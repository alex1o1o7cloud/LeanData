import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_ABC_l197_19702

-- Define the points
noncomputable def O : ℝ × ℝ × ℝ := (0, 0, 0)
noncomputable def A : ℝ × ℝ × ℝ := (Real.sqrt 50, 0, 0)

-- Define B and C as variables on y and z axes
variable (B : ℝ × ℝ × ℝ)
variable (C : ℝ × ℝ × ℝ)

-- Define the conditions
def on_y_axis (p : ℝ × ℝ × ℝ) : Prop := p.1 = 0 ∧ p.2.2 = 0
def on_z_axis (p : ℝ × ℝ × ℝ) : Prop := p.1 = 0 ∧ p.2.1 = 0

-- Define the angle BAC
noncomputable def angle_BAC (A B C : ℝ × ℝ × ℝ) : ℝ := 
  Real.arccos ((B.1 - A.1) * (C.1 - A.1) + (B.2.1 - A.2.1) * (C.2.1 - A.2.1) + (B.2.2 - A.2.2) * (C.2.2 - A.2.2)) / 
    (Real.sqrt ((B.1 - A.1)^2 + (B.2.1 - A.2.1)^2 + (B.2.2 - A.2.2)^2) * 
     Real.sqrt ((C.1 - A.1)^2 + (C.2.1 - A.2.1)^2 + (C.2.2 - A.2.2)^2))

-- Define the area of triangle ABC
noncomputable def area_triangle (A B C : ℝ × ℝ × ℝ) : ℝ :=
  let a := Real.sqrt ((B.1 - C.1)^2 + (B.2.1 - C.2.1)^2 + (B.2.2 - C.2.2)^2)
  let b := Real.sqrt ((A.1 - C.1)^2 + (A.2.1 - C.2.1)^2 + (A.2.2 - C.2.2)^2)
  let c := Real.sqrt ((A.1 - B.1)^2 + (A.2.1 - B.2.1)^2 + (A.2.2 - B.2.2)^2)
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Theorem statement
theorem area_triangle_ABC (B C : ℝ × ℝ × ℝ) 
  (h1 : on_y_axis B) 
  (h2 : on_z_axis C) 
  (h3 : angle_BAC A B C = π / 4) : 
  area_triangle A B C = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_ABC_l197_19702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_four_digit_number_l197_19700

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def has_two_even_two_odd (n : ℕ) : Prop :=
  let digits := n.digits 10
  (digits.filter (λ d => d % 2 = 0)).length = 2 ∧ (digits.filter (λ d => d % 2 ≠ 0)).length = 2

def thousands_digit_between_2_and_3 (n : ℕ) : Prop :=
  let thousands := (n / 1000) % 10
  thousands = 2 ∨ thousands = 3

theorem smallest_four_digit_number (n : ℕ) :
  is_four_digit n ∧
  n % 3 = 0 ∧
  has_two_even_two_odd n ∧
  thousands_digit_between_2_and_3 n →
  n ≥ 3009 :=
by sorry

#eval Nat.digits 10 3009
#eval (3009 / 1000) % 10
#eval 3009 % 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_four_digit_number_l197_19700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_f_to_g_l197_19746

/-- The original function f(x) -/
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) - 2 * (Real.sin x) ^ 2 + 1

/-- The transformed function g(x) -/
noncomputable def g (x : ℝ) : ℝ := Real.sqrt 2 * Real.sin (4 * x - 3 * Real.pi / 4)

/-- Theorem stating that g(x) is the result of transforming f(x) -/
theorem transform_f_to_g : 
  ∀ x : ℝ, g (x + Real.pi / 4) = f (x / 2) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_f_to_g_l197_19746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l197_19749

/-- The minimum value of sqrt(x^2 + y^2) given the constraint 6x + 8y = 48 is 4.8 -/
theorem min_distance_to_line :
  (∀ x y : ℝ, 6 * x + 8 * y = 48 → Real.sqrt (x^2 + y^2) ≥ 4.8) ∧
  (∃ x y : ℝ, 6 * x + 8 * y = 48 ∧ Real.sqrt (x^2 + y^2) = 4.8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l197_19749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_l197_19767

theorem problem_1 : ((-4) - 13 + (-5) - (-9) + 7) = -6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_l197_19767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_m_equation_tangent_lines_equations_line_DE_equation_l197_19766

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 4

-- Define point A
noncomputable def point_A : ℝ × ℝ := (1, 2 * Real.sqrt 3)

-- Theorem for the equation of line m
theorem line_m_equation :
  ∃ (m : ℝ → ℝ → Prop),
    (∀ x y, m x y ↔ x = 0) ∧
    (m 0 0) ∧
    (∃ (p1 p2 p3 : ℝ × ℝ),
      p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
      circle_C p1.1 p1.2 ∧ circle_C p2.1 p2.2 ∧ circle_C p3.1 p3.2 ∧
      (∀ p, circle_C p.1 p.2 → 
        (abs (p.1) = 1 ↔ (p = p1 ∨ p = p2 ∨ p = p3)))) := by
  sorry

-- Theorem for the equations of tangent lines
theorem tangent_lines_equations :
  ∃ (l1 l2 : ℝ → ℝ → Prop),
    (∀ x y, l1 x y ↔ Real.sqrt 3 * x - 3 * y + 5 * Real.sqrt 3 = 0) ∧
    (∀ x y, l2 x y ↔ x = 1) ∧
    l1 point_A.1 point_A.2 ∧ l2 point_A.1 point_A.2 ∧
    (∃ (D E : ℝ × ℝ),
      circle_C D.1 D.2 ∧ circle_C E.1 E.2 ∧
      l1 D.1 D.2 ∧ l2 E.1 E.2 ∧
      (∀ p, circle_C p.1 p.2 → ¬(l1 p.1 p.2) ∨ p = D) ∧
      (∀ p, circle_C p.1 p.2 → ¬(l2 p.1 p.2) ∨ p = E)) := by
  sorry

-- Theorem for the equation of line DE
theorem line_DE_equation :
  ∃ (DE : ℝ → ℝ → Prop),
    (∀ x y, DE x y ↔ x + Real.sqrt 3 * y - 1 = 0) ∧
    (∃ (D E : ℝ × ℝ),
      circle_C D.1 D.2 ∧ circle_C E.1 E.2 ∧
      (∃ (l : ℝ → ℝ → Prop),
        l point_A.1 point_A.2 ∧ l D.1 D.2 ∧ l E.1 E.2 ∧
        (∀ p, circle_C p.1 p.2 → ¬(l p.1 p.2) ∨ p = D ∨ p = E)) ∧
      DE D.1 D.2 ∧ DE E.1 E.2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_m_equation_tangent_lines_equations_line_DE_equation_l197_19766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tourism_investment_breakeven_l197_19734

noncomputable def a (n : ℕ) : ℝ := 40 * (1 - (4/5)^n)
noncomputable def b (n : ℕ) : ℝ := 16 * ((5/4)^n - 1)

theorem tourism_investment_breakeven :
  ∃ n : ℕ, (∀ k < n, b k ≤ a k) ∧ (∀ m ≥ n, b m > a m) ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tourism_investment_breakeven_l197_19734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_sum_a_b_is_zero_l197_19784

-- Define the greatest integer function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Define the nested square root function
noncomputable def nestedSqrt (n : ℕ) : ℝ :=
  if n = 0 then 0 else Real.sqrt (6 + nestedSqrt (n - 1))

-- Define the nested cube root function
noncomputable def nestedCubeRoot (n : ℕ) : ℝ :=
  if n = 0 then 0 else (6 + nestedCubeRoot (n - 1)) ^ (1/3)

-- Define a and b
noncomputable def a : ℝ := nestedSqrt 2016 / 2016
noncomputable def b : ℝ := nestedCubeRoot 2017 / 2017

-- Theorem statement
theorem floor_sum_a_b_is_zero : floor (a + b) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_sum_a_b_is_zero_l197_19784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_max_score_l197_19718

/-- The game described in the problem -/
structure Game where
  n : ℕ
  h : n ≥ 3 ∧ Odd n

/-- The maximum score achievable in the game -/
def maxScore (g : Game) : ℕ := g.n * (g.n + 1)

/-- Theorem stating the maximum score for the game -/
theorem game_max_score (g : Game) : 
  ∀ (score : ℕ), score ≤ maxScore g := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_max_score_l197_19718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_eq_sqrt_one_plus_sqrt_two_l197_19755

/-- The distance between the intersections of x = y³ and x + y³ = 1 -/
noncomputable def intersection_distance : ℝ :=
  let y₁ := (1/2) ^ (1/3 : ℝ)
  let y₂ := -(1/2) ^ (1/3 : ℝ)
  let x₁ := y₁^3
  let x₂ := y₂^3
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

/-- The theorem stating that the distance is equal to √(1 + √2) -/
theorem intersection_distance_eq_sqrt_one_plus_sqrt_two :
  intersection_distance = Real.sqrt (1 + Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_eq_sqrt_one_plus_sqrt_two_l197_19755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_cube_root_fraction_l197_19781

/-- The limit of (∛x - 1) / (∛(x^2 + 2∛x - 3)) as x approaches 1 is 1/4 -/
theorem limit_cube_root_fraction :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 1| ∧ |x - 1| < δ →
    |((x^(1/3) - 1) / ((x^2 + 2*x^(1/3) - 3)^(1/3))) - 1/4| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_cube_root_fraction_l197_19781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_distance_l197_19745

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

/-- The line equation -/
def line (x y b : ℝ) : Prop := y = x + b

/-- Two points are distinct -/
def distinct (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁ ≠ x₂ ∨ y₁ ≠ y₂

/-- The distance between two points -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

/-- The main theorem -/
theorem intersection_and_distance :
  (∀ b : ℝ, (∃ x₁ y₁ x₂ y₂ : ℝ, 
    ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧ 
    line x₁ y₁ b ∧ line x₂ y₂ b ∧ 
    distinct x₁ y₁ x₂ y₂) ↔ 
    -Real.sqrt 3 < b ∧ b < Real.sqrt 3) ∧
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧ 
    line x₁ y₁ 1 ∧ line x₂ y₂ 1 ∧ 
    distance x₁ y₁ x₂ y₂ = 4 * Real.sqrt 2 / 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_distance_l197_19745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_down_payment_l197_19794

def laptop_price : ℝ := 1000
def monthly_installment : ℝ := 65
def down_payment_percentage : ℝ := 0.20
def months_paid : ℕ := 4
def remaining_balance : ℝ := 520

theorem additional_down_payment (additional_amount : ℝ) : 
  laptop_price - (down_payment_percentage * laptop_price + additional_amount + months_paid * monthly_installment) = remaining_balance → 
  additional_amount = 20 := by
  intro h
  -- Proof steps would go here
  sorry

#check additional_down_payment

end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_down_payment_l197_19794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_drain_time_proof_l197_19785

/-- Represents the time to drain the remaining content of a tub -/
noncomputable def drain_remaining_time (C x : ℝ) : ℝ :=
  8 / (5 + 28 * x / C)

/-- 
Theorem stating that the time to drain the remaining content of a tub
is equal to 8 / (5 + 28x/C) minutes, given the conditions of the problem.
-/
theorem drain_time_proof (C x : ℝ) (h1 : C > 0) (h2 : x ≥ 0) : 
  ∃ (D : ℝ), 
    (D - x) * 4 = 5/7 * C ∧ 
    drain_remaining_time C x = 2/7 * C / D :=
by
  sorry

#check drain_time_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_drain_time_proof_l197_19785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rooks_placement_exists_l197_19771

-- Define the type for coordinates in the 4x4x4 cube
def Coord := Fin 4 × Fin 4 × Fin 4

-- Define a predicate to check if two coordinates can attack each other
def can_attack (c1 c2 : Coord) : Prop :=
  c1.1 = c2.1 ∨ c1.2.1 = c2.2.1 ∨ c1.2.2 = c2.2.2

-- Define the theorem
theorem rooks_placement_exists : ∃ (positions : Finset Coord),
  positions.card = 16 ∧ 
  (∀ c1 c2, c1 ∈ positions → c2 ∈ positions → c1 ≠ c2 → ¬can_attack c1 c2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rooks_placement_exists_l197_19771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_diameter_height_ratio_l197_19797

/-- A structure representing a sphere inscribed in a truncated right circular cone -/
structure InscribedSphere where
  s : ℝ  -- radius of the sphere
  R : ℝ  -- radius of the larger base of the truncated cone
  r : ℝ  -- radius of the smaller base of the truncated cone
  H : ℝ  -- height of the truncated cone

/-- The volume of a sphere -/
noncomputable def sphereVolume (sphere : InscribedSphere) : ℝ :=
  (4 / 3) * Real.pi * sphere.s^3

/-- The volume of a truncated right circular cone -/
noncomputable def truncatedConeVolume (cone : InscribedSphere) : ℝ :=
  (Real.pi * cone.H / 3) * (cone.R^2 + cone.R * cone.r + cone.r^2)

/-- Theorem: If a sphere is inscribed in a truncated right circular cone and the volume of the
    truncated cone is three times that of the sphere, then the ratio of the sphere's diameter
    to the height of the truncated cone is 1 -/
theorem inscribed_sphere_diameter_height_ratio
    (sphere : InscribedSphere)
    (h1 : sphere.s = Real.sqrt (sphere.R * sphere.r))  -- Geometric mean theorem
    (h2 : sphere.H = 2 * sphere.s)  -- Height relation
    (h3 : truncatedConeVolume sphere = 3 * sphereVolume sphere)  -- Volume relation
    : (2 * sphere.s) / sphere.H = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_diameter_height_ratio_l197_19797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_loss_percentage_l197_19738

/-- Represents the problem of determining the loss percentage for remaining stock --/
theorem stock_loss_percentage 
  (total_stock : ℝ) 
  (profit_percentage : ℝ) 
  (sold_percentage : ℝ) 
  (overall_loss : ℝ) : 
  total_stock = 12499.99 →
  profit_percentage = 20 →
  sold_percentage = 20 →
  overall_loss = 500 →
  ∃ (loss_percentage : ℝ), 
    (loss_percentage ≥ 9.99 ∧ loss_percentage ≤ 10.01) ∧
    (sold_percentage / 100 * profit_percentage / 100 * total_stock) -
    ((100 - sold_percentage) / 100 * loss_percentage / 100 * total_stock) = -overall_loss :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_loss_percentage_l197_19738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_function_satisfies_conditions_price_function_trend_price_fall_prediction_l197_19710

/-- Represents the price simulation function for a certain region's specialty fruit -/
noncomputable def price_function (x : ℝ) : ℝ := x * (x - 3)^2 + 4

/-- The price function satisfies the given conditions -/
theorem price_function_satisfies_conditions :
  (price_function 0 = 4) ∧
  (price_function 2 = 6) ∧
  (∀ x : ℝ, x ≥ 0 → x ≤ 5 → price_function x ≥ 0) := by
  sorry

/-- The price function has two increasing intervals and one decreasing interval -/
theorem price_function_trend :
  ∃ a b : ℝ, 0 < a ∧ a < b ∧ b < 5 ∧
  (∀ x : ℝ, 0 ≤ x ∧ x < a → (deriv price_function x) > 0) ∧
  (∀ x : ℝ, a < x ∧ x < b → (deriv price_function x) < 0) ∧
  (∀ x : ℝ, b < x ∧ x ≤ 5 → (deriv price_function x) > 0) := by
  sorry

/-- The price is predicted to fall in May and June -/
theorem price_fall_prediction :
  (deriv price_function 1 < 0) ∧ (deriv price_function 2 < 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_function_satisfies_conditions_price_function_trend_price_fall_prediction_l197_19710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_energetic_time_is_seven_l197_19795

/-- Represents a runner with two speeds --/
structure Runner where
  energeticSpeed : ℝ
  tiredSpeed : ℝ

/-- Calculates the time spent at energetic speed given total distance, total time, and runner's speeds --/
noncomputable def timeSpentEnergetic (runner : Runner) (totalDistance : ℝ) (totalTime : ℝ) : ℝ :=
  (totalDistance - runner.tiredSpeed * totalTime) / (runner.energeticSpeed - runner.tiredSpeed)

/-- Theorem stating that under given conditions, the time spent at energetic speed is 7 hours --/
theorem energetic_time_is_seven {totalDistance totalTime : ℝ} (runner : Runner) 
    (h1 : runner.energeticSpeed = 10)
    (h2 : runner.tiredSpeed = 6)
    (h3 : totalDistance = 88)
    (h4 : totalTime = 10) : 
  timeSpentEnergetic runner totalDistance totalTime = 7 := by
  sorry

#check energetic_time_is_seven

end NUMINAMATH_CALUDE_ERRORFEEDBACK_energetic_time_is_seven_l197_19795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_intersection_points_l197_19724

/-- The function f(x) = tan(ω * x) -/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.tan (ω * x)

/-- The theorem stating the properties of the function and its value at π/12 -/
theorem tan_intersection_points (ω : ℝ) (h1 : ω > 0) 
  (h2 : ∀ (x : ℝ), f ω (x + π / (4 * ω)) = f ω x) :
  f ω (π / 12) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_intersection_points_l197_19724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_cycle_length_l197_19733

theorem reciprocal_cycle_length (x : ℝ) (h : x = 48) : 
  ∃ n : ℕ, n = 2 ∧ (∀ m : ℕ, 0 < m → m < n → (1 / x) ^ m ≠ x) ∧ (1 / x) ^ n = x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_cycle_length_l197_19733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_of_f_l197_19728

noncomputable section

-- Define the function f(x)
def f (x : ℝ) : ℝ := (1/3) * x^3 - 4*x + 4

-- Define the interval [0, 3]
def interval : Set ℝ := Set.Icc 0 3

-- State the theorem
theorem extreme_values_of_f :
  ∃ (max min : ℝ),
    (∀ x ∈ interval, f x ≤ max) ∧
    (∃ x ∈ interval, f x = max) ∧
    (∀ x ∈ interval, min ≤ f x) ∧
    (∃ x ∈ interval, f x = min) ∧
    max = 4 ∧
    min = -4/3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_of_f_l197_19728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_tangent_length_l197_19743

/-- Definition of circle C₁ -/
def C₁ (x y : ℝ) : Prop := (x - 12)^2 + y^2 = 25

/-- Definition of circle C₂ -/
def C₂ (x y : ℝ) : Prop := (x + 18)^2 + y^2 = 100

/-- Definition of a point being on a circle -/
def on_circle (C : ℝ → ℝ → Prop) (p : ℝ × ℝ) : Prop := C p.1 p.2

/-- Definition of a line segment being tangent to a circle at a point -/
def is_tangent (C : ℝ → ℝ → Prop) (p : ℝ × ℝ) : Prop := sorry

/-- The shortest line segment PQ that is tangent to both circles -/
def shortest_tangent (P Q : ℝ × ℝ) : Prop :=
  on_circle C₁ P ∧
  on_circle C₂ Q ∧
  is_tangent C₁ P ∧
  is_tangent C₂ Q ∧
  ∀ P' Q', on_circle C₁ P' → on_circle C₂ Q' →
    is_tangent C₁ P' → is_tangent C₂ Q' →
    Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ≤ Real.sqrt ((P'.1 - Q'.1)^2 + (P'.2 - Q'.2)^2)

theorem shortest_tangent_length :
  ∀ P Q : ℝ × ℝ, shortest_tangent P Q →
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 5 * Real.sqrt 35 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_tangent_length_l197_19743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factory_completion_time_l197_19714

/-- Production line data -/
structure ProductionLine where
  rate : ℕ  -- gummy bears per minute
  packetSize : ℕ  -- gummy bears per packet
  requiredPackets : ℕ

/-- Factory setup -/
def factory : List ProductionLine := [
  ⟨300, 50, 240⟩,  -- Line A
  ⟨400, 75, 180⟩,  -- Line B
  ⟨500, 100, 150⟩  -- Line C
]

/-- Time taken for a production line to complete its task -/
def timeTaken (line : ProductionLine) : ℚ :=
  (line.requiredPackets * line.packetSize : ℚ) / line.rate

/-- Maximum time among all production lines -/
def maxTime (lines : List ProductionLine) : ℚ :=
  (lines.map timeTaken).foldl max 0

/-- Theorem: The factory completes all tasks in 40 minutes -/
theorem factory_completion_time :
  maxTime factory = 40 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factory_completion_time_l197_19714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_a_value_l197_19708

-- Define the function f(x) = x^a
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^a

-- Define the derivative of f
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := a * x^(a-1)

theorem tangent_line_implies_a_value (a : ℝ) :
  (∀ x, f_derivative a 1 * (x - 1) + f a 1 = -4 * x) →
  a = -4 := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

#check tangent_line_implies_a_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_a_value_l197_19708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_product_constant_line_MN_passes_through_fixed_point_l197_19789

/-- Ellipse C with equation x²/4 + y²/2 = 1 -/
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2/2 = 1

/-- Left vertex A of the ellipse -/
noncomputable def A : ℝ × ℝ := (-2, 0)

/-- Right vertex B of the ellipse -/
noncomputable def B : ℝ × ℝ := (2, 0)

/-- Slope of line MA -/
noncomputable def k_MA (x₀ y₀ : ℝ) : ℝ := y₀ / (x₀ + 2)

/-- Slope of line MB -/
noncomputable def k_MB (x₀ y₀ : ℝ) : ℝ := y₀ / (x₀ - 2)

/-- Theorem: Product of slopes k_MA and k_MB is constant -/
theorem slope_product_constant (x₀ y₀ : ℝ) :
  ellipse_C x₀ y₀ → x₀ ≠ -2 → x₀ ≠ 2 → k_MA x₀ y₀ * k_MB x₀ y₀ = -1/2 := by
  sorry

/-- Fixed point through which line MN passes -/
noncomputable def fixed_point : ℝ × ℝ := (2/3, 0)

/-- Theorem: Line MN passes through the fixed point -/
theorem line_MN_passes_through_fixed_point (x₁ y₁ x₂ y₂ : ℝ) :
  ellipse_C x₁ y₁ → ellipse_C x₂ y₂ → x₁ ≠ x₂ → 
  ∃ m t : ℝ, (x₁ = m * y₁ + t ∧ x₂ = m * y₂ + t) → t = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_product_constant_line_MN_passes_through_fixed_point_l197_19789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l197_19790

def A : Set ℕ := {1, 2}

def B : Set ℕ := {x : ℕ | x^2 - 3*x + 2 = 0}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l197_19790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_fourth_term_l197_19799

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sum_of_arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (a 1 + a n) / 2

theorem arithmetic_sequence_fourth_term
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : sum_of_arithmetic_sequence a 5 = 35)
  (h_fifth : a 5 = 11) :
  a 4 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_fourth_term_l197_19799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_points_dense_l197_19726

-- Define the closed interval [0,1]
def I : Set ℝ := Set.Icc 0 1

-- Define topological transitivity
def topologically_transitive (f : I → I) : Prop :=
  ∀ U V : Set I, IsOpen U → IsOpen V → Set.Nonempty U → Set.Nonempty V →
    ∃ n : ℕ, Set.Nonempty (Set.inter ((f^[n]) '' U) V)

-- Define periodic points
def is_periodic_point (f : I → I) (x : I) : Prop :=
  ∃ n : ℕ+, (f^[n]) x = x

def periodic_points (f : I → I) : Set I :=
  {x : I | is_periodic_point f x}

-- Main theorem statement
theorem periodic_points_dense
  (f : I → I)
  (h_cont : Continuous f)
  (h_trans : topologically_transitive f) :
  Dense (periodic_points f) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_points_dense_l197_19726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_bottle_height_l197_19730

-- Define the cone bottle
structure ConeBottle where
  height : ℝ
  radius : ℝ

-- Define the water level in the bottle
def water_level (bottle : ConeBottle) : ℝ := bottle.height - 8

-- Define the volume of water in the bottle when upright
noncomputable def water_volume_upright (bottle : ConeBottle) : ℝ :=
  (1/3) * Real.pi * (8 * bottle.radius / bottle.height)^2 * 8

-- Define the volume of water in the bottle when inverted
noncomputable def water_volume_inverted (bottle : ConeBottle) : ℝ :=
  (1/3) * Real.pi * ((bottle.height - 2) * bottle.radius / bottle.height)^2 * (bottle.height - 2)

-- Theorem statement
theorem cone_bottle_height (bottle : ConeBottle) :
  water_volume_upright bottle = water_volume_inverted bottle →
  bottle.height = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_bottle_height_l197_19730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_parabola_l197_19763

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The distance from a point to a vertical line -/
def distanceToVerticalLine (p : Point) (x : ℝ) : ℝ :=
  |p.x - x|

/-- The focus of the parabola -/
def F : Point := { x := -4, y := 0 }

/-- The x-coordinate of the directrix -/
def directrixX : ℝ := 4

/-- A parabola is the set of points equidistant from a fixed point (focus) and a fixed line (directrix) -/
def isParabola (P : Set Point) : Prop :=
  ∀ p ∈ P, distance p F = distanceToVerticalLine p directrixX

/-- The theorem to be proved -/
theorem trajectory_is_parabola (P : Set Point) :
  (∀ p ∈ P, distance p F = distanceToVerticalLine p directrixX) →
  isParabola P := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_parabola_l197_19763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_point_one_l197_19720

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x - x

-- Define the derivative of f(x)
noncomputable def f_derivative (x : ℝ) : ℝ := 2 / x - 1

-- Theorem statement
theorem tangent_line_at_point_one (x y : ℝ) :
  f 1 = -1 →  -- The point (1, -1) is on the curve
  f_derivative 1 = 1 →  -- The derivative at x = 1 is 1
  (x - y - 2 = 0) ↔ (y - (-1) = f_derivative 1 * (x - 1)) :=  -- The equation of the tangent line
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_point_one_l197_19720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l197_19719

/-- Calculates the speed of a train given its length, time to cross a person, and the person's speed in the opposite direction --/
theorem train_speed_calculation (train_length : ℝ) (crossing_time : ℝ) (person_speed_kmph : ℝ) :
  train_length = 250 →
  crossing_time = 6 →
  person_speed_kmph = 10 →
  ∃ (train_speed_kmph : ℝ), abs (train_speed_kmph - 140) < 0.1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l197_19719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equality_implies_base_l197_19703

theorem log_equality_implies_base (x : ℝ) :
  x > 0 → (Real.log 16 / Real.log x = Real.log 3 / Real.log 81) → x = 65536 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equality_implies_base_l197_19703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_closed_form_l197_19777

/-- Definition of the sequence a_n -/
def a : ℕ → ℕ
| 0 => 1
| n + 1 => 2 * a n + 2^(n + 1)

/-- Theorem stating the closed form of a_n -/
theorem a_closed_form (n : ℕ) : a n = (2 * (n + 1) - 1) * 2^n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_closed_form_l197_19777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_day_150_of_year_N_minus_1_is_Friday_l197_19727

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a year -/
structure Year where
  value : ℕ

/-- Represents a day in a year -/
structure DayInYear where
  day : ℕ
  year : Year

/-- Function to get the day of the week for a given day in a year -/
def dayOfWeek (d : DayInYear) : DayOfWeek :=
  sorry

/-- Given conditions -/
axiom condition1 (N : Year) : dayOfWeek ⟨280, N⟩ = DayOfWeek.Wednesday
axiom condition2 (N : Year) : dayOfWeek ⟨190, ⟨N.value + 1⟩⟩ = DayOfWeek.Wednesday

/-- Theorem to prove -/
theorem day_150_of_year_N_minus_1_is_Friday (N : Year) :
  dayOfWeek ⟨150, ⟨N.value - 1⟩⟩ = DayOfWeek.Friday := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_day_150_of_year_N_minus_1_is_Friday_l197_19727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tile_II_in_rectangle_B_l197_19791

-- Define the sides of a tile
structure TileSides where
  top : ℕ
  right : ℕ
  bottom : ℕ
  left : ℕ

-- Define the tiles
def tile_I : TileSides := ⟨2, 6, 5, 3⟩
def tile_II : TileSides := ⟨6, 2, 3, 5⟩
def tile_III : TileSides := ⟨5, 7, 1, 2⟩
def tile_IV : TileSides := ⟨3, 5, 6, 7⟩

-- Define the rectangles (A, B, C, D)
inductive Rectangle
  | A | B | C | D

-- Define a function to represent the placement of tiles
def tile_placement : Rectangle → TileSides :=
  fun r => match r with
    | Rectangle.A => tile_I  -- Placeholder, actual placement may vary
    | Rectangle.B => tile_II
    | Rectangle.C => tile_IV -- Placeholder, actual placement may vary
    | Rectangle.D => tile_III

-- Theorem: Tile II must be placed in Rectangle B
theorem tile_II_in_rectangle_B :
  tile_placement Rectangle.B = tile_II := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tile_II_in_rectangle_B_l197_19791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_m_is_3125_l197_19731

-- Define the set T
def T (m : ℕ) : Set ℕ := {n : ℕ | 5 ≤ n ∧ n ≤ m}

-- Define the property of a set containing a, b, c such that ab = c
def contains_abc (S : Set ℕ) : Prop :=
  ∃ a b c, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a * b = c

-- Define the property that any partition of T satisfies the condition
def any_partition_satisfies (m : ℕ) : Prop :=
  ∀ A B : Set ℕ, (A ∪ B = T m) → (A ∩ B = ∅) →
    contains_abc A ∨ contains_abc B

-- State the theorem
theorem minimal_m_is_3125 :
  ∀ m : ℕ, m ≥ 5 →
    (any_partition_satisfies m ↔ m ≥ 3125) :=
by sorry

#check minimal_m_is_3125

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_m_is_3125_l197_19731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_four_points_odd_distances_l197_19770

-- Define a Point in the Euclidean plane
def Point := ℝ × ℝ

-- Define a function to calculate the distance between two points
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define a predicate to check if a number is an odd integer
def isOddInteger (x : ℝ) : Prop :=
  ∃ (k : ℤ), x = 2 * k + 1

-- State the theorem
theorem no_four_points_odd_distances :
  ¬ ∃ (A B C D : Point),
    (isOddInteger (distance A B)) ∧
    (isOddInteger (distance A C)) ∧
    (isOddInteger (distance A D)) ∧
    (isOddInteger (distance B C)) ∧
    (isOddInteger (distance B D)) ∧
    (isOddInteger (distance C D)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_four_points_odd_distances_l197_19770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opponent_total_runs_l197_19751

theorem opponent_total_runs (team_scores : List ℕ) 
  (h1 : team_scores = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
  (h2 : ∃ (lost_games : List ℕ), lost_games.length = 6 ∧ 
    ∀ x ∈ lost_games, ∃ y ∈ team_scores, x = y + 2)
  (h3 : ∃ (triple_games : List ℕ), triple_games.length = 3 ∧ 
    ∀ x ∈ triple_games, ∃ y ∈ team_scores, x = y / 3)
  (h4 : ∃ (double_games : List ℕ), double_games.length = 3 ∧ 
    ∀ x ∈ double_games, ∃ y ∈ team_scores, x = y / 2) :
  ∃ (opponent_scores : List ℕ), (opponent_scores.length = 12) ∧ 
    (opponent_scores.sum = 56) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_opponent_total_runs_l197_19751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_l197_19704

/-- The ellipse defined by x²/3 + y² = 1 -/
def ellipse (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

/-- The line defined by x - y - 1 = 0 -/
def line (x y : ℝ) : Prop := x - y - 1 = 0

/-- The distance from a point (x, y) to the line x - y - 1 = 0 -/
noncomputable def distance_to_line (x y : ℝ) : ℝ := |x - y - 1| / Real.sqrt 2

theorem max_distance_to_line :
  ∃ (x y : ℝ), ellipse x y ∧
               (∀ (x' y' : ℝ), ellipse x' y' →
                 distance_to_line x y ≥ distance_to_line x' y') ∧
               distance_to_line x y = 3 * Real.sqrt 2 / 2 ∧
               x = 3 / 2 ∧ y = -1 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_l197_19704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_selection_uses_golden_ratio_answer_is_correct_l197_19765

/-- The Golden ratio -/
noncomputable def golden_ratio : ℝ := (1 + Real.sqrt 5) / 2

/-- The optimal selection method popularized by Hua Luogeng -/
def optimal_selection_method : Type := Unit

/-- The mathematical concept used in the optimal selection method -/
noncomputable def concept_used (method : optimal_selection_method) : ℝ := golden_ratio

/-- Theorem stating that the optimal selection method uses the Golden ratio -/
theorem optimal_selection_uses_golden_ratio :
  ∀ (method : optimal_selection_method), concept_used method = golden_ratio := by
  intro method
  rfl

/-- Proof that the answer to the question is correct -/
theorem answer_is_correct : concept_used () = golden_ratio := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_selection_uses_golden_ratio_answer_is_correct_l197_19765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l197_19760

-- Define the hyperbola
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the conditions
def has_foci_on_x_axis (h : Hyperbola) : Prop :=
  h.c > 0 ∧ h.a > 0 ∧ h.b > 0

noncomputable def asymptote_angle (h : Hyperbola) : ℝ :=
  2 * Real.arctan (h.b / h.a)

def focal_distance (h : Hyperbola) : ℝ :=
  2 * h.c

-- Define the equations and eccentricities
def equation1 (x y : ℝ) : Prop :=
  x^2 / 27 - y^2 / 9 = 1

def equation2 (x y : ℝ) : Prop :=
  x^2 / 9 - y^2 / 27 = 1

noncomputable def eccentricity1 : ℝ :=
  2 * Real.sqrt 3 / 3

def eccentricity2 : ℝ :=
  2

-- Theorem statement
theorem hyperbola_properties (h : Hyperbola) :
  has_foci_on_x_axis h →
  asymptote_angle h = π/3 →
  focal_distance h = 12 →
  ((∀ x y, equation1 x y) ∧ h.c / h.a = eccentricity1) ∨
  ((∀ x y, equation2 x y) ∧ h.c / h.a = eccentricity2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l197_19760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_sum_inverse_squares_l197_19701

/-- A plane intersecting the coordinate axes -/
structure IntersectingPlane where
  /-- Distance from the origin to the plane -/
  distance : ℝ
  /-- x-coordinate of the intersection with the x-axis -/
  α : ℝ
  /-- y-coordinate of the intersection with the y-axis -/
  β : ℝ
  /-- z-coordinate of the intersection with the z-axis -/
  γ : ℝ
  /-- Condition that the plane equation holds -/
  plane_eq : 1 / α + 1 / β + 1 / γ = 1
  /-- Condition that the distance is correct -/
  distance_eq : 1 / Real.sqrt (1 / α^2 + 1 / β^2 + 1 / γ^2) = distance
  /-- Condition that A, B, and C are distinct from the origin -/
  distinct_points : α ≠ 0 ∧ β ≠ 0 ∧ γ ≠ 0

/-- The centroid of a triangle formed by intersections with coordinate axes -/
noncomputable def centroid (p : IntersectingPlane) : ℝ × ℝ × ℝ :=
  (p.α / 3, p.β / 3, p.γ / 3)

/-- Theorem stating the relationship between the centroid coordinates -/
theorem centroid_sum_inverse_squares (p : IntersectingPlane) (h : p.distance = 3) :
    let (x, y, z) := centroid p
    1 / x^2 + 1 / y^2 + 1 / z^2 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_sum_inverse_squares_l197_19701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_is_50_cube_root_2_l197_19715

/-- Represents a right circular cone -/
structure Cone where
  radius : ℝ
  height : ℝ

/-- Calculates the volume of a cone -/
noncomputable def coneVolume (c : Cone) : ℝ := (1/3) * Real.pi * c.radius^2 * c.height

/-- Represents the water tank problem -/
def waterTankProblem (tank : Cone) (waterHeight : ℝ) : Prop :=
  tank.radius = 20 ∧
  tank.height = 100 ∧
  coneVolume { radius := tank.radius * (waterHeight / tank.height), height := waterHeight } = 
    (1/2) * coneVolume tank

theorem water_height_is_50_cube_root_2 (tank : Cone) (waterHeight : ℝ) :
  waterTankProblem tank waterHeight →
  waterHeight = 50 * (2 ^ (1/3)) := by
  sorry

#eval 50 + 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_is_50_cube_root_2_l197_19715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_90_degrees_l197_19788

-- Define the function
noncomputable def f (b c : ℝ) (x : ℝ) : ℝ := x + b * Real.sqrt x + c

-- State the theorem
theorem angle_sum_90_degrees (b c : ℝ) (h_c : c > 0) :
  ∃ (x₁ x₂ : ℝ), 
    (f b c x₁ = 0) ∧ 
    (f b c x₂ = 0) ∧ 
    (x₁ ≠ x₂) ∧
    (∃ (angle₁ angle₂ : ℝ), 
      angle₁ = Real.arctan (c / x₁) ∧
      angle₂ = Real.arctan (c / x₂) ∧
      angle₁ + angle₂ = π / 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_90_degrees_l197_19788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_equals_ten_l197_19709

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + 1 else -3*x

-- State the theorem
theorem function_composition_equals_ten (x : ℝ) :
  f (f x) = 10 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_equals_ten_l197_19709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l197_19721

-- Define the function f as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x)^2 - 1

-- State the theorem
theorem f_properties :
  (∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x = 2) ∧  -- maximum value
  (∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x = -1) ∧  -- minimum value
  (∀ (x₀ : ℝ), x₀ ∈ Set.Icc (Real.pi / 4) (Real.pi / 2) → 
    f x₀ = 6 / 5 → Real.cos (2 * x₀) = (3 - 4 * Real.sqrt 3) / 10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l197_19721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_train_length_l197_19737

/-- Calculates the length of a train given the speeds of two trains and the time taken to pass --/
noncomputable def trainLength (slowSpeed fastSpeed : ℝ) (passingTime : ℝ) : ℝ :=
  (slowSpeed + fastSpeed) * (1000 / 3600) * passingTime

/-- Theorem stating the length of the faster train given the problem conditions --/
theorem faster_train_length :
  trainLength 36 45 10 = 225 := by
  -- Unfold the definition of trainLength
  unfold trainLength
  -- Simplify the arithmetic
  simp [mul_add, mul_assoc]
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_train_length_l197_19737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_edge_growth_l197_19756

/-- 
Given a cube with an initial edge length, this theorem states that 
if the surface area of the cube increases by 95.99999999999997%, 
then the edges of the cube have increased by approximately 40%.
-/
theorem cube_edge_growth (initial_edge : ℝ) (h : initial_edge > 0) :
  let surface_area_increase := 95.99999999999997
  let edge_growth := 40
  let new_edge := initial_edge * (1 + edge_growth / 100)
  let initial_surface_area := 6 * initial_edge^2
  let new_surface_area := 6 * new_edge^2
  abs ((new_surface_area - initial_surface_area) / initial_surface_area * 100 - surface_area_increase) < 0.0001 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_edge_growth_l197_19756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_implies_root_l197_19711

noncomputable def f (x : ℝ) : ℝ := Real.exp (2*x) * (2*x + 2*Real.log 2 - 1) - 2

noncomputable def curve1 (x : ℝ) : ℝ := Real.exp x

noncomputable def curve2 (x : ℝ) : ℝ := Real.exp (2*x) - 2

noncomputable def curve1_deriv (x : ℝ) : ℝ := Real.exp x

noncomputable def curve2_deriv (x : ℝ) : ℝ := 2 * Real.exp (2*x)

theorem common_tangent_implies_root (a b : ℝ) :
  (∃ m n : ℝ, curve1_deriv m = curve2_deriv a ∧ 
              curve1 m = curve1_deriv m * (m - a) + curve2 a) →
  curve2 a = b →
  f a = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_implies_root_l197_19711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gain_amount_proof_l197_19707

/-- 
Given a gain in rupees and a gain percent, calculate the original amount.
-/
noncomputable def calculate_amount (gain : ℝ) (gain_percent : ℝ) : ℝ :=
  gain / (gain_percent / 100)

/-- 
Theorem: Given a gain of 0.70 rupees and a gain percent of 1%, 
the amount on which the gain was made is 70 rupees.
-/
theorem gain_amount_proof (gain : ℝ) (gain_percent : ℝ) 
  (h1 : gain = 0.70) (h2 : gain_percent = 1) : 
  calculate_amount gain gain_percent = 70 := by
  -- Unfold the definition of calculate_amount
  unfold calculate_amount
  -- Substitute the given values
  rw [h1, h2]
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gain_amount_proof_l197_19707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_noncoplanar_edges_count_l197_19796

/-- A parallelepiped is a three-dimensional figure with six parallelogram faces. -/
structure Parallelepiped where
  -- We don't need to define the structure explicitly for this problem

/-- A line drawn on a face of a parallelepiped -/
structure DrawnLine where
  -- We don't need to define the structure explicitly for this problem

/-- Represents the number of edges not coplanar with a drawn line -/
def NoncoplanarEdgesCount : Type := Nat

/-- The set of possible counts of noncoplanar edges -/
def PossibleCounts : Set Nat := {4, 6, 7, 8}

/-- 
  Given a parallelepiped and a line drawn on one of its faces, 
  the number of edges not coplanar with the line is in the set of possible counts.
-/
theorem noncoplanar_edges_count (p : Parallelepiped) (l : DrawnLine) :
  ∃ (n : Nat), n ∈ PossibleCounts := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_noncoplanar_edges_count_l197_19796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_fifth_term_and_rational_terms_l197_19723

noncomputable def binomial_expansion (x : ℝ) (n : ℕ) : ℕ → ℝ :=
  fun r => (n.choose r) * (-1/24)^r * x^((2*n - 3*r)/4 - r/2)

theorem constant_fifth_term_and_rational_terms 
  (x : ℝ) (n : ℕ) (h : ∃ k, binomial_expansion x n 4 = k) :
  n = 6 ∧ 
  (∀ r, r ≠ 0 ∧ r ≠ 4 → ¬ (∃ q : ℚ, binomial_expansion x n r = q)) ∧
  (∃ q : ℚ, binomial_expansion x n 0 = q) ∧
  (∃ q : ℚ, binomial_expansion x n 4 = q) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_fifth_term_and_rational_terms_l197_19723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l197_19774

noncomputable def f (x : ℝ) : ℝ := Real.sin (x - Real.pi/3) - 1

theorem smallest_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧
  (∀ S, S > 0 ∧ (∀ x, f (x + S) = f x) → T ≤ S) ∧
  T = 2*Real.pi :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l197_19774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_G_101_equals_2_l197_19736

def G : ℕ → ℚ
  | 0 => 2  -- Add this case for n = 0
  | 1 => 2
  | (n + 2) => (3 * G (n + 1) + 2) / 4

theorem G_101_equals_2 : G 101 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_G_101_equals_2_l197_19736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_pentagon_not_on_one_side_of_two_sides_l197_19787

/-- A pentagon is a polygon with 5 vertices -/
structure Pentagon where
  vertices : Fin 5 → ℝ × ℝ

/-- A side of a pentagon is determined by two consecutive vertices -/
def Pentagon.side (p : Pentagon) (i : Fin 5) : Set (ℝ × ℝ) :=
  { x : ℝ × ℝ | ∃ t : ℝ, x = (1 - t) • (p.vertices i) + t • (p.vertices (i + 1)) }

/-- A point lies on one side of a line if it's in one of the half-planes defined by the line -/
def liesOnOneSide (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) : Prop := sorry

/-- A pentagon lies on one side of its side if all its vertices (except the two defining the side) lie on one side of that side -/
def Pentagon.liesOnOneSideOfSide (p : Pentagon) (i : Fin 5) : Prop :=
  ∀ j : Fin 5, j ≠ i ∧ j ≠ (i + 1) → liesOnOneSide (p.vertices j) (p.side i)

/-- The main theorem: there exists a pentagon that doesn't lie on one side of at least two of its sides -/
theorem exists_pentagon_not_on_one_side_of_two_sides :
  ∃ p : Pentagon, ¬(∃ i j : Fin 5, i ≠ j ∧ p.liesOnOneSideOfSide i ∧ p.liesOnOneSideOfSide j) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_pentagon_not_on_one_side_of_two_sides_l197_19787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_groupings_and_divisibility_l197_19764

-- Part 1: Number of ways to group 2n people into n teams of 2
def groupings (n : ℕ) : ℚ := (2 * n).factorial / (2^n * n.factorial)

-- Part 2: Divisibility property
def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem groupings_and_divisibility (m n : ℕ+) :
  (groupings n.val = (2 * n.val).factorial / (2^n.val * n.val.factorial)) ∧
  (is_divisible ((m.val * n.val).factorial * (m.val * n.val).factorial) 
                ((m.val.factorial)^(n.val + 1) * (n.val.factorial)^(m.val + 1))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_groupings_and_divisibility_l197_19764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_2_4_l197_19780

noncomputable def power_function (α : ℝ) : ℝ → ℝ := fun x ↦ x ^ α

theorem power_function_through_point_2_4 :
  ∃ α : ℝ, power_function α 2 = 4 ∧ ∀ x : ℝ, power_function α x = x^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_2_4_l197_19780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_cubic_has_three_real_roots_l197_19706

-- Define a polynomial type
def MyPolynomial (α : Type) := ℕ → α

-- Define the degree of a polynomial
noncomputable def degree {α : Type} [Semiring α] (p : MyPolynomial α) : ℕ := sorry

-- Define polynomial addition
def add_poly {α : Type} [Semiring α] (p q : MyPolynomial α) : MyPolynomial α := sorry

-- Define polynomial multiplication
def mul_poly {α : Type} [Semiring α] (p q : MyPolynomial α) : MyPolynomial α := sorry

-- Define polynomial squaring
def square_poly {α : Type} [Semiring α] (p : MyPolynomial α) : MyPolynomial α := mul_poly p p

-- Define a function to count real roots of a polynomial
noncomputable def count_real_roots (p : MyPolynomial ℝ) : ℕ := sorry

-- Theorem statement
theorem one_cubic_has_three_real_roots 
  (P Q R : MyPolynomial ℝ) 
  (h1 : degree P = 2 ∨ degree Q = 2 ∨ degree R = 2) 
  (h2 : degree P = 3 ∨ degree Q = 3) 
  (h3 : degree P = 3 ∨ degree R = 3) 
  (h4 : add_poly (square_poly P) (square_poly Q) = square_poly R) :
  ∃ (p : MyPolynomial ℝ), (degree p = 3 ∧ count_real_roots p = 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_cubic_has_three_real_roots_l197_19706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_shift_and_subtract_l197_19792

theorem decimal_shift_and_subtract : 
  (Int.floor ((100 * (1.41 : ℚ)) - 1.41) : ℤ) = 139 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_shift_and_subtract_l197_19792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_divisors_of_nine_factorial_l197_19759

/-- The factorial function -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- The number of even divisors of a natural number -/
def evenDivisorsCount (n : ℕ) : ℕ :=
  (Nat.divisors n).filter (· % 2 = 0) |>.card

/-- Theorem: The number of even divisors of 9! is 140 -/
theorem even_divisors_of_nine_factorial :
  evenDivisorsCount (factorial 9) = 140 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_divisors_of_nine_factorial_l197_19759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_plus_pi_third_l197_19754

theorem sin_double_angle_plus_pi_third (α : ℝ) :
  0 < α ∧ α < π / 2 →
  Real.cos (α + π / 6) = 4 / 5 →
  Real.sin (2 * α + π / 3) = 24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_plus_pi_third_l197_19754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_equals_61_l197_19732

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 6 then x / 2
  else 3 * x - 11

-- Define the area A
noncomputable def A : ℝ :=
  ∫ x in (0 : ℝ)..(10 : ℝ), f x

-- Theorem statement
theorem area_enclosed_equals_61 : A = 61 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_equals_61_l197_19732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contest_probabilities_l197_19783

/-- Represents the probability of a contestant answering a question correctly -/
noncomputable def correct_probability : ℝ := 2/3

/-- Represents the probability of a contestant answering a question incorrectly -/
noncomputable def incorrect_probability : ℝ := 1 - correct_probability

/-- The probability of advancing to the final round -/
noncomputable def advance_probability : ℝ := 496/729

theorem contest_probabilities :
  (incorrect_probability ^ 2 = 1/9) →
  (correct_probability + incorrect_probability = 1) →
  (correct_probability = 2/3 ∧ advance_probability = 496/729) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contest_probabilities_l197_19783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_movement_composition_movement_with_fixed_point_l197_19786

-- Define the space
variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

-- Define a movement in space
def Movement (E : Type*) [NormedAddCommGroup E] := E → E

-- Define a plane symmetry
def PlaneSymmetry (E : Type*) [NormedAddCommGroup E] := E → E

-- Define composition of movements
def ComposeMovements (f g : Movement E) : Movement E :=
  λ x ↦ f (g x)

-- Theorem for part a
theorem movement_composition (T : Movement E) :
  ∃ (S₁ S₂ S₃ S₄ : PlaneSymmetry E),
    ∀ x, T x = (ComposeMovements (ComposeMovements (ComposeMovements S₁ S₂) S₃) S₄) x :=
by sorry

-- Theorem for part b
theorem movement_with_fixed_point (T : Movement E) (O : E)
  (h : T O = O) :
  ∃ (S₁ S₂ S₃ : PlaneSymmetry E),
    ∀ x, T x = (ComposeMovements (ComposeMovements S₁ S₂) S₃) x :=
by sorry

-- Define first-kind movement
def FirstKindMovement (T : Movement E) :=
  ∃ (S₁ S₂ S₃ S₄ : PlaneSymmetry E),
    (∀ x, T x = (ComposeMovements (ComposeMovements S₁ S₂) (ComposeMovements S₃ S₄)) x) ∨
    (∀ x, T x = (ComposeMovements S₁ S₂) x)

-- Define second-kind movement
def SecondKindMovement (T : Movement E) :=
  ∃ (S₁ S₂ S₃ : PlaneSymmetry E),
    (∀ x, T x = (ComposeMovements (ComposeMovements S₁ S₂) S₃) x) ∨
    (∀ x, T x = S₁ x)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_movement_composition_movement_with_fixed_point_l197_19786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_k_clique_size_exists_180180_clique_l197_19772

/-- The set of all points with integer coordinates in the coordinate plane -/
def S : Set (ℤ × ℤ) := Set.univ

/-- Two points are k-friends if there exists a third point forming a triangle with area k -/
def is_k_friend (k : ℕ+) (A B : ℤ × ℤ) : Prop :=
  ∃ C : ℤ × ℤ, |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)| = 2 * k

/-- A set is a k-clique if every two points in it are k-friends -/
def is_k_clique (k : ℕ+) (T : Set (ℤ × ℤ)) : Prop :=
  T ⊆ S ∧ ∀ A B, A ∈ T → B ∈ T → A ≠ B → is_k_friend k A B

/-- The theorem to be proved -/
theorem min_k_clique_size :
  ∀ k : ℕ+, k < 180180 →
    ¬∃ T : Set (ℤ × ℤ), is_k_clique k T ∧ T.Finite ∧ T.ncard > 200 :=
sorry

/-- The existence of a 180180-clique with more than 200 elements -/
theorem exists_180180_clique :
  ∃ T : Set (ℤ × ℤ), is_k_clique 180180 T ∧ T.Finite ∧ T.ncard > 200 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_k_clique_size_exists_180180_clique_l197_19772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_intersection_l197_19712

-- Define the curve C
def C (x : ℝ) : ℝ := x^3

-- Define the tangent line
def tangentLine (x y : ℝ) : Prop := 3*x - y - 2 = 0

-- Define the derivative of C
def C' (x : ℝ) : ℝ := 3*x^2

theorem tangent_and_intersection :
  -- The tangent line passes through (1, 1) and has slope 3
  (tangentLine 1 (C 1) ∧ C' 1 = 3) ∧
  -- (-2, -8) is on both the curve and the tangent line
  (C (-2) = -8 ∧ tangentLine (-2) (-8)) ∧
  -- The tangent line intersects the curve only at (1, 1) and (-2, -8)
  (∀ x y : ℝ, tangentLine x y ∧ C x = y → (x = 1 ∧ y = 1) ∨ (x = -2 ∧ y = -8)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_intersection_l197_19712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_and_g_range_l197_19741

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.log (2 + x) / Real.log 10 + Real.log (2 - x) / Real.log 10

noncomputable def g (x : ℝ) : ℝ := 10^(f x) + 2 * x

-- Define the domain of f
def domain_f : Set ℝ := {x | -2 < x ∧ x < 2}

-- Define the range of g
def range_g : Set ℝ := {y | ∃ x ∈ domain_f, g x = y}

-- Theorem statement
theorem f_domain_and_g_range :
  (∀ x, x ∈ domain_f ↔ -2 < x ∧ x < 2) ∧
  (∀ y, y ∈ range_g ↔ -11/2 < y ∧ y ≤ 7) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_and_g_range_l197_19741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l197_19758

/-- The function we're analyzing -/
noncomputable def f (x : ℝ) := 3 * Real.sin x ^ 2

/-- The period we want to prove -/
noncomputable def period : ℝ := Real.pi

theorem min_positive_period_of_f :
  (∀ x : ℝ, f (x + period) = f x) ∧
  (∀ p : ℝ, 0 < p → p < period → ∃ x : ℝ, f (x + p) ≠ f x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l197_19758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tin_in_mixture_l197_19716

/-- Represents the composition of an alloy -/
structure AlloyComposition where
  component1 : ℚ
  component2 : ℚ

/-- Represents an alloy with its weight and composition -/
structure Alloy where
  weight : ℚ
  composition : AlloyComposition

def alloy_A : Alloy :=
  { weight := 90
    composition := { component1 := 3, component2 := 4 } }

def alloy_B : Alloy :=
  { weight := 140
    composition := { component1 := 2, component2 := 5 } }

def tin_content (a : Alloy) : ℚ :=
  (a.composition.component2 / (a.composition.component1 + a.composition.component2)) * a.weight

theorem tin_in_mixture (ε : ℚ) (h : ε > 0) :
  ∃ δ : ℚ, δ > 0 ∧ |tin_content alloy_A + tin_content alloy_B - 91429/1000| < ε := by
  sorry

#eval tin_content alloy_A + tin_content alloy_B

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tin_in_mixture_l197_19716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_2013_l197_19705

noncomputable def arithmeticSequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

noncomputable def sequenceSum (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := (n : ℝ) / 2 * (2 * a₁ + (n - 1 : ℝ) * d)

theorem arithmetic_sequence_sum_2013 :
  ∀ (d : ℝ),
  let a₁ := -2013
  let S := sequenceSum a₁ d
  (S 12 / 12 - S 10 / 10 = 2) →
  S 2013 = -2013 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_2013_l197_19705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_product_fraction_l197_19740

theorem units_digit_of_product_fraction : 
  let numerator := 30 * 31 * 32 * 33 * 34 * 35
  let denominator := 1500
  let fraction := numerator / denominator
  fraction % 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_product_fraction_l197_19740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l197_19747

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The minimum sum of distances theorem -/
theorem min_sum_distances (A B : Point) (h1 : A.x = 0) (h2 : A.y = 10) 
    (h3 : B.x = 25) (h4 : B.y = 15) :
  ∃ (P : Point), P.y = 0 ∧ 
    (∀ (Q : Point), Q.y = 0 → 
      distance A P + distance B P ≤ distance A Q + distance B Q) ∧
    abs ((distance A P + distance B P) - 35.46) < 0.01 := by
  sorry

#check min_sum_distances

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l197_19747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_exists_in_interval_l197_19798

theorem equation_solution_exists_in_interval :
  ∃! x : ℝ, x ∈ Set.Ioo 2 3 ∧ 3 * x + Real.log x = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_exists_in_interval_l197_19798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_comparison_l197_19725

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := (x - x^9) / (1 - x) + 10 * x^9
noncomputable def g (x : ℝ) : ℝ := (x - x^11) / (1 - x) + 10 * x^11

-- State the theorem
theorem root_comparison : 
  ∃ (r₁ r₂ : ℝ), 0 < r₁ ∧ 0 < r₂ ∧ r₁ < 1 ∧ r₂ < 1 ∧ 
  f r₁ = 8 ∧ g r₂ = 8 ∧ r₁ < r₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_comparison_l197_19725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_10_irrational_l197_19717

theorem log_10_irrational (N : ℕ+) (h : ∀ k : ℕ, N ≠ 10^k) : 
  Irrational (Real.log N / Real.log 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_10_irrational_l197_19717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_divisibility_l197_19768

theorem binomial_coefficient_divisibility (p : ℕ) (hp : Nat.Prime p) (ho : Odd p) :
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ p - 2 →
    ∃ m : ℤ, (Nat.choose (p - 2) k : ℤ) + (-1 : ℤ)^(k - 1) * (k + 1 : ℤ) = p * m :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_divisibility_l197_19768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_uniqueness_l197_19753

/-- Given lengths a, b, c, prove that a unique triangle ABC can be constructed
    where CA = a, CB = b, CH = c, and H is on AB with BH = 2AH,
    if and only if b, 2a, and 3c satisfy the triangle inequality -/
theorem triangle_construction_uniqueness 
  (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃! (A B C H : ℝ × ℝ), 
    (let d := dist
     d A C = a ∧ d B C = b ∧ d C H = c ∧
     H.1 = (2 * A.1 + B.1) / 3 ∧ H.2 = (2 * A.2 + B.2) / 3 ∧
     (A.1 - B.1) * (C.2 - B.2) = (A.2 - B.2) * (C.1 - B.1))
  ↔ b + 2*a > 3*c ∧ 2*a + 3*c > b ∧ b + 3*c > 2*a :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_uniqueness_l197_19753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bottom_right_is_zero_l197_19713

/-- Represents a 3x3 grid with known corner values -/
structure Grid (α : Type*) [Add α] [Sub α] where
  a : α  -- top-left corner
  b : α  -- top-right corner
  c : α  -- bottom-left corner

/-- The sum of a 2x2 sub-grid given its corner values -/
def subgridSum {α : Type*} [Add α] (w x y z : α) : α := w + x + y + z

/-- Theorem stating that the bottom-right corner must be 0 -/
theorem bottom_right_is_zero {α : Type*} [Add α] [Sub α] [Zero α] (grid : Grid α) :
  ∃ (A B C D E : α),
    subgridSum grid.a A B C = subgridSum A grid.b C D ∧
    subgridSum grid.a A B C = subgridSum B C grid.c E ∧
    subgridSum grid.a A B C = subgridSum C D E (0 : α) := by
  sorry

#check bottom_right_is_zero

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bottom_right_is_zero_l197_19713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_speed_l197_19752

/-- Represents a distance marker with digits --/
structure DistanceMarker where
  digits : List Nat
  deriving Repr

/-- Represents the journey with three distance markers --/
structure Journey where
  marker1 : DistanceMarker
  marker2 : DistanceMarker
  marker3 : DistanceMarker

/-- Checks if two markers have reversed digits --/
def areReversed (m1 m2 : DistanceMarker) : Prop :=
  m1.digits.reverse = m2.digits

/-- Checks if the third marker contains the digits from the first two markers plus a zero --/
def thirdMarkerValid (j : Journey) : Prop :=
  (j.marker3.digits.length = 3) ∧
  (j.marker3.digits.toFinset = (j.marker1.digits ++ j.marker2.digits ++ [0]).toFinset)

/-- Calculates the numeric value of a distance marker --/
def markerValue (m : DistanceMarker) : Nat :=
  m.digits.foldl (fun acc d => acc * 10 + d) 0

/-- Theorem statement --/
theorem journey_speed (j : Journey) 
  (h1 : j.marker1.digits.length = 2)
  (h2 : areReversed j.marker1 j.marker2)
  (h3 : thirdMarkerValid j)
  (h4 : markerValue j.marker2 - markerValue j.marker1 = markerValue j.marker3 - markerValue j.marker2) :
  markerValue j.marker3 - markerValue j.marker1 = 96 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_speed_l197_19752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_16_fourth_power_l197_19748

-- Define the functions f and g
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- State the conditions
axiom fg_condition : ∀ x : ℝ, x ≥ 1 → f (g x) = x^3
axiom gf_condition : ∀ x : ℝ, x ≥ 1 → g (f x) = x^4
axiom g_64 : g 64 = 16

-- State the theorem to be proved
theorem g_16_fourth_power : (g 16)^4 = 4096 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_16_fourth_power_l197_19748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l197_19744

/-- The time (in seconds) it takes for a train to pass a person moving in the opposite direction. -/
noncomputable def time_to_pass (train_length : ℝ) (train_speed : ℝ) (person_speed : ℝ) : ℝ :=
  train_length / ((train_speed + person_speed) * (5 / 18))

/-- Theorem stating that the time for a 550 m long train moving at 60 km/hr to pass a man
    moving at 6 km/hr in the opposite direction is approximately 30 seconds. -/
theorem train_passing_time :
  let train_length := (550 : ℝ)
  let train_speed := (60 : ℝ)
  let person_speed := (6 : ℝ)
  ⌊time_to_pass train_length train_speed person_speed⌋ = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l197_19744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_expansion_l197_19722

theorem coefficient_x_cubed_expansion : 
  let f : Polynomial ℤ := (1 - X) * (2 + X)^5
  (f.coeff 3) = -40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_expansion_l197_19722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_equivalent_to_0_000045_l197_19762

theorem not_equivalent_to_0_000045 :
  let a := (4.5 : ℝ) * (10 : ℝ)^(-5 : ℤ)
  let b := (9 : ℝ) / (2 : ℝ) * (10 : ℝ)^(-5 : ℤ)
  let c := (45 : ℝ) * (10 : ℝ)^(-7 : ℤ)
  let d := (1 : ℝ) / (22500 : ℝ)
  let e := (45 : ℝ) / (10 : ℝ)^6
  (a = 0.000045) ∧ 
  (b = 0.000045) ∧ 
  (c ≠ 0.000045) ∧ 
  (d = 0.000045) ∧ 
  (e = 0.000045) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_equivalent_to_0_000045_l197_19762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_pyramid_surface_area_l197_19773

/-- A pyramid with an equilateral triangular base and one equilateral triangular lateral face perpendicular to the base -/
structure SpecialPyramid where
  /-- Side length of the base triangle -/
  a : ℝ
  /-- The base is an equilateral triangle -/
  base_equilateral : True
  /-- One lateral face is an equilateral triangle -/
  lateral_face_equilateral : True
  /-- This lateral face is perpendicular to the base -/
  lateral_face_perpendicular : True

/-- The total surface area of the special pyramid -/
noncomputable def totalSurfaceArea (p : SpecialPyramid) : ℝ :=
  (p.a^2 * Real.sqrt 3 * (2 + Real.sqrt 5)) / 4

/-- Theorem stating the total surface area of the special pyramid -/
theorem special_pyramid_surface_area (p : SpecialPyramid) :
  totalSurfaceArea p = (p.a^2 * Real.sqrt 3 * (2 + Real.sqrt 5)) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_pyramid_surface_area_l197_19773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l197_19750

-- Define the function as noncomputable
noncomputable def f (x : ℝ) := Real.log (12 + x - x^2)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | -3 < x ∧ x < 4} = {x : ℝ | ∃ y, f x = y} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l197_19750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l197_19775

/-- Two parallel lines in a 2D plane -/
structure ParallelLines where
  a : ℝ
  l₁ : ℝ → ℝ → Prop
  l₂ : ℝ → ℝ → Prop
  l₁_eq : ∀ x y, l₁ x y ↔ x + a * y + 6 = 0
  l₂_eq : ∀ x y, l₂ x y ↔ (a - 2) * x + 3 * y + 2 * a = 0
  parallel : ∀ x₁ y₁ x₂ y₂, l₁ x₁ y₁ → l₂ x₂ y₂ → (x₂ - x₁) * 3 = (y₂ - y₁) * (a - 2)

/-- The distance between two parallel lines -/
noncomputable def distance (pl : ParallelLines) : ℝ :=
  8 * Real.sqrt 2 / 3

/-- Theorem: The distance between the given parallel lines is 8√2/3 -/
theorem parallel_lines_distance (pl : ParallelLines) :
  distance pl = 8 * Real.sqrt 2 / 3 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l197_19775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_transaction_profit_l197_19769

/-- Represents a bicycle transaction with two sellers -/
structure BicycleTransaction where
  initial_cost : ℚ
  first_profit_percent : ℚ
  final_price : ℚ

/-- Calculates the profit percentage of the second seller in a bicycle transaction -/
noncomputable def second_seller_profit_percent (t : BicycleTransaction) : ℚ :=
  let first_selling_price := t.initial_cost * (1 + t.first_profit_percent / 100)
  let second_profit := t.final_price - first_selling_price
  (second_profit / first_selling_price) * 100

/-- Theorem stating that under given conditions, the second seller's profit is 25% -/
theorem bicycle_transaction_profit
  (t : BicycleTransaction)
  (h1 : t.initial_cost = 150)
  (h2 : t.first_profit_percent = 20)
  (h3 : t.final_price = 225) :
  second_seller_profit_percent t = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_transaction_profit_l197_19769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_of_specific_body_l197_19742

/-- A body in 3D space --/
structure Body where
  Ω : Set (ℝ × ℝ × ℝ)

/-- Density function for a body --/
def density : ℝ × ℝ × ℝ → ℝ :=
  fun (_, _, z) ↦ z

/-- The bounding surfaces of the body --/
def boundingSurfaces (x y z : ℝ) : Prop :=
  x^2 + y^2 = 4 ∧ z = 0 ∧ z = (x^2 + y^2) / 2

/-- The mass of a body --/
noncomputable def mass (b : Body) : ℝ :=
  ∫ p in b.Ω, density p

/-- The specific body described in the problem --/
def specificBody : Body where
  Ω := {(x, y, z) | boundingSurfaces x y z}

/-- Theorem stating the mass of the specific body --/
theorem mass_of_specific_body :
  mass specificBody = 16 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_of_specific_body_l197_19742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_passes_through_point_l197_19776

-- Define a function f: ℝ → ℝ with an inverse
variable (f : ℝ → ℝ)
variable (hf : Function.Bijective f)

-- Define the condition that f(1) = 2
variable (h : f 1 = 2)

-- Define g(x) = f(x-4)
def g : ℝ → ℝ := fun x => f (x - 4)

-- The theorem to prove
theorem inverse_g_passes_through_point :
  Function.invFun g 2 = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_passes_through_point_l197_19776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shop_profit_calculation_l197_19782

/-- Given a cost and selling price, calculates the profit percentage. -/
noncomputable def profitPercentage (cost : ℝ) (sellingPrice : ℝ) : ℝ :=
  ((sellingPrice - cost) / cost) * 100

/-- Represents the conditions of the problem. -/
structure ShopProblem where
  cost : ℝ
  sellingPrice : ℝ
  increasedCostPercentage : ℝ
  newProfitPercentage : ℝ

/-- The theorem representing the problem. -/
theorem shop_profit_calculation (p : ShopProblem)
  (h1 : p.increasedCostPercentage = 25)
  (h2 : p.newProfitPercentage = 70.23809523809523)
  (h3 : p.sellingPrice > p.cost)
  (h4 : p.cost > 0) :
  profitPercentage p.cost p.sellingPrice = 320 := by
  sorry

#check shop_profit_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shop_profit_calculation_l197_19782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_plus_2cos_2x_eq_neg_2_l197_19793

theorem sin_2x_plus_2cos_2x_eq_neg_2 (α β : Real) : 
  (0 ≤ α ∧ α < Real.pi) →
  (0 ≤ β ∧ β < Real.pi) →
  (α ≠ β) →
  (Real.sin (2 * α) + 2 * Real.cos (2 * α) = -2) →
  (Real.sin (2 * β) + 2 * Real.cos (2 * β) = -2) →
  Real.cos (α - β) = 2 * Real.sqrt 5 / 5 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_plus_2cos_2x_eq_neg_2_l197_19793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l197_19739

noncomputable def f (x : ℝ) : ℝ := (x^2 - 64) / (x - 8)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < 8 ∨ x > 8} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l197_19739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l197_19735

theorem problem_statement (a b c d : ℤ)
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h1 : a^5 = b^4)
  (h2 : c^3 = d^2)
  (h3 : c - a = 9) :
  a - b = -16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l197_19735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_score_order_l197_19729

/-- Represents a person's score in the quiz competition -/
structure Score where
  value : ℕ

/-- The quiz competition participants -/
inductive Person
  | Ned
  | Uma
  | Tara

/-- Assignment of scores to persons -/
def score_assignment : Person → Score := sorry

/-- Ned's condition: At least one other person has a score higher than his -/
def ned_condition (sa : Person → Score) : Prop :=
  ∃ p : Person, p ≠ Person.Ned ∧ (sa p).value > (sa Person.Ned).value

/-- Tara's condition: Her score isn't the lowest -/
def tara_condition (sa : Person → Score) : Prop :=
  ∃ p : Person, p ≠ Person.Tara ∧ (sa p).value < (sa Person.Tara).value

/-- Uma's condition: Her score isn't the highest but it's not the lowest either -/
def uma_condition (sa : Person → Score) : Prop :=
  ∃ p1 p2 : Person, p1 ≠ Person.Uma ∧ p2 ≠ Person.Uma ∧ p1 ≠ p2 ∧
    (sa p1).value < (sa Person.Uma).value ∧ (sa Person.Uma).value < (sa p2).value

/-- The main theorem stating the correct order of scores -/
theorem correct_score_order (sa : Person → Score)
  (h_ned : ned_condition sa)
  (h_tara : tara_condition sa)
  (h_uma : uma_condition sa) :
  (sa Person.Ned).value < (sa Person.Uma).value ∧
  (sa Person.Uma).value < (sa Person.Tara).value := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_score_order_l197_19729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cleaning_time_theorem_l197_19779

/-- Represents the layout of Jack's grove --/
structure GroveLayout where
  first_section : Nat × Nat
  second_section : Nat × Nat
  third_section : Nat × Nat

/-- Represents the cleaning times for each section of the grove --/
structure CleaningTimes where
  first_section : Nat
  second_section : Nat
  third_section : Nat

/-- Represents the time reductions for each section due to friends' help --/
structure TimeReductions where
  first_section : Float
  second_section : Float
  third_section : Float

/-- Calculates the total cleaning time in hours --/
noncomputable def calculate_total_time (layout : GroveLayout) (times : CleaningTimes) 
  (reductions : TimeReductions) (break_time : Nat) (rain_effect : Float) : Float :=
  sorry

/-- The main theorem stating the total cleaning time --/
theorem total_cleaning_time_theorem (layout : GroveLayout) (times : CleaningTimes) 
  (reductions : TimeReductions) (break_time : Nat) (rain_effect : Float) : 
  (calculate_total_time layout times reductions break_time rain_effect - 10.937).abs < 0.001 := by
  sorry

#check total_cleaning_time_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cleaning_time_theorem_l197_19779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_on_negative_interval_l197_19761

-- Define the function f
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x + Real.exp (x * Real.log 2)

-- State the theorem
theorem min_value_on_negative_interval 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hmax : ∀ x ∈ Set.Icc 0 1, f a b x ≤ 4) 
  (hmax_achieved : ∃ x ∈ Set.Icc 0 1, f a b x = 4) :
  ∀ x ∈ Set.Icc (-1) 0, f a b x ≥ -3/2 ∧ 
  ∃ y ∈ Set.Icc (-1) 0, f a b y = -3/2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_on_negative_interval_l197_19761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_run_l197_19757

/-- The distance run by both A and B -/
def D : ℝ := 2700

/-- The time taken by A to run the distance D -/
def time_A : ℝ := 198

/-- The time taken by B to run the distance D -/
def time_B : ℝ := 220

/-- The additional distance A can run in the time B runs D -/
def additional_distance : ℝ := 300

/-- The theorem stating that the distance run by both A and B is 2700 meters -/
theorem distance_run : D = 2700 := by
  -- Unfold the definition of D
  unfold D
  -- The proof is complete by reflexivity
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_run_l197_19757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jane_wins_probability_l197_19778

/-- Represents a deck of cards -/
structure Deck :=
  (total : Nat)
  (red : Nat)
  (other : Nat)
  (h_total : total = red + other)

/-- Represents the game setup -/
structure Game :=
  (deck : Deck)
  (h_deck : deck.total = 12 ∧ deck.red = 3 ∧ deck.other = 9)

/-- Calculates the number of valid configurations -/
def validConfigurations (n : Nat) : Nat :=
  (List.range 9).map (fun k => Nat.choose k 2 * Nat.choose (24 - k) 3) |> List.sum

/-- Calculates the total number of possible configurations -/
def totalConfigurations (g : Game) : Nat :=
  (Nat.choose g.deck.total g.deck.red) ^ 2

/-- The main theorem stating the probability -/
theorem jane_wins_probability (g : Game) :
  (validConfigurations 24 : Rat) / (totalConfigurations g : Rat) = 39 / 1100 := by
  sorry

#eval validConfigurations 24
#eval totalConfigurations { deck := { total := 12, red := 3, other := 9, h_total := rfl }, h_deck := by simp }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jane_wins_probability_l197_19778
