import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_theorem_l169_16918

/-- The function f(x) = ln x - x^2 --/
noncomputable def f (x : ℝ) : ℝ := Real.log x - x^2

/-- The line equation x + y - 3 = 0 --/
def line_equation (x y : ℝ) : Prop := x + y - 3 = 0

/-- Theorem: The shortest distance between a point on f(x) and the line is 3√2/2 --/
theorem shortest_distance_theorem :
  ∃ (x₀ y₀ : ℝ), 
    (y₀ = f x₀) ∧ 
    (∀ (x y : ℝ), line_equation x y → 
      (x₀ - x)^2 + (y₀ - y)^2 ≥ (3 * Real.sqrt 2 / 2)^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_theorem_l169_16918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rainfall_tuesday_l169_16964

/-- Rainfall on a specific day in a week with special conditions -/
noncomputable def rainfall_on_specific_day (total_rainfall : ℝ) (num_days : ℕ) (avg_rainfall : ℝ) : ℝ :=
  total_rainfall / 2

theorem rainfall_tuesday (total_rainfall : ℝ) (num_days : ℕ) (avg_rainfall : ℝ) 
    (h1 : num_days = 7)
    (h2 : avg_rainfall = 3)
    (h3 : total_rainfall = avg_rainfall * num_days) : 
  rainfall_on_specific_day total_rainfall num_days avg_rainfall = 10.5 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval rainfall_on_specific_day 21 7 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rainfall_tuesday_l169_16964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_list_count_l169_16944

def arithmetic_sequence_count (start end_ diff : ℤ) : ℕ :=
  let n := ((start - end_) / diff.natAbs + 1).natAbs
  n

theorem list_count : arithmetic_sequence_count 195 12 (-3) = 62 := by
  -- Proof goes here
  sorry

#eval arithmetic_sequence_count 195 12 (-3)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_list_count_l169_16944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y₀_range_l169_16925

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the point M
noncomputable def point_M (y₀ : ℝ) : ℝ × ℝ := (Real.sqrt 3, y₀)

-- Define the angle OMN
noncomputable def angle_OMN (y₀ : ℝ) : ℝ := Real.arccos (1 / (2 * Real.sqrt (3 + y₀^2)))

-- Theorem statement
theorem y₀_range (y₀ : ℝ) :
  (angle_OMN y₀ ≥ π / 6) → (-1 ≤ y₀ ∧ y₀ ≤ 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_y₀_range_l169_16925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l169_16972

/-- A plane in 3D space -/
structure Plane where

/-- A point in 3D space -/
structure Point where

/-- A triangle in 3D space -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Projection of a triangle onto a plane from a point -/
def project (T : Triangle) (S : Plane) (P : Point) : Triangle :=
  sorry

/-- Predicate to check if a triangle is isosceles right -/
def isIsoscelesRight (T : Triangle) : Prop :=
  sorry

/-- Predicate to check if a triangle lies on a plane -/
def liesOn (T : Triangle) (S : Plane) : Prop :=
  sorry

/-- Predicate to check if two planes are parallel -/
def isParallel (S1 S2 : Plane) : Prop :=
  sorry

/-- The plane containing a triangle -/
def planeOf (T : Triangle) : Plane :=
  sorry

theorem projection_theorem (S : Plane) (ABC : Triangle) :
  ¬(liesOn ABC S) →
  (∃ P : Point, isIsoscelesRight (project ABC S P)) ∨
  (isParallel (planeOf ABC) S ∧ ¬(isIsoscelesRight ABC)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l169_16972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_angle_third_quadrant_l169_16913

theorem tan_half_angle_third_quadrant (a : Real) :
  (π < a ∧ a < 3*π/2) →  -- a is in the third quadrant
  Real.cos a = -3/5 →
  Real.tan (a/2) = -2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_angle_third_quadrant_l169_16913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_greater_than_c_greater_than_b_l169_16948

-- Define the constants
noncomputable def a : ℝ := 2^(3/2 : ℝ)
noncomputable def b : ℝ := Real.log 1.5 / Real.log (1/2)
noncomputable def c : ℝ := (1/2)^(3/2 : ℝ)

-- State the theorem
theorem a_greater_than_c_greater_than_b : a > c ∧ c > b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_greater_than_c_greater_than_b_l169_16948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_20_10_l169_16930

theorem binomial_coefficient_20_10 (h1 : Nat.choose 18 8 = 31824) 
                                   (h2 : Nat.choose 18 9 = 48620) 
                                   (h3 : Nat.choose 18 10 = 43758) : 
  Nat.choose 20 10 = 172822 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_20_10_l169_16930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_specific_l169_16994

/-- The area of the shaded region formed by the overlap of two sectors --/
noncomputable def shaded_area (r : ℝ) (θ : ℝ) : ℝ :=
  2 * (θ / 360 * Real.pi * r^2 - 1/2 * r^2 * Real.sin (θ * Real.pi / 180))

/-- Theorem stating the area of the shaded region for the given problem --/
theorem shaded_area_specific : shaded_area 15 45 = (225 * Real.pi - 450 * Real.sqrt 2) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_specific_l169_16994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_at_5_50_l169_16934

-- Define the angle formula
noncomputable def clock_angle (h : ℕ) (m : ℕ) : ℝ :=
  |60 * (h : ℝ) - 11 * (m : ℝ)| / 2

-- Theorem statement
theorem angle_at_5_50 : clock_angle 5 50 = 125 := by
  -- Unfold the definition of clock_angle
  unfold clock_angle
  -- Simplify the expression
  simp
  -- The proof is completed with sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_at_5_50_l169_16934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_even_l169_16940

-- Define the function g as noncomputable
noncomputable def g (x : ℝ) : ℝ := Real.log (Real.cos x + Real.sqrt (1 + Real.sin x ^ 2))

-- Theorem statement
theorem g_is_even : ∀ x : ℝ, g (-x) = g x := by
  -- The proof is omitted and replaced with 'sorry'
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_even_l169_16940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l169_16963

-- Define the function f(x) as noncomputable
noncomputable def f (a b x : ℝ) : ℝ := Real.log x / Real.log a + x - b

-- State the theorem
theorem zero_in_interval (a b : ℝ) (ha : 0 < a) (ha_neq_1 : a ≠ 1) 
  (ha_bound : 2 < a ∧ a < 3) (hb_bound : 3 < b ∧ b < 4) :
  ∃ x₀ : ℝ, f a b x₀ = 0 ∧ 2 < x₀ ∧ x₀ < 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l169_16963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_in_binomial_expansion_l169_16904

/-- The coefficient of x^2 in the binomial expansion of (x^2/2 - 1/√x)^6 -/
def coefficient_x_squared : ℚ := 15/4

/-- The binomial expansion of (x^2/2 - 1/√x)^6 -/
noncomputable def binomial_expansion (x : ℝ) : ℝ := (x^2/2 - 1/Real.sqrt x)^6

/-- Theorem stating that the coefficient of x^2 in the binomial expansion is correct -/
theorem coefficient_x_squared_in_binomial_expansion :
  ∃ (a b : ℝ), ∀ x, x > 0 → 
    binomial_expansion x = a * x^2 + b * coefficient_x_squared + 
      (binomial_expansion x - a * x^2 - b * coefficient_x_squared) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_in_binomial_expansion_l169_16904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_l169_16973

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 2 * Real.sin (x - Real.pi / 4) - Real.sin x

theorem f_is_even : ∀ x : ℝ, f (-x) = f x := by
  intro x
  simp [f]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_l169_16973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_pumped_in_30_minutes_l169_16993

/-- Represents the water pumping rate in gallons per hour -/
noncomputable def pump_rate : ℚ := 600

/-- Represents the time in hours for which we want to calculate the water pumped -/
noncomputable def time : ℚ := 1/2

/-- Calculates the amount of water pumped given a rate and time -/
noncomputable def water_pumped (rate : ℚ) (t : ℚ) : ℚ := rate * t

/-- Theorem stating that the machine pumps 300 gallons in 30 minutes -/
theorem water_pumped_in_30_minutes : 
  water_pumped pump_rate time = 300 := by
  -- Unfold the definitions
  unfold water_pumped pump_rate time
  -- Simplify the arithmetic
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_pumped_in_30_minutes_l169_16993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_consecutive_numbers_with_digit_sum_property_l169_16987

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

def isValidSequence (start : ℕ) (length : ℕ) : Prop :=
  ∀ i : ℕ, i ∈ Finset.range length → (i.succ ∣ sumOfDigits (start + i))

theorem max_consecutive_numbers_with_digit_sum_property :
  (∃ start : ℕ, isValidSequence start 21) ∧
  (∀ n : ℕ, n > 21 → ¬∃ start : ℕ, isValidSequence start n) :=
sorry

#check max_consecutive_numbers_with_digit_sum_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_consecutive_numbers_with_digit_sum_property_l169_16987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_surface_area_from_hexagon_section_l169_16986

/-- Given a cube whose cross-section is a regular hexagon with area Q,
    prove that the total surface area of the cube is (8Q√3)/3 -/
theorem cube_surface_area_from_hexagon_section (Q : ℝ) (Q_pos : Q > 0) :
  ∃ (a : ℝ), a > 0 ∧
    (3 * Real.sqrt 3 / 2) * a^2 = Q ∧
    6 * ((2 * a) / Real.sqrt 2)^2 = (8 * Q * Real.sqrt 3) / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_surface_area_from_hexagon_section_l169_16986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_break_even_proof_l169_16950

def expenses : ℚ := 11 + 3 + 16 + 8 + 10 + 35 + 3.5
def wifi_connection_fee : ℚ := 5
def hourly_rate : ℚ := 12
def wifi_rate : ℚ := 1

def break_even_hours : ℕ :=
  (((expenses + wifi_connection_fee) / (hourly_rate - wifi_rate)).ceil).toNat

theorem break_even_proof :
  break_even_hours = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_break_even_proof_l169_16950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_shapes_sum_of_areas_formula_l169_16977

/-- Represents the dimensions of a rectangular paper -/
structure PaperDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular paper -/
noncomputable def area (d : PaperDimensions) : ℝ := d.length * d.width

/-- Represents the state of the paper after folding -/
structure FoldedPaper where
  n : ℕ  -- number of folds
  shapes : List PaperDimensions

/-- The initial paper dimensions -/
def initial_paper : PaperDimensions := { length := 20, width := 12 }

/-- Function to calculate the sum of areas after n folds -/
noncomputable def sum_of_areas (n : ℕ) : ℝ := 240 * (3 - (n + 3) / 2^n)

/-- Theorem stating the number of shapes after n folds -/
theorem number_of_shapes (paper : FoldedPaper) :
  paper.shapes.length = paper.n + 1 := by sorry

/-- Theorem stating the sum of areas after n folds -/
theorem sum_of_areas_formula (paper : FoldedPaper) :
  (paper.shapes.map area).sum = sum_of_areas paper.n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_shapes_sum_of_areas_formula_l169_16977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l169_16916

-- Define the function f
noncomputable def f (x a : ℝ) : ℝ := 2 * Real.log x + x^2 - a * x + 2

-- Define the theorem
theorem range_of_m (a : ℝ) (h₁ : a ∈ Set.Icc (-2) 0) :
  ∃ (x₀ : ℝ) (h₂ : x₀ ∈ Set.Ioc 0 1),
    ∀ (m : ℝ), (f x₀ a > a^2 + 3*a + 2 - 2*m*Real.exp (a*(a+1))) →
      m ∈ Set.Icc (-1/2) (5*Real.exp 2/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l169_16916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_n_14_sufficient_not_necessary_l169_16975

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 3 * x + 4 * y - 3 = 0
def l₂ (x y n : ℝ) : Prop := 6 * x + 8 * y + n = 0

-- Define the distance function between two lines
noncomputable def distance (a b c₁ c₂ : ℝ) : ℝ := |c₁ - c₂| / Real.sqrt (a^2 + b^2)

-- Theorem statement
theorem n_14_sufficient_not_necessary :
  (∀ n : ℝ, n = 14 → distance 3 4 (-3) (-n/2) = 2) ∧
  (∃ n : ℝ, n ≠ 14 ∧ distance 3 4 (-3) (-n/2) = 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_n_14_sufficient_not_necessary_l169_16975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_properties_l169_16936

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_decreasing : ∀ x y, x < y → f x > f y
axiom f_domain : Set.Icc 1 3 = {x | ∃ y, f x = y}
axiom f_range : Set.Icc 4 7 = {y | ∃ x, f x = y}
axiom f_has_inverse : Function.Bijective f

-- Define the inverse function
noncomputable def f_inv : ℝ → ℝ := Function.invFun f

-- Theorem to prove
theorem inverse_properties :
  (∀ x y, x < y → f_inv x < f_inv y) ∧
  (∀ x, f_inv x ≤ 3) ∧
  (∃ x, f_inv x = 3) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_properties_l169_16936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_6_value_l169_16949

def sequence_a : ℕ → ℚ
  | 0 => 7  -- Added case for 0
  | 1 => 7
  | n + 2 => 1/2 * sequence_a (n + 1) + 3

theorem a_6_value : sequence_a 6 = 193/32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_6_value_l169_16949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_function_properties_l169_16943

noncomputable def f (a b x : ℝ) := a - b * Real.cos (2 * x)
noncomputable def g (a b x : ℝ) := -4 * a * Real.sin (3 * b * x + Real.pi / 3)

theorem trigonometric_function_properties
  (a b : ℝ)
  (h_b_pos : b > 0)
  (h_max : ∀ x, f a b x ≤ 3/2)
  (h_min : ∀ x, f a b x ≥ -1/2)
  (h_max_exists : ∃ x, f a b x = 3/2)
  (h_min_exists : ∃ x, f a b x = -1/2) :
  (∀ x, g a b (x + 2*Real.pi/(3*b)) = g a b x) ∧
  (∀ x, g a b x ≤ 2) ∧
  (∀ k : ℤ, g a b (-5*Real.pi/(18*b) + 2*k*Real.pi/(3*b)) = 2) ∧
  (∀ x, g a b x = 2 → ∃ k : ℤ, x = -5*Real.pi/(18*b) + 2*k*Real.pi/(3*b)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_function_properties_l169_16943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_exists_l169_16946

-- Define the function fn
noncomputable def fn (n : ℕ) (x : ℝ) : ℝ := (Real.log x) / (x^n)

-- Define Tn
noncomputable def Tn (n : ℕ) (t : ℝ) : ℝ := (t - 1) * fn n t

-- Define Sn
noncomputable def Sn (n : ℕ) (t : ℝ) : ℝ := ∫ x in Set.Icc 1 t, fn n x

-- Theorem statement
theorem unique_intersection_exists (n : ℕ) (h : n ≥ 2) :
  ∃! t : ℝ, t > 1 ∧ Tn n t = Sn n t :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_exists_l169_16946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_implies_a_range_l169_16922

theorem function_inequality_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → ∃ f : ℝ → ℝ, f x = a^x + x^2 - x * Real.log a) →
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 1 → x₂ ∈ Set.Icc 0 1 → 
    ∀ f : ℝ → ℝ, f x₁ = a^x₁ + x₁^2 - x₁ * Real.log a → 
                 f x₂ = a^x₂ + x₂^2 - x₂ * Real.log a → 
                 |f x₁ - f x₂| ≤ a - 1) →
  a ≥ Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_implies_a_range_l169_16922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_decreasing_on_open_interval_l169_16995

open Real

-- Define the interval (0, π/2)
def openInterval : Set ℝ := {x | 0 < x ∧ x < π/2}

-- Define the monotonicity property
def isDecreasing (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ S → y ∈ S → x < y → f y < f x

-- State the theorem
theorem cos_decreasing_on_open_interval :
  isDecreasing (fun x => x^(-(1/3 : ℝ))) openInterval →
  isDecreasing cos openInterval := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_decreasing_on_open_interval_l169_16995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_bed_area_increase_l169_16938

noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2

noncomputable def percentage_increase (smaller larger : ℝ) : ℝ :=
  ((larger - smaller) / smaller) * 100

theorem flower_bed_area_increase :
  let r1 : ℝ := 4
  let r2 : ℝ := 6
  let area1 := circle_area r1
  let area2 := circle_area r2
  let increase := percentage_increase area1 area2
  ∃ (n : ℕ), n = 125 ∧ ∀ m : ℕ, abs (increase - n) ≤ abs (increase - m) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_bed_area_increase_l169_16938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_and_a_l169_16960

/-- A parabola with equation y = ax² and directrix y = -1/4 has focus at (0, 1/4) and a = 1 -/
theorem parabola_focus_and_a (a : ℝ) : 
  (∀ x y : ℝ, y = a * x^2 ↔ (x, y) ∈ {p : ℝ × ℝ | p.2 = a * p.1^2}) →
  (∀ x : ℝ, -1/4 = (x, -1/4).2) →
  ((0, 1/4) = (0, 1 / (4 * a))) ∧ 
  (a = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_and_a_l169_16960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_sin_satisfies_conditions_l169_16935

open Real

-- Define the properties of the function
def is_periodic_pi (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + Real.pi) = f x

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def satisfies_conditions (f : ℝ → ℝ) : Prop :=
  (∀ x, f (x + Real.pi) = f (-x)) ∧ is_even f

-- State the theorem
theorem abs_sin_satisfies_conditions :
  satisfies_conditions (fun x ↦ |sin x|) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_sin_satisfies_conditions_l169_16935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_fractional_parts_zeta_l169_16923

-- Define the Riemann zeta function
noncomputable def riemann_zeta (x : ℝ) : ℝ := ∑' n, (1 : ℝ) / (n ^ x)

-- Define the fractional part function
noncomputable def frac (x : ℝ) : ℝ := x - ⌊x⌋

-- State the theorem
theorem sum_fractional_parts_zeta :
  (∑' k : ℕ, frac (riemann_zeta (2 * (k + 3)))) = (1 : ℝ) / 36 := by
  sorry

#check sum_fractional_parts_zeta

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_fractional_parts_zeta_l169_16923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_tan_l169_16915

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_tan (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 + a 8 + a 15 = Real.pi →
  Real.tan (a 4 + a 12) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_tan_l169_16915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_to_nearest_tenth_problem_l169_16900

/-- Rounds a real number to the nearest tenth -/
noncomputable def roundToNearestTenth (x : ℝ) : ℝ := 
  ⌊x * 10 + 0.5⌋ / 10

/-- The problem number -/
def problemNumber : ℝ := 78.46582

/-- Theorem stating that rounding the problem number to the nearest tenth equals 78.5 -/
theorem round_to_nearest_tenth_problem : roundToNearestTenth problemNumber = 78.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_to_nearest_tenth_problem_l169_16900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_sum_a_b_l169_16978

theorem least_sum_a_b : ∃ (a b : ℕ), 
  (Nat.gcd (a + b) 330 = 1) ∧ 
  (a^a % b^b = 0) ∧ 
  (¬ ∃ (k : ℕ), a = k * b) ∧
  (∀ (c d : ℕ), 
    (Nat.gcd (c + d) 330 = 1) → 
    (c^c % d^d = 0) → 
    (¬ ∃ (k : ℕ), c = k * d) → 
    (a + b ≤ c + d)) ∧
  (a + b = 105) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_sum_a_b_l169_16978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bottles_needed_l169_16979

-- Define the conversion factors and requirements
def flOzPerLiter : ℝ := 33.8
def mlPerLiter : ℝ := 1000
def bottleSize : ℝ := 250
def requiredFlOz : ℝ := 60

-- Define the function to calculate the number of bottles needed
noncomputable def bottlesNeeded : ℕ :=
  let litersNeeded := requiredFlOz / flOzPerLiter
  let mlNeeded := litersNeeded * mlPerLiter
  (Int.ceil (mlNeeded / bottleSize)).toNat

-- Theorem statement
theorem min_bottles_needed : bottlesNeeded = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bottles_needed_l169_16979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_from_intersecting_lines_l169_16966

/-- Two lines intersecting at P(2,8) with slopes 3 and -1 form a triangle with the x-axis -/
theorem triangle_area_from_intersecting_lines :
  let p : ℝ × ℝ := (2, 8)
  let m₁ : ℝ := 3
  let m₂ : ℝ := -1
  let line₁ (x : ℝ) := m₁ * (x - p.1) + p.2
  let line₂ (x : ℝ) := m₂ * (x - p.1) + p.2
  let q : ℝ × ℝ := (-2/3, 0)  -- x-intercept of line₁
  let r : ℝ × ℝ := (10, 0)    -- x-intercept of line₂
  let area := (1/2) * (r.1 - q.1) * p.2
  area = 128/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_from_intersecting_lines_l169_16966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_function_value_l169_16901

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := 1 + Real.log x / Real.log 2

-- Define the property of g being symmetric to f about y = x
def symmetric_about_y_eq_x (f g : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ g y = x

-- State the theorem
theorem symmetric_function_value (g : ℝ → ℝ) 
  (h : symmetric_about_y_eq_x f g) : g 3 = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_function_value_l169_16901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_number_probability_l169_16926

def alice_range : Finset ℕ := Finset.filter (λ n => n > 0 ∧ n < 300 ∧ 20 ∣ n) (Finset.range 300)
def alex_range : Finset ℕ := Finset.filter (λ n => n > 0 ∧ n < 300 ∧ 36 ∣ n) (Finset.range 300)

theorem same_number_probability :
  (Finset.card (alice_range ∩ alex_range) : ℚ) / 
  ((Finset.card alice_range : ℚ) * (Finset.card alex_range : ℚ)) = 1 / 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_number_probability_l169_16926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_difference_l169_16961

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem factorial_difference (x : ℕ) (h1 : x > 0) (h2 : factorial x - factorial (x - 4) = 120) : x = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_difference_l169_16961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_relationships_l169_16952

/-- Represents the units of measurement used in the problem -/
inductive MeasurementUnit
| Meter
| Yuan
| Hectare
| SquareCentimeter

/-- Represents a measurement with a value and a unit -/
structure Measurement where
  value : ℝ
  unit : MeasurementUnit

/-- Conversion factors between units -/
noncomputable def meterToHectare : ℝ := 1 / 10000
noncomputable def squareCentimeterToHectare : ℝ := 1 / 100000000

/-- The blackboard length -/
def blackboardLength : Measurement := ⟨4, MeasurementUnit.Meter⟩

/-- The pencil case price -/
def pencilCasePrice : Measurement := ⟨9.50, MeasurementUnit.Yuan⟩

/-- The school campus area -/
def schoolCampusArea : Measurement := ⟨3, MeasurementUnit.Hectare⟩

/-- The fingernail area -/
def fingernailArea : Measurement := ⟨1, MeasurementUnit.SquareCentimeter⟩

theorem unit_relationships :
  (blackboardLength.value * blackboardLength.value * meterToHectare < schoolCampusArea.value) ∧
  (fingernailArea.value * squareCentimeterToHectare < schoolCampusArea.value) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_relationships_l169_16952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_defined_iff_l169_16907

-- Define the expression as noncomputable
noncomputable def f (x : ℝ) := (Real.log (9 - x^2)) / Real.sqrt (2*x - 1)

-- Theorem statement
theorem f_defined_iff (x : ℝ) : 
  (∃ y, f x = y) ↔ (1/2 < x ∧ x < 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_defined_iff_l169_16907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_equation_chord_length_l169_16902

/-- Line l -/
def line_l (x y : ℝ) : Prop := x - y + 2 = 0

/-- Point A -/
def point_A : ℝ × ℝ := (3, -1)

/-- Circle equation -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 4*y - 12 = 0

/-- Theorem for parallel line equation -/
theorem parallel_line_equation :
  ∃ (m b : ℝ), ∀ (x y : ℝ),
    (line_l x y → y = m*x + (m + 2)) ∧
    (y = m*x + b ∧ (x, y) = point_A) →
    y = x - 4 :=
sorry

/-- Theorem for chord length -/
theorem chord_length :
  ∃ (M N : ℝ × ℝ),
    (∀ (x y : ℝ), line_l x y → (x, y) = M ∨ (x, y) = N) ∧
    (∀ (x y : ℝ), circle_eq x y → (x, y) = M ∨ (x, y) = N) ∧
    Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_equation_chord_length_l169_16902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_quadruple_characterization_l169_16941

/-- A quadruple of real numbers satisfying the product condition -/
structure ProductQuadruple where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  prod_cond : (a = b * c ∧ a = b * d ∧ a = c * d) ∧
              (b = a * c ∧ b = a * d ∧ b = c * d) ∧
              (c = a * b ∧ c = a * d ∧ c = b * d) ∧
              (d = a * b ∧ d = a * c ∧ d = b * c)

/-- The set of all valid ProductQuadruples -/
def ValidQuadruples : Set ProductQuadruple :=
  {q | q.a = 0 ∧ q.b = 0 ∧ q.c = 0 ∧ q.d = 0 ∨
       q.a = 1 ∧ q.b = 1 ∧ q.c = 1 ∧ q.d = 1 ∨
       (q.a = -1 ∧ q.b = -1 ∧ q.c = 1 ∧ q.d = 1) ∨
       (q.a = -1 ∧ q.b = -1 ∧ q.c = -1 ∧ q.d = 1) ∨
       (q.a = -1 ∧ q.b = 1 ∧ q.c = 1 ∧ q.d = -1) ∨
       (q.a = 1 ∧ q.b = -1 ∧ q.c = 1 ∧ q.d = -1) ∨
       (q.a = 1 ∧ q.b = 1 ∧ q.c = -1 ∧ q.d = -1)}

theorem product_quadruple_characterization (q : ProductQuadruple) : 
  q ∈ ValidQuadruples := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_quadruple_characterization_l169_16941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_rose_more_expensive_than_two_carnations_l169_16957

/-- The price of a rose in yuan -/
def x : ℝ := sorry

/-- The price of a carnation in yuan -/
def y : ℝ := sorry

/-- 3 roses and 2 carnations cost more than 8 yuan -/
axiom condition1 : 3 * x + 2 * y > 8

/-- 2 roses and 3 carnations cost less than 7 yuan -/
axiom condition2 : 2 * x + 3 * y < 7

/-- Theorem: One rose is more expensive than two carnations -/
theorem one_rose_more_expensive_than_two_carnations : x > 2 * y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_rose_more_expensive_than_two_carnations_l169_16957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l169_16959

noncomputable def f (ω : Real) (x : Real) : Real := 
  Real.sqrt 3 * Real.sin (ω * x) * Real.cos (ω * x) + (Real.cos (ω * x))^2 - 1/2

noncomputable def g (x : Real) : Real := Real.sin (2 * x - Real.pi / 3)

theorem function_properties (ω : Real) (h1 : ω > 0) 
  (h2 : ∀ x, f ω (x + Real.pi / (2 * ω)) = f ω x) 
  (h3 : ∀ T > 0, (∀ x, f ω (x + T) = f ω x) → T ≥ Real.pi / (2 * ω)) :
  (∀ x, f ω x = Real.sin (4 * x + Real.pi / 6)) ∧
  (∀ k, (∃! x, x ∈ Set.Icc 0 (Real.pi / 2) ∧ g x + k = 0) ↔ 
    (-Real.sqrt 3 / 2 < k ∧ k ≤ Real.sqrt 3 / 2) ∨ k = -1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l169_16959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_d_value_l169_16945

theorem least_d_value (c d : ℕ) : 
  (Nat.card (Nat.divisors c) = 4) →
  (Nat.card (Nat.divisors d) = c) →
  (d % c = 0) →
  (∀ e : ℕ, 
    (Nat.card (Nat.divisors e) = c) → 
    (e % c = 0) → 
    d ≤ e) →
  d = 18 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_d_value_l169_16945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_power_line_placement_l169_16974

/-- The distance between factories A and B -/
def AB : ℝ := 0.6

/-- The distance from substation C to both A and B -/
def AC : ℝ := 0.5

/-- The distance from the midpoint of AB to point D -/
noncomputable def x : ℝ := Real.sqrt 0.03

/-- The total length of the power line -/
noncomputable def L (x : ℝ) : ℝ := (0.4 - x) + 2 * Real.sqrt (0.09 + x^2)

theorem optimal_power_line_placement :
  ∀ y : ℝ, L x ≤ L y := by
  sorry

#eval AB
#eval AC

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_power_line_placement_l169_16974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_correct_example_bridge_length_approx_l169_16905

/-- Calculates the length of a bridge crossed by a train -/
noncomputable def bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (time_seconds : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let total_distance := train_speed_ms * time_seconds
  total_distance - train_length

/-- Theorem stating that the bridge length calculation is correct -/
theorem bridge_length_correct (train_length : ℝ) (train_speed_kmh : ℝ) (time_seconds : ℝ) :
  bridge_length train_length train_speed_kmh time_seconds =
  train_speed_kmh * 1000 / 3600 * time_seconds - train_length :=
by sorry

/-- Example calculation for the given problem -/
noncomputable def example_bridge_length : ℝ :=
  bridge_length 285 62 68

/-- Theorem stating that the example calculation is approximately correct -/
theorem example_bridge_length_approx :
  ‖example_bridge_length - 886.111‖ < 0.001 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_correct_example_bridge_length_approx_l169_16905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_pi_fourth_g_monotone_decreasing_sin_less_than_x_for_positive_x_l169_16928

noncomputable def f (x : ℝ) := Real.sin x
noncomputable def g (m : ℝ) (x : ℝ) := m * x - x^3 / 6

theorem tangent_line_at_pi_fourth :
  let slope := Real.sqrt 2 / 2
  let intercept := 1 / 2 - Real.sqrt 2 * Real.pi / 8
  ∀ x : ℝ, f (Real.pi / 4) + slope * (x - Real.pi / 4) = slope * x + intercept := by
  sorry

theorem g_monotone_decreasing (m : ℝ) :
  ∀ x : ℝ, x ≤ -Real.sqrt (2 * m) ∨ x ≥ Real.sqrt (2 * m) →
  ∀ y : ℝ, y > x → g m y < g m x := by
  sorry

theorem sin_less_than_x_for_positive_x :
  ∀ x : ℝ, x > 0 → f x < x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_pi_fourth_g_monotone_decreasing_sin_less_than_x_for_positive_x_l169_16928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_one_solution_l169_16921

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^3 + 1 else x^2 + 2

-- Theorem statement
theorem f_eq_one_solution (x : ℝ) : f x = 1 ↔ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_one_solution_l169_16921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apples_picked_l169_16967

theorem apples_picked (initial_apples : Float) (total_apples : Float) 
  (h1 : initial_apples = 56.0)
  (h2 : total_apples = 161.0) :
  total_apples - initial_apples = 105.0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apples_picked_l169_16967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_pentagon_area_l169_16931

theorem equilateral_pentagon_area :
  let side_length : ℝ := 2
  let triangle_area : ℝ := (Real.sqrt 3 / 4) * side_length^2
  let num_triangles : ℕ := 5
  triangle_area * num_triangles = 5 * Real.sqrt 3 :=
by
  -- Introduce the local definitions
  intro side_length triangle_area num_triangles
  
  -- Simplify the left-hand side
  have h1 : triangle_area * num_triangles = ((Real.sqrt 3 / 4) * 2^2) * 5 := by rfl
  
  -- Simplify further
  have h2 : ((Real.sqrt 3 / 4) * 2^2) * 5 = 5 * Real.sqrt 3 := by
    ring
    
  -- Conclude the proof
  exact h2


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_pentagon_area_l169_16931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_difference_l169_16917

theorem sin_cos_difference (A : Real) (h : Real.sin A + Real.cos A = Real.sqrt 5 / 5) :
  Real.sin A - Real.cos A = 3 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_difference_l169_16917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l169_16955

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0

/-- The focal length of a hyperbola -/
noncomputable def focal_length (h : Hyperbola) : ℝ := 2 * Real.sqrt (h.a^2 + h.b^2)

/-- Checks if a point (x, y) lies on the asymptote of a hyperbola -/
def on_asymptote (h : Hyperbola) (x y : ℝ) : Prop :=
  y = (h.b / h.a) * x ∨ y = -(h.b / h.a) * x

/-- The main theorem -/
theorem hyperbola_equation (h : Hyperbola) 
  (h_focal : focal_length h = 2 * Real.sqrt 5)
  (h_asymptote : on_asymptote h 2 1) :
  h.a = 2 ∧ h.b = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l169_16955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_BEC_is_six_l169_16968

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a trapezoid ABCD with point E on DC -/
structure Trapezoid where
  A : Point
  B : Point
  C : Point
  D : Point
  E : Point

/-- Given trapezoid satisfies the problem conditions -/
def satisfiesConditions (t : Trapezoid) : Prop :=
  -- AD is perpendicular to DC
  (t.A.x - t.D.x) * (t.D.x - t.C.x) + (t.A.y - t.D.y) * (t.D.y - t.C.y) = 0 ∧
  -- AD = AB = 3
  (t.A.x - t.D.x)^2 + (t.A.y - t.D.y)^2 = 9 ∧
  (t.A.x - t.B.x)^2 + (t.A.y - t.B.y)^2 = 9 ∧
  -- DC = 7
  (t.D.x - t.C.x)^2 + (t.D.y - t.C.y)^2 = 49 ∧
  -- E is on DC
  ∃ k : ℝ, 0 ≤ k ∧ k ≤ 1 ∧ t.E.x = t.D.x + k * (t.C.x - t.D.x) ∧ t.E.y = t.D.y + k * (t.C.y - t.D.y) ∧
  -- BE is parallel to AD
  (t.B.x - t.E.x) * (t.A.y - t.D.y) = (t.B.y - t.E.y) * (t.A.x - t.D.x) ∧
  -- BE equally divides DC
  (t.D.x - t.E.x)^2 + (t.D.y - t.E.y)^2 = (t.E.x - t.C.x)^2 + (t.E.y - t.C.y)^2 ∧
  -- DE = 3
  (t.D.x - t.E.x)^2 + (t.D.y - t.E.y)^2 = 9

/-- Calculate the area of triangle BEC -/
noncomputable def areaOfTriangleBEC (t : Trapezoid) : ℝ :=
  let BE := ((t.B.x - t.E.x)^2 + (t.B.y - t.E.y)^2).sqrt
  let EC := ((t.E.x - t.C.x)^2 + (t.E.y - t.C.y)^2).sqrt
  0.5 * BE * EC

/-- Theorem: The area of triangle BEC is 6 -/
theorem area_of_triangle_BEC_is_six (t : Trapezoid) (h : satisfiesConditions t) :
  areaOfTriangleBEC t = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_BEC_is_six_l169_16968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_equality_l169_16954

theorem angle_equality (α β x y : Real) : 
  (0 ≤ α ∧ α < 360) → 
  (0 ≤ β ∧ β < 360) → 
  (0 ≤ x ∧ x < 360) → 
  (0 ≤ y ∧ y < 360) → 
  (Real.sin x + Real.sin y = Real.sin α + Real.sin β) → 
  (Real.cos x + Real.cos y = Real.cos α + Real.cos β) → 
  ((x = α ∧ y = β) ∨ (x = β ∧ y = α)) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_equality_l169_16954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_period_approx_3_years_l169_16988

/-- Calculates the time period for compound interest -/
noncomputable def compound_interest_period (P A r : ℝ) : ℝ :=
  Real.log (A / P) / Real.log (1 + r)

/-- Theorem: The compound interest period is approximately 3 years -/
theorem compound_interest_period_approx_3_years
  (P : ℝ)
  (A : ℝ)
  (r : ℝ)
  (h_P : P = 2000)
  (h_A : A = 2662)
  (h_r : r = 0.1) :
  ∃ ε > 0, |compound_interest_period P A r - 3| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_period_approx_3_years_l169_16988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coordinates_A_l169_16976

/-- Given points A, B, and C in ℝ², where AC/AB = CB/AB = 1/3, B = (2, 15), and C = (-4, 5),
    prove that the sum of the coordinates of A is 22 + 1/3. -/
theorem sum_of_coordinates_A (A B C : ℝ × ℝ) : 
  (dist A C) / (dist A B) = 1/3 →
  (dist C B) / (dist A B) = 1/3 →
  B = (2, 15) →
  C = (-4, 5) →
  (A.1 + A.2 = 22 + 1/3) :=
by sorry

-- Define the distance function
noncomputable def dist (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coordinates_A_l169_16976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_N_mod_1000_l169_16970

def N : ℕ := 3^2011

theorem N_mod_1000 : N % 1000 = 183 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_N_mod_1000_l169_16970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_length_is_1000_l169_16914

/-- The length of a race, given the positions of two runners at the finish of one runner. -/
def race_length (distance_from_start : ℕ) (distance_between_runners : ℕ) : ℕ :=
  distance_from_start + distance_between_runners

/-- Proof that the race length is 1000 meters under the given conditions. -/
theorem race_length_is_1000 : race_length 184 816 = 1000 := by
  rfl

#eval race_length 184 816

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_length_is_1000_l169_16914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_equality_l169_16919

theorem sequence_equality (x : Fin 1995 → ℝ) 
  (h1 : ∀ n : Fin 1994, 2 * Real.sqrt (x n - n.val + 1) ≥ x (n.succ) - n.val + 1)
  (h2 : 2 * Real.sqrt (x ⟨1994, by norm_num⟩ - 1994) ≥ x ⟨0, by norm_num⟩ + 1) :
  ∀ n : Fin 1995, x n = n.val := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_equality_l169_16919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_close_points_l169_16910

/-- An equilateral triangle with side length 1 -/
structure EquilateralTriangle where
  side_length : ℝ
  is_equilateral : side_length = 1

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- A set of 5 points -/
def FivePoints := Fin 5 → Point

theorem exists_close_points (triangle : EquilateralTriangle) (points : FivePoints) :
  ∃ (i j : Fin 5), i ≠ j ∧ distance (points i) (points j) ≤ 0.5 := by
  sorry

#check exists_close_points

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_close_points_l169_16910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_not_necessarily_axisymmetric_l169_16951

-- Define what it means for a figure to be axisymmetric
def is_axisymmetric (figure : Type) : Prop :=
  ∃ (axis : Set ℝ), ∀ (point : figure), 
    ∃ (reflected_point : figure), 
      point = reflected_point  -- Simplified condition for symmetry

-- Define a right-angled triangle
structure RightTriangle :=
  (a b c : ℝ)
  (right_angle : a^2 + b^2 = c^2)

-- State the theorem
theorem right_triangle_not_necessarily_axisymmetric :
  ¬ ∀ (t : RightTriangle), is_axisymmetric RightTriangle :=
by
  -- We use a proof by contradiction
  intro h
  -- Consider a non-isosceles right triangle
  let t : RightTriangle := ⟨3, 4, 5, by norm_num⟩
  -- Apply the hypothesis to this triangle
  have : is_axisymmetric RightTriangle := h t
  -- This leads to a contradiction, as a 3-4-5 triangle is not axisymmetric
  sorry  -- We skip the detailed proof of the contradiction

-- Note: The actual proof would involve showing that no axis of symmetry exists for this triangle,
-- which is beyond the scope of this basic framework.

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_not_necessarily_axisymmetric_l169_16951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_is_150_degrees_l169_16991

/-- The inclination angle of a line with equation ax + by + c = 0 -/
noncomputable def inclination_angle (a b : ℝ) : ℝ :=
  Real.arctan (-a / b)

theorem line_inclination_angle_is_150_degrees :
  let a : ℝ := Real.sqrt 3
  let b : ℝ := 3
  0 < inclination_angle a b ∧ inclination_angle a b < π →
  inclination_angle a b = 5 * π / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_is_150_degrees_l169_16991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_64_l169_16939

theorem cube_root_of_64 : (64 : ℝ) ^ (1/3) = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_64_l169_16939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l169_16984

noncomputable def f (x : ℝ) : ℝ := Real.sin ((Real.pi / 3) * x + 1 / 3)

theorem smallest_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l169_16984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_molecular_weight_l169_16996

/-- Calculates the molecular weight of a compound given the number of atoms and their atomic weights -/
def molecular_weight (carbon_atoms hydrogen_atoms oxygen_atoms : ℕ) 
  (carbon_weight hydrogen_weight oxygen_weight : ℝ) : ℝ :=
  (carbon_atoms : ℝ) * carbon_weight + 
  (hydrogen_atoms : ℝ) * hydrogen_weight + 
  (oxygen_atoms : ℝ) * oxygen_weight

/-- Theorem stating the molecular weight of the given compound -/
theorem compound_molecular_weight :
  let carbon_atoms : ℕ := 7
  let hydrogen_atoms : ℕ := 6
  let oxygen_atoms : ℕ := 2
  let carbon_weight : ℝ := 12.01
  let hydrogen_weight : ℝ := 1.008
  let oxygen_weight : ℝ := 16.00
  ∃ ε > 0, |molecular_weight carbon_atoms hydrogen_atoms oxygen_atoms 
    carbon_weight hydrogen_weight oxygen_weight - 122.118| < ε := by
  sorry

#check compound_molecular_weight

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_molecular_weight_l169_16996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_dot_product_and_range_l169_16962

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x + Real.cos x, -2 * Real.sin x)

noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sin x - Real.cos x, Real.sqrt 3 * Real.cos x)

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem vector_dot_product_and_range :
  (∃! x : ℝ, x ∈ Set.Ioo 0 π ∧ f x = 2 ∧ x = 2 * π / 3) ∧
  (∀ x : ℝ, x ∈ Set.Ioo 0 (π / 2) → -2 < f x ∧ f x < -1) ∧
  (∀ ε > 0, ∃ x₁ x₂ : ℝ, x₁ ∈ Set.Ioo 0 (π / 2) ∧ x₂ ∈ Set.Ioo 0 (π / 2) ∧ f x₁ < -2 + ε ∧ f x₂ > -1 - ε) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_dot_product_and_range_l169_16962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_capital_city_survey_not_representative_l169_16953

/-- Represents a survey method -/
structure SurveyMethod where
  sample : Set String
  population : Set String

/-- Defines what it means for a survey method to be representative -/
def isRepresentative (method : SurveyMethod) : Prop :=
  method.sample ⊆ method.population ∧ 
  method.sample.Nonempty ∧
  ∀ x ∈ method.population, ∃ y ∈ method.sample, true  -- Placeholder for characteristics comparison

/-- Theorem: A survey method that uses only the capital city to represent an entire province is not representative -/
theorem capital_city_survey_not_representative (province : Set String) (capital : Set String) :
  capital ⊂ province →
  ¬isRepresentative { sample := capital, population := province } := by
  intro h
  simp [isRepresentative]
  sorry  -- Proof omitted

#check capital_city_survey_not_representative

end NUMINAMATH_CALUDE_ERRORFEEDBACK_capital_city_survey_not_representative_l169_16953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cube_volume_main_theorem_l169_16998

/-- Regular quadrilateral pyramid with an inscribed cube -/
structure PyramidWithCube where
  a : ℝ  -- Side length of the pyramid's base and lateral edge
  h : a > 0  -- Ensure positive side length

/-- Volume of the inscribed cube -/
noncomputable def cubeVolume (p : PyramidWithCube) : ℝ :=
  (p.a * (Real.sqrt 2 - 1) / 3) ^ 3

/-- Theorem stating the volume of the inscribed cube -/
theorem inscribed_cube_volume (p : PyramidWithCube) :
  cubeVolume p = (p.a * (Real.sqrt 2 - 1) / 3) ^ 3 := by
  rfl

/-- Main theorem proving the volume of the inscribed cube -/
theorem main_theorem (p : PyramidWithCube) :
  ∃ (v : ℝ), v = cubeVolume p ∧ v = (p.a * (Real.sqrt 2 - 1) / 3) ^ 3 := by
  use cubeVolume p
  constructor
  · rfl
  · rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cube_volume_main_theorem_l169_16998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l169_16929

/-- Represents a line in 2D space --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  nonzero : a ≠ 0 ∨ b ≠ 0

/-- Checks if a point lies on a line --/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Calculates the x-intercept of a line --/
noncomputable def Line.xIntercept (l : Line) : ℝ :=
  -l.c / l.a

/-- Calculates the y-intercept of a line --/
noncomputable def Line.yIntercept (l : Line) : ℝ :=
  -l.c / l.b

theorem line_equation_proof (l : Line) : 
  (l.a = 1 ∧ l.b = 2 ∧ l.c = -17) →
  l.contains 5 6 ∧ 
  l.xIntercept = 2 * l.yIntercept :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l169_16929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_bricks_l169_16920

/-- Represents the time taken to build the wall -/
structure WallTime where
  becky : ℚ
  bob : ℚ
  together : ℚ

/-- Represents the efficiency decrease when working together -/
def efficiency_decrease : ℚ := 15

/-- Calculates the number of bricks in the wall -/
noncomputable def bricks_in_wall (wt : WallTime) : ℚ :=
  let becky_rate := 1 / wt.becky
  let bob_rate := 1 / wt.bob
  let combined_rate := becky_rate + bob_rate - efficiency_decrease
  combined_rate * wt.together

/-- The main theorem stating the number of bricks in the wall -/
theorem wall_bricks (wt : WallTime) 
  (h_becky : wt.becky = 8)
  (h_bob : wt.bob = 12)
  (h_together : wt.together = 6) :
  bricks_in_wall wt = 360 := by
  sorry

#check wall_bricks

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_bricks_l169_16920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_winning_strategy_l169_16906

/-- Represents a player in the Y2K Game -/
inductive Player : Type
| First : Player
| Second : Player

/-- Represents a cell in the Y2K Game grid -/
inductive Cell : Type
| Empty : Cell
| S : Cell
| O : Cell

/-- Represents the Y2K Game board -/
def Board : Type := Fin 2000 → Cell

/-- Checks if a player has won by forming SOS in three consecutive cells -/
def hasWon (board : Board) : Prop :=
  ∃ i : Fin 1998, board i = Cell.S ∧ board (i + 1) = Cell.O ∧ board (i + 2) = Cell.S

/-- Represents a strategy for a player in the Y2K Game -/
def Strategy : Type := Board → Fin 2000

/-- Updates the board with a player's move -/
def updateBoard (board : Board) (player : Player) (move : Fin 2000) : Board :=
  fun i => if i = move then 
    match player with
    | Player.First => Cell.S
    | Player.Second => Cell.O
  else board i

/-- Checks if a strategy is winning for the second player -/
def isWinningStrategy (strat : Strategy) : Prop :=
  ∀ (game : Nat → Board), 
    (∀ n : Nat, game (n + 1) = updateBoard (game n) (if n % 2 = 0 then Player.First else Player.Second) (strat (game n))) →
    (∃ n : Nat, hasWon (game n) ∧ n % 2 = 1) ∨ 
    (∀ i : Fin 2000, game 2000 i ≠ Cell.Empty)

/-- The main theorem stating that the second player has a winning strategy -/
theorem second_player_winning_strategy : 
  ∃ (strat : Strategy), isWinningStrategy strat := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_winning_strategy_l169_16906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l169_16985

/-- The function f(x) defined in the problem -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.sqrt (|x + 1| + |x - 3| - m)

/-- Theorem stating the maximum value of m and the minimum value of 7a+4b -/
theorem problem_solution :
  (∀ x : ℝ, f 4 x ≥ 0) ∧
  (∀ m : ℝ, (∀ x : ℝ, f m x ≥ 0) → m ≤ 4) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → 2 / (3*a + b) + 1 / (a + 2*b) = 4 →
    7*a + 4*b ≥ 9/4 ∧ (∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧
      2 / (3*a₀ + b₀) + 1 / (a₀ + 2*b₀) = 4 ∧ 7*a₀ + 4*b₀ = 9/4)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l169_16985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_eq_two_l169_16981

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = (x * e^x) / (e^(ax) - 1) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (x * Real.exp x) / (Real.exp (a * x) - 1)

/-- If f(x) = (x * e^x) / (e^(ax) - 1) is an even function, then a = 2 -/
theorem even_function_implies_a_eq_two :
  ∀ a : ℝ, IsEven (f a) → a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_eq_two_l169_16981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_visit_charge_correct_l169_16908

/-- Represents the charge for the first visit at a tanning salon --/
def first_visit_charge : ℚ := 9.20

/-- Represents the charge for subsequent visits at a tanning salon --/
def subsequent_visit_charge : ℚ := 8

/-- Represents the total number of customers who visited the salon --/
def total_customers : ℕ := 100

/-- Represents the number of customers who made a second visit --/
def second_visit_customers : ℕ := 30

/-- Represents the number of customers who made a third visit --/
def third_visit_customers : ℕ := 10

/-- Represents the total revenue for the last calendar month --/
def total_revenue : ℚ := 1240

/-- Theorem stating that the charge for the first visit is correct given the conditions --/
theorem first_visit_charge_correct : 
  first_visit_charge * total_customers + 
  subsequent_visit_charge * (second_visit_customers + third_visit_customers) = 
  total_revenue := by
  sorry

#eval first_visit_charge * total_customers + 
  subsequent_visit_charge * (second_visit_customers + third_visit_customers)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_visit_charge_correct_l169_16908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_points_coordinates_l169_16990

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (log x)^2 + 2 * log x

-- Define the derivative of f
noncomputable def f_deriv (x : ℝ) : ℝ := 2 * (log x + 1) / x

-- Theorem statement
theorem tangent_points_coordinates (a b : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : a < b)
  (h4 : f a = a * f_deriv a) (h5 : f b = b * f_deriv b) :
  a = Real.exp (-Real.sqrt 2) ∧ b = Real.exp (Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_points_coordinates_l169_16990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_ratio_years_l169_16969

/-- Tom's current age -/
def t : ℕ := sorry

/-- Jerry's current age -/
def j : ℕ := sorry

/-- Tom's age was three times Jerry's age three years ago -/
axiom condition1 : t - 3 = 3 * (j - 3)

/-- Tom's age was five times Jerry's age seven years ago -/
axiom condition2 : t - 7 = 5 * (j - 7)

/-- The number of years until the ratio of Tom's age to Jerry's age is 3:2 -/
def x : ℕ := sorry

/-- The ratio of Tom's age to Jerry's age after x years is 3:2 -/
axiom future_ratio : (t + x) * 2 = (j + x) * 3

theorem age_ratio_years : x = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_ratio_years_l169_16969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_niatirp_sum_equivalence_l169_16956

-- Define the Niatirp numeral system
def niatirp_to_arabic (n : ℕ) : ℕ :=
  let digits := n.digits 10
  let converted := digits.map (λ d ↦ 9 - d)
  converted.foldl (λ acc d ↦ acc * 10 + d) 0

-- Define the inverse conversion
def arabic_to_niatirp (n : ℕ) : ℕ :=
  let digits := n.digits 10
  let converted := digits.map (λ d ↦ 9 - d)
  converted.foldl (λ acc d ↦ acc * 10 + d) 0

-- Theorem statement
theorem niatirp_sum_equivalence :
  arabic_to_niatirp (niatirp_to_arabic 837 + niatirp_to_arabic 742) = 419 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_niatirp_sum_equivalence_l169_16956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_theta_plus_pi_third_l169_16911

theorem sin_theta_plus_pi_third (θ : Real) 
  (h1 : Real.cos θ = -3/5) 
  (h2 : θ ∈ Set.Ioo (Real.pi/2) Real.pi) : 
  Real.sin (θ + Real.pi/3) = (4 - 3 * Real.sqrt 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_theta_plus_pi_third_l169_16911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_coords_of_neg_five_minus_five_sqrt_three_l169_16927

/-- The polar coordinates of a point in the complex plane. -/
structure PolarCoord where
  r : ℝ
  θ : ℝ

/-- Convert a complex number to polar coordinates. -/
noncomputable def toPolar (z : ℂ) : PolarCoord :=
  { r := Complex.abs z
    θ := Complex.arg z }

/-- The theorem stating that the polar coordinates of -5-5√3 are (10, 4π/3). -/
theorem polar_coords_of_neg_five_minus_five_sqrt_three :
  toPolar (-5 - 5 * Real.sqrt 3 : ℂ) = PolarCoord.mk 10 ((4 : ℝ) * Real.pi / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_coords_of_neg_five_minus_five_sqrt_three_l169_16927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_increased_by_35_percent_l169_16932

theorem number_increased_by_35_percent (x : ℝ) : x * 1.35 = 935 ↔ Int.floor (x + 0.5) = 693 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_increased_by_35_percent_l169_16932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_AB_line_and_length_l169_16909

-- Define the perpendicular lines
def line1 (x y : ℝ) : Prop := 2*x - y = 0
def line2 (x y : ℝ) (a : ℝ) : Prop := x + a*y = 0

-- Define the points A and B
noncomputable def A : ℝ × ℝ := (4, 8)
noncomputable def B : ℝ × ℝ := (-4, 2)

-- Define the midpoint P
noncomputable def P (a : ℝ) : ℝ × ℝ := (0, 10/a)

-- State the theorem
theorem AB_line_and_length (a : ℝ) 
  (h1 : line1 A.1 A.2)
  (h2 : line2 B.1 B.2 a)
  (h3 : (A.1 + B.1) / 2 = (P a).1 ∧ (A.2 + B.2) / 2 = (P a).2)
  (h4 : (∀ x y, line1 x y → line2 x y a → x = 0 ∧ y = 0) → a = 2) :
  (∃ (k : ℝ), ∀ (x y : ℝ), 3*x - 4*y + 20 = k * (x - A.1) ∧ y - A.2 = (B.2 - A.2)/(B.1 - A.1) * (x - A.1)) ∧
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 10 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_AB_line_and_length_l169_16909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l169_16958

noncomputable def f (x : ℝ) := Real.sqrt (x + 2) / (2 * x - 1)

theorem domain_of_f :
  {x : ℝ | x ≥ -2 ∧ x ≠ 1/2} = {x : ℝ | ∃ y, f x = y} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l169_16958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_calculation_l169_16997

/-- Represents the monthly salaries of three employees -/
structure EmployeeSalaries where
  M : ℝ
  N : ℝ
  P : ℝ

/-- Checks if the given salaries satisfy the problem conditions -/
def satisfiesConditions (s : EmployeeSalaries) : Prop :=
  s.M + s.N + s.P = 3200 ∧
  s.M = 1.2 * s.N ∧
  s.P = 0.65 * s.M ∧
  (s.N ≤ s.M ∧ s.P ≤ s.N)

/-- Theorem stating that the calculated salaries satisfy the problem conditions -/
theorem salary_calculation :
  ∃ (s : EmployeeSalaries),
    satisfiesConditions s ∧
    (|s.M - 1288.58| < 0.01) ∧
    (|s.N - 1073.82| < 0.01) ∧
    (|s.P - 837.38| < 0.01) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_calculation_l169_16997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_of_sum_le_common_difference_l169_16983

/-- A monic quadratic polynomial with real coefficients -/
structure MonicQuadratic where
  a : ℝ
  b : ℝ

/-- The difference between the roots of a monic quadratic polynomial -/
noncomputable def root_difference (q : MonicQuadratic) : ℝ :=
  Real.sqrt (q.a^2 - 4 * q.b)

/-- The sum of two monic quadratic polynomials -/
def sum_quadratics (q1 q2 : MonicQuadratic) : MonicQuadratic :=
  { a := q1.a + q2.a, b := q1.b + q2.b }

theorem root_difference_of_sum_le_common_difference 
  (f g : MonicQuadratic) 
  (hf : root_difference f > 0)
  (hg : root_difference g > 0)
  (hfg : root_difference (sum_quadratics f g) > 0)
  (h_equal : root_difference f = root_difference g) :
  root_difference (sum_quadratics f g) ≤ root_difference f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_of_sum_le_common_difference_l169_16983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_division_bound_l169_16912

/-- Represents a piece of chocolate -/
structure ChocolatePiece where
  mass : ℚ
  deriving Repr

/-- Represents the state of the chocolate division process -/
structure ChocolateState where
  pieces : List ChocolatePiece
  steps : ℕ
  deriving Repr

/-- A function that represents a single step of division -/
def divideStep (state : ChocolateState) : ChocolateState :=
  sorry

/-- The main theorem to be proved -/
theorem chocolate_division_bound 
  (initial_mass : ℚ)
  (k : ℕ)
  (h_initial_mass : initial_mass > 0)
  (h_k : k > 0)
  : 
  ∀ (final_state : ChocolateState),
    final_state.steps = k →
    (∀ piece ∈ final_state.pieces, piece.mass < (2 / (k + 1)) * initial_mass) :=
by
  sorry

#check chocolate_division_bound

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_division_bound_l169_16912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wedge_volume_l169_16980

/-- The volume of a wedge cut from a cylindrical log --/
theorem wedge_volume (d : ℝ) (α : ℝ) : 
  d = 16 → α = 60 → (π * d^2 * d / 8) * (1 - Real.cos (α * π / 180)) / 2 = 128 * π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wedge_volume_l169_16980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_hexagon_area_final_result_l169_16942

/-- The side length of the rectangle parallel to the x-axis -/
noncomputable def rectangle_width : ℝ := 20

/-- The side length of the rectangle parallel to the y-axis -/
noncomputable def rectangle_height : ℝ := 22

/-- The area of a regular hexagon with side length s -/
noncomputable def hexagon_area (s : ℝ) : ℝ := (3 * Real.sqrt 3 / 2) * s^2

/-- The largest regular hexagon that can fit inside the rectangle -/
noncomputable def largest_inscribed_hexagon : ℝ := 
  let s := Real.sqrt (884 - 440 * Real.sqrt 3)
  hexagon_area s

theorem largest_inscribed_hexagon_area : 
  largest_inscribed_hexagon = 1326 * Real.sqrt 3 - 1980 := by sorry

/-- The final result of the problem -/
theorem final_result : 
  (100 * 1326 + 10 * 3 + 1980 : ℕ) = 134610 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_hexagon_area_final_result_l169_16942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ben_walking_time_l169_16933

/-- Given Ben's walking speed, prove the formula for time taken to travel any distance -/
theorem ben_walking_time (D : ℝ) : 
  (D / 1.5) * 60 = (D / 1.5) * 60 := by
  -- Define the given values
  let initial_distance : ℝ := 3
  let initial_time : ℝ := 2
  let second_distance : ℝ := 12
  let second_time : ℝ := 480 / 60
  let speed : ℝ := initial_distance / initial_time

  -- Prove that the speed is consistent
  have speed_consistent : speed = second_distance / second_time := by
    sorry

  -- Prove the main theorem
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ben_walking_time_l169_16933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_one_l169_16903

-- Define the function f(x) = xe^x
noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

-- State the theorem
theorem tangent_slope_at_one :
  (deriv f) 1 = 2 * Real.exp 1 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_one_l169_16903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_sum_equals_54_l169_16924

theorem power_sum_equals_54 (x y : ℝ) (h1 : (3 : ℝ)^x = 6) (h2 : (3 : ℝ)^y = 9) : (3 : ℝ)^(x+y) = 54 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_sum_equals_54_l169_16924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_equals_mean_l169_16992

def number_set : List ℝ := [75, 77, 79, 81, 83, 85, 87]

theorem median_equals_mean
  (x : ℝ)
  (h_mean : (List.sum number_set + x) / 8 = 82)
  (h_sorted : List.Sorted (· ≤ ·) (number_set ++ [x]))
  : (List.get! (number_set ++ [x]) 3 +
     List.get! (number_set ++ [x]) 4) / 2 = 82 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_equals_mean_l169_16992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ladybug_count_l169_16965

/-- Represents the types of ladybugs -/
inductive LadybugType
  | TruthTeller
  | Liar

/-- The meadow of ladybugs -/
structure Meadow where
  ladybugs : List LadybugType

def spots (t : LadybugType) : Nat :=
  match t with
  | LadybugType.TruthTeller => 6
  | LadybugType.Liar => 4

def totalSpots (m : Meadow) : Nat :=
  m.ladybugs.foldl (fun acc t => acc + spots t) 0

theorem ladybug_count (m : Meadow) : 
  (∀ l ∈ m.ladybugs, l = LadybugType.TruthTeller ∨ l = LadybugType.Liar) →
  m.ladybugs.length ≥ 3 →
  m.ladybugs.get? 0 = some LadybugType.Liar →
  m.ladybugs.get? 1 = some LadybugType.Liar →
  m.ladybugs.get? 2 = some LadybugType.TruthTeller →
  (∀ i, 3 ≤ i → i < m.ladybugs.length → m.ladybugs.get? i = some LadybugType.TruthTeller) →
  totalSpots m = 26 →
  m.ladybugs.length = 5 := by
  sorry

#check ladybug_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ladybug_count_l169_16965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_max_y_coordinate_l169_16947

/-- Given a hyperbola with equation x²/4 - y²/b = 1 where b > 0,
    a point P in the first quadrant satisfying |OP| = 1/2|F₁F₂|,
    and eccentricity e ∈ (1, 2], prove that the maximum y-coordinate of P is 3. -/
theorem hyperbola_max_y_coordinate (b : ℝ) (e : ℝ) (x y : ℝ) :
  b > 0 →
  e ∈ Set.Ioo 1 2 →
  x > 0 →
  y > 0 →
  x^2 / 4 - y^2 / b = 1 →
  x^2 + y^2 = 4 + b →
  ∀ (x' y' : ℝ), x' > 0 → y' > 0 → x'^2 / 4 - y'^2 / b = 1 → x'^2 + y'^2 = 4 + b → y ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_max_y_coordinate_l169_16947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_non_multiple_subset_l169_16971

def oddSet : Set ℕ := {n : ℕ | n ≤ 199 ∧ n % 2 = 1}

def notMultiple (s : Set ℕ) : Prop :=
  ∀ a b, a ∈ s → b ∈ s → a ≠ b → a ∣ b → False

theorem max_non_multiple_subset :
  ∃ (s : Finset ℕ), ↑s ⊆ oddSet ∧ s.card = 67 ∧ notMultiple ↑s ∧
    ∀ (t : Finset ℕ), ↑t ⊆ oddSet → notMultiple ↑t → t.card ≤ 67 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_non_multiple_subset_l169_16971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_iff_a_ge_four_fifths_l169_16999

/-- The function f(x) = (1/2)x^2 - 2ax - a*ln(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - 2*a*x - a * Real.log x

theorem f_decreasing_iff_a_ge_four_fifths :
  (∀ x₁ x₂ : ℝ, 1 < x₁ → x₁ < 2 → 1 < x₂ → x₂ < 2 → x₁ ≠ x₂ → 
    (f a x₂ - f a x₁) / (x₂ - x₁) < 0) ↔ 
  a ≥ 4/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_iff_a_ge_four_fifths_l169_16999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l169_16982

-- Define the constants
noncomputable def a : ℝ := Real.log 3 / Real.log 7
noncomputable def b : ℝ := -(Real.log 7 / Real.log 3)
noncomputable def c : ℝ := 3^(7/10)

-- State the theorem
theorem relationship_abc : b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l169_16982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_four_digit_solution_l169_16989

theorem least_four_digit_solution (x : ℤ) : x = 1329 ↔ 
  (x ≥ 1000 ∧ x < 10000) ∧ 
  (∀ y : ℤ, y ≥ 1000 ∧ y < 10000 →
    (y % 3 = 1 ∧
     (2 * y + 5) % 8 = 11 ∧
     (10 * y + 2) % 13 = (2 * y) % 13 ∧
     (5 * y - 3) % 7 = 12) → 
    x ≤ y) ∧
  x % 3 = 1 ∧
  (2 * x + 5) % 8 = 11 ∧
  (10 * x + 2) % 13 = (2 * x) % 13 ∧
  (5 * x - 3) % 7 = 12 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_four_digit_solution_l169_16989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_solution_l169_16937

/-- Three points in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Check if three points are collinear -/
def collinear (p q r : Point3D) : Prop :=
  ∃ t s : ℝ, q.x - p.x = t * (r.x - p.x) ∧
             q.y - p.y = t * (r.y - p.y) ∧
             q.z - p.z = t * (r.z - p.z) ∧
             q.x - p.x = s * (r.x - q.x) ∧
             q.y - p.y = s * (r.y - q.y) ∧
             q.z - p.z = s * (r.z - q.z)

theorem collinear_points_solution (a b : ℝ) :
  let p := Point3D.mk 2 (a + 1) (b + 1)
  let q := Point3D.mk (a + 1) 3 (b + 1)
  let r := Point3D.mk (a + 1) (b + 1) 4
  collinear p q r → a = 1 ∧ b = 3 ∧ a + b = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_solution_l169_16937
