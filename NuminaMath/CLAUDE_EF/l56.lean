import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l56_5617

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x * Real.exp x + 1

-- Define the point of tangency
def x₀ : ℝ := 0
def y₀ : ℝ := 1

-- State the theorem
theorem tangent_line_equation :
  ∃ (m : ℝ), ∀ (x y : ℝ), y = m * (x - x₀) + y₀ → x - y + 1 = 0 :=
by
  -- Introduce the slope m
  use 1
  -- For all x and y satisfying the point-slope form
  intros x y h
  -- Substitute the known values
  rw [x₀, y₀] at h
  -- Simplify the equation
  simp at h
  -- Rearrange to match the desired form
  linarith [h]

-- The proof is complete, so we don't need 'sorry'

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l56_5617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l56_5688

noncomputable section

def IsRectangle (A B C D : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let BC := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
  let CD := Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2)
  let DA := Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2)
  let AC := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  AB = CD ∧ BC = DA ∧ AB * BC = AC * DA

def area (A B C D : ℝ × ℝ) : ℝ :=
  let AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let BC := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
  AB * BC

theorem rectangle_area (A B C D : ℝ × ℝ) : 
  let AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let AC := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  IsRectangle A B C D ∧ AB = 15 ∧ AC = 17 →
  area A B C D = 120 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l56_5688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_susan_loan_days_l56_5635

/-- The minimum number of days for a loan to triple with simple interest --/
def min_days_to_triple_loan (loan : ℚ) (daily_rate : ℚ) : ℕ :=
  let triple_amount := 3 * loan
  let daily_interest := loan * daily_rate
  Nat.ceil ((triple_amount - loan) / daily_interest)

/-- Proof of the specific problem --/
theorem susan_loan_days : min_days_to_triple_loan 20 (1/10) = 20 := by
  rw [min_days_to_triple_loan]
  simp
  norm_num
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_susan_loan_days_l56_5635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_satisfies_conditions_l56_5682

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

-- Define the asymptotes of the hyperbola
def asymptote (x y : ℝ) : Prop := y = (3/4) * x ∨ y = -(3/4) * x

-- Define the circle
def circle_equation (x y : ℝ) : Prop := (x - 5)^2 + y^2 = 9

-- Theorem statement
theorem circle_satisfies_conditions :
  -- The center is on the positive x-axis
  ∃ m : ℝ, m > 0 ∧
  -- The radius is equal to the length of the imaginary semi-axis of the hyperbola
  (∀ x y : ℝ, hyperbola x y → 3 = Real.sqrt 9) ∧
  -- The circle is tangent to the hyperbola's asymptotes
  (∀ x y : ℝ, asymptote x y → ∃ t : ℝ, circle_equation t ((3/4) * t) ∨ circle_equation t (-(3/4) * t)) ∧
  -- The equation (x - 5)^2 + y^2 = 9 represents the circle
  (∀ x y : ℝ, circle_equation x y ↔ (x - 5)^2 + y^2 = 9) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_satisfies_conditions_l56_5682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_teresa_total_spend_l56_5657

noncomputable def sandwich_price : ℚ := 775/100
def num_sandwiches : ℕ := 2
noncomputable def salami_price : ℚ := 4
noncomputable def olive_price_per_pound : ℚ := 10
noncomputable def olive_amount : ℚ := 1/4
noncomputable def feta_price_per_pound : ℚ := 8
noncomputable def feta_amount : ℚ := 1/2
noncomputable def french_bread_price : ℚ := 2

noncomputable def total_cost : ℚ :=
  sandwich_price * num_sandwiches +
  salami_price +
  3 * salami_price +
  olive_price_per_pound * olive_amount +
  feta_price_per_pound * feta_amount +
  french_bread_price

theorem teresa_total_spend : total_cost = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_teresa_total_spend_l56_5657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_c_sum_of_cubes_l56_5687

theorem smallest_c_sum_of_cubes : 
  ∃ (c : ℕ), c > 0 ∧ 
  (∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ c = a * b ∧
    (∃ (x y : ℤ), (a : ℤ) = x^3 + y^3) ∧
    (∃ (x y : ℤ), (b : ℤ) = x^3 + y^3)) ∧
  (∀ (x y : ℤ), (c : ℤ) ≠ x^3 + y^3) ∧
  (∀ (c' : ℕ), 0 < c' ∧ c' < c →
    ¬(∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ c' = a * b ∧
      (∃ (x y : ℤ), (a : ℤ) = x^3 + y^3) ∧
      (∃ (x y : ℤ), (b : ℤ) = x^3 + y^3) ∧
      (∀ (x y : ℤ), (c' : ℤ) ≠ x^3 + y^3))) :=
by
  use 4
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_c_sum_of_cubes_l56_5687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_fixed_point_f_fixed_point_1093_largest_fixed_point_1093_l56_5680

def f : ℕ → ℕ
  | 0 => 1  -- Adding case for 0
  | 1 => 1
  | 2 => 1
  | (n + 3) => match n % 3 with
    | 0 => 3 * f (n / 3) - 2
    | 1 => 3 * f ((n - 1) / 3) + 1
    | _ => 3 * f ((n - 2) / 3) + 4

theorem largest_fixed_point_f (n : ℕ) :
  n ≤ 1992 → f n = n → n ≤ 1093 :=
by sorry

theorem fixed_point_1093 : f 1093 = 1093 :=
by sorry

theorem largest_fixed_point_1093 :
  ∀ n : ℕ, n ≤ 1992 → f n = n → n ≤ 1093 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_fixed_point_f_fixed_point_1093_largest_fixed_point_1093_l56_5680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_and_width_of_g_l56_5668

-- Define the function h with domain [-12, 6]
def h : Set ℝ := Set.Icc (-12) 6

-- Define the function g in terms of h
def g (x : ℝ) : Prop := (2 * x / 3) ∈ h

-- Theorem statement
theorem domain_and_width_of_g :
  (∀ x, g x ↔ x ∈ Set.Icc (-18) 9) ∧
  (9 - (-18) = 27) := by
  sorry

#check domain_and_width_of_g

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_and_width_of_g_l56_5668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_coefficient_properties_l56_5696

-- Define the correlation coefficient
noncomputable def correlation_coefficient (X Y : Type*) [NormedAddCommGroup X] [InnerProductSpace ℝ X] 
  [NormedAddCommGroup Y] [InnerProductSpace ℝ Y] : ℝ := 
  sorry

-- Define the degree of correlation
noncomputable def degree_of_correlation (r : ℝ) : ℝ := 
  sorry

-- State the properties of the correlation coefficient
theorem correlation_coefficient_properties 
  (X Y : Type*) [NormedAddCommGroup X] [InnerProductSpace ℝ X] 
  [NormedAddCommGroup Y] [InnerProductSpace ℝ Y] :
  let r := correlation_coefficient X Y
  (∀ x : X, ∀ y : Y, |r| ≤ 1) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ r₁ r₂ : ℝ, 
    |r₁| > 1 - δ ∧ |r₂| > 1 - δ → 
    |degree_of_correlation r₁ - degree_of_correlation r₂| < ε) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ r₁ r₂ : ℝ,
    |r₁| < δ ∧ |r₂| < δ →
    |degree_of_correlation r₁ - degree_of_correlation r₂| < ε) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_coefficient_properties_l56_5696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lightning_distance_l56_5648

/-- The speed of sound in feet per second -/
def speed_of_sound : ℚ := 1120

/-- The number of feet in a mile -/
def feet_per_mile : ℚ := 5280

/-- The time in seconds between seeing lightning and hearing thunder -/
def time_delay : ℚ := 15

/-- Rounds a rational number to the nearest quarter -/
def round_to_quarter (x : ℚ) : ℚ :=
  ⌊(x * 4 + 1/2)⌋ / 4

/-- The theorem stating that the distance to the lightning strike is 3.25 miles -/
theorem lightning_distance :
  round_to_quarter ((speed_of_sound * time_delay) / feet_per_mile) = 13/4 := by
  sorry

#eval round_to_quarter ((speed_of_sound * time_delay) / feet_per_mile)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lightning_distance_l56_5648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_of_BC_line_bisected_by_P_l56_5630

-- Define points B, C, and P
def B : ℝ × ℝ := (-4, 0)
def C : ℝ × ℝ := (-2, 4)
def P : ℝ × ℝ := (-1, -2)

-- Define the equations of the lines
def perpendicular_bisector (x y : ℝ) : Prop := x + 2 * y - 1 = 0
def line_through_P (x y : ℝ) : Prop := 2 * x + y + 4 = 0

-- Helper functions
def lies_on (point : ℝ × ℝ) (line : ℝ → ℝ → Prop) : Prop := line point.1 point.2
def perpendicular_bisector_of (p1 p2 : ℝ × ℝ) : ℝ → ℝ → Prop := sorry
def line_through (x y : ℝ) : ℝ → ℝ → Prop := sorry

-- Theorem for the perpendicular bisector
theorem perpendicular_bisector_of_BC :
  ∀ x y : ℝ, perpendicular_bisector x y ↔ 
  lies_on (x, y) (perpendicular_bisector_of B C) := by sorry

-- Theorem for the line through P
theorem line_bisected_by_P :
  ∀ x y : ℝ, line_through_P x y ↔
  (∃ a b : ℝ, lies_on (a, 0) (line_through x y) ∧
              lies_on (0, b) (line_through x y) ∧
              P = ((a + 0) / 2, (0 + b) / 2)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_of_BC_line_bisected_by_P_l56_5630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_equals_one_l56_5627

theorem tan_alpha_equals_one (α β : Real) : 
  0 < α ∧ α < Real.pi/2 →  -- α is acute
  0 < β ∧ β < Real.pi/2 →  -- β is acute
  Real.cos (α + β) = Real.sin (α - β) → 
  Real.tan α = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_equals_one_l56_5627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_intersect_N_l56_5660

def M : Set ℤ := {m | -3 < m ∧ m < 2}
def N : Set ℤ := {n : ℤ | 0 ≤ n ∧ n ≤ 3}

theorem M_intersect_N : M ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_intersect_N_l56_5660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symbol_table_theorem_l56_5610

/-- A symbol in the table -/
inductive TableSymbol
| Cross
| Circle

/-- A table filled with symbols -/
def Table (m n : ℕ) := Fin m → Fin n → TableSymbol

/-- Check if two positions are adjacent in the table -/
def adjacent {m n : ℕ} (p q : Fin m × Fin n) : Prop :=
  (p.1 = q.1 ∧ (p.2.val + 1 = q.2.val ∨ p.2.val = q.2.val + 1)) ∨
  (p.2 = q.2 ∧ (p.1.val + 1 = q.1.val ∨ p.1.val = q.1.val + 1))

/-- Check if a table satisfies the adjacency condition -/
def satisfies_condition (m n : ℕ) (t : Table m n) : Prop :=
  ∀ (i : Fin m) (j : Fin n),
    (t i j = TableSymbol.Cross →
      ∃! (k : Fin m) (l : Fin n), adjacent (i, j) (k, l) ∧ t k l = TableSymbol.Circle) ∧
    (t i j = TableSymbol.Circle →
      ∃! (k : Fin m) (l : Fin n), adjacent (i, j) (k, l) ∧ t k l = TableSymbol.Cross)

/-- Theorem stating the impossibility for 3x3 table and possibility for 198x8 table -/
theorem symbol_table_theorem :
  (¬ ∃ (t : Table 3 3), satisfies_condition 3 3 t) ∧
  (∃ (t : Table 198 8), satisfies_condition 198 8 t) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symbol_table_theorem_l56_5610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_difference_2500_l56_5642

/-- The number of digits in the base-b representation of a positive integer n -/
noncomputable def numDigits (n : ℕ) (b : ℕ) : ℕ :=
  if n < b then 1 else Nat.floor (Real.log (n : ℝ) / Real.log (b : ℝ)) + 1

/-- The difference in the number of digits between base-2 and base-7 representations of 2500 -/
theorem digit_difference_2500 :
  numDigits 2500 2 - numDigits 2500 7 = 7 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_difference_2500_l56_5642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_properties_l56_5605

/-- Represents a circular sector -/
structure Sector where
  radius : ℝ
  centralAngle : ℝ

/-- Calculate the perimeter of a sector -/
noncomputable def sectorPerimeter (s : Sector) : ℝ :=
  2 * s.radius + s.centralAngle * s.radius

/-- Calculate the area of a sector -/
noncomputable def sectorArea (s : Sector) : ℝ :=
  0.5 * s.radius * s.radius * s.centralAngle

/-- Calculate the length of the chord AB -/
noncomputable def chordLength (s : Sector) : ℝ :=
  2 * s.radius * Real.sin (s.centralAngle / 2)

theorem sector_properties (s : Sector) 
  (h_perimeter : sectorPerimeter s = 8)
  (h_radius_bounds : 0 < s.radius ∧ s.radius < 4) :
  (sectorArea s = 3 → s.centralAngle = 6 ∨ s.centralAngle = 2/3) ∧
  (∃ s_max : Sector, 
    sectorPerimeter s_max = 8 ∧
    (∀ s' : Sector, sectorPerimeter s' = 8 → sectorArea s' ≤ sectorArea s_max) ∧
    s_max.centralAngle = 2 ∧
    chordLength s_max = 4 * Real.sin 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_properties_l56_5605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_break_even_point_l56_5663

/-- The break-even point for a board game business -/
def board_game_break_even_point 
  (initial_investment : ℚ) 
  (manufacturing_cost : ℚ) 
  (marketing_shipping_cost : ℚ) 
  (selling_price : ℚ) : ℕ :=
let total_cost_per_game := manufacturing_cost + marketing_shipping_cost
let profit_per_game := selling_price - total_cost_per_game
let break_even_point := initial_investment / profit_per_game
(break_even_point.ceil).toNat

/-- The specific break-even point for the given problem -/
theorem specific_break_even_point : 
  board_game_break_even_point 10410 2.65 3.35 20 = 744 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_break_even_point_l56_5663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l56_5628

theorem triangle_side_length (A B C BC S : Real) :
  (Real.sin A + Real.cos A = Real.sqrt 2) →
  (S = 3) →
  (3 * BC * Real.sin A / 2 = S) →
  BC = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l56_5628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_value_l56_5695

theorem sine_function_value (φ : ℝ) (ω : ℝ) (f : ℝ → ℝ) :
  Real.sin φ = 3/5 →
  φ > π/2 →
  φ < π →
  ω > 0 →
  (∀ x, f x = Real.sin (ω * x + φ)) →
  (π / ω = π/2) →
  f (π/4) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_value_l56_5695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_progression_l56_5607

def factorial (r : ℕ) : ℕ := Nat.factorial r

def binom (j k : ℕ) : ℕ := 
  if k ≤ j then
    factorial j / (factorial k * factorial (j - k))
  else
    0

def isArithmeticProgression (a b c : ℕ) : Prop :=
  b - a = c - b

theorem binomial_progression (n : ℕ) (h1 : n > 3) :
  isArithmeticProgression (binom n 1) (binom n 2) (binom n 3) → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_progression_l56_5607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_power_less_than_500_l56_5647

theorem greatest_power_less_than_500 (a b : ℕ) (h1 : b > 1) 
  (h2 : ∀ (x y : ℕ), (y > 1 ∧ x^y < 500) → a^b ≥ x^y) 
  (h3 : a^b < 500) : a + b = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_power_less_than_500_l56_5647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bounded_function_equation_solution_l56_5678

noncomputable def is_bounded (f : ℝ → ℝ) : Prop :=
  ∃ M : ℝ, ∀ x : ℝ, |f x| ≤ M

def satisfies_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * f y) + y * f x = x * f y + f (x * y)

noncomputable def solution_function (x : ℝ) : ℝ :=
  if x ≥ 0 then 0 else -2 * x

theorem bounded_function_equation_solution :
  ∀ f : ℝ → ℝ, is_bounded f → satisfies_equation f →
  f = solution_function := by
  sorry

#check bounded_function_equation_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bounded_function_equation_solution_l56_5678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_calculation_l56_5616

/-- The probability of selecting a real number x from [-2, 2] such that 1 ≤ 2^x ≤ 2 -/
noncomputable def probability_event : ℝ := 1 / 4

/-- The interval from which x is randomly selected -/
def selection_interval : Set ℝ := Set.Icc (-2) 2

/-- The event condition -/
def event_condition (x : ℝ) : Prop := 1 ≤ (2 : ℝ)^x ∧ (2 : ℝ)^x ≤ 2

/-- The measure of the selection interval -/
noncomputable def selection_interval_measure : ℝ := 4

/-- The measure of the event set -/
noncomputable def event_set_measure : ℝ := 1

theorem probability_calculation :
  probability_event = event_set_measure / selection_interval_measure :=
by
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_calculation_l56_5616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_rise_ratio_l56_5671

/-- Represents a right circular cone -/
structure Cone where
  radius : ℝ
  height : ℝ

/-- Represents a spherical marble -/
structure Marble where
  radius : ℝ

/-- Calculates the volume of a cone -/
noncomputable def coneVolume (c : Cone) : ℝ := (1/3) * Real.pi * c.radius^2 * c.height

/-- Calculates the volume of a sphere -/
noncomputable def sphereVolume (m : Marble) : ℝ := (4/3) * Real.pi * m.radius^3

/-- Calculates the new height of a cone after adding a marble -/
noncomputable def newHeight (c : Cone) (m : Marble) : ℝ :=
  c.height + sphereVolume m / (Real.pi * c.radius^2)

/-- Theorem: The ratio of liquid level rise in narrow cone to wide cone is 4:1 -/
theorem liquid_rise_ratio :
  ∀ (narrow_cone wide_cone : Cone) (marble : Marble),
    narrow_cone.radius = 4 →
    wide_cone.radius = 8 →
    marble.radius = 2 →
    coneVolume narrow_cone = coneVolume wide_cone →
    (newHeight narrow_cone marble - narrow_cone.height) /
    (newHeight wide_cone marble - wide_cone.height) = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_rise_ratio_l56_5671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_focal_distances_l56_5698

-- Define the focal distance of an ellipse
noncomputable def ellipse_focal_distance (a b : ℝ) : ℝ := Real.sqrt (a^2 - b^2)

-- Define the focal distance of a hyperbola
noncomputable def hyperbola_focal_distance (a b : ℝ) : ℝ := Real.sqrt (a^2 - b^2)

-- Theorem statement
theorem equal_focal_distances (k : ℝ) (h : 12 < k ∧ k < 16) :
  ellipse_focal_distance 4 (2 * Real.sqrt 3) = hyperbola_focal_distance (Real.sqrt (16 - k)) (Real.sqrt (12 - k)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_focal_distances_l56_5698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_coins_theorem_l56_5685

-- Define the circles
variable (Γ₁ Γ₂ Γ₃ Γ₄ : Set (EuclideanSpace ℝ (Fin 2)))

-- Define the points of tangency
variable (A B C D : EuclideanSpace ℝ (Fin 2))

-- Define the property of external tangency
def externally_tangent (c1 c2 : Set (EuclideanSpace ℝ (Fin 2))) : Prop :=
  ∃ p, p ∈ c1 ∧ p ∈ c2 ∧ ∀ q, q ∈ c1 → q ∈ c2 → q = p

-- Define the concyclic property
def concyclic (A B C D : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ c : Set (EuclideanSpace ℝ (Fin 2)), A ∈ c ∧ B ∈ c ∧ C ∈ c ∧ D ∈ c

-- State the theorem
theorem four_coins_theorem 
  (h1 : externally_tangent Γ₁ Γ₂)
  (h2 : externally_tangent Γ₂ Γ₃)
  (h3 : externally_tangent Γ₃ Γ₄)
  (h4 : externally_tangent Γ₄ Γ₁)
  (hA : A ∈ Γ₁ ∧ A ∈ Γ₂)
  (hB : B ∈ Γ₂ ∧ B ∈ Γ₃)
  (hC : C ∈ Γ₃ ∧ C ∈ Γ₄)
  (hD : D ∈ Γ₄ ∧ D ∈ Γ₁) :
  concyclic A B C D :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_coins_theorem_l56_5685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_positive_probability_l56_5697

def S : Finset Int := {-3, -7, 10, 2, -1}

theorem product_positive_probability :
  let pairs := S.product S |>.filter (λ p => p.1 ≠ p.2)
  (pairs.filter (λ p => p.1 * p.2 > 0)).card / pairs.card = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_positive_probability_l56_5697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l56_5683

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x + 3 * Real.pi / 4) * Real.cos x

theorem f_properties :
  ∃ (T : ℝ),
    (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
    T = Real.pi ∧
    (∀ (x : ℝ), x ∈ Set.Icc (-5 * Real.pi / 4) (5 * Real.pi / 4) → f x ≤ 1 + Real.sqrt 3) ∧
    (∃ (x : ℝ), x ∈ Set.Icc (-5 * Real.pi / 4) (5 * Real.pi / 4) ∧ f x = 1 + Real.sqrt 3) ∧
    (∀ (x : ℝ), x ∈ Set.Icc (-5 * Real.pi / 4) (5 * Real.pi / 4) → f x ≥ -3 + Real.sqrt 3) ∧
    (∃ (x : ℝ), x ∈ Set.Icc (-5 * Real.pi / 4) (5 * Real.pi / 4) ∧ f x = -3 + Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l56_5683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cubes_required_l56_5658

/-- Represents a 3D figure constructed from unit cubes --/
structure Figure3D where
  cubes : Finset (ℕ × ℕ × ℕ)

/-- Checks if the figure satisfies the front view --/
def satisfiesFrontView (f : Figure3D) : Prop :=
  ∃ (c1 c2 : Finset (ℕ × ℕ × ℕ)), 
    c1 ⊆ f.cubes ∧ c2 ⊆ f.cubes ∧
    c1.card = 9 ∧ c2.card = 2 ∧
    (∀ (x y z : ℕ), (x, y, z) ∈ c1 → x = 0 ∧ y < 3 ∧ z < 3) ∧
    (∀ (x y z : ℕ), (x, y, z) ∈ c2 → x = 1 ∧ y ≥ 1 ∧ y < 3 ∧ z = 0)

/-- Checks if the figure satisfies the side view --/
def satisfiesSideView (f : Figure3D) : Prop :=
  ∃ (c : Finset (ℕ × ℕ × ℕ)),
    c ⊆ f.cubes ∧
    c.card = 11 ∧
    (∀ (x y z : ℕ), (x, y, z) ∈ c → z < 3) ∧
    (∃ (x y : ℕ), (x, y, 2) ∈ c)

/-- Checks if each cube shares at least one face with another cube --/
def sharesAtLeastOneFace (f : Figure3D) : Prop :=
  ∀ (x1 y1 z1 : ℕ), (x1, y1, z1) ∈ f.cubes →
    ∃ (x2 y2 z2 : ℕ), (x2, y2, z2) ∈ f.cubes ∧ (x2, y2, z2) ≠ (x1, y1, z1) ∧
      ((x2 = x1 ∧ y2 = y1 ∧ (z2 = z1 + 1 ∨ z2 = z1 - 1)) ∨
       (x2 = x1 ∧ z2 = z1 ∧ (y2 = y1 + 1 ∨ y2 = y1 - 1)) ∨
       (y2 = y1 ∧ z2 = z1 ∧ (x2 = x1 + 1 ∨ x2 = x1 - 1)))

/-- The main theorem stating that the minimum number of cubes required is 11 --/
theorem min_cubes_required : 
  ∀ (f : Figure3D), 
    satisfiesFrontView f → 
    satisfiesSideView f → 
    sharesAtLeastOneFace f → 
    f.cubes.card ≥ 11 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cubes_required_l56_5658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boiling_water_theorem_l56_5652

/-- The amount of boiling water needed to achieve a desired temperature -/
noncomputable def boiling_water_amount (a t₁ t₂ : ℝ) : ℝ :=
  (a * (t₂ - t₁)) / (100 - t₂)

theorem boiling_water_theorem (a t₁ t₂ : ℝ) (h₁ : t₁ < 100) (h₂ : t₂ < 100) :
  let x := boiling_water_amount a t₁ t₂
  x * (100 - t₂) = a * (t₂ - t₁) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boiling_water_theorem_l56_5652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l56_5626

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of a triangle -/
noncomputable def area (t : Triangle) : ℝ := (1/2) * t.a * t.b * Real.sin t.C

/-- Theorem: The area of triangle ABC is 132/25, given B = 2C, b = 6, and c = 5 -/
theorem triangle_area (t : Triangle) 
  (h1 : t.B = 2 * t.C) 
  (h2 : t.b = 6) 
  (h3 : t.c = 5) : 
  area t = 132/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l56_5626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_m_circle_line_intersection_m_l56_5694

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 - y^2 - 2*x - 4*y + m = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  x + 2*y - 4 = 0

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Theorem 1
theorem circle_radius_m (m : ℝ) :
  (∀ x y, circle_equation x y m → distance x y 1 2 = 2) → m = 1 :=
by sorry

-- Theorem 2
theorem circle_line_intersection_m (m : ℝ) :
  (∃ x1 y1 x2 y2, 
    circle_equation x1 y1 m ∧ 
    circle_equation x2 y2 m ∧ 
    line_equation x1 y1 ∧ 
    line_equation x2 y2 ∧ 
    distance x1 y1 x2 y2 = 4 * Real.sqrt 5 / 5) → 
  m = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_m_circle_line_intersection_m_l56_5694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_length_l56_5637

/-- The common focus of the ellipse and parabola -/
def F : ℝ × ℝ := (1, 0)

/-- The ellipse C₁ -/
def C₁ (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- The parabola C₂ -/
def C₂ (p x y : ℝ) : Prop := y^2 = 2*p*x

/-- The parameter p of the parabola -/
noncomputable def p : ℝ := 2

/-- The intersection points of C₁ and C₂ -/
def intersectionPoints : Set (ℝ × ℝ) := {(x, y) | C₁ x y ∧ C₂ p x y}

/-- The length of segment AB -/
noncomputable def lengthAB : ℝ := Real.sqrt 6 * 4/3

theorem intersection_length : 
  F = (1, 0) →
  p > 0 →
  ∃ (A B : ℝ × ℝ), A ∈ intersectionPoints ∧ B ∈ intersectionPoints ∧
    dist A B = lengthAB :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_length_l56_5637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_prism_faces_l56_5649

/-- A shape representing a prism -/
structure Prism where
  total_edges : ℕ
  base_shape : Shape
  total_faces : ℕ

/-- Possible shapes for the base of a prism -/
inductive Shape where
  | Hexagon

/-- A prism with hexagonal bases and 18 edges has 8 faces. -/
theorem hexagonal_prism_faces :
  ∀ (prism : Prism),
  prism.total_edges = 18 ∧
  prism.base_shape = Shape.Hexagon →
  prism.total_faces = 8 :=
by
  intro prism h
  sorry

#check hexagonal_prism_faces

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_prism_faces_l56_5649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l56_5655

def vector1 : ℝ × ℝ := (4, -5)

theorem perpendicular_vectors : 
  ∃ b : ℝ, (vector1.1 * b + vector1.2 * 3 = 0) ∧ b = 15/4 := by
  use 15/4
  constructor
  · simp [vector1]
    norm_num
  · rfl

#check perpendicular_vectors

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l56_5655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_at_one_l56_5661

-- Define the function f(x) = 1/x
noncomputable def f (x : ℝ) : ℝ := 1 / x

-- State the theorem
theorem limit_at_one (ε : ℝ) (hε : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧ ∀ h : ℝ, 0 < |h| ∧ |h| < δ →
    |(f (1 - h) - f 1) / h - 1| < ε := by
  sorry

#check limit_at_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_at_one_l56_5661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alloy_mixture_theorem_l56_5684

/-- Represents an alloy with a specific chromium percentage -/
structure Alloy where
  chromium_percentage : ℝ
  mass : ℝ

/-- Calculates the mass of chromium in an alloy -/
noncomputable def chromium_mass (a : Alloy) : ℝ := a.chromium_percentage * a.mass / 100

/-- Represents the mixture of two alloys -/
noncomputable def mix_alloys (a1 a2 : Alloy) : Alloy :=
  { chromium_percentage := (chromium_mass a1 + chromium_mass a2) / (a1.mass + a2.mass) * 100,
    mass := a1.mass + a2.mass }

theorem alloy_mixture_theorem (x : ℝ) :
  let alloy1 : Alloy := { chromium_percentage := 12, mass := x }
  let alloy2 : Alloy := { chromium_percentage := 8, mass := 30 }
  let mixture := mix_alloys alloy1 alloy2
  mixture.chromium_percentage = 9.333333333333334 → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alloy_mixture_theorem_l56_5684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_equation_l56_5624

theorem unique_solution_for_equation : 
  ∃! (n : ℕ), (n + 1100) / 80 = ⌊Real.sqrt n⌋ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_equation_l56_5624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_approx_l56_5604

-- Define the given parameters
noncomputable def train_length : ℝ := 110
noncomputable def bridge_length : ℝ := 340
noncomputable def crossing_time : ℝ := 26.997840172786177

-- Define the total distance traveled
noncomputable def total_distance : ℝ := train_length + bridge_length

-- Define the speed of the train
noncomputable def train_speed : ℝ := total_distance / crossing_time

-- Theorem to prove
theorem train_speed_approx : 
  ∀ ε > 0, |train_speed - 16.669| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_approx_l56_5604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_lateral_surface_area_l56_5618

/-- The lateral surface area of a right square cone -/
noncomputable def lateral_surface_area (base_edge : ℝ) (height : ℝ) : ℝ :=
  2 * base_edge * Real.sqrt (height^2 + (base_edge/2)^2)

/-- Theorem: The lateral surface area of a right square cone with base edge length 3 and height √17/2 is 3√26 -/
theorem cone_lateral_surface_area :
  lateral_surface_area 3 (Real.sqrt 17 / 2) = 3 * Real.sqrt 26 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_lateral_surface_area_l56_5618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_distance_l56_5679

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Check if a point is on the hyperbola -/
def isOnHyperbola (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / (64 : ℝ) - p.y^2 / (36 : ℝ) = 1

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: For a point P on the given hyperbola, if |PF₁| = 17, then |PF₂| = 33 -/
theorem hyperbola_focal_distance (h : Hyperbola) (p f1 f2 : Point)
    (h_on_hyperbola : isOnHyperbola h p)
    (h_foci : distance f1 f2 = 20) -- Distance between foci is 2c = 20
    (h_pf1 : distance p f1 = 17) :
    distance p f2 = 33 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_distance_l56_5679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_amount_is_correct_l56_5645

-- Define the initial amount
def initial_amount : ℚ := 204

-- Define the spending percentages
def toy_percentage : ℚ := 40 / 100
def book_percentage : ℚ := 50 / 100
def charity_percentage : ℚ := 20 / 100
def gift_percentage : ℚ := 30 / 100
def future_expenses_percentage : ℚ := 10 / 100

-- Function to calculate remaining amount after a transaction
def remaining_after_transaction (amount : ℚ) (percentage : ℚ) : ℚ :=
  amount * (1 - percentage)

-- Calculate the final amount
def final_amount : ℚ :=
  let after_toy := remaining_after_transaction initial_amount toy_percentage
  let after_book := remaining_after_transaction after_toy book_percentage
  let after_charity := remaining_after_transaction after_book charity_percentage
  let after_gift := remaining_after_transaction after_charity gift_percentage
  remaining_after_transaction after_gift future_expenses_percentage

-- Function to round to two decimal places
def round_to_cents (q : ℚ) : ℚ :=
  (q * 100).floor / 100

-- Theorem to prove
theorem final_amount_is_correct :
  round_to_cents final_amount = 23.13 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_amount_is_correct_l56_5645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_l56_5699

-- Define the custom operation
noncomputable def circle_slash (a b : ℝ) : ℝ := (Real.sqrt (3 * a + b)) ^ 3

-- Theorem statement
theorem solve_equation (x : ℝ) (h : circle_slash 4 x = 64) : x = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_l56_5699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_A_approximation_l56_5666

noncomputable def A : ℝ := 1/9 * (1 - 10^(-10 : ℤ))

theorem sqrt_A_approximation :
  ∃ (error : ℝ), error < 10^(-20 : ℤ) ∧
  |Real.sqrt A - 0.33333333331666666666| ≤ error := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_A_approximation_l56_5666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_l56_5650

theorem coefficient_of_x (some_number : ℝ) (x : ℝ) : 
  (4 : ℝ) ^ (some_number * x + 2) = (16 : ℝ) ^ (3 * x - 1) → x = 1 → some_number = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_l56_5650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l56_5600

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * (Real.sin x + Real.cos x)

theorem f_properties :
  let period : ℝ := π
  let monotonic_interval (k : ℤ) : Set ℝ := Set.Icc (k * π - 3 * π / 8) (k * π + π / 8)
  let min_value : ℝ := 0
  let max_value : ℝ := Real.sqrt 2 + 1
  -- Smallest positive period
  (∀ (x : ℝ), f (x + period) = f x) ∧
  (∀ (t : ℝ), 0 < t ∧ t < period → ∃ (x : ℝ), f (x + t) ≠ f x) ∧
  -- Monotonically increasing intervals
  (∀ (k : ℤ) (x y : ℝ), x ∈ monotonic_interval k → y ∈ monotonic_interval k → x ≤ y → f x ≤ f y) ∧
  -- Minimum and maximum values on [-π/4, π/4]
  (∀ (x : ℝ), x ∈ Set.Icc (-π/4) (π/4) → f x ≥ min_value) ∧
  (∀ (x : ℝ), x ∈ Set.Icc (-π/4) (π/4) → f x ≤ max_value) ∧
  (∃ (x : ℝ), x ∈ Set.Icc (-π/4) (π/4) ∧ f x = min_value) ∧
  (∃ (x : ℝ), x ∈ Set.Icc (-π/4) (π/4) ∧ f x = max_value) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l56_5600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_roots_property_l56_5603

/-- A polynomial of degree 3 with integer coefficients -/
def polynomial (c d : ℤ) (x : ℝ) : ℝ := x^3 + c*x^2 + d*x + 15*c

theorem polynomial_roots_property (c d : ℤ) :
  c ≠ 0 ∧ d ≠ 0 →
  ∃ (u v : ℤ), 
    (∀ x : ℝ, polynomial c d x = (x - (u : ℝ))^2 * (x - (v : ℝ))) →
    |c * d| = 840 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_roots_property_l56_5603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_properties_l56_5622

theorem power_of_two_properties (n : ℕ) :
  (∃ k : ℕ, n = 3 * k ↔ (2^n - 1) % 7 = 0) ∧
  (2^n + 1) % 7 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_properties_l56_5622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l56_5693

/-- The minimum value of ω for the given cosine function with specified properties -/
theorem min_omega_value (ω φ T : ℝ) (h_ω_pos : ω > 0) (h_φ_range : 0 < φ ∧ φ < π)
  (h_period : T = 2 * π / ω)
  (h_f_T : Real.cos (ω * T + φ) = 1/2)
  (h_critical_point : ∃ k : ℤ, ω * (7 * π / 3) + φ = k * π) :
  ω ≥ 2/7 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l56_5693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_four_solutions_l56_5669

/-- The set of solutions to the equation (x-3) * sin(π*x) = 1 where x > 0 -/
def SolutionSet : Set ℝ :=
  {x | (x - 3) * Real.sin (Real.pi * x) = 1 ∧ x > 0}

/-- Theorem stating that the sum of any four elements from the solution set is at least 12 -/
theorem min_sum_of_four_solutions :
  ∀ (x₁ x₂ x₃ x₄ : ℝ), x₁ ∈ SolutionSet → x₂ ∈ SolutionSet → x₃ ∈ SolutionSet → x₄ ∈ SolutionSet →
  x₁ + x₂ + x₃ + x₄ ≥ 12 := by
  sorry

#check min_sum_of_four_solutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_four_solutions_l56_5669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_n_value_l56_5651

def is_valid_n (n : ℕ) : Prop :=
  n < 60 ∧ n > 0 ∧ ∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ p ∣ n ∧ q ∣ n

def unit_digit (n : ℕ) : ℕ := n % 10

def num_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

theorem unique_n_value :
  ∃! n : ℕ, is_valid_n n ∧
    (∀ m : ℕ, is_valid_n m → unit_digit m = unit_digit n → m = n) ∧
    (∀ m : ℕ, is_valid_n m → num_divisors m = num_divisors n → m = n) ∧
    n = 10 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_n_value_l56_5651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_l56_5619

/-- A function f is symmetric about a line x = a if f(a + x) = f(a - x) for all x -/
def SymmetricAboutLine (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

theorem function_symmetry (ω φ : ℝ) (h_ω : ω > 0) (h_φ : |φ| < π/2)
  (h_period : ∀ x, Real.sin (ω * (x + π) + φ) = Real.sin (ω * x + φ))
  (h_shift : ∀ x, Real.sin (ω * (x - π/6) + φ) = Real.sin (ω * x)) :
  SymmetricAboutLine (fun x ↦ Real.sin (ω * x + φ)) (π/12) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_l56_5619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_implies_equilateral_l56_5629

-- Define the triangle and point P₀
variable (A₁ A₂ A₃ P₀ : ℂ)

-- Define the sequence of points
noncomputable def P : ℕ → ℂ
  | 0 => P₀
  | (k + 1) => let A := if k % 3 = 0 then A₁ else if k % 3 = 1 then A₂ else A₃
               A + (P k - A) * Complex.exp (Complex.I * (2 * Real.pi / 3))

-- Define the condition P₁₉₈₆ = P₀
def rotation_condition (A₁ A₂ A₃ P₀ : ℂ) : Prop :=
  P A₁ A₂ A₃ P₀ 1986 = P₀

-- Define an equilateral triangle
def is_equilateral (A₁ A₂ A₃ : ℂ) : Prop :=
  Complex.abs (A₂ - A₁) = Complex.abs (A₃ - A₂) ∧
  Complex.abs (A₃ - A₂) = Complex.abs (A₁ - A₃)

-- The theorem to prove
theorem rotation_implies_equilateral (A₁ A₂ A₃ P₀ : ℂ) :
  rotation_condition A₁ A₂ A₃ P₀ → is_equilateral A₁ A₂ A₃ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_implies_equilateral_l56_5629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_turtle_hatchlings_l56_5662

/-- The number of eggs laid by each turtle -/
def eggs_per_turtle : ℕ := 20

/-- The success rate of hatching -/
def hatch_rate : ℚ := 2/5

/-- The number of turtles -/
def num_turtles : ℕ := 6

/-- The number of hatchlings produced by the given number of turtles -/
def total_hatchlings : ℕ := 
  (num_turtles * (eggs_per_turtle * hatch_rate)).floor.toNat

theorem turtle_hatchlings :
  total_hatchlings = 48 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_turtle_hatchlings_l56_5662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_hyperbola_asymptote_angle_l56_5665

noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

noncomputable def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

def same_foci (e_foci h_foci : ℝ × ℝ) : Prop := e_foci = h_foci

def reciprocal_eccentricities (e_ecc h_ecc : ℝ) : Prop := e_ecc * h_ecc = 1

noncomputable def asymptote_angle_sine (a b : ℝ) : ℝ := b / Real.sqrt (a^2 + b^2)

theorem ellipse_hyperbola_asymptote_angle 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (e_foci h_foci : ℝ × ℝ) 
  (e_ecc h_ecc : ℝ) 
  (h_same_foci : same_foci e_foci h_foci)
  (h_reciprocal_ecc : reciprocal_eccentricities e_ecc h_ecc) :
  asymptote_angle_sine a b = Real.sqrt 3 / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_hyperbola_asymptote_angle_l56_5665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_quadrilateral_area_l56_5654

/-- Definition of a convex quadrilateral -/
def ConvexQuadrilateral (A B C D : EuclideanSpace ℝ (Fin 2)) : Prop :=
sorry

/-- Helper function to calculate the area of a quadrilateral -/
noncomputable def area_of_quadrilateral (A B C D : EuclideanSpace ℝ (Fin 2)) : ℝ :=
sorry

/-- Given a convex quadrilateral ABCD with area s, if we extend each side by its own length
    to form a new quadrilateral A₁B₁C₁D₁, then the area of A₁B₁C₁D₁ is equal to 5s. -/
theorem extended_quadrilateral_area
  (A B C D A₁ B₁ C₁ D₁ : EuclideanSpace ℝ (Fin 2))
  (s : ℝ)
  (convex : ConvexQuadrilateral A B C D)
  (area_ABCD : area_of_quadrilateral A B C D = s)
  (extend_AB : B₁ - A = 2 • (B - A))
  (extend_BC : C₁ - B = 2 • (C - B))
  (extend_CD : D₁ - C = 2 • (D - C))
  (extend_DA : A₁ - D = 2 • (A - D)) :
  area_of_quadrilateral A₁ B₁ C₁ D₁ = 5 * s :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_quadrilateral_area_l56_5654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_difference_theorem_l56_5664

/-- Represents the number of years after 2001 -/
def n : ℕ := sorry

/-- Represents the annual price increase of commodity Y in cents -/
def y : ℚ := sorry

/-- The price of commodity X in a given year -/
def price_x (n : ℕ) : ℚ := 420/100 + 45/100 * n

/-- The price of commodity Y in a given year -/
def price_y (n : ℕ) (y : ℚ) : ℚ := 630/100 + y/100 * n

/-- Theorem stating the relationship between n and y when the price difference is 65 cents -/
theorem price_difference_theorem (n : ℕ) (y : ℚ) :
  price_x n = price_y n y + 65/100 ↔ n = 275 / (45 - y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_difference_theorem_l56_5664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closed_geometric_sequence_propositions_l56_5639

/-- A geometric sequence with first term a and common ratio q -/
def geometric_sequence (a q : ℝ) : ℕ → ℝ := fun n => a * q ^ (n - 1)

/-- A closed geometric sequence is one where the product of any two terms is also a term in the sequence -/
def is_closed_geometric_sequence (s : ℕ → ℝ) : Prop :=
  ∀ m n, ∃ k, s m * s n = s k

/-- Proposition 1: If a₁ = 3 and q = 2, then {aₙ} is a closed geometric sequence -/
def proposition1 : Prop :=
  is_closed_geometric_sequence (geometric_sequence 3 2)

/-- Proposition 2: If a₁ = 1/2 and q = 2, then {aₙ} is a closed geometric sequence -/
def proposition2 : Prop :=
  is_closed_geometric_sequence (geometric_sequence (1/2) 2)

/-- Proposition 3: If {aₙ} and {bₙ} are both closed geometric sequences, 
    then {aₙ * bₙ} and {aₙ + bₙ} are also closed geometric sequences -/
def proposition3 : Prop :=
  ∀ a b : ℕ → ℝ, is_closed_geometric_sequence a → is_closed_geometric_sequence b →
    is_closed_geometric_sequence (fun n => a n * b n) ∧ 
    is_closed_geometric_sequence (fun n => a n + b n)

/-- Proposition 4: There does not exist a sequence {aₙ} such that 
    both {aₙ} and {aₙ²} are closed geometric sequences -/
def proposition4 : Prop :=
  ¬∃ a : ℕ → ℝ, is_closed_geometric_sequence a ∧ is_closed_geometric_sequence (fun n => (a n)^2)

theorem closed_geometric_sequence_propositions :
  ¬proposition1 ∧ proposition2 ∧ ¬proposition3 ∧ ¬proposition4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closed_geometric_sequence_propositions_l56_5639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_at_one_l56_5656

/-- A polynomial with real coefficients -/
def MyPolynomial := ℝ → ℝ

/-- The mean of nonzero coefficients of a polynomial -/
noncomputable def mean_nonzero_coeff (P : MyPolynomial) : ℝ := sorry

/-- Create a new polynomial by replacing nonzero coefficients with their mean -/
noncomputable def replace_with_mean (P : MyPolynomial) : MyPolynomial := 
  λ x => mean_nonzero_coeff P * (P x)

/-- Theorem: For any polynomial P, P(1) equals Q(1) where Q is formed by replacing
    nonzero coefficients of P with their mean -/
theorem equal_at_one (P : MyPolynomial) : 
  P 1 = (replace_with_mean P) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_at_one_l56_5656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_is_real_z_in_third_quadrant_l56_5620

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := ⟨m^2 + 5*m + 6, m^2 - 2*m - 15⟩

-- Theorem for when z is a real number
theorem z_is_real (m : ℝ) : (z m).im = 0 ↔ m = -3 ∨ m = 5 := by sorry

-- Theorem for when z lies in the third quadrant
theorem z_in_third_quadrant (m : ℝ) : 
  (z m).re < 0 ∧ (z m).im < 0 ↔ m > -3 ∧ m < -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_is_real_z_in_third_quadrant_l56_5620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gerald_tasks_per_month_l56_5646

noncomputable def monthly_supply_cost : ℝ := 100
def season_length : ℕ := 4
noncomputable def glove_cost : ℝ := 80
def saving_months : ℕ := 8
noncomputable def raking_charge : ℝ := 8
noncomputable def shoveling_charge : ℝ := 12
noncomputable def mowing_charge : ℝ := 15

noncomputable def total_season_cost : ℝ := monthly_supply_cost * season_length + glove_cost

noncomputable def monthly_saving_goal : ℝ := total_season_cost / saving_months

noncomputable def average_task_charge : ℝ := (raking_charge + shoveling_charge + mowing_charge) / 3

theorem gerald_tasks_per_month :
  ∃ n : ℕ, n ≥ 6 ∧ n * average_task_charge ≥ monthly_saving_goal := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gerald_tasks_per_month_l56_5646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_1964_not_divisible_by_4_l56_5608

def a : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => a (n + 1) * a n + 1

theorem a_1964_not_divisible_by_4 : ¬ (4 ∣ a 1964) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_1964_not_divisible_by_4_l56_5608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oh_count_independent_of_order_oh_count_is_n_minus_100_l56_5692

/-- Represents a spectator with their assigned seat number -/
structure Spectator where
  seat : Nat

/-- Represents the state of the seating arrangement -/
structure SeatingState where
  occupied : Finset Nat
  ohCount : Nat

/-- The total number of seats -/
def totalSeats : Nat := 1000

/-- The number of tickets sold -/
def n : Nat := 101  -- Example value, can be changed as needed

/-- Assumption that 100 < n < 1000 -/
axiom n_bounds : 100 < n ∧ n < 1000

/-- Function to simulate seating a spectator -/
def seatSpectator (state : SeatingState) (s : Spectator) : SeatingState :=
  sorry

/-- Theorem stating that the number of "Oh!"s is independent of arrival order -/
theorem oh_count_independent_of_order 
  (spectators : List Spectator) 
  (h_spectators : spectators.length = n) :
  ∀ (perm : List Spectator), 
    perm.length = n → 
    (perm.Perm spectators) → 
    (spectators.foldl seatSpectator ⟨∅, 0⟩).ohCount = 
    (perm.foldl seatSpectator ⟨∅, 0⟩).ohCount :=
  sorry

/-- Theorem stating that the number of "Oh!"s is always n - 100 -/
theorem oh_count_is_n_minus_100 
  (spectators : List Spectator) 
  (h_spectators : spectators.length = n) :
  (spectators.foldl seatSpectator ⟨∅, 0⟩).ohCount = n - 100 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oh_count_independent_of_order_oh_count_is_n_minus_100_l56_5692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_two_l56_5602

/-- A monic quartic polynomial satisfying specific conditions -/
noncomputable def f : ℝ → ℝ := sorry

/-- f is a monic quartic polynomial -/
axiom f_monic_quartic : ∃ a b c : ℝ, ∀ x, f x = x^4 + a*x^3 + b*x^2 + c*x + (f 0)

/-- Conditions on f -/
axiom f_cond1 : f (-2) = -4
axiom f_cond2 : f 1 = -1
axiom f_cond3 : f 3 = -9
axiom f_cond4 : f (-4) = -16

/-- Theorem: f(2) = -28 -/
theorem f_at_two : f 2 = -28 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_two_l56_5602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_B_proportional_l56_5686

def group_A : List ℝ := [1, 2, 3, 4]
def group_B : List ℝ := [1, 2, 2, 4]
def group_C : List ℝ := [3, 5, 9, 13]
def group_D : List ℝ := [1, 2, 2, 3]

def is_proportional (l : List ℝ) : Prop :=
  l.length = 4 ∧ l[0]! * l[3]! = l[1]! * l[2]!

theorem only_B_proportional :
  ¬(is_proportional group_A) ∧
  (is_proportional group_B) ∧
  ¬(is_proportional group_C) ∧
  ¬(is_proportional group_D) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_B_proportional_l56_5686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l56_5644

-- Define the line passing through (3, 2) for all real a
noncomputable def line_through_point (a : ℝ) : ℝ → ℝ := λ x ↦ a * x - 3 * a + 2

-- Define the line y = 3x - 2
noncomputable def line_with_y_intercept : ℝ → ℝ := λ x ↦ 3 * x - 2

-- Define the two perpendicular lines
noncomputable def line1 : ℝ → ℝ := λ x ↦ (x - 3) / 2
noncomputable def line2 : ℝ → ℝ := λ x ↦ -2 * x

theorem line_properties :
  (∀ a : ℝ, line_through_point a 3 = 2) ∧
  (line_with_y_intercept 0 = -2) ∧
  (line2 (-1) = 2 ∧ (∀ x : ℝ, line1 x * line2 x = -1)) := by
  sorry

#check line_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l56_5644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l56_5621

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ a ≥ b

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line perpendicular to the x-axis -/
structure PerpendicularLine where
  x : ℝ

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a) ^ 2)

/-- The right focus of an ellipse -/
noncomputable def rightFocus (e : Ellipse) : Point :=
  { x := Real.sqrt (e.a ^ 2 - e.b ^ 2), y := 0 }

/-- The left focus of an ellipse -/
noncomputable def leftFocus (e : Ellipse) : Point :=
  { x := -Real.sqrt (e.a ^ 2 - e.b ^ 2), y := 0 }

/-- Predicate to check if three points form an equilateral triangle -/
def isEquilateralTriangle (p1 p2 p3 : Point) : Prop :=
  let d12 := Real.sqrt ((p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2)
  let d23 := Real.sqrt ((p2.x - p3.x) ^ 2 + (p2.y - p3.y) ^ 2)
  let d31 := Real.sqrt ((p3.x - p1.x) ^ 2 + (p3.y - p1.y) ^ 2)
  d12 = d23 ∧ d23 = d31

/-- The main theorem -/
theorem ellipse_eccentricity (e : Ellipse) (l : PerpendicularLine) (A B : Point) :
  l.x = (rightFocus e).x →
  A.x = l.x ∧ B.x = l.x →
  (∀ p : Point, p.x = A.x ∧ p.y = A.y ∨ p.x = B.x ∧ p.y = B.y → 
    (p.x / e.a) ^ 2 + (p.y / e.b) ^ 2 = 1) →
  isEquilateralTriangle A B (leftFocus e) →
  eccentricity e = Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l56_5621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_plus_b_equals_one_l56_5634

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  if x > 0 then x - 1
  else if x = 0 then a
  else x + b

def isOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_implies_a_plus_b_equals_one (a b : ℝ) :
  isOdd (f · a b) → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_plus_b_equals_one_l56_5634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_row_is_531_l56_5631

/-- A type representing the numbers 1 to 5 -/
inductive FiveNum
  | one
  | two
  | three
  | four
  | five

/-- A 5x5 grid filled with FiveNum values -/
def Grid := Fin 5 → Fin 5 → FiveNum

/-- Check if a grid satisfies the no-repeat condition -/
def noRepeat (g : Grid) : Prop :=
  ∀ i j k, i ≠ k → g i j ≠ g k j ∧ g j i ≠ g j k

/-- Convert FiveNum to ℕ -/
def fiveNumToNat : FiveNum → ℕ
  | FiveNum.one => 1
  | FiveNum.two => 2
  | FiveNum.three => 3
  | FiveNum.four => 4
  | FiveNum.five => 5

/-- Check if division is integer for adjacent cells -/
def integerDivision (g : Grid) : Prop :=
  ∀ i j, i < 4 → j < 4 →
    (fiveNumToNat (g i j) % fiveNumToNat (g (i+1) j) = 0) ∧
    (fiveNumToNat (g j i) % fiveNumToNat (g j (i+1)) = 0)

/-- The main theorem -/
theorem second_row_is_531 (g : Grid) 
  (h1 : noRepeat g) 
  (h2 : integerDivision g) : 
  fiveNumToNat (g 1 0) = 5 ∧ fiveNumToNat (g 1 1) = 3 ∧ fiveNumToNat (g 1 2) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_row_is_531_l56_5631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_continuous_l56_5632

-- Define the function f(x) = cos(x^2)
noncomputable def f (x : ℝ) : ℝ := Real.cos (x^2)

-- State the theorem
theorem f_continuous : Continuous f := by
  sorry

-- Note: The proof is omitted as per the instructions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_continuous_l56_5632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bf_length_l56_5673

structure Quadrilateral :=
  (A B C D E F P : ℝ × ℝ)

def is_right_angle (p q r : ℝ × ℝ) : Prop := sorry

def on_line_segment (p q r : ℝ × ℝ) : Prop := sorry

def perpendicular (p q r s : ℝ × ℝ) : Prop := sorry

def extend_to_point (p q r : ℝ × ℝ) : Prop := sorry

noncomputable def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem bf_length (quad : Quadrilateral) : 
  is_right_angle quad.A quad.C quad.D →
  is_right_angle quad.C quad.A quad.B →
  on_line_segment quad.A quad.C quad.E →
  on_line_segment quad.A quad.C quad.F →
  perpendicular quad.D quad.E quad.A quad.C →
  perpendicular quad.B quad.F quad.A quad.C →
  extend_to_point quad.A quad.B quad.P →
  extend_to_point quad.C quad.D quad.P →
  distance quad.A quad.E = 4 →
  distance quad.D quad.E = 6 →
  distance quad.C quad.E = 8 →
  ∃ (ε : ℝ), abs (distance quad.B quad.F - 8.47) < ε ∧ ε > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bf_length_l56_5673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_given_vector_sum_l56_5609

open Real

/-- Helper function to define a triangle given its vertices -/
def triangle (A B C : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {P | ∃ (α β γ : ℝ), α ≥ 0 ∧ β ≥ 0 ∧ γ ≥ 0 ∧ α + β + γ = 1 ∧
    P = (α * A.1 + β * B.1 + γ * C.1, α * A.2 + β * B.2 + γ * C.2)}

/-- Helper function to calculate the area of a triangle given its vertices -/
noncomputable def area (t : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : ℝ := 
  let (A, B, C) := t
  abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2)

/-- Given a triangle ABC and a point O inside it, if the vector sum condition is satisfied,
    then the ratio of areas of triangles AOC and ABC is 1:2 -/
theorem area_ratio_given_vector_sum (A B C O : ℝ × ℝ) : 
  O ∈ interior (triangle A B C) →
  (A.1 - O.1, A.2 - O.2) + (C.1 - O.1, C.2 - O.2) + 2 • (B.1 - O.1, B.2 - O.2) = (0, 0) →
  area (A, O, C) / area (A, B, C) = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_given_vector_sum_l56_5609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_theorem_l56_5641

-- Define the trapezium area function
noncomputable def trapezium_area (a b h : ℝ) : ℝ := (a + b) * h / 2

-- Theorem statement
theorem trapezium_area_theorem : 
  let a := (20 : ℝ) -- Length of one parallel side
  let b := (18 : ℝ) -- Length of the other parallel side
  let h := (11 : ℝ) -- Distance between parallel sides
  trapezium_area a b h = 209 := by
  -- Unfold the definition of trapezium_area
  unfold trapezium_area
  -- Simplify the expression
  simp
  -- The proof is completed with sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_theorem_l56_5641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cow_spot_percentage_l56_5690

theorem cow_spot_percentage (total_cows : ℕ) (red_spot_percentage : ℚ) (no_spot_cows : ℕ) :
  total_cows = 140 →
  red_spot_percentage = 2/5 →
  no_spot_cows = 63 →
  (total_cows - (red_spot_percentage * ↑total_cows).floor - no_spot_cows) / 
  (total_cows - (red_spot_percentage * ↑total_cows).floor) = 1/4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cow_spot_percentage_l56_5690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_factorable_b_l56_5638

/-- A function that checks if a quadratic expression can be factored into two binomials with integer coefficients -/
def is_factorable (b : ℤ) : Prop :=
  ∃ (r s : ℤ), ∀ (x : ℤ), x^2 + b*x + 1200 = (x + r) * (x + s)

/-- Theorem stating that 70 is the smallest positive integer b for which x^2 + bx + 1200 can be factored into two binomials with integer coefficients -/
theorem smallest_factorable_b :
  (∀ b : ℤ, 0 < b → b < 70 → ¬ is_factorable b) ∧ is_factorable 70 := by
  sorry

#check smallest_factorable_b

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_factorable_b_l56_5638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_three_similar_piles_l56_5675

theorem impossible_three_similar_piles :
  ¬ ∃ (a b c : ℝ), 
    (a > 0 ∧ b > 0 ∧ c > 0) ∧
    (a ≤ Real.sqrt 2 * b ∧ b ≤ Real.sqrt 2 * a) ∧
    (a ≤ Real.sqrt 2 * c ∧ c ≤ Real.sqrt 2 * a) ∧
    (b ≤ Real.sqrt 2 * c ∧ c ≤ Real.sqrt 2 * b) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_three_similar_piles_l56_5675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_odd_divisor_15_factorial_l56_5672

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def divisors (n : ℕ) : Finset ℕ :=
  Finset.filter (λ d => n % d = 0) (Finset.range (n + 1))

def is_odd (n : ℕ) : Bool := n % 2 = 1

theorem prob_odd_divisor_15_factorial :
  let n := factorial 15
  let all_divisors := divisors n
  let odd_divisors := all_divisors.filter (λ x => x % 2 = 1)
  (odd_divisors.card : ℚ) / all_divisors.card = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_odd_divisor_15_factorial_l56_5672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_unit_vectors_l56_5601

def a : Fin 2 → ℝ := ![5, 12]

theorem perpendicular_unit_vectors :
  let b₁ : Fin 2 → ℝ := ![-12/13, 5/13]
  let b₂ : Fin 2 → ℝ := ![12/13, -5/13]
  (∀ i, (a i) * (b₁ i) + (a (i.succ)) * (b₁ (i.succ)) = 0) ∧
  (∀ i, (a i) * (b₂ i) + (a (i.succ)) * (b₂ (i.succ)) = 0) ∧
  (b₁ 0)^2 + (b₁ 1)^2 = 1 ∧
  (b₂ 0)^2 + (b₂ 1)^2 = 1 := by
  sorry

#check perpendicular_unit_vectors

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_unit_vectors_l56_5601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_faces_area_theorem_l56_5689

/-- The total area of the four triangular faces of a right, square-based pyramid -/
noncomputable def pyramidFacesArea (baseEdge : ℝ) (lateralEdge : ℝ) : ℝ :=
  4 * (1 / 2 * baseEdge * Real.sqrt (lateralEdge ^ 2 - (baseEdge / 2) ^ 2))

/-- Theorem: The total area of the four triangular faces of a right, square-based pyramid
    with base edges of 8 units and lateral edges of 7 units is equal to 16√33 square units -/
theorem pyramid_faces_area_theorem :
  pyramidFacesArea 8 7 = 16 * Real.sqrt 33 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_faces_area_theorem_l56_5689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_trig_l56_5677

theorem right_triangle_trig (AB BC : ℝ) (h1 : AB = 15) (h2 : BC = 20) :
  let AC : ℝ := Real.sqrt (AB^2 + BC^2)
  Real.cos (Real.arcsin (BC / AC)) = 3/5 ∧ Real.sin (Real.arccos (AB / AC)) = 3/5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_trig_l56_5677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_number_problem_l56_5643

/-- Given a real number x satisfying (5/7)x + 123 = 984, 
    prove that 0.7396x - 45 is approximately 844.85 -/
theorem johns_number_problem (x : ℝ) (h : (5/7)*x + 123 = 984) : 
  abs (0.7396*x - 45 - 844.85) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_number_problem_l56_5643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dichromate_mass_percentage_single_cr_closer_to_target_l56_5667

/-- Represents the mass percentage of an element in a compound -/
noncomputable def MassPercentage (elementMass : ℝ) (totalMass : ℝ) : ℝ :=
  (elementMass / totalMass) * 100

/-- The atomic mass of Chromium in g/mol -/
def CrMass : ℝ := 52.00

/-- The atomic mass of Oxygen in g/mol -/
def OMass : ℝ := 16.00

/-- The total mass of Dichromate (Cr2O7^2-) in g/mol -/
def DichromateMass : ℝ := 2 * CrMass + 7 * OMass

/-- Theorem stating that neither Chromium nor Oxygen in Dichromate has a mass percentage of exactly 27.03% -/
theorem dichromate_mass_percentage :
  MassPercentage (2 * CrMass) DichromateMass ≠ 27.03 ∧
  MassPercentage (7 * OMass) DichromateMass ≠ 27.03 := by
  sorry

/-- Theorem stating that a single Chromium atom in Dichromate has a mass percentage closer to 27.03% than Oxygen -/
theorem single_cr_closer_to_target :
  |MassPercentage CrMass DichromateMass - 27.03| <
  |MassPercentage (7 * OMass) DichromateMass - 27.03| := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dichromate_mass_percentage_single_cr_closer_to_target_l56_5667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eliza_ironing_time_l56_5676

/-- Represents the time in hours Eliza spent ironing blouses -/
noncomputable def blouse_hours : ℝ := 2

/-- Represents the number of blouses Eliza can iron in one hour -/
noncomputable def blouses_per_hour : ℝ := 60 / 15

/-- Represents the number of dresses Eliza can iron in one hour -/
noncomputable def dresses_per_hour : ℝ := 60 / 20

/-- Represents the number of hours Eliza spent ironing dresses -/
noncomputable def dress_hours : ℝ := 3

/-- Represents the total number of clothes Eliza ironed -/
def total_clothes : ℕ := 17

theorem eliza_ironing_time : 
  blouse_hours * blouses_per_hour + dress_hours * dresses_per_hour = total_clothes ∧ 
  blouse_hours = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eliza_ironing_time_l56_5676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_bound_l56_5611

noncomputable def a (n : ℕ) : ℝ :=
  match n with
  | 0 => 1  -- We use 0-based indexing in Lean, so a₁ corresponds to a 0
  | n + 1 => a n + 1 / a n

theorem sequence_bound : 14 < a 99 ∧ a 99 < 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_bound_l56_5611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_opposite_parts_l56_5615

theorem complex_opposite_parts (b : ℝ) : 
  (Complex.re ((2 - b * Complex.I) * Complex.I) = 
   -Complex.im ((2 - b * Complex.I) * Complex.I)) → b = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_opposite_parts_l56_5615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_quadrants_l56_5691

noncomputable def inverse_proportion (x : ℝ) : ℝ := 2 / x

def in_first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

def in_third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

theorem inverse_proportion_quadrants :
  ∀ x : ℝ, x ≠ 0 →
    (in_first_quadrant x (inverse_proportion x) ∨ 
     in_third_quadrant x (inverse_proportion x)) ∧
    ¬(x > 0 ∧ inverse_proportion x < 0) ∧
    ¬(x < 0 ∧ inverse_proportion x > 0) := by
  sorry

#check inverse_proportion_quadrants

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_quadrants_l56_5691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_simplification_l56_5614

theorem problem_simplification :
  (Real.sqrt 8 / Real.sqrt 2 + (Real.sqrt 5 + 3) * (Real.sqrt 5 - 3) = -2) ∧
  (Real.sqrt 48 / Real.sqrt 3 - Real.sqrt (1/2) * Real.sqrt 12 + Real.sqrt 24 = 4 + Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_simplification_l56_5614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_l56_5613

noncomputable section

/-- Definition of the ellipse C -/
def ellipse_C (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

/-- Definition of eccentricity -/
def eccentricity (a c : ℝ) : ℝ := c / a

/-- Definition of the point M -/
def point_M : ℝ × ℝ := (-3, 0)

/-- Definition of the condition MF₂ = 2MF₁ -/
def MF_condition (F₁ F₂ : ℝ × ℝ) : Prop :=
  let M := point_M
  (F₂.1 - M.1, F₂.2 - M.2) = (2 * (F₁.1 - M.1), 2 * (F₁.2 - M.2))

/-- Definition of line l -/
def line_l (k m : ℝ) (x : ℝ) : ℝ := k * x + m

/-- Definition of points A and B -/
def point_A (k m : ℝ) : ℝ × ℝ := (2, line_l k m 2)
def point_B (k m : ℝ) : ℝ × ℝ := (-2, line_l k m (-2))

/-- Definition of the condition F₂A · F₂B = 0 -/
def F₂AB_condition (F₂ : ℝ × ℝ) (k m : ℝ) : Prop :=
  let A := point_A k m
  let B := point_B k m
  (A.1 - F₂.1) * (B.1 - F₂.1) + (A.2 - F₂.2) * (B.2 - F₂.2) = 0

/-- The main theorem -/
theorem ellipse_line_intersection
  (a b c : ℝ)
  (F₁ F₂ : ℝ × ℝ)
  (k m : ℝ)
  (h_ellipse : ∀ x y, ellipse_C x y a b)
  (h_eccentricity : eccentricity a c = 1/2)
  (h_MF : MF_condition F₁ F₂)
  (h_F₂AB : F₂AB_condition F₂ k m) :
  ∃! p : ℝ × ℝ, ellipse_C p.1 p.2 a b ∧ p.2 = line_l k m p.1 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_l56_5613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_l56_5633

-- Define the trapezoid ABCD and points X and Y
structure Trapezoid where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  X : ℝ × ℝ
  Y : ℝ × ℝ

-- Define the properties of the trapezoid
def isIsoscelesTrapezoid (t : Trapezoid) : Prop :=
  (t.A.1 = t.D.1) ∧ (t.B.1 = t.C.1) ∧ (t.A.2 = t.B.2) ∧ (t.C.2 = t.D.2)

def isParallel (l1 l2 : (ℝ × ℝ) × (ℝ × ℝ)) : Prop :=
  (l1.1.2 - l1.2.2) / (l1.1.1 - l1.2.1) = (l2.1.2 - l2.2.2) / (l2.1.1 - l2.2.1)

def isRightAngle (p1 p2 p3 : ℝ × ℝ) : Prop :=
  (p2.1 - p1.1) * (p3.1 - p2.1) + (p2.2 - p1.2) * (p3.2 - p2.2) = 0

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem trapezoid_area (t : Trapezoid) :
  isIsoscelesTrapezoid t →
  isParallel (t.B, t.C) (t.A, t.D) →
  distance t.A t.B = distance t.C t.D →
  t.X.1 > t.A.1 ∧ t.X.1 < t.Y.1 ∧ t.Y.1 < t.C.1 →
  isRightAngle t.A t.X t.D →
  isRightAngle t.B t.Y t.C →
  distance t.A t.X = 5 →
  distance t.X t.Y = 3 →
  distance t.Y t.C = 4 →
  let area := (t.C.1 - t.A.1) * (t.B.2 - t.D.2) / 2
  area = 26 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_l56_5633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pieces_after_cuts_l56_5623

/-- Represents a rectangle on a square sheet of paper -/
structure Rectangle where
  -- We don't need to define the exact properties of a rectangle here
  -- as the problem doesn't require specific dimensions

/-- Represents a square sheet of paper with rectangles drawn on it -/
structure Sheet where
  rectangles : List Rectangle
  no_overlap : ∀ r1 r2, r1 ∈ rectangles → r2 ∈ rectangles → r1 ≠ r2 → 
    True -- Placeholder for the non-overlap condition

/-- Helper function to calculate the maximum number of pieces -/
def max_num_pieces_after_cuts (sheet : Sheet) : Nat :=
  sorry -- Implementation not required for the theorem statement

/-- 
Theorem: For a square sheet of paper with n non-overlapping rectangles 
drawn with sides parallel to the sheet's sides, the maximum number of 
pieces formed after cutting out these rectangles is n+1.
-/
theorem max_pieces_after_cuts (sheet : Sheet) : 
  sheet.rectangles.length + 1 ≥ max_num_pieces_after_cuts sheet := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pieces_after_cuts_l56_5623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lemming_average_distance_l56_5625

/-- The side length of the square --/
noncomputable def side_length : ℝ := 10

/-- The distance the lemming moves along the diagonal --/
noncomputable def diagonal_move : ℝ := 6.2

/-- The distance the lemming moves after turning --/
noncomputable def perpendicular_move : ℝ := 2

/-- The lemming's final position after movement --/
noncomputable def lemming_position : ℝ × ℝ :=
  let diag_fraction := diagonal_move / (side_length * Real.sqrt 2)
  let diag_coord := diag_fraction * side_length
  (diag_coord + perpendicular_move, diag_coord)

/-- The shortest distances from the lemming to each side of the square --/
noncomputable def distances_to_sides : List ℝ :=
  let (x, y) := lemming_position
  [x, y, side_length - x, side_length - y]

/-- The average of the shortest distances from the lemming to each side of the square --/
noncomputable def average_distance : ℝ :=
  (distances_to_sides.sum) / 4

theorem lemming_average_distance :
  average_distance = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lemming_average_distance_l56_5625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_PA_max_min_values_l56_5640

noncomputable def curve_C (θ : ℝ) : ℝ × ℝ := (3 * Real.cos θ, 2 * Real.sin θ)

def line_l (t : ℝ) : ℝ × ℝ := (2 + t, 2 - 2 * t)

noncomputable def point_P (θ : ℝ) : ℝ × ℝ := curve_C θ

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def angle_between_lines (l1 l2 : ℝ → ℝ × ℝ) : ℝ → ℝ := sorry

def intersection_point (l1 l2 : ℝ → ℝ × ℝ) : ℝ × ℝ := sorry

def line_through_P (θ : ℝ) : ℝ → ℝ × ℝ := sorry

theorem PA_max_min_values :
  let max_PA := (6 * Real.sqrt 10 + 20) / 5
  let min_PA := (20 - 6 * Real.sqrt 10) / 5
  ∀ θ : ℝ,
    let P := point_P θ
    let l_P := line_through_P θ
    let A := intersection_point l_P line_l
    let PA := distance P A
    (PA ≤ max_PA) ∧ (min_PA ≤ PA) ∧
    (∃ θ₁ θ₂ : ℝ, distance (point_P θ₁) (intersection_point (line_through_P θ₁) line_l) = max_PA ∧
                  distance (point_P θ₂) (intersection_point (line_through_P θ₂) line_l) = min_PA) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_PA_max_min_values_l56_5640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_break_even_price_is_20_l56_5612

-- Define the given constants
def initial_investment : ℚ := 10410
def manufacturing_cost_per_game : ℚ := 2.65
def games_to_break_even : ℕ := 600

-- Define the break-even selling price
noncomputable def break_even_price : ℚ := (initial_investment + manufacturing_cost_per_game * games_to_break_even) / games_to_break_even

-- Theorem statement
theorem break_even_price_is_20 : break_even_price = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_break_even_price_is_20_l56_5612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_lines_l56_5659

-- Define the triangle ABC
def A : ℝ × ℝ := (-3, 0)
def B : ℝ × ℝ := (2, 1)
def C : ℝ × ℝ := (-2, 3)

-- Define the equations of lines
def line_BC (x y : ℝ) : Prop := x + 2*y - 4 = 0
def median_AD (x y : ℝ) : Prop := 2*x - 3*y + 6 = 0
def perp_bisector_DE (x y : ℝ) : Prop := y = 2*x + 2

-- Helper function to create a line through two points
def line_through (p1 p2 : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, p = (1 - t) • p1 + t • p2}

-- Helper function for perpendicular bisector
def perp_bisector (p1 p2 : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - (p1.1 + p2.1)/2)^2 + (p.2 - (p1.2 + p2.2)/2)^2 = 
               ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)/4}

-- Theorem stating the equations of the lines
theorem triangle_lines :
  (∀ x y, line_BC x y ↔ (x, y) ∈ line_through B C) ∧
  (∀ x y, median_AD x y ↔ (x, y) ∈ line_through A ((B.1 + C.1)/2, (B.2 + C.2)/2)) ∧
  (∀ x y, perp_bisector_DE x y ↔ (x, y) ∈ perp_bisector B C) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_lines_l56_5659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_T_l56_5636

/-- The absolute value of T, where T = (1+√3i)^19 - (1-√3i)^19 and i = √(-1) -/
theorem abs_T : 
  let i : ℂ := Complex.I
  let T : ℂ := (1 + Real.sqrt 3 * i)^19 - (1 - Real.sqrt 3 * i)^19
  Complex.abs T = 2^19 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_T_l56_5636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_dot_product_l56_5606

/-- The ellipse with equation x²/4 + y²/3 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 / 3 = 1}

/-- The left focus of the ellipse -/
def F₁ : ℝ × ℝ := (-1, 0)

/-- The right focus of the ellipse -/
def F₂ : ℝ × ℝ := (1, 0)

/-- The incircle radius of triangle PF₁F₂ -/
noncomputable def incircle_radius (P : ℝ × ℝ) : ℝ := 1/2

/-- The dot product of vectors PF₁ and PF₂ -/
noncomputable def dot_product (P : ℝ × ℝ) : ℝ :=
  (F₁.1 - P.1) * (F₂.1 - P.1) + (F₁.2 - P.2) * (F₂.2 - P.2)

theorem ellipse_dot_product (P : ℝ × ℝ) 
  (h₁ : P ∈ Ellipse) 
  (h₂ : incircle_radius P = 1/2) : 
  dot_product P = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_dot_product_l56_5606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_proof_l56_5681

theorem problem_proof (x y : ℝ) 
  (h1 : (4 : ℝ) ^ x = (16 : ℝ) ^ (y + 2))
  (h2 : (16 : ℝ) ^ y = (4 : ℝ) ^ (x - 4)) : 
  x + y = 4 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_proof_l56_5681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_to_line_l56_5674

-- Define the curve C
noncomputable def curve_C (θ : ℝ) : ℝ × ℝ :=
  (8 * (Real.tan θ) ^ 2, 8 * Real.tan θ)

-- Define the line l
def line_l (x y : ℝ) : Prop :=
  x + y + 8 = 0

-- Define the distance function between a point on C and line l
noncomputable def distance_to_line (θ : ℝ) : ℝ :=
  let (x, y) := curve_C θ
  |x + y + 8| / Real.sqrt 2

-- Statement to prove
theorem min_distance_curve_to_line :
  ∃ (min_dist : ℝ), min_dist = 3 * Real.sqrt 2 ∧
  ∀ θ, -π/2 < θ ∧ θ < π/2 → distance_to_line θ ≥ min_dist := by
  sorry

#check min_distance_curve_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_to_line_l56_5674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_semicircle_area_l56_5653

/-- The area of a figure formed by rotating a semicircle of radius R by angle α -/
noncomputable def area_of_rotated_semicircle (R α : ℝ) : ℝ := 
  (2 * R)^2 * α / 2

/-- The area of a figure formed by rotating a semicircle around one of its ends by 45° -/
theorem rotated_semicircle_area (R : ℝ) (h : R > 0) : 
  let α : ℝ := 45 * π / 180  -- 45° in radians
  area_of_rotated_semicircle R α = π * R^2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_semicircle_area_l56_5653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_sum_less_than_n_l56_5670

theorem gcd_sum_less_than_n (n m : ℕ) (h_n : n > 1) (h_m : m > 1) 
  (a : Fin m → ℕ) (h_a : ∀ i, a i ≤ n^m) :
  ∃ b : Fin m → ℕ, (∀ i, 0 < b i ∧ b i ≤ n) ∧ 
  Nat.gcd (Finset.univ.sum (λ i ↦ a i + b i)) < n := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_sum_less_than_n_l56_5670
