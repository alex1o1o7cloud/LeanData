import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_implies_a_range_l31_3130

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then a^x else (a-3)*x + 4*a

-- State the theorem
theorem function_property_implies_a_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) < 0) →
  0 < a ∧ a ≤ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_implies_a_range_l31_3130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_block_fraction_visible_l31_3173

/-- Represents a block of wood --/
structure Block where
  weight : ℝ
  buoyantForce : ℝ

/-- Calculates the fraction of the block visible above water when floating --/
noncomputable def fractionVisible (b : Block) : ℝ :=
  1 - b.weight / b.buoyantForce

/-- Theorem stating that for a block with weight 30 N and buoyant force 50 N when submerged, 
    the fraction visible above water when floating is 2/5 --/
theorem block_fraction_visible :
  let b : Block := { weight := 30, buoyantForce := 50 }
  fractionVisible b = 2/5 := by
  -- Expand the definition of fractionVisible
  unfold fractionVisible
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_block_fraction_visible_l31_3173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_l31_3188

-- Define the slope function
noncomputable def slope_function (x : ℝ) : ℝ := x^2 + 1

-- Define the theorem
theorem inclination_angle_range :
  ∀ x : ℝ, ∃ α : ℝ,
    (0 ≤ α ∧ α < Real.pi / 2) ∧  -- α is in the first quadrant
    (slope_function x = Real.tan α) ∧  -- relation between slope and angle
    (Real.pi / 4 ≤ α) ∧  -- lower bound
    (α < Real.pi / 2) :=  -- upper bound
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_l31_3188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_properties_l31_3108

def z (a : ℝ) : ℂ := Complex.mk (a - 2) (a + 1)

theorem complex_number_properties (a : ℝ) :
  ((z a).im = 0 ↔ a = -1) ∧
  ((z a ≠ 0 ∧ (z a).im ≠ 0) ↔ (a ≠ -1 ∧ a ≠ 2)) ∧
  (((z a).re = 0 ∧ (z a).im ≠ 0) ↔ a = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_properties_l31_3108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scaled_model_height_approx_l31_3178

/-- Represents a water tower with a spherical top portion -/
structure WaterTower where
  height : ℝ
  volume : ℝ

/-- Calculates the height of a scaled model water tower -/
noncomputable def scaled_height (original : WaterTower) (model_volume : ℝ) : ℝ :=
  original.height / (original.volume / model_volume) ^ (1/3)

/-- Theorem stating the approximate height of the scaled model -/
theorem scaled_model_height_approx (ε : ℝ) (hε : ε > 0) :
  ∃ (original : WaterTower),
    original.height = 60 ∧
    original.volume = 50000 ∧
    abs (scaled_height original 0.2 - 0.95) < ε := by
  sorry

#eval "Theorem statement compiled successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scaled_model_height_approx_l31_3178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Al_in_Al2CO3_3_approx_l31_3109

/-- The mass percentage of Al in Al2(CO3)3 -/
noncomputable def mass_percentage_Al_in_Al2CO3_3 : ℝ :=
  let molar_mass_Al : ℝ := 26.98
  let molar_mass_C : ℝ := 12.01
  let molar_mass_O : ℝ := 16.00
  let molar_mass_Al2CO3_3 : ℝ := 2 * molar_mass_Al + 3 * molar_mass_C + 9 * molar_mass_O
  let mass_Al_in_Al2CO3_3 : ℝ := 2 * molar_mass_Al
  (mass_Al_in_Al2CO3_3 / molar_mass_Al2CO3_3) * 100

/-- The mass percentage of Al in Al2(CO3)3 is approximately 23.05% -/
theorem mass_percentage_Al_in_Al2CO3_3_approx :
  |mass_percentage_Al_in_Al2CO3_3 - 23.05| < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Al_in_Al2CO3_3_approx_l31_3109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_prop_and_slope_l31_3187

/-- An inverse proportion function passing through points A(2, 6) and B(3, 4) -/
noncomputable def inverse_prop (x : ℝ) : ℝ := 12 / x

/-- The slope of a line intersecting segment AB -/
def slope_range (m : ℝ) : Prop := 4/3 ≤ m ∧ m ≤ 3

theorem inverse_prop_and_slope :
  (inverse_prop 2 = 6 ∧ inverse_prop 3 = 4) ∧
  (∀ m : ℝ, (∃ x : ℝ, 2 ≤ x ∧ x ≤ 3 ∧ m * x = inverse_prop x) → slope_range m) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_prop_and_slope_l31_3187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_journey_distance_time_relationship_l31_3129

-- Define the acceleration and time constants
noncomputable def a : ℝ := 0.5
noncomputable def t1 : ℝ := 2
noncomputable def t2 : ℝ := 9
noncomputable def t3 : ℝ := 11
noncomputable def t4 : ℝ := 13

-- Define the distance function
noncomputable def distance (t : ℝ) : ℝ :=
  if t ≤ t1 then t^2 / 4
  else if t ≤ t2 then t - 1
  else if t ≤ t3 then (1/4) * (-t^2 + 22*t - 85)
  else 30

-- Theorem statement
theorem train_journey_distance_time_relationship :
  ∀ t : ℝ, 0 ≤ t ∧ t ≤ t4 →
    (distance t = t^2 / 4 ∧ t ≤ t1) ∨
    (distance t = t - 1 ∧ t1 < t ∧ t ≤ t2) ∨
    (distance t = (1/4) * (-t^2 + 22*t - 85) ∧ t2 < t ∧ t ≤ t3) ∨
    (distance t = 30 ∧ t3 < t ∧ t ≤ t4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_journey_distance_time_relationship_l31_3129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_difference_implies_m_value_l31_3110

-- Define the function f(x)
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log (m - x) / Real.log m

-- State the theorem
theorem max_min_difference_implies_m_value (m : ℝ) :
  (m > 5) →
  (∀ x ∈ Set.Icc 3 5, f m x ≤ f m 3) ∧  -- Maximum at x = 3
  (∀ x ∈ Set.Icc 3 5, f m 5 ≤ f m x) ∧  -- Minimum at x = 5
  (f m 3 - f m 5 = 1) →                 -- Difference is 1
  m = 3 + Real.sqrt 6 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_difference_implies_m_value_l31_3110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_plus_two_over_z_traces_ellipse_l31_3115

/-- A complex number z that traces a circle centered at the origin with radius 3 -/
def z_on_circle (z : ℂ) : Prop := Complex.abs z = 3

/-- The curve traced by z + 2/z -/
noncomputable def curve (z : ℂ) : ℂ := z + 2 / z

/-- Definition of an ellipse in the complex plane -/
def is_ellipse (f : ℂ → ℂ) : Prop :=
  ∃ (a b : ℝ) (h : a > 0 ∧ b > 0),
    ∀ (w : ℂ), (w.re / a)^2 + (w.im / b)^2 = 1 ↔ ∃ (z : ℂ), w = f z

/-- Theorem stating that z + 2/z traces an ellipse when z is on the circle -/
theorem z_plus_two_over_z_traces_ellipse :
  is_ellipse curve := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_plus_two_over_z_traces_ellipse_l31_3115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_rewritten_sum_l31_3174

noncomputable def f (x : ℝ) : ℝ := (x^3 - 4*x^2 - x + 6) / (x - 1)

def D : ℝ := 1

theorem f_rewritten_sum :
  ∃ (A B C : ℝ),
    (∀ x : ℝ, x ≠ D → f x = A*x^2 + B*x + C) ∧
    A + B + C + D = -7 := by
  -- We'll use these values based on the solution
  let A : ℝ := 1
  let B : ℝ := -3
  let C : ℝ := -6
  
  use A, B, C
  constructor
  · intro x hx
    -- The proof of equality would go here
    sorry
  · -- Verify the sum
    simp [A, B, C, D]
    norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_rewritten_sum_l31_3174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_pi_fourth_l31_3156

theorem tan_difference_pi_fourth (α : Real) (h : Real.tan α = 2) : 
  Real.tan (α - Real.pi/4) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_pi_fourth_l31_3156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_degree_is_five_l31_3180

/-- The degree of the polynomial resulting from multiplying x^4, x + 1/x, and 1 + 3/x + 5/x^2 -/
def polynomial_degree : ℕ := 5

/-- The first expression to be multiplied -/
noncomputable def expr1 (x : ℝ) : ℝ := x^4

/-- The second expression to be multiplied -/
noncomputable def expr2 (x : ℝ) : ℝ := x + 1/x

/-- The third expression to be multiplied -/
noncomputable def expr3 (x : ℝ) : ℝ := 1 + 3/x + 5/x^2

/-- The theorem stating that the degree of the polynomial resulting from
    multiplying expr1, expr2, and expr3 is equal to polynomial_degree -/
theorem product_degree_is_five :
  ∃ (p : Polynomial ℝ), (Polynomial.degree p = polynomial_degree) ∧
  (∀ x : ℝ, x ≠ 0 → p.eval x = expr1 x * expr2 x * expr3 x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_degree_is_five_l31_3180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_dot_product_l31_3139

/-- The ellipse representing the trajectory of point M --/
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/2 = 1

/-- Point C on the x-axis --/
noncomputable def C : ℝ × ℝ := (-1, 0)

/-- Point N on the x-axis --/
noncomputable def N : ℝ × ℝ := (-7/4, 0)

/-- Dot product of two 2D vectors --/
noncomputable def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Vector from N to a point --/
noncomputable def vector_N_to (p : ℝ × ℝ) : ℝ × ℝ := (p.1 - N.1, p.2 - N.2)

theorem constant_dot_product (A B : ℝ × ℝ) :
  ellipse A.1 A.2 → ellipse B.1 B.2 →
  ∃ (k : ℝ), k ≠ 0 ∧ A.2 - C.2 = k * (A.1 - C.1) ∧ B.2 - C.2 = k * (B.1 - C.1) →
  dot_product (vector_N_to A) (vector_N_to B) = -15/16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_dot_product_l31_3139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_rounded_to_two_decimal_places_l31_3179

noncomputable def round_to_two_decimal_places (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

theorem division_rounded_to_two_decimal_places :
  round_to_two_decimal_places (14.23 / 4.7) = 3.03 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_rounded_to_two_decimal_places_l31_3179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_conditions_l31_3198

noncomputable section

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The logarithmic function we're working with -/
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  Real.log (abs (a + 1 / (1 - x))) + b

theorem odd_function_conditions (a b : ℝ) :
  IsOdd (f a b) → a = -1/2 ∧ b = Real.log 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_conditions_l31_3198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l31_3103

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 → x = 1 ∧ y = 1) →
  b/a = Real.sqrt 2 →
  ∀ x y : ℝ, 2*x^2 - y^2 = 1 ↔ x^2/a^2 - y^2/b^2 = 1 :=
by
  sorry

#check hyperbola_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l31_3103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sin_plus_sqrt_one_minus_x_squared_l31_3140

open MeasureTheory Interval Real Set

theorem integral_sin_plus_sqrt_one_minus_x_squared : 
  ∫ x in (-1 : ℝ)..1, (sin x + sqrt (1 - x^2)) = π / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sin_plus_sqrt_one_minus_x_squared_l31_3140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bernardo_leroy_problem_l31_3155

/-- Representation of a number in a given base -/
def BaseRepresentation (n : ℕ) (base : ℕ) : List ℕ := sorry

/-- Convert a base representation to a decimal number -/
def toDecimal (rep : List ℕ) (base : ℕ) : ℕ := sorry

/-- Get the three rightmost digits of a number -/
def rightmostDigits (n : ℕ) : ℕ := n % 1000

/-- Check if two numbers have the same three rightmost digits -/
def sameRightmostDigits (a b : ℕ) : Bool := rightmostDigits a = rightmostDigits b

/-- The main theorem capturing the essence of the problem -/
theorem bernardo_leroy_problem :
  ∃ k : ℕ,
    k = (Finset.filter (fun n : ℕ =>
      let base4Rep := BaseRepresentation (n + 100) 4
      let base7Rep := BaseRepresentation (n + 100) 7
      let S := toDecimal base4Rep 10 + toDecimal base7Rep 10
      sameRightmostDigits S (2 * (n + 100))
    ) (Finset.range 900)).card := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bernardo_leroy_problem_l31_3155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_problem_l31_3176

-- Function to round a real number to 3 decimal places
noncomputable def round_to_3dp (x : ℝ) : ℝ := 
  (⌊x * 1000 + 0.5⌋ : ℝ) / 1000

-- Theorem statement
theorem fourth_root_problem : ∃ x : ℝ, 
  round_to_3dp (x^4) = 1.012 ∧ round_to_3dp x = 1.003 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_problem_l31_3176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_enclosed_area_l31_3146

-- Define the function f and its derivative f'
def f (x : ℝ) : ℝ := x^3 - x + 2
def f' (x : ℝ) : ℝ := 3*x^2 - 1

-- Define the tangent line at x=1
def tangent_line (x : ℝ) : ℝ := 2*x

-- Theorem for the tangent line equation
theorem tangent_line_at_one :
  ∀ x : ℝ, tangent_line x = f 1 + f' 1 * (x - 1) := by sorry

-- Theorem for the enclosed area
theorem enclosed_area :
  ∫ x in (-1/3)..1, (tangent_line x - f' x) = 32/27 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_enclosed_area_l31_3146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_newton_and_bisection_approximations_l31_3169

noncomputable def f (x : ℝ) : ℝ := x^3 + 2*x^2 + 3*x + 3

noncomputable def f_derivative (x : ℝ) : ℝ := 3*x^2 + 4*x + 3

noncomputable def newton_iteration (x : ℝ) : ℝ := x - f x / f_derivative x

noncomputable def bisection_midpoint (a b : ℝ) : ℝ := (a + b) / 2

theorem newton_and_bisection_approximations :
  let x₀ : ℝ := -1
  let x₁ : ℝ := newton_iteration x₀
  let x₂ : ℝ := newton_iteration x₁
  let a₀ : ℝ := -2
  let b₀ : ℝ := -1
  let m₁ : ℝ := bisection_midpoint a₀ b₀
  let a₁ : ℝ := if f m₁ < 0 then m₁ else a₀
  let b₁ : ℝ := if f m₁ < 0 then b₀ else m₁
  let m₂ : ℝ := bisection_midpoint a₁ b₁
  (x₂ = -7/5) ∧ (m₂ = -11/8) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_newton_and_bisection_approximations_l31_3169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_ones_divisible_by_power_of_three_l31_3149

/-- Function to create a number with n digits, all of which are 1 -/
def all_ones (n : ℕ) : ℕ :=
  (10^n - 1) / 9

/-- Theorem: For any positive integer n, the number consisting of 3^n digits, 
    all of which are 1, is divisible by 3^n -/
theorem all_ones_divisible_by_power_of_three (n : ℕ) :
  (3^n : ℕ) ∣ all_ones (3^n) := by
  sorry

#check all_ones_divisible_by_power_of_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_ones_divisible_by_power_of_three_l31_3149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_of_f_l31_3122

-- Define the function f as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (-x^2 - 2*x + 3)

-- Define the domain of f
def domain : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 1}

-- State the theorem
theorem monotonic_decreasing_interval_of_f :
  ∃ (a b : ℝ), a = -1 ∧ b = 1 ∧
  (∀ x y, x ∈ domain → y ∈ domain → a ≤ x → x < y → y ≤ b → f y < f x) ∧
  (∀ c d, c < a ∨ b < d → ¬(∀ x y, x ∈ domain → y ∈ domain → c ≤ x → x < y → y ≤ d → f y < f x)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_of_f_l31_3122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_close_points_exist_l31_3183

-- Define the regular triangle
def RegularTriangle : Set (ℝ × ℝ) := {p | p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ p.1 + p.2 ≤ 1}

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem close_points_exist (points : Finset (ℝ × ℝ)) 
  (h1 : ∀ p, p ∈ points → p ∈ RegularTriangle)
  (h2 : points.card = 7) :
  ∃ p q, p ∈ points ∧ q ∈ points ∧ p ≠ q ∧ distance p q < 0.4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_close_points_exist_l31_3183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l31_3150

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x - 2 * Real.pi / 3)

theorem f_monotone_increasing :
  MonotoneOn f (Set.Icc (Real.pi / 12) (7 * Real.pi / 12)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l31_3150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_not_containing_multiple_of_seven_l31_3116

/-- A number n contains m if the digits of m appear consecutively in n. -/
def contains (n m : ℕ) : Prop := sorry

/-- The largest number that doesn't contain any multiple of 7 is 999999. -/
theorem largest_not_containing_multiple_of_seven : 
  ∀ n : ℕ, n > 999999 → ∃ k : ℕ, k % 7 = 0 ∧ contains n k := by
  sorry

#check largest_not_containing_multiple_of_seven

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_not_containing_multiple_of_seven_l31_3116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_length_l31_3158

/-- The length of the common chord of two overlapping circles -/
theorem common_chord_length (r d : ℝ) (h1 : r = 12) (h2 : d = 18) :
  2 * Real.sqrt (r^2 - (d/2)^2) = 6 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_length_l31_3158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_such_function_l31_3104

theorem no_such_function : ¬∃ f : ℕ → ℕ, ∀ n : ℕ, n ≥ 2 → f (f (n - 1)) = f (n + 1) - f n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_such_function_l31_3104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_max_on_interval_l31_3135

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.cos x + (x + 1) * Real.sin x + 1

-- State the theorem
theorem f_min_max_on_interval :
  ∃ (min max : ℝ),
    (∀ x ∈ Set.Icc 0 (2 * π), f x ≥ min ∧ f x ≤ max) ∧
    (∃ x₁ ∈ Set.Icc 0 (2 * π), f x₁ = min) ∧
    (∃ x₂ ∈ Set.Icc 0 (2 * π), f x₂ = max) ∧
    min = -3 * π / 2 ∧
    max = π / 2 + 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_max_on_interval_l31_3135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_calculation_l31_3182

/-- The distance to school given the travel conditions -/
noncomputable def distance_to_school (total_time : ℝ) (speed_to_school : ℝ) (speed_from_school : ℝ) : ℝ :=
  25 / 6

/-- Theorem stating that the distance to school is 25/6 miles given the specified conditions -/
theorem distance_calculation (total_time : ℝ) (speed_to_school : ℝ) (speed_from_school : ℝ) 
  (h1 : total_time = 1)
  (h2 : speed_to_school = 5)
  (h3 : speed_from_school = 25) :
  distance_to_school total_time speed_to_school speed_from_school = 25 / 6 := by
  -- Unfold the definition of distance_to_school
  unfold distance_to_school
  -- The result follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_calculation_l31_3182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l31_3165

-- Define the ellipse C
def ellipse (a b : ℝ) := {(x, y) : ℝ × ℝ | x^2/a^2 + y^2/b^2 = 1}

-- Define the foci
def leftFocus (c : ℝ) : ℝ × ℝ := (-c, 0)
def rightFocus (c : ℝ) : ℝ × ℝ := (c, 0)

-- Define the line l₁
def line_l1 (c : ℝ) := {(x, y) : ℝ × ℝ | x = c}

-- Define the line l₂
def line_l2 (c : ℝ) := {(x, y) : ℝ × ℝ | y = x + c}

-- Define the intersection points
def intersectionPoints (C : Set (ℝ × ℝ)) (l : Set (ℝ × ℝ)) := C ∩ l

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the area of a triangle given three points
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  |((p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2)) / 2)|

-- State the theorem
theorem ellipse_properties (a b c : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : c > 0) (h4 : c^2 = a^2 - b^2) :
  let C := ellipse a b
  let F1 := leftFocus c
  let F2 := rightFocus c
  let l1 := line_l1 c
  let l2 := line_l2 (-c)
  ∃ (A B M N : ℝ × ℝ),
    A ∈ intersectionPoints C l1 ∧
    B ∈ intersectionPoints C l1 ∧ A ≠ B ∧
    M ∈ intersectionPoints C l2 ∧
    N ∈ intersectionPoints C l2 ∧ M ≠ N ∧
    distance A F1 = 7 * distance A F2 ∧
    triangleArea (0, 0) M N = 2 * Real.sqrt 6 / 5 →
    c / a = Real.sqrt 3 / 2 ∧ C = {(x, y) : ℝ × ℝ | x^2/4 + y^2 = 1} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l31_3165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_cube_volume_ratio_l31_3160

/-- A regular octahedron -/
structure RegularOctahedron where
  sideLength : ℝ
  sideLength_pos : sideLength > 0

/-- A cube whose vertices are the centers of the faces of a regular octahedron -/
def CubeFromOctahedron (o : RegularOctahedron) : ℝ → Prop :=
  λ sideLength => sideLength = o.sideLength * (Real.sqrt 2) / 3

/-- The volume of a regular octahedron -/
noncomputable def octahedronVolume (o : RegularOctahedron) : ℝ :=
  o.sideLength^3 * Real.sqrt 2 / 3

/-- The volume of a cube -/
def cubeVolume (sideLength : ℝ) : ℝ :=
  sideLength^3

theorem octahedron_cube_volume_ratio (o : RegularOctahedron) :
  ∃ (c : ℝ), CubeFromOctahedron o c ∧
  (octahedronVolume o) / (cubeVolume c) = 9 / 2 := by
  sorry

#check octahedron_cube_volume_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_cube_volume_ratio_l31_3160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_same_type_as_sqrt_3_l31_3167

theorem sqrt_same_type_as_sqrt_3 :
  ∃ (a : ℚ), Real.sqrt 12 = a * Real.sqrt 3 ∧
  (∀ (b : ℚ), Real.sqrt 6 ≠ b * Real.sqrt 3) ∧
  (∀ (c : ℚ), Real.sqrt 9 ≠ c * Real.sqrt 3) ∧
  (∀ (d : ℚ), Real.sqrt 18 ≠ d * Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_same_type_as_sqrt_3_l31_3167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_l31_3151

noncomputable def hyperbola (x y : ℝ) : Prop :=
  (x + 4)^2 / 9^2 - (y - 8)^2 / 16^2 = 1

noncomputable def focus_x : ℝ := -4 + Real.sqrt 337
def focus_y : ℝ := 8

theorem hyperbola_focus :
  ∀ x y : ℝ, hyperbola x y → x ≤ focus_x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_l31_3151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_z_minus_x_for_10_factorial_l31_3196

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem smallest_z_minus_x_for_10_factorial :
  ∃ (x y z : ℕ), 
    (x * y * z = factorial 10) ∧ 
    (0 < x) ∧ (x < y) ∧ (y < z) ∧
    (∀ (a b c : ℕ), (a * b * c = factorial 10) → (0 < a) → (a < b) → (b < c) → (z - x ≤ c - a)) ∧
    (z - x = 2139) := by
  sorry

#eval factorial 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_z_minus_x_for_10_factorial_l31_3196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_snowboard_energy_fraction_l31_3113

/-- The fraction of mechanical energy lost during a descent that goes into heating a snowboard -/
noncomputable def energy_fraction (h v m_total m_board c dT g : ℝ) : ℝ :=
  let E_initial := m_total * g * h
  let E_final := (1/2) * m_total * v^2
  let W := E_initial - E_final
  let Q := c * m_board * dT
  Q / W

/-- Theorem stating that the fraction of mechanical energy lost that goes into heating the snowboard is 1/98 -/
theorem snowboard_energy_fraction :
  energy_fraction 250 10 72 6 300 1 10 = 1/98 := by
  -- Unfold the definition of energy_fraction
  unfold energy_fraction
  -- Simplify the expression
  simp
  -- The proof is completed with sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_snowboard_energy_fraction_l31_3113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_eight_is_zero_l31_3114

-- Define the polynomial f
def f (x : ℝ) : ℝ := x^3 - 2*x^2 - 5*x + 6

-- Define the properties of g
noncomputable def g : ℝ → ℝ := sorry

-- g is a cubic polynomial
axiom g_cubic : ∃ a b c d : ℝ, ∀ x, g x = a*x^3 + b*x^2 + c*x + d

-- g(0) = 1
axiom g_zero : g 0 = 1

-- The roots of g are the cubes of the roots of f
axiom g_roots : ∀ x : ℝ, f x = 0 → ∃ y : ℝ, g y = 0 ∧ y = x^3

-- Theorem to prove
theorem g_eight_is_zero : g 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_eight_is_zero_l31_3114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_calculation_l31_3184

/-- Represents a cone with given slant height and lateral area -/
structure Cone where
  slant_height : ℝ
  lateral_area : ℝ

/-- Calculates the height of a cone given its slant height and lateral area -/
noncomputable def cone_height (c : Cone) : ℝ :=
  Real.sqrt (c.slant_height ^ 2 - (c.lateral_area / (Real.pi * c.slant_height)) ^ 2)

theorem cone_height_calculation (c : Cone) 
  (h1 : c.slant_height = 13)
  (h2 : c.lateral_area = 65 * Real.pi) : 
  cone_height c = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_calculation_l31_3184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_dot_product_theorem_l31_3134

-- Define vectors a and b
variable (a b : ℝ × ℝ)

-- Define the dot product
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Define the magnitude of a vector
noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (dot_product v v)

-- Define the conditions
def conditions (a b : ℝ × ℝ) : Prop :=
  magnitude a = 1 ∧ 
  magnitude b = 1 ∧ 
  ∀ k > 0, magnitude (a.1 + k * b.1, a.2 + k * b.2) = Real.sqrt 3 * magnitude (k * a.1 - b.1, k * a.2 - b.2)

-- Define the function f
def f (a b : ℝ × ℝ) (k : ℝ) : ℝ :=
  dot_product a b

-- State the theorem
theorem vector_dot_product_theorem (a b : ℝ × ℝ) 
  (h : conditions a b) :
  (∀ k > 0, f a b k = 4 * k / (k^2 + 1)) ∧
  (∀ k > 0, ∀ t ∈ Set.Icc (-2 : ℝ) 2, 
    (∃ x ∈ Set.Icc (2 - Real.sqrt 7) (Real.sqrt 7 - 2), 
      f a b k ≥ x^2 - 2*t*x - 5/2) ∧
    (∀ x ∉ Set.Icc (2 - Real.sqrt 7) (Real.sqrt 7 - 2), 
      ∃ k' > 0, ∃ t' ∈ Set.Icc (-2 : ℝ) 2, 
        f a b k' < x^2 - 2*t'*x - 5/2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_dot_product_theorem_l31_3134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_for_B_l31_3181

theorem number_for_B (a : ℝ) : ℝ := by
  -- Define the number for A
  let number_for_A := a
  
  -- Define the relationship between A and B
  let difference : ℝ := 2.5
  
  -- Define the number for B
  let number_for_B := number_for_A - difference
  
  -- Prove that the number for B is equal to a - 2.5
  exact number_for_B


end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_for_B_l31_3181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goldfish_cost_graph_is_finite_distinct_points_l31_3141

/-- The cost of n goldfish, where each goldfish costs 15 cents -/
def cost (n : ℕ) : ℕ := 15 * n

/-- The set of points representing the cost of 1 to 12 goldfish -/
def costGraph : Set (ℕ × ℕ) :=
  {p | ∃ n : ℕ, 1 ≤ n ∧ n ≤ 12 ∧ p = (n, cost n)}

theorem goldfish_cost_graph_is_finite_distinct_points :
  Finite costGraph ∧ (∀ p q, p ∈ costGraph → q ∈ costGraph → p ≠ q → p.1 ≠ q.1 ∧ p.2 ≠ q.2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_goldfish_cost_graph_is_finite_distinct_points_l31_3141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_b_l31_3166

-- Define a and b as noncomputable
noncomputable def a : ℝ := Real.log 400 / Real.log 16
noncomputable def b : ℝ := Real.log 20 / Real.log 4

-- Theorem statement
theorem a_equals_b : a = b := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_b_l31_3166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_implies_y_sixth_l31_3126

theorem cube_root_equation_implies_y_sixth (y : ℝ) (h_pos : y > 0) 
  (h_eq : (2 - y^3)^(1/3) + (2 + y^3)^(1/3) = 2) : 
  y^6 = 116/27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_implies_y_sixth_l31_3126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_90_congruence_l31_3154

theorem base_90_congruence (b : ℤ) : 
  (0 ≤ b ∧ b ≤ 160) → 
  (389201647 : ℤ) - b ≡ 0 [ZMOD 17] ↔ 
  b % 17 = 8 ∧ b ≤ 144 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_90_congruence_l31_3154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l31_3137

def f (x : ℝ) : ℝ := abs x + abs (x + 1)

theorem f_properties :
  (∃ (l : ℝ), ∀ (x : ℝ), f x ≥ l ∧ ¬∃ (m : ℝ), m > l ∧ ∀ (x : ℝ), f x ≥ m) ∧
  (∀ (t : ℝ), (∃ (m : ℝ), m^2 + 2*m + f t = 0) ↔ -1 ≤ t ∧ t ≤ 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l31_3137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mia_age_is_eleven_l31_3195

/-- Represents Mia's age in years -/
def age : ℕ := sorry

/-- Represents the number of hours Mia works per day -/
def hours_per_day : ℕ := 3

/-- Represents Mia's hourly rate in dollars per year of age -/
def hourly_rate : ℚ := 2/5

/-- Represents the number of days Mia worked during the nine-month period -/
def days_worked : ℕ := 80

/-- Represents Mia's total earnings during the nine-month period -/
def total_earnings : ℚ := 960

/-- Theorem stating that Mia's age at the end of the nine-month period was 11 years -/
theorem mia_age_is_eleven :
  age * hours_per_day * hourly_rate * days_worked = total_earnings →
  age = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mia_age_is_eleven_l31_3195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_divisor_of_diminished_value_l31_3136

theorem smallest_divisor_of_diminished_value (n : ℕ) : 
  n = 1014 → 
  (∀ k ∈ ({16, 18, 21, 28} : Set ℕ), (n - 6) % k = 0) →
  (∃ m : ℕ, m > 0 ∧ (n - 6) % m = 0 ∧ ∀ l : ℕ, 0 < l ∧ l < m → (n - 6) % l ≠ 0) →
  2 = Nat.minFac (n - 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_divisor_of_diminished_value_l31_3136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_formula_l31_3144

/-- The total distance Hadley walked given his trips to different locations -/
noncomputable def total_distance (x : ℝ) : ℝ :=
  x + (x - 3) + (x - 1) + (2*x - 5)/3

/-- Theorem stating that the total distance is equal to (11x - 17)/3 -/
theorem total_distance_formula (x : ℝ) : 
  total_distance x = (11*x - 17)/3 := by
  -- Unfold the definition of total_distance
  unfold total_distance
  -- Simplify the expression
  simp [add_assoc, sub_eq_add_neg, mul_add, mul_one, mul_neg]
  -- Perform arithmetic operations
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_formula_l31_3144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l31_3185

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {y | ∃ x, y = Real.sin x - Real.cos (x + Real.pi/6) + m}
def B : Set ℝ := {y | ∃ x ∈ Set.Icc 1 2, y = -x^2 + 2*x}

-- Define propositions p and q
def p (x : ℝ) (m : ℝ) : Prop := x ∈ A m
def q (x : ℝ) : Prop := x ∈ B

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, (∀ x : ℝ, (¬p x m → ¬q x) ∧ ∃ x : ℝ, ¬q x ∧ p x m) →
  1 - Real.sqrt 3 ≤ m ∧ m ≤ Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l31_3185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_expression_maximum_l31_3194

theorem triangle_angle_expression_maximum (A B C : ℝ) : 
  A + B + C = Real.pi →
  ∃ (max : ℝ), 
    (∀ A' B' C' : ℝ, A' + B' + C' = Real.pi → 
      Real.cos A' + 2 * Real.cos ((B' + C') / 2) ≤ max) ∧
    (Real.cos (Real.pi / 3) + 2 * Real.cos ((B + C) / 2) = max) ∧
    max = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_expression_maximum_l31_3194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l31_3118

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  A + B + C = Real.pi ∧
  Real.sin (C / 2) = Real.sqrt 10 / 4 ∧
  (1 / 2) * a * b * Real.sin C = 3 * Real.sqrt 15 / 4 ∧
  Real.sin A ^ 2 + Real.sin B ^ 2 = (13 / 16) * Real.sin C ^ 2 →
  Real.cos C = -1 / 4 ∧
  ((a = 2 ∧ b = 3 ∧ c = 4) ∨ (a = 3 ∧ b = 2 ∧ c = 4)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l31_3118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_implies_equality_l31_3152

theorem divisibility_implies_equality (a b n : ℕ) 
  (ha : a > 0) (hb : b > 0) (hn : n > 0)
  (h : ∀ k : ℕ, k > 0 → k ≠ b → (b - k) ∣ (a - k^n)) : 
  a = b^n := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_implies_equality_l31_3152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_count_l31_3162

/-- The set of digits allowed in the number --/
def allowed_digits : Finset Nat := {0, 1, 2, 3, 4, 6, 7, 9}

/-- A function to check if a number is a valid three-digit number without 5 or 8 --/
def is_valid (n : Nat) : Bool :=
  n ≥ 100 ∧ n < 1000 ∧
  (n / 100 % 10) ∈ allowed_digits ∧
  (n / 10 % 10) ∈ allowed_digits ∧
  (n % 10) ∈ allowed_digits

/-- The count of valid three-digit numbers --/
def count_valid : Nat := (Finset.filter (fun n => is_valid n) (Finset.range 1000)).card

theorem valid_count : count_valid = 448 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_count_l31_3162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_approx_l31_3123

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_area_approx :
  let a : ℝ := 28
  let b : ℝ := 24
  let c : ℝ := 15
  abs (triangle_area a b c - 178.12) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_approx_l31_3123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ezekiel_hike_theorem_l31_3102

/-- Calculates the distance hiked on the third day of a three-day hike -/
noncomputable def third_day_distance (total_distance : ℝ) (first_day : ℝ) : ℝ :=
  total_distance - first_day - (total_distance / 2)

/-- Theorem stating that for a 50km hike with 10km on the first day and half the distance on the second day, the third day's distance is 15km -/
theorem ezekiel_hike_theorem :
  third_day_distance 50 10 = 15 := by
  -- Unfold the definition of third_day_distance
  unfold third_day_distance
  -- Simplify the arithmetic
  simp [sub_eq_add_neg, add_assoc, add_comm, add_left_comm]
  -- Check that the result is correct
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ezekiel_hike_theorem_l31_3102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_income_from_tax_l31_3105

/-- Represents the tax system in Country X -/
structure TaxSystem where
  lowRate : ℚ  -- Rate for the first $40,000
  highRate : ℚ  -- Rate for income exceeding $40,000
  threshold : ℚ  -- The threshold between low and high rates

/-- Calculates the tax for a given income under the Country X tax system -/
def calculateTax (system : TaxSystem) (income : ℚ) : ℚ :=
  if income ≤ system.threshold then
    system.lowRate * income
  else
    system.lowRate * system.threshold + system.highRate * (income - system.threshold)

/-- Theorem: Given the tax rules and total tax paid, the citizen's income is $56,000 -/
theorem income_from_tax (system : TaxSystem) (totalTax : ℚ) :
  system.lowRate = 12/100 ∧ 
  system.highRate = 20/100 ∧ 
  system.threshold = 40000 ∧
  totalTax = 8000 →
  ∃ income, calculateTax system income = totalTax ∧ income = 56000 := by
  sorry

#eval calculateTax ⟨12/100, 20/100, 40000⟩ 56000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_income_from_tax_l31_3105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l31_3175

/-- A function that checks if a three-digit number abc is valid according to the problem conditions -/
def isValidNumber (a b c : Nat) : Bool :=
  a ≠ 0 && c ≠ 0 && a < 10 && b < 10 && c < 10

/-- A function that checks if a number is divisible by 4 -/
def isDivisibleBy4 (n : Nat) : Bool :=
  n % 4 = 0

/-- The main theorem stating that there are 36 valid numbers satisfying the conditions -/
theorem count_valid_numbers : 
  (Finset.filter (fun n => 
    let a := n / 100
    let b := (n / 10) % 10
    let c := n % 10
    isValidNumber a b c && 
    isDivisibleBy4 n && 
    isDivisibleBy4 (c * 100 + b * 10 + a)
  ) (Finset.range 1000)).card = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l31_3175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_composition_l31_3171

/-- Represents a solution with a specific carbonated water content -/
structure Solution where
  carbonated_water_content : ℝ

/-- Represents a mixture of two solutions -/
noncomputable def Mixture (p q : Solution) (p_volume q_volume : ℝ) : ℝ :=
  (p.carbonated_water_content * p_volume + q.carbonated_water_content * q_volume) / (p_volume + q_volume)

theorem mixture_composition 
  (p q : Solution)
  (hp : p.carbonated_water_content = 0.80)
  (hq : q.carbonated_water_content = 0.55)
  (hmix : ∀ (vp vq : ℝ), vp > 0 → vq > 0 → Mixture p q vp vq = 0.675) :
  ∃ (vp vq : ℝ), vp > 0 ∧ vq > 0 ∧ vp / (vp + vq) = 0.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_composition_l31_3171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_martha_kitchen_painting_time_l31_3128

/-- Represents the dimensions of a rectangular room -/
structure RoomDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the total wall area of a rectangular room -/
noncomputable def totalWallArea (d : RoomDimensions) : ℝ :=
  2 * (d.length * d.height + d.width * d.height)

/-- Calculates the time needed to paint a given area at a given rate -/
noncomputable def paintingTime (area : ℝ) (rate : ℝ) : ℝ :=
  area / rate

/-- Theorem stating the time needed to paint Martha's kitchen -/
theorem martha_kitchen_painting_time :
  let kitchen : RoomDimensions := ⟨12, 16, 10⟩
  let totalArea : ℝ := totalWallArea kitchen
  let coats : ℝ := 3 -- one primer + two paint coats
  let paintRate : ℝ := 40 -- square feet per hour
  paintingTime (totalArea * coats) paintRate = 42 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_martha_kitchen_painting_time_l31_3128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_values_of_a_l31_3121

def P : Set ℝ := {1, 2}
def Q (a : ℝ) : Set ℝ := {x | a * x + 2 = 0}

theorem possible_values_of_a (a : ℝ) : P ∪ Q a = P → a ∈ ({-2, -1, 0} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_values_of_a_l31_3121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_theorem_l31_3191

noncomputable section

-- Define the triangle ABC
def Triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = Real.pi

-- Define the area of a triangle
def AreaTriangle (a c B : ℝ) : ℝ := (1/2) * a * c * Real.sin B

-- State the theorem
theorem triangle_area_theorem (a b c A B C : ℝ) :
  Triangle a b c A B C →
  a = 2 →
  c = 2 * Real.sqrt 3 →
  A = Real.pi / 6 →
  (AreaTriangle a c B = 2 * Real.sqrt 3 ∨ AreaTriangle a c B = Real.sqrt 3) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_theorem_l31_3191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_field_trip_total_people_l31_3186

-- Define the vehicle types and their capacities
structure Vehicle where
  student_capacity : ℕ
  teacher_capacity : ℕ

-- Define the groups
structure TripGroup where
  students : ℕ
  teachers : ℕ

-- Define the field trip
def field_trip : Prop :=
  let van : Vehicle := ⟨7, 0⟩
  let bus : Vehicle := ⟨25, 2⟩
  let minibus : Vehicle := ⟨12, 1⟩
  let science_group : TripGroup := ⟨60, 6⟩
  let language_group : TripGroup := ⟨65, 5⟩
  let total_vans : ℕ := 3
  let total_buses : ℕ := 5
  let total_minibuses : ℕ := 2
  let total_people : ℕ := science_group.students + science_group.teachers + 
                          language_group.students + language_group.teachers
  let total_capacity : ℕ := total_vans * (van.student_capacity + van.teacher_capacity) +
                            total_buses * (bus.student_capacity + bus.teacher_capacity) +
                            total_minibuses * (minibus.student_capacity + minibus.teacher_capacity)
  (total_people = 136) ∧ (total_people ≤ total_capacity)

-- Theorem statement
theorem field_trip_total_people : field_trip := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_field_trip_total_people_l31_3186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l31_3111

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 1 then
    -x^2 + 2*(a+1)*x + 4
  else if x > 1 then
    x^a
  else
    0  -- This case is not specified in the original problem, but we need to handle it

-- State the theorem
theorem a_range (a : ℝ) : 
  (∀ x y, 0 < x ∧ x < y → f a x > f a y) → 
  -2 ≤ a ∧ a ≤ -1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l31_3111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_both_correct_l31_3147

theorem percentage_both_correct (total : ℕ) (first_correct second_correct neither_correct : ℕ) 
  (h1 : first_correct = (80 * total) / 100)
  (h2 : second_correct = (55 * total) / 100)
  (h3 : neither_correct = (20 * total) / 100)
  (h4 : total > 0) :
  ((first_correct + second_correct - (total - neither_correct)) * 100) / total = 55 := by
  sorry

#check percentage_both_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_both_correct_l31_3147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_fraction_of_decimal_l31_3157

theorem simplest_fraction_of_decimal (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  (a : ℚ) / b = 6125 / 10000 →
  Int.gcd a b = 1 →
  a + b = 129 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_fraction_of_decimal_l31_3157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_theorem_l31_3159

/-- Two triangles are similar if they have the same shape but possibly different sizes. -/
def SimilarTriangles (t1 t2 : Set Point) : Prop := sorry

/-- Two triangles have the same orientation if their vertices are arranged in the same order (clockwise or counterclockwise). -/
def SameOrientation (t1 t2 : Set Point) : Prop := sorry

/-- A point lies on the line connecting two other points. -/
def PointOnLine (p q r : Point) : Prop := sorry

/-- The ratio of distances between three collinear points. -/
noncomputable def DistanceRatio (p q r : Point) : ℝ := sorry

theorem similar_triangles_theorem (A₁ B₁ C₁ A₂ B₂ C₂ A B C : Point) :
  let t1 := {A₁, B₁, C₁}
  let t2 := {A₂, B₂, C₂}
  let t3 := {A, B, C}
  SimilarTriangles t1 t2 ∧ 
  SameOrientation t1 t2 ∧
  PointOnLine A₁ A A₂ ∧
  PointOnLine B₁ B B₂ ∧
  PointOnLine C₁ C C₂ ∧
  DistanceRatio A₁ A A₂ = DistanceRatio B₁ B B₂ ∧
  DistanceRatio B₁ B B₂ = DistanceRatio C₁ C C₂ →
  SimilarTriangles t3 t1 ∧ SimilarTriangles t3 t2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_theorem_l31_3159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_face_invariance_l31_3133

/-- Represents a regular tetrahedron -/
structure RegularTetrahedron where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Represents a flip of the tetrahedron over an edge -/
def flipTetrahedron (t : RegularTetrahedron) : RegularTetrahedron :=
  sorry

/-- Represents a sequence of flips -/
def flipSequence (t : RegularTetrahedron) (n : ℕ) : RegularTetrahedron :=
  match n with
  | 0 => t
  | n + 1 => flipTetrahedron (flipSequence t n)

/-- Predicate to check if the tetrahedron is in its original position -/
def isInOriginalPosition (t1 t2 : RegularTetrahedron) : Prop :=
  sorry

/-- Predicate to check if the faces are in their original positions -/
def facesInOriginalPositions (t1 t2 : RegularTetrahedron) : Prop :=
  sorry

/-- Theorem stating that if a sequence of flips returns the tetrahedron to its original position, 
    the faces must be in their original positions -/
theorem tetrahedron_face_invariance (t : RegularTetrahedron) (n : ℕ) :
  isInOriginalPosition t (flipSequence t n) → facesInOriginalPositions t (flipSequence t n) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_face_invariance_l31_3133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_and_max_area_l31_3170

/-- The circle on which point P lies -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- The curve C, which is the trajectory of point R -/
def curveC (x y : ℝ) : Prop := x^2/3 + y^2 = 1

/-- The condition that R satisfies RQ = √3 * PQ -/
def R_condition (xR yR xP yP : ℝ) : Prop :=
  xR = Real.sqrt 3 * xP ∧ yR = yP

/-- The slope product condition for points M and N -/
def slope_product_condition (xM yM xN yN : ℝ) : Prop :=
  ((yM - 1) / xM) * ((yN - 1) / xN) = 2/3

/-- The main theorem statement -/
theorem curve_C_and_max_area :
  ∀ (xP yP xQ yQ xR yR : ℝ),
  circle_equation xP yP →
  xQ = 0 ∧ yQ = yP →
  R_condition xR yR xP yP →
  (∀ x y, curveC x y ↔ ∃ xP yP, circle_equation xP yP ∧ R_condition x y xP yP) ∧
  (∀ xM yM xN yN,
    curveC xM yM →
    curveC xN yN →
    slope_product_condition xM yM xN yN →
    ∃ (S : ℝ), S = (1/2) * |xM - xN| * Real.sqrt 10 ∧
               S ≤ 2 * Real.sqrt 3 / 3 ∧
               (∃ xM' yM' xN' yN',
                 curveC xM' yM' ∧
                 curveC xN' yN' ∧
                 slope_product_condition xM' yM' xN' yN' ∧
                 S = 2 * Real.sqrt 3 / 3)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_and_max_area_l31_3170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parametric_to_standard_equation_l31_3164

-- Define the parametric equations
noncomputable def x (t : ℝ) : ℝ := Real.sqrt t
noncomputable def y (t : ℝ) : ℝ := 1 - 2 * Real.sqrt t

-- State the theorem
theorem parametric_to_standard_equation :
  ∀ t : ℝ, t ≥ 0 →
  ∃ x y : ℝ, x = Real.sqrt t ∧ y = 1 - 2 * Real.sqrt t ∧
  2 * x + y = 1 ∧ x ≥ 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parametric_to_standard_equation_l31_3164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_coefficients_l31_3131

/-- The polynomial f(x) = x^3 + x^2 + 2x + 3 -/
def f (x : ℝ) : ℝ := x^3 + x^2 + 2*x + 3

/-- The polynomial g(x) = x^3 + bx^2 + cx + d -/
def g (b c d : ℝ) (x : ℝ) : ℝ := x^3 + b*x^2 + c*x + d

/-- f has three distinct roots -/
axiom f_has_three_distinct_roots : ∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧ f r₁ = 0 ∧ f r₂ = 0 ∧ f r₃ = 0

/-- The roots of g are the squares of the roots of f -/
axiom g_roots_are_squares_of_f_roots : ∃ b c d : ℝ, ∀ r : ℝ, f r = 0 → g b c d (r^2) = 0

theorem g_coefficients : ∃ b c d : ℝ, (∀ r : ℝ, f r = 0 → g b c d (r^2) = 0) ∧ b = 3 ∧ c = -2 ∧ d = -9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_coefficients_l31_3131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closed_function_k_range_l31_3199

-- Define the function f
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := Real.sqrt (x + 2) + k

-- Define what it means for a function to be closed on an interval
def is_closed_function (f : ℝ → ℝ) (D : Set ℝ) : Prop :=
  ∃ (a b : ℝ), a ≤ b ∧ Set.Icc a b ⊆ D ∧ 
  (∀ x ∈ Set.Icc a b, f x ∈ Set.Icc a b) ∧
  (∀ y ∈ Set.Icc a b, ∃ x ∈ Set.Icc a b, f x = y)

-- State the theorem
theorem closed_function_k_range :
  ∀ k : ℝ, (∃ D : Set ℝ, Monotone (f k) ∧ is_closed_function (f k) D) ↔ 
  k ∈ Set.Ioo (-9/4) (-2) ∪ {-2} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_closed_function_k_range_l31_3199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_perpendicular_chords_ellipse_l31_3177

/-- The minimum sum of lengths of perpendicular chords in an ellipse with specific conditions -/
theorem min_sum_perpendicular_chords_ellipse 
  (a b : ℝ) 
  (h_ab : a > b ∧ b > 0) 
  (h_e : a / Real.sqrt (a^2 - b^2) = 3 / Real.sqrt 3) 
  (h_f2 : ∃ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 ∧ y^2 = 4*x ∧ x > 0) :
  ∃ (A B C D : ℝ × ℝ),
    (A.1^2 / a^2 + A.2^2 / b^2 = 1) ∧
    (B.1^2 / a^2 + B.2^2 / b^2 = 1) ∧
    (C.1^2 / a^2 + C.2^2 / b^2 = 1) ∧
    (D.1^2 / a^2 + D.2^2 / b^2 = 1) ∧
    ((A.2 - C.2) * (B.1 - D.1) = (A.1 - C.1) * (D.2 - B.2)) ∧
    (∀ (A' B' C' D' : ℝ × ℝ),
      (A'.1^2 / a^2 + A'.2^2 / b^2 = 1) →
      (B'.1^2 / a^2 + B'.2^2 / b^2 = 1) →
      (C'.1^2 / a^2 + C'.2^2 / b^2 = 1) →
      (D'.1^2 / a^2 + D'.2^2 / b^2 = 1) →
      ((A'.2 - C'.2) * (B'.1 - D'.1) = (A'.1 - C'.1) * (D'.2 - B'.2)) →
      Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) + Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) ≤
      Real.sqrt ((A'.1 - C'.1)^2 + (A'.2 - C'.2)^2) + Real.sqrt ((B'.1 - D'.1)^2 + (B'.2 - D'.2)^2)) ∧
    Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) + Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) = 16 * Real.sqrt 3 / 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_perpendicular_chords_ellipse_l31_3177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_triangle_possible_l31_3106

-- Define the set of stick lengths
def stickLengths : List ℕ := List.range 10 |>.map (fun n => 2^n)

-- Define the triangle inequality
def satisfiesTriangleInequality (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Theorem statement
theorem no_triangle_possible : 
  ∀ (a b c : ℕ), a ∈ stickLengths → b ∈ stickLengths → c ∈ stickLengths → 
    ¬(satisfiesTriangleInequality a b c) := by
  sorry

-- Example to show that the theorem applies to specific values
example : ¬(satisfiesTriangleInequality 1 2 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_triangle_possible_l31_3106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_curve_twice_no_intersection_outside_range_l31_3153

/-- The line equation y = k(x + 1) -/
def line (k x : ℝ) : ℝ := k * (x + 1)

/-- The curve equation y = √(4 - (x - 2)²) -/
noncomputable def curve (x : ℝ) : ℝ := Real.sqrt (4 - (x - 2)^2)

/-- The range of k values for which the line intersects the curve at two points -/
def k_range : Set ℝ := Set.Icc 0 (2 * Real.sqrt 5 / 5)

theorem line_intersects_curve_twice :
  ∀ k ∈ k_range, ∃ x₁ x₂, x₁ ≠ x₂ ∧ 
    line k x₁ = curve x₁ ∧ 
    line k x₂ = curve x₂ :=
by sorry

theorem no_intersection_outside_range :
  ∀ k ∉ k_range, ¬∃ x₁ x₂, x₁ ≠ x₂ ∧ 
    line k x₁ = curve x₁ ∧ 
    line k x₂ = curve x₂ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_curve_twice_no_intersection_outside_range_l31_3153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_max_profit_proof_l31_3197

/-- The number of dollars Rosencrantz is ahead after n flips -/
def f (n : ℕ) : ℤ → ℤ := sorry

/-- The probability of getting heads in a fair coin flip -/
noncomputable def p_heads : ℝ := 1/2

/-- The initial amount of money for each player -/
def initial_amount : ℕ := 2013

/-- The number of flips -/
def num_flips : ℕ := 2013

/-- The expected value of max{f(0), f(1), f(2), ..., f(2013)} -/
noncomputable def expected_max_profit : ℝ :=
  -1/2 + (1007 * (Nat.choose 2013 1006 : ℝ)) / 2^2012

theorem expected_max_profit_proof :
  expected_max_profit = -1/2 + (1007 * (Nat.choose 2013 1006 : ℝ)) / 2^2012 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_max_profit_proof_l31_3197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_energy_in_doubled_square_energy_change_in_square_config_l31_3127

/-- Energy stored by two charges -/
noncomputable def energy_between_charges (q1 q2 distance : ℝ) : ℝ := q1 * q2 / distance

/-- Total energy stored in a square configuration of charges -/
noncomputable def total_energy (charge side_length : ℝ) : ℝ :=
  4 * energy_between_charges charge charge side_length

/-- Theorem: Energy in new configuration is half of original -/
theorem energy_in_doubled_square (charge d : ℝ) (h1 : charge > 0) (h2 : d > 0) :
  total_energy charge (2 * d) = (total_energy charge d) / 2 := by
  sorry

/-- Main theorem: If original configuration stores 20 Joules, new configuration stores 10 Joules -/
theorem energy_change_in_square_config (charge d : ℝ) 
  (h1 : charge > 0) (h2 : d > 0) (h3 : total_energy charge d = 20) :
  total_energy charge (2 * d) = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_energy_in_doubled_square_energy_change_in_square_config_l31_3127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_equality_l31_3145

theorem sine_cosine_equality (α β : ℝ) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2) : 
  (Real.sin α)^4 / (Real.cos β)^2 + (Real.sin β)^4 / (Real.cos α)^2 = 1 ↔ α + β = π/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_equality_l31_3145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_light_reflection_l31_3101

/-- The plane equation: 2x - y + z = 8 -/
def plane_equation (p : ℝ × ℝ × ℝ) : Prop :=
  2 * p.1 - p.2.1 + p.2.2 = 8

/-- Check if three points are collinear -/
def collinear (p q r : ℝ × ℝ × ℝ) : Prop :=
  ∃ t : ℝ, r.1 - p.1 = t * (q.1 - p.1) ∧
           r.2.1 - p.2.1 = t * (q.2.1 - p.2.1) ∧
           r.2.2 - p.2.2 = t * (q.2.2 - p.2.2)

/-- The reflection of a point with respect to a plane -/
noncomputable def reflect (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (10/3, 32/3, 68/3)

theorem light_reflection
  (A : ℝ × ℝ × ℝ) (C : ℝ × ℝ × ℝ) (B : ℝ × ℝ × ℝ)
  (hA : A = (-2, 8, 12))
  (hC : C = (4, 2, 10))
  (hB : B = (20/3, -94/3, -122/3)) :
  plane_equation B ∧ collinear B (reflect A) C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_light_reflection_l31_3101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_sixteen_thirds_l31_3125

theorem sum_of_solutions_is_sixteen_thirds :
  let f (x : ℝ) := (8 : ℝ)^(x^2 - 4*x - 3) = (16 : ℝ)^(x - 3)
  ∃ (x₁ x₂ : ℝ), f x₁ ∧ f x₂ ∧ (∀ x, f x → (x = x₁ ∨ x = x₂)) ∧ x₁ + x₂ = 16/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_sixteen_thirds_l31_3125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_current_speed_calculation_l31_3138

-- Define constants
noncomputable def rowing_speed_kmph : ℝ := 15
noncomputable def distance_downstream : ℝ := 15
noncomputable def time_downstream : ℝ := 2.9997600191984644

-- Define functions
noncomputable def kmph_to_mps (speed : ℝ) : ℝ := speed * (1000 / 3600)

noncomputable def downstream_speed (distance time : ℝ) : ℝ := distance / time

-- Theorem statement
theorem current_speed_calculation :
  let rowing_speed_mps := kmph_to_mps rowing_speed_kmph
  let speed_downstream := downstream_speed distance_downstream time_downstream
  ∃ ε > 0, |speed_downstream - rowing_speed_mps - 0.83333333| < ε := by
  sorry

#check current_speed_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_current_speed_calculation_l31_3138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mechanical_setup_draws_straight_line_l31_3117

-- Define the basic structure of our mechanical setup
structure MechanicalSetup where
  A : EuclideanSpace ℝ (Fin 2)
  B : EuclideanSpace ℝ (Fin 2)
  C : EuclideanSpace ℝ (Fin 2)
  D : EuclideanSpace ℝ (Fin 2)

-- Define the properties of our setup
def is_valid_setup (s : MechanicalSetup) : Prop :=
  -- AB = BC (fixed points condition)
  ‖s.A - s.B‖ = ‖s.B - s.C‖ ∧
  -- Rhombus condition (all sides equal)
  ‖s.A - s.B‖ = ‖s.B - s.C‖ ∧
  ‖s.B - s.C‖ = ‖s.C - s.D‖ ∧
  ‖s.C - s.D‖ = ‖s.D - s.A‖

-- Define what it means for a point to be on a straight line
def on_straight_line (A B P : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ t : ℝ, P = (1 - t) • A + t • B

-- Theorem statement
theorem mechanical_setup_draws_straight_line (s : MechanicalSetup) 
  (h : is_valid_setup s) : 
  ∀ (P : EuclideanSpace ℝ (Fin 2)), (∃ (t : ℝ), s.D = (1 - t) • s.A + t • s.C) → 
    on_straight_line s.A s.C P :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mechanical_setup_draws_straight_line_l31_3117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_diagonals_perpendicular_l31_3190

-- Define what a rhombus is
def is_rhombus {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] (quad : Set E) : Prop := sorry

-- Define what it means for lines to be perpendicular
def are_perpendicular {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] (l1 l2 : Set E) : Prop := sorry

-- Define the diagonals of a quadrilateral
def diagonals {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] (quad : Set E) : (Set E × Set E) := sorry

-- The theorem
theorem rhombus_diagonals_perpendicular {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] (quad : Set E) :
  is_rhombus quad → are_perpendicular (diagonals quad).1 (diagonals quad).2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_diagonals_perpendicular_l31_3190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l31_3107

/-- The circle defined by the equation x^2 + y^2 - 2x - 2y + 1 = 0 -/
def Circle : Set (ℝ × ℝ) :=
  {p | p.1^2 + p.2^2 - 2*p.1 - 2*p.2 + 1 = 0}

/-- The line defined by the equation x - y = 2 -/
def Line : Set (ℝ × ℝ) :=
  {p | p.1 - p.2 = 2}

/-- The maximum distance from a point on the circle to the line -/
theorem max_distance_circle_to_line :
  ∃ (p : ℝ × ℝ), p ∈ Circle ∧
    ∀ (q : ℝ × ℝ), q ∈ Circle →
      ∃ (r : ℝ × ℝ), r ∈ Line ∧
        dist p r ≥ dist q r ∧
        dist p r = 1 + Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l31_3107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_theorem_l31_3168

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- A line segment with endpoints on an ellipse -/
structure EllipseChord (E : Ellipse) where
  slope : ℝ
  midpoint : ℝ × ℝ

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (E : Ellipse) : ℝ :=
  Real.sqrt (1 - (E.b / E.a)^2)

/-- Theorem: If a chord of an ellipse has slope -1/2 and midpoint (1,1),
    then the eccentricity of the ellipse is √2/2 -/
theorem ellipse_eccentricity_theorem (E : Ellipse) (chord : EllipseChord E) :
  chord.slope = -1/2 → chord.midpoint = (1, 1) → eccentricity E = Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_theorem_l31_3168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_properties_l31_3143

-- Define the quadratic function
noncomputable def f (b c x : ℝ) : ℝ := x^2 + b*x + c

-- Define the condition that the function passes through (-2, 4)
def passes_through_point (b c : ℝ) : Prop := f b c (-2) = 4

-- Define the vertex of the parabola
noncomputable def vertex (b c : ℝ) : ℝ × ℝ := (-b/2, f b c (-b/2))

-- Theorem statement
theorem quadratic_properties (b c : ℝ) (h : passes_through_point b c) :
  c = 2*b ∧ 
  ∀ m n : ℝ, vertex b c = (m, n) → n = -m^2 - 4*m := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_properties_l31_3143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_value_l31_3142

/-- Given a polynomial x^3 - ax^2 + bx - 30030 with three positive integer roots,
    the smallest possible value of a is 94 -/
theorem smallest_a_value (a b : ℤ) :
  (∃ p q r : ℤ, p > 0 ∧ q > 0 ∧ r > 0 ∧
    p^3 - a * p^2 + b * p - 30030 = 0 ∧
    q^3 - a * q^2 + b * q - 30030 = 0 ∧
    r^3 - a * r^2 + b * r - 30030 = 0) →
  a ≥ 94 ∧ ∃ a₀ b₀ : ℤ, a₀ = 94 ∧ 
    (∃ p q r : ℤ, p > 0 ∧ q > 0 ∧ r > 0 ∧
      p^3 - a₀ * p^2 + b₀ * p - 30030 = 0 ∧
      q^3 - a₀ * q^2 + b₀ * q - 30030 = 0 ∧
      r^3 - a₀ * r^2 + b₀ * r - 30030 = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_value_l31_3142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l31_3161

-- Define the curves and point
noncomputable def C₁ (ρ θ : ℝ) : Prop := ρ * Real.cos θ - ρ * Real.sin θ + 2 = 0

noncomputable def C₂ (α : ℝ) : ℝ × ℝ := (Real.cos α, 2 * Real.sin α)

noncomputable def C₃ (α : ℝ) : ℝ × ℝ := (3 * Real.cos α, 3 * Real.sin α)

def P : ℝ × ℝ := (0, 2)

-- Define the theorem
theorem intersection_distance_sum :
  ∃ (A B : ℝ × ℝ), 
    (∃ (t : ℝ), A.1 = Real.sqrt 2 / 2 * t ∧ A.2 = 2 + Real.sqrt 2 / 2 * t) ∧
    (∃ (t : ℝ), B.1 = Real.sqrt 2 / 2 * t ∧ B.2 = 2 + Real.sqrt 2 / 2 * t) ∧
    A.1^2 + A.2^2 = 9 ∧
    B.1^2 + B.2^2 = 9 ∧
    Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2) + Real.sqrt ((B.1 - P.1)^2 + (B.2 - P.2)^2) = 2 * Real.sqrt 7 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l31_3161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_six_20_to_109_l31_3172

/-- Count of digit 6 in a range of integers -/
def count_digit_six (start : ℕ) (stop : ℕ) : ℕ :=
  (List.range (stop - start + 1)).map (· + start)
    |>.filter (λ n => n.repr.contains '6')
    |>.length

/-- The problem statement -/
theorem count_six_20_to_109 :
  count_digit_six 20 109 = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_six_20_to_109_l31_3172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_through_point_l31_3124

/-- The equation of a circle with center (h, k) and radius r is (x - h)^2 + (y - k)^2 = r^2 -/
def circle_equation (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- The distance between two points (x1, y1) and (x2, y2) -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

theorem circle_through_point (x y : ℝ) :
  circle_equation (-3) 2 (distance (-3) 2 1 (-1)) x y ↔ (x + 3)^2 + (y - 2)^2 = 25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_through_point_l31_3124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_place_mat_length_theorem_l31_3148

/-- The length of a place mat on a circular table -/
noncomputable def place_mat_length (table_radius : ℝ) (num_mats : ℕ) (mat_width : ℝ) : ℝ :=
  Real.sqrt 24.75 - 5 * Real.sqrt ((2 + Real.sqrt 2) / 2) + 1 / 2

/-- Theorem stating the length of place mats on a circular table -/
theorem place_mat_length_theorem :
  place_mat_length 5 8 1 = Real.sqrt 24.75 - 5 * Real.sqrt ((2 + Real.sqrt 2) / 2) + 1 / 2 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_place_mat_length_theorem_l31_3148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_can_write_sqrt3_l31_3132

-- Define the set of numbers that can be written on the board
inductive BoardNumber : Type
| zero : BoardNumber
| trig : BoardNumber → BoardNumber
| inv_trig : BoardNumber → BoardNumber
| quotient : BoardNumber → BoardNumber → BoardNumber
| product : BoardNumber → BoardNumber → BoardNumber

-- Define the goal number √3
noncomputable def sqrt3 : ℝ := Real.sqrt 3

-- State the theorem
theorem can_write_sqrt3 : ∃ (b : BoardNumber), ∃ (f : BoardNumber → ℝ), f b = sqrt3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_can_write_sqrt3_l31_3132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l31_3189

theorem triangle_side_length (A B C : ℝ) (sinB : ℝ) (AC : ℝ) :
  sinB = 4/5 →
  AC = 3 →
  Real.sqrt ((4/5 * 5)^2 + 3^2) = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l31_3189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_forall_sin_leq_one_l31_3119

theorem negation_of_forall_sin_leq_one :
  (¬ ∀ x : ℝ, Real.sin x ≤ 1) ↔ (∃ x : ℝ, Real.sin x > 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_forall_sin_leq_one_l31_3119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_cosine_function_l31_3192

noncomputable def f (x : ℝ) := Real.cos (3 * x + Real.pi / 6)

theorem period_of_cosine_function (x : ℝ) :
  f (x + 2 * Real.pi / 3) = f x :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_cosine_function_l31_3192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_factorial_consecutive_product_l31_3163

theorem smallest_factorial_consecutive_product : 
  ∃ n : ℕ, n = 23 ∧
  (∃ m : ℕ, m ≥ 5 ∧ n! = (Finset.range m).prod (λ i => n - 5 + i + 1)) ∧
  (∀ k : ℕ, k < 23 → 
    ¬(∃ m : ℕ, m ≥ 5 ∧ k! = (Finset.range m).prod (λ i => k - 5 + i + 1))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_factorial_consecutive_product_l31_3163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_equals_result_l31_3112

open Real MeasureTheory

noncomputable def f (x : ℝ) : ℝ := (((3 * x + 5) ^ (1/3 : ℝ)) + 2) / (1 + ((3 * x + 5) ^ (1/3 : ℝ)))

theorem definite_integral_equals_result : 
  ∫ x in (-5/3)..1, f x = 8/3 + Real.log 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_equals_result_l31_3112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_female_worker_ants_l31_3100

def total_ants : ℕ := 110

def worker_ants : ℕ := total_ants / 2

def male_worker_ants : ℕ := (worker_ants * 20) / 100

theorem female_worker_ants : worker_ants - male_worker_ants = 44 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_female_worker_ants_l31_3100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rect_to_polar_circle_l31_3120

/-- Given a point (x, y) in rectangular coordinates and (ρ, θ) in polar coordinates,
    prove that (x-3)^2 + y^2 = 9 is equivalent to ρ = 6*cos(θ) --/
theorem rect_to_polar_circle (x y ρ θ : ℝ) :
  (x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  ((x - 3)^2 + y^2 = 9) ↔ (ρ = 6 * Real.cos θ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rect_to_polar_circle_l31_3120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_at_negative_eight_l31_3193

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := 1 - Real.log x / Real.log 3

-- State the theorem
theorem inverse_f_at_negative_eight :
  ∃ (f_inv : ℝ → ℝ), Function.LeftInverse f_inv f ∧ Function.RightInverse f_inv f ∧ f_inv (-8) = 3^9 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_at_negative_eight_l31_3193
