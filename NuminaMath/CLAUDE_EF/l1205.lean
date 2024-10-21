import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sin_B_in_arithmetic_sequence_triangle_l1205_120589

theorem max_sin_B_in_arithmetic_sequence_triangle (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  2 * b^2 = a^2 + c^2 →
  (∃ (A B C : ℝ), 
    0 < A ∧ A < π ∧
    0 < B ∧ B < π ∧
    0 < C ∧ C < π ∧
    A + B + C = π ∧
    a = b * Real.sin C ∧
    b = c * Real.sin A ∧
    c = a * Real.sin B) →
  (∀ B : ℝ, 0 < B ∧ B < π ∧ c = a * Real.sin B → Real.sin B ≤ Real.sqrt 3 / 2) ∧
  (∃ B : ℝ, 0 < B ∧ B < π ∧ c = a * Real.sin B ∧ Real.sin B = Real.sqrt 3 / 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sin_B_in_arithmetic_sequence_triangle_l1205_120589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f₃_minus_f₁_of_6_pow_2020_l1205_120562

/-- Ω(n) denotes the number of prime factors of n, counting multiplicity -/
def Ω : ℕ+ → ℕ := sorry

/-- f₁(n) is the sum of positive divisors d|n where Ω(d) ≡ 1 (mod 4) -/
def f₁ (n : ℕ+) : ℕ := sorry

/-- f₃(n) is the sum of positive divisors d|n where Ω(d) ≡ 3 (mod 4) -/
def f₃ (n : ℕ+) : ℕ := sorry

/-- The main theorem to prove -/
theorem f₃_minus_f₁_of_6_pow_2020 :
  f₃ (6 ^ 2020 : ℕ+) - f₁ (6 ^ 2020 : ℕ+) = (6 ^ 2021 - 3 ^ 2021 - 2 ^ 2021 - 1) / 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f₃_minus_f₁_of_6_pow_2020_l1205_120562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_1_expression_value_2_l1205_120587

-- Part 1
theorem expression_value_1 : 
  Real.log 8 + Real.log 125 - (1/7)^(-2 : ℤ) + 16^(3/4) + (Real.sqrt 3 - 1)^(0 : ℕ) = -37 := by sorry

-- Part 2
theorem expression_value_2 : 
  Real.sin (25 * Real.pi / 6) + Real.cos (25 * Real.pi / 3) + Real.tan (-25 * Real.pi / 4) = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_1_expression_value_2_l1205_120587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_quadratic_composition_with_eight_roots_l1205_120547

/-- A quadratic polynomial over ℝ -/
def QuadraticPolynomial := ℝ → ℝ

/-- The set of solutions to the equation -/
def SolutionSet : Set ℝ := {1, 2, 3, 4, 5, 6, 7, 8}

/-- The main theorem statement -/
theorem no_quadratic_composition_with_eight_roots :
  ¬ ∃ (f g h : QuadraticPolynomial),
    (∀ x ∈ SolutionSet, f (g (h x)) = 0) ∧
    (∃ a b c : ℝ, f = λ x ↦ a*x^2 + b*x + c) ∧
    (∃ d e k : ℝ, g = λ x ↦ d*x^2 + e*x + k) ∧
    (∃ l m n : ℝ, h = λ x ↦ l*x^2 + m*x + n) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_quadratic_composition_with_eight_roots_l1205_120547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_and_side_c_l1205_120598

noncomputable def f (x : ℝ) := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x)^2 - 1

theorem max_value_and_side_c :
  (∃ (x : ℝ), f x = 2 ∧ ∀ (y : ℝ), f y ≤ 2) ∧
  ∀ (A B C : ℝ) (a b c : ℝ),
    a = Real.sqrt 7 →
    b = Real.sqrt 3 →
    f (A / 2) = Real.sqrt 3 →
    (c = 4 ∨ c = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_and_side_c_l1205_120598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_g_equals_half_l1205_120538

/-- The function g(n) defined as the sum of reciprocals of powers of integers starting from 3 -/
noncomputable def g (n : ℕ) : ℝ := ∑' k, (1 : ℝ) / (k + 2 : ℝ) ^ n

/-- The theorem stating that the sum of g(n) from n=1 to infinity equals 1/2 -/
theorem sum_of_g_equals_half : ∑' n, g n = (1 : ℝ) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_g_equals_half_l1205_120538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_employed_males_percentage_l1205_120554

theorem employed_males_percentage 
  (total_population : ℕ) 
  (employed_percentage : ℚ) 
  (employed_females_percentage : ℚ)
  (h1 : employed_percentage = 60 / 100)
  (h2 : employed_females_percentage = 20 / 100)
  : ℚ := by
  let employed := (total_population : ℚ) * employed_percentage
  let employed_females := employed * employed_females_percentage
  let employed_males := employed - employed_females
  have h3 : employed_males / (total_population : ℚ) = 48 / 100 := by sorry
  exact 48 / 100


end NUMINAMATH_CALUDE_ERRORFEEDBACK_employed_males_percentage_l1205_120554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_exp_sin_difference_l1205_120543

/-- The limit of (e^(αx) - e^(βx)) / (sin(αx) - sin(βx)) as x approaches 0 is 1 -/
theorem limit_exp_sin_difference (α β : ℝ) :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x| ∧ |x| < δ →
    |((Real.exp (α * x) - Real.exp (β * x)) / (Real.sin (α * x) - Real.sin (β * x))) - 1| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_exp_sin_difference_l1205_120543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_distance_l1205_120512

noncomputable section

-- Define the hyperbola
def Hyperbola (a b c : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) - (p.2^2 / b^2) = 1}

-- Define the circle
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 6)^2 + p.2^2 = 20}

-- Define the asymptote
def Asymptote (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = (b / a) * p.1}

-- Define the distance between two points
noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

theorem hyperbola_focus_distance 
  (a b c : ℝ) 
  (h_c : c = 6) 
  (h_a : a^2 + b^2 = c^2) 
  (h_tangent : ∃ p ∈ Circle ∩ Asymptote a b, True)
  (P : ℝ × ℝ)
  (h_P : P ∈ Hyperbola a b c)
  (F₁ F₂ : ℝ × ℝ)
  (h_F₁ : F₁ = (c, 0))
  (h_F₂ : F₂ = (-c, 0))
  (h_PF₁ : distance P F₁ = 9) :
  distance P F₂ = 17 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_distance_l1205_120512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_and_eccentricity_l1205_120516

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 (a > 0, b > 0) and an asymptote
    with inclination angle π/6, prove its asymptote equation and eccentricity. -/
theorem hyperbola_asymptote_and_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ → ℝ), ∀ t, (x t)^2 / a^2 - (y t)^2 / b^2 = 1 ∧ 
   ∃ (m : ℝ), Real.tan (π/6) = m ∧ (∀ t, y t = m * x t ∨ y t = -m * x t)) →
  (∀ (x y : ℝ), y = (Real.sqrt 3/3) * x ∨ y = -(Real.sqrt 3/3) * x) ∧
  Real.sqrt (1 + b^2/a^2) = 2*(Real.sqrt 3)/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_and_eccentricity_l1205_120516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_M_l1205_120550

def M : ℕ := 2^6 * 3^5 * 5^3 * 7^1 * 11^2

theorem number_of_factors_of_M : (Finset.filter (· ∣ M) (Finset.range (M + 1))).card = 1008 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_M_l1205_120550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_with_given_arcs_l1205_120566

/-- The area of a triangle inscribed in a circle, given the arc lengths --/
noncomputable def triangleArea (arc1 arc2 arc3 : ℝ) : ℝ :=
  let circumference := arc1 + arc2 + arc3
  let radius := circumference / (2 * Real.pi)
  (2 * radius^2 / Real.pi) * (1 + (Real.sqrt ((2 + Real.sqrt 2) / 4)) + (Real.sqrt ((2 - Real.sqrt 2) / 4)))

/-- Theorem stating the area of a triangle inscribed in a circle with specific arc lengths --/
theorem triangle_area_with_given_arcs :
  triangleArea 4 5 7 = (32 / Real.pi^2) * (1 + (Real.sqrt ((2 + Real.sqrt 2) / 4)) + (Real.sqrt ((2 - Real.sqrt 2) / 4))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_with_given_arcs_l1205_120566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_and_directrix_l1205_120575

/-- A parabola with equation x = -1/8 * y^2 has focus at (-1/32, 0) and directrix x = 1/32 -/
theorem parabola_focus_and_directrix :
  let parabola := {p : ℝ × ℝ | p.1 = -1/8 * p.2^2}
  ∃ (focus : ℝ × ℝ) (directrix : Set ℝ),
    focus = (-1/32, 0) ∧
    directrix = {x : ℝ | x = 1/32} ∧
    ∀ (p : ℝ × ℝ), p ∈ parabola ↔
      Real.sqrt ((p.1 - focus.1)^2 + (p.2 - focus.2)^2) =
      |p.1 - 1/32| := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_and_directrix_l1205_120575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l1205_120507

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 7 - y^2 / 3 = 1

-- Define the asymptotes
def asymptotes (x y : ℝ) : Prop := y = (Real.sqrt 21 / 7) * x ∨ y = -(Real.sqrt 21 / 7) * x

-- Theorem statement
theorem hyperbola_asymptotes :
  ∀ x y : ℝ, asymptotes x y ↔ (∃ ε > 0, ∀ δ > ε, ∃ x' y', hyperbola x' y' ∧ |y - y'| < δ ∧ |x - x'| < δ) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l1205_120507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_staff_final_price_l1205_120518

/-- Calculates the final price for a staff member buying three dresses -/
noncomputable def final_price (d : ℝ) : ℝ :=
  let discount1 := 0.3
  let discount2 := 0.5
  let discount3 := 0.6
  let staff_discount := 0.5
  let sales_tax := 0.1
  let env_surcharge := 0.05
  let discounted_price := (1 - discount1) * d + (1 - discount2) * d + (1 - discount3) * d
  let staff_reduction := staff_discount * (1 - discount3) * d
  let price_after_staff_discount := discounted_price - staff_reduction
  let price_with_tax := price_after_staff_discount * (1 + sales_tax)
  if d ≤ 100 then
    price_with_tax
  else
    price_with_tax + 3 * env_surcharge * d

theorem staff_final_price (d : ℝ) :
  (d ≤ 100 → final_price d = 1.54 * d) ∧
  (d > 100 → final_price d = 1.69 * d) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_staff_final_price_l1205_120518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_integer_satisfying_inequality_l1205_120505

theorem largest_integer_satisfying_inequality :
  ∃ (x : ℤ), (3 * x.natAbs + 10 ≤ 25) ∧ (∀ y : ℤ, 3 * y.natAbs + 10 ≤ 25 → y ≤ x) ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_integer_satisfying_inequality_l1205_120505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_odd_numbers_l1205_120506

theorem consecutive_odd_numbers (a b c d e x : ℤ) : 
  (∀ n : ℤ, n ∈ [a, b, c, d, e] → n % 2 = 1) → -- All numbers are odd
  (b = a + 2) → (c = b + 2) → (d = c + 2) → (e = d + 2) → -- Consecutive
  (e = 79) → -- Given value of e
  (a + x = 146) → -- Sum of a and another odd number
  (x % 2 = 1) → -- x is odd
  x = 75 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_odd_numbers_l1205_120506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grandmother_money_sum_l1205_120584

/-- The amount of money Pete received in cents -/
def P : ℕ := sorry

/-- The amount of money Raymond received in cents -/
def R : ℕ := sorry

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The number of nickels Pete spent -/
def pete_spent_nickels : ℕ := 4

/-- The number of dimes Raymond has left -/
def raymond_left_dimes : ℕ := 7

/-- The total amount they spent in cents -/
def total_spent : ℕ := 200

theorem grandmother_money_sum :
  P + R = total_spent + (pete_spent_nickels * nickel_value) + (raymond_left_dimes * dime_value) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grandmother_money_sum_l1205_120584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_casey_pumping_time_l1205_120510

/-- Represents Casey's farm and water pumping scenario -/
structure CaseyFarm where
  pump_rate : ℚ
  corn_rows : ℕ
  corn_plants_per_row : ℕ
  corn_water_need : ℚ
  pig_count : ℕ
  pig_water_need : ℚ
  duck_count : ℕ
  duck_water_need : ℚ

/-- Calculates the time needed to pump water for Casey's farm -/
def time_to_pump (farm : CaseyFarm) : ℚ :=
  let total_corn_plants := farm.corn_rows * farm.corn_plants_per_row
  let total_water_needed := 
    (total_corn_plants : ℚ) * farm.corn_water_need +
    (farm.pig_count : ℚ) * farm.pig_water_need +
    (farm.duck_count : ℚ) * farm.duck_water_need
  total_water_needed / farm.pump_rate

/-- Theorem stating that Casey needs 25 minutes to pump water -/
theorem casey_pumping_time :
  let farm := CaseyFarm.mk 3 4 15 (1/2) 10 4 20 (1/4)
  time_to_pump farm = 25 := by
  sorry

#eval time_to_pump (CaseyFarm.mk 3 4 15 (1/2) 10 4 20 (1/4))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_casey_pumping_time_l1205_120510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jones_license_plate_problem_l1205_120581

/-- Represents a 4-digit number where each of two digits appears twice -/
structure RepeatDigitNumber where
  value : ℕ
  is_repeat_digit : ∃ a b : ℕ, a ≠ b ∧ a < 10 ∧ b < 10 ∧ 
    (value = 1000*a + 100*a + 10*b + b ∨
     value = 1000*a + 100*b + 10*b + a ∨
     value = 1000*a + 100*b + 10*a + b)

/-- Represents the ages of Mr. Jones's children -/
structure JonesChildren where
  ages : Finset ℕ
  count_eight : ages.card = 8
  all_different : ∀ (a b : ℕ), a ∈ ages → b ∈ ages → a = b → a == b
  max_age_nine : ∀ a, a ∈ ages → a ≤ 9
  has_nine : 9 ∈ ages

theorem jones_license_plate_problem 
  (plate : RepeatDigitNumber)
  (children : JonesChildren)
  (h1 : ∀ a, a ∈ children.ages → plate.value % a = 0)
  (h2 : plate.value % 100 ≤ 99)
  : plate.value = 5544 ∧ (∀ n : ℕ, n ∈ Finset.range 10 → n ≠ 5 → plate.value % n = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jones_license_plate_problem_l1205_120581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l1205_120561

theorem polynomial_division_remainder :
  ∃ q : Polynomial ℝ, (X : Polynomial ℝ)^5 + 3*(X : Polynomial ℝ)^3 + 3 = q * (X - 1)^2 + (14*X - 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l1205_120561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1205_120534

-- Define the function f
noncomputable def f (x : ℝ) : ℤ := ⌊x⌋ + ⌊1 - x⌋

-- Theorem stating the properties of f
theorem f_properties (x : ℝ) :
  (∃ n : ℤ, x = n → f x = 1) ∧
  (∀ n : ℤ, x ≠ n → f x = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1205_120534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_correct_transformation_l1205_120556

-- Define the type for equation transformations
inductive EquationTransformation
  | add_to_both_sides : ℚ → ℚ → ℚ → EquationTransformation
  | divide_both_sides : ℚ → ℚ → EquationTransformation
  | multiply_both_sides : ℚ → ℚ → EquationTransformation
  | subtract_from_both_sides : ℚ → ℚ → ℚ → EquationTransformation

-- Function to check if a transformation is correct
def is_correct_transformation (t : EquationTransformation) : Bool :=
  match t with
  | EquationTransformation.add_to_both_sides a b c => a + b = c
  | EquationTransformation.divide_both_sides a b => a ≠ 0 ∧ b / a = -4 / 7
  | EquationTransformation.multiply_both_sides a b => a ≠ 0 ∧ a * b = 0
  | EquationTransformation.subtract_from_both_sides a b c => a - b = c

-- Define the list of transformations
def transformations : List EquationTransformation :=
  [ EquationTransformation.add_to_both_sides 3 5 8
  , EquationTransformation.divide_both_sides 7 (-4)
  , EquationTransformation.multiply_both_sides (1/2) 2
  , EquationTransformation.subtract_from_both_sides 3 (-2) (-5)
  ]

-- Theorem to prove
theorem exactly_one_correct_transformation :
  (transformations.filter is_correct_transformation).length = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_correct_transformation_l1205_120556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_sqrt_two_l1205_120579

-- Define m
noncomputable def m : ℝ := 2 + Real.sqrt 2

-- Define the expression
noncomputable def expression (x : ℝ) : ℝ := 
  (1 - x / (x + 2)) / ((x^2 - 4*x + 4) / (x^2 - 4))

-- Theorem statement
theorem expression_equals_sqrt_two : expression m = Real.sqrt 2 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_sqrt_two_l1205_120579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_inequality_l1205_120528

noncomputable def a : ℝ := ∫ x in (0:ℝ)..1, x^(1/3)
noncomputable def b : ℝ := ∫ x in (0:ℝ)..1, Real.sqrt x
noncomputable def c : ℝ := ∫ x in (0:ℝ)..1, Real.sin x

theorem integral_inequality : c < b ∧ b < a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_inequality_l1205_120528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_on_ellipse_l1205_120571

-- Define the ellipse G
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the unit circle (renamed to avoid conflict)
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define a tangent line to the circle passing through (m, 0)
def tangent_line (m k : ℝ) (x y : ℝ) : Prop := y = k * (x - m) ∧ m^2 * k^2 = k^2 + 1

-- Define the intersection points of the tangent line with the ellipse
def intersection_points (m k x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧ tangent_line m k x₁ y₁ ∧ tangent_line m k x₂ y₂

-- Define the distance between two points
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

-- Theorem statement
theorem max_distance_on_ellipse :
  ∀ m k x₁ y₁ x₂ y₂ : ℝ,
    intersection_points m k x₁ y₁ x₂ y₂ →
    distance x₁ y₁ x₂ y₂ ≤ 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_on_ellipse_l1205_120571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_theorem_l1205_120546

/-- An odd, strictly decreasing function on ℝ. -/
noncomputable def f : ℝ → ℝ :=
  sorry

/-- f is odd. -/
axiom f_odd : ∀ x, f (-x) = -f x

/-- f is strictly decreasing. -/
axiom f_decreasing : ∀ x y, x < y → f y < f x

/-- f(1) = -1 -/
axiom f_one : f 1 = -1

/-- The main theorem: the range of x that satisfies -1 ≤ f(x-2) ≤ 1 is [1,3]. -/
theorem range_theorem : Set.Icc 1 3 = {x | -1 ≤ f (x - 2) ∧ f (x - 2) ≤ 1} := by
  sorry

#check range_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_theorem_l1205_120546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_surface_area_formula_l1205_120572

/-- A right prism with an isosceles triangle base -/
structure RightPrism where
  -- Base triangle
  α : ℝ  -- Angle ABC
  -- Edge properties
  b : ℝ  -- Length of CD
  β : ℝ  -- Angle DCA

/-- The lateral surface area of the right prism -/
noncomputable def lateralSurfaceArea (prism : RightPrism) : ℝ :=
  4 * prism.b^2 * Real.sin (2 * prism.β) * Real.cos (prism.α / 2)^2

/-- Theorem: The lateral surface area of the right prism is 4b² sin(2β) cos²(α/2) -/
theorem lateral_surface_area_formula (prism : RightPrism) :
  lateralSurfaceArea prism = 4 * prism.b^2 * Real.sin (2 * prism.β) * Real.cos (prism.α / 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_surface_area_formula_l1205_120572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decal_enlargement_l1205_120569

/-- Given a rectangular decal with original width and height, and a new width,
    calculate the new height while maintaining the same proportions -/
noncomputable def enlargedHeight (originalWidth originalHeight newWidth : ℝ) : ℝ :=
  (newWidth / originalWidth) * originalHeight

/-- Theorem stating that enlarging a 3x2 inch decal to 15 inches wide
    results in a height of 10 inches -/
theorem decal_enlargement :
  let originalWidth : ℝ := 3
  let originalHeight : ℝ := 2
  let newWidth : ℝ := 15
  enlargedHeight originalWidth originalHeight newWidth = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decal_enlargement_l1205_120569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_closed_figure_l1205_120513

-- Define the bounds of the figure
noncomputable def lower_bound : ℝ := 1/2
noncomputable def upper_bound : ℝ := 2

-- Define the curve
noncomputable def curve (x : ℝ) : ℝ := 1/x

-- State the theorem
theorem area_of_closed_figure :
  (∫ (y : ℝ) in lower_bound..upper_bound, 1/y) = 2 * Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_closed_figure_l1205_120513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lower_limit_of_prime_set_l1205_120522

def isPrime (n : ℕ) : Prop := Nat.Prime n

def isInRange (n a b : ℕ) : Prop := a ≤ n ∧ n ≤ b

theorem lower_limit_of_prime_set (W : Set ℕ) : 
  (∀ n, n ∈ W → isPrime n ∧ isInRange n 7 25) → 
  (∃ a b, a ∈ W ∧ b ∈ W ∧ b - a = 12 ∧ ∀ c ∈ W, a ≤ c ∧ c ≤ b) →
  (∀ n, n < 7 → n ∉ W) →
  7 ∈ W :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lower_limit_of_prime_set_l1205_120522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_intercepts_l1205_120542

/-- A line in the xy-plane defined by the equation ax + y - 2 - a = 0 -/
structure Line (a : ℝ) where
  equation : ∀ x y : ℝ, a * x + y - 2 - a = 0

/-- The x-intercept of a line -/
noncomputable def x_intercept (a : ℝ) : ℝ := (2 + a) / a

/-- The y-intercept of a line -/
def y_intercept (a : ℝ) : ℝ := 2 + a

/-- The theorem stating that the x-intercept and y-intercept are equal if and only if a = -2 or a = 1 -/
theorem equal_intercepts (a : ℝ) :
  x_intercept a = y_intercept a ↔ a = -2 ∨ a = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_intercepts_l1205_120542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1205_120500

/-- The time taken to complete a piece of work given the conditions -/
theorem work_completion_time 
  (initial_men : ℝ)
  (initial_days : ℝ)
  (added_men : ℝ)
  (days_before_joining : ℝ)
  (h1 : initial_men > 0)
  (h2 : initial_days > 0)
  (h3 : added_men ≥ 0)
  (h4 : days_before_joining ≥ 0)
  (h5 : days_before_joining < initial_days) :
  let total_men := initial_men + added_men
  let work_rate := 1 / (initial_men * initial_days)
  let work_done := initial_men * work_rate * days_before_joining
  let remaining_work := 1 - work_done
  let remaining_days := remaining_work / (total_men * work_rate)
  days_before_joining + remaining_days = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1205_120500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_gt_g_plus_half_exists_a_min_value_three_l1205_120537

open Real

noncomputable section

-- Define the interval (0, e]
def I : Set ℝ := Set.Ioc 0 (Real.exp 1)

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := a * x - log x

def g (x : ℝ) : ℝ := (log x) / x

-- Statement 1
theorem f_gt_g_plus_half : ∀ x ∈ I, |f 1 x| > g x + 1/2 := by sorry

-- Statement 2
theorem exists_a_min_value_three :
  ∃ a : ℝ, (∀ x ∈ I, f a x ≥ 3) ∧ (∃ x ∈ I, f a x = 3) ∧ a = (Real.exp 1) ^ 2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_gt_g_plus_half_exists_a_min_value_three_l1205_120537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_intersection_distance_exists_l1205_120514

-- Define the basic geometric objects
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

def parallel (l1 l2 : Line) : Prop :=
  l1.direction = l2.direction

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def intersectionPoints (c : Circle) (l : Line) : Set (ℝ × ℝ) :=
  { p | ∃ t, p = (l.point.1 + t * l.direction.1, l.point.2 + t * l.direction.2) ∧ distance p c.center = c.radius }

-- The main theorem
theorem parallel_line_intersection_distance_exists 
  (S₁ S₂ : Circle) (l : Line) (a : ℝ) : 
  ∃ (l' : Line), parallel l l' ∧ 
    ∃ (p₁ p₂ : ℝ × ℝ), p₁ ∈ intersectionPoints S₁ l' ∧ 
                        p₂ ∈ intersectionPoints S₂ l' ∧ 
                        distance p₁ p₂ = a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_intersection_distance_exists_l1205_120514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_value_implies_x_value_l1205_120530

/-- Angle in the second quadrant -/
def SecondQuadrantAngle (α : Real) : Prop :=
  Real.pi / 2 < α ∧ α < Real.pi

/-- Point on the terminal side of an angle -/
def PointOnTerminalSide (x y α : Real) : Prop :=
  x = y * Real.tan α

theorem cosine_value_implies_x_value
  (α : Real)
  (x : Real)
  (h1 : SecondQuadrantAngle α)
  (h2 : PointOnTerminalSide x 4 α)
  (h3 : Real.cos α = (1/5) * x) :
  x = -3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_value_implies_x_value_l1205_120530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jose_land_share_l1205_120597

/-- Calculates the amount of land Jose will have after equal division -/
noncomputable def land_for_jose (total_land : ℝ) (num_siblings : ℕ) : ℝ :=
  total_land / (num_siblings + 1 : ℝ)

/-- Theorem stating that Jose will have 4000 square meters of land -/
theorem jose_land_share : 
  land_for_jose 20000 4 = 4000 := by
  -- Unfold the definition of land_for_jose
  unfold land_for_jose
  -- Simplify the expression
  simp
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jose_land_share_l1205_120597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_capacity_is_100_liters_l1205_120502

/-- Represents a water tank with a certain capacity -/
structure WaterTank where
  capacity : ℚ
  capacity_positive : capacity > 0

/-- The amount of water in the tank when it's at a certain percentage full -/
def water_amount (tank : WaterTank) (percentage : ℚ) : ℚ :=
  percentage * tank.capacity / 100

theorem tank_capacity_is_100_liters : ∃ tank : WaterTank, 
  water_amount tank 90 - water_amount tank 40 = 50 ∧ tank.capacity = 100 := by
  -- We'll construct a tank with capacity 100
  let tank : WaterTank := ⟨100, by norm_num⟩
  
  -- Show this tank satisfies our conditions
  have h1 : water_amount tank 90 - water_amount tank 40 = 50 := by
    simp [water_amount]
    norm_num
  
  have h2 : tank.capacity = 100 := rfl
  
  -- Prove the existence
  exact ⟨tank, h1, h2⟩

#check tank_capacity_is_100_liters

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_capacity_is_100_liters_l1205_120502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1205_120586

-- Define the function f(x) = 2x + 1/(x-1)
noncomputable def f (x : ℝ) : ℝ := 2 * x + 1 / (x - 1)

-- State the theorem
theorem min_value_of_f :
  (∀ x > 1, f x ≥ 2 * Real.sqrt 2 + 2) ∧
  (∃ x > 1, f x = 2 * Real.sqrt 2 + 2) :=
by
  sorry

#check min_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1205_120586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_frustum_from_pyramid_l1205_120540

-- Define the geometric bodies
inductive GeometricBody
| HexagonalPyramid
| HexagonalPrism
| RectangularPrism
| Cube

-- Define a type for shapes
inductive Shape
| Hexagon
| Rectangle
| Square

-- Define the concept of a plane being parallel to a shape
def is_parallel_to (plane : Type) (shape : Shape) : Prop := sorry

-- Define the concept of cutting a body with a plane
def cut_by (body : GeometricBody) (plane : Type) (result : Shape) : Prop := sorry

-- Define the concept of a frustum
def is_frustum (body : GeometricBody) : Prop :=
  ∃ (base_shape : Shape) (cutting_plane : Type),
    is_parallel_to cutting_plane base_shape ∧
    cut_by body cutting_plane base_shape

-- State the theorem
theorem hexagonal_frustum_from_pyramid :
  is_frustum GeometricBody.HexagonalPyramid →
  ∃ (cutting_plane : Type),
    is_parallel_to cutting_plane Shape.Hexagon ∧
    cut_by GeometricBody.HexagonalPyramid cutting_plane Shape.Hexagon :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_frustum_from_pyramid_l1205_120540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_obtuse_isosceles_triangle_l1205_120570

structure Quadrilateral where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  angle4 : ℝ
  sum_angles : angle1 + angle2 + angle3 + angle4 = 360
  has_right_angle : angle1 = 90 ∨ angle2 = 90 ∨ angle3 = 90 ∨ angle4 = 90
  has_120_angle : angle1 = 120 ∨ angle2 = 120 ∨ angle3 = 120 ∨ angle4 = 120

structure Diagonal where
  length : ℝ
  intersect_angle : ℝ

def diagonals_are_equal_and_perpendicular (d1 d2 : Diagonal) : Prop :=
  d1.length = d2.length ∧ d1.intersect_angle = 90

-- Define the concepts of obtuse isosceles triangle and subset of quadrilateral
def is_obtuse_isosceles_triangle (t : Set ℝ) : Prop := sorry

def is_subset_of_quadrilateral (t : Set ℝ) (q : Quadrilateral) : Prop := sorry

theorem no_obtuse_isosceles_triangle 
  (q : Quadrilateral) 
  (d1 d2 : Diagonal) 
  (h : diagonals_are_equal_and_perpendicular d1 d2) : 
  ¬∃ (t : Set ℝ), is_obtuse_isosceles_triangle t ∧ is_subset_of_quadrilateral t q :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_obtuse_isosceles_triangle_l1205_120570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_perimeter_l1205_120536

/-- An isosceles trapezoid with given dimensions -/
structure IsoscelesTrapezoid where
  AB : ℝ
  CD : ℝ
  height : ℝ
  ab_eq_10 : AB = 10
  cd_eq_18 : CD = 18
  height_eq_4 : height = 4

/-- The perimeter of an isosceles trapezoid -/
noncomputable def perimeter (t : IsoscelesTrapezoid) : ℝ :=
  t.AB + t.CD + 2 * Real.sqrt ((t.CD - t.AB)^2 / 4 + t.height^2)

/-- Theorem: The perimeter of the given isosceles trapezoid is 28 + 8√2 -/
theorem isosceles_trapezoid_perimeter (t : IsoscelesTrapezoid) :
  perimeter t = 28 + 8 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_perimeter_l1205_120536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_mean_BC_proof_l1205_120599

/-- Represents a pile of rocks -/
structure RockPile where
  weight : ℝ  -- Total weight of the pile
  count : ℝ   -- Number of rocks in the pile

/-- Calculates the mean weight of a rock pile -/
noncomputable def mean_weight (pile : RockPile) : ℝ := pile.weight / pile.count

/-- Combines two rock piles -/
def combine_piles (p1 p2 : RockPile) : RockPile :=
  { weight := p1.weight + p2.weight, count := p1.count + p2.count }

/-- The greatest possible integer mean weight of combined piles B and C -/
def max_mean_BC : ℕ := 62

theorem max_mean_BC_proof 
  (A B C : RockPile)
  (hA : mean_weight A = 30)
  (hB : mean_weight B = 55)
  (hAB : mean_weight (combine_piles A B) = 35)
  (hAC : mean_weight (combine_piles A C) = 32) :
  ∀ n : ℕ, n ≤ max_mean_BC → 
    n ≤ Int.floor (mean_weight (combine_piles B C)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_mean_BC_proof_l1205_120599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coplanar_iff_b_eq_a_squared_l1205_120573

/-- A point in 3D space represented by its coordinates -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The determinant of a 3x3 matrix represented by three 3D vectors -/
def det3 (v1 v2 v3 : ℝ × ℝ × ℝ) : ℝ :=
  let (a1, a2, a3) := v1
  let (b1, b2, b3) := v2
  let (c1, c2, c3) := v3
  a1 * (b2 * c3 - b3 * c2) - a2 * (b1 * c3 - b3 * c1) + a3 * (b1 * c2 - b2 * c1)

/-- Check if four points in 3D space are coplanar -/
def areCoplanar (p1 p2 p3 p4 : Point3D) : Prop :=
  let v1 := (p2.x - p1.x, p2.y - p1.y, p2.z - p1.z)
  let v2 := (p3.x - p1.x, p3.y - p1.y, p3.z - p1.z)
  let v3 := (p4.x - p1.x, p4.y - p1.y, p4.z - p1.z)
  det3 v1 v2 v3 = 0

theorem coplanar_iff_b_eq_a_squared (a b : ℝ) :
  areCoplanar
    (Point3D.mk 0 0 0)
    (Point3D.mk b a 0)
    (Point3D.mk 0 1 a)
    (Point3D.mk a 0 1) ↔
  b = a^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coplanar_iff_b_eq_a_squared_l1205_120573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l1205_120531

-- Define the circle C
noncomputable def circle_C (θ : Real) : Real × Real := (4 * Real.cos θ, 4 * Real.sin θ)

-- Define the line l
noncomputable def line_l (t : Real) : Real × Real := (1 + (Real.sqrt 3 / 2) * t, 2 + (1 / 2) * t)

-- Define the inclination angle
noncomputable def inclination_angle : Real := Real.pi / 6

-- Theorem statement
theorem intersection_distance_product :
  let P : Real × Real := (1, 2)
  let intersection_points := {A : Real × Real | ∃ t, line_l t = A ∧ ∃ θ, circle_C θ = A}
  ∀ A B, A ∈ intersection_points → B ∈ intersection_points → A ≠ B →
    (Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2)) *
    (Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2)) = 11 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l1205_120531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farm_plot_length_l1205_120529

/-- Represents the dimensions and rental information of a rectangular plot of farmland. -/
structure FarmPlot where
  width : ℚ  -- Width of the plot in feet
  monthlyRent : ℚ  -- Monthly rent in dollars
  ratePerAcre : ℚ  -- Rental rate per acre per month in dollars
  sqFtPerAcre : ℚ  -- Square feet per acre

/-- Calculates the length of a rectangular farm plot given its characteristics. -/
def calculatePlotLength (plot : FarmPlot) : ℚ :=
  let acres := plot.monthlyRent / plot.ratePerAcre
  let totalSqFt := acres * plot.sqFtPerAcre
  totalSqFt / plot.width

/-- Theorem stating that for the given plot characteristics, the length is 360 feet. -/
theorem farm_plot_length :
  let plot : FarmPlot := {
    width := 1210,
    monthlyRent := 600,
    ratePerAcre := 60,
    sqFtPerAcre := 43560
  }
  calculatePlotLength plot = 360 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farm_plot_length_l1205_120529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_incenter_lambda_l1205_120590

-- Define the hyperbola structure
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : a > 0 ∧ b > 0

-- Define the point type
def Point := ℝ × ℝ

-- Define the foci of the hyperbola
noncomputable def leftFocus (h : Hyperbola) : Point := sorry
noncomputable def rightFocus (h : Hyperbola) : Point := sorry

-- Define a predicate for a point being on the right branch of the hyperbola
def isOnRightBranch (h : Hyperbola) (p : Point) : Prop := sorry

-- Define the incenter of a triangle
noncomputable def incenter (p1 p2 p3 : Point) : Point := sorry

-- Define the area of a triangle
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ := sorry

-- Define the eccentricity of a hyperbola
noncomputable def eccentricity (h : Hyperbola) : ℝ := sorry

-- The main theorem
theorem hyperbola_incenter_lambda (h : Hyperbola) (p : Point) (lambda : ℝ) :
  isOnRightBranch h p →
  let f1 := leftFocus h
  let f2 := rightFocus h
  let m := incenter p f1 f2
  triangleArea m p f1 = triangleArea m p f2 + lambda * triangleArea m f1 f2 →
  eccentricity h = 3 →
  lambda = 1/3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_incenter_lambda_l1205_120590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_g_l1205_120560

/-- The function f(x) = x/(x^2) -/
noncomputable def f (x : ℝ) : ℝ := x / (x^2)

/-- The function g(x) = 1/x -/
noncomputable def g (x : ℝ) : ℝ := 1 / x

/-- Theorem stating that f and g are identical for all non-zero real numbers -/
theorem f_eq_g : ∀ (x : ℝ), x ≠ 0 → f x = g x := by
  intro x hx
  simp [f, g]
  field_simp [hx]
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_g_l1205_120560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_projection_sum_l1205_120541

/-- Given an equilateral triangle with side length a, the sum of the squares of the projections
of its sides onto any axis is equal to 3/2 * a^2. -/
theorem equilateral_triangle_projection_sum (a : ℝ) (h : a > 0) :
  ∀ α : ℝ, 
  (a * Real.cos α)^2 + (a * Real.cos (π/3 - α))^2 + (a * Real.cos (π/3 + α))^2 = (3/2) * a^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_projection_sum_l1205_120541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_distances_odd_l1205_120553

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Predicate to check if a number is an odd integer -/
def isOddInteger (n : ℝ) : Prop :=
  ∃ k : ℤ, n = 2 * k + 1

/-- Theorem: For any 4 points in a plane, at least one pair of points has a distance that is not an odd integer -/
theorem not_all_distances_odd (A B C D : Point) : 
  ¬(isOddInteger (distance A B) ∧ 
    isOddInteger (distance A C) ∧ 
    isOddInteger (distance A D) ∧ 
    isOddInteger (distance B C) ∧ 
    isOddInteger (distance B D) ∧ 
    isOddInteger (distance C D)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_distances_odd_l1205_120553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_pythagorean_triples_with_2013_l1205_120577

def isPythagoreanTriple (a b c : ℕ) : Prop := a^2 + b^2 = c^2

def isPythagoreanTripleWith2013 (b c : ℕ) : Prop := isPythagoreanTriple 2013 b c

noncomputable def countPythagoreanTriplesWith2013 : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => isPythagoreanTripleWith2013 p.1 p.2 = true) (Finset.product (Finset.range 10000) (Finset.range 10000))).card

theorem count_pythagorean_triples_with_2013 :
  countPythagoreanTriplesWith2013 = 13 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_pythagorean_triples_with_2013_l1205_120577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cattle_profit_calculation_l1205_120504

/-- Calculates the profit from a cattle business transaction -/
theorem cattle_profit_calculation 
  (num_cattle : ℕ) 
  (purchase_price feed_cost_percentage cattle_weight selling_price_per_pound : ℚ) : 
  (num_cattle = 100 ∧ 
   purchase_price = 40000 ∧ 
   feed_cost_percentage = 1/5 ∧ 
   cattle_weight = 1000 ∧ 
   selling_price_per_pound = 2) →
  (let feed_cost := purchase_price * (1 + feed_cost_percentage)
   let total_cost := purchase_price + feed_cost
   let revenue_per_cattle := cattle_weight * selling_price_per_pound
   let total_revenue := revenue_per_cattle * num_cattle
   let profit := total_revenue - total_cost
   profit = 112000) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cattle_profit_calculation_l1205_120504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_of_smaller_sphere_l1205_120585

/-- Represents the properties of a hollow sphere -/
structure HollowSphere where
  radius : ℝ
  weight : ℝ

/-- The surface area of a sphere given its radius -/
noncomputable def surfaceArea (r : ℝ) : ℝ := 4 * Real.pi * r^2

/-- The weight of a hollow sphere is directly proportional to its surface area -/
axiom weight_proportional_to_area {s1 s2 : HollowSphere} :
  s1.weight / surfaceArea s1.radius = s2.weight / surfaceArea s2.radius

/-- The theorem to be proved -/
theorem weight_of_smaller_sphere (s1 s2 : HollowSphere)
  (h1 : s1.radius = 0.15)
  (h2 : s2.radius = 0.3)
  (h3 : s2.weight = 32) :
  s1.weight = 8 := by
  sorry

#check weight_of_smaller_sphere

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_of_smaller_sphere_l1205_120585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asphalt_coverage_l1205_120596

/-- Calculates the area covered by each truckload of asphalt given the road dimensions, cost per truckload, tax rate, and total cost after tax. -/
theorem asphalt_coverage (road_length road_width : ℝ) (cost_per_truckload : ℝ) (tax_rate : ℝ) (total_cost_after_tax : ℝ) :
  road_length = 2000 →
  road_width = 20 →
  cost_per_truckload = 75 →
  tax_rate = 0.2 →
  total_cost_after_tax = 4500 →
  (road_length * road_width) / (total_cost_after_tax / (1 + tax_rate) / cost_per_truckload) = 800 := by
  sorry

#check asphalt_coverage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_asphalt_coverage_l1205_120596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapping_squares_area_l1205_120509

/-- The side length of each square sheet -/
def side_length : ℝ := 8

/-- The rotation angle of the middle sheet in radians -/
noncomputable def middle_rotation : ℝ := 20 * Real.pi / 180

/-- The rotation angle of the top sheet in radians -/
noncomputable def top_rotation : ℝ := 50 * Real.pi / 180

/-- The resulting polygon formed by overlapping the rotated squares -/
def resulting_polygon (s : ℝ) (θ₁ θ₂ : ℝ) : Set (ℝ × ℝ) := sorry

/-- The area of the resulting polygon -/
noncomputable def polygon_area (s : ℝ) (θ₁ θ₂ : ℝ) : ℝ := sorry

theorem overlapping_squares_area :
  polygon_area side_length middle_rotation top_rotation = 192 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapping_squares_area_l1205_120509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_correct_l1205_120563

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Calculates the slope given an angle in degrees -/
noncomputable def slopeFromAngle (angle : ℝ) : ℝ :=
  Real.tan (angle * Real.pi / 180)

/-- Checks if a point lies on a line -/
def pointOnLine (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.yIntercept

theorem line_equation_correct (x y : ℝ) :
  let slope := slopeFromAngle 60
  let line : Line := { slope := slope, yIntercept := 2 - slope * (-3) }
  pointOnLine line (-3) 2 ∧ 
  ∀ x y, pointOnLine line x y ↔ y - 2 = Real.sqrt 3 * (x + 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_correct_l1205_120563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surf_festival_weighted_average_l1205_120576

/-- The Rip Curl Myrtle Beach Surf Festival problem -/
theorem surf_festival_weighted_average (total_surfers : ℕ) (total_days : ℕ) 
  (ratio_first_two_days : Rat) (ratio_last_two_days : Rat) :
  total_surfers = 12000 →
  total_days = 4 →
  ratio_first_two_days = (5 : ℚ) / (7 : ℚ) →
  ratio_last_two_days = (3 : ℚ) / (2 : ℚ) →
  (total_surfers : ℚ) / total_days = 3000 := by
  sorry

#check surf_festival_weighted_average

end NUMINAMATH_CALUDE_ERRORFEEDBACK_surf_festival_weighted_average_l1205_120576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1205_120557

-- Define the function f
noncomputable def f (x φ : ℝ) : ℝ := 2 * Real.sin (2 * x + φ) + 1

-- State the theorem
theorem function_properties :
  ∀ φ : ℝ, -π/2 < φ → φ < 0 → f 0 φ = 0 →
  ∃ (k : ℤ),
    (φ = -π/6) ∧
    (∀ x : ℝ, f x φ ≤ 3) ∧
    (f (↑k * π + 2*π/3) φ = 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1205_120557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_4_equals_negative_5_l1205_120574

def sequence_a : ℕ → ℤ
  | 0 => 1  -- Adding the base case for 0
  | 1 => 1
  | (n + 1) => sequence_a n - n

theorem a_4_equals_negative_5 : sequence_a 4 = -5 := by
  -- Expand the definition of sequence_a for n = 4
  have h1 : sequence_a 4 = sequence_a 3 - 3 := rfl
  have h2 : sequence_a 3 = sequence_a 2 - 2 := rfl
  have h3 : sequence_a 2 = sequence_a 1 - 1 := rfl
  have h4 : sequence_a 1 = 1 := rfl

  -- Calculate step by step
  calc
    sequence_a 4 = sequence_a 3 - 3 := h1
    _ = (sequence_a 2 - 2) - 3 := by rw [h2]
    _ = ((sequence_a 1 - 1) - 2) - 3 := by rw [h3]
    _ = ((1 - 1) - 2) - 3 := by rw [h4]
    _ = (0 - 2) - 3 := by ring
    _ = -2 - 3 := by ring
    _ = -5 := by ring

  -- Alternative concise proof
  -- simp [sequence_a]
  -- ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_4_equals_negative_5_l1205_120574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_factor_of_36_is_one_fourth_l1205_120517

/-- The set of positive integers less than or equal to 36 -/
def integers_up_to_36 : Finset ℕ := Finset.range 36 \ {0}

/-- The set of factors of 36 -/
def factors_of_36 : Finset ℕ := Finset.filter (·∣36) integers_up_to_36

/-- The probability of a positive integer less than or equal to 36 being a factor of 36 -/
def probability_factor_of_36 : ℚ :=
  (factors_of_36.card : ℚ) / (integers_up_to_36.card : ℚ)

theorem probability_factor_of_36_is_one_fourth :
  probability_factor_of_36 = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_factor_of_36_is_one_fourth_l1205_120517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1205_120558

noncomputable def f (c : ℝ) (x : ℝ) : ℝ :=
  if 0 < x ∧ x < c then c * x + 1
  else if c ≤ x ∧ x < 1 then 2^(-x/c^2) + 1
  else 0  -- undefined for other x values

theorem function_properties (c : ℝ) :
  (0 < c ∧ c < 1) →
  (f c (c^2) = 9/8) →
  (c = 1/2) ∧
  (∀ x : ℝ, f (1/2) x > Real.sqrt 2 / 8 + 1 ↔ Real.sqrt 2 / 4 < x ∧ x < 5/8) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1205_120558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_sin_is_arcsin_l1205_120578

-- Define the original function
noncomputable def f (x : ℝ) := Real.sin x

-- Define the domain of the original function
def domain : Set ℝ := { x | -Real.pi / 2 ≤ x ∧ x ≤ Real.pi / 2 }

-- Define the inverse function
noncomputable def g (x : ℝ) := Real.arcsin x

-- Define the range of the original function (domain of the inverse function)
def range : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }

-- Theorem stating that g is the inverse of f on the given domain and range
theorem inverse_sin_is_arcsin :
  ∀ x ∈ domain, ∀ y ∈ range,
    f x = y ↔ g y = x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_sin_is_arcsin_l1205_120578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_numbers_with_2_or_5_odd_ending_l1205_120532

theorem three_digit_numbers_with_2_or_5_odd_ending : 
  let total_three_digit : Nat := 900
  let digits_without_2_or_5 : Finset Nat := {0, 1, 3, 4, 6, 7, 8, 9}
  let odd_digits_without_2_or_5 : Finset Nat := {1, 3, 7, 9}
  let numbers_without_2_or_5_odd_ending : Nat := 7 * 8 * 4
  total_three_digit - numbers_without_2_or_5_odd_ending = 676 :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_numbers_with_2_or_5_odd_ending_l1205_120532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_couch_price_before_tax_l1205_120525

/-- Represents the price of furniture items and calculates the total cost --/
structure FurniturePrices where
  chair : ℚ
  table : ℚ := 3 * chair
  couch : ℚ := 5 * table
  bookshelf : ℚ := couch / 2
  tax_rate : ℚ := 1 / 10

  total_cost : ℚ := (chair + table + couch + bookshelf) * (1 + tax_rate)

/-- Theorem stating that given the conditions, the price of the couch before tax is $288.75 --/
theorem couch_price_before_tax (prices : FurniturePrices) :
  prices.total_cost = 561 → prices.couch = 2887500 / 10000 := by
  sorry

/-- Compute the couch price for the given chair price --/
def compute_couch_price (chair_price : ℚ) : ℚ :=
  let prices : FurniturePrices := { chair := chair_price }
  prices.couch

#eval compute_couch_price (1925 / 100)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_couch_price_before_tax_l1205_120525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_theorem_l1205_120521

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  pos_a : a > 0
  pos_b : b > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  -- We'll define a line by its slope and y-intercept
  slope : ℝ
  intercept : ℝ

/-- The left focus of a hyperbola -/
def left_focus (h : Hyperbola a b) : Point := sorry

/-- Check if two points are symmetric about a line -/
def symmetric_about (p1 p2 : Point) (l : Line) : Prop := sorry

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- Check if a line passes through a point -/
def passes_through (l : Line) (p : Point) : Prop := 
  p.y = l.slope * p.x + l.intercept

/-- Theorem: If points (a, 0) and (0, b) are symmetric about a line passing through
    the left focus of a hyperbola x²/a² - y²/b² = 1, then its eccentricity is √3 + 1 -/
theorem hyperbola_eccentricity_theorem 
  (a b : ℝ) (h : Hyperbola a b) (l : Line) :
  symmetric_about (Point.mk a 0) (Point.mk 0 b) l →
  passes_through l (left_focus h) →
  eccentricity h = Real.sqrt 3 + 1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_theorem_l1205_120521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_two_solutions_l1205_120503

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x * (x - 1) else Real.log (1 - x) / Real.log 3

theorem f_eq_two_solutions :
  ∃! (s : Set ℝ), s = {m : ℝ | f m = 2} ∧ s = {-8, 2} := by
  sorry

#check f_eq_two_solutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_two_solutions_l1205_120503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_right_angle_l1205_120552

theorem triangle_right_angle (A B C : ℝ) (h : Real.sin A * Real.cos B = 0) : 
  ∃ (θ : ℝ), θ = Real.pi / 2 ∧ (θ = A ∨ θ = B ∨ θ = C) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_right_angle_l1205_120552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1205_120549

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * (Real.cos x)^2 + 2 * Real.sin x * Real.cos x - Real.sqrt 3

/-- Theorem stating the two properties of f(x) to be proved -/
theorem f_properties :
  (∀ x, f x = 2 * Real.cos (2 * x - π / 6)) ∧
  (f (2023 * π) = Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1205_120549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_iff_m_ge_one_l1205_120526

-- Define the function f(x) with parameter m
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1/2) * m * x^2 - 2*x + Real.log x

-- State the theorem
theorem f_increasing_iff_m_ge_one :
  ∀ m : ℝ, (∀ x > 0, StrictMono (f m)) ↔ m ≥ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_iff_m_ge_one_l1205_120526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horse_catch_up_problem_l1205_120520

/-- Represents the problem of a fast horse catching up to a slow horse --/
theorem horse_catch_up_problem (x : ℝ) :
  240 * x = 150 * (x + 12) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horse_catch_up_problem_l1205_120520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_identification_l1205_120583

noncomputable section

-- Define what a fraction is
def is_fraction (expr : ℝ → ℝ) : Prop :=
  ∃ (num : ℝ) (denom : ℝ → ℝ), 
    (∀ x, expr x = num / (denom x)) ∧ 
    (∀ x, denom x ≠ 0) ∧
    (∀ x, ¬(∃ k, denom x = k * x + Real.pi - 3)) ∧  -- not a monomial
    (∀ x, ¬(∃ k, denom x = k))  -- not a constant

-- Define the expressions
def expr_A (a : ℝ) : ℝ := 1 / (2 - a)
def expr_B (x : ℝ) : ℝ := x / (Real.pi - 3)
def expr_C (y : ℝ) : ℝ := -y / 5
def expr_D (x y : ℝ) : ℝ := x / 2 + y

-- State the theorem
theorem fraction_identification :
  is_fraction expr_A ∧ 
  ¬is_fraction expr_B ∧ 
  ¬is_fraction expr_C ∧ 
  ¬is_fraction (λ x ↦ expr_D x x) := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_identification_l1205_120583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_odd_in_sequence_l1205_120595

/-- Given a sequence of 39 consecutive odd numbers with sum 1989, 
    the largest number in the sequence is 89. -/
theorem largest_odd_in_sequence : 
  ∀ (seq : List ℕ), 
    seq.length = 39 ∧ 
    (∀ i, i < seq.length - 1 → seq[i]?.isSome ∧ seq[i+1]?.isSome ∧ 
      (∃ (a b : ℕ), seq[i]? = some a ∧ seq[i+1]? = some b ∧ Odd a ∧ Odd b ∧ b = a + 2)) ∧
    seq.sum = 1989 →
    seq.maximum? = some 89 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_odd_in_sequence_l1205_120595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_decimal_to_fraction_l1205_120544

theorem periodic_decimal_to_fraction :
  (∃ (x : ℚ), x = 2 + 6/99) ∧ (2/99 = 2/99) →
  (∃ (y : ℚ), y = 2 + 6/99 ∧ y = 68/33) :=
by
  intro h
  rcases h with ⟨⟨x, hx⟩, _⟩
  use x
  constructor
  · exact hx
  · calc
      x = 2 + 6/99 := hx
      _ = (2 * 99 + 6) / 99 := by ring
      _ = 204 / 99 := by norm_num
      _ = 68 / 33 := by norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_decimal_to_fraction_l1205_120544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_ef_sum_l1205_120592

/-- Square ABCD with side length and points E, F on AB -/
structure SquareEF where
  side_length : ℝ
  E : ℝ
  F : ℝ

/-- Properties of the square and points E, F -/
def SquareProperties (s : SquareEF) : Prop :=
  s.side_length = 900 ∧
  0 < s.E ∧ s.E < s.F ∧ s.F < s.side_length ∧
  Real.sqrt 2 / 2 * (s.F - s.E) = (s.side_length / 2 - s.E) ∧
  s.F - s.E = 400

/-- Expression for BF in terms of p, q, and r -/
def BFExpression (s : SquareEF) (p q r : ℕ) : Prop :=
  s.side_length - s.F = p + q * Real.sqrt r ∧
  0 < p ∧ 0 < q ∧ 0 < r ∧
  ∀ (x : ℕ), x > 1 → Nat.Prime x → ¬(x * x ∣ r)

/-- Main theorem -/
theorem square_ef_sum (s : SquareEF) (p q r : ℕ) :
  SquareProperties s → BFExpression s p q r → p + q + r = 307 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_ef_sum_l1205_120592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_sum_l1205_120594

theorem complex_power_sum (z : ℂ) (h : z + z⁻¹ = 1) : z^12 + (z⁻¹)^12 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_sum_l1205_120594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octal_sum_proof_l1205_120591

/-- Represents a number in base 8 -/
structure OctalNumber where
  value : ℕ

/-- Converts an OctalNumber to its decimal (ℕ) representation -/
def octal_to_decimal (n : OctalNumber) : ℕ := sorry

/-- Converts a decimal (ℕ) to its OctalNumber representation -/
def decimal_to_octal (n : ℕ) : OctalNumber := sorry

instance : OfNat OctalNumber n where
  ofNat := decimal_to_octal n

/-- Sum of arithmetic sequence in base 8 -/
def octal_arithmetic_sum (a l n : OctalNumber) : OctalNumber :=
  decimal_to_octal ((octal_to_decimal n * (octal_to_decimal a + octal_to_decimal l)) / 2)

theorem octal_sum_proof :
  let a : OctalNumber := 1
  let l : OctalNumber := 32
  let n : OctalNumber := 32
  octal_arithmetic_sum a l n = 633 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octal_sum_proof_l1205_120591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_inequality_l1205_120511

theorem trigonometric_inequality (x : ℝ) (h : x ∈ Set.Ioo (-1/2) 0) :
  Real.cos ((x + 1) * π) < Real.sin (Real.cos (x * π)) ∧ 
  Real.sin (Real.cos (x * π)) < Real.cos (Real.sin (x * π)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_inequality_l1205_120511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_rate_satisfies_equation_compound_interest_rate_approx_close_l1205_120545

/-- The rate of compound interest that satisfies the given conditions -/
noncomputable def compound_interest_rate : ℝ := 
  let simple_interest_principal := 3225
  let simple_interest_rate := 8
  let simple_interest_time := 5
  let compound_interest_principal := 8000
  let compound_interest_time := 2
  Real.sqrt (1 + (516 / 8000)) - 1

/-- Theorem stating that the compound interest rate satisfies the given equation -/
theorem compound_interest_rate_satisfies_equation : 
  let simple_interest_principal := 3225
  let simple_interest_rate := 8
  let simple_interest_time := 5
  let compound_interest_principal := 8000
  let compound_interest_time := 2
  (simple_interest_principal * simple_interest_rate * simple_interest_time) / 100 = 
  (1/2) * compound_interest_principal * ((1 + compound_interest_rate/100)^compound_interest_time - 1) :=
by
  sorry

/-- Approximation of the compound interest rate -/
def compound_interest_rate_approx : ℝ := 3.17

/-- Theorem stating that the approximate rate is close to the actual rate -/
theorem compound_interest_rate_approx_close :
  abs (compound_interest_rate - compound_interest_rate_approx) < 0.01 :=
by
  sorry

#eval compound_interest_rate_approx

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_rate_satisfies_equation_compound_interest_rate_approx_close_l1205_120545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1205_120548

/-- Ellipse (C) with properties as described in the problem -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0
  h_AB : 2 * b^2 / a = 8 / Real.sqrt 5
  h_cos : (2 * a^2 - 4 * (a^2 - b^2)) / (2 * a^2) = 3 / 5

/-- The dot product of vectors F₂M and F₂N -/
noncomputable def dot_product (e : Ellipse) (m : ℝ) : ℝ :=
  -4 + 61 / (4 * m^2 + 5)

theorem ellipse_properties (e : Ellipse) :
  e.a = Real.sqrt 5 ∧ e.b = 2 ∧
  (∀ x y, x^2 / 5 + y^2 / 4 = 1 ↔ x^2 / e.a^2 + y^2 / e.b^2 = 1) ∧
  (∀ m, -4 ≤ dot_product e m ∧ dot_product e m ≤ 41 / 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1205_120548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_BC_fraction_of_AD_l1205_120593

-- Define the points and line segments
variable (A B C D E : EuclideanSpace ℝ (Fin 2))
variable (AB BC CD DE EA AD : ℝ)

-- State the conditions
axiom points_on_line : (B - A).isParallelTo (D - A) ∧ 
                       (C - A).isParallelTo (D - A) ∧ 
                       (E - A).isParallelTo (D - A)
axiom AB_length : AB = 3 * (AD - AB)
axiom AC_length : dist A C = 5 * CD
axiom DE_length : DE = 2 * EA

-- Define AD as the sum of its parts
axiom AD_sum : AD = AB + (AD - AB)

-- Define the theorem to be proved
theorem BC_fraction_of_AD : BC = (1 / 12) * AD := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_BC_fraction_of_AD_l1205_120593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l1205_120501

noncomputable section

-- Define the coefficients of the two lines
def a : ℝ := 3
def b : ℝ := -2
def c₁ : ℝ := -5
def c₂ : ℝ := 3

-- Define the distance formula
noncomputable def distance (a b c₁ c₂ : ℝ) : ℝ :=
  |c₂ - c₁| / (Real.sqrt (a^2 + b^2))

-- Theorem statement
theorem parallel_lines_distance :
  distance a b c₁ c₂ = Real.sqrt 13 / 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l1205_120501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_axes_symmetry_range_l1205_120523

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 4)

theorem two_axes_symmetry_range (ω : ℝ) :
  ω > 0 ∧
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 0 ≤ x₁ ∧ x₁ ≤ Real.pi ∧ 0 ≤ x₂ ∧ x₂ ≤ Real.pi ∧
    (∀ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi → f ω (x₁ - x) = f ω (x₁ + x) ∧ f ω (x₂ - x) = f ω (x₂ + x)) ∧
    (∀ (x₃ : ℝ), x₃ ≠ x₁ ∧ x₃ ≠ x₂ → ∃ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi ∧ f ω (x₃ - x) ≠ f ω (x₃ + x))) ↔
  5 / 4 ≤ ω ∧ ω < 9 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_axes_symmetry_range_l1205_120523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_primes_under_20_l1205_120559

theorem sum_of_primes_under_20 : 
  (Finset.filter (fun n => Nat.Prime n ∧ n < 20) (Finset.range 20)).sum id = 77 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_primes_under_20_l1205_120559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_negative_three_l1205_120565

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x - 6

noncomputable def g (x : ℝ) : ℝ := 3 * ((4⁻¹ * (x + 6)))^2 + 4 * ((4⁻¹ * (x + 6))) - 2

-- State the theorem
theorem g_of_negative_three : g (-3) = 43 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_negative_three_l1205_120565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_a_in_range_l1205_120539

open Real

/-- The function f(x) in the inequality -/
noncomputable def f (a x : ℝ) : ℝ := (1/2) * x^2 + (1-a) * x - a * log x

/-- The theorem to prove -/
theorem inequality_holds_iff_a_in_range (a : ℝ) : 
  (a > 0 ∧ ∀ x > 0, f a x > 2*a - (3/2)*a^2) ↔ (0 < a ∧ a ≠ 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_a_in_range_l1205_120539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_words_l1205_120533

/-- Represents the Antarctican language --/
structure AntarcticanLanguage where
  alphabet : Finset Char
  words : Finset (Char × Char × Char)

/-- The properties of the Antarctican language --/
def valid_language (lang : AntarcticanLanguage) : Prop :=
  (lang.alphabet.card = 16) ∧
  (∀ w ∈ lang.words, w.1 ≠ w.2.2) ∧
  (∀ w ∈ lang.words, w.1 ∈ lang.alphabet ∧ w.2.1 ∈ lang.alphabet ∧ w.2.2 ∈ lang.alphabet)

/-- The theorem stating the maximum number of words --/
theorem max_words (lang : AntarcticanLanguage) (h : valid_language lang) :
  lang.words.card ≤ 1024 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_words_l1205_120533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_tetrahedra_inequality_l1205_120519

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  edge_length : ℝ
  edge_length_pos : edge_length > 0

/-- The circumradius of a regular tetrahedron -/
noncomputable def circumradius (t : RegularTetrahedron) : ℝ :=
  t.edge_length * Real.sqrt 6 / 4

/-- The inradius of a regular tetrahedron -/
noncomputable def inradius (t : RegularTetrahedron) : ℝ :=
  t.edge_length * Real.sqrt 6 / 12

/-- Two regular tetrahedra where one is inscribed in the other -/
structure InscribedTetrahedra where
  inner : RegularTetrahedron
  outer : RegularTetrahedron
  inscribed : circumradius inner ≥ inradius outer

/-- Theorem: If a regular tetrahedron is inscribed in another regular tetrahedron
    such that each vertex of the inner tetrahedron lies on a face of the outer tetrahedron,
    then 3 times the edge length of the inner tetrahedron is greater than or equal to
    the edge length of the outer tetrahedron. -/
theorem inscribed_tetrahedra_inequality (t : InscribedTetrahedra) :
    3 * t.inner.edge_length ≥ t.outer.edge_length := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_tetrahedra_inequality_l1205_120519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_classification_l1205_120588

/-- A complex number z is defined as m + 1 + (m - 1)i, where m is a real number -/
def z (m : ℝ) : ℂ := m + 1 + (m - 1) * Complex.I

/-- Theorem stating the conditions for z to be real, complex, or pure imaginary -/
theorem z_classification (m : ℝ) : 
  (z m ∈ Set.range (Complex.ofReal) ↔ m = 1) ∧ 
  (z m ∈ {w : ℂ | w.im ≠ 0} ↔ m ≠ 1) ∧ 
  (z m = Complex.I * (z m).im ↔ m = -1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_classification_l1205_120588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sunscreen_cost_theorem_l1205_120582

/-- The cost of each bottle of sunscreen before the discount -/
def cost_per_bottle : ℚ := 30

/-- The number of bottles purchased for a year -/
def bottles_per_year : ℕ := 12

/-- The discount rate applied to the purchase -/
def discount_rate : ℚ := 3 / 10

/-- The total cost after applying the discount -/
def total_cost_after_discount : ℚ := 252

theorem sunscreen_cost_theorem :
  cost_per_bottle * (1 - discount_rate) * (bottles_per_year : ℚ) = total_cost_after_discount :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sunscreen_cost_theorem_l1205_120582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_existence_min_n_is_minimal_min_n_correct_l1205_120568

/-- The minimum positive integer n for which the binomial expansion of (√x + 3/∛x)^n contains a constant term -/
def min_n : ℕ := 5

/-- The expression (√x + 3/∛x)^n -/
noncomputable def expression (x : ℝ) (n : ℕ) : ℝ := (Real.sqrt x + 3 / Real.rpow x (1/3)) ^ n

theorem constant_term_existence (x : ℝ) (n : ℕ) (h : n ≥ min_n) :
  ∃ k : ℕ, k ≤ n ∧ 3 * n = 5 * k :=
by sorry

theorem min_n_is_minimal :
  ∀ m : ℕ, m < min_n → ¬∃ k : ℕ, k ≤ m ∧ 3 * m = 5 * k :=
by sorry

theorem min_n_correct :
  min_n = 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_existence_min_n_is_minimal_min_n_correct_l1205_120568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_ratio_l1205_120508

-- Define the lengths of the two pieces of wire
variable (a b : ℝ)

-- Define that a and b are positive
variable (ha : a > 0) (hb : b > 0)

-- Define the area of the square formed by a
noncomputable def square_area (a : ℝ) : ℝ := (a / 4) ^ 2

-- Define the area of the regular octagon formed by b
noncomputable def octagon_area (b : ℝ) : ℝ := 2 * ((b / 8) ^ 2) * (1 + Real.sqrt 2)

-- State the theorem
theorem wire_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  square_area a = octagon_area b →
  a / b = Real.sqrt (2 + Real.sqrt 2) / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_ratio_l1205_120508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_digit_square_with_sqrt_last_three_digits_l1205_120515

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

def last_three_digits (n : ℕ) : ℕ := n % 1000

def first_three_digits (n : ℕ) : ℕ := n / 1000

theorem six_digit_square_with_sqrt_last_three_digits :
  ∀ n : ℕ, is_six_digit n → Nat.sqrt n * Nat.sqrt n = n →
  (∃ m : ℕ, n = m^2 ∧ last_three_digits n = m) →
  (n = 141376 ∨ n = 390625) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_digit_square_with_sqrt_last_three_digits_l1205_120515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_circle_l1205_120564

/-- Given a circle with center (5,3) and a point (8,8) on the circle,
    the slope of the tangent line at (8,8) is -3/5. -/
theorem tangent_slope_circle (center : ℝ × ℝ) (point : ℝ × ℝ) : 
  center = (5, 3) → point = (8, 8) → 
  (((point.2 - center.2) / (point.1 - center.1))⁻¹ * (-1)) = -3/5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_circle_l1205_120564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1205_120555

/-- Defines an ellipse with foci on the x-axis -/
def defines_ellipse_with_foci_on_x_axis (f : ℝ → ℝ → Prop) : Prop :=
  ∃ a b : ℝ, a > b ∧ b > 0 ∧ ∀ x y : ℝ, f x y ↔ x^2/a^2 + y^2/b^2 = 1

/-- The range of values for m given the conditions -/
theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + (1/2)*m = 0) ∧ 
  defines_ellipse_with_foci_on_x_axis (λ x y : ℝ ↦ x^2/(m+3) + y^2/4 = 1) → 
  1 < m ∧ m ≤ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1205_120555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_linear_functions_minimum_value_l1205_120567

/-- Two linear functions whose graphs are parallel lines not parallel to the coordinate axes -/
def parallel_linear_functions (f g : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ (∀ x, f x = a * x + b) ∧ (∀ x, g x = a * x + c)

/-- The minimum value of a quadratic function -/
def quadratic_minimum (h : ℝ → ℝ) (v : ℝ) : Prop :=
  ∃ a b c : ℝ, a > 0 ∧ (∀ x, h x = a * x^2 + b * x + c) ∧ v = c - b^2 / (4 * a)

theorem parallel_linear_functions_minimum_value 
  (f g : ℝ → ℝ) 
  (h : parallel_linear_functions f g) 
  (min_f_sq_plus_g : quadratic_minimum (λ x ↦ (f x)^2 + g x) (-6)) :
  quadratic_minimum (λ x ↦ (g x)^2 + f x) (11/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_linear_functions_minimum_value_l1205_120567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pie_ingredients_theorem_l1205_120527

theorem pie_ingredients_theorem (total_pies : ℕ) 
  (chocolate_fraction marshmallow_fraction cayenne_fraction soy_fraction : ℚ)
  (h_total : total_pies = 48)
  (h_chocolate : chocolate_fraction = 1/3)
  (h_marshmallow : marshmallow_fraction = 1/2)
  (h_cayenne : cayenne_fraction = 3/8)
  (h_soy : soy_fraction = 1/8) :
  ∃ (plain_pies : ℕ), 
    plain_pies ≤ total_pies ∧
    plain_pies = total_pies - max 
      (Nat.floor (chocolate_fraction * total_pies)) 
      (max (Nat.floor (marshmallow_fraction * total_pies)) 
        (max (Nat.floor (cayenne_fraction * total_pies)) 
          (Nat.floor (soy_fraction * total_pies)))) ∧
    plain_pies = 24 ∧
    ∀ (other_plain : ℕ), 
      other_plain ≤ total_pies → 
      other_plain ≤ plain_pies :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pie_ingredients_theorem_l1205_120527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_factor_of_249_l1205_120524

theorem greatest_prime_factor_of_249 : 
  (Nat.factors 249).maximum? = some 19 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_factor_of_249_l1205_120524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_square_root_problem_l1205_120535

theorem cube_root_square_root_problem (a b c : ℝ) : 
  (a - 4) ^ (1/3) = 1 →
  Real.sqrt b = 2 →
  c = Int.floor (Real.sqrt 11) →
  a = 5 ∧ b = 4 ∧ c = 3 ∧ (Real.sqrt (2*a - 3*b + c) = 1 ∨ Real.sqrt (2*a - 3*b + c) = -1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_square_root_problem_l1205_120535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_B_in_A_l1205_120551

theorem complement_of_B_in_A : Set ℕ := by
  -- Define sets A and B
  let A : Set ℕ := {0, 2, 4, 6, 8, 10}
  let B : Set ℕ := {4, 8}

  -- Define the complement of B in A
  let complement_B_in_A := A \ B

  -- Prove that the complement equals {0, 2, 6, 10}
  have : complement_B_in_A = {0, 2, 6, 10} := by
    -- The proof would go here, but we'll use sorry for now
    sorry

  -- Return the result
  exact {0, 2, 6, 10}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_B_in_A_l1205_120551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1205_120580

-- Define the function f
noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x + m / x

-- State the theorem
theorem f_properties (m : ℝ) (h : f 1 m = 2) :
  -- 1. f is an odd function
  (∀ x, f (-x) m = -(f x m)) ∧
  -- 2. f is increasing on (1, +∞)
  (∀ x y, 1 < x → x < y → f x m < f y m) ∧
  -- 3. When f(a) > 2, a ∈ (0, 1) ∪ (1, +∞)
  (∀ a, f a m > 2 → (0 < a ∧ a < 1) ∨ (1 < a)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1205_120580
