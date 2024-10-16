import Mathlib

namespace NUMINAMATH_CALUDE_complex_sum_theorem_l3495_349574

theorem complex_sum_theorem (a b c d e f g h : ℝ) : 
  b = 2 → 
  g = -(a + c + e) → 
  (a + b * Complex.I) + (c + d * Complex.I) + (e + f * Complex.I) + (g + h * Complex.I) = 3 * Complex.I → 
  d + f + h = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_theorem_l3495_349574


namespace NUMINAMATH_CALUDE_wrapping_paper_area_formula_l3495_349520

/-- The area of wrapping paper required for a box -/
def wrapping_paper_area (w : ℝ) (h : ℝ) : ℝ :=
  (4 * w + h) * (2 * w + h)

/-- Theorem: The area of the wrapping paper for a box with width w, length 2w, and height h -/
theorem wrapping_paper_area_formula (w : ℝ) (h : ℝ) :
  wrapping_paper_area w h = 8 * w^2 + 6 * w * h + h^2 := by
  sorry

end NUMINAMATH_CALUDE_wrapping_paper_area_formula_l3495_349520


namespace NUMINAMATH_CALUDE_trigonometric_identities_l3495_349548

noncomputable def α : ℝ := Real.arcsin (2/3 - Real.sqrt (5/9))
noncomputable def β : ℝ := Real.arctan 2

theorem trigonometric_identities 
  (h1 : Real.sin α + Real.cos α = 2/3)
  (h2 : π/2 < α ∧ α < π)
  (h3 : Real.tan β = 2) :
  (Real.sin (3*π/2 - α) * Real.cos (-π/2 - α) = -5/18) ∧
  ((1 / Real.sin (π - α)) - (1 / Real.cos (2*π - α)) + 
   (Real.sin β - Real.cos β) / (2*Real.sin β + Real.cos β) = (6*Real.sqrt 14 + 1)/5) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l3495_349548


namespace NUMINAMATH_CALUDE_point_c_coordinates_l3495_349583

/-- Given point A, vector AB, and vector BC in a 2D Cartesian coordinate system,
    prove that the coordinates of point C are as calculated. -/
theorem point_c_coordinates (A B C : ℝ × ℝ) (AB BC : ℝ × ℝ) :
  A = (0, 1) →
  AB = (-4, -3) →
  BC = (-7, -4) →
  B = (A.1 + AB.1, A.2 + AB.2) →
  C = (B.1 + BC.1, B.2 + BC.2) →
  C = (-11, -6) := by
  sorry

end NUMINAMATH_CALUDE_point_c_coordinates_l3495_349583


namespace NUMINAMATH_CALUDE_magnitude_of_complex_reciprocal_l3495_349591

open Complex

theorem magnitude_of_complex_reciprocal (z : ℂ) : z = (1 : ℂ) / (1 - I) → abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_reciprocal_l3495_349591


namespace NUMINAMATH_CALUDE_brian_drove_200_miles_more_l3495_349509

/-- Represents the driving scenario of Mike, Steve, and Brian --/
structure DrivingScenario where
  t : ℝ  -- Mike's driving time
  s : ℝ  -- Mike's driving speed
  d : ℝ  -- Mike's driving distance
  steve_distance : ℝ  -- Steve's driving distance
  brian_distance : ℝ  -- Brian's driving distance

/-- The conditions of the driving scenario --/
def scenario_conditions (scenario : DrivingScenario) : Prop :=
  scenario.d = scenario.s * scenario.t ∧  -- Mike's distance equation
  scenario.steve_distance = (scenario.s + 6) * (scenario.t + 1.5) ∧  -- Steve's distance equation
  scenario.brian_distance = (scenario.s + 12) * (scenario.t + 3) ∧  -- Brian's distance equation
  scenario.steve_distance = scenario.d + 90  -- Steve drove 90 miles more than Mike

/-- The theorem stating that Brian drove 200 miles more than Mike --/
theorem brian_drove_200_miles_more (scenario : DrivingScenario) 
  (h : scenario_conditions scenario) : 
  scenario.brian_distance = scenario.d + 200 := by
  sorry


end NUMINAMATH_CALUDE_brian_drove_200_miles_more_l3495_349509


namespace NUMINAMATH_CALUDE_same_color_pairs_l3495_349507

def white_socks : ℕ := 5
def brown_socks : ℕ := 6
def blue_socks : ℕ := 3
def red_socks : ℕ := 2

def total_socks : ℕ := white_socks + brown_socks + blue_socks + red_socks

theorem same_color_pairs : 
  (Nat.choose white_socks 2) + (Nat.choose brown_socks 2) + 
  (Nat.choose blue_socks 2) + (Nat.choose red_socks 2) = 29 := by
sorry

end NUMINAMATH_CALUDE_same_color_pairs_l3495_349507


namespace NUMINAMATH_CALUDE_additional_round_trips_l3495_349580

/-- Represents the number of passengers on a one-way trip -/
def one_way_passengers : ℕ := 100

/-- Represents the number of passengers on a return trip -/
def return_passengers : ℕ := 60

/-- Represents the total number of passengers transported that day -/
def total_passengers : ℕ := 640

/-- Calculates the number of passengers in one round trip -/
def passengers_per_round_trip : ℕ := one_way_passengers + return_passengers

/-- Theorem: The number of additional round trips is 3 -/
theorem additional_round_trips :
  (total_passengers - passengers_per_round_trip) / passengers_per_round_trip = 3 := by
  sorry

end NUMINAMATH_CALUDE_additional_round_trips_l3495_349580


namespace NUMINAMATH_CALUDE_sum_of_digits_of_f_l3495_349578

/-- The number of digits in (10^2020 + 2020)^2 when written out in full -/
def num_digits : ℕ := 4041

/-- The function that calculates (10^2020 + 2020)^2 -/
def f : ℕ := (10^2020 + 2020)^2

/-- The sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_of_f : sum_of_digits f = 25 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_f_l3495_349578


namespace NUMINAMATH_CALUDE_tangent_and_normal_lines_l3495_349565

-- Define the curve
def x (t : ℝ) : ℝ := t - t^4
def y (t : ℝ) : ℝ := t^2 - t^3

-- Define the parameter value
def t₀ : ℝ := 1

-- State the theorem
theorem tangent_and_normal_lines :
  let x₀ := x t₀
  let y₀ := y t₀
  let dx := deriv x t₀
  let dy := deriv y t₀
  let m_tangent := dy / dx
  let m_normal := -1 / m_tangent
  (∀ t : ℝ, y t - y₀ = m_tangent * (x t - x₀)) ∧
  (∀ t : ℝ, y t - y₀ = m_normal * (x t - x₀)) := by
  sorry

end NUMINAMATH_CALUDE_tangent_and_normal_lines_l3495_349565


namespace NUMINAMATH_CALUDE_worksheets_graded_correct_l3495_349594

/-- Represents the number of worksheets graded before new ones were turned in -/
def worksheets_graded : ℕ := 7

/-- The initial number of worksheets to grade -/
def initial_worksheets : ℕ := 34

/-- The number of new worksheets turned in -/
def new_worksheets : ℕ := 36

/-- The final number of worksheets to grade -/
def final_worksheets : ℕ := 63

/-- Theorem stating that the number of worksheets graded before new ones were turned in is correct -/
theorem worksheets_graded_correct :
  initial_worksheets - worksheets_graded + new_worksheets = final_worksheets :=
by sorry

end NUMINAMATH_CALUDE_worksheets_graded_correct_l3495_349594


namespace NUMINAMATH_CALUDE_arrangements_eq_combinations_l3495_349538

/-- The number of ways to arrange nine 1s and four 0s in a row, where no two 0s are adjacent -/
def arrangements : ℕ := sorry

/-- The number of ways to choose 4 items from 10 items -/
def combinations : ℕ := Nat.choose 10 4

/-- Theorem stating that the number of arrangements is equal to the number of combinations -/
theorem arrangements_eq_combinations : arrangements = combinations := by sorry

end NUMINAMATH_CALUDE_arrangements_eq_combinations_l3495_349538


namespace NUMINAMATH_CALUDE_sqrt_three_minus_pi_squared_l3495_349570

theorem sqrt_three_minus_pi_squared : Real.sqrt ((3 - Real.pi) ^ 2) = Real.pi - 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_minus_pi_squared_l3495_349570


namespace NUMINAMATH_CALUDE_max_k_value_l3495_349535

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 15 = 0

-- Define the line
def line (k : ℝ) (x y : ℝ) : Prop := y = k*x - 2

-- Define the condition for intersection
def has_intersection (k : ℝ) : Prop :=
  ∃ (x y : ℝ), line k x y ∧ 
  (∃ (x' y' : ℝ), circle_C x' y' ∧ (x - x')^2 + (y - y')^2 ≤ 1)

-- Theorem statement
theorem max_k_value :
  (∀ k : ℝ, k ≤ 4/3 → has_intersection k) ∧
  (∀ k : ℝ, k > 4/3 → ¬has_intersection k) :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l3495_349535


namespace NUMINAMATH_CALUDE_tangent_line_properties_l3495_349529

def curve_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 1 = 0

def line_l (α : ℝ) (x y : ℝ) : Prop := 
  ∃ t : ℝ, x = 4 + t * Real.sin α ∧ y = t * Real.cos α

def is_tangent (α : ℝ) : Prop :=
  ∃ x y : ℝ, curve_C x y ∧ line_l α x y ∧
  ∀ x' y' : ℝ, curve_C x' y' ∧ line_l α x' y' → (x', y') = (x, y)

theorem tangent_line_properties :
  ∀ α : ℝ, 0 ≤ α ∧ α < Real.pi → is_tangent α →
    α = Real.pi / 6 ∧
    ∃ x y : ℝ, curve_C x y ∧ line_l α x y ∧ x = 7/2 ∧ y = -Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_properties_l3495_349529


namespace NUMINAMATH_CALUDE_compound_interest_rate_l3495_349575

/-- Given an initial amount P at compound interest that sums to 17640 after 2 years
    and 22050 after 3 years, the annual interest rate is 25%. -/
theorem compound_interest_rate (P : ℝ) : 
  P * (1 + 0.25)^2 = 17640 ∧ P * (1 + 0.25)^3 = 22050 → 0.25 = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_rate_l3495_349575


namespace NUMINAMATH_CALUDE_delta_quotient_equals_two_plus_delta_x_l3495_349508

/-- The function f(x) = x^2 + 1 -/
def f (x : ℝ) : ℝ := x^2 + 1

/-- Theorem: For the function f(x) = x^2 + 1, given points (1, 2) and (1 + Δx, 2 + Δy) on the graph,
    Δy / Δx = 2 + Δx for any non-zero Δx -/
theorem delta_quotient_equals_two_plus_delta_x (Δx : ℝ) (Δy : ℝ) (h : Δx ≠ 0) :
  f (1 + Δx) = 2 + Δy →
  Δy / Δx = 2 + Δx :=
by sorry

end NUMINAMATH_CALUDE_delta_quotient_equals_two_plus_delta_x_l3495_349508


namespace NUMINAMATH_CALUDE_lcm_problem_l3495_349533

theorem lcm_problem (a b : ℕ+) (h1 : Nat.gcd a b = 11) (h2 : a * b = 1991) :
  Nat.lcm a b = 181 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l3495_349533


namespace NUMINAMATH_CALUDE_sum_A_B_equals_24_l3495_349576

theorem sum_A_B_equals_24 (A B : ℚ) (h1 : (1 : ℚ) / 6 * (1 : ℚ) / 3 = 1 / (A * 3))
  (h2 : (1 : ℚ) / 6 * (1 : ℚ) / 3 = 1 / B) : A + B = 24 := by
  sorry

end NUMINAMATH_CALUDE_sum_A_B_equals_24_l3495_349576


namespace NUMINAMATH_CALUDE_total_squares_is_83_l3495_349551

/-- Represents the count of squares of a specific size in the figure -/
structure SquareCount where
  size : Nat
  count : Nat

/-- Represents the figure composed of squares and isosceles right triangles -/
structure Figure where
  squareCounts : List SquareCount

/-- Calculates the total number of squares in the figure -/
def totalSquares (f : Figure) : Nat :=
  f.squareCounts.foldl (fun acc sc => acc + sc.count) 0

/-- The specific figure described in the problem -/
def problemFigure : Figure :=
  { squareCounts := [
      { size := 1, count := 40 },
      { size := 2, count := 25 },
      { size := 3, count := 12 },
      { size := 4, count := 5 },
      { size := 5, count := 1 }
    ] }

theorem total_squares_is_83 : totalSquares problemFigure = 83 := by
  sorry

end NUMINAMATH_CALUDE_total_squares_is_83_l3495_349551


namespace NUMINAMATH_CALUDE_perpendicular_unit_vectors_l3495_349547

def a : ℝ × ℝ := (2, -2)

theorem perpendicular_unit_vectors :
  let v₁ : ℝ × ℝ := (Real.sqrt 2 / 2, Real.sqrt 2 / 2)
  let v₂ : ℝ × ℝ := (-Real.sqrt 2 / 2, -Real.sqrt 2 / 2)
  (v₁.1 * a.1 + v₁.2 * a.2 = 0 ∧ v₁.1^2 + v₁.2^2 = 1) ∧
  (v₂.1 * a.1 + v₂.2 * a.2 = 0 ∧ v₂.1^2 + v₂.2^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_unit_vectors_l3495_349547


namespace NUMINAMATH_CALUDE_quadratic_sum_l3495_349546

/-- Given a quadratic function f(x) = 4x^2 - 28x - 108, prove that when written in the form
    a(x+b)^2 + c, the sum of a, b, and c is -156.5 -/
theorem quadratic_sum (f : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f x = 4 * x^2 - 28 * x - 108) →
  (∀ x, f x = a * (x + b)^2 + c) →
  a + b + c = -156.5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l3495_349546


namespace NUMINAMATH_CALUDE_slope_of_line_l3495_349554

theorem slope_of_line (x y : ℝ) :
  4 * x + 7 * y = 28 → (y - 4) / x = -4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_l3495_349554


namespace NUMINAMATH_CALUDE_expand_expression_l3495_349510

theorem expand_expression (x y : ℝ) : 12 * (3 * x + 4 - 2 * y) = 36 * x + 48 - 24 * y := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3495_349510


namespace NUMINAMATH_CALUDE_cube_root_neg_eight_plus_sqrt_nine_equals_one_l3495_349588

theorem cube_root_neg_eight_plus_sqrt_nine_equals_one :
  ((-8 : ℝ) ^ (1/3 : ℝ)) + (9 : ℝ).sqrt = 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_neg_eight_plus_sqrt_nine_equals_one_l3495_349588


namespace NUMINAMATH_CALUDE_base5_to_octal_polynomial_evaluation_l3495_349530

-- Define the base-5 number 1234₅
def base5_number : ℕ := 1 * 5^3 + 2 * 5^2 + 3 * 5^1 + 4 * 5^0

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := 7*x^7 + 6*x^6 + 5*x^5 + 4*x^4 + 3*x^3 + 2*x^2 + x

-- Theorem 1: Converting base-5 to octal
theorem base5_to_octal : 
  (base5_number : ℕ).digits 8 = [3, 0, 2] := by sorry

-- Theorem 2: Evaluating the polynomial at x = 3
theorem polynomial_evaluation :
  f 3 = 21324 := by sorry

end NUMINAMATH_CALUDE_base5_to_octal_polynomial_evaluation_l3495_349530


namespace NUMINAMATH_CALUDE_min_coefficient_value_l3495_349504

theorem min_coefficient_value (c d : ℤ) (box : ℤ) : 
  (c * d = 42) →
  (c ≠ d) → (c ≠ box) → (d ≠ box) →
  (∀ x, (c * x + d) * (d * x + c) = 42 * x^2 + box * x + 42) →
  (∀ c' d' box' : ℤ, 
    (c' * d' = 42) → 
    (c' ≠ d') → (c' ≠ box') → (d' ≠ box') →
    (∀ x, (c' * x + d') * (d' * x + c') = 42 * x^2 + box' * x + 42) →
    box ≤ box') →
  box = 85 := by
sorry

end NUMINAMATH_CALUDE_min_coefficient_value_l3495_349504


namespace NUMINAMATH_CALUDE_cubic_function_properties_l3495_349543

-- Define the function f(x) = ax^3 + bx
def f (a b x : ℝ) : ℝ := a * x^3 + b * x

-- Define the derivative of f
def f_derivative (a b x : ℝ) : ℝ := 3 * a * x^2 + b

theorem cubic_function_properties (a b : ℝ) :
  f a b 2 = 2 ∧ f_derivative a b 2 = 9 →
  a * b = -3 ∧
  Set.Icc (-3/2 : ℝ) 3 ⊆ f a b ⁻¹' Set.Icc (-2 : ℝ) 18 ∧
  ∃ (x₁ x₂ : ℝ), x₁ ∈ Set.Icc (-3/2 : ℝ) 3 ∧ x₂ ∈ Set.Icc (-3/2 : ℝ) 3 ∧
    f a b x₁ = -2 ∧ f a b x₂ = 18 :=
by
  sorry

#check cubic_function_properties

end NUMINAMATH_CALUDE_cubic_function_properties_l3495_349543


namespace NUMINAMATH_CALUDE_pancake_diameter_l3495_349596

/-- The diameter of a circular object with radius 7 centimeters is 14 centimeters. -/
theorem pancake_diameter (r : ℝ) (h : r = 7) : 2 * r = 14 := by
  sorry

end NUMINAMATH_CALUDE_pancake_diameter_l3495_349596


namespace NUMINAMATH_CALUDE_pen_pencil_difference_l3495_349584

theorem pen_pencil_difference :
  ∀ (pens pencils : ℕ),
    pens * 6 = pencils * 5 →  -- ratio of pens to pencils is 5:6
    pencils = 30 →            -- there are 30 pencils
    pencils - pens = 5        -- prove that there are 5 more pencils than pens
:= by sorry

end NUMINAMATH_CALUDE_pen_pencil_difference_l3495_349584


namespace NUMINAMATH_CALUDE_complex_sum_parts_l3495_349577

theorem complex_sum_parts (z : ℂ) (h : z / (1 + 2*I) = 2 + I) : 
  (z + 5).re + (z + 5).im = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_parts_l3495_349577


namespace NUMINAMATH_CALUDE_greatest_number_with_odd_factors_has_odd_factors_196_is_less_than_200_196_greatest_number_less_than_200_with_odd_factors_l3495_349544

def has_odd_number_of_factors (n : ℕ) : Prop :=
  Odd (Finset.card (Finset.filter (·∣n) (Finset.range (n + 1))))

theorem greatest_number_with_odd_factors : 
  ∀ n : ℕ, n < 200 → has_odd_number_of_factors n → n ≤ 196 :=
by sorry

theorem has_odd_factors_196 : has_odd_number_of_factors 196 :=
by sorry

theorem is_less_than_200_196 : 196 < 200 :=
by sorry

theorem greatest_number_less_than_200_with_odd_factors :
  ∃ n : ℕ, n < 200 ∧ has_odd_number_of_factors n ∧
  ∀ m : ℕ, m < 200 → has_odd_number_of_factors m → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_number_with_odd_factors_has_odd_factors_196_is_less_than_200_196_greatest_number_less_than_200_with_odd_factors_l3495_349544


namespace NUMINAMATH_CALUDE_factors_of_M_l3495_349589

/-- The number of natural-number factors of M, where M = 2^4 * 3^3 * 5^2 * 7^1 -/
def num_factors (M : ℕ) : ℕ :=
  (5 : ℕ) * 4 * 3 * 2

/-- M is defined as 2^4 * 3^3 * 5^2 * 7^1 -/
def M : ℕ := 2^4 * 3^3 * 5^2 * 7^1

theorem factors_of_M : num_factors M = 120 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_M_l3495_349589


namespace NUMINAMATH_CALUDE_hyperbola_foci_l3495_349550

/-- The equation of a hyperbola in standard form -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 7 - y^2 / 9 = 1

/-- The coordinates of a focus of the hyperbola -/
def focus_coordinate : ℝ × ℝ := (4, 0)

/-- Theorem: The foci of the given hyperbola are located at (±4, 0) -/
theorem hyperbola_foci :
  let (a, b) := focus_coordinate
  (hyperbola_equation a b ∨ hyperbola_equation (-a) b) ∧
  ∀ (x y : ℝ), (x, y) ≠ (a, b) ∧ (x, y) ≠ (-a, b) →
    ¬(hyperbola_equation x y ∧ x^2 - y^2 = a^2) :=
by sorry


end NUMINAMATH_CALUDE_hyperbola_foci_l3495_349550


namespace NUMINAMATH_CALUDE_water_bottles_stolen_solve_water_bottle_theft_l3495_349552

theorem water_bottles_stolen (initial_bottles : ℕ) (lost_bottles : ℕ) (stickers_per_bottle : ℕ) (total_stickers : ℕ) : ℕ :=
  let remaining_after_loss := initial_bottles - lost_bottles
  let remaining_after_theft := total_stickers / stickers_per_bottle
  remaining_after_loss - remaining_after_theft

theorem solve_water_bottle_theft : water_bottles_stolen 10 2 3 21 = 1 := by
  sorry

end NUMINAMATH_CALUDE_water_bottles_stolen_solve_water_bottle_theft_l3495_349552


namespace NUMINAMATH_CALUDE_expression_evaluation_l3495_349560

theorem expression_evaluation (a b c : ℝ) 
  (h1 : c = b + 1) 
  (h2 : b = a + 5) 
  (h3 : a = 3) 
  (h4 : a + 2 ≠ 0) 
  (h5 : b - 3 ≠ 0) 
  (h6 : c + 7 ≠ 0) : 
  ((a + 3) / (a + 2)) * ((b + 1) / (b - 3)) * ((c + 9) / (c + 7)) = 243 / 100 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3495_349560


namespace NUMINAMATH_CALUDE_bens_initial_money_l3495_349531

theorem bens_initial_money (initial_amount : ℕ) : 
  (((initial_amount - 600) + 800) - 1200 = 1000) → 
  initial_amount = 2000 := by
  sorry

end NUMINAMATH_CALUDE_bens_initial_money_l3495_349531


namespace NUMINAMATH_CALUDE_chemistry_class_section_size_l3495_349545

theorem chemistry_class_section_size :
  let section1_size : ℕ := 65
  let section2_size : ℕ := 35
  let section4_size : ℕ := 42
  let section1_mean : ℚ := 50
  let section2_mean : ℚ := 60
  let section3_mean : ℚ := 55
  let section4_mean : ℚ := 45
  let overall_mean : ℚ := 5195 / 100

  ∃ (section3_size : ℕ),
    (section1_size * section1_mean + section2_size * section2_mean + 
     section3_size * section3_mean + section4_size * section4_mean) / 
    (section1_size + section2_size + section3_size + section4_size : ℚ) = overall_mean ∧
    section3_size = 45
  := by sorry

end NUMINAMATH_CALUDE_chemistry_class_section_size_l3495_349545


namespace NUMINAMATH_CALUDE_unique_integer_sqrt_l3495_349521

theorem unique_integer_sqrt (x y : ℕ) : x = 25530 ∧ y = 29464 ↔ 
  ∃ (z : ℕ), z > 0 ∧ z * z = x * x + y * y ∧
  ∀ (a b : ℕ), (a = 37615 ∧ b = 26855) ∨ 
               (a = 15123 ∧ b = 32477) ∨ 
               (a = 28326 ∧ b = 28614) ∨ 
               (a = 22536 ∧ b = 27462) →
               ¬∃ (w : ℕ), w > 0 ∧ w * w = a * a + b * b :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_sqrt_l3495_349521


namespace NUMINAMATH_CALUDE_sum_of_squares_inequality_range_of_a_l3495_349540

-- Part I
theorem sum_of_squares_inequality (a b c : ℝ) (h : a + b + c = 1) :
  (a + 1)^2 + (b + 1)^2 + (c + 1)^2 ≥ 16/3 := by sorry

-- Part II
theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, |x - a| + |2*x - 1| ≥ 2) :
  a ≤ -3/2 ∨ a ≥ 5/2 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_inequality_range_of_a_l3495_349540


namespace NUMINAMATH_CALUDE_set_operation_result_l3495_349586

def X : Set ℕ := {0, 1, 2, 4, 5, 7}
def Y : Set ℕ := {1, 3, 6, 8, 9}
def Z : Set ℕ := {3, 7, 8}

theorem set_operation_result : (X ∩ Y) ∪ Z = {1, 3, 7, 8} := by sorry

end NUMINAMATH_CALUDE_set_operation_result_l3495_349586


namespace NUMINAMATH_CALUDE_probability_diamond_then_face_correct_l3495_349526

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- Represents the suit of a card -/
inductive Suit
| hearts | diamonds | clubs | spades

/-- Represents the rank of a card -/
inductive Rank
| two | three | four | five | six | seven | eight | nine | ten
| jack | queen | king | ace

/-- Represents a playing card -/
structure Card :=
  (suit : Suit)
  (rank : Rank)

/-- Checks if a card is a diamond -/
def is_diamond (c : Card) : Prop :=
  c.suit = Suit.diamonds

/-- Checks if a card is a face card -/
def is_face_card (c : Card) : Prop :=
  c.rank = Rank.jack ∨ c.rank = Rank.queen ∨ c.rank = Rank.king

/-- The number of diamonds in a standard deck -/
def diamond_count : Nat := 13

/-- The number of face cards in a standard deck -/
def face_card_count : Nat := 12

/-- The probability of drawing a diamond as the first card and a face card as the second card -/
def probability_diamond_then_face (d : Deck) : ℚ :=
  47 / 884

theorem probability_diamond_then_face_correct (d : Deck) :
  probability_diamond_then_face d = 47 / 884 :=
sorry

end NUMINAMATH_CALUDE_probability_diamond_then_face_correct_l3495_349526


namespace NUMINAMATH_CALUDE_binomial_square_constant_l3495_349549

theorem binomial_square_constant (k : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, x^2 - 60*x + k = (a*x + b)^2) → k = 900 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_constant_l3495_349549


namespace NUMINAMATH_CALUDE_spoiled_cross_to_square_l3495_349553

/-- Represents a symmetrical Greek cross -/
structure GreekCross where
  arm_length : ℝ
  arm_width : ℝ
  symmetrical : arm_length > 0 ∧ arm_width > 0

/-- Represents a square -/
structure Square where
  side_length : ℝ
  is_positive : side_length > 0

/-- Represents a Greek cross with a square cut out -/
structure SpoiledGreekCross where
  cross : GreekCross
  cut_out : Square
  fits_end : cut_out.side_length = cross.arm_width

/-- Represents a piece obtained from cutting the spoiled Greek cross -/
structure Piece where
  area : ℝ
  is_positive : area > 0

/-- Theorem stating that a spoiled Greek cross can be cut into four pieces
    that can be reassembled into a square -/
theorem spoiled_cross_to_square (sc : SpoiledGreekCross) :
  ∃ (p1 p2 p3 p4 : Piece) (result : Square),
    p1.area + p2.area + p3.area + p4.area = result.side_length ^ 2 :=
sorry

end NUMINAMATH_CALUDE_spoiled_cross_to_square_l3495_349553


namespace NUMINAMATH_CALUDE_polynomial_sum_theorem_l3495_349564

theorem polynomial_sum_theorem (d : ℝ) (h : d ≠ 0) :
  ∃ (a b c : ℤ), (10 * d - 3 + 16 * d^2) + (4 * d + 7) = a * d + b + c * d^2 ∧ a + b + c = 34 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_theorem_l3495_349564


namespace NUMINAMATH_CALUDE_p_equiv_not_q_l3495_349525

theorem p_equiv_not_q (P Q : Prop) 
  (h1 : P ∨ Q) 
  (h2 : ¬(P ∧ Q)) : 
  P ↔ ¬Q := by
  sorry

end NUMINAMATH_CALUDE_p_equiv_not_q_l3495_349525


namespace NUMINAMATH_CALUDE_circle_line_intersection_l3495_349567

/-- A circle in the xy-plane -/
structure Circle where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  m : ℝ

/-- A line in the xy-plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The origin point (0,0) -/
def origin : Point := ⟨0, 0⟩

/-- Check if a point lies on a circle -/
def Circle.contains (c : Circle) (p : Point) : Prop :=
  p.x^2 + p.y^2 + c.a * p.x + c.b * p.y + c.m = 0

/-- Check if a point lies on a line -/
def Line.contains (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two points are distinct -/
def Point.distinct (p q : Point) : Prop :=
  p.x ≠ q.x ∨ p.y ≠ q.y

/-- The circle with a given diameter passes through the origin -/
def circle_through_origin (p q : Point) : Prop :=
  ∃ (c : Circle), c.contains p ∧ c.contains q ∧ c.contains origin

/-- The main theorem -/
theorem circle_line_intersection (c : Circle) (l : Line) (p q : Point) :
  c.a = 1 ∧ c.b = -6 ∧ c.c = 1 ∧ c.d = 1 ∧
  l.a = 1 ∧ l.b = 2 ∧ l.c = -3 ∧
  c.contains p ∧ c.contains q ∧
  l.contains p ∧ l.contains q ∧
  Point.distinct p q ∧
  circle_through_origin p q →
  c.m = 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l3495_349567


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l3495_349522

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 17 = 0 → n ≤ 986 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l3495_349522


namespace NUMINAMATH_CALUDE_max_area_OAP_l3495_349590

noncomputable section

/-- The maximum area of triangle OAP given the conditions --/
theorem max_area_OAP (a m : ℝ) (h1 : 0 < a) (h2 : a < 1/2) 
  (h3 : -a < m) (h4 : m ≤ (a^2 + 1)/2)
  (h5 : ∃! P : ℝ × ℝ, P.2 > 0 ∧ P.1^2 + a^2*P.2^2 = a^2 ∧ P.2^2 = 2*(P.1 + m)) :
  ∃ (S : ℝ), S = (1/54)*Real.sqrt 6 ∧ 
  (∀ A P : ℝ × ℝ, A.2 = 0 ∧ A.1^2 + a^2*A.2^2 = a^2 ∧ A.1 < 0 ∧
   P.2 > 0 ∧ P.1^2 + a^2*P.2^2 = a^2 ∧ P.2^2 = 2*(P.1 + m) →
   (1/2) * abs (A.1 * P.2) ≤ S) := by
  sorry

end

end NUMINAMATH_CALUDE_max_area_OAP_l3495_349590


namespace NUMINAMATH_CALUDE_g_inv_composite_equals_three_l3495_349502

-- Define the function g
def g : ℕ → ℕ
| 1 => 4
| 2 => 12
| 3 => 7
| 5 => 2
| 8 => 1
| 13 => 6
| _ => 0  -- Default value for undefined inputs

-- Axiom: g has an inverse
axiom g_has_inverse : Function.Bijective g

-- Define g_inv using g_has_inverse
noncomputable def g_inv : ℕ → ℕ := Function.invFun g

-- State the theorem
theorem g_inv_composite_equals_three :
  g_inv ((g_inv 6 + g_inv 12) / g_inv 2) = 3 := by sorry

end NUMINAMATH_CALUDE_g_inv_composite_equals_three_l3495_349502


namespace NUMINAMATH_CALUDE_two_books_cost_l3495_349593

/-- The cost of two books, where one is sold at a loss and the other at a gain --/
theorem two_books_cost (C₁ C₂ : ℝ) (h1 : C₁ = 274.1666666666667) 
  (h2 : C₁ * 0.85 = C₂ * 1.19) : 
  abs (C₁ + C₂ - 470) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_two_books_cost_l3495_349593


namespace NUMINAMATH_CALUDE_system_solution_l3495_349542

theorem system_solution :
  ∀ (x y : ℤ) (m : ℝ),
    x < 0 ∧ y > 0 ∧
    -2 * x + 3 * y = 2 * m ∧
    x - 5 * y = -11 →
    ((x = -6 ∧ y = 1 ∧ m = 7.5) ∨ (x = -1 ∧ y = 2 ∧ m = 4)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3495_349542


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3495_349528

theorem absolute_value_inequality (x : ℝ) : 
  (|x - 2| + |x - 3| < 9) ↔ (-2 < x ∧ x < 7) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3495_349528


namespace NUMINAMATH_CALUDE_club_sports_theorem_l3495_349527

/-- The number of people who do not play a sport in a club -/
def people_not_playing (total : ℕ) (tennis : ℕ) (baseball : ℕ) (both : ℕ) : ℕ :=
  total - (tennis + baseball - both)

/-- Theorem: In a club with 310 people, where 138 play tennis, 255 play baseball, 
    and 94 play both sports, 11 people do not play a sport. -/
theorem club_sports_theorem : people_not_playing 310 138 255 94 = 11 := by
  sorry

end NUMINAMATH_CALUDE_club_sports_theorem_l3495_349527


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3495_349556

-- Define the quadratic function
def f (a x : ℝ) : ℝ := a * x^2 + (1 - a) * x - 1

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ :=
  {x | f a x > 0}

-- Theorem statement
theorem quadratic_inequality_solution (a : ℝ) (h : a < 0) :
  solution_set a = 
    if -1 < a then {x | 1 < x ∧ x < -1/a}
    else if a = -1 then ∅
    else {x | -1/a < x ∧ x < 1} :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3495_349556


namespace NUMINAMATH_CALUDE_product_of_conjugates_l3495_349500

theorem product_of_conjugates (P Q R S : ℝ) : 
  P = Real.sqrt 2023 + Real.sqrt 2024 →
  Q = -Real.sqrt 2023 - Real.sqrt 2024 →
  R = Real.sqrt 2023 - Real.sqrt 2024 →
  S = Real.sqrt 2024 - Real.sqrt 2023 →
  P * Q * R * S = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_of_conjugates_l3495_349500


namespace NUMINAMATH_CALUDE_kiwi_profit_optimization_l3495_349555

/-- Kiwi prices and profit optimization problem -/
theorem kiwi_profit_optimization 
  (green_price red_price : ℕ) 
  (green_cost red_cost : ℕ) 
  (total_boxes : ℕ) 
  (max_expenditure : ℕ) :
  green_cost = 80 →
  red_cost = 100 →
  total_boxes = 21 →
  max_expenditure = 2000 →
  red_price = green_price + 25 →
  6 * green_price = 5 * red_price - 25 →
  green_price = 100 ∧ 
  red_price = 125 ∧
  (∃ (green_boxes red_boxes : ℕ),
    green_boxes + red_boxes = total_boxes ∧
    green_boxes * green_cost + red_boxes * red_cost ≤ max_expenditure ∧
    green_boxes = 5 ∧ 
    red_boxes = 16 ∧
    (green_boxes * (green_price - green_cost) + red_boxes * (red_price - red_cost)) = 500 ∧
    ∀ (g r : ℕ), 
      g + r = total_boxes → 
      g * green_cost + r * red_cost ≤ max_expenditure →
      g * (green_price - green_cost) + r * (red_price - red_cost) ≤ 500) :=
by sorry

end NUMINAMATH_CALUDE_kiwi_profit_optimization_l3495_349555


namespace NUMINAMATH_CALUDE_power_difference_square_sum_l3495_349581

theorem power_difference_square_sum (m n : ℕ+) : 
  2^(m : ℕ) - 2^(n : ℕ) = 1792 → m^2 + n^2 = 185 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_square_sum_l3495_349581


namespace NUMINAMATH_CALUDE_no_solutions_squared_l3495_349503

theorem no_solutions_squared (n : ℕ) (h : n > 2) :
  (∀ x y z : ℕ+, x^n + y^n ≠ z^n) →
  (∀ x y z : ℕ+, x^(2*n) + y^(2*n) ≠ z^2) :=
by sorry

end NUMINAMATH_CALUDE_no_solutions_squared_l3495_349503


namespace NUMINAMATH_CALUDE_p_and_q_false_iff_a_range_l3495_349598

/-- The logarithm function with base 10 -/
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

/-- The function f(x) = lg(ax^2 - x + a/16) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := lg (a * x^2 - x + a/16)

/-- Proposition p: The range of f(x) is ℝ -/
def p (a : ℝ) : Prop := ∀ y : ℝ, ∃ x : ℝ, f a x = y

/-- Proposition q: 3^x - 9^x < a holds for all real numbers x -/
def q (a : ℝ) : Prop := ∀ x : ℝ, 3^x - 9^x < a

/-- Theorem: "p and q" is false iff a > 2 or a ≤ 1/4 -/
theorem p_and_q_false_iff_a_range (a : ℝ) : ¬(p a ∧ q a) ↔ a > 2 ∨ a ≤ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_p_and_q_false_iff_a_range_l3495_349598


namespace NUMINAMATH_CALUDE_sin_cubed_identity_l3495_349569

theorem sin_cubed_identity (θ : Real) : 
  Real.sin θ ^ 3 = -1/4 * Real.sin (3 * θ) + 3/4 * Real.sin θ := by
  sorry

end NUMINAMATH_CALUDE_sin_cubed_identity_l3495_349569


namespace NUMINAMATH_CALUDE_truck_mileage_l3495_349517

/-- Given a truck that travels 240 miles on 5 gallons of gas, 
    prove that it can travel 336 miles on 7 gallons of gas. -/
theorem truck_mileage (miles_on_five : ℝ) (gallons_five : ℝ) (gallons_seven : ℝ) 
  (h1 : miles_on_five = 240)
  (h2 : gallons_five = 5)
  (h3 : gallons_seven = 7) :
  (miles_on_five / gallons_five) * gallons_seven = 336 := by
sorry

end NUMINAMATH_CALUDE_truck_mileage_l3495_349517


namespace NUMINAMATH_CALUDE_hyperbola_sum_l3495_349513

/-- Given a hyperbola with center (-2, 0), one focus at (-2 + √41, 0), and one vertex at (-7, 0),
    prove that h + k + a + b = 7, where (h, k) is the center, a is the distance from the center
    to a vertex, and b is the length of the conjugate axis. -/
theorem hyperbola_sum (h k a b c : ℝ) : 
  h = -2 ∧ 
  k = 0 ∧ 
  (h + Real.sqrt 41 - h)^2 = c^2 ∧
  (h - 5 - h)^2 = a^2 ∧
  c^2 = a^2 + b^2 →
  h + k + a + b = 7 := by sorry

end NUMINAMATH_CALUDE_hyperbola_sum_l3495_349513


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l3495_349534

theorem regular_polygon_sides (n : ℕ) (h : n > 0) :
  (360 : ℝ) / n = 15 → n = 24 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l3495_349534


namespace NUMINAMATH_CALUDE_equation_solution_l3495_349559

theorem equation_solution (x : ℝ) : 
  (2*x - 3) / (x + 4) = (3*x + 1) / (2*x - 5) ↔ 
  x = (29 + Real.sqrt 797) / 2 ∨ x = (29 - Real.sqrt 797) / 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3495_349559


namespace NUMINAMATH_CALUDE_intersection_implies_a_range_l3495_349506

def M : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 2}
def N (a : ℝ) : Set ℝ := {x : ℝ | 1 - 3*a < x ∧ x ≤ 2*a}

theorem intersection_implies_a_range (a : ℝ) : M ∩ N a = M → a ∈ Set.Ici 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_range_l3495_349506


namespace NUMINAMATH_CALUDE_triangle_properties_l3495_349557

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC exists
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π →
  -- Given condition
  2 * a = Real.sqrt 3 * c * Real.sin A - a * Real.cos C →
  -- Part 1: Prove C = 2π/3
  C = 2 * π / 3 ∧
  -- Part 2: Prove maximum area is √3/4 when c = √3
  (c = Real.sqrt 3 →
    ∀ (a' b' : ℝ), 
      0 < a' ∧ 0 < b' ∧
      2 * a' = Real.sqrt 3 * c * Real.sin A - a' * Real.cos C →
      1/2 * a' * b' * Real.sin C ≤ Real.sqrt 3 / 4) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3495_349557


namespace NUMINAMATH_CALUDE_line_through_point_l3495_349585

theorem line_through_point (b : ℚ) : 
  (b * (-3) - (b - 1) * 5 = b - 3) ↔ (b = 8 / 9) := by sorry

end NUMINAMATH_CALUDE_line_through_point_l3495_349585


namespace NUMINAMATH_CALUDE_amy_soup_count_l3495_349524

/-- The number of cans of chicken soup Amy bought -/
def chicken_soup : ℕ := 6

/-- The number of cans of tomato soup Amy bought -/
def tomato_soup : ℕ := 3

/-- The number of cans of vegetable soup Amy bought -/
def vegetable_soup : ℕ := 4

/-- The number of cans of clam chowder Amy bought -/
def clam_chowder : ℕ := 2

/-- The number of cans of French onion soup Amy bought -/
def french_onion_soup : ℕ := 1

/-- The number of cans of minestrone soup Amy bought -/
def minestrone_soup : ℕ := 5

/-- The total number of cans of soup Amy bought -/
def total_soups : ℕ := chicken_soup + tomato_soup + vegetable_soup + clam_chowder + french_onion_soup + minestrone_soup

theorem amy_soup_count : total_soups = 21 := by
  sorry

end NUMINAMATH_CALUDE_amy_soup_count_l3495_349524


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l3495_349579

theorem square_area_from_diagonal (d : ℝ) (h : d = 8 * Real.sqrt 2) : 
  (d / Real.sqrt 2) ^ 2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l3495_349579


namespace NUMINAMATH_CALUDE_pq_length_l3495_349516

-- Define the triangle PQR
def Triangle (P Q R : ℝ × ℝ) : Prop :=
  -- PQR is a right-angled triangle
  (Q.1 - P.1) * (R.2 - P.2) = (R.1 - P.1) * (Q.2 - P.2) ∧
  -- Angle PQR is 45°
  (Q.1 - P.1) * (R.1 - Q.1) + (Q.2 - P.2) * (R.2 - Q.2) = 
  Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2) * Real.sqrt ((R.1 - Q.1)^2 + (R.2 - Q.2)^2) / Real.sqrt 2 ∧
  -- PR = 10
  (R.1 - P.1)^2 + (R.2 - P.2)^2 = 100

-- Theorem statement
theorem pq_length (P Q R : ℝ × ℝ) (h : Triangle P Q R) :
  (Q.1 - P.1)^2 + (Q.2 - P.2)^2 = 50 := by
  sorry

end NUMINAMATH_CALUDE_pq_length_l3495_349516


namespace NUMINAMATH_CALUDE_integral_cos_plus_exp_l3495_349523

theorem integral_cos_plus_exp : 
  ∫ x in -Real.pi..0, (Real.cos x + Real.exp x) = 1 - 1 / Real.exp Real.pi := by
  sorry

end NUMINAMATH_CALUDE_integral_cos_plus_exp_l3495_349523


namespace NUMINAMATH_CALUDE_candy_mixture_weight_l3495_349561

/-- Proves that a candy mixture weighs 80 pounds given specific conditions -/
theorem candy_mixture_weight :
  ∀ (x : ℝ),
  x ≥ 0 →
  2 * x + 3 * 16 = 2.20 * (x + 16) →
  x + 16 = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_candy_mixture_weight_l3495_349561


namespace NUMINAMATH_CALUDE_power_mod_eleven_l3495_349532

theorem power_mod_eleven : 5^303 % 11 = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_eleven_l3495_349532


namespace NUMINAMATH_CALUDE_star_properties_l3495_349595

def star (x y : ℤ) : ℤ := (x + 2) * (y + 2) - 3

theorem star_properties :
  (∀ x y : ℤ, star x y = star y x) ∧
  (∃ x y z : ℤ, star x (y + z) ≠ star x y + star x z) ∧
  (∃ x : ℤ, star (x - 2) (x + 2) ≠ star x x - 3) ∧
  (∃ x : ℤ, star x 1 ≠ x) := by
  sorry

end NUMINAMATH_CALUDE_star_properties_l3495_349595


namespace NUMINAMATH_CALUDE_sector_central_angle_l3495_349519

theorem sector_central_angle (circumference area : ℝ) (h_circ : circumference = 6) (h_area : area = 2) :
  ∃ (r l : ℝ), r > 0 ∧ l > 0 ∧ 2 * r + l = circumference ∧ (1 / 2) * r * l = area ∧
  (l / r = 1 ∨ l / r = 4) :=
sorry

end NUMINAMATH_CALUDE_sector_central_angle_l3495_349519


namespace NUMINAMATH_CALUDE_students_present_l3495_349514

theorem students_present (total : ℕ) (absent_percent : ℚ) (present : ℕ) : 
  total = 50 → 
  absent_percent = 1/10 → 
  present = total - (total * (absent_percent : ℚ)).num / (absent_percent : ℚ).den → 
  present = 45 := by sorry

end NUMINAMATH_CALUDE_students_present_l3495_349514


namespace NUMINAMATH_CALUDE_count_special_numbers_eq_51_l3495_349566

/-- Sum of digits of a positive integer -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Counts the number of three-digit numbers x such that digit_sum(digit_sum(x)) = 4 -/
def count_special_numbers : ℕ := sorry

theorem count_special_numbers_eq_51 : count_special_numbers = 51 := by sorry

end NUMINAMATH_CALUDE_count_special_numbers_eq_51_l3495_349566


namespace NUMINAMATH_CALUDE_agency_a_cheaper_l3495_349541

/-- Represents a travel agency with a pricing function -/
structure TravelAgency where
  price : ℕ → ℝ

/-- The initial price per person -/
def initialPrice : ℝ := 200

/-- Travel Agency A with 25% discount for all -/
def agencyA : TravelAgency :=
  { price := λ x => initialPrice * 0.75 * x }

/-- Travel Agency B with one free and 20% discount for the rest -/
def agencyB : TravelAgency :=
  { price := λ x => initialPrice * 0.8 * (x - 1) }

/-- Theorem stating when Agency A is cheaper than Agency B -/
theorem agency_a_cheaper (x : ℕ) :
  x > 16 → agencyA.price x < agencyB.price x :=
sorry

end NUMINAMATH_CALUDE_agency_a_cheaper_l3495_349541


namespace NUMINAMATH_CALUDE_wallpaper_three_layers_l3495_349558

/-- Given wallpaper covering conditions, prove the area covered by three layers -/
theorem wallpaper_three_layers
  (total_area : ℝ)
  (wall_area : ℝ)
  (two_layer_area : ℝ)
  (h1 : total_area = 300)
  (h2 : wall_area = 180)
  (h3 : two_layer_area = 30)
  : ∃ (three_layer_area : ℝ),
    three_layer_area = total_area - (wall_area - two_layer_area + two_layer_area) ∧
    three_layer_area = 120 :=
by sorry

end NUMINAMATH_CALUDE_wallpaper_three_layers_l3495_349558


namespace NUMINAMATH_CALUDE_upload_time_calculation_l3495_349572

def file_size : ℝ := 160
def upload_speed : ℝ := 8

theorem upload_time_calculation : 
  file_size / upload_speed = 20 := by sorry

end NUMINAMATH_CALUDE_upload_time_calculation_l3495_349572


namespace NUMINAMATH_CALUDE_sum_of_cubes_negative_l3495_349511

theorem sum_of_cubes_negative : 
  (Real.sqrt 2021 - Real.sqrt 2020)^3 + 
  (Real.sqrt 2020 - Real.sqrt 2019)^3 + 
  (Real.sqrt 2019 - Real.sqrt 2018)^3 < 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_negative_l3495_349511


namespace NUMINAMATH_CALUDE_no_valid_arrangement_l3495_349539

theorem no_valid_arrangement :
  ¬ ∃ (x y : ℕ), 
    90 = x * y ∧ 
    5 ≤ x ∧ x ≤ 20 ∧ 
    Even y :=
by sorry

end NUMINAMATH_CALUDE_no_valid_arrangement_l3495_349539


namespace NUMINAMATH_CALUDE_tangent_and_normal_equations_l3495_349571

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 5

/-- The x-coordinate of the point of interest -/
def x₀ : ℝ := 2

/-- The y-coordinate of the point of interest -/
def y₀ : ℝ := f x₀

/-- The slope of the tangent line at x₀ -/
def m : ℝ := 2*x₀ - 2

theorem tangent_and_normal_equations :
  (∀ x y, 2*x - y + 1 = 0 ↔ y = m*(x - x₀) + y₀) ∧
  (∀ x y, x + 2*y - 12 = 0 ↔ y = -1/(2*m)*(x - x₀) + y₀) := by
  sorry


end NUMINAMATH_CALUDE_tangent_and_normal_equations_l3495_349571


namespace NUMINAMATH_CALUDE_curve_slope_range_l3495_349505

/-- The curve y = ln x + ax² - 2x has no tangent lines with negative slope -/
def no_negative_slope (a : ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → (1 / x + 2 * a * x - 2) ≥ 0

/-- The range of a for which the curve has no negative slope tangents -/
theorem curve_slope_range (a : ℝ) : no_negative_slope a → a ≥ (1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_curve_slope_range_l3495_349505


namespace NUMINAMATH_CALUDE_vertex_of_quadratic_l3495_349597

/-- The quadratic function f(x) = 2 - (x+1)^2 -/
def f (x : ℝ) : ℝ := 2 - (x + 1)^2

/-- The vertex of a quadratic function -/
structure Vertex where
  x : ℝ
  y : ℝ

/-- Theorem: The vertex of f(x) = 2 - (x+1)^2 is at (-1, 2) -/
theorem vertex_of_quadratic : 
  ∃ (v : Vertex), v.x = -1 ∧ v.y = 2 ∧ 
  ∀ (x : ℝ), f x ≤ f v.x := by
  sorry

end NUMINAMATH_CALUDE_vertex_of_quadratic_l3495_349597


namespace NUMINAMATH_CALUDE_martin_trip_distance_l3495_349537

/-- Calculates the total distance traveled during a two-part journey -/
def total_distance (total_time hours_per_half : ℝ) (speed1 speed2 : ℝ) : ℝ :=
  speed1 * hours_per_half + speed2 * hours_per_half

/-- Proves that the total distance traveled in the given conditions is 620 km -/
theorem martin_trip_distance :
  let total_time : ℝ := 8
  let speed1 : ℝ := 70
  let speed2 : ℝ := 85
  let hours_per_half : ℝ := total_time / 2
  total_distance total_time hours_per_half speed1 speed2 = 620 := by
  sorry

#eval total_distance 8 4 70 85

end NUMINAMATH_CALUDE_martin_trip_distance_l3495_349537


namespace NUMINAMATH_CALUDE_beyonce_album_songs_l3495_349573

/-- The number of songs in Beyonce's first two albums -/
def songs_in_first_two_albums (total_songs num_singles num_albums songs_in_third_album : ℕ) : ℕ :=
  total_songs - num_singles - songs_in_third_album

theorem beyonce_album_songs :
  songs_in_first_two_albums 55 5 3 20 = 30 := by
  sorry

end NUMINAMATH_CALUDE_beyonce_album_songs_l3495_349573


namespace NUMINAMATH_CALUDE_unique_n_exists_l3495_349562

theorem unique_n_exists : ∃! n : ℤ,
  50 < n ∧ n < 150 ∧
  n % 7 = 0 ∧
  n % 9 = 3 ∧
  n % 6 = 3 ∧
  n = 63 := by
sorry

end NUMINAMATH_CALUDE_unique_n_exists_l3495_349562


namespace NUMINAMATH_CALUDE_pyramid_height_equals_cube_volume_l3495_349582

theorem pyramid_height_equals_cube_volume (cube_edge : ℝ) (pyramid_base : ℝ) (pyramid_height : ℝ) :
  cube_edge = 5 →
  pyramid_base = 6 →
  (1 / 3) * pyramid_base^2 * pyramid_height = cube_edge^3 →
  pyramid_height = 125 / 12 :=
by sorry

end NUMINAMATH_CALUDE_pyramid_height_equals_cube_volume_l3495_349582


namespace NUMINAMATH_CALUDE_probability_of_white_ball_is_five_eighths_l3495_349515

/-- Represents the color of a ball -/
inductive Color
| White
| NonWhite

/-- Represents a bag of balls -/
def Bag := List Color

/-- The number of balls initially in the bag -/
def initialBallCount : Nat := 3

/-- Generates all possible initial configurations of the bag -/
def allPossibleInitialBags : List Bag :=
  sorry

/-- Adds a white ball to a bag -/
def addWhiteBall (bag : Bag) : Bag :=
  sorry

/-- Calculates the probability of drawing a white ball from a bag -/
def probabilityOfWhite (bag : Bag) : Rat :=
  sorry

/-- Calculates the average probability across all possible scenarios -/
def averageProbability (bags : List Bag) : Rat :=
  sorry

/-- The main theorem to prove -/
theorem probability_of_white_ball_is_five_eighths :
  averageProbability (allPossibleInitialBags.map addWhiteBall) = 5/8 :=
  sorry

end NUMINAMATH_CALUDE_probability_of_white_ball_is_five_eighths_l3495_349515


namespace NUMINAMATH_CALUDE_fruit_filled_mooncake_probability_l3495_349512

def num_fruits : ℕ := 5
def num_meats : ℕ := 4

def combinations (n : ℕ) : ℕ := n * (n - 1) / 2

theorem fruit_filled_mooncake_probability :
  let total_combinations := combinations num_fruits + combinations num_meats
  let fruit_combinations := combinations num_fruits
  (fruit_combinations : ℚ) / total_combinations = 5 / 8 := by sorry

end NUMINAMATH_CALUDE_fruit_filled_mooncake_probability_l3495_349512


namespace NUMINAMATH_CALUDE_cats_not_liking_either_l3495_349599

theorem cats_not_liking_either (total : ℕ) (cheese : ℕ) (tuna : ℕ) (both : ℕ) 
  (h_total : total = 100)
  (h_cheese : cheese = 25)
  (h_tuna : tuna = 70)
  (h_both : both = 15) :
  total - (cheese + tuna - both) = 20 := by
  sorry

end NUMINAMATH_CALUDE_cats_not_liking_either_l3495_349599


namespace NUMINAMATH_CALUDE_attitude_gender_relationship_expected_value_X_l3495_349587

-- Define the survey data
def total_sample : ℕ := 200
def male_agree : ℕ := 70
def male_disagree : ℕ := 30
def female_agree : ℕ := 50
def female_disagree : ℕ := 50

-- Define the chi-square function
def chi_square (n a b c d : ℕ) : ℚ :=
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the critical value for 99% confidence
def critical_value : ℚ := 6635 / 1000

-- Define the probability of agreeing
def p_agree : ℚ := (male_agree + female_agree) / total_sample

-- Theorem 1: Relationship between attitudes and gender
theorem attitude_gender_relationship :
  chi_square total_sample male_agree female_agree male_disagree female_disagree > critical_value :=
sorry

-- Theorem 2: Expected value of X
theorem expected_value_X :
  (3 : ℚ) * p_agree = 9 / 5 :=
sorry

end NUMINAMATH_CALUDE_attitude_gender_relationship_expected_value_X_l3495_349587


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l3495_349536

theorem right_triangle_third_side 
  (a b : ℝ) 
  (h : Real.sqrt (a^2 - 6*a + 9) + |b - 4| = 0) : 
  ∃ c : ℝ, (c = 5 ∨ c = Real.sqrt 7) ∧ 
    ((a^2 + b^2 = c^2) ∨ (a^2 + c^2 = b^2) ∨ (b^2 + c^2 = a^2)) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l3495_349536


namespace NUMINAMATH_CALUDE_total_recovery_time_l3495_349501

/-- Calculates the total recovery time for James after a hand burn, considering initial healing,
    post-surgery recovery, physical therapy sessions, and medication effects. -/
theorem total_recovery_time (initial_healing : ℝ) (A : ℝ) : 
  initial_healing = 4 →
  let post_surgery := initial_healing * 1.5
  let total_before_reduction := post_surgery
  let therapy_reduction := total_before_reduction * (0.1 * A)
  let medication_reduction := total_before_reduction * 0.2
  total_before_reduction - therapy_reduction - medication_reduction = 4.8 - 0.6 * A := by
  sorry

end NUMINAMATH_CALUDE_total_recovery_time_l3495_349501


namespace NUMINAMATH_CALUDE_fifth_largest_divisor_l3495_349563

def n : ℕ := 1020000000

def is_fifth_largest_divisor (d : ℕ) : Prop :=
  d ∣ n ∧ (∃ (a b c e : ℕ), a > b ∧ b > c ∧ c > d ∧ d > e ∧ 
    a ∣ n ∧ b ∣ n ∧ c ∣ n ∧ e ∣ n ∧
    ∀ (x : ℕ), x ∣ n → x ≤ e ∨ x = d ∨ x = c ∨ x = b ∨ x = a ∨ x = n)

theorem fifth_largest_divisor : is_fifth_largest_divisor 63750000 := by
  sorry

end NUMINAMATH_CALUDE_fifth_largest_divisor_l3495_349563


namespace NUMINAMATH_CALUDE_g_composition_15_l3495_349518

def g (x : ℤ) : ℤ :=
  if x % 3 = 0 then x / 3 else 5 * x + 2

theorem g_composition_15 : g (g (g (g 15))) = 3 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_15_l3495_349518


namespace NUMINAMATH_CALUDE_intersection_and_union_when_m_is_3_intersection_empty_iff_l3495_349592

-- Define sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ 2 * m + 1}

-- Part 1
theorem intersection_and_union_when_m_is_3 :
  (A ∩ B 3 = {x | 2 ≤ x ∧ x ≤ 5}) ∧
  ((Set.univ \ A) ∪ B 3 = {x | x < -2 ∨ x ≥ 2}) := by sorry

-- Part 2
theorem intersection_empty_iff :
  ∀ m : ℝ, A ∩ B m = ∅ ↔ m < -3/2 ∨ m > 6 := by sorry

end NUMINAMATH_CALUDE_intersection_and_union_when_m_is_3_intersection_empty_iff_l3495_349592


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3495_349568

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The 7th term of the arithmetic sequence is 1 -/
theorem arithmetic_sequence_problem (a : ℕ → ℚ) 
    (h_arith : ArithmeticSequence a) 
    (h_a4 : a 4 = 4)
    (h_sum : a 3 + a 8 = 5) : 
  a 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3495_349568
