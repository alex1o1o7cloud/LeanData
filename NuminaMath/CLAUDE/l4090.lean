import Mathlib

namespace NUMINAMATH_CALUDE_isosceles_with_120_exterior_is_equilateral_equal_exterior_angles_is_equilateral_l4090_409004

-- Define an isosceles triangle
structure IsoscelesTriangle :=
  (a b c : ℝ)
  (ab_eq_ac : a = c)

-- Define an equilateral triangle
structure EquilateralTriangle :=
  (a b c : ℝ)
  (all_sides_equal : a = b ∧ b = c)

-- Define exterior angle
def exterior_angle (interior_angle : ℝ) : ℝ := 180 - interior_angle

theorem isosceles_with_120_exterior_is_equilateral 
  (t : IsoscelesTriangle) 
  (h : ∃ (angle : ℝ), exterior_angle angle = 120) : 
  EquilateralTriangle :=
sorry

theorem equal_exterior_angles_is_equilateral 
  (t : IsoscelesTriangle) 
  (h : ∃ (angle : ℝ), 
    exterior_angle angle = exterior_angle (exterior_angle angle) ∧ 
    exterior_angle angle = exterior_angle (exterior_angle (exterior_angle angle))) : 
  EquilateralTriangle :=
sorry

end NUMINAMATH_CALUDE_isosceles_with_120_exterior_is_equilateral_equal_exterior_angles_is_equilateral_l4090_409004


namespace NUMINAMATH_CALUDE_hexagonal_lattice_triangles_l4090_409061

/-- Represents a point in the hexagonal lattice -/
structure LatticePoint where
  x : ℝ
  y : ℝ

/-- The hexagonal lattice with two concentric hexagons -/
structure HexagonalLattice where
  center : LatticePoint
  inner_hexagon : List LatticePoint
  outer_hexagon : List LatticePoint

/-- Checks if three points form an equilateral triangle -/
def is_equilateral_triangle (p1 p2 p3 : LatticePoint) : Prop :=
  sorry

/-- Counts the number of equilateral triangles in the lattice -/
def count_equilateral_triangles (lattice : HexagonalLattice) : ℕ :=
  sorry

/-- Main theorem: The number of equilateral triangles in the described hexagonal lattice is 18 -/
theorem hexagonal_lattice_triangles 
  (lattice : HexagonalLattice)
  (h1 : lattice.inner_hexagon.length = 6)
  (h2 : lattice.outer_hexagon.length = 6)
  (h3 : ∀ p ∈ lattice.inner_hexagon, 
    ∃ q ∈ lattice.inner_hexagon, 
    (p.x - q.x)^2 + (p.y - q.y)^2 = 1)
  (h4 : ∀ p ∈ lattice.outer_hexagon, 
    ∃ q ∈ lattice.inner_hexagon, 
    (p.x - q.x)^2 + (p.y - q.y)^2 = 4) :
  count_equilateral_triangles lattice = 18 :=
sorry

end NUMINAMATH_CALUDE_hexagonal_lattice_triangles_l4090_409061


namespace NUMINAMATH_CALUDE_find_first_number_l4090_409058

/-- A sequence where the sum of two numbers is always 1 less than their actual arithmetic sum -/
def SpecialSequence (a b c : ℕ) : Prop := a + b = c + 1

/-- The theorem to prove -/
theorem find_first_number (x : ℕ) :
  SpecialSequence x 9 16 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_find_first_number_l4090_409058


namespace NUMINAMATH_CALUDE_chris_hockey_stick_cost_l4090_409011

/-- The cost of a hockey stick, given the total spent, helmet cost, and number of sticks. -/
def hockey_stick_cost (total_spent helmet_cost num_sticks : ℚ) : ℚ :=
  (total_spent - helmet_cost) / num_sticks

/-- Theorem stating the cost of one hockey stick given Chris's purchase. -/
theorem chris_hockey_stick_cost :
  let total_spent : ℚ := 68
  let helmet_cost : ℚ := 25
  let num_sticks : ℚ := 2
  hockey_stick_cost total_spent helmet_cost num_sticks = 21.5 := by
  sorry

end NUMINAMATH_CALUDE_chris_hockey_stick_cost_l4090_409011


namespace NUMINAMATH_CALUDE_intersection_complement_l4090_409013

def U : Set ℕ := {1, 2, 3, 4}

theorem intersection_complement (A B : Set ℕ) 
  (h1 : (A ∪ B)ᶜ = {4})
  (h2 : B = {1, 2}) : 
  A ∩ Bᶜ = {3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_l4090_409013


namespace NUMINAMATH_CALUDE_geometric_progression_common_ratio_l4090_409098

theorem geometric_progression_common_ratio
  (q : ℝ)
  (h1 : |q| < 1)
  (h2 : ∀ (a : ℝ), a ≠ 0 → a = 4 * (a / (1 - q) - a)) :
  q = 1/5 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_common_ratio_l4090_409098


namespace NUMINAMATH_CALUDE_certain_number_proof_l4090_409022

theorem certain_number_proof : ∃ x : ℕ, 9873 + x = 13200 ∧ x = 3327 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l4090_409022


namespace NUMINAMATH_CALUDE_f_two_equals_two_thirds_l4090_409014

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x ↦ x / (x + 1)

-- State the theorem
theorem f_two_equals_two_thirds :
  (∀ x : ℝ, x ≠ 0 → f (1 / x) = 1 / (x + 1)) →
  f 2 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_f_two_equals_two_thirds_l4090_409014


namespace NUMINAMATH_CALUDE_pencil_price_in_units_l4090_409047

-- Define the base price in won
def base_price : ℝ := 5000

-- Define the additional cost in won
def additional_cost : ℝ := 200

-- Define the conversion factor from won to 10,000 won units
def conversion_factor : ℝ := 10000

-- Theorem statement
theorem pencil_price_in_units (price : ℝ) : 
  price = base_price + additional_cost → 
  price / conversion_factor = 0.52 := by
sorry

end NUMINAMATH_CALUDE_pencil_price_in_units_l4090_409047


namespace NUMINAMATH_CALUDE_correct_time_is_two_five_and_five_elevenths_l4090_409040

/-- Represents a time between 2 and 3 o'clock --/
structure Time where
  hour : ℕ
  minute : ℚ
  h_hour : hour = 2
  h_minute : 0 ≤ minute ∧ minute < 60

/-- Converts a Time to minutes past 2:00 --/
def timeToMinutes (t : Time) : ℚ :=
  60 * (t.hour - 2) + t.minute

/-- Represents the misread time by swapping hour and minute hands --/
def misreadTime (t : Time) : ℚ :=
  60 * (t.minute / 5) + 5 * t.hour

theorem correct_time_is_two_five_and_five_elevenths (t : Time) :
  misreadTime t = timeToMinutes t - 55 →
  t.hour = 2 ∧ t.minute = 5 + 5 / 11 := by
  sorry

end NUMINAMATH_CALUDE_correct_time_is_two_five_and_five_elevenths_l4090_409040


namespace NUMINAMATH_CALUDE_periodic_decimal_is_rational_l4090_409074

/-- A real number with a periodic decimal expansion can be expressed as a rational number. -/
theorem periodic_decimal_is_rational (x : ℝ) (d : ℕ) (k : ℕ) (a b : ℕ) 
  (h1 : x = (a : ℝ) / 10^k + (b : ℝ) / (10^k * (10^d - 1)))
  (h2 : b < 10^d) :
  ∃ (p q : ℤ), x = (p : ℝ) / (q : ℝ) ∧ q ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_periodic_decimal_is_rational_l4090_409074


namespace NUMINAMATH_CALUDE_yard_length_l4090_409006

theorem yard_length (num_trees : ℕ) (distance : ℝ) :
  num_trees = 11 →
  distance = 18 →
  (num_trees - 1) * distance = 180 :=
by sorry

end NUMINAMATH_CALUDE_yard_length_l4090_409006


namespace NUMINAMATH_CALUDE_estimate_sqrt_expression_l4090_409055

theorem estimate_sqrt_expression :
  6 < Real.sqrt 5 * (2 * Real.sqrt 5 - Real.sqrt 2) ∧
  Real.sqrt 5 * (2 * Real.sqrt 5 - Real.sqrt 2) < 7 :=
by sorry

end NUMINAMATH_CALUDE_estimate_sqrt_expression_l4090_409055


namespace NUMINAMATH_CALUDE_complex_distance_and_midpoint_l4090_409096

/-- Given two complex numbers, prove the distance between them and their midpoint -/
theorem complex_distance_and_midpoint (z1 z2 : ℂ) 
  (hz1 : z1 = 3 + 4*I) (hz2 : z2 = -2 - 3*I) : 
  Complex.abs (z1 - z2) = Real.sqrt 74 ∧ 
  (z1 + z2) / 2 = (1/2 : ℂ) + (1/2 : ℂ)*I := by
  sorry

end NUMINAMATH_CALUDE_complex_distance_and_midpoint_l4090_409096


namespace NUMINAMATH_CALUDE_square_perimeter_with_circles_l4090_409038

/-- Represents a square with inscribed circles -/
structure SquareWithCircles where
  /-- Side length of the square -/
  side : ℝ
  /-- Radius of each inscribed circle -/
  circle_radius : ℝ
  /-- The circles touch two sides of the square and two respective corners -/
  circles_touch_sides : circle_radius > 0

/-- The perimeter of a square with inscribed circles of radius 4 is 64√2 -/
theorem square_perimeter_with_circles (s : SquareWithCircles) 
  (h : s.circle_radius = 4) : s.side * 4 = 64 * Real.sqrt 2 := by
  sorry

#check square_perimeter_with_circles

end NUMINAMATH_CALUDE_square_perimeter_with_circles_l4090_409038


namespace NUMINAMATH_CALUDE_first_year_selection_probability_l4090_409060

/-- Represents the number of students in each year --/
structure StudentCounts where
  firstYear : ℕ
  secondYear : ℕ
  thirdYear : ℕ

/-- Represents the sampling information --/
structure SamplingInfo where
  thirdYearSample : ℕ

/-- Calculates the probability of a student being selected in stratified sampling --/
def stratifiedSamplingProbability (counts : StudentCounts) (info : SamplingInfo) : ℚ :=
  info.thirdYearSample / counts.thirdYear

/-- Theorem stating the probability of a first-year student being selected --/
theorem first_year_selection_probability 
  (counts : StudentCounts) 
  (info : SamplingInfo) 
  (h1 : counts.firstYear = 800)
  (h2 : counts.thirdYear = 500)
  (h3 : info.thirdYearSample = 25) :
  stratifiedSamplingProbability counts info = 1 / 20 := by
  sorry


end NUMINAMATH_CALUDE_first_year_selection_probability_l4090_409060


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l4090_409009

theorem completing_square_equivalence (x : ℝ) :
  (x^2 + 8*x + 9 = 0) ↔ ((x + 4)^2 = 7) :=
by sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l4090_409009


namespace NUMINAMATH_CALUDE_gambler_max_return_l4090_409008

/-- Represents the maximum amount a gambler can receive back after losing chips at a casino. -/
def max_amount_received (initial_value : ℕ) (chip_20_value : ℕ) (chip_100_value : ℕ) 
  (total_chips_lost : ℕ) : ℕ :=
  let chip_20_lost := (total_chips_lost + 2) / 2
  let chip_100_lost := total_chips_lost - chip_20_lost
  let value_lost := chip_20_lost * chip_20_value + chip_100_lost * chip_100_value
  initial_value - value_lost

/-- Theorem stating the maximum amount a gambler can receive back under specific conditions. -/
theorem gambler_max_return :
  max_amount_received 3000 20 100 16 = 2120 := by
  sorry

end NUMINAMATH_CALUDE_gambler_max_return_l4090_409008


namespace NUMINAMATH_CALUDE_easter_egg_hunt_l4090_409079

theorem easter_egg_hunt (bonnie george cheryl kevin : ℕ) : 
  bonnie = 13 →
  george = 9 →
  cheryl = 56 →
  cheryl = bonnie + george + kevin + 29 →
  kevin = 5 := by
sorry

end NUMINAMATH_CALUDE_easter_egg_hunt_l4090_409079


namespace NUMINAMATH_CALUDE_binomial_coefficient_10_3_l4090_409002

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 := by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_10_3_l4090_409002


namespace NUMINAMATH_CALUDE_smallest_n_sqrt_difference_l4090_409081

theorem smallest_n_sqrt_difference (n : ℕ) : 
  (n ≥ 2501) ↔ (Real.sqrt n - Real.sqrt (n - 1) < 0.01) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_sqrt_difference_l4090_409081


namespace NUMINAMATH_CALUDE_polynomial_remainder_l4090_409084

/-- Given a polynomial q(x) = Ax^6 + Bx^4 + Cx^2 + 10, if the remainder when
    q(x) is divided by x - 2 is 20, then the remainder when q(x) is divided
    by x + 2 is also 20. -/
theorem polynomial_remainder (A B C : ℝ) : 
  let q : ℝ → ℝ := λ x ↦ A * x^6 + B * x^4 + C * x^2 + 10
  (q 2 = 20) → (q (-2) = 20) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l4090_409084


namespace NUMINAMATH_CALUDE_valid_numbering_exists_l4090_409065

/-- Represents a numbering system for 7 contacts and 7 holes -/
def Numbering := Fin 7 → Fin 7

/-- Checks if a numbering system satisfies the alignment condition for all rotations -/
def isValidNumbering (n : Numbering) : Prop :=
  ∀ k : Fin 7, ∃ i : Fin 7, n i = (i + k : Fin 7)

/-- The main theorem stating that a valid numbering system exists -/
theorem valid_numbering_exists : ∃ n : Numbering, isValidNumbering n := by
  sorry


end NUMINAMATH_CALUDE_valid_numbering_exists_l4090_409065


namespace NUMINAMATH_CALUDE_sector_forms_cylinder_l4090_409012

-- Define the sector
def sector_angle : ℝ := 300
def sector_radius : ℝ := 12

-- Define the cylinder
def cylinder_base_radius : ℝ := 10
def cylinder_height : ℝ := 12

-- Theorem statement
theorem sector_forms_cylinder :
  2 * Real.pi * cylinder_base_radius = (sector_angle / 360) * 2 * Real.pi * sector_radius ∧
  cylinder_height = sector_radius :=
by sorry

end NUMINAMATH_CALUDE_sector_forms_cylinder_l4090_409012


namespace NUMINAMATH_CALUDE_sandwich_cost_calculation_l4090_409067

-- Define the costs and quantities
def selling_price : ℚ := 3
def bread_cost : ℚ := 0.15
def ham_cost : ℚ := 0.25
def cheese_cost : ℚ := 0.35
def mayo_cost : ℚ := 0.10
def lettuce_cost : ℚ := 0.05
def tomato_cost : ℚ := 0.08
def packaging_cost : ℚ := 0.02

def bread_qty : ℕ := 2
def ham_qty : ℕ := 2
def cheese_qty : ℕ := 2
def mayo_qty : ℕ := 1
def lettuce_qty : ℕ := 1
def tomato_qty : ℕ := 2

def discount_rate : ℚ := 0.1
def tax_rate : ℚ := 0.05

-- Define the theorem
theorem sandwich_cost_calculation :
  let ingredient_cost := bread_cost * bread_qty + ham_cost * ham_qty + cheese_cost * cheese_qty +
                         mayo_cost * mayo_qty + lettuce_cost * lettuce_qty + tomato_cost * tomato_qty
  let discount := (ham_cost * ham_qty + cheese_cost * cheese_qty) * discount_rate
  let adjusted_cost := ingredient_cost - discount + packaging_cost
  let tax := selling_price * tax_rate
  let total_cost := adjusted_cost + tax
  total_cost = 1.86 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_cost_calculation_l4090_409067


namespace NUMINAMATH_CALUDE_absolute_value_properties_l4090_409032

theorem absolute_value_properties :
  (∀ a : ℚ, a = 5 → |a| / a = 1) ∧
  (∀ a : ℚ, a = -2 → a / |a| = -1) ∧
  (∀ a b : ℚ, a * b > 0 → a / |a| + |b| / b = 2 ∨ a / |a| + |b| / b = -2) ∧
  (∀ a b c : ℚ, a * b * c < 0 → 
    a / |a| + b / |b| + c / |c| + (a * b * c) / |a * b * c| = 0 ∨
    a / |a| + b / |b| + c / |c| + (a * b * c) / |a * b * c| = -4) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_properties_l4090_409032


namespace NUMINAMATH_CALUDE_distance_is_approx_7_38_l4090_409054

/-- Represents a circle intersected by three equally spaced parallel lines -/
structure CircleWithParallelLines where
  radius : ℝ
  chord_length_1 : ℝ
  chord_length_2 : ℝ
  chord_length_3 : ℝ
  parallel_line_distance : ℝ
  chord_length_1_eq : chord_length_1 = 40
  chord_length_2_eq : chord_length_2 = 36
  chord_length_3_eq : chord_length_3 = 40
  equally_spaced : True  -- Assumption that lines are equally spaced

/-- The distance between adjacent parallel lines in the given configuration -/
def distance_between_lines (c : CircleWithParallelLines) : ℝ :=
  c.parallel_line_distance

/-- Theorem stating that the distance between adjacent parallel lines is approximately 7.38 -/
theorem distance_is_approx_7_38 (c : CircleWithParallelLines) :
  ∃ ε > 0, |distance_between_lines c - 7.38| < ε :=
sorry

#check distance_is_approx_7_38

end NUMINAMATH_CALUDE_distance_is_approx_7_38_l4090_409054


namespace NUMINAMATH_CALUDE_jason_initial_cards_l4090_409050

/-- The number of Pokemon cards Jason had initially -/
def initial_cards : ℕ := sorry

/-- The number of Pokemon cards Alyssa bought from Jason -/
def cards_bought : ℕ := 224

/-- The number of Pokemon cards Jason has now -/
def remaining_cards : ℕ := 452

/-- Theorem stating that Jason's initial number of Pokemon cards was 676 -/
theorem jason_initial_cards : initial_cards = 676 := by
  sorry

end NUMINAMATH_CALUDE_jason_initial_cards_l4090_409050


namespace NUMINAMATH_CALUDE_second_player_seven_moves_l4090_409016

/-- Represents a game on a polygon where two players mark vertices. -/
structure PolygonGame where
  sides : Nat
  marked : Finset Nat

/-- The maximum number of moves the second player can guarantee. -/
def maxGuaranteedMoves (game : PolygonGame) : Nat :=
  sorry

/-- Theorem stating that in a 129-sided polygon game, the second player can always make at least 7 moves. -/
theorem second_player_seven_moves :
  ∀ (game : PolygonGame), game.sides = 129 →
  maxGuaranteedMoves game ≥ 7 :=
by sorry

end NUMINAMATH_CALUDE_second_player_seven_moves_l4090_409016


namespace NUMINAMATH_CALUDE_investment_interest_proof_l4090_409097

/-- Calculates the total annual interest for a two-fund investment --/
def total_annual_interest (total_investment : ℝ) (rate1 rate2 : ℝ) (amount_in_fund1 : ℝ) : ℝ :=
  let amount_in_fund2 := total_investment - amount_in_fund1
  let interest1 := amount_in_fund1 * rate1
  let interest2 := amount_in_fund2 * rate2
  interest1 + interest2

/-- Proves that the total annual interest for the given investment scenario is $4,120 --/
theorem investment_interest_proof :
  total_annual_interest 50000 0.08 0.085 26000 = 4120 := by
  sorry

end NUMINAMATH_CALUDE_investment_interest_proof_l4090_409097


namespace NUMINAMATH_CALUDE_apple_pie_division_l4090_409021

/-- The number of apple pies Sedrach has -/
def total_pies : ℕ := 13

/-- The number of bite-size samples each part of an apple pie can be split into -/
def samples_per_part : ℕ := 5

/-- The total number of people who can taste the pies -/
def total_tasters : ℕ := 130

/-- The number of parts each apple pie is divided into -/
def parts_per_pie : ℕ := 2

theorem apple_pie_division :
  total_pies * parts_per_pie * samples_per_part = total_tasters := by sorry

end NUMINAMATH_CALUDE_apple_pie_division_l4090_409021


namespace NUMINAMATH_CALUDE_dividend_calculation_l4090_409001

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 14)
  (h2 : quotient = 9)
  (h3 : remainder = 5) :
  divisor * quotient + remainder = 131 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l4090_409001


namespace NUMINAMATH_CALUDE_rightmost_three_digits_of_3_to_1987_l4090_409029

theorem rightmost_three_digits_of_3_to_1987 : 3^1987 % 1000 = 187 := by sorry

end NUMINAMATH_CALUDE_rightmost_three_digits_of_3_to_1987_l4090_409029


namespace NUMINAMATH_CALUDE_not_prime_n_fourth_plus_four_to_n_l4090_409092

theorem not_prime_n_fourth_plus_four_to_n (n : ℕ) (h : n > 1) :
  ¬ Nat.Prime (n^4 + 4^n) := by
  sorry

end NUMINAMATH_CALUDE_not_prime_n_fourth_plus_four_to_n_l4090_409092


namespace NUMINAMATH_CALUDE_line_through_three_points_l4090_409034

/-- Given that the points (0, 2), (10, m), and (25, -3) lie on the same line, prove that m = 0. -/
theorem line_through_three_points (m : ℝ) : 
  (∀ t : ℝ, ∃ a b : ℝ, t * (10 - 0) + 0 = 10 ∧ 
                       t * (m - 2) + 2 = m ∧ 
                       t * (25 - 0) + 0 = 25 ∧ 
                       t * (-3 - 2) + 2 = -3) → 
  m = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_through_three_points_l4090_409034


namespace NUMINAMATH_CALUDE_apple_cost_proof_l4090_409056

/-- The cost of apples for the first 30 kgs (in rupees per kg) -/
def l : ℝ := sorry

/-- The cost of apples for each additional kg beyond 30 kgs (in rupees per kg) -/
def q : ℝ := sorry

/-- The total cost of 33 kgs of apples (in rupees) -/
def cost_33 : ℝ := 11.67

/-- The total cost of 36 kgs of apples (in rupees) -/
def cost_36 : ℝ := 12.48

/-- The cost of the first 10 kgs of apples (in rupees) -/
def cost_10 : ℝ := 10 * l

theorem apple_cost_proof :
  (30 * l + 3 * q = cost_33) ∧
  (30 * l + 6 * q = cost_36) →
  cost_10 = 3.62 := by
  sorry

end NUMINAMATH_CALUDE_apple_cost_proof_l4090_409056


namespace NUMINAMATH_CALUDE_function_inequality_l4090_409023

open Real

theorem function_inequality (x₁ x₂ : ℝ) (h₁ : 0 < x₁) (h₂ : x₁ < x₂) (h₃ : -1 < log x₁) (h₄ : -1 < log x₂) : 
  let f := fun (x : ℝ) => x * log x
  x₁ * f x₁ + x₂ * f x₂ > 2 * x₂ * f x₁ := by
sorry

end NUMINAMATH_CALUDE_function_inequality_l4090_409023


namespace NUMINAMATH_CALUDE_shoes_difference_l4090_409091

/-- Represents the number of shoes tried on at each store --/
structure ShoesTried where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- The total number of shoes tried on across all stores --/
def totalShoesTried (s : ShoesTried) : ℕ :=
  s.first + s.second + s.third + s.fourth

/-- The conditions from the problem --/
def problemConditions (s : ShoesTried) : Prop :=
  s.first = 7 ∧
  s.third = 0 ∧
  s.fourth = 2 * (s.first + s.second + s.third) ∧
  totalShoesTried s = 48

/-- The theorem to prove --/
theorem shoes_difference (s : ShoesTried) 
  (h : problemConditions s) : s.second - s.first = 2 := by
  sorry


end NUMINAMATH_CALUDE_shoes_difference_l4090_409091


namespace NUMINAMATH_CALUDE_no_adjacent_standing_probability_l4090_409088

/-- Represents the number of valid arrangements for n people where no two adjacent people stand -/
def validArrangements : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | 2 => 3
  | n + 3 => validArrangements (n + 1) + validArrangements (n + 2)

/-- The probability of no two adjacent people standing in a circular arrangement of n people -/
def probability (n : ℕ) : ℚ := (validArrangements n : ℚ) / (2^n : ℚ)

theorem no_adjacent_standing_probability :
  probability 10 = 123 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_no_adjacent_standing_probability_l4090_409088


namespace NUMINAMATH_CALUDE_trig_identity_l4090_409068

theorem trig_identity (x : Real) (h : Real.tan x = -1/2) : 
  Real.sin x ^ 2 + 3 * Real.sin x * Real.cos x - 1 = -2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l4090_409068


namespace NUMINAMATH_CALUDE_fraction_sum_l4090_409007

theorem fraction_sum : (2 : ℚ) / 5 + 3 / 8 + 1 = 71 / 40 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_l4090_409007


namespace NUMINAMATH_CALUDE_intersection_on_ellipse_l4090_409053

/-- Ellipse C with given properties -/
structure EllipseC where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- The eccentricity of ellipse C is √3/2 -/
axiom eccentricity (C : EllipseC) : (Real.sqrt (C.a^2 - C.b^2)) / C.a = Real.sqrt 3 / 2

/-- A circle centered at the origin with diameter equal to the minor axis of C
    is tangent to the line x - y + 2 = 0 -/
axiom circle_tangent (C : EllipseC) : C.b = Real.sqrt 2

/-- Point on ellipse C -/
def on_ellipse (C : EllipseC) (x y : ℝ) : Prop :=
  x^2 / C.a^2 + y^2 / C.b^2 = 1

/-- Theorem: If M and N are symmetric points on C, and T is the intersection of PM and QN,
    then T lies on the ellipse C -/
theorem intersection_on_ellipse (C : EllipseC) (x₀ y₀ x y : ℝ) :
  on_ellipse C x₀ y₀ →
  on_ellipse C (-x₀) y₀ →
  x = 2 * x₀ * (y - 1) →
  y = (3 * y₀ - 4) / (2 * y₀ - 3) →
  on_ellipse C x y := by
  sorry

end NUMINAMATH_CALUDE_intersection_on_ellipse_l4090_409053


namespace NUMINAMATH_CALUDE_polygon_sides_from_exterior_angle_l4090_409028

theorem polygon_sides_from_exterior_angle :
  ∀ (n : ℕ) (exterior_angle : ℝ),
    exterior_angle = 30 →
    (n : ℝ) * exterior_angle = 360 →
    n = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_from_exterior_angle_l4090_409028


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l4090_409000

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = x^2 + 1}
def N : Set ℝ := {x | ∃ y, y = Real.sqrt (x + 1)}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | x ≥ 0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l4090_409000


namespace NUMINAMATH_CALUDE_c_value_for_four_roots_l4090_409033

/-- A complex number is a root of the polynomial Q(x) if Q(x) = 0 -/
def is_root (Q : ℂ → ℂ) (z : ℂ) : Prop := Q z = 0

/-- The polynomial Q(x) -/
def Q (c : ℂ) (x : ℂ) : ℂ := (x^2 - 3*x + 3) * (x^2 - c*x + 2) * (x^2 - 5*x + 5)

/-- The theorem stating the value of |c| for Q(x) with exactly 4 distinct roots -/
theorem c_value_for_four_roots :
  ∃ (c : ℂ), (∃ (z₁ z₂ z₃ z₄ : ℂ), z₁ ≠ z₂ ∧ z₁ ≠ z₃ ∧ z₁ ≠ z₄ ∧ z₂ ≠ z₃ ∧ z₂ ≠ z₄ ∧ z₃ ≠ z₄ ∧
    is_root (Q c) z₁ ∧ is_root (Q c) z₂ ∧ is_root (Q c) z₃ ∧ is_root (Q c) z₄ ∧
    (∀ (z : ℂ), is_root (Q c) z → z = z₁ ∨ z = z₂ ∨ z = z₃ ∨ z = z₄)) →
  Complex.abs c = Real.sqrt (18 - Real.sqrt 15 / 2) :=
sorry

end NUMINAMATH_CALUDE_c_value_for_four_roots_l4090_409033


namespace NUMINAMATH_CALUDE_point_on_line_l4090_409031

/-- Given two points on a line and a third point with a known y-coordinate,
    prove that the x-coordinate of the third point is -6. -/
theorem point_on_line (x : ℝ) :
  let p1 : ℝ × ℝ := (0, 8)
  let p2 : ℝ × ℝ := (-4, 0)
  let p3 : ℝ × ℝ := (x, -4)
  (p3.2 - p1.2) / (p3.1 - p1.1) = (p2.2 - p1.2) / (p2.1 - p1.1) →
  x = -6 := by
sorry

end NUMINAMATH_CALUDE_point_on_line_l4090_409031


namespace NUMINAMATH_CALUDE_committee_selection_ways_l4090_409027

theorem committee_selection_ways (n m : ℕ) (hn : n = 30) (hm : m = 5) :
  Nat.choose n m = 54810 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_ways_l4090_409027


namespace NUMINAMATH_CALUDE_t_shirt_jersey_cost_difference_l4090_409041

/-- The cost difference between a t-shirt and a jersey -/
def cost_difference (t_shirt_price jersey_price : ℕ) : ℕ :=
  t_shirt_price - jersey_price

/-- Theorem: The cost difference between a t-shirt and a jersey is $158 -/
theorem t_shirt_jersey_cost_difference :
  cost_difference 192 34 = 158 := by
  sorry

end NUMINAMATH_CALUDE_t_shirt_jersey_cost_difference_l4090_409041


namespace NUMINAMATH_CALUDE_triangle_inequality_l4090_409089

theorem triangle_inequality (α β γ : ℝ) (h_a l_a r R : ℝ) 
  (h1 : h_a / l_a = Real.cos ((β - γ) / 2))
  (h2 : 2 * r / R = 8 * Real.sin (α / 2) * Real.sin (β / 2) * Real.sin (γ / 2)) :
  h_a / l_a ≥ Real.sqrt (2 * r / R) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l4090_409089


namespace NUMINAMATH_CALUDE_inequality_proof_l4090_409080

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_sum : a + b + c = 1) : 
  (a * b + b * c + c * a ≤ 1 / 3) ∧ 
  (a^2 / b + b^2 / c + c^2 / a ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4090_409080


namespace NUMINAMATH_CALUDE_quartic_root_sum_l4090_409087

theorem quartic_root_sum (a b c d : ℂ) : 
  (a^4 - 34*a^3 + 15*a^2 - 42*a - 8 = 0) →
  (b^4 - 34*b^3 + 15*b^2 - 42*b - 8 = 0) →
  (c^4 - 34*c^3 + 15*c^2 - 42*c - 8 = 0) →
  (d^4 - 34*d^3 + 15*d^2 - 42*d - 8 = 0) →
  (a / ((1/a) + b*c*d) + b / ((1/b) + a*c*d) + c / ((1/c) + a*b*d) + d / ((1/d) + a*b*c) = -161) :=
by sorry

end NUMINAMATH_CALUDE_quartic_root_sum_l4090_409087


namespace NUMINAMATH_CALUDE_smallest_sum_of_three_l4090_409086

def S : Finset Int := {-8, 2, -5, 17, -3}

theorem smallest_sum_of_three (a b c : Int) 
  (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  (∀ x y z : Int, x ∈ S → y ∈ S → z ∈ S → 
    x ≠ y ∧ y ≠ z ∧ x ≠ z → a + b + c ≤ x + y + z) ∧ 
  (∃ x y z : Int, x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ 
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x + y + z = -16) :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_three_l4090_409086


namespace NUMINAMATH_CALUDE_inequality_solutions_imply_range_l4090_409057

theorem inequality_solutions_imply_range (a : ℝ) : 
  (∃ x₁ x₂ : ℕ+, x₁ ≠ x₂ ∧ 
    (∀ x : ℕ+, 2 * (x : ℝ) + a ≤ 1 ↔ (x = x₁ ∨ x = x₂))) →
  -5 < a ∧ a ≤ -3 := by
sorry

end NUMINAMATH_CALUDE_inequality_solutions_imply_range_l4090_409057


namespace NUMINAMATH_CALUDE_factorization_xy_plus_3x_l4090_409052

theorem factorization_xy_plus_3x (x y : ℝ) : x * y + 3 * x = x * (y + 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_xy_plus_3x_l4090_409052


namespace NUMINAMATH_CALUDE_greatest_sum_36_l4090_409018

/-- The sum of the first n positive integers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The property that n is the greatest number of consecutive positive integers starting from 1 whose sum is 36 -/
def is_greatest_sum_36 (n : ℕ) : Prop :=
  sum_first_n n = 36 ∧ ∀ m : ℕ, m > n → sum_first_n m > 36

theorem greatest_sum_36 : is_greatest_sum_36 8 := by
  sorry

end NUMINAMATH_CALUDE_greatest_sum_36_l4090_409018


namespace NUMINAMATH_CALUDE_discount_percentage_proof_l4090_409019

theorem discount_percentage_proof (original_price sale_price : ℝ) 
  (h1 : original_price = 128)
  (h2 : sale_price = 83.2) :
  (original_price - sale_price) / original_price * 100 = 35 := by
sorry

end NUMINAMATH_CALUDE_discount_percentage_proof_l4090_409019


namespace NUMINAMATH_CALUDE_intersection_complement_equals_set_l4090_409044

def M : Set ℝ := {-1, 1, 2, 4}
def N : Set ℝ := {x | x^2 - 2*x ≥ 3}

theorem intersection_complement_equals_set : M ∩ (Set.univ \ N) = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_set_l4090_409044


namespace NUMINAMATH_CALUDE_solve_equation_l4090_409017

theorem solve_equation : ∃ X : ℝ, 
  (((4 - 3.5 * (2 + 1/7 - (1 + 1/5))) / 0.16) / X = 
  ((3 + 2/7 - (3/14 / (1/6))) / (41 + 23/84 - (40 + 49/60))) ∧ X = 1) := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l4090_409017


namespace NUMINAMATH_CALUDE_degenerate_iff_c_eq_52_l4090_409070

/-- A point in R^2 -/
structure Point where
  x : ℝ
  y : ℝ

/-- The equation of the graph -/
def equation (p : Point) (c : ℝ) : Prop :=
  3 * p.x^2 + p.y^2 + 6 * p.x - 14 * p.y + c = 0

/-- The graph is degenerate (represents a single point) -/
def is_degenerate (c : ℝ) : Prop :=
  ∃! p : Point, equation p c

theorem degenerate_iff_c_eq_52 :
  ∀ c : ℝ, is_degenerate c ↔ c = 52 := by sorry

end NUMINAMATH_CALUDE_degenerate_iff_c_eq_52_l4090_409070


namespace NUMINAMATH_CALUDE_florist_roses_problem_l4090_409077

/-- A florist problem involving roses -/
theorem florist_roses_problem (initial_roses : ℕ) (picked_roses : ℕ) (final_roses : ℕ) :
  initial_roses = 50 →
  picked_roses = 21 →
  final_roses = 56 →
  ∃ (sold_roses : ℕ), initial_roses - sold_roses + picked_roses = final_roses ∧ sold_roses = 15 :=
by sorry

end NUMINAMATH_CALUDE_florist_roses_problem_l4090_409077


namespace NUMINAMATH_CALUDE_total_students_suggestion_l4090_409082

theorem total_students_suggestion (mashed_potatoes bacon tomatoes : ℕ) 
  (h1 : mashed_potatoes = 324)
  (h2 : bacon = 374)
  (h3 : tomatoes = 128) :
  mashed_potatoes + bacon + tomatoes = 826 := by
  sorry

end NUMINAMATH_CALUDE_total_students_suggestion_l4090_409082


namespace NUMINAMATH_CALUDE_sin_300_deg_l4090_409045

theorem sin_300_deg : Real.sin (300 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_300_deg_l4090_409045


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l4090_409020

-- Define an arithmetic sequence
def isArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h1 : isArithmeticSequence a) 
    (h2 : a 1 + 3 * a 6 + a 11 = 120) : 
  2 * a 7 - a 8 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l4090_409020


namespace NUMINAMATH_CALUDE_max_points_for_one_participant_l4090_409042

theorem max_points_for_one_participant 
  (n : ℕ) 
  (avg : ℚ) 
  (min_points : ℕ) 
  (h1 : n = 50) 
  (h2 : avg = 8) 
  (h3 : min_points = 2) 
  (h4 : ∀ p : ℕ, p ≤ n → p ≥ min_points) : 
  ∃ max_points : ℕ, max_points = 302 ∧ 
  ∀ p : ℕ, p ≤ n → p ≤ max_points := by
sorry


end NUMINAMATH_CALUDE_max_points_for_one_participant_l4090_409042


namespace NUMINAMATH_CALUDE_unique_three_digit_factorial_sum_l4090_409024

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def digits_factorial_sum (n : ℕ) : ℕ :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  factorial d1 + factorial d2 + factorial d3

theorem unique_three_digit_factorial_sum :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧
  digits_factorial_sum n = n ∧
  (n / 100 = 3 ∨ (n / 10) % 10 = 3 ∨ n % 10 = 3) ∧
  n = 145 :=
sorry

end NUMINAMATH_CALUDE_unique_three_digit_factorial_sum_l4090_409024


namespace NUMINAMATH_CALUDE_quadratic_sequence_sum_l4090_409030

theorem quadratic_sequence_sum (x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ : ℝ) 
  (eq1 : x₁ + 4*x₂ + 9*x₃ + 16*x₄ + 25*x₅ + 36*x₆ + 49*x₇ + 64*x₈ = 10)
  (eq2 : 4*x₁ + 9*x₂ + 16*x₃ + 25*x₄ + 36*x₅ + 49*x₆ + 64*x₇ + 81*x₈ = 40)
  (eq3 : 9*x₁ + 16*x₂ + 25*x₃ + 36*x₄ + 49*x₅ + 64*x₆ + 81*x₇ + 100*x₈ = 170) :
  16*x₁ + 25*x₂ + 36*x₃ + 49*x₄ + 64*x₅ + 81*x₆ + 100*x₇ + 121*x₈ = 400 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sequence_sum_l4090_409030


namespace NUMINAMATH_CALUDE_triangle_probability_ten_points_triangle_probability_ten_points_with_conditions_l4090_409036

/-- Given 10 points in a plane where no three are collinear, this function
    calculates the probability that three out of four randomly chosen
    distinct segments connecting pairs of these points will form a triangle. -/
def probability_triangle_from_segments (n : ℕ) : ℚ :=
  if n = 10 then 16 / 473
  else 0

/-- Theorem stating that the probability of forming a triangle
    from three out of four randomly chosen segments is 16/473
    when there are 10 points in the plane and no three are collinear. -/
theorem triangle_probability_ten_points :
  probability_triangle_from_segments 10 = 16 / 473 := by
  sorry

/-- Assumption that no three points are collinear in the given set of points. -/
axiom no_three_collinear (n : ℕ) : Prop

/-- Theorem stating that given 10 points in a plane where no three are collinear,
    the probability that three out of four randomly chosen distinct segments
    connecting pairs of these points will form a triangle is 16/473. -/
theorem triangle_probability_ten_points_with_conditions :
  no_three_collinear 10 →
  probability_triangle_from_segments 10 = 16 / 473 := by
  sorry

end NUMINAMATH_CALUDE_triangle_probability_ten_points_triangle_probability_ten_points_with_conditions_l4090_409036


namespace NUMINAMATH_CALUDE_solve_system_l4090_409072

theorem solve_system (a b : ℚ) 
  (h1 : -3 / (a - 3) = 3 / (a + 2))
  (h2 : (a^2 - b^2)/(a - b) = 7) :
  a = 1/2 ∧ b = 13/2 := by
sorry

end NUMINAMATH_CALUDE_solve_system_l4090_409072


namespace NUMINAMATH_CALUDE_units_digit_of_special_two_digit_number_l4090_409078

/-- 
Given a two-digit number M = 10a + b, where a and b are single digits,
if M = ab + (a + b) + 5, then b = 8.
-/
theorem units_digit_of_special_two_digit_number (a b : ℕ) : 
  a ≥ 1 ∧ a ≤ 9 ∧ b ≥ 0 ∧ b ≤ 9 →
  (10 * a + b = a * b + a + b + 5) →
  b = 8 := by
sorry

end NUMINAMATH_CALUDE_units_digit_of_special_two_digit_number_l4090_409078


namespace NUMINAMATH_CALUDE_line_equation_l4090_409037

/-- Given a line passing through (a, 0) and cutting a triangular region with area T from the second quadrant, 
    the equation of the line is -2Tx + a²y + 2aT = 0 -/
theorem line_equation (a T : ℝ) (h1 : a ≠ 0) (h2 : T > 0) : 
  ∃ (f : ℝ → ℝ), (∀ x y, f x = y ↔ -2 * T * x + a^2 * y + 2 * a * T = 0) ∧ 
                  (f a = 0) ∧
                  (∀ x y, x > 0 → y > 0 → f x = y → 
                    (1/2) * a * y = T) :=
sorry

end NUMINAMATH_CALUDE_line_equation_l4090_409037


namespace NUMINAMATH_CALUDE_senior_mean_score_l4090_409026

theorem senior_mean_score 
  (total_students : ℕ) 
  (overall_mean : ℝ) 
  (senior_count : ℝ) 
  (non_senior_count : ℝ) 
  (senior_mean : ℝ) 
  (non_senior_mean : ℝ) :
  total_students = 120 →
  overall_mean = 150 →
  non_senior_count = senior_count + 0.75 * senior_count →
  senior_mean = 2 * non_senior_mean →
  senior_count + non_senior_count = total_students →
  senior_count * senior_mean + non_senior_count * non_senior_mean = total_students * overall_mean →
  senior_mean = 220 :=
by sorry

end NUMINAMATH_CALUDE_senior_mean_score_l4090_409026


namespace NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_l4090_409073

/-- Represents different sampling methods --/
inductive SamplingMethod
  | Lottery
  | RandomNumber
  | Systematic
  | Stratified

/-- Represents a school population --/
structure SchoolPopulation where
  male_students : Nat
  female_students : Nat

/-- Represents a survey plan --/
structure SurveyPlan where
  population : SchoolPopulation
  sample_size : Nat
  goal : String

/-- Determines the most appropriate sampling method for a given survey plan --/
def most_appropriate_sampling_method (plan : SurveyPlan) : SamplingMethod :=
  sorry

/-- The theorem stating that stratified sampling is most appropriate for the given scenario --/
theorem stratified_sampling_most_appropriate (plan : SurveyPlan) :
  plan.population.male_students = 500 →
  plan.population.female_students = 500 →
  plan.sample_size = 100 →
  plan.goal = "investigate differences in study interests and hobbies between male and female students" →
  most_appropriate_sampling_method plan = SamplingMethod.Stratified :=
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_l4090_409073


namespace NUMINAMATH_CALUDE_sqrt_7225_minus_55_cube_l4090_409035

theorem sqrt_7225_minus_55_cube (c d : ℕ) (hc : c > 0) (hd : d > 0) 
  (h : Real.sqrt 7225 - 55 = (Real.sqrt c - d)^3) : c + d = 19 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_7225_minus_55_cube_l4090_409035


namespace NUMINAMATH_CALUDE_triangle_similarity_and_area_l4090_409049

/-- Triangle similarity and area theorem -/
theorem triangle_similarity_and_area (PQ QR YZ : ℝ) (area_XYZ : ℝ) :
  PQ = 8 →
  QR = 16 →
  YZ = 24 →
  area_XYZ = 144 →
  ∃ (XY : ℝ),
    (XY / PQ = YZ / QR) ∧
    (area_XYZ = (1/2) * YZ * (2 * area_XYZ / YZ)) ∧
    XY = 12 := by
  sorry

end NUMINAMATH_CALUDE_triangle_similarity_and_area_l4090_409049


namespace NUMINAMATH_CALUDE_bella_current_beads_l4090_409075

/-- The number of friends Bella is making bracelets for -/
def num_friends : ℕ := 6

/-- The number of beads needed per bracelet -/
def beads_per_bracelet : ℕ := 8

/-- The number of additional beads Bella needs -/
def additional_beads_needed : ℕ := 12

/-- The total number of beads Bella needs for all bracelets -/
def total_beads_needed : ℕ := num_friends * beads_per_bracelet

/-- Theorem: Bella currently has 36 beads -/
theorem bella_current_beads : 
  total_beads_needed - additional_beads_needed = 36 := by
  sorry

end NUMINAMATH_CALUDE_bella_current_beads_l4090_409075


namespace NUMINAMATH_CALUDE_cube_sum_one_l4090_409059

theorem cube_sum_one (a b c : ℝ) 
  (sum_one : a + b + c = 1)
  (sum_products : a * b + a * c + b * c = -5)
  (product : a * b * c = 5) :
  a^3 + b^3 + c^3 = 1 := by sorry

end NUMINAMATH_CALUDE_cube_sum_one_l4090_409059


namespace NUMINAMATH_CALUDE_watch_cost_price_l4090_409048

theorem watch_cost_price (selling_price loss_percent gain_percent additional_amount : ℝ) 
  (h1 : selling_price = (1 - loss_percent / 100) * 1500)
  (h2 : selling_price + additional_amount = (1 + gain_percent / 100) * 1500)
  (h3 : loss_percent = 10)
  (h4 : gain_percent = 5)
  (h5 : additional_amount = 225) : 
  1500 = 1500 := by sorry

end NUMINAMATH_CALUDE_watch_cost_price_l4090_409048


namespace NUMINAMATH_CALUDE_talent_school_problem_l4090_409025

theorem talent_school_problem (total : ℕ) (cant_sing cant_dance cant_act : ℕ) :
  total = 150 ∧ 
  cant_sing = 90 ∧ 
  cant_dance = 100 ∧ 
  cant_act = 60 →
  ∃ (two_talents : ℕ),
    two_talents = 50 ∧
    two_talents = total - (total - cant_sing) - (total - cant_dance) - (total - cant_act) + 2 * total - cant_sing - cant_dance - cant_act :=
by sorry

end NUMINAMATH_CALUDE_talent_school_problem_l4090_409025


namespace NUMINAMATH_CALUDE_solution_set_of_system_l4090_409043

theorem solution_set_of_system : ∃! S : Set (ℝ × ℝ),
  S = {(-1, 2), (2, -1), (-2, 7)} ∧
  ∀ (x y : ℝ), (x, y) ∈ S ↔ 
    (y^2 + 2*x*y + x^2 - 6*y - 6*x + 5 = 0) ∧ 
    (y - x + 1 = x^2 - 3*x) ∧ 
    (x ≠ 0) ∧ 
    (x ≠ 3) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_system_l4090_409043


namespace NUMINAMATH_CALUDE_larger_integer_problem_l4090_409090

theorem larger_integer_problem (x y : ℕ) (h1 : y = 4 * x) (h2 : (x + 6) / y = 1 / 2) : y = 24 := by
  sorry

end NUMINAMATH_CALUDE_larger_integer_problem_l4090_409090


namespace NUMINAMATH_CALUDE_determinant_k_value_l4090_409066

def determinant (a b c d e f g h i : ℝ) : ℝ :=
  a * e * i + b * f * g + c * d * h - c * e * g - b * d * i - a * f * h

def algebraic_cofactor_1_2 (a b c d e f g h i : ℝ) : ℝ :=
  -(b * i - c * h)

theorem determinant_k_value (k : ℝ) :
  algebraic_cofactor_1_2 4 2 k (-3) 5 4 (-1) 1 (-2) = -10 →
  k = -14 := by
  sorry

end NUMINAMATH_CALUDE_determinant_k_value_l4090_409066


namespace NUMINAMATH_CALUDE_fast_food_cost_l4090_409071

/-- The total cost of buying fast food -/
def total_cost (a b : ℕ) : ℕ := 30 * a + 20 * b

/-- The price of one serving of type A fast food -/
def price_A : ℕ := 30

/-- The price of one serving of type B fast food -/
def price_B : ℕ := 20

/-- Theorem: The total cost of buying 'a' servings of type A fast food and 'b' servings of type B fast food is 30a + 20b yuan -/
theorem fast_food_cost (a b : ℕ) : total_cost a b = price_A * a + price_B * b := by
  sorry

end NUMINAMATH_CALUDE_fast_food_cost_l4090_409071


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l4090_409051

theorem quadratic_equation_solution : 
  ∃ y : ℝ, y^2 - 6*y + 5 = 0 ↔ y = 1 ∨ y = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l4090_409051


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l4090_409094

theorem fractional_equation_solution (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 0) :
  (4 / (x - 1) = 3 / x) ↔ x = -3 := by
sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l4090_409094


namespace NUMINAMATH_CALUDE_geometric_sequence_values_l4090_409093

theorem geometric_sequence_values (a b c : ℝ) : 
  (∃ q : ℝ, q ≠ 0 ∧ 2 * q = a ∧ a * q = b ∧ b * q = c ∧ c * q = 32) → 
  ((a = 4 ∧ b = 8 ∧ c = 16) ∨ (a = -4 ∧ b = 8 ∧ c = -16)) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_values_l4090_409093


namespace NUMINAMATH_CALUDE_part_one_part_two_l4090_409095

-- Part 1
theorem part_one (x : ℝ) (h : x = Real.sqrt 3 - 1) :
  (x - 1) / (x^2 + 2*x + 1) / (1 - 2 / (x + 1)) = Real.sqrt 3 / 3 := by
  sorry

-- Part 2
theorem part_two (a b : ℝ) (h1 : a + b = 1) (h2 : a * b = 3 / 16) :
  a^3 * b - 2 * a^2 * b^2 + a * b^3 = 3 / 64 := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l4090_409095


namespace NUMINAMATH_CALUDE_tangent_slope_implies_a_l4090_409039

/-- Given a curve y = x^4 + ax + 1 with a tangent at (-1, a+2) having slope 8, prove a = -6 -/
theorem tangent_slope_implies_a (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^4 + a*x + 1
  let point : ℝ × ℝ := (-1, a + 2)
  let slope : ℝ := 8
  (f (-1) = a + 2) ∧ 
  (deriv f (-1) = slope) → 
  a = -6 := by
sorry

end NUMINAMATH_CALUDE_tangent_slope_implies_a_l4090_409039


namespace NUMINAMATH_CALUDE_gift_wrapping_combinations_l4090_409010

/-- The number of wrapping paper varieties -/
def wrapping_paper_varieties : ℕ := 12

/-- The number of ribbon colors -/
def ribbon_colors : ℕ := 3

/-- The number of gift card types -/
def gift_card_types : ℕ := 6

/-- The number of ribbon colors available for small gifts -/
def small_gift_ribbon_colors : ℕ := 2

/-- Calculates the number of wrapping combinations for small gifts -/
def small_gift_combinations : ℕ :=
  wrapping_paper_varieties * small_gift_ribbon_colors * gift_card_types

/-- Calculates the number of wrapping combinations for large gifts -/
def large_gift_combinations : ℕ :=
  wrapping_paper_varieties * ribbon_colors * gift_card_types

theorem gift_wrapping_combinations :
  small_gift_combinations = 144 ∧ large_gift_combinations = 216 := by
  sorry

end NUMINAMATH_CALUDE_gift_wrapping_combinations_l4090_409010


namespace NUMINAMATH_CALUDE_complement_A_in_U_l4090_409085

def U : Set ℝ := {x | x > 0}
def A : Set ℝ := {x | x > 2}

theorem complement_A_in_U : 
  (U \ A) = {x : ℝ | 0 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_l4090_409085


namespace NUMINAMATH_CALUDE_smallest_difference_for_8_factorial_l4090_409083

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem smallest_difference_for_8_factorial :
  ∀ a b c : ℕ+,
  a * b * c = factorial 8 →
  a < b →
  b < c →
  ∀ a' b' c' : ℕ+,
  a' * b' * c' = factorial 8 →
  a' < b' →
  b' < c' →
  c - a ≤ c' - a' :=
sorry

end NUMINAMATH_CALUDE_smallest_difference_for_8_factorial_l4090_409083


namespace NUMINAMATH_CALUDE_jersey_profit_calculation_l4090_409003

-- Define the given conditions
def tshirt_profit : ℝ := 25
def tshirts_sold : ℕ := 113
def jerseys_sold : ℕ := 78
def jersey_price_difference : ℝ := 90

-- Define the theorem to be proved
theorem jersey_profit_calculation :
  let jersey_profit := tshirt_profit + jersey_price_difference
  jersey_profit = 115 := by sorry

end NUMINAMATH_CALUDE_jersey_profit_calculation_l4090_409003


namespace NUMINAMATH_CALUDE_cheryl_material_problem_l4090_409099

theorem cheryl_material_problem (x : ℚ) : 
  (x + 2/3 : ℚ) - 8/18 = 2/3 → x = 4/9 := by sorry

end NUMINAMATH_CALUDE_cheryl_material_problem_l4090_409099


namespace NUMINAMATH_CALUDE_cube_surface_area_increase_l4090_409076

theorem cube_surface_area_increase (L : ℝ) (L_new : ℝ) (h : L > 0) :
  L_new = 1.3 * L →
  (6 * L_new^2 - 6 * L^2) / (6 * L^2) = 0.69 := by
sorry

end NUMINAMATH_CALUDE_cube_surface_area_increase_l4090_409076


namespace NUMINAMATH_CALUDE_anna_bills_count_l4090_409063

theorem anna_bills_count (five_dollar_bills : ℕ) (ten_dollar_bills : ℕ) : 
  five_dollar_bills = 4 → ten_dollar_bills = 8 → five_dollar_bills + ten_dollar_bills = 12 := by
  sorry

end NUMINAMATH_CALUDE_anna_bills_count_l4090_409063


namespace NUMINAMATH_CALUDE_classics_section_books_l4090_409015

/-- The number of classic authors in Jack's collection -/
def num_authors : ℕ := 6

/-- The number of books per author -/
def books_per_author : ℕ := 33

/-- The total number of books in the classics section -/
def total_books : ℕ := num_authors * books_per_author

theorem classics_section_books :
  total_books = 198 := by sorry

end NUMINAMATH_CALUDE_classics_section_books_l4090_409015


namespace NUMINAMATH_CALUDE_sequence_sum_property_l4090_409046

theorem sequence_sum_property (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h : ∀ n : ℕ, n ≥ 1 → S n = 2 * a n - n) :
  2 / (a 1 * a 2) + 4 / (a 2 * a 3) + 8 / (a 3 * a 4) + 16 / (a 4 * a 5) = 30 / 31 :=
by sorry

end NUMINAMATH_CALUDE_sequence_sum_property_l4090_409046


namespace NUMINAMATH_CALUDE_initial_mean_calculation_l4090_409069

/-- Given 20 observations with an initial mean, prove that correcting one observation
    from 40 to 25 results in a new mean of 34.9 if and only if the initial mean was 35.65 -/
theorem initial_mean_calculation (n : ℕ) (initial_mean corrected_mean : ℝ) :
  n = 20 ∧
  corrected_mean = 34.9 ∧
  (n : ℝ) * initial_mean - 15 = (n : ℝ) * corrected_mean →
  initial_mean = 35.65 := by
  sorry

end NUMINAMATH_CALUDE_initial_mean_calculation_l4090_409069


namespace NUMINAMATH_CALUDE_tangent_line_perpendicular_l4090_409064

theorem tangent_line_perpendicular (a : ℝ) : 
  let f (x : ℝ) := Real.exp (2 * a * x)
  let f' (x : ℝ) := 2 * a * Real.exp (2 * a * x)
  let tangent_slope := f' 0
  let perpendicular_line_slope := -1 / 2
  (tangent_slope = perpendicular_line_slope) → a = -1/4 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_perpendicular_l4090_409064


namespace NUMINAMATH_CALUDE_min_values_ab_and_a_plus_2b_l4090_409005

theorem min_values_ab_and_a_plus_2b 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (h : 1/a + 2/b = 2) : 
  a * b ≥ 2 ∧ 
  a + 2*b ≥ 9/2 ∧ 
  (a + 2*b = 9/2 ↔ a = 3/2 ∧ b = 3/2) :=
by sorry

end NUMINAMATH_CALUDE_min_values_ab_and_a_plus_2b_l4090_409005


namespace NUMINAMATH_CALUDE_two_hundred_squared_minus_399_is_composite_l4090_409062

theorem two_hundred_squared_minus_399_is_composite : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ 200^2 - 399 = a * b :=
by
  sorry

end NUMINAMATH_CALUDE_two_hundred_squared_minus_399_is_composite_l4090_409062
