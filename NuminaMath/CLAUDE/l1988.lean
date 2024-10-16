import Mathlib

namespace NUMINAMATH_CALUDE_normal_distribution_probability_l1988_198821

/-- A random variable following a normal distribution -/
structure NormalRandomVariable where
  μ : ℝ
  σ : ℝ
  hσ : σ > 0

/-- The cumulative distribution function (CDF) of a normal random variable -/
noncomputable def normalCDF (X : NormalRandomVariable) (x : ℝ) : ℝ := sorry

/-- The probability that a random variable falls within an interval -/
noncomputable def probInterval (X : NormalRandomVariable) (a b : ℝ) : ℝ := 
  normalCDF X b - normalCDF X a

theorem normal_distribution_probability 
  (X : NormalRandomVariable) 
  (h1 : X.μ = 3) 
  (h2 : normalCDF X 4 = 0.84) : 
  probInterval X 2 4 = 0.68 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_probability_l1988_198821


namespace NUMINAMATH_CALUDE_not_prime_257_pow_1092_plus_1092_l1988_198875

theorem not_prime_257_pow_1092_plus_1092 : ¬ Nat.Prime (257^1092 + 1092) := by
  sorry

end NUMINAMATH_CALUDE_not_prime_257_pow_1092_plus_1092_l1988_198875


namespace NUMINAMATH_CALUDE_probability_sum_le_four_l1988_198883

-- Define the type for a die
def Die := Fin 6

-- Define the sum of two dice
def diceSum (d1 d2 : Die) : Nat := d1.val + d2.val + 2

-- Define the condition for the sum being less than or equal to 4
def sumLEFour (d1 d2 : Die) : Prop := diceSum d1 d2 ≤ 4

-- Define the probability space
def totalOutcomes : Nat := 36

-- Define the number of favorable outcomes
def favorableOutcomes : Nat := 6

-- Theorem statement
theorem probability_sum_le_four :
  (favorableOutcomes : ℚ) / totalOutcomes = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_probability_sum_le_four_l1988_198883


namespace NUMINAMATH_CALUDE_root_difference_range_l1988_198843

noncomputable section

variables (a b c d : ℝ) (x₁ x₂ : ℝ)

def g (x : ℝ) := a * x^3 + b * x^2 + c * x + d

def f (x : ℝ) := 3 * a * x^2 + 2 * b * x + c

theorem root_difference_range (ha : a ≠ 0) 
  (h_sum : a + b + c = 0) 
  (h_prod : f 0 * f 1 > 0) 
  (h_roots : f x₁ = 0 ∧ f x₂ = 0) :
  ∃ (l u : ℝ), l = Real.sqrt 3 / 3 ∧ u = 2 / 3 ∧ 
  l ≤ |x₁ - x₂| ∧ |x₁ - x₂| < u :=
sorry

end NUMINAMATH_CALUDE_root_difference_range_l1988_198843


namespace NUMINAMATH_CALUDE_marys_unique_score_l1988_198802

/-- Represents the score in a mathematics competition. -/
structure Score where
  total : ℕ
  correct : ℕ
  wrong : ℕ
  h_total : total = 35 + 5 * correct - 2 * wrong

/-- Determines if a score is uniquely determinable. -/
def isUniqueDeterminable (s : Score) : Prop :=
  ∀ s' : Score, s'.total = s.total → s'.correct = s.correct ∧ s'.wrong = s.wrong

/-- The theorem stating Mary's unique score. -/
theorem marys_unique_score :
  ∃! s : Score,
    s.total > 90 ∧
    isUniqueDeterminable s ∧
    ∀ s' : Score, s'.total > 90 ∧ s'.total < s.total → ¬isUniqueDeterminable s' :=
by sorry

end NUMINAMATH_CALUDE_marys_unique_score_l1988_198802


namespace NUMINAMATH_CALUDE_unique_solution_l1988_198812

/-- The equation has two solutions and their sum is 12 -/
def has_two_solutions_sum_12 (a b : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ x ≠ 1 ∧ x ≠ -1 ∧ y ≠ 1 ∧ y ≠ -1 ∧
    (a * x^2 - 24 * x + b) / (x^2 - 1) = x ∧
    (a * y^2 - 24 * y + b) / (y^2 - 1) = y ∧
    x + y = 12

theorem unique_solution :
  ∀ a b : ℝ, has_two_solutions_sum_12 a b ↔ a = 35 ∧ b = -5819 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l1988_198812


namespace NUMINAMATH_CALUDE_triangle_segment_sum_squares_l1988_198884

-- Define the triangle ABC and points D and E
def Triangle (A B C : ℝ × ℝ) : Prop :=
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = a^2 ∧
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = b^2 ∧
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = c^2

def RightTriangle (A B C : ℝ × ℝ) : Prop :=
  Triangle A B C ∧ 
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

def DivideHypotenuse (A B C D E : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), t > 0 ∧
  D = ((2*B.1 + C.1)/3, (2*B.2 + C.2)/3) ∧
  E = ((B.1 + 2*C.1)/3, (B.2 + 2*C.2)/3)

-- State the theorem
theorem triangle_segment_sum_squares 
  (A B C D E : ℝ × ℝ) 
  (h1 : RightTriangle A B C) 
  (h2 : DivideHypotenuse A B C D E) : 
  ((D.1 - A.1)^2 + (D.2 - A.2)^2) + 
  ((E.1 - D.1)^2 + (E.2 - D.2)^2) + 
  ((E.1 - A.1)^2 + (E.2 - A.2)^2) = 
  2/3 * ((C.1 - B.1)^2 + (C.2 - B.2)^2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_segment_sum_squares_l1988_198884


namespace NUMINAMATH_CALUDE_min_value_sqrt_expression_l1988_198806

theorem min_value_sqrt_expression (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 16) :
  Real.sqrt (x + 36) + Real.sqrt (16 - x) + 2 * Real.sqrt x ≥ 13.46 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sqrt_expression_l1988_198806


namespace NUMINAMATH_CALUDE_wills_hourly_rate_l1988_198879

/-- Proof of Will's hourly rate given his work hours and total earnings -/
theorem wills_hourly_rate (monday_hours tuesday_hours total_earnings : ℕ) 
  (h1 : monday_hours = 8)
  (h2 : tuesday_hours = 2)
  (h3 : total_earnings = 80) :
  total_earnings / (monday_hours + tuesday_hours) = 8 := by
  sorry

#check wills_hourly_rate

end NUMINAMATH_CALUDE_wills_hourly_rate_l1988_198879


namespace NUMINAMATH_CALUDE_highway_vehicles_l1988_198836

/-- The number of vehicles involved in accidents per 100 million vehicles -/
def accident_rate : ℕ := 40

/-- The total number of vehicles involved in accidents last year -/
def total_accidents : ℕ := 800

/-- The number of vehicles per 100 million for the accident rate calculation -/
def base_vehicles : ℕ := 100000000

/-- The number of vehicles that traveled on the highway last year -/
def total_vehicles : ℕ := 2000000000

theorem highway_vehicles :
  total_vehicles = (total_accidents * base_vehicles) / accident_rate :=
sorry

end NUMINAMATH_CALUDE_highway_vehicles_l1988_198836


namespace NUMINAMATH_CALUDE_same_color_choices_l1988_198866

theorem same_color_choices (m : ℕ) : 
  let total_objects := 2 * m
  let red_objects := m
  let blue_objects := m
  (number_of_ways_to_choose_same_color : ℕ) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_same_color_choices_l1988_198866


namespace NUMINAMATH_CALUDE_intercepted_arc_is_60_degrees_l1988_198819

/-- An equilateral triangle with a circle rolling along its side -/
structure RollingCircleTriangle where
  -- The side length of the equilateral triangle
  side : ℝ
  -- The radius of the rolling circle
  radius : ℝ
  -- The radius equals the height of the triangle
  height_eq_radius : radius = (side * Real.sqrt 3) / 2

/-- The angular measure of the arc intercepted on the circle by the sides of the triangle -/
def intercepted_arc_measure (t : RollingCircleTriangle) : ℝ := 
  -- Definition to be proved
  60

/-- Theorem: The angular measure of the arc intercepted on the circle 
    by the sides of the triangle is always 60° -/
theorem intercepted_arc_is_60_degrees (t : RollingCircleTriangle) : 
  intercepted_arc_measure t = 60 := by
  sorry

end NUMINAMATH_CALUDE_intercepted_arc_is_60_degrees_l1988_198819


namespace NUMINAMATH_CALUDE_arrangement_count_l1988_198898

def num_people : ℕ := 8

theorem arrangement_count :
  (num_people.factorial) / 6 * 2 = 13440 := by sorry

end NUMINAMATH_CALUDE_arrangement_count_l1988_198898


namespace NUMINAMATH_CALUDE_number_problem_l1988_198816

theorem number_problem (x : ℝ) : 5 * (x - 12) = 40 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1988_198816


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l1988_198832

-- Define a triangle type
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : ℝ
  hb : ℝ
  hc : ℝ
  area : ℝ
  inradius : ℝ

-- Define the conditions
def validTriangle (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.ha > 0 ∧ t.hb > 0 ∧ t.hc > 0 ∧
  t.area > 0 ∧ t.inradius > 0

def integerAltitudes (t : Triangle) : Prop :=
  ∃ (n₁ n₂ n₃ : ℕ), t.ha = n₁ ∧ t.hb = n₂ ∧ t.hc = n₃

def altitudesSumLessThan20 (t : Triangle) : Prop :=
  t.ha + t.hb + t.hc < 20

def integerInradius (t : Triangle) : Prop :=
  ∃ (n : ℕ), t.inradius = n

-- State the theorem
theorem triangle_area_theorem (t : Triangle) 
  (h1 : validTriangle t)
  (h2 : integerAltitudes t)
  (h3 : altitudesSumLessThan20 t)
  (h4 : integerInradius t) :
  t.area = 6 ∨ t.area = 12 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_theorem_l1988_198832


namespace NUMINAMATH_CALUDE_square_area_on_parabola_prove_square_area_l1988_198867

theorem square_area_on_parabola : ℝ → Prop :=
  fun area =>
    ∃ (x₁ x₂ : ℝ),
      -- The endpoints lie on the parabola
      x₁^2 + 4*x₁ + 3 = 6 ∧
      x₂^2 + 4*x₂ + 3 = 6 ∧
      -- The side length is the distance between x-coordinates
      (x₂ - x₁)^2 = area ∧
      -- The area is 28
      area = 28

theorem prove_square_area : square_area_on_parabola 28 := by
  sorry

end NUMINAMATH_CALUDE_square_area_on_parabola_prove_square_area_l1988_198867


namespace NUMINAMATH_CALUDE_units_digit_of_expression_l1988_198847

-- Define a function to get the units digit of a natural number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define the expression we're interested in
def expression : ℕ := 24^3 + 42^3

-- Theorem statement
theorem units_digit_of_expression : unitsDigit expression = 6 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_expression_l1988_198847


namespace NUMINAMATH_CALUDE_base_10_to_base_7_157_l1988_198857

def base_7_digit (n : Nat) : Char :=
  if n < 7 then Char.ofNat (n + 48) else Char.ofNat (n + 55)

def to_base_7 (n : Nat) : List Char :=
  if n < 7 then [base_7_digit n]
  else base_7_digit (n % 7) :: to_base_7 (n / 7)

theorem base_10_to_base_7_157 :
  to_base_7 157 = ['3', '1', '3'] := by sorry

end NUMINAMATH_CALUDE_base_10_to_base_7_157_l1988_198857


namespace NUMINAMATH_CALUDE_rectangular_plot_area_l1988_198844

/-- Given a rectangular plot where the length is thrice the breadth and the breadth is 15 meters,
    prove that the area of the plot is 675 square meters. -/
theorem rectangular_plot_area (breadth : ℝ) (length : ℝ) (area : ℝ) : 
  breadth = 15 →
  length = 3 * breadth →
  area = length * breadth →
  area = 675 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_area_l1988_198844


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l1988_198813

/-- Given two quadratic equations with real roots and specific conditions, prove that m = 4 --/
theorem quadratic_roots_relation (m n : ℝ) (x₁ x₂ y₁ y₂ : ℝ) : 
  n < 0 →
  x₁^2 + m^2*x₁ + n = 0 →
  x₂^2 + m^2*x₂ + n = 0 →
  y₁^2 + 5*m*y₁ + 7 = 0 →
  y₂^2 + 5*m*y₂ + 7 = 0 →
  x₁ - y₁ = 2 →
  x₂ - y₂ = 2 →
  m = 4 := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l1988_198813


namespace NUMINAMATH_CALUDE_rachel_cookies_l1988_198807

theorem rachel_cookies (mona jasmine rachel : ℕ) : 
  mona = 20 →
  jasmine = mona - 5 →
  rachel > jasmine →
  mona + jasmine + rachel = 60 →
  rachel = 25 := by
sorry

end NUMINAMATH_CALUDE_rachel_cookies_l1988_198807


namespace NUMINAMATH_CALUDE_modular_inverse_13_mod_101_l1988_198853

theorem modular_inverse_13_mod_101 : ∃ x : ℕ, x ≤ 100 ∧ (13 * x) % 101 = 1 :=
by
  use 70
  sorry

end NUMINAMATH_CALUDE_modular_inverse_13_mod_101_l1988_198853


namespace NUMINAMATH_CALUDE_xy_positive_sufficient_not_necessary_l1988_198869

theorem xy_positive_sufficient_not_necessary (x y : ℝ) :
  (x * y > 0 → |x + y| = |x| + |y|) ∧
  ¬(∀ x y : ℝ, |x + y| = |x| + |y| → x * y > 0) :=
by sorry

end NUMINAMATH_CALUDE_xy_positive_sufficient_not_necessary_l1988_198869


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l1988_198858

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 1 + 3 * a 3 + a 15 = 10) :
  a 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l1988_198858


namespace NUMINAMATH_CALUDE_time_calculation_correct_l1988_198855

/-- Represents a date and time -/
structure DateTime where
  year : Nat
  month : Nat
  day : Nat
  hour : Nat
  minute : Nat

/-- Adds minutes to a DateTime -/
def addMinutes (dt : DateTime) (minutes : Nat) : DateTime :=
  sorry

/-- The start date and time -/
def startDateTime : DateTime :=
  { year := 2021, month := 1, day := 5, hour := 15, minute := 0 }

/-- The number of minutes to add -/
def minutesToAdd : Nat := 5050

/-- The expected end date and time -/
def expectedEndDateTime : DateTime :=
  { year := 2021, month := 1, day := 9, hour := 3, minute := 10 }

theorem time_calculation_correct :
  addMinutes startDateTime minutesToAdd = expectedEndDateTime := by sorry

end NUMINAMATH_CALUDE_time_calculation_correct_l1988_198855


namespace NUMINAMATH_CALUDE_system_solution_l1988_198870

/-- Given a system of equations, prove that x and y have specific values. -/
theorem system_solution (a x y : ℝ) (h1 : Real.log (x^2 + y^2) / Real.log (Real.sqrt 10) = 2 * Real.log (2*a) / Real.log 10 + 2 * Real.log (x^2 - y^2) / Real.log 100) (h2 : x * y = a^2) :
  (x = a * Real.sqrt (Real.sqrt 2 + 1) ∧ y = a * Real.sqrt (Real.sqrt 2 - 1)) ∨
  (x = -a * Real.sqrt (Real.sqrt 2 + 1) ∧ y = -a * Real.sqrt (Real.sqrt 2 - 1)) ∨
  (x = a * Real.sqrt (Real.sqrt 2 + 1) ∧ y = -a * Real.sqrt (Real.sqrt 2 - 1)) ∨
  (x = -a * Real.sqrt (Real.sqrt 2 + 1) ∧ y = a * Real.sqrt (Real.sqrt 2 - 1)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1988_198870


namespace NUMINAMATH_CALUDE_f_squared_properties_l1988_198818

-- Define a real-valued function f with period T
def f (T : ℝ) (h : T > 0) : ℝ → ℝ := sorry

-- Define the property of f being periodic with period T
def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

-- Define the property of f being monotonic on an interval
def is_monotonic_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

-- Main theorem
theorem f_squared_properties (T : ℝ) (h : T > 0) :
  (is_periodic (f T h) T) →
  (is_monotonic_on (f T h) 0 T) →
  (¬ ∃ P, is_periodic (fun x ↦ f T h (x^2)) P) ∧
  (is_monotonic_on (fun x ↦ f T h (x^2)) 0 (Real.sqrt T)) := by
  sorry

end NUMINAMATH_CALUDE_f_squared_properties_l1988_198818


namespace NUMINAMATH_CALUDE_tank_emptying_time_l1988_198871

-- Define constants
def tank_volume_cubic_feet : ℝ := 20
def inlet_rate : ℝ := 5
def outlet_rate_1 : ℝ := 9
def outlet_rate_2 : ℝ := 8
def inches_per_foot : ℝ := 12

-- Theorem statement
theorem tank_emptying_time :
  let tank_volume_cubic_inches := tank_volume_cubic_feet * (inches_per_foot ^ 3)
  let net_emptying_rate := outlet_rate_1 + outlet_rate_2 - inlet_rate
  tank_volume_cubic_inches / net_emptying_rate = 2880 := by
  sorry

end NUMINAMATH_CALUDE_tank_emptying_time_l1988_198871


namespace NUMINAMATH_CALUDE_inscribed_circle_ratio_l1988_198830

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a circle -/
structure Circle :=
  (center : Point)
  (radius : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Checks if a triangle is isosceles with base AB -/
def IsIsoscelesAB (t : Triangle) : Prop :=
  (t.A.x - t.C.x)^2 + (t.A.y - t.C.y)^2 = (t.B.x - t.C.x)^2 + (t.B.y - t.C.y)^2

/-- Checks if a circle is inscribed in a triangle -/
def IsInscribed (c : Circle) (t : Triangle) : Prop := sorry

/-- Checks if a point is on a line segment -/
def IsOnSegment (p : Point) (a : Point) (b : Point) : Prop := sorry

/-- Checks if a point is on a circle -/
def IsOnCircle (p : Point) (c : Circle) : Prop := sorry

/-- Calculates the distance between two points -/
def Distance (a : Point) (b : Point) : ℝ := sorry

/-- The main theorem -/
theorem inscribed_circle_ratio 
  (t : Triangle) 
  (c : Circle) 
  (M N : Point) 
  (k : ℝ) :
  IsIsoscelesAB t →
  IsInscribed c t →
  IsOnSegment M t.B t.C →
  IsOnCircle M c →
  IsOnSegment N t.A M →
  IsOnCircle N c →
  Distance t.A t.B / Distance t.B t.C = k →
  Distance M N / Distance t.A N = 2 * (2 - k) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_ratio_l1988_198830


namespace NUMINAMATH_CALUDE_open_box_volume_is_5208_l1988_198835

/-- Calculates the volume of an open box created from a metallic sheet --/
def open_box_volume (sheet_length : ℝ) (sheet_width : ℝ) (thickness : ℝ) (cut_size : ℝ) : ℝ :=
  let internal_length := sheet_length - 2 * cut_size - 2 * thickness
  let internal_width := sheet_width - 2 * cut_size - 2 * thickness
  let height := cut_size
  internal_length * internal_width * height

/-- Theorem stating that the volume of the open box is 5208 m³ --/
theorem open_box_volume_is_5208 :
  open_box_volume 48 38 0.5 8 = 5208 := by
  sorry

end NUMINAMATH_CALUDE_open_box_volume_is_5208_l1988_198835


namespace NUMINAMATH_CALUDE_line_properties_l1988_198880

-- Define the lines
def line (A B C : ℝ) := {(x, y) : ℝ × ℝ | A * x + B * y + C = 0}

-- Define when two lines intersect
def intersect (l1 l2 : Set (ℝ × ℝ)) := ∃ p, p ∈ l1 ∧ p ∈ l2

-- Define when two lines are perpendicular
def perpendicular (A1 B1 A2 B2 : ℝ) := A1 * A2 + B1 * B2 = 0

-- Theorem statement
theorem line_properties (A1 B1 C1 A2 B2 C2 : ℝ) :
  (A1 * B2 - A2 * B1 ≠ 0 → intersect (line A1 B1 C1) (line A2 B2 C2)) ∧
  (perpendicular A1 B1 A2 B2 → 
    ∃ (x1 y1 x2 y2 : ℝ), 
      (x1, y1) ∈ line A1 B1 C1 ∧ 
      (x2, y2) ∈ line A2 B2 C2 ∧ 
      (x2 - x1) * (y2 - y1) = 0) :=
sorry

end NUMINAMATH_CALUDE_line_properties_l1988_198880


namespace NUMINAMATH_CALUDE_license_plate_count_l1988_198845

/-- The set of odd single-digit numbers -/
def oddDigits : Finset Nat := {1, 3, 5, 7, 9}

/-- The set of prime numbers less than 10 -/
def primesLessThan10 : Finset Nat := {2, 3, 5, 7}

/-- The set of even single-digit numbers -/
def evenDigits : Finset Nat := {0, 2, 4, 6, 8}

/-- The number of letters in the alphabet -/
def alphabetSize : Nat := 26

theorem license_plate_count :
  (alphabetSize ^ 2) * oddDigits.card * primesLessThan10.card * evenDigits.card = 67600 := by
  sorry

#eval (alphabetSize ^ 2) * oddDigits.card * primesLessThan10.card * evenDigits.card

end NUMINAMATH_CALUDE_license_plate_count_l1988_198845


namespace NUMINAMATH_CALUDE_complex_sum_powers_of_i_l1988_198840

theorem complex_sum_powers_of_i : ∃ (i : ℂ), i^2 = -1 ∧ i + i^2 + i^3 + i^4 = 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_sum_powers_of_i_l1988_198840


namespace NUMINAMATH_CALUDE_inequality_proof_l1988_198854

theorem inequality_proof (a b c : ℝ) (h : a ≠ b) :
  Real.sqrt ((a - c)^2 + b^2) + Real.sqrt (a^2 + (b - c)^2) > Real.sqrt 2 * abs (a - b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1988_198854


namespace NUMINAMATH_CALUDE_sqrt_inequality_range_l1988_198831

theorem sqrt_inequality_range (m : ℝ) : 
  Real.sqrt (2 * m + 1) > Real.sqrt (m^2 + m - 1) ↔ 
  m ∈ Set.Ici ((Real.sqrt 5 - 1) / 2) ∩ Set.Iio 2 :=
sorry

end NUMINAMATH_CALUDE_sqrt_inequality_range_l1988_198831


namespace NUMINAMATH_CALUDE_student_calculation_mistake_l1988_198872

theorem student_calculation_mistake (x y z : ℝ) 
  (h1 : x - (y - z) = 15) 
  (h2 : x - y - z = 7) : 
  x - y = 11 := by
sorry

end NUMINAMATH_CALUDE_student_calculation_mistake_l1988_198872


namespace NUMINAMATH_CALUDE_probability_three_tails_l1988_198825

/-- The probability of getting exactly k successes in n trials
    with probability p for each trial -/
def binomialProbability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

/-- The number of coin flips -/
def numFlips : ℕ := 8

/-- The probability of getting tails -/
def probTails : ℚ := 2/3

/-- The number of tails we're interested in -/
def numTails : ℕ := 3

theorem probability_three_tails :
  binomialProbability numFlips numTails probTails = 448/6561 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_tails_l1988_198825


namespace NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l1988_198846

theorem arithmetic_expression_evaluation :
  10 - 9 * 8 + 7^2 - 6 / 3 * 2 + 1 = -16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l1988_198846


namespace NUMINAMATH_CALUDE_inequality_preservation_l1988_198863

theorem inequality_preservation (a b : ℝ) (h : a > b) : (1/3)*a - 1 > (1/3)*b - 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l1988_198863


namespace NUMINAMATH_CALUDE_jovana_shells_total_l1988_198805

/-- The total amount of shells in Jovana's bucket -/
def total_shells (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem stating that given 5 pounds initial and 12 pounds additional shells, the total is 17 pounds -/
theorem jovana_shells_total :
  total_shells 5 12 = 17 := by
  sorry

end NUMINAMATH_CALUDE_jovana_shells_total_l1988_198805


namespace NUMINAMATH_CALUDE_birthday_stickers_calculation_l1988_198833

/-- The number of stickers Mika received for her birthday -/
def birthday_stickers : ℝ := sorry

/-- Mika's initial number of stickers -/
def initial_stickers : ℝ := 20.0

/-- Number of stickers Mika bought -/
def bought_stickers : ℝ := 26.0

/-- Number of stickers Mika received from her sister -/
def sister_stickers : ℝ := 6.0

/-- Number of stickers Mika received from her mother -/
def mother_stickers : ℝ := 58.0

/-- Mika's final total number of stickers -/
def final_stickers : ℝ := 130.0

theorem birthday_stickers_calculation :
  birthday_stickers = final_stickers - (initial_stickers + bought_stickers + sister_stickers + mother_stickers) :=
by sorry

end NUMINAMATH_CALUDE_birthday_stickers_calculation_l1988_198833


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1988_198837

def I : Set Nat := {1,2,3,4,5,6,7,8}
def A : Set Nat := {3,4,5}
def B : Set Nat := {1,3,6}

theorem complement_intersection_theorem : 
  (I \ A) ∩ (I \ B) = {2,7,8} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1988_198837


namespace NUMINAMATH_CALUDE_group_size_is_ten_l1988_198850

/-- The number of people in a group that can hold a certain number of boxes. -/
def group_size (total_boxes : ℕ) (boxes_per_person : ℕ) : ℕ :=
  total_boxes / boxes_per_person

/-- Theorem: The group size is 10 when the total boxes is 20 and each person can hold 2 boxes. -/
theorem group_size_is_ten : group_size 20 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_group_size_is_ten_l1988_198850


namespace NUMINAMATH_CALUDE_pharmacy_loss_l1988_198888

theorem pharmacy_loss (a b : ℝ) (h : a < b) : 
  100 * ((a + b) / 2) - (41 * a + 59 * b) < 0 := by
  sorry

#check pharmacy_loss

end NUMINAMATH_CALUDE_pharmacy_loss_l1988_198888


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_remainder_l1988_198826

def arithmetic_sequence_sum (a : ℕ) (d : ℕ) (last : ℕ) : ℕ :=
  let n := (last - a) / d + 1
  n * (a + last) / 2

theorem arithmetic_sequence_sum_remainder
  (h1 : arithmetic_sequence_sum 3 6 273 % 6 = 0) : 
  arithmetic_sequence_sum 3 6 273 % 6 = 0 := by
  sorry

#check arithmetic_sequence_sum_remainder

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_remainder_l1988_198826


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l1988_198873

/-- Given a rhombus with diagonals of 10 inches and 24 inches, its perimeter is 52 inches. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) :
  let side := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  4 * side = 52 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l1988_198873


namespace NUMINAMATH_CALUDE_sin_half_range_l1988_198882

theorem sin_half_range (α : Real) (h1 : 0 < α ∧ α < π/2) (h2 : Real.sin (α/2) > Real.cos (α/2)) :
  ∃ x, Real.sqrt 2 / 2 < x ∧ x < 1 ∧ x = Real.sin (α/2) :=
sorry

end NUMINAMATH_CALUDE_sin_half_range_l1988_198882


namespace NUMINAMATH_CALUDE_john_total_paint_l1988_198865

/-- The number of primary colors John has -/
def num_colors : ℕ := 3

/-- The amount of paint John has for each color (in liters) -/
def paint_per_color : ℕ := 5

/-- The total amount of paint John has (in liters) -/
def total_paint : ℕ := num_colors * paint_per_color

theorem john_total_paint : total_paint = 15 := by
  sorry

end NUMINAMATH_CALUDE_john_total_paint_l1988_198865


namespace NUMINAMATH_CALUDE_largest_quantity_l1988_198894

theorem largest_quantity (A B C : ℚ) : 
  A = 3003 / 3002 + 3003 / 3004 →
  B = 2 / 1 + 4 / 2 + 3005 / 3004 →
  C = 3004 / 3003 + 3004 / 3005 →
  B > A ∧ B > C := by
sorry


end NUMINAMATH_CALUDE_largest_quantity_l1988_198894


namespace NUMINAMATH_CALUDE_lawnmower_value_drop_l1988_198868

theorem lawnmower_value_drop (initial_price : ℝ) (first_drop_percent : ℝ) (final_value : ℝ) :
  initial_price = 100 →
  first_drop_percent = 25 →
  final_value = 60 →
  let value_after_six_months := initial_price * (1 - first_drop_percent / 100)
  let drop_over_next_year := value_after_six_months - final_value
  let drop_percent_next_year := (drop_over_next_year / value_after_six_months) * 100
  drop_percent_next_year = 20 := by
  sorry

end NUMINAMATH_CALUDE_lawnmower_value_drop_l1988_198868


namespace NUMINAMATH_CALUDE_island_distance_l1988_198827

/-- Given two islands A and B that are 10 nautical miles apart, with an angle of view from A to C and B of 60°, and an angle of view from B to A and C of 75°, the distance between islands B and C is 5√6 nautical miles. -/
theorem island_distance (A B C : ℝ × ℝ) : 
  let distance := λ p q : ℝ × ℝ => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  let angle := λ p q r : ℝ × ℝ => Real.arccos ((distance p q)^2 + (distance p r)^2 - (distance q r)^2) / (2 * distance p q * distance p r)
  distance A B = 10 →
  angle A C B = π / 3 →
  angle B A C = 5 * π / 12 →
  distance B C = 5 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_island_distance_l1988_198827


namespace NUMINAMATH_CALUDE_custom_mult_theorem_l1988_198864

/-- Custom multiplication operation for integers -/
def customMult (a b : ℤ) : ℤ := (a - 1) * (b - 1)

/-- Theorem stating that if y10 = 90 under the custom multiplication, then y = 11 -/
theorem custom_mult_theorem (y : ℤ) (h : customMult y 10 = 90) : y = 11 := by
  sorry

end NUMINAMATH_CALUDE_custom_mult_theorem_l1988_198864


namespace NUMINAMATH_CALUDE_system_solutions_l1988_198899

-- Define the logarithm base 4
noncomputable def log4 (x : ℝ) : ℝ := Real.log x / Real.log 4

-- Define the system of equations
def system (x y : ℝ) : Prop :=
  x + y - 20 = 0 ∧ log4 x + log4 y = 1 + log4 9

-- Theorem stating the solutions
theorem system_solutions :
  ∃ (x y : ℝ), system x y ∧ ((x = 18 ∧ y = 2) ∨ (x = 2 ∧ y = 18)) :=
sorry

end NUMINAMATH_CALUDE_system_solutions_l1988_198899


namespace NUMINAMATH_CALUDE_system_solution_l1988_198859

theorem system_solution (a : ℝ) :
  ∃ (x y z : ℝ),
    x^2 + y^2 - 2*z^2 = 2*a^2 ∧
    x + y + 2*z = 4*(a^2 + 1) ∧
    z^2 - x*y = a^2 ∧
    ((x = a^2 + a + 1 ∧ y = a^2 - a + 1) ∨ (x = a^2 - a + 1 ∧ y = a^2 + a + 1)) ∧
    z = a^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1988_198859


namespace NUMINAMATH_CALUDE_farm_cows_count_l1988_198838

/-- Represents the number of bags of husk eaten by a group of cows in 30 days -/
def total_bags : ℕ := 30

/-- Represents the number of bags of husk eaten by one cow in 30 days -/
def bags_per_cow : ℕ := 1

/-- Calculates the number of cows on the farm -/
def num_cows : ℕ := total_bags / bags_per_cow

/-- Proves that the number of cows on the farm is 30 -/
theorem farm_cows_count : num_cows = 30 := by
  sorry

end NUMINAMATH_CALUDE_farm_cows_count_l1988_198838


namespace NUMINAMATH_CALUDE_equation_solution_l1988_198851

theorem equation_solution : ∀ x : ℝ, (3 / (x - 3) = 3 / (x^2 - 9)) ↔ x = -2 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1988_198851


namespace NUMINAMATH_CALUDE_f_is_quadratic_l1988_198810

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The given equation -/
def f (x : ℝ) : ℝ := 3 - 5*x^2 - x

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end NUMINAMATH_CALUDE_f_is_quadratic_l1988_198810


namespace NUMINAMATH_CALUDE_words_with_consonants_l1988_198887

/-- The number of letters in the alphabet --/
def alphabet_size : ℕ := 6

/-- The number of vowels in the alphabet --/
def vowel_count : ℕ := 2

/-- The length of words we're considering --/
def word_length : ℕ := 5

/-- The total number of possible words --/
def total_words : ℕ := alphabet_size ^ word_length

/-- The number of words containing only vowels --/
def vowel_only_words : ℕ := vowel_count ^ word_length

theorem words_with_consonants :
  total_words - vowel_only_words = 7744 := by sorry

end NUMINAMATH_CALUDE_words_with_consonants_l1988_198887


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l1988_198804

/-- 
Given an arithmetic sequence with:
- First term a = -36
- Common difference d = 6
- Last term l = 66

Prove that the number of terms in the sequence is 18.
-/
theorem arithmetic_sequence_length :
  ∀ (a d l : ℤ) (n : ℕ),
    a = -36 →
    d = 6 →
    l = 66 →
    l = a + (n - 1) * d →
    n = 18 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l1988_198804


namespace NUMINAMATH_CALUDE_secret_spread_day_l1988_198815

/-- The number of people who know the secret after n days -/
def secret_spread (n : ℕ) : ℕ := (3^(n+1) - 1) / 2

/-- The day when 3280 students know the secret -/
theorem secret_spread_day :
  ∃ n : ℕ, secret_spread n = 3280 ∧ n = 7 :=
sorry

end NUMINAMATH_CALUDE_secret_spread_day_l1988_198815


namespace NUMINAMATH_CALUDE_negative_sixty_four_to_four_thirds_l1988_198800

theorem negative_sixty_four_to_four_thirds : (-64 : ℝ) ^ (4/3) = 256 := by
  sorry

end NUMINAMATH_CALUDE_negative_sixty_four_to_four_thirds_l1988_198800


namespace NUMINAMATH_CALUDE_parity_of_F_l1988_198861

/-- F(n) is the number of ways to express n as the sum of three different positive integers -/
def F (n : ℕ) : ℕ := sorry

/-- Main theorem about the parity of F(n) -/
theorem parity_of_F (n : ℕ) (hn : n > 0) :
  (n % 6 = 2 ∨ n % 6 = 4 → F n % 2 = 0) ∧
  (n % 6 = 0 → F n % 2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_parity_of_F_l1988_198861


namespace NUMINAMATH_CALUDE_correct_matching_probability_l1988_198896

-- Define the number of celebrities and child photos
def num_celebrities : ℕ := 4

-- Define the function to calculate the number of possible arrangements
def num_arrangements (n : ℕ) : ℕ := Nat.factorial n

-- Define the probability of correct matching
def probability_correct_matching : ℚ := 1 / num_arrangements num_celebrities

-- Theorem statement
theorem correct_matching_probability :
  probability_correct_matching = 1 / 24 := by
  sorry

end NUMINAMATH_CALUDE_correct_matching_probability_l1988_198896


namespace NUMINAMATH_CALUDE_rectangle_area_42_implies_y_7_l1988_198886

/-- Rectangle PQRS with vertices P(0, 0), Q(0, 6), R(y, 6), and S(y, 0) -/
structure Rectangle where
  y : ℝ
  h_positive : y > 0

/-- The area of a rectangle is the product of its length and width -/
def area (rect : Rectangle) : ℝ := 6 * rect.y

theorem rectangle_area_42_implies_y_7 (rect : Rectangle) (h_area : area rect = 42) : rect.y = 7 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_42_implies_y_7_l1988_198886


namespace NUMINAMATH_CALUDE_workers_wage_increase_l1988_198811

/-- Proves that if a worker's daily wage is increased by 50% to $42, then the original daily wage was $28. -/
theorem workers_wage_increase (original_wage : ℝ) (increased_wage : ℝ) : 
  increased_wage = 42 ∧ increased_wage = original_wage * 1.5 → original_wage = 28 := by
  sorry

end NUMINAMATH_CALUDE_workers_wage_increase_l1988_198811


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1988_198828

def M : Set ℝ := {x | -1 < x ∧ x < 1}
def N : Set ℝ := {x | ∃ y, y = Real.sqrt x}

theorem intersection_of_M_and_N : M ∩ N = {x | 0 ≤ x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1988_198828


namespace NUMINAMATH_CALUDE_reflection_line_correct_l1988_198852

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The reflection line that transforms one point to another -/
def reflection_line (p1 p2 : Point) : Line :=
  sorry

/-- Check if a point lies on a line -/
def point_on_line (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- The given points in the problem -/
def point1 : Point := ⟨5, 3⟩
def point2 : Point := ⟨1, -1⟩

/-- The proposed reflection line -/
def line_l : Line := ⟨-1, 4⟩

theorem reflection_line_correct :
  reflection_line point1 point2 = line_l ∧
  point_on_line (⟨3, 1⟩ : Point) line_l :=
sorry

end NUMINAMATH_CALUDE_reflection_line_correct_l1988_198852


namespace NUMINAMATH_CALUDE_quadratic_root_implies_b_value_l1988_198824

theorem quadratic_root_implies_b_value (b : ℝ) : 
  (Complex.I + 3 : ℂ) ^ 2 - 6 * (Complex.I + 3 : ℂ) + b = 0 → b = 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_b_value_l1988_198824


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l1988_198885

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | ∃ y, y = Real.log (1 - x^2)}
def B : Set ℝ := {y : ℝ | ∃ x, y = (4 : ℝ)^(x - 2)}

-- State the theorem
theorem intersection_complement_equality :
  A ∩ (Set.univ \ B) = Set.Ioc (-1) 0 := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l1988_198885


namespace NUMINAMATH_CALUDE_angle_value_l1988_198874

theorem angle_value (PQR PQS QRS : ℝ) (x : ℝ) : 
  PQR = 120 → PQS = 2*x → QRS = x → PQR = PQS + QRS → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_angle_value_l1988_198874


namespace NUMINAMATH_CALUDE_dinner_arrangements_l1988_198808

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- There are 5 people in the group -/
def total_people : ℕ := 5

/-- The number of people who cook -/
def cooks : ℕ := 2

theorem dinner_arrangements :
  choose total_people cooks = 10 := by
  sorry

end NUMINAMATH_CALUDE_dinner_arrangements_l1988_198808


namespace NUMINAMATH_CALUDE_intersect_point_m_bisecting_line_equation_l1988_198881

-- Define the lines
def l₁ (x y : ℝ) : Prop := 2 * x + y - 1 = 0
def l₂ (x y : ℝ) : Prop := x - y + 2 = 0
def l₃ (m x y : ℝ) : Prop := 3 * x + m * y - 6 = 0

-- Define point M
def M : ℝ × ℝ := (2, 0)

-- Theorem for part (1)
theorem intersect_point_m : 
  ∃ (x y : ℝ), l₁ x y ∧ l₂ x y ∧ (∃ (m : ℝ), l₃ m x y) → 
  (∃ (x y : ℝ), l₁ x y ∧ l₂ x y ∧ l₃ (21/5) x y) :=
sorry

-- Theorem for part (2)
theorem bisecting_line_equation :
  ∃ (A B : ℝ × ℝ) (k : ℝ), 
    l₁ A.1 A.2 ∧ l₂ B.1 B.2 ∧ 
    ((A.1 + B.1) / 2 = M.1 ∧ (A.2 + B.2) / 2 = M.2) →
    (∀ (x y : ℝ), 11 * x + y - 22 = 0 ↔ 
      ∃ (t : ℝ), x = A.1 + t * (M.1 - A.1) ∧ y = A.2 + t * (M.2 - A.2)) :=
sorry

end NUMINAMATH_CALUDE_intersect_point_m_bisecting_line_equation_l1988_198881


namespace NUMINAMATH_CALUDE_complex_magnitude_squared_l1988_198897

theorem complex_magnitude_squared (w : ℂ) (h : w^2 = -48 + 14*I) : 
  Complex.abs w = 5 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_squared_l1988_198897


namespace NUMINAMATH_CALUDE_paintbrush_cost_l1988_198860

/-- The amount spent on paintbrushes given the total spent and costs of other items. -/
theorem paintbrush_cost (total_spent : ℝ) (canvas_cost : ℝ) (paint_cost : ℝ) (easel_cost : ℝ) 
  (h1 : total_spent = 90)
  (h2 : canvas_cost = 40)
  (h3 : paint_cost = canvas_cost / 2)
  (h4 : easel_cost = 15) :
  total_spent - (canvas_cost + paint_cost + easel_cost) = 15 :=
by sorry

end NUMINAMATH_CALUDE_paintbrush_cost_l1988_198860


namespace NUMINAMATH_CALUDE_car_distance_theorem_l1988_198849

/-- The distance covered by a car given initial time and adjusted speed -/
theorem car_distance_theorem (initial_time : ℝ) (adjusted_speed : ℝ) : 
  initial_time = 6 →
  adjusted_speed = 60 →
  (initial_time * 3 / 2) * adjusted_speed = 540 := by
sorry

end NUMINAMATH_CALUDE_car_distance_theorem_l1988_198849


namespace NUMINAMATH_CALUDE_symmetry_properties_l1988_198801

-- Define the shapes
inductive Shape
  | Parallelogram
  | Rectangle
  | Square
  | Rhombus
  | IsoscelesTrapezoid

-- Define the symmetry properties
def isAxisymmetric (s : Shape) : Prop :=
  match s with
  | Shape.Rectangle => true
  | Shape.Square => true
  | Shape.Rhombus => true
  | Shape.IsoscelesTrapezoid => true
  | _ => false

def isCentrallySymmetric (s : Shape) : Prop :=
  match s with
  | Shape.Parallelogram => true
  | Shape.Rectangle => true
  | Shape.Square => true
  | Shape.Rhombus => true
  | _ => false

-- Theorem statement
theorem symmetry_properties :
  ∀ s : Shape,
    (isAxisymmetric s ∧ isCentrallySymmetric s) ↔
    (s = Shape.Rectangle ∨ s = Shape.Square ∨ s = Shape.Rhombus) :=
by sorry

end NUMINAMATH_CALUDE_symmetry_properties_l1988_198801


namespace NUMINAMATH_CALUDE_reflection_across_y_axis_l1988_198876

/-- Given a point A with coordinates (-4, 8, 6), 
    prove that its reflection across the y-axis has coordinates (4, 8, 6) -/
theorem reflection_across_y_axis :
  let A : ℝ × ℝ × ℝ := (-4, 8, 6)
  let reflection : ℝ × ℝ × ℝ → ℝ × ℝ × ℝ := fun (x, y, z) ↦ (-x, y, z)
  reflection A = (4, 8, 6) := by
  sorry

end NUMINAMATH_CALUDE_reflection_across_y_axis_l1988_198876


namespace NUMINAMATH_CALUDE_power_product_square_l1988_198890

theorem power_product_square (a b : ℝ) : (a^2 * b)^2 = a^4 * b^2 := by sorry

end NUMINAMATH_CALUDE_power_product_square_l1988_198890


namespace NUMINAMATH_CALUDE_resulting_surface_area_l1988_198889

/-- Represents the dimensions of the large cube -/
def large_cube_dim : ℕ := 12

/-- Represents the dimensions of the small cubes -/
def small_cube_dim : ℕ := 3

/-- The number of small cubes in the original large cube -/
def total_small_cubes : ℕ := 64

/-- The number of small cubes removed -/
def removed_cubes : ℕ := 8

/-- The number of remaining small cubes after removal -/
def remaining_cubes : ℕ := total_small_cubes - removed_cubes

/-- The surface area of a single small cube before modification -/
def small_cube_surface : ℕ := 6 * small_cube_dim ^ 2

/-- The number of new surfaces exposed per small cube after modification -/
def new_surfaces_per_cube : ℕ := 12

/-- The number of edge-shared internal faces -/
def edge_shared_faces : ℕ := 12

/-- The area of each edge-shared face -/
def edge_shared_face_area : ℕ := small_cube_dim ^ 2

/-- Theorem stating the surface area of the resulting structure -/
theorem resulting_surface_area :
  (remaining_cubes * (small_cube_surface + new_surfaces_per_cube)) -
  (4 * 3 * edge_shared_faces * edge_shared_face_area) = 3408 := by
  sorry

end NUMINAMATH_CALUDE_resulting_surface_area_l1988_198889


namespace NUMINAMATH_CALUDE_smallest_natural_solution_l1988_198848

theorem smallest_natural_solution (n : ℕ) : 
  (2023 / 2022 : ℝ) ^ (36 * (1 - (2/3 : ℝ)^(n+1)) / (1 - 2/3)) > (2023 / 2022 : ℝ) ^ 96 ↔ n ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_natural_solution_l1988_198848


namespace NUMINAMATH_CALUDE_unknown_number_proof_l1988_198814

theorem unknown_number_proof : (12^1 * 6^4) / 432 = 36 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_proof_l1988_198814


namespace NUMINAMATH_CALUDE_cat_food_consumed_by_wednesday_l1988_198856

/-- Represents the days of the week -/
inductive Day : Type
| Monday : Day
| Tuesday : Day
| Wednesday : Day
| Thursday : Day
| Friday : Day
| Saturday : Day
| Sunday : Day

/-- Calculates the number of days until all cat food is consumed -/
def daysUntilFoodConsumed (morningPortion : ℚ) (eveningPortion : ℚ) (fullCans : ℕ) (leftoverCan : ℚ) (leftoverExpiry : Day) : Day :=
  sorry

/-- Theorem stating that all cat food will be consumed by Wednesday -/
theorem cat_food_consumed_by_wednesday :
  let morningPortion : ℚ := 1/4
  let eveningPortion : ℚ := 1/6
  let fullCans : ℕ := 10
  let leftoverCan : ℚ := 1/2
  let leftoverExpiry : Day := Day.Tuesday
  daysUntilFoodConsumed morningPortion eveningPortion fullCans leftoverCan leftoverExpiry = Day.Wednesday :=
by sorry

end NUMINAMATH_CALUDE_cat_food_consumed_by_wednesday_l1988_198856


namespace NUMINAMATH_CALUDE_invalid_external_diagonals_l1988_198878

def is_valid_external_diagonals (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  (a^2 + b^2 ≥ c^2 ∧ a^2 + c^2 ≥ b^2 ∧ b^2 + c^2 ≥ a^2)

theorem invalid_external_diagonals : 
  ¬(is_valid_external_diagonals 8 15 18) := by
  sorry

end NUMINAMATH_CALUDE_invalid_external_diagonals_l1988_198878


namespace NUMINAMATH_CALUDE_one_zero_one_zero_in_interval_l1988_198823

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 1

-- Theorem for the case of only one zero
theorem one_zero (a : ℝ) :
  (∃! x, f a x = 0) ↔ (a = 2 ∨ a = -2) :=
sorry

-- Theorem for the case of only one zero in (0, 1)
theorem one_zero_in_interval (a : ℝ) :
  (∃! x, x ∈ (Set.Ioo 0 1) ∧ f a x = 0) ↔ a > 2 :=
sorry

end NUMINAMATH_CALUDE_one_zero_one_zero_in_interval_l1988_198823


namespace NUMINAMATH_CALUDE_cost_of_shirt_l1988_198822

/-- Given Sandy's shopping trip details, prove the cost of the shirt. -/
theorem cost_of_shirt (pants_cost change : ℚ) (bill : ℚ) : 
  pants_cost = 9.24 →
  change = 2.51 →
  bill = 20 →
  bill - change - pants_cost = 8.25 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_shirt_l1988_198822


namespace NUMINAMATH_CALUDE_gcd_72_108_150_l1988_198841

theorem gcd_72_108_150 : Nat.gcd 72 (Nat.gcd 108 150) = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_72_108_150_l1988_198841


namespace NUMINAMATH_CALUDE_smallest_linear_combination_3003_55555_l1988_198803

theorem smallest_linear_combination_3003_55555 :
  ∃ (k : ℕ), k > 0 ∧ (∃ (m n : ℤ), k = 3003 * m + 55555 * n) ∧
  ∀ (j : ℕ), j > 0 → (∃ (a b : ℤ), j = 3003 * a + 55555 * b) → j ≥ k :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_linear_combination_3003_55555_l1988_198803


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l1988_198862

theorem quadratic_roots_property (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 - 4*x₁ + k + 1 = 0 ∧ 
    x₂^2 - 4*x₂ + k + 1 = 0 ∧
    3/x₁ + 3/x₂ = x₁*x₂ - 4) →
  k = -3 ∧ k ≤ 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l1988_198862


namespace NUMINAMATH_CALUDE_bookshelf_average_l1988_198820

theorem bookshelf_average (initial_books : ℕ) (new_books : ℕ) (shelves : ℕ) (leftover : ℕ) 
  (h1 : initial_books = 56)
  (h2 : new_books = 26)
  (h3 : shelves = 4)
  (h4 : leftover = 2) :
  (initial_books + new_books - leftover) / shelves = 20 := by
  sorry

end NUMINAMATH_CALUDE_bookshelf_average_l1988_198820


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_related_quadratic_inequality_solution_l1988_198809

/-- Solution set type -/
inductive SolutionSet
  | Empty
  | Interval (lower upper : ℝ)
  | Union (s1 s2 : SolutionSet)

/-- Solve quadratic inequality -/
noncomputable def solveQuadraticInequality (a : ℝ) : SolutionSet :=
  if a = 0 then
    SolutionSet.Empty
  else if a > 0 then
    SolutionSet.Interval (-a) (2 * a)
  else
    SolutionSet.Interval (2 * a) (-a)

/-- Theorem for part 1 -/
theorem quadratic_inequality_solution (a : ℝ) :
  solveQuadraticInequality a =
    if a = 0 then
      SolutionSet.Empty
    else if a > 0 then
      SolutionSet.Interval (-a) (2 * a)
    else
      SolutionSet.Interval (2 * a) (-a) :=
by sorry

/-- Theorem for part 2 -/
theorem related_quadratic_inequality_solution :
  ∃ (a b : ℝ), 
    (∀ x, x^2 - a*x - b < 0 ↔ -1 < x ∧ x < 2) →
    (∀ x, a*x^2 + x - b > 0 ↔ x < -2 ∨ x > 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_related_quadratic_inequality_solution_l1988_198809


namespace NUMINAMATH_CALUDE_train_distance_l1988_198842

theorem train_distance (x : ℝ) :
  (x > 0) →
  (x / 40 + 2*x / 20 = (x + 2*x) / 48) →
  (x + 2*x = 6) :=
by sorry

end NUMINAMATH_CALUDE_train_distance_l1988_198842


namespace NUMINAMATH_CALUDE_sum_first_ten_terms_l1988_198891

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

def sum_arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem sum_first_ten_terms :
  sum_arithmetic_sequence (-3) 4 10 = 150 := by
sorry

end NUMINAMATH_CALUDE_sum_first_ten_terms_l1988_198891


namespace NUMINAMATH_CALUDE_binary_101101_to_decimal_l1988_198877

def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (λ acc (i, b) => acc + if b then 2^i else 0) 0

def binary_101101 : List Bool := [true, false, true, true, false, true]

theorem binary_101101_to_decimal :
  binary_to_decimal binary_101101 = 45 := by
  sorry

end NUMINAMATH_CALUDE_binary_101101_to_decimal_l1988_198877


namespace NUMINAMATH_CALUDE_dragon_boat_festival_visitors_scientific_notation_l1988_198892

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem dragon_boat_festival_visitors_scientific_notation :
  toScientificNotation 82600000 = ScientificNotation.mk 8.26 7 sorry := by
  sorry

end NUMINAMATH_CALUDE_dragon_boat_festival_visitors_scientific_notation_l1988_198892


namespace NUMINAMATH_CALUDE_solve_bowtie_equation_l1988_198893

-- Define the operation ⊗
noncomputable def bowtie (a b : ℝ) : ℝ := a + Real.sqrt (b + Real.sqrt (b + Real.sqrt (b + Real.sqrt b)))

-- Theorem statement
theorem solve_bowtie_equation (y : ℝ) : bowtie 5 y = 15 → y = 90 := by
  sorry

end NUMINAMATH_CALUDE_solve_bowtie_equation_l1988_198893


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1988_198817

theorem least_subtraction_for_divisibility : ∃ (n : ℕ), n = 8 ∧ 
  (∀ (m : ℕ), m < n → ¬(20 ∣ (50248 - m))) ∧ (20 ∣ (50248 - n)) := by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1988_198817


namespace NUMINAMATH_CALUDE_limit_fraction_to_one_third_l1988_198834

theorem limit_fraction_to_one_third :
  ∀ ε > 0, ∃ N : ℝ, ∀ n : ℝ, n > N → |((n + 20) / (3 * n + 1)) - (1 / 3)| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_limit_fraction_to_one_third_l1988_198834


namespace NUMINAMATH_CALUDE_prob_not_adjacent_ten_chairs_l1988_198839

-- Define the number of chairs
def n : ℕ := 10

-- Define the probability function
def prob_not_adjacent (n : ℕ) : ℚ :=
  1 - (n - 1 : ℚ) / (n.choose 2 : ℚ)

-- Theorem statement
theorem prob_not_adjacent_ten_chairs :
  prob_not_adjacent n = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_prob_not_adjacent_ten_chairs_l1988_198839


namespace NUMINAMATH_CALUDE_ball_fall_height_l1988_198895

/-- Given a ball falling from a certain height, this theorem calculates its final height from the ground. -/
theorem ball_fall_height (initial_height : ℝ) (fall_time : ℝ) (fall_speed : ℝ) :
  initial_height = 120 →
  fall_time = 20 →
  fall_speed = 4 →
  initial_height - fall_time * fall_speed = 40 := by
sorry

end NUMINAMATH_CALUDE_ball_fall_height_l1988_198895


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l1988_198829

/-- Two points are symmetric with respect to the x-axis if their x-coordinates are equal
    and their y-coordinates are negatives of each other -/
def symmetric_x_axis (A B : ℝ × ℝ) : Prop :=
  A.1 = B.1 ∧ A.2 = -B.2

/-- Given that point A(2, a) is symmetric to point B(b, -3) with respect to the x-axis,
    prove that a + b = 5 -/
theorem symmetric_points_sum (a b : ℝ) 
  (h : symmetric_x_axis (2, a) (b, -3)) : a + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l1988_198829
