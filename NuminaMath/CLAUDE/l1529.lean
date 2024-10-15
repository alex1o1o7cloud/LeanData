import Mathlib

namespace NUMINAMATH_CALUDE_relation_abc_l1529_152962

theorem relation_abc : 
  let a := (2 : ℝ) ^ (1/5 : ℝ)
  let b := (2/5 : ℝ) ^ (1/5 : ℝ)
  let c := (2/5 : ℝ) ^ (3/5 : ℝ)
  a > b ∧ b > c := by sorry

end NUMINAMATH_CALUDE_relation_abc_l1529_152962


namespace NUMINAMATH_CALUDE_granary_circumference_l1529_152906

/-- Represents the height of the granary in chi -/
def granary_height : ℝ := 13.325

/-- Represents the volume of the granary in cubic chi -/
def granary_volume : ℝ := 2000 * 1.62

/-- Approximation of π -/
def π_approx : ℝ := 3

theorem granary_circumference :
  let base_area := granary_volume / granary_height
  let radius := Real.sqrt (base_area / π_approx)
  2 * π_approx * radius = 54 := by sorry

end NUMINAMATH_CALUDE_granary_circumference_l1529_152906


namespace NUMINAMATH_CALUDE_quadrilateral_is_trapezoid_or_parallelogram_l1529_152924

/-- A quadrilateral with angles A, B, C, and D, where the products of cosines of opposite angles are equal. -/
structure Quadrilateral where
  A : Real
  B : Real
  C : Real
  D : Real
  angle_sum : A + B + C + D = 2 * Real.pi
  cosine_product : Real.cos A * Real.cos C = Real.cos B * Real.cos D

/-- A quadrilateral is either a trapezoid or a parallelogram if the products of cosines of opposite angles are equal. -/
theorem quadrilateral_is_trapezoid_or_parallelogram (q : Quadrilateral) :
  (∃ (x y : Real), x + y = Real.pi ∧ (q.A = x ∧ q.C = x) ∨ (q.B = y ∧ q.D = y)) ∨
  (q.A = q.C ∧ q.B = q.D) := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_is_trapezoid_or_parallelogram_l1529_152924


namespace NUMINAMATH_CALUDE_abs_2x_plus_4_not_positive_l1529_152941

theorem abs_2x_plus_4_not_positive (x : ℝ) : |2*x + 4| ≤ 0 ↔ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_abs_2x_plus_4_not_positive_l1529_152941


namespace NUMINAMATH_CALUDE_distinct_arrangements_l1529_152982

/-- The number of distinct arrangements of 6 indistinguishable objects of one type
    and 4 indistinguishable objects of another type in a row of 10 positions -/
def arrangement_count : ℕ := Nat.choose 10 4

/-- Theorem stating that the number of distinct arrangements is 210 -/
theorem distinct_arrangements :
  arrangement_count = 210 := by sorry

end NUMINAMATH_CALUDE_distinct_arrangements_l1529_152982


namespace NUMINAMATH_CALUDE_class_composition_solution_l1529_152987

/-- Represents the class composition problem --/
structure ClassComposition where
  total_students : ℕ
  girls : ℕ
  boys : ℕ

/-- Checks if the given class composition satisfies the problem conditions --/
def satisfies_conditions (c : ClassComposition) : Prop :=
  c.total_students = c.girls + c.boys ∧
  c.girls * 2 = c.boys * 3 ∧
  (c.total_students * 2 - 150 = c.girls * 5)

/-- The theorem stating the solution to the class composition problem --/
theorem class_composition_solution :
  ∃ c : ClassComposition, c.total_students = 300 ∧ c.girls = 180 ∧ c.boys = 120 ∧
  satisfies_conditions c := by
  sorry

#check class_composition_solution

end NUMINAMATH_CALUDE_class_composition_solution_l1529_152987


namespace NUMINAMATH_CALUDE_equation_solution_l1529_152907

theorem equation_solution (x y : ℕ) : x^y + y^x = 2408 ∧ x = 2407 → y = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1529_152907


namespace NUMINAMATH_CALUDE_percentage_men_is_seventy_l1529_152944

/-- The number of women in the engineering department -/
def num_women : ℕ := 180

/-- The number of men in the engineering department -/
def num_men : ℕ := 420

/-- The total number of students in the engineering department -/
def total_students : ℕ := num_women + num_men

/-- The percentage of men in the engineering department -/
def percentage_men : ℚ := (num_men : ℚ) / (total_students : ℚ) * 100

theorem percentage_men_is_seventy :
  percentage_men = 70 :=
sorry

end NUMINAMATH_CALUDE_percentage_men_is_seventy_l1529_152944


namespace NUMINAMATH_CALUDE_officer_average_salary_l1529_152902

theorem officer_average_salary
  (total_avg : ℝ)
  (non_officer_avg : ℝ)
  (officer_count : ℕ)
  (non_officer_count : ℕ)
  (h1 : total_avg = 120)
  (h2 : non_officer_avg = 110)
  (h3 : officer_count = 15)
  (h4 : non_officer_count = 525) :
  let total_count := officer_count + non_officer_count
  let officer_total := total_avg * total_count - non_officer_avg * non_officer_count
  officer_total / officer_count = 470 := by
sorry

end NUMINAMATH_CALUDE_officer_average_salary_l1529_152902


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_4_and_16_l1529_152947

theorem arithmetic_mean_of_4_and_16 (m : ℝ) : 
  m = (4 + 16) / 2 → m = 10 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_4_and_16_l1529_152947


namespace NUMINAMATH_CALUDE_ratio_sum_quotient_l1529_152928

theorem ratio_sum_quotient (x y : ℚ) (h : x / y = 1 / 2) : (x + y) / y = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_quotient_l1529_152928


namespace NUMINAMATH_CALUDE_polynomial_division_degree_l1529_152920

theorem polynomial_division_degree (f d q r : Polynomial ℝ) : 
  Polynomial.degree f = 15 →
  f = d * q + r →
  Polynomial.degree q = 8 →
  r = 5 * X^2 + 3 * X - 9 →
  Polynomial.degree d = 7 := by
sorry

end NUMINAMATH_CALUDE_polynomial_division_degree_l1529_152920


namespace NUMINAMATH_CALUDE_power_zero_equations_l1529_152966

theorem power_zero_equations (a : ℝ) (h : a ≠ 0) :
  (∃ x, (x + 2)^0 ≠ 1) ∧
  ((a^2 + 1)^0 = 1) ∧
  ((-6*a)^0 = 1) ∧
  ((1/a)^0 = 1) :=
sorry

end NUMINAMATH_CALUDE_power_zero_equations_l1529_152966


namespace NUMINAMATH_CALUDE_bear_census_l1529_152955

def total_bears (black_a : ℕ) : ℕ :=
  let black_b := 3 * black_a
  let black_c := 2 * black_b
  let white_a := black_a / 2
  let white_b := black_b / 2
  let white_c := black_c / 2
  let brown_a := black_a + 40
  let brown_b := black_b + 40
  let brown_c := black_c + 40
  black_a + black_b + black_c +
  white_a + white_b + white_c +
  brown_a + brown_b + brown_c

theorem bear_census (black_a : ℕ) (h1 : black_a = 60) :
  total_bears black_a = 1620 := by
  sorry

end NUMINAMATH_CALUDE_bear_census_l1529_152955


namespace NUMINAMATH_CALUDE_gcd_of_256_180_600_l1529_152910

theorem gcd_of_256_180_600 : Nat.gcd 256 (Nat.gcd 180 600) = 4 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_256_180_600_l1529_152910


namespace NUMINAMATH_CALUDE_expand_expression_l1529_152993

theorem expand_expression (y : ℝ) : 5 * (y - 2) * (y + 7) = 5 * y^2 + 25 * y - 70 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1529_152993


namespace NUMINAMATH_CALUDE_complex_modulus_squared_l1529_152984

theorem complex_modulus_squared : Complex.abs (3/4 + 3*Complex.I)^2 = 153/16 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_squared_l1529_152984


namespace NUMINAMATH_CALUDE_max_abs_z5_l1529_152904

theorem max_abs_z5 (z₁ z₂ z₃ z₄ z₅ : ℂ)
  (h1 : Complex.abs z₁ ≤ 1)
  (h2 : Complex.abs z₂ ≤ 1)
  (h3 : Complex.abs (2 * z₃ - (z₁ + z₂)) ≤ Complex.abs (z₁ - z₂))
  (h4 : Complex.abs (2 * z₄ - (z₁ + z₂)) ≤ Complex.abs (z₁ - z₂))
  (h5 : Complex.abs (2 * z₅ - (z₃ + z₄)) ≤ Complex.abs (z₃ - z₄)) :
  Complex.abs z₅ ≤ Real.sqrt 3 ∧ ∃ z₁ z₂ z₃ z₄ z₅ : ℂ, 
    Complex.abs z₁ ≤ 1 ∧
    Complex.abs z₂ ≤ 1 ∧
    Complex.abs (2 * z₃ - (z₁ + z₂)) ≤ Complex.abs (z₁ - z₂) ∧
    Complex.abs (2 * z₄ - (z₁ + z₂)) ≤ Complex.abs (z₁ - z₂) ∧
    Complex.abs (2 * z₅ - (z₃ + z₄)) ≤ Complex.abs (z₃ - z₄) ∧
    Complex.abs z₅ = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_max_abs_z5_l1529_152904


namespace NUMINAMATH_CALUDE_goldfish_ratio_l1529_152911

/-- Proves the ratio of goldfish Bexley brought to Hershel's initial goldfish -/
theorem goldfish_ratio :
  ∀ (hershel_betta hershel_goldfish bexley_goldfish : ℕ),
  hershel_betta = 10 →
  hershel_goldfish = 15 →
  ∃ (total_after_gift : ℕ),
    total_after_gift = 17 ∧
    (hershel_betta + (2 / 5 : ℚ) * hershel_betta + hershel_goldfish + bexley_goldfish) / 2 = total_after_gift →
    bexley_goldfish * 3 = hershel_goldfish :=
by
  sorry

end NUMINAMATH_CALUDE_goldfish_ratio_l1529_152911


namespace NUMINAMATH_CALUDE_least_five_digit_square_cube_l1529_152931

theorem least_five_digit_square_cube : ∃ n : ℕ,
  (10000 ≤ n ∧ n < 100000) ∧  -- five-digit number
  (∃ a : ℕ, n = a^2) ∧        -- perfect square
  (∃ b : ℕ, n = b^3) ∧        -- perfect cube
  (∀ m : ℕ, 
    (10000 ≤ m ∧ m < 100000) ∧ 
    (∃ x : ℕ, m = x^2) ∧ 
    (∃ y : ℕ, m = y^3) → 
    n ≤ m) ∧                  -- least such number
  n = 15625 := by
sorry

end NUMINAMATH_CALUDE_least_five_digit_square_cube_l1529_152931


namespace NUMINAMATH_CALUDE_total_mangoes_l1529_152991

theorem total_mangoes (alexis_mangoes : ℕ) (dilan_ashley_mangoes : ℕ) : 
  alexis_mangoes = 60 →
  alexis_mangoes = 4 * dilan_ashley_mangoes →
  alexis_mangoes + dilan_ashley_mangoes = 75 :=
by
  sorry

end NUMINAMATH_CALUDE_total_mangoes_l1529_152991


namespace NUMINAMATH_CALUDE_max_a_right_angle_circle_l1529_152969

/-- Given points A(-a, 0) and B(a, 0) where a > 0, and a point C on the circle (x-2)²+(y-2)²=2
    such that ∠ACB = 90°, the maximum value of a is 3√2. -/
theorem max_a_right_angle_circle (a : ℝ) (C : ℝ × ℝ) : 
  a > 0 → 
  (C.1 - 2)^2 + (C.2 - 2)^2 = 2 →
  (C.1 + a) * (C.1 - a) + C.2 * C.2 = 0 →
  a ≤ 3 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_a_right_angle_circle_l1529_152969


namespace NUMINAMATH_CALUDE_perpendicular_lines_condition_l1529_152921

theorem perpendicular_lines_condition (m : ℝ) :
  (∀ x y, m * x + 2 * y - 1 = 0 → 3 * x + (m + 1) * y + 1 = 0 → 
    (m * 3 + 2 * (m + 1) = 0)) ↔ m = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_condition_l1529_152921


namespace NUMINAMATH_CALUDE_car_speed_equality_l1529_152938

/-- Proves that given the conditions of the car problem, the average speed of Car Y is equal to the average speed of Car X -/
theorem car_speed_equality (speed_x : ℝ) (start_delay : ℝ) (distance_after_y_starts : ℝ) : 
  speed_x = 35 →
  start_delay = 72 / 60 →
  distance_after_y_starts = 98 →
  ∃ (speed_y : ℝ), speed_y = speed_x :=
by
  sorry

#check car_speed_equality

end NUMINAMATH_CALUDE_car_speed_equality_l1529_152938


namespace NUMINAMATH_CALUDE_cubic_roots_from_known_root_l1529_152916

/-- Given a cubic polynomial P(x) = x^3 + ax^2 + bx + c and a known root α,
    the other roots of P(x) are the roots of the quadratic polynomial Q(x)
    obtained by dividing P(x) by (x - α). -/
theorem cubic_roots_from_known_root (a b c α : ℝ) :
  (α^3 + a*α^2 + b*α + c = 0) →
  ∃ (p q : ℝ),
    (∀ x, x^3 + a*x^2 + b*x + c = (x - α) * (x^2 + p*x + q)) ∧
    (∀ x, x ≠ α ∧ x^3 + a*x^2 + b*x + c = 0 ↔ x^2 + p*x + q = 0) :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_from_known_root_l1529_152916


namespace NUMINAMATH_CALUDE_simplify_absolute_value_l1529_152995

theorem simplify_absolute_value : |-4^2 + (6 - 2)| = 12 := by
  sorry

end NUMINAMATH_CALUDE_simplify_absolute_value_l1529_152995


namespace NUMINAMATH_CALUDE_pies_from_apples_l1529_152978

/-- Given the rate of pies per apples and a new number of apples, calculate the number of pies that can be made -/
def calculate_pies (initial_apples : ℕ) (initial_pies : ℕ) (new_apples : ℕ) : ℕ :=
  (new_apples * initial_pies) / initial_apples

/-- Theorem stating that given 3 pies can be made from 15 apples, 45 apples will yield 9 pies -/
theorem pies_from_apples :
  calculate_pies 15 3 45 = 9 := by
  sorry

end NUMINAMATH_CALUDE_pies_from_apples_l1529_152978


namespace NUMINAMATH_CALUDE_car_speed_proof_l1529_152999

/-- The speed of a car in km/h -/
def car_speed : ℝ := 30

/-- The reference speed in km/h -/
def reference_speed : ℝ := 36

/-- The additional time taken in seconds -/
def additional_time : ℝ := 20

/-- The distance traveled in km -/
def distance : ℝ := 1

theorem car_speed_proof :
  car_speed = 30 ∧
  (distance / car_speed) * 3600 = (distance / reference_speed) * 3600 + additional_time :=
sorry

end NUMINAMATH_CALUDE_car_speed_proof_l1529_152999


namespace NUMINAMATH_CALUDE_will_uses_six_pages_l1529_152952

/-- The number of cards Will can put on each page -/
def cards_per_page : ℕ := 3

/-- The number of new cards Will has -/
def new_cards : ℕ := 8

/-- The number of old cards Will has -/
def old_cards : ℕ := 10

/-- The total number of cards Will has -/
def total_cards : ℕ := new_cards + old_cards

/-- The number of pages Will uses -/
def pages_used : ℕ := total_cards / cards_per_page

theorem will_uses_six_pages : pages_used = 6 := by
  sorry

end NUMINAMATH_CALUDE_will_uses_six_pages_l1529_152952


namespace NUMINAMATH_CALUDE_alcohol_percentage_in_first_vessel_alcohol_percentage_proof_l1529_152948

theorem alcohol_percentage_in_first_vessel : ℝ → Prop :=
  fun x =>
    let vessel1_capacity : ℝ := 2
    let vessel2_capacity : ℝ := 6
    let vessel2_alcohol_percentage : ℝ := 40
    let total_liquid : ℝ := 8
    let new_mixture_concentration : ℝ := 30

    let vessel2_alcohol_amount : ℝ := vessel2_capacity * (vessel2_alcohol_percentage / 100)
    let total_alcohol_amount : ℝ := total_liquid * (new_mixture_concentration / 100)
    let vessel1_alcohol_amount : ℝ := vessel1_capacity * (x / 100)

    vessel1_alcohol_amount + vessel2_alcohol_amount = total_alcohol_amount →
    x = 0

theorem alcohol_percentage_proof : alcohol_percentage_in_first_vessel 0 := by
  sorry

end NUMINAMATH_CALUDE_alcohol_percentage_in_first_vessel_alcohol_percentage_proof_l1529_152948


namespace NUMINAMATH_CALUDE_smallest_a_l1529_152958

/-- The polynomial with four positive integer roots -/
def p (a b c : ℤ) (x : ℤ) : ℤ := x^4 - a*x^3 + b*x^2 - c*x + 2520

/-- The proposition that the polynomial has four positive integer roots -/
def has_four_positive_integer_roots (a b c : ℤ) : Prop :=
  ∃ w x y z : ℤ, w > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧
    ∀ t : ℤ, p a b c t = 0 ↔ t = w ∨ t = x ∨ t = y ∨ t = z

/-- The theorem stating that 29 is the smallest possible value of a -/
theorem smallest_a :
  ∀ a b c : ℤ, has_four_positive_integer_roots a b c →
  (∀ a' : ℤ, has_four_positive_integer_roots a' b c → a ≤ a') →
  a = 29 :=
sorry

end NUMINAMATH_CALUDE_smallest_a_l1529_152958


namespace NUMINAMATH_CALUDE_last_nonzero_digit_factorial_not_periodic_l1529_152965

/-- The last nonzero digit of a natural number -/
def lastNonzeroDigit (n : ℕ) : ℕ := sorry

/-- The sequence of last nonzero digits of factorials -/
def a (n : ℕ) : ℕ := lastNonzeroDigit (n.factorial)

/-- A sequence is eventually periodic if there exists some point after which it repeats with a fixed period -/
def EventuallyPeriodic (f : ℕ → ℕ) : Prop :=
  ∃ (N p : ℕ), p > 0 ∧ ∀ n ≥ N, f (n + p) = f n

theorem last_nonzero_digit_factorial_not_periodic :
  ¬ EventuallyPeriodic a := sorry

end NUMINAMATH_CALUDE_last_nonzero_digit_factorial_not_periodic_l1529_152965


namespace NUMINAMATH_CALUDE_x_intercept_of_line_l1529_152926

/-- The x-intercept of the line 4x + 6y = 24 is (6, 0) -/
theorem x_intercept_of_line (x y : ℝ) : 
  4 * x + 6 * y = 24 → y = 0 → x = 6 := by sorry

end NUMINAMATH_CALUDE_x_intercept_of_line_l1529_152926


namespace NUMINAMATH_CALUDE_size_relationship_l1529_152922

theorem size_relationship : ∀ a b c : ℝ,
  a = 2^(1/2) → b = 3^(1/3) → c = 5^(1/5) →
  b > a ∧ a > c := by
  sorry

end NUMINAMATH_CALUDE_size_relationship_l1529_152922


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1529_152954

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geom : ∀ n, a (n + 1) = a n * q) 
  (h_sum1 : a 2 + a 3 = 2) 
  (h_sum2 : a 4 + a 5 = 32) : 
  q = 4 ∨ q = -4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1529_152954


namespace NUMINAMATH_CALUDE_expected_sides_formula_rectangle_limit_sides_l1529_152908

/-- Represents a polygon with a given number of sides -/
structure Polygon where
  sides : ℕ

/-- Represents the state after cutting a polygon -/
structure CutState where
  initialPolygon : Polygon
  numCuts : ℕ

/-- Calculates the expected number of sides after cuts -/
def expectedSides (state : CutState) : ℚ :=
  (state.initialPolygon.sides + 4 * state.numCuts) / (state.numCuts + 1)

/-- Theorem: The expected number of sides after cuts is (n + 4k) / (k + 1) -/
theorem expected_sides_formula (state : CutState) :
  expectedSides state = (state.initialPolygon.sides + 4 * state.numCuts) / (state.numCuts + 1) := by
  sorry

/-- Corollary: For a rectangle (4 sides), as cuts approach infinity, expected sides approach 4 -/
theorem rectangle_limit_sides (initialRect : Polygon) (h : initialRect.sides = 4) :
  ∀ ε > 0, ∃ N, ∀ k ≥ N,
    |expectedSides { initialPolygon := initialRect, numCuts := k } - 4| < ε := by
  sorry

end NUMINAMATH_CALUDE_expected_sides_formula_rectangle_limit_sides_l1529_152908


namespace NUMINAMATH_CALUDE_rosalina_gifts_l1529_152913

/-- The number of gifts Rosalina received from Emilio -/
def emilio_gifts : ℕ := 11

/-- The number of gifts Rosalina received from Jorge -/
def jorge_gifts : ℕ := 6

/-- The number of gifts Rosalina received from Pedro -/
def pedro_gifts : ℕ := 4

/-- The total number of gifts Rosalina received -/
def total_gifts : ℕ := emilio_gifts + jorge_gifts + pedro_gifts

theorem rosalina_gifts : total_gifts = 21 := by
  sorry

end NUMINAMATH_CALUDE_rosalina_gifts_l1529_152913


namespace NUMINAMATH_CALUDE_cake_segment_length_squared_l1529_152923

theorem cake_segment_length_squared (d : ℝ) (n : ℕ) (m : ℝ) : 
  d = 20 → n = 4 → m = (d / 2) * Real.sqrt 2 → m^2 = 200 := by
  sorry

end NUMINAMATH_CALUDE_cake_segment_length_squared_l1529_152923


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l1529_152956

theorem fraction_equation_solution :
  ∀ y : ℚ, (2 / 5 : ℚ) - (1 / 3 : ℚ) = 4 / y → y = 60 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l1529_152956


namespace NUMINAMATH_CALUDE_sphere_volume_increase_l1529_152994

theorem sphere_volume_increase (r : ℝ) (h : r > 0) : 
  (4 / 3 * Real.pi * (2 * r)^3) / (4 / 3 * Real.pi * r^3) = 8 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_increase_l1529_152994


namespace NUMINAMATH_CALUDE_number_puzzle_l1529_152914

theorem number_puzzle (x : ℝ) : x / 3 = x - 42 → x = 63 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l1529_152914


namespace NUMINAMATH_CALUDE_overlap_angle_is_90_degrees_l1529_152996

/-- A regular octagon -/
structure RegularOctagon where
  sides : Fin 8 → ℝ × ℝ
  is_regular : ∀ (i j : Fin 8), dist (sides i) (sides ((i + 1) % 8)) = dist (sides j) (sides ((j + 1) % 8))

/-- The angle at the intersection point when two non-adjacent sides of a regular octagon overlap -/
def overlap_angle (octagon : RegularOctagon) : ℝ :=
  sorry

/-- Theorem: The angle at the intersection point when two non-adjacent sides of a regular octagon overlap is 90° -/
theorem overlap_angle_is_90_degrees (octagon : RegularOctagon) : 
  overlap_angle octagon = 90 :=
sorry

end NUMINAMATH_CALUDE_overlap_angle_is_90_degrees_l1529_152996


namespace NUMINAMATH_CALUDE_power_of_2_ending_probabilities_l1529_152998

/-- The probability that 2^n ends with the digit 2, where n is a randomly chosen positive integer -/
def prob_ends_with_2 : ℚ := 1 / 4

/-- The probability that 2^n ends with the digits 12, where n is a randomly chosen positive integer -/
def prob_ends_with_12 : ℚ := 1 / 20

/-- Theorem stating the probabilities for 2^n ending with 2 and 12 -/
theorem power_of_2_ending_probabilities :
  (prob_ends_with_2 = 1 / 4) ∧ (prob_ends_with_12 = 1 / 20) := by
  sorry

end NUMINAMATH_CALUDE_power_of_2_ending_probabilities_l1529_152998


namespace NUMINAMATH_CALUDE_hyperbola_m_range_l1529_152980

-- Define the hyperbola equation
def is_hyperbola (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (2 + m) + y^2 / (m + 1) = 1

-- Define the range of m
def m_range (m : ℝ) : Prop :=
  -2 < m ∧ m < -1

-- Theorem statement
theorem hyperbola_m_range :
  ∀ m : ℝ, is_hyperbola m → m_range m :=
by
  sorry

end NUMINAMATH_CALUDE_hyperbola_m_range_l1529_152980


namespace NUMINAMATH_CALUDE_guayaquilean_sum_of_digits_l1529_152946

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A natural number is guayaquilean if the sum of its digits equals the sum of digits of its square -/
def is_guayaquilean (n : ℕ) : Prop :=
  sum_of_digits n = sum_of_digits (n^2)

/-- The sum of digits of a guayaquilean number is either 9k or 9k + 1 for some k -/
theorem guayaquilean_sum_of_digits (n : ℕ) (h : is_guayaquilean n) :
  ∃ k : ℕ, sum_of_digits n = 9 * k ∨ sum_of_digits n = 9 * k + 1 :=
sorry

end NUMINAMATH_CALUDE_guayaquilean_sum_of_digits_l1529_152946


namespace NUMINAMATH_CALUDE_book_purchase_equations_l1529_152959

/-- Represents the problem of students pooling money to buy a book. -/
theorem book_purchase_equations (x y : ℝ) :
  (∀ (excess shortage : ℝ),
    excess = 4 ∧ shortage = 3 →
    (9 * x - y = excess ∧ y - 8 * x = shortage)) ↔
  (9 * x - y = 4 ∧ y - 8 * x = 3) :=
sorry

end NUMINAMATH_CALUDE_book_purchase_equations_l1529_152959


namespace NUMINAMATH_CALUDE_ratio_problem_l1529_152961

theorem ratio_problem (a b c d : ℝ) 
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 4)
  (h3 : c / d = 7) :
  d / a = 4 / 35 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l1529_152961


namespace NUMINAMATH_CALUDE_quadratic_increasing_condition_l1529_152974

/-- Given a quadratic function f(x) = x^2 - 2ax + 1 that is increasing on [1, +∞),
    prove that a ≤ 1 -/
theorem quadratic_increasing_condition (a : ℝ) : 
  (∀ x ≥ 1, ∀ y ≥ x, (y^2 - 2*a*y + 1) ≥ (x^2 - 2*a*x + 1)) → a ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_increasing_condition_l1529_152974


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1529_152905

theorem right_triangle_hypotenuse (a q : ℝ) (ha : a > 0) (hq : q > 0) :
  ∃ (b c : ℝ),
    c > 0 ∧
    a^2 + b^2 = c^2 ∧
    q * c = b^2 ∧
    c = q / 2 + Real.sqrt ((q / 2)^2 + a^2) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1529_152905


namespace NUMINAMATH_CALUDE_f_geq_a_iff_a_in_range_l1529_152901

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 2

-- Define the domain
def domain : Set ℝ := {x : ℝ | x ≥ -1}

-- State the theorem
theorem f_geq_a_iff_a_in_range (a : ℝ) : 
  (∀ x ∈ domain, f a x ≥ a) ↔ a ∈ Set.Icc (-3) 1 := by sorry

end NUMINAMATH_CALUDE_f_geq_a_iff_a_in_range_l1529_152901


namespace NUMINAMATH_CALUDE_waiter_earnings_l1529_152930

theorem waiter_earnings (total_customers : ℕ) (non_tipping_customers : ℕ) (tip_amount : ℕ) : 
  total_customers = 7 → 
  non_tipping_customers = 5 → 
  tip_amount = 3 → 
  (total_customers - non_tipping_customers) * tip_amount = 6 := by
sorry

end NUMINAMATH_CALUDE_waiter_earnings_l1529_152930


namespace NUMINAMATH_CALUDE_sphere_segment_height_ratio_l1529_152939

/-- Given a sphere of radius R and a plane cutting a segment from it, 
    if the ratio of the segment's volume to the volume of a cone with 
    the same base and height is n, then the height h of the segment 
    is given by h = R / (3 - n), where n < 3 -/
theorem sphere_segment_height_ratio 
  (R : ℝ) 
  (n : ℝ) 
  (h : ℝ) 
  (hn : n < 3) 
  (hR : R > 0) :
  (π * R^2 * (h - R/3)) / ((1/3) * π * R^2 * h) = n → 
  h = R / (3 - n) :=
by sorry

end NUMINAMATH_CALUDE_sphere_segment_height_ratio_l1529_152939


namespace NUMINAMATH_CALUDE_leifs_apples_l1529_152963

def num_oranges : ℕ := 24 -- 2 dozen oranges

theorem leifs_apples :
  ∃ (num_apples : ℕ), num_apples = num_oranges - 10 ∧ num_apples = 14 :=
by sorry

end NUMINAMATH_CALUDE_leifs_apples_l1529_152963


namespace NUMINAMATH_CALUDE_common_difference_is_two_l1529_152927

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_condition : a 2 + a 6 = 8
  fifth_term : a 5 = 6

/-- The common difference of an arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℝ :=
  seq.a 2 - seq.a 1

theorem common_difference_is_two (seq : ArithmeticSequence) :
  common_difference seq = 2 := by
  sorry

end NUMINAMATH_CALUDE_common_difference_is_two_l1529_152927


namespace NUMINAMATH_CALUDE_percent_relation_l1529_152989

theorem percent_relation (a b c : ℝ) (h1 : c = 0.25 * a) (h2 : b = 2.5 * a) :
  c = 0.1 * b := by sorry

end NUMINAMATH_CALUDE_percent_relation_l1529_152989


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1529_152983

theorem inequality_equivalence (x : ℝ) :
  (x - 2) * (2 * x + 3) ≠ 0 →
  ((10 * x^3 - x^2 - 38 * x + 40) / ((x - 2) * (2 * x + 3)) < 2) ↔ (x < 4/5) := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1529_152983


namespace NUMINAMATH_CALUDE_rug_coverage_area_l1529_152976

/-- Given three rugs with specified overlapping areas, calculate the total floor area covered. -/
theorem rug_coverage_area (total_rug_area double_layer triple_layer : ℝ) 
  (h1 : total_rug_area = 212)
  (h2 : double_layer = 24)
  (h3 : triple_layer = 24) :
  total_rug_area - double_layer - 2 * triple_layer = 140 :=
by sorry

end NUMINAMATH_CALUDE_rug_coverage_area_l1529_152976


namespace NUMINAMATH_CALUDE_local_max_is_four_l1529_152940

/-- Given that x = 1 is a point of local minimum for f(x) = x³ - 3ax + 2,
    prove that the point of local maximum for f(x) is 4. -/
theorem local_max_is_four (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^3 - 3*a*x + 2
  (∀ h ∈ Set.Ioo (1 - ε) (1 + ε), f 1 ≤ f h) →
  ∃ ε > 0, ∀ h ∈ Set.Ioo (-1 - ε) (-1 + ε), f h ≤ f (-1) ∧ f (-1) = 4 :=
by sorry

end NUMINAMATH_CALUDE_local_max_is_four_l1529_152940


namespace NUMINAMATH_CALUDE_school_meeting_attendance_l1529_152968

theorem school_meeting_attendance
  (seated_students : ℕ)
  (seated_teachers : ℕ)
  (standing_students : ℕ)
  (h1 : seated_students = 300)
  (h2 : seated_teachers = 30)
  (h3 : standing_students = 25) :
  seated_students + seated_teachers + standing_students = 355 :=
by sorry

end NUMINAMATH_CALUDE_school_meeting_attendance_l1529_152968


namespace NUMINAMATH_CALUDE_flower_cost_minimization_l1529_152970

/-- The cost of flowers given the number of carnations -/
def cost (x : ℕ) : ℕ := 55 - x

/-- The problem statement -/
theorem flower_cost_minimization :
  let total_flowers : ℕ := 11
  let min_lilies : ℕ := 2
  let carnation_cost : ℕ := 4
  let lily_cost : ℕ := 5
  (2 * lily_cost + carnation_cost = 14) →
  (3 * carnation_cost = 2 * lily_cost + 2) →
  (∀ x : ℕ, x ≤ total_flowers - min_lilies → cost x = 55 - x) →
  (∃ x : ℕ, x ≤ total_flowers - min_lilies ∧ cost x = 46 ∧ 
    ∀ y : ℕ, y ≤ total_flowers - min_lilies → cost y ≥ cost x) := by
  sorry

end NUMINAMATH_CALUDE_flower_cost_minimization_l1529_152970


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l1529_152932

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) :=
by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 > 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l1529_152932


namespace NUMINAMATH_CALUDE_seq_of_nat_countable_l1529_152942

/-- The set of all sequences of n natural numbers -/
def SeqOfNat (n : ℕ) : Set (Fin n → ℕ) := Set.univ

/-- A set is countable if there exists an injection from the set to ℕ -/
def IsCountable (α : Type*) : Prop := ∃ f : α → ℕ, Function.Injective f

/-- For any natural number n, the set of all sequences of n natural numbers is countable -/
theorem seq_of_nat_countable (n : ℕ) : IsCountable (SeqOfNat n) := by sorry

end NUMINAMATH_CALUDE_seq_of_nat_countable_l1529_152942


namespace NUMINAMATH_CALUDE_largest_B_term_l1529_152943

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The term B_k in the binomial expansion -/
def B (k : ℕ) : ℝ := (binomial 500 k) * (0.3 ^ k)

/-- The theorem stating that B_k is largest when k = 125 -/
theorem largest_B_term : ∀ k : ℕ, k ≤ 500 → B k ≤ B 125 := by sorry

end NUMINAMATH_CALUDE_largest_B_term_l1529_152943


namespace NUMINAMATH_CALUDE_f_inequality_solution_l1529_152945

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then -Real.log x - x else -Real.log (-x) + x

-- Define the solution set
def solution_set : Set ℝ := {m : ℝ | m ∈ Set.Ioo (-1/2) 0 ∪ Set.Ioo 0 (1/2)}

-- State the theorem
theorem f_inequality_solution :
  ∀ m : ℝ, m ≠ 0 → (f (1/m) < Real.log (1/2) - 2 ↔ m ∈ solution_set) :=
sorry

end NUMINAMATH_CALUDE_f_inequality_solution_l1529_152945


namespace NUMINAMATH_CALUDE_quadratic_polynomial_property_l1529_152917

-- Define the quadratic polynomial f
def f (p q r : ℝ) (x : ℝ) : ℝ := p * x^2 + q * x + r

-- State the theorem
theorem quadratic_polynomial_property 
  (a b c : ℝ) 
  (ha : a ≠ b) (hb : b ≠ c) (hc : a ≠ c)
  (hf : ∃ (p q r : ℝ), 
    f p q r a = b * c ∧ 
    f p q r b = c * a ∧ 
    f p q r c = a * b) : 
  ∃ (p q r : ℝ), f p q r (a + b + c) = a * b + b * c + a * c := by
sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_property_l1529_152917


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l1529_152909

theorem arithmetic_sequence_length (a₁ aₙ d : ℤ) (n : ℕ) 
  (h1 : a₁ = -33)
  (h2 : aₙ = 72)
  (h3 : d = 7)
  (h4 : aₙ = a₁ + (n - 1) * d) :
  n = 16 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l1529_152909


namespace NUMINAMATH_CALUDE_binary_1100_is_12_l1529_152935

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + (if bit then 2^i else 0)) 0

theorem binary_1100_is_12 : 
  binary_to_decimal [false, false, true, true] = 12 := by
  sorry

end NUMINAMATH_CALUDE_binary_1100_is_12_l1529_152935


namespace NUMINAMATH_CALUDE_array_sum_divisibility_l1529_152929

/-- Represents the sum of all terms in a 1/2011-array -/
def arraySum : ℚ :=
  (2011^2 : ℚ) / ((4011 : ℚ) * 2010)

/-- Numerator of the array sum when expressed as a simplified fraction -/
def m : ℕ := 2011^2

/-- Denominator of the array sum when expressed as a simplified fraction -/
def n : ℕ := 4011 * 2010

/-- Theorem stating that m + n is divisible by 2011 -/
theorem array_sum_divisibility : (m + n) % 2011 = 0 := by
  sorry

end NUMINAMATH_CALUDE_array_sum_divisibility_l1529_152929


namespace NUMINAMATH_CALUDE_airplane_passengers_virginia_l1529_152964

/-- Calculates the number of people landing in Virginia given the flight conditions -/
theorem airplane_passengers_virginia
  (initial_passengers : ℕ)
  (texas_off texas_on : ℕ)
  (nc_off nc_on : ℕ)
  (crew : ℕ)
  (h1 : initial_passengers = 124)
  (h2 : texas_off = 58)
  (h3 : texas_on = 24)
  (h4 : nc_off = 47)
  (h5 : nc_on = 14)
  (h6 : crew = 10) :
  initial_passengers - texas_off + texas_on - nc_off + nc_on + crew = 67 :=
by sorry

end NUMINAMATH_CALUDE_airplane_passengers_virginia_l1529_152964


namespace NUMINAMATH_CALUDE_dihedral_angle_relationship_not_determined_l1529_152953

/-- Two dihedral angles with perpendicular half-planes -/
structure PerpendicularDihedralAngles where
  angle1 : ℝ
  angle2 : ℝ
  perpendicular_half_planes : Bool

/-- The relationship between the sizes of two dihedral angles with perpendicular half-planes is not determined -/
theorem dihedral_angle_relationship_not_determined (angles : PerpendicularDihedralAngles) :
  angles.perpendicular_half_planes →
  ¬(∀ a : PerpendicularDihedralAngles, a.angle1 = a.angle2) ∧
  ¬(∀ a : PerpendicularDihedralAngles, a.angle1 + a.angle2 = π) ∧
  ¬(∀ a : PerpendicularDihedralAngles, a.angle1 = a.angle2 ∨ a.angle1 + a.angle2 = π) :=
by sorry

end NUMINAMATH_CALUDE_dihedral_angle_relationship_not_determined_l1529_152953


namespace NUMINAMATH_CALUDE_cos_cube_decomposition_l1529_152903

theorem cos_cube_decomposition (b₁ b₂ b₃ : ℝ) :
  (∀ θ : ℝ, Real.cos θ ^ 3 = b₁ * Real.cos θ + b₂ * Real.cos (2 * θ) + b₃ * Real.cos (3 * θ)) →
  b₁^2 + b₂^2 + b₃^2 = 5/8 := by
  sorry

end NUMINAMATH_CALUDE_cos_cube_decomposition_l1529_152903


namespace NUMINAMATH_CALUDE_inequalities_for_positive_reals_l1529_152972

theorem inequalities_for_positive_reals (a b : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_sum : a + b = 2) : 
  a * b ≤ 1 ∧ a^2 + b^2 ≥ 2 ∧ Real.sqrt a + Real.sqrt b ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequalities_for_positive_reals_l1529_152972


namespace NUMINAMATH_CALUDE_y_relationship_l1529_152949

/-- The quadratic function f(x) = x² + 4x - 5 --/
def f (x : ℝ) : ℝ := x^2 + 4*x - 5

/-- y₁ is the y-coordinate of point A(-4, y₁) on the graph of f --/
def y₁ : ℝ := f (-4)

/-- y₂ is the y-coordinate of point B(-3, y₂) on the graph of f --/
def y₂ : ℝ := f (-3)

/-- y₃ is the y-coordinate of point C(1, y₃) on the graph of f --/
def y₃ : ℝ := f 1

/-- Theorem stating the relationship between y₁, y₂, and y₃ --/
theorem y_relationship : y₂ < y₁ ∧ y₁ < y₃ := by
  sorry

end NUMINAMATH_CALUDE_y_relationship_l1529_152949


namespace NUMINAMATH_CALUDE_max_volume_triangular_cone_l1529_152975

/-- A quadrilateral cone with a square base -/
structure QuadrilateralCone where
  /-- Side length of the square base -/
  baseSideLength : ℝ
  /-- Sum of distances from apex to two adjacent vertices of the base -/
  sumOfDistances : ℝ

/-- Theorem: Maximum volume of triangular cone (A-BCM) -/
theorem max_volume_triangular_cone (cone : QuadrilateralCone) 
  (h1 : cone.baseSideLength = 6)
  (h2 : cone.sumOfDistances = 10) : 
  ∃ (v : ℝ), v = 24 ∧ ∀ (volume : ℝ), volume ≤ v :=
by
  sorry

end NUMINAMATH_CALUDE_max_volume_triangular_cone_l1529_152975


namespace NUMINAMATH_CALUDE_ellipse_condition_l1529_152997

/-- A non-degenerate ellipse is represented by the equation x^2 + 9y^2 - 6x + 27y = b
    if and only if b > -145/4 -/
theorem ellipse_condition (b : ℝ) :
  (∃ (x y : ℝ), x^2 + 9*y^2 - 6*x + 27*y = b) ∧
  (∀ (x y : ℝ), x^2 + 9*y^2 - 6*x + 27*y = b → (x, y) ≠ (0, 0)) ↔
  b > -145/4 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_condition_l1529_152997


namespace NUMINAMATH_CALUDE_triangle_properties_l1529_152986

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.b = 3)
  (h2 : (Real.cos t.A / Real.cos t.B) + (Real.sin t.A / Real.sin t.B) = 2 * t.c / t.b)
  (h3 : t.A + t.B + t.C = Real.pi)
  (h4 : t.A > 0 ∧ t.B > 0 ∧ t.C > 0)
  (h5 : t.a > 0 ∧ t.b > 0 ∧ t.c > 0) :
  t.B = Real.pi / 3 ∧ 
  (∀ (t' : Triangle), t'.b = 3 → t'.a + t'.b + t'.c ≤ 9) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1529_152986


namespace NUMINAMATH_CALUDE_pepper_difference_l1529_152990

/-- Represents the types of curry based on spice level -/
inductive CurryType
| VerySpicy
| Spicy
| Mild

/-- Returns the number of peppers needed for a given curry type -/
def peppersNeeded (c : CurryType) : ℕ :=
  match c with
  | .VerySpicy => 3
  | .Spicy => 2
  | .Mild => 1

/-- Calculates the total number of peppers needed for a given number of curries of each type -/
def totalPeppers (verySpicy spicy mild : ℕ) : ℕ :=
  verySpicy * peppersNeeded CurryType.VerySpicy +
  spicy * peppersNeeded CurryType.Spicy +
  mild * peppersNeeded CurryType.Mild

/-- The main theorem stating the difference in peppers bought -/
theorem pepper_difference : 
  totalPeppers 30 30 10 - totalPeppers 0 15 90 = 40 := by
  sorry

#eval totalPeppers 30 30 10 - totalPeppers 0 15 90

end NUMINAMATH_CALUDE_pepper_difference_l1529_152990


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1529_152915

theorem quadratic_inequality_solution_set :
  let f : ℝ → ℝ := λ x => 2 * x^2 + 7 * x + 3
  let solution_set : Set ℝ := {x | f x > 0}
  solution_set = {x | x < -3 ∨ x > -0.5} := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1529_152915


namespace NUMINAMATH_CALUDE_smallest_x_value_l1529_152900

theorem smallest_x_value : ∃ x : ℚ, 
  (∀ y : ℚ, 7 * (8 * y^2 + 8 * y + 11) = y * (8 * y - 35) → x ≤ y) ∧
  7 * (8 * x^2 + 8 * x + 11) = x * (8 * x - 35) ∧
  x = -7/3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_value_l1529_152900


namespace NUMINAMATH_CALUDE_subset_intersection_condition_solution_set_eq_interval_l1529_152912

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | 2*a + 1 ≤ x ∧ x ≤ 3*a - 5}
def B : Set ℝ := {x | 3 ≤ x ∧ x ≤ 22}

-- Define the theorem
theorem subset_intersection_condition (a : ℝ) :
  (A a).Nonempty ∧ (A a) ⊆ (A a) ∩ B ↔ 6 ≤ a ∧ a ≤ 9 := by
  sorry

-- Define the set of all 'a' that satisfies the condition
def solution_set : Set ℝ := {a | (A a).Nonempty ∧ (A a) ⊆ (A a) ∩ B}

-- Prove that the solution set is equal to the interval [6, 9]
theorem solution_set_eq_interval :
  solution_set = {a | 6 ≤ a ∧ a ≤ 9} := by
  sorry

end NUMINAMATH_CALUDE_subset_intersection_condition_solution_set_eq_interval_l1529_152912


namespace NUMINAMATH_CALUDE_lcm_18_20_l1529_152971

theorem lcm_18_20 : Nat.lcm 18 20 = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_20_l1529_152971


namespace NUMINAMATH_CALUDE_rectangular_plot_breadth_l1529_152992

/-- A rectangular plot with length thrice its breadth and area 363 sq m has a breadth of 11 m -/
theorem rectangular_plot_breadth : 
  ∀ (breadth : ℝ),
  breadth > 0 →
  3 * breadth * breadth = 363 →
  breadth = 11 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_breadth_l1529_152992


namespace NUMINAMATH_CALUDE_factorization_result_quadratic_factorization_l1529_152967

-- Part 1
theorem factorization_result (a b : ℤ) :
  (∀ x, (2*x - 21) * (3*x - 7) - (3*x - 7) * (x - 13) = (3*x + a) * (x + b)) →
  a + 3*b = -31 := by sorry

-- Part 2
theorem quadratic_factorization :
  ∀ x, x^2 - 3*x + 2 = (x - 1) * (x - 2) := by sorry

end NUMINAMATH_CALUDE_factorization_result_quadratic_factorization_l1529_152967


namespace NUMINAMATH_CALUDE_inscribed_square_side_length_l1529_152918

/-- A right triangle with sides 5, 12, and 13 containing an inscribed square -/
structure InscribedSquare where
  /-- Side length of the inscribed square -/
  t : ℝ
  /-- The inscribed square has side length t -/
  is_square : t > 0
  /-- The triangle is a right triangle with sides 5, 12, and 13 -/
  is_right_triangle : 5^2 + 12^2 = 13^2
  /-- The square is inscribed in the triangle -/
  is_inscribed : ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 13 ∧ t / x = 5 / 13 ∧ t / y = 12 / 13

/-- The side length of the inscribed square is 780/169 -/
theorem inscribed_square_side_length (s : InscribedSquare) : s.t = 780 / 169 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_side_length_l1529_152918


namespace NUMINAMATH_CALUDE_fraction_sum_l1529_152934

theorem fraction_sum (a b : ℚ) (h : a / b = 2 / 5) : (a + b) / b = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l1529_152934


namespace NUMINAMATH_CALUDE_y_intercept_of_line_with_slope_3_and_x_intercept_4_l1529_152988

/-- A line is defined by its slope and a point it passes through. -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The y-intercept of a line is the y-coordinate of the point where the line crosses the y-axis. -/
def y_intercept (l : Line) : ℝ :=
  l.slope * (-l.point.1) + l.point.2

theorem y_intercept_of_line_with_slope_3_and_x_intercept_4 :
  let l : Line := { slope := 3, point := (4, 0) }
  y_intercept l = -12 := by sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_with_slope_3_and_x_intercept_4_l1529_152988


namespace NUMINAMATH_CALUDE_revenue_decrease_percentage_l1529_152973

theorem revenue_decrease_percentage (old_revenue new_revenue : ℝ) 
  (h1 : old_revenue = 69.0)
  (h2 : new_revenue = 52.0) :
  ∃ (ε : ℝ), abs ((old_revenue - new_revenue) / old_revenue * 100 - 24.64) < ε ∧ ε > 0 :=
by sorry

end NUMINAMATH_CALUDE_revenue_decrease_percentage_l1529_152973


namespace NUMINAMATH_CALUDE_overlap_area_of_rectangles_l1529_152985

theorem overlap_area_of_rectangles (a b x y : ℝ) : 
  a = 3 ∧ b = 9 ∧  -- Rectangle dimensions
  x^2 + a^2 = y^2 ∧ -- Pythagorean theorem for the corner triangle
  x + y = b ∧ -- Sum of triangle sides equals longer rectangle side
  0 < x ∧ 0 < y -- Positive lengths
  → (b * a - 2 * (x * a / 2)) = 15 := by sorry

end NUMINAMATH_CALUDE_overlap_area_of_rectangles_l1529_152985


namespace NUMINAMATH_CALUDE_greatest_k_for_inequality_l1529_152957

theorem greatest_k_for_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ k : ℝ, k > 0 ∧ 
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → 
    a/b + b/c + c/a - 3 ≥ k * (a/(b+c) + b/(c+a) + c/(a+b) - 3/2)) ∧
  (∀ k' : ℝ, k' > k → 
    ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧
    a/b + b/c + c/a - 3 < k' * (a/(b+c) + b/(c+a) + c/(a+b) - 3/2)) ∧
  k = 1 :=
sorry

end NUMINAMATH_CALUDE_greatest_k_for_inequality_l1529_152957


namespace NUMINAMATH_CALUDE_puppy_cost_l1529_152933

theorem puppy_cost (items_cost : ℝ) (discount_rate : ℝ) (total_spent : ℝ)
  (h1 : items_cost = 95)
  (h2 : discount_rate = 0.2)
  (h3 : total_spent = 96) :
  total_spent - items_cost * (1 - discount_rate) = 20 :=
by sorry

end NUMINAMATH_CALUDE_puppy_cost_l1529_152933


namespace NUMINAMATH_CALUDE_abs_sum_inequality_l1529_152951

theorem abs_sum_inequality (k : ℝ) : 
  (∀ x : ℝ, |x + 2| + |x + 1| > k) → k < 1 :=
by sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_l1529_152951


namespace NUMINAMATH_CALUDE_additional_discount_percentage_l1529_152979

def initial_budget : ℝ := 1000
def first_discount : ℝ := 100
def total_discount : ℝ := 280

def price_after_first_discount : ℝ := initial_budget - first_discount
def final_price : ℝ := initial_budget - total_discount
def additional_discount : ℝ := price_after_first_discount - final_price

theorem additional_discount_percentage : 
  (additional_discount / price_after_first_discount) * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_additional_discount_percentage_l1529_152979


namespace NUMINAMATH_CALUDE_prism_18_edges_has_8_faces_l1529_152977

/-- A prism is a polyhedron with a specific structure. -/
structure Prism where
  edges : ℕ
  lateral_faces : ℕ
  total_faces : ℕ

/-- Theorem: A prism with 18 edges has 8 faces. -/
theorem prism_18_edges_has_8_faces :
  ∀ (p : Prism), p.edges = 18 → p.total_faces = 8 := by
  sorry


end NUMINAMATH_CALUDE_prism_18_edges_has_8_faces_l1529_152977


namespace NUMINAMATH_CALUDE_tomorrow_sunny_is_uncertain_l1529_152925

-- Define the type for events
inductive Event : Type
  | certain : Event
  | impossible : Event
  | inevitable : Event
  | uncertain : Event

-- Define the event "Tomorrow will be sunny"
def tomorrow_sunny : Event := Event.uncertain

-- Define the properties of events
def is_guaranteed (e : Event) : Prop :=
  e = Event.certain ∨ e = Event.inevitable

def cannot_happen (e : Event) : Prop :=
  e = Event.impossible

def is_not_guaranteed (e : Event) : Prop :=
  e = Event.uncertain

-- Theorem statement
theorem tomorrow_sunny_is_uncertain :
  is_not_guaranteed tomorrow_sunny ∧
  ¬is_guaranteed tomorrow_sunny ∧
  ¬cannot_happen tomorrow_sunny :=
by sorry

end NUMINAMATH_CALUDE_tomorrow_sunny_is_uncertain_l1529_152925


namespace NUMINAMATH_CALUDE_fraction_equality_l1529_152919

theorem fraction_equality (a b : ℝ) (h : (2*a - b) / (a + b) = 3/4) : b / a = 5/7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1529_152919


namespace NUMINAMATH_CALUDE_factorization_of_cubic_l1529_152936

theorem factorization_of_cubic (x : ℝ) : 2 * x^3 - 8 * x = 2 * x * (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_cubic_l1529_152936


namespace NUMINAMATH_CALUDE_cow_chicken_leg_excess_l1529_152981

/-- Represents the number of legs more than twice the number of heads in a group of cows and chickens -/
def excess_legs (num_chickens : ℕ) : ℕ :=
  (4 * 10 + 2 * num_chickens) - 2 * (10 + num_chickens)

theorem cow_chicken_leg_excess :
  ∀ num_chickens : ℕ, excess_legs num_chickens = 20 := by
  sorry

end NUMINAMATH_CALUDE_cow_chicken_leg_excess_l1529_152981


namespace NUMINAMATH_CALUDE_bisection_method_calculations_l1529_152937

theorem bisection_method_calculations (a b : Real) (accuracy : Real) :
  a = 1.4 →
  b = 1.5 →
  accuracy = 0.001 →
  ∃ n : ℕ, (((b - a) / (2 ^ n : Real)) < accuracy) ∧ 
    (∀ m : ℕ, m < n → ((b - a) / (2 ^ m : Real)) ≥ accuracy) ∧
    n = 7 :=
by sorry

end NUMINAMATH_CALUDE_bisection_method_calculations_l1529_152937


namespace NUMINAMATH_CALUDE_factorization_equality_l1529_152960

theorem factorization_equality (m n : ℝ) : m^2 - n^2 + 2*m - 2*n = (m-n)*(m+n+2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1529_152960


namespace NUMINAMATH_CALUDE_goat_difference_l1529_152950

theorem goat_difference (adam_goats andrew_goats ahmed_goats : ℕ) : 
  adam_goats = 7 →
  ahmed_goats = 13 →
  andrew_goats = ahmed_goats + 6 →
  andrew_goats - 2 * adam_goats = 5 := by
sorry

end NUMINAMATH_CALUDE_goat_difference_l1529_152950
