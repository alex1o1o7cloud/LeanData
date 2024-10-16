import Mathlib

namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l4190_419016

theorem complex_magnitude_problem (z : ℂ) (h : (3 + 4*Complex.I)*z = 2 + Complex.I) :
  Complex.abs z = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l4190_419016


namespace NUMINAMATH_CALUDE_symmetry_classification_l4190_419006

-- Define the shape type
inductive Shape
  | Parallelogram
  | Rectangle
  | RightTrapezoid
  | Square
  | EquilateralTriangle
  | LineSegment

-- Define properties
def isAxiallySymmetric (s : Shape) : Prop :=
  match s with
  | Shape.Rectangle => True
  | Shape.Square => True
  | Shape.EquilateralTriangle => True
  | Shape.LineSegment => True
  | _ => False

def isCentrallySymmetric (s : Shape) : Prop :=
  match s with
  | Shape.Parallelogram => True
  | Shape.Rectangle => True
  | Shape.Square => True
  | Shape.LineSegment => True
  | _ => False

-- Theorem statement
theorem symmetry_classification (s : Shape) :
  (isAxiallySymmetric s ∧ isCentrallySymmetric s) ↔
  (s = Shape.Rectangle ∨ s = Shape.Square ∨ s = Shape.LineSegment) :=
by sorry

end NUMINAMATH_CALUDE_symmetry_classification_l4190_419006


namespace NUMINAMATH_CALUDE_parallelogram_altitude_base_ratio_l4190_419007

theorem parallelogram_altitude_base_ratio 
  (area : ℝ) 
  (base : ℝ) 
  (altitude : ℝ) 
  (h1 : area = 450) 
  (h2 : base = 15) 
  (h3 : area = base * altitude) 
  (h4 : ∃ k : ℝ, altitude = k * base) : 
  altitude / base = 2 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_altitude_base_ratio_l4190_419007


namespace NUMINAMATH_CALUDE_eighth_term_value_l4190_419001

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1
  first_third_product : a 1 * a 3 = 4
  ninth_term : a 9 = 256

/-- The eighth term of the geometric sequence is either 128 or -128 -/
theorem eighth_term_value (seq : GeometricSequence) : seq.a 8 = 128 ∨ seq.a 8 = -128 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_value_l4190_419001


namespace NUMINAMATH_CALUDE_central_diamond_area_l4190_419010

/-- The area of the central diamond-shaped region in a 10x10 square --/
theorem central_diamond_area (square_side : ℝ) (h : square_side = 10) : 
  let diagonal_length : ℝ := square_side * Real.sqrt 2
  let midpoint_distance : ℝ := square_side / 2
  let diamond_area : ℝ := diagonal_length * midpoint_distance / 2
  diamond_area = 50 := by sorry

end NUMINAMATH_CALUDE_central_diamond_area_l4190_419010


namespace NUMINAMATH_CALUDE_archimedes_schools_l4190_419028

/-- The number of students in Euclid's contest -/
def euclid_participants : ℕ := 69

/-- The number of students per school team -/
def team_size : ℕ := 4

/-- The total number of participants in Archimedes' contest -/
def total_participants : ℕ := euclid_participants + 100

/-- Beth's rank in the contest -/
def beth_rank : ℕ := 45

/-- Carla's rank in the contest -/
def carla_rank : ℕ := 80

/-- Andrea's teammates with lower scores -/
def andreas_lower_teammates : ℕ := 2

theorem archimedes_schools :
  ∃ (num_schools : ℕ), 
    num_schools * team_size = total_participants ∧
    num_schools = 43 :=
sorry

end NUMINAMATH_CALUDE_archimedes_schools_l4190_419028


namespace NUMINAMATH_CALUDE_no_four_consecutive_lucky_numbers_l4190_419012

/-- A function that checks if a number is lucky -/
def is_lucky (n : ℕ) : Prop :=
  1000000 ≤ n ∧ n ≤ 9999999 ∧ 
  ∃ (a b c d e f g : ℕ),
    n = a * 1000000 + b * 100000 + c * 10000 + d * 1000 + e * 100 + f * 10 + g ∧
    a ≠ 0 ∧ (n % (a * b * c * d * e * f * g) = 0)

/-- Theorem stating that four consecutive lucky numbers do not exist -/
theorem no_four_consecutive_lucky_numbers :
  ¬ ∃ (n : ℕ), is_lucky n ∧ is_lucky (n + 1) ∧ is_lucky (n + 2) ∧ is_lucky (n + 3) :=
sorry

end NUMINAMATH_CALUDE_no_four_consecutive_lucky_numbers_l4190_419012


namespace NUMINAMATH_CALUDE_point_transformation_theorem_l4190_419019

/-- Rotation of a point (x, y) by 180° around (h, k) -/
def rotate180 (x y h k : ℝ) : ℝ × ℝ :=
  (2 * h - x, 2 * k - y)

/-- Reflection of a point (x, y) about the line y = x -/
def reflectAboutYEqualX (x y : ℝ) : ℝ × ℝ :=
  (y, x)

theorem point_transformation_theorem (a b : ℝ) :
  let P := (a, b)
  let rotated := rotate180 a b 2 3
  let final := reflectAboutYEqualX rotated.1 rotated.2
  final = (5, -4) →
  a - b = 7 := by
sorry

end NUMINAMATH_CALUDE_point_transformation_theorem_l4190_419019


namespace NUMINAMATH_CALUDE_profit_difference_approx_42_l4190_419000

-- Define the original selling price
def original_selling_price : ℝ := 659.9999999999994

-- Define the original profit percentage
def original_profit_percentage : ℝ := 0.10

-- Define the new purchase price reduction percentage
def new_purchase_reduction : ℝ := 0.10

-- Define the new profit percentage
def new_profit_percentage : ℝ := 0.30

-- Theorem to prove
theorem profit_difference_approx_42 :
  let original_purchase_price := original_selling_price / (1 + original_profit_percentage)
  let new_purchase_price := original_purchase_price * (1 - new_purchase_reduction)
  let new_selling_price := new_purchase_price * (1 + new_profit_percentage)
  ∃ ε > 0, |new_selling_price - original_selling_price - 42| < ε :=
sorry

end NUMINAMATH_CALUDE_profit_difference_approx_42_l4190_419000


namespace NUMINAMATH_CALUDE_one_point_inside_circle_l4190_419026

-- Define a type for points in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a predicate for three points being collinear
def collinear (p q r : Point) : Prop := sorry

-- Define a predicate for a point being on a circle determined by three other points
def onCircle (p q r s : Point) : Prop := sorry

-- Define a predicate for a point being inside a circle determined by three other points
def insideCircle (p q r s : Point) : Prop := sorry

-- Theorem statement
theorem one_point_inside_circle (A B C D : Point) 
  (h_not_collinear : ¬(collinear A B C) ∧ ¬(collinear A B D) ∧ ¬(collinear A C D) ∧ ¬(collinear B C D))
  (h_not_on_circle : ¬(onCircle A B C D) ∧ ¬(onCircle A B D C) ∧ ¬(onCircle A C D B) ∧ ¬(onCircle B C D A)) :
  insideCircle A B C D ∨ insideCircle A B D C ∨ insideCircle A C D B ∨ insideCircle B C D A :=
by sorry

end NUMINAMATH_CALUDE_one_point_inside_circle_l4190_419026


namespace NUMINAMATH_CALUDE_sakshi_investment_dividend_l4190_419025

/-- Calculate the total dividend per annum for Sakshi's investment --/
theorem sakshi_investment_dividend
  (total_investment : ℝ)
  (investment_12_percent : ℝ)
  (price_12_percent : ℝ)
  (price_15_percent : ℝ)
  (dividend_rate_12_percent : ℝ)
  (dividend_rate_15_percent : ℝ)
  (h1 : total_investment = 12000)
  (h2 : investment_12_percent = 4000.000000000002)
  (h3 : price_12_percent = 120)
  (h4 : price_15_percent = 125)
  (h5 : dividend_rate_12_percent = 0.12)
  (h6 : dividend_rate_15_percent = 0.15) :
  ∃ (total_dividend : ℝ), abs (total_dividend - 1680) < 1 :=
sorry

end NUMINAMATH_CALUDE_sakshi_investment_dividend_l4190_419025


namespace NUMINAMATH_CALUDE_bird_families_left_l4190_419002

theorem bird_families_left (total : ℕ) (to_africa : ℕ) (to_asia : ℕ) 
  (h1 : total = 85) (h2 : to_africa = 23) (h3 : to_asia = 37) : 
  total - (to_africa + to_asia) = 25 := by
  sorry

end NUMINAMATH_CALUDE_bird_families_left_l4190_419002


namespace NUMINAMATH_CALUDE_plane_equation_from_perpendicular_foot_l4190_419008

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the coefficients of a plane equation Ax + By + Cz + D = 0 -/
structure PlaneCoefficients where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- Check if a point lies on a plane -/
def pointOnPlane (p : Point3D) (c : PlaneCoefficients) : Prop :=
  c.A * p.x + c.B * p.y + c.C * p.z + c.D = 0

/-- Check if a vector is perpendicular to a plane -/
def vectorPerpendicularToPlane (v : Point3D) (c : PlaneCoefficients) : Prop :=
  ∃ (k : ℝ), v.x = k * c.A ∧ v.y = k * c.B ∧ v.z = k * c.C

/-- The main theorem -/
theorem plane_equation_from_perpendicular_foot : 
  ∃ (c : PlaneCoefficients),
    c.A = 4 ∧ c.B = -3 ∧ c.C = 1 ∧ c.D = -52 ∧
    pointOnPlane ⟨8, -6, 2⟩ c ∧
    vectorPerpendicularToPlane ⟨8, -6, 2⟩ c ∧
    c.A > 0 ∧
    Nat.gcd (Nat.gcd (Nat.gcd (Int.natAbs c.A) (Int.natAbs c.B)) (Int.natAbs c.C)) (Int.natAbs c.D) = 1 :=
by sorry

end NUMINAMATH_CALUDE_plane_equation_from_perpendicular_foot_l4190_419008


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l4190_419024

def f (x : ℝ) : ℝ := -x^2 + x + 6

theorem quadratic_function_properties :
  (f (-3) = -6 ∧ f 0 = 6 ∧ f 2 = 4) →
  (∀ x : ℝ, f x = -x^2 + x + 6) ∧
  (∀ x : ℝ, f x ≤ 25/4) ∧
  (f (1/2) = 25/4) ∧
  (∀ x : ℝ, f (x + 1/2) = f (1/2 - x)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l4190_419024


namespace NUMINAMATH_CALUDE_sqrt_five_squared_minus_four_squared_l4190_419018

theorem sqrt_five_squared_minus_four_squared : 
  Real.sqrt (5^2 - 4^2) = 3 := by sorry

end NUMINAMATH_CALUDE_sqrt_five_squared_minus_four_squared_l4190_419018


namespace NUMINAMATH_CALUDE_total_candy_count_l4190_419029

theorem total_candy_count (brother_candy : ℕ) (wendy_boxes : ℕ) (pieces_per_box : ℕ) : 
  brother_candy = 6 → wendy_boxes = 2 → pieces_per_box = 3 →
  brother_candy + wendy_boxes * pieces_per_box = 12 :=
by sorry

end NUMINAMATH_CALUDE_total_candy_count_l4190_419029


namespace NUMINAMATH_CALUDE_x_minus_y_equals_106_over_21_l4190_419021

theorem x_minus_y_equals_106_over_21 (x y : ℚ) : 
  x + 2*y = 16/3 → 5*x + 3*y = 26 → x - y = 106/21 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_106_over_21_l4190_419021


namespace NUMINAMATH_CALUDE_fraction_leading_zeros_l4190_419017

-- Define the fraction
def fraction : ℚ := 7 / 5000

-- Define a function to count leading zeros in a decimal representation
def countLeadingZeros (q : ℚ) : ℕ := sorry

-- Theorem statement
theorem fraction_leading_zeros :
  countLeadingZeros fraction = 2 := by sorry

end NUMINAMATH_CALUDE_fraction_leading_zeros_l4190_419017


namespace NUMINAMATH_CALUDE_floor_sqrt_eight_count_l4190_419005

theorem floor_sqrt_eight_count : 
  (Finset.range 81 \ Finset.range 64).card = 17 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_eight_count_l4190_419005


namespace NUMINAMATH_CALUDE_seminar_attendees_seminar_attendees_solution_l4190_419015

theorem seminar_attendees (total : ℕ) (company_a : ℕ) : ℕ :=
  let company_b := 2 * company_a
  let company_c := company_a + 10
  let company_d := company_c - 5
  let from_companies := company_a + company_b + company_c + company_d
  total - from_companies

theorem seminar_attendees_solution :
  seminar_attendees 185 30 = 20 := by
  sorry

end NUMINAMATH_CALUDE_seminar_attendees_seminar_attendees_solution_l4190_419015


namespace NUMINAMATH_CALUDE_product_repeating_nine_and_nine_l4190_419009

/-- The repeating decimal 0.999... -/
def repeating_decimal_nine : ℚ := 1

theorem product_repeating_nine_and_nine :
  repeating_decimal_nine * 9 = 9 := by sorry

end NUMINAMATH_CALUDE_product_repeating_nine_and_nine_l4190_419009


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l4190_419027

theorem quadratic_equation_solutions :
  ∀ x : ℝ, x^2 = x ↔ x = 0 ∨ x = 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l4190_419027


namespace NUMINAMATH_CALUDE_worker_a_payment_share_l4190_419022

/-- Calculates the share of payment for worker A given the total days needed for each worker and the total payment -/
def worker_a_share (days_a days_b : ℕ) (total_payment : ℚ) : ℚ :=
  let work_rate_a := 1 / days_a
  let work_rate_b := 1 / days_b
  let combined_rate := work_rate_a + work_rate_b
  let a_share_ratio := work_rate_a / combined_rate
  a_share_ratio * total_payment

/-- Theorem stating that worker A's share is 89.55 given the problem conditions -/
theorem worker_a_payment_share :
  worker_a_share 12 18 (149.25 : ℚ) = (8955 : ℚ) / 100 := by
  sorry

#eval worker_a_share 12 18 (149.25 : ℚ)

end NUMINAMATH_CALUDE_worker_a_payment_share_l4190_419022


namespace NUMINAMATH_CALUDE_roots_quadratic_equation_l4190_419004

theorem roots_quadratic_equation (a b : ℝ) : 
  (a^2 - a - 2013 = 0) → (b^2 - b - 2013 = 0) → (a^2 + 2*a + 3*b - 2 = 2014) := by
  sorry

end NUMINAMATH_CALUDE_roots_quadratic_equation_l4190_419004


namespace NUMINAMATH_CALUDE_polynomial_root_relation_l4190_419011

-- Define the polynomials h(x) and k(x)
def h (p : ℝ) (x : ℝ) : ℝ := x^3 + p*x^2 + x + 15
def k (q s : ℝ) (x : ℝ) : ℝ := x^4 + x^3 + q*x^2 + 150*x + s

-- State the theorem
theorem polynomial_root_relation (p q s : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    h p x = 0 ∧ h p y = 0 ∧ h p z = 0) →
  (∀ x : ℝ, h p x = 0 → k q s x = 0) →
  k q s 1 = -16048 :=
sorry

end NUMINAMATH_CALUDE_polynomial_root_relation_l4190_419011


namespace NUMINAMATH_CALUDE_expression_simplification_l4190_419023

theorem expression_simplification (x : ℝ) (h : x ≠ 0) :
  (x * (3 - 4 * x) + 2 * x^2 * (x - 1)) / (-2 * x) = -x^2 + 3 * x - 3/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l4190_419023


namespace NUMINAMATH_CALUDE_min_max_sum_x_l4190_419014

theorem min_max_sum_x (x y z : ℝ) 
  (sum_eq : x + y + z = 6)
  (sum_sq_eq : x^2 + y^2 + z^2 = 14) :
  ∃ (m M : ℝ), (∀ t, m ≤ t ∧ t ≤ M → ∃ u v, t + u + v = 6 ∧ t^2 + u^2 + v^2 = 14) ∧
                (∀ s, (∃ u v, s + u + v = 6 ∧ s^2 + u^2 + v^2 = 14) → m ≤ s ∧ s ≤ M) ∧
                m + M = 10/3 :=
sorry

end NUMINAMATH_CALUDE_min_max_sum_x_l4190_419014


namespace NUMINAMATH_CALUDE_percentage_difference_l4190_419003

theorem percentage_difference (p j t : ℝ) 
  (hj : j = 0.75 * p) 
  (ht : t = 0.9375 * p) : 
  (t - j) / t = 0.2 := by
sorry

end NUMINAMATH_CALUDE_percentage_difference_l4190_419003


namespace NUMINAMATH_CALUDE_tangent_double_angle_identity_l4190_419020

theorem tangent_double_angle_identity (α : Real) (h : 0 < α ∧ α < π/4) : 
  Real.tan (2 * α) / Real.tan α = 1 + 1 / Real.cos (2 * α) := by
  sorry

end NUMINAMATH_CALUDE_tangent_double_angle_identity_l4190_419020


namespace NUMINAMATH_CALUDE_determine_c_l4190_419013

/-- Given integers a and b, where there exist unique x, y, z satisfying the LCM conditions,
    the value of c can be uniquely determined. -/
theorem determine_c (a b : ℕ) 
  (h_exists : ∃! (x y z : ℕ), a = Nat.lcm y z ∧ b = Nat.lcm x z ∧ ∃ c, c = Nat.lcm x y) :
  ∃! c, ∀ (x y z : ℕ), 
    (a = Nat.lcm y z ∧ b = Nat.lcm x z ∧ c = Nat.lcm x y) → 
    (∀ (x' y' z' : ℕ), a = Nat.lcm y' z' ∧ b = Nat.lcm x' z' → (x = x' ∧ y = y' ∧ z = z')) :=
by sorry


end NUMINAMATH_CALUDE_determine_c_l4190_419013
