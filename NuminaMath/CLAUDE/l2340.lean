import Mathlib

namespace NUMINAMATH_CALUDE_isosceles_right_triangle_l2340_234000

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the properties of the triangle
def isRightTriangle (t : Triangle) : Prop := sorry

def angleAEqualsAngleB (t : Triangle) : Prop := sorry

def sideABLength (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem isosceles_right_triangle 
  (t : Triangle) 
  (h1 : isRightTriangle t) 
  (h2 : angleAEqualsAngleB t) 
  (h3 : sideABLength t = 12) : 
  ∃ (ac_length area : ℝ), 
    ac_length = 6 * Real.sqrt 2 ∧ 
    area = 36 := by sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_l2340_234000


namespace NUMINAMATH_CALUDE_stratified_sample_size_l2340_234073

/-- Proves that for a population of 600 with 250 young employees,
    a stratified sample with 5 young employees has a total size of 12 -/
theorem stratified_sample_size
  (total_population : ℕ)
  (young_population : ℕ)
  (young_sample : ℕ)
  (h1 : total_population = 600)
  (h2 : young_population = 250)
  (h3 : young_sample = 5)
  (h4 : young_population ≤ total_population)
  (h5 : young_sample > 0) :
  ∃ (sample_size : ℕ),
    sample_size * young_population = young_sample * total_population ∧
    sample_size = 12 := by
  sorry


end NUMINAMATH_CALUDE_stratified_sample_size_l2340_234073


namespace NUMINAMATH_CALUDE_hyperbola_y_relationship_l2340_234078

theorem hyperbola_y_relationship (k : ℝ) (y₁ y₂ : ℝ) (h_k_pos : k > 0) 
  (h_A : y₁ = k / 2) (h_B : y₂ = k / 3) : y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_y_relationship_l2340_234078


namespace NUMINAMATH_CALUDE_stadium_attendance_l2340_234075

/-- The number of people in a stadium at the start and end of a game -/
theorem stadium_attendance (boys_start girls_start boys_end girls_end : ℕ) :
  girls_start = 240 →
  boys_end = boys_start - boys_start / 4 →
  girls_end = girls_start - girls_start / 8 →
  boys_end + girls_end = 480 →
  boys_start + girls_start = 600 := by
sorry

end NUMINAMATH_CALUDE_stadium_attendance_l2340_234075


namespace NUMINAMATH_CALUDE_power_of_product_l2340_234082

theorem power_of_product (a b : ℝ) : (a^2 * b)^3 = a^6 * b^3 := by sorry

end NUMINAMATH_CALUDE_power_of_product_l2340_234082


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2340_234032

/-- Given a right triangle with one leg of length 15 and the angle opposite that leg measuring 30°, 
    the hypotenuse has length 30. -/
theorem right_triangle_hypotenuse (leg : ℝ) (angle : ℝ) (h1 : leg = 15) (h2 : angle = 30) :
  let hypotenuse := 2 * leg
  hypotenuse = 30 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2340_234032


namespace NUMINAMATH_CALUDE_exists_empty_subsquare_l2340_234055

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A square in a 2D plane -/
structure Square where
  bottomLeft : Point
  sideLength : ℝ

/-- Checks if a point is inside a square -/
def isInside (p : Point) (s : Square) : Prop :=
  s.bottomLeft.x ≤ p.x ∧ p.x < s.bottomLeft.x + s.sideLength ∧
  s.bottomLeft.y ≤ p.y ∧ p.y < s.bottomLeft.y + s.sideLength

theorem exists_empty_subsquare (bigSquare : Square) (points : Finset Point) :
  bigSquare.sideLength = 4 →
  points.card = 15 →
  (∀ p ∈ points, isInside p bigSquare) →
  ∃ (smallSquare : Square),
    smallSquare.sideLength = 1 ∧
    isInside smallSquare.bottomLeft bigSquare ∧
    (∀ p ∈ points, ¬isInside p smallSquare) :=
by sorry

end NUMINAMATH_CALUDE_exists_empty_subsquare_l2340_234055


namespace NUMINAMATH_CALUDE_min_triangles_to_cover_l2340_234039

/-- The minimum number of unit equilateral triangles needed to cover a larger equilateral triangle and a square -/
theorem min_triangles_to_cover (small_side : ℝ) (large_side : ℝ) (square_side : ℝ) :
  small_side = 1 →
  large_side = 12 →
  square_side = 4 →
  ∃ (n : ℕ), n = ⌈145 * Real.sqrt 3 + 64⌉ ∧
    n * (Real.sqrt 3 / 4 * small_side^2) ≥
      (Real.sqrt 3 / 4 * large_side^2) + square_side^2 :=
by sorry

end NUMINAMATH_CALUDE_min_triangles_to_cover_l2340_234039


namespace NUMINAMATH_CALUDE_complex_expression_equals_negative_two_l2340_234031

theorem complex_expression_equals_negative_two :
  let A := (Real.sqrt 6 + Real.sqrt 2) * (Real.sqrt 3 - 2) * Real.sqrt (Real.sqrt 3 + 2)
  A = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equals_negative_two_l2340_234031


namespace NUMINAMATH_CALUDE_line_segment_length_l2340_234091

/-- Given four points A, B, C, and D on a line in that order, with AB = 2, BD = 6, and CD = 3,
    prove that AC = 1. -/
theorem line_segment_length (A B C D : ℝ) : 
  A < B ∧ B < C ∧ C < D →  -- Points are in order on the line
  B - A = 2 →              -- AB = 2
  D - B = 6 →              -- BD = 6
  D - C = 3 →              -- CD = 3
  C - A = 1 := by          -- AC = 1
sorry


end NUMINAMATH_CALUDE_line_segment_length_l2340_234091


namespace NUMINAMATH_CALUDE_optimal_deposit_rate_l2340_234025

/-- The bank's profit function -/
def profit (k : ℝ) (x : ℝ) : ℝ := 0.048 * k * x^2 - k * x^3

/-- The derivative of the profit function -/
def profit_derivative (k : ℝ) (x : ℝ) : ℝ := 0.096 * k * x - 3 * k * x^2

theorem optimal_deposit_rate (k : ℝ) (h : k > 0) :
  ∃ (x : ℝ), x ∈ Set.Ioo 0 0.048 ∧ 
  (∀ (y : ℝ), y ∈ Set.Ioo 0 0.048 → profit k x ≥ profit k y) ∧
  x = 0.032 := by
  sorry

#eval (0.032 : ℝ) * 100  -- Should output 3.2

end NUMINAMATH_CALUDE_optimal_deposit_rate_l2340_234025


namespace NUMINAMATH_CALUDE_gcd_1230_990_l2340_234086

theorem gcd_1230_990 : Nat.gcd 1230 990 = 30 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1230_990_l2340_234086


namespace NUMINAMATH_CALUDE_T_5_value_l2340_234017

/-- An arithmetic sequence with first term 1 and common difference 1 -/
def a (n : ℕ) : ℚ := n

/-- Sum of the first n terms of the arithmetic sequence -/
def S (n : ℕ) : ℚ := n * (n + 1) / 2

/-- Sum of the first n terms of the sequence {1/S_n} -/
def T (n : ℕ) : ℚ := 2 * n / (n + 1)

/-- Theorem: T_5 = 5/3 -/
theorem T_5_value : T 5 = 5 / 3 := by sorry

end NUMINAMATH_CALUDE_T_5_value_l2340_234017


namespace NUMINAMATH_CALUDE_equation_equality_l2340_234097

theorem equation_equality : 27474 + 3699 + 1985 - 2047 = 31111 := by
  sorry

end NUMINAMATH_CALUDE_equation_equality_l2340_234097


namespace NUMINAMATH_CALUDE_return_trip_duration_l2340_234072

/-- Represents the flight scenario between two towns -/
structure FlightScenario where
  d : ℝ  -- distance between towns
  p : ℝ  -- speed of the plane in still air
  w₁ : ℝ  -- wind speed against the plane
  w₂ : ℝ  -- wind speed with the plane

/-- The conditions of the flight scenario -/
def flight_conditions (f : FlightScenario) : Prop :=
  f.d / (f.p - f.w₁) = 120 ∧  -- outbound trip takes 120 minutes
  f.d / (f.p + f.w₂) = f.d / f.p - 10  -- return trip is 10 minutes faster than in still air

/-- The theorem to prove -/
theorem return_trip_duration (f : FlightScenario) 
  (h : flight_conditions f) : f.d / (f.p + f.w₂) = 72 := by
  sorry


end NUMINAMATH_CALUDE_return_trip_duration_l2340_234072


namespace NUMINAMATH_CALUDE_max_popsicles_is_18_l2340_234019

/-- Represents the number of popsicles in a package -/
inductive Package
| Single : Package
| FourPack : Package
| SevenPack : Package
| NinePack : Package

/-- Returns the cost of a package in dollars -/
def cost (p : Package) : ℕ :=
  match p with
  | Package.Single => 2
  | Package.FourPack => 5
  | Package.SevenPack => 8
  | Package.NinePack => 10

/-- Returns the number of popsicles in a package -/
def popsicles (p : Package) : ℕ :=
  match p with
  | Package.Single => 1
  | Package.FourPack => 4
  | Package.SevenPack => 7
  | Package.NinePack => 9

/-- Represents a combination of packages -/
def Combination := List Package

/-- Calculates the total cost of a combination -/
def totalCost (c : Combination) : ℕ :=
  c.map cost |>.sum

/-- Calculates the total number of popsicles in a combination -/
def totalPopsicles (c : Combination) : ℕ :=
  c.map popsicles |>.sum

/-- Checks if a combination is within budget -/
def withinBudget (c : Combination) : Prop :=
  totalCost c ≤ 20

/-- Theorem: The maximum number of popsicles Pablo can buy with $20 is 18 -/
theorem max_popsicles_is_18 :
  ∀ c : Combination, withinBudget c → totalPopsicles c ≤ 18 :=
by sorry

end NUMINAMATH_CALUDE_max_popsicles_is_18_l2340_234019


namespace NUMINAMATH_CALUDE_equal_sum_sequence_definition_l2340_234012

def is_equal_sum_sequence (a : ℕ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ n : ℕ, n ≥ 1 → a n + a (n + 1) = k

theorem equal_sum_sequence_definition (a : ℕ → ℝ) :
  is_equal_sum_sequence a ↔
  ∃ k : ℝ, ∀ n : ℕ, n ≥ 1 → a n + a (n + 1) = k :=
by sorry

end NUMINAMATH_CALUDE_equal_sum_sequence_definition_l2340_234012


namespace NUMINAMATH_CALUDE_f_composition_at_two_l2340_234074

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem
theorem f_composition_at_two : f (f (f 2)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_at_two_l2340_234074


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l2340_234067

/-- Represents a geometric sequence with positive terms -/
structure GeometricSequence where
  a : ℕ → ℝ
  q : ℝ
  pos_terms : ∀ n, a n > 0
  geom_prop : ∀ n, a (n + 1) = q * a n

/-- Sum of first n terms of a geometric sequence -/
def S (g : GeometricSequence) (n : ℕ) : ℝ := sorry

theorem geometric_sequence_properties (g : GeometricSequence) :
  (-1 : ℝ) < S g 5 ∧ S g 5 < S g 10 ∧  -- S_5 and S_10 are positive
  (S g 5 - (-1) = S g 10 - S g 5) →    -- -1, S_5, S_10 form an arithmetic sequence
  (S g 10 - 2 * S g 5 = 1) ∧           -- First result
  (∀ h : GeometricSequence, 
    ((-1 : ℝ) < S h 5 ∧ S h 5 < S h 10 ∧ 
     S h 5 - (-1) = S h 10 - S h 5) → 
    S g 15 - S g 10 ≤ S h 15 - S h 10) ∧  -- S_15 - S_10 is minimized for g
  (S g 15 - S g 10 = 4) :=              -- Minimum value is 4
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l2340_234067


namespace NUMINAMATH_CALUDE_cylinder_height_relationship_l2340_234061

theorem cylinder_height_relationship (r1 h1 r2 h2 : ℝ) :
  r1 > 0 ∧ h1 > 0 ∧ r2 > 0 ∧ h2 > 0 →
  r2 = 1.2 * r1 →
  π * r1^2 * h1 = π * r2^2 * h2 →
  h1 = 1.44 * h2 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_height_relationship_l2340_234061


namespace NUMINAMATH_CALUDE_two_point_questions_count_l2340_234041

/-- A test with two types of questions -/
structure Test where
  total_points : ℕ
  total_questions : ℕ
  two_point_questions : ℕ
  four_point_questions : ℕ

/-- The test satisfies the given conditions -/
def valid_test (t : Test) : Prop :=
  t.total_points = 100 ∧
  t.total_questions = 40 ∧
  t.two_point_questions + t.four_point_questions = t.total_questions ∧
  2 * t.two_point_questions + 4 * t.four_point_questions = t.total_points

theorem two_point_questions_count (t : Test) (h : valid_test t) :
  t.two_point_questions = 30 :=
by sorry

end NUMINAMATH_CALUDE_two_point_questions_count_l2340_234041


namespace NUMINAMATH_CALUDE_product_of_special_integers_l2340_234016

theorem product_of_special_integers (p q r : ℤ) 
  (h1 : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0)
  (h2 : p + q + r = 30)
  (h3 : 1 / p + 1 / q + 1 / r + 1000 / (p * q * r) = 1) :
  p * q * r = 1600 := by
  sorry

end NUMINAMATH_CALUDE_product_of_special_integers_l2340_234016


namespace NUMINAMATH_CALUDE_ball_radius_l2340_234063

theorem ball_radius (hole_diameter : ℝ) (hole_depth : ℝ) (ball_radius : ℝ) : 
  hole_diameter = 30 ∧ hole_depth = 10 → ball_radius = 16.25 := by
  sorry

end NUMINAMATH_CALUDE_ball_radius_l2340_234063


namespace NUMINAMATH_CALUDE_smallest_integer_y_l2340_234065

theorem smallest_integer_y : ∃ y : ℤ, (y : ℚ) / 4 + 3 / 7 > 2 / 3 ∧ ∀ z : ℤ, z < y → (z : ℚ) / 4 + 3 / 7 ≤ 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_y_l2340_234065


namespace NUMINAMATH_CALUDE_increase_in_average_commission_l2340_234087

/-- Calculates the increase in average commission after a big sale -/
theorem increase_in_average_commission 
  (big_sale_commission : ℕ) 
  (new_average_commission : ℕ) 
  (total_sales : ℕ) 
  (h1 : big_sale_commission = 1000)
  (h2 : new_average_commission = 250)
  (h3 : total_sales = 6) :
  new_average_commission - (new_average_commission * total_sales - big_sale_commission) / (total_sales - 1) = 150 := by
  sorry

end NUMINAMATH_CALUDE_increase_in_average_commission_l2340_234087


namespace NUMINAMATH_CALUDE_board_game_cost_l2340_234002

def number_of_games : ℕ := 6
def total_paid : ℕ := 100
def change_bill_value : ℕ := 5
def number_of_change_bills : ℕ := 2

theorem board_game_cost :
  (total_paid - (change_bill_value * number_of_change_bills)) / number_of_games = 15 := by
  sorry

end NUMINAMATH_CALUDE_board_game_cost_l2340_234002


namespace NUMINAMATH_CALUDE_fraction_sum_inequality_l2340_234095

theorem fraction_sum_inequality (a b : ℝ) (h : a * b < 0) :
  a / b + b / a ≤ -2 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_inequality_l2340_234095


namespace NUMINAMATH_CALUDE_expression_evaluation_l2340_234045

theorem expression_evaluation (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a = 2 / b) :
  (a - 2 / a) * (b + 2 / b) = a^2 - 4 / a^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2340_234045


namespace NUMINAMATH_CALUDE_simplify_radical_product_l2340_234053

theorem simplify_radical_product (x : ℝ) (h : x ≥ 0) :
  Real.sqrt (40 * x) * Real.sqrt (45 * x) * Real.sqrt (56 * x) = 120 * Real.sqrt (7 * x) :=
by sorry

end NUMINAMATH_CALUDE_simplify_radical_product_l2340_234053


namespace NUMINAMATH_CALUDE_number_of_blue_balls_l2340_234034

/-- Given a set of balls with red, blue, and green colors, prove the number of blue balls. -/
theorem number_of_blue_balls
  (total : ℕ)
  (green : ℕ)
  (h1 : total = 40)
  (h2 : green = 7)
  (h3 : ∃ (blue : ℕ), total = green + blue + 2 * blue) :
  ∃ (blue : ℕ), blue = 11 ∧ total = green + blue + 2 * blue :=
sorry

end NUMINAMATH_CALUDE_number_of_blue_balls_l2340_234034


namespace NUMINAMATH_CALUDE_ps_length_l2340_234069

-- Define the triangle PQR
structure Triangle (P Q R : ℝ × ℝ) : Prop where
  right_angle : (Q.1 - P.1) * (R.1 - Q.1) + (Q.2 - P.2) * (R.2 - Q.2) = 0
  pq_length : Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2) = 15
  qr_length : Real.sqrt ((R.1 - Q.1)^2 + (R.2 - Q.2)^2) = 20

-- Define points S and T
def S (P R : ℝ × ℝ) : ℝ × ℝ := sorry
def T (Q R : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the theorem
theorem ps_length (P Q R : ℝ × ℝ) (h : Triangle P Q R) :
  let S := S P R
  let T := T Q R
  (S.1 - P.1) * (T.1 - S.1) + (S.2 - P.2) * (T.2 - S.2) = 0 →
  Real.sqrt ((T.1 - S.1)^2 + (T.2 - S.2)^2) = 12 →
  Real.sqrt ((S.1 - P.1)^2 + (S.2 - P.2)^2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_ps_length_l2340_234069


namespace NUMINAMATH_CALUDE_factorization_3x_squared_minus_27_l2340_234014

theorem factorization_3x_squared_minus_27 (x : ℝ) : 3 * x^2 - 27 = 3 * (x + 3) * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_3x_squared_minus_27_l2340_234014


namespace NUMINAMATH_CALUDE_modulo_eleven_residue_l2340_234043

theorem modulo_eleven_residue : (312 - 3 * 52 + 9 * 165 + 6 * 22) % 11 = 2 := by
  sorry

end NUMINAMATH_CALUDE_modulo_eleven_residue_l2340_234043


namespace NUMINAMATH_CALUDE_expansion_and_a4_imply_a_and_sum_l2340_234020

/-- The expansion of (2x - a)^7 in terms of (x+1) -/
def expansion (a : ℝ) (x : ℝ) : ℝ → ℝ → ℝ → ℝ → ℝ → ℝ → ℝ → ℝ → ℝ :=
  λ a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ =>
    a₀ + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + a₅*(x+1)^5 + a₆*(x+1)^6 + a₇*(x+1)^7

theorem expansion_and_a4_imply_a_and_sum :
  ∀ a : ℝ, ∀ a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ,
    (∀ x : ℝ, (2*x - a)^7 = expansion a x a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇) →
    a₄ = -560 →
    a = -1 ∧ |a₁| + |a₂| + |a₃| + |a₅| + |a₆| + |a₇| = 2186 :=
by sorry

end NUMINAMATH_CALUDE_expansion_and_a4_imply_a_and_sum_l2340_234020


namespace NUMINAMATH_CALUDE_lcm_of_9_12_15_l2340_234080

theorem lcm_of_9_12_15 : Nat.lcm (Nat.lcm 9 12) 15 = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_9_12_15_l2340_234080


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l2340_234056

/-- An arithmetic sequence -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def is_geometric_sequence (b : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, b (n + 1) = q * b n

/-- The theorem statement -/
theorem arithmetic_geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  is_arithmetic_sequence a →
  is_geometric_sequence (λ n => a (2*n - 1) - (2*n - 1)) q →
  q = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l2340_234056


namespace NUMINAMATH_CALUDE_person_c_payment_l2340_234094

def personA : ℕ := 560
def personB : ℕ := 350
def personC : ℕ := 180
def totalDuty : ℕ := 100

def totalMoney : ℕ := personA + personB + personC

def proportionalPayment (money : ℕ) : ℚ :=
  (totalDuty : ℚ) * (money : ℚ) / (totalMoney : ℚ)

theorem person_c_payment :
  round (proportionalPayment personC) = 17 := by
  sorry

end NUMINAMATH_CALUDE_person_c_payment_l2340_234094


namespace NUMINAMATH_CALUDE_triangle_problem_l2340_234066

-- Define the triangle ABC
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side opposite to A
  b : ℝ  -- Side opposite to B
  c : ℝ  -- Side opposite to C

-- Define the theorem
theorem triangle_problem (abc : Triangle) 
  (h1 : Real.cos (abc.A / 2) = 2 * Real.sqrt 5 / 5)
  (h2 : abc.b * abc.c * Real.cos abc.A = 15)
  (h3 : Real.tan abc.B = 2) : 
  -- Part 1: Area of triangle
  (1 / 2 * abc.b * abc.c * Real.sin abc.A = 10) ∧
  -- Part 2: Value of side a
  (abc.a = 2 * Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l2340_234066


namespace NUMINAMATH_CALUDE_estimation_correct_l2340_234050

/-- Represents a school population --/
structure School where
  total_students : ℕ
  sample_size : ℕ
  sample_enthusiasts : ℕ

/-- Calculates the estimated number of enthusiasts in the entire school population --/
def estimate_enthusiasts (s : School) : ℕ :=
  (s.total_students * s.sample_enthusiasts) / s.sample_size

/-- Theorem stating that the estimation method in statement D is correct --/
theorem estimation_correct (s : School) 
  (h1 : s.total_students = 3200)
  (h2 : s.sample_size = 200)
  (h3 : s.sample_enthusiasts = 85) :
  estimate_enthusiasts s = 1360 := by
  sorry

#eval estimate_enthusiasts { total_students := 3200, sample_size := 200, sample_enthusiasts := 85 }

end NUMINAMATH_CALUDE_estimation_correct_l2340_234050


namespace NUMINAMATH_CALUDE_triangle_inequality_l2340_234057

theorem triangle_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
    (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) (h7 : a + b + c = 1) :
    5 * (a^2 + b^2 + c^2) = 18 * a * b * c ∧ 18 * a * b * c ≥ 7/3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2340_234057


namespace NUMINAMATH_CALUDE_abs_value_equivalence_l2340_234008

theorem abs_value_equivalence (x : ℝ) : -1 < x ∧ x < 1 ↔ |x| < 1 := by sorry

end NUMINAMATH_CALUDE_abs_value_equivalence_l2340_234008


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l2340_234018

/-- Given an algebraic expression mx^2 - 2x + n that equals 2 when x = 2,
    prove that it equals 10 when x = -2 -/
theorem algebraic_expression_value (m n : ℝ) 
  (h : m * 2^2 - 2 * 2 + n = 2) : 
  m * (-2)^2 - 2 * (-2) + n = 10 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l2340_234018


namespace NUMINAMATH_CALUDE_min_perimeter_isosceles_triangles_l2340_234013

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  side : ℕ
  base : ℕ

/-- Calculates the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ := 2 * t.side + t.base

/-- Calculates the area of an isosceles triangle -/
noncomputable def area (t : IsoscelesTriangle) : ℝ :=
  (t.base : ℝ) * Real.sqrt ((t.side : ℝ)^2 - ((t.base : ℝ) / 2)^2) / 2

theorem min_perimeter_isosceles_triangles :
  ∃ (t1 t2 : IsoscelesTriangle),
    t1 ≠ t2 ∧
    (t1.base : ℚ) / (t2.base : ℚ) = 5 / 4 ∧
    area t1 = area t2 ∧
    perimeter t1 = perimeter t2 ∧
    perimeter t1 = 192 ∧
    (∀ (s1 s2 : IsoscelesTriangle),
      s1 ≠ s2 →
      (s1.base : ℚ) / (s2.base : ℚ) = 5 / 4 →
      area s1 = area s2 →
      perimeter s1 = perimeter s2 →
      perimeter s1 ≥ 192) :=
by
  sorry

end NUMINAMATH_CALUDE_min_perimeter_isosceles_triangles_l2340_234013


namespace NUMINAMATH_CALUDE_miranda_pillow_stuffing_l2340_234059

/-- 
Given:
- Two pounds of feathers are needed for each pillow
- A pound of goose feathers is approximately 300 feathers
- A pound of duck feathers is approximately 500 feathers
- Miranda's goose has approximately 3600 feathers
- Miranda's duck has approximately 4000 feathers

Prove that Miranda can stuff 10 pillows.
-/
theorem miranda_pillow_stuffing (
  feathers_per_pillow : ℕ)
  (goose_feathers_per_pound : ℕ)
  (duck_feathers_per_pound : ℕ)
  (goose_total_feathers : ℕ)
  (duck_total_feathers : ℕ)
  (h1 : feathers_per_pillow = 2)
  (h2 : goose_feathers_per_pound = 300)
  (h3 : duck_feathers_per_pound = 500)
  (h4 : goose_total_feathers = 3600)
  (h5 : duck_total_feathers = 4000) :
  (goose_total_feathers / goose_feathers_per_pound + 
   duck_total_feathers / duck_feathers_per_pound) / 
  feathers_per_pillow = 10 := by
  sorry

end NUMINAMATH_CALUDE_miranda_pillow_stuffing_l2340_234059


namespace NUMINAMATH_CALUDE_saly_needs_ten_eggs_l2340_234068

/-- The number of eggs needed by various individuals and produced by the farm --/
structure EggNeeds where
  ben_weekly : ℕ  -- Ben's weekly egg needs
  ked_weekly : ℕ  -- Ked's weekly egg needs
  monthly_total : ℕ  -- Total eggs produced by the farm in a month
  weeks_in_month : ℕ  -- Number of weeks in a month

/-- Calculates Saly's weekly egg needs based on the given conditions --/
def saly_weekly_needs (e : EggNeeds) : ℕ :=
  (e.monthly_total - (e.ben_weekly + e.ked_weekly) * e.weeks_in_month) / e.weeks_in_month

/-- Theorem stating that Saly needs 10 eggs per week given the conditions --/
theorem saly_needs_ten_eggs (e : EggNeeds) 
  (h1 : e.ben_weekly = 14)
  (h2 : e.ked_weekly = e.ben_weekly / 2)
  (h3 : e.monthly_total = 124)
  (h4 : e.weeks_in_month = 4) : 
  saly_weekly_needs e = 10 := by
  sorry

end NUMINAMATH_CALUDE_saly_needs_ten_eggs_l2340_234068


namespace NUMINAMATH_CALUDE_virginia_adrienne_difference_l2340_234044

/-- The combined total years of teaching for Virginia, Adrienne, and Dennis -/
def total_years : ℕ := 93

/-- The number of years Dennis has taught -/
def dennis_years : ℕ := 40

/-- The number of years Virginia has taught -/
def virginia_years : ℕ := dennis_years - 9

/-- The number of years Adrienne has taught -/
def adrienne_years : ℕ := total_years - dennis_years - virginia_years

/-- Theorem stating the difference in teaching years between Virginia and Adrienne -/
theorem virginia_adrienne_difference : virginia_years - adrienne_years = 9 := by
  sorry

end NUMINAMATH_CALUDE_virginia_adrienne_difference_l2340_234044


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2340_234089

theorem complex_modulus_problem (z : ℂ) (h : z^2 = -4) : 
  Complex.abs (1 + z) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2340_234089


namespace NUMINAMATH_CALUDE_triangle_is_right_angle_l2340_234077

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)
  (pos_a : 0 < a)
  (pos_b : 0 < b)
  (pos_c : 0 < c)
  (angle_sum : A + B + C = π)

-- State the theorem
theorem triangle_is_right_angle (t : Triangle) 
  (h : (Real.cos (t.A / 2))^2 = (t.b + t.c) / (2 * t.c)) : 
  t.c^2 = t.a^2 + t.b^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_is_right_angle_l2340_234077


namespace NUMINAMATH_CALUDE_number_problem_l2340_234062

theorem number_problem (x : ℚ) : x^2 + 105 = (x - 19)^2 → x = 128/19 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2340_234062


namespace NUMINAMATH_CALUDE_smallest_integer_absolute_value_l2340_234011

theorem smallest_integer_absolute_value (x : ℤ) :
  (∀ y : ℤ, |3 * y - 4| ≤ 22 → x ≤ y) ↔ x = -6 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_absolute_value_l2340_234011


namespace NUMINAMATH_CALUDE_curve_C_properties_l2340_234083

-- Define the curve C
def C (t : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / (4 - t) + p.2^2 / (t - 1) = 1}

-- Define what it means for C to be a hyperbola
def is_hyperbola (t : ℝ) : Prop :=
  t < 1 ∨ t > 4

-- Define what it means for C to be an ellipse with foci on the X-axis
def is_ellipse_x_axis (t : ℝ) : Prop :=
  1 < t ∧ t < 5/2

-- State the theorem
theorem curve_C_properties :
  ∀ t : ℝ,
    (is_hyperbola t ↔ ∃ a b : ℝ, C t = {p : ℝ × ℝ | p.1^2/a^2 - p.2^2/b^2 = 1}) ∧
    (is_ellipse_x_axis t ↔ ∃ a b : ℝ, C t = {p : ℝ × ℝ | p.1^2/a^2 + p.2^2/b^2 = 1} ∧ a > b) :=
by sorry

end NUMINAMATH_CALUDE_curve_C_properties_l2340_234083


namespace NUMINAMATH_CALUDE_quadratic_polynomial_sequence_bound_l2340_234070

/-- A real quadratic polynomial with positive leading coefficient and no fixed point -/
structure QuadraticPolynomial where
  f : ℝ → ℝ
  is_quadratic : ∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c
  positive_leading : ∃ (a b c : ℝ), (∀ x, f x = a * x^2 + b * x + c) ∧ a > 0
  no_fixed_point : ∀ α : ℝ, f α ≠ α

/-- The theorem statement -/
theorem quadratic_polynomial_sequence_bound (f : QuadraticPolynomial) :
  ∃ n : ℕ+, ∀ (a : ℕ → ℝ),
    (∀ i : ℕ, i ≥ 1 → i ≤ n → a i = f.f (a (i-1))) →
    a n > 2021 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_sequence_bound_l2340_234070


namespace NUMINAMATH_CALUDE_steven_peach_apple_difference_l2340_234081

-- Define the number of peaches and apples Steven has
def steven_peaches : ℕ := 18
def steven_apples : ℕ := 11

-- Theorem to prove the difference between peaches and apples
theorem steven_peach_apple_difference :
  steven_peaches - steven_apples = 7 := by
  sorry

end NUMINAMATH_CALUDE_steven_peach_apple_difference_l2340_234081


namespace NUMINAMATH_CALUDE_coeff_x3_in_expansion_l2340_234064

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the coefficient of x^3 in (x-4)^5
def coeff_x3 (x : ℝ) : ℝ := binomial 5 2 * (-4)^2

-- Theorem statement
theorem coeff_x3_in_expansion :
  coeff_x3 x = 160 := by sorry

end NUMINAMATH_CALUDE_coeff_x3_in_expansion_l2340_234064


namespace NUMINAMATH_CALUDE_three_four_five_triangle_l2340_234098

/-- Triangle inequality theorem: the sum of the lengths of any two sides of a triangle
    must be greater than the length of the remaining side. -/
def satisfies_triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A function to check if three given lengths can form a triangle. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ satisfies_triangle_inequality a b c

/-- Theorem stating that the lengths 3, 4, and 5 can form a triangle. -/
theorem three_four_five_triangle :
  can_form_triangle 3 4 5 := by
  sorry


end NUMINAMATH_CALUDE_three_four_five_triangle_l2340_234098


namespace NUMINAMATH_CALUDE_angela_beth_ages_l2340_234029

/-- Angela and Beth's ages problem -/
theorem angela_beth_ages (angela beth : ℕ) 
  (h1 : angela = 4 * beth) 
  (h2 : angela + beth = 55) : 
  angela + 5 = 49 := by sorry

end NUMINAMATH_CALUDE_angela_beth_ages_l2340_234029


namespace NUMINAMATH_CALUDE_right_triangle_division_l2340_234036

theorem right_triangle_division (n : ℝ) (h : n > 0) :
  ∃ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a^2 + b^2 = c^2 ∧
    ∃ (x y : ℝ),
      0 < x ∧ x < c ∧
      0 < y ∧ y < c ∧
      x * y = a * b ∧
      (1/2) * x * a = n * x * y ∧
      (1/2) * y * b = (1/(4*n)) * x * y :=
sorry

end NUMINAMATH_CALUDE_right_triangle_division_l2340_234036


namespace NUMINAMATH_CALUDE_sqrt_of_three_minus_negative_one_equals_two_l2340_234076

theorem sqrt_of_three_minus_negative_one_equals_two :
  Real.sqrt (3 - (-1)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_three_minus_negative_one_equals_two_l2340_234076


namespace NUMINAMATH_CALUDE_negation_equivalence_l2340_234007

theorem negation_equivalence :
  (¬ ∀ (n : ℕ), ∃ (x : ℝ), n^2 < x) ↔ (∃ (n : ℕ), ∀ (x : ℝ), n^2 ≥ x) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2340_234007


namespace NUMINAMATH_CALUDE_weight_of_raisins_l2340_234026

/-- Given that Kelly bought peanuts and raisins, prove the weight of raisins. -/
theorem weight_of_raisins 
  (total_weight : ℝ) 
  (peanut_weight : ℝ) 
  (h1 : total_weight = 0.5) 
  (h2 : peanut_weight = 0.1) : 
  total_weight - peanut_weight = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_raisins_l2340_234026


namespace NUMINAMATH_CALUDE_russian_number_sequence_next_two_elements_l2340_234021

/-- Represents the first letter of a Russian number word -/
inductive RussianNumberLetter
| O  -- Один (One)
| D  -- Два (Two)
| T  -- Три (Three)
| C  -- Четыре (Four)
| P  -- Пять (Five)
| S  -- Шесть (Six)
| S' -- Семь (Seven)
| V  -- Восемь (Eight)

/-- Returns the RussianNumberLetter for a given natural number -/
def russianNumberLetter (n : ℕ) : RussianNumberLetter :=
  match n with
  | 1 => RussianNumberLetter.O
  | 2 => RussianNumberLetter.D
  | 3 => RussianNumberLetter.T
  | 4 => RussianNumberLetter.C
  | 5 => RussianNumberLetter.P
  | 6 => RussianNumberLetter.S
  | 7 => RussianNumberLetter.S'
  | 8 => RussianNumberLetter.V
  | _ => RussianNumberLetter.O  -- Default case, should not be reached for 1-8

theorem russian_number_sequence_next_two_elements :
  russianNumberLetter 7 = RussianNumberLetter.S' ∧
  russianNumberLetter 8 = RussianNumberLetter.V :=
by sorry

end NUMINAMATH_CALUDE_russian_number_sequence_next_two_elements_l2340_234021


namespace NUMINAMATH_CALUDE_five_ruble_coins_l2340_234027

/-- Represents the number of coins of each denomination -/
structure CoinCounts where
  one : Nat
  two : Nat
  five : Nat
  ten : Nat

/-- The problem setup -/
def coin_problem (c : CoinCounts) : Prop :=
  c.one + c.two + c.five + c.ten = 25 ∧
  c.one + c.five + c.ten = 19 ∧
  c.one + c.two + c.five = 20 ∧
  c.two + c.five + c.ten = 16

/-- The theorem to be proved -/
theorem five_ruble_coins (c : CoinCounts) : 
  coin_problem c → c.five = 5 := by
  sorry

end NUMINAMATH_CALUDE_five_ruble_coins_l2340_234027


namespace NUMINAMATH_CALUDE_point_on_linear_function_l2340_234023

/-- Given a linear function y = -2x + 3 and a point (a, -4) on its graph, prove that a = 7/2 -/
theorem point_on_linear_function (a : ℝ) : -2 * a + 3 = -4 → a = 7/2 := by sorry

end NUMINAMATH_CALUDE_point_on_linear_function_l2340_234023


namespace NUMINAMATH_CALUDE_solve_equation_l2340_234048

theorem solve_equation (x : ℝ) : (x^3).sqrt = 9 * (81^(1/9)) → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2340_234048


namespace NUMINAMATH_CALUDE_square_sum_equals_eighteen_l2340_234005

theorem square_sum_equals_eighteen (a b : ℝ) (h1 : a - b = Real.sqrt 2) (h2 : a * b = 4) :
  (a + b)^2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_eighteen_l2340_234005


namespace NUMINAMATH_CALUDE_sorcerer_elixir_combinations_l2340_234088

/-- The number of magical herbs available. -/
def num_herbs : ℕ := 4

/-- The number of enchanted crystals available. -/
def num_crystals : ℕ := 6

/-- The number of crystals that are incompatible with some herbs. -/
def num_incompatible_crystals : ℕ := 2

/-- The number of herbs that each incompatible crystal cannot be used with. -/
def num_incompatible_herbs_per_crystal : ℕ := 3

/-- The total number of valid combinations for the sorcerer's elixir. -/
def valid_combinations : ℕ := 18

theorem sorcerer_elixir_combinations :
  (num_herbs * num_crystals) - (num_incompatible_crystals * num_incompatible_herbs_per_crystal) = valid_combinations :=
by sorry

end NUMINAMATH_CALUDE_sorcerer_elixir_combinations_l2340_234088


namespace NUMINAMATH_CALUDE_train_crossing_time_l2340_234058

/-- Proves that a train crosses a man in 18 seconds given its speed and time to cross a platform. -/
theorem train_crossing_time (train_speed : ℝ) (platform_length : ℝ) (platform_crossing_time : ℝ) :
  train_speed = 20 →
  platform_length = 340 →
  platform_crossing_time = 35 →
  (train_speed * platform_crossing_time - platform_length) / train_speed = 18 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l2340_234058


namespace NUMINAMATH_CALUDE_two_std_dev_below_mean_l2340_234093

/-- Represents a normal distribution --/
structure NormalDistribution where
  mean : ℝ
  std_dev : ℝ

/-- Calculates the value that is a given number of standard deviations away from the mean --/
def value_at_std_devs (d : NormalDistribution) (n : ℝ) : ℝ :=
  d.mean - n * d.std_dev

/-- Theorem: For a normal distribution with mean 15 and standard deviation 1.5,
    the value 2 standard deviations below the mean is 12 --/
theorem two_std_dev_below_mean :
  let d : NormalDistribution := { mean := 15, std_dev := 1.5 }
  value_at_std_devs d 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_two_std_dev_below_mean_l2340_234093


namespace NUMINAMATH_CALUDE_points_on_line_l2340_234092

/-- Three points are collinear if they lie on the same straight line -/
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  (p3.2 - p1.2) * (p2.1 - p1.1) = (p2.2 - p1.2) * (p3.1 - p1.1)

/-- The problem statement -/
theorem points_on_line (k : ℝ) :
  collinear (1, 2) (3, -2) (4, k/3) → k = -12 := by
  sorry

end NUMINAMATH_CALUDE_points_on_line_l2340_234092


namespace NUMINAMATH_CALUDE_factors_of_N_l2340_234071

/-- The number of natural-number factors of N, where N = 2^4 * 3^2 * 5^1 * 7^2 -/
def number_of_factors (N : ℕ) : ℕ :=
  (5 : ℕ) * (3 : ℕ) * (2 : ℕ) * (3 : ℕ)

/-- N is defined as 2^4 * 3^2 * 5^1 * 7^2 -/
def N : ℕ := 2^4 * 3^2 * 5^1 * 7^2

theorem factors_of_N : number_of_factors N = 90 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_N_l2340_234071


namespace NUMINAMATH_CALUDE_missing_chess_pieces_l2340_234030

/-- The number of pieces in a standard chess set -/
def standard_chess_set_pieces : ℕ := 32

/-- The number of pieces present -/
def present_pieces : ℕ := 24

/-- The number of missing chess pieces -/
def missing_pieces : ℕ := standard_chess_set_pieces - present_pieces

theorem missing_chess_pieces :
  missing_pieces = 8 := by sorry

end NUMINAMATH_CALUDE_missing_chess_pieces_l2340_234030


namespace NUMINAMATH_CALUDE_fibonacci_period_correct_l2340_234010

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- The period of the Fibonacci sequence modulo 127 -/
def fibonacci_period : ℕ := 256

theorem fibonacci_period_correct :
  fibonacci_period = 256 ∧
  (∀ m : ℕ, m > 0 → m < 256 → ¬(fib m % 127 = 0 ∧ fib (m + 1) % 127 = 1)) ∧
  fib 256 % 127 = 0 ∧
  fib 257 % 127 = 1 := by
  sorry

#check fibonacci_period_correct

end NUMINAMATH_CALUDE_fibonacci_period_correct_l2340_234010


namespace NUMINAMATH_CALUDE_moon_weight_calculation_l2340_234028

/-- The weight of the moon in tons -/
def moon_weight : ℝ := 250

/-- The weight of Mars in tons -/
def mars_weight : ℝ := 500

/-- The percentage of iron in the composition -/
def iron_percentage : ℝ := 50

/-- The percentage of carbon in the composition -/
def carbon_percentage : ℝ := 20

/-- The percentage of other elements in the composition -/
def other_percentage : ℝ := 100 - iron_percentage - carbon_percentage

/-- The weight of other elements on Mars in tons -/
def mars_other_elements : ℝ := 150

theorem moon_weight_calculation :
  moon_weight = mars_weight / 2 ∧
  mars_weight = mars_other_elements / (other_percentage / 100) :=
by sorry

end NUMINAMATH_CALUDE_moon_weight_calculation_l2340_234028


namespace NUMINAMATH_CALUDE_x_range_for_inequality_l2340_234004

theorem x_range_for_inequality (x : ℝ) :
  (∀ m : ℝ, |m| ≤ 1 → x^2 - 2 > m*x) →
  x < -2 ∨ x > 2 :=
by sorry

end NUMINAMATH_CALUDE_x_range_for_inequality_l2340_234004


namespace NUMINAMATH_CALUDE_tangent_line_y_intercept_l2340_234006

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def circle1 : Circle := ⟨(3, 0), 3⟩
def circle2 : Circle := ⟨(7, 0), 2⟩
def circle3 : Circle := ⟨(11, 0), 1⟩

-- Define the tangent line
structure TangentLine where
  slope : ℝ
  yIntercept : ℝ

-- Function to check if a line is tangent to a circle
def isTangent (l : TangentLine) (c : Circle) : Prop :=
  let (x₀, y₀) := c.center
  let r := c.radius
  let m := l.slope
  let b := l.yIntercept
  (y₀ - m * x₀ - b)^2 = (m^2 + 1) * r^2

-- Theorem statement
theorem tangent_line_y_intercept :
  ∃ l : TangentLine,
    isTangent l circle1 ∧
    isTangent l circle2 ∧
    isTangent l circle3 ∧
    l.yIntercept = 36 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_y_intercept_l2340_234006


namespace NUMINAMATH_CALUDE_javier_exercise_time_l2340_234024

theorem javier_exercise_time :
  ∀ (d : ℕ),
  (50 * d + 90 * 3 = 620) →
  (50 * d = 350) := by
sorry

end NUMINAMATH_CALUDE_javier_exercise_time_l2340_234024


namespace NUMINAMATH_CALUDE_last_locker_opened_l2340_234009

/-- Represents the state of a locker (open or closed) -/
inductive LockerState
| Open
| Closed

/-- Represents the direction of the student's movement -/
inductive Direction
| Forward
| Backward

/-- Defines the locker opening process -/
def openLockers (n : Nat) : Nat :=
  sorry

/-- Theorem stating that the last locker opened is number 86 -/
theorem last_locker_opened (n : Nat) (h : n = 512) : openLockers n = 86 := by
  sorry

end NUMINAMATH_CALUDE_last_locker_opened_l2340_234009


namespace NUMINAMATH_CALUDE_probability_two_defective_approx_l2340_234099

/-- The probability of selecting two defective smartphones from a shipment -/
def probability_two_defective (total : ℕ) (defective : ℕ) : ℚ :=
  (defective : ℚ) / total * ((defective - 1) : ℚ) / (total - 1)

/-- Theorem stating the probability of selecting two defective smartphones -/
theorem probability_two_defective_approx :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/1000 ∧ 
  abs (probability_two_defective 220 84 - 1447/10000) < ε :=
sorry

end NUMINAMATH_CALUDE_probability_two_defective_approx_l2340_234099


namespace NUMINAMATH_CALUDE_minimum_race_distance_l2340_234001

/-- The minimum distance a runner must travel in a race with given conditions -/
theorem minimum_race_distance (wall_length : ℝ) (dist_A_to_wall : ℝ) (dist_wall_to_B : ℝ) :
  wall_length = 1600 →
  dist_A_to_wall = 600 →
  dist_wall_to_B = 800 →
  round (Real.sqrt ((wall_length ^ 2) + (dist_A_to_wall + dist_wall_to_B) ^ 2)) = 2127 :=
by sorry

end NUMINAMATH_CALUDE_minimum_race_distance_l2340_234001


namespace NUMINAMATH_CALUDE_crypto_puzzle_l2340_234051

theorem crypto_puzzle (A B C D : Nat) : 
  A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧  -- Digits are 0-9
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧  -- Unique digits
  A + B + C = D ∧
  B + C = 7 ∧
  A - B = 1 →
  D = 9 := by
sorry

end NUMINAMATH_CALUDE_crypto_puzzle_l2340_234051


namespace NUMINAMATH_CALUDE_min_value_of_function_l2340_234052

theorem min_value_of_function (x : ℝ) (h1 : 0 < x) (h2 : x < 1/3) :
  (3 / (2 * x) + 2 / (1 - 3 * x)) ≥ 25/2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l2340_234052


namespace NUMINAMATH_CALUDE_chefs_wage_difference_l2340_234047

theorem chefs_wage_difference (dishwasher1_wage dishwasher2_wage dishwasher3_wage : ℚ)
  (chef1_percentage chef2_percentage chef3_percentage : ℚ)
  (manager_wage : ℚ) :
  dishwasher1_wage = 6 →
  dishwasher2_wage = 7 →
  dishwasher3_wage = 8 →
  chef1_percentage = 1.2 →
  chef2_percentage = 1.25 →
  chef3_percentage = 1.3 →
  manager_wage = 12.5 →
  manager_wage - (dishwasher1_wage * chef1_percentage + 
                  dishwasher2_wage * chef2_percentage + 
                  dishwasher3_wage * chef3_percentage) = 13.85 := by
  sorry

end NUMINAMATH_CALUDE_chefs_wage_difference_l2340_234047


namespace NUMINAMATH_CALUDE_parallel_perpendicular_lines_l2340_234035

/-- Given a point P and a line l, prove the equations of parallel and perpendicular lines through P -/
theorem parallel_perpendicular_lines
  (P : ℝ × ℝ)  -- Point P
  (l : ℝ → ℝ → Prop)  -- Line l
  (hl : ∀ x y, l x y ↔ 3 * x - 2 * y - 7 = 0)  -- Equation of line l
  (hP : P = (-4, 2))  -- Coordinates of point P
  : 
  -- 1. Equation of parallel line through P
  (∀ x y, (3 * x - 2 * y + 16 = 0) ↔ 
    (∃ k, k ≠ 0 ∧ ∀ a b, l a b → (3 * x - 2 * y = 3 * a - 2 * b + k))) ∧
    (3 * P.1 - 2 * P.2 + 16 = 0) ∧

  -- 2. Equation of perpendicular line through P
  (∀ x y, (2 * x + 3 * y + 2 = 0) ↔ 
    (∀ a b, l a b → (3 * (x - a) + 2 * (y - b) = 0))) ∧
    (2 * P.1 + 3 * P.2 + 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_parallel_perpendicular_lines_l2340_234035


namespace NUMINAMATH_CALUDE_find_b_value_l2340_234003

theorem find_b_value (a b : ℚ) (h1 : 3 * a - 2 = 1) (h2 : 2 * b - 3 * a = 2) : b = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_find_b_value_l2340_234003


namespace NUMINAMATH_CALUDE_rainville_rainfall_2006_l2340_234060

/-- The total rainfall in Rainville in 2006 given the average monthly rainfall in 2005 and the increase in 2006 -/
theorem rainville_rainfall_2006 (rainfall_2005 rainfall_increase : ℝ) : 
  rainfall_2005 = 50.0 →
  rainfall_increase = 3 →
  (rainfall_2005 + rainfall_increase) * 12 = 636 := by
  sorry

end NUMINAMATH_CALUDE_rainville_rainfall_2006_l2340_234060


namespace NUMINAMATH_CALUDE_circle_constraint_extrema_l2340_234022

theorem circle_constraint_extrema :
  ∀ x y : ℝ, x^2 + y^2 = 1 →
  (∀ a b : ℝ, a^2 + b^2 = 1 → (1 + x*y)*(1 - x*y) ≤ (1 + a*b)*(1 - a*b)) ∧
  (∃ a b : ℝ, a^2 + b^2 = 1 ∧ (1 + a*b)*(1 - a*b) = 1) ∧
  (∀ a b : ℝ, a^2 + b^2 = 1 → (1 + a*b)*(1 - a*b) ≥ 3/4) ∧
  (∃ a b : ℝ, a^2 + b^2 = 1 ∧ (1 + a*b)*(1 - a*b) = 3/4) :=
by sorry

end NUMINAMATH_CALUDE_circle_constraint_extrema_l2340_234022


namespace NUMINAMATH_CALUDE_percentage_calculation_l2340_234033

theorem percentage_calculation : 
  (0.2 * 120 + 0.25 * 250 + 0.15 * 80) - 0.1 * 600 = 38.5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l2340_234033


namespace NUMINAMATH_CALUDE_unknown_number_proof_l2340_234038

theorem unknown_number_proof :
  let N : ℕ := 15222392625570
  let a : ℕ := 1155
  let b : ℕ := 1845
  let product : ℕ := a * b
  let difference : ℕ := b - a
  let quotient : ℕ := 15 * (difference * difference)
  N / product = quotient ∧ N % product = 570 :=
by sorry

end NUMINAMATH_CALUDE_unknown_number_proof_l2340_234038


namespace NUMINAMATH_CALUDE_sqrt_difference_simplification_l2340_234015

theorem sqrt_difference_simplification :
  3 * Real.sqrt 2 - |Real.sqrt 2 - Real.sqrt 3| = 4 * Real.sqrt 2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_simplification_l2340_234015


namespace NUMINAMATH_CALUDE_quadratic_function_inequality_l2340_234085

theorem quadratic_function_inequality (a : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : 0 < a) (h₂ : a < 3) 
  (h₃ : x₁ < x₂) (h₄ : x₁ + x₂ = 1 - a) : 
  let f := fun x => a * x^2 + 2 * a * x + 4
  f x₁ < f x₂ := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_inequality_l2340_234085


namespace NUMINAMATH_CALUDE_tims_income_percentage_l2340_234096

theorem tims_income_percentage (tim mart juan : ℝ) 
  (h1 : mart = 1.6 * tim) 
  (h2 : mart = 0.8 * juan) : 
  tim = 0.5 * juan := by
  sorry

end NUMINAMATH_CALUDE_tims_income_percentage_l2340_234096


namespace NUMINAMATH_CALUDE_fraction_simplification_l2340_234054

theorem fraction_simplification :
  (1 - 2 + 4 - 8 + 16 - 32 + 64) / (2 - 4 + 8 - 16 + 32 - 64 + 128) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2340_234054


namespace NUMINAMATH_CALUDE_coefficient_of_x_4_in_expansion_l2340_234037

def binomial_expansion (n : ℕ) (x : ℝ) : ℝ → ℝ := 
  fun a => (1 + a * x)^n

def coefficient_of_x_power (f : ℝ → ℝ) (n : ℕ) : ℝ := sorry

theorem coefficient_of_x_4_in_expansion : 
  coefficient_of_x_power (fun x => (1 + x^2) * binomial_expansion 5 (-2) x) 4 = 120 := by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_4_in_expansion_l2340_234037


namespace NUMINAMATH_CALUDE_solve_equation_l2340_234046

theorem solve_equation : ∃ x : ℚ, (3/4 : ℚ) - (1/2 : ℚ) = 1/x ∧ x = 4 := by sorry

end NUMINAMATH_CALUDE_solve_equation_l2340_234046


namespace NUMINAMATH_CALUDE_range_of_m_l2340_234040

/-- The function f(x) = x^2 - 4x + 5 -/
def f (x : ℝ) := x^2 - 4*x + 5

/-- The maximum value of f on [0, m] is 5 -/
def max_value := 5

/-- The minimum value of f on [0, m] is 1 -/
def min_value := 1

/-- The range of m for which f has max_value and min_value on [0, m] -/
theorem range_of_m :
  ∃ (m : ℝ), m ∈ Set.Icc 2 4 ∧
  (∀ x ∈ Set.Icc 0 m, f x ≤ max_value) ∧
  (∃ x ∈ Set.Icc 0 m, f x = max_value) ∧
  (∀ x ∈ Set.Icc 0 m, f x ≥ min_value) ∧
  (∃ x ∈ Set.Icc 0 m, f x = min_value) ∧
  (∀ m' > 4, ∃ x ∈ Set.Icc 0 m', f x > max_value) ∧
  (∀ m' < 2, ∀ x ∈ Set.Icc 0 m', f x > min_value) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2340_234040


namespace NUMINAMATH_CALUDE_marble_redistribution_l2340_234049

/-- Given Tyrone's initial marbles -/
def tyrone_initial : ℕ := 150

/-- Given Eric's initial marbles -/
def eric_initial : ℕ := 30

/-- The number of marbles Tyrone gives to Eric -/
def marbles_given : ℕ := 15

theorem marble_redistribution :
  (tyrone_initial - marbles_given = 3 * (eric_initial + marbles_given)) ∧
  (0 < marbles_given) ∧ (marbles_given < tyrone_initial) := by
  sorry

end NUMINAMATH_CALUDE_marble_redistribution_l2340_234049


namespace NUMINAMATH_CALUDE_divisible_by_seven_l2340_234084

theorem divisible_by_seven : ∃ k : ℤ, (1 + 5)^4 - 1 = 7 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_seven_l2340_234084


namespace NUMINAMATH_CALUDE_nikolai_wins_l2340_234042

/-- Represents a mountain goat with its jump distance -/
structure Goat where
  name : String
  jump_distance : ℕ

/-- Calculates the number of jumps needed to cover a given distance -/
def jumps_needed (goat : Goat) (distance : ℕ) : ℕ :=
  (distance + goat.jump_distance - 1) / goat.jump_distance

/-- Represents the race between two goats -/
structure Race where
  goat1 : Goat
  goat2 : Goat
  distance : ℕ

/-- Determines if the first goat is faster than the second goat -/
def is_faster (race : Race) : Prop :=
  jumps_needed race.goat1 race.distance < jumps_needed race.goat2 race.distance

theorem nikolai_wins (gennady nikolai : Goat) (h1 : gennady.jump_distance = 6)
    (h2 : nikolai.jump_distance = 4) : is_faster { goat1 := nikolai, goat2 := gennady, distance := 2000 } := by
  sorry

#check nikolai_wins

end NUMINAMATH_CALUDE_nikolai_wins_l2340_234042


namespace NUMINAMATH_CALUDE_largest_number_l2340_234079

-- Define the numbers as real numbers
def a : ℝ := 9.12445
def b : ℝ := 9.124555555555555555555555555555555555555555555555555
def c : ℝ := 9.124545454545454545454545454545454545454545454545454
def d : ℝ := 9.124524524524524524524524524524524524524524524524524
def e : ℝ := 9.124512451245124512451245124512451245124512451245124

-- Theorem statement
theorem largest_number : b > a ∧ b > c ∧ b > d ∧ b > e := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l2340_234079


namespace NUMINAMATH_CALUDE_point_coordinates_l2340_234090

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def is_in_second_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Distance to x-axis -/
def distance_to_x_axis (p : Point) : ℝ :=
  |p.y|

/-- Distance to y-axis -/
def distance_to_y_axis (p : Point) : ℝ :=
  |p.x|

/-- The main theorem -/
theorem point_coordinates (p : Point) 
  (h1 : is_in_second_quadrant p)
  (h2 : distance_to_x_axis p = 7)
  (h3 : distance_to_y_axis p = 3) :
  p = Point.mk (-3) 7 :=
sorry

end NUMINAMATH_CALUDE_point_coordinates_l2340_234090
