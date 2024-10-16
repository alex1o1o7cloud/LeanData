import Mathlib

namespace NUMINAMATH_CALUDE_composite_function_equality_l3692_369225

/-- Given two functions f and g, prove that f(g(f(3))) = 332 -/
theorem composite_function_equality (f g : ℝ → ℝ) 
  (hf : ∀ x, f x = 4 * x + 4) 
  (hg : ∀ x, g x = 5 * x + 2) : 
  f (g (f 3)) = 332 := by
  sorry

end NUMINAMATH_CALUDE_composite_function_equality_l3692_369225


namespace NUMINAMATH_CALUDE_line_y_coordinate_at_15_l3692_369293

/-- A line passing through three given points -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ
  point3 : ℝ × ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1)

theorem line_y_coordinate_at_15 (l : Line) 
    (h1 : l.point1 = (4, 5))
    (h2 : l.point2 = (8, 17))
    (h3 : l.point3 = (12, 29))
    (h4 : collinear l.point1 l.point2 l.point3) :
    ∃ t : ℝ, collinear l.point1 l.point2 (15, t) ∧ t = 38 := by
  sorry

end NUMINAMATH_CALUDE_line_y_coordinate_at_15_l3692_369293


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3692_369227

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x + 4 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2*x + 4 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3692_369227


namespace NUMINAMATH_CALUDE_square_construction_l3692_369209

/-- A point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A line in a 2D plane defined by two points -/
structure Line2D where
  p1 : Point2D
  p2 : Point2D

/-- A square in a 2D plane -/
structure Square where
  vertices : Fin 4 → Point2D

/-- Check if a point lies on a line -/
def pointOnLine (p : Point2D) (l : Line2D) : Prop := sorry

/-- Check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line2D) : Prop := sorry

/-- Check if all sides of a square are equal length -/
def equalSides (s : Square) : Prop := sorry

/-- The main theorem -/
theorem square_construction (A B C D : Point2D) :
  ∃ (s : Square),
    (∀ i : Fin 4, ∃ p ∈ [A, B, C, D], pointOnLine (s.vertices i) (Line2D.mk (s.vertices i) (s.vertices ((i + 1) % 4)))) ∧
    (∀ i : Fin 4, perpendicular (Line2D.mk (s.vertices i) (s.vertices ((i + 1) % 4))) (Line2D.mk (s.vertices ((i + 1) % 4)) (s.vertices ((i + 2) % 4)))) ∧
    equalSides s :=
sorry

end NUMINAMATH_CALUDE_square_construction_l3692_369209


namespace NUMINAMATH_CALUDE_quadratic_solutions_solution_at_minus_four_solution_at_minus_three_no_solution_at_minus_five_l3692_369242

-- Define the quadratic equation
def quadratic (a x : ℝ) : ℝ := x^2 - (a - 12) * x + 36 - 5 * a

-- Define the condition for x
def x_condition (x : ℝ) : Prop := -6 < x ∧ x ≤ -2 ∧ x ≠ -5 ∧ x ≠ -4 ∧ x ≠ -3

-- Define the range for a
def a_range (a : ℝ) : Prop := (4 < a ∧ a < 4.5) ∨ (4.5 < a ∧ a ≤ 16/3)

-- Main theorem
theorem quadratic_solutions (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x_condition x₁ ∧ x_condition x₂ ∧ 
   quadratic a x₁ = 0 ∧ quadratic a x₂ = 0) ↔ a_range a :=
sorry

-- Theorems for specific points
theorem solution_at_minus_four :
  quadratic 4 (-4) = 0 :=
sorry

theorem solution_at_minus_three :
  quadratic 4.5 (-3) = 0 :=
sorry

theorem no_solution_at_minus_five (a : ℝ) :
  quadratic a (-5) ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_solutions_solution_at_minus_four_solution_at_minus_three_no_solution_at_minus_five_l3692_369242


namespace NUMINAMATH_CALUDE_window_installation_time_l3692_369286

theorem window_installation_time (total_windows : ℕ) (installed_windows : ℕ) (time_per_window : ℕ) 
  (h1 : total_windows = 14)
  (h2 : installed_windows = 5)
  (h3 : time_per_window = 4) : 
  (total_windows - installed_windows) * time_per_window = 36 := by
  sorry

end NUMINAMATH_CALUDE_window_installation_time_l3692_369286


namespace NUMINAMATH_CALUDE_wheel_moves_200cm_per_rotation_l3692_369290

/-- Represents the properties of a rotating wheel -/
structure RotatingWheel where
  rotations_per_minute : ℕ
  distance_per_hour : ℕ

/-- Calculates the distance moved during each rotation of the wheel -/
def distance_per_rotation (wheel : RotatingWheel) : ℚ :=
  wheel.distance_per_hour / (wheel.rotations_per_minute * 60)

/-- Theorem stating that a wheel with the given properties moves 200 cm per rotation -/
theorem wheel_moves_200cm_per_rotation (wheel : RotatingWheel) 
    (h1 : wheel.rotations_per_minute = 10)
    (h2 : wheel.distance_per_hour = 120000) : 
  distance_per_rotation wheel = 200 := by
  sorry

end NUMINAMATH_CALUDE_wheel_moves_200cm_per_rotation_l3692_369290


namespace NUMINAMATH_CALUDE_opposite_reciprocal_sum_l3692_369244

theorem opposite_reciprocal_sum (a b c : ℝ) 
  (h1 : a + b = 0)  -- a and b are opposite numbers
  (h2 : c = 1/4)    -- the reciprocal of c is 4
  : 3*a + 3*b - 4*c = -1 := by
  sorry

end NUMINAMATH_CALUDE_opposite_reciprocal_sum_l3692_369244


namespace NUMINAMATH_CALUDE_dolls_count_l3692_369271

theorem dolls_count (total_toys : ℕ) (action_figure_fraction : ℚ) : 
  total_toys = 24 → action_figure_fraction = 1/4 → 
  total_toys - (total_toys * action_figure_fraction).floor = 18 := by
sorry

end NUMINAMATH_CALUDE_dolls_count_l3692_369271


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l3692_369249

/-- Two vectors are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (-2, m)
  parallel a b → m = -4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l3692_369249


namespace NUMINAMATH_CALUDE_cody_initial_money_l3692_369235

theorem cody_initial_money : 
  ∀ (initial : ℕ), 
  (initial + 9 - 19 = 35) → 
  initial = 45 := by
sorry

end NUMINAMATH_CALUDE_cody_initial_money_l3692_369235


namespace NUMINAMATH_CALUDE_circle_projection_bodies_l3692_369207

/-- A type representing geometric bodies -/
inductive GeometricBody
  | Cone
  | Cylinder
  | Sphere
  | Other

/-- A predicate that determines if a geometric body appears as a circle from a certain perspective -/
def appearsAsCircle (body : GeometricBody) : Prop :=
  sorry

/-- The theorem stating that cones, cylinders, and spheres appear as circles from certain perspectives -/
theorem circle_projection_bodies :
  ∃ (cone cylinder sphere : GeometricBody),
    cone = GeometricBody.Cone ∧
    cylinder = GeometricBody.Cylinder ∧
    sphere = GeometricBody.Sphere ∧
    appearsAsCircle cone ∧
    appearsAsCircle cylinder ∧
    appearsAsCircle sphere :=
  sorry

end NUMINAMATH_CALUDE_circle_projection_bodies_l3692_369207


namespace NUMINAMATH_CALUDE_linear_equation_solution_l3692_369201

theorem linear_equation_solution (x y : ℚ) :
  4 * x - 5 * y = 9 → y = (4 * x - 9) / 5 := by
sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l3692_369201


namespace NUMINAMATH_CALUDE_gcd_1234_1987_l3692_369265

theorem gcd_1234_1987 : Int.gcd 1234 1987 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1234_1987_l3692_369265


namespace NUMINAMATH_CALUDE_base5_division_l3692_369240

/-- Converts a base 5 number represented as a list of digits to its decimal equivalent -/
def toDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 5 * acc + d) 0

/-- Converts a decimal number to its base 5 representation as a list of digits -/
def toBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec go (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else go (m / 5) ((m % 5) :: acc)
    go n []

/-- Theorem stating that the quotient of 1121₅ ÷ 12₅ in base 5 is equal to 43₅ -/
theorem base5_division :
  toBase5 (toDecimal [1, 1, 2, 1] / toDecimal [1, 2]) = [4, 3] := by
  sorry

end NUMINAMATH_CALUDE_base5_division_l3692_369240


namespace NUMINAMATH_CALUDE_inequality_proof_l3692_369245

theorem inequality_proof (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) :
  (((1 - a) ^ (1 / b) ≤ (1 - a) ^ b) ∧
   ((1 + a) ^ a ≤ (1 + b) ^ b) ∧
   ((1 - a) ^ b ≤ (1 - a) ^ (b / 2))) ∧
  ((1 - a) ^ a > (1 - b) ^ b) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3692_369245


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l3692_369298

theorem solution_set_of_inequality (x : ℝ) :
  (x - 3) / (x + 2) < 0 ↔ -2 < x ∧ x < 3 := by
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l3692_369298


namespace NUMINAMATH_CALUDE_quadrilateral_inequality_l3692_369237

-- Define a structure for a quadrilateral
structure Quadrilateral :=
  (a b c d e f : ℝ)
  (a_pos : a > 0)
  (b_pos : b > 0)
  (c_pos : c > 0)
  (d_pos : d > 0)
  (e_pos : e > 0)
  (f_pos : f > 0)

-- Define what it means for a quadrilateral to be cyclic
def is_cyclic (q : Quadrilateral) : Prop :=
  q.e^2 + q.f^2 = q.b^2 + q.d^2 + 2*q.a*q.c

-- State the theorem
theorem quadrilateral_inequality (q : Quadrilateral) :
  q.e^2 + q.f^2 ≤ q.b^2 + q.d^2 + 2*q.a*q.c ∧
  (q.e^2 + q.f^2 = q.b^2 + q.d^2 + 2*q.a*q.c ↔ is_cyclic q) :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_inequality_l3692_369237


namespace NUMINAMATH_CALUDE_circle_condition_l3692_369250

-- Define the equation of the circle
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 4*y + m = 0

-- Theorem statement
theorem circle_condition (m : ℝ) :
  (∃ x y : ℝ, circle_equation x y m) ↔ m < 5 := by
  sorry

end NUMINAMATH_CALUDE_circle_condition_l3692_369250


namespace NUMINAMATH_CALUDE_equilateral_triangle_division_l3692_369263

/-- Given an equilateral triangle with sides divided into three equal parts and an inner equilateral
    triangle formed by connecting corresponding division points, if the inscribed circle in the inner
    triangle has radius 6 cm, then the side length of the inner triangle is 12√3 cm and the side
    length of the outer triangle is 36 cm. -/
theorem equilateral_triangle_division (r : ℝ) (inner_side outer_side : ℝ) :
  r = 6 →
  inner_side = 12 * Real.sqrt 3 →
  outer_side = 36 :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_division_l3692_369263


namespace NUMINAMATH_CALUDE_min_value_given_condition_l3692_369206

theorem min_value_given_condition (a b : ℝ) : 
  (|a - 2| + (b + 3)^2 = 0) → 
  (min (min (min (a + b) (a - b)) (b^a)) (a * b) = a * b) :=
by sorry

end NUMINAMATH_CALUDE_min_value_given_condition_l3692_369206


namespace NUMINAMATH_CALUDE_s_square_minus_product_abs_eq_eight_l3692_369226

/-- The sequence s_n defined for three real numbers a, b, c -/
def s (a b c : ℝ) : ℕ → ℝ
  | 0 => 3  -- s_0 = a^0 + b^0 + c^0 = 3
  | n + 1 => a^(n + 1) + b^(n + 1) + c^(n + 1)

/-- The theorem statement -/
theorem s_square_minus_product_abs_eq_eight
  (a b c : ℝ)
  (h1 : s a b c 1 = 2)
  (h2 : s a b c 2 = 6)
  (h3 : s a b c 3 = 14) :
  ∀ n : ℕ, n > 1 → |(s a b c n)^2 - (s a b c (n-1)) * (s a b c (n+1))| = 8 := by
  sorry

end NUMINAMATH_CALUDE_s_square_minus_product_abs_eq_eight_l3692_369226


namespace NUMINAMATH_CALUDE_remainder_problem_l3692_369259

theorem remainder_problem : (7 * 7^10 + 1^10) % 11 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3692_369259


namespace NUMINAMATH_CALUDE_triangle_interior_point_inequality_l3692_369258

open Real

variable (A B C M : ℝ × ℝ)

def isInside (M A B C : ℝ × ℝ) : Prop := sorry

def distance (P Q : ℝ × ℝ) : ℝ := sorry

theorem triangle_interior_point_inequality 
  (h : isInside M A B C) :
  min (distance M A) (min (distance M B) (distance M C)) + 
  distance M A + distance M B + distance M C < 
  distance A B + distance B C + distance C A := by
  sorry

end NUMINAMATH_CALUDE_triangle_interior_point_inequality_l3692_369258


namespace NUMINAMATH_CALUDE_handshaking_theorem_l3692_369295

/-- Represents a handshaking arrangement for 11 people -/
def HandshakingArrangement := Fin 11 → Finset (Fin 11)

/-- The number of people in the group -/
def group_size : Nat := 11

/-- The number of handshakes per person -/
def handshakes_per_person : Nat := 3

/-- Predicate for a valid handshaking arrangement -/
def is_valid_arrangement (a : HandshakingArrangement) : Prop :=
  ∀ i : Fin group_size, (a i).card = handshakes_per_person ∧ i ∉ a i

/-- The number of valid handshaking arrangements -/
def num_arrangements : Nat := 1814400

/-- The theorem to be proved -/
theorem handshaking_theorem :
  (∃ (S : Finset HandshakingArrangement),
    (∀ a ∈ S, is_valid_arrangement a) ∧
    S.card = num_arrangements) ∧
  num_arrangements % 1000 = 400 := by sorry

end NUMINAMATH_CALUDE_handshaking_theorem_l3692_369295


namespace NUMINAMATH_CALUDE_tangent_line_at_origin_l3692_369248

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x

-- Theorem: The equation of the tangent line to y = x^3 - 3x^2 + 1 at (0, 1) is y = 1
theorem tangent_line_at_origin (x : ℝ) : 
  (f' 0) * x + f 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_at_origin_l3692_369248


namespace NUMINAMATH_CALUDE_negative_inequality_l3692_369204

theorem negative_inequality (h : 3.14 < Real.pi) : -3.14 > -Real.pi := by
  sorry

end NUMINAMATH_CALUDE_negative_inequality_l3692_369204


namespace NUMINAMATH_CALUDE_birdseed_mix_problem_l3692_369280

/-- Represents the composition of a birdseed brand -/
structure BirdseedBrand where
  millet : Float
  sunflower : Float

/-- Represents a mix of two birdseed brands -/
structure BirdseedMix where
  brandA : BirdseedBrand
  brandB : BirdseedBrand
  proportionA : Float

theorem birdseed_mix_problem (mixA : BirdseedBrand) (mixB : BirdseedBrand) (mix : BirdseedMix) :
  mixA.millet = 0.4 →
  mixA.sunflower = 0.6 →
  mixB.millet = 0.65 →
  mix.brandA = mixA →
  mix.brandB = mixB →
  mix.proportionA = 0.6 →
  (mix.proportionA * mixA.sunflower + (1 - mix.proportionA) * mixB.sunflower = 0.5) →
  mixB.sunflower = 0.35 := by
  sorry

#check birdseed_mix_problem

end NUMINAMATH_CALUDE_birdseed_mix_problem_l3692_369280


namespace NUMINAMATH_CALUDE_appended_number_divisibility_l3692_369251

theorem appended_number_divisibility : ∃ n : ℕ, 
  27700 ≤ n ∧ n ≤ 27799 ∧ 
  (∀ k : ℕ, 2 ≤ k ∧ k ≤ 12 → n % k = 0) :=
sorry

end NUMINAMATH_CALUDE_appended_number_divisibility_l3692_369251


namespace NUMINAMATH_CALUDE_investment_time_ratio_l3692_369276

/-- Represents the business investment scenario of Krishan and Nandan -/
structure Investment where
  nandan_amount : ℝ
  nandan_time : ℝ
  krishan_time_ratio : ℝ
  nandan_gain : ℝ
  total_gain : ℝ

/-- The conditions of the investment scenario -/
def investment_conditions (i : Investment) : Prop :=
  i.nandan_gain = i.nandan_amount * i.nandan_time ∧
  i.total_gain = i.nandan_gain + 6 * i.nandan_amount * i.krishan_time_ratio * i.nandan_time ∧
  i.nandan_gain = 6000 ∧
  i.total_gain = 78000

/-- The theorem stating that under the given conditions, 
    Krishan's investment time is twice that of Nandan's -/
theorem investment_time_ratio 
  (i : Investment) 
  (h : investment_conditions i) : 
  i.krishan_time_ratio = 2 := by
  sorry

end NUMINAMATH_CALUDE_investment_time_ratio_l3692_369276


namespace NUMINAMATH_CALUDE_two_heads_in_succession_probability_l3692_369239

-- Define a function to count sequences without two heads in succession
def g : ℕ → ℕ
| 0 => 1
| 1 => 2
| n + 2 => g (n + 1) + g n

-- Theorem statement
theorem two_heads_in_succession_probability :
  (1024 - g 10 : ℚ) / 1024 = 55 / 64 := by sorry

end NUMINAMATH_CALUDE_two_heads_in_succession_probability_l3692_369239


namespace NUMINAMATH_CALUDE_triangle_area_l3692_369262

/-- The area of a triangle with vertices at (2, 2), (2, -3), and (7, 2) is 12.5 square units. -/
theorem triangle_area : 
  let A : ℝ × ℝ := (2, 2)
  let B : ℝ × ℝ := (2, -3)
  let C : ℝ × ℝ := (7, 2)
  (1/2 : ℝ) * |A.1 - C.1| * |A.2 - B.2| = 12.5 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l3692_369262


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3692_369246

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 5*x + 3) * (x^2 + 9*x + 20) + (x^2 + 7*x - 8) = 
  (x^2 + 7*x + 8) * (x^2 + 7*x + 14) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3692_369246


namespace NUMINAMATH_CALUDE_odd_function_sum_zero_l3692_369289

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Theorem statement
theorem odd_function_sum_zero (f : ℝ → ℝ) (h : OddFunction f) :
  f (-2) + f 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_sum_zero_l3692_369289


namespace NUMINAMATH_CALUDE_cab_driver_income_l3692_369285

/-- Theorem: Given a cab driver's income for 5 days where 4 days are known and the average income,
    prove that the income for the unknown day is as calculated. -/
theorem cab_driver_income 
  (day1 day2 day3 day5 : ℕ) 
  (average : ℕ) 
  (h1 : day1 = 300)
  (h2 : day2 = 150)
  (h3 : day3 = 750)
  (h5 : day5 = 500)
  (h_avg : average = 420)
  : ∃ day4 : ℕ, 
    day4 = 400 ∧ 
    (day1 + day2 + day3 + day4 + day5) / 5 = average :=
by
  sorry


end NUMINAMATH_CALUDE_cab_driver_income_l3692_369285


namespace NUMINAMATH_CALUDE_total_dots_is_78_l3692_369234

/-- The number of ladybugs Andre caught on Monday -/
def monday_ladybugs : ℕ := 8

/-- The number of ladybugs Andre caught on Tuesday -/
def tuesday_ladybugs : ℕ := 5

/-- The number of dots each ladybug has -/
def dots_per_ladybug : ℕ := 6

/-- The total number of dots on all ladybugs Andre caught -/
def total_dots : ℕ := (monday_ladybugs + tuesday_ladybugs) * dots_per_ladybug

theorem total_dots_is_78 : total_dots = 78 := by
  sorry

end NUMINAMATH_CALUDE_total_dots_is_78_l3692_369234


namespace NUMINAMATH_CALUDE_bridge_length_l3692_369299

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : Real) (train_speed_kmh : Real) (crossing_time_s : Real) :
  train_length = 120 ∧ 
  train_speed_kmh = 45 ∧ 
  crossing_time_s = 30 →
  (train_speed_kmh * 1000 / 3600 * crossing_time_s) - train_length = 255 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l3692_369299


namespace NUMINAMATH_CALUDE_work_time_solution_l3692_369233

def work_time (T : ℝ) : Prop :=
  let A_alone := T + 8
  let B_alone := T + 4.5
  (1 / A_alone) + (1 / B_alone) = 1 / T

theorem work_time_solution : ∃ T : ℝ, work_time T ∧ T = 6 := by sorry

end NUMINAMATH_CALUDE_work_time_solution_l3692_369233


namespace NUMINAMATH_CALUDE_janet_jasmine_shampoo_l3692_369256

/-- The amount of rose shampoo Janet has, in bottles -/
def rose_shampoo : ℚ := 1/3

/-- The amount of shampoo Janet uses per day, in bottles -/
def daily_usage : ℚ := 1/12

/-- The number of days Janet's shampoo will last -/
def days : ℕ := 7

/-- The total amount of shampoo Janet has, in bottles -/
def total_shampoo : ℚ := daily_usage * days

/-- The amount of jasmine shampoo Janet has, in bottles -/
def jasmine_shampoo : ℚ := total_shampoo - rose_shampoo

theorem janet_jasmine_shampoo : jasmine_shampoo = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_janet_jasmine_shampoo_l3692_369256


namespace NUMINAMATH_CALUDE_expression_simplification_l3692_369216

theorem expression_simplification (a : ℝ) (h : a^2 + 2*a - 8 = 0) :
  ((a^2 - 4) / (a^2 - 4*a + 4) - a / (a - 2)) / ((a^2 + 2*a) / (a - 2)) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3692_369216


namespace NUMINAMATH_CALUDE_jack_money_l3692_369238

theorem jack_money (jack ben eric : ℕ) : 
  (eric = ben - 10) → 
  (ben = jack - 9) → 
  (jack + ben + eric = 50) → 
  jack = 26 := by sorry

end NUMINAMATH_CALUDE_jack_money_l3692_369238


namespace NUMINAMATH_CALUDE_complex_fraction_squared_difference_l3692_369232

theorem complex_fraction_squared_difference (a b : ℝ) :
  (Complex.I : ℂ)^2 = -1 →
  (1 - Complex.I) / (1 + Complex.I) = a + b * Complex.I →
  a^2 - b^2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_squared_difference_l3692_369232


namespace NUMINAMATH_CALUDE_parabola_equilateral_triangle_p_value_l3692_369261

/-- Parabola defined by x^2 = 2py where p > 0 -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Circle defined by center and radius -/
structure Circle where
  center : Point
  radius : ℝ

/-- Theorem: For a parabola C: x^2 = 2py (p > 0), if there exists a point A on C
    such that A is equidistant from O(0,0) and M(0,9), and triangle ABO is equilateral
    (where B is another point on the circle with center M and radius |OA|),
    then p = 3/4 -/
theorem parabola_equilateral_triangle_p_value
  (C : Parabola)
  (A : Point)
  (h_A_on_C : A.x^2 = 2 * C.p * A.y)
  (h_A_equidistant : A.x^2 + A.y^2 = A.x^2 + (A.y - 9)^2)
  (h_ABO_equilateral : ∃ B : Point, B.x^2 + (B.y - 9)^2 = A.x^2 + A.y^2 ∧
                       A.x^2 + A.y^2 = (A.x - B.x)^2 + (A.y - B.y)^2) :
  C.p = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_equilateral_triangle_p_value_l3692_369261


namespace NUMINAMATH_CALUDE_range_of_f_l3692_369202

def f (x : ℤ) : ℤ := (x - 1)^2 - 1

def domain : Set ℤ := {-1, 0, 1, 2, 3}

theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {-1, 0, 3} := by
  sorry

end NUMINAMATH_CALUDE_range_of_f_l3692_369202


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_squares_l3692_369218

theorem quadratic_roots_sum_squares (h : ℝ) : 
  (∃ x y : ℝ, x^2 + 2*h*x = 8 ∧ y^2 + 2*h*y = 8 ∧ x^2 + y^2 = 20) → 
  |h| = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_squares_l3692_369218


namespace NUMINAMATH_CALUDE_additional_oil_needed_l3692_369223

/-- Calculates the additional oil needed for a car engine -/
theorem additional_oil_needed
  (oil_per_cylinder : ℕ)
  (num_cylinders : ℕ)
  (oil_already_added : ℕ)
  (h1 : oil_per_cylinder = 8)
  (h2 : num_cylinders = 6)
  (h3 : oil_already_added = 16) :
  oil_per_cylinder * num_cylinders - oil_already_added = 32 :=
by sorry

end NUMINAMATH_CALUDE_additional_oil_needed_l3692_369223


namespace NUMINAMATH_CALUDE_trig_identity_proof_l3692_369288

theorem trig_identity_proof : 
  (Real.cos (63 * π / 180) * Real.cos (3 * π / 180) - 
   Real.cos (87 * π / 180) * Real.cos (27 * π / 180)) / 
  (Real.cos (132 * π / 180) * Real.cos (72 * π / 180) - 
   Real.cos (42 * π / 180) * Real.cos (18 * π / 180)) = 
  -Real.tan (24 * π / 180) := by
sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l3692_369288


namespace NUMINAMATH_CALUDE_exists_prime_divisor_l3692_369278

-- Define the sequence a_n
def a (c : ℕ+) : ℕ → ℕ
  | 0 => c
  | n + 1 => (a c n)^3 - 4 * c * (a c n)^2 + 5 * c^2 * (a c n) + c

-- State the theorem
theorem exists_prime_divisor (c : ℕ+) (n : ℕ) (hn : n ≥ 2) :
  ∃ p : ℕ, Prime p ∧ p ∣ a c (n - 1) ∧ ∀ k : ℕ, k < n - 1 → ¬(p ∣ a c k) := by
  sorry

end NUMINAMATH_CALUDE_exists_prime_divisor_l3692_369278


namespace NUMINAMATH_CALUDE_water_jars_problem_l3692_369272

/-- Proves that given 28 gallons of water stored in equal numbers of quart, half-gallon, and one-gallon jars, the total number of water-filled jars is 48. -/
theorem water_jars_problem (total_water : ℚ) (num_each_jar : ℕ) : 
  total_water = 28 →
  (1/4 : ℚ) * num_each_jar + (1/2 : ℚ) * num_each_jar + 1 * num_each_jar = total_water →
  3 * num_each_jar = 48 := by
  sorry

end NUMINAMATH_CALUDE_water_jars_problem_l3692_369272


namespace NUMINAMATH_CALUDE_earth_pile_fraction_l3692_369264

theorem earth_pile_fraction (P : ℚ) (P_pos : P > 0) : 
  P * (1 - 1/2) * (1 - 1/3) * (1 - 1/4) * (1 - 1/5) * (1 - 1/6) * (1 - 1/7) = P * (1/7) := by
  sorry

end NUMINAMATH_CALUDE_earth_pile_fraction_l3692_369264


namespace NUMINAMATH_CALUDE_min_value_quadratic_sum_l3692_369297

theorem min_value_quadratic_sum (x y z : ℝ) (h : x - 2*y + 2*z = 5) :
  ∃ (m : ℝ), m = 36 ∧ ∀ (a b c : ℝ), a - 2*b + 2*c = 5 → (a + 5)^2 + (b - 1)^2 + (c + 3)^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_sum_l3692_369297


namespace NUMINAMATH_CALUDE_sum_in_base7_l3692_369294

/-- Converts a number from base 7 to base 10 -/
def toBase10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * 7^i) 0

/-- Converts a number from base 10 to base 7 -/
def toBase7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else aux (m / 7) ((m % 7) :: acc)
    aux n []

/-- Theorem: The sum of 666₇, 66₇, and 6₇ in base 7 is equal to 104₇ -/
theorem sum_in_base7 :
  toBase7 (toBase10 [6, 6, 6] + toBase10 [6, 6] + toBase10 [6]) = [1, 0, 4] :=
sorry

end NUMINAMATH_CALUDE_sum_in_base7_l3692_369294


namespace NUMINAMATH_CALUDE_inequality_proof_l3692_369257

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x * y + y * z + z * x = 6) : 
  1 / (2 * Real.sqrt 2 + x^2 * (y + z)) + 
  1 / (2 * Real.sqrt 2 + y^2 * (x + z)) + 
  1 / (2 * Real.sqrt 2 + z^2 * (x + y)) ≤ 
  1 / (x * y * z) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3692_369257


namespace NUMINAMATH_CALUDE_yellas_computer_usage_l3692_369268

theorem yellas_computer_usage (last_week_hours : ℕ) (reduction : ℕ) : 
  last_week_hours = 91 → 
  reduction = 35 → 
  (last_week_hours - reduction) / 7 = 8 := by
sorry

end NUMINAMATH_CALUDE_yellas_computer_usage_l3692_369268


namespace NUMINAMATH_CALUDE_certain_number_proof_l3692_369213

theorem certain_number_proof (a b x : ℝ) (h1 : x * a = 3 * b) (h2 : a * b ≠ 0) (h3 : a / 3 = b / 2) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3692_369213


namespace NUMINAMATH_CALUDE_jacket_pricing_l3692_369287

theorem jacket_pricing (x : ℝ) : x + 28 = 0.8 * (1.5 * x) → 
  (∃ (markup_percent : ℝ) (discount_percent : ℝ) (profit : ℝ),
    markup_percent = 0.5 ∧
    discount_percent = 0.2 ∧
    profit = 28 ∧
    x + profit = (1 - discount_percent) * (1 + markup_percent) * x) :=
by
  sorry

end NUMINAMATH_CALUDE_jacket_pricing_l3692_369287


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l3692_369200

/-- Definition of the diamond operation -/
def diamond (A B : ℝ) : ℝ := 4 * A + 3 * B + 7

/-- Theorem stating the solution to the equation A ◇ 5 = 71 -/
theorem diamond_equation_solution :
  ∃ A : ℝ, diamond A 5 = 71 ∧ A = 12.25 := by
  sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l3692_369200


namespace NUMINAMATH_CALUDE_solution_set_range_no_k_exists_positive_roots_k_range_l3692_369243

/-- The quadratic function y(x) = kx² - 2kx + 2k - 1 -/
def y (k x : ℝ) : ℝ := k * x^2 - 2 * k * x + 2 * k - 1

/-- The solution set of y ≥ 4k - 2 is all real numbers iff k ∈ [0, 1/3] -/
theorem solution_set_range (k : ℝ) :
  (∀ x, y k x ≥ 4 * k - 2) ↔ k ∈ Set.Icc 0 (1/3) := by sorry

/-- No k ∈ (0, 1) satisfies x₁² + x₂² = 3x₁x₂ - 4 for roots of y(x) = 0 -/
theorem no_k_exists (k : ℝ) (hk : k ∈ Set.Ioo 0 1) :
  ¬∃ x₁ x₂ : ℝ, y k x₁ = 0 ∧ y k x₂ = 0 ∧ x₁ ≠ x₂ ∧ x₁^2 + x₂^2 = 3*x₁*x₂ - 4 := by sorry

/-- If roots of y(x) = 0 are positive, then k ∈ (1/2, 1) -/
theorem positive_roots_k_range (k : ℝ) :
  (∃ x₁ x₂ : ℝ, y k x₁ = 0 ∧ y k x₂ = 0 ∧ x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0) →
  k ∈ Set.Ioo (1/2) 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_range_no_k_exists_positive_roots_k_range_l3692_369243


namespace NUMINAMATH_CALUDE_f_4_solutions_l3692_369236

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x

-- Define the composite function f^4
def f_4 (x : ℝ) : ℝ := f (f (f (f x)))

-- Theorem statement
theorem f_4_solutions :
  ∃! (s : Finset ℝ), (∀ c ∈ s, f_4 c = 3) ∧ s.card = 3 :=
sorry

end NUMINAMATH_CALUDE_f_4_solutions_l3692_369236


namespace NUMINAMATH_CALUDE_isotopes_same_count_atom_molecule_same_count_molecules_same_count_cations_same_count_anions_same_count_different_elements_different_count_atom_ion_different_count_molecule_ion_different_count_anion_cation_different_count_l3692_369274

-- Define the basic types
inductive ParticleType
| Atom
| Molecule
| Cation
| Anion

-- Define a particle
structure Particle where
  type : ParticleType
  protons : ℕ
  electrons : ℕ

-- Define the property of having the same number of protons and electrons
def sameProtonElectronCount (p1 p2 : Particle) : Prop :=
  p1.protons = p2.protons ∧ p1.electrons = p2.electrons

-- Theorem: Two different atoms (isotopes) can have the same number of protons and electrons
theorem isotopes_same_count :
  ∃ (p1 p2 : Particle), p1.type = ParticleType.Atom ∧ p2.type = ParticleType.Atom ∧
  p1 ≠ p2 ∧ sameProtonElectronCount p1 p2 :=
sorry

-- Theorem: An atom and a molecule can have the same number of protons and electrons
theorem atom_molecule_same_count :
  ∃ (p1 p2 : Particle), p1.type = ParticleType.Atom ∧ p2.type = ParticleType.Molecule ∧
  sameProtonElectronCount p1 p2 :=
sorry

-- Theorem: Two different molecules can have the same number of protons and electrons
theorem molecules_same_count :
  ∃ (p1 p2 : Particle), p1.type = ParticleType.Molecule ∧ p2.type = ParticleType.Molecule ∧
  p1 ≠ p2 ∧ sameProtonElectronCount p1 p2 :=
sorry

-- Theorem: Two different cations can have the same number of protons and electrons
theorem cations_same_count :
  ∃ (p1 p2 : Particle), p1.type = ParticleType.Cation ∧ p2.type = ParticleType.Cation ∧
  p1 ≠ p2 ∧ sameProtonElectronCount p1 p2 :=
sorry

-- Theorem: Two different anions can have the same number of protons and electrons
theorem anions_same_count :
  ∃ (p1 p2 : Particle), p1.type = ParticleType.Anion ∧ p2.type = ParticleType.Anion ∧
  p1 ≠ p2 ∧ sameProtonElectronCount p1 p2 :=
sorry

-- Theorem: Atoms of two different elements cannot have the same number of protons and electrons
theorem different_elements_different_count :
  ∀ (p1 p2 : Particle), p1.type = ParticleType.Atom ∧ p2.type = ParticleType.Atom ∧
  p1.protons ≠ p2.protons → ¬(sameProtonElectronCount p1 p2) :=
sorry

-- Theorem: An atom and an ion cannot have the same number of protons and electrons
theorem atom_ion_different_count :
  ∀ (p1 p2 : Particle), p1.type = ParticleType.Atom ∧ (p2.type = ParticleType.Cation ∨ p2.type = ParticleType.Anion) →
  ¬(sameProtonElectronCount p1 p2) :=
sorry

-- Theorem: A molecule and an ion cannot have the same number of protons and electrons
theorem molecule_ion_different_count :
  ∀ (p1 p2 : Particle), p1.type = ParticleType.Molecule ∧ (p2.type = ParticleType.Cation ∨ p2.type = ParticleType.Anion) →
  ¬(sameProtonElectronCount p1 p2) :=
sorry

-- Theorem: An anion and a cation cannot have the same number of protons and electrons
theorem anion_cation_different_count :
  ∀ (p1 p2 : Particle), p1.type = ParticleType.Anion ∧ p2.type = ParticleType.Cation →
  ¬(sameProtonElectronCount p1 p2) :=
sorry

end NUMINAMATH_CALUDE_isotopes_same_count_atom_molecule_same_count_molecules_same_count_cations_same_count_anions_same_count_different_elements_different_count_atom_ion_different_count_molecule_ion_different_count_anion_cation_different_count_l3692_369274


namespace NUMINAMATH_CALUDE_scientists_born_in_july_percentage_l3692_369215

theorem scientists_born_in_july_percentage :
  let total_scientists : ℕ := 120
  let born_in_july : ℕ := 20
  let percentage : ℚ := (born_in_july : ℚ) / total_scientists * 100
  percentage = 50 / 3 := by sorry

end NUMINAMATH_CALUDE_scientists_born_in_july_percentage_l3692_369215


namespace NUMINAMATH_CALUDE_remove_one_gives_avg_seven_point_five_l3692_369275

def original_sequence : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

def remove_element (lst : List Nat) (n : Nat) : List Nat :=
  lst.filter (· ≠ n)

def average (lst : List Nat) : Rat :=
  (lst.sum : Rat) / lst.length

theorem remove_one_gives_avg_seven_point_five :
  average (remove_element original_sequence 1) = 15/2 := by
  sorry

end NUMINAMATH_CALUDE_remove_one_gives_avg_seven_point_five_l3692_369275


namespace NUMINAMATH_CALUDE_total_thumbtacks_l3692_369282

/-- The number of boards tested -/
def boards_tested : ℕ := 120

/-- The number of thumbtacks used per board -/
def tacks_per_board : ℕ := 3

/-- The number of thumbtacks remaining in each can after testing -/
def tacks_remaining_per_can : ℕ := 30

/-- The number of cans used -/
def num_cans : ℕ := 3

/-- Theorem stating that the total number of thumbtacks in three full cans is 450 -/
theorem total_thumbtacks :
  boards_tested * tacks_per_board + num_cans * tacks_remaining_per_can = 450 :=
by sorry

end NUMINAMATH_CALUDE_total_thumbtacks_l3692_369282


namespace NUMINAMATH_CALUDE_gcd_factorial_problem_l3692_369220

theorem gcd_factorial_problem : Nat.gcd (Nat.factorial 7) ((Nat.factorial 9) / (Nat.factorial 4)) = 2520 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_problem_l3692_369220


namespace NUMINAMATH_CALUDE_average_square_ge_product_l3692_369255

theorem average_square_ge_product (a b : ℝ) : (a^2 + b^2) / 2 ≥ a * b := by
  sorry

end NUMINAMATH_CALUDE_average_square_ge_product_l3692_369255


namespace NUMINAMATH_CALUDE_queen_diamond_probability_l3692_369273

/-- Represents a standard deck of 52 playing cards -/
def Deck : Type := Unit

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of Queens in a standard deck -/
def num_queens : ℕ := 4

/-- The number of diamonds in a standard deck -/
def num_diamonds : ℕ := 13

/-- Represents the event of drawing a Queen as the first card and a diamond as the second card -/
def queen_then_diamond (d : Deck) : Prop := sorry

/-- The probability of the queen_then_diamond event -/
def prob_queen_then_diamond (d : Deck) : ℚ := sorry

theorem queen_diamond_probability (d : Deck) : 
  prob_queen_then_diamond d = 1 / deck_size := by sorry

end NUMINAMATH_CALUDE_queen_diamond_probability_l3692_369273


namespace NUMINAMATH_CALUDE_correct_stratified_sample_l3692_369281

/-- Represents a stratified sample from a high school population -/
structure StratifiedSample where
  total_students : ℕ
  freshmen : ℕ
  sophomores : ℕ
  juniors : ℕ
  sample_size : ℕ
  sampled_freshmen : ℕ
  sampled_sophomores : ℕ
  sampled_juniors : ℕ

/-- Checks if a stratified sample is valid according to the problem conditions -/
def is_valid_sample (s : StratifiedSample) : Prop :=
  s.total_students = 2000 ∧
  s.freshmen = 800 ∧
  s.sophomores = 600 ∧
  s.juniors = 600 ∧
  s.sample_size = 50 ∧
  s.sampled_freshmen + s.sampled_sophomores + s.sampled_juniors = s.sample_size

/-- Theorem stating that the correct stratified sample is 20 freshmen, 15 sophomores, and 15 juniors -/
theorem correct_stratified_sample (s : StratifiedSample) :
  is_valid_sample s →
  s.sampled_freshmen = 20 ∧ s.sampled_sophomores = 15 ∧ s.sampled_juniors = 15 := by
  sorry


end NUMINAMATH_CALUDE_correct_stratified_sample_l3692_369281


namespace NUMINAMATH_CALUDE_continuity_at_4_l3692_369266

def f (x : ℝ) : ℝ := -2 * x^2 + 9

theorem continuity_at_4 :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 4| < δ → |f x - f 4| < ε :=
by sorry

end NUMINAMATH_CALUDE_continuity_at_4_l3692_369266


namespace NUMINAMATH_CALUDE_base4_addition_subtraction_l3692_369247

/-- Converts a base 4 number to base 10 --/
def base4ToBase10 (a b c : ℕ) : ℕ := a * 4^2 + b * 4^1 + c * 4^0

/-- Converts a base 10 number to base 4 --/
def base10ToBase4 (n : ℕ) : ℕ × ℕ × ℕ × ℕ :=
  let d := n / 64
  let r := n % 64
  let c := r / 16
  let r' := r % 16
  let b := r' / 4
  let a := r' % 4
  (d, c, b, a)

theorem base4_addition_subtraction :
  let x := base4ToBase10 2 0 3
  let y := base4ToBase10 3 2 1
  let z := base4ToBase10 1 1 2
  base10ToBase4 (x + y - z) = (1, 0, 1, 2) := by sorry

end NUMINAMATH_CALUDE_base4_addition_subtraction_l3692_369247


namespace NUMINAMATH_CALUDE_tan_alpha_point_one_two_l3692_369296

/-- For an angle α whose terminal side passes through the point (1,2), tan α = 2 -/
theorem tan_alpha_point_one_two (α : Real) :
  (∃ (P : ℝ × ℝ), P.1 = 1 ∧ P.2 = 2 ∧ (∃ (r : ℝ), r > 0 ∧ P = (r * Real.cos α, r * Real.sin α))) →
  Real.tan α = 2 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_point_one_two_l3692_369296


namespace NUMINAMATH_CALUDE_min_tablets_extracted_l3692_369277

theorem min_tablets_extracted (total_A : ℕ) (total_B : ℕ) : 
  total_A = 10 → total_B = 10 → 
  ∃ (min_extracted : ℕ), 
    (∀ (n : ℕ), n < min_extracted → 
      ∃ (a b : ℕ), a + b = n ∧ (a < 2 ∨ b < 2)) ∧
    (∀ (a b : ℕ), a + b = min_extracted → a ≥ 2 ∧ b ≥ 2) ∧
    min_extracted = 12 :=
by sorry

end NUMINAMATH_CALUDE_min_tablets_extracted_l3692_369277


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l3692_369231

def geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) : ℚ := a * r ^ (n - 1)

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ := a * (1 - r^n) / (1 - r)

theorem geometric_sequence_properties :
  let a : ℚ := 1/5
  let r : ℚ := 1/2
  let n : ℕ := 8
  (geometric_sequence a r n = 1/640) ∧
  (geometric_sum a r n = 255/320) := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l3692_369231


namespace NUMINAMATH_CALUDE_value_of_x_l3692_369279

theorem value_of_x (x y z : ℝ) : x = 3 * y ∧ y = z / 3 ∧ z = 90 → x = 90 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l3692_369279


namespace NUMINAMATH_CALUDE_positive_reals_inequality_arithmetic_geometric_mean_inequality_l3692_369212

theorem positive_reals_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 := by
  sorry

theorem arithmetic_geometric_mean_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt ((a^2 + b^2 + c^2) / 3) ≥ (a + b + c) / 3 := by
  sorry

end NUMINAMATH_CALUDE_positive_reals_inequality_arithmetic_geometric_mean_inequality_l3692_369212


namespace NUMINAMATH_CALUDE_problem_solution_l3692_369203

theorem problem_solution (x y z w : ℕ+) 
  (h1 : x^3 = y^2) 
  (h2 : z^5 = w^4) 
  (h3 : z - x = 31) : 
  (w : ℤ) - y = -2351 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3692_369203


namespace NUMINAMATH_CALUDE_new_persons_joined_l3692_369252

/-- Proves that 20 new persons joined the group given the initial conditions and final average age -/
theorem new_persons_joined (initial_avg : ℝ) (new_avg : ℝ) (final_avg : ℝ) (initial_count : ℕ) : 
  initial_avg = 16 → new_avg = 15 → final_avg = 15.5 → initial_count = 20 → 
  ∃ (new_count : ℕ), 
    (initial_count * initial_avg + new_count * new_avg) / (initial_count + new_count) = final_avg ∧
    new_count = 20 := by
  sorry

end NUMINAMATH_CALUDE_new_persons_joined_l3692_369252


namespace NUMINAMATH_CALUDE_equation_solution_l3692_369219

theorem equation_solution :
  ∀ y : ℝ, (5 + 3.2 * y = 2.1 * y - 25) ↔ (y = -300 / 11) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3692_369219


namespace NUMINAMATH_CALUDE_quadratic_inequality_proof_l3692_369211

theorem quadratic_inequality_proof (x : ℝ) : 
  x^2 + 6*x + 8 ≥ -(x + 4)*(x + 6) ∧ 
  (x^2 + 6*x + 8 = -(x + 4)*(x + 6) ↔ x = -4) := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_proof_l3692_369211


namespace NUMINAMATH_CALUDE_cauliflower_increase_l3692_369224

/-- Represents a square garden for growing cauliflowers -/
structure CauliflowerGarden where
  side : ℕ

/-- Calculates the number of cauliflowers in a garden -/
def cauliflowers (garden : CauliflowerGarden) : ℕ := garden.side * garden.side

/-- Theorem: If a square garden's cauliflower output increases by 401 while
    maintaining a square shape, the new total is 40,401 cauliflowers -/
theorem cauliflower_increase (old_garden new_garden : CauliflowerGarden) :
  cauliflowers new_garden - cauliflowers old_garden = 401 →
  cauliflowers new_garden = 40401 := by
  sorry


end NUMINAMATH_CALUDE_cauliflower_increase_l3692_369224


namespace NUMINAMATH_CALUDE_binary_sum_theorem_l3692_369253

/-- Converts a binary number (represented as a list of 0s and 1s) to decimal -/
def binary_to_decimal (binary : List Nat) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + b * 2^i) 0

theorem binary_sum_theorem :
  let binary1 := [1, 0, 1, 1, 0, 0, 1]
  let binary2 := [0, 0, 0, 1, 1, 1]
  let binary3 := [0, 1, 0, 1]
  (binary_to_decimal binary1) + (binary_to_decimal binary2) + (binary_to_decimal binary3) = 143 := by
  sorry

end NUMINAMATH_CALUDE_binary_sum_theorem_l3692_369253


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l3692_369284

theorem inequality_system_solution_set :
  ∀ x : ℝ, (x + 2 > 3 * (1 - x) ∧ 1 - 2 * x ≤ 2) ↔ x > (1 : ℝ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l3692_369284


namespace NUMINAMATH_CALUDE_probability_different_tens_digits_l3692_369208

/-- The number of integers in the range 10 to 79, inclusive. -/
def total_integers : ℕ := 70

/-- The number of different tens digits in the range 10 to 79. -/
def different_tens_digits : ℕ := 7

/-- The number of integers to be chosen. -/
def chosen_integers : ℕ := 6

/-- The number of integers for each tens digit. -/
def integers_per_tens : ℕ := 10

theorem probability_different_tens_digits :
  (different_tens_digits.choose chosen_integers * integers_per_tens ^ chosen_integers : ℚ) /
  (total_integers.choose chosen_integers) = 1750 / 2980131 := by sorry

end NUMINAMATH_CALUDE_probability_different_tens_digits_l3692_369208


namespace NUMINAMATH_CALUDE_equation_equivalence_l3692_369230

theorem equation_equivalence (x : ℝ) : (5 = 3 * x - 2) ↔ (5 + 2 = 3 * x) := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l3692_369230


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3692_369205

-- Define sets A and B
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {3, 4, 5}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3692_369205


namespace NUMINAMATH_CALUDE_intersection_A_B_l3692_369269

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -2 < x ∧ x < 1}
def B : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x : ℝ | 0 < x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3692_369269


namespace NUMINAMATH_CALUDE_complement_of_union_l3692_369229

def I : Finset Int := {-2, -1, 0, 1, 2, 3, 4, 5}
def A : Finset Int := {-1, 0, 1, 2, 3}
def B : Finset Int := {-2, 0, 2}

theorem complement_of_union :
  (I \ (A ∪ B)) = {4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l3692_369229


namespace NUMINAMATH_CALUDE_product_of_fractions_l3692_369283

theorem product_of_fractions : (2 : ℚ) / 3 * 5 / 8 * 1 / 4 = 5 / 48 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l3692_369283


namespace NUMINAMATH_CALUDE_nonagon_diagonal_count_l3692_369260

/-- The number of diagonals in a regular nonagon -/
def nonagon_diagonals : ℕ := 27

/-- A regular nonagon has 9 sides -/
def nonagon_sides : ℕ := 9

/-- The number of vertices in a regular nonagon -/
def nonagon_vertices : ℕ := 9

theorem nonagon_diagonal_count :
  nonagon_diagonals = (nonagon_vertices.choose 2) - nonagon_sides := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonal_count_l3692_369260


namespace NUMINAMATH_CALUDE_highest_level_backers_count_l3692_369254

/-- Represents the crowdfunding scenario for an entrepreneur --/
structure CrowdfundingScenario where
  highest_level : ℕ
  second_level : ℕ
  lowest_level : ℕ
  target_amount : ℕ
  second_level_backers : ℕ
  lowest_level_backers : ℕ
  
/-- Defines the crowdfunding scenario based on the given problem --/
def entrepreneur_scenario : CrowdfundingScenario :=
  { highest_level := 5000
  , second_level := 500
  , lowest_level := 50
  , target_amount := 12000
  , second_level_backers := 3
  , lowest_level_backers := 10 }

/-- Theorem stating that the number of highest level backers is 2 --/
theorem highest_level_backers_count (scenario : CrowdfundingScenario := entrepreneur_scenario) :
  ∃ (x : ℕ), 
    scenario.highest_level * x + 
    scenario.second_level * scenario.second_level_backers + 
    scenario.lowest_level * scenario.lowest_level_backers = 
    scenario.target_amount ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_highest_level_backers_count_l3692_369254


namespace NUMINAMATH_CALUDE_greatest_multiple_of_30_under_1000_l3692_369214

theorem greatest_multiple_of_30_under_1000 : 
  ∀ n : ℕ, n * 30 < 1000 → n * 30 ≤ 990 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_30_under_1000_l3692_369214


namespace NUMINAMATH_CALUDE_corn_acreage_l3692_369222

theorem corn_acreage (total_land : ℕ) (bean_ratio wheat_ratio corn_ratio : ℕ) 
  (h1 : total_land = 1034)
  (h2 : bean_ratio = 5)
  (h3 : wheat_ratio = 2)
  (h4 : corn_ratio = 4) :
  (total_land * corn_ratio) / (bean_ratio + wheat_ratio + corn_ratio) = 376 := by
  sorry

end NUMINAMATH_CALUDE_corn_acreage_l3692_369222


namespace NUMINAMATH_CALUDE_range_of_independent_variable_l3692_369221

theorem range_of_independent_variable (x : ℝ) :
  (∃ y : ℝ, y = 1 / Real.sqrt (2 - 3 * x)) ↔ x < 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_range_of_independent_variable_l3692_369221


namespace NUMINAMATH_CALUDE_coefficient_of_x5y2_l3692_369292

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the polynomial (x^2 + 3x - y)^5
def polynomial (x y : ℤ) : ℤ := (x^2 + 3*x - y)^5

-- Theorem statement
theorem coefficient_of_x5y2 :
  ∃ (coeff : ℤ), coeff = 90 ∧
  ∀ (x y : ℤ), 
    ∃ (rest : ℤ), 
      polynomial x y = coeff * x^5 * y^2 + rest ∧ 
      (∀ (a b : ℕ), a ≤ 5 ∧ b ≤ 2 ∧ (a, b) ≠ (5, 2) → 
        ∃ (other_terms : ℤ), rest = other_terms * x^a * y^b + other_terms) :=
sorry

end NUMINAMATH_CALUDE_coefficient_of_x5y2_l3692_369292


namespace NUMINAMATH_CALUDE_best_approximation_log5_10_l3692_369241

/-- Approximation of log₁₀2 -/
def log10_2 : ℝ := 0.301

/-- Approximation of log₁₀3 -/
def log10_3 : ℝ := 0.477

/-- The set of possible fractions for approximating log₅10 -/
def fraction_options : List ℚ := [8/7, 9/7, 10/7, 11/7, 12/7]

/-- Statement: The fraction 10/7 is the closest approximation to log₅10 among the given options -/
theorem best_approximation_log5_10 : 
  ∃ (x : ℚ), x ∈ fraction_options ∧ 
  ∀ (y : ℚ), y ∈ fraction_options → |x - (1 / (1 - log10_2))| ≤ |y - (1 / (1 - log10_2))| ∧
  x = 10/7 := by
  sorry

end NUMINAMATH_CALUDE_best_approximation_log5_10_l3692_369241


namespace NUMINAMATH_CALUDE_cube_surface_area_equal_volume_cylinder_l3692_369217

/-- The surface area of a cube with volume equal to a cylinder of radius 4 and height 12 -/
theorem cube_surface_area_equal_volume_cylinder (π : ℝ) :
  let cylinder_volume := π * 4^2 * 12
  let cube_edge := (cylinder_volume)^(1/3)
  let cube_surface_area := 6 * cube_edge^2
  cube_surface_area = 6 * (192 * π)^(2/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_equal_volume_cylinder_l3692_369217


namespace NUMINAMATH_CALUDE_complex_addition_l3692_369267

theorem complex_addition (z₁ z₂ : ℂ) (h₁ : z₁ = 1 + 2*I) (h₂ : z₂ = 3 + 4*I) : 
  z₁ + z₂ = 4 + 6*I := by
sorry

end NUMINAMATH_CALUDE_complex_addition_l3692_369267


namespace NUMINAMATH_CALUDE_power_of_product_exponent_l3692_369228

theorem power_of_product_exponent (a b : ℝ) : (a^2 * b^3)^2 = a^4 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_exponent_l3692_369228


namespace NUMINAMATH_CALUDE_parabola_properties_l3692_369291

-- Define the parabola
def parabola (b c x : ℝ) : ℝ := -x^2 + b*x + c

-- Define the roots of the parabola
def roots (b c : ℝ) : Set ℝ := {x | parabola b c x = 0}

-- Theorem statement
theorem parabola_properties :
  ∀ b c : ℝ,
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ + x₂ = 4 ∧ {x₁, x₂} ⊆ roots b c) →
  (b = 4 ∧ c > -4) ∧
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ {x₁, x₂} ⊆ roots b c ∧ |x₁ - x₂| = 2 → c = -3) ∧
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ {x₁, x₂} ⊆ roots b c ∧
    (c = (1 + Real.sqrt 17) / 2 ∨ c = (1 - Real.sqrt 17) / 2) ∧
    |c| = c - parabola b c 2) :=
sorry

end NUMINAMATH_CALUDE_parabola_properties_l3692_369291


namespace NUMINAMATH_CALUDE_factorial_10_mod_13_l3692_369270

/-- Definition of factorial for positive integers -/
def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

/-- The remainder when 10! is divided by 13 is 7 -/
theorem factorial_10_mod_13 : factorial 10 % 13 = 7 := by
  sorry

end NUMINAMATH_CALUDE_factorial_10_mod_13_l3692_369270


namespace NUMINAMATH_CALUDE_negative_fraction_range_l3692_369210

theorem negative_fraction_range (x : ℝ) : (x - 1) / x^2 < 0 → x < 1 ∧ x ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_range_l3692_369210
