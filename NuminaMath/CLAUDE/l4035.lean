import Mathlib

namespace NUMINAMATH_CALUDE_a_range_l4035_403597

/-- The function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then a^x else x^2 + 4/x + a * Real.log x

/-- The theorem statement -/
theorem a_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : ∀ x y : ℝ, x < y → f a x < f a y) : 
  2 ≤ a ∧ a ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_a_range_l4035_403597


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_abs_sum_l4035_403515

theorem sqrt_sum_equals_abs_sum (x : ℝ) : 
  Real.sqrt (x^2 - 4*x + 4) + Real.sqrt (x^2 + 6*x + 9) = |x - 2| + |x + 3| := by
sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_abs_sum_l4035_403515


namespace NUMINAMATH_CALUDE_expected_turns_to_second_ace_prove_expected_turns_l4035_403500

/-- A deck of cards -/
structure Deck :=
  (n : ℕ)  -- Total number of cards
  (h : n ≥ 3)  -- There are at least 3 cards (for the 3 aces)

/-- The expected number of cards turned up until the second ace appears -/
def expectedTurnsToSecondAce (d : Deck) : ℚ :=
  (d.n + 1) / 2

/-- Theorem stating that the expected number of cards turned up until the second ace appears is (n+1)/2 -/
theorem expected_turns_to_second_ace (d : Deck) :
  expectedTurnsToSecondAce d = (d.n + 1) / 2 := by
  sorry

/-- Main theorem proving the expected number of cards turned up -/
theorem prove_expected_turns (d : Deck) :
  ∃ (e : ℚ), e = expectedTurnsToSecondAce d ∧ e = (d.n + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_expected_turns_to_second_ace_prove_expected_turns_l4035_403500


namespace NUMINAMATH_CALUDE_perimeter_is_20_l4035_403595

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 16 = 1

-- Define the foci
def left_focus : ℝ × ℝ := sorry
def right_focus : ℝ × ℝ := sorry

-- Define a line passing through the right focus
def line_through_right_focus (x y : ℝ) : Prop := sorry

-- Define points A and B on the ellipse and the line
def point_A : ℝ × ℝ := sorry
def point_B : ℝ × ℝ := sorry

-- Assumption that A and B are on the ellipse and the line
axiom A_on_ellipse : ellipse point_A.1 point_A.2
axiom B_on_ellipse : ellipse point_B.1 point_B.2
axiom A_on_line : line_through_right_focus point_A.1 point_A.2
axiom B_on_line : line_through_right_focus point_B.1 point_B.2

-- Define the perimeter of triangle F₁AB
def perimeter_F1AB : ℝ := sorry

-- Theorem statement
theorem perimeter_is_20 : perimeter_F1AB = 20 := by sorry

end NUMINAMATH_CALUDE_perimeter_is_20_l4035_403595


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l4035_403590

/-- Given a geometric sequence {aₙ} where a₁ + a₂ = 3 and a₂ + a₃ = 6, prove that a₃ = 4 -/
theorem geometric_sequence_third_term
  (a : ℕ → ℝ)  -- a is a sequence of real numbers
  (h_geom : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1))  -- a is a geometric sequence
  (h_sum1 : a 1 + a 2 = 3)  -- first condition
  (h_sum2 : a 2 + a 3 = 6)  -- second condition
  : a 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l4035_403590


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l4035_403526

theorem complex_fraction_equality : ∃ (i : ℂ), i^2 = -1 ∧ i^3 / (1 + i) = -1/2 - 1/2 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l4035_403526


namespace NUMINAMATH_CALUDE_obtuse_triangle_area_l4035_403524

theorem obtuse_triangle_area (a b : ℝ) (C : ℝ) (h1 : a = 8) (h2 : b = 12) (h3 : C = 150 * π / 180) :
  let area := (1/2) * a * b * Real.sin C
  area = 24 := by
sorry

end NUMINAMATH_CALUDE_obtuse_triangle_area_l4035_403524


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l4035_403547

theorem polynomial_evaluation (x : ℝ) (h : x = 2) : 3 * x^2 + 5 * x - 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l4035_403547


namespace NUMINAMATH_CALUDE_faster_train_speed_l4035_403550

/-- Calculates the speed of the faster train given the conditions of the problem -/
theorem faster_train_speed
  (train_length : ℝ)
  (slower_speed : ℝ)
  (overtake_time : ℝ)
  (h1 : train_length = 50)
  (h2 : slower_speed = 36)
  (h3 : overtake_time = 36)
  : ∃ (faster_speed : ℝ), faster_speed = 46 :=
by
  sorry

#check faster_train_speed

end NUMINAMATH_CALUDE_faster_train_speed_l4035_403550


namespace NUMINAMATH_CALUDE_perpendicular_bisector_and_parallel_line_l4035_403553

/-- Given two points A and B in the plane, this theorem proves:
    1. The equation of the perpendicular bisector of AB
    2. The equation of a line passing through P and parallel to AB -/
theorem perpendicular_bisector_and_parallel_line 
  (A B P : ℝ × ℝ) 
  (hA : A = (8, -6)) 
  (hB : B = (2, 2)) 
  (hP : P = (2, -3)) : 
  (∃ (a b c : ℝ), a * 3 = b * 4 ∧ c = 23 ∧ 
    (∀ (x y : ℝ), (a * x + b * y + c = 0) ↔ 
      (x - (A.1 + B.1) / 2)^2 + (y - (A.2 + B.2) / 2)^2 = 
      ((A.1 - B.1)^2 + (A.2 - B.2)^2) / 4)) ∧
  (∃ (d e f : ℝ), d * 4 = -e * 3 ∧ f = 1 ∧
    (∀ (x y : ℝ), (d * x + e * y + f = 0) ↔ 
      (y - P.2) = ((B.2 - A.2) / (B.1 - A.1)) * (x - P.1))) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_and_parallel_line_l4035_403553


namespace NUMINAMATH_CALUDE_unpainted_side_length_approx_l4035_403579

/-- A rectangular parking space with three painted sides -/
structure ParkingSpace where
  length : ℝ
  width : ℝ
  painted_sides_sum : length + 2 * width = 37
  area : length * width = 125

/-- The length of the unpainted side of the parking space -/
def unpainted_side_length (p : ParkingSpace) : ℝ := p.length

/-- The unpainted side length is approximately 8.90 feet -/
theorem unpainted_side_length_approx (p : ParkingSpace) :
  ∃ ε > 0, |unpainted_side_length p - 8.90| < ε :=
sorry

end NUMINAMATH_CALUDE_unpainted_side_length_approx_l4035_403579


namespace NUMINAMATH_CALUDE_log_equation_implies_c_eq_a_to_three_halves_l4035_403561

/-- Given the equation relating logarithms of x with bases c and a, prove that c = a^(3/2) -/
theorem log_equation_implies_c_eq_a_to_three_halves
  (a c x : ℝ)
  (h_pos_a : 0 < a)
  (h_pos_c : 0 < c)
  (h_pos_x : 0 < x)
  (h_eq : 2 * (Real.log x / Real.log c)^2 + 5 * (Real.log x / Real.log a)^2 = 12 * (Real.log x)^2 / (Real.log a * Real.log c)) :
  c = a^(3/2) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_implies_c_eq_a_to_three_halves_l4035_403561


namespace NUMINAMATH_CALUDE_normal_vector_perpendicular_cosine_angle_between_lines_distance_point_to_line_l4035_403511

/-- A line in 2D space represented by the equation Ax + By + C = 0 -/
structure Line2D where
  A : ℝ
  B : ℝ
  C : ℝ
  nonzero : A ≠ 0 ∨ B ≠ 0

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The vector perpendicular to a line is its normal vector -/
theorem normal_vector_perpendicular (l : Line2D) :
  let dir_vec := (-l.B, l.A)
  let normal_vec := (l.A, l.B)
  (dir_vec.1 * normal_vec.1 + dir_vec.2 * normal_vec.2 = 0) :=
sorry

/-- The cosine of the angle between two intersecting lines -/
theorem cosine_angle_between_lines (l₁ l₂ : Line2D) :
  let cos_theta := |(l₁.A * l₂.A + l₁.B * l₂.B) / (Real.sqrt (l₁.A^2 + l₁.B^2) * Real.sqrt (l₂.A^2 + l₂.B^2))|
  (0 ≤ cos_theta ∧ cos_theta ≤ 1) :=
sorry

/-- The distance from a point to a line -/
theorem distance_point_to_line (p : Point2D) (l : Line2D) :
  let d := |l.A * p.x + l.B * p.y + l.C| / Real.sqrt (l.A^2 + l.B^2)
  (d ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_normal_vector_perpendicular_cosine_angle_between_lines_distance_point_to_line_l4035_403511


namespace NUMINAMATH_CALUDE_dans_remaining_cards_l4035_403528

/-- Given Dan's initial number of baseball cards, the number of torn cards,
    and the number of cards sold to Sam, prove that Dan now has 82 baseball cards. -/
theorem dans_remaining_cards
  (initial_cards : ℕ)
  (torn_cards : ℕ)
  (sold_cards : ℕ)
  (h1 : initial_cards = 97)
  (h2 : torn_cards = 8)
  (h3 : sold_cards = 15) :
  initial_cards - sold_cards = 82 := by
  sorry

end NUMINAMATH_CALUDE_dans_remaining_cards_l4035_403528


namespace NUMINAMATH_CALUDE_john_phone_bill_cost_l4035_403509

/-- Calculates the total cost of a phone bill given the monthly fee, per-minute rate, and minutes used. -/
def phoneBillCost (monthlyFee : ℝ) (perMinuteRate : ℝ) (minutesUsed : ℝ) : ℝ :=
  monthlyFee + perMinuteRate * minutesUsed

theorem john_phone_bill_cost :
  phoneBillCost 5 0.25 28.08 = 12.02 := by
  sorry

end NUMINAMATH_CALUDE_john_phone_bill_cost_l4035_403509


namespace NUMINAMATH_CALUDE_cone_surface_area_l4035_403569

theorem cone_surface_area (r h : ℝ) (hr : r = 4) (hh : h = 2 * Real.sqrt 5) :
  let slant_height := Real.sqrt (r^2 + h^2)
  let base_area := π * r^2
  let lateral_area := π * r * slant_height
  base_area + lateral_area = 40 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_surface_area_l4035_403569


namespace NUMINAMATH_CALUDE_u_value_when_m_is_3_l4035_403570

-- Define the functions u and t
def t (m : ℕ) : ℕ := 3^m + m
def u (m : ℕ) : ℕ := 4^(t m) - 3*(t m)

-- State the theorem
theorem u_value_when_m_is_3 : u 3 = 4^30 - 90 := by
  sorry

end NUMINAMATH_CALUDE_u_value_when_m_is_3_l4035_403570


namespace NUMINAMATH_CALUDE_function_inequality_l4035_403507

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the property of being an even function
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define the property of being monotonically decreasing on an interval
def MonoDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

-- State the theorem
theorem function_inequality (h1 : IsEven f) (h2 : MonoDecreasing (fun x => f (x - 2)) 0 2) :
  f 0 < f (-1) ∧ f (-1) < f 2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l4035_403507


namespace NUMINAMATH_CALUDE_sqrt_inequality_l4035_403585

theorem sqrt_inequality (x : ℝ) : 
  Real.sqrt (x^2 - 3*x + 2) > x + 5 ↔ x < -23/13 := by sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l4035_403585


namespace NUMINAMATH_CALUDE_marina_olympiad_supplies_l4035_403551

/-- The cost of school supplies for Marina's olympiad participation. -/
def school_supplies_cost 
  (notebook : ℕ) 
  (pencil : ℕ) 
  (eraser : ℕ) 
  (ruler : ℕ) 
  (pen : ℕ) : Prop :=
  notebook = 15 ∧ 
  notebook + pencil + eraser = 47 ∧
  notebook + ruler + pen = 58 →
  notebook + pencil + eraser + ruler + pen = 90

theorem marina_olympiad_supplies : 
  ∃ (notebook pencil eraser ruler pen : ℕ), 
  school_supplies_cost notebook pencil eraser ruler pen :=
sorry

end NUMINAMATH_CALUDE_marina_olympiad_supplies_l4035_403551


namespace NUMINAMATH_CALUDE_janas_height_l4035_403527

/-- Given the heights of several people and their relationships, prove Jana's height. -/
theorem janas_height
  (kelly_jess : ℝ) -- Height difference between Kelly and Jess
  (jana_kelly : ℝ) -- Height difference between Jana and Kelly
  (jess_height : ℝ) -- Jess's height
  (jess_alex : ℝ) -- Height difference between Jess and Alex
  (alex_sam : ℝ) -- Height difference between Alex and Sam
  (h1 : jana_kelly = 5.5)
  (h2 : kelly_jess = -3.75)
  (h3 : jess_height = 72)
  (h4 : jess_alex = -1.25)
  (h5 : alex_sam = 0.5)
  : jess_height - kelly_jess + jana_kelly = 73.75 := by
  sorry

#check janas_height

end NUMINAMATH_CALUDE_janas_height_l4035_403527


namespace NUMINAMATH_CALUDE_triangle_area_ratio_l4035_403557

/-- A rectangle on a coordinate grid with vertices at (0,0), (x,0), (0,y), and (x,y) -/
structure Rectangle where
  x : ℝ
  y : ℝ

/-- The number of parts the diagonals are divided into -/
structure DiagonalDivisions where
  n : ℕ  -- number of parts for diagonal from (0,0) to (x,y)
  m : ℕ  -- number of parts for diagonal from (x,0) to (0,y)

/-- Triangle formed by joining a point on a diagonal to the rectangle's center -/
inductive Triangle
  | A  -- formed from diagonal (0,0) to (x,y)
  | B  -- formed from diagonal (x,0) to (0,y)

/-- The area of a triangle -/
def triangleArea (t : Triangle) (r : Rectangle) (d : DiagonalDivisions) : ℝ :=
  sorry  -- definition omitted as it's not directly given in the problem conditions

/-- The theorem to be proved -/
theorem triangle_area_ratio (r : Rectangle) (d : DiagonalDivisions) :
  triangleArea Triangle.A r d / triangleArea Triangle.B r d = d.m / d.n :=
sorry

end NUMINAMATH_CALUDE_triangle_area_ratio_l4035_403557


namespace NUMINAMATH_CALUDE_leap_year_inequality_l4035_403559

/-- Represents the dataset for a leap year as described in the problem -/
def leapYearData : List ℕ := sorry

/-- Calculates the median of modes for the leap year dataset -/
def medianOfModes (data : List ℕ) : ℚ := sorry

/-- Calculates the mean for the leap year dataset -/
def mean (data : List ℕ) : ℚ := sorry

/-- Calculates the median for the leap year dataset -/
def median (data : List ℕ) : ℚ := sorry

theorem leap_year_inequality :
  let d := medianOfModes leapYearData
  let μ := mean leapYearData
  let M := median leapYearData
  d < μ ∧ μ < M := by sorry

end NUMINAMATH_CALUDE_leap_year_inequality_l4035_403559


namespace NUMINAMATH_CALUDE_parabola_max_value_l4035_403530

theorem parabola_max_value :
  ∃ (max : ℝ), max = 4 ∧ ∀ (x : ℝ), -x^2 + 2*x + 3 ≤ max :=
by sorry

end NUMINAMATH_CALUDE_parabola_max_value_l4035_403530


namespace NUMINAMATH_CALUDE_special_number_unique_l4035_403508

/-- The unique integer between 10000 and 99999 satisfying the given conditions -/
def special_number : ℕ := 11311

/-- Checks if a natural number is between 10000 and 99999 -/
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

/-- Extracts the digits of a five-digit number -/
def digits (n : ℕ) : Fin 5 → ℕ
| 0 => n / 10000
| 1 => (n / 1000) % 10
| 2 => (n / 100) % 10
| 3 => (n / 10) % 10
| 4 => n % 10

theorem special_number_unique :
  ∀ n : ℕ, is_five_digit n →
    (digits n 0 = n % 2) →
    (digits n 1 = n % 3) →
    (digits n 2 = n % 4) →
    (digits n 3 = n % 5) →
    (digits n 4 = n % 6) →
    n = special_number := by sorry

end NUMINAMATH_CALUDE_special_number_unique_l4035_403508


namespace NUMINAMATH_CALUDE_square_area_increase_l4035_403581

theorem square_area_increase (x : ℝ) : (x + 3)^2 - x^2 = 45 → x^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_area_increase_l4035_403581


namespace NUMINAMATH_CALUDE_max_value_fraction_l4035_403563

theorem max_value_fraction (x y : ℝ) (hx : -5 ≤ x ∧ x ≤ -3) (hy : 1 ≤ y ∧ y ≤ 3) :
  (∀ x' y', -5 ≤ x' ∧ x' ≤ -3 ∧ 1 ≤ y' ∧ y' ≤ 3 → (x' + 2*y') / x' ≤ (x + 2*y) / x) →
  (x + 2*y) / x = -1/5 :=
by sorry

end NUMINAMATH_CALUDE_max_value_fraction_l4035_403563


namespace NUMINAMATH_CALUDE_tangent_line_to_exp_curve_l4035_403506

/-- The value of k for which the line y = kx is tangent to the curve y = e^x -/
theorem tangent_line_to_exp_curve (k : ℝ) : 
  (∃ x₀ : ℝ, k * x₀ = Real.exp x₀ ∧ k = Real.exp x₀) → k = Real.exp 1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_to_exp_curve_l4035_403506


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l4035_403558

theorem trigonometric_simplification (x y : ℝ) :
  Real.sin x ^ 2 + Real.sin (x + y) ^ 2 - 2 * Real.sin x * Real.sin y * Real.sin (x + y) = Real.sin y ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l4035_403558


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l4035_403577

theorem hyperbola_eccentricity :
  let hyperbola := fun (x y : ℝ) => x^2 / 5 - y^2 / 4 = 1
  ∃ (e : ℝ), e = (3 * Real.sqrt 5) / 5 ∧
    ∀ (x y : ℝ), hyperbola x y → 
      e = Real.sqrt ((x^2 / 5) + (y^2 / 4)) / Real.sqrt (x^2 / 5) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l4035_403577


namespace NUMINAMATH_CALUDE_school_transfer_percentage_l4035_403594

theorem school_transfer_percentage : 
  ∀ (total_students : ℕ) (school_A_percent school_C_percent : ℚ),
    school_A_percent = 60 / 100 →
    (30 / 100 * school_A_percent + 
     (school_C_percent - 30 / 100 * school_A_percent) / (1 - school_A_percent)) * total_students = 
    school_C_percent * total_students →
    school_C_percent = 34 / 100 →
    (school_C_percent - 30 / 100 * school_A_percent) / (1 - school_A_percent) = 40 / 100 :=
by sorry

end NUMINAMATH_CALUDE_school_transfer_percentage_l4035_403594


namespace NUMINAMATH_CALUDE_special_op_nine_ten_l4035_403502

-- Define the ⊕ operation
def special_op (A B : ℚ) : ℚ := 1 / (A * B) + 1 / ((A + 1) * (B + 2))

-- State the theorem
theorem special_op_nine_ten :
  special_op 9 10 = 7 / 360 :=
by
  -- The proof goes here
  sorry

-- Additional fact given in the problem
axiom special_op_one_two : special_op 1 2 = 5 / 8

end NUMINAMATH_CALUDE_special_op_nine_ten_l4035_403502


namespace NUMINAMATH_CALUDE_a_range_l4035_403536

/-- A cubic function with a parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 - 2

/-- Theorem stating the range of a given the conditions -/
theorem a_range (a : ℝ) :
  (∃! x, f a x = 0) ∧ 
  (∀ x, f a x = 0 → x < 0) →
  a < -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_a_range_l4035_403536


namespace NUMINAMATH_CALUDE_ball_count_l4035_403566

/-- Given a box of balls with specific properties, prove the total number of balls -/
theorem ball_count (orange purple yellow total : ℕ) : 
  orange + purple + yellow = total →  -- Total balls
  orange = 2 * n →  -- Ratio condition for orange
  purple = 3 * n →  -- Ratio condition for purple
  yellow = 4 * n →  -- Ratio condition for yellow
  yellow = 32 →     -- Given number of yellow balls
  total = 72 :=     -- Prove total number of balls
by sorry

end NUMINAMATH_CALUDE_ball_count_l4035_403566


namespace NUMINAMATH_CALUDE_machine_A_rate_l4035_403596

/-- Production rates of machines A, P, and Q -/
structure MachineRates where
  rateA : ℝ
  rateP : ℝ
  rateQ : ℝ

/-- Time taken by machines P and Q to produce 220 sprockets -/
structure MachineTimes where
  timeP : ℝ
  timeQ : ℝ

/-- Conditions of the sprocket manufacturing problem -/
def sprocketProblem (r : MachineRates) (t : MachineTimes) : Prop :=
  220 / t.timeP = r.rateP
  ∧ 220 / t.timeQ = r.rateQ
  ∧ t.timeP = t.timeQ + 10
  ∧ r.rateQ = 1.1 * r.rateA
  ∧ r.rateA > 0
  ∧ r.rateP > 0
  ∧ r.rateQ > 0
  ∧ t.timeP > 0
  ∧ t.timeQ > 0

/-- Theorem stating that machine A's production rate is 20/9 sprockets per hour -/
theorem machine_A_rate (r : MachineRates) (t : MachineTimes) 
  (h : sprocketProblem r t) : r.rateA = 20/9 := by
  sorry

end NUMINAMATH_CALUDE_machine_A_rate_l4035_403596


namespace NUMINAMATH_CALUDE_triangle_property_l4035_403575

open Real

theorem triangle_property (A B C a b c : ℝ) :
  A > 0 → B > 0 → C > 0 →
  a > 0 → b > 0 → c > 0 →
  A + B + C = π →
  a / sin A = b / sin B →
  a / sin A = c / sin C →
  1 / tan A + 1 / tan C = 1 / sin B →
  b^2 = a * c :=
by sorry

end NUMINAMATH_CALUDE_triangle_property_l4035_403575


namespace NUMINAMATH_CALUDE_percentage_not_sold_l4035_403518

def initial_stock : ℕ := 1200
def monday_sales : ℕ := 75
def tuesday_sales : ℕ := 50
def wednesday_sales : ℕ := 64
def thursday_sales : ℕ := 78
def friday_sales : ℕ := 135

def total_sales : ℕ := monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales
def books_not_sold : ℕ := initial_stock - total_sales

theorem percentage_not_sold : 
  (books_not_sold : ℚ) / initial_stock * 100 = 66.5 := by sorry

end NUMINAMATH_CALUDE_percentage_not_sold_l4035_403518


namespace NUMINAMATH_CALUDE_parallelogram_height_l4035_403523

/-- Given a parallelogram with area 576 cm² and base 32 cm, its height is 18 cm. -/
theorem parallelogram_height (area : ℝ) (base : ℝ) (height : ℝ) : 
  area = 576 ∧ base = 32 ∧ area = base * height → height = 18 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_height_l4035_403523


namespace NUMINAMATH_CALUDE_halfway_between_fractions_average_of_fractions_l4035_403522

theorem halfway_between_fractions : 
  (2 : ℚ) / 7 + (4 : ℚ) / 9 = (46 : ℚ) / 63 :=
by sorry

theorem average_of_fractions : 
  ((2 : ℚ) / 7 + (4 : ℚ) / 9) / 2 = (23 : ℚ) / 63 :=
by sorry

end NUMINAMATH_CALUDE_halfway_between_fractions_average_of_fractions_l4035_403522


namespace NUMINAMATH_CALUDE_sin_3phi_value_l4035_403556

theorem sin_3phi_value (φ : ℝ) (h : Complex.exp (Complex.I * φ) = (3 + Complex.I * Real.sqrt 8) / 5) :
  Real.sin (3 * φ) = 19 * Real.sqrt 8 / 125 := by
  sorry

end NUMINAMATH_CALUDE_sin_3phi_value_l4035_403556


namespace NUMINAMATH_CALUDE_janet_stickers_l4035_403531

theorem janet_stickers (initial_stickers received_stickers : ℕ) : 
  initial_stickers = 3 → received_stickers = 53 → initial_stickers + received_stickers = 56 := by
  sorry

end NUMINAMATH_CALUDE_janet_stickers_l4035_403531


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l4035_403517

theorem sufficient_not_necessary (a b : ℝ) :
  (∀ a b, a > 1 ∧ b > 2 → a + b > 3 ∧ a * b > 2) ∧
  (∃ a b, a + b > 3 ∧ a * b > 2 ∧ ¬(a > 1 ∧ b > 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l4035_403517


namespace NUMINAMATH_CALUDE_competition_results_l4035_403543

/-- Represents the weights of lifts for an athlete -/
structure AthleteLifts where
  first : ℕ
  second : ℕ

/-- The competition results -/
def Competition : Type :=
  AthleteLifts × AthleteLifts × AthleteLifts

def joe_total (c : Competition) : ℕ := c.1.first + c.1.second
def mike_total (c : Competition) : ℕ := c.2.1.first + c.2.1.second
def lisa_total (c : Competition) : ℕ := c.2.2.first + c.2.2.second

def joe_condition (c : Competition) : Prop :=
  2 * c.1.first = c.1.second + 300

def mike_condition (c : Competition) : Prop :=
  c.2.1.second = c.2.1.first + 200

def lisa_condition (c : Competition) : Prop :=
  c.2.2.first = 3 * c.2.2.second

theorem competition_results (c : Competition) 
  (h1 : joe_total c = 900)
  (h2 : mike_total c = 1100)
  (h3 : lisa_total c = 1000)
  (h4 : joe_condition c)
  (h5 : mike_condition c)
  (h6 : lisa_condition c) :
  c.1.first = 400 ∧ c.2.1.first = 450 ∧ c.2.2.second = 250 := by
  sorry

end NUMINAMATH_CALUDE_competition_results_l4035_403543


namespace NUMINAMATH_CALUDE_all_triangles_congruent_l4035_403573

/-- Represents a square tablecloth with hanging triangles -/
structure Tablecloth where
  -- Side length of the square tablecloth
  side : ℝ
  -- Heights of the hanging triangles
  hA : ℝ
  hB : ℝ
  hC : ℝ
  hD : ℝ
  -- Condition that all heights are positive
  hA_pos : hA > 0
  hB_pos : hB > 0
  hC_pos : hC > 0
  hD_pos : hD > 0
  -- Condition that △A and △B are congruent (given)
  hA_eq_hB : hA = hB

/-- Theorem stating that if △A and △B are congruent, then all hanging triangles are congruent -/
theorem all_triangles_congruent (t : Tablecloth) :
  t.hA = t.hB ∧ t.hA = t.hC ∧ t.hA = t.hD :=
sorry

end NUMINAMATH_CALUDE_all_triangles_congruent_l4035_403573


namespace NUMINAMATH_CALUDE_matrix_power_four_l4035_403599

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -1; 1, 1]

theorem matrix_power_four :
  A ^ 4 = !![0, -9; 9, -9] := by sorry

end NUMINAMATH_CALUDE_matrix_power_four_l4035_403599


namespace NUMINAMATH_CALUDE_regular_octagon_interior_angle_l4035_403510

/-- The measure of one interior angle of a regular octagon is 135 degrees. -/
theorem regular_octagon_interior_angle : ℝ := by
  sorry

end NUMINAMATH_CALUDE_regular_octagon_interior_angle_l4035_403510


namespace NUMINAMATH_CALUDE_last_term_is_one_l4035_403582

/-- A sequence is k-th order repeatable if there exist two sets of consecutive k terms that match in order. -/
def kth_order_repeatable (a : ℕ → Fin 2) (m k : ℕ) : Prop :=
  ∃ i j, i ≠ j ∧ i + k ≤ m ∧ j + k ≤ m ∧ ∀ t, t < k → a (i + t) = a (j + t)

theorem last_term_is_one
  (a : ℕ → Fin 2)
  (m : ℕ)
  (h_m : m ≥ 3)
  (h_not_5th : ¬ kth_order_repeatable a m 5)
  (h_5th_after : ∀ b : Fin 2, kth_order_repeatable (Function.update a m b) (m + 1) 5)
  (h_a4 : a 4 ≠ 1) :
  a m = 1 :=
sorry

end NUMINAMATH_CALUDE_last_term_is_one_l4035_403582


namespace NUMINAMATH_CALUDE_correct_subtraction_l4035_403576

theorem correct_subtraction (x : ℤ) (h1 : x - 32 = 25) (h2 : 23 ≠ 32) : x - 23 = 34 := by
  sorry

end NUMINAMATH_CALUDE_correct_subtraction_l4035_403576


namespace NUMINAMATH_CALUDE_complex_modulus_l4035_403565

theorem complex_modulus (z : ℂ) (h : (1 + Complex.I) * z = 1 - 2 * Complex.I) : 
  Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l4035_403565


namespace NUMINAMATH_CALUDE_price_difference_year_l4035_403512

/-- 
Given:
- The price of commodity X increases by 45 cents every year
- The price of commodity Y increases by 20 cents every year
- In 2001, the price of commodity X was $4.20
- The price of commodity Y in 2001 is Y dollars

Prove that the number of years n after 2001 when the price of X is 65 cents more than 
the price of Y is given by n = (Y - 3.55) / 0.25
-/
theorem price_difference_year (Y : ℝ) : 
  let n : ℝ := (Y - 3.55) / 0.25
  let price_X (t : ℝ) : ℝ := 4.20 + 0.45 * t
  let price_Y (t : ℝ) : ℝ := Y + 0.20 * t
  price_X n = price_Y n + 0.65 :=
by sorry

end NUMINAMATH_CALUDE_price_difference_year_l4035_403512


namespace NUMINAMATH_CALUDE_power_fraction_evaluation_l4035_403525

theorem power_fraction_evaluation (a b : ℕ) : 
  (2^a : ℕ) ∣ 360 ∧ 
  (3^b : ℕ) ∣ 360 ∧ 
  ∀ k > a, ¬((2^k : ℕ) ∣ 360) ∧ 
  ∀ l > b, ¬((3^l : ℕ) ∣ 360) →
  ((1/4 : ℚ) ^ (b - a) : ℚ) = 4 := by
sorry

end NUMINAMATH_CALUDE_power_fraction_evaluation_l4035_403525


namespace NUMINAMATH_CALUDE_probability_all_red_balls_l4035_403504

def total_balls : ℕ := 10
def red_balls : ℕ := 5
def blue_balls : ℕ := 5
def drawn_balls : ℕ := 5

theorem probability_all_red_balls :
  (Nat.choose red_balls drawn_balls) / (Nat.choose total_balls drawn_balls) = 1 / 252 := by
  sorry

end NUMINAMATH_CALUDE_probability_all_red_balls_l4035_403504


namespace NUMINAMATH_CALUDE_original_bill_calculation_l4035_403549

theorem original_bill_calculation (num_friends : ℕ) (discount_rate : ℚ) (individual_payment : ℚ) :
  num_friends = 5 →
  discount_rate = 6 / 100 →
  individual_payment = 188 / 10 →
  ∃ (original_bill : ℚ), 
    (1 - discount_rate) * original_bill = num_friends * individual_payment ∧
    original_bill = 100 := by
  sorry

#check original_bill_calculation

end NUMINAMATH_CALUDE_original_bill_calculation_l4035_403549


namespace NUMINAMATH_CALUDE_number_of_schedules_l4035_403540

/-- Represents the number of periods in a day -/
def total_periods : ℕ := 6

/-- Represents the number of morning periods -/
def morning_periods : ℕ := 3

/-- Represents the number of afternoon periods -/
def afternoon_periods : ℕ := 3

/-- Represents the total number of classes -/
def total_classes : ℕ := 6

/-- Represents the constraint that Mathematics must be in the morning -/
def math_in_morning : Prop := true

/-- Represents the constraint that Art must be in the afternoon -/
def art_in_afternoon : Prop := true

/-- The main theorem stating the number of possible schedules -/
theorem number_of_schedules :
  math_in_morning →
  art_in_afternoon →
  (total_periods = morning_periods + afternoon_periods) →
  (∃ (n : ℕ), n = 216 ∧ n = number_of_possible_schedules) :=
sorry

end NUMINAMATH_CALUDE_number_of_schedules_l4035_403540


namespace NUMINAMATH_CALUDE_max_value_implies_ratio_l4035_403548

def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x - a^2 - 7*a

theorem max_value_implies_ratio (a b : ℝ) :
  (∀ x, f a b x ≤ f a b 1) ∧
  (f a b 1 = 10) →
  a / b = -2/3 :=
sorry

end NUMINAMATH_CALUDE_max_value_implies_ratio_l4035_403548


namespace NUMINAMATH_CALUDE_root_product_theorem_l4035_403541

theorem root_product_theorem (x₁ x₂ x₃ x₄ x₅ : ℂ) : 
  (x₁^5 - 3*x₁^3 + x₁ + 6 = 0) →
  (x₂^5 - 3*x₂^3 + x₂ + 6 = 0) →
  (x₃^5 - 3*x₃^3 + x₃ + 6 = 0) →
  (x₄^5 - 3*x₄^3 + x₄ + 6 = 0) →
  (x₅^5 - 3*x₅^3 + x₅ + 6 = 0) →
  ((x₁^2 - 2) * (x₂^2 - 2) * (x₃^2 - 2) * (x₄^2 - 2) * (x₅^2 - 2) = 10) := by
  sorry

end NUMINAMATH_CALUDE_root_product_theorem_l4035_403541


namespace NUMINAMATH_CALUDE_quadratic_sum_zero_l4035_403598

/-- A quadratic function f(x) = ax^2 + bx + c -/
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_sum_zero 
  (a b c : ℝ) 
  (h1 : quadratic a b c 1 = 0)
  (h2 : quadratic a b c 5 = 0)
  (h3 : ∃ (k : ℝ), ∀ (x : ℝ), quadratic a b c x ≥ 36 ∧ quadratic a b c k = 36) :
  a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_zero_l4035_403598


namespace NUMINAMATH_CALUDE_equation_solution_inequality_solution_l4035_403505

-- Equation problem
theorem equation_solution :
  ∀ x : ℝ, 6 * x - 2 * (x - 3) = 14 ↔ x = 2 := by sorry

-- Inequality problem
theorem inequality_solution :
  ∀ x : ℝ, 3 * (x + 3) < x + 7 ↔ x < -1 := by sorry

end NUMINAMATH_CALUDE_equation_solution_inequality_solution_l4035_403505


namespace NUMINAMATH_CALUDE_larger_cross_section_distance_l4035_403534

/-- Right octagonal pyramid with two parallel cross sections -/
structure OctagonalPyramid where
  /-- Ratio of areas of two cross sections -/
  area_ratio : ℝ
  /-- Distance between the two cross sections -/
  cross_section_distance : ℝ

/-- Theorem about the distance of the larger cross section from the apex -/
theorem larger_cross_section_distance (pyramid : OctagonalPyramid) 
  (h_ratio : pyramid.area_ratio = 4 / 9)
  (h_distance : pyramid.cross_section_distance = 10) :
  ∃ (apex_distance : ℝ), apex_distance = 30 := by
  sorry

end NUMINAMATH_CALUDE_larger_cross_section_distance_l4035_403534


namespace NUMINAMATH_CALUDE_is_quadratic_equation_l4035_403592

theorem is_quadratic_equation (x : ℝ) : ∃ (a b c : ℝ), a ≠ 0 ∧ (x - 1)^2 = 2*(3 - x)^2 ↔ a*x^2 + b*x + c = 0 :=
sorry

end NUMINAMATH_CALUDE_is_quadratic_equation_l4035_403592


namespace NUMINAMATH_CALUDE_rhombus_properties_l4035_403572

-- Define a rhombus
structure Rhombus (V : Type*) [NormedAddCommGroup V] :=
  (A B C D : V)
  (is_rhombus : True)  -- This is a placeholder for the rhombus property

-- Define the theorem
theorem rhombus_properties {V : Type*} [NormedAddCommGroup V] (r : Rhombus V) :
  (‖r.A - r.B‖ = ‖r.B - r.C‖) ∧ 
  (‖r.A - r.B - (r.C - r.D)‖ = ‖r.A - r.D + (r.B - r.C)‖) ∧
  (‖r.A - r.C‖^2 + ‖r.B - r.D‖^2 = 4 * ‖r.A - r.B‖^2) :=
by sorry

end NUMINAMATH_CALUDE_rhombus_properties_l4035_403572


namespace NUMINAMATH_CALUDE_sum_relations_l4035_403580

theorem sum_relations (a b c d : ℝ) 
  (hab : a + b = 4)
  (hcd : c + d = 3)
  (had : a + d = 2) :
  b + c = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_relations_l4035_403580


namespace NUMINAMATH_CALUDE_numeric_methods_students_l4035_403516

/-- The total number of students in the faculty -/
def total_students : ℕ := 653

/-- The number of second-year students studying automatic control -/
def auto_control_students : ℕ := 423

/-- The number of second-year students studying both numeric methods and automatic control -/
def both_subjects_students : ℕ := 134

/-- The approximate percentage of second-year students in the faculty -/
def second_year_percentage : ℚ := 80/100

/-- The number of second-year students (rounded) -/
def second_year_students : ℕ := 522

/-- Theorem stating the number of second-year students studying numeric methods -/
theorem numeric_methods_students : 
  ∃ (n : ℕ), n = second_year_students - (auto_control_students - both_subjects_students) ∧ n = 233 :=
sorry

end NUMINAMATH_CALUDE_numeric_methods_students_l4035_403516


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_35_l4035_403578

theorem smallest_four_digit_divisible_by_35 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 → n ≥ 1015 := by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_35_l4035_403578


namespace NUMINAMATH_CALUDE_ab_difference_l4035_403538

theorem ab_difference (a b : ℝ) (h : |a - 2| + |b + 3| = 0) : a - b = 5 := by
  sorry

end NUMINAMATH_CALUDE_ab_difference_l4035_403538


namespace NUMINAMATH_CALUDE_crazy_silly_school_books_l4035_403571

theorem crazy_silly_school_books (num_movies : ℕ) (movie_book_diff : ℕ) : 
  num_movies = 17 → movie_book_diff = 6 → num_movies - movie_book_diff = 11 := by
  sorry

end NUMINAMATH_CALUDE_crazy_silly_school_books_l4035_403571


namespace NUMINAMATH_CALUDE_lcm_of_150_and_456_l4035_403584

theorem lcm_of_150_and_456 : Nat.lcm 150 456 = 11400 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_150_and_456_l4035_403584


namespace NUMINAMATH_CALUDE_y_satisfies_equation_l4035_403503

noncomputable section

variable (x : ℝ)

def y (x : ℝ) : ℝ := (2 * x) / (x^3 + 1) + 1 / x

def equation (x : ℝ) (y : ℝ → ℝ) : Prop :=
  x * (x^3 + 1) * (deriv y x) + (2 * x^3 - 1) * (y x) = (x^3 - 2) / x

theorem y_satisfies_equation :
  ∀ x ≠ 0, equation x y
  := by sorry

end NUMINAMATH_CALUDE_y_satisfies_equation_l4035_403503


namespace NUMINAMATH_CALUDE_square_perimeter_l4035_403514

theorem square_perimeter (area : ℝ) (side : ℝ) (h1 : area = 200) (h2 : side^2 = area) :
  4 * side = 40 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l4035_403514


namespace NUMINAMATH_CALUDE_self_inverse_solutions_l4035_403533

def is_self_inverse (a d : ℝ) : Prop :=
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![a, 4; -9, d]
  M * M = 1

theorem self_inverse_solutions :
  ∃! n : ℕ, ∃ S : Finset (ℝ × ℝ),
    S.card = n ∧
    (∀ p : ℝ × ℝ, p ∈ S ↔ is_self_inverse p.1 p.2) :=
by sorry

end NUMINAMATH_CALUDE_self_inverse_solutions_l4035_403533


namespace NUMINAMATH_CALUDE_square_differences_sum_l4035_403545

theorem square_differences_sum : 1010^2 - 990^2 - 1005^2 + 995^2 - 1002^2 + 998^2 = 28000 := by
  sorry

end NUMINAMATH_CALUDE_square_differences_sum_l4035_403545


namespace NUMINAMATH_CALUDE_purchase_savings_l4035_403537

/-- Calculates the total savings on a purchase given the original and discounted prices -/
def calculateSavings (originalPrice discountedPrice : ℚ) (quantity : ℕ) : ℚ :=
  (originalPrice - discountedPrice) * quantity

/-- Calculates the discounted price given the original price and discount percentage -/
def calculateDiscountedPrice (originalPrice : ℚ) (discountPercentage : ℚ) : ℚ :=
  originalPrice * (1 - discountPercentage)

theorem purchase_savings :
  let folderQuantity : ℕ := 7
  let folderPrice : ℚ := 3
  let folderDiscount : ℚ := 0.25
  let penQuantity : ℕ := 4
  let penPrice : ℚ := 1.5
  let penDiscount : ℚ := 0.1
  let folderSavings := calculateSavings folderPrice (calculateDiscountedPrice folderPrice folderDiscount) folderQuantity
  let penSavings := calculateSavings penPrice (calculateDiscountedPrice penPrice penDiscount) penQuantity
  folderSavings + penSavings = 5.85 := by
  sorry


end NUMINAMATH_CALUDE_purchase_savings_l4035_403537


namespace NUMINAMATH_CALUDE_problem_statement_l4035_403535

theorem problem_statement (a b m : ℝ) 
  (h1 : 2^a = m) 
  (h2 : 5^b = m) 
  (h3 : 1/a + 1/b = 1) : 
  m = 10 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l4035_403535


namespace NUMINAMATH_CALUDE_distance_relationships_l4035_403588

structure Distance where
  r : ℝ
  a : ℝ
  b : ℝ

def perpendicular_to_x12 (d : Distance) : Prop := sorry
def parallel_to_H (d : Distance) : Prop := sorry
def parallel_to_P1 (d : Distance) : Prop := sorry
def perpendicular_to_H (d : Distance) : Prop := sorry
def perpendicular_to_P1 (d : Distance) : Prop := sorry
def parallel_to_x12 (d : Distance) : Prop := sorry

theorem distance_relationships (d : Distance) :
  (∃ α β : ℝ, d.a = d.r * Real.cos α ∧ d.b = d.r * Real.cos β) ∧
  (perpendicular_to_x12 d → d.a^2 + d.b^2 = d.r^2) ∧
  (parallel_to_H d → d.a = d.b) ∧
  (parallel_to_P1 d → d.a = d.r ∧ ∃ β : ℝ, d.b = d.a * Real.cos β) ∧
  (perpendicular_to_H d → d.a = d.b ∧ d.a = d.r * Real.sqrt 2 / 2) ∧
  (perpendicular_to_P1 d → d.a = 0 ∧ d.b = d.r) ∧
  (parallel_to_x12 d → d.a = d.b ∧ d.a = d.r) :=
by sorry

end NUMINAMATH_CALUDE_distance_relationships_l4035_403588


namespace NUMINAMATH_CALUDE_greatest_solution_quadratic_l4035_403589

theorem greatest_solution_quadratic : 
  ∃ (x : ℝ), x = 4/5 ∧ 5*x^2 - 3*x - 4 = 0 ∧ 
  ∀ (y : ℝ), 5*y^2 - 3*y - 4 = 0 → y ≤ x :=
by sorry

end NUMINAMATH_CALUDE_greatest_solution_quadratic_l4035_403589


namespace NUMINAMATH_CALUDE_last_score_is_71_l4035_403501

def scores : List Nat := [71, 74, 79, 85, 88, 92]

def is_valid_last_score (last_score : Nat) : Prop :=
  last_score ∈ scores ∧
  ∀ n : Nat, 1 ≤ n ∧ n ≤ 6 → 
    (scores.sum - last_score) % n = 0

theorem last_score_is_71 : 
  ∃! last_score, is_valid_last_score last_score ∧ last_score = 71 := by sorry

end NUMINAMATH_CALUDE_last_score_is_71_l4035_403501


namespace NUMINAMATH_CALUDE_star_polygon_n_value_l4035_403529

/-- Represents an n-pointed regular star polygon -/
structure StarPolygon (n : ℕ) where
  -- All 2n edges are congruent (implicit in the structure)
  -- Alternate angles A₁, A₂, ..., Aₙ are congruent (implicit)
  -- Alternate angles B₁, B₂, ..., Bₙ are congruent (implicit)
  angle_A : ℝ  -- Acute angle at each Aᵢ
  angle_B : ℝ  -- Acute angle at each Bᵢ
  angle_diff : angle_B = angle_A + 20  -- Angle difference condition
  sum_external : n * (angle_A + angle_B) = 360  -- Sum of external angles

/-- Theorem: For a star polygon satisfying the given conditions, n = 36 -/
theorem star_polygon_n_value :
  ∀ (n : ℕ) (s : StarPolygon n), n = 36 :=
by sorry

end NUMINAMATH_CALUDE_star_polygon_n_value_l4035_403529


namespace NUMINAMATH_CALUDE_soda_consumption_proof_l4035_403513

/-- The number of soda bottles Debby bought -/
def total_soda_bottles : ℕ := 360

/-- The number of days the soda bottles lasted -/
def days_lasted : ℕ := 40

/-- The number of soda bottles Debby drank per day -/
def soda_bottles_per_day : ℕ := total_soda_bottles / days_lasted

theorem soda_consumption_proof : soda_bottles_per_day = 9 := by
  sorry

end NUMINAMATH_CALUDE_soda_consumption_proof_l4035_403513


namespace NUMINAMATH_CALUDE_cylinder_radius_calculation_l4035_403568

/-- Regular prism with a cylinder -/
structure PrismWithCylinder where
  -- Base side length of the prism
  base_side : ℝ
  -- Lateral edge length of the prism
  lateral_edge : ℝ
  -- Distance between cylinder axis and line AB₁
  axis_distance : ℝ
  -- Radius of the cylinder
  cylinder_radius : ℝ

/-- Theorem stating the radius of the cylinder given the prism dimensions -/
theorem cylinder_radius_calculation (p : PrismWithCylinder) 
  (h1 : p.base_side = 1)
  (h2 : p.lateral_edge = 1 / Real.sqrt 3)
  (h3 : p.axis_distance = 1 / 4) :
  p.cylinder_radius = Real.sqrt 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_radius_calculation_l4035_403568


namespace NUMINAMATH_CALUDE_original_class_size_l4035_403521

theorem original_class_size (x : ℕ) : 
  (x > 0) →                        -- Ensure the class has at least one student
  (40 * x + 12 * 32) / (x + 12) = 36 →  -- New average age equation
  x = 12 :=
by sorry

end NUMINAMATH_CALUDE_original_class_size_l4035_403521


namespace NUMINAMATH_CALUDE_specific_quadrilateral_area_l4035_403532

/-- Represents a quadrilateral ABCD with given side lengths and a right angle at C -/
structure Quadrilateral where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  AD : ℝ
  right_angle_at_C : Bool

/-- Calculates the area of the quadrilateral ABCD -/
noncomputable def area (q : Quadrilateral) : ℝ :=
  sorry

/-- Theorem stating that the area of the specific quadrilateral is 106 -/
theorem specific_quadrilateral_area :
  ∃ (q : Quadrilateral),
    q.AB = 15 ∧
    q.BC = 5 ∧
    q.CD = 12 ∧
    q.AD = 13 ∧
    q.right_angle_at_C = true ∧
    area q = 106 := by
  sorry

end NUMINAMATH_CALUDE_specific_quadrilateral_area_l4035_403532


namespace NUMINAMATH_CALUDE_joe_cars_l4035_403546

theorem joe_cars (initial_cars additional_cars : ℕ) :
  initial_cars = 50 → additional_cars = 12 → initial_cars + additional_cars = 62 := by
  sorry

end NUMINAMATH_CALUDE_joe_cars_l4035_403546


namespace NUMINAMATH_CALUDE_system_solution_l4035_403539

theorem system_solution (x y z : ℝ) : 
  x + y = 5 ∧ y + z = -1 ∧ x + z = -2 → x = 2 ∧ y = 3 ∧ z = -4 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l4035_403539


namespace NUMINAMATH_CALUDE_translated_function_coefficient_sum_l4035_403562

def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 5

def translation (h : ℝ) (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x + h)

theorem translated_function_coefficient_sum :
  ∃ (a b c : ℝ),
    (∀ x, translation 3 f x = a * x^2 + b * x + c) ∧
    a + b + c = 51 :=
by sorry

end NUMINAMATH_CALUDE_translated_function_coefficient_sum_l4035_403562


namespace NUMINAMATH_CALUDE_diagonal_path_crosses_12_tiles_l4035_403555

/-- Represents a rectangular floor tiled with 1x2 foot tiles -/
structure TiledFloor where
  width : ℕ
  length : ℕ

/-- Calculates the number of tiles crossed by a diagonal path on a tiled floor -/
def tilesCrossed (floor : TiledFloor) : ℕ :=
  floor.width / Nat.gcd floor.width floor.length +
  floor.length / Nat.gcd floor.width floor.length - 1

/-- Theorem stating that a diagonal path on an 8x18 foot floor crosses 12 tiles -/
theorem diagonal_path_crosses_12_tiles :
  let floor : TiledFloor := { width := 8, length := 18 }
  tilesCrossed floor = 12 := by sorry

end NUMINAMATH_CALUDE_diagonal_path_crosses_12_tiles_l4035_403555


namespace NUMINAMATH_CALUDE_golden_ratio_logarithm_l4035_403519

theorem golden_ratio_logarithm (p q : ℝ) (hp : p > 0) (hq : q > 0) 
  (h : Real.log p / Real.log 8 = Real.log q / Real.log 15 ∧ 
       Real.log p / Real.log 8 = Real.log (p + q) / Real.log 18) : 
  q / p = (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_golden_ratio_logarithm_l4035_403519


namespace NUMINAMATH_CALUDE_profit_is_120_l4035_403560

/-- Calculates the profit from book sales given the selling price, number of customers,
    production cost, and books per customer. -/
def calculate_profit (selling_price : ℕ) (num_customers : ℕ) (production_cost : ℕ) (books_per_customer : ℕ) : ℕ :=
  let total_books := num_customers * books_per_customer
  let revenue := selling_price * total_books
  let total_cost := production_cost * total_books
  revenue - total_cost

/-- Proves that the profit is $120 given the specified conditions. -/
theorem profit_is_120 :
  let selling_price := 20
  let num_customers := 4
  let production_cost := 5
  let books_per_customer := 2
  calculate_profit selling_price num_customers production_cost books_per_customer = 120 := by
  sorry

end NUMINAMATH_CALUDE_profit_is_120_l4035_403560


namespace NUMINAMATH_CALUDE_intersection_sum_coordinates_l4035_403586

/-- The quartic equation -/
def f (x : ℝ) : ℝ := x^4 - 4*x^3 + 4*x + 1

/-- The linear equation -/
def g (x y : ℝ) : ℝ := 2*x - 3*y - 6

/-- The intersection points of f and g -/
def intersection_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | f p.1 = p.2 ∧ g p.1 p.2 = 0}

theorem intersection_sum_coordinates :
  ∃ (x₁ x₂ x₃ x₄ y₁ y₂ y₃ y₄ : ℝ),
    (x₁, y₁) ∈ intersection_points ∧
    (x₂, y₂) ∈ intersection_points ∧
    (x₃, y₃) ∈ intersection_points ∧
    (x₄, y₄) ∈ intersection_points ∧
    x₁ + x₂ + x₃ + x₄ = 3 ∧
    y₁ + y₂ + y₃ + y₄ = -6 :=
by sorry

end NUMINAMATH_CALUDE_intersection_sum_coordinates_l4035_403586


namespace NUMINAMATH_CALUDE_leilas_savings_l4035_403520

theorem leilas_savings (savings : ℚ) : 
  (3 / 4 : ℚ) * savings + 20 = savings → savings = 80 :=
by sorry

end NUMINAMATH_CALUDE_leilas_savings_l4035_403520


namespace NUMINAMATH_CALUDE_porch_width_calculation_l4035_403542

/-- Given a house and porch with specific dimensions, calculate the width of the porch. -/
theorem porch_width_calculation (house_length house_width porch_length total_shingle_area : ℝ)
  (h1 : house_length = 20.5)
  (h2 : house_width = 10)
  (h3 : porch_length = 6)
  (h4 : total_shingle_area = 232) :
  let house_area := house_length * house_width
  let porch_area := total_shingle_area - house_area
  porch_area / porch_length = 4.5 := by sorry

end NUMINAMATH_CALUDE_porch_width_calculation_l4035_403542


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l4035_403567

theorem quadratic_form_sum (b c : ℝ) : 
  (∀ x, x^2 - 12*x + 49 = (x + b)^2 + c) → b + c = 7 := by
sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l4035_403567


namespace NUMINAMATH_CALUDE_fraction_simplification_l4035_403564

theorem fraction_simplification (x : ℝ) 
  (h1 : x ≠ 3) (h2 : x ≠ 4) (h3 : x ≠ 2) (h4 : x ≠ 5) :
  (x^2 - 4*x + 3) / (x^2 - 6*x + 9) / ((x^2 - 6*x + 8) / (x^2 - 8*x + 15)) = 
  ((x - 1) * (x - 5)) / ((x - 4) * (x - 2)) := by
sorry

end NUMINAMATH_CALUDE_fraction_simplification_l4035_403564


namespace NUMINAMATH_CALUDE_function_periodic_l4035_403552

/-- A function satisfying the given conditions is periodic with period 1 -/
theorem function_periodic (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, |f x| ≤ 1)
  (h2 : ∀ x : ℝ, f (x + 13/42) + f x = f (x + 1/6) + f (x + 1/7)) :
  ∀ x : ℝ, f (x + 1) = f x := by
  sorry

end NUMINAMATH_CALUDE_function_periodic_l4035_403552


namespace NUMINAMATH_CALUDE_trapezoid_area_l4035_403574

/-- Trapezoid ABCD with given properties -/
structure Trapezoid where
  -- Length of AD
  ad : ℝ
  -- Length of BC
  bc : ℝ
  -- Length of CD
  cd : ℝ
  -- BC is parallel to AD
  parallel : True
  -- Ratio of BC to AD is 5:7
  ratio_bc_ad : bc / ad = 5 / 7
  -- AF:FD = 4:3
  ratio_af_fd : (4 / 7 * ad) / (3 / 7 * ad) = 4 / 3
  -- CE:ED = 2:3
  ratio_ce_ed : (2 / 5 * cd) / (3 / 5 * cd) = 2 / 3
  -- Area of ABEF is 123
  area_abef : (ad * cd - (3 / 7 * ad) * (3 / 5 * cd) - bc * (2 / 5 * cd)) / 2 = 123

/-- The area of trapezoid ABCD is 180 -/
theorem trapezoid_area (t : Trapezoid) : (t.ad + t.bc) * t.cd / 2 = 180 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_l4035_403574


namespace NUMINAMATH_CALUDE_f_not_satisfy_double_property_l4035_403554

-- Define the function f(x) = x + 1
def f (x : ℝ) : ℝ := x + 1

-- State the theorem
theorem f_not_satisfy_double_property : ∃ x : ℝ, f (2 * x) ≠ 2 * f x := by
  sorry

end NUMINAMATH_CALUDE_f_not_satisfy_double_property_l4035_403554


namespace NUMINAMATH_CALUDE_smaller_number_in_ratio_l4035_403587

theorem smaller_number_in_ratio (a b : ℕ) : 
  a > 0 → b > 0 → a * 3 = b * 2 → lcm a b = 120 → a = 80 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_in_ratio_l4035_403587


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l4035_403593

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x₀ : ℝ, x₀^2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l4035_403593


namespace NUMINAMATH_CALUDE_common_tangent_sum_l4035_403544

/-- Parabola P₁ -/
def P₁ (x y : ℚ) : Prop := y = x^2 + 52/25

/-- Parabola P₂ -/
def P₂ (x y : ℚ) : Prop := x = y^2 + 81/16

/-- Common tangent line L -/
def L (a b c : ℕ) (x y : ℚ) : Prop := a * x + b * y = c

/-- L has rational slope -/
def rational_slope (a b : ℕ) : Prop := ∃ (p q : ℤ), p ≠ 0 ∧ q ≠ 0 ∧ (a : ℚ) / b = p / q

theorem common_tangent_sum (a b c : ℕ) :
  (∀ x y : ℚ, P₁ x y → L a b c x y → (∃ t : ℚ, ∀ x' y', P₁ x' y' → L a b c x' y' → (x' - x)^2 + (y' - y)^2 ≤ t^2)) →
  (∀ x y : ℚ, P₂ x y → L a b c x y → (∃ t : ℚ, ∀ x' y', P₂ x' y' → L a b c x' y' → (x' - x)^2 + (y' - y)^2 ≤ t^2)) →
  rational_slope a b →
  a > 0 → b > 0 → c > 0 →
  Nat.gcd a (Nat.gcd b c) = 1 →
  a + b + c = 168 :=
sorry

end NUMINAMATH_CALUDE_common_tangent_sum_l4035_403544


namespace NUMINAMATH_CALUDE_orange_ribbons_l4035_403583

theorem orange_ribbons (total : ℚ) 
  (yellow_frac : ℚ) (purple_frac : ℚ) (orange_frac : ℚ) (black_count : ℕ) :
  yellow_frac = 1/3 →
  purple_frac = 1/4 →
  orange_frac = 1/6 →
  black_count = 40 →
  (1 - yellow_frac - purple_frac - orange_frac) * total = black_count →
  orange_frac * total = 80/3 := by
sorry

end NUMINAMATH_CALUDE_orange_ribbons_l4035_403583


namespace NUMINAMATH_CALUDE_monitor_student_ratio_l4035_403591

/-- The ratio of monitors to students in a lunchroom --/
theorem monitor_student_ratio :
  ∀ (S : ℕ) (G B : ℝ),
    G = 0.4 * S →
    B = 0.6 * S →
    2 * G + B = 168 →
    (8 : ℝ) / S = 1 / 15 :=
by sorry

end NUMINAMATH_CALUDE_monitor_student_ratio_l4035_403591
