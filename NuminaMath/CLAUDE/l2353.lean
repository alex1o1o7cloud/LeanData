import Mathlib

namespace NUMINAMATH_CALUDE_average_monthly_balance_l2353_235305

def initial_balance : ℝ := 120
def february_change : ℝ := 80
def march_change : ℝ := -50
def april_change : ℝ := 70
def may_change : ℝ := 0
def june_change : ℝ := 100
def num_months : ℕ := 6

def monthly_balances : List ℝ := [
  initial_balance,
  initial_balance + february_change,
  initial_balance + february_change + march_change,
  initial_balance + february_change + march_change + april_change,
  initial_balance + february_change + march_change + april_change + may_change,
  initial_balance + february_change + march_change + april_change + may_change + june_change
]

theorem average_monthly_balance :
  (monthly_balances.sum / num_months) = 205 := by sorry

end NUMINAMATH_CALUDE_average_monthly_balance_l2353_235305


namespace NUMINAMATH_CALUDE_tan_fifteen_equals_sqrt_three_l2353_235370

theorem tan_fifteen_equals_sqrt_three : (1 + Real.tan (15 * π / 180)) / (1 - Real.tan (15 * π / 180)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_fifteen_equals_sqrt_three_l2353_235370


namespace NUMINAMATH_CALUDE_cosine_angle_POQ_l2353_235345

/-- Given points P, Q, and O, prove that the cosine of angle POQ is -√10/10 -/
theorem cosine_angle_POQ :
  let P : ℝ × ℝ := (1, 1)
  let Q : ℝ × ℝ := (-2, 1)
  let O : ℝ × ℝ := (0, 0)
  let OP : ℝ × ℝ := (P.1 - O.1, P.2 - O.2)
  let OQ : ℝ × ℝ := (Q.1 - O.1, Q.2 - O.2)
  let dot_product : ℝ := OP.1 * OQ.1 + OP.2 * OQ.2
  let magnitude_OP : ℝ := Real.sqrt (OP.1^2 + OP.2^2)
  let magnitude_OQ : ℝ := Real.sqrt (OQ.1^2 + OQ.2^2)
  dot_product / (magnitude_OP * magnitude_OQ) = -Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_cosine_angle_POQ_l2353_235345


namespace NUMINAMATH_CALUDE_chessboard_division_impossible_l2353_235347

/-- Represents a chessboard --/
structure Chessboard :=
  (size : ℕ)

/-- Represents a line on the chessboard --/
structure Line

/-- Represents a division of the chessboard --/
def ChessboardDivision := List Line

/-- Function to check if a division is valid --/
def is_valid_division (board : Chessboard) (division : ChessboardDivision) : Prop :=
  sorry

/-- Theorem: It's impossible to divide an 8x8 chessboard with 13 lines
    such that each region contains at most one square center --/
theorem chessboard_division_impossible :
  ∀ (board : Chessboard) (division : ChessboardDivision),
    board.size = 8 →
    division.length = 13 →
    ¬(is_valid_division board division) :=
sorry

end NUMINAMATH_CALUDE_chessboard_division_impossible_l2353_235347


namespace NUMINAMATH_CALUDE_cube_side_length_l2353_235352

theorem cube_side_length (s₂ : ℝ) : 
  s₂ > 0 →
  (6 * s₂^2) / (6 * 1^2) = 36 →
  s₂ = 6 := by
sorry

end NUMINAMATH_CALUDE_cube_side_length_l2353_235352


namespace NUMINAMATH_CALUDE_smallest_multiplier_for_perfect_cube_l2353_235334

def y : Nat := 2^3 * 3^5 * 4^5 * 5^4 * 6^3 * 7^5 * 8^2

def is_perfect_cube (n : Nat) : Prop :=
  ∃ m : Nat, n = m^3

theorem smallest_multiplier_for_perfect_cube :
  (∀ z < 350, ¬ is_perfect_cube (y * z)) ∧ is_perfect_cube (y * 350) :=
sorry

end NUMINAMATH_CALUDE_smallest_multiplier_for_perfect_cube_l2353_235334


namespace NUMINAMATH_CALUDE_max_sum_of_factors_max_sum_is_884_l2353_235306

theorem max_sum_of_factors (a b : ℕ+) : 
  a * b = 1764 → ∀ x y : ℕ+, x * y = 1764 → a + b ≥ x + y :=
by sorry

theorem max_sum_is_884 : 
  ∃ a b : ℕ+, a * b = 1764 ∧ a + b = 884 ∧ 
  (∀ x y : ℕ+, x * y = 1764 → x + y ≤ 884) :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_max_sum_is_884_l2353_235306


namespace NUMINAMATH_CALUDE_ellipse_focal_distances_l2353_235386

theorem ellipse_focal_distances (x y : ℝ) (P : ℝ × ℝ) (F1 F2 : ℝ × ℝ) :
  x^2 / 25 + y^2 = 1 →  -- P is on the ellipse
  P = (x, y) →  -- P's coordinates
  (∃ d : ℝ, d = 2 ∧ (Real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2) = d ∨
                     Real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2) = d)) →
  Real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2) +
  Real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2) = 10 →
  (∃ d : ℝ, d = 8 ∧ (Real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2) = d ∨
                     Real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2) = d)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_focal_distances_l2353_235386


namespace NUMINAMATH_CALUDE_base_8_calculation_l2353_235368

/-- Addition in base 8 -/
def add_base_8 (a b : ℕ) : ℕ := sorry

/-- Subtraction in base 8 -/
def sub_base_8 (a b : ℕ) : ℕ := sorry

/-- Convert a natural number to its base 8 representation -/
def to_base_8 (n : ℕ) : ℕ := sorry

/-- Convert a base 8 number to its decimal representation -/
def from_base_8 (n : ℕ) : ℕ := sorry

theorem base_8_calculation : 
  sub_base_8 (add_base_8 (from_base_8 452) (from_base_8 167)) (from_base_8 53) = from_base_8 570 := by
  sorry

end NUMINAMATH_CALUDE_base_8_calculation_l2353_235368


namespace NUMINAMATH_CALUDE_annulus_area_l2353_235328

theorem annulus_area (R r s : ℝ) (h1 : R > r) (h2 : R^2 - r^2 = s^2) :
  π * s^2 = π * R^2 - π * r^2 := by
  sorry

end NUMINAMATH_CALUDE_annulus_area_l2353_235328


namespace NUMINAMATH_CALUDE_number_exceeding_fraction_l2353_235300

theorem number_exceeding_fraction (x : ℚ) : x = (3 / 8) * x + 35 → x = 56 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeding_fraction_l2353_235300


namespace NUMINAMATH_CALUDE_representation_of_real_number_l2353_235379

theorem representation_of_real_number (x : ℝ) (hx : 0 < x ∧ x ≤ 1) :
  ∃ (n : ℕ → ℕ), 
    (∀ k, n (k + 1) / n k ∈ ({2, 3, 4} : Set ℕ)) ∧ 
    (∑' k, (1 : ℝ) / n k) = x :=
sorry

end NUMINAMATH_CALUDE_representation_of_real_number_l2353_235379


namespace NUMINAMATH_CALUDE_right_triangle_area_l2353_235384

/-- The area of a right-angled triangle with base 12 cm and height 15 cm is 90 square centimeters -/
theorem right_triangle_area : 
  ∀ (base height area : ℝ), 
  base = 12 → 
  height = 15 → 
  area = (1/2) * base * height → 
  area = 90 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2353_235384


namespace NUMINAMATH_CALUDE_competition_necessarily_laughable_l2353_235374

/-- Represents the number of questions in the math competition -/
def num_questions : ℕ := 10

/-- Represents the threshold for laughable performance -/
def laughable_threshold : ℕ := 57

/-- Represents the minimum number of students for which the performance is necessarily laughable -/
def min_laughable_students : ℕ := 253

/-- Represents a student's performance on the math competition -/
structure StudentPerformance where
  correct_answers : Finset (Fin num_questions)

/-- Represents the collective performance of students in the math competition -/
def Competition (n : ℕ) := Fin n → StudentPerformance

/-- Defines when a competition performance is laughable -/
def is_laughable (comp : Competition n) : Prop :=
  ∃ (i j : Fin num_questions), i ≠ j ∧
    (∃ (students : Finset (Fin n)), students.card = laughable_threshold ∧
      (∀ s ∈ students, (i ∈ (comp s).correct_answers ∧ j ∈ (comp s).correct_answers) ∨
                       (i ∉ (comp s).correct_answers ∧ j ∉ (comp s).correct_answers)))

/-- The main theorem: any competition with at least min_laughable_students is necessarily laughable -/
theorem competition_necessarily_laughable (n : ℕ) (h : n ≥ min_laughable_students) :
  ∀ (comp : Competition n), is_laughable comp :=
sorry

end NUMINAMATH_CALUDE_competition_necessarily_laughable_l2353_235374


namespace NUMINAMATH_CALUDE_work_completion_time_l2353_235330

theorem work_completion_time (b : ℝ) (c : ℝ) (d : ℝ) (h1 : b = 14) (h2 : c = 2) (h3 : d = 5.000000000000001) : 
  ∃ a : ℝ, a = 4 ∧ 
  (c * (1 / a + 1 / b) + d * (1 / b) = 1) :=
sorry

end NUMINAMATH_CALUDE_work_completion_time_l2353_235330


namespace NUMINAMATH_CALUDE_celias_savings_l2353_235367

def weeks : ℕ := 4
def food_budget_per_week : ℕ := 100
def rent : ℕ := 1500
def streaming_cost : ℕ := 30
def phone_cost : ℕ := 50
def savings_rate : ℚ := 1 / 10

def total_spending : ℕ := weeks * food_budget_per_week + rent + streaming_cost + phone_cost

def savings_amount : ℚ := (total_spending : ℚ) * savings_rate

theorem celias_savings : savings_amount = 198 := by
  sorry

end NUMINAMATH_CALUDE_celias_savings_l2353_235367


namespace NUMINAMATH_CALUDE_return_flight_theorem_l2353_235361

/-- Represents a direction in degrees relative to a cardinal direction -/
structure Direction where
  angle : ℝ
  cardinal : String
  relative : String

/-- Represents a flight path -/
structure FlightPath where
  distance : ℝ
  direction : Direction

/-- Returns the opposite direction for a given flight path -/
def oppositeDirection (fp : FlightPath) : Direction :=
  { angle := fp.direction.angle,
    cardinal := if fp.direction.cardinal = "east" then "west" else "east",
    relative := if fp.direction.relative = "south" then "north" else "south" }

theorem return_flight_theorem (outbound : FlightPath) 
  (h1 : outbound.distance = 1200)
  (h2 : outbound.direction.angle = 30)
  (h3 : outbound.direction.cardinal = "east")
  (h4 : outbound.direction.relative = "south") :
  ∃ (inbound : FlightPath),
    inbound.distance = outbound.distance ∧
    inbound.direction = oppositeDirection outbound :=
  sorry

end NUMINAMATH_CALUDE_return_flight_theorem_l2353_235361


namespace NUMINAMATH_CALUDE_current_speed_l2353_235338

/-- Proves that the speed of the current is 20 kmph, given the boat's speed in still water and upstream. -/
theorem current_speed (boat_still_speed upstream_speed : ℝ) 
  (h1 : boat_still_speed = 50)
  (h2 : upstream_speed = 30) :
  boat_still_speed - upstream_speed = 20 := by
  sorry

#check current_speed

end NUMINAMATH_CALUDE_current_speed_l2353_235338


namespace NUMINAMATH_CALUDE_solutions_periodic_l2353_235315

/-- A system of differential equations with given initial conditions -/
structure DiffSystem where
  f : ℝ → ℝ  -- y = f(x)
  g : ℝ → ℝ  -- z = g(x)
  eqn1 : ∀ x, deriv f x = -(g x)^3
  eqn2 : ∀ x, deriv g x = (f x)^3
  init1 : f 0 = 1
  init2 : g 0 = 0
  unique : ∀ f' g', (∀ x, deriv f' x = -(g' x)^3) →
                    (∀ x, deriv g' x = (f' x)^3) →
                    f' 0 = 1 → g' 0 = 0 →
                    f' = f ∧ g' = g

/-- Definition of a periodic function -/
def Periodic (f : ℝ → ℝ) :=
  ∃ k : ℝ, k > 0 ∧ ∀ x, f (x + k) = f x

/-- The main theorem stating that solutions are periodic with the same period -/
theorem solutions_periodic (sys : DiffSystem) :
  ∃ k : ℝ, k > 0 ∧ Periodic sys.f ∧ Periodic sys.g ∧
  ∀ x, sys.f (x + k) = sys.f x ∧ sys.g (x + k) = sys.g x :=
sorry

end NUMINAMATH_CALUDE_solutions_periodic_l2353_235315


namespace NUMINAMATH_CALUDE_exists_composite_carmichael_number_l2353_235373

theorem exists_composite_carmichael_number : ∃ n : ℕ, 
  n > 1 ∧ 
  ¬ Nat.Prime n ∧ 
  ∀ a : ℤ, (n : ℤ) ∣ (a^n - a) := by
  sorry

end NUMINAMATH_CALUDE_exists_composite_carmichael_number_l2353_235373


namespace NUMINAMATH_CALUDE_right_triangle_from_parabolas_l2353_235320

theorem right_triangle_from_parabolas (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hac : a ≠ c)
  (h_intersect : ∃ x₀ : ℝ, x₀ ≠ 0 ∧ x₀^2 + 2*a*x₀ + b^2 = 0 ∧ x₀^2 + 2*c*x₀ - b^2 = 0) :
  a^2 = b^2 + c^2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_from_parabolas_l2353_235320


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l2353_235378

theorem quadratic_solution_sum (p q : ℝ) : 
  (∀ x : ℂ, (2 * x^2 + 5 = 7 * x - 2) ↔ (x = p + q * I ∨ x = p - q * I)) →
  p + q^2 = 35/16 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l2353_235378


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l2353_235313

theorem fraction_equals_zero (x : ℝ) : (2 * x - 6) / (5 * x + 10) = 0 ↔ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l2353_235313


namespace NUMINAMATH_CALUDE_max_side_length_of_triangle_l2353_235372

theorem max_side_length_of_triangle (a b c : ℕ) : 
  a < b → b < c →  -- Three different integer side lengths
  a + b + c = 24 → -- Perimeter is 24 units
  a + b > c →      -- Triangle inequality
  b + c > a →      -- Triangle inequality
  a + c > b →      -- Triangle inequality
  c ≤ 11 :=        -- Maximum length of any side is 11
by sorry

end NUMINAMATH_CALUDE_max_side_length_of_triangle_l2353_235372


namespace NUMINAMATH_CALUDE_ring_area_equals_three_circles_l2353_235344

theorem ring_area_equals_three_circles 
  (r₁ r₂ r₃ d R r : ℝ) (h_positive : r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0 ∧ d > 0) :
  (R^2 - r^2 = r₁^2 + r₂^2 + r₃^2) ∧ (R - r = d) →
  (R = ((r₁^2 + r₂^2 + r₃^2) + d^2) / (2*d)) ∧ (r = R - d) := by
sorry

end NUMINAMATH_CALUDE_ring_area_equals_three_circles_l2353_235344


namespace NUMINAMATH_CALUDE_rational_sum_squares_l2353_235383

theorem rational_sum_squares (a b c : ℚ) :
  1 / (b - c)^2 + 1 / (c - a)^2 + 1 / (a - b)^2 = (1 / (a - b) + 1 / (b - c) + 1 / (c - a))^2 :=
by sorry

end NUMINAMATH_CALUDE_rational_sum_squares_l2353_235383


namespace NUMINAMATH_CALUDE_max_value_of_expression_l2353_235343

theorem max_value_of_expression (x : ℝ) :
  (4 * x^2 + 8 * x + 19) / (4 * x^2 + 8 * x + 9) ≤ 3 ∧
  ∀ ε > 0, ∃ y : ℝ, (4 * y^2 + 8 * y + 19) / (4 * y^2 + 8 * y + 9) > 3 - ε :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l2353_235343


namespace NUMINAMATH_CALUDE_siding_total_cost_l2353_235377

def wall_width : ℝ := 10
def wall_height : ℝ := 7
def roof_width : ℝ := 10
def roof_height : ℝ := 6
def roof_sections : ℕ := 2
def siding_width : ℝ := 10
def siding_height : ℝ := 15
def siding_cost : ℝ := 35

theorem siding_total_cost :
  let total_area := wall_width * wall_height + roof_width * roof_height * roof_sections
  let siding_area := siding_width * siding_height
  let sections_needed := Int.ceil (total_area / siding_area)
  sections_needed * siding_cost = 70 := by sorry

end NUMINAMATH_CALUDE_siding_total_cost_l2353_235377


namespace NUMINAMATH_CALUDE_marble_problem_l2353_235395

theorem marble_problem (R : ℕ) : 
  (1 - (1 - 10 / (R + 16)) ^ 2 = 3/4) → R = 4 := by
  sorry

end NUMINAMATH_CALUDE_marble_problem_l2353_235395


namespace NUMINAMATH_CALUDE_triangle_area_l2353_235359

/-- The area of a triangle with sides 10, 24, and 26 is 120 square units -/
theorem triangle_area : ∀ (a b c : ℝ),
  a = 10 ∧ b = 24 ∧ c = 26 →
  (∃ (s : ℝ), s = (a + b + c) / 2 ∧ 
   Real.sqrt (s * (s - a) * (s - b) * (s - c)) = 120) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l2353_235359


namespace NUMINAMATH_CALUDE_prime_value_problem_l2353_235358

theorem prime_value_problem : ∃ p : ℕ, 
  Prime p ∧ 
  (5 * p) % 4 = 3 ∧ 
  Prime (13 * p + 2) ∧ 
  13 * p + 2 = 41 := by
sorry

end NUMINAMATH_CALUDE_prime_value_problem_l2353_235358


namespace NUMINAMATH_CALUDE_sum_200_consecutive_integers_l2353_235353

theorem sum_200_consecutive_integers (n : ℕ) : 
  (n = 2000200000 ∨ n = 3000300000 ∨ n = 4000400000 ∨ n = 5000500000 ∨ n = 6000600000) →
  ¬∃ k : ℕ, n = (200 * (k + 100)) + 10050 := by
  sorry

end NUMINAMATH_CALUDE_sum_200_consecutive_integers_l2353_235353


namespace NUMINAMATH_CALUDE_abhay_sameer_speed_comparison_l2353_235302

theorem abhay_sameer_speed_comparison 
  (distance : ℝ) 
  (abhay_speed : ℝ) 
  (time_difference : ℝ) :
  distance = 42 →
  abhay_speed = 7 →
  time_difference = 2 →
  distance / abhay_speed = distance / (distance / (distance / abhay_speed - time_difference)) + time_difference →
  distance / (2 * abhay_speed) = (distance / (distance / (distance / abhay_speed - time_difference))) - 1 :=
by sorry

end NUMINAMATH_CALUDE_abhay_sameer_speed_comparison_l2353_235302


namespace NUMINAMATH_CALUDE_scientific_notation_of_0_00000065_l2353_235356

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_0_00000065 :
  toScientificNotation 0.00000065 = ScientificNotation.mk 6.5 (-7) sorry := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_0_00000065_l2353_235356


namespace NUMINAMATH_CALUDE_tangent_and_max_chord_length_l2353_235331

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the point M
def point_M (a : ℝ) : ℝ × ℝ := (1, a)

theorem tangent_and_max_chord_length :
  -- Part I: Point M is on the circle if and only if a = ±√3
  (∃ a : ℝ, circle_O (point_M a).1 (point_M a).2 ↔ a = Real.sqrt 3 ∨ a = -Real.sqrt 3) ∧
  -- Part II: Maximum value of |AC| + |BD| is 2√10
  (let a : ℝ := Real.sqrt 2
   ∀ A B C D : ℝ × ℝ,
   circle_O A.1 A.2 →
   circle_O B.1 B.2 →
   circle_O C.1 C.2 →
   circle_O D.1 D.2 →
   (A.1 - C.1) * (B.1 - D.1) + (A.2 - C.2) * (B.2 - D.2) = 0 →  -- AC ⊥ BD
   (point_M a).1 = (A.1 + C.1) / 2 →  -- M is midpoint of AC
   (point_M a).1 = (B.1 + D.1) / 2 →  -- M is midpoint of BD
   (point_M a).2 = (A.2 + C.2) / 2 →  -- M is midpoint of AC
   (point_M a).2 = (B.2 + D.2) / 2 →  -- M is midpoint of BD
   Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) + Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) ≤ 2 * Real.sqrt 10) := by
sorry

end NUMINAMATH_CALUDE_tangent_and_max_chord_length_l2353_235331


namespace NUMINAMATH_CALUDE_fraction_inequality_solution_set_l2353_235311

theorem fraction_inequality_solution_set (x : ℝ) (h : x ≠ 0) :
  1 / x ≤ 1 ↔ x ∈ Set.Ioo 0 1 ∪ Set.Ici 1 := by sorry

end NUMINAMATH_CALUDE_fraction_inequality_solution_set_l2353_235311


namespace NUMINAMATH_CALUDE_largest_root_ratio_l2353_235336

-- Define the polynomials
def f (x : ℝ) : ℝ := 1 - x - 4*x^2 + x^4
def g (x : ℝ) : ℝ := 16 - 8*x - 16*x^2 + x^4

-- Define x₁ as the largest root of f
def x₁ : ℝ := sorry

-- Define x₂ as the largest root of g
def x₂ : ℝ := sorry

-- Theorem statement
theorem largest_root_ratio :
  x₂ / x₁ = 2 :=
sorry

end NUMINAMATH_CALUDE_largest_root_ratio_l2353_235336


namespace NUMINAMATH_CALUDE_intersection_equals_B_l2353_235381

/-- The set A of solutions to x^2 - 4x + 3 = 0 -/
def A : Set ℝ := {x | x^2 - 4*x + 3 = 0}

/-- The set B of solutions to mx + 1 = 0 for some real m -/
def B (m : ℝ) : Set ℝ := {x | m*x + 1 = 0}

/-- The theorem stating the set of values for m that satisfy A ∩ B = B -/
theorem intersection_equals_B : 
  {m : ℝ | A ∩ B m = B m} = {-1, -1/3, 0} := by sorry

end NUMINAMATH_CALUDE_intersection_equals_B_l2353_235381


namespace NUMINAMATH_CALUDE_intersection_point_with_median_line_l2353_235393

open Complex

/-- Given complex numbers and a curve, prove the intersection point with the median line -/
theorem intersection_point_with_median_line 
  (a b c : ℝ) 
  (z₁₁ : ℂ) 
  (z₁ : ℂ) 
  (z₂ : ℂ) 
  (h_z₁₁ : z₁₁ = Complex.I * a) 
  (h_z₁ : z₁ = (1/2 : ℝ) + Complex.I * b) 
  (h_z₂ : z₂ = 1 + Complex.I * c) 
  (h_non_collinear : a + c ≠ 2 * b) 
  (z : ℝ → ℂ) 
  (h_z : ∀ t, z t = z₁ * (cos t)^4 + 2 * z₁ * (cos t)^2 * (sin t)^2 + z₂ * (sin t)^4) :
  ∃! p : ℂ, p ∈ Set.range z ∧ 
    p.re = (1/2 : ℝ) ∧ 
    p.im = (a + c + 2*b) / 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_with_median_line_l2353_235393


namespace NUMINAMATH_CALUDE_average_divisible_by_six_l2353_235318

theorem average_divisible_by_six : ∃ (S : Finset ℕ),
  (∀ n ∈ S, 7 < n ∧ n ≤ 49 ∧ 6 ∣ n) ∧
  (∀ n, 7 < n → n ≤ 49 → 6 ∣ n → n ∈ S) ∧
  (S.sum id / S.card : ℚ) = 30 := by
sorry

end NUMINAMATH_CALUDE_average_divisible_by_six_l2353_235318


namespace NUMINAMATH_CALUDE_certain_number_proof_l2353_235310

theorem certain_number_proof (p q : ℝ) 
  (h1 : 3 / p = 8) 
  (h2 : p - q = 0.20833333333333334) : 
  3 / q = 18 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2353_235310


namespace NUMINAMATH_CALUDE_basketball_shots_mode_and_median_l2353_235397

def data_set : List Nat := [6, 7, 6, 9, 8]

def mode (l : List Nat) : Nat := sorry

def median (l : List Nat) : Nat := sorry

theorem basketball_shots_mode_and_median :
  mode data_set = 6 ∧ median data_set = 7 := by sorry

end NUMINAMATH_CALUDE_basketball_shots_mode_and_median_l2353_235397


namespace NUMINAMATH_CALUDE_units_digit_of_seven_power_l2353_235329

theorem units_digit_of_seven_power (n : ℕ) : 7^(6^5) ≡ 1 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_seven_power_l2353_235329


namespace NUMINAMATH_CALUDE_ordering_of_logarithms_and_exponential_l2353_235362

theorem ordering_of_logarithms_and_exponential : 
  let a := Real.log 3 / Real.log 5
  let b := Real.log 8 / Real.log 13
  let c := Real.exp (-1/2)
  c < a ∧ a < b :=
by sorry

end NUMINAMATH_CALUDE_ordering_of_logarithms_and_exponential_l2353_235362


namespace NUMINAMATH_CALUDE_difference_of_squares_consecutive_evens_l2353_235321

def consecutive_even_integers (a b c : ℤ) : Prop :=
  b = a + 2 ∧ c = b + 2

theorem difference_of_squares_consecutive_evens (a b c : ℤ) :
  consecutive_even_integers a b c →
  a + b + c = 1992 →
  c^2 - a^2 = 5312 :=
by sorry

end NUMINAMATH_CALUDE_difference_of_squares_consecutive_evens_l2353_235321


namespace NUMINAMATH_CALUDE_new_average_weight_l2353_235351

def original_players : ℕ := 7
def original_average_weight : ℝ := 112
def new_player1_weight : ℝ := 110
def new_player2_weight : ℝ := 60

theorem new_average_weight :
  let total_original_weight := original_players * original_average_weight
  let total_new_weight := total_original_weight + new_player1_weight + new_player2_weight
  let new_total_players := original_players + 2
  (total_new_weight / new_total_players : ℝ) = 106 := by
  sorry

end NUMINAMATH_CALUDE_new_average_weight_l2353_235351


namespace NUMINAMATH_CALUDE_not_integer_proofs_l2353_235332

theorem not_integer_proofs (a b c d : ℝ) (n : ℕ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) : 
  let E := (a/(a+b+d)) + (b/(b+c+a)) + (c/(c+d+b)) + (d/(d+a+c))
  (1 < E ∧ E < 2) ∧ (n < Real.sqrt (n^2 + n) ∧ Real.sqrt (n^2 + n) < n + 1) := by
  sorry

end NUMINAMATH_CALUDE_not_integer_proofs_l2353_235332


namespace NUMINAMATH_CALUDE_remainder_problem_l2353_235354

theorem remainder_problem (x y z w : ℕ) 
  (hx : 4 ∣ x) (hy : 4 ∣ y) (hz : 4 ∣ z) (hw : 3 ∣ w) (hpos_x : x > 0) (hpos_y : y > 0) (hpos_z : z > 0) :
  (x^2 * (y*w + z*(x + y)^2) + 7) % 6 = 1 :=
sorry

end NUMINAMATH_CALUDE_remainder_problem_l2353_235354


namespace NUMINAMATH_CALUDE_system_solution_l2353_235341

theorem system_solution (a b : ℤ) :
  (b * (-1) + 2 * 2 = 8) →
  (a * 1 + 3 * 4 = 5) →
  (a = -7 ∧ b = -4) ∧
  ((-7) * 7 + 3 * 18 = 5) ∧
  ((-4) * 7 + 2 * 18 = 8) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2353_235341


namespace NUMINAMATH_CALUDE_max_value_of_exponential_difference_l2353_235304

theorem max_value_of_exponential_difference : 
  ∃ (M : ℝ), M = 1/4 ∧ ∀ (x : ℝ), 2^x - 16^x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_exponential_difference_l2353_235304


namespace NUMINAMATH_CALUDE_square_diagonal_length_l2353_235319

theorem square_diagonal_length (side_length : ℝ) (h : side_length = 100 * Real.sqrt 3) :
  Real.sqrt (2 * side_length ^ 2) = 100 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_square_diagonal_length_l2353_235319


namespace NUMINAMATH_CALUDE_product_49_sum_14_l2353_235380

theorem product_49_sum_14 (a b c d : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a * b * c * d = 49 →
  a + b + c + d = 14 :=
by sorry

end NUMINAMATH_CALUDE_product_49_sum_14_l2353_235380


namespace NUMINAMATH_CALUDE_max_non_managers_proof_l2353_235324

structure Department where
  name : String
  managers : ℕ
  ratio_managers : ℕ
  ratio_non_managers : ℕ
  active_projects : ℕ

def calculate_non_managers (d : Department) : ℕ :=
  d.managers * d.ratio_non_managers / d.ratio_managers +
  (d.active_projects + 2) / 3 +
  2

def total_non_managers (departments : List Department) : ℕ :=
  departments.foldl (fun acc d => acc + calculate_non_managers d) 0

theorem max_non_managers_proof :
  let departments : List Department := [
    { name := "Marketing", managers := 9, ratio_managers := 9, ratio_non_managers := 38, active_projects := 6 },
    { name := "HR", managers := 5, ratio_managers := 5, ratio_non_managers := 23, active_projects := 4 },
    { name := "Finance", managers := 6, ratio_managers := 6, ratio_non_managers := 31, active_projects := 5 }
  ]
  total_non_managers departments = 104 :=
by sorry

end NUMINAMATH_CALUDE_max_non_managers_proof_l2353_235324


namespace NUMINAMATH_CALUDE_discount_equation_l2353_235327

theorem discount_equation (original_price final_price : ℝ) (x : ℝ) 
  (h1 : original_price = 200)
  (h2 : final_price = 164)
  (h3 : final_price = original_price * (1 - x)^2) :
  200 * (1 - x)^2 = 164 := by
  sorry

end NUMINAMATH_CALUDE_discount_equation_l2353_235327


namespace NUMINAMATH_CALUDE_right_triangle_leg_length_l2353_235322

/-- Proves that in a right triangle with an area of 800 square feet and one leg of 40 feet, 
    the length of the other leg is also 40 feet. -/
theorem right_triangle_leg_length 
  (area : ℝ) 
  (base : ℝ) 
  (h : area = 800) 
  (b : base = 40) : 
  (2 * area) / base = 40 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_leg_length_l2353_235322


namespace NUMINAMATH_CALUDE_intersection_A_B_intersection_C_R_A_B_l2353_235363

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 10}

-- Define the complement of A in ℝ
def C_R_A : Set ℝ := {x : ℝ | x < 3 ∨ x ≥ 7}

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {x : ℝ | 3 ≤ x ∧ x < 7} := by sorry

-- Theorem for (C_R A) ∩ B
theorem intersection_C_R_A_B : (C_R_A ∩ B) = {x : ℝ | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_intersection_C_R_A_B_l2353_235363


namespace NUMINAMATH_CALUDE_square_sum_product_l2353_235307

theorem square_sum_product (a b : ℝ) (h1 : a + b = -3) (h2 : a * b = 2) :
  a^2 * b + a * b^2 = -6 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_product_l2353_235307


namespace NUMINAMATH_CALUDE_bicycle_spokes_theorem_l2353_235399

/-- The number of spokes on each bicycle wheel given the total number of bicycles and spokes -/
def spokes_per_wheel (num_bicycles : ℕ) (total_spokes : ℕ) : ℕ :=
  total_spokes / (num_bicycles * 2)

/-- Theorem stating that 4 bicycles with a total of 80 spokes have 10 spokes per wheel -/
theorem bicycle_spokes_theorem :
  spokes_per_wheel 4 80 = 10 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_spokes_theorem_l2353_235399


namespace NUMINAMATH_CALUDE_cylinder_volume_equals_cube_surface_l2353_235364

/-- The volume of a cylinder with surface area equal to a cube of side length 4 and height equal to its diameter --/
theorem cylinder_volume_equals_cube_surface (π : ℝ) (h : π > 0) : 
  ∃ (r : ℝ), r > 0 ∧ 
  6 * π * r^2 = 96 ∧ 
  π * r^2 * (2 * r) = 128 * Real.sqrt 2 / π :=
sorry

end NUMINAMATH_CALUDE_cylinder_volume_equals_cube_surface_l2353_235364


namespace NUMINAMATH_CALUDE_eccentricity_of_hyperbola_with_diagonal_asymptotes_l2353_235312

/-- A hyperbola with given asymptotes -/
structure Hyperbola where
  -- Asymptotes of the hyperbola are y = ±x
  asymptotes : (ℝ → ℝ) × (ℝ → ℝ)
  asymptotes_prop : asymptotes = ((fun x => x), (fun x => -x))

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- Theorem: The eccentricity of a hyperbola with asymptotes y = ±x is √2 -/
theorem eccentricity_of_hyperbola_with_diagonal_asymptotes (h : Hyperbola) :
  eccentricity h = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_eccentricity_of_hyperbola_with_diagonal_asymptotes_l2353_235312


namespace NUMINAMATH_CALUDE_optimal_purchase_l2353_235389

/-- Represents the cost and quantity of soccer balls and basketballs --/
structure BallPurchase where
  soccer_price : ℝ
  basketball_price : ℝ
  soccer_quantity : ℕ
  basketball_quantity : ℕ

/-- Defines the conditions of the ball purchase problem --/
def valid_purchase (p : BallPurchase) : Prop :=
  p.soccer_price + 3 * p.basketball_price = 275 ∧
  3 * p.soccer_price + 2 * p.basketball_price = 300 ∧
  p.soccer_quantity + p.basketball_quantity = 80 ∧
  p.soccer_quantity ≤ 3 * p.basketball_quantity

/-- Calculates the total cost of a ball purchase --/
def total_cost (p : BallPurchase) : ℝ :=
  p.soccer_price * p.soccer_quantity + p.basketball_price * p.basketball_quantity

/-- Theorem stating the most cost-effective purchase plan --/
theorem optimal_purchase :
  ∃ (p : BallPurchase),
    valid_purchase p ∧
    p.soccer_price = 50 ∧
    p.basketball_price = 75 ∧
    p.soccer_quantity = 60 ∧
    p.basketball_quantity = 20 ∧
    (∀ (q : BallPurchase), valid_purchase q → total_cost p ≤ total_cost q) :=
  sorry

end NUMINAMATH_CALUDE_optimal_purchase_l2353_235389


namespace NUMINAMATH_CALUDE_defective_books_relative_frequency_l2353_235339

/-- The relative frequency of an event is the ratio of the number of times 
    the event occurs to the total number of trials or experiments. -/
def relative_frequency (event_occurrences : ℕ) (total_trials : ℕ) : ℚ :=
  event_occurrences / total_trials

/-- Given a batch of 100 randomly selected books with 5 defective books,
    prove that the relative frequency of defective books is 0.05. -/
theorem defective_books_relative_frequency :
  let total_books : ℕ := 100
  let defective_books : ℕ := 5
  relative_frequency defective_books total_books = 5 / 100 := by
  sorry

#eval (5 : ℚ) / 100  -- To verify the result is indeed 0.05

end NUMINAMATH_CALUDE_defective_books_relative_frequency_l2353_235339


namespace NUMINAMATH_CALUDE_sum_of_38_and_twice_43_l2353_235333

theorem sum_of_38_and_twice_43 : 38 + 2 * 43 = 124 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_38_and_twice_43_l2353_235333


namespace NUMINAMATH_CALUDE_correct_minus_position_l2353_235391

def numbers : List ℕ := [6, 9, 12, 15, 18, 21]

def place_signs (nums : List ℕ) (minus_pos : ℕ) : ℤ :=
  (nums.take minus_pos).sum - nums[minus_pos]! + (nums.drop (minus_pos + 1)).sum

theorem correct_minus_position (nums : List ℕ) (h : nums = numbers) :
  ∃! pos : ℕ, pos < nums.length - 1 ∧ place_signs nums pos = 45 :=
by sorry

end NUMINAMATH_CALUDE_correct_minus_position_l2353_235391


namespace NUMINAMATH_CALUDE_min_xyz_value_l2353_235349

/-- Given real numbers x, y, z satisfying the given conditions, 
    the minimum value of xyz is 9√11 - 32 -/
theorem min_xyz_value (x y z : ℝ) 
    (h1 : x * y + 2 * z = 1) 
    (h2 : x^2 + y^2 + z^2 = 5) : 
  ∀ (a b c : ℝ), a * b + 2 * c = 1 → a^2 + b^2 + c^2 = 5 → 
    x * y * z ≤ a * b * c ∧ 
    ∃ (x₀ y₀ z₀ : ℝ), x₀ * y₀ + 2 * z₀ = 1 ∧ x₀^2 + y₀^2 + z₀^2 = 5 ∧ 
      x₀ * y₀ * z₀ = 9 * Real.sqrt 11 - 32 :=
by
  sorry

#check min_xyz_value

end NUMINAMATH_CALUDE_min_xyz_value_l2353_235349


namespace NUMINAMATH_CALUDE_quadratic_equation_q_value_l2353_235326

theorem quadratic_equation_q_value : ∀ (p q : ℝ),
  (∃ x : ℝ, 3 * x^2 + p * x + q = 0 ∧ x = -3) →
  (∃ x₁ x₂ : ℝ, 3 * x₁^2 + p * x₁ + q = 0 ∧ 3 * x₂^2 + p * x₂ + q = 0 ∧ x₁ + x₂ = -2) →
  q = -9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_q_value_l2353_235326


namespace NUMINAMATH_CALUDE_sixteen_squares_covered_l2353_235394

/-- Represents a square on the checkerboard -/
structure Square where
  x : Int
  y : Int

/-- Represents the circular disc -/
structure Disc where
  diameter : ℝ
  center : Square

/-- Represents the checkerboard -/
structure Checkerboard where
  size : Nat
  squares : List Square

/-- Checks if a square is completely covered by the disc -/
def is_covered (s : Square) (d : Disc) : Bool :=
  sorry

/-- Counts the number of squares completely covered by the disc -/
def count_covered_squares (cb : Checkerboard) (d : Disc) : Nat :=
  sorry

/-- Main theorem: 16 squares are completely covered -/
theorem sixteen_squares_covered (cb : Checkerboard) (d : Disc) : 
  cb.size = 6 → d.diameter = 2 → count_covered_squares cb d = 16 := by
  sorry

end NUMINAMATH_CALUDE_sixteen_squares_covered_l2353_235394


namespace NUMINAMATH_CALUDE_three_distinct_solutions_l2353_235337

theorem three_distinct_solutions : ∃ (x₁ x₂ x₃ : ℝ), 
  (356 * x₁ = 2492) ∧ 
  (x₂ / 39 = 235) ∧ 
  (1908 - x₃ = 529) ∧ 
  (x₁ ≠ x₂) ∧ (x₁ ≠ x₃) ∧ (x₂ ≠ x₃) :=
by sorry

end NUMINAMATH_CALUDE_three_distinct_solutions_l2353_235337


namespace NUMINAMATH_CALUDE_bullseye_value_l2353_235375

/-- 
Given a dart game with the following conditions:
- Three darts are thrown
- One dart is a bullseye worth B points
- One dart completely misses (0 points)
- One dart is worth half the bullseye points
- The total score is 75 points

Prove that the bullseye is worth 50 points
-/
theorem bullseye_value (B : ℝ) 
  (total_score : B + 0 + B/2 = 75) : 
  B = 50 := by
  sorry

end NUMINAMATH_CALUDE_bullseye_value_l2353_235375


namespace NUMINAMATH_CALUDE_total_cases_giving_one_card_l2353_235301

def blue_cards : ℕ := 3
def yellow_cards : ℕ := 5

theorem total_cases_giving_one_card : blue_cards + yellow_cards = 8 := by
  sorry

end NUMINAMATH_CALUDE_total_cases_giving_one_card_l2353_235301


namespace NUMINAMATH_CALUDE_triangle_side_difference_range_l2353_235369

theorem triangle_side_difference_range (A B C a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Side lengths are positive
  a = 1 ∧  -- Given condition
  C - B = π / 2 ∧  -- Given condition
  a / Real.sin A = b / Real.sin B ∧  -- Law of Sines
  b / Real.sin B = c / Real.sin C  -- Law of Sines
  → Real.sqrt 2 / 2 < c - b ∧ c - b < 1 := by sorry

end NUMINAMATH_CALUDE_triangle_side_difference_range_l2353_235369


namespace NUMINAMATH_CALUDE_probability_two_girls_chosen_l2353_235360

-- Define the total number of members
def total_members : ℕ := 12

-- Define the number of girls
def num_girls : ℕ := 6

-- Define the number of boys
def num_boys : ℕ := 6

-- Define a function to calculate combinations
def combination (n k : ℕ) : ℕ := (n.factorial) / (k.factorial * (n - k).factorial)

-- Theorem statement
theorem probability_two_girls_chosen :
  (combination num_girls 2 : ℚ) / (combination total_members 2) = 5 / 22 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_girls_chosen_l2353_235360


namespace NUMINAMATH_CALUDE_ivy_room_spiders_l2353_235314

/-- Given the total number of spider legs in a room, calculate the number of spiders. -/
def spiders_in_room (total_legs : ℕ) : ℕ :=
  total_legs / 8

/-- Theorem: There are 4 spiders in Ivy's room given 32 total spider legs. -/
theorem ivy_room_spiders : spiders_in_room 32 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ivy_room_spiders_l2353_235314


namespace NUMINAMATH_CALUDE_ratio_to_nine_l2353_235390

/-- Given a ratio of 5:1 and a number 9, prove that the number x which satisfies this ratio is 45. -/
theorem ratio_to_nine : ∃ x : ℚ, (5 : ℚ) / 1 = x / 9 ∧ x = 45 := by
  sorry

end NUMINAMATH_CALUDE_ratio_to_nine_l2353_235390


namespace NUMINAMATH_CALUDE_largest_constant_inequality_l2353_235317

theorem largest_constant_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ (K : ℝ), K = Real.sqrt 3 ∧ 
  (∀ (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0),
    Real.sqrt (x * y / z) + Real.sqrt (y * z / x) + Real.sqrt (x * z / y) ≥ K * Real.sqrt (x + y + z)) ∧
  (∀ (L : ℝ), 
    (∀ (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0),
      Real.sqrt (x * y / z) + Real.sqrt (y * z / x) + Real.sqrt (x * z / y) ≥ L * Real.sqrt (x + y + z)) →
    L ≤ K) :=
by sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_l2353_235317


namespace NUMINAMATH_CALUDE_optimal_price_maximizes_profit_l2353_235348

/-- Represents the profit function for a product with given pricing and sales conditions -/
def profit_function (x : ℝ) : ℝ :=
  (x - 8) * (100 - (x - 10) * 10)

/-- The selling price that maximizes profit -/
def optimal_price : ℝ := 14

theorem optimal_price_maximizes_profit :
  ∀ (x : ℝ), profit_function x ≤ profit_function optimal_price :=
sorry

#check optimal_price_maximizes_profit

end NUMINAMATH_CALUDE_optimal_price_maximizes_profit_l2353_235348


namespace NUMINAMATH_CALUDE_sin_cos_sum_l2353_235325

theorem sin_cos_sum (θ : Real) (h : Real.sin θ * Real.cos θ = 1/8) :
  Real.sin θ + Real.cos θ = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_l2353_235325


namespace NUMINAMATH_CALUDE_max_b_letters_l2353_235346

/-- The maximum number of "B" letters that can be formed with 47 sticks -/
theorem max_b_letters (total_sticks : ℕ) (sticks_per_b : ℕ) (sticks_per_v : ℕ)
  (h_total : total_sticks = 47)
  (h_b : sticks_per_b = 4)
  (h_v : sticks_per_v = 5)
  (h_all_used : ∃ (b v : ℕ), total_sticks = b * sticks_per_b + v * sticks_per_v) :
  ∃ (max_b : ℕ), 
    (max_b * sticks_per_b ≤ total_sticks) ∧ 
    (∀ b : ℕ, b * sticks_per_b ≤ total_sticks → b ≤ max_b) ∧
    (∃ v : ℕ, total_sticks = max_b * sticks_per_b + v * sticks_per_v) ∧
    max_b = 8 :=
sorry

end NUMINAMATH_CALUDE_max_b_letters_l2353_235346


namespace NUMINAMATH_CALUDE_casper_candy_problem_l2353_235365

def candy_sequence (initial : ℕ) : List ℕ :=
  let day1 := initial / 2 - 3
  let day2 := day1 / 2 - 5
  let day3 := day2 / 2 - 2
  let day4 := day3 / 2
  [initial, day1, day2, day3, day4]

theorem casper_candy_problem (initial : ℕ) :
  candy_sequence initial = [initial, initial / 2 - 3, (initial / 2 - 3) / 2 - 5, ((initial / 2 - 3) / 2 - 5) / 2 - 2, 10] →
  initial = 122 := by
  sorry

end NUMINAMATH_CALUDE_casper_candy_problem_l2353_235365


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2353_235316

theorem negation_of_proposition :
  ¬(∀ x y : ℤ, Even (x + y) → (Even x ∧ Even y)) ↔
  (∀ x y : ℤ, ¬Even (x + y) → ¬(Even x ∧ Even y)) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2353_235316


namespace NUMINAMATH_CALUDE_buckingham_palace_visitor_difference_l2353_235355

/-- The number of visitors to Buckingham Palace on the previous day -/
def previous_day_visitors : ℕ := 100

/-- The number of visitors to Buckingham Palace on that day -/
def that_day_visitors : ℕ := 666

/-- The difference in visitors between that day and the previous day -/
def visitor_difference : ℕ := that_day_visitors - previous_day_visitors

theorem buckingham_palace_visitor_difference :
  visitor_difference = 566 :=
by sorry

end NUMINAMATH_CALUDE_buckingham_palace_visitor_difference_l2353_235355


namespace NUMINAMATH_CALUDE_children_tickets_sold_l2353_235308

theorem children_tickets_sold (adult_price child_price total_tickets total_revenue : ℚ)
  (h1 : adult_price = 6)
  (h2 : child_price = 9/2)
  (h3 : total_tickets = 400)
  (h4 : total_revenue = 2100)
  (h5 : ∃ (adult_tickets child_tickets : ℚ),
    adult_tickets + child_tickets = total_tickets ∧
    adult_price * adult_tickets + child_price * child_tickets = total_revenue) :
  ∃ (child_tickets : ℚ), child_tickets = 200 := by
  sorry

end NUMINAMATH_CALUDE_children_tickets_sold_l2353_235308


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_equation_l2353_235342

theorem sum_of_roots_quadratic_equation :
  ∀ x₁ x₂ : ℝ, (x₁^2 - 3*x₁ + 2 = 0) ∧ (x₂^2 - 3*x₂ + 2 = 0) → x₁ + x₂ = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_equation_l2353_235342


namespace NUMINAMATH_CALUDE_parabola_transformation_l2353_235366

/-- Represents a parabola in the form y = a(x-h)^2 + k --/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Shifts a parabola horizontally and vertically --/
def shift_parabola (p : Parabola) (dx dy : ℝ) : Parabola :=
  { a := p.a, h := p.h + dx, k := p.k + dy }

/-- The original parabola y = -x^2 + 1 --/
def original_parabola : Parabola :=
  { a := -1, h := 0, k := 1 }

/-- The transformed parabola after shifting --/
def transformed_parabola : Parabola :=
  shift_parabola original_parabola 2 (-2)

theorem parabola_transformation :
  transformed_parabola = { a := -1, h := 2, k := -1 } :=
sorry

end NUMINAMATH_CALUDE_parabola_transformation_l2353_235366


namespace NUMINAMATH_CALUDE_unknowns_and_variables_l2353_235392

-- Define a type for equations
structure Equation where
  f : ℝ → ℝ → ℝ
  c : ℝ

-- Define a type for systems of equations
structure SystemOfEquations where
  eq1 : Equation
  eq2 : Equation

-- Define a type for single equations
structure SingleEquation where
  eq : Equation

-- Define a property for being an unknown
def isUnknown (x : ℝ) (y : ℝ) (system : SystemOfEquations) : Prop :=
  ∀ (sol_x sol_y : ℝ), system.eq1.f sol_x sol_y = system.eq1.c ∧ 
                        system.eq2.f sol_x sol_y = system.eq2.c →
                        x = sol_x ∧ y = sol_y

-- Define a property for being a variable
def isVariable (x : ℝ) (y : ℝ) (single : SingleEquation) : Prop :=
  ∀ (val_x : ℝ), ∃ (val_y : ℝ), single.eq.f val_x val_y = single.eq.c

-- Theorem statement
theorem unknowns_and_variables 
  (x y : ℝ) (system : SystemOfEquations) (single : SingleEquation) : 
  (isUnknown x y system) ∧ (isVariable x y single) := by
  sorry

end NUMINAMATH_CALUDE_unknowns_and_variables_l2353_235392


namespace NUMINAMATH_CALUDE_balloon_difference_l2353_235335

theorem balloon_difference (x y z : ℚ) 
  (eq1 : x = 3 * z - 2)
  (eq2 : y = z / 4 + 5)
  (eq3 : z = y + 3) :
  x + y - z = 27 := by
  sorry

end NUMINAMATH_CALUDE_balloon_difference_l2353_235335


namespace NUMINAMATH_CALUDE_sqrt_eight_plus_sqrt_two_l2353_235309

theorem sqrt_eight_plus_sqrt_two : Real.sqrt 8 + Real.sqrt 2 = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_plus_sqrt_two_l2353_235309


namespace NUMINAMATH_CALUDE_quadratic_increasing_l2353_235376

/-- Given a quadratic function y = (x - 1)^2 + 2, prove that y is increasing when x > 1 -/
theorem quadratic_increasing (x : ℝ) : 
  let y : ℝ → ℝ := λ x ↦ (x - 1)^2 + 2
  x > 1 → ∀ h > 0, y (x + h) > y x :=
by sorry

end NUMINAMATH_CALUDE_quadratic_increasing_l2353_235376


namespace NUMINAMATH_CALUDE_paula_and_karl_ages_sum_l2353_235323

theorem paula_and_karl_ages_sum (P K : ℕ) : 
  (P - 5 = 3 * (K - 5)) →  -- 5 years ago, Paula was 3 times as old as Karl
  (P + 6 = 2 * (K + 6)) →  -- In 6 years, Paula will be twice as old as Karl
  P + K = 54 :=            -- The sum of their current ages is 54
by sorry

end NUMINAMATH_CALUDE_paula_and_karl_ages_sum_l2353_235323


namespace NUMINAMATH_CALUDE_curve_C_equation_sum_of_slopes_constant_l2353_235357

noncomputable section

def Circle (O : ℝ × ℝ) (r : ℝ) := {P : ℝ × ℝ | (P.1 - O.1)^2 + (P.2 - O.2)^2 = r^2}

def Curve_C := {N : ℝ × ℝ | N.1^2 / 6 + N.2^2 / 3 = 1}

def Point_on_circle (P : ℝ × ℝ) := P.1^2 + P.2^2 = 6

def Point_N (P N : ℝ × ℝ) := 
  ∃ (M : ℝ × ℝ), M.2 = 0 ∧ (P.1 - M.1)^2 + (P.2 - M.2)^2 = 2 * ((N.1 - M.1)^2 + (N.2 - M.2)^2)

def Line_through_B (k : ℝ) := {P : ℝ × ℝ | P.2 = k * (P.1 - 3)}

def Slope (A B : ℝ × ℝ) := (B.2 - A.2) / (B.1 - A.1)

theorem curve_C_equation :
  ∀ N : ℝ × ℝ, (∃ P : ℝ × ℝ, Point_on_circle P ∧ Point_N P N) → N ∈ Curve_C := by sorry

theorem sum_of_slopes_constant :
  ∀ k : ℝ, ∀ D E : ℝ × ℝ,
    D ∈ Curve_C ∧ E ∈ Curve_C ∧ D ∈ Line_through_B k ∧ E ∈ Line_through_B k ∧ D ≠ E →
    Slope (2, 1) D + Slope (2, 1) E = -2 := by sorry

end NUMINAMATH_CALUDE_curve_C_equation_sum_of_slopes_constant_l2353_235357


namespace NUMINAMATH_CALUDE_combination_sum_equals_c_11_3_l2353_235388

theorem combination_sum_equals_c_11_3 :
  (Finset.range 9).sum (fun k => Nat.choose (k + 2) 2) = Nat.choose 11 3 := by
  sorry

end NUMINAMATH_CALUDE_combination_sum_equals_c_11_3_l2353_235388


namespace NUMINAMATH_CALUDE_egg_processing_plant_l2353_235371

theorem egg_processing_plant (E : ℕ) : 
  (∃ A R : ℕ, 
    E = A + R ∧ 
    A = 388 * (R / 12) ∧
    (A + 37) / R = 405 / 3) →
  E = 125763 := by
sorry

end NUMINAMATH_CALUDE_egg_processing_plant_l2353_235371


namespace NUMINAMATH_CALUDE_percentage_of_C_grades_l2353_235385

/-- Represents a grade with its lower and upper bounds -/
structure Grade where
  letter : String
  lower : Nat
  upper : Nat

/-- Checks if a score falls within a grade range -/
def isInGradeRange (score : Nat) (grade : Grade) : Bool :=
  score >= grade.lower ∧ score <= grade.upper

/-- The grading scale -/
def gradingScale : List Grade := [
  ⟨"A", 95, 100⟩,
  ⟨"A-", 90, 94⟩,
  ⟨"B+", 85, 89⟩,
  ⟨"B", 80, 84⟩,
  ⟨"C+", 77, 79⟩,
  ⟨"C", 73, 76⟩,
  ⟨"D", 70, 72⟩,
  ⟨"F", 0, 69⟩
]

/-- The list of student scores -/
def scores : List Nat := [98, 75, 86, 77, 60, 94, 72, 79, 69, 82, 70, 93, 74, 87, 78, 84, 95, 73]

/-- Theorem stating that the percentage of students who received a grade of C is 16.67% -/
theorem percentage_of_C_grades (ε : Real) (h : ε > 0) : 
  ∃ (p : Real), abs (p - 16.67) < ε ∧ 
  p = (100 : Real) * (scores.filter (fun score => 
    ∃ (g : Grade), g ∈ gradingScale ∧ g.letter = "C" ∧ isInGradeRange score g
  )).length / scores.length :=
sorry

end NUMINAMATH_CALUDE_percentage_of_C_grades_l2353_235385


namespace NUMINAMATH_CALUDE_odd_sum_difference_l2353_235398

def sum_odd_range (a b : ℕ) : ℕ :=
  let first := if a % 2 = 1 then a else a + 1
  let last := if b % 2 = 1 then b else b - 1
  let n := (last - first) / 2 + 1
  n * (first + last) / 2

theorem odd_sum_difference : 
  sum_odd_range 101 300 - sum_odd_range 3 70 = 18776 := by
  sorry

end NUMINAMATH_CALUDE_odd_sum_difference_l2353_235398


namespace NUMINAMATH_CALUDE_smallest_group_size_l2353_235350

theorem smallest_group_size (n : ℕ) : 
  (n % 18 = 0 ∧ n % 60 = 0) → n ≥ Nat.lcm 18 60 := by
  sorry

#eval Nat.lcm 18 60

end NUMINAMATH_CALUDE_smallest_group_size_l2353_235350


namespace NUMINAMATH_CALUDE_xy_sum_l2353_235396

theorem xy_sum (x y : ℕ) (hx : x < 15) (hy : y < 25) (hxy : x + y + x * y = 119) :
  x + y = 20 ∨ x + y = 21 := by
  sorry

end NUMINAMATH_CALUDE_xy_sum_l2353_235396


namespace NUMINAMATH_CALUDE_cylinder_volume_l2353_235303

/-- The volume of a cylinder with diameter 8 cm and height 5 cm is 80π cubic centimeters. -/
theorem cylinder_volume (π : ℝ) (h : π > 0) : 
  let d : ℝ := 8 -- diameter in cm
  let h : ℝ := 5 -- height in cm
  let r : ℝ := d / 2 -- radius in cm
  let volume : ℝ := π * r^2 * h
  volume = 80 * π := by sorry

end NUMINAMATH_CALUDE_cylinder_volume_l2353_235303


namespace NUMINAMATH_CALUDE_grass_seed_problem_l2353_235387

/-- Represents the cost and weight of a bag of grass seed -/
structure SeedBag where
  weight : Nat
  cost : Rat

/-- Represents a purchase of grass seed -/
structure Purchase where
  bags : List SeedBag
  totalWeight : Nat
  totalCost : Rat

def validPurchase (p : Purchase) : Prop :=
  p.totalWeight ≥ 65 ∧ p.totalWeight ≤ 80

def optimalPurchase (p : Purchase) : Prop :=
  validPurchase p ∧ p.totalCost = 98.75

/-- The theorem to be proved -/
theorem grass_seed_problem :
  ∃ (cost_5lb : Rat),
    let bag_5lb : SeedBag := ⟨5, cost_5lb⟩
    let bag_10lb : SeedBag := ⟨10, 20.40⟩
    let bag_25lb : SeedBag := ⟨25, 32.25⟩
    ∃ (p : Purchase),
      optimalPurchase p ∧
      bag_5lb ∈ p.bags ∧
      cost_5lb = 2.00 :=
sorry

end NUMINAMATH_CALUDE_grass_seed_problem_l2353_235387


namespace NUMINAMATH_CALUDE_cube_with_holes_surface_area_l2353_235340

/-- Represents a cube with holes cut through each face -/
structure CubeWithHoles where
  edge_length : ℝ
  hole_side_length : ℝ

/-- Calculates the total surface area of a cube with holes, including inside surfaces -/
def total_surface_area (cube : CubeWithHoles) : ℝ :=
  let original_surface_area := 6 * cube.edge_length^2
  let area_removed_by_holes := 6 * cube.hole_side_length^2
  let new_exposed_area := 6 * 6 * cube.hole_side_length^2
  original_surface_area - area_removed_by_holes + new_exposed_area

/-- Theorem stating that a cube with edge length 5 and hole side length 2 has total surface area 270 -/
theorem cube_with_holes_surface_area :
  total_surface_area { edge_length := 5, hole_side_length := 2 } = 270 := by
  sorry

end NUMINAMATH_CALUDE_cube_with_holes_surface_area_l2353_235340


namespace NUMINAMATH_CALUDE_square_sum_product_divisibility_l2353_235382

/-- A structure representing a triple of positive integers -/
structure PositiveTriple where
  x : ℕ+
  y : ℕ+
  z : ℕ+

/-- The set of valid triples for k = 3 -/
def validTriplesForK3 : Set PositiveTriple :=
  {⟨1, 1, 1⟩, ⟨1, 1, 2⟩, ⟨1, 2, 5⟩, ⟨1, 5, 13⟩, ⟨2, 5, 29⟩, ⟨5, 29, 169⟩}

/-- The set of valid triples for k = 1 -/
def validTriplesForK1 : Set PositiveTriple :=
  {⟨3, 3, 3⟩, ⟨3, 3, 6⟩, ⟨3, 6, 15⟩, ⟨6, 15, 39⟩, ⟨6, 15, 87⟩}

/-- The main theorem -/
theorem square_sum_product_divisibility
  (x y z : ℕ+)
  (h_bound : x ≤ 1000 ∧ y ≤ 1000 ∧ z ≤ 1000) :
  (∃ k : ℤ, (x : ℤ)^2 + (y : ℤ)^2 + (z : ℤ)^2 = k * (x : ℤ) * (y : ℤ) * (z : ℤ)) ↔
  (k = 1 ∨ k = 3) ∧ (⟨x, y, z⟩ ∈ validTriplesForK1 ∪ validTriplesForK3) :=
sorry

end NUMINAMATH_CALUDE_square_sum_product_divisibility_l2353_235382
