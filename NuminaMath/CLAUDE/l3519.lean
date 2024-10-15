import Mathlib

namespace NUMINAMATH_CALUDE_oil_ratio_in_first_bottle_l3519_351901

theorem oil_ratio_in_first_bottle 
  (C : ℝ) 
  (h1 : C > 0)
  (oil_in_second : ℝ)
  (h2 : oil_in_second = C / 2)
  (total_content : ℝ)
  (h3 : total_content = 3 * C)
  (total_oil : ℝ)
  (h4 : total_oil = total_content / 3)
  (oil_in_first : ℝ)
  (h5 : oil_in_first + oil_in_second = total_oil) :
  oil_in_first / C = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_oil_ratio_in_first_bottle_l3519_351901


namespace NUMINAMATH_CALUDE_cubic_roots_relation_l3519_351953

theorem cubic_roots_relation (p q r : ℝ) (u v w : ℝ) : 
  (∀ x : ℝ, x^3 - 6*x^2 + 11*x + 10 = (x - p) * (x - q) * (x - r)) →
  (∀ x : ℝ, x^3 + u*x^2 + v*x + w = (x - (p + q)) * (x - (q + r)) * (x - (r + p))) →
  w = 80 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_relation_l3519_351953


namespace NUMINAMATH_CALUDE_root_implies_a_values_l3519_351994

theorem root_implies_a_values (a : ℝ) : 
  (2 * (-1)^2 + a * (-1) - a^2 = 0) → (a = 1 ∨ a = -2) := by
  sorry

end NUMINAMATH_CALUDE_root_implies_a_values_l3519_351994


namespace NUMINAMATH_CALUDE_cone_base_radius_l3519_351960

/-- Given a cone with slant height 5 and lateral area 15π, its base radius is 3 -/
theorem cone_base_radius (s : ℝ) (L : ℝ) (r : ℝ) : 
  s = 5 → L = 15 * Real.pi → L = Real.pi * r * s → r = 3 := by sorry

end NUMINAMATH_CALUDE_cone_base_radius_l3519_351960


namespace NUMINAMATH_CALUDE_rectangular_field_laps_l3519_351984

theorem rectangular_field_laps (length width total_distance : ℝ) 
  (h_length : length = 75)
  (h_width : width = 15)
  (h_total_distance : total_distance = 540) :
  total_distance / (2 * (length + width)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_laps_l3519_351984


namespace NUMINAMATH_CALUDE_tank_length_is_six_l3519_351917

def tank_volume : ℝ := 72
def tank_width : ℝ := 4
def tank_depth : ℝ := 3

theorem tank_length_is_six :
  let length := tank_volume / (tank_width * tank_depth)
  length = 6 := by sorry

end NUMINAMATH_CALUDE_tank_length_is_six_l3519_351917


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l3519_351949

/-- Given that the arithmetic mean of six expressions is 30, prove that x = 18.5 and y = 10. -/
theorem arithmetic_mean_problem (x y : ℝ) :
  ((2*x - y) + 20 + (3*x + y) + 16 + (x + 5) + (y + 8)) / 6 = 30 →
  x = 18.5 ∧ y = 10 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l3519_351949


namespace NUMINAMATH_CALUDE_local_road_speed_l3519_351978

/-- Proves that the speed of a car on local roads is 20 mph given the specified conditions -/
theorem local_road_speed (local_distance : ℝ) (highway_distance : ℝ) (highway_speed : ℝ) (average_speed : ℝ) :
  local_distance = 60 →
  highway_distance = 120 →
  highway_speed = 60 →
  average_speed = 36 →
  (local_distance + highway_distance) / (local_distance / (local_distance / (local_distance / average_speed - highway_distance / highway_speed)) + highway_distance / highway_speed) = average_speed →
  local_distance / (local_distance / average_speed - highway_distance / highway_speed) = 20 :=
by sorry

end NUMINAMATH_CALUDE_local_road_speed_l3519_351978


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a3_l3519_351923

/-- An arithmetic sequence with common difference 2 where a₂ is the geometric mean of a₁ and a₅ -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) = a n + 2) ∧ a 2 ^ 2 = a 1 * a 5

theorem arithmetic_sequence_a3 (a : ℕ → ℝ) (h : ArithmeticSequence a) : a 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a3_l3519_351923


namespace NUMINAMATH_CALUDE_football_players_count_l3519_351974

theorem football_players_count (total : ℕ) (basketball : ℕ) (neither : ℕ) (both : ℕ) 
  (h1 : total = 22)
  (h2 : basketball = 13)
  (h3 : neither = 3)
  (h4 : both = 18) :
  total - neither - (basketball - both) = 19 :=
by sorry

end NUMINAMATH_CALUDE_football_players_count_l3519_351974


namespace NUMINAMATH_CALUDE_concertHallSeats_l3519_351996

/-- Represents a concert hall with a specific seating arrangement. -/
structure ConcertHall where
  rows : ℕ
  middleRowSeats : ℕ
  middleRowIndex : ℕ
  increaseFactor : ℕ

/-- Calculates the total number of seats in the concert hall. -/
def totalSeats (hall : ConcertHall) : ℕ :=
  let firstRowSeats := hall.middleRowSeats - 2 * (hall.middleRowIndex - 1)
  let lastRowSeats := hall.middleRowSeats + 2 * (hall.rows - hall.middleRowIndex)
  hall.rows * (firstRowSeats + lastRowSeats) / 2

/-- Theorem stating that a concert hall with the given properties has 1984 seats. -/
theorem concertHallSeats (hall : ConcertHall) 
    (h1 : hall.rows = 31)
    (h2 : hall.middleRowSeats = 64)
    (h3 : hall.middleRowIndex = 16)
    (h4 : hall.increaseFactor = 2) : 
  totalSeats hall = 1984 := by
  sorry


end NUMINAMATH_CALUDE_concertHallSeats_l3519_351996


namespace NUMINAMATH_CALUDE_total_pencils_is_fifty_l3519_351997

/-- The number of pencils Sabrina has -/
def sabrina_pencils : ℕ := 14

/-- The number of pencils Justin has -/
def justin_pencils : ℕ := 2 * sabrina_pencils + 8

/-- The total number of pencils Justin and Sabrina have combined -/
def total_pencils : ℕ := justin_pencils + sabrina_pencils

/-- Theorem stating that the total number of pencils is 50 -/
theorem total_pencils_is_fifty : total_pencils = 50 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_is_fifty_l3519_351997


namespace NUMINAMATH_CALUDE_intersection_M_N_l3519_351916

def M : Set ℤ := {-1, 0, 1}

def N : Set ℤ := {x | ∃ a ∈ M, x = 2 * a}

theorem intersection_M_N : M ∩ N = {0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3519_351916


namespace NUMINAMATH_CALUDE_peter_and_laura_seating_probability_l3519_351990

-- Define the number of chairs
def num_chairs : ℕ := 10

-- Define the probability of not sitting next to each other
def prob_not_adjacent : ℚ := 4 / 5

-- Theorem statement
theorem peter_and_laura_seating_probability :
  let total_ways := num_chairs.choose 2
  let adjacent_ways := num_chairs - 1
  prob_not_adjacent = 1 - (adjacent_ways : ℚ) / (total_ways : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_peter_and_laura_seating_probability_l3519_351990


namespace NUMINAMATH_CALUDE_add_point_four_five_to_fifty_seven_point_two_five_l3519_351952

theorem add_point_four_five_to_fifty_seven_point_two_five :
  57.25 + 0.45 = 57.7 := by
  sorry

end NUMINAMATH_CALUDE_add_point_four_five_to_fifty_seven_point_two_five_l3519_351952


namespace NUMINAMATH_CALUDE_apple_count_correct_l3519_351998

/-- The number of apples in a box containing apples and oranges -/
def num_apples : ℕ := 14

/-- The initial number of oranges in the box -/
def initial_oranges : ℕ := 20

/-- The number of oranges removed from the box -/
def removed_oranges : ℕ := 14

/-- The percentage of apples after removing oranges -/
def apple_percentage : ℝ := 0.7

theorem apple_count_correct :
  num_apples = 14 ∧
  initial_oranges = 20 ∧
  removed_oranges = 14 ∧
  apple_percentage = 0.7 ∧
  (num_apples : ℝ) / ((num_apples : ℝ) + (initial_oranges - removed_oranges : ℝ)) = apple_percentage :=
by sorry

end NUMINAMATH_CALUDE_apple_count_correct_l3519_351998


namespace NUMINAMATH_CALUDE_friendship_theorem_l3519_351989

/-- Represents a simple undirected graph with 6 vertices -/
def Graph := Fin 6 → Fin 6 → Bool

/-- The friendship relation is symmetric -/
def symmetric (g : Graph) : Prop :=
  ∀ i j : Fin 6, g i j = g j i

/-- A set of three vertices form a triangle in the graph -/
def isTriangle (g : Graph) (v1 v2 v3 : Fin 6) : Prop :=
  v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3 ∧
  ((g v1 v2 ∧ g v2 v3 ∧ g v1 v3) ∨ (¬g v1 v2 ∧ ¬g v2 v3 ∧ ¬g v1 v3))

/-- Main theorem: any graph with 6 vertices contains a monochromatic triangle -/
theorem friendship_theorem (g : Graph) (h : symmetric g) :
  ∃ v1 v2 v3 : Fin 6, isTriangle g v1 v2 v3 := by
  sorry

end NUMINAMATH_CALUDE_friendship_theorem_l3519_351989


namespace NUMINAMATH_CALUDE_fourth_term_is_negative_24_l3519_351900

-- Define a geometric sequence
def geometric_sequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r^(n - 1)

-- Define the conditions of our specific sequence
def our_sequence (x : ℝ) (n : ℕ) : ℝ :=
  match n with
  | 1 => x
  | 2 => 3*x + 3
  | 3 => 6*x + 6
  | _ => geometric_sequence x 2 n

-- Theorem statement
theorem fourth_term_is_negative_24 : 
  ∀ x : ℝ, our_sequence x 4 = -24 := by sorry

end NUMINAMATH_CALUDE_fourth_term_is_negative_24_l3519_351900


namespace NUMINAMATH_CALUDE_number_equation_solution_l3519_351965

theorem number_equation_solution : 
  ∃ x : ℝ, 0.4 * x + 60 = x ∧ x = 100 := by sorry

end NUMINAMATH_CALUDE_number_equation_solution_l3519_351965


namespace NUMINAMATH_CALUDE_abs_value_of_specific_complex_l3519_351987

/-- Given a complex number z = (1-i)/i, prove that its absolute value |z| is equal to √2 -/
theorem abs_value_of_specific_complex : let z : ℂ := (1 - Complex.I) / Complex.I
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_value_of_specific_complex_l3519_351987


namespace NUMINAMATH_CALUDE_symmetric_circle_equation_l3519_351969

-- Define the original circle
def original_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop := y = -x

-- Define the symmetric circle C
def symmetric_circle (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 1

-- Theorem stating that the symmetric circle C has the equation x^2 + (y + 1)^2 = 1
theorem symmetric_circle_equation :
  ∀ x y : ℝ,
  (∃ x' y' : ℝ, original_circle x' y' ∧ 
   symmetry_line ((x + x') / 2) ((y + y') / 2)) →
  symmetric_circle x y :=
sorry

end NUMINAMATH_CALUDE_symmetric_circle_equation_l3519_351969


namespace NUMINAMATH_CALUDE_percentage_of_students_passed_l3519_351903

/-- The percentage of students who passed an examination, given the total number of students and the number of students who failed. -/
theorem percentage_of_students_passed (total : ℕ) (failed : ℕ) (h1 : total = 840) (h2 : failed = 546) :
  (((total - failed : ℚ) / total) * 100 : ℚ) = 35 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_students_passed_l3519_351903


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3519_351911

/-- Given a hyperbola with the following properties:
  - Standard form equation: x²/a² - y²/b² = 1
  - a > 0 and b > 0
  - A focus at (2, 0)
  - Asymptotes: y = ±√3x
  Prove that the equation of the hyperbola is x² - y²/3 = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (focus : (2 : ℝ) = (a^2 + b^2).sqrt)
  (asymptote : b/a = Real.sqrt 3) :
  ∀ x y : ℝ, x^2 - y^2/3 = 1 ↔ x^2/a^2 - y^2/b^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3519_351911


namespace NUMINAMATH_CALUDE_binary_to_decimal_101001_l3519_351967

/-- Converts a list of binary digits to its decimal representation -/
def binaryToDecimal (bits : List Nat) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + b * 2^i) 0

/-- The binary representation of the number we want to convert -/
def binaryNumber : List Nat := [1, 0, 1, 0, 0, 1]

/-- Theorem stating that the binary number 101001 is equal to the decimal number 41 -/
theorem binary_to_decimal_101001 :
  binaryToDecimal binaryNumber = 41 := by
  sorry

#eval binaryToDecimal binaryNumber

end NUMINAMATH_CALUDE_binary_to_decimal_101001_l3519_351967


namespace NUMINAMATH_CALUDE_bus_stop_speed_fraction_l3519_351983

theorem bus_stop_speed_fraction (usual_time normal_delay : ℕ) (fraction : ℚ) : 
  usual_time = 28 →
  normal_delay = 7 →
  fraction * (usual_time + normal_delay) = usual_time →
  fraction = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_bus_stop_speed_fraction_l3519_351983


namespace NUMINAMATH_CALUDE_correct_statements_count_l3519_351946

/-- Represents the correctness of a statement -/
inductive Correctness
| correct
| incorrect

/-- Evaluates the correctness of statement 1 -/
def statement1 : Correctness := Correctness.correct

/-- Evaluates the correctness of statement 2 -/
def statement2 : Correctness := Correctness.incorrect

/-- Evaluates the correctness of statement 3 -/
def statement3 : Correctness := Correctness.incorrect

/-- Evaluates the correctness of statement 4 -/
def statement4 : Correctness := Correctness.correct

/-- Counts the number of correct statements -/
def countCorrect (s1 s2 s3 s4 : Correctness) : Nat :=
  match s1, s2, s3, s4 with
  | Correctness.correct, Correctness.correct, Correctness.correct, Correctness.correct => 4
  | Correctness.correct, Correctness.correct, Correctness.correct, Correctness.incorrect => 3
  | Correctness.correct, Correctness.correct, Correctness.incorrect, Correctness.correct => 3
  | Correctness.correct, Correctness.correct, Correctness.incorrect, Correctness.incorrect => 2
  | Correctness.correct, Correctness.incorrect, Correctness.correct, Correctness.correct => 3
  | Correctness.correct, Correctness.incorrect, Correctness.correct, Correctness.incorrect => 2
  | Correctness.correct, Correctness.incorrect, Correctness.incorrect, Correctness.correct => 2
  | Correctness.correct, Correctness.incorrect, Correctness.incorrect, Correctness.incorrect => 1
  | Correctness.incorrect, Correctness.correct, Correctness.correct, Correctness.correct => 3
  | Correctness.incorrect, Correctness.correct, Correctness.correct, Correctness.incorrect => 2
  | Correctness.incorrect, Correctness.correct, Correctness.incorrect, Correctness.correct => 2
  | Correctness.incorrect, Correctness.correct, Correctness.incorrect, Correctness.incorrect => 1
  | Correctness.incorrect, Correctness.incorrect, Correctness.correct, Correctness.correct => 2
  | Correctness.incorrect, Correctness.incorrect, Correctness.correct, Correctness.incorrect => 1
  | Correctness.incorrect, Correctness.incorrect, Correctness.incorrect, Correctness.correct => 1
  | Correctness.incorrect, Correctness.incorrect, Correctness.incorrect, Correctness.incorrect => 0

theorem correct_statements_count :
  countCorrect statement1 statement2 statement3 statement4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_correct_statements_count_l3519_351946


namespace NUMINAMATH_CALUDE_megacorp_fine_l3519_351931

def daily_mining_revenue : ℝ := 3000000
def daily_oil_revenue : ℝ := 5000000
def monthly_expenses : ℝ := 30000000
def fine_percentage : ℝ := 0.01
def days_in_year : ℕ := 365
def months_in_year : ℕ := 12

theorem megacorp_fine :
  let daily_revenue := daily_mining_revenue + daily_oil_revenue
  let annual_revenue := daily_revenue * days_in_year
  let annual_expenses := monthly_expenses * months_in_year
  let annual_profit := annual_revenue - annual_expenses
  let fine := annual_profit * fine_percentage
  fine = 25600000 := by sorry

end NUMINAMATH_CALUDE_megacorp_fine_l3519_351931


namespace NUMINAMATH_CALUDE_ST_length_l3519_351941

-- Define the points
variable (P Q R S T : ℝ × ℝ)

-- Define the distances
def distance (A B : ℝ × ℝ) : ℝ := sorry

-- State the conditions
axiom PQ_eq_6 : distance P Q = 6
axiom QR_eq_6 : distance Q R = 6
axiom RS_eq_6 : distance R S = 6
axiom SP_eq_6 : distance S P = 6
axiom SQ_eq_6 : distance S Q = 6
axiom PT_eq_14 : distance P T = 14
axiom RT_eq_14 : distance R T = 14

-- PQRS is a rhombus
axiom is_rhombus : distance P Q = distance Q R ∧ distance Q R = distance R S ∧ distance R S = distance S P

-- PQS and RQS are equilateral triangles
axiom PQS_equilateral : distance P Q = distance Q S ∧ distance Q S = distance S P
axiom RQS_equilateral : distance R Q = distance Q S ∧ distance Q S = distance R S

-- The theorem to prove
theorem ST_length : distance S T = 10 :=
sorry

end NUMINAMATH_CALUDE_ST_length_l3519_351941


namespace NUMINAMATH_CALUDE_charity_event_probability_l3519_351954

theorem charity_event_probability :
  let n : ℕ := 5  -- number of students
  let d : ℕ := 2  -- number of days (Saturday and Sunday)
  let total_outcomes : ℕ := d^n
  let same_day_outcomes : ℕ := 2  -- all choose Saturday or all choose Sunday
  let both_days_outcomes : ℕ := total_outcomes - same_day_outcomes
  (both_days_outcomes : ℚ) / total_outcomes = 15 / 16 :=
by sorry

end NUMINAMATH_CALUDE_charity_event_probability_l3519_351954


namespace NUMINAMATH_CALUDE_jeff_donuts_per_day_l3519_351945

/-- The number of days Jeff makes donuts -/
def days : ℕ := 12

/-- The number of donuts Jeff eats per day -/
def jeff_eats_per_day : ℕ := 1

/-- The total number of donuts Chris eats -/
def chris_eats_total : ℕ := 8

/-- The number of donuts that fit in each box -/
def donuts_per_box : ℕ := 10

/-- The number of boxes Jeff can fill -/
def boxes_filled : ℕ := 10

/-- The number of donuts Jeff makes each day -/
def donuts_per_day : ℕ := 10

theorem jeff_donuts_per_day :
  ∃ (d : ℕ), 
    d * days - (jeff_eats_per_day * days) - chris_eats_total = boxes_filled * donuts_per_box ∧
    d = donuts_per_day :=
by sorry

end NUMINAMATH_CALUDE_jeff_donuts_per_day_l3519_351945


namespace NUMINAMATH_CALUDE_total_squat_bench_press_l3519_351906

/-- Represents the weight Tony can lift in various exercises --/
structure TonyLift where
  curl : ℝ
  military_press : ℝ
  squat : ℝ
  bench_press : ℝ

/-- Defines Tony's lifting capabilities based on the given conditions --/
def tony_lift : TonyLift where
  curl := 90
  military_press := 2 * 90
  squat := 5 * (2 * 90)
  bench_press := 1.5 * (2 * 90)

/-- Theorem stating the total weight Tony can lift in squat and bench press combined --/
theorem total_squat_bench_press (t : TonyLift) (h : t = tony_lift) : 
  t.squat + t.bench_press = 1170 := by
  sorry

end NUMINAMATH_CALUDE_total_squat_bench_press_l3519_351906


namespace NUMINAMATH_CALUDE_remaining_kittens_l3519_351975

def initial_kittens : ℕ := 8
def given_away : ℕ := 2

theorem remaining_kittens : initial_kittens - given_away = 6 := by
  sorry

end NUMINAMATH_CALUDE_remaining_kittens_l3519_351975


namespace NUMINAMATH_CALUDE_middle_of_five_consecutive_sum_60_l3519_351902

theorem middle_of_five_consecutive_sum_60 (a b c d e : ℕ) : 
  (a + b + c + d + e = 60) → 
  (b = a + 1) → 
  (c = b + 1) → 
  (d = c + 1) → 
  (e = d + 1) → 
  c = 12 := by
sorry

end NUMINAMATH_CALUDE_middle_of_five_consecutive_sum_60_l3519_351902


namespace NUMINAMATH_CALUDE_arithmetic_sequence_tan_l3519_351908

/-- Given an arithmetic sequence {a_n} where a₁ + a₇ + a₁₃ = π, 
    prove that tan(a₂ + a₁₂) = -√3 -/
theorem arithmetic_sequence_tan (a : ℕ → ℝ) :
  (∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n) →  -- arithmetic sequence
  a 1 + a 7 + a 13 = Real.pi →                      -- given condition
  Real.tan (a 2 + a 12) = -Real.sqrt 3 :=           -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_tan_l3519_351908


namespace NUMINAMATH_CALUDE_definite_integral_2x_plus_exp_l3519_351964

theorem definite_integral_2x_plus_exp : ∫ x in (0:ℝ)..1, (2 * x + Real.exp x) = Real.exp 1 - 1 := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_2x_plus_exp_l3519_351964


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3519_351985

theorem polynomial_simplification (x : ℝ) :
  (3 * x - 2) * (5 * x^9 + 3 * x^8 + 2 * x^7 + x^6) =
  15 * x^10 - x^9 + 3 * x^7 - 2 * x^6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3519_351985


namespace NUMINAMATH_CALUDE_chocolate_milk_probability_l3519_351925

theorem chocolate_milk_probability (n : ℕ) (k : ℕ) (p : ℚ) :
  n = 6 →
  k = 5 →
  p = 2/3 →
  Nat.choose n k * p^k * (1 - p)^(n - k) = 64/243 :=
by sorry

end NUMINAMATH_CALUDE_chocolate_milk_probability_l3519_351925


namespace NUMINAMATH_CALUDE_circle_radius_from_tangents_l3519_351934

/-- A circle with two parallel tangents and a third tangent -/
structure CircleWithTangents where
  r : ℝ  -- radius of the circle
  xy : ℝ  -- length of tangent XY
  xpyp : ℝ  -- length of tangent X'Y'

/-- The theorem stating the relationship between the tangents and the radius -/
theorem circle_radius_from_tangents (c : CircleWithTangents) 
  (h1 : c.xy = 7)
  (h2 : c.xpyp = 12) :
  c.r = 4 * Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_tangents_l3519_351934


namespace NUMINAMATH_CALUDE_gcd_180_450_l3519_351926

theorem gcd_180_450 : Nat.gcd 180 450 = 90 := by
  sorry

end NUMINAMATH_CALUDE_gcd_180_450_l3519_351926


namespace NUMINAMATH_CALUDE_factor_tree_value_l3519_351936

theorem factor_tree_value (X Y Z F G : ℕ) : 
  X = Y * Z ∧
  Y = 7 * F ∧
  F = 2 * 5 ∧
  Z = 11 * G ∧
  G = 3 * 7 →
  X = 16170 := by sorry

end NUMINAMATH_CALUDE_factor_tree_value_l3519_351936


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_meaningful_only_four_satisfies_l3519_351935

theorem sqrt_x_minus_one_meaningful (x : ℝ) : x - 1 ≥ 0 ↔ x ≥ 1 := by sorry

theorem only_four_satisfies :
  (4 - 1 ≥ 0) ∧ 
  ¬(-4 - 1 ≥ 0) ∧ 
  ¬(-1 - 1 ≥ 0) ∧ 
  ¬(0 - 1 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_meaningful_only_four_satisfies_l3519_351935


namespace NUMINAMATH_CALUDE_area_bounded_by_parabola_and_x_axis_l3519_351904

/-- The area of the figure bounded by y = 2x - x^2 and y = 0 is 4/3 square units. -/
theorem area_bounded_by_parabola_and_x_axis : 
  let f (x : ℝ) := 2 * x - x^2
  ∫ x in (0)..(2), max 0 (f x) = 4/3 := by sorry

end NUMINAMATH_CALUDE_area_bounded_by_parabola_and_x_axis_l3519_351904


namespace NUMINAMATH_CALUDE_minutes_from_2222_to_midnight_l3519_351915

def minutes_until_midnight (hour : Nat) (minute : Nat) : Nat :=
  (23 - hour) * 60 + (60 - minute)

theorem minutes_from_2222_to_midnight :
  minutes_until_midnight 22 22 = 98 := by
  sorry

end NUMINAMATH_CALUDE_minutes_from_2222_to_midnight_l3519_351915


namespace NUMINAMATH_CALUDE_nine_grams_combinations_l3519_351961

def weight_combinations (n : ℕ) : ℕ :=
  let ones := Finset.range 4
  let twos := Finset.range 4
  let fives := Finset.range 2
  (ones.product twos).product fives
    |>.filter (fun ((a, b), c) => a + 2*b + 5*c == n)
    |>.card

theorem nine_grams_combinations : weight_combinations 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_nine_grams_combinations_l3519_351961


namespace NUMINAMATH_CALUDE_quadratic_root_complex_l3519_351971

theorem quadratic_root_complex (c d : ℝ) : 
  (Complex.I : ℂ).re = 0 ∧ (Complex.I : ℂ).im = 1 →
  (3 - 4 * Complex.I : ℂ) ^ 2 + c * (3 - 4 * Complex.I : ℂ) + d = 0 →
  c = -6 ∧ d = 25 := by sorry

end NUMINAMATH_CALUDE_quadratic_root_complex_l3519_351971


namespace NUMINAMATH_CALUDE_rebecca_eggs_l3519_351933

/-- The number of eggs Rebecca has -/
def number_of_eggs : ℕ := 3 * 3

/-- The size of each group of eggs -/
def group_size : ℕ := 3

/-- The number of groups Rebecca created -/
def number_of_groups : ℕ := 3

theorem rebecca_eggs : 
  number_of_eggs = group_size * number_of_groups := by sorry

end NUMINAMATH_CALUDE_rebecca_eggs_l3519_351933


namespace NUMINAMATH_CALUDE_stewart_farm_sheep_count_l3519_351943

/-- The Stewart farm problem -/
theorem stewart_farm_sheep_count :
  ∀ (sheep horses : ℕ),
  sheep * 7 = horses * 6 →
  horses * 230 = 12880 →
  sheep = 48 :=
by
  sorry

end NUMINAMATH_CALUDE_stewart_farm_sheep_count_l3519_351943


namespace NUMINAMATH_CALUDE_pool_filling_time_l3519_351924

/-- Proves that filling a pool of given capacity with a specific number of hoses 
    and flow rate takes the calculated number of hours -/
theorem pool_filling_time 
  (pool_capacity : ℕ) 
  (num_hoses : ℕ) 
  (flow_rate_per_hose : ℕ) 
  (hours_to_fill : ℕ) 
  (h1 : pool_capacity = 32000)
  (h2 : num_hoses = 3)
  (h3 : flow_rate_per_hose = 4)
  (h4 : hours_to_fill = 44) : 
  pool_capacity = num_hoses * flow_rate_per_hose * 60 * hours_to_fill :=
by
  sorry

#check pool_filling_time

end NUMINAMATH_CALUDE_pool_filling_time_l3519_351924


namespace NUMINAMATH_CALUDE_base_of_equation_l3519_351905

theorem base_of_equation (e : ℕ) (h : e = 35) :
  ∃ b : ℚ, b^e * (1/4)^18 = 1/(2*(10^35)) ∧ b = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_base_of_equation_l3519_351905


namespace NUMINAMATH_CALUDE_sum_of_corners_is_164_l3519_351968

/-- Represents a square on the checkerboard -/
structure Square where
  row : Nat
  col : Nat

/-- The size of the checkerboard -/
def boardSize : Nat := 9

/-- The total number of squares on the board -/
def totalSquares : Nat := boardSize * boardSize

/-- Function to get the number in a given square -/
def getNumber (s : Square) : Nat :=
  (s.row - 1) * boardSize + s.col

/-- The four corner squares of the board -/
def corners : List Square := [
  { row := 1, col := 1 },       -- Top left
  { row := 1, col := boardSize },  -- Top right
  { row := boardSize, col := 1 },  -- Bottom left
  { row := boardSize, col := boardSize }  -- Bottom right
]

/-- Theorem stating that the sum of numbers in the corners is 164 -/
theorem sum_of_corners_is_164 :
  (corners.map getNumber).sum = 164 := by sorry

end NUMINAMATH_CALUDE_sum_of_corners_is_164_l3519_351968


namespace NUMINAMATH_CALUDE_ginos_popsicle_sticks_l3519_351982

def my_popsicle_sticks : ℕ := 50
def total_popsicle_sticks : ℕ := 113

theorem ginos_popsicle_sticks :
  total_popsicle_sticks - my_popsicle_sticks = 63 := by sorry

end NUMINAMATH_CALUDE_ginos_popsicle_sticks_l3519_351982


namespace NUMINAMATH_CALUDE_complex_equation_ratio_l3519_351942

theorem complex_equation_ratio (a b : ℝ) : 
  (Complex.mk a b) * (Complex.mk 1 1) = Complex.mk 7 (-3) → a / b = -2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_ratio_l3519_351942


namespace NUMINAMATH_CALUDE_equation_solution_l3519_351922

theorem equation_solution : ∃ x : ℝ, 45 * x = 0.4 * 900 ∧ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3519_351922


namespace NUMINAMATH_CALUDE_painted_cubes_count_l3519_351950

/-- Represents a cube constructed from unit cubes -/
structure LargeCube where
  side_length : ℕ
  unpainted_cubes : ℕ

/-- Calculates the number of cubes with at least one face painted -/
def painted_cubes (c : LargeCube) : ℕ :=
  c.side_length ^ 3 - c.unpainted_cubes

/-- The theorem states that for a cube with 22 unpainted cubes,
    42 cubes have at least one face painted red -/
theorem painted_cubes_count (c : LargeCube) 
  (h1 : c.unpainted_cubes = 22) 
  (h2 : c.side_length = 4) : 
  painted_cubes c = 42 := by
  sorry

#check painted_cubes_count

end NUMINAMATH_CALUDE_painted_cubes_count_l3519_351950


namespace NUMINAMATH_CALUDE_dance_off_combined_time_l3519_351981

/-- Given John and James' dancing schedules, prove their combined dancing time is 20 hours --/
theorem dance_off_combined_time (john_first_session : ℝ) (john_break : ℝ) (john_second_session : ℝ) 
  (james_extra_fraction : ℝ) : 
  john_first_session = 3 ∧ 
  john_break = 1 ∧ 
  john_second_session = 5 ∧ 
  james_extra_fraction = 1/3 → 
  (john_first_session + john_second_session) + 
  ((john_first_session + john_break + john_second_session) + 
   (john_first_session + john_break + john_second_session) * james_extra_fraction) = 20 := by
sorry

end NUMINAMATH_CALUDE_dance_off_combined_time_l3519_351981


namespace NUMINAMATH_CALUDE_max_sum_on_integer_circle_l3519_351910

theorem max_sum_on_integer_circle : 
  ∀ x y : ℤ, x^2 + y^2 = 100 → (∀ a b : ℤ, a^2 + b^2 = 100 → x + y ≥ a + b) → x + y = 14 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_on_integer_circle_l3519_351910


namespace NUMINAMATH_CALUDE_arithmetic_seq_sum_l3519_351963

/-- An arithmetic sequence with its partial sums -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  S : ℕ → ℚ  -- Partial sums
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- The main theorem -/
theorem arithmetic_seq_sum (seq : ArithmeticSequence) 
  (h3 : seq.S 3 = 3) 
  (h6 : seq.S 6 = 15) : 
  seq.a 10 + seq.a 11 + seq.a 12 = 30 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_seq_sum_l3519_351963


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l3519_351966

theorem tangent_line_to_circle (p : ℝ) : 
  (∀ x y : ℝ, x = -p/2 → x^2 + y^2 + 6*x + 8 = 0 → 
    ∃! y : ℝ, x^2 + y^2 + 6*x + 8 = 0) → 
  p = 4 ∨ p = 8 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l3519_351966


namespace NUMINAMATH_CALUDE_quadratic_root_property_l3519_351992

theorem quadratic_root_property (m : ℝ) : 
  m^2 - m - 3 = 0 → 2023 - m^2 + m = 2020 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_property_l3519_351992


namespace NUMINAMATH_CALUDE_square_difference_equality_l3519_351912

theorem square_difference_equality : 1010^2 - 990^2 - 1005^2 + 995^2 = 20000 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l3519_351912


namespace NUMINAMATH_CALUDE_puppies_adoption_time_l3519_351947

/-- The number of days required to adopt all puppies -/
def adoption_days (initial_puppies : ℕ) (additional_puppies : ℕ) (adoption_rate : ℕ) : ℕ :=
  (initial_puppies + additional_puppies) / adoption_rate

/-- Theorem: Given the initial conditions, it takes 9 days to adopt all puppies -/
theorem puppies_adoption_time :
  adoption_days 2 34 4 = 9 := by
  sorry

end NUMINAMATH_CALUDE_puppies_adoption_time_l3519_351947


namespace NUMINAMATH_CALUDE_sqrt_six_over_sqrt_two_equals_sqrt_three_l3519_351988

theorem sqrt_six_over_sqrt_two_equals_sqrt_three : 
  Real.sqrt 6 / Real.sqrt 2 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_six_over_sqrt_two_equals_sqrt_three_l3519_351988


namespace NUMINAMATH_CALUDE_monic_quartic_value_at_zero_l3519_351958

def is_monic_quartic (f : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, f x = x^4 + a*x^3 + b*x^2 + c*x + d

theorem monic_quartic_value_at_zero 
  (f : ℝ → ℝ) 
  (h_monic : is_monic_quartic f)
  (h_m2 : f (-2) = -4)
  (h_1 : f 1 = -1)
  (h_3 : f 3 = -9)
  (h_5 : f 5 = -25) :
  f 0 = 30 := by
sorry

end NUMINAMATH_CALUDE_monic_quartic_value_at_zero_l3519_351958


namespace NUMINAMATH_CALUDE_group_dynamics_index_difference_l3519_351938

theorem group_dynamics_index_difference :
  let n : ℕ := 35
  let k1 : ℕ := 15
  let k2 : ℕ := 5
  let k3 : ℕ := 8
  let l1 : ℕ := 6
  let l2 : ℕ := 10
  let index_females : ℚ := ((n - k1 + k2) / n : ℚ) * (1 + k3/10)
  let index_males : ℚ := ((n - (n - k1) + l1) / n : ℚ) * (1 + l2/10)
  index_females - index_males = 3/35 := by
  sorry

end NUMINAMATH_CALUDE_group_dynamics_index_difference_l3519_351938


namespace NUMINAMATH_CALUDE_sum_of_inscribed_circles_limit_l3519_351957

/-- The sum of areas of inscribed circles in a rectangle --/
def sum_of_circle_areas (m : ℝ) (n : ℕ) : ℝ :=
  sorry

/-- The limit of the sum as n approaches infinity --/
def limit_of_sum (m : ℝ) : ℝ :=
  sorry

/-- Theorem: The limit of the sum of areas of inscribed circles approaches 5πm^2 --/
theorem sum_of_inscribed_circles_limit (m : ℝ) (h : m > 0) :
  limit_of_sum m = 5 * Real.pi * m^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_inscribed_circles_limit_l3519_351957


namespace NUMINAMATH_CALUDE_club_officer_selection_l3519_351962

theorem club_officer_selection (n : ℕ) (e : ℕ) (h1 : n = 12) (h2 : e = 5) (h3 : e ≤ n) :
  (n * (n - 1) * e * (n - 2)) = 6600 :=
sorry

end NUMINAMATH_CALUDE_club_officer_selection_l3519_351962


namespace NUMINAMATH_CALUDE_equation_solution_l3519_351956

theorem equation_solution : ∃ x : ℚ, 
  x = 81 / 16 ∧ 
  Real.sqrt x + 4 * Real.sqrt (x^2 + 9*x) + Real.sqrt (x + 9) = 45 - 2*x :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3519_351956


namespace NUMINAMATH_CALUDE_ab_value_is_32_l3519_351914

def is_distinct (a b c d e f : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f

def is_valid_digit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 6

theorem ab_value_is_32 :
  ∃ (a b c d e f : ℕ),
    is_distinct a b c d e f ∧
    is_valid_digit a ∧ is_valid_digit b ∧ is_valid_digit c ∧
    is_valid_digit d ∧ is_valid_digit e ∧ is_valid_digit f ∧
    (10 * a + b) * ((10 * c + d) - e) + f = 2021 ∧
    10 * a + b = 32 :=
by sorry

end NUMINAMATH_CALUDE_ab_value_is_32_l3519_351914


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l3519_351976

theorem complex_expression_simplification (x : ℝ) :
  x * (x * (x * (3 - x) - 3) + 5) + 1 = -x^4 + 3*x^3 - 3*x^2 + 5*x + 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l3519_351976


namespace NUMINAMATH_CALUDE_fuel_mixture_problem_l3519_351921

theorem fuel_mixture_problem (tank_capacity : ℝ) (ethanol_A : ℝ) (ethanol_B : ℝ) (total_ethanol : ℝ) :
  tank_capacity = 218 →
  ethanol_A = 0.12 →
  ethanol_B = 0.16 →
  total_ethanol = 30 →
  ∃ (V_A : ℝ), V_A = 122 ∧
    ∃ (V_B : ℝ), V_A + V_B = tank_capacity ∧
    ethanol_A * V_A + ethanol_B * V_B = total_ethanol :=
by sorry

end NUMINAMATH_CALUDE_fuel_mixture_problem_l3519_351921


namespace NUMINAMATH_CALUDE_mailman_junk_mail_l3519_351980

/-- Given a total number of mail pieces and a number of magazines, 
    calculate the number of junk mail pieces. -/
def junk_mail (total : ℕ) (magazines : ℕ) : ℕ :=
  total - magazines

theorem mailman_junk_mail :
  junk_mail 11 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_mailman_junk_mail_l3519_351980


namespace NUMINAMATH_CALUDE_green_pepper_weight_is_half_total_l3519_351907

/-- The weight of green peppers bought by Hannah's Vegetarian Restaurant -/
def green_pepper_weight : ℝ := 0.33333333335

/-- The total weight of peppers bought by Hannah's Vegetarian Restaurant -/
def total_pepper_weight : ℝ := 0.6666666667

/-- Theorem stating that the weight of green peppers is half the total weight -/
theorem green_pepper_weight_is_half_total :
  green_pepper_weight = total_pepper_weight / 2 := by sorry

end NUMINAMATH_CALUDE_green_pepper_weight_is_half_total_l3519_351907


namespace NUMINAMATH_CALUDE_tens_digit_of_8_power_23_l3519_351973

theorem tens_digit_of_8_power_23 : ∃ n : ℕ, 8^23 = 10 * n + 12 :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_8_power_23_l3519_351973


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l3519_351913

theorem geometric_sequence_first_term
  (a : ℝ) -- first term
  (r : ℝ) -- common ratio
  (h1 : a * r^2 = 18) -- third term is 18
  (h2 : a * r^4 = 72) -- fifth term is 72
  : a = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l3519_351913


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l3519_351995

theorem least_subtraction_for_divisibility : 
  ∃! x : ℕ, x ≤ 86 ∧ (13605 - x) % 87 = 0 ∧ ∀ y : ℕ, y < x → (13605 - y) % 87 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l3519_351995


namespace NUMINAMATH_CALUDE_system_solution_l3519_351977

theorem system_solution : ∃ (x y : ℝ), (4 * x - y = 7) ∧ (3 * x + 4 * y = 10) ∧ (x = 2) ∧ (y = 1) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3519_351977


namespace NUMINAMATH_CALUDE_max_xy_value_l3519_351939

theorem max_xy_value (x y : ℕ+) (h : 7 * x + 4 * y = 140) : x * y ≤ 112 := by
  sorry

end NUMINAMATH_CALUDE_max_xy_value_l3519_351939


namespace NUMINAMATH_CALUDE_project_budget_increase_l3519_351909

/-- Proves that the annual increase in budget for project Q is $50,000 -/
theorem project_budget_increase (initial_q initial_v decrease_v : ℕ) 
  (h1 : initial_q = 540000)
  (h2 : initial_v = 780000)
  (h3 : decrease_v = 10000)
  (h4 : ∃ (increase_q : ℕ), initial_q + 4 * increase_q = initial_v - 4 * decrease_v) :
  ∃ (increase_q : ℕ), increase_q = 50000 := by
sorry


end NUMINAMATH_CALUDE_project_budget_increase_l3519_351909


namespace NUMINAMATH_CALUDE_triangle_similarity_l3519_351918

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Represents an excircle of a triangle -/
structure Excircle where
  center : Point
  radius : ℝ

/-- Defines if a triangle is acute and scalene -/
def isAcuteScalene (t : Triangle) : Prop := sorry

/-- Defines the C-excircle of a triangle -/
def cExcircle (t : Triangle) : Excircle := sorry

/-- Defines the B-excircle of a triangle -/
def bExcircle (t : Triangle) : Excircle := sorry

/-- Defines the intersection point of two lines -/
def lineIntersection (p1 p2 p3 p4 : Point) : Point := sorry

/-- Defines a point symmetric to another point with respect to a third point -/
def symmetricPoint (p center : Point) : Point := sorry

/-- Checks if two triangles are similar -/
def areSimilar (t1 t2 : Triangle) : Prop := sorry

/-- Main theorem -/
theorem triangle_similarity (t : Triangle) (h : isAcuteScalene t) :
  let c_ex := cExcircle t
  let b_ex := bExcircle t
  let M := sorry -- Point where C-excircle is tangent to AB
  let N := sorry -- Point where C-excircle is tangent to extension of BC
  let P := sorry -- Point where B-excircle is tangent to AC
  let Q := sorry -- Point where B-excircle is tangent to extension of BC
  let A1 := lineIntersection M N P Q
  let A2 := symmetricPoint t.A A1
  let B1 := sorry -- Defined analogously to A1
  let B2 := symmetricPoint t.B B1
  let C1 := sorry -- Defined analogously to A1
  let C2 := symmetricPoint t.C C1
  let t2 : Triangle := ⟨A2, B2, C2⟩
  areSimilar t t2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_similarity_l3519_351918


namespace NUMINAMATH_CALUDE_f_2023_of_2_eq_one_seventh_l3519_351979

-- Define the function f
def f (x : ℚ) : ℚ := (1 + x) / (1 - 3*x)

-- Define f_n recursively
def f_n : ℕ → (ℚ → ℚ)
  | 0 => f
  | 1 => λ x => f (f x)
  | (n+2) => λ x => f (f_n (n+1) x)

-- Theorem statement
theorem f_2023_of_2_eq_one_seventh : f_n 2023 2 = 1/7 := by sorry

end NUMINAMATH_CALUDE_f_2023_of_2_eq_one_seventh_l3519_351979


namespace NUMINAMATH_CALUDE_mias_socks_theorem_l3519_351927

/-- Represents the number of pairs of socks at each price point --/
structure SockInventory where
  one_dollar : ℕ
  two_dollar : ℕ
  three_dollar : ℕ
  four_dollar : ℕ

/-- Calculates the total number of pairs of socks --/
def total_pairs (s : SockInventory) : ℕ :=
  s.one_dollar + s.two_dollar + s.three_dollar + s.four_dollar

/-- Calculates the total cost of all socks --/
def total_cost (s : SockInventory) : ℕ :=
  s.one_dollar + 2 * s.two_dollar + 3 * s.three_dollar + 4 * s.four_dollar

/-- Checks if at least one pair of each type was bought --/
def at_least_one_each (s : SockInventory) : Prop :=
  s.one_dollar ≥ 1 ∧ s.two_dollar ≥ 1 ∧ s.three_dollar ≥ 1 ∧ s.four_dollar ≥ 1

theorem mias_socks_theorem (s : SockInventory) 
  (h1 : total_pairs s = 16)
  (h2 : total_cost s = 36)
  (h3 : at_least_one_each s) :
  s.one_dollar = 3 := by
  sorry

end NUMINAMATH_CALUDE_mias_socks_theorem_l3519_351927


namespace NUMINAMATH_CALUDE_shoe_color_probability_l3519_351932

theorem shoe_color_probability (n : ℕ) (h : n = 6) :
  let total_shoes := 2 * n
  let same_color_selections := n
  let total_selections := total_shoes.choose 2
  (same_color_selections : ℚ) / total_selections = 1 / 11 :=
by sorry

end NUMINAMATH_CALUDE_shoe_color_probability_l3519_351932


namespace NUMINAMATH_CALUDE_function_domain_range_equality_l3519_351948

/-- The function f(x) = x^2 - 2x + 2 -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 2

/-- The theorem stating that b = 2 for the given conditions -/
theorem function_domain_range_equality (b : ℝ) (h1 : b > 1) 
  (h2 : Set.Icc 1 b = Set.range f)
  (h3 : ∀ x, x ∈ Set.Icc 1 b → f x ∈ Set.Icc 1 b) : b = 2 := by
  sorry

#check function_domain_range_equality

end NUMINAMATH_CALUDE_function_domain_range_equality_l3519_351948


namespace NUMINAMATH_CALUDE_quadratic_vertex_x_coordinate_l3519_351959

/-- Given a quadratic function f(x) = ax^2 + bx + c that passes through 
    the points (2, 3), (8, -1), and (11, 8), prove that the x-coordinate 
    of its vertex is 142/23. -/
theorem quadratic_vertex_x_coordinate 
  (f : ℝ → ℝ) 
  (a b c : ℝ) 
  (h1 : ∀ x, f x = a * x^2 + b * x + c) 
  (h2 : f 2 = 3) 
  (h3 : f 8 = -1) 
  (h4 : f 11 = 8) : 
  -b / (2 * a) = 142 / 23 := by
sorry

end NUMINAMATH_CALUDE_quadratic_vertex_x_coordinate_l3519_351959


namespace NUMINAMATH_CALUDE_roll_two_dice_prob_at_least_one_two_l3519_351929

/-- The number of sides on each die -/
def num_sides : ℕ := 8

/-- The probability of rolling at least one 2 with two fair 8-sided dice -/
def prob_at_least_one_two : ℚ := 15 / 64

/-- Theorem stating the probability of rolling at least one 2 with two fair 8-sided dice -/
theorem roll_two_dice_prob_at_least_one_two :
  prob_at_least_one_two = 15 / 64 := by
  sorry

#check roll_two_dice_prob_at_least_one_two

end NUMINAMATH_CALUDE_roll_two_dice_prob_at_least_one_two_l3519_351929


namespace NUMINAMATH_CALUDE_track_circumference_l3519_351991

/-- Represents the circular track and the movement of A and B -/
structure CircularTrack where
  circumference : ℝ
  speed_A : ℝ
  speed_B : ℝ

/-- The conditions of the problem -/
def problem_conditions (track : CircularTrack) : Prop :=
  ∃ (first_meet second_meet : ℝ),
    -- B has traveled 150 yards at first meeting
    track.speed_B * first_meet = 150 ∧
    -- A is 90 yards away from completing one lap at second meeting
    track.speed_A * second_meet = track.circumference - 90 ∧
    -- B's total distance at second meeting
    track.speed_B * second_meet = track.circumference / 2 + 90 ∧
    -- A and B start from opposite points and move in opposite directions
    track.speed_A > 0 ∧ track.speed_B > 0

/-- The theorem to prove -/
theorem track_circumference :
  ∀ (track : CircularTrack),
    problem_conditions track →
    track.circumference = 720 := by
  sorry

end NUMINAMATH_CALUDE_track_circumference_l3519_351991


namespace NUMINAMATH_CALUDE_star_commutative_star_associative_star_identity_star_not_distributive_l3519_351993

-- Define the binary operation
def star (x y : ℝ) : ℝ := (x + 2) * (y + 2) - 2

-- Commutativity
theorem star_commutative : ∀ x y : ℝ, star x y = star y x := by sorry

-- Associativity
theorem star_associative : ∀ x y z : ℝ, star (star x y) z = star x (star y z) := by sorry

-- Identity element
theorem star_identity : ∃ e : ℝ, ∀ x : ℝ, star x e = x ∧ star e x = x := by sorry

-- Not distributive over addition
theorem star_not_distributive : ¬(∀ x y z : ℝ, star x (y + z) = star x y + star x z) := by sorry

end NUMINAMATH_CALUDE_star_commutative_star_associative_star_identity_star_not_distributive_l3519_351993


namespace NUMINAMATH_CALUDE_pencil_price_in_units_l3519_351944

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

end NUMINAMATH_CALUDE_pencil_price_in_units_l3519_351944


namespace NUMINAMATH_CALUDE_book_length_calculation_l3519_351970

theorem book_length_calculation (B₁ B₂ : ℕ) : 
  (2 : ℚ) / 3 * B₁ - (1 : ℚ) / 3 * B₁ = 90 →
  (3 : ℚ) / 4 * B₂ - (1 : ℚ) / 4 * B₂ = 120 →
  B₁ + B₂ = 510 := by
sorry

end NUMINAMATH_CALUDE_book_length_calculation_l3519_351970


namespace NUMINAMATH_CALUDE_range_of_2x_minus_3_l3519_351920

theorem range_of_2x_minus_3 (x : ℝ) (h : -1 < 2*x + 3 ∧ 2*x + 3 < 1) :
  ∃! (n : ℤ), ∃ (y : ℝ), 2*y - 3 = ↑n ∧ -1 < 2*y + 3 ∧ 2*y + 3 < 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_2x_minus_3_l3519_351920


namespace NUMINAMATH_CALUDE_correct_equation_l3519_351986

theorem correct_equation (a b : ℝ) : 3 * a^2 * b - 4 * b * a^2 = -a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_correct_equation_l3519_351986


namespace NUMINAMATH_CALUDE_adult_books_count_l3519_351999

theorem adult_books_count (total : ℕ) (children_percent : ℚ) (h1 : total = 160) (h2 : children_percent = 35 / 100) :
  (total : ℚ) * (1 - children_percent) = 104 := by
  sorry

end NUMINAMATH_CALUDE_adult_books_count_l3519_351999


namespace NUMINAMATH_CALUDE_half_x_is_32_implies_2x_is_128_l3519_351955

theorem half_x_is_32_implies_2x_is_128 (x : ℝ) (h : x / 2 = 32) : 2 * x = 128 := by
  sorry

end NUMINAMATH_CALUDE_half_x_is_32_implies_2x_is_128_l3519_351955


namespace NUMINAMATH_CALUDE_marble_probability_difference_l3519_351940

/-- The number of red marbles in the box -/
def red_marbles : ℕ := 501

/-- The number of black marbles in the box -/
def black_marbles : ℕ := 1501

/-- The number of blue marbles in the box -/
def blue_marbles : ℕ := 1000

/-- The total number of marbles in the box -/
def total_marbles : ℕ := red_marbles + black_marbles + blue_marbles

/-- The probability of drawing two marbles of the same color -/
def Ps : ℚ := (red_marbles.choose 2 + black_marbles.choose 2 + blue_marbles.choose 2) / total_marbles.choose 2

/-- The probability of drawing two marbles of different colors -/
def Pd : ℚ := 1 - Ps

/-- The theorem stating that the absolute difference between Ps and Pd is 2/9 -/
theorem marble_probability_difference : |Ps - Pd| = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_difference_l3519_351940


namespace NUMINAMATH_CALUDE_smallest_z_value_l3519_351972

/-- Given an equation of consecutive perfect cubes, find the smallest possible value of the largest cube. -/
theorem smallest_z_value (u w x y z : ℕ) : 
  u^3 + w^3 + x^3 + y^3 = z^3 ∧ 
  (∃ k : ℕ, u = k ∧ w = k + 1 ∧ x = k + 2 ∧ y = k + 3 ∧ z = k + 4) ∧
  0 < u ∧ u < w ∧ w < x ∧ x < y ∧ y < z →
  z = 6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_z_value_l3519_351972


namespace NUMINAMATH_CALUDE_square_plus_n_eq_square_plus_k_implies_m_le_n_l3519_351930

theorem square_plus_n_eq_square_plus_k_implies_m_le_n
  (k m n : ℕ+) 
  (h : m^2 + n = k^2 + k) : 
  m ≤ n := by
sorry

end NUMINAMATH_CALUDE_square_plus_n_eq_square_plus_k_implies_m_le_n_l3519_351930


namespace NUMINAMATH_CALUDE_ellipse_problem_l3519_351937

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/2 + y^2 = 1

-- Define points A, B, and P
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry
def P : ℝ × ℝ := (0, 1)

-- Define line AB
def lineAB (x y : ℝ) : Prop := sorry

-- Define line y = -x + 2
def intersectLine (x y : ℝ) : Prop := y = -x + 2

-- Define points C and D
def C : ℝ × ℝ := sorry
def D : ℝ × ℝ := sorry

-- Define slopes
def slopePA : ℝ := sorry
def slopeAB : ℝ := sorry
def slopePB : ℝ := sorry

-- Theorem statement
theorem ellipse_problem :
  ellipse A.1 A.2 ∧ 
  ellipse B.1 B.2 ∧ 
  A ≠ P ∧ 
  B ≠ P ∧ 
  lineAB 0 0 ∧
  intersectLine C.1 C.2 ∧
  intersectLine D.1 D.2 →
  (∃ k : ℝ, slopePA + slopePB = 2 * slopeAB) ∧
  (∃ minArea : ℝ, minArea = Real.sqrt 2 / 3 ∧ 
    ∀ area : ℝ, area ≥ minArea) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_problem_l3519_351937


namespace NUMINAMATH_CALUDE_statements_equivalence_l3519_351928

-- Define the propositions
variable (S : Prop) -- Saturn is visible from Earth tonight
variable (M : Prop) -- Mars is visible

-- Define the statements
def statement1 : Prop := S → ¬M
def statement2 : Prop := M → ¬S
def statement3 : Prop := ¬S ∨ ¬M

-- Theorem stating the equivalence of the statements
theorem statements_equivalence : statement1 S M ↔ statement2 S M ∧ statement3 S M := by
  sorry

end NUMINAMATH_CALUDE_statements_equivalence_l3519_351928


namespace NUMINAMATH_CALUDE_quadratic_root_property_l3519_351951

theorem quadratic_root_property (a : ℝ) : 
  a^2 - a - 50 = 0 → a^3 - 51*a = 50 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_property_l3519_351951


namespace NUMINAMATH_CALUDE_cheerleader_ratio_is_half_l3519_351919

/-- Represents the number of cheerleaders for each uniform size -/
structure CheerleaderCounts where
  total : ℕ
  size2 : ℕ
  size6 : ℕ
  size12 : ℕ

/-- The ratio of cheerleaders needing size 12 to those needing size 6 -/
def size12to6Ratio (counts : CheerleaderCounts) : ℚ :=
  counts.size12 / counts.size6

/-- Theorem stating the ratio of cheerleaders needing size 12 to those needing size 6 -/
theorem cheerleader_ratio_is_half (counts : CheerleaderCounts)
  (h_total : counts.total = 19)
  (h_size2 : counts.size2 = 4)
  (h_size6 : counts.size6 = 10)
  (h_sum : counts.total = counts.size2 + counts.size6 + counts.size12) :
  size12to6Ratio counts = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_cheerleader_ratio_is_half_l3519_351919
