import Mathlib

namespace NUMINAMATH_CALUDE_jose_profit_share_l491_49160

structure Partner where
  investment : ℕ
  duration : ℕ

def totalInvestmentTime (partners : List Partner) : ℕ :=
  partners.foldl (fun acc p => acc + p.investment * p.duration) 0

def profitShare (partner : Partner) (partners : List Partner) (totalProfit : ℕ) : ℚ :=
  (partner.investment * partner.duration : ℚ) / (totalInvestmentTime partners : ℚ) * totalProfit

theorem jose_profit_share :
  let tom : Partner := { investment := 30000, duration := 12 }
  let jose : Partner := { investment := 45000, duration := 10 }
  let angela : Partner := { investment := 60000, duration := 8 }
  let rebecca : Partner := { investment := 75000, duration := 6 }
  let partners : List Partner := [tom, jose, angela, rebecca]
  let totalProfit : ℕ := 72000
  abs (profitShare jose partners totalProfit - 18620.69) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_jose_profit_share_l491_49160


namespace NUMINAMATH_CALUDE_square_arrangement_exists_l491_49177

-- Define the structure of the square
structure Square where
  bottomLeft : ℕ
  topRight : ℕ
  bottomRight : ℕ
  topLeft : ℕ
  center : ℕ

-- Define the property of having a common divisor greater than 1
def hasCommonDivisorGreaterThanOne (m n : ℕ) : Prop :=
  ∃ k : ℕ, k > 1 ∧ k ∣ m ∧ k ∣ n

-- Define the property of being relatively prime
def isRelativelyPrime (m n : ℕ) : Prop :=
  Nat.gcd m n = 1

-- Main theorem
theorem square_arrangement_exists : ∃ (a b c d : ℕ), ∃ (s : Square),
  s.bottomLeft = a * b ∧
  s.topRight = c * d ∧
  s.bottomRight = a * d ∧
  s.topLeft = b * c ∧
  s.center = a * b * c * d ∧
  (hasCommonDivisorGreaterThanOne s.bottomLeft s.center) ∧
  (hasCommonDivisorGreaterThanOne s.topRight s.center) ∧
  (hasCommonDivisorGreaterThanOne s.bottomRight s.center) ∧
  (hasCommonDivisorGreaterThanOne s.topLeft s.center) ∧
  (isRelativelyPrime s.bottomLeft s.topRight) ∧
  (isRelativelyPrime s.bottomRight s.topLeft) :=
sorry

end NUMINAMATH_CALUDE_square_arrangement_exists_l491_49177


namespace NUMINAMATH_CALUDE_fraction_simplification_l491_49192

theorem fraction_simplification :
  (1722^2 - 1715^2) / (1731^2 - 1706^2) = 7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l491_49192


namespace NUMINAMATH_CALUDE_quadratic_function_property_l491_49184

theorem quadratic_function_property (a b : ℝ) (h1 : a ≠ b) : 
  let f : ℝ → ℝ := λ x ↦ x^2 + a*x + b
  (f a = f b) → f 2 = 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l491_49184


namespace NUMINAMATH_CALUDE_car_distance_l491_49157

theorem car_distance (total_distance : ℝ) (foot_fraction : ℝ) (bus_fraction : ℝ) :
  total_distance = 90 →
  foot_fraction = 1/5 →
  bus_fraction = 2/3 →
  total_distance * (1 - foot_fraction - bus_fraction) = 12 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_l491_49157


namespace NUMINAMATH_CALUDE_triangle_side_length_expression_l491_49181

/-- Given a triangle with side lengths a, b, and c, 
    the expression |a-b+c| - |a-b-c| simplifies to 2a - 2b -/
theorem triangle_side_length_expression (a b c : ℝ) 
  (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  |a - b + c| - |a - b - c| = 2*a - 2*b := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_expression_l491_49181


namespace NUMINAMATH_CALUDE_cake_brownie_calorie_difference_l491_49137

def cake_slices : ℕ := 8
def calories_per_cake_slice : ℕ := 347
def brownies : ℕ := 6
def calories_per_brownie : ℕ := 375

theorem cake_brownie_calorie_difference :
  cake_slices * calories_per_cake_slice - brownies * calories_per_brownie = 526 := by
sorry

end NUMINAMATH_CALUDE_cake_brownie_calorie_difference_l491_49137


namespace NUMINAMATH_CALUDE_nights_stayed_is_five_l491_49104

/-- Represents a date in May or June --/
inductive Date
| may (day : Nat)
| june (day : Nat)

def Date.daysInMay : Nat := 31

/-- Calculates the number of nights between two dates --/
def nightsBetween (arrival : Date) (departure : Date) : Nat :=
  match arrival, departure with
  | Date.may arrivalDay, Date.june departureDay =>
      (Date.daysInMay - arrivalDay) + departureDay
  | _, _ => 0  -- Handle other cases (should not occur in this problem)

theorem nights_stayed_is_five :
  let arrival := Date.may 30
  let departure := Date.june 4
  nightsBetween arrival departure = 5 := by
  sorry

#eval nightsBetween (Date.may 30) (Date.june 4)

end NUMINAMATH_CALUDE_nights_stayed_is_five_l491_49104


namespace NUMINAMATH_CALUDE_total_defective_rate_proof_l491_49102

/-- The fraction of products checked by worker y -/
def worker_y_fraction : ℝ := 0.1666666666666668

/-- The defective rate for products checked by worker x -/
def worker_x_defective_rate : ℝ := 0.005

/-- The defective rate for products checked by worker y -/
def worker_y_defective_rate : ℝ := 0.008

/-- The total defective rate for all products -/
def total_defective_rate : ℝ := 0.0055

theorem total_defective_rate_proof :
  (1 - worker_y_fraction) * worker_x_defective_rate +
  worker_y_fraction * worker_y_defective_rate = total_defective_rate := by
  sorry

end NUMINAMATH_CALUDE_total_defective_rate_proof_l491_49102


namespace NUMINAMATH_CALUDE_calculator_problem_l491_49132

/-- Represents the possible operations on the calculator --/
inductive Operation
| addOne
| addThree
| double

/-- Applies a single operation to a number --/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.addOne => n + 1
  | Operation.addThree => n + 3
  | Operation.double => n * 2

/-- Applies a sequence of operations to a number --/
def applySequence (start : ℕ) (ops : List Operation) : ℕ :=
  ops.foldl applyOperation start

/-- Checks if a sequence of operations transforms start into target --/
def isValidSequence (start target : ℕ) (ops : List Operation) : Prop :=
  applySequence start ops = target

/-- The main theorem to be proved --/
theorem calculator_problem :
  ∃ (ops : List Operation),
    ops.length = 10 ∧
    isValidSequence 1 410 ops ∧
    ∀ (shorter_ops : List Operation),
      shorter_ops.length < 10 →
      ¬ isValidSequence 1 410 shorter_ops :=
sorry

end NUMINAMATH_CALUDE_calculator_problem_l491_49132


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_cubic_equation_solutions_l491_49153

theorem quadratic_equation_solutions (x : ℝ) :
  (x^2 + 2*x - 4 = 0) ↔ (x = Real.sqrt 5 - 1 ∨ x = -Real.sqrt 5 - 1) :=
sorry

theorem cubic_equation_solutions (x : ℝ) :
  (3*x*(x-5) = 5-x) ↔ (x = 5 ∨ x = -1/3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_cubic_equation_solutions_l491_49153


namespace NUMINAMATH_CALUDE_circuit_length_difference_l491_49165

/-- The length of the small circuit in meters -/
def small_circuit_length : ℕ := 400

/-- The number of laps Jana runs -/
def jana_laps : ℕ := 3

/-- The number of laps Father runs -/
def father_laps : ℕ := 4

/-- The total distance Jana runs in meters -/
def jana_distance : ℕ := small_circuit_length * jana_laps

/-- The total distance Father runs in meters -/
def father_distance : ℕ := 2 * jana_distance

/-- The length of the large circuit in meters -/
def large_circuit_length : ℕ := father_distance / father_laps

theorem circuit_length_difference :
  large_circuit_length - small_circuit_length = 200 := by
  sorry

end NUMINAMATH_CALUDE_circuit_length_difference_l491_49165


namespace NUMINAMATH_CALUDE_unique_geometric_sequence_l491_49145

/-- A geometric sequence with the given properties has a unique first term of 1/3 -/
theorem unique_geometric_sequence (a : ℝ) (a_n : ℕ → ℝ) : 
  a > 0 ∧ 
  (∀ n, a_n (n + 1) = a_n n * (a_n 2 / a_n 1)) ∧ 
  a_n 1 = a ∧
  (∃ q : ℝ, (a_n 1 + 1) * q = a_n 2 + 2 ∧ (a_n 2 + 2) * q = a_n 3 + 3) ∧
  (∃! q : ℝ, q ≠ 0 ∧ a * q^2 - 4 * a * q + 3 * a - 1 = 0) →
  a = 1/3 := by
sorry

end NUMINAMATH_CALUDE_unique_geometric_sequence_l491_49145


namespace NUMINAMATH_CALUDE_brownie_pieces_count_l491_49161

/-- Represents the dimensions of a rectangular shape -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangular shape given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Calculates the number of smaller rectangles that can fit into a larger rectangle -/
def number_of_pieces (tray : Dimensions) (piece : Dimensions) : ℕ :=
  (area tray) / (area piece)

theorem brownie_pieces_count :
  let tray := Dimensions.mk 24 16
  let piece := Dimensions.mk 2 2
  number_of_pieces tray piece = 96 := by
  sorry

end NUMINAMATH_CALUDE_brownie_pieces_count_l491_49161


namespace NUMINAMATH_CALUDE_min_value_x_plus_inverse_equality_condition_l491_49134

theorem min_value_x_plus_inverse (x : ℝ) (hx : x > 0) : x + 1/x ≥ 2 :=
by
  sorry

theorem equality_condition (x : ℝ) (hx : x > 0) : x + 1/x = 2 ↔ x = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_inverse_equality_condition_l491_49134


namespace NUMINAMATH_CALUDE_company_fund_problem_l491_49197

theorem company_fund_problem (n : ℕ) (initial_fund : ℕ) : 
  initial_fund = 60 * n - 10 →  -- The fund initially contained $10 less than needed for $60 bonuses
  initial_fund = 55 * n + 120 → -- Each employee received $55, and $120 remained
  initial_fund = 1550 :=        -- The initial fund amount was $1550
by sorry

end NUMINAMATH_CALUDE_company_fund_problem_l491_49197


namespace NUMINAMATH_CALUDE_min_value_of_sum_equality_condition_l491_49114

theorem min_value_of_sum (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b) / c + (a + c) / d + (b + d) / a + (c + d) / b ≥ 8 :=
by sorry

theorem equality_condition (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b) / c + (a + c) / d + (b + d) / a + (c + d) / b = 8 ↔ a = b ∧ b = c ∧ c = d :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_equality_condition_l491_49114


namespace NUMINAMATH_CALUDE_library_book_distribution_l491_49199

/-- The number of ways to distribute books between the library and checked out -/
def distributeBooks (total : ℕ) (minInLibrary : ℕ) (minCheckedOut : ℕ) : ℕ :=
  (total - minInLibrary - minCheckedOut + 1)

/-- Theorem: There are 6 ways to distribute 10 books with at least 2 in the library and 3 checked out -/
theorem library_book_distribution :
  distributeBooks 10 2 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_library_book_distribution_l491_49199


namespace NUMINAMATH_CALUDE_condition_relation_l491_49194

theorem condition_relation (A B C : Prop) 
  (h1 : B → A)  -- A is a necessary condition for B
  (h2 : C → B)  -- C is a sufficient condition for B
  (h3 : ¬(B → C))  -- C is not a necessary condition for B
  : (C → A) ∧ ¬(A → C) := by
  sorry

end NUMINAMATH_CALUDE_condition_relation_l491_49194


namespace NUMINAMATH_CALUDE_square_area_l491_49143

theorem square_area (side_length : ℝ) (h : side_length = 7) : side_length ^ 2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_square_area_l491_49143


namespace NUMINAMATH_CALUDE_rectangle_opposite_sides_equal_square_all_sides_equal_l491_49139

-- Define a rectangle
structure Rectangle where
  width : ℝ
  height : ℝ

-- Define a square
structure Square where
  side : ℝ

-- Theorem for rectangle
theorem rectangle_opposite_sides_equal (r : Rectangle) : 
  r.width = r.width ∧ r.height = r.height := by
  sorry

-- Theorem for square
theorem square_all_sides_equal (s : Square) : 
  s.side = s.side ∧ s.side = s.side ∧ s.side = s.side ∧ s.side = s.side := by
  sorry

end NUMINAMATH_CALUDE_rectangle_opposite_sides_equal_square_all_sides_equal_l491_49139


namespace NUMINAMATH_CALUDE_inscribed_tetrahedron_volume_ratio_l491_49164

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  sideLength : ℝ
  sideLength_pos : sideLength > 0

/-- Given two regular tetrahedra where one is inscribed inside the other
    such that its vertices are at the midpoints of the edges of the larger tetrahedron,
    the ratio of their volumes is 1/8 -/
theorem inscribed_tetrahedron_volume_ratio
  (large : RegularTetrahedron) (small : RegularTetrahedron)
  (h : small.sideLength = large.sideLength / 2) :
  (small.sideLength ^ 3) / (large.sideLength ^ 3) = 1 / 8 := by
  sorry

#check inscribed_tetrahedron_volume_ratio

end NUMINAMATH_CALUDE_inscribed_tetrahedron_volume_ratio_l491_49164


namespace NUMINAMATH_CALUDE_square_side_length_l491_49186

/-- Given a square ABCD with specific points and conditions, prove its side length is 10 -/
theorem square_side_length (A B C D P Q R S Z : ℝ × ℝ) : 
  (∃ s : ℝ, 
    -- Square ABCD
    A = (0, 0) ∧ B = (s, 0) ∧ C = (s, s) ∧ D = (0, s) ∧
    -- P on AB, Q on BC, R on CD, S on DA
    (∃ t₁ : ℝ, 0 ≤ t₁ ∧ t₁ ≤ 1 ∧ P = (t₁ * s, 0)) ∧
    (∃ t₂ : ℝ, 0 ≤ t₂ ∧ t₂ ≤ 1 ∧ Q = (s, (1 - t₂) * s)) ∧
    (∃ t₃ : ℝ, 0 ≤ t₃ ∧ t₃ ≤ 1 ∧ R = ((1 - t₃) * s, s)) ∧
    (∃ t₄ : ℝ, 0 ≤ t₄ ∧ t₄ ≤ 1 ∧ S = (0, t₄ * s)) ∧
    -- PR parallel to BC, SQ parallel to AB
    (R.1 - P.1) * (C.2 - B.2) = (R.2 - P.2) * (C.1 - B.1) ∧
    (Q.1 - S.1) * (B.2 - A.2) = (Q.2 - S.2) * (B.1 - A.1) ∧
    -- Z is intersection of PR and SQ
    (Z.1 - P.1) * (R.2 - P.2) = (Z.2 - P.2) * (R.1 - P.1) ∧
    (Z.1 - S.1) * (Q.2 - S.2) = (Z.2 - S.2) * (Q.1 - S.1) ∧
    -- Given distances
    ‖B - P‖ = 7 ∧
    ‖B - Q‖ = 6 ∧
    ‖D - Z‖ = 5) →
  s = 10 := by
sorry

end NUMINAMATH_CALUDE_square_side_length_l491_49186


namespace NUMINAMATH_CALUDE_pieces_left_l491_49116

/-- The number of medieval art pieces Alicia originally had -/
def original_pieces : ℕ := 70

/-- The number of medieval art pieces Alicia donated -/
def donated_pieces : ℕ := 46

/-- Theorem: The number of medieval art pieces Alicia has left is 24 -/
theorem pieces_left : original_pieces - donated_pieces = 24 := by
  sorry

end NUMINAMATH_CALUDE_pieces_left_l491_49116


namespace NUMINAMATH_CALUDE_complex_modulus_equality_l491_49142

theorem complex_modulus_equality (x y : ℝ) (h : (1 + Complex.I) * x = 1 + y * Complex.I) :
  Complex.abs (x + y * Complex.I) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equality_l491_49142


namespace NUMINAMATH_CALUDE_mango_boxes_count_l491_49129

/-- Given a number of mangoes per dozen, total mangoes, and mangoes per box,
    calculate the number of boxes. -/
def calculate_boxes (mangoes_per_dozen : ℕ) (total_mangoes : ℕ) (dozens_per_box : ℕ) : ℕ :=
  total_mangoes / (mangoes_per_dozen * dozens_per_box)

/-- Prove that there are 36 boxes of mangoes given the problem conditions. -/
theorem mango_boxes_count :
  let mangoes_per_dozen : ℕ := 12
  let total_mangoes : ℕ := 4320
  let dozens_per_box : ℕ := 10
  calculate_boxes mangoes_per_dozen total_mangoes dozens_per_box = 36 := by
  sorry

#eval calculate_boxes 12 4320 10

end NUMINAMATH_CALUDE_mango_boxes_count_l491_49129


namespace NUMINAMATH_CALUDE_negative_reals_inequality_l491_49170

theorem negative_reals_inequality (a b c : ℝ) (ha : a < 0) (hb : b < 0) (hc : c < 0) :
  a + b + c ≤ (a^2 + b^2) / (2*c) + (b^2 + c^2) / (2*a) + (c^2 + a^2) / (2*b) ∧
  (a^2 + b^2) / (2*c) + (b^2 + c^2) / (2*a) + (c^2 + a^2) / (2*b) ≤ a^2 / (b*c) + b^2 / (c*a) + c^2 / (a*b) :=
by sorry

end NUMINAMATH_CALUDE_negative_reals_inequality_l491_49170


namespace NUMINAMATH_CALUDE_correct_operation_l491_49119

theorem correct_operation (a : ℝ) : 2 * a^2 * (3 * a) = 6 * a^3 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l491_49119


namespace NUMINAMATH_CALUDE_parallelogram_area_l491_49111

/-- Represents a parallelogram with given dimensions -/
structure Parallelogram where
  base : ℝ
  height : ℝ
  shift : ℝ

/-- Calculates the area of a parallelogram -/
def area (p : Parallelogram) : ℝ := p.base * p.height

/-- Theorem: The area of the specific parallelogram is 140 square feet -/
theorem parallelogram_area :
  let p := Parallelogram.mk 20 7 8
  area p = 140 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l491_49111


namespace NUMINAMATH_CALUDE_expected_bounces_l491_49141

/-- The expected number of bounces for a ball on a rectangular billiard table -/
theorem expected_bounces (table_length table_width ball_travel : ℝ) 
  (h_length : table_length = 3)
  (h_width : table_width = 1)
  (h_travel : ball_travel = 2) :
  ∃ (E : ℝ), E = 1 + (2 / Real.pi) * (Real.arccos (3/4) + Real.arccos (1/4) - Real.arcsin (3/4)) :=
by sorry

end NUMINAMATH_CALUDE_expected_bounces_l491_49141


namespace NUMINAMATH_CALUDE_hockey_league_games_l491_49189

theorem hockey_league_games (n : ℕ) (total_games : ℕ) (h1 : n = 15) (h2 : total_games = 1050) :
  ∃ (games_per_pair : ℕ), 
    games_per_pair * (n * (n - 1) / 2) = total_games ∧ 
    games_per_pair = 10 := by
sorry

end NUMINAMATH_CALUDE_hockey_league_games_l491_49189


namespace NUMINAMATH_CALUDE_circle_center_coordinate_sum_l491_49187

/-- Given a circle with diameter endpoints (10, -6) and (-6, 2), 
    the sum of the coordinates of its center is 0. -/
theorem circle_center_coordinate_sum : 
  let x1 : ℝ := 10
  let y1 : ℝ := -6
  let x2 : ℝ := -6
  let y2 : ℝ := 2
  let center_x : ℝ := (x1 + x2) / 2
  let center_y : ℝ := (y1 + y2) / 2
  center_x + center_y = 0 := by sorry

end NUMINAMATH_CALUDE_circle_center_coordinate_sum_l491_49187


namespace NUMINAMATH_CALUDE_smallest_base_perfect_square_l491_49180

theorem smallest_base_perfect_square : 
  ∃ (b : ℕ), b > 3 ∧ 
  (∃ (n : ℕ), 4 * b + 5 = n^2) ∧ 
  (∀ (x : ℕ), x > 3 ∧ x < b → ¬∃ (m : ℕ), 4 * x + 5 = m^2) ∧
  b = 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_base_perfect_square_l491_49180


namespace NUMINAMATH_CALUDE_errors_per_debug_session_l491_49130

theorem errors_per_debug_session 
  (total_lines : ℕ) 
  (debug_interval : ℕ) 
  (total_errors : ℕ) 
  (h1 : total_lines = 4300)
  (h2 : debug_interval = 100)
  (h3 : total_errors = 129) :
  total_errors / (total_lines / debug_interval) = 3 := by
sorry

end NUMINAMATH_CALUDE_errors_per_debug_session_l491_49130


namespace NUMINAMATH_CALUDE_min_value_quadratic_roots_l491_49182

theorem min_value_quadratic_roots (m : ℝ) (x₁ x₂ : ℝ) : 
  (x₁^2 + 2*m*x₁ + m^2 + 3*m - 2 = 0) →
  (x₂^2 + 2*m*x₂ + m^2 + 3*m - 2 = 0) →
  (∃ (min : ℝ), ∀ (m : ℝ), x₁*(x₂ + x₁) + x₂^2 ≥ min ∧ 
  ∃ (m₀ : ℝ), x₁*(x₂ + x₁) + x₂^2 = min) →
  (∃ (min : ℝ), min = 5/4 ∧ 
  ∀ (m : ℝ), x₁*(x₂ + x₁) + x₂^2 ≥ min ∧ 
  ∃ (m₀ : ℝ), x₁*(x₂ + x₁) + x₂^2 = min) :=
sorry

end NUMINAMATH_CALUDE_min_value_quadratic_roots_l491_49182


namespace NUMINAMATH_CALUDE_lunch_calories_calculation_l491_49124

def daily_calorie_allowance : ℕ := 2200
def breakfast_calories : ℕ := 353
def snack_calories : ℕ := 130
def dinner_calories_left : ℕ := 832

theorem lunch_calories_calculation : 
  daily_calorie_allowance - breakfast_calories - snack_calories - dinner_calories_left = 885 := by
  sorry

end NUMINAMATH_CALUDE_lunch_calories_calculation_l491_49124


namespace NUMINAMATH_CALUDE_larger_number_problem_l491_49121

theorem larger_number_problem (x y : ℝ) 
  (sum : x + y = 40)
  (diff : x - y = 10)
  (prod : x * y = 375)
  (greater : x > y) : x = 25 := by
sorry

end NUMINAMATH_CALUDE_larger_number_problem_l491_49121


namespace NUMINAMATH_CALUDE_sales_growth_rate_l491_49179

theorem sales_growth_rate (x : ℝ) : (1 + x)^2 = 1 + 0.44 → x < 0.22 := by
  sorry

end NUMINAMATH_CALUDE_sales_growth_rate_l491_49179


namespace NUMINAMATH_CALUDE_car_trading_problem_l491_49120

-- Define the profit per car for models A and B (in thousand yuan)
variable (profit_A profit_B : ℚ)

-- Define the number of cars of each model
variable (num_A num_B : ℕ)

-- Define the given conditions
axiom profit_condition_1 : 3 * profit_A + 2 * profit_B = 34
axiom profit_condition_2 : profit_A + 4 * profit_B = 28

-- Define the purchase prices (in thousand yuan)
def price_A : ℚ := 160
def price_B : ℚ := 140

-- Define the total number of cars and budget (in thousand yuan)
def total_cars : ℕ := 30
def max_budget : ℚ := 4400

-- Define the minimum profit (in thousand yuan)
def min_profit : ℚ := 177

-- Theorem statement
theorem car_trading_problem :
  (profit_A = 8 ∧ profit_B = 5) ∧
  ((num_A = 9 ∧ num_B = 21) ∨ (num_A = 10 ∧ num_B = 20)) ∧
  (num_A + num_B = total_cars) ∧
  (num_A * price_A + num_B * price_B ≤ max_budget) ∧
  (num_A * profit_A + num_B * profit_B ≥ min_profit) :=
sorry

end NUMINAMATH_CALUDE_car_trading_problem_l491_49120


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l491_49195

/-- If a, b, and c form a geometric sequence, then ax^2 + bx + c has no real roots -/
theorem quadratic_no_real_roots (a b c : ℝ) (h : b^2 = a*c) : 
  ∀ x : ℝ, a*x^2 + b*x + c ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l491_49195


namespace NUMINAMATH_CALUDE_roots_of_cubic_polynomial_l491_49101

theorem roots_of_cubic_polynomial :
  let p : ℝ → ℝ := λ x => x^3 - 2*x^2 - 5*x + 6
  (∀ x : ℝ, p x = 0 ↔ x = 1 ∨ x = -2 ∨ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_roots_of_cubic_polynomial_l491_49101


namespace NUMINAMATH_CALUDE_total_hot_dog_cost_l491_49125

def hot_dog_cost (group : ℕ) (quantity : ℕ) (price : ℚ) : ℚ :=
  quantity * price

theorem total_hot_dog_cost : 
  let group1_cost := hot_dog_cost 1 4 0.60
  let group2_cost := hot_dog_cost 2 5 0.75
  let group3_cost := hot_dog_cost 3 3 0.90
  group1_cost + group2_cost + group3_cost = 8.85 := by
  sorry

end NUMINAMATH_CALUDE_total_hot_dog_cost_l491_49125


namespace NUMINAMATH_CALUDE_kates_retirement_fund_l491_49155

/-- 
Given an initial retirement fund value and a decrease amount, 
calculate the current value of the retirement fund.
-/
def current_fund_value (initial_value decrease : ℕ) : ℕ :=
  initial_value - decrease

/-- 
Theorem: Given Kate's initial retirement fund value of $1472 and a decrease of $12, 
the current value of her retirement fund is $1460.
-/
theorem kates_retirement_fund : 
  current_fund_value 1472 12 = 1460 := by
  sorry

end NUMINAMATH_CALUDE_kates_retirement_fund_l491_49155


namespace NUMINAMATH_CALUDE_infinitely_many_primes_4k_minus_1_l491_49113

theorem infinitely_many_primes_4k_minus_1 : 
  Set.Infinite {p : ℕ | Nat.Prime p ∧ ∃ k : ℕ, p = 4 * k - 1} := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_4k_minus_1_l491_49113


namespace NUMINAMATH_CALUDE_rectangle_cannot_fit_l491_49135

theorem rectangle_cannot_fit (square_area : ℝ) (rect_area : ℝ) (ratio : ℝ) : 
  square_area = 400 ∧ rect_area = 300 ∧ ratio = 3/2 →
  ∃ (square_side : ℝ) (rect_length : ℝ) (rect_width : ℝ),
    square_side^2 = square_area ∧
    rect_length * rect_width = rect_area ∧
    rect_length / rect_width = ratio ∧
    rect_length > square_side :=
by
  sorry

#check rectangle_cannot_fit

end NUMINAMATH_CALUDE_rectangle_cannot_fit_l491_49135


namespace NUMINAMATH_CALUDE_total_books_count_l491_49115

/-- The total number of Iesha's books -/
def total_books : ℕ := sorry

/-- The number of Iesha's school books -/
def school_books : ℕ := 19

/-- The number of Iesha's sports books -/
def sports_books : ℕ := 39

/-- Theorem: The total number of Iesha's books is 58 -/
theorem total_books_count : total_books = school_books + sports_books ∧ total_books = 58 := by
  sorry

end NUMINAMATH_CALUDE_total_books_count_l491_49115


namespace NUMINAMATH_CALUDE_football_player_average_increase_l491_49176

theorem football_player_average_increase :
  ∀ (total_goals : ℕ) (goals_fifth_match : ℕ) (num_matches : ℕ),
    total_goals = 16 →
    goals_fifth_match = 4 →
    num_matches = 5 →
    (total_goals : ℚ) / num_matches - ((total_goals - goals_fifth_match) : ℚ) / (num_matches - 1) = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_football_player_average_increase_l491_49176


namespace NUMINAMATH_CALUDE_product_digits_sum_l491_49146

/-- Converts a base-7 number to decimal --/
def base7ToDecimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to base-7 --/
def decimalToBase7 (n : ℕ) : ℕ := sorry

/-- Computes the sum of digits of a base-7 number --/
def sumOfDigitsBase7 (n : ℕ) : ℕ := sorry

/-- The product of 24₇ and 35₇ in base-7 --/
def productBase7 : ℕ := decimalToBase7 (base7ToDecimal 24 * base7ToDecimal 35)

theorem product_digits_sum :
  sumOfDigitsBase7 productBase7 = 15 :=
sorry

end NUMINAMATH_CALUDE_product_digits_sum_l491_49146


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_tangent_l491_49150

/-- The value of n for which the ellipse 2x^2 + 3y^2 = 6 and the hyperbola 3x^2 - n(y-1)^2 = 3 are tangent -/
def tangent_n : ℝ := -6

/-- The equation of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := 2 * x^2 + 3 * y^2 = 6

/-- The equation of the hyperbola -/
def is_on_hyperbola (x y n : ℝ) : Prop := 3 * x^2 - n * (y - 1)^2 = 3

/-- Two curves are tangent if they intersect at exactly one point -/
def are_tangent (f g : ℝ → ℝ → Prop) : Prop :=
  ∃! p : ℝ × ℝ, f p.1 p.2 ∧ g p.1 p.2

theorem ellipse_hyperbola_tangent :
  are_tangent (λ x y => is_on_ellipse x y) (λ x y => is_on_hyperbola x y tangent_n) :=
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_tangent_l491_49150


namespace NUMINAMATH_CALUDE_dorchester_earnings_l491_49123

def daily_fixed_pay : ℝ := 40
def pay_per_puppy : ℝ := 2.25
def puppies_washed : ℕ := 16

theorem dorchester_earnings :
  daily_fixed_pay + pay_per_puppy * (puppies_washed : ℝ) = 76 := by
  sorry

end NUMINAMATH_CALUDE_dorchester_earnings_l491_49123


namespace NUMINAMATH_CALUDE_factor_x4_plus_64_l491_49122

theorem factor_x4_plus_64 (x : ℝ) : 
  x^4 + 64 = (x^2 + 4*x + 8) * (x^2 - 4*x + 8) := by
sorry

end NUMINAMATH_CALUDE_factor_x4_plus_64_l491_49122


namespace NUMINAMATH_CALUDE_spiral_stripe_length_l491_49191

/-- The length of a spiral stripe on a right circular cylinder -/
theorem spiral_stripe_length (base_circumference height : ℝ) (h1 : base_circumference = 18) (h2 : height = 8) :
  let stripe_length := Real.sqrt (height^2 + (2 * base_circumference)^2)
  stripe_length = Real.sqrt 1360 := by
sorry

end NUMINAMATH_CALUDE_spiral_stripe_length_l491_49191


namespace NUMINAMATH_CALUDE_young_worker_proportion_is_three_fifths_l491_49196

/-- The proportion of young workers in a steel works -/
def young_worker_proportion : ℚ := 3/5

/-- The statement that the proportion of young workers is three-fifths -/
theorem young_worker_proportion_is_three_fifths : 
  young_worker_proportion = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_young_worker_proportion_is_three_fifths_l491_49196


namespace NUMINAMATH_CALUDE_at_least_one_greater_than_one_l491_49159

theorem at_least_one_greater_than_one (x y : ℝ) (h : x + y > 2) :
  x > 1 ∨ y > 1 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_greater_than_one_l491_49159


namespace NUMINAMATH_CALUDE_adams_total_school_time_l491_49118

/-- The time Adam spent at school on each day of the week --/
structure SchoolWeek where
  monday : Float
  tuesday : Float
  wednesday : Float
  thursday : Float
  friday : Float

/-- Calculate the total time Adam spent at school during the week --/
def totalSchoolTime (week : SchoolWeek) : Float :=
  week.monday + week.tuesday + week.wednesday + week.thursday + week.friday

/-- Adam's actual school week --/
def adamsWeek : SchoolWeek := {
  monday := 7.75,
  tuesday := 5.75,
  wednesday := 13.5,
  thursday := 8,
  friday := 6.75
}

/-- Theorem stating that Adam's total school time for the week is 41.75 hours --/
theorem adams_total_school_time :
  totalSchoolTime adamsWeek = 41.75 := by
  sorry


end NUMINAMATH_CALUDE_adams_total_school_time_l491_49118


namespace NUMINAMATH_CALUDE_antons_number_l491_49128

def matches_one_digit (n m : ℕ) : Prop :=
  (n / 100 = m / 100 ∧ n / 10 % 10 ≠ m / 10 % 10 ∧ n % 10 ≠ m % 10) ∨
  (n / 100 ≠ m / 100 ∧ n / 10 % 10 = m / 10 % 10 ∧ n % 10 ≠ m % 10) ∨
  (n / 100 ≠ m / 100 ∧ n / 10 % 10 ≠ m / 10 % 10 ∧ n % 10 = m % 10)

theorem antons_number :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧
    matches_one_digit n 109 ∧
    matches_one_digit n 704 ∧
    matches_one_digit n 124 ∧
    n = 729 :=
by
  sorry

end NUMINAMATH_CALUDE_antons_number_l491_49128


namespace NUMINAMATH_CALUDE_sea_glass_collection_l491_49144

/-- Sea glass collection problem -/
theorem sea_glass_collection (blanche_green blanche_red rose_red rose_blue : ℕ) 
  (h1 : blanche_green = 12)
  (h2 : blanche_red = 3)
  (h3 : rose_red = 9)
  (h4 : rose_blue = 11)
  : 2 * (blanche_red + rose_red) + 3 * rose_blue = 57 := by
  sorry

end NUMINAMATH_CALUDE_sea_glass_collection_l491_49144


namespace NUMINAMATH_CALUDE_set_membership_implies_m_values_l491_49169

theorem set_membership_implies_m_values (m : ℝ) :
  let A : Set ℝ := {1, m + 2, m^2 + 4}
  5 ∈ A → m = 3 ∨ m = 1 := by
sorry

end NUMINAMATH_CALUDE_set_membership_implies_m_values_l491_49169


namespace NUMINAMATH_CALUDE_proposition_variants_l491_49173

theorem proposition_variants (a b : ℝ) : 
  (∀ a b, a ≤ b → a - 2 ≤ b - 2) ∧ 
  (∀ a b, a - 2 > b - 2 → a > b) ∧ 
  (∀ a b, a - 2 ≤ b - 2 → a ≤ b) ∧ 
  ¬(∀ a b, a > b → a - 2 ≤ b - 2) := by
  sorry

end NUMINAMATH_CALUDE_proposition_variants_l491_49173


namespace NUMINAMATH_CALUDE_seventh_oblong_number_l491_49109

/-- Definition of an oblong number -/
def oblong_number (n : ℕ) : ℕ := n * (n + 1)

/-- Theorem: The 7th oblong number is 56 -/
theorem seventh_oblong_number : oblong_number 7 = 56 := by
  sorry

end NUMINAMATH_CALUDE_seventh_oblong_number_l491_49109


namespace NUMINAMATH_CALUDE_speed_increase_proof_l491_49183

/-- Proves that given a total distance of 210 km, a forward journey time of 7 hours,
    and a return journey time of 5 hours, the increase in speed during the return journey is 12 km/hr. -/
theorem speed_increase_proof (distance : ℝ) (forward_time : ℝ) (return_time : ℝ) 
    (h1 : distance = 210)
    (h2 : forward_time = 7)
    (h3 : return_time = 5) :
    (distance / return_time) - (distance / forward_time) = 12 := by
  sorry

end NUMINAMATH_CALUDE_speed_increase_proof_l491_49183


namespace NUMINAMATH_CALUDE_amy_total_tickets_l491_49166

/-- Amy's initial number of tickets -/
def initial_tickets : ℕ := 33

/-- Number of tickets Amy bought additionally -/
def additional_tickets : ℕ := 21

/-- Theorem stating the total number of tickets Amy has -/
theorem amy_total_tickets : initial_tickets + additional_tickets = 54 := by
  sorry

end NUMINAMATH_CALUDE_amy_total_tickets_l491_49166


namespace NUMINAMATH_CALUDE_prime_sum_squares_l491_49108

theorem prime_sum_squares (p q m : ℕ) : 
  p.Prime → q.Prime → p ≠ q →
  p^2 - 2001*p + m = 0 →
  q^2 - 2001*q + m = 0 →
  p^2 + q^2 = 3996005 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_squares_l491_49108


namespace NUMINAMATH_CALUDE_factorization_cubic_minus_quadratic_l491_49110

theorem factorization_cubic_minus_quadratic (x y : ℝ) :
  y^3 - 4*x^2*y = y*(y+2*x)*(y-2*x) := by sorry

end NUMINAMATH_CALUDE_factorization_cubic_minus_quadratic_l491_49110


namespace NUMINAMATH_CALUDE_river_road_cars_l491_49168

theorem river_road_cars (buses cars : ℕ) : 
  (buses : ℚ) / cars = 1 / 13 →
  buses = cars - 60 →
  cars = 65 := by
sorry

end NUMINAMATH_CALUDE_river_road_cars_l491_49168


namespace NUMINAMATH_CALUDE_cubic_resonance_intervals_sqrt_resonance_interval_l491_49105

-- Definition of a resonance interval
def is_resonance_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  a ≤ b ∧
  Monotone f ∧
  (∀ x ∈ Set.Icc a b, f x ∈ Set.Icc a b) ∧
  (∀ y ∈ Set.Icc a b, ∃ x ∈ Set.Icc a b, f x = y)

-- Theorem for the cubic function
theorem cubic_resonance_intervals :
  (is_resonance_interval (fun x ↦ x^3) (-1) 0) ∧
  (is_resonance_interval (fun x ↦ x^3) (-1) 1) ∧
  (is_resonance_interval (fun x ↦ x^3) 0 1) :=
sorry

-- Theorem for the square root function
theorem sqrt_resonance_interval (k : ℝ) :
  (∃ a b, is_resonance_interval (fun x ↦ Real.sqrt (x + 1) - k) a b) ↔
  (1 ≤ k ∧ k < 5/4) :=
sorry

end NUMINAMATH_CALUDE_cubic_resonance_intervals_sqrt_resonance_interval_l491_49105


namespace NUMINAMATH_CALUDE_wall_width_l491_49158

theorem wall_width (w h l : ℝ) (volume : ℝ) : 
  h = 4 * w →
  l = 3 * h →
  volume = w * h * l →
  volume = 10368 →
  w = 6 := by
sorry

end NUMINAMATH_CALUDE_wall_width_l491_49158


namespace NUMINAMATH_CALUDE_line_through_points_l491_49171

/-- The equation of a line passing through two points (x₁, y₁) and (x₂, y₂) -/
def line_equation (x₁ y₁ x₂ y₂ : ℝ) (x y : ℝ) : Prop :=
  (y - y₁) * (x₂ - x₁) = (x - x₁) * (y₂ - y₁)

/-- Theorem: The equation of the line passing through (1, 0) and (0, 1) is x + y - 1 = 0 -/
theorem line_through_points : 
  ∀ x y : ℝ, line_equation 1 0 0 1 x y ↔ x + y - 1 = 0 := by
  sorry

#check line_through_points

end NUMINAMATH_CALUDE_line_through_points_l491_49171


namespace NUMINAMATH_CALUDE_notebook_purchase_cost_l491_49178

def pen_cost : ℝ := 1.50
def notebook_cost : ℝ := 3 * pen_cost
def number_of_notebooks : ℕ := 4

theorem notebook_purchase_cost : 
  number_of_notebooks * notebook_cost = 18 := by
  sorry

end NUMINAMATH_CALUDE_notebook_purchase_cost_l491_49178


namespace NUMINAMATH_CALUDE_no_valid_A_l491_49140

theorem no_valid_A : ¬∃ (A : ℕ), A ≤ 9 ∧ 45 % A = 0 ∧ (456204 + A * 10) % 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_valid_A_l491_49140


namespace NUMINAMATH_CALUDE_days_worked_l491_49138

/-- Proves that given the conditions of the problem, the number of days worked is 23 -/
theorem days_worked (total_days : ℕ) (daily_wage : ℕ) (daily_forfeit : ℕ) (net_earnings : ℕ) 
  (h1 : total_days = 25)
  (h2 : daily_wage = 20)
  (h3 : daily_forfeit = 5)
  (h4 : net_earnings = 450) :
  ∃ (worked_days : ℕ), 
    worked_days * daily_wage - (total_days - worked_days) * daily_forfeit = net_earnings ∧ 
    worked_days = 23 := by
  sorry

#check days_worked

end NUMINAMATH_CALUDE_days_worked_l491_49138


namespace NUMINAMATH_CALUDE_crazy_silly_school_books_l491_49127

/-- The number of books yet to be read in the 'crazy silly school' series. -/
def books_to_read (total_books : ℕ) (books_read : ℕ) : ℕ :=
  total_books - books_read

/-- Theorem stating that given 20 total books and 15 books read, there are 5 books left to read. -/
theorem crazy_silly_school_books : books_to_read 20 15 = 5 := by
  sorry

end NUMINAMATH_CALUDE_crazy_silly_school_books_l491_49127


namespace NUMINAMATH_CALUDE_modulo_nine_sum_product_l491_49149

theorem modulo_nine_sum_product : 
  (2 * (1 + 222 + 3333 + 44444 + 555555 + 6666666 + 77777777 + 888888888)) % 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_modulo_nine_sum_product_l491_49149


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l491_49174

/-- Given that i² = -1, prove that (2-i)/(1+4i) = -2/17 - 9/17*i -/
theorem complex_fraction_simplification (i : ℂ) (h : i^2 = -1) :
  (2 - i) / (1 + 4*i) = -2/17 - 9/17*i := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l491_49174


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l491_49112

/-- A quadratic trinomial x^2 + mx + 1 is a perfect square if and only if m = ±2 -/
theorem perfect_square_trinomial (m : ℝ) :
  (∀ x, ∃ a, x^2 + m*x + 1 = (x + a)^2) ↔ (m = 2 ∨ m = -2) :=
sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l491_49112


namespace NUMINAMATH_CALUDE_manolo_face_mask_production_l491_49172

/-- Represents the face-mask production scenario for Manolo -/
structure FaceMaskProduction where
  initial_rate : ℕ  -- Rate of production in the first hour (minutes per mask)
  total_masks : ℕ   -- Total masks produced in a 4-hour shift
  shift_duration : ℕ -- Total duration of the shift in hours

/-- Calculates the time required to make one face-mask after the first hour -/
def time_per_mask_after_first_hour (p : FaceMaskProduction) : ℕ :=
  let masks_in_first_hour := 60 / p.initial_rate
  let remaining_masks := p.total_masks - masks_in_first_hour
  let remaining_time := (p.shift_duration - 1) * 60
  remaining_time / remaining_masks

/-- Theorem stating that given the initial conditions, the time per mask after the first hour is 6 minutes -/
theorem manolo_face_mask_production :
  ∀ (p : FaceMaskProduction),
    p.initial_rate = 4 ∧
    p.total_masks = 45 ∧
    p.shift_duration = 4 →
    time_per_mask_after_first_hour p = 6 := by
  sorry

end NUMINAMATH_CALUDE_manolo_face_mask_production_l491_49172


namespace NUMINAMATH_CALUDE_line_circle_intersection_slope_range_l491_49148

/-- Given a line passing through (4,0) and intersecting the circle (x-2)^2 + y^2 = 1,
    prove that its slope k is between -√3/3 and √3/3 inclusive. -/
theorem line_circle_intersection_slope_range :
  ∀ (k : ℝ),
  (∃ (x y : ℝ), y = k * (x - 4) ∧ (x - 2)^2 + y^2 = 1) →
  -Real.sqrt 3 / 3 ≤ k ∧ k ≤ Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_line_circle_intersection_slope_range_l491_49148


namespace NUMINAMATH_CALUDE_nail_sizes_sum_l491_49185

theorem nail_sizes_sum (size_2d : ℚ) (size_4d : ℚ) (size_6d : ℚ) (size_8d : ℚ) 
  (h1 : size_2d = 1/5)
  (h2 : size_4d = 3/10)
  (h3 : size_6d = 1/4)
  (h4 : size_8d = 1/8) :
  size_2d + size_4d = 1/2 := by
sorry

end NUMINAMATH_CALUDE_nail_sizes_sum_l491_49185


namespace NUMINAMATH_CALUDE_circle_equation_proof_l491_49163

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := x/4 + y/2 = 1

/-- Point A is where the line intersects the x-axis -/
def point_A : ℝ × ℝ := (4, 0)

/-- Point B is where the line intersects the y-axis -/
def point_B : ℝ × ℝ := (0, 2)

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y = 0

/-- Theorem: The equation of the circle with diameter AB is x^2 + y^2 - 4x - 2y = 0 -/
theorem circle_equation_proof :
  ∀ x y : ℝ, line_equation x y →
  (∃ t : ℝ, x = t * (point_B.1 - point_A.1) + point_A.1 ∧
            y = t * (point_B.2 - point_A.2) + point_A.2) →
  circle_equation x y :=
sorry

end NUMINAMATH_CALUDE_circle_equation_proof_l491_49163


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l491_49154

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x + 2 ≤ Real.sin x) ↔ (∃ x : ℝ, x^2 - 2*x + 2 > Real.sin x) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l491_49154


namespace NUMINAMATH_CALUDE_linear_equation_condition_l491_49162

theorem linear_equation_condition (a : ℝ) : 
  (∀ x y : ℝ, ∃ k m n : ℝ, (a - 2) * x^(|a| - 1) + 3 * y = k * x + m * y + n) → 
  a = -2 :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_condition_l491_49162


namespace NUMINAMATH_CALUDE_annie_ride_distance_l491_49117

/-- Taxi fare calculation --/
def taxi_fare (start_fee : ℚ) (toll : ℚ) (per_mile : ℚ) (miles : ℚ) : ℚ :=
  start_fee + toll + per_mile * miles

theorem annie_ride_distance :
  let mike_start_fee : ℚ := 25/10
  let annie_start_fee : ℚ := 25/10
  let mike_toll : ℚ := 0
  let annie_toll : ℚ := 5
  let per_mile : ℚ := 1/4
  let mike_miles : ℚ := 34
  let annie_miles : ℚ := 14

  taxi_fare mike_start_fee mike_toll per_mile mike_miles =
  taxi_fare annie_start_fee annie_toll per_mile annie_miles :=
by
  sorry


end NUMINAMATH_CALUDE_annie_ride_distance_l491_49117


namespace NUMINAMATH_CALUDE_three_digit_numbers_with_one_or_six_l491_49156

/-- The number of three-digit whole numbers -/
def total_three_digit_numbers : ℕ := 900

/-- The number of choices for the first digit (excluding 1 and 6) -/
def first_digit_choices : ℕ := 7

/-- The number of choices for the second and third digits (excluding 1 and 6) -/
def other_digit_choices : ℕ := 8

/-- The number of three-digit numbers without 1 or 6 -/
def numbers_without_one_or_six : ℕ := first_digit_choices * other_digit_choices * other_digit_choices

theorem three_digit_numbers_with_one_or_six : 
  total_three_digit_numbers - numbers_without_one_or_six = 452 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_numbers_with_one_or_six_l491_49156


namespace NUMINAMATH_CALUDE_factorization_equality_l491_49133

theorem factorization_equality (a x y : ℝ) : a * x^2 + 2 * a * x * y + a * y^2 = a * (x + y)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l491_49133


namespace NUMINAMATH_CALUDE_chord_slope_l491_49167

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 20 + y^2 / 16 = 1

-- Define the point P
def P : ℝ × ℝ := (3, -2)

-- Define the line l
def line_l (m : ℝ) (x y : ℝ) : Prop := y - P.2 = m * (x - P.1)

-- Define the midpoint property
def is_midpoint (P A B : ℝ × ℝ) : Prop :=
  P.1 = (A.1 + B.1) / 2 ∧ P.2 = (A.2 + B.2) / 2

-- Theorem statement
theorem chord_slope :
  ∃ (A B : ℝ × ℝ) (m : ℝ),
    ellipse A.1 A.2 ∧
    ellipse B.1 B.2 ∧
    line_l m A.1 A.2 ∧
    line_l m B.1 B.2 ∧
    is_midpoint P A B ∧
    m = 6/5 := by sorry

end NUMINAMATH_CALUDE_chord_slope_l491_49167


namespace NUMINAMATH_CALUDE_smallest_among_given_numbers_l491_49188

theorem smallest_among_given_numbers :
  ∀ (a b c d : ℝ), a = -1 ∧ b = 0 ∧ c = -Real.sqrt 2 ∧ d = 2 →
  c ≤ a ∧ c ≤ b ∧ c ≤ d :=
by sorry

end NUMINAMATH_CALUDE_smallest_among_given_numbers_l491_49188


namespace NUMINAMATH_CALUDE_min_value_expression_l491_49175

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + 1/b) * (b + 4/a) ≥ 9 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ (a₀ + 1/b₀) * (b₀ + 4/a₀) = 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l491_49175


namespace NUMINAMATH_CALUDE_carries_trip_l491_49190

theorem carries_trip (day1 : ℕ) (day3 : ℕ) (day4 : ℕ) (charge_distance : ℕ) (charge_count : ℕ) : 
  day1 = 135 → 
  day3 = 159 → 
  day4 = 189 → 
  charge_distance = 106 → 
  charge_count = 7 → 
  ∃ day2 : ℕ, day2 - day1 = 124 ∧ day1 + day2 + day3 + day4 = charge_distance * charge_count :=
by sorry

end NUMINAMATH_CALUDE_carries_trip_l491_49190


namespace NUMINAMATH_CALUDE_exam_score_deviation_l491_49131

/-- Given an exam with mean score 74 and standard deviation σ, 
    prove that 58 is 2 standard deviations below the mean. -/
theorem exam_score_deviation :
  ∀ σ : ℝ,
  74 + 3 * σ = 98 →
  74 - 2 * σ = 58 :=
by
  sorry

end NUMINAMATH_CALUDE_exam_score_deviation_l491_49131


namespace NUMINAMATH_CALUDE_oil_truck_tank_radius_l491_49126

/-- Represents a right circular cylinder -/
structure RightCircularCylinder where
  radius : ℝ
  height : ℝ

/-- The problem statement -/
theorem oil_truck_tank_radius 
  (stationary_tank : RightCircularCylinder)
  (oil_truck_tank : RightCircularCylinder)
  (oil_level_drop : ℝ)
  (h_stationary_radius : stationary_tank.radius = 100)
  (h_stationary_height : stationary_tank.height = 25)
  (h_truck_height : oil_truck_tank.height = 10)
  (h_oil_drop : oil_level_drop = 0.025)
  (h_volume_equality : π * stationary_tank.radius^2 * oil_level_drop = 
                       π * oil_truck_tank.radius^2 * oil_truck_tank.height) :
  oil_truck_tank.radius = 5 := by
  sorry

#check oil_truck_tank_radius

end NUMINAMATH_CALUDE_oil_truck_tank_radius_l491_49126


namespace NUMINAMATH_CALUDE_distance_negative_five_to_negative_fourteen_l491_49103

/-- The distance between two points on a number line -/
def numberLineDistance (a b : ℝ) : ℝ := |a - b|

/-- Theorem: The distance between -5 and -14 on a number line is 9 -/
theorem distance_negative_five_to_negative_fourteen :
  numberLineDistance (-5) (-14) = 9 := by
  sorry

end NUMINAMATH_CALUDE_distance_negative_five_to_negative_fourteen_l491_49103


namespace NUMINAMATH_CALUDE_root_implies_m_value_l491_49198

theorem root_implies_m_value (x m : ℝ) : 
  x = 2 → x^2 - m*x + 6 = 0 → m = 5 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_m_value_l491_49198


namespace NUMINAMATH_CALUDE_hotel_expenditure_l491_49136

theorem hotel_expenditure (num_men : ℕ) (standard_cost : ℚ) (extra_cost : ℚ) :
  num_men = 9 →
  standard_cost = 3 →
  extra_cost = 2 →
  (((num_men - 1) * standard_cost + 
    (standard_cost + extra_cost + 
      ((num_men - 1) * standard_cost + (standard_cost + extra_cost)) / num_men)) = 29.25) := by
  sorry

end NUMINAMATH_CALUDE_hotel_expenditure_l491_49136


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l491_49193

theorem unique_solution_quadratic (k : ℚ) : 
  (∃! x : ℝ, (x + 6) * (x + 2) = k + 3 * x) ↔ k = 23 / 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l491_49193


namespace NUMINAMATH_CALUDE_negation_equivalence_l491_49106

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x^2 + x + 1 > 0) ↔ (∃ x : ℝ, x^2 + x + 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l491_49106


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_given_l491_49107

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in the form y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

def Line.passes_through (l : Line) (p : Point) : Prop :=
  p.y = l.m * p.x + l.b

def Line.parallel_to (l1 l2 : Line) : Prop :=
  l1.m = l2.m

theorem line_through_point_parallel_to_given : 
  let P : Point := ⟨1, 2⟩
  let given_line : Line := ⟨2, 3⟩
  let parallel_line : Line := ⟨2, 0⟩
  parallel_line.passes_through P ∧ parallel_line.parallel_to given_line := by
  sorry


end NUMINAMATH_CALUDE_line_through_point_parallel_to_given_l491_49107


namespace NUMINAMATH_CALUDE_largest_subsequence_number_l491_49151

def original_number : ℕ := 778157260669103

def is_subsequence (sub seq : List ℕ) : Prop :=
  ∃ (l1 l2 : List ℕ), seq = l1 ++ sub ++ l2

def digits_to_nat (digits : List ℕ) : ℕ :=
  digits.foldl (fun acc d => 10 * acc + d) 0

def nat_to_digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 10) ((m % 10) :: acc)
    aux n []

theorem largest_subsequence_number :
  let orig_digits := nat_to_digits original_number
  let result_digits := nat_to_digits 879103
  (result_digits.length = 6) ∧
  (is_subsequence result_digits orig_digits) ∧
  (∀ (other : List ℕ), other.length = 6 →
    is_subsequence other orig_digits →
    digits_to_nat other ≤ digits_to_nat result_digits) :=
by sorry

end NUMINAMATH_CALUDE_largest_subsequence_number_l491_49151


namespace NUMINAMATH_CALUDE_unique_a_for_system_solution_l491_49100

/-- The system of equations --/
def system (a b x y : ℝ) : Prop :=
  2^(b*x) + (a+1)*b*y^2 = a^2 ∧ (a-1)*x^3 + y^3 = 1

/-- The theorem statement --/
theorem unique_a_for_system_solution :
  ∃! a : ℝ, ∀ b : ℝ, ∃ x y : ℝ, system a b x y :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_a_for_system_solution_l491_49100


namespace NUMINAMATH_CALUDE_optimal_large_trucks_for_fruit_loading_l491_49152

/-- Represents the problem of loading fruits onto trucks -/
structure FruitLoading where
  total_fruits : ℕ
  large_truck_capacity : ℕ
  small_truck_capacity : ℕ

/-- Checks if a given number of large trucks is optimal for the fruit loading problem -/
def is_optimal_large_trucks (problem : FruitLoading) (num_large_trucks : ℕ) : Prop :=
  let remaining_fruits := problem.total_fruits - num_large_trucks * problem.large_truck_capacity
  -- The remaining fruits can be loaded onto small trucks without leftovers
  remaining_fruits % problem.small_truck_capacity = 0 ∧
  -- Using one more large truck would exceed the total fruits
  (num_large_trucks + 1) * problem.large_truck_capacity > problem.total_fruits

/-- Theorem stating that 8 large trucks is the optimal solution for the given problem -/
theorem optimal_large_trucks_for_fruit_loading :
  let problem : FruitLoading := ⟨134, 15, 7⟩
  is_optimal_large_trucks problem 8 :=
by sorry

end NUMINAMATH_CALUDE_optimal_large_trucks_for_fruit_loading_l491_49152


namespace NUMINAMATH_CALUDE_logarithm_expression_equals_two_plus_pi_l491_49147

-- Define the common logarithm (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem logarithm_expression_equals_two_plus_pi :
  lg 5 * (Real.log 20 / Real.log (Real.sqrt 10)) + (lg (2 ^ Real.sqrt 2))^2 + Real.exp (Real.log π) = 2 + π := by
  sorry

end NUMINAMATH_CALUDE_logarithm_expression_equals_two_plus_pi_l491_49147
