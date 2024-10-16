import Mathlib

namespace NUMINAMATH_CALUDE_union_of_P_and_Q_l2352_235240

-- Define the sets P and Q
def P : Set ℝ := {x | -1 < x ∧ x < 1}
def Q : Set ℝ := {x | 0 < x ∧ x < 2}

-- State the theorem
theorem union_of_P_and_Q : P ∪ Q = {x : ℝ | -1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_union_of_P_and_Q_l2352_235240


namespace NUMINAMATH_CALUDE_octahedron_flattenable_l2352_235220

/-- Represents a cube -/
structure Cube where
  vertices : Finset (Fin 8)
  edges : Finset (Fin 12)
  faces : Finset (Fin 6)

/-- Represents an octahedron -/
structure Octahedron where
  vertices : Finset (Fin 6)
  edges : Finset (Fin 12)
  faces : Finset (Fin 8)

/-- Defines the relationship between a cube and its corresponding octahedron -/
def correspondingOctahedron (c : Cube) : Octahedron :=
  sorry

/-- Defines what it means for a set of faces to be connected -/
def isConnected {α : Type*} (s : Finset α) : Prop :=
  sorry

/-- Defines what it means for a set of faces to be flattenable -/
def isFlattenable {α : Type*} (s : Finset α) : Prop :=
  sorry

/-- Defines the operation of cutting edges on a polyhedron -/
def cutEdges {α : Type*} (edges : Finset α) (toCut : Finset α) : Finset α :=
  sorry

theorem octahedron_flattenable (c : Cube) (cubeCuts : Finset (Fin 12)) :
  (cubeCuts.card = 7) →
  (isConnected (cutEdges c.edges cubeCuts)) →
  (isFlattenable (cutEdges c.edges cubeCuts)) →
  let o := correspondingOctahedron c
  let octaCuts := c.edges \ cubeCuts
  (isConnected (cutEdges o.edges octaCuts)) ∧
  (isFlattenable (cutEdges o.edges octaCuts)) := by
  sorry

end NUMINAMATH_CALUDE_octahedron_flattenable_l2352_235220


namespace NUMINAMATH_CALUDE_dvd_rental_cost_l2352_235237

theorem dvd_rental_cost (total_cost : ℝ) (num_dvds : ℕ) (cost_per_dvd : ℝ) 
  (h1 : total_cost = 4.8)
  (h2 : num_dvds = 4)
  (h3 : cost_per_dvd = total_cost / num_dvds) :
  cost_per_dvd = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_dvd_rental_cost_l2352_235237


namespace NUMINAMATH_CALUDE_max_value_implies_tan_2alpha_l2352_235203

/-- Given a function f(x) = 3sin(x) + cos(x) that attains its maximum value when x = α,
    prove that tan(2α) = -3/4 -/
theorem max_value_implies_tan_2alpha (α : ℝ) 
  (h : ∀ x, 3 * Real.sin x + Real.cos x ≤ 3 * Real.sin α + Real.cos α) : 
  Real.tan (2 * α) = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_implies_tan_2alpha_l2352_235203


namespace NUMINAMATH_CALUDE_coordinates_of_point_b_l2352_235270

/-- Given a line segment AB with length 3, parallel to the y-axis, and point A at coordinates (-1, 2),
    the coordinates of point B must be either (-1, 5) or (-1, -1). -/
theorem coordinates_of_point_b (A B : ℝ × ℝ) : 
  A = (-1, 2) → 
  (B.1 - A.1 = 0) →  -- AB is parallel to y-axis
  ((B.1 - A.1)^2 + (B.2 - A.2)^2 = 3^2) →  -- AB length is 3
  (B = (-1, 5) ∨ B = (-1, -1)) := by
sorry

end NUMINAMATH_CALUDE_coordinates_of_point_b_l2352_235270


namespace NUMINAMATH_CALUDE_football_season_games_l2352_235291

/-- The number of months in the football season -/
def season_months : ℕ := 17

/-- The number of games played each month -/
def games_per_month : ℕ := 19

/-- The total number of games played during the season -/
def total_games : ℕ := season_months * games_per_month

theorem football_season_games :
  total_games = 323 :=
by sorry

end NUMINAMATH_CALUDE_football_season_games_l2352_235291


namespace NUMINAMATH_CALUDE_special_numbers_theorem_l2352_235278

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def has_distinct_digits (n : ℕ) : Prop :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3

def replace_greatest_with_one (n : ℕ) : ℕ :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  let max_digit := max d1 (max d2 d3)
  if d1 = max_digit then
    100 + d2 * 10 + d3
  else if d2 = max_digit then
    d1 * 100 + 10 + d3
  else
    d1 * 100 + d2 * 10 + 1

theorem special_numbers_theorem :
  {n : ℕ | is_three_digit n ∧ 
           has_distinct_digits n ∧ 
           (replace_greatest_with_one n) % 30 = 0} =
  {230, 320, 560, 650, 890, 980} := by sorry

end NUMINAMATH_CALUDE_special_numbers_theorem_l2352_235278


namespace NUMINAMATH_CALUDE_alice_wins_iff_m_even_or_n_odd_l2352_235244

/-- The game state on an n×n grid where players can color an m×m subgrid or a single cell -/
structure GameState (m n : ℕ+) where
  grid : Fin n → Fin n → Bool

/-- The result of the game -/
inductive GameResult
  | AliceWins
  | BobWins

/-- An optimal strategy for the game -/
def OptimalStrategy (m n : ℕ+) : GameState m n → GameResult := sorry

/-- The main theorem: Alice wins with optimal play if and only if m is even or n is odd -/
theorem alice_wins_iff_m_even_or_n_odd (m n : ℕ+) :
  (∀ initial : GameState m n, OptimalStrategy m n initial = GameResult.AliceWins) ↔ 
  (Even m.val ∨ Odd n.val) := by sorry

end NUMINAMATH_CALUDE_alice_wins_iff_m_even_or_n_odd_l2352_235244


namespace NUMINAMATH_CALUDE_average_age_decrease_l2352_235236

theorem average_age_decrease (initial_avg : ℝ) (new_students : ℕ) (new_avg : ℝ) (total_students : ℕ) : 
  initial_avg = 48 →
  new_students = 120 →
  new_avg = 32 →
  total_students = 160 →
  let original_students := total_students - new_students
  let total_age := initial_avg * original_students + new_avg * new_students
  let new_avg_age := total_age / total_students
  initial_avg - new_avg_age = 12 := by
sorry

end NUMINAMATH_CALUDE_average_age_decrease_l2352_235236


namespace NUMINAMATH_CALUDE_product_terminal_zeros_l2352_235285

/-- The number of terminal zeros in a natural number -/
def terminalZeros (n : ℕ) : ℕ := sorry

/-- The product of 50, 480, and 7 -/
def product : ℕ := 50 * 480 * 7

theorem product_terminal_zeros : terminalZeros product = 3 := by sorry

end NUMINAMATH_CALUDE_product_terminal_zeros_l2352_235285


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2352_235269

theorem inequality_equivalence (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (3 : ℝ) ^ (1 / (x + 1/x)) > (3 : ℝ) ^ (1 / (y + 1/y)) ↔ 
  (x > 0 ∧ y < 0) ∨ 
  (x > y ∧ y > 0 ∧ x * y > 1) ∨ 
  (x < y ∧ y < 0 ∧ 0 < x * y ∧ x * y < 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2352_235269


namespace NUMINAMATH_CALUDE_simplify_sqrt_x_squared_y_second_quadrant_l2352_235213

theorem simplify_sqrt_x_squared_y_second_quadrant (x y : ℝ) (h1 : x < 0) (h2 : y > 0) :
  Real.sqrt (x^2 * y) = -x * Real.sqrt y := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_x_squared_y_second_quadrant_l2352_235213


namespace NUMINAMATH_CALUDE_sqrt_x_plus_2_equals_2_implies_x_plus_2_squared_equals_16_l2352_235234

theorem sqrt_x_plus_2_equals_2_implies_x_plus_2_squared_equals_16 (x : ℝ) :
  Real.sqrt (x + 2) = 2 → (x + 2)^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_2_equals_2_implies_x_plus_2_squared_equals_16_l2352_235234


namespace NUMINAMATH_CALUDE_no_solution_for_functional_equation_l2352_235217

theorem no_solution_for_functional_equation :
  ¬∃ (f : ℕ → ℕ), ∀ (x : ℕ), f (f x) = x + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_functional_equation_l2352_235217


namespace NUMINAMATH_CALUDE_constant_difference_expressions_l2352_235230

theorem constant_difference_expressions (x : ℤ) : 
  (∃ k : ℤ, (x^2 - 4*x + 5) - (2*x - 6) = k ∧ 
             (4*x - 8) - (x^2 - 4*x + 5) = k ∧ 
             (3*x^2 - 12*x + 11) - (4*x - 8) = k) ↔ 
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_constant_difference_expressions_l2352_235230


namespace NUMINAMATH_CALUDE_function_characterization_l2352_235261

def DivisibilityCondition (f : ℕ+ → ℕ+) : Prop :=
  ∀ (a b : ℕ+), a + b > 2019 → (a + f b) ∣ (a^2 + b * f a)

theorem function_characterization (f : ℕ+ → ℕ+) 
  (h : DivisibilityCondition f) : 
  ∃ (r : ℕ+), ∀ (x : ℕ+), f x = r * x := by
  sorry

end NUMINAMATH_CALUDE_function_characterization_l2352_235261


namespace NUMINAMATH_CALUDE_no_real_roots_l2352_235271

theorem no_real_roots (A B : ℝ) : 
  (∀ x y : ℝ, x^2 + x*y + y = A ∧ y / (y - x) = B → False) ↔ A = 2 ∧ B = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_no_real_roots_l2352_235271


namespace NUMINAMATH_CALUDE_sequence_min_value_and_ratio_l2352_235289

/-- Given a positive integer m ≥ 3, an arithmetic sequence {a_n} with positive terms,
    and a geometric sequence {b_n} with positive terms, such that:
    1. The first term of {a_n} equals the common ratio of {b_n}
    2. The first term of {b_n} equals the common difference of {a_n}
    3. a_m = b_m
    This theorem proves the minimum value of a_m and the ratio of a_1 to b_1 when a_m is minimum. -/
theorem sequence_min_value_and_ratio (m : ℕ) (a b : ℝ → ℝ) (h_m : m ≥ 3) 
  (h_a_pos : ∀ n, a n > 0) (h_b_pos : ∀ n, b n > 0)
  (h_a_arith : ∀ n, a (n + 1) - a n = a 2 - a 1)
  (h_b_geom : ∀ n, b (n + 1) / b n = b 2 / b 1)
  (h_first_term : a 1 = b 2 / b 1)
  (h_common_diff : b 1 = a 2 - a 1)
  (h_m_equal : a m = b m) :
  ∃ (min_am : ℝ) (ratio : ℝ),
    min_am = ((m^m : ℝ) / ((m - 1 : ℝ)^(m - 2)))^(1 / (m - 1 : ℝ)) ∧
    ratio = (m - 1 : ℝ)^2 ∧
    a m ≥ min_am ∧
    (a m = min_am → a 1 / b 1 = ratio) := by
  sorry

end NUMINAMATH_CALUDE_sequence_min_value_and_ratio_l2352_235289


namespace NUMINAMATH_CALUDE_gizi_does_not_catch_up_l2352_235263

/-- Represents the work progress of Kató and Gizi -/
structure WorkProgress where
  kato_lines : ℕ
  gizi_lines : ℕ

/-- Represents the copying rates and page capacities -/
structure CopyingParameters where
  kato_lines_per_page : ℕ
  gizi_lines_per_page : ℕ
  kato_rate : ℕ
  gizi_rate : ℕ

def initial_state : WorkProgress :=
  { kato_lines := 80,  -- 4 pages * 20 lines per page
    gizi_lines := 0 }

def copying_params : CopyingParameters :=
  { kato_lines_per_page := 20,
    gizi_lines_per_page := 30,
    kato_rate := 3,
    gizi_rate := 4 }

def setup_time_progress (wp : WorkProgress) : WorkProgress :=
  { kato_lines := wp.kato_lines + 3,  -- 2.5 rounded up to 3
    gizi_lines := wp.gizi_lines }

def update_progress (wp : WorkProgress) (cp : CopyingParameters) : WorkProgress :=
  { kato_lines := wp.kato_lines + cp.kato_rate,
    gizi_lines := wp.gizi_lines + cp.gizi_rate }

def gizi_catches_up (wp : WorkProgress) : Prop :=
  wp.gizi_lines * 4 ≥ wp.kato_lines * 3

theorem gizi_does_not_catch_up :
  ¬∃ n : ℕ, gizi_catches_up (n.iterate (update_progress · copying_params) (setup_time_progress initial_state)) ∧
            (n.iterate (update_progress · copying_params) (setup_time_progress initial_state)).gizi_lines ≤ 150 :=
sorry

end NUMINAMATH_CALUDE_gizi_does_not_catch_up_l2352_235263


namespace NUMINAMATH_CALUDE_integer_equation_solution_l2352_235294

theorem integer_equation_solution (x y z : ℤ) :
  x^2 * (y - z) + y^2 * (z - x) + z^2 * (x - y) = 2 ↔ 
  ∃ k : ℤ, x = k + 1 ∧ y = k ∧ z = k - 1 :=
by sorry

end NUMINAMATH_CALUDE_integer_equation_solution_l2352_235294


namespace NUMINAMATH_CALUDE_simplify_expression_l2352_235226

theorem simplify_expression : 
  (18 * 10^9 - 6 * 10^9) / (6 * 10^4 + 3 * 10^4) = 400000 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2352_235226


namespace NUMINAMATH_CALUDE_cricket_team_size_l2352_235222

theorem cricket_team_size :
  ∀ (n : ℕ) (initial_avg final_avg : ℝ),
  initial_avg = 29 →
  final_avg = 26 →
  (n * final_avg = (n - 2) * (initial_avg - 1) + (initial_avg + 3) + initial_avg) →
  n = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_cricket_team_size_l2352_235222


namespace NUMINAMATH_CALUDE_ship_storm_problem_l2352_235247

/-- A problem about a ship's journey and a storm -/
theorem ship_storm_problem (initial_speed : ℝ) (initial_time : ℝ) 
  (h1 : initial_speed = 30)
  (h2 : initial_time = 20)
  (h3 : initial_speed * initial_time = (1/2) * (total_distance : ℝ))
  (h4 : distance_after_storm = (1/3) * total_distance) : 
  initial_speed * initial_time - distance_after_storm = 200 := by
  sorry

#check ship_storm_problem

end NUMINAMATH_CALUDE_ship_storm_problem_l2352_235247


namespace NUMINAMATH_CALUDE_journey_time_calculation_l2352_235238

/-- Proves that given a journey of 240 km completed in 5 hours, 
    where the first part is traveled at 40 kmph and the second part at 60 kmph, 
    the time spent on the first part of the journey is 3 hours. -/
theorem journey_time_calculation (total_distance : ℝ) (total_time : ℝ) 
    (speed_first_part : ℝ) (speed_second_part : ℝ) 
    (h1 : total_distance = 240)
    (h2 : total_time = 5)
    (h3 : speed_first_part = 40)
    (h4 : speed_second_part = 60) :
    ∃ (first_part_time : ℝ), 
      first_part_time * speed_first_part + 
      (total_time - first_part_time) * speed_second_part = total_distance ∧
      first_part_time = 3 :=
by sorry

end NUMINAMATH_CALUDE_journey_time_calculation_l2352_235238


namespace NUMINAMATH_CALUDE_fraction_problem_l2352_235251

theorem fraction_problem (p q : ℚ) : 
  q = 5 → 
  1/7 + (2*q - p)/(2*q + p) = 4/7 → 
  p = 4 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l2352_235251


namespace NUMINAMATH_CALUDE_summer_reading_challenge_l2352_235259

def books_to_coupons (books : ℕ) : ℕ := books / 5

def quinn_books : ℕ := 5 * 5

def taylor_books : ℕ := 1 + 4 * 9

def jordan_books : ℕ := 3 * 10

theorem summer_reading_challenge : 
  books_to_coupons quinn_books + books_to_coupons taylor_books + books_to_coupons jordan_books = 18 := by
  sorry

end NUMINAMATH_CALUDE_summer_reading_challenge_l2352_235259


namespace NUMINAMATH_CALUDE_lcm_product_hcf_l2352_235290

theorem lcm_product_hcf (a b : ℕ+) (h1 : Nat.lcm a b = 750) (h2 : a * b = 18750) :
  Nat.gcd a b = 25 := by
  sorry

end NUMINAMATH_CALUDE_lcm_product_hcf_l2352_235290


namespace NUMINAMATH_CALUDE_horror_movie_tickets_l2352_235218

theorem horror_movie_tickets (romance_tickets horror_tickets : ℕ) : 
  romance_tickets = 25 →
  horror_tickets = 3 * romance_tickets + 18 →
  horror_tickets = 93 := by
sorry

end NUMINAMATH_CALUDE_horror_movie_tickets_l2352_235218


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l2352_235284

/-- Given a line segment from (1, 3) to (x, -4) with length 15 and x > 0, prove x = 1 + √176 -/
theorem line_segment_endpoint (x : ℝ) : 
  x > 0 → 
  ((x - 1)^2 + (-4 - 3)^2).sqrt = 15 → 
  x = 1 + Real.sqrt 176 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l2352_235284


namespace NUMINAMATH_CALUDE_max_stamps_proof_l2352_235206

/-- The price of a stamp in cents -/
def stamp_price : ℕ := 50

/-- The total budget in cents -/
def total_budget : ℕ := 5000

/-- The number of stamps required for discount eligibility -/
def discount_threshold : ℕ := 80

/-- The discount amount per stamp in cents -/
def discount_amount : ℕ := 5

/-- The maximum number of stamps that can be purchased with the given conditions -/
def max_stamps : ℕ := 111

theorem max_stamps_proof :
  ∀ n : ℕ,
  n ≤ max_stamps ∧
  (n > discount_threshold → n * (stamp_price - discount_amount) ≤ total_budget) ∧
  (n ≤ discount_threshold → n * stamp_price ≤ total_budget) ∧
  (max_stamps > discount_threshold → max_stamps * (stamp_price - discount_amount) ≤ total_budget) ∧
  (max_stamps + 1 > discount_threshold → (max_stamps + 1) * (stamp_price - discount_amount) > total_budget) := by
  sorry

end NUMINAMATH_CALUDE_max_stamps_proof_l2352_235206


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l2352_235286

theorem quadratic_roots_relation (p A B : ℤ) : 
  (∃ α β : ℝ, α + 1 ≠ β + 1 ∧ 
    (∀ x : ℝ, x^2 + p*x + 19 = 0 ↔ x = α + 1 ∨ x = β + 1) ∧
    (∀ x : ℝ, x^2 - A*x + B = 0 ↔ x = α ∨ x = β)) →
  A + B = 18 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l2352_235286


namespace NUMINAMATH_CALUDE_triangle_inequality_l2352_235214

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : (a^2 + b^2 + c^2)^2 > 2*(a^4 + b^4 + c^4)) :
  a + b > c ∧ a + c > b ∧ b + c > a := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2352_235214


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2352_235215

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x^2 - 2*x
  ∃ x₁ x₂ : ℝ, x₁ = 0 ∧ x₂ = 2 ∧ (∀ x : ℝ, f x = 0 ↔ x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2352_235215


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l2352_235258

theorem algebraic_expression_value :
  ∀ x : ℝ, x = 2 * Real.sqrt 3 - 1 → x^2 + 2*x - 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l2352_235258


namespace NUMINAMATH_CALUDE_find_number_l2352_235295

theorem find_number : ∃! x : ℕ, 220080 = (x + 445) * (2 * (x - 445)) + 80 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l2352_235295


namespace NUMINAMATH_CALUDE_nine_zeros_in_binary_representation_l2352_235229

/-- The number of zeros in the binary representation of a natural number -/
def countZeros (n : ℕ) : ℕ := sorry

/-- An unknown non-negative integer -/
def someNumber : ℕ := sorry

/-- The main expression: 6 * 1024 + 4 * 64 + someNumber -/
def mainExpression : ℕ := 6 * 1024 + 4 * 64 + someNumber

theorem nine_zeros_in_binary_representation :
  countZeros mainExpression = 9 := by sorry

end NUMINAMATH_CALUDE_nine_zeros_in_binary_representation_l2352_235229


namespace NUMINAMATH_CALUDE_smallest_integer_y_l2352_235266

def is_integer (x : ℚ) : Prop := ∃ n : ℤ, x = n

def expression (y : ℤ) : ℚ := (y^2 - 3*y + 11) / (y - 5)

theorem smallest_integer_y : 
  (∀ y : ℤ, y < 6 → ¬(is_integer (expression y))) ∧ 
  (is_integer (expression 6)) := by sorry

end NUMINAMATH_CALUDE_smallest_integer_y_l2352_235266


namespace NUMINAMATH_CALUDE_complex_magnitude_equals_one_l2352_235221

theorem complex_magnitude_equals_one : ∀ (z : ℂ), z = (2 * Complex.I + 1) / (Complex.I - 2) → Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equals_one_l2352_235221


namespace NUMINAMATH_CALUDE_girl_boy_ratio_l2352_235212

/-- Represents the number of students in the class -/
def total_students : ℕ := 28

/-- Represents the difference between the number of girls and boys -/
def girl_boy_difference : ℕ := 4

/-- Theorem stating that the ratio of girls to boys is 4:3 -/
theorem girl_boy_ratio :
  ∃ (girls boys : ℕ),
    girls + boys = total_students ∧
    girls = boys + girl_boy_difference ∧
    girls * 3 = boys * 4 :=
by sorry

end NUMINAMATH_CALUDE_girl_boy_ratio_l2352_235212


namespace NUMINAMATH_CALUDE_shell_calculation_l2352_235287

theorem shell_calculation (initial : Real) (add1 : Real) (add2 : Real) (subtract : Real) (final : Real) :
  initial = 5.2 ∧ add1 = 15.7 ∧ add2 = 17.5 ∧ subtract = 4.3 ∧ final = 102.3 →
  final = 3 * ((initial + add1 + add2 - subtract)) :=
by sorry

end NUMINAMATH_CALUDE_shell_calculation_l2352_235287


namespace NUMINAMATH_CALUDE_complement_of_union_l2352_235232

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {2, 3, 5}
def N : Set ℕ := {4, 5}

theorem complement_of_union (U M N : Set ℕ) 
  (hU : U = {1, 2, 3, 4, 5, 6})
  (hM : M = {2, 3, 5})
  (hN : N = {4, 5}) :
  (M ∪ N)ᶜ = {1, 6} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_l2352_235232


namespace NUMINAMATH_CALUDE_sin_alpha_plus_pi_half_l2352_235201

-- Define the point P
def P : ℝ × ℝ := (2, 1)

-- Define the angle α
variable (α : ℝ)

-- State the theorem
theorem sin_alpha_plus_pi_half (h : ∃ (t : ℝ), t > 0 ∧ P = (t * Real.cos α, t * Real.sin α)) : 
  Real.sin (α + π/2) = 2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_plus_pi_half_l2352_235201


namespace NUMINAMATH_CALUDE_brads_running_speed_l2352_235246

/-- Proves that Brad's running speed is 6 km/h given the conditions of the problem -/
theorem brads_running_speed (maxwell_speed : ℝ) (total_distance : ℝ) (maxwell_time : ℝ) 
  (h1 : maxwell_speed = 4)
  (h2 : total_distance = 94)
  (h3 : maxwell_time = 10) : 
  let brad_time := maxwell_time - 1
  let maxwell_distance := maxwell_speed * maxwell_time
  let brad_distance := total_distance - maxwell_distance
  brad_distance / brad_time = 6 := by
  sorry

end NUMINAMATH_CALUDE_brads_running_speed_l2352_235246


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l2352_235233

theorem quadratic_root_relation (p q : ℤ) : 
  (∃ x y : ℝ, x^2 + p*x + q = 0 ∧ y^2 + p*y + q = 0 ∧ x = 4*y) ∧ 
  (abs p < 100) ∧ (abs q < 100) ↔ 
  ((p = 5 ∨ p = -5) ∧ q = 4) ∨
  ((p = 10 ∨ p = -10) ∧ q = 16) ∨
  ((p = 15 ∨ p = -15) ∧ q = 36) ∨
  ((p = 20 ∨ p = -20) ∧ q = 64) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l2352_235233


namespace NUMINAMATH_CALUDE_m_range_l2352_235275

-- Define the propositions p and q as functions of m
def p (m : ℝ) : Prop := 2 < m ∧ m < 4

def q (m : ℝ) : Prop := m < 0 ∨ m > 3

-- Define the range of m
def range_m (m : ℝ) : Prop := m < 0 ∨ (2 < m ∧ m < 3) ∨ m > 3

-- State the theorem
theorem m_range : 
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m) ↔ range_m m :=
by sorry

end NUMINAMATH_CALUDE_m_range_l2352_235275


namespace NUMINAMATH_CALUDE_square_of_sum_twice_x_plus_y_l2352_235265

theorem square_of_sum_twice_x_plus_y (x y : ℝ) : (2*x + y)^2 = (2*x + y)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_sum_twice_x_plus_y_l2352_235265


namespace NUMINAMATH_CALUDE_heartsuit_calculation_l2352_235207

def heartsuit (u v : ℝ) : ℝ := (u + 2*v) * (u - v)

theorem heartsuit_calculation : heartsuit 2 (heartsuit 3 4) = -260 := by
  sorry

end NUMINAMATH_CALUDE_heartsuit_calculation_l2352_235207


namespace NUMINAMATH_CALUDE_max_teams_tied_for_most_wins_l2352_235205

/-- Represents a round-robin tournament --/
structure Tournament where
  num_teams : Nat
  games : Fin num_teams → Fin num_teams → Bool
  
/-- Tournament conditions --/
def valid_tournament (t : Tournament) : Prop :=
  t.num_teams = 7 ∧
  (∀ i j, i ≠ j → (t.games i j ↔ ¬t.games j i)) ∧
  (∀ i, ¬t.games i i)

/-- Number of wins for a team --/
def wins (t : Tournament) (team : Fin t.num_teams) : Nat :=
  (Finset.univ.filter (λ j => t.games team j)).card

/-- Maximum number of wins in the tournament --/
def max_wins (t : Tournament) : Nat :=
  Finset.univ.sup (λ team => wins t team)

/-- Number of teams tied for the maximum number of wins --/
def num_teams_with_max_wins (t : Tournament) : Nat :=
  (Finset.univ.filter (λ team => wins t team = max_wins t)).card

/-- The main theorem --/
theorem max_teams_tied_for_most_wins (t : Tournament) 
  (h : valid_tournament t) : 
  num_teams_with_max_wins t ≤ 6 ∧ 
  ∃ t' : Tournament, valid_tournament t' ∧ num_teams_with_max_wins t' = 6 := by
  sorry


end NUMINAMATH_CALUDE_max_teams_tied_for_most_wins_l2352_235205


namespace NUMINAMATH_CALUDE_mutually_exclusive_not_opposite_l2352_235253

structure PencilCase where
  pencils : ℕ
  pens : ℕ

def case : PencilCase := { pencils := 2, pens := 2 }

def select_two (pc : PencilCase) : ℕ := 2

def exactly_one_pen (pc : PencilCase) : Prop :=
  ∃ (x : ℕ), x = 1 ∧ x ≤ pc.pens

def exactly_two_pencils (pc : PencilCase) : Prop :=
  ∃ (x : ℕ), x = 2 ∧ x ≤ pc.pencils

theorem mutually_exclusive_not_opposite :
  (exactly_one_pen case ∧ exactly_two_pencils case → False) ∧
  ¬(exactly_one_pen case ↔ ¬exactly_two_pencils case) :=
by sorry

end NUMINAMATH_CALUDE_mutually_exclusive_not_opposite_l2352_235253


namespace NUMINAMATH_CALUDE_complex_equality_l2352_235245

theorem complex_equality (a : ℝ) : 
  (1 + (a - 2) * Complex.I).im = 0 → (a + Complex.I) / Complex.I = 1 - 2 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_equality_l2352_235245


namespace NUMINAMATH_CALUDE_master_zhang_apple_sales_l2352_235260

/-- The number of apples Master Zhang must sell to make a profit of 15 yuan -/
def apples_to_sell : ℕ := 100

/-- The buying price in yuan per apple -/
def buying_price : ℚ := 1 / 4

/-- The selling price in yuan per apple -/
def selling_price : ℚ := 2 / 5

/-- The desired profit in yuan -/
def desired_profit : ℕ := 15

theorem master_zhang_apple_sales :
  apples_to_sell = (desired_profit : ℚ) / (selling_price - buying_price) := by sorry

end NUMINAMATH_CALUDE_master_zhang_apple_sales_l2352_235260


namespace NUMINAMATH_CALUDE_fraction_product_result_l2352_235293

def fraction_product (n : ℕ) : ℚ :=
  let seq (k : ℕ) := 2 + 3 * k
  (seq 0) / (seq n)

theorem fraction_product_result :
  fraction_product 667 = 2 / 2007 := by sorry

end NUMINAMATH_CALUDE_fraction_product_result_l2352_235293


namespace NUMINAMATH_CALUDE_dave_guitar_strings_l2352_235267

/-- The number of guitar strings Dave breaks per night -/
def strings_per_night : ℕ := 2

/-- The number of shows Dave performs per week -/
def shows_per_week : ℕ := 6

/-- The number of weeks Dave performs -/
def total_weeks : ℕ := 12

/-- The total number of guitar strings Dave needs to replace -/
def total_strings : ℕ := strings_per_night * shows_per_week * total_weeks

theorem dave_guitar_strings :
  total_strings = 144 := by sorry

end NUMINAMATH_CALUDE_dave_guitar_strings_l2352_235267


namespace NUMINAMATH_CALUDE_new_person_weight_l2352_235228

/-- Given 4 people, with one weighing 95 kg, if the average weight increases by 8.5 kg
    when a new person replaces the 95 kg person, then the new person weighs 129 kg. -/
theorem new_person_weight (initial_count : Nat) (replaced_weight : Real) (avg_increase : Real) :
  initial_count = 4 →
  replaced_weight = 95 →
  avg_increase = 8.5 →
  (initial_count : Real) * avg_increase + replaced_weight = 129 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l2352_235228


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l2352_235288

theorem simplify_fraction_product : 5 * (18 / 7) * (21 / -63) = -30 / 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l2352_235288


namespace NUMINAMATH_CALUDE_stage_20_toothpicks_l2352_235256

/-- Calculates the number of toothpicks in a given stage of the pattern -/
def toothpicks (stage : ℕ) : ℕ :=
  3 + 3 * (stage - 1)

/-- Theorem: The 20th stage of the pattern has 60 toothpicks -/
theorem stage_20_toothpicks : toothpicks 20 = 60 := by
  sorry

end NUMINAMATH_CALUDE_stage_20_toothpicks_l2352_235256


namespace NUMINAMATH_CALUDE_tree_height_proof_l2352_235255

/-- Proves that a tree with current height 180 inches, which is 50% taller than its original height, had an original height of 10 feet. -/
theorem tree_height_proof (current_height : ℝ) (height_increase_percent : ℝ) 
  (h1 : current_height = 180)
  (h2 : height_increase_percent = 50)
  (h3 : current_height = (1 + height_increase_percent / 100) * (12 * 10)) : 
  ∃ (original_height_feet : ℝ), original_height_feet = 10 :=
by
  sorry

#check tree_height_proof

end NUMINAMATH_CALUDE_tree_height_proof_l2352_235255


namespace NUMINAMATH_CALUDE_sequence_formula_l2352_235209

def S (n : ℕ+) : ℤ := -n^2 + 7*n

def a (n : ℕ+) : ℤ := -2*n + 8

theorem sequence_formula (n : ℕ+) : 
  (∀ k : ℕ+, S k = -k^2 + 7*k) → 
  a n = -2*n + 8 := by
  sorry

end NUMINAMATH_CALUDE_sequence_formula_l2352_235209


namespace NUMINAMATH_CALUDE_negation_of_inequality_proposition_l2352_235268

theorem negation_of_inequality_proposition :
  (¬ ∀ a b : ℝ, a^2 + b^2 ≥ 2*a*b) ↔ (∃ a b : ℝ, a^2 + b^2 < 2*a*b) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_inequality_proposition_l2352_235268


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2352_235202

/-- Represents a hyperbola with foci on the y-axis -/
structure Hyperbola where
  /-- The distance from the center to a focus -/
  c : ℝ
  /-- The length of the semi-major axis -/
  a : ℝ
  /-- The length of the semi-minor axis -/
  b : ℝ
  /-- One focus lies on the line 5x-2y+20=0 -/
  focus_on_line : c = 10
  /-- The ratio of c to a is 5/3 -/
  c_a_ratio : c / a = 5 / 3
  /-- Relationship between a, b, and c -/
  abc_relation : b^2 = c^2 - a^2

/-- The equation of the hyperbola is x²/64 - y²/36 = -1 -/
theorem hyperbola_equation (h : Hyperbola) :
  ∀ x y : ℝ, (x^2 / 64 - y^2 / 36 = -1) ↔ h.b^2 * y^2 - h.a^2 * x^2 = h.a^2 * h.b^2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2352_235202


namespace NUMINAMATH_CALUDE_number_equation_solution_l2352_235223

theorem number_equation_solution : ∃ x : ℝ, x + x + 2*x + 4*x = 104 ∧ x = 13 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l2352_235223


namespace NUMINAMATH_CALUDE_sin_alpha_plus_pi_l2352_235210

theorem sin_alpha_plus_pi (α : Real) :
  (∃ P : ℝ × ℝ, P.1 = Real.sin (5 * Real.pi / 3) ∧ P.2 = Real.cos (5 * Real.pi / 3) ∧
   P.1 = Real.sin α ∧ P.2 = Real.cos α) →
  Real.sin (α + Real.pi) = -1/2 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_plus_pi_l2352_235210


namespace NUMINAMATH_CALUDE_expression_not_constant_l2352_235252

theorem expression_not_constant : 
  ∀ x y : ℝ, x ≠ 3 → x ≠ -2 → y ≠ 3 → y ≠ -2 → x ≠ y → 
  (3*x^2 + 2*x - 5) / ((x-3)*(x+2)) - (5*x - 7) / ((x-3)*(x+2)) ≠ 
  (3*y^2 + 2*y - 5) / ((y-3)*(y+2)) - (5*y - 7) / ((y-3)*(y+2)) := by
  sorry

end NUMINAMATH_CALUDE_expression_not_constant_l2352_235252


namespace NUMINAMATH_CALUDE_average_score_theorem_l2352_235273

def max_score : ℕ := 900
def amar_percent : ℕ := 64
def bhavan_percent : ℕ := 36
def chetan_percent : ℕ := 44
def num_boys : ℕ := 3

theorem average_score_theorem :
  let amar_score := max_score * amar_percent / 100
  let bhavan_score := max_score * bhavan_percent / 100
  let chetan_score := max_score * chetan_percent / 100
  let total_score := amar_score + bhavan_score + chetan_score
  (total_score / num_boys : ℚ) = 432 := by sorry

end NUMINAMATH_CALUDE_average_score_theorem_l2352_235273


namespace NUMINAMATH_CALUDE_city_population_l2352_235280

theorem city_population (population_percentage : Real) (partial_population : ℕ) (total_population : ℕ) : 
  population_percentage = 0.85 →
  partial_population = 85000 →
  population_percentage * (total_population : Real) = partial_population →
  total_population = 100000 :=
by
  sorry

end NUMINAMATH_CALUDE_city_population_l2352_235280


namespace NUMINAMATH_CALUDE_initial_mixture_volume_l2352_235249

/-- Given a mixture of milk and water with an initial ratio of 4:1,
    adding 3 litres of water results in a new ratio of 3:1.
    This theorem proves that the initial volume of the mixture was 45 litres. -/
theorem initial_mixture_volume
  (initial_milk : ℝ)
  (initial_water : ℝ)
  (h1 : initial_milk / initial_water = 4)
  (h2 : initial_milk / (initial_water + 3) = 3) :
  initial_milk + initial_water = 45 := by
sorry

end NUMINAMATH_CALUDE_initial_mixture_volume_l2352_235249


namespace NUMINAMATH_CALUDE_valid_integers_exist_l2352_235257

def is_valid_integer (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧
  (n / 1000 + (n / 100 % 10) + (n / 10 % 10) + (n % 10) = 18) ∧
  ((n / 100 % 10) + (n / 10 % 10) = 11) ∧
  (n / 1000 - n % 10 = 3) ∧
  n % 9 = 0

theorem valid_integers_exist : ∃ n : ℕ, is_valid_integer n :=
sorry

end NUMINAMATH_CALUDE_valid_integers_exist_l2352_235257


namespace NUMINAMATH_CALUDE_modified_counting_game_45th_number_l2352_235242

/-- Represents the modified counting game sequence -/
def modifiedSequence (n : ℕ) : ℕ :=
  n + (n - 1) / 10

/-- The 45th number in the modified counting game is 54 -/
theorem modified_counting_game_45th_number : modifiedSequence 45 = 54 := by
  sorry

end NUMINAMATH_CALUDE_modified_counting_game_45th_number_l2352_235242


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2352_235235

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, -1/2 < x ∧ x < 1/3 ↔ a * x^2 + b * x + 2 > 0) →
  a - b = -10 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2352_235235


namespace NUMINAMATH_CALUDE_polynomial_equality_unique_solution_l2352_235241

theorem polynomial_equality_unique_solution :
  ∃! (a b c : ℤ), ∀ (x : ℝ), (x - a) * (x - 11) + 2 = (x + b) * (x + c) ∧
  a = 13 ∧ b = -13 ∧ c = -12 :=
sorry

end NUMINAMATH_CALUDE_polynomial_equality_unique_solution_l2352_235241


namespace NUMINAMATH_CALUDE_book_chapters_l2352_235276

/-- The number of chapters in a book, given the number of chapters read per day and the number of days taken to finish the book. -/
def total_chapters (chapters_per_day : ℕ) (days_to_finish : ℕ) : ℕ :=
  chapters_per_day * days_to_finish

/-- Theorem stating that the total number of chapters in the book is 220,448. -/
theorem book_chapters :
  total_chapters 332 664 = 220448 := by
  sorry

end NUMINAMATH_CALUDE_book_chapters_l2352_235276


namespace NUMINAMATH_CALUDE_fraction_comparison_l2352_235211

theorem fraction_comparison (a b c d : ℝ) (h1 : a/b < c/d) (h2 : b > d) (h3 : d > 0) :
  (a+c)/(b+d) < (1/2) * (a/b + c/d) := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l2352_235211


namespace NUMINAMATH_CALUDE_inaccurate_tape_measurement_l2352_235200

theorem inaccurate_tape_measurement 
  (wholesale_price : ℝ) 
  (tape_length : ℝ) 
  (retail_markup : ℝ) 
  (actual_profit : ℝ) 
  (h1 : retail_markup = 0.4)
  (h2 : actual_profit = 0.39)
  (h3 : ((1 + retail_markup) * wholesale_price - tape_length * wholesale_price) / (tape_length * wholesale_price) = actual_profit) :
  tape_length = 140 / 139 :=
sorry

end NUMINAMATH_CALUDE_inaccurate_tape_measurement_l2352_235200


namespace NUMINAMATH_CALUDE_modulo_17_residue_l2352_235264

theorem modulo_17_residue : (305 + 7 * 51 + 11 * 187 + 6 * 23) % 17 = 3 := by
  sorry

end NUMINAMATH_CALUDE_modulo_17_residue_l2352_235264


namespace NUMINAMATH_CALUDE_round_robin_matches_l2352_235239

/-- The number of matches in a round-robin tournament -/
def num_matches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a round-robin tournament with n players, where each player plays 
    every other player exactly once, the total number of matches is (n * (n - 1)) / 2 -/
theorem round_robin_matches (n : ℕ) (h : n > 1) : 
  num_matches n = n * (n - 1) / 2 := by
  sorry

#eval num_matches 10  -- Should evaluate to 45

end NUMINAMATH_CALUDE_round_robin_matches_l2352_235239


namespace NUMINAMATH_CALUDE_line_m_equation_l2352_235292

/-- Two distinct lines in the xy-plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflection of a point about a line -/
def reflect (p : Point) (l : Line) : Point :=
  sorry

theorem line_m_equation (ℓ m : Line) (Q Q'' : Point) :
  ℓ.a = 1 ∧ ℓ.b = 3 ∧ ℓ.c = 7 ∧  -- Equation of line ℓ: x + 3y = 7
  Q.x = 2 ∧ Q.y = 5 ∧  -- Coordinates of Q
  Q''.x = 5 ∧ Q''.y = 0 ∧  -- Coordinates of Q''
  (1 : ℝ) * ℓ.a + 2 * ℓ.b = ℓ.c ∧  -- ℓ passes through (1, 2)
  (1 : ℝ) * m.a + 2 * m.b = m.c ∧  -- m passes through (1, 2)
  Q'' = reflect (reflect Q ℓ) m →  -- Q'' is the result of reflecting Q about ℓ and then m
  m.a = 2 ∧ m.b = -1 ∧ m.c = 2  -- Equation of line m: 2x - y = 2
  := by sorry

end NUMINAMATH_CALUDE_line_m_equation_l2352_235292


namespace NUMINAMATH_CALUDE_movements_correctly_classified_l2352_235231

-- Define an enumeration for movement types
inductive MovementType
  | Translation
  | Rotation

-- Define a structure for a movement
structure Movement where
  description : String
  classification : MovementType

-- Define the list of movements
def movements : List Movement := [
  { description := "Xiaoming walking forward 3 meters", classification := MovementType.Translation },
  { description := "Rocket launching into the sky", classification := MovementType.Translation },
  { description := "Car wheels constantly rotating", classification := MovementType.Rotation },
  { description := "Archer shooting an arrow onto the target", classification := MovementType.Translation }
]

-- Theorem statement
theorem movements_correctly_classified :
  movements.map (λ m => m.classification) = 
    [MovementType.Translation, MovementType.Translation, MovementType.Rotation, MovementType.Translation] := by
  sorry


end NUMINAMATH_CALUDE_movements_correctly_classified_l2352_235231


namespace NUMINAMATH_CALUDE_min_value_sum_squares_l2352_235243

theorem min_value_sum_squares (a b s : ℝ) (h : 2 * a + 2 * b = s) :
  ∃ (min : ℝ), min = s^2 / 2 ∧ ∀ (x y : ℝ), 2 * x + 2 * y = s → 2 * x^2 + 2 * y^2 ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_l2352_235243


namespace NUMINAMATH_CALUDE_inverse_variation_example_l2352_235296

/-- Two quantities vary inversely if their product is constant -/
def VaryInversely (a b : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, a x * b x = k

theorem inverse_variation_example :
  ∀ a b : ℝ → ℝ,
  VaryInversely a b →
  a 1500 = 1500 →
  b 1500 = 0.45 →
  a 3000 = 3000 →
  b 3000 = 0.225 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_example_l2352_235296


namespace NUMINAMATH_CALUDE_ellipse_foci_condition_l2352_235225

/-- Represents an ellipse defined by the equation mx^2 + ny^2 = 1 -/
structure Ellipse (m n : ℝ) where
  eq : ∀ x y : ℝ, m * x^2 + n * y^2 = 1

/-- Indicates whether an ellipse has foci on the x-axis -/
def has_foci_on_x_axis (e : Ellipse m n) : Prop :=
  n > m ∧ m > 0

/-- The condition m > n > 0 is neither sufficient nor necessary for an ellipse to have foci on the x-axis -/
theorem ellipse_foci_condition (m n : ℝ) :
  ¬(∀ e : Ellipse m n, m > n ∧ n > 0 → has_foci_on_x_axis e) ∧
  ¬(∀ e : Ellipse m n, has_foci_on_x_axis e → m > n ∧ n > 0) :=
sorry

end NUMINAMATH_CALUDE_ellipse_foci_condition_l2352_235225


namespace NUMINAMATH_CALUDE_x_squared_congruence_l2352_235262

theorem x_squared_congruence (x : ℤ) : 
  (5 * x ≡ 10 [ZMOD 25]) → (4 * x ≡ 20 [ZMOD 25]) → (x^2 ≡ 0 [ZMOD 25]) := by
sorry

end NUMINAMATH_CALUDE_x_squared_congruence_l2352_235262


namespace NUMINAMATH_CALUDE_remaining_money_l2352_235299

-- Define the plant sales
def orchid_sales : ℕ := 30
def orchid_price : ℕ := 50
def money_plant_sales : ℕ := 25
def money_plant_price : ℕ := 30
def bonsai_sales : ℕ := 15
def bonsai_price : ℕ := 75
def cacti_sales : ℕ := 20
def cacti_price : ℕ := 20

-- Define the expenses
def num_workers : ℕ := 4
def worker_pay : ℕ := 60
def new_pots_cost : ℕ := 250
def utility_bill : ℕ := 200
def tax : ℕ := 500

-- Calculate total earnings
def total_earnings : ℕ := 
  orchid_sales * orchid_price + 
  money_plant_sales * money_plant_price + 
  bonsai_sales * bonsai_price + 
  cacti_sales * cacti_price

-- Calculate total expenses
def total_expenses : ℕ := 
  num_workers * worker_pay + 
  new_pots_cost + 
  utility_bill + 
  tax

-- Theorem to prove
theorem remaining_money : 
  total_earnings - total_expenses = 2585 := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_l2352_235299


namespace NUMINAMATH_CALUDE_corner_sum_9x9_board_l2352_235282

/- Define the size of the checkerboard -/
def boardSize : Nat := 9

/- Define the total number of squares -/
def totalSquares : Nat := boardSize * boardSize

/- Define the positions of the corner and adjacent numbers -/
def cornerPositions : List Nat := [1, 2, 8, 9, 73, 74, 80, 81]

/- Theorem statement -/
theorem corner_sum_9x9_board :
  (List.sum cornerPositions) = 328 := by
  sorry

end NUMINAMATH_CALUDE_corner_sum_9x9_board_l2352_235282


namespace NUMINAMATH_CALUDE_power_division_equality_l2352_235219

theorem power_division_equality : (3 : ℕ)^15 / (27 : ℕ)^3 = 729 := by sorry

end NUMINAMATH_CALUDE_power_division_equality_l2352_235219


namespace NUMINAMATH_CALUDE_triangle_angle_not_all_greater_60_l2352_235277

theorem triangle_angle_not_all_greater_60 :
  ∀ (a b c : ℝ), 
  (a > 0 ∧ b > 0 ∧ c > 0) →  -- Angles are positive
  (a + b + c = 180) →        -- Sum of angles in a triangle is 180°
  ¬(a > 60 ∧ b > 60 ∧ c > 60) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_not_all_greater_60_l2352_235277


namespace NUMINAMATH_CALUDE_larger_number_with_hcf_and_lcm_factors_l2352_235204

/-- Given two positive integers with HCF 23 and LCM factors 13 and 14, the larger is 322 -/
theorem larger_number_with_hcf_and_lcm_factors (a b : ℕ+) : 
  (Nat.gcd a b = 23) → 
  (∃ k : ℕ+, Nat.lcm a b = 23 * 13 * 14 * k) → 
  (max a b = 322) := by
sorry

end NUMINAMATH_CALUDE_larger_number_with_hcf_and_lcm_factors_l2352_235204


namespace NUMINAMATH_CALUDE_train_probability_is_half_l2352_235283

-- Define the time interval (in minutes)
def timeInterval : ℝ := 60

-- Define the waiting time of the train (in minutes)
def waitingTime : ℝ := 30

-- Define a function to calculate the probability
noncomputable def trainProbability : ℝ :=
  let triangleArea := (1 / 2) * waitingTime * waitingTime
  let trapezoidArea := (1 / 2) * (waitingTime + timeInterval) * (timeInterval - waitingTime)
  (triangleArea + trapezoidArea) / (timeInterval * timeInterval)

-- Theorem statement
theorem train_probability_is_half :
  trainProbability = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_train_probability_is_half_l2352_235283


namespace NUMINAMATH_CALUDE_apple_distribution_problem_l2352_235274

/-- The number of ways to distribute n indistinguishable objects among k distinguishable boxes --/
def stars_and_bars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of ways to distribute apples among people --/
def apple_distribution (total_apples min_apples people : ℕ) : ℕ :=
  stars_and_bars (total_apples - people * min_apples) people

theorem apple_distribution_problem : apple_distribution 30 3 3 = 253 := by
  sorry

end NUMINAMATH_CALUDE_apple_distribution_problem_l2352_235274


namespace NUMINAMATH_CALUDE_negation_of_existential_l2352_235272

theorem negation_of_existential (P : α → Prop) : 
  (¬∃ x, P x) ↔ (∀ x, ¬P x) := by sorry

end NUMINAMATH_CALUDE_negation_of_existential_l2352_235272


namespace NUMINAMATH_CALUDE_garden_area_unchanged_l2352_235248

/-- Represents a rectangular garden with given length and width -/
structure RectangularGarden where
  length : ℝ
  width : ℝ

/-- Represents a square garden with a given side length -/
structure SquareGarden where
  side : ℝ

/-- Calculates the area of a rectangular garden -/
def area_rectangular (g : RectangularGarden) : ℝ := g.length * g.width

/-- Calculates the perimeter of a rectangular garden -/
def perimeter_rectangular (g : RectangularGarden) : ℝ := 2 * (g.length + g.width)

/-- Calculates the area of a square garden -/
def area_square (g : SquareGarden) : ℝ := g.side * g.side

/-- Calculates the perimeter of a square garden -/
def perimeter_square (g : SquareGarden) : ℝ := 4 * g.side

theorem garden_area_unchanged 
  (rect : RectangularGarden) 
  (sq : SquareGarden) 
  (partition_length : ℝ) :
  rect.length = 60 →
  rect.width = 15 →
  partition_length = 30 →
  perimeter_rectangular rect = perimeter_square sq + partition_length →
  area_rectangular rect = area_square sq :=
by sorry

end NUMINAMATH_CALUDE_garden_area_unchanged_l2352_235248


namespace NUMINAMATH_CALUDE_transportation_theorem_l2352_235224

/-- Represents the capacity and cost of vehicles --/
structure VehicleInfo where
  typeA_capacity : ℝ
  typeB_capacity : ℝ
  typeA_cost : ℝ
  typeB_cost : ℝ

/-- Represents the transportation problem --/
structure TransportationProblem where
  info : VehicleInfo
  total_vehicles : ℕ
  min_transport : ℝ
  max_cost : ℝ

/-- Solves the transportation problem --/
def solve_transportation (p : TransportationProblem) :
  (ℝ × ℝ) × ℕ × (ℕ × ℕ × ℝ) :=
sorry

/-- The main theorem --/
theorem transportation_theorem (p : TransportationProblem) :
  let vi := VehicleInfo.mk 50 40 3000 2000
  let tp := TransportationProblem.mk vi 20 955 58800
  let ((typeA_cap, typeB_cap), min_typeA, (opt_typeA, opt_typeB, min_cost)) := solve_transportation tp
  typeA_cap = 50 ∧ 
  typeB_cap = 40 ∧ 
  min_typeA = 16 ∧ 
  opt_typeA = 16 ∧ 
  opt_typeB = 4 ∧ 
  min_cost = 56000 ∧
  5 * typeA_cap + 3 * typeB_cap = 370 ∧
  4 * typeA_cap + 7 * typeB_cap = 480 ∧
  opt_typeA + opt_typeB = p.total_vehicles ∧
  opt_typeA * typeA_cap + opt_typeB * typeB_cap ≥ p.min_transport ∧
  opt_typeA * p.info.typeA_cost + opt_typeB * p.info.typeB_cost ≤ p.max_cost :=
by sorry


end NUMINAMATH_CALUDE_transportation_theorem_l2352_235224


namespace NUMINAMATH_CALUDE_asymptote_coincidence_l2352_235216

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 = 1

-- Define the parabola
def parabola (p x y : ℝ) : Prop := y^2 = 2 * p * x

-- Define the asymptote of the hyperbola
def hyperbola_asymptote (x : ℝ) : Prop := x = -3/2 ∨ x = 3/2

-- Define the asymptote of the parabola
def parabola_asymptote (p x : ℝ) : Prop := x = -p/2

-- State the theorem
theorem asymptote_coincidence (p : ℝ) :
  (p > 0) →
  (∃ x : ℝ, hyperbola_asymptote x ∧ parabola_asymptote p x) →
  p = 3 :=
sorry

end NUMINAMATH_CALUDE_asymptote_coincidence_l2352_235216


namespace NUMINAMATH_CALUDE_john_finish_time_l2352_235281

-- Define the start time of the first task
def start_time : Nat := 14 * 60 + 30  -- 2:30 PM in minutes since midnight

-- Define the end time of the second task
def end_second_task : Nat := 16 * 60 + 20  -- 4:20 PM in minutes since midnight

-- Define the number of tasks
def num_tasks : Nat := 4

-- Theorem statement
theorem john_finish_time :
  let task_duration := (end_second_task - start_time) / 2
  let finish_time := end_second_task + 2 * task_duration
  finish_time = 18 * 60 + 10  -- 6:10 PM in minutes since midnight
  := by sorry

end NUMINAMATH_CALUDE_john_finish_time_l2352_235281


namespace NUMINAMATH_CALUDE_F_composition_result_l2352_235227

def F (x : ℝ) : ℝ := 2 * x - 1

theorem F_composition_result : F (F (F (F (F 2)))) = 33 := by
  sorry

end NUMINAMATH_CALUDE_F_composition_result_l2352_235227


namespace NUMINAMATH_CALUDE_second_next_perfect_square_l2352_235250

theorem second_next_perfect_square (x : ℕ) (h : ∃ k : ℕ, x = k^2) :
  ∃ m : ℕ, m^2 = x + 4 * (x : ℝ).sqrt + 4 :=
sorry

end NUMINAMATH_CALUDE_second_next_perfect_square_l2352_235250


namespace NUMINAMATH_CALUDE_enrico_earnings_l2352_235298

/-- Calculates the earnings from selling roosters -/
def rooster_earnings (price_per_kg : ℚ) (weights : List ℚ) : ℚ :=
  (weights.map (· * price_per_kg)).sum

/-- Proves that Enrico's earnings from selling two roosters are $35 -/
theorem enrico_earnings : 
  let price_per_kg : ℚ := 1/2
  let weights : List ℚ := [30, 40]
  rooster_earnings price_per_kg weights = 35 := by
sorry

#eval rooster_earnings (1/2) [30, 40]

end NUMINAMATH_CALUDE_enrico_earnings_l2352_235298


namespace NUMINAMATH_CALUDE_right_triangle_has_one_right_angle_l2352_235254

-- Define a right triangle
structure RightTriangle where
  angles : Fin 3 → ℝ
  sum_180 : angles 0 + angles 1 + angles 2 = 180
  has_right_angle : ∃ i, angles i = 90

-- Theorem: A right triangle has exactly one right angle
theorem right_triangle_has_one_right_angle (t : RightTriangle) : 
  (∃! i, t.angles i = 90) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_has_one_right_angle_l2352_235254


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_ratio_l2352_235279

theorem arithmetic_geometric_mean_ratio (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) :
  let A := (x + y) / 2
  let G := Real.sqrt (x * y)
  A / G = 5 / 4 → x / y = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_ratio_l2352_235279


namespace NUMINAMATH_CALUDE_monthly_growth_rate_correct_max_daily_tourists_may_correct_l2352_235297

-- Define the number of tourists in February and April
def tourists_february : ℝ := 16000
def tourists_april : ℝ := 25000

-- Define the number of tourists from May 1st to May 21st
def tourists_may_21 : ℝ := 21250

-- Define the monthly average growth rate
def monthly_growth_rate : ℝ := 0.25

-- Define the function to calculate the growth over two months
def two_month_growth (initial : ℝ) (rate : ℝ) : ℝ :=
  initial * (1 + rate) ^ 2

-- Define the function to calculate the maximum number of tourists in May
def max_tourists_may (rate : ℝ) : ℝ :=
  tourists_april * (1 + rate)

-- Theorem 1: Prove the monthly average growth rate
theorem monthly_growth_rate_correct :
  two_month_growth tourists_february monthly_growth_rate = tourists_april :=
sorry

-- Theorem 2: Prove the maximum average number of tourists per day in the last 10 days of May
theorem max_daily_tourists_may_correct :
  (max_tourists_may monthly_growth_rate - tourists_may_21) / 10 = 100000 :=
sorry

end NUMINAMATH_CALUDE_monthly_growth_rate_correct_max_daily_tourists_may_correct_l2352_235297


namespace NUMINAMATH_CALUDE_total_bears_is_98_l2352_235208

/-- The maximum number of teddy bears that can be placed on each shelf. -/
def max_bears_per_shelf : ℕ := 7

/-- The number of filled shelves. -/
def filled_shelves : ℕ := 14

/-- The total number of teddy bears. -/
def total_bears : ℕ := max_bears_per_shelf * filled_shelves

/-- Theorem stating that the total number of teddy bears is 98. -/
theorem total_bears_is_98 : total_bears = 98 := by
  sorry

end NUMINAMATH_CALUDE_total_bears_is_98_l2352_235208
