import Mathlib

namespace NUMINAMATH_CALUDE_wire_length_proof_l3349_334947

theorem wire_length_proof (piece1 piece2 piece3 piece4 : ℝ) 
  (ratio_condition : piece1 / piece4 = 5 / 2 ∧ piece2 / piece4 = 7 / 2 ∧ piece3 / piece4 = 3 / 2)
  (shortest_piece : piece4 = 16) : 
  piece1 + piece2 + piece3 + piece4 = 136 := by
sorry

end NUMINAMATH_CALUDE_wire_length_proof_l3349_334947


namespace NUMINAMATH_CALUDE_parabola_c_value_l3349_334981

/-- A parabola passing through two points -/
def Parabola (b c : ℝ) :=
  {f : ℝ → ℝ | ∃ (x : ℝ), f x = x^2 + b*x + c}

/-- The parabola passes through the point (1,4) -/
def passes_through_1_4 (b c : ℝ) : Prop :=
  1^2 + b*1 + c = 4

/-- The parabola passes through the point (5,4) -/
def passes_through_5_4 (b c : ℝ) : Prop :=
  5^2 + b*5 + c = 4

/-- Theorem: For a parabola y = x² + bx + c passing through (1,4) and (5,4), c = 9 -/
theorem parabola_c_value (b c : ℝ) 
  (h1 : passes_through_1_4 b c) 
  (h2 : passes_through_5_4 b c) : 
  c = 9 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l3349_334981


namespace NUMINAMATH_CALUDE_product_of_sums_inequality_l3349_334941

theorem product_of_sums_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b) * (b + c) * (c + a) ≥ 8 * a * b * c :=
sorry

end NUMINAMATH_CALUDE_product_of_sums_inequality_l3349_334941


namespace NUMINAMATH_CALUDE_min_length_3rd_order_repeatable_last_term_value_l3349_334970

/-- Definition of a kth-order repeatable sequence -/
def is_kth_order_repeatable (a : ℕ → Fin 2) (m k : ℕ) : Prop :=
  ∃ i j, 1 ≤ i ∧ i + k - 1 ≤ m ∧ 1 ≤ j ∧ j + k - 1 ≤ m ∧ i ≠ j ∧
  ∀ t, 0 ≤ t ∧ t < k → a (i + t) = a (j + t)

theorem min_length_3rd_order_repeatable :
  ∀ m : ℕ, m ≥ 3 →
  ((∀ a : ℕ → Fin 2, is_kth_order_repeatable a m 3) ↔ m ≥ 11) :=
sorry

theorem last_term_value (a : ℕ → Fin 2) (m : ℕ) :
  m ≥ 3 →
  a 4 ≠ 1 →
  (¬ is_kth_order_repeatable a m 5) →
  (∃ b : Fin 2, is_kth_order_repeatable (Function.update a (m + 1) b) (m + 1) 5) →
  a m = 0 :=
sorry

end NUMINAMATH_CALUDE_min_length_3rd_order_repeatable_last_term_value_l3349_334970


namespace NUMINAMATH_CALUDE_transportation_theorem_l3349_334950

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


end NUMINAMATH_CALUDE_transportation_theorem_l3349_334950


namespace NUMINAMATH_CALUDE_factorization_valid_l3349_334985

theorem factorization_valid (x : ℝ) : 10 * x^2 - 5 * x = 5 * x * (2 * x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_valid_l3349_334985


namespace NUMINAMATH_CALUDE_quadratic_form_k_value_l3349_334961

theorem quadratic_form_k_value :
  ∀ (a h k : ℝ), (∀ x, x^2 - 7*x = a*(x - h)^2 + k) → k = -49/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_k_value_l3349_334961


namespace NUMINAMATH_CALUDE_square_root_problem_l3349_334949

theorem square_root_problem (h1 : Real.sqrt 99225 = 315) (h2 : Real.sqrt x = 3.15) : x = 9.9225 := by
  sorry

end NUMINAMATH_CALUDE_square_root_problem_l3349_334949


namespace NUMINAMATH_CALUDE_all_circles_contain_common_point_l3349_334931

/-- A parabola of the form y = x² + 2px + q -/
structure Parabola where
  p : ℝ
  q : ℝ

/-- The circle passing through the intersection points of a parabola with the coordinate axes -/
def circle_through_intersections (par : Parabola) : Set (ℝ × ℝ) :=
  {(x, y) | (x + par.p)^2 + (y - par.q/2)^2 = par.p^2 + par.q^2/4}

/-- Predicate to check if a parabola intersects the coordinate axes in three distinct points -/
def has_three_distinct_intersections (par : Parabola) : Prop :=
  par.p^2 > par.q ∧ par.q ≠ 0

theorem all_circles_contain_common_point :
  ∀ (par : Parabola), has_three_distinct_intersections par →
  (0, 1) ∈ circle_through_intersections par :=
sorry

end NUMINAMATH_CALUDE_all_circles_contain_common_point_l3349_334931


namespace NUMINAMATH_CALUDE_intersection_values_l3349_334968

/-- The function f(x) = mx² - 6x + 2 -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 6 * x + 2

/-- The graph of f intersects the x-axis at only one point -/
def single_intersection (m : ℝ) : Prop :=
  ∃! x, f m x = 0

theorem intersection_values (m : ℝ) :
  single_intersection m → m = 0 ∨ m = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_values_l3349_334968


namespace NUMINAMATH_CALUDE_ellipse_fixed_point_intersection_l3349_334920

theorem ellipse_fixed_point_intersection :
  ∀ (k : ℝ) (A B : ℝ × ℝ),
  k ≠ 0 →
  A ≠ (2, 0) →
  B ≠ (2, 0) →
  A.1^2 / 4 + A.2^2 / 3 = 1 →
  B.1^2 / 4 + B.2^2 / 3 = 1 →
  A.2 = k * (A.1 - 2/7) →
  B.2 = k * (B.1 - 2/7) →
  (A.1 - 2)^2 + A.2^2 = (B.1 - 2)^2 + B.2^2 →
  (A.1 - 2)^2 + A.2^2 = (A.1 - B.1)^2 + (A.2 - B.2)^2 →
  ∃ (m : ℝ), A.2 = k * (A.1 - 2/7) ∧ B.2 = k * (B.1 - 2/7) := by
sorry

end NUMINAMATH_CALUDE_ellipse_fixed_point_intersection_l3349_334920


namespace NUMINAMATH_CALUDE_distance_AC_proof_l3349_334997

/-- The distance between two cities A and C, given specific travel conditions. -/
def distance_AC : ℝ := 17.5

/-- The speed of the truck in km/h. -/
def truck_speed : ℝ := 50

/-- The distance traveled by delivery person A before meeting the truck, in km. -/
def distance_A_meeting : ℝ := 3

/-- The time between the meeting point and arrival at C, in hours. -/
def time_after_meeting : ℝ := 0.2  -- 12 minutes = 0.2 hours

/-- Theorem stating the distance between cities A and C under given conditions. -/
theorem distance_AC_proof :
  ∃ (speed_delivery : ℝ),
    speed_delivery > 0 ∧
    distance_AC = truck_speed * (time_after_meeting + distance_A_meeting / truck_speed) :=
by sorry


end NUMINAMATH_CALUDE_distance_AC_proof_l3349_334997


namespace NUMINAMATH_CALUDE_eggs_taken_l3349_334956

theorem eggs_taken (initial : ℕ) (remaining : ℕ) (taken : ℕ) : 
  initial = 47 → remaining = 42 → taken = initial - remaining → taken = 5 := by
  sorry

end NUMINAMATH_CALUDE_eggs_taken_l3349_334956


namespace NUMINAMATH_CALUDE_reflected_ray_equation_l3349_334913

/-- Given that:
  - Point A is at (-1/2, 0)
  - Point B is at (0, 1)
  - A' is the reflection of A across the y-axis
Prove that the line passing through A' and B has the equation 2x + y - 1 = 0 -/
theorem reflected_ray_equation (A : ℝ × ℝ) (B : ℝ × ℝ) (A' : ℝ × ℝ) :
  A = (-1/2, 0) →
  B = (0, 1) →
  A'.1 = -A.1 →  -- A' is reflection of A across y-axis
  A'.2 = A.2 →   -- A' is reflection of A across y-axis
  ∀ (x y : ℝ), (x = A'.1 ∧ y = A'.2) ∨ (x = B.1 ∧ y = B.2) →
    2 * x + y - 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_reflected_ray_equation_l3349_334913


namespace NUMINAMATH_CALUDE_math_homework_pages_l3349_334908

theorem math_homework_pages 
  (total_pages : ℕ) 
  (math_pages : ℕ) 
  (reading_pages : ℕ) 
  (problems_per_page : ℕ) 
  (total_problems : ℕ) :
  total_pages = math_pages + reading_pages →
  reading_pages = 6 →
  problems_per_page = 4 →
  total_problems = 40 →
  math_pages = 4 := by
sorry

end NUMINAMATH_CALUDE_math_homework_pages_l3349_334908


namespace NUMINAMATH_CALUDE_circular_field_diameter_circular_field_diameter_proof_l3349_334923

/-- The diameter of a circular field given the cost of fencing per meter and the total cost -/
theorem circular_field_diameter (cost_per_meter : ℝ) (total_cost : ℝ) : ℝ :=
  let circumference := total_cost / cost_per_meter
  circumference / Real.pi

/-- Proof that the diameter of the circular field is approximately 28 meters -/
theorem circular_field_diameter_proof :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |circular_field_diameter 1.50 131.95 - 28| < ε :=
sorry

end NUMINAMATH_CALUDE_circular_field_diameter_circular_field_diameter_proof_l3349_334923


namespace NUMINAMATH_CALUDE_sum_of_squares_specific_numbers_l3349_334940

theorem sum_of_squares_specific_numbers : 52^2 + 81^2 + 111^2 = 21586 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_specific_numbers_l3349_334940


namespace NUMINAMATH_CALUDE_smallest_positive_root_l3349_334952

noncomputable def α : Real := Real.arctan (2 / 9)
noncomputable def β : Real := Real.arctan (6 / 7)

def equation (x : Real) : Prop :=
  2 * Real.sin (6 * x) + 9 * Real.cos (6 * x) = 6 * Real.sin (2 * x) + 7 * Real.cos (2 * x)

theorem smallest_positive_root :
  ∃ (x : Real), x > 0 ∧ equation x ∧ ∀ (y : Real), y > 0 ∧ equation y → x ≤ y :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_root_l3349_334952


namespace NUMINAMATH_CALUDE_set_A_equals_explicit_set_l3349_334959

def set_A : Set (ℤ × ℤ) :=
  {p | p.1^2 = p.2 + 1 ∧ |p.1| < 2}

theorem set_A_equals_explicit_set : 
  set_A = {(-1, 0), (0, -1), (1, 0)} := by sorry

end NUMINAMATH_CALUDE_set_A_equals_explicit_set_l3349_334959


namespace NUMINAMATH_CALUDE_fourth_year_exam_count_l3349_334965

/-- Represents the number of exams taken in each year -/
structure ExamCount where
  year1 : ℕ
  year2 : ℕ
  year3 : ℕ
  year4 : ℕ
  year5 : ℕ

/-- Conditions for the exam count problem -/
def ValidExamCount (e : ExamCount) : Prop :=
  e.year1 + e.year2 + e.year3 + e.year4 + e.year5 = 31 ∧
  e.year1 < e.year2 ∧ e.year2 < e.year3 ∧ e.year3 < e.year4 ∧ e.year4 < e.year5 ∧
  e.year5 = 3 * e.year1

/-- The theorem stating that if the exam count is valid, the fourth year must have 8 exams -/
theorem fourth_year_exam_count (e : ExamCount) : ValidExamCount e → e.year4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_fourth_year_exam_count_l3349_334965


namespace NUMINAMATH_CALUDE_game_results_l3349_334918

/-- Represents a strategy for choosing digits -/
def Strategy := Nat → Nat

/-- Represents the result of the game -/
inductive GameResult
| FirstPlayerWins
| SecondPlayerWins

/-- Determines if a list of digits is divisible by 9 -/
def isDivisibleBy9 (digits : List Nat) : Prop :=
  digits.sum % 9 = 0

/-- Simulates the game for a given k and returns the result -/
def playGame (k : Nat) (firstPlayerStrategy : Strategy) (secondPlayerStrategy : Strategy) : GameResult :=
  sorry

/-- Theorem stating the game results for k = 10 and k = 15 -/
theorem game_results :
  (∀ (firstPlayerStrategy : Strategy),
    ∃ (secondPlayerStrategy : Strategy),
      playGame 10 firstPlayerStrategy secondPlayerStrategy = GameResult.SecondPlayerWins) ∧
  (∃ (firstPlayerStrategy : Strategy),
    ∀ (secondPlayerStrategy : Strategy),
      playGame 15 firstPlayerStrategy secondPlayerStrategy = GameResult.FirstPlayerWins) :=
sorry

end NUMINAMATH_CALUDE_game_results_l3349_334918


namespace NUMINAMATH_CALUDE_point_outside_circle_l3349_334936

/-- A line intersects a circle if and only if the distance from the center of the circle to the line is less than the radius of the circle. -/
axiom line_intersects_circle (a b : ℝ) : 
  (∃ x y, a * x + b * y = 1 ∧ x^2 + y^2 = 1) ↔ (1 / Real.sqrt (a^2 + b^2) < 1)

/-- A point (x, y) is outside a circle centered at the origin with radius r if and only if x^2 + y^2 > r^2. -/
def outside_circle (x y r : ℝ) : Prop := x^2 + y^2 > r^2

theorem point_outside_circle (a b : ℝ) :
  (∃ x y, a * x + b * y = 1 ∧ x^2 + y^2 = 1) → outside_circle a b 1 := by
  sorry

end NUMINAMATH_CALUDE_point_outside_circle_l3349_334936


namespace NUMINAMATH_CALUDE_unique_solution_abs_equation_l3349_334972

theorem unique_solution_abs_equation :
  ∃! x : ℝ, |x - 10| + |x - 14| = |3*x - 42| :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_abs_equation_l3349_334972


namespace NUMINAMATH_CALUDE_emily_egg_collection_l3349_334910

/-- The number of baskets Emily used -/
def num_baskets : ℕ := 303

/-- The number of eggs in each basket -/
def eggs_per_basket : ℕ := 28

/-- The total number of eggs Emily collected -/
def total_eggs : ℕ := num_baskets * eggs_per_basket

theorem emily_egg_collection : total_eggs = 8484 := by
  sorry

end NUMINAMATH_CALUDE_emily_egg_collection_l3349_334910


namespace NUMINAMATH_CALUDE_simplify_expression_l3349_334977

theorem simplify_expression : 
  (18 * 10^9 - 6 * 10^9) / (6 * 10^4 + 3 * 10^4) = 400000 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3349_334977


namespace NUMINAMATH_CALUDE_puzzle_time_relationship_l3349_334924

/-- Represents the time needed to complete a puzzle given the gluing rate -/
def puzzle_completion_time (initial_pieces : ℕ) (pieces_per_minute : ℕ) : ℕ :=
  (initial_pieces - 1) / (pieces_per_minute - 1)

/-- Theorem stating the relationship between puzzle completion times
    with different gluing rates -/
theorem puzzle_time_relationship :
  ∀ (initial_pieces : ℕ),
    initial_pieces > 1 →
    puzzle_completion_time initial_pieces 2 = 120 →
    puzzle_completion_time initial_pieces 3 = 60 := by
  sorry

end NUMINAMATH_CALUDE_puzzle_time_relationship_l3349_334924


namespace NUMINAMATH_CALUDE_largest_integer_inequality_l3349_334945

theorem largest_integer_inequality :
  ∃ (n : ℕ), (∀ (m : ℕ), (1/4 : ℚ) + (m : ℚ)/8 < 7/8 → m ≤ n) ∧
             ((1/4 : ℚ) + (n : ℚ)/8 < 7/8) ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_inequality_l3349_334945


namespace NUMINAMATH_CALUDE_max_final_number_l3349_334916

/-- The game function that takes a list of integers and returns the largest prime divisor of their sum -/
def game (pair : List Nat) : Nat :=
  sorry

/-- Function to perform one round of the game on a list of numbers -/
def gameRound (numbers : List Nat) : List Nat :=
  sorry

/-- Function to play the game until only one number remains -/
def playUntilOne (numbers : List Nat) : Nat :=
  sorry

theorem max_final_number : 
  ∃ (finalPairing : List (List Nat)), 
    (finalPairing.join = List.range 32) ∧ 
    (∀ pair ∈ finalPairing, pair.length = 2) ∧
    (playUntilOne (finalPairing.map game) = 11) ∧
    (∀ otherPairing : List (List Nat), 
      (otherPairing.join = List.range 32) → 
      (∀ pair ∈ otherPairing, pair.length = 2) →
      playUntilOne (otherPairing.map game) ≤ 11) :=
sorry

end NUMINAMATH_CALUDE_max_final_number_l3349_334916


namespace NUMINAMATH_CALUDE_students_left_on_bus_l3349_334933

def initial_students : ℕ := 10
def students_who_left : ℕ := 3

theorem students_left_on_bus : initial_students - students_who_left = 7 := by
  sorry

end NUMINAMATH_CALUDE_students_left_on_bus_l3349_334933


namespace NUMINAMATH_CALUDE_t_level_quasi_increasing_range_l3349_334948

/-- Definition of t-level quasi-increasing function -/
def is_t_level_quasi_increasing (f : ℝ → ℝ) (t : ℝ) (M : Set ℝ) : Prop :=
  ∀ x ∈ M, (x + t) ∈ M ∧ f (x + t) ≥ f x

/-- The function we're considering -/
def f (x : ℝ) : ℝ := x^2 - 3*x

/-- The interval we're considering -/
def M : Set ℝ := {x | x ≥ 1}

/-- The main theorem -/
theorem t_level_quasi_increasing_range :
  {t : ℝ | is_t_level_quasi_increasing f t M} = {t : ℝ | t ≥ 1} := by sorry

end NUMINAMATH_CALUDE_t_level_quasi_increasing_range_l3349_334948


namespace NUMINAMATH_CALUDE_score_difference_l3349_334979

def sammy_score : ℝ := 20

def gab_score : ℝ := 2 * sammy_score

def cher_score : ℝ := 2 * gab_score

def alex_score : ℝ := cher_score * 1.1

def team1_score : ℝ := sammy_score + gab_score + cher_score + alex_score

def opponent_initial_score : ℝ := 85

def opponent_final_score : ℝ := opponent_initial_score * 1.5

theorem score_difference :
  team1_score - opponent_final_score = 100.5 := by
  sorry

end NUMINAMATH_CALUDE_score_difference_l3349_334979


namespace NUMINAMATH_CALUDE_a3_value_l3349_334999

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def geometric_sequence (a b c : ℝ) :=
  b ^ 2 = a * c

theorem a3_value (a : ℕ → ℝ) :
  arithmetic_sequence a 2 →
  geometric_sequence (a 1) (a 3) (a 4) →
  a 3 = -4 :=
by sorry

end NUMINAMATH_CALUDE_a3_value_l3349_334999


namespace NUMINAMATH_CALUDE_chord_intersection_probability_l3349_334907

/-- The probability that a chord intersects the inner circle when two points are chosen randomly
    on the outer circle of two concentric circles with radii 2 and 3 -/
theorem chord_intersection_probability (r₁ r₂ : ℝ) (h₁ : r₁ = 2) (h₂ : r₂ = 3) :
  let θ := 2 * Real.arctan (r₁ / Real.sqrt (r₂^2 - r₁^2))
  (θ / (2 * Real.pi)) = 0.2148 := by sorry

end NUMINAMATH_CALUDE_chord_intersection_probability_l3349_334907


namespace NUMINAMATH_CALUDE_linear_function_intersection_l3349_334987

-- Define the linear function
def f (k : ℝ) (x : ℝ) : ℝ := k * x + 3

-- Define the theorem
theorem linear_function_intersection (k : ℝ) :
  (∃ t : ℝ, t > 0 ∧ f k t = 0) →  -- x-axis intersection exists and is positive
  (f k 0 = 3) →  -- y-axis intersection is (0, 3)
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f k x₁ > f k x₂) →  -- y decreases as x increases
  (∃ t : ℝ, t > 0 ∧ f k t = 0 ∧ t^2 + 3^2 = 5^2) →  -- distance between intersections is 5
  k = -3/4 := by
sorry

end NUMINAMATH_CALUDE_linear_function_intersection_l3349_334987


namespace NUMINAMATH_CALUDE_trip_time_difference_l3349_334969

theorem trip_time_difference (distance1 distance2 speed : ℝ) 
  (h1 : distance1 = 160)
  (h2 : distance2 = 280)
  (h3 : speed = 40)
  : distance2 / speed - distance1 / speed = 3 := by
  sorry

end NUMINAMATH_CALUDE_trip_time_difference_l3349_334969


namespace NUMINAMATH_CALUDE_library_schedule_l3349_334991

theorem library_schedule (sam fran mike julio : ℕ) 
  (h_sam : sam = 5)
  (h_fran : fran = 8)
  (h_mike : mike = 10)
  (h_julio : julio = 12) :
  Nat.lcm (Nat.lcm (Nat.lcm sam fran) mike) julio = 120 := by
  sorry

end NUMINAMATH_CALUDE_library_schedule_l3349_334991


namespace NUMINAMATH_CALUDE_triangle_side_length_l3349_334984

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem triangle_side_length (t : Triangle) : 
  t.B = π / 3 ∧ 
  t.b = 6 ∧ 
  Real.sin t.A - 2 * Real.sin t.C = 0 → 
  t.a = 4 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_side_length_l3349_334984


namespace NUMINAMATH_CALUDE_min_perimeter_of_divided_rectangle_l3349_334975

/-- Represents the side lengths of the two main squares in the rectangle -/
structure MainSquares where
  a : ℕ
  b : ℕ

/-- Represents the dimensions of the rectangle -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Calculates the perimeter of a rectangle -/
def perimeter (rect : Rectangle) : ℕ :=
  2 * (rect.width + rect.height)

/-- Checks if the given main square side lengths satisfy the rectangle division conditions -/
def satisfiesConditions (squares : MainSquares) : Prop :=
  5 * squares.a + 2 * squares.b = 20 * squares.a - 3 * squares.b

/-- Calculates the rectangle dimensions from the main square side lengths -/
def calculateRectangle (squares : MainSquares) : Rectangle :=
  { width := 2 * squares.a + 2 * squares.b
  , height := 3 * squares.a + 2 * squares.b }

theorem min_perimeter_of_divided_rectangle :
  ∃ (squares : MainSquares),
    satisfiesConditions squares ∧
    ∀ (other : MainSquares),
      satisfiesConditions other →
      perimeter (calculateRectangle squares) ≤ perimeter (calculateRectangle other) ∧
      perimeter (calculateRectangle squares) = 52 :=
sorry

end NUMINAMATH_CALUDE_min_perimeter_of_divided_rectangle_l3349_334975


namespace NUMINAMATH_CALUDE_circular_bed_circumference_circular_bed_specific_circumference_l3349_334943

/-- The circumference of a circular bed containing a given number of plants -/
theorem circular_bed_circumference (num_plants : Real) (area_per_plant : Real) : Real :=
  let total_area := num_plants * area_per_plant
  let radius := (total_area / Real.pi).sqrt
  2 * Real.pi * radius

/-- Proof that the circular bed with given specifications has the expected circumference -/
theorem circular_bed_specific_circumference : 
  ∃ (ε : Real), ε > 0 ∧ ε < 0.000001 ∧ 
  |circular_bed_circumference 22.997889276778874 4 - 34.007194| < ε :=
sorry

end NUMINAMATH_CALUDE_circular_bed_circumference_circular_bed_specific_circumference_l3349_334943


namespace NUMINAMATH_CALUDE_expression_value_l3349_334922

theorem expression_value (x y : ℚ) (hx : x = -5/4) (hy : y = -3/2) :
  -2 * x - y^2 = 1/4 := by sorry

end NUMINAMATH_CALUDE_expression_value_l3349_334922


namespace NUMINAMATH_CALUDE_quadratic_increasing_implies_a_bound_l3349_334903

/-- A function f is increasing on an interval [a, +∞) if for all x₁, x₂ in the interval with x₁ < x₂, f(x₁) < f(x₂) --/
def IncreasingOn (f : ℝ → ℝ) (a : ℝ) :=
  ∀ x₁ x₂, a ≤ x₁ → x₁ < x₂ → f x₁ < f x₂

/-- The quadratic function f(x) = x^2 + 2ax + 1 --/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 1

theorem quadratic_increasing_implies_a_bound (a : ℝ) :
  IncreasingOn (f a) 2 → a ≥ -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_increasing_implies_a_bound_l3349_334903


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l3349_334982

/-- Given planar vectors a and b, prove that m = 1 makes ma + b perpendicular to a -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (h1 : a = (-1, 3)) (h2 : b = (4, -2)) :
  ∃ m : ℝ, m = 1 ∧ (m • a + b) • a = 0 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l3349_334982


namespace NUMINAMATH_CALUDE_cistern_fill_time_l3349_334994

/-- Represents a tap that can fill or empty a cistern -/
structure Tap where
  rate : ℚ  -- Rate at which the tap fills (positive) or empties (negative) the cistern per hour

/-- Calculates the time to fill a cistern given a list of taps -/
def timeTofill (taps : List Tap) : ℚ :=
  1 / (taps.map (λ t => t.rate) |>.sum)

theorem cistern_fill_time (tapA tapB tapC : Tap)
  (hA : tapA.rate = 1/3)
  (hB : tapB.rate = -1/6)
  (hC : tapC.rate = 1/2) :
  timeTofill [tapA, tapB, tapC] = 3/2 := by
  sorry

#eval timeTofill [{ rate := 1/3 }, { rate := -1/6 }, { rate := 1/2 }]

end NUMINAMATH_CALUDE_cistern_fill_time_l3349_334994


namespace NUMINAMATH_CALUDE_teahouse_on_tuesday_or_thursday_not_all_plays_on_tuesday_heavenly_sound_not_on_wednesday_thunderstorm_not_only_on_tuesday_l3349_334989

-- Define the days of the week
inductive Day
| Monday
| Tuesday
| Wednesday
| Thursday

-- Define the plays
inductive Play
| Thunderstorm
| Teahouse
| HeavenlySound
| ShatteredHoofbeats

def Schedule := Day → Play

def valid_schedule (s : Schedule) : Prop :=
  (s Day.Monday ≠ Play.Thunderstorm) ∧
  (s Day.Thursday ≠ Play.Thunderstorm) ∧
  (s Day.Monday ≠ Play.Teahouse) ∧
  (s Day.Wednesday ≠ Play.Teahouse) ∧
  (s Day.Wednesday ≠ Play.HeavenlySound) ∧
  (s Day.Thursday ≠ Play.HeavenlySound) ∧
  (s Day.Monday ≠ Play.ShatteredHoofbeats) ∧
  (s Day.Thursday ≠ Play.ShatteredHoofbeats) ∧
  (∀ d1 d2, d1 ≠ d2 → s d1 ≠ s d2)

theorem teahouse_on_tuesday_or_thursday :
  ∃ (s : Schedule), valid_schedule s ∧
    (s Day.Tuesday = Play.Teahouse ∨ s Day.Thursday = Play.Teahouse) :=
by sorry

theorem not_all_plays_on_tuesday :
  ¬∃ (s : Schedule), valid_schedule s ∧
    (s Day.Tuesday = Play.Thunderstorm ∧
     s Day.Tuesday = Play.Teahouse ∧
     s Day.Tuesday = Play.HeavenlySound ∧
     s Day.Tuesday = Play.ShatteredHoofbeats) :=
by sorry

theorem heavenly_sound_not_on_wednesday :
  ∀ (s : Schedule), valid_schedule s →
    s Day.Wednesday ≠ Play.HeavenlySound :=
by sorry

theorem thunderstorm_not_only_on_tuesday :
  ∃ (s1 s2 : Schedule), valid_schedule s1 ∧ valid_schedule s2 ∧
    s1 Day.Tuesday = Play.Thunderstorm ∧
    s2 Day.Wednesday = Play.Thunderstorm :=
by sorry

end NUMINAMATH_CALUDE_teahouse_on_tuesday_or_thursday_not_all_plays_on_tuesday_heavenly_sound_not_on_wednesday_thunderstorm_not_only_on_tuesday_l3349_334989


namespace NUMINAMATH_CALUDE_total_items_for_58_slices_l3349_334927

/-- Given the number of slices of bread, calculate the total number of items -/
def totalItems (slices : ℕ) : ℕ :=
  let milk := slices - 18
  let cookies := slices + 27
  slices + milk + cookies

theorem total_items_for_58_slices :
  totalItems 58 = 183 := by
  sorry

end NUMINAMATH_CALUDE_total_items_for_58_slices_l3349_334927


namespace NUMINAMATH_CALUDE_right_triangle_area_l3349_334946

theorem right_triangle_area (hypotenuse : ℝ) (angle : ℝ) : 
  hypotenuse = 10 * Real.sqrt 2 →
  angle = 45 →
  (1 / 2) * (hypotenuse / Real.sqrt 2) * (hypotenuse / Real.sqrt 2) = 50 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3349_334946


namespace NUMINAMATH_CALUDE_problem_solution_l3349_334958

theorem problem_solution (a b : ℝ) (h1 : a - b = 4) (h2 : a * b = 3) :
  (a^2 + b^2 = 22) ∧ ((a - 2) * (b + 2) = 7) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3349_334958


namespace NUMINAMATH_CALUDE_pencil_count_l3349_334980

-- Define the number of items in the pencil case
def total_items : ℕ := 13

-- Define the relationship between pens and pencils
def pen_pencil_relation (pencils : ℕ) : ℕ := 2 * pencils

-- Define the number of erasers
def erasers : ℕ := 1

-- Theorem statement
theorem pencil_count : 
  ∃ (pencils : ℕ), 
    pencils + pen_pencil_relation pencils + erasers = total_items ∧ 
    pencils = 4 := by
  sorry

end NUMINAMATH_CALUDE_pencil_count_l3349_334980


namespace NUMINAMATH_CALUDE_valid_numbers_l3349_334996

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧  -- three-digit number
  (n / 100 = n % 10) ∧  -- hundreds and units digits are the same
  n % 15 = 0            -- divisible by 15

theorem valid_numbers :
  {n : ℕ | is_valid_number n} = {525, 555, 585} := by sorry

end NUMINAMATH_CALUDE_valid_numbers_l3349_334996


namespace NUMINAMATH_CALUDE_exponentiation_equality_l3349_334901

theorem exponentiation_equality : 
  (-2 : ℤ)^3 = -2^3 ∧ 
  (-4 : ℤ)^2 ≠ -4^2 ∧ 
  (-1 : ℤ)^2020 ≠ (-1 : ℤ)^2021 ∧ 
  (2/3 : ℚ)^3 = (2/3 : ℚ)^3 := by sorry

end NUMINAMATH_CALUDE_exponentiation_equality_l3349_334901


namespace NUMINAMATH_CALUDE_incorrect_exponent_equality_l3349_334953

theorem incorrect_exponent_equality : (-2)^2 ≠ -(2^2) :=
by
  -- Assuming the other equalities are true
  have h1 : 2^0 = 1 := by sorry
  have h2 : (-5)^3 = -(5^3) := by sorry
  have h3 : (-1/2)^3 = -1/8 := by sorry
  
  -- Proof that (-2)^2 ≠ -(2^2)
  sorry

end NUMINAMATH_CALUDE_incorrect_exponent_equality_l3349_334953


namespace NUMINAMATH_CALUDE_angle_relations_l3349_334993

theorem angle_relations (θ : Real) 
  (h1 : θ ∈ Set.Icc (3 * Real.pi / 2) (2 * Real.pi)) -- θ is in the fourth quadrant
  (h2 : Real.sin θ + Real.cos θ = 1/5) :
  (Real.sin θ - Real.cos θ = -7/5) ∧ (Real.tan θ = -3/4) := by
  sorry

end NUMINAMATH_CALUDE_angle_relations_l3349_334993


namespace NUMINAMATH_CALUDE_marble_distribution_proof_l3349_334917

/-- The number of marbles in the jar -/
def total_marbles : ℕ := 312

/-- The number of people in the group today -/
def group_size : ℕ := 24

/-- The number of additional people joining in the future scenario -/
def additional_people : ℕ := 2

/-- The decrease in marbles per person in the future scenario -/
def marble_decrease : ℕ := 1

theorem marble_distribution_proof :
  (total_marbles / group_size = total_marbles / (group_size + additional_people) + marble_decrease) ∧
  (total_marbles % group_size = 0) :=
sorry

end NUMINAMATH_CALUDE_marble_distribution_proof_l3349_334917


namespace NUMINAMATH_CALUDE_school_population_l3349_334921

theorem school_population (x : ℝ) : 
  (242 = (x / 100) * (50 / 100 * x)) → x = 220 := by
  sorry

end NUMINAMATH_CALUDE_school_population_l3349_334921


namespace NUMINAMATH_CALUDE_bags_difference_l3349_334932

/-- The number of bags Tiffany had on Monday -/
def monday_bags : ℕ := 7

/-- The number of bags Tiffany found on the next day -/
def next_day_bags : ℕ := 12

/-- Theorem: The difference between the number of bags found on the next day
    and the number of bags on Monday is equal to 5 -/
theorem bags_difference : next_day_bags - monday_bags = 5 := by
  sorry

end NUMINAMATH_CALUDE_bags_difference_l3349_334932


namespace NUMINAMATH_CALUDE_circle_area_from_circumference_l3349_334900

theorem circle_area_from_circumference : ∀ (r : ℝ), 
  (2 * π * r = 18 * π) → (π * r^2 = 81 * π) := by
  sorry

end NUMINAMATH_CALUDE_circle_area_from_circumference_l3349_334900


namespace NUMINAMATH_CALUDE_paint_brush_square_ratio_l3349_334930

/-- Given a square with side length s and a paint brush of width w that sweeps along both diagonals,
    if half the area of the square is painted, then the ratio of the square's diagonal length to the brush width is 2√2 + 2. -/
theorem paint_brush_square_ratio (s w : ℝ) (h_positive : s > 0 ∧ w > 0) 
  (h_half_painted : w^2 + (s - w)^2 / 2 = s^2 / 2) : 
  s * Real.sqrt 2 / w = 2 * Real.sqrt 2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_paint_brush_square_ratio_l3349_334930


namespace NUMINAMATH_CALUDE_carlos_singles_percentage_l3349_334929

/-- Represents the statistics of Carlos's baseball hits -/
structure BaseballStats where
  total_hits : ℕ
  home_runs : ℕ
  triples : ℕ
  doubles : ℕ

/-- Calculates the percentage of singles in Carlos's hits -/
def percentage_singles (stats : BaseballStats) : ℚ :=
  let non_singles := stats.home_runs + stats.triples + stats.doubles
  let singles := stats.total_hits - non_singles
  (singles : ℚ) / stats.total_hits * 100

/-- Carlos's baseball statistics -/
def carlos_stats : BaseballStats :=
  { total_hits := 50
  , home_runs := 3
  , triples := 2
  , doubles := 8 }

/-- Theorem stating that the percentage of singles in Carlos's hits is 74% -/
theorem carlos_singles_percentage :
  percentage_singles carlos_stats = 74 := by
  sorry


end NUMINAMATH_CALUDE_carlos_singles_percentage_l3349_334929


namespace NUMINAMATH_CALUDE_sequence_sum_equals_n_squared_l3349_334919

def sequence_sum (n : ℕ) : ℕ :=
  (List.range (n + 1)).sum + (List.range n).sum

theorem sequence_sum_equals_n_squared (n : ℕ) : sequence_sum n = n^2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_equals_n_squared_l3349_334919


namespace NUMINAMATH_CALUDE_registration_scientific_correct_l3349_334942

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The number of people registered for the national college entrance examination in 2023 -/
def registration_number : ℕ := 12910000

/-- The scientific notation representation of the registration number -/
def registration_scientific : ScientificNotation :=
  { coefficient := 1.291,
    exponent := 7,
    is_valid := by sorry }

/-- Theorem stating that the registration number is correctly represented in scientific notation -/
theorem registration_scientific_correct :
  (registration_scientific.coefficient * (10 : ℝ) ^ registration_scientific.exponent) = registration_number := by
  sorry

end NUMINAMATH_CALUDE_registration_scientific_correct_l3349_334942


namespace NUMINAMATH_CALUDE_radio_show_song_time_l3349_334935

/-- Calculates the time spent on songs in a radio show -/
theorem radio_show_song_time (total_show_time : ℕ) (talking_segment_duration : ℕ) 
  (ad_break_duration : ℕ) (num_talking_segments : ℕ) (num_ad_breaks : ℕ) :
  total_show_time = 3 * 60 →
  talking_segment_duration = 10 →
  ad_break_duration = 5 →
  num_talking_segments = 3 →
  num_ad_breaks = 5 →
  total_show_time - (num_talking_segments * talking_segment_duration + num_ad_breaks * ad_break_duration) = 125 := by
  sorry

end NUMINAMATH_CALUDE_radio_show_song_time_l3349_334935


namespace NUMINAMATH_CALUDE_min_k_for_triangle_inequality_l3349_334939

theorem min_k_for_triangle_inequality : 
  ∃ (k : ℕ), k > 0 ∧ 
  (∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → 
    k * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2) → 
    (a + b > c ∧ b + c > a ∧ c + a > b)) ∧
  (∀ (k' : ℕ), k' > 0 → k' < k → 
    ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    k' * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2) ∧
    ¬(a + b > c ∧ b + c > a ∧ c + a > b)) ∧
  k = 6 :=
sorry

end NUMINAMATH_CALUDE_min_k_for_triangle_inequality_l3349_334939


namespace NUMINAMATH_CALUDE_number_equation_solution_l3349_334934

theorem number_equation_solution : ∃ x : ℝ, x + x + 2*x + 4*x = 104 ∧ x = 13 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l3349_334934


namespace NUMINAMATH_CALUDE_lily_remaining_milk_l3349_334938

theorem lily_remaining_milk (initial_milk : ℚ) (james_milk : ℚ) (maria_milk : ℚ) :
  initial_milk = 5 →
  james_milk = 15 / 4 →
  maria_milk = 3 / 4 →
  initial_milk - (james_milk + maria_milk) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_lily_remaining_milk_l3349_334938


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_roots_l3349_334966

/-- 
Given complex numbers a and b, this theorem states that a^2 = 2b ≠ 0 
if and only if the roots of the polynomial x^2 + ax + b form an isosceles 
right triangle on the complex plane with the right angle at the origin.
-/
theorem isosceles_right_triangle_roots 
  (a b : ℂ) : a^2 = 2*b ∧ b ≠ 0 ↔ 
  ∃ (x₁ x₂ : ℂ), x₁^2 + a*x₁ + b = 0 ∧ 
                 x₂^2 + a*x₂ + b = 0 ∧ 
                 x₁ ≠ x₂ ∧
                 (x₁ = Complex.I * x₂ ∨ x₂ = Complex.I * x₁) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_roots_l3349_334966


namespace NUMINAMATH_CALUDE_inequality_proof_l3349_334983

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (a^2 / (b + c)) + (b^2 / (c + a)) + (c^2 / (a + b)) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3349_334983


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l3349_334988

/-- Given two vectors in ℝ², prove that the magnitude of their difference is 5 -/
theorem vector_difference_magnitude (a b : ℝ × ℝ) : 
  a = (2, 1) → b = (-2, 4) → ‖a - b‖ = 5 := by sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l3349_334988


namespace NUMINAMATH_CALUDE_derivative_of_y_l3349_334905

noncomputable def y (x : ℝ) : ℝ := Real.sin (2 * x) - Real.cos (2 * x)

theorem derivative_of_y (x : ℝ) :
  deriv y x = 2 * Real.sqrt 2 * Real.cos (2 * x - Real.pi / 4) := by sorry

end NUMINAMATH_CALUDE_derivative_of_y_l3349_334905


namespace NUMINAMATH_CALUDE_bruce_goals_l3349_334925

theorem bruce_goals (bruce_goals : ℕ) 
  (michael_goals : ℕ)
  (h1 : michael_goals = 3 * bruce_goals)
  (h2 : bruce_goals + michael_goals = 16) : 
  bruce_goals = 4 := by
sorry

end NUMINAMATH_CALUDE_bruce_goals_l3349_334925


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3349_334912

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt (4 - 5 * x) = 10 → x = -19.2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3349_334912


namespace NUMINAMATH_CALUDE_parabola_directrix_l3349_334955

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop :=
  y = (x^2 - 6*x + 5) / 12

/-- The directrix equation -/
def directrix (y : ℝ) : Prop :=
  y = -10/3

/-- Theorem stating that the given directrix is correct for the parabola -/
theorem parabola_directrix :
  ∀ x y : ℝ, parabola x y → ∃ d : ℝ, directrix d ∧ 
  (∀ p : ℝ × ℝ, p.1 = x ∧ p.2 = y → 
    ∃ f : ℝ × ℝ, ∃ q : ℝ × ℝ, 
      q.2 = d ∧ 
      (p.1 - f.1)^2 + (p.2 - f.2)^2 = (p.1 - q.1)^2 + (p.2 - q.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3349_334955


namespace NUMINAMATH_CALUDE_shenny_vacation_shirts_l3349_334964

/-- The number of shirts Shenny needs to pack for her vacation -/
def shirts_to_pack (vacation_days : ℕ) (same_shirt_days : ℕ) (different_shirts_per_day : ℕ) : ℕ :=
  (vacation_days - same_shirt_days) * different_shirts_per_day + 1

/-- Proof that Shenny needs to pack 11 shirts for her vacation -/
theorem shenny_vacation_shirts :
  shirts_to_pack 7 2 2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_shenny_vacation_shirts_l3349_334964


namespace NUMINAMATH_CALUDE_walking_speed_problem_l3349_334902

/-- Proves that given a circular track of 640 m, two people walking in opposite directions 
    from the same starting point, meeting after 4.8 minutes, with one person walking at 3.8 km/hr, 
    the other person's speed is 4.2 km/hr. -/
theorem walking_speed_problem (track_length : ℝ) (meeting_time : ℝ) (geeta_speed : ℝ) :
  track_length = 640 →
  meeting_time = 4.8 →
  geeta_speed = 3.8 →
  ∃ lata_speed : ℝ,
    lata_speed = 4.2 ∧
    (lata_speed + geeta_speed) * meeting_time / 60 = track_length / 1000 :=
by sorry

end NUMINAMATH_CALUDE_walking_speed_problem_l3349_334902


namespace NUMINAMATH_CALUDE_bobs_sandwich_cost_l3349_334995

/-- Proves that the cost of each of Bob's sandwiches after discount and before tax is $2.412 -/
theorem bobs_sandwich_cost 
  (andy_soda : ℝ) (andy_hamburger : ℝ) (andy_chips : ℝ) (andy_tax_rate : ℝ)
  (bob_sandwich_before_discount : ℝ) (bob_sandwich_count : ℕ) (bob_water : ℝ)
  (bob_sandwich_discount_rate : ℝ) (bob_water_tax_rate : ℝ)
  (h_andy_soda : andy_soda = 1.50)
  (h_andy_hamburger : andy_hamburger = 2.75)
  (h_andy_chips : andy_chips = 1.25)
  (h_andy_tax_rate : andy_tax_rate = 0.08)
  (h_bob_sandwich_before_discount : bob_sandwich_before_discount = 2.68)
  (h_bob_sandwich_count : bob_sandwich_count = 5)
  (h_bob_water : bob_water = 1.25)
  (h_bob_sandwich_discount_rate : bob_sandwich_discount_rate = 0.10)
  (h_bob_water_tax_rate : bob_water_tax_rate = 0.07)
  (h_equal_total : 
    (andy_soda + 3 * andy_hamburger + andy_chips) * (1 + andy_tax_rate) = 
    bob_sandwich_count * bob_sandwich_before_discount * (1 - bob_sandwich_discount_rate) + 
    bob_water * (1 + bob_water_tax_rate)) :
  bob_sandwich_before_discount * (1 - bob_sandwich_discount_rate) = 2.412 := by
  sorry


end NUMINAMATH_CALUDE_bobs_sandwich_cost_l3349_334995


namespace NUMINAMATH_CALUDE_unique_three_digit_even_with_digit_sum_26_l3349_334998

/-- The digit sum of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Checks if a natural number is a 3-digit number -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- The set of 3-digit even numbers with digit sum 26 -/
def S : Set ℕ := {n : ℕ | is_three_digit n ∧ Even n ∧ digit_sum n = 26}

theorem unique_three_digit_even_with_digit_sum_26 : ∃! n, n ∈ S := by sorry

end NUMINAMATH_CALUDE_unique_three_digit_even_with_digit_sum_26_l3349_334998


namespace NUMINAMATH_CALUDE_opposite_reciprocal_absolute_value_l3349_334904

theorem opposite_reciprocal_absolute_value (a b c d m : ℝ) : 
  (a = -b) →  -- a and b are opposite numbers
  (c * d = 1) →  -- c and d are reciprocals
  (abs m = 5) →  -- absolute value of m is 5
  (-a - m * c * d - b = 5 ∨ -a - m * c * d - b = -5) :=
by
  sorry

end NUMINAMATH_CALUDE_opposite_reciprocal_absolute_value_l3349_334904


namespace NUMINAMATH_CALUDE_earliest_time_84_degrees_l3349_334928

/-- Temperature function representing the temperature in Austin, TX on a summer day -/
def T (t : ℝ) : ℝ := -t^2 + 14*t + 40

/-- The earliest positive real solution to the temperature equation when it equals 84 degrees -/
theorem earliest_time_84_degrees :
  ∀ t : ℝ, t > 0 → T t = 84 → t ≥ 22 :=
by sorry

end NUMINAMATH_CALUDE_earliest_time_84_degrees_l3349_334928


namespace NUMINAMATH_CALUDE_elder_sister_savings_l3349_334951

theorem elder_sister_savings (total : ℝ) (elder_donation_rate : ℝ) (younger_donation_rate : ℝ)
  (h_total : total = 108)
  (h_elder_rate : elder_donation_rate = 0.75)
  (h_younger_rate : younger_donation_rate = 0.8)
  (h_equal_remainder : ∃ (elder younger : ℝ), 
    elder + younger = total ∧ 
    elder * (1 - elder_donation_rate) = younger * (1 - younger_donation_rate)) :
  ∃ (elder : ℝ), elder = 48 ∧ 
    ∃ (younger : ℝ), younger = total - elder ∧
    elder * (1 - elder_donation_rate) = younger * (1 - younger_donation_rate) := by
  sorry

end NUMINAMATH_CALUDE_elder_sister_savings_l3349_334951


namespace NUMINAMATH_CALUDE_initial_bananas_per_child_l3349_334963

/-- Proves that the initial number of bananas per child is 2 --/
theorem initial_bananas_per_child (total_children : ℕ) (absent_children : ℕ) (extra_bananas : ℕ) : 
  total_children = 320 →
  absent_children = 160 →
  extra_bananas = 2 →
  ∃ (initial_bananas : ℕ), 
    (total_children - absent_children) * (initial_bananas + extra_bananas) = 
    total_children * initial_bananas ∧
    initial_bananas = 2 := by
  sorry

end NUMINAMATH_CALUDE_initial_bananas_per_child_l3349_334963


namespace NUMINAMATH_CALUDE_integer_root_count_l3349_334944

theorem integer_root_count : ∃! (S : Finset ℝ), 
  (∀ x ∈ S, ∃ k : ℤ, Real.sqrt (123 - Real.sqrt x) = k) ∧ 
  (∀ x : ℝ, (∃ k : ℤ, Real.sqrt (123 - Real.sqrt x) = k) → x ∈ S) ∧ 
  Finset.card S = 12 := by sorry

end NUMINAMATH_CALUDE_integer_root_count_l3349_334944


namespace NUMINAMATH_CALUDE_shaded_area_is_nine_l3349_334974

/-- Represents a point on the grid -/
structure Point where
  x : ℕ
  y : ℕ

/-- Represents the spinner shape -/
structure Spinner where
  center : Point
  armLength : ℕ

/-- Represents the entire shaded shape -/
structure ShadedShape where
  spinner : Spinner
  cornerSquares : List Point

/-- Calculates the area of the shaded shape -/
def shadedArea (shape : ShadedShape) : ℕ :=
  let spinnerArea := 2 * shape.spinner.armLength * 2 + 1
  let cornerSquaresArea := shape.cornerSquares.length
  spinnerArea + cornerSquaresArea

/-- The theorem to be proved -/
theorem shaded_area_is_nine :
  ∀ (shape : ShadedShape),
    shape.spinner.center = ⟨3, 3⟩ →
    shape.spinner.armLength = 1 →
    shape.cornerSquares.length = 4 →
    shadedArea shape = 9 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_nine_l3349_334974


namespace NUMINAMATH_CALUDE_farmer_animals_count_l3349_334960

/-- Represents the number of animals a farmer has -/
structure FarmAnimals where
  goats : ℕ
  cows : ℕ
  pigs : ℕ

/-- Calculates the total number of animals -/
def totalAnimals (animals : FarmAnimals) : ℕ :=
  animals.goats + animals.cows + animals.pigs

/-- Theorem stating the total number of animals given the conditions -/
theorem farmer_animals_count :
  ∀ (animals : FarmAnimals),
    animals.goats = 11 →
    animals.cows = animals.goats + 4 →
    animals.pigs = 2 * animals.cows →
    totalAnimals animals = 56 := by
  sorry

end NUMINAMATH_CALUDE_farmer_animals_count_l3349_334960


namespace NUMINAMATH_CALUDE_team_selection_count_l3349_334967

theorem team_selection_count (total : ℕ) (veterans : ℕ) (new : ℕ) (team_size : ℕ) (max_veterans : ℕ) :
  total = veterans + new →
  total = 10 →
  veterans = 2 →
  new = 8 →
  team_size = 3 →
  max_veterans = 1 →
  Nat.choose (new - 1) team_size + veterans * Nat.choose (new - 1) (team_size - 1) = 77 :=
by sorry

end NUMINAMATH_CALUDE_team_selection_count_l3349_334967


namespace NUMINAMATH_CALUDE_two_derived_point_of_neg_two_three_original_point_from_three_derived_k_range_for_distance_condition_l3349_334973

/-- Definition of k-derived point -/
def k_derived_point (k : ℝ) (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.1 + k * P.2, k * P.1 + P.2)

/-- Theorem 1: The 2-derived point of (-2,3) is (4, -1) -/
theorem two_derived_point_of_neg_two_three :
  k_derived_point 2 (-2, 3) = (4, -1) := by sorry

/-- Theorem 2: If the 3-derived point of P is (9,11), then P is (3,2) -/
theorem original_point_from_three_derived :
  ∀ P : ℝ × ℝ, k_derived_point 3 P = (9, 11) → P = (3, 2) := by sorry

/-- Theorem 3: For a point P(0,b) on the positive y-axis, its k-derived point P'(kb,b) 
    has |kb| ≥ 5b if and only if k ≥ 5 or k ≤ -5 -/
theorem k_range_for_distance_condition :
  ∀ k b : ℝ, b > 0 → (|k * b| ≥ 5 * b ↔ k ≥ 5 ∨ k ≤ -5) := by sorry

end NUMINAMATH_CALUDE_two_derived_point_of_neg_two_three_original_point_from_three_derived_k_range_for_distance_condition_l3349_334973


namespace NUMINAMATH_CALUDE_AM_GM_inequality_counterexample_AM_GM_inequality_l3349_334976

theorem AM_GM_inequality_counterexample : ¬ ∀ x : ℝ, x + 1/x ≥ 2 * Real.sqrt (x * (1/x)) :=
by
  sorry

theorem AM_GM_inequality {a b : ℝ} (ha : a > 0) (hb : b > 0) : a + b ≥ 2 * Real.sqrt (a * b) :=
by
  sorry

end NUMINAMATH_CALUDE_AM_GM_inequality_counterexample_AM_GM_inequality_l3349_334976


namespace NUMINAMATH_CALUDE_congruence_solution_l3349_334914

theorem congruence_solution : ∃! n : ℤ, 0 ≤ n ∧ n < 23 ∧ -250 ≡ n [ZMOD 23] ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l3349_334914


namespace NUMINAMATH_CALUDE_ferris_wheel_seats_l3349_334986

theorem ferris_wheel_seats (people_per_seat : ℕ) (total_people : ℕ) (h1 : people_per_seat = 6) (h2 : total_people = 84) :
  total_people / people_per_seat = 14 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_seats_l3349_334986


namespace NUMINAMATH_CALUDE_right_angled_triangle_not_axisymmetric_l3349_334971

-- Define the types of geometric figures
inductive GeometricFigure
  | Angle
  | EquilateralTriangle
  | LineSegment
  | RightAngledTriangle

-- Define the property of being axisymmetric
def isAxisymmetric : GeometricFigure → Prop :=
  fun figure =>
    match figure with
    | GeometricFigure.Angle => true
    | GeometricFigure.EquilateralTriangle => true
    | GeometricFigure.LineSegment => true
    | GeometricFigure.RightAngledTriangle => false

-- Theorem statement
theorem right_angled_triangle_not_axisymmetric :
  ∀ (figure : GeometricFigure),
    ¬(isAxisymmetric figure) ↔ figure = GeometricFigure.RightAngledTriangle :=
by
  sorry

end NUMINAMATH_CALUDE_right_angled_triangle_not_axisymmetric_l3349_334971


namespace NUMINAMATH_CALUDE_f_odd_and_increasing_l3349_334990

-- Define the function f(x) = x|x|
def f (x : ℝ) : ℝ := x * abs x

-- State the theorem
theorem f_odd_and_increasing : 
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_f_odd_and_increasing_l3349_334990


namespace NUMINAMATH_CALUDE_bales_equation_initial_bales_count_l3349_334937

/-- The initial number of bales in the barn -/
def initial_bales : ℕ := sorry

/-- The number of bales added to the barn -/
def added_bales : ℕ := 35

/-- The final number of bales in the barn -/
def final_bales : ℕ := 82

/-- Theorem stating that the initial number of bales plus the added bales equals the final number of bales -/
theorem bales_equation : initial_bales + added_bales = final_bales := by sorry

/-- Theorem proving that the initial number of bales was 47 -/
theorem initial_bales_count : initial_bales = 47 := by sorry

end NUMINAMATH_CALUDE_bales_equation_initial_bales_count_l3349_334937


namespace NUMINAMATH_CALUDE_unshaded_area_between_circles_l3349_334957

/-- The area of the unshaded region between two concentric circles -/
theorem unshaded_area_between_circles (r₁ r₂ : ℝ) (h₁ : r₁ = 4) (h₂ : r₂ = 7) :
  π * r₂^2 - π * r₁^2 = 33 * π :=
by sorry

end NUMINAMATH_CALUDE_unshaded_area_between_circles_l3349_334957


namespace NUMINAMATH_CALUDE_parabola_point_distance_l3349_334906

theorem parabola_point_distance (m n : ℝ) : 
  n^2 = 4*m →                             -- P(m,n) is on the parabola y^2 = 4x
  (m + 1)^2 = (m - 5)^2 + n^2 →           -- Distance from P to x=-1 equals distance from P to A(5,0)
  m = 3 := by sorry

end NUMINAMATH_CALUDE_parabola_point_distance_l3349_334906


namespace NUMINAMATH_CALUDE_min_participants_is_100_l3349_334915

/-- Represents the number of correct answers for each question in the quiz. -/
structure QuizResults where
  q1 : Nat
  q2 : Nat
  q3 : Nat
  q4 : Nat

/-- Calculates the minimum number of participants given quiz results. -/
def minParticipants (results : QuizResults) : Nat :=
  ((results.q1 + results.q2 + results.q3 + results.q4 + 1) / 2)

/-- Theorem: The minimum number of participants in the quiz is 100. -/
theorem min_participants_is_100 (results : QuizResults) 
  (h1 : results.q1 = 90)
  (h2 : results.q2 = 50)
  (h3 : results.q3 = 40)
  (h4 : results.q4 = 20)
  (h5 : ∀ n : Nat, n ≤ minParticipants results → 
       2 * n ≥ results.q1 + results.q2 + results.q3 + results.q4) :
  minParticipants results = 100 := by
  sorry

#eval minParticipants ⟨90, 50, 40, 20⟩

end NUMINAMATH_CALUDE_min_participants_is_100_l3349_334915


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3349_334962

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x + 7) + 5 = 14 → x = 74 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3349_334962


namespace NUMINAMATH_CALUDE_smallest_integer_for_quadratic_inequality_l3349_334954

theorem smallest_integer_for_quadratic_inequality :
  ∃ n : ℤ, (∀ m : ℤ, m^2 - 13*m + 40 ≤ 0 → n ≤ m) ∧ (n^2 - 13*n + 40 ≤ 0) ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_for_quadratic_inequality_l3349_334954


namespace NUMINAMATH_CALUDE_supplement_of_complement_35_l3349_334978

/-- The complement of an angle in degrees -/
def complement (α : ℝ) : ℝ := 90 - α

/-- The supplement of an angle in degrees -/
def supplement (β : ℝ) : ℝ := 180 - β

/-- The original angle in degrees -/
def original_angle : ℝ := 35

/-- Theorem: The degree measure of the supplement of the complement of a 35-degree angle is 125° -/
theorem supplement_of_complement_35 : 
  supplement (complement original_angle) = 125 := by
  sorry

end NUMINAMATH_CALUDE_supplement_of_complement_35_l3349_334978


namespace NUMINAMATH_CALUDE_max_product_l3349_334911

def digits : Finset Nat := {1, 3, 5, 8, 9}

def valid_combination (a b c d e : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ e ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

def three_digit (a b c : Nat) : Nat := 100 * a + 10 * b + c

def two_digit (d e : Nat) : Nat := 10 * d + e

theorem max_product :
  ∀ a b c d e,
    valid_combination a b c d e →
    (three_digit a b c) * (two_digit d e) ≤ (three_digit 9 3 1) * (two_digit 8 5) :=
by sorry

end NUMINAMATH_CALUDE_max_product_l3349_334911


namespace NUMINAMATH_CALUDE_price_per_pack_is_one_l3349_334992

/-- Represents the number of boxes in a carton -/
def boxes_per_carton : ℕ := 12

/-- Represents the number of packs of cheese cookies in a box -/
def packs_per_box : ℕ := 10

/-- Represents the cost of a dozen cartons in dollars -/
def cost_dozen_cartons : ℕ := 1440

/-- Represents the number of cartons in a dozen -/
def cartons_in_dozen : ℕ := 12

/-- Theorem stating that the price of a pack of cheese cookies is $1 -/
theorem price_per_pack_is_one :
  (cost_dozen_cartons : ℚ) / (cartons_in_dozen * boxes_per_carton * packs_per_box) = 1 := by
  sorry

end NUMINAMATH_CALUDE_price_per_pack_is_one_l3349_334992


namespace NUMINAMATH_CALUDE_polygon_sides_greater_than_diagonals_l3349_334909

theorem polygon_sides_greater_than_diagonals (n : ℕ) (d : ℕ) : 
  (n ≥ 3 ∧ d = n * (n - 3) / 2) → (n > d ↔ n = 3 ∨ n = 4) := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_greater_than_diagonals_l3349_334909


namespace NUMINAMATH_CALUDE_matrix_inverse_from_eigenvectors_l3349_334926

theorem matrix_inverse_from_eigenvectors :
  ∀ (a b c d : ℝ),
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d]
  (A.mulVec ![1, 1] = (6 : ℝ) • ![1, 1]) →
  (A.mulVec ![3, -2] = (1 : ℝ) • ![3, -2]) →
  A⁻¹ = !![2/3, -1/2; -1/3, 1/2] :=
by sorry

end NUMINAMATH_CALUDE_matrix_inverse_from_eigenvectors_l3349_334926
