import Mathlib

namespace triangle_determinant_zero_l3514_351445

theorem triangle_determinant_zero (A B C : Real) 
  (h : A + B + C = π) : -- condition that A, B, C are angles of a triangle
  let M : Matrix (Fin 3) (Fin 3) Real := 
    ![![Real.cos A ^ 2, Real.tan A, 1],
      ![Real.cos B ^ 2, Real.tan B, 1],
      ![Real.cos C ^ 2, Real.tan C, 1]]
  Matrix.det M = 0 := by
    sorry

end triangle_determinant_zero_l3514_351445


namespace q_divisibility_q_values_q_cubic_form_q_10_expression_l3514_351466

/-- A cubic polynomial q(x) such that [q(x)]^2 - x is divisible by (x - 2)(x + 2)(x - 5)(x - 7) -/
def q (x : ℝ) : ℝ := sorry

theorem q_divisibility (x : ℝ) : 
  ∃ k : ℝ, q x ^ 2 - x = k * ((x - 2) * (x + 2) * (x - 5) * (x - 7)) := sorry

theorem q_values : 
  q 2 = Real.sqrt 2 ∧ q (-2) = -Real.sqrt 2 ∧ q 5 = Real.sqrt 5 ∧ q 7 = Real.sqrt 7 := sorry

theorem q_cubic_form : 
  ∃ a b c d : ℝ, ∀ x : ℝ, q x = a * x^3 + b * x^2 + c * x + d := sorry

theorem q_10_expression (a b c d : ℝ) 
  (h : ∀ x : ℝ, q x = a * x^3 + b * x^2 + c * x + d) : 
  q 10 = 1000 * a + 100 * b + 10 * c + d := sorry

end q_divisibility_q_values_q_cubic_form_q_10_expression_l3514_351466


namespace negation_equivalence_l3514_351459

/-- A number is even if it's divisible by 2 -/
def IsEven (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

/-- A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself -/
def IsPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

/-- The negation of "There is an even number that is a prime number" is equivalent to "No even number is a prime number" -/
theorem negation_equivalence : 
  (¬ ∃ n : ℕ, IsEven n ∧ IsPrime n) ↔ (∀ n : ℕ, IsEven n → ¬ IsPrime n) := by
  sorry

end negation_equivalence_l3514_351459


namespace geometric_sequence_sum_l3514_351405

/-- Given a geometric sequence {a_n} with common ratio q < 0,
    if a_2 = 1 - a_1 and a_4 = 4 - a_3, then a_4 + a_5 = 16 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) (h1 : q < 0)
  (h2 : ∀ n, a (n + 1) = q * a n)  -- Definition of geometric sequence
  (h3 : a 2 = 1 - a 1)
  (h4 : a 4 = 4 - a 3) :
  a 4 + a 5 = 16 := by
sorry

end geometric_sequence_sum_l3514_351405


namespace geometric_sequence_sum_l3514_351492

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n, a n > 0) →
  a 4 * a 6 + 2 * a 5 * a 7 + a 6 * a 8 = 36 →
  a 5 + a 7 = 6 := by
  sorry

end geometric_sequence_sum_l3514_351492


namespace peters_marbles_l3514_351449

theorem peters_marbles (n : ℕ) (orange purple silver white : ℕ) : 
  n > 0 →
  orange = n / 2 →
  purple = n / 5 →
  silver = 8 →
  white = n - (orange + purple + silver) →
  n = orange + purple + silver + white →
  (∀ m : ℕ, m > 0 ∧ m < n → 
    m / 2 + m / 5 + 8 + (m - (m / 2 + m / 5 + 8)) ≠ m) →
  white = 1 := by
sorry

end peters_marbles_l3514_351449


namespace rectangle_region_perimeter_l3514_351426

/-- Given a region formed by four congruent rectangles with a total area of 360 square centimeters
    and each rectangle having a width to length ratio of 3:4, the perimeter of the region is 14√7.5 cm. -/
theorem rectangle_region_perimeter (total_area : ℝ) (width : ℝ) (length : ℝ) : 
  total_area = 360 →
  width / length = 3 / 4 →
  width * length = total_area / 4 →
  2 * (2 * width + 2 * length) = 14 * Real.sqrt 7.5 := by
  sorry

end rectangle_region_perimeter_l3514_351426


namespace perpendicular_tangents_ratio_l3514_351454

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + x

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Theorem statement
theorem perpendicular_tangents_ratio (a b : ℝ) :
  -- Line equation
  (∀ x y, a * x - b * y - 2 = 0 → True) →
  -- Curve equation
  (∀ x, f x = x^3 + x) →
  -- Point P
  f 1 = 2 →
  -- Perpendicular tangents at P
  (a / b) * (f' 1) = -1 →
  -- Conclusion
  a / b = -1/4 := by
  sorry

end perpendicular_tangents_ratio_l3514_351454


namespace football_season_games_l3514_351433

/-- Represents a football team's season statistics -/
structure SeasonStats where
  totalGames : ℕ
  tieGames : ℕ
  firstHundredWins : ℕ
  remainingWins : ℕ
  maxConsecutiveLosses : ℕ
  minConsecutiveWins : ℕ

/-- Calculates the total wins for a season -/
def totalWins (stats : SeasonStats) : ℕ :=
  stats.firstHundredWins + stats.remainingWins

/-- Calculates the win percentage for a season, excluding tie games -/
def winPercentage (stats : SeasonStats) : ℚ :=
  (totalWins stats : ℚ) / ((stats.totalGames - stats.tieGames) : ℚ)

/-- Theorem stating the total number of games played in the season -/
theorem football_season_games (stats : SeasonStats) 
  (h1 : 150 ≤ stats.totalGames ∧ stats.totalGames ≤ 200)
  (h2 : stats.firstHundredWins = 63)
  (h3 : stats.remainingWins = (stats.totalGames - 100) * 48 / 100)
  (h4 : stats.tieGames = 5)
  (h5 : winPercentage stats = 58 / 100)
  (h6 : stats.minConsecutiveWins ≥ 20)
  (h7 : stats.maxConsecutiveLosses ≤ 10) :
  stats.totalGames = 179 := by
  sorry


end football_season_games_l3514_351433


namespace license_plate_count_l3514_351430

/-- The number of consonants in the English alphabet -/
def num_consonants : ℕ := 20

/-- The number of even digits -/
def num_even_digits : ℕ := 5

/-- The number of possible license plates meeting the specified criteria -/
def num_license_plates : ℕ := num_consonants * 1 * num_consonants * num_even_digits

/-- Theorem stating that the number of license plates meeting the criteria is 2000 -/
theorem license_plate_count : num_license_plates = 2000 := by
  sorry

end license_plate_count_l3514_351430


namespace inequality_system_solution_l3514_351490

theorem inequality_system_solution (x : ℝ) : 
  (1 / x < 1 ∧ |4 * x - 1| > 2) ↔ (x < -1/4 ∨ x > 1) :=
by sorry

end inequality_system_solution_l3514_351490


namespace data_transmission_time_l3514_351442

theorem data_transmission_time (blocks : Nat) (chunks_per_block : Nat) (transmission_rate : Nat) :
  blocks = 50 →
  chunks_per_block = 1024 →
  transmission_rate = 100 →
  (blocks * chunks_per_block : ℚ) / transmission_rate / 60 = 9 := by
  sorry

end data_transmission_time_l3514_351442


namespace sodas_consumed_l3514_351467

def potluck_sodas (brought : ℕ) (taken_back : ℕ) : ℕ :=
  brought - taken_back

theorem sodas_consumed (brought : ℕ) (taken_back : ℕ) 
  (h : brought ≥ taken_back) : 
  potluck_sodas brought taken_back = brought - taken_back :=
by sorry

end sodas_consumed_l3514_351467


namespace initial_selling_price_theorem_l3514_351431

/-- The number of articles sold at a gain -/
def articles_sold_gain : ℝ := 20

/-- The gain percentage -/
def gain_percentage : ℝ := 0.20

/-- The number of articles that would be sold at a loss -/
def articles_sold_loss : ℝ := 29.99999625000047

/-- The loss percentage -/
def loss_percentage : ℝ := 0.20

/-- Theorem stating that the initial selling price for articles sold at a gain
    is 24 times the cost price of one article -/
theorem initial_selling_price_theorem (cost_price : ℝ) :
  let selling_price_gain := cost_price * (1 + gain_percentage)
  let selling_price_loss := cost_price * (1 - loss_percentage)
  articles_sold_gain * selling_price_gain = articles_sold_loss * selling_price_loss →
  articles_sold_gain * selling_price_gain = 24 * cost_price :=
by sorry

end initial_selling_price_theorem_l3514_351431


namespace characterization_of_special_numbers_l3514_351465

-- Define a structure for numbers of the form p^n - 1
structure PrimeExponentMinusOne where
  p : Nat
  n : Nat
  isPrime : Nat.Prime p

-- Define a predicate for numbers whose all divisors are of the form p^n - 1
def allDivisorsArePrimeExponentMinusOne (m : Nat) : Prop :=
  ∀ d : Nat, d ∣ m → ∃ (p n : Nat), Nat.Prime p ∧ d = p^n - 1

-- Main theorem
theorem characterization_of_special_numbers (m : Nat) 
  (h1 : ∃ (p n : Nat), Nat.Prime p ∧ m = p^n - 1)
  (h2 : allDivisorsArePrimeExponentMinusOne m) :
  (∃ k : Nat, m = 2^k - 1 ∧ Nat.Prime m) ∨ m ∣ 48 :=
sorry

end characterization_of_special_numbers_l3514_351465


namespace dataset_mode_l3514_351470

def dataset : List ℕ := [24, 23, 24, 25, 22]

def mode (l : List ℕ) : ℕ :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem dataset_mode : mode dataset = 24 := by
  sorry

end dataset_mode_l3514_351470


namespace chiefs_gold_l3514_351453

/-- A graph representing druids and their willingness to shake hands. -/
structure DruidGraph where
  /-- The set of vertices (druids) in the graph. -/
  V : Type
  /-- The edge relation, representing willingness to shake hands. -/
  E : V → V → Prop
  /-- The graph has no cycles of length 4 or more. -/
  no_long_cycles : ∀ (a b c d : V), E a b → E b c → E c d → E d a → (a = c ∨ b = d)

/-- The number of vertices in a DruidGraph. -/
def num_vertices (G : DruidGraph) : ℕ := sorry

/-- The number of edges in a DruidGraph. -/
def num_edges (G : DruidGraph) : ℕ := sorry

/-- 
The chief's gold theorem: In a DruidGraph, the chief can keep at least 3 gold coins.
This is equivalent to showing that 3n - 2e ≥ 3, where n is the number of vertices and e is the number of edges.
-/
theorem chiefs_gold (G : DruidGraph) : 
  3 * (num_vertices G) - 2 * (num_edges G) ≥ 3 := by sorry

end chiefs_gold_l3514_351453


namespace completing_square_equivalence_l3514_351435

theorem completing_square_equivalence (x : ℝ) : 
  (x^2 + 2*x - 5 = 0) ↔ ((x + 1)^2 = 6) :=
by sorry

end completing_square_equivalence_l3514_351435


namespace quadrilateral_diagonal_length_l3514_351425

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Calculates the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ := sorry

/-- Checks if a quadrilateral is convex -/
def isConvex (q : Quadrilateral) : Prop := sorry

/-- Finds the intersection point of two line segments -/
def lineIntersection (p1 p2 p3 p4 : Point) : Point := sorry

theorem quadrilateral_diagonal_length 
  (q : Quadrilateral) 
  (hConvex : isConvex q) 
  (hAB : distance q.A q.B = 8)
  (hCD : distance q.C q.D = 18)
  (hAC : distance q.A q.C = 20)
  (E : Point)
  (hE : E = lineIntersection q.A q.C q.B q.D)
  (hAreas : triangleArea q.A E q.D = triangleArea q.B E q.C) :
  distance q.A E = 80 / 13 := by sorry

end quadrilateral_diagonal_length_l3514_351425


namespace eleventh_number_with_digit_sum_13_l3514_351468

/-- A function that returns the sum of digits of a positive integer -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- A function that returns the nth positive integer whose digits sum to 13 -/
def nthNumberWithDigitSum13 (n : ℕ) : ℕ := sorry

/-- Theorem stating that the 11th number with digit sum 13 is 175 -/
theorem eleventh_number_with_digit_sum_13 : nthNumberWithDigitSum13 11 = 175 := by sorry

end eleventh_number_with_digit_sum_13_l3514_351468


namespace trapezium_height_l3514_351434

theorem trapezium_height (a b area : ℝ) (ha : a = 20) (hb : b = 16) (harea : area = 270) :
  (2 * area) / (a + b) = 15 := by
  sorry

end trapezium_height_l3514_351434


namespace sum_of_tenth_powers_l3514_351483

theorem sum_of_tenth_powers (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^10 + b^10 = 123 := by
  sorry

end sum_of_tenth_powers_l3514_351483


namespace complex_arithmetic_expression_evaluation_l3514_351413

theorem complex_arithmetic_expression_evaluation : 
  let a := (8/7 - 23/49) / (22/147)
  let b := (0.6 / (15/4)) * (5/2)
  let c := 3.75 / (3/2)
  ((a - b + c) / 2.2) = 3 := by sorry

end complex_arithmetic_expression_evaluation_l3514_351413


namespace max_c_value_l3514_351414

theorem max_c_value (c d : ℝ) (h : 5 * c + (d - 12)^2 = 235) :
  c ≤ 47 ∧ ∃ d₀, 5 * 47 + (d₀ - 12)^2 = 235 := by
  sorry

end max_c_value_l3514_351414


namespace number_representation_and_addition_l3514_351499

theorem number_representation_and_addition :
  (4090000 = 409 * 10000) ∧ (800000 + 5000 + 20 + 4 = 805024) := by
  sorry

end number_representation_and_addition_l3514_351499


namespace second_car_speed_l3514_351484

theorem second_car_speed 
  (highway_length : ℝ) 
  (first_car_speed : ℝ) 
  (meeting_time : ℝ) 
  (h1 : highway_length = 175) 
  (h2 : first_car_speed = 25) 
  (h3 : meeting_time = 2.5) : 
  ∃ second_car_speed : ℝ, 
    first_car_speed * meeting_time + second_car_speed * meeting_time = highway_length ∧ 
    second_car_speed = 45 := by
sorry

end second_car_speed_l3514_351484


namespace age_problem_solution_l3514_351461

/-- Represents the age relationship between a father and daughter -/
structure AgeProblem where
  daughter_age : ℕ
  father_age : ℕ
  years_ago : ℕ
  years_future : ℕ

/-- The conditions of the problem -/
def problem_conditions (p : AgeProblem) : Prop :=
  p.father_age = 3 * p.daughter_age ∧
  (p.father_age - p.years_ago) = 5 * (p.daughter_age - p.years_ago)

/-- The future condition we want to prove -/
def future_condition (p : AgeProblem) : Prop :=
  (p.father_age + p.years_future) = 2 * (p.daughter_age + p.years_future)

/-- The theorem to prove -/
theorem age_problem_solution :
  ∀ p : AgeProblem,
    problem_conditions p →
    (p.years_future = 14 ↔ future_condition p) :=
by
  sorry


end age_problem_solution_l3514_351461


namespace amusement_park_groups_l3514_351403

theorem amusement_park_groups (n : ℕ) (k : ℕ) : n = 7 ∧ k = 4 → Nat.choose n k = 35 := by
  sorry

end amusement_park_groups_l3514_351403


namespace no_three_naturals_sum_power_of_three_l3514_351463

theorem no_three_naturals_sum_power_of_three :
  ¬ ∃ (a b c : ℕ), 
    (∃ k : ℕ, a + b = 3^k) ∧
    (∃ m : ℕ, b + c = 3^m) ∧
    (∃ n : ℕ, c + a = 3^n) :=
sorry

end no_three_naturals_sum_power_of_three_l3514_351463


namespace least_value_f_1998_l3514_351422

/-- A function from positive integers to positive integers satisfying the given property -/
def FunctionF :=
  {f : ℕ+ → ℕ+ | ∀ m n : ℕ+, f (n^2 * f m) = m * (f n)^2}

/-- The theorem stating the least possible value of f(1998) -/
theorem least_value_f_1998 :
  (∃ f ∈ FunctionF, f 1998 = 120) ∧
  (∀ f ∈ FunctionF, f 1998 ≥ 120) :=
sorry

end least_value_f_1998_l3514_351422


namespace semesters_per_year_l3514_351450

/-- Given the cost per semester and total cost for 13 years of school,
    prove that there are 2 semesters in a year. -/
theorem semesters_per_year :
  let cost_per_semester : ℕ := 20000
  let total_cost : ℕ := 520000
  let years : ℕ := 13
  let total_semesters : ℕ := total_cost / cost_per_semester
  total_semesters / years = 2 := by
  sorry

end semesters_per_year_l3514_351450


namespace total_cost_in_dollars_l3514_351420

/-- The cost of a single pencil in cents -/
def pencil_cost : ℚ := 2

/-- The cost of a single eraser in cents -/
def eraser_cost : ℚ := 5

/-- The number of pencils to be purchased -/
def num_pencils : ℕ := 500

/-- The number of erasers to be purchased -/
def num_erasers : ℕ := 250

/-- The conversion rate from cents to dollars -/
def cents_to_dollars : ℚ := 1 / 100

theorem total_cost_in_dollars : 
  (pencil_cost * num_pencils + eraser_cost * num_erasers) * cents_to_dollars = 22.5 := by
  sorry

end total_cost_in_dollars_l3514_351420


namespace eighth_term_is_six_l3514_351475

/-- An arithmetic progression with given conditions -/
structure ArithmeticProgression where
  a : ℕ → ℚ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1
  first_term : a 1 = 2
  sum_condition : a 3 + a 6 = 8

/-- The 8th term of the arithmetic progression is 6 -/
theorem eighth_term_is_six (ap : ArithmeticProgression) : ap.a 8 = 6 := by
  sorry

end eighth_term_is_six_l3514_351475


namespace parabola_equation_l3514_351452

-- Define the parabola C
def Parabola : Type := ℝ → ℝ → Prop

-- Define the line x - y = 0
def Line (x y : ℝ) : Prop := x - y = 0

-- Define a point on the 2D plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the midpoint of two points
def Midpoint (A B P : Point) : Prop :=
  P.x = (A.x + B.x) / 2 ∧ P.y = (A.y + B.y) / 2

-- State the theorem
theorem parabola_equation (C : Parabola) (A B P : Point) :
  -- The vertex of parabola C is at the origin
  C 0 0 →
  -- The focus of parabola C is on the x-axis (we don't need to specify the exact location)
  ∃ f : ℝ, C f 0 →
  -- The line x - y = 0 intersects parabola C at points A and B
  Line A.x A.y ∧ C A.x A.y ∧ Line B.x B.y ∧ C B.x B.y →
  -- P(1,1) is the midpoint of segment AB
  P.x = 1 ∧ P.y = 1 ∧ Midpoint A B P →
  -- The equation of parabola C is x^2 = 2y
  ∀ x y : ℝ, C x y ↔ x^2 = 2*y :=
by sorry

end parabola_equation_l3514_351452


namespace twice_slope_line_equation_l3514_351421

/-- Given a line L1: 2x + 3y + 3 = 0, prove that the line L2 passing through (1,0) 
    with a slope twice that of L1 has the equation 4x + 3y = 4. -/
theorem twice_slope_line_equation : 
  let L1 : ℝ → ℝ → Prop := λ x y ↦ 2 * x + 3 * y + 3 = 0
  let m1 : ℝ := -2 / 3  -- slope of L1
  let m2 : ℝ := 2 * m1  -- slope of L2
  let L2 : ℝ → ℝ → Prop := λ x y ↦ 4 * x + 3 * y = 4
  (∀ x y, L2 x y ↔ y - 0 = m2 * (x - 1)) ∧ L2 1 0 := by
  sorry


end twice_slope_line_equation_l3514_351421


namespace square_area_ratio_l3514_351481

theorem square_area_ratio (s₁ s₂ : ℝ) (h : s₁ = 2 * s₂ * Real.sqrt 2) :
  s₁^2 / s₂^2 = 8 := by
  sorry

end square_area_ratio_l3514_351481


namespace two_tower_100_gt_3_three_tower_100_gt_three_tower_99_three_tower_100_gt_four_tower_99_l3514_351471

-- Define a function to represent the power tower
def powerTower (base : ℕ) (height : ℕ) : ℕ :=
  match height with
  | 0 => 1
  | n + 1 => base ^ (powerTower base n)

-- Theorem 1
theorem two_tower_100_gt_3 : powerTower 2 100 > 3 := by sorry

-- Theorem 2
theorem three_tower_100_gt_three_tower_99 : powerTower 3 100 > powerTower 3 99 := by sorry

-- Theorem 3
theorem three_tower_100_gt_four_tower_99 : powerTower 3 100 > powerTower 4 99 := by sorry

end two_tower_100_gt_3_three_tower_100_gt_three_tower_99_three_tower_100_gt_four_tower_99_l3514_351471


namespace no_solution_to_diophantine_equation_l3514_351495

theorem no_solution_to_diophantine_equation :
  ¬ ∃ (x y z t : ℕ), 3 * x^4 + 5 * y^4 + 7 * z^4 = 11 * t^4 := by
  sorry

end no_solution_to_diophantine_equation_l3514_351495


namespace student_rank_problem_l3514_351494

/-- Given a total number of students and a student's rank from the right,
    calculates the student's rank from the left. -/
def rank_from_left (total : ℕ) (rank_from_right : ℕ) : ℕ :=
  total - rank_from_right + 1

/-- Proves that for 21 total students and a student ranked 16th from the right,
    the student's rank from the left is 6. -/
theorem student_rank_problem :
  rank_from_left 21 16 = 6 := by
  sorry

#eval rank_from_left 21 16

end student_rank_problem_l3514_351494


namespace fraction_sum_equality_l3514_351489

theorem fraction_sum_equality (a b c : ℝ) 
  (h : a / (36 - a) + b / (49 - b) + c / (81 - c) = 9) :
  6 / (36 - a) + 7 / (49 - b) + 9 / (81 - c) = 5047 / 1000 := by
  sorry

end fraction_sum_equality_l3514_351489


namespace lagoon_island_male_alligators_l3514_351479

/-- Represents the population of alligators on Lagoon island -/
structure AlligatorPopulation where
  total : ℕ
  males : ℕ
  females : ℕ
  juvenileFemales : ℕ
  adultFemales : ℕ

/-- Conditions for the Lagoon island alligator population -/
def lagoonIslandConditions (pop : AlligatorPopulation) : Prop :=
  pop.males = pop.females ∧
  pop.females = pop.juvenileFemales + pop.adultFemales ∧
  pop.juvenileFemales = (2 * pop.females) / 5 ∧
  pop.adultFemales = 15

theorem lagoon_island_male_alligators (pop : AlligatorPopulation) 
  (h : lagoonIslandConditions pop) : 
  pop.males = pop.adultFemales / (3 : ℚ) / (10 : ℚ) := by
  sorry

#check lagoon_island_male_alligators

end lagoon_island_male_alligators_l3514_351479


namespace outfit_combinations_l3514_351458

theorem outfit_combinations : 
  let total_items : ℕ := 3  -- shirts, pants, hats
  let colors_per_item : ℕ := 5
  let total_combinations := colors_per_item ^ total_items
  let same_color_combinations := 
    (total_items * colors_per_item * (colors_per_item - 1)) + colors_per_item
  total_combinations - same_color_combinations = 60 :=
by sorry

end outfit_combinations_l3514_351458


namespace log_equation_implies_m_value_l3514_351437

theorem log_equation_implies_m_value 
  (m n : ℝ) (c : ℝ) 
  (h : Real.log (m^2) = c - 2 * Real.log n) :
  m = Real.sqrt (Real.exp c / n) :=
by sorry

end log_equation_implies_m_value_l3514_351437


namespace four_numbers_product_sum_prime_l3514_351428

theorem four_numbers_product_sum_prime :
  ∃ (a b c d : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    Nat.Prime (a * b + c * d) ∧
    Nat.Prime (a * c + b * d) ∧
    Nat.Prime (a * d + b * c) := by
  sorry

end four_numbers_product_sum_prime_l3514_351428


namespace same_solution_equations_l3514_351485

theorem same_solution_equations (b : ℚ) : 
  (∃ x : ℚ, 3 * x + 9 = 0 ∧ 2 * b * x - 15 = -5) → b = -5/3 := by
  sorry

end same_solution_equations_l3514_351485


namespace rancher_problem_solution_l3514_351462

/-- Represents the rancher's cattle problem -/
structure CattleProblem where
  initial_cattle : ℕ
  dead_cattle : ℕ
  price_reduction : ℚ
  loss_amount : ℚ

/-- Calculates the original price per head of cattle -/
def original_price (p : CattleProblem) : ℚ :=
  p.loss_amount / (p.initial_cattle - p.dead_cattle : ℚ)

/-- Calculates the total amount the rancher would have made -/
def total_amount (p : CattleProblem) : ℚ :=
  (p.initial_cattle : ℚ) * original_price p

/-- Theorem stating the solution to the rancher's problem -/
theorem rancher_problem_solution (p : CattleProblem) 
  (h1 : p.initial_cattle = 340)
  (h2 : p.dead_cattle = 172)
  (h3 : p.price_reduction = 150)
  (h4 : p.loss_amount = 25200) :
  total_amount p = 49813.40 := by
  sorry

end rancher_problem_solution_l3514_351462


namespace average_difference_l3514_351469

theorem average_difference (a b c : ℝ) 
  (h1 : (a + b) / 2 = 35) 
  (h2 : (b + c) / 2 = 80) : 
  c - a = 90 := by
sorry

end average_difference_l3514_351469


namespace justin_tim_same_game_l3514_351408

/-- The number of players in the league -/
def total_players : ℕ := 12

/-- The number of players in each game -/
def players_per_game : ℕ := 6

/-- The number of players to be selected (excluding Justin and Tim) -/
def players_to_select : ℕ := players_per_game - 2

/-- The number of remaining players after excluding Justin and Tim -/
def remaining_players : ℕ := total_players - 2

/-- The number of times Justin and Tim play in the same game -/
def same_game_count : ℕ := Nat.choose remaining_players players_to_select

theorem justin_tim_same_game :
  same_game_count = 210 := by sorry

end justin_tim_same_game_l3514_351408


namespace border_area_l3514_351497

/-- The area of the border around a rectangular painting -/
theorem border_area (height width border_width : ℕ) : 
  height = 12 → width = 16 → border_width = 3 →
  (height + 2 * border_width) * (width + 2 * border_width) - height * width = 204 := by
  sorry

end border_area_l3514_351497


namespace unique_function_satisfying_conditions_l3514_351488

theorem unique_function_satisfying_conditions
  (f : ℕ → ℕ → ℕ)
  (h1 : ∀ a b c : ℕ, f (Nat.gcd a b) c = Nat.gcd a (f c b))
  (h2 : ∀ a : ℕ, f a a ≥ a) :
  ∀ a b : ℕ, f a b = Nat.gcd a b :=
by sorry

end unique_function_satisfying_conditions_l3514_351488


namespace sam_gave_thirteen_cards_l3514_351415

/-- The number of baseball cards Sam gave to Mike -/
def cards_from_sam (initial_cards final_cards : ℕ) : ℕ :=
  final_cards - initial_cards

theorem sam_gave_thirteen_cards :
  let initial_cards : ℕ := 87
  let final_cards : ℕ := 100
  cards_from_sam initial_cards final_cards = 13 := by
  sorry

end sam_gave_thirteen_cards_l3514_351415


namespace area_ratio_of_triangles_l3514_351448

/-- Given two triangles PQR and XYZ with known base and height measurements,
    prove that the area of PQR is 1/3 of the area of XYZ. -/
theorem area_ratio_of_triangles (base_PQR height_PQR base_XYZ height_XYZ : ℝ)
  (h1 : base_PQR = 3)
  (h2 : height_PQR = 2)
  (h3 : base_XYZ = 6)
  (h4 : height_XYZ = 3) :
  (1 / 2 * base_PQR * height_PQR) / (1 / 2 * base_XYZ * height_XYZ) = 1 / 3 := by
  sorry

end area_ratio_of_triangles_l3514_351448


namespace students_only_english_l3514_351460

theorem students_only_english (total : ℕ) (both : ℕ) (german : ℕ) 
  (h1 : total = 32)
  (h2 : both = 12)
  (h3 : german = 22)
  (h4 : total = (german - both) + both + (total - german)) :
  total - german = 10 := by
  sorry

end students_only_english_l3514_351460


namespace equation_root_implies_m_values_l3514_351418

theorem equation_root_implies_m_values (m : ℝ) : 
  (∃ x : ℝ, x^2 + 2*m*x + m^2 - 1 = 0) ∧ 
  (3^2 + 2*m*3 + m^2 - 1 = 0) →
  m = -2 ∨ m = -4 := by
sorry

end equation_root_implies_m_values_l3514_351418


namespace apple_sales_theorem_l3514_351446

/-- Calculate the total money earned from selling apples from a rectangular plot of trees -/
def apple_sales_revenue (rows : ℕ) (cols : ℕ) (apples_per_tree : ℕ) (price_per_apple : ℚ) : ℚ :=
  (rows * cols * apples_per_tree : ℕ) * price_per_apple

/-- Theorem: The total money earned from selling apples from a 3x4 plot of trees,
    where each tree produces 5 apples and each apple is sold for $0.5, is equal to $30 -/
theorem apple_sales_theorem :
  apple_sales_revenue 3 4 5 (1/2) = 30 := by
  sorry

end apple_sales_theorem_l3514_351446


namespace donation_change_l3514_351451

def original_donations : List ℕ := [5, 3, 6, 5, 10]

def median (l : List ℕ) : ℕ := sorry
def mode (l : List ℕ) : List ℕ := sorry

def new_donations (a : ℕ) : List ℕ :=
  let index := original_donations.indexOf 3
  original_donations.set index (3 + a)

theorem donation_change (a : ℕ) :
  (median (new_donations a) = median original_donations ∧
   mode (new_donations a) = mode original_donations) ↔
  (a = 1 ∨ a = 2) := by sorry

end donation_change_l3514_351451


namespace square_sum_given_linear_and_product_l3514_351411

theorem square_sum_given_linear_and_product (x y : ℝ) 
  (h1 : x + 2*y = 6) (h2 : x*y = -12) : x^2 + 4*y^2 = 84 := by
  sorry

end square_sum_given_linear_and_product_l3514_351411


namespace CCl4_formation_l3514_351457

-- Define the initial amounts of reactants
def initial_C2H6 : ℝ := 2
def initial_Cl2 : ℝ := 14

-- Define the stoichiometric ratio for each step
def stoichiometric_ratio : ℝ := 1

-- Define the number of reaction steps
def num_steps : ℕ := 4

-- Theorem statement
theorem CCl4_formation (remaining_Cl2 : ℝ → ℝ) 
  (h1 : remaining_Cl2 0 = initial_Cl2)
  (h2 : ∀ n : ℕ, n < num_steps → 
    remaining_Cl2 (n + 1) = remaining_Cl2 n - stoichiometric_ratio * initial_C2H6)
  (h3 : ∀ n : ℕ, n ≤ num_steps → remaining_Cl2 n ≥ 0) :
  remaining_Cl2 num_steps = initial_Cl2 - num_steps * stoichiometric_ratio * initial_C2H6 ∧
  initial_C2H6 = initial_C2H6 :=
by sorry

end CCl4_formation_l3514_351457


namespace radical_product_simplification_l3514_351486

theorem radical_product_simplification (q : ℝ) (hq : q ≥ 0) :
  Real.sqrt (50 * q) * Real.sqrt (10 * q) * Real.sqrt (15 * q) = 50 * q * Real.sqrt q :=
by sorry

end radical_product_simplification_l3514_351486


namespace david_average_marks_l3514_351493

def david_marks : List ℕ := [76, 65, 82, 67, 85]

theorem david_average_marks :
  (david_marks.sum / david_marks.length : ℚ) = 75 := by sorry

end david_average_marks_l3514_351493


namespace inverse_g_at_43_16_l3514_351464

/-- Given a function g(x) = (x^3 - 5) / 4, prove that g⁻¹(43/16) = 3 * ∛7 / 2 -/
theorem inverse_g_at_43_16 (g : ℝ → ℝ) (h : ∀ x, g x = (x^3 - 5) / 4) :
  g⁻¹ (43/16) = 3 * Real.rpow 7 (1/3) / 2 := by
  sorry

end inverse_g_at_43_16_l3514_351464


namespace alcohol_percentage_after_dilution_l3514_351412

/-- Given a mixture of water and alcohol, calculate the new alcohol percentage after adding water. -/
theorem alcohol_percentage_after_dilution
  (initial_volume : ℝ)
  (initial_percentage : ℝ)
  (added_water : ℝ)
  (h1 : initial_volume = 15)
  (h2 : initial_percentage = 20)
  (h3 : added_water = 5)
  : (initial_volume * initial_percentage / 100) / (initial_volume + added_water) * 100 = 15 := by
  sorry

#check alcohol_percentage_after_dilution

end alcohol_percentage_after_dilution_l3514_351412


namespace current_age_ratio_l3514_351424

def age_ratio (p q : ℕ) : ℚ := p / q

theorem current_age_ratio :
  ∀ (p q : ℕ),
  (∃ k : ℕ, p = k * q) →
  (p + 11 = 2 * (q + 11)) →
  (p = 30 + 3) →
  age_ratio p q = 3 / 1 :=
by
  sorry

end current_age_ratio_l3514_351424


namespace probability_of_selecting_one_each_l3514_351444

/-- The probability of selecting one shirt, one pair of shorts, and one pair of socks
    when randomly choosing three items from a drawer containing 4 shirts, 5 pairs of shorts,
    and 6 pairs of socks. -/
theorem probability_of_selecting_one_each (num_shirts : ℕ) (num_shorts : ℕ) (num_socks : ℕ) :
  num_shirts = 4 →
  num_shorts = 5 →
  num_socks = 6 →
  (num_shirts * num_shorts * num_socks : ℚ) / (Nat.choose (num_shirts + num_shorts + num_socks) 3) = 24 / 91 := by
  sorry

#check probability_of_selecting_one_each

end probability_of_selecting_one_each_l3514_351444


namespace ratio_to_twelve_l3514_351472

theorem ratio_to_twelve : ∃ x : ℝ, (5 : ℝ) / 1 = x / 12 → x = 60 :=
by sorry

end ratio_to_twelve_l3514_351472


namespace algebraic_simplification_l3514_351477

theorem algebraic_simplification (a b c : ℝ) :
  -32 * a^4 * b^5 * c / (-2 * a * b)^3 * (-3/4 * a * c) = -3 * a^2 * b^2 * c^2 := by
  sorry

end algebraic_simplification_l3514_351477


namespace jesse_carpet_problem_l3514_351498

/-- Given a room with length and width, and some carpet already available,
    calculate the additional carpet needed to cover the whole floor. -/
def additional_carpet_needed (length width available_carpet : ℝ) : ℝ :=
  length * width - available_carpet

/-- Theorem: Given a room that is 4 feet long and 20 feet wide, with 18 square feet
    of carpet already available, the additional carpet needed is 62 square feet. -/
theorem jesse_carpet_problem :
  additional_carpet_needed 4 20 18 = 62 := by
  sorry

end jesse_carpet_problem_l3514_351498


namespace remainder_equality_l3514_351487

theorem remainder_equality (A A' D S S' s s' : ℕ) 
  (h1 : A > A')
  (h2 : S = A % D)
  (h3 : S' = A' % D)
  (h4 : s = (A + A') % D)
  (h5 : s' = (S + S') % D) :
  s = s' :=
sorry

end remainder_equality_l3514_351487


namespace gcd_triple_existence_l3514_351473

theorem gcd_triple_existence (S : Set ℕ+) 
  (h_infinite : Set.Infinite S)
  (h_distinct_gcd : ∃ (a b c d : ℕ+), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
    Nat.gcd a.val b.val ≠ Nat.gcd c.val d.val) :
  ∃ (x y z : ℕ+), x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ 
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    Nat.gcd x.val y.val = Nat.gcd y.val z.val ∧ 
    Nat.gcd y.val z.val ≠ Nat.gcd z.val x.val :=
sorry

end gcd_triple_existence_l3514_351473


namespace monomial_degree_equality_l3514_351429

-- Define the degree of a monomial
def degree (x y z : ℕ) (m : ℕ) : ℕ := x + y

-- Define the theorem
theorem monomial_degree_equality (m : ℕ) :
  degree 2 4 0 0 = degree 0 1 (m + 2) m →
  3 * m - 2 = 7 := by sorry

end monomial_degree_equality_l3514_351429


namespace parabola_directrix_l3514_351441

-- Define the parabola
def parabola (x y : ℝ) : Prop := y = 16 * x^2

-- Define the directrix
def directrix (y : ℝ) : Prop := y = -1/64

-- Theorem statement
theorem parabola_directrix :
  ∀ x y : ℝ, parabola x y → ∃ d : ℝ, directrix d ∧ d = -1/64 :=
by sorry

end parabola_directrix_l3514_351441


namespace total_clam_shells_is_43_l3514_351400

/-- The number of clam shells found by Sam, Mary, and Lucy -/
def clam_shells (name : String) : ℕ :=
  match name with
  | "Sam" => 8
  | "Mary" => 20
  | "Lucy" => 15
  | _ => 0

/-- The total number of clam shells found by Sam, Mary, and Lucy -/
def total_clam_shells : ℕ :=
  clam_shells "Sam" + clam_shells "Mary" + clam_shells "Lucy"

/-- Theorem stating that the total number of clam shells found is 43 -/
theorem total_clam_shells_is_43 : total_clam_shells = 43 := by
  sorry

end total_clam_shells_is_43_l3514_351400


namespace complex_fraction_equality_l3514_351447

theorem complex_fraction_equality : ∃ (i : ℂ), i^2 = -1 ∧ (1 - 2*i) / (2 + i) = -i := by sorry

end complex_fraction_equality_l3514_351447


namespace linear_function_quadrants_l3514_351436

/-- A linear function passing through the first, third, and fourth quadrants -/
def passes_through_134_quadrants (k : ℝ) : Prop :=
  (k - 3 > 0) ∧ (-k + 2 < 0)

/-- Theorem stating that if a linear function y=(k-3)x-k+2 passes through
    the first, third, and fourth quadrants, then k > 3 -/
theorem linear_function_quadrants (k : ℝ) :
  passes_through_134_quadrants k → k > 3 := by
  sorry

end linear_function_quadrants_l3514_351436


namespace parallel_vectors_t_value_l3514_351474

theorem parallel_vectors_t_value (t : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![-2, t]
  (∃ (k : ℝ), k ≠ 0 ∧ (∀ i, b i = k * a i)) → t = -4 := by
  sorry

end parallel_vectors_t_value_l3514_351474


namespace regions_in_circle_l3514_351419

/-- The number of regions created by radii and concentric circles in a larger circle -/
def num_regions (num_radii : ℕ) (num_circles : ℕ) : ℕ :=
  (num_circles + 1) * num_radii

/-- Theorem: 16 radii and 10 concentric circles create 176 regions -/
theorem regions_in_circle (num_radii num_circles : ℕ) 
  (h1 : num_radii = 16) 
  (h2 : num_circles = 10) : 
  num_regions num_radii num_circles = 176 := by
  sorry

#eval num_regions 16 10

end regions_in_circle_l3514_351419


namespace circle_center_distance_to_line_l3514_351438

theorem circle_center_distance_to_line : ∃ (center : ℝ × ℝ),
  (∀ (x y : ℝ), x^2 + 2*x + y^2 = 0 ↔ (x - center.1)^2 + (y - center.2)^2 = 1) ∧
  |center.1 - 3| = 4 := by
  sorry

end circle_center_distance_to_line_l3514_351438


namespace sqrt_function_theorem_linear_function_theorem_l3514_351423

-- Problem 1
theorem sqrt_function_theorem (f : ℝ → ℝ) :
  (∀ x ≥ 0, f (Real.sqrt x) = x - 1) →
  (∀ x ≥ 0, f x = x^2 - 1) :=
by sorry

-- Problem 2
theorem linear_function_theorem (f : ℝ → ℝ) :
  (∃ a b : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x + b) →
  (∀ x, f (f x) = f x + 2) →
  (∀ x, f x = x + 2) :=
by sorry

end sqrt_function_theorem_linear_function_theorem_l3514_351423


namespace chemical_mixture_percentage_l3514_351401

theorem chemical_mixture_percentage (initial_volume : ℝ) (initial_percentage : ℝ) (added_volume : ℝ) :
  initial_volume = 80 →
  initial_percentage = 0.3 →
  added_volume = 20 →
  let final_volume := initial_volume + added_volume
  let initial_x_volume := initial_volume * initial_percentage
  let final_x_volume := initial_x_volume + added_volume
  final_x_volume / final_volume = 0.44 := by
  sorry

end chemical_mixture_percentage_l3514_351401


namespace sum_of_squared_coefficients_l3514_351432

theorem sum_of_squared_coefficients : 
  let expression := fun x : ℝ => 5 * (x^3 - x) - 3 * (x^2 - 4*x + 3)
  let simplified := fun x : ℝ => 5*x^3 - 3*x^2 + 7*x - 9
  (∀ x : ℝ, expression x = simplified x) →
  (5^2 + (-3)^2 + 7^2 + (-9)^2 = 164) := by
  sorry

end sum_of_squared_coefficients_l3514_351432


namespace inequality_range_l3514_351409

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, |2*x + 1| - |2*x - 1| < a) → a > 2 :=
by sorry

end inequality_range_l3514_351409


namespace remaining_eggs_l3514_351476

theorem remaining_eggs (initial_eggs : ℕ) (morning_eaten : ℕ) (afternoon_eaten : ℕ) 
  (h1 : initial_eggs = 20)
  (h2 : morning_eaten = 4)
  (h3 : afternoon_eaten = 3) :
  initial_eggs - (morning_eaten + afternoon_eaten) = 13 := by
  sorry

end remaining_eggs_l3514_351476


namespace walking_speed_l3514_351416

/-- Given a constant walking speed, prove that traveling 30 km in 6 hours results in a speed of 5 kmph -/
theorem walking_speed (distance : ℝ) (time : ℝ) (speed : ℝ) 
    (h1 : distance = 30) 
    (h2 : time = 6) 
    (h3 : speed = distance / time) : speed = 5 := by
  sorry

end walking_speed_l3514_351416


namespace star_product_scaling_l3514_351455

/-- Given that 2994 ã · 14.5 = 179, prove that 29.94 ã · 1.45 = 0.179 -/
theorem star_product_scaling (h : 2994 * 14.5 = 179) : 29.94 * 1.45 = 0.179 := by
  sorry

end star_product_scaling_l3514_351455


namespace line_intersects_and_passes_through_point_l3514_351427

-- Define the line l
def line_l (m x y : ℝ) : Prop := (m + 1) * x + 2 * y + 2 * m - 2 = 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*y - 8 = 0

-- Theorem statement
theorem line_intersects_and_passes_through_point :
  ∀ m : ℝ,
  (∃ x y : ℝ, line_l m x y ∧ circle_C x y) ∧
  (line_l m (-2) 2) :=
by sorry

end line_intersects_and_passes_through_point_l3514_351427


namespace regression_lines_intersection_l3514_351404

/-- A regression line in 2D space -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- The point where a regression line passes through -/
def RegressionLine.point_on_line (l : RegressionLine) (x : ℝ) : ℝ × ℝ :=
  (x, l.slope * x + l.intercept)

theorem regression_lines_intersection
  (l₁ l₂ : RegressionLine)
  (s t : ℝ)
  (h₁ : (s, t) = l₁.point_on_line s)
  (h₂ : (s, t) = l₂.point_on_line s) :
  ∃ (x y : ℝ), l₁.point_on_line x = (x, y) ∧ l₂.point_on_line x = (x, y) ∧ x = s ∧ y = t :=
sorry

end regression_lines_intersection_l3514_351404


namespace sum_range_l3514_351440

theorem sum_range : ∃ (x : ℚ), 10.5 < x ∧ x < 11 ∧ x = 2 + 1/8 + 3 + 1/3 + 5 + 1/18 := by
  sorry

end sum_range_l3514_351440


namespace equation_solution_l3514_351496

theorem equation_solution : ∃ x : ℝ, (2 / (x - 4) + 3 = (x - 2) / (4 - x)) ∧ x = 3 := by
  sorry

end equation_solution_l3514_351496


namespace last_locker_is_2046_l3514_351480

/-- Represents the state of a locker (open or closed) -/
inductive LockerState
  | Open
  | Closed

/-- Represents the corridor of lockers -/
def Corridor := Fin 2048 → LockerState

/-- Represents the student's locker opening strategy -/
def OpeningStrategy := Corridor → Nat → Nat

/-- The final locker opened by the student -/
def lastOpenedLocker (strategy : OpeningStrategy) : Nat :=
  2046

/-- The theorem stating that the last opened locker is 2046 -/
theorem last_locker_is_2046 (strategy : OpeningStrategy) :
  lastOpenedLocker strategy = 2046 := by
  sorry

#check last_locker_is_2046

end last_locker_is_2046_l3514_351480


namespace inverse_function_equality_f_equals_f_inverse_l3514_351456

def f (x : ℝ) : ℝ := 4 * x - 5

theorem inverse_function_equality (f : ℝ → ℝ) (h : Function.Bijective f) :
  ∃ x : ℝ, f x = Function.invFun f x :=
by
  sorry

theorem f_equals_f_inverse :
  ∃ x : ℝ, f x = Function.invFun f x ∧ x = 5/3 :=
by
  sorry

end inverse_function_equality_f_equals_f_inverse_l3514_351456


namespace actor_stage_time_l3514_351402

theorem actor_stage_time (actors_at_once : ℕ) (total_actors : ℕ) (show_duration : ℕ) : 
  actors_at_once = 5 → total_actors = 20 → show_duration = 60 → 
  (show_duration / (total_actors / actors_at_once) : ℚ) = 15 := by
  sorry

end actor_stage_time_l3514_351402


namespace turquoise_survey_result_l3514_351410

/-- Represents the survey about turquoise color perception -/
structure TurquoiseSurvey where
  total : ℕ
  blue : ℕ
  both : ℕ
  neither : ℕ

/-- The number of people who believe turquoise is "green-ish" -/
def green_count (s : TurquoiseSurvey) : ℕ :=
  s.total - (s.blue - s.both) - s.both - s.neither

/-- Theorem stating the result of the survey -/
theorem turquoise_survey_result (s : TurquoiseSurvey) 
  (h1 : s.total = 150)
  (h2 : s.blue = 90)
  (h3 : s.both = 40)
  (h4 : s.neither = 30) :
  green_count s = 70 := by
  sorry

#eval green_count ⟨150, 90, 40, 30⟩

end turquoise_survey_result_l3514_351410


namespace correct_pairings_l3514_351443

/-- The number of possible pairings for the first round of a tennis tournament with 2n players -/
def numPairings (n : ℕ) : ℚ :=
  (Nat.factorial (2 * n)) / ((2 ^ n) * Nat.factorial n)

/-- Theorem stating that numPairings gives the correct number of possible pairings -/
theorem correct_pairings (n : ℕ) :
  numPairings n = (Nat.factorial (2 * n)) / ((2 ^ n) * Nat.factorial n) := by
  sorry

end correct_pairings_l3514_351443


namespace f_equals_g_l3514_351482

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - 1
def g (t : ℝ) : ℝ := t^2 - 1

-- Theorem stating that f and g are the same function
theorem f_equals_g : ∀ x : ℝ, f x = g x := by sorry

end f_equals_g_l3514_351482


namespace simplify_expression_l3514_351439

theorem simplify_expression (a b : ℝ) : (-a^2 * b^3)^3 = -a^6 * b^9 := by
  sorry

end simplify_expression_l3514_351439


namespace danicas_car_arrangement_l3514_351417

/-- The number of cars Danica currently has -/
def current_cars : ℕ := 29

/-- The number of cars required in each row -/
def cars_per_row : ℕ := 8

/-- The function to calculate the number of additional cars needed -/
def additional_cars_needed (current : ℕ) (per_row : ℕ) : ℕ :=
  (per_row - (current % per_row)) % per_row

theorem danicas_car_arrangement :
  additional_cars_needed current_cars cars_per_row = 3 := by
  sorry

end danicas_car_arrangement_l3514_351417


namespace complex_number_magnitude_l3514_351406

theorem complex_number_magnitude (z : ℂ) : z = 2 / (1 - Complex.I) + Complex.I → Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_number_magnitude_l3514_351406


namespace binary_1011001_to_base6_l3514_351407

/-- Converts a binary (base-2) number to its decimal (base-10) representation -/
def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

/-- Converts a decimal (base-10) number to its base-6 representation -/
def decimal_to_base6 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
    aux n []

/-- The binary representation of 1011001 -/
def binary_1011001 : List Bool := [true, false, false, true, true, false, true]

theorem binary_1011001_to_base6 :
  decimal_to_base6 (binary_to_decimal binary_1011001) = [2, 2, 5] :=
sorry

end binary_1011001_to_base6_l3514_351407


namespace usual_time_is_eight_l3514_351491

-- Define the usual speed and time
variable (S : ℝ) -- Usual speed
variable (T : ℝ) -- Usual time

-- Define the theorem
theorem usual_time_is_eight
  (h1 : S > 0) -- Assume speed is positive
  (h2 : T > 0) -- Assume time is positive
  (h3 : S / (0.25 * S) = (T + 24) / T) -- Equation from the problem
  : T = 8 := by
sorry


end usual_time_is_eight_l3514_351491


namespace divisor_problem_l3514_351478

theorem divisor_problem (x : ℕ) : x > 0 ∧ x ∣ 1058 ∧ ∀ y, 0 < y ∧ y < x → ¬(y ∣ 1058) → x = 2 := by
  sorry

end divisor_problem_l3514_351478
