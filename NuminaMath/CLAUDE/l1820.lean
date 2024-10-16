import Mathlib

namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l1820_182092

theorem simplify_complex_fraction :
  1 / ((3 / (Real.sqrt 5 + 2)) - (4 / (Real.sqrt 7 + 2))) =
  3 * (9 * Real.sqrt 5 + 4 * Real.sqrt 7 + 10) /
  ((9 * Real.sqrt 5 - 4 * Real.sqrt 7 - 10) * (9 * Real.sqrt 5 + 4 * Real.sqrt 7 + 10)) :=
by sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l1820_182092


namespace NUMINAMATH_CALUDE_max_projection_area_parallelepiped_l1820_182024

/-- The maximum area of the orthogonal projection of a rectangular parallelepiped -/
theorem max_projection_area_parallelepiped (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (S : ℝ), S = a * Real.sqrt (a^2 + b^2) ∧
  ∀ (S' : ℝ), S' ≤ S :=
sorry

end NUMINAMATH_CALUDE_max_projection_area_parallelepiped_l1820_182024


namespace NUMINAMATH_CALUDE_sphere_packing_radius_l1820_182031

/-- A sphere in a unit cube -/
structure SpherePacking where
  radius : ℝ
  center_at_vertex : Bool
  touches_three_faces : Bool
  tangent_to_six_neighbors : Bool

/-- The theorem stating the radius of spheres in the specific packing -/
theorem sphere_packing_radius (s : SpherePacking) :
  s.center_at_vertex ∧ 
  s.touches_three_faces ∧ 
  s.tangent_to_six_neighbors →
  s.radius = (Real.sqrt 3 * (Real.sqrt 3 - 1)) / 4 :=
sorry

end NUMINAMATH_CALUDE_sphere_packing_radius_l1820_182031


namespace NUMINAMATH_CALUDE_shorter_side_is_eight_l1820_182077

/-- A rectangle with given area and perimeter -/
structure Rectangle where
  length : ℝ
  width : ℝ
  area_eq : length * width = 104
  perimeter_eq : 2 * (length + width) = 42

/-- The shorter side of the rectangle is 8 feet -/
theorem shorter_side_is_eight (r : Rectangle) : min r.length r.width = 8 := by
  sorry

end NUMINAMATH_CALUDE_shorter_side_is_eight_l1820_182077


namespace NUMINAMATH_CALUDE_ball_tower_levels_l1820_182053

/-- Represents a tower of balls with the given properties -/
structure BallTower where
  topLevel : ℕ
  difference : ℕ
  totalBalls : ℕ

/-- Calculates the number of balls in the nth level of the tower -/
def ballsInLevel (tower : BallTower) (n : ℕ) : ℕ :=
  tower.topLevel + (n - 1) * tower.difference

/-- Calculates the total number of balls in a tower with n levels -/
def totalBallsInTower (tower : BallTower) (n : ℕ) : ℕ :=
  n * (2 * tower.topLevel + (n - 1) * tower.difference) / 2

/-- Theorem stating that a tower with the given properties has 12 levels -/
theorem ball_tower_levels (tower : BallTower) 
  (h1 : tower.topLevel = 2)
  (h2 : tower.difference = 3)
  (h3 : tower.totalBalls = 225) :
  ∃ n : ℕ, n = 12 ∧ totalBallsInTower tower n = tower.totalBalls :=
by
  sorry


end NUMINAMATH_CALUDE_ball_tower_levels_l1820_182053


namespace NUMINAMATH_CALUDE_count_divisible_numbers_l1820_182012

theorem count_divisible_numbers : 
  (Finset.filter 
    (fun k : ℕ => k ≤ 267000 ∧ (k^2 - 1) % 267 = 0) 
    (Finset.range 267001)).card = 4000 :=
by sorry

end NUMINAMATH_CALUDE_count_divisible_numbers_l1820_182012


namespace NUMINAMATH_CALUDE_problem_solution_l1820_182096

-- Define the solution set
def solution_set (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 1

-- Define the inequality
def inequality (x m : ℝ) : Prop := |x + 2| + |x - m| ≤ 3

theorem problem_solution :
  (∀ x, solution_set x ↔ inequality x 1) ∧
  ∀ a b c : ℝ, a^2 + 2*b^2 + 3*c^2 = 1 → -Real.sqrt 6 ≤ a + 2*b + 3*c ∧ a + 2*b + 3*c ≤ Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l1820_182096


namespace NUMINAMATH_CALUDE_picture_books_count_l1820_182083

theorem picture_books_count (total : ℕ) (fiction : ℕ) (non_fiction : ℕ) (autobiographies : ℕ) (picture : ℕ) : 
  total = 35 →
  fiction = 5 →
  non_fiction = fiction + 4 →
  autobiographies = 2 * fiction →
  total = fiction + non_fiction + autobiographies + picture →
  picture = 11 := by
sorry

end NUMINAMATH_CALUDE_picture_books_count_l1820_182083


namespace NUMINAMATH_CALUDE_four_fours_theorem_l1820_182068

def is_valid_expression (e : ℕ → ℕ) : Prop :=
  ∃ (a b c d f : ℕ), 
    (a = 4 ∧ b = 4 ∧ c = 4 ∧ d = 4 ∧ f = 4) ∧
    (∀ n : ℕ, 1 ≤ n ∧ n ≤ 22 → e n = n)

theorem four_fours_theorem :
  ∃ e : ℕ → ℕ, is_valid_expression e :=
sorry

end NUMINAMATH_CALUDE_four_fours_theorem_l1820_182068


namespace NUMINAMATH_CALUDE_intersection_point_x_coordinate_l1820_182019

theorem intersection_point_x_coordinate (x y : ℝ) : 
  y = 4 * x - 19 ∧ 2 * x + y = 95 → x = 19 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_x_coordinate_l1820_182019


namespace NUMINAMATH_CALUDE_unique_polynomial_function_l1820_182026

-- Define a polynomial function of degree 3
def PolynomialDegree3 (f : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^3 + b * x^2 + c * x + d

-- Define the conditions given in the problem
def SatisfiesConditions (f : ℝ → ℝ) : Prop :=
  (∀ x, f (x^2) = (f x)^2) ∧
  (∀ x, f (x^2) = f (f x)) ∧
  f 1 = f (-1)

-- Theorem statement
theorem unique_polynomial_function :
  ∃! f : ℝ → ℝ, PolynomialDegree3 f ∧ SatisfiesConditions f ∧ (∀ x, f x = x^3) :=
sorry

end NUMINAMATH_CALUDE_unique_polynomial_function_l1820_182026


namespace NUMINAMATH_CALUDE_hiker_rate_ratio_l1820_182093

/-- Proves that the ratio of the rate down to the rate up is 1.5 given the hiking conditions --/
theorem hiker_rate_ratio 
  (rate_up : ℝ) 
  (time_up : ℝ) 
  (distance_down : ℝ) 
  (h1 : rate_up = 3) 
  (h2 : time_up = 2) 
  (h3 : distance_down = 9) 
  (h4 : time_up = distance_down / (distance_down / time_up)) : 
  (distance_down / time_up) / rate_up = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_hiker_rate_ratio_l1820_182093


namespace NUMINAMATH_CALUDE_multiples_properties_l1820_182080

theorem multiples_properties (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 4 * k) 
  (hb : ∃ m : ℤ, b = 8 * m) : 
  (∃ n : ℤ, b = 4 * n) ∧ 
  (∃ p : ℤ, a - b = 4 * p) := by
sorry

end NUMINAMATH_CALUDE_multiples_properties_l1820_182080


namespace NUMINAMATH_CALUDE_jake_kendra_weight_ratio_l1820_182001

/-- The problem of Jake and Kendra's weight ratio -/
theorem jake_kendra_weight_ratio :
  ∀ (j k : ℝ),
  j + k = 293 →
  j - 8 = 2 * k →
  (j - 8) / k = 2 :=
by sorry

end NUMINAMATH_CALUDE_jake_kendra_weight_ratio_l1820_182001


namespace NUMINAMATH_CALUDE_salary_increase_after_reduction_l1820_182052

theorem salary_increase_after_reduction : ∀ (original_salary : ℝ),
  original_salary > 0 →
  let reduced_salary := original_salary * (1 - 0.25)
  let increase_factor := (1 + 1/3)
  reduced_salary * increase_factor = original_salary :=
by
  sorry

end NUMINAMATH_CALUDE_salary_increase_after_reduction_l1820_182052


namespace NUMINAMATH_CALUDE_negative_three_a_plus_two_a_equals_negative_a_l1820_182084

theorem negative_three_a_plus_two_a_equals_negative_a (a : ℝ) : -3*a + 2*a = -a := by
  sorry

end NUMINAMATH_CALUDE_negative_three_a_plus_two_a_equals_negative_a_l1820_182084


namespace NUMINAMATH_CALUDE_function_value_at_negative_two_l1820_182011

/-- Given a function f(x) = ax^5 + bx^3 + cx + 1 where f(2) = -1, prove that f(-2) = 3 -/
theorem function_value_at_negative_two 
  (a b c : ℝ) 
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = a * x^5 + b * x^3 + c * x + 1)
  (h2 : f 2 = -1) : 
  f (-2) = 3 := by
sorry

end NUMINAMATH_CALUDE_function_value_at_negative_two_l1820_182011


namespace NUMINAMATH_CALUDE_book_arrangement_l1820_182048

theorem book_arrangement (n m : ℕ) (h : n + m = 11) :
  Nat.choose (n + m) n = 462 :=
sorry

end NUMINAMATH_CALUDE_book_arrangement_l1820_182048


namespace NUMINAMATH_CALUDE_carpool_arrangement_count_l1820_182043

def num_students : ℕ := 8
def num_grades : ℕ := 4
def students_per_grade : ℕ := 2
def car_capacity : ℕ := 4

def has_twin_sisters : Prop := true

theorem carpool_arrangement_count : ℕ := by
  sorry

end NUMINAMATH_CALUDE_carpool_arrangement_count_l1820_182043


namespace NUMINAMATH_CALUDE_square_difference_theorem_l1820_182046

theorem square_difference_theorem (x : ℝ) : 
  (x + 2)^2 - x^2 = 32 → x + 2 = 9 := by sorry

end NUMINAMATH_CALUDE_square_difference_theorem_l1820_182046


namespace NUMINAMATH_CALUDE_sequence_product_l1820_182039

theorem sequence_product (n a : ℕ) :
  ∃ u v : ℕ, n / (n + a) = (u / (u + a)) * (v / (v + a)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_product_l1820_182039


namespace NUMINAMATH_CALUDE_binary_multiplication_theorem_l1820_182090

def binary_to_nat (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def nat_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec to_binary_aux (m : Nat) (acc : List Bool) : List Bool :=
    if m = 0 then acc
    else to_binary_aux (m / 2) ((m % 2 = 1) :: acc)
  to_binary_aux n []

def binary_mult (a b : List Bool) : List Bool :=
  nat_to_binary (binary_to_nat a * binary_to_nat b)

theorem binary_multiplication_theorem :
  binary_mult [true, false, false, true, true] [true, true, true] = 
  [true, true, true, true, true, false, true, false, true] := by
  sorry

end NUMINAMATH_CALUDE_binary_multiplication_theorem_l1820_182090


namespace NUMINAMATH_CALUDE_distance_to_school_is_two_prove_distance_to_school_l1820_182078

/-- The distance to school in miles -/
def distance_to_school : ℝ := 2

/-- Jerry's one-way trip time in minutes -/
def jerry_one_way_time : ℝ := 15

/-- Carson's speed in miles per hour -/
def carson_speed : ℝ := 8

/-- Theorem stating that the distance to school is 2 miles -/
theorem distance_to_school_is_two :
  distance_to_school = 2 :=
by
  sorry

/-- Lemma: Jerry's round trip time equals Carson's one-way trip time -/
lemma jerry_round_trip_equals_carson_one_way :
  2 * jerry_one_way_time = distance_to_school / (carson_speed / 60) :=
by
  sorry

/-- Main theorem proving the distance to school -/
theorem prove_distance_to_school :
  distance_to_school = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_distance_to_school_is_two_prove_distance_to_school_l1820_182078


namespace NUMINAMATH_CALUDE_max_plain_cupcakes_l1820_182009

structure Cupcakes :=
  (total : ℕ)
  (blueberries : ℕ)
  (sprinkles : ℕ)
  (frosting : ℕ)
  (pecans : ℕ)

def has_no_ingredients (c : Cupcakes) : ℕ :=
  c.total - (c.blueberries + c.sprinkles + c.frosting + c.pecans)

theorem max_plain_cupcakes (c : Cupcakes) 
  (h_total : c.total = 60)
  (h_blueberries : c.blueberries ≥ c.total / 3)
  (h_sprinkles : c.sprinkles ≥ c.total / 4)
  (h_frosting : c.frosting ≥ c.total / 2)
  (h_pecans : c.pecans ≥ c.total / 5) :
  has_no_ingredients c ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_max_plain_cupcakes_l1820_182009


namespace NUMINAMATH_CALUDE_rect_to_polar_conversion_l1820_182028

/-- Conversion from rectangular coordinates to polar coordinates -/
theorem rect_to_polar_conversion :
  ∀ (x y : ℝ), x = -1 ∧ y = 1 →
  ∃ (r θ : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧
  r = Real.sqrt 2 ∧ θ = 3 * Real.pi / 4 ∧
  x = r * Real.cos θ ∧ y = r * Real.sin θ :=
by sorry

end NUMINAMATH_CALUDE_rect_to_polar_conversion_l1820_182028


namespace NUMINAMATH_CALUDE_gcf_of_36_60_90_l1820_182018

theorem gcf_of_36_60_90 : Nat.gcd 36 (Nat.gcd 60 90) = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_36_60_90_l1820_182018


namespace NUMINAMATH_CALUDE_max_abs_sum_on_circle_l1820_182035

theorem max_abs_sum_on_circle (x y : ℝ) (h : x^2 + y^2 = 2) : 
  |x| + |y| ≤ 2 ∧ ∃ (a b : ℝ), a^2 + b^2 = 2 ∧ |a| + |b| = 2 := by
  sorry

end NUMINAMATH_CALUDE_max_abs_sum_on_circle_l1820_182035


namespace NUMINAMATH_CALUDE_chess_tournament_square_players_l1820_182033

/-- Represents a chess tournament with men and women players -/
structure ChessTournament where
  k : ℕ  -- number of men
  m : ℕ  -- number of women

/-- The total number of players in the tournament -/
def ChessTournament.totalPlayers (t : ChessTournament) : ℕ := t.k + t.m

/-- The condition that total points scored by men against women equals total points scored by women against men -/
def ChessTournament.equalCrossScores (t : ChessTournament) : Prop :=
  (t.k * (t.k - 1)) / 2 + (t.m * (t.m - 1)) / 2 = t.k * t.m

theorem chess_tournament_square_players (t : ChessTournament) 
  (h : t.equalCrossScores) : 
  ∃ n : ℕ, t.totalPlayers = n^2 := by
  sorry


end NUMINAMATH_CALUDE_chess_tournament_square_players_l1820_182033


namespace NUMINAMATH_CALUDE_stewart_farm_sheep_horse_ratio_l1820_182040

/-- Represents the Stewart farm with sheep and horses -/
structure StewartFarm where
  sheep : ℕ
  horses : ℕ
  horseFoodPerHorse : ℕ
  totalHorseFood : ℕ

/-- The ratio between two natural numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Calculates the simplified ratio between two natural numbers -/
def simplifiedRatio (a b : ℕ) : Ratio :=
  let gcd := Nat.gcd a b
  { numerator := a / gcd, denominator := b / gcd }

/-- Theorem: The ratio of sheep to horses on the Stewart farm is 5:7 -/
theorem stewart_farm_sheep_horse_ratio (farm : StewartFarm)
    (h1 : farm.sheep = 40)
    (h2 : farm.horseFoodPerHorse = 230)
    (h3 : farm.totalHorseFood = 12880)
    (h4 : farm.horses * farm.horseFoodPerHorse = farm.totalHorseFood) :
    simplifiedRatio farm.sheep farm.horses = { numerator := 5, denominator := 7 } := by
  sorry

end NUMINAMATH_CALUDE_stewart_farm_sheep_horse_ratio_l1820_182040


namespace NUMINAMATH_CALUDE_bella_ella_meeting_l1820_182069

/-- The distance between Bella's and Ella's houses in feet -/
def distance : ℕ := 15840

/-- The length of Bella's step in feet -/
def step_length : ℕ := 3

/-- The ratio of Ella's speed to Bella's speed -/
def speed_ratio : ℕ := 5

/-- The number of steps Bella takes before meeting Ella -/
def steps_taken : ℕ := 880

theorem bella_ella_meeting :
  distance = 15840 ∧
  step_length = 3 ∧
  speed_ratio = 5 →
  steps_taken = 880 :=
by sorry

end NUMINAMATH_CALUDE_bella_ella_meeting_l1820_182069


namespace NUMINAMATH_CALUDE_plywood_perimeter_difference_l1820_182030

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle --/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Represents the plywood and its cutting --/
structure Plywood where
  length : ℝ
  width : ℝ
  num_pieces : ℕ

/-- Generates all possible ways to cut the plywood into congruent rectangles --/
def possible_cuts (p : Plywood) : List Rectangle :=
  sorry -- Implementation details omitted

/-- Finds the maximum perimeter among the possible cuts --/
def max_perimeter (cuts : List Rectangle) : ℝ :=
  sorry -- Implementation details omitted

/-- Finds the minimum perimeter among the possible cuts --/
def min_perimeter (cuts : List Rectangle) : ℝ :=
  sorry -- Implementation details omitted

theorem plywood_perimeter_difference :
  let p : Plywood := { length := 6, width := 9, num_pieces := 6 }
  let cuts := possible_cuts p
  max_perimeter cuts - min_perimeter cuts = 11 := by
  sorry

end NUMINAMATH_CALUDE_plywood_perimeter_difference_l1820_182030


namespace NUMINAMATH_CALUDE_chessboard_coloring_limit_l1820_182025

/-- Represents the minimum number of colored vertices required on an n × n chessboard
    such that any k × k square has at least one edge with a colored vertex. -/
noncomputable def l (n : ℕ) : ℕ := sorry

/-- The limit of l(n)/n² as n approaches infinity is 2/7. -/
theorem chessboard_coloring_limit :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |l n / (n^2 : ℝ) - 2/7| < ε :=
sorry

end NUMINAMATH_CALUDE_chessboard_coloring_limit_l1820_182025


namespace NUMINAMATH_CALUDE_decimal_29_to_binary_l1820_182020

def decimal_to_binary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 2) ((m % 2) :: acc)
    aux n []

theorem decimal_29_to_binary :
  decimal_to_binary 29 = [1, 1, 1, 0, 1] := by sorry

end NUMINAMATH_CALUDE_decimal_29_to_binary_l1820_182020


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1820_182002

-- Define the inequality system
def inequality_system (x : ℝ) : Prop := x + 1 > 0 ∧ x > -3

-- Define the solution set
def solution_set : Set ℝ := {x | x > -1}

-- Theorem statement
theorem inequality_system_solution :
  {x : ℝ | inequality_system x} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1820_182002


namespace NUMINAMATH_CALUDE_power_equation_solution_l1820_182085

theorem power_equation_solution : ∃ K : ℕ, 16^3 * 8^3 = 2^K ∧ K = 21 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l1820_182085


namespace NUMINAMATH_CALUDE_sum_abc_equals_three_l1820_182016

theorem sum_abc_equals_three (a b c : ℝ) 
  (eq1 : a^2 + 2*b = 7)
  (eq2 : b^2 - 2*c = -1)
  (eq3 : c^2 - 6*a = -17) : 
  a + b + c = 3 := by
sorry

end NUMINAMATH_CALUDE_sum_abc_equals_three_l1820_182016


namespace NUMINAMATH_CALUDE_terry_lunch_options_l1820_182032

/-- The number of lunch combination options for Terry's salad bar lunch. -/
def lunch_combinations (lettuce_types : ℕ) (tomato_types : ℕ) (olive_types : ℕ) 
                       (bread_types : ℕ) (fruit_types : ℕ) (soup_types : ℕ) : ℕ :=
  lettuce_types * tomato_types * olive_types * bread_types * fruit_types * soup_types

/-- Theorem stating that Terry's lunch combinations equal 4320. -/
theorem terry_lunch_options :
  lunch_combinations 4 5 6 3 4 3 = 4320 := by
  sorry

end NUMINAMATH_CALUDE_terry_lunch_options_l1820_182032


namespace NUMINAMATH_CALUDE_binary_sum_to_octal_to_decimal_l1820_182095

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a decimal number to its octal representation -/
def decimal_to_octal (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc else aux (m / 8) ((m % 8) :: acc)
    aux n []

/-- Converts an octal number represented as a list of digits to its decimal equivalent -/
def octal_to_decimal (digits : List ℕ) : ℕ :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 8^(digits.length - 1 - i)) 0

/-- The main theorem to be proved -/
theorem binary_sum_to_octal_to_decimal : 
  let a := binary_to_decimal [true, true, true, true, true, true, true, true]
  let b := binary_to_decimal [true, true, true, true, true]
  let sum := a + b
  let octal := decimal_to_octal sum
  octal_to_decimal octal = 286 := by
  sorry

end NUMINAMATH_CALUDE_binary_sum_to_octal_to_decimal_l1820_182095


namespace NUMINAMATH_CALUDE_phillips_apples_l1820_182027

theorem phillips_apples (ben phillip tom : ℕ) : 
  ben = phillip + 8 →
  3 * ben = 8 * tom →
  tom = 18 →
  phillip = 40 := by
sorry

end NUMINAMATH_CALUDE_phillips_apples_l1820_182027


namespace NUMINAMATH_CALUDE_gcd_of_powers_of_101_l1820_182022

theorem gcd_of_powers_of_101 : 
  Nat.Prime 101 → Nat.gcd (101^11 + 1) (101^11 + 101^3 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_powers_of_101_l1820_182022


namespace NUMINAMATH_CALUDE_tan_two_implies_sum_l1820_182097

theorem tan_two_implies_sum (θ : ℝ) (h : Real.tan θ = 2) : 
  2 * Real.sin θ + Real.sin θ * Real.cos θ = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_two_implies_sum_l1820_182097


namespace NUMINAMATH_CALUDE_polynomial_division_quotient_l1820_182047

theorem polynomial_division_quotient :
  let dividend : Polynomial ℤ := X^6 + 3*X^4 - 2*X^3 + X + 12
  let divisor : Polynomial ℤ := X - 2
  let quotient : Polynomial ℤ := X^5 + 2*X^4 + 6*X^3 + 10*X^2 + 18*X + 34
  dividend = divisor * quotient + 80 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_quotient_l1820_182047


namespace NUMINAMATH_CALUDE_paper_length_proof_l1820_182067

theorem paper_length_proof (cube_volume : ℝ) (paper_width : ℝ) (inches_per_foot : ℝ) :
  cube_volume = 8 →
  paper_width = 72 →
  inches_per_foot = 12 →
  ∃ (paper_length : ℝ),
    paper_length * paper_width = (cube_volume^(1/3) * inches_per_foot)^2 ∧
    paper_length = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_paper_length_proof_l1820_182067


namespace NUMINAMATH_CALUDE_part_one_part_two_l1820_182061

-- Part 1
theorem part_one (α : Real) (h : Real.tan α = 2) :
  (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = -3 := by sorry

-- Part 2
theorem part_two (α : Real) :
  (Real.sin (α - π/2) * Real.cos (π/2 - α) * Real.tan (π - α)) / 
  (Real.tan (π + α) * Real.sin (π + α)) = -Real.cos α := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1820_182061


namespace NUMINAMATH_CALUDE_percentage_problem_l1820_182038

theorem percentage_problem (P : ℝ) : 
  (P / 100) * 100 - 40 = 30 → P = 70 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1820_182038


namespace NUMINAMATH_CALUDE_probability_divisible_by_15_l1820_182079

/-- The set of digits used to form the six-digit number -/
def digits : Finset Nat := {1, 2, 3, 4, 5, 9}

/-- The number of digits -/
def n : Nat := 6

/-- A permutation of the digits -/
def Permutation := Fin n → Fin n

/-- The set of all permutations -/
def allPermutations : Finset Permutation := sorry

/-- Predicate to check if a permutation results in a number divisible by 15 -/
def isDivisibleBy15 (p : Permutation) : Prop := sorry

/-- The number of permutations that result in a number divisible by 15 -/
def divisibleBy15Count : Nat := sorry

/-- The total number of permutations -/
def totalPermutations : Nat := Finset.card allPermutations

theorem probability_divisible_by_15 :
  (divisibleBy15Count : ℚ) / totalPermutations = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_probability_divisible_by_15_l1820_182079


namespace NUMINAMATH_CALUDE_vector_at_t_5_l1820_182082

/-- A parameterized line in 2D space -/
structure ParameterizedLine where
  /-- The vector on the line at parameter t -/
  vector_at : ℝ → ℝ × ℝ

/-- Theorem: Given a parameterized line with specific points, the vector at t=5 is (10, -11) -/
theorem vector_at_t_5 
  (line : ParameterizedLine) 
  (h1 : line.vector_at 1 = (2, 5)) 
  (h4 : line.vector_at 4 = (8, -7)) : 
  line.vector_at 5 = (10, -11) := by
sorry


end NUMINAMATH_CALUDE_vector_at_t_5_l1820_182082


namespace NUMINAMATH_CALUDE_orange_harvest_problem_l1820_182064

/-- Proves that the number of sacks harvested per day is 66, given the conditions of the orange harvest problem. -/
theorem orange_harvest_problem (oranges_per_sack : ℕ) (harvest_days : ℕ) (total_oranges : ℕ) 
  (h1 : oranges_per_sack = 25)
  (h2 : harvest_days = 87)
  (h3 : total_oranges = 143550) :
  total_oranges / (oranges_per_sack * harvest_days) = 66 := by
  sorry

#eval 143550 / (25 * 87)  -- Should output 66

end NUMINAMATH_CALUDE_orange_harvest_problem_l1820_182064


namespace NUMINAMATH_CALUDE_circle_radius_l1820_182091

/-- The equation of a circle in the form x^2 + y^2 + 2x = 0 has radius 1 -/
theorem circle_radius (x y : ℝ) : x^2 + y^2 + 2*x = 0 → ∃ (h k : ℝ), (x - h)^2 + (y - k)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l1820_182091


namespace NUMINAMATH_CALUDE_solve_otimes_equation_l1820_182023

-- Define the ⊗ operation
def otimes (a b : ℝ) : ℝ := a - 3 * b

-- State the theorem
theorem solve_otimes_equation :
  ∃! x : ℝ, otimes x 1 + otimes 2 x = 1 ∧ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_solve_otimes_equation_l1820_182023


namespace NUMINAMATH_CALUDE_square_perimeter_l1820_182004

/-- Given a square with area 400 square meters, its perimeter is 80 meters. -/
theorem square_perimeter (s : ℝ) (h : s^2 = 400) : 4 * s = 80 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l1820_182004


namespace NUMINAMATH_CALUDE_problem_solution_l1820_182055

-- Define the solution set for |x-m| < |x|
def solution_set (m : ℝ) : Set ℝ := {x : ℝ | 1 < x}

-- Define the inequality condition
def inequality_condition (a m : ℝ) (x : ℝ) : Prop :=
  (a - 5) / x < |1 + 1/x| - |1 - m/x| ∧ |1 + 1/x| - |1 - m/x| < (a + 2) / x

theorem problem_solution :
  (∀ x : ℝ, x ∈ solution_set m ↔ |x - m| < |x|) →
  m = 2 ∧
  (∀ a : ℝ, (∀ x : ℝ, x > 0 → inequality_condition a m x) ↔ 1 < a ∧ a ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l1820_182055


namespace NUMINAMATH_CALUDE_james_weekly_income_l1820_182073

/-- Calculates the weekly income from car rental given hourly rate, hours per day, and days per week. -/
def weekly_income (hourly_rate : ℝ) (hours_per_day : ℝ) (days_per_week : ℝ) : ℝ :=
  hourly_rate * hours_per_day * days_per_week

/-- Proves that James' weekly income from car rental is $640 given the specified conditions. -/
theorem james_weekly_income :
  let hourly_rate : ℝ := 20
  let hours_per_day : ℝ := 8
  let days_per_week : ℝ := 4
  weekly_income hourly_rate hours_per_day days_per_week = 640 := by
  sorry

#eval weekly_income 20 8 4

end NUMINAMATH_CALUDE_james_weekly_income_l1820_182073


namespace NUMINAMATH_CALUDE_only_three_four_five_is_right_triangle_l1820_182086

def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem only_three_four_five_is_right_triangle :
  (¬ is_right_triangle 1 2 3) ∧
  (¬ is_right_triangle 2 3 4) ∧
  (is_right_triangle 3 4 5) ∧
  (¬ is_right_triangle 1 2 3) :=
sorry

end NUMINAMATH_CALUDE_only_three_four_five_is_right_triangle_l1820_182086


namespace NUMINAMATH_CALUDE_product_of_x_values_l1820_182000

theorem product_of_x_values (x₁ x₂ : ℝ) : 
  (|20 / x₁ + 4| = 3 ∧ |20 / x₂ + 4| = 3 ∧ x₁ ≠ x₂) → x₁ * x₂ = 400 / 7 :=
by sorry

end NUMINAMATH_CALUDE_product_of_x_values_l1820_182000


namespace NUMINAMATH_CALUDE_valid_distribution_exists_l1820_182066

/-- Represents a part of the city -/
structure CityPart where
  id : Nat

/-- Represents a currency exchange point -/
structure ExchangePoint where
  id : Nat

/-- A distribution of exchange points across city parts -/
def Distribution := CityPart → Finset ExchangePoint

/-- The property that each city part contains exactly two exchange points -/
def ValidDistribution (d : Distribution) (cityParts : Finset CityPart) (exchangePoints : Finset ExchangePoint) : Prop :=
  ∀ cp ∈ cityParts, (d cp).card = 2

/-- The main theorem stating that a valid distribution exists -/
theorem valid_distribution_exists (cityParts : Finset CityPart) (exchangePoints : Finset ExchangePoint)
    (h1 : cityParts.card = 4) (h2 : exchangePoints.card = 4) :
    ∃ d : Distribution, ValidDistribution d cityParts exchangePoints := by
  sorry

end NUMINAMATH_CALUDE_valid_distribution_exists_l1820_182066


namespace NUMINAMATH_CALUDE_chain_breaking_theorem_l1820_182094

/-- Represents a chain with n links -/
structure Chain (n : ℕ) where
  links : Fin n → ℕ
  all_links_one : ∀ i, links i = 1

/-- Represents a set of chain segments after breaking -/
structure Segments (n : ℕ) where
  pieces : List ℕ
  sum_pieces : pieces.sum = n

/-- Function to break a chain into segments -/
def break_chain (n : ℕ) (k : ℕ) (break_points : Fin (k-1) → ℕ) : Segments n :=
  sorry

/-- Function to check if a weight can be measured using given segments -/
def can_measure (segments : List ℕ) (weight : ℕ) : Prop :=
  sorry

theorem chain_breaking_theorem (k : ℕ) :
  let n := k * 2^k - 1
  ∃ (break_points : Fin (k-1) → ℕ),
    let segments := (break_chain n k break_points).pieces
    ∀ w : ℕ, w ≤ n → can_measure segments w :=
  sorry

end NUMINAMATH_CALUDE_chain_breaking_theorem_l1820_182094


namespace NUMINAMATH_CALUDE_laptop_original_price_l1820_182051

/-- Proves that if a laptop's price is reduced by 15% and the new price is $680, then the original price was $800. -/
theorem laptop_original_price (discount_percent : ℝ) (discounted_price : ℝ) (original_price : ℝ) : 
  discount_percent = 15 →
  discounted_price = 680 →
  discounted_price = original_price * (1 - discount_percent / 100) →
  original_price = 800 := by
sorry

end NUMINAMATH_CALUDE_laptop_original_price_l1820_182051


namespace NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l1820_182062

theorem count_integers_satisfying_inequality :
  (Finset.filter (fun n : ℤ => (n - 3) * (n + 2) * (n + 6) < 0)
    (Finset.Icc (-11 : ℤ) 11)).card = 12 := by
  sorry

end NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l1820_182062


namespace NUMINAMATH_CALUDE_inscribed_rectangle_area_l1820_182010

/-- Given a triangle with base 12 and altitude 8, and an inscribed rectangle with height 4,
    the area of the rectangle is 48. -/
theorem inscribed_rectangle_area (b h x : ℝ) : 
  b = 12 → h = 8 → x = h / 2 → x = 4 → 
  ∃ (w : ℝ), w * x = 48 := by sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_area_l1820_182010


namespace NUMINAMATH_CALUDE_largest_three_digit_multiple_of_6_with_digit_sum_15_l1820_182050

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem largest_three_digit_multiple_of_6_with_digit_sum_15 :
  ∀ n : ℕ, is_three_digit n → n % 6 = 0 → digit_sum n = 15 → n ≤ 960 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_multiple_of_6_with_digit_sum_15_l1820_182050


namespace NUMINAMATH_CALUDE_sum_of_sequences_is_435_l1820_182003

def sequence1 : List ℕ := [2, 14, 26, 38, 50]
def sequence2 : List ℕ := [12, 24, 36, 48, 60]
def sequence3 : List ℕ := [5, 15, 25, 35, 45]

theorem sum_of_sequences_is_435 :
  (sequence1.sum + sequence2.sum + sequence3.sum) = 435 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_sequences_is_435_l1820_182003


namespace NUMINAMATH_CALUDE_concert_guests_combinations_l1820_182042

theorem concert_guests_combinations : Nat.choose 10 5 = 252 := by
  sorry

end NUMINAMATH_CALUDE_concert_guests_combinations_l1820_182042


namespace NUMINAMATH_CALUDE_medical_team_selection_l1820_182088

/-- The number of ways to select k items from n items. -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of internists available. -/
def num_internists : ℕ := 5

/-- The number of surgeons available. -/
def num_surgeons : ℕ := 6

/-- The total number of doctors needed for the team. -/
def team_size : ℕ := 4

/-- The number of ways to select the medical team. -/
def select_team : ℕ :=
  choose num_internists 1 * choose num_surgeons 3 +
  choose num_internists 2 * choose num_surgeons 2 +
  choose num_internists 3 * choose num_surgeons 1

theorem medical_team_selection :
  select_team = 310 := by sorry

end NUMINAMATH_CALUDE_medical_team_selection_l1820_182088


namespace NUMINAMATH_CALUDE_arithmetic_geometric_harmonic_mean_sum_of_squares_l1820_182006

theorem arithmetic_geometric_harmonic_mean_sum_of_squares
  (a b c : ℝ)
  (h_arithmetic : (a + b + c) / 3 = 8)
  (h_geometric : (a * b * c) ^ (1/3 : ℝ) = 5)
  (h_harmonic : 3 / (1/a + 1/b + 1/c) = 3) :
  a^2 + b^2 + c^2 = 326 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_harmonic_mean_sum_of_squares_l1820_182006


namespace NUMINAMATH_CALUDE_expression_equals_one_l1820_182041

theorem expression_equals_one (x : ℝ) (h1 : x^3 ≠ -1) (h2 : x^3 ≠ 1) : 
  ((x+1)^3 * (x^2-x+1)^3 / (x^3+1)^3)^2 * ((x-1)^3 * (x^2+x+1)^3 / (x^3-1)^3)^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_expression_equals_one_l1820_182041


namespace NUMINAMATH_CALUDE_sum_cubic_over_power_of_three_l1820_182037

/-- The sum of the infinite series ∑_{k = 1}^∞ (k^3 / 3^k) is equal to 1.5 -/
theorem sum_cubic_over_power_of_three :
  ∑' k : ℕ, (k^3 : ℝ) / 3^k = (3/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_sum_cubic_over_power_of_three_l1820_182037


namespace NUMINAMATH_CALUDE_complex_imaginary_solution_l1820_182005

/-- Given that z = m^2 - (1-i)m is an imaginary number, prove that m = 1 -/
theorem complex_imaginary_solution (m : ℂ) : 
  let z := m^2 - (1 - Complex.I) * m
  (∃ (y : ℝ), z = Complex.I * y) ∧ z ≠ 0 → m = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_imaginary_solution_l1820_182005


namespace NUMINAMATH_CALUDE_citizenship_test_study_time_l1820_182017

/-- Represents the time in minutes to learn each fill-in-the-blank question -/
def time_per_blank_question (total_questions : ℕ) (multiple_choice : ℕ) (fill_blank : ℕ) 
  (time_per_mc : ℕ) (total_study_time : ℕ) : ℕ :=
  ((total_study_time * 60) - (multiple_choice * time_per_mc)) / fill_blank

/-- Theorem stating that given the conditions, the time to learn each fill-in-the-blank question is 25 minutes -/
theorem citizenship_test_study_time :
  time_per_blank_question 60 30 30 15 20 = 25 := by
  sorry

end NUMINAMATH_CALUDE_citizenship_test_study_time_l1820_182017


namespace NUMINAMATH_CALUDE_train_crossing_contradiction_l1820_182089

theorem train_crossing_contradiction (V₁ V₂ L₁ L₂ T₂ : ℝ) : 
  V₁ > 0 → V₂ > 0 → L₁ > 0 → L₂ > 0 → T₂ > 0 →
  (L₁ / V₁ = 20) →  -- First train crosses man in 20 seconds
  (L₂ / V₂ = T₂) →  -- Second train crosses man in T₂ seconds
  ((L₁ + L₂) / (V₁ + V₂) = 19) →  -- Trains cross each other in 19 seconds
  (V₁ = V₂) →  -- Ratio of speeds is 1
  False :=  -- This leads to a contradiction
by
  sorry

#check train_crossing_contradiction

end NUMINAMATH_CALUDE_train_crossing_contradiction_l1820_182089


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1820_182076

/-- Given a hyperbola and a circle with specific properties, prove the equation of the hyperbola -/
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 → 
    (∃ t : ℝ, (b*x + a*y = 0 ∨ b*x - a*y = 0) ∧ 
      ((x - 3)^2 + y^2 = 4 ↔ t = 0))) → 
  (∃ c : ℝ, c > 0 ∧ c^2 = a^2 + b^2 ∧ c = 3) →
  (a^2 = 5 ∧ b^2 = 4) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1820_182076


namespace NUMINAMATH_CALUDE_triangle_inequality_l1820_182013

/-- Triangle inequality theorem -/
theorem triangle_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  2 * (a^2 + b^2) > c^2 ∧ 
  ∀ ε > 0, ∃ a' b' c', a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ 
    (2 - ε) * (a'^2 + b'^2) ≤ c'^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1820_182013


namespace NUMINAMATH_CALUDE_completing_square_quadratic_l1820_182057

theorem completing_square_quadratic (x : ℝ) : 
  (x^2 - 4*x - 11 = 0) ↔ ((x - 2)^2 = 15) :=
by sorry

end NUMINAMATH_CALUDE_completing_square_quadratic_l1820_182057


namespace NUMINAMATH_CALUDE_equation_solution_l1820_182098

theorem equation_solution : ∃ x : ℝ, 5*x + 9*x = 420 - 12*(x - 4) ∧ x = 18 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1820_182098


namespace NUMINAMATH_CALUDE_external_tangent_same_color_l1820_182021

/-- A point on a line --/
structure Point where
  x : ℝ

/-- A circle with diameter endpoints --/
structure Circle where
  p1 : Point
  p2 : Point

/-- A color represented as a natural number --/
def Color := ℕ

/-- The set of all circles formed by pairs of points --/
def allCircles (points : List Point) : List Circle :=
  sorry

/-- Checks if two circles are externally tangent --/
def areExternallyTangent (c1 c2 : Circle) : Prop :=
  sorry

/-- Assigns a color to each circle --/
def colorAssignment (circles : List Circle) (n : ℕ) : Circle → Color :=
  sorry

/-- Main theorem --/
theorem external_tangent_same_color 
  (k n : ℕ) (points : List Point) (h : k > 2^n) (h2 : points.length = k) :
  ∃ (c1 c2 : Circle), c1 ∈ allCircles points ∧ c2 ∈ allCircles points ∧ 
    c1 ≠ c2 ∧
    areExternallyTangent c1 c2 ∧
    colorAssignment (allCircles points) n c1 = colorAssignment (allCircles points) n c2 :=
  sorry

end NUMINAMATH_CALUDE_external_tangent_same_color_l1820_182021


namespace NUMINAMATH_CALUDE_certain_number_exists_and_unique_l1820_182065

theorem certain_number_exists_and_unique : 
  ∃! x : ℝ, x / 5 + x + 5 = 65 := by sorry

end NUMINAMATH_CALUDE_certain_number_exists_and_unique_l1820_182065


namespace NUMINAMATH_CALUDE_sqrt_three_irrational_l1820_182036

theorem sqrt_three_irrational : Irrational (Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_irrational_l1820_182036


namespace NUMINAMATH_CALUDE_linear_function_through_origin_l1820_182071

/-- A linear function y = nx + (n^2 - 7) passing through (0, 2) with negative slope has n = -3 -/
theorem linear_function_through_origin (n : ℝ) : 
  (2 = n^2 - 7) →  -- The graph passes through (0, 2)
  (n < 0) →        -- y decreases as x increases (negative slope)
  n = -3 := by
sorry

end NUMINAMATH_CALUDE_linear_function_through_origin_l1820_182071


namespace NUMINAMATH_CALUDE_cube_isosceles_right_probability_l1820_182058

/-- A cube with 8 vertices -/
structure Cube :=
  (vertices : Fin 8)

/-- A triangle formed by 3 vertices of a cube -/
structure CubeTriangle :=
  (v1 v2 v3 : Fin 8)
  (distinct : v1 ≠ v2 ∧ v1 ≠ v3 ∧ v2 ≠ v3)

/-- An isosceles right triangle on a cube face -/
def IsIsoscelesRight (t : CubeTriangle) : Prop :=
  sorry

/-- The number of isosceles right triangles that can be formed on a cube -/
def numIsoscelesRight : ℕ := 24

/-- The total number of ways to select 3 vertices from 8 -/
def totalTriangles : ℕ := 56

/-- The probability of forming an isosceles right triangle -/
def probabilityIsoscelesRight : ℚ := 3/7

theorem cube_isosceles_right_probability :
  (numIsoscelesRight : ℚ) / totalTriangles = probabilityIsoscelesRight :=
sorry

end NUMINAMATH_CALUDE_cube_isosceles_right_probability_l1820_182058


namespace NUMINAMATH_CALUDE_male_democrat_ratio_l1820_182075

/-- Proves the ratio of male democrats to total male participants in a meeting --/
theorem male_democrat_ratio (total_participants : ℕ) 
  (female_democrats : ℕ) (h1 : total_participants = 660) 
  (h2 : female_democrats = 110) 
  (h3 : female_democrats * 2 = total_participants / 3) : 
  (total_participants / 3 - female_democrats) * 4 = 
  (total_participants - female_democrats * 2) :=
sorry

end NUMINAMATH_CALUDE_male_democrat_ratio_l1820_182075


namespace NUMINAMATH_CALUDE_bipartite_graph_completion_l1820_182070

/-- A bipartite graph with n vertices in each partition -/
structure BipartiteGraph (n : ℕ) :=
  (A B : Finset (Fin n))
  (edges : Finset (Fin n × Fin n))
  (bipartite : ∀ (e : Fin n × Fin n), e ∈ edges → (e.1 ∈ A ∧ e.2 ∈ B) ∨ (e.1 ∈ B ∧ e.2 ∈ A))

/-- The degree of a vertex in a bipartite graph -/
def degree (G : BipartiteGraph n) (v : Fin n) : ℕ :=
  (G.edges.filter (λ e => e.1 = v ∨ e.2 = v)).card

/-- The theorem statement -/
theorem bipartite_graph_completion
  (n d : ℕ) (h_pos : 0 < n ∧ 0 < d) (h_bound : d < n / 2)
  (G : BipartiteGraph n)
  (h_degree : ∀ v, degree G v ≤ d) :
  ∃ G' : BipartiteGraph n,
    (∀ e ∈ G.edges, e ∈ G'.edges) ∧
    (∀ v, degree G' v = 2 * d) :=
sorry

end NUMINAMATH_CALUDE_bipartite_graph_completion_l1820_182070


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1820_182072

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 + 2*x₁ - 4 = 0) → 
  (x₂^2 + 2*x₂ - 4 = 0) → 
  (x₁ + x₂ = -2) := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1820_182072


namespace NUMINAMATH_CALUDE_constant_value_theorem_l1820_182063

/-- Given constants a and b, if f(x) = x^2 + 4x + 3 and f(ax + b) = x^2 + 10x + 24, then 5a - b = 2 -/
theorem constant_value_theorem (a b : ℝ) : 
  (∀ x, (x^2 + 4*x + 3 : ℝ) = ((a*x + b)^2 + 4*(a*x + b) + 3 : ℝ)) → 
  (∀ x, (x^2 + 4*x + 3 : ℝ) = (x^2 + 10*x + 24 : ℝ)) → 
  (5*a - b : ℝ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_constant_value_theorem_l1820_182063


namespace NUMINAMATH_CALUDE_remaining_oranges_l1820_182015

def initial_oranges : ℝ := 77.0
def eaten_oranges : ℝ := 2.0

theorem remaining_oranges : initial_oranges - eaten_oranges = 75.0 := by
  sorry

end NUMINAMATH_CALUDE_remaining_oranges_l1820_182015


namespace NUMINAMATH_CALUDE_symmetric_point_correct_l1820_182008

-- Define the original point
def original_point : ℝ × ℝ := (-1, 1)

-- Define the line of symmetry
def line_of_symmetry (x y : ℝ) : Prop := x - y - 1 = 0

-- Define the symmetric point
def symmetric_point : ℝ × ℝ := (2, -2)

-- Theorem statement
theorem symmetric_point_correct : 
  let (x₁, y₁) := original_point
  let (x₂, y₂) := symmetric_point
  (line_of_symmetry ((x₁ + x₂) / 2) ((y₁ + y₂) / 2) ∧ 
   (x₂ - x₁) = (y₂ - y₁) ∧
   (x₂ - x₁)^2 + (y₂ - y₁)^2 = 4 * ((x₁ - (x₁ + x₂) / 2)^2 + (y₁ - (y₁ + y₂) / 2)^2)) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_point_correct_l1820_182008


namespace NUMINAMATH_CALUDE_sum_of_first_four_terms_l1820_182014

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem sum_of_first_four_terms 
  (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a)
  (h_a2 : a 2 = 9)
  (h_a5 : a 5 = 243) :
  (a 1) + (a 2) + (a 3) + (a 4) = 120 :=
sorry

end NUMINAMATH_CALUDE_sum_of_first_four_terms_l1820_182014


namespace NUMINAMATH_CALUDE_equations_truth_l1820_182060

-- Define the theorem
theorem equations_truth :
  -- Equation 1
  (∀ a : ℝ, Real.sqrt ((a^2 + 1)^2) = a^2 + 1) ∧
  -- Equation 2
  (∀ a : ℝ, Real.sqrt (a^2) = abs a) ∧
  -- Equation 4
  (∀ x : ℝ, x ≥ 1 → Real.sqrt ((x + 1) * (x - 1)) = Real.sqrt (x + 1) * Real.sqrt (x - 1)) ∧
  -- Equation 3 (counterexample)
  (∃ a b : ℝ, Real.sqrt (a * b) ≠ Real.sqrt a * Real.sqrt b) :=
by
  sorry

end NUMINAMATH_CALUDE_equations_truth_l1820_182060


namespace NUMINAMATH_CALUDE_symmetric_point_about_origin_l1820_182034

/-- Given a point P with coordinates (2, -4), this theorem proves that its symmetric point
    about the origin has coordinates (-2, 4). -/
theorem symmetric_point_about_origin :
  let P : ℝ × ℝ := (2, -4)
  let symmetric_point : ℝ × ℝ := (-P.1, -P.2)
  symmetric_point = (-2, 4) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_about_origin_l1820_182034


namespace NUMINAMATH_CALUDE_triangle_sin_c_equals_one_l1820_182044

theorem triangle_sin_c_equals_one (a b c A B C : ℝ) : 
  a = 1 → 
  b = Real.sqrt 3 → 
  A + C = 2 * B → 
  0 < a ∧ 0 < b ∧ 0 < c → 
  0 < A ∧ 0 < B ∧ 0 < C → 
  A + B + C = π → 
  a / (Real.sin A) = b / (Real.sin B) → 
  a / (Real.sin A) = c / (Real.sin C) → 
  a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A → 
  b ^ 2 = a ^ 2 + c ^ 2 - 2 * a * c * Real.cos B → 
  c ^ 2 = a ^ 2 + b ^ 2 - 2 * a * b * Real.cos C → 
  Real.sin C = 1 := by
sorry

end NUMINAMATH_CALUDE_triangle_sin_c_equals_one_l1820_182044


namespace NUMINAMATH_CALUDE_three_diamonds_balance_six_dots_l1820_182029

-- Define the symbols
variable (triangle diamond dot : ℕ)

-- Define the balance relation
def balances (left right : ℕ) : Prop := left = right

-- State the given conditions
axiom balance1 : balances (4 * triangle + 2 * diamond) (12 * dot)
axiom balance2 : balances (2 * triangle) (diamond + 2 * dot)

-- State the theorem to be proved
theorem three_diamonds_balance_six_dots : balances (3 * diamond) (6 * dot) := by
  sorry

end NUMINAMATH_CALUDE_three_diamonds_balance_six_dots_l1820_182029


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1820_182045

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : 2 * x + y = 2) :
  ∀ z : ℝ, (1 / x + 1 / y) ≥ z → z ≤ 3 / 2 + Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1820_182045


namespace NUMINAMATH_CALUDE_solve_for_x_l1820_182099

-- Define the € operation
def euro (x y : ℝ) : ℝ := 2 * x * y

-- State the theorem
theorem solve_for_x (x : ℝ) : euro x (euro 4 5) = 480 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l1820_182099


namespace NUMINAMATH_CALUDE_sum_always_four_digits_l1820_182087

theorem sum_always_four_digits :
  ∀ (A B : ℕ), 1 ≤ A ∧ A ≤ 9 → 1 ≤ B ∧ B ≤ 9 →
  ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧ n = 7654 + (900 + 10 * A + 7) + (10 + B) :=
by sorry

end NUMINAMATH_CALUDE_sum_always_four_digits_l1820_182087


namespace NUMINAMATH_CALUDE_profit_percentage_previous_year_l1820_182056

/-- Given the following conditions for a company's finances over two years:
    1. In the previous year, profits were a percentage of revenues
    2. In 2009, revenues fell by 20%
    3. In 2009, profits were 20% of revenues
    4. Profits in 2009 were 160% of profits in the previous year
    
    This theorem proves that the percentage of profits to revenues in the previous year was 10%. -/
theorem profit_percentage_previous_year 
  (R : ℝ) -- Revenues in the previous year
  (P : ℝ) -- Profits in the previous year
  (h1 : P > 0) -- Ensure profits are positive
  (h2 : R > 0) -- Ensure revenues are positive
  (h3 : 0.8 * R * 0.2 = 1.6 * P) -- Condition relating 2009 profits to previous year
  : P / R = 0.1 := by
  sorry

#check profit_percentage_previous_year

end NUMINAMATH_CALUDE_profit_percentage_previous_year_l1820_182056


namespace NUMINAMATH_CALUDE_income_expenditure_ratio_l1820_182054

def income : ℕ := 17000
def savings : ℕ := 3400

theorem income_expenditure_ratio :
  let expenditure := income - savings
  (income / (income.gcd expenditure)) = 5 ∧
  (expenditure / (income.gcd expenditure)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_income_expenditure_ratio_l1820_182054


namespace NUMINAMATH_CALUDE_unique_four_digit_square_with_property_l1820_182074

theorem unique_four_digit_square_with_property : ∃! n : ℕ,
  (1000 ≤ n ∧ n ≤ 9999) ∧  -- four-digit number
  (∃ m : ℕ, n = m^2) ∧     -- perfect square
  (n / 100 = 3 * (n % 100) + 1) ∧  -- satisfies the equation
  n = 2809 := by
sorry

end NUMINAMATH_CALUDE_unique_four_digit_square_with_property_l1820_182074


namespace NUMINAMATH_CALUDE_fair_coin_five_flips_probability_l1820_182007

/-- Represents the probability of a specific outcome when flipping a fair coin n times -/
def coin_flip_probability (n : ℕ) (heads : Finset ℕ) : ℚ :=
  (1 / 2) ^ n

theorem fair_coin_five_flips_probability :
  coin_flip_probability 5 {0, 1} = 1 / 32 := by
  sorry

end NUMINAMATH_CALUDE_fair_coin_five_flips_probability_l1820_182007


namespace NUMINAMATH_CALUDE_g_neg_two_l1820_182049

def g (x : ℝ) : ℝ := x^3 - x^2 + x

theorem g_neg_two : g (-2) = -14 := by sorry

end NUMINAMATH_CALUDE_g_neg_two_l1820_182049


namespace NUMINAMATH_CALUDE_fifteen_fishers_tomorrow_l1820_182059

/-- Represents the fishing schedule in the coastal village -/
structure FishingSchedule where
  daily : ℕ
  everyOtherDay : ℕ
  everyThreeDay : ℕ
  yesterdayCount : ℕ
  todayCount : ℕ

/-- Calculates the number of people fishing tomorrow given the fishing schedule -/
def tomorrowFishers (schedule : FishingSchedule) : ℕ :=
  schedule.daily + schedule.everyThreeDay + (schedule.everyOtherDay - (schedule.yesterdayCount - schedule.daily))

/-- Theorem stating that given the specific fishing schedule, 15 people will fish tomorrow -/
theorem fifteen_fishers_tomorrow (schedule : FishingSchedule) 
  (h1 : schedule.daily = 7)
  (h2 : schedule.everyOtherDay = 8)
  (h3 : schedule.everyThreeDay = 3)
  (h4 : schedule.yesterdayCount = 12)
  (h5 : schedule.todayCount = 10) :
  tomorrowFishers schedule = 15 := by
  sorry

#eval tomorrowFishers { daily := 7, everyOtherDay := 8, everyThreeDay := 3, yesterdayCount := 12, todayCount := 10 }

end NUMINAMATH_CALUDE_fifteen_fishers_tomorrow_l1820_182059


namespace NUMINAMATH_CALUDE_regular_hexagon_perimeter_l1820_182081

/-- The perimeter of a regular hexagon given the distance between opposite sides -/
theorem regular_hexagon_perimeter (d : ℝ) (h : d = 15) : 
  let s := 2 * d / Real.sqrt 3
  6 * s = 60 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_regular_hexagon_perimeter_l1820_182081
