import Mathlib

namespace NUMINAMATH_CALUDE_two_talent_students_l2579_257919

theorem two_talent_students (total : ℕ) (all_three : ℕ) (cant_sing : ℕ) (cant_dance : ℕ) (cant_act : ℕ) : 
  total = 150 →
  all_three = 10 →
  cant_sing = 70 →
  cant_dance = 90 →
  cant_act = 50 →
  ∃ (two_talents : ℕ), two_talents = 80 ∧ 
    (total - cant_sing) + (total - cant_dance) + (total - cant_act) - two_talents - 2 * all_three = total :=
by sorry

end NUMINAMATH_CALUDE_two_talent_students_l2579_257919


namespace NUMINAMATH_CALUDE_craig_walking_distance_l2579_257975

/-- The distance Craig rode on the bus in miles -/
def bus_distance : ℝ := 3.83

/-- The difference between the bus distance and walking distance in miles -/
def distance_difference : ℝ := 3.67

/-- The distance Craig walked in miles -/
def walking_distance : ℝ := bus_distance - distance_difference

theorem craig_walking_distance : walking_distance = 0.16 := by
  sorry

end NUMINAMATH_CALUDE_craig_walking_distance_l2579_257975


namespace NUMINAMATH_CALUDE_alissa_presents_l2579_257979

theorem alissa_presents (ethan_presents : ℕ) (alissa_additional : ℕ) :
  ethan_presents = 31 →
  alissa_additional = 22 →
  ethan_presents + alissa_additional = 53 :=
by sorry

end NUMINAMATH_CALUDE_alissa_presents_l2579_257979


namespace NUMINAMATH_CALUDE_descending_order_xy_xy2_x_l2579_257991

theorem descending_order_xy_xy2_x
  (x y : ℝ)
  (hx : x < 0)
  (hy : -1 < y ∧ y < 0) :
  xy > xy^2 ∧ xy^2 > x :=
by sorry

end NUMINAMATH_CALUDE_descending_order_xy_xy2_x_l2579_257991


namespace NUMINAMATH_CALUDE_solution_volume_l2579_257965

/-- Given a solution with 1.5 liters of pure acid and a concentration of 30%,
    prove that the total volume of the solution is 5 liters. -/
theorem solution_volume (volume_acid : ℝ) (concentration : ℝ) :
  volume_acid = 1.5 →
  concentration = 0.30 →
  (volume_acid / concentration) = 5 := by
  sorry

end NUMINAMATH_CALUDE_solution_volume_l2579_257965


namespace NUMINAMATH_CALUDE_sequence_a_property_sequence_a_formula_l2579_257926

def sequence_a : ℕ → ℚ
  | 0 => 1
  | 1 => 2
  | (n + 2) => sorry

theorem sequence_a_property (n : ℕ) (h : n ≥ 1) :
  n * (n + 1) * sequence_a (n + 1) = n * (n - 1) * sequence_a n - (n - 2) * sequence_a (n - 1) := by sorry

theorem sequence_a_formula (n : ℕ) (h : n ≥ 2) : sequence_a n = 1 / n.factorial := by sorry

end NUMINAMATH_CALUDE_sequence_a_property_sequence_a_formula_l2579_257926


namespace NUMINAMATH_CALUDE_secret_reaches_2186_l2579_257954

def secret_spread (day : ℕ) : ℕ :=
  if day = 0 then 1
  else secret_spread (day - 1) + 3^day

theorem secret_reaches_2186 :
  ∃ d : ℕ, d ≤ 7 ∧ secret_spread d ≥ 2186 :=
by sorry

end NUMINAMATH_CALUDE_secret_reaches_2186_l2579_257954


namespace NUMINAMATH_CALUDE_exponent_multiplication_l2579_257934

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l2579_257934


namespace NUMINAMATH_CALUDE_trapezoid_has_two_heights_l2579_257949

/-- A trapezoid is a quadrilateral with at least one pair of parallel sides. -/
structure Trapezoid where
  vertices : Fin 4 → ℝ × ℝ
  parallel_sides : ∃ (i j : Fin 4), i ≠ j ∧ (vertices i).1 = (vertices j).1

/-- The number of heights in a trapezoid -/
def num_heights (t : Trapezoid) : ℕ := 2

/-- Theorem: A trapezoid has exactly 2 heights -/
theorem trapezoid_has_two_heights (t : Trapezoid) : num_heights t = 2 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_has_two_heights_l2579_257949


namespace NUMINAMATH_CALUDE_cookies_received_l2579_257904

theorem cookies_received (brother sister cousin self : ℕ) 
  (h1 : brother = 12)
  (h2 : sister = 9)
  (h3 : cousin = 7)
  (h4 : self = 17) :
  brother + sister + cousin + self = 45 := by
  sorry

end NUMINAMATH_CALUDE_cookies_received_l2579_257904


namespace NUMINAMATH_CALUDE_article_price_decrease_l2579_257931

theorem article_price_decrease (P : ℝ) : 
  (P * (1 - 0.24) * (1 - 0.10) = 760) → 
  ∃ ε > 0, |P - 111| < ε :=
sorry

end NUMINAMATH_CALUDE_article_price_decrease_l2579_257931


namespace NUMINAMATH_CALUDE_new_person_weight_l2579_257995

theorem new_person_weight (initial_count : ℕ) (replaced_weight : ℝ) (avg_increase : ℝ) :
  initial_count = 4 →
  replaced_weight = 70 →
  avg_increase = 3 →
  (initial_count : ℝ) * avg_increase + replaced_weight = 82 := by
  sorry

#check new_person_weight

end NUMINAMATH_CALUDE_new_person_weight_l2579_257995


namespace NUMINAMATH_CALUDE_quadratic_transformation_l2579_257987

theorem quadratic_transformation (x : ℝ) :
  (x^2 - 4*x + 3 = 0) → (∃ h k : ℝ, (x + h)^2 = k ∧ k = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l2579_257987


namespace NUMINAMATH_CALUDE_opposite_reciprocal_abs_one_result_l2579_257989

theorem opposite_reciprocal_abs_one_result (a b c d m : ℝ) : 
  (a = -b) → 
  (c * d = 1) → 
  (|m| = 1) → 
  ((a + b) * c * d - 2009 * m = -2009 ∨ (a + b) * c * d - 2009 * m = 2009) :=
by sorry

end NUMINAMATH_CALUDE_opposite_reciprocal_abs_one_result_l2579_257989


namespace NUMINAMATH_CALUDE_reflection_line_sum_l2579_257940

/-- Given a line y = mx + b, where the point (2,3) is reflected to (8,7) across this line,
    prove that m + b = 9.5 -/
theorem reflection_line_sum (m b : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ = 2 ∧ y₁ = 3 ∧ x₂ = 8 ∧ y₂ = 7 ∧
    (y₂ - y₁) / (x₂ - x₁) = -1 / m ∧
    ((x₁ + x₂) / 2, (y₁ + y₂) / 2) ∈ {(x, y) | y = m * x + b}) →
  m + b = 9.5 := by
sorry


end NUMINAMATH_CALUDE_reflection_line_sum_l2579_257940


namespace NUMINAMATH_CALUDE_commission_change_point_l2579_257938

/-- The sales amount where the commission rate changes -/
def X : ℝ := 1822.98

/-- The total sales amount -/
def total_sales : ℝ := 15885.42

/-- The amount remitted to the parent company -/
def remitted_amount : ℝ := 15000

/-- The commission rate for sales up to X -/
def low_rate : ℝ := 0.10

/-- The commission rate for sales exceeding X -/
def high_rate : ℝ := 0.05

theorem commission_change_point : 
  X * low_rate + (total_sales - X) * high_rate = total_sales - remitted_amount :=
sorry

end NUMINAMATH_CALUDE_commission_change_point_l2579_257938


namespace NUMINAMATH_CALUDE_sin_eleven_pi_thirds_l2579_257913

theorem sin_eleven_pi_thirds : Real.sin (11 * π / 3) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_eleven_pi_thirds_l2579_257913


namespace NUMINAMATH_CALUDE_parallel_perpendicular_implication_l2579_257910

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem parallel_perpendicular_implication 
  (m n : Line) (a : Plane) : 
  parallel m n → perpendicular m a → perpendicular n a :=
sorry

end NUMINAMATH_CALUDE_parallel_perpendicular_implication_l2579_257910


namespace NUMINAMATH_CALUDE_rubber_boat_lost_at_4pm_l2579_257903

/-- Represents the time when the rubber boat was lost (in hours before 5 PM) -/
def time_lost : ℝ := 1

/-- Represents the speed of the ship in still water -/
def ship_speed : ℝ := 1

/-- Represents the speed of the river flow -/
def river_speed : ℝ := 1

/-- Theorem stating that the rubber boat was lost at 4 PM -/
theorem rubber_boat_lost_at_4pm :
  (5 - time_lost) * (ship_speed - river_speed) + (6 - time_lost) * river_speed = ship_speed + river_speed :=
by sorry

end NUMINAMATH_CALUDE_rubber_boat_lost_at_4pm_l2579_257903


namespace NUMINAMATH_CALUDE_handmade_ornaments_excess_l2579_257956

/-- Proves that the number of handmade ornaments exceeds 1/6 of the total ornaments by 20 -/
theorem handmade_ornaments_excess (total : ℕ) (handmade : ℕ) (antique : ℕ) : 
  total = 60 →
  3 * antique = total →
  2 * antique = handmade →
  antique = 20 →
  handmade - (total / 6) = 20 := by
  sorry

end NUMINAMATH_CALUDE_handmade_ornaments_excess_l2579_257956


namespace NUMINAMATH_CALUDE_equivalent_operation_l2579_257973

theorem equivalent_operation (x : ℝ) : 
  (x * (2/5)) / (3/7) = x * (14/15) := by
  sorry

end NUMINAMATH_CALUDE_equivalent_operation_l2579_257973


namespace NUMINAMATH_CALUDE_tessa_initial_apples_l2579_257946

/-- The initial number of apples Tessa had -/
def initial_apples : ℝ := sorry

/-- The number of apples Anita gives to Tessa -/
def apples_from_anita : ℝ := 5.0

/-- The number of apples needed to make a pie -/
def apples_for_pie : ℝ := 4.0

/-- The number of apples left after making the pie -/
def apples_left : ℝ := 11

/-- Theorem stating that Tessa initially had 10 apples -/
theorem tessa_initial_apples : 
  initial_apples + apples_from_anita - apples_for_pie = apples_left ∧ 
  initial_apples = 10 := by sorry

end NUMINAMATH_CALUDE_tessa_initial_apples_l2579_257946


namespace NUMINAMATH_CALUDE_cube_split_with_2023_l2579_257996

theorem cube_split_with_2023 (m : ℕ) (h1 : m > 1) : 
  (∃ (k : ℕ), 2 * k + 1 = 2023 ∧ 
   k ≥ (m + 2) * (m - 1) / 2 - m + 1 ∧ 
   k < (m + 2) * (m - 1) / 2 + 1) → 
  m = 45 := by
sorry

end NUMINAMATH_CALUDE_cube_split_with_2023_l2579_257996


namespace NUMINAMATH_CALUDE_sqrt_equation_solutions_l2579_257984

theorem sqrt_equation_solutions :
  ∀ x : ℝ, (Real.sqrt (9 * x - 4) + 15 / Real.sqrt (9 * x - 4) = 8) ↔ (x = 29 / 9 ∨ x = 13 / 9) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solutions_l2579_257984


namespace NUMINAMATH_CALUDE_weight_of_replaced_person_l2579_257978

/-- Given a group of 6 persons, if replacing one person with a new person weighing 74 kg
    increases the average weight by 1.5 kg, then the weight of the person being replaced is 65 kg. -/
theorem weight_of_replaced_person (group_size : ℕ) (new_person_weight : ℝ) (average_increase : ℝ) :
  group_size = 6 →
  new_person_weight = 74 →
  average_increase = 1.5 →
  ∃ (original_average : ℝ) (replaced_person_weight : ℝ),
    group_size * (original_average + average_increase) =
    group_size * original_average - replaced_person_weight + new_person_weight ∧
    replaced_person_weight = 65 :=
by sorry

end NUMINAMATH_CALUDE_weight_of_replaced_person_l2579_257978


namespace NUMINAMATH_CALUDE_largest_n_base_7_double_l2579_257907

def to_base_7 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
  aux n []

def from_base_7 (digits : List ℕ) : ℕ :=
  digits.foldl (fun acc d => acc * 7 + d) 0

theorem largest_n_base_7_double : ∀ n : ℕ, n > 156 → 2 * n ≠ from_base_7 (to_base_7 n) :=
sorry

end NUMINAMATH_CALUDE_largest_n_base_7_double_l2579_257907


namespace NUMINAMATH_CALUDE_sequence_existence_l2579_257960

theorem sequence_existence (a b : ℤ) (ha : a > 2) (hb : b > 2) :
  ∃ (k : ℕ) (n : ℕ → ℤ), 
    n 1 = a ∧ 
    n k = b ∧ 
    (∀ i : ℕ, 1 ≤ i ∧ i < k → (n i + n (i + 1)) ∣ (n i * n (i + 1))) :=
sorry

end NUMINAMATH_CALUDE_sequence_existence_l2579_257960


namespace NUMINAMATH_CALUDE_coin_stack_arrangements_l2579_257937

/-- Represents a coin with a color and a face side -/
inductive Coin
  | Gold : Bool → Coin
  | Silver : Bool → Coin

/-- A stack of coins -/
def CoinStack := List Coin

/-- Checks if two adjacent coins are not face to face -/
def validAdjacent : Coin → Coin → Bool
  | Coin.Gold true, Coin.Gold true => false
  | Coin.Gold true, Coin.Silver true => false
  | Coin.Silver true, Coin.Gold true => false
  | Coin.Silver true, Coin.Silver true => false
  | _, _ => true

/-- Checks if a stack of coins is valid (no adjacent face to face) -/
def validStack : CoinStack → Bool
  | [] => true
  | [_] => true
  | (c1 :: c2 :: rest) => validAdjacent c1 c2 && validStack (c2 :: rest)

/-- Counts the number of gold coins in a stack -/
def countGold : CoinStack → Nat
  | [] => 0
  | (Coin.Gold _) :: rest => 1 + countGold rest
  | _ :: rest => countGold rest

/-- Counts the number of silver coins in a stack -/
def countSilver : CoinStack → Nat
  | [] => 0
  | (Coin.Silver _) :: rest => 1 + countSilver rest
  | _ :: rest => countSilver rest

/-- The main theorem to prove -/
theorem coin_stack_arrangements :
  (∃ (validStacks : List CoinStack),
    (∀ s ∈ validStacks, validStack s = true) ∧
    (∀ s ∈ validStacks, countGold s = 5) ∧
    (∀ s ∈ validStacks, countSilver s = 5) ∧
    validStacks.length = 2772) := by
  sorry

end NUMINAMATH_CALUDE_coin_stack_arrangements_l2579_257937


namespace NUMINAMATH_CALUDE_second_day_study_hours_l2579_257970

/-- Represents the relationship between study hours and performance score for a given day -/
structure StudyDay where
  hours : ℝ
  score : ℝ

/-- The constant product of hours and score, representing the inverse relationship -/
def inverse_constant (day : StudyDay) : ℝ := day.hours * day.score

theorem second_day_study_hours 
  (day1 : StudyDay)
  (avg_score : ℝ)
  (h1 : day1.hours = 5)
  (h2 : day1.score = 80)
  (h3 : avg_score = 85) :
  ∃ (day2 : StudyDay), 
    inverse_constant day1 = inverse_constant day2 ∧
    (day1.score + day2.score) / 2 = avg_score ∧
    day2.hours = 40 / 9 := by
  sorry

end NUMINAMATH_CALUDE_second_day_study_hours_l2579_257970


namespace NUMINAMATH_CALUDE_johns_share_l2579_257920

theorem johns_share (total_amount : ℕ) (john_ratio jose_ratio binoy_ratio : ℕ) 
  (h1 : total_amount = 6000)
  (h2 : john_ratio = 2)
  (h3 : jose_ratio = 4)
  (h4 : binoy_ratio = 6) :
  (john_ratio : ℚ) / (john_ratio + jose_ratio + binoy_ratio : ℚ) * total_amount = 1000 :=
by sorry

end NUMINAMATH_CALUDE_johns_share_l2579_257920


namespace NUMINAMATH_CALUDE_cube_vertex_shapes_l2579_257963

-- Define a cube
structure Cube where
  vertices : Finset (Fin 8)
  edges : Finset (Fin 8 × Fin 8)
  faces : Finset (Finset (Fin 8))

-- Define the types of shapes we're interested in
inductive ShapeType
  | Rectangle
  | NonRectangleParallelogram
  | IsoscelesRightTetrahedron
  | RegularTetrahedron
  | RightTetrahedron

-- Function to check if 4 vertices form a specific shape
def formsShape (c : Cube) (v : Finset (Fin 8)) (s : ShapeType) : Prop :=
  v.card = 4 ∧ v ⊆ c.vertices ∧ match s with
    | ShapeType.Rectangle => sorry
    | ShapeType.NonRectangleParallelogram => sorry
    | ShapeType.IsoscelesRightTetrahedron => sorry
    | ShapeType.RegularTetrahedron => sorry
    | ShapeType.RightTetrahedron => sorry

-- Theorem statement
theorem cube_vertex_shapes (c : Cube) :
  (∃ v, formsShape c v ShapeType.Rectangle) ∧
  (∃ v, formsShape c v ShapeType.IsoscelesRightTetrahedron) ∧
  (∃ v, formsShape c v ShapeType.RegularTetrahedron) ∧
  (∃ v, formsShape c v ShapeType.RightTetrahedron) ∧
  (¬ ∃ v, formsShape c v ShapeType.NonRectangleParallelogram) :=
sorry

end NUMINAMATH_CALUDE_cube_vertex_shapes_l2579_257963


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2579_257976

theorem quadratic_inequality_solution (x : ℝ) :
  (x^2 + 3*x < 10) ↔ (-5 < x ∧ x < 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2579_257976


namespace NUMINAMATH_CALUDE_second_player_wins_l2579_257983

/-- Represents the state of the game board as a list of integers -/
def GameBoard := List Nat

/-- The initial game board with 2022 ones -/
def initialBoard : GameBoard := List.replicate 2022 1

/-- A player in the game -/
inductive Player
| First
| Second

/-- The result of a game -/
inductive GameResult
| FirstWin
| SecondWin
| Draw

/-- A move in the game, represented by the index of the first number to be replaced -/
def Move := Nat

/-- Apply a move to the game board -/
def applyMove (board : GameBoard) (move : Move) : GameBoard :=
  sorry

/-- Check if a player has won -/
def hasWon (board : GameBoard) : Bool :=
  sorry

/-- Check if the game is a draw -/
def isDraw (board : GameBoard) : Bool :=
  sorry

/-- A strategy for a player -/
def Strategy := GameBoard → Move

/-- The second player's strategy -/
def secondPlayerStrategy : Strategy :=
  sorry

/-- The game result when both players play optimally -/
def gameResult (firstStrategy secondStrategy : Strategy) : GameResult :=
  sorry

/-- Theorem stating that the second player has a winning strategy -/
theorem second_player_wins :
  ∃ (secondStrategy : Strategy),
    ∀ (firstStrategy : Strategy),
      gameResult firstStrategy secondStrategy = GameResult.SecondWin :=
sorry

end NUMINAMATH_CALUDE_second_player_wins_l2579_257983


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l2579_257929

/-- The perimeter of a rhombus with diagonals of lengths 72 and 30 is 156 -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 72) (h2 : d2 = 30) : 
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 156 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l2579_257929


namespace NUMINAMATH_CALUDE_prime_quadratic_l2579_257981

theorem prime_quadratic (a : ℕ) : 
  Nat.Prime (a^2 - 10*a + 21) ↔ a = 2 ∨ a = 8 := by sorry

end NUMINAMATH_CALUDE_prime_quadratic_l2579_257981


namespace NUMINAMATH_CALUDE_multiply_23_by_4_l2579_257902

theorem multiply_23_by_4 : 23 * 4 = 20 * 4 + 3 * 4 := by
  sorry

end NUMINAMATH_CALUDE_multiply_23_by_4_l2579_257902


namespace NUMINAMATH_CALUDE_school_gender_difference_l2579_257962

theorem school_gender_difference (initial_girls boys additional_girls : ℕ) 
  (h1 : initial_girls = 632)
  (h2 : boys = 410)
  (h3 : additional_girls = 465) :
  initial_girls + additional_girls - boys = 687 := by
  sorry

end NUMINAMATH_CALUDE_school_gender_difference_l2579_257962


namespace NUMINAMATH_CALUDE_base_conversion_problem_l2579_257966

-- Define a function to convert a number from base n to decimal
def to_decimal (digits : List Nat) (n : Nat) : Nat :=
  digits.enum.foldr (fun (i, digit) acc => acc + digit * n ^ i) 0

-- Define the problem statement
theorem base_conversion_problem (n : Nat) (d : Nat) :
  n > 0 →  -- n is a positive integer
  d < 10 →  -- d is a digit
  to_decimal [4, 5, d] n = 392 →  -- 45d in base n equals 392
  to_decimal [4, 5, 7] n = to_decimal [2, 1, d, 5] 7 →  -- 457 in base n equals 21d5 in base 7
  n + d = 12 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_problem_l2579_257966


namespace NUMINAMATH_CALUDE_average_weight_increase_l2579_257930

theorem average_weight_increase (initial_count : ℕ) (old_weight new_weight : ℝ) :
  initial_count = 10 →
  old_weight = 65 →
  new_weight = 90 →
  (new_weight - old_weight) / initial_count = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_increase_l2579_257930


namespace NUMINAMATH_CALUDE_smallest_equal_hotdogs_and_buns_l2579_257943

theorem smallest_equal_hotdogs_and_buns :
  (∃ n : ℕ+, ∀ k : ℕ+, (∃ m : ℕ+, 6 * k = 8 * m) → n ≤ k) ∧
  (∃ m : ℕ+, 6 * 4 = 8 * m) :=
by sorry

end NUMINAMATH_CALUDE_smallest_equal_hotdogs_and_buns_l2579_257943


namespace NUMINAMATH_CALUDE_rectangle_area_l2579_257914

/-- Rectangle PQRS with given coordinates and properties -/
structure Rectangle where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  S : ℝ × ℝ
  is_rectangle : Bool

/-- The area of the rectangle PQRS is 200000 -/
theorem rectangle_area (rect : Rectangle) : 
  rect.P = (-15, 30) →
  rect.Q = (985, 230) →
  rect.S.1 = -13 →
  rect.is_rectangle = true →
  (rect.Q.1 - rect.P.1) * (rect.S.2 - rect.P.2) = 200000 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_area_l2579_257914


namespace NUMINAMATH_CALUDE_jeopardy_episode_length_l2579_257982

/-- The length of one episode of Jeopardy in minutes -/
def jeopardy_length : ℝ := sorry

/-- The length of one episode of Wheel of Fortune in minutes -/
def wheel_of_fortune_length : ℝ := sorry

/-- The total number of episodes James watched -/
def total_episodes : ℕ := sorry

/-- The total time James spent watching TV in minutes -/
def total_watch_time : ℝ := sorry

theorem jeopardy_episode_length :
  jeopardy_length = 20 ∧
  wheel_of_fortune_length = 2 * jeopardy_length ∧
  total_episodes = 4 ∧
  total_watch_time = 120 ∧
  total_watch_time = 2 * jeopardy_length + 2 * wheel_of_fortune_length :=
by sorry

end NUMINAMATH_CALUDE_jeopardy_episode_length_l2579_257982


namespace NUMINAMATH_CALUDE_smallest_cube_root_with_small_fraction_l2579_257972

theorem smallest_cube_root_with_small_fraction : 
  ∃ (m : ℕ) (r : ℝ), 
    0 < r ∧ r < 1/2000 ∧ 
    (∃ (n : ℕ), n > 0 ∧ m = (n + r)^3) ∧
    (∀ (k : ℕ) (s : ℝ), 0 < k ∧ k < 26 → 0 < s ∧ s < 1/2000 → ¬(∃ (l : ℕ), l = (k + s)^3)) ∧
    (∃ (n : ℕ) (r : ℝ), n = 26 ∧ 0 < r ∧ r < 1/2000 ∧ (∃ (m : ℕ), m = (n + r)^3)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_cube_root_with_small_fraction_l2579_257972


namespace NUMINAMATH_CALUDE_vector_perpendicular_condition_l2579_257992

theorem vector_perpendicular_condition (a b : ℝ × ℝ) (m : ℝ) : 
  ‖a‖ = Real.sqrt 3 →
  ‖b‖ = 2 →
  (a.1 * b.1 + a.2 * b.2) / (‖a‖ * ‖b‖) = Real.cos (π / 6) →
  (a.1 - m * b.1) * a.1 + (a.2 - m * b.2) * a.2 = 0 →
  m = 1 := by
  sorry

end NUMINAMATH_CALUDE_vector_perpendicular_condition_l2579_257992


namespace NUMINAMATH_CALUDE_carl_weekly_earnings_l2579_257935

/-- Represents Carl's earnings and candy bar purchases over 4 weeks -/
structure CarlEarnings where
  weeks : ℕ
  candyBars : ℕ
  candyBarPrice : ℚ
  weeklyEarnings : ℚ

/-- Theorem stating that Carl's weekly earnings are $0.75 given the conditions -/
theorem carl_weekly_earnings (e : CarlEarnings) 
  (h_weeks : e.weeks = 4)
  (h_candyBars : e.candyBars = 6)
  (h_candyBarPrice : e.candyBarPrice = 1/2) :
  e.weeklyEarnings = 3/4 := by
sorry

end NUMINAMATH_CALUDE_carl_weekly_earnings_l2579_257935


namespace NUMINAMATH_CALUDE_rectangle_area_is_100_l2579_257923

-- Define the rectangle
def Rectangle (width : ℝ) (length : ℝ) : Type :=
  { w : ℝ // w = width } × { l : ℝ // l = length }

-- Define the properties of the rectangle
def rectangle_properties (r : Rectangle 5 (4 * 5)) : Prop :=
  r.2.1 = 4 * r.1.1

-- Define the area of a rectangle
def area (r : Rectangle 5 (4 * 5)) : ℝ :=
  r.1.1 * r.2.1

-- Theorem statement
theorem rectangle_area_is_100 (r : Rectangle 5 (4 * 5)) 
  (h : rectangle_properties r) : area r = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_is_100_l2579_257923


namespace NUMINAMATH_CALUDE_initial_inventory_l2579_257932

def bookshop_inventory (initial_books : ℕ) : Prop :=
  let saturday_instore := 37
  let saturday_online := 128
  let sunday_instore := 2 * saturday_instore
  let sunday_online := saturday_online + 34
  let shipment := 160
  let current_books := 502
  initial_books = current_books + saturday_instore + saturday_online + sunday_instore + sunday_online - shipment

theorem initial_inventory : ∃ (x : ℕ), bookshop_inventory x ∧ x = 743 := by
  sorry

end NUMINAMATH_CALUDE_initial_inventory_l2579_257932


namespace NUMINAMATH_CALUDE_positive_number_square_sum_l2579_257950

theorem positive_number_square_sum (n : ℝ) : n > 0 → n^2 + n = 245 → n = 14 := by
  sorry

end NUMINAMATH_CALUDE_positive_number_square_sum_l2579_257950


namespace NUMINAMATH_CALUDE_circle_symmetry_l2579_257917

-- Define the original circle
def original_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 1 = 0

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop :=
  2*x - y + 3 = 0

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop :=
  (x+3)^2 + (y-2)^2 = 2

-- Theorem statement
theorem circle_symmetry :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
  original_circle x₁ y₁ →
  symmetric_circle x₂ y₂ →
  ∃ (x_m y_m : ℝ),
    symmetry_line x_m y_m ∧
    x_m = (x₁ + x₂) / 2 ∧
    y_m = (y₁ + y₂) / 2 :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l2579_257917


namespace NUMINAMATH_CALUDE_triangle_problem_l2579_257999

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  3 * a * Real.cos C = 2 * c * Real.cos A →
  b = 2 * Real.sqrt 5 →
  c = 3 →
  (a = Real.sqrt 5 ∧
   Real.sin (B + π / 4) = Real.sqrt 10 / 10) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l2579_257999


namespace NUMINAMATH_CALUDE_least_seven_digit_binary_l2579_257968

theorem least_seven_digit_binary : ∀ n : ℕ, 
  (n < 64 → (Nat.log2 n).succ < 7) ∧ 
  ((Nat.log2 64).succ = 7) :=
sorry

end NUMINAMATH_CALUDE_least_seven_digit_binary_l2579_257968


namespace NUMINAMATH_CALUDE_curve_C_symmetry_l2579_257967

/-- The curve C in the Cartesian coordinate system -/
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ((p.1 - 1)^2 + p.2^2) * ((p.1 + 1)^2 + p.2^2) = 4}

/-- A point is symmetric about the x-axis -/
def symmetric_x (p : ℝ × ℝ) : Prop := (p.1, -p.2) ∈ C ↔ p ∈ C

/-- A point is symmetric about the y-axis -/
def symmetric_y (p : ℝ × ℝ) : Prop := (-p.1, p.2) ∈ C ↔ p ∈ C

/-- A point is symmetric about the origin -/
def symmetric_origin (p : ℝ × ℝ) : Prop := (-p.1, -p.2) ∈ C ↔ p ∈ C

theorem curve_C_symmetry :
  (∀ p ∈ C, symmetric_x p ∧ symmetric_y p) ∧
  (∀ p ∈ C, symmetric_origin p) := by sorry

end NUMINAMATH_CALUDE_curve_C_symmetry_l2579_257967


namespace NUMINAMATH_CALUDE_triangle_area_l2579_257936

/-- The area of a triangle with sides 5, 4, and 4 units is (5√39)/4 square units. -/
theorem triangle_area (a b c : ℝ) (h1 : a = 5) (h2 : b = 4) (h3 : c = 4) :
  (1/2 : ℝ) * a * (((b^2 - (a/2)^2).sqrt : ℝ)) = (5 * Real.sqrt 39) / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2579_257936


namespace NUMINAMATH_CALUDE_cube_of_0_09_times_0_0007_l2579_257921

theorem cube_of_0_09_times_0_0007 : (0.09 : ℝ)^3 * 0.0007 = 0.0000005103 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_0_09_times_0_0007_l2579_257921


namespace NUMINAMATH_CALUDE_todays_production_l2579_257958

theorem todays_production (n : ℕ) (past_average : ℝ) (new_average : ℝ) 
  (h1 : n = 9)
  (h2 : past_average = 50)
  (h3 : new_average = 54) :
  (n + 1) * new_average - n * past_average = 90 := by
  sorry

end NUMINAMATH_CALUDE_todays_production_l2579_257958


namespace NUMINAMATH_CALUDE_c_less_than_a_l2579_257986

theorem c_less_than_a (a b c : ℝ) (ha : a < 0) (hb : b < 0) (hc : c < 0)
  (h1 : c / (a + b) = 2) (h2 : c / (b - a) = 3) : c < a := by
  sorry

end NUMINAMATH_CALUDE_c_less_than_a_l2579_257986


namespace NUMINAMATH_CALUDE_specific_trapezoid_area_l2579_257977

/-- Represents a circumscribed isosceles trapezoid -/
structure CircumscribedIsoscelesTrapezoid where
  long_base : ℝ
  base_angle : ℝ

/-- Calculates the area of a circumscribed isosceles trapezoid -/
def area (t : CircumscribedIsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem stating that the area of the specific trapezoid is 84 -/
theorem specific_trapezoid_area :
  let t : CircumscribedIsoscelesTrapezoid := {
    long_base := 24,
    base_angle := Real.arcsin 0.6
  }
  area t = 84 := by sorry

end NUMINAMATH_CALUDE_specific_trapezoid_area_l2579_257977


namespace NUMINAMATH_CALUDE_die_roll_probability_l2579_257994

/-- A fair six-sided die is rolled six times. -/
def num_rolls : ℕ := 6

/-- The probability of rolling a 5 or 6 on a fair six-sided die. -/
def prob_success : ℚ := 1/3

/-- The probability of not rolling a 5 or 6 on a fair six-sided die. -/
def prob_failure : ℚ := 1 - prob_success

/-- The number of successful outcomes we're interested in (at least 5 times). -/
def min_successes : ℕ := 5

/-- Calculates the binomial coefficient (n choose k). -/
def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

/-- Calculates the probability of exactly k successes in n trials. -/
def prob_exactly (n k : ℕ) (p : ℚ) : ℚ :=
  (binomial_coefficient n k : ℚ) * p^k * (1-p)^(n-k)

/-- The main theorem to prove. -/
theorem die_roll_probability : 
  prob_exactly num_rolls min_successes prob_success + 
  prob_exactly num_rolls num_rolls prob_success = 13/729 := by
  sorry

end NUMINAMATH_CALUDE_die_roll_probability_l2579_257994


namespace NUMINAMATH_CALUDE_x_values_when_two_in_set_l2579_257952

theorem x_values_when_two_in_set (x : ℝ) : 2 ∈ ({1, x^2 + x} : Set ℝ) → x = 1 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_x_values_when_two_in_set_l2579_257952


namespace NUMINAMATH_CALUDE_gcd_30_and_70_to_80_l2579_257927

theorem gcd_30_and_70_to_80 : ∃! n : ℕ, 70 ≤ n ∧ n ≤ 80 ∧ Nat.gcd 30 n = 10 := by sorry

end NUMINAMATH_CALUDE_gcd_30_and_70_to_80_l2579_257927


namespace NUMINAMATH_CALUDE_product_75_180_trailing_zeros_l2579_257925

/-- The number of trailing zeros in the product of two positive integers -/
def trailingZeros (a b : ℕ+) : ℕ :=
  sorry

/-- Theorem: The number of trailing zeros in the product of 75 and 180 is 2 -/
theorem product_75_180_trailing_zeros :
  trailingZeros 75 180 = 2 := by
  sorry

end NUMINAMATH_CALUDE_product_75_180_trailing_zeros_l2579_257925


namespace NUMINAMATH_CALUDE_log_problem_l2579_257928

theorem log_problem (y : ℝ) (h : y = (Real.log 16 / Real.log 4) ^ (Real.log 4 / Real.log 16)) :
  Real.log y / Real.log 12 = 1 / (4 + 2 * Real.log 3 / Real.log 2) := by
  sorry

end NUMINAMATH_CALUDE_log_problem_l2579_257928


namespace NUMINAMATH_CALUDE_inverse_sum_equals_negative_six_l2579_257945

-- Define the function f(x) = x|x|
def f (x : ℝ) : ℝ := x * |x|

-- Define the inverse function f^(-1)
noncomputable def f_inv (y : ℝ) : ℝ :=
  if y ≥ 0 then Real.sqrt y else -Real.sqrt (-y)

-- Theorem statement
theorem inverse_sum_equals_negative_six :
  f_inv 9 + f_inv (-81) = -6 := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_equals_negative_six_l2579_257945


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_specific_root_implies_m_and_other_root_l2579_257948

/-- Given a quadratic equation x^2 + 2x - (m-2) = 0 with real roots -/
def quadratic_equation (x m : ℝ) : Prop := x^2 + 2*x - (m-2) = 0

/-- The discriminant of the quadratic equation is non-negative -/
def has_real_roots (m : ℝ) : Prop := 4*m - 4 ≥ 0

theorem quadratic_real_roots_condition (m : ℝ) :
  has_real_roots m ↔ m ≥ 1 := by sorry

theorem specific_root_implies_m_and_other_root :
  ∀ m : ℝ, quadratic_equation 1 m → m = 3 ∧ quadratic_equation (-3) m := by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_specific_root_implies_m_and_other_root_l2579_257948


namespace NUMINAMATH_CALUDE_green_ball_probability_l2579_257944

structure Container where
  red : ℕ
  green : ℕ

def Set1 : List Container := [
  ⟨2, 8⟩,  -- Container A
  ⟨8, 2⟩,  -- Container B
  ⟨8, 2⟩   -- Container C
]

def Set2 : List Container := [
  ⟨8, 2⟩,  -- Container A
  ⟨2, 8⟩,  -- Container B
  ⟨2, 8⟩   -- Container C
]

def probability_green (set : List Container) : ℚ :=
  let total_balls (c : Container) := c.red + c.green
  let green_prob (c : Container) := c.green / (total_balls c)
  (set.map green_prob).sum / set.length

theorem green_ball_probability :
  (1 / 2 : ℚ) * probability_green Set1 + (1 / 2 : ℚ) * probability_green Set2 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_green_ball_probability_l2579_257944


namespace NUMINAMATH_CALUDE_weight_on_switch_l2579_257916

theorem weight_on_switch (total_weight : ℕ) (additional_weight : ℕ) 
  (h1 : total_weight = 712)
  (h2 : additional_weight = 478) :
  total_weight - additional_weight = 234 := by
  sorry

end NUMINAMATH_CALUDE_weight_on_switch_l2579_257916


namespace NUMINAMATH_CALUDE_nested_sqrt_simplification_l2579_257964

theorem nested_sqrt_simplification (y : ℝ) (h : y ≥ 0) :
  Real.sqrt (y * Real.sqrt (y^3 * Real.sqrt y)) = y^(9/4) := by sorry

end NUMINAMATH_CALUDE_nested_sqrt_simplification_l2579_257964


namespace NUMINAMATH_CALUDE_some_number_equals_37_l2579_257911

theorem some_number_equals_37 : ∃ x : ℤ, 45 - (28 - (x - (15 - 20))) = 59 ∧ x = 37 := by
  sorry

end NUMINAMATH_CALUDE_some_number_equals_37_l2579_257911


namespace NUMINAMATH_CALUDE_sasha_plucked_leaves_l2579_257912

/-- The number of leaves Sasha plucked -/
def leaves_plucked : ℕ := 22

/-- The number of apple trees -/
def apple_trees : ℕ := 17

/-- The number of poplar trees -/
def poplar_trees : ℕ := 18

/-- The position of the apple tree after which Masha's phone memory was full -/
def masha_last_photo : ℕ := 10

/-- The number of trees that remained unphotographed by Masha -/
def unphotographed_trees : ℕ := 13

/-- The position of the apple tree from which Sasha started plucking leaves -/
def sasha_start : ℕ := 8

theorem sasha_plucked_leaves : 
  apple_trees = 17 ∧ 
  poplar_trees = 18 ∧ 
  masha_last_photo = 10 ∧ 
  unphotographed_trees = 13 ∧ 
  sasha_start = 8 → 
  leaves_plucked = 22 := by
  sorry

end NUMINAMATH_CALUDE_sasha_plucked_leaves_l2579_257912


namespace NUMINAMATH_CALUDE_second_quadrant_transformation_l2579_257971

/-- A point in the 2D plane. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point is in the second quadrant. -/
def isInSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Theorem: If P(a,b) is in the second quadrant, then Q(-b,1-a) is also in the second quadrant. -/
theorem second_quadrant_transformation (a b : ℝ) :
  isInSecondQuadrant ⟨a, b⟩ → isInSecondQuadrant ⟨-b, 1-a⟩ := by
  sorry


end NUMINAMATH_CALUDE_second_quadrant_transformation_l2579_257971


namespace NUMINAMATH_CALUDE_pascal_ninth_row_interior_sum_l2579_257905

/-- Sum of elements in row n of Pascal's Triangle -/
def pascal_row_sum (n : ℕ) : ℕ := 2^(n-1)

/-- Sum of interior elements in row n of Pascal's Triangle -/
def pascal_interior_sum (n : ℕ) : ℕ := pascal_row_sum n - 2

theorem pascal_ninth_row_interior_sum :
  pascal_interior_sum 9 = 254 := by sorry

end NUMINAMATH_CALUDE_pascal_ninth_row_interior_sum_l2579_257905


namespace NUMINAMATH_CALUDE_seventh_root_of_unity_product_l2579_257955

theorem seventh_root_of_unity_product (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1) = 8 := by
  sorry

end NUMINAMATH_CALUDE_seventh_root_of_unity_product_l2579_257955


namespace NUMINAMATH_CALUDE_boxes_with_neither_l2579_257993

theorem boxes_with_neither (total : ℕ) (pencils : ℕ) (pens : ℕ) (both : ℕ) 
  (h1 : total = 15)
  (h2 : pencils = 8)
  (h3 : pens = 5)
  (h4 : both = 4) :
  total - (pencils + pens - both) = 6 := by
sorry

end NUMINAMATH_CALUDE_boxes_with_neither_l2579_257993


namespace NUMINAMATH_CALUDE_kolya_parallelepiped_edge_length_l2579_257908

/-- A rectangular parallelepiped constructed from unit cubes -/
structure Parallelepiped where
  length : ℕ
  width : ℕ
  height : ℕ
  volume : ℕ
  edge_min : ℕ

/-- The total length of all edges of a rectangular parallelepiped -/
def total_edge_length (p : Parallelepiped) : ℕ :=
  4 * (p.length + p.width + p.height)

/-- Theorem stating the total edge length of the specific parallelepiped -/
theorem kolya_parallelepiped_edge_length :
  ∃ (p : Parallelepiped),
    p.volume = 440 ∧
    p.edge_min = 5 ∧
    p.length ≥ p.edge_min ∧
    p.width ≥ p.edge_min ∧
    p.height ≥ p.edge_min ∧
    p.volume = p.length * p.width * p.height ∧
    total_edge_length p = 96 := by
  sorry

end NUMINAMATH_CALUDE_kolya_parallelepiped_edge_length_l2579_257908


namespace NUMINAMATH_CALUDE_jake_weight_proof_l2579_257933

/-- Jake's present weight in pounds -/
def jake_weight : ℝ := 196

/-- Jake's sister's weight in pounds -/
def sister_weight : ℝ := 290 - jake_weight

theorem jake_weight_proof :
  (jake_weight - 8 = 2 * sister_weight) ∧
  (jake_weight + sister_weight = 290) :=
by sorry

end NUMINAMATH_CALUDE_jake_weight_proof_l2579_257933


namespace NUMINAMATH_CALUDE_triangle_properties_l2579_257969

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = Real.pi →
  a / (Real.sin A) = b / (Real.sin B) →
  b / (Real.sin B) = c / (Real.sin C) →
  c * (Real.cos A) + Real.sqrt 3 * c * (Real.sin A) - b - a = 0 →
  (C = Real.pi / 3 ∧
   (c = 1 → ∀ (a' b' : ℝ), a' > 0 → b' > 0 → a' + b' > c →
     1 / 2 * a' * b' * Real.sin C ≤ Real.sqrt 3 / 4)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l2579_257969


namespace NUMINAMATH_CALUDE_max_value_when_a_zero_one_zero_iff_a_positive_l2579_257997

noncomputable section

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := a * x - 1 / x - (a + 1) * Real.log x

-- Theorem for part 1
theorem max_value_when_a_zero :
  ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f 0 y ≤ f 0 x ∧ f 0 x = -1 :=
sorry

-- Theorem for part 2
theorem one_zero_iff_a_positive :
  ∀ (a : ℝ), (∃! (x : ℝ), x > 0 ∧ f a x = 0) ↔ a > 0 :=
sorry

end

end NUMINAMATH_CALUDE_max_value_when_a_zero_one_zero_iff_a_positive_l2579_257997


namespace NUMINAMATH_CALUDE_lg_100_equals_2_l2579_257988

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem lg_100_equals_2 : lg 100 = 2 := by
  sorry

end NUMINAMATH_CALUDE_lg_100_equals_2_l2579_257988


namespace NUMINAMATH_CALUDE_smallest_square_area_l2579_257909

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Calculates the minimum side length of a square that can contain two rectangles -/
def minSquareSide (r1 r2 : Rectangle) : ℕ :=
  min (max r1.width r2.width + min r1.height r2.height)
      (max r1.height r2.height + min r1.width r2.width)

/-- The theorem stating the smallest possible area of the square -/
theorem smallest_square_area (r1 r2 : Rectangle)
  (h1 : r1 = ⟨3, 5⟩)
  (h2 : r2 = ⟨4, 6⟩) :
  (minSquareSide r1 r2) ^ 2 = 81 := by
  sorry

#eval (minSquareSide ⟨3, 5⟩ ⟨4, 6⟩) ^ 2

end NUMINAMATH_CALUDE_smallest_square_area_l2579_257909


namespace NUMINAMATH_CALUDE_employee_savings_l2579_257953

/-- Calculate the combined savings of three employees over a period of time. -/
def combined_savings (hourly_wage : ℚ) (hours_per_day : ℕ) (days_per_week : ℕ) 
  (robby_save_ratio : ℚ) (jaylen_save_ratio : ℚ) (miranda_save_ratio : ℚ) 
  (num_weeks : ℕ) : ℚ :=
  let weekly_salary := hourly_wage * hours_per_day * days_per_week
  let robby_savings := robby_save_ratio * weekly_salary
  let jaylen_savings := jaylen_save_ratio * weekly_salary
  let miranda_savings := miranda_save_ratio * weekly_salary
  (robby_savings + jaylen_savings + miranda_savings) * num_weeks

/-- The combined savings of three employees after four weeks is $3000. -/
theorem employee_savings : 
  combined_savings 10 10 5 (2/5) (3/5) (1/2) 4 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_employee_savings_l2579_257953


namespace NUMINAMATH_CALUDE_phone_call_duration_l2579_257924

/-- Calculates the duration of a phone call given the initial card value, cost per minute, and remaining credit. -/
def call_duration (initial_value : ℚ) (cost_per_minute : ℚ) (remaining_credit : ℚ) : ℚ :=
  (initial_value - remaining_credit) / cost_per_minute

/-- Theorem stating that given the specific values from the problem, the call duration is 22 minutes. -/
theorem phone_call_duration :
  let initial_value : ℚ := 30
  let cost_per_minute : ℚ := 16/100
  let remaining_credit : ℚ := 2648/100
  call_duration initial_value cost_per_minute remaining_credit = 22 := by
sorry

end NUMINAMATH_CALUDE_phone_call_duration_l2579_257924


namespace NUMINAMATH_CALUDE_natalia_documentaries_l2579_257959

/-- The number of documentaries in Natalia's library --/
def documentaries (novels comics albums crates_used crate_capacity : ℕ) : ℕ :=
  crates_used * crate_capacity - (novels + comics + albums)

/-- Theorem stating the number of documentaries in Natalia's library --/
theorem natalia_documentaries :
  documentaries 145 271 209 116 9 = 419 := by
  sorry

end NUMINAMATH_CALUDE_natalia_documentaries_l2579_257959


namespace NUMINAMATH_CALUDE_square_perimeter_l2579_257901

theorem square_perimeter (area : ℝ) (side : ℝ) (h1 : area = 36) (h2 : side^2 = area) :
  4 * side = 24 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l2579_257901


namespace NUMINAMATH_CALUDE_xy_value_l2579_257942

theorem xy_value (x y : ℝ) (h : |x - 1| + (x + y)^2 = 0) : x * y = -1 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l2579_257942


namespace NUMINAMATH_CALUDE_trees_difference_l2579_257900

theorem trees_difference (initial_trees : ℕ) (dead_trees : ℕ) 
  (h1 : initial_trees = 14) (h2 : dead_trees = 9) : 
  dead_trees - (initial_trees - dead_trees) = 4 := by
  sorry

end NUMINAMATH_CALUDE_trees_difference_l2579_257900


namespace NUMINAMATH_CALUDE_total_soap_cost_two_years_l2579_257918

/-- Represents the types of soap --/
inductive SoapType
  | Lavender
  | Lemon
  | Sandalwood

/-- Returns the price of a given soap type --/
def soapPrice (s : SoapType) : ℚ :=
  match s with
  | SoapType.Lavender => 4
  | SoapType.Lemon => 5
  | SoapType.Sandalwood => 6

/-- Applies the bulk discount to a given quantity and price --/
def applyDiscount (quantity : ℕ) (price : ℚ) : ℚ :=
  let totalPrice := price * quantity
  if quantity ≥ 10 then totalPrice * (1 - 0.15)
  else if quantity ≥ 7 then totalPrice * (1 - 0.10)
  else if quantity ≥ 4 then totalPrice * (1 - 0.05)
  else totalPrice

/-- Calculates the cost of soap for a given type over 2 years --/
def soapCostTwoYears (s : SoapType) : ℚ :=
  let price := soapPrice s
  applyDiscount 7 price + price

/-- Theorem: The total amount Elias spends on soap in 2 years is $109.50 --/
theorem total_soap_cost_two_years :
  soapCostTwoYears SoapType.Lavender +
  soapCostTwoYears SoapType.Lemon +
  soapCostTwoYears SoapType.Sandalwood = 109.5 := by
  sorry

end NUMINAMATH_CALUDE_total_soap_cost_two_years_l2579_257918


namespace NUMINAMATH_CALUDE_area_between_sine_and_constant_line_l2579_257980

theorem area_between_sine_and_constant_line : 
  let f : ℝ → ℝ := λ x => Real.sin x
  let g : ℝ → ℝ := λ _ => (1/2 : ℝ)
  let lower_bound : ℝ := 0
  let upper_bound : ℝ := Real.pi
  ∃ (area : ℝ), area = ∫ x in lower_bound..upper_bound, |f x - g x| ∧ area = Real.sqrt 3 - Real.pi / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_area_between_sine_and_constant_line_l2579_257980


namespace NUMINAMATH_CALUDE_rectangle_roots_l2579_257951

def polynomial (b : ℝ) (z : ℂ) : ℂ :=
  z^4 - 10*z^3 + 16*b*z^2 - 2*(3*b^2 - 5*b + 4)*z + 6

def forms_rectangle (b : ℝ) : Prop :=
  ∃ (z₁ z₂ z₃ z₄ : ℂ),
    polynomial b z₁ = 0 ∧
    polynomial b z₂ = 0 ∧
    polynomial b z₃ = 0 ∧
    polynomial b z₄ = 0 ∧
    (z₁.re = z₂.re ∧ z₁.im = -z₂.im) ∧
    (z₃.re = z₄.re ∧ z₃.im = -z₄.im) ∧
    (z₁.re - z₃.re = z₂.im - z₄.im) ∧
    (z₁.im - z₃.im = z₄.re - z₂.re)

theorem rectangle_roots :
  ∀ b : ℝ, forms_rectangle b ↔ (b = 5/3 ∨ b = 2) :=
sorry

end NUMINAMATH_CALUDE_rectangle_roots_l2579_257951


namespace NUMINAMATH_CALUDE_expected_total_rainfall_l2579_257915

/-- Represents the weather forecast for a day --/
structure WeatherForecast where
  sun_chance : ℝ
  rain_chance1 : ℝ
  rain_amount1 : ℝ
  rain_chance2 : ℝ
  rain_amount2 : ℝ

/-- Calculates the expected rainfall for a given weather forecast --/
def expected_rainfall (forecast : WeatherForecast) : ℝ :=
  forecast.sun_chance * 0 + forecast.rain_chance1 * forecast.rain_amount1 + 
  forecast.rain_chance2 * forecast.rain_amount2

/-- The weather forecast for weekdays --/
def weekday_forecast : WeatherForecast := {
  sun_chance := 0.3,
  rain_chance1 := 0.2,
  rain_amount1 := 5,
  rain_chance2 := 0.5,
  rain_amount2 := 8
}

/-- The weather forecast for weekend days --/
def weekend_forecast : WeatherForecast := {
  sun_chance := 0.5,
  rain_chance1 := 0.25,
  rain_amount1 := 2,
  rain_chance2 := 0.25,
  rain_amount2 := 6
}

/-- The number of weekdays --/
def num_weekdays : ℕ := 5

/-- The number of weekend days --/
def num_weekend_days : ℕ := 2

theorem expected_total_rainfall : 
  (num_weekdays : ℝ) * expected_rainfall weekday_forecast + 
  (num_weekend_days : ℝ) * expected_rainfall weekend_forecast = 29 := by
  sorry

end NUMINAMATH_CALUDE_expected_total_rainfall_l2579_257915


namespace NUMINAMATH_CALUDE_orange_marbles_count_l2579_257947

theorem orange_marbles_count (total : ℕ) (red : ℕ) (blue : ℕ) (orange : ℕ) : 
  total = 24 →
  blue = total / 2 →
  red = 6 →
  orange = total - blue - red →
  orange = 6 := by
sorry

end NUMINAMATH_CALUDE_orange_marbles_count_l2579_257947


namespace NUMINAMATH_CALUDE_not_square_of_integer_l2579_257922

theorem not_square_of_integer (n : ℕ+) : ¬ ∃ m : ℤ, m^2 = 2*(n.val^2 + 1) - n.val := by
  sorry

end NUMINAMATH_CALUDE_not_square_of_integer_l2579_257922


namespace NUMINAMATH_CALUDE_min_value_of_log_expression_four_is_minimum_l2579_257906

theorem min_value_of_log_expression (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  (Real.log 2011 / Real.log x + Real.log 2011 / Real.log y) / (Real.log 2011 / (Real.log x + Real.log y)) ≥ 4 :=
by sorry

theorem four_is_minimum (ε : ℝ) (hε : ε > 0) :
  ∃ x y : ℝ, x > 1 ∧ y > 1 ∧
  (Real.log 2011 / Real.log x + Real.log 2011 / Real.log y) / (Real.log 2011 / (Real.log x + Real.log y)) < 4 + ε :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_log_expression_four_is_minimum_l2579_257906


namespace NUMINAMATH_CALUDE_unique_solution_system_l2579_257990

/-- The system of equations has a unique solution (67/9, 1254/171) -/
theorem unique_solution_system :
  ∃! (x y : ℚ), (3 * x - 4 * y = -7) ∧ (6 * x - 5 * y = 8) :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_system_l2579_257990


namespace NUMINAMATH_CALUDE_diamond_spade_ratio_l2579_257957

structure Deck :=
  (clubs : ℕ)
  (diamonds : ℕ)
  (hearts : ℕ)
  (spades : ℕ)

def is_valid_deck (d : Deck) : Prop :=
  d.clubs + d.diamonds + d.hearts + d.spades = 13 ∧
  d.clubs + d.spades = 7 ∧
  d.diamonds + d.hearts = 6 ∧
  d.hearts = 2 * d.diamonds ∧
  d.clubs = 6

theorem diamond_spade_ratio (d : Deck) (h : is_valid_deck d) :
  d.diamonds = 2 ∧ d.spades = 1 :=
sorry

end NUMINAMATH_CALUDE_diamond_spade_ratio_l2579_257957


namespace NUMINAMATH_CALUDE_circle_graph_proportion_l2579_257985

theorem circle_graph_proportion (total_degrees : ℝ) (sector_degrees : ℝ) 
  (h1 : total_degrees = 360) 
  (h2 : sector_degrees = 180) : 
  sector_degrees / total_degrees = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_circle_graph_proportion_l2579_257985


namespace NUMINAMATH_CALUDE_sphere_plane_intersection_area_l2579_257939

theorem sphere_plane_intersection_area (r : ℝ) (h : r = 1) :
  ∃ (d : ℝ), 0 < d ∧ d < r ∧
  (2 * π * r * (r - d) = π * r^2) ∧
  (2 * π * r * d = 3 * π * r^2) ∧
  π * (r^2 - d^2) = (3 * π) / 4 := by
sorry

end NUMINAMATH_CALUDE_sphere_plane_intersection_area_l2579_257939


namespace NUMINAMATH_CALUDE_f_t_plus_one_l2579_257974

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x - 1

-- State the theorem
theorem f_t_plus_one (t : ℝ) : f (t + 1) = 3 * t + 2 := by
  sorry

end NUMINAMATH_CALUDE_f_t_plus_one_l2579_257974


namespace NUMINAMATH_CALUDE_cookie_boxes_theorem_l2579_257961

theorem cookie_boxes_theorem (n : ℕ) : 
  (∃ (m a : ℕ), 
    m = n - 8 ∧ 
    a = n - 2 ∧ 
    m ≥ 1 ∧ 
    a ≥ 1 ∧ 
    m + a < n) → 
  n = 9 := by
sorry

end NUMINAMATH_CALUDE_cookie_boxes_theorem_l2579_257961


namespace NUMINAMATH_CALUDE_july_birth_percentage_l2579_257941

/-- The percentage of scientists born in July, given the total number of scientists and the number born in July. -/
theorem july_birth_percentage 
  (total_scientists : ℕ) 
  (july_births : ℕ) 
  (h1 : total_scientists = 200) 
  (h2 : july_births = 17) : 
  (july_births : ℚ) / total_scientists * 100 = 8.5 := by
sorry

end NUMINAMATH_CALUDE_july_birth_percentage_l2579_257941


namespace NUMINAMATH_CALUDE_divisibility_by_eleven_l2579_257998

theorem divisibility_by_eleven (n : ℤ) : 
  11 ∣ (n^2001 - n^4) ↔ n % 11 = 0 ∨ n % 11 = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_eleven_l2579_257998
