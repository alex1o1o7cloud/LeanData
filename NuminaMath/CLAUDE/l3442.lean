import Mathlib

namespace NUMINAMATH_CALUDE_triangle_vector_dot_product_l3442_344277

/-- Given a triangle ABC with vectors AB and AC, prove that the dot product of AB and BC equals 5 -/
theorem triangle_vector_dot_product (A B C : ℝ × ℝ) : 
  let AB : ℝ × ℝ := (2, 3)
  let AC : ℝ × ℝ := (3, 4)
  let BC : ℝ × ℝ := AC - AB
  (AB.1 * BC.1 + AB.2 * BC.2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_vector_dot_product_l3442_344277


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l3442_344298

theorem inscribed_circle_radius (d : ℝ) (h : d = Real.sqrt 12) : 
  let R := d / 2
  let s := R * Real.sqrt 3
  let h := (Real.sqrt 3 / 2) * s
  let a := Real.sqrt (h^2 - (h/2)^2)
  let r := (a * Real.sqrt 3) / 6
  r = 9/8 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l3442_344298


namespace NUMINAMATH_CALUDE_f_inequality_l3442_344249

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem f_inequality (h1 : f 1 = 1) (h2 : ∀ x, deriv f x < 2) :
  ∀ x, f x < 2 * x - 1 ↔ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l3442_344249


namespace NUMINAMATH_CALUDE_percentage_increase_l3442_344276

theorem percentage_increase (original : ℝ) (new : ℝ) : 
  original = 50 → new = 80 → (new - original) / original * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l3442_344276


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_roots_min_value_achievable_l3442_344238

theorem min_value_of_sum_of_roots (x : ℝ) :
  Real.sqrt (x^2 + (x - 2)^2) + Real.sqrt ((x - 2)^2 + (x + 2)^2) ≥ 2 * Real.sqrt 5 :=
by sorry

theorem min_value_achievable :
  ∃ x : ℝ, Real.sqrt (x^2 + (x - 2)^2) + Real.sqrt ((x - 2)^2 + (x + 2)^2) = 2 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_roots_min_value_achievable_l3442_344238


namespace NUMINAMATH_CALUDE_longest_distance_rectangle_in_square_l3442_344205

/-- The longest distance between a vertex of a rectangle inscribed in a square and a vertex of the square -/
theorem longest_distance_rectangle_in_square 
  (rectangle_perimeter : ℝ) 
  (square_perimeter : ℝ) 
  (h_rectangle : rectangle_perimeter = 26) 
  (h_square : square_perimeter = 36) :
  ∃ (x y : ℝ), 
    2 * (x + y) = rectangle_perimeter ∧ 
    x > 0 ∧ 
    y > 0 ∧
    x ≤ square_perimeter / 4 ∧
    y ≤ square_perimeter / 4 ∧
    (square_perimeter / 4 * Real.sqrt 2 + Real.sqrt (x^2 + y^2)) / 2 = 
      (9 * Real.sqrt 2 + Real.sqrt 89) / 2 := by
  sorry

end NUMINAMATH_CALUDE_longest_distance_rectangle_in_square_l3442_344205


namespace NUMINAMATH_CALUDE_remaining_cube_volume_l3442_344289

/-- The remaining volume of a cube after removing a cylindrical section -/
theorem remaining_cube_volume (cube_side : ℝ) (cylinder_radius : ℝ) :
  cube_side = 4 →
  cylinder_radius = 2 →
  (cube_side ^ 3) - (π * cylinder_radius ^ 2 * cube_side) = 64 - 16 * π :=
by sorry

end NUMINAMATH_CALUDE_remaining_cube_volume_l3442_344289


namespace NUMINAMATH_CALUDE_minimum_guests_l3442_344224

theorem minimum_guests (total_food : ℝ) (max_per_guest : ℝ) (min_guests : ℕ) :
  total_food = 406 →
  max_per_guest = 2.5 →
  min_guests = 163 →
  (↑min_guests : ℝ) * max_per_guest ≥ total_food ∧
  ∀ n : ℕ, (↑n : ℝ) * max_per_guest ≥ total_food → n ≥ min_guests :=
by sorry

end NUMINAMATH_CALUDE_minimum_guests_l3442_344224


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_341_l3442_344212

theorem greatest_prime_factor_of_341 : ∃ p : ℕ, 
  Nat.Prime p ∧ p ∣ 341 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 341 → q ≤ p :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_341_l3442_344212


namespace NUMINAMATH_CALUDE_train_crossing_time_l3442_344250

/-- Proves that a train with given length and speed takes the calculated time to cross an electric pole -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) (crossing_time : Real) :
  train_length = 2500 ∧ 
  train_speed_kmh = 90 →
  crossing_time = 100 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l3442_344250


namespace NUMINAMATH_CALUDE_coin_value_equality_l3442_344236

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Theorem stating that if the value of 25 quarters, 15 dimes, and 10 nickels 
    equals the value of 15 quarters, n dimes, and 20 nickels, then n = 35 -/
theorem coin_value_equality (n : ℕ) : 
  25 * quarter_value + 15 * dime_value + 10 * nickel_value = 
  15 * quarter_value + n * dime_value + 20 * nickel_value → n = 35 := by
  sorry

end NUMINAMATH_CALUDE_coin_value_equality_l3442_344236


namespace NUMINAMATH_CALUDE_accepted_to_rejected_ratio_egg_processing_change_l3442_344263

/-- Represents the daily egg processing at a plant -/
structure EggProcessing where
  total : ℕ
  accepted : ℕ
  rejected : ℕ
  h_total : total = accepted + rejected

/-- The original egg processing scenario -/
def original : EggProcessing :=
  { total := 400,
    accepted := 384,
    rejected := 16,
    h_total := rfl }

/-- The modified egg processing scenario -/
def modified : EggProcessing :=
  { total := 400,
    accepted := 396,
    rejected := 4,
    h_total := rfl }

/-- Theorem stating the ratio of accepted to rejected eggs in the modified scenario -/
theorem accepted_to_rejected_ratio :
  modified.accepted / modified.rejected = 99 := by
  sorry

/-- Proof that the ratio of accepted to rejected eggs changes as described -/
theorem egg_processing_change (orig : EggProcessing) (mod : EggProcessing)
  (h_orig : orig = original)
  (h_mod : mod = modified)
  (h_total_unchanged : orig.total = mod.total)
  (h_accepted_increase : mod.accepted = orig.accepted + 12) :
  mod.accepted / mod.rejected = 99 := by
  sorry

end NUMINAMATH_CALUDE_accepted_to_rejected_ratio_egg_processing_change_l3442_344263


namespace NUMINAMATH_CALUDE_inequality_solution_l3442_344260

theorem inequality_solution (x : ℝ) : (x - 2) / (x - 4) ≥ 3 ↔ x < 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3442_344260


namespace NUMINAMATH_CALUDE_vacation_pictures_deleted_l3442_344280

theorem vacation_pictures_deleted (zoo_pics : ℕ) (museum_pics : ℕ) (remaining_pics : ℕ) : 
  zoo_pics = 41 → museum_pics = 29 → remaining_pics = 55 → 
  zoo_pics + museum_pics - remaining_pics = 15 := by
  sorry

end NUMINAMATH_CALUDE_vacation_pictures_deleted_l3442_344280


namespace NUMINAMATH_CALUDE_no_five_cent_combination_l3442_344292

/-- Represents the types of coins available -/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | HalfDollar

/-- Returns the value of a coin in cents -/
def coinValue (c : Coin) : ℕ :=
  match c with
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.HalfDollar => 50

/-- A function that takes a list of 5 coins and returns their total value in cents -/
def totalValue (coins : List Coin) : ℕ :=
  coins.map coinValue |>.sum

/-- Theorem stating that it's impossible to select 5 coins with a total value of 5 cents -/
theorem no_five_cent_combination :
  ¬ ∃ (coins : List Coin), coins.length = 5 ∧ totalValue coins = 5 := by
  sorry


end NUMINAMATH_CALUDE_no_five_cent_combination_l3442_344292


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l3442_344248

theorem modulus_of_complex_fraction (i : ℂ) : i * i = -1 → Complex.abs ((3 - 4 * i) / i) = 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l3442_344248


namespace NUMINAMATH_CALUDE_halfway_fraction_l3442_344297

theorem halfway_fraction (a b c d : ℕ) (h1 : a = 3 ∧ b = 4) (h2 : c = 5 ∧ d = 7) :
  (a / b + c / d) / 2 = 41 / 56 :=
sorry

end NUMINAMATH_CALUDE_halfway_fraction_l3442_344297


namespace NUMINAMATH_CALUDE_transformed_function_eq_g_l3442_344228

/-- The original quadratic function -/
def f (x : ℝ) : ℝ := 2 * (x - 3)^2 + 4

/-- The transformed quadratic function -/
def g (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 3

/-- The horizontal shift transformation -/
def shift_left (h : ℝ) (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x + h)

/-- The vertical shift transformation -/
def shift_down (k : ℝ) (f : ℝ → ℝ) (x : ℝ) : ℝ := f x - k

/-- Theorem stating that the transformed function is equivalent to g -/
theorem transformed_function_eq_g :
  ∀ x, shift_down 3 (shift_left 2 f) x = g x := by sorry

end NUMINAMATH_CALUDE_transformed_function_eq_g_l3442_344228


namespace NUMINAMATH_CALUDE_max_points_top_four_teams_l3442_344222

/-- Represents a tournament with the given conditions -/
structure Tournament :=
  (num_teams : ℕ)
  (points_for_win : ℕ)
  (points_for_draw : ℕ)
  (points_for_loss : ℕ)

/-- Calculates the total number of games in the tournament -/
def total_games (t : Tournament) : ℕ :=
  t.num_teams * (t.num_teams - 1) / 2

/-- Represents the maximum possible points for top teams -/
def max_points_for_top_teams (t : Tournament) (num_top_teams : ℕ) : ℕ :=
  sorry

/-- Theorem stating the maximum possible points for each of the top four teams -/
theorem max_points_top_four_teams (t : Tournament) :
  t.num_teams = 7 →
  t.points_for_win = 3 →
  t.points_for_draw = 1 →
  t.points_for_loss = 0 →
  max_points_for_top_teams t 4 = 18 := by
  sorry

end NUMINAMATH_CALUDE_max_points_top_four_teams_l3442_344222


namespace NUMINAMATH_CALUDE_player1_can_achieve_6_player2_can_prevent_above_6_max_achievable_sum_is_6_l3442_344279

/-- Represents a cell on the 5x5 board -/
inductive Cell
| mk (row : Fin 5) (col : Fin 5)

/-- Represents the state of a cell (Empty, Marked by Player 1, or Marked by Player 2) -/
inductive CellState
| Empty
| Player1
| Player2

/-- Represents the game board -/
def Board := Cell → CellState

/-- Checks if a given 3x3 sub-square is valid on the 5x5 board -/
def isValid3x3Square (topLeft : Cell) : Prop :=
  ∃ (r c : Fin 3), topLeft = Cell.mk r c

/-- Computes the sum of a 3x3 sub-square -/
def subSquareSum (b : Board) (topLeft : Cell) : ℕ :=
  sorry

/-- The maximum sum of any 3x3 sub-square on the board -/
def maxSubSquareSum (b : Board) : ℕ :=
  sorry

/-- A strategy for Player 1 -/
def Player1Strategy := Board → Cell

/-- A strategy for Player 2 -/
def Player2Strategy := Board → Cell

/-- Simulates a game given strategies for both players -/
def playGame (s1 : Player1Strategy) (s2 : Player2Strategy) : Board :=
  sorry

/-- Theorem stating that Player 1 can always achieve a maximum 3x3 sub-square sum of at least 6 -/
theorem player1_can_achieve_6 :
  ∃ (s1 : Player1Strategy), ∀ (s2 : Player2Strategy),
    maxSubSquareSum (playGame s1 s2) ≥ 6 :=
  sorry

/-- Theorem stating that Player 2 can always prevent the maximum 3x3 sub-square sum from exceeding 6 -/
theorem player2_can_prevent_above_6 :
  ∃ (s2 : Player2Strategy), ∀ (s1 : Player1Strategy),
    maxSubSquareSum (playGame s1 s2) ≤ 6 :=
  sorry

/-- Main theorem combining the above results -/
theorem max_achievable_sum_is_6 :
  (∃ (s1 : Player1Strategy), ∀ (s2 : Player2Strategy),
    maxSubSquareSum (playGame s1 s2) ≥ 6) ∧
  (∃ (s2 : Player2Strategy), ∀ (s1 : Player1Strategy),
    maxSubSquareSum (playGame s1 s2) ≤ 6) :=
  sorry

end NUMINAMATH_CALUDE_player1_can_achieve_6_player2_can_prevent_above_6_max_achievable_sum_is_6_l3442_344279


namespace NUMINAMATH_CALUDE_equation_solution_l3442_344244

theorem equation_solution : ∃ x : ℝ, 2 * x + 6 = 2 + 3 * x ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3442_344244


namespace NUMINAMATH_CALUDE_ferry_route_ratio_l3442_344215

-- Define the parameters
def ferry_p_speed : ℝ := 6
def ferry_p_time : ℝ := 3
def ferry_q_speed_difference : ℝ := 3
def ferry_q_time_difference : ℝ := 3

-- Define the theorem
theorem ferry_route_ratio :
  let ferry_p_distance := ferry_p_speed * ferry_p_time
  let ferry_q_speed := ferry_p_speed + ferry_q_speed_difference
  let ferry_q_time := ferry_p_time + ferry_q_time_difference
  let ferry_q_distance := ferry_q_speed * ferry_q_time
  ferry_q_distance / ferry_p_distance = 3 := by
  sorry


end NUMINAMATH_CALUDE_ferry_route_ratio_l3442_344215


namespace NUMINAMATH_CALUDE_beth_score_l3442_344246

/-- The score of a basketball game between two teams -/
structure BasketballScore where
  team1_player1 : ℕ  -- Beth's score
  team1_player2 : ℕ  -- Jan's score
  team2_player1 : ℕ  -- Judy's score
  team2_player2 : ℕ  -- Angel's score

/-- The conditions of the basketball game -/
def game_conditions (score : BasketballScore) : Prop :=
  score.team1_player2 = 10 ∧
  score.team2_player1 = 8 ∧
  score.team2_player2 = 11 ∧
  score.team1_player1 + score.team1_player2 = score.team2_player1 + score.team2_player2 + 3

/-- Theorem: Given the game conditions, Beth scored 12 points -/
theorem beth_score (score : BasketballScore) 
  (h : game_conditions score) : score.team1_player1 = 12 := by
  sorry


end NUMINAMATH_CALUDE_beth_score_l3442_344246


namespace NUMINAMATH_CALUDE_point_on_600_degree_angle_l3442_344294

theorem point_on_600_degree_angle (a : ℝ) : 
  (∃ θ : ℝ, θ = 600 * Real.pi / 180 ∧ 
   (-1 : ℝ) = Real.cos θ ∧ 
   a = Real.sin θ) → 
  a = -Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_point_on_600_degree_angle_l3442_344294


namespace NUMINAMATH_CALUDE_area_of_circle_portion_l3442_344214

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop := x^2 - 12*x + y^2 = 28

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := y = x - 4

/-- The region of interest -/
def region_of_interest (x y : ℝ) : Prop :=
  circle_equation x y ∧ y ≥ 0 ∧ y ≥ x - 4

/-- The area of the region of interest -/
noncomputable def area_of_region : ℝ := sorry

theorem area_of_circle_portion : area_of_region = 48 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_area_of_circle_portion_l3442_344214


namespace NUMINAMATH_CALUDE_total_animals_after_addition_l3442_344288

/-- Represents the number of animals on a farm --/
structure FarmAnimals where
  cows : ℕ
  pigs : ℕ
  goats : ℕ

/-- Calculates the total number of animals on the farm --/
def totalAnimals (farm : FarmAnimals) : ℕ :=
  farm.cows + farm.pigs + farm.goats

/-- The initial number of animals on the farm --/
def initialFarm : FarmAnimals :=
  { cows := 2, pigs := 3, goats := 6 }

/-- The number of animals to be added to the farm --/
def addedAnimals : FarmAnimals :=
  { cows := 3, pigs := 5, goats := 2 }

/-- Theorem stating that the total number of animals after addition is 21 --/
theorem total_animals_after_addition :
  totalAnimals initialFarm + totalAnimals addedAnimals = 21 := by
  sorry


end NUMINAMATH_CALUDE_total_animals_after_addition_l3442_344288


namespace NUMINAMATH_CALUDE_line_contains_point_l3442_344217

/-- The value of k for which the line 2 - 2kx = -4y contains the point (3, -2) -/
theorem line_contains_point (k : ℝ) : 
  (2 - 2 * k * 3 = -4 * (-2)) ↔ k = -1 := by sorry

end NUMINAMATH_CALUDE_line_contains_point_l3442_344217


namespace NUMINAMATH_CALUDE_second_number_calculation_l3442_344295

theorem second_number_calculation (A : ℝ) (X : ℝ) (h1 : A = 1280) 
  (h2 : 0.25 * A = 0.20 * X + 190) : X = 650 := by
  sorry

end NUMINAMATH_CALUDE_second_number_calculation_l3442_344295


namespace NUMINAMATH_CALUDE_atomic_number_relation_l3442_344216

-- Define the compound Y₂X₃
structure Compound where
  X : ℕ  -- Atomic number of X
  Y : ℕ  -- Atomic number of Y

-- Define the property of X being a short-period non-metal element
def isShortPeriodNonMetal (x : ℕ) : Prop :=
  x ≤ 18  -- Assuming short-period elements have atomic numbers up to 18

-- Define the compound formation rule
def formsCompound (c : Compound) : Prop :=
  isShortPeriodNonMetal c.X ∧ c.Y > 0

-- Theorem statement
theorem atomic_number_relation (n : ℕ) :
  ∀ c : Compound, formsCompound c → c.X = n → c.Y ≠ n + 2 := by
  sorry

end NUMINAMATH_CALUDE_atomic_number_relation_l3442_344216


namespace NUMINAMATH_CALUDE_probability_of_banana_lunch_l3442_344286

-- Define the types of meats and fruits
inductive Meat
| beef
| chicken

inductive Fruit
| apple
| pear
| banana

-- Define a lunch as a pair of meat and fruit
def Lunch := Meat × Fruit

-- Define the set of all possible lunches
def allLunches : Finset Lunch := sorry

-- Define the set of lunches with banana
def lunchesWithBanana : Finset Lunch := sorry

-- Theorem statement
theorem probability_of_banana_lunch :
  (Finset.card lunchesWithBanana : ℚ) / (Finset.card allLunches : ℚ) = 1/3 := by sorry

end NUMINAMATH_CALUDE_probability_of_banana_lunch_l3442_344286


namespace NUMINAMATH_CALUDE_cube_root_simplification_l3442_344268

theorem cube_root_simplification : (5488000 : ℝ)^(1/3) = 140 * 2^(1/3) := by sorry

end NUMINAMATH_CALUDE_cube_root_simplification_l3442_344268


namespace NUMINAMATH_CALUDE_girls_insects_count_l3442_344210

/-- The number of insects collected by boys -/
def boys_insects : ℕ := 200

/-- The number of groups the class was divided into -/
def num_groups : ℕ := 4

/-- The number of insects each group received -/
def insects_per_group : ℕ := 125

/-- The number of insects collected by girls -/
def girls_insects : ℕ := num_groups * insects_per_group - boys_insects

theorem girls_insects_count : girls_insects = 300 := by
  sorry

end NUMINAMATH_CALUDE_girls_insects_count_l3442_344210


namespace NUMINAMATH_CALUDE_sum_of_cubes_l3442_344285

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 4) (h2 : a * b = -5) : a^3 + b^3 = 124 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l3442_344285


namespace NUMINAMATH_CALUDE_initial_money_calculation_l3442_344247

def toy_car_price : ℕ := 11
def scarf_price : ℕ := 10
def beanie_price : ℕ := 14
def remaining_money : ℕ := 7

def total_spent : ℕ := 2 * toy_car_price + scarf_price + beanie_price

theorem initial_money_calculation :
  total_spent + remaining_money = 53 := by sorry

end NUMINAMATH_CALUDE_initial_money_calculation_l3442_344247


namespace NUMINAMATH_CALUDE_fixed_points_are_corresponding_l3442_344211

/-- A type representing a geometric figure -/
structure Figure where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Similarity transformation between two figures -/
def similarity (F1 F2 : Figure) : Prop :=
  sorry

/-- A point in a geometric figure -/
structure Point where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Defines if a point is fixed under a similarity transformation -/
def is_fixed_point (p : Point) (F1 F2 : Figure) : Prop :=
  sorry

/-- Defines if two points are corresponding in similar figures -/
def are_corresponding (p1 p2 : Point) (F1 F2 : Figure) : Prop :=
  sorry

/-- Main theorem: Fixed points of three similar figures are their corresponding points -/
theorem fixed_points_are_corresponding 
  (F1 F2 F3 : Figure) 
  (h1 : similarity F1 F2) 
  (h2 : similarity F2 F3) 
  (h3 : similarity F3 F1) 
  (p1 : Point) 
  (p2 : Point) 
  (p3 : Point) 
  (hf1 : is_fixed_point p1 F1 F2) 
  (hf2 : is_fixed_point p2 F2 F3) 
  (hf3 : is_fixed_point p3 F3 F1) : 
  are_corresponding p1 p2 F1 F2 ∧ 
  are_corresponding p2 p3 F2 F3 ∧ 
  are_corresponding p3 p1 F3 F1 :=
sorry

end NUMINAMATH_CALUDE_fixed_points_are_corresponding_l3442_344211


namespace NUMINAMATH_CALUDE_consecutive_coin_tosses_l3442_344220

theorem consecutive_coin_tosses (p : ℝ) (h : p = 1 / 2) :
  p ^ 5 = 1 / 32 := by
sorry

end NUMINAMATH_CALUDE_consecutive_coin_tosses_l3442_344220


namespace NUMINAMATH_CALUDE_soccer_substitutions_remainder_l3442_344275

/-- The number of ways to make substitutions in a soccer game -/
def substitution_ways (total_players start_players max_substitutions : ℕ) : ℕ :=
  sorry

/-- The main theorem about the remainder of substitution ways when divided by 1000 -/
theorem soccer_substitutions_remainder :
  let total_players := 22
  let start_players := 11
  let max_substitutions := 4
  (substitution_ways total_players start_players max_substitutions) % 1000 = 122 := by
  sorry

end NUMINAMATH_CALUDE_soccer_substitutions_remainder_l3442_344275


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3442_344261

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

-- State the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 + a 3 = 5 →
  a 3 + a 5 = 20 →
  a 5 + a 7 = 80 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3442_344261


namespace NUMINAMATH_CALUDE_pet_store_siamese_cats_l3442_344204

/-- The number of Siamese cats initially in the pet store. -/
def initial_siamese_cats : ℕ := 13

/-- The number of house cats initially in the pet store. -/
def initial_house_cats : ℕ := 5

/-- The number of cats sold during the sale. -/
def cats_sold : ℕ := 10

/-- The number of cats left after the sale. -/
def cats_remaining : ℕ := 8

/-- Theorem stating that the initial number of Siamese cats is correct. -/
theorem pet_store_siamese_cats :
  initial_siamese_cats + initial_house_cats - cats_sold = cats_remaining :=
by sorry

end NUMINAMATH_CALUDE_pet_store_siamese_cats_l3442_344204


namespace NUMINAMATH_CALUDE_min_recolor_is_n_minus_one_l3442_344225

/-- A complete graph of order n (≥ 3) with edges colored using three colors. -/
structure ColoredCompleteGraph where
  n : ℕ
  n_ge_3 : n ≥ 3
  colors : Fin 3 → Type
  edge_coloring : Fin n → Fin n → Fin 3
  each_color_used : ∀ c : Fin 3, ∃ i j : Fin n, i ≠ j ∧ edge_coloring i j = c

/-- The minimum number of edges that need to be recolored to make the graph connected by one color. -/
def min_recolor (G : ColoredCompleteGraph) : ℕ := G.n - 1

/-- Theorem stating that the minimum number of edges to recolor is n - 1. -/
theorem min_recolor_is_n_minus_one (G : ColoredCompleteGraph) :
  min_recolor G = G.n - 1 := by sorry

end NUMINAMATH_CALUDE_min_recolor_is_n_minus_one_l3442_344225


namespace NUMINAMATH_CALUDE_quadratic_equation_distinct_roots_l3442_344227

theorem quadratic_equation_distinct_roots (p q : ℚ) : 
  (∀ x : ℚ, x^2 + p*x + q = 0 ↔ x = 2*p ∨ x = p + q) ∧ 
  (2*p ≠ p + q) → 
  p = 2/3 ∧ q = -8/3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_distinct_roots_l3442_344227


namespace NUMINAMATH_CALUDE_max_a6_value_l3442_344229

theorem max_a6_value (a : ℕ → ℝ) 
  (h1 : ∀ n, a (n + 1) ≤ (a (n + 2) + a n) / 2)
  (h2 : a 1 = 1)
  (h3 : a 404 = 2016) :
  ∃ M, a 6 ≤ M ∧ M = 26 :=
by sorry

end NUMINAMATH_CALUDE_max_a6_value_l3442_344229


namespace NUMINAMATH_CALUDE_intersection_k_value_l3442_344293

/-- Given two lines that intersect at a point, find the value of k -/
theorem intersection_k_value (m n : ℝ → ℝ) (k : ℝ) :
  (∀ x, m x = 4 * x + 2) →  -- Line m equation
  (∀ x, n x = k * x - 8) →  -- Line n equation
  m (-2) = -6 →             -- Lines intersect at (-2, -6)
  n (-2) = -6 →             -- Lines intersect at (-2, -6)
  k = -1 := by
sorry

end NUMINAMATH_CALUDE_intersection_k_value_l3442_344293


namespace NUMINAMATH_CALUDE_distance_PF_is_five_l3442_344237

/-- Parabola structure with focus and directrix -/
structure Parabola :=
  (focus : ℝ × ℝ)
  (directrix : ℝ)

/-- Point on a parabola -/
structure PointOnParabola (p : Parabola) :=
  (point : ℝ × ℝ)
  (on_parabola : (point.2)^2 = 4 * point.1)

/-- Given parabola y^2 = 4x -/
def given_parabola : Parabola :=
  { focus := (1, 0),
    directrix := -1 }

/-- Point P on the parabola with x-coordinate 4 -/
def point_P : PointOnParabola given_parabola :=
  { point := (4, 4),
    on_parabola := by sorry }

/-- Theorem: The distance between P and F is 5 -/
theorem distance_PF_is_five :
  let F := given_parabola.focus
  let P := point_P.point
  Real.sqrt ((P.1 - F.1)^2 + (P.2 - F.2)^2) = 5 := by sorry

end NUMINAMATH_CALUDE_distance_PF_is_five_l3442_344237


namespace NUMINAMATH_CALUDE_investment_rate_proof_l3442_344284

/-- Proves that the required interest rate for the remaining investment is 6.4% --/
theorem investment_rate_proof (total_investment : ℝ) (first_investment : ℝ) (second_investment : ℝ)
  (first_rate : ℝ) (second_rate : ℝ) (desired_income : ℝ)
  (h1 : total_investment = 10000)
  (h2 : first_investment = 4000)
  (h3 : second_investment = 3500)
  (h4 : first_rate = 0.05)
  (h5 : second_rate = 0.04)
  (h6 : desired_income = 500) :
  (desired_income - (first_investment * first_rate + second_investment * second_rate)) / 
  (total_investment - first_investment - second_investment) = 0.064 := by
sorry

end NUMINAMATH_CALUDE_investment_rate_proof_l3442_344284


namespace NUMINAMATH_CALUDE_smallest_multiplier_perfect_square_l3442_344245

theorem smallest_multiplier_perfect_square (x : ℕ+) :
  (∃ y : ℕ+, y = 2 ∧ 
    (∃ z : ℕ+, x * y = z^2) ∧
    (∀ w : ℕ+, w < y → ¬∃ v : ℕ+, x * w = v^2)) →
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_smallest_multiplier_perfect_square_l3442_344245


namespace NUMINAMATH_CALUDE_juice_dispenser_capacity_l3442_344230

/-- A cylindrical juice dispenser with capacity x cups -/
structure JuiceDispenser where
  capacity : ℝ
  cylindrical : Bool

/-- Theorem: A cylindrical juice dispenser that contains 60 cups when 48% full has a total capacity of 125 cups -/
theorem juice_dispenser_capacity (d : JuiceDispenser) 
  (h_cylindrical : d.cylindrical = true) 
  (h_partial : 0.48 * d.capacity = 60) : 
  d.capacity = 125 := by
  sorry

end NUMINAMATH_CALUDE_juice_dispenser_capacity_l3442_344230


namespace NUMINAMATH_CALUDE_modulus_sum_complex_numbers_l3442_344290

theorem modulus_sum_complex_numbers : 
  Complex.abs ((3 : ℂ) - 8*I + (4 : ℂ) + 6*I) = Real.sqrt 53 := by sorry

end NUMINAMATH_CALUDE_modulus_sum_complex_numbers_l3442_344290


namespace NUMINAMATH_CALUDE_solve_equation_l3442_344203

theorem solve_equation (x : ℝ) (h : 5 - 5 / x = 4 + 4 / x) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3442_344203


namespace NUMINAMATH_CALUDE_max_carlson_jars_l3442_344283

/-- Represents the initial state of jam jars --/
structure JamState where
  carlson_weight : ℕ  -- Total weight of Carlson's jars
  baby_weight : ℕ     -- Total weight of Baby's jars
  carlson_jars : ℕ    -- Number of Carlson's jars

/-- Represents the state after Carlson gives his smallest jar to Baby --/
structure NewJamState where
  carlson_weight : ℕ  -- New total weight of Carlson's jars
  baby_weight : ℕ     -- New total weight of Baby's jars

/-- Conditions of the problem --/
def jam_problem (initial : JamState) (final : NewJamState) : Prop :=
  initial.carlson_weight = 13 * initial.baby_weight ∧
  final.carlson_weight = 8 * final.baby_weight ∧
  initial.carlson_weight = final.carlson_weight + (final.baby_weight - initial.baby_weight) ∧
  initial.carlson_jars > 0

/-- The theorem to be proved --/
theorem max_carlson_jars :
  ∀ (initial : JamState) (final : NewJamState),
    jam_problem initial final →
    initial.carlson_jars ≤ 23 :=
sorry

end NUMINAMATH_CALUDE_max_carlson_jars_l3442_344283


namespace NUMINAMATH_CALUDE_simplify_algebraic_expression_l3442_344213

theorem simplify_algebraic_expression (x : ℝ) 
  (h1 : x ≠ 1) (h2 : x ≠ -1) (h3 : x ≠ 0) : 
  (x - x / (x + 1)) / (1 + 1 / (x^2 - 1)) = x - 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_algebraic_expression_l3442_344213


namespace NUMINAMATH_CALUDE_sum_of_integers_and_squares_l3442_344266

-- Define the sum of integers from a to b, inclusive
def sumIntegers (a b : Int) : Int :=
  (b - a + 1) * (a + b) / 2

-- Define the sum of squares from a to b, inclusive
def sumSquares (a b : Int) : Int :=
  (b * (b + 1) * (2 * b + 1) - (a - 1) * a * (2 * a - 1)) / 6

theorem sum_of_integers_and_squares : 
  sumIntegers (-50) 40 + sumSquares 10 40 = 21220 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_and_squares_l3442_344266


namespace NUMINAMATH_CALUDE_factors_720_l3442_344299

/-- The number of distinct positive factors of 720 -/
def num_factors_720 : ℕ := sorry

/-- 720 has exactly 30 distinct positive factors -/
theorem factors_720 : num_factors_720 = 30 := by sorry

end NUMINAMATH_CALUDE_factors_720_l3442_344299


namespace NUMINAMATH_CALUDE_field_length_proof_l3442_344218

theorem field_length_proof (width : ℝ) (length : ℝ) (pond_side : ℝ) :
  length = 2 * width →
  pond_side = 8 →
  pond_side ^ 2 = (1 / 18) * (length * width) →
  length = 48 := by
  sorry

end NUMINAMATH_CALUDE_field_length_proof_l3442_344218


namespace NUMINAMATH_CALUDE_max_min_product_l3442_344282

theorem max_min_product (p q r : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)
  (sum_eq : p + q + r = 13) (sum_prod_eq : p * q + q * r + r * p = 30) :
  ∃ (n : ℝ), n = min (p * q) (min (q * r) (r * p)) ∧ n ≤ 10 ∧
  ∀ (m : ℝ), m = min (p * q) (min (q * r) (r * p)) → m ≤ n :=
sorry

end NUMINAMATH_CALUDE_max_min_product_l3442_344282


namespace NUMINAMATH_CALUDE_diagonals_27_sided_polygon_l3442_344296

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

/-- Theorem: A convex polygon with 27 sides has 324 diagonals -/
theorem diagonals_27_sided_polygon :
  num_diagonals 27 = 324 := by
  sorry

end NUMINAMATH_CALUDE_diagonals_27_sided_polygon_l3442_344296


namespace NUMINAMATH_CALUDE_same_color_config_prob_is_correct_l3442_344200

def total_candies : ℕ := 40
def red_candies : ℕ := 15
def blue_candies : ℕ := 15
def green_candies : ℕ := 10

def same_color_config_prob : ℚ :=
  let prob_both_red := (red_candies * (red_candies - 1) * (red_candies - 2) * (red_candies - 3)) / 
                       (total_candies * (total_candies - 1) * (total_candies - 2) * (total_candies - 3))
  let prob_both_blue := (blue_candies * (blue_candies - 1) * (blue_candies - 2) * (blue_candies - 3)) / 
                        (total_candies * (total_candies - 1) * (total_candies - 2) * (total_candies - 3))
  let prob_both_green := (green_candies * (green_candies - 1) * (green_candies - 2) * (green_candies - 3)) / 
                         (total_candies * (total_candies - 1) * (total_candies - 2) * (total_candies - 3))
  let prob_both_red_blue := (red_candies * blue_candies * (red_candies - 1) * (blue_candies - 1)) / 
                            (total_candies * (total_candies - 1) * (total_candies - 2) * (total_candies - 3))
  2 * prob_both_red + 2 * prob_both_blue + prob_both_green + 2 * prob_both_red_blue

theorem same_color_config_prob_is_correct : same_color_config_prob = 579 / 8686 := by
  sorry

end NUMINAMATH_CALUDE_same_color_config_prob_is_correct_l3442_344200


namespace NUMINAMATH_CALUDE_problem_solution_l3442_344226

noncomputable def f (a x : ℝ) : ℝ := x + Real.exp (x - a)

noncomputable def g (a x : ℝ) : ℝ := Real.log (x + 2) - 4 * Real.exp (a - x)

theorem problem_solution (a : ℝ) :
  (∃ x₀ : ℝ, f a x₀ - g a x₀ = 3) → a = -Real.log 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3442_344226


namespace NUMINAMATH_CALUDE_dummies_leftover_l3442_344233

theorem dummies_leftover (n : ℕ) (h : n % 10 = 3) : (4 * n) % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_dummies_leftover_l3442_344233


namespace NUMINAMATH_CALUDE_elise_comic_book_cost_l3442_344201

/-- Calculates the amount spent on a comic book given initial money, saved money, puzzle cost, and final money --/
def comic_book_cost (initial_money saved_money puzzle_cost final_money : ℕ) : ℕ :=
  initial_money + saved_money - puzzle_cost - final_money

/-- Proves that Elise spent $2 on the comic book --/
theorem elise_comic_book_cost :
  comic_book_cost 8 13 18 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_elise_comic_book_cost_l3442_344201


namespace NUMINAMATH_CALUDE_jenny_cat_expenditure_first_year_l3442_344269

/-- Calculates Jenny's expenditure on a cat for the first year -/
def jennys_cat_expenditure (adoption_fee : ℕ) (vet_costs : ℕ) (monthly_food_cost : ℕ) (jenny_toy_costs : ℕ) : ℕ :=
  let shared_costs := adoption_fee + vet_costs
  let jenny_shared_costs := shared_costs / 2
  let annual_food_cost := monthly_food_cost * 12
  let jenny_food_cost := annual_food_cost / 2
  jenny_shared_costs + jenny_food_cost + jenny_toy_costs

/-- Theorem stating Jenny's total expenditure on the cat in the first year -/
theorem jenny_cat_expenditure_first_year : 
  jennys_cat_expenditure 50 500 25 200 = 625 := by
  sorry

end NUMINAMATH_CALUDE_jenny_cat_expenditure_first_year_l3442_344269


namespace NUMINAMATH_CALUDE_marathon_end_time_l3442_344272

-- Define the start time of the marathon
def start_time : Nat := 15  -- 3:00 p.m. in 24-hour format

-- Define the duration of the marathon in minutes
def duration : Nat := 780

-- Define a function to calculate the end time
def calculate_end_time (start : Nat) (duration_minutes : Nat) : Nat :=
  (start + duration_minutes / 60) % 24

-- Theorem to prove
theorem marathon_end_time :
  calculate_end_time start_time duration = 4 := by
  sorry


end NUMINAMATH_CALUDE_marathon_end_time_l3442_344272


namespace NUMINAMATH_CALUDE_factorization_theorem_l3442_344223

variable (x : ℝ)

theorem factorization_theorem :
  (x^2 - 4*x + 3 = (x-1)*(x-3)) ∧
  (4*x^2 + 12*x - 7 = (2*x+7)*(2*x-1)) := by
  sorry

end NUMINAMATH_CALUDE_factorization_theorem_l3442_344223


namespace NUMINAMATH_CALUDE_circle_construction_cases_l3442_344208

/-- Two lines in a plane -/
structure Line where
  -- Add necessary fields for a line

/-- A point in a plane -/
structure Point where
  -- Add necessary fields for a point

/-- A circle in a plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- Predicate to check if a point is on a line -/
def point_on_line (p : Point) (l : Line) : Prop :=
  sorry

/-- Predicate to check if two lines intersect -/
def lines_intersect (l1 l2 : Line) : Prop :=
  sorry

/-- Predicate to check if two lines are perpendicular -/
def lines_perpendicular (l1 l2 : Line) : Prop :=
  sorry

/-- Predicate to check if a circle is tangent to a line -/
def circle_tangent_to_line (c : Circle) (l : Line) : Prop :=
  sorry

/-- Predicate to check if a point is on a circle -/
def point_on_circle (p : Point) (c : Circle) : Prop :=
  sorry

/-- Main theorem -/
theorem circle_construction_cases
  (a b : Line) (P : Point)
  (h1 : lines_intersect a b)
  (h2 : point_on_line P b) :
  (∃ c1 c2 : Circle,
    c1 ≠ c2 ∧
    circle_tangent_to_line c1 a ∧
    circle_tangent_to_line c2 a ∧
    point_on_circle P c1 ∧
    point_on_circle P c2 ∧
    point_on_line c1.center b ∧
    point_on_line c2.center b) ∨
  (∃ Q : Point, point_on_line Q a ∧ point_on_line Q b ∧ P = Q) ∨
  (lines_perpendicular a b) :=
sorry

end NUMINAMATH_CALUDE_circle_construction_cases_l3442_344208


namespace NUMINAMATH_CALUDE_matrices_are_inverses_l3442_344243

theorem matrices_are_inverses : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![4, -7; -5, 9]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![9, 7; 5, 4]
  A * B = 1 ∧ B * A = 1 := by
  sorry

end NUMINAMATH_CALUDE_matrices_are_inverses_l3442_344243


namespace NUMINAMATH_CALUDE_angle_complement_problem_l3442_344274

theorem angle_complement_problem (x : ℝ) : 
  x + 2 * (4 * x + 10) = 90 → x = 70 / 9 :=
by sorry

end NUMINAMATH_CALUDE_angle_complement_problem_l3442_344274


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3442_344287

theorem sqrt_equation_solution (y : ℝ) :
  Real.sqrt (3 + Real.sqrt (3 * y - 4)) = Real.sqrt 10 → y = 53 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3442_344287


namespace NUMINAMATH_CALUDE_pizza_toppings_count_l3442_344232

theorem pizza_toppings_count (n : ℕ) (h : n = 8) : 
  (n) + (n.choose 2) + (n.choose 3) = 92 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_count_l3442_344232


namespace NUMINAMATH_CALUDE_curve_has_axis_of_symmetry_l3442_344253

/-- The equation of the curve -/
def curve_equation (x y : ℝ) : Prop :=
  x^2 - x*y + y^2 + x - y - 1 = 0

/-- The proposed axis of symmetry -/
def axis_of_symmetry (x y : ℝ) : Prop :=
  x + y = 0

/-- Theorem stating that the curve has the given axis of symmetry -/
theorem curve_has_axis_of_symmetry :
  ∀ (x y : ℝ), curve_equation x y ↔ curve_equation (-y) (-x) :=
sorry

end NUMINAMATH_CALUDE_curve_has_axis_of_symmetry_l3442_344253


namespace NUMINAMATH_CALUDE_strawberry_jam_money_l3442_344271

-- Define the given conditions
def betty_strawberries : ℕ := 16
def matthew_strawberries : ℕ := betty_strawberries + 20
def natalie_strawberries : ℕ := matthew_strawberries / 2
def strawberries_per_jar : ℕ := 7
def price_per_jar : ℕ := 4

-- Define the theorem
theorem strawberry_jam_money : 
  (betty_strawberries + matthew_strawberries + natalie_strawberries) / strawberries_per_jar * price_per_jar = 40 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_jam_money_l3442_344271


namespace NUMINAMATH_CALUDE_one_hundred_twenty_fifth_number_with_digit_sum_5_l3442_344256

/-- A function that returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- A function that returns the nth natural number whose digits sum to 5 -/
def nth_number_with_digit_sum_5 (n : ℕ) : ℕ := sorry

/-- The main theorem stating that the 125th number with digit sum 5 is 41000 -/
theorem one_hundred_twenty_fifth_number_with_digit_sum_5 :
  nth_number_with_digit_sum_5 125 = 41000 := by sorry

end NUMINAMATH_CALUDE_one_hundred_twenty_fifth_number_with_digit_sum_5_l3442_344256


namespace NUMINAMATH_CALUDE_find_k_l3442_344262

theorem find_k (f g : ℝ → ℝ) (k : ℝ) : 
  (∀ x, f x = 7 * x^3 - 1/x + 5) →
  (∀ x, g x = x^3 - k) →
  f 3 - g 3 = 5 →
  k = -485/3 := by sorry

end NUMINAMATH_CALUDE_find_k_l3442_344262


namespace NUMINAMATH_CALUDE_torn_sheets_count_l3442_344234

/-- Represents a book with consecutively numbered pages, two per sheet. -/
structure Book where
  /-- The last page number in the book -/
  last_page : ℕ

/-- Represents a set of consecutively torn-out sheets from a book -/
structure TornSheets where
  /-- The first torn-out page number -/
  first_page : ℕ
  /-- The last torn-out page number -/
  last_page : ℕ

/-- Check if two numbers have the same digits -/
def same_digits (a b : ℕ) : Prop := sorry

/-- Calculate the number of sheets torn out -/
def sheets_torn_out (ts : TornSheets) : ℕ :=
  (ts.last_page - ts.first_page + 1) / 2

/-- Main theorem -/
theorem torn_sheets_count (b : Book) (ts : TornSheets) :
  ts.first_page = 185 →
  same_digits ts.first_page ts.last_page →
  Even ts.last_page →
  ts.last_page > ts.first_page →
  ts.last_page ≤ b.last_page →
  sheets_torn_out ts = 167 := by sorry

end NUMINAMATH_CALUDE_torn_sheets_count_l3442_344234


namespace NUMINAMATH_CALUDE_adlai_animal_legs_l3442_344240

/-- The number of legs a dog has -/
def dog_legs : ℕ := 4

/-- The number of legs a chicken has -/
def chicken_legs : ℕ := 2

/-- The number of dogs Adlai has -/
def adlai_dogs : ℕ := 2

/-- The number of chickens Adlai has -/
def adlai_chickens : ℕ := 1

/-- The total number of animal legs Adlai has -/
def total_legs : ℕ := adlai_dogs * dog_legs + adlai_chickens * chicken_legs

theorem adlai_animal_legs : total_legs = 10 := by
  sorry

end NUMINAMATH_CALUDE_adlai_animal_legs_l3442_344240


namespace NUMINAMATH_CALUDE_expansion_terms_imply_n_12_l3442_344219

def binomial_coefficient (n k : ℕ) : ℕ := sorry

theorem expansion_terms_imply_n_12 (x a : ℝ) (n : ℕ) :
  (binomial_coefficient n 3 * x^(n-3) * a^3 = 120) →
  (binomial_coefficient n 4 * x^(n-4) * a^4 = 360) →
  (binomial_coefficient n 5 * x^(n-5) * a^5 = 720) →
  n = 12 :=
sorry

end NUMINAMATH_CALUDE_expansion_terms_imply_n_12_l3442_344219


namespace NUMINAMATH_CALUDE_garbage_collection_difference_l3442_344259

theorem garbage_collection_difference (daliah_amount dewei_amount zane_amount : ℝ) : 
  daliah_amount = 17.5 →
  zane_amount = 62 →
  zane_amount = 4 * dewei_amount →
  dewei_amount < daliah_amount →
  daliah_amount - dewei_amount = 2 := by
sorry

end NUMINAMATH_CALUDE_garbage_collection_difference_l3442_344259


namespace NUMINAMATH_CALUDE_range_of_f_l3442_344265

noncomputable def f (x : ℝ) : ℝ := Real.arctan x + Real.arctan ((1 - x) / (1 + x)) + Real.arctan (2 * x)

theorem range_of_f :
  Set.range f = Set.Ioo (-Real.pi / 2) (Real.pi / 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l3442_344265


namespace NUMINAMATH_CALUDE_f_of_g_of_3_l3442_344231

/-- Given functions f and g, prove that f(2 + g(3)) = 44 -/
theorem f_of_g_of_3 (f g : ℝ → ℝ) 
    (hf : ∀ x, f x = 3 * x - 4)
    (hg : ∀ x, g x = x^2 + 2 * x - 1) : 
  f (2 + g 3) = 44 := by
  sorry

end NUMINAMATH_CALUDE_f_of_g_of_3_l3442_344231


namespace NUMINAMATH_CALUDE_unique_solution_l3442_344273

/-- A function from positive reals to positive reals -/
def PositiveRealFunction := {f : ℝ → ℝ // ∀ x, x > 0 → f x > 0}

/-- The functional equation that f must satisfy -/
def SatisfiesEquation (f : PositiveRealFunction) : Prop :=
  ∀ x y, x > 0 → y > 0 → f.val (x + f.val y) = f.val (x + y) + f.val y

/-- The theorem stating that the only solution is f(x) = 2x -/
theorem unique_solution (f : PositiveRealFunction) (h : SatisfiesEquation f) :
  ∀ x, x > 0 → f.val x = 2 * x :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l3442_344273


namespace NUMINAMATH_CALUDE_absolute_value_equation_roots_l3442_344278

theorem absolute_value_equation_roots : ∃ (x₁ x₂ : ℝ), 
  (x₁ ≠ x₂) ∧ 
  (|x₁|^2 + |x₁| - 12 = 0) ∧ 
  (|x₂|^2 + |x₂| - 12 = 0) ∧
  (x₁ + x₂ = 0) ∧
  (∀ x : ℝ, |x|^2 + |x| - 12 = 0 → (x = x₁ ∨ x = x₂)) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_roots_l3442_344278


namespace NUMINAMATH_CALUDE_expression_evaluation_l3442_344264

theorem expression_evaluation : (4 + 6 + 2) / 3 - 2 / 3 = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3442_344264


namespace NUMINAMATH_CALUDE_tan_pi_minus_alpha_l3442_344252

theorem tan_pi_minus_alpha (α : Real) (h : 3 * Real.sin (α - Real.pi) = Real.cos α) :
  Real.tan (Real.pi - α) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_pi_minus_alpha_l3442_344252


namespace NUMINAMATH_CALUDE_betty_cupcake_rate_l3442_344258

theorem betty_cupcake_rate : 
  ∀ (B : ℕ), -- Betty's cupcake rate per hour
  (5 * 8 - 3 * B = 10) → -- Difference in cupcakes after 5 hours
  B = 10 := by
sorry

end NUMINAMATH_CALUDE_betty_cupcake_rate_l3442_344258


namespace NUMINAMATH_CALUDE_some_number_value_l3442_344207

theorem some_number_value (x : ℝ) : 65 + 5 * x / (180 / 3) = 66 → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l3442_344207


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l3442_344206

def z₁ : ℂ := -3 + Complex.I
def z₂ : ℂ := 1 - Complex.I
def z : ℂ := z₁ - z₂

theorem z_in_second_quadrant : 
  z.re < 0 ∧ z.im > 0 := by sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l3442_344206


namespace NUMINAMATH_CALUDE_product_and_reciprocal_relation_sum_l3442_344241

theorem product_and_reciprocal_relation_sum (x y : ℝ) : 
  x > 0 → y > 0 → x * y = 16 → 1 / x = 3 / y → x + y = (16 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_product_and_reciprocal_relation_sum_l3442_344241


namespace NUMINAMATH_CALUDE_sandy_fish_count_l3442_344281

/-- The number of pet fish Sandy has after buying more -/
def total_fish (initial : ℕ) (bought : ℕ) : ℕ :=
  initial + bought

/-- Theorem stating that Sandy now has 32 pet fish -/
theorem sandy_fish_count : total_fish 26 6 = 32 := by
  sorry

end NUMINAMATH_CALUDE_sandy_fish_count_l3442_344281


namespace NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l3442_344267

theorem sum_of_absolute_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) : 
  (∀ x, (1 - 3*x)^9 = a₀ + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + 
                      a₅*(x+1)^5 + a₆*(x+1)^6 + a₇*(x+1)^7 + a₈*(x+1)^8 + a₉*(x+1)^9) →
  |a₀| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| + |a₇| + |a₈| + |a₉| = 7^9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l3442_344267


namespace NUMINAMATH_CALUDE_least_prime_factor_of_11_5_minus_11_4_l3442_344251

theorem least_prime_factor_of_11_5_minus_11_4 :
  Nat.minFac (11^5 - 11^4) = 2 := by sorry

end NUMINAMATH_CALUDE_least_prime_factor_of_11_5_minus_11_4_l3442_344251


namespace NUMINAMATH_CALUDE_nancy_football_games_l3442_344221

theorem nancy_football_games (games_this_month games_last_month games_next_month total_games : ℕ) :
  games_this_month = 9 →
  games_last_month = 8 →
  games_next_month = 7 →
  total_games = 24 →
  games_this_month + games_last_month + games_next_month = total_games :=
by sorry

end NUMINAMATH_CALUDE_nancy_football_games_l3442_344221


namespace NUMINAMATH_CALUDE_one_positive_integer_satisfies_condition_l3442_344254

theorem one_positive_integer_satisfies_condition : 
  ∃! (n : ℕ), n > 0 ∧ 21 - 3 * n > 15 :=
sorry

end NUMINAMATH_CALUDE_one_positive_integer_satisfies_condition_l3442_344254


namespace NUMINAMATH_CALUDE_parallelogram_area_l3442_344202

/-- The area of a parallelogram with base 30 cm and height 12 cm is 360 square centimeters. -/
theorem parallelogram_area : 
  ∀ (base height area : ℝ), 
  base = 30 → 
  height = 12 → 
  area = base * height →
  area = 360 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l3442_344202


namespace NUMINAMATH_CALUDE_sandwich_jam_cost_l3442_344239

theorem sandwich_jam_cost (N B J : ℕ) : 
  N > 1 → 
  B > 0 → 
  J > 0 → 
  N * (4 * B + 6 * J) = 351 → 
  N * J * 6 = 162 :=
by sorry

end NUMINAMATH_CALUDE_sandwich_jam_cost_l3442_344239


namespace NUMINAMATH_CALUDE_triangle_special_case_triangle_inequality_l3442_344257

-- Define a triangle structure
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  area : ℝ
  hpos : 0 < a ∧ 0 < b ∧ 0 < c
  htri : a + b > c ∧ b + c > a ∧ c + a > b

-- Part (a)
theorem triangle_special_case (t : Triangle) 
  (h : 6 * t.area = 2 * t.a^2 + t.b * t.c) : 
  t.b = t.c ∧ t.b = Real.sqrt (5/2) * t.a :=
sorry

-- Part (b)
theorem triangle_inequality (t : Triangle) :
  3 * t.a^2 + 3 * t.b^2 - t.c^2 ≥ 4 * Real.sqrt 3 * t.area :=
sorry

end NUMINAMATH_CALUDE_triangle_special_case_triangle_inequality_l3442_344257


namespace NUMINAMATH_CALUDE_water_tank_capacity_l3442_344291

theorem water_tank_capacity : ∀ (c : ℝ),
  (c / 3 : ℝ) / c = 1 / 3 →
  ((c / 3 + 7) : ℝ) / c = 2 / 5 →
  c = 105 := by
sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l3442_344291


namespace NUMINAMATH_CALUDE_triangle_max_area_l3442_344209

theorem triangle_max_area (A B C : ℝ) (h1 : 0 < A ∧ A < π) (h2 : 0 < B ∧ B < π) (h3 : 0 < C ∧ C < π) 
  (h4 : A + B + C = π) (h5 : Real.cos A / Real.sin B + Real.cos B / Real.sin A = 2) 
  (h6 : ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 12 ∧ 
    a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C) :
  ∃ (S : ℝ), S ≤ 36 * (3 - 2 * Real.sqrt 2) ∧ 
    (∀ (S' : ℝ), (∃ (a' b' c' : ℝ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ a' + b' + c' = 12 ∧ 
      a' / Real.sin A = b' / Real.sin B ∧ b' / Real.sin B = c' / Real.sin C ∧ 
      S' = 1/2 * a' * b' * Real.sin C) → S' ≤ S) :=
sorry

end NUMINAMATH_CALUDE_triangle_max_area_l3442_344209


namespace NUMINAMATH_CALUDE_exists_four_unacquainted_l3442_344270

/-- A type representing a person in the group -/
def Person : Type := Fin 10

/-- The acquaintance relation between people -/
def acquainted : Person → Person → Prop := sorry

theorem exists_four_unacquainted 
  (h1 : ∀ p : Person, ∃! (q r : Person), q ≠ r ∧ acquainted p q ∧ acquainted p r)
  (h2 : ∀ p q : Person, acquainted p q → acquainted q p) :
  ∃ (a b c d : Person), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    ¬acquainted a b ∧ ¬acquainted a c ∧ ¬acquainted a d ∧
    ¬acquainted b c ∧ ¬acquainted b d ∧ ¬acquainted c d :=
sorry

end NUMINAMATH_CALUDE_exists_four_unacquainted_l3442_344270


namespace NUMINAMATH_CALUDE_d_values_l3442_344242

def a (n : ℕ) : ℕ := 20 + n^2

def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem d_values : {n : ℕ | n > 0} → {d n | n : ℕ} = {1, 3, 9, 27, 81} := by sorry

end NUMINAMATH_CALUDE_d_values_l3442_344242


namespace NUMINAMATH_CALUDE_student_selection_theorem_l3442_344235

def number_of_boys : ℕ := 4
def number_of_girls : ℕ := 3
def total_to_select : ℕ := 3

theorem student_selection_theorem :
  (Nat.choose number_of_boys 2 * Nat.choose number_of_girls 1) +
  (Nat.choose number_of_boys 1 * Nat.choose number_of_girls 2) = 30 := by
  sorry

end NUMINAMATH_CALUDE_student_selection_theorem_l3442_344235


namespace NUMINAMATH_CALUDE_fraction_equality_l3442_344255

theorem fraction_equality (a b : ℝ) (h1 : a ≠ b) (h2 : b ≠ 0) :
  let x := a / b
  (2 * a + b) / (a - 2 * b) = (2 * x + 1) / (x - 2) := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l3442_344255
