import Mathlib

namespace NUMINAMATH_CALUDE_students_per_class_l1392_139204

theorem students_per_class 
  (total_students : ℕ) 
  (num_classrooms : ℕ) 
  (h1 : total_students = 120) 
  (h2 : num_classrooms = 24) 
  (h3 : total_students % num_classrooms = 0) : 
  total_students / num_classrooms = 5 := by
sorry

end NUMINAMATH_CALUDE_students_per_class_l1392_139204


namespace NUMINAMATH_CALUDE_x_plus_y_value_l1392_139238

theorem x_plus_y_value (x y : ℝ) (h1 : 1/x = 2) (h2 : 1/x + 3/y = 3) : x + y = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l1392_139238


namespace NUMINAMATH_CALUDE_imaginary_part_of_one_minus_two_i_squared_l1392_139280

theorem imaginary_part_of_one_minus_two_i_squared (i : ℂ) : 
  i * i = -1 → Complex.im ((1 - 2*i)^2) = -4 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_one_minus_two_i_squared_l1392_139280


namespace NUMINAMATH_CALUDE_river_crossing_possible_l1392_139260

/-- Represents the state of the river crossing -/
structure RiverState where
  left_soldiers : Nat
  left_robbers : Nat
  right_soldiers : Nat
  right_robbers : Nat

/-- Represents a boat trip -/
inductive BoatTrip
  | SoldierSoldier
  | SoldierRobber
  | RobberRobber
  | Soldier
  | Robber

/-- Checks if a state is safe (soldiers not outnumbered by robbers) -/
def is_safe_state (state : RiverState) : Prop :=
  (state.left_soldiers ≥ state.left_robbers || state.left_soldiers = 0) &&
  (state.right_soldiers ≥ state.right_robbers || state.right_soldiers = 0)

/-- Applies a boat trip to a state -/
def apply_trip (state : RiverState) (trip : BoatTrip) (direction : Bool) : RiverState :=
  sorry

/-- Checks if the final state is reached -/
def is_final_state (state : RiverState) : Prop :=
  state.left_soldiers = 0 && state.left_robbers = 0 &&
  state.right_soldiers = 3 && state.right_robbers = 3

/-- Theorem: There exists a sequence of boat trips that safely transports everyone across -/
theorem river_crossing_possible : ∃ (trips : List (BoatTrip × Bool)),
  let final_state := trips.foldl (λ s (trip, dir) => apply_trip s trip dir)
    (RiverState.mk 3 3 0 0)
  is_final_state final_state ∧
  ∀ (intermediate_state : RiverState),
    intermediate_state ∈ trips.scanl (λ s (trip, dir) => apply_trip s trip dir)
      (RiverState.mk 3 3 0 0) →
    is_safe_state intermediate_state :=
  sorry

end NUMINAMATH_CALUDE_river_crossing_possible_l1392_139260


namespace NUMINAMATH_CALUDE_scale_model_height_l1392_139246

/-- The scale ratio used for the model -/
def scale_ratio : ℚ := 1 / 20

/-- The actual height of the United States Capitol in feet -/
def actual_height : ℕ := 289

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (q : ℚ) : ℤ :=
  ⌊q + 1/2⌋

/-- The height of the scale model rounded to the nearest foot -/
def model_height : ℕ := (round_to_nearest ((actual_height : ℚ) / scale_ratio)).toNat

theorem scale_model_height :
  model_height = 14 := by sorry

end NUMINAMATH_CALUDE_scale_model_height_l1392_139246


namespace NUMINAMATH_CALUDE_star_properties_l1392_139286

-- Define the binary operation
def star (x y : ℝ) : ℝ := (x + 1) * (y + 1) - 1

-- Theorem statement
theorem star_properties :
  (∀ x y : ℝ, star x y = star y x) ∧ 
  (∀ x : ℝ, star x (-1) = x ∧ star (-1) x = x) ∧
  (∀ x : ℝ, star x x = x^2 + 2*x) := by
  sorry

end NUMINAMATH_CALUDE_star_properties_l1392_139286


namespace NUMINAMATH_CALUDE_abc_over_def_value_l1392_139208

theorem abc_over_def_value (a b c d e f : ℝ) 
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : e / f = 1 / 10)
  (h_nonzero : b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ f ≠ 0) : 
  a * b * c / (d * e * f) = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_abc_over_def_value_l1392_139208


namespace NUMINAMATH_CALUDE_payment_calculation_l1392_139243

theorem payment_calculation (rate : ℚ) (rooms : ℚ) : 
  rate = 13/3 → rooms = 8/5 → rate * rooms = 104/15 := by
  sorry

end NUMINAMATH_CALUDE_payment_calculation_l1392_139243


namespace NUMINAMATH_CALUDE_area_of_triangle_WRX_l1392_139203

-- Define the points
variable (W X Y Z P Q R : ℝ × ℝ)

-- Define the conditions
def is_rectangle (A B C D : ℝ × ℝ) : Prop := sorry
def on_line (P A B : ℝ × ℝ) : Prop := sorry
def distance (A B : ℝ × ℝ) : ℝ := sorry
def intersect (A B C D E : ℝ × ℝ) : Prop := sorry
def area_triangle (A B C : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem area_of_triangle_WRX 
  (h1 : is_rectangle W X Y Z)
  (h2 : distance W Z = 7)
  (h3 : distance X Y = 4)
  (h4 : on_line P Y Z)
  (h5 : on_line Q Y Z)
  (h6 : distance Y P = 2)
  (h7 : distance Q Z = 3)
  (h8 : intersect W P X Q R) :
  area_triangle W R X = 98/5 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_WRX_l1392_139203


namespace NUMINAMATH_CALUDE_max_cards_per_box_l1392_139266

/-- Given a total of 94 cards and 6 cards in an unfilled box, 
    prove that the maximum number of cards a full box can hold is 22. -/
theorem max_cards_per_box (total_cards : ℕ) (cards_in_unfilled_box : ℕ) 
  (h1 : total_cards = 94) (h2 : cards_in_unfilled_box = 6) :
  ∃ (max_cards_per_box : ℕ), 
    max_cards_per_box = 22 ∧ 
    max_cards_per_box > cards_in_unfilled_box ∧
    (total_cards - cards_in_unfilled_box) % max_cards_per_box = 0 ∧
    ∀ n : ℕ, n > max_cards_per_box → (total_cards - cards_in_unfilled_box) % n ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_max_cards_per_box_l1392_139266


namespace NUMINAMATH_CALUDE_total_committees_is_160_l1392_139292

/-- Represents the structure of a committee -/
inductive CommitteeStructure
  | FiveSenators
  | FourSenatorsAndFourAides
  | TwoSenatorsAndTwelveAides

/-- The number of senators -/
def numSenators : ℕ := 100

/-- The number of aides each senator has -/
def aidesPerSenator : ℕ := 4

/-- The number of committees each senator serves on -/
def committeesPerSenator : ℕ := 5

/-- The number of committees each aide serves on -/
def committeesPerAide : ℕ := 3

/-- The total number of committees -/
def totalCommittees : ℕ := 160

/-- Theorem stating that the total number of committees is 160 -/
theorem total_committees_is_160 :
  totalCommittees = 160 :=
by sorry

end NUMINAMATH_CALUDE_total_committees_is_160_l1392_139292


namespace NUMINAMATH_CALUDE_prob_different_colors_specific_l1392_139255

/-- The probability of drawing two chips of different colors -/
def prob_different_colors (blue yellow red : ℕ) : ℚ :=
  let total := blue + yellow + red
  let p_blue := blue / total
  let p_yellow := yellow / total
  let p_red := red / total
  p_blue * (p_yellow + p_red) + p_yellow * (p_blue + p_red) + p_red * (p_blue + p_yellow)

/-- Theorem stating the probability of drawing two chips of different colors -/
theorem prob_different_colors_specific : prob_different_colors 6 4 2 = 11 / 18 := by
  sorry

#eval prob_different_colors 6 4 2

end NUMINAMATH_CALUDE_prob_different_colors_specific_l1392_139255


namespace NUMINAMATH_CALUDE_unique_k_for_lcm_l1392_139247

def lcm (a b c : ℕ) : ℕ := Nat.lcm a (Nat.lcm b c)

theorem unique_k_for_lcm : ∃! k : ℕ+, lcm (6^6) (9^9) k = 18^18 := by
  sorry

end NUMINAMATH_CALUDE_unique_k_for_lcm_l1392_139247


namespace NUMINAMATH_CALUDE_sum_of_cyclic_equations_l1392_139235

theorem sum_of_cyclic_equations (p q r : ℕ+) 
  (eq1 : p * q + r = 47)
  (eq2 : q * r + p = 47)
  (eq3 : r * p + q = 47) :
  p + q + r = 48 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cyclic_equations_l1392_139235


namespace NUMINAMATH_CALUDE_inequality_solution_l1392_139274

theorem inequality_solution (a b c : ℝ) (h1 : a < b)
  (h2 : ∀ x : ℝ, (x - a) * (x - b) / (x - c) ≤ 0 ↔ x < -6 ∨ |x - 31| ≤ 1) :
  a + 2 * b + 3 * c = 76 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1392_139274


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l1392_139265

theorem arithmetic_sequence_middle_term (a₁ a₃ z : ℤ) : 
  a₁ = 3^2 → a₃ = 3^4 → (a₃ - z = z - a₁) → z = 45 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l1392_139265


namespace NUMINAMATH_CALUDE_quadratic_equation_from_roots_l1392_139272

theorem quadratic_equation_from_roots (α β : ℝ) (h1 : α + β = 5) (h2 : α * β = 6) :
  ∃ a b c : ℝ, a ≠ 0 ∧ a * α^2 + b * α + c = 0 ∧ a * β^2 + b * β + c = 0 ∧ 
  a = 1 ∧ b = -5 ∧ c = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_from_roots_l1392_139272


namespace NUMINAMATH_CALUDE_quarters_in_jar_l1392_139206

/-- Represents the number of coins of each type in the jar -/
structure CoinCounts where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ
  half_dollars : ℕ
  dollar_coins : ℕ
  two_dollar_coins : ℕ

/-- Represents the cost of a sundae and its modifications -/
structure SundaeCost where
  base : ℚ
  special_topping : ℚ
  featured_flavor : ℚ

/-- Represents the family's ice cream trip details -/
structure IceCreamTrip where
  family_size : ℕ
  special_toppings : ℕ
  featured_flavors : ℕ
  leftover : ℚ

def count_quarters (coins : CoinCounts) (sundae : SundaeCost) (trip : IceCreamTrip) : ℕ :=
  sorry

theorem quarters_in_jar 
  (coins : CoinCounts)
  (sundae : SundaeCost)
  (trip : IceCreamTrip) :
  coins.pennies = 123 ∧ 
  coins.nickels = 85 ∧ 
  coins.dimes = 35 ∧ 
  coins.half_dollars = 15 ∧ 
  coins.dollar_coins = 5 ∧ 
  coins.two_dollar_coins = 4 ∧
  sundae.base = 5.25 ∧
  sundae.special_topping = 0.5 ∧
  sundae.featured_flavor = 0.25 ∧
  trip.family_size = 8 ∧
  trip.special_toppings = 3 ∧
  trip.featured_flavors = 5 ∧
  trip.leftover = 0.97 →
  count_quarters coins sundae trip = 54 :=
by sorry

end NUMINAMATH_CALUDE_quarters_in_jar_l1392_139206


namespace NUMINAMATH_CALUDE_log_expression_equals_two_l1392_139249

-- Define base 10 logarithm
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_expression_equals_two :
  lg 2 ^ 2 + lg 2 * lg 5 + lg 50 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equals_two_l1392_139249


namespace NUMINAMATH_CALUDE_largest_a_less_than_l1392_139299

theorem largest_a_less_than (a b : ℤ) : 
  9 < a → 
  19 < b → 
  b < 31 → 
  (a : ℚ) / (b : ℚ) ≤ 2/3 → 
  a < 21 :=
by sorry

end NUMINAMATH_CALUDE_largest_a_less_than_l1392_139299


namespace NUMINAMATH_CALUDE_car_distance_calculation_l1392_139261

/-- Proves that a car traveling at 260 km/h for 2 2/5 hours covers a distance of 624 km -/
theorem car_distance_calculation (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 260 → time = 2 + 2/5 → distance = speed * time → distance = 624 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_calculation_l1392_139261


namespace NUMINAMATH_CALUDE_tammy_orange_picking_l1392_139250

/-- Proves that given the conditions of Tammy's orange selling business, 
    she picks 12 oranges from each tree each day. -/
theorem tammy_orange_picking :
  let num_trees : ℕ := 10
  let oranges_per_pack : ℕ := 6
  let price_per_pack : ℕ := 2
  let total_earnings : ℕ := 840
  let num_weeks : ℕ := 3
  let days_per_week : ℕ := 7

  (num_trees > 0) →
  (oranges_per_pack > 0) →
  (price_per_pack > 0) →
  (total_earnings > 0) →
  (num_weeks > 0) →
  (days_per_week > 0) →

  (total_earnings / price_per_pack * oranges_per_pack) / (num_weeks * days_per_week) / num_trees = 12 :=
by
  sorry


end NUMINAMATH_CALUDE_tammy_orange_picking_l1392_139250


namespace NUMINAMATH_CALUDE_unique_modular_solution_l1392_139254

theorem unique_modular_solution : ∃! n : ℕ, n < 251 ∧ (250 * n) % 251 = 123 % 251 := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_solution_l1392_139254


namespace NUMINAMATH_CALUDE_solve_inequality_1_solve_inequality_2_l1392_139289

-- Inequality 1
theorem solve_inequality_1 : 
  {x : ℝ | x^2 + x - 6 < 0} = {x : ℝ | -3 < x ∧ x < 2} := by sorry

-- Inequality 2
theorem solve_inequality_2 : 
  {x : ℝ | -6*x^2 - x + 2 ≤ 0} = {x : ℝ | x ≤ -2/3 ∨ x ≥ 1/2} := by sorry

end NUMINAMATH_CALUDE_solve_inequality_1_solve_inequality_2_l1392_139289


namespace NUMINAMATH_CALUDE_adjacent_above_350_l1392_139234

/-- Represents a position in the triangular grid -/
structure GridPosition where
  row : ℕ
  column : ℕ

/-- Returns the number at a given position in the triangular grid -/
def numberAt (pos : GridPosition) : ℕ := sorry

/-- Returns the position of a given number in the triangular grid -/
def positionOf (n : ℕ) : GridPosition := sorry

/-- Returns the number in the horizontally adjacent triangle in the row above -/
def adjacentAbove (n : ℕ) : ℕ := sorry

theorem adjacent_above_350 : adjacentAbove 350 = 314 := by sorry

end NUMINAMATH_CALUDE_adjacent_above_350_l1392_139234


namespace NUMINAMATH_CALUDE_sock_pairs_same_color_l1392_139242

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem sock_pairs_same_color (red green yellow : ℕ) 
  (h_red : red = 5) (h_green : green = 6) (h_yellow : yellow = 4) :
  choose red 2 + choose green 2 + choose yellow 2 = 31 := by
  sorry

end NUMINAMATH_CALUDE_sock_pairs_same_color_l1392_139242


namespace NUMINAMATH_CALUDE_quadratic_equations_solution_l1392_139207

theorem quadratic_equations_solution :
  -- Part I
  let eq1 : ℝ → Prop := λ x ↦ x^2 + 6*x + 5 = 0
  ∃ x1 x2 : ℝ, eq1 x1 ∧ eq1 x2 ∧ x1 = -5 ∧ x2 = -1 ∧
  -- Part II
  ∀ k : ℝ,
    let eq2 : ℝ → Prop := λ x ↦ x^2 - 3*x + k = 0
    (∃ x1 x2 : ℝ, eq2 x1 ∧ eq2 x2 ∧ (x1 - 1) * (x2 - 1) = -6) →
    k = -4 ∧ ∃ x1 x2 : ℝ, eq2 x1 ∧ eq2 x2 ∧ x1 = 4 ∧ x2 = -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solution_l1392_139207


namespace NUMINAMATH_CALUDE_root_sum_theorem_l1392_139264

-- Define the polynomial
def polynomial (a b c d x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- State the theorem
theorem root_sum_theorem (a b c d : ℝ) (h_a : a ≠ 0) :
  polynomial a b c d 4 = 0 ∧
  polynomial a b c d (-1) = 0 ∧
  polynomial a b c d (-3) = 0 →
  (b + c) / a = -1441 / 37 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l1392_139264


namespace NUMINAMATH_CALUDE_exists_special_number_l1392_139252

def is_twelve_digit (n : ℕ) : Prop := 10^11 ≤ n ∧ n < 10^12

def is_not_perfect_square (n : ℕ) : Prop := ∃ (d : ℕ), n % 10 = d ∧ (d = 2 ∨ d = 3 ∨ d = 7 ∨ d = 8)

def is_ambiguous_cube (n : ℕ) : Prop := ∀ (d : ℕ), d < 10 → ∃ (k : ℕ), k^3 % 10 = d

theorem exists_special_number : ∃ (n : ℕ), 
  is_twelve_digit n ∧ 
  is_not_perfect_square n ∧ 
  is_ambiguous_cube n := by
  sorry

end NUMINAMATH_CALUDE_exists_special_number_l1392_139252


namespace NUMINAMATH_CALUDE_furniture_cost_price_l1392_139298

theorem furniture_cost_price (price : ℝ) (discount : ℝ) (profit : ℝ) :
  price = 132 ∧ 
  discount = 0.1 ∧ 
  profit = 0.1 ∧ 
  price * (1 - discount) = (1 + profit) * (price * (1 - discount) / (1 + profit)) →
  price * (1 - discount) / (1 + profit) = 108 :=
by sorry

end NUMINAMATH_CALUDE_furniture_cost_price_l1392_139298


namespace NUMINAMATH_CALUDE_sequence_equation_l1392_139239

theorem sequence_equation (n : ℕ+) : 9 * n + (n - 1) = 10 * n - 1 := by
  sorry

#check sequence_equation

end NUMINAMATH_CALUDE_sequence_equation_l1392_139239


namespace NUMINAMATH_CALUDE_correct_selling_price_B_l1392_139273

/-- Represents the pricing and sales data for laundry detergents --/
structure LaundryDetergentData where
  cost_diff : ℝ               -- Cost difference between brands
  total_cost_A : ℝ            -- Total cost for brand A
  total_cost_B : ℝ            -- Total cost for brand B
  sell_price_A : ℝ            -- Selling price of brand A
  daily_sales_A : ℝ           -- Daily sales of brand A
  base_price_B : ℝ            -- Base selling price of brand B
  base_sales_B : ℝ            -- Base daily sales of brand B
  price_sales_ratio : ℝ       -- Ratio of price increase to sales decrease for B

/-- Calculates the selling price of brand B for a given total daily profit --/
def calculate_selling_price_B (data : LaundryDetergentData) (total_profit : ℝ) : ℝ :=
  sorry

/-- Theorem stating the correct selling price for brand B --/
theorem correct_selling_price_B (data : LaundryDetergentData) :
  let d := {
    cost_diff := 10,
    total_cost_A := 3000,
    total_cost_B := 4000,
    sell_price_A := 45,
    daily_sales_A := 100,
    base_price_B := 50,
    base_sales_B := 140,
    price_sales_ratio := 2
  }
  calculate_selling_price_B d 4700 = 80 := by sorry

end NUMINAMATH_CALUDE_correct_selling_price_B_l1392_139273


namespace NUMINAMATH_CALUDE_one_more_tile_possible_l1392_139210

/-- Represents a checkerboard -/
structure Checkerboard :=
  (size : ℕ)

/-- Represents a T-shaped tile -/
structure TTile :=
  (squares_covered : ℕ)

/-- The number of squares that remain uncovered after placing T-tiles -/
def uncovered_squares (board : Checkerboard) (tiles : ℕ) (tile : TTile) : ℕ :=
  board.size ^ 2 - tiles * tile.squares_covered

/-- Theorem stating that one more T-tile can be placed on the checkerboard -/
theorem one_more_tile_possible (board : Checkerboard) (tiles : ℕ) (tile : TTile) :
  board.size = 100 →
  tiles = 800 →
  tile.squares_covered = 4 →
  uncovered_squares board tiles tile ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_one_more_tile_possible_l1392_139210


namespace NUMINAMATH_CALUDE_exists_valid_superchess_configuration_l1392_139251

/-- Represents a chess piece in the game of superchess -/
structure Piece where
  id : Fin 20

/-- Represents a position on the superchess board -/
structure Position where
  x : Fin 100
  y : Fin 100

/-- Represents the superchess board -/
def Board := Fin 100 → Fin 100 → Option Piece

/-- Predicate to check if a piece attacks a position -/
def attacks (p : Piece) (pos : Position) (board : Board) : Prop :=
  ∃ (attacked : Finset Position), attacked.card ≤ 20 ∧ pos ∈ attacked

/-- Predicate to check if a board configuration is valid (no piece attacks another) -/
def valid_board (board : Board) : Prop :=
  ∀ (p₁ p₂ : Piece) (pos₁ pos₂ : Position),
    board pos₁.x pos₁.y = some p₁ →
    board pos₂.x pos₂.y = some p₂ →
    p₁ ≠ p₂ →
    ¬(attacks p₁ pos₂ board ∨ attacks p₂ pos₁ board)

/-- Theorem stating that there exists a valid board configuration -/
theorem exists_valid_superchess_configuration :
  ∃ (board : Board), (∀ p : Piece, ∃ pos : Position, board pos.x pos.y = some p) ∧ valid_board board :=
sorry

end NUMINAMATH_CALUDE_exists_valid_superchess_configuration_l1392_139251


namespace NUMINAMATH_CALUDE_ellipse_range_theorem_l1392_139291

theorem ellipse_range_theorem :
  ∀ x y : ℝ, x^2/4 + y^2 = 1 → -Real.sqrt 17 ≤ 2*x + y ∧ 2*x + y ≤ Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_range_theorem_l1392_139291


namespace NUMINAMATH_CALUDE_perpendicular_planes_l1392_139223

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (is_perpendicular : Line → Plane → Prop)
variable (is_subset : Line → Plane → Prop)
variable (planes_perpendicular : Plane → Plane → Prop)

-- Define m, n as different lines
variable (m n : Line)
variable (h_diff_lines : m ≠ n)

-- Define α, β as different planes
variable (α β : Plane)
variable (h_diff_planes : α ≠ β)

-- State the theorem
theorem perpendicular_planes 
  (h1 : is_perpendicular m α) 
  (h2 : is_subset m β) : 
  planes_perpendicular α β := by sorry

end NUMINAMATH_CALUDE_perpendicular_planes_l1392_139223


namespace NUMINAMATH_CALUDE_sine_function_properties_l1392_139269

theorem sine_function_properties (ω φ : ℝ) (f : ℝ → ℝ) 
  (h_ω_pos : ω > 0)
  (h_φ_bound : |φ| < π / 2)
  (h_f_def : ∀ x, f x = Real.sin (ω * x + φ))
  (h_period : ∀ x, f (x + π) = f x)
  (h_f_zero : f 0 = 1 / 2) :
  (ω = 2) ∧ 
  (∀ x, f (π / 3 - x) = f (π / 3 + x)) ∧
  (∀ k : ℤ, ∀ x ∈ Set.Icc (k * π - π / 3) (k * π + π / 6), 
    ∀ y ∈ Set.Icc (k * π - π / 3) (k * π + π / 6),
    x ≤ y → f x ≤ f y) :=
by sorry

end NUMINAMATH_CALUDE_sine_function_properties_l1392_139269


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1392_139282

def A : Set ℝ := {x | x^2 - x ≤ 0}
def B : Set ℝ := {0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1392_139282


namespace NUMINAMATH_CALUDE_gcd_8251_6105_l1392_139232

theorem gcd_8251_6105 : Nat.gcd 8251 6105 = 37 := by
  sorry

end NUMINAMATH_CALUDE_gcd_8251_6105_l1392_139232


namespace NUMINAMATH_CALUDE_circle_sum_bounds_l1392_139270

/-- The circle defined by the equation x² + y² - 4x + 2 = 0 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 4*p.1 + 2 = 0}

/-- The sum function x + y for points (x, y) on the circle -/
def sum_func (p : ℝ × ℝ) : ℝ := p.1 + p.2

theorem circle_sum_bounds :
  ∃ (min max : ℝ), min = 0 ∧ max = 4 ∧
  ∀ p ∈ Circle, min ≤ sum_func p ∧ sum_func p ≤ max :=
sorry

end NUMINAMATH_CALUDE_circle_sum_bounds_l1392_139270


namespace NUMINAMATH_CALUDE_basketball_team_selection_with_twins_l1392_139279

def number_of_players : ℕ := 16
def number_of_starters : ℕ := 7

theorem basketball_team_selection_with_twins :
  (Nat.choose (number_of_players - 2) (number_of_starters - 2)) +
  (Nat.choose (number_of_players - 2) number_of_starters) =
  (Nat.choose 14 5) + (Nat.choose 14 7) :=
by sorry

end NUMINAMATH_CALUDE_basketball_team_selection_with_twins_l1392_139279


namespace NUMINAMATH_CALUDE_mod_thirteen_four_eleven_l1392_139236

theorem mod_thirteen_four_eleven (m : ℕ) : 
  13^4 % 11 = m ∧ 0 ≤ m ∧ m < 11 → m = 5 := by
  sorry

end NUMINAMATH_CALUDE_mod_thirteen_four_eleven_l1392_139236


namespace NUMINAMATH_CALUDE_magic_square_g_value_l1392_139297

/-- Represents a 3x3 multiplicative magic square --/
structure MagicSquare where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  d : ℕ+
  e : ℕ+
  f : ℕ+
  g : ℕ+
  h : ℕ+
  i : ℕ+
  row_product : a * b * c = d * e * f ∧ d * e * f = g * h * i
  col_product : a * d * g = b * e * h ∧ b * e * h = c * f * i
  diag_product : a * e * i = c * e * g

/-- The theorem stating that the only possible value for g is 3 --/
theorem magic_square_g_value (ms : MagicSquare) (h1 : ms.a = 90) (h2 : ms.i = 3) :
  ms.g = 3 :=
sorry

end NUMINAMATH_CALUDE_magic_square_g_value_l1392_139297


namespace NUMINAMATH_CALUDE_percent_value_in_quarters_l1392_139224

theorem percent_value_in_quarters : 
  let num_dimes : ℕ := 40
  let num_quarters : ℕ := 30
  let num_nickels : ℕ := 10
  let value_dime : ℕ := 10
  let value_quarter : ℕ := 25
  let value_nickel : ℕ := 5
  let total_value : ℕ := num_dimes * value_dime + num_quarters * value_quarter + num_nickels * value_nickel
  let quarter_value : ℕ := num_quarters * value_quarter
  (quarter_value : ℚ) / (total_value : ℚ) * 100 = 62.5
  := by sorry

end NUMINAMATH_CALUDE_percent_value_in_quarters_l1392_139224


namespace NUMINAMATH_CALUDE_basketball_probabilities_l1392_139220

def probability_A : ℝ := 0.7
def shots : ℕ := 3

theorem basketball_probabilities (a : ℝ) 
  (h1 : 0 ≤ a ∧ a ≤ 1) 
  (h2 : (Nat.choose 3 2 : ℝ) * (1 - probability_A) * probability_A^2 + probability_A^3 - a^3 = 0.659) :
  a = 0.5 ∧ 
  (1 - probability_A)^3 * (1 - a)^3 + 
  (Nat.choose 3 1 : ℝ) * (1 - probability_A)^2 * probability_A * 
  (Nat.choose 3 1 : ℝ) * (1 - a)^2 * a = 0.07425 := by
  sorry

end NUMINAMATH_CALUDE_basketball_probabilities_l1392_139220


namespace NUMINAMATH_CALUDE_cost_price_calculation_l1392_139276

/-- Proves that given a selling price of 400 and a profit percentage of 60%, 
    the cost price of the article is 250. -/
theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) :
  selling_price = 400 ∧ profit_percentage = 60 →
  (selling_price / (1 + profit_percentage / 100) : ℝ) = 250 := by
sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l1392_139276


namespace NUMINAMATH_CALUDE_base_conversion_subtraction_l1392_139227

-- Define a function to convert a number from base 8 to base 10
def base8ToBase10 (n : Nat) : Nat :=
  -- Implementation details are omitted
  sorry

-- Define a function to convert a number from base 9 to base 10
def base9ToBase10 (n : Nat) : Nat :=
  -- Implementation details are omitted
  sorry

-- Theorem statement
theorem base_conversion_subtraction :
  base8ToBase10 76432 - base9ToBase10 2541 = 30126 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_subtraction_l1392_139227


namespace NUMINAMATH_CALUDE_apartment_exchange_in_two_days_l1392_139221

universe u

theorem apartment_exchange_in_two_days {α : Type u} [Finite α] :
  ∀ (f : α → α), Function.Bijective f →
  ∃ (g h : α → α), Function.Involutive g ∧ Function.Involutive h ∧ f = g ∘ h :=
by sorry

end NUMINAMATH_CALUDE_apartment_exchange_in_two_days_l1392_139221


namespace NUMINAMATH_CALUDE_function_equivalence_l1392_139216

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * (cos x)^2 - Real.sqrt 3 * sin (2 * x)

noncomputable def g (x : ℝ) : ℝ := 2 * sin (2 * x) + 1

theorem function_equivalence : ∀ x : ℝ, f x = g (x + 5 * π / 12) := by sorry

end NUMINAMATH_CALUDE_function_equivalence_l1392_139216


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l1392_139245

theorem quadratic_equation_solutions :
  (∀ x, x^2 - 9 = 0 ↔ x = 3 ∨ x = -3) ∧
  (∀ x, x^2 + 2*x - 1 = 0 ↔ x = -1 + Real.sqrt 2 ∨ x = -1 - Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l1392_139245


namespace NUMINAMATH_CALUDE_train_speed_proof_l1392_139209

/-- The average speed of a train with stoppages, in km/h -/
def speed_with_stoppages : ℝ := 60

/-- The duration of stoppages per hour, in minutes -/
def stoppage_duration : ℝ := 15

/-- The average speed of a train without stoppages, in km/h -/
def speed_without_stoppages : ℝ := 80

theorem train_speed_proof :
  speed_without_stoppages * ((60 - stoppage_duration) / 60) = speed_with_stoppages :=
by sorry

end NUMINAMATH_CALUDE_train_speed_proof_l1392_139209


namespace NUMINAMATH_CALUDE_lcm_hcf_problem_l1392_139226

/-- Given two natural numbers a and b, their LCM is 2310 and a is 462, prove that their HCF is 1 -/
theorem lcm_hcf_problem (a b : ℕ) (h1 : a = 462) (h2 : Nat.lcm a b = 2310) : Nat.gcd a b = 1 := by
  sorry

end NUMINAMATH_CALUDE_lcm_hcf_problem_l1392_139226


namespace NUMINAMATH_CALUDE_semicircle_perimeter_approx_l1392_139200

/-- The perimeter of a semicircle with radius 12 is approximately 61.7 -/
theorem semicircle_perimeter_approx : ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  abs ((12 * Real.pi + 24) - 61.7) < ε :=
by sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_approx_l1392_139200


namespace NUMINAMATH_CALUDE_last_period_production_theorem_l1392_139222

/-- Represents the TV production scenario in a factory --/
structure TVProduction where
  total_days : ℕ
  first_period_days : ℕ
  first_period_avg : ℕ
  monthly_avg : ℕ

/-- Calculates the average daily production for the last period --/
def last_period_avg (prod : TVProduction) : ℚ :=
  let total_production := prod.total_days * prod.monthly_avg
  let first_period_production := prod.first_period_days * prod.first_period_avg
  let last_period_days := prod.total_days - prod.first_period_days
  (total_production - first_period_production) / last_period_days

/-- Theorem stating the average production for the last 5 days --/
theorem last_period_production_theorem (prod : TVProduction) 
  (h1 : prod.total_days = 30)
  (h2 : prod.first_period_days = 25)
  (h3 : prod.first_period_avg = 63)
  (h4 : prod.monthly_avg = 58) :
  last_period_avg prod = 33 := by
  sorry

end NUMINAMATH_CALUDE_last_period_production_theorem_l1392_139222


namespace NUMINAMATH_CALUDE_square_side_length_l1392_139284

/-- A square with four identical isosceles triangles on its sides -/
structure SquareWithTriangles where
  /-- Side length of the square -/
  s : ℝ
  /-- Area of one isosceles triangle -/
  triangle_area : ℝ
  /-- The total area of the isosceles triangles equals the area of the remaining region -/
  area_equality : 4 * triangle_area = s^2 - 4 * triangle_area
  /-- The distance between the apexes of two opposite isosceles triangles is 12 -/
  apex_distance : s + 2 * (triangle_area / s) = 12

/-- Theorem: The side length of the square is 24 -/
theorem square_side_length (sq : SquareWithTriangles) : sq.s = 24 :=
sorry

end NUMINAMATH_CALUDE_square_side_length_l1392_139284


namespace NUMINAMATH_CALUDE_remainder_17_pow_63_mod_7_l1392_139225

theorem remainder_17_pow_63_mod_7 : 17^63 % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_17_pow_63_mod_7_l1392_139225


namespace NUMINAMATH_CALUDE_derivative_log2_l1392_139263

-- Define the base-2 logarithm function
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem derivative_log2 (x : ℝ) (h : x > 0) :
  deriv log2 x = 1 / (x * Real.log 2) :=
by sorry

end NUMINAMATH_CALUDE_derivative_log2_l1392_139263


namespace NUMINAMATH_CALUDE_max_m_value_min_objective_value_l1392_139230

-- Define the inequality function
def inequality (x m : ℝ) : Prop := |x - 3| + |x - m| ≥ 2 * m

-- Theorem for the maximum value of m
theorem max_m_value : 
  (∀ x : ℝ, inequality x 1) ∧ 
  (∀ m : ℝ, m > 1 → ∃ x : ℝ, ¬(inequality x m)) :=
sorry

-- Define the constraint function
def constraint (a b c : ℝ) : Prop := a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 1

-- Define the objective function
def objective (a b c : ℝ) : ℝ := 4 * a^2 + 9 * b^2 + c^2

-- Theorem for the minimum value of the objective function
theorem min_objective_value :
  (∀ a b c : ℝ, constraint a b c → objective a b c ≥ 36/49) ∧
  (∃ a b c : ℝ, constraint a b c ∧ objective a b c = 36/49 ∧ 
    a = 9/49 ∧ b = 4/49 ∧ c = 36/49) :=
sorry

end NUMINAMATH_CALUDE_max_m_value_min_objective_value_l1392_139230


namespace NUMINAMATH_CALUDE_down_payment_calculation_l1392_139205

theorem down_payment_calculation (purchase_price : ℝ) (monthly_payment : ℝ) 
  (num_payments : ℕ) (interest_rate : ℝ) (down_payment : ℝ) : 
  purchase_price = 130 ∧ 
  monthly_payment = 10 ∧ 
  num_payments = 12 ∧ 
  interest_rate = 0.23076923076923077 →
  down_payment = purchase_price + interest_rate * purchase_price - num_payments * monthly_payment :=
by
  sorry

end NUMINAMATH_CALUDE_down_payment_calculation_l1392_139205


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1392_139262

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.log (1 - x)}
def N : Set ℝ := {x | ∃ y, y = x^2 - 2*x + 1}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x | x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1392_139262


namespace NUMINAMATH_CALUDE_jose_investment_is_4500_l1392_139214

/-- Represents the investment and profit scenario of Tom and Jose's shop --/
structure ShopInvestment where
  tom_investment : ℕ
  jose_join_delay : ℕ
  total_profit : ℕ
  jose_profit_share : ℕ

/-- Calculates Jose's investment given the shop investment scenario --/
def calculate_jose_investment (s : ShopInvestment) : ℕ :=
  sorry

/-- Theorem stating that Jose's investment is 4500 given the specific scenario --/
theorem jose_investment_is_4500 :
  let s : ShopInvestment := {
    tom_investment := 3000,
    jose_join_delay := 2,
    total_profit := 6300,
    jose_profit_share := 3500
  }
  calculate_jose_investment s = 4500 := by sorry

end NUMINAMATH_CALUDE_jose_investment_is_4500_l1392_139214


namespace NUMINAMATH_CALUDE_fraction_under_21_l1392_139258

theorem fraction_under_21 (total : ℕ) (under_21 : ℕ) (over_65 : ℚ) :
  total > 50 →
  total < 100 →
  over_65 = 5/10 →
  under_21 = 30 →
  (under_21 : ℚ) / total = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_under_21_l1392_139258


namespace NUMINAMATH_CALUDE_orthogonal_lines_sweep_l1392_139278

-- Define the circle S
def S (a : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ a^2}

-- Define the point O outside the circle
def O (x₀ y₀ : ℝ) : ℝ × ℝ := (x₀, y₀)

-- Define a point X inside the circle
def X (c : ℝ) (a : ℝ) : ℝ × ℝ := (c, 0)

-- Define the set of points swept by lines l
def swept_points (a c : ℝ) : Set (ℝ × ℝ) :=
  {p | (c^2 - a^2) * p.1^2 - a^2 * p.2^2 ≤ a^2 * (c^2 - a^2)}

-- State the theorem
theorem orthogonal_lines_sweep (a : ℝ) (x₀ y₀ : ℝ) (h₁ : a > 0) (h₂ : x₀^2 + y₀^2 ≠ a^2) :
  ∀ c, c^2 < a^2 →
    swept_points a c =
    {p | ∃ (X : ℝ × ℝ), X ∈ S a ∧ (p.1 - X.1) * (O x₀ y₀).1 + (p.2 - X.2) * (O x₀ y₀).2 = 0} :=
by sorry

end NUMINAMATH_CALUDE_orthogonal_lines_sweep_l1392_139278


namespace NUMINAMATH_CALUDE_proportion_equality_l1392_139231

theorem proportion_equality (m n : ℝ) (h1 : 6 * m = 7 * n) (h2 : n ≠ 0) :
  m / 7 = n / 6 := by
  sorry

end NUMINAMATH_CALUDE_proportion_equality_l1392_139231


namespace NUMINAMATH_CALUDE_zoo_giraffe_difference_l1392_139213

theorem zoo_giraffe_difference (total_giraffes : ℕ) (other_animals : ℕ) : 
  total_giraffes = 300 →
  total_giraffes = 3 * other_animals →
  total_giraffes - other_animals = 200 := by
sorry

end NUMINAMATH_CALUDE_zoo_giraffe_difference_l1392_139213


namespace NUMINAMATH_CALUDE_triangle_prime_count_l1392_139283

def is_prime (n : ℕ) : Prop := sorry

def count_primes (a b : ℕ) : ℕ := sorry

def triangle_sides_valid (n : ℕ) : Prop :=
  let side1 := Real.log 16 / Real.log 8
  let side2 := Real.log 128 / Real.log 8
  let side3 := Real.log n / Real.log 8
  side1 + side2 > side3 ∧ side1 + side3 > side2 ∧ side2 + side3 > side1

theorem triangle_prime_count :
  ∀ n : ℕ, 
    n > 0 → 
    is_prime n → 
    triangle_sides_valid n →
    ∃ (count : ℕ), count = count_primes 9 4095 := by
  sorry

end NUMINAMATH_CALUDE_triangle_prime_count_l1392_139283


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1392_139287

/-- Represents a hyperbola with given properties -/
structure Hyperbola where
  asymptote_slope : ℝ
  real_axis_length : ℝ
  foci_on_x_axis : Bool

/-- The standard equation of a hyperbola -/
def standard_equation (h : Hyperbola) : Prop :=
  ∀ x y : ℝ, x^2 - y^2 / 9 = 1

/-- Theorem: Given a hyperbola with asymptote slope 3, real axis length 2, and foci on x-axis,
    its standard equation is x² - y²/9 = 1 -/
theorem hyperbola_equation (h : Hyperbola) 
    (h_asymptote : h.asymptote_slope = 3)
    (h_real_axis : h.real_axis_length = 2)
    (h_foci : h.foci_on_x_axis = true) :
    standard_equation h :=
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1392_139287


namespace NUMINAMATH_CALUDE_roots_property_l1392_139211

theorem roots_property (a b : ℝ) : 
  (a^2 - a - 2 = 0) → (b^2 - b - 2 = 0) → (a - 1) * (b - 1) = -2 := by
  sorry

end NUMINAMATH_CALUDE_roots_property_l1392_139211


namespace NUMINAMATH_CALUDE_sin_593_degrees_l1392_139268

theorem sin_593_degrees (h : Real.sin (37 * π / 180) = 3 / 5) :
  Real.sin (593 * π / 180) = -(3 / 5) := by
  sorry

end NUMINAMATH_CALUDE_sin_593_degrees_l1392_139268


namespace NUMINAMATH_CALUDE_president_secretary_selection_l1392_139271

/-- The number of ways to select one president and one secretary from five different people. -/
def select_president_and_secretary (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: The number of ways to select one president and one secretary from five different people is 20. -/
theorem president_secretary_selection :
  select_president_and_secretary 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_president_secretary_selection_l1392_139271


namespace NUMINAMATH_CALUDE_grade_multiplier_is_five_l1392_139218

def grades : List ℕ := [2, 2, 2, 3, 3, 3, 3, 4, 5]
def total_reward : ℚ := 15

theorem grade_multiplier_is_five :
  let average_grade := (grades.sum : ℚ) / grades.length
  let multiplier := total_reward / average_grade
  multiplier = 5 := by sorry

end NUMINAMATH_CALUDE_grade_multiplier_is_five_l1392_139218


namespace NUMINAMATH_CALUDE_gcd_of_100_and_250_l1392_139294

theorem gcd_of_100_and_250 : Nat.gcd 100 250 = 50 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_100_and_250_l1392_139294


namespace NUMINAMATH_CALUDE_cos_sixty_degrees_l1392_139248

theorem cos_sixty_degrees : Real.cos (π / 3) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sixty_degrees_l1392_139248


namespace NUMINAMATH_CALUDE_polynomial_not_divisible_l1392_139215

theorem polynomial_not_divisible (k : ℕ) : 
  (∀ x : ℂ, x^2 + x + 1 = 0 → x^(2*k) + 1 + (x+1)^(2*k) ≠ 0) ↔ ¬(3 ∣ k) :=
sorry

end NUMINAMATH_CALUDE_polynomial_not_divisible_l1392_139215


namespace NUMINAMATH_CALUDE_smallest_number_l1392_139290

/-- Converts a number from base b to decimal --/
def to_decimal (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b^i) 0

/-- The number 85 in base 9 --/
def num1 : Nat := to_decimal [5, 8] 9

/-- The number 1000 in base 4 --/
def num2 : Nat := to_decimal [0, 0, 0, 1] 4

/-- The number 111111 in base 2 --/
def num3 : Nat := to_decimal [1, 1, 1, 1, 1, 1] 2

theorem smallest_number : num3 < num2 ∧ num3 < num1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l1392_139290


namespace NUMINAMATH_CALUDE_problem_proof_l1392_139240

theorem problem_proof : -1^2023 + (Real.pi - 3.14)^0 + |-2| = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_proof_l1392_139240


namespace NUMINAMATH_CALUDE_youngest_boy_age_l1392_139228

/-- Given three boys whose ages are in proportion 2 : 6 : 8 and whose average age is 120 years,
    the age of the youngest boy is 45 years. -/
theorem youngest_boy_age (a b c : ℕ) : 
  a + b + c = 360 →  -- Sum of ages is 360 (3 * 120)
  3 * a = b →        -- b is 3 times a
  4 * a = c →        -- c is 4 times a
  a = 45 :=          -- The age of the youngest boy (a) is 45
by sorry

end NUMINAMATH_CALUDE_youngest_boy_age_l1392_139228


namespace NUMINAMATH_CALUDE_sample_size_is_120_l1392_139219

/-- Represents the sizes of three population groups -/
structure PopulationGroups where
  group1 : ℕ
  group2 : ℕ
  group3 : ℕ

/-- Calculates the total sample size in a stratified sampling -/
def calculateSampleSize (groups : PopulationGroups) (samplesFromGroup3 : ℕ) : ℕ :=
  samplesFromGroup3 * (groups.group1 + groups.group2 + groups.group3) / groups.group3

/-- Theorem stating that the sample size is 120 under given conditions -/
theorem sample_size_is_120 (groups : PopulationGroups) (h1 : groups.group1 = 2400) 
    (h2 : groups.group2 = 3600) (h3 : groups.group3 = 6000) (samplesFromGroup3 : ℕ) 
    (h4 : samplesFromGroup3 = 60) : 
  calculateSampleSize groups samplesFromGroup3 = 120 := by
  sorry

#eval calculateSampleSize ⟨2400, 3600, 6000⟩ 60

end NUMINAMATH_CALUDE_sample_size_is_120_l1392_139219


namespace NUMINAMATH_CALUDE_six_different_squares_cannot_form_rectangle_l1392_139296

/-- A square with a given side length -/
structure Square where
  sideLength : ℝ
  positive : sideLength > 0

/-- A collection of squares -/
def SquareCollection := List Square

/-- Predicate to check if all squares in a collection have different sizes -/
def allDifferentSizes (squares : SquareCollection) : Prop :=
  ∀ i j, i ≠ j → (squares.get i).sideLength ≠ (squares.get j).sideLength

/-- Predicate to check if squares can form a rectangle -/
def canFormRectangle (squares : SquareCollection) : Prop :=
  ∃ (width height : ℝ), width > 0 ∧ height > 0 ∧
    (squares.map (λ s => s.sideLength ^ 2)).sum = width * height

theorem six_different_squares_cannot_form_rectangle :
  ∀ (squares : SquareCollection),
    squares.length = 6 →
    allDifferentSizes squares →
    ¬ canFormRectangle squares :=
by
  sorry

end NUMINAMATH_CALUDE_six_different_squares_cannot_form_rectangle_l1392_139296


namespace NUMINAMATH_CALUDE_mariela_get_well_cards_l1392_139237

theorem mariela_get_well_cards (cards_from_home : ℝ) (cards_from_country : ℕ) 
  (h1 : cards_from_home = 287.0) 
  (h2 : cards_from_country = 116) : 
  ↑cards_from_country + cards_from_home = 403 := by
  sorry

end NUMINAMATH_CALUDE_mariela_get_well_cards_l1392_139237


namespace NUMINAMATH_CALUDE_area_ratio_in_square_l1392_139256

/-- Given a unit square ABCD with points X on BC and Y on CD such that
    triangles ABX, XCY, and YDA have equal areas, the ratio of the area of
    triangle AXY to the area of triangle XCY is √5. -/
theorem area_ratio_in_square (A B C D X Y : ℝ × ℝ) : 
  let square_side_length : ℝ := 1
  let on_side (P Q R : ℝ × ℝ) := ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • Q + t • R
  let area (P Q R : ℝ × ℝ) : ℝ := abs ((P.1 - R.1) * (Q.2 - R.2) - (Q.1 - R.1) * (P.2 - R.2)) / 2
  square_side_length = 1 →
  (A.1 = 0 ∧ A.2 = 0) →
  (B.1 = 1 ∧ B.2 = 0) →
  (C.1 = 1 ∧ C.2 = 1) →
  (D.1 = 0 ∧ D.2 = 1) →
  on_side X B C →
  on_side Y C D →
  area A B X = area X C Y →
  area X C Y = area Y D A →
  area A X Y / area X C Y = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_area_ratio_in_square_l1392_139256


namespace NUMINAMATH_CALUDE_abc_product_l1392_139233

theorem abc_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a * (b + c) = 171) (h2 : b * (c + a) = 180) (h3 : c * (a + b) = 189) :
  a * b * c = 270 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_l1392_139233


namespace NUMINAMATH_CALUDE_min_value_xy_l1392_139202

theorem min_value_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 5/x + 3/y = 2) : 
  ∀ a b : ℝ, a > 0 → b > 0 → 5/a + 3/b = 2 → x * y ≤ a * b :=
by sorry

end NUMINAMATH_CALUDE_min_value_xy_l1392_139202


namespace NUMINAMATH_CALUDE_arithmetic_to_geometric_sequence_ratio_l1392_139229

/-- 
Given three distinct real numbers a, b, c forming an arithmetic sequence with a < b < c,
if swapping two of these numbers results in a geometric sequence,
then (a² + c²) / b² = 20.
-/
theorem arithmetic_to_geometric_sequence_ratio (a b c : ℝ) : 
  a < b → b < c → 
  (∃ d : ℝ, c - b = b - a ∧ d = b - a) →
  (∃ (x y z : ℝ) (σ : Equiv.Perm (Fin 3)), 
    ({x, y, z} : Finset ℝ) = {a, b, c} ∧ 
    (y * y = x * z)) →
  (a * a + c * c) / (b * b) = 20 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_to_geometric_sequence_ratio_l1392_139229


namespace NUMINAMATH_CALUDE_quadratic_intersection_l1392_139212

/-- A quadratic function of the form y = x^2 + px + q where p + q = 2002 -/
def QuadraticFunction (p q : ℝ) : ℝ → ℝ := fun x ↦ x^2 + p*x + q

/-- The theorem stating that all quadratic functions satisfying the condition
    p + q = 2002 intersect at the point (1, 2003) -/
theorem quadratic_intersection (p q : ℝ) (h : p + q = 2002) :
  QuadraticFunction p q 1 = 2003 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_intersection_l1392_139212


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l1392_139288

/-- Given a principal sum and a time period of 8 years, if the simple interest
    is one-fifth of the principal sum, then the rate of interest per annum is 2.5%. -/
theorem interest_rate_calculation (P : ℝ) (P_pos : P > 0) : 
  (P * 2.5 * 8) / 100 = P / 5 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l1392_139288


namespace NUMINAMATH_CALUDE_siblings_average_age_l1392_139259

/-- Given 4 siblings where the youngest is 25.75 years old and the others are 3, 6, and 7 years older,
    the average age of all siblings is 29.75 years. -/
theorem siblings_average_age :
  let youngest_age : ℝ := 25.75
  let sibling_age_differences : List ℝ := [3, 6, 7]
  let all_ages : List ℝ := youngest_age :: (sibling_age_differences.map (λ x => youngest_age + x))
  (all_ages.sum / all_ages.length : ℝ) = 29.75 := by
  sorry

end NUMINAMATH_CALUDE_siblings_average_age_l1392_139259


namespace NUMINAMATH_CALUDE_baxter_spent_105_l1392_139281

/-- The cost of peanuts per pound -/
def cost_per_pound : ℕ := 3

/-- The minimum purchase requirement in pounds -/
def minimum_purchase : ℕ := 15

/-- The amount Baxter purchased over the minimum, in pounds -/
def over_minimum : ℕ := 20

/-- Calculates the total amount Baxter spent on peanuts -/
def baxter_spent : ℕ := cost_per_pound * (minimum_purchase + over_minimum)

/-- Proves that Baxter spent $105 on peanuts -/
theorem baxter_spent_105 : baxter_spent = 105 := by
  sorry

end NUMINAMATH_CALUDE_baxter_spent_105_l1392_139281


namespace NUMINAMATH_CALUDE_contest_winner_l1392_139253

theorem contest_winner (n : ℕ) : 
  (∀ k : ℕ, k > 0 → n % 100 = 0 ∧ n % 40 = 0) → n ≥ 200 :=
sorry

end NUMINAMATH_CALUDE_contest_winner_l1392_139253


namespace NUMINAMATH_CALUDE_half_radius_circle_y_l1392_139285

theorem half_radius_circle_y (x y : Real) :
  (∃ (r : Real), x = π * r^2 ∧ y = π * r^2) →  -- circles x and y have the same area
  (∃ (r : Real), 18 * π = 2 * π * r) →         -- circle x has circumference 18π
  (∃ (r : Real), y = π * r^2 ∧ r / 2 = 4.5) := by
sorry

end NUMINAMATH_CALUDE_half_radius_circle_y_l1392_139285


namespace NUMINAMATH_CALUDE_well_depth_l1392_139275

/-- Represents a circular well -/
structure CircularWell where
  diameter : ℝ
  volume : ℝ
  depth : ℝ

/-- Theorem stating the depth of a specific circular well -/
theorem well_depth (w : CircularWell) 
  (h1 : w.diameter = 4)
  (h2 : w.volume = 175.92918860102841) :
  w.depth = 14 := by
  sorry

end NUMINAMATH_CALUDE_well_depth_l1392_139275


namespace NUMINAMATH_CALUDE_consecutive_product_square_append_l1392_139241

theorem consecutive_product_square_append (n : ℕ) : ∃ m : ℕ, 100 * (n * (n + 1)) + 25 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_product_square_append_l1392_139241


namespace NUMINAMATH_CALUDE_baking_powder_difference_l1392_139295

theorem baking_powder_difference (yesterday_amount today_amount : ℝ) 
  (h1 : yesterday_amount = 0.4)
  (h2 : today_amount = 0.3) : 
  yesterday_amount - today_amount = 0.1 := by
sorry

end NUMINAMATH_CALUDE_baking_powder_difference_l1392_139295


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_implies_c_less_than_one_l1392_139217

theorem quadratic_distinct_roots_implies_c_less_than_one :
  ∀ c : ℝ, (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + c = 0 ∧ y^2 - 2*y + c = 0) → c < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_implies_c_less_than_one_l1392_139217


namespace NUMINAMATH_CALUDE_eliminate_alpha_l1392_139277

theorem eliminate_alpha (x y : ℝ) (α : ℝ) 
  (hx : x = Real.tan α ^ 2) 
  (hy : y = Real.sin α ^ 2) : 
  x - y = x * y := by
  sorry

end NUMINAMATH_CALUDE_eliminate_alpha_l1392_139277


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l1392_139257

-- Problem 1
theorem problem_1 : Real.sqrt 32 + 3 * Real.sqrt (1/2) - Real.sqrt 2 = (9 * Real.sqrt 2) / 2 := by sorry

-- Problem 2
theorem problem_2 : (Real.sqrt 50 * Real.sqrt 32) / Real.sqrt 8 - 4 * Real.sqrt 2 = 6 * Real.sqrt 2 := by sorry

-- Problem 3
theorem problem_3 : (Real.sqrt 5 - 3)^2 + (Real.sqrt 11 + 3) * (Real.sqrt 11 - 3) = 16 - 6 * Real.sqrt 5 := by sorry

-- Problem 4
theorem problem_4 : (2 * Real.sqrt 6 + Real.sqrt 12) * Real.sqrt 3 - 12 * Real.sqrt (1/2) = 6 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l1392_139257


namespace NUMINAMATH_CALUDE_rope_knot_reduction_l1392_139293

theorem rope_knot_reduction 
  (total_length : ℝ) 
  (num_pieces : ℕ) 
  (tied_pieces : ℕ) 
  (final_length : ℝ) 
  (h1 : total_length = 72) 
  (h2 : num_pieces = 12) 
  (h3 : tied_pieces = 3) 
  (h4 : final_length = 15) : 
  (total_length / num_pieces * tied_pieces - final_length) / (tied_pieces - 1) = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_rope_knot_reduction_l1392_139293


namespace NUMINAMATH_CALUDE_solve_for_k_l1392_139267

-- Define the system of equations
def system (x y k : ℝ) : Prop :=
  (2 * x + y = 4 * k) ∧ (x - y = k)

-- Define the additional equation
def additional_eq (x y : ℝ) : Prop :=
  x + 2 * y = 12

-- Theorem statement
theorem solve_for_k :
  ∀ x y k : ℝ, system x y k → additional_eq x y → k = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_for_k_l1392_139267


namespace NUMINAMATH_CALUDE_perpendicular_slope_l1392_139201

/-- The slope of a line perpendicular to the line passing through (3, -4) and (-2, 5) is 5/9 -/
theorem perpendicular_slope : 
  let x₁ : ℚ := 3
  let y₁ : ℚ := -4
  let x₂ : ℚ := -2
  let y₂ : ℚ := 5
  let m : ℚ := (y₂ - y₁) / (x₂ - x₁)
  (- (1 / m)) = 5 / 9 := by sorry

end NUMINAMATH_CALUDE_perpendicular_slope_l1392_139201


namespace NUMINAMATH_CALUDE_prob_all_red_is_one_third_l1392_139244

/-- Represents the number of red chips in the hat -/
def num_red : ℕ := 4

/-- Represents the number of green chips in the hat -/
def num_green : ℕ := 2

/-- Represents the total number of chips in the hat -/
def total_chips : ℕ := num_red + num_green

/-- Represents the probability of drawing all red chips before both green chips -/
def prob_all_red : ℚ := 1 / 3

/-- Theorem stating that the probability of drawing all red chips before both green chips is 1/3 -/
theorem prob_all_red_is_one_third :
  prob_all_red = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_prob_all_red_is_one_third_l1392_139244
