import Mathlib

namespace NUMINAMATH_CALUDE_overall_percentage_increase_l1901_190181

def initial_price_A : ℝ := 300
def initial_price_B : ℝ := 150
def initial_price_C : ℝ := 50
def initial_price_D : ℝ := 100

def new_price_A : ℝ := 390
def new_price_B : ℝ := 180
def new_price_C : ℝ := 70
def new_price_D : ℝ := 110

def total_initial_price : ℝ := initial_price_A + initial_price_B + initial_price_C + initial_price_D
def total_new_price : ℝ := new_price_A + new_price_B + new_price_C + new_price_D

theorem overall_percentage_increase :
  (total_new_price - total_initial_price) / total_initial_price * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_overall_percentage_increase_l1901_190181


namespace NUMINAMATH_CALUDE_marys_overtime_rate_increase_l1901_190187

/-- Represents Mary's work schedule and pay structure --/
structure WorkSchedule where
  maxHours : ℕ
  regularHours : ℕ
  regularRate : ℚ
  maxWeeklyEarnings : ℚ

/-- Calculates the percentage increase in overtime rate compared to regular rate --/
def overtimeRateIncrease (w : WorkSchedule) : ℚ :=
  let regularEarnings := w.regularRate * w.regularHours
  let overtimeEarnings := w.maxWeeklyEarnings - regularEarnings
  let overtimeHours := w.maxHours - w.regularHours
  let overtimeRate := overtimeEarnings / overtimeHours
  ((overtimeRate - w.regularRate) / w.regularRate) * 100

/-- Mary's work schedule --/
def marysSchedule : WorkSchedule :=
  { maxHours := 50
  , regularHours := 20
  , regularRate := 8
  , maxWeeklyEarnings := 460 }

/-- Theorem stating that Mary's overtime rate increase is 25% --/
theorem marys_overtime_rate_increase :
  overtimeRateIncrease marysSchedule = 25 := by
  sorry


end NUMINAMATH_CALUDE_marys_overtime_rate_increase_l1901_190187


namespace NUMINAMATH_CALUDE_equations_equivalence_l1901_190120

-- Define the equations
def equation1 (x : ℝ) : Prop := (-x - 2) / (x - 3) = (x + 1) / (x - 3)
def equation2 (x : ℝ) : Prop := -x - 2 = x + 1
def equation3 (x : ℝ) : Prop := (-x - 2) * (x - 3) = (x + 1) * (x - 3)

-- Theorem statement
theorem equations_equivalence :
  (∀ x : ℝ, x ≠ 3 → (equation1 x ↔ equation2 x)) ∧
  (¬ ∀ x : ℝ, equation2 x ↔ equation3 x) ∧
  (¬ ∀ x : ℝ, equation1 x ↔ equation3 x) :=
sorry

end NUMINAMATH_CALUDE_equations_equivalence_l1901_190120


namespace NUMINAMATH_CALUDE_complex_number_magnitude_l1901_190146

theorem complex_number_magnitude (z : ℂ) (h : z * Complex.I = 1) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_magnitude_l1901_190146


namespace NUMINAMATH_CALUDE_stating_escalator_step_count_l1901_190194

/-- Represents the number of steps counted on an escalator under different conditions -/
structure EscalatorSteps where
  down : ℕ  -- steps counted running down
  up : ℕ    -- steps counted running up
  stationary : ℕ  -- steps counted on a stationary escalator

/-- 
Given the number of steps counted running down and up a moving escalator,
calculates the number of steps on a stationary escalator
-/
def calculateStationarySteps (e : EscalatorSteps) : Prop :=
  e.down = 30 ∧ e.up = 150 → e.stationary = 50

/-- 
Theorem stating that if a person counts 30 steps running down a moving escalator
and 150 steps running up the same escalator at the same speed relative to the escalator,
then they would count 50 steps on a stationary escalator
-/
theorem escalator_step_count : ∃ e : EscalatorSteps, calculateStationarySteps e :=
  sorry

end NUMINAMATH_CALUDE_stating_escalator_step_count_l1901_190194


namespace NUMINAMATH_CALUDE_distance_at_two_point_five_l1901_190169

/-- The distance traveled by a ball rolling down an inclined plane -/
def distance (t : ℝ) : ℝ := 10 * t^2

/-- Theorem: The distance traveled at t = 2.5 seconds is 62.5 feet -/
theorem distance_at_two_point_five :
  distance 2.5 = 62.5 := by sorry

end NUMINAMATH_CALUDE_distance_at_two_point_five_l1901_190169


namespace NUMINAMATH_CALUDE_louis_fabric_purchase_l1901_190136

/-- The cost of velvet fabric per yard -/
def fabric_cost_per_yard : ℚ := 24

/-- The cost of the pattern -/
def pattern_cost : ℚ := 15

/-- The total cost of silver thread -/
def thread_cost : ℚ := 6

/-- The total amount spent -/
def total_spent : ℚ := 141

/-- The number of yards of fabric bought -/
def yards_bought : ℚ := (total_spent - pattern_cost - thread_cost) / fabric_cost_per_yard

theorem louis_fabric_purchase : yards_bought = 5 := by
  sorry

end NUMINAMATH_CALUDE_louis_fabric_purchase_l1901_190136


namespace NUMINAMATH_CALUDE_alice_game_theorem_l1901_190103

/-- The game state, representing the positions of the red and blue beads -/
structure GameState where
  red : ℚ
  blue : ℚ

/-- The move function that updates the game state -/
def move (r : ℚ) (state : GameState) (k : ℤ) (moveRed : Bool) : GameState :=
  if moveRed then
    { red := state.blue + r^k * (state.red - state.blue), blue := state.blue }
  else
    { red := state.red, blue := state.red + r^k * (state.blue - state.red) }

/-- Predicate to check if a rational number is of the form (b+1)/b for 1 ≤ b ≤ 1010 -/
def isValidR (r : ℚ) : Prop :=
  ∃ b : ℕ, 1 ≤ b ∧ b ≤ 1010 ∧ r = (b + 1) / b

/-- Main theorem statement -/
theorem alice_game_theorem (r : ℚ) (hr : r > 1) :
  (∃ (moves : List (ℤ × Bool)), moves.length ≤ 2021 ∧
    (moves.foldl (λ state (k, moveRed) => move r state k moveRed)
      { red := 0, blue := 1 }).red = 1) ↔
  isValidR r :=
sorry

end NUMINAMATH_CALUDE_alice_game_theorem_l1901_190103


namespace NUMINAMATH_CALUDE_room_height_is_12_l1901_190125

def room_length : ℝ := 25
def room_width : ℝ := 15
def door_area : ℝ := 6 * 3
def window_area : ℝ := 4 * 3
def num_windows : ℕ := 3
def whitewash_cost_per_sqft : ℝ := 2
def total_cost : ℝ := 1812

theorem room_height_is_12 (h : ℝ) :
  (2 * (room_length + room_width) * h - (door_area + num_windows * window_area)) * whitewash_cost_per_sqft = total_cost →
  h = 12 := by
  sorry

end NUMINAMATH_CALUDE_room_height_is_12_l1901_190125


namespace NUMINAMATH_CALUDE_book_cost_proof_l1901_190155

/-- Given that Mark started with $85, bought 10 books, and was left with $35, prove that each book cost $5. -/
theorem book_cost_proof (initial_amount : ℕ) (books_bought : ℕ) (remaining_amount : ℕ) :
  initial_amount = 85 ∧ books_bought = 10 ∧ remaining_amount = 35 →
  (initial_amount - remaining_amount) / books_bought = 5 :=
by sorry

end NUMINAMATH_CALUDE_book_cost_proof_l1901_190155


namespace NUMINAMATH_CALUDE_pool_concrete_weight_l1901_190126

/-- Represents the dimensions and properties of a swimming pool --/
structure Pool where
  tileLength : ℝ
  wallHeight : ℝ
  wallThickness : ℝ
  perimeterUnits : ℕ
  outerCorners : ℕ
  innerCorners : ℕ
  concreteWeight : ℝ

/-- Calculates the weight of concrete used for the walls of a pool --/
def concreteWeightForWalls (p : Pool) : ℝ :=
  let adjustedPerimeter := p.perimeterUnits * p.tileLength + p.outerCorners * p.wallThickness - p.innerCorners * p.wallThickness
  let wallVolume := adjustedPerimeter * p.wallHeight * p.wallThickness
  wallVolume * p.concreteWeight

/-- The theorem to be proved --/
theorem pool_concrete_weight :
  let p : Pool := {
    tileLength := 2,
    wallHeight := 3,
    wallThickness := 0.5,
    perimeterUnits := 32,
    outerCorners := 10,
    innerCorners := 6,
    concreteWeight := 2000
  }
  concreteWeightForWalls p = 198000 := by sorry

end NUMINAMATH_CALUDE_pool_concrete_weight_l1901_190126


namespace NUMINAMATH_CALUDE_deck_total_cost_l1901_190111

def deck_length : ℝ := 30
def deck_width : ℝ := 40
def base_cost_per_sqft : ℝ := 3
def sealant_cost_per_sqft : ℝ := 1

theorem deck_total_cost :
  deck_length * deck_width * (base_cost_per_sqft + sealant_cost_per_sqft) = 4800 := by
  sorry

end NUMINAMATH_CALUDE_deck_total_cost_l1901_190111


namespace NUMINAMATH_CALUDE_petya_max_candies_l1901_190176

/-- Represents the state of a pile of candies -/
structure Pile :=
  (count : Nat)

/-- Represents the game state -/
structure GameState :=
  (piles : List Pile)

/-- Defines a player's move -/
inductive Move
  | take : Nat → Move

/-- Defines the result of a move -/
inductive MoveResult
  | eat : MoveResult
  | throw : MoveResult

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Move) : Option (GameState × MoveResult) :=
  sorry

/-- Checks if the game is over -/
def isGameOver (state : GameState) : Bool :=
  sorry

/-- Represents a strategy for playing the game -/
def Strategy := GameState → Move

/-- Simulates the game with given strategies -/
def playGame (initialState : GameState) (petyaStrategy : Strategy) (vasyaStrategy : Strategy) : Nat :=
  sorry

/-- The initial game state -/
def initialGameState : GameState :=
  { piles := List.range 55 |>.map (fun i => { count := i + 1 }) }

theorem petya_max_candies :
  ∀ (petyaStrategy : Strategy),
  ∃ (vasyaStrategy : Strategy),
  playGame initialGameState petyaStrategy vasyaStrategy ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_petya_max_candies_l1901_190176


namespace NUMINAMATH_CALUDE_circle_A_tangent_to_x_axis_l1901_190117

def circle_A_center : ℝ × ℝ := (-4, -3)
def circle_A_radius : ℝ := 3

theorem circle_A_tangent_to_x_axis :
  let (x, y) := circle_A_center
  abs y = circle_A_radius := by sorry

end NUMINAMATH_CALUDE_circle_A_tangent_to_x_axis_l1901_190117


namespace NUMINAMATH_CALUDE_integer_root_count_theorem_l1901_190196

/-- A polynomial of degree 5 with integer coefficients -/
def IntPolynomial5 (x b c d e f : ℤ) : ℤ := x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f

/-- The set of possible numbers of integer roots for IntPolynomial5 -/
def PossibleRootCounts : Set ℕ := {0, 1, 2, 5}

/-- Theorem stating that the number of integer roots of IntPolynomial5 is in PossibleRootCounts -/
theorem integer_root_count_theorem (b c d e f : ℤ) :
  ∃ (n : ℕ), n ∈ PossibleRootCounts ∧
  (∃ (roots : List ℤ), (∀ x ∈ roots, IntPolynomial5 x b c d e f = 0) ∧
                       roots.length = n) :=
sorry

end NUMINAMATH_CALUDE_integer_root_count_theorem_l1901_190196


namespace NUMINAMATH_CALUDE_factor_implies_b_value_l1901_190167

theorem factor_implies_b_value (a b : ℝ) :
  (∃ c : ℝ, ∀ x : ℝ, a * x^3 + b * x^2 + 1 = (x^2 - x - 1) * (x + c)) →
  b = -2 :=
by sorry

end NUMINAMATH_CALUDE_factor_implies_b_value_l1901_190167


namespace NUMINAMATH_CALUDE_initial_trees_count_l1901_190139

/-- The number of oak trees initially in the park -/
def initial_trees : ℕ := sorry

/-- The number of oak trees cut down -/
def cut_trees : ℕ := 2

/-- The number of oak trees remaining after cutting -/
def remaining_trees : ℕ := 7

/-- Theorem stating that the initial number of trees is 9 -/
theorem initial_trees_count : initial_trees = 9 := by sorry

end NUMINAMATH_CALUDE_initial_trees_count_l1901_190139


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_implies_m_eq_six_l1901_190172

/-- Given a hyperbola with equation x²/m - y²/6 = 1, where m is a real number,
    if one of its asymptotes is y = x, then m = 6. -/
theorem hyperbola_asymptote_implies_m_eq_six (m : ℝ) :
  (∃ (x y : ℝ), x^2 / m - y^2 / 6 = 1) →
  (∃ (x : ℝ), x = x) →
  m = 6 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_implies_m_eq_six_l1901_190172


namespace NUMINAMATH_CALUDE_rica_spent_fraction_l1901_190158

theorem rica_spent_fraction (total_prize : ℝ) (rica_fraction : ℝ) (rica_left : ℝ) : 
  total_prize = 1000 →
  rica_fraction = 3/8 →
  rica_left = 300 →
  (total_prize * rica_fraction - rica_left) / (total_prize * rica_fraction) = 1/5 :=
by sorry

end NUMINAMATH_CALUDE_rica_spent_fraction_l1901_190158


namespace NUMINAMATH_CALUDE_bank_interest_rate_is_five_percent_l1901_190148

/-- Proves that the bank interest rate is 5% given the investment conditions -/
theorem bank_interest_rate_is_five_percent 
  (total_investment : ℝ)
  (bank_investment : ℝ)
  (bond_investment : ℝ)
  (total_annual_income : ℝ)
  (bond_return_rate : ℝ)
  (h1 : total_investment = 10000)
  (h2 : bank_investment = 6000)
  (h3 : bond_investment = 4000)
  (h4 : total_annual_income = 660)
  (h5 : bond_return_rate = 0.09)
  (h6 : total_investment = bank_investment + bond_investment)
  (h7 : total_annual_income = bank_investment * bank_interest_rate + bond_investment * bond_return_rate) :
  bank_interest_rate = 0.05 := by
  sorry

#check bank_interest_rate_is_five_percent

end NUMINAMATH_CALUDE_bank_interest_rate_is_five_percent_l1901_190148


namespace NUMINAMATH_CALUDE_calculation_proof_l1901_190162

theorem calculation_proof : 15 * 30 + 45 * 15 - 15 * 10 = 975 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1901_190162


namespace NUMINAMATH_CALUDE_target_breaking_sequences_l1901_190149

/-- The number of unique permutations of a string with repeated characters -/
def multinomial_permutations (char_counts : List Nat) : Nat :=
  Nat.factorial (char_counts.sum) / (char_counts.map Nat.factorial).prod

/-- The target arrangement represented as character counts -/
def target_arrangement : List Nat := [4, 3, 3]

theorem target_breaking_sequences :
  multinomial_permutations target_arrangement = 4200 := by
  sorry

end NUMINAMATH_CALUDE_target_breaking_sequences_l1901_190149


namespace NUMINAMATH_CALUDE_flour_bag_cost_l1901_190174

/-- Represents the problem of calculating the cost of flour bags for Tom's dough ball project --/
theorem flour_bag_cost (flour_needed : ℕ) (flour_bag_size : ℕ) (salt_needed : ℕ) (salt_cost : ℚ)
  (promotion_cost : ℕ) (ticket_price : ℕ) (tickets_sold : ℕ) (profit : ℕ) :
  flour_needed = 500 →
  flour_bag_size = 50 →
  salt_needed = 10 →
  salt_cost = 1/5 →
  promotion_cost = 1000 →
  ticket_price = 20 →
  tickets_sold = 500 →
  profit = 8798 →
  (ticket_price * tickets_sold - promotion_cost - (salt_needed : ℚ) * salt_cost - profit) /
    (flour_needed / flour_bag_size) = 120 := by
  sorry

#check flour_bag_cost

end NUMINAMATH_CALUDE_flour_bag_cost_l1901_190174


namespace NUMINAMATH_CALUDE_grasshopper_cannot_return_after_25_jumps_l1901_190132

/-- The sum of the first n positive integers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The position of the grasshopper after n jumps -/
def grasshopper_position (n : ℕ) : ℕ := sum_first_n n

theorem grasshopper_cannot_return_after_25_jumps :
  ∃ k : ℕ, grasshopper_position 25 = 2 * k + 1 :=
sorry

end NUMINAMATH_CALUDE_grasshopper_cannot_return_after_25_jumps_l1901_190132


namespace NUMINAMATH_CALUDE_least_sum_of_exponents_l1901_190189

def target_number : ℕ := 3124

def is_sum_of_distinct_powers_of_two (n : ℕ) (exponents : List ℕ) : Prop :=
  n = (exponents.map (fun e => 2^e)).sum ∧ exponents.Nodup

theorem least_sum_of_exponents :
  ∃ (exponents : List ℕ),
    is_sum_of_distinct_powers_of_two target_number exponents ∧
    ∀ (other_exponents : List ℕ),
      is_sum_of_distinct_powers_of_two target_number other_exponents →
      exponents.sum ≤ other_exponents.sum ∧
      exponents.sum = 32 :=
sorry

end NUMINAMATH_CALUDE_least_sum_of_exponents_l1901_190189


namespace NUMINAMATH_CALUDE_ken_share_l1901_190143

def total_amount : ℕ := 5250

theorem ken_share (ken : ℕ) (tony : ℕ) 
  (h1 : ken + tony = total_amount) 
  (h2 : tony = 2 * ken) : 
  ken = 1750 := by
  sorry

end NUMINAMATH_CALUDE_ken_share_l1901_190143


namespace NUMINAMATH_CALUDE_triangle_problem_l1901_190119

theorem triangle_problem (A B C : Real) (a b c : Real) :
  -- Given conditions
  (b * Real.cos B = (a * Real.cos C + c * Real.cos A) / 2) →
  (a + c = Real.sqrt 10) →
  (b = 2) →
  -- Conclusions
  (B = π / 3) ∧
  (1/2 * a * c * Real.sin B = Real.sqrt 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l1901_190119


namespace NUMINAMATH_CALUDE_fraction_simplification_l1901_190106

theorem fraction_simplification (m : ℝ) (h : m^2 ≠ 1) :
  (m^2 - m) / (m^2 - 1) = m / (m + 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1901_190106


namespace NUMINAMATH_CALUDE_valid_diagonals_150_sided_polygon_l1901_190118

/-- The number of sides in the polygon -/
def n : ℕ := 150

/-- The total number of diagonals in an n-sided polygon -/
def total_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The number of diagonals to be excluded (those connecting vertices whose indices differ by a multiple of 4) -/
def excluded_diagonals (n : ℕ) : ℕ := n * (n / 4)

/-- The number of valid diagonals in the polygon -/
def valid_diagonals (n : ℕ) : ℕ := total_diagonals n - excluded_diagonals n

theorem valid_diagonals_150_sided_polygon :
  valid_diagonals n = 5400 := by
  sorry


end NUMINAMATH_CALUDE_valid_diagonals_150_sided_polygon_l1901_190118


namespace NUMINAMATH_CALUDE_tilly_star_count_l1901_190150

theorem tilly_star_count (stars_east : ℕ) (stars_west : ℕ) : 
  stars_east = 120 →
  stars_west = 6 * stars_east →
  stars_east + stars_west = 840 := by
sorry

end NUMINAMATH_CALUDE_tilly_star_count_l1901_190150


namespace NUMINAMATH_CALUDE_sports_equipment_purchase_l1901_190114

/-- Represents the purchase of sports equipment --/
structure Equipment where
  price_a : ℕ  -- price of type A equipment
  price_b : ℕ  -- price of type B equipment
  quantity_a : ℕ  -- quantity of type A equipment purchased
  quantity_b : ℕ  -- quantity of type B equipment purchased

/-- The main theorem about the sports equipment purchase --/
theorem sports_equipment_purchase 
  (e : Equipment) 
  (h1 : e.price_b = e.price_a + 10)  -- price difference condition
  (h2 : e.quantity_a * e.price_a = 300)  -- total cost of A
  (h3 : e.quantity_b * e.price_b = 360)  -- total cost of B
  (h4 : e.quantity_a = e.quantity_b)  -- equal quantities purchased
  : 
  (e.price_a = 50 ∧ e.price_b = 60) ∧  -- correct prices
  (∀ m n : ℕ, 
    50 * m + 60 * n = 1000 ↔ 
    ((m = 14 ∧ n = 5) ∨ (m = 8 ∧ n = 10) ∨ (m = 2 ∧ n = 15))) -- possible scenarios
  := by sorry


end NUMINAMATH_CALUDE_sports_equipment_purchase_l1901_190114


namespace NUMINAMATH_CALUDE_vector_identity_l1901_190164

variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- For any four points A, B, C, and D in a real inner product space,
    the vector DA + CD - CB is equal to BA. -/
theorem vector_identity (A B C D : V) :
  (D - A) + (C - D) - (C - B) = B - A :=
sorry

end NUMINAMATH_CALUDE_vector_identity_l1901_190164


namespace NUMINAMATH_CALUDE_quadratic_always_positive_implies_a_range_l1901_190163

theorem quadratic_always_positive_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1) * x + 1 > 0) → -1 < a ∧ a < 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_implies_a_range_l1901_190163


namespace NUMINAMATH_CALUDE_distinct_arrangements_of_six_l1901_190115

theorem distinct_arrangements_of_six (n : ℕ) (h : n = 6) : 
  Nat.factorial n = 720 := by
  sorry

end NUMINAMATH_CALUDE_distinct_arrangements_of_six_l1901_190115


namespace NUMINAMATH_CALUDE_camp_cedar_counselors_l1901_190141

/-- Calculates the number of counselors needed for a camp given the number of boys and the ratio of girls to boys. -/
def counselors_needed (num_boys : ℕ) (girls_to_boys_ratio : ℕ) (children_per_counselor : ℕ) : ℕ :=
  let num_girls := num_boys * girls_to_boys_ratio
  let total_children := num_boys + num_girls
  total_children / children_per_counselor

/-- Theorem stating that Camp Cedar needs 20 counselors. -/
theorem camp_cedar_counselors : 
  counselors_needed 40 3 8 = 20 := by
  sorry

end NUMINAMATH_CALUDE_camp_cedar_counselors_l1901_190141


namespace NUMINAMATH_CALUDE_sum_of_two_sequences_l1901_190157

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => a₁ + i * d)

def sum_list (l : List ℕ) : ℕ :=
  l.foldl (· + ·) 0

theorem sum_of_two_sequences : 
  let seq1 := arithmetic_sequence 2 12 4
  let seq2 := arithmetic_sequence 18 12 4
  sum_list (seq1 ++ seq2) = 224 := by
sorry

end NUMINAMATH_CALUDE_sum_of_two_sequences_l1901_190157


namespace NUMINAMATH_CALUDE_same_color_probability_l1901_190121

/-- The number of red candies in the box -/
def num_red : ℕ := 12

/-- The number of green candies in the box -/
def num_green : ℕ := 8

/-- The number of candies Alice and Bob each pick -/
def num_pick : ℕ := 3

/-- The probability that Alice and Bob pick the same number of candies of each color -/
def same_color_prob : ℚ := 231 / 1060

theorem same_color_probability :
  let total := num_red + num_green
  same_color_prob = (Nat.choose num_red num_pick * Nat.choose (num_red - num_pick) num_pick) / 
    (Nat.choose total num_pick * Nat.choose (total - num_pick) num_pick) +
    (Nat.choose num_red 2 * Nat.choose num_green 1 * Nat.choose (num_red - 2) 2 * 
    Nat.choose (num_green - 1) 1) / (Nat.choose total num_pick * Nat.choose (total - num_pick) num_pick) := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l1901_190121


namespace NUMINAMATH_CALUDE_point_on_line_l1901_190109

/-- Given a line in 3D space defined by the vector equation (x,y,z) = (5,0,3) + t(0,3,0),
    this theorem proves that the point on the line when t = 1/2 has coordinates (5,3/2,3). -/
theorem point_on_line (x y z t : ℝ) : 
  (x, y, z) = (5, 0, 3) + t • (0, 3, 0) → 
  t = 1/2 → 
  (x, y, z) = (5, 3/2, 3) := by
sorry

end NUMINAMATH_CALUDE_point_on_line_l1901_190109


namespace NUMINAMATH_CALUDE_ellipse_focal_length_l1901_190152

/-- An ellipse with equation y^2/2 + x^2 = 1 -/
def Ellipse := {p : ℝ × ℝ | p.2^2 / 2 + p.1^2 = 1}

/-- The focal length of an ellipse -/
def focalLength (E : Set (ℝ × ℝ)) : ℝ := sorry

/-- Theorem: The focal length of the ellipse y^2/2 + x^2 = 1 is 2 -/
theorem ellipse_focal_length : focalLength Ellipse = 2 := by sorry

end NUMINAMATH_CALUDE_ellipse_focal_length_l1901_190152


namespace NUMINAMATH_CALUDE_arccos_neg_one_eq_pi_l1901_190186

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = Real.pi := by sorry

end NUMINAMATH_CALUDE_arccos_neg_one_eq_pi_l1901_190186


namespace NUMINAMATH_CALUDE_graph_single_point_implies_d_value_l1901_190173

/-- The equation of the graph -/
def equation (x y d : ℝ) : Prop :=
  3 * x^2 + y^2 + 6 * x - 12 * y + d = 0

/-- The graph consists of a single point -/
def single_point (d : ℝ) : Prop :=
  ∃! p : ℝ × ℝ, equation p.1 p.2 d

/-- If the graph of 3x^2 + y^2 + 6x - 12y + d = 0 consists of a single point, then d = 39 -/
theorem graph_single_point_implies_d_value :
  ∀ d : ℝ, single_point d → d = 39 := by sorry

end NUMINAMATH_CALUDE_graph_single_point_implies_d_value_l1901_190173


namespace NUMINAMATH_CALUDE_min_distance_to_line_l1901_190182

/-- Given a point P(a,b) on the line y = √3x - √3, 
    the minimum value of (a+1)^2 + b^2 is 3 -/
theorem min_distance_to_line (a b : ℝ) : 
  b = Real.sqrt 3 * a - Real.sqrt 3 → 
  (∀ x y : ℝ, y = Real.sqrt 3 * x - Real.sqrt 3 → (a + 1)^2 + b^2 ≤ (x + 1)^2 + y^2) → 
  (a + 1)^2 + b^2 = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l1901_190182


namespace NUMINAMATH_CALUDE_expression_evaluation_l1901_190192

theorem expression_evaluation : 1 - (-2) - 3 - (-4) - 5 - (-6) = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1901_190192


namespace NUMINAMATH_CALUDE_unique_base7_digit_divisible_by_13_l1901_190168

/-- Converts a base-7 number of the form 3dd6_7 to base-10 --/
def base7ToBase10 (d : ℕ) : ℕ := 3 * 7^3 + d * 7^2 + d * 7 + 6

/-- Checks if a number is divisible by 13 --/
def isDivisibleBy13 (n : ℕ) : Prop := n % 13 = 0

/-- Represents a base-7 digit --/
def isBase7Digit (d : ℕ) : Prop := d ≤ 6

theorem unique_base7_digit_divisible_by_13 :
  ∃! d : ℕ, isBase7Digit d ∧ isDivisibleBy13 (base7ToBase10 d) ∧ d = 2 := by sorry

end NUMINAMATH_CALUDE_unique_base7_digit_divisible_by_13_l1901_190168


namespace NUMINAMATH_CALUDE_prob_only_one_value_l1901_190197

/-- The probability that student A solves the problem -/
def prob_A : ℚ := 1/2

/-- The probability that student B solves the problem -/
def prob_B : ℚ := 1/3

/-- The probability that student C solves the problem -/
def prob_C : ℚ := 1/4

/-- The probability that only one student solves the problem -/
def prob_only_one : ℚ :=
  prob_A * (1 - prob_B) * (1 - prob_C) +
  prob_B * (1 - prob_A) * (1 - prob_C) +
  prob_C * (1 - prob_A) * (1 - prob_B)

theorem prob_only_one_value : prob_only_one = 11/24 := by
  sorry

end NUMINAMATH_CALUDE_prob_only_one_value_l1901_190197


namespace NUMINAMATH_CALUDE_intersection_not_roots_l1901_190138

theorem intersection_not_roots : ∀ x : ℝ,
  (x^2 - 1 = x + 7) → (x^2 + x - 6 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_intersection_not_roots_l1901_190138


namespace NUMINAMATH_CALUDE_positive_solution_x_l1901_190142

theorem positive_solution_x (x y z : ℝ) 
  (eq1 : x * y = 12 - 2 * x - 3 * y)
  (eq2 : y * z = 8 - 4 * y - 2 * z)
  (eq3 : x * z = 24 - 4 * x - 3 * z)
  (x_pos : x > 0) :
  x = 3 := by sorry

end NUMINAMATH_CALUDE_positive_solution_x_l1901_190142


namespace NUMINAMATH_CALUDE_integral_sine_product_zero_and_no_beta_solution_l1901_190151

theorem integral_sine_product_zero_and_no_beta_solution 
  (m n : ℕ) (h_distinct : m ≠ n) (h_positive_m : m > 0) (h_positive_n : n > 0) :
  (∀ α : ℝ, |α| < 1 → ∫ x in -π..π, Real.sin ((m : ℝ) + α) * x * Real.sin ((n : ℝ) + α) * x = 0) ∧
  ¬ ∃ β : ℝ, (∫ x in -π..π, Real.sin ((m : ℝ) + β) * x ^ 2 = π + 2 / (4 * m - 1)) ∧
             (∫ x in -π..π, Real.sin ((n : ℝ) + β) * x ^ 2 = π + 2 / (4 * n - 1)) :=
by sorry

end NUMINAMATH_CALUDE_integral_sine_product_zero_and_no_beta_solution_l1901_190151


namespace NUMINAMATH_CALUDE_function_inequality_l1901_190160

theorem function_inequality (a b : ℝ) (f g : ℝ → ℝ) 
  (h₁ : a ≤ b)
  (h₂ : DifferentiableOn ℝ f (Set.Icc a b))
  (h₃ : DifferentiableOn ℝ g (Set.Icc a b))
  (h₄ : ∀ x ∈ Set.Icc a b, deriv f x > deriv g x)
  (h₅ : f a = g a) :
  ∀ x ∈ Set.Icc a b, f x ≥ g x :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_l1901_190160


namespace NUMINAMATH_CALUDE_octagon_area_l1901_190195

/-- The area of an octagon formed by the intersection of two concentric squares -/
theorem octagon_area (side_large : ℝ) (side_small : ℝ) (octagon_side : ℝ) : 
  side_large = 2 →
  side_small = 1 →
  octagon_side = 17/36 →
  let octagon_area := 8 * (1/2 * octagon_side * side_small)
  octagon_area = 17/9 := by
sorry

end NUMINAMATH_CALUDE_octagon_area_l1901_190195


namespace NUMINAMATH_CALUDE_players_who_quit_correct_players_who_quit_l1901_190145

theorem players_who_quit (initial_players : ℕ) (lives_per_player : ℕ) (total_lives : ℕ) : ℕ :=
  initial_players - (total_lives / lives_per_player)

theorem correct_players_who_quit :
  players_who_quit 8 3 15 = 3 := by
  sorry

end NUMINAMATH_CALUDE_players_who_quit_correct_players_who_quit_l1901_190145


namespace NUMINAMATH_CALUDE_even_games_player_exists_l1901_190161

/-- Represents a player in the chess tournament -/
structure Player where
  id : Nat
  gamesPlayed : Nat

/-- Represents the state of a round-robin chess tournament -/
structure ChessTournament where
  players : Finset Player
  numPlayers : Nat
  h_numPlayers : numPlayers = 17

/-- The main theorem to prove -/
theorem even_games_player_exists (tournament : ChessTournament) :
  ∃ p ∈ tournament.players, Even p.gamesPlayed :=
sorry

end NUMINAMATH_CALUDE_even_games_player_exists_l1901_190161


namespace NUMINAMATH_CALUDE_hyperbola_real_axis_x_implies_mn_negative_mn_negative_not_sufficient_for_real_axis_x_mn_negative_necessary_not_sufficient_l1901_190144

/-- Represents a hyperbola equation of the form x²/m + y²/n = 1 -/
structure Hyperbola (m n : ℝ) where
  equation : ∀ (x y : ℝ), x^2 / m + y^2 / n = 1

/-- Predicate to check if a hyperbola has its real axis on the x-axis -/
def has_real_axis_on_x (h : Hyperbola m n) : Prop :=
  m > 0 ∧ n < 0

theorem hyperbola_real_axis_x_implies_mn_negative 
  (m n : ℝ) (h : Hyperbola m n) :
  has_real_axis_on_x h → m * n < 0 := by
  sorry

theorem mn_negative_not_sufficient_for_real_axis_x :
  ∃ (m n : ℝ), m * n < 0 ∧ 
  ∃ (h : Hyperbola m n), ¬(has_real_axis_on_x h) := by
  sorry

/-- The main theorem stating that m * n < 0 is a necessary but not sufficient condition -/
theorem mn_negative_necessary_not_sufficient (m n : ℝ) (h : Hyperbola m n) :
  (has_real_axis_on_x h → m * n < 0) ∧
  ¬(m * n < 0 → has_real_axis_on_x h) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_real_axis_x_implies_mn_negative_mn_negative_not_sufficient_for_real_axis_x_mn_negative_necessary_not_sufficient_l1901_190144


namespace NUMINAMATH_CALUDE_pencil_sharpening_theorem_l1901_190170

/-- Calculates the final length of a pencil after sharpening on two consecutive days. -/
def pencil_length (initial_length : ℕ) (day1_sharpening : ℕ) (day2_sharpening : ℕ) : ℕ :=
  initial_length - day1_sharpening - day2_sharpening

/-- Proves that a 22-inch pencil sharpened by 2 inches on two consecutive days will be 18 inches long. -/
theorem pencil_sharpening_theorem :
  pencil_length 22 2 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_pencil_sharpening_theorem_l1901_190170


namespace NUMINAMATH_CALUDE_expected_abs_difference_10_days_l1901_190135

/-- Represents the wealth difference between two entities -/
def WealthDifference := ℤ

/-- Probability of each outcome -/
def p_cat_wins : ℝ := 0.25
def p_fox_wins : ℝ := 0.25
def p_both_lose : ℝ := 0.5

/-- Number of days -/
def num_days : ℕ := 10

/-- Expected value of absolute wealth difference after n days -/
def expected_abs_difference (n : ℕ) : ℝ := sorry

/-- Theorem: Expected absolute wealth difference after 10 days is 1 -/
theorem expected_abs_difference_10_days :
  expected_abs_difference num_days = 1 := by sorry

end NUMINAMATH_CALUDE_expected_abs_difference_10_days_l1901_190135


namespace NUMINAMATH_CALUDE_bird_cage_problem_l1901_190180

theorem bird_cage_problem (N : ℚ) : 
  (5/8 * (4/5 * (1/2 * N + 12) + 20) = 60) → N = 166 := by
  sorry

end NUMINAMATH_CALUDE_bird_cage_problem_l1901_190180


namespace NUMINAMATH_CALUDE_P_and_S_not_fourth_l1901_190107

-- Define the set of runners
inductive Runner : Type
| P | Q | R | S | T | U

-- Define the relation "finishes before"
def finishes_before (a b : Runner) : Prop := sorry

-- Define the conditions
axiom P_beats_R : finishes_before Runner.P Runner.R
axiom P_beats_S : finishes_before Runner.P Runner.S
axiom Q_beats_S : finishes_before Runner.Q Runner.S
axiom Q_before_U : finishes_before Runner.Q Runner.U
axiom U_before_P : finishes_before Runner.U Runner.P
axiom T_before_U : finishes_before Runner.T Runner.U
axiom T_before_Q : finishes_before Runner.T Runner.Q

-- Define what it means to finish fourth
def finishes_fourth (r : Runner) : Prop := 
  ∃ a b c : Runner, 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    a ≠ r ∧ b ≠ r ∧ c ≠ r ∧
    finishes_before a r ∧ 
    finishes_before b r ∧ 
    finishes_before c r ∧
    (∀ x : Runner, x ≠ r → x ≠ a → x ≠ b → x ≠ c → finishes_before r x)

-- Theorem to prove
theorem P_and_S_not_fourth : 
  ¬(finishes_fourth Runner.P) ∧ ¬(finishes_fourth Runner.S) :=
sorry

end NUMINAMATH_CALUDE_P_and_S_not_fourth_l1901_190107


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l1901_190122

theorem gcd_of_three_numbers : Nat.gcd 8650 (Nat.gcd 11570 28980) = 10 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l1901_190122


namespace NUMINAMATH_CALUDE_kelly_snacks_weight_l1901_190185

/-- The weight of peanuts Kelly bought in pounds -/
def peanuts_weight : ℝ := 0.1

/-- The weight of raisins Kelly bought in pounds -/
def raisins_weight : ℝ := 0.4

/-- The weight of almonds Kelly bought in pounds -/
def almonds_weight : ℝ := 0.3

/-- The total weight of snacks Kelly bought -/
def total_weight : ℝ := peanuts_weight + raisins_weight + almonds_weight

theorem kelly_snacks_weight : total_weight = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_kelly_snacks_weight_l1901_190185


namespace NUMINAMATH_CALUDE_contrapositive_not_true_l1901_190175

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = k • b ∨ b = k • a

/-- Two vectors have the same direction if they are positive scalar multiples of each other -/
def same_direction (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ (a = k • b ∨ b = k • a)

/-- The original proposition -/
def original_proposition : Prop :=
  ∀ a b : ℝ × ℝ, collinear a b → same_direction a b

/-- The contrapositive of the original proposition -/
def contrapositive : Prop :=
  ∀ a b : ℝ × ℝ, ¬ same_direction a b → ¬ collinear a b

theorem contrapositive_not_true : ¬ contrapositive := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_not_true_l1901_190175


namespace NUMINAMATH_CALUDE_triangle_semiperimeter_inequality_l1901_190165

/-- 
For any triangle with semiperimeter p, incircle radius r, and circumcircle radius R,
the inequality p ≥ (3/2) * sqrt(6 * R * r) holds.
-/
theorem triangle_semiperimeter_inequality (p r R : ℝ) 
  (hp : p > 0) (hr : r > 0) (hR : R > 0) : p ≥ (3/2) * Real.sqrt (6 * R * r) := by
  sorry

end NUMINAMATH_CALUDE_triangle_semiperimeter_inequality_l1901_190165


namespace NUMINAMATH_CALUDE_factorization_cubic_minus_linear_l1901_190147

theorem factorization_cubic_minus_linear (x : ℝ) : x^3 - 9*x = x*(x+3)*(x-3) := by sorry

end NUMINAMATH_CALUDE_factorization_cubic_minus_linear_l1901_190147


namespace NUMINAMATH_CALUDE_brothers_age_difference_l1901_190131

/-- Represents a year in the format ABCD where A, B, C, D are digits --/
structure Year :=
  (value : ℕ)
  (in_19th_century : value ≥ 1800 ∧ value < 1900)

/-- Represents a year in the format ABCD where A, B, C, D are digits --/
structure Year' :=
  (value : ℕ)
  (in_20th_century : value ≥ 1900 ∧ value < 2000)

/-- Sum of digits of a number --/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- Age of a person born in year y at the current year current_year --/
def age (y : ℕ) (current_year : ℕ) : ℕ := current_year - y

theorem brothers_age_difference 
  (peter_birth : Year) 
  (paul_birth : Year') 
  (current_year : ℕ) 
  (h1 : age peter_birth.value current_year = sum_of_digits peter_birth.value)
  (h2 : age paul_birth.value current_year = sum_of_digits paul_birth.value) :
  age peter_birth.value current_year - age paul_birth.value current_year = 9 :=
sorry

end NUMINAMATH_CALUDE_brothers_age_difference_l1901_190131


namespace NUMINAMATH_CALUDE_intersection_A_B_l1901_190159

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.log (-x^2 + 2*x)}

-- Define set B
def B : Set ℝ := {x | |x| ≤ 1}

-- Theorem statement
theorem intersection_A_B :
  A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1901_190159


namespace NUMINAMATH_CALUDE_sqrt_product_equation_l1901_190108

theorem sqrt_product_equation (y : ℝ) (h_pos : y > 0) 
  (h_eq : Real.sqrt (12 * y) * Real.sqrt (25 * y) * Real.sqrt (5 * y) * Real.sqrt (20 * y) = 40) :
  y = (Real.sqrt 30 * Real.rpow 3 (1/4)) / 15 := by
sorry

end NUMINAMATH_CALUDE_sqrt_product_equation_l1901_190108


namespace NUMINAMATH_CALUDE_certain_number_problem_l1901_190105

theorem certain_number_problem (x : ℚ) :
  (2 / 5 : ℚ) * 300 - (3 / 5 : ℚ) * x = 45 → x = 125 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l1901_190105


namespace NUMINAMATH_CALUDE_quadratic_monotone_decreasing_implies_m_range_l1901_190140

/-- If the quadratic function f(x) = x^2 - 2mx + 1 is monotonically decreasing
    on the interval (-∞, 1), then m is greater than or equal to 1. -/
theorem quadratic_monotone_decreasing_implies_m_range 
  (f : ℝ → ℝ) 
  (h_def : ∀ x, f x = x^2 - 2*m*x + 1) 
  (h_decreasing : ∀ x y, x < y → x < 1 → f x > f y) : 
  m ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_monotone_decreasing_implies_m_range_l1901_190140


namespace NUMINAMATH_CALUDE_pascal_triangle_sum_l1901_190124

/-- The number of elements in a row of Pascal's Triangle -/
def elementsInRow (n : ℕ) : ℕ := n + 1

/-- The sum of elements in the first n rows of Pascal's Triangle -/
def sumOfElements (n : ℕ) : ℕ := (n + 1) * (n + 2) / 2

theorem pascal_triangle_sum :
  sumOfElements 29 = 465 := by sorry

end NUMINAMATH_CALUDE_pascal_triangle_sum_l1901_190124


namespace NUMINAMATH_CALUDE_right_rectangular_prism_volume_l1901_190184

theorem right_rectangular_prism_volume (a b c : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  a * b = 54 →
  b * c = 56 →
  a * c = 60 →
  abs (a * b * c - 426) < 0.5 :=
by sorry

end NUMINAMATH_CALUDE_right_rectangular_prism_volume_l1901_190184


namespace NUMINAMATH_CALUDE_largest_number_l1901_190153

theorem largest_number : 
  0.9989 > 0.998 ∧ 
  0.9989 > 0.9899 ∧ 
  0.9989 > 0.9 ∧ 
  0.9989 > 0.8999 :=
by sorry

end NUMINAMATH_CALUDE_largest_number_l1901_190153


namespace NUMINAMATH_CALUDE_f_differentiable_at_sqrt_non_square_l1901_190156

/-- A function f: ℝ → ℝ defined as follows:
    f(x) = 0 if x is irrational
    f(p/q) = 1/q³ if p ∈ ℤ, q ∈ ℕ, and p/q is in lowest terms -/
def f : ℝ → ℝ := sorry

/-- Predicate to check if a natural number is not a perfect square -/
def is_not_perfect_square (k : ℕ) : Prop := ∀ n : ℕ, n^2 ≠ k

theorem f_differentiable_at_sqrt_non_square (k : ℕ) (h : is_not_perfect_square k) :
  DifferentiableAt ℝ f (Real.sqrt k) ∧ deriv f (Real.sqrt k) = 0 := by sorry

end NUMINAMATH_CALUDE_f_differentiable_at_sqrt_non_square_l1901_190156


namespace NUMINAMATH_CALUDE_highway_length_is_500_l1901_190191

/-- The length of a highway where two cars meet after traveling from opposite ends -/
def highway_length (speed1 speed2 : ℝ) (time : ℝ) : ℝ :=
  speed1 * time + speed2 * time

/-- Theorem stating the length of the highway is 500 miles -/
theorem highway_length_is_500 :
  highway_length 40 60 5 = 500 := by
  sorry

end NUMINAMATH_CALUDE_highway_length_is_500_l1901_190191


namespace NUMINAMATH_CALUDE_correct_factorization_l1901_190127

theorem correct_factorization (x : ℝ) : x^2 - 3*x + 2 = (x - 1)*(x - 2) := by
  sorry

end NUMINAMATH_CALUDE_correct_factorization_l1901_190127


namespace NUMINAMATH_CALUDE_ms_jones_class_size_l1901_190130

theorem ms_jones_class_size :
  ∀ (num_students : ℕ),
    (num_students : ℝ) * 0.3 * (1/3) * 10 = 50 →
    num_students = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_ms_jones_class_size_l1901_190130


namespace NUMINAMATH_CALUDE_divide_eight_by_repeating_third_l1901_190100

-- Define the repeating decimal 0.overline{3}
def repeating_third : ℚ := 1/3

-- State the theorem
theorem divide_eight_by_repeating_third : 8 / repeating_third = 24 := by
  sorry

end NUMINAMATH_CALUDE_divide_eight_by_repeating_third_l1901_190100


namespace NUMINAMATH_CALUDE_banana_permutations_eq_60_l1901_190171

/-- The number of distinct permutations of the letters in "BANANA" -/
def banana_permutations : ℕ :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

/-- Theorem stating that the number of distinct permutations of "BANANA" is 60 -/
theorem banana_permutations_eq_60 : banana_permutations = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_permutations_eq_60_l1901_190171


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1901_190129

theorem inequality_solution_set (a : ℝ) (h : a > 1) :
  {x : ℝ | (x - a) * (x - 1/a) > 0} = {x : ℝ | x < 1/a ∨ x > a} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1901_190129


namespace NUMINAMATH_CALUDE_system_three_solutions_l1901_190133

def system (a : ℝ) (x y : ℝ) : Prop :=
  y = |x - Real.sqrt a| + Real.sqrt a - 2 ∧
  (|x| - 4)^2 + (|y| - 3)^2 = 25

def has_exactly_three_solutions (a : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    system a x₁ y₁ ∧ system a x₂ y₂ ∧ system a x₃ y₃ ∧
    (∀ x y, system a x y → (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂) ∨ (x = x₃ ∧ y = y₃))

theorem system_three_solutions :
  ∀ a : ℝ, has_exactly_three_solutions a ↔ a = 1 ∨ a = 16 ∨ a = ((5 * Real.sqrt 2 + 1) / 2)^2 :=
sorry

end NUMINAMATH_CALUDE_system_three_solutions_l1901_190133


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l1901_190198

/-- Given a complex number z such that z / (1 - z) = 2i, prove that z is in the first quadrant -/
theorem z_in_first_quadrant (z : ℂ) (h : z / (1 - z) = Complex.I * 2) : 
  Complex.re z > 0 ∧ Complex.im z > 0 := by
  sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l1901_190198


namespace NUMINAMATH_CALUDE_integer_square_root_l1901_190123

theorem integer_square_root (x : ℤ) : 
  (∃ n : ℤ, n ≥ 0 ∧ n^2 = x^2 - x + 1) ↔ (x = 0 ∨ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_integer_square_root_l1901_190123


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1901_190178

theorem sufficient_not_necessary (a b : ℝ) :
  (∀ a b : ℝ, a > 2 ∧ b > 2 → a + b > 4) ∧
  (∃ a b : ℝ, a + b > 4 ∧ ¬(a > 2 ∧ b > 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1901_190178


namespace NUMINAMATH_CALUDE_sin_50_plus_sqrt3_tan_10_equals_1_l1901_190193

theorem sin_50_plus_sqrt3_tan_10_equals_1 : 
  Real.sin (50 * π / 180) * (1 + Real.sqrt 3 * Real.tan (10 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_50_plus_sqrt3_tan_10_equals_1_l1901_190193


namespace NUMINAMATH_CALUDE_negative_quarter_power_times_four_power_l1901_190116

theorem negative_quarter_power_times_four_power (n : ℕ) :
  ((-0.25 : ℝ) ^ n) * (4 ^ (n + 1)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_negative_quarter_power_times_four_power_l1901_190116


namespace NUMINAMATH_CALUDE_basketball_teams_count_l1901_190166

/-- The number of combinations of n items taken k at a time -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The total number of people available for the basketball game -/
def total_people : ℕ := 7

/-- The number of players needed for each team -/
def team_size : ℕ := 4

/-- Theorem: The number of different teams of 4 that can be formed from 7 people is 35 -/
theorem basketball_teams_count : binomial total_people team_size = 35 := by sorry

end NUMINAMATH_CALUDE_basketball_teams_count_l1901_190166


namespace NUMINAMATH_CALUDE_bucket_weight_calculation_l1901_190179

/-- Given an initial weight of shells and an additional weight of shells,
    calculate the total weight of shells in the bucket. -/
def total_weight (initial_weight additional_weight : ℕ) : ℕ :=
  initial_weight + additional_weight

/-- Theorem stating that given 5 pounds of initial weight and 12 pounds of additional weight,
    the total weight of shells in the bucket is 17 pounds. -/
theorem bucket_weight_calculation :
  total_weight 5 12 = 17 := by
  sorry

end NUMINAMATH_CALUDE_bucket_weight_calculation_l1901_190179


namespace NUMINAMATH_CALUDE_addition_puzzle_l1901_190154

theorem addition_puzzle (E S X : Nat) : 
  E ≠ 0 → S ≠ 0 → X ≠ 0 →
  E ≠ S → E ≠ X → S ≠ X →
  E * 100 + E * 10 + E + E * 100 + E * 10 + E = S * 100 + X * 10 + S →
  X = 7 := by
sorry

end NUMINAMATH_CALUDE_addition_puzzle_l1901_190154


namespace NUMINAMATH_CALUDE_cycling_equation_correct_l1901_190113

/-- Represents the scenario of two employees cycling to work -/
def cycling_scenario (x : ℝ) : Prop :=
  let distance : ℝ := 5000
  let speed_ratio : ℝ := 1.5
  let time_difference : ℝ := 10
  (distance / x) - (distance / (speed_ratio * x)) = time_difference

/-- Proves that the equation correctly represents the cycling scenario -/
theorem cycling_equation_correct :
  ∀ x : ℝ, x > 0 → cycling_scenario x :=
by
  sorry

end NUMINAMATH_CALUDE_cycling_equation_correct_l1901_190113


namespace NUMINAMATH_CALUDE_airport_exchange_rate_l1901_190102

theorem airport_exchange_rate (euros : ℝ) (official_rate : ℝ) (airport_rate_factor : ℝ) :
  euros = 70 →
  official_rate = 5 →
  airport_rate_factor = 5 / 7 →
  (euros / official_rate) * airport_rate_factor = 10 := by
  sorry

end NUMINAMATH_CALUDE_airport_exchange_rate_l1901_190102


namespace NUMINAMATH_CALUDE_courtyard_width_l1901_190177

/-- Represents the dimensions of a brick in centimeters -/
structure Brick where
  length : ℝ
  width : ℝ

/-- Represents a rectangular courtyard -/
structure Courtyard where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle given its length and width -/
def area (length width : ℝ) : ℝ := length * width

/-- Theorem: The width of the courtyard is 15 meters -/
theorem courtyard_width (b : Brick) (c : Courtyard) (total_bricks : ℕ) :
  b.length = 0.2 →
  b.width = 0.1 →
  c.length = 25 →
  total_bricks = 18750 →
  area c.length c.width = (total_bricks : ℝ) * area b.length b.width →
  c.width = 15 := by
  sorry

#check courtyard_width

end NUMINAMATH_CALUDE_courtyard_width_l1901_190177


namespace NUMINAMATH_CALUDE_cube_preserves_order_l1901_190104

theorem cube_preserves_order (a b : ℝ) (h : a > b) : a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_preserves_order_l1901_190104


namespace NUMINAMATH_CALUDE_cube_of_m_equals_64_l1901_190110

theorem cube_of_m_equals_64 (m : ℕ) (h : 3^m = 81) : m^3 = 64 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_m_equals_64_l1901_190110


namespace NUMINAMATH_CALUDE_not_all_n_squared_plus_n_plus_41_prime_l1901_190128

theorem not_all_n_squared_plus_n_plus_41_prime :
  ∃ n : ℕ, ¬(Nat.Prime (n^2 + n + 41)) := by
  sorry

end NUMINAMATH_CALUDE_not_all_n_squared_plus_n_plus_41_prime_l1901_190128


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l1901_190199

theorem fraction_equation_solution (a b : ℝ) (h : a / b = 5 / 4) :
  ∃ x : ℝ, (4 * a + x * b) / (4 * a - x * b) = 4 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l1901_190199


namespace NUMINAMATH_CALUDE_total_fish_caught_l1901_190112

/-- The number of times Chris goes fishing -/
def chris_trips : ℕ := 10

/-- The number of fish Brian catches per trip -/
def brian_fish_per_trip : ℕ := 400

/-- The ratio of Brian's fishing frequency to Chris's -/
def brian_frequency_ratio : ℚ := 2

/-- The fraction of fish Brian catches compared to Chris per trip -/
def brian_catch_fraction : ℚ := 3/5

theorem total_fish_caught :
  let brian_trips := chris_trips * brian_frequency_ratio
  let chris_fish_per_trip := brian_fish_per_trip / brian_catch_fraction
  let brian_total := brian_trips * brian_fish_per_trip
  let chris_total := chris_trips * chris_fish_per_trip.floor
  brian_total + chris_total = 14660 := by
sorry

end NUMINAMATH_CALUDE_total_fish_caught_l1901_190112


namespace NUMINAMATH_CALUDE_smallest_angle_in_3_4_5_ratio_triangle_l1901_190134

theorem smallest_angle_in_3_4_5_ratio_triangle (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- angles are positive
  a + b + c = 180 →  -- sum of angles in a triangle
  ∃ (k : ℝ), a = 3*k ∧ b = 4*k ∧ c = 5*k →  -- angles are in ratio 3:4:5
  min a (min b c) = 45 :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_in_3_4_5_ratio_triangle_l1901_190134


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l1901_190137

theorem unique_three_digit_number : ∃! x : ℕ, 
  100 ≤ x ∧ x < 1000 ∧ 
  (∃ k : ℤ, x - 7 = 7 * k) ∧
  (∃ l : ℤ, x - 8 = 8 * l) ∧
  (∃ m : ℤ, x - 9 = 9 * m) :=
by sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l1901_190137


namespace NUMINAMATH_CALUDE_arrangements_count_l1901_190101

/-- Represents the number of students -/
def total_students : ℕ := 6

/-- Represents the number of male students -/
def male_students : ℕ := 3

/-- Represents the number of female students -/
def female_students : ℕ := 3

/-- Represents that exactly two female students stand next to each other -/
def adjacent_female_students : ℕ := 2

/-- Calculates the number of arrangements satisfying the given conditions -/
def num_arrangements : ℕ := 288

/-- Theorem stating that the number of arrangements satisfying the given conditions is 288 -/
theorem arrangements_count :
  (total_students = male_students + female_students) →
  (male_students = 3) →
  (female_students = 3) →
  (adjacent_female_students = 2) →
  num_arrangements = 288 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_count_l1901_190101


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1901_190183

theorem complex_equation_solution (b : ℂ) : (1 + b * Complex.I) * Complex.I = -1 + Complex.I → b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1901_190183


namespace NUMINAMATH_CALUDE_min_quotient_four_digit_number_l1901_190190

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧
  (∃ a b c d : ℕ, n = 1000 * a + 100 * b + 10 * c + d ∧
    0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (Even a ∨ Even b ∨ Even c ∨ Even d) ∧
    (Even a ∧ Even b ∨ Even a ∧ Even c ∨ Even a ∧ Even d ∨
     Even b ∧ Even c ∨ Even b ∧ Even d ∨ Even c ∧ Even d))

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  List.sum digits

theorem min_quotient_four_digit_number :
  ∀ n : ℕ, is_valid_number n → (n : ℚ) / (digit_sum n : ℚ) ≥ 87 :=
by sorry

end NUMINAMATH_CALUDE_min_quotient_four_digit_number_l1901_190190


namespace NUMINAMATH_CALUDE_license_plate_count_l1901_190188

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of positions for letters on the license plate -/
def letter_positions : ℕ := 4

/-- The number of digits on the license plate -/
def digit_positions : ℕ := 2

/-- The number of possible digits (0-9) -/
def digit_options : ℕ := 10

/-- Calculates the number of license plate combinations -/
def license_plate_combinations : ℕ :=
  alphabet_size * (alphabet_size - 1).choose 2 * letter_positions.choose 2 * 2 * digit_options * (digit_options - 1)

theorem license_plate_count :
  license_plate_combinations = 8424000 :=
by sorry

end NUMINAMATH_CALUDE_license_plate_count_l1901_190188
