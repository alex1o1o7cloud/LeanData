import Mathlib

namespace NUMINAMATH_CALUDE_ali_wallet_l3594_359434

def wallet_problem (num_five_dollar_bills : ℕ) (total_amount : ℕ) : ℕ := 
  let five_dollar_amount := 5 * num_five_dollar_bills
  let ten_dollar_amount := total_amount - five_dollar_amount
  ten_dollar_amount / 10

theorem ali_wallet :
  wallet_problem 7 45 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ali_wallet_l3594_359434


namespace NUMINAMATH_CALUDE_first_row_valid_l3594_359480

def is_valid_row (row : List Nat) : Prop :=
  row.length = 5 ∧ row.toFinset = {1, 2, 3, 4, 5}

theorem first_row_valid : is_valid_row [2, 5, 1, 3, 4] := by
  sorry

end NUMINAMATH_CALUDE_first_row_valid_l3594_359480


namespace NUMINAMATH_CALUDE_sum_product_ratio_theorem_l3594_359493

theorem sum_product_ratio_theorem (x y z : ℝ) (hxy : x ≠ y) (hyz : y ≠ z) (hxz : x ≠ z) (hsum : x + y + z = 12) :
  (x * y + y * z + z * x) / (x^2 + y^2 + z^2) = (144 - (x^2 + y^2 + z^2)) / (2 * (x^2 + y^2 + z^2)) := by
  sorry

end NUMINAMATH_CALUDE_sum_product_ratio_theorem_l3594_359493


namespace NUMINAMATH_CALUDE_deal_or_no_deal_probability_l3594_359482

theorem deal_or_no_deal_probability (total_boxes : Nat) (high_value_boxes : Nat) 
  (h1 : total_boxes = 26)
  (h2 : high_value_boxes = 7) :
  total_boxes - (high_value_boxes + high_value_boxes) = 12 := by
sorry

end NUMINAMATH_CALUDE_deal_or_no_deal_probability_l3594_359482


namespace NUMINAMATH_CALUDE_remainder_eight_power_2002_mod_9_l3594_359438

theorem remainder_eight_power_2002_mod_9 : 8^2002 % 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_eight_power_2002_mod_9_l3594_359438


namespace NUMINAMATH_CALUDE_daily_lottery_expected_profit_l3594_359415

/-- The expected profit from purchasing one "Daily Lottery" ticket -/
def expected_profit : ℝ := -0.9

/-- The price of one lottery ticket -/
def ticket_price : ℝ := 2

/-- The probability of winning the first prize -/
def first_prize_prob : ℝ := 0.001

/-- The probability of winning the second prize -/
def second_prize_prob : ℝ := 0.1

/-- The amount of the first prize -/
def first_prize_amount : ℝ := 100

/-- The amount of the second prize -/
def second_prize_amount : ℝ := 10

theorem daily_lottery_expected_profit :
  expected_profit = 
    first_prize_prob * first_prize_amount + 
    second_prize_prob * second_prize_amount - 
    ticket_price := by
  sorry

end NUMINAMATH_CALUDE_daily_lottery_expected_profit_l3594_359415


namespace NUMINAMATH_CALUDE_grunters_win_probability_l3594_359413

theorem grunters_win_probability : 
  let n_games : ℕ := 6
  let p_first_half : ℚ := 3/4
  let p_second_half : ℚ := 4/5
  let n_first_half : ℕ := 3
  let n_second_half : ℕ := 3
  
  (n_first_half + n_second_half = n_games) →
  (p_first_half ^ n_first_half * p_second_half ^ n_second_half = 27/125) :=
by sorry

end NUMINAMATH_CALUDE_grunters_win_probability_l3594_359413


namespace NUMINAMATH_CALUDE_exam_score_l3594_359428

/-- Calculates the total marks in an examination based on given parameters. -/
def totalMarks (totalQuestions : ℕ) (correctMarks : ℤ) (wrongMarks : ℤ) (correctAnswers : ℕ) : ℤ :=
  (correctAnswers : ℤ) * correctMarks + (totalQuestions - correctAnswers : ℤ) * wrongMarks

/-- Theorem stating that under the given conditions, the student secures 130 marks. -/
theorem exam_score :
  totalMarks 80 4 (-1) 42 = 130 := by
  sorry

end NUMINAMATH_CALUDE_exam_score_l3594_359428


namespace NUMINAMATH_CALUDE_sqrt_sum_comparison_l3594_359492

theorem sqrt_sum_comparison : Real.sqrt 11 + Real.sqrt 7 > Real.sqrt 13 + Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_comparison_l3594_359492


namespace NUMINAMATH_CALUDE_haley_marbles_count_l3594_359412

def number_of_boys : ℕ := 2
def marbles_per_boy : ℕ := 10

theorem haley_marbles_count :
  number_of_boys * marbles_per_boy = 20 :=
by sorry

end NUMINAMATH_CALUDE_haley_marbles_count_l3594_359412


namespace NUMINAMATH_CALUDE_local_value_of_three_is_300_l3594_359443

/-- Represents a four-digit number -/
structure FourDigitNumber where
  thousands : Nat
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_valid : thousands ≥ 1 ∧ thousands ≤ 9 ∧
             hundreds ≥ 0 ∧ hundreds ≤ 9 ∧
             tens ≥ 0 ∧ tens ≤ 9 ∧
             ones ≥ 0 ∧ ones ≤ 9

/-- Calculate the local value of a digit given its place value -/
def localValue (digit : Nat) (placeValue : Nat) : Nat :=
  digit * placeValue

/-- Theorem: In the number 2345, if the sum of local values of all digits is 2345,
    then the local value of the digit 3 is 300 -/
theorem local_value_of_three_is_300 (n : FourDigitNumber)
    (h1 : n.thousands = 2 ∧ n.hundreds = 3 ∧ n.tens = 4 ∧ n.ones = 5)
    (h2 : localValue n.thousands 1000 + localValue n.hundreds 100 +
          localValue n.tens 10 + localValue n.ones 1 = 2345) :
    localValue n.hundreds 100 = 300 := by
  sorry

end NUMINAMATH_CALUDE_local_value_of_three_is_300_l3594_359443


namespace NUMINAMATH_CALUDE_initial_bees_correct_l3594_359433

/-- The initial number of bees in the colony. -/
def initial_bees : ℕ := 80000

/-- The daily loss of bees. -/
def daily_loss : ℕ := 1200

/-- The number of days after which the colony reaches a fourth of its initial size. -/
def days : ℕ := 50

/-- Theorem stating that the initial number of bees is correct given the conditions. -/
theorem initial_bees_correct : 
  initial_bees = daily_loss * days * 4 / 3 :=
by sorry

end NUMINAMATH_CALUDE_initial_bees_correct_l3594_359433


namespace NUMINAMATH_CALUDE_factorial_20_19_div_5_is_perfect_square_l3594_359444

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem factorial_20_19_div_5_is_perfect_square :
  is_perfect_square ((factorial 20 * factorial 19) / 5) := by
  sorry

end NUMINAMATH_CALUDE_factorial_20_19_div_5_is_perfect_square_l3594_359444


namespace NUMINAMATH_CALUDE_matthews_crackers_l3594_359445

theorem matthews_crackers (num_friends : ℕ) (crackers_eaten_per_friend : ℕ) 
  (h1 : num_friends = 18)
  (h2 : crackers_eaten_per_friend = 2) :
  num_friends * crackers_eaten_per_friend = 36 := by
  sorry

end NUMINAMATH_CALUDE_matthews_crackers_l3594_359445


namespace NUMINAMATH_CALUDE_exists_equal_face_products_l3594_359449

/-- A cube arrangement is a function from the set of 12 edges to the set of numbers 1 to 12 -/
def CubeArrangement := Fin 12 → Fin 12

/-- The set of edges on the top face of the cube -/
def topFace : Finset (Fin 12) := {0, 1, 2, 3}

/-- The set of edges on the bottom face of the cube -/
def bottomFace : Finset (Fin 12) := {4, 5, 6, 7}

/-- The product of numbers on a given face for a given arrangement -/
def faceProduct (arrangement : CubeArrangement) (face : Finset (Fin 12)) : ℕ :=
  face.prod (fun edge => (arrangement edge).val + 1)

/-- Theorem stating that there exists a cube arrangement where the product of
    numbers on the top face equals the product of numbers on the bottom face -/
theorem exists_equal_face_products : ∃ (arrangement : CubeArrangement),
  faceProduct arrangement topFace = faceProduct arrangement bottomFace := by
  sorry

end NUMINAMATH_CALUDE_exists_equal_face_products_l3594_359449


namespace NUMINAMATH_CALUDE_max_adjacent_squares_l3594_359473

/-- A square with side length 1 -/
def UnitSquare : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

/-- Two squares are adjacent if they share at least one point on their boundaries -/
def Adjacent (s1 s2 : Set (ℝ × ℝ)) : Prop :=
  ∃ p, p ∈ (frontier s1) ∩ (frontier s2)

/-- Two squares are non-overlapping if their interiors are disjoint -/
def NonOverlapping (s1 s2 : Set (ℝ × ℝ)) : Prop :=
  interior s1 ∩ interior s2 = ∅

/-- A configuration of squares adjacent to a given square -/
def AdjacentSquares (n : ℕ) : Prop :=
  ∃ (squares : Fin n → Set (ℝ × ℝ)),
    (∀ i, squares i = UnitSquare) ∧
    (∀ i, Adjacent (squares i) UnitSquare) ∧
    (∀ i j, i ≠ j → NonOverlapping (squares i) (squares j))

/-- The maximum number of non-overlapping unit squares that can be placed adjacent to a given unit square is 8 -/
theorem max_adjacent_squares :
  (∀ n, AdjacentSquares n → n ≤ 8) ∧ AdjacentSquares 8 := by sorry

end NUMINAMATH_CALUDE_max_adjacent_squares_l3594_359473


namespace NUMINAMATH_CALUDE_total_egg_rolls_l3594_359494

theorem total_egg_rolls (omar_rolls karen_rolls : ℕ) 
  (h1 : omar_rolls = 219) 
  (h2 : karen_rolls = 229) : 
  omar_rolls + karen_rolls = 448 := by
sorry

end NUMINAMATH_CALUDE_total_egg_rolls_l3594_359494


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3594_359405

theorem quadratic_inequality_solution_set :
  let f : ℝ → ℝ := λ x => 3 * x^2 - 7 * x - 10
  let S : Set ℝ := {x | f x ≥ 0}
  S = {x | x ≥ 10/3 ∨ x ≤ -1} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3594_359405


namespace NUMINAMATH_CALUDE_playground_children_count_l3594_359496

theorem playground_children_count (boys girls : ℕ) 
  (h1 : boys = 44) 
  (h2 : girls = 53) : 
  boys + girls = 97 := by
  sorry

end NUMINAMATH_CALUDE_playground_children_count_l3594_359496


namespace NUMINAMATH_CALUDE_prob_red_then_black_54_card_deck_l3594_359429

/-- A deck of cards with red and black cards, including jokers -/
structure Deck :=
  (total_cards : ℕ)
  (red_cards : ℕ)
  (black_cards : ℕ)

/-- The probability of drawing a red card first and a black card second -/
def prob_red_then_black (d : Deck) : ℚ :=
  (d.red_cards : ℚ) * d.black_cards / (d.total_cards * (d.total_cards - 1))

/-- The theorem stating the probability for the specific deck -/
theorem prob_red_then_black_54_card_deck :
  prob_red_then_black ⟨54, 26, 28⟩ = 364 / 1431 := by sorry

end NUMINAMATH_CALUDE_prob_red_then_black_54_card_deck_l3594_359429


namespace NUMINAMATH_CALUDE_amy_green_balloons_l3594_359481

/-- The number of green balloons Amy has -/
def num_green_balloons (total red blue : ℕ) : ℕ := total - red - blue

/-- Theorem stating that Amy has 17 green balloons -/
theorem amy_green_balloons : 
  num_green_balloons 67 29 21 = 17 := by sorry

end NUMINAMATH_CALUDE_amy_green_balloons_l3594_359481


namespace NUMINAMATH_CALUDE_solitaire_game_solvable_l3594_359464

/-- Represents the state of a marker on the solitaire board -/
inductive MarkerState
| White
| Black

/-- Represents the solitaire game board -/
def Board (m n : ℕ) := Fin m → Fin n → MarkerState

/-- Initializes the board with all white markers except one black corner -/
def initBoard (m n : ℕ) : Board m n := sorry

/-- Represents a valid move in the game -/
def validMove (b : Board m n) (i : Fin m) (j : Fin n) : Prop := sorry

/-- The state of the board after making a move -/
def makeMove (b : Board m n) (i : Fin m) (j : Fin n) : Board m n := sorry

/-- Predicate to check if all markers have been removed from the board -/
def allMarkersRemoved (b : Board m n) : Prop := sorry

/-- Predicate to check if it's possible to remove all markers from the board -/
def canRemoveAllMarkers (m n : ℕ) : Prop := 
  ∃ (moves : List (Fin m × Fin n)), 
    let finalBoard := moves.foldl (λ b move => makeMove b move.1 move.2) (initBoard m n)
    allMarkersRemoved finalBoard

/-- The main theorem stating the condition for removing all markers -/
theorem solitaire_game_solvable (m n : ℕ) : 
  canRemoveAllMarkers m n ↔ m % 2 = 1 ∨ n % 2 = 1 := by sorry

end NUMINAMATH_CALUDE_solitaire_game_solvable_l3594_359464


namespace NUMINAMATH_CALUDE_part_one_part_two_l3594_359446

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 2*a - 1 < x ∧ x < -a}
def B : Set ℝ := {x | |x - 1| < 2}

-- Part 1
theorem part_one : (Aᶜ (-1) ∪ B) = {x | x ≤ -3 ∨ x > -1} := by sorry

-- Part 2
theorem part_two : ∀ a : ℝ, (A a ⊆ B ∧ A a ≠ B) ↔ a ≥ 0 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3594_359446


namespace NUMINAMATH_CALUDE_fraction_calculation_l3594_359447

theorem fraction_calculation : (1 / 3 : ℚ) * (4 / 7 : ℚ) * (9 / 13 : ℚ) + (1 / 2 : ℚ) = 49 / 78 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l3594_359447


namespace NUMINAMATH_CALUDE_multiplier_problem_l3594_359477

theorem multiplier_problem (m : ℝ) : m * 5.0 - 7 = 13 → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_multiplier_problem_l3594_359477


namespace NUMINAMATH_CALUDE_initial_water_temp_l3594_359458

-- Define the constants
def total_time : ℕ := 73
def temp_increase_per_minute : ℕ := 3
def boiling_point : ℕ := 212
def pasta_cooking_time : ℕ := 12

-- Define the theorem
theorem initial_water_temp (mixing_time : ℕ) (boiling_time : ℕ) 
  (h1 : mixing_time = pasta_cooking_time / 3)
  (h2 : boiling_time = total_time - (pasta_cooking_time + mixing_time))
  (h3 : boiling_point = temp_increase_per_minute * boiling_time + 41) :
  41 = boiling_point - temp_increase_per_minute * boiling_time :=
by sorry

end NUMINAMATH_CALUDE_initial_water_temp_l3594_359458


namespace NUMINAMATH_CALUDE_square_of_binomial_constant_l3594_359437

theorem square_of_binomial_constant (b : ℝ) : 
  (∃ (a c : ℝ), ∀ x, 16 * x^2 + 40 * x + b = (a * x + c)^2) → b = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_constant_l3594_359437


namespace NUMINAMATH_CALUDE_wire_cutting_is_random_event_l3594_359404

/-- An event that can occur but is not certain to occur -/
structure PossibleEvent where
  can_occur : Bool
  not_certain : Bool

/-- A random event is a possible event that exhibits regularity in repeated trials -/
structure RandomEvent extends PossibleEvent where
  exhibits_regularity : Bool

/-- The event of cutting a wire into three pieces to form a triangle -/
def wire_cutting_event (a : ℝ) : PossibleEvent :=
  { can_occur := true,
    not_certain := true }

/-- Theorem: The wire cutting event is a random event -/
theorem wire_cutting_is_random_event (a : ℝ) :
  ∃ (e : RandomEvent), (e.toPossibleEvent = wire_cutting_event a) :=
sorry

end NUMINAMATH_CALUDE_wire_cutting_is_random_event_l3594_359404


namespace NUMINAMATH_CALUDE_min_three_digit_quotient_l3594_359483

def three_digit_quotient (a b : ℕ) : ℚ :=
  (100 * a + 10 * b + 1) / (a + b + 1)

theorem min_three_digit_quotient :
  ∀ a b : ℕ, 2 ≤ a → a ≤ 9 → 2 ≤ b → b ≤ 9 → a ≠ b →
  three_digit_quotient a b ≥ 24.25 ∧
  ∃ a₀ b₀ : ℕ, 2 ≤ a₀ ∧ a₀ ≤ 9 ∧ 2 ≤ b₀ ∧ b₀ ≤ 9 ∧ a₀ ≠ b₀ ∧
  three_digit_quotient a₀ b₀ = 24.25 :=
sorry

end NUMINAMATH_CALUDE_min_three_digit_quotient_l3594_359483


namespace NUMINAMATH_CALUDE_tom_dimes_count_l3594_359459

/-- The number of dimes Tom initially had -/
def initial_dimes : ℕ := 15

/-- The number of dimes Tom's dad gave him -/
def dimes_from_dad : ℕ := 33

/-- The total number of dimes Tom has after receiving dimes from his dad -/
def total_dimes : ℕ := initial_dimes + dimes_from_dad

theorem tom_dimes_count : total_dimes = 48 := by
  sorry

end NUMINAMATH_CALUDE_tom_dimes_count_l3594_359459


namespace NUMINAMATH_CALUDE_new_speed_calculation_l3594_359423

theorem new_speed_calculation (distance : ℝ) (original_time : ℝ) 
  (h1 : distance = 469)
  (h2 : original_time = 6)
  (h3 : original_time > 0) :
  let new_time := original_time * (3/2)
  let new_speed := distance / new_time
  new_speed = distance / (original_time * (3/2)) := by
sorry

end NUMINAMATH_CALUDE_new_speed_calculation_l3594_359423


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l3594_359431

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem f_derivative_at_zero : 
  deriv f 0 = 1 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l3594_359431


namespace NUMINAMATH_CALUDE_right_triangle_leg_divisible_by_three_l3594_359495

theorem right_triangle_leg_divisible_by_three 
  (a b c : ℕ) -- a, b are legs, c is hypotenuse
  (h_right : a^2 + b^2 = c^2) -- Pythagorean theorem
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) -- Positive sides
  : 3 ∣ a ∨ 3 ∣ b := by
sorry

end NUMINAMATH_CALUDE_right_triangle_leg_divisible_by_three_l3594_359495


namespace NUMINAMATH_CALUDE_light_bulb_ratio_l3594_359410

theorem light_bulb_ratio (initial : ℕ) (used : ℕ) (left : ℕ) : 
  initial = 40 → used = 16 → left = 12 → 
  (initial - used - left) = left := by
  sorry

end NUMINAMATH_CALUDE_light_bulb_ratio_l3594_359410


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3594_359448

theorem simplify_and_evaluate (x y : ℤ) (hx : x = -3) (hy : y = 2) :
  (x + y)^2 - y * (2 * x - y) = 17 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3594_359448


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l3594_359421

theorem cube_root_equation_solution :
  ∃! x : ℝ, (3 - x / 2) ^ (1/3 : ℝ) = -4 :=
by
  -- The unique solution is x = 134
  use 134
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l3594_359421


namespace NUMINAMATH_CALUDE_rectangle_same_color_corners_l3594_359487

/-- A coloring of a rectangular board -/
def Coloring (m n : ℕ) := Fin m → Fin n → Bool

/-- A rectangle on the board -/
structure Rectangle (m n : ℕ) where
  top : Fin m
  bottom : Fin m
  left : Fin n
  right : Fin n
  h_top_lt_bottom : top < bottom

/-- The corners of a rectangle have the same color -/
def sameColorCorners (c : Coloring m n) (r : Rectangle m n) : Prop :=
  c r.top r.left = c r.top r.right ∧
  c r.top r.left = c r.bottom r.left ∧
  c r.top r.left = c r.bottom r.right

theorem rectangle_same_color_corners :
  ∀ (c : Coloring 4 7), ∃ (r : Rectangle 4 7), sameColorCorners c r :=
sorry

end NUMINAMATH_CALUDE_rectangle_same_color_corners_l3594_359487


namespace NUMINAMATH_CALUDE_first_number_in_sequence_l3594_359436

def sequence_sum (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n ≥ 4 → n ≤ 10 → a n = a (n-1) + a (n-2) + a (n-3)

theorem first_number_in_sequence 
  (a : ℕ → ℤ) 
  (h_sum : sequence_sum a) 
  (h_8 : a 8 = 29) 
  (h_9 : a 9 = 56) 
  (h_10 : a 10 = 108) : 
  a 1 = 32 := by sorry

end NUMINAMATH_CALUDE_first_number_in_sequence_l3594_359436


namespace NUMINAMATH_CALUDE_rachels_budget_l3594_359497

/-- Rachel's budget for a beauty and modeling contest -/
theorem rachels_budget (sara_shoes : ℕ) (sara_dress : ℕ) : 
  sara_shoes = 50 → sara_dress = 200 → 2 * (sara_shoes + sara_dress) = 500 := by
  sorry

end NUMINAMATH_CALUDE_rachels_budget_l3594_359497


namespace NUMINAMATH_CALUDE_insurance_company_expenses_percentage_l3594_359489

/-- Proves that given the conditions from the problem, the expenses in 2006 were 55.2% of the revenue in 2006 -/
theorem insurance_company_expenses_percentage (revenue2005 expenses2005 : ℝ) 
  (h1 : revenue2005 > 0)
  (h2 : expenses2005 > 0)
  (h3 : revenue2005 > expenses2005)
  (h4 : (1.25 * revenue2005 - 1.15 * expenses2005) = 1.4 * (revenue2005 - expenses2005)) :
  (1.15 * expenses2005) / (1.25 * revenue2005) = 0.552 := by
sorry

end NUMINAMATH_CALUDE_insurance_company_expenses_percentage_l3594_359489


namespace NUMINAMATH_CALUDE_two_digit_number_with_divisibility_properties_l3594_359484

theorem two_digit_number_with_divisibility_properties : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ 
  (n + 3) % 3 = 0 ∧ 
  (n + 7) % 7 = 0 ∧ 
  (n - 4) % 4 = 0 ∧
  n = 84 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_with_divisibility_properties_l3594_359484


namespace NUMINAMATH_CALUDE_change_in_expression_l3594_359478

theorem change_in_expression (x a : ℝ) (k : ℝ) (h : k > 0) :
  let f := fun x => 3 * x^2 - k
  (f (x + a) - f x = 6 * a * x + 3 * a^2) ∧
  (f (x - a) - f x = -6 * a * x + 3 * a^2) :=
by sorry

end NUMINAMATH_CALUDE_change_in_expression_l3594_359478


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_squared_l3594_359479

theorem sum_of_reciprocals_squared (a b c d : ℝ) : 
  a = 2 * Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 35 →
  b = -2 * Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 35 →
  c = 2 * Real.sqrt 5 - Real.sqrt 7 + Real.sqrt 35 →
  d = -2 * Real.sqrt 5 - Real.sqrt 7 + Real.sqrt 35 →
  (1/a + 1/b + 1/c + 1/d)^2 = 560 / 155432121 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_squared_l3594_359479


namespace NUMINAMATH_CALUDE_inequality_proof_l3594_359416

theorem inequality_proof (k m n : ℕ+) (h1 : 1 < k) (h2 : k ≤ m) (h3 : m < n) :
  (1 + m.val : ℝ)^2 > (1 + n.val : ℝ)^m.val := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3594_359416


namespace NUMINAMATH_CALUDE_three_digit_integers_with_remainders_l3594_359427

theorem three_digit_integers_with_remainders : 
  ∃! (S : Finset ℕ), 
    (∀ n ∈ S, 100 ≤ n ∧ n < 1000 ∧ 
              n % 7 = 3 ∧ 
              n % 10 = 4 ∧ 
              n % 12 = 8) ∧
    S.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_integers_with_remainders_l3594_359427


namespace NUMINAMATH_CALUDE_michelle_gas_usage_l3594_359422

theorem michelle_gas_usage (initial_gas final_gas : ℚ) : 
  initial_gas = 1/2 → 
  final_gas = 1/6 → 
  initial_gas - final_gas = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_michelle_gas_usage_l3594_359422


namespace NUMINAMATH_CALUDE_largest_n_multiple_of_three_n_99998_is_solution_n_99998_is_largest_l3594_359491

theorem largest_n_multiple_of_three (n : ℕ) : 
  n < 100000 → 
  (∃ k : ℤ, (n - 3)^5 - n^2 + 10*n - 30 = 3*k) → 
  n ≤ 99998 :=
sorry

theorem n_99998_is_solution : 
  ∃ k : ℤ, (99998 - 3)^5 - 99998^2 + 10*99998 - 30 = 3*k :=
sorry

theorem n_99998_is_largest : 
  ¬∃ n : ℕ, n > 99998 ∧ n < 100000 ∧ 
  (∃ k : ℤ, (n - 3)^5 - n^2 + 10*n - 30 = 3*k) :=
sorry

end NUMINAMATH_CALUDE_largest_n_multiple_of_three_n_99998_is_solution_n_99998_is_largest_l3594_359491


namespace NUMINAMATH_CALUDE_shekar_biology_score_l3594_359485

/-- Given a student's scores in four subjects and their average score for five subjects,
    calculate the score in the fifth subject. -/
def calculate_fifth_subject_score (math science social_studies english average : ℕ) : ℕ :=
  5 * average - (math + science + social_studies + english)

/-- Theorem stating that Shekar's biology score is 95 given his other scores and average -/
theorem shekar_biology_score :
  let math := 76
  let science := 65
  let social_studies := 82
  let english := 67
  let average := 77
  calculate_fifth_subject_score math science social_studies english average = 95 := by
  sorry

#eval calculate_fifth_subject_score 76 65 82 67 77

end NUMINAMATH_CALUDE_shekar_biology_score_l3594_359485


namespace NUMINAMATH_CALUDE_count_pairs_satisfying_equation_l3594_359486

/-- The number of pairs of positive integers (a, b) satisfying the given equation and condition -/
def solution_count : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ =>
    let a := p.1
    let b := p.2
    a > 0 ∧ b > 0 ∧
    (a + 2 / b) / (1 / a + 2 * b) = 17 ∧
    a + b ≤ 150)
  (Finset.product (Finset.range 151) (Finset.range 151))).card

/-- The theorem stating that there are exactly 8 pairs satisfying the conditions -/
theorem count_pairs_satisfying_equation : solution_count = 8 := by
  sorry

end NUMINAMATH_CALUDE_count_pairs_satisfying_equation_l3594_359486


namespace NUMINAMATH_CALUDE_mod_multiplication_equivalence_l3594_359475

theorem mod_multiplication_equivalence : 98 * 202 ≡ 71 [ZMOD 75] := by sorry

end NUMINAMATH_CALUDE_mod_multiplication_equivalence_l3594_359475


namespace NUMINAMATH_CALUDE_complement_A_complement_A_intersect_B_l3594_359452

-- Define the universal set U
def U : Set ℝ := {x | x ≤ 4}

-- Define set A
def A : Set ℝ := {x | 2 * x + 4 < 0}

-- Define set B
def B : Set ℝ := {x | x^2 + 2*x - 3 ≤ 0}

-- Theorem for the complement of A with respect to U
theorem complement_A : (U \ A) = {x : ℝ | -2 ≤ x ∧ x ≤ 4} := by sorry

-- Theorem for the complement of (A ∩ B) with respect to U
theorem complement_A_intersect_B : (U \ (A ∩ B)) = {x : ℝ | x < -3 ∨ (-2 ≤ x ∧ x ≤ 4)} := by sorry

end NUMINAMATH_CALUDE_complement_A_complement_A_intersect_B_l3594_359452


namespace NUMINAMATH_CALUDE_max_gcd_13n_plus_4_8n_plus_3_l3594_359425

theorem max_gcd_13n_plus_4_8n_plus_3 :
  (∃ (max : ℕ), 
    (∀ (n : ℕ), n > 0 → Nat.gcd (13*n + 4) (8*n + 3) ≤ max) ∧ 
    (∃ (n : ℕ), n > 0 ∧ Nat.gcd (13*n + 4) (8*n + 3) = max)) ∧
  (∀ (m : ℕ), 
    (∀ (n : ℕ), n > 0 → Nat.gcd (13*n + 4) (8*n + 3) ≤ m) →
    (∃ (n : ℕ), n > 0 ∧ Nat.gcd (13*n + 4) (8*n + 3) = m) →
    m ≤ 9) :=
by sorry

end NUMINAMATH_CALUDE_max_gcd_13n_plus_4_8n_plus_3_l3594_359425


namespace NUMINAMATH_CALUDE_smallest_sum_sequence_l3594_359426

/-- Given positive integers A, B, C, D satisfying the conditions, 
    the smallest possible sum A + B + C + D is 97 -/
theorem smallest_sum_sequence (A B C D : ℕ+) : 
  (∃ (d : ℤ), (A : ℤ) + d = B ∧ (B : ℤ) + d = C) →  -- arithmetic sequence
  (C : ℚ) / B = D / C →  -- geometric sequence
  (C : ℚ) / B = 7 / 4 →
  (∀ (A' B' C' D' : ℕ+), 
    (∃ (d' : ℤ), (A' : ℤ) + d' = B' ∧ (B' : ℤ) + d' = C') →
    (C' : ℚ) / B' = D' / C' →
    (C' : ℚ) / B' = 7 / 4 →
    (A : ℕ) + B + C + D ≤ (A' : ℕ) + B' + C' + D') →
  (A : ℕ) + B + C + D = 97 := by
sorry

end NUMINAMATH_CALUDE_smallest_sum_sequence_l3594_359426


namespace NUMINAMATH_CALUDE_river_road_bus_car_ratio_l3594_359430

/-- The ratio of buses to cars on River Road -/
def busCarRatio (numBuses : ℕ) (numCars : ℕ) : ℚ :=
  numBuses / numCars

theorem river_road_bus_car_ratio : 
  let numCars : ℕ := 60
  let numBuses : ℕ := numCars - 40
  busCarRatio numBuses numCars = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_river_road_bus_car_ratio_l3594_359430


namespace NUMINAMATH_CALUDE_interval_cardinality_equal_l3594_359498

/-- Two sets are equinumerous if there exists a bijection between them -/
def Equinumerous (α β : Type*) : Prop :=
  ∃ f : α → β, Function.Bijective f

theorem interval_cardinality_equal (a b : ℝ) (h : a < b) :
  Equinumerous (Set.Icc a b) (Set.Ioo a b) ∧
  Equinumerous (Set.Icc a b) (Set.Ico a b) ∧
  Equinumerous (Set.Icc a b) (Set.Ioc a b) :=
sorry

end NUMINAMATH_CALUDE_interval_cardinality_equal_l3594_359498


namespace NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l3594_359441

theorem sufficient_condition_for_inequality (a : ℝ) (h : a > 4) :
  ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≤ 0 := by
sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l3594_359441


namespace NUMINAMATH_CALUDE_abs_neg_one_ninth_l3594_359406

theorem abs_neg_one_ninth : |(-1 : ℚ) / 9| = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_one_ninth_l3594_359406


namespace NUMINAMATH_CALUDE_edwards_initial_money_l3594_359450

theorem edwards_initial_money (initial_amount : ℝ) : 
  initial_amount > 0 →
  (initial_amount * 0.6 * 0.75 * 1.2) = 28 →
  initial_amount = 77.78 := by
sorry

end NUMINAMATH_CALUDE_edwards_initial_money_l3594_359450


namespace NUMINAMATH_CALUDE_cheryl_material_used_l3594_359465

-- Define the amounts of materials
def material1 : ℚ := 2 / 9
def material2 : ℚ := 1 / 8
def leftover : ℚ := 4 / 18

-- Define the total amount bought
def total_bought : ℚ := material1 + material2

-- Define the theorem
theorem cheryl_material_used :
  total_bought - leftover = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_cheryl_material_used_l3594_359465


namespace NUMINAMATH_CALUDE_probability_of_red_ball_l3594_359424

theorem probability_of_red_ball (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ) :
  total_balls = red_balls + white_balls →
  red_balls = 2 →
  white_balls = 5 →
  (red_balls : ℚ) / total_balls = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_red_ball_l3594_359424


namespace NUMINAMATH_CALUDE_popcorn_profit_l3594_359432

def buying_price : ℝ := 4
def selling_price : ℝ := 8
def bags_sold : ℕ := 30

def profit_per_bag : ℝ := selling_price - buying_price
def total_profit : ℝ := bags_sold * profit_per_bag

theorem popcorn_profit : total_profit = 120 := by
  sorry

end NUMINAMATH_CALUDE_popcorn_profit_l3594_359432


namespace NUMINAMATH_CALUDE_coat_price_calculation_l3594_359442

/-- Calculates the final price of a coat after discounts, coupons, rebates, and tax -/
def finalPrice (initialPrice : ℝ) (discountRate : ℝ) (couponValue : ℝ) (rebateValue : ℝ) (taxRate : ℝ) : ℝ :=
  let discountedPrice := initialPrice * (1 - discountRate)
  let afterCoupon := discountedPrice - couponValue
  let afterRebate := afterCoupon - rebateValue
  afterRebate * (1 + taxRate)

/-- Theorem stating that the final price of the coat is $72.45 -/
theorem coat_price_calculation :
  finalPrice 120 0.30 10 5 0.05 = 72.45 := by
  sorry

#eval finalPrice 120 0.30 10 5 0.05

end NUMINAMATH_CALUDE_coat_price_calculation_l3594_359442


namespace NUMINAMATH_CALUDE_corina_calculation_l3594_359402

theorem corina_calculation (P Q : ℤ) 
  (h1 : P + Q = 16) 
  (h2 : P - Q = 4) : 
  P = 10 := by
sorry

end NUMINAMATH_CALUDE_corina_calculation_l3594_359402


namespace NUMINAMATH_CALUDE_sum_of_squares_with_given_means_l3594_359461

theorem sum_of_squares_with_given_means (a b : ℝ) :
  (a + b) / 2 = 8 → Real.sqrt (a * b) = 2 * Real.sqrt 5 → a^2 + b^2 = 216 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_with_given_means_l3594_359461


namespace NUMINAMATH_CALUDE_parallel_lines_m_equal_intercepts_equal_intercept_equations_l3594_359453

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 2 * x - y - 3 = 0
def l₂ (m x y : ℝ) : Prop := x - m * y + 1 - 3 * m = 0

-- Part 1: Parallel lines
theorem parallel_lines_m (m : ℝ) : 
  (∀ x y : ℝ, l₁ x y ↔ l₂ m x y) → m = 1/2 :=
sorry

-- Part 2: Equal intercepts
theorem equal_intercepts :
  ∃ m : ℝ, m ≠ 0 ∧ 
  ((∃ y : ℝ, l₂ m 0 y) ∧ (∃ x : ℝ, l₂ m x 0)) ∧
  (∀ y : ℝ, l₂ m 0 y → y = 3 * m - 1) ∧
  (∀ x : ℝ, l₂ m x 0 → x = 3 * m - 1) →
  (m = -1 ∨ m = 1/3) :=
sorry

-- Final equations for l₂ with equal intercepts
theorem equal_intercept_equations (x y : ℝ) :
  (x + y + 4 = 0 ∨ 3 * x - y = 0) ↔
  (l₂ (-1) x y ∨ l₂ (1/3) x y) :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_m_equal_intercepts_equal_intercept_equations_l3594_359453


namespace NUMINAMATH_CALUDE_smallest_prime_twelve_less_than_square_l3594_359466

theorem smallest_prime_twelve_less_than_square : ∃ n : ℕ, 
  (n > 0) ∧ 
  (Nat.Prime n) ∧ 
  (∃ m : ℕ, n = m^2 - 12) ∧
  (∀ k : ℕ, k > 0 → Nat.Prime k → (∃ l : ℕ, k = l^2 - 12) → k ≥ n) ∧
  n = 13 := by
sorry

end NUMINAMATH_CALUDE_smallest_prime_twelve_less_than_square_l3594_359466


namespace NUMINAMATH_CALUDE_oscar_review_questions_l3594_359463

/-- The number of questions Professor Oscar must review -/
def total_questions (num_classes : ℕ) (students_per_class : ℕ) (questions_per_exam : ℕ) : ℕ :=
  num_classes * students_per_class * questions_per_exam

/-- Proof that Professor Oscar must review 1750 questions -/
theorem oscar_review_questions :
  total_questions 5 35 10 = 1750 := by
  sorry

end NUMINAMATH_CALUDE_oscar_review_questions_l3594_359463


namespace NUMINAMATH_CALUDE_greatest_integer_b_no_real_roots_l3594_359407

theorem greatest_integer_b_no_real_roots : 
  ∀ b : ℤ, (∀ x : ℝ, x^2 + b*x + 10 ≠ 0) → b ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_b_no_real_roots_l3594_359407


namespace NUMINAMATH_CALUDE_monotone_increasing_condition_l3594_359419

/-- The function f(x) = (ax - 1)e^x is monotonically increasing on [0,1] if and only if a ≥ 1 -/
theorem monotone_increasing_condition (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, Monotone (fun x => (a * x - 1) * Real.exp x)) ↔ a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_monotone_increasing_condition_l3594_359419


namespace NUMINAMATH_CALUDE_stadium_entry_count_l3594_359418

def basket_capacity : ℕ := 4634
def placards_per_person : ℕ := 2

theorem stadium_entry_count :
  let total_placards : ℕ := basket_capacity
  let people_entered : ℕ := total_placards / placards_per_person
  people_entered = 2317 := by sorry

end NUMINAMATH_CALUDE_stadium_entry_count_l3594_359418


namespace NUMINAMATH_CALUDE_birthday_pigeonhole_l3594_359470

theorem birthday_pigeonhole (n m : ℕ) (h1 : n = 39) (h2 : m = 12) :
  ∃ k : ℕ, k ≤ m ∧ 4 ≤ (n / m + (if n % m = 0 then 0 else 1)) := by
  sorry

end NUMINAMATH_CALUDE_birthday_pigeonhole_l3594_359470


namespace NUMINAMATH_CALUDE_rectangle_height_double_area_square_side_double_area_cube_side_double_volume_rectangle_half_width_triple_height_rectangle_double_length_triple_width_geometric_transformations_l3594_359420

-- Define geometric shapes
def Rectangle (w h : ℝ) := w * h
def Square (s : ℝ) := s * s
def Cube (s : ℝ) := s * s * s

-- Theorem for statement (A)
theorem rectangle_height_double_area (w h : ℝ) :
  Rectangle w (2 * h) = 2 * Rectangle w h := by sorry

-- Theorem for statement (B)
theorem square_side_double_area (s : ℝ) :
  Square (2 * s) = 4 * Square s := by sorry

-- Theorem for statement (C)
theorem cube_side_double_volume (s : ℝ) :
  Cube (2 * s) = 8 * Cube s := by sorry

-- Theorem for statement (D)
theorem rectangle_half_width_triple_height (w h : ℝ) :
  Rectangle (w / 2) (3 * h) = (3 / 2) * Rectangle w h := by sorry

-- Theorem for statement (E)
theorem rectangle_double_length_triple_width (l w : ℝ) :
  Rectangle (2 * l) (3 * w) = 6 * Rectangle l w := by sorry

-- Main theorem proving (A) is false and others are true
theorem geometric_transformations :
  (∃ w h : ℝ, Rectangle w (2 * h) ≠ 3 * Rectangle w h) ∧
  (∀ s : ℝ, Square (2 * s) = 4 * Square s) ∧
  (∀ s : ℝ, Cube (2 * s) = 8 * Cube s) ∧
  (∀ w h : ℝ, Rectangle (w / 2) (3 * h) = (3 / 2) * Rectangle w h) ∧
  (∀ l w : ℝ, Rectangle (2 * l) (3 * w) = 6 * Rectangle l w) := by sorry

end NUMINAMATH_CALUDE_rectangle_height_double_area_square_side_double_area_cube_side_double_volume_rectangle_half_width_triple_height_rectangle_double_length_triple_width_geometric_transformations_l3594_359420


namespace NUMINAMATH_CALUDE_population_increase_rate_l3594_359417

def birth_rate : ℚ := 32 / 1000
def death_rate : ℚ := 11 / 1000

theorem population_increase_rate : 
  (birth_rate - death_rate) * 100 = 2.1 := by sorry

end NUMINAMATH_CALUDE_population_increase_rate_l3594_359417


namespace NUMINAMATH_CALUDE_solve_equation_l3594_359472

theorem solve_equation (x : ℚ) : (4 / 7) * (1 / 5) * x = 12 → x = 105 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3594_359472


namespace NUMINAMATH_CALUDE_probability_5_heads_in_7_flips_l3594_359476

def fair_coin_probability (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) / (2 ^ n : ℚ)

theorem probability_5_heads_in_7_flips :
  fair_coin_probability 7 5 = 21 / 128 := by
  sorry

end NUMINAMATH_CALUDE_probability_5_heads_in_7_flips_l3594_359476


namespace NUMINAMATH_CALUDE_composite_sum_of_squares_l3594_359460

theorem composite_sum_of_squares (a b : ℤ) : 
  (∃ x y : ℤ, x^2 + a*x + 1 = b ∧ x ≠ y) → 
  b ≠ 1 → 
  ∃ m n : ℤ, m > 1 ∧ n > 1 ∧ m * n = a^2 + b^2 := by
  sorry

end NUMINAMATH_CALUDE_composite_sum_of_squares_l3594_359460


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l3594_359490

theorem quadratic_equal_roots (k : ℝ) : 
  (∃ x : ℝ, x^2 - 2*k*x + 6 = 0 ∧ 
   ∀ y : ℝ, y^2 - 2*k*y + 6 = 0 → y = x) ↔ 
  k = Real.sqrt 6 ∨ k = -Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l3594_359490


namespace NUMINAMATH_CALUDE_euler_dedekind_divisibility_l3594_359457

-- Define the Euler totient function
def Φ : ℕ → ℕ := sorry

-- Define the Dedekind's totient function
def Ψ : ℕ → ℕ := sorry

-- Define the set of numbers of the form 2^n₁, 2^n₁3^n₂, or 2^n₁5^n₂
def S : Set ℕ :=
  {n : ℕ | n = 1 ∨ (∃ n₁ n₂ : ℕ, n = 2^n₁ ∨ n = 2^n₁ * 3^n₂ ∨ n = 2^n₁ * 5^n₂)}

-- State the theorem
theorem euler_dedekind_divisibility (n : ℕ) :
  (n ∈ S) ↔ (Φ n ∣ n + Ψ n) := by sorry

end NUMINAMATH_CALUDE_euler_dedekind_divisibility_l3594_359457


namespace NUMINAMATH_CALUDE_trigonometric_values_l3594_359471

theorem trigonometric_values (α : Real) 
  (h1 : α ∈ Set.Ioo (π/3) (π/2))
  (h2 : Real.cos (π/6 + α) * Real.cos (π/3 - α) = -1/4) : 
  Real.sin (2*α) = Real.sqrt 3 / 2 ∧ 
  Real.tan α - 1 / Real.tan α = 2 * Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_values_l3594_359471


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3594_359454

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product
  (a : ℕ → ℝ)
  (h_geo : geometric_sequence a)
  (h_a3 : a 3 = 2) :
  a 1 * a 2 * a 3 * a 4 * a 5 = 32 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l3594_359454


namespace NUMINAMATH_CALUDE_coin_toss_probability_l3594_359467

/-- The probability of getting exactly k heads in n tosses of a coin with probability r of landing heads -/
def binomial_probability (n k : ℕ) (r : ℚ) : ℚ :=
  (n.choose k : ℚ) * r^k * (1 - r)^(n - k)

/-- The main theorem -/
theorem coin_toss_probability : ∀ r : ℚ,
  0 < r →
  r < 1 →
  binomial_probability 5 1 r = binomial_probability 5 2 r →
  binomial_probability 5 3 r = 40 / 243 := by
  sorry

end NUMINAMATH_CALUDE_coin_toss_probability_l3594_359467


namespace NUMINAMATH_CALUDE_complex_magnitude_equality_l3594_359401

theorem complex_magnitude_equality (n : ℝ) (hn : n > 0) :
  Complex.abs (4 + 2 * n * Complex.I) = 4 * Real.sqrt 5 ↔ n = 4 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_equality_l3594_359401


namespace NUMINAMATH_CALUDE_quartic_sum_theorem_l3594_359456

/-- A quartic polynomial with specific properties -/
structure QuarticPolynomial (m : ℝ) where
  Q : ℝ → ℝ
  is_quartic : ∃ (a b c d : ℝ), ∀ x, Q x = a * x^4 + b * x^3 + c * x^2 + d * x + m
  at_zero : Q 0 = m
  at_one : Q 1 = 2 * m
  at_neg_one : Q (-1) = 4 * m
  at_two : Q 2 = 5 * m

/-- Theorem: For a quartic polynomial Q satisfying specific conditions, Q(2) + Q(-2) = 66m -/
theorem quartic_sum_theorem (m : ℝ) (qp : QuarticPolynomial m) : qp.Q 2 + qp.Q (-2) = 66 * m := by
  sorry

end NUMINAMATH_CALUDE_quartic_sum_theorem_l3594_359456


namespace NUMINAMATH_CALUDE_f_passes_through_six_zero_f_vertex_at_four_neg_eight_l3594_359440

/-- A quadratic function passing through (6, 0) with vertex at (4, -8) -/
def f (x : ℝ) : ℝ := 2 * (x - 4)^2 - 8

/-- The function f passes through the point (6, 0) -/
theorem f_passes_through_six_zero : f 6 = 0 := by sorry

/-- The vertex of f is at (4, -8) -/
theorem f_vertex_at_four_neg_eight :
  (∃ (a : ℝ), ∀ (x : ℝ), f x = a * (x - 4)^2 - 8) ∧
  (∀ (x : ℝ), f x ≥ f 4) := by sorry

end NUMINAMATH_CALUDE_f_passes_through_six_zero_f_vertex_at_four_neg_eight_l3594_359440


namespace NUMINAMATH_CALUDE_michael_cleaning_count_l3594_359474

/-- The number of times Michael takes a bath per week -/
def baths_per_week : ℕ := 2

/-- The number of times Michael takes a shower per week -/
def showers_per_week : ℕ := 1

/-- The number of weeks in the given time period -/
def weeks : ℕ := 52

/-- The total number of times Michael cleans himself in the given time period -/
def total_cleanings : ℕ := weeks * (baths_per_week + showers_per_week)

theorem michael_cleaning_count : total_cleanings = 156 := by
  sorry

end NUMINAMATH_CALUDE_michael_cleaning_count_l3594_359474


namespace NUMINAMATH_CALUDE_max_value_theorem_l3594_359400

/-- Represents the crop types --/
inductive Crop
| Melon
| Fruit
| Vegetable

/-- Represents the problem parameters --/
structure ProblemParams where
  totalLaborers : ℕ
  totalLand : ℕ
  laborRequirement : Crop → ℚ
  valuePerAcre : Crop → ℚ

/-- Represents the allocation of land to each crop --/
structure Allocation where
  melon : ℕ
  fruit : ℕ
  vegetable : ℕ

/-- The main theorem statement --/
theorem max_value_theorem (params : ProblemParams) 
  (h1 : params.totalLaborers = 20)
  (h2 : params.totalLand = 50)
  (h3 : params.laborRequirement Crop.Melon = 1/2)
  (h4 : params.laborRequirement Crop.Fruit = 1/3)
  (h5 : params.laborRequirement Crop.Vegetable = 1/4)
  (h6 : params.valuePerAcre Crop.Melon = 6/10)
  (h7 : params.valuePerAcre Crop.Fruit = 5/10)
  (h8 : params.valuePerAcre Crop.Vegetable = 3/10) :
  ∃ (alloc : Allocation),
    (alloc.melon + alloc.fruit + alloc.vegetable = params.totalLand) ∧
    (alloc.melon * params.laborRequirement Crop.Melon + 
     alloc.fruit * params.laborRequirement Crop.Fruit + 
     alloc.vegetable * params.laborRequirement Crop.Vegetable = params.totalLaborers) ∧
    (∀ (other : Allocation),
      (other.melon + other.fruit + other.vegetable = params.totalLand) →
      (other.melon * params.laborRequirement Crop.Melon + 
       other.fruit * params.laborRequirement Crop.Fruit + 
       other.vegetable * params.laborRequirement Crop.Vegetable = params.totalLaborers) →
      (alloc.melon * params.valuePerAcre Crop.Melon + 
       alloc.fruit * params.valuePerAcre Crop.Fruit + 
       alloc.vegetable * params.valuePerAcre Crop.Vegetable ≥
       other.melon * params.valuePerAcre Crop.Melon + 
       other.fruit * params.valuePerAcre Crop.Fruit + 
       other.vegetable * params.valuePerAcre Crop.Vegetable)) ∧
    (alloc.melon * params.valuePerAcre Crop.Melon + 
     alloc.fruit * params.valuePerAcre Crop.Fruit + 
     alloc.vegetable * params.valuePerAcre Crop.Vegetable = 27) :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3594_359400


namespace NUMINAMATH_CALUDE_min_xy_value_l3594_359409

theorem min_xy_value (x y : ℕ+) (h : (1 : ℚ) / x + (1 : ℚ) / (3 * y) = (1 : ℚ) / 8) : 
  (x * y : ℕ) ≥ 96 := by
sorry

end NUMINAMATH_CALUDE_min_xy_value_l3594_359409


namespace NUMINAMATH_CALUDE_parallel_segment_length_l3594_359435

/-- Given a triangle ABC with side AC = 8 cm, if two segments parallel to AC divide the triangle
    into three equal areas, then the length of the parallel segment closest to AC is 8√3/3. -/
theorem parallel_segment_length (A B C : ℝ × ℝ) (a b : ℝ) :
  let triangle_area := (4 : ℝ) * b
  let segment_de_length := (8 : ℝ) * Real.sqrt 6 / 3
  let segment_fg_length := (8 : ℝ) * Real.sqrt 3 / 3
  A = (0, 0) →
  B = (a, b) →
  C = (8, 0) →
  triangle_area / 3 = b * (a * Real.sqrt (8 / 3))^2 / (2 * a) →
  segment_de_length > segment_fg_length →
  segment_fg_length = 8 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_segment_length_l3594_359435


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l3594_359499

-- Define the operation
def det (a b c d : ℂ) : ℂ := a * d - b * c

-- Define the imaginary unit
def i : ℂ := Complex.I

-- Define z using the operation
def z : ℂ := det 1 2 i (i^4)

-- Define the fourth quadrant
def fourth_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

-- Theorem statement
theorem z_in_fourth_quadrant : fourth_quadrant z := by
  sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l3594_359499


namespace NUMINAMATH_CALUDE_simplify_square_roots_l3594_359488

theorem simplify_square_roots : 
  (Real.sqrt 450 / Real.sqrt 200) + (Real.sqrt 98 / Real.sqrt 49) = 1457 / 500 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l3594_359488


namespace NUMINAMATH_CALUDE_constant_sum_and_square_sum_implies_constant_S_l3594_359439

theorem constant_sum_and_square_sum_implies_constant_S 
  (a b c d : ℝ) 
  (h1 : a + b + c + d = 10) 
  (h2 : a^2 + b^2 + c^2 + d^2 = 30) : 
  3 * (a^3 + b^3 + c^3 + d^3) - 3 * (a^2 + b^2 + c^2 + d^2) = 7.5 := by
sorry

end NUMINAMATH_CALUDE_constant_sum_and_square_sum_implies_constant_S_l3594_359439


namespace NUMINAMATH_CALUDE_box_volume_calculation_l3594_359451

/-- Calculates the total volume occupied by boxes given their dimensions, cost per box, and total monthly payment -/
theorem box_volume_calculation (length width height cost_per_box total_payment : ℝ) :
  length = 15 ∧ 
  width = 12 ∧ 
  height = 10 ∧ 
  cost_per_box = 0.8 ∧ 
  total_payment = 480 →
  (total_payment / cost_per_box) * (length * width * height) = 1080000 := by
  sorry

#check box_volume_calculation

end NUMINAMATH_CALUDE_box_volume_calculation_l3594_359451


namespace NUMINAMATH_CALUDE_complement_of_union_l3594_359403

def U : Set Int := {-2, -1, 0, 1, 2, 3}
def A : Set Int := {-1, 0, 1}
def B : Set Int := {1, 2}

theorem complement_of_union : (U \ (A ∪ B)) = {-2, 3} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_l3594_359403


namespace NUMINAMATH_CALUDE_min_height_of_box_l3594_359462

/-- Represents a rectangular box with square bases -/
structure Box where
  base : ℝ  -- side length of the square base
  height : ℝ -- height of the box
  h_positive : 0 < height
  b_positive : 0 < base

/-- The surface area of a box -/
def surface_area (box : Box) : ℝ :=
  2 * box.base^2 + 4 * box.base * box.height

/-- The constraint that the height is 5 units greater than the base -/
def height_constraint (box : Box) : Prop :=
  box.height = box.base + 5

theorem min_height_of_box (box : Box) 
  (h_constraint : height_constraint box)
  (h_surface_area : surface_area box ≥ 150) :
  box.height ≥ 10 ∧ ∃ (b : Box), height_constraint b ∧ surface_area b ≥ 150 ∧ b.height = 10 :=
sorry

end NUMINAMATH_CALUDE_min_height_of_box_l3594_359462


namespace NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l3594_359411

theorem min_value_of_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geometric_mean : Real.sqrt 3 = Real.sqrt (3^a * 3^b)) : 
  (∀ x y, x > 0 → y > 0 → Real.sqrt 3 = Real.sqrt (3^x * 3^y) → 1/x + 1/y ≥ 1/a + 1/b) → 
  1/a + 1/b = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l3594_359411


namespace NUMINAMATH_CALUDE_sphere_radius_l3594_359455

/-- Given two spheres A and B, where A has radius 40 cm and the ratio of their surface areas is 16,
    prove that the radius of sphere B is 20 cm. -/
theorem sphere_radius (r : ℝ) : 
  let surface_area (radius : ℝ) := 4 * Real.pi * radius^2
  surface_area 40 / surface_area r = 16 → r = 20 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_l3594_359455


namespace NUMINAMATH_CALUDE_problem_solution_l3594_359414

def f (a : ℝ) : ℝ → ℝ := fun x ↦ |x - a|

theorem problem_solution :
  (∃ a : ℝ, (∀ x : ℝ, f a x ≤ 2 ↔ 1 ≤ x ∧ x ≤ 5) ∧
   (∀ m : ℝ, (∀ x : ℝ, f 3 (2*x) + f 3 (x+2) ≥ m) ↔ m ≤ 1/2)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3594_359414


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l3594_359468

theorem min_value_trig_expression (α β : ℝ) :
  (3 * Real.cos α + 4 * Real.sin β - 10)^2 + (3 * Real.sin α + 4 * Real.cos β - 12)^2 ≥ (Real.sqrt 244 - 7)^2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l3594_359468


namespace NUMINAMATH_CALUDE_original_number_exists_l3594_359469

theorem original_number_exists : ∃ x : ℝ, 4 * ((x^3 / 5)^2 + 15) = 224 := by
  sorry

end NUMINAMATH_CALUDE_original_number_exists_l3594_359469


namespace NUMINAMATH_CALUDE_candy_bag_problem_l3594_359408

theorem candy_bag_problem (n : ℕ) (r : ℕ) : 
  n > 0 →  -- Ensure the bag is not empty
  r > 0 →  -- Ensure there are red candies
  r ≤ n →  -- Ensure the number of red candies doesn't exceed the total
  (r : ℚ) / n = 5 / 6 →  -- Probability of choosing a red candy
  n = 6 :=
by sorry

end NUMINAMATH_CALUDE_candy_bag_problem_l3594_359408
