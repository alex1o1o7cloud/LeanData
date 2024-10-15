import Mathlib

namespace NUMINAMATH_CALUDE_classroom_children_count_l284_28436

theorem classroom_children_count :
  ∀ (total_children : ℕ),
  (total_children : ℚ) / 3 = total_children - 30 →
  30 ≤ total_children →
  total_children = 45 := by
sorry

end NUMINAMATH_CALUDE_classroom_children_count_l284_28436


namespace NUMINAMATH_CALUDE_room_length_is_25_l284_28419

/-- Represents the dimensions and whitewashing details of a room --/
structure Room where
  length : ℝ
  width : ℝ
  height : ℝ
  whitewashCost : ℝ
  doorArea : ℝ
  windowArea : ℝ
  totalCost : ℝ

/-- Calculates the whitewashable area of the room --/
def whitewashableArea (r : Room) : ℝ :=
  2 * (r.length * r.height + r.width * r.height) - r.doorArea - 3 * r.windowArea

/-- Theorem stating that the room length is 25 feet given the specified conditions --/
theorem room_length_is_25 (r : Room) 
    (h1 : r.width = 15)
    (h2 : r.height = 12)
    (h3 : r.whitewashCost = 6)
    (h4 : r.doorArea = 18)
    (h5 : r.windowArea = 12)
    (h6 : r.totalCost = 5436)
    (h7 : r.totalCost = r.whitewashCost * whitewashableArea r) :
    r.length = 25 := by
  sorry


end NUMINAMATH_CALUDE_room_length_is_25_l284_28419


namespace NUMINAMATH_CALUDE_marcus_initial_cards_l284_28426

/-- The number of baseball cards Carter gave to Marcus -/
def cards_from_carter : ℝ := 58.0

/-- The total number of baseball cards Marcus has after receiving cards from Carter -/
def total_cards : ℕ := 268

/-- The initial number of baseball cards Marcus had -/
def initial_cards : ℕ := 210

theorem marcus_initial_cards : 
  (total_cards : ℝ) - cards_from_carter = initial_cards := by sorry

end NUMINAMATH_CALUDE_marcus_initial_cards_l284_28426


namespace NUMINAMATH_CALUDE_sqrt_product_sqrt_l284_28481

theorem sqrt_product_sqrt : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_sqrt_l284_28481


namespace NUMINAMATH_CALUDE_last_two_digits_of_sum_factorials_l284_28486

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def last_two_digits (n : ℕ) : ℕ := n % 100

def sum_factorials : ℕ := List.range 31 |>.map (fun i => factorial (6 + 3 * i)) |>.sum

theorem last_two_digits_of_sum_factorials :
  last_two_digits sum_factorials = 20 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_of_sum_factorials_l284_28486


namespace NUMINAMATH_CALUDE_linear_systems_solutions_l284_28410

/-- Prove that the solutions to the given systems of linear equations are correct -/
theorem linear_systems_solutions :
  -- System 1
  (∃ x y : ℝ, 3 * x - 2 * y = -1 ∧ 2 * x + 3 * y = 8 ∧ x = 1 ∧ y = 2) ∧
  -- System 2
  (∃ x y : ℝ, 2 * x + y = 1 ∧ 2 * x - y = 7 ∧ x = 2 ∧ y = -3) :=
by sorry

end NUMINAMATH_CALUDE_linear_systems_solutions_l284_28410


namespace NUMINAMATH_CALUDE_pencil_count_l284_28457

/-- Given an initial number of pencils and additional pencils added, 
    calculate the total number of pencils -/
def total_pencils (initial : ℕ) (added : ℕ) : ℕ := initial + added

/-- Theorem stating that 115 initial pencils plus 100 additional pencils equals 215 total pencils -/
theorem pencil_count : total_pencils 115 100 = 215 := by
  sorry

end NUMINAMATH_CALUDE_pencil_count_l284_28457


namespace NUMINAMATH_CALUDE_blackboard_operation_theorem_l284_28465

/-- Operation that replaces a number with two new numbers -/
def replace_operation (r : ℝ) : ℝ × ℝ :=
  let a := r
  let b := 2 * r
  (a, b)

/-- Applies the replace_operation n times to an initial set of numbers -/
def apply_operations (initial : Set ℝ) (n : ℕ) : Set ℝ :=
  sorry

theorem blackboard_operation_theorem (r : ℝ) (k : ℕ) (h_r : r > 0) :
  ∃ s ∈ apply_operations {r} (k^2 - 1), s ≤ k * r :=
sorry

end NUMINAMATH_CALUDE_blackboard_operation_theorem_l284_28465


namespace NUMINAMATH_CALUDE_inverse_of_B_cubed_l284_28456

open Matrix

theorem inverse_of_B_cubed (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B⁻¹ = ![![3, -1], ![0, 5]]) : 
  (B^3)⁻¹ = ![![27, -49], ![0, 125]] := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_B_cubed_l284_28456


namespace NUMINAMATH_CALUDE_vector_magnitude_proof_l284_28452

theorem vector_magnitude_proof (OA AB : ℂ) (h1 : OA = -2 + I) (h2 : AB = 3 + 2*I) :
  Complex.abs (OA + AB) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_proof_l284_28452


namespace NUMINAMATH_CALUDE_points_three_units_from_negative_one_l284_28482

theorem points_three_units_from_negative_one : 
  {x : ℝ | |x - (-1)| = 3} = {2, -4} := by sorry

end NUMINAMATH_CALUDE_points_three_units_from_negative_one_l284_28482


namespace NUMINAMATH_CALUDE_house_transaction_gain_l284_28468

/-- Calculates the net gain from selling a house at a profit and buying it back at a loss --/
def net_gain (initial_worth : ℝ) (sell_profit_percent : ℝ) (buy_loss_percent : ℝ) : ℝ :=
  let sell_price := initial_worth * (1 + sell_profit_percent)
  let buy_back_price := sell_price * (1 - buy_loss_percent)
  sell_price - buy_back_price

/-- Theorem stating that selling a $12,000 house at 20% profit and buying it back at 15% loss results in $2,160 gain --/
theorem house_transaction_gain :
  net_gain 12000 0.2 0.15 = 2160 := by
  sorry

end NUMINAMATH_CALUDE_house_transaction_gain_l284_28468


namespace NUMINAMATH_CALUDE_comic_pages_calculation_l284_28491

theorem comic_pages_calculation (total_pages : ℕ) (extra_pages : ℕ) : 
  total_pages = 220 → 
  extra_pages = 4 → 
  ∃ (first_issue : ℕ), 
    first_issue * 2 + (first_issue + extra_pages) = total_pages ∧ 
    first_issue = 72 := by
  sorry

end NUMINAMATH_CALUDE_comic_pages_calculation_l284_28491


namespace NUMINAMATH_CALUDE_triangle_side_length_l284_28418

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    if the area is √3, B = 60°, and a² + c² = 3ac, then b = 2√2. -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  (1/2 * a * c * Real.sin B = Real.sqrt 3) →
  (B = π/3) →
  (a^2 + c^2 = 3*a*c) →
  (b = 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l284_28418


namespace NUMINAMATH_CALUDE_rows_sum_equal_l284_28469

def first_row : List ℕ := [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 200]
def second_row : List ℕ := [1, 4, 7, 10, 13, 16, 19, 22, 25, 28]

theorem rows_sum_equal : 
  (first_row.sum = second_row.sum + 155) := by sorry

end NUMINAMATH_CALUDE_rows_sum_equal_l284_28469


namespace NUMINAMATH_CALUDE_log_equation_solution_l284_28459

theorem log_equation_solution (p q : ℝ) 
  (h1 : Real.log p + Real.log (q^2) = Real.log (p + 3*q^2))
  (h2 : p ≠ -3*q^2)
  (h3 : q ≠ 0)
  (h4 : q^2 ≠ 1) : 
  p = 3*q^2 / (q^2 - 1) := by
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l284_28459


namespace NUMINAMATH_CALUDE_system_has_no_solution_l284_28463

theorem system_has_no_solution :
  ¬ (∃ (x₁ x₂ x₃ x₄ : ℝ),
    (5 * x₁ + 12 * x₂ + 19 * x₃ + 25 * x₄ = 25) ∧
    (10 * x₁ + 22 * x₂ + 16 * x₃ + 39 * x₄ = 25) ∧
    (5 * x₁ + 12 * x₂ + 9 * x₃ + 25 * x₄ = 30) ∧
    (20 * x₁ + 46 * x₂ + 34 * x₃ + 89 * x₄ = 70)) :=
by
  sorry


end NUMINAMATH_CALUDE_system_has_no_solution_l284_28463


namespace NUMINAMATH_CALUDE_oatmeal_boxes_sold_problem_l284_28431

/-- The number of oatmeal biscuit boxes sold to the neighbor -/
def oatmeal_boxes_sold (total_boxes : ℕ) (lemon_boxes : ℕ) (chocolate_boxes : ℕ) (boxes_to_sell : ℕ) : ℕ :=
  total_boxes - lemon_boxes - chocolate_boxes - boxes_to_sell

theorem oatmeal_boxes_sold_problem (total_boxes : ℕ) (lemon_boxes : ℕ) (chocolate_boxes : ℕ) (boxes_to_sell : ℕ)
  (h1 : total_boxes = 33)
  (h2 : lemon_boxes = 12)
  (h3 : chocolate_boxes = 5)
  (h4 : boxes_to_sell = 12) :
  oatmeal_boxes_sold total_boxes lemon_boxes chocolate_boxes boxes_to_sell = 4 := by
  sorry

end NUMINAMATH_CALUDE_oatmeal_boxes_sold_problem_l284_28431


namespace NUMINAMATH_CALUDE_range_of_m_l284_28480

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the property of f being decreasing on [-2, 2]
def isDecreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x y, -2 ≤ x ∧ x < y ∧ y ≤ 2 → f y < f x

-- State the theorem
theorem range_of_m (f : ℝ → ℝ) (m : ℝ) 
  (h1 : isDecreasingOn f) 
  (h2 : f (m - 1) < f (-m)) : 
  1/2 < m ∧ m ≤ 2 := by sorry

end NUMINAMATH_CALUDE_range_of_m_l284_28480


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l284_28446

-- Define repeating decimals
def repeating_decimal_6 : ℚ := 2/3
def repeating_decimal_2 : ℚ := 2/9
def repeating_decimal_4 : ℚ := 4/9

-- Theorem statement
theorem repeating_decimal_sum :
  repeating_decimal_6 + repeating_decimal_2 - repeating_decimal_4 = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l284_28446


namespace NUMINAMATH_CALUDE_gcf_of_lcm_equals_15_l284_28445

-- Define GCF (Greatest Common Factor)
def GCF (a b : ℕ) : ℕ := Nat.gcd a b

-- Define LCM (Least Common Multiple)
def LCM (c d : ℕ) : ℕ := Nat.lcm c d

-- Theorem statement
theorem gcf_of_lcm_equals_15 : GCF (LCM 9 15) (LCM 10 21) = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_lcm_equals_15_l284_28445


namespace NUMINAMATH_CALUDE_equal_roots_condition_l284_28484

/-- For a quadratic equation of the form (2kx^2 + Bx + 2) = 0 to have equal roots, B must be equal to 4√k -/
theorem equal_roots_condition (k : ℝ) (B : ℝ) :
  (∀ x : ℝ, (2 * k * x^2 + B * x + 2 = 0) → (∃! r : ℝ, 2 * k * r^2 + B * r + 2 = 0)) ↔ B = 4 * Real.sqrt k := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_condition_l284_28484


namespace NUMINAMATH_CALUDE_starting_number_is_eight_l284_28428

def is_valid_start (n : ℕ) : Prop :=
  n ≤ 38 ∧ n % 4 = 0

def numbers_between (n : ℕ) : List ℕ :=
  (List.range ((38 - n) / 4 + 1)).map (fun i => n + 4 * i)

def average (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

theorem starting_number_is_eight (n : ℕ) (h1 : is_valid_start n) 
    (h2 : average (numbers_between n) = 22) : n = 8 := by
  sorry

end NUMINAMATH_CALUDE_starting_number_is_eight_l284_28428


namespace NUMINAMATH_CALUDE_chess_tournament_games_l284_28474

theorem chess_tournament_games (n : ℕ) (h : n = 50) : 
  (n * (n - 1)) / 2 = 1225 := by sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l284_28474


namespace NUMINAMATH_CALUDE_square_root_of_sixteen_l284_28420

theorem square_root_of_sixteen : 
  {x : ℝ | x^2 = 16} = {4, -4} := by sorry

end NUMINAMATH_CALUDE_square_root_of_sixteen_l284_28420


namespace NUMINAMATH_CALUDE_largest_sample_is_433_l284_28487

/-- Represents a systematic sampling scheme. -/
structure SystematicSampling where
  totalItems : ℕ
  sampleSize : ℕ
  knownSample : ℕ
  firstItem : ℕ
  interval : ℕ

/-- Calculates the largest sampled number in a systematic sampling scheme. -/
def largestSampledNumber (s : SystematicSampling) : ℕ :=
  ((s.firstItem - 1 + (s.sampleSize - 1) * s.interval) % s.totalItems) + 1

/-- Theorem stating that for the given systematic sampling scheme,
    the largest sampled number is 433. -/
theorem largest_sample_is_433 :
  ∃ s : SystematicSampling,
    s.totalItems = 360 ∧
    s.sampleSize = 30 ∧
    s.knownSample = 105 ∧
    s.firstItem = 97 ∧
    s.interval = 12 ∧
    largestSampledNumber s = 433 :=
  sorry

end NUMINAMATH_CALUDE_largest_sample_is_433_l284_28487


namespace NUMINAMATH_CALUDE_x_with_18_factors_l284_28423

theorem x_with_18_factors (x : ℕ) : 
  (∃ (factors : Finset ℕ), factors.card = 18 ∧ (∀ f ∈ factors, f ∣ x)) → 
  18 ∣ x → 
  20 ∣ x → 
  x = 180 := by
sorry

end NUMINAMATH_CALUDE_x_with_18_factors_l284_28423


namespace NUMINAMATH_CALUDE_prob_non_red_is_half_l284_28464

-- Define the die
def total_faces : ℕ := 10
def red_faces : ℕ := 5
def yellow_faces : ℕ := 3
def blue_faces : ℕ := 1
def green_faces : ℕ := 1

-- Define the probability of rolling a non-red face
def prob_non_red : ℚ := (yellow_faces + blue_faces + green_faces : ℚ) / total_faces

-- Theorem statement
theorem prob_non_red_is_half : prob_non_red = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_prob_non_red_is_half_l284_28464


namespace NUMINAMATH_CALUDE_min_value_of_d_l284_28499

/-- The function d(x, y) to be minimized -/
def d (x y : ℝ) : ℝ := x^2 + y^2 - 2*x - 4*y + 6

/-- Theorem stating that the minimum value of d(x, y) is 1 -/
theorem min_value_of_d :
  ∃ (min : ℝ), min = 1 ∧ ∀ (x y : ℝ), d x y ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_of_d_l284_28499


namespace NUMINAMATH_CALUDE_fraction_subtraction_l284_28488

theorem fraction_subtraction : (3/8 : ℚ) + (5/12 : ℚ) - (1/6 : ℚ) = (5/8 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l284_28488


namespace NUMINAMATH_CALUDE_frog_jump_distance_l284_28461

theorem frog_jump_distance (grasshopper_jump : ℕ) (total_jump : ℕ) (frog_jump : ℕ) : 
  grasshopper_jump = 31 → total_jump = 66 → frog_jump = total_jump - grasshopper_jump → frog_jump = 35 := by
  sorry

end NUMINAMATH_CALUDE_frog_jump_distance_l284_28461


namespace NUMINAMATH_CALUDE_martha_cards_total_l284_28433

/-- Given that Martha starts with 3 cards and receives 76 more cards,
    prove that she ends up with 79 cards in total. -/
theorem martha_cards_total : 
  let initial_cards : ℕ := 3
  let received_cards : ℕ := 76
  initial_cards + received_cards = 79 := by
  sorry

end NUMINAMATH_CALUDE_martha_cards_total_l284_28433


namespace NUMINAMATH_CALUDE_integral_minus_x_squared_plus_one_l284_28421

theorem integral_minus_x_squared_plus_one : ∫ (x : ℝ) in (0)..(1), -x^2 + 1 = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_integral_minus_x_squared_plus_one_l284_28421


namespace NUMINAMATH_CALUDE_sandys_shopping_money_l284_28443

theorem sandys_shopping_money (initial_amount : ℝ) : 
  (initial_amount * 0.7 = 217) → initial_amount = 310 := by
  sorry

end NUMINAMATH_CALUDE_sandys_shopping_money_l284_28443


namespace NUMINAMATH_CALUDE_optimal_price_reduction_l284_28409

/-- Represents the price reduction of thermal shirts -/
def price_reduction : ℝ := 20

/-- Initial average daily sales -/
def initial_sales : ℝ := 20

/-- Initial profit per piece -/
def initial_profit_per_piece : ℝ := 40

/-- Sales increase per dollar of price reduction -/
def sales_increase_rate : ℝ := 2

/-- Target daily profit -/
def target_profit : ℝ := 1200

/-- New daily sales after price reduction -/
def new_sales (x : ℝ) : ℝ := initial_sales + sales_increase_rate * x

/-- New profit per piece after price reduction -/
def new_profit_per_piece (x : ℝ) : ℝ := initial_profit_per_piece - x

/-- Daily profit function -/
def daily_profit (x : ℝ) : ℝ := new_sales x * new_profit_per_piece x

theorem optimal_price_reduction :
  daily_profit price_reduction = target_profit ∧
  ∀ y, y ≠ price_reduction → daily_profit y ≤ daily_profit price_reduction :=
by sorry

end NUMINAMATH_CALUDE_optimal_price_reduction_l284_28409


namespace NUMINAMATH_CALUDE_three_digit_powers_of_three_l284_28444

theorem three_digit_powers_of_three (n : ℕ) : 
  (100 ≤ 3^n ∧ 3^n ≤ 999) ↔ (n = 5 ∨ n = 6) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_powers_of_three_l284_28444


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l284_28453

theorem simplify_trig_expression :
  (Real.sin (58 * π / 180) - Real.sin (28 * π / 180) * Real.cos (30 * π / 180)) / Real.cos (28 * π / 180) = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l284_28453


namespace NUMINAMATH_CALUDE_monotonicity_condition_max_value_condition_l284_28473

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := (m - 3) * x^3 + 9 * x

-- Part 1: Monotonicity condition
theorem monotonicity_condition (m : ℝ) :
  (∀ x y : ℝ, x < y → f m x < f m y) ↔ m ≥ 3 :=
sorry

-- Part 2: Maximum value condition
theorem max_value_condition :
  ∃ m : ℝ, (∀ x ∈ Set.Icc 1 2, f m x ≤ 4) ∧
           (∃ x ∈ Set.Icc 1 2, f m x = 4) ∧
           m = -2 :=
sorry

end NUMINAMATH_CALUDE_monotonicity_condition_max_value_condition_l284_28473


namespace NUMINAMATH_CALUDE_two_queens_or_one_king_probability_l284_28485

-- Define a standard deck
def standard_deck : ℕ := 52

-- Define the number of queens in a deck
def num_queens : ℕ := 4

-- Define the number of kings in a deck
def num_kings : ℕ := 4

-- Define the number of cards drawn
def cards_drawn : ℕ := 3

-- Define the probability of the event
def event_probability : ℚ := 49 / 221

-- Theorem statement
theorem two_queens_or_one_king_probability :
  let p_two_queens := (num_queens / standard_deck) * ((num_queens - 1) / (standard_deck - 1))
  let p_no_kings := (standard_deck - num_kings) / standard_deck *
                    ((standard_deck - num_kings - 1) / (standard_deck - 1)) *
                    ((standard_deck - num_kings - 2) / (standard_deck - 2))
  let p_at_least_one_king := 1 - p_no_kings
  p_two_queens + p_at_least_one_king = event_probability :=
by sorry

end NUMINAMATH_CALUDE_two_queens_or_one_king_probability_l284_28485


namespace NUMINAMATH_CALUDE_one_eighth_percent_of_800_l284_28497

theorem one_eighth_percent_of_800 : (1 / 8 * (1 / 100) * 800 : ℚ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_one_eighth_percent_of_800_l284_28497


namespace NUMINAMATH_CALUDE_jerry_money_left_l284_28408

/-- The amount of money Jerry has left after grocery shopping -/
def money_left (mustard_oil_price : ℝ) (mustard_oil_quantity : ℝ)
                (pasta_price : ℝ) (pasta_quantity : ℝ)
                (sauce_price : ℝ) (sauce_quantity : ℝ)
                (initial_money : ℝ) : ℝ :=
  initial_money - (mustard_oil_price * mustard_oil_quantity +
                   pasta_price * pasta_quantity +
                   sauce_price * sauce_quantity)

/-- Theorem stating that Jerry has $7 left after grocery shopping -/
theorem jerry_money_left :
  money_left 13 2 4 3 5 1 50 = 7 := by
  sorry

end NUMINAMATH_CALUDE_jerry_money_left_l284_28408


namespace NUMINAMATH_CALUDE_wire_cutting_l284_28429

theorem wire_cutting (wire_length : ℚ) (num_parts : ℕ) :
  wire_length = 4/5 →
  num_parts = 3 →
  (wire_length / num_parts) / wire_length = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_wire_cutting_l284_28429


namespace NUMINAMATH_CALUDE_divisibility_problem_l284_28439

theorem divisibility_problem (n : ℤ) (h : n % 30 = 16) : (2 * n) % 30 = 2 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l284_28439


namespace NUMINAMATH_CALUDE_loonie_value_l284_28402

/-- Represents the types of coins Antonella has --/
inductive Coin
| Loonie
| Toonie

/-- The problem setup --/
def AntonellaProblem (loonie_value : ℚ) : Prop :=
  ∃ (num_loonies num_toonies : ℕ),
    -- Total number of coins is 10
    num_loonies + num_toonies = 10 ∧
    -- She initially had 4 toonies
    num_toonies = 4 ∧
    -- Value of a toonie is $2
    -- Total value before buying Frappuccino
    num_loonies * loonie_value + num_toonies * 2 = 14 ∧
    -- She still has $11 after buying $3 Frappuccino
    num_loonies * loonie_value + num_toonies * 2 - 3 = 11

/-- The theorem stating that the value of a loonie is $1 --/
theorem loonie_value : AntonellaProblem 1 := by
  sorry

end NUMINAMATH_CALUDE_loonie_value_l284_28402


namespace NUMINAMATH_CALUDE_angle_measure_60_degrees_l284_28427

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to A, B, C respectively

-- State the theorem
theorem angle_measure_60_degrees (t : Triangle) 
  (h : t.b^2 + t.c^2 - t.a^2 = t.b * t.c) : 
  t.A = 60 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_60_degrees_l284_28427


namespace NUMINAMATH_CALUDE_sector_area_l284_28437

/-- Given a sector with radius 8 cm and central angle 45°, its area is 8π cm². -/
theorem sector_area (r : ℝ) (θ : ℝ) : 
  r = 8 → θ = 45 * (π / 180) → (1/2) * r^2 * θ = 8 * π := by sorry

end NUMINAMATH_CALUDE_sector_area_l284_28437


namespace NUMINAMATH_CALUDE_product_and_reciprocal_sum_l284_28406

theorem product_and_reciprocal_sum (x y : ℝ) : 
  x > 0 → y > 0 → x * y = 9 → (1 / x) = (4 / y) → x + y = 15 / 2 := by
  sorry

end NUMINAMATH_CALUDE_product_and_reciprocal_sum_l284_28406


namespace NUMINAMATH_CALUDE_job_completion_time_l284_28489

/-- Given that A can complete a job in 6 hours and A and D together can complete the job in 4 hours,
    prove that D can complete the job alone in 12 hours. -/
theorem job_completion_time (a_time d_time ad_time : ℝ) 
  (ha : a_time = 6)
  (had : ad_time = 4)
  (h_job_rate : 1 / a_time + 1 / d_time = 1 / ad_time) : 
  d_time = 12 := by
sorry

end NUMINAMATH_CALUDE_job_completion_time_l284_28489


namespace NUMINAMATH_CALUDE_sum_of_y_coordinates_l284_28432

/-- A rectangle in a 2D plane --/
structure Rectangle where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ

/-- Predicate to check if two points are opposite vertices of a rectangle --/
def are_opposite_vertices (r : Rectangle) (p1 p2 : ℝ × ℝ) : Prop :=
  (r.v1 = p1 ∧ r.v3 = p2) ∨ (r.v1 = p2 ∧ r.v3 = p1) ∨
  (r.v2 = p1 ∧ r.v4 = p2) ∨ (r.v2 = p2 ∧ r.v4 = p1)

/-- Theorem: Sum of y-coordinates of two vertices given the other two --/
theorem sum_of_y_coordinates (r : Rectangle) :
  are_opposite_vertices r (4, 20) (12, -6) →
  (r.v1.2 + r.v2.2 + r.v3.2 + r.v4.2) = 14 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_y_coordinates_l284_28432


namespace NUMINAMATH_CALUDE_dollar_to_yen_exchange_l284_28496

/-- Given an exchange rate where 5000 yen equals 48 dollars, 
    prove that 3 dollars can be exchanged for 312.5 yen. -/
theorem dollar_to_yen_exchange : 
  ∀ (dollar_to_yen : ℝ → ℝ),
  (dollar_to_yen 48 = 5000) →  -- Exchange rate condition
  (dollar_to_yen 3 = 312.5) :=  -- What we want to prove
by
  sorry

end NUMINAMATH_CALUDE_dollar_to_yen_exchange_l284_28496


namespace NUMINAMATH_CALUDE_good_number_decomposition_l284_28407

/-- A natural number is good if it can be represented as the product of two consecutive natural numbers. -/
def is_good (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * (k + 1)

/-- Theorem: Any good number greater than 6 can be represented as the sum of a good number and a number that is 3 times a good number. -/
theorem good_number_decomposition (a : ℕ) (h1 : is_good a) (h2 : a > 6) :
  ∃ x y : ℕ, is_good x ∧ is_good y ∧ a * (a + 1) = x * (x + 1) + 3 * (y * (y + 1)) :=
sorry

end NUMINAMATH_CALUDE_good_number_decomposition_l284_28407


namespace NUMINAMATH_CALUDE_spinner_probability_l284_28455

-- Define the spinners
def spinner_C : Finset ℕ := {1, 3, 5, 7}
def spinner_D : Finset ℕ := {2, 4, 6}

-- Define the probability space
def Ω : Finset (ℕ × ℕ) := spinner_C.product spinner_D

-- Define the event of sum being divisible by 3
def E : Finset (ℕ × ℕ) := Ω.filter (λ p => (p.1 + p.2) % 3 = 0)

theorem spinner_probability : 
  (Finset.card E : ℚ) / (Finset.card Ω : ℚ) = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_spinner_probability_l284_28455


namespace NUMINAMATH_CALUDE_range_of_a_l284_28494

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc a (a + 2), |x + a| ≥ 2 * |x|) → a ≤ -3/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l284_28494


namespace NUMINAMATH_CALUDE_number_problem_l284_28483

theorem number_problem (N : ℝ) : 
  (3/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N * (1/2 : ℝ) = 45 → 
  (60/100 : ℝ) * N = 540 := by
sorry

end NUMINAMATH_CALUDE_number_problem_l284_28483


namespace NUMINAMATH_CALUDE_lucas_age_in_three_years_l284_28411

/-- Proves Lucas's age in three years given the relationships between Gladys, Billy, and Lucas's ages -/
theorem lucas_age_in_three_years (gladys billy lucas : ℕ) : 
  gladys = 2 * (billy + lucas) → 
  gladys = 3 * billy → 
  gladys = 30 → 
  lucas + 3 = 8 := by
  sorry

#check lucas_age_in_three_years

end NUMINAMATH_CALUDE_lucas_age_in_three_years_l284_28411


namespace NUMINAMATH_CALUDE_rectangle_area_increase_l284_28438

/-- 
Theorem: Area increase of a rectangle
Given a rectangle with its length increased by 30% and width increased by 15%,
prove that the area increases by 49.5%.
-/
theorem rectangle_area_increase : 
  ∀ (l w : ℝ), l > 0 → w > 0 → 
  (1.3 * l) * (1.15 * w) = 1.495 * (l * w) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_l284_28438


namespace NUMINAMATH_CALUDE_managers_salary_l284_28416

theorem managers_salary (num_employees : ℕ) (avg_salary : ℝ) (salary_increase : ℝ) :
  num_employees = 20 →
  avg_salary = 1500 →
  salary_increase = 150 →
  (num_employees * avg_salary + (num_employees + 1) * salary_increase) / (num_employees + 1) - avg_salary = salary_increase →
  (num_employees + 1) * ((num_employees * avg_salary + (num_employees + 1) * salary_increase) / (num_employees + 1)) - num_employees * avg_salary = 4650 := by
  sorry

end NUMINAMATH_CALUDE_managers_salary_l284_28416


namespace NUMINAMATH_CALUDE_consecutive_digit_product_is_square_l284_28471

/-- Represents a 16-digit positive integer -/
def SixteenDigitInteger := { n : ℕ // 10^15 ≤ n ∧ n < 10^16 }

/-- Extracts a consecutive sequence of digits from a natural number -/
def extractDigitSequence (n : ℕ) (start : ℕ) (len : ℕ) : ℕ := sorry

/-- Checks if a natural number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

/-- Main theorem: For any 16-digit positive integer, there exists a consecutive
    sequence of digits whose product is a perfect square -/
theorem consecutive_digit_product_is_square (A : SixteenDigitInteger) : 
  ∃ start len, isPerfectSquare (extractDigitSequence A.val start len) :=
sorry

end NUMINAMATH_CALUDE_consecutive_digit_product_is_square_l284_28471


namespace NUMINAMATH_CALUDE_cosine_sine_inequality_l284_28447

theorem cosine_sine_inequality (x : ℝ) : 
  (Real.cos (x / 2) + Real.sin (x / 2) ≤ (Real.sin x - 3) / Real.sqrt 2) ↔ 
  ∃ k : ℤ, x = -3 * Real.pi / 2 + 4 * Real.pi * ↑k :=
sorry

end NUMINAMATH_CALUDE_cosine_sine_inequality_l284_28447


namespace NUMINAMATH_CALUDE_kanul_total_amount_l284_28405

/-- The total amount Kanul had -/
def T : ℝ := 93750

/-- The amount spent on raw materials -/
def raw_materials : ℝ := 35000

/-- The amount spent on machinery -/
def machinery : ℝ := 40000

/-- The percentage of total amount spent as cash -/
def cash_percentage : ℝ := 0.20

theorem kanul_total_amount :
  raw_materials + machinery + cash_percentage * T = T := by sorry

end NUMINAMATH_CALUDE_kanul_total_amount_l284_28405


namespace NUMINAMATH_CALUDE_order_of_roots_l284_28475

theorem order_of_roots : 3^(2/3) < 2^(4/3) ∧ 2^(4/3) < 25^(1/3) := by sorry

end NUMINAMATH_CALUDE_order_of_roots_l284_28475


namespace NUMINAMATH_CALUDE_hyperbola_condition_l284_28434

theorem hyperbola_condition (m : ℝ) (h1 : -3 < m) (h2 : m < 0) :
  ∃ (x y : ℝ), (x^2 / (m - 2) + y^2 / (m + 3) = 1) ∧ 
  (∀ (a b : ℝ), a^2 / (m - 2) + b^2 / (m + 3) = 1 → 
    (a, b) ≠ (0, 0) ∧ (a / (m - 2), b / (m + 3)) ≠ (0, 0)) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l284_28434


namespace NUMINAMATH_CALUDE_regular_tetrahedron_edge_sum_plus_three_l284_28448

/-- A regular tetrahedron is a tetrahedron with all faces congruent equilateral triangles. -/
structure RegularTetrahedron where
  sideLength : ℝ
  sideLength_pos : sideLength > 0

/-- The number of edges in a tetrahedron -/
def tetrahedronEdgeCount : ℕ := 6

/-- Calculate the sum of edge lengths of a regular tetrahedron plus an additional length -/
def sumEdgeLengthsPlusExtra (t : RegularTetrahedron) (extra : ℝ) : ℝ :=
  (t.sideLength * tetrahedronEdgeCount : ℝ) + extra

/-- Theorem: For a regular tetrahedron with side length 3 cm, 
    the sum of its edge lengths plus 3 cm is 21 cm -/
theorem regular_tetrahedron_edge_sum_plus_three
  (t : RegularTetrahedron)
  (h : t.sideLength = 3) :
  sumEdgeLengthsPlusExtra t 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_regular_tetrahedron_edge_sum_plus_three_l284_28448


namespace NUMINAMATH_CALUDE_function_value_l284_28466

-- Define the function f
noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + b else Real.log x / Real.log 2

-- State the theorem
theorem function_value (b : ℝ) : f b (f b (1/2)) = 3 → b = 2 := by
  sorry

end NUMINAMATH_CALUDE_function_value_l284_28466


namespace NUMINAMATH_CALUDE_hostel_provisions_l284_28495

/-- The number of days the provisions would last for the initial number of men -/
def initial_days : ℕ := 48

/-- The number of days the provisions would last after some men left -/
def final_days : ℕ := 60

/-- The number of men who left the hostel -/
def men_left : ℕ := 50

/-- The initial number of men in the hostel -/
def initial_men : ℕ := 250

theorem hostel_provisions :
  initial_men * initial_days = (initial_men - men_left) * final_days :=
by sorry

end NUMINAMATH_CALUDE_hostel_provisions_l284_28495


namespace NUMINAMATH_CALUDE_big_sale_commission_l284_28493

def commission_problem (new_average : ℝ) (num_sales : ℕ) (average_increase : ℝ) : Prop :=
  let old_average := new_average - average_increase
  let old_total := old_average * (num_sales - 1 : ℝ)
  let new_total := new_average * num_sales
  new_total - old_total = 1150

theorem big_sale_commission : 
  commission_problem 400 6 150 := by sorry

end NUMINAMATH_CALUDE_big_sale_commission_l284_28493


namespace NUMINAMATH_CALUDE_tangent_slope_angle_l284_28450

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 + 1)

theorem tangent_slope_angle (x : ℝ) : 
  let slope := (deriv f) 1
  Real.arctan slope = π / 4 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_angle_l284_28450


namespace NUMINAMATH_CALUDE_car_problem_solution_l284_28403

def car_problem (t : ℝ) : Prop :=
  let v1 : ℝ := 60  -- First speed in km/h
  let v2 : ℝ := 90  -- Second speed in km/h
  let t2 : ℝ := 2/3 -- Time at second speed in hours (40 minutes = 2/3 hour)
  let v_avg : ℝ := 80 -- Average speed in km/h
  
  -- Total distance
  let d_total : ℝ := v1 * t + v2 * t2
  
  -- Total time
  let t_total : ℝ := t + t2
  
  -- Average speed equation
  v_avg = d_total / t_total

theorem car_problem_solution :
  ∃ t : ℝ, car_problem t ∧ t = 1/3 := by sorry

end NUMINAMATH_CALUDE_car_problem_solution_l284_28403


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l284_28451

theorem complex_fraction_equality : 1 + 1 / (2 + 1 / 3) = 10 / 7 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l284_28451


namespace NUMINAMATH_CALUDE_correct_assignment_plans_l284_28425

/-- The number of students in the group -/
def total_students : ℕ := 6

/-- The number of tasks to be assigned -/
def total_tasks : ℕ := 4

/-- The number of students who cannot perform the first task -/
def restricted_students : ℕ := 2

/-- Calculates the number of distinct assignment plans -/
def assignment_plans : ℕ :=
  (total_students - restricted_students) * (total_students - 1) * (total_students - 2) * (total_students - 3)

theorem correct_assignment_plans :
  assignment_plans = 240 :=
sorry

end NUMINAMATH_CALUDE_correct_assignment_plans_l284_28425


namespace NUMINAMATH_CALUDE_finite_sum_evaluation_l284_28498

theorem finite_sum_evaluation : 
  let S := (1 : ℚ) / 4^1 + 2 / 4^2 + 3 / 4^3 + 4 / 4^4 + 5 / 4^5
  S = 4/3 * (1 - 1/4^6) :=
by sorry

end NUMINAMATH_CALUDE_finite_sum_evaluation_l284_28498


namespace NUMINAMATH_CALUDE_beatrice_on_beach_probability_l284_28492

theorem beatrice_on_beach_probability :
  let p_beach : ℝ := 1/2  -- Probability of Béatrice being on the beach
  let p_tennis : ℝ := 1/4  -- Probability of Béatrice being on the tennis court
  let p_cafe : ℝ := 1/4  -- Probability of Béatrice being in the cafe
  let p_not_found_beach : ℝ := 1/2  -- Probability of not finding Béatrice if she's on the beach
  let p_not_found_tennis : ℝ := 1/3  -- Probability of not finding Béatrice if she's on the tennis court
  let p_not_found_cafe : ℝ := 0  -- Probability of not finding Béatrice if she's in the cafe

  let p_beach_and_not_found : ℝ := p_beach * p_not_found_beach
  let p_tennis_and_not_found : ℝ := p_tennis * p_not_found_tennis
  let p_cafe_and_not_found : ℝ := p_cafe * p_not_found_cafe
  let p_not_found_total : ℝ := p_beach_and_not_found + p_tennis_and_not_found + p_cafe_and_not_found

  p_beach_and_not_found / p_not_found_total = 3/5 :=
by
  sorry

#check beatrice_on_beach_probability

end NUMINAMATH_CALUDE_beatrice_on_beach_probability_l284_28492


namespace NUMINAMATH_CALUDE_ruble_bill_combination_l284_28454

theorem ruble_bill_combination : ∃ x y z : ℕ, x + y + z = 11 ∧ x + 3 * y + 5 * z = 25 := by
  sorry

end NUMINAMATH_CALUDE_ruble_bill_combination_l284_28454


namespace NUMINAMATH_CALUDE_chair_arrangement_count_l284_28412

/-- The number of ways to arrange 45 chairs in a rectangular array with at least 3 chairs in each row and column -/
def rectangular_chair_arrangements : ℕ := 4

/-- The total number of chairs -/
def total_chairs : ℕ := 45

/-- The minimum number of chairs required in each row and column -/
def min_chairs_per_line : ℕ := 3

theorem chair_arrangement_count :
  ∀ (arrangement : ℕ × ℕ),
    arrangement.1 * arrangement.2 = total_chairs ∧
    arrangement.1 ≥ min_chairs_per_line ∧
    arrangement.2 ≥ min_chairs_per_line →
    rectangular_chair_arrangements = (Finset.filter
      (λ arr : ℕ × ℕ => arr.1 * arr.2 = total_chairs ∧
                        arr.1 ≥ min_chairs_per_line ∧
                        arr.2 ≥ min_chairs_per_line)
      (Finset.product (Finset.range (total_chairs + 1)) (Finset.range (total_chairs + 1)))).card :=
by sorry

end NUMINAMATH_CALUDE_chair_arrangement_count_l284_28412


namespace NUMINAMATH_CALUDE_minimize_difference_product_l284_28414

theorem minimize_difference_product (x y : ℤ) : 
  (20 * x + 19 * y = 2019) →
  (∀ (a b : ℤ), 20 * a + 19 * b = 2019 → |x - y| ≤ |a - b|) →
  x * y = 2623 := by
sorry

end NUMINAMATH_CALUDE_minimize_difference_product_l284_28414


namespace NUMINAMATH_CALUDE_triangle_15_6_13_valid_and_perimeter_l284_28401

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if the given side lengths form a valid triangle -/
def isValidTriangle (t : Triangle) : Prop :=
  t.a + t.b > t.c ∧ t.a + t.c > t.b ∧ t.b + t.c > t.a

/-- Calculates the perimeter of a triangle -/
def perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

/-- Theorem: The triangle with sides 15, 6, and 13 is valid and has a perimeter of 34 -/
theorem triangle_15_6_13_valid_and_perimeter :
  let t : Triangle := { a := 15, b := 6, c := 13 }
  isValidTriangle t ∧ perimeter t = 34 := by
  sorry

end NUMINAMATH_CALUDE_triangle_15_6_13_valid_and_perimeter_l284_28401


namespace NUMINAMATH_CALUDE_similarity_transformation_result_l284_28490

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a similarity transformation with a given ratio -/
structure SimilarityTransformation where
  ratio : ℝ
  center : Point

/-- Applies a similarity transformation to a point -/
def applyTransformation (t : SimilarityTransformation) (p : Point) : Point :=
  { x := t.center.x + t.ratio * (p.x - t.center.x)
  , y := t.center.y + t.ratio * (p.y - t.center.y) }

theorem similarity_transformation_result :
  let A : Point := { x := 2, y := 3 }
  let O : Point := { x := 0, y := 0 }
  let t : SimilarityTransformation := { ratio := 2, center := O }
  let A' := applyTransformation t A
  (A'.x = 4 ∧ A'.y = 6) ∨ (A'.x = -4 ∧ A'.y = -6) := by sorry

end NUMINAMATH_CALUDE_similarity_transformation_result_l284_28490


namespace NUMINAMATH_CALUDE_average_marks_l284_28462

theorem average_marks (n : ℕ) (avg_five : ℝ) (sixth_mark : ℝ) (h1 : n = 6) (h2 : avg_five = 74) (h3 : sixth_mark = 86) :
  ((n - 1) * avg_five + sixth_mark) / n = 76 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_l284_28462


namespace NUMINAMATH_CALUDE_f_properties_l284_28476

def f_property (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x + y) + f (x - y) + f (2 * x) = 4 * f x * f ((x + y) / 2) * f ((y - x) / 2) - 1) ∧
  f 1 = 0

theorem f_properties (f : ℝ → ℝ) (h : f_property f) :
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x : ℝ, f x = f (x + 4)) ∧
  (∀ n : ℤ, f n = if n % 4 = 0 then 1 else if n % 4 = 1 ∨ n % 4 = 3 then 0 else -1) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l284_28476


namespace NUMINAMATH_CALUDE_day_before_day_after_tomorrow_l284_28404

-- Define the days of the week
inductive Day : Type
  | monday : Day
  | tuesday : Day
  | wednesday : Day
  | thursday : Day
  | friday : Day
  | saturday : Day
  | sunday : Day

-- Define a function to get the next day
def nextDay (d : Day) : Day :=
  match d with
  | Day.monday => Day.tuesday
  | Day.tuesday => Day.wednesday
  | Day.wednesday => Day.thursday
  | Day.thursday => Day.friday
  | Day.friday => Day.saturday
  | Day.saturday => Day.sunday
  | Day.sunday => Day.monday

-- Define a function to get the previous day
def prevDay (d : Day) : Day :=
  match d with
  | Day.monday => Day.sunday
  | Day.tuesday => Day.monday
  | Day.wednesday => Day.tuesday
  | Day.thursday => Day.wednesday
  | Day.friday => Day.thursday
  | Day.saturday => Day.friday
  | Day.sunday => Day.saturday

-- Theorem statement
theorem day_before_day_after_tomorrow (today : Day) :
  today = Day.thursday →
  prevDay (nextDay (nextDay today)) = Day.friday :=
by
  sorry


end NUMINAMATH_CALUDE_day_before_day_after_tomorrow_l284_28404


namespace NUMINAMATH_CALUDE_distance_equality_implies_m_equals_negative_one_l284_28470

theorem distance_equality_implies_m_equals_negative_one : 
  ∀ m : ℝ, (|m| = |m + 2|) → m = -1 := by
sorry

end NUMINAMATH_CALUDE_distance_equality_implies_m_equals_negative_one_l284_28470


namespace NUMINAMATH_CALUDE_first_expression_value_l284_28417

theorem first_expression_value (a : ℝ) (E : ℝ) : 
  a = 30 → 
  (E + (3 * a - 8)) / 2 = 79 → 
  E = 76 := by
sorry

end NUMINAMATH_CALUDE_first_expression_value_l284_28417


namespace NUMINAMATH_CALUDE_revenue_change_after_price_and_quantity_adjustment_l284_28430

/-- Proves that a 60% price increase and 20% quantity decrease results in a 28% revenue increase -/
theorem revenue_change_after_price_and_quantity_adjustment 
  (P Q : ℝ) 
  (P_new : ℝ := 1.60 * P) 
  (Q_new : ℝ := 0.80 * Q) 
  (h_P : P > 0) 
  (h_Q : Q > 0) : 
  (P_new * Q_new) / (P * Q) = 1.28 := by
  sorry

end NUMINAMATH_CALUDE_revenue_change_after_price_and_quantity_adjustment_l284_28430


namespace NUMINAMATH_CALUDE_smallest_multiple_of_6_and_15_l284_28478

theorem smallest_multiple_of_6_and_15 (b : ℕ) : 
  (∃ k : ℕ, b = 6 * k) ∧ 
  (∃ m : ℕ, b = 15 * m) ∧ 
  (∀ c : ℕ, c > 0 ∧ (∃ p : ℕ, c = 6 * p) ∧ (∃ q : ℕ, c = 15 * q) → b ≤ c) →
  b = 30 := by
sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_6_and_15_l284_28478


namespace NUMINAMATH_CALUDE_fourth_number_in_sequence_l284_28479

theorem fourth_number_in_sequence (a : Fin 6 → ℝ) 
  (h1 : (a 0 + a 1 + a 2 + a 3 + a 4 + a 5) / 6 = 27)
  (h2 : (a 0 + a 1 + a 2 + a 3) / 4 = 23)
  (h3 : (a 3 + a 4 + a 5) / 3 = 34) :
  a 3 = 32 := by
sorry

end NUMINAMATH_CALUDE_fourth_number_in_sequence_l284_28479


namespace NUMINAMATH_CALUDE_joyce_initial_eggs_l284_28460

/-- 
Given that Joyce has an initial number of eggs, receives 6 more eggs from Marie,
and ends up with 14 eggs in total, prove that Joyce initially had 8 eggs.
-/
theorem joyce_initial_eggs : ℕ → Prop :=
  fun initial_eggs =>
    initial_eggs + 6 = 14 → initial_eggs = 8

/-- Proof of the theorem -/
lemma joyce_initial_eggs_proof : joyce_initial_eggs 8 := by
  sorry

end NUMINAMATH_CALUDE_joyce_initial_eggs_l284_28460


namespace NUMINAMATH_CALUDE_sum_base5_equals_2112_l284_28413

/-- Converts a base 5 number represented as a list of digits to its decimal equivalent -/
def base5ToDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => 5 * acc + d) 0

/-- Converts a decimal number to its base 5 representation as a list of digits -/
def decimalToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc
    else aux (m / 5) ((m % 5) :: acc)
  aux n []

/-- The sum of 1234₅, 234₅, and 34₅ in base 5 is equal to 2112₅ -/
theorem sum_base5_equals_2112 :
  let a := base5ToDecimal [1, 2, 3, 4]
  let b := base5ToDecimal [2, 3, 4]
  let c := base5ToDecimal [3, 4]
  decimalToBase5 (a + b + c) = [2, 1, 1, 2] := by
  sorry


end NUMINAMATH_CALUDE_sum_base5_equals_2112_l284_28413


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l284_28467

theorem completing_square_equivalence :
  ∀ x : ℝ, x^2 - 6*x + 5 = 0 ↔ (x - 3)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l284_28467


namespace NUMINAMATH_CALUDE_min_distance_to_complex_locus_l284_28458

/-- Given a complex number z satisfying |z - 4i| + |z + 2| = 7, 
    the minimum value of |z - i| is 3/√5 -/
theorem min_distance_to_complex_locus (z : ℂ) 
  (h : Complex.abs (z - 4*Complex.I) + Complex.abs (z + 2) = 7) :
  ∃ (w : ℂ), Complex.abs (w - Complex.I) = 3 / Real.sqrt 5 ∧
    ∀ (u : ℂ), Complex.abs (u - Complex.I) ≥ Complex.abs (w - Complex.I) :=
sorry

end NUMINAMATH_CALUDE_min_distance_to_complex_locus_l284_28458


namespace NUMINAMATH_CALUDE_equation_solution_l284_28472

theorem equation_solution : ∃! x : ℚ, (3 - x) / (x + 2) + (3 * x - 6) / (3 - x) = 1 ∧ x = -11 / 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l284_28472


namespace NUMINAMATH_CALUDE_max_regions_11_rays_l284_28477

/-- The number of regions a plane is divided into by n rays -/
def num_regions (n : ℕ) : ℕ := 1 + n * (n + 1) / 2

/-- Theorem: The maximum number of regions a plane can be divided into by 11 rays is 67 -/
theorem max_regions_11_rays : num_regions 11 = 67 := by
  sorry

end NUMINAMATH_CALUDE_max_regions_11_rays_l284_28477


namespace NUMINAMATH_CALUDE_sum_and_equal_after_changes_l284_28449

theorem sum_and_equal_after_changes (a b c : ℝ) 
  (sum_eq : a + b + c = 120)
  (equal_after_changes : a + 4 = b - 12 ∧ b - 12 = 3 * c) :
  b = 60 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_equal_after_changes_l284_28449


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_seven_l284_28424

theorem sum_of_fractions_equals_seven :
  let S := 1 / (4 - Real.sqrt 15) + 1 / (Real.sqrt 15 - Real.sqrt 14) - 
           1 / (Real.sqrt 14 - Real.sqrt 13) + 1 / (Real.sqrt 13 - 3)
  S = 7 := by
sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_seven_l284_28424


namespace NUMINAMATH_CALUDE_fraction_evaluation_l284_28422

theorem fraction_evaluation : (20 - 4) / (6 - 3) = 16 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l284_28422


namespace NUMINAMATH_CALUDE_susan_fewer_cars_than_carol_l284_28400

/-- Given the information about car ownership, prove that Susan owns 2 fewer cars than Carol. -/
theorem susan_fewer_cars_than_carol :
  ∀ (cathy lindsey carol susan : ℕ),
  lindsey = cathy + 4 →
  susan < carol →
  carol = 2 * cathy →
  cathy = 5 →
  cathy + lindsey + carol + susan = 32 →
  carol - susan = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_susan_fewer_cars_than_carol_l284_28400


namespace NUMINAMATH_CALUDE_exactly_one_two_digit_number_satisfies_condition_l284_28435

/-- Reverses the digits of a two-digit number -/
def reverseDigits (n : Nat) : Nat :=
  10 * (n % 10) + (n / 10)

/-- Checks if a number is a two-digit positive integer -/
def isTwoDigit (n : Nat) : Prop :=
  10 ≤ n ∧ n ≤ 99

/-- Checks if a number is thrice a perfect square -/
def isThricePerfectSquare (n : Nat) : Prop :=
  ∃ m : Nat, n = 3 * m * m

theorem exactly_one_two_digit_number_satisfies_condition : 
  ∃! n : Nat, isTwoDigit n ∧ 
    isThricePerfectSquare (n + 2 * (reverseDigits n)) :=
sorry

end NUMINAMATH_CALUDE_exactly_one_two_digit_number_satisfies_condition_l284_28435


namespace NUMINAMATH_CALUDE_power_product_squared_l284_28441

theorem power_product_squared (a b : ℝ) : (a * b^2)^2 = a^2 * b^4 := by
  sorry

end NUMINAMATH_CALUDE_power_product_squared_l284_28441


namespace NUMINAMATH_CALUDE_cube_decomposition_smallest_term_l284_28415

theorem cube_decomposition_smallest_term (m : ℕ) (h : m > 0) :
  m^2 - m + 1 = 73 → m = 9 := by
  sorry

end NUMINAMATH_CALUDE_cube_decomposition_smallest_term_l284_28415


namespace NUMINAMATH_CALUDE_unknown_score_value_l284_28442

/-- Given 5 scores where 4 are known and the average of all 5 scores is 9.3, 
    prove that the unknown score x must be 9.5 -/
theorem unknown_score_value (s1 s2 s3 s4 : ℝ) (x : ℝ) (h_s1 : s1 = 9.1) 
    (h_s2 : s2 = 9.3) (h_s3 : s3 = 9.2) (h_s4 : s4 = 9.4) 
    (h_avg : (s1 + s2 + x + s3 + s4) / 5 = 9.3) : x = 9.5 := by
  sorry

#check unknown_score_value

end NUMINAMATH_CALUDE_unknown_score_value_l284_28442


namespace NUMINAMATH_CALUDE_triangle_existence_l284_28440

theorem triangle_existence (k : ℕ) (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_k : k = 6) 
  (h_ineq : k * (x * y + y * z + z * x) > 5 * (x^2 + y^2 + z^2)) : 
  x + y > z ∧ y + z > x ∧ z + x > y := by
sorry

end NUMINAMATH_CALUDE_triangle_existence_l284_28440
