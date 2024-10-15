import Mathlib

namespace NUMINAMATH_CALUDE_min_cost_to_win_l1004_100433

/-- Represents the possible coin types -/
inductive Coin
| One : Coin
| Two : Coin

/-- The game state -/
structure GameState where
  points : Nat
  cost : Nat

/-- Applies a coin to the game state -/
def apply_coin (s : GameState) (c : Coin) : GameState :=
  match c with
  | Coin.One => { points := s.points + 1, cost := s.cost + 1 }
  | Coin.Two => { points := s.points * 2, cost := s.cost + 2 }

/-- Checks if a game state is valid (50 points or less) -/
def is_valid (s : GameState) : Prop := s.points ≤ 50

/-- Checks if a game state is winning (exactly 50 points) -/
def is_winning (s : GameState) : Prop := s.points = 50

/-- The theorem to prove -/
theorem min_cost_to_win : 
  ∃ (sequence : List Coin), 
    let final_state := sequence.foldl apply_coin { points := 0, cost := 0 }
    is_winning final_state ∧ 
    final_state.cost = 11 ∧ 
    (∀ (other_sequence : List Coin), 
      let other_final_state := other_sequence.foldl apply_coin { points := 0, cost := 0 }
      is_winning other_final_state → other_final_state.cost ≥ 11) :=
by sorry

end NUMINAMATH_CALUDE_min_cost_to_win_l1004_100433


namespace NUMINAMATH_CALUDE_four_digit_numbers_count_l1004_100467

/-- The number of digits in the given number -/
def total_digits : ℕ := 4

/-- The number of distinct digits in the given number -/
def distinct_digits : ℕ := 2

/-- The number of occurrences of the first digit (3) -/
def count_first_digit : ℕ := 2

/-- The number of occurrences of the second digit (0) -/
def count_second_digit : ℕ := 2

/-- The function to calculate the number of permutations -/
def permutations (n : ℕ) (n1 : ℕ) (n2 : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial n1 * Nat.factorial n2)

/-- The theorem stating the number of different four-digit numbers -/
theorem four_digit_numbers_count : 
  permutations (total_digits - 1) (count_first_digit - 1) count_second_digit = 3 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_numbers_count_l1004_100467


namespace NUMINAMATH_CALUDE_consecutive_integers_divisibility_l1004_100405

theorem consecutive_integers_divisibility (k : ℤ) : 
  let n := k * (k + 1) * (k + 2)
  (∃ m : ℤ, n = 11 * m) →
  (∃ m : ℤ, n = 6 * m) ∧
  (∃ m : ℤ, n = 22 * m) ∧
  (∃ m : ℤ, n = 33 * m) ∧
  (∃ m : ℤ, n = 66 * m) ∧
  ¬(∀ k : ℤ, ∃ m : ℤ, k * (k + 1) * (k + 2) = 36 * m) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_divisibility_l1004_100405


namespace NUMINAMATH_CALUDE_fundraising_problem_l1004_100412

/-- The fundraising problem -/
theorem fundraising_problem (total_goal : ℕ) (num_people : ℕ) (fee_per_person : ℕ) 
  (h1 : total_goal = 2400)
  (h2 : num_people = 8)
  (h3 : fee_per_person = 20) :
  (total_goal + num_people * fee_per_person) / num_people = 320 := by
  sorry

#check fundraising_problem

end NUMINAMATH_CALUDE_fundraising_problem_l1004_100412


namespace NUMINAMATH_CALUDE_card_cost_correct_l1004_100416

/-- The cost of cards in the first box -/
def cost_box1 : ℝ := 1.25

/-- The cost of cards in the second box -/
def cost_box2 : ℝ := 1.75

/-- The number of cards bought from each box -/
def cards_per_box : ℕ := 6

/-- The total amount spent -/
def total_spent : ℝ := 18

/-- Theorem stating that the cost of cards in the first box is correct -/
theorem card_cost_correct : 
  cost_box1 * cards_per_box + cost_box2 * cards_per_box = total_spent := by
  sorry

end NUMINAMATH_CALUDE_card_cost_correct_l1004_100416


namespace NUMINAMATH_CALUDE_pizza_consumption_order_l1004_100439

/-- Represents the fraction of pizza eaten by each sibling -/
structure PizzaConsumption where
  alex : Rat
  beth : Rat
  cyril : Rat
  dan : Rat

/-- Compares two rational numbers -/
def ratGreater (a b : Rat) : Prop := a > b

theorem pizza_consumption_order (pc : PizzaConsumption) : 
  pc.alex = 1/7 ∧ 
  pc.beth = 2/5 ∧ 
  pc.cyril = 3/10 ∧ 
  pc.dan = 2 * (1 - (pc.alex + pc.beth + pc.cyril)) →
  ratGreater pc.beth pc.dan ∧ 
  ratGreater pc.dan pc.cyril ∧ 
  ratGreater pc.cyril pc.alex :=
by sorry

end NUMINAMATH_CALUDE_pizza_consumption_order_l1004_100439


namespace NUMINAMATH_CALUDE_max_a_value_l1004_100466

theorem max_a_value (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 + 2 + |x^3 - 2*x| ≥ a*x) → 
  a ≤ 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_max_a_value_l1004_100466


namespace NUMINAMATH_CALUDE_xyz_sum_sqrt_l1004_100451

theorem xyz_sum_sqrt (x y z : ℝ) 
  (h1 : y + z = 18) 
  (h2 : z + x = 19) 
  (h3 : x + y = 20) : 
  Real.sqrt (x * y * z * (x + y + z)) = Real.sqrt 24150.1875 := by
  sorry

end NUMINAMATH_CALUDE_xyz_sum_sqrt_l1004_100451


namespace NUMINAMATH_CALUDE_unique_number_property_l1004_100450

theorem unique_number_property : ∃! x : ℝ, x / 3 = x - 5 := by sorry

end NUMINAMATH_CALUDE_unique_number_property_l1004_100450


namespace NUMINAMATH_CALUDE_pages_read_difference_l1004_100428

theorem pages_read_difference (total_pages : ℕ) (fraction_read : ℚ) : 
  total_pages = 90 → fraction_read = 2/3 → 
  (total_pages : ℚ) * fraction_read - (total_pages : ℚ) * (1 - fraction_read) = 30 := by
  sorry

end NUMINAMATH_CALUDE_pages_read_difference_l1004_100428


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l1004_100401

theorem square_area_from_diagonal (diagonal : ℝ) (area : ℝ) :
  diagonal = 16 →
  area = diagonal^2 / 2 →
  area = 128 :=
by sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l1004_100401


namespace NUMINAMATH_CALUDE_simplify_expression_l1004_100497

theorem simplify_expression (y : ℝ) : 4*y + 8*y^3 + 6 - (3 - 4*y - 8*y^3) = 16*y^3 + 8*y + 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1004_100497


namespace NUMINAMATH_CALUDE_quadratic_function_max_abs_value_ge_one_l1004_100446

/-- Given a quadratic function f(x) = 2x^2 + mx + n, 
    prove that the maximum absolute value of f(1), f(2), and f(3) is at least 1. -/
theorem quadratic_function_max_abs_value_ge_one (m n : ℝ) : 
  let f := fun (x : ℝ) => 2 * x^2 + m * x + n
  max (|f 1|) (max (|f 2|) (|f 3|)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_max_abs_value_ge_one_l1004_100446


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l1004_100496

def p (x : ℝ) : Prop := |2*x - 3| < 1

def q (x : ℝ) : Prop := x * (x - 3) < 0

theorem p_necessary_not_sufficient_for_q :
  (∀ x, q x → p x) ∧ (∃ x, p x ∧ ¬q x) := by sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l1004_100496


namespace NUMINAMATH_CALUDE_multiple_of_n_divisible_by_60_l1004_100431

theorem multiple_of_n_divisible_by_60 (n : ℕ) :
  0 < n →
  n < 200 →
  (∃ k : ℕ, k > 0 ∧ 60 ∣ (k * n)) →
  (∃ p q r : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ n = p * q * r) →
  (∃ m : ℕ, m > 0 ∧ 60 ∣ (m * n) ∧ ∀ k : ℕ, (k > 0 ∧ 60 ∣ (k * n)) → m ≤ k) →
  (∃ m : ℕ, m > 0 ∧ 60 ∣ (m * n) ∧ ∀ k : ℕ, (k > 0 ∧ 60 ∣ (k * n)) → m ≤ k) ∧ m = 60 :=
by sorry

end NUMINAMATH_CALUDE_multiple_of_n_divisible_by_60_l1004_100431


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_log_l1004_100448

theorem arithmetic_geometric_sequence_log (a b : ℝ) : 
  a ≠ b →
  (2 * a = 1 + b) →
  (b ^ 2 = a) →
  7 * a * (Real.log (-b) / Real.log a) = 7/8 := by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_log_l1004_100448


namespace NUMINAMATH_CALUDE_wall_volume_l1004_100429

/-- The volume of a rectangular wall with specific proportions -/
theorem wall_volume (width : ℝ) (height : ℝ) (length : ℝ) : 
  width = 4 → 
  height = 6 * width → 
  length = 7 * height → 
  width * height * length = 16128 := by
sorry

end NUMINAMATH_CALUDE_wall_volume_l1004_100429


namespace NUMINAMATH_CALUDE_integer_solution_congruence_l1004_100495

theorem integer_solution_congruence (x y z : ℤ) 
  (eq1 : x - 3*y + 2*z = 1)
  (eq2 : 2*x + y - 5*z = 7) :
  z ≡ 1 [ZMOD 7] :=
sorry

end NUMINAMATH_CALUDE_integer_solution_congruence_l1004_100495


namespace NUMINAMATH_CALUDE_not_sufficient_for_geometric_sequence_l1004_100485

theorem not_sufficient_for_geometric_sequence 
  (a : ℕ → ℝ) 
  (h : ∀ n : ℕ, n ≥ 2 → a (n + 1) * a (n - 1) = a n ^ 2) :
  ¬ (∀ n : ℕ, ∃ r : ℝ, ∀ k : ℕ, a (n + k) = a n * r ^ k) :=
sorry

end NUMINAMATH_CALUDE_not_sufficient_for_geometric_sequence_l1004_100485


namespace NUMINAMATH_CALUDE_expression_value_l1004_100425

theorem expression_value (a b c d : ℤ) 
  (ha : a = 15) (hb : b = 19) (hc : c = 3) (hd : d = 2) : 
  (a - (b - c)) - ((a - b) - c + d) = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1004_100425


namespace NUMINAMATH_CALUDE_quadratic_roots_bounds_l1004_100476

theorem quadratic_roots_bounds (a b : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 + a*x + b = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ < x₂ →
  (∀ x, -1 < x ∧ x < 1 → x^2 + a*x + b < 0) →
  -1 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_bounds_l1004_100476


namespace NUMINAMATH_CALUDE_complex_fraction_calculation_l1004_100489

theorem complex_fraction_calculation : (9 * 9 - 2 * 2) / ((1 / 12) - (1 / 19)) = 2508 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_calculation_l1004_100489


namespace NUMINAMATH_CALUDE_unique_q_13_l1004_100435

-- Define the cubic polynomial q(x)
def q (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- State the theorem
theorem unique_q_13 (a b c d : ℝ) :
  (∀ x : ℝ, (q a b c d x)^3 - x = 0 → x = 2 ∨ x = -2 ∨ x = 5) →
  q a b c d 2 = 2 →
  q a b c d (-2) = -2 →
  q a b c d 5 = 3 →
  ∃! y : ℝ, q a b c d 13 = y :=
sorry

end NUMINAMATH_CALUDE_unique_q_13_l1004_100435


namespace NUMINAMATH_CALUDE_mary_juan_income_ratio_l1004_100427

theorem mary_juan_income_ratio (juan tim mary : ℝ) 
  (h1 : mary = 1.4 * tim) 
  (h2 : tim = 0.6 * juan) : 
  mary = 0.84 * juan := by
  sorry

end NUMINAMATH_CALUDE_mary_juan_income_ratio_l1004_100427


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l1004_100474

/-- The perimeter of a rhombus with diagonals of 12 inches and 30 inches is 4√261 inches. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 12) (h2 : d2 = 30) :
  4 * Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2) = 4 * Real.sqrt 261 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l1004_100474


namespace NUMINAMATH_CALUDE_factorial_equation_solution_l1004_100403

theorem factorial_equation_solution (n : ℕ) : n * n! + n! = 5040 → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_factorial_equation_solution_l1004_100403


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l1004_100442

theorem shaded_area_calculation (π : ℝ) (h : π > 0) : 
  let square_side : ℝ := 8
  let quarter_circle_radius : ℝ := 0.6 * square_side
  let square_area : ℝ := square_side ^ 2
  let quarter_circles_area : ℝ := π * quarter_circle_radius ^ 2
  square_area - quarter_circles_area = 64 - 23.04 * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l1004_100442


namespace NUMINAMATH_CALUDE_largest_number_in_sample_l1004_100475

/-- Represents a systematic sampling scenario -/
structure SystematicSample where
  total_items : ℕ
  first_number : ℕ
  second_number : ℕ
  sample_size : ℕ

/-- Calculates the largest number in a systematic sample -/
def largest_sample_number (s : SystematicSample) : ℕ :=
  s.first_number + (s.sample_size - 1) * (s.second_number - s.first_number)

/-- Theorem stating the largest number in the given systematic sample -/
theorem largest_number_in_sample :
  let s : SystematicSample := {
    total_items := 400,
    first_number := 8,
    second_number := 33,
    sample_size := 16
  }
  largest_sample_number s = 383 := by sorry

end NUMINAMATH_CALUDE_largest_number_in_sample_l1004_100475


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l1004_100420

theorem sqrt_product_equality : Real.sqrt (49 + 121) * Real.sqrt (64 - 49) = Real.sqrt 2550 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l1004_100420


namespace NUMINAMATH_CALUDE_extended_quadrilateral_area_l1004_100462

/-- Represents a quadrilateral with extended sides -/
structure ExtendedQuadrilateral where
  -- Original quadrilateral
  area : ℝ
  -- Lengths of sides
  wz : ℝ
  zx : ℝ
  xy : ℝ
  yw : ℝ
  -- Conditions for extended sides
  wz_extended : ℝ
  zx_extended : ℝ
  xy_extended : ℝ
  yw_extended : ℝ
  -- Conditions for double length
  wz_double : wz_extended = 2 * wz
  zx_double : zx_extended = 2 * zx
  xy_double : xy_extended = 2 * xy
  yw_double : yw_extended = 2 * yw

/-- Theorem stating the relationship between areas of original and extended quadrilaterals -/
theorem extended_quadrilateral_area 
  (q : ExtendedQuadrilateral) : 
  ∃ (extended_area : ℝ), extended_area = 9 * q.area := by
  sorry

end NUMINAMATH_CALUDE_extended_quadrilateral_area_l1004_100462


namespace NUMINAMATH_CALUDE_smallest_valid_debt_proof_l1004_100447

/-- The value of one sheep in dollars -/
def sheep_value : ℕ := 250

/-- The value of one lamb in dollars -/
def lamb_value : ℕ := 150

/-- A debt resolution is valid if it can be expressed as an integer combination of sheep and lambs -/
def is_valid_debt (d : ℕ) : Prop :=
  ∃ (s l : ℤ), d = sheep_value * s + lamb_value * l

/-- The smallest positive debt that can be resolved -/
def smallest_valid_debt : ℕ := 50

theorem smallest_valid_debt_proof :
  (∀ d : ℕ, d > 0 ∧ d < smallest_valid_debt → ¬is_valid_debt d) ∧
  is_valid_debt smallest_valid_debt :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_debt_proof_l1004_100447


namespace NUMINAMATH_CALUDE_least_n_satisfying_inequality_l1004_100491

theorem least_n_satisfying_inequality : ∃ n : ℕ, 
  (∀ k : ℕ, k < n → (1 : ℚ) / k - (1 : ℚ) / (k + 1) ≥ (1 : ℚ) / 15) ∧
  ((1 : ℚ) / n - (1 : ℚ) / (n + 1) < (1 : ℚ) / 15) ∧
  n = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_least_n_satisfying_inequality_l1004_100491


namespace NUMINAMATH_CALUDE_percentage_gain_calculation_l1004_100482

def calculate_percentage_gain (total_bought : ℕ) (cost_per_bowl : ℚ) (total_sold : ℕ) (sell_per_bowl : ℚ) : ℚ :=
  let total_cost := total_bought * cost_per_bowl
  let total_revenue := total_sold * sell_per_bowl
  let profit := total_revenue - total_cost
  (profit / total_cost) * 100

theorem percentage_gain_calculation :
  let total_bought : ℕ := 114
  let cost_per_bowl : ℚ := 13
  let total_sold : ℕ := 108
  let sell_per_bowl : ℚ := 17
  abs (calculate_percentage_gain total_bought cost_per_bowl total_sold sell_per_bowl - 23.88) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_percentage_gain_calculation_l1004_100482


namespace NUMINAMATH_CALUDE_a_5_equals_5_l1004_100443

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  h1 : a 3 + a 11 = 18  -- Condition 1
  h2 : (a 1 + a 2 + a 3) = -3  -- Condition 2 (S₃ = -3)

/-- The theorem stating that a₅ = 5 for the given arithmetic sequence -/
theorem a_5_equals_5 (seq : ArithmeticSequence) : seq.a 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_a_5_equals_5_l1004_100443


namespace NUMINAMATH_CALUDE_grocery_cost_is_correct_l1004_100469

def grocery_cost (egg_quantity : ℕ) (egg_price : ℚ) (milk_quantity : ℕ) (milk_price : ℚ)
  (bread_quantity : ℕ) (bread_price : ℚ) (egg_milk_tax : ℚ) (bread_tax : ℚ)
  (egg_discount : ℚ) (milk_discount : ℚ) : ℚ :=
  let egg_subtotal := egg_quantity * egg_price
  let milk_subtotal := milk_quantity * milk_price
  let bread_subtotal := bread_quantity * bread_price
  let egg_discounted := egg_subtotal * (1 - egg_discount)
  let milk_discounted := milk_subtotal * (1 - milk_discount)
  let egg_with_tax := egg_discounted * (1 + egg_milk_tax)
  let milk_with_tax := milk_discounted * (1 + egg_milk_tax)
  let bread_with_tax := bread_subtotal * (1 + bread_tax)
  egg_with_tax + milk_with_tax + bread_with_tax

theorem grocery_cost_is_correct :
  grocery_cost 36 0.5 2 3 4 1.25 0.05 0.02 0.1 0.05 = 12.51 := by
  sorry

end NUMINAMATH_CALUDE_grocery_cost_is_correct_l1004_100469


namespace NUMINAMATH_CALUDE_number_of_ways_to_draw_l1004_100459

/-- The number of balls in the bin -/
def total_balls : ℕ := 15

/-- The number of balls to be drawn -/
def drawn_balls : ℕ := 4

/-- The sequence of colors to be drawn -/
def color_sequence : List String := ["Red", "Green", "Blue", "Yellow"]

/-- Function to calculate the number of ways to draw the balls -/
def ways_to_draw : ℕ := (total_balls - 0) * (total_balls - 1) * (total_balls - 2) * (total_balls - 3)

/-- Theorem stating the number of ways to draw the balls -/
theorem number_of_ways_to_draw :
  ways_to_draw = 32760 := by sorry

end NUMINAMATH_CALUDE_number_of_ways_to_draw_l1004_100459


namespace NUMINAMATH_CALUDE_physics_marks_l1004_100441

theorem physics_marks (P C M : ℝ) 
  (avg_all : (P + C + M) / 3 = 70)
  (avg_pm : (P + M) / 2 = 90)
  (avg_pc : (P + C) / 2 = 70) :
  P = 110 := by sorry

end NUMINAMATH_CALUDE_physics_marks_l1004_100441


namespace NUMINAMATH_CALUDE_shopkeeper_sold_450_meters_l1004_100406

/-- Represents the sale of cloth by a shopkeeper -/
structure ClothSale where
  totalSellingPrice : ℕ  -- Total selling price in Rupees
  lossPerMeter : ℕ       -- Loss per meter in Rupees
  costPricePerMeter : ℕ  -- Cost price per meter in Rupees

/-- Calculates the number of meters of cloth sold -/
def metersOfClothSold (sale : ClothSale) : ℕ :=
  sale.totalSellingPrice / (sale.costPricePerMeter - sale.lossPerMeter)

/-- Theorem stating that the shopkeeper sold 450 meters of cloth -/
theorem shopkeeper_sold_450_meters :
  let sale : ClothSale := {
    totalSellingPrice := 18000,
    lossPerMeter := 5,
    costPricePerMeter := 45
  }
  metersOfClothSold sale = 450 := by
  sorry


end NUMINAMATH_CALUDE_shopkeeper_sold_450_meters_l1004_100406


namespace NUMINAMATH_CALUDE_lg2_bounds_l1004_100465

theorem lg2_bounds :
  (10 : ℝ)^3 = 1000 ∧ (10 : ℝ)^4 = 10000 ∧
  (2 : ℝ)^10 = 1024 ∧ (2 : ℝ)^11 = 2048 ∧
  (2 : ℝ)^12 = 4096 ∧ (2 : ℝ)^13 = 8192 →
  3/10 < Real.log 2 / Real.log 10 ∧ Real.log 2 / Real.log 10 < 4/13 := by
  sorry

end NUMINAMATH_CALUDE_lg2_bounds_l1004_100465


namespace NUMINAMATH_CALUDE_largest_area_quadrilateral_in_sector_l1004_100452

/-- The largest area of a right-angled quadrilateral inscribed in a circular sector -/
theorem largest_area_quadrilateral_in_sector (r : ℝ) (h : r > 0) :
  let max_area (α : ℝ) := 
    (2 * r^2 * Real.sin (α/2) * Real.sin (α/2)) / Real.sin α
  (max_area (2*π/3) = (r^2 * Real.sqrt 3) / 3) ∧ 
  (max_area (4*π/3) = r^2 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_largest_area_quadrilateral_in_sector_l1004_100452


namespace NUMINAMATH_CALUDE_video_game_lives_l1004_100471

theorem video_game_lives (initial_lives won_lives gained_lives : Float) 
  (h1 : initial_lives = 43.0)
  (h2 : won_lives = 14.0)
  (h3 : gained_lives = 27.0) :
  initial_lives + won_lives + gained_lives = 84.0 := by
  sorry

end NUMINAMATH_CALUDE_video_game_lives_l1004_100471


namespace NUMINAMATH_CALUDE_plank_length_l1004_100437

/-- The length of a plank given specific movements of its ends -/
theorem plank_length (a b : ℝ) : 
  (∀ x y, x^2 + y^2 = a^2 + b^2 → (x - 8)^2 + (y + 4)^2 = a^2 + b^2) →
  (∀ x y, x^2 + y^2 = a^2 + b^2 → (x - 17)^2 + (y + 7)^2 = a^2 + b^2) →
  a^2 + b^2 = 65^2 := by
  sorry

end NUMINAMATH_CALUDE_plank_length_l1004_100437


namespace NUMINAMATH_CALUDE_fraction_simplification_l1004_100445

theorem fraction_simplification (m : ℝ) (h : m ≠ 3) :
  m^2 / (m - 3) + 9 / (3 - m) = m + 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1004_100445


namespace NUMINAMATH_CALUDE_total_distance_after_five_days_l1004_100417

/-- The total distance run by Peter and Andrew after 5 days -/
def total_distance (andrew_distance : ℕ) (peter_extra : ℕ) (days : ℕ) : ℕ :=
  (andrew_distance + peter_extra + andrew_distance) * days

/-- Theorem stating the total distance run by Peter and Andrew after 5 days -/
theorem total_distance_after_five_days :
  total_distance 2 3 5 = 35 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_after_five_days_l1004_100417


namespace NUMINAMATH_CALUDE_petya_vasya_equal_numbers_possible_l1004_100453

theorem petya_vasya_equal_numbers_possible :
  ∃ (n : ℤ) (k : ℕ), n ≠ 0 ∧ 
  (n + 10 * k) * 2014 = (n - 10 * k) / 2014 := by
  sorry

end NUMINAMATH_CALUDE_petya_vasya_equal_numbers_possible_l1004_100453


namespace NUMINAMATH_CALUDE_binary_is_largest_l1004_100419

/-- Convert a number from base b to decimal --/
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

/-- The given numbers in their respective bases --/
def binary : List Nat := [1, 1, 0, 1, 1]
def base_4 : List Nat := [3, 0, 1]
def base_5 : List Nat := [4, 4]
def decimal : Nat := 25

/-- Theorem stating that the binary number is the largest --/
theorem binary_is_largest :
  let a := to_decimal binary 2
  let b := to_decimal base_4 4
  let c := to_decimal base_5 5
  let d := decimal
  a > b ∧ a > c ∧ a > d :=
by sorry


end NUMINAMATH_CALUDE_binary_is_largest_l1004_100419


namespace NUMINAMATH_CALUDE_max_difference_of_reversed_digits_l1004_100432

/-- Represents a three-digit positive integer -/
structure ThreeDigitInt where
  value : ℕ
  is_three_digit : 100 ≤ value ∧ value ≤ 999

/-- Returns true if two ThreeDigitInt have the same digits in reverse order -/
def reverse_digits (a b : ThreeDigitInt) : Prop :=
  ∃ (x y z : ℕ), x ≤ 9 ∧ y ≤ 9 ∧ z ≤ 9 ∧
    a.value = 100 * x + 10 * y + z ∧
    b.value = 100 * z + 10 * y + x

/-- The theorem to be proved -/
theorem max_difference_of_reversed_digits (q r : ThreeDigitInt) :
  reverse_digits q r →
  (∃ p : ℕ, Nat.Prime p ∧ p ∣ (q.value - r.value)) →
  q.value - r.value < 300 →
  (∀ s t : ThreeDigitInt, reverse_digits s t →
    (∃ p : ℕ, Nat.Prime p ∧ p ∣ (s.value - t.value)) →
    s.value - t.value < 300 →
    s.value - t.value ≤ q.value - r.value) →
  q.value - r.value = 297 := by
sorry

end NUMINAMATH_CALUDE_max_difference_of_reversed_digits_l1004_100432


namespace NUMINAMATH_CALUDE_inequality_solution_l1004_100411

theorem inequality_solution (x : ℝ) : 
  (x^2 + x^3 - 2*x^4) / (x + x^2 - 2*x^3) ≥ -1 ↔ 
  (x ≥ -1 ∧ x ≠ -1/2 ∧ x ≠ 0 ∧ x ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1004_100411


namespace NUMINAMATH_CALUDE_ferry_river_crossing_l1004_100456

/-- The width of a river crossed by two ferries --/
def river_width : ℝ := 1280

/-- The distance from the nearest shore where the ferries first meet --/
def first_meeting_distance : ℝ := 720

/-- The distance from the other shore where the ferries meet on the return trip --/
def second_meeting_distance : ℝ := 400

/-- Theorem stating that the width of the river is 1280 meters given the conditions --/
theorem ferry_river_crossing :
  let w := river_width
  let d1 := first_meeting_distance
  let d2 := second_meeting_distance
  (d1 + (w - d1) = w) ∧
  (3 * w = 2 * w + 2 * d1) ∧
  (3 * d1 = 2 * w - d2) →
  w = 1280 := by sorry


end NUMINAMATH_CALUDE_ferry_river_crossing_l1004_100456


namespace NUMINAMATH_CALUDE_circumradius_ge_twice_inradius_l1004_100463

/-- A triangle is represented by its three vertices in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The radius of the circumscribed circle of a triangle -/
noncomputable def circumradius (t : Triangle) : ℝ := sorry

/-- The radius of the inscribed circle of a triangle -/
noncomputable def inradius (t : Triangle) : ℝ := sorry

/-- Checks if a triangle is equilateral -/
def is_equilateral (t : Triangle) : Prop := sorry

/-- Theorem: The circumradius is greater than or equal to twice the inradius for any triangle,
    with equality if and only if the triangle is equilateral -/
theorem circumradius_ge_twice_inradius (t : Triangle) :
  circumradius t ≥ 2 * inradius t ∧
  (circumradius t = 2 * inradius t ↔ is_equilateral t) := by
  sorry

end NUMINAMATH_CALUDE_circumradius_ge_twice_inradius_l1004_100463


namespace NUMINAMATH_CALUDE_yellow_crayon_count_prove_yellow_crayons_l1004_100499

/-- Proves that the number of yellow crayons is 32 given the conditions of the problem. -/
theorem yellow_crayon_count : ℕ → ℕ → ℕ → Prop :=
  fun red blue yellow =>
    (red = 14) →
    (blue = red + 5) →
    (yellow = 2 * blue - 6) →
    (yellow = 32)

/-- The main theorem that proves the number of yellow crayons. -/
theorem prove_yellow_crayons :
  ∃ (red blue yellow : ℕ),
    yellow_crayon_count red blue yellow :=
by
  sorry

end NUMINAMATH_CALUDE_yellow_crayon_count_prove_yellow_crayons_l1004_100499


namespace NUMINAMATH_CALUDE_reciprocal_of_25_l1004_100444

theorem reciprocal_of_25 (x : ℝ) : (1 / x = 25) → (x = 1 / 25) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_25_l1004_100444


namespace NUMINAMATH_CALUDE_total_blisters_l1004_100430

/-- Given a person with 60 blisters on each arm and 80 blisters on the rest of their body,
    the total number of blisters is 200. -/
theorem total_blisters (blisters_per_arm : ℕ) (blisters_rest : ℕ) :
  blisters_per_arm = 60 →
  blisters_rest = 80 →
  blisters_per_arm * 2 + blisters_rest = 200 :=
by sorry

end NUMINAMATH_CALUDE_total_blisters_l1004_100430


namespace NUMINAMATH_CALUDE_remainder_67_power_67_plus_67_mod_68_l1004_100484

theorem remainder_67_power_67_plus_67_mod_68 : (67^67 + 67) % 68 = 66 := by
  sorry

end NUMINAMATH_CALUDE_remainder_67_power_67_plus_67_mod_68_l1004_100484


namespace NUMINAMATH_CALUDE_trigonometric_identities_and_circle_parametrization_l1004_100440

theorem trigonometric_identities_and_circle_parametrization (a t : ℝ) 
  (h : t = Real.tan (a / 2)) : 
  Real.cos a = (1 - t^2) / (1 + t^2) ∧ 
  Real.sin a = 2 * t / (1 + t^2) ∧ 
  Real.tan a = 2 * t / (1 - t^2) ∧ 
  ∀ x y : ℝ, x = (1 - t^2) / (1 + t^2) ∧ y = 2 * t / (1 + t^2) → x^2 + y^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_and_circle_parametrization_l1004_100440


namespace NUMINAMATH_CALUDE_solve_for_S_l1004_100410

theorem solve_for_S : ∃ S : ℚ, (1/2 : ℚ) * (1/7 : ℚ) * S = (1/4 : ℚ) * (1/6 : ℚ) * 120 ∧ S = 70 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_S_l1004_100410


namespace NUMINAMATH_CALUDE_quadratic_equation_set_l1004_100483

theorem quadratic_equation_set (a : ℝ) : 
  (∃! x, a * x^2 - 3 * x + 2 = 0) → (a = 0 ∨ a = 9/8) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_set_l1004_100483


namespace NUMINAMATH_CALUDE_root_product_theorem_l1004_100480

theorem root_product_theorem (a b m p q : ℝ) : 
  (a^2 - m*a + 3 = 0) → 
  (b^2 - m*b + 3 = 0) → 
  ((a + 1/b)^2 - p*(a + 1/b) + q = 0) → 
  ((b + 1/a)^2 - p*(b + 1/a) + q = 0) → 
  q = 13/3 := by
sorry

end NUMINAMATH_CALUDE_root_product_theorem_l1004_100480


namespace NUMINAMATH_CALUDE_max_knights_count_l1004_100472

/-- Represents the type of islander: Knight or Liar -/
inductive IslanderType
  | Knight
  | Liar

/-- Represents the statement made by an islander -/
inductive Statement
  | BothNeighborsLiars
  | OneNeighborLiar

/-- Configuration of islanders around the table -/
structure IslanderConfig where
  total : Nat
  half_both_liars : Nat
  half_one_liar : Nat
  knight_count : Nat

/-- Checks if the given configuration is valid -/
def is_valid_config (config : IslanderConfig) : Prop :=
  config.total = 100 ∧
  config.half_both_liars = 50 ∧
  config.half_one_liar = 50 ∧
  config.knight_count ≤ config.total

/-- Theorem stating the maximum number of knights possible -/
theorem max_knights_count (config : IslanderConfig) 
  (h_valid : is_valid_config config) : 
  config.knight_count ≤ 67 :=
sorry

end NUMINAMATH_CALUDE_max_knights_count_l1004_100472


namespace NUMINAMATH_CALUDE_largest_divisor_of_expression_l1004_100461

theorem largest_divisor_of_expression (x : ℤ) (h : Odd x) :
  (∃ (k : ℤ), (15 * x + 3) * (15 * x + 9) * (10 * x + 10) = 1920 * k) ∧
  (∀ (n : ℤ), n > 1920 → ∃ (y : ℤ), Odd y ∧ ¬(∃ (m : ℤ), (15 * y + 3) * (15 * y + 9) * (10 * y + 10) = n * m)) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_expression_l1004_100461


namespace NUMINAMATH_CALUDE_number_multiplied_by_9999_l1004_100449

theorem number_multiplied_by_9999 : ∃ x : ℕ, x * 9999 = 5865863355 ∧ x = 586650 := by
  sorry

end NUMINAMATH_CALUDE_number_multiplied_by_9999_l1004_100449


namespace NUMINAMATH_CALUDE_solution_set_abs_inequality_l1004_100490

theorem solution_set_abs_inequality (x : ℝ) :
  (|x - 3| < 2) ↔ (1 < x ∧ x < 5) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_abs_inequality_l1004_100490


namespace NUMINAMATH_CALUDE_monster_consumption_l1004_100402

theorem monster_consumption (a : ℕ → ℕ) (h1 : a 0 = 121) (h2 : ∀ n, a (n + 1) = 2 * a n) : 
  a 0 + a 1 + a 2 = 847 := by
sorry

end NUMINAMATH_CALUDE_monster_consumption_l1004_100402


namespace NUMINAMATH_CALUDE_log3_20_approximation_l1004_100438

-- Define the approximate values given in the problem
def log10_2_approx : ℝ := 0.301
def log10_3_approx : ℝ := 0.477

-- Define the target approximation
def log3_20_target : ℝ := 2.786

-- State the theorem
theorem log3_20_approximation :
  abs (Real.log 20 / Real.log 3 - log3_20_target) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_log3_20_approximation_l1004_100438


namespace NUMINAMATH_CALUDE_gcf_64_80_l1004_100413

theorem gcf_64_80 : Nat.gcd 64 80 = 16 := by
  sorry

end NUMINAMATH_CALUDE_gcf_64_80_l1004_100413


namespace NUMINAMATH_CALUDE_prob_end_two_tails_after_second_head_l1004_100458

/-- A fair coin flip can result in either heads or tails with equal probability -/
def FairCoin : Type := Bool

/-- The outcome of a sequence of coin flips -/
inductive FlipOutcome
| TwoHeads
| TwoTails
| Incomplete

/-- The state of the coin flipping process -/
structure FlipState :=
  (seenSecondHead : Bool)
  (lastFlip : Option Bool)
  (outcome : FlipOutcome)

/-- Simulates a single coin flip and updates the state -/
def flipCoin (state : FlipState) : FlipState := sorry

/-- Calculates the probability of ending with two tails after seeing the second head -/
def probEndTwoTailsAfterSecondHead : ℝ := sorry

/-- The main theorem to prove -/
theorem prob_end_two_tails_after_second_head :
  probEndTwoTailsAfterSecondHead = 1 / 24 := by sorry

end NUMINAMATH_CALUDE_prob_end_two_tails_after_second_head_l1004_100458


namespace NUMINAMATH_CALUDE_smallest_prime_digit_sum_23_l1004_100487

/-- A function that returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number is prime -/
def is_prime (n : ℕ) : Prop := sorry

/-- Theorem stating that 599 is the smallest prime number whose digits sum to 23 -/
theorem smallest_prime_digit_sum_23 :
  (is_prime 599) ∧ 
  (digit_sum 599 = 23) ∧ 
  (∀ n : ℕ, n < 599 → ¬(is_prime n ∧ digit_sum n = 23)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_digit_sum_23_l1004_100487


namespace NUMINAMATH_CALUDE_lcm_gcf_ratio_l1004_100407

theorem lcm_gcf_ratio : (Nat.lcm 144 756) / (Nat.gcd 144 756) = 84 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_ratio_l1004_100407


namespace NUMINAMATH_CALUDE_exponential_function_fixed_point_l1004_100454

theorem exponential_function_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  a^(0 : ℝ) = 1 := by sorry

end NUMINAMATH_CALUDE_exponential_function_fixed_point_l1004_100454


namespace NUMINAMATH_CALUDE_mysterious_division_l1004_100421

theorem mysterious_division :
  ∃! (d q : ℕ),
    d ∈ Finset.range 900 ∧ d ≥ 100 ∧
    q ∈ Finset.range 90000 ∧ q ≥ 10000 ∧
    10000000 = d * q + (10000000 % d) ∧
    d = 124 ∧ q = 80809 := by
  sorry

end NUMINAMATH_CALUDE_mysterious_division_l1004_100421


namespace NUMINAMATH_CALUDE_max_distance_circle_to_line_l1004_100400

/-- The maximum distance from any point on the circle (x-1)² + (y-1)² = 2 to the line x + y - 4 = 0 is 2√2. -/
theorem max_distance_circle_to_line :
  let circle := {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 1)^2 = 2}
  let line := {p : ℝ × ℝ | p.1 + p.2 - 4 = 0}
  ∃ (d : ℝ), d = 2 * Real.sqrt 2 ∧
    (∀ p ∈ circle, ∀ q ∈ line, Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≤ d) ∧
    (∃ p ∈ circle, ∃ q ∈ line, Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = d) :=
by sorry

end NUMINAMATH_CALUDE_max_distance_circle_to_line_l1004_100400


namespace NUMINAMATH_CALUDE_partnership_profit_calculation_l1004_100470

/-- Represents a business partnership --/
structure Partnership where
  investment_A : ℕ
  investment_B : ℕ
  investment_C : ℕ
  investment_D : ℕ
  profit_ratio_A : ℕ
  profit_ratio_B : ℕ
  profit_ratio_C : ℕ
  profit_ratio_D : ℕ
  C_profit_share : ℕ

/-- Calculates the total profit of a partnership --/
def calculate_total_profit (p : Partnership) : ℕ :=
  let x := p.C_profit_share / p.profit_ratio_C
  x * (p.profit_ratio_A + p.profit_ratio_B + p.profit_ratio_C + p.profit_ratio_D)

/-- Theorem stating that for the given partnership, the total profit is 144000 --/
theorem partnership_profit_calculation (p : Partnership)
  (h1 : p.investment_A = 27000)
  (h2 : p.investment_B = 72000)
  (h3 : p.investment_C = 81000)
  (h4 : p.investment_D = 63000)
  (h5 : p.profit_ratio_A = 2)
  (h6 : p.profit_ratio_B = 3)
  (h7 : p.profit_ratio_C = 4)
  (h8 : p.profit_ratio_D = 3)
  (h9 : p.C_profit_share = 48000) :
  calculate_total_profit p = 144000 := by
  sorry

end NUMINAMATH_CALUDE_partnership_profit_calculation_l1004_100470


namespace NUMINAMATH_CALUDE_burrito_cheese_amount_l1004_100434

/-- The amount of cheese (in ounces) required for a burrito -/
def cheese_per_burrito : ℝ := 4

/-- The amount of cheese (in ounces) required for a taco -/
def cheese_per_taco : ℝ := 9

/-- The total amount of cheese (in ounces) required for 7 burritos and 1 taco -/
def total_cheese : ℝ := 37

/-- Theorem stating that the amount of cheese required for a burrito is 4 ounces -/
theorem burrito_cheese_amount :
  cheese_per_burrito = 4 ∧
  cheese_per_taco = 9 ∧
  7 * cheese_per_burrito + cheese_per_taco = total_cheese :=
by sorry

end NUMINAMATH_CALUDE_burrito_cheese_amount_l1004_100434


namespace NUMINAMATH_CALUDE_mysterious_quadratic_polynomial_value_at_zero_l1004_100460

/-- A quadratic polynomial with real coefficients -/
def QuadraticPolynomial (b c : ℝ) : ℝ → ℝ := fun x ↦ x^2 + b*x + c

/-- A polynomial is mysterious if p(p(x))=0 has exactly four real roots, including multiplicities -/
def IsMysteri6ous (p : ℝ → ℝ) : Prop :=
  ∃ (roots : Finset ℝ), (∀ x, x ∈ roots ↔ p (p x) = 0) ∧ roots.card = 4

/-- The sum of roots of a quadratic polynomial -/
def SumOfRoots (b c : ℝ) : ℝ := -b

theorem mysterious_quadratic_polynomial_value_at_zero
  (b c : ℝ)
  (h_mysterious : IsMysteri6ous (QuadraticPolynomial b c))
  (h_minimal_sum : ∀ b' c', IsMysteri6ous (QuadraticPolynomial b' c') → SumOfRoots b c ≤ SumOfRoots b' c') :
  QuadraticPolynomial b c 0 = 4 := by
  sorry

end NUMINAMATH_CALUDE_mysterious_quadratic_polynomial_value_at_zero_l1004_100460


namespace NUMINAMATH_CALUDE_cone_lateral_area_l1004_100404

/-- The lateral area of a cone with slant height 8 cm and base diameter 6 cm is 24π cm² -/
theorem cone_lateral_area (slant_height : ℝ) (base_diameter : ℝ) :
  slant_height = 8 →
  base_diameter = 6 →
  (1 / 2 : ℝ) * π * base_diameter * slant_height = 24 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_area_l1004_100404


namespace NUMINAMATH_CALUDE_system_solution_l1004_100477

theorem system_solution :
  let S := {(x, y) : ℝ × ℝ | x + y = 3 ∧ 2*x - 3*y = 1}
  S = {(2, 1)} := by sorry

end NUMINAMATH_CALUDE_system_solution_l1004_100477


namespace NUMINAMATH_CALUDE_remainder_problem_l1004_100436

theorem remainder_problem (k : ℕ+) (h : ∃ b : ℕ, 120 = b * k^2 + 12) : 
  ∃ q : ℕ, 200 = q * k + 2 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l1004_100436


namespace NUMINAMATH_CALUDE_shaded_areas_equality_l1004_100423

theorem shaded_areas_equality (θ : Real) (h1 : 0 < θ) (h2 : θ < π / 4) :
  (∃ (r : Real), r > 0 ∧ θ * r^2 = (r^2 * Real.tan (2 * θ)) / 2) ↔ Real.tan (2 * θ) = 2 * θ := by
  sorry

end NUMINAMATH_CALUDE_shaded_areas_equality_l1004_100423


namespace NUMINAMATH_CALUDE_parabola_m_values_l1004_100468

-- Define the parabola function
def parabola (a h k : ℝ) (x : ℝ) : ℝ := a * (x - h)^2 + k

-- State the theorem
theorem parabola_m_values (a h k m : ℝ) :
  (parabola a h k (-1) = 0) →
  (parabola a h k 5 = 0) →
  (a * (4 - h + m)^2 + k = 0) →
  (m = -5 ∨ m = 1) :=
by sorry

end NUMINAMATH_CALUDE_parabola_m_values_l1004_100468


namespace NUMINAMATH_CALUDE_e_pow_pi_gt_pi_pow_e_l1004_100492

/-- Prove that e^π > π^e, given that π > e -/
theorem e_pow_pi_gt_pi_pow_e : Real.exp π > π ^ Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_e_pow_pi_gt_pi_pow_e_l1004_100492


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l1004_100424

theorem necessary_but_not_sufficient_condition (a b : ℝ) :
  (a < b → a < b + 1) ∧ ¬(∀ a b : ℝ, a < b + 1 → a < b) := by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l1004_100424


namespace NUMINAMATH_CALUDE_min_value_expression_l1004_100455

theorem min_value_expression (m n : ℝ) (h : m > 2 * n) :
  ∃ (min : ℝ), min = 6 ∧ 
  (∀ x, x = m + (4 * n^2 - 2 * m * n + 9) / (m - 2 * n) → x ≥ min) ∧
  (∃ y, y = m + (4 * n^2 - 2 * m * n + 9) / (m - 2 * n) ∧ y = min) :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l1004_100455


namespace NUMINAMATH_CALUDE_circle_equation_proof_l1004_100473

/-- A circle with center on the x-axis passing through two given points -/
structure CircleOnXAxis where
  center : ℝ  -- x-coordinate of the center
  passesThrough : (ℝ × ℝ) → (ℝ × ℝ) → Prop

/-- The equation of a circle given its center and a point on the circle -/
def circleEquation (h : ℝ) (k : ℝ) (x : ℝ) (y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = ((5 - h)^2 + 2^2)

theorem circle_equation_proof (c : CircleOnXAxis) 
  (h1 : c.passesThrough (5, 2) (-1, 4)) :
  ∀ x y, circleEquation 1 0 x y ↔ (x - 1)^2 + y^2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_proof_l1004_100473


namespace NUMINAMATH_CALUDE_greatest_integer_fraction_inequality_l1004_100486

theorem greatest_integer_fraction_inequality :
  ∀ x : ℤ, (8 : ℚ) / 11 > (x : ℚ) / 17 ↔ x ≤ 12 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_fraction_inequality_l1004_100486


namespace NUMINAMATH_CALUDE_anya_pancakes_l1004_100457

theorem anya_pancakes (x : ℝ) (x_pos : x > 0) : 
  let flipped := x * (2/3)
  let not_burnt := flipped * 0.6
  let not_dropped := not_burnt * 0.8
  not_dropped / x = 0.32 := by sorry

end NUMINAMATH_CALUDE_anya_pancakes_l1004_100457


namespace NUMINAMATH_CALUDE_cube_edge_product_equality_l1004_100479

-- Define a cube as a structure with 12 edges
structure Cube :=
  (edges : Fin 12 → ℕ)

-- Define a predicate to check if the edges contain all numbers from 1 to 12
def validEdges (c : Cube) : Prop :=
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 12 → ∃ i : Fin 12, c.edges i = n

-- Define the top and bottom faces of the cube
def topFace (c : Cube) : Fin 4 → ℕ := λ i => c.edges i
def bottomFace (c : Cube) : Fin 4 → ℕ := λ i => c.edges (i + 8)

-- Define the product of numbers on a face
def faceProduct (face : Fin 4 → ℕ) : ℕ := (face 0) * (face 1) * (face 2) * (face 3)

-- Theorem statement
theorem cube_edge_product_equality :
  ∃ c : Cube, validEdges c ∧ faceProduct (topFace c) = faceProduct (bottomFace c) := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_product_equality_l1004_100479


namespace NUMINAMATH_CALUDE_P_neither_sufficient_nor_necessary_for_Q_l1004_100415

-- Define the quadratic equation
def quadratic_equation (a x : ℝ) : ℝ := x^2 + (a-2)*x + 2*a - 8

-- Define the condition P
def condition_P (a : ℝ) : Prop := -1 < a ∧ a < 1

-- Define the condition Q
def condition_Q (a : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ < 0 ∧
    quadratic_equation a x₁ = 0 ∧ quadratic_equation a x₂ = 0

-- Theorem stating that P is neither sufficient nor necessary for Q
theorem P_neither_sufficient_nor_necessary_for_Q :
  (¬∀ a : ℝ, condition_P a → condition_Q a) ∧
  (¬∀ a : ℝ, condition_Q a → condition_P a) :=
sorry

end NUMINAMATH_CALUDE_P_neither_sufficient_nor_necessary_for_Q_l1004_100415


namespace NUMINAMATH_CALUDE_game_probability_l1004_100488

theorem game_probability (n : ℕ) (p_alex p_mel p_chelsea : ℝ) : 
  n = 8 →
  p_alex = 1/2 →
  p_mel = 3 * p_chelsea →
  p_alex + p_mel + p_chelsea = 1 →
  (n.choose 4 * n.choose 3 * n.choose 1) * p_alex^4 * p_mel^3 * p_chelsea = 945/8192 :=
by sorry

end NUMINAMATH_CALUDE_game_probability_l1004_100488


namespace NUMINAMATH_CALUDE_number_problem_l1004_100422

theorem number_problem (x : ℝ) : x = 456 ↔ 0.5 * x = 0.4 * 120 + 180 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1004_100422


namespace NUMINAMATH_CALUDE_pascal_triangle_30_rows_l1004_100418

/-- The number of elements in the first n rows of Pascal's Triangle -/
def pascal_triangle_elements (n : ℕ) : ℕ := (n + 1) * (n + 2) / 2

/-- Theorem: The number of elements in the first 30 rows of Pascal's Triangle is 465 -/
theorem pascal_triangle_30_rows : pascal_triangle_elements 29 = 465 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_30_rows_l1004_100418


namespace NUMINAMATH_CALUDE_passing_percentage_is_33_percent_l1004_100478

/-- The passing percentage for an exam -/
def passing_percentage (marks_obtained : ℕ) (marks_failed_by : ℕ) (max_marks : ℕ) : ℚ :=
  ((marks_obtained + marks_failed_by : ℚ) / max_marks) * 100

/-- Theorem: The passing percentage is 33% given the problem conditions -/
theorem passing_percentage_is_33_percent : 
  passing_percentage 59 40 300 = 33 := by sorry

end NUMINAMATH_CALUDE_passing_percentage_is_33_percent_l1004_100478


namespace NUMINAMATH_CALUDE_tree_house_wood_needed_l1004_100414

-- Define the components of the tree house
structure TreeHouse where
  pillar_short : ℝ
  pillar_long : ℝ
  wall_short : ℝ
  wall_long : ℝ
  floor_avg : ℝ
  roof_first : ℝ
  roof_diff : ℝ

-- Define the function to calculate total wood needed
def total_wood (t : TreeHouse) : ℝ :=
  -- Pillars
  4 * t.pillar_short + 4 * t.pillar_long +
  -- Walls
  10 * t.wall_short + 10 * t.wall_long +
  -- Floor
  8 * t.floor_avg +
  -- Roof (arithmetic sequence sum formula)
  6 * t.roof_first + 15 * t.roof_diff

-- Theorem statement
theorem tree_house_wood_needed (t : TreeHouse) 
  (h1 : t.pillar_short = 4)
  (h2 : t.pillar_long = 5 * Real.sqrt t.pillar_short)
  (h3 : t.wall_short = 6)
  (h4 : t.wall_long = (2/3) * (t.wall_short ^ (3/2)))
  (h5 : t.floor_avg = 5.5)
  (h6 : t.roof_first = 2 * t.floor_avg)
  (h7 : t.roof_diff = (1/3) * t.pillar_short) :
  total_wood t = 344 := by
  sorry

end NUMINAMATH_CALUDE_tree_house_wood_needed_l1004_100414


namespace NUMINAMATH_CALUDE_factorization_equality_l1004_100409

theorem factorization_equality (a m n : ℝ) :
  -3 * a * m^2 + 12 * a * n^2 = -3 * a * (m + 2*n) * (m - 2*n) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1004_100409


namespace NUMINAMATH_CALUDE_sum_squares_first_12_base6_l1004_100426

-- Define a function to convert a number from base 10 to base 6
def toBase6 (n : ℕ) : List ℕ := sorry

-- Define a function to square a number
def square (n : ℕ) : ℕ := n * n

-- Define a function to sum a list of numbers in base 6
def sumBase6 (list : List (List ℕ)) : List ℕ := sorry

-- Main theorem
theorem sum_squares_first_12_base6 : 
  sumBase6 (List.map (λ n => toBase6 (square n)) (List.range 12)) = [5, 1, 5, 0, 1] := by sorry

end NUMINAMATH_CALUDE_sum_squares_first_12_base6_l1004_100426


namespace NUMINAMATH_CALUDE_lower_right_is_four_l1004_100481

-- Define the grid type
def Grid := Fin 5 → Fin 5 → Fin 5

-- Define a valid grid
def is_valid_grid (g : Grid) : Prop :=
  (∀ i j k, i ≠ j → g i k ≠ g j k) ∧ 
  (∀ i j k, i ≠ j → g k i ≠ g k j)

-- Define the initial configuration
def initial_config (g : Grid) : Prop :=
  g 0 0 = 1 ∧ g 0 2 = 2 ∧ g 0 3 = 3 ∧
  g 1 0 = 2 ∧ g 1 1 = 3 ∧ g 1 4 = 1 ∧
  g 2 1 = 1 ∧ g 2 3 = 5 ∧
  g 4 2 = 4

-- Theorem statement
theorem lower_right_is_four :
  ∀ g : Grid, is_valid_grid g → initial_config g → g 4 4 = 4 :=
sorry

end NUMINAMATH_CALUDE_lower_right_is_four_l1004_100481


namespace NUMINAMATH_CALUDE_park_grass_area_calculation_l1004_100464

/-- Represents the geometry of a circular park with a path and square plot -/
structure ParkGeometry where
  circle_diameter : ℝ
  path_width : ℝ
  square_side : ℝ

/-- Calculates the remaining grass area in the park -/
def remaining_grass_area (park : ParkGeometry) : ℝ :=
  sorry

/-- Theorem stating the remaining grass area for the given park configuration -/
theorem park_grass_area_calculation (park : ParkGeometry) 
  (h1 : park.circle_diameter = 20)
  (h2 : park.path_width = 4)
  (h3 : park.square_side = 6) :
  remaining_grass_area park = 78.21 * Real.pi + 13 := by
  sorry

end NUMINAMATH_CALUDE_park_grass_area_calculation_l1004_100464


namespace NUMINAMATH_CALUDE_base_four_for_64_l1004_100493

theorem base_four_for_64 : ∃! b : ℕ, b > 1 ∧ b ^ 3 ≤ 64 ∧ 64 < b ^ 4 := by
  sorry

end NUMINAMATH_CALUDE_base_four_for_64_l1004_100493


namespace NUMINAMATH_CALUDE_wangwa_smallest_growth_rate_l1004_100498

structure BreedingBase where
  name : String
  growthRate : Float

def liwa : BreedingBase := { name := "Liwa", growthRate := 3.25 }
def wangwa : BreedingBase := { name := "Wangwa", growthRate := -2.75 }
def jiazhuang : BreedingBase := { name := "Jiazhuang", growthRate := 4.6 }
def wuzhuang : BreedingBase := { name := "Wuzhuang", growthRate := -1.76 }

def breedingBases : List BreedingBase := [liwa, wangwa, jiazhuang, wuzhuang]

theorem wangwa_smallest_growth_rate :
  ∀ b ∈ breedingBases, wangwa.growthRate ≤ b.growthRate :=
by sorry

end NUMINAMATH_CALUDE_wangwa_smallest_growth_rate_l1004_100498


namespace NUMINAMATH_CALUDE_cubic_root_sum_cubes_l1004_100494

theorem cubic_root_sum_cubes (a b c : ℝ) : 
  (5 * a^3 + 505 * a + 1010 = 0) →
  (5 * b^3 + 505 * b + 1010 = 0) →
  (5 * c^3 + 505 * c + 1010 = 0) →
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 606 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_cubes_l1004_100494


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1004_100408

theorem quadratic_inequality_solution (a : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 - 2*a*x - 8*a^2 < 0 ↔ x₁ < x ∧ x < x₂) →
  a > 0 →
  x₂ + x₁ = 15 →
  a = 15/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1004_100408
