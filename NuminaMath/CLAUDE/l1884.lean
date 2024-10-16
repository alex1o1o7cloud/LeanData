import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_z_values_l1884_188445

-- Define the function f
def f (x : ℝ) : ℝ := (2*x)^2 - 3*(2*x) + 2

-- State the theorem
theorem sum_of_z_values (z : ℝ) : 
  (∃ z₁ z₂, f z₁ = 4 ∧ f z₂ = 4 ∧ z₁ ≠ z₂ ∧ z₁ + z₂ = 3/4) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_z_values_l1884_188445


namespace NUMINAMATH_CALUDE_units_digit_of_sum_l1884_188402

-- Define the function to get the units digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- State the theorem
theorem units_digit_of_sum : unitsDigit ((56 ^ 78) + (87 ^ 65)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_l1884_188402


namespace NUMINAMATH_CALUDE_grape_yield_after_change_l1884_188458

/-- Represents the number of jars that can be made from one can of juice -/
structure JuiceYield where
  apple : ℚ
  grape : ℚ

/-- Represents the recipe for the beverage -/
structure Recipe where
  apple : ℚ
  grape : ℚ

/-- The initial recipe yield -/
def initial_yield : JuiceYield :=
  { apple := 6,
    grape := 10 }

/-- The changed recipe yield for apple juice -/
def changed_apple_yield : ℚ := 5

/-- Theorem stating that after the recipe change, one can of grape juice makes 15 jars -/
theorem grape_yield_after_change
  (initial : JuiceYield)
  (changed_apple : ℚ)
  (h_initial : initial = initial_yield)
  (h_changed_apple : changed_apple = changed_apple_yield)
  : ∃ (changed : JuiceYield), changed.grape = 15 :=
sorry

end NUMINAMATH_CALUDE_grape_yield_after_change_l1884_188458


namespace NUMINAMATH_CALUDE_fraction_invariance_l1884_188403

theorem fraction_invariance (x y : ℝ) (h : x ≠ y) : x / (x - y) = (2 * x) / (2 * x - 2 * y) := by
  sorry

end NUMINAMATH_CALUDE_fraction_invariance_l1884_188403


namespace NUMINAMATH_CALUDE_chicken_pasta_pieces_is_two_l1884_188487

/-- Represents the number of chicken pieces in different orders and the total needed -/
structure ChickenOrders where
  barbecue_pieces : ℕ
  fried_dinner_pieces : ℕ
  fried_dinner_orders : ℕ
  chicken_pasta_orders : ℕ
  barbecue_orders : ℕ
  total_pieces : ℕ

/-- Calculates the number of chicken pieces in a Chicken Pasta order -/
def chicken_pasta_pieces (orders : ChickenOrders) : ℕ :=
  (orders.total_pieces -
   (orders.fried_dinner_pieces * orders.fried_dinner_orders +
    orders.barbecue_pieces * orders.barbecue_orders)) /
  orders.chicken_pasta_orders

/-- Theorem stating that the number of chicken pieces in a Chicken Pasta order is 2 -/
theorem chicken_pasta_pieces_is_two (orders : ChickenOrders)
  (h1 : orders.barbecue_pieces = 3)
  (h2 : orders.fried_dinner_pieces = 8)
  (h3 : orders.fried_dinner_orders = 2)
  (h4 : orders.chicken_pasta_orders = 6)
  (h5 : orders.barbecue_orders = 3)
  (h6 : orders.total_pieces = 37) :
  chicken_pasta_pieces orders = 2 := by
  sorry

end NUMINAMATH_CALUDE_chicken_pasta_pieces_is_two_l1884_188487


namespace NUMINAMATH_CALUDE_highway_traffic_l1884_188489

/-- The number of vehicles involved in accidents per 100 million vehicles -/
def accident_rate : ℚ := 75

/-- The total number of vehicles involved in accidents -/
def total_accidents : ℕ := 4500

/-- The number of vehicles (in millions) that traveled on the highway -/
def total_vehicles : ℕ := 6000

theorem highway_traffic :
  (accident_rate / 100000000) * (total_vehicles * 1000000) = total_accidents :=
sorry

end NUMINAMATH_CALUDE_highway_traffic_l1884_188489


namespace NUMINAMATH_CALUDE_age_ratio_l1884_188439

theorem age_ratio (current_age : ℕ) (years_ago : ℕ) : 
  current_age = 10 → 
  years_ago = 5 → 
  (current_age : ℚ) / ((current_age - years_ago) : ℚ) = 2 := by
sorry

end NUMINAMATH_CALUDE_age_ratio_l1884_188439


namespace NUMINAMATH_CALUDE_sqrt_seven_minus_fraction_greater_than_reciprocal_l1884_188451

theorem sqrt_seven_minus_fraction_greater_than_reciprocal 
  (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : Real.sqrt 7 - m / n > 0) : 
  Real.sqrt 7 - m / n > 1 / (m * n) := by
sorry

end NUMINAMATH_CALUDE_sqrt_seven_minus_fraction_greater_than_reciprocal_l1884_188451


namespace NUMINAMATH_CALUDE_books_read_together_l1884_188446

theorem books_read_together (tony_books dean_books breanna_books tony_dean_overlap total_different : ℕ)
  (h1 : tony_books = 23)
  (h2 : dean_books = 12)
  (h3 : breanna_books = 17)
  (h4 : tony_dean_overlap = 3)
  (h5 : total_different = 47) :
  tony_books + dean_books + breanna_books - tony_dean_overlap - total_different = 2 :=
by sorry

end NUMINAMATH_CALUDE_books_read_together_l1884_188446


namespace NUMINAMATH_CALUDE_perfect_square_triples_l1884_188476

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

def satisfies_condition (a b c : ℕ) : Prop :=
  is_perfect_square (2^a + 2^b + 2^c + 3)

theorem perfect_square_triples :
  ∀ a b c : ℕ, satisfies_condition a b c ↔ (a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = 3 ∧ b = 2 ∧ c = 1) :=
sorry

end NUMINAMATH_CALUDE_perfect_square_triples_l1884_188476


namespace NUMINAMATH_CALUDE_initial_candies_count_l1884_188421

/-- The number of candies initially in the box -/
def initial_candies : ℕ := sorry

/-- The number of candies Diana took from the box -/
def candies_taken : ℕ := 6

/-- The number of candies left in the box after Diana took some -/
def candies_left : ℕ := 82

/-- Theorem stating that the initial number of candies is 88 -/
theorem initial_candies_count : initial_candies = 88 :=
  by sorry

end NUMINAMATH_CALUDE_initial_candies_count_l1884_188421


namespace NUMINAMATH_CALUDE_first_digit_of_1122001_base_3_in_base_9_l1884_188423

def base_3_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

def first_digit_base_9 (n : Nat) : Nat :=
  if n = 0 then 0
  else
    let log9 := Nat.log 9 n
    n / (9 ^ log9)

theorem first_digit_of_1122001_base_3_in_base_9 :
  let x := base_3_to_10 [1, 0, 0, 2, 2, 1, 1]
  first_digit_base_9 x = 1 := by
  sorry

end NUMINAMATH_CALUDE_first_digit_of_1122001_base_3_in_base_9_l1884_188423


namespace NUMINAMATH_CALUDE_remainder_sum_of_powers_l1884_188450

theorem remainder_sum_of_powers (n : ℕ) : (8^6 + 7^7 + 6^8) % 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_of_powers_l1884_188450


namespace NUMINAMATH_CALUDE_certain_number_base_l1884_188460

theorem certain_number_base (x y : ℕ) (a : ℝ) 
  (h1 : 3^x * a^y = 3^12) 
  (h2 : x - y = 12) 
  (h3 : x = 12) : 
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_base_l1884_188460


namespace NUMINAMATH_CALUDE_mom_tshirt_packages_l1884_188479

/-- The number of t-shirts in each package -/
def package_size : ℕ := 13

/-- The total number of t-shirts mom buys -/
def total_tshirts : ℕ := 39

/-- The number of packages mom will have -/
def num_packages : ℕ := total_tshirts / package_size

theorem mom_tshirt_packages : num_packages = 3 := by
  sorry

end NUMINAMATH_CALUDE_mom_tshirt_packages_l1884_188479


namespace NUMINAMATH_CALUDE_largest_integer_solution_l1884_188436

theorem largest_integer_solution : 
  ∀ x : ℤ, (3 * x - 4 : ℚ) / 2 < x - 1 → x ≤ 1 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integer_solution_l1884_188436


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1884_188463

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, a * x^2 - 2 * a * x + 3 ≤ 0) ↔ (a < 0 ∨ a ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1884_188463


namespace NUMINAMATH_CALUDE_cost_of_three_l1884_188411

/-- Represents the prices of fruits and vegetables -/
structure Prices where
  apples : ℝ
  bananas : ℝ
  cantaloupe : ℝ
  dates : ℝ
  eggplant : ℝ

/-- The total cost of all items is $30 -/
def total_cost (p : Prices) : Prop :=
  p.apples + p.bananas + p.cantaloupe + p.dates + p.eggplant = 30

/-- The carton of dates costs twice as much as the sack of apples -/
def dates_cost (p : Prices) : Prop :=
  p.dates = 2 * p.apples

/-- The price of cantaloupe equals price of apples minus price of bananas -/
def cantaloupe_cost (p : Prices) : Prop :=
  p.cantaloupe = p.apples - p.bananas

/-- The price of eggplant is the sum of apples and bananas prices -/
def eggplant_cost (p : Prices) : Prop :=
  p.eggplant = p.apples + p.bananas

/-- The main theorem: Given the conditions, the cost of bananas, cantaloupe, and eggplant is $12 -/
theorem cost_of_three (p : Prices) 
  (h1 : total_cost p) 
  (h2 : dates_cost p) 
  (h3 : cantaloupe_cost p) 
  (h4 : eggplant_cost p) : 
  p.bananas + p.cantaloupe + p.eggplant = 12 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_three_l1884_188411


namespace NUMINAMATH_CALUDE_tournament_handshakes_l1884_188413

theorem tournament_handshakes (n : ℕ) (m : ℕ) (h : n = 4 ∧ m = 2) :
  let total_players := n * m
  let handshakes_per_player := total_players - m
  (total_players * handshakes_per_player) / 2 = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_tournament_handshakes_l1884_188413


namespace NUMINAMATH_CALUDE_matrix_operation_result_l1884_188493

def A : Matrix (Fin 2) (Fin 2) ℤ := !![5, -3; 6, 4]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![-1, 8; -7, 3]

theorem matrix_operation_result :
  (2 : ℤ) • (A + B) = !![8, 10; -2, 14] := by sorry

end NUMINAMATH_CALUDE_matrix_operation_result_l1884_188493


namespace NUMINAMATH_CALUDE_octal_243_equals_decimal_163_l1884_188419

/-- Converts a base-8 number to base-10 --/
def octal_to_decimal (octal : ℕ) : ℕ :=
  let hundreds := octal / 100
  let tens := (octal % 100) / 10
  let ones := octal % 10
  hundreds * 8^2 + tens * 8^1 + ones * 8^0

/-- The octal number 243 is equal to 163 in decimal --/
theorem octal_243_equals_decimal_163 : octal_to_decimal 243 = 163 := by
  sorry

end NUMINAMATH_CALUDE_octal_243_equals_decimal_163_l1884_188419


namespace NUMINAMATH_CALUDE_machine_worked_two_minutes_l1884_188409

/-- Calculates the working time of a machine given its production rate and total output -/
def machine_working_time (shirts_per_minute : ℕ) (total_shirts : ℕ) : ℚ :=
  (total_shirts : ℚ) / (shirts_per_minute : ℚ)

/-- Proves that a machine making 3 shirts per minute that made 6 shirts worked for 2 minutes -/
theorem machine_worked_two_minutes :
  machine_working_time 3 6 = 2 := by sorry

end NUMINAMATH_CALUDE_machine_worked_two_minutes_l1884_188409


namespace NUMINAMATH_CALUDE_cos_alpha_minus_pi_half_l1884_188426

/-- If the terminal side of angle α passes through the point P(-1, √3), then cos(α - π/2) = √3/2 -/
theorem cos_alpha_minus_pi_half (α : Real) :
  (∃ (x y : Real), x = -1 ∧ y = Real.sqrt 3 ∧ x = Real.cos α * Real.cos 0 - Real.sin α * Real.sin 0 ∧ 
                    y = Real.sin α * Real.cos 0 + Real.cos α * Real.sin 0) →
  Real.cos (α - π/2) = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_minus_pi_half_l1884_188426


namespace NUMINAMATH_CALUDE_no_solution_sqrt_eq_negative_l1884_188422

theorem no_solution_sqrt_eq_negative :
  ¬∃ x : ℝ, Real.sqrt (5 - x) = -3 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_sqrt_eq_negative_l1884_188422


namespace NUMINAMATH_CALUDE_easter_egg_distribution_l1884_188431

theorem easter_egg_distribution (total_people : ℕ) (eggs_per_person : ℕ) (num_baskets : ℕ) :
  total_people = 20 →
  eggs_per_person = 9 →
  num_baskets = 15 →
  (total_people * eggs_per_person) / num_baskets = 12 := by
sorry

end NUMINAMATH_CALUDE_easter_egg_distribution_l1884_188431


namespace NUMINAMATH_CALUDE_problem_solution_l1884_188418

theorem problem_solution (f : ℝ → ℝ) (m a b c : ℝ) 
  (h1 : ∀ x, f x = |x - m|)
  (h2 : Set.Icc (-1) 5 = {x | f x ≤ 3})
  (h3 : a - 2*b + 2*c = m) : 
  m = 2 ∧ (∃ (min : ℝ), min = 4/9 ∧ a^2 + b^2 + c^2 ≥ min) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1884_188418


namespace NUMINAMATH_CALUDE_floor_equation_solution_l1884_188494

theorem floor_equation_solution :
  ∃! x : ℝ, ⌊x⌋ + x + (1/2 : ℝ) = 20.7 :=
by sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l1884_188494


namespace NUMINAMATH_CALUDE_units_digit_sum_base_8_l1884_188468

-- Define a function to get the units digit in base 8
def units_digit_base_8 (n : ℕ) : ℕ := n % 8

-- Define the numbers in base 8
def num1 : ℕ := 64
def num2 : ℕ := 34

-- Theorem statement
theorem units_digit_sum_base_8 :
  units_digit_base_8 (num1 + num2) = 0 :=
by sorry

end NUMINAMATH_CALUDE_units_digit_sum_base_8_l1884_188468


namespace NUMINAMATH_CALUDE_high_correlation_implies_r_close_to_one_l1884_188459

/-- Represents the correlation coefficient between two variables -/
def correlation_coefficient (x y : ℝ → ℝ) : ℝ := sorry

/-- Represents the degree of linear correlation between two variables -/
def linear_correlation_degree (x y : ℝ → ℝ) : ℝ := sorry

/-- A high degree of linear correlation -/
def high_correlation : ℝ := sorry

theorem high_correlation_implies_r_close_to_one (x y : ℝ → ℝ) :
  linear_correlation_degree x y ≥ high_correlation →
  ∀ ε > 0, ∃ δ > 0, linear_correlation_degree x y > 1 - δ →
  |correlation_coefficient x y| > 1 - ε :=
sorry

end NUMINAMATH_CALUDE_high_correlation_implies_r_close_to_one_l1884_188459


namespace NUMINAMATH_CALUDE_remainder_problem_l1884_188412

theorem remainder_problem (p : Nat) (h : Prime p) (h1 : p = 13) :
  (7 * 12^24 + 2^24) % p = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1884_188412


namespace NUMINAMATH_CALUDE_correct_stratified_sample_l1884_188434

/-- Represents the number of water heaters in a sample from a specific factory -/
structure FactorySample where
  total : ℕ
  factory_a : ℕ
  factory_b : ℕ
  sample_size : ℕ

/-- Calculates the stratified sample size for a factory -/
def stratified_sample_size (total : ℕ) (factory : ℕ) (sample_size : ℕ) : ℕ :=
  (factory * sample_size) / total

/-- Theorem stating the correct stratified sample sizes for factories A and B -/
theorem correct_stratified_sample (fs : FactorySample) 
  (h1 : fs.total = 98)
  (h2 : fs.factory_a = 56)
  (h3 : fs.factory_b = 42)
  (h4 : fs.sample_size = 14) :
  stratified_sample_size fs.total fs.factory_a fs.sample_size = 8 ∧
  stratified_sample_size fs.total fs.factory_b fs.sample_size = 6 := by
  sorry

end NUMINAMATH_CALUDE_correct_stratified_sample_l1884_188434


namespace NUMINAMATH_CALUDE_inequality_chain_l1884_188484

theorem inequality_chain (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) :
  a * b > a * b^2 ∧ a * b^2 > a := by sorry

end NUMINAMATH_CALUDE_inequality_chain_l1884_188484


namespace NUMINAMATH_CALUDE_line_equation_l1884_188408

/-- A line passing through the point (-2, 5) with slope -3/4 has the equation 3x + 4y - 14 = 0. -/
theorem line_equation (x y : ℝ) : 
  (∃ (L : Set (ℝ × ℝ)), 
    ((-2, 5) ∈ L) ∧ 
    (∀ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ L → (x₂, y₂) ∈ L → x₁ ≠ x₂ → (y₂ - y₁) / (x₂ - x₁) = -3/4) ∧
    ((x, y) ∈ L ↔ 3*x + 4*y - 14 = 0)) :=
by
  sorry

end NUMINAMATH_CALUDE_line_equation_l1884_188408


namespace NUMINAMATH_CALUDE_power_function_passes_through_one_l1884_188407

theorem power_function_passes_through_one (α : ℝ) : 
  let f : ℝ → ℝ := fun x ↦ x ^ α
  f 1 = 1 := by sorry

end NUMINAMATH_CALUDE_power_function_passes_through_one_l1884_188407


namespace NUMINAMATH_CALUDE_simplify_sqrt_sum_l1884_188461

theorem simplify_sqrt_sum : 
  Real.sqrt 1 + Real.sqrt (1 + 2) + Real.sqrt (1 + 2 + 3) + Real.sqrt (1 + 2 + 3 + 4) = 
  1 + Real.sqrt 3 + Real.sqrt 6 + Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_sum_l1884_188461


namespace NUMINAMATH_CALUDE_exists_same_answer_question_l1884_188485

/-- Represents a person who either always tells the truth or always lies -/
inductive Person
| TruthTeller
| Liar

/-- Represents a question that can be asked to a person -/
def Question := Type

/-- Represents an answer to a question -/
def Answer := Type

/-- The response function that determines how a person answers a question -/
def respond (p : Person) (q : Question) : Answer :=
  sorry

/-- Theorem stating that there exists a question that elicits the same answer from both a truth-teller and a liar -/
theorem exists_same_answer_question :
  ∃ (q : Question), ∀ (p1 p2 : Person), p1 ≠ p2 → respond p1 q = respond p2 q :=
sorry

end NUMINAMATH_CALUDE_exists_same_answer_question_l1884_188485


namespace NUMINAMATH_CALUDE_mike_mark_height_difference_l1884_188443

/-- Converts feet and inches to total inches -/
def height_to_inches (feet : ℕ) (inches : ℕ) : ℕ := feet * 12 + inches

/-- The height difference between two people in inches -/
def height_difference (height1 : ℕ) (height2 : ℕ) : ℕ := 
  if height1 ≥ height2 then height1 - height2 else height2 - height1

theorem mike_mark_height_difference :
  let mark_height := height_to_inches 5 3
  let mike_height := height_to_inches 6 1
  height_difference mike_height mark_height = 10 := by
sorry

end NUMINAMATH_CALUDE_mike_mark_height_difference_l1884_188443


namespace NUMINAMATH_CALUDE_chess_tournament_players_l1884_188482

theorem chess_tournament_players (total_games : ℕ) (h : total_games = 56) :
  ∃ (n : ℕ), n > 0 ∧ total_games = n * (n - 1) ∧ n = 14 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_players_l1884_188482


namespace NUMINAMATH_CALUDE_evaluate_expression_l1884_188455

theorem evaluate_expression : 225 + 2 * 15 * 8 + 64 = 529 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1884_188455


namespace NUMINAMATH_CALUDE_constant_dot_product_implies_ratio_l1884_188417

/-- Given that O is the origin, P is any point on the line 2x + y - 2 = 0,
    a = (m, n) is a non-zero vector, and the dot product of OP and a is always constant,
    then m/n = 2. -/
theorem constant_dot_product_implies_ratio (m n : ℝ) :
  (∀ x y : ℝ, 2 * x + y - 2 = 0 →
    ∃ k : ℝ, ∀ x' y' : ℝ, 2 * x' + y' - 2 = 0 →
      m * x' + n * y' = k) →
  m ≠ 0 ∨ n ≠ 0 →
  m / n = 2 :=
by sorry

end NUMINAMATH_CALUDE_constant_dot_product_implies_ratio_l1884_188417


namespace NUMINAMATH_CALUDE_function_identically_zero_l1884_188449

/-- A function satisfying the given conditions is identically zero. -/
theorem function_identically_zero (f : ℝ → ℝ) 
  (h_diff : Differentiable ℝ f)
  (h_zero : f 0 = 0)
  (h_bound : ∀ x : ℝ, 0 < |f x| → |f x| < (1/2) → 
    |deriv f x| ≤ |f x * Real.log (|f x|)|) : 
  ∀ x : ℝ, f x = 0 := by
sorry

end NUMINAMATH_CALUDE_function_identically_zero_l1884_188449


namespace NUMINAMATH_CALUDE_pulley_system_theorem_l1884_188481

/-- Represents the configuration of three pulleys --/
structure PulleySystem where
  r : ℝ  -- radius of pulleys
  d12 : ℝ  -- distance between O₁ and O₂
  d13 : ℝ  -- distance between O₁ and O₃
  h : ℝ  -- height of O₃ above the plane of O₁ and O₂

/-- Calculates the possible belt lengths for the pulley system --/
def beltLengths (p : PulleySystem) : Set ℝ :=
  { 32 + 4 * Real.pi, 22 + 2 * Real.sqrt 97 + 4 * Real.pi }

/-- Checks if a given cord length is always sufficient --/
def isAlwaysSufficient (p : PulleySystem) (cordLength : ℝ) : Prop :=
  ∀ l ∈ beltLengths p, l ≤ cordLength

theorem pulley_system_theorem (p : PulleySystem) 
    (h1 : p.r = 2)
    (h2 : p.d12 = 12)
    (h3 : p.d13 = 10)
    (h4 : p.h = 8) :
    (beltLengths p = { 32 + 4 * Real.pi, 22 + 2 * Real.sqrt 97 + 4 * Real.pi }) ∧
    (¬ isAlwaysSufficient p 54) := by
  sorry

end NUMINAMATH_CALUDE_pulley_system_theorem_l1884_188481


namespace NUMINAMATH_CALUDE_unique_triple_l1884_188427

/-- Least common multiple of two positive integers -/
def lcm (x y : ℕ+) : ℕ+ := sorry

/-- The number of ordered triples (a,b,c) satisfying the given LCM conditions -/
def count_triples : ℕ := sorry

theorem unique_triple : count_triples = 1 := by sorry

end NUMINAMATH_CALUDE_unique_triple_l1884_188427


namespace NUMINAMATH_CALUDE_sum_of_roots_cubic_sum_of_roots_specific_cubic_l1884_188469

theorem sum_of_roots_cubic : ∀ (a b c d : ℝ),
  (∃ x y z : ℝ, a * x^3 + b * x^2 + c * x + d = 0 ∧
                a * y^3 + b * y^2 + c * y + d = 0 ∧
                a * z^3 + b * z^2 + c * z + d = 0 ∧
                (∀ w : ℝ, a * w^3 + b * w^2 + c * w + d = 0 → w = x ∨ w = y ∨ w = z)) →
  x + y + z = -b / a :=
by sorry

theorem sum_of_roots_specific_cubic :
  ∃ x y z : ℝ, x^3 - 3*x^2 - 12*x - 7 = 0 ∧
              y^3 - 3*y^2 - 12*y - 7 = 0 ∧
              z^3 - 3*z^2 - 12*z - 7 = 0 ∧
              (∀ w : ℝ, w^3 - 3*w^2 - 12*w - 7 = 0 → w = x ∨ w = y ∨ w = z) ∧
              x + y + z = 3 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_cubic_sum_of_roots_specific_cubic_l1884_188469


namespace NUMINAMATH_CALUDE_no_square_root_among_options_l1884_188428

theorem no_square_root_among_options : ∃ (x : ℝ), x ^ 2 = 0 ∧
                                       ∃ (x : ℝ), x ^ 2 = (-2)^2 ∧
                                       ∃ (x : ℝ), x ^ 2 = |9| ∧
                                       ¬∃ (x : ℝ), x ^ 2 = -|(-5)| := by
  sorry

#check no_square_root_among_options

end NUMINAMATH_CALUDE_no_square_root_among_options_l1884_188428


namespace NUMINAMATH_CALUDE_circle_symmetry_l1884_188454

/-- The equation of the original circle -/
def original_circle (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 2*y + 1 = 0

/-- The equation of the line of symmetry -/
def symmetry_line (x y : ℝ) : Prop :=
  x - y + 3 = 0

/-- The equation of the symmetric circle -/
def symmetric_circle (x y : ℝ) : Prop :=
  (x + 2)^2 + (y - 2)^2 = 1

/-- Theorem stating that the symmetric_circle is indeed symmetric to the original_circle
    with respect to the symmetry_line -/
theorem circle_symmetry :
  ∀ (x y : ℝ), original_circle x y ↔ 
  ∃ (x' y' : ℝ), symmetric_circle x' y' ∧ 
  ((x + x')/2 - (y + y')/2 + 3 = 0) ∧
  ((y' - y)/(x' - x) = -1) :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l1884_188454


namespace NUMINAMATH_CALUDE_divisors_of_ten_factorial_greater_than_nine_factorial_l1884_188456

theorem divisors_of_ten_factorial_greater_than_nine_factorial : 
  (Finset.filter (fun d => d > Nat.factorial 9 ∧ Nat.factorial 10 % d = 0) 
    (Finset.range (Nat.factorial 10 + 1))).card = 9 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_ten_factorial_greater_than_nine_factorial_l1884_188456


namespace NUMINAMATH_CALUDE_solve_equation_and_evaluate_l1884_188486

theorem solve_equation_and_evaluate (x : ℚ) : 
  (4 * x - 3 = 13 * x + 12) → (5 * (x + 4) = 35 / 3) := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_and_evaluate_l1884_188486


namespace NUMINAMATH_CALUDE_high_sulfur_oil_count_l1884_188414

/-- Represents the properties of an oil sample set -/
structure OilSampleSet where
  total_samples : Nat
  heavy_oil_prob : Rat
  light_low_sulfur_prob : Rat

/-- Theorem stating the number of high-sulfur oil samples in a given set -/
theorem high_sulfur_oil_count (s : OilSampleSet)
  (h1 : s.total_samples % 7 = 0)
  (h2 : s.total_samples ≤ 100 ∧ ∀ n, n % 7 = 0 → n ≤ 100 → s.total_samples ≥ n)
  (h3 : s.heavy_oil_prob = 1 / 7)
  (h4 : s.light_low_sulfur_prob = 9 / 14) :
  (s.total_samples : Rat) * s.heavy_oil_prob +
  (s.total_samples : Rat) * (1 - s.heavy_oil_prob) * (1 - s.light_low_sulfur_prob) = 44 := by
  sorry

end NUMINAMATH_CALUDE_high_sulfur_oil_count_l1884_188414


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1884_188496

theorem quadratic_equation_roots : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (x₁^2 + x₁ - 1 = 0) ∧ (x₂^2 + x₂ - 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1884_188496


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l1884_188410

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n m : ℕ, a (n + m) - a n = m * (a 2 - a 1)

theorem arithmetic_sequence_formula (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 = 2 →
  (∀ n : ℕ, a n < a (n + 1)) →
  (a 2 / a 1 = a 4 / a 2) →
  ∀ n : ℕ, a n = 2 * n :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l1884_188410


namespace NUMINAMATH_CALUDE_display_rows_l1884_188444

/-- Represents the number of cans in a row given its position from the top. -/
def cans_in_row (n : ℕ) : ℕ := 2 + 3 * (n - 1)

/-- Represents the total number of cans in the first n rows. -/
def total_cans (n : ℕ) : ℕ := n * (cans_in_row 1 + cans_in_row n) / 2

/-- The number of rows in the display is 10, given the conditions. -/
theorem display_rows :
  ∃ (n : ℕ), n > 0 ∧ total_cans n = 145 ∧ n = 10 :=
sorry

end NUMINAMATH_CALUDE_display_rows_l1884_188444


namespace NUMINAMATH_CALUDE_solution_value_l1884_188441

theorem solution_value (x y m : ℝ) : 
  x = 2 ∧ y = -1 ∧ 2*x - 3*y = m → m = 7 := by sorry

end NUMINAMATH_CALUDE_solution_value_l1884_188441


namespace NUMINAMATH_CALUDE_count_words_with_e_l1884_188429

/-- The number of letters in our alphabet -/
def n : ℕ := 5

/-- The length of the words we're creating -/
def k : ℕ := 4

/-- The number of letters in our alphabet excluding E -/
def m : ℕ := 4

/-- The number of 4-letter words that can be made from 5 letters (A, B, C, D, E) with repetition allowed -/
def total_words : ℕ := n ^ k

/-- The number of 4-letter words that can be made from 4 letters (A, B, C, D) with repetition allowed -/
def words_without_e : ℕ := m ^ k

/-- The number of 4-letter words that can be made from 5 letters (A, B, C, D, E) with repetition allowed and using E at least once -/
def words_with_e : ℕ := total_words - words_without_e

theorem count_words_with_e : words_with_e = 369 := by
  sorry

end NUMINAMATH_CALUDE_count_words_with_e_l1884_188429


namespace NUMINAMATH_CALUDE_banana_orange_relation_bananas_to_oranges_l1884_188440

/-- The value of one banana in terms of oranges -/
def banana_value : ℚ := 1

/-- The given relationship between bananas and oranges -/
theorem banana_orange_relation : (3/4 : ℚ) * 16 * banana_value = 12 := by sorry

/-- Theorem to prove: If 3/4 of 16 bananas are worth 12 oranges, 
    then 1/3 of 9 bananas are worth 3 oranges -/
theorem bananas_to_oranges : 
  ((1/3 : ℚ) * 9 * banana_value = 3) := by sorry

end NUMINAMATH_CALUDE_banana_orange_relation_bananas_to_oranges_l1884_188440


namespace NUMINAMATH_CALUDE_limit_p_n_sqrt_n_l1884_188406

/-- The probability that the sum of two randomly selected integers from {1,2,...,n} is a perfect square -/
def p (n : ℕ) : ℝ := sorry

/-- The main theorem stating that the limit of p_n√n as n approaches infinity is 2/3 -/
theorem limit_p_n_sqrt_n :
  ∃ (L : ℝ), L = 2/3 ∧ ∀ ε > 0, ∃ N, ∀ n ≥ N, |p n * Real.sqrt n - L| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_p_n_sqrt_n_l1884_188406


namespace NUMINAMATH_CALUDE_rectangle_dimension_increase_l1884_188420

theorem rectangle_dimension_increase (L B : ℝ) (L' B' : ℝ) (h1 : B' = 1.3 * B) (h2 : L' * B' = 1.43 * L * B) : L' = 1.1 * L := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimension_increase_l1884_188420


namespace NUMINAMATH_CALUDE_square_value_l1884_188442

theorem square_value : ∃ (square : ℚ), (7863 : ℚ) / 13 = 604 + square / 13 ∧ square = 11 := by
  sorry

end NUMINAMATH_CALUDE_square_value_l1884_188442


namespace NUMINAMATH_CALUDE_rival_to_jessie_award_ratio_l1884_188488

/-- Given that Scott won 4 awards, Jessie won 3 times as many awards as Scott,
    and the rival won 24 awards, prove that the ratio of awards won by the rival
    to Jessie is 2:1. -/
theorem rival_to_jessie_award_ratio :
  let scott_awards : ℕ := 4
  let jessie_awards : ℕ := 3 * scott_awards
  let rival_awards : ℕ := 24
  (rival_awards : ℚ) / jessie_awards = 2 := by sorry

end NUMINAMATH_CALUDE_rival_to_jessie_award_ratio_l1884_188488


namespace NUMINAMATH_CALUDE_chocolate_bar_breaks_l1884_188491

/-- Represents a rectangular chocolate bar -/
structure ChocolateBar where
  rows : ℕ
  cols : ℕ

/-- Calculates the minimum number of breaks required to separate a chocolate bar into individual pieces -/
def min_breaks (bar : ChocolateBar) : ℕ :=
  (bar.rows - 1) * bar.cols + (bar.cols - 1)

theorem chocolate_bar_breaks (bar : ChocolateBar) (h1 : bar.rows = 5) (h2 : bar.cols = 8) :
  min_breaks bar = 39 := by
  sorry

#eval min_breaks ⟨5, 8⟩

end NUMINAMATH_CALUDE_chocolate_bar_breaks_l1884_188491


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1884_188492

def A : Set ℕ := {2, 4}
def B : Set ℕ := {3, 4}

theorem intersection_of_A_and_B : A ∩ B = {4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1884_188492


namespace NUMINAMATH_CALUDE_shopping_trip_percentage_l1884_188473

/-- Represents the percentage of the total amount spent on other items -/
def percentage_other : ℝ := sorry

theorem shopping_trip_percentage :
  let total_amount : ℝ := 100 -- Assume total amount is 100 for percentage calculations
  let clothing_percent : ℝ := 60
  let food_percent : ℝ := 10
  let clothing_tax_rate : ℝ := 4
  let other_tax_rate : ℝ := 8
  let total_tax_percent : ℝ := 4.8

  -- Condition 1, 2, and 3
  clothing_percent + food_percent + percentage_other = total_amount ∧
  -- Condition 4, 5, and 6 (tax calculations)
  clothing_percent * clothing_tax_rate / 100 + percentage_other * other_tax_rate / 100 =
    total_tax_percent ∧
  -- Conclusion
  percentage_other = 30 := by sorry

end NUMINAMATH_CALUDE_shopping_trip_percentage_l1884_188473


namespace NUMINAMATH_CALUDE_digit_2457_is_5_l1884_188470

/-- The decimal number constructed by concatenating integers from 1 to 999 -/
def x : ℝ := sorry

/-- The nth digit after the decimal point in the number x -/
def digit_at (n : ℕ) : ℕ := sorry

/-- The theorem stating that the 2457th digit of x is 5 -/
theorem digit_2457_is_5 : digit_at 2457 = 5 := by sorry

end NUMINAMATH_CALUDE_digit_2457_is_5_l1884_188470


namespace NUMINAMATH_CALUDE_max_xy_value_max_xy_attained_l1884_188433

theorem max_xy_value (x y : ℕ+) (h : 7 * x + 4 * y = 140) : x * y ≤ 168 := by
  sorry

theorem max_xy_attained : ∃ (x y : ℕ+), 7 * x + 4 * y = 140 ∧ x * y = 168 := by
  sorry

end NUMINAMATH_CALUDE_max_xy_value_max_xy_attained_l1884_188433


namespace NUMINAMATH_CALUDE_simplified_cow_bull_ratio_l1884_188405

/-- Represents the number of cattle on the farm -/
def total_cattle : ℕ := 555

/-- Represents the number of bulls on the farm -/
def bulls : ℕ := 405

/-- Calculates the number of cows on the farm -/
def cows : ℕ := total_cattle - bulls

/-- Represents the ratio of cows to bulls as a pair of natural numbers -/
def cow_bull_ratio : ℕ × ℕ := (cows, bulls)

/-- The theorem stating that the simplified ratio of cows to bulls is 10:27 -/
theorem simplified_cow_bull_ratio : 
  ∃ (k : ℕ), k > 0 ∧ cow_bull_ratio.1 = 10 * k ∧ cow_bull_ratio.2 = 27 * k := by
  sorry

end NUMINAMATH_CALUDE_simplified_cow_bull_ratio_l1884_188405


namespace NUMINAMATH_CALUDE_motorcycle_journey_time_ratio_l1884_188490

/-- Proves that the time taken to travel from A to B is 2 times the time taken to travel from B to C -/
theorem motorcycle_journey_time_ratio :
  ∀ (total_distance AB_distance BC_distance average_speed : ℝ),
  total_distance = 180 →
  AB_distance = 120 →
  BC_distance = 60 →
  average_speed = 20 →
  AB_distance = 2 * BC_distance →
  ∃ (AB_time BC_time : ℝ),
    AB_time > 0 ∧ BC_time > 0 ∧
    AB_time + BC_time = total_distance / average_speed ∧
    AB_time = AB_distance / average_speed ∧
    BC_time = BC_distance / average_speed ∧
    AB_time = 2 * BC_time :=
by sorry

end NUMINAMATH_CALUDE_motorcycle_journey_time_ratio_l1884_188490


namespace NUMINAMATH_CALUDE_factorization_x_cubed_minus_x_l1884_188472

/-- Factorization of x^3 - x --/
theorem factorization_x_cubed_minus_x :
  ∀ x : ℝ, x^3 - x = x * (x + 1) * (x - 1) := by sorry

end NUMINAMATH_CALUDE_factorization_x_cubed_minus_x_l1884_188472


namespace NUMINAMATH_CALUDE_discount_rate_is_four_percent_l1884_188415

def marked_price : ℝ := 125
def selling_price : ℝ := 120

theorem discount_rate_is_four_percent :
  (marked_price - selling_price) / marked_price * 100 = 4 := by
  sorry

end NUMINAMATH_CALUDE_discount_rate_is_four_percent_l1884_188415


namespace NUMINAMATH_CALUDE_num_children_picked_apples_l1884_188466

/-- The number of baskets -/
def num_baskets : ℕ := 11

/-- The sum of apples picked by each child from all baskets -/
def apples_per_child : ℕ := (num_baskets * (num_baskets + 1)) / 2

/-- The total number of apples picked by all children -/
def total_apples_picked : ℕ := 660

/-- Theorem stating that the number of children who picked apples is 10 -/
theorem num_children_picked_apples : 
  total_apples_picked / apples_per_child = 10 := by
  sorry

end NUMINAMATH_CALUDE_num_children_picked_apples_l1884_188466


namespace NUMINAMATH_CALUDE_no_valid_balanced_coloring_l1884_188477

/-- A chessboard is a 2D grid of squares that can be colored black or white -/
def Chessboard := Fin 1900 → Fin 1900 → Bool

/-- A point on the chessboard -/
def Point := Fin 1900 × Fin 1900

/-- The center point of the chessboard -/
def center : Point := (949, 949)

/-- Two points are symmetric if they are equidistant from the center in opposite directions -/
def symmetric (p q : Point) : Prop :=
  p.1 + q.1 = 2 * center.1 ∧ p.2 + q.2 = 2 * center.2

/-- A valid coloring satisfies the symmetry condition -/
def valid_coloring (c : Chessboard) : Prop :=
  ∀ p q : Point, symmetric p q → c p.1 p.2 ≠ c q.1 q.2

/-- A balanced coloring has an equal number of black and white squares in each row and column -/
def balanced_coloring (c : Chessboard) : Prop :=
  (∀ i : Fin 1900, (Finset.filter (λ j => c i j) Finset.univ).card = 950) ∧
  (∀ j : Fin 1900, (Finset.filter (λ i => c i j) Finset.univ).card = 950)

/-- The main theorem: it's impossible to have a valid and balanced coloring -/
theorem no_valid_balanced_coloring :
  ¬∃ c : Chessboard, valid_coloring c ∧ balanced_coloring c := by
  sorry

end NUMINAMATH_CALUDE_no_valid_balanced_coloring_l1884_188477


namespace NUMINAMATH_CALUDE_problem_solution_l1884_188432

theorem problem_solution : 
  let A : ℤ := -5 * -3
  let B : ℤ := 2 - 2
  A + B = 15 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l1884_188432


namespace NUMINAMATH_CALUDE_evaluation_ratio_l1884_188424

def relevance_percentage : ℚ := 45 / 100
def language_percentage : ℚ := 25 / 100
def structure_percentage : ℚ := 30 / 100

theorem evaluation_ratio :
  let r := relevance_percentage
  let l := language_percentage
  let s := structure_percentage
  let gcd := (r * 100).num.gcd ((l * 100).num.gcd (s * 100).num)
  ((r * 100).num / gcd, (l * 100).num / gcd, (s * 100).num / gcd) = (9, 5, 6) := by
  sorry

end NUMINAMATH_CALUDE_evaluation_ratio_l1884_188424


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l1884_188404

def f (x : ℝ) : ℝ := x^2 - 4*x + 2

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_formula (a : ℕ → ℝ) (x : ℝ) :
  arithmetic_sequence a →
  a 1 = f (x + 1) →
  a 2 = 0 →
  a 3 = f (x - 1) →
  (∀ n : ℕ, a n = 2*n - 4) ∨ (∀ n : ℕ, a n = 4 - 2*n) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l1884_188404


namespace NUMINAMATH_CALUDE_g_at_negative_three_l1884_188464

-- Define the property of g being a rational function satisfying the given equation
def is_valid_g (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → 4 * g (1 / x) + 3 * g x / x = 3 * x^2

-- State the theorem
theorem g_at_negative_three (g : ℝ → ℝ) (h : is_valid_g g) : g (-3) = 247 / 39 := by
  sorry

end NUMINAMATH_CALUDE_g_at_negative_three_l1884_188464


namespace NUMINAMATH_CALUDE_fraction_simplification_l1884_188457

theorem fraction_simplification :
  (30 : ℚ) / 35 * 21 / 45 * 70 / 63 - 2 / 3 = -8 / 15 := by
sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1884_188457


namespace NUMINAMATH_CALUDE_john_needs_thirteen_more_l1884_188483

def saturday_earnings : ℕ := 18
def sunday_earnings : ℕ := saturday_earnings / 2
def previous_weekend_earnings : ℕ := 20
def pogo_stick_cost : ℕ := 60

theorem john_needs_thirteen_more : 
  pogo_stick_cost - (saturday_earnings + sunday_earnings + previous_weekend_earnings) = 13 := by
  sorry

end NUMINAMATH_CALUDE_john_needs_thirteen_more_l1884_188483


namespace NUMINAMATH_CALUDE_paint_set_cost_l1884_188462

def total_cost (has : ℝ) (needs : ℝ) : ℝ := has + needs
def paintbrush_cost : ℝ := 1.50
def easel_cost : ℝ := 12.65
def albert_has : ℝ := 6.50
def albert_needs : ℝ := 12.00

theorem paint_set_cost :
  total_cost albert_has albert_needs - (paintbrush_cost + easel_cost) = 4.35 := by
  sorry

end NUMINAMATH_CALUDE_paint_set_cost_l1884_188462


namespace NUMINAMATH_CALUDE_exp_inequality_l1884_188474

theorem exp_inequality (a b : ℝ) (h : a > b) : Real.exp (-a) - Real.exp (-b) < 0 := by
  sorry

end NUMINAMATH_CALUDE_exp_inequality_l1884_188474


namespace NUMINAMATH_CALUDE_alcohol_water_ratio_three_fifths_l1884_188435

/-- Represents a mixture of alcohol and water -/
structure Mixture where
  alcohol_fraction : ℚ
  water_fraction : ℚ
  sum_is_one : alcohol_fraction + water_fraction = 1

/-- The ratio of alcohol to water in a mixture -/
def alcohol_to_water_ratio (m : Mixture) : ℚ := m.alcohol_fraction / m.water_fraction

/-- Theorem stating that for a mixture with 3/5 alcohol and 2/5 water, 
    the ratio of alcohol to water is 3:2 -/
theorem alcohol_water_ratio_three_fifths 
  (m : Mixture) 
  (h1 : m.alcohol_fraction = 3/5) 
  (h2 : m.water_fraction = 2/5) : 
  alcohol_to_water_ratio m = 3/2 := by
  sorry


end NUMINAMATH_CALUDE_alcohol_water_ratio_three_fifths_l1884_188435


namespace NUMINAMATH_CALUDE_parking_lot_wheels_l1884_188471

/-- Calculates the total number of wheels in a parking lot with specific vehicles and conditions -/
theorem parking_lot_wheels : 
  let cars := 14
  let bikes := 5
  let unicycles := 3
  let twelve_wheeler_trucks := 2
  let eighteen_wheeler_truck := 1
  let cars_with_missing_wheel := 2
  let truck_with_damaged_wheels := 1
  let damaged_wheels := 3

  let car_wheels := cars * 4 - cars_with_missing_wheel * 1
  let bike_wheels := bikes * 2
  let unicycle_wheels := unicycles * 1
  let twelve_wheeler_truck_wheels := twelve_wheeler_trucks * 12 - damaged_wheels
  let eighteen_wheeler_truck_wheels := eighteen_wheeler_truck * 18

  car_wheels + bike_wheels + unicycle_wheels + twelve_wheeler_truck_wheels + eighteen_wheeler_truck_wheels = 106 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_wheels_l1884_188471


namespace NUMINAMATH_CALUDE_angle_measure_proof_l1884_188453

theorem angle_measure_proof (x : ℝ) : 
  x + (3 * x - 10) = 180 → x = 47.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l1884_188453


namespace NUMINAMATH_CALUDE_taps_teps_equivalence_l1884_188475

/-- Given the equivalences between taps, tops, and teps, prove that 15 taps are equivalent to 48 teps -/
theorem taps_teps_equivalence (tap top tep : ℕ → ℚ) 
  (h1 : 5 * tap 1 = 4 * top 1)  -- 5 taps are equivalent to 4 tops
  (h2 : 3 * top 1 = 12 * tep 1) -- 3 tops are equivalent to 12 teps
  : 15 * tap 1 = 48 * tep 1 := by
  sorry

end NUMINAMATH_CALUDE_taps_teps_equivalence_l1884_188475


namespace NUMINAMATH_CALUDE_fence_poles_count_l1884_188425

-- Define the parameters
def total_path_length : ℕ := 900
def bridge_length : ℕ := 42
def pole_spacing : ℕ := 6

-- Define the function to calculate the number of fence poles
def fence_poles : ℕ :=
  let path_to_line := total_path_length - bridge_length
  let poles_one_side := path_to_line / pole_spacing
  2 * poles_one_side

-- Theorem statement
theorem fence_poles_count : fence_poles = 286 := by
  sorry

end NUMINAMATH_CALUDE_fence_poles_count_l1884_188425


namespace NUMINAMATH_CALUDE_heart_diamond_spade_probability_l1884_188430

/-- Probability of drawing a heart, then a diamond, then a spade from a standard 52-card deck -/
theorem heart_diamond_spade_probability : 
  let total_cards : ℕ := 52
  let hearts : ℕ := 13
  let diamonds : ℕ := 13
  let spades : ℕ := 13
  (hearts : ℚ) / total_cards * 
  (diamonds : ℚ) / (total_cards - 1) * 
  (spades : ℚ) / (total_cards - 2) = 2197 / 132600 := by
sorry

end NUMINAMATH_CALUDE_heart_diamond_spade_probability_l1884_188430


namespace NUMINAMATH_CALUDE_acute_angle_tan_value_l1884_188480

theorem acute_angle_tan_value (α : Real) (h : α > 0 ∧ α < Real.pi / 2) 
  (h_eq : Real.sqrt (369 - 360 * Real.cos α) + Real.sqrt (544 - 480 * Real.sin α) - 25 = 0) : 
  40 * Real.tan α = 30 := by
  sorry

end NUMINAMATH_CALUDE_acute_angle_tan_value_l1884_188480


namespace NUMINAMATH_CALUDE_casino_chips_problem_l1884_188400

theorem casino_chips_problem (x y : ℕ) : 
  20 * x + 100 * y = 3000 →
  x + y = 14 →
  3000 - (20 * (x - x_lost) + 100 * (y - y_lost)) = 2240 →
  x_lost + y_lost = 14 →
  x_lost - y_lost = 2 :=
by sorry

end NUMINAMATH_CALUDE_casino_chips_problem_l1884_188400


namespace NUMINAMATH_CALUDE_parallelogram_area_32_14_l1884_188401

/-- The area of a parallelogram given its base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 32 cm and height 14 cm is 448 square centimeters -/
theorem parallelogram_area_32_14 : parallelogram_area 32 14 = 448 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_32_14_l1884_188401


namespace NUMINAMATH_CALUDE_square_greater_than_negative_l1884_188452

theorem square_greater_than_negative (x : ℝ) : x < 0 → x^2 > x := by
  sorry

end NUMINAMATH_CALUDE_square_greater_than_negative_l1884_188452


namespace NUMINAMATH_CALUDE_water_current_speed_l1884_188416

/-- Proves that the speed of a water current is 2 km/h given specific swimming conditions -/
theorem water_current_speed 
  (swimmer_speed : ℝ) 
  (distance : ℝ) 
  (time : ℝ) 
  (h1 : swimmer_speed = 4) 
  (h2 : distance = 7) 
  (h3 : time = 3.5) : 
  ∃ (current_speed : ℝ), 
    current_speed = 2 ∧ 
    (swimmer_speed - current_speed) * time = distance :=
by sorry

end NUMINAMATH_CALUDE_water_current_speed_l1884_188416


namespace NUMINAMATH_CALUDE_comic_arrangement_count_l1884_188499

/-- The number of different Batman comic books --/
def batman_comics : ℕ := 8

/-- The number of different Superman comic books --/
def superman_comics : ℕ := 7

/-- The number of different Wonder Woman comic books --/
def wonder_woman_comics : ℕ := 5

/-- The total number of comic books --/
def total_comics : ℕ := batman_comics + superman_comics + wonder_woman_comics

/-- The number of different comic book types --/
def comic_types : ℕ := 3

theorem comic_arrangement_count :
  (Nat.factorial batman_comics) * (Nat.factorial superman_comics) * 
  (Nat.factorial wonder_woman_comics) * (Nat.factorial comic_types) = 12203212800 := by
  sorry

end NUMINAMATH_CALUDE_comic_arrangement_count_l1884_188499


namespace NUMINAMATH_CALUDE_ampersand_composition_l1884_188437

-- Define the operations
def ampersand_right (x : ℝ) : ℝ := 9 - x
def ampersand_left (x : ℝ) : ℝ := x - 9

-- State the theorem
theorem ampersand_composition : ampersand_left (ampersand_right 15) = -15 := by
  sorry

end NUMINAMATH_CALUDE_ampersand_composition_l1884_188437


namespace NUMINAMATH_CALUDE_susan_money_left_l1884_188447

def susan_problem (swimming_income babysitting_income : ℝ) 
  (clothes_percentage books_percentage gifts_percentage : ℝ) : ℝ :=
  let total_income := swimming_income + babysitting_income
  let after_clothes := total_income * (1 - clothes_percentage)
  let after_books := after_clothes * (1 - books_percentage)
  let final_amount := after_books * (1 - gifts_percentage)
  final_amount

theorem susan_money_left : 
  susan_problem 1200 600 0.4 0.25 0.15 = 688.5 := by
  sorry

end NUMINAMATH_CALUDE_susan_money_left_l1884_188447


namespace NUMINAMATH_CALUDE_power_mod_29_l1884_188465

theorem power_mod_29 : 17^2003 % 29 = 26 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_29_l1884_188465


namespace NUMINAMATH_CALUDE_solve_equation_l1884_188467

theorem solve_equation (a : ℚ) (h : 3 * a + 6 * a / 4 = 6) : a = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1884_188467


namespace NUMINAMATH_CALUDE_population_growth_two_periods_l1884_188478

/-- Theorem: Population growth over two periods --/
theorem population_growth_two_periods (P : ℝ) (h : P > 0) :
  let first_half := P * 3
  let second_half := first_half * 4
  (second_half - P) / P * 100 = 1100 := by
  sorry

end NUMINAMATH_CALUDE_population_growth_two_periods_l1884_188478


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l1884_188498

theorem contrapositive_equivalence (x y : ℝ) :
  (((x - 1) * (y + 2) ≠ 0 → x ≠ 1 ∧ y ≠ -2) ↔
   (x = 1 ∨ y = -2 → (x - 1) * (y + 2) = 0)) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l1884_188498


namespace NUMINAMATH_CALUDE_complex_number_problem_l1884_188497

theorem complex_number_problem (z : ℂ) (i : ℂ) : 
  i * i = -1 → z / (-i) = 1 + 2*i → z = 2 - i := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l1884_188497


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l1884_188438

/-- Given a hyperbola with equation x²/(2m) - y²/m = 1, if one of its asymptotes
    has the equation y = 1, then m = -3 -/
theorem hyperbola_asymptote (m : ℝ) : 
  (∀ x y : ℝ, x^2/(2*m) - y^2/m = 1) →
  (∃ y : ℝ → ℝ, y = λ _ => 1) →
  m = -3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l1884_188438


namespace NUMINAMATH_CALUDE_task_assignment_count_l1884_188495

/-- The number of ways to assign 4 students to 3 tasks -/
def task_assignments : ℕ := 12

/-- The number of students -/
def num_students : ℕ := 4

/-- The number of tasks -/
def num_tasks : ℕ := 3

/-- The number of students assigned to clean the podium -/
def podium_cleaners : ℕ := 1

/-- The number of students assigned to sweep the floor -/
def floor_sweepers : ℕ := 1

/-- The number of students assigned to mop the floor -/
def floor_moppers : ℕ := 2

theorem task_assignment_count :
  task_assignments = num_students * (num_students - 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_task_assignment_count_l1884_188495


namespace NUMINAMATH_CALUDE_infinite_sqrt_twelve_l1884_188448

theorem infinite_sqrt_twelve (x : ℝ) : x = Real.sqrt (12 + x) → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_infinite_sqrt_twelve_l1884_188448
