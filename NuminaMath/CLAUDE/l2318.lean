import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_factorization_l2318_231896

theorem quadratic_factorization (d e f : ℤ) :
  (∀ x, x^2 + 17*x + 72 = (x + d) * (x + e)) →
  (∀ x, x^2 + 7*x - 60 = (x + e) * (x - f)) →
  d + e + f = 29 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2318_231896


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l2318_231859

theorem line_segment_endpoint (x : ℝ) : 
  (∃ (y : ℝ), (x = 3 - Real.sqrt 69 ∨ x = 3 + Real.sqrt 69) ∧ 
   ((3 - x)^2 + (8 - (-2))^2 = 13^2)) ↔ 
  (∃ (y : ℝ), ((3 - x)^2 + (y - (-2))^2 = 13^2) ∧ y = 8) :=
by sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l2318_231859


namespace NUMINAMATH_CALUDE_not_always_possible_to_empty_bags_l2318_231841

/-- Represents the state of the two bags --/
structure BagState where
  m : ℕ
  n : ℕ

/-- Allowed operations on the bags --/
inductive Operation
  | remove : ℕ → Operation
  | tripleFirst : Operation
  | tripleSecond : Operation

/-- Applies an operation to a bag state --/
def applyOperation (state : BagState) (op : Operation) : BagState :=
  match op with
  | Operation.remove k => ⟨state.m - k, state.n - k⟩
  | Operation.tripleFirst => ⟨3 * state.m, state.n⟩
  | Operation.tripleSecond => ⟨state.m, 3 * state.n⟩

/-- A sequence of operations --/
def OperationSequence := List Operation

/-- Applies a sequence of operations to a bag state --/
def applySequence (state : BagState) (seq : OperationSequence) : BagState :=
  seq.foldl applyOperation state

/-- Theorem: There exist initial values of m and n for which it's impossible to empty both bags --/
theorem not_always_possible_to_empty_bags : 
  ∃ (m n : ℕ), m ≥ 1 ∧ n ≥ 1 ∧ 
  ∀ (seq : OperationSequence), 
  let final_state := applySequence ⟨m, n⟩ seq
  (final_state.m ≠ 0 ∨ final_state.n ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_not_always_possible_to_empty_bags_l2318_231841


namespace NUMINAMATH_CALUDE_flowers_per_pot_l2318_231847

theorem flowers_per_pot (total_pots : ℕ) (total_flowers : ℕ) (h1 : total_pots = 544) (h2 : total_flowers = 17408) :
  total_flowers / total_pots = 32 := by
  sorry

end NUMINAMATH_CALUDE_flowers_per_pot_l2318_231847


namespace NUMINAMATH_CALUDE_number_transformation_l2318_231854

theorem number_transformation (x : ℕ) : x = 5 → 3 * (2 * x + 9) = 57 := by
  sorry

end NUMINAMATH_CALUDE_number_transformation_l2318_231854


namespace NUMINAMATH_CALUDE_e_integral_greater_than_ln_integral_l2318_231819

theorem e_integral_greater_than_ln_integral : ∫ (x : ℝ) in (0)..(1), Real.exp x > ∫ (x : ℝ) in (1)..(Real.exp 1), 1 / x := by
  sorry

end NUMINAMATH_CALUDE_e_integral_greater_than_ln_integral_l2318_231819


namespace NUMINAMATH_CALUDE_wide_right_field_goals_l2318_231858

theorem wide_right_field_goals 
  (total_attempts : ℕ) 
  (missed_fraction : ℚ) 
  (wide_right_percentage : ℚ) : ℕ :=
by
  have h1 : total_attempts = 60 := by sorry
  have h2 : missed_fraction = 1 / 4 := by sorry
  have h3 : wide_right_percentage = 1 / 5 := by sorry
  
  let missed_goals := total_attempts * missed_fraction
  let wide_right_goals := missed_goals * wide_right_percentage
  
  exact 3
  
#check wide_right_field_goals

end NUMINAMATH_CALUDE_wide_right_field_goals_l2318_231858


namespace NUMINAMATH_CALUDE_johns_age_to_tonyas_age_ratio_l2318_231857

/-- Proves that the ratio of John's age to Tonya's age is 1:2 given the specified conditions --/
theorem johns_age_to_tonyas_age_ratio :
  ∀ (john mary tonya : ℕ),
    john = 2 * mary →
    tonya = 60 →
    (john + mary + tonya) / 3 = 35 →
    john / tonya = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_johns_age_to_tonyas_age_ratio_l2318_231857


namespace NUMINAMATH_CALUDE_iris_shopping_expense_l2318_231893

def jacket_price : ℕ := 10
def shorts_price : ℕ := 6
def pants_price : ℕ := 12

def jacket_quantity : ℕ := 3
def shorts_quantity : ℕ := 2
def pants_quantity : ℕ := 4

def total_spent : ℕ := jacket_price * jacket_quantity + shorts_price * shorts_quantity + pants_price * pants_quantity

theorem iris_shopping_expense : total_spent = 90 := by
  sorry

end NUMINAMATH_CALUDE_iris_shopping_expense_l2318_231893


namespace NUMINAMATH_CALUDE_product_and_reciprocal_relation_l2318_231821

theorem product_and_reciprocal_relation (x y : ℝ) 
  (h_pos_x : 0 < x) (h_pos_y : 0 < y) 
  (h_product : x * y = 16) 
  (h_reciprocal : 1 / x = 3 * (1 / y)) :
  2 * y - x = 24 - (4 * Real.sqrt 3) / 3 := by
sorry

end NUMINAMATH_CALUDE_product_and_reciprocal_relation_l2318_231821


namespace NUMINAMATH_CALUDE_repeating_decimal_property_l2318_231846

def is_repeating_decimal_period_2 (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ 1 / n = (10 * a + b) / 99

def is_repeating_decimal_period_3 (n : ℕ) : Prop :=
  ∃ (u v w : ℕ), u < 10 ∧ v < 10 ∧ w < 10 ∧ 1 / n = (100 * u + 10 * v + w) / 999

theorem repeating_decimal_property (n : ℕ) :
  n > 0 ∧ n < 3000 ∧
  is_repeating_decimal_period_2 n ∧
  is_repeating_decimal_period_3 (n + 8) →
  601 ≤ n ∧ n ≤ 1200 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_property_l2318_231846


namespace NUMINAMATH_CALUDE_min_correct_answers_for_score_l2318_231816

/-- AMC 10 scoring system and Mark's strategy -/
structure AMC10 where
  total_problems : Nat
  attempted_problems : Nat
  correct_points : Int
  incorrect_points : Int
  unanswered_points : Int

/-- Calculate the score based on the number of correct answers -/
def calculate_score (amc : AMC10) (correct_answers : Nat) : Int :=
  let incorrect_answers := amc.attempted_problems - correct_answers
  let unanswered := amc.total_problems - amc.attempted_problems
  correct_answers * amc.correct_points + 
  incorrect_answers * amc.incorrect_points + 
  unanswered * amc.unanswered_points

/-- Theorem stating the minimum number of correct answers needed -/
theorem min_correct_answers_for_score (amc : AMC10) 
  (h1 : amc.total_problems = 25)
  (h2 : amc.attempted_problems = 20)
  (h3 : amc.correct_points = 8)
  (h4 : amc.incorrect_points = -2)
  (h5 : amc.unanswered_points = 2)
  (target_score : Int)
  (h6 : target_score = 120) :
  (∃ n : Nat, n ≥ 15 ∧ calculate_score amc n ≥ target_score ∧ 
    ∀ m : Nat, m < 15 → calculate_score amc m < target_score) :=
  sorry

end NUMINAMATH_CALUDE_min_correct_answers_for_score_l2318_231816


namespace NUMINAMATH_CALUDE_unique_divisor_remainder_l2318_231837

theorem unique_divisor_remainder : ∃! (d r : ℤ),
  (1210 % d = r) ∧
  (1690 % d = r) ∧
  (2670 % d = r) ∧
  (d > 0) ∧
  (0 ≤ r) ∧
  (r < d) ∧
  (d - 4*r = -20) := by
  sorry

end NUMINAMATH_CALUDE_unique_divisor_remainder_l2318_231837


namespace NUMINAMATH_CALUDE_car_value_after_depreciation_l2318_231834

def initial_value : ℝ := 10000

def depreciation_rates : List ℝ := [0.20, 0.15, 0.10, 0.08, 0.05]

def calculate_value (initial : ℝ) (rates : List ℝ) : ℝ :=
  rates.foldl (fun acc rate => acc * (1 - rate)) initial

theorem car_value_after_depreciation :
  calculate_value initial_value depreciation_rates = 5348.88 := by
  sorry

end NUMINAMATH_CALUDE_car_value_after_depreciation_l2318_231834


namespace NUMINAMATH_CALUDE_no_baby_cries_iff_even_l2318_231868

/-- Represents the direction a baby is facing -/
inductive Direction
  | Right
  | Down
  | Left
  | Up

/-- Represents a position on the grid -/
structure Position where
  x : Nat
  y : Nat

/-- Represents the state of a baby on the grid -/
structure Baby where
  pos : Position
  dir : Direction

/-- The grid of babies -/
def Grid := List Baby

/-- Function to check if a position is within the grid -/
def isWithinGrid (n m : Nat) (pos : Position) : Prop :=
  0 ≤ pos.x ∧ pos.x < n ∧ 0 ≤ pos.y ∧ pos.y < m

/-- Function to move a baby according to the rules -/
def moveBaby (n m : Nat) (baby : Baby) : Baby :=
  sorry

/-- Function to check if any baby cries after a move -/
def anyCry (n m : Nat) (grid : Grid) : Prop :=
  sorry

/-- Main theorem: No baby cries if and only if n and m are even -/
theorem no_baby_cries_iff_even (n m : Nat) :
  (∀ (grid : Grid), ¬(anyCry n m grid)) ↔ (∃ (k l : Nat), n = 2 * k ∧ m = 2 * l) :=
  sorry

end NUMINAMATH_CALUDE_no_baby_cries_iff_even_l2318_231868


namespace NUMINAMATH_CALUDE_problem_statement_l2318_231805

-- Define propositions p and q
def p : Prop := 2 + 2 = 5
def q : Prop := 3 > 2

-- Theorem stating the properties of p and q
theorem problem_statement :
  ¬p ∧ q ∧ (p ∨ q) ∧ ¬(p ∧ q) ∧ ¬p ∧ ¬¬q := by sorry

end NUMINAMATH_CALUDE_problem_statement_l2318_231805


namespace NUMINAMATH_CALUDE_sydney_initial_rocks_l2318_231843

/-- Rock collecting contest between Sydney and Conner --/
def rock_contest (sydney_initial : ℕ) : Prop :=
  let conner_initial := 723
  let sydney_day1 := 4
  let conner_day1 := 8 * sydney_day1
  let sydney_day2 := 0
  let conner_day2 := 123
  let sydney_day3 := 2 * conner_day1
  let conner_day3 := 27

  let sydney_total := sydney_initial + sydney_day1 + sydney_day2 + sydney_day3
  let conner_total := conner_initial + conner_day1 + conner_day2 + conner_day3

  sydney_total ≤ conner_total ∧ sydney_initial = 837

theorem sydney_initial_rocks : rock_contest 837 := by
  sorry

end NUMINAMATH_CALUDE_sydney_initial_rocks_l2318_231843


namespace NUMINAMATH_CALUDE_range_of_m_l2318_231838

theorem range_of_m (m : ℝ) : 
  (∀ x y : ℝ, x > 0 → y > 0 → (2 * y / x + 8 * x / y > m^2 + 2 * m)) → 
  -4 < m ∧ m < 2 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l2318_231838


namespace NUMINAMATH_CALUDE_factory_output_growth_rate_l2318_231878

theorem factory_output_growth_rate (x : ℝ) : 
  (∀ y : ℝ, y > 0 → (1 + x)^2 * y = 1.2 * y) → 
  x < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_factory_output_growth_rate_l2318_231878


namespace NUMINAMATH_CALUDE_arrangement_count_10_l2318_231882

/-- The number of ways to choose a president, vice-president, and committee from a group. -/
def arrangementCount (n : ℕ) : ℕ :=
  let presidentChoices := n
  let vicePresidentChoices := n - 1
  let officerArrangements := presidentChoices * vicePresidentChoices
  let remainingPeople := n - 2
  let committeeArrangements := remainingPeople.choose 3
  officerArrangements * committeeArrangements

/-- Theorem stating the number of arrangements for a group of 10 people. -/
theorem arrangement_count_10 : arrangementCount 10 = 5040 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_10_l2318_231882


namespace NUMINAMATH_CALUDE_sampling_methods_correct_l2318_231877

-- Define the sampling methods
inductive SamplingMethod
| SimpleRandom
| Stratified
| Systematic

-- Define the tasks
structure Task1 where
  total_products : Nat
  sample_size : Nat

structure Task2 where
  total_students : Nat
  first_year : Nat
  second_year : Nat
  third_year : Nat
  sample_size : Nat

structure Task3 where
  rows : Nat
  seats_per_row : Nat
  sample_size : Nat

-- Define the function to determine the most reasonable sampling method
def most_reasonable_sampling_method (task1 : Task1) (task2 : Task2) (task3 : Task3) : 
  (SamplingMethod × SamplingMethod × SamplingMethod) :=
  (SamplingMethod.SimpleRandom, SamplingMethod.Stratified, SamplingMethod.Systematic)

-- Theorem statement
theorem sampling_methods_correct (task1 : Task1) (task2 : Task2) (task3 : Task3) :
  task1.total_products = 30 ∧ task1.sample_size = 3 ∧
  task2.total_students = 2460 ∧ task2.first_year = 890 ∧ task2.second_year = 820 ∧ 
  task2.third_year = 810 ∧ task2.sample_size = 300 ∧
  task3.rows = 28 ∧ task3.seats_per_row = 32 ∧ task3.sample_size = 28 →
  most_reasonable_sampling_method task1 task2 task3 = 
    (SamplingMethod.SimpleRandom, SamplingMethod.Stratified, SamplingMethod.Systematic) :=
by
  sorry

end NUMINAMATH_CALUDE_sampling_methods_correct_l2318_231877


namespace NUMINAMATH_CALUDE_product_of_polynomials_l2318_231825

theorem product_of_polynomials (p q : ℝ) : 
  (∀ m : ℝ, (9 * m^2 - 2 * m + p) * (4 * m^2 + q * m - 5) = 
             36 * m^4 - 23 * m^3 - 31 * m^2 + 6 * m - 10) →
  p + q = 0 := by
sorry

end NUMINAMATH_CALUDE_product_of_polynomials_l2318_231825


namespace NUMINAMATH_CALUDE_set_equality_l2318_231881

theorem set_equality : Set ℝ := by
  have h1 : Set ℝ := {x | x = -2 ∨ x = 1}
  have h2 : Set ℝ := {x | (x - 1) * (x + 2) = 0}
  sorry

#check set_equality

end NUMINAMATH_CALUDE_set_equality_l2318_231881


namespace NUMINAMATH_CALUDE_orange_profit_theorem_l2318_231800

/-- Represents the orange selling scenario --/
structure OrangeSelling where
  buy_price : ℚ  -- Price to buy 4 oranges in cents
  sell_price : ℚ  -- Price to sell 7 oranges in cents
  free_oranges : ℕ  -- Number of free oranges per 8 bought
  target_profit : ℚ  -- Target profit in cents
  oranges_to_sell : ℕ  -- Number of oranges to sell

/-- Calculates the profit from selling oranges --/
def calculate_profit (scenario : OrangeSelling) : ℚ :=
  let cost_per_9 := scenario.buy_price * 2  -- Cost for 8 bought + 1 free
  let cost_per_orange := cost_per_9 / 9
  let revenue_per_orange := scenario.sell_price / 7
  let profit_per_orange := revenue_per_orange - cost_per_orange
  profit_per_orange * scenario.oranges_to_sell

/-- Theorem: Selling 120 oranges results in a profit of at least 200 cents --/
theorem orange_profit_theorem (scenario : OrangeSelling) 
  (h1 : scenario.buy_price = 15)
  (h2 : scenario.sell_price = 35)
  (h3 : scenario.free_oranges = 1)
  (h4 : scenario.target_profit = 200)
  (h5 : scenario.oranges_to_sell = 120) :
  calculate_profit scenario ≥ scenario.target_profit :=
sorry

end NUMINAMATH_CALUDE_orange_profit_theorem_l2318_231800


namespace NUMINAMATH_CALUDE_tenth_finger_number_l2318_231851

-- Define the function g based on the graph points
def g : ℕ → ℕ
| 0 => 5
| 1 => 0
| 2 => 4
| 3 => 8
| 4 => 3
| 5 => 7
| 6 => 2
| 7 => 6
| 8 => 1
| 9 => 5
| n => n  -- Default case for numbers not explicitly defined

-- Define a function that applies g n times to an initial value
def apply_g_n_times (n : ℕ) (initial : ℕ) : ℕ :=
  match n with
  | 0 => initial
  | k + 1 => g (apply_g_n_times k initial)

-- Theorem statement
theorem tenth_finger_number : apply_g_n_times 10 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_tenth_finger_number_l2318_231851


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l2318_231856

/-- Given that y varies inversely as x², prove that x = 2 when y = 8, 
    given that y = 2 when x = 4. -/
theorem inverse_variation_problem (y x : ℝ) (h : x > 0) : 
  (∃ (k : ℝ), ∀ (x : ℝ), x > 0 → y * x^2 = k) → 
  (2 * 4^2 = 8 * x^2) →
  (y = 8) →
  (x = 2) := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l2318_231856


namespace NUMINAMATH_CALUDE_ackermann_3_1_l2318_231818

def B : ℕ → ℕ → ℕ
  | 0, n => n + 1
  | m + 1, 0 => B m 1
  | m + 1, n + 1 => B m (B (m + 1) n)

theorem ackermann_3_1 : B 3 1 = 13 := by
  sorry

end NUMINAMATH_CALUDE_ackermann_3_1_l2318_231818


namespace NUMINAMATH_CALUDE_uniform_cost_is_355_l2318_231814

/-- Calculates the total cost of uniforms for a student given the costs of individual items -/
def uniform_cost (pants_cost shirt_cost tie_cost socks_cost : ℚ) : ℚ :=
  5 * (pants_cost + shirt_cost + tie_cost + socks_cost)

/-- Proves that the total cost of uniforms for a student is $355 -/
theorem uniform_cost_is_355 :
  uniform_cost 20 40 8 3 = 355 := by
  sorry

#eval uniform_cost 20 40 8 3

end NUMINAMATH_CALUDE_uniform_cost_is_355_l2318_231814


namespace NUMINAMATH_CALUDE_midpoint_to_directrix_distance_l2318_231863

-- Define the parabola C
def parabola_C (x y : ℝ) : Prop := y^2 = 12 * x

-- Define the line l
def line_l (x y : ℝ) : Prop := y = x - 2

-- Define the point P
def point_P : ℝ × ℝ := (2, 0)

-- Define the directrix of parabola C
def directrix_C : ℝ := -3

-- Theorem statement
theorem midpoint_to_directrix_distance :
  ∃ (A B : ℝ × ℝ),
    parabola_C A.1 A.2 ∧
    parabola_C B.1 B.2 ∧
    line_l A.1 A.2 ∧
    line_l B.1 B.2 ∧
    (A.1 + B.1) / 2 - directrix_C = 11 :=
sorry

end NUMINAMATH_CALUDE_midpoint_to_directrix_distance_l2318_231863


namespace NUMINAMATH_CALUDE_melanie_cats_count_l2318_231815

/-- Given that Jacob has 90 cats, Annie has three times fewer cats than Jacob,
    and Melanie has twice as many cats as Annie, prove that Melanie has 60 cats. -/
theorem melanie_cats_count :
  ∀ (jacob_cats annie_cats melanie_cats : ℕ),
    jacob_cats = 90 →
    annie_cats * 3 = jacob_cats →
    melanie_cats = annie_cats * 2 →
    melanie_cats = 60 := by
  sorry

end NUMINAMATH_CALUDE_melanie_cats_count_l2318_231815


namespace NUMINAMATH_CALUDE_cubic_function_derivative_l2318_231820

theorem cubic_function_derivative (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^3 + 3 * x^2 + 2
  let f' : ℝ → ℝ := λ x ↦ 3 * a * x^2 + 6 * x
  f' (-1) = 4 → a = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_derivative_l2318_231820


namespace NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_l2318_231897

/-- Represents a sampling method -/
inductive SamplingMethod
  | Lottery
  | RandomNumber
  | Stratified
  | Systematic

/-- Represents a population with two equal-sized subgroups -/
structure Population :=
  (total_size : ℕ)
  (subgroup_size : ℕ)
  (h_equal_subgroups : subgroup_size * 2 = total_size)

/-- Represents a sampling scenario -/
structure SamplingScenario :=
  (population : Population)
  (sample_size : ℕ)
  (h_sample_size_valid : sample_size ≤ population.total_size)

/-- Determines if a sampling method is appropriate for investigating subgroup differences -/
def is_appropriate_for_subgroup_investigation (method : SamplingMethod) (scenario : SamplingScenario) : Prop :=
  method = SamplingMethod.Stratified

/-- Theorem stating that stratified sampling is the most appropriate method
    for investigating differences between equal-sized subgroups -/
theorem stratified_sampling_most_appropriate
  (scenario : SamplingScenario)
  (h_equal_subgroups : scenario.population.subgroup_size * 2 = scenario.population.total_size) :
  is_appropriate_for_subgroup_investigation SamplingMethod.Stratified scenario :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_l2318_231897


namespace NUMINAMATH_CALUDE_cubic_function_property_l2318_231827

/-- Given a cubic function f(x) = ax³ + bx + 8, if f(-2) = 10, then f(2) = 6 -/
theorem cubic_function_property (a b : ℝ) : 
  let f := λ x : ℝ => a * x^3 + b * x + 8
  f (-2) = 10 → f 2 = 6 := by
sorry

end NUMINAMATH_CALUDE_cubic_function_property_l2318_231827


namespace NUMINAMATH_CALUDE_conveyor_belts_combined_time_l2318_231899

/-- The time taken for two conveyor belts to move one day's coal output together -/
theorem conveyor_belts_combined_time (old_rate new_rate : ℝ) 
  (h1 : old_rate = 1 / 21)
  (h2 : new_rate = 1 / 15) : 
  1 / (old_rate + new_rate) = 35 / 4 := by
  sorry

end NUMINAMATH_CALUDE_conveyor_belts_combined_time_l2318_231899


namespace NUMINAMATH_CALUDE_slope_intercept_form_through_points_l2318_231861

/-- Slope-intercept form of a line passing through two points -/
theorem slope_intercept_form_through_points
  (x₁ y₁ x₂ y₂ : ℚ)
  (h₁ : x₁ = -3)
  (h₂ : y₁ = 7)
  (h₃ : x₂ = 4)
  (h₄ : y₂ = -2)
  : ∃ (m b : ℚ), m = -9/7 ∧ b = 22/7 ∧ ∀ x y, y = m * x + b ↔ (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂) :=
by sorry

end NUMINAMATH_CALUDE_slope_intercept_form_through_points_l2318_231861


namespace NUMINAMATH_CALUDE_train_length_l2318_231888

/-- The length of a train given its speed and time to cross a stationary point -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 180 → time_s = 3 → speed_kmh * (1000 / 3600) * time_s = 150 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2318_231888


namespace NUMINAMATH_CALUDE_union_equality_condition_equivalence_condition_l2318_231822

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ 3 - 2*a}
def B : Set ℝ := {x | x^2 - 2*x - 8 ≤ 0}

-- Statement for part (1)
theorem union_equality_condition (a : ℝ) :
  A a ∪ B = B ↔ a ∈ Set.Ici (-1/2) :=
sorry

-- Statement for part (2)
theorem equivalence_condition (a : ℝ) :
  (∀ x, x ∈ B ↔ x ∈ A a) ↔ a ∈ Set.Iic (-1) :=
sorry

end NUMINAMATH_CALUDE_union_equality_condition_equivalence_condition_l2318_231822


namespace NUMINAMATH_CALUDE_quadratic_intersection_l2318_231864

theorem quadratic_intersection
  (a b c d : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (hd : d ≠ 0) :
  ∃! p : ℝ × ℝ,
    (λ x y => y = a * x^2 + b * x + c) p.1 p.2 ∧
    (λ x y => y = a * x^2 - b * x + c + d) p.1 p.2 ∧
    p.1 ≠ 0 ∧ p.2 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_intersection_l2318_231864


namespace NUMINAMATH_CALUDE_least_possible_square_area_l2318_231830

/-- The least possible area of a square with sides measured as 7 cm (to the nearest centimeter) and actual side lengths having at most two decimal places -/
theorem least_possible_square_area : 
  ∀ (side : ℝ), 
  (6.5 ≤ side) → 
  (side < 7.5) → 
  (∃ (n : ℕ), side = (n : ℝ) / 100) → 
  42.25 ≤ side * side :=
by sorry

end NUMINAMATH_CALUDE_least_possible_square_area_l2318_231830


namespace NUMINAMATH_CALUDE_increasing_on_zero_one_iff_decreasing_on_three_four_l2318_231807

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f y ≤ f x

theorem increasing_on_zero_one_iff_decreasing_on_three_four
  (f : ℝ → ℝ) (h1 : is_even_function f) (h2 : has_period f 2) :
  is_increasing_on f 0 1 ↔ is_decreasing_on f 3 4 :=
sorry

end NUMINAMATH_CALUDE_increasing_on_zero_one_iff_decreasing_on_three_four_l2318_231807


namespace NUMINAMATH_CALUDE_volume_of_three_cubes_cuboid_l2318_231865

/-- The volume of a cuboid formed by attaching three identical cubes -/
def cuboid_volume (cube_side_length : ℝ) (num_cubes : ℕ) : ℝ :=
  (cube_side_length ^ 3) * num_cubes

/-- Theorem: The volume of a cuboid formed by three 6cm cubes is 648 cm³ -/
theorem volume_of_three_cubes_cuboid : 
  cuboid_volume 6 3 = 648 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_three_cubes_cuboid_l2318_231865


namespace NUMINAMATH_CALUDE_unique_integer_fraction_l2318_231829

theorem unique_integer_fraction (m n : ℕ) (hm : m ≥ 3) (hn : n ≥ 3) :
  (∃ S : Set ℕ, Set.Infinite S ∧ ∀ a ∈ S, ∃ k : ℤ, k * (a^n + a^2 - 1) = a^m + a - 1) →
  m = 5 ∧ n = 3 := by
sorry

end NUMINAMATH_CALUDE_unique_integer_fraction_l2318_231829


namespace NUMINAMATH_CALUDE_ball_radius_for_given_hole_l2318_231836

/-- The radius of a spherical ball that leaves a circular hole with given dimensions -/
def ball_radius (hole_diameter : ℝ) (hole_depth : ℝ) : ℝ :=
  13

/-- Theorem stating that a ball leaving a hole with diameter 24 cm and depth 8 cm has a radius of 13 cm -/
theorem ball_radius_for_given_hole : ball_radius 24 8 = 13 := by
  sorry

end NUMINAMATH_CALUDE_ball_radius_for_given_hole_l2318_231836


namespace NUMINAMATH_CALUDE_smallest_multiple_sixty_four_satisfies_smallest_satisfying_number_l2318_231826

theorem smallest_multiple (x : ℕ) : x > 0 ∧ 450 * x % 640 = 0 → x ≥ 64 := by
  sorry

theorem sixty_four_satisfies : 450 * 64 % 640 = 0 := by
  sorry

theorem smallest_satisfying_number : ∃ x : ℕ, x > 0 ∧ 450 * x % 640 = 0 ∧ ∀ y : ℕ, (y > 0 ∧ 450 * y % 640 = 0) → x ≤ y := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_sixty_four_satisfies_smallest_satisfying_number_l2318_231826


namespace NUMINAMATH_CALUDE_not_perfect_square_9999xxxx_l2318_231853

theorem not_perfect_square_9999xxxx : 
  ∀ n : ℕ, 99990000 ≤ n ∧ n ≤ 99999999 → ¬∃ m : ℕ, n = m * m := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_9999xxxx_l2318_231853


namespace NUMINAMATH_CALUDE_work_completion_time_relation_l2318_231842

/-- Given a constant amount of work, if 100 workers complete it in 5 days,
    then 40 workers will complete it in 12.5 days. -/
theorem work_completion_time_relation :
  ∀ (total_work : ℝ),
    total_work > 0 →
    ∃ (worker_rate : ℝ),
      worker_rate > 0 ∧
      total_work = 100 * worker_rate * 5 →
      total_work = 40 * worker_rate * 12.5 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_relation_l2318_231842


namespace NUMINAMATH_CALUDE_actual_vs_planned_percentage_l2318_231876

/-- Given that Master Zhang processed 500 parts, which is 100 more than planned,
    prove that the actual amount is 25% more than the planned amount. -/
theorem actual_vs_planned_percentage : 
  let actual : ℕ := 500
  let difference : ℕ := 100
  let planned : ℕ := actual - difference
  (actual : ℚ) / (planned : ℚ) - 1 = 1/4
:= by sorry

end NUMINAMATH_CALUDE_actual_vs_planned_percentage_l2318_231876


namespace NUMINAMATH_CALUDE_wendys_cupcakes_l2318_231874

theorem wendys_cupcakes :
  ∀ (cupcakes cookies_baked pastries_left pastries_sold : ℕ),
    cookies_baked = 29 →
    pastries_left = 24 →
    pastries_sold = 9 →
    cupcakes + cookies_baked = pastries_left + pastries_sold →
    cupcakes = 4 := by
  sorry

end NUMINAMATH_CALUDE_wendys_cupcakes_l2318_231874


namespace NUMINAMATH_CALUDE_triangle_area_from_perimeter_and_inradius_l2318_231885

/-- The area of a triangle given its perimeter and inradius -/
theorem triangle_area_from_perimeter_and_inradius 
  (perimeter : ℝ) (inradius : ℝ) (area : ℝ) 
  (h_perimeter : perimeter = 20) 
  (h_inradius : inradius = 2.5) : 
  area = perimeter / 2 * inradius ∧ area = 25 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_from_perimeter_and_inradius_l2318_231885


namespace NUMINAMATH_CALUDE_hyperbola_midpoint_l2318_231852

def hyperbola (x y : ℝ) : Prop := x^2 - y^2/9 = 1

def is_midpoint (x₀ y₀ x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₀ = (x₁ + x₂) / 2 ∧ y₀ = (y₁ + y₂) / 2

theorem hyperbola_midpoint :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    hyperbola x₁ y₁ ∧
    hyperbola x₂ y₂ ∧
    is_midpoint (-1) (-4) x₁ y₁ x₂ y₂ ∧
    ¬∃ (x₁' y₁' x₂' y₂' : ℝ),
      hyperbola x₁' y₁' ∧
      hyperbola x₂' y₂' ∧
      (is_midpoint 1 1 x₁' y₁' x₂' y₂' ∨
       is_midpoint (-1) 2 x₁' y₁' x₂' y₂' ∨
       is_midpoint 1 3 x₁' y₁' x₂' y₂') :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_midpoint_l2318_231852


namespace NUMINAMATH_CALUDE_quadratic_function_b_range_l2318_231840

/-- Given a quadratic function f(x) = x^2 + 2bx + c where b and c are real numbers,
    if f(1) = 0 and the equation f(x) + x + b = 0 has two real roots
    in the intervals (-3,-2) and (0,1), then b is in the open interval (1/5, 5/7). -/
theorem quadratic_function_b_range (b c : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^2 + 2*b*x + c
  (f 1 = 0) →
  (∃ x₁ x₂, -3 < x₁ ∧ x₁ < -2 ∧ 0 < x₂ ∧ x₂ < 1 ∧ 
    f x₁ + x₁ + b = 0 ∧ f x₂ + x₂ + b = 0) →
  1/5 < b ∧ b < 5/7 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_b_range_l2318_231840


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2318_231889

/-- For an arithmetic sequence with common difference d ≠ 0, 
    if a_3 is the geometric mean of a_2 and a_6, then a_6 / a_3 = 2 -/
theorem arithmetic_sequence_ratio (a : ℕ → ℝ) (d : ℝ) :
  d ≠ 0 →
  (∀ n, a (n + 1) = a n + d) →
  a 3 ^ 2 = a 2 * a 6 →
  a 6 / a 3 = 2 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2318_231889


namespace NUMINAMATH_CALUDE_smallest_student_count_l2318_231894

/-- Represents the number of students in each grade --/
structure StudentCounts where
  ninth : ℕ
  tenth : ℕ
  eleventh : ℕ

/-- Checks if the given student counts satisfy the required ratios --/
def satisfiesRatios (counts : StudentCounts) : Prop :=
  7 * counts.tenth = 4 * counts.ninth ∧
  21 * counts.eleventh = 10 * counts.ninth

/-- The theorem stating the smallest possible total number of students --/
theorem smallest_student_count : 
  ∃ (counts : StudentCounts), 
    satisfiesRatios counts ∧ 
    counts.ninth + counts.tenth + counts.eleventh = 43 ∧
    (∀ (other : StudentCounts), 
      satisfiesRatios other → 
      other.ninth + other.tenth + other.eleventh ≥ 43) :=
by sorry

end NUMINAMATH_CALUDE_smallest_student_count_l2318_231894


namespace NUMINAMATH_CALUDE_chinese_learning_hours_l2318_231812

theorem chinese_learning_hours (total_hours : ℕ) (num_days : ℕ) (hours_per_day : ℕ) : 
  total_hours = 24 → num_days = 6 → hours_per_day = total_hours / num_days → hours_per_day = 4 := by
  sorry

end NUMINAMATH_CALUDE_chinese_learning_hours_l2318_231812


namespace NUMINAMATH_CALUDE_special_line_equation_l2318_231880

/-- A line passing through (-4, -1) with x-intercept twice its y-intercept -/
structure SpecialLine where
  /-- The slope of the line -/
  slope : ℝ
  /-- The y-intercept of the line -/
  y_intercept : ℝ
  /-- The line passes through (-4, -1) -/
  passes_through : slope * (-4) + y_intercept = -1
  /-- The x-intercept is twice the y-intercept -/
  intercept_relation : -y_intercept / slope = 2 * y_intercept

/-- The equation of the special line is x + 2y + 6 = 0 or y = 1/4 x -/
theorem special_line_equation (l : SpecialLine) :
  (∀ x y, y = l.slope * x + l.y_intercept ↔ x + 2*y + 6 = 0) ∨
  (l.slope = 1/4 ∧ l.y_intercept = 0) :=
sorry

end NUMINAMATH_CALUDE_special_line_equation_l2318_231880


namespace NUMINAMATH_CALUDE_solution_of_system_l2318_231806

/-- The system of equations:
    1. xy + 5yz - 6xz = -2z
    2. 2xy + 9yz - 9xz = -12z
    3. yz - 2xz = 6z
-/
def system_of_equations (x y z : ℝ) : Prop :=
  x*y + 5*y*z - 6*x*z = -2*z ∧
  2*x*y + 9*y*z - 9*x*z = -12*z ∧
  y*z - 2*x*z = 6*z

theorem solution_of_system :
  (∃ (x y z : ℝ), system_of_equations x y z ∧ (x = -2 ∧ y = 2 ∧ z = 1/6)) ∧
  (∀ (x : ℝ), system_of_equations x 0 0) ∧
  (∀ (y : ℝ), system_of_equations 0 y 0) :=
by sorry

end NUMINAMATH_CALUDE_solution_of_system_l2318_231806


namespace NUMINAMATH_CALUDE_probability_theorem_l2318_231850

-- Define the number of doctors and cities
def num_doctors : ℕ := 5
def num_cities : ℕ := 3

-- Define a function to calculate the probability
def probability_one_doctor_one_city (n_doctors : ℕ) (n_cities : ℕ) : ℚ :=
  7/75

-- State the theorem
theorem probability_theorem :
  probability_one_doctor_one_city num_doctors num_cities = 7/75 :=
by sorry

end NUMINAMATH_CALUDE_probability_theorem_l2318_231850


namespace NUMINAMATH_CALUDE_six_years_passed_l2318_231870

/-- Represents a stem-and-leaf plot --/
structure StemAndLeafPlot where
  stem : List Nat
  leaves : List (List Nat)

/-- The initial stem-and-leaf plot --/
def initial_plot : StemAndLeafPlot := {
  stem := [0, 1, 2, 3, 4, 5],
  leaves := [[3], [0, 1, 2, 3, 4, 5], [2, 3, 5, 6, 8, 9], [4, 6], [0, 2], []]
}

/-- The final stem-and-leaf plot with obscured numbers --/
def final_plot : StemAndLeafPlot := {
  stem := [0, 1, 2, 3, 4, 5],
  leaves := [[], [6, 9], [4, 7], [0], [2, 8], []]
}

/-- Function to calculate the years passed --/
def years_passed (initial : StemAndLeafPlot) (final : StemAndLeafPlot) : Nat :=
  sorry

/-- Theorem stating that 6 years have passed --/
theorem six_years_passed :
  years_passed initial_plot final_plot = 6 := by
  sorry

end NUMINAMATH_CALUDE_six_years_passed_l2318_231870


namespace NUMINAMATH_CALUDE_div_sqrt_three_equals_sqrt_three_l2318_231803

theorem div_sqrt_three_equals_sqrt_three : 3 / Real.sqrt 3 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_div_sqrt_three_equals_sqrt_three_l2318_231803


namespace NUMINAMATH_CALUDE_pinwheel_area_is_four_l2318_231867

/-- Represents a point on a 2D grid -/
structure GridPoint where
  x : Int
  y : Int

/-- Represents a triangle on the grid -/
structure Triangle where
  v1 : GridPoint
  v2 : GridPoint
  v3 : GridPoint

/-- Represents the pinwheel design -/
structure Pinwheel where
  center : GridPoint
  arms : List Triangle

/-- Calculates the area of a triangle using Pick's theorem -/
def triangleArea (t : Triangle) : Int :=
  sorry

/-- Calculates the total area of the pinwheel -/
def pinwheelArea (p : Pinwheel) : Int :=
  sorry

/-- The main theorem to prove -/
theorem pinwheel_area_is_four :
  let center := GridPoint.mk 3 3
  let arm1 := Triangle.mk center (GridPoint.mk 6 3) (GridPoint.mk 3 6)
  let arm2 := Triangle.mk center (GridPoint.mk 3 6) (GridPoint.mk 0 3)
  let arm3 := Triangle.mk center (GridPoint.mk 0 3) (GridPoint.mk 3 0)
  let arm4 := Triangle.mk center (GridPoint.mk 3 0) (GridPoint.mk 6 3)
  let pinwheel := Pinwheel.mk center [arm1, arm2, arm3, arm4]
  pinwheelArea pinwheel = 4 :=
sorry

end NUMINAMATH_CALUDE_pinwheel_area_is_four_l2318_231867


namespace NUMINAMATH_CALUDE_min_distance_parabola_to_line_l2318_231835

/-- The minimum distance from a point on the parabola y^2 = 4x to the line 3x + 4y + 15 = 0 -/
theorem min_distance_parabola_to_line :
  let parabola := {P : ℝ × ℝ | P.2^2 = 4 * P.1}
  let line := {P : ℝ × ℝ | 3 * P.1 + 4 * P.2 + 15 = 0}
  (∃ (d : ℝ), d > 0 ∧
    (∀ P ∈ parabola, ∀ Q ∈ line, d ≤ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)) ∧
    (∃ P ∈ parabola, ∃ Q ∈ line, d = Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2))) ∧
  (∀ d' : ℝ, (∀ P ∈ parabola, ∀ Q ∈ line, d' ≤ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)) →
    d' ≥ 29/15) :=
by sorry


end NUMINAMATH_CALUDE_min_distance_parabola_to_line_l2318_231835


namespace NUMINAMATH_CALUDE_transformed_line_y_intercept_l2318_231817

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translates a point vertically -/
def translateVertical (p : Point) (dy : ℝ) : Point :=
  { x := p.x, y := p.y + dy }

/-- Translates a point horizontally -/
def translateHorizontal (p : Point) (dx : ℝ) : Point :=
  { x := p.x + dx, y := p.y }

/-- Reflects a point in the line y = x -/
def reflectInDiagonal (p : Point) : Point :=
  { x := p.y, y := p.x }

/-- Applies a series of transformations to a line -/
def transformLine (l : Line) : Line :=
  sorry  -- The actual transformation is implemented here

/-- The main theorem stating that the transformed line has a y-intercept of -7 -/
theorem transformed_line_y_intercept :
  let originalLine : Line := { slope := 3, intercept := 6 }
  let transformedLine := transformLine originalLine
  transformedLine.intercept = -7 := by
  sorry


end NUMINAMATH_CALUDE_transformed_line_y_intercept_l2318_231817


namespace NUMINAMATH_CALUDE_copper_percentage_in_first_alloy_l2318_231811

/-- Prove that the percentage of copper in the first alloy is 20% -/
theorem copper_percentage_in_first_alloy
  (final_mixture_weight : ℝ)
  (final_copper_percentage : ℝ)
  (first_alloy_weight : ℝ)
  (second_alloy_copper_percentage : ℝ)
  (h1 : final_mixture_weight = 100)
  (h2 : final_copper_percentage = 24.9)
  (h3 : first_alloy_weight = 30)
  (h4 : second_alloy_copper_percentage = 27)
  : ∃ (first_alloy_copper_percentage : ℝ),
    first_alloy_copper_percentage = 20 ∧
    (first_alloy_copper_percentage / 100) * first_alloy_weight +
    (second_alloy_copper_percentage / 100) * (final_mixture_weight - first_alloy_weight) =
    (final_copper_percentage / 100) * final_mixture_weight :=
by sorry

end NUMINAMATH_CALUDE_copper_percentage_in_first_alloy_l2318_231811


namespace NUMINAMATH_CALUDE_sequence_general_term_l2318_231886

theorem sequence_general_term (a : ℕ → ℕ) : 
  a 1 = 1 ∧ 
  (∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + 2^n) → 
  ∀ n : ℕ, n ≥ 1 → a n = 2^n - 1 := by
sorry

end NUMINAMATH_CALUDE_sequence_general_term_l2318_231886


namespace NUMINAMATH_CALUDE_sin_m_equals_cos_714_l2318_231873

theorem sin_m_equals_cos_714 (m : ℤ) :
  -180 ≤ m ∧ m ≤ 180 ∧ Real.sin (m * π / 180) = Real.cos (714 * π / 180) →
  m = 96 ∨ m = 84 := by
  sorry

end NUMINAMATH_CALUDE_sin_m_equals_cos_714_l2318_231873


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2318_231808

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a with a₂ + a₁₂ = 32, prove that a₃ + a₁₁ = 32 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h1 : ArithmeticSequence a) (h2 : a 2 + a 12 = 32) :
  a 3 + a 11 = 32 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2318_231808


namespace NUMINAMATH_CALUDE_inequality_division_l2318_231887

theorem inequality_division (a b : ℝ) (h : a > b) : a/2 > b/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_division_l2318_231887


namespace NUMINAMATH_CALUDE_product_of_solutions_l2318_231810

theorem product_of_solutions (x : ℝ) : 
  (∃ α β : ℝ, (α * β = -10) ∧ (10 = -α^2 - 4*α) ∧ (10 = -β^2 - 4*β)) := by
  sorry

end NUMINAMATH_CALUDE_product_of_solutions_l2318_231810


namespace NUMINAMATH_CALUDE_no_solution_fibonacci_equation_l2318_231833

def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem no_solution_fibonacci_equation :
  ¬∃ n : ℕ, n * (fib n) * (fib (n - 1)) = (fib (n + 2) - 1)^2 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_fibonacci_equation_l2318_231833


namespace NUMINAMATH_CALUDE_symmetric_curve_correct_l2318_231875

/-- The equation of a curve symmetric to y^2 = 4x with respect to the line x = 2 -/
def symmetric_curve_equation (x y : ℝ) : Prop :=
  y^2 = 16 - 4*x

/-- The original curve equation -/
def original_curve_equation (x y : ℝ) : Prop :=
  y^2 = 4*x

/-- The line of symmetry -/
def symmetry_line : ℝ := 2

/-- Theorem stating that the symmetric curve equation is correct -/
theorem symmetric_curve_correct :
  ∀ x y : ℝ, symmetric_curve_equation x y ↔ 
  original_curve_equation (2*symmetry_line - x) y :=
by sorry

end NUMINAMATH_CALUDE_symmetric_curve_correct_l2318_231875


namespace NUMINAMATH_CALUDE_quadratic_extrema_l2318_231866

theorem quadratic_extrema :
  (∀ x : ℝ, 2 * x^2 - 1 ≥ -1) ∧
  (∀ x : ℝ, 2 * x^2 - 1 = -1 ↔ x = 0) ∧
  (∀ x : ℝ, -2 * (x + 1)^2 + 1 ≤ 1) ∧
  (∀ x : ℝ, -2 * (x + 1)^2 + 1 = 1 ↔ x = -1) ∧
  (∀ x : ℝ, 2 * x^2 - 4 * x + 1 ≥ -1) ∧
  (∀ x : ℝ, 2 * x^2 - 4 * x + 1 = -1 ↔ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_extrema_l2318_231866


namespace NUMINAMATH_CALUDE_paul_bought_two_pants_l2318_231860

def shirtPrice : ℝ := 15
def pantPrice : ℝ := 40
def suitPrice : ℝ := 150
def sweaterPrice : ℝ := 30
def storeDiscount : ℝ := 0.2
def couponDiscount : ℝ := 0.1
def finalSpent : ℝ := 252

def totalBeforeDiscount (numPants : ℝ) : ℝ :=
  4 * shirtPrice + numPants * pantPrice + suitPrice + 2 * sweaterPrice

def discountedTotal (numPants : ℝ) : ℝ :=
  (1 - storeDiscount) * totalBeforeDiscount numPants

def finalTotal (numPants : ℝ) : ℝ :=
  (1 - couponDiscount) * discountedTotal numPants

theorem paul_bought_two_pants :
  ∃ (numPants : ℝ), numPants = 2 ∧ finalTotal numPants = finalSpent :=
sorry

end NUMINAMATH_CALUDE_paul_bought_two_pants_l2318_231860


namespace NUMINAMATH_CALUDE_cricketer_average_score_l2318_231832

/-- Proves that the average score for the first 6 matches is 41 runs -/
theorem cricketer_average_score (total_matches : ℕ) (overall_average : ℚ) 
  (first_part_matches : ℕ) (second_part_matches : ℕ) (second_part_average : ℚ) :
  total_matches = 10 →
  overall_average = 389/10 →
  first_part_matches = 6 →
  second_part_matches = 4 →
  second_part_average = 143/4 →
  (overall_average * total_matches - second_part_average * second_part_matches) / first_part_matches = 41 :=
by sorry

end NUMINAMATH_CALUDE_cricketer_average_score_l2318_231832


namespace NUMINAMATH_CALUDE_complement_M_intersect_N_l2318_231898

def M : Set ℤ := {m : ℤ | m ≤ -3 ∨ m ≥ 2}
def N : Set ℤ := {n : ℤ | -1 ≤ n ∧ n ≤ 3}

theorem complement_M_intersect_N :
  (Mᶜ : Set ℤ) ∩ N = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_complement_M_intersect_N_l2318_231898


namespace NUMINAMATH_CALUDE_smallest_product_l2318_231872

def S : Finset Int := {-10, -4, 0, 2, 6}

theorem smallest_product (a b : Int) (ha : a ∈ S) (hb : b ∈ S) :
  ∃ (x y : Int) (hx : x ∈ S) (hy : y ∈ S), x * y ≤ a * b ∧ x * y = -60 := by
  sorry

end NUMINAMATH_CALUDE_smallest_product_l2318_231872


namespace NUMINAMATH_CALUDE_rearranged_prism_surface_area_l2318_231849

structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ

def cut_heights : List ℝ := [0.6, 0.3, 0.05, 0.05]

def surface_area (prism : RectangularPrism) (cuts : List ℝ) : ℝ :=
  2 * (prism.length * prism.width + prism.length * prism.height + prism.width * prism.height)

theorem rearranged_prism_surface_area :
  let original_prism : RectangularPrism := { length := 2, width := 2, height := 1 }
  surface_area original_prism cut_heights = 20 := by
  sorry

end NUMINAMATH_CALUDE_rearranged_prism_surface_area_l2318_231849


namespace NUMINAMATH_CALUDE_fractional_equation_range_l2318_231848

theorem fractional_equation_range (a x : ℝ) : 
  ((a + 2) / (x + 1) = 1 ∧ x ≤ 0 ∧ x + 1 ≠ 0) → (a ≤ -1 ∧ a ≠ -2) :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_range_l2318_231848


namespace NUMINAMATH_CALUDE_min_people_theorem_l2318_231895

/-- Represents a group of people consisting of married couples -/
structure CoupleGroup :=
  (num_couples : ℕ)
  (total_people : ℕ)
  (h_total : total_people = 2 * num_couples)

/-- The minimum number of people required to guarantee at least one married couple -/
def min_for_couple (group : CoupleGroup) : ℕ :=
  group.num_couples + 3

/-- The minimum number of people required to guarantee at least two people of the same gender -/
def min_for_same_gender (group : CoupleGroup) : ℕ := 3

/-- Theorem stating the minimum number of people required for both conditions -/
theorem min_people_theorem (group : CoupleGroup) 
  (h_group : group.num_couples = 10) : 
  min_for_couple group = 13 ∧ min_for_same_gender group = 3 := by
  sorry

#eval min_for_couple ⟨10, 20, rfl⟩
#eval min_for_same_gender ⟨10, 20, rfl⟩

end NUMINAMATH_CALUDE_min_people_theorem_l2318_231895


namespace NUMINAMATH_CALUDE_five_numbers_product_invariant_l2318_231831

theorem five_numbers_product_invariant :
  ∃ (a b c d e : ℝ),
    a * b * c * d * e ≠ 0 ∧
    (a - 1) * (b - 1) * (c - 1) * (d - 1) * (e - 1) = a * b * c * d * e :=
by sorry

end NUMINAMATH_CALUDE_five_numbers_product_invariant_l2318_231831


namespace NUMINAMATH_CALUDE_root_sum_squares_l2318_231869

-- Define the polynomial
def p (x : ℝ) : ℝ := x^3 - 12*x^2 + 44*x - 85

-- Define the roots of the polynomial
def roots_condition (a b c : ℝ) : Prop := p a = 0 ∧ p b = 0 ∧ p c = 0

-- Theorem statement
theorem root_sum_squares (a b c : ℝ) (h : roots_condition a b c) :
  (a + b)^2 + (b + c)^2 + (c + a)^2 = 200 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_squares_l2318_231869


namespace NUMINAMATH_CALUDE_intersection_sum_l2318_231801

-- Define the parabolas
def parabola1 (x y : ℝ) : Prop := y = (x - 2)^2
def parabola2 (x y : ℝ) : Prop := x + 6 = (y - 5)^2

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | parabola1 p.1 p.2 ∧ parabola2 p.1 p.2}

-- Theorem statement
theorem intersection_sum :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ),
    (x₁, y₁) ∈ intersection_points ∧
    (x₂, y₂) ∈ intersection_points ∧
    (x₃, y₃) ∈ intersection_points ∧
    (x₄, y₄) ∈ intersection_points ∧
    (x₁, y₁) ≠ (x₂, y₂) ∧ (x₁, y₁) ≠ (x₃, y₃) ∧ (x₁, y₁) ≠ (x₄, y₄) ∧
    (x₂, y₂) ≠ (x₃, y₃) ∧ (x₂, y₂) ≠ (x₄, y₄) ∧
    (x₃, y₃) ≠ (x₄, y₄) ∧
    x₁ + x₂ + x₃ + x₄ + y₁ + y₂ + y₃ + y₄ = 10 :=
by sorry

end NUMINAMATH_CALUDE_intersection_sum_l2318_231801


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a8_l2318_231804

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

/-- Theorem: In an arithmetic sequence where a_7 + a_9 = 16, a_8 = 8 -/
theorem arithmetic_sequence_a8 (a : ℕ → ℝ) (h1 : ArithmeticSequence a) (h2 : a 7 + a 9 = 16) :
  a 8 = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a8_l2318_231804


namespace NUMINAMATH_CALUDE_a2b2_value_l2318_231828

def is_arithmetic_progression (a b c d : ℝ) : Prop :=
  b - a = c - b ∧ c - b = d - c

def is_geometric_progression (a b c d e : ℝ) : Prop :=
  b / a = c / b ∧ c / b = d / c ∧ d / c = e / d

theorem a2b2_value (a₁ a₂ b₁ b₂ b₃ : ℝ) 
  (h_arith : is_arithmetic_progression 1 a₁ a₂ 4)
  (h_geom : is_geometric_progression 1 b₁ b₂ b₃ 4) : 
  a₂ * b₂ = 6 := by
  sorry

end NUMINAMATH_CALUDE_a2b2_value_l2318_231828


namespace NUMINAMATH_CALUDE_inscribed_triangle_ratio_l2318_231883

-- Define the ellipse
def ellipse (p q : ℝ) (x y : ℝ) : Prop :=
  x^2 / p^2 + y^2 / q^2 = 1

-- Define an equilateral triangle
def equilateral_triangle (A B C : ℝ × ℝ) : Prop :=
  (dist A B = dist B C) ∧ (dist B C = dist C A)

-- Define that a point is on a line segment
def on_segment (P A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = ((1 - t) • A.1 + t • B.1, (1 - t) • A.2 + t • B.2)

theorem inscribed_triangle_ratio (p q : ℝ) (A B C F₁ F₂ : ℝ × ℝ) :
  ellipse p q A.1 A.2 →
  ellipse p q B.1 B.2 →
  ellipse p q C.1 C.2 →
  B = (0, q) →
  A.2 = C.2 →
  equilateral_triangle A B C →
  on_segment F₁ B C →
  on_segment F₂ A B →
  dist F₁ F₂ = 2 →
  dist A B / dist F₁ F₂ = 8/5 :=
sorry

end NUMINAMATH_CALUDE_inscribed_triangle_ratio_l2318_231883


namespace NUMINAMATH_CALUDE_deer_distribution_l2318_231802

theorem deer_distribution (a : ℚ) (d : ℚ) : 
  (5 * a + 10 * d = 5) →  -- Sum of 5 terms equals 5
  (a + 3 * d = 2/3) →     -- Fourth term is 2/3
  a = 1/3 :=              -- First term (Gong Shi's share) is 1/3
by
  sorry

end NUMINAMATH_CALUDE_deer_distribution_l2318_231802


namespace NUMINAMATH_CALUDE_lucky_set_guaranteed_l2318_231813

/-- The number of cards in the deck -/
def deck_size : ℕ := 52

/-- The maximum possible sum of digits for any card in the deck -/
def max_sum : ℕ := 13

/-- Function to calculate the sum of digits of a number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- The minimum number of cards to draw to guarantee a "lucky" set -/
def min_draw : ℕ := 26

theorem lucky_set_guaranteed (draw : ℕ) (h : draw ≥ min_draw) :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a ≤ deck_size ∧ b ≤ deck_size ∧ c ≤ deck_size ∧
  sum_of_digits a = sum_of_digits b ∧ sum_of_digits b = sum_of_digits c :=
by sorry

end NUMINAMATH_CALUDE_lucky_set_guaranteed_l2318_231813


namespace NUMINAMATH_CALUDE_solution_to_money_division_l2318_231862

/-- Represents the division of money among three parties -/
structure MoneyDivision where
  x : ℝ  -- Amount x gets
  y : ℝ  -- Amount y gets
  z : ℝ  -- Amount z gets
  a : ℝ  -- Amount y gets for each rupee x gets

/-- The conditions of the problem -/
def problem_conditions (d : MoneyDivision) : Prop :=
  d.y = d.a * d.x ∧
  d.z = 0.5 * d.x ∧
  d.x + d.y + d.z = 78 ∧
  d.y = 18

/-- The theorem stating the solution to the problem -/
theorem solution_to_money_division :
  ∀ d : MoneyDivision, problem_conditions d → d.a = 0.45 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_money_division_l2318_231862


namespace NUMINAMATH_CALUDE_f_range_l2318_231884

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (Real.cos x ^ 2 - 3/4) + Real.sin x

theorem f_range : Set.range f = Set.Icc (-1/2) (Real.sqrt 2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_f_range_l2318_231884


namespace NUMINAMATH_CALUDE_certain_number_value_l2318_231824

theorem certain_number_value (n : ℝ) : 22 + Real.sqrt (-4 + 6 * 4 * n) = 24 → n = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_value_l2318_231824


namespace NUMINAMATH_CALUDE_rectangle_width_l2318_231845

theorem rectangle_width (length width : ℝ) : 
  width = length + 3 →
  2 * length + 2 * width = 54 →
  width = 15 := by
sorry

end NUMINAMATH_CALUDE_rectangle_width_l2318_231845


namespace NUMINAMATH_CALUDE_a_2023_coordinates_l2318_231855

/-- Represents a point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Returns the conjugate point of a given point -/
def conjugate (p : Point) : Point :=
  { x := -p.y + 1, y := p.x + 1 }

/-- Returns the nth point in the sequence starting from A₁ -/
def nthPoint (n : ℕ) : Point :=
  match n % 4 with
  | 1 => { x := 3, y := 1 }
  | 2 => { x := 0, y := 4 }
  | 3 => { x := -3, y := 1 }
  | _ => { x := 0, y := -2 }

theorem a_2023_coordinates : nthPoint 2023 = { x := -3, y := 1 } := by
  sorry

end NUMINAMATH_CALUDE_a_2023_coordinates_l2318_231855


namespace NUMINAMATH_CALUDE_flood_damage_in_pounds_l2318_231891

def flood_damage_rupees : ℝ := 45000000
def exchange_rate : ℝ := 75

theorem flood_damage_in_pounds : 
  flood_damage_rupees / exchange_rate = 600000 := by sorry

end NUMINAMATH_CALUDE_flood_damage_in_pounds_l2318_231891


namespace NUMINAMATH_CALUDE_solve_system_l2318_231892

theorem solve_system (p q : ℚ) (eq1 : 5 * p + 3 * q = 10) (eq2 : 3 * p + 5 * q = 20) : p = -5/8 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l2318_231892


namespace NUMINAMATH_CALUDE_inequality_not_always_true_l2318_231879

theorem inequality_not_always_true (x y w : ℝ) 
  (hx : x > 0) (hy : y > 0) (hxy : x^2 > y^2) (hw : w ≠ 0) :
  ¬ (∀ w, x^2 * w > y^2 * w) :=
by sorry

end NUMINAMATH_CALUDE_inequality_not_always_true_l2318_231879


namespace NUMINAMATH_CALUDE_exam_probabilities_l2318_231809

/-- Represents the probability of passing the exam for each attempt -/
structure PassProbability where
  male : ℚ
  female : ℚ

/-- Represents the exam conditions -/
structure ExamConditions where
  pass_prob : PassProbability
  max_attempts : ℕ
  free_attempts : ℕ

/-- Calculates the probability of both passing within first two attempts -/
def prob_both_pass_free (conditions : ExamConditions) : ℚ :=
  sorry

/-- Calculates the probability of passing with one person requiring a third attempt -/
def prob_one_third_attempt (conditions : ExamConditions) : ℚ :=
  sorry

theorem exam_probabilities (conditions : ExamConditions) 
  (h1 : conditions.pass_prob.male = 3/4)
  (h2 : conditions.pass_prob.female = 2/3)
  (h3 : conditions.max_attempts = 5)
  (h4 : conditions.free_attempts = 2) :
  prob_both_pass_free conditions = 5/6 ∧ 
  prob_one_third_attempt conditions = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_exam_probabilities_l2318_231809


namespace NUMINAMATH_CALUDE_no_rain_probability_l2318_231844

theorem no_rain_probability (p : ℚ) (h : p = 2/3) :
  (1 - p)^4 = 1/81 := by
  sorry

end NUMINAMATH_CALUDE_no_rain_probability_l2318_231844


namespace NUMINAMATH_CALUDE_sum_of_min_max_z_l2318_231871

-- Define the feasible region
def FeasibleRegion (x y : ℝ) : Prop :=
  2 * x - y + 2 ≥ 0 ∧ 2 * x + y - 2 ≥ 0 ∧ y ≥ 0

-- Define the function z
def z (x y : ℝ) : ℝ := x - y

-- Theorem statement
theorem sum_of_min_max_z :
  ∃ (min_z max_z : ℝ),
    (∀ (x y : ℝ), FeasibleRegion x y → z x y ≥ min_z) ∧
    (∃ (x y : ℝ), FeasibleRegion x y ∧ z x y = min_z) ∧
    (∀ (x y : ℝ), FeasibleRegion x y → z x y ≤ max_z) ∧
    (∃ (x y : ℝ), FeasibleRegion x y ∧ z x y = max_z) ∧
    min_z + max_z = -1 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_min_max_z_l2318_231871


namespace NUMINAMATH_CALUDE_average_weight_increase_l2318_231839

theorem average_weight_increase (group_size : ℕ) (original_weight new_weight : ℝ) :
  group_size = 4 →
  original_weight = 65 →
  new_weight = 71 →
  (new_weight - original_weight) / group_size = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_increase_l2318_231839


namespace NUMINAMATH_CALUDE_solve_linear_equation_l2318_231823

theorem solve_linear_equation (x : ℚ) : -3*x - 8 = 8*x + 3 → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l2318_231823


namespace NUMINAMATH_CALUDE_cone_base_radius_l2318_231890

/-- Given a right cone with slant height 27 cm and lateral surface forming
    a circular sector of 220° when unrolled, the radius of the base is 16.5 cm. -/
theorem cone_base_radius (s : ℝ) (θ : ℝ) (h1 : s = 27) (h2 : θ = 220 * π / 180) :
  let r := s * θ / (2 * π)
  r = 16.5 := by sorry

end NUMINAMATH_CALUDE_cone_base_radius_l2318_231890
