import Mathlib

namespace target_primes_are_5_13_17_29_l3179_317985

/-- The set of prime numbers less than 30 -/
def primes_less_than_30 : Set ℕ :=
  {p | p < 30 ∧ Nat.Prime p}

/-- A function that checks if a number becomes a multiple of 4 after adding 3 -/
def becomes_multiple_of_4 (n : ℕ) : Prop :=
  (n + 3) % 4 = 0

/-- The set of prime numbers less than 30 that become multiples of 4 after adding 3 -/
def target_primes : Set ℕ :=
  {p ∈ primes_less_than_30 | becomes_multiple_of_4 p}

theorem target_primes_are_5_13_17_29 : target_primes = {5, 13, 17, 29} := by
  sorry

end target_primes_are_5_13_17_29_l3179_317985


namespace train_B_speed_l3179_317910

-- Define the problem parameters
def distance_between_cities : ℝ := 330
def speed_train_A : ℝ := 60
def time_train_A : ℝ := 3
def time_train_B : ℝ := 2

-- Theorem statement
theorem train_B_speed : 
  ∃ (speed_train_B : ℝ),
    speed_train_B * time_train_B + speed_train_A * time_train_A = distance_between_cities ∧
    speed_train_B = 75 := by
  sorry

end train_B_speed_l3179_317910


namespace symmetric_line_passes_through_point_l3179_317908

-- Define a line type
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define a point type
structure Point where
  x : ℝ
  y : ℝ

-- Function to check if a point is on a line
def point_on_line (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

-- Function to check if two lines are symmetric about a point
def symmetric_lines (l₁ l₂ : Line) (p : Point) : Prop :=
  ∀ (x y : ℝ), point_on_line ⟨x, y⟩ l₁ ↔ 
    point_on_line ⟨2*p.x - x, 2*p.y - y⟩ l₂

-- Theorem statement
theorem symmetric_line_passes_through_point :
  ∀ (k : ℝ),
  let l₁ : Line := ⟨k, -4*k⟩
  let l₂ : Line := sorry
  let p : Point := ⟨2, 1⟩
  symmetric_lines l₁ l₂ p →
  point_on_line ⟨0, 2⟩ l₂ := by
  sorry

end symmetric_line_passes_through_point_l3179_317908


namespace files_deleted_amy_deleted_files_l3179_317941

theorem files_deleted (initial_music : ℕ) (initial_video : ℕ) (remaining : ℕ) : ℕ :=
  (initial_music + initial_video) - remaining

theorem amy_deleted_files : files_deleted 4 21 2 = 23 := by
  sorry

end files_deleted_amy_deleted_files_l3179_317941


namespace factorial_square_root_simplification_l3179_317946

theorem factorial_square_root_simplification :
  Real.sqrt ((4 * 3 * 2 * 1) * (4 * 3 * 2 * 1) + 4) = 2 * Real.sqrt 145 := by
  sorry

end factorial_square_root_simplification_l3179_317946


namespace halfway_between_fractions_l3179_317982

theorem halfway_between_fractions :
  (1 / 8 : ℚ) + ((1 / 3 : ℚ) - (1 / 8 : ℚ)) / 2 = 11 / 48 := by
  sorry

end halfway_between_fractions_l3179_317982


namespace chase_theorem_l3179_317999

/-- Represents the chase scenario between a greyhound and a rabbit. -/
structure ChaseScenario where
  n : ℕ  -- Initial lead of the rabbit in rabbit hops
  a : ℕ  -- Number of rabbit hops
  b : ℕ  -- Number of greyhound hops
  c : ℕ  -- Equivalent rabbit hops
  d : ℕ  -- Greyhound hops

/-- Calculates the number of hops the rabbit can make before being caught. -/
def rabbit_hops (scenario : ChaseScenario) : ℚ :=
  (scenario.a * scenario.d * scenario.n : ℚ) / (scenario.b * scenario.c - scenario.a * scenario.d)

/-- Calculates the number of hops the greyhound makes before catching the rabbit. -/
def greyhound_hops (scenario : ChaseScenario) : ℚ :=
  (scenario.b * scenario.d * scenario.n : ℚ) / (scenario.b * scenario.c - scenario.a * scenario.d)

/-- Theorem stating the correctness of the chase calculations. -/
theorem chase_theorem (scenario : ChaseScenario) 
  (h : scenario.b * scenario.c ≠ scenario.a * scenario.d) : 
  rabbit_hops scenario * (scenario.b * scenario.c : ℚ) / (scenario.a * scenario.d) = 
  greyhound_hops scenario * (scenario.c : ℚ) / scenario.d + scenario.n := by
  sorry

end chase_theorem_l3179_317999


namespace largest_n_with_unique_k_l3179_317975

theorem largest_n_with_unique_k : 
  (∃! (n : ℕ), n > 0 ∧ 
    (∃! (k : ℤ), (9:ℚ)/17 < (n:ℚ)/(n + k) ∧ (n:ℚ)/(n + k) < 8/15)) → 
  (∃! (n : ℕ), n = 72 ∧ n > 0 ∧ 
    (∃! (k : ℤ), (9:ℚ)/17 < (n:ℚ)/(n + k) ∧ (n:ℚ)/(n + k) < 8/15)) :=
by sorry

end largest_n_with_unique_k_l3179_317975


namespace angle_with_special_supplementary_complementary_relation_l3179_317979

theorem angle_with_special_supplementary_complementary_relation :
  ∀ x : ℝ, (180 - x = 3 * (90 - x)) → x = 45 := by
  sorry

end angle_with_special_supplementary_complementary_relation_l3179_317979


namespace shirt_price_l3179_317902

theorem shirt_price (total_cost : ℝ) (price_difference : ℝ) (shirt_price : ℝ) :
  total_cost = 80.34 →
  shirt_price = (total_cost + price_difference) / 2 - price_difference →
  price_difference = 7.43 →
  shirt_price = 36.455 := by
  sorry

end shirt_price_l3179_317902


namespace quadratic_discriminant_nonnegative_l3179_317907

/-- 
Given a quadratic equation ax^2 + 4bx + c = 0 where a, b, and c form an arithmetic progression,
prove that the discriminant Δ is always non-negative.
-/
theorem quadratic_discriminant_nonnegative 
  (a b c : ℝ) 
  (h_progression : ∃ d : ℝ, b = a + d ∧ c = a + 2*d) : 
  (4*b)^2 - 4*a*c ≥ 0 := by
  sorry

end quadratic_discriminant_nonnegative_l3179_317907


namespace monomial_properties_l3179_317942

/-- Represents a monomial -3a²bc/5 -/
structure Monomial where
  coefficient : ℚ
  a_exponent : ℕ
  b_exponent : ℕ
  c_exponent : ℕ

/-- The specific monomial -3a²bc/5 -/
def our_monomial : Monomial :=
  { coefficient := -3/5
    a_exponent := 2
    b_exponent := 1
    c_exponent := 1 }

/-- The coefficient of a monomial is its numerical factor -/
def get_coefficient (m : Monomial) : ℚ := m.coefficient

/-- The degree of a monomial is the sum of its variable exponents -/
def get_degree (m : Monomial) : ℕ := m.a_exponent + m.b_exponent + m.c_exponent

theorem monomial_properties :
  (get_coefficient our_monomial = -3/5) ∧ (get_degree our_monomial = 4) := by
  sorry


end monomial_properties_l3179_317942


namespace committee_selection_ways_l3179_317967

theorem committee_selection_ways (n m : ℕ) (hn : n = 30) (hm : m = 5) :
  Nat.choose n m = 54810 := by
  sorry

end committee_selection_ways_l3179_317967


namespace f_composition_of_3_l3179_317955

def f (x : ℤ) : ℤ :=
  if x % 2 = 0 then x / 2 else 3 * x + 1

theorem f_composition_of_3 : f (f (f (f (f 3)))) = 4 := by
  sorry

end f_composition_of_3_l3179_317955


namespace digit_interchange_theorem_l3179_317918

theorem digit_interchange_theorem (a b m : ℕ) (h1 : a > 0 ∧ a < 10) (h2 : b < 10) 
  (h3 : 10 * a + b = m * (a * b)) :
  10 * b + a = (11 - m) * (a * b) :=
sorry

end digit_interchange_theorem_l3179_317918


namespace range_of_a_l3179_317938

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- Define set A
def A : Set ℝ := {x | f (x - 3) < 1}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | f (x - 2*a) < a^2}

-- Theorem statement
theorem range_of_a (a : ℝ) :
  (A ∪ B a = B a) → (1 ≤ a ∧ a ≤ 2) :=
by sorry

end range_of_a_l3179_317938


namespace original_quadratic_equation_l3179_317950

/-- The original quadratic equation given Xiaoming and Xiaohua's mistakes -/
theorem original_quadratic_equation :
  ∀ (a b c : ℝ),
  (∃ (x y : ℝ), x * y = -6 ∧ x + y = 2 - (-3)) →  -- Xiaoming's roots condition
  (∃ (u v : ℝ), u + v = -2 + 5) →                 -- Xiaohua's roots condition
  a = 1 →                                         -- Coefficient of x^2 is 1
  (a * X^2 + b * X + c = 0 ↔ X^2 - 3 * X - 6 = 0) -- The original equation
  := by sorry

end original_quadratic_equation_l3179_317950


namespace sum_neq_two_implies_both_neq_one_l3179_317961

theorem sum_neq_two_implies_both_neq_one (x y : ℝ) : x + y ≠ 2 → x ≠ 1 ∧ y ≠ 1 := by
  sorry

end sum_neq_two_implies_both_neq_one_l3179_317961


namespace square_sum_problem_l3179_317990

theorem square_sum_problem (x₁ y₁ x₂ y₂ : ℝ) 
  (h1 : x₁^2 + 5*x₂^2 = 10)
  (h2 : x₂*y₁ - x₁*y₂ = 5)
  (h3 : x₁*y₁ + 5*x₂*y₂ = Real.sqrt 105) :
  y₁^2 + 5*y₂^2 = 23 := by
sorry

end square_sum_problem_l3179_317990


namespace equation_solution_l3179_317991

theorem equation_solution : 
  {x : ℝ | x^2 + 6*x + 11 = |2*x + 5 - 5*x|} = {-6, -1} := by sorry

end equation_solution_l3179_317991


namespace constant_b_value_l3179_317909

theorem constant_b_value (x y b : ℝ) 
  (h1 : (7 * x + b * y) / (x - 2 * y) = 25)
  (h2 : x / (2 * y) = 3 / 2) : 
  b = 4 := by sorry

end constant_b_value_l3179_317909


namespace quadratic_negative_root_l3179_317923

theorem quadratic_negative_root (m : ℝ) :
  (∃ x : ℝ, x < 0 ∧ m * x^2 + 2 * x + 1 = 0) ↔ m ≤ 1 := by
  sorry

end quadratic_negative_root_l3179_317923


namespace banana_cost_l3179_317932

/-- Given that 4 bananas cost $20, prove that one banana costs $5. -/
theorem banana_cost : 
  ∀ (cost : ℝ), (4 * cost = 20) → (cost = 5) := by
  sorry

end banana_cost_l3179_317932


namespace color_change_probability_l3179_317953

/-- Represents the duration of each color in the traffic light cycle -/
structure TrafficLightCycle where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the total cycle duration -/
def cycleDuration (cycle : TrafficLightCycle) : ℕ :=
  cycle.green + cycle.yellow + cycle.red

/-- Calculates the number of 5-second intervals where a color change occurs -/
def colorChangeIntervals (cycle : TrafficLightCycle) : ℕ := 3

/-- Represents the duration of the observation interval -/
def observationInterval : ℕ := 5

/-- Theorem: The probability of observing a color change is 3/20 -/
theorem color_change_probability (cycle : TrafficLightCycle)
    (h1 : cycle.green = 45)
    (h2 : cycle.yellow = 5)
    (h3 : cycle.red = 50) :
    (colorChangeIntervals cycle : ℚ) * observationInterval / (cycleDuration cycle) = 3 / 20 := by
  sorry


end color_change_probability_l3179_317953


namespace loan_interest_percentage_l3179_317994

theorem loan_interest_percentage 
  (loan_amount : ℝ) 
  (monthly_payment : ℝ) 
  (num_months : ℕ) 
  (h1 : loan_amount = 150)
  (h2 : monthly_payment = 15)
  (h3 : num_months = 11) : 
  (monthly_payment * num_months - loan_amount) / loan_amount * 100 = 10 := by
  sorry

end loan_interest_percentage_l3179_317994


namespace article_markups_l3179_317977

/-- Calculate the markup for an article given its purchase price, overhead percentage, and desired net profit. -/
def calculateMarkup (purchasePrice : ℝ) (overheadPercentage : ℝ) (netProfit : ℝ) : ℝ :=
  let overheadCost := overheadPercentage * purchasePrice
  let totalCost := purchasePrice + overheadCost
  let sellingPrice := totalCost + netProfit
  sellingPrice - purchasePrice

/-- The markups for two articles with given purchase prices, overhead percentages, and desired net profits. -/
theorem article_markups :
  let article1Markup := calculateMarkup 48 0.35 18
  let article2Markup := calculateMarkup 60 0.40 22
  article1Markup = 34.80 ∧ article2Markup = 46 := by
  sorry

#eval calculateMarkup 48 0.35 18
#eval calculateMarkup 60 0.40 22

end article_markups_l3179_317977


namespace unique_base_for_256_four_digits_l3179_317934

/-- A number n has exactly d digits in base b if and only if b^(d-1) ≤ n < b^d -/
def has_exactly_d_digits (n : ℕ) (b : ℕ) (d : ℕ) : Prop :=
  b^(d-1) ≤ n ∧ n < b^d

/-- The theorem statement -/
theorem unique_base_for_256_four_digits :
  ∃! b : ℕ, b ≥ 2 ∧ has_exactly_d_digits 256 b 4 :=
sorry

end unique_base_for_256_four_digits_l3179_317934


namespace special_function_at_one_third_l3179_317992

/-- A function satisfying the given properties -/
def special_function (g : ℝ → ℝ) : Prop :=
  g 1 = 1 ∧ ∀ x y : ℝ, g (x * y + g x) = x * g y + g x

/-- The main theorem -/
theorem special_function_at_one_third {g : ℝ → ℝ} (hg : special_function g) : 
  g (1/3) = 1/3 := by
  sorry

end special_function_at_one_third_l3179_317992


namespace typists_problem_l3179_317996

theorem typists_problem (n : ℕ) : 
  (∃ (k : ℕ), k > 0 ∧ k * n = 46) → 
  (30 * (3 * 46) / n = 207) → 
  n = 20 := by
sorry

end typists_problem_l3179_317996


namespace tangent_sum_simplification_l3179_317966

theorem tangent_sum_simplification :
  (Real.tan (20 * π / 180) + Real.tan (30 * π / 180) + Real.tan (60 * π / 180) + Real.tan (70 * π / 180)) / Real.cos (40 * π / 180) =
  (Real.sin (50 * π / 180) * (Real.cos (60 * π / 180) * Real.cos (70 * π / 180) + Real.cos (20 * π / 180) * Real.cos (30 * π / 180))) /
  (Real.cos (20 * π / 180) * Real.cos (30 * π / 180) * Real.cos (40 * π / 180) * Real.cos (60 * π / 180) * Real.cos (70 * π / 180)) :=
by sorry

end tangent_sum_simplification_l3179_317966


namespace pies_sold_theorem_l3179_317926

/-- Represents the number of slices in an apple pie -/
def apple_slices : ℕ := 8

/-- Represents the number of slices in a peach pie -/
def peach_slices : ℕ := 6

/-- Represents the number of customers who ordered apple pie slices -/
def apple_customers : ℕ := 56

/-- Represents the number of customers who ordered peach pie slices -/
def peach_customers : ℕ := 48

/-- Calculates the total number of pies sold given the number of slices per pie and the number of customers -/
def total_pies (apple_slices peach_slices apple_customers peach_customers : ℕ) : ℕ :=
  (apple_customers / apple_slices) + (peach_customers / peach_slices)

/-- Theorem stating that the total number of pies sold is 15 -/
theorem pies_sold_theorem : total_pies apple_slices peach_slices apple_customers peach_customers = 15 := by
  sorry

end pies_sold_theorem_l3179_317926


namespace souvenir_profit_maximization_l3179_317970

/-- Represents the problem of maximizing profit for a souvenir seller --/
theorem souvenir_profit_maximization
  (cost_price : ℕ)
  (initial_price : ℕ)
  (initial_sales : ℕ)
  (price_increase : ℕ → ℕ)
  (sales_decrease : ℕ → ℕ)
  (profit : ℕ → ℕ)
  (h_cost : cost_price = 5)
  (h_initial_price : initial_price = 9)
  (h_initial_sales : initial_sales = 32)
  (h_price_increase : ∀ x, price_increase x = x)
  (h_sales_decrease : ∀ x, sales_decrease x = 4 * x)
  (h_profit : ∀ x, profit x = (initial_price + price_increase x - cost_price) * (initial_sales - sales_decrease x)) :
  ∃ (optimal_increase : ℕ),
    optimal_increase = 2 ∧
    ∀ x, x ≠ optimal_increase → profit x ≤ profit optimal_increase ∧
    profit optimal_increase = 144 := by
  sorry


end souvenir_profit_maximization_l3179_317970


namespace square_sum_problem_l3179_317978

theorem square_sum_problem (a b c d m n : ℕ) 
  (h1 : a^2 + b^2 + c^2 + d^2 = 1989)
  (h2 : a + b + c + d = m^2)
  (h3 : max a (max b (max c d)) = n^2) :
  m = 9 ∧ n = 6 := by
  sorry

end square_sum_problem_l3179_317978


namespace patio_length_l3179_317937

theorem patio_length (width : ℝ) (length : ℝ) (perimeter : ℝ) : 
  width > 0 →
  length = 4 * width →
  perimeter = 2 * (length + width) →
  perimeter = 100 →
  length = 40 := by
sorry

end patio_length_l3179_317937


namespace xiaoming_mother_money_l3179_317939

/-- The amount of money Xiaoming's mother brought to buy soap. -/
def money : ℕ := 36

/-- The price of one unit of brand A soap in yuan. -/
def price_A : ℕ := 6

/-- The price of one unit of brand B soap in yuan. -/
def price_B : ℕ := 9

/-- The number of units of brand A soap that can be bought with the money. -/
def units_A : ℕ := money / price_A

/-- The number of units of brand B soap that can be bought with the money. -/
def units_B : ℕ := money / price_B

theorem xiaoming_mother_money :
  (units_A = units_B + 2) ∧
  (money = units_A * price_A) ∧
  (money = units_B * price_B) := by
  sorry

end xiaoming_mother_money_l3179_317939


namespace product_of_roots_l3179_317919

theorem product_of_roots (x : ℝ) : 
  (∃ r₁ r₂ r₃ : ℝ, x^3 - 12*x^2 + 48*x + 28 = (x - r₁) * (x - r₂) * (x - r₃)) →
  (∃ r₁ r₂ r₃ : ℝ, x^3 - 12*x^2 + 48*x + 28 = (x - r₁) * (x - r₂) * (x - r₃) ∧ r₁ * r₂ * r₃ = -28) :=
by sorry

end product_of_roots_l3179_317919


namespace square_difference_formula_inapplicable_l3179_317920

theorem square_difference_formula_inapplicable (a b : ℝ) :
  ¬∃ (x y : ℝ), (a - b) * (b - a) = x^2 - y^2 :=
sorry

end square_difference_formula_inapplicable_l3179_317920


namespace short_trees_planted_l3179_317949

/-- The number of short trees planted in a park. -/
theorem short_trees_planted (current : ℕ) (final : ℕ) (planted : ℕ) : 
  current = 3 → final = 12 → planted = final - current → planted = 9 := by
sorry

end short_trees_planted_l3179_317949


namespace initial_pizza_slices_l3179_317989

-- Define the number of slices eaten at each meal and the number of slices left
def breakfast_slices : ℕ := 4
def lunch_slices : ℕ := 2
def snack_slices : ℕ := 2
def dinner_slices : ℕ := 5
def slices_left : ℕ := 2

-- Define the total number of slices eaten
def total_eaten : ℕ := breakfast_slices + lunch_slices + snack_slices + dinner_slices

-- Theorem: The initial number of pizza slices is 15
theorem initial_pizza_slices : 
  total_eaten + slices_left = 15 := by
  sorry

end initial_pizza_slices_l3179_317989


namespace total_weight_moved_tom_total_weight_l3179_317969

/-- Calculate the total weight Tom is moving with. -/
theorem total_weight_moved (tom_weight : ℝ) (hand_weight_ratio : ℝ) (vest_weight_ratio : ℝ) : ℝ :=
  let vest_weight := vest_weight_ratio * tom_weight
  let hand_weight := hand_weight_ratio * tom_weight
  let total_hand_weight := 2 * hand_weight
  total_hand_weight + vest_weight

/-- Prove that Tom is moving a total weight of 525 kg. -/
theorem tom_total_weight :
  total_weight_moved 150 1.5 0.5 = 525 := by
  sorry

end total_weight_moved_tom_total_weight_l3179_317969


namespace stating_height_represents_frequency_ratio_l3179_317980

/-- Represents a frequency distribution histogram --/
structure FrequencyHistogram where
  /-- The height of a bar in the histogram --/
  height : ℝ → ℝ
  /-- The frequency of individuals in a group --/
  frequency : ℝ → ℝ
  /-- The class interval for a group --/
  classInterval : ℝ → ℝ

/-- 
Theorem stating that the height of a frequency distribution histogram
represents the ratio of the frequency to the class interval
-/
theorem height_represents_frequency_ratio (h : FrequencyHistogram) :
  ∀ x, h.height x = h.frequency x / h.classInterval x := by
  sorry

end stating_height_represents_frequency_ratio_l3179_317980


namespace simplify_expression_l3179_317972

theorem simplify_expression : (256 : ℝ) ^ (1/4) * (144 : ℝ) ^ (1/2) = 48 := by
  sorry

end simplify_expression_l3179_317972


namespace fourth_rectangle_perimeter_l3179_317904

theorem fourth_rectangle_perimeter 
  (a b c d : ℝ) 
  (h1 : 2 * (c + b) = 6) 
  (h2 : 2 * (a + c) = 10) 
  (h3 : 2 * (a + d) = 12) : 
  2 * (b + d) = 8 := by
sorry

end fourth_rectangle_perimeter_l3179_317904


namespace average_weight_ab_is_40_l3179_317951

def average_weight_abc : ℝ := 42
def average_weight_bc : ℝ := 43
def weight_b : ℝ := 40

theorem average_weight_ab_is_40 :
  let weight_c := 2 * average_weight_bc - weight_b
  let weight_a := 3 * average_weight_abc - weight_b - weight_c
  (weight_a + weight_b) / 2 = 40 := by sorry

end average_weight_ab_is_40_l3179_317951


namespace total_books_count_l3179_317916

-- Define the number of books per shelf
def booksPerShelf : ℕ := 6

-- Define the number of shelves for each category
def mysteryShelvesCount : ℕ := 8
def pictureShelvesCount : ℕ := 5
def sciFiShelvesCount : ℕ := 4
def nonFictionShelvesCount : ℕ := 3

-- Define the total number of books
def totalBooks : ℕ := 
  booksPerShelf * (mysteryShelvesCount + pictureShelvesCount + sciFiShelvesCount + nonFictionShelvesCount)

-- Theorem statement
theorem total_books_count : totalBooks = 120 := by
  sorry

end total_books_count_l3179_317916


namespace five_solutions_l3179_317940

/-- The system of equations has exactly 5 distinct real solutions -/
theorem five_solutions (x y z w : ℝ) : 
  (x = w + z + z*w*x) ∧
  (y = z + x + z*x*y) ∧
  (z = x + y + x*y*z) ∧
  (w = y + z + y*z*w) →
  ∃! (sol : Finset (ℝ × ℝ × ℝ × ℝ)), sol.card = 5 ∧ ∀ (a b c d : ℝ), (a, b, c, d) ∈ sol ↔
    (a = d + c + c*d*a) ∧
    (b = c + a + c*a*b) ∧
    (c = a + b + a*b*c) ∧
    (d = b + c + b*c*d) := by
  sorry

end five_solutions_l3179_317940


namespace integral_sqrt_plus_linear_l3179_317925

theorem integral_sqrt_plus_linear (f g : ℝ → ℝ) :
  (∫ x in (0 : ℝ)..1, (Real.sqrt (1 - x^2) + 3*x)) = π/4 + 3/2 := by sorry

end integral_sqrt_plus_linear_l3179_317925


namespace sqrt_expression_equality_l3179_317915

theorem sqrt_expression_equality : 
  (Real.sqrt 3 + Real.sqrt 2)^2 - (Real.sqrt 3 - Real.sqrt 2) * (Real.sqrt 3 + Real.sqrt 2) = 4 + 2 * Real.sqrt 6 := by
  sorry

end sqrt_expression_equality_l3179_317915


namespace range_of_a_l3179_317974

def equation1 (a x : ℝ) : Prop := x^2 + 4*a*x - 4*a + 3 = 0

def equation2 (a x : ℝ) : Prop := x^2 + (a-1)*x + a^2 = 0

def equation3 (a x : ℝ) : Prop := x^2 + 2*a*x - 2*a = 0

def has_real_root (a : ℝ) : Prop :=
  ∃ x : ℝ, equation1 a x ∨ equation2 a x ∨ equation3 a x

theorem range_of_a : ∀ a : ℝ, has_real_root a ↔ a ≥ -1 ∨ a ≤ -3/2 :=
sorry

end range_of_a_l3179_317974


namespace fraction_equality_l3179_317922

theorem fraction_equality (a : ℕ+) : 
  (a : ℚ) / (a + 27 : ℚ) = 865 / 1000 → a = 173 := by
  sorry

end fraction_equality_l3179_317922


namespace hall_length_l3179_317901

/-- The length of a rectangular hall given its width, height, and total area to be covered. -/
theorem hall_length (width height total_area : ℝ) (hw : width = 15) (hh : height = 5) 
  (ha : total_area = 950) : 
  ∃ length : ℝ, length = 32 ∧ total_area = length * width + 2 * (height * length + height * width) :=
by sorry

end hall_length_l3179_317901


namespace jack_additional_apples_l3179_317988

/-- Represents the capacity of apple baskets and current apple counts -/
structure AppleBaskets where
  jack_capacity : ℕ
  jill_capacity : ℕ
  jack_current : ℕ

/-- The conditions of the apple picking problem -/
def apple_picking_conditions (ab : AppleBaskets) : Prop :=
  ab.jill_capacity = 2 * ab.jack_capacity ∧
  ab.jack_capacity = 12 ∧
  3 * ab.jack_current = ab.jill_capacity

/-- The theorem stating how many more apples Jack's basket can hold -/
theorem jack_additional_apples (ab : AppleBaskets) 
  (h : apple_picking_conditions ab) : 
  ab.jack_capacity - ab.jack_current = 4 := by
  sorry


end jack_additional_apples_l3179_317988


namespace distinct_points_on_curve_l3179_317963

theorem distinct_points_on_curve : ∃ (a b : ℝ), 
  a ≠ b ∧ 
  (a^3 + Real.sqrt e^4 = 2 * (Real.sqrt e)^2 * a + 1) ∧
  (b^3 + Real.sqrt e^4 = 2 * (Real.sqrt e)^2 * b + 1) ∧
  |a - b| = 3 := by
  sorry

end distinct_points_on_curve_l3179_317963


namespace expanded_product_terms_l3179_317981

theorem expanded_product_terms (a b c : ℕ) (ha : a = 6) (hb : b = 7) (hc : c = 5) :
  a * b * c = 210 := by
  sorry

end expanded_product_terms_l3179_317981


namespace triangle_abc_properties_l3179_317983

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  B = π/4 ∧ 
  b = Real.sqrt 10 ∧
  Real.cos C = 2 * Real.sqrt 5 / 5 →
  Real.sin A = 3 * Real.sqrt 10 / 10 ∧
  a = 3 * Real.sqrt 2 ∧
  1/2 * a * b * Real.sin C = 3 :=
by sorry

end triangle_abc_properties_l3179_317983


namespace probability_no_adjacent_standing_is_correct_l3179_317995

/-- Represents the number of valid arrangements for n people where no two adjacent people stand. -/
def validArrangements : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n + 2 => validArrangements (n + 1) + validArrangements n

/-- The number of people sitting around the circular table. -/
def numPeople : ℕ := 10

/-- The probability of no two adjacent people standing when numPeople flip fair coins. -/
def probabilityNoAdjacentStanding : ℚ :=
  validArrangements numPeople / 2^numPeople

theorem probability_no_adjacent_standing_is_correct :
  probabilityNoAdjacentStanding = 123 / 1024 := by
  sorry

end probability_no_adjacent_standing_is_correct_l3179_317995


namespace base_12_remainder_l3179_317965

/-- Converts a base-12 number to base-10 --/
def base12ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (12 ^ i)) 0

/-- The problem statement --/
theorem base_12_remainder (digits : List Nat) (h : digits = [3, 4, 7, 2]) :
  base12ToBase10 digits % 10 = 5 := by
  sorry

#eval base12ToBase10 [3, 4, 7, 2]

end base_12_remainder_l3179_317965


namespace base4_multiplication_division_l3179_317929

-- Define a function to convert from base 4 to base 10
def base4ToBase10 (n : ℕ) : ℕ := sorry

-- Define a function to convert from base 10 to base 4
def base10ToBase4 (n : ℕ) : ℕ := sorry

-- Define the theorem
theorem base4_multiplication_division :
  base10ToBase4 ((base4ToBase10 131 * base4ToBase10 21) / base4ToBase10 3) = 1113 := by sorry

end base4_multiplication_division_l3179_317929


namespace equidistant_point_x_coordinate_l3179_317906

theorem equidistant_point_x_coordinate :
  ∃ (x y : ℝ),
    (abs y = abs x) ∧  -- Distance from y-axis equals distance from x-axis
    (abs y = abs ((x + y - 4) / Real.sqrt 2)) ∧  -- Distance from x-axis equals distance from line x + y = 4
    (x = 2) := by
  sorry

end equidistant_point_x_coordinate_l3179_317906


namespace dodecahedron_diagonals_l3179_317935

/-- Represents a dodecahedron -/
structure Dodecahedron where
  vertices : Finset (Fin 20)
  faces : Finset (Finset (Fin 5))
  vertex_face_incidence : vertices → Finset faces
  diag : Fin 20 → Fin 20 → Prop

/-- Properties of a dodecahedron -/
axiom dodecahedron_properties (D : Dodecahedron) :
  (D.vertices.card = 20) ∧
  (D.faces.card = 12) ∧
  (∀ v : D.vertices, (D.vertex_face_incidence v).card = 3) ∧
  (∀ v w : D.vertices, D.diag v w ↔ v ≠ w ∧ (D.vertex_face_incidence v ∩ D.vertex_face_incidence w).card = 0)

/-- The number of interior diagonals in a dodecahedron -/
def interior_diagonals (D : Dodecahedron) : ℕ :=
  (D.vertices.card * (D.vertices.card - 4)) / 2

/-- Theorem: A dodecahedron has 160 interior diagonals -/
theorem dodecahedron_diagonals (D : Dodecahedron) : interior_diagonals D = 160 := by
  sorry

end dodecahedron_diagonals_l3179_317935


namespace ryan_learning_days_l3179_317984

def daily_english_hours : ℕ := 6
def daily_chinese_hours : ℕ := 7
def total_hours : ℕ := 65

theorem ryan_learning_days : 
  total_hours / (daily_english_hours + daily_chinese_hours) = 5 := by
sorry

end ryan_learning_days_l3179_317984


namespace bus_stop_problem_l3179_317993

/-- The number of students who got off the bus at the first stop -/
def students_who_got_off (initial_students : ℕ) (remaining_students : ℕ) : ℕ :=
  initial_students - remaining_students

theorem bus_stop_problem (initial_students remaining_students : ℕ) 
  (h1 : initial_students = 10)
  (h2 : remaining_students = 7) :
  students_who_got_off initial_students remaining_students = 3 := by
  sorry

end bus_stop_problem_l3179_317993


namespace square_roots_problem_l3179_317903

theorem square_roots_problem (a : ℝ) (n : ℝ) :
  n > 0 ∧ 
  (∃ x y : ℝ, x * x = n ∧ y * y = n ∧ x = a ∧ y = 2 * a - 6) →
  a = 6 ∧ 
  n = 36 ∧
  (∃ b : ℝ, b * b * b = 10 * 2 + 7 ∧ b = 3) :=
by sorry

end square_roots_problem_l3179_317903


namespace minimal_distance_point_l3179_317933

/-- The point that minimizes the sum of distances to two fixed points on a given line -/
theorem minimal_distance_point 
  (A B P : ℝ × ℝ) 
  (h_A : A = (-3, 1)) 
  (h_B : B = (5, -1)) 
  (h_P : P.2 = -2) : 
  (P = (3, -2)) ↔ 
  (∀ Q : ℝ × ℝ, Q.2 = -2 → 
    Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) + Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) ≤ 
    Real.sqrt ((Q.1 - A.1)^2 + (Q.2 - A.2)^2) + Real.sqrt ((Q.1 - B.1)^2 + (Q.2 - B.2)^2)) :=
by sorry

end minimal_distance_point_l3179_317933


namespace min_sum_of_cubes_when_sum_is_eight_l3179_317976

theorem min_sum_of_cubes_when_sum_is_eight :
  ∀ x y : ℝ, x + y = 8 →
  x^3 + y^3 ≥ 4^3 + 4^3 :=
by sorry

end min_sum_of_cubes_when_sum_is_eight_l3179_317976


namespace mascot_sales_equation_l3179_317943

/-- Represents the sales growth of a mascot over two months -/
def sales_growth (initial_sales : ℝ) (final_sales : ℝ) (growth_rate : ℝ) : Prop :=
  initial_sales * (1 + growth_rate)^2 = final_sales

/-- Theorem stating the correct equation for the given sales scenario -/
theorem mascot_sales_equation :
  ∀ (x : ℝ), x > 0 →
  sales_growth 10 11.5 x :=
by
  sorry

end mascot_sales_equation_l3179_317943


namespace johns_purchase_cost_l3179_317962

/-- Calculates the total cost of John's metal purchase in USD -/
def total_cost (silver_oz : ℝ) (gold_oz : ℝ) (platinum_oz : ℝ) 
                (silver_price_usd : ℝ) (gold_multiplier : ℝ) 
                (platinum_price_gbp : ℝ) (usd_gbp_rate : ℝ) : ℝ :=
  let silver_cost := silver_oz * silver_price_usd
  let gold_cost := gold_oz * (silver_price_usd * gold_multiplier)
  let platinum_cost := platinum_oz * (platinum_price_gbp * usd_gbp_rate)
  silver_cost + gold_cost + platinum_cost

/-- Theorem stating that John's total cost is $5780.5 -/
theorem johns_purchase_cost : 
  total_cost 2.5 3.5 4.5 25 60 80 1.3 = 5780.5 := by
  sorry

end johns_purchase_cost_l3179_317962


namespace shape_is_regular_tetrahedron_l3179_317905

/-- A 3D shape with diagonals of adjacent sides meeting at a 60-degree angle -/
structure Shape3D where
  diagonalAngle : ℝ
  diagonalAngleIs60 : diagonalAngle = 60

/-- Definition of a regular tetrahedron -/
def RegularTetrahedron : Type := Unit

/-- Theorem: A 3D shape with diagonals of adjacent sides meeting at a 60-degree angle is a regular tetrahedron -/
theorem shape_is_regular_tetrahedron (s : Shape3D) : RegularTetrahedron := by
  sorry

end shape_is_regular_tetrahedron_l3179_317905


namespace price_increase_percentage_l3179_317917

theorem price_increase_percentage (old_price new_price : ℝ) (h1 : old_price = 300) (h2 : new_price = 450) :
  (new_price - old_price) / old_price * 100 = 50 := by
  sorry

end price_increase_percentage_l3179_317917


namespace angle_supplement_theorem_l3179_317957

-- Define a type for angles in degrees and minutes
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

-- Define the complement of an angle
def complement (α : Angle) : Angle :=
  { degrees := 90 - α.degrees - 1,
    minutes := 60 - α.minutes }

-- Define the supplement of an angle
def supplement (α : Angle) : Angle :=
  { degrees := 180 - α.degrees - 1,
    minutes := 60 - α.minutes }

theorem angle_supplement_theorem (α : Angle) :
  complement α = { degrees := 54, minutes := 32 } →
  supplement α = { degrees := 144, minutes := 32 } :=
by sorry

end angle_supplement_theorem_l3179_317957


namespace amount_with_r_l3179_317924

theorem amount_with_r (total : ℝ) (p q r : ℝ) : 
  total = 4000 →
  p + q + r = total →
  r = (2/3) * (p + q) →
  r = 1600 := by
sorry

end amount_with_r_l3179_317924


namespace sector_area_l3179_317960

/-- The area of a circular sector with a central angle of 150° and a radius of √3 is 5π/4 -/
theorem sector_area (α : Real) (r : Real) : 
  α = 150 * π / 180 →  -- Convert 150° to radians
  r = Real.sqrt 3 →
  (1 / 2) * α * r^2 = (5 * π) / 4 := by
  sorry

end sector_area_l3179_317960


namespace sums_are_equal_l3179_317936

def sum1 : ℕ :=
  1 + 22 + 333 + 4444 + 55555 + 666666 + 7777777 + 88888888 + 999999999

def sum2 : ℕ :=
  9 + 98 + 987 + 9876 + 98765 + 987654 + 9876543 + 98765432 + 987654321

theorem sums_are_equal : sum1 = sum2 := by
  sorry

end sums_are_equal_l3179_317936


namespace bag_counter_problem_l3179_317958

theorem bag_counter_problem (Y X : ℕ) : 
  (Y > 0) →  -- Y is positive
  (X > 0) →  -- X is positive
  (Y / (Y + 10) = (Y + 2) / (X + Y + 12)) →  -- Proportion remains unchanged
  (Y * X = 20) →  -- Derived from the equality of proportions
  (Y = 1 ∨ Y = 2 ∨ Y = 4 ∨ Y = 5 ∨ Y = 10 ∨ Y = 20) :=
by sorry

end bag_counter_problem_l3179_317958


namespace bread_slice_cost_l3179_317931

-- Define the problem parameters
def num_loaves : ℕ := 3
def slices_per_loaf : ℕ := 20
def payment_amount : ℕ := 40  -- in dollars
def change_received : ℕ := 16  -- in dollars

-- Define the theorem
theorem bread_slice_cost :
  let total_cost : ℕ := payment_amount - change_received
  let total_slices : ℕ := num_loaves * slices_per_loaf
  let cost_per_slice_cents : ℕ := (total_cost * 100) / total_slices
  cost_per_slice_cents = 40 := by
  sorry

end bread_slice_cost_l3179_317931


namespace middle_of_five_consecutive_integers_l3179_317921

/-- Given 5 consecutive integers with a sum of 60, prove that the middle number is 12 -/
theorem middle_of_five_consecutive_integers (a b c d e : ℤ) : 
  (a + b + c + d + e = 60) → 
  (b = a + 1) → (c = b + 1) → (d = c + 1) → (e = d + 1) →
  c = 12 := by
sorry

end middle_of_five_consecutive_integers_l3179_317921


namespace guitar_sales_l3179_317913

theorem guitar_sales (total_revenue : ℕ) (electric_price acoustic_price : ℕ) (electric_sold : ℕ) : 
  total_revenue = 3611 →
  electric_price = 479 →
  acoustic_price = 339 →
  electric_sold = 4 →
  ∃ (acoustic_sold : ℕ), electric_sold + acoustic_sold = 9 ∧ 
    electric_sold * electric_price + acoustic_sold * acoustic_price = total_revenue := by
  sorry

end guitar_sales_l3179_317913


namespace order_of_abc_l3179_317945

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define a, b, and c as real numbers
variable (a b c : ℝ)

-- State the theorem
theorem order_of_abc (hf : Monotone f) (ha : a = f 2 ∧ a < 0) 
  (hb : f b = 2) (hc : f c = 0) : b > c ∧ c > a := by
  sorry

end order_of_abc_l3179_317945


namespace mosaic_completion_time_l3179_317954

-- Define the start time
def start_time : ℕ := 9 * 60  -- 9:00 AM in minutes since midnight

-- Define the time when 1/4 of the mosaic is completed
def quarter_time : ℕ := (12 + 12) * 60 + 45  -- 12:45 PM in minutes since midnight

-- Define the fraction of work completed
def fraction_completed : ℚ := 1/4

-- Define the duration to complete 1/4 of the mosaic
def quarter_duration : ℕ := quarter_time - start_time

-- Theorem to prove
theorem mosaic_completion_time :
  let total_duration : ℕ := (quarter_duration * 4)
  let finish_time : ℕ := (start_time + total_duration) % (24 * 60)
  finish_time = 0  -- 0 minutes past midnight (12:00 AM)
  := by sorry

end mosaic_completion_time_l3179_317954


namespace max_ab_bisecting_line_l3179_317971

theorem max_ab_bisecting_line (a b : ℝ) : 
  (∀ x y : ℝ, 2*a*x - b*y + 2 = 0 → (x+1)^2 + (y-2)^2 = 4) → 
  (a * b ≤ 1/4) ∧ (∃ a₀ b₀ : ℝ, a₀ * b₀ = 1/4 ∧ 
    (∀ x y : ℝ, 2*a₀*x - b₀*y + 2 = 0 → (x+1)^2 + (y-2)^2 = 4)) :=
by sorry

end max_ab_bisecting_line_l3179_317971


namespace sphere_radius_when_volume_equals_surface_area_l3179_317998

theorem sphere_radius_when_volume_equals_surface_area :
  ∀ r : ℝ,
  (4 / 3) * Real.pi * r^3 = 4 * Real.pi * r^2 →
  r = 3 :=
by sorry

end sphere_radius_when_volume_equals_surface_area_l3179_317998


namespace negation_of_universal_proposition_l3179_317956

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℕ, x > 0) ↔ (∃ x : ℕ, x ≤ 0) := by sorry

end negation_of_universal_proposition_l3179_317956


namespace expression_evaluation_l3179_317973

theorem expression_evaluation : 
  (2024^3 - 3 * 2024^2 * 2025 + 4 * 2024 * 2025^2 - 2025^3 + 2) / (2024 * 2025) = 2025 - 1 / (2024 * 2025) := by
  sorry

end expression_evaluation_l3179_317973


namespace total_balls_in_box_l3179_317900

/-- Given a box with blue and red balls, calculate the total number of balls -/
theorem total_balls_in_box (blue_balls : ℕ) (red_balls : ℕ) : 
  blue_balls = 3 → red_balls = 2 → blue_balls + red_balls = 5 := by
  sorry

end total_balls_in_box_l3179_317900


namespace prob_three_correct_five_l3179_317948

/-- The number of houses and packages --/
def n : ℕ := 5

/-- The probability of exactly 3 out of n packages being delivered to the correct houses --/
def prob_three_correct (n : ℕ) : ℚ :=
  (n.choose 3 : ℚ) * (1 / n) * (1 / (n - 1)) * (1 / (n - 2)) * (1 / 2)

/-- Theorem stating that the probability of exactly 3 out of 5 packages 
    being delivered to the correct houses is 1/12 --/
theorem prob_three_correct_five : prob_three_correct n = 1 / 12 := by
  sorry

end prob_three_correct_five_l3179_317948


namespace marker_difference_l3179_317928

theorem marker_difference (price : ℚ) (hector_count alicia_count : ℕ) : 
  price > 1/100 →  -- More than a penny each
  price * hector_count = 276/100 →  -- Hector paid $2.76
  price * alicia_count = 407/100 →  -- Alicia paid $4.07
  alicia_count - hector_count = 13 := by
  sorry

end marker_difference_l3179_317928


namespace other_diagonal_length_l3179_317927

/-- Represents a rhombus with given diagonals and area -/
structure Rhombus where
  d1 : ℝ  -- Length of the first diagonal
  d2 : ℝ  -- Length of the second diagonal
  area : ℝ -- Area of the rhombus

/-- The area of a rhombus is half the product of its diagonals -/
axiom rhombus_area (r : Rhombus) : r.area = (r.d1 * r.d2) / 2

/-- Given a rhombus with one diagonal of 15 cm and an area of 90 cm²,
    the length of the other diagonal is 12 cm -/
theorem other_diagonal_length :
  ∀ r : Rhombus, r.d1 = 15 ∧ r.area = 90 → r.d2 = 12 := by
  sorry

end other_diagonal_length_l3179_317927


namespace ice_cream_jog_speed_l3179_317930

/-- Calculates the required speed in miles per hour to cover a given distance within a time limit -/
def required_speed (time_limit : ℚ) (distance_blocks : ℕ) (block_length : ℚ) : ℚ :=
  (distance_blocks : ℚ) * block_length * (60 / time_limit)

theorem ice_cream_jog_speed :
  let time_limit : ℚ := 10  -- Time limit in minutes
  let distance_blocks : ℕ := 16  -- Distance in blocks
  let block_length : ℚ := 1/8  -- Length of each block in miles
  required_speed time_limit distance_blocks block_length = 12 := by
sorry

end ice_cream_jog_speed_l3179_317930


namespace equation_solution_l3179_317914

theorem equation_solution (x : ℝ) : 
  (3 * x + 25 ≠ 0) → 
  ((8 * x^2 + 75 * x - 3) / (3 * x + 25) = 2 * x + 5 ↔ x = -16 ∨ x = 4) :=
by sorry

end equation_solution_l3179_317914


namespace system_solution_l3179_317986

theorem system_solution (x y k : ℝ) 
  (eq1 : 2 * x - y = 5 * k + 6)
  (eq2 : 4 * x + 7 * y = k)
  (eq3 : x + y = 2024) :
  k = 2023 := by
sorry

end system_solution_l3179_317986


namespace certain_number_proof_l3179_317911

theorem certain_number_proof (x q : ℝ) 
  (h1 : 3 / x = 8)
  (h2 : 3 / q = 18)
  (h3 : x - q = 0.20833333333333334) :
  x = 0.375 := by
sorry

end certain_number_proof_l3179_317911


namespace irrational_density_l3179_317912

theorem irrational_density (α : ℝ) (h_irrational : Irrational α) (a b : ℝ) (h_lt : a < b) :
  ∃ (m n : ℕ), a < m * α - n ∧ m * α - n < b :=
sorry

end irrational_density_l3179_317912


namespace inequality_holds_l3179_317959

theorem inequality_holds (a b : ℝ) (h1 : a > 1) (h2 : 1 > b) (h3 : b > 0) :
  (1 / Real.log a) > (1 / Real.log b) := by
  sorry

end inequality_holds_l3179_317959


namespace fencing_length_l3179_317944

/-- Calculates the required fencing length for a rectangular field -/
theorem fencing_length (area : ℝ) (uncovered_side : ℝ) : area = 680 ∧ uncovered_side = 40 →
  2 * (area / uncovered_side) + uncovered_side = 74 := by
  sorry


end fencing_length_l3179_317944


namespace intersection_empty_at_m_zero_l3179_317964

theorem intersection_empty_at_m_zero :
  ∃ m : ℝ, m = 0 ∧ (Set.Icc 0 1 : Set ℝ) ∩ {x : ℝ | x^2 - 2*x + m > 0} = ∅ :=
by sorry

end intersection_empty_at_m_zero_l3179_317964


namespace triangle_area_problem_l3179_317968

theorem triangle_area_problem (base_small : ℝ) (base_large : ℝ) (area_small : ℝ) :
  base_small = 14 →
  base_large = 24 →
  area_small = 35 →
  let height_small := (2 * area_small) / base_small
  let height_large := (height_small * base_large) / base_small
  (1/2 : ℝ) * base_large * height_large = 144 :=
by sorry

end triangle_area_problem_l3179_317968


namespace pears_picked_by_keith_l3179_317952

/-- The number of pears Keith picked -/
def keiths_pears : ℝ := 0

theorem pears_picked_by_keith :
  let mikes_apples : ℝ := 7.0
  let nancys_eaten_apples : ℝ := 3.0
  let keiths_apples : ℝ := 6.0
  let apples_left : ℝ := 10.0
  keiths_pears = 0 := by sorry

end pears_picked_by_keith_l3179_317952


namespace smallest_with_20_divisors_l3179_317947

/-- The number of positive divisors of a positive integer n -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- n has exactly 20 positive divisors -/
def has_20_divisors (n : ℕ+) : Prop := num_divisors n = 20

theorem smallest_with_20_divisors : 
  has_20_divisors 432 ∧ ∀ m : ℕ+, m < 432 → ¬(has_20_divisors m) := by sorry

end smallest_with_20_divisors_l3179_317947


namespace range_of_a_theorem_l3179_317997

/-- Proposition p: For any x ∈ ℝ, x^2 - 2x > a -/
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*x > a

/-- Proposition q: The function f(x) = x^2 + 2ax + 2 - a has a zero point on ℝ -/
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

/-- The range of values for a given the conditions -/
def range_of_a : Set ℝ := {a | a ∈ Set.Ioo (-2) (-1) ∨ a ∈ Set.Ici 1}

theorem range_of_a_theorem (a : ℝ) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → a ∈ range_of_a := by
  sorry

end range_of_a_theorem_l3179_317997


namespace mapping_properties_l3179_317987

-- Define the sets A and B
variable {A B : Type}

-- Define the mapping f from A to B
variable (f : A → B)

-- Theorem stating the properties of the mapping
theorem mapping_properties :
  (∀ a : A, ∃! b : B, f a = b) ∧
  (∃ a₁ a₂ : A, a₁ ≠ a₂ ∧ f a₁ = f a₂) :=
by sorry

end mapping_properties_l3179_317987
