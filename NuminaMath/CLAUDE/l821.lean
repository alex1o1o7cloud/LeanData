import Mathlib

namespace stable_polygon_sides_l821_82195

/-- A polygon is stable if connecting all vertices from a point on one of its edges
    (not a vertex) results in a certain number of triangles. -/
def is_stable_polygon (n : ℕ) (t : ℕ) : Prop :=
  n > 2 ∧ t = n - 1

theorem stable_polygon_sides :
  ∀ n : ℕ, is_stable_polygon n 2022 → n = 2023 :=
by sorry

end stable_polygon_sides_l821_82195


namespace jerry_money_left_l821_82166

/-- Calculates the amount of money Jerry has left after grocery shopping --/
def money_left (budget : ℚ) (mustard_oil_price : ℚ) (mustard_oil_quantity : ℚ) 
  (mustard_oil_discount : ℚ) (pasta_price : ℚ) (pasta_quantity : ℚ) 
  (pasta_sauce_price : ℚ) (pasta_sauce_quantity : ℚ) : ℚ :=
  let mustard_oil_cost := mustard_oil_price * mustard_oil_quantity * (1 - mustard_oil_discount)
  let pasta_cost := pasta_price * (pasta_quantity - 1)  -- Buy 2, Get the 3rd one free
  let pasta_sauce_cost := pasta_sauce_price * pasta_sauce_quantity
  budget - (mustard_oil_cost + pasta_cost + pasta_sauce_cost)

theorem jerry_money_left :
  money_left 100 13 2 0.1 4 3 5 1 = 63.6 := by
  sorry

end jerry_money_left_l821_82166


namespace inequality_proof_l821_82186

theorem inequality_proof (a b c : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)
  (sum_cond : a + b + c = 2) :
  (1 / (1 + a*b)) + (1 / (1 + b*c)) + (1 / (1 + c*a)) ≥ 27/13 ∧
  ((1 / (1 + a*b)) + (1 / (1 + b*c)) + (1 / (1 + c*a)) = 27/13 ↔ a = 2/3 ∧ b = 2/3 ∧ c = 2/3) :=
by sorry

end inequality_proof_l821_82186


namespace factorization_equality_l821_82123

theorem factorization_equality (x y : ℝ) : 2 * x^3 - 8 * x^2 * y + 8 * x * y^2 = 2 * x * (x - 2*y)^2 := by
  sorry

end factorization_equality_l821_82123


namespace unique_a_value_l821_82161

def A (a : ℝ) : Set ℝ := {-4, 2*a-1, a^2}
def B (a : ℝ) : Set ℝ := {a-5, 1-a, 9}

theorem unique_a_value : ∃! a : ℝ, A a ∩ B a = {9} := by sorry

end unique_a_value_l821_82161


namespace farm_fencing_cost_l821_82114

/-- Proves that the cost of fencing per meter is 15 for a rectangular farm with given conditions -/
theorem farm_fencing_cost (area : ℝ) (short_side : ℝ) (total_cost : ℝ) :
  area = 1200 →
  short_side = 30 →
  total_cost = 1800 →
  let long_side := area / short_side
  let diagonal := Real.sqrt (long_side ^ 2 + short_side ^ 2)
  let total_length := long_side + short_side + diagonal
  total_cost / total_length = 15 := by
  sorry

end farm_fencing_cost_l821_82114


namespace shirt_price_calculation_l821_82137

/-- The price of a single shirt -/
def shirt_price : ℝ := 45

/-- The price of a single pair of pants -/
def pants_price : ℝ := 25

/-- The total cost of the initial purchase -/
def total_cost : ℝ := 120

/-- The refund percentage -/
def refund_percentage : ℝ := 0.25

theorem shirt_price_calculation :
  (∀ s₁ s₂ : ℝ, s₁ = s₂ → s₁ = shirt_price) →  -- All shirts have the same price
  (∀ p₁ p₂ : ℝ, p₁ = p₂ → p₁ = pants_price) →  -- All pants have the same price
  shirt_price ≠ pants_price →                  -- Shirt price ≠ pants price
  2 * shirt_price + 3 * pants_price = total_cost →  -- 2 shirts + 3 pants = $120
  3 * pants_price = refund_percentage * total_cost →  -- Refund for 3 pants = 25% of $120
  shirt_price = 45 := by sorry

end shirt_price_calculation_l821_82137


namespace smallest_even_abundant_after_12_l821_82174

def is_abundant (n : ℕ) : Prop :=
  n < (Finset.sum (Finset.filter (λ x => x < n ∧ n % x = 0) (Finset.range n)) id)

theorem smallest_even_abundant_after_12 :
  ∀ n : ℕ, n > 12 → n % 2 = 0 → is_abundant n → n ≥ 18 :=
sorry

end smallest_even_abundant_after_12_l821_82174


namespace union_of_M_and_N_l821_82150

def M : Set ℝ := {x | x^2 = 1}
def N : Set ℝ := {1, 2}

theorem union_of_M_and_N : M ∪ N = {-1, 1, 2} := by sorry

end union_of_M_and_N_l821_82150


namespace geometric_sequence_term_number_l821_82160

theorem geometric_sequence_term_number (a₁ q aₙ : ℚ) (n : ℕ) :
  a₁ = 1/2 →
  q = 1/2 →
  aₙ = 1/64 →
  aₙ = a₁ * q^(n - 1) →
  n = 6 :=
by sorry

end geometric_sequence_term_number_l821_82160


namespace sum_of_numbers_l821_82196

theorem sum_of_numbers (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_prod : x * y = 16) (h_recip : 1 / x = 3 / y) : 
  x + y = 16 * Real.sqrt 3 / 3 := by
  sorry

end sum_of_numbers_l821_82196


namespace cut_string_theorem_l821_82151

/-- Represents the number of pieces resulting from cutting a string at all points marked by two different equal-spacing schemes. -/
def cut_string_pieces (total_length : ℝ) (divisions1 divisions2 : ℕ) : ℕ :=
  let marks1 := divisions1 - 1
  let marks2 := divisions2 - 1
  marks1 + marks2 + 1

/-- Theorem stating that cutting a string at points marked for 9 equal pieces and 8 equal pieces results in 16 pieces. -/
theorem cut_string_theorem : cut_string_pieces 1 9 8 = 16 := by
  sorry

#eval cut_string_pieces 1 9 8

end cut_string_theorem_l821_82151


namespace scientific_notation_45400_l821_82129

theorem scientific_notation_45400 : 
  ∃ (a : ℝ) (n : ℤ), 45400 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 4.54 ∧ n = 4 :=
sorry

end scientific_notation_45400_l821_82129


namespace probability_sum_17_l821_82141

def roll_sum_17 : ℕ := 56

def total_outcomes : ℕ := 6^4

theorem probability_sum_17 : 
  (roll_sum_17 : ℚ) / total_outcomes = 7 / 162 :=
sorry

end probability_sum_17_l821_82141


namespace g_of_2_eq_6_l821_82132

def g (x : ℝ) : ℝ := x^3 - x

theorem g_of_2_eq_6 : g 2 = 6 := by sorry

end g_of_2_eq_6_l821_82132


namespace xiao_wang_exam_scores_xiao_wang_final_results_l821_82199

theorem xiao_wang_exam_scores :
  ∀ (x y : ℝ),
  (x * y + 98) / (x + 1) = y + 1 →
  (x * y + 98 + 70) / (x + 2) = y - 1 →
  x = 8 ∧ y = 89 :=
by
  sorry

theorem xiao_wang_final_results (x y : ℝ) 
  (h : x = 8 ∧ y = 89) :
  (x + 2 : ℝ) = 10 ∧ (y - 1 : ℝ) = 88 :=
by
  sorry

end xiao_wang_exam_scores_xiao_wang_final_results_l821_82199


namespace lowest_common_multiple_8_12_l821_82156

theorem lowest_common_multiple_8_12 : ∃ n : ℕ, n > 0 ∧ 8 ∣ n ∧ 12 ∣ n ∧ ∀ m : ℕ, m > 0 → 8 ∣ m → 12 ∣ m → n ≤ m := by
  sorry

end lowest_common_multiple_8_12_l821_82156


namespace common_number_in_list_l821_82146

theorem common_number_in_list (l : List ℝ) : 
  l.length = 7 →
  (l.take 4).sum / 4 = 6 →
  (l.drop 3).sum / 4 = 9 →
  l.sum / 7 = 55 / 7 →
  ∃ x ∈ l.take 4 ∩ l.drop 3, x = 5 :=
by sorry

end common_number_in_list_l821_82146


namespace cubic_sum_greater_than_mixed_product_l821_82147

theorem cubic_sum_greater_than_mixed_product (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a ≠ b) : a^3 + b^3 > a^2*b + a*b^2 := by
  sorry

end cubic_sum_greater_than_mixed_product_l821_82147


namespace direct_proportion_when_constant_quotient_l821_82115

-- Define the variables
variable (a b c : ℝ)

-- Define the theorem
theorem direct_proportion_when_constant_quotient :
  (∀ x y : ℝ, x ≠ 0 → a = y / x → c ≠ 0 → a = b / c) →
  (∃ k : ℝ, ∀ x y : ℝ, x ≠ 0 → a = y / x → y = k * x) :=
sorry

end direct_proportion_when_constant_quotient_l821_82115


namespace not_prime_a_l821_82170

theorem not_prime_a (a b : ℕ+) (h : ∃ k : ℤ, k * (b.val^4 + 3*b.val^2 + 4) = 5*a.val^4 + a.val^2) : 
  ¬ Nat.Prime a.val := by
sorry

end not_prime_a_l821_82170


namespace count_words_l821_82121

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 6

/-- The maximum word length -/
def max_word_length : ℕ := 3

/-- Counts the number of words with exactly one letter -/
def one_letter_words : ℕ := alphabet_size

/-- Counts the number of words with exactly two letters -/
def two_letter_words : ℕ := alphabet_size * alphabet_size

/-- Counts the number of three-letter words with all letters the same -/
def three_same_letter_words : ℕ := alphabet_size

/-- Counts the number of three-letter words with exactly two letters the same -/
def three_two_same_letter_words : ℕ := alphabet_size * (alphabet_size - 1) * 3

/-- The total number of words in the language -/
def total_words : ℕ := one_letter_words + two_letter_words + three_same_letter_words + three_two_same_letter_words

/-- Theorem stating that the total number of words is 138 -/
theorem count_words : total_words = 138 := by
  sorry

end count_words_l821_82121


namespace multiply_mistake_l821_82107

theorem multiply_mistake (x : ℝ) : 97 * x - 89 * x = 4926 → x = 615.75 := by
  sorry

end multiply_mistake_l821_82107


namespace expression_evaluation_l821_82188

theorem expression_evaluation : 
  2016 / ((13 + 5/7) - (8 + 8/11)) * (5/7 - 5/11) = 105 := by sorry

end expression_evaluation_l821_82188


namespace jessica_letter_paper_weight_l821_82145

/-- The weight of each piece of paper in Jessica's letter -/
def paper_weight (num_papers : ℕ) (envelope_weight total_weight : ℚ) : ℚ :=
  (total_weight - envelope_weight) / num_papers

/-- Theorem stating that each piece of paper in Jessica's letter weighs 1/5 of an ounce -/
theorem jessica_letter_paper_weight :
  let num_papers : ℕ := 8
  let envelope_weight : ℚ := 2/5
  let total_weight : ℚ := 2
  paper_weight num_papers envelope_weight total_weight = 1/5 := by
  sorry

end jessica_letter_paper_weight_l821_82145


namespace repeating_decimal_three_thirty_six_l821_82138

/-- The repeating decimal 3.363636... is equal to 10/3 -/
theorem repeating_decimal_three_thirty_six : ∃ (x : ℚ), x = 10 / 3 ∧ x = 3 + (36 / 99) := by sorry

end repeating_decimal_three_thirty_six_l821_82138


namespace inequality_proof_l821_82103

theorem inequality_proof (k n : ℕ) (hk : k > 0) (hn : n > 0) (hkn : k ≤ n) :
  1 + (k : ℝ) / n ≤ (1 + 1 / n) ^ k ∧ (1 + 1 / n) ^ k < 1 + (k : ℝ) / n + (k : ℝ)^2 / n^2 := by
  sorry

end inequality_proof_l821_82103


namespace magnitude_of_complex_fraction_l821_82177

theorem magnitude_of_complex_fraction :
  let z : ℂ := (1 + Complex.I) / (2 - 2 * Complex.I)
  Complex.abs z = 1 / 2 := by sorry

end magnitude_of_complex_fraction_l821_82177


namespace x_squared_minus_y_squared_l821_82191

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 5/8)
  (h2 : 3*x - 3*y = 3/8) : 
  x^2 - y^2 = 5/64 := by
  sorry

end x_squared_minus_y_squared_l821_82191


namespace unanswered_test_completion_ways_l821_82165

/-- Represents a multiple-choice test -/
structure MultipleChoiceTest where
  num_questions : ℕ
  choices_per_question : ℕ

/-- Calculates the number of ways to complete the test with all questions unanswered -/
def ways_to_leave_unanswered (test : MultipleChoiceTest) : ℕ := 1

/-- Theorem: For a test with 4 questions and 5 choices per question,
    there is only one way to leave all questions unanswered -/
theorem unanswered_test_completion_ways
  (test : MultipleChoiceTest)
  (h1 : test.num_questions = 4)
  (h2 : test.choices_per_question = 5) :
  ways_to_leave_unanswered test = 1 := by
  sorry


end unanswered_test_completion_ways_l821_82165


namespace gift_card_spending_ratio_l821_82194

theorem gift_card_spending_ratio :
  ∀ (M : ℚ),
  (200 - M - (1/4) * (200 - M) = 75) →
  (M / 200 = 1/2) := by
sorry

end gift_card_spending_ratio_l821_82194


namespace amusement_park_price_calculation_l821_82116

/-- Calculates the total price for a family visiting an amusement park on a weekend -/
def amusement_park_price (adult_price : ℝ) (child_price : ℝ) (adult_count : ℕ) (child_count : ℕ) (adult_discount : ℝ) (sales_tax : ℝ) : ℝ :=
  let total_before_discount := adult_price * adult_count + child_price * child_count
  let adult_discount_amount := adult_price * adult_count * adult_discount
  let total_after_discount := total_before_discount - adult_discount_amount
  let tax_amount := total_after_discount * sales_tax
  total_after_discount + tax_amount

/-- Theorem: The total price for a family of 2 adults and 2 children visiting the amusement park on a weekend is $66 -/
theorem amusement_park_price_calculation :
  amusement_park_price 25 10 2 2 0.2 0.1 = 66 := by
  sorry

end amusement_park_price_calculation_l821_82116


namespace complex_equation_solution_l821_82171

theorem complex_equation_solution (a : ℝ) : 
  (Complex.I * (a + Complex.I) = -1 - 2 * Complex.I) → a = -2 := by
  sorry

end complex_equation_solution_l821_82171


namespace cone_sphere_ratio_l821_82158

theorem cone_sphere_ratio (r h : ℝ) (hr : r > 0) :
  (1 / 3 * π * r^2 * h = 1 / 2 * (4 / 3 * π * r^3)) →
  h / r = 2 := by
  sorry

end cone_sphere_ratio_l821_82158


namespace valid_placements_count_l821_82187

/-- Represents a valid placement of letters in the grid -/
def ValidPlacement := Fin 16 → Fin 2

/-- The total number of cells in the grid -/
def gridSize : Nat := 16

/-- The number of rows (or columns) in the grid -/
def gridDimension : Nat := 4

/-- The number of each letter to be placed -/
def letterCount : Nat := 2

/-- Checks if a placement is valid (no same letter in any row or column) -/
def isValidPlacement (p : ValidPlacement) : Prop := sorry

/-- Counts the number of valid placements -/
def countValidPlacements : Nat := sorry

/-- The main theorem stating the correct number of valid placements -/
theorem valid_placements_count : countValidPlacements = 3960 := by sorry

end valid_placements_count_l821_82187


namespace a_fraction_is_one_third_l821_82102

-- Define the partnership
structure Partnership :=
  (total_capital : ℝ)
  (a_fraction : ℝ)
  (b_fraction : ℝ)
  (c_fraction : ℝ)
  (d_fraction : ℝ)
  (total_profit : ℝ)
  (a_profit : ℝ)

-- Define the conditions
def partnership_conditions (p : Partnership) : Prop :=
  p.b_fraction = 1/4 ∧
  p.c_fraction = 1/5 ∧
  p.d_fraction = 1 - (p.a_fraction + p.b_fraction + p.c_fraction) ∧
  p.total_profit = 2445 ∧
  p.a_profit = 815 ∧
  p.a_profit / p.total_profit = p.a_fraction

-- Theorem statement
theorem a_fraction_is_one_third (p : Partnership) :
  partnership_conditions p → p.a_fraction = 1/3 := by
  sorry

end a_fraction_is_one_third_l821_82102


namespace smallest_integer_a_for_unique_solution_l821_82184

-- Define the system of equations
def equation1 (x y a : ℝ) : Prop := y / (a - Real.sqrt x - 1) = 4
def equation2 (x y : ℝ) : Prop := y = (Real.sqrt x + 5) / (Real.sqrt x + 1)

-- Define the property of having a unique solution
def has_unique_solution (a : ℝ) : Prop :=
  ∃! (x y : ℝ), equation1 x y a ∧ equation2 x y

-- State the theorem
theorem smallest_integer_a_for_unique_solution :
  (∀ a : ℤ, a < 3 → ¬(has_unique_solution (a : ℝ))) ∧
  has_unique_solution 3 :=
sorry

end smallest_integer_a_for_unique_solution_l821_82184


namespace ceiling_floor_sum_l821_82109

theorem ceiling_floor_sum : ⌈(7 : ℝ) / 3⌉ + ⌊-(7 : ℝ) / 3⌋ = 0 := by
  sorry

end ceiling_floor_sum_l821_82109


namespace dart_board_probability_l821_82139

/-- The probability of a dart landing within the center square of a regular hexagonal dart board -/
theorem dart_board_probability (s : ℝ) (h : s > 0) : 
  let hexagon_area := (3 * Real.sqrt 3 / 2) * s^2
  let square_area := s^2
  square_area / hexagon_area = 2 * Real.sqrt 3 / 9 := by sorry

end dart_board_probability_l821_82139


namespace concentrate_water_ratio_is_one_to_three_l821_82180

/-- The ratio of concentrate to water for orange juice -/
def concentrate_to_water_ratio : ℚ := 1 / 3

/-- The number of cans of concentrate used -/
def concentrate_cans : ℕ := 40

/-- The number of cans of water per can of concentrate -/
def water_cans_per_concentrate : ℕ := 3

/-- Theorem: The ratio of cans of concentrate to cans of water is 1:3 -/
theorem concentrate_water_ratio_is_one_to_three :
  concentrate_to_water_ratio = 1 / (water_cans_per_concentrate : ℚ) ∧
  concentrate_to_water_ratio = (concentrate_cans : ℚ) / ((water_cans_per_concentrate * concentrate_cans) : ℚ) := by
  sorry

end concentrate_water_ratio_is_one_to_three_l821_82180


namespace marble_probability_l821_82120

theorem marble_probability (total : ℕ) (p_white p_green p_black : ℚ) :
  total = 120 ∧
  p_white = 1/4 ∧
  p_green = 1/6 ∧
  p_black = 1/8 →
  1 - (p_white + p_green + p_black) = 11/24 :=
by sorry

end marble_probability_l821_82120


namespace vacation_costs_l821_82179

/-- Vacation expenses for Xiao Ming's family --/
structure VacationExpenses where
  tickets : ℕ
  meals : ℕ
  accommodation : ℕ

/-- The actual expenses for Xiao Ming's family's vacation --/
def actualExpenses : VacationExpenses :=
  { tickets := 456
  , meals := 385
  , accommodation := 396 }

/-- The total cost of the vacation --/
def totalCost (e : VacationExpenses) : ℕ :=
  e.tickets + e.meals + e.accommodation

/-- The approximate total cost of the vacation --/
def approximateTotalCost (e : VacationExpenses) : ℕ :=
  ((e.tickets + 50) / 100 * 100) +
  ((e.meals + 50) / 100 * 100) +
  ((e.accommodation + 50) / 100 * 100)

theorem vacation_costs (e : VacationExpenses) 
  (h : e = actualExpenses) : 
  approximateTotalCost e = 1300 ∧ totalCost e = 1237 := by
  sorry

end vacation_costs_l821_82179


namespace type_a_sample_size_l821_82142

/-- Represents the ratio of quantities for product types A, B, and C -/
structure ProductRatio :=
  (a : ℕ) (b : ℕ) (c : ℕ)

/-- Calculates the number of units to be selected for a given product type -/
def unitsToSelect (total : ℕ) (sampleSize : ℕ) (ratio : ProductRatio) (typeRatio : ℕ) : ℕ :=
  (sampleSize * typeRatio) / (ratio.a + ratio.b + ratio.c)

theorem type_a_sample_size 
  (totalProduction : ℕ)
  (sampleSize : ℕ)
  (ratio : ProductRatio)
  (h1 : totalProduction = 600)
  (h2 : sampleSize = 120)
  (h3 : ratio = ⟨1, 2, 3⟩) :
  unitsToSelect totalProduction sampleSize ratio ratio.a = 20 := by
  sorry

end type_a_sample_size_l821_82142


namespace walnut_price_l821_82127

/-- Represents the price of a nut in Forints -/
structure NutPrice where
  price : ℕ
  is_two_digit : price ≥ 10 ∧ price < 100

/-- Checks if two digits are consecutive -/
def consecutive_digits (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ (a = b + 1 ∨ b = a + 1) ∧ n = 10 * a + b

/-- Checks if two prices are digit swaps of each other -/
def is_digit_swap (p1 p2 : NutPrice) : Prop :=
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ p1.price = 10 * a + b ∧ p2.price = 10 * b + a

theorem walnut_price (walnut hazelnut : NutPrice)
  (total_value : ℕ) (total_weight : ℕ)
  (h1 : total_value = 1978)
  (h2 : total_weight = 55)
  (h3 : walnut.price > hazelnut.price)
  (h4 : is_digit_swap walnut hazelnut)
  (h5 : consecutive_digits walnut.price)
  (h6 : ∃ (w h : ℕ), w * walnut.price + h * hazelnut.price = total_value ∧ w + h = total_weight) :
  walnut.price = 43 := by
  sorry

end walnut_price_l821_82127


namespace davids_weighted_average_l821_82159

def english_mark : ℝ := 76
def math_mark : ℝ := 65
def physics_mark : ℝ := 82
def chemistry_mark : ℝ := 67
def biology_mark : ℝ := 85
def history_mark : ℝ := 78
def cs_mark : ℝ := 81

def english_weight : ℝ := 0.10
def math_weight : ℝ := 0.20
def physics_weight : ℝ := 0.15
def chemistry_weight : ℝ := 0.15
def biology_weight : ℝ := 0.10
def history_weight : ℝ := 0.20
def cs_weight : ℝ := 0.10

def weighted_average : ℝ := 
  english_mark * english_weight +
  math_mark * math_weight +
  physics_mark * physics_weight +
  chemistry_mark * chemistry_weight +
  biology_mark * biology_weight +
  history_mark * history_weight +
  cs_mark * cs_weight

theorem davids_weighted_average : weighted_average = 75.15 := by
  sorry

end davids_weighted_average_l821_82159


namespace mismatching_socks_count_l821_82172

def total_socks : ℕ := 65
def ankle_sock_pairs : ℕ := 13
def crew_sock_pairs : ℕ := 10

theorem mismatching_socks_count :
  total_socks - 2 * (ankle_sock_pairs + crew_sock_pairs) = 19 :=
by sorry

end mismatching_socks_count_l821_82172


namespace somin_solved_most_l821_82119

def suhyeon_remaining : ℚ := 1/4
def somin_remaining : ℚ := 1/8
def jisoo_remaining : ℚ := 1/5

theorem somin_solved_most : 
  somin_remaining < suhyeon_remaining ∧ somin_remaining < jisoo_remaining :=
sorry

end somin_solved_most_l821_82119


namespace equation_solution_l821_82122

theorem equation_solution : ∃! x : ℝ, x^2 - ⌊x⌋ = 3 ∧ x = (4 : ℝ)^(1/3) := by
  sorry

end equation_solution_l821_82122


namespace min_value_of_sum_l821_82185

theorem min_value_of_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y + x*y = 2) :
  x + y ≥ 2 * Real.sqrt 3 - 2 :=
sorry

end min_value_of_sum_l821_82185


namespace fourth_root_equation_l821_82124

theorem fourth_root_equation (P : ℝ) : (P^3)^(1/4) = 81 * 81^(1/16) → P = 27 * 3^(1/3) := by
  sorry

end fourth_root_equation_l821_82124


namespace division_sum_theorem_l821_82144

theorem division_sum_theorem (quotient divisor remainder : ℕ) 
  (h_quotient : quotient = 40)
  (h_divisor : divisor = 72)
  (h_remainder : remainder = 64) :
  divisor * quotient + remainder = 2944 :=
by sorry

end division_sum_theorem_l821_82144


namespace total_disks_is_126_l821_82169

/-- Represents the colors of disks in the bag -/
inductive DiskColor
  | Blue
  | Yellow
  | Green

/-- Represents the bag of disks -/
structure DiskBag where
  blue : ℕ
  yellow : ℕ
  green : ℕ

/-- The ratio of blue to yellow to green disks -/
def diskRatio : DiskBag → Prop
  | ⟨b, y, g⟩ => ∃ (x : ℕ), b = 3 * x ∧ y = 7 * x ∧ g = 8 * x

/-- The difference between green and blue disks -/
def greenBlueDifference (bag : DiskBag) : Prop :=
  bag.green = bag.blue + 35

/-- The total number of disks in the bag -/
def totalDisks (bag : DiskBag) : ℕ :=
  bag.blue + bag.yellow + bag.green

/-- Theorem: Given the conditions, the total number of disks is 126 -/
theorem total_disks_is_126 (bag : DiskBag) 
  (h1 : diskRatio bag) 
  (h2 : greenBlueDifference bag) : 
  totalDisks bag = 126 := by
  sorry

end total_disks_is_126_l821_82169


namespace leo_and_kendra_combined_weight_l821_82153

/-- The combined weight of Leo and Kendra is 150 pounds -/
theorem leo_and_kendra_combined_weight :
  let leo_weight : ℝ := 86
  let kendra_weight : ℝ := (leo_weight + 10) / 1.5
  leo_weight + kendra_weight = 150 :=
by sorry

end leo_and_kendra_combined_weight_l821_82153


namespace jerry_lawsuit_percentage_l821_82143

/-- Calculates the percentage of a lawsuit claim received -/
def lawsuit_claim_percentage (annual_salary : ℕ) (years : ℕ) (medical_bills : ℕ) (received_amount : ℕ) : ℚ :=
  let salary_damages := annual_salary * years
  let punitive_multiplier := 3
  let punitive_damages := punitive_multiplier * (salary_damages + medical_bills)
  let total_claim := salary_damages + medical_bills + punitive_damages
  (received_amount : ℚ) / (total_claim : ℚ) * 100

theorem jerry_lawsuit_percentage :
  let result := lawsuit_claim_percentage 50000 30 200000 5440000
  (result > 79.9 ∧ result < 80.1) :=
by sorry

end jerry_lawsuit_percentage_l821_82143


namespace trapezoid_to_square_l821_82198

/-- A trapezoid composed of a square and a triangle -/
structure Trapezoid where
  square_area : ℝ
  triangle_area : ℝ

/-- The given trapezoid -/
def given_trapezoid : Trapezoid where
  square_area := 4
  triangle_area := 1

/-- The total area of the trapezoid -/
def trapezoid_area (t : Trapezoid) : ℝ := t.square_area + t.triangle_area

/-- A function to check if a trapezoid can be rearranged into a square -/
def can_form_square (t : Trapezoid) : Prop :=
  ∃ (side : ℝ), side^2 = trapezoid_area t ∧
  ∃ (a b : ℝ), a^2 + b^2 = side^2 ∧ a * b = t.triangle_area

/-- Theorem: The given trapezoid can be cut and rearranged to form a square -/
theorem trapezoid_to_square :
  can_form_square given_trapezoid := by
  sorry

end trapezoid_to_square_l821_82198


namespace prob_not_all_same_eight_sided_dice_l821_82167

theorem prob_not_all_same_eight_sided_dice (n : ℕ) (h : n = 5) :
  (1 - (8 : ℚ) / 8^n) = 4095 / 4096 := by
  sorry

end prob_not_all_same_eight_sided_dice_l821_82167


namespace prob_two_twos_given_sum7_is_one_fourth_l821_82173

/-- Represents the outcome of three draws from an urn containing balls numbered 1 to 4 -/
structure ThreeDraws where
  first : Fin 4
  second : Fin 4
  third : Fin 4

/-- The set of all possible outcomes of three draws -/
def allOutcomes : Finset ThreeDraws := sorry

/-- The probability of each individual outcome, assuming uniform distribution -/
def probOfOutcome (outcome : ThreeDraws) : ℚ := 1 / 64

/-- The sum of the numbers drawn in a given outcome -/
def sumOfDraws (outcome : ThreeDraws) : Nat :=
  outcome.first.val + 1 + outcome.second.val + 1 + outcome.third.val + 1

/-- The set of outcomes where the sum of draws is 7 -/
def outcomesWithSum7 : Finset ThreeDraws :=
  allOutcomes.filter (λ o => sumOfDraws o = 7)

/-- The number of times 2 is drawn in a given outcome -/
def countTwos (outcome : ThreeDraws) : Nat :=
  (if outcome.first = 1 then 1 else 0) +
  (if outcome.second = 1 then 1 else 0) +
  (if outcome.third = 1 then 1 else 0)

/-- The set of outcomes where 2 is drawn at least twice and the sum is 7 -/
def outcomesWithTwoTwosAndSum7 : Finset ThreeDraws :=
  outcomesWithSum7.filter (λ o => countTwos o ≥ 2)

/-- The probability of drawing 2 at least twice given that the sum is 7 -/
def probTwoTwosGivenSum7 : ℚ :=
  (outcomesWithTwoTwosAndSum7.sum probOfOutcome) /
  (outcomesWithSum7.sum probOfOutcome)

theorem prob_two_twos_given_sum7_is_one_fourth :
  probTwoTwosGivenSum7 = 1 / 4 := by sorry

end prob_two_twos_given_sum7_is_one_fourth_l821_82173


namespace range_of_a_l821_82101

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ 2}
def B (a : ℝ) : Set ℝ := {x | x ≥ a}

-- State the theorem
theorem range_of_a (a : ℝ) (h : A ⊆ B a) : a ≤ -2 := by
  sorry

end range_of_a_l821_82101


namespace outfit_count_l821_82148

/-- The number of shirts -/
def num_shirts : ℕ := 8

/-- The number of ties that can be paired with each shirt -/
def ties_per_shirt : ℕ := 4

/-- The total number of shirt-and-tie outfits -/
def total_outfits : ℕ := num_shirts * ties_per_shirt

theorem outfit_count : total_outfits = 32 := by
  sorry

end outfit_count_l821_82148


namespace dolphin_training_ratio_l821_82140

theorem dolphin_training_ratio (total : ℕ) (fully_trained_fraction : ℚ) (to_be_trained : ℕ) :
  total = 20 →
  fully_trained_fraction = 1/4 →
  to_be_trained = 5 →
  (total - (total * fully_trained_fraction).num - to_be_trained : ℚ) / to_be_trained = 2 := by
sorry

end dolphin_training_ratio_l821_82140


namespace distance_after_time_l821_82104

theorem distance_after_time (adam_speed simon_speed : ℝ) (time : ℝ) (distance : ℝ) : 
  adam_speed = 5 →
  simon_speed = 12 →
  time = 5 →
  distance = 65 →
  (adam_speed * time) ^ 2 + (simon_speed * time) ^ 2 = distance ^ 2 := by
sorry

end distance_after_time_l821_82104


namespace line_inclination_angle_l821_82176

/-- Given a line l with equation represented by the determinant |1 0 2; x 2 3; y -1 2| = 0,
    prove that its inclination angle is π - arctan(1/2) -/
theorem line_inclination_angle (x y : ℝ) : 
  let l : Set (ℝ × ℝ) := {(x, y) | Matrix.det !![1, 0, 2; x, 2, 3; y, -1, 2] = 0}
  ∃ θ : ℝ, θ = π - Real.arctan (1/2) ∧ 
    ∀ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ l → (x₂, y₂) ∈ l → 
      x₁ ≠ x₂ → θ = Real.arctan ((y₂ - y₁) / (x₂ - x₁)) := by
  sorry

end line_inclination_angle_l821_82176


namespace f_decreasing_implies_a_range_l821_82111

/-- A piecewise function f defined on ℝ -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then (a - 5) * x - 1 else (x + a) / (x - 1)

/-- Theorem stating that if f is decreasing on ℝ, then a ∈ (-1, 1] -/
theorem f_decreasing_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) →
  a ∈ Set.Ioc (-1) 1 :=
by sorry

end f_decreasing_implies_a_range_l821_82111


namespace yoonseok_handshakes_l821_82149

/-- Represents a group of people arranged in a dodecagon -/
structure DodecagonGroup :=
  (size : Nat)
  (handshake_rule : Nat → Nat)

/-- The number of handshakes for a person in the DodecagonGroup -/
def handshakes_count (g : DodecagonGroup) : Nat :=
  g.handshake_rule g.size

theorem yoonseok_handshakes (g : DodecagonGroup) :
  g.size = 12 →
  (∀ n : Nat, n ≤ g.size → g.handshake_rule n = n - 3) →
  handshakes_count g = 9 := by
  sorry

#check yoonseok_handshakes

end yoonseok_handshakes_l821_82149


namespace common_chord_length_l821_82110

/-- The length of the common chord of two overlapping circles -/
theorem common_chord_length (r : ℝ) (h : r = 15) : 
  let chord_length := 2 * r * Real.sqrt 3 / 2
  chord_length = 15 * Real.sqrt 3 := by sorry

end common_chord_length_l821_82110


namespace magical_red_knights_fraction_l821_82192

/-- Represents the fraction of knights of a certain color who are magical -/
structure MagicalFraction where
  numerator : ℕ
  denominator : ℕ
  denominator_pos : denominator > 0

/-- Represents the distribution of knights in the kingdom -/
structure KnightDistribution where
  total : ℕ
  red : ℕ
  blue : ℕ
  magical : ℕ
  red_fraction : red = (3 * total) / 8
  blue_fraction : blue = total - red
  magical_fraction : magical = total / 4

/-- The main theorem about magical knights -/
theorem magical_red_knights_fraction 
  (dist : KnightDistribution) 
  (red_magical : MagicalFraction) 
  (blue_magical : MagicalFraction) :
  (3 * dist.total) / 8 * red_magical.numerator / red_magical.denominator + 
  (5 * dist.total) / 8 * blue_magical.numerator / blue_magical.denominator = dist.total / 4 →
  red_magical.numerator * blue_magical.denominator = 
  3 * blue_magical.numerator * red_magical.denominator →
  red_magical.numerator = 6 ∧ red_magical.denominator = 19 :=
sorry

end magical_red_knights_fraction_l821_82192


namespace comparison_of_roots_l821_82128

theorem comparison_of_roots : 
  let a := (16 : ℝ) ^ (1/4)
  let b := (27 : ℝ) ^ (1/3)
  let c := (25 : ℝ) ^ (1/2)
  let d := (32 : ℝ) ^ (1/5)
  (c > b ∧ b > a ∧ b > d) := by sorry

end comparison_of_roots_l821_82128


namespace arithmetic_sequence_equal_sums_l821_82117

/-- Sum of first n terms of an arithmetic sequence -/
def arithmetic_sum (a : ℚ) (d : ℚ) (n : ℤ) : ℚ :=
  n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequence_equal_sums :
  ∀ n : ℤ, n ≠ 0 →
    arithmetic_sum 5 6 n = arithmetic_sum 3 5 n ↔ n = -3 :=
by sorry

end arithmetic_sequence_equal_sums_l821_82117


namespace items_not_washed_l821_82154

theorem items_not_washed (total_items : ℕ) (items_washed : ℕ) : 
  total_items = 129 → items_washed = 20 → total_items - items_washed = 109 := by
  sorry

end items_not_washed_l821_82154


namespace min_steps_to_eliminate_zeroes_l821_82125

/-- Represents the state of the blackboard with zeroes and ones -/
structure BlackboardState where
  zeroes : Nat
  ones : Nat

/-- Defines a step operation on the blackboard state -/
def step (state : BlackboardState) : BlackboardState :=
  { zeroes := state.zeroes - 1, ones := state.ones + 1 }

/-- The initial state of the blackboard -/
def initial_state : BlackboardState := { zeroes := 150, ones := 151 }

/-- Predicate to check if a state has no zeroes -/
def no_zeroes (state : BlackboardState) : Prop := state.zeroes = 0

/-- The theorem to be proved -/
theorem min_steps_to_eliminate_zeroes :
  ∃ (n : Nat), n = 150 ∧ no_zeroes (n.iterate step initial_state) ∧
  ∀ (m : Nat), m < n → ¬no_zeroes (m.iterate step initial_state) :=
sorry

end min_steps_to_eliminate_zeroes_l821_82125


namespace triangle_ratio_sum_l821_82181

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_ratio_sum (t : Triangle) (h : t.B = 60) :
  (t.c / (t.a + t.b)) + (t.a / (t.b + t.c)) = 1 := by
  sorry

end triangle_ratio_sum_l821_82181


namespace odd_solutions_count_l821_82108

theorem odd_solutions_count (x : ℕ) : 
  (∃ (s : Finset ℕ), 
    (∀ y ∈ s, 20 ≤ y ∧ y ≤ 150 ∧ Odd y ∧ (y + 17) % 29 = 65 % 29) ∧ 
    (∀ y, 20 ≤ y ∧ y ≤ 150 ∧ Odd y ∧ (y + 17) % 29 = 65 % 29 → y ∈ s) ∧
    Finset.card s = 3) := by
  sorry

end odd_solutions_count_l821_82108


namespace proposition_l821_82105

theorem proposition (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) :
  (Real.sqrt (b^2 - a*c)) / a < Real.sqrt 3 := by
  sorry

end proposition_l821_82105


namespace even_function_domain_symmetric_l821_82197

/-- An even function from ℝ to ℝ -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- The domain of the function -/
def Domain (f : ℝ → ℝ) (t : ℝ) : Set ℝ :=
  {x | t - 4 ≤ x ∧ x ≤ t}

/-- Theorem: For an even function with domain [t-4, t], t = 2 -/
theorem even_function_domain_symmetric (f : ℝ → ℝ) (t : ℝ) 
    (h1 : EvenFunction f) (h2 : Domain f t = Set.Icc (t - 4) t) : t = 2 := by
  sorry

end even_function_domain_symmetric_l821_82197


namespace polynomial_divisibility_l821_82182

theorem polynomial_divisibility : 
  ∃ (q : ℝ → ℝ), ∀ x : ℝ, 5 * x^2 - 6 * x - 95 = (x - 5) * q x := by
  sorry

end polynomial_divisibility_l821_82182


namespace average_first_50_naturals_l821_82183

theorem average_first_50_naturals : 
  let n : ℕ := 50
  let sum : ℕ := n * (n + 1) / 2
  (sum : ℚ) / n = 25.5 := by sorry

end average_first_50_naturals_l821_82183


namespace sqrt_pattern_l821_82155

theorem sqrt_pattern (a b : ℝ) : 
  (∀ n : ℕ, n ≥ 2 → n ≤ 4 → Real.sqrt (n + n / (n^2 - 1)) = n * Real.sqrt (n / (n^2 - 1))) →
  Real.sqrt (6 + a / b) = 6 * Real.sqrt (a / b) →
  a = 6 ∧ b = 35 := by
  sorry

end sqrt_pattern_l821_82155


namespace tuition_cost_l821_82163

/-- Proves that the tuition cost per semester is $22,000 given the specified conditions --/
theorem tuition_cost (T : ℝ) : 
  (T / 2)                     -- Parents' contribution
  + 3000                      -- Scholarship
  + (2 * 3000)                -- Student loan (twice scholarship amount)
  + (200 * 10)                -- Work earnings (200 hours at $10/hour)
  = T                         -- Total equals tuition cost
  → T = 22000 := by
  sorry

end tuition_cost_l821_82163


namespace total_revenue_is_21040_l821_82157

/-- Represents the seating capacity and ticket price for a section of the circus tent. -/
structure Section where
  capacity : ℕ
  price : ℕ

/-- Calculates the revenue for a given section when all seats are occupied. -/
def sectionRevenue (s : Section) : ℕ := s.capacity * s.price

/-- Represents the seating arrangement of the circus tent. -/
def circusTent : List Section :=
  [{ capacity := 246, price := 15 },
   { capacity := 246, price := 15 },
   { capacity := 246, price := 15 },
   { capacity := 246, price := 15 },
   { capacity := 314, price := 20 }]

/-- Calculates the total revenue when all seats are occupied. -/
def totalRevenue : ℕ := (circusTent.map sectionRevenue).sum

/-- Theorem stating that the total revenue when all seats are occupied is $21,040. -/
theorem total_revenue_is_21040 : totalRevenue = 21040 := by
  sorry

end total_revenue_is_21040_l821_82157


namespace xyz_fraction_l821_82152

theorem xyz_fraction (x y z : ℝ) 
  (h1 : x * y / (x + y) = 1 / 3)
  (h2 : y * z / (y + z) = 1 / 5)
  (h3 : z * x / (z + x) = 1 / 6) :
  x * y * z / (x * y + y * z + z * x) = 1 / 7 := by
sorry

end xyz_fraction_l821_82152


namespace product_inequality_l821_82190

theorem product_inequality (a b c : ℝ) (d : ℝ) 
  (sum_zero : a + b + c = 0)
  (d_def : d = max (|a|) (max (|b|) (|c|))) :
  |(1 + a) * (1 + b) * (1 + c)| ≥ 1 - d^2 := by
  sorry

end product_inequality_l821_82190


namespace alpha_plus_beta_l821_82134

theorem alpha_plus_beta (α β : ℝ) : 
  (∀ x : ℝ, (x - α) / (x + β) = (x^2 - 64*x + 975) / (x^2 + 99*x - 2200)) → 
  α + β = 138 := by
  sorry

end alpha_plus_beta_l821_82134


namespace assembly_line_arrangements_eq_120_l821_82162

/-- Represents the number of tasks in the assembly line -/
def num_tasks : ℕ := 6

/-- Represents the number of independent arrangements -/
def num_independent_arrangements : ℕ := 5

/-- The number of ways to arrange the assembly line -/
def assembly_line_arrangements : ℕ := Nat.factorial num_independent_arrangements

/-- Theorem stating that the number of ways to arrange the assembly line is 120 -/
theorem assembly_line_arrangements_eq_120 : 
  assembly_line_arrangements = 120 := by sorry

end assembly_line_arrangements_eq_120_l821_82162


namespace parallel_line_vector_l821_82133

theorem parallel_line_vector (m : ℝ) : 
  (∀ x y : ℝ, m * x + 2 * y + 6 = 0 → (1 - m) * y = x) → 
  m = -1 ∨ m = 2 := by
  sorry

end parallel_line_vector_l821_82133


namespace min_value_reciprocal_sum_l821_82118

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a + 2 / b) ≥ 3 + 2 * Real.sqrt 2 := by sorry

end min_value_reciprocal_sum_l821_82118


namespace hyperbola_focus_distance_l821_82130

/-- Represents a point on a hyperbola -/
structure HyperbolaPoint where
  x : ℝ
  y : ℝ
  on_hyperbola : x^2 / 4 - y^2 = 1

/-- Theorem: For a hyperbola defined by x^2/4 - y^2 = 1, if the distance from a point
    on the hyperbola to one focus is 12, then the distance to the other focus is either 16 or 8 -/
theorem hyperbola_focus_distance (p : HyperbolaPoint) (d1 : ℝ) (d2 : ℝ)
  (h1 : d1 = 12) -- Distance to one focus is 12
  (h2 : d2 = |d1 - 4| ∨ d2 = |d1 + 4|) -- Distance to other focus based on hyperbola properties
  : d2 = 16 ∨ d2 = 8 := by
  sorry


end hyperbola_focus_distance_l821_82130


namespace fractional_equation_solution_l821_82106

theorem fractional_equation_solution :
  ∃ x : ℝ, (x + 2) / (x - 1) = 0 ∧ x = -2 :=
by sorry

end fractional_equation_solution_l821_82106


namespace largest_angle_convex_hexagon_l821_82175

/-- The measure of the largest angle in a convex hexagon with specific interior angles -/
theorem largest_angle_convex_hexagon :
  ∀ x : ℝ,
  (x + 2) + (2 * x + 4) + (3 * x - 6) + (4 * x + 8) + (5 * x - 10) + (6 * x + 12) = 720 →
  max (x + 2) (max (2 * x + 4) (max (3 * x - 6) (max (4 * x + 8) (max (5 * x - 10) (6 * x + 12))))) = 215 := by
sorry

end largest_angle_convex_hexagon_l821_82175


namespace sum_of_fifth_powers_l821_82135

theorem sum_of_fifth_powers (ζ₁ ζ₂ ζ₃ : ℂ) 
  (h1 : ζ₁ + ζ₂ + ζ₃ = 2)
  (h2 : ζ₁^2 + ζ₂^2 + ζ₃^2 = 6)
  (h3 : ζ₁^3 + ζ₂^3 + ζ₃^3 = 8) :
  ζ₁^5 + ζ₂^5 + ζ₃^5 = 30 := by
  sorry

end sum_of_fifth_powers_l821_82135


namespace acid_solution_replacement_l821_82189

theorem acid_solution_replacement (V : ℝ) (h : V > 0) :
  let x : ℝ := 0.5
  let initial_concentration : ℝ := 0.5
  let replacement_concentration : ℝ := 0.3
  let final_concentration : ℝ := 0.4
  initial_concentration * V - initial_concentration * x * V + replacement_concentration * x * V = final_concentration * V :=
by sorry

end acid_solution_replacement_l821_82189


namespace problem_solution_l821_82193

theorem problem_solution (m n : ℚ) (h : m - n = -2/3) : 7 - 3*m + 3*n = 9 := by
  sorry

end problem_solution_l821_82193


namespace complex_fraction_squared_complex_fraction_minus_z_l821_82126

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Theorem 1
theorem complex_fraction_squared :
  ((3 - i) / (1 + i))^2 = -3 - 4*i := by sorry

-- Theorem 2
theorem complex_fraction_minus_z (z : ℂ) (h : z = 1 + i) :
  2 / z - z = -2*i := by sorry

end complex_fraction_squared_complex_fraction_minus_z_l821_82126


namespace triangle_angle_problem_l821_82131

theorem triangle_angle_problem (A B C : ℝ) : 
  A = B + 21 →
  C = B + 36 →
  A + B + C = 180 →
  B = 41 :=
by sorry

end triangle_angle_problem_l821_82131


namespace mean_height_of_volleyball_team_l821_82112

def volleyball_heights : List ℕ := [58, 59, 60, 61, 62, 65, 65, 66, 67, 70, 71, 71, 72, 74, 75, 79, 79]

theorem mean_height_of_volleyball_team (heights : List ℕ) (h1 : heights = volleyball_heights) :
  (heights.sum / heights.length : ℚ) = 68 := by
  sorry

end mean_height_of_volleyball_team_l821_82112


namespace probability_greater_than_400_probability_even_l821_82168

def digits : List ℕ := [1, 5, 6]

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧ (∀ d, d ∈ digits → (n / 100 = d ∨ (n / 10) % 10 = d ∨ n % 10 = d))

def valid_numbers : List ℕ := [156, 165, 516, 561, 615, 651]

theorem probability_greater_than_400 :
  (valid_numbers.filter (λ n => n > 400)).length / valid_numbers.length = 2 / 3 := by sorry

theorem probability_even :
  (valid_numbers.filter (λ n => n % 2 = 0)).length / valid_numbers.length = 1 / 3 := by sorry

end probability_greater_than_400_probability_even_l821_82168


namespace equation_satisfied_l821_82100

theorem equation_satisfied (x y z : ℤ) (h1 : x = z + 1) (h2 : y = z) :
  x * (x - y) + y * (y - z) + z * (z - x) = 2 := by
  sorry

end equation_satisfied_l821_82100


namespace number_of_girls_l821_82113

-- Define the total number of polished nails
def total_nails : ℕ := 40

-- Define the number of nails per girl
def nails_per_girl : ℕ := 20

-- Theorem to prove the number of girls
theorem number_of_girls : total_nails / nails_per_girl = 2 := by
  sorry

end number_of_girls_l821_82113


namespace fixed_point_exponential_function_l821_82164

/-- Given a function f(x) = 2a^(x-b) + 1 where a > 0 and a ≠ 1, 
    if f(2) = 3, then b = 2 -/
theorem fixed_point_exponential_function (a b : ℝ) (ha : a > 0) (ha' : a ≠ 1) :
  (∀ x, 2 * a^(x - b) + 1 = 3) → b = 2 := by
  sorry

end fixed_point_exponential_function_l821_82164


namespace second_fraction_base_l821_82178

theorem second_fraction_base (x k : ℝ) (h1 : (1/2)^18 * (1/x)^k = 1/18^18) (h2 : k = 9) : x = 9 := by
  sorry

end second_fraction_base_l821_82178


namespace equation_system_solution_l821_82136

theorem equation_system_solution (x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ : ℝ) 
  (eq1 : x₁ + 4*x₂ + 9*x₃ + 16*x₄ + 25*x₅ + 36*x₆ + 49*x₇ + 64*x₈ = 2)
  (eq2 : 4*x₁ + 9*x₂ + 16*x₃ + 25*x₄ + 36*x₅ + 49*x₆ + 64*x₇ + 81*x₈ = 15)
  (eq3 : 9*x₁ + 16*x₂ + 25*x₃ + 36*x₄ + 49*x₅ + 64*x₆ + 81*x₇ + 100*x₈ = 136)
  (eq4 : 16*x₁ + 25*x₂ + 36*x₃ + 49*x₄ + 64*x₅ + 81*x₆ + 100*x₇ + 121*x₈ = 1234) :
  25*x₁ + 36*x₂ + 49*x₃ + 64*x₄ + 81*x₅ + 100*x₆ + 121*x₇ + 144*x₈ = 1242 :=
by sorry


end equation_system_solution_l821_82136
