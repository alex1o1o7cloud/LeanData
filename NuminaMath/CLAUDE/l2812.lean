import Mathlib

namespace arithmetic_progression_common_difference_l2812_281271

theorem arithmetic_progression_common_difference 
  (n : ℕ) 
  (d : ℚ) 
  (sum_original : ℚ) 
  (sum_decrease_min : ℚ) 
  (sum_decrease_max : ℚ) :
  (n > 0) →
  (sum_original = 63) →
  (sum_original = (n / 2) * (3 * d + (n - 1) * d)) →
  (sum_decrease_min = 7) →
  (sum_decrease_max = 8) →
  (sum_original - (n / 2) * (2 * d + (n - 1) * d) ≥ sum_decrease_min) →
  (sum_original - (n / 2) * (2 * d + (n - 1) * d) ≤ sum_decrease_max) →
  (d = 21/8 ∨ d = 2) :=
by sorry

end arithmetic_progression_common_difference_l2812_281271


namespace line_passes_through_fixed_point_l2812_281242

theorem line_passes_through_fixed_point :
  ∀ (a : ℝ), (3 * a - 1 + 1 - 3 * a = 0) := by
  sorry

end line_passes_through_fixed_point_l2812_281242


namespace max_volume_at_eight_l2812_281206

/-- The side length of the original square plate in cm -/
def plate_side : ℝ := 48

/-- The volume of the container as a function of the cut square's side length -/
def volume (x : ℝ) : ℝ := (plate_side - 2*x)^2 * x

/-- The derivative of the volume function -/
def volume_derivative (x : ℝ) : ℝ := (plate_side - 2*x) * (plate_side - 6*x)

theorem max_volume_at_eight :
  ∃ (max_x : ℝ), max_x = 8 ∧
  ∀ (x : ℝ), 0 < x ∧ x < plate_side / 2 → volume x ≤ volume max_x :=
sorry

end max_volume_at_eight_l2812_281206


namespace first_player_wins_l2812_281237

def Game := List Nat → List Nat

def validMove (n : Nat) (history : List Nat) : Prop :=
  n ∣ 328 ∧ n ∉ history ∧ ∀ m ∈ history, ¬(n ∣ m)

def gameOver (history : List Nat) : Prop :=
  328 ∈ history

def winningStrategy (strategy : Game) : Prop :=
  ∀ history : List Nat,
    ¬gameOver history →
    ∃ move,
      validMove move history ∧
      ∀ opponent_move,
        validMove opponent_move (move :: history) →
        gameOver (opponent_move :: move :: history)

theorem first_player_wins :
  ∃ strategy : Game, winningStrategy strategy :=
sorry

end first_player_wins_l2812_281237


namespace daxton_water_usage_l2812_281262

theorem daxton_water_usage 
  (tank_capacity : ℝ)
  (initial_fill_ratio : ℝ)
  (refill_ratio : ℝ)
  (final_volume : ℝ)
  (h1 : tank_capacity = 8000)
  (h2 : initial_fill_ratio = 3/4)
  (h3 : refill_ratio = 0.3)
  (h4 : final_volume = 4680) :
  let initial_volume := tank_capacity * initial_fill_ratio
  let usage_percentage := 
    (initial_volume - (final_volume - refill_ratio * (initial_volume - usage_volume))) / initial_volume
  let usage_volume := usage_percentage * initial_volume
  usage_percentage = 0.4 := by
sorry

end daxton_water_usage_l2812_281262


namespace forty_second_card_is_eight_of_spades_l2812_281243

-- Define the card suits
inductive Suit
| Hearts
| Spades

-- Define the card ranks
inductive Rank
| Ace | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King

-- Define a card as a pair of rank and suit
structure Card where
  rank : Rank
  suit : Suit

-- Define the cycle of cards
def cardCycle : List Card := sorry

-- Define a function to get the nth card in the cycle
def nthCard (n : Nat) : Card := sorry

-- Theorem to prove
theorem forty_second_card_is_eight_of_spades :
  nthCard 42 = Card.mk Rank.Eight Suit.Spades := by sorry

end forty_second_card_is_eight_of_spades_l2812_281243


namespace ball_bearing_bulk_discount_percentage_l2812_281247

/-- Calculates the bulk discount percentage for John's ball bearing purchase --/
theorem ball_bearing_bulk_discount_percentage : 
  let num_machines : ℕ := 10
  let bearings_per_machine : ℕ := 30
  let normal_price : ℚ := 1
  let sale_price : ℚ := 3/4
  let total_savings : ℚ := 120
  let total_bearings := num_machines * bearings_per_machine
  let original_cost := total_bearings * normal_price
  let sale_cost := total_bearings * sale_price
  let discounted_cost := original_cost - total_savings
  let bulk_discount := sale_cost - discounted_cost
  let discount_percentage := (bulk_discount / sale_cost) * 100
  discount_percentage = 20 := by
sorry

end ball_bearing_bulk_discount_percentage_l2812_281247


namespace fruit_display_total_l2812_281256

/-- Proves that the total number of fruits on a display is 35, given the specified conditions. -/
theorem fruit_display_total (bananas oranges apples : ℕ) : 
  bananas = 5 → 
  oranges = 2 * bananas → 
  apples = 2 * oranges → 
  bananas + oranges + apples = 35 := by
sorry

end fruit_display_total_l2812_281256


namespace pet_store_combinations_l2812_281295

def num_puppies : ℕ := 20
def num_kittens : ℕ := 6
def num_hamsters : ℕ := 8

def alice_choices : ℕ := num_puppies
def bob_pet_type_choices : ℕ := 2
def bob_specific_pet_choices : ℕ := num_kittens
def charlie_choices : ℕ := num_hamsters

theorem pet_store_combinations : 
  alice_choices * bob_pet_type_choices * bob_specific_pet_choices * charlie_choices = 1920 := by
  sorry

end pet_store_combinations_l2812_281295


namespace min_value_cubic_expression_l2812_281207

theorem min_value_cubic_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 15 * x - y = 22) :
  x^3 + y^3 - x^2 - y^2 ≥ 1 ∧
  (x^3 + y^3 - x^2 - y^2 = 1 ↔ x = 3/2 ∧ y = 1/2) :=
by sorry

end min_value_cubic_expression_l2812_281207


namespace count_true_propositions_l2812_281277

/-- Represents a line in the form ax + by + c = 0 --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The proposition p --/
def p (l1 l2 : Line) : Prop :=
  (∃ (k : ℝ), k ≠ 0 ∧ l1.a = k * l2.a ∧ l1.b = k * l2.b) → l1.a * l2.b - l2.a * l1.b = 0

/-- The converse of p --/
def p_converse (l1 l2 : Line) : Prop :=
  l1.a * l2.b - l2.a * l1.b = 0 → (∃ (k : ℝ), k ≠ 0 ∧ l1.a = k * l2.a ∧ l1.b = k * l2.b)

/-- Count of true propositions among p, its converse, negation, and contrapositive --/
def f_p : ℕ := 2

/-- The main theorem --/
theorem count_true_propositions :
  (∀ l1 l2 : Line, p l1 l2) ∧
  (∃ l1 l2 : Line, ¬(p_converse l1 l2)) ∧
  f_p = 2 := by sorry

end count_true_propositions_l2812_281277


namespace linear_functions_relation_l2812_281228

/-- Given two linear functions f and g, prove that A + B = 2A under certain conditions -/
theorem linear_functions_relation (A B : ℝ) 
  (f : ℝ → ℝ) (g : ℝ → ℝ)
  (hf : ∀ x, f x = A * x + B + 1)
  (hg : ∀ x, g x = B * x + A - 1)
  (hAB : A ≠ -B)
  (h_comp : ∀ x, f (g x) - g (f x) = A - 2 * B) :
  A + B = 2 * A :=
sorry

end linear_functions_relation_l2812_281228


namespace tenth_term_value_l2812_281239

/-- An arithmetic sequence with 30 terms, first term 3, and last term 88 -/
def arithmetic_sequence (n : ℕ) : ℚ :=
  let d := (88 - 3) / 29
  3 + (n - 1) * d

/-- The 10th term of the arithmetic sequence is 852/29 -/
theorem tenth_term_value : arithmetic_sequence 10 = 852 / 29 := by
  sorry

end tenth_term_value_l2812_281239


namespace parabola_point_value_l2812_281263

/-- Given a parabola y = x^2 + (a+1)x + a that passes through the point (-1, m),
    prove that m = 0 -/
theorem parabola_point_value (a m : ℝ) : 
  ((-1)^2 + (a+1)*(-1) + a = m) → m = 0 := by
  sorry

end parabola_point_value_l2812_281263


namespace square_difference_401_399_l2812_281266

theorem square_difference_401_399 : 401^2 - 399^2 = 1600 := by sorry

end square_difference_401_399_l2812_281266


namespace circle_area_theorem_l2812_281238

theorem circle_area_theorem (r : ℝ) (h : r > 0) :
  (2 * (1 / (2 * Real.pi * r)) = r / 2) → (Real.pi * r^2 = 2) := by
  sorry

end circle_area_theorem_l2812_281238


namespace books_returned_percentage_l2812_281205

/-- Calculates the percentage of loaned books that were returned -/
def percentage_books_returned (initial_books : ℕ) (final_books : ℕ) (loaned_books : ℕ) : ℚ :=
  let returned_books := final_books - (initial_books - loaned_books)
  (returned_books : ℚ) / (loaned_books : ℚ) * 100

/-- Proves that the percentage of loaned books returned is 70% -/
theorem books_returned_percentage :
  percentage_books_returned 75 60 50 = 70 := by
  sorry

#eval percentage_books_returned 75 60 50

end books_returned_percentage_l2812_281205


namespace stock_percentage_return_l2812_281255

def stock_yield : ℝ := 0.08
def market_value : ℝ := 137.5

theorem stock_percentage_return :
  (stock_yield * market_value) / market_value * 100 = stock_yield * 100 := by
  sorry

end stock_percentage_return_l2812_281255


namespace number_of_factors_7200_l2812_281216

theorem number_of_factors_7200 : Nat.card (Nat.divisors 7200) = 45 := by
  sorry

end number_of_factors_7200_l2812_281216


namespace function_value_theorem_l2812_281291

theorem function_value_theorem (f : ℝ → ℝ) (h : ∀ x, f ((1/2) * x - 1) = 2 * x + 3) :
  f (-3/4) = 4 :=
by sorry

end function_value_theorem_l2812_281291


namespace watch_loss_percentage_l2812_281226

/-- Proves that the loss percentage is 5% when a watch is sold for Rs. 1140,
    given that selling it for Rs. 1260 would result in a 5% profit. -/
theorem watch_loss_percentage
  (loss_price : ℝ)
  (profit_price : ℝ)
  (profit_percentage : ℝ)
  (h1 : loss_price = 1140)
  (h2 : profit_price = 1260)
  (h3 : profit_percentage = 0.05)
  : (profit_price / (1 + profit_percentage) - loss_price) / (profit_price / (1 + profit_percentage)) = 0.05 := by
  sorry

#check watch_loss_percentage

end watch_loss_percentage_l2812_281226


namespace probability_sum_10_four_dice_l2812_281292

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The number of dice thrown -/
def numDice : ℕ := 4

/-- The target sum we're looking for -/
def targetSum : ℕ := 10

/-- The total number of possible outcomes when throwing four dice -/
def totalOutcomes : ℕ := numFaces ^ numDice

/-- The number of favorable outcomes (combinations that sum to 10) -/
def favorableOutcomes : ℕ := 46

/-- The probability of getting a sum of 10 when throwing four 6-sided dice -/
theorem probability_sum_10_four_dice : 
  (favorableOutcomes : ℚ) / totalOutcomes = 23 / 648 := by sorry

end probability_sum_10_four_dice_l2812_281292


namespace number_of_sets_l2812_281261

/-- Represents a four-digit number in the game "Set" -/
def SetNumber := Fin 4 → Fin 3

/-- Checks if three numbers form a valid set in the game "Set" -/
def is_valid_set (a b c : SetNumber) : Prop :=
  ∀ i : Fin 4, (a i = b i ∧ b i = c i) ∨ (a i ≠ b i ∧ b i ≠ c i ∧ a i ≠ c i)

/-- The set of all possible four-digit numbers in the game "Set" -/
def all_set_numbers : Finset SetNumber :=
  sorry

/-- The set of all valid sets in the game "Set" -/
def all_valid_sets : Finset (Finset SetNumber) :=
  sorry

/-- The main theorem stating the number of valid sets in the game "Set" -/
theorem number_of_sets : Finset.card all_valid_sets = 1080 :=
  sorry

end number_of_sets_l2812_281261


namespace composite_function_evaluation_l2812_281229

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x + 2
def g (x : ℝ) : ℝ := 5 * x + 2

-- State the theorem
theorem composite_function_evaluation : g (f (g 1)) = 82 := by
  sorry

end composite_function_evaluation_l2812_281229


namespace factorization_difference_of_squares_l2812_281286

theorem factorization_difference_of_squares (a b : ℝ) : a^2 * b^2 - 9 = (a*b + 3) * (a*b - 3) := by
  sorry

end factorization_difference_of_squares_l2812_281286


namespace hcf_problem_l2812_281257

theorem hcf_problem (a b : ℕ+) (h1 : a * b = 2560) (h2 : Nat.lcm a b = 160) :
  Nat.gcd a b = 16 := by
  sorry

end hcf_problem_l2812_281257


namespace average_z_squared_l2812_281202

theorem average_z_squared (z : ℝ) : 
  (0 + 2 * z^2 + 4 * z^2 + 8 * z^2 + 16 * z^2) / 5 = 6 * z^2 := by
  sorry

end average_z_squared_l2812_281202


namespace polynomial_product_no_x_terms_l2812_281220

theorem polynomial_product_no_x_terms
  (a b : ℚ)
  (h1 : a ≠ 0)
  (h2 : ∀ x : ℚ, (a * x^2 + b * x + 1) * (3 * x - 2) = 3 * a * x^3 - 2) :
  a = 9/4 ∧ b = 3/2 := by
sorry

end polynomial_product_no_x_terms_l2812_281220


namespace find_unknown_number_l2812_281258

theorem find_unknown_number (n : ℕ) : 
  (∀ m : ℕ, m < 3555 → ¬(711 ∣ m ∧ n ∣ m)) → 
  (711 ∣ 3555 ∧ n ∣ 3555) → 
  n = 5 := by
sorry

end find_unknown_number_l2812_281258


namespace isosceles_triangle_from_square_l2812_281282

/-- Given a square with side length a, there exists an isosceles triangle with the specified properties --/
theorem isosceles_triangle_from_square (a : ℝ) (h : a > 0) :
  ∃ (x y z : ℝ),
    -- The base of the triangle
    x = a * Real.sqrt 3 ∧
    -- The height of the triangle
    y = (2 * x) / 3 ∧
    -- The equal sides of the triangle
    z = (5 * a * Real.sqrt 3) / 6 ∧
    -- Area equality
    (1 / 2) * x * y = a^2 ∧
    -- Sum of base and height equals sum of equal sides
    x + y = 2 * z ∧
    -- Pythagorean theorem
    y^2 + (x / 2)^2 = z^2 := by
  sorry

end isosceles_triangle_from_square_l2812_281282


namespace vector_addition_l2812_281268

-- Define the vectors a and b
def a : Fin 2 → ℝ := ![2, 1]
def b : Fin 2 → ℝ := ![1, 5]

-- Define the operation 2a + b
def result : Fin 2 → ℝ := fun i => 2 * a i + b i

-- Theorem statement
theorem vector_addition : result = ![5, 7] := by sorry

end vector_addition_l2812_281268


namespace card_sequence_periodicity_l2812_281201

def planet_value : ℕ := 2010
def hegemon_value (planets : ℕ) : ℕ := 4 * planets

def card_choice (n : ℕ) : ℕ := 
  if n ≤ 503 then 0 else (n - 503) % 2

theorem card_sequence_periodicity :
  ∃ (k : ℕ), k > 0 ∧ ∀ (n : ℕ), n ≥ 503 → card_choice (n + k) = card_choice n :=
sorry

end card_sequence_periodicity_l2812_281201


namespace bug_safe_probability_l2812_281279

theorem bug_safe_probability (r : ℝ) (h : r = 3) :
  let safe_radius := r - 1
  let total_volume := (4 / 3) * Real.pi * r^3
  let safe_volume := (4 / 3) * Real.pi * safe_radius^3
  safe_volume / total_volume = 8 / 27 := by
sorry

end bug_safe_probability_l2812_281279


namespace willy_distance_theorem_l2812_281235

/-- Represents the distances from Willy to the corners of the square lot -/
structure Distances where
  d₁ : ℝ
  d₂ : ℝ
  d₃ : ℝ
  d₄ : ℝ

/-- The conditions given in the problem -/
def satisfies_conditions (d : Distances) : Prop :=
  d.d₁ < d.d₂ ∧ d.d₂ < d.d₄ ∧ d.d₄ < d.d₃ ∧
  d.d₂ = (d.d₁ + d.d₃) / 2 ∧
  d.d₄ ^ 2 = d.d₂ * d.d₃

/-- The theorem to be proved -/
theorem willy_distance_theorem (d : Distances) (h : satisfies_conditions d) :
  d.d₁ ^ 2 = (4 * d.d₁ * d.d₃ - d.d₃ ^ 2) / 3 := by
  sorry

end willy_distance_theorem_l2812_281235


namespace cats_not_eating_apples_or_chicken_l2812_281212

theorem cats_not_eating_apples_or_chicken
  (total_cats : ℕ)
  (cats_liking_apples : ℕ)
  (cats_liking_chicken : ℕ)
  (cats_liking_both : ℕ)
  (h1 : total_cats = 80)
  (h2 : cats_liking_apples = 15)
  (h3 : cats_liking_chicken = 60)
  (h4 : cats_liking_both = 10) :
  total_cats - (cats_liking_apples + cats_liking_chicken - cats_liking_both) = 15 := by
  sorry

end cats_not_eating_apples_or_chicken_l2812_281212


namespace inequalities_for_positive_reals_l2812_281267

theorem inequalities_for_positive_reals (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  a * b ≤ 1 ∧ a^2 + b^2 ≥ 2 ∧ Real.sqrt a + Real.sqrt b ≤ 2 := by
  sorry

end inequalities_for_positive_reals_l2812_281267


namespace gcd_b_consecutive_is_one_l2812_281284

def b (n : ℕ) : ℤ := (7^n - 1) / 6

theorem gcd_b_consecutive_is_one (n : ℕ) : 
  Nat.gcd (Int.natAbs (b n)) (Int.natAbs (b (n + 1))) = 1 := by sorry

end gcd_b_consecutive_is_one_l2812_281284


namespace son_age_proof_l2812_281265

theorem son_age_proof (father_age son_age : ℕ) : 
  father_age = 36 →
  4 * son_age = father_age →
  father_age - son_age = 27 →
  son_age = 9 := by
sorry

end son_age_proof_l2812_281265


namespace product_trailing_zeros_l2812_281233

/-- The number of trailing zeros in the product of all multiples of 5 from 5 to 2015 -/
def trailingZeros : ℕ :=
  let n := 2015 / 5  -- number of terms in the product
  let factorsOf2 := (n / 2) + (n / 4) + (n / 8) + (n / 16) + (n / 32) + (n / 64) + (n / 128) + (n / 256)
  factorsOf2

theorem product_trailing_zeros : trailingZeros = 398 := by
  sorry

end product_trailing_zeros_l2812_281233


namespace optimization_problem_l2812_281222

theorem optimization_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 1) :
  ((2 / x) + (1 / y) ≥ 9) ∧
  (4 * x^2 + y^2 ≥ 1/2) ∧
  (Real.sqrt (2 * x) + Real.sqrt y ≤ Real.sqrt 2) := by
sorry

end optimization_problem_l2812_281222


namespace sum_in_range_l2812_281232

theorem sum_in_range : ∃ (x : ℚ), 
  (x = 3 + 3/8 + 4 + 1/3 + 6 + 1/21 - 2) ∧ 
  (11.5 < x) ∧ 
  (x < 12) := by
  sorry

end sum_in_range_l2812_281232


namespace gold_coin_percentage_is_49_percent_l2812_281280

/-- Represents the composition of objects in an urn -/
structure UrnComposition where
  bead_percentage : ℝ
  silver_coin_percentage : ℝ

/-- Calculates the percentage of gold coins in the urn -/
def gold_coin_percentage (urn : UrnComposition) : ℝ :=
  (1 - urn.bead_percentage) * (1 - urn.silver_coin_percentage)

/-- Theorem stating that for the given urn composition, 
    the percentage of gold coins is 49% -/
theorem gold_coin_percentage_is_49_percent 
  (urn : UrnComposition) 
  (h1 : urn.bead_percentage = 0.3) 
  (h2 : urn.silver_coin_percentage = 0.3) : 
  gold_coin_percentage urn = 0.49 := by
  sorry

#eval gold_coin_percentage ⟨0.3, 0.3⟩

end gold_coin_percentage_is_49_percent_l2812_281280


namespace polynomial_factorization_l2812_281298

theorem polynomial_factorization (x : ℝ) :
  x^4 + 2021*x^2 + 2020*x + 2021 = (x^2 + x + 1)*(x^2 - x + 2021) := by
  sorry

end polynomial_factorization_l2812_281298


namespace garden_walkway_area_l2812_281290

/-- Represents the configuration of a garden with flower beds and walkways -/
structure Garden where
  bed_width : ℕ
  bed_height : ℕ
  walkway_width : ℕ
  rows : ℕ
  beds_in_first_row : ℕ
  beds_in_other_rows : ℕ

/-- Calculates the total area of walkways in the garden -/
def walkway_area (g : Garden) : ℕ :=
  let total_width := g.bed_width * g.beds_in_first_row + (g.beds_in_first_row + 1) * g.walkway_width
  let total_height := g.bed_height * g.rows + (g.rows + 1) * g.walkway_width
  let total_area := total_width * total_height
  let bed_area := (g.bed_width * g.bed_height) * (g.beds_in_first_row + (g.rows - 1) * g.beds_in_other_rows)
  total_area - bed_area

/-- The theorem stating that for the given garden configuration, the walkway area is 488 square feet -/
theorem garden_walkway_area :
  let g : Garden := {
    bed_width := 8,
    bed_height := 3,
    walkway_width := 2,
    rows := 4,
    beds_in_first_row := 3,
    beds_in_other_rows := 2
  }
  walkway_area g = 488 := by sorry

end garden_walkway_area_l2812_281290


namespace sqrt_product_equals_three_halves_l2812_281269

theorem sqrt_product_equals_three_halves : 
  Real.sqrt 5 * Real.sqrt (9 / 20) = 3 / 2 := by
  sorry

end sqrt_product_equals_three_halves_l2812_281269


namespace equal_digit_probability_l2812_281244

def num_dice : ℕ := 5
def sides_per_die : ℕ := 20
def one_digit_sides : ℕ := 9
def two_digit_sides : ℕ := 11

theorem equal_digit_probability : 
  (num_dice.choose (num_dice / 2)) * 
  ((two_digit_sides : ℚ) / sides_per_die) ^ (num_dice / 2) * 
  ((one_digit_sides : ℚ) / sides_per_die) ^ (num_dice - num_dice / 2) = 
  1062889 / 128000000 := by sorry

end equal_digit_probability_l2812_281244


namespace symmetry_yOz_correct_l2812_281285

/-- Given a point (x, y, z) in 3D space, this function returns its symmetrical point
    with respect to the yOz plane -/
def symmetry_yOz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-x, y, z)

theorem symmetry_yOz_correct :
  symmetry_yOz (1, 2, 1) = (-1, 2, 1) := by
  sorry

end symmetry_yOz_correct_l2812_281285


namespace poster_width_l2812_281296

theorem poster_width (height : ℝ) (area : ℝ) (width : ℝ) 
  (h1 : height = 7)
  (h2 : area = 28)
  (h3 : area = width * height) : 
  width = 4 := by sorry

end poster_width_l2812_281296


namespace inequality_solution_equation_solution_range_l2812_281203

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 4| + |x + 1|

-- Theorem for the first part of the problem
theorem inequality_solution :
  {x : ℝ | f x ≤ 9} = {x : ℝ | -2 ≤ x ∧ x ≤ 4} :=
sorry

-- Theorem for the second part of the problem
theorem equation_solution_range :
  {a : ℝ | ∃ x ∈ Set.Icc 0 2, f x = -x^2 + a} = Set.Icc (19/4) 7 :=
sorry

end inequality_solution_equation_solution_range_l2812_281203


namespace probability_of_convex_quadrilateral_l2812_281215

-- Define the number of points on the circle
def num_points : ℕ := 8

-- Define the number of chords to be selected
def num_selected_chords : ℕ := 4

-- Define the total number of possible chords
def total_chords : ℕ := num_points.choose 2

-- Define the number of ways to select the chords
def ways_to_select_chords : ℕ := total_chords.choose num_selected_chords

-- Define the number of ways to form a convex quadrilateral
def convex_quadrilaterals : ℕ := num_points.choose 4

-- State the theorem
theorem probability_of_convex_quadrilateral :
  (convex_quadrilaterals : ℚ) / ways_to_select_chords = 2 / 585 :=
sorry

end probability_of_convex_quadrilateral_l2812_281215


namespace sin_cos_sum_equals_sqrt_sum_half_l2812_281297

theorem sin_cos_sum_equals_sqrt_sum_half :
  Real.sin (14 * π / 3) + Real.cos (-25 * π / 4) = (Real.sqrt 3 + Real.sqrt 2) / 2 := by
  sorry

end sin_cos_sum_equals_sqrt_sum_half_l2812_281297


namespace trapezoid_to_square_l2812_281251

/-- Represents a trapezoid with bases a and b, and height h -/
structure Trapezoid where
  a : ℝ
  b : ℝ
  h : ℝ
  area_eq : (a + b) * h / 2 = 5

/-- Represents a square with side length s -/
structure Square where
  s : ℝ
  area_eq : s^2 = 5

/-- Theorem stating that a trapezoid with area 5 can be cut into three parts to form a square -/
theorem trapezoid_to_square (t : Trapezoid) : ∃ (sq : Square), True := by
  sorry

end trapezoid_to_square_l2812_281251


namespace ways_to_choose_all_suits_formula_l2812_281281

/-- The number of ways to choose 13 cards from a 52-card deck such that all four suits are represented -/
def waysToChooseAllSuits : ℕ :=
  Nat.choose 52 13 - 4 * Nat.choose 39 13 + 6 * Nat.choose 26 13 - 4 * Nat.choose 13 13

/-- Theorem stating that the number of ways to choose 13 cards from a 52-card deck
    such that all four suits are represented is equal to the given formula -/
theorem ways_to_choose_all_suits_formula :
  waysToChooseAllSuits =
    Nat.choose 52 13 - 4 * Nat.choose 39 13 + 6 * Nat.choose 26 13 - 4 * Nat.choose 13 13 := by
  sorry

#eval waysToChooseAllSuits

end ways_to_choose_all_suits_formula_l2812_281281


namespace largest_binomial_coefficient_sum_l2812_281240

theorem largest_binomial_coefficient_sum (n : ℕ) : 
  (Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 n) → n ≤ 6 :=
by sorry

end largest_binomial_coefficient_sum_l2812_281240


namespace problem_solid_surface_area_l2812_281293

/-- Represents a 3D solid composed of unit cubes -/
structure CubeSolid where
  cubes : ℕ
  top_layer : ℕ
  bottom_layer : ℕ
  height : ℕ
  length : ℕ
  depth : ℕ

/-- Calculates the surface area of a CubeSolid -/
def surface_area (s : CubeSolid) : ℕ := sorry

/-- The specific solid described in the problem -/
def problem_solid : CubeSolid :=
  { cubes := 15
  , top_layer := 5
  , bottom_layer := 5
  , height := 3
  , length := 5
  , depth := 3 }

/-- Theorem stating that the surface area of the problem_solid is 26 square units -/
theorem problem_solid_surface_area :
  surface_area problem_solid = 26 := by sorry

end problem_solid_surface_area_l2812_281293


namespace perp_foot_curve_equation_l2812_281276

/-- The curve traced by the feet of perpendiculars from the origin to a moving unit segment -/
def PerpFootCurve (x y : ℝ) : Prop :=
  (x^2 + y^2)^3 = x^2 * y^2

/-- A point on the x-axis -/
def PointOnXAxis (p : ℝ × ℝ) : Prop :=
  p.2 = 0

/-- A point on the y-axis -/
def PointOnYAxis (p : ℝ × ℝ) : Prop :=
  p.1 = 0

/-- The distance between two points is 1 -/
def UnitDistance (p q : ℝ × ℝ) : Prop :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2 = 1

/-- The perpendicular foot from the origin to a line segment -/
def PerpFoot (p : ℝ × ℝ) (a b : ℝ × ℝ) : Prop :=
  (p.1 * (b.1 - a.1) + p.2 * (b.2 - a.2) = 0) ∧
  (∃ t : ℝ, p = (a.1 + t * (b.1 - a.1), a.2 + t * (b.2 - a.2)) ∧ 0 ≤ t ∧ t ≤ 1)

theorem perp_foot_curve_equation (x y : ℝ) :
  (∃ a b : ℝ × ℝ, PointOnXAxis a ∧ PointOnYAxis b ∧ UnitDistance a b ∧
    PerpFoot (x, y) a b) →
  PerpFootCurve x y :=
sorry

end perp_foot_curve_equation_l2812_281276


namespace shelf_filling_theorem_l2812_281250

/-- Represents the thickness of a programming book -/
def t : ℝ := sorry

/-- The number of programming books that can fill the shelf -/
def P : ℕ := sorry

/-- The number of biology books in a combination that can fill the shelf -/
def B : ℕ := sorry

/-- The number of physics books in a combination that can fill the shelf -/
def F : ℕ := sorry

/-- The number of programming books in a combination with biology books that can fill the shelf -/
def R : ℕ := sorry

/-- The number of biology books in a combination with programming books that can fill the shelf -/
def C : ℕ := sorry

/-- The number of programming books that alone would fill the shelf (to be proven) -/
def Q : ℕ := sorry

/-- The length of the shelf -/
def shelf_length : ℝ := P * t

/-- Theorem stating that Q equals R + 2C -/
theorem shelf_filling_theorem (h1 : P * t = shelf_length)
                               (h2 : 2 * B * t + 3 * F * t = shelf_length)
                               (h3 : R * t + 2 * C * t = shelf_length)
                               (h4 : Q * t = shelf_length)
                               (h5 : P ≠ B ∧ P ≠ F ∧ P ≠ R ∧ P ≠ C ∧ B ≠ F ∧ B ≠ R ∧ B ≠ C ∧ F ≠ R ∧ F ≠ C ∧ R ≠ C)
                               (h6 : P > 0 ∧ B > 0 ∧ F > 0 ∧ R > 0 ∧ C > 0) :
  Q = R + 2 * C :=
sorry

end shelf_filling_theorem_l2812_281250


namespace regular_toenails_in_jar_l2812_281249

def jar_capacity : ℕ := 100
def big_toenail_count : ℕ := 20
def remaining_space : ℕ := 20

def big_toenail_size : ℕ := 2
def regular_toenail_size : ℕ := 1

theorem regular_toenails_in_jar :
  ∃ (regular_count : ℕ),
    regular_count * regular_toenail_size +
    big_toenail_count * big_toenail_size +
    remaining_space * regular_toenail_size = jar_capacity ∧
    regular_count = 40 := by
  sorry

end regular_toenails_in_jar_l2812_281249


namespace parabola_symmetric_intersection_l2812_281253

/-- Represents a parabola of the form y = ax^2 + bx + c --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- Represents a point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem: For a parabola with symmetric axis x=1 and one x-axis intersection at (3,0),
    the other x-axis intersection is at (-1,0) --/
theorem parabola_symmetric_intersection
  (p : Parabola)
  (symmetric_axis : ℝ)
  (intersection : Point)
  (h1 : symmetric_axis = 1)
  (h2 : intersection = Point.mk 3 0)
  (h3 : p.a * intersection.x^2 + p.b * intersection.x + p.c = 0)
  : ∃ (other : Point), 
    other = Point.mk (-1) 0 ∧ 
    p.a * other.x^2 + p.b * other.x + p.c = 0 :=
by sorry

end parabola_symmetric_intersection_l2812_281253


namespace power_72_equals_m3n2_l2812_281204

theorem power_72_equals_m3n2 (a m n : ℝ) (h1 : 2^a = m) (h2 : 3^a = n) : 72^a = m^3 * n^2 := by
  sorry

end power_72_equals_m3n2_l2812_281204


namespace relationship_abc_l2812_281236

theorem relationship_abc (a b c : ℝ) 
  (ha : a = (2/3)^(-(1/3 : ℝ))) 
  (hb : b = (5/3)^(-(2/3 : ℝ))) 
  (hc : c = (3/2)^(2/3 : ℝ)) : 
  b < a ∧ a < c := by sorry

end relationship_abc_l2812_281236


namespace min_lines_proof_l2812_281289

/-- The number of regions created by n lines in a plane -/
def regions (n : ℕ) : ℕ := 1 + n * (n + 1) / 2

/-- The minimum number of lines needed to divide a plane into at least 1000 regions -/
def min_lines_for_1000_regions : ℕ := 45

theorem min_lines_proof :
  (∀ k < min_lines_for_1000_regions, regions k < 1000) ∧
  regions min_lines_for_1000_regions ≥ 1000 := by
  sorry

#eval regions min_lines_for_1000_regions

end min_lines_proof_l2812_281289


namespace sequence_odd_terms_l2812_281210

theorem sequence_odd_terms (a : ℕ → ℤ) 
  (h1 : a 1 = 2)
  (h2 : a 2 = 7)
  (h3 : ∀ n ≥ 2, -1/2 < (a (n+1) : ℚ) - (a n)^2 / (a (n-1))^2 ∧ 
                 (a (n+1) : ℚ) - (a n)^2 / (a (n-1))^2 ≤ 1/2) :
  ∀ n > 1, Odd (a n) := by
sorry

end sequence_odd_terms_l2812_281210


namespace hexagon_triangle_area_l2812_281252

/-- The area of a regular hexagon with side length 2, topped by an equilateral triangle with side length 2, is 7√3 square units. -/
theorem hexagon_triangle_area : 
  let hexagon_side : ℝ := 2
  let triangle_side : ℝ := 2
  let hexagon_area : ℝ := (3 * Real.sqrt 3 / 2) * hexagon_side^2
  let triangle_area : ℝ := (Real.sqrt 3 / 4) * triangle_side^2
  hexagon_area + triangle_area = 7 * Real.sqrt 3 := by
sorry


end hexagon_triangle_area_l2812_281252


namespace kitty_vacuuming_time_l2812_281294

/-- Represents the weekly cleaning routine for a living room -/
structure LivingRoomCleaning where
  pickup_time : ℕ
  window_time : ℕ
  dusting_time : ℕ
  total_time_4weeks : ℕ

/-- Calculates the time spent vacuuming per week -/
def vacuuming_time_per_week (cleaning : LivingRoomCleaning) : ℕ :=
  let other_tasks_time := cleaning.pickup_time + cleaning.window_time + cleaning.dusting_time
  let total_other_tasks_4weeks := other_tasks_time * 4
  let total_vacuuming_4weeks := cleaning.total_time_4weeks - total_other_tasks_4weeks
  total_vacuuming_4weeks / 4

/-- Theorem stating that Kitty spends 20 minutes vacuuming per week -/
theorem kitty_vacuuming_time (cleaning : LivingRoomCleaning)
    (h1 : cleaning.pickup_time = 5)
    (h2 : cleaning.window_time = 15)
    (h3 : cleaning.dusting_time = 10)
    (h4 : cleaning.total_time_4weeks = 200) :
    vacuuming_time_per_week cleaning = 20 := by
  sorry

end kitty_vacuuming_time_l2812_281294


namespace min_value_a_l2812_281299

theorem min_value_a : 
  (∀ (x y : ℝ), x > 0 → y > 0 → x + Real.sqrt (x * y) ≤ a * (x + y)) → 
  a ≥ (Real.sqrt 2 + 1) / 2 := by
sorry

end min_value_a_l2812_281299


namespace amber_pieces_count_l2812_281213

theorem amber_pieces_count (green clear : ℕ) (h1 : green = 35) (h2 : clear = 85) 
  (h3 : green = (green + clear + amber) / 4) : amber = 20 := by
  sorry

end amber_pieces_count_l2812_281213


namespace dictionary_page_count_l2812_281211

/-- Count the occurrences of digit 1 in a number -/
def countOnesInNumber (n : ℕ) : ℕ := sorry

/-- Count the occurrences of digit 1 in a range of numbers from 1 to n -/
def countOnesInRange (n : ℕ) : ℕ := sorry

/-- The number of pages in the dictionary -/
def dictionaryPages : ℕ := 3152

/-- The total count of digit 1 appearances -/
def totalOnesCount : ℕ := 1988

theorem dictionary_page_count :
  countOnesInRange dictionaryPages = totalOnesCount ∧
  ∀ m : ℕ, m < dictionaryPages → countOnesInRange m < totalOnesCount :=
sorry

end dictionary_page_count_l2812_281211


namespace number_problem_l2812_281275

theorem number_problem (x : ℝ) : 0.35 * x = 0.50 * x - 24 → x = 160 := by
  sorry

end number_problem_l2812_281275


namespace souvenir_spending_difference_l2812_281214

def total_spent : ℚ := 548
def keychain_bracelet_spent : ℚ := 347

theorem souvenir_spending_difference :
  keychain_bracelet_spent - (total_spent - keychain_bracelet_spent) = 146 := by
  sorry

end souvenir_spending_difference_l2812_281214


namespace train_distance_difference_l2812_281224

/-- Proves that the difference in distance traveled by two trains is 70 km -/
theorem train_distance_difference (v1 v2 total_distance : ℝ) 
  (h1 : v1 = 20) 
  (h2 : v2 = 25) 
  (h3 : total_distance = 630) : ∃ (t : ℝ), v2 * t - v1 * t = 70 := by
  sorry

end train_distance_difference_l2812_281224


namespace ricky_sold_nine_l2812_281200

/-- Represents the number of glasses of lemonade sold by each person -/
structure LemonadeSales where
  katya : ℕ
  ricky : ℕ
  tina : ℕ

/-- The conditions of the lemonade sales problem -/
def lemonade_problem (sales : LemonadeSales) : Prop :=
  sales.katya = 8 ∧
  sales.tina = 2 * (sales.katya + sales.ricky) ∧
  sales.tina = sales.katya + 26

/-- Theorem stating that under the given conditions, Ricky sold 9 glasses of lemonade -/
theorem ricky_sold_nine (sales : LemonadeSales) 
  (h : lemonade_problem sales) : sales.ricky = 9 := by
  sorry

end ricky_sold_nine_l2812_281200


namespace february_highest_percentage_difference_l2812_281221

/-- Represents the sales data for a vendor in a given month -/
structure SalesData where
  quantity : Nat
  price : Float

/-- Calculates the revenue from sales data -/
def revenue (data : SalesData) : Float :=
  data.quantity.toFloat * data.price

/-- Calculates the percentage difference between two revenues -/
def percentageDifference (r1 r2 : Float) : Float :=
  (max r1 r2 - min r1 r2) / (min r1 r2) * 100

/-- Represents a month -/
inductive Month
  | January | February | March | April | May | June

/-- Andy's sales data for each month -/
def andySales : Month → SalesData
  | .January => ⟨100, 2⟩
  | .February => ⟨150, 1.5⟩
  | .March => ⟨120, 2.5⟩
  | .April => ⟨80, 4⟩
  | .May => ⟨140, 1.75⟩
  | .June => ⟨110, 3⟩

/-- Bella's sales data for each month -/
def bellaSales : Month → SalesData
  | .January => ⟨90, 2.2⟩
  | .February => ⟨100, 1.75⟩
  | .March => ⟨80, 3⟩
  | .April => ⟨85, 3.5⟩
  | .May => ⟨135, 2⟩
  | .June => ⟨160, 2.5⟩

/-- Theorem: February has the highest percentage difference in revenue -/
theorem february_highest_percentage_difference :
  ∀ m : Month, m ≠ Month.February →
    percentageDifference (revenue (andySales Month.February)) (revenue (bellaSales Month.February)) ≥
    percentageDifference (revenue (andySales m)) (revenue (bellaSales m)) :=
by sorry

end february_highest_percentage_difference_l2812_281221


namespace algebraic_expression_correct_l2812_281241

/-- The algebraic expression for "three times x minus the cube of y" -/
def algebraic_expression (x y : ℝ) : ℝ := 3 * x - y^3

/-- Theorem stating that the algebraic expression is correct -/
theorem algebraic_expression_correct (x y : ℝ) : 
  algebraic_expression x y = 3 * x - y^3 := by
  sorry

end algebraic_expression_correct_l2812_281241


namespace num_valid_committees_l2812_281259

/-- Represents a community with speakers of different languages -/
structure Community where
  total : ℕ
  english : ℕ
  german : ℕ
  french : ℕ

/-- Defines a valid committee in the community -/
def ValidCommittee (c : Community) : Prop :=
  c.total = 20 ∧ c.english = 10 ∧ c.german = 10 ∧ c.french = 10

/-- Calculates the number of valid committees -/
noncomputable def NumValidCommittees (c : Community) : ℕ :=
  Nat.choose c.total 3 - Nat.choose (c.total - c.english) 3

/-- Theorem stating the number of valid committees -/
theorem num_valid_committees (c : Community) (h : ValidCommittee c) : 
  NumValidCommittees c = 1020 := by
  sorry


end num_valid_committees_l2812_281259


namespace intersection_sum_l2812_281272

/-- Two lines intersect at a point (3,1). -/
def intersection_point : ℝ × ℝ := (3, 1)

/-- The first line equation: x = (1/3)y + a -/
def line1 (a : ℝ) (x y : ℝ) : Prop := x = (1/3) * y + a

/-- The second line equation: y = (1/3)x + b -/
def line2 (b : ℝ) (x y : ℝ) : Prop := y = (1/3) * x + b

/-- The theorem states that if two lines intersect at (3,1), then a + b = 8/3 -/
theorem intersection_sum (a b : ℝ) : 
  line1 a (intersection_point.1) (intersection_point.2) ∧ 
  line2 b (intersection_point.1) (intersection_point.2) → 
  a + b = 8/3 :=
by sorry

end intersection_sum_l2812_281272


namespace rectangle_dimensions_quadratic_equation_l2812_281219

theorem rectangle_dimensions_quadratic_equation 
  (L W : ℝ) 
  (h1 : L + W = 15) 
  (h2 : L * W = 2 * W^2) : 
  (L = (15 + Real.sqrt 25) / 2 ∧ W = (15 - Real.sqrt 25) / 2) ∨ 
  (L = (15 - Real.sqrt 25) / 2 ∧ W = (15 + Real.sqrt 25) / 2) := by
  sorry

end rectangle_dimensions_quadratic_equation_l2812_281219


namespace min_value_geometric_sequence_l2812_281231

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem min_value_geometric_sequence (a : ℕ → ℝ) :
  geometric_sequence a →
  a 2015 + a 2017 = π →
  ∀ x : ℝ, a 2016 * (a 2014 + a 2018) ≥ π^2 / 2 :=
sorry

end min_value_geometric_sequence_l2812_281231


namespace birthday_number_l2812_281234

theorem birthday_number (T : ℕ) (x y : ℕ+) : 
  200 < T → T < 225 → T^2 = 4 * 10000 + x * 1000 + y * 100 + 29 → T = 223 := by
sorry

end birthday_number_l2812_281234


namespace rectangle_area_l2812_281225

/-- Given a rectangle where the length is thrice the breadth and the perimeter is 40,
    prove that its area is 75. -/
theorem rectangle_area (b : ℝ) (h1 : b > 0) : 
  let l := 3 * b
  let p := 2 * (l + b)
  p = 40 →
  l * b = 75 := by sorry

end rectangle_area_l2812_281225


namespace cosine_pi_third_derivative_l2812_281278

theorem cosine_pi_third_derivative :
  let y : ℝ → ℝ := λ _ => Real.cos (π / 3)
  ∀ x : ℝ, deriv y x = 0 := by
sorry

end cosine_pi_third_derivative_l2812_281278


namespace volume_ratio_of_pyramids_l2812_281208

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a triangular pyramid -/
structure TriangularPyramid where
  apex : Point3D
  base1 : Point3D
  base2 : Point3D
  base3 : Point3D

/-- Calculates the volume of a triangular pyramid -/
def volumeOfTriangularPyramid (pyramid : TriangularPyramid) : ℝ :=
  sorry

/-- Given two points, returns a point that divides the line segment in a given ratio -/
def divideSegment (p1 p2 : Point3D) (ratio : ℝ) : Point3D :=
  sorry

theorem volume_ratio_of_pyramids (P A B C : Point3D) : 
  let PABC := TriangularPyramid.mk P A B C
  let M := divideSegment P C (1/3)
  let N := divideSegment P B (2/3)
  let PAMN := TriangularPyramid.mk P A M N
  (volumeOfTriangularPyramid PAMN) / (volumeOfTriangularPyramid PABC) = 2/9 := by
  sorry

end volume_ratio_of_pyramids_l2812_281208


namespace percentage_calculation_l2812_281223

theorem percentage_calculation (n : ℝ) (h : n = 4800) : n * 0.5 * 0.3 * 0.15 = 108 := by
  sorry

end percentage_calculation_l2812_281223


namespace right_triangle_in_sets_l2812_281245

/-- Checks if three numbers can form a right-angled triangle --/
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

/-- The given sets of numbers --/
def number_sets : List (ℕ × ℕ × ℕ) :=
  [(1, 2, 3), (2, 3, 4), (3, 4, 5), (9, 13, 17)]

theorem right_triangle_in_sets :
  ∃! (a b c : ℕ), (a, b, c) ∈ number_sets ∧ is_right_triangle a b c :=
by
  sorry

end right_triangle_in_sets_l2812_281245


namespace pastries_count_l2812_281209

/-- The number of pastries made by Lola and Lulu -/
def total_pastries (lola_cupcakes lola_poptarts lola_pies lulu_cupcakes lulu_poptarts lulu_pies : ℕ) : ℕ :=
  lola_cupcakes + lola_poptarts + lola_pies + lulu_cupcakes + lulu_poptarts + lulu_pies

/-- Theorem stating the total number of pastries made by Lola and Lulu -/
theorem pastries_count : total_pastries 13 10 8 16 12 14 = 73 := by
  sorry

end pastries_count_l2812_281209


namespace part_one_part_two_l2812_281218

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {x | m - 1 < x ∧ x < 2*m + 1}
def B : Set ℝ := {x | (x - 7) / (x - 2) < 0}

-- Part 1
theorem part_one : A 2 ∩ (Set.univ \ B) = Set.Ioc 1 2 := by sorry

-- Part 2
theorem part_two : ∀ m : ℝ, A m ∪ B = B ↔ m ∈ Set.Iic (-2) ∪ {3} := by sorry

end part_one_part_two_l2812_281218


namespace fraction_problem_l2812_281283

theorem fraction_problem (a b c : ℝ) 
  (h1 : a * b / (a + b) = 3)
  (h2 : b * c / (b + c) = 6)
  (h3 : a * c / (a + c) = 9) :
  c / (a * b) = -35 / 36 := by
  sorry

end fraction_problem_l2812_281283


namespace flour_needed_l2812_281227

theorem flour_needed (total : ℕ) (added : ℕ) (needed : ℕ) : 
  total = 8 ∧ added = 2 → needed = 6 := by
  sorry

end flour_needed_l2812_281227


namespace johnny_practice_days_l2812_281260

/-- The number of days Johnny has been practicing up to now -/
def current_practice_days : ℕ := 40

/-- The number of days in the future when Johnny will have tripled his practice -/
def future_days : ℕ := 80

/-- Represents that Johnny practices the same amount each day -/
axiom consistent_practice : True

/-- In 80 days, Johnny will have 3 times as much practice as he does currently -/
axiom future_practice : current_practice_days + future_days = 3 * current_practice_days

/-- The number of days ago when Johnny had half as much practice -/
def half_practice_days : ℕ := current_practice_days / 2

theorem johnny_practice_days : half_practice_days = 20 := by sorry

end johnny_practice_days_l2812_281260


namespace triangle_properties_l2812_281270

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given conditions for the triangle -/
def TriangleConditions (t : Triangle) : Prop :=
  2 * t.c = t.a + Real.cos t.A * t.b / Real.cos t.B ∧
  t.b = 4 ∧
  t.a + t.c = 3 * Real.sqrt 2

theorem triangle_properties (t : Triangle) (h : TriangleConditions t) :
  t.B = π / 3 ∧ 
  (1 / 2 * t.a * t.c * Real.sin t.B : ℝ) = Real.sqrt 3 / 6 := by
  sorry


end triangle_properties_l2812_281270


namespace smaller_number_l2812_281274

theorem smaller_number (x y : ℝ) (sum_eq : x + y = 18) (diff_eq : x - y = 8) : 
  min x y = 5 := by sorry

end smaller_number_l2812_281274


namespace clown_balloon_count_l2812_281217

/-- The number of balloons a clown has after a series of events -/
def final_balloon_count (initial : ℕ) (additional : ℕ) (given_away : ℕ) (popped : ℕ) : ℕ :=
  initial + additional - given_away - popped

/-- Theorem stating the final number of balloons the clown has -/
theorem clown_balloon_count :
  final_balloon_count 47 13 20 5 = 35 := by
  sorry

end clown_balloon_count_l2812_281217


namespace roots_difference_squared_l2812_281248

theorem roots_difference_squared (α β : ℝ) : 
  α^2 - 3*α + 1 = 0 → β^2 - 3*β + 1 = 0 → (α - β)^2 = 5 := by
  sorry

end roots_difference_squared_l2812_281248


namespace roots_sum_cubes_fourth_powers_l2812_281254

theorem roots_sum_cubes_fourth_powers (α β : ℝ) : 
  α^2 - 3*α - 2 = 0 → β^2 - 3*β - 2 = 0 → 3*α^3 + 8*β^4 = 1229 := by
  sorry

end roots_sum_cubes_fourth_powers_l2812_281254


namespace range_of_2a_plus_3b_l2812_281264

theorem range_of_2a_plus_3b (a b : ℝ) 
  (h1 : -1 ≤ a + b ∧ a + b ≤ 1) 
  (h2 : -1 ≤ a - b ∧ a - b ≤ 1) : 
  (∀ x, 2*a + 3*b ≤ x → 3 ≤ x) ∧ 
  (∀ y, y ≤ 2*a + 3*b → y ≤ -3) :=
by sorry

end range_of_2a_plus_3b_l2812_281264


namespace other_number_proof_l2812_281273

theorem other_number_proof (a b : ℕ+) (h1 : Nat.gcd a b = 16) (h2 : Nat.lcm a b = 396) (h3 : a = 176) : b = 36 := by
  sorry

end other_number_proof_l2812_281273


namespace geometric_sequence_fourth_term_l2812_281246

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_fourth_term
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_positive : ∀ n, a n > 0)
  (h_product : a 1 * a 7 = 36) :
  a 4 = 6 := by
sorry

end geometric_sequence_fourth_term_l2812_281246


namespace vector_simplification_l2812_281230

variable {V : Type*} [AddCommGroup V]

theorem vector_simplification 
  (O P Q S : V) : 
  (O - P) - (Q - P) + (P - S) + (S - P) = O - Q := by sorry

end vector_simplification_l2812_281230


namespace pred_rohem_30_more_pred_rohem_triple_total_sold_is_60_l2812_281288

/-- The number of alarm clocks sold at "Za Rohem" -/
def za_rohem : ℕ := 15

/-- The number of alarm clocks sold at "Před Rohem" -/
def pred_rohem : ℕ := za_rohem + 30

/-- The claim that "Před Rohem" sold 30 more alarm clocks than "Za Rohem" -/
theorem pred_rohem_30_more : pred_rohem = za_rohem + 30 := by sorry

/-- The claim that "Před Rohem" sold three times as many alarm clocks as "Za Rohem" -/
theorem pred_rohem_triple : pred_rohem = 3 * za_rohem := by sorry

/-- The total number of alarm clocks sold at both shops -/
def total_sold : ℕ := za_rohem + pred_rohem

/-- Proof that the total number of alarm clocks sold at both shops is 60 -/
theorem total_sold_is_60 : total_sold = 60 := by sorry

end pred_rohem_30_more_pred_rohem_triple_total_sold_is_60_l2812_281288


namespace copper_carbonate_molecular_weight_l2812_281287

/-- The molecular weight of Copper(II) carbonate for a given number of moles -/
def molecular_weight (moles : ℝ) : ℝ := sorry

/-- Theorem: The molecular weight of one mole of Copper(II) carbonate is 124 grams/mole -/
theorem copper_carbonate_molecular_weight :
  molecular_weight 1 = 124 :=
by
  have h : molecular_weight 8 = 992 := sorry
  sorry

end copper_carbonate_molecular_weight_l2812_281287
