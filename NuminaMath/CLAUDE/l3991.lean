import Mathlib

namespace NUMINAMATH_CALUDE_lottery_profit_l3991_399177

-- Define the card colors
inductive Color
| Black
| Red

-- Define the card values
inductive Value
| One
| Two
| Three
| Four

-- Define a card as a pair of color and value
structure Card where
  color : Color
  value : Value

-- Define the set of cards
def cards : Finset Card := sorry

-- Define the categories
inductive Category
| A  -- Flush
| B  -- Same color
| C  -- Straight
| D  -- Pair
| E  -- Others

-- Function to determine the category of a pair of cards
def categorize : Card → Card → Category := sorry

-- Function to calculate the probability of a category
def probability (c : Category) : Rat := sorry

-- Define the prize values
def prizeValue : Category → Nat
| Category.D => 9  -- First prize
| Category.B => 3  -- Second prize
| _ => 1           -- Third prize

-- Number of participants
def participants : Nat := 300

-- Theorem to prove
theorem lottery_profit :
  (∀ c : Category, c ≠ Category.D → probability Category.D ≤ probability c) ∧
  (∀ c : Category, c ≠ Category.B → probability c ≤ probability Category.B) ∧
  (participants * 3 - (participants * probability Category.D * prizeValue Category.D +
                       participants * probability Category.B * prizeValue Category.B +
                       participants * (1 - probability Category.D - probability Category.B) * 1) = 120) := by
  sorry

end NUMINAMATH_CALUDE_lottery_profit_l3991_399177


namespace NUMINAMATH_CALUDE_investment_problem_l3991_399162

theorem investment_problem (P : ℝ) : 
  (P * 0.15 * 2 - P * 0.12 * 2 = 840) → P = 14000 := by
  sorry

end NUMINAMATH_CALUDE_investment_problem_l3991_399162


namespace NUMINAMATH_CALUDE_square_root_condition_l3991_399121

theorem square_root_condition (x : ℝ) : 
  Real.sqrt ((x - 1)^2) = x - 1 → x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_square_root_condition_l3991_399121


namespace NUMINAMATH_CALUDE_weight_difference_l3991_399101

/-- Given the weights of Heather and Emily, prove the difference in their weights -/
theorem weight_difference (heather_weight emily_weight : ℕ) 
  (h1 : heather_weight = 87)
  (h2 : emily_weight = 9) :
  heather_weight - emily_weight = 78 := by
  sorry

end NUMINAMATH_CALUDE_weight_difference_l3991_399101


namespace NUMINAMATH_CALUDE_combined_list_size_l3991_399147

def combined_friends_list (james_friends john_friends shared_friends : ℕ) : ℕ :=
  james_friends + john_friends - shared_friends

theorem combined_list_size :
  let james_friends : ℕ := 75
  let john_friends : ℕ := 3 * james_friends
  let shared_friends : ℕ := 25
  combined_friends_list james_friends john_friends shared_friends = 275 := by
  sorry

end NUMINAMATH_CALUDE_combined_list_size_l3991_399147


namespace NUMINAMATH_CALUDE_f_has_two_zeros_l3991_399116

-- Define the piecewise function
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + 2*x - 3 else -2 + Real.log x

-- State the theorem
theorem f_has_two_zeros :
  ∃ (a b : ℝ), a ≠ b ∧ f a = 0 ∧ f b = 0 ∧ ∀ x, f x = 0 → x = a ∨ x = b :=
sorry

end NUMINAMATH_CALUDE_f_has_two_zeros_l3991_399116


namespace NUMINAMATH_CALUDE_compute_expression_l3991_399197

theorem compute_expression : 2⁻¹ + |-5| - Real.sin (30 * π / 180) + (π - 1)^0 = 6 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l3991_399197


namespace NUMINAMATH_CALUDE_prob_X_equals_three_l3991_399167

/-- X is a random variable following a binomial distribution B(6, 1/2) -/
def X : Real → Real := sorry

/-- The probability mass function of X -/
def pmf (k : ℕ) : Real := sorry

/-- Theorem: The probability of X = 3 is 5/16 -/
theorem prob_X_equals_three : pmf 3 = 5/16 := by sorry

end NUMINAMATH_CALUDE_prob_X_equals_three_l3991_399167


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3991_399161

/-- Arithmetic sequence properties -/
structure ArithmeticSequence where
  first_term : ℝ
  last_term : ℝ
  sum : ℝ
  num_terms : ℕ

/-- Theorem: Common difference of a specific arithmetic sequence -/
theorem arithmetic_sequence_common_difference
  (seq : ArithmeticSequence)
  (h1 : seq.first_term = 5)
  (h2 : seq.last_term = 45)
  (h3 : seq.sum = 250) :
  let d := (seq.last_term - seq.first_term) / (seq.num_terms - 1)
  d = 40 / 9 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3991_399161


namespace NUMINAMATH_CALUDE_cube_root_of_decimal_l3991_399190

theorem cube_root_of_decimal (x : ℚ) : x = 1/4 → x^3 = 15625/1000000 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_decimal_l3991_399190


namespace NUMINAMATH_CALUDE_proportion_third_number_l3991_399109

theorem proportion_third_number (y : ℝ) : 
  (0.75 : ℝ) / 1.05 = y / 7 → y = 5 := by
  sorry

end NUMINAMATH_CALUDE_proportion_third_number_l3991_399109


namespace NUMINAMATH_CALUDE_one_meeting_before_first_lap_l3991_399104

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℝ
  direction : Bool  -- True for clockwise, False for counterclockwise

/-- Calculates the number of meetings between two runners on a circular track -/
def meetings (track_length : ℝ) (runner1 runner2 : Runner) : ℕ :=
  sorry

theorem one_meeting_before_first_lap (track_length : ℝ) (runner1 runner2 : Runner) :
  track_length = 190 →
  runner1.speed = 7 →
  runner2.speed = 12 →
  runner1.direction ≠ runner2.direction →
  meetings track_length runner1 runner2 = 1 :=
sorry

end NUMINAMATH_CALUDE_one_meeting_before_first_lap_l3991_399104


namespace NUMINAMATH_CALUDE_fraction_irreducible_l3991_399102

theorem fraction_irreducible (n : ℤ) : Int.gcd (14 * n + 3) (21 * n + 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_irreducible_l3991_399102


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3991_399136

theorem absolute_value_inequality (x : ℝ) :
  2 ≤ |x - 3| ∧ |x - 3| ≤ 8 ↔ (-5 ≤ x ∧ x ≤ 1) ∨ (5 ≤ x ∧ x ≤ 11) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3991_399136


namespace NUMINAMATH_CALUDE_chair_rows_theorem_l3991_399194

/-- Given a total number of chairs and chairs per row, calculates the number of rows -/
def calculate_rows (total_chairs : ℕ) (chairs_per_row : ℕ) : ℕ :=
  total_chairs / chairs_per_row

/-- Theorem stating that for 432 total chairs and 16 chairs per row, there are 27 rows -/
theorem chair_rows_theorem :
  calculate_rows 432 16 = 27 := by
  sorry

end NUMINAMATH_CALUDE_chair_rows_theorem_l3991_399194


namespace NUMINAMATH_CALUDE_third_group_size_l3991_399160

theorem third_group_size (total : ℕ) (first_fraction : ℚ) (second_fraction : ℚ)
  (h_total : total = 45)
  (h_first : first_fraction = 1 / 3)
  (h_second : second_fraction = 2 / 5)
  : total - (total * first_fraction).floor - (total * second_fraction).floor = 12 :=
by sorry

end NUMINAMATH_CALUDE_third_group_size_l3991_399160


namespace NUMINAMATH_CALUDE_symmetry_implies_coordinates_l3991_399134

/-- Two points are symmetric with respect to the y-axis if their x-coordinates are opposites and their y-coordinates are the same. -/
def symmetric_wrt_y_axis (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = q.2

theorem symmetry_implies_coordinates (a b : ℝ) :
  symmetric_wrt_y_axis (a, 3) (-2, b) → a = 2 ∧ b = 3 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_coordinates_l3991_399134


namespace NUMINAMATH_CALUDE_negation_at_most_three_l3991_399172

theorem negation_at_most_three (x : ℝ) : ¬(x ≤ 3) ↔ x > 3 := by sorry

end NUMINAMATH_CALUDE_negation_at_most_three_l3991_399172


namespace NUMINAMATH_CALUDE_geometric_progression_terms_l3991_399122

/-- A finite geometric progression with first term 3, second term 12, and last term 3072 has 6 terms -/
theorem geometric_progression_terms : 
  ∀ (b : ℕ → ℝ), 
    b 1 = 3 → 
    b 2 = 12 → 
    (∃ n : ℕ, n > 2 ∧ b n = 3072 ∧ ∀ k : ℕ, 1 < k → k < n → b k / b (k-1) = b 2 / b 1) →
    ∃ n : ℕ, n = 6 ∧ b n = 3072 ∧ ∀ k : ℕ, 1 < k → k < n → b k / b (k-1) = b 2 / b 1 :=
by sorry


end NUMINAMATH_CALUDE_geometric_progression_terms_l3991_399122


namespace NUMINAMATH_CALUDE_rhombus_longer_diagonal_l3991_399198

/-- Given a rhombus with diagonal ratio 2:3 and area 12 cm², prove the longer diagonal is 6 cm -/
theorem rhombus_longer_diagonal (d1 d2 : ℝ) : 
  d1 / d2 = 2 / 3 →  -- ratio of diagonals
  d1 * d2 / 2 = 12 →  -- area of rhombus
  d2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_longer_diagonal_l3991_399198


namespace NUMINAMATH_CALUDE_joan_balloons_l3991_399189

theorem joan_balloons (total sally jessica : ℕ) (h1 : total = 16) (h2 : sally = 5) (h3 : jessica = 2) :
  ∃ joan : ℕ, joan + sally + jessica = total ∧ joan = 9 := by
  sorry

end NUMINAMATH_CALUDE_joan_balloons_l3991_399189


namespace NUMINAMATH_CALUDE_projection_onto_common_vector_l3991_399181

/-- Given two vectors v1 and v2 in ℝ², prove that their projection onto a common vector u results in the vector q. -/
theorem projection_onto_common_vector (v1 v2 u q : ℝ × ℝ) : 
  v1 = (3, 2) → 
  v2 = (2, 5) → 
  q = (27/8, 7/8) → 
  ∃ (t : ℝ), q = v1 + t • (v2 - v1) ∧ 
  (q - v1) • (v2 - v1) = 0 ∧ 
  (q - v2) • (v2 - v1) = 0 :=
by sorry

end NUMINAMATH_CALUDE_projection_onto_common_vector_l3991_399181


namespace NUMINAMATH_CALUDE_subtraction_of_reciprocals_l3991_399128

theorem subtraction_of_reciprocals (p q : ℚ) : 
  (4 / p = 8) → (4 / q = 18) → (p - q = 5 / 18) := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_reciprocals_l3991_399128


namespace NUMINAMATH_CALUDE_more_sad_players_left_l3991_399168

/-- Represents the state of a player in the game -/
inductive PlayerState
| Sad
| Cheerful

/-- Represents the game with its rules and initial state -/
structure Game where
  initialPlayers : Nat
  remainingPlayers : Nat
  sadPlayers : Nat
  cheerfulPlayers : Nat

/-- Definition of a valid game state -/
def validGameState (g : Game) : Prop :=
  g.initialPlayers = 36 ∧
  g.remainingPlayers + g.sadPlayers + g.cheerfulPlayers = g.initialPlayers ∧
  g.remainingPlayers ≥ 1

/-- The game ends when only one player remains -/
def gameEnded (g : Game) : Prop :=
  g.remainingPlayers = 1

/-- Theorem stating that more sad players have left the game than cheerful players when the game ends -/
theorem more_sad_players_left (g : Game) 
  (h1 : validGameState g) 
  (h2 : gameEnded g) : 
  g.sadPlayers > g.cheerfulPlayers :=
sorry

end NUMINAMATH_CALUDE_more_sad_players_left_l3991_399168


namespace NUMINAMATH_CALUDE_geometric_series_sum_l3991_399158

theorem geometric_series_sum : 
  let a : ℚ := 1/4
  let r : ℚ := 1/4
  let n : ℕ := 5
  let series_sum := (a * (1 - r^n)) / (1 - r)
  series_sum = 341/1024 := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l3991_399158


namespace NUMINAMATH_CALUDE_infinite_male_lineage_l3991_399151

/-- Represents a male person --/
structure Male where
  name : String

/-- Represents the son relationship between two males --/
def is_son_of (son father : Male) : Prop := sorry

/-- Adam, the first male --/
def adam : Male := ⟨"Adam"⟩

/-- An infinite sequence of males --/
def male_sequence : ℕ → Male
| 0 => adam
| n + 1 => sorry

/-- Theorem stating the existence of an infinite male lineage starting from Adam --/
theorem infinite_male_lineage :
  (∀ n : ℕ, is_son_of (male_sequence (n + 1)) (male_sequence n)) ∧
  (∀ n : ℕ, ∃ m : ℕ, m > n) :=
sorry

end NUMINAMATH_CALUDE_infinite_male_lineage_l3991_399151


namespace NUMINAMATH_CALUDE_union_A_B_range_of_a_l3991_399154

-- Define the sets A, B, and C
def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 8}
def B : Set ℝ := {x | 1 < x ∧ x < 6}
def C (a : ℝ) : Set ℝ := {x | x > a}

-- Theorem for the union of A and B
theorem union_A_B : A ∪ B = {x | 1 < x ∧ x ≤ 8} := by sorry

-- Theorem for the range of a when A ∩ C is nonempty
theorem range_of_a (a : ℝ) : (A ∩ C a).Nonempty → a < 8 := by sorry

end NUMINAMATH_CALUDE_union_A_B_range_of_a_l3991_399154


namespace NUMINAMATH_CALUDE_swap_digits_formula_l3991_399174

/-- Represents a two-digit number with sum of digits equal to 13 -/
structure TwoDigitNumber where
  units : ℕ
  tens : ℕ
  sum_is_13 : units + tens = 13

/-- The result of swapping digits in a TwoDigitNumber -/
def swap_digits (n : TwoDigitNumber) : ℕ := 10 * n.units + n.tens

theorem swap_digits_formula (n : TwoDigitNumber) : 
  swap_digits n = 9 * n.units + 13 := by
  sorry

end NUMINAMATH_CALUDE_swap_digits_formula_l3991_399174


namespace NUMINAMATH_CALUDE_problem_solution_l3991_399126

def digits : Finset Nat := {0, 1, 2, 3, 4, 5}

def naturalNumbersWithoutRepeats (d : Finset Nat) : Nat :=
  sorry

def fourDigitEvenWithoutRepeats (d : Finset Nat) : Nat :=
  sorry

def fourDigitGreaterThan4023WithoutRepeats (d : Finset Nat) : Nat :=
  sorry

theorem problem_solution (d : Finset Nat) (h : d = digits) :
  naturalNumbersWithoutRepeats d = 1631 ∧
  fourDigitEvenWithoutRepeats d = 156 ∧
  fourDigitGreaterThan4023WithoutRepeats d = 115 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3991_399126


namespace NUMINAMATH_CALUDE_distinct_positions_selection_l3991_399125

theorem distinct_positions_selection (n : ℕ) (k : ℕ) (ways : ℕ) : 
  n = 12 → k = 2 → ways = 132 → ways = n * (n - 1) :=
by sorry

end NUMINAMATH_CALUDE_distinct_positions_selection_l3991_399125


namespace NUMINAMATH_CALUDE_max_min_x_values_l3991_399166

theorem max_min_x_values (x y z : ℝ) 
  (sum_zero : x + y + z = 0)
  (inequality : (x - y)^2 + (y - z)^2 + (z - x)^2 ≤ 2) :
  (∀ w, w = x → w ≤ 2/3) ∧ 
  (∃ v, v = x ∧ v = 2/3) ∧
  (∀ u, u = x → u ≥ -2/3) ∧
  (∃ t, t = x ∧ t = -2/3) :=
sorry

end NUMINAMATH_CALUDE_max_min_x_values_l3991_399166


namespace NUMINAMATH_CALUDE_books_sold_on_monday_l3991_399114

theorem books_sold_on_monday (initial_stock : ℕ) (tuesday_sold : ℕ) (wednesday_sold : ℕ) (thursday_sold : ℕ) (friday_sold : ℕ) (unsold : ℕ) : 
  initial_stock = 800 →
  tuesday_sold = 10 →
  wednesday_sold = 20 →
  thursday_sold = 44 →
  friday_sold = 66 →
  unsold = 600 →
  initial_stock - (tuesday_sold + wednesday_sold + thursday_sold + friday_sold + unsold) = 60 := by
  sorry


end NUMINAMATH_CALUDE_books_sold_on_monday_l3991_399114


namespace NUMINAMATH_CALUDE_simplify_expression_l3991_399175

theorem simplify_expression (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
  (h_sum : x + y + z = 3) :
  (1 / (y^2 + z^2 - x^2)) + (1 / (x^2 + z^2 - y^2)) + (1 / (x^2 + y^2 - z^2)) =
  3 / (-9 + 6*y + 6*z - 2*y*z) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3991_399175


namespace NUMINAMATH_CALUDE_simplify_K_simplify_L_l3991_399133

-- Part (a)
theorem simplify_K (x y : ℝ) (h : x ≥ y^2) :
  Real.sqrt (x + 2*y*Real.sqrt (x - y^2)) + Real.sqrt (x - 2*y*Real.sqrt (x - y^2)) = 
  max (2*abs y) (2*Real.sqrt (x - y^2)) := by sorry

-- Part (b)
theorem simplify_L (x y z : ℝ) (h : x*y + y*z + z*x = 1) :
  (2*x*y*z) / Real.sqrt ((1 + x^2)*(1 + y^2)*(1 + z^2)) = 
  (2*x*y*z) / abs (x + y + z - x*y*z) := by sorry

end NUMINAMATH_CALUDE_simplify_K_simplify_L_l3991_399133


namespace NUMINAMATH_CALUDE_equidistant_point_y_coordinate_l3991_399148

/-- The y-coordinate of the point on the y-axis that is equidistant from A(-3, 0) and B(2, 5) is 2. -/
theorem equidistant_point_y_coordinate : ∃ y : ℝ, 
  ((-3 - 0)^2 + (0 - y)^2 = (2 - 0)^2 + (5 - y)^2) ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_y_coordinate_l3991_399148


namespace NUMINAMATH_CALUDE_gum_cost_1000_l3991_399146

/-- The cost of buying a given number of pieces of gum, considering bulk discount --/
def gumCost (pieces : ℕ) : ℚ :=
  let baseCost := 2 * pieces
  let discountedCost := if pieces > 500 then baseCost * (9/10) else baseCost
  discountedCost / 100

theorem gum_cost_1000 :
  gumCost 1000 = 18 := by sorry

end NUMINAMATH_CALUDE_gum_cost_1000_l3991_399146


namespace NUMINAMATH_CALUDE_initial_cards_l3991_399183

theorem initial_cards (x : ℚ) : 
  (x ≥ 0) → 
  (3 * (1/2) * ((x/3) + (4/3)) = 34) → 
  (x = 64) := by
sorry

end NUMINAMATH_CALUDE_initial_cards_l3991_399183


namespace NUMINAMATH_CALUDE_oliver_william_money_difference_l3991_399143

/-- Calculates the total amount of money given the number of bills of different denominations -/
def calculate_total (twenty_bills ten_bills five_bills : ℕ) : ℕ :=
  20 * twenty_bills + 10 * ten_bills + 5 * five_bills

/-- Represents the problem of comparing Oliver's and William's money -/
theorem oliver_william_money_difference :
  let oliver_total := calculate_total 10 0 3
  let william_total := calculate_total 0 15 4
  oliver_total - william_total = 45 := by sorry

end NUMINAMATH_CALUDE_oliver_william_money_difference_l3991_399143


namespace NUMINAMATH_CALUDE_square_equals_multiplication_l3991_399165

theorem square_equals_multiplication (a : ℝ) : a * a = a ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_square_equals_multiplication_l3991_399165


namespace NUMINAMATH_CALUDE_f_84_value_l3991_399145

def is_increasing (f : ℕ+ → ℕ+) : Prop :=
  ∀ n : ℕ+, f (n + 1) > f n

def multiplicative (f : ℕ+ → ℕ+) : Prop :=
  ∀ m n : ℕ+, f (m * n) = f m * f n

def special_condition (f : ℕ+ → ℕ+) : Prop :=
  ∀ m n : ℕ+, m ≠ n → m ^ (n : ℕ) = n ^ (m : ℕ) → f m = n ∨ f n = m

theorem f_84_value (f : ℕ+ → ℕ+)
  (h_inc : is_increasing f)
  (h_mult : multiplicative f)
  (h_special : special_condition f) :
  f 84 = 1764 := by
  sorry

end NUMINAMATH_CALUDE_f_84_value_l3991_399145


namespace NUMINAMATH_CALUDE_batsman_average_proof_l3991_399139

/-- Calculates the average runs for a batsman given two sets of matches with different averages -/
def calculateAverageRuns (matches1 : ℕ) (average1 : ℚ) (matches2 : ℕ) (average2 : ℚ) : ℚ :=
  ((matches1 : ℚ) * average1 + (matches2 : ℚ) * average2) / ((matches1 + matches2) : ℚ)

/-- Theorem: Given a batsman's performance in two sets of matches, prove the overall average -/
theorem batsman_average_proof (matches1 matches2 : ℕ) (average1 average2 : ℚ) :
  matches1 = 20 ∧ matches2 = 10 ∧ average1 = 30 ∧ average2 = 15 →
  calculateAverageRuns matches1 average1 matches2 average2 = 25 := by
  sorry

#eval calculateAverageRuns 20 30 10 15

end NUMINAMATH_CALUDE_batsman_average_proof_l3991_399139


namespace NUMINAMATH_CALUDE_grinder_purchase_price_l3991_399152

theorem grinder_purchase_price 
  (x : ℝ) -- purchase price of grinder
  (h1 : 0.96 * x + 9200 = x + 8600) -- equation representing the overall transaction
  : x = 15000 := by
  sorry

end NUMINAMATH_CALUDE_grinder_purchase_price_l3991_399152


namespace NUMINAMATH_CALUDE_problem_statement_l3991_399156

theorem problem_statement : 
  (∃ x : ℝ, x - x + 1 ≥ 0) ∧ ¬(∀ a b : ℝ, a^2 < b^2 → a < b) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3991_399156


namespace NUMINAMATH_CALUDE_cubic_root_function_l3991_399173

theorem cubic_root_function (k : ℝ) :
  (∀ x : ℝ, x > 0 → ∃ y : ℝ, y = k * x^(1/3)) →
  (∃ y : ℝ, y = 4 * Real.sqrt 3 ∧ 64^(1/3) * k = y) →
  (∃ y : ℝ, y = 2 * Real.sqrt 3 ∧ 8^(1/3) * k = y) :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_function_l3991_399173


namespace NUMINAMATH_CALUDE_sum_of_digits_l3991_399129

theorem sum_of_digits (a₁ a₂ b c : ℕ) :
  a₁ < 10 → a₂ < 10 → b < 10 → c < 10 →
  100 * (10 * a₁ + a₂) + 10 * b + 7 * c = 2024 →
  a₁ + a₂ + b + c = 5 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_digits_l3991_399129


namespace NUMINAMATH_CALUDE_least_multiple_of_15_greater_than_520_l3991_399196

theorem least_multiple_of_15_greater_than_520 : 
  ∀ n : ℕ, n > 0 ∧ 15 ∣ n ∧ n > 520 → n ≥ 525 := by
  sorry

end NUMINAMATH_CALUDE_least_multiple_of_15_greater_than_520_l3991_399196


namespace NUMINAMATH_CALUDE_max_sum_of_goods_l3991_399157

theorem max_sum_of_goods (a b : ℕ+) : 
  7 * a + 19 * b = 213 →
  ∀ x y : ℕ+, 7 * x + 19 * y = 213 → a + b ≥ x + y →
  a + b = 27 := by
sorry

end NUMINAMATH_CALUDE_max_sum_of_goods_l3991_399157


namespace NUMINAMATH_CALUDE_ring_cost_calculation_l3991_399178

/-- The cost of a single ring given the total sales and necklace price -/
def ring_cost (total_sales necklace_price : ℕ) (num_necklaces num_rings : ℕ) : ℕ :=
  (total_sales - necklace_price * num_necklaces) / num_rings

theorem ring_cost_calculation (total_sales necklace_price : ℕ) 
  (h1 : total_sales = 80)
  (h2 : necklace_price = 12)
  (h3 : ring_cost total_sales necklace_price 4 8 = 4) : 
  ∃ (x : ℕ), x = ring_cost 80 12 4 8 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_ring_cost_calculation_l3991_399178


namespace NUMINAMATH_CALUDE_only_q_is_true_l3991_399108

theorem only_q_is_true (p q m : Prop) 
  (h1 : (p ∨ q ∨ m) ∧ (¬(p ∧ q) ∧ ¬(p ∧ m) ∧ ¬(q ∧ m)))  -- Only one of p, q, and m is true
  (h2 : (p ∨ ¬(p ∨ q) ∨ m) ∧ (¬(p ∧ ¬(p ∨ q)) ∧ ¬(p ∧ m) ∧ ¬(¬(p ∨ q) ∧ m)))  -- Only one judgment is incorrect
  : q := by
sorry


end NUMINAMATH_CALUDE_only_q_is_true_l3991_399108


namespace NUMINAMATH_CALUDE_differential_of_y_l3991_399187

open Real

noncomputable def y (x : ℝ) : ℝ := cos x * log (tan x) - log (tan (x / 2))

theorem differential_of_y (x : ℝ) (h : x ≠ 0) (h' : x ≠ π/2) :
  deriv y x = -sin x * log (tan x) :=
by sorry

end NUMINAMATH_CALUDE_differential_of_y_l3991_399187


namespace NUMINAMATH_CALUDE_gcd_cube_plus_27_and_plus_3_l3991_399184

theorem gcd_cube_plus_27_and_plus_3 (n : ℕ) (h : n > 27) :
  Nat.gcd (n^3 + 27) (n + 3) = n + 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_cube_plus_27_and_plus_3_l3991_399184


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3991_399127

/-- A positive geometric sequence -/
def IsPositiveGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product (a : ℕ → ℝ) :
  IsPositiveGeometricSequence a →
  a 1 * a 19 = 16 →
  a 8 * a 10 * a 12 = 64 := by
    sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l3991_399127


namespace NUMINAMATH_CALUDE_june_earnings_l3991_399131

/-- Represents the number of clovers June picks -/
def total_clovers : ℕ := 200

/-- Represents the percentage of clovers with 3 petals -/
def three_petal_percentage : ℚ := 75 / 100

/-- Represents the percentage of clovers with 2 petals -/
def two_petal_percentage : ℚ := 24 / 100

/-- Represents the percentage of clovers with 4 petals -/
def four_petal_percentage : ℚ := 1 / 100

/-- Represents the payment in cents for each clover -/
def payment_per_clover : ℕ := 1

/-- Theorem stating that June earns 200 cents -/
theorem june_earnings : 
  (total_clovers * payment_per_clover : ℕ) = 200 := by
  sorry

end NUMINAMATH_CALUDE_june_earnings_l3991_399131


namespace NUMINAMATH_CALUDE_forint_bill_solution_exists_l3991_399123

def is_valid_solution (x y z : ℕ) : Prop :=
  10 * x + 5 * y + z = 682 ∧ x = y + z

def is_one_of_solutions (x y z : ℕ) : Prop :=
  (x = 58 ∧ y = 11 ∧ z = 47) ∨
  (x = 54 ∧ y = 22 ∧ z = 32) ∨
  (x = 50 ∧ y = 33 ∧ z = 17) ∨
  (x = 46 ∧ y = 44 ∧ z = 2)

theorem forint_bill_solution_exists :
  ∃ x y z : ℕ, is_valid_solution x y z ∧ is_one_of_solutions x y z := by
  sorry

end NUMINAMATH_CALUDE_forint_bill_solution_exists_l3991_399123


namespace NUMINAMATH_CALUDE_total_rainfall_sum_l3991_399153

/-- The total rainfall recorded over three days equals the sum of individual daily rainfall amounts. -/
theorem total_rainfall_sum (monday tuesday wednesday : Real) 
  (h1 : monday = 0.17)
  (h2 : tuesday = 0.42)
  (h3 : wednesday = 0.08) :
  monday + tuesday + wednesday = 0.67 := by
  sorry

end NUMINAMATH_CALUDE_total_rainfall_sum_l3991_399153


namespace NUMINAMATH_CALUDE_sandwich_combinations_l3991_399169

theorem sandwich_combinations (meat : ℕ) (cheese : ℕ) (bread : ℕ) :
  meat = 12 → cheese = 11 → bread = 5 →
  (meat * (cheese.choose 3) * bread) = 9900 :=
by
  sorry

end NUMINAMATH_CALUDE_sandwich_combinations_l3991_399169


namespace NUMINAMATH_CALUDE_difference_of_squares_l3991_399120

theorem difference_of_squares (n : ℝ) : n^2 - 9 = (n + 3) * (n - 3) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3991_399120


namespace NUMINAMATH_CALUDE_range_of_m_l3991_399110

theorem range_of_m (x : ℝ) (m : ℝ) : 
  (∃ x ∈ Set.Ioo (π/2) π, 2 * Real.sin x ^ 2 - Real.sqrt 3 * Real.sin (2 * x) + m - 1 = 0) →
  m ∈ Set.Ioo (-2) (-1) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3991_399110


namespace NUMINAMATH_CALUDE_smallest_base_for_61_digits_l3991_399193

theorem smallest_base_for_61_digits : ∃ (b : ℕ), b > 1 ∧ 
  (∀ (n : ℕ), n > 1 → n < b → (Nat.log 10 (n^200) + 1 < 61)) ∧ 
  (Nat.log 10 (b^200) + 1 = 61) := by
  sorry

end NUMINAMATH_CALUDE_smallest_base_for_61_digits_l3991_399193


namespace NUMINAMATH_CALUDE_tenth_graders_truth_count_l3991_399179

def is_valid_response_count (n : ℕ) (truth_tellers : ℕ) : Prop :=
  n > 0 ∧ truth_tellers ≤ n ∧ 
  (truth_tellers * (n - 1) + (n - truth_tellers) * truth_tellers = 44) ∧
  (truth_tellers * (n - truth_tellers) + (n - truth_tellers) * (n - 1 - truth_tellers) = 28)

theorem tenth_graders_truth_count :
  ∃ (n : ℕ) (t : ℕ), 
    is_valid_response_count n t ∧ 
    (t * (n - 1) = 16 ∨ t * (n - 1) = 56) := by
  sorry

end NUMINAMATH_CALUDE_tenth_graders_truth_count_l3991_399179


namespace NUMINAMATH_CALUDE_crosswalk_lines_total_l3991_399155

theorem crosswalk_lines_total (num_intersections : ℕ) (crosswalks_per_intersection : ℕ) (lines_per_crosswalk : ℕ) : 
  num_intersections = 5 → 
  crosswalks_per_intersection = 4 → 
  lines_per_crosswalk = 20 → 
  num_intersections * crosswalks_per_intersection * lines_per_crosswalk = 400 :=
by sorry

end NUMINAMATH_CALUDE_crosswalk_lines_total_l3991_399155


namespace NUMINAMATH_CALUDE_triangle_problem_l3991_399182

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if (b² + c² - a²) / cos A = 2 and (a cos B - b cos A) / (a cos B + b cos A) - b / c = 1,
    then bc = 1 and the area of triangle ABC is √3 / 4 -/
theorem triangle_problem (a b c A B C : ℝ) (h1 : (b^2 + c^2 - a^2) / Real.cos A = 2)
    (h2 : (a * Real.cos B - b * Real.cos A) / (a * Real.cos B + b * Real.cos A) - b / c = 1) :
    b * c = 1 ∧ (1/2 : ℝ) * b * c * Real.sin A = Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l3991_399182


namespace NUMINAMATH_CALUDE_instrument_players_fraction_l3991_399115

theorem instrument_players_fraction (total : ℕ) (two_or_more : ℕ) (prob_exactly_one : ℚ) :
  total = 800 →
  two_or_more = 64 →
  prob_exactly_one = 12 / 100 →
  (prob_exactly_one * total + two_or_more : ℚ) / total = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_instrument_players_fraction_l3991_399115


namespace NUMINAMATH_CALUDE_min_value_of_expression_equality_condition_l3991_399171

theorem min_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + 8*b*c)) + (b / Real.sqrt (b^2 + 8*a*c)) + (c / Real.sqrt (c^2 + 8*a*b)) ≥ 1 :=
by sorry

theorem equality_condition (a : ℝ) (ha : a > 0) :
  (a / Real.sqrt (a^2 + 8*a*a)) + (a / Real.sqrt (a^2 + 8*a*a)) + (a / Real.sqrt (a^2 + 8*a*a)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_equality_condition_l3991_399171


namespace NUMINAMATH_CALUDE_correct_num_teams_l3991_399118

/-- The number of teams in a league where each team plays every other team exactly once -/
def num_teams : ℕ := 14

/-- The total number of games played in the league -/
def total_games : ℕ := 91

/-- Theorem stating that the number of teams is correct given the conditions -/
theorem correct_num_teams :
  (num_teams * (num_teams - 1)) / 2 = total_games :=
by sorry

end NUMINAMATH_CALUDE_correct_num_teams_l3991_399118


namespace NUMINAMATH_CALUDE_standard_deviation_best_dispersion_measure_l3991_399163

-- Define the possible measures of central tendency and dispersion
inductive DataMeasure
  | Mode
  | Mean
  | StandardDeviation
  | Range

-- Define a function to determine if a measure reflects dispersion
def reflectsDispersion (measure : DataMeasure) : Prop :=
  match measure with
  | DataMeasure.StandardDeviation => true
  | _ => false

-- Theorem stating that standard deviation is the best measure of dispersion
theorem standard_deviation_best_dispersion_measure :
  ∀ (measure : DataMeasure),
    reflectsDispersion measure ↔ measure = DataMeasure.StandardDeviation :=
by sorry

end NUMINAMATH_CALUDE_standard_deviation_best_dispersion_measure_l3991_399163


namespace NUMINAMATH_CALUDE_range_of_a_l3991_399186

-- Define the sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x | x - a < 0}

-- State the theorem
theorem range_of_a (a : ℝ) : A ⊆ B a → a ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3991_399186


namespace NUMINAMATH_CALUDE_ellipse_condition_iff_l3991_399119

-- Define the condition
def condition (m n : ℝ) : Prop := m > n ∧ n > 0

-- Define what it means for the equation to represent an ellipse with foci on the y-axis
def is_ellipse_with_foci_on_y_axis (m n : ℝ) : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ m = 1 / (a^2) ∧ n = 1 / (b^2)

-- State the theorem
theorem ellipse_condition_iff (m n : ℝ) :
  condition m n ↔ is_ellipse_with_foci_on_y_axis m n := by
  sorry

end NUMINAMATH_CALUDE_ellipse_condition_iff_l3991_399119


namespace NUMINAMATH_CALUDE_line_intercepts_l3991_399142

/-- Given a line with equation 2x - 3y = 6, prove that its x-intercept is 3 and y-intercept is -2 -/
theorem line_intercepts :
  let line : ℝ → ℝ → Prop := λ x y => 2 * x - 3 * y = 6
  ∃ (x y : ℝ), (line x 0 ∧ x = 3) ∧ (line 0 y ∧ y = -2) := by
  sorry

end NUMINAMATH_CALUDE_line_intercepts_l3991_399142


namespace NUMINAMATH_CALUDE_bird_migration_problem_l3991_399180

theorem bird_migration_problem (distance_jim_disney : ℕ) (distance_disney_london : ℕ) (total_distance : ℕ) :
  distance_jim_disney = 50 →
  distance_disney_london = 60 →
  total_distance = 2200 →
  ∃ (num_birds : ℕ), num_birds * (distance_jim_disney + distance_disney_london) = total_distance ∧ num_birds = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_bird_migration_problem_l3991_399180


namespace NUMINAMATH_CALUDE_jacks_distance_l3991_399103

theorem jacks_distance (initial_students : ℕ) (initial_average : ℝ) (new_average : ℝ) :
  initial_students = 20 →
  initial_average = 3 →
  new_average = 3.1 →
  (initial_students + 1) * new_average - initial_students * initial_average = 5.1 :=
by sorry

end NUMINAMATH_CALUDE_jacks_distance_l3991_399103


namespace NUMINAMATH_CALUDE_system_solution_l3991_399149

theorem system_solution (a b c : ℝ) (h : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∃ (x y z : ℝ),
    (z + a*y + a^2*x + a^3 = 0) ∧
    (z + b*y + b^2*x + b^3 = 0) ∧
    (z + c*y + c^2*x + c^3 = 0) ∧
    (x = -(a+b+c)) ∧
    (y = a*b + a*c + b*c) ∧
    (z = -a*b*c) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3991_399149


namespace NUMINAMATH_CALUDE_min_value_of_function_min_value_achievable_l3991_399137

theorem min_value_of_function (x : ℝ) (h : x > 0) : 2 * x + 3 / x ≥ 2 * Real.sqrt 6 := by
  sorry

theorem min_value_achievable : ∃ x : ℝ, x > 0 ∧ 2 * x + 3 / x = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_min_value_achievable_l3991_399137


namespace NUMINAMATH_CALUDE_triangle_side_length_l3991_399199

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- Ensure positive side lengths
  A = π/3 →  -- 60 degrees in radians
  B = π/4 →  -- 45 degrees in radians
  b = Real.sqrt 6 →
  a + b + c = A + B + C →  -- Triangle angle sum theorem
  a / Real.sin A = b / Real.sin B →  -- Sine rule
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3991_399199


namespace NUMINAMATH_CALUDE_same_side_theorem_l3991_399130

/-- The set of values for parameter a where points A and B lie on the same side of the line 2x - y = 5 -/
def same_side_values : Set ℝ :=
  {a : ℝ | a ∈ Set.Ioo (-5/2) (-1/2) ∪ Set.Ioo 0 3}

/-- The equation of point A in the plane -/
def point_A_equation (a x y : ℝ) : Prop :=
  5 * a^2 - 4 * a * y + 8 * x^2 - 4 * x * y + y^2 + 12 * a * x = 0

/-- The equation of the parabola with vertex B -/
def parabola_B_equation (a x y : ℝ) : Prop :=
  a * x^2 - 2 * a^2 * x - a * y + a^3 + 3 = 0

/-- The line equation 2x - y = 5 -/
def line_equation (x y : ℝ) : Prop :=
  2 * x - y = 5

theorem same_side_theorem (a : ℝ) :
  (∃ x y : ℝ, point_A_equation a x y) ∧
  (∃ x y : ℝ, parabola_B_equation a x y) ∧
  (∀ x y : ℝ, point_A_equation a x y → ¬line_equation x y) ∧
  (∀ x y : ℝ, parabola_B_equation a x y → ¬line_equation x y) →
  (a ∈ same_side_values ↔
    (∃ x₁ y₁ x₂ y₂ : ℝ,
      point_A_equation a x₁ y₁ ∧
      parabola_B_equation a x₂ y₂ ∧
      (2 * x₁ - y₁ - 5) * (2 * x₂ - y₂ - 5) > 0)) :=
sorry

end NUMINAMATH_CALUDE_same_side_theorem_l3991_399130


namespace NUMINAMATH_CALUDE_units_digit_of_product_l3991_399140

theorem units_digit_of_product (a b c : ℕ) : 
  (4^503 * 3^401 * 15^402) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_product_l3991_399140


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l3991_399188

/-- A geometric sequence with third term 16 and seventh term 2 has fifth term 8 -/
theorem geometric_sequence_fifth_term (a : ℝ) (r : ℝ) 
  (h1 : a * r^2 = 16)  -- third term is 16
  (h2 : a * r^6 = 2)   -- seventh term is 2
  : a * r^4 = 8 :=     -- fifth term is 8
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l3991_399188


namespace NUMINAMATH_CALUDE_inscribed_triangle_area_bound_l3991_399135

/-- A convex polygon with area 1 -/
structure ConvexPolygon where
  area : ℝ
  isConvex : Bool
  area_eq_one : area = 1
  is_convex : isConvex = true

/-- A triangle inscribed in a convex polygon -/
structure InscribedTriangle (p : ConvexPolygon) where
  area : ℝ
  is_inscribed : Bool

/-- Theorem: Any convex polygon with area 1 contains a triangle with area at least 3/8 -/
theorem inscribed_triangle_area_bound (p : ConvexPolygon) : 
  ∃ (t : InscribedTriangle p), t.area ≥ 3/8 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_triangle_area_bound_l3991_399135


namespace NUMINAMATH_CALUDE_inequality_range_l3991_399138

theorem inequality_range (m : ℝ) : 
  (∀ x y : ℝ, 3 * x^2 + y^2 ≥ m * x * (x + y)) → 
  -6 ≤ m ∧ m ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_range_l3991_399138


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3991_399159

noncomputable def f (x : ℝ) : ℝ := x^3 - x + 3

theorem tangent_line_equation :
  let P : ℝ × ℝ := (1, f 1)
  let m : ℝ := (3 * P.1^2 - 1)  -- Derivative of f at x = 1
  (2 : ℝ) * x - y + 1 = 0 ↔ y - P.2 = m * (x - P.1) := by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3991_399159


namespace NUMINAMATH_CALUDE_power_function_quadrants_l3991_399105

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, f x = x ^ a

-- Define the condition f(1/3) = 9
def satisfiesCondition (f : ℝ → ℝ) : Prop :=
  f (1/3) = 9

-- Define the property of being in first and second quadrants
def isInFirstAndSecondQuadrants (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → f x > 0

-- Theorem statement
theorem power_function_quadrants (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) 
  (h2 : satisfiesCondition f) : 
  isInFirstAndSecondQuadrants f :=
sorry

end NUMINAMATH_CALUDE_power_function_quadrants_l3991_399105


namespace NUMINAMATH_CALUDE_bridget_apples_bridget_apples_proof_l3991_399144

theorem bridget_apples : ℕ → Prop :=
  fun total_apples =>
    ∃ (ann_apples cassie_apples : ℕ),
      -- Bridget gave 4 apples to Tom
      -- She split the remaining apples equally between Ann and Cassie
      ann_apples = cassie_apples ∧
      -- After distribution, she was left with 5 apples
      total_apples = 4 + ann_apples + cassie_apples + 5 ∧
      -- The total number of apples is 13
      total_apples = 13

theorem bridget_apples_proof : bridget_apples 13 := by
  sorry

end NUMINAMATH_CALUDE_bridget_apples_bridget_apples_proof_l3991_399144


namespace NUMINAMATH_CALUDE_abc_inequalities_l3991_399170

theorem abc_inequalities (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (sum_eq_one : a + b + c = 1) : 
  (a * b + b * c + a * c ≤ 1/3) ∧ (1/a + 1/b + 1/c ≥ 9) := by
  sorry

end NUMINAMATH_CALUDE_abc_inequalities_l3991_399170


namespace NUMINAMATH_CALUDE_total_cost_is_87_60_l3991_399192

/-- Calculate the total cost of T-shirts bought by Dave -/
def calculate_total_cost : ℝ :=
  let white_packs := 3
  let blue_packs := 2
  let red_packs := 4
  let green_packs := 1

  let white_price := 12
  let blue_price := 8
  let red_price := 10
  let green_price := 6

  let white_discount := 0.10
  let blue_discount := 0.05
  let red_discount := 0.15
  let green_discount := 0

  let white_cost := white_packs * white_price * (1 - white_discount)
  let blue_cost := blue_packs * blue_price * (1 - blue_discount)
  let red_cost := red_packs * red_price * (1 - red_discount)
  let green_cost := green_packs * green_price * (1 - green_discount)

  white_cost + blue_cost + red_cost + green_cost

/-- The total cost of T-shirts bought by Dave is $87.60 -/
theorem total_cost_is_87_60 : calculate_total_cost = 87.60 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_87_60_l3991_399192


namespace NUMINAMATH_CALUDE_can_mark_any_rational_ratio_l3991_399107

/-- Represents the ability to mark points on a segment -/
structure SegmentMarker where
  /-- Mark a point that divides a segment in half -/
  mark_half : ∀ (a b : ℝ), ∃ (c : ℝ), c = (a + b) / 2
  /-- Mark a point that divides a segment in the ratio n:(n+1) -/
  mark_ratio : ∀ (a b : ℝ) (n : ℕ), ∃ (c : ℝ), (c - a) / (b - c) = n / (n + 1)

/-- Theorem stating that with given marking abilities, any rational ratio can be achieved -/
theorem can_mark_any_rational_ratio (marker : SegmentMarker) :
  ∀ (a b : ℝ) (p q : ℕ), ∃ (c : ℝ), (c - a) / (b - c) = p / q :=
sorry

end NUMINAMATH_CALUDE_can_mark_any_rational_ratio_l3991_399107


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3991_399176

theorem trigonometric_identity (α : Real) 
  (h : Real.sin (π / 6 - α) = 1 / 3) : 
  2 * (Real.cos (π / 6 + α / 2))^2 - 1 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3991_399176


namespace NUMINAMATH_CALUDE_min_teams_in_tournament_l3991_399185

/-- Represents a football team in the tournament -/
structure Team where
  wins : Nat
  draws : Nat
  losses : Nat

/-- Calculates the score of a team -/
def score (t : Team) : Nat := 3 * t.wins + t.draws

/-- Represents a football tournament -/
structure Tournament where
  teams : List Team
  /-- Each team plays against every other team once -/
  matches_played : ∀ t ∈ teams, t.wins + t.draws + t.losses = teams.length - 1
  /-- There exists a team with the highest score -/
  highest_scorer : ∃ t ∈ teams, ∀ t' ∈ teams, t ≠ t' → score t > score t'
  /-- The highest scoring team has the fewest wins -/
  fewest_wins : ∃ t ∈ teams, (∀ t' ∈ teams, score t ≥ score t') ∧ 
                              (∀ t' ∈ teams, t ≠ t' → t.wins < t'.wins)

/-- The minimum number of teams in a valid tournament is 8 -/
theorem min_teams_in_tournament : 
  ∀ t : Tournament, t.teams.length ≥ 8 ∧ 
  (∃ t' : Tournament, t'.teams.length = 8) := by sorry

end NUMINAMATH_CALUDE_min_teams_in_tournament_l3991_399185


namespace NUMINAMATH_CALUDE_area_R_specific_rhombus_l3991_399191

/-- Represents a rhombus ABCD -/
structure Rhombus :=
  (side_length : ℝ)
  (angle_B : ℝ)

/-- Represents the region R inside the rhombus -/
def region_R (r : Rhombus) : Set (ℝ × ℝ) := sorry

/-- The area of region R in the rhombus -/
def area_R (r : Rhombus) : ℝ := sorry

/-- Theorem: The area of region R in a rhombus with side length 3 and angle B = 150° -/
theorem area_R_specific_rhombus :
  let r : Rhombus := { side_length := 3, angle_B := 150 }
  area_R r = (9 * (Real.sqrt 6 - Real.sqrt 2)) / 8 := by sorry

end NUMINAMATH_CALUDE_area_R_specific_rhombus_l3991_399191


namespace NUMINAMATH_CALUDE_gcd_1260_924_l3991_399100

theorem gcd_1260_924 : Nat.gcd 1260 924 = 84 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1260_924_l3991_399100


namespace NUMINAMATH_CALUDE_age_problem_l3991_399113

theorem age_problem (my_age : ℕ) : 
  (∃ (older_brother younger_sister youngest_brother : ℕ),
    -- Ten years ago, my older brother was exactly twice my age
    older_brother = 2 * (my_age - 10) ∧
    -- Ten years ago, my younger sister's age was half of mine
    younger_sister = (my_age - 10) / 2 ∧
    -- Ten years ago, my youngest brother was the same age as my sister
    youngest_brother = younger_sister ∧
    -- In fifteen years, the combined age of the four of us will be 110
    (my_age + 15) + (older_brother + 15) + (younger_sister + 15) + (youngest_brother + 15) = 110) →
  my_age = 16 :=
by sorry

end NUMINAMATH_CALUDE_age_problem_l3991_399113


namespace NUMINAMATH_CALUDE_root_of_cubic_l3991_399164

theorem root_of_cubic (x₁ x₂ x₃ : ℝ) (p q r : ℝ) :
  (∀ x, x^3 + p*x^2 + q*x + r = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) →
  (Real.sqrt 2)^3 - 3*(Real.sqrt 2)^2*(Real.sqrt 2) + 7*(Real.sqrt 2) - 3*(Real.sqrt 2) = 0 :=
by sorry

end NUMINAMATH_CALUDE_root_of_cubic_l3991_399164


namespace NUMINAMATH_CALUDE_complement_union_theorem_l3991_399111

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {2, 3, 4}

theorem complement_union_theorem :
  (U \ A) ∪ (U \ B) = {0, 1, 4} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l3991_399111


namespace NUMINAMATH_CALUDE_factorial_ratio_l3991_399106

theorem factorial_ratio : Nat.factorial 50 / Nat.factorial 48 = 2450 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l3991_399106


namespace NUMINAMATH_CALUDE_trailing_zeros_302_factorial_l3991_399150

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: The number of trailing zeros in 302! is 74 -/
theorem trailing_zeros_302_factorial :
  trailingZeros 302 = 74 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_302_factorial_l3991_399150


namespace NUMINAMATH_CALUDE_solution_replacement_l3991_399117

theorem solution_replacement (initial_conc : ℚ) (replacing_conc : ℚ) (final_conc : ℚ) 
  (h1 : initial_conc = 70/100)
  (h2 : replacing_conc = 25/100)
  (h3 : final_conc = 35/100) :
  ∃ (x : ℚ), x = 7/9 ∧ initial_conc * (1 - x) + replacing_conc * x = final_conc :=
by sorry

end NUMINAMATH_CALUDE_solution_replacement_l3991_399117


namespace NUMINAMATH_CALUDE_cakes_left_is_two_l3991_399195

def cakes_baked_yesterday : ℕ := 3
def cakes_baked_lunch : ℕ := 5
def cakes_sold_dinner : ℕ := 6

def cakes_left : ℕ := cakes_baked_yesterday + cakes_baked_lunch - cakes_sold_dinner

theorem cakes_left_is_two : cakes_left = 2 := by
  sorry

end NUMINAMATH_CALUDE_cakes_left_is_two_l3991_399195


namespace NUMINAMATH_CALUDE_rebecca_earnings_l3991_399132

/-- Rebecca's hair salon earnings calculation --/
theorem rebecca_earnings : 
  let haircut_price : ℕ := 30
  let perm_price : ℕ := 40
  let dye_job_price : ℕ := 60
  let dye_cost : ℕ := 10
  let haircut_count : ℕ := 4
  let perm_count : ℕ := 1
  let dye_job_count : ℕ := 2
  let tips : ℕ := 50
  
  haircut_price * haircut_count + 
  perm_price * perm_count + 
  (dye_job_price - dye_cost) * dye_job_count + 
  tips = 310 :=
by
  sorry


end NUMINAMATH_CALUDE_rebecca_earnings_l3991_399132


namespace NUMINAMATH_CALUDE_optimal_sales_distribution_l3991_399141

/-- Represents the sales and profit model for a company selling robots in two locations --/
structure RobotSales where
  x : ℝ  -- Monthly sales volume in both locations
  production_cost : ℝ := 200
  price_A : ℝ := 500
  price_B : ℝ → ℝ := λ x => 1200 - x
  advert_cost_A : ℝ → ℝ := λ x => 100 * x + 10000
  advert_cost_B : ℝ := 50000
  total_sales : ℝ := 1000

/-- Calculates the profit for location A --/
def profit_A (model : RobotSales) : ℝ :=
  model.x * model.price_A - model.x * model.production_cost - model.advert_cost_A model.x

/-- Calculates the profit for location B --/
def profit_B (model : RobotSales) : ℝ :=
  model.x * model.price_B model.x - model.x * model.production_cost - model.advert_cost_B

/-- Calculates the total profit for both locations --/
def total_profit (model : RobotSales) : ℝ :=
  profit_A model + profit_B model

/-- Theorem stating the optimal sales distribution --/
theorem optimal_sales_distribution (model : RobotSales) :
  ∃ (x_A x_B : ℝ),
    x_A + x_B = model.total_sales ∧
    x_A = 600 ∧
    x_B = 400 ∧
    ∀ (y_A y_B : ℝ),
      y_A + y_B = model.total_sales →
      total_profit { model with x := y_A } + total_profit { model with x := y_B } ≤
      total_profit { model with x := x_A } + total_profit { model with x := x_B } :=
sorry

end NUMINAMATH_CALUDE_optimal_sales_distribution_l3991_399141


namespace NUMINAMATH_CALUDE_train_length_l3991_399124

/-- The length of a train given its speed and time to cross an electric pole -/
theorem train_length (speed_kmh : ℝ) (time_sec : ℝ) (h1 : speed_kmh = 144) (h2 : time_sec = 20) :
  speed_kmh * (1000 / 3600) * time_sec = 800 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3991_399124


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l3991_399112

theorem cylinder_surface_area (r h : ℝ) (base_area : ℝ) : 
  base_area = π * r^2 →
  h = 2 * r →
  2 * base_area + 2 * π * r * h = 384 * π := by
  sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l3991_399112
