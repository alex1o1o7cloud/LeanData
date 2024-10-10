import Mathlib

namespace stock_percent_change_l1461_146147

theorem stock_percent_change (initial_value : ℝ) : 
  let day1_value := initial_value * (1 - 0.1)
  let day2_value := day1_value * (1 + 0.2)
  (day2_value - initial_value) / initial_value = 0.08 := by
sorry

end stock_percent_change_l1461_146147


namespace absolute_value_equation_l1461_146113

theorem absolute_value_equation (x : ℝ) : |x + 3| = |x - 5| → x = 1 := by
  sorry

end absolute_value_equation_l1461_146113


namespace twentieth_number_is_381_l1461_146135

/-- The last number of the nth row in the sequence -/
def last_number (n : ℕ) : ℕ := n^2

/-- The 20th number in the 20th row of the sequence -/
def twentieth_number : ℕ := last_number 19 + 20

theorem twentieth_number_is_381 : twentieth_number = 381 := by
  sorry

end twentieth_number_is_381_l1461_146135


namespace polynomial_expansion_property_l1461_146154

theorem polynomial_expansion_property (a a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2 + x)^4 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  a₂ - a₁ + a₄ - a₃ = -15 := by
sorry

end polynomial_expansion_property_l1461_146154


namespace tangent_line_at_one_condition_holds_iff_l1461_146139

-- Define the function f(x) = 2x³ - 3ax²
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 3 * a * x^2

-- Define the derivative of f
def f_prime (a : ℝ) (x : ℝ) : ℝ := 6 * x^2 - 6 * a * x

theorem tangent_line_at_one (a : ℝ) (h : a = 2) :
  ∃ m b : ℝ, m = -6 ∧ b = 2 ∧
  ∀ x : ℝ, f a x + (f_prime a 1) * (x - 1) = m * x + b :=
sorry

theorem condition_holds_iff (a : ℝ) :
  (∀ x₁ : ℝ, x₁ ∈ Set.Icc 0 2 →
    ∃ x₂ : ℝ, x₂ ∈ Set.Icc 0 1 ∧ f a x₁ ≥ f_prime a x₂) ↔
  a ≤ 3/2 :=
sorry

end tangent_line_at_one_condition_holds_iff_l1461_146139


namespace sara_bouquets_l1461_146136

theorem sara_bouquets (red yellow blue : ℕ) 
  (h_red : red = 42) 
  (h_yellow : yellow = 63) 
  (h_blue : blue = 54) : 
  Nat.gcd red (Nat.gcd yellow blue) = 21 := by
  sorry

end sara_bouquets_l1461_146136


namespace range_of_a_l1461_146179

-- Define proposition p
def p (a : ℝ) : Prop :=
  ∀ m : ℝ, m ∈ Set.Icc (-1) 1 → a^2 - 5*a - 3 ≥ Real.sqrt (m^2 + 8)

-- Define proposition q
def q (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 4*x + a^2 > 0

-- Define the range of a
def range_a : Set ℝ := Set.Icc (-2) (-1) ∪ Set.Ioo 2 6

-- Theorem statement
theorem range_of_a (a : ℝ) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → a ∈ range_a :=
by sorry

end range_of_a_l1461_146179


namespace diagonals_25_sided_polygon_l1461_146105

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

/-- Theorem: A convex polygon with 25 sides has 275 diagonals -/
theorem diagonals_25_sided_polygon :
  num_diagonals 25 = 275 := by sorry

end diagonals_25_sided_polygon_l1461_146105


namespace count_pears_l1461_146157

/-- Given a box of fruits with apples and pears, prove the number of pears. -/
theorem count_pears (total_fruits : ℕ) (apples : ℕ) (pears : ℕ) : 
  total_fruits = 51 → apples = 12 → total_fruits = pears + apples → pears = 39 := by
  sorry

end count_pears_l1461_146157


namespace probability_6_or_7_heads_in_8_flips_l1461_146142

def n : ℕ := 8  -- number of coin flips

-- Define the probability of getting exactly k heads in n flips
def prob_k_heads (k : ℕ) : ℚ :=
  (n.choose k) / 2^n

-- Define the probability of getting exactly 6 or 7 heads in n flips
def prob_6_or_7_heads : ℚ :=
  prob_k_heads 6 + prob_k_heads 7

-- Theorem statement
theorem probability_6_or_7_heads_in_8_flips :
  prob_6_or_7_heads = 9 / 64 := by sorry

end probability_6_or_7_heads_in_8_flips_l1461_146142


namespace exactly_one_black_and_two_red_mutually_exclusive_but_not_complementary_l1461_146188

/-- Represents the color of a ball -/
inductive BallColor
| Red
| Black

/-- Represents the outcome of drawing two balls -/
structure TwoDrawOutcome :=
  (first second : BallColor)

/-- The sample space of all possible outcomes when drawing two balls -/
def sampleSpace : Finset TwoDrawOutcome := sorry

/-- The event of drawing exactly one black ball -/
def exactlyOneBlack (outcome : TwoDrawOutcome) : Prop := sorry

/-- The event of drawing exactly two red balls -/
def exactlyTwoRed (outcome : TwoDrawOutcome) : Prop := sorry

/-- Two events are mutually exclusive if they cannot occur simultaneously -/
def mutuallyExclusive (event1 event2 : TwoDrawOutcome → Prop) : Prop := sorry

/-- Two events are complementary if their union is the entire sample space -/
def complementary (event1 event2 : TwoDrawOutcome → Prop) : Prop := sorry

theorem exactly_one_black_and_two_red_mutually_exclusive_but_not_complementary :
  mutuallyExclusive exactlyOneBlack exactlyTwoRed ∧
  ¬complementary exactlyOneBlack exactlyTwoRed := by sorry

end exactly_one_black_and_two_red_mutually_exclusive_but_not_complementary_l1461_146188


namespace bakery_sales_percentage_l1461_146141

theorem bakery_sales_percentage (cake_percent cookie_percent : ℝ) 
  (h_cake : cake_percent = 42)
  (h_cookie : cookie_percent = 25) :
  100 - (cake_percent + cookie_percent) = 33 := by
  sorry

end bakery_sales_percentage_l1461_146141


namespace class_size_l1461_146193

theorem class_size (n : ℕ) (h1 : n > 0) :
  (∃ (x : ℕ), x > 0 ∧ x = 6 + 7 - 1) →
  3 * n = 33 :=
by
  sorry

end class_size_l1461_146193


namespace alvin_friend_gave_wood_l1461_146103

/-- The number of pieces of wood Alvin needs in total -/
def total_needed : ℕ := 376

/-- The number of pieces of wood Alvin's brother gave him -/
def brother_gave : ℕ := 136

/-- The number of pieces of wood Alvin still needs to gather -/
def still_needed : ℕ := 117

/-- The number of pieces of wood Alvin's friend gave him -/
def friend_gave : ℕ := total_needed - brother_gave - still_needed

theorem alvin_friend_gave_wood : friend_gave = 123 := by
  sorry

end alvin_friend_gave_wood_l1461_146103


namespace c_range_l1461_146184

def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

def solution_set_is_real (f : ℝ → ℝ) : Prop :=
  ∀ x, f x > 1

theorem c_range (c : ℝ) (hc : c > 0) :
  let p := is_increasing (λ x => Real.log ((1 - c) * x - 1) / Real.log 10)
  let q := solution_set_is_real (λ x => x + |x - 2 * c|)
  (p ∨ q) ∧ ¬(p ∧ q) →
  c ∈ Set.Ioo 0 (1/2) ∪ Set.Ici 1 :=
sorry

end c_range_l1461_146184


namespace inequality_not_always_preserved_l1461_146156

theorem inequality_not_always_preserved (a b : ℝ) (h : a < b) :
  ∃ m : ℝ, m^2 * a ≤ m^2 * b :=
sorry

end inequality_not_always_preserved_l1461_146156


namespace multiply_63_57_l1461_146155

theorem multiply_63_57 : 63 * 57 = 3591 := by
  sorry

end multiply_63_57_l1461_146155


namespace complex_number_problem_l1461_146131

theorem complex_number_problem (z : ℂ) 
  (h1 : ∃ (r1 : ℝ), z / (1 + z^2) = r1)
  (h2 : ∃ (r2 : ℝ), z^2 / (1 + z) = r2) :
  z = -1/2 + (Complex.I * Real.sqrt 3)/2 ∨ 
  z = -1/2 - (Complex.I * Real.sqrt 3)/2 :=
sorry

end complex_number_problem_l1461_146131


namespace expression_inequality_l1461_146151

theorem expression_inequality : 
  let x : ℚ := 3 + 1/10 + 4/100
  let y : ℚ := 3 + 5/110
  x ≠ y :=
by
  sorry

end expression_inequality_l1461_146151


namespace work_increase_percentage_l1461_146187

theorem work_increase_percentage (p : ℕ) (W : ℝ) (h : p > 0) :
  let original_work_per_person := W / p
  let remaining_persons := (2 : ℝ) / 3 * p
  let new_work_per_person := W / remaining_persons
  (new_work_per_person - original_work_per_person) / original_work_per_person * 100 = 50 := by
sorry

end work_increase_percentage_l1461_146187


namespace certain_number_problem_l1461_146176

theorem certain_number_problem :
  ∃! x : ℝ,
    (28 + x + 42 + 78 + 104) / 5 = 62 ∧
    (48 + 62 + 98 + 124 + x) / 5 = 78 ∧
    x = 58 := by
  sorry

end certain_number_problem_l1461_146176


namespace shirts_per_pants_l1461_146150

/-- 
Given:
- Mr. Jones has 40 pants.
- The total number of pieces of clothes he owns is 280.
- Mr. Jones has a certain number of shirts for every pair of pants.

Prove that Mr. Jones has 6 shirts for every pair of pants.
-/
theorem shirts_per_pants (num_pants : ℕ) (total_clothes : ℕ) (shirts_per_pants : ℕ) : 
  num_pants = 40 → total_clothes = 280 → shirts_per_pants * num_pants + num_pants = total_clothes → 
  shirts_per_pants = 6 := by
  sorry

end shirts_per_pants_l1461_146150


namespace upstream_speed_is_25_l1461_146104

/-- Represents the speed of a man rowing in different conditions -/
structure RowingSpeed where
  stillWater : ℝ  -- Speed in still water
  downstream : ℝ  -- Speed downstream

/-- Calculates the speed of the man rowing upstream -/
def upstreamSpeed (s : RowingSpeed) : ℝ :=
  2 * s.stillWater - s.downstream

/-- Theorem: Given the conditions, the upstream speed is 25 kmph -/
theorem upstream_speed_is_25 (s : RowingSpeed) 
  (h1 : s.stillWater = 32) 
  (h2 : s.downstream = 39) : 
  upstreamSpeed s = 25 := by
  sorry

#eval upstreamSpeed { stillWater := 32, downstream := 39 }

end upstream_speed_is_25_l1461_146104


namespace sum_of_valid_numbers_mod_1000_l1461_146124

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧
  (∀ d, d ∈ n.digits 10 → d ≠ 0) ∧
  n % 99 = 0 ∧
  150 % (n / 100) = 0 ∧
  168 % (n % 100) = 0

def sum_of_valid_numbers : ℕ := sorry

theorem sum_of_valid_numbers_mod_1000 :
  sum_of_valid_numbers % 1000 = 108 := by sorry

end sum_of_valid_numbers_mod_1000_l1461_146124


namespace product_over_sum_minus_four_l1461_146109

theorem product_over_sum_minus_four :
  (1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9) /
  (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 - 4) = 362880 / 41 := by
  sorry

end product_over_sum_minus_four_l1461_146109


namespace min_guaranteed_meeting_distance_l1461_146100

/-- Represents the state of a player on the train -/
structure PlayerState :=
  (position : Real)
  (facing_forward : Bool)
  (at_front : Bool)
  (at_end : Bool)

/-- Represents the game state -/
structure GameState :=
  (alice : PlayerState)
  (bob : PlayerState)
  (total_distance : Real)

/-- Defines the train length -/
def train_length : Real := 1

/-- Theorem stating the minimum guaranteed meeting distance -/
theorem min_guaranteed_meeting_distance :
  ∀ (initial_state : GameState),
  ∃ (strategy : GameState → GameState),
  ∀ (final_state : GameState),
  (final_state.alice.position = final_state.bob.position) →
  (final_state.total_distance ≤ 1.5) :=
sorry

end min_guaranteed_meeting_distance_l1461_146100


namespace circle_center_transformation_l1461_146153

def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

def translate_down (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (p.1, p.2 - d)

theorem circle_center_transformation :
  let initial_center : ℝ × ℝ := (-2, 6)
  let reflected := reflect_y initial_center
  let final_position := translate_down reflected 8
  final_position = (2, -2) := by sorry

end circle_center_transformation_l1461_146153


namespace max_value_sqrt_inequality_l1461_146122

theorem max_value_sqrt_inequality (x : ℝ) (h1 : 3 ≤ x) (h2 : x ≤ 6) :
  ∃ (k : ℝ), (∀ y : ℝ, y ≥ k → ∃ x : ℝ, 3 ≤ x ∧ x ≤ 6 ∧ Real.sqrt (x - 3) + Real.sqrt (6 - x) ≥ y) ∧
  (∀ z : ℝ, z > k → ¬∃ x : ℝ, 3 ≤ x ∧ x ≤ 6 ∧ Real.sqrt (x - 3) + Real.sqrt (6 - x) ≥ z) ∧
  k = Real.sqrt 6 :=
sorry

end max_value_sqrt_inequality_l1461_146122


namespace riza_son_age_l1461_146177

/-- Represents the age difference between Riza and her son -/
def age_difference : ℕ := 25

/-- Represents the sum of Riza's and her son's current ages -/
def current_age_sum : ℕ := 105

/-- Represents Riza's son's current age -/
def son_age : ℕ := (current_age_sum - age_difference) / 2

theorem riza_son_age : son_age = 40 := by
  sorry

end riza_son_age_l1461_146177


namespace factor_expression_l1461_146162

theorem factor_expression (x : ℝ) : 5 * x * (x + 2) + 9 * (x + 2) = (x + 2) * (5 * x + 9) := by
  sorry

end factor_expression_l1461_146162


namespace probability_two_hearts_one_spade_l1461_146119

/-- The number of cards in a standard deck -/
def standardDeckSize : ℕ := 52

/-- The number of hearts in a standard deck -/
def numberOfHearts : ℕ := 13

/-- The number of spades in a standard deck -/
def numberOfSpades : ℕ := 13

/-- The probability of drawing two hearts followed by a spade from a standard 52-card deck -/
theorem probability_two_hearts_one_spade :
  (numberOfHearts * (numberOfHearts - 1) * numberOfSpades : ℚ) / 
  (standardDeckSize * (standardDeckSize - 1) * (standardDeckSize - 2)) = 78 / 5115 :=
by sorry

end probability_two_hearts_one_spade_l1461_146119


namespace largest_value_l1461_146181

theorem largest_value (a b c d e : ℝ) 
  (ha : a = 15372 + 2/3074)
  (hb : b = 15372 - 2/3074)
  (hc : c = 15372 / (2/3074))
  (hd : d = 15372 * (2/3074))
  (he : e = 15372.3074) :
  c > a ∧ c > b ∧ c > d ∧ c > e :=
by sorry

end largest_value_l1461_146181


namespace duck_cow_problem_l1461_146152

theorem duck_cow_problem (D C : ℕ) : 
  2 * D + 4 * C = 2 * (D + C) + 24 → C = 12 := by
  sorry

end duck_cow_problem_l1461_146152


namespace fifth_triple_is_pythagorean_l1461_146115

/-- A Pythagorean triple is a tuple of three positive integers (a, b, c) such that a² + b² = c² -/
def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * a + b * b = c * c

/-- The fifth group in the sequence of Pythagorean triples -/
def fifth_pythagorean_triple : ℕ × ℕ × ℕ := (11, 60, 61)

theorem fifth_triple_is_pythagorean :
  let (a, b, c) := fifth_pythagorean_triple
  is_pythagorean_triple a b c :=
by sorry

end fifth_triple_is_pythagorean_l1461_146115


namespace tim_total_score_l1461_146120

/-- The score for a single line in the game -/
def single_line_score : ℕ := 1000

/-- The score multiplier for a tetris -/
def tetris_multiplier : ℕ := 8

/-- The number of single lines Tim scored -/
def tim_singles : ℕ := 6

/-- The number of tetrises Tim scored -/
def tim_tetrises : ℕ := 4

/-- Theorem: Tim's total score is 38000 points -/
theorem tim_total_score : 
  tim_singles * single_line_score + tim_tetrises * (tetris_multiplier * single_line_score) = 38000 := by
  sorry

end tim_total_score_l1461_146120


namespace polynomial_remainder_theorem_l1461_146185

theorem polynomial_remainder_theorem (x : ℝ) : 
  (8 * x^3 - 20 * x^2 + 28 * x - 30) % (4 * x - 8) = 10 := by
  sorry

end polynomial_remainder_theorem_l1461_146185


namespace handshakes_theorem_l1461_146143

/-- Calculate the number of handshakes in a single meeting -/
def handshakes_in_meeting (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Calculate the total number of handshakes in two meetings -/
def total_handshakes (first_meeting_attendees second_meeting_attendees overlap : ℕ) : ℕ :=
  handshakes_in_meeting first_meeting_attendees +
  handshakes_in_meeting second_meeting_attendees -
  handshakes_in_meeting overlap

/-- Prove that the total number of handshakes in the two meetings is 41 -/
theorem handshakes_theorem :
  let first_meeting_attendees : ℕ := 7
  let second_meeting_attendees : ℕ := 7
  let overlap : ℕ := 2
  total_handshakes first_meeting_attendees second_meeting_attendees overlap = 41 := by
  sorry

#eval total_handshakes 7 7 2

end handshakes_theorem_l1461_146143


namespace P_in_fourth_quadrant_iff_m_gt_two_l1461_146194

/-- A point P in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def in_fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The point P with coordinates (m, 2-m) -/
def P (m : ℝ) : Point :=
  ⟨m, 2 - m⟩

/-- Theorem stating that for P(m, 2-m) to be in the fourth quadrant, m > 2 -/
theorem P_in_fourth_quadrant_iff_m_gt_two (m : ℝ) :
  in_fourth_quadrant (P m) ↔ m > 2 := by
  sorry


end P_in_fourth_quadrant_iff_m_gt_two_l1461_146194


namespace unique_row_contains_101_l1461_146127

/-- The number of rows in Pascal's Triangle that contain the number 101 -/
def rows_containing_101 : ℕ := 1

/-- 101 is a prime number -/
axiom prime_101 : Nat.Prime 101

/-- A number appears in Pascal's Triangle if it's a binomial coefficient -/
def appears_in_pascals_triangle (n : ℕ) : Prop :=
  ∃ (row k : ℕ), Nat.choose row k = n

theorem unique_row_contains_101 :
  (∃! row : ℕ, appears_in_pascals_triangle 101 ∧ row > 0) ∧
  rows_containing_101 = 1 :=
sorry

end unique_row_contains_101_l1461_146127


namespace students_with_B_in_donovans_class_l1461_146130

theorem students_with_B_in_donovans_class 
  (christopher_total : ℕ) 
  (christopher_B : ℕ) 
  (donovan_total : ℕ) 
  (h1 : christopher_total = 20) 
  (h2 : christopher_B = 12) 
  (h3 : donovan_total = 30) 
  (h4 : (christopher_B : ℚ) / christopher_total = (donovan_B : ℚ) / donovan_total) :
  donovan_B = 18 :=
by
  sorry

end students_with_B_in_donovans_class_l1461_146130


namespace system_solutions_l1461_146125

/-- The polynomial f(t) = t³ - 4t² - 16t + 60 -/
def f (t : ℤ) : ℤ := t^3 - 4*t^2 - 16*t + 60

/-- The system of equations -/
def system (x y z : ℤ) : Prop :=
  f x = y ∧ f y = z ∧ f z = x

/-- The theorem stating the only integer solutions to the system -/
theorem system_solutions :
  ∀ x y z : ℤ, system x y z ↔ (x = 3 ∧ y = 3 ∧ z = 3) ∨ 
                               (x = 5 ∧ y = 5 ∧ z = 5) ∨ 
                               (x = -4 ∧ y = -4 ∧ z = -4) :=
sorry

end system_solutions_l1461_146125


namespace complex_equation_solution_l1461_146116

theorem complex_equation_solution (z : ℂ) (h : z * (1 - Complex.I) = 1 + Complex.I) : z = Complex.I := by
  sorry

end complex_equation_solution_l1461_146116


namespace binomial_unique_parameters_l1461_146171

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  hp : 0 ≤ p ∧ p ≤ 1

/-- The expectation of a binomial random variable -/
def expectation (X : BinomialRV) : ℝ := X.n * X.p

/-- The variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

/-- Theorem: For a binomial random variable X with E(X) = 1.6 and D(X) = 1.28, n = 8 and p = 0.2 -/
theorem binomial_unique_parameters :
  ∀ X : BinomialRV,
  expectation X = 1.6 →
  variance X = 1.28 →
  X.n = 8 ∧ X.p = 0.2 := by
  sorry

end binomial_unique_parameters_l1461_146171


namespace polynomial_value_at_three_l1461_146111

/-- Polynomial of degree 5 -/
def P (a b c d : ℝ) (x : ℝ) : ℝ := a * x^5 + b * x^3 + c * x + d

theorem polynomial_value_at_three
  (a b c d : ℝ)
  (h1 : P a b c d 0 = -5)
  (h2 : P a b c d (-3) = 7) :
  P a b c d 3 = -17 := by
  sorry

end polynomial_value_at_three_l1461_146111


namespace supplementary_angles_ratio_l1461_146133

theorem supplementary_angles_ratio (a b : ℝ) : 
  a + b = 180 → -- angles are supplementary
  a / b = 5 / 3 → -- ratio of angles is 5:3
  b = 67.5 -- smaller angle is 67.5°
  := by sorry

end supplementary_angles_ratio_l1461_146133


namespace office_work_distribution_l1461_146149

theorem office_work_distribution (P : ℕ) : P > 0 → (6 / 7 : ℚ) * P * (6 / 5 : ℚ) = P → P ≥ 35 := by
  sorry

end office_work_distribution_l1461_146149


namespace cube_volume_from_face_perimeter_l1461_146165

theorem cube_volume_from_face_perimeter (face_perimeter : ℝ) (h : face_perimeter = 40) :
  let side_length := face_perimeter / 4
  let volume := side_length ^ 3
  volume = 1000 := by sorry

end cube_volume_from_face_perimeter_l1461_146165


namespace solution_set_for_a_equals_one_a_value_for_minimum_four_l1461_146195

def f (a : ℝ) (x : ℝ) : ℝ := |2*x - 1| + |a*x - 5|

theorem solution_set_for_a_equals_one :
  {x : ℝ | f 1 x ≥ 9} = {x : ℝ | x ≤ -1 ∨ x > 5} := by sorry

theorem a_value_for_minimum_four :
  ∀ a : ℝ, 0 < a → a < 5 → (∃ m : ℝ, ∀ x : ℝ, f a x ≥ m ∧ (∃ y : ℝ, f a y = m) ∧ m = 4) → a = 2 := by sorry

end solution_set_for_a_equals_one_a_value_for_minimum_four_l1461_146195


namespace equal_area_rectangles_l1461_146129

/-- Given two rectangles of equal area, where one has dimensions 8 by 45 and the other has width 24,
    prove that the length of the second rectangle is 15. -/
theorem equal_area_rectangles (area : ℝ) (length₁ width₁ width₂ : ℝ) 
  (h₁ : area = length₁ * width₁)
  (h₂ : length₁ = 8)
  (h₃ : width₁ = 45)
  (h₄ : width₂ = 24) :
  area / width₂ = 15 := by
  sorry

end equal_area_rectangles_l1461_146129


namespace recurrence_sequence_has_composite_l1461_146126

/-- A sequence of positive integers satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → (a (n + 1) = 2 * a n + 1 ∨ a (n + 1) = 2 * a n - 1)

/-- A number is composite if it's greater than 1 and not prime -/
def IsComposite (n : ℕ) : Prop :=
  n > 1 ∧ ¬ Nat.Prime n

/-- The main theorem stating that any sequence satisfying the recurrence relation contains a composite number -/
theorem recurrence_sequence_has_composite
  (a : ℕ → ℕ)
  (h_seq : RecurrenceSequence a)
  (h_positive : ∀ n, a n > 0)
  (h_nonconstant : ∃ m n, m ≠ n ∧ a m ≠ a n) :
  ∃ k, IsComposite (a k) :=
sorry

end recurrence_sequence_has_composite_l1461_146126


namespace rotate_point_D_l1461_146198

def rotate90Clockwise (x y : ℝ) : ℝ × ℝ := (y, -x)

theorem rotate_point_D : 
  let D : ℝ × ℝ := (-3, 2)
  rotate90Clockwise D.1 D.2 = (2, -3) := by sorry

end rotate_point_D_l1461_146198


namespace shaniqua_haircut_price_l1461_146108

/-- The amount Shaniqua makes for each haircut -/
def haircut_price : ℚ := sorry

/-- The amount Shaniqua makes for each style -/
def style_price : ℚ := 25

/-- The total number of haircuts Shaniqua gave -/
def num_haircuts : ℕ := 8

/-- The total number of styles Shaniqua gave -/
def num_styles : ℕ := 5

/-- The total amount Shaniqua made -/
def total_amount : ℚ := 221

theorem shaniqua_haircut_price : 
  haircut_price * num_haircuts + style_price * num_styles = total_amount ∧ 
  haircut_price = 12 := by sorry

end shaniqua_haircut_price_l1461_146108


namespace not_a_implies_condition_l1461_146163

/-- Represents a student in the course -/
structure Student :=
  (name : String)

/-- Represents the exam result for a student -/
structure ExamResult :=
  (student : Student)
  (allMultipleChoiceCorrect : Bool)
  (essayScore : ℝ)
  (receivedA : Bool)

/-- The professor's grading policy -/
axiom grading_policy : 
  ∀ (result : ExamResult), 
    result.allMultipleChoiceCorrect ∧ result.essayScore ≥ 80 → result.receivedA

/-- The theorem to be proved -/
theorem not_a_implies_condition (result : ExamResult) : 
  ¬result.receivedA → ¬result.allMultipleChoiceCorrect ∨ result.essayScore < 80 :=
sorry

end not_a_implies_condition_l1461_146163


namespace allocation_schemes_eq_ten_l1461_146128

/-- Represents the number of classes. -/
def num_classes : ℕ := 3

/-- Represents the total number of spots to be allocated. -/
def total_spots : ℕ := 6

/-- Represents the minimum number of spots each class must receive. -/
def min_spots_per_class : ℕ := 1

/-- A function that calculates the number of ways to allocate spots among classes. -/
def allocation_schemes (n c m : ℕ) : ℕ := sorry

/-- Theorem stating that the number of allocation schemes is 10. -/
theorem allocation_schemes_eq_ten : 
  allocation_schemes total_spots num_classes min_spots_per_class = 10 := by sorry

end allocation_schemes_eq_ten_l1461_146128


namespace sugar_salt_price_l1461_146164

/-- Given the price of 2 kg sugar and 5 kg salt, and the price of 1 kg sugar,
    prove the price of 3 kg sugar and 1 kg salt. -/
theorem sugar_salt_price
  (total_price : ℝ)
  (sugar_price : ℝ)
  (h1 : total_price = 5.5)
  (h2 : sugar_price = 1.5)
  (h3 : 2 * sugar_price + 5 * ((total_price - 2 * sugar_price) / 5) = total_price) :
  3 * sugar_price + ((total_price - 2 * sugar_price) / 5) = 5 :=
by sorry

end sugar_salt_price_l1461_146164


namespace smallest_k_no_real_roots_four_is_smallest_k_l1461_146144

/-- The quadratic equation 3x(kx-5)-x^2+7=0 has no real roots when k ≥ 4 -/
theorem smallest_k_no_real_roots : 
  ∀ k : ℤ, (∀ x : ℝ, 3 * x * (k * x - 5) - x^2 + 7 ≠ 0) ↔ k ≥ 4 :=
by sorry

/-- 4 is the smallest integer k for which 3x(kx-5)-x^2+7=0 has no real roots -/
theorem four_is_smallest_k : 
  ∀ k : ℤ, k < 4 → ∃ x : ℝ, 3 * x * (k * x - 5) - x^2 + 7 = 0 :=
by sorry

end smallest_k_no_real_roots_four_is_smallest_k_l1461_146144


namespace temperature_altitude_relationship_l1461_146145

/-- Given that the ground temperature is 20°C and the temperature decreases by 6°C
    for every 1000m increase in altitude, prove that the functional relationship
    between temperature t(°C) and altitude h(m) is t = -0.006h + 20. -/
theorem temperature_altitude_relationship (h : ℝ) :
  let ground_temp : ℝ := 20
  let temp_decrease_per_km : ℝ := 6
  let altitude_increase : ℝ := 1000
  let t : ℝ → ℝ := fun h => -((temp_decrease_per_km / altitude_increase) * h) + ground_temp
  t h = -0.006 * h + 20 := by
  sorry

end temperature_altitude_relationship_l1461_146145


namespace cross_section_area_theorem_l1461_146134

/-- Regular hexagonal pyramid -/
structure HexagonalPyramid where
  base_side : ℝ
  height : ℝ

/-- Cutting plane for the pyramid -/
structure CuttingPlane where
  distance_from_apex : ℝ

/-- The area of the cross-section of a regular hexagonal pyramid -/
noncomputable def cross_section_area (p : HexagonalPyramid) (c : CuttingPlane) : ℝ :=
  sorry

/-- Theorem stating the area of the cross-section for the given conditions -/
theorem cross_section_area_theorem (p : HexagonalPyramid) (c : CuttingPlane) :
  p.base_side = 2 →
  c.distance_from_apex = 1 →
  cross_section_area p c = 34 * Real.sqrt 3 / 35 :=
sorry

end cross_section_area_theorem_l1461_146134


namespace ten_men_joined_l1461_146137

/-- Represents the number of men who joined the camp -/
def men_joined : ℕ := sorry

/-- The initial number of men in the camp -/
def initial_men : ℕ := 10

/-- The initial duration of the food supply in days -/
def initial_duration : ℕ := 50

/-- The new duration of the food supply after more men join -/
def new_duration : ℕ := 25

/-- The total amount of food available in man-days -/
def total_food : ℕ := initial_men * initial_duration

/-- Theorem stating that 10 men joined the camp -/
theorem ten_men_joined : men_joined = 10 := by
  sorry

end ten_men_joined_l1461_146137


namespace miriam_pushups_l1461_146169

/-- Calculates the number of push-ups Miriam does on Friday given her schedule for the week. -/
theorem miriam_pushups (monday : ℕ) (tuesday : ℕ) (wednesday : ℕ) (thursday : ℕ) 
  (h1 : monday = 5)
  (h2 : tuesday = 7)
  (h3 : wednesday = 2 * tuesday)
  (h4 : thursday = (monday + tuesday + wednesday) / 2)
  : monday + tuesday + wednesday + thursday = 39 := by
  sorry

end miriam_pushups_l1461_146169


namespace age_difference_proof_l1461_146166

theorem age_difference_proof (p m n : ℕ) 
  (h1 : 5 * p = 3 * m)  -- p:m = 3:5
  (h2 : 5 * m = 3 * n)  -- m:n = 3:5
  (h3 : p + m + n = 245) : 
  n - p = 80 := by
sorry

end age_difference_proof_l1461_146166


namespace complete_square_sum_l1461_146175

theorem complete_square_sum (b c : ℤ) : 
  (∀ x : ℝ, x^2 - 10*x + 15 = 0 ↔ (x + b)^2 = c) → 
  b + c = 5 := by
sorry

end complete_square_sum_l1461_146175


namespace fruits_in_good_condition_l1461_146110

theorem fruits_in_good_condition 
  (oranges : ℕ) 
  (bananas : ℕ) 
  (rotten_oranges_percent : ℚ) 
  (rotten_bananas_percent : ℚ) 
  (h1 : oranges = 600) 
  (h2 : bananas = 400) 
  (h3 : rotten_oranges_percent = 15/100) 
  (h4 : rotten_bananas_percent = 5/100) : 
  (oranges + bananas - (oranges * rotten_oranges_percent + bananas * rotten_bananas_percent)) / (oranges + bananas) = 89/100 := by
sorry

end fruits_in_good_condition_l1461_146110


namespace arithmetic_sequence_fourth_term_l1461_146160

/-- Given an arithmetic sequence where the sum of the third and fifth terms is 10,
    prove that the fourth term is 5. -/
theorem arithmetic_sequence_fourth_term (b y : ℝ) 
  (h : b + (b + 2*y) = 10) : b + y = 5 := by
  sorry

end arithmetic_sequence_fourth_term_l1461_146160


namespace smallest_b_value_l1461_146174

def is_factor (m n : ℕ) : Prop := n % m = 0

theorem smallest_b_value (a b : ℕ) : 
  a = 363 → 
  is_factor 112 (a * 43 * 62 * b) → 
  is_factor 33 (a * 43 * 62 * b) → 
  b ≥ 56 ∧ is_factor 112 (a * 43 * 62 * 56) ∧ is_factor 33 (a * 43 * 62 * 56) :=
sorry

end smallest_b_value_l1461_146174


namespace H_range_l1461_146190

def H (x : ℝ) : ℝ := |x + 2| - |x - 4| + 3

theorem H_range : 
  (∀ x, 5 ≤ H x ∧ H x ≤ 9) ∧ 
  (∃ x, H x = 9) ∧
  (∀ ε > 0, ∃ x, H x < 5 + ε) :=
sorry

end H_range_l1461_146190


namespace inequality_proof_l1461_146138

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (b - c)^2 * (b + c) / a + (c - a)^2 * (c + a) / b + (a - b)^2 * (a + b) / c ≥ 
  2 * (a^2 + b^2 + c^2 - a*b - b*c - c*a) := by
  sorry

end inequality_proof_l1461_146138


namespace bird_photo_combinations_l1461_146106

/-- Represents the number of pairs of birds -/
def num_pairs : ℕ := 5

/-- Calculates the number of ways to photograph birds with alternating genders -/
def photo_combinations (n : ℕ) : ℕ :=
  let female_choices := List.range n
  let male_choices := List.range (n - 1)
  (female_choices.foldl (· * ·) 1) * (male_choices.foldl (· * ·) 1)

/-- Theorem stating the number of ways to photograph the birds -/
theorem bird_photo_combinations :
  photo_combinations num_pairs = 2880 := by
  sorry

end bird_photo_combinations_l1461_146106


namespace math_score_calculation_math_score_is_83_l1461_146182

theorem math_score_calculation (average_three : ℝ) (average_decrease : ℝ) : ℝ :=
  let total_three := 3 * average_three
  let new_average := average_three - average_decrease
  let total_four := 4 * new_average
  total_four - total_three

theorem math_score_is_83 :
  math_score_calculation 95 3 = 83 := by
  sorry

end math_score_calculation_math_score_is_83_l1461_146182


namespace f_minus_two_range_l1461_146117

def f (a b x : ℝ) : ℝ := a * x^2 + b * x

theorem f_minus_two_range (a b : ℝ) :
  (1 ≤ f a b (-1) ∧ f a b (-1) ≤ 2) →
  (2 ≤ f a b 1 ∧ f a b 1 ≤ 4) →
  ∃ (y : ℝ), y = f a b (-2) ∧ 3 ≤ y ∧ y ≤ 12 :=
by sorry

end f_minus_two_range_l1461_146117


namespace list_property_l1461_146159

theorem list_property (list : List ℝ) (n : ℝ) : 
  list.Nodup →
  n ∈ list →
  n = 4 * ((list.sum - n) / (list.length - 1)) →
  n = (1 / 6) * list.sum →
  list.length = 21 := by
  sorry

end list_property_l1461_146159


namespace sum_of_exterior_angles_constant_l1461_146173

/-- A convex polygon with n sides, where n ≥ 3 -/
structure ConvexPolygon where
  n : ℕ
  sides_ge_three : n ≥ 3

/-- The sum of exterior angles of a convex polygon -/
def sum_of_exterior_angles (p : ConvexPolygon) : ℝ := sorry

/-- Theorem: The sum of exterior angles of any convex polygon is 360° -/
theorem sum_of_exterior_angles_constant (p : ConvexPolygon) :
  sum_of_exterior_angles p = 360 := by sorry

end sum_of_exterior_angles_constant_l1461_146173


namespace quadratic_equation_solution_l1461_146183

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁^2 - 2*x₁ - 8 = 0) ∧ 
  (x₂^2 - 2*x₂ - 8 = 0) ∧ 
  x₁ = 4 ∧ 
  x₂ = -2 := by
  sorry

end quadratic_equation_solution_l1461_146183


namespace monotonic_decreasing_interval_l1461_146178

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 5

-- State the theorem
theorem monotonic_decreasing_interval :
  ∀ x : ℝ, 0 < x ∧ x < 2 ↔ ∀ y : ℝ, 0 < y ∧ y < x → f y > f x :=
sorry

end monotonic_decreasing_interval_l1461_146178


namespace consecutive_integer_averages_l1461_146192

theorem consecutive_integer_averages (a : ℤ) (c : ℚ) : 
  (a > 0) →
  (c = (7 * a + 21) / 7) →
  ((7 * c + 21) / 7 : ℚ) = a + 6 := by
  sorry

end consecutive_integer_averages_l1461_146192


namespace planes_distance_l1461_146140

/-- The total distance traveled by two planes moving towards each other -/
def total_distance (speed : ℝ) (time : ℝ) : ℝ :=
  2 * speed * time

/-- Theorem: The total distance traveled by two planes moving towards each other
    at 283 miles per hour for 2 hours is 1132 miles. -/
theorem planes_distance :
  total_distance 283 2 = 1132 :=
by sorry

end planes_distance_l1461_146140


namespace apple_pear_equivalence_l1461_146161

theorem apple_pear_equivalence : 
  ∀ (apple_value pear_value : ℚ),
  (3/4 * 16 : ℚ) * apple_value = 10 * pear_value →
  (2/5 * 20 : ℚ) * apple_value = (20/3 : ℚ) * pear_value := by
sorry

end apple_pear_equivalence_l1461_146161


namespace opposite_of_negative_seven_thirds_l1461_146118

theorem opposite_of_negative_seven_thirds :
  ∃ y : ℚ, -7/3 + y = 0 ∧ y = 7/3 := by
sorry

end opposite_of_negative_seven_thirds_l1461_146118


namespace equation_holds_iff_b_equals_c_l1461_146197

theorem equation_holds_iff_b_equals_c (a b c : ℕ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_less_than_10 : a < 10 ∧ b < 10 ∧ c < 10) :
  (10 * a + b + 1) * (10 * a + c) = 100 * a^2 + 100 * a + b + c ↔ b = c :=
sorry

end equation_holds_iff_b_equals_c_l1461_146197


namespace line_passes_through_intersection_and_perpendicular_l1461_146107

-- Define the lines
def line1 (x y : ℝ) : Prop := 2 * x + y + 2 = 0
def line2 (x y : ℝ) : Prop := 3 * x + 4 * y - 2 = 0
def line3 (x y : ℝ) : Prop := 3 * x - 2 * y + 4 = 0
def line4 (x y : ℝ) : Prop := 2 * x + 3 * y - 2 = 0

-- Define the intersection point
def intersection_point (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Define perpendicularity
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Theorem statement
theorem line_passes_through_intersection_and_perpendicular :
  ∃ (x y : ℝ),
    intersection_point x y ∧
    line4 x y ∧
    perpendicular 
      ((3 : ℝ) / 2) -- slope of line3
      (-(2 : ℝ) / 3) -- slope of line4
  := by sorry

end line_passes_through_intersection_and_perpendicular_l1461_146107


namespace minimal_fraction_difference_l1461_146121

theorem minimal_fraction_difference (p q : ℕ+) : 
  (3 : ℚ) / 5 < p / q ∧ p / q < (2 : ℚ) / 3 ∧ 
  (∀ (p' q' : ℕ+), (3 : ℚ) / 5 < p' / q' ∧ p' / q' < (2 : ℚ) / 3 → q' ≥ q) →
  q - p = 11 := by
  sorry

end minimal_fraction_difference_l1461_146121


namespace function_rate_comparison_l1461_146189

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - x
def g (x : ℝ) : ℝ := x^2 + x

-- Define the derivatives of f and g
def f' (x : ℝ) : ℝ := 2*x - 1
def g' (x : ℝ) : ℝ := 2*x + 1

theorem function_rate_comparison :
  (∃ x : ℝ, f' x = 2 * g' x ∧ x = -3/2) ∧
  (¬ ∃ x : ℝ, f' x = g' x) := by
  sorry


end function_rate_comparison_l1461_146189


namespace root_condition_implies_k_range_l1461_146148

theorem root_condition_implies_k_range (n : ℕ) (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    2*n - 1 < x₁ ∧ x₁ ≤ 2*n + 1 ∧
    2*n - 1 < x₂ ∧ x₂ ≤ 2*n + 1 ∧
    |x₁ - 2*n| = k * Real.sqrt x₁ ∧
    |x₂ - 2*n| = k * Real.sqrt x₂) →
  (0 < k ∧ k ≤ 1 / Real.sqrt (2*n + 1)) :=
by sorry

end root_condition_implies_k_range_l1461_146148


namespace inner_rectangle_length_l1461_146114

/-- Represents the dimensions of a rectangular region -/
structure RectDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle given its dimensions -/
def rectangleArea (dim : RectDimensions) : ℝ :=
  dim.length * dim.width

/-- Represents the three regions of the rug -/
structure RugRegions where
  inner : RectDimensions
  middle : RectDimensions
  outer : RectDimensions

/-- Theorem stating that the length of the inner rectangle is 4 feet -/
theorem inner_rectangle_length (rug : RugRegions) : rug.inner.length = 4 :=
  by
  -- Assuming the following conditions:
  have inner_width : rug.inner.width = 2 := by sorry
  have middle_surround : rug.middle.length = rug.inner.length + 4 ∧ 
                         rug.middle.width = rug.inner.width + 4 := by sorry
  have outer_surround : rug.outer.length = rug.middle.length + 4 ∧ 
                        rug.outer.width = rug.middle.width + 4 := by sorry
  have areas_arithmetic_progression : 
    (rectangleArea rug.middle - rectangleArea rug.inner) = 
    (rectangleArea rug.outer - rectangleArea rug.middle) := by sorry
  
  sorry -- Proof goes here

end inner_rectangle_length_l1461_146114


namespace sugar_amount_l1461_146196

/-- Represents the amounts of ingredients in a bakery storage room. -/
structure BakeryStorage where
  sugar : ℝ
  flour : ℝ
  bakingSoda : ℝ

/-- Checks if the given storage satisfies the bakery's ratios. -/
def satisfiesRatios (storage : BakeryStorage) : Prop :=
  storage.sugar / storage.flour = 5 / 4 ∧
  storage.flour / storage.bakingSoda = 10 / 1

/-- Checks if adding 60 pounds of baking soda changes the ratio as specified. -/
def satisfiesNewRatio (storage : BakeryStorage) : Prop :=
  storage.flour / (storage.bakingSoda + 60) = 8 / 1

/-- Theorem: Given the conditions, the amount of sugar in the storage is 3000 pounds. -/
theorem sugar_amount (storage : BakeryStorage) 
  (h1 : satisfiesRatios storage) 
  (h2 : satisfiesNewRatio storage) : 
  storage.sugar = 3000 := by
sorry

end sugar_amount_l1461_146196


namespace nunzio_pizza_consumption_l1461_146191

/-- Represents the number of pieces in a whole pizza -/
def pieces_per_pizza : ℕ := 8

/-- Represents the number of pizzas Nunzio eats in the given period -/
def total_pizzas : ℕ := 27

/-- Represents the number of days in the given period -/
def total_days : ℕ := 72

/-- Calculates the number of pizza pieces Nunzio eats per day -/
def pieces_per_day : ℕ := (total_pizzas * pieces_per_pizza) / total_days

/-- Theorem stating that Nunzio eats 3 pieces of pizza per day -/
theorem nunzio_pizza_consumption : pieces_per_day = 3 := by
  sorry

end nunzio_pizza_consumption_l1461_146191


namespace arithmetic_sequence_properties_l1461_146101

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

def sum_arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := n * a₁ + (n * (n - 1) / 2) * d

theorem arithmetic_sequence_properties 
  (a₁ : ℤ) 
  (d : ℤ) 
  (h1 : a₁ = 23)
  (h2 : ∀ n : ℕ, n ≤ 6 → arithmetic_sequence a₁ d n > 0)
  (h3 : arithmetic_sequence a₁ d 7 < 0) :
  (d = -4) ∧ 
  (∃ n : ℕ, sum_arithmetic_sequence a₁ d n = 78 ∧ 
    ∀ m : ℕ, sum_arithmetic_sequence a₁ d m ≤ 78) ∧
  (∃ n : ℕ, n = 12 ∧ sum_arithmetic_sequence a₁ d n > 0 ∧ 
    ∀ m : ℕ, m > 12 → sum_arithmetic_sequence a₁ d m ≤ 0) :=
sorry

end arithmetic_sequence_properties_l1461_146101


namespace seventh_term_of_geometric_sequence_l1461_146172

/-- Given a geometric sequence where the first term is 3 and the second term is 6,
    prove that the seventh term is 192. -/
theorem seventh_term_of_geometric_sequence :
  ∀ (a : ℕ → ℝ), 
    a 1 = 3 →
    a 2 = 6 →
    (∀ n : ℕ, a (n + 1) = a n * (a 2 / a 1)) →
    a 7 = 192 := by
  sorry

end seventh_term_of_geometric_sequence_l1461_146172


namespace q_gt_one_not_sufficient_nor_necessary_l1461_146168

/-- A geometric sequence with common ratio q -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

/-- An increasing sequence -/
def IncreasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

/-- Theorem: "q > 1" is neither sufficient nor necessary for a geometric sequence to be increasing -/
theorem q_gt_one_not_sufficient_nor_necessary (a : ℕ → ℝ) (q : ℝ) :
  (∃ a q, GeometricSequence a q ∧ q > 1 ∧ ¬IncreasingSequence a) ∧
  (∃ a q, GeometricSequence a q ∧ IncreasingSequence a ∧ ¬(q > 1)) :=
sorry

end q_gt_one_not_sufficient_nor_necessary_l1461_146168


namespace parallel_vectors_x_value_l1461_146132

/-- Two-dimensional vector -/
structure Vec2D where
  x : ℝ
  y : ℝ

/-- Parallelism of two vectors -/
def parallel (v w : Vec2D) : Prop :=
  ∃ (μ : ℝ), μ ≠ 0 ∧ v.x = μ * w.x ∧ v.y = μ * w.y

theorem parallel_vectors_x_value :
  ∀ (x : ℝ),
  let a : Vec2D := ⟨x, 1⟩
  let b : Vec2D := ⟨3, 6⟩
  parallel b a → x = 1/2 := by
  sorry

end parallel_vectors_x_value_l1461_146132


namespace inequality_proof_l1461_146170

theorem inequality_proof (x y z : ℝ) 
  (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) 
  (hsum : x + y + z = 1) : 
  (2 * (x^2 + y^2 + z^2) + 9*x*y*z ≥ 1) ∧ 
  (x*y + y*z + z*x - 3*x*y*z ≤ 1/4) := by
  sorry

end inequality_proof_l1461_146170


namespace quadratic_function_satisfies_conditions_l1461_146112

-- Define the quadratic function
def f (x : ℝ) := -4 * x^2 + 4 * x + 7

-- State the theorem
theorem quadratic_function_satisfies_conditions :
  (f 2 = -1) ∧ 
  (f (-1) = -1) ∧ 
  (∀ x : ℝ, f x ≤ 8) ∧
  (∃ x : ℝ, f x = 8) := by
  sorry

end quadratic_function_satisfies_conditions_l1461_146112


namespace instantaneous_velocity_at_3_seconds_l1461_146158

-- Define the equation of motion
def s (t : ℝ) : ℝ := -t + t^2

-- Define the velocity function
def v (t : ℝ) : ℝ := (-1) + 2*t

-- Theorem statement
theorem instantaneous_velocity_at_3_seconds :
  v 3 = 5 := by sorry

end instantaneous_velocity_at_3_seconds_l1461_146158


namespace pie_slices_sold_today_l1461_146180

theorem pie_slices_sold_today (total : ℕ) (yesterday : ℕ) (today : ℕ) 
  (h1 : total = 7) 
  (h2 : yesterday = 5) 
  (h3 : total = yesterday + today) : 
  today = 2 := by
  sorry

end pie_slices_sold_today_l1461_146180


namespace cube_sum_of_roots_l1461_146199

theorem cube_sum_of_roots (p q r : ℝ) : 
  (p^3 - 2*p^2 + p - 3 = 0) → 
  (q^3 - 2*q^2 + q - 3 = 0) → 
  (r^3 - 2*r^2 + r - 3 = 0) → 
  p^3 + q^3 + r^3 = 11 := by sorry

end cube_sum_of_roots_l1461_146199


namespace train_speed_l1461_146146

/-- Proves that a train with given length and time to cross a stationary object has the specified speed -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (speed : ℝ) : 
  train_length = 250 →
  crossing_time = 12.857142857142858 →
  speed = (train_length / 1000) / (crossing_time / 3600) →
  speed = 70 := by
  sorry

#check train_speed

end train_speed_l1461_146146


namespace trajectory_of_Q_l1461_146123

/-- Given a circle ρ = 2cos θ, and a point Q on the extension of a chord OP such that OP/PQ = 2/3,
    prove that the trajectory of Q is a circle with equation ρ = 5cos θ. -/
theorem trajectory_of_Q (θ : Real) (ρ ρ_0 : Real → Real) :
  (∀ θ, ρ_0 θ = 2 * Real.cos θ) →  -- Given circle equation
  (∀ θ, ρ_0 θ / (ρ θ - ρ_0 θ) = 2 / 3) →  -- Ratio condition
  (∀ θ, ρ θ = 5 * Real.cos θ) :=  -- Trajectory equation to prove
by sorry

end trajectory_of_Q_l1461_146123


namespace parallel_lines_distance_l1461_146167

/-- Given three equally spaced parallel lines intersecting a circle and creating
    chords of lengths 42, 36, and 36, prove that the distance between two
    adjacent parallel lines is 2√2006. -/
theorem parallel_lines_distance (r : ℝ) (d : ℝ) : 
  (∃ (x y z : ℝ), x = 42 ∧ y = 36 ∧ z = 36 ∧
   21 * x * 21 + (d/2)^2 * x = 21 * r^2 + 21 * r^2 ∧
   18 * y * 18 + (d/2)^2 * y = 18 * r^2 + 18 * r^2) →
  d = 2 * Real.sqrt 2006 :=
by sorry

end parallel_lines_distance_l1461_146167


namespace largest_positive_root_bound_l1461_146186

theorem largest_positive_root_bound (b₂ b₁ b₀ : ℝ) 
  (h₂ : |b₂| ≤ 3) (h₁ : |b₁| ≤ 5) (h₀ : |b₀| ≤ 3) :
  ∃ s : ℝ, s > 4 ∧ s < 5 ∧
  (∀ x : ℝ, x > 0 → x^3 + b₂*x^2 + b₁*x + b₀ = 0 → x ≤ s) ∧
  (∃ b₂' b₁' b₀' : ℝ, |b₂'| ≤ 3 ∧ |b₁'| ≤ 5 ∧ |b₀'| ≤ 3 ∧
    s^3 + b₂'*s^2 + b₁'*s + b₀' = 0) :=
by sorry

end largest_positive_root_bound_l1461_146186


namespace sphere_cylinder_volume_ratio_l1461_146102

/-- The ratio of the volume of a sphere inscribed in a right circular cylinder
    to the volume of the cylinder. -/
theorem sphere_cylinder_volume_ratio :
  ∀ (r : ℝ), r > 0 →
  (4 / 3 * π * r^3) / (π * r^2 * (2 * r)) = 2 * Real.sqrt 3 * π / 27 := by
  sorry

end sphere_cylinder_volume_ratio_l1461_146102
