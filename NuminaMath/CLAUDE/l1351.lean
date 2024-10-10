import Mathlib

namespace six_minutes_to_hours_l1351_135179

-- Define the conversion factor from minutes to hours
def minutes_to_hours (minutes : ℚ) : ℚ := minutes / 60

-- State the theorem
theorem six_minutes_to_hours : 
  minutes_to_hours 6 = 0.1 := by
  sorry

end six_minutes_to_hours_l1351_135179


namespace smaller_circle_circumference_l1351_135116

/-- Given a square and two circles with specific relationships, 
    prove the circumference of the smaller circle -/
theorem smaller_circle_circumference 
  (square_area : ℝ) 
  (larger_radius smaller_radius : ℝ) 
  (h1 : square_area = 784)
  (h2 : square_area = (2 * larger_radius)^2)
  (h3 : larger_radius = (7/3) * smaller_radius) : 
  2 * Real.pi * smaller_radius = 12 * Real.pi := by
sorry

end smaller_circle_circumference_l1351_135116


namespace walter_seal_time_l1351_135140

/-- The time Walter spends at the zoo -/
def total_time : ℕ := 260

/-- Walter's initial time spent looking at seals -/
def initial_seal_time : ℕ := 20

/-- Time spent looking at penguins -/
def penguin_time (s : ℕ) : ℕ := 8 * s

/-- Time spent looking at elephants -/
def elephant_time : ℕ := 13

/-- Time spent on second visit to seals -/
def second_seal_time (s : ℕ) : ℕ := s / 2

/-- Time spent at giraffe exhibit -/
def giraffe_time (s : ℕ) : ℕ := 3 * s

/-- Total time spent looking at seals -/
def total_seal_time (s : ℕ) : ℕ := s + (s / 2)

theorem walter_seal_time :
  total_seal_time initial_seal_time = 30 ∧
  initial_seal_time + penguin_time initial_seal_time + elephant_time +
  second_seal_time initial_seal_time + giraffe_time initial_seal_time = total_time :=
sorry

end walter_seal_time_l1351_135140


namespace price_of_pants_l1351_135152

theorem price_of_pants (total_cost shirt_price pants_price shoes_price : ℝ) : 
  total_cost = 340 →
  shirt_price = (3/4) * pants_price →
  shoes_price = pants_price + 10 →
  total_cost = shirt_price + pants_price + shoes_price →
  pants_price = 120 := by
sorry

end price_of_pants_l1351_135152


namespace fraction_problem_l1351_135115

theorem fraction_problem (x : ℝ) : 
  (x * 7000 - (1 / 1000) * 7000 = 700) ↔ (x = 0.101) :=
by sorry

end fraction_problem_l1351_135115


namespace intersection_condition_l1351_135138

-- Define the curves
def C₁ (x : ℝ) : Prop := ∃ y : ℝ, y = x^2 ∧ -2 ≤ x ∧ x ≤ 2

def C₂ (m : ℝ) (x y : ℝ) : Prop := x - y + m = 0

-- Theorem statement
theorem intersection_condition (m : ℝ) :
  (∃ x y : ℝ, C₁ x ∧ C₂ m x y) ↔ -1/4 ≤ m ∧ m ≤ 6 := by
  sorry

end intersection_condition_l1351_135138


namespace shaniqua_style_price_l1351_135145

/-- Proves that Shaniqua makes $25 for every style given the conditions -/
theorem shaniqua_style_price 
  (haircut_price : ℕ) 
  (total_earned : ℕ) 
  (num_haircuts : ℕ) 
  (num_styles : ℕ) 
  (h1 : haircut_price = 12)
  (h2 : total_earned = 221)
  (h3 : num_haircuts = 8)
  (h4 : num_styles = 5)
  (h5 : total_earned = num_haircuts * haircut_price + num_styles * (total_earned - num_haircuts * haircut_price) / num_styles) : 
  (total_earned - num_haircuts * haircut_price) / num_styles = 25 := by
  sorry

end shaniqua_style_price_l1351_135145


namespace circle_intersection_theorem_l1351_135156

-- Define the circles
def circle_O₁ (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 4
def circle_O₂ (x y r : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = r^2

-- Define the theorem
theorem circle_intersection_theorem :
  -- Part 1: Tangent case
  (∀ x y : ℝ, circle_O₁ x y → ¬(circle_O₂ x y (12 - 8 * Real.sqrt 2))) →
  -- Part 2: Intersection case
  (∃ A B : ℝ × ℝ, 
    circle_O₁ A.1 A.2 ∧ circle_O₁ B.1 B.2 ∧
    circle_O₂ A.1 A.2 2 ∧ circle_O₂ B.1 B.2 2 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 8) →
  (∀ x y : ℝ, circle_O₂ x y 2 ∨ circle_O₂ x y (Real.sqrt 20)) :=
by sorry

end circle_intersection_theorem_l1351_135156


namespace power_multiplication_l1351_135119

theorem power_multiplication (a : ℝ) : a^3 * a = a^4 := by
  sorry

end power_multiplication_l1351_135119


namespace city_male_population_l1351_135160

theorem city_male_population (total_population : ℕ) (num_parts : ℕ) (male_parts : ℕ) :
  total_population = 800 →
  num_parts = 4 →
  male_parts = 2 →
  (total_population / num_parts) * male_parts = 400 :=
by sorry

end city_male_population_l1351_135160


namespace apple_bags_l1351_135173

theorem apple_bags (A B C : ℕ) 
  (h1 : A + B + C = 24) 
  (h2 : A + B = 11) 
  (h3 : B + C = 18) : 
  A + C = 19 := by
  sorry

end apple_bags_l1351_135173


namespace john_ray_difference_l1351_135174

/-- The number of chickens each person took -/
structure ChickenCount where
  john : ℕ
  mary : ℕ
  ray : ℕ

/-- The conditions of the chicken problem -/
def chicken_problem (c : ChickenCount) : Prop :=
  c.john = c.mary + 5 ∧
  c.mary = c.ray + 6 ∧
  c.ray = 10

/-- The theorem stating the difference between John's and Ray's chickens -/
theorem john_ray_difference (c : ChickenCount) 
  (h : chicken_problem c) : c.john - c.ray = 11 := by
  sorry

end john_ray_difference_l1351_135174


namespace computer_price_ratio_l1351_135195

theorem computer_price_ratio (d : ℝ) : 
  d + 0.3 * d = 377 → (d + 377) / d = 2.3 := by
  sorry

end computer_price_ratio_l1351_135195


namespace total_balls_l1351_135153

theorem total_balls (jungkook_balls yoongi_balls : ℕ) : 
  jungkook_balls = 3 → yoongi_balls = 2 → jungkook_balls + yoongi_balls = 5 := by
  sorry

end total_balls_l1351_135153


namespace bart_burning_period_l1351_135191

/-- The number of pieces of firewood Bart gets from one tree -/
def pieces_per_tree : ℕ := 75

/-- The number of logs Bart burns per day -/
def logs_per_day : ℕ := 5

/-- The number of trees Bart cuts down for the period -/
def trees_cut : ℕ := 8

/-- The period (in days) that Bart burns the logs -/
def burning_period : ℕ := (pieces_per_tree * trees_cut) / logs_per_day

theorem bart_burning_period :
  burning_period = 120 := by sorry

end bart_burning_period_l1351_135191


namespace team_capacity_ratio_l1351_135199

/-- The working capacity ratio of two teams -/
def working_capacity_ratio (p_engineers q_engineers : ℕ) (p_days q_days : ℕ) : ℚ × ℚ :=
  let p_capacity := p_engineers * p_days / p_engineers
  let q_capacity := q_engineers * q_days / q_engineers
  (p_capacity, q_capacity)

/-- Theorem: The ratio of working capacity for the given teams is 16:15 -/
theorem team_capacity_ratio :
  let (p_cap, q_cap) := working_capacity_ratio 20 16 32 30
  p_cap / q_cap = 16 / 15 := by
  sorry

end team_capacity_ratio_l1351_135199


namespace knights_seating_probability_correct_l1351_135117

/-- The probability of three knights being seated with empty chairs on either side
    when randomly placed around a circular table with n chairs. -/
def knights_seating_probability (n : ℕ) : ℚ :=
  if n ≥ 6 then
    (n - 4 : ℚ) * (n - 5) / ((n - 1 : ℚ) * (n - 2))
  else
    0

/-- Theorem stating the probability of three knights being seated with empty chairs
    on either side when randomly placed around a circular table with n chairs. -/
theorem knights_seating_probability_correct (n : ℕ) (h : n ≥ 6) :
  knights_seating_probability n =
    (n - 4 : ℚ) * (n - 5) / ((n - 1 : ℚ) * (n - 2)) :=
by sorry

end knights_seating_probability_correct_l1351_135117


namespace bank_deposit_exceeds_500_first_day_exceeding_500_l1351_135150

def bank_deposit (n : ℕ) : ℚ :=
  3 * (3^n - 1) / 2

theorem bank_deposit_exceeds_500 :
  ∃ n : ℕ, bank_deposit n > 500 ∧ ∀ m : ℕ, m < n → bank_deposit m ≤ 500 :=
by
  sorry

theorem first_day_exceeding_500 :
  (∃ n : ℕ, bank_deposit n > 500 ∧ ∀ m : ℕ, m < n → bank_deposit m ≤ 500) →
  (∃ n : ℕ, n = 6 ∧ bank_deposit n > 500 ∧ ∀ m : ℕ, m < n → bank_deposit m ≤ 500) :=
by
  sorry

end bank_deposit_exceeds_500_first_day_exceeding_500_l1351_135150


namespace latia_work_hours_l1351_135186

/-- Proves that Latia works 30 hours per week given the problem conditions -/
theorem latia_work_hours :
  ∀ (tv_price : ℕ) (hourly_rate : ℕ) (additional_hours : ℕ) (weeks_per_month : ℕ),
  tv_price = 1700 →
  hourly_rate = 10 →
  additional_hours = 50 →
  weeks_per_month = 4 →
  ∃ (hours_per_week : ℕ),
    hours_per_week * weeks_per_month * hourly_rate + additional_hours * hourly_rate = tv_price ∧
    hours_per_week = 30 :=
by
  sorry


end latia_work_hours_l1351_135186


namespace chef_nut_purchase_l1351_135111

/-- The weight of almonds bought by the chef in kilograms -/
def almond_weight : ℝ := 0.14

/-- The weight of pecans bought by the chef in kilograms -/
def pecan_weight : ℝ := 0.38

/-- The total weight of nuts bought by the chef in kilograms -/
def total_nut_weight : ℝ := almond_weight + pecan_weight

theorem chef_nut_purchase : total_nut_weight = 0.52 := by
  sorry

end chef_nut_purchase_l1351_135111


namespace ratio_problem_l1351_135188

theorem ratio_problem (A B C : ℝ) (h1 : A + B + C = 98) (h2 : B / C = 5 / 8) (h3 : B = 30) :
  A / B = 2 / 3 := by
  sorry

end ratio_problem_l1351_135188


namespace range_of_M_l1351_135107

theorem range_of_M (x y z : ℝ) (h1 : x + y + z = 30) (h2 : 3 * x + y - z = 50)
  (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  let M := 5 * x + 4 * y + 2 * z
  ∀ m, (m = M) → 120 ≤ m ∧ m ≤ 130 :=
by sorry

end range_of_M_l1351_135107


namespace mary_has_more_euros_l1351_135159

-- Define initial amounts
def michelle_initial : ℚ := 30
def alice_initial : ℚ := 18
def marco_initial : ℚ := 24
def mary_initial : ℚ := 15

-- Define conversion rate
def usd_to_eur : ℚ := 0.85

-- Define transactions
def marco_to_mary : ℚ := marco_initial / 2
def michelle_to_alice : ℚ := michelle_initial * (40 / 100)
def mary_spend : ℚ := 5
def alice_convert : ℚ := 10

-- Calculate final amounts
def marco_final : ℚ := marco_initial - marco_to_mary
def mary_final : ℚ := mary_initial + marco_to_mary - mary_spend
def alice_final_usd : ℚ := alice_initial + michelle_to_alice - alice_convert
def alice_final_eur : ℚ := alice_convert * usd_to_eur

-- Theorem statement
theorem mary_has_more_euros :
  mary_final = marco_final + alice_final_eur + (3/2) := by sorry

end mary_has_more_euros_l1351_135159


namespace solution_set_is_x_gt_one_l1351_135135

/-- A linear function y = kx + b with a table of x and y values -/
structure LinearFunction where
  k : ℝ
  b : ℝ
  k_nonzero : k ≠ 0
  x_values : List ℝ := [-2, -1, 0, 1, 2, 3]
  y_values : List ℝ := [3, 2, 1, 0, -1, -2]
  table_valid : x_values.length = y_values.length

/-- The solution set of kx + b < 0 for the given linear function -/
def solutionSet (f : LinearFunction) : Set ℝ :=
  {x | f.k * x + f.b < 0}

/-- Theorem stating that the solution set is x > 1 -/
theorem solution_set_is_x_gt_one (f : LinearFunction) : 
  solutionSet f = {x | x > 1} := by
  sorry

end solution_set_is_x_gt_one_l1351_135135


namespace union_of_A_and_B_complement_of_A_l1351_135127

-- Define the sets
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x^2 < 4}

-- State the theorems
theorem union_of_A_and_B : A ∪ B = {x | -2 < x ∧ x ≤ 3} := by sorry

theorem complement_of_A : (Set.univ \ A) = {x | x < -1 ∨ x > 3} := by sorry

end union_of_A_and_B_complement_of_A_l1351_135127


namespace books_at_end_of_month_l1351_135112

/-- Given a special collection of books, calculate the number of books at the end of the month. -/
theorem books_at_end_of_month 
  (initial_books : ℕ) 
  (loaned_books : ℕ) 
  (return_rate : ℚ) 
  (h1 : initial_books = 75)
  (h2 : loaned_books = 40)
  (h3 : return_rate = 65 / 100) : 
  initial_books - loaned_books + (return_rate * loaned_books).floor = 61 := by
  sorry

#check books_at_end_of_month

end books_at_end_of_month_l1351_135112


namespace arthur_walked_four_point_five_miles_l1351_135126

/-- The distance Arthur walked in miles -/
def arthurs_distance (blocks_west : ℕ) (blocks_south : ℕ) (miles_per_block : ℚ) : ℚ :=
  (blocks_west + blocks_south : ℚ) * miles_per_block

/-- Theorem stating that Arthur walked 4.5 miles -/
theorem arthur_walked_four_point_five_miles :
  arthurs_distance 8 10 (1/4) = 4.5 := by
  sorry

end arthur_walked_four_point_five_miles_l1351_135126


namespace problem_types_not_mutually_exclusive_l1351_135120

/-- Represents a mathematical problem type -/
inductive ProblemType
  | Proof
  | Computation
  | Construction

/-- Represents a mathematical problem -/
structure Problem where
  type : ProblemType
  hasProofElement : Bool
  hasComputationElement : Bool
  hasConstructionElement : Bool

/-- Theorem stating that problem types are not mutually exclusive -/
theorem problem_types_not_mutually_exclusive :
  ∃ (p : Problem), (p.type = ProblemType.Proof ∨ p.type = ProblemType.Computation ∨ p.type = ProblemType.Construction) ∧
    p.hasProofElement ∧ p.hasComputationElement ∧ p.hasConstructionElement :=
sorry

end problem_types_not_mutually_exclusive_l1351_135120


namespace polynomial_factorization_l1351_135169

theorem polynomial_factorization (x : ℝ) : 
  x^8 - 16 = (x^2 - 2) * (x^2 + 2) * (x^2 - 2*x + 2) * (x^2 + 2*x + 2) := by
  sorry

end polynomial_factorization_l1351_135169


namespace inequality_proof_l1351_135184

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (a^2 + 1) * (b^3 + 2) * (c^5 + 4) ≥ 30 ∧
  ((a^2 + 1) * (b^3 + 2) * (c^5 + 4) = 30 ↔ a = 1 ∧ b = 1 ∧ c = 1) :=
sorry

end inequality_proof_l1351_135184


namespace smallest_a1_l1351_135170

def is_valid_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∀ n > 1, a n = 11 * a (n - 1) - n)

theorem smallest_a1 (a : ℕ → ℝ) (h : is_valid_sequence a) :
  ∀ ε > 0, a 1 ≥ 21 / 100 - ε :=
sorry

end smallest_a1_l1351_135170


namespace max_actors_in_tournament_l1351_135141

/-- Represents the result of a chess match -/
inductive MatchResult
  | Win
  | Draw
  | Loss

/-- Calculates the score for a given match result -/
def scoreForResult (result : MatchResult) : Rat :=
  match result with
  | MatchResult.Win => 1
  | MatchResult.Draw => 1/2
  | MatchResult.Loss => 0

/-- Represents a chess tournament -/
structure ChessTournament (n : ℕ) where
  /-- The results of all matches in the tournament -/
  results : Fin n → Fin n → MatchResult
  /-- Each player plays exactly one match against each other player -/
  no_self_play : ∀ i, results i i = MatchResult.Draw
  /-- Matches are symmetric: if A wins against B, B loses against A -/
  symmetry : ∀ i j, results i j = MatchResult.Win ↔ results j i = MatchResult.Loss

/-- Calculates the score of player i against player j -/
def score (tournament : ChessTournament n) (i j : Fin n) : Rat :=
  scoreForResult (tournament.results i j)

/-- The tournament satisfies the "1.5 solido" condition -/
def satisfies_condition (tournament : ChessTournament n) : Prop :=
  ∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k →
    (score tournament i j + score tournament i k = 3/2) ∨
    (score tournament j i + score tournament j k = 3/2) ∨
    (score tournament k i + score tournament k j = 3/2)

/-- The main theorem: the maximum number of actors in a valid tournament is 5 -/
theorem max_actors_in_tournament :
  (∃ (tournament : ChessTournament 5), satisfies_condition tournament) ∧
  (∀ n > 5, ¬∃ (tournament : ChessTournament n), satisfies_condition tournament) :=
sorry

end max_actors_in_tournament_l1351_135141


namespace floor_equation_solution_l1351_135172

theorem floor_equation_solution (a b : ℝ) : 
  (∀ x y : ℝ, ⌊a*x + b*y⌋ + ⌊b*x + a*y⌋ = (a + b)*⌊x + y⌋) ↔ 
  ((a = 0 ∧ b = 0) ∨ (a = 1 ∧ b = 1)) :=
sorry

end floor_equation_solution_l1351_135172


namespace average_of_next_sequence_l1351_135110

def consecutive_integers_average (a b : ℕ) : Prop :=
  (a > 0) ∧ 
  (b = (a + (a + 1) + (a + 2) + (a + 3) + (a + 4)) / 5)

theorem average_of_next_sequence (a b : ℕ) :
  consecutive_integers_average a b →
  ((b + (b + 1) + (b + 2) + (b + 3) + (b + 4)) / 5 : ℚ) = a + 4 := by
  sorry

end average_of_next_sequence_l1351_135110


namespace sum_is_five_digits_l1351_135185

/-- A nonzero digit is a natural number between 1 and 9, inclusive. -/
def NonzeroDigit : Type := {n : ℕ // 1 ≤ n ∧ n ≤ 9}

/-- Convert a nonzero digit to a natural number. -/
def to_nat (d : NonzeroDigit) : ℕ := d.val

/-- The first number in the sum. -/
def num1 : ℕ := 59876

/-- The second number in the sum, parameterized by a nonzero digit A. -/
def num2 (A : NonzeroDigit) : ℕ := 1000 + 100 * (to_nat A) + 32

/-- The third number in the sum, parameterized by a nonzero digit B. -/
def num3 (B : NonzeroDigit) : ℕ := 10 * (to_nat B) + 1

/-- The sum of the three numbers. -/
def total_sum (A B : NonzeroDigit) : ℕ := num1 + num2 A + num3 B

/-- A number is a 5-digit number if it's between 10000 and 99999, inclusive. -/
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

theorem sum_is_five_digits (A B : NonzeroDigit) : is_five_digit (total_sum A B) := by
  sorry

end sum_is_five_digits_l1351_135185


namespace one_four_digit_perfect_square_palindrome_l1351_135192

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

theorem one_four_digit_perfect_square_palindrome :
  ∃! n : ℕ, is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n :=
sorry

end one_four_digit_perfect_square_palindrome_l1351_135192


namespace harry_needs_five_spellbooks_l1351_135176

/-- Represents the cost and quantity of items Harry needs to buy --/
structure HarrysPurchase where
  spellbookCost : ℕ
  potionKitCost : ℕ
  owlCost : ℕ
  silverToGoldRatio : ℕ
  totalSilver : ℕ
  potionKitQuantity : ℕ

/-- Calculates the number of spellbooks Harry needs to buy --/
def calculateSpellbooks (purchase : HarrysPurchase) : ℕ :=
  let remainingSilver := purchase.totalSilver -
    (purchase.owlCost * purchase.silverToGoldRatio + 
     purchase.potionKitCost * purchase.potionKitQuantity)
  remainingSilver / (purchase.spellbookCost * purchase.silverToGoldRatio)

/-- Theorem stating that Harry needs to buy 5 spellbooks --/
theorem harry_needs_five_spellbooks (purchase : HarrysPurchase) 
  (h1 : purchase.spellbookCost = 5)
  (h2 : purchase.potionKitCost = 20)
  (h3 : purchase.owlCost = 28)
  (h4 : purchase.silverToGoldRatio = 9)
  (h5 : purchase.totalSilver = 537)
  (h6 : purchase.potionKitQuantity = 3) :
  calculateSpellbooks purchase = 5 := by
  sorry


end harry_needs_five_spellbooks_l1351_135176


namespace first_alloy_copper_percentage_l1351_135181

/-- The percentage of copper in the final alloy -/
def final_alloy_percentage : ℝ := 15

/-- The total amount of the final alloy in ounces -/
def total_alloy : ℝ := 121

/-- The amount of the first alloy used in ounces -/
def first_alloy_amount : ℝ := 66

/-- The percentage of copper in the second alloy -/
def second_alloy_percentage : ℝ := 21

/-- The percentage of copper in the first alloy -/
def first_alloy_percentage : ℝ := 10

theorem first_alloy_copper_percentage :
  first_alloy_amount * (first_alloy_percentage / 100) +
  (total_alloy - first_alloy_amount) * (second_alloy_percentage / 100) =
  total_alloy * (final_alloy_percentage / 100) :=
by sorry

end first_alloy_copper_percentage_l1351_135181


namespace guaranteed_scores_l1351_135132

/-- Represents a player in the card game -/
inductive Player : Type
| One
| Two

/-- The deck of cards for each player -/
def player_deck (p : Player) : List Nat :=
  match p with
  | Player.One => List.range 1000 |>.map (fun n => 2 * n + 2)
  | Player.Two => List.range 1001 |>.map (fun n => 2 * n + 1)

/-- The number of turns in the game -/
def num_turns : Nat := 1000

/-- The result of the game -/
structure GameResult where
  player1_score : Nat
  player2_score : Nat

/-- A strategy for playing the game -/
def Strategy := List Nat → Nat

/-- Play the game with given strategies -/
def play_game (s1 s2 : Strategy) : GameResult :=
  sorry

/-- The theorem stating the guaranteed minimum scores for both players -/
theorem guaranteed_scores :
  ∃ (s1 : Strategy), ∀ (s2 : Strategy), (play_game s1 s2).player1_score ≥ 499 ∧
  ∃ (s2 : Strategy), ∀ (s1 : Strategy), (play_game s1 s2).player2_score ≥ 501 :=
  sorry

end guaranteed_scores_l1351_135132


namespace triangle_angle_theorem_l1351_135104

/-- A triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The circumcenter of a triangle -/
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

/-- The area of a triangle given three points -/
def area (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- Angle in degrees between three points -/
def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

theorem triangle_angle_theorem (t : Triangle) :
  let O := circumcenter t
  angle t.B t.C t.A = 75 →
  area O t.A t.B + area O t.B t.C = Real.sqrt 3 * area O t.C t.A →
  angle t.B t.A t.C = 45 := by
    sorry

end triangle_angle_theorem_l1351_135104


namespace s_range_l1351_135105

noncomputable def s (x : ℝ) : ℝ := 1 / (2 - x)^3

theorem s_range :
  {y : ℝ | ∃ x ≠ 2, s x = y} = {y : ℝ | y < 0 ∨ y > 0} :=
by sorry

end s_range_l1351_135105


namespace freezer_ice_cubes_l1351_135118

/-- The minimum number of ice cubes in Jerry's freezer -/
def min_ice_cubes (num_cups : ℕ) (ice_per_cup : ℕ) : ℕ :=
  num_cups * ice_per_cup

/-- Theorem stating that the minimum number of ice cubes is the product of cups and ice per cup -/
theorem freezer_ice_cubes (num_cups : ℕ) (ice_per_cup : ℕ) :
  min_ice_cubes num_cups ice_per_cup = num_cups * ice_per_cup :=
by sorry

end freezer_ice_cubes_l1351_135118


namespace book_cost_price_l1351_135178

theorem book_cost_price (cost_price : ℝ) : cost_price = 2200 :=
  let selling_price_10_percent := 1.10 * cost_price
  let selling_price_15_percent := 1.15 * cost_price
  have h1 : selling_price_15_percent - selling_price_10_percent = 110 := by sorry
  sorry

end book_cost_price_l1351_135178


namespace fox_weasel_hunting_average_l1351_135190

/-- Proves that given the initial conditions and the number of animals remaining after 3 weeks,
    the average number of weasels caught by each fox per week is 4. -/
theorem fox_weasel_hunting_average :
  let initial_weasels : ℕ := 100
  let initial_rabbits : ℕ := 50
  let num_foxes : ℕ := 3
  let rabbits_per_fox_per_week : ℕ := 2
  let weeks : ℕ := 3
  let animals_left : ℕ := 96
  let total_animals_caught := initial_weasels + initial_rabbits - animals_left
  let total_rabbits_caught := num_foxes * rabbits_per_fox_per_week * weeks
  let total_weasels_caught := total_animals_caught - total_rabbits_caught
  let weasels_per_fox := total_weasels_caught / num_foxes
  let avg_weasels_per_fox_per_week := weasels_per_fox / weeks
  avg_weasels_per_fox_per_week = 4 :=
by
  sorry

end fox_weasel_hunting_average_l1351_135190


namespace units_digit_of_7_to_2023_l1351_135165

theorem units_digit_of_7_to_2023 : 7^2023 % 10 = 3 := by
  sorry

end units_digit_of_7_to_2023_l1351_135165


namespace range_of_a_l1351_135197

/-- The solution set of the inequality |x+a|+|2x-1| ≤ |2x+1| with respect to x -/
def A (a : ℝ) : Set ℝ :=
  {x : ℝ | |x + a| + |2*x - 1| ≤ |2*x + 1|}

/-- The theorem stating the range of a given the conditions -/
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, x ∈ A a) → a ∈ Set.Icc (-3) 1 := by
  sorry

end range_of_a_l1351_135197


namespace milk_water_ratio_l1351_135149

theorem milk_water_ratio (initial_volume : ℝ) (water_added : ℝ) 
  (milk : ℝ) (water : ℝ) : 
  initial_volume = 115 →
  water_added = 46 →
  milk + water = initial_volume →
  milk / (water + water_added) = 3 / 4 →
  milk / water = 3 / 2 := by
sorry

end milk_water_ratio_l1351_135149


namespace water_bottles_needed_l1351_135158

/-- Calculates the total number of water bottles needed for a family road trip. -/
theorem water_bottles_needed
  (family_size : ℕ)
  (travel_time : ℕ)
  (water_consumption : ℚ)
  (h1 : family_size = 4)
  (h2 : travel_time = 16)
  (h3 : water_consumption = 1/2) :
  ↑family_size * ↑travel_time * water_consumption = 32 :=
by sorry

end water_bottles_needed_l1351_135158


namespace difference_of_squares_l1351_135106

theorem difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end difference_of_squares_l1351_135106


namespace smallest_n_congruence_l1351_135137

theorem smallest_n_congruence (n : ℕ) : 
  (n > 0 ∧ 19 * n ≡ 789 [ZMOD 11]) → n = 1 :=
sorry

end smallest_n_congruence_l1351_135137


namespace sequence_problem_l1351_135133

theorem sequence_problem (a : ℕ → ℝ) :
  a 1 = 1 ∧
  (∀ n : ℕ, n ≥ 1 → a (n + 1) - a n ≤ 3^n) ∧
  (∀ n : ℕ, n ≥ 1 → a (n + 2) - a n ≥ 4 * 3^n) →
  a 2017 = (3^2017 - 1) / 2 := by
sorry

end sequence_problem_l1351_135133


namespace problem_statement_l1351_135155

theorem problem_statement (x₁ x₂ : ℝ) 
  (h₁ : |x₁ - 2| < 1) 
  (h₂ : |x₂ - 2| < 1) : 
  (2 < x₁ + x₂ ∧ x₁ + x₂ < 6 ∧ |x₁ - x₂| < 2) ∧ 
  (let f := fun x => x^2 - x + 1
   |x₁ - x₂| < |f x₁ - f x₂| ∧ |f x₁ - f x₂| < 5 * |x₁ - x₂|) := by
  sorry

end problem_statement_l1351_135155


namespace cubical_box_edge_length_cubical_box_edge_length_proof_l1351_135168

/-- The edge length of a cubical box that can hold 64 cubes with edge length 25 cm is 1 meter. -/
theorem cubical_box_edge_length : Real → Prop := fun edge_length =>
  let small_cube_volume := (25 / 100) ^ 3
  let box_volume := 64 * small_cube_volume
  edge_length ^ 3 = box_volume → edge_length = 1

/-- Proof of the cubical box edge length theorem -/
theorem cubical_box_edge_length_proof : cubical_box_edge_length 1 := by
  sorry


end cubical_box_edge_length_cubical_box_edge_length_proof_l1351_135168


namespace probability_sum_10_l1351_135166

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The set of possible outcomes when throwing two dice -/
def outcomes : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range numFaces) (Finset.range numFaces)

/-- The total number of possible outcomes -/
def totalOutcomes : ℕ := numFaces * numFaces

/-- The sum of two numbers -/
def sum (pair : ℕ × ℕ) : ℕ := pair.1 + pair.2

/-- The set of favorable outcomes (sum equals 10) -/
def favorableOutcomes : Finset (ℕ × ℕ) :=
  outcomes.filter (fun pair => sum pair = 10)

/-- The number of favorable outcomes -/
def numFavorableOutcomes : ℕ := favorableOutcomes.card

theorem probability_sum_10 :
  (numFavorableOutcomes : ℚ) / totalOutcomes = 5 / 36 := by
  sorry

#eval numFavorableOutcomes -- Should output 5
#eval totalOutcomes -- Should output 36

end probability_sum_10_l1351_135166


namespace chopped_cube_height_l1351_135100

theorem chopped_cube_height (cube_side_length : ℝ) (h_side : cube_side_length = 2) :
  let chopped_corner_height := cube_side_length - (1 / Real.sqrt 3)
  chopped_corner_height = (5 * Real.sqrt 3) / 3 := by
  sorry

end chopped_cube_height_l1351_135100


namespace lo_length_l1351_135175

/-- Represents a parallelogram LMNO with given properties -/
structure Parallelogram where
  -- Length of side MN
  mn_length : ℝ
  -- Altitude from O to MN
  altitude_o_to_mn : ℝ
  -- Altitude from N to LO
  altitude_n_to_lo : ℝ
  -- Condition that LMNO is a parallelogram
  is_parallelogram : True

/-- Theorem stating the length of LO in the parallelogram LMNO -/
theorem lo_length (p : Parallelogram)
  (h1 : p.mn_length = 15)
  (h2 : p.altitude_o_to_mn = 9)
  (h3 : p.altitude_n_to_lo = 7) :
  ∃ (lo_length : ℝ), lo_length = 19 + 2 / 7 ∧ 
  p.mn_length * p.altitude_o_to_mn = lo_length * p.altitude_n_to_lo :=
sorry

end lo_length_l1351_135175


namespace time_2nd_to_7th_floor_l1351_135157

/-- Time needed to go from floor a to floor b, given the time to go from floor c to floor d -/
def time_between_floors (a b c d : ℕ) (time_cd : ℕ) : ℕ :=
  ((b - a) * time_cd) / (d - c)

/-- The theorem stating that it takes 50 seconds to go from the 2nd to the 7th floor -/
theorem time_2nd_to_7th_floor : 
  time_between_floors 2 7 1 5 40 = 50 := by sorry

end time_2nd_to_7th_floor_l1351_135157


namespace stating_optimal_swap_distance_maximizes_total_distance_l1351_135193

/-- Front tire lifespan in kilometers -/
def front_lifespan : ℝ := 11000

/-- Rear tire lifespan in kilometers -/
def rear_lifespan : ℝ := 9000

/-- The optimal swap distance in kilometers -/
def optimal_swap_distance : ℝ := 4950

/-- 
Theorem stating that the optimal swap distance maximizes total distance traveled
while ensuring both tires wear out simultaneously.
-/
theorem optimal_swap_distance_maximizes_total_distance :
  let total_distance := front_lifespan + rear_lifespan
  let front_remaining := 1 - (optimal_swap_distance / front_lifespan)
  let rear_remaining := 1 - (optimal_swap_distance / rear_lifespan)
  let distance_after_swap := front_remaining * rear_lifespan
  (front_remaining * rear_lifespan = rear_remaining * front_lifespan) ∧
  (optimal_swap_distance + distance_after_swap = total_distance) ∧
  (∀ x : ℝ, x ≠ optimal_swap_distance →
    let front_remaining' := 1 - (x / front_lifespan)
    let rear_remaining' := 1 - (x / rear_lifespan)
    let distance_after_swap' := min (front_remaining' * rear_lifespan) (rear_remaining' * front_lifespan)
    x + distance_after_swap' ≤ total_distance) :=
by
  sorry

end stating_optimal_swap_distance_maximizes_total_distance_l1351_135193


namespace projection_composition_l1351_135113

open Matrix

/-- The matrix that projects a vector onto (4, 2) -/
def proj_matrix_1 : Matrix (Fin 2) (Fin 2) ℚ :=
  !![4/5, 2/5; 2/5, 1/5]

/-- The matrix that projects a vector onto (2, 1) -/
def proj_matrix_2 : Matrix (Fin 2) (Fin 2) ℚ :=
  !![4/5, 2/5; 2/5, 1/5]

/-- The theorem stating that the composition of the two projection matrices
    results in the same matrix -/
theorem projection_composition :
  proj_matrix_2 * proj_matrix_1 = !![4/5, 2/5; 2/5, 1/5] := by sorry

end projection_composition_l1351_135113


namespace batsman_total_score_l1351_135148

/-- Represents a batsman's score in cricket -/
structure BatsmanScore where
  boundaries : ℕ
  sixes : ℕ
  runningPercentage : ℚ

/-- Calculates the total score of a batsman -/
def totalScore (score : BatsmanScore) : ℕ :=
  sorry

theorem batsman_total_score (score : BatsmanScore) 
  (h1 : score.boundaries = 6)
  (h2 : score.sixes = 4)
  (h3 : score.runningPercentage = 60/100) :
  totalScore score = 120 := by
  sorry

end batsman_total_score_l1351_135148


namespace contrapositive_equivalence_l1351_135187

theorem contrapositive_equivalence (x : ℝ) :
  (x = 1 → x^2 - 3*x + 2 = 0) ↔ (x^2 - 3*x + 2 ≠ 0 → x ≠ 1) := by
  sorry

end contrapositive_equivalence_l1351_135187


namespace greatest_divisor_l1351_135146

theorem greatest_divisor (G : ℕ) : G = 127 ↔ 
  G > 0 ∧ 
  (∀ n : ℕ, n > G → ¬(1657 % n = 6 ∧ 2037 % n = 5)) ∧
  1657 % G = 6 ∧ 
  2037 % G = 5 := by
sorry

end greatest_divisor_l1351_135146


namespace complement_A_when_a_5_union_A_B_when_a_2_l1351_135102

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ 2*a + 1}
def B : Set ℝ := {x | x < 0 ∨ x > 5}

-- Theorem 1: Complement of A when a = 5
theorem complement_A_when_a_5 : 
  (A 5)ᶜ = {x : ℝ | x < 4 ∨ x > 11} := by sorry

-- Theorem 2: Union of A and B when a = 2
theorem union_A_B_when_a_2 : 
  A 2 ∪ B = {x : ℝ | x < 0 ∨ x ≥ 1} := by sorry

end complement_A_when_a_5_union_A_B_when_a_2_l1351_135102


namespace spade_calculation_l1351_135134

-- Define the spade operation
def spade (x y : ℝ) : ℝ := (x + y + 1) * (x - y)

-- Theorem statement
theorem spade_calculation : spade 2 (spade 3 6) = -864 := by
  sorry

end spade_calculation_l1351_135134


namespace range_of_a_l1351_135144

/-- The range of values for real number a given specific conditions -/
theorem range_of_a (a : ℝ) : 
  (∃ x, x^2 - 4*a*x + 3*a^2 < 0 ∧ a < 0) →
  (∃ x, x^2 + 2*x - 8 > 0) →
  (∀ x, (x^2 - 4*a*x + 3*a^2 ≥ 0 ∨ a ≥ 0) → (x^2 + 2*x - 8 ≤ 0)) →
  (∃ x, (x^2 + 2*x - 8 ≤ 0) ∧ (x^2 - 4*a*x + 3*a^2 ≥ 0 ∨ a ≥ 0)) →
  a ≤ -4 :=
by sorry

end range_of_a_l1351_135144


namespace mikes_weekly_exercises_l1351_135163

/-- Represents the number of repetitions for each exercise -/
structure ExerciseReps where
  pullUps : ℕ
  pushUps : ℕ
  squats : ℕ

/-- Represents the number of daily visits to each room -/
structure RoomVisits where
  office : ℕ
  kitchen : ℕ
  livingRoom : ℕ

/-- Calculates the total number of exercises performed in a week -/
def weeklyExercises (reps : ExerciseReps) (visits : RoomVisits) : ExerciseReps :=
  { pullUps := reps.pullUps * visits.office * 7,
    pushUps := reps.pushUps * visits.kitchen * 7,
    squats := reps.squats * visits.livingRoom * 7 }

/-- Mike's exercise routine -/
def mikesRoutine : ExerciseReps :=
  { pullUps := 2, pushUps := 5, squats := 10 }

/-- Mike's daily room visits -/
def mikesVisits : RoomVisits :=
  { office := 5, kitchen := 8, livingRoom := 7 }

theorem mikes_weekly_exercises :
  weeklyExercises mikesRoutine mikesVisits = { pullUps := 70, pushUps := 280, squats := 490 } := by
  sorry

end mikes_weekly_exercises_l1351_135163


namespace quadratic_inequality_coefficient_sum_l1351_135123

/-- Given a quadratic inequality ax^2 + bx + 2 > 0 with solution set (-1/2, 1/3),
    prove that a + b = -14 -/
theorem quadratic_inequality_coefficient_sum (a b : ℝ) : 
  (∀ x, a * x^2 + b * x + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) →
  a + b = -14 :=
by sorry

end quadratic_inequality_coefficient_sum_l1351_135123


namespace tangent_through_origin_l1351_135171

theorem tangent_through_origin (x : ℝ) :
  (∃ y : ℝ, y = Real.exp x ∧ 
   (Real.exp x) * (0 - x) = 0 - y) →
  x = 1 ∧ Real.exp x = Real.exp 1 := by
  sorry

end tangent_through_origin_l1351_135171


namespace rectangular_to_polar_conversion_l1351_135122

theorem rectangular_to_polar_conversion :
  ∀ (x y : ℝ),
  x = -3 ∧ y = 1 →
  ∃ (r θ : ℝ),
  r > 0 ∧
  0 ≤ θ ∧ θ < 2 * Real.pi ∧
  r = Real.sqrt 10 ∧
  θ = Real.pi - Real.arctan (1 / 3) ∧
  x = r * Real.cos θ ∧
  y = r * Real.sin θ :=
by sorry

end rectangular_to_polar_conversion_l1351_135122


namespace total_pizza_combinations_l1351_135198

def num_toppings : ℕ := 8

def num_one_topping (n : ℕ) : ℕ := n

def num_two_toppings (n : ℕ) : ℕ := n.choose 2

def num_three_toppings (n : ℕ) : ℕ := n.choose 3

theorem total_pizza_combinations :
  num_one_topping num_toppings + num_two_toppings num_toppings + num_three_toppings num_toppings = 92 := by
  sorry

end total_pizza_combinations_l1351_135198


namespace inequality_implies_x_equals_one_l1351_135162

theorem inequality_implies_x_equals_one (x : ℝ) : 
  (∀ m : ℝ, m > 0 → (m * x - 1) * (3 * m^2 - (x + 1) * m - 1) ≥ 0) → 
  x = 1 := by
sorry

end inequality_implies_x_equals_one_l1351_135162


namespace decimal_fraction_equality_l1351_135196

theorem decimal_fraction_equality (b : ℕ+) : 
  (5 * b + 17 : ℚ) / (7 * b + 12) = 85 / 100 ↔ b = 7 := by sorry

end decimal_fraction_equality_l1351_135196


namespace bridge_length_calculation_l1351_135121

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 250 →
  train_speed_kmh = 60 →
  crossing_time = 20 →
  ∃ (bridge_length : ℝ), (abs (bridge_length - 83.4) < 0.1) ∧
    (bridge_length = train_speed_kmh * 1000 / 3600 * crossing_time - train_length) :=
by sorry

end bridge_length_calculation_l1351_135121


namespace shaded_area_ratio_l1351_135147

/-- Given a rectangle divided into a 4x5 grid of 1cm x 1cm smaller rectangles,
    with a shaded area consisting of 3 full small rectangles and 4 half small rectangles,
    prove that the ratio of the shaded area to the total area is 1/4. -/
theorem shaded_area_ratio (total_width : ℝ) (total_height : ℝ) 
  (full_rectangles : ℕ) (half_rectangles : ℕ) :
  total_width = 4 →
  total_height = 5 →
  full_rectangles = 3 →
  half_rectangles = 4 →
  (full_rectangles + half_rectangles / 2) / (total_width * total_height) = 1 / 4 := by
  sorry

end shaded_area_ratio_l1351_135147


namespace car_speed_comparison_l1351_135177

theorem car_speed_comparison (u v w : ℝ) (hu : u > 0) (hv : v > 0) (hw : w > 0) :
  3 / (1/u + 1/v + 1/w) ≤ (u + v) / 2 := by
  sorry

end car_speed_comparison_l1351_135177


namespace bee_path_distance_l1351_135131

open Complex

-- Define ω as e^(πi/4)
noncomputable def ω : ℂ := exp (I * Real.pi / 4)

-- Define the path of the bee
noncomputable def z : ℂ := 1 + 2 * ω + 3 * ω^2 + 4 * ω^3 + 5 * ω^4 + 6 * ω^5 + 7 * ω^6

-- Theorem stating the distance from P₀ to P₇
theorem bee_path_distance : abs z = Real.sqrt (25 - 7 * Real.sqrt 2 / 2) := by sorry

end bee_path_distance_l1351_135131


namespace rabbit_population_growth_l1351_135151

theorem rabbit_population_growth (initial_rabbits new_rabbits : ℕ) 
  (h1 : initial_rabbits = 8) 
  (h2 : new_rabbits = 5) : 
  initial_rabbits + new_rabbits = 13 := by
  sorry

end rabbit_population_growth_l1351_135151


namespace train_crossing_time_l1351_135189

theorem train_crossing_time (train_length : ℝ) (platform1_length : ℝ) (platform2_length : ℝ) (time1 : ℝ) :
  train_length = 100 →
  platform1_length = 350 →
  platform2_length = 500 →
  time1 = 15 →
  let total_distance1 := train_length + platform1_length
  let speed := total_distance1 / time1
  let total_distance2 := train_length + platform2_length
  let time2 := total_distance2 / speed
  time2 = 20 := by sorry

end train_crossing_time_l1351_135189


namespace not_sufficient_nor_necessary_l1351_135125

/-- Two lines are parallel if their slopes are equal and they are not coincident -/
def parallel (a b c d e f : ℝ) : Prop :=
  a / d = b / e ∧ a / d ≠ c / f

/-- First line equation: ax + 2y + 3a = 0 -/
def line1 (a : ℝ) (x y : ℝ) : Prop :=
  a * x + 2 * y + 3 * a = 0

/-- Second line equation: 3x + (a-1)y + a^2 - a + 3 = 0 -/
def line2 (a : ℝ) (x y : ℝ) : Prop :=
  3 * x + (a - 1) * y + a^2 - a + 3 = 0

/-- Theorem stating that a=3 is neither sufficient nor necessary for the lines to be parallel -/
theorem not_sufficient_nor_necessary :
  ¬(∀ a : ℝ, a = 3 → parallel a 2 (3*a) 3 (a-1) (a^2 - a + 3)) ∧
  ¬(∀ a : ℝ, parallel a 2 (3*a) 3 (a-1) (a^2 - a + 3) → a = 3) :=
sorry

end not_sufficient_nor_necessary_l1351_135125


namespace range_of_m_l1351_135154

/-- The probability that two lines l₁: ax + by = 2 and l₂: x + 2y = 2 are parallel,
    where a and b are results of two dice throws. -/
def P₁ : ℚ := 1/18

/-- The probability that two lines l₁: ax + by = 2 and l₂: x + 2y = 2 intersect,
    where a and b are results of two dice throws. -/
def P₂ : ℚ := 11/12

/-- The theorem stating the range of m for which the point (P₁, P₂) is inside
    the circle (x-m)² + y² = 137/144. -/
theorem range_of_m : ∀ m : ℚ, 
  (P₁ - m)^2 + P₂^2 < 137/144 ↔ -5/18 < m ∧ m < 7/18 := by sorry

end range_of_m_l1351_135154


namespace no_equal_sum_partition_l1351_135139

/-- A group of four consecutive natural numbers -/
structure NumberGroup :=
  (start : ℕ)
  (h : start > 0 ∧ start ≤ 69)

/-- The product of four consecutive natural numbers starting from n -/
def groupProduct (g : NumberGroup) : ℕ :=
  g.start * (g.start + 1) * (g.start + 2) * (g.start + 3)

/-- The sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- A partition of 72 consecutive natural numbers into 18 groups -/
def Partition := Fin 18 → NumberGroup

/-- The theorem stating that no partition exists where all groups have the same sum of digits of their product -/
theorem no_equal_sum_partition :
  ¬ ∃ (p : Partition), ∃ (s : ℕ), ∀ i : Fin 18, sumOfDigits (groupProduct (p i)) = s :=
sorry

end no_equal_sum_partition_l1351_135139


namespace square_difference_l1351_135101

theorem square_difference (x y : ℚ) 
  (h1 : x + y = 3/8) 
  (h2 : x - y = 1/8) : 
  x^2 - y^2 = 3/64 := by
  sorry

end square_difference_l1351_135101


namespace purple_shoes_count_l1351_135136

/-- Prove the number of purple shoes in a warehouse --/
theorem purple_shoes_count (total : ℕ) (blue : ℕ) (green : ℕ) (purple : ℕ) : 
  total = 1250 →
  blue = 540 →
  green + purple = total - blue →
  green = purple →
  purple = 355 := by
sorry

end purple_shoes_count_l1351_135136


namespace catch_up_point_l1351_135103

/-- Represents a car traveling between two cities -/
structure Car where
  speed : ℝ
  startTime : ℝ
  arrivalTime : ℝ

/-- The problem setup -/
def travelProblem (distanceAB : ℝ) (carA carB : Car) : Prop :=
  distanceAB > 0 ∧
  carA.startTime = carB.startTime + 1 ∧
  carA.arrivalTime + 1 = carB.arrivalTime ∧
  distanceAB = carA.speed * (carA.arrivalTime - carA.startTime) ∧
  distanceAB = carB.speed * (carB.arrivalTime - carB.startTime)

/-- The theorem to be proved -/
theorem catch_up_point (distanceAB : ℝ) (carA carB : Car) 
  (h : travelProblem distanceAB carA carB) : 
  ∃ (t : ℝ), carA.speed * (t - carA.startTime) = carB.speed * (t - carB.startTime) ∧ 
              carA.speed * (t - carA.startTime) = distanceAB - 150 := by
  sorry

end catch_up_point_l1351_135103


namespace bobs_work_hours_l1351_135128

/-- Given Bob's wage increase, benefit reduction, and net weekly gain, 
    prove that he works 40 hours per week. -/
theorem bobs_work_hours : 
  ∀ (h : ℝ), 
    (0.50 * h - 15 = 5) → 
    h = 40 :=
by
  sorry

end bobs_work_hours_l1351_135128


namespace non_juniors_playing_sport_l1351_135108

theorem non_juniors_playing_sport (total_students : ℕ) 
  (juniors_play_percent : ℚ) (non_juniors_not_play_percent : ℚ) 
  (total_not_play_percent : ℚ) : ℕ :=
  
  -- Define the given conditions
  let total_students := 600
  let juniors_play_percent := 1/2
  let non_juniors_not_play_percent := 2/5
  let total_not_play_percent := 13/25

  -- Define the number of non-juniors who play a sport
  let non_juniors_play := 72

  -- Proof statement (not implemented)
  by sorry

end non_juniors_playing_sport_l1351_135108


namespace plane_line_perpendicular_parallel_perpendicular_parallel_transitive_l1351_135109

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations between planes and lines
variable (perpendicular : Plane → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (linePerpendicular : Line → Plane → Prop)

-- Define distinct planes
variable (α β γ : Plane)
variable (distinct_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ)

-- Define a line
variable (l : Line)

theorem plane_line_perpendicular_parallel 
  (h1 : linePerpendicular l α) 
  (h2 : linePerpendicular l β) : 
  parallel α β := by sorry

theorem perpendicular_parallel_transitive 
  (h1 : perpendicular α γ) 
  (h2 : parallel β γ) : 
  perpendicular α β := by sorry

end plane_line_perpendicular_parallel_perpendicular_parallel_transitive_l1351_135109


namespace even_function_condition_l1351_135124

/-- A function f is even if f(-x) = f(x) for all x in ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The function f(x) = (x+a)(x-4) -/
def f (a : ℝ) (x : ℝ) : ℝ := (x + a) * (x - 4)

theorem even_function_condition (a : ℝ) : IsEven (f a) ↔ a = 4 := by
  sorry

end even_function_condition_l1351_135124


namespace largest_angle_in_triangle_l1351_135194

-- Define the triangle's angles
def angle1 : ℝ := 40
def angle2 : ℝ := 70
def angle3 : ℝ := 180 - angle1 - angle2

-- Theorem statement
theorem largest_angle_in_triangle : 
  max angle1 (max angle2 angle3) = 70 := by
  sorry

end largest_angle_in_triangle_l1351_135194


namespace absolute_value_negative_2023_l1351_135130

theorem absolute_value_negative_2023 : |(-2023 : ℤ)| = 2023 := by
  sorry

end absolute_value_negative_2023_l1351_135130


namespace talent_show_girls_l1351_135114

theorem talent_show_girls (total : ℕ) (difference : ℕ) (girls : ℕ) : 
  total = 34 → difference = 22 → girls = total - (total - difference) / 2 → girls = 28 := by
sorry

end talent_show_girls_l1351_135114


namespace inequality_infimum_l1351_135129

theorem inequality_infimum (m : ℝ) : 
  (∃ (a b : ℝ), ∀ (x : ℝ), x ∈ Set.Icc 0 1 → x^3 - m ≤ a*x + b ∧ a*x + b ≤ x^3 + m) →
  m ≥ Real.sqrt 3 / 9 :=
sorry

end inequality_infimum_l1351_135129


namespace problem_solution_l1351_135164

theorem problem_solution (a b c : ℝ) 
  (h1 : a + 2*b + 3*c = 12) 
  (h2 : a^2 + b^2 + c^2 = a*b + a*c + b*c) : 
  a + b^2 + c^3 = 14 := by
sorry

end problem_solution_l1351_135164


namespace perfect_square_sum_l1351_135142

theorem perfect_square_sum : 529 + 2 * 23 * 7 + 49 = 900 := by
  sorry

end perfect_square_sum_l1351_135142


namespace circle_equation_tangent_to_line_l1351_135167

/-- The equation of a circle with center (-1, 1) that is tangent to the line x - y = 0 -/
theorem circle_equation_tangent_to_line (x y : ℝ) : 
  (∃ (r : ℝ), (x + 1)^2 + (y - 1)^2 = r^2 ∧ 
  r = |(-1 - 1 + 0)| / Real.sqrt (1^2 + (-1)^2) ∧
  r > 0) ↔ 
  (x + 1)^2 + (y - 1)^2 = 2 :=
sorry

end circle_equation_tangent_to_line_l1351_135167


namespace greatest_3digit_base9_divisible_by_7_l1351_135143

def base9ToDecimal (n : Nat) : Nat :=
  (n / 100) * 9^2 + ((n / 10) % 10) * 9 + (n % 10)

theorem greatest_3digit_base9_divisible_by_7 :
  ∃ (n : Nat), 
    n < 1000 ∧ 
    base9ToDecimal n % 7 = 0 ∧
    (∀ m : Nat, m < 1000 → base9ToDecimal m % 7 = 0 → m ≤ n) ∧
    n = 888 := by
  sorry

end greatest_3digit_base9_divisible_by_7_l1351_135143


namespace sum_reciprocal_complements_l1351_135180

theorem sum_reciprocal_complements (a b c d : ℝ) 
  (h1 : a + b + c + d = 2) 
  (h2 : 1/a + 1/b + 1/c + 1/d = 2) : 
  1/(1-a) + 1/(1-b) + 1/(1-c) + 1/(1-d) = 2 := by
  sorry

end sum_reciprocal_complements_l1351_135180


namespace perpendicular_tangents_and_inequality_l1351_135182

noncomputable def f (x : ℝ) := x^2 + 4*x + 2

noncomputable def g (t x : ℝ) := t * Real.exp x * ((2*x + 4) - 2)

theorem perpendicular_tangents_and_inequality (t k : ℝ) : 
  (((2 * (-17/8) + 4) * (2 * t * Real.exp 0 * (0 + 2)) = -1) ∧
   (∀ x : ℝ, x ≥ 2 → k * g 1 x ≥ 2 * f x)) ↔ 
  (t = 1 ∧ 2 ≤ k ∧ k ≤ 2 * Real.exp 2) :=
sorry

end perpendicular_tangents_and_inequality_l1351_135182


namespace min_distance_circle_to_line_l1351_135183

/-- The minimum distance from any point on a circle to a line --/
theorem min_distance_circle_to_line :
  let circle := {(x, y) : ℝ × ℝ | (x - 1)^2 + (y - 1)^2 = 4}
  let line := {(x, y) : ℝ × ℝ | x - y + 4 = 0}
  (∃ (d : ℝ), d = 2 * Real.sqrt 2 - 2 ∧
    ∀ (p : ℝ × ℝ), p ∈ circle →
      ∀ (q : ℝ × ℝ), q ∈ line →
        d ≤ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)) :=
by sorry

end min_distance_circle_to_line_l1351_135183


namespace intersection_area_is_sqrt_80_l1351_135161

/-- Represents a square pyramid -/
structure SquarePyramid where
  base_side : ℝ
  edge_length : ℝ

/-- Represents a plane intersecting the pyramid -/
structure IntersectingPlane where
  pyramid : SquarePyramid
  -- The plane passes through midpoints of one lateral edge and two base edges

/-- The area of intersection between the plane and the pyramid -/
noncomputable def intersection_area (plane : IntersectingPlane) : ℝ := sorry

theorem intersection_area_is_sqrt_80 (plane : IntersectingPlane) :
  plane.pyramid.base_side = 4 →
  plane.pyramid.edge_length = 4 →
  intersection_area plane = Real.sqrt 80 := by
  sorry

end intersection_area_is_sqrt_80_l1351_135161
