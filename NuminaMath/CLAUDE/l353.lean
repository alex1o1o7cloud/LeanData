import Mathlib

namespace prob_four_genuine_given_equal_weights_l353_35396

-- Define the total number of coins
def total_coins : ℕ := 20

-- Define the number of genuine coins
def genuine_coins : ℕ := 12

-- Define the number of counterfeit coins
def counterfeit_coins : ℕ := 8

-- Define a function to calculate the probability of selecting genuine coins
def prob_genuine_selection (selected : ℕ) (remaining : ℕ) : ℚ :=
  (genuine_coins.choose selected) / (total_coins.choose selected)

-- Define the probability of selecting four genuine coins
def prob_four_genuine : ℚ :=
  (prob_genuine_selection 2 total_coins) * (prob_genuine_selection 2 (total_coins - 2))

-- Define the probability of equal weights (approximation)
def prob_equal_weights : ℚ :=
  prob_four_genuine + (counterfeit_coins / total_coins) * ((counterfeit_coins - 1) / (total_coins - 1)) *
  ((counterfeit_coins - 2) / (total_coins - 2)) * (1 / (total_coins - 3))

-- State the theorem
theorem prob_four_genuine_given_equal_weights :
  prob_four_genuine / prob_equal_weights = 550 / 703 := by
  sorry

end prob_four_genuine_given_equal_weights_l353_35396


namespace inconsistent_pricing_problem_l353_35329

theorem inconsistent_pricing_problem (shirt trouser tie : ℕ → ℚ) :
  (∃ x : ℕ, 6 * shirt 1 + 4 * trouser 1 + x * tie 1 = 80) →
  (4 * shirt 1 + 2 * trouser 1 + 2 * tie 1 = 140) →
  (5 * shirt 1 + 3 * trouser 1 + 2 * tie 1 = 110) →
  False :=
by
  sorry

end inconsistent_pricing_problem_l353_35329


namespace nineteen_only_vegetarian_l353_35364

/-- Represents the number of people in different dietary categories in a family -/
structure FamilyDiet where
  only_non_veg : ℕ
  both_veg_and_non_veg : ℕ
  total_veg : ℕ

/-- Calculates the number of people who eat only vegetarian -/
def only_vegetarian (f : FamilyDiet) : ℕ :=
  f.total_veg - f.both_veg_and_non_veg

/-- Theorem stating that 19 people eat only vegetarian in the given family -/
theorem nineteen_only_vegetarian (f : FamilyDiet) 
  (h1 : f.only_non_veg = 9)
  (h2 : f.both_veg_and_non_veg = 12)
  (h3 : f.total_veg = 31) :
  only_vegetarian f = 19 := by
  sorry

end nineteen_only_vegetarian_l353_35364


namespace sqrt_18_div_sqrt_2_equals_3_l353_35361

theorem sqrt_18_div_sqrt_2_equals_3 : Real.sqrt 18 / Real.sqrt 2 = 3 := by
  sorry

end sqrt_18_div_sqrt_2_equals_3_l353_35361


namespace anthony_lunch_money_l353_35330

theorem anthony_lunch_money (initial_money juice_cost cupcake_cost : ℕ) 
  (h1 : initial_money = 75)
  (h2 : juice_cost = 27)
  (h3 : cupcake_cost = 40) :
  initial_money - (juice_cost + cupcake_cost) = 8 :=
by
  sorry

end anthony_lunch_money_l353_35330


namespace final_color_is_yellow_l353_35346

/-- Represents the color of an elf -/
inductive ElfColor
  | Blue
  | Red
  | Yellow

/-- Represents the state of elves on the island -/
structure ElfState where
  blue : Nat
  red : Nat
  yellow : Nat
  total : Nat
  h_total : blue + red + yellow = total

/-- The score assigned to each color -/
def colorScore (c : ElfColor) : Nat :=
  match c with
  | ElfColor.Blue => 1
  | ElfColor.Red => 2
  | ElfColor.Yellow => 3

/-- The total score of all elves -/
def totalScore (state : ElfState) : Nat :=
  state.blue * colorScore ElfColor.Blue +
  state.red * colorScore ElfColor.Red +
  state.yellow * colorScore ElfColor.Yellow

/-- Theorem: The final color of all elves is yellow -/
theorem final_color_is_yellow (initial_state : ElfState)
  (h_initial : initial_state.blue = 7 ∧ initial_state.red = 10 ∧ initial_state.yellow = 17 ∧ initial_state.total = 34)
  (h_change : ∀ (state : ElfState), totalScore state % 3 = totalScore initial_state % 3)
  (h_final : ∃ (final_state : ElfState), (final_state.blue = final_state.total ∨ final_state.red = final_state.total ∨ final_state.yellow = final_state.total) ∧
              totalScore final_state % 3 = totalScore initial_state % 3) :
  ∃ (final_state : ElfState), final_state.yellow = final_state.total :=
sorry

end final_color_is_yellow_l353_35346


namespace logarithm_inequality_l353_35328

theorem logarithm_inequality (a b c : ℝ) (ha : a ≥ 2) (hb : b ≥ 2) (hc : c ≥ 2) :
  Real.log a ^ 2 / Real.log (b + c) + Real.log b ^ 2 / Real.log (c + a) + Real.log c ^ 2 / Real.log (a + b) ≥ 3 := by
  sorry

end logarithm_inequality_l353_35328


namespace tax_diminished_percentage_l353_35378

/-- Proves that a 12% increase in consumption and a 23.84% decrease in revenue
    implies a 32% decrease in tax rate. -/
theorem tax_diminished_percentage
  (original_tax : ℝ)
  (original_consumption : ℝ)
  (new_tax : ℝ)
  (new_consumption : ℝ)
  (original_revenue : ℝ)
  (new_revenue : ℝ)
  (h1 : original_tax > 0)
  (h2 : original_consumption > 0)
  (h3 : new_consumption = original_consumption * 1.12)
  (h4 : new_revenue = original_revenue * 0.7616)
  (h5 : original_revenue = original_tax * original_consumption)
  (h6 : new_revenue = new_tax * new_consumption) :
  new_tax = original_tax * 0.68 :=
sorry

end tax_diminished_percentage_l353_35378


namespace simplify_expression_l353_35327

theorem simplify_expression (x : ℝ) : (2*x)^5 + (4*x)*(x^4) + 5*x^3 = 36*x^5 + 5*x^3 := by
  sorry

end simplify_expression_l353_35327


namespace least_n_divisibility_l353_35305

theorem least_n_divisibility : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (∃ (k : ℕ), k ≥ 1 ∧ k ≤ n + 1 ∧ (n - 1)^2 % k = 0) ∧
  (∃ (k : ℕ), k ≥ 1 ∧ k ≤ n + 1 ∧ (n - 1)^2 % k ≠ 0) ∧
  (∀ (m : ℕ), m > 0 ∧ m < n → 
    (∀ (k : ℕ), k ≥ 1 ∧ k ≤ m + 1 → (m - 1)^2 % k = 0) ∨
    (∀ (k : ℕ), k ≥ 1 ∧ k ≤ m + 1 → (m - 1)^2 % k ≠ 0)) ∧
  n = 3 :=
by sorry

end least_n_divisibility_l353_35305


namespace diagonals_150_sided_polygon_l353_35371

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: The number of diagonals in a polygon with 150 sides is 11025 -/
theorem diagonals_150_sided_polygon : num_diagonals 150 = 11025 := by
  sorry

end diagonals_150_sided_polygon_l353_35371


namespace no_infinite_prime_sequence_l353_35342

theorem no_infinite_prime_sequence : 
  ¬ ∃ (p : ℕ → ℕ), (∀ n, Prime (p n)) ∧ 
    (∀ n, p n < p (n + 1)) ∧
    (∀ k, p (k + 1) = 2 * p k + 1 ∨ p (k + 1) = 2 * p k - 1) :=
by sorry

end no_infinite_prime_sequence_l353_35342


namespace travel_theorem_l353_35318

def travel_problem (total_time : ℝ) (foot_speed : ℝ) (bike_speed : ℝ) (foot_distance : ℝ) : Prop :=
  let foot_time : ℝ := foot_distance / foot_speed
  let bike_time : ℝ := total_time - foot_time
  let bike_distance : ℝ := bike_speed * bike_time
  let total_distance : ℝ := foot_distance + bike_distance
  total_distance = 80

theorem travel_theorem :
  travel_problem 7 8 16 32 := by
  sorry

end travel_theorem_l353_35318


namespace student_count_proof_l353_35354

theorem student_count_proof (n : ℕ) 
  (h1 : n < 600) 
  (h2 : n % 25 = 24) 
  (h3 : n % 19 = 15) : 
  n = 399 := by
sorry

end student_count_proof_l353_35354


namespace dog_food_preferences_l353_35312

theorem dog_food_preferences (total : ℕ) (carrot : ℕ) (chicken : ℕ) (both : ℕ) 
  (h1 : total = 85)
  (h2 : carrot = 12)
  (h3 : chicken = 62)
  (h4 : both = 8) :
  total - (carrot + chicken - both) = 19 := by
  sorry

end dog_food_preferences_l353_35312


namespace intersection_points_on_horizontal_line_l353_35343

/-- Given two lines parameterized by a real number s, 
    prove that their intersection points lie on a horizontal line -/
theorem intersection_points_on_horizontal_line :
  ∀ (s : ℝ), 
  ∃ (x y : ℝ), 
  (2 * x + 3 * y = 6 * s + 4) ∧ 
  (x + 2 * y = 3 * s - 1) → 
  y = -6 := by
sorry

end intersection_points_on_horizontal_line_l353_35343


namespace total_heads_count_l353_35320

/-- Proves that the total number of heads is 48 given the conditions of the problem -/
theorem total_heads_count (hens cows : ℕ) : 
  hens = 28 →
  2 * hens + 4 * cows = 136 →
  hens + cows = 48 := by
  sorry

end total_heads_count_l353_35320


namespace same_gender_probability_same_school_probability_l353_35360

structure School where
  male_count : Nat
  female_count : Nat

def total_teachers (s : School) : Nat :=
  s.male_count + s.female_count

def school_A : School :=
  { male_count := 2, female_count := 1 }

def school_B : School :=
  { male_count := 1, female_count := 2 }

def total_schools : Nat := 2

def total_all_teachers : Nat :=
  total_teachers school_A + total_teachers school_B

theorem same_gender_probability :
  (school_A.male_count * school_B.male_count + school_A.female_count * school_B.female_count) /
  (total_teachers school_A * total_teachers school_B) = 4 / 9 := by
  sorry

theorem same_school_probability :
  (Nat.choose (total_teachers school_A) 2 + Nat.choose (total_teachers school_B) 2) /
  Nat.choose total_all_teachers 2 = 2 / 5 := by
  sorry

end same_gender_probability_same_school_probability_l353_35360


namespace base7_addition_l353_35379

/-- Addition of numbers in base 7 -/
def base7_add (a b c : ℕ) : ℕ :=
  (a + b + c) % 7^3

/-- Conversion from base 7 to decimal -/
def base7_to_decimal (n : ℕ) : ℕ :=
  (n / 7^2) * 7^2 + ((n / 7) % 7) * 7 + (n % 7)

theorem base7_addition :
  base7_add (base7_to_decimal 26) (base7_to_decimal 64) (base7_to_decimal 135) = base7_to_decimal 261 :=
sorry

end base7_addition_l353_35379


namespace min_distance_to_line_l353_35339

/-- The minimum distance from the origin (0,0) to the line x + y - 4 = 0 is 2√2 -/
theorem min_distance_to_line : 
  let line := {p : ℝ × ℝ | p.1 + p.2 = 4}
  ∃ (d : ℝ), d = 2 * Real.sqrt 2 ∧ 
    ∀ (p : ℝ × ℝ), p ∈ line → Real.sqrt (p.1^2 + p.2^2) ≥ d :=
by sorry

end min_distance_to_line_l353_35339


namespace quadratic_equations_solutions_l353_35388

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, (2 * x₁^2 + x₁ - 3 = 0 ∧ x₁ = 1) ∧
                (2 * x₂^2 + x₂ - 3 = 0 ∧ x₂ = -3/2)) ∧
  (∃ y₁ y₂ : ℝ, ((y₁ - 3)^2 = 2 * y₁ * (3 - y₁) ∧ y₁ = 3) ∧
                ((y₂ - 3)^2 = 2 * y₂ * (3 - y₂) ∧ y₂ = 1)) :=
by sorry

end quadratic_equations_solutions_l353_35388


namespace geometric_sequence_tan_result_l353_35314

/-- Given a geometric sequence {a_n} with the specified conditions, 
    prove that tan((a_4 * a_6 / 3) * π) = -√3 -/
theorem geometric_sequence_tan_result (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) →  -- geometric sequence condition
  a 2 * a 3 * a 4 = -a 7^2 →                        -- given condition
  a 7^2 = 64 →                                      -- given condition
  Real.tan ((a 4 * a 6 / 3) * Real.pi) = -Real.sqrt 3 := by
  sorry

end geometric_sequence_tan_result_l353_35314


namespace second_place_limit_l353_35335

/-- Represents an election with five candidates -/
structure Election where
  totalVoters : ℕ
  nonParticipationRate : ℚ
  invalidVotes : ℕ
  winnerVoteShare : ℚ
  winnerMargin : ℕ

/-- Conditions for a valid election -/
def validElection (e : Election) : Prop :=
  e.nonParticipationRate = 15/100 ∧
  e.invalidVotes = 250 ∧
  e.winnerVoteShare = 38/100 ∧
  e.winnerMargin = 300

/-- Calculate the percentage of valid votes for the second-place candidate -/
def secondPlacePercentage (e : Election) : ℚ :=
  let validVotes := e.totalVoters * (1 - e.nonParticipationRate) - e.invalidVotes
  let secondPlaceVotes := e.totalVoters * e.winnerVoteShare - e.winnerMargin
  secondPlaceVotes / validVotes * 100

/-- Theorem stating that as the number of voters approaches infinity, 
    the percentage of valid votes for the second-place candidate approaches 44.71% -/
theorem second_place_limit (ε : ℚ) (hε : ε > 0) : 
  ∃ N : ℕ, ∀ e : Election, validElection e → e.totalVoters ≥ N → 
    |secondPlacePercentage e - 4471/100| < ε :=
sorry

end second_place_limit_l353_35335


namespace tom_reading_pages_l353_35358

/-- Tom's initial reading speed in pages per hour -/
def initial_speed : ℕ := 12

/-- The factor by which Tom increases his reading speed -/
def speed_increase : ℕ := 3

/-- The number of hours Tom reads -/
def reading_time : ℕ := 2

/-- Theorem stating the number of pages Tom can read with increased speed -/
theorem tom_reading_pages : initial_speed * speed_increase * reading_time = 72 := by
  sorry

end tom_reading_pages_l353_35358


namespace perpendicular_vectors_result_l353_35362

def a : ℝ × ℝ := (1, -2)
def b : ℝ → ℝ × ℝ := λ m ↦ (4, m)

theorem perpendicular_vectors_result (m : ℝ) 
  (h : a.1 * (b m).1 + a.2 * (b m).2 = 0) : 
  (5 : ℝ) • a - (3 : ℝ) • (b m) = (-7, -16) := by
  sorry

end perpendicular_vectors_result_l353_35362


namespace parabola_directrix_l353_35398

/-- The directrix of a parabola given by y = -3x^2 + 6x - 5 -/
theorem parabola_directrix : 
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, y = -3 * x^2 + 6 * x - 5 ↔ 4 * a * y = (x - b)^2 + c) ∧ 
    (a = -1/12 ∧ b = 1 ∧ c = -23/3) := by
  sorry

end parabola_directrix_l353_35398


namespace blocks_differing_in_three_ways_l353_35366

/-- Represents the number of options for each attribute of a block -/
structure BlockOptions :=
  (materials : Nat)
  (sizes : Nat)
  (colors : Nat)
  (shapes : Nat)

/-- Calculates the number of blocks that differ in exactly k ways from a specific block -/
def countDifferingBlocks (options : BlockOptions) (k : Nat) : Nat :=
  sorry

/-- The specific block options for our problem -/
def ourBlockOptions : BlockOptions :=
  { materials := 2, sizes := 4, colors := 4, shapes := 4 }

/-- The main theorem: 45 blocks differ in exactly 3 ways from a specific block -/
theorem blocks_differing_in_three_ways :
  countDifferingBlocks ourBlockOptions 3 = 45 := by
  sorry

end blocks_differing_in_three_ways_l353_35366


namespace greatest_integer_solution_l353_35351

theorem greatest_integer_solution : 
  ∃ (n : ℤ), (∀ (x : ℤ), 6*x^2 + 5*x - 8 < 3*x^2 - 4*x + 1 → x ≤ n) ∧ 
  (6*n^2 + 5*n - 8 < 3*n^2 - 4*n + 1) ∧ 
  n = 0 :=
by sorry

end greatest_integer_solution_l353_35351


namespace line_equation_through_two_points_l353_35381

/-- The equation of a line passing through two points -/
theorem line_equation_through_two_points 
  (x₁ y₁ x₂ y₂ x y : ℝ) : 
  (x - x₁) * (y₂ - y₁) = (y - y₁) * (x₂ - x₁) ↔ 
  (x₁ = x₂ ∧ y₁ = y₂) ∨ 
  (∃ (t : ℝ), x = x₁ + t * (x₂ - x₁) ∧ y = y₁ + t * (y₂ - y₁)) :=
by sorry

end line_equation_through_two_points_l353_35381


namespace amy_initial_amount_l353_35315

/-- The amount of money Amy had when she got to the fair -/
def initial_amount : ℕ := sorry

/-- The amount of money Amy had when she left the fair -/
def final_amount : ℕ := 11

/-- The amount of money Amy spent at the fair -/
def spent_amount : ℕ := 4

/-- Theorem: Amy had $15 when she got to the fair -/
theorem amy_initial_amount : initial_amount = 15 := by
  sorry

end amy_initial_amount_l353_35315


namespace arithmetic_geometric_sequence_properties_l353_35359

-- Define the arithmetic sequence
def a (n : ℕ) : ℚ := n

-- Define the sum of the first n terms
def S (n : ℕ) : ℚ := n * (n + 1) / 2

-- Define T_n as the sum of the first n terms of {1/S_n}
def T (n : ℕ) : ℚ := 2 * (1 - 1 / (n + 1))

theorem arithmetic_geometric_sequence_properties :
  -- The sequence {a_n} is arithmetic with common difference 1
  (∀ n : ℕ, a (n + 1) - a n = 1) ∧
  -- a_1, a_3, a_9 form a geometric sequence
  (a 3)^2 = a 1 * a 9 →
  -- Prove the following:
  (-- 1. General term formula
   (∀ n : ℕ, n ≥ 1 → a n = n) ∧
   -- 2. Sum of first n terms
   (∀ n : ℕ, n ≥ 1 → S n = n * (n + 1) / 2) ∧
   -- 3. T_n < 2
   (∀ n : ℕ, n ≥ 1 → T n < 2)) :=
by sorry

end arithmetic_geometric_sequence_properties_l353_35359


namespace wand_original_price_l353_35390

theorem wand_original_price (price_paid : ℝ) (original_price : ℝ) 
  (h1 : price_paid = 8)
  (h2 : price_paid = original_price / 8) : 
  original_price = 64 := by
  sorry

end wand_original_price_l353_35390


namespace cookies_left_after_ted_l353_35308

/-- Calculates the number of cookies left after Frank's baking and consumption, and Ted's visit -/
def cookies_left (days : ℕ) (trays_per_day : ℕ) (cookies_per_tray : ℕ) 
                 (frank_daily_consumption : ℕ) (ted_consumption : ℕ) : ℕ :=
  days * trays_per_day * cookies_per_tray - days * frank_daily_consumption - ted_consumption

/-- Proves that 134 cookies are left after 6 days of Frank's baking and Ted's visit -/
theorem cookies_left_after_ted : cookies_left 6 2 12 1 4 = 134 := by
  sorry

end cookies_left_after_ted_l353_35308


namespace quadratic_roots_l353_35306

-- Define the quadratic equation
def quadratic (k : ℝ) (x : ℝ) : ℝ := x^2 - (k + 2) * x + 2 * k

theorem quadratic_roots (k : ℝ) :
  (quadratic k 1 = 0 → k = 1 ∧ quadratic k 2 = 0) ∧
  (∃ x y : ℝ, x ≠ y ∧ quadratic k x = 0 ∧ quadratic k y = 0) :=
sorry

end quadratic_roots_l353_35306


namespace prob_king_queen_standard_deck_l353_35302

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (ranks : Nat)
  (suits : Nat)
  (red_cards : Nat)
  (black_cards : Nat)

/-- A standard 52-card deck -/
def standard_deck : Deck :=
  { cards := 52,
    ranks := 13,
    suits := 4,
    red_cards := 26,
    black_cards := 26 }

/-- The number of Kings in a standard deck -/
def num_kings (d : Deck) : Nat := d.suits

/-- The number of Queens in a standard deck -/
def num_queens (d : Deck) : Nat := d.suits

/-- The probability of drawing a King then a Queen from a shuffled deck -/
def prob_king_queen (d : Deck) : Rat :=
  (num_kings d * num_queens d) / (d.cards * (d.cards - 1))

/-- Theorem: The probability of drawing a King then a Queen from a standard 52-card deck is 4/663 -/
theorem prob_king_queen_standard_deck :
  prob_king_queen standard_deck = 4 / 663 := by
  sorry

end prob_king_queen_standard_deck_l353_35302


namespace cassidy_grounding_l353_35348

/-- Calculates the number of extra days grounded per grade below B -/
def extraDaysPerGrade (totalDays : ℕ) (baseDays : ℕ) (gradesBelowB : ℕ) : ℕ :=
  if gradesBelowB = 0 then 0 else (totalDays - baseDays) / gradesBelowB

theorem cassidy_grounding (totalDays : ℕ) (baseDays : ℕ) (gradesBelowB : ℕ) 
  (h1 : totalDays = 26)
  (h2 : baseDays = 14)
  (h3 : gradesBelowB = 4) :
  extraDaysPerGrade totalDays baseDays gradesBelowB = 3 := by
  sorry

#eval extraDaysPerGrade 26 14 4

end cassidy_grounding_l353_35348


namespace largest_number_divisible_by_88_has_4_digits_l353_35382

theorem largest_number_divisible_by_88_has_4_digits :
  let n : ℕ := 9944
  (∀ m : ℕ, m > n → m % 88 ≠ 0 ∨ (String.length (toString m) > String.length (toString n))) →
  n % 88 = 0 →
  String.length (toString n) = 4 :=
by sorry

end largest_number_divisible_by_88_has_4_digits_l353_35382


namespace trajectory_and_equilateral_triangle_l353_35391

-- Define the points
def H : ℝ × ℝ := (-3, 0)
def T : ℝ × ℝ := (-1, 0)

-- Define the trajectory C
def C : Set (ℝ × ℝ) := {(x, y) | y^2 = 4*x ∧ x > 0}

-- Define the conditions
def on_y_axis (P : ℝ × ℝ) : Prop := P.1 = 0
def on_positive_x_axis (Q : ℝ × ℝ) : Prop := Q.2 = 0 ∧ Q.1 > 0
def on_line (P Q M : ℝ × ℝ) : Prop := ∃ t : ℝ, M = (1 - t) • P + t • Q

def orthogonal (HP PM : ℝ × ℝ) : Prop := HP.1 * PM.1 + HP.2 * PM.2 = 0
def vector_ratio (PM MQ : ℝ × ℝ) : Prop := PM = (-3/2) • MQ

-- Main theorem
theorem trajectory_and_equilateral_triangle 
  (P Q M : ℝ × ℝ) 
  (hP : on_y_axis P) 
  (hQ : on_positive_x_axis Q) 
  (hM : on_line P Q M) 
  (hOrth : orthogonal (H.1 - P.1, H.2 - P.2) (M.1 - P.1, M.2 - P.2))
  (hRatio : vector_ratio (M.1 - P.1, M.2 - P.2) (Q.1 - M.1, Q.2 - M.2)) :
  (M ∈ C) ∧ 
  (∀ (A B : ℝ × ℝ) (l : Set (ℝ × ℝ)) (E : ℝ × ℝ),
    (A ∈ C ∧ B ∈ C ∧ T ∈ l ∧ A ∈ l ∧ B ∈ l ∧ E.2 = 0) →
    (∃ (x₀ : ℝ), E.1 = x₀ ∧ 
      (norm (A - E) = norm (B - E) ∧ norm (A - E) = norm (A - B)) →
      x₀ = 11/3)) :=
sorry

end trajectory_and_equilateral_triangle_l353_35391


namespace negative_third_greater_than_negative_half_l353_35376

theorem negative_third_greater_than_negative_half : -1/3 > -1/2 := by
  sorry

end negative_third_greater_than_negative_half_l353_35376


namespace probability_theorem_l353_35377

def club_sizes : List Nat := [6, 9, 11, 13]

def probability_select_officers (sizes : List Nat) : Rat :=
  let total_probability := sizes.map (fun n => 1 / Nat.choose n 3)
  (1 / sizes.length) * total_probability.sum

theorem probability_theorem :
  probability_select_officers club_sizes = 905 / 55440 := by
  sorry

end probability_theorem_l353_35377


namespace money_left_after_debts_l353_35334

def lottery_winnings : ℕ := 100
def payment_to_colin : ℕ := 20

def payment_to_helen (colin_payment : ℕ) : ℕ := 2 * colin_payment

def payment_to_benedict (helen_payment : ℕ) : ℕ := helen_payment / 2

def total_payments (colin : ℕ) (helen : ℕ) (benedict : ℕ) : ℕ := colin + helen + benedict

theorem money_left_after_debts :
  lottery_winnings - total_payments payment_to_colin (payment_to_helen payment_to_colin) (payment_to_benedict (payment_to_helen payment_to_colin)) = 20 := by
  sorry

end money_left_after_debts_l353_35334


namespace student_claim_incorrect_l353_35394

theorem student_claim_incorrect (m n : ℤ) (hn : 0 < n) (hn_bound : n ≤ 100) :
  ¬ (167 * n ≤ 1000 * m ∧ 1000 * m < 168 * n) := by
  sorry

end student_claim_incorrect_l353_35394


namespace smallest_club_size_club_size_exists_l353_35389

theorem smallest_club_size (n : ℕ) : 
  (n % 6 = 1) ∧ (n % 8 = 2) ∧ (n % 9 = 3) → n ≥ 343 :=
by sorry

theorem club_size_exists : 
  ∃ n : ℕ, (n % 6 = 1) ∧ (n % 8 = 2) ∧ (n % 9 = 3) ∧ n = 343 :=
by sorry

end smallest_club_size_club_size_exists_l353_35389


namespace bill_sunday_miles_l353_35345

/-- Proves that Bill ran 9 miles on Sunday given the problem conditions --/
theorem bill_sunday_miles : ℕ → ℕ → Prop :=
  fun (bill_saturday : ℕ) (bill_sunday : ℕ) =>
    let julia_sunday := 2 * bill_sunday
    bill_saturday + bill_sunday + julia_sunday = 32 ∧
    bill_sunday = bill_saturday + 4 →
    bill_sunday = 9

/-- Proof of the theorem --/
lemma bill_sunday_miles_proof : ∃ (bill_saturday : ℕ), bill_sunday_miles bill_saturday (bill_saturday + 4) :=
  sorry

end bill_sunday_miles_l353_35345


namespace scalene_triangle_area_l353_35301

/-- Given an outer triangle enclosing a regular hexagon, prove the area of one scalene triangle -/
theorem scalene_triangle_area 
  (outer_triangle_area : ℝ) 
  (hexagon_area : ℝ) 
  (num_scalene_triangles : ℕ)
  (h1 : outer_triangle_area = 25)
  (h2 : hexagon_area = 4)
  (h3 : num_scalene_triangles = 6) :
  (outer_triangle_area - hexagon_area) / num_scalene_triangles = 3.5 := by
sorry

end scalene_triangle_area_l353_35301


namespace coupon1_best_at_229_95_l353_35369

def coupon1_discount (price : ℝ) : ℝ := 0.15 * price

def coupon2_discount (price : ℝ) : ℝ := 30

def coupon3_discount (price : ℝ) : ℝ := 0.2 * (price - 150)

def price_list : List ℝ := [199.95, 229.95, 249.95, 289.95, 319.95]

theorem coupon1_best_at_229_95 :
  let p := 229.95
  (p ≥ 50) ∧
  (p ≥ 150) ∧
  (coupon1_discount p > coupon2_discount p) ∧
  (coupon1_discount p > coupon3_discount p) ∧
  (∀ q ∈ price_list, q < p → 
    coupon1_discount q ≤ coupon2_discount q ∨ 
    coupon1_discount q ≤ coupon3_discount q) :=
by sorry

end coupon1_best_at_229_95_l353_35369


namespace folded_rectangle_perimeter_ratio_l353_35333

/-- Represents a rectangle with given length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

theorem folded_rectangle_perimeter_ratio :
  let original := Rectangle.mk 8 4
  let folded := Rectangle.mk 4 2
  (perimeter folded) / (perimeter original) = 1/2 := by sorry

end folded_rectangle_perimeter_ratio_l353_35333


namespace pet_shop_legs_l353_35338

/-- The number of legs for each animal type --/
def bird_legs : ℕ := 2
def dog_legs : ℕ := 4
def snake_legs : ℕ := 0
def spider_legs : ℕ := 8

/-- The number of each animal type --/
def num_birds : ℕ := 3
def num_dogs : ℕ := 5
def num_snakes : ℕ := 4
def num_spiders : ℕ := 1

/-- The total number of legs in the pet shop --/
def total_legs : ℕ := 
  num_birds * bird_legs + 
  num_dogs * dog_legs + 
  num_snakes * snake_legs + 
  num_spiders * spider_legs

theorem pet_shop_legs : total_legs = 34 := by
  sorry

end pet_shop_legs_l353_35338


namespace prize_probabilities_l353_35317

/-- Represents the outcome of drawing a ball from a box -/
inductive BallColor
| Red
| White

/-- Represents a box with red and white balls -/
structure Box where
  red : Nat
  white : Nat

/-- Probability of drawing a red ball from a box -/
def probRed (box : Box) : Rat :=
  box.red / (box.red + box.white)

/-- Probability of winning first prize in one draw -/
def probFirstPrize (boxA boxB : Box) : Rat :=
  probRed boxA * probRed boxB

/-- Probability of winning second prize in one draw -/
def probSecondPrize (boxA boxB : Box) : Rat :=
  probRed boxA * (1 - probRed boxB) + (1 - probRed boxA) * probRed boxB

/-- Probability of winning a prize in one draw -/
def probWinPrize (boxA boxB : Box) : Rat :=
  probFirstPrize boxA boxB + probSecondPrize boxA boxB

/-- Expected number of first prizes in n draws -/
def expectedFirstPrizes (boxA boxB : Box) (n : Nat) : Rat :=
  n * probFirstPrize boxA boxB

theorem prize_probabilities (boxA boxB : Box) :
  boxA.red = 4 ∧ boxA.white = 6 ∧ boxB.red = 5 ∧ boxB.white = 5 →
  probWinPrize boxA boxB = 7/10 ∧ expectedFirstPrizes boxA boxB 3 = 3/5 := by
  sorry


end prize_probabilities_l353_35317


namespace star_calculation_l353_35392

def star (a b : ℝ) : ℝ := a * b + a + b

theorem star_calculation : star 1 2 + star 2 3 = 16 := by
  sorry

end star_calculation_l353_35392


namespace inverse_variation_problem_l353_35304

/-- The inverse relationship between y^5 and z^(1/5) -/
def inverse_relation (y z : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ y^5 * z^(1/5) = k

theorem inverse_variation_problem (y₁ y₂ z₁ z₂ : ℝ) 
  (h1 : inverse_relation y₁ z₁)
  (h2 : inverse_relation y₂ z₂)
  (h3 : y₁ = 3)
  (h4 : z₁ = 8)
  (h5 : y₂ = 6) :
  z₂ = 1 / 1048576 := by
sorry

end inverse_variation_problem_l353_35304


namespace sum_a_b_equals_21_over_8_l353_35323

/-- Operation ⊕ defined for real numbers -/
def circle_plus (x y : ℝ) : ℝ := x + 2*y + 3

/-- Theorem stating the result of a + b given the conditions -/
theorem sum_a_b_equals_21_over_8 (a b : ℝ) 
  (h : (circle_plus (circle_plus (a^3) (a^2)) a) = (circle_plus (a^3) (circle_plus (a^2) a)) ∧ 
       (circle_plus (circle_plus (a^3) (a^2)) a) = b) : 
  a + b = 21/8 := by sorry

end sum_a_b_equals_21_over_8_l353_35323


namespace smallest_high_efficiency_l353_35331

def efficiency (n : ℕ) : ℚ :=
  (n - (Nat.totient n)) / n

theorem smallest_high_efficiency : 
  ∀ m : ℕ, m < 30030 → efficiency m ≤ 4/5 ∧ efficiency 30030 > 4/5 :=
sorry

end smallest_high_efficiency_l353_35331


namespace root_properties_l353_35347

theorem root_properties (a b : ℝ) :
  (a - b)^3 + 3*a*b*(a - b) + b^3 - a^3 = 0 ∧
  (∀ a : ℝ, (a - 1)^3 - a*(a - 1)^2 + 1 = 0 ↔ a = 0 ∨ a = 2) :=
by sorry

end root_properties_l353_35347


namespace floor_expression_equals_eight_l353_35322

theorem floor_expression_equals_eight :
  ⌊(2021^3 : ℝ) / (2019 * 2020) - (2019^3 : ℝ) / (2020 * 2021)⌋ = 8 := by
  sorry

#check floor_expression_equals_eight

end floor_expression_equals_eight_l353_35322


namespace negation_equivalence_l353_35309

theorem negation_equivalence (x : ℝ) :
  ¬(x^2 - x ≥ 0 → x > 2) ↔ (x^2 - x < 0 → x ≤ 2) := by
  sorry

end negation_equivalence_l353_35309


namespace employees_using_public_transportation_l353_35368

theorem employees_using_public_transportation
  (total_employees : ℕ)
  (drive_percentage : ℚ)
  (public_transport_fraction : ℚ)
  (h1 : total_employees = 100)
  (h2 : drive_percentage = 60 / 100)
  (h3 : public_transport_fraction = 1 / 2) :
  (total_employees : ℚ) * (1 - drive_percentage) * public_transport_fraction = 20 := by
  sorry

end employees_using_public_transportation_l353_35368


namespace jessie_weight_loss_l353_35321

/-- Jessie's weight loss journey -/
theorem jessie_weight_loss (current_weight weight_lost : ℕ) 
  (h1 : current_weight = 66)
  (h2 : weight_lost = 126) : 
  current_weight + weight_lost = 192 := by
  sorry

end jessie_weight_loss_l353_35321


namespace total_spent_equals_sum_of_games_l353_35393

/-- The total amount Tom spent on video games -/
def total_spent : ℝ := 35.52

/-- The cost of the football game -/
def football_cost : ℝ := 14.02

/-- The cost of the strategy game -/
def strategy_cost : ℝ := 9.46

/-- The cost of the Batman game -/
def batman_cost : ℝ := 12.04

/-- Theorem: The total amount Tom spent on video games is equal to the sum of the costs of the football game, strategy game, and Batman game -/
theorem total_spent_equals_sum_of_games : 
  total_spent = football_cost + strategy_cost + batman_cost := by
  sorry

end total_spent_equals_sum_of_games_l353_35393


namespace min_value_of_f_l353_35341

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := ∫ t, (2 * t - 4)

-- State the theorem
theorem min_value_of_f :
  ∃ (min : ℝ), min = -4 ∧ ∀ x ∈ Set.Icc (-1) 3, f x ≥ min :=
sorry

end min_value_of_f_l353_35341


namespace smallest_perfect_square_divisible_by_2_3_5_l353_35386

def is_divisible_by (n m : ℕ) : Prop := m ∣ n

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem smallest_perfect_square_divisible_by_2_3_5 :
  ∃ n : ℕ, n > 0 ∧ 
    is_perfect_square n ∧
    is_divisible_by n 2 ∧
    is_divisible_by n 3 ∧
    is_divisible_by n 5 ∧
    (∀ m : ℕ, m > 0 ∧ 
      is_perfect_square m ∧
      is_divisible_by m 2 ∧
      is_divisible_by m 3 ∧
      is_divisible_by m 5 →
      n ≤ m) ∧
    n = 900 := by
  sorry

end smallest_perfect_square_divisible_by_2_3_5_l353_35386


namespace changsha_tourism_l353_35340

/-- The number of visitors (in millions) to Changsha during May Day holiday in 2021 -/
def visitors_2021 : ℝ := 2

/-- The number of visitors (in millions) to Changsha during May Day holiday in 2023 -/
def visitors_2023 : ℝ := 2.88

/-- The amount spent on Youlan Latte -/
def spent_youlan : ℝ := 216

/-- The amount spent on Shengsheng Oolong -/
def spent_oolong : ℝ := 96

/-- The price difference between Youlan Latte and Shengsheng Oolong -/
def price_difference : ℝ := 2

theorem changsha_tourism (r x : ℝ) : 
  ((1 + r)^2 = visitors_2023 / visitors_2021) ∧ 
  (spent_youlan / x = 2 * spent_oolong / (x - price_difference)) → 
  (r = 0.2 ∧ x = 18) := by sorry

end changsha_tourism_l353_35340


namespace average_daily_sales_l353_35325

/-- Represents the sales data for a baker's pastry shop over a week. -/
structure BakerSales where
  weekdayPrice : ℕ
  weekendPrice : ℕ
  mondaySales : ℕ
  weekdayIncrease : ℕ
  weekendIncrease : ℕ

/-- Calculates the total pastries sold in a week based on the given sales data. -/
def totalWeeklySales (sales : BakerSales) : ℕ :=
  let tue := sales.mondaySales + sales.weekdayIncrease
  let wed := tue + sales.weekdayIncrease
  let thu := wed + sales.weekdayIncrease
  let fri := thu + sales.weekdayIncrease
  let sat := fri + sales.weekendIncrease
  let sun := sat + sales.weekendIncrease
  sales.mondaySales + tue + wed + thu + fri + sat + sun

/-- Theorem stating that the average daily sales for the given conditions is 59/7. -/
theorem average_daily_sales (sales : BakerSales)
    (h1 : sales.weekdayPrice = 5)
    (h2 : sales.weekendPrice = 6)
    (h3 : sales.mondaySales = 2)
    (h4 : sales.weekdayIncrease = 2)
    (h5 : sales.weekendIncrease = 3) :
    (totalWeeklySales sales : ℚ) / 7 = 59 / 7 := by
  sorry

end average_daily_sales_l353_35325


namespace complete_square_formula_not_complete_square_A_not_complete_square_B_not_complete_square_C_l353_35383

theorem complete_square_formula (a b : ℝ) : 
  (a - b) * (b - a) = -(a - b)^2 :=
sorry

theorem not_complete_square_A (a b : ℝ) :
  (a - b) * (a + b) = a^2 - b^2 :=
sorry

theorem not_complete_square_B (a b : ℝ) :
  -(a + b) * (b - a) = a^2 - b^2 :=
sorry

theorem not_complete_square_C (a b : ℝ) :
  (a + b) * (b - a) = b^2 - a^2 :=
sorry

end complete_square_formula_not_complete_square_A_not_complete_square_B_not_complete_square_C_l353_35383


namespace imaginary_part_of_z_l353_35363

theorem imaginary_part_of_z (z : ℂ) : z = -2 * Complex.I * (-1 + Real.sqrt 3 * Complex.I) → z.im = 2 := by
  sorry

end imaginary_part_of_z_l353_35363


namespace cosine_amplitude_l353_35313

theorem cosine_amplitude (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x, a * Real.cos (b * x) ≤ 3) ∧ (∃ x, a * Real.cos (b * x) = 3) → a = 3 := by
  sorry

end cosine_amplitude_l353_35313


namespace problem_solution_l353_35352

theorem problem_solution (a : ℝ) (h : a^2 - 4*a + 3 = 0) :
  (9 - 3*a) / (2*a - 4) / (a + 2 - 5 / (a - 2)) = -3/8 :=
by sorry

end problem_solution_l353_35352


namespace sum_of_coefficients_zero_l353_35350

/-- A parabola with equation y = ax^2 + bx + c, vertex (3, 4), and x-intercepts at (1, 0) and (5, 0) -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  vertex_x : ℝ := 3
  vertex_y : ℝ := 4
  intercept1_x : ℝ := 1
  intercept2_x : ℝ := 5

/-- The parabola satisfies its vertex condition -/
axiom vertex_condition (p : Parabola) : p.vertex_y = p.a * p.vertex_x^2 + p.b * p.vertex_x + p.c

/-- The parabola satisfies its first x-intercept condition -/
axiom intercept1_condition (p : Parabola) : 0 = p.a * p.intercept1_x^2 + p.b * p.intercept1_x + p.c

/-- The parabola satisfies its second x-intercept condition -/
axiom intercept2_condition (p : Parabola) : 0 = p.a * p.intercept2_x^2 + p.b * p.intercept2_x + p.c

/-- The sum of coefficients a, b, and c is zero for a parabola satisfying the given conditions -/
theorem sum_of_coefficients_zero (p : Parabola) : p.a + p.b + p.c = 0 := by
  sorry

end sum_of_coefficients_zero_l353_35350


namespace problem_solution_l353_35311

theorem problem_solution : 
  ((-1/2 - 1/3 + 3/4) * (-60) = 5) ∧ 
  ((-1)^4 - 1/6 * (3 - (-3)^2) = 2) := by
sorry

end problem_solution_l353_35311


namespace sqrt_5_simplest_l353_35385

def is_simplest_sqrt (x : ℝ) : Prop :=
  ∀ y : ℝ, y > 0 → x = Real.sqrt y → ¬∃ (a b : ℚ), y = a / b ∧ b ≠ 1

theorem sqrt_5_simplest :
  is_simplest_sqrt (Real.sqrt 5) ∧
  ¬is_simplest_sqrt (Real.sqrt 2.5) ∧
  ¬is_simplest_sqrt (Real.sqrt 8) ∧
  ¬is_simplest_sqrt (Real.sqrt (1/3)) :=
sorry

end sqrt_5_simplest_l353_35385


namespace expression_simplification_l353_35303

theorem expression_simplification (x y : ℝ) :
  (x + 2*y) * (x - 2*y) - x * (x + 3*y) = -4*y^2 - 3*x*y ∧
  (x - 1 - 3/(x + 1)) / ((x^2 - 4*x + 4) / (x + 1)) = (x + 2) / (x - 2) :=
by sorry

end expression_simplification_l353_35303


namespace power_sum_value_l353_35349

theorem power_sum_value (a : ℝ) (x y : ℝ) (h1 : a^x = 2) (h2 : a^y = 3) : a^(x+y) = 6 := by
  sorry

end power_sum_value_l353_35349


namespace solution_product_l353_35336

-- Define the equation
def equation (x : ℝ) : Prop :=
  (x - 3) * (3 * x + 7) = x^2 - 12 * x + 27

-- State the theorem
theorem solution_product (a b : ℝ) : 
  a ≠ b ∧ equation a ∧ equation b → (a + 2) * (b + 2) = -30 := by
  sorry

end solution_product_l353_35336


namespace exists_rational_with_prime_multiples_l353_35373

theorem exists_rational_with_prime_multiples : ∃ x : ℚ, 
  (Nat.Prime (Int.natAbs (Int.floor (10 * x)))) ∧ 
  (Nat.Prime (Int.natAbs (Int.floor (15 * x)))) := by
  sorry

end exists_rational_with_prime_multiples_l353_35373


namespace hostel_expenditure_l353_35332

/-- Calculates the new total expenditure of a hostel after accommodating additional students --/
def new_total_expenditure (initial_students : ℕ) (additional_students : ℕ) (average_decrease : ℕ) (total_increase : ℕ) : ℕ :=
  let new_students := initial_students + additional_students
  let original_average := (total_increase + new_students * average_decrease) / (new_students - initial_students)
  new_students * (original_average - average_decrease)

/-- Theorem stating that the new total expenditure is 5400 rupees --/
theorem hostel_expenditure :
  new_total_expenditure 100 20 5 400 = 5400 := by
  sorry

end hostel_expenditure_l353_35332


namespace trapezoid_area_l353_35353

/-- Given a trapezoid with bases a and b, prove that its area is 150 -/
theorem trapezoid_area (a b : ℝ) : 
  ((a + b) / 2) * ((a - b) / 2) = 25 →
  ∃ h : ℝ, h = 3 * (a - b) →
  (1 / 2) * (a + b) * h = 150 :=
by sorry

end trapezoid_area_l353_35353


namespace min_voters_to_win_is_24_l353_35310

/-- Represents the voting structure and outcome of a giraffe beauty contest. -/
structure GiraffeContest where
  total_voters : Nat
  num_districts : Nat
  sections_per_district : Nat
  voters_per_section : Nat
  (total_voters_eq : total_voters = num_districts * sections_per_district * voters_per_section)
  (num_districts_eq : num_districts = 5)
  (sections_per_district_eq : sections_per_district = 7)
  (voters_per_section_eq : voters_per_section = 3)

/-- Calculates the minimum number of voters required to win the contest. -/
def min_voters_to_win (contest : GiraffeContest) : Nat :=
  let districts_to_win := contest.num_districts / 2 + 1
  let sections_to_win := contest.sections_per_district / 2 + 1
  let voters_to_win_section := contest.voters_per_section / 2 + 1
  districts_to_win * sections_to_win * voters_to_win_section

/-- Theorem stating that the minimum number of voters required to win the contest is 24. -/
theorem min_voters_to_win_is_24 (contest : GiraffeContest) :
  min_voters_to_win contest = 24 := by
  sorry

#eval min_voters_to_win {
  total_voters := 105,
  num_districts := 5,
  sections_per_district := 7,
  voters_per_section := 3,
  total_voters_eq := rfl,
  num_districts_eq := rfl,
  sections_per_district_eq := rfl,
  voters_per_section_eq := rfl
}

end min_voters_to_win_is_24_l353_35310


namespace sec_330_deg_l353_35326

/-- Prove that sec 330° = 2√3 / 3 -/
theorem sec_330_deg : 
  let sec : Real → Real := λ θ ↦ 1 / Real.cos θ
  let θ : Real := 330 * Real.pi / 180
  sec θ = 2 * Real.sqrt 3 / 3 := by
  sorry

end sec_330_deg_l353_35326


namespace marker_selection_combinations_l353_35365

theorem marker_selection_combinations : ∀ n r : ℕ, 
  n = 15 → r = 5 → (n.choose r) = 3003 := by
  sorry

end marker_selection_combinations_l353_35365


namespace exactly_one_correct_proposition_l353_35374

open Real

theorem exactly_one_correct_proposition : ∃! n : Nat, n = 1 ∧
  (¬ (∀ x : ℝ, (x^2 < 1 → -1 < x ∧ x < 1) ↔ ((x > 1 ∨ x < -1) → x^2 > 1))) ∧
  (¬ ((∀ x : ℝ, sin x ≤ 1) ∧ (∀ a b : ℝ, a < b → a^2 < b^2))) ∧
  ((∀ x : ℝ, ¬(x^2 - x > 0)) ↔ (∀ x : ℝ, x^2 - x ≤ 0)) ∧
  (¬ (∀ x : ℝ, x^2 > 4 → x > 2)) :=
by sorry

end exactly_one_correct_proposition_l353_35374


namespace units_digit_characteristic_l353_35337

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- A predicate to check if a natural number is even -/
def isEven (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem units_digit_characteristic (p : ℕ) 
  (h1 : p > 0) 
  (h2 : isEven p) 
  (h3 : unitsDigit (p^3) - unitsDigit (p^2) = 0)
  (h4 : unitsDigit (p + 4) = 0) : 
  unitsDigit p = 6 := by
  sorry

end units_digit_characteristic_l353_35337


namespace equation_solution_l353_35344

theorem equation_solution : 
  let f : ℝ → ℝ := λ x => 1/3 + 1/x + 1/(x^2)
  ∃ x₁ x₂ : ℝ, x₁ = (3 + Real.sqrt 33) / 4 ∧ 
              x₂ = (3 - Real.sqrt 33) / 4 ∧ 
              f x₁ = 1 ∧ 
              f x₂ = 1 ∧ 
              ∀ x : ℝ, f x = 1 → x = x₁ ∨ x = x₂ := by
  sorry

end equation_solution_l353_35344


namespace sqrt_16_minus_pi_minus_3_pow_0_l353_35307

theorem sqrt_16_minus_pi_minus_3_pow_0 : Real.sqrt 16 - (π - 3)^0 = 3 := by sorry

end sqrt_16_minus_pi_minus_3_pow_0_l353_35307


namespace basketball_lineup_count_l353_35380

/-- The number of players in the basketball team -/
def total_players : ℕ := 18

/-- The number of players in a lineup excluding the point guard -/
def lineup_size : ℕ := 7

/-- The number of different lineups that can be chosen -/
def number_of_lineups : ℕ := total_players * (Nat.choose (total_players - 1) lineup_size)

/-- Theorem stating the number of different lineups -/
theorem basketball_lineup_count : number_of_lineups = 349464 := by
  sorry

end basketball_lineup_count_l353_35380


namespace quadratic_inequality_always_positive_l353_35356

theorem quadratic_inequality_always_positive (a : ℝ) :
  (∀ x : ℝ, x^2 - 2*x + a > 0) → a > 1 := by
  sorry

end quadratic_inequality_always_positive_l353_35356


namespace annie_village_trick_or_treat_l353_35300

/-- The number of blocks in Annie's village -/
def num_blocks : ℕ := 9

/-- The number of children on each block -/
def children_per_block : ℕ := 6

/-- The total number of children going trick or treating in Annie's village -/
def total_children : ℕ := num_blocks * children_per_block

theorem annie_village_trick_or_treat : total_children = 54 := by
  sorry

end annie_village_trick_or_treat_l353_35300


namespace power_equality_l353_35319

theorem power_equality (q : ℕ) : 16^10 = 4^q → q = 20 := by
  sorry

end power_equality_l353_35319


namespace circumcircle_radius_is_13_l353_35372

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  /-- Ratio of the shorter base to the longer base -/
  base_ratio : ℚ
  /-- Height of the trapezoid -/
  height : ℝ
  /-- The midline of the trapezoid equals its height -/
  midline_eq_height : True

/-- Calculate the radius of the circumcircle of an isosceles trapezoid -/
def circumcircle_radius (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem stating that for a trapezoid with given properties, its circumcircle radius is 13 -/
theorem circumcircle_radius_is_13 (t : IsoscelesTrapezoid) 
  (h1 : t.base_ratio = 5 / 12)
  (h2 : t.height = 17) : 
  circumcircle_radius t = 13 := by
  sorry

end circumcircle_radius_is_13_l353_35372


namespace jake_shooting_improvement_l353_35395

theorem jake_shooting_improvement (initial_shots : ℕ) (additional_shots : ℕ) 
  (initial_percentage : ℚ) (final_percentage : ℚ) :
  initial_shots = 30 →
  additional_shots = 10 →
  initial_percentage = 60 / 100 →
  final_percentage = 62 / 100 →
  ∃ (last_successful_shots : ℕ),
    last_successful_shots = 7 ∧
    (initial_percentage * initial_shots).floor + last_successful_shots = 
      (final_percentage * (initial_shots + additional_shots)).floor :=
by sorry

end jake_shooting_improvement_l353_35395


namespace at_least_one_negative_l353_35399

theorem at_least_one_negative (a b c d : ℝ) 
  (sum_eq_one : a + b = 1 ∧ c + d = 1) 
  (product_gt_one : a * c + b * d > 1) : 
  a < 0 ∨ b < 0 ∨ c < 0 ∨ d < 0 := by
  sorry

end at_least_one_negative_l353_35399


namespace f_three_pow_ge_f_two_pow_l353_35375

/-- A quadratic function f(x) = ax^2 + bx + c with a > 0 and symmetric about x = 1 -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The theorem stating that f(3^x) ≥ f(2^x) for all x ∈ ℝ -/
theorem f_three_pow_ge_f_two_pow (a b c : ℝ) (h_a : a > 0) 
  (h_sym : ∀ x, f a b c (1 - x) = f a b c (1 + x)) :
  ∀ x, f a b c (3^x) ≥ f a b c (2^x) := by
  sorry

end f_three_pow_ge_f_two_pow_l353_35375


namespace unique_f_3_l353_35316

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  f 2 = 3 ∧ ∀ x y : ℝ, f (x^2 + y^2) = (x + y) * (f x - f y)

/-- The main theorem -/
theorem unique_f_3 (f : ℝ → ℝ) (hf : special_function f) : f 3 = 3 * Real.sqrt 3 := by
  sorry

end unique_f_3_l353_35316


namespace sarah_hair_products_usage_l353_35367

/-- Given Sarah's daily shampoo and conditioner usage, calculate the total volume used in 14 days -/
theorem sarah_hair_products_usage 
  (shampoo_daily : ℝ) 
  (conditioner_daily : ℝ) 
  (h1 : shampoo_daily = 1) 
  (h2 : conditioner_daily = shampoo_daily / 2) 
  (days : ℕ) 
  (h3 : days = 14) : 
  shampoo_daily * days + conditioner_daily * days = 21 := by
  sorry


end sarah_hair_products_usage_l353_35367


namespace distance_city_AC_l353_35397

/-- The distance between two cities given travel times and speeds -/
theorem distance_city_AC (time_eddy time_freddy : ℝ) (distance_AB : ℝ) (speed_ratio : ℝ) 
  (h1 : time_eddy = 3)
  (h2 : time_freddy = 4)
  (h3 : distance_AB = 480)
  (h4 : speed_ratio = 2.1333333333333333)
  (h5 : speed_ratio = (distance_AB / time_eddy) / ((distance_AB / time_eddy) / speed_ratio)) :
  (distance_AB / time_eddy) / speed_ratio * time_freddy = 300 := by
  sorry

#eval (480 / 3) / 2.1333333333333333 * 4

end distance_city_AC_l353_35397


namespace candy_pencils_count_l353_35384

/-- The number of pencils Candy has -/
def candy_pencils : ℕ := 9

/-- The number of pencils Caleb has -/
def caleb_pencils : ℕ := 2 * candy_pencils - 3

/-- The original number of pencils Calen had -/
def calen_original_pencils : ℕ := caleb_pencils + 5

/-- The number of pencils Calen lost -/
def calen_lost_pencils : ℕ := 10

/-- The number of pencils Calen has now -/
def calen_current_pencils : ℕ := 10

theorem candy_pencils_count :
  calen_original_pencils - calen_lost_pencils = calen_current_pencils :=
by sorry

end candy_pencils_count_l353_35384


namespace bug_return_probability_l353_35357

/-- Probability of a bug returning to the starting vertex of a regular tetrahedron after n steps -/
def P (n : ℕ) : ℚ :=
  if n = 0 then 1
  else (1 - P (n - 1)) / 3

/-- The regular tetrahedron has edge length 1 and the bug starts at vertex A -/
theorem bug_return_probability :
  P 10 = 4921 / 59049 :=
sorry

end bug_return_probability_l353_35357


namespace greatest_fraction_l353_35355

theorem greatest_fraction : 
  let f1 := 44444 / 55555
  let f2 := 5555 / 6666
  let f3 := 666 / 777
  let f4 := 77 / 88
  let f5 := 8 / 9
  (f5 > f1) ∧ (f5 > f2) ∧ (f5 > f3) ∧ (f5 > f4) := by
  sorry

end greatest_fraction_l353_35355


namespace smallest_fraction_above_four_fifths_l353_35387

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem smallest_fraction_above_four_fifths :
  ∀ (a b : ℕ), is_two_digit a → is_two_digit b → (a : ℚ) / b > 4 / 5 → Nat.gcd a b = 1 →
  (77 : ℚ) / 96 ≤ (a : ℚ) / b :=
sorry

end smallest_fraction_above_four_fifths_l353_35387


namespace intersection_of_A_and_B_l353_35370

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {0, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {0, 2} := by
  sorry

end intersection_of_A_and_B_l353_35370


namespace parallel_lines_condition_l353_35324

/-- Two lines are parallel if and only if their slopes are equal and they are not the same line -/
def are_parallel (m₁ n₁ c₁ m₂ n₂ c₂ : ℝ) : Prop :=
  m₁ * n₂ = m₂ * n₁ ∧ (m₁, n₁, c₁) ≠ (m₂, n₂, c₂)

/-- The theorem states that a = 3 is a necessary and sufficient condition for the given lines to be parallel -/
theorem parallel_lines_condition (a : ℝ) :
  are_parallel a 2 (3*a) 3 (a-1) (a-7) ↔ a = 3 := by sorry

end parallel_lines_condition_l353_35324
