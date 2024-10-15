import Mathlib

namespace NUMINAMATH_CALUDE_correct_statements_count_l2873_287351

-- Define a structure for sampling statements
structure SamplingStatement :=
  (id : Nat)
  (content : String)
  (isCorrect : Bool)

-- Define the four statements
def statement1 : SamplingStatement :=
  { id := 1
  , content := "When the total number of individuals in a population is not large, it is appropriate to use simple random sampling"
  , isCorrect := true }

def statement2 : SamplingStatement :=
  { id := 2
  , content := "In systematic sampling, after the population is divided evenly, simple random sampling is used in each part"
  , isCorrect := false }

def statement3 : SamplingStatement :=
  { id := 3
  , content := "The lottery activities in department stores are a method of drawing lots"
  , isCorrect := true }

def statement4 : SamplingStatement :=
  { id := 4
  , content := "In systematic sampling, the probability of each individual being selected is equal throughout the entire sampling process (except when exclusions are made)"
  , isCorrect := true }

-- Define the list of all statements
def allStatements : List SamplingStatement := [statement1, statement2, statement3, statement4]

-- Theorem: The number of correct statements is 3
theorem correct_statements_count :
  (allStatements.filter (λ s => s.isCorrect)).length = 3 := by
  sorry


end NUMINAMATH_CALUDE_correct_statements_count_l2873_287351


namespace NUMINAMATH_CALUDE_circle_radius_sqrt_29_l2873_287379

/-- Given a circle with center on the x-axis that passes through points (2,2) and (-1,5),
    prove that its radius is √29 -/
theorem circle_radius_sqrt_29 :
  ∃ (x : ℝ), 
    (x - 2)^2 + 2^2 = (x + 1)^2 + 5^2 →
    Real.sqrt ((x - 2)^2 + 2^2) = Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_sqrt_29_l2873_287379


namespace NUMINAMATH_CALUDE_rate_percent_proof_l2873_287365

/-- Simple interest formula -/
def simple_interest (principal rate time : ℚ) : ℚ :=
  principal * rate * time / 100

theorem rate_percent_proof (principal interest time : ℚ) 
  (h1 : principal = 800)
  (h2 : interest = 192)
  (h3 : time = 4)
  (h4 : simple_interest principal (6 : ℚ) time = interest) : 
  (6 : ℚ) = (interest * 100) / (principal * time) := by
  sorry

end NUMINAMATH_CALUDE_rate_percent_proof_l2873_287365


namespace NUMINAMATH_CALUDE_binomial_10_choose_4_l2873_287385

theorem binomial_10_choose_4 : Nat.choose 10 4 = 210 := by sorry

end NUMINAMATH_CALUDE_binomial_10_choose_4_l2873_287385


namespace NUMINAMATH_CALUDE_chord_length_theorem_l2873_287358

/-- Represents a circle with a given radius and center point -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two circles are externally tangent -/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 = (c1.radius + c2.radius)^2

/-- Checks if a smaller circle is internally tangent to a larger circle -/
def is_internally_tangent (small large : Circle) : Prop :=
  (small.center.1 - large.center.1)^2 + (small.center.2 - large.center.2)^2 = (large.radius - small.radius)^2

/-- Represents the common external tangent chord length -/
def common_external_tangent_chord_length_squared (c1 c2 c3 : Circle) : ℝ := 72

theorem chord_length_theorem (c1 c2 c3 : Circle)
  (h1 : c1.radius = 3)
  (h2 : c2.radius = 6)
  (h3 : c3.radius = 9)
  (h4 : are_externally_tangent c1 c2)
  (h5 : is_internally_tangent c1 c3)
  (h6 : is_internally_tangent c2 c3) :
  common_external_tangent_chord_length_squared c1 c2 c3 = 72 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_theorem_l2873_287358


namespace NUMINAMATH_CALUDE_race_finish_order_l2873_287327

/-- Represents a sprinter in the race -/
inductive Sprinter
  | A
  | B
  | C

/-- Represents the order of sprinters -/
def RaceOrder := List Sprinter

/-- Represents the number of position changes for each sprinter -/
def PositionChanges := Sprinter → Nat

/-- Determines if a sprinter started later than another -/
def StartedLater (s1 s2 : Sprinter) : Prop := sorry

/-- Determines if a sprinter finished before another -/
def FinishedBefore (s1 s2 : Sprinter) : Prop := sorry

/-- Determines if a sprinter was delayed at the start -/
def DelayedAtStart (s : Sprinter) : Prop := sorry

theorem race_finish_order :
  ∀ (changes : PositionChanges),
    changes Sprinter.C = 6 →
    changes Sprinter.A = 5 →
    StartedLater Sprinter.B Sprinter.A →
    FinishedBefore Sprinter.B Sprinter.A →
    DelayedAtStart Sprinter.C →
    ∃ (order : RaceOrder),
      order = [Sprinter.B, Sprinter.A, Sprinter.C] :=
by sorry

end NUMINAMATH_CALUDE_race_finish_order_l2873_287327


namespace NUMINAMATH_CALUDE_base_conversion_142_to_7_l2873_287335

/-- Converts a natural number from base 10 to base 7 --/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 7 to a natural number in base 10 --/
def fromBase7 (digits : List ℕ) : ℕ :=
  sorry

theorem base_conversion_142_to_7 :
  toBase7 142 = [2, 6, 2] ∧ fromBase7 [2, 6, 2] = 142 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_142_to_7_l2873_287335


namespace NUMINAMATH_CALUDE_randy_baseball_gloves_l2873_287339

theorem randy_baseball_gloves (bats : ℕ) (gloves : ℕ) : 
  bats = 4 → gloves = 7 * bats + 1 → gloves = 29 := by
  sorry

end NUMINAMATH_CALUDE_randy_baseball_gloves_l2873_287339


namespace NUMINAMATH_CALUDE_fraction_ordering_l2873_287397

theorem fraction_ordering : 
  let a : ℚ := 6 / 29
  let b : ℚ := 8 / 31
  let c : ℚ := 10 / 39
  a < c ∧ c < b :=
by sorry

end NUMINAMATH_CALUDE_fraction_ordering_l2873_287397


namespace NUMINAMATH_CALUDE_divisibility_of_Z_l2873_287381

/-- Represents a 7-digit positive integer in the form abcabca -/
def Z (a b c : ℕ) : ℕ :=
  1000000 * a + 100000 * b + 10000 * c + 1000 * a + 100 * b + 10 * c + a

/-- Theorem stating that 1001 divides Z for any valid a, b, c -/
theorem divisibility_of_Z (a b c : ℕ) (ha : 0 < a) (ha' : a < 10) (hb : b < 10) (hc : c < 10) :
  1001 ∣ Z a b c := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_Z_l2873_287381


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2873_287398

/-- Represents the sum of the first n terms of an arithmetic sequence -/
def S (n : ℕ) (a₁ d : ℚ) : ℚ := n / 2 * (2 * a₁ + (n - 1) * d)

/-- Theorem stating that for an arithmetic sequence with S₅ = 5 and S₉ = 27, S₇ = 14 -/
theorem arithmetic_sequence_sum (a₁ d : ℚ) 
  (h₁ : S 5 a₁ d = 5)
  (h₂ : S 9 a₁ d = 27) : 
  S 7 a₁ d = 14 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2873_287398


namespace NUMINAMATH_CALUDE_suresh_investment_l2873_287360

/-- Given the total profit, Ramesh's investment, and Ramesh's share of profit, 
    prove that Suresh's investment is Rs. 24,000. -/
theorem suresh_investment 
  (total_profit : ℕ) 
  (ramesh_investment : ℕ) 
  (ramesh_profit : ℕ) 
  (h1 : total_profit = 19000)
  (h2 : ramesh_investment = 40000)
  (h3 : ramesh_profit = 11875) :
  (total_profit - ramesh_profit) * ramesh_investment / ramesh_profit = 24000 := by
  sorry

end NUMINAMATH_CALUDE_suresh_investment_l2873_287360


namespace NUMINAMATH_CALUDE_grid_whitening_theorem_l2873_287392

/-- Represents the color of a square -/
inductive Color
| Black
| White

/-- Represents a grid of squares -/
def Grid := Matrix (Fin 98) (Fin 98) Color

/-- Represents a sub-rectangle in the grid -/
structure SubRectangle where
  top_left : Fin 98 × Fin 98
  width : Nat
  height : Nat
  width_valid : width > 1
  height_valid : height > 1
  in_bounds : top_left.1 + width ≤ 98 ∧ top_left.2 + height ≤ 98

/-- Represents a color-flipping operation on a sub-rectangle -/
def flip_operation (grid : Grid) (rect : SubRectangle) : Grid :=
  sorry

/-- Represents a sequence of color-flipping operations -/
def operation_sequence := List SubRectangle

/-- Applies a sequence of operations to a grid -/
def apply_operations (grid : Grid) (ops : operation_sequence) : Grid :=
  sorry

/-- Checks if all squares in the grid are white -/
def all_white (grid : Grid) : Prop :=
  sorry

/-- Main theorem: There exists a finite sequence of operations that turns any grid all white -/
theorem grid_whitening_theorem (initial_grid : Grid) :
  ∃ (ops : operation_sequence), all_white (apply_operations initial_grid ops) :=
sorry

end NUMINAMATH_CALUDE_grid_whitening_theorem_l2873_287392


namespace NUMINAMATH_CALUDE_art_arrangement_probability_l2873_287366

/-- The probability of arranging n items with k specific items consecutive -/
def consecutive_probability (n : ℕ) (k : ℕ) : ℚ :=
  if k ≤ n ∧ k > 0 then
    (Nat.factorial (n - k + 1) * Nat.factorial k) / Nat.factorial n
  else
    0

/-- Theorem: The probability of arranging 12 pieces of art with 4 specific pieces consecutive is 1/55 -/
theorem art_arrangement_probability :
  consecutive_probability 12 4 = 1 / 55 := by
  sorry

#eval consecutive_probability 12 4

end NUMINAMATH_CALUDE_art_arrangement_probability_l2873_287366


namespace NUMINAMATH_CALUDE_triangle_property_l2873_287326

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    if a*cos(B) - b*cos(A) = (3/5)*c, then tan(A)/tan(B) = 4 and max(tan(A-B)) = 3/4 -/
theorem triangle_property (a b c : ℝ) (A B C : ℝ) :
  (0 < a) → (0 < b) → (0 < c) →
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  (a * Real.cos B - b * Real.cos A = (3/5) * c) →
  (a / Real.sin A = b / Real.sin B) →
  (b / Real.sin B = c / Real.sin C) →
  (Real.tan A / Real.tan B = 4) ∧
  (∀ x y : ℝ, Real.tan (A - B) ≤ (3/4)) ∧
  (∃ x y : ℝ, Real.tan (A - B) = (3/4)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_property_l2873_287326


namespace NUMINAMATH_CALUDE_pizza_parlor_cost_theorem_l2873_287386

/-- Calculates the total cost including gratuity for a group celebration at a pizza parlor -/
def pizza_parlor_cost (total_people : ℕ) (child_pizza_cost adult_pizza_cost child_drink_cost adult_drink_cost : ℚ) (gratuity_rate : ℚ) : ℚ :=
  let num_adults : ℕ := total_people / 3
  let num_children : ℕ := 2 * num_adults
  let child_cost : ℚ := num_children * (child_pizza_cost + child_drink_cost)
  let adult_cost : ℚ := num_adults * (adult_pizza_cost + adult_drink_cost)
  let subtotal : ℚ := child_cost + adult_cost
  let gratuity : ℚ := subtotal * gratuity_rate
  subtotal + gratuity

/-- The total cost including gratuity for the group celebration at the pizza parlor is $1932 -/
theorem pizza_parlor_cost_theorem : 
  pizza_parlor_cost 120 10 12 3 4 (15/100) = 1932 :=
by sorry

end NUMINAMATH_CALUDE_pizza_parlor_cost_theorem_l2873_287386


namespace NUMINAMATH_CALUDE_spinner_probability_l2873_287345

theorem spinner_probability (pA pB pC pD : ℚ) : 
  pA = 1/4 → pB = 1/3 → pD = 1/6 → pA + pB + pC + pD = 1 → pC = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_spinner_probability_l2873_287345


namespace NUMINAMATH_CALUDE_defective_engine_fraction_l2873_287393

theorem defective_engine_fraction :
  let total_batches : ℕ := 5
  let engines_per_batch : ℕ := 80
  let non_defective_engines : ℕ := 300
  let total_engines : ℕ := total_batches * engines_per_batch
  let defective_engines : ℕ := total_engines - non_defective_engines
  (defective_engines : ℚ) / total_engines = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_defective_engine_fraction_l2873_287393


namespace NUMINAMATH_CALUDE_min_rolls_for_two_sixes_l2873_287364

/-- The probability of getting two sixes in a single roll of two dice -/
def p : ℚ := 1 / 36

/-- The probability of not getting two sixes in a single roll of two dice -/
def q : ℚ := 1 - p

/-- The number of rolls -/
def n : ℕ := 25

/-- The theorem stating that n is the minimum number of rolls required -/
theorem min_rolls_for_two_sixes (n : ℕ) : 
  (1 - q ^ n > (1 : ℚ) / 2) ∧ ∀ m < n, (1 - q ^ m ≤ (1 : ℚ) / 2) :=
sorry

end NUMINAMATH_CALUDE_min_rolls_for_two_sixes_l2873_287364


namespace NUMINAMATH_CALUDE_exists_universal_program_l2873_287342

/- Define the maze structure -/
def Maze := Fin 10 → Fin 10 → Bool

/- Define the robot's position -/
structure Position where
  x : Fin 10
  y : Fin 10

/- Define the possible robot commands -/
inductive Command
| L
| R
| U
| D

/- Define a program as a list of commands -/
def Program := List Command

/- Function to check if a cell is accessible -/
def isAccessible (maze : Maze) (pos : Position) : Bool :=
  maze pos.x pos.y

/- Function to apply a command to a position -/
def applyCommand (maze : Maze) (pos : Position) (cmd : Command) : Position :=
  sorry

/- Function to check if a program visits all accessible cells -/
def visitsAllCells (maze : Maze) (start : Position) (prog : Program) : Prop :=
  sorry

/- The main theorem -/
theorem exists_universal_program :
  ∃ (prog : Program),
    ∀ (maze : Maze) (start : Position),
      visitsAllCells maze start prog :=
sorry

end NUMINAMATH_CALUDE_exists_universal_program_l2873_287342


namespace NUMINAMATH_CALUDE_three_heads_after_three_tails_probability_l2873_287348

/-- Represents the outcome of a coin flip -/
inductive CoinFlip
| Heads
| Tails

/-- Represents a sequence of coin flips -/
def FlipSequence := List CoinFlip

/-- A fair coin has equal probability of heads and tails -/
def isFairCoin (p : CoinFlip → ℝ) : Prop :=
  p CoinFlip.Heads = 1/2 ∧ p CoinFlip.Tails = 1/2

/-- Checks if a sequence ends with three heads in a row -/
def endsWithThreeHeads : FlipSequence → Bool := sorry

/-- Checks if a sequence contains three tails before three heads -/
def hasThreeTailsBeforeThreeHeads : FlipSequence → Bool := sorry

/-- The probability of a specific flip sequence occurring -/
def sequenceProbability (s : FlipSequence) (p : CoinFlip → ℝ) : ℝ := sorry

/-- The main theorem to prove -/
theorem three_heads_after_three_tails_probability 
  (p : CoinFlip → ℝ) (h : isFairCoin p) :
  (∃ s : FlipSequence, endsWithThreeHeads s ∧ hasThreeTailsBeforeThreeHeads s ∧
    sequenceProbability s p = 1/192) :=
by sorry

end NUMINAMATH_CALUDE_three_heads_after_three_tails_probability_l2873_287348


namespace NUMINAMATH_CALUDE_point_coordinate_sum_l2873_287320

/-- Given two points A(0,0) and B(x,-3) where the slope of AB is 4/5, 
    the sum of B's coordinates is -27/4 -/
theorem point_coordinate_sum (x : ℚ) : 
  let A : ℚ × ℚ := (0, 0)
  let B : ℚ × ℚ := (x, -3)
  let slope : ℚ := (B.2 - A.2) / (B.1 - A.1)
  slope = 4/5 → x + B.2 = -27/4 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinate_sum_l2873_287320


namespace NUMINAMATH_CALUDE_binomial_6_choose_2_l2873_287316

theorem binomial_6_choose_2 : Nat.choose 6 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_binomial_6_choose_2_l2873_287316


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l2873_287310

theorem quadratic_equation_solutions :
  (∀ x, 3 * x^2 - 6 * x = 0 ↔ x = 0 ∨ x = 2) ∧
  (∀ x, x^2 + 4 * x - 1 = 0 ↔ x = -2 + Real.sqrt 5 ∨ x = -2 - Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l2873_287310


namespace NUMINAMATH_CALUDE_select_chess_team_l2873_287307

/-- The number of ways to select a team of 4 players from 10, where two are twins and both twins can't be on the team -/
def select_team (total_players : ℕ) (team_size : ℕ) (num_twins : ℕ) : ℕ :=
  Nat.choose total_players team_size - Nat.choose (total_players - num_twins) (team_size - num_twins)

/-- Theorem stating that selecting 4 players from 10, where two are twins and both twins can't be on the team, results in 182 ways -/
theorem select_chess_team : select_team 10 4 2 = 182 := by
  sorry

end NUMINAMATH_CALUDE_select_chess_team_l2873_287307


namespace NUMINAMATH_CALUDE_last_four_digits_of_5_to_2011_last_four_digits_of_5_to_7_last_four_digits_of_5_to_2011_is_8125_l2873_287374

/-- The last four digits of 5^n -/
def lastFourDigits (n : ℕ) : ℕ := 5^n % 10000

/-- The cycle length of the last four digits of powers of 5 -/
def cycleLength : ℕ := 4

theorem last_four_digits_of_5_to_2011 :
  lastFourDigits 2011 = lastFourDigits 7 :=
by sorry

theorem last_four_digits_of_5_to_7 :
  lastFourDigits 7 = 8125 :=
by sorry

theorem last_four_digits_of_5_to_2011_is_8125 :
  lastFourDigits 2011 = 8125 :=
by sorry

end NUMINAMATH_CALUDE_last_four_digits_of_5_to_2011_last_four_digits_of_5_to_7_last_four_digits_of_5_to_2011_is_8125_l2873_287374


namespace NUMINAMATH_CALUDE_sister_ages_l2873_287375

theorem sister_ages (x y : ℕ) (h1 : x - y = 4) (h2 : x^3 - y^3 = 988) : x = 11 ∧ y = 7 := by
  sorry

end NUMINAMATH_CALUDE_sister_ages_l2873_287375


namespace NUMINAMATH_CALUDE_orchestra_members_count_l2873_287322

theorem orchestra_members_count :
  ∃! n : ℕ, 200 < n ∧ n < 300 ∧
    n % 4 = 2 ∧
    n % 5 = 3 ∧
    n % 7 = 4 ∧
    n = 158 :=
by sorry

end NUMINAMATH_CALUDE_orchestra_members_count_l2873_287322


namespace NUMINAMATH_CALUDE_rope_problem_l2873_287325

theorem rope_problem (x : ℝ) :
  (8 : ℝ)^2 + (x - 3)^2 = x^2 :=
by sorry

end NUMINAMATH_CALUDE_rope_problem_l2873_287325


namespace NUMINAMATH_CALUDE_rationalize_and_simplify_l2873_287330

theorem rationalize_and_simplify :
  (32 / Real.sqrt 8) + (8 / Real.sqrt 32) = 9 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_and_simplify_l2873_287330


namespace NUMINAMATH_CALUDE_circle_center_x_coordinate_range_l2873_287390

/-- The problem statement as a theorem in Lean 4 -/
theorem circle_center_x_coordinate_range :
  ∀ (O A C M : ℝ × ℝ) (l : ℝ → ℝ) (a : ℝ),
    O = (0, 0) →
    A = (0, 3) →
    (∀ x, l x = x + 1) →
    C.2 = l C.1 →
    C.1 = a →
    ∃ r : ℝ, r = 1 ∧ ∀ p : ℝ × ℝ, (p.1 - C.1)^2 + (p.2 - C.2)^2 = r^2 →
      ∃ M : ℝ × ℝ, (M.1 - C.1)^2 + (M.2 - C.2)^2 = r^2 ∧
        (M.1 - A.1)^2 + (M.2 - A.2)^2 = 4 * ((M.1 - O.1)^2 + (M.2 - O.2)^2) →
          -1 - Real.sqrt 7 / 2 ≤ a ∧ a ≤ -1 + Real.sqrt 7 / 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_x_coordinate_range_l2873_287390


namespace NUMINAMATH_CALUDE_koolaid_percentage_l2873_287301

/-- Calculates the percentage of Kool-Aid powder in a mixture after evaporation and water addition -/
theorem koolaid_percentage
  (initial_powder : ℚ)
  (initial_water : ℚ)
  (evaporation_rate : ℚ)
  (water_increase_factor : ℚ)
  (h1 : initial_powder = 3)
  (h2 : initial_water = 20)
  (h3 : evaporation_rate = 1/4)
  (h4 : water_increase_factor = 5) :
  let remaining_water := initial_water * (1 - evaporation_rate)
  let final_water := remaining_water * water_increase_factor
  let final_mixture := initial_powder + final_water
  initial_powder / final_mixture = 1/26 :=
sorry

end NUMINAMATH_CALUDE_koolaid_percentage_l2873_287301


namespace NUMINAMATH_CALUDE_twentieth_term_is_79_l2873_287362

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1 : ℝ) * d

/-- The 20th term of the specific arithmetic sequence is 79 -/
theorem twentieth_term_is_79 :
  arithmetic_sequence 3 4 20 = 79 := by
  sorry

end NUMINAMATH_CALUDE_twentieth_term_is_79_l2873_287362


namespace NUMINAMATH_CALUDE_max_true_statements_l2873_287331

theorem max_true_statements (a b : ℝ) : 
  let statements := [
    (1/a > 1/b),
    (a^2 < b^2),
    (a > b),
    (a > 0),
    (b > 0)
  ]
  ∃ (trueStatements : List Bool), 
    trueStatements.length ≤ 3 ∧ 
    ∀ (i : Nat), i < statements.length → 
      (trueStatements.get? i = some true → statements.get! i) ∧
      (statements.get! i → trueStatements.get? i = some true) :=
sorry

end NUMINAMATH_CALUDE_max_true_statements_l2873_287331


namespace NUMINAMATH_CALUDE_moss_pollen_scientific_notation_l2873_287352

theorem moss_pollen_scientific_notation (d : ℝ) (n : ℤ) :
  d = 0.0000084 →
  d = 8.4 * (10 : ℝ) ^ n →
  n = -6 :=
by
  sorry

end NUMINAMATH_CALUDE_moss_pollen_scientific_notation_l2873_287352


namespace NUMINAMATH_CALUDE_remainder_of_product_div_12_l2873_287323

theorem remainder_of_product_div_12 : (1125 * 1127 * 1129) % 12 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_product_div_12_l2873_287323


namespace NUMINAMATH_CALUDE_sunset_time_calculation_l2873_287350

/-- Represents time in 24-hour format -/
structure Time24 where
  hours : Nat
  minutes : Nat
  valid : hours < 24 ∧ minutes < 60

/-- Represents a duration in hours and minutes -/
structure Duration where
  hours : Nat
  minutes : Nat
  valid : minutes < 60

/-- Adds a duration to a time, wrapping around 24 hours if necessary -/
def addDuration (t : Time24) (d : Duration) : Time24 := sorry

/-- Converts 24-hour time to 12-hour time string (AM/PM) -/
def to12Hour (t : Time24) : String := sorry

theorem sunset_time_calculation 
  (sunrise : Time24) 
  (daylight : Duration) 
  (h_sunrise : sunrise.hours = 7 ∧ sunrise.minutes = 30)
  (h_daylight : daylight.hours = 11 ∧ daylight.minutes = 10) :
  to12Hour (addDuration sunrise daylight) = "6:40 PM" := by sorry

end NUMINAMATH_CALUDE_sunset_time_calculation_l2873_287350


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2873_287334

/-- Triangle ABC with vertices A(-1,5), B(-2,-1), and C(4,3) -/
structure Triangle :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)

/-- The specific triangle ABC from the problem -/
def triangleABC : Triangle :=
  { A := (-1, 5)
  , B := (-2, -1)
  , C := (4, 3) }

/-- Equation of a line in the form ax + by + c = 0 -/
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- Definition of an altitude in a triangle -/
def isAltitude (t : Triangle) (l : Line) : Prop :=
  sorry

/-- The area of a triangle -/
def triangleArea (t : Triangle) : ℝ :=
  sorry

/-- Main theorem stating the properties of triangle ABC -/
theorem triangle_abc_properties :
  let t := triangleABC
  let altitude := Line.mk 3 2 (-7)
  isAltitude t altitude ∧ triangleArea t = 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l2873_287334


namespace NUMINAMATH_CALUDE_cat_roaming_area_l2873_287338

/-- The area accessible to a cat tethered to a circular water tank -/
theorem cat_roaming_area (tank_radius rope_length : ℝ) (h1 : tank_radius = 20) (h2 : rope_length = 10) :
  π * (tank_radius + rope_length)^2 - π * tank_radius^2 = 500 * π :=
by sorry

end NUMINAMATH_CALUDE_cat_roaming_area_l2873_287338


namespace NUMINAMATH_CALUDE_number_categorization_l2873_287353

def S : Set ℝ := {-2.5, 0, 8, -2, Real.pi/2, 0.7, -2/3, -1.12112112, 3/4}

theorem number_categorization :
  (∃ P I R : Set ℝ,
    P = {x ∈ S | x > 0} ∧
    I = {x ∈ S | ∃ n : ℤ, x = n} ∧
    R = {x ∈ S | ¬∃ q : ℚ, x = q} ∧
    P = {8, Real.pi/2, 0.7, 3/4} ∧
    I = {0, 8, -2} ∧
    R = {Real.pi/2, -1.12112112}) :=
by sorry

end NUMINAMATH_CALUDE_number_categorization_l2873_287353


namespace NUMINAMATH_CALUDE_journey_distance_l2873_287333

/-- Proves that the total distance of a journey is 70 km given specific travel conditions. -/
theorem journey_distance (v1 v2 : ℝ) (t_late : ℝ) : 
  v1 = 40 →  -- Average speed for on-time arrival (km/h)
  v2 = 35 →  -- Average speed for late arrival (km/h)
  t_late = 0.25 →  -- Time of late arrival (hours)
  ∃ (d t : ℝ), 
    d = v1 * t ∧  -- Distance equation for on-time arrival
    d = v2 * (t + t_late) ∧  -- Distance equation for late arrival
    d = 70  -- Total distance of the journey (km)
  := by sorry

end NUMINAMATH_CALUDE_journey_distance_l2873_287333


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2873_287389

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_eq : a 2 + a 8 = 15 - a 5) : 
  a 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2873_287389


namespace NUMINAMATH_CALUDE_yellow_ball_probability_l2873_287394

-- Define the number of balls of each color
def red_balls : ℕ := 2
def yellow_balls : ℕ := 5
def blue_balls : ℕ := 4

-- Define the total number of balls
def total_balls : ℕ := red_balls + yellow_balls + blue_balls

-- Define the probability of choosing a yellow ball
def prob_yellow : ℚ := yellow_balls / total_balls

-- Theorem statement
theorem yellow_ball_probability : prob_yellow = 5 / 11 := by
  sorry

end NUMINAMATH_CALUDE_yellow_ball_probability_l2873_287394


namespace NUMINAMATH_CALUDE_consecutive_non_prime_powers_l2873_287344

theorem consecutive_non_prime_powers (n : ℕ+) :
  ∃ x : ℕ, ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → ¬∃ (p : ℕ) (k : ℕ), Prime p ∧ (x + i = p^k) :=
sorry

end NUMINAMATH_CALUDE_consecutive_non_prime_powers_l2873_287344


namespace NUMINAMATH_CALUDE_unique_solution_l2873_287312

def repeating_decimal_2 (a b : ℕ) : ℚ :=
  (10 * a + b) / 99

def repeating_decimal_3 (a b c : ℕ) : ℚ :=
  (100 * a + 10 * b + c) / 999

def is_digit (n : ℕ) : Prop :=
  n ≥ 0 ∧ n ≤ 9

theorem unique_solution (a b c : ℕ) :
  is_digit a ∧ is_digit b ∧ is_digit c →
  repeating_decimal_2 a b + repeating_decimal_3 a b c = 35 / 37 →
  a = 5 ∧ b = 3 ∧ c = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l2873_287312


namespace NUMINAMATH_CALUDE_absolute_value_implies_inequality_l2873_287321

theorem absolute_value_implies_inequality (x : ℝ) : 
  |x - 1| = 1 - x → x ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_implies_inequality_l2873_287321


namespace NUMINAMATH_CALUDE_B_power_2023_l2873_287368

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![0, -1, 0],
    ![1,  0, 0],
    ![0,  0, -1]]

theorem B_power_2023 :
  B ^ 2023 = ![![ 0,  1,  0],
               ![-1,  0,  0],
               ![ 0,  0, -1]] := by sorry

end NUMINAMATH_CALUDE_B_power_2023_l2873_287368


namespace NUMINAMATH_CALUDE_darrel_took_48_candies_l2873_287399

/-- Represents the number of candies on the table -/
structure CandyCount where
  red : ℕ
  blue : ℕ

/-- Represents the state of candies on the table at different stages -/
structure CandyState where
  initial : CandyCount
  afterDarrel : CandyCount
  afterCloe : CandyCount

/-- Darrel's action of taking candies -/
def darrelAction (x : ℕ) (c : CandyCount) : CandyCount :=
  { red := c.red - x, blue := c.blue - x }

/-- Cloe's action of taking candies -/
def cloeAction (c : CandyCount) : CandyCount :=
  { red := c.red - 12, blue := c.blue - 12 }

/-- The theorem to be proved -/
theorem darrel_took_48_candies (state : CandyState) (x : ℕ) :
  state.initial.red = 3 * state.initial.blue →
  state.afterDarrel = darrelAction x state.initial →
  state.afterDarrel.red = 4 * state.afterDarrel.blue →
  state.afterCloe = cloeAction state.afterDarrel →
  state.afterCloe.red = 5 * state.afterCloe.blue →
  2 * x = 48 := by
  sorry


end NUMINAMATH_CALUDE_darrel_took_48_candies_l2873_287399


namespace NUMINAMATH_CALUDE_fifth_day_temperature_l2873_287302

/-- Given the average temperatures and ratio of temperatures for specific days,
    prove that the temperature on the fifth day is 32 degrees. -/
theorem fifth_day_temperature
  (avg_first_four : ℝ)
  (avg_second_to_fifth : ℝ)
  (temp_first : ℝ)
  (temp_fifth : ℝ)
  (h1 : avg_first_four = 58)
  (h2 : avg_second_to_fifth = 59)
  (h3 : temp_fifth = (8 / 7) * temp_first)
  (h4 : temp_first + (avg_first_four * 4 - temp_first) = avg_first_four * 4)
  (h5 : (avg_first_four * 4 - temp_first) + temp_fifth = avg_second_to_fifth * 4) :
  temp_fifth = 32 :=
sorry

end NUMINAMATH_CALUDE_fifth_day_temperature_l2873_287302


namespace NUMINAMATH_CALUDE_work_days_calculation_l2873_287369

theorem work_days_calculation (days_a days_b : ℕ) (wage_c : ℕ) (total_earning : ℕ) :
  days_a = 6 →
  days_b = 9 →
  wage_c = 95 →
  total_earning = 1406 →
  ∃ (days_c : ℕ),
    (3 * wage_c * days_a + 4 * wage_c * days_b + 5 * wage_c * days_c = 5 * total_earning) ∧
    days_c = 4 :=
by sorry

end NUMINAMATH_CALUDE_work_days_calculation_l2873_287369


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_example_l2873_287395

/-- Calculates the sampling interval for systematic sampling -/
def systematicSamplingInterval (populationSize sampleSize : ℕ) : ℕ :=
  populationSize / sampleSize

/-- Theorem: The systematic sampling interval for a population of 2000 and sample size of 50 is 40 -/
theorem systematic_sampling_interval_example :
  systematicSamplingInterval 2000 50 = 40 := by
  sorry

#eval systematicSamplingInterval 2000 50

end NUMINAMATH_CALUDE_systematic_sampling_interval_example_l2873_287395


namespace NUMINAMATH_CALUDE_red_balls_count_l2873_287387

/-- Given a bag of balls with the following properties:
  * The total number of balls is 60
  * The frequency of picking red balls is 0.15
  Prove that the number of red balls in the bag is 9 -/
theorem red_balls_count (total_balls : ℕ) (red_frequency : ℝ) 
  (h1 : total_balls = 60)
  (h2 : red_frequency = 0.15) :
  ⌊total_balls * red_frequency⌋ = 9 := by
  sorry

end NUMINAMATH_CALUDE_red_balls_count_l2873_287387


namespace NUMINAMATH_CALUDE_purchase_equivalence_l2873_287315

/-- Proves that if a person can buy exactly 6 items at price x and exactly 8 items at price (x - 1.5),
    then the total amount of money the person has is 36. -/
theorem purchase_equivalence (x : ℝ) :
  (6 * x = 8 * (x - 1.5)) → 6 * x = 36 := by
  sorry

end NUMINAMATH_CALUDE_purchase_equivalence_l2873_287315


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2873_287384

def set_A : Set ℝ := {x | ∃ y, y = Real.sqrt (-x^2 + 1)}
def set_B : Set ℝ := Set.Ioo 0 1

theorem intersection_of_A_and_B :
  set_A ∩ set_B = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2873_287384


namespace NUMINAMATH_CALUDE_smallest_lcm_with_gcd_11_l2873_287314

theorem smallest_lcm_with_gcd_11 :
  ∃ (k l : ℕ), 
    1000 ≤ k ∧ k < 10000 ∧
    1000 ≤ l ∧ l < 10000 ∧
    Nat.gcd k l = 11 ∧
    Nat.lcm k l = 92092 ∧
    ∀ (m n : ℕ), 
      1000 ≤ m ∧ m < 10000 ∧
      1000 ≤ n ∧ n < 10000 ∧
      Nat.gcd m n = 11 →
      Nat.lcm m n ≥ 92092 :=
by sorry

end NUMINAMATH_CALUDE_smallest_lcm_with_gcd_11_l2873_287314


namespace NUMINAMATH_CALUDE_mary_weight_change_ratio_l2873_287363

/-- Represents the sequence of weight changes in Mary's diet journey -/
structure WeightChange where
  initial_weight : ℝ
  initial_loss : ℝ
  second_gain : ℝ
  third_loss : ℝ
  final_gain : ℝ
  final_weight : ℝ

/-- Theorem representing Mary's weight change problem -/
theorem mary_weight_change_ratio (w : WeightChange)
  (h1 : w.initial_weight = 99)
  (h2 : w.initial_loss = 12)
  (h3 : w.second_gain = 2 * w.initial_loss)
  (h4 : w.final_gain = 6)
  (h5 : w.final_weight = 81)
  (h6 : w.initial_weight - w.initial_loss + w.second_gain - w.third_loss + w.final_gain = w.final_weight) :
  w.third_loss / w.initial_loss = 3 := by
  sorry


end NUMINAMATH_CALUDE_mary_weight_change_ratio_l2873_287363


namespace NUMINAMATH_CALUDE_harry_last_mile_water_consumption_l2873_287319

/-- Represents the hike scenario --/
structure HikeScenario where
  totalDistance : ℝ
  initialWater : ℝ
  finalWater : ℝ
  timeTaken : ℝ
  leakRate : ℝ
  waterConsumptionFirstThreeMiles : ℝ

/-- Calculates the water consumed in the last mile of the hike --/
def waterConsumedLastMile (h : HikeScenario) : ℝ :=
  h.initialWater - h.finalWater - (h.leakRate * h.timeTaken) - (h.waterConsumptionFirstThreeMiles * (h.totalDistance - 1))

/-- Theorem stating that Harry drank 3 cups of water in the last mile --/
theorem harry_last_mile_water_consumption :
  let h : HikeScenario := {
    totalDistance := 4
    initialWater := 10
    finalWater := 2
    timeTaken := 2
    leakRate := 1
    waterConsumptionFirstThreeMiles := 1
  }
  waterConsumedLastMile h = 3 := by
  sorry


end NUMINAMATH_CALUDE_harry_last_mile_water_consumption_l2873_287319


namespace NUMINAMATH_CALUDE_largest_integer_with_conditions_l2873_287337

def digit_sum_of_squares (n : ℕ) : ℕ := sorry

def digits_increasing (n : ℕ) : Prop := sorry

def product_of_digits (n : ℕ) : ℕ := sorry

theorem largest_integer_with_conditions (n : ℕ) :
  (digit_sum_of_squares n = 82) →
  digits_increasing n →
  product_of_digits n ≤ 9 := by sorry

end NUMINAMATH_CALUDE_largest_integer_with_conditions_l2873_287337


namespace NUMINAMATH_CALUDE_expression_value_l2873_287377

theorem expression_value (x y : ℤ) (hx : x = 3) (hy : y = 2) : 3 * x - 4 * y + 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2873_287377


namespace NUMINAMATH_CALUDE_equation_equivalence_l2873_287383

theorem equation_equivalence (x : ℝ) (h : x ≠ 3 ∧ x ≠ 4 ∧ x ≠ 5 ∧ x ≠ 7) : 
  1 / (x - 3) + 1 / (x - 5) + 1 / (x - 7) = 4 / (x - 4) ↔ 
  x^3 - 13*x^2 + 48*x - 64 = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l2873_287383


namespace NUMINAMATH_CALUDE_lakers_win_in_seven_l2873_287382

/-- The probability of the Celtics winning a single game -/
def p_celtics : ℚ := 2/3

/-- The probability of the Lakers winning a single game -/
def p_lakers : ℚ := 1 - p_celtics

/-- The number of ways to choose 3 games out of 6 -/
def ways_to_choose_3_of_6 : ℕ := 20

/-- The probability that the Lakers win the NBA finals in exactly 7 games -/
theorem lakers_win_in_seven (p_celtics : ℚ) (p_lakers : ℚ) (ways_to_choose_3_of_6 : ℕ) :
  p_celtics = 2/3 →
  p_lakers = 1 - p_celtics →
  ways_to_choose_3_of_6 = 20 →
  (ways_to_choose_3_of_6 : ℚ) * p_lakers^3 * p_celtics^3 * p_lakers = 160/2187 :=
by sorry

end NUMINAMATH_CALUDE_lakers_win_in_seven_l2873_287382


namespace NUMINAMATH_CALUDE_rice_cost_is_ten_cents_l2873_287308

/-- The cost of rice per plate in cents -/
def rice_cost_per_plate (total_plates : ℕ) (chicken_cost_per_plate : ℚ) (total_spent : ℚ) : ℚ :=
  (total_spent - (chicken_cost_per_plate * total_plates)) / total_plates * 100

/-- Theorem: The cost of rice per plate is 10 cents -/
theorem rice_cost_is_ten_cents :
  rice_cost_per_plate 100 0.40 50 = 10 := by
  sorry

end NUMINAMATH_CALUDE_rice_cost_is_ten_cents_l2873_287308


namespace NUMINAMATH_CALUDE_ratio_of_squares_nonnegative_l2873_287340

theorem ratio_of_squares_nonnegative (x : ℝ) (h : x ≠ 5) : (x^2) / ((x - 5)^2) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_squares_nonnegative_l2873_287340


namespace NUMINAMATH_CALUDE_cos_negative_seventy_nine_pi_sixths_l2873_287313

theorem cos_negative_seventy_nine_pi_sixths : 
  Real.cos (-79 * Real.pi / 6) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_seventy_nine_pi_sixths_l2873_287313


namespace NUMINAMATH_CALUDE_circle_center_correct_l2873_287357

/-- The equation of a circle in the form ax² + bx + cy² + dy + e = 0 --/
structure CircleEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- The center of a circle --/
structure CircleCenter where
  x : ℝ
  y : ℝ

/-- Function to find the center of a circle given its equation --/
def findCircleCenter (eq : CircleEquation) : CircleCenter :=
  sorry

theorem circle_center_correct :
  let eq := CircleEquation.mk 4 8 4 (-24) 96
  let center := findCircleCenter eq
  center.x = -1 ∧ center.y = 3 := by sorry

end NUMINAMATH_CALUDE_circle_center_correct_l2873_287357


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l2873_287332

theorem quadratic_equation_coefficients (b c : ℝ) :
  (∀ x : ℝ, x^2 + b*x + c = 0 ↔ x = 1 ∨ x = -7) →
  (∀ x : ℝ, |x + 3| = 4 ↔ x = 1 ∨ x = -7) →
  b = 6 ∧ c = -7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l2873_287332


namespace NUMINAMATH_CALUDE_total_silver_dollars_l2873_287378

theorem total_silver_dollars (chiu phung ha lin : ℕ) : 
  chiu = 56 →
  phung = chiu + 16 →
  ha = phung + 5 →
  lin = (chiu + phung + ha) + 25 →
  chiu + phung + ha + lin = 435 :=
by sorry

end NUMINAMATH_CALUDE_total_silver_dollars_l2873_287378


namespace NUMINAMATH_CALUDE_fox_invasion_count_l2873_287343

/-- The number of foxes that invaded the forest region --/
def num_foxes : ℕ := 3

/-- The initial number of rodents in the forest --/
def initial_rodents : ℕ := 150

/-- The number of rodents each fox catches per week --/
def rodents_per_fox_per_week : ℕ := 6

/-- The number of weeks the foxes hunted --/
def weeks : ℕ := 3

/-- The number of rodents remaining after the foxes hunted --/
def remaining_rodents : ℕ := 96

theorem fox_invasion_count :
  num_foxes * (rodents_per_fox_per_week * weeks) = initial_rodents - remaining_rodents :=
by sorry

end NUMINAMATH_CALUDE_fox_invasion_count_l2873_287343


namespace NUMINAMATH_CALUDE_train_length_l2873_287300

/-- The length of a train given its speed, the speed of a person running in the opposite direction, and the time it takes for the train to pass the person. -/
theorem train_length (train_speed : ℝ) (man_speed : ℝ) (passing_time : ℝ) 
  (h1 : train_speed = 69) 
  (h2 : man_speed = 3) 
  (h3 : passing_time = 10) : 
  train_speed * (5/18) * passing_time + man_speed * (5/18) * passing_time = 200 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l2873_287300


namespace NUMINAMATH_CALUDE_parallel_lines_minimum_value_l2873_287371

theorem parallel_lines_minimum_value (m n : ℕ+) 
  (h_parallel : (2 : ℝ) / (n - 1 : ℝ) = (m : ℝ) / (n : ℝ)) : 
  (∀ k l : ℕ+, (2 : ℝ) / (l - 1 : ℝ) = (k : ℝ) / (l : ℝ) → 2 * m + n ≤ 2 * k + l) ∧ 
  (∃ k l : ℕ+, (2 : ℝ) / (l - 1 : ℝ) = (k : ℝ) / (l : ℝ) ∧ 2 * k + l = 9) :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_minimum_value_l2873_287371


namespace NUMINAMATH_CALUDE_shirt_cost_l2873_287346

/-- Given the cost of jeans and shirts in two scenarios, prove the cost of one shirt. -/
theorem shirt_cost (j s : ℚ) 
  (scenario1 : 3 * j + 2 * s = 69)
  (scenario2 : 2 * j + 3 * s = 66) :
  s = 12 := by
  sorry

end NUMINAMATH_CALUDE_shirt_cost_l2873_287346


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l2873_287373

theorem smallest_number_divisible (n : ℕ) : 
  (∃ (k : ℕ), n - k = 44 ∧ 
   9 ∣ (n - k) ∧ 
   6 ∣ (n - k) ∧ 
   12 ∣ (n - k) ∧ 
   18 ∣ (n - k)) →
  (∀ (m : ℕ), m < n → 
    ¬(∃ (k : ℕ), m - k = 44 ∧ 
      9 ∣ (m - k) ∧ 
      6 ∣ (m - k) ∧ 
      12 ∣ (m - k) ∧ 
      18 ∣ (m - k))) →
  n = 80 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_l2873_287373


namespace NUMINAMATH_CALUDE_max_sum_of_factors_48_l2873_287341

theorem max_sum_of_factors_48 :
  ∃ (a b : ℕ), a * b = 48 ∧ a + b = 49 ∧
  ∀ (x y : ℕ), x * y = 48 → x + y ≤ 49 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_48_l2873_287341


namespace NUMINAMATH_CALUDE_dad_catch_is_27_l2873_287328

/-- The number of salmons Hazel caught -/
def hazel_catch : ℕ := 24

/-- The total number of salmons caught by Hazel and her dad -/
def total_catch : ℕ := 51

/-- The number of salmons Hazel's dad caught -/
def dad_catch : ℕ := total_catch - hazel_catch

theorem dad_catch_is_27 : dad_catch = 27 := by
  sorry

end NUMINAMATH_CALUDE_dad_catch_is_27_l2873_287328


namespace NUMINAMATH_CALUDE_numbers_less_than_reciprocals_l2873_287359

theorem numbers_less_than_reciprocals :
  let numbers : List ℚ := [-1/2, -3, 1/4, 4, 1/3]
  ∀ x ∈ numbers, (x < 1 / x) ↔ (x = -3 ∨ x = 1/4 ∨ x = 1/3) :=
by sorry

end NUMINAMATH_CALUDE_numbers_less_than_reciprocals_l2873_287359


namespace NUMINAMATH_CALUDE_range_of_a_l2873_287303

def p (x : ℝ) : Prop := |x + 1| > 3

def q (x a : ℝ) : Prop := x > a

theorem range_of_a (h1 : ∀ x, q x a → p x) 
                   (h2 : ∃ x, p x ∧ ¬q x a) : 
  a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l2873_287303


namespace NUMINAMATH_CALUDE_smallest_m_is_correct_l2873_287372

/-- The smallest positive value of m for which the equation 15x^2 - mx + 630 = 0 has integral solutions -/
def smallest_m : ℕ := 195

/-- Predicate to check if a quadratic equation ax^2 + bx + c = 0 has integral solutions -/
def has_integral_solutions (a b c : ℤ) : Prop :=
  ∃ x : ℤ, a * x^2 + b * x + c = 0

theorem smallest_m_is_correct :
  (∀ m : ℕ, m < smallest_m → ¬(has_integral_solutions 15 (-m) 630)) ∧
  (has_integral_solutions 15 (-smallest_m) 630) :=
sorry

end NUMINAMATH_CALUDE_smallest_m_is_correct_l2873_287372


namespace NUMINAMATH_CALUDE_logarithm_equality_l2873_287380

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the logarithm base 5 function
noncomputable def log5 (x : ℝ) : ℝ := Real.log x / Real.log 5

-- State the theorem
theorem logarithm_equality : lg 2 + lg 5 + 2 * log5 10 - log5 20 = 2 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_equality_l2873_287380


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l2873_287361

theorem arithmetic_calculations : 
  (1 - 2 + 3 + (-4) = -2) ∧ 
  ((-6) / 3 - (-10) - |(-8)| = 0) := by sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l2873_287361


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l2873_287370

theorem tangent_line_to_circle (k : ℝ) : 
  (∀ x y : ℝ, y = k * x + 3 → x^2 + y^2 = 1 → 
    ∀ ε > 0, ∃ δ > 0, ∀ x' y' : ℝ, 
      x'^2 + y'^2 = 1 → (x' - x)^2 + (y' - y)^2 < δ^2 → 
        (y' - (k * x' + 3))^2 > ε^2 * ((x' - x)^2 + (y' - y)^2)) →
  k = 2 * Real.sqrt 2 ∨ k = -2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l2873_287370


namespace NUMINAMATH_CALUDE_descent_problem_l2873_287388

/-- A function that calculates the final elevation after descending --/
def final_elevation (initial_elevation rate_of_descent duration : ℝ) : ℝ :=
  initial_elevation - rate_of_descent * duration

/-- Theorem stating that descending from 400 feet at 10 feet per minute for 5 minutes results in an elevation of 350 feet --/
theorem descent_problem :
  final_elevation 400 10 5 = 350 := by
  sorry

end NUMINAMATH_CALUDE_descent_problem_l2873_287388


namespace NUMINAMATH_CALUDE_room_size_l2873_287349

/-- Given two square carpets in a square room, prove the room's side length is 19 meters. -/
theorem room_size (small_carpet big_carpet room : ℝ) : 
  small_carpet > 0 ∧ 
  big_carpet = 2 * small_carpet ∧
  (room - small_carpet - big_carpet)^2 = 4 ∧
  (room - big_carpet) * (room - small_carpet) = 14 →
  room = 19 := by
  sorry

end NUMINAMATH_CALUDE_room_size_l2873_287349


namespace NUMINAMATH_CALUDE_floor_sum_product_l2873_287309

theorem floor_sum_product : 3 * (⌊(12.7 : ℝ)⌋ + ⌊(-12.7 : ℝ)⌋) = -3 := by sorry

end NUMINAMATH_CALUDE_floor_sum_product_l2873_287309


namespace NUMINAMATH_CALUDE_min_value_expression_l2873_287356

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b - a - 2 * b = 0) :
  (a^2 / 4 - 2 / a + b^2 - 1 / b) ≥ 7 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l2873_287356


namespace NUMINAMATH_CALUDE_average_weight_increase_l2873_287336

/-- Proves that replacing a person in a group of 5 increases the average weight by 1.5 kg -/
theorem average_weight_increase (group_size : ℕ) (old_weight new_weight : ℝ) :
  group_size = 5 →
  old_weight = 65 →
  new_weight = 72.5 →
  (new_weight - old_weight) / group_size = 1.5 := by
sorry

end NUMINAMATH_CALUDE_average_weight_increase_l2873_287336


namespace NUMINAMATH_CALUDE_nut_mixture_price_l2873_287318

/-- Calculates the total selling price of a nut mixture -/
def total_selling_price (total_weight : ℝ) (cashew_weight : ℝ) (cashew_price : ℝ) (peanut_price : ℝ) : ℝ :=
  cashew_weight * cashew_price + (total_weight - cashew_weight) * peanut_price

/-- The total selling price of the nut mixture is $83.00 -/
theorem nut_mixture_price : total_selling_price 25 11 5 2 = 83 := by
  sorry

end NUMINAMATH_CALUDE_nut_mixture_price_l2873_287318


namespace NUMINAMATH_CALUDE_larger_number_proof_l2873_287305

theorem larger_number_proof (x y : ℝ) (h_diff : x - y = 3) (h_sum : x + y = 31) :
  max x y = 17 := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l2873_287305


namespace NUMINAMATH_CALUDE_weight_of_three_moles_l2873_287306

/-- Given a compound with molecular weight of 882 g/mol, 
    prove that 3 moles of this compound weigh 2646 grams. -/
theorem weight_of_three_moles (molecular_weight : ℝ) (moles : ℝ) : 
  molecular_weight = 882 → moles = 3 → moles * molecular_weight = 2646 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_three_moles_l2873_287306


namespace NUMINAMATH_CALUDE_total_coins_last_month_l2873_287324

/-- The number of coins Mathilde had at the start of this month -/
def mathilde_this_month : ℕ := 100

/-- The number of coins Salah had at the start of this month -/
def salah_this_month : ℕ := 100

/-- The percentage increase in Mathilde's coins from last month to this month -/
def mathilde_increase : ℚ := 25/100

/-- The percentage decrease in Salah's coins from last month to this month -/
def salah_decrease : ℚ := 20/100

/-- Theorem stating that the total number of coins Mathilde and Salah had at the start of last month was 205 -/
theorem total_coins_last_month : 
  ∃ (mathilde_last_month salah_last_month : ℕ),
    (mathilde_this_month : ℚ) = mathilde_last_month * (1 + mathilde_increase) ∧
    (salah_this_month : ℚ) = salah_last_month * (1 - salah_decrease) ∧
    mathilde_last_month + salah_last_month = 205 := by
  sorry

end NUMINAMATH_CALUDE_total_coins_last_month_l2873_287324


namespace NUMINAMATH_CALUDE_probability_of_sum_22_l2873_287317

/-- A function representing the probability of rolling a specific sum with four standard 6-faced dice -/
def probability_of_sum (sum : ℕ) : ℚ :=
  sorry

/-- The number of faces on each die -/
def num_faces : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 4

/-- Theorem stating the probability of rolling a sum of 22 with four standard 6-faced dice -/
theorem probability_of_sum_22 : probability_of_sum 22 = 5 / 648 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_sum_22_l2873_287317


namespace NUMINAMATH_CALUDE_james_weekly_take_home_pay_l2873_287311

/-- Calculates James' weekly take-home pay given his work and tax conditions --/
def jamesTakeHomePay (mainJobRate hourlyRate : ℝ) 
                     (secondJobRatePercentage : ℝ) 
                     (mainJobHours overtimeHours : ℕ) 
                     (secondJobHours : ℕ) 
                     (weekendDays : ℕ)
                     (weekendRate : ℝ)
                     (taxDeductions : ℝ)
                     (federalTaxRate stateTaxRate : ℝ) : ℝ :=
  let secondJobRate := mainJobRate * (1 - secondJobRatePercentage)
  let regularHours := mainJobHours - overtimeHours
  let mainJobEarnings := regularHours * mainJobRate + overtimeHours * mainJobRate * 1.5
  let secondJobEarnings := secondJobHours * secondJobRate
  let weekendEarnings := weekendDays * weekendRate
  let totalEarnings := mainJobEarnings + secondJobEarnings + weekendEarnings
  let taxableIncome := totalEarnings - taxDeductions
  let federalTax := taxableIncome * federalTaxRate
  let stateTax := taxableIncome * stateTaxRate
  let totalTaxes := federalTax + stateTax
  totalEarnings - totalTaxes

/-- Theorem stating that James' weekly take-home pay is $885.30 --/
theorem james_weekly_take_home_pay :
  jamesTakeHomePay 20 20 0.2 30 5 15 2 100 200 0.18 0.05 = 885.30 := by
  sorry

end NUMINAMATH_CALUDE_james_weekly_take_home_pay_l2873_287311


namespace NUMINAMATH_CALUDE_triangle_proof_l2873_287376

theorem triangle_proof (A B C : ℝ) (a b c : ℝ) :
  a = Real.sqrt 7 →
  b = 2 →
  a * Real.sin B - Real.sqrt 3 * b * Real.cos A = 0 →
  (A = π / 3 ∧ 
   (1 / 2) * b * c * Real.sin A = (3 * Real.sqrt 3) / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_proof_l2873_287376


namespace NUMINAMATH_CALUDE_distance_to_point_l2873_287367

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 4*x + 2*y + 4

-- Define the center of the circle
def circle_center : ℝ × ℝ := sorry

-- Define the distance function between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem distance_to_point : distance circle_center (10, 5) = 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_point_l2873_287367


namespace NUMINAMATH_CALUDE_inequality_properties_l2873_287329

theorem inequality_properties (a b c : ℝ) : 
  (a^2 > b^2 → abs a > abs b) ∧ 
  (a > b ↔ a + c > b + c) := by sorry

end NUMINAMATH_CALUDE_inequality_properties_l2873_287329


namespace NUMINAMATH_CALUDE_school_survey_methods_l2873_287304

/-- Represents a sampling method --/
inductive SamplingMethod
  | SimpleRandomSampling
  | StratifiedSampling
  | SystematicSampling

/-- Represents a survey with two sampling methods --/
structure Survey where
  totalStudents : Nat
  firstMethod : SamplingMethod
  secondMethod : SamplingMethod

/-- Defines the specific survey conducted by the school --/
def schoolSurvey : Survey :=
  { totalStudents := 200,
    firstMethod := SamplingMethod.SimpleRandomSampling,
    secondMethod := SamplingMethod.SystematicSampling }

/-- Theorem stating that the school survey uses Simple Random Sampling for the first method
    and Systematic Sampling for the second method --/
theorem school_survey_methods :
  schoolSurvey.firstMethod = SamplingMethod.SimpleRandomSampling ∧
  schoolSurvey.secondMethod = SamplingMethod.SystematicSampling :=
by sorry

end NUMINAMATH_CALUDE_school_survey_methods_l2873_287304


namespace NUMINAMATH_CALUDE_cos_15_degrees_l2873_287391

theorem cos_15_degrees : Real.cos (15 * π / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_15_degrees_l2873_287391


namespace NUMINAMATH_CALUDE_slower_speed_calculation_l2873_287347

/-- Proves that the slower speed is 8.4 km/hr given the conditions of the problem -/
theorem slower_speed_calculation (actual_distance : ℝ) (faster_speed : ℝ) (additional_distance : ℝ)
  (h1 : actual_distance = 50)
  (h2 : faster_speed = 14)
  (h3 : additional_distance = 20)
  : ∃ slower_speed : ℝ,
    slower_speed = 8.4 ∧
    (actual_distance / faster_speed = (actual_distance - additional_distance) / slower_speed) := by
  sorry

end NUMINAMATH_CALUDE_slower_speed_calculation_l2873_287347


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2873_287354

theorem inequality_solution_set (m : ℝ) : 
  (∀ x : ℝ, m * (x - 1) > x^2 - x ↔ 1 < x ∧ x < 2) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2873_287354


namespace NUMINAMATH_CALUDE_triangle_special_angles_l2873_287355

theorem triangle_special_angles (A B C : ℝ) (a b c : ℝ) :
  A > 0 ∧ B > 0 ∧ C > 0 ∧  -- Angles are positive
  A + B + C = Real.pi ∧    -- Sum of angles in a triangle
  C = 2 * A ∧              -- Angle C is twice angle A
  b = 2 * a ∧              -- Side b is twice side a
  a * Real.sin B = b * Real.sin A ∧  -- Law of sines
  a * Real.sin C = c * Real.sin A ∧  -- Law of sines
  a^2 + b^2 = c^2          -- Pythagorean theorem
  →
  A = Real.pi / 6 ∧ B = Real.pi / 2 ∧ C = Real.pi / 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_special_angles_l2873_287355


namespace NUMINAMATH_CALUDE_line_through_intersection_l2873_287396

/-- The line l: ax - y + b = 0 passes through the intersection point of 
    lines l₁: 2x - 2y - 3 = 0 and l₂: 3x - 5y + 1 = 0 
    if and only if 17a + 4b = 11 -/
theorem line_through_intersection (a b : ℝ) : 
  (∃ x y : ℝ, 2*x - 2*y - 3 = 0 ∧ 3*x - 5*y + 1 = 0 ∧ a*x - y + b = 0) ↔ 
  17*a + 4*b = 11 := by
sorry

end NUMINAMATH_CALUDE_line_through_intersection_l2873_287396
