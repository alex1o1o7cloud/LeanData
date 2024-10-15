import Mathlib

namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l437_43709

theorem sqrt_equation_solution :
  ∀ x : ℝ, Real.sqrt (3 + Real.sqrt x) = 4 → x = 169 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l437_43709


namespace NUMINAMATH_CALUDE_fraction_problem_l437_43746

theorem fraction_problem (N : ℝ) (F : ℝ) 
  (h1 : (1/4) * (1/3) * F * N = 15)
  (h2 : 0.40 * N = 180) : 
  F = 2/5 := by sorry

end NUMINAMATH_CALUDE_fraction_problem_l437_43746


namespace NUMINAMATH_CALUDE_least_multiple_36_with_digit_product_multiple_9_l437_43725

def is_multiple_of (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def digit_product (n : ℕ) : ℕ :=
  if n = 0 then 0 else
  let digits := n.digits 10
  digits.foldl (· * ·) 1

theorem least_multiple_36_with_digit_product_multiple_9 :
  ∀ n : ℕ, n > 0 →
    is_multiple_of n 36 →
    is_multiple_of (digit_product n) 9 →
    n ≥ 36 :=
sorry

end NUMINAMATH_CALUDE_least_multiple_36_with_digit_product_multiple_9_l437_43725


namespace NUMINAMATH_CALUDE_croissant_making_time_l437_43718

-- Define the constants
def fold_time : ℕ := 5
def fold_count : ℕ := 4
def rest_time : ℕ := 75
def mixing_time : ℕ := 10
def baking_time : ℕ := 30

-- Define the theorem
theorem croissant_making_time :
  (fold_time * fold_count + 
   rest_time * fold_count + 
   mixing_time + 
   baking_time) / 60 = 6 := by
  sorry

end NUMINAMATH_CALUDE_croissant_making_time_l437_43718


namespace NUMINAMATH_CALUDE_power_calculation_l437_43740

theorem power_calculation : (-8 : ℝ)^2023 * (1/8 : ℝ)^2024 = -1/8 := by sorry

end NUMINAMATH_CALUDE_power_calculation_l437_43740


namespace NUMINAMATH_CALUDE_nested_sqrt_24_l437_43741

/-- The solution to the equation x = √(24 + x), where x is non-negative -/
theorem nested_sqrt_24 : 
  ∃ x : ℝ, x ≥ 0 ∧ x = Real.sqrt (24 + x) → x = 6 := by sorry

end NUMINAMATH_CALUDE_nested_sqrt_24_l437_43741


namespace NUMINAMATH_CALUDE_inequality_proof_l437_43762

theorem inequality_proof (x₁ x₂ y₁ y₂ : ℝ) (h : x₁^2 + x₂^2 ≤ 1) :
  (x₁*y₁ + x₂*y₂ - 1)^2 ≥ (x₁^2 + x₂^2 - 1)*(y₁^2 + y₂^2 - 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l437_43762


namespace NUMINAMATH_CALUDE_at_least_one_fails_l437_43700

-- Define the propositions
variable (p : Prop) -- "Student A passes the driving test"
variable (q : Prop) -- "Student B passes the driving test"

-- Define the theorem
theorem at_least_one_fails : (¬p ∨ ¬q) ↔ (∃ student, student = p ∨ student = q) ∧ (¬student) :=
sorry

end NUMINAMATH_CALUDE_at_least_one_fails_l437_43700


namespace NUMINAMATH_CALUDE_exchange_impossibility_l437_43764

theorem exchange_impossibility : ¬ ∃ (N : ℤ), 5 * N = 2001 := by sorry

end NUMINAMATH_CALUDE_exchange_impossibility_l437_43764


namespace NUMINAMATH_CALUDE_london_to_edinburgh_distance_l437_43747

theorem london_to_edinburgh_distance :
  ∀ D : ℝ,
  (∃ x : ℝ, x = 200 ∧ x + 3.5 = D / 2) →
  D = 393 :=
by
  sorry

end NUMINAMATH_CALUDE_london_to_edinburgh_distance_l437_43747


namespace NUMINAMATH_CALUDE_cubic_function_property_l437_43768

/-- A cubic function g(x) with specific properties -/
def g (p q r s : ℝ) (x : ℝ) : ℝ := p * x^3 + q * x^2 + r * x + s

theorem cubic_function_property (p q r s : ℝ) :
  g p q r s 1 = 1 →
  g p q r s 3 = 1 →
  g p q r s 2 = 2 →
  (fun x ↦ 3 * p * x^2 + 2 * q * x + r) 2 = 0 →
  3 * p - 2 * q + r - 4 * s = -2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_property_l437_43768


namespace NUMINAMATH_CALUDE_expected_prize_money_l437_43745

theorem expected_prize_money (a₁ : ℝ) : 
  a₁ > 0 →  -- Probability should be positive
  a₁ + 2 * a₁ + 4 * a₁ = 1 →  -- Sum of probabilities is 1
  700 * a₁ + 560 * (2 * a₁) + 420 * (4 * a₁) = 500 := by
  sorry

end NUMINAMATH_CALUDE_expected_prize_money_l437_43745


namespace NUMINAMATH_CALUDE_jason_pokemon_cards_l437_43712

theorem jason_pokemon_cards (initial : ℕ) : 
  initial - 9 = 4 → initial = 13 := by
  sorry

end NUMINAMATH_CALUDE_jason_pokemon_cards_l437_43712


namespace NUMINAMATH_CALUDE_field_trip_students_l437_43758

theorem field_trip_students (teachers : ℕ) (student_ticket_cost adult_ticket_cost total_cost : ℚ) :
  teachers = 4 →
  student_ticket_cost = 1 →
  adult_ticket_cost = 3 →
  total_cost = 24 →
  ∃ (students : ℕ), students * student_ticket_cost + teachers * adult_ticket_cost = total_cost ∧ students = 12 :=
by sorry

end NUMINAMATH_CALUDE_field_trip_students_l437_43758


namespace NUMINAMATH_CALUDE_equal_split_probability_eight_dice_l437_43717

theorem equal_split_probability_eight_dice (n : ℕ) (p : ℝ) : 
  n = 8 →
  p = 1 / 2 →
  (n.choose (n / 2)) * p^n = 35 / 128 :=
by sorry

end NUMINAMATH_CALUDE_equal_split_probability_eight_dice_l437_43717


namespace NUMINAMATH_CALUDE_pirate_costume_cost_l437_43750

theorem pirate_costume_cost (num_friends : ℕ) (cost_per_costume : ℕ) : 
  num_friends = 8 → cost_per_costume = 5 → num_friends * cost_per_costume = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_pirate_costume_cost_l437_43750


namespace NUMINAMATH_CALUDE_jacobs_age_l437_43793

/-- Proves Jacob's age given the conditions of the problem -/
theorem jacobs_age :
  ∀ (rehana_age phoebe_age jacob_age : ℕ),
  rehana_age = 25 →
  rehana_age + 5 = 3 * (phoebe_age + 5) →
  jacob_age = (3 * phoebe_age) / 5 →
  jacob_age = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_jacobs_age_l437_43793


namespace NUMINAMATH_CALUDE_least_four_digit_7_heavy_l437_43770

def is_7_heavy (n : ℕ) : Prop := n % 7 > 3

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem least_four_digit_7_heavy : 
  (∀ n : ℕ, is_four_digit n → is_7_heavy n → 1000 ≤ n) ∧ 
  is_four_digit 1000 ∧ 
  is_7_heavy 1000 :=
sorry

end NUMINAMATH_CALUDE_least_four_digit_7_heavy_l437_43770


namespace NUMINAMATH_CALUDE_max_altitude_product_right_triangle_l437_43736

/-- Given a fixed side length and area, the product of altitudes is maximum for a right triangle --/
theorem max_altitude_product_right_triangle 
  (l : ℝ) (S : ℝ) (h_pos_l : l > 0) (h_pos_S : S > 0) :
  ∃ (a b : ℝ),
    a > 0 ∧ b > 0 ∧
    (1/2) * l * b = S ∧
    ∀ (x y : ℝ), x > 0 → y > 0 → (1/2) * l * y = S →
      (2*S/l) * (2*S/(l*x)) * (2*S/(l*y)) ≤ (2*S/l) * (2*S/(l*a)) * (2*S/(l*b)) :=
sorry

end NUMINAMATH_CALUDE_max_altitude_product_right_triangle_l437_43736


namespace NUMINAMATH_CALUDE_transylvanian_identity_l437_43767

-- Define the possible states of being
inductive State
| Human
| Vampire

-- Define the possible states of mind
inductive Mind
| Sane
| Insane

-- Define a person as a combination of state and mind
structure Person :=
  (state : State)
  (mind : Mind)

-- Define the statement made by the Transylvanian
def transylvanian_statement (p : Person) : Prop :=
  p.state = State.Human ∨ p.mind = Mind.Sane

-- Define the condition that insane vampires only make true statements
axiom insane_vampire_truth (p : Person) :
  p.state = State.Vampire ∧ p.mind = Mind.Insane → transylvanian_statement p

-- Theorem: The Transylvanian must be a human and sane
theorem transylvanian_identity :
  ∃ (p : Person), p.state = State.Human ∧ p.mind = Mind.Sane ∧ transylvanian_statement p :=
by sorry

end NUMINAMATH_CALUDE_transylvanian_identity_l437_43767


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l437_43776

def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B : Set ℝ := {-3, -1, 1, 3}

theorem intersection_of_A_and_B : A ∩ B = {-1, 1, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l437_43776


namespace NUMINAMATH_CALUDE_seven_square_side_length_l437_43755

/-- Represents a shape composed of seven equal squares -/
structure SevenSquareShape :=
  (side_length : ℝ)

/-- Represents a line that divides the shape into two equal areas -/
structure DividingLine :=
  (shape : SevenSquareShape)
  (divides_equally : Bool)

/-- Represents the intersection points of the dividing line with the shape -/
structure IntersectionPoints :=
  (line : DividingLine)
  (cf_length : ℝ)
  (ae_length : ℝ)
  (sum_cf_ae : cf_length + ae_length = 91)

/-- Theorem: The side length of each small square is 26 cm -/
theorem seven_square_side_length
  (shape : SevenSquareShape)
  (line : DividingLine)
  (points : IntersectionPoints)
  (h1 : line.shape = shape)
  (h2 : line.divides_equally = true)
  (h3 : points.line = line)
  : shape.side_length = 26 :=
by sorry

end NUMINAMATH_CALUDE_seven_square_side_length_l437_43755


namespace NUMINAMATH_CALUDE_largest_divisor_of_n4_minus_n2_l437_43715

theorem largest_divisor_of_n4_minus_n2 (n : ℤ) : 
  (∃ (k : ℤ), n^4 - n^2 = 12 * k) ∧ 
  (∀ (m : ℤ), m > 12 → ∃ (n : ℤ), ¬∃ (k : ℤ), n^4 - n^2 = m * k) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n4_minus_n2_l437_43715


namespace NUMINAMATH_CALUDE_tire_repair_cost_is_seven_l437_43731

/-- The cost of repairing one tire without sales tax, given the total cost for 4 tires and the sales tax per tire. -/
def tire_repair_cost (total_cost : ℚ) (sales_tax : ℚ) : ℚ :=
  (total_cost - 4 * sales_tax) / 4

/-- Theorem stating that the cost of repairing one tire without sales tax is $7,
    given a total cost of $30 for 4 tires and a sales tax of $0.50 per tire. -/
theorem tire_repair_cost_is_seven :
  tire_repair_cost 30 0.5 = 7 := by
  sorry

end NUMINAMATH_CALUDE_tire_repair_cost_is_seven_l437_43731


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l437_43765

/-- A geometric sequence with positive terms where a₁ and a₉₉ are roots of x² - 10x + 16 = 0 -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ 
  (∃ r, ∀ n, a (n + 1) = r * a n) ∧
  (a 1 * a 99 = 16) ∧
  (a 1 + a 99 = 10)

/-- The product of specific terms in the geometric sequence equals 64 -/
theorem geometric_sequence_product (a : ℕ → ℝ) (h : GeometricSequence a) :
  a 20 * a 50 * a 80 = 64 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l437_43765


namespace NUMINAMATH_CALUDE_not_perfect_square_l437_43742

theorem not_perfect_square (m n : ℕ) : ¬ ∃ k : ℕ, 1 + 3^m + 3^n = k^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l437_43742


namespace NUMINAMATH_CALUDE_permutations_three_distinct_l437_43714

/-- The number of distinct permutations of three distinct elements -/
def num_permutations_three_distinct : ℕ := 6

/-- Theorem: The number of distinct permutations of three distinct elements is 6 -/
theorem permutations_three_distinct :
  num_permutations_three_distinct = 6 := by
  sorry

end NUMINAMATH_CALUDE_permutations_three_distinct_l437_43714


namespace NUMINAMATH_CALUDE_new_assessed_value_calculation_l437_43757

/-- Represents the property tax calculation in Township K -/
structure PropertyTax where
  initialValue : ℝ
  newValue : ℝ
  taxRate : ℝ
  taxIncrease : ℝ

/-- Theorem stating the relationship between tax increase and new assessed value -/
theorem new_assessed_value_calculation (p : PropertyTax)
  (h1 : p.initialValue = 20000)
  (h2 : p.taxRate = 0.1)
  (h3 : p.taxIncrease = 800)
  (h4 : p.taxRate * p.newValue - p.taxRate * p.initialValue = p.taxIncrease) :
  p.newValue = 28000 := by
  sorry

end NUMINAMATH_CALUDE_new_assessed_value_calculation_l437_43757


namespace NUMINAMATH_CALUDE_solve_age_problem_l437_43722

def age_problem (a b : ℕ) : Prop :=
  (a + 10 = 2 * (b - 10)) ∧ (a = b + 9)

theorem solve_age_problem :
  ∃ (a b : ℕ), age_problem a b ∧ b = 39 :=
sorry

end NUMINAMATH_CALUDE_solve_age_problem_l437_43722


namespace NUMINAMATH_CALUDE_paint_wall_theorem_l437_43774

/-- The number of people needed to paint a wall in a given time, assuming a constant rate of painting. -/
def people_needed (initial_people : ℕ) (initial_time : ℕ) (new_time : ℕ) : ℕ :=
  (initial_people * initial_time) / new_time

/-- The additional number of people needed to paint a wall in a shorter time. -/
def additional_people_needed (initial_people : ℕ) (initial_time : ℕ) (new_time : ℕ) : ℕ :=
  (people_needed initial_people initial_time new_time) - initial_people

theorem paint_wall_theorem (initial_people : ℕ) (initial_time : ℕ) (new_time : ℕ) 
  (h1 : initial_people = 8) 
  (h2 : initial_time = 3) 
  (h3 : new_time = 2) :
  additional_people_needed initial_people initial_time new_time = 4 := by
  sorry

#check paint_wall_theorem

end NUMINAMATH_CALUDE_paint_wall_theorem_l437_43774


namespace NUMINAMATH_CALUDE_arithmetic_evaluation_l437_43779

theorem arithmetic_evaluation : 5 + 2 * (8 - 3) = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_evaluation_l437_43779


namespace NUMINAMATH_CALUDE_function_properties_l437_43782

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define g' as the derivative of g
variable (g' : ℝ → ℝ)
variable (h : ∀ x, HasDerivAt g (g' x) x)

-- Define the conditions
variable (cond1 : ∀ x, f x + g' x - 10 = 0)
variable (cond2 : ∀ x, f x - g' (4 - x) - 10 = 0)
variable (cond3 : ∀ x, g x = g (-x))  -- g is an even function

-- Theorem statement
theorem function_properties :
  (f 1 + f 3 = 20) ∧ (f 4 = 10) ∧ (f 2022 = 10) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l437_43782


namespace NUMINAMATH_CALUDE_min_value_expression_l437_43778

theorem min_value_expression (p x : ℝ) (h1 : 0 < p) (h2 : p < 15) (h3 : p ≤ x) (h4 : x ≤ 15) :
  (∀ y, p ≤ y ∧ y ≤ 15 → |x - p| + |x - 15| + |x - p - 15| ≤ |y - p| + |y - 15| + |y - p - 15|) →
  |x - p| + |x - 15| + |x - p - 15| = 15 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l437_43778


namespace NUMINAMATH_CALUDE_x_wins_in_six_moves_l437_43702

/-- Represents a position on the infinite grid -/
structure Position :=
  (x : ℤ) (y : ℤ)

/-- Represents a player in the game -/
inductive Player
  | X
  | O

/-- Represents the state of the game -/
structure GameState :=
  (moves : List (Player × Position))
  (currentPlayer : Player)

/-- Checks if a given list of positions forms a winning line -/
def isWinningLine (line : List Position) : Bool :=
  sorry

/-- Checks if the current game state is a win for the given player -/
def isWin (state : GameState) (player : Player) : Bool :=
  sorry

/-- Represents a strategy for playing the game -/
def Strategy := GameState → Position

/-- The theorem stating that X has a winning strategy in at most 6 moves -/
theorem x_wins_in_six_moves :
  ∃ (strategy : Strategy),
    ∀ (opponent_strategy : Strategy),
      ∃ (final_state : GameState),
        (final_state.moves.length ≤ 6) ∧
        (isWin final_state Player.X) :=
  sorry

end NUMINAMATH_CALUDE_x_wins_in_six_moves_l437_43702


namespace NUMINAMATH_CALUDE_negation_equivalence_l437_43797

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l437_43797


namespace NUMINAMATH_CALUDE_article_cost_l437_43769

theorem article_cost (C : ℝ) (S : ℝ) : 
  S = 1.25 * C →                            -- 25% profit
  (0.8 * C + 0.3 * (0.8 * C) = S - 10.50) → -- 30% profit on reduced cost and price
  C = 50 := by
sorry

end NUMINAMATH_CALUDE_article_cost_l437_43769


namespace NUMINAMATH_CALUDE_fractional_inequality_solution_set_l437_43723

theorem fractional_inequality_solution_set (x : ℝ) : 
  (x - 1) / (2 * x - 1) ≤ 0 ↔ 1/2 < x ∧ x ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_fractional_inequality_solution_set_l437_43723


namespace NUMINAMATH_CALUDE_intersection_property_l437_43777

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in polar form √2ρcos(θ + π/4) = 1 -/
def Line : Type := Unit

/-- Represents a curve in polar form ρ = 2acosθ -/
def Curve (a : ℝ) : Type := Unit

/-- Returns true if the point is on the given line -/
def Point.onLine (p : Point) (l : Line) : Prop := sorry

/-- Returns true if the point is on the given curve -/
def Point.onCurve (p : Point) (c : Curve a) : Prop := sorry

/-- Calculates the squared distance between two points -/
def Point.distanceSquared (p q : Point) : ℝ := sorry

/-- Theorem: Given the conditions, prove that a = 3 -/
theorem intersection_property (a : ℝ) (l : Line) (c : Curve a) (M P Q : Point)
    (h₁ : a > 0)
    (h₂ : M.x = 0 ∧ M.y = -1)
    (h₃ : P.onLine l ∧ P.onCurve c)
    (h₄ : Q.onLine l ∧ Q.onCurve c)
    (h₅ : P.distanceSquared Q = 4 * P.distanceSquared M * Q.distanceSquared M) :
    a = 3 := by sorry

end NUMINAMATH_CALUDE_intersection_property_l437_43777


namespace NUMINAMATH_CALUDE_lcm_18_24_30_l437_43792

theorem lcm_18_24_30 : Nat.lcm (Nat.lcm 18 24) 30 = 360 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_24_30_l437_43792


namespace NUMINAMATH_CALUDE_jose_wrong_questions_l437_43711

theorem jose_wrong_questions (total_questions : ℕ) (marks_per_question : ℕ) 
  (meghan_score jose_score alisson_score : ℕ) : 
  total_questions = 50 →
  marks_per_question = 2 →
  meghan_score = jose_score - 20 →
  jose_score = alisson_score + 40 →
  meghan_score + jose_score + alisson_score = 210 →
  total_questions * marks_per_question - jose_score = 5 * marks_per_question :=
by sorry

end NUMINAMATH_CALUDE_jose_wrong_questions_l437_43711


namespace NUMINAMATH_CALUDE_amy_balloons_l437_43763

theorem amy_balloons (red green blue : ℕ) (h1 : red = 29) (h2 : green = 17) (h3 : blue = 21) :
  red + green + blue = 67 := by
  sorry

end NUMINAMATH_CALUDE_amy_balloons_l437_43763


namespace NUMINAMATH_CALUDE_part_to_whole_ratio_l437_43737

theorem part_to_whole_ratio (N P : ℝ) 
  (h1 : (1/4) * (1/3) * P = 10)
  (h2 : 0.40 * N = 120) : 
  P/N = 1/2.5 := by
sorry

end NUMINAMATH_CALUDE_part_to_whole_ratio_l437_43737


namespace NUMINAMATH_CALUDE_total_trees_planted_l437_43713

theorem total_trees_planted (apricot_trees peach_trees : ℕ) : 
  apricot_trees = 58 →
  peach_trees = 3 * apricot_trees →
  apricot_trees + peach_trees = 232 := by
sorry

end NUMINAMATH_CALUDE_total_trees_planted_l437_43713


namespace NUMINAMATH_CALUDE_cherry_weekly_earnings_l437_43739

/-- Represents Cherry's delivery service earnings --/
def cherry_earnings : ℕ → ℚ
| 5 => 2.5  -- $2.50 for 5 kg cargo
| 8 => 4    -- $4 for 8 kg cargo
| _ => 0    -- Default case

/-- Calculates Cherry's daily earnings --/
def daily_earnings : ℚ :=
  4 * cherry_earnings 5 + 2 * cherry_earnings 8

/-- Theorem: Cherry's weekly earnings are $126 --/
theorem cherry_weekly_earnings :
  7 * daily_earnings = 126 := by
  sorry

end NUMINAMATH_CALUDE_cherry_weekly_earnings_l437_43739


namespace NUMINAMATH_CALUDE_cube_of_product_l437_43705

theorem cube_of_product (a b : ℝ) : (-2 * a * b^2)^3 = -8 * a^3 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_product_l437_43705


namespace NUMINAMATH_CALUDE_committee_formation_count_l437_43719

def schoolchildren : ℕ := 12
def teachers : ℕ := 3
def committee_size : ℕ := 9

theorem committee_formation_count :
  (Nat.choose (schoolchildren + teachers) committee_size) -
  (Nat.choose schoolchildren committee_size) = 4785 :=
by sorry

end NUMINAMATH_CALUDE_committee_formation_count_l437_43719


namespace NUMINAMATH_CALUDE_bob_pizza_calorie_intake_l437_43771

/-- Calculates the average calorie intake per slice for the slices Bob ate from a pizza -/
def average_calorie_intake (total_slices : ℕ) (low_cal_slices : ℕ) (high_cal_slices : ℕ) (low_cal : ℕ) (high_cal : ℕ) : ℚ :=
  (low_cal_slices * low_cal + high_cal_slices * high_cal) / (low_cal_slices + high_cal_slices)

/-- Theorem stating that the average calorie intake per slice for the slices Bob ate is approximately 357.14 calories -/
theorem bob_pizza_calorie_intake :
  average_calorie_intake 12 3 4 300 400 = 2500 / 7 := by
  sorry

end NUMINAMATH_CALUDE_bob_pizza_calorie_intake_l437_43771


namespace NUMINAMATH_CALUDE_system_solution_l437_43798

theorem system_solution (x y : ℝ) : x + y = -5 ∧ 2*y = -2 → x = -4 ∧ y = -1 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l437_43798


namespace NUMINAMATH_CALUDE_cube_cutting_theorem_l437_43733

/-- A plane in 3D space --/
structure Plane where
  normal : ℝ × ℝ × ℝ
  distance : ℝ

/-- A part of a cube resulting from cuts --/
structure CubePart where
  points : Set (ℝ × ℝ × ℝ)

/-- Function to calculate the maximum distance between any two points in a set --/
def maxDistance (s : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

/-- Function to cut a unit cube with given planes --/
def cutCube (planes : List Plane) : List CubePart := sorry

theorem cube_cutting_theorem :
  (∃ (planes : List Plane), planes.length = 4 ∧ 
    ∀ part ∈ cutCube planes, maxDistance part.points < 4/5) ∧
  (¬ ∃ (planes : List Plane), planes.length = 4 ∧ 
    ∀ part ∈ cutCube planes, maxDistance part.points < 4/7) := by
  sorry

end NUMINAMATH_CALUDE_cube_cutting_theorem_l437_43733


namespace NUMINAMATH_CALUDE_aristocrat_spending_l437_43761

theorem aristocrat_spending (total_people : ℕ) (men_amount : ℕ) (women_amount : ℕ)
  (men_fraction : ℚ) (women_fraction : ℚ) :
  total_people = 3552 →
  men_amount = 45 →
  women_amount = 60 →
  men_fraction = 1/9 →
  women_fraction = 1/12 →
  ∃ (men women : ℕ),
    men + women = total_people ∧
    (men_fraction * men * men_amount + women_fraction * women * women_amount : ℚ) = 17760 :=
by sorry

end NUMINAMATH_CALUDE_aristocrat_spending_l437_43761


namespace NUMINAMATH_CALUDE_value_of_expression_l437_43735

-- Define the function g
def g (p q r s : ℝ) (x : ℝ) : ℝ := p * x^3 + q * x^2 + r * x + s

-- State the theorem
theorem value_of_expression (p q r s : ℝ) : g p q r s 3 = 6 → 6*p - 3*q + 2*r - s = 0 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l437_43735


namespace NUMINAMATH_CALUDE_cos_arithmetic_sequence_product_l437_43780

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

def S (a₁ : ℝ) : Set ℝ := {x | ∃ n : ℕ+, x = Real.cos (arithmetic_sequence a₁ (2 * Real.pi / 3) n)}

theorem cos_arithmetic_sequence_product (a₁ : ℝ) :
  ∃ a b : ℝ, S a₁ = {a, b} → a * b = -1/2 := by sorry

end NUMINAMATH_CALUDE_cos_arithmetic_sequence_product_l437_43780


namespace NUMINAMATH_CALUDE_solution_set_l437_43790

noncomputable def Solutions (a : ℝ) : Set (ℝ × ℝ × ℝ) :=
  { (1, Real.sqrt (-a), -Real.sqrt (-a)),
    (1, -Real.sqrt (-a), Real.sqrt (-a)),
    (Real.sqrt (-a), -Real.sqrt (-a), 1),
    (-Real.sqrt (-a), 1, Real.sqrt (-a)),
    (Real.sqrt (-a), 1, -Real.sqrt (-a)),
    (-Real.sqrt (-a), Real.sqrt (-a), 1) }

theorem solution_set (a : ℝ) :
  ∀ (x y z : ℝ),
    (x + y + z = 1 ∧
     1/x + 1/y + 1/z = 1 ∧
     x*y*z = a) ↔
    (x, y, z) ∈ Solutions a := by
  sorry

end NUMINAMATH_CALUDE_solution_set_l437_43790


namespace NUMINAMATH_CALUDE_smallest_of_three_consecutive_odds_l437_43789

theorem smallest_of_three_consecutive_odds (x y z : ℤ) : 
  (∃ k : ℤ, x = 2*k + 1) →  -- x is odd
  y = x + 2 →               -- y is the next consecutive odd number
  z = y + 2 →               -- z is the next consecutive odd number after y
  x + y + z = 69 →          -- their sum is 69
  x = 21 :=                 -- the smallest number (x) is 21
by
  sorry

end NUMINAMATH_CALUDE_smallest_of_three_consecutive_odds_l437_43789


namespace NUMINAMATH_CALUDE_complex_equation_solution_l437_43748

theorem complex_equation_solution (z : ℂ) : z + Complex.abs z * Complex.I = 3 + 9 * Complex.I → z = 3 + 4 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l437_43748


namespace NUMINAMATH_CALUDE_remainder_theorem_l437_43772

theorem remainder_theorem (n : ℤ) (k : ℤ) (h : n = 60 * k - 3) :
  (n^3 + 2*n^2 + 3*n + 4) % 60 = 46 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l437_43772


namespace NUMINAMATH_CALUDE_special_function_characterization_l437_43720

/-- A function f: ℝ² → ℝ satisfying specific conditions -/
def SpecialFunction (f : ℝ → ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f x y = f y x) ∧
  (∀ x y z : ℝ, (f x y - f y z) * (f y z - f z x) * (f z x - f x y) = 0) ∧
  (∀ x y a : ℝ, f (x + a) (y + a) = f x y + a) ∧
  (∀ x y : ℝ, x ≤ y → f 0 x ≤ f 0 y)

/-- The theorem stating that any SpecialFunction must be of a specific form -/
theorem special_function_characterization (f : ℝ → ℝ → ℝ) (hf : SpecialFunction f) :
  ∃ a : ℝ, (∀ x y : ℝ, f x y = a + min x y) ∨ (∀ x y : ℝ, f x y = a + max x y) := by
  sorry

end NUMINAMATH_CALUDE_special_function_characterization_l437_43720


namespace NUMINAMATH_CALUDE_divides_condition_l437_43783

theorem divides_condition (p k r : ℕ) : 
  Prime p → 
  k > 0 → 
  r > 0 → 
  p > r → 
  (pk + r) ∣ (p^p + 1) → 
  r ∣ k := by
sorry

end NUMINAMATH_CALUDE_divides_condition_l437_43783


namespace NUMINAMATH_CALUDE_lucky_larry_calculation_l437_43781

theorem lucky_larry_calculation (a b c d e : ℚ) : 
  a = 2 ∧ b = 3 ∧ c = 4 ∧ d = 6 →
  a * b + c * d - c * e + 1 = a * (b + (c * (d - e))) →
  e = 23/4 := by
sorry

end NUMINAMATH_CALUDE_lucky_larry_calculation_l437_43781


namespace NUMINAMATH_CALUDE_stating_no_room_for_other_animals_l437_43788

/-- Represents the composition of animals in a circus --/
structure CircusAnimals where
  total : ℕ
  lions : ℕ
  tigers : ℕ
  h_lions : lions = (total - lions) / 5
  h_tigers : tigers = total - tigers + 5

/-- 
Theorem stating that in a circus where the number of lions is 1/5 of the number of non-lions, 
and the number of tigers is 5 more than the number of non-tigers, 
there is no room for any other animals.
-/
theorem no_room_for_other_animals (c : CircusAnimals) : 
  c.lions + c.tigers = c.total :=
sorry

end NUMINAMATH_CALUDE_stating_no_room_for_other_animals_l437_43788


namespace NUMINAMATH_CALUDE_opposite_face_is_A_l437_43773

/-- Represents the labels of the squares --/
inductive Label
  | A | B | C | D | E | F

/-- Represents a cube formed by folding six squares --/
structure Cube where
  top : Label
  bottom : Label
  front : Label
  back : Label
  left : Label
  right : Label

/-- Represents the linear arrangement of squares before folding --/
def LinearArrangement := List Label

/-- Function to create a cube from a linear arrangement of squares --/
def foldCube (arrangement : LinearArrangement) (top : Label) : Cube :=
  sorry

/-- The theorem to be proved --/
theorem opposite_face_is_A 
  (arrangement : LinearArrangement) 
  (h1 : arrangement = [Label.A, Label.B, Label.C, Label.D, Label.E, Label.F]) 
  (cube : Cube) 
  (h2 : cube = foldCube arrangement Label.B) : 
  cube.bottom = Label.A :=
sorry

end NUMINAMATH_CALUDE_opposite_face_is_A_l437_43773


namespace NUMINAMATH_CALUDE_village_foods_monthly_sales_l437_43729

/-- Represents the monthly sales data for Village Foods --/
structure VillageFoodsSales where
  customers_per_month : ℕ
  lettuce_per_customer : ℕ
  lettuce_price : ℚ
  tomatoes_per_customer : ℕ
  tomato_price : ℚ

/-- Calculates the total monthly sales of lettuce and tomatoes --/
def total_monthly_sales (sales : VillageFoodsSales) : ℚ :=
  sales.customers_per_month * 
  (sales.lettuce_per_customer * sales.lettuce_price + 
   sales.tomatoes_per_customer * sales.tomato_price)

/-- Theorem stating that the total monthly sales of lettuce and tomatoes is $2000 --/
theorem village_foods_monthly_sales :
  let sales : VillageFoodsSales := {
    customers_per_month := 500,
    lettuce_per_customer := 2,
    lettuce_price := 1,
    tomatoes_per_customer := 4,
    tomato_price := 1/2
  }
  total_monthly_sales sales = 2000 := by sorry

end NUMINAMATH_CALUDE_village_foods_monthly_sales_l437_43729


namespace NUMINAMATH_CALUDE_some_number_solution_l437_43708

theorem some_number_solution : 
  ∃ x : ℝ, 4.7 * x + 4.7 * 9.43 + 4.7 * 77.31 = 470 ∧ x = 13.26 := by
  sorry

end NUMINAMATH_CALUDE_some_number_solution_l437_43708


namespace NUMINAMATH_CALUDE_second_integer_value_l437_43751

theorem second_integer_value (a b c d : ℤ) : 
  (∃ x : ℤ, a = x ∧ b = x + 2 ∧ c = x + 4 ∧ d = x + 6) →  -- consecutive even integers
  (a + d = 156) →                                        -- sum of first and fourth is 156
  b = 77                                                 -- second integer is 77
:= by sorry

end NUMINAMATH_CALUDE_second_integer_value_l437_43751


namespace NUMINAMATH_CALUDE_symmetric_function_value_l437_43710

/-- A function symmetric about x=1 -/
def SymmetricAboutOne (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (2 - x)

/-- The main theorem -/
theorem symmetric_function_value (f : ℝ → ℝ) 
  (h_sym : SymmetricAboutOne f)
  (h_def : ∀ x ≥ 1, f x = x * (1 - x)) : 
  f (-2) = -12 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_function_value_l437_43710


namespace NUMINAMATH_CALUDE_probability_tamika_greater_carlos_l437_43786

def tamika_set : Finset ℕ := {10, 11, 12}
def carlos_set : Finset ℕ := {4, 6, 7}

def tamika_sums : Finset ℕ := {21, 22, 23}
def carlos_sums : Finset ℕ := {10, 11, 13}

def favorable_outcomes : ℕ := (tamika_sums.card * carlos_sums.card)

def total_outcomes : ℕ := (tamika_sums.card * carlos_sums.card)

theorem probability_tamika_greater_carlos :
  (favorable_outcomes : ℚ) / total_outcomes = 1 := by sorry

end NUMINAMATH_CALUDE_probability_tamika_greater_carlos_l437_43786


namespace NUMINAMATH_CALUDE_negative_seven_x_is_product_l437_43706

theorem negative_seven_x_is_product : ∀ x : ℝ, -7 * x = -7 * x := by sorry

end NUMINAMATH_CALUDE_negative_seven_x_is_product_l437_43706


namespace NUMINAMATH_CALUDE_triangle_side_length_l437_43785

noncomputable def triangleConfiguration (OA OC OD OB BD : ℝ) : ℝ → Prop :=
  λ y => OA = 5 ∧ OC = 12 ∧ OD = 5 ∧ OB = 3 ∧ BD = 6 ∧ 
    y^2 = OA^2 + OC^2 - 2 * OA * OC * ((OD^2 + OB^2 - BD^2) / (2 * OD * OB))

theorem triangle_side_length : 
  ∃ (OA OC OD OB BD : ℝ), triangleConfiguration OA OC OD OB BD (3 * Real.sqrt 67) :=
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l437_43785


namespace NUMINAMATH_CALUDE_complement_of_P_P_subset_Q_range_P_inter_Q_eq_Q_range_final_range_of_m_l437_43734

-- Define sets P and Q
def P : Set ℝ := {x | -2 ≤ x ∧ x ≤ 10}
def Q (m : ℝ) : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ 1 + m}

-- Theorem for the complement of P
theorem complement_of_P : (Set.univ \ P) = {x | x < -2 ∨ x > 10} := by sorry

-- Theorem for P being a subset of Q
theorem P_subset_Q_range (m : ℝ) : P ⊆ Q m ↔ m ≥ 9 := by sorry

-- Theorem for intersection of P and Q equals Q
theorem P_inter_Q_eq_Q_range (m : ℝ) : P ∩ Q m = Q m ↔ m ≤ 9 := by sorry

-- Theorem for the final range of m satisfying both conditions
theorem final_range_of_m : 
  {m : ℝ | P ⊆ Q m ∧ P ∩ Q m = Q m} = {m : ℝ | 9 ≤ m ∧ m ≤ 9} := by sorry

end NUMINAMATH_CALUDE_complement_of_P_P_subset_Q_range_P_inter_Q_eq_Q_range_final_range_of_m_l437_43734


namespace NUMINAMATH_CALUDE_reservoir_capacity_difference_l437_43707

/-- Proves that the difference between total capacity and normal level is 25 million gallons --/
theorem reservoir_capacity_difference (current_amount : ℝ) (normal_level : ℝ) (total_capacity : ℝ)
  (h1 : current_amount = 30)
  (h2 : current_amount = 2 * normal_level)
  (h3 : current_amount = 0.75 * total_capacity) :
  total_capacity - normal_level = 25 := by
  sorry

end NUMINAMATH_CALUDE_reservoir_capacity_difference_l437_43707


namespace NUMINAMATH_CALUDE_expression_one_equality_expression_two_equality_l437_43795

-- Expression 1
theorem expression_one_equality : 
  0.25 * (-1/2)^(-4) - 4 / 2^0 - (1/16)^(-1/2) = -4 := by sorry

-- Expression 2
theorem expression_two_equality :
  2 * (Real.log 2 / Real.log 3) - 
  (Real.log (32/9) / Real.log 3) + 
  (Real.log 8 / Real.log 3) - 
  ((Real.log 3 / Real.log 4) + (Real.log 3 / Real.log 8)) * 
  ((Real.log 2 / Real.log 3) + (Real.log 2 / Real.log 9)) = 3/4 := by sorry

end NUMINAMATH_CALUDE_expression_one_equality_expression_two_equality_l437_43795


namespace NUMINAMATH_CALUDE_second_shop_amount_calculation_l437_43704

/-- The amount paid for books from the second shop -/
def second_shop_amount (books_shop1 books_shop2 : ℕ) (amount_shop1 avg_price : ℚ) : ℚ :=
  (books_shop1 + books_shop2 : ℚ) * avg_price - amount_shop1

/-- Theorem stating the amount paid for books from the second shop -/
theorem second_shop_amount_calculation :
  second_shop_amount 27 20 581 25 = 594 := by
  sorry

end NUMINAMATH_CALUDE_second_shop_amount_calculation_l437_43704


namespace NUMINAMATH_CALUDE_farm_chicken_count_l437_43724

/-- Represents the number of chickens on a farm -/
structure FarmChickens where
  roosters : ℕ
  hens : ℕ

/-- Given a farm with the specified conditions, proves that the total number of chickens is 75 -/
theorem farm_chicken_count (farm : FarmChickens) 
  (hen_count : farm.hens = 67)
  (rooster_hen_relation : farm.hens = 9 * farm.roosters - 5) :
  farm.roosters + farm.hens = 75 := by
  sorry


end NUMINAMATH_CALUDE_farm_chicken_count_l437_43724


namespace NUMINAMATH_CALUDE_valuable_heirlooms_percentage_l437_43716

theorem valuable_heirlooms_percentage
  (useful_percentage : Real)
  (junk_percentage : Real)
  (useful_items : Nat)
  (junk_items : Nat)
  (h1 : useful_percentage = 0.2)
  (h2 : junk_percentage = 0.7)
  (h3 : useful_items = 8)
  (h4 : junk_items = 28) :
  ∃ (total_items : Nat),
    (useful_items : Real) / total_items = useful_percentage ∧
    (junk_items : Real) / total_items = junk_percentage ∧
    1 - useful_percentage - junk_percentage = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_valuable_heirlooms_percentage_l437_43716


namespace NUMINAMATH_CALUDE_alcohol_mixture_theorem_alcohol_mixture_validity_l437_43732

/-- Proves that adding the calculated amount of alcohol results in the desired concentration -/
theorem alcohol_mixture_theorem (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c ≥ 0) (hd : d ≥ 0) 
  (hac : a ≠ c) (had : a ≠ d) (hcd : c ≠ d) :
  let x := b * (d - c) / (a - d)
  (b * c + x * a) / (b + x) = d :=
by sorry

/-- Proves that the solution is valid when d is between a and c -/
theorem alcohol_mixture_validity (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c ≥ 0) (hd : d ≥ 0) 
  (hac : a ≠ c) (had : a ≠ d) (hcd : c ≠ d) :
  (min a c < d ∧ d < max a c) → 
  let x := b * (d - c) / (a - d)
  x > 0 :=
by sorry

end NUMINAMATH_CALUDE_alcohol_mixture_theorem_alcohol_mixture_validity_l437_43732


namespace NUMINAMATH_CALUDE_afternoon_sales_l437_43701

/-- Represents the amount of pears sold in kilograms during different parts of the day -/
structure PearSales where
  morning : ℝ
  afternoon : ℝ
  evening : ℝ

/-- Defines the relationship between sales in different parts of the day -/
def valid_sales (s : PearSales) : Prop :=
  s.afternoon = 2 * s.morning ∧ 
  s.evening = 3 * s.afternoon ∧ 
  s.morning + s.afternoon + s.evening = 510

/-- Theorem stating that the afternoon sales equal 510 / 4.5 kg given the conditions -/
theorem afternoon_sales (s : PearSales) (h : valid_sales s) : 
  s.afternoon = 510 / 4.5 := by
  sorry

end NUMINAMATH_CALUDE_afternoon_sales_l437_43701


namespace NUMINAMATH_CALUDE_simplify_scientific_notation_l437_43796

theorem simplify_scientific_notation :
  (12 * 10^10) / (6 * 10^2) = 2 * 10^8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_scientific_notation_l437_43796


namespace NUMINAMATH_CALUDE_total_area_is_68_l437_43799

/-- Represents the dimensions of a rectangle -/
structure RectDimensions where
  width : ℕ
  height : ℕ

/-- Calculates the area of a rectangle given its dimensions -/
def rectangleArea (rect : RectDimensions) : ℕ :=
  rect.width * rect.height

/-- The dimensions of the four rectangles in the figure -/
def rect1 : RectDimensions := ⟨5, 7⟩
def rect2 : RectDimensions := ⟨3, 3⟩
def rect3 : RectDimensions := ⟨4, 1⟩
def rect4 : RectDimensions := ⟨5, 4⟩

/-- Theorem: The total area of the composite shape is 68 square units -/
theorem total_area_is_68 : 
  rectangleArea rect1 + rectangleArea rect2 + rectangleArea rect3 + rectangleArea rect4 = 68 := by
  sorry

end NUMINAMATH_CALUDE_total_area_is_68_l437_43799


namespace NUMINAMATH_CALUDE_system_solution_l437_43759

-- Define the system of equations
def system (x y : ℝ) : Prop :=
  x + y = 3 ∧ x - y = 1

-- Define the solution set
def solution_set : Set (ℝ × ℝ) :=
  {pair | system pair.1 pair.2}

-- Theorem statement
theorem system_solution :
  solution_set = {(2, 1)} := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l437_43759


namespace NUMINAMATH_CALUDE_shoe_size_ratio_l437_43730

def jasmine_shoe_size : ℕ := 7
def combined_shoe_size : ℕ := 21

def alexa_shoe_size : ℕ := combined_shoe_size - jasmine_shoe_size

theorem shoe_size_ratio : 
  alexa_shoe_size / jasmine_shoe_size = 2 := by sorry

end NUMINAMATH_CALUDE_shoe_size_ratio_l437_43730


namespace NUMINAMATH_CALUDE_smallest_consecutive_sequence_sum_l437_43754

theorem smallest_consecutive_sequence_sum (B : ℤ) : B = 1011 ↔ 
  (∀ k < B, ¬∃ n : ℕ+, (n : ℤ) * (2 * k + n - 1) = 2023 ∧ n > 1) ∧
  (∃ n : ℕ+, (n : ℤ) * (2 * B + n - 1) = 2023 ∧ n > 1) :=
by sorry

end NUMINAMATH_CALUDE_smallest_consecutive_sequence_sum_l437_43754


namespace NUMINAMATH_CALUDE_arcsin_sin_eq_x_div_3_solutions_l437_43775

theorem arcsin_sin_eq_x_div_3_solutions (x : ℝ) :
  -((3 * π) / 2) ≤ x ∧ x ≤ (3 * π) / 2 →
  (Real.arcsin (Real.sin x) = x / 3) ↔ 
  x ∈ ({-3*π, -2*π, -π, 0, π, 2*π, 3*π} : Set ℝ) :=
by sorry

end NUMINAMATH_CALUDE_arcsin_sin_eq_x_div_3_solutions_l437_43775


namespace NUMINAMATH_CALUDE_equation_solutions_l437_43784

theorem equation_solutions :
  (∀ x : ℝ, x^2 - 81 = 0 ↔ x = 9 ∨ x = -9) ∧
  (∀ x : ℝ, x^3 - 3 = 3/8 ↔ x = 3/2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l437_43784


namespace NUMINAMATH_CALUDE_water_addition_changes_ratio_l437_43721

/-- Proves that adding 3 litres of water to a 45-litre mixture with initial milk to water ratio of 4:1 results in a new mixture with milk to water ratio of 3:1 -/
theorem water_addition_changes_ratio :
  let initial_volume : ℝ := 45
  let initial_milk_ratio : ℝ := 4
  let initial_water_ratio : ℝ := 1
  let added_water : ℝ := 3
  let final_milk_ratio : ℝ := 3
  let final_water_ratio : ℝ := 1

  let initial_milk := initial_volume * initial_milk_ratio / (initial_milk_ratio + initial_water_ratio)
  let initial_water := initial_volume * initial_water_ratio / (initial_milk_ratio + initial_water_ratio)
  let final_water := initial_water + added_water

  initial_milk / final_water = final_milk_ratio / final_water_ratio :=
by
  sorry

#check water_addition_changes_ratio

end NUMINAMATH_CALUDE_water_addition_changes_ratio_l437_43721


namespace NUMINAMATH_CALUDE_integer_pair_inequality_l437_43756

theorem integer_pair_inequality (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  (1 ≤ m^n - n^m ∧ m^n - n^m ≤ m*n) ↔ 
  ((n = 1 ∧ m ≥ 2) ∨ (m = 2 ∧ n = 5) ∨ (m = 3 ∧ n = 2)) := by
sorry

end NUMINAMATH_CALUDE_integer_pair_inequality_l437_43756


namespace NUMINAMATH_CALUDE_initial_condition_recurrence_relation_diamonds_in_25th_figure_l437_43753

/-- The number of diamonds in the n-th figure of the sequence -/
def num_diamonds (n : ℕ) : ℕ :=
  if n = 1 then 1
  else 2 * n^2 + 2 * n - 3

/-- The sequence starts with one diamond in the first figure -/
theorem initial_condition : num_diamonds 1 = 1 := by sorry

/-- The recurrence relation for n ≥ 2 -/
theorem recurrence_relation (n : ℕ) (h : n ≥ 2) :
  num_diamonds n = num_diamonds (n-1) + 4*n := by sorry

/-- The main theorem: The 25th figure contains 1297 diamonds -/
theorem diamonds_in_25th_figure : num_diamonds 25 = 1297 := by sorry

end NUMINAMATH_CALUDE_initial_condition_recurrence_relation_diamonds_in_25th_figure_l437_43753


namespace NUMINAMATH_CALUDE_quadratic_equation_general_form_l437_43787

theorem quadratic_equation_general_form :
  ∀ x : ℝ, 3 * x * (x - 3) = 4 ↔ 3 * x^2 - 9 * x - 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_general_form_l437_43787


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l437_43766

theorem quadratic_solution_sum (c d : ℝ) : 
  (∀ x, x^2 - 6*x + 15 = 27 ↔ x = c ∨ x = d) →
  c ≥ d →
  3*c + 2*d = 15 + Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l437_43766


namespace NUMINAMATH_CALUDE_sum_of_specific_numbers_l437_43749

theorem sum_of_specific_numbers : 
  217 + 2.017 + 0.217 + 2.0017 = 221.2357 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_numbers_l437_43749


namespace NUMINAMATH_CALUDE_travel_problem_solvable_l437_43738

/-- A strategy for three friends to travel between two cities --/
structure TravelStrategy where
  /-- The time taken for all friends to reach their destinations --/
  total_time : ℝ
  /-- Assertion that the strategy is valid --/
  is_valid : Prop

/-- The travel problem setup --/
structure TravelProblem where
  /-- Distance between the two cities in km --/
  distance : ℝ
  /-- Maximum walking speed in km/h --/
  walk_speed : ℝ
  /-- Maximum cycling speed in km/h --/
  cycle_speed : ℝ

/-- The existence of a valid strategy for the given travel problem --/
def exists_valid_strategy (problem : TravelProblem) : Prop :=
  ∃ (strategy : TravelStrategy), 
    strategy.is_valid ∧ 
    strategy.total_time ≤ 160/60 ∧  -- 2 hours and 40 minutes in hours
    problem.distance = 24 ∧
    problem.walk_speed ≤ 6 ∧
    problem.cycle_speed ≤ 18

/-- Theorem stating that there exists a valid strategy for the given problem --/
theorem travel_problem_solvable : 
  ∃ (problem : TravelProblem), exists_valid_strategy problem :=
sorry

end NUMINAMATH_CALUDE_travel_problem_solvable_l437_43738


namespace NUMINAMATH_CALUDE_prime_square_in_A_implies_prime_in_A_l437_43726

/-- The set of positive integers of the form a^2 + 2b^2, where a and b are integers and b ≠ 0 -/
def A : Set ℕ+ :=
  {n : ℕ+ | ∃ (a b : ℤ), (b ≠ 0) ∧ (n : ℤ) = a^2 + 2*b^2}

/-- Theorem: If p is a prime number and p^2 is in A, then p is in A -/
theorem prime_square_in_A_implies_prime_in_A (p : ℕ+) (hp : Nat.Prime p) 
    (h_p_sq : (p^2 : ℕ+) ∈ A) : p ∈ A := by
  sorry

end NUMINAMATH_CALUDE_prime_square_in_A_implies_prime_in_A_l437_43726


namespace NUMINAMATH_CALUDE_ratio_problem_l437_43743

theorem ratio_problem (a b : ℕ) (h1 : a = 55) (h2 : a = 5 * b) : b = 11 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l437_43743


namespace NUMINAMATH_CALUDE_dog_weight_problem_l437_43703

theorem dog_weight_problem (x y : ℝ) :
  -- Define the weights of the dogs
  let w₂ : ℝ := 31
  let w₃ : ℝ := 35
  let w₄ : ℝ := 33
  let w₅ : ℝ := y
  -- The average of the first 4 dogs equals the average of all 5 dogs
  (x + w₂ + w₃ + w₄) / 4 = (x + w₂ + w₃ + w₄ + w₅) / 5 →
  -- The weight of the fifth dog is 31 pounds
  y = 31 →
  -- The weight of the first dog is 25 pounds
  x = 25 := by
sorry

end NUMINAMATH_CALUDE_dog_weight_problem_l437_43703


namespace NUMINAMATH_CALUDE_mice_meet_in_six_days_l437_43791

/-- The thickness of the wall in feet -/
def wall_thickness : ℚ := 64 + 31/32

/-- The distance burrowed by both mice after n days -/
def total_distance (n : ℕ) : ℚ := 2^n - 1/(2^(n-1)) + 1

/-- The number of days it takes for the mice to meet -/
def days_to_meet : ℕ := 6

/-- Theorem stating that the mice meet after 6 days -/
theorem mice_meet_in_six_days :
  total_distance days_to_meet = wall_thickness :=
sorry

end NUMINAMATH_CALUDE_mice_meet_in_six_days_l437_43791


namespace NUMINAMATH_CALUDE_hyperbola_equation_from_parameters_l437_43744

/-- Represents a hyperbola with given eccentricity and foci -/
structure Hyperbola where
  eccentricity : ℝ
  focus1 : ℝ × ℝ
  focus2 : ℝ × ℝ

/-- The equation of a hyperbola given its parameters -/
def hyperbola_equation (h : Hyperbola) : ℝ → ℝ → Prop :=
  fun x y => x^2 / 4 - y^2 / 12 = 1

/-- Theorem stating that a hyperbola with eccentricity 2 and foci at (-4, 0) and (4, 0)
    has the equation x^2/4 - y^2/12 = 1 -/
theorem hyperbola_equation_from_parameters :
  ∀ h : Hyperbola,
    h.eccentricity = 2 ∧
    h.focus1 = (-4, 0) ∧
    h.focus2 = (4, 0) →
    hyperbola_equation h = fun x y => x^2 / 4 - y^2 / 12 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_from_parameters_l437_43744


namespace NUMINAMATH_CALUDE_kellys_apples_l437_43760

/-- The total number of apples Kelly has after picking more apples -/
def total_apples (initial : Float) (picked : Float) : Float :=
  initial + picked

/-- Theorem stating that Kelly's total apples is 161.0 -/
theorem kellys_apples :
  let initial := 56.0
  let picked := 105.0
  total_apples initial picked = 161.0 := by
  sorry

end NUMINAMATH_CALUDE_kellys_apples_l437_43760


namespace NUMINAMATH_CALUDE_fashion_show_total_time_l437_43752

/-- Represents the different types of clothing in the fashion show -/
inductive ClothingType
  | EveningWear
  | BathingSuit
  | FormalWear
  | CasualWear

/-- Returns the time in minutes for a runway walk based on the clothing type -/
def walkTime (c : ClothingType) : ℝ :=
  match c with
  | ClothingType.EveningWear => 4
  | ClothingType.BathingSuit => 2
  | ClothingType.FormalWear => 3
  | ClothingType.CasualWear => 2.5

/-- The number of models in the show -/
def numModels : ℕ := 10

/-- Returns the number of sets for each clothing type -/
def numSets (c : ClothingType) : ℕ :=
  match c with
  | ClothingType.EveningWear => 4
  | ClothingType.BathingSuit => 2
  | ClothingType.FormalWear => 3
  | ClothingType.CasualWear => 5

/-- Calculates the total time for all runway walks of a specific clothing type -/
def totalTimeForClothingType (c : ClothingType) : ℝ :=
  (walkTime c) * (numSets c : ℝ) * (numModels : ℝ)

/-- Theorem: The total time for all runway trips during the fashion show is 415 minutes -/
theorem fashion_show_total_time :
  (totalTimeForClothingType ClothingType.EveningWear) +
  (totalTimeForClothingType ClothingType.BathingSuit) +
  (totalTimeForClothingType ClothingType.FormalWear) +
  (totalTimeForClothingType ClothingType.CasualWear) = 415 := by
  sorry


end NUMINAMATH_CALUDE_fashion_show_total_time_l437_43752


namespace NUMINAMATH_CALUDE_ones_digit_of_17_power_l437_43728

theorem ones_digit_of_17_power : ∃ n : ℕ, 17^(17*(13^13)) ≡ 7 [ZMOD 10] := by sorry

end NUMINAMATH_CALUDE_ones_digit_of_17_power_l437_43728


namespace NUMINAMATH_CALUDE_number_of_fours_is_even_l437_43727

theorem number_of_fours_is_even (x y z : ℕ) : 
  x + y + z = 80 →
  3 * x + 4 * y + 5 * z = 276 →
  Even y :=
by sorry

end NUMINAMATH_CALUDE_number_of_fours_is_even_l437_43727


namespace NUMINAMATH_CALUDE_monkey_pole_height_l437_43794

/-- Calculates the height of a pole given the ascent pattern and time taken by a monkey to reach the top -/
def poleHeight (ascent : ℕ) (descent : ℕ) (totalTime : ℕ) : ℕ :=
  let fullCycles := (totalTime - 1) / 2
  let remainingDistance := ascent
  fullCycles * (ascent - descent) + remainingDistance

/-- The height of the pole given the monkey's climbing pattern and time -/
theorem monkey_pole_height : poleHeight 2 1 17 = 10 := by
  sorry

end NUMINAMATH_CALUDE_monkey_pole_height_l437_43794
