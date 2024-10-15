import Mathlib

namespace NUMINAMATH_CALUDE_divisors_of_18m_squared_l563_56358

/-- A function that returns the number of positive divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number is even -/
def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem divisors_of_18m_squared (m : ℕ) 
  (h1 : is_even m) 
  (h2 : num_divisors m = 9) : 
  num_divisors (18 * m^2) = 54 := by sorry

end NUMINAMATH_CALUDE_divisors_of_18m_squared_l563_56358


namespace NUMINAMATH_CALUDE_remaining_boys_average_weight_l563_56341

/-- Proves that given a class of 30 boys where 22 boys have an average weight of 50.25 kg
    and the average weight of all boys is 48.89 kg, the average weight of the remaining boys is 45.15 kg. -/
theorem remaining_boys_average_weight :
  let total_boys : ℕ := 30
  let known_boys : ℕ := 22
  let known_boys_avg_weight : ℝ := 50.25
  let all_boys_avg_weight : ℝ := 48.89
  let remaining_boys : ℕ := total_boys - known_boys
  let remaining_boys_avg_weight : ℝ := (total_boys * all_boys_avg_weight - known_boys * known_boys_avg_weight) / remaining_boys
  remaining_boys_avg_weight = 45.15 := by
sorry

end NUMINAMATH_CALUDE_remaining_boys_average_weight_l563_56341


namespace NUMINAMATH_CALUDE_segment_length_after_reflection_l563_56329

-- Define the points
def Z : ℝ × ℝ := (-5, 3)
def Z' : ℝ × ℝ := (5, 3)

-- Define the reflection over y-axis
def reflect_over_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

-- Theorem statement
theorem segment_length_after_reflection :
  Z' = reflect_over_y_axis Z ∧ 
  Real.sqrt ((Z'.1 - Z.1)^2 + (Z'.2 - Z.2)^2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_segment_length_after_reflection_l563_56329


namespace NUMINAMATH_CALUDE_iron_percentage_in_alloy_l563_56315

/-- The percentage of alloy in the ore -/
def alloy_percentage : ℝ := 0.25

/-- The total amount of ore in kg -/
def total_ore : ℝ := 266.6666666666667

/-- The amount of pure iron obtained in kg -/
def pure_iron : ℝ := 60

/-- The percentage of iron in the alloy -/
def iron_percentage : ℝ := 0.9

theorem iron_percentage_in_alloy :
  alloy_percentage * total_ore * iron_percentage = pure_iron :=
sorry

end NUMINAMATH_CALUDE_iron_percentage_in_alloy_l563_56315


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l563_56393

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given condition: a_2 + a_5 + a_8 = 6 -/
def GivenCondition (a : ℕ → ℝ) : Prop :=
  a 2 + a 5 + a 8 = 6

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h1 : ArithmeticSequence a) (h2 : GivenCondition a) : a 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l563_56393


namespace NUMINAMATH_CALUDE_jose_wandering_time_l563_56386

/-- Given a distance of 4 kilometers and a speed of 2 kilometers per hour,
    the time taken is 2 hours. -/
theorem jose_wandering_time :
  let distance : ℝ := 4  -- Distance in kilometers
  let speed : ℝ := 2     -- Speed in kilometers per hour
  let time := distance / speed
  time = 2 := by sorry

end NUMINAMATH_CALUDE_jose_wandering_time_l563_56386


namespace NUMINAMATH_CALUDE_probability_no_three_consecutive_as_l563_56309

/-- A string of length 6 using symbols A, B, and C -/
def String6ABC := Fin 6 → Fin 3

/-- Check if a string contains three consecutive A's -/
def hasThreeConsecutiveAs (s : String6ABC) : Prop :=
  ∃ i : Fin 4, s i = 0 ∧ s (i + 1) = 0 ∧ s (i + 2) = 0

/-- The total number of possible strings -/
def totalStrings : ℕ := 3^6

/-- The number of strings without three consecutive A's -/
def stringsWithoutThreeAs : ℕ := 680

/-- The probability of a random string not having three consecutive A's -/
def probabilityNoThreeAs : ℚ := stringsWithoutThreeAs / totalStrings

theorem probability_no_three_consecutive_as :
  probabilityNoThreeAs = 680 / 729 :=
sorry

end NUMINAMATH_CALUDE_probability_no_three_consecutive_as_l563_56309


namespace NUMINAMATH_CALUDE_greatest_display_groups_l563_56334

theorem greatest_display_groups (plates spoons glasses bowls : ℕ) 
  (h_plates : plates = 3219)
  (h_spoons : spoons = 5641)
  (h_glasses : glasses = 1509)
  (h_bowls : bowls = 2387) :
  Nat.gcd (Nat.gcd (Nat.gcd plates spoons) glasses) bowls = 1 := by
  sorry

end NUMINAMATH_CALUDE_greatest_display_groups_l563_56334


namespace NUMINAMATH_CALUDE_field_division_proof_l563_56337

theorem field_division_proof (total_area smaller_area larger_area certain_value : ℝ) : 
  total_area = 900 ∧ 
  smaller_area = 405 ∧ 
  larger_area = total_area - smaller_area ∧ 
  larger_area - smaller_area = (1 / 5) * certain_value →
  certain_value = 450 := by
sorry

end NUMINAMATH_CALUDE_field_division_proof_l563_56337


namespace NUMINAMATH_CALUDE_x_less_than_one_necessary_not_sufficient_for_x_squared_less_than_one_l563_56319

theorem x_less_than_one_necessary_not_sufficient_for_x_squared_less_than_one :
  ∀ x : ℝ, (x^2 < 1 → x < 1) ∧ ¬(x < 1 → x^2 < 1) := by sorry

end NUMINAMATH_CALUDE_x_less_than_one_necessary_not_sufficient_for_x_squared_less_than_one_l563_56319


namespace NUMINAMATH_CALUDE_BAB_better_than_ABA_l563_56367

/-- Represents a wrestler's opponent -/
inductive Opponent
| A  -- Andrei
| B  -- Boris

/-- Represents a schedule of three matches -/
def Schedule := List Opponent

/-- The probability of Vladimir losing to a given opponent -/
def losing_probability (o : Opponent) : ℝ :=
  match o with
  | Opponent.A => 0.4
  | Opponent.B => 0.3

/-- The probability of Vladimir winning against a given opponent -/
def winning_probability (o : Opponent) : ℝ :=
  1 - losing_probability o

/-- Calculates the probability of Vladimir qualifying given a schedule -/
def qualifying_probability (s : Schedule) : ℝ :=
  match s with
  | [o1, o2, o3] =>
    winning_probability o1 * losing_probability o2 * winning_probability o3 +
    winning_probability o1 * winning_probability o2 +
    losing_probability o1 * winning_probability o2 * winning_probability o3
  | _ => 0  -- Invalid schedule

def ABA : Schedule := [Opponent.A, Opponent.B, Opponent.A]
def BAB : Schedule := [Opponent.B, Opponent.A, Opponent.B]

theorem BAB_better_than_ABA :
  qualifying_probability BAB > qualifying_probability ABA :=
sorry

end NUMINAMATH_CALUDE_BAB_better_than_ABA_l563_56367


namespace NUMINAMATH_CALUDE_quadratic_equation_prime_solutions_l563_56394

theorem quadratic_equation_prime_solutions :
  ∀ (p q x₁ x₂ : ℤ),
    Prime p →
    Prime q →
    x₁^2 + p*x₁ + 3*q = 0 →
    x₂^2 + p*x₂ + 3*q = 0 →
    x₁ + x₂ = -p →
    x₁ * x₂ = 3*q →
    ((p = 7 ∧ q = 2 ∧ x₁ = -1 ∧ x₂ = -6) ∨
     (p = 5 ∧ q = 2 ∧ x₁ = -3 ∧ x₂ = -2)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_prime_solutions_l563_56394


namespace NUMINAMATH_CALUDE_rational_numbers_equivalence_l563_56374

-- Define the set of integers
def Integers : Set ℚ := {q : ℚ | ∃ (n : ℤ), q = n}

-- Define the set of fractions
def Fractions : Set ℚ := {q : ℚ | ∃ (a b : ℤ), b ≠ 0 ∧ q = a / b}

-- Theorem statement
theorem rational_numbers_equivalence :
  Set.univ = Integers ∪ Fractions :=
sorry

end NUMINAMATH_CALUDE_rational_numbers_equivalence_l563_56374


namespace NUMINAMATH_CALUDE_dwayne_class_a_count_l563_56330

/-- Proves that given the conditions from Mrs. Carter's and Mr. Dwayne's classes,
    the number of students who received an 'A' in Mr. Dwayne's class is 12. -/
theorem dwayne_class_a_count :
  let carter_total : ℕ := 20
  let carter_a_count : ℕ := 8
  let dwayne_total : ℕ := 30
  let ratio : ℚ := carter_a_count / carter_total
  ∃ (dwayne_a_count : ℕ), 
    (dwayne_a_count : ℚ) / dwayne_total = ratio ∧ 
    dwayne_a_count = 12 :=
by sorry

end NUMINAMATH_CALUDE_dwayne_class_a_count_l563_56330


namespace NUMINAMATH_CALUDE_function_property_l563_56332

/-- Given a function f(x) = (x^2 + ax + b)(e^x - e), where a and b are real numbers,
    and f(x) ≥ 0 for all x > 0, then a ≥ -1. -/
theorem function_property (a b : ℝ) :
  (∀ x > 0, (x^2 + a*x + b) * (Real.exp x - Real.exp 1) ≥ 0) →
  a ≥ -1 :=
by sorry

end NUMINAMATH_CALUDE_function_property_l563_56332


namespace NUMINAMATH_CALUDE_wire_cutting_problem_l563_56324

theorem wire_cutting_problem (total_length : ℝ) (ratio : ℝ) :
  total_length = 35 →
  ratio = 2 / 5 →
  ∃ shorter_length longer_length : ℝ,
    shorter_length + longer_length = total_length ∧
    shorter_length = ratio * longer_length ∧
    shorter_length = 10 := by
  sorry

end NUMINAMATH_CALUDE_wire_cutting_problem_l563_56324


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l563_56340

def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmeticSequence a →
  a 1 = 2 →
  a 2 + a 5 = 13 →
  a 5 + a 6 + a 7 = 33 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l563_56340


namespace NUMINAMATH_CALUDE_prism_dimensions_l563_56360

/-- Represents the dimensions of a rectangular prism -/
structure PrismDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Checks if the given dimensions satisfy the conditions of the problem -/
def satisfiesConditions (d : PrismDimensions) : Prop :=
  -- One edge is five times longer than another
  (d.length = 5 * d.width ∨ d.width = 5 * d.length ∨ d.length = 5 * d.height ∨
   d.height = 5 * d.length ∨ d.width = 5 * d.height ∨ d.height = 5 * d.width) ∧
  -- Increasing height by 2 increases volume by 90
  d.length * d.width * 2 = 90 ∧
  -- Changing height to half of (height + 2) makes volume three-fifths of original
  (d.height + 2) / 2 = 3 / 5 * d.height

/-- The theorem stating the only possible dimensions for the rectangular prism -/
theorem prism_dimensions :
  ∀ d : PrismDimensions,
    satisfiesConditions d →
    (d = ⟨0.9, 50, 10⟩ ∨ d = ⟨50, 0.9, 10⟩ ∨
     d = ⟨2, 22.5, 10⟩ ∨ d = ⟨22.5, 2, 10⟩ ∨
     d = ⟨3, 15, 10⟩ ∨ d = ⟨15, 3, 10⟩) :=
by
  sorry

end NUMINAMATH_CALUDE_prism_dimensions_l563_56360


namespace NUMINAMATH_CALUDE_sara_payment_l563_56396

/-- The amount Sara gave to the seller -/
def amount_given (book1_price book2_price change : ℝ) : ℝ :=
  book1_price + book2_price + change

/-- Theorem stating the amount Sara gave to the seller -/
theorem sara_payment (book1_price book2_price change : ℝ) 
  (h1 : book1_price = 5.5)
  (h2 : book2_price = 6.5)
  (h3 : change = 8) :
  amount_given book1_price book2_price change = 20 := by
sorry

end NUMINAMATH_CALUDE_sara_payment_l563_56396


namespace NUMINAMATH_CALUDE_probability_at_least_one_woman_l563_56311

/-- The probability of selecting at least one woman when choosing three people
    at random from a group of eight men and four women -/
theorem probability_at_least_one_woman (men : ℕ) (women : ℕ) : 
  men = 8 → women = 4 → 
  (1 - (Nat.choose men 3 : ℚ) / (Nat.choose (men + women) 3 : ℚ)) = 41 / 55 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_woman_l563_56311


namespace NUMINAMATH_CALUDE_function_minimum_condition_l563_56391

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x - 1

theorem function_minimum_condition (a : ℝ) :
  (∃ x₀ : ℝ, ∀ x : ℝ, f a x₀ ≤ f a x) ∧ 
  (∀ x : ℝ, f a x ≥ 2 * a^2 - a - 1) ↔ 
  0 < a ∧ a ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_function_minimum_condition_l563_56391


namespace NUMINAMATH_CALUDE_joan_football_games_l563_56339

/-- The number of football games Joan attended this year -/
def games_this_year : ℕ := 4

/-- The number of football games Joan attended last year -/
def games_last_year : ℕ := 9

/-- The total number of football games Joan attended -/
def total_games : ℕ := games_this_year + games_last_year

theorem joan_football_games : total_games = 13 := by
  sorry

end NUMINAMATH_CALUDE_joan_football_games_l563_56339


namespace NUMINAMATH_CALUDE_smallest_common_multiple_of_8_and_6_l563_56390

theorem smallest_common_multiple_of_8_and_6 : ∃ n : ℕ+, (∀ m : ℕ+, 8 ∣ m ∧ 6 ∣ m → n ≤ m) ∧ 8 ∣ n ∧ 6 ∣ n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_of_8_and_6_l563_56390


namespace NUMINAMATH_CALUDE_consecutive_integer_sum_l563_56348

theorem consecutive_integer_sum (n : ℕ) :
  (∃ k : ℤ, (k - 2) + (k - 1) + k + (k + 1) + (k + 2) = n) ∧
  (¬ ∃ m : ℤ, (m - 1) + m + (m + 1) + (m + 2) = n) :=
by
  sorry

#check consecutive_integer_sum 225

end NUMINAMATH_CALUDE_consecutive_integer_sum_l563_56348


namespace NUMINAMATH_CALUDE_tangent_line_equation_f_positive_when_a_is_one_minimum_value_when_a_is_e_squared_l563_56378

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := 1 - (a * x^2) / Real.exp x

-- State the theorems to be proved
theorem tangent_line_equation (a : ℝ) :
  (∃ k, ∀ x, k * x + (f a 1 - k) = f a x + (deriv (f a)) 1 * (x - 1)) →
  ∃ k, k = 1 ∧ ∀ x, x + 1 = f a x + (deriv (f a)) 1 * (x - 1) :=
sorry

theorem f_positive_when_a_is_one :
  ∀ x > 0, f 1 x > 0 :=
sorry

theorem minimum_value_when_a_is_e_squared :
  (∃ x, f (Real.exp 2) x = -3) ∧ (∀ x, f (Real.exp 2) x ≥ -3) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_f_positive_when_a_is_one_minimum_value_when_a_is_e_squared_l563_56378


namespace NUMINAMATH_CALUDE_stating_call_ratio_theorem_l563_56349

/-- Represents the ratio of calls processed by team members -/
structure CallRatio where
  team_a : ℚ
  team_b : ℚ

/-- Represents the distribution of calls and agents between two teams -/
structure CallCenter where
  agent_ratio : ℚ  -- Ratio of team A agents to team B agents
  team_b_calls : ℚ -- Fraction of total calls processed by team B

/-- 
Given a call center with specified agent ratio and call distribution,
calculates the ratio of calls processed by each member of team A to team B
-/
def calculate_call_ratio (cc : CallCenter) : CallRatio :=
  { team_a := 7,
    team_b := 5 }

/-- 
Theorem stating that for a call center where team A has 5/8 as many agents as team B,
and team B processes 8/15 of the calls, the ratio of calls processed per agent
of team A to team B is 7:5
-/
theorem call_ratio_theorem (cc : CallCenter) 
  (h1 : cc.agent_ratio = 5 / 8)
  (h2 : cc.team_b_calls = 8 / 15) :
  calculate_call_ratio cc = { team_a := 7, team_b := 5 } := by
  sorry

end NUMINAMATH_CALUDE_stating_call_ratio_theorem_l563_56349


namespace NUMINAMATH_CALUDE_arithmetic_sequence_condition_l563_56316

/-- For an arithmetic sequence with first term a₁ and common difference d,
    the condition 2a₁ + 11d > 0 is sufficient but not necessary for 2a₁ + 11d ≥ 0 -/
theorem arithmetic_sequence_condition (a₁ d : ℝ) :
  (∃ x y : ℝ, (x > y) ∧ (x ≥ 0) ∧ (y < 0)) ∧
  (2 * a₁ + 11 * d > 0 → 2 * a₁ + 11 * d ≥ 0) ∧
  ¬(2 * a₁ + 11 * d ≥ 0 → 2 * a₁ + 11 * d > 0) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_condition_l563_56316


namespace NUMINAMATH_CALUDE_parrot_count_theorem_l563_56331

/-- Represents the types of parrots -/
inductive ParrotType
  | Green
  | Yellow
  | Mottled

/-- Represents the behavior of parrots -/
def ParrotBehavior : ParrotType → Bool → Bool
  | ParrotType.Green, _ => true
  | ParrotType.Yellow, _ => false
  | ParrotType.Mottled, b => b

theorem parrot_count_theorem 
  (total_parrots : Nat)
  (green_count : Nat)
  (yellow_count : Nat)
  (mottled_count : Nat)
  (h_total : total_parrots = 100)
  (h_sum : green_count + yellow_count + mottled_count = total_parrots)
  (h_first_statement : green_count + (mottled_count / 2) = 50)
  (h_second_statement : yellow_count + (mottled_count / 2) = 50)
  : yellow_count = green_count :=
by sorry

end NUMINAMATH_CALUDE_parrot_count_theorem_l563_56331


namespace NUMINAMATH_CALUDE_addition_subtraction_equality_l563_56389

theorem addition_subtraction_equality : 147 + 31 - 19 + 21 = 180 := by sorry

end NUMINAMATH_CALUDE_addition_subtraction_equality_l563_56389


namespace NUMINAMATH_CALUDE_tshirt_cost_l563_56373

/-- The Razorback t-shirt Shop problem -/
theorem tshirt_cost (total_sales : ℝ) (num_shirts : ℕ) (cost_per_shirt : ℝ)
  (h1 : total_sales = 720)
  (h2 : num_shirts = 45)
  (h3 : cost_per_shirt = total_sales / num_shirts) :
  cost_per_shirt = 16 := by
  sorry

end NUMINAMATH_CALUDE_tshirt_cost_l563_56373


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l563_56385

theorem cube_root_equation_solution :
  ∃ x : ℝ, x = 1/11 ∧ (5 + 2/x)^(1/3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l563_56385


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l563_56310

/-- A geometric sequence with positive terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 = 3 →
  a 1 + a 2 + a 3 = 21 →
  a 4 + a 5 + a 6 = 168 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l563_56310


namespace NUMINAMATH_CALUDE_consecutive_integers_divisible_by_three_l563_56392

theorem consecutive_integers_divisible_by_three (a b c d e : ℕ) : 
  (70 < a) ∧ (a < 100) ∧
  (b = a + 1) ∧ (c = b + 1) ∧ (d = c + 1) ∧ (e = d + 1) ∧
  (a % 3 = 0) ∧ (b % 3 = 0) ∧ (c % 3 = 0) ∧ (d % 3 = 0) ∧ (e % 3 = 0) →
  e = 84 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_divisible_by_three_l563_56392


namespace NUMINAMATH_CALUDE_election_votes_count_l563_56328

theorem election_votes_count :
  ∀ (total_votes : ℕ) (losing_candidate_votes winning_candidate_votes : ℕ),
    losing_candidate_votes = (35 * total_votes) / 100 →
    winning_candidate_votes = losing_candidate_votes + 2370 →
    losing_candidate_votes + winning_candidate_votes = total_votes →
    total_votes = 7900 :=
by
  sorry

end NUMINAMATH_CALUDE_election_votes_count_l563_56328


namespace NUMINAMATH_CALUDE_rosy_fish_count_l563_56344

/-- The number of fish Lilly has -/
def lillys_fish : ℕ := 10

/-- The total number of fish Lilly and Rosy have together -/
def total_fish : ℕ := 19

/-- The number of fish Rosy has -/
def rosys_fish : ℕ := total_fish - lillys_fish

theorem rosy_fish_count : rosys_fish = 9 := by
  sorry

end NUMINAMATH_CALUDE_rosy_fish_count_l563_56344


namespace NUMINAMATH_CALUDE_sum_of_squares_of_divisors_1800_l563_56317

def sumOfSquaresOfDivisors (n : ℕ) : ℕ := sorry

theorem sum_of_squares_of_divisors_1800 :
  sumOfSquaresOfDivisors 1800 = 5035485 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_divisors_1800_l563_56317


namespace NUMINAMATH_CALUDE_negation_of_existence_square_gt_power_negation_l563_56368

theorem negation_of_existence (p : ℕ → Prop) :
  (¬∃ n : ℕ, n > 1 ∧ p n) ↔ (∀ n : ℕ, n > 1 → ¬(p n)) := by sorry

theorem square_gt_power_negation :
  (¬∃ n : ℕ, n > 1 ∧ n^2 > 2^n) ↔ (∀ n : ℕ, n > 1 → n^2 ≤ 2^n) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_square_gt_power_negation_l563_56368


namespace NUMINAMATH_CALUDE_spider_trade_l563_56308

/-- The number of spiders Pugsley and Wednesday trade --/
theorem spider_trade (P W x : ℕ) : 
  P = 4 →  -- Pugsley's initial number of spiders
  W + x = 9 * (P - x) →  -- First scenario equation
  P + 6 = W - 6 →  -- Second scenario equation
  x = 2  -- Number of spiders Pugsley gives to Wednesday
:= by sorry

end NUMINAMATH_CALUDE_spider_trade_l563_56308


namespace NUMINAMATH_CALUDE_cookie_batch_size_l563_56320

theorem cookie_batch_size 
  (num_batches : ℕ) 
  (num_people : ℕ) 
  (cookies_per_person : ℕ) 
  (h1 : num_batches = 4)
  (h2 : num_people = 16)
  (h3 : cookies_per_person = 6) :
  (num_people * cookies_per_person) / num_batches / 12 = 2 := by
sorry

end NUMINAMATH_CALUDE_cookie_batch_size_l563_56320


namespace NUMINAMATH_CALUDE_six_students_arrangement_l563_56326

/-- The number of ways to arrange n elements -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The number of ways to arrange n elements, excluding arrangements where two specific elements are adjacent -/
def arrangementWithoutAdjacent (n : ℕ) : ℕ :=
  factorial n - (factorial (n - 1) * 2)

theorem six_students_arrangement :
  arrangementWithoutAdjacent 6 = 480 := by
  sorry

end NUMINAMATH_CALUDE_six_students_arrangement_l563_56326


namespace NUMINAMATH_CALUDE_circle_problem_l563_56357

-- Define the equation of the general circle
def general_circle (k : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*k*x - (2*k+6)*y - 2*k - 31 = 0

-- Define the specific circle E
def circle_E (x y : ℝ) : Prop :=
  (x+2)^2 + (y-1)^2 = 32

-- Theorem statement
theorem circle_problem :
  (∀ k : ℝ, general_circle k (-6) 5 ∧ general_circle k 2 (-3)) ∧
  (circle_E (-6) 5 ∧ circle_E 2 (-3)) ∧
  (∀ P : ℝ × ℝ, ¬circle_E P.1 P.2 →
    ∃ A B : ℝ × ℝ,
      circle_E A.1 A.2 ∧
      circle_E B.1 B.2 ∧
      (∀ X : ℝ × ℝ, circle_E X.1 X.2 →
        (P.1 - A.1) * (X.1 - A.1) + (P.2 - A.2) * (X.2 - A.2) = 0 ∧
        (P.1 - B.1) * (X.1 - B.1) + (P.2 - B.2) * (X.2 - B.2) = 0) ∧
      ((P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2) ≥ 64 * Real.sqrt 2 - 96)) :=
sorry

end NUMINAMATH_CALUDE_circle_problem_l563_56357


namespace NUMINAMATH_CALUDE_yellow_ball_count_l563_56366

/-- Given a bag with red and yellow balls, if the probability of drawing a red ball is 0.2,
    then the number of yellow balls is 20. -/
theorem yellow_ball_count (red_balls : ℕ) (yellow_balls : ℕ) :
  red_balls = 5 →
  (red_balls : ℚ) / ((red_balls : ℚ) + (yellow_balls : ℚ)) = 1/5 →
  yellow_balls = 20 := by
  sorry


end NUMINAMATH_CALUDE_yellow_ball_count_l563_56366


namespace NUMINAMATH_CALUDE_quadratic_m_value_l563_56359

/-- A quadratic function with specific properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  min_value : ℝ
  min_x : ℝ
  point_zero : ℝ
  point_five : ℝ

/-- The properties of the given quadratic function -/
def given_quadratic : QuadraticFunction where
  a := 0
  b := 0
  c := 0
  min_value := -10
  min_x := -1
  point_zero := 8
  point_five := 0  -- This is the m we want to prove

/-- The theorem stating the value of m -/
theorem quadratic_m_value (f : QuadraticFunction) (h1 : f.min_value = -10) 
    (h2 : f.min_x = -1) (h3 : f.point_zero = 8) : f.point_five = 638 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_m_value_l563_56359


namespace NUMINAMATH_CALUDE_least_x_squared_divisible_by_240_l563_56370

theorem least_x_squared_divisible_by_240 :
  ∀ x : ℕ, x > 0 → x^2 % 240 = 0 → x ≥ 60 :=
by
  sorry

end NUMINAMATH_CALUDE_least_x_squared_divisible_by_240_l563_56370


namespace NUMINAMATH_CALUDE_system_solution_l563_56313

/-- Given a system of two linear equations in two variables,
    prove that the solution satisfies both equations. -/
theorem system_solution (x y : ℚ) : 
  x = 14 ∧ y = 29/5 →
  -x + 5*y = 15 ∧ 4*x - 10*y = -2 := by sorry

end NUMINAMATH_CALUDE_system_solution_l563_56313


namespace NUMINAMATH_CALUDE_age_difference_l563_56372

/-- Given that the total age of A and B is 16 years more than the total age of B and C,
    prove that C is 16 years younger than A. -/
theorem age_difference (A B C : ℕ) (h : A + B = B + C + 16) : A = C + 16 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l563_56372


namespace NUMINAMATH_CALUDE_inequality_solution_set_l563_56325

theorem inequality_solution_set (a b : ℝ) (h : |a - b| > 2) : ∀ x : ℝ, |x - a| + |x - b| > 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l563_56325


namespace NUMINAMATH_CALUDE_parabola_x_intercepts_l563_56353

/-- The number of x-intercepts for the parabola x = -3y^2 + 2y + 3 -/
theorem parabola_x_intercepts : ∃! x : ℝ, ∃ y : ℝ, x = -3 * y^2 + 2 * y + 3 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_parabola_x_intercepts_l563_56353


namespace NUMINAMATH_CALUDE_prime_factor_congruence_l563_56302

theorem prime_factor_congruence (p : ℕ) (h_prime : Prime p) :
  ∃ q : ℕ, Prime q ∧ q ∣ (p^p - 1) ∧ q ≡ 1 [ZMOD p] := by
  sorry

end NUMINAMATH_CALUDE_prime_factor_congruence_l563_56302


namespace NUMINAMATH_CALUDE_electric_car_charging_cost_l563_56364

/-- Calculates the total cost of charging an electric car -/
def total_charging_cost (charges_per_week : ℕ) (num_weeks : ℕ) (cost_per_charge : ℚ) : ℚ :=
  (charges_per_week * num_weeks : ℕ) * cost_per_charge

/-- Proves that the total cost of charging an electric car under given conditions is $121.68 -/
theorem electric_car_charging_cost :
  total_charging_cost 3 52 (78/100) = 12168/100 := by
  sorry

end NUMINAMATH_CALUDE_electric_car_charging_cost_l563_56364


namespace NUMINAMATH_CALUDE_trig_expression_evaluation_l563_56347

open Real

theorem trig_expression_evaluation (x : ℝ) 
  (f : ℝ → ℝ) 
  (hf : f = fun x => sin x + cos x) 
  (hf' : deriv f = fun x => 3 * f x) : 
  (sin x)^2 - 3 / ((cos x)^2 + 1) = -14/9 := by
sorry

end NUMINAMATH_CALUDE_trig_expression_evaluation_l563_56347


namespace NUMINAMATH_CALUDE_fruit_combination_count_l563_56346

/-- The number of combinations when choosing k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of fruit types available -/
def fruit_types : ℕ := 4

/-- The number of fruits to be chosen -/
def fruits_to_choose : ℕ := 3

/-- Theorem: The number of combinations when choosing 3 fruits from 4 types is 4 -/
theorem fruit_combination_count : choose fruit_types fruits_to_choose = 4 := by
  sorry

end NUMINAMATH_CALUDE_fruit_combination_count_l563_56346


namespace NUMINAMATH_CALUDE_bobs_speed_l563_56336

theorem bobs_speed (initial_time : ℝ) (construction_time : ℝ) (construction_speed : ℝ) (total_distance : ℝ) :
  initial_time = 1.5 →
  construction_time = 2 →
  construction_speed = 45 →
  total_distance = 180 →
  ∃ (initial_speed : ℝ),
    initial_speed * initial_time + construction_speed * construction_time = total_distance ∧
    initial_speed = 60 :=
by sorry

end NUMINAMATH_CALUDE_bobs_speed_l563_56336


namespace NUMINAMATH_CALUDE_f_properties_l563_56369

noncomputable def f (x : ℝ) : ℝ := Real.sin x * (2 * Real.sqrt 3 * Real.cos x - Real.sin x) + 1

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T')) ∧
  (∀ (x y : ℝ), -π/4 ≤ x ∧ x < y ∧ y ≤ π/6 → f x < f y) ∧
  (∀ (x y : ℝ), π/6 ≤ x ∧ x < y ∧ y ≤ π/4 → f x > f y) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l563_56369


namespace NUMINAMATH_CALUDE_system_solution_l563_56361

theorem system_solution (m : ℝ) : 
  (∃ x y : ℝ, x + y = 3*m ∧ x - y = 5*m ∧ 2*x + 3*y = 10) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l563_56361


namespace NUMINAMATH_CALUDE_sum_of_seventh_powers_l563_56343

theorem sum_of_seventh_powers (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (hsum : x + y + z = 0) (hprod : x*y + x*z + y*z ≠ 0) :
  (x^7 + y^7 + z^7) / (x*y*z * (x*y + x*z + y*z)) = -7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_seventh_powers_l563_56343


namespace NUMINAMATH_CALUDE_diorama_time_factor_l563_56381

def total_time : ℕ := 67
def building_time : ℕ := 49

theorem diorama_time_factor :
  ∃ (planning_time : ℕ) (factor : ℚ),
    planning_time + building_time = total_time ∧
    building_time = planning_time * factor - 5 ∧
    factor = 3 := by
  sorry

end NUMINAMATH_CALUDE_diorama_time_factor_l563_56381


namespace NUMINAMATH_CALUDE_first_digit_base_7_of_528_l563_56304

/-- The first digit of the base 7 representation of a natural number -/
def first_digit_base_7 (n : ℕ) : ℕ :=
  if n = 0 then 0
  else
    let k := (Nat.log n 7).succ
    (n / 7^(k-1)) % 7

/-- Theorem: The first digit of the base 7 representation of 528 is 1 -/
theorem first_digit_base_7_of_528 :
  first_digit_base_7 528 = 1 := by sorry

end NUMINAMATH_CALUDE_first_digit_base_7_of_528_l563_56304


namespace NUMINAMATH_CALUDE_middle_part_of_proportional_division_l563_56338

theorem middle_part_of_proportional_division (total : ℚ) (a b c : ℚ) 
  (h_total : total = 120)
  (h_prop : ∃ (x : ℚ), a = x ∧ b = (1/2) * x ∧ c = (1/4) * x)
  (h_sum : a + b + c = total) :
  b = 34 + 2/7 := by
  sorry

end NUMINAMATH_CALUDE_middle_part_of_proportional_division_l563_56338


namespace NUMINAMATH_CALUDE_wire_cutting_problem_l563_56301

/-- The length of wire pieces that satisfies the given conditions -/
def wire_piece_length : ℕ := 83

theorem wire_cutting_problem (initial_length second_length : ℕ) 
  (h1 : initial_length = 1000)
  (h2 : second_length = 1070)
  (h3 : 12 * wire_piece_length ≤ initial_length)
  (h4 : 12 * wire_piece_length ≤ second_length)
  (h5 : ∀ x : ℕ, x > wire_piece_length → 12 * x > second_length) :
  wire_piece_length = 83 := by
  sorry

end NUMINAMATH_CALUDE_wire_cutting_problem_l563_56301


namespace NUMINAMATH_CALUDE_sqrt_inequality_solution_set_l563_56300

theorem sqrt_inequality_solution_set (x : ℝ) :
  (x^3 - 8) / x ≥ 0 →
  (Real.sqrt ((x^3 - 8) / x) > x - 2 ↔ x ∈ Set.Ioi 2 ∪ Set.Iio 0) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_inequality_solution_set_l563_56300


namespace NUMINAMATH_CALUDE_winter_clothing_mittens_per_box_l563_56303

theorem winter_clothing_mittens_per_box 
  (num_boxes : ℕ) 
  (scarves_per_box : ℕ) 
  (total_pieces : ℕ) 
  (h1 : num_boxes = 3)
  (h2 : scarves_per_box = 3)
  (h3 : total_pieces = 21) :
  (total_pieces - num_boxes * scarves_per_box) / num_boxes = 4 := by
sorry

end NUMINAMATH_CALUDE_winter_clothing_mittens_per_box_l563_56303


namespace NUMINAMATH_CALUDE_root_value_theorem_l563_56355

theorem root_value_theorem (m : ℝ) : 
  (2 * m^2 - 3 * m - 1 = 0) → (3 * m * (2 * m - 3) - 1 = 2) := by
  sorry

end NUMINAMATH_CALUDE_root_value_theorem_l563_56355


namespace NUMINAMATH_CALUDE_velocity_at_4_seconds_l563_56363

-- Define the motion equation
def motion_equation (t : ℝ) : ℝ := t^2 - t + 2

-- Define the instantaneous velocity function
def instantaneous_velocity (t : ℝ) : ℝ := 2 * t - 1

-- Theorem statement
theorem velocity_at_4_seconds :
  instantaneous_velocity 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_velocity_at_4_seconds_l563_56363


namespace NUMINAMATH_CALUDE_sphere_surface_area_from_rectangular_solid_l563_56323

/-- Given a rectangular solid with adjacent face areas of 2, 3, and 6, 
    and all vertices lying on a sphere, the surface area of this sphere is 14π. -/
theorem sphere_surface_area_from_rectangular_solid (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  a * b = 6 →
  b * c = 2 →
  a * c = 3 →
  (∃ (r : ℝ), r > 0 ∧ a^2 + b^2 + c^2 = (2*r)^2) →
  4 * π * ((a^2 + b^2 + c^2) / 4) = 14 * π :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_from_rectangular_solid_l563_56323


namespace NUMINAMATH_CALUDE_right_triangle_product_divisible_by_30_l563_56356

theorem right_triangle_product_divisible_by_30 (a b c : ℤ) :
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →
  (30 : ℤ) ∣ (a * b * c) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_product_divisible_by_30_l563_56356


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l563_56342

theorem arithmetic_sequence_problem (a : ℕ → ℚ) (S : ℕ → ℚ) : 
  (∀ n, a (n + 1) - a n = 1) →  -- arithmetic sequence with common difference 1
  (∀ n, S n = n * a 1 + n * (n - 1) / 2) →  -- sum formula for arithmetic sequence
  (S 8 = 4 * S 4) →  -- given condition
  a 10 = 19 / 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l563_56342


namespace NUMINAMATH_CALUDE_min_moves_to_equalize_l563_56380

/-- Represents a stack of coins -/
structure CoinStack :=
  (coins : ℕ)

/-- Represents the state of all coin stacks -/
structure CoinStacks :=
  (stacks : Fin 4 → CoinStack)

/-- Represents a move that adds one coin to three different stacks -/
structure Move :=
  (targets : Fin 3 → Fin 4)
  (different : targets 0 ≠ targets 1 ∧ targets 0 ≠ targets 2 ∧ targets 1 ≠ targets 2)

/-- The initial state of the coin stacks -/
def initial_stacks : CoinStacks :=
  CoinStacks.mk (fun i => match i with
    | 0 => CoinStack.mk 9
    | 1 => CoinStack.mk 7
    | 2 => CoinStack.mk 5
    | 3 => CoinStack.mk 10)

/-- Applies a move to a given state of coin stacks -/
def apply_move (stacks : CoinStacks) (move : Move) : CoinStacks :=
  sorry

/-- Checks if all stacks have an equal number of coins -/
def all_equal (stacks : CoinStacks) : Prop :=
  sorry

/-- The main theorem to prove -/
theorem min_moves_to_equalize :
  ∃ (moves : List Move),
    moves.length = 11 ∧
    all_equal (moves.foldl apply_move initial_stacks) ∧
    ∀ (other_moves : List Move),
      all_equal (other_moves.foldl apply_move initial_stacks) →
      other_moves.length ≥ 11 :=
  sorry

end NUMINAMATH_CALUDE_min_moves_to_equalize_l563_56380


namespace NUMINAMATH_CALUDE_joshua_share_l563_56399

/-- Given that Joshua and Justin shared $40 and Joshua's share was thrice as much as Justin's,
    prove that Joshua's share is $30. -/
theorem joshua_share (total : ℕ) (justin : ℕ) (joshua : ℕ) : 
  total = 40 → joshua = 3 * justin → total = joshua + justin → joshua = 30 := by
  sorry

end NUMINAMATH_CALUDE_joshua_share_l563_56399


namespace NUMINAMATH_CALUDE_smallest_n_for_inequality_l563_56322

theorem smallest_n_for_inequality : 
  ∃ n : ℕ, n > 0 ∧ (∀ k : ℕ, k > 0 → (1 : ℚ) / k - (1 : ℚ) / (k + 1) < (1 : ℚ) / 15 → k ≥ n) ∧ 
  ((1 : ℚ) / n - (1 : ℚ) / (n + 1) < (1 : ℚ) / 15) ∧ n = 4 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_inequality_l563_56322


namespace NUMINAMATH_CALUDE_stone150_is_8_l563_56398

/-- Represents the circular arrangement of stones with the given counting pattern. -/
def StoneCircle := Fin 15

/-- The number of counts before the pattern repeats. -/
def patternLength : ℕ := 28

/-- Maps a count to its corresponding stone in the circle. -/
def countToStone (count : ℕ) : StoneCircle :=
  sorry

/-- The stone that is counted as 150. -/
def stone150 : StoneCircle :=
  countToStone 150

/-- The original stone number that corresponds to the 150th count. -/
theorem stone150_is_8 : stone150 = ⟨8, sorry⟩ :=
  sorry

end NUMINAMATH_CALUDE_stone150_is_8_l563_56398


namespace NUMINAMATH_CALUDE_present_giving_property_l563_56387

/-- Represents a child in the class -/
structure Child where
  id : Nat

/-- Represents a triple of children -/
structure Triple where
  a : Child
  b : Child
  c : Child

/-- The main theorem to be proved -/
theorem present_giving_property (n : Nat) (h : Odd n) :
  ∃ (children : Finset Child) (S : Finset Triple),
    (children.card = 3 * n) ∧
    (∀ (x y : Child), x ∈ children → y ∈ children → x ≠ y →
      ∃! (t : Triple), t ∈ S ∧ (t.a = x ∧ t.b = y ∨ t.a = x ∧ t.c = y ∨ t.b = x ∧ t.c = y)) ∧
    (∀ (t : Triple), t ∈ S →
      ∃ (t' : Triple), t' ∈ S ∧ t'.a = t.a ∧ t'.b = t.c ∧ t'.c = t.b) := by
  sorry

end NUMINAMATH_CALUDE_present_giving_property_l563_56387


namespace NUMINAMATH_CALUDE_adjacent_triangles_toothpicks_l563_56306

/-- Calculates the number of toothpicks needed for an equilateral triangle -/
def toothpicks_for_triangle (base : ℕ) : ℕ :=
  3 * (base * (base + 1) / 2) / 2

/-- The number of toothpicks needed for two adjacent equilateral triangles -/
def total_toothpicks (large_base small_base : ℕ) : ℕ :=
  toothpicks_for_triangle large_base + toothpicks_for_triangle small_base - small_base

theorem adjacent_triangles_toothpicks :
  total_toothpicks 100 50 = 9462 :=
sorry

end NUMINAMATH_CALUDE_adjacent_triangles_toothpicks_l563_56306


namespace NUMINAMATH_CALUDE_binomial_expansion_example_l563_56318

theorem binomial_expansion_example : 
  (0.5 : ℝ)^3 + 3 * (0.5 : ℝ)^2 * (-1.5) + 3 * (0.5 : ℝ) * (-1.5)^2 + (-1.5)^3 = -1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_example_l563_56318


namespace NUMINAMATH_CALUDE_trapezoid_area_is_147_l563_56382

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a trapezoid ABCD with intersection point E of diagonals -/
structure Trapezoid :=
  (A B C D E : Point)

/-- The area of a triangle -/
def triangle_area (p1 p2 p3 : Point) : ℝ := sorry

/-- The area of a trapezoid -/
def trapezoid_area (t : Trapezoid) : ℝ := sorry

/-- Theorem: Area of trapezoid ABCD is 147 square units -/
theorem trapezoid_area_is_147 (ABCD : Trapezoid) :
  (ABCD.A.x - ABCD.B.x) * (ABCD.C.y - ABCD.D.y) = (ABCD.C.x - ABCD.D.x) * (ABCD.A.y - ABCD.B.y) →
  triangle_area ABCD.A ABCD.B ABCD.E = 75 →
  triangle_area ABCD.A ABCD.D ABCD.E = 30 →
  trapezoid_area ABCD = 147 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_is_147_l563_56382


namespace NUMINAMATH_CALUDE_carbonated_water_percent_in_specific_mixture_l563_56383

/-- Represents a solution with a given percentage of carbonated water -/
structure Solution :=
  (carbonated_water_percent : ℝ)

/-- Represents a mixture of two solutions -/
structure Mixture :=
  (solution1 : Solution)
  (solution2 : Solution)
  (solution1_volume_percent : ℝ)

/-- Calculates the percentage of carbonated water in a mixture -/
def carbonated_water_percent_in_mixture (m : Mixture) : ℝ :=
  m.solution1.carbonated_water_percent * m.solution1_volume_percent +
  m.solution2.carbonated_water_percent * (1 - m.solution1_volume_percent)

/-- Theorem stating that the percentage of carbonated water in the specific mixture is 67.5% -/
theorem carbonated_water_percent_in_specific_mixture :
  let solution1 : Solution := ⟨0.8⟩
  let solution2 : Solution := ⟨0.55⟩
  let mixture : Mixture := ⟨solution1, solution2, 0.5⟩
  carbonated_water_percent_in_mixture mixture = 0.675 := by
  sorry

end NUMINAMATH_CALUDE_carbonated_water_percent_in_specific_mixture_l563_56383


namespace NUMINAMATH_CALUDE_village_foods_sales_l563_56307

/-- Represents the pricing structure for lettuce -/
structure LettucePricing where
  first : Float
  second : Float
  additional : Float

/-- Represents the pricing structure for tomatoes -/
structure TomatoPricing where
  firstTwo : Float
  nextTwo : Float
  additional : Float

/-- Calculates the total sales per month for Village Foods -/
def totalSalesPerMonth (
  customersPerMonth : Nat
) (
  lettucePerCustomer : Nat
) (
  tomatoesPerCustomer : Nat
) (
  lettucePricing : LettucePricing
) (
  tomatoPricing : TomatoPricing
) (
  discountThreshold : Float
) (
  discountRate : Float
) : Float :=
  sorry

/-- Theorem stating that the total sales per month is $2350 -/
theorem village_foods_sales :
  totalSalesPerMonth
    500  -- customers per month
    2    -- lettuce per customer
    4    -- tomatoes per customer
    { first := 1.50, second := 1.00, additional := 0.75 }  -- lettuce pricing
    { firstTwo := 0.60, nextTwo := 0.50, additional := 0.40 }  -- tomato pricing
    10.00  -- discount threshold
    0.10   -- discount rate
  = 2350.00 :=
by sorry

end NUMINAMATH_CALUDE_village_foods_sales_l563_56307


namespace NUMINAMATH_CALUDE_after_school_program_l563_56305

/-- Represents the number of students in each class combination --/
structure ClassCombinations where
  drawing_only : ℕ
  chess_only : ℕ
  music_only : ℕ
  drawing_chess : ℕ
  drawing_music : ℕ
  chess_music : ℕ
  all_three : ℕ

/-- The after-school program problem --/
theorem after_school_program 
  (total_students : ℕ) 
  (drawing_students : ℕ) 
  (chess_students : ℕ) 
  (music_students : ℕ) 
  (multi_class_students : ℕ) 
  (h1 : total_students = 30)
  (h2 : drawing_students = 15)
  (h3 : chess_students = 17)
  (h4 : music_students = 12)
  (h5 : multi_class_students = 14)
  : ∃ (c : ClassCombinations), 
    c.drawing_only + c.chess_only + c.music_only + 
    c.drawing_chess + c.drawing_music + c.chess_music + c.all_three = total_students ∧
    c.drawing_only + c.drawing_chess + c.drawing_music + c.all_three = drawing_students ∧
    c.chess_only + c.drawing_chess + c.chess_music + c.all_three = chess_students ∧
    c.music_only + c.drawing_music + c.chess_music + c.all_three = music_students ∧
    c.drawing_chess + c.drawing_music + c.chess_music + c.all_three = multi_class_students ∧
    c.all_three = 2 := by
  sorry

end NUMINAMATH_CALUDE_after_school_program_l563_56305


namespace NUMINAMATH_CALUDE_first_term_to_common_diff_ratio_l563_56350

/-- An arithmetic progression with a specific property -/
structure ArithmeticProgression where
  a : ℝ  -- First term
  d : ℝ  -- Common difference
  sum_15_eq_3sum_5 : (15 * a + 105 * d) = 3 * (5 * a + 10 * d)

/-- The ratio of the first term to the common difference is 5:1 -/
theorem first_term_to_common_diff_ratio 
  (ap : ArithmeticProgression) : ap.a / ap.d = 5 := by
  sorry

end NUMINAMATH_CALUDE_first_term_to_common_diff_ratio_l563_56350


namespace NUMINAMATH_CALUDE_cube_of_square_of_third_smallest_prime_l563_56371

def third_smallest_prime : ℕ := sorry

theorem cube_of_square_of_third_smallest_prime :
  (third_smallest_prime ^ 2) ^ 3 = 15625 := by sorry

end NUMINAMATH_CALUDE_cube_of_square_of_third_smallest_prime_l563_56371


namespace NUMINAMATH_CALUDE_intersection_M_N_l563_56388

def M : Set ℝ := {x | 0 ≤ x ∧ x < 2}
def N : Set ℝ := {x | x^2 - 2*x - 3 < 0}

theorem intersection_M_N : M ∩ N = {x | 0 ≤ x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l563_56388


namespace NUMINAMATH_CALUDE_inequality_solution_set_l563_56377

-- Define the function f
def f (b c x : ℝ) : ℝ := x^2 - b*x + c

-- Define the theorem
theorem inequality_solution_set 
  (b c : ℝ) 
  (hb : b > 0) 
  (hc : c > 0) 
  (x₁ x₂ : ℝ) 
  (h_zeros : f b c x₁ = 0 ∧ f b c x₂ = 0) 
  (h_progression : (∃ r : ℝ, x₁ = -1 * r ∧ x₂ = -1 / r) ∨ 
                   (∃ d : ℝ, x₁ = -1 - d ∧ x₂ = -1 + d)) :
  {x : ℝ | (x - b) / (x - c) ≤ 0} = Set.Ioo 1 (5/2) ∪ {5/2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l563_56377


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l563_56352

theorem absolute_value_equation_solution :
  ∀ x : ℝ, (|2*x - 6| = 3*x + 5) ↔ (x = 1/5) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l563_56352


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_twelve_l563_56314

theorem sqrt_sum_equals_twelve : 
  Real.sqrt ((5 - 3 * Real.sqrt 2)^2) + Real.sqrt ((5 + 3 * Real.sqrt 2)^2) + 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_twelve_l563_56314


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l563_56312

/-- The eccentricity of a hyperbola with equation x²/4 - y² = 1 is √5/2 -/
theorem hyperbola_eccentricity : 
  let hyperbola := {(x, y) : ℝ × ℝ | x^2 / 4 - y^2 = 1}
  let a : ℝ := 2  -- semi-major axis
  let b : ℝ := 1  -- semi-minor axis
  let c : ℝ := Real.sqrt (a^2 + b^2)  -- focal distance
  let e : ℝ := c / a  -- eccentricity
  e = Real.sqrt 5 / 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l563_56312


namespace NUMINAMATH_CALUDE_base_8_2453_equals_1323_l563_56335

def base_8_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (8 ^ i)) 0

theorem base_8_2453_equals_1323 :
  base_8_to_10 [3, 5, 4, 2] = 1323 := by
  sorry

end NUMINAMATH_CALUDE_base_8_2453_equals_1323_l563_56335


namespace NUMINAMATH_CALUDE_sum_2012_terms_equals_negative_2012_l563_56362

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

def sum_arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem sum_2012_terms_equals_negative_2012 :
  let a₁ : ℤ := -2012
  let d : ℤ := 2
  let n : ℕ := 2012
  sum_arithmetic_sequence a₁ d n = -2012 := by sorry

end NUMINAMATH_CALUDE_sum_2012_terms_equals_negative_2012_l563_56362


namespace NUMINAMATH_CALUDE_unique_valid_stamp_set_l563_56333

/-- Given stamps of denominations 7, n, and n+1 cents, 
    110 cents is the greatest postage that cannot be formed -/
def is_valid_stamp_set (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 110 → ∃ (a b c : ℕ), m = 7 * a + n * b + (n + 1) * c ∧
  ¬∃ (a b c : ℕ), 110 = 7 * a + n * b + (n + 1) * c

theorem unique_valid_stamp_set :
  ∃! n : ℕ, n > 0 ∧ is_valid_stamp_set n :=
by sorry

end NUMINAMATH_CALUDE_unique_valid_stamp_set_l563_56333


namespace NUMINAMATH_CALUDE_three_planes_max_regions_l563_56384

/-- The maximum number of regions into which n planes can divide 3D space -/
def maxRegions (n : ℕ) : ℕ := sorry

/-- Theorem: Three planes can divide 3D space into at most 8 regions -/
theorem three_planes_max_regions :
  maxRegions 3 = 8 := by sorry

end NUMINAMATH_CALUDE_three_planes_max_regions_l563_56384


namespace NUMINAMATH_CALUDE_jumping_contest_l563_56354

/-- The jumping contest problem -/
theorem jumping_contest (grasshopper_jump mouse_jump : ℕ) 
  (h1 : grasshopper_jump = 25)
  (h2 : mouse_jump = 31) : 
  (grasshopper_jump + 32) - mouse_jump = 26 := by
  sorry


end NUMINAMATH_CALUDE_jumping_contest_l563_56354


namespace NUMINAMATH_CALUDE_min_difference_theorem_l563_56321

noncomputable def f (x : ℝ) : ℝ := Real.exp x

noncomputable def g (x : ℝ) : ℝ := Real.log (x / 2) + 1 / 2

theorem min_difference_theorem (m : ℝ) (hm : m > 0) :
  ∃ (a b : ℝ), f a = m ∧ f b = m ∧
  ∀ (a' b' : ℝ), f a' = m → f b' = m → b - a ≤ b' - a' ∧ b - a = 2 + Real.log 2 :=
sorry

end NUMINAMATH_CALUDE_min_difference_theorem_l563_56321


namespace NUMINAMATH_CALUDE_point_d_coordinates_l563_56376

/-- Given two points P and Q in the plane, and a point D on the line segment PQ such that
    PD = 2DQ, prove that D has specific coordinates. -/
theorem point_d_coordinates (P Q D : ℝ × ℝ) : 
  P = (-3, -2) →
  Q = (5, 10) →
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (1 - t) • P + t • Q) →
  (P.1 - D.1)^2 + (P.2 - D.2)^2 = 4 * ((Q.1 - D.1)^2 + (Q.2 - D.2)^2) →
  D = (3, 7) := by
sorry


end NUMINAMATH_CALUDE_point_d_coordinates_l563_56376


namespace NUMINAMATH_CALUDE_system_integer_solutions_determinant_l563_56395

theorem system_integer_solutions_determinant (a b c d : ℤ) :
  (∀ m n : ℤ, ∃ x y : ℤ, a * x + b * y = m ∧ c * x + d * y = n) →
  (a * d - b * c = 1 ∨ a * d - b * c = -1) :=
by sorry

end NUMINAMATH_CALUDE_system_integer_solutions_determinant_l563_56395


namespace NUMINAMATH_CALUDE_dibromoalkane_formula_l563_56379

/-- The mass fraction of bromine in a dibromoalkane -/
def bromine_mass_fraction : ℝ := 0.851

/-- The atomic mass of carbon in g/mol -/
def carbon_mass : ℝ := 12

/-- The atomic mass of hydrogen in g/mol -/
def hydrogen_mass : ℝ := 1

/-- The atomic mass of bromine in g/mol -/
def bromine_mass : ℝ := 80

/-- The general formula of a dibromoalkane is CₙH₂ₙBr₂ -/
def dibromoalkane_mass (n : ℕ) : ℝ :=
  n * carbon_mass + 2 * n * hydrogen_mass + 2 * bromine_mass

/-- Theorem: If the mass fraction of bromine in a dibromoalkane is 85.1%, then n = 2 -/
theorem dibromoalkane_formula :
  ∃ (n : ℕ), (2 * bromine_mass) / (dibromoalkane_mass n) = bromine_mass_fraction ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_dibromoalkane_formula_l563_56379


namespace NUMINAMATH_CALUDE_combined_weight_l563_56345

theorem combined_weight (person baby nurse : ℝ)
  (h1 : person + baby = 78)
  (h2 : nurse + baby = 69)
  (h3 : person + nurse = 137) :
  person + nurse + baby = 142 :=
by sorry

end NUMINAMATH_CALUDE_combined_weight_l563_56345


namespace NUMINAMATH_CALUDE_blue_pill_cost_correct_l563_56327

/-- The cost of a blue pill in dollars -/
def blue_pill_cost : ℝ := 23.50

/-- The cost of a red pill in dollars -/
def red_pill_cost : ℝ := blue_pill_cost - 2

/-- The number of days of medication -/
def days : ℕ := 21

/-- The total cost of medication for the entire period -/
def total_cost : ℝ := 945

theorem blue_pill_cost_correct :
  blue_pill_cost * days + red_pill_cost * days = total_cost :=
by sorry

end NUMINAMATH_CALUDE_blue_pill_cost_correct_l563_56327


namespace NUMINAMATH_CALUDE_statue_cost_l563_56397

theorem statue_cost (selling_price : ℝ) (profit_percentage : ℝ) (original_cost : ℝ) :
  selling_price = 670 ∧ 
  profit_percentage = 35 ∧ 
  selling_price = original_cost * (1 + profit_percentage / 100) →
  original_cost = 496.30 := by
sorry

end NUMINAMATH_CALUDE_statue_cost_l563_56397


namespace NUMINAMATH_CALUDE_rationalize_sqrt_five_twelfths_l563_56375

theorem rationalize_sqrt_five_twelfths : 
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := by sorry

end NUMINAMATH_CALUDE_rationalize_sqrt_five_twelfths_l563_56375


namespace NUMINAMATH_CALUDE_circle_tangent_to_parabola_directrix_l563_56365

/-- The value of m for which the circle x^2 + y^2 + mx - 1/4 = 0 is tangent to the directrix of the parabola y^2 = 4x -/
theorem circle_tangent_to_parabola_directrix (x y m : ℝ) : 
  (∃ x y, x^2 + y^2 + m*x - 1/4 = 0) → -- Circle equation
  (∃ x y, y^2 = 4*x) → -- Parabola equation
  (∃ x y, x^2 + y^2 + m*x - 1/4 = 0 ∧ x = -1) → -- Circle is tangent to directrix (x = -1)
  m = 3/4 := by
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_parabola_directrix_l563_56365


namespace NUMINAMATH_CALUDE_jim_out_of_pocket_l563_56351

def out_of_pocket (first_ring_cost second_ring_cost first_ring_sale_price : ℕ) : ℕ :=
  first_ring_cost + second_ring_cost - first_ring_sale_price

theorem jim_out_of_pocket :
  let first_ring_cost : ℕ := 10000
  let second_ring_cost : ℕ := 2 * first_ring_cost
  let first_ring_sale_price : ℕ := first_ring_cost / 2
  out_of_pocket first_ring_cost second_ring_cost first_ring_sale_price = 25000 := by
  sorry

end NUMINAMATH_CALUDE_jim_out_of_pocket_l563_56351
