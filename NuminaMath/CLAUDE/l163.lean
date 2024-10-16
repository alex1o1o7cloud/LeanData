import Mathlib

namespace NUMINAMATH_CALUDE_a_plus_reward_is_ten_l163_16369

/-- Represents the grading system and reward structure for Paul's courses. -/
structure GradingSystem where
  num_courses : ℕ
  reward_b_plus : ℚ
  reward_a : ℚ
  max_reward : ℚ

/-- Calculates the maximum reward Paul can receive given a grading system and A+ reward. -/
def max_reward (gs : GradingSystem) (reward_a_plus : ℚ) : ℚ :=
  let doubled_reward_b_plus := 2 * gs.reward_b_plus
  let doubled_reward_a := 2 * gs.reward_a
  max (gs.num_courses * doubled_reward_a)
      (((gs.num_courses - 1) * doubled_reward_a) + reward_a_plus)

/-- Theorem stating that the A+ reward must be $10 to achieve the maximum possible reward. -/
theorem a_plus_reward_is_ten (gs : GradingSystem) 
    (h_num_courses : gs.num_courses = 10)
    (h_reward_b_plus : gs.reward_b_plus = 5)
    (h_reward_a : gs.reward_a = 10)
    (h_max_reward : gs.max_reward = 190) :
    ∃ (reward_a_plus : ℚ), reward_a_plus = 10 ∧ max_reward gs reward_a_plus = gs.max_reward :=
  sorry


end NUMINAMATH_CALUDE_a_plus_reward_is_ten_l163_16369


namespace NUMINAMATH_CALUDE_jean_grandchildren_l163_16361

/-- The number of cards Jean buys for each grandchild per year -/
def cards_per_grandchild : ℕ := 2

/-- The amount of money Jean puts in each card -/
def money_per_card : ℕ := 80

/-- The total amount of money Jean gives away to her grandchildren per year -/
def total_money_given : ℕ := 480

/-- The number of grandchildren Jean has -/
def num_grandchildren : ℕ := total_money_given / (cards_per_grandchild * money_per_card)

theorem jean_grandchildren :
  num_grandchildren = 3 :=
sorry

end NUMINAMATH_CALUDE_jean_grandchildren_l163_16361


namespace NUMINAMATH_CALUDE_valid_word_count_l163_16364

def letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'I'}
def vowels : Finset Char := {'A', 'E', 'I'}
def consonants : Finset Char := {'B', 'C', 'D'}

def is_valid_word (word : List Char) : Prop :=
  word.length = 5 ∧ word.toFinset ⊆ letters ∧ (∃ c ∈ word, c ∈ consonants)

def count_valid_words : ℕ := (letters.powerset.filter (λ s => s.card = 5)).card

theorem valid_word_count : count_valid_words = 7533 := by
  sorry

end NUMINAMATH_CALUDE_valid_word_count_l163_16364


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l163_16338

/-- Given a principal amount and an interest rate, if the simple interest for 2 years is 320
    and the compound interest for 2 years is 340, then the interest rate is 12.5% per annum. -/
theorem interest_rate_calculation (P R : ℝ) 
  (h_simple : (P * R * 2) / 100 = 320)
  (h_compound : P * ((1 + R / 100)^2 - 1) = 340) :
  R = 12.5 := by
sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l163_16338


namespace NUMINAMATH_CALUDE_not_always_equal_distribution_l163_16388

/-- Represents the state of the pies on plates -/
structure PieState where
  numPlates : Nat
  totalPies : Nat
  blackPies : Nat
  whitePies : Nat

/-- Represents a move in the game -/
inductive Move
  | transfer : Nat → Move

/-- Checks if a pie state is valid -/
def isValidState (state : PieState) : Prop :=
  state.numPlates = 20 ∧
  state.totalPies = 40 ∧
  state.blackPies + state.whitePies = state.totalPies

/-- Checks if a pie state has equal distribution -/
def hasEqualDistribution (state : PieState) : Prop :=
  state.blackPies = state.whitePies

/-- Applies a move to a pie state -/
def applyMove (state : PieState) (move : Move) : PieState :=
  match move with
  | Move.transfer n => 
      { state with 
        blackPies := state.blackPies + n,
        whitePies := state.whitePies - n
      }

/-- Theorem: It's not always possible to achieve equal distribution -/
theorem not_always_equal_distribution :
  ∃ (initialState : PieState),
    isValidState initialState ∧
    ∀ (moves : List Move),
      ¬hasEqualDistribution (moves.foldl applyMove initialState) :=
sorry

end NUMINAMATH_CALUDE_not_always_equal_distribution_l163_16388


namespace NUMINAMATH_CALUDE_complex_equation_solution_l163_16396

theorem complex_equation_solution (z : ℂ) : 
  (1 - Complex.I)^2 * z = 3 + 2 * Complex.I → z = -1 + (3/2) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l163_16396


namespace NUMINAMATH_CALUDE_distinct_values_of_3_3_3_3_l163_16353

-- Define a function to represent the expression with different parenthesizations
def exprParenthesization : List (ℕ → ℕ → ℕ → ℕ) :=
  [ (λ a b c => a^(b^(c^c))),
    (λ a b c => a^((b^c)^c)),
    (λ a b c => ((a^b)^c)^c),
    (λ a b c => (a^(b^c))^c),
    (λ a b c => (a^b)^(c^c)) ]

-- Define a function to evaluate the expression for a given base
def evaluateExpr (base : ℕ) : List ℕ :=
  exprParenthesization.map (λ f => f base base base)

-- Theorem statement
theorem distinct_values_of_3_3_3_3 :
  (evaluateExpr 3).toFinset.card = 3 := by sorry


end NUMINAMATH_CALUDE_distinct_values_of_3_3_3_3_l163_16353


namespace NUMINAMATH_CALUDE_new_members_weight_l163_16339

/-- Theorem: Calculate the combined weight of new group members -/
theorem new_members_weight (original_size : ℕ) (weight_increase : ℝ) 
  (original_member1 original_member2 original_member3 : ℝ) :
  original_size = 8 →
  weight_increase = 4.2 →
  original_member1 = 60 →
  original_member2 = 75 →
  original_member3 = 65 →
  (original_member1 + original_member2 + original_member3 + 
    original_size * weight_increase) = 233.6 := by
  sorry

end NUMINAMATH_CALUDE_new_members_weight_l163_16339


namespace NUMINAMATH_CALUDE_santiago_roses_count_l163_16365

/-- The number of red roses Mrs. Garrett has -/
def garrett_roses : ℕ := 24

/-- The number of additional roses Mrs. Santiago has compared to Mrs. Garrett -/
def additional_roses : ℕ := 34

/-- The number of red roses Mrs. Santiago has -/
def santiago_roses : ℕ := garrett_roses + additional_roses

theorem santiago_roses_count : santiago_roses = 58 := by
  sorry

end NUMINAMATH_CALUDE_santiago_roses_count_l163_16365


namespace NUMINAMATH_CALUDE_complex_magnitude_equality_l163_16360

theorem complex_magnitude_equality (n : ℝ) : 
  n > 0 → (Complex.abs (5 + Complex.I * n) = 5 * Real.sqrt 13 ↔ n = 10 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equality_l163_16360


namespace NUMINAMATH_CALUDE_prob_good_or_excellent_grade_l163_16356

/-- Represents the types of students in the group -/
inductive StudentType
| Excellent
| Good
| Poor

/-- Represents the possible grades a student can receive -/
inductive Grade
| Excellent
| Good
| Satisfactory
| Unsatisfactory

/-- The total number of students -/
def totalStudents : ℕ := 21

/-- The number of excellent students -/
def excellentCount : ℕ := 5

/-- The number of good students -/
def goodCount : ℕ := 10

/-- The number of poorly performing students -/
def poorCount : ℕ := 6

/-- The probability of selecting an excellent student -/
def probExcellent : ℚ := excellentCount / totalStudents

/-- The probability of selecting a good student -/
def probGood : ℚ := goodCount / totalStudents

/-- The probability of selecting a poor student -/
def probPoor : ℚ := poorCount / totalStudents

/-- The probability of an excellent student receiving an excellent grade -/
def probExcellentGrade (s : StudentType) : ℚ :=
  match s with
  | StudentType.Excellent => 1
  | _ => 0

/-- The probability of a good student receiving a good or excellent grade -/
def probGoodOrExcellentGrade (s : StudentType) : ℚ :=
  match s with
  | StudentType.Excellent => 1
  | StudentType.Good => 1
  | StudentType.Poor => 1/3

/-- The probability of a randomly selected student receiving a good or excellent grade -/
theorem prob_good_or_excellent_grade :
  probExcellent * probExcellentGrade StudentType.Excellent +
  probGood * probGoodOrExcellentGrade StudentType.Good +
  probPoor * probGoodOrExcellentGrade StudentType.Poor = 17/21 := by
  sorry


end NUMINAMATH_CALUDE_prob_good_or_excellent_grade_l163_16356


namespace NUMINAMATH_CALUDE_water_consumption_rate_l163_16377

/-- 
Given a person drinks water at a rate of 1 cup every 20 minutes,
prove that they will drink 11.25 cups in 225 minutes.
-/
theorem water_consumption_rate (drinking_rate : ℚ) (time : ℚ) (cups : ℚ) : 
  drinking_rate = 1 / 20 → time = 225 → cups = time * drinking_rate → cups = 11.25 := by
  sorry

end NUMINAMATH_CALUDE_water_consumption_rate_l163_16377


namespace NUMINAMATH_CALUDE_solve_equation_l163_16300

theorem solve_equation (x : ℝ) : 3 * x + 15 = (1/3) * (6 * x + 45) → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l163_16300


namespace NUMINAMATH_CALUDE_largest_divisor_of_n_l163_16397

theorem largest_divisor_of_n (n : ℕ) (h1 : 0 < n) (h2 : 72 ∣ n^2) :
  ∃ (v : ℕ), v = 12 ∧ v ∣ n ∧ ∀ (k : ℕ), k ∣ n → k ≤ v :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_l163_16397


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l163_16371

theorem quadratic_inequality_solution (x : ℝ) :
  (-5 * x^2 + 10 * x - 3 > 0) ↔ (x > 1 - Real.sqrt 10 / 5 ∧ x < 1 + Real.sqrt 10 / 5) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l163_16371


namespace NUMINAMATH_CALUDE_sphere_volume_surface_area_relation_l163_16347

theorem sphere_volume_surface_area_relation (r : ℝ) : 
  (4 / 3 * Real.pi * r^3) = 2 * (4 * Real.pi * r^2) → r = 6 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_surface_area_relation_l163_16347


namespace NUMINAMATH_CALUDE_committee_probability_l163_16329

def science_club_size : ℕ := 24
def num_boys : ℕ := 12
def num_girls : ℕ := 12
def committee_size : ℕ := 5

def probability_at_least_two_of_each : ℚ :=
  4704 / 7084

theorem committee_probability :
  let total_committees := Nat.choose science_club_size committee_size
  let valid_committees := total_committees - (
    Nat.choose num_boys 0 * Nat.choose num_girls 5 +
    Nat.choose num_boys 1 * Nat.choose num_girls 4 +
    Nat.choose num_boys 4 * Nat.choose num_girls 1 +
    Nat.choose num_boys 5 * Nat.choose num_girls 0
  )
  (valid_committees : ℚ) / total_committees = probability_at_least_two_of_each :=
by sorry

end NUMINAMATH_CALUDE_committee_probability_l163_16329


namespace NUMINAMATH_CALUDE_vector_properties_l163_16332

/-- Given vectors in R², prove properties about their relationships -/
theorem vector_properties (a b : ℝ) :
  let m : Fin 2 → ℝ := ![a, b^2 - b + 7/3]
  let n : Fin 2 → ℝ := ![a + b + 2, 1]
  let μ : Fin 2 → ℝ := ![2, 1]
  (∃ (k : ℝ), m = k • μ) →
  (∃ (a_min : ℝ), a_min = 25/6 ∧ ∀ (a' : ℝ), (∃ (k : ℝ), ![a', b^2 - b + 7/3] = k • μ) → a' ≥ a_min) ∧
  (m • n ≥ 0) := by
sorry


end NUMINAMATH_CALUDE_vector_properties_l163_16332


namespace NUMINAMATH_CALUDE_base5_44_to_decimal_l163_16359

/-- Converts a base-5 number to its decimal equivalent -/
def base5ToDecimal (d₁ d₀ : ℕ) : ℕ := d₁ * 5^1 + d₀ * 5^0

/-- The base-5 number 44₅ -/
def base5_44 : ℕ × ℕ := (4, 4)

theorem base5_44_to_decimal :
  base5ToDecimal base5_44.1 base5_44.2 = 24 := by sorry

end NUMINAMATH_CALUDE_base5_44_to_decimal_l163_16359


namespace NUMINAMATH_CALUDE_polynomial_value_at_n_plus_one_l163_16346

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℚ :=
  if k > n then 0
  else (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Define the polynomial P
noncomputable def P (n : ℕ) (x : ℚ) : ℚ :=
  sorry  -- The actual definition is not provided in the problem statement

-- State the theorem
theorem polynomial_value_at_n_plus_one (n : ℕ) :
  (∀ k : ℕ, k ≤ n → P n k = 1 / binomial n k) →
  P n (n + 1) = if n % 2 = 0 then 1 else 0 :=
sorry

end NUMINAMATH_CALUDE_polynomial_value_at_n_plus_one_l163_16346


namespace NUMINAMATH_CALUDE_wong_valentines_l163_16390

/-- Mrs. Wong's Valentine problem -/
theorem wong_valentines (initial : ℕ) (given_away : ℕ) (remaining : ℕ) : 
  initial = 30 → given_away = 8 → remaining = initial - given_away → remaining = 22 := by
  sorry

end NUMINAMATH_CALUDE_wong_valentines_l163_16390


namespace NUMINAMATH_CALUDE_smallest_three_star_three_star_divisibility_l163_16394

/-- A three-star number is a positive three-digit integer that is the product of three distinct prime numbers. -/
def is_three_star (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ ∃ p q r : ℕ, 
    Prime p ∧ Prime q ∧ Prime r ∧ 
    p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
    n = p * q * r

/-- The smallest three-star number is 102. -/
theorem smallest_three_star : 
  is_three_star 102 ∧ ∀ n, is_three_star n → 102 ≤ n :=
sorry

/-- Every three-star number is divisible by 2, 3, or 5. -/
theorem three_star_divisibility (n : ℕ) :
  is_three_star n → (2 ∣ n) ∨ (3 ∣ n) ∨ (5 ∣ n) :=
sorry

end NUMINAMATH_CALUDE_smallest_three_star_three_star_divisibility_l163_16394


namespace NUMINAMATH_CALUDE_complex_expression_equality_l163_16343

theorem complex_expression_equality : 
  let z₁ : ℂ := (-2 * Real.sqrt 3 + I) / (1 + 2 * Real.sqrt 3 * I)
  let z₂ : ℂ := (Real.sqrt 2 / (1 - I)) ^ 2017
  z₁ + z₂ = Real.sqrt 2 / 2 + (Real.sqrt 2 / 2 + 1) * I :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l163_16343


namespace NUMINAMATH_CALUDE_largest_rank_3_less_than_quarter_proof_l163_16304

def rank (q : ℚ) : ℕ :=
  sorry

def largest_rank_3_less_than_quarter : ℚ :=
  sorry

theorem largest_rank_3_less_than_quarter_proof :
  rank largest_rank_3_less_than_quarter = 3 ∧
  largest_rank_3_less_than_quarter < 1/4 ∧
  largest_rank_3_less_than_quarter = 1/5 + 1/21 + 1/421 ∧
  ∀ q : ℚ, rank q = 3 → q < 1/4 → q ≤ largest_rank_3_less_than_quarter :=
by sorry

end NUMINAMATH_CALUDE_largest_rank_3_less_than_quarter_proof_l163_16304


namespace NUMINAMATH_CALUDE_prime_sum_product_l163_16311

theorem prime_sum_product (x y z : ℕ) : 
  Nat.Prime x ∧ Nat.Prime y ∧ Nat.Prime z ∧
  x ≤ y ∧ y ≤ z ∧
  x + y + z = 12 ∧
  x * y + y * z + x * z = 41 →
  x + 2 * y + 3 * z = 29 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_product_l163_16311


namespace NUMINAMATH_CALUDE_star_operation_result_l163_16354

-- Define the set of elements
inductive Element : Type
  | one : Element
  | two : Element
  | three : Element
  | four : Element

-- Define the operation
def star : Element → Element → Element
  | Element.one, Element.one => Element.four
  | Element.one, Element.two => Element.one
  | Element.one, Element.three => Element.two
  | Element.one, Element.four => Element.three
  | Element.two, Element.one => Element.one
  | Element.two, Element.two => Element.three
  | Element.two, Element.three => Element.four
  | Element.two, Element.four => Element.two
  | Element.three, Element.one => Element.two
  | Element.three, Element.two => Element.four
  | Element.three, Element.three => Element.one
  | Element.three, Element.four => Element.three
  | Element.four, Element.one => Element.three
  | Element.four, Element.two => Element.two
  | Element.four, Element.three => Element.three
  | Element.four, Element.four => Element.four

theorem star_operation_result :
  star (star Element.three Element.one) (star Element.four Element.two) = Element.three := by
  sorry

end NUMINAMATH_CALUDE_star_operation_result_l163_16354


namespace NUMINAMATH_CALUDE_new_pet_ratio_l163_16314

/-- Represents the number of pets of each type -/
structure PetCount where
  dogs : ℕ
  cats : ℕ
  birds : ℕ

/-- Calculates the new pet count after changes -/
def newPetCount (initial : PetCount) : PetCount :=
  { dogs := initial.dogs - 15,
    cats := initial.cats + 4 - 12,
    birds := initial.birds + 7 - 5 }

/-- Theorem stating the new ratio of pets after changes -/
theorem new_pet_ratio (initial : PetCount) :
  initial.dogs + initial.cats + initial.birds = 315 →
  initial.dogs * 35 = 315 * 10 →
  initial.cats * 35 = 315 * 17 →
  initial.birds * 35 = 315 * 8 →
  let final := newPetCount initial
  (final.dogs, final.cats, final.birds) = (75, 145, 74) :=
by sorry

end NUMINAMATH_CALUDE_new_pet_ratio_l163_16314


namespace NUMINAMATH_CALUDE_find_divisor_l163_16376

theorem find_divisor (dividend quotient remainder divisor : ℕ) 
  (h1 : dividend = 127)
  (h2 : quotient = 5)
  (h3 : remainder = 2)
  (h4 : dividend = divisor * quotient + remainder) :
  divisor = 25 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l163_16376


namespace NUMINAMATH_CALUDE_min_value_squares_l163_16391

theorem min_value_squares (a b t : ℝ) (h : a + b = t) :
  (∀ x y : ℝ, x + y = t → (a^2 + 1)^2 + (b^2 + 1)^2 ≤ (x^2 + 1)^2 + (y^2 + 1)^2) →
  (a^2 + 1)^2 + (b^2 + 1)^2 = (t^4 + 8*t^2 + 16) / 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_squares_l163_16391


namespace NUMINAMATH_CALUDE_set_A_nonempty_iff_l163_16313

def A (a : ℝ) : Set ℝ := {x : ℝ | x^2 + 2*x - a = 0}

theorem set_A_nonempty_iff (a : ℝ) : Set.Nonempty (A a) ↔ a ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_set_A_nonempty_iff_l163_16313


namespace NUMINAMATH_CALUDE_bus_average_speed_l163_16370

/-- Proves that given a bicycle traveling at 15 km/h and a bus starting 195 km behind it,
    if the bus catches up to the bicycle in 3 hours, then the average speed of the bus is 80 km/h. -/
theorem bus_average_speed
  (bicycle_speed : ℝ)
  (initial_distance : ℝ)
  (catch_up_time : ℝ)
  (h1 : bicycle_speed = 15)
  (h2 : initial_distance = 195)
  (h3 : catch_up_time = 3)
  : (initial_distance + bicycle_speed * catch_up_time) / catch_up_time = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_bus_average_speed_l163_16370


namespace NUMINAMATH_CALUDE_combined_data_mode_l163_16302

/-- Given two sets of data with specified averages, proves that the mode of the combined set is 8 -/
theorem combined_data_mode (x y : ℝ) : 
  (3 + x + 2*y + 5) / 4 = 6 →
  (x + 6 + y) / 3 = 6 →
  let combined_set := [3, x, 2*y, 5, x, 6, y]
  ∃ (mode : ℝ), mode = 8 ∧ 
    (∀ z ∈ combined_set, (combined_set.filter (λ t => t = z)).length ≤ 
                         (combined_set.filter (λ t => t = mode)).length) :=
by sorry

end NUMINAMATH_CALUDE_combined_data_mode_l163_16302


namespace NUMINAMATH_CALUDE_extreme_value_condition_l163_16333

/-- The function f(x) with parameters a and b -/
def f (a b x : ℝ) : ℝ := x^3 - a*x^2 + b*x + a^2

/-- The derivative of f(x) with respect to x -/
def f_derivative (a b x : ℝ) : ℝ := 3*x^2 - 2*a*x + b

theorem extreme_value_condition (a b : ℝ) :
  f a b 1 = 10 ∧ f_derivative a b 1 = 0 → a = -4 := by
  sorry

end NUMINAMATH_CALUDE_extreme_value_condition_l163_16333


namespace NUMINAMATH_CALUDE_intersection_line_l163_16340

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 10
def circle2 (x y : ℝ) : Prop := (x-1)^2 + (y-3)^2 = 10

-- Define the line
def line (x y : ℝ) : Prop := x + 3*y - 5 = 0

-- Theorem statement
theorem intersection_line :
  ∀ x y : ℝ, circle1 x y ∧ circle2 x y → line x y :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_line_l163_16340


namespace NUMINAMATH_CALUDE_student_failed_by_89_marks_l163_16351

def total_marks : ℕ := 800
def passing_percentage : ℚ := 33 / 100
def student_marks : ℕ := 175

theorem student_failed_by_89_marks :
  ⌈(passing_percentage * total_marks : ℚ)⌉ - student_marks = 89 :=
sorry

end NUMINAMATH_CALUDE_student_failed_by_89_marks_l163_16351


namespace NUMINAMATH_CALUDE_ratio_transformation_l163_16366

theorem ratio_transformation (a c : ℝ) (h : c ≠ 0) :
  (3 * a) / (c / 3) = 9 * (a / c) := by sorry

end NUMINAMATH_CALUDE_ratio_transformation_l163_16366


namespace NUMINAMATH_CALUDE_sin_kpi_minus_x_is_odd_cos_squared_when_tan_pi_minus_x_is_two_cos_2x_plus_pi_third_symmetry_l163_16362

-- Statement 1
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem sin_kpi_minus_x_is_odd (k : ℤ) :
  is_odd_function (λ x => Real.sin (k * Real.pi - x)) :=
sorry

-- Statement 2
theorem cos_squared_when_tan_pi_minus_x_is_two :
  ∀ x, Real.tan (Real.pi - x) = 2 → Real.cos x ^ 2 = 1/5 :=
sorry

-- Statement 3
def is_line_of_symmetry (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

theorem cos_2x_plus_pi_third_symmetry :
  is_line_of_symmetry (λ x => Real.cos (2*x + Real.pi/3)) (-2*Real.pi/3) :=
sorry

end NUMINAMATH_CALUDE_sin_kpi_minus_x_is_odd_cos_squared_when_tan_pi_minus_x_is_two_cos_2x_plus_pi_third_symmetry_l163_16362


namespace NUMINAMATH_CALUDE_not_odd_implies_exists_neq_l163_16384

/-- A function is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem not_odd_implies_exists_neq (f : ℝ → ℝ) (h : ¬IsOdd f) : 
  ∃ x, f (-x) ≠ -f x := by
  sorry

end NUMINAMATH_CALUDE_not_odd_implies_exists_neq_l163_16384


namespace NUMINAMATH_CALUDE_parallel_vectors_l163_16320

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def IsParallel (v w : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, v.1 = t * w.1 ∧ v.2 = t * w.2

theorem parallel_vectors (k : ℝ) :
  let a : ℝ × ℝ := (-1, 1)
  let b : ℝ × ℝ := (2, 3)
  let c : ℝ × ℝ := (-2, k)
  IsParallel (a.1 + b.1, a.2 + b.2) c → k = -8 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_l163_16320


namespace NUMINAMATH_CALUDE_power_of_729_l163_16381

theorem power_of_729 : (729 : ℝ) ^ (4/6 : ℝ) = 81 :=
by
  have h : 729 = 3^6 := by sorry
  sorry

end NUMINAMATH_CALUDE_power_of_729_l163_16381


namespace NUMINAMATH_CALUDE_simple_interest_problem_l163_16326

theorem simple_interest_problem (principal rate time : ℝ) : 
  principal = 2100 →
  principal * (rate + 1) * time / 100 = principal * rate * time / 100 + 63 →
  time = 3 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l163_16326


namespace NUMINAMATH_CALUDE_tank_filling_time_l163_16303

def pipe1_rate : ℚ := 1 / 8
def pipe2_rate : ℚ := 1 / 12

def combined_rate : ℚ := pipe1_rate + pipe2_rate

theorem tank_filling_time : (1 : ℚ) / combined_rate = 24 / 5 := by sorry

end NUMINAMATH_CALUDE_tank_filling_time_l163_16303


namespace NUMINAMATH_CALUDE_min_value_expression_l163_16309

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  a^2 / b + b^2 / c + c^2 / a ≥ 3 ∧
  (a^2 / b + b^2 / c + c^2 / a = 3 ↔ a = 1 ∧ b = 1 ∧ c = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l163_16309


namespace NUMINAMATH_CALUDE_six_students_three_competitions_l163_16345

/-- The number of ways to assign students to competitions -/
def registration_methods (num_students : ℕ) (num_competitions : ℕ) : ℕ :=
  num_competitions ^ num_students

/-- Theorem: The number of ways to assign 6 students to 3 competitions is 729 -/
theorem six_students_three_competitions :
  registration_methods 6 3 = 729 := by
  sorry

end NUMINAMATH_CALUDE_six_students_three_competitions_l163_16345


namespace NUMINAMATH_CALUDE_gumball_difference_l163_16392

theorem gumball_difference (x y : ℕ) : 
  let total := 16 + 12 + 20 + x + y
  (18 * 5 ≤ total ∧ total ≤ 27 * 5) →
  (∃ (x' y' : ℕ), 
    let total' := 16 + 12 + 20 + x' + y'
    (18 * 5 ≤ total' ∧ total' ≤ 27 * 5) ∧
    x' + y' - (x + y) = 45) :=
by sorry

end NUMINAMATH_CALUDE_gumball_difference_l163_16392


namespace NUMINAMATH_CALUDE_leos_laundry_bill_l163_16316

/-- The total bill amount for Leo's laundry -/
def total_bill_amount (trousers_count : ℕ) (initial_shirts_count : ℕ) (missing_shirts_count : ℕ) 
  (trouser_price : ℕ) (shirt_price : ℕ) : ℕ :=
  trousers_count * trouser_price + (initial_shirts_count + missing_shirts_count) * shirt_price

/-- Theorem stating that Leo's total bill amount is $140 -/
theorem leos_laundry_bill : 
  total_bill_amount 10 2 8 9 5 = 140 := by
  sorry

#eval total_bill_amount 10 2 8 9 5

end NUMINAMATH_CALUDE_leos_laundry_bill_l163_16316


namespace NUMINAMATH_CALUDE_right_triangle_area_l163_16349

/-- The area of a right triangle formed by two perpendicular vectors -/
theorem right_triangle_area (a b : ℝ × ℝ) (h : a.1 * b.1 + a.2 * b.2 = 0) :
  let area := (1/2) * abs (a.1 * b.2 - a.2 * b.1)
  (a = (3, 4) ∧ b = (-4, 3)) → area = 12.5 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l163_16349


namespace NUMINAMATH_CALUDE_card_sum_problem_l163_16330

theorem card_sum_problem (a b c d e f g h : ℕ) :
  (a + b) * (c + d) * (e + f) * (g + h) = 330 →
  a + b + c + d + e + f + g + h = 21 := by
sorry

end NUMINAMATH_CALUDE_card_sum_problem_l163_16330


namespace NUMINAMATH_CALUDE_invariant_preserved_not_all_blue_l163_16352

/-- Represents the state of chameleons on the island -/
structure ChameleonState where
  red : ℕ
  green : ℕ
  blue : ℕ

/-- The initial state of chameleons -/
def initial_state : ChameleonState :=
  { red := 25, green := 12, blue := 8 }

/-- Represents a single interaction between chameleons -/
inductive Interaction
  | SameColor : Interaction
  | DifferentColor : Interaction

/-- Applies an interaction to the current state -/
def apply_interaction (state : ChameleonState) (interaction : Interaction) : ChameleonState :=
  sorry

/-- The invariant that remains constant after each interaction -/
def invariant (state : ChameleonState) : ℕ :=
  (state.red - state.green) % 3

/-- Theorem stating that the invariant remains constant after any interaction -/
theorem invariant_preserved (state : ChameleonState) (interaction : Interaction) :
  invariant state = invariant (apply_interaction state interaction) :=
  sorry

/-- Theorem stating that it's impossible for all chameleons to be blue -/
theorem not_all_blue (state : ChameleonState) :
  (∃ n : ℕ, (state.red = 0 ∧ state.green = 0 ∧ state.blue = n)) →
  state ≠ initial_state ∧ 
  ¬∃ (interactions : List Interaction), 
    state = List.foldl apply_interaction initial_state interactions :=
  sorry

end NUMINAMATH_CALUDE_invariant_preserved_not_all_blue_l163_16352


namespace NUMINAMATH_CALUDE_min_students_per_bench_l163_16344

theorem min_students_per_bench (male_students : ℕ) (benches : ℕ) : 
  male_students = 29 →
  benches = 29 →
  let female_students := 4 * male_students
  let total_students := male_students + female_students
  (total_students + benches - 1) / benches = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_students_per_bench_l163_16344


namespace NUMINAMATH_CALUDE_equation_is_quadratic_l163_16323

/-- Definition of a quadratic equation in one variable -/
def is_quadratic_equation_in_one_variable (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x^2 = 3 -/
def equation (x : ℝ) : ℝ := x^2 - 3

theorem equation_is_quadratic : is_quadratic_equation_in_one_variable equation := by
  sorry

end NUMINAMATH_CALUDE_equation_is_quadratic_l163_16323


namespace NUMINAMATH_CALUDE_addition_closed_in_P_l163_16373

-- Define the set P
def P : Set ℝ := {n | ∃ k : ℕ+, n = Real.log k}

-- State the theorem
theorem addition_closed_in_P (a b : ℝ) (ha : a ∈ P) (hb : b ∈ P) : 
  a + b ∈ P := by sorry

end NUMINAMATH_CALUDE_addition_closed_in_P_l163_16373


namespace NUMINAMATH_CALUDE_field_length_proof_l163_16325

theorem field_length_proof (width : ℝ) (length : ℝ) (pond_side : ℝ) :
  length = 2 * width →
  pond_side = 5 →
  pond_side^2 = (1/8) * (length * width) →
  length = 20 := by
  sorry

end NUMINAMATH_CALUDE_field_length_proof_l163_16325


namespace NUMINAMATH_CALUDE_pencil_distribution_l163_16312

theorem pencil_distribution (num_pens : ℕ) (num_pencils : ℕ) (num_students : ℕ) :
  num_pens = 1048 →
  num_students = 4 →
  num_pens % num_students = 0 →
  num_pencils % num_students = 0 →
  num_students = Nat.gcd num_pens num_pencils →
  num_pencils % 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_pencil_distribution_l163_16312


namespace NUMINAMATH_CALUDE_expression_simplification_l163_16327

theorem expression_simplification (a b : ℝ) 
  (h : |2*a - 1| + (b + 4)^2 = 0) : 
  a^3*b - a^2*b^3 - 1/2*(4*a*b - 6*a^2*b^3 - 1) + 2*(a*b - a^2*b^3) = 0 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l163_16327


namespace NUMINAMATH_CALUDE_largest_inscribed_square_side_length_l163_16389

/-- The side length of the largest inscribed square in a specific configuration -/
theorem largest_inscribed_square_side_length :
  ∃ (large_square_side : ℝ) (triangle_side : ℝ) (inscribed_square_side : ℝ),
    large_square_side = 12 ∧
    triangle_side = 4 * Real.sqrt 6 ∧
    inscribed_square_side = 6 - Real.sqrt 6 ∧
    2 * inscribed_square_side * Real.sqrt 2 + triangle_side = large_square_side * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_largest_inscribed_square_side_length_l163_16389


namespace NUMINAMATH_CALUDE_perpendicular_from_point_to_line_l163_16379

-- Define the plane
variable (Plane : Type)

-- Define points and lines
variable (Point : Plane → Type)
variable (Line : Plane → Type)

-- Define the relation of a point being on a line
variable (on_line : ∀ {p : Plane}, Point p → Line p → Prop)

-- Define perpendicularity of lines
variable (perpendicular : ∀ {p : Plane}, Line p → Line p → Prop)

-- Define the operation of drawing a line through two points
variable (line_through : ∀ {p : Plane}, Point p → Point p → Line p)

-- Define the operation of erecting a perpendicular to a line at a point
variable (erect_perpendicular : ∀ {p : Plane}, Line p → Point p → Line p)

-- Theorem statement
theorem perpendicular_from_point_to_line 
  {p : Plane} (A : Point p) (L : Line p) 
  (h : ¬ on_line A L) : 
  ∃ (M : Line p), perpendicular M L ∧ on_line A M := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_from_point_to_line_l163_16379


namespace NUMINAMATH_CALUDE_correct_guess_probability_l163_16380

/-- The probability of guessing the correct last digit of a 6-digit password in no more than 2 attempts -/
def guess_probability : ℚ := 1/5

/-- The number of possible digits for each position in the password -/
def digit_options : ℕ := 10

/-- The number of attempts allowed to guess the last digit -/
def max_attempts : ℕ := 2

theorem correct_guess_probability :
  guess_probability = 1 / digit_options + (1 - 1 / digit_options) * (1 / (digit_options - 1)) :=
sorry

end NUMINAMATH_CALUDE_correct_guess_probability_l163_16380


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l163_16342

def is_in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

theorem z_in_second_quadrant (z : ℂ) (h : z * Complex.I = -1 - Complex.I) :
  is_in_second_quadrant z := by
  sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l163_16342


namespace NUMINAMATH_CALUDE_interest_for_one_rupee_l163_16387

/-- Given that for 5000 rs, the interest is 200 paise, prove that the interest for 1 rs is 0.04 paise. -/
theorem interest_for_one_rupee (interest_5000 : ℝ) (h : interest_5000 = 200) :
  interest_5000 / 5000 = 0.04 := by
sorry

end NUMINAMATH_CALUDE_interest_for_one_rupee_l163_16387


namespace NUMINAMATH_CALUDE_least_integer_satisfying_inequality_l163_16308

theorem least_integer_satisfying_inequality :
  ∀ x : ℤ, (3 * |x| + 4 < 19) → x ≥ -4 ∧
  ∃ y : ℤ, y = -4 ∧ (3 * |y| + 4 < 19) :=
by sorry

end NUMINAMATH_CALUDE_least_integer_satisfying_inequality_l163_16308


namespace NUMINAMATH_CALUDE_lemming_average_distance_l163_16306

/-- The average distance from a point to the sides of a square --/
theorem lemming_average_distance (side_length : ℝ) (diagonal_distance : ℝ) (turn_angle : ℝ) (final_distance : ℝ) : 
  side_length = 12 →
  diagonal_distance = 7.8 →
  turn_angle = 60 * π / 180 →
  final_distance = 3 →
  let d := (diagonal_distance / (side_length * Real.sqrt 2))
  let x := d * side_length + final_distance * Real.cos (π/2 - turn_angle)
  let y := d * side_length + final_distance * Real.sin (π/2 - turn_angle)
  (x + y + (side_length - x) + (side_length - y)) / 4 = 6 := by
sorry

end NUMINAMATH_CALUDE_lemming_average_distance_l163_16306


namespace NUMINAMATH_CALUDE_max_digit_sum_l163_16317

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def decimal_to_fraction (a b c : ℕ) : ℚ :=
  (a * 100000 + a * 10000 + b * 1000 + b * 100 + c * 10 + c) / 1000000

theorem max_digit_sum (a b c y : ℕ) :
  is_digit a → is_digit b → is_digit c →
  decimal_to_fraction a b c = 1 / y →
  y > 0 → y ≤ 16 →
  a + b + c ≤ 13 :=
sorry

end NUMINAMATH_CALUDE_max_digit_sum_l163_16317


namespace NUMINAMATH_CALUDE_boating_group_size_l163_16382

theorem boating_group_size : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 4 = 10 ∧ 
  n % 5 = 1 ∧ 
  n = 46 := by
  sorry

end NUMINAMATH_CALUDE_boating_group_size_l163_16382


namespace NUMINAMATH_CALUDE_triangle_problem_l163_16335

-- Define the triangle ABC
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side opposite to A
  b : ℝ  -- Side opposite to B
  c : ℝ  -- Side opposite to C

-- Define the theorem
theorem triangle_problem (t : Triangle) 
  (h1 : t.a = 2 * t.a * Real.cos t.A * Real.cos t.B - 2 * t.b * Real.sin t.A * Real.sin t.A)
  (h2 : t.a * t.b * Real.sin t.C / 2 = 15 * Real.sqrt 3 / 4)
  (h3 : t.a + t.b + t.c = 15) :
  t.C = 2 * Real.pi / 3 ∧ t.c = 7 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l163_16335


namespace NUMINAMATH_CALUDE_division_remainder_problem_l163_16350

theorem division_remainder_problem (a b : ℕ) (h1 : a - b = 1000) (h2 : ∃ q r, a = b * q + r ∧ q = 10) (h3 : a = 1100) : 
  ∃ r, a = b * 10 + r ∧ r = 100 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l163_16350


namespace NUMINAMATH_CALUDE_hexagon_area_for_given_triangle_l163_16348

/-- Given an isosceles triangle PQR with circumcircle radius r and perimeter p,
    calculate the area of the hexagon formed by the intersections of the
    perpendicular bisectors of the sides with the circumcircle. -/
def hexagon_area (r p : ℝ) : ℝ :=
  5 * p

theorem hexagon_area_for_given_triangle :
  hexagon_area 10 42 = 210 := by sorry

end NUMINAMATH_CALUDE_hexagon_area_for_given_triangle_l163_16348


namespace NUMINAMATH_CALUDE_consecutive_numbers_with_perfect_square_digit_sums_l163_16319

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Check if a natural number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

/-- Theorem: There exist two consecutive natural numbers greater than 1,000,000 
    whose sums of digits are perfect squares -/
theorem consecutive_numbers_with_perfect_square_digit_sums : 
  ∃ n : ℕ, n > 1000000 ∧ 
    is_perfect_square (sum_of_digits n) ∧ 
    is_perfect_square (sum_of_digits (n + 1)) := by sorry

end NUMINAMATH_CALUDE_consecutive_numbers_with_perfect_square_digit_sums_l163_16319


namespace NUMINAMATH_CALUDE_complex_equation_roots_l163_16368

theorem complex_equation_roots : ∃ (z₁ z₂ : ℂ), 
  z₁ = 3 - I ∧ z₂ = -2 + I ∧ 
  z₁^2 - z₁ = 5 - 5*I ∧ 
  z₂^2 - z₂ = 5 - 5*I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_roots_l163_16368


namespace NUMINAMATH_CALUDE_patricia_money_l163_16310

theorem patricia_money (jethro carmen patricia : ℕ) : 
  carmen = 2 * jethro - 7 →
  patricia = 3 * jethro →
  jethro + carmen + patricia = 113 →
  patricia = 60 := by
sorry

end NUMINAMATH_CALUDE_patricia_money_l163_16310


namespace NUMINAMATH_CALUDE_sqrt_two_irrational_others_rational_l163_16307

theorem sqrt_two_irrational_others_rational : 
  (∃ (q : ℚ), Real.sqrt 2 = (q : ℝ)) ∧ 
  (∃ (q : ℚ), (1 : ℝ) = (q : ℝ)) ∧ 
  (∃ (q : ℚ), (0 : ℝ) = (q : ℝ)) ∧ 
  (∃ (q : ℚ), (-1 : ℝ) = (q : ℝ)) →
  False :=
by sorry

end NUMINAMATH_CALUDE_sqrt_two_irrational_others_rational_l163_16307


namespace NUMINAMATH_CALUDE_sum_of_specific_numbers_l163_16375

theorem sum_of_specific_numbers : 1235 + 2351 + 3512 + 5123 = 12221 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_numbers_l163_16375


namespace NUMINAMATH_CALUDE_problem_hexagon_area_l163_16395

/-- Represents a hexagon formed by stretching a rubber band over pegs on a grid. -/
structure Hexagon where
  interior_points : ℕ
  boundary_points : ℕ

/-- Calculates the area of a hexagon using Pick's Theorem. -/
def area (h : Hexagon) : ℕ :=
  h.interior_points + h.boundary_points / 2 - 1

/-- The hexagon formed on the 5x5 grid as described in the problem. -/
def problem_hexagon : Hexagon :=
  { interior_points := 11
  , boundary_points := 6 }

/-- Theorem stating that the area of the problem hexagon is 13 square units. -/
theorem problem_hexagon_area :
  area problem_hexagon = 13 := by
  sorry

#eval area problem_hexagon  -- Should output 13

end NUMINAMATH_CALUDE_problem_hexagon_area_l163_16395


namespace NUMINAMATH_CALUDE_student_math_percentage_l163_16385

/-- The percentage a student got in math, given their history score, third subject score,
    and desired overall average. -/
def math_percentage (history : ℝ) (third_subject : ℝ) (overall_average : ℝ) : ℝ :=
  3 * overall_average - history - third_subject

/-- Theorem stating that the student got 74% in math, given the conditions. -/
theorem student_math_percentage :
  math_percentage 81 70 75 = 74 := by
  sorry

end NUMINAMATH_CALUDE_student_math_percentage_l163_16385


namespace NUMINAMATH_CALUDE_square_difference_1989_l163_16378

theorem square_difference_1989 :
  {(a, b) : ℕ × ℕ | a > b ∧ a ^ 2 - b ^ 2 = 1989} =
  {(995, 994), (333, 330), (115, 106), (83, 70), (67, 50), (45, 6)} :=
by sorry

end NUMINAMATH_CALUDE_square_difference_1989_l163_16378


namespace NUMINAMATH_CALUDE_lcm_of_24_and_16_l163_16318

theorem lcm_of_24_and_16 :
  let n : ℕ := 24
  let m : ℕ := 16
  Nat.gcd n m = 8 →
  Nat.lcm n m = 48 := by
sorry

end NUMINAMATH_CALUDE_lcm_of_24_and_16_l163_16318


namespace NUMINAMATH_CALUDE_inequality_solution_l163_16331

theorem inequality_solution (x : ℝ) :
  x ≠ 3 →
  (x * (x + 2) / (x - 3)^2 ≥ 8 ↔ x ∈ Set.Iic (18/7) ∪ Set.Ioi 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l163_16331


namespace NUMINAMATH_CALUDE_geometric_sequence_11th_term_l163_16334

/-- Given a geometric sequence where the 5th term is 8 and the 8th term is 64,
    prove that the 11th term is 512. -/
theorem geometric_sequence_11th_term (a : ℝ) (r : ℝ) :
  a * r^4 = 8 → a * r^7 = 64 → a * r^10 = 512 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_11th_term_l163_16334


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l163_16399

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - 2 * a * x + 1 > 0) → (0 ≤ a ∧ a < 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l163_16399


namespace NUMINAMATH_CALUDE_statement_C_is_false_l163_16315

-- Define the concept of lines in space
variable (Line : Type)

-- Define the perpendicular relationship between lines
variable (perpendicular : Line → Line → Prop)

-- Define the parallel relationship between lines
variable (parallel : Line → Line → Prop)

-- State the theorem to be proven false
theorem statement_C_is_false :
  ¬(∀ (a b c : Line), perpendicular a c → perpendicular b c → parallel a b) :=
sorry

end NUMINAMATH_CALUDE_statement_C_is_false_l163_16315


namespace NUMINAMATH_CALUDE_light_wattage_increase_l163_16305

theorem light_wattage_increase (original_wattage new_wattage : ℝ) 
  (h1 : original_wattage = 80)
  (h2 : new_wattage = 100) :
  (new_wattage - original_wattage) / original_wattage * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_light_wattage_increase_l163_16305


namespace NUMINAMATH_CALUDE_quadrilateral_property_implication_l163_16383

-- Define a quadrilateral type
structure Quadrilateral :=
  (A B C D : Point)

-- Define the three properties
def diagonals_perpendicular (q : Quadrilateral) : Prop := sorry

def inscribed_in_circle (q : Quadrilateral) : Prop := sorry

def perpendicular_through_intersection (q : Quadrilateral) : Prop := sorry

-- Main theorem
theorem quadrilateral_property_implication (q : Quadrilateral) :
  (diagonals_perpendicular q ∧ inscribed_in_circle q) ∨
  (diagonals_perpendicular q ∧ perpendicular_through_intersection q) ∨
  (inscribed_in_circle q ∧ perpendicular_through_intersection q) →
  diagonals_perpendicular q ∧ inscribed_in_circle q ∧ perpendicular_through_intersection q :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_property_implication_l163_16383


namespace NUMINAMATH_CALUDE_unanswered_questions_count_l163_16358

/-- Represents the scoring system for a math test. -/
structure ScoringSystem where
  correct : Int
  wrong : Int
  unanswered : Int
  initial : Int

/-- Represents the result of a math test. -/
structure TestResult where
  correct : Nat
  wrong : Nat
  unanswered : Nat
  total_score : Int

/-- Calculates the score based on a given scoring system and test result. -/
def calculate_score (system : ScoringSystem) (result : TestResult) : Int :=
  system.initial +
  system.correct * result.correct +
  system.wrong * result.wrong +
  system.unanswered * result.unanswered

theorem unanswered_questions_count
  (new_system : ScoringSystem)
  (old_system : ScoringSystem)
  (result : TestResult)
  (h1 : new_system = { correct := 6, wrong := 0, unanswered := 3, initial := 0 })
  (h2 : old_system = { correct := 4, wrong := -1, unanswered := 0, initial := 40 })
  (h3 : result.correct + result.wrong + result.unanswered = 35)
  (h4 : calculate_score new_system result = 120)
  (h5 : calculate_score old_system result = 100) :
  result.unanswered = 5 := by
  sorry

end NUMINAMATH_CALUDE_unanswered_questions_count_l163_16358


namespace NUMINAMATH_CALUDE_problem_statement_l163_16372

theorem problem_statement (x y : ℝ) 
  (eq1 : x + x*y + y = 2 + 3*Real.sqrt 2) 
  (eq2 : x^2 + y^2 = 6) : 
  |x + y + 1| = 3 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l163_16372


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l163_16363

theorem cubic_equation_solutions :
  let x₁ : ℂ := 4
  let x₂ : ℂ := -2 + 2 * Complex.I * Real.sqrt 3
  let x₃ : ℂ := -2 - 2 * Complex.I * Real.sqrt 3
  (∀ x : ℂ, 2 * x^3 = 128 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l163_16363


namespace NUMINAMATH_CALUDE_equation_solution_difference_l163_16328

theorem equation_solution_difference : ∃ (s₁ s₂ : ℝ),
  s₁ ≠ s₂ ∧
  s₁ ≠ -6 ∧
  s₂ ≠ -6 ∧
  (s₁^2 - 5*s₁ - 24) / (s₁ + 6) = 3*s₁ + 10 ∧
  (s₂^2 - 5*s₂ - 24) / (s₂ + 6) = 3*s₂ + 10 ∧
  |s₁ - s₂| = 6.5 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_difference_l163_16328


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l163_16322

theorem arithmetic_calculation : 3127 + 240 / 60 * 5 - 227 = 2920 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l163_16322


namespace NUMINAMATH_CALUDE_taxi_charge_proof_l163_16398

/-- Calculates the total charge for a taxi trip -/
def total_charge (initial_fee : ℚ) (rate_per_increment : ℚ) (increment_distance : ℚ) (trip_distance : ℚ) : ℚ :=
  initial_fee + (trip_distance / increment_distance).floor * rate_per_increment

/-- Proves that the total charge for a 3.6-mile trip is $7.65 -/
theorem taxi_charge_proof :
  let initial_fee : ℚ := 9/4  -- $2.25
  let rate_per_increment : ℚ := 3/10  -- $0.3
  let increment_distance : ℚ := 2/5  -- 2/5 mile
  let trip_distance : ℚ := 18/5  -- 3.6 miles
  total_charge initial_fee rate_per_increment increment_distance trip_distance = 153/20  -- $7.65
  := by sorry


end NUMINAMATH_CALUDE_taxi_charge_proof_l163_16398


namespace NUMINAMATH_CALUDE_quadratic_equations_integer_roots_l163_16386

theorem quadratic_equations_integer_roots :
  ∃ (p q : ℤ), ∀ k : ℕ, k ≤ 9 →
    ∃ (x y : ℤ), x^2 + (p + k) * x + (q + k) = 0 ∧
                 y^2 + (p + k) * y + (q + k) = 0 ∧
                 x ≠ y :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_integer_roots_l163_16386


namespace NUMINAMATH_CALUDE_diane_age_to_25_diane_age_problem_l163_16337

theorem diane_age_to_25 (denise_future_age : ℕ) (years_until_denise_future : ℕ) (age_difference : ℕ) : ℕ :=
  let denise_current_age := denise_future_age - years_until_denise_future
  let diane_current_age := denise_current_age - age_difference
  let years_until_diane_25 := denise_future_age - diane_current_age
  years_until_diane_25

theorem diane_age_problem :
  diane_age_to_25 25 2 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_diane_age_to_25_diane_age_problem_l163_16337


namespace NUMINAMATH_CALUDE_seven_digit_numbers_with_zero_l163_16336

def seven_digit_numbers : ℕ := 9 * (10 ^ 6)
def seven_digit_numbers_without_zero : ℕ := 9 ^ 7

theorem seven_digit_numbers_with_zero (h1 : seven_digit_numbers = 9 * (10 ^ 6))
                                      (h2 : seven_digit_numbers_without_zero = 9 ^ 7) :
  seven_digit_numbers - seven_digit_numbers_without_zero = 8521704 := by
  sorry

end NUMINAMATH_CALUDE_seven_digit_numbers_with_zero_l163_16336


namespace NUMINAMATH_CALUDE_soap_box_length_proof_l163_16374

/-- Represents the dimensions of a box in inches -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (box : BoxDimensions) : ℝ :=
  box.length * box.width * box.height

theorem soap_box_length_proof 
  (carton : BoxDimensions) 
  (soap_box : BoxDimensions) 
  (max_boxes : ℕ) :
  carton.length = 25 ∧ 
  carton.width = 42 ∧ 
  carton.height = 60 ∧
  soap_box.width = 6 ∧ 
  soap_box.height = 10 ∧
  max_boxes = 150 ∧
  (max_boxes : ℝ) * boxVolume soap_box = boxVolume carton →
  soap_box.length = 7 := by
sorry

end NUMINAMATH_CALUDE_soap_box_length_proof_l163_16374


namespace NUMINAMATH_CALUDE_internally_tangent_circles_l163_16321

/-- Given two circles, where one is internally tangent to the other, prove the possible values of m -/
theorem internally_tangent_circles (m : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 = m) ∧ 
  (∃ (x y : ℝ), x^2 + y^2 + 6*x - 8*y - 11 = 0) ∧ 
  (∃ (x y : ℝ), x^2 + y^2 = m ∧ x^2 + y^2 + 6*x - 8*y - 11 = 0) →
  m = 1 ∨ m = 121 :=
by sorry

end NUMINAMATH_CALUDE_internally_tangent_circles_l163_16321


namespace NUMINAMATH_CALUDE_difference_of_squares_l163_16357

theorem difference_of_squares (x y : ℝ) 
  (sum_eq : x + y = 24) 
  (diff_eq : x - y = 8) : 
  x^2 - y^2 = 192 := by
sorry

end NUMINAMATH_CALUDE_difference_of_squares_l163_16357


namespace NUMINAMATH_CALUDE_midpoint_property_l163_16355

/-- Given two points A and B in the plane, if C is their midpoint,
    then 2x - 4y = 0, where (x, y) are the coordinates of C. -/
theorem midpoint_property (A B : ℝ × ℝ) (hA : A = (20, 10)) (hB : B = (10, 5)) :
  let C := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  2 * C.1 - 4 * C.2 = 0 := by
sorry

end NUMINAMATH_CALUDE_midpoint_property_l163_16355


namespace NUMINAMATH_CALUDE_smallest_perimeter_is_23_l163_16301

/-- A scalene triangle with prime side lengths greater than 3 and prime perimeter. -/
structure ScaleneTriangle where
  /-- First side length -/
  a : ℕ
  /-- Second side length -/
  b : ℕ
  /-- Third side length -/
  c : ℕ
  /-- Proof that a is prime -/
  a_prime : Nat.Prime a
  /-- Proof that b is prime -/
  b_prime : Nat.Prime b
  /-- Proof that c is prime -/
  c_prime : Nat.Prime c
  /-- Proof that a is greater than 3 -/
  a_gt_three : a > 3
  /-- Proof that b is greater than 3 -/
  b_gt_three : b > 3
  /-- Proof that c is greater than 3 -/
  c_gt_three : c > 3
  /-- Proof that a, b, and c are distinct -/
  distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c
  /-- Proof that a, b, and c form a valid triangle -/
  triangle_inequality : a + b > c ∧ a + c > b ∧ b + c > a
  /-- Proof that the perimeter is prime -/
  perimeter_prime : Nat.Prime (a + b + c)

/-- The smallest possible perimeter of a scalene triangle with the given conditions is 23. -/
theorem smallest_perimeter_is_23 : ∀ t : ScaleneTriangle, t.a + t.b + t.c ≥ 23 := by
  sorry

end NUMINAMATH_CALUDE_smallest_perimeter_is_23_l163_16301


namespace NUMINAMATH_CALUDE_area_of_region_l163_16367

/-- Rectangle with sides of length 2 -/
structure Rectangle :=
  (width : ℝ)
  (height : ℝ)
  (is_2x2 : width = 2 ∧ height = 2)

/-- Equilateral triangle with side length 2 -/
structure EquilateralTriangle :=
  (side_length : ℝ)
  (is_side_2 : side_length = 2)

/-- Region R inside rectangle and outside triangle -/
structure Region (rect : Rectangle) (tri : EquilateralTriangle) :=
  (inside_rectangle : Prop)
  (outside_triangle : Prop)
  (distance_from_AD : ℝ → Prop)

/-- The theorem to be proved -/
theorem area_of_region 
  (rect : Rectangle) 
  (tri : EquilateralTriangle) 
  (R : Region rect tri) : 
  ∃ (area : ℝ), 
    area = (4 - Real.sqrt 3) / 6 ∧ 
    (∀ x, R.distance_from_AD x → 2/3 ≤ x ∧ x ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_area_of_region_l163_16367


namespace NUMINAMATH_CALUDE_complex_expression_equality_combinatorial_equality_l163_16341

-- Part I
theorem complex_expression_equality : 
  (((Complex.abs (1 - Complex.I)) / Real.sqrt 2) ^ 16 + 
   ((1 + 2 * Complex.I) ^ 2) / (1 - Complex.I)) = 
  (-5 / 2 : ℂ) + (1 / 2 : ℂ) * Complex.I := by sorry

-- Part II
theorem combinatorial_equality (m : ℕ) : 
  (1 / Nat.choose 5 m : ℚ) - (1 / Nat.choose 6 m : ℚ) = 
  (7 : ℚ) / (10 * Nat.choose 7 m) → 
  Nat.choose 8 m = 28 := by sorry

end NUMINAMATH_CALUDE_complex_expression_equality_combinatorial_equality_l163_16341


namespace NUMINAMATH_CALUDE_girls_attending_event_l163_16324

theorem girls_attending_event (total_students : ℕ) (total_attending : ℕ) 
  (h_total : total_students = 1500)
  (h_attending : total_attending = 900)
  (h_girls_ratio : ∀ g : ℕ, g ≤ total_students → (3 * g) / 4 ≤ total_attending)
  (h_boys_ratio : ∀ b : ℕ, b ≤ total_students → (2 * b) / 5 ≤ total_attending)
  (h_all_students : ∀ g b : ℕ, g + b = total_students → (3 * g) / 4 + (2 * b) / 5 = total_attending) :
  ∃ g : ℕ, g ≤ total_students ∧ (3 * g) / 4 = 643 := by
sorry

end NUMINAMATH_CALUDE_girls_attending_event_l163_16324


namespace NUMINAMATH_CALUDE_sum_of_ages_l163_16393

/-- Given information about the ages of Nacho, Divya, and Samantha, prove that the sum of their current ages is 80 years. -/
theorem sum_of_ages (nacho divya samantha : ℕ) : 
  (divya = 5) →
  (nacho + 5 = 3 * (divya + 5)) →
  (samantha = 2 * nacho) →
  (nacho + divya + samantha = 80) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_ages_l163_16393
