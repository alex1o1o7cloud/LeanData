import Mathlib

namespace NUMINAMATH_CALUDE_fencing_cost_per_foot_l2653_265357

/-- The cost of fencing per foot -/
def cost_per_foot (side_length back_length total_cost : ℚ) : ℚ :=
  let total_length := 2 * side_length + back_length
  let neighbor_back_contribution := back_length / 2
  let neighbor_left_contribution := side_length / 3
  let cole_length := total_length - neighbor_back_contribution - neighbor_left_contribution
  total_cost / cole_length

/-- Theorem stating that the cost per foot of fencing is $3 -/
theorem fencing_cost_per_foot :
  cost_per_foot 9 18 72 = 3 :=
sorry

end NUMINAMATH_CALUDE_fencing_cost_per_foot_l2653_265357


namespace NUMINAMATH_CALUDE_B_2_1_equals_12_l2653_265305

-- Define the function B using the given recurrence relation
def B : ℕ → ℕ → ℕ
| 0, n => n + 2
| m + 1, 0 => B m 2
| m + 1, n + 1 => B m (B (m + 1) n)

-- Theorem statement
theorem B_2_1_equals_12 : B 2 1 = 12 := by
  sorry

end NUMINAMATH_CALUDE_B_2_1_equals_12_l2653_265305


namespace NUMINAMATH_CALUDE_cardinality_of_C_l2653_265319

def A : Finset ℕ := {0, 2, 3, 4, 5, 7}
def B : Finset ℕ := {1, 2, 3, 4, 6}
def C : Finset ℕ := A \ B

theorem cardinality_of_C : Finset.card C = 3 := by
  sorry

end NUMINAMATH_CALUDE_cardinality_of_C_l2653_265319


namespace NUMINAMATH_CALUDE_even_function_with_range_l2653_265385

/-- Given a function f(x) = (x + a)(bx - a) where a and b are real constants,
    if f is an even function and its range is [-4, +∞),
    then f(x) = x^2 - 4 -/
theorem even_function_with_range (a b : ℝ) :
  (∀ x, (x + a) * (b * x - a) = ((-(x : ℝ)) + a) * (b * (-x) - a)) →
  (∀ y ≥ -4, ∃ x, (x + a) * (b * x - a) = y) →
  (∀ x, (x + a) * (b * x - a) = x^2 - 4) :=
by sorry

end NUMINAMATH_CALUDE_even_function_with_range_l2653_265385


namespace NUMINAMATH_CALUDE_dance_group_average_age_l2653_265363

/-- Calculates the average age of a dance group given the number and average ages of children and adults. -/
theorem dance_group_average_age 
  (num_children : ℕ) 
  (num_adults : ℕ) 
  (avg_age_children : ℚ) 
  (avg_age_adults : ℚ) 
  (h1 : num_children = 8) 
  (h2 : num_adults = 12) 
  (h3 : avg_age_children = 12) 
  (h4 : avg_age_adults = 40) :
  let total_members := num_children + num_adults
  let total_age := num_children * avg_age_children + num_adults * avg_age_adults
  total_age / total_members = 288 / 10 := by
  sorry

end NUMINAMATH_CALUDE_dance_group_average_age_l2653_265363


namespace NUMINAMATH_CALUDE_smallest_spend_l2653_265335

/-- Represents a gift set with its composition and price -/
structure GiftSet where
  chocolates : ℕ
  caramels : ℕ
  price : ℕ

/-- The first type of gift set -/
def gift1 : GiftSet := { chocolates := 3, caramels := 15, price := 350 }

/-- The second type of gift set -/
def gift2 : GiftSet := { chocolates := 20, caramels := 5, price := 500 }

/-- Calculates the total cost of buying gift sets -/
def totalCost (m n : ℕ) : ℕ := m * gift1.price + n * gift2.price

/-- Calculates the total number of chocolate candies -/
def totalChocolates (m n : ℕ) : ℕ := m * gift1.chocolates + n * gift2.chocolates

/-- Calculates the total number of caramel candies -/
def totalCaramels (m n : ℕ) : ℕ := m * gift1.caramels + n * gift2.caramels

/-- Theorem stating the smallest non-zero amount Eugene needs to spend -/
theorem smallest_spend : 
  ∃ m n : ℕ, m + n > 0 ∧ 
    totalChocolates m n = totalCaramels m n ∧
    totalCost m n = 3750 ∧
    ∀ k l : ℕ, k + l > 0 → 
      totalChocolates k l = totalCaramels k l → 
      totalCost k l ≥ 3750 := by sorry

end NUMINAMATH_CALUDE_smallest_spend_l2653_265335


namespace NUMINAMATH_CALUDE_set_cardinality_relation_l2653_265393

theorem set_cardinality_relation (a b : ℕ+) (A B : Finset ℕ+) :
  (A ∩ B = ∅) →
  (∀ i ∈ A ∪ B, (i + a) ∈ A ∨ (i - b) ∈ B) →
  a * A.card = b * B.card :=
sorry

end NUMINAMATH_CALUDE_set_cardinality_relation_l2653_265393


namespace NUMINAMATH_CALUDE_max_containers_proof_l2653_265332

def oatmeal_cookies : ℕ := 50
def chocolate_chip_cookies : ℕ := 75
def sugar_cookies : ℕ := 36

theorem max_containers_proof :
  let gcd := Nat.gcd oatmeal_cookies (Nat.gcd chocolate_chip_cookies sugar_cookies)
  (sugar_cookies / gcd) = 7 ∧ 
  (oatmeal_cookies / gcd) ≥ 7 ∧ 
  (chocolate_chip_cookies / gcd) ≥ 7 :=
by sorry

end NUMINAMATH_CALUDE_max_containers_proof_l2653_265332


namespace NUMINAMATH_CALUDE_ashley_exam_marks_l2653_265367

theorem ashley_exam_marks (marks_secured : ℕ) (percentage : ℚ) (max_marks : ℕ) : 
  marks_secured = 332 → percentage = 83/100 → 
  (marks_secured : ℚ) / (max_marks : ℚ) = percentage →
  max_marks = 400 := by
sorry

end NUMINAMATH_CALUDE_ashley_exam_marks_l2653_265367


namespace NUMINAMATH_CALUDE_sum_interior_angles_polygon_l2653_265331

/-- The sum of interior angles of an n-sided polygon -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- The number of triangles formed by drawing diagonals from one vertex -/
def num_triangles (n : ℕ) : ℕ := n - 2

/-- The number of diagonals drawn from one vertex -/
def num_diagonals (n : ℕ) : ℕ := n - 3

theorem sum_interior_angles_polygon (n : ℕ) (h : n ≥ 3) :
  sum_interior_angles n = (num_triangles n) * 180 :=
by sorry


end NUMINAMATH_CALUDE_sum_interior_angles_polygon_l2653_265331


namespace NUMINAMATH_CALUDE_nala_seashells_l2653_265316

/-- The number of seashells Nala found on the first day -/
def first_day : ℕ := 5

/-- The total number of seashells Nala has -/
def total : ℕ := 36

/-- The number of seashells Nala found on the second day -/
def second_day : ℕ := 7

theorem nala_seashells : 
  first_day + second_day + 2 * (first_day + second_day) = total := by
  sorry

#check nala_seashells

end NUMINAMATH_CALUDE_nala_seashells_l2653_265316


namespace NUMINAMATH_CALUDE_polygon_interior_angles_l2653_265390

theorem polygon_interior_angles (n : ℕ) (h : n > 2) :
  (∃ x : ℝ, x > 0 ∧ x < 180 ∧ 2400 + x = (n - 2) * 180) → n = 16 := by
  sorry

end NUMINAMATH_CALUDE_polygon_interior_angles_l2653_265390


namespace NUMINAMATH_CALUDE_sum_of_solutions_squared_equation_l2653_265324

theorem sum_of_solutions_squared_equation (x : ℝ) :
  (∃ a b : ℝ, (a - 8)^2 = 49 ∧ (b - 8)^2 = 49 ∧ a ≠ b) →
  (∃ s : ℝ, s = a + b ∧ s = 16) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_squared_equation_l2653_265324


namespace NUMINAMATH_CALUDE_greatest_mpn_l2653_265302

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  ones : Nat
  tens_nonzero : tens ≠ 0
  tens_single_digit : tens < 10
  ones_single_digit : ones < 10

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  hundreds_nonzero : hundreds ≠ 0
  hundreds_single_digit : hundreds < 10
  tens_single_digit : tens < 10
  ones_single_digit : ones < 10

def is_valid_mpn (m n : Nat) (mpn : ThreeDigitNumber) : Prop :=
  m ≠ n ∧
  m < 10 ∧
  n < 10 ∧
  mpn.hundreds = m ∧
  mpn.ones = m ∧
  (10 * m + n) * m = 100 * mpn.hundreds + 10 * mpn.tens + mpn.ones

theorem greatest_mpn :
  ∀ m n : Nat,
  ∀ mpn : ThreeDigitNumber,
  is_valid_mpn m n mpn →
  mpn.hundreds * 100 + mpn.tens * 10 + mpn.ones ≤ 898 :=
sorry

end NUMINAMATH_CALUDE_greatest_mpn_l2653_265302


namespace NUMINAMATH_CALUDE_chess_club_girls_l2653_265379

theorem chess_club_girls (total : ℕ) (present : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 30 →
  present = 18 →
  boys + girls = total →
  boys + (1/3 : ℚ) * girls = present →
  girls = 18 :=
by sorry

end NUMINAMATH_CALUDE_chess_club_girls_l2653_265379


namespace NUMINAMATH_CALUDE_find_m_l2653_265396

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem find_m : ∃ m : ℕ, m * factorial m + 2 * factorial m = 5040 ∧ m = 5 := by
  sorry

end NUMINAMATH_CALUDE_find_m_l2653_265396


namespace NUMINAMATH_CALUDE_probability_theorem_l2653_265313

def team_sizes : List Nat := [6, 9, 10]
def co_captains_per_team : Nat := 3
def members_selected : Nat := 3

def probability_all_co_captains (sizes : List Nat) (co_captains : Nat) (selected : Nat) : ℚ :=
  let team_probabilities := sizes.map (λ n => (co_captains.factorial * (n - co_captains).choose (selected - co_captains)) / n.choose selected)
  (1 / sizes.length) * team_probabilities.sum

theorem probability_theorem :
  probability_all_co_captains team_sizes co_captains_per_team members_selected = 177 / 12600 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l2653_265313


namespace NUMINAMATH_CALUDE_system_has_solution_l2653_265329

/-- Given a system of equations {sin x + a = b x, cos x = b} where a and b are real numbers,
    and the equation sin x + a = b x has exactly two solutions,
    prove that the system has at least one solution. -/
theorem system_has_solution (a b : ℝ) 
    (h : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
         (∀ x, Real.sin x + a = b * x ↔ x = x₁ ∨ x = x₂)) :
  ∃ x, Real.sin x + a = b * x ∧ Real.cos x = b := by
  sorry

end NUMINAMATH_CALUDE_system_has_solution_l2653_265329


namespace NUMINAMATH_CALUDE_max_value_implies_a_value_l2653_265358

-- Define the function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * a * x

-- Define the theorem
theorem max_value_implies_a_value (a : ℝ) (h1 : a ≠ 0) :
  (∃ (M : ℝ), M = 3 ∧ ∀ x ∈ Set.Icc 0 3, f a x ≤ M) →
  (a = 1 ∨ a = -3) := by
  sorry

end NUMINAMATH_CALUDE_max_value_implies_a_value_l2653_265358


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_range_of_m_for_full_solution_set_solution_set_eq_nonnegative_reals_l2653_265347

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 1| + |x - 3|

-- Theorem for part I
theorem solution_set_of_inequality (x : ℝ) :
  f x ≤ 3 * x + 4 ↔ x ≥ 0 := by sorry

-- Theorem for part II
theorem range_of_m_for_full_solution_set :
  ∀ m : ℝ, (∀ x : ℝ, f x ≥ m) ↔ m ∈ Set.Iic 4 := by sorry

-- Define the solution set for part I
def solution_set : Set ℝ := {x : ℝ | f x ≤ 3 * x + 4}

-- Theorem stating that the solution set is equivalent to [0, +∞)
theorem solution_set_eq_nonnegative_reals :
  solution_set = Set.Ici 0 := by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_range_of_m_for_full_solution_set_solution_set_eq_nonnegative_reals_l2653_265347


namespace NUMINAMATH_CALUDE_residue_of_negative_1001_mod_37_l2653_265300

theorem residue_of_negative_1001_mod_37 :
  -1001 ≡ 35 [ZMOD 37] := by sorry

end NUMINAMATH_CALUDE_residue_of_negative_1001_mod_37_l2653_265300


namespace NUMINAMATH_CALUDE_chandler_apples_per_week_l2653_265368

/-- The number of apples Chandler can eat per week -/
def chandler_apples : ℕ := 23

/-- The number of apples Lucy can eat per week -/
def lucy_apples : ℕ := 19

/-- The number of apples ordered for a month -/
def monthly_order : ℕ := 168

/-- The number of weeks in a month -/
def weeks_per_month : ℕ := 4

/-- Theorem stating that Chandler can eat 23 apples per week -/
theorem chandler_apples_per_week :
  chandler_apples * weeks_per_month + lucy_apples * weeks_per_month = monthly_order :=
by sorry

end NUMINAMATH_CALUDE_chandler_apples_per_week_l2653_265368


namespace NUMINAMATH_CALUDE_largest_810_triple_l2653_265343

/-- Converts a base-10 number to its base-8 representation as a list of digits -/
def toBase8 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) : List ℕ :=
    if m = 0 then [] else (m % 8) :: aux (m / 8)
  aux n |>.reverse

/-- Converts a list of digits to its base-10 representation -/
def fromDigits (digits : List ℕ) : ℕ :=
  digits.foldl (fun acc d => 10 * acc + d) 0

/-- Checks if a number is an 8-10 triple -/
def is810Triple (n : ℕ) : Prop :=
  fromDigits (toBase8 n) = 3 * n

/-- Statement: 273 is the largest 8-10 triple -/
theorem largest_810_triple : 
  (∀ m : ℕ, m > 273 → ¬ is810Triple m) ∧ is810Triple 273 :=
sorry

end NUMINAMATH_CALUDE_largest_810_triple_l2653_265343


namespace NUMINAMATH_CALUDE_bryden_receives_correct_amount_l2653_265323

/-- The face value of a state quarter in dollars -/
def face_value : ℚ := 1/4

/-- The number of quarters Bryden is selling -/
def num_quarters : ℕ := 6

/-- The percentage of face value offered by the collector -/
def offer_percentage : ℕ := 1500

/-- The amount Bryden will receive in dollars -/
def amount_received : ℚ := (offer_percentage : ℚ) / 100 * face_value * num_quarters

theorem bryden_receives_correct_amount : amount_received = 45/2 := by
  sorry

end NUMINAMATH_CALUDE_bryden_receives_correct_amount_l2653_265323


namespace NUMINAMATH_CALUDE_power_product_equality_l2653_265333

theorem power_product_equality (a : ℝ) : a^2 * (-a)^2 = a^4 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l2653_265333


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l2653_265371

/-- Given two orthogonal vectors a = (x-1, y) and b = (1, 2), with x > 0 and y > 0,
    the minimum value of 1/x + 1/y is 3 + 2√2 -/
theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) 
    (h_orthogonal : (x - 1) * 1 + y * 2 = 0) :
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → (x' - 1) * 1 + y' * 2 = 0 → 1 / x' + 1 / y' ≥ 1 / x + 1 / y) ∧
  1 / x + 1 / y = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l2653_265371


namespace NUMINAMATH_CALUDE_gwen_final_amount_l2653_265370

def initial_amount : ℚ := 5.00
def candy_expense : ℚ := 3.25
def recycling_income : ℚ := 1.50
def card_expense : ℚ := 0.70

theorem gwen_final_amount :
  initial_amount - candy_expense + recycling_income - card_expense = 2.55 := by
  sorry

end NUMINAMATH_CALUDE_gwen_final_amount_l2653_265370


namespace NUMINAMATH_CALUDE_bobby_shoe_count_bobby_shoe_count_proof_l2653_265383

/-- Given the relationships between Bonny's, Becky's, and Bobby's shoe counts, 
    prove that Bobby has 27 pairs of shoes. -/
theorem bobby_shoe_count : ℕ → ℕ → Prop :=
  fun becky_shoes bobby_shoes =>
    -- Bonny has 13 pairs of shoes
    -- Bonny's shoe count is 5 less than twice Becky's
    13 = 2 * becky_shoes - 5 →
    -- Bobby has 3 times as many shoes as Becky
    bobby_shoes = 3 * becky_shoes →
    -- Prove that Bobby has 27 pairs of shoes
    bobby_shoes = 27

/-- Proof of the theorem -/
theorem bobby_shoe_count_proof : ∃ (becky_shoes : ℕ), bobby_shoe_count becky_shoes 27 := by
  sorry

end NUMINAMATH_CALUDE_bobby_shoe_count_bobby_shoe_count_proof_l2653_265383


namespace NUMINAMATH_CALUDE_seven_correct_guesses_l2653_265386

/-- A guess is either a lower bound (not less than) or an upper bound (not more than) -/
inductive Guess
  | LowerBound (n : Nat)
  | UpperBound (n : Nat)

/-- The set of guesses made by the teachers -/
def teacherGuesses : List Guess := [
  Guess.LowerBound 1, Guess.UpperBound 2,
  Guess.LowerBound 3, Guess.UpperBound 4,
  Guess.LowerBound 5, Guess.UpperBound 6,
  Guess.LowerBound 7, Guess.UpperBound 8,
  Guess.LowerBound 9, Guess.UpperBound 10,
  Guess.LowerBound 11, Guess.UpperBound 12
]

/-- A guess is correct if it's satisfied by the given number -/
def isCorrectGuess (x : Nat) (g : Guess) : Bool :=
  match g with
  | Guess.LowerBound n => x ≥ n
  | Guess.UpperBound n => x ≤ n

/-- The number of correct guesses for a given number -/
def correctGuessCount (x : Nat) : Nat :=
  (teacherGuesses.filter (isCorrectGuess x)).length

/-- There exists a number for which exactly 7 guesses are correct -/
theorem seven_correct_guesses : ∃ x, correctGuessCount x = 7 := by
  sorry

end NUMINAMATH_CALUDE_seven_correct_guesses_l2653_265386


namespace NUMINAMATH_CALUDE_prob_three_same_color_l2653_265377

def total_marbles : ℕ := 23
def red_marbles : ℕ := 6
def white_marbles : ℕ := 8
def blue_marbles : ℕ := 9

def prob_same_color : ℚ := 160 / 1771

theorem prob_three_same_color :
  let prob_red := (red_marbles / total_marbles) * ((red_marbles - 1) / (total_marbles - 1)) * ((red_marbles - 2) / (total_marbles - 2))
  let prob_white := (white_marbles / total_marbles) * ((white_marbles - 1) / (total_marbles - 1)) * ((white_marbles - 2) / (total_marbles - 2))
  let prob_blue := (blue_marbles / total_marbles) * ((blue_marbles - 1) / (total_marbles - 1)) * ((blue_marbles - 2) / (total_marbles - 2))
  prob_red + prob_white + prob_blue = prob_same_color := by
sorry

end NUMINAMATH_CALUDE_prob_three_same_color_l2653_265377


namespace NUMINAMATH_CALUDE_chess_team_girls_l2653_265356

theorem chess_team_girls (total : ℕ) (attended : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 26 →
  attended = 16 →
  boys + girls = total →
  boys + (girls / 2) = attended →
  girls = 20 := by
sorry

end NUMINAMATH_CALUDE_chess_team_girls_l2653_265356


namespace NUMINAMATH_CALUDE_arithmetic_simplification_l2653_265354

theorem arithmetic_simplification :
  (5 + 7 + 3) / 3 - 2 / 3 = 13 / 3 := by sorry

end NUMINAMATH_CALUDE_arithmetic_simplification_l2653_265354


namespace NUMINAMATH_CALUDE_expression_evaluation_l2653_265355

-- Define the ceiling function
def ceiling (x : ℚ) : ℤ := Int.ceil x

-- Define the main expression
def main_expression : ℚ :=
  (ceiling ((21 : ℚ) / 5 - ceiling ((35 : ℚ) / 23))) /
  (ceiling ((35 : ℚ) / 5 + ceiling ((5 * 23 : ℚ) / 35)))

-- Theorem statement
theorem expression_evaluation :
  main_expression = 3 / 11 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2653_265355


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l2653_265373

def a : Fin 2 → ℝ := ![1, 2]
def b (x : ℝ) : Fin 2 → ℝ := ![x, 6]

theorem vector_difference_magnitude 
  (h_parallel : ∃ (k : ℝ), ∀ i, a i = k * b x i) :
  ‖a - b x‖ = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l2653_265373


namespace NUMINAMATH_CALUDE_cos_135_degrees_l2653_265375

theorem cos_135_degrees : Real.cos (135 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_135_degrees_l2653_265375


namespace NUMINAMATH_CALUDE_vanessa_missed_days_l2653_265304

theorem vanessa_missed_days (total : ℕ) (vanessa_mike : ℕ) (mike_sarah : ℕ)
  (h1 : total = 17)
  (h2 : vanessa_mike = 14)
  (h3 : mike_sarah = 12) :
  ∃ (vanessa mike sarah : ℕ),
    vanessa + mike + sarah = total ∧
    vanessa + mike = vanessa_mike ∧
    mike + sarah = mike_sarah ∧
    vanessa = 5 := by
  sorry

end NUMINAMATH_CALUDE_vanessa_missed_days_l2653_265304


namespace NUMINAMATH_CALUDE_parabola_directrix_l2653_265384

/-- The equation of the directrix of the parabola y = x^2 is y = -1/4 -/
theorem parabola_directrix : 
  ∀ (x y : ℝ), y = x^2 → (∃ (k : ℝ), y = k ∧ k = -1/4) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l2653_265384


namespace NUMINAMATH_CALUDE_middle_share_is_forty_l2653_265359

/-- Represents the distribution of marbles among three people -/
structure MarbleDistribution where
  total : ℕ
  ratio1 : ℕ
  ratio2 : ℕ
  ratio3 : ℕ

/-- Calculates the number of marbles for the person with the middle ratio -/
def middleShare (d : MarbleDistribution) : ℕ :=
  d.total * d.ratio2 / (d.ratio1 + d.ratio2 + d.ratio3)

/-- Theorem: In a distribution of 120 marbles with ratio 4:5:6, the middle share is 40 -/
theorem middle_share_is_forty : 
  let d : MarbleDistribution := ⟨120, 4, 5, 6⟩
  middleShare d = 40 := by sorry


end NUMINAMATH_CALUDE_middle_share_is_forty_l2653_265359


namespace NUMINAMATH_CALUDE_num_solutions_eq_1176_l2653_265339

/-- The number of distinct ordered triples (a, b, c) of positive integers satisfying a + b + c = 50 -/
def num_solutions : ℕ :=
  (Finset.range 49).sum (λ k ↦ 49 - k)

/-- Theorem stating that the number of solutions is 1176 -/
theorem num_solutions_eq_1176 : num_solutions = 1176 := by
  sorry

end NUMINAMATH_CALUDE_num_solutions_eq_1176_l2653_265339


namespace NUMINAMATH_CALUDE_track_circumference_l2653_265340

/-- The circumference of a circular track given specific running conditions -/
theorem track_circumference (brenda_first_meeting : ℝ) (sally_second_meeting : ℝ) 
  (h1 : brenda_first_meeting = 120)
  (h2 : sally_second_meeting = 180) :
  let circumference := brenda_first_meeting * 3/2
  circumference = 180 := by sorry

end NUMINAMATH_CALUDE_track_circumference_l2653_265340


namespace NUMINAMATH_CALUDE_expression_evaluation_l2653_265351

theorem expression_evaluation :
  let a : ℚ := -1/2
  let b : ℚ := 3
  3 * a^2 - b^2 - (a^2 - 6*a) - 2*(-b^2 + 3*a) = 19/2 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2653_265351


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_l2653_265314

theorem largest_integer_with_remainder (n : ℕ) : 
  n < 80 ∧ n % 8 = 5 ∧ ∀ m, m < 80 ∧ m % 8 = 5 → m ≤ n → n = 77 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_l2653_265314


namespace NUMINAMATH_CALUDE_function_value_at_three_pi_four_l2653_265376

noncomputable def f (A φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (x + φ)

theorem function_value_at_three_pi_four
  (A φ : ℝ)
  (h1 : A > 0)
  (h2 : 0 < φ)
  (h3 : φ < Real.pi)
  (h4 : ∀ x, f A φ x ≤ 1)
  (h5 : ∃ x, f A φ x = 1)
  (h6 : f A φ (Real.pi / 3) = 1 / 2) :
  f A φ (3 * Real.pi / 4) = -Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_function_value_at_three_pi_four_l2653_265376


namespace NUMINAMATH_CALUDE_cody_initial_tickets_l2653_265361

def initial_tickets : ℕ → Prop
  | t => (t - 25 + 6 = 30)

theorem cody_initial_tickets : ∃ t : ℕ, initial_tickets t ∧ t = 49 := by
  sorry

end NUMINAMATH_CALUDE_cody_initial_tickets_l2653_265361


namespace NUMINAMATH_CALUDE_no_mn_divisibility_l2653_265387

theorem no_mn_divisibility : ¬∃ (m n : ℕ+), 
  (m.val * n.val ∣ 3^m.val + 1) ∧ (m.val * n.val ∣ 3^n.val + 1) := by
  sorry

end NUMINAMATH_CALUDE_no_mn_divisibility_l2653_265387


namespace NUMINAMATH_CALUDE_apple_pear_puzzle_l2653_265318

theorem apple_pear_puzzle (apples pears : ℕ) : 
  (apples : ℚ) / 3 = (pears : ℚ) / 2 + 1 →
  (apples : ℚ) / 5 = (pears : ℚ) / 4 - 3 →
  apples = 23 ∧ pears = 16 := by
sorry

end NUMINAMATH_CALUDE_apple_pear_puzzle_l2653_265318


namespace NUMINAMATH_CALUDE_square_sum_value_l2653_265317

theorem square_sum_value (x y : ℝ) (h1 : (x + y)^2 = 36) (h2 : x * y = 8) : x^2 + y^2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_value_l2653_265317


namespace NUMINAMATH_CALUDE_specific_trapezoid_area_l2653_265322

/-- Represents an isosceles trapezoid with given measurements -/
structure IsoscelesTrapezoid where
  leg_length : ℝ
  diagonal_length : ℝ
  longer_base : ℝ

/-- Calculates the area of the isosceles trapezoid -/
def trapezoid_area (t : IsoscelesTrapezoid) : ℝ :=
  -- The actual calculation is not implemented here
  sorry

/-- Theorem stating that the area of the specific trapezoid is approximately 318.93 -/
theorem specific_trapezoid_area :
  let t : IsoscelesTrapezoid := {
    leg_length := 20,
    diagonal_length := 25,
    longer_base := 30
  }
  abs (trapezoid_area t - 318.93) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_specific_trapezoid_area_l2653_265322


namespace NUMINAMATH_CALUDE_sum_of_square_roots_inequality_l2653_265326

theorem sum_of_square_roots_inequality (a b c d : ℝ) 
  (non_neg_a : 0 ≤ a) (non_neg_b : 0 ≤ b) (non_neg_c : 0 ≤ c) (non_neg_d : 0 ≤ d)
  (sum_eq_four : a + b + c + d = 4) : 
  Real.sqrt (a + b + c) + Real.sqrt (b + c + d) + Real.sqrt (c + d + a) + Real.sqrt (d + a + b) ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_square_roots_inequality_l2653_265326


namespace NUMINAMATH_CALUDE_art_museum_cost_l2653_265389

def total_cost (initial_fee : ℕ) (initial_visits_per_year : ℕ) (new_fee : ℕ) (new_visits_per_year : ℕ) (total_years : ℕ) : ℕ :=
  (initial_fee * initial_visits_per_year) + (new_fee * new_visits_per_year * (total_years - 1))

theorem art_museum_cost : 
  total_cost 5 12 7 4 3 = 116 := by sorry

end NUMINAMATH_CALUDE_art_museum_cost_l2653_265389


namespace NUMINAMATH_CALUDE_c_rent_share_l2653_265394

/-- Represents a person's pasture usage -/
structure PastureUsage where
  oxen : ℕ
  months : ℕ

/-- Calculates the total oxen-months for a given pasture usage -/
def oxenMonths (usage : PastureUsage) : ℕ :=
  usage.oxen * usage.months

/-- Calculates the share of rent for a given usage and total usage -/
def rentShare (usage : PastureUsage) (totalUsage : ℕ) (totalRent : ℕ) : ℚ :=
  (oxenMonths usage : ℚ) / (totalUsage : ℚ) * (totalRent : ℚ)

theorem c_rent_share :
  let a := PastureUsage.mk 10 7
  let b := PastureUsage.mk 12 5
  let c := PastureUsage.mk 15 3
  let totalRent := 175
  let totalUsage := oxenMonths a + oxenMonths b + oxenMonths c
  rentShare c totalUsage totalRent = 45 := by
  sorry

end NUMINAMATH_CALUDE_c_rent_share_l2653_265394


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2653_265381

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  c = 2 →
  C = π / 3 →
  (1/2 * a * b * Real.sin C = Real.sqrt 3 → a = 2 ∧ b = 2) ∧
  (Real.sin B = 2 * Real.sin A → 1/2 * a * b * Real.sin C = 2 * Real.sqrt 3 / 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l2653_265381


namespace NUMINAMATH_CALUDE_initial_balance_was_200_l2653_265327

/-- Represents the balance of Yasmin's bank account throughout the week --/
structure BankAccount where
  initial : ℝ
  monday : ℝ
  tuesday : ℝ
  wednesday : ℝ
  thursday : ℝ

/-- Calculates the final balance of Yasmin's account after all transactions --/
def finalBalance (account : BankAccount) : ℝ :=
  account.thursday

/-- Theorem stating that the initial balance was $200 --/
theorem initial_balance_was_200 (account : BankAccount) :
  account.initial = 200 ∧
  account.monday = account.initial / 2 ∧
  account.tuesday = account.monday + 30 ∧
  account.wednesday = 200 ∧
  account.thursday = account.wednesday - 20 ∧
  finalBalance account = 160 :=
by sorry

end NUMINAMATH_CALUDE_initial_balance_was_200_l2653_265327


namespace NUMINAMATH_CALUDE_kelly_peanuts_weight_l2653_265399

/-- Given the total weight of snacks and the weight of raisins, 
    calculate the weight of peanuts Kelly bought. -/
theorem kelly_peanuts_weight 
  (total_snacks : ℝ) 
  (raisins_weight : ℝ) 
  (h1 : total_snacks = 0.5) 
  (h2 : raisins_weight = 0.4) : 
  total_snacks - raisins_weight = 0.1 := by
  sorry

#check kelly_peanuts_weight

end NUMINAMATH_CALUDE_kelly_peanuts_weight_l2653_265399


namespace NUMINAMATH_CALUDE_quincy_peter_difference_l2653_265312

/-- The number of pictures Randy drew -/
def randy_pictures : ℕ := 5

/-- The number of additional pictures Peter drew compared to Randy -/
def peter_additional : ℕ := 3

/-- The total number of pictures drawn by all three -/
def total_pictures : ℕ := 41

/-- The number of pictures Peter drew -/
def peter_pictures : ℕ := randy_pictures + peter_additional

/-- The number of pictures Quincy drew -/
def quincy_pictures : ℕ := total_pictures - randy_pictures - peter_pictures

theorem quincy_peter_difference : quincy_pictures - peter_pictures = 20 := by
  sorry

end NUMINAMATH_CALUDE_quincy_peter_difference_l2653_265312


namespace NUMINAMATH_CALUDE_eggs_leftover_l2653_265346

theorem eggs_leftover (abigail beatrice carson : ℕ) 
  (h_abigail : abigail = 60)
  (h_beatrice : beatrice = 75)
  (h_carson : carson = 27) :
  (abigail + beatrice + carson) % 18 = 0 := by
sorry

end NUMINAMATH_CALUDE_eggs_leftover_l2653_265346


namespace NUMINAMATH_CALUDE_water_needed_for_mixture_l2653_265369

/-- Given a mixture of nutrient concentrate and water, calculate the amount of water needed to prepare a larger volume of the same mixture. -/
theorem water_needed_for_mixture (concentrate : ℝ) (initial_water : ℝ) (total_desired : ℝ) : 
  concentrate = 0.05 → 
  initial_water = 0.03 → 
  total_desired = 0.72 → 
  (initial_water / (concentrate + initial_water)) * total_desired = 0.27 := by
sorry

end NUMINAMATH_CALUDE_water_needed_for_mixture_l2653_265369


namespace NUMINAMATH_CALUDE_volume_equivalence_l2653_265325

/-- A parallelepiped with congruent rhombic faces and a special vertex -/
structure RhombicParallelepiped where
  -- Side length of the rhombic face
  a : ℝ
  -- Angle between edges at the special vertex
  α : ℝ
  -- Diagonals of the rhombic face
  e : ℝ
  f : ℝ
  -- Conditions
  a_pos : 0 < a
  α_pos : 0 < α
  α_not_right : α ≠ π / 2
  α_less_120 : α < 2 * π / 3
  e_pos : 0 < e
  f_pos : 0 < f
  diag_relation : a = (1 / 2) * Real.sqrt (e^2 + f^2)
  angle_relation : Real.tan (α / 2) = f / e

/-- The volume of a rhombic parallelepiped -/
noncomputable def volume (p : RhombicParallelepiped) : ℝ :=
  p.a^3 * Real.sin p.α * Real.sqrt (Real.sin p.α^2 - Real.cos p.α^2 * Real.tan (p.α / 2)^2)

/-- The volume of a rhombic parallelepiped in terms of diagonals -/
noncomputable def volume_diag (p : RhombicParallelepiped) : ℝ :=
  (p.f / (8 * p.e)) * (p.e^2 + p.f^2) * Real.sqrt (3 * p.e^2 - p.f^2)

/-- The main theorem: equivalence of volume formulas -/
theorem volume_equivalence (p : RhombicParallelepiped) : volume p = volume_diag p := by
  sorry

end NUMINAMATH_CALUDE_volume_equivalence_l2653_265325


namespace NUMINAMATH_CALUDE_sum_of_coordinates_symmetric_points_l2653_265308

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = -x₂ ∧ y₁ = -y₂

/-- The theorem states that if A(a, 2) and B(-3, b) are symmetric with respect to the origin, then a + b = 1 -/
theorem sum_of_coordinates_symmetric_points (a b : ℝ) 
  (h : symmetric_wrt_origin a 2 (-3) b) : a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_symmetric_points_l2653_265308


namespace NUMINAMATH_CALUDE_stock_price_calculation_l2653_265320

def stock_price_evolution (initial_price : ℝ) (first_year_increase : ℝ) (second_year_decrease : ℝ) : ℝ :=
  let price_after_first_year := initial_price * (1 + first_year_increase)
  price_after_first_year * (1 - second_year_decrease)

theorem stock_price_calculation :
  stock_price_evolution 150 0.5 0.3 = 157.5 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_calculation_l2653_265320


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l2653_265310

theorem fraction_equation_solution : 
  ∃ x : ℚ, (3/4 * 60 - x * 60 + 63 = 12) ∧ (x = 8/5) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l2653_265310


namespace NUMINAMATH_CALUDE_pairball_playtime_l2653_265366

/-- Given a game of pairball with the following conditions:
  * There are 12 children participating.
  * Only 2 children can play at a time.
  * The game runs continuously for 120 minutes.
  * Every child has an equal amount of playtime.
  Prove that each child plays for 20 minutes. -/
theorem pairball_playtime (num_children : ℕ) (players_per_game : ℕ) (total_time : ℕ) 
  (h1 : num_children = 12)
  (h2 : players_per_game = 2)
  (h3 : total_time = 120)
  : (total_time * players_per_game) / num_children = 20 := by
  sorry

end NUMINAMATH_CALUDE_pairball_playtime_l2653_265366


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l2653_265362

theorem fractional_equation_solution :
  ∀ x : ℚ, x ≠ 0 → x ≠ 1 → (3 / (x - 1) = 1 / x) ↔ (x = -1/2) :=
by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l2653_265362


namespace NUMINAMATH_CALUDE_painted_cube_theorem_l2653_265328

/-- Represents a cube composed of unit cubes -/
structure PaintedCube where
  size : ℕ
  totalCubes : ℕ
  surfacePainted : Bool

/-- Counts the number of unit cubes with a specific number of faces painted -/
def countPaintedFaces (cube : PaintedCube) (numFaces : ℕ) : ℕ :=
  match numFaces with
  | 3 => 8
  | 2 => 12 * (cube.size - 2)
  | 1 => 6 * (cube.size - 2)^2
  | 0 => (cube.size - 2)^3
  | _ => 0

theorem painted_cube_theorem (cube : PaintedCube) 
  (h1 : cube.size = 10) 
  (h2 : cube.totalCubes = 1000) 
  (h3 : cube.surfacePainted = true) :
  (countPaintedFaces cube 3 = 8) ∧
  (countPaintedFaces cube 2 = 96) ∧
  (countPaintedFaces cube 1 = 384) ∧
  (countPaintedFaces cube 0 = 512) := by
  sorry

end NUMINAMATH_CALUDE_painted_cube_theorem_l2653_265328


namespace NUMINAMATH_CALUDE_square_perimeter_l2653_265315

theorem square_perimeter (s : ℝ) (h : s * s = 625) : 4 * s = 100 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l2653_265315


namespace NUMINAMATH_CALUDE_velocity_equal_distance_time_l2653_265365

/-- For uniform motion, the velocity that makes the distance equal to time is 1. -/
theorem velocity_equal_distance_time (s t v : ℝ) (h : s = v * t) (h2 : s = t) : v = 1 := by
  sorry

end NUMINAMATH_CALUDE_velocity_equal_distance_time_l2653_265365


namespace NUMINAMATH_CALUDE_roots_are_zero_neg_five_and_a_l2653_265388

variable (a : ℝ)

def roots : Set ℝ := {x : ℝ | x * (x + 5)^2 * (a - x) = 0}

theorem roots_are_zero_neg_five_and_a : roots a = {0, -5, a} := by
  sorry

end NUMINAMATH_CALUDE_roots_are_zero_neg_five_and_a_l2653_265388


namespace NUMINAMATH_CALUDE_lcm_18_24_30_l2653_265321

theorem lcm_18_24_30 : Nat.lcm 18 (Nat.lcm 24 30) = 360 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_24_30_l2653_265321


namespace NUMINAMATH_CALUDE_product_probability_l2653_265345

/-- Claire's spinner has 7 equally probable outcomes -/
def claire_spinner : ℕ := 7

/-- Jamie's spinner has 12 equally probable outcomes -/
def jamie_spinner : ℕ := 12

/-- The threshold for the product of spins -/
def threshold : ℕ := 42

/-- The probability that the product of Claire's and Jamie's spins is less than the threshold -/
theorem product_probability : 
  (Finset.filter (λ (pair : ℕ × ℕ) => pair.1 * pair.2 < threshold) 
    (Finset.product (Finset.range claire_spinner) (Finset.range jamie_spinner))).card / 
  (claire_spinner * jamie_spinner : ℚ) = 31 / 42 := by sorry

end NUMINAMATH_CALUDE_product_probability_l2653_265345


namespace NUMINAMATH_CALUDE_equation_relationship_l2653_265391

/-- Represents a relationship between x and y --/
inductive Relationship
  | Direct
  | Inverse
  | Neither

/-- Determines the relationship between x and y in the equation 2x + 3y = 15 --/
def relationshipInEquation : Relationship := sorry

/-- Theorem stating that the relationship in the equation 2x + 3y = 15 is neither direct nor inverse proportionality --/
theorem equation_relationship :
  relationshipInEquation = Relationship.Neither := by sorry

end NUMINAMATH_CALUDE_equation_relationship_l2653_265391


namespace NUMINAMATH_CALUDE_sum_of_possible_radii_l2653_265395

/-- A circle with center C(r, r) is tangent to the positive x-axis and y-axis,
    and externally tangent to another circle centered at (3,3) with radius 2.
    This theorem states that the sum of all possible radii r is 16. -/
theorem sum_of_possible_radii : ∃ r₁ r₂ : ℝ,
  (r₁ - 3)^2 + (r₁ - 3)^2 = (r₁ + 2)^2 ∧
  (r₂ - 3)^2 + (r₂ - 3)^2 = (r₂ + 2)^2 ∧
  r₁ + r₂ = 16 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_possible_radii_l2653_265395


namespace NUMINAMATH_CALUDE_smallest_no_inverse_mod_77_88_l2653_265338

theorem smallest_no_inverse_mod_77_88 : 
  ∀ a : ℕ, a > 0 → (Nat.gcd a 77 > 1 ∧ Nat.gcd a 88 > 1) → a ≥ 14 :=
by sorry

end NUMINAMATH_CALUDE_smallest_no_inverse_mod_77_88_l2653_265338


namespace NUMINAMATH_CALUDE_jam_solution_l2653_265372

/-- Represents the amount and consumption rate of jam for a person -/
structure JamConsumption where
  amount : ℝ
  rate : ℝ

/-- The problem of determining jam consumption for Ponchik and Syropchik -/
def jam_problem (ponchik : JamConsumption) (syropchik : JamConsumption) : Prop :=
  -- Total amount of jam
  ponchik.amount + syropchik.amount = 100 ∧
  -- Same time to consume their own supplies
  ponchik.amount / ponchik.rate = syropchik.amount / syropchik.rate ∧
  -- Ponchik's consumption time if he had Syropchik's amount
  syropchik.amount / ponchik.rate = 45 ∧
  -- Syropchik's consumption time if he had Ponchik's amount
  ponchik.amount / syropchik.rate = 20

/-- The solution to the jam consumption problem -/
theorem jam_solution :
  ∃ (ponchik syropchik : JamConsumption),
    jam_problem ponchik syropchik ∧
    ponchik.amount = 40 ∧
    ponchik.rate = 4/3 ∧
    syropchik.amount = 60 ∧
    syropchik.rate = 2 :=
by sorry

end NUMINAMATH_CALUDE_jam_solution_l2653_265372


namespace NUMINAMATH_CALUDE_lavinia_son_katie_daughter_age_ratio_l2653_265311

/-- Proves that the ratio of Lavinia's son's age to Katie's daughter's age is 2:1 given the specified conditions. -/
theorem lavinia_son_katie_daughter_age_ratio :
  ∀ (katie_daughter_age : ℕ) 
    (lavinia_daughter_age : ℕ) 
    (lavinia_son_age : ℕ),
  katie_daughter_age = 12 →
  lavinia_daughter_age = katie_daughter_age - 10 →
  lavinia_son_age = lavinia_daughter_age + 22 →
  ∃ (k : ℕ), k * katie_daughter_age = lavinia_son_age →
  lavinia_son_age / katie_daughter_age = 2 :=
by sorry

end NUMINAMATH_CALUDE_lavinia_son_katie_daughter_age_ratio_l2653_265311


namespace NUMINAMATH_CALUDE_museum_ticket_cost_l2653_265344

theorem museum_ticket_cost (num_students num_teachers : ℕ) 
  (student_ticket_price teacher_ticket_price : ℚ) : 
  num_students = 12 →
  num_teachers = 4 →
  student_ticket_price = 1 →
  teacher_ticket_price = 3 →
  (num_students : ℚ) * student_ticket_price + (num_teachers : ℚ) * teacher_ticket_price = 24 :=
by sorry

end NUMINAMATH_CALUDE_museum_ticket_cost_l2653_265344


namespace NUMINAMATH_CALUDE_fixed_point_exponential_function_l2653_265301

theorem fixed_point_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ 3 + a^(x + 2)
  f (-2) = 4 := by sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_function_l2653_265301


namespace NUMINAMATH_CALUDE_cos_one_sufficient_not_necessary_l2653_265306

theorem cos_one_sufficient_not_necessary (x : ℝ) : 
  (∀ x, Real.cos x = 1 → Real.sin x = 0) ∧ 
  (∃ x, Real.sin x = 0 ∧ Real.cos x ≠ 1) := by
  sorry

end NUMINAMATH_CALUDE_cos_one_sufficient_not_necessary_l2653_265306


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l2653_265364

/-- 
Given a quadratic equation kx^2 + 2x + 1 = 0 with two equal real roots,
prove that k = 1.
-/
theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, k * x^2 + 2 * x + 1 = 0 ∧ 
   ∀ y : ℝ, k * y^2 + 2 * y + 1 = 0 → y = x) → 
  k = 1 := by sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l2653_265364


namespace NUMINAMATH_CALUDE_total_chairs_is_59_l2653_265398

/-- The number of chairs in the office canteen -/
def total_chairs : ℕ :=
  let round_tables : ℕ := 3
  let rectangular_tables : ℕ := 4
  let square_tables : ℕ := 2
  let chairs_per_round_table : ℕ := 6
  let chairs_per_rectangular_table : ℕ := 7
  let chairs_per_square_table : ℕ := 4
  let extra_chairs : ℕ := 5
  (round_tables * chairs_per_round_table) +
  (rectangular_tables * chairs_per_rectangular_table) +
  (square_tables * chairs_per_square_table) +
  extra_chairs

/-- Theorem stating that the total number of chairs in the office canteen is 59 -/
theorem total_chairs_is_59 : total_chairs = 59 := by
  sorry

end NUMINAMATH_CALUDE_total_chairs_is_59_l2653_265398


namespace NUMINAMATH_CALUDE_billy_tickets_left_l2653_265330

theorem billy_tickets_left (tickets_won : ℕ) (difference : ℕ) (tickets_left : ℕ) : 
  tickets_won = 48 → 
  difference = 16 → 
  tickets_won - tickets_left = difference → 
  tickets_left = 32 := by
sorry

end NUMINAMATH_CALUDE_billy_tickets_left_l2653_265330


namespace NUMINAMATH_CALUDE_hyperbola_cosine_theorem_l2653_265352

/-- A hyperbola with equation x^2 - y^2 = 2 -/
def Hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 2

/-- The left focus of the hyperbola -/
def F₁ : ℝ × ℝ := sorry

/-- The right focus of the hyperbola -/
def F₂ : ℝ × ℝ := sorry

/-- A point on the hyperbola -/
def P : ℝ × ℝ := sorry

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem hyperbola_cosine_theorem :
  Hyperbola P.1 P.2 →
  distance P F₁ = 2 * distance P F₂ →
  let cosine_angle := (distance P F₁)^2 + (distance P F₂)^2 - (distance F₁ F₂)^2
                    / (2 * distance P F₁ * distance P F₂)
  cosine_angle = 3/4 := by sorry

end NUMINAMATH_CALUDE_hyperbola_cosine_theorem_l2653_265352


namespace NUMINAMATH_CALUDE_fifth_root_of_102030201_l2653_265350

theorem fifth_root_of_102030201 : (102030201 : ℝ) ^ (1/5 : ℝ) = 101 := by
  sorry

end NUMINAMATH_CALUDE_fifth_root_of_102030201_l2653_265350


namespace NUMINAMATH_CALUDE_upperclassmen_sport_players_l2653_265309

/-- Represents the number of students who play a sport in a college --/
structure SportPlayers where
  total : ℕ
  freshmen : ℕ
  upperclassmen : ℕ
  freshmenPercent : ℚ
  upperclassmenPercent : ℚ
  totalNonPlayersPercent : ℚ

/-- Theorem stating that given the conditions, 383 upperclassmen play a sport --/
theorem upperclassmen_sport_players (sp : SportPlayers)
  (h1 : sp.total = 800)
  (h2 : sp.freshmenPercent = 35 / 100)
  (h3 : sp.upperclassmenPercent = 75 / 100)
  (h4 : sp.totalNonPlayersPercent = 395 / 1000)
  : sp.upperclassmen = 383 := by
  sorry

#check upperclassmen_sport_players

end NUMINAMATH_CALUDE_upperclassmen_sport_players_l2653_265309


namespace NUMINAMATH_CALUDE_tangent_line_circle_range_l2653_265374

theorem tangent_line_circle_range (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_tangent : ∀ x y : ℝ, x + y + a = 0 → (x - b)^2 + (y - 1)^2 ≥ 2) 
  (h_touch : ∃ x y : ℝ, x + y + a = 0 ∧ (x - b)^2 + (y - 1)^2 = 2) :
  ∀ z : ℝ, z > 0 → ∃ c : ℝ, c > 0 ∧ c < 1 ∧ z = c^2 / (1 - c) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_circle_range_l2653_265374


namespace NUMINAMATH_CALUDE_inverse_proportion_order_l2653_265380

theorem inverse_proportion_order (y₁ y₂ y₃ : ℝ) : 
  y₁ = -12 / (-3) → 
  y₂ = -12 / (-2) → 
  y₃ = -12 / 2 → 
  y₃ < y₁ ∧ y₁ < y₂ := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_order_l2653_265380


namespace NUMINAMATH_CALUDE_sum_x_coordinates_preserved_l2653_265307

/-- A polygon in the Cartesian plane -/
structure Polygon where
  vertices : List (ℝ × ℝ)

/-- Create a new polygon from the midpoints of another polygon's sides -/
def midpointPolygon (p : Polygon) : Polygon :=
  sorry

/-- Sum of x-coordinates of a polygon's vertices -/
def sumXCoordinates (p : Polygon) : ℝ :=
  sorry

theorem sum_x_coordinates_preserved (n : ℕ) (q1 q2 q3 : Polygon) :
  n = 44 →
  q1.vertices.length = n →
  sumXCoordinates q1 = 176 →
  q2 = midpointPolygon q1 →
  q3 = midpointPolygon q2 →
  sumXCoordinates q3 = 176 := by
  sorry

end NUMINAMATH_CALUDE_sum_x_coordinates_preserved_l2653_265307


namespace NUMINAMATH_CALUDE_total_buses_is_816_l2653_265342

/-- Represents the bus schedule for different types of days -/
structure BusSchedule where
  weekday : Nat
  saturday : Nat
  sunday_holiday : Nat

/-- Calculates the total number of buses in a month -/
def total_buses_in_month (schedule : BusSchedule) (public_holidays : Nat) : Nat :=
  let weekdays := 20 - public_holidays
  let saturdays := 4
  let sundays_holidays := 4 + public_holidays
  weekdays * schedule.weekday + saturdays * schedule.saturday + sundays_holidays * schedule.sunday_holiday

/-- The bus schedule for the given problem -/
def problem_schedule : BusSchedule :=
  { weekday := 36
  , saturday := 24
  , sunday_holiday := 12 }

/-- Theorem stating that the total number of buses in the month is 816 -/
theorem total_buses_is_816 :
  total_buses_in_month problem_schedule 2 = 816 := by
  sorry

end NUMINAMATH_CALUDE_total_buses_is_816_l2653_265342


namespace NUMINAMATH_CALUDE_pen_bag_discount_l2653_265382

theorem pen_bag_discount (price : ℝ) (discount : ℝ) (savings : ℝ) :
  price = 18 →
  discount = 0.1 →
  savings = 36 →
  ∃ (x : ℝ),
    price * (x + 1) * (1 - discount) = price * x - savings ∧
    x = 30 ∧
    price * (x + 1) * (1 - discount) = 486 :=
by
  sorry

end NUMINAMATH_CALUDE_pen_bag_discount_l2653_265382


namespace NUMINAMATH_CALUDE_vector_collinear_same_direction_l2653_265360

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = (k * a.1, k * a.2)

/-- Two vectors have the same direction if one is a positive scalar multiple of the other -/
def same_direction (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ b = (k * a.1, k * a.2)

/-- Main theorem: If vectors a = (-1, x) and b = (-x, 2) are collinear and have the same direction, then x = √2 -/
theorem vector_collinear_same_direction (x : ℝ) :
  let a : ℝ × ℝ := (-1, x)
  let b : ℝ × ℝ := (-x, 2)
  collinear a b → same_direction a b → x = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_collinear_same_direction_l2653_265360


namespace NUMINAMATH_CALUDE_problem_statement_l2653_265341

theorem problem_statement (a b c : ℝ) (h : a^2 + b^2 + c^2 = 1) :
  (a + b + c = 0 → a*b + b*c + c*a = -1/2) ∧
  ((a + b + c)^2 ≤ 3 ∧ ∃ (x y z : ℝ), x^2 + y^2 + z^2 = 1 ∧ (x + y + z)^2 = 3) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2653_265341


namespace NUMINAMATH_CALUDE_blue_balls_count_l2653_265397

theorem blue_balls_count (black_balls : ℕ) (blue_balls : ℕ) : 
  (black_balls : ℚ) / blue_balls = 5 / 3 → 
  black_balls = 15 → 
  blue_balls = 9 := by
sorry

end NUMINAMATH_CALUDE_blue_balls_count_l2653_265397


namespace NUMINAMATH_CALUDE_matrix_power_1000_l2653_265392

def A : Matrix (Fin 2) (Fin 2) ℝ := !![1, 0; 2, 1]

theorem matrix_power_1000 :
  A ^ 1000 = !![1, 0; 2000, 1] := by sorry

end NUMINAMATH_CALUDE_matrix_power_1000_l2653_265392


namespace NUMINAMATH_CALUDE_part_one_part_two_l2653_265334

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | -x^2 + 2*x + m > 0}

-- Part 1
theorem part_one : A ∩ (Set.univ \ B 3) = {x | 3 ≤ x ∧ x ≤ 5} := by sorry

-- Part 2
theorem part_two : ∃ m : ℝ, m = 8 ∧ A ∩ B m = {x | -1 < x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2653_265334


namespace NUMINAMATH_CALUDE_cubes_fill_box_completely_l2653_265336

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents a cube with a given side length -/
structure Cube where
  sideLength : ℕ

/-- Calculates the volume of a rectangular box -/
def boxVolume (box : BoxDimensions) : ℕ :=
  box.length * box.width * box.height

/-- Calculates the volume of a cube -/
def cubeVolume (cube : Cube) : ℕ :=
  cube.sideLength ^ 3

/-- Calculates the number of cubes that can fit along each dimension of the box -/
def cubesPerDimension (box : BoxDimensions) (cube : Cube) : ℕ × ℕ × ℕ :=
  (box.length / cube.sideLength, box.width / cube.sideLength, box.height / cube.sideLength)

/-- Calculates the total number of cubes that can fit in the box -/
def totalCubes (box : BoxDimensions) (cube : Cube) : ℕ :=
  let (l, w, h) := cubesPerDimension box cube
  l * w * h

/-- Calculates the total volume occupied by the cubes in the box -/
def totalCubeVolume (box : BoxDimensions) (cube : Cube) : ℕ :=
  totalCubes box cube * cubeVolume cube

/-- Theorem: The volume occupied by 4-inch cubes in an 8x4x12 inch box is 100% of the box's volume -/
theorem cubes_fill_box_completely (box : BoxDimensions) (cube : Cube) :
  box.length = 8 ∧ box.width = 4 ∧ box.height = 12 ∧ cube.sideLength = 4 →
  totalCubeVolume box cube = boxVolume box := by
  sorry

#check cubes_fill_box_completely

end NUMINAMATH_CALUDE_cubes_fill_box_completely_l2653_265336


namespace NUMINAMATH_CALUDE_two_digit_number_remainder_l2653_265337

theorem two_digit_number_remainder (n : ℕ) : 
  10 ≤ n ∧ n < 100 →  -- n is a two-digit number
  n % 9 = 1 →         -- remainder when divided by 9 is 1
  n % 10 = 3 →        -- remainder when divided by 10 is 3
  n % 11 = 7 :=       -- remainder when divided by 11 is 7
by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_remainder_l2653_265337


namespace NUMINAMATH_CALUDE_sqrt_6_irrational_l2653_265348

/-- A number is rational if it can be expressed as a ratio of two integers -/
def IsRational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

/-- A number is irrational if it is not rational -/
def IsIrrational (x : ℝ) : Prop := ¬ IsRational x

/-- √6 is irrational -/
theorem sqrt_6_irrational : IsIrrational (Real.sqrt 6) := by sorry

end NUMINAMATH_CALUDE_sqrt_6_irrational_l2653_265348


namespace NUMINAMATH_CALUDE_train_length_l2653_265378

/-- The length of a train given its speed and time to pass a fixed point -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 72 → time = 14 → speed * time * (5 / 18) = 280 :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_l2653_265378


namespace NUMINAMATH_CALUDE_basketball_team_probability_l2653_265303

def team_size : ℕ := 12
def main_players : ℕ := 6
def classes_with_two_students : ℕ := 2
def classes_with_one_student : ℕ := 8

theorem basketball_team_probability :
  (Nat.choose classes_with_two_students 1 * Nat.choose classes_with_two_students 1 * Nat.choose classes_with_one_student 4) / 
  Nat.choose team_size main_players = 10 / 33 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_probability_l2653_265303


namespace NUMINAMATH_CALUDE_price_decrease_approx_l2653_265353

/-- Original price in dollars for 6 cups -/
def original_price : ℚ := 8

/-- Number of cups in original offer -/
def original_cups : ℕ := 6

/-- Promotional price in dollars for 8 cups -/
def promo_price : ℚ := 6

/-- Number of cups in promotional offer -/
def promo_cups : ℕ := 8

/-- Calculate the percent decrease in price per cup -/
def percent_decrease : ℚ :=
  (original_price / original_cups - promo_price / promo_cups) / (original_price / original_cups) * 100

/-- Theorem stating that the percent decrease is approximately 43.6% -/
theorem price_decrease_approx :
  abs (percent_decrease - 43.6) < 0.1 := by sorry

end NUMINAMATH_CALUDE_price_decrease_approx_l2653_265353


namespace NUMINAMATH_CALUDE_smallest_k_inequality_k_is_smallest_l2653_265349

theorem smallest_k_inequality (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  (x * y) ^ (1/3 : ℝ) + (3/8 : ℝ) * (x - y)^2 ≥ (3/8 : ℝ) * (x + y) :=
sorry

theorem k_is_smallest :
  ∀ k : ℝ, k > 0 → k < 3/8 →
  ∃ x y : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ (x * y) ^ (1/3 : ℝ) + k * (x - y)^2 < (3/8 : ℝ) * (x + y) :=
sorry

end NUMINAMATH_CALUDE_smallest_k_inequality_k_is_smallest_l2653_265349
