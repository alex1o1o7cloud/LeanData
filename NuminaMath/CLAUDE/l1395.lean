import Mathlib

namespace NUMINAMATH_CALUDE_fraction_equality_implies_equality_l1395_139597

theorem fraction_equality_implies_equality (a b : ℝ) : 
  a / (-5 : ℝ) = b / (-5 : ℝ) → a = b := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_equality_l1395_139597


namespace NUMINAMATH_CALUDE_circle_k_range_l1395_139565

-- Define the equation
def circle_equation (x y k : ℝ) : Prop :=
  x^2 + y^2 + 4*x + 2*y + 4*k + 1 = 0

-- Define what it means for the equation to represent a circle
def is_circle (k : ℝ) : Prop :=
  ∃ (h r : ℝ), r > 0 ∧ ∀ (x y : ℝ),
    circle_equation x y k ↔ (x + 2)^2 + (y + 1)^2 = r^2

-- Theorem statement
theorem circle_k_range :
  ∀ k : ℝ, is_circle k ↔ k < 1 :=
sorry

end NUMINAMATH_CALUDE_circle_k_range_l1395_139565


namespace NUMINAMATH_CALUDE_order_of_roots_l1395_139501

theorem order_of_roots : 5^(2/3) > 16^(1/3) ∧ 16^(1/3) > 2^(4/5) := by
  sorry

end NUMINAMATH_CALUDE_order_of_roots_l1395_139501


namespace NUMINAMATH_CALUDE_complex_arithmetic_expression_l1395_139547

theorem complex_arithmetic_expression : 
  ((520 * 0.43) / 0.26 - 217 * (2 + 3/7)) - (31.5 / (12 + 3/5) + 114 * (2 + 1/3) + (61 + 1/2)) = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_expression_l1395_139547


namespace NUMINAMATH_CALUDE_count_propositions_and_true_propositions_l1395_139541

-- Define the type for statements
inductive Statement
| RhetoricalQuestion
| Question
| Proposition (isTrue : Bool)
| ExclamatoryStatement
| ConstructionLanguage

-- Define the list of statements
def statements : List Statement := [
  Statement.RhetoricalQuestion,
  Statement.Question,
  Statement.Proposition false,
  Statement.ExclamatoryStatement,
  Statement.Proposition false,
  Statement.ConstructionLanguage
]

-- Theorem to prove
theorem count_propositions_and_true_propositions :
  (statements.filter (fun s => match s with
    | Statement.Proposition _ => true
    | _ => false
  )).length = 2 ∧
  (statements.filter (fun s => match s with
    | Statement.Proposition true => true
    | _ => false
  )).length = 0 := by
  sorry

end NUMINAMATH_CALUDE_count_propositions_and_true_propositions_l1395_139541


namespace NUMINAMATH_CALUDE_books_to_decorations_ratio_l1395_139530

theorem books_to_decorations_ratio 
  (total_books : ℕ) 
  (books_per_shelf : ℕ) 
  (decorations_per_shelf : ℕ) 
  (initial_shelves : ℕ) 
  (h1 : total_books = 42)
  (h2 : books_per_shelf = 2)
  (h3 : decorations_per_shelf = 1)
  (h4 : initial_shelves = 3) :
  (total_books : ℚ) / ((total_books / (books_per_shelf * initial_shelves)) * decorations_per_shelf) = 6 / 1 := by
sorry

end NUMINAMATH_CALUDE_books_to_decorations_ratio_l1395_139530


namespace NUMINAMATH_CALUDE_component_scrap_probability_l1395_139583

/-- The probability of a component passing the first inspection -/
def prob_pass_first : ℝ := 0.8

/-- The probability of a component passing the second inspection -/
def prob_pass_second : ℝ := 0.9

/-- The probability of a component being scrapped -/
def prob_scrapped : ℝ := (1 - prob_pass_first) * (1 - prob_pass_second)

theorem component_scrap_probability : prob_scrapped = 0.02 := by
  sorry

end NUMINAMATH_CALUDE_component_scrap_probability_l1395_139583


namespace NUMINAMATH_CALUDE_probability_masters_degree_expected_value_bachelors_or_higher_male_education_greater_than_female_l1395_139592

/-- Represents the education levels in the census data -/
inductive EducationLevel
  | NoSchooling
  | PrimarySchool
  | JuniorHighSchool
  | HighSchool
  | CollegeAssociate
  | CollegeBachelor
  | MastersDegree
  | DoctoralDegree

/-- Represents the gender in the census data -/
inductive Gender
  | Male
  | Female

/-- Census data for City Z -/
def censusData : Gender → EducationLevel → Float
  | Gender.Male, EducationLevel.NoSchooling => 0.00
  | Gender.Male, EducationLevel.PrimarySchool => 0.03
  | Gender.Male, EducationLevel.JuniorHighSchool => 0.14
  | Gender.Male, EducationLevel.HighSchool => 0.11
  | Gender.Male, EducationLevel.CollegeAssociate => 0.07
  | Gender.Male, EducationLevel.CollegeBachelor => 0.11
  | Gender.Male, EducationLevel.MastersDegree => 0.03
  | Gender.Male, EducationLevel.DoctoralDegree => 0.01
  | Gender.Female, EducationLevel.NoSchooling => 0.01
  | Gender.Female, EducationLevel.PrimarySchool => 0.04
  | Gender.Female, EducationLevel.JuniorHighSchool => 0.11
  | Gender.Female, EducationLevel.HighSchool => 0.11
  | Gender.Female, EducationLevel.CollegeAssociate => 0.08
  | Gender.Female, EducationLevel.CollegeBachelor => 0.12
  | Gender.Female, EducationLevel.MastersDegree => 0.03
  | Gender.Female, EducationLevel.DoctoralDegree => 0.00

/-- Proportion of residents aged 15 and above in City Z -/
def proportionAged15AndAbove : Float := 0.85

/-- Theorem 1: Probability of selecting a person aged 15 and above with a Master's degree -/
theorem probability_masters_degree : 
  proportionAged15AndAbove * (censusData Gender.Male EducationLevel.MastersDegree + 
  censusData Gender.Female EducationLevel.MastersDegree) = 0.051 := by sorry

/-- Theorem 2: Expected value of X (number of people with Bachelor's degree or higher among two randomly selected residents aged 15 and above) -/
theorem expected_value_bachelors_or_higher : 
  let p := censusData Gender.Male EducationLevel.CollegeBachelor + 
           censusData Gender.Female EducationLevel.CollegeBachelor +
           censusData Gender.Male EducationLevel.MastersDegree + 
           censusData Gender.Female EducationLevel.MastersDegree +
           censusData Gender.Male EducationLevel.DoctoralDegree + 
           censusData Gender.Female EducationLevel.DoctoralDegree
  2 * p * (1 - p) + 2 * p * p = 0.6 := by sorry

/-- Theorem 3: Relationship between average years of education for male and female residents -/
theorem male_education_greater_than_female :
  let male_avg := 0 * censusData Gender.Male EducationLevel.NoSchooling +
                  6 * censusData Gender.Male EducationLevel.PrimarySchool +
                  9 * censusData Gender.Male EducationLevel.JuniorHighSchool +
                  12 * censusData Gender.Male EducationLevel.HighSchool +
                  16 * (censusData Gender.Male EducationLevel.CollegeAssociate +
                        censusData Gender.Male EducationLevel.CollegeBachelor +
                        censusData Gender.Male EducationLevel.MastersDegree +
                        censusData Gender.Male EducationLevel.DoctoralDegree)
  let female_avg := 0 * censusData Gender.Female EducationLevel.NoSchooling +
                    6 * censusData Gender.Female EducationLevel.PrimarySchool +
                    9 * censusData Gender.Female EducationLevel.JuniorHighSchool +
                    12 * censusData Gender.Female EducationLevel.HighSchool +
                    16 * (censusData Gender.Female EducationLevel.CollegeAssociate +
                          censusData Gender.Female EducationLevel.CollegeBachelor +
                          censusData Gender.Female EducationLevel.MastersDegree +
                          censusData Gender.Female EducationLevel.DoctoralDegree)
  male_avg > female_avg := by sorry

end NUMINAMATH_CALUDE_probability_masters_degree_expected_value_bachelors_or_higher_male_education_greater_than_female_l1395_139592


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l1395_139517

theorem sqrt_sum_inequality (a b c d e : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) :
  Real.sqrt (a / (b + c + d + e)) +
  Real.sqrt (b / (a + c + d + e)) +
  Real.sqrt (c / (a + b + d + e)) +
  Real.sqrt (d / (a + b + c + e)) +
  Real.sqrt (e / (a + b + c + d)) > 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l1395_139517


namespace NUMINAMATH_CALUDE_negation_equivalence_l1395_139595

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, x₀ > 0 ∧ Real.log x₀ = x₀ - 1) ↔ 
  (∀ x : ℝ, x > 0 → Real.log x ≠ x - 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1395_139595


namespace NUMINAMATH_CALUDE_find_divisor_l1395_139525

theorem find_divisor : ∃ d : ℕ, d > 1 ∧ (1077 + 4) % d = 0 ∧ d = 1081 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l1395_139525


namespace NUMINAMATH_CALUDE_evaluate_complex_exponential_l1395_139576

theorem evaluate_complex_exponential : (3^2)^(3^(3^2)) = 9^19683 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_complex_exponential_l1395_139576


namespace NUMINAMATH_CALUDE_problem_statement_l1395_139598

theorem problem_statement : (-3)^7 / 3^5 + 5^5 - 8^2 = 3052 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1395_139598


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1395_139568

theorem quadratic_equation_solution : 
  ∃ x₁ x₂ : ℝ, x₁ = 2 ∧ x₂ = (1/2 : ℝ) ∧ 
  (∀ x : ℝ, 2 * x^2 - 5 * x + 2 = 0 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1395_139568


namespace NUMINAMATH_CALUDE_females_with_advanced_degrees_l1395_139596

theorem females_with_advanced_degrees 
  (total_employees : ℕ)
  (female_employees : ℕ)
  (advanced_degree_employees : ℕ)
  (males_with_college_only : ℕ)
  (h1 : total_employees = 148)
  (h2 : female_employees = 92)
  (h3 : advanced_degree_employees = 78)
  (h4 : males_with_college_only = 31) :
  total_employees - female_employees - males_with_college_only - 
  (advanced_degree_employees - (total_employees - female_employees - males_with_college_only)) = 53 := by
  sorry

end NUMINAMATH_CALUDE_females_with_advanced_degrees_l1395_139596


namespace NUMINAMATH_CALUDE_expression_evaluation_l1395_139589

theorem expression_evaluation : 
  -|-(3 + 3/5)| - (-(2 + 2/5)) + 4/5 = -2/5 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1395_139589


namespace NUMINAMATH_CALUDE_sin_square_equation_solution_l1395_139557

theorem sin_square_equation_solution (x : ℝ) :
  (Real.sin (3 * x))^2 + (Real.sin (4 * x))^2 = (Real.sin (5 * x))^2 + (Real.sin (6 * x))^2 →
  (∃ l : ℤ, x = l * π / 2) ∨ (∃ n : ℤ, x = n * π / 9) :=
by sorry

end NUMINAMATH_CALUDE_sin_square_equation_solution_l1395_139557


namespace NUMINAMATH_CALUDE_number_above_345_l1395_139593

/-- Represents the triangular array structure -/
structure TriangularArray where
  /-- Returns the number of elements in the k-th row -/
  elementsInRow : ℕ → ℕ
  /-- Returns the sum of elements up to and including the k-th row -/
  sumUpToRow : ℕ → ℕ
  /-- First row has one element -/
  first_row_one : elementsInRow 1 = 1
  /-- Each row has three more elements than the previous -/
  row_increment : ∀ k, elementsInRow (k + 1) = elementsInRow k + 3
  /-- Sum formula for elements up to k-th row -/
  sum_formula : ∀ k, sumUpToRow k = k * (3 * k - 1) / 2

theorem number_above_345 (arr : TriangularArray) :
  ∃ (row : ℕ) (pos : ℕ),
    arr.sumUpToRow (row - 1) < 345 ∧
    345 ≤ arr.sumUpToRow row ∧
    pos = 345 - arr.sumUpToRow (row - 1) ∧
    arr.sumUpToRow (row - 2) + pos = 308 :=
  sorry

end NUMINAMATH_CALUDE_number_above_345_l1395_139593


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l1395_139502

theorem simplify_and_rationalize (x : ℝ) : 
  1 / (2 + 1 / (Real.sqrt 5 + 2)) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l1395_139502


namespace NUMINAMATH_CALUDE_weight_replacement_l1395_139532

/-- Given 5 people, if replacing one person with a new person weighing 70 kg
    increases the average weight by 4 kg, then the replaced person weighed 50 kg. -/
theorem weight_replacement (initial_count : ℕ) (weight_increase : ℝ) (new_weight : ℝ) :
  initial_count = 5 →
  weight_increase = 4 →
  new_weight = 70 →
  (initial_count : ℝ) * weight_increase = new_weight - 50 := by
  sorry

end NUMINAMATH_CALUDE_weight_replacement_l1395_139532


namespace NUMINAMATH_CALUDE_min_sum_with_constraint_l1395_139506

theorem min_sum_with_constraint (x y z : ℝ) (h : (4 / x) + (2 / y) + (1 / z) = 1) :
  x + 8 * y + 4 * z ≥ 64 ∧ ∃ (x₀ y₀ z₀ : ℝ), (4 / x₀) + (2 / y₀) + (1 / z₀) = 1 ∧ x₀ + 8 * y₀ + 4 * z₀ = 64 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_with_constraint_l1395_139506


namespace NUMINAMATH_CALUDE_unique_function_theorem_l1395_139550

def S : Set ℕ := {n | 1 ≤ n ∧ n ≤ 10^10}

def is_valid_function (f : ℕ → ℕ) : Prop :=
  (∀ x ∈ S, f x ∈ S) ∧
  (∀ x ∈ S, f (x + 1) ≡ f (f x) + 1 [MOD 10^10]) ∧
  (f (10^10 + 1) = f 1)

theorem unique_function_theorem :
  ∀ f : ℕ → ℕ, is_valid_function f →
    ∀ x ∈ S, f x ≡ x [MOD 10^10] :=
by sorry

end NUMINAMATH_CALUDE_unique_function_theorem_l1395_139550


namespace NUMINAMATH_CALUDE_pairs_sold_proof_l1395_139591

def total_amount : ℝ := 588
def average_price : ℝ := 9.8

theorem pairs_sold_proof :
  total_amount / average_price = 60 :=
by sorry

end NUMINAMATH_CALUDE_pairs_sold_proof_l1395_139591


namespace NUMINAMATH_CALUDE_circle_equation_proof_l1395_139507

/-- Given a parabola and a hyperbola, prove the equation of a circle with specific properties -/
theorem circle_equation_proof (x y : ℝ) : 
  (∃ (p : ℝ × ℝ), y^2 = 20*x ∧ p = (5, 0)) → -- Parabola equation and its focus
  (x^2/9 - y^2/16 = 1) →                    -- Hyperbola equation
  (∃ (c : ℝ × ℝ) (r : ℝ),                   -- Circle properties
    c = (5, 0) ∧                            -- Circle center at parabola focus
    r = 4 ∧                                 -- Circle radius
    (∀ (x' y' : ℝ), (y' = 4*x'/3 ∨ y' = -4*x'/3) →  -- Asymptotes of hyperbola
      (x' - c.1)^2 + (y' - c.2)^2 = r^2)) →  -- Circle tangent to asymptotes
  (x - 5)^2 + y^2 = 16                       -- Equation of the circle
  := by sorry

end NUMINAMATH_CALUDE_circle_equation_proof_l1395_139507


namespace NUMINAMATH_CALUDE_average_geometric_sequence_l1395_139587

theorem average_geometric_sequence (y : ℝ) : 
  let sequence := [0, 3*y, 9*y, 27*y, 81*y]
  (sequence.sum / sequence.length : ℝ) = 24*y := by
  sorry

end NUMINAMATH_CALUDE_average_geometric_sequence_l1395_139587


namespace NUMINAMATH_CALUDE_bob_and_bill_transfer_probability_l1395_139570

theorem bob_and_bill_transfer_probability (total_students : ℕ) (transfer_students : ℕ) (num_classes : ℕ) :
  total_students = 32 →
  transfer_students = 2 →
  num_classes = 2 →
  (1 : ℚ) / (Nat.choose total_students transfer_students * num_classes) = 1 / 992 :=
by sorry

end NUMINAMATH_CALUDE_bob_and_bill_transfer_probability_l1395_139570


namespace NUMINAMATH_CALUDE_price_difference_l1395_139563

def coupon_A (P : ℝ) : ℝ := 0.20 * P
def coupon_B : ℝ := 40
def coupon_C (P : ℝ) : ℝ := 0.30 * (P - 150)

def valid_price (P : ℝ) : Prop :=
  P > 150 ∧ coupon_A P ≥ max coupon_B (coupon_C P)

theorem price_difference : 
  ∃ (x y : ℝ), valid_price x ∧ valid_price y ∧
  (∀ P, valid_price P → x ≤ P ∧ P ≤ y) ∧
  y - x = 250 :=
sorry

end NUMINAMATH_CALUDE_price_difference_l1395_139563


namespace NUMINAMATH_CALUDE_scientific_notation_110000_l1395_139539

theorem scientific_notation_110000 : 
  110000 = 1.1 * (10 : ℝ) ^ 5 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_110000_l1395_139539


namespace NUMINAMATH_CALUDE_triangle_trig_ratio_l1395_139599

theorem triangle_trig_ratio (A B C : ℝ) (h : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) 
  (ratio : Real.sin A / Real.sin B = 2/3 ∧ Real.sin B / Real.sin C = 3/4) : 
  Real.cos C = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_trig_ratio_l1395_139599


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1395_139571

theorem polynomial_coefficient_sum (a b c d : ℤ) : 
  (∀ x : ℝ, (x^2 + a*x + b) * (x^2 + c*x + d) = x^4 + 2*x^3 - 3*x^2 + 7*x - 6) →
  a + b + c + d = 7 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1395_139571


namespace NUMINAMATH_CALUDE_shell_collection_ratio_l1395_139545

theorem shell_collection_ratio :
  ∀ (laurie_shells ben_shells alan_shells : ℕ),
    laurie_shells = 36 →
    ben_shells = laurie_shells / 3 →
    alan_shells = 48 →
    alan_shells / ben_shells = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_shell_collection_ratio_l1395_139545


namespace NUMINAMATH_CALUDE_game_ends_after_33_rounds_l1395_139505

/-- Represents a player in the token redistribution game -/
inductive Player
| P
| Q
| R

/-- State of the game, tracking token counts for each player and the number of rounds played -/
structure GameState where
  tokens : Player → ℕ
  rounds : ℕ

/-- Determines if the game has ended (any player has 0 tokens) -/
def gameEnded (state : GameState) : Prop :=
  ∃ p : Player, state.tokens p = 0

/-- Simulates one round of the game -/
def playRound (state : GameState) : GameState :=
  sorry

/-- The initial state of the game -/
def initialState : GameState :=
  { tokens := λ p => match p with
    | Player.P => 12
    | Player.Q => 10
    | Player.R => 8,
    rounds := 0 }

/-- The main theorem: the game ends after 33 rounds -/
theorem game_ends_after_33_rounds :
  ∃ finalState : GameState,
    finalState.rounds = 33 ∧
    gameEnded finalState ∧
    (∀ n : ℕ, n < 33 → ¬gameEnded ((playRound^[n]) initialState)) :=
  sorry

end NUMINAMATH_CALUDE_game_ends_after_33_rounds_l1395_139505


namespace NUMINAMATH_CALUDE_sum_even_integers_l1395_139582

theorem sum_even_integers (x y : ℕ) : 
  (x = (40 + 60) * ((60 - 40) / 2 + 1) / 2) →  -- Sum formula for arithmetic sequence
  (y = (60 - 40) / 2 + 1) →                    -- Number of terms in arithmetic sequence
  (x + y = 561) → 
  (x = 550) := by
sorry

end NUMINAMATH_CALUDE_sum_even_integers_l1395_139582


namespace NUMINAMATH_CALUDE_infinite_solutions_l1395_139552

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The first equation: 3x - 4y = 10 -/
def equation1 (p : Point) : Prop := 3 * p.x - 4 * p.y = 10

/-- The second equation: 9x - 12y = 30 -/
def equation2 (p : Point) : Prop := 9 * p.x - 12 * p.y = 30

/-- A solution satisfies both equations -/
def is_solution (p : Point) : Prop := equation1 p ∧ equation2 p

/-- The set of all solutions -/
def solution_set : Set Point := {p | is_solution p}

/-- The theorem stating that there are infinitely many solutions -/
theorem infinite_solutions : Set.Infinite solution_set := by sorry

end NUMINAMATH_CALUDE_infinite_solutions_l1395_139552


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_three_l1395_139535

theorem smallest_four_digit_multiple_of_three :
  ∃ n : ℕ, n = 1002 ∧ 
    (∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 3 = 0 → n ≤ m) ∧
    1000 ≤ n ∧ n < 10000 ∧ n % 3 = 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_three_l1395_139535


namespace NUMINAMATH_CALUDE_find_number_multiplied_by_9999_l1395_139560

theorem find_number_multiplied_by_9999 :
  ∃! x : ℤ, x * 9999 = 724807415 :=
by
  sorry

end NUMINAMATH_CALUDE_find_number_multiplied_by_9999_l1395_139560


namespace NUMINAMATH_CALUDE_seashells_given_to_jason_l1395_139584

theorem seashells_given_to_jason (initial_seashells : ℕ) (remaining_seashells : ℕ) 
  (h1 : initial_seashells = 66) (h2 : remaining_seashells = 14) : 
  initial_seashells - remaining_seashells = 52 := by
  sorry

end NUMINAMATH_CALUDE_seashells_given_to_jason_l1395_139584


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l1395_139540

def U : Set Int := {-2, -1, 0, 1, 2}

def A : Set Int := {x | x ∈ U ∧ x^2 + x - 2 < 0}

theorem complement_of_A_in_U :
  {x | x ∈ U ∧ x ∉ A} = {-2, 1, 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l1395_139540


namespace NUMINAMATH_CALUDE_remainder_theorem_l1395_139543

theorem remainder_theorem (n : ℤ) (h : ∃ k : ℤ, n = 100 * k - 1) :
  (n^2 + 2*n + 3 + n^3) % 100 = 1 := by
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1395_139543


namespace NUMINAMATH_CALUDE_three_digit_rotations_divisibility_l1395_139564

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  h_hundreds : hundreds ≥ 1 ∧ hundreds ≤ 9
  h_tens : tens ≥ 0 ∧ tens ≤ 9
  h_ones : ones ≥ 0 ∧ ones ≤ 9

/-- Converts a ThreeDigitNumber to its numeric value -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- Rotates the digits of a ThreeDigitNumber once to the left -/
def ThreeDigitNumber.rotateLeft (n : ThreeDigitNumber) : ThreeDigitNumber where
  hundreds := n.tens
  tens := n.ones
  ones := n.hundreds
  h_hundreds := by sorry
  h_tens := by sorry
  h_ones := by sorry

/-- Rotates the digits of a ThreeDigitNumber twice to the left -/
def ThreeDigitNumber.rotateLeftTwice (n : ThreeDigitNumber) : ThreeDigitNumber where
  hundreds := n.ones
  tens := n.hundreds
  ones := n.tens
  h_hundreds := by sorry
  h_tens := by sorry
  h_ones := by sorry

theorem three_digit_rotations_divisibility (n : ThreeDigitNumber) :
  27 ∣ n.toNat → 27 ∣ (n.rotateLeft).toNat ∧ 27 ∣ (n.rotateLeftTwice).toNat := by
  sorry

end NUMINAMATH_CALUDE_three_digit_rotations_divisibility_l1395_139564


namespace NUMINAMATH_CALUDE_evans_books_multiple_l1395_139577

/-- Proves the multiple of Evan's current books in 5 years --/
theorem evans_books_multiple (books_two_years_ago : ℕ) (books_decrease : ℕ) (books_in_five_years : ℕ) : 
  books_two_years_ago = 200 →
  books_decrease = 40 →
  books_in_five_years = 860 →
  ∃ (current_books : ℕ) (multiple : ℕ),
    current_books = books_two_years_ago - books_decrease ∧
    books_in_five_years = multiple * current_books + 60 ∧
    multiple = 5 := by
  sorry

#check evans_books_multiple

end NUMINAMATH_CALUDE_evans_books_multiple_l1395_139577


namespace NUMINAMATH_CALUDE_smallest_block_size_l1395_139548

/-- Given a rectangular block of dimensions l × m × n formed by N unit cubes,
    where (l - 1) × (m - 1) × (n - 1) = 143, the smallest possible value of N is 336. -/
theorem smallest_block_size (l m n : ℕ) (h : (l - 1) * (m - 1) * (n - 1) = 143) :
  ∃ (N : ℕ), N = l * m * n ∧ N = 336 ∧ ∀ (l' m' n' : ℕ), 
    ((l' - 1) * (m' - 1) * (n' - 1) = 143) → l' * m' * n' ≥ N :=
by sorry

end NUMINAMATH_CALUDE_smallest_block_size_l1395_139548


namespace NUMINAMATH_CALUDE_recurrence_sequence_property_l1395_139566

/-- An integer sequence satisfying the given recurrence relation -/
def RecurrenceSequence (m : ℤ) (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a (n + 2) = a (n + 1) - m * a n

/-- The main theorem -/
theorem recurrence_sequence_property
  (m : ℤ) (a : ℕ → ℤ) (h_m : |m| ≥ 2)
  (h_nonzero : ¬(a 1 = 0 ∧ a 2 = 0))
  (h_recurrence : RecurrenceSequence m a)
  (r s : ℕ) (h_rs : r > s ∧ s ≥ 2)
  (h_equal : a r = a 1 ∧ a s = a 1) :
  r - s ≥ |m| := by sorry

end NUMINAMATH_CALUDE_recurrence_sequence_property_l1395_139566


namespace NUMINAMATH_CALUDE_ellipse_m_range_l1395_139549

-- Define the equation
def is_ellipse (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (m + 2) - y^2 / (m + 1) = 1

-- State the theorem
theorem ellipse_m_range :
  ∀ m : ℝ, is_ellipse m → m ∈ Set.Ioo (-2 : ℝ) (-3/2) ∪ Set.Ioo (-3/2 : ℝ) (-1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_m_range_l1395_139549


namespace NUMINAMATH_CALUDE_glenburgh_parade_squad_l1395_139521

theorem glenburgh_parade_squad (m : ℕ) : 
  (∃ k : ℕ, 20 * m = 28 * k + 6) → 
  20 * m < 1200 →
  (∀ n : ℕ, (∃ j : ℕ, 20 * n = 28 * j + 6) → 20 * n < 1200 → 20 * n ≤ 20 * m) →
  20 * m = 1160 := by
sorry

end NUMINAMATH_CALUDE_glenburgh_parade_squad_l1395_139521


namespace NUMINAMATH_CALUDE_inverse_proportion_point_difference_l1395_139578

/-- 
Given two points A(x₁, y₁) and B(x₂, y₂) on the graph of y = -2/x,
where x₁ < 0 < x₂, prove that y₁ - y₂ > 0.
-/
theorem inverse_proportion_point_difference (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : y₁ = -2 / x₁)
  (h2 : y₂ = -2 / x₂)
  (h3 : x₁ < 0)
  (h4 : 0 < x₂) : 
  y₁ - y₂ > 0 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_point_difference_l1395_139578


namespace NUMINAMATH_CALUDE_import_value_calculation_l1395_139551

theorem import_value_calculation (export_value import_value : ℝ) : 
  export_value = 8.07 ∧ 
  export_value = 1.5 * import_value + 1.11 → 
sorry

end NUMINAMATH_CALUDE_import_value_calculation_l1395_139551


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1395_139586

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  (6 * x₁^2 + 5 * x₁ - 4 = 0) → 
  (6 * x₂^2 + 5 * x₂ - 4 = 0) → 
  (x₁ ≠ x₂) →
  (x₁^2 + x₂^2 = 73/36) := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1395_139586


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l1395_139533

/-- Given a geometric sequence with first term 1000 and sixth term 125,
    prove that the third term is equal to 301. -/
theorem geometric_sequence_third_term :
  ∀ (a : ℝ) (r : ℝ),
    a = 1000 →
    a * r^5 = 125 →
    a * r^2 = 301 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l1395_139533


namespace NUMINAMATH_CALUDE_searchlight_dark_time_l1395_139558

/-- The number of revolutions per minute for the searchlight -/
def revolutions_per_minute : ℝ := 4

/-- The probability of staying in the dark for at least a certain number of seconds -/
def probability : ℝ := 0.6666666666666667

/-- The time in seconds for which the probability applies -/
def dark_time : ℝ := 10

theorem searchlight_dark_time :
  revolutions_per_minute = 4 ∧ probability = 0.6666666666666667 →
  dark_time = 10 := by sorry

end NUMINAMATH_CALUDE_searchlight_dark_time_l1395_139558


namespace NUMINAMATH_CALUDE_train_length_calculation_l1395_139516

/-- The length of a train given jogger and train speeds, initial distance, and time to pass. -/
theorem train_length_calculation (jogger_speed : ℝ) (train_speed : ℝ) (initial_distance : ℝ) (time_to_pass : ℝ) : 
  jogger_speed = 9 * (1000 / 3600) →
  train_speed = 45 * (1000 / 3600) →
  initial_distance = 240 →
  time_to_pass = 39 →
  (train_speed - jogger_speed) * time_to_pass - initial_distance = 150 := by
sorry

end NUMINAMATH_CALUDE_train_length_calculation_l1395_139516


namespace NUMINAMATH_CALUDE_division_remainder_problem_l1395_139554

theorem division_remainder_problem (dividend : Nat) (divisor : Nat) : 
  Prime dividend → Prime divisor → dividend = divisor * 7 + 1054 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l1395_139554


namespace NUMINAMATH_CALUDE_opposite_of_negative_fraction_l1395_139573

theorem opposite_of_negative_fraction (n : ℕ) (hn : n ≠ 0) :
  (-(1 : ℚ) / n) + (1 : ℚ) / n = 0 :=
by sorry

#check opposite_of_negative_fraction

end NUMINAMATH_CALUDE_opposite_of_negative_fraction_l1395_139573


namespace NUMINAMATH_CALUDE_ice_block_volume_l1395_139522

theorem ice_block_volume (V : ℝ) : 
  V > 0 →
  (8/35 : ℝ) * V = 0.15 →
  V = 0.65625 := by sorry

end NUMINAMATH_CALUDE_ice_block_volume_l1395_139522


namespace NUMINAMATH_CALUDE_zach_needs_six_dollars_l1395_139500

/-- The amount of money Zach needs to earn to buy the bike -/
def money_needed (bike_cost allowance lawn_pay babysit_rate current_savings babysit_hours : ℕ) : ℕ :=
  let total_earnings := allowance + lawn_pay + babysit_rate * babysit_hours
  let total_savings := current_savings + total_earnings
  if total_savings ≥ bike_cost then 0
  else bike_cost - total_savings

theorem zach_needs_six_dollars :
  money_needed 100 5 10 7 65 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_zach_needs_six_dollars_l1395_139500


namespace NUMINAMATH_CALUDE_farmer_earnings_proof_l1395_139523

/-- Calculates the farmer's earnings after the market fee -/
def farmer_earnings (potatoes carrots tomatoes : ℕ) 
  (potato_bundle_size potato_bundle_price : ℚ)
  (carrot_bundle_size carrot_bundle_price : ℚ)
  (tomato_price canned_tomato_set_size canned_tomato_set_price : ℚ)
  (market_fee_rate : ℚ) : ℚ :=
  let potato_sales := (potatoes / potato_bundle_size) * potato_bundle_price
  let carrot_sales := (carrots / carrot_bundle_size) * carrot_bundle_price
  let fresh_tomato_sales := (tomatoes / 2) * tomato_price
  let canned_tomato_sales := ((tomatoes / 2) / canned_tomato_set_size) * canned_tomato_set_price
  let total_sales := potato_sales + carrot_sales + fresh_tomato_sales + canned_tomato_sales
  let market_fee := total_sales * market_fee_rate
  total_sales - market_fee

/-- The farmer's earnings after the market fee is $618.45 -/
theorem farmer_earnings_proof :
  farmer_earnings 250 320 480 25 1.9 20 2 1 10 15 0.05 = 618.45 := by
  sorry

end NUMINAMATH_CALUDE_farmer_earnings_proof_l1395_139523


namespace NUMINAMATH_CALUDE_M_equals_N_l1395_139504

def M : Set ℤ := {u | ∃ m n l : ℤ, u = 12*m + 8*n + 4*l}
def N : Set ℤ := {u | ∃ p q r : ℤ, u = 20*p + 16*q + 12*r}

theorem M_equals_N : M = N := by sorry

end NUMINAMATH_CALUDE_M_equals_N_l1395_139504


namespace NUMINAMATH_CALUDE_fifteenth_student_age_l1395_139509

theorem fifteenth_student_age 
  (total_students : Nat) 
  (group1_students : Nat) 
  (group2_students : Nat) 
  (total_average_age : ℝ) 
  (group1_average_age : ℝ) 
  (group2_average_age : ℝ) :
  total_students = 15 →
  group1_students = 5 →
  group2_students = 9 →
  total_average_age = 15 →
  group1_average_age = 14 →
  group2_average_age = 16 →
  (total_students * total_average_age) - 
    (group1_students * group1_average_age + group2_students * group2_average_age) = 11 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_student_age_l1395_139509


namespace NUMINAMATH_CALUDE_sqrt_81_div_3_l1395_139538

theorem sqrt_81_div_3 : Real.sqrt 81 / 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_81_div_3_l1395_139538


namespace NUMINAMATH_CALUDE_triangle_isosceles_or_right_angled_l1395_139503

-- Define a triangle with angles α, β, and γ
structure Triangle where
  α : Real
  β : Real
  γ : Real
  sum_angles : α + β + γ = Real.pi

-- Define the theorem
theorem triangle_isosceles_or_right_angled (t : Triangle) :
  Real.tan t.β * Real.sin t.γ * Real.sin t.γ = Real.tan t.γ * Real.sin t.β * Real.sin t.β →
  (t.β = t.γ ∨ t.β + t.γ = Real.pi / 2) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_isosceles_or_right_angled_l1395_139503


namespace NUMINAMATH_CALUDE_inscribed_circle_ratio_l1395_139510

-- Define the quadrilateral ABCD
variable (A B C D : Point)

-- Define the inscribed circle P
variable (P : Point)

-- Define that P is the center of the inscribed circle
def is_inscribed_center (P : Point) (A B C D : Point) : Prop := sorry

-- Define the distance function
def distance (P Q : Point) : ℝ := sorry

-- State the theorem
theorem inscribed_circle_ratio 
  (h : is_inscribed_center P A B C D) :
  (distance P A)^2 / (distance P C)^2 = 
  (distance A B * distance A D) / (distance B C * distance C D) := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_ratio_l1395_139510


namespace NUMINAMATH_CALUDE_baymax_testing_system_l1395_139572

theorem baymax_testing_system (x y : ℕ) : 
  (200 * y = x + 18 ∧ 180 * y = x - 42) ↔ 
  (∀ (z : ℕ), z = 200 → z * y = x + 18) ∧ 
  (∀ (w : ℕ), w = 180 → w * y + 42 = x) :=
sorry

end NUMINAMATH_CALUDE_baymax_testing_system_l1395_139572


namespace NUMINAMATH_CALUDE_unique_exam_scores_l1395_139511

def is_valid_score_set (scores : List Nat) : Prop :=
  scores.length = 5 ∧
  scores.all (λ x => x % 2 = 1 ∧ x < 100) ∧
  scores.Nodup ∧
  scores.sum / scores.length = 80 ∧
  [95, 85, 75, 65].all (λ x => x ∈ scores)

theorem unique_exam_scores :
  ∃! scores : List Nat, is_valid_score_set scores ∧ scores = [95, 85, 79, 75, 65] := by
  sorry

end NUMINAMATH_CALUDE_unique_exam_scores_l1395_139511


namespace NUMINAMATH_CALUDE_theta_half_quadrants_l1395_139542

theorem theta_half_quadrants (θ : Real) 
  (h1 : |Real.cos θ| = Real.cos θ) 
  (h2 : |Real.tan θ| = -Real.tan θ) : 
  (∃ (k : ℤ), 2 * k * Real.pi + Real.pi / 2 < θ / 2 ∧ θ / 2 ≤ 2 * k * Real.pi + Real.pi) ∨ 
  (∃ (k : ℤ), 2 * k * Real.pi + 3 * Real.pi / 2 < θ / 2 ∧ θ / 2 ≤ 2 * k * Real.pi + 2 * Real.pi) ∨
  (∃ (k : ℤ), θ / 2 = k * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_theta_half_quadrants_l1395_139542


namespace NUMINAMATH_CALUDE_uniform_price_is_200_l1395_139534

/-- Represents the agreement between a man and his servant --/
structure Agreement where
  full_year_salary : ℕ
  service_duration : ℕ
  actual_duration : ℕ
  partial_payment : ℕ

/-- Calculates the price of the uniform given the agreement details --/
def uniform_price (a : Agreement) : ℕ :=
  let expected_payment := a.full_year_salary * a.actual_duration / a.service_duration
  expected_payment - a.partial_payment

/-- Theorem stating that the price of the uniform is 200 given the problem conditions --/
theorem uniform_price_is_200 : 
  uniform_price { full_year_salary := 800
                , service_duration := 12
                , actual_duration := 9
                , partial_payment := 400 } = 200 := by
  sorry

end NUMINAMATH_CALUDE_uniform_price_is_200_l1395_139534


namespace NUMINAMATH_CALUDE_theta_range_l1395_139585

theorem theta_range (θ : Real) (h1 : θ ∈ Set.Icc 0 (2 * Real.pi)) 
  (h2 : Real.cos θ ^ 5 - Real.sin θ ^ 5 < 7 * (Real.sin θ ^ 3 - Real.cos θ ^ 3)) :
  θ ∈ Set.Ioo (Real.pi / 4) (5 * Real.pi / 4) := by
  sorry

end NUMINAMATH_CALUDE_theta_range_l1395_139585


namespace NUMINAMATH_CALUDE_right_triangle_sin_cos_l1395_139512

/-- In a right triangle XYZ with ∠Y = 90°, hypotenuse XZ = 15, and leg XY = 9, sin X = 4/5 and cos X = 3/5 -/
theorem right_triangle_sin_cos (X Y Z : ℝ) (h1 : X^2 + Y^2 = Z^2) (h2 : Z = 15) (h3 : X = 9) :
  Real.sin (Real.arccos (X / Z)) = 4/5 ∧ Real.cos (Real.arccos (X / Z)) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sin_cos_l1395_139512


namespace NUMINAMATH_CALUDE_implicit_function_derivative_l1395_139520

/-- Given an implicitly defined function y²(x) + x² - 1 = 0,
    prove that the derivative of y with respect to x is -x / y(x) -/
theorem implicit_function_derivative 
  (y : ℝ → ℝ) 
  (h : ∀ x, y x ^ 2 + x ^ 2 - 1 = 0) :
  ∀ x, HasDerivAt y (-(x / y x)) x :=
sorry

end NUMINAMATH_CALUDE_implicit_function_derivative_l1395_139520


namespace NUMINAMATH_CALUDE_trainers_average_age_l1395_139508

/-- The average age of trainers in a sports club --/
theorem trainers_average_age
  (total_members : ℕ)
  (overall_average : ℚ)
  (num_women : ℕ)
  (num_men : ℕ)
  (num_trainers : ℕ)
  (women_average : ℚ)
  (men_average : ℚ)
  (h_total : total_members = 70)
  (h_overall : overall_average = 23)
  (h_women : num_women = 30)
  (h_men : num_men = 25)
  (h_trainers : num_trainers = 15)
  (h_women_avg : women_average = 20)
  (h_men_avg : men_average = 25)
  (h_sum : total_members = num_women + num_men + num_trainers) :
  (total_members * overall_average - num_women * women_average - num_men * men_average) / num_trainers = 25 + 2/3 :=
by sorry

end NUMINAMATH_CALUDE_trainers_average_age_l1395_139508


namespace NUMINAMATH_CALUDE_star_value_l1395_139567

/-- The operation * for non-zero integers -/
def star (a b : ℤ) : ℚ := (a : ℚ)⁻¹ + (b : ℚ)⁻¹

/-- Theorem: If a + b = 10 and ab = 24, then a * b = 5/12 -/
theorem star_value (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (sum_eq : a + b = 10) (prod_eq : a * b = 24) : 
  star a b = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_star_value_l1395_139567


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l1395_139513

theorem equal_roots_quadratic (a : ℝ) : 
  (∃ x : ℝ, a * x^2 - 4 * x + 3 = 0 ∧ 
   ∀ y : ℝ, a * y^2 - 4 * y + 3 = 0 → y = x) → 
  a = 4/3 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l1395_139513


namespace NUMINAMATH_CALUDE_circle_radius_proof_l1395_139528

theorem circle_radius_proof (A₁ A₂ : ℝ) (h1 : A₁ > 0) (h2 : A₂ > 0) : 
  (A₁ + A₂ = 16 * Real.pi) →
  (2 * A₁ = 16 * Real.pi - A₁) →
  (∃ (r : ℝ), r > 0 ∧ A₁ = Real.pi * r^2 ∧ r = 4 * Real.sqrt 3 / 3) := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_proof_l1395_139528


namespace NUMINAMATH_CALUDE_sally_received_quarters_l1395_139536

/-- The number of quarters Sally initially had -/
def initial_quarters : ℕ := 760

/-- The number of quarters Sally now has -/
def final_quarters : ℕ := 1178

/-- The number of quarters Sally received -/
def received_quarters : ℕ := final_quarters - initial_quarters

theorem sally_received_quarters : received_quarters = 418 := by
  sorry

end NUMINAMATH_CALUDE_sally_received_quarters_l1395_139536


namespace NUMINAMATH_CALUDE_race_finish_orders_l1395_139561

/-- The number of possible finish orders for a race with 4 participants and no ties -/
def finish_orders : ℕ := 24

/-- The number of participants in the race -/
def num_participants : ℕ := 4

/-- Theorem: The number of possible finish orders for a race with 4 participants and no ties is 24 -/
theorem race_finish_orders : 
  finish_orders = Nat.factorial num_participants :=
sorry

end NUMINAMATH_CALUDE_race_finish_orders_l1395_139561


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1395_139555

-- Define the sets M and N
def M : Set ℝ := {x | -1 < x ∧ x < 1}
def N : Set ℝ := {x | x / (x - 1) ≤ 0}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | 0 ≤ x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1395_139555


namespace NUMINAMATH_CALUDE_train_passenger_problem_l1395_139519

theorem train_passenger_problem (P : ℚ) : 
  (((P * (2/3) + 280) * (1/2) + 12) = 242) → P = 270 := by
  sorry

end NUMINAMATH_CALUDE_train_passenger_problem_l1395_139519


namespace NUMINAMATH_CALUDE_ellipse_slope_product_l1395_139526

/-- Given an ellipse with equation x^2/25 + y^2/9 = 1, 
    this theorem states that for any point P on the ellipse 
    (distinct from the endpoints of the major axis), 
    the product of the slopes of the lines connecting P 
    to the endpoints of the major axis is -9/25. -/
theorem ellipse_slope_product : 
  ∀ (x y : ℝ), 
  x^2/25 + y^2/9 = 1 →  -- P is on the ellipse
  x ≠ 5 →              -- P is not the right endpoint
  x ≠ -5 →             -- P is not the left endpoint
  ∃ (m₁ m₂ : ℝ),       -- slopes exist
  (m₁ = y / (x - 5) ∧ m₂ = y / (x + 5)) ∧  -- definition of slopes
  m₁ * m₂ = -9/25 :=   -- product of slopes
by sorry


end NUMINAMATH_CALUDE_ellipse_slope_product_l1395_139526


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1395_139559

/-- An arithmetic sequence with first term 2 and sum of first 3 terms equal to 12 has its 6th term equal to 12 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  a 1 = 2 →                            -- first term is 2
  a 1 + a 2 + a 3 = 12 →                -- sum of first 3 terms is 12
  a 6 = 12 := by                        -- 6th term is 12
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1395_139559


namespace NUMINAMATH_CALUDE_remainder_problem_l1395_139594

theorem remainder_problem (k : ℕ) (h1 : k > 0) (h2 : 90 % (k^2) = 10) : 150 % k = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1395_139594


namespace NUMINAMATH_CALUDE_division_problem_l1395_139531

theorem division_problem (n : ℕ) : 
  n / 18 = 11 ∧ n % 18 = 1 → n = 199 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1395_139531


namespace NUMINAMATH_CALUDE_first_car_speed_l1395_139580

/-- Proves that the speed of the first car is 54 miles per hour given the conditions of the problem -/
theorem first_car_speed (total_distance : ℝ) (second_car_speed : ℝ) (time_difference : ℝ) (total_time : ℝ)
  (h1 : total_distance = 80)
  (h2 : second_car_speed = 60)
  (h3 : time_difference = 1/6)
  (h4 : total_time = 1.5) :
  ∃ (first_car_speed : ℝ), first_car_speed = 54 ∧
    second_car_speed * total_time = first_car_speed * (total_time + time_difference) := by
  sorry


end NUMINAMATH_CALUDE_first_car_speed_l1395_139580


namespace NUMINAMATH_CALUDE_geometric_sequence_b_value_l1395_139518

theorem geometric_sequence_b_value (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 180) (h₂ : a₃ = 64/25) (h₃ : a₂ > 0) 
  (h₄ : ∃ r : ℝ, a₂ = a₁ * r ∧ a₃ = a₂ * r) : a₂ = 21.6 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_b_value_l1395_139518


namespace NUMINAMATH_CALUDE_sqrt_50_plus_sqrt_32_l1395_139514

theorem sqrt_50_plus_sqrt_32 : Real.sqrt 50 + Real.sqrt 32 = 9 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_50_plus_sqrt_32_l1395_139514


namespace NUMINAMATH_CALUDE_fraction_conversions_l1395_139529

theorem fraction_conversions :
  (7 / 9 : ℚ) = 7 / 9 ∧
  (12 / 7 : ℚ) = 12 / 7 ∧
  (3 + 5 / 8 : ℚ) = 29 / 8 ∧
  (6 : ℚ) = 66 / 11 := by
sorry

end NUMINAMATH_CALUDE_fraction_conversions_l1395_139529


namespace NUMINAMATH_CALUDE_probability_is_three_fifths_l1395_139556

/-- The number of red balls in the box -/
def num_red : ℕ := 2

/-- The number of black balls in the box -/
def num_black : ℕ := 3

/-- The total number of balls in the box -/
def total_balls : ℕ := num_red + num_black

/-- The number of ways to choose 2 balls from the box -/
def total_combinations : ℕ := (total_balls * (total_balls - 1)) / 2

/-- The number of ways to choose 1 red ball and 1 black ball -/
def different_color_combinations : ℕ := num_red * num_black

/-- The probability of drawing two balls with different colors -/
def probability_different_colors : ℚ := different_color_combinations / total_combinations

theorem probability_is_three_fifths :
  probability_different_colors = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_three_fifths_l1395_139556


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_attainable_l1395_139544

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  4 * a^4 + 8 * b^4 + 16 * c^4 + 1 / (a * b * c) ≥ 10 := by
  sorry

theorem min_value_attainable :
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧
  4 * a^4 + 8 * b^4 + 16 * c^4 + 1 / (a * b * c) = 10 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_attainable_l1395_139544


namespace NUMINAMATH_CALUDE_solutions_sum_and_product_l1395_139574

theorem solutions_sum_and_product : ∃ (x₁ x₂ : ℝ),
  (x₁ - 6)^2 = 49 ∧
  (x₂ - 6)^2 = 49 ∧
  x₁ + x₂ = 12 ∧
  x₁ * x₂ = -13 :=
by sorry

end NUMINAMATH_CALUDE_solutions_sum_and_product_l1395_139574


namespace NUMINAMATH_CALUDE_parabola_vertex_on_x_axis_l1395_139562

/-- A parabola with equation y = x^2 - 6x + c has its vertex on the x-axis if and only if c = 9 -/
theorem parabola_vertex_on_x_axis (c : ℝ) : 
  (∃ x : ℝ, x^2 - 6*x + c = 0 ∧ ∀ y : ℝ, y^2 - 6*y + c ≥ x^2 - 6*x + c) ↔ c = 9 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_on_x_axis_l1395_139562


namespace NUMINAMATH_CALUDE_cassie_water_refills_l1395_139575

/-- Represents the number of cups of water Cassie aims to drink daily -/
def daily_cups : ℕ := 12

/-- Represents the capacity of Cassie's water bottle in ounces -/
def bottle_capacity : ℕ := 16

/-- Represents the number of ounces in a cup -/
def ounces_per_cup : ℕ := 8

/-- Represents the number of times Cassie needs to refill her water bottle -/
def refills : ℕ := 6

/-- Theorem stating that Cassie needs to refill her water bottle 6 times
    to meet her daily water intake goal -/
theorem cassie_water_refills :
  (daily_cups * ounces_per_cup) / bottle_capacity = refills :=
by sorry

end NUMINAMATH_CALUDE_cassie_water_refills_l1395_139575


namespace NUMINAMATH_CALUDE_power_of_three_mod_ten_l1395_139527

theorem power_of_three_mod_ten : 3^19 % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_mod_ten_l1395_139527


namespace NUMINAMATH_CALUDE_average_temperature_l1395_139537

def temperatures : List ℝ := [55, 59, 60, 57, 64]

theorem average_temperature : 
  (temperatures.sum / temperatures.length : ℝ) = 59.0 := by
  sorry

end NUMINAMATH_CALUDE_average_temperature_l1395_139537


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l1395_139553

/-- A positive integer is a multiple of 3, 5, 7, and 9 if and only if it's a multiple of their LCM -/
axiom multiple_of_3_5_7_9 (n : ℕ) : (3 ∣ n) ∧ (5 ∣ n) ∧ (7 ∣ n) ∧ (9 ∣ n) ↔ 315 ∣ n

/-- The theorem stating that 314 is the unique three-digit positive integer
    that is one less than a multiple of 3, 5, 7, and 9 -/
theorem unique_three_digit_number : 
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ ∃ m : ℕ, n + 1 = 315 * m :=
sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l1395_139553


namespace NUMINAMATH_CALUDE_average_speed_round_trip_l1395_139590

/-- Calculates the average speed of a round trip journey given the distance and times for each leg. -/
theorem average_speed_round_trip 
  (uphill_distance : ℝ) 
  (uphill_time : ℝ) 
  (downhill_time : ℝ) : 
  (2 * uphill_distance) / (uphill_time + downhill_time) = 4 :=
by
  sorry

#check average_speed_round_trip 2 (45/60) (15/60)

end NUMINAMATH_CALUDE_average_speed_round_trip_l1395_139590


namespace NUMINAMATH_CALUDE_coprime_count_2016_l1395_139579

theorem coprime_count_2016 : Nat.totient 2016 = 576 := by
  sorry

end NUMINAMATH_CALUDE_coprime_count_2016_l1395_139579


namespace NUMINAMATH_CALUDE_expression_simplification_l1395_139569

theorem expression_simplification (y : ℝ) :
  y * (4 * y^2 - 3) - 6 * (y^2 - 3 * y + 8) = 4 * y^3 - 6 * y^2 + 15 * y - 48 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1395_139569


namespace NUMINAMATH_CALUDE_payment_calculation_l1395_139581

/-- Calculates the payment per safely delivered bowl -/
def payment_per_bowl (total_bowls : ℕ) (fee : ℚ) (cost_per_damaged : ℚ) 
  (lost_bowls : ℕ) (broken_bowls : ℕ) (total_payment : ℚ) : ℚ :=
  let safely_delivered := total_bowls - lost_bowls - broken_bowls
  (total_payment - fee) / safely_delivered

theorem payment_calculation : 
  let result := payment_per_bowl 638 100 4 12 15 1825
  ∃ (ε : ℚ), ε > 0 ∧ ε < (1/100) ∧ |result - (282/100)| < ε := by
  sorry

end NUMINAMATH_CALUDE_payment_calculation_l1395_139581


namespace NUMINAMATH_CALUDE_investment_rate_proof_l1395_139515

def total_investment : ℝ := 3000
def high_interest_amount : ℝ := 800
def high_interest_rate : ℝ := 0.1
def total_interest : ℝ := 256

def remaining_investment : ℝ := total_investment - high_interest_amount
def high_interest : ℝ := high_interest_amount * high_interest_rate
def remaining_interest : ℝ := total_interest - high_interest

theorem investment_rate_proof :
  remaining_interest / remaining_investment = 0.08 :=
sorry

end NUMINAMATH_CALUDE_investment_rate_proof_l1395_139515


namespace NUMINAMATH_CALUDE_katrina_cookies_l1395_139546

/-- The number of cookies in a dozen -/
def dozen : ℕ := 12

/-- The number of cookies Katrina sold in the morning -/
def morning_sales : ℕ := 3 * dozen

/-- The number of cookies Katrina sold during lunch rush -/
def lunch_sales : ℕ := 57

/-- The number of cookies Katrina sold in the afternoon -/
def afternoon_sales : ℕ := 16

/-- The number of cookies Katrina has left to take home -/
def cookies_left : ℕ := 11

/-- The total number of cookies Katrina had initially -/
def initial_cookies : ℕ := morning_sales + lunch_sales + afternoon_sales + cookies_left

theorem katrina_cookies : initial_cookies = 120 := by sorry

end NUMINAMATH_CALUDE_katrina_cookies_l1395_139546


namespace NUMINAMATH_CALUDE_min_difference_is_one_l1395_139524

/-- Represents the side lengths of a triangle --/
structure TriangleSides where
  xz : ℕ
  yz : ℕ
  xy : ℕ

/-- Checks if the given side lengths form a valid triangle --/
def isValidTriangle (t : TriangleSides) : Prop :=
  t.xz + t.yz > t.xy ∧ t.xz + t.xy > t.yz ∧ t.yz + t.xy > t.xz

/-- Checks if the given side lengths satisfy the problem conditions --/
def satisfiesConditions (t : TriangleSides) : Prop :=
  t.xz + t.yz + t.xy = 3001 ∧ t.xz < t.yz ∧ t.yz ≤ t.xy

theorem min_difference_is_one :
  ∀ t : TriangleSides,
    isValidTriangle t →
    satisfiesConditions t →
    ∀ u : TriangleSides,
      isValidTriangle u →
      satisfiesConditions u →
      t.yz - t.xz ≤ u.yz - u.xz →
      t.yz - t.xz = 1 :=
sorry

end NUMINAMATH_CALUDE_min_difference_is_one_l1395_139524


namespace NUMINAMATH_CALUDE_number_of_students_l1395_139588

theorem number_of_students (total_books : ℕ) : 
  (∃ (x : ℕ), 3 * x + 20 = total_books ∧ 4 * x = total_books + 25) → 
  (∃ (x : ℕ), x = 45 ∧ 3 * x + 20 = total_books ∧ 4 * x = total_books + 25) :=
by sorry

end NUMINAMATH_CALUDE_number_of_students_l1395_139588
