import Mathlib

namespace garden_fence_area_l2405_240590

/-- Given an L-shaped fence and two straight fence sections of 13m and 14m,
    prove that it's possible to create a rectangular area of at least 200 m². -/
theorem garden_fence_area (length : ℝ) (width : ℝ) : 
  length = 13 → width = 17 → length * width ≥ 200 := by
  sorry

end garden_fence_area_l2405_240590


namespace count_pairs_eq_15_l2405_240526

def count_pairs : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => p.1 > 0 ∧ p.2 > 0 ∧ p.1 + p.2 ≤ 6) (Finset.product (Finset.range 6) (Finset.range 6))).card

theorem count_pairs_eq_15 : count_pairs = 15 := by
  sorry

end count_pairs_eq_15_l2405_240526


namespace hyperbola_decreasing_condition_l2405_240555

/-- For a hyperbola y = (1-m)/x, y decreases as x increases when x > 0 if and only if m < 1 -/
theorem hyperbola_decreasing_condition (m : ℝ) : 
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ x₁ < x₂ → (1-m)/x₁ > (1-m)/x₂) ↔ m < 1 :=
by sorry

end hyperbola_decreasing_condition_l2405_240555


namespace dish_price_l2405_240578

/-- The original price of a dish given specific discount and tip conditions --/
def original_price : ℝ → Prop :=
  λ price =>
    let john_payment := price * 0.9 + price * 0.15
    let jane_payment := price * 0.9 + price * 0.9 * 0.15
    john_payment - jane_payment = 0.60

theorem dish_price : ∃ (price : ℝ), original_price price ∧ price = 40 := by
  sorry

end dish_price_l2405_240578


namespace sin_cube_identity_l2405_240553

theorem sin_cube_identity (θ : Real) : 
  Real.sin θ ^ 3 = (-1/4) * Real.sin (3 * θ) + (3/4) * Real.sin θ := by
  sorry

end sin_cube_identity_l2405_240553


namespace arithmetic_sequence_theorem_l2405_240571

/-- An arithmetic sequence with given third and seventeenth terms -/
structure ArithmeticSequence where
  a₃ : ℚ
  a₁₇ : ℚ
  is_arithmetic : ∃ d, a₁₇ = a₃ + 14 * d

/-- The properties we want to prove about this arithmetic sequence -/
def ArithmeticSequenceProperties (seq : ArithmeticSequence) : Prop :=
  ∃ (a₁₀ : ℚ),
    (seq.a₃ = 11/15) ∧
    (seq.a₁₇ = 2/3) ∧
    (a₁₀ = 7/10) ∧
    (seq.a₃ + a₁₀ + seq.a₁₇ = 21/10)

/-- The main theorem stating that our arithmetic sequence has the desired properties -/
theorem arithmetic_sequence_theorem (seq : ArithmeticSequence) 
    (h₁ : seq.a₃ = 11/15) (h₂ : seq.a₁₇ = 2/3) : 
    ArithmeticSequenceProperties seq := by
  sorry


end arithmetic_sequence_theorem_l2405_240571


namespace sqrt_trig_identity_l2405_240554

theorem sqrt_trig_identity : 
  Real.sqrt (2 - Real.sin 2 ^ 2 + Real.cos 4) = -Real.sqrt 3 * Real.cos 2 := by
  sorry

end sqrt_trig_identity_l2405_240554


namespace larger_integer_problem_l2405_240591

theorem larger_integer_problem (x y : ℕ) (h1 : y = 4 * x) (h2 : (x + 6) / y = 1 / 2) : y = 24 := by
  sorry

end larger_integer_problem_l2405_240591


namespace common_difference_is_two_l2405_240507

/-- An arithmetic sequence with sum of first n terms Sₙ -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum of first n terms
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- The common difference of an arithmetic sequence is 2 given the condition -/
theorem common_difference_is_two (seq : ArithmeticSequence) 
    (h : seq.S 2016 / 2016 = seq.S 2015 / 2015 + 1) : 
    ∃ d, d = 2 ∧ ∀ n, seq.a (n + 1) - seq.a n = d :=
  sorry

end common_difference_is_two_l2405_240507


namespace identical_differences_exist_l2405_240518

theorem identical_differences_exist (a : Fin 20 → ℕ) 
  (h_increasing : ∀ i j, i < j → a i < a j) 
  (h_bounded : ∀ i, a i ≤ 70) : 
  ∃ (i₁ j₁ i₂ j₂ i₃ j₃ i₄ j₄ : Fin 20), 
    i₁ < j₁ ∧ i₂ < j₂ ∧ i₃ < j₃ ∧ i₄ < j₄ ∧ 
    (i₁ ≠ i₂ ∨ j₁ ≠ j₂) ∧ (i₁ ≠ i₃ ∨ j₁ ≠ j₃) ∧ (i₁ ≠ i₄ ∨ j₁ ≠ j₄) ∧
    (i₂ ≠ i₃ ∨ j₂ ≠ j₃) ∧ (i₂ ≠ i₄ ∨ j₂ ≠ j₄) ∧ (i₃ ≠ i₄ ∨ j₃ ≠ j₄) ∧
    a j₁ - a i₁ = a j₂ - a i₂ ∧ 
    a j₁ - a i₁ = a j₃ - a i₃ ∧ 
    a j₁ - a i₁ = a j₄ - a i₄ :=
by sorry

end identical_differences_exist_l2405_240518


namespace division_remainder_l2405_240550

theorem division_remainder : ∃ q : ℕ, 1234567 = 321 * q + 264 ∧ 264 < 321 := by
  sorry

end division_remainder_l2405_240550


namespace jack_evening_emails_l2405_240542

/-- The number of emails Jack received in a day -/
def total_emails : ℕ := 10

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 3

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 6

/-- The number of emails Jack received in the evening -/
def evening_emails : ℕ := total_emails - (afternoon_emails + morning_emails)

theorem jack_evening_emails : evening_emails = 1 := by
  sorry

end jack_evening_emails_l2405_240542


namespace yeongsoo_initial_amount_l2405_240580

/-- Given the initial amounts of money for Yeongsoo, Hyogeun, and Woong,
    this function returns their final amounts after the transactions. -/
def final_amounts (y h w : ℕ) : ℕ × ℕ × ℕ :=
  (y - 200 + 1000, h + 200 - 500, w + 500 - 1000)

/-- Theorem stating that Yeongsoo's initial amount was 1200 won -/
theorem yeongsoo_initial_amount :
  ∃ (h w : ℕ), final_amounts 1200 h w = (2000, 2000, 2000) :=
sorry

end yeongsoo_initial_amount_l2405_240580


namespace person_age_puzzle_l2405_240566

theorem person_age_puzzle : ∃ (age : ℕ), 
  (3 * (age + 3) - 3 * (age - 3) = age) ∧ (age = 18) := by
  sorry

end person_age_puzzle_l2405_240566


namespace minimum_value_of_expression_l2405_240595

theorem minimum_value_of_expression (x : ℝ) (h : x > 1) :
  x + 1 / (x - 1) ≥ 3 ∧ ∃ y > 1, y + 1 / (y - 1) = 3 :=
by sorry

end minimum_value_of_expression_l2405_240595


namespace birthday_money_theorem_l2405_240515

def birthday_money_problem (initial_amount : ℚ) (video_game_fraction : ℚ) (goggles_fraction : ℚ) : ℚ :=
  let remaining_after_game := initial_amount * (1 - video_game_fraction)
  remaining_after_game * (1 - goggles_fraction)

theorem birthday_money_theorem :
  birthday_money_problem 100 (1/4) (1/5) = 60 := by
  sorry

end birthday_money_theorem_l2405_240515


namespace perpendicular_lines_parallel_l2405_240533

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (para : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_parallel (m n : Line) (α : Plane) :
  perp m α → perp n α → para m n :=
sorry

end perpendicular_lines_parallel_l2405_240533


namespace f_of_2_equals_4_l2405_240587

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x + 2

-- Theorem statement
theorem f_of_2_equals_4 : f 2 = 4 := by
  sorry

end f_of_2_equals_4_l2405_240587


namespace basketball_players_l2405_240582

theorem basketball_players (cricket : ℕ) (both : ℕ) (total : ℕ)
  (h1 : cricket = 8)
  (h2 : both = 6)
  (h3 : total = 11)
  (h4 : total = cricket + basketball - both) :
  basketball = 9 :=
by
  sorry

end basketball_players_l2405_240582


namespace pear_arrangement_l2405_240589

theorem pear_arrangement (n : ℕ) (weights : Fin (2*n+2) → ℝ) :
  ∃ (perm : Fin (2*n+2) ≃ Fin (2*n+2)),
    ∀ i : Fin (2*n+2), |weights (perm i) - weights (perm (i+1))| ≤ 1 :=
sorry

end pear_arrangement_l2405_240589


namespace smallest_sum_of_three_l2405_240563

def S : Finset Int := {-8, 2, -5, 17, -3}

theorem smallest_sum_of_three (a b c : Int) 
  (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  (∀ x y z : Int, x ∈ S → y ∈ S → z ∈ S → 
    x ≠ y ∧ y ≠ z ∧ x ≠ z → a + b + c ≤ x + y + z) ∧ 
  (∃ x y z : Int, x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ 
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x + y + z = -16) :=
by sorry

end smallest_sum_of_three_l2405_240563


namespace eight_operations_proof_l2405_240512

theorem eight_operations_proof :
  (((8 : ℚ) / 8) * ((8 : ℚ) / 8) = 1) ∧
  (((8 : ℚ) / 8) + ((8 : ℚ) / 8) = 2) := by
  sorry

end eight_operations_proof_l2405_240512


namespace prob_a_b_not_same_class_l2405_240575

/-- The number of students to be distributed -/
def num_students : ℕ := 4

/-- The number of classes -/
def num_classes : ℕ := 3

/-- The probability that students A and B are not in the same class -/
def prob_not_same_class : ℚ := 5/6

/-- The total number of ways to distribute students into classes -/
def total_distributions : ℕ := num_students.choose 2 * num_classes.factorial

/-- The number of distributions where A and B are in different classes -/
def favorable_distributions : ℕ := total_distributions - num_classes.factorial

theorem prob_a_b_not_same_class :
  (favorable_distributions : ℚ) / total_distributions = prob_not_same_class :=
sorry

end prob_a_b_not_same_class_l2405_240575


namespace tan_equality_225_l2405_240531

theorem tan_equality_225 (m : ℤ) :
  -180 < m ∧ m < 180 →
  (Real.tan (m * π / 180) = Real.tan (225 * π / 180) ↔ m = 45 ∨ m = -135) := by
  sorry

end tan_equality_225_l2405_240531


namespace a_annual_income_l2405_240560

/-- Prove that A's annual income is Rs. 504000 -/
theorem a_annual_income (a_monthly b_monthly c_monthly : ℕ) : 
  (a_monthly : ℚ) / b_monthly = 5 / 2 →
  b_monthly = (112 * c_monthly) / 100 →
  c_monthly = 15000 →
  12 * a_monthly = 504000 := by
  sorry

end a_annual_income_l2405_240560


namespace arithmetic_sequence_150th_term_l2405_240544

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

/-- The first term of our specific sequence -/
def a₁ : ℝ := 3

/-- The common difference of our specific sequence -/
def d : ℝ := 5

/-- The 150th term of our specific sequence -/
def a₁₅₀ : ℝ := arithmetic_sequence a₁ d 150

theorem arithmetic_sequence_150th_term : a₁₅₀ = 748 := by
  sorry

end arithmetic_sequence_150th_term_l2405_240544


namespace fraction_subtraction_l2405_240599

theorem fraction_subtraction : 
  (2 + 4 + 6 + 8) / (1 + 3 + 5 + 7) - (1 + 3 + 5 + 7) / (2 + 4 + 6 + 8) = 9 / 20 := by
  sorry

end fraction_subtraction_l2405_240599


namespace negation_of_proposition_l2405_240540

theorem negation_of_proposition :
  ¬(∀ x : ℝ, x > 0 → x^2 - x ≤ 1) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - x > 1) :=
by sorry

end negation_of_proposition_l2405_240540


namespace functional_equation_solution_l2405_240545

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x ^ 2 + f y) = x * f x + y

/-- The main theorem stating that any function satisfying the equation
    must be either the identity function or the negation function -/
theorem functional_equation_solution (f : ℝ → ℝ) 
    (h : SatisfiesEquation f) : 
    (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) := by
  sorry

end functional_equation_solution_l2405_240545


namespace fraction_equivalence_l2405_240511

theorem fraction_equivalence : 
  ∀ (n : ℕ), (4 + n : ℚ) / (7 + n) = 7 / 8 → n = 17 :=
by sorry

end fraction_equivalence_l2405_240511


namespace matrix_not_invertible_l2405_240500

theorem matrix_not_invertible : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2 + 16/19, 9; 4 - 16/19, 10]
  ¬(IsUnit (Matrix.det A)) := by
sorry

end matrix_not_invertible_l2405_240500


namespace a_in_A_l2405_240551

def A : Set ℝ := {x | x ≥ 2 * Real.sqrt 2}

theorem a_in_A : 3 ∈ A := by
  sorry

end a_in_A_l2405_240551


namespace persistent_iff_two_l2405_240546

/-- A number T is persistent if for any a, b, c, d ∈ ℝ \ {0, 1} satisfying
    a + b + c + d = T and 1/a + 1/b + 1/c + 1/d = T,
    we also have 1/(1-a) + 1/(1-b) + 1/(1-c) + 1/(1-d) = T -/
def isPersistent (T : ℝ) : Prop :=
  ∀ a b c d : ℝ, a ≠ 0 ∧ a ≠ 1 ∧ b ≠ 0 ∧ b ≠ 1 ∧ c ≠ 0 ∧ c ≠ 1 ∧ d ≠ 0 ∧ d ≠ 1 →
    a + b + c + d = T →
    1/a + 1/b + 1/c + 1/d = T →
    1/(1-a) + 1/(1-b) + 1/(1-c) + 1/(1-d) = T

/-- The only persistent number is 2 -/
theorem persistent_iff_two : ∀ T : ℝ, isPersistent T ↔ T = 2 := by
  sorry

end persistent_iff_two_l2405_240546


namespace overlap_area_63_l2405_240521

/-- Represents the geometric shapes and their movement --/
structure GeometricSetup where
  square_side : ℝ
  triangle_hypotenuse : ℝ
  initial_distance : ℝ
  relative_speed : ℝ

/-- Calculates the overlapping area at a given time --/
def overlapping_area (setup : GeometricSetup) (t : ℝ) : ℝ :=
  sorry

/-- The main theorem stating when the overlapping area is 63 square centimeters --/
theorem overlap_area_63 (setup : GeometricSetup) 
  (h1 : setup.square_side = 12)
  (h2 : setup.triangle_hypotenuse = 18)
  (h3 : setup.initial_distance = 13)
  (h4 : setup.relative_speed = 5) :
  (∃ t : ℝ, t = 5 ∨ t = 6.2) ∧ (overlapping_area setup t = 63) :=
sorry

end overlap_area_63_l2405_240521


namespace solve_system_equations_solve_system_inequalities_l2405_240519

-- Part 1: System of equations
theorem solve_system_equations :
  ∃! (x y : ℝ), 3 * x + 2 * y = 13 ∧ 2 * x + 3 * y = -8 ∧ x = 11 ∧ y = -10 := by sorry

-- Part 2: System of inequalities
theorem solve_system_inequalities :
  ∀ y : ℝ, ((5 * y - 2) / 3 - 1 > (3 * y - 5) / 2 ∧ 2 * (y - 3) ≤ 0) ↔ (-5 < y ∧ y ≤ 3) := by sorry

end solve_system_equations_solve_system_inequalities_l2405_240519


namespace total_students_accommodated_l2405_240557

/-- Represents a bus with its seating configuration and broken seats -/
structure Bus where
  columns : Nat
  rows : Nat
  broken_seats : Nat

/-- Calculates the number of usable seats in a bus -/
def usable_seats (bus : Bus) : Nat :=
  bus.columns * bus.rows - bus.broken_seats

/-- The list of buses with their configurations -/
def buses : List Bus := [
  ⟨4, 10, 2⟩,
  ⟨5, 8, 4⟩,
  ⟨3, 12, 3⟩,
  ⟨4, 12, 1⟩,
  ⟨6, 8, 5⟩,
  ⟨5, 10, 2⟩
]

/-- Theorem: The total number of students that can be accommodated is 245 -/
theorem total_students_accommodated : (buses.map usable_seats).sum = 245 := by
  sorry


end total_students_accommodated_l2405_240557


namespace team_a_games_l2405_240552

theorem team_a_games (a : ℕ) : 
  (2 : ℚ) / 3 * a = (5 : ℚ) / 8 * (a + 14) - 7 → a = 42 := by
  sorry

end team_a_games_l2405_240552


namespace quadratic_inequality_solution_set_l2405_240593

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - 2*x - 3 < 0} = Set.Ioo (-1 : ℝ) 3 := by
  sorry

end quadratic_inequality_solution_set_l2405_240593


namespace gratuity_calculation_l2405_240585

-- Define the given values
def total_bill : ℝ := 140
def tax_rate : ℝ := 0.10
def striploin_cost : ℝ := 80
def wine_cost : ℝ := 10

-- Define the theorem
theorem gratuity_calculation :
  let pre_tax_total := striploin_cost + wine_cost
  let tax_amount := pre_tax_total * tax_rate
  let bill_with_tax := pre_tax_total + tax_amount
  let gratuity := total_bill - bill_with_tax
  gratuity = 41 := by sorry

end gratuity_calculation_l2405_240585


namespace complex_equation_solution_l2405_240503

theorem complex_equation_solution (z : ℂ) : 
  Complex.I * (z - 1) = 1 + Complex.I → z = 2 - Complex.I := by
  sorry

end complex_equation_solution_l2405_240503


namespace probability_club_after_removal_l2405_240598

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (ranks : ℕ)
  (suits : ℕ)

/-- Represents the deck after removing spade cards -/
structure ModifiedDeck :=
  (remaining_cards : ℕ)
  (club_cards : ℕ)

/-- The probability of drawing a club card from the modified deck -/
def probability_club (d : ModifiedDeck) : ℚ :=
  d.club_cards / d.remaining_cards

theorem probability_club_after_removal (standard_deck : Deck) (modified_deck : ModifiedDeck) :
  standard_deck.total_cards = 52 →
  standard_deck.ranks = 13 →
  standard_deck.suits = 4 →
  modified_deck.remaining_cards = 48 →
  modified_deck.club_cards = 13 →
  probability_club modified_deck = 13 / 48 := by
  sorry

#eval (13 : ℚ) / 48

end probability_club_after_removal_l2405_240598


namespace sum_of_cubes_constraint_l2405_240539

theorem sum_of_cubes_constraint (a b : ℝ) :
  a^3 + b^3 = 1 - 3*a*b → (a + b = 1 ∨ a + b = -2) :=
by sorry

end sum_of_cubes_constraint_l2405_240539


namespace square_root_of_81_l2405_240583

theorem square_root_of_81 : Real.sqrt 81 = 9 := by
  sorry

end square_root_of_81_l2405_240583


namespace shaded_square_area_fraction_l2405_240529

/-- The area of a square with vertices at (2,1), (4,3), (2,5), and (0,3) divided by the area of a 5x5 square -/
theorem shaded_square_area_fraction : 
  let vertices : List (ℤ × ℤ) := [(2,1), (4,3), (2,5), (0,3)]
  let side_length := Real.sqrt ((4 - 2)^2 + (3 - 1)^2)
  let shaded_area := side_length ^ 2
  let grid_area := 5^2
  shaded_area / grid_area = 8 / 25 := by sorry

end shaded_square_area_fraction_l2405_240529


namespace subset_ratio_eight_elements_l2405_240538

theorem subset_ratio_eight_elements :
  let n : ℕ := 8
  let S : ℕ := 2^n
  let T : ℕ := n.choose 3
  (T : ℚ) / S = 7 / 32 := by
sorry

end subset_ratio_eight_elements_l2405_240538


namespace connectivity_determination_bound_l2405_240523

/-- A graph with n vertices -/
structure Graph (n : ℕ) where
  adj : Fin n → Fin n → Bool

/-- Distance between two vertices in a graph -/
def distance (G : Graph n) (u v : Fin n) : ℕ := sorry

/-- Whether a graph is connected -/
def is_connected (G : Graph n) : Prop := sorry

/-- A query about the distance between two vertices -/
structure Query (n : ℕ) where
  u : Fin n
  v : Fin n

/-- Result of a query -/
inductive QueryResult
  | LessThan
  | EqualTo
  | GreaterThan

/-- Function to determine if a graph is connected using queries -/
def determine_connectivity (n k : ℕ) (h : k ≤ n) (G : Graph n) : 
  ∃ (queries : List (Query n)), 
    queries.length ≤ 2 * n^2 / k ∧ 
    (∀ q : Query n, q ∈ queries → ∃ r : QueryResult, r = sorry) → 
    ∃ b : Bool, b = is_connected G := sorry

/-- Main theorem -/
theorem connectivity_determination_bound (n k : ℕ) (h : k ≤ n) :
  ∀ G : Graph n, ∃ (queries : List (Query n)), 
    queries.length ≤ 2 * n^2 / k ∧ 
    (∀ q : Query n, q ∈ queries → ∃ r : QueryResult, r = sorry) → 
    ∃ b : Bool, b = is_connected G := by
  sorry

end connectivity_determination_bound_l2405_240523


namespace wizard_collection_value_l2405_240527

def base7ToBase10 (n : List Nat) : Nat :=
  n.enum.foldr (fun (i, d) acc => acc + d * (7 ^ i)) 0

theorem wizard_collection_value :
  let crystal_ball := [3, 4, 2, 6]
  let wand := [0, 5, 6, 1]
  let book := [2, 0, 2]
  base7ToBase10 crystal_ball + base7ToBase10 wand + base7ToBase10 book = 2959 := by
  sorry

end wizard_collection_value_l2405_240527


namespace botanist_flower_distribution_l2405_240588

theorem botanist_flower_distribution (total_flowers : ℕ) (num_bouquets : ℕ) (additional_flowers : ℕ) : 
  total_flowers = 601 →
  num_bouquets = 8 →
  additional_flowers = 7 →
  (total_flowers + additional_flowers) % num_bouquets = 0 ∧
  (total_flowers + additional_flowers - 1) % num_bouquets ≠ 0 :=
by
  sorry

end botanist_flower_distribution_l2405_240588


namespace point_A_coordinates_l2405_240569

/-- A translation that moves any point (a,b) to (a+2,b-6) -/
def translation (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + 2, p.2 - 6)

/-- The point A₁ after translation -/
def A1 : ℝ × ℝ := (4, -3)

theorem point_A_coordinates :
  ∃ A : ℝ × ℝ, translation A = A1 ∧ A = (2, 3) := by sorry

end point_A_coordinates_l2405_240569


namespace equal_spaced_roots_value_l2405_240570

theorem equal_spaced_roots_value (k : ℝ) : 
  (∃ a b c d : ℝ, 
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
    (a^2 - 1) * (a^2 - 4) = k ∧
    (b^2 - 1) * (b^2 - 4) = k ∧
    (c^2 - 1) * (c^2 - 4) = k ∧
    (d^2 - 1) * (d^2 - 4) = k ∧
    b - a = c - b ∧ c - b = d - c) →
  k = 7/4 := by
sorry

end equal_spaced_roots_value_l2405_240570


namespace ramanujan_number_l2405_240509

def hardy_number : ℂ := 4 + 2 * Complex.I
def product : ℂ := 18 - 34 * Complex.I

theorem ramanujan_number : 
  ∃ r : ℂ, r * hardy_number = product ∧ r = 0.2 - 8.6 * Complex.I :=
by sorry

end ramanujan_number_l2405_240509


namespace min_freight_cost_l2405_240577

/-- Represents the freight problem with given parameters -/
structure FreightProblem where
  totalOre : ℕ
  truckCapacity1 : ℕ
  truckCapacity2 : ℕ
  truckCost1 : ℕ
  truckCost2 : ℕ

/-- Calculates the total cost for a given number of trucks -/
def totalCost (p : FreightProblem) (trucks1 : ℕ) (trucks2 : ℕ) : ℕ :=
  trucks1 * p.truckCost1 + trucks2 * p.truckCost2

/-- Checks if a combination of trucks can transport the required amount of ore -/
def isValidCombination (p : FreightProblem) (trucks1 : ℕ) (trucks2 : ℕ) : Prop :=
  trucks1 * p.truckCapacity1 + trucks2 * p.truckCapacity2 ≥ p.totalOre

/-- The main theorem stating that 685 is the minimum freight cost -/
theorem min_freight_cost (p : FreightProblem) 
  (h1 : p.totalOre = 73)
  (h2 : p.truckCapacity1 = 7)
  (h3 : p.truckCapacity2 = 5)
  (h4 : p.truckCost1 = 65)
  (h5 : p.truckCost2 = 50) :
  (∀ trucks1 trucks2 : ℕ, isValidCombination p trucks1 trucks2 → totalCost p trucks1 trucks2 ≥ 685) ∧ 
  (∃ trucks1 trucks2 : ℕ, isValidCombination p trucks1 trucks2 ∧ totalCost p trucks1 trucks2 = 685) :=
sorry


end min_freight_cost_l2405_240577


namespace shoes_difference_l2405_240592

/-- Represents the number of shoes tried on at each store --/
structure ShoesTried where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- The total number of shoes tried on across all stores --/
def totalShoesTried (s : ShoesTried) : ℕ :=
  s.first + s.second + s.third + s.fourth

/-- The conditions from the problem --/
def problemConditions (s : ShoesTried) : Prop :=
  s.first = 7 ∧
  s.third = 0 ∧
  s.fourth = 2 * (s.first + s.second + s.third) ∧
  totalShoesTried s = 48

/-- The theorem to prove --/
theorem shoes_difference (s : ShoesTried) 
  (h : problemConditions s) : s.second - s.first = 2 := by
  sorry


end shoes_difference_l2405_240592


namespace group_purchase_equation_l2405_240524

/-- Represents a group purchase scenario -/
structure GroupPurchase where
  price : ℝ  -- Price of the item
  contribution1 : ℝ  -- First contribution amount per person
  excess : ℝ  -- Excess amount for first contribution
  contribution2 : ℝ  -- Second contribution amount per person
  shortage : ℝ  -- Shortage amount for second contribution

/-- Theorem stating the equation for the group purchase scenario -/
theorem group_purchase_equation (gp : GroupPurchase) 
  (h1 : gp.contribution1 = 8) 
  (h2 : gp.excess = 3) 
  (h3 : gp.contribution2 = 7) 
  (h4 : gp.shortage = 4) :
  (gp.price + gp.excess) / gp.contribution1 = (gp.price - gp.shortage) / gp.contribution2 := by
  sorry

end group_purchase_equation_l2405_240524


namespace geometric_sequence_sum_l2405_240530

/-- Given a geometric sequence {a_n} with sum S_n = 3^(n-1) + t for all n ≥ 1,
    prove that t + a_3 = 17/3 -/
theorem geometric_sequence_sum (a : ℕ → ℚ) (S : ℕ → ℚ) (t : ℚ) 
  (h1 : ∀ n : ℕ, n ≥ 1 → S n = 3^(n-1) + t)
  (h2 : ∀ n : ℕ, n ≥ 2 → a n = S n - S (n-1))
  (h3 : ∀ n m : ℕ, n ≥ 1 → m ≥ 1 → a (n+1) / a n = a (m+1) / a m) :
  t + a 3 = 17/3 := by
sorry

end geometric_sequence_sum_l2405_240530


namespace parallel_vectors_x_value_l2405_240567

theorem parallel_vectors_x_value (x : ℝ) :
  let a : Fin 2 → ℝ := ![1, x]
  let b : Fin 2 → ℝ := ![2, 2 - x]
  (∃ (k : ℝ), a = k • b) → x = 2/3 := by
  sorry

end parallel_vectors_x_value_l2405_240567


namespace max_a_condition_1_range_a_condition_2_l2405_240579

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |2*x - a| + a
def g (x : ℝ) : ℝ := |2*x - 1|

-- Theorem for the first part of the problem
theorem max_a_condition_1 :
  (∀ a : ℝ, (∀ x : ℝ, g x ≤ 5 → f a x ≤ 6) → a ≤ 1) ∧
  (∃ a : ℝ, a = 1 ∧ ∀ x : ℝ, g x ≤ 5 → f a x ≤ 6) :=
sorry

-- Theorem for the second part of the problem
theorem range_a_condition_2 :
  ∀ a : ℝ, (∀ x : ℝ, f a x + g x ≥ 3) → a ≥ 2 :=
sorry

end max_a_condition_1_range_a_condition_2_l2405_240579


namespace solve_for_x_l2405_240525

theorem solve_for_x (x y : ℝ) (h1 : x + 2 * y = 10) (h2 : y = 3) : x = 4 := by
  sorry

end solve_for_x_l2405_240525


namespace intersection_N_complement_M_l2405_240532

def M : Set ℝ := {x | x > 2}
def N : Set ℝ := {x | 1 < x ∧ x ≤ 3}

theorem intersection_N_complement_M : N ∩ (Mᶜ) = {x | 1 < x ∧ x ≤ 2} := by sorry

end intersection_N_complement_M_l2405_240532


namespace socks_cost_proof_l2405_240558

/-- The cost of a uniform item without discount -/
structure UniformItem where
  shirt : ℝ
  pants : ℝ
  socks : ℝ

/-- The cost of a uniform item with discount -/
structure DiscountedUniformItem where
  shirt : ℝ
  pants : ℝ
  socks : ℝ

def team_size : ℕ := 12
def team_savings : ℝ := 36

def regular_uniform : UniformItem :=
  { shirt := 7.5,
    pants := 15,
    socks := 4.5 }  -- We use the answer here as we're proving this value

def discounted_uniform : DiscountedUniformItem :=
  { shirt := 6.75,
    pants := 13.5,
    socks := 3.75 }

theorem socks_cost_proof :
  let regular_total := team_size * (regular_uniform.shirt + regular_uniform.pants + regular_uniform.socks)
  let discounted_total := team_size * (discounted_uniform.shirt + discounted_uniform.pants + discounted_uniform.socks)
  regular_total - discounted_total = team_savings :=
by sorry

end socks_cost_proof_l2405_240558


namespace breakfast_expectation_l2405_240516

/-- Represents the possible outcomes of rolling a fair six-sided die, excluding 1 (which leads to a reroll) -/
inductive DieOutcome
| two
| three
| four
| five
| six

/-- The probability of rolling an even number (2, 4, or 6) after accounting for rerolls on 1 -/
def prob_even : ℚ := 3/5

/-- The probability of rolling an odd number (3 or 5) after accounting for rerolls on 1 -/
def prob_odd : ℚ := 2/5

/-- The number of days in a non-leap year -/
def days_in_year : ℕ := 365

/-- The expected difference between days eating pancakes and days eating oatmeal -/
def expected_difference : ℚ := prob_even * days_in_year - prob_odd * days_in_year

theorem breakfast_expectation :
  expected_difference = 73 := by sorry

end breakfast_expectation_l2405_240516


namespace perfect_square_polynomial_l2405_240505

theorem perfect_square_polynomial (g : ℕ) : 
  (∃ k : ℕ, g^4 + g^3 + g^2 + g + 1 = k^2) → g = 3 :=
by sorry

end perfect_square_polynomial_l2405_240505


namespace machine_y_efficiency_l2405_240561

/-- The number of widgets produced by both machines -/
def total_widgets : ℕ := 1080

/-- The number of widgets Machine X produces per hour -/
def machine_x_rate : ℕ := 3

/-- The difference in hours between Machine X and Machine Y to produce the total widgets -/
def time_difference : ℕ := 60

/-- Calculate the percentage difference between two numbers -/
def percentage_difference (a b : ℚ) : ℚ := (b - a) / a * 100

/-- Theorem stating that Machine Y produces 20% more widgets per hour than Machine X -/
theorem machine_y_efficiency : 
  let machine_x_time := total_widgets / machine_x_rate
  let machine_y_time := machine_x_time - time_difference
  let machine_y_rate := total_widgets / machine_y_time
  percentage_difference machine_x_rate machine_y_rate = 20 := by sorry

end machine_y_efficiency_l2405_240561


namespace mixed_grains_in_rice_l2405_240564

theorem mixed_grains_in_rice (total_stones : ℕ) (sample_size : ℕ) (mixed_in_sample : ℕ) :
  total_stones = 1536 →
  sample_size = 256 →
  mixed_in_sample = 18 →
  (total_stones * mixed_in_sample) / sample_size = 108 :=
by
  sorry

#check mixed_grains_in_rice

end mixed_grains_in_rice_l2405_240564


namespace complement_A_in_U_l2405_240562

def U : Set ℝ := {x | x > 0}
def A : Set ℝ := {x | x > 2}

theorem complement_A_in_U : 
  (U \ A) = {x : ℝ | 0 < x ∧ x ≤ 2} := by sorry

end complement_A_in_U_l2405_240562


namespace parallelogram_area_and_scaling_l2405_240565

theorem parallelogram_area_and_scaling :
  let base : ℝ := 6
  let height : ℝ := 20
  let area := base * height
  let scaled_base := 3 * base
  let scaled_height := 3 * height
  let scaled_area := scaled_base * scaled_height
  (area = 120) ∧ 
  (scaled_area = 9 * area) ∧ 
  (scaled_area = 1080) := by
  sorry

end parallelogram_area_and_scaling_l2405_240565


namespace arithmetic_mean_of_fractions_l2405_240559

theorem arithmetic_mean_of_fractions (x a : ℝ) (hx : x ≠ 0) :
  ((x^2 + a) / x^2 + (x^2 - a) / x^2) / 2 = 1 := by
  sorry

end arithmetic_mean_of_fractions_l2405_240559


namespace equation_solution_l2405_240536

theorem equation_solution : 
  let f (x : ℝ) := 1 / (x^2 + 13*x - 10) + 1 / (x^2 + 4*x - 10) + 1 / (x^2 - 11*x - 10)
  ∀ x : ℝ, f x = 0 ↔ 
    x = (-15 + Real.sqrt 265) / 2 ∨ 
    x = (-15 - Real.sqrt 265) / 2 ∨ 
    x = (6 + Real.sqrt 76) / 2 ∨ 
    x = (6 - Real.sqrt 76) / 2 :=
by sorry

end equation_solution_l2405_240536


namespace running_track_area_l2405_240597

theorem running_track_area (r : ℝ) (w : ℝ) (h1 : r = 50) (h2 : w = 3) :
  π * ((r + w)^2 - r^2) = 309 * π := by
  sorry

end running_track_area_l2405_240597


namespace satisfying_polynomial_form_l2405_240501

/-- A polynomial satisfying the given condition -/
def SatisfyingPolynomial (p : ℝ → ℝ) : Prop :=
  ∀ a b c : ℝ, p (a + b - 2*c) + p (b + c - 2*a) + p (c + a - 2*b) = 
               3 * (p (a - b) + p (b - c) + p (c - a))

/-- The theorem stating the form of polynomials satisfying the condition -/
theorem satisfying_polynomial_form (p : ℝ → ℝ) 
  (h : SatisfyingPolynomial p) :
  ∃ a₂ a₁ : ℝ, ∀ x, p x = a₂ * x^2 + a₁ * x :=
sorry

end satisfying_polynomial_form_l2405_240501


namespace society_of_beggars_voting_l2405_240586

/-- The Society of Beggars voting problem -/
theorem society_of_beggars_voting (initial_for : ℕ) (initial_against : ℕ) (no_chair : ℕ) : 
  initial_for = 115 → 
  initial_against = 92 → 
  no_chair = 12 → 
  initial_for + initial_against + no_chair = 207 := by
sorry

end society_of_beggars_voting_l2405_240586


namespace linear_function_k_value_l2405_240522

/-- Given a linear function y = kx - 2 that passes through the point (-1, 3), prove that k = -5 -/
theorem linear_function_k_value (k : ℝ) : 
  (∀ x y : ℝ, y = k * x - 2) → -- The function is y = kx - 2
  (3 : ℝ) = k * (-1 : ℝ) - 2 → -- The function passes through the point (-1, 3)
  k = -5 := by
sorry

end linear_function_k_value_l2405_240522


namespace expression_behavior_l2405_240528

/-- Given a > b > c, this theorem characterizes the behavior of the expression (a-x)(b-x)/(c-x) for different values of x. -/
theorem expression_behavior (a b c x : ℝ) (h : a > b ∧ b > c) :
  let f := fun (x : ℝ) => (a - x) * (b - x) / (c - x)
  (x < c ∨ (b < x ∧ x < a) → f x > 0) ∧
  ((c < x ∧ x < b) ∨ x > a → f x < 0) ∧
  (x = a ∨ x = b → f x = 0) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ y, 0 < |y - c| ∧ |y - c| < δ → |f y| > 1/ε) :=
by sorry

end expression_behavior_l2405_240528


namespace expression_evaluation_l2405_240556

theorem expression_evaluation : 2 * 0 + 1 - 9 = -8 := by
  sorry

end expression_evaluation_l2405_240556


namespace jellybean_theorem_l2405_240543

/-- Calculates the final number of jellybeans in a jar after a series of actions. -/
def final_jellybean_count (initial : ℕ) (samantha_took : ℕ) (shelby_ate : ℕ) : ℕ :=
  let scarlett_took := 2 * shelby_ate
  let scarlett_returned := (scarlett_took * 2) / 5  -- 40% rounded down
  let shannon_refilled := (samantha_took + shelby_ate) / 2
  initial - samantha_took - shelby_ate + scarlett_returned + shannon_refilled

/-- Theorem stating that given the initial conditions, the final number of jellybeans is 81. -/
theorem jellybean_theorem : final_jellybean_count 90 24 12 = 81 := by
  sorry

#eval final_jellybean_count 90 24 12

end jellybean_theorem_l2405_240543


namespace laptop_price_proof_l2405_240581

/-- The original sticker price of the laptop -/
def original_price : ℝ := 500

/-- The price at Store A after discount and rebate -/
def store_a_price (x : ℝ) : ℝ := 0.82 * x - 100

/-- The price at Store B after discount -/
def store_b_price (x : ℝ) : ℝ := 0.7 * x

/-- Theorem stating that the original price satisfies the given conditions -/
theorem laptop_price_proof :
  store_a_price original_price = store_b_price original_price - 40 := by
  sorry

end laptop_price_proof_l2405_240581


namespace deposit_percentage_l2405_240514

def deposit : ℝ := 3800
def monthly_income : ℝ := 11875

theorem deposit_percentage : (deposit / monthly_income) * 100 = 32 := by
  sorry

end deposit_percentage_l2405_240514


namespace cone_lateral_surface_area_l2405_240537

theorem cone_lateral_surface_area (r h : ℝ) (hr : r = 4) (hh : h = 5) :
  π * r * h = 20 * π := by
  sorry

end cone_lateral_surface_area_l2405_240537


namespace max_value_problem_l2405_240508

theorem max_value_problem (a b : ℝ) 
  (h1 : 4 * a + 3 * b ≤ 10) 
  (h2 : 3 * a + 5 * b ≤ 11) : 
  2 * a + b ≤ 48 / 11 :=
by sorry

end max_value_problem_l2405_240508


namespace product_586645_9999_l2405_240502

theorem product_586645_9999 : 586645 * 9999 = 5865885355 := by
  sorry

end product_586645_9999_l2405_240502


namespace not_prime_n_fourth_plus_four_to_n_l2405_240568

theorem not_prime_n_fourth_plus_four_to_n (n : ℕ) (h : n > 1) :
  ¬ Nat.Prime (n^4 + 4^n) := by
  sorry

end not_prime_n_fourth_plus_four_to_n_l2405_240568


namespace base_for_four_digit_256_l2405_240510

theorem base_for_four_digit_256 : ∃! b : ℕ, 
  b > 1 ∧ b^3 ≤ 256 ∧ 256 < b^4 :=
by sorry

end base_for_four_digit_256_l2405_240510


namespace fraction_subtraction_l2405_240584

theorem fraction_subtraction : 
  let a := 3 + 6 + 9 + 12
  let b := 2 + 5 + 8 + 11
  (a / b) - (b / a) = 56 / 195 := by
sorry

end fraction_subtraction_l2405_240584


namespace decimal_to_fraction_l2405_240504

theorem decimal_to_fraction :
  (3.375 : ℚ) = 27 / 8 := by sorry

end decimal_to_fraction_l2405_240504


namespace currency_notes_total_l2405_240513

/-- Proves that given the specified conditions, the total amount of currency notes is 5000 rupees. -/
theorem currency_notes_total (total_notes : ℕ) (amount_50 : ℕ) : 
  total_notes = 85 →
  amount_50 = 3500 →
  ∃ (notes_100 notes_50 : ℕ),
    notes_100 + notes_50 = total_notes ∧
    50 * notes_50 = amount_50 ∧
    100 * notes_100 + 50 * notes_50 = 5000 := by
  sorry

end currency_notes_total_l2405_240513


namespace consecutive_integers_average_l2405_240594

theorem consecutive_integers_average (c d : ℝ) : 
  (d = (c + (c+1) + (c+2) + (c+3) + (c+4) + (c+5)) / 6) →
  ((d-2) + (d-1) + d + (d+1) + (d+2) + (d+3)) / 6 = c + 3 := by
  sorry

end consecutive_integers_average_l2405_240594


namespace omega_real_iff_m_eq_4_or_neg_3_omega_in_fourth_quadrant_iff_3_lt_m_lt_4_l2405_240534

-- Define the complex number ω as a function of m
def ω (m : ℝ) : ℂ := Complex.mk (m^2 - 2*m - 3) (m^2 - m - 12)

-- Theorem 1: ω is real iff m = 4 or m = -3
theorem omega_real_iff_m_eq_4_or_neg_3 (m : ℝ) :
  ω m ∈ Set.range Complex.ofReal ↔ m = 4 ∨ m = -3 :=
sorry

-- Theorem 2: ω is in the fourth quadrant iff 3 < m < 4
theorem omega_in_fourth_quadrant_iff_3_lt_m_lt_4 (m : ℝ) :
  (Complex.re (ω m) > 0 ∧ Complex.im (ω m) < 0) ↔ 3 < m ∧ m < 4 :=
sorry

end omega_real_iff_m_eq_4_or_neg_3_omega_in_fourth_quadrant_iff_3_lt_m_lt_4_l2405_240534


namespace probability_three_same_group_l2405_240596

/-- The number of students in the school -/
def total_students : ℕ := 600

/-- The number of lunch groups -/
def num_groups : ℕ := 3

/-- Assumption that the groups are of equal size -/
axiom groups_equal_size : total_students % num_groups = 0

/-- The probability of a student being assigned to a specific group -/
def prob_one_group : ℚ := 1 / num_groups

/-- The probability of three specific students being assigned to the same lunch group -/
def prob_three_same_group : ℚ := prob_one_group * prob_one_group

theorem probability_three_same_group :
  prob_three_same_group = 1 / 9 :=
sorry

end probability_three_same_group_l2405_240596


namespace smallest_four_digit_divisible_by_6_l2405_240572

/-- A number is a four-digit number if it's greater than or equal to 1000 and less than 10000 -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

/-- The smallest four-digit number divisible by 6 -/
def smallest_four_digit_div_by_6 : ℕ := 1002

theorem smallest_four_digit_divisible_by_6 :
  (is_four_digit smallest_four_digit_div_by_6) ∧
  (smallest_four_digit_div_by_6 % 6 = 0) ∧
  (∀ n : ℕ, is_four_digit n ∧ n % 6 = 0 → smallest_four_digit_div_by_6 ≤ n) :=
by sorry

end smallest_four_digit_divisible_by_6_l2405_240572


namespace coefficient_sum_l2405_240548

-- Define the function f
def f (x : ℝ) : ℝ := sorry

-- Define the coefficients a, b, c, d
def a : ℝ := sorry
def b : ℝ := sorry
def c : ℝ := sorry
def d : ℝ := sorry

-- State the theorem
theorem coefficient_sum :
  (∀ x, f (x + 3) = 3 * x^2 + 7 * x + 4) →
  (∀ x, f x = a * x^3 + b * x^2 + c * x + d) →
  a + b + c + d = -7 := by sorry

end coefficient_sum_l2405_240548


namespace min_value_and_nonexistence_l2405_240549

theorem min_value_and_nonexistence (a b : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : 1/a + 1/b = Real.sqrt (a*b)) : 
  (∀ x y : ℝ, x > 0 → y > 0 → 1/x + 1/y = Real.sqrt (x*y) → x^3 + y^3 ≥ 4 * Real.sqrt 2) ∧ 
  (¬∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 1/x + 1/y = Real.sqrt (x*y) ∧ 2*x + 3*y = 6) :=
by sorry

end min_value_and_nonexistence_l2405_240549


namespace medians_intersect_l2405_240506

/-- Definition of a triangle -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Definition of a point being the midpoint of a line segment -/
def isMidpoint (M : ℝ × ℝ) (P Q : ℝ × ℝ) : Prop :=
  M.1 = (P.1 + Q.1) / 2 ∧ M.2 = (P.2 + Q.2) / 2

/-- Definition of a median -/
def isMedian (P Q R S : ℝ × ℝ) : Prop :=
  isMidpoint S Q R

/-- Theorem: The medians of a triangle intersect at a single point -/
theorem medians_intersect (t : Triangle) 
  (A' : ℝ × ℝ) (B' : ℝ × ℝ) (C' : ℝ × ℝ)
  (h1 : isMidpoint A' t.B t.C)
  (h2 : isMidpoint B' t.C t.A)
  (h3 : isMidpoint C' t.A t.B)
  (h4 : isMedian t.A t.B t.C A')
  (h5 : isMedian t.B t.C t.A B')
  (h6 : isMedian t.C t.A t.B C') :
  ∃ G : ℝ × ℝ, (∃ k₁ k₂ k₃ : ℝ, 
    G = k₁ • t.A + (1 - k₁) • A' ∧
    G = k₂ • t.B + (1 - k₂) • B' ∧
    G = k₃ • t.C + (1 - k₃) • C') :=
  sorry


end medians_intersect_l2405_240506


namespace f_2_neg3_neg1_eq_half_l2405_240535

def f (a b c : ℚ) : ℚ := (c + a) / (c - b)

theorem f_2_neg3_neg1_eq_half : f 2 (-3) (-1) = 1/2 := by
  sorry

end f_2_neg3_neg1_eq_half_l2405_240535


namespace smallest_x_value_l2405_240574

theorem smallest_x_value : 
  ∃ (x : ℝ), x > 1 ∧ 
  ((5*x - 20) / (4*x - 5))^2 + (5*x - 20) / (4*x - 5) = 20 ∧
  (∀ (y : ℝ), y > 1 ∧ 
   ((5*y - 20) / (4*y - 5))^2 + (5*y - 20) / (4*y - 5) = 20 → 
   x ≤ y) ∧
  x = 9/5 :=
by sorry

end smallest_x_value_l2405_240574


namespace saltwater_solution_l2405_240547

/-- Represents the saltwater tank problem --/
def saltwater_problem (x : ℝ) : Prop :=
  let original_salt := 0.2 * x
  let volume_after_evaporation := 0.75 * x
  let salt_after_addition := original_salt + 14
  let final_volume := salt_after_addition / (1/3)
  let water_added := final_volume - volume_after_evaporation
  (x = 104.99999999999997) ∧ (water_added = 26.25)

/-- Theorem stating the solution to the saltwater problem --/
theorem saltwater_solution :
  ∃ (x : ℝ), saltwater_problem x :=
sorry

end saltwater_solution_l2405_240547


namespace quadratic_transformation_l2405_240520

/-- Given a quadratic expression px^2 + qx + r that can be expressed as 5(x + 3)^2 - 15,
    prove that when 4px^2 + 4qx + 4r is written in the form m(x - h)^2 + k, then h = -3 -/
theorem quadratic_transformation (p q r : ℝ) 
  (h : ∀ x, p * x^2 + q * x + r = 5 * (x + 3)^2 - 15) :
  ∃ (m k : ℝ), ∀ x, 4 * p * x^2 + 4 * q * x + 4 * r = m * (x - (-3))^2 + k :=
sorry

end quadratic_transformation_l2405_240520


namespace sphere_volume_ratio_l2405_240573

theorem sphere_volume_ratio (R : ℝ) (h : R > 0) :
  (4 / 3 * Real.pi * (2 * R)^3) / (4 / 3 * Real.pi * R^3) = 8 := by
  sorry

end sphere_volume_ratio_l2405_240573


namespace arithmetic_sequence_sum_l2405_240576

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 3 + a 4 + a 5 + a 6 + a 7 = 25) →
  (a 2 + a 8 = 10) := by
  sorry

end arithmetic_sequence_sum_l2405_240576


namespace range_of_a_l2405_240517

/-- The function f(x) = x³ + x + 1 -/
def f (x : ℝ) : ℝ := x^3 + x + 1

/-- Theorem stating that if f(x² + a) + f(ax) > 2 for all x, then 0 < a < 4 -/
theorem range_of_a (a : ℝ) : (∀ x : ℝ, f (x^2 + a) + f (a*x) > 2) → 0 < a ∧ a < 4 := by
  sorry

#check range_of_a

end range_of_a_l2405_240517


namespace seventeen_pairs_sold_l2405_240541

/-- Represents the sales data for an optometrist's contact lens business --/
structure ContactLensSales where
  soft_price : ℝ
  hard_price : ℝ
  soft_hard_difference : ℕ
  discount_rate : ℝ
  total_sales : ℝ

/-- Calculates the total number of contact lens pairs sold given the sales data --/
def total_pairs_sold (sales : ContactLensSales) : ℕ :=
  sorry

/-- Theorem stating that given the specific sales data, 17 pairs of lenses were sold --/
theorem seventeen_pairs_sold :
  let sales := ContactLensSales.mk 175 95 7 0.1 2469
  total_pairs_sold sales = 17 := by
  sorry

end seventeen_pairs_sold_l2405_240541
