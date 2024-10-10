import Mathlib

namespace yanna_change_l3037_303797

/-- The change Yanna received after buying shirts and sandals -/
def change_received (shirt_price shirt_quantity sandal_price sandal_quantity payment : ℕ) : ℕ :=
  payment - (shirt_price * shirt_quantity + sandal_price * sandal_quantity)

/-- Theorem stating that Yanna received $41 in change -/
theorem yanna_change :
  change_received 5 10 3 3 100 = 41 := by
  sorry

end yanna_change_l3037_303797


namespace four_men_absent_l3037_303764

/-- Represents the work completion scenario with absent workers -/
structure WorkScenario where
  totalMen : ℕ
  originalDays : ℕ
  actualDays : ℕ
  absentMen : ℕ

/-- Calculates the number of absent men given the work scenario -/
def calculateAbsentMen (scenario : WorkScenario) : ℕ :=
  scenario.totalMen - (scenario.totalMen * scenario.originalDays) / scenario.actualDays

/-- Theorem stating that 4 men became absent in the given scenario -/
theorem four_men_absent :
  let scenario := WorkScenario.mk 8 6 12 4
  calculateAbsentMen scenario = 4 := by
  sorry

#eval calculateAbsentMen (WorkScenario.mk 8 6 12 4)

end four_men_absent_l3037_303764


namespace marble_sculpture_weight_l3037_303723

theorem marble_sculpture_weight (original_weight : ℝ) : 
  original_weight > 0 →
  (0.75 * (0.80 * (0.70 * original_weight))) = 105 →
  original_weight = 250 :=
by
  sorry

end marble_sculpture_weight_l3037_303723


namespace smallest_student_count_l3037_303751

/-- Represents the number of students in each grade --/
structure GradeCount where
  ninth : ℕ
  tenth : ℕ
  eleventh : ℕ

/-- Checks if the given grade counts satisfy the required ratios --/
def satisfies_ratios (gc : GradeCount) : Prop :=
  4 * gc.ninth = 3 * gc.tenth ∧ 6 * gc.tenth = 5 * gc.eleventh

/-- The total number of students across the three grades --/
def total_students (gc : GradeCount) : ℕ :=
  gc.ninth + gc.tenth + gc.eleventh

/-- Theorem stating that 59 is the smallest number of students satisfying the ratios --/
theorem smallest_student_count : 
  ∃ (gc : GradeCount), satisfies_ratios gc ∧ total_students gc = 59 ∧
  ∀ (gc' : GradeCount), satisfies_ratios gc' → total_students gc' ≥ 59 :=
sorry

end smallest_student_count_l3037_303751


namespace janelle_initial_green_marbles_l3037_303783

/-- The number of bags of blue marbles Janelle bought -/
def blue_bags : ℕ := 6

/-- The number of marbles in each bag -/
def marbles_per_bag : ℕ := 10

/-- The number of green marbles in the gift -/
def green_marbles_gift : ℕ := 6

/-- The number of blue marbles in the gift -/
def blue_marbles_gift : ℕ := 8

/-- The number of marbles Janelle has left after giving the gift -/
def marbles_left : ℕ := 72

/-- The initial number of green marbles Janelle had -/
def initial_green_marbles : ℕ := 26

theorem janelle_initial_green_marbles :
  initial_green_marbles = green_marbles_gift + (marbles_left - (blue_bags * marbles_per_bag - blue_marbles_gift)) :=
by sorry

end janelle_initial_green_marbles_l3037_303783


namespace g_composition_theorem_l3037_303704

/-- The function g defined as g(x) = bx^3 - 1 --/
def g (b : ℝ) (x : ℝ) : ℝ := b * x^3 - 1

/-- Theorem stating that if g(g(1)) = -1 and b is positive, then b = 1 --/
theorem g_composition_theorem (b : ℝ) (h1 : b > 0) (h2 : g b (g b 1) = -1) : b = 1 := by
  sorry

end g_composition_theorem_l3037_303704


namespace equation_is_linear_with_one_var_l3037_303715

/-- A linear equation with one variable is an equation of the form ax + b = c, where a ≠ 0 and x is the variable. --/
def is_linear_equation_one_var (eq : ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, eq x ↔ a * x + b = c)

/-- The equation 4a - 1 = 8 --/
def equation (a : ℝ) : Prop := 4 * a - 1 = 8

theorem equation_is_linear_with_one_var : is_linear_equation_one_var equation := by
  sorry

end equation_is_linear_with_one_var_l3037_303715


namespace sum_over_subsets_equals_power_of_two_l3037_303769

def S : Finset ℕ := Finset.range 1999

def f (T : Finset ℕ) : ℕ := T.sum id

theorem sum_over_subsets_equals_power_of_two :
  (Finset.powerset S).sum (fun E => (f E : ℚ) / (f S : ℚ)) = 2^1998 := by sorry

end sum_over_subsets_equals_power_of_two_l3037_303769


namespace total_friends_count_l3037_303762

-- Define the number of friends who can pay Rs. 60 each
def standard_payers : ℕ := 10

-- Define the amount each standard payer would pay
def standard_payment : ℕ := 60

-- Define the extra amount paid by one friend
def extra_payment : ℕ := 50

-- Define the total amount paid by the friend who paid extra
def total_extra_payer_amount : ℕ := 115

-- Theorem to prove
theorem total_friends_count : 
  ∃ (n : ℕ), 
    n = standard_payers + 1 ∧ 
    n * (total_extra_payer_amount - extra_payment) = 
      standard_payers * standard_payment + extra_payment :=
by
  sorry

#check total_friends_count

end total_friends_count_l3037_303762


namespace parabola_shift_correct_l3037_303716

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The original parabola y = 2x^2 -/
def original_parabola : Parabola := { a := 2, b := 0, c := 0 }

/-- Shift a parabola horizontally and vertically -/
def shift_parabola (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c - v }

theorem parabola_shift_correct :
  let shifted := shift_parabola original_parabola 1 5
  shifted.a = 2 ∧ shifted.b = -4 ∧ shifted.c = -5 := by sorry

end parabola_shift_correct_l3037_303716


namespace perfect_square_divisibility_l3037_303706

theorem perfect_square_divisibility (x y : ℕ+) (h : (2 * x * y) ∣ (x^2 + y^2 - x)) : 
  ∃ (n : ℕ+), x = n^2 := by
sorry

end perfect_square_divisibility_l3037_303706


namespace cube_guessing_game_l3037_303709

/-- The maximum amount Alexei can guarantee himself in the cube guessing game -/
def maxGuaranteedAmount (m : ℕ) (n : ℕ) : ℚ :=
  2^m / (Nat.choose m n)

/-- The problem statement for the cube guessing game -/
theorem cube_guessing_game (n : ℕ) (hn : n ≤ 100) :
  /- Part a: One blue cube -/
  maxGuaranteedAmount 100 1 = 2^100 / 100 ∧
  /- Part b: n blue cubes -/
  maxGuaranteedAmount 100 n = 2^100 / (Nat.choose 100 n) :=
sorry

end cube_guessing_game_l3037_303709


namespace work_hours_per_day_l3037_303742

theorem work_hours_per_day (days : ℕ) (total_hours : ℕ) (h1 : days = 5) (h2 : total_hours = 40) :
  total_hours / days = 8 :=
by sorry

end work_hours_per_day_l3037_303742


namespace quadratic_roots_property_l3037_303755

theorem quadratic_roots_property (m n : ℝ) : 
  (m^2 + 2*m - 5 = 0) → (n^2 + 2*n - 5 = 0) → (m^2 + m*n + 2*m = 0) := by
  sorry

end quadratic_roots_property_l3037_303755


namespace labor_cost_per_hour_l3037_303799

theorem labor_cost_per_hour 
  (total_repair_cost : ℝ)
  (part_cost : ℝ)
  (labor_hours : ℝ)
  (h1 : total_repair_cost = 2400)
  (h2 : part_cost = 1200)
  (h3 : labor_hours = 16) :
  (total_repair_cost - part_cost) / labor_hours = 75 := by
sorry

end labor_cost_per_hour_l3037_303799


namespace min_correct_answers_to_pass_l3037_303725

/-- Represents a test with given parameters -/
structure Test where
  total_questions : ℕ
  points_correct : ℕ
  points_wrong : ℕ
  passing_score : ℕ

/-- Calculates the score based on the number of correct answers -/
def calculate_score (test : Test) (correct_answers : ℕ) : ℤ :=
  (test.points_correct * correct_answers : ℤ) - 
  (test.points_wrong * (test.total_questions - correct_answers) : ℤ)

/-- Theorem stating the minimum number of correct answers needed to pass the test -/
theorem min_correct_answers_to_pass (test : Test) 
  (h1 : test.total_questions = 20)
  (h2 : test.points_correct = 5)
  (h3 : test.points_wrong = 3)
  (h4 : test.passing_score = 60) :
  ∀ n : ℕ, n ≥ 15 ↔ calculate_score test n ≥ test.passing_score := by
  sorry

#check min_correct_answers_to_pass

end min_correct_answers_to_pass_l3037_303725


namespace fraction_equality_l3037_303774

theorem fraction_equality (a b : ℝ) (h : a / b = 6 / 5) :
  (5 * a + 4 * b) / (5 * a - 4 * b) = 5 := by
  sorry

end fraction_equality_l3037_303774


namespace greatest_prime_factor_of_5_pow_5_plus_10_pow_4_l3037_303761

theorem greatest_prime_factor_of_5_pow_5_plus_10_pow_4 :
  (Nat.factors (5^5 + 10^4)).maximum? = some 7 :=
sorry

end greatest_prime_factor_of_5_pow_5_plus_10_pow_4_l3037_303761


namespace f_increasing_iff_m_range_l3037_303790

def f (x m : ℝ) : ℝ := |x^2 + (m-1)*x + (m^2 - 3*m + 1)|

theorem f_increasing_iff_m_range :
  ∀ m : ℝ, (∀ x₁ x₂ : ℝ, -1 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 0 → f x₁ m < f x₂ m) ↔ (m = 1 ∨ m ≥ 3) :=
by sorry

end f_increasing_iff_m_range_l3037_303790


namespace compare_exponentials_l3037_303765

theorem compare_exponentials (h : 0 < 0.5 ∧ 0.5 < 1) : 0.5^(-2) > 0.5^(-0.8) := by
  sorry

end compare_exponentials_l3037_303765


namespace negative_cube_squared_l3037_303759

theorem negative_cube_squared (a : ℝ) : (-a^3)^2 = a^6 := by
  sorry

end negative_cube_squared_l3037_303759


namespace arithmetic_sequence_2011_l3037_303789

def arithmetic_sequence (a₁ d n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem arithmetic_sequence_2011 :
  arithmetic_sequence 1 3 671 = 2011 := by sorry

end arithmetic_sequence_2011_l3037_303789


namespace art_display_side_length_l3037_303719

/-- Represents the dimensions of a glass pane -/
structure GlassPane where
  width : ℝ
  height : ℝ
  ratio : height = 3 * width

/-- Represents the square art display -/
structure ArtDisplay where
  pane : GlassPane
  border_width : ℝ
  horizontal_panes : ℕ
  vertical_panes : ℕ
  is_square : horizontal_panes * pane.width + (horizontal_panes + 1) * border_width = 
              vertical_panes * pane.height + (vertical_panes + 1) * border_width

/-- The side length of the square display is 17.4 inches -/
theorem art_display_side_length (display : ArtDisplay) 
  (h1 : display.horizontal_panes = 4)
  (h2 : display.vertical_panes = 3)
  (h3 : display.border_width = 3) :
  display.horizontal_panes * display.pane.width + (display.horizontal_panes + 1) * display.border_width = 17.4 := by
  sorry

end art_display_side_length_l3037_303719


namespace largest_n_divisible_by_three_l3037_303782

def is_divisible_by_three (n : ℕ) : Prop :=
  ∃ k : ℕ, 8*(n+2)^5 - n^2 + 14*n - 30 = 3*k

theorem largest_n_divisible_by_three :
  (∀ m : ℕ, m < 100000 → is_divisible_by_three m) ∧
  (∀ m : ℕ, m > 99999 → m < 100000 → ¬is_divisible_by_three m) :=
sorry

end largest_n_divisible_by_three_l3037_303782


namespace weight_of_compound_approx_l3037_303771

/-- The atomic mass of carbon in g/mol -/
def carbon_mass : ℝ := 12.01

/-- The atomic mass of hydrogen in g/mol -/
def hydrogen_mass : ℝ := 1.008

/-- The atomic mass of oxygen in g/mol -/
def oxygen_mass : ℝ := 16.00

/-- The chemical formula of the compound -/
structure ChemicalFormula where
  carbon : ℕ
  hydrogen : ℕ
  oxygen : ℕ

/-- The compound C6H8O7 -/
def compound : ChemicalFormula := ⟨6, 8, 7⟩

/-- Calculate the molar mass of a chemical formula -/
def molar_mass (formula : ChemicalFormula) : ℝ :=
  formula.carbon * carbon_mass + 
  formula.hydrogen * hydrogen_mass + 
  formula.oxygen * oxygen_mass

/-- The number of moles -/
def num_moles : ℝ := 3

/-- The total weight in grams -/
def total_weight : ℝ := 576

/-- Theorem stating that the weight of 3 moles of C6H8O7 is approximately 576 grams -/
theorem weight_of_compound_approx (ε : ℝ) (ε_pos : ε > 0) : 
  |num_moles * molar_mass compound - total_weight| < ε := by
  sorry

end weight_of_compound_approx_l3037_303771


namespace joshua_toy_cars_l3037_303785

theorem joshua_toy_cars (total_boxes : ℕ) (cars_box1 : ℕ) (cars_box2 : ℕ) (total_cars : ℕ)
  (h1 : total_boxes = 3)
  (h2 : cars_box1 = 21)
  (h3 : cars_box2 = 31)
  (h4 : total_cars = 71) :
  total_cars - (cars_box1 + cars_box2) = 19 := by
  sorry

end joshua_toy_cars_l3037_303785


namespace counterfeit_probability_l3037_303749

def total_bills : ℕ := 20
def counterfeit_bills : ℕ := 5
def selected_bills : ℕ := 2

def prob_both_counterfeit : ℚ := (counterfeit_bills.choose selected_bills : ℚ) / (total_bills.choose selected_bills)
def prob_at_least_one_counterfeit : ℚ := 1 - ((total_bills - counterfeit_bills).choose selected_bills : ℚ) / (total_bills.choose selected_bills)

theorem counterfeit_probability :
  prob_both_counterfeit / prob_at_least_one_counterfeit = 2 / 17 := by
  sorry

end counterfeit_probability_l3037_303749


namespace part_one_part_two_l3037_303756

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (a + 1) * x + 2

-- Part I
theorem part_one : 
  {x : ℝ | f 2 x > 1} = {x : ℝ | x < 1/2 ∨ x > 1} := by sorry

-- Part II
theorem part_two : 
  (∀ x ∈ Set.Icc (-1) 3, f a x ≥ 0) → 
  1/6 ≤ a ∧ a ≤ 3 + 2 * Real.sqrt 2 := by sorry

end part_one_part_two_l3037_303756


namespace cube_preserves_order_l3037_303710

theorem cube_preserves_order (a b : ℝ) : a > b → a^3 > b^3 := by
  sorry

end cube_preserves_order_l3037_303710


namespace complex_fraction_equality_l3037_303793

theorem complex_fraction_equality : (1 - 2*I) / (1 + I) = (-1 - 3*I) / 2 := by
  sorry

end complex_fraction_equality_l3037_303793


namespace initial_rope_length_l3037_303788

theorem initial_rope_length 
  (r_initial : ℝ) 
  (h1 : r_initial > 0) 
  (h2 : π * (21^2 - r_initial^2) = 933.4285714285714) : 
  r_initial = 12 := by
sorry

end initial_rope_length_l3037_303788


namespace sin_graph_transformation_l3037_303787

theorem sin_graph_transformation (x : ℝ) :
  let f (x : ℝ) := Real.sin (2 * x)
  let g (x : ℝ) := f (x - π / 3)
  let h (x : ℝ) := g (-x)
  h x = Real.sin (-2 * x - 2 * π / 3) := by
  sorry

end sin_graph_transformation_l3037_303787


namespace ranked_choice_voting_theorem_l3037_303747

theorem ranked_choice_voting_theorem 
  (initial_votes_A initial_votes_B initial_votes_C initial_votes_D initial_votes_E : ℚ)
  (redistribution_D_to_A redistribution_D_to_B redistribution_D_to_C : ℚ)
  (redistribution_E_to_A redistribution_E_to_B redistribution_E_to_C : ℚ)
  (majority_difference : ℕ)
  (h1 : initial_votes_A = 35/100)
  (h2 : initial_votes_B = 25/100)
  (h3 : initial_votes_C = 20/100)
  (h4 : initial_votes_D = 15/100)
  (h5 : initial_votes_E = 5/100)
  (h6 : redistribution_D_to_A = 60/100)
  (h7 : redistribution_D_to_B = 25/100)
  (h8 : redistribution_D_to_C = 15/100)
  (h9 : redistribution_E_to_A = 50/100)
  (h10 : redistribution_E_to_B = 30/100)
  (h11 : redistribution_E_to_C = 20/100)
  (h12 : majority_difference = 1890) :
  ∃ (total_votes : ℕ),
    total_votes = 11631 ∧
    (initial_votes_A + redistribution_D_to_A * initial_votes_D + redistribution_E_to_A * initial_votes_E) * total_votes -
    (initial_votes_B + redistribution_D_to_B * initial_votes_D + redistribution_E_to_B * initial_votes_E) * total_votes =
    majority_difference := by
  sorry


end ranked_choice_voting_theorem_l3037_303747


namespace intersection_of_lines_l3037_303745

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (-1/4, -1/4)

/-- First line equation -/
def line1 (x y : ℚ) : Prop := y = 5 * x + 1

/-- Second line equation -/
def line2 (x y : ℚ) : Prop := 2 * y - 3 = -6 * x

theorem intersection_of_lines :
  let (x, y) := intersection_point
  (line1 x y ∧ line2 x y) ∧
  ∀ x' y', (line1 x' y' ∧ line2 x' y') → (x' = x ∧ y' = y) := by
  sorry

end intersection_of_lines_l3037_303745


namespace f_extrema_on_interval_l3037_303753

def f (x : ℝ) : ℝ := 2 * x^3 + 3 * x^2 - 36 * x + 1

theorem f_extrema_on_interval :
  let a : ℝ := 1
  let b : ℝ := 11
  ∀ x ∈ Set.Icc a b, 
    f x ≥ -43 ∧ f x ≤ 2630 ∧ 
    (∃ x₁ ∈ Set.Icc a b, f x₁ = -43) ∧ 
    (∃ x₂ ∈ Set.Icc a b, f x₂ = 2630) := by
  sorry

end f_extrema_on_interval_l3037_303753


namespace quarters_count_l3037_303724

/-- Proves that given 21 coins consisting of nickels and quarters with a total value of $3.65, the number of quarters is 13. -/
theorem quarters_count (total_coins : ℕ) (total_value : ℚ) (nickels : ℕ) (quarters : ℕ) : 
  total_coins = 21 →
  total_value = 365/100 →
  total_coins = nickels + quarters →
  total_value = (5 * nickels + 25 * quarters) / 100 →
  quarters = 13 := by
  sorry

end quarters_count_l3037_303724


namespace marks_tomatoes_l3037_303737

/-- Given that Mark bought tomatoes at $5 per pound and 5 pounds of apples at $6 per pound,
    spending a total of $40, prove that he bought 2 pounds of tomatoes. -/
theorem marks_tomatoes :
  ∀ (tomato_price apple_price : ℝ) (apple_pounds : ℝ) (total_spent : ℝ),
    tomato_price = 5 →
    apple_price = 6 →
    apple_pounds = 5 →
    total_spent = 40 →
    ∃ (tomato_pounds : ℝ),
      tomato_pounds * tomato_price + apple_pounds * apple_price = total_spent ∧
      tomato_pounds = 2 :=
by sorry

end marks_tomatoes_l3037_303737


namespace no_infinite_sequence_with_sqrt_difference_l3037_303781

theorem no_infinite_sequence_with_sqrt_difference :
  ¬∃ (x : ℕ → ℝ), 
    (∀ n, x n > 0) ∧ 
    (∀ n, x (n + 2) = Real.sqrt (x (n + 1)) - Real.sqrt (x n)) := by
  sorry

end no_infinite_sequence_with_sqrt_difference_l3037_303781


namespace choose_and_assign_officers_l3037_303732

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The factorial of a natural number -/
def factorial (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to choose 3 people from 5 and assign them to 3 distinct roles -/
def waysToChooseAndAssign : ℕ := choose 5 3 * factorial 3

theorem choose_and_assign_officers :
  waysToChooseAndAssign = 60 := by
  sorry

end choose_and_assign_officers_l3037_303732


namespace solution_k_l3037_303757

theorem solution_k (h : 2 * k - (-4) = 2) : k = -1 := by
  sorry

end solution_k_l3037_303757


namespace total_length_QP_PL_l3037_303730

-- Define the triangle XYZ
def X : ℝ × ℝ := (1, 4)
def Y : ℝ × ℝ := (0, 0)
def Z : ℝ × ℝ := (3, 0)

-- Define the altitudes
def XK : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = X.1 ∧ p.2 ≤ X.2}
def YL : Set (ℝ × ℝ) := {p : ℝ × ℝ | (Z.1 - X.1) * (p.1 - X.1) = (Z.2 - X.2) * (p.2 - X.2)}

-- Define the angle bisectors
def ZD : Set (ℝ × ℝ) := {p : ℝ × ℝ | (X.1 - Z.1) * (p.2 - Z.2) = (X.2 - Z.2) * (p.1 - Z.1)}
def XE : Set (ℝ × ℝ) := {p : ℝ × ℝ | (Y.1 - X.1) * (p.2 - X.2) = (Y.2 - X.2) * (p.1 - X.1)}

-- Define Q and P
def Q : ℝ × ℝ := (1, 1)
noncomputable def P : ℝ × ℝ := (0.5, 3)

-- Theorem statement
theorem total_length_QP_PL : 
  let qp_length := Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)
  let pl_length := Real.sqrt ((P.1 - (3/4))^2 + (P.2 - 3)^2)
  qp_length + pl_length = 1.5 := by sorry

end total_length_QP_PL_l3037_303730


namespace lunch_theorem_l3037_303766

def lunch_problem (total_spent friend_spent : ℕ) : Prop :=
  friend_spent > total_spent - friend_spent ∧
  friend_spent - (total_spent - friend_spent) = 1

theorem lunch_theorem :
  lunch_problem 15 8 := by
  sorry

end lunch_theorem_l3037_303766


namespace infinitely_many_primes_congruent_one_mod_power_of_two_l3037_303720

theorem infinitely_many_primes_congruent_one_mod_power_of_two (r : ℕ) (hr : r ≥ 1) :
  Set.Infinite {p : ℕ | Nat.Prime p ∧ p ≡ 1 [MOD 2^r]} := by
  sorry

end infinitely_many_primes_congruent_one_mod_power_of_two_l3037_303720


namespace fishing_problem_l3037_303794

theorem fishing_problem (a b c d : ℕ) : 
  a + b + c + d = 25 →
  a > b ∧ b > c ∧ c > d →
  a = b + c →
  b = c + d →
  (a = 11 ∧ b = 7 ∧ c = 4 ∧ d = 3) := by
sorry

end fishing_problem_l3037_303794


namespace triangle_AOB_properties_l3037_303708

def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (3, 1)

def OA : ℝ × ℝ := A
def OB : ℝ × ℝ := B
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

theorem triangle_AOB_properties :
  let magnitude_AB := Real.sqrt ((AB.1)^2 + (AB.2)^2)
  let dot_product_AB_OA := AB.1 * OA.1 + AB.2 * OA.2
  let cos_angle_OA_OB := (OA.1 * OB.1 + OA.2 * OB.2) / 
    (Real.sqrt (OA.1^2 + OA.2^2) * Real.sqrt (OB.1^2 + OB.2^2))
  (magnitude_AB = Real.sqrt 5) ∧
  (dot_product_AB_OA = 0) ∧
  (cos_angle_OA_OB = Real.sqrt 2 / 2) := by
  sorry

end triangle_AOB_properties_l3037_303708


namespace dunkers_lineup_count_l3037_303741

/-- The number of players in the team -/
def team_size : ℕ := 15

/-- The number of players who refuse to play together -/
def excluded_players : ℕ := 3

/-- The size of a starting lineup -/
def lineup_size : ℕ := 5

/-- The number of ways to select a lineup from the team, excluding the combinations
    where the excluded players play together -/
def valid_lineups : ℕ := 2277

theorem dunkers_lineup_count :
  (excluded_players * (Nat.choose (team_size - excluded_players) (lineup_size - 1))) +
  (Nat.choose (team_size - excluded_players) lineup_size) = valid_lineups :=
sorry

end dunkers_lineup_count_l3037_303741


namespace pentagon_centroid_intersection_l3037_303760

/-- Given a convex pentagon ABCDE in a real vector space, prove that the point P defined as
    (1/5)(A + B + C + D + E) is the intersection point of all segments connecting the midpoint
    of each side to the centroid of the triangle formed by the other three vertices. -/
theorem pentagon_centroid_intersection
  (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]
  (A B C D E : V) :
  let P := (1/5 : ℝ) • (A + B + C + D + E)
  let midpoint (X Y : V) := (1/2 : ℝ) • (X + Y)
  let centroid (X Y Z : V) := (1/3 : ℝ) • (X + Y + Z)
  ∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 →
    (midpoint A B + t • (centroid C D E - midpoint A B) = P) ∧
    (midpoint B C + t • (centroid D E A - midpoint B C) = P) ∧
    (midpoint C D + t • (centroid E A B - midpoint C D) = P) ∧
    (midpoint D E + t • (centroid A B C - midpoint D E) = P) ∧
    (midpoint E A + t • (centroid B C D - midpoint E A) = P) :=
by sorry


end pentagon_centroid_intersection_l3037_303760


namespace harry_apples_l3037_303775

/-- Proves that Harry has 19 apples given the conditions of the problem -/
theorem harry_apples :
  ∀ (martha_apples tim_apples harry_apples jane_apples : ℕ),
  martha_apples = 68 →
  tim_apples = martha_apples - 30 →
  harry_apples = tim_apples / 2 →
  jane_apples = ((martha_apples + tim_apples) * 25) / 100 →
  harry_apples = 19 := by
  sorry

#check harry_apples

end harry_apples_l3037_303775


namespace geometric_sequence_101st_term_l3037_303726

def geometric_sequence (a : ℕ → ℝ) := ∃ (r : ℝ), ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_101st_term
  (a : ℕ → ℝ)
  (h_geo : geometric_sequence a)
  (h_3rd : a 3 = 3)
  (h_sum : a 2016 + a 2017 = 0) :
  a 101 = 3 := by
sorry

end geometric_sequence_101st_term_l3037_303726


namespace valid_galaxish_words_remainder_l3037_303752

/-- Represents the set of letters in Galaxish --/
inductive GalaxishLetter
| S
| T
| U

/-- Represents a Galaxish word as a list of letters --/
def GalaxishWord := List GalaxishLetter

/-- Checks if a letter is a consonant --/
def is_consonant (l : GalaxishLetter) : Bool :=
  match l with
  | GalaxishLetter.S => true
  | GalaxishLetter.T => true
  | GalaxishLetter.U => false

/-- Checks if a Galaxish word is valid --/
def is_valid_galaxish_word (word : GalaxishWord) : Bool :=
  let rec check (w : GalaxishWord) (consonant_count : Nat) : Bool :=
    match w with
    | [] => true
    | l::ls => 
      if is_consonant l then
        check ls (consonant_count + 1)
      else if consonant_count >= 3 then
        check ls 0
      else
        false
  check word 0

/-- Counts the number of valid 8-letter Galaxish words --/
def count_valid_galaxish_words : Nat :=
  sorry

theorem valid_galaxish_words_remainder :
  count_valid_galaxish_words % 1000 = 56 := by sorry

end valid_galaxish_words_remainder_l3037_303752


namespace tangent_line_at_one_monotonicity_intervals_negative_condition_l3037_303703

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x - a * x

theorem tangent_line_at_one (h : ℝ → ℝ) :
  (∀ x > 0, h x = f (-2) x) →
  (∀ x, h x = -x + 1) →
  ∃ c, h c = f (-2) c ∧ (∀ x, h x - (f (-2) c) = -(x - c)) :=
sorry

theorem monotonicity_intervals (a : ℝ) :
  (a ≤ 0 → ∀ x y, 0 < x → 0 < y → x < y → f a x < f a y) ∧
  (a > 0 → ∀ x y, 0 < x → x < 1/a → x < y → f a x < f a y) ∧
  (a > 0 → ∀ x y, 1/a < x → x < y → f a y < f a x) :=
sorry

theorem negative_condition (a : ℝ) :
  (∀ x > 0, f a x < 0) ↔ a > 1/exp 1 :=
sorry

end tangent_line_at_one_monotonicity_intervals_negative_condition_l3037_303703


namespace even_function_implies_a_equals_four_l3037_303770

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The function f(x) = (x + a)(x - 4) -/
def f (a : ℝ) (x : ℝ) : ℝ := (x + a) * (x - 4)

/-- If f(x) = (x + a)(x - 4) is an even function, then a = 4 -/
theorem even_function_implies_a_equals_four :
  ∀ a : ℝ, IsEven (f a) → a = 4 := by
  sorry

end even_function_implies_a_equals_four_l3037_303770


namespace fraction_sum_l3037_303707

theorem fraction_sum : (3 : ℚ) / 8 + (9 : ℚ) / 12 = (9 : ℚ) / 8 := by
  sorry

end fraction_sum_l3037_303707


namespace missed_both_equiv_l3037_303786

-- Define propositions
variable (p q : Prop)

-- Define the meaning of "missed the target on both shots"
def missed_both (p q : Prop) : Prop := (¬p) ∧ (¬q)

-- Theorem: "missed the target on both shots" is equivalent to ¬(p ∨ q)
theorem missed_both_equiv (p q : Prop) : missed_both p q ↔ ¬(p ∨ q) := by
  sorry

end missed_both_equiv_l3037_303786


namespace notebook_cost_l3037_303767

/-- Represents the problem of determining the cost of notebooks --/
theorem notebook_cost (total_students : ℕ) 
  (buyers : ℕ) 
  (notebooks_per_buyer : ℕ) 
  (cost_per_notebook : ℕ) 
  (total_cost : ℕ) :
  total_students = 36 →
  buyers > total_students / 2 →
  buyers ≤ total_students →
  cost_per_notebook > notebooks_per_buyer →
  buyers * notebooks_per_buyer * cost_per_notebook = total_cost →
  total_cost = 2275 →
  cost_per_notebook = 13 :=
by sorry

end notebook_cost_l3037_303767


namespace expression_evaluation_l3037_303796

theorem expression_evaluation (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
  ((((x - 2)^2 * (x^2 + x + 1)^2) / (x^3 - 1)^2)^2 * 
   (((x + 2)^2 * (x^2 - x + 1)^2) / (x^3 + 1)^2)^2) = (x^2 - 4)^4 := by
  sorry

end expression_evaluation_l3037_303796


namespace jason_games_last_month_l3037_303779

/-- Calculates the number of games Jason planned to attend last month -/
def games_planned_last_month (games_this_month games_missed games_attended : ℕ) : ℕ :=
  (games_attended + games_missed) - games_this_month

theorem jason_games_last_month :
  games_planned_last_month 11 16 12 = 17 := by
  sorry

end jason_games_last_month_l3037_303779


namespace offspring_different_genes_l3037_303736

structure Eukaryote where
  genes : Set String

def sexualReproduction (parent1 parent2 : Eukaryote) : Eukaryote :=
  sorry

theorem offspring_different_genes (parent1 parent2 : Eukaryote) :
  let offspring := sexualReproduction parent1 parent2
  ∃ (gene : String), (gene ∈ offspring.genes ∧ gene ∉ parent1.genes) ∨
                     (gene ∈ offspring.genes ∧ gene ∉ parent2.genes) :=
  sorry

end offspring_different_genes_l3037_303736


namespace equal_area_rectangles_length_l3037_303768

/-- Given two rectangles of equal area, where one rectangle has dimensions 4 inches by 30 inches,
    and the other has a width of 24 inches, prove that the length of the second rectangle is 5 inches. -/
theorem equal_area_rectangles_length (area : ℝ) (width : ℝ) :
  area = 4 * 30 →
  width = 24 →
  area = width * 5 :=
by sorry

end equal_area_rectangles_length_l3037_303768


namespace finley_class_size_l3037_303773

/-- The number of students in Mrs. Finley's class -/
def finley_class : ℕ := sorry

/-- The number of students in Mr. Johnson's class -/
def johnson_class : ℕ := 22

/-- Mr. Johnson's class has 10 more than half the number in Mrs. Finley's class -/
axiom johnson_class_size : johnson_class = finley_class / 2 + 10

theorem finley_class_size : finley_class = 24 := by sorry

end finley_class_size_l3037_303773


namespace complex_quadrant_l3037_303734

theorem complex_quadrant (a b : ℝ) (z : ℂ) :
  z = a + b * Complex.I →
  (a / (1 - Complex.I) + b / (1 - 2 * Complex.I) = 5 / (3 + Complex.I)) →
  0 < a ∧ b < 0 := by
  sorry

end complex_quadrant_l3037_303734


namespace second_smallest_packs_l3037_303746

/-- The number of hot dogs in each pack -/
def hot_dogs_per_pack : ℕ := 10

/-- The number of buns in each pack -/
def buns_per_pack : ℕ := 8

/-- The number of hot dogs left over -/
def leftover_hot_dogs : ℕ := 4

/-- A function that checks if a given number of packs satisfies the condition -/
def satisfies_condition (n : ℕ) : Prop :=
  (hot_dogs_per_pack * n) % buns_per_pack = leftover_hot_dogs

/-- The theorem stating that 6 is the second smallest number of packs satisfying the condition -/
theorem second_smallest_packs : 
  ∃ (m : ℕ), m < 6 ∧ satisfies_condition m ∧ 
  (∀ (k : ℕ), k < m → ¬satisfies_condition k) ∧
  (∀ (k : ℕ), m < k → k < 6 → ¬satisfies_condition k) ∧
  satisfies_condition 6 := by
  sorry

end second_smallest_packs_l3037_303746


namespace correct_prices_l3037_303718

/-- Represents the purchase and sale of golden passion fruit -/
structure GoldenPassionFruit where
  first_batch_cost : ℝ
  second_batch_cost : ℝ
  weight_ratio : ℝ
  price_difference : ℝ
  profit_margin : ℝ

/-- Calculates the unit prices and minimum selling price for golden passion fruit -/
def calculate_prices (gpf : GoldenPassionFruit) : 
  (ℝ × ℝ × ℝ) :=
  let first_batch_price := 20
  let second_batch_price := first_batch_price - gpf.price_difference
  let min_selling_price := 25
  (first_batch_price, second_batch_price, min_selling_price)

/-- Theorem stating the correctness of the calculated prices -/
theorem correct_prices (gpf : GoldenPassionFruit) 
  (h1 : gpf.first_batch_cost = 3600)
  (h2 : gpf.second_batch_cost = 5400)
  (h3 : gpf.weight_ratio = 2)
  (h4 : gpf.price_difference = 5)
  (h5 : gpf.profit_margin = 0.5) :
  let (first_price, second_price, min_price) := calculate_prices gpf
  first_price = 20 ∧ 
  second_price = 15 ∧ 
  min_price ≥ 25 ∧
  min_price * (gpf.first_batch_cost / first_price + gpf.second_batch_cost / second_price) ≥ 
    (gpf.first_batch_cost + gpf.second_batch_cost) * (1 + gpf.profit_margin) :=
by sorry


end correct_prices_l3037_303718


namespace polar_to_rectangular_conversion_l3037_303738

theorem polar_to_rectangular_conversion :
  let r : ℝ := 6
  let θ : ℝ := 5 * Real.pi / 3
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  x = 3 ∧ y = -3 * Real.sqrt 3 := by
  sorry

end polar_to_rectangular_conversion_l3037_303738


namespace students_not_excelling_l3037_303731

theorem students_not_excelling (total : ℕ) (basketball : ℕ) (soccer : ℕ) (both : ℕ) : 
  total = 40 → basketball = 12 → soccer = 18 → both = 6 → 
  total - (basketball + soccer - both) = 16 := by
  sorry

end students_not_excelling_l3037_303731


namespace unique_n_value_l3037_303722

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem unique_n_value (m n : ℕ) 
  (h1 : m > 0)
  (h2 : is_three_digit n)
  (h3 : Nat.lcm m n = 690)
  (h4 : ¬(3 ∣ n))
  (h5 : ¬(2 ∣ m)) :
  n = 230 := by
sorry

end unique_n_value_l3037_303722


namespace fifth_day_distance_l3037_303739

/-- Represents the daily walking distance of a man -/
def walkingSequence (firstDay : ℕ) (dailyIncrease : ℕ) : ℕ → ℕ :=
  fun n => firstDay + (n - 1) * dailyIncrease

theorem fifth_day_distance
  (firstDay : ℕ)
  (dailyIncrease : ℕ)
  (h1 : firstDay = 100)
  (h2 : (Finset.range 9).sum (walkingSequence firstDay dailyIncrease) = 1260) :
  walkingSequence firstDay dailyIncrease 5 = 140 := by
  sorry

#check fifth_day_distance

end fifth_day_distance_l3037_303739


namespace gina_credits_l3037_303735

/-- Proves that given the conditions of Gina's college expenses, she is taking 14 credits -/
theorem gina_credits : 
  let credit_cost : ℕ := 450
  let textbook_count : ℕ := 5
  let textbook_cost : ℕ := 120
  let facilities_fee : ℕ := 200
  let total_cost : ℕ := 7100
  ∃ (credits : ℕ), 
    credits * credit_cost + 
    textbook_count * textbook_cost + 
    facilities_fee = total_cost ∧ 
    credits = 14 :=
by sorry

end gina_credits_l3037_303735


namespace sqrt_product_equality_l3037_303744

theorem sqrt_product_equality : Real.sqrt 50 * Real.sqrt 18 * Real.sqrt 8 = 60 * Real.sqrt 2 := by
  sorry

end sqrt_product_equality_l3037_303744


namespace water_volume_in_cylinder_l3037_303791

theorem water_volume_in_cylinder (r : ℝ) (h : r = 2) : 
  let cylinder_base_area := π * r^2
  let ball_volume := (4/3) * π * r^3
  let water_height_with_ball := 2 * r
  let total_volume_with_ball := cylinder_base_area * water_height_with_ball
  let original_water_volume := total_volume_with_ball - ball_volume
  original_water_volume = (16 * π) / 3 := by
sorry

end water_volume_in_cylinder_l3037_303791


namespace lemonade_ratio_l3037_303714

/-- Given that 36 lemons make 48 gallons of lemonade, this theorem proves that 4.5 lemons are needed for 6 gallons of lemonade. -/
theorem lemonade_ratio (lemons : ℝ) (gallons : ℝ) 
  (h1 : lemons / gallons = 36 / 48) 
  (h2 : gallons = 6) : 
  lemons = 4.5 := by
  sorry

end lemonade_ratio_l3037_303714


namespace min_value_theorem_l3037_303750

theorem min_value_theorem (a : ℝ) (ha : a > 0) :
  a + (a + 4) / a ≥ 5 ∧ ∃ a₀ > 0, a₀ + (a₀ + 4) / a₀ = 5 :=
by sorry

end min_value_theorem_l3037_303750


namespace two_digit_numbers_satisfying_condition_l3037_303711

def sumOfDigits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def isTwoDigit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

def satisfiesCondition (n : ℕ) : Prop :=
  isTwoDigit n ∧ Nat.Prime (n - 7 * sumOfDigits n)

theorem two_digit_numbers_satisfying_condition :
  {n : ℕ | satisfiesCondition n} = {10, 31, 52, 73, 94} := by
  sorry

end two_digit_numbers_satisfying_condition_l3037_303711


namespace angle_in_first_quadrant_l3037_303748

theorem angle_in_first_quadrant (α : Real) 
  (h1 : Real.tan α > 0) 
  (h2 : Real.sin α + Real.cos α > 0) : 
  0 < α ∧ α < Real.pi / 2 := by
  sorry

end angle_in_first_quadrant_l3037_303748


namespace binary_1011_is_11_l3037_303758

def binary_to_decimal (b : List Bool) : ℕ :=
  List.foldl (λ acc d => 2 * acc + if d then 1 else 0) 0 b

theorem binary_1011_is_11 :
  binary_to_decimal [true, false, true, true] = 11 := by
  sorry

end binary_1011_is_11_l3037_303758


namespace flowers_to_grandma_vs_mom_l3037_303701

theorem flowers_to_grandma_vs_mom (total : ℕ) (to_mom : ℕ) (in_vase : ℕ) :
  total = 52 →
  to_mom = 15 →
  in_vase = 16 →
  total - to_mom - in_vase - to_mom = 6 := by
  sorry

end flowers_to_grandma_vs_mom_l3037_303701


namespace housing_relocation_problem_l3037_303795

/-- Represents the housing relocation problem -/
theorem housing_relocation_problem 
  (household_area : ℝ) 
  (initial_green_space_ratio : ℝ)
  (final_green_space_ratio : ℝ)
  (additional_households : ℕ)
  (min_green_space_ratio : ℝ)
  (h1 : household_area = 150)
  (h2 : initial_green_space_ratio = 0.4)
  (h3 : final_green_space_ratio = 0.15)
  (h4 : additional_households = 20)
  (h5 : min_green_space_ratio = 0.2) :
  ∃ (initial_households : ℕ) (total_area : ℝ) (withdraw_households : ℕ),
    initial_households = 48 ∧ 
    total_area = 12000 ∧ 
    withdraw_households ≥ 4 ∧
    total_area - household_area * initial_households = initial_green_space_ratio * total_area ∧
    total_area - household_area * (initial_households + additional_households) = final_green_space_ratio * total_area ∧
    total_area - household_area * (initial_households + additional_households - withdraw_households) ≥ min_green_space_ratio * total_area :=
by sorry

end housing_relocation_problem_l3037_303795


namespace absolute_value_problem_l3037_303772

theorem absolute_value_problem (a b c : ℝ) 
  (ha : |a| = 5)
  (hb : |b| = 3)
  (hc : |c| = 6)
  (hab : |a+b| = -(a+b))
  (hac : |a+c| = a+c) :
  a - b + c = 4 ∨ a - b + c = -2 :=
by
  sorry

end absolute_value_problem_l3037_303772


namespace boat_speed_correct_l3037_303780

/-- The speed of the boat in still water -/
def boat_speed : ℝ := 15

/-- The speed of the stream -/
def stream_speed : ℝ := 3

/-- The time taken to travel downstream -/
def downstream_time : ℝ := 1

/-- The time taken to travel upstream -/
def upstream_time : ℝ := 1.5

/-- Theorem stating that the boat speed is correct given the conditions -/
theorem boat_speed_correct :
  ∃ (distance : ℝ),
    distance = (boat_speed + stream_speed) * downstream_time ∧
    distance = (boat_speed - stream_speed) * upstream_time :=
by
  sorry

end boat_speed_correct_l3037_303780


namespace students_liking_both_sea_and_mountains_l3037_303717

theorem students_liking_both_sea_and_mountains 
  (total_students : ℕ)
  (sea_lovers : ℕ)
  (mountain_lovers : ℕ)
  (neither_lovers : ℕ)
  (h1 : total_students = 500)
  (h2 : sea_lovers = 337)
  (h3 : mountain_lovers = 289)
  (h4 : neither_lovers = 56) :
  sea_lovers + mountain_lovers - (total_students - neither_lovers) = 182 := by
  sorry

end students_liking_both_sea_and_mountains_l3037_303717


namespace fishing_trip_cost_l3037_303754

/-- The fishing trip cost problem -/
theorem fishing_trip_cost (alice_paid bob_paid chris_paid : ℝ) 
  (h1 : alice_paid = 135)
  (h2 : bob_paid = 165)
  (h3 : chris_paid = 225)
  (x y : ℝ) 
  (h4 : x = (alice_paid + bob_paid + chris_paid) / 3 - alice_paid)
  (h5 : y = (alice_paid + bob_paid + chris_paid) / 3 - bob_paid) :
  x - y = 30 := by
sorry

end fishing_trip_cost_l3037_303754


namespace arithmetic_expression_equals_28_l3037_303778

theorem arithmetic_expression_equals_28 : 12 / 4 - 3 - 10 + 3 * 10 + 2^3 = 28 := by
  sorry

end arithmetic_expression_equals_28_l3037_303778


namespace two_fruits_from_five_l3037_303727

/-- The number of ways to choose k items from a set of n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of fruit types available -/
def num_fruits : ℕ := 5

/-- The number of fruits to be chosen -/
def fruits_to_choose : ℕ := 2

theorem two_fruits_from_five :
  choose num_fruits fruits_to_choose = 10 := by
  sorry

end two_fruits_from_five_l3037_303727


namespace store_payback_time_l3037_303763

/-- Calculates the time required to pay back an initial investment given monthly revenue and expenses -/
def payback_time (initial_cost : ℕ) (monthly_revenue : ℕ) (monthly_expenses : ℕ) : ℕ :=
  let monthly_profit := monthly_revenue - monthly_expenses
  initial_cost / monthly_profit

theorem store_payback_time :
  payback_time 25000 4000 1500 = 10 := by
  sorry

end store_payback_time_l3037_303763


namespace people_who_left_train_l3037_303733

/-- The number of people who left a train given initial, boarding, and final passenger counts. -/
theorem people_who_left_train (initial : ℕ) (boarded : ℕ) (final : ℕ) : 
  initial = 82 → boarded = 17 → final = 73 → initial + boarded - final = 26 := by
  sorry

end people_who_left_train_l3037_303733


namespace remainder_mod_88_l3037_303728

theorem remainder_mod_88 : (1 - 90) ^ 10 ≡ 1 [MOD 88] := by sorry

end remainder_mod_88_l3037_303728


namespace nates_matches_l3037_303700

/-- The number of matches Nate started with -/
def initial_matches : ℕ := 70

/-- The number of matches Nate dropped in the creek -/
def dropped_matches : ℕ := 10

/-- The number of matches eaten by the dog -/
def eaten_matches : ℕ := 2 * dropped_matches

/-- The number of matches Nate has left -/
def remaining_matches : ℕ := 40

theorem nates_matches :
  initial_matches = remaining_matches + dropped_matches + eaten_matches :=
by sorry

end nates_matches_l3037_303700


namespace sum_of_powers_inequality_l3037_303729

theorem sum_of_powers_inequality (x : ℝ) (hx : x > 0) :
  1 + x + x^2 + x^3 + x^4 ≥ 5 * x^2 := by
  sorry

end sum_of_powers_inequality_l3037_303729


namespace inequality_proof_l3037_303784

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (x + y) / (2 + x + y) < x / (2 + x) + y / (2 + y) := by
  sorry

end inequality_proof_l3037_303784


namespace solve_equation_l3037_303702

theorem solve_equation (x : ℝ) : 3 * x = (62 - x) + 26 → x = 22 := by
  sorry

end solve_equation_l3037_303702


namespace union_subset_iff_m_nonpositive_no_m_exists_for_equality_l3037_303713

-- Define the sets P and S
def P : Set ℝ := {x | x^2 - 8*x - 20 ≤ 0}
def S (m : ℝ) : Set ℝ := {x | |x - 1| ≤ m}

-- Question 1
theorem union_subset_iff_m_nonpositive (m : ℝ) :
  (P ∪ S m) ⊆ P ↔ m ≤ 0 :=
sorry

-- Question 2
theorem no_m_exists_for_equality :
  ¬∃ m : ℝ, P = S m :=
sorry

end union_subset_iff_m_nonpositive_no_m_exists_for_equality_l3037_303713


namespace mrs_hilt_has_more_money_l3037_303777

-- Define the value of each coin type
def penny_value : ℚ := 0.01
def nickel_value : ℚ := 0.05
def dime_value : ℚ := 0.10

-- Define the number of coins each person has
def mrs_hilt_pennies : ℕ := 2
def mrs_hilt_nickels : ℕ := 2
def mrs_hilt_dimes : ℕ := 2

def jacob_pennies : ℕ := 4
def jacob_nickels : ℕ := 1
def jacob_dimes : ℕ := 1

-- Calculate the total amount for each person
def mrs_hilt_total : ℚ :=
  mrs_hilt_pennies * penny_value +
  mrs_hilt_nickels * nickel_value +
  mrs_hilt_dimes * dime_value

def jacob_total : ℚ :=
  jacob_pennies * penny_value +
  jacob_nickels * nickel_value +
  jacob_dimes * dime_value

-- State the theorem
theorem mrs_hilt_has_more_money :
  mrs_hilt_total - jacob_total = 0.13 := by
  sorry

end mrs_hilt_has_more_money_l3037_303777


namespace sum_equals_negative_six_l3037_303743

theorem sum_equals_negative_six (a b c d : ℤ) :
  (∃ x : ℤ, a + 2 = x ∧ b + 3 = x ∧ c + 4 = x ∧ d + 5 = x ∧ a + b + c + d + 8 = x) →
  a + b + c + d = -6 := by
sorry

end sum_equals_negative_six_l3037_303743


namespace f_decreasing_after_one_l3037_303712

def f (x : ℝ) : ℝ := -(x - 1)^2 + 2

theorem f_decreasing_after_one :
  ∀ x₁ x₂ : ℝ, 1 < x₁ → x₁ < x₂ → f x₂ < f x₁ := by
  sorry

end f_decreasing_after_one_l3037_303712


namespace treasure_chest_rubies_l3037_303721

/-- Given a treasure chest with gems, calculate the number of rubies. -/
theorem treasure_chest_rubies (total_gems diamonds : ℕ) (h1 : total_gems = 5155) (h2 : diamonds = 45) :
  total_gems - diamonds = 5110 := by
  sorry

end treasure_chest_rubies_l3037_303721


namespace line_curve_intersection_implies_m_geq_3_l3037_303776

-- Define the line equation
def line (k x : ℝ) : ℝ := k * x - k + 1

-- Define the curve equation
def curve (x y m : ℝ) : Prop := x^2 + 2 * y^2 = m

-- Theorem statement
theorem line_curve_intersection_implies_m_geq_3 (k m : ℝ) :
  (∃ x y : ℝ, line k x = y ∧ curve x y m) → m ≥ 3 := by sorry

end line_curve_intersection_implies_m_geq_3_l3037_303776


namespace student_scores_average_l3037_303705

theorem student_scores_average (math physics chem : ℕ) : 
  math + physics = 30 → 
  chem = physics + 20 → 
  (math + chem) / 2 = 25 := by
sorry

end student_scores_average_l3037_303705


namespace cupcake_package_size_l3037_303798

/-- The number of cupcakes in the smaller package -/
def smaller_package : ℕ := 10

/-- The number of cupcakes in the larger package -/
def larger_package : ℕ := 15

/-- The number of packs of each size bought -/
def packs_bought : ℕ := 4

/-- The total number of children to receive cupcakes -/
def total_children : ℕ := 100

theorem cupcake_package_size :
  packs_bought * larger_package + packs_bought * smaller_package = total_children :=
sorry

end cupcake_package_size_l3037_303798


namespace probability_same_result_is_seven_twentyfourths_l3037_303740

/-- Represents a 12-sided die with specific colored sides -/
structure TwelveSidedDie :=
  (purple : Nat)
  (green : Nat)
  (orange : Nat)
  (glittery : Nat)
  (total_sides : Nat)
  (is_valid : purple + green + orange + glittery = total_sides)

/-- Calculates the probability of two dice showing the same result -/
def probability_same_result (d : TwelveSidedDie) : Rat :=
  let p_purple := (d.purple : Rat) / d.total_sides
  let p_green := (d.green : Rat) / d.total_sides
  let p_orange := (d.orange : Rat) / d.total_sides
  let p_glittery := (d.glittery : Rat) / d.total_sides
  p_purple * p_purple + p_green * p_green + p_orange * p_orange + p_glittery * p_glittery

/-- Theorem: The probability of two 12-sided dice with specific colored sides showing the same result is 7/24 -/
theorem probability_same_result_is_seven_twentyfourths :
  let d : TwelveSidedDie := {
    purple := 3,
    green := 4,
    orange := 4,
    glittery := 1,
    total_sides := 12,
    is_valid := by simp
  }
  probability_same_result d = 7 / 24 := by
  sorry


end probability_same_result_is_seven_twentyfourths_l3037_303740


namespace cube_minus_self_div_by_six_l3037_303792

theorem cube_minus_self_div_by_six (n : ℕ) : 6 ∣ (n^3 - n) := by
  sorry

end cube_minus_self_div_by_six_l3037_303792
