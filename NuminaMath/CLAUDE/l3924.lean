import Mathlib

namespace elon_has_13_teslas_l3924_392447

/-- The number of teslas Chris has -/
def chris_teslas : ℕ := 6

/-- The number of teslas Sam has -/
def sam_teslas : ℕ := chris_teslas / 2

/-- The number of teslas Elon has -/
def elon_teslas : ℕ := sam_teslas + 10

/-- Theorem stating that Elon has 13 teslas -/
theorem elon_has_13_teslas : elon_teslas = 13 := by
  sorry

end elon_has_13_teslas_l3924_392447


namespace root_sum_theorem_l3924_392459

def cubic_equation (x : ℝ) : Prop := 60 * x^3 - 70 * x^2 + 24 * x - 2 = 0

theorem root_sum_theorem (p q r : ℝ) :
  cubic_equation p ∧ cubic_equation q ∧ cubic_equation r ∧
  p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
  0 < p ∧ p < 2 ∧ 0 < q ∧ q < 2 ∧ 0 < r ∧ r < 2 →
  1 / (2 - p) + 1 / (2 - q) + 1 / (2 - r) = 116 / 15 :=
by sorry

end root_sum_theorem_l3924_392459


namespace beth_has_winning_strategy_l3924_392465

/-- Represents the state of a wall of bricks -/
structure Wall :=
  (bricks : ℕ)

/-- Represents the game state with multiple walls -/
structure GameState :=
  (walls : List Wall)

/-- Calculates the nim-value of a single wall -/
def nimValue (w : Wall) : ℕ :=
  sorry

/-- Calculates the combined nim-value of a game state -/
def combinedNimValue (gs : GameState) : ℕ :=
  sorry

/-- Determines if a given game state is a winning position for the current player -/
def isWinningPosition (gs : GameState) : Prop :=
  combinedNimValue gs ≠ 0

/-- The initial game state -/
def initialState : GameState :=
  { walls := [{ bricks := 7 }, { bricks := 3 }, { bricks := 2 }] }

theorem beth_has_winning_strategy :
  ¬ isWinningPosition initialState :=
sorry

end beth_has_winning_strategy_l3924_392465


namespace smaller_integer_problem_l3924_392462

theorem smaller_integer_problem (x y : ℤ) : 
  y = 2 * x → x + y = 96 → x = 32 := by
  sorry

end smaller_integer_problem_l3924_392462


namespace base_conversion_arithmetic_l3924_392483

/-- Converts a number from base b to base 10 --/
def toBase10 (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun d acc => d + b * acc) 0

/-- Rounds a rational number to the nearest integer --/
def roundToNearest (x : ℚ) : ℤ :=
  (x + 1/2).floor

theorem base_conversion_arithmetic : 
  let base8_2468 := toBase10 [8, 6, 4, 2] 8
  let base4_110 := toBase10 [0, 1, 1] 4
  let base9_3571 := toBase10 [1, 7, 5, 3] 9
  let base10_1357 := 1357
  roundToNearest (base8_2468 / base4_110) - base9_3571 + base10_1357 = -1232 := by
  sorry

end base_conversion_arithmetic_l3924_392483


namespace geometric_sequence_first_term_l3924_392460

/-- Given a geometric sequence where the last four terms are a, b, 243, 729,
    prove that the first term of the sequence is 3. -/
theorem geometric_sequence_first_term
  (a b : ℝ)
  (h1 : ∃ (r : ℝ), r ≠ 0 ∧ b = a * r ∧ 243 = b * r ∧ 729 = 243 * r)
  : ∃ (n : ℕ), 3 * (a / 243) ^ n = 1 :=
by sorry

end geometric_sequence_first_term_l3924_392460


namespace equation_solution_range_l3924_392437

theorem equation_solution_range (m : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ (x + m) / (x - 3) + (3 * m) / (3 - x) = 3) →
  m < 9 / 2 ∧ m ≠ 3 / 2 :=
by sorry

end equation_solution_range_l3924_392437


namespace quadratic_equations_solutions_l3924_392457

theorem quadratic_equations_solutions :
  -- Equation 1
  (∃ x : ℝ, x^2 - 4 = 0) ∧
  (∀ x : ℝ, x^2 - 4 = 0 → x = 2 ∨ x = -2) ∧
  -- Equation 2
  (∃ x : ℝ, x^2 - 6*x + 9 = 0) ∧
  (∀ x : ℝ, x^2 - 6*x + 9 = 0 → x = 3) ∧
  -- Equation 3
  (∃ x : ℝ, x^2 - 7*x + 12 = 0) ∧
  (∀ x : ℝ, x^2 - 7*x + 12 = 0 → x = 3 ∨ x = 4) ∧
  -- Equation 4
  (∃ x : ℝ, 2*x^2 - 3*x = 5) ∧
  (∀ x : ℝ, 2*x^2 - 3*x = 5 → x = 5/2 ∨ x = -1) := by
  sorry


end quadratic_equations_solutions_l3924_392457


namespace prob_k_white_balls_correct_l3924_392429

/-- The probability of drawing exactly k white balls from an urn containing n white and n black balls,
    when drawing n balls in total. -/
def prob_k_white_balls (n k : ℕ) : ℚ :=
  (Nat.choose n k)^2 / Nat.choose (2*n) n

/-- Theorem stating that the probability of drawing exactly k white balls from an urn
    containing n white balls and n black balls, when drawing n balls in total,
    is equal to (n choose k)^2 / (2n choose n). -/
theorem prob_k_white_balls_correct (n k : ℕ) (h : k ≤ n) :
  prob_k_white_balls n k = (Nat.choose n k)^2 / Nat.choose (2*n) n :=
by sorry

end prob_k_white_balls_correct_l3924_392429


namespace triangle_reflection_slope_l3924_392469

/-- Triangle DEF with vertices D(3,2), E(5,4), and F(2,6) reflected across y=2x -/
theorem triangle_reflection_slope (D E F D' E' F' : ℝ × ℝ) :
  D = (3, 2) →
  E = (5, 4) →
  F = (2, 6) →
  D' = (1, 3/2) →
  E' = (2, 5/2) →
  F' = (3, 1) →
  (D'.2 - D.2) / (D'.1 - D.1) ≠ -1/2 :=
by sorry

end triangle_reflection_slope_l3924_392469


namespace divisibility_of_p_l3924_392421

theorem divisibility_of_p (p q r s : ℕ+) 
  (h1 : Nat.gcd p.val q.val = 40)
  (h2 : Nat.gcd q.val r.val = 50)
  (h3 : Nat.gcd r.val s.val = 75)
  (h4 : 80 < Nat.gcd s.val p.val)
  (h5 : Nat.gcd s.val p.val < 120) :
  5 ∣ p.val := by
  sorry

end divisibility_of_p_l3924_392421


namespace grape_lollipops_count_l3924_392476

/-- Given a total number of lollipops and the number of flavors for non-cherry lollipops,
    calculate the number of lollipops of a specific non-cherry flavor. -/
def grape_lollipops (total : ℕ) (non_cherry_flavors : ℕ) : ℕ :=
  (total / 2) / non_cherry_flavors

/-- Theorem stating that with 42 total lollipops and 3 non-cherry flavors,
    the number of grape lollipops is 7. -/
theorem grape_lollipops_count :
  grape_lollipops 42 3 = 7 := by
  sorry

end grape_lollipops_count_l3924_392476


namespace rotation_equivalence_l3924_392439

/-- Given that a point A is rotated 450 degrees clockwise and y degrees counterclockwise
    about the same center point B, both rotations resulting in the same final position C,
    and y < 360, prove that y = 270. -/
theorem rotation_equivalence (y : ℝ) : 
  (450 % 360 : ℝ) = (360 - y) % 360 → y < 360 → y = 270 := by sorry

end rotation_equivalence_l3924_392439


namespace trig_fraction_equality_l3924_392441

theorem trig_fraction_equality (x : ℝ) (h : (1 + Real.sin x) / Real.cos x = -1/2) :
  Real.cos x / (Real.sin x - 1) = 1/2 := by
  sorry

end trig_fraction_equality_l3924_392441


namespace bankers_gain_example_l3924_392435

/-- Calculate the banker's gain given the banker's discount, time, and interest rate. -/
def bankers_gain (bankers_discount : ℚ) (time : ℚ) (rate : ℚ) : ℚ :=
  let face_value := (bankers_discount * 100) / (rate * time)
  let true_discount := (face_value * rate * time) / (100 + rate * time)
  bankers_discount - true_discount

/-- Theorem stating that the banker's gain is 360 given the specified conditions. -/
theorem bankers_gain_example : 
  bankers_gain 1360 3 12 = 360 := by
  sorry

end bankers_gain_example_l3924_392435


namespace february_monthly_fee_calculation_l3924_392423

/-- Represents the monthly membership fee and per-class fee structure -/
structure FeeStructure where
  monthly_fee : ℝ
  per_class_fee : ℝ

/-- Calculates the total bill given a fee structure and number of classes -/
def total_bill (fs : FeeStructure) (classes : ℕ) : ℝ :=
  fs.monthly_fee + fs.per_class_fee * classes

/-- Represents the fee structure with a 10% increase in monthly fee -/
def increased_fee_structure (fs : FeeStructure) : FeeStructure :=
  { monthly_fee := 1.1 * fs.monthly_fee
    per_class_fee := fs.per_class_fee }

theorem february_monthly_fee_calculation 
  (feb_fs : FeeStructure)
  (h1 : total_bill feb_fs 4 = 30.72)
  (h2 : total_bill (increased_fee_structure feb_fs) 8 = 54.72) :
  feb_fs.monthly_fee = 7.47 := by
  sorry

#eval (7.47 : Float).toString

end february_monthly_fee_calculation_l3924_392423


namespace cooler_cans_count_l3924_392418

/-- Given a cooler with cherry soda and orange pop, where there are twice as many
    cans of orange pop as cherry soda, and there are 8 cherry sodas,
    prove that the total number of cans in the cooler is 24. -/
theorem cooler_cans_count (cherry_soda orange_pop : ℕ) : 
  cherry_soda = 8 →
  orange_pop = 2 * cherry_soda →
  cherry_soda + orange_pop = 24 := by
  sorry

end cooler_cans_count_l3924_392418


namespace factorial_of_factorial_divided_by_factorial_l3924_392417

theorem factorial_of_factorial_divided_by_factorial :
  (Nat.factorial (Nat.factorial 4)) / (Nat.factorial 4) = 25852016738884976640000 := by
  sorry

end factorial_of_factorial_divided_by_factorial_l3924_392417


namespace min_reciprocal_81_l3924_392495

/-- The reciprocal function -/
def reciprocal (x : ℚ) : ℚ := 1 / x

/-- Apply the reciprocal function n times -/
def apply_reciprocal (x : ℚ) (n : ℕ) : ℚ :=
  match n with
  | 0 => x
  | n + 1 => reciprocal (apply_reciprocal x n)

/-- Theorem: The minimum number of times to apply the reciprocal function to 81 to return to 81 is 2 -/
theorem min_reciprocal_81 :
  (∃ n : ℕ, apply_reciprocal 81 n = 81 ∧ n > 0) ∧
  (∀ m : ℕ, m > 0 ∧ m < 2 → apply_reciprocal 81 m ≠ 81) ∧
  apply_reciprocal 81 2 = 81 :=
sorry

end min_reciprocal_81_l3924_392495


namespace swapped_divisible_by_37_l3924_392458

/-- Represents a nine-digit number split into two parts -/
structure SplitNumber where
  x : ℕ
  y : ℕ
  k : ℕ
  h1 : k > 0
  h2 : k < 10

/-- The original nine-digit number -/
def originalNumber (n : SplitNumber) : ℕ :=
  n.x * 10^(9 - n.k) + n.y

/-- The swapped nine-digit number -/
def swappedNumber (n : SplitNumber) : ℕ :=
  n.y * 10^n.k + n.x

/-- Theorem stating that if the original number is divisible by 37,
    then the swapped number is also divisible by 37 -/
theorem swapped_divisible_by_37 (n : SplitNumber) :
  37 ∣ originalNumber n → 37 ∣ swappedNumber n := by
  sorry


end swapped_divisible_by_37_l3924_392458


namespace sufficient_to_necessary_contrapositive_l3924_392491

theorem sufficient_to_necessary_contrapositive (A B : Prop) 
  (h : A → B) : ¬B → ¬A := by sorry

end sufficient_to_necessary_contrapositive_l3924_392491


namespace henry_tic_tac_toe_games_l3924_392468

theorem henry_tic_tac_toe_games (wins losses draws : ℕ) 
  (h_wins : wins = 2)
  (h_losses : losses = 2)
  (h_draws : draws = 10) :
  wins + losses + draws = 14 := by
  sorry

end henry_tic_tac_toe_games_l3924_392468


namespace polynomial_with_negative_integer_roots_l3924_392408

/-- A polynomial of degree 4 with integer coefficients -/
structure Polynomial4 where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ

/-- The polynomial function corresponding to a Polynomial4 -/
def poly_func (p : Polynomial4) : ℝ → ℝ :=
  fun x ↦ x^4 + p.a * x^3 + p.b * x^2 + p.c * x + p.d

/-- Predicate stating that all roots of a polynomial are negative integers -/
def all_roots_negative_integers (p : Polynomial4) : Prop :=
  ∀ x : ℝ, poly_func p x = 0 → (∃ n : ℤ, x = ↑n ∧ n < 0)

theorem polynomial_with_negative_integer_roots
  (p : Polynomial4)
  (h_roots : all_roots_negative_integers p)
  (h_sum : p.a + p.b + p.c + p.d = 2009) :
  p.d = 528 := by
  sorry

end polynomial_with_negative_integer_roots_l3924_392408


namespace f_properties_l3924_392484

noncomputable def f (x : ℝ) := (1 + Real.sqrt 3 * Real.tan x) * (Real.cos x)^2

theorem f_properties :
  (∀ x : ℝ, f x ≠ 0 → ∃ k : ℤ, x ≠ Real.pi / 2 + k * Real.pi) ∧
  (∀ x : ℝ, f (x + Real.pi) = f x) ∧
  (∀ x : ℝ, x ∈ Set.Ioo 0 (Real.pi / 2) → f x ∈ Set.Ioc 0 (3 / 2)) :=
by sorry

end f_properties_l3924_392484


namespace andy_late_demerits_l3924_392482

/-- The maximum number of demerits Andy can get before being fired -/
def max_demerits : ℕ := 50

/-- The number of times Andy showed up late -/
def late_instances : ℕ := 6

/-- The number of demerits Andy got for making an inappropriate joke -/
def joke_demerits : ℕ := 15

/-- The number of additional demerits Andy can get this month before being fired -/
def remaining_demerits : ℕ := 23

/-- The number of demerits Andy gets per instance of being late -/
def demerits_per_late_instance : ℕ := 2

theorem andy_late_demerits :
  late_instances * demerits_per_late_instance + joke_demerits = max_demerits - remaining_demerits :=
sorry

end andy_late_demerits_l3924_392482


namespace rectangle_area_rectangle_area_is_140_l3924_392473

theorem rectangle_area (square_area : ℝ) (rectangle_breadth : ℝ) : ℝ :=
  let square_side : ℝ := Real.sqrt square_area
  let circle_radius : ℝ := square_side
  let rectangle_length : ℝ := (2 / 5) * circle_radius
  let rectangle_area : ℝ := rectangle_length * rectangle_breadth
  rectangle_area

theorem rectangle_area_is_140 :
  rectangle_area 1225 10 = 140 := by
  sorry

end rectangle_area_rectangle_area_is_140_l3924_392473


namespace parallel_transitivity_l3924_392456

-- Define a type for lines in a plane
def Line : Type := ℝ → ℝ → Prop

-- Define a relation for parallel lines
def Parallel (l₁ l₂ : Line) : Prop := sorry

-- Theorem statement
theorem parallel_transitivity (l₁ l₂ l₃ : Line) :
  Parallel l₁ l₃ → Parallel l₂ l₃ → Parallel l₁ l₂ :=
sorry

end parallel_transitivity_l3924_392456


namespace polynomial_simplification_l3924_392453

theorem polynomial_simplification (x : ℝ) :
  (3 * x^3 + 4 * x^2 - 5 * x + 8) - (2 * x^3 + x^2 + 3 * x - 15) = x^3 + 3 * x^2 - 8 * x + 23 := by
  sorry

end polynomial_simplification_l3924_392453


namespace range_encoding_l3924_392426

/-- Represents a coding scheme for words -/
structure CodeScheme where
  random : Nat
  rand : Nat

/-- Defines the coding for a word given a CodeScheme -/
def encode (scheme : CodeScheme) (word : String) : Nat :=
  sorry

/-- Theorem: Given the coding scheme where 'random' is 123678 and 'rand' is 1236,
    the code for 'range' is 12378 -/
theorem range_encoding (scheme : CodeScheme)
    (h1 : scheme.random = 123678)
    (h2 : scheme.rand = 1236) :
    encode scheme "range" = 12378 :=
  sorry

end range_encoding_l3924_392426


namespace power_exceeds_any_number_l3924_392403

theorem power_exceeds_any_number (p M : ℝ) (hp : p > 0) (hM : M > 0) :
  ∃ n : ℕ, (1 + p)^n > M := by sorry

end power_exceeds_any_number_l3924_392403


namespace max_sections_five_lines_l3924_392404

/- Define a function that calculates the maximum number of sections -/
def max_sections (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 2
  else 2 + (n - 1) * n / 2

/- Theorem statement -/
theorem max_sections_five_lines :
  max_sections 5 = 16 :=
by sorry

end max_sections_five_lines_l3924_392404


namespace team_x_games_l3924_392489

/-- Prove that Team X played 24 games given the conditions -/
theorem team_x_games (x : ℕ) 
  (h1 : (3 : ℚ) / 4 * x = x - (1 : ℚ) / 4 * x)  -- Team X wins 3/4 of its games
  (h2 : (2 : ℚ) / 3 * (x + 9) = (x + 9) - (1 : ℚ) / 3 * (x + 9))  -- Team Y wins 2/3 of its games
  (h3 : (2 : ℚ) / 3 * (x + 9) = (3 : ℚ) / 4 * x + 4)  -- Team Y won 4 more games than Team X
  : x = 24 := by
  sorry

end team_x_games_l3924_392489


namespace fish_count_l3924_392488

/-- The number of fish Lilly has -/
def lilly_fish : ℕ := 10

/-- The number of fish Rosy has -/
def rosy_fish : ℕ := 9

/-- The total number of fish Lilly and Rosy have together -/
def total_fish : ℕ := lilly_fish + rosy_fish

theorem fish_count : total_fish = 19 := by
  sorry

end fish_count_l3924_392488


namespace quintic_polynomial_minimum_value_l3924_392428

/-- A quintic polynomial with real coefficients -/
def QuinticPolynomial (P : ℝ → ℝ) : Prop :=
  ∃ a b c d e f : ℝ, ∀ x, P x = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f

/-- All complex roots of P have magnitude 1 -/
def AllRootsOnUnitCircle (P : ℂ → ℂ) : Prop :=
  ∀ z : ℂ, P z = 0 → Complex.abs z = 1

theorem quintic_polynomial_minimum_value (P : ℝ → ℝ) 
  (h_quintic : QuinticPolynomial P)
  (h_P0 : P 0 = 2)
  (h_P1 : P 1 = 3)
  (h_roots : AllRootsOnUnitCircle (fun z => P z.re)) :
  (∀ Q : ℝ → ℝ, QuinticPolynomial Q → Q 0 = 2 → Q 1 = 3 → 
    AllRootsOnUnitCircle (fun z => Q z.re) → P 2 ≤ Q 2) ∧ P 2 = 54 := by
  sorry

end quintic_polynomial_minimum_value_l3924_392428


namespace student_selection_problem_l3924_392431

theorem student_selection_problem (n_male : ℕ) (n_female : ℕ) (n_select : ℕ) (n_competitions : ℕ) :
  n_male = 5 →
  n_female = 4 →
  n_select = 3 →
  n_competitions = 2 →
  (Nat.choose (n_male + n_female) n_select - Nat.choose n_male n_select - Nat.choose n_female n_select) *
  (Nat.factorial n_select) = 420 := by
  sorry

end student_selection_problem_l3924_392431


namespace games_played_calculation_l3924_392467

/-- Represents the number of games played by a baseball team --/
def GamesPlayed : ℕ → ℕ → ℕ → ℕ
  | games_won, games_left, games_to_win_more => games_won + games_left - games_to_win_more

/-- Represents the total number of games in a season --/
def TotalGames : ℕ → ℕ → ℕ
  | games_played, games_left => games_played + games_left

theorem games_played_calculation (games_won : ℕ) (games_left : ℕ) (games_to_win_more : ℕ) :
  games_won = 12 →
  games_left = 10 →
  games_to_win_more = 8 →
  (3 * (games_won + games_to_win_more) = 2 * TotalGames (GamesPlayed games_won games_left games_to_win_more) games_left) →
  GamesPlayed games_won games_left games_to_win_more = 20 := by
  sorry

end games_played_calculation_l3924_392467


namespace ellipse_condition_l3924_392411

/-- The equation of the graph -/
def graph_equation (x y k : ℝ) : Prop :=
  3 * x^2 + 9 * y^2 - 6 * x + 27 * y = k

/-- The condition for a non-degenerate ellipse -/
def is_non_degenerate_ellipse (k : ℝ) : Prop :=
  k > -93/4

/-- Theorem: The graph is a non-degenerate ellipse iff k > -93/4 -/
theorem ellipse_condition :
  ∀ k, (∃ x y, graph_equation x y k) ↔ is_non_degenerate_ellipse k :=
by sorry

end ellipse_condition_l3924_392411


namespace natural_solutions_count_l3924_392416

theorem natural_solutions_count :
  ∃! (s : Finset (ℕ × ℕ)), s.card = 4 ∧ ∀ (x y : ℕ), (x, y) ∈ s ↔ 2 * x + y = 7 := by
  sorry

end natural_solutions_count_l3924_392416


namespace abc_value_l3924_392407

theorem abc_value (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a + 1/b = 5)
  (h2 : b + 1/c = 2)
  (h3 : c + 1/a = 9/4) :
  a * b * c = (7 + Real.sqrt 21) / 8 := by
sorry

end abc_value_l3924_392407


namespace area_maximized_at_m_pm1_l3924_392448

/-- Ellipse E with equation x²/6 + y²/2 = 1 -/
def ellipse_E (x y : ℝ) : Prop := x^2 / 6 + y^2 / 2 = 1

/-- Focus F₁ at (-2, 0) -/
def F₁ : ℝ × ℝ := (-2, 0)

/-- Line l with equation x - my - 2 = 0 -/
def line_l (m x y : ℝ) : Prop := x - m * y - 2 = 0

/-- Intersection points of ellipse E and line l -/
def intersection_points (m : ℝ) : Set (ℝ × ℝ) :=
  {p | ellipse_E p.1 p.2 ∧ line_l m p.1 p.2}

/-- Area of quadrilateral AF₁BC -/
noncomputable def area_AF₁BC (m : ℝ) : ℝ := sorry

/-- Theorem: Area of AF₁BC is maximized when m = ±1 -/
theorem area_maximized_at_m_pm1 :
  ∀ m : ℝ, area_AF₁BC m ≤ area_AF₁BC 1 ∧ area_AF₁BC m ≤ area_AF₁BC (-1) :=
sorry

end area_maximized_at_m_pm1_l3924_392448


namespace simplest_quadratic_radical_example_l3924_392438

def is_simplest_quadratic_radical (n : ℝ) : Prop :=
  ∃ (a : ℕ), n = a ∧ ¬∃ (b : ℕ), b * b = a ∧ b > 1

theorem simplest_quadratic_radical_example : 
  ∃ (x : ℝ), is_simplest_quadratic_radical (x + 3) ∧ x = 2 := by
  sorry

#check simplest_quadratic_radical_example

end simplest_quadratic_radical_example_l3924_392438


namespace unique_root_quadratic_l3924_392420

theorem unique_root_quadratic (k : ℝ) : 
  (∃! a : ℝ, (k^2 - 9) * a^2 - 2 * (k + 1) * a + 1 = 0) → 
  (k = 3 ∨ k = -3 ∨ k = -5) :=
by sorry

end unique_root_quadratic_l3924_392420


namespace original_sequence_reappearance_l3924_392412

/-- The cycle length of the letter sequence -/
def letter_cycle_length : ℕ := 8

/-- The cycle length of the digit sequence -/
def digit_cycle_length : ℕ := 5

/-- The line number where the original sequence reappears -/
def reappearance_line : ℕ := 40

theorem original_sequence_reappearance :
  Nat.lcm letter_cycle_length digit_cycle_length = reappearance_line :=
by sorry

end original_sequence_reappearance_l3924_392412


namespace exponent_multiplication_calculate_expression_l3924_392479

theorem exponent_multiplication (a : ℕ) (m n : ℕ) : 
  a * (a ^ n) = a ^ (n + 1) :=
by sorry

theorem calculate_expression : 3000 * (3000 ^ 2500) = 3000 ^ 2501 :=
by sorry

end exponent_multiplication_calculate_expression_l3924_392479


namespace conic_section_eccentricity_l3924_392490

/-- Given three numbers 2, m, 8 forming a geometric sequence, 
    the eccentricity of the conic section x^2/m + y^2/2 = 1 is either √2/2 or √3 -/
theorem conic_section_eccentricity (m : ℝ) :
  (2 * m = m * 8) →
  let e := if m > 0 then Real.sqrt (1 - 2 / m) else Real.sqrt (1 + m / 2)
  e = Real.sqrt 2 / 2 ∨ e = Real.sqrt 3 := by
  sorry

end conic_section_eccentricity_l3924_392490


namespace sum_seven_smallest_multiples_of_12_l3924_392492

theorem sum_seven_smallest_multiples_of_12 : 
  (Finset.range 7).sum (fun i => 12 * (i + 1)) = 336 := by
  sorry

end sum_seven_smallest_multiples_of_12_l3924_392492


namespace surface_area_of_sliced_prism_l3924_392466

/-- A right prism with equilateral triangular bases -/
structure RightPrism where
  base_side_length : ℝ
  height : ℝ

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The solid CXYZ formed by slicing the prism -/
structure SolidCXYZ where
  prism : RightPrism
  C : Point3D
  X : Point3D
  Y : Point3D
  Z : Point3D

/-- Function to calculate the surface area of SolidCXYZ -/
def surface_area_CXYZ (solid : SolidCXYZ) : ℝ :=
  sorry

/-- Theorem statement -/
theorem surface_area_of_sliced_prism (solid : SolidCXYZ) 
  (h1 : solid.prism.base_side_length = 12)
  (h2 : solid.prism.height = 16)
  (h3 : solid.X.x = (solid.C.x + solid.prism.base_side_length / 2))
  (h4 : solid.Y.x = (solid.C.x + solid.prism.base_side_length))
  (h5 : solid.Z.z = (solid.C.z + solid.prism.height / 2)) :
  surface_area_CXYZ solid = 48 + 9 * Real.sqrt 3 + 3 * Real.sqrt 91 :=
sorry

end surface_area_of_sliced_prism_l3924_392466


namespace smaller_circle_with_integer_points_l3924_392430

/-- Given a circle centered at the origin with radius R, there exists a circle
    with radius R/√2 that contains at least as many points with integer coordinates. -/
theorem smaller_circle_with_integer_points (R : ℝ) (R_pos : R > 0) :
  ∃ (R' : ℝ), R' = R / Real.sqrt 2 ∧
  (∀ (x y : ℤ), x^2 + y^2 ≤ R^2 →
    ∃ (x' y' : ℤ), x'^2 + y'^2 ≤ R'^2) :=
by sorry

end smaller_circle_with_integer_points_l3924_392430


namespace moses_percentage_l3924_392485

theorem moses_percentage (total : ℝ) (moses_amount : ℝ) (esther_amount : ℝ) : 
  total = 50 ∧
  moses_amount = esther_amount + 5 ∧
  moses_amount + 2 * esther_amount = total →
  moses_amount / total = 0.4 := by
  sorry

end moses_percentage_l3924_392485


namespace fraction_equality_l3924_392449

theorem fraction_equality (x : ℝ) (h : x / (x^2 + x - 1) = 1 / 7) :
  x^2 / (x^4 - x^2 + 1) = 1 / 37 := by
  sorry

end fraction_equality_l3924_392449


namespace fraction_equals_121_l3924_392406

theorem fraction_equals_121 : (1100^2 : ℚ) / (260^2 - 240^2) = 121 := by sorry

end fraction_equals_121_l3924_392406


namespace cost_per_set_is_correct_l3924_392480

/-- The cost of each set of drill bits -/
def cost_per_set : ℝ := 6

/-- The number of sets bought -/
def num_sets : ℕ := 5

/-- The tax rate -/
def tax_rate : ℝ := 0.1

/-- The total amount paid -/
def total_paid : ℝ := 33

/-- Theorem stating that the cost per set is correct given the conditions -/
theorem cost_per_set_is_correct : 
  num_sets * cost_per_set * (1 + tax_rate) = total_paid :=
sorry

end cost_per_set_is_correct_l3924_392480


namespace product_terminal_zeros_l3924_392410

/-- The number of terminal zeros in a positive integer -/
def terminalZeros (n : ℕ) : ℕ := sorry

/-- The product of 50 and 480 -/
def product : ℕ := 50 * 480

theorem product_terminal_zeros :
  terminalZeros product = 3 := by sorry

end product_terminal_zeros_l3924_392410


namespace logarithm_expression_equals_two_l3924_392451

-- Define the base 10 logarithm
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem logarithm_expression_equals_two :
  lg 4 + lg 9 + 2 * Real.sqrt ((lg 6)^2 - lg 36 + 1) = 2 := by
  sorry

end logarithm_expression_equals_two_l3924_392451


namespace statue_increase_factor_l3924_392494

/-- The factor by which the number of statues increased in the second year --/
def increase_factor : ℝ := 4

/-- The initial number of statues --/
def initial_statues : ℕ := 4

/-- The number of statues added in the third year --/
def added_third_year : ℕ := 12

/-- The number of statues broken in the third year --/
def broken_third_year : ℕ := 3

/-- The final number of statues after four years --/
def final_statues : ℕ := 31

theorem statue_increase_factor : 
  (initial_statues : ℝ) * increase_factor + 
  (added_third_year : ℝ) - (broken_third_year : ℝ) + 
  2 * (broken_third_year : ℝ) = (final_statues : ℝ) := by
  sorry

end statue_increase_factor_l3924_392494


namespace negative_64_to_four_thirds_equals_256_l3924_392481

theorem negative_64_to_four_thirds_equals_256 : (-64 : ℝ) ^ (4/3) = 256 := by
  sorry

end negative_64_to_four_thirds_equals_256_l3924_392481


namespace cooper_pies_per_day_l3924_392436

/-- The number of days Cooper makes pies -/
def days : ℕ := 12

/-- The number of pies Ashley eats -/
def pies_eaten : ℕ := 50

/-- The number of pies remaining -/
def pies_remaining : ℕ := 34

/-- The number of pies Cooper makes per day -/
def pies_per_day : ℕ := 7

theorem cooper_pies_per_day :
  days * pies_per_day - pies_eaten = pies_remaining :=
by sorry

end cooper_pies_per_day_l3924_392436


namespace flight_duration_sum_l3924_392477

-- Define the flight departure and arrival times in minutes since midnight
def departure_time : ℕ := 10 * 60 + 34
def arrival_time : ℕ := 13 * 60 + 18

-- Define the flight duration in hours and minutes
def flight_duration (h m : ℕ) : Prop :=
  h * 60 + m = arrival_time - departure_time ∧ 0 < m ∧ m < 60

-- Theorem statement
theorem flight_duration_sum :
  ∃ (h m : ℕ), flight_duration h m ∧ h + m = 46 :=
sorry

end flight_duration_sum_l3924_392477


namespace point_outside_circle_l3924_392400

theorem point_outside_circle (m : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + 4*m*x - 2*y + 5*m = 0 → (x - 1)^2 + (y - 1)^2 > 0) ↔ 
  (0 < m ∧ m < 1/4) ∨ m > 1 := by sorry

end point_outside_circle_l3924_392400


namespace max_d_is_one_l3924_392422

/-- The sequence a_n defined as (10^n - 1) / 9 -/
def a (n : ℕ) : ℕ := (10^n - 1) / 9

/-- The greatest common divisor of a_n and a_{n+1} -/
def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

/-- Theorem: The maximum value of d_n is 1 -/
theorem max_d_is_one : ∀ n : ℕ, d n = 1 := by sorry

end max_d_is_one_l3924_392422


namespace exists_g_for_f_l3924_392478

-- Define the function f: ℝ² → ℝ
variable (f : ℝ × ℝ → ℝ)

-- State the condition for f
axiom f_condition : ∀ (x y z : ℝ), f (x, y) + f (y, z) + f (z, x) = 0

-- Theorem statement
theorem exists_g_for_f : 
  ∃ (g : ℝ → ℝ), ∀ (x y : ℝ), f (x, y) = g x - g y := by sorry

end exists_g_for_f_l3924_392478


namespace hyperbola_asymptote_slope_l3924_392454

/-- A hyperbola with equation x^2 - y^2/b^2 = 1 where b > 0 -/
structure Hyperbola where
  b : ℝ
  h_pos : b > 0

/-- The asymptote of a hyperbola is parallel to a line -/
def asymptote_parallel (h : Hyperbola) (m : ℝ) : Prop :=
  ∃ (c : ℝ), ∀ (x y : ℝ), y = m * x + c → (x^2 - y^2 / h.b^2 = 1 → False)

theorem hyperbola_asymptote_slope (h : Hyperbola) 
  (parallel : asymptote_parallel h 2) : h.b = 2 := by
  sorry

end hyperbola_asymptote_slope_l3924_392454


namespace helen_cookies_l3924_392413

/-- The number of chocolate chip cookies Helen baked yesterday -/
def cookies_yesterday : ℕ := 1081 - 554

/-- The total number of chocolate chip cookies Helen baked -/
def total_cookies : ℕ := 1081

/-- The number of chocolate chip cookies Helen baked this morning -/
def cookies_this_morning : ℕ := 554

theorem helen_cookies : cookies_yesterday = 527 := by
  sorry

end helen_cookies_l3924_392413


namespace distribute_seven_balls_four_boxes_l3924_392401

/-- Represents the number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 7 distinguishable balls into 4 indistinguishable boxes is 495 -/
theorem distribute_seven_balls_four_boxes : distribute_balls 7 4 = 495 := by sorry

end distribute_seven_balls_four_boxes_l3924_392401


namespace root_domain_implies_a_bound_l3924_392474

/-- The equation has real roots for m in this set -/
def A : Set ℝ := {m | ∃ x, (m + 1) * x^2 - m * x + m - 1 = 0}

/-- The domain of the function f(x) -/
def B (a : ℝ) : Set ℝ := {x | x^2 - (a + 2) * x + 2 * a > 0}

/-- The main theorem -/
theorem root_domain_implies_a_bound (a : ℝ) : A ⊆ B a → a > 2/3 * Real.sqrt 3 := by
  sorry

end root_domain_implies_a_bound_l3924_392474


namespace sufficiency_not_necessity_l3924_392493

theorem sufficiency_not_necessity (p q : Prop) : 
  (¬p ∧ ¬q → ¬(p ∧ q)) ∧ 
  ∃ (p q : Prop), ¬(p ∧ q) ∧ ¬(¬p ∧ ¬q) := by
sorry

end sufficiency_not_necessity_l3924_392493


namespace range_of_a_l3924_392486

/-- The function f(x) = |x+a| + |x-2| -/
def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x - 2|

/-- The solution set A for f(x) ≤ |x-4| -/
def A (a : ℝ) : Set ℝ := {x | f a x ≤ |x - 4|}

/-- Theorem stating the range of a given the conditions -/
theorem range_of_a :
  ∀ a : ℝ, (∀ x ∈ Set.Icc 1 2, x ∈ A a) → a ∈ Set.Icc (-3) 0 :=
by sorry

end range_of_a_l3924_392486


namespace inequality_solution_sets_l3924_392425

-- Define the types for our variables
variables {a b c : ℝ}

-- Define the solution set of the first inequality
def solution_set_1 : Set ℝ := {x | 1 ≤ x ∧ x ≤ 2}

-- Define the solution set of the second inequality
def solution_set_2 : Set ℝ := {x | x ≤ -1 ∨ x ≥ -1/2}

-- State the theorem
theorem inequality_solution_sets :
  (∀ x, ax^2 - b*x + c ≥ 0 ↔ x ∈ solution_set_1) →
  (∀ x, c*x^2 + b*x + a ≤ 0 ↔ x ∈ solution_set_2) :=
by sorry

end inequality_solution_sets_l3924_392425


namespace fifteenth_student_age_l3924_392440

theorem fifteenth_student_age 
  (total_students : Nat) 
  (average_age : ℝ) 
  (group1_size : Nat) 
  (group1_average : ℝ) 
  (group2_size : Nat) 
  (group2_average : ℝ) 
  (h1 : total_students = 15)
  (h2 : average_age = 15)
  (h3 : group1_size = 5)
  (h4 : group1_average = 14)
  (h5 : group2_size = 9)
  (h6 : group2_average = 16)
  (h7 : group1_size + group2_size + 1 = total_students) :
  ∃ (fifteenth_age : ℝ),
    fifteenth_age = total_students * average_age - (group1_size * group1_average + group2_size * group2_average) ∧
    fifteenth_age = 11 := by
  sorry

end fifteenth_student_age_l3924_392440


namespace problem_solution_l3924_392499

theorem problem_solution (x : ℝ) (h_pos : x > 0) 
  (h_eq : Real.sqrt (12 * x) * Real.sqrt (6 * x) * Real.sqrt (5 * x) * Real.sqrt (20 * x) = 20) : 
  x = (1 / 18) ^ (1 / 4) := by
  sorry

end problem_solution_l3924_392499


namespace inverse_graph_coordinate_sum_l3924_392432

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the theorem
theorem inverse_graph_coordinate_sum :
  (∃ (f : ℝ → ℝ), f 2 = 4 ∧ (∃ (x : ℝ), f⁻¹ x = 2 ∧ x / 4 = 1 / 2)) →
  (∃ (x y : ℝ), y = f⁻¹ x / 4 ∧ x + y = 9 / 2) :=
by sorry

end inverse_graph_coordinate_sum_l3924_392432


namespace min_value_of_expression_l3924_392427

-- Define the circles
def circle1 (x y a : ℝ) : Prop := x^2 + y^2 + 2*a*x + a^2 - 9 = 0
def circle2 (x y b : ℝ) : Prop := x^2 + y^2 - 4*b*y - 1 + 4*b^2 = 0

-- Define the theorem
theorem min_value_of_expression (a b : ℝ) 
  (h1 : ∃ x y, circle1 x y a)
  (h2 : ∃ x y, circle2 x y b)
  (h3 : ∃ t1 t2 t3 : ℝ × ℝ, t1 ≠ t2 ∧ t2 ≠ t3 ∧ t1 ≠ t3 ∧ 
    (∀ x y, circle1 x y a → (t1.1 * x + t1.2 * y = 1 ∨ t2.1 * x + t2.2 * y = 1 ∨ t3.1 * x + t3.2 * y = 1)) ∧
    (∀ x y, circle2 x y b → (t1.1 * x + t1.2 * y = 1 ∨ t2.1 * x + t2.2 * y = 1 ∨ t3.1 * x + t3.2 * y = 1)))
  (h4 : a ≠ 0)
  (h5 : b ≠ 0) :
  ∃ m : ℝ, m = 1 ∧ ∀ a b : ℝ, a ≠ 0 → b ≠ 0 → 4 / a^2 + 1 / b^2 ≥ m :=
by sorry

end min_value_of_expression_l3924_392427


namespace max_value_of_min_expression_l3924_392497

theorem max_value_of_min_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (⨅ x ∈ ({1/a, 2/b, 4/c, (a*b*c)^(1/3)} : Set ℝ), x) ≤ Real.sqrt 2 ∧ 
  ∃ (a' b' c' : ℝ), 0 < a' ∧ 0 < b' ∧ 0 < c' ∧ 
    (⨅ x ∈ ({1/a', 2/b', 4/c', (a'*b'*c')^(1/3)} : Set ℝ), x) = Real.sqrt 2 := by
  sorry

end max_value_of_min_expression_l3924_392497


namespace min_value_system_min_value_exact_l3924_392496

open Real

theorem min_value_system (x y z : ℝ) 
  (eq1 : 2 * cos x = 1 / tan y)
  (eq2 : 2 * sin y = tan z)
  (eq3 : cos z = 1 / tan x) :
  ∀ (a b c : ℝ), 
    (2 * cos a = 1 / tan b) → 
    (2 * sin b = tan c) → 
    (cos c = 1 / tan a) → 
    sin x + cos z ≤ sin a + cos c :=
by sorry

theorem min_value_exact (x y z : ℝ) 
  (eq1 : 2 * cos x = 1 / tan y)
  (eq2 : 2 * sin y = tan z)
  (eq3 : cos z = 1 / tan x) :
  ∃ (a b c : ℝ), 
    (2 * cos a = 1 / tan b) ∧ 
    (2 * sin b = tan c) ∧ 
    (cos c = 1 / tan a) ∧ 
    sin a + cos c = -5 * Real.sqrt 3 / 6 :=
by sorry

end min_value_system_min_value_exact_l3924_392496


namespace bee_multiplier_l3924_392445

/-- Given the number of bees seen on two consecutive days, 
    prove that the ratio of bees on the second day to the first day is 3 -/
theorem bee_multiplier (bees_day1 bees_day2 : ℕ) 
  (h1 : bees_day1 = 144) 
  (h2 : bees_day2 = 432) : 
  (bees_day2 : ℚ) / bees_day1 = 3 := by
  sorry

end bee_multiplier_l3924_392445


namespace greatest_integer_for_all_real_domain_l3924_392446

theorem greatest_integer_for_all_real_domain (a : ℤ) : 
  (∀ x : ℝ, (x^2 + a*x + 15 ≠ 0)) ↔ a ≤ 7 :=
by sorry

end greatest_integer_for_all_real_domain_l3924_392446


namespace parabola_intersection_theorem_l3924_392444

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the line
def line (k : ℝ) (x y : ℝ) : Prop := y - 1 = k * (x + 3)

-- Define the point on the parabola
def point_on_parabola (a : ℝ) : Prop := parabola 3 a ∧ (3 - 2)^2 + a^2 = 5^2

-- Theorem statement
theorem parabola_intersection_theorem (k : ℝ) :
  (∃! p : ℝ × ℝ, parabola p.1 p.2 ∧ line k p.1 p.2) ↔ k = 0 ∨ k = -1 ∨ k = 2/3 :=
sorry

end parabola_intersection_theorem_l3924_392444


namespace product_of_solutions_l3924_392424

theorem product_of_solutions (x : ℝ) : 
  (|18 / x + 4| = 3) → 
  (∃ y : ℝ, (|18 / y + 4| = 3) ∧ x * y = 324 / 7) :=
sorry

end product_of_solutions_l3924_392424


namespace parabola_reflection_y_axis_l3924_392409

/-- Represents a parabola in the form y = a(x - h)^2 + k --/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Reflects a parabola along the y-axis --/
def reflect_y (p : Parabola) : Parabola :=
  { a := p.a, h := -p.h, k := p.k }

theorem parabola_reflection_y_axis :
  let original := Parabola.mk 2 1 (-4)
  let reflected := reflect_y original
  reflected = Parabola.mk 2 (-1) (-4) := by sorry

end parabola_reflection_y_axis_l3924_392409


namespace interest_difference_implies_principal_l3924_392414

/-- Proves that given specific interest conditions, if the difference between compound and simple interest is 36, the principal is 3600. -/
theorem interest_difference_implies_principal : 
  let rate : ℝ := 10  -- Interest rate (%)
  let time : ℝ := 2   -- Time period in years
  let diff : ℝ := 36  -- Difference between compound and simple interest
  ∀ principal : ℝ,
    (principal * (1 + rate / 100) ^ time - principal) -  -- Compound interest
    (principal * rate * time / 100) =                    -- Simple interest
    diff →
    principal = 3600 := by
  sorry

end interest_difference_implies_principal_l3924_392414


namespace savings_amount_correct_l3924_392464

def calculate_savings (lightweight_price medium_price heavyweight_price : ℚ)
  (home_lightweight grandparents_medium_factor neighbor_heavyweight : ℕ)
  (dad_total dad_lightweight_percent dad_medium_percent dad_heavyweight_percent : ℚ) : ℚ :=
  let home_medium := home_lightweight * grandparents_medium_factor
  let dad_lightweight := dad_total * dad_lightweight_percent
  let dad_medium := dad_total * dad_medium_percent
  let dad_heavyweight := dad_total * dad_heavyweight_percent
  let total_amount := 
    lightweight_price * (home_lightweight + dad_lightweight) +
    medium_price * (home_medium + dad_medium) +
    heavyweight_price * (neighbor_heavyweight + dad_heavyweight)
  total_amount / 2

theorem savings_amount_correct :
  calculate_savings 0.15 0.25 0.35 12 3 46 250 0.5 0.3 0.2 = 41.45 :=
by sorry

end savings_amount_correct_l3924_392464


namespace satellite_sensor_upgrade_fraction_l3924_392463

theorem satellite_sensor_upgrade_fraction :
  ∀ (total_units : ℕ) (non_upgraded_per_unit : ℕ) (total_upgraded : ℕ),
    total_units = 24 →
    non_upgraded_per_unit * 4 = total_upgraded →
    (total_upgraded : ℚ) / (total_upgraded + total_units * non_upgraded_per_unit) = 1 / 7 := by
  sorry

end satellite_sensor_upgrade_fraction_l3924_392463


namespace parabola_equation_l3924_392433

/-- A parabola passing through points (0, 5) and (3, 2) -/
def Parabola (x y : ℝ) : Prop :=
  ∃ (b c : ℝ), y = x^2 + b*x + c ∧ 5 = c ∧ 2 = 9 + 3*b + c

/-- The specific parabola y = x^2 - 4x + 5 -/
def SpecificParabola (x y : ℝ) : Prop :=
  y = x^2 - 4*x + 5

theorem parabola_equation : ∀ x y : ℝ, Parabola x y ↔ SpecificParabola x y :=
sorry

end parabola_equation_l3924_392433


namespace meaningful_expression_l3924_392471

theorem meaningful_expression (a : ℝ) : 
  (∃ x : ℝ, x = (Real.sqrt (a + 1)) / (a - 2)) ↔ (a ≥ -1 ∧ a ≠ 2) := by
  sorry

end meaningful_expression_l3924_392471


namespace boys_in_basketball_camp_l3924_392450

theorem boys_in_basketball_camp (total : ℕ) (boy_ratio girl_ratio : ℕ) (boys girls : ℕ) : 
  total = 48 →
  boy_ratio = 3 →
  girl_ratio = 5 →
  boys + girls = total →
  boy_ratio * girls = girl_ratio * boys →
  boys = 18 :=
by
  sorry

end boys_in_basketball_camp_l3924_392450


namespace shaded_area_circles_l3924_392405

/-- The area of the shaded region formed by a larger circle and two smaller circles --/
theorem shaded_area_circles (R : ℝ) (h : R = 8) : 
  let r := R / 2
  let large_circle_area := π * R^2
  let small_circle_area := π * r^2
  large_circle_area - 2 * small_circle_area = 32 * π :=
by sorry

end shaded_area_circles_l3924_392405


namespace day_crew_fraction_of_boxes_l3924_392419

/-- Represents the fraction of boxes loaded by the day crew given the relative productivity
    and size of the night crew compared to the day crew. -/
theorem day_crew_fraction_of_boxes
  (night_worker_productivity : ℚ)  -- Productivity of night worker relative to day worker
  (night_crew_size : ℚ)            -- Size of night crew relative to day crew
  (h1 : night_worker_productivity = 1 / 4)
  (h2 : night_crew_size = 4 / 5) :
  (1 : ℚ) / (1 + night_worker_productivity * night_crew_size) = 5 / 6 := by
  sorry


end day_crew_fraction_of_boxes_l3924_392419


namespace line_tangent_to_ellipse_l3924_392498

/-- The line y = mx + 2 is tangent to the ellipse x² + y²/4 = 1 if and only if m² = 0 -/
theorem line_tangent_to_ellipse (m : ℝ) :
  (∃! p : ℝ × ℝ, p.1^2 + (p.2^2 / 4) = 1 ∧ p.2 = m * p.1 + 2) ↔ m^2 = 0 := by
  sorry

end line_tangent_to_ellipse_l3924_392498


namespace arc_length_120_degrees_l3924_392487

/-- Given a circle with circumference 90 meters and an arc subtended by a 120° central angle,
    prove that the length of the arc is 30 meters. -/
theorem arc_length_120_degrees (circle_circumference : ℝ) (central_angle : ℝ) (arc_length : ℝ) :
  circle_circumference = 90 →
  central_angle = 120 →
  arc_length = (central_angle / 360) * circle_circumference →
  arc_length = 30 := by
sorry

end arc_length_120_degrees_l3924_392487


namespace right_triangle_hypotenuse_l3924_392475

theorem right_triangle_hypotenuse (p q : ℝ) (hp : p > 0) (hq : q > 0) (hpq : q < p) (hpq2 : p < q * Real.sqrt 1.8) :
  ∃ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b = p ∧
    (1/3 : ℝ) * Real.sqrt (a^2 + 4*b^2) + (1/3 : ℝ) * Real.sqrt (4*a^2 + b^2) = q ∧
    c^2 = a^2 + b^2 ∧
    c^2 = (p^4 - 9*q^4) / (2*(p^2 - 5*q^2)) := by
  sorry

end right_triangle_hypotenuse_l3924_392475


namespace reciprocal_of_negative_three_l3924_392434

theorem reciprocal_of_negative_three :
  (1 : ℚ) / (-3 : ℚ) = -1/3 := by sorry

end reciprocal_of_negative_three_l3924_392434


namespace cascade_properties_l3924_392472

/-- A cascade generated by a natural number r -/
def Cascade (r : ℕ) : Finset ℕ :=
  Finset.image (λ i => i * r) (Finset.range 12)

/-- The property that a pair of natural numbers belongs to exactly six cascades -/
def BelongsToSixCascades (a b : ℕ) : Prop :=
  ∃ (cascades : Finset ℕ), cascades.card = 6 ∧
    ∀ r ∈ cascades, a ∈ Cascade r ∧ b ∈ Cascade r

/-- A coloring function from natural numbers to 12 colors -/
def ColoringFunction := ℕ → Fin 12

/-- The property that a coloring function assigns different colors to all elements in any cascade -/
def ValidColoring (f : ColoringFunction) : Prop :=
  ∀ r : ℕ, ∀ i j : Fin 12, i ≠ j → f (r * (i.val + 1)) ≠ f (r * (j.val + 1))

theorem cascade_properties :
  (∃ a b : ℕ, BelongsToSixCascades a b) ∧
  (∃ f : ColoringFunction, ValidColoring f) := by sorry

end cascade_properties_l3924_392472


namespace smallest_positive_largest_negative_smallest_absolute_l3924_392443

theorem smallest_positive_largest_negative_smallest_absolute (triangle : ℕ) (O : ℤ) (square : ℚ) : 
  (∀ n : ℕ, n > 0 → triangle ≤ n) →
  (∀ z : ℤ, z < 0 → z ≤ O) →
  (∀ q : ℚ, q ≠ 0 → |square| ≤ |q|) →
  triangle > 0 →
  O < 0 →
  (square + triangle) * O = -1 := by
  sorry

end smallest_positive_largest_negative_smallest_absolute_l3924_392443


namespace min_value_theorem_l3924_392415

theorem min_value_theorem (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  (x^2 + 1) / (y - 2) + (y^2 + 1) / (x - 2) ≥ 20 ∧
  ((x^2 + 1) / (y - 2) + (y^2 + 1) / (x - 2) = 20 ↔ x = 3 ∧ y = 3) :=
by sorry

end min_value_theorem_l3924_392415


namespace exam_student_count_l3924_392402

theorem exam_student_count (N : ℕ) (average_all : ℝ) (average_excluded : ℝ) (average_remaining : ℝ) 
  (h1 : average_all = 70)
  (h2 : average_excluded = 50)
  (h3 : average_remaining = 90)
  (h4 : N * average_all = 250 + (N - 5) * average_remaining) :
  N = 10 := by sorry

end exam_student_count_l3924_392402


namespace estimate_greater_than_exact_l3924_392461

theorem estimate_greater_than_exact 
  (a b c a' b' c' : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (ha' : a' ≥ a) (hb' : b' ≤ b) (hc' : c' ≤ c) :
  (a' / b') - c' > (a / b) - c :=
sorry

end estimate_greater_than_exact_l3924_392461


namespace rectangle_to_square_l3924_392442

theorem rectangle_to_square (x y : ℚ) :
  (x - 5 = y + 2) →
  (x * y = (x - 5) * (y + 2)) →
  (x = 25/3 ∧ y = 4/3) :=
by sorry

end rectangle_to_square_l3924_392442


namespace reflected_ray_equation_l3924_392455

/-- Given an incident ray along the line 2x - y + 2 = 0 reflected off the y-axis,
    the equation of the line containing the reflected ray is 2x + y - 2 = 0 -/
theorem reflected_ray_equation (x y : ℝ) :
  (2 * x - y + 2 = 0) →  -- incident ray equation
  (∃ (x' y' : ℝ), 2 * x' + y' - 2 = 0) -- reflected ray equation
  := by sorry

end reflected_ray_equation_l3924_392455


namespace sin_2alpha_value_l3924_392452

theorem sin_2alpha_value (α : Real) 
  (h : Real.cos (π / 4 - α) = 3 / 5) : 
  Real.sin (2 * α) = -7 / 25 := by
  sorry

end sin_2alpha_value_l3924_392452


namespace card_area_reduction_l3924_392470

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle --/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- The theorem to be proved --/
theorem card_area_reduction (initial : Rectangle) :
  initial.length = 5 ∧ initial.width = 8 →
  ∃ (reduced : Rectangle),
    (reduced.length = initial.length - 2 ∨ reduced.width = initial.width - 2) ∧
    area reduced = 21 →
  ∃ (other_reduced : Rectangle),
    (other_reduced.length = initial.length - 2 ∨ other_reduced.width = initial.width - 2) ∧
    other_reduced ≠ reduced ∧
    area other_reduced = 24 := by
  sorry

end card_area_reduction_l3924_392470
