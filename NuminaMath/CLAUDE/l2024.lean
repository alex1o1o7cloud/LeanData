import Mathlib

namespace NUMINAMATH_CALUDE_expected_value_of_game_l2024_202409

def roll_value (n : ℕ) : ℝ :=
  if n % 2 = 0 then 3 * n else 0

def fair_8_sided_die : Finset ℕ := Finset.range 8

theorem expected_value_of_game : 
  (fair_8_sided_die.sum (λ i => (roll_value (i + 1)) / 8)) = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_of_game_l2024_202409


namespace NUMINAMATH_CALUDE_cos_function_property_l2024_202407

theorem cos_function_property (x : ℝ) (n : ℤ) (f : ℝ → ℝ) 
  (h : f (Real.sin x) = Real.sin ((4 * ↑n + 1) * x)) :
  f (Real.cos x) = Real.cos ((4 * ↑n + 1) * x) := by
  sorry

end NUMINAMATH_CALUDE_cos_function_property_l2024_202407


namespace NUMINAMATH_CALUDE_acid_solution_mixture_l2024_202483

/-- Proves that adding 40 ounces of pure water and 200/9 ounces of 10% acid solution
    to 40 ounces of 25% acid solution results in a 15% acid solution. -/
theorem acid_solution_mixture : 
  let initial_volume : ℝ := 40
  let initial_concentration : ℝ := 0.25
  let water_added : ℝ := 40
  let dilute_solution_added : ℝ := 200 / 9
  let dilute_concentration : ℝ := 0.1
  let final_concentration : ℝ := 0.15
  let final_volume : ℝ := initial_volume + water_added + dilute_solution_added
  let final_acid_amount : ℝ := initial_volume * initial_concentration + 
                                dilute_solution_added * dilute_concentration
  final_acid_amount / final_volume = final_concentration :=
by
  sorry


end NUMINAMATH_CALUDE_acid_solution_mixture_l2024_202483


namespace NUMINAMATH_CALUDE_transformed_line_equation_l2024_202492

-- Define the original line
def original_line (x y : ℝ) : Prop := x - 2 * y = 2

-- Define the stretch transformation
def stretch_transform (x y x' y' : ℝ) : Prop := x' = x ∧ y' = 2 * y

-- Theorem: The equation of line l after transformation is x - y - 2 = 0
theorem transformed_line_equation (x' y' : ℝ) :
  (∃ x y, original_line x y ∧ stretch_transform x y x' y') →
  x' - y' - 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_transformed_line_equation_l2024_202492


namespace NUMINAMATH_CALUDE_abs_opposite_neg_six_l2024_202495

theorem abs_opposite_neg_six : |-(- 6)| = 6 := by
  sorry

end NUMINAMATH_CALUDE_abs_opposite_neg_six_l2024_202495


namespace NUMINAMATH_CALUDE_mock_exam_girls_count_l2024_202474

theorem mock_exam_girls_count :
  ∀ (total_students : ℕ) (boys girls : ℕ) (boys_cleared girls_cleared : ℕ),
    total_students = 400 →
    boys + girls = total_students →
    boys_cleared = (60 * boys) / 100 →
    girls_cleared = (80 * girls) / 100 →
    boys_cleared + girls_cleared = (65 * total_students) / 100 →
    girls = 100 := by
  sorry

end NUMINAMATH_CALUDE_mock_exam_girls_count_l2024_202474


namespace NUMINAMATH_CALUDE_jack_socks_problem_l2024_202444

theorem jack_socks_problem :
  ∀ (x y z : ℕ),
    x + y + z = 15 →
    2 * x + 4 * y + 5 * z = 36 →
    x ≥ 1 →
    y ≥ 1 →
    z ≥ 1 →
    x = 4 :=
by sorry

end NUMINAMATH_CALUDE_jack_socks_problem_l2024_202444


namespace NUMINAMATH_CALUDE_total_gray_trees_count_l2024_202422

/-- Represents an aerial photo with tree counts -/
structure AerialPhoto where
  totalTrees : ℕ
  whiteTrees : ℕ

/-- Calculates the number of trees in the gray area of a photo -/
def grayTrees (photo : AerialPhoto) : ℕ :=
  photo.totalTrees - photo.whiteTrees

theorem total_gray_trees_count 
  (photo1 photo2 photo3 : AerialPhoto)
  (h1 : photo1.totalTrees = 100)
  (h2 : photo1.whiteTrees = 82)
  (h3 : photo2.totalTrees = 90)
  (h4 : photo2.whiteTrees = 82)
  (h5 : photo3.whiteTrees = 75)
  (h6 : photo1.totalTrees = photo2.totalTrees)
  (h7 : photo2.totalTrees = photo3.totalTrees) :
  grayTrees photo1 + grayTrees photo2 + grayTrees photo3 = 26 := by
  sorry

end NUMINAMATH_CALUDE_total_gray_trees_count_l2024_202422


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l2024_202479

theorem smallest_solution_of_equation (x : ℝ) :
  (3 * x^2 + 33 * x - 90 = x * (x + 15)) →
  x ≥ -15 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l2024_202479


namespace NUMINAMATH_CALUDE_digit_58_is_8_l2024_202434

/-- The decimal representation of 1/7 -/
def decimal_rep_1_7 : ℕ → ℕ
| 0 => 1
| 1 => 4
| 2 => 2
| 3 => 8
| 4 => 5
| 5 => 7
| n + 6 => decimal_rep_1_7 n

/-- The period of the decimal representation of 1/7 -/
def period : ℕ := 6

/-- The 58th digit after the decimal point in the decimal representation of 1/7 -/
def digit_58 : ℕ := decimal_rep_1_7 ((58 - 1) % period)

theorem digit_58_is_8 : digit_58 = 8 := by sorry

end NUMINAMATH_CALUDE_digit_58_is_8_l2024_202434


namespace NUMINAMATH_CALUDE_f_value_l2024_202464

-- Define the ceiling function
def ceiling (x : ℚ) : ℤ := Int.ceil x

-- Define the function f
def f (x y : ℚ) : ℚ := x - y * ceiling (x / y)

-- State the theorem
theorem f_value : f (1/3) (-3/7) = -2/21 := by
  sorry

end NUMINAMATH_CALUDE_f_value_l2024_202464


namespace NUMINAMATH_CALUDE_jeans_wednesday_calls_l2024_202454

/-- Represents the number of calls Jean answered each day of the week --/
structure WeekCalls where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- The average number of calls per day --/
def average_calls : ℕ := 40

/-- The number of working days --/
def working_days : ℕ := 5

/-- Calculates the total number of calls in a week --/
def total_calls (w : WeekCalls) : ℕ :=
  w.monday + w.tuesday + w.wednesday + w.thursday + w.friday

/-- Jean's calls for the week --/
def jeans_calls : WeekCalls := {
  monday := 35,
  tuesday := 46,
  wednesday := 27,  -- This is what we want to prove
  thursday := 61,
  friday := 31
}

/-- Theorem stating that Jean answered 27 calls on Wednesday --/
theorem jeans_wednesday_calls :
  jeans_calls.wednesday = 27 ∧
  total_calls jeans_calls = average_calls * working_days :=
sorry

end NUMINAMATH_CALUDE_jeans_wednesday_calls_l2024_202454


namespace NUMINAMATH_CALUDE_solve_percentage_equation_l2024_202457

theorem solve_percentage_equation (x : ℝ) : 
  (70 / 100) * 600 = (40 / 100) * x → x = 1050 := by
  sorry

end NUMINAMATH_CALUDE_solve_percentage_equation_l2024_202457


namespace NUMINAMATH_CALUDE_circle_arrangement_exists_l2024_202460

theorem circle_arrangement_exists : ∃ (a : Fin 12 → Fin 12), Function.Bijective a ∧
  ∀ (i j : Fin 12), i < j → |a i - a j| ≠ |i - j| := by
  sorry

end NUMINAMATH_CALUDE_circle_arrangement_exists_l2024_202460


namespace NUMINAMATH_CALUDE_equation_one_solution_l2024_202435

-- Define the equation
def equation (x p : ℝ) : Prop :=
  2 * |x - p| + |x - 2| = 1

-- Define the property of having exactly one solution
def has_exactly_one_solution (p : ℝ) : Prop :=
  ∃! x, equation x p

-- Theorem statement
theorem equation_one_solution :
  ∀ p : ℝ, has_exactly_one_solution p ↔ (p = 1 ∨ p = 3) :=
sorry

end NUMINAMATH_CALUDE_equation_one_solution_l2024_202435


namespace NUMINAMATH_CALUDE_kenzo_office_chairs_l2024_202458

theorem kenzo_office_chairs :
  ∀ (initial_chairs : ℕ),
    (∃ (chairs_legs tables_legs remaining_chairs_legs : ℕ),
      chairs_legs = 5 * initial_chairs ∧
      tables_legs = 20 * 3 ∧
      remaining_chairs_legs = (6 * chairs_legs) / 10 ∧
      remaining_chairs_legs + tables_legs = 300) →
    initial_chairs = 80 := by
  sorry

end NUMINAMATH_CALUDE_kenzo_office_chairs_l2024_202458


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2024_202484

-- Problem 1
theorem problem_1 : 0.108 / 1.2 + 0.7 = 0.79 := by sorry

-- Problem 2
theorem problem_2 : (9.8 - 3.75) / 25 / 0.4 = 0.605 := by sorry

-- Problem 3
theorem problem_3 : 6.3 * 15 + 1/3 * 75/100 = 94.75 := by sorry

-- Problem 4
theorem problem_4 : 8 * 0.56 + 5.4 * 0.8 - 80/100 = 8 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2024_202484


namespace NUMINAMATH_CALUDE_inequality_proof_l2024_202428

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) 
  (h_product : a * b * c = 1) : 
  a^2 + b^2 + c^2 + 3 ≥ 2 * (1/a + 1/b + 1/c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2024_202428


namespace NUMINAMATH_CALUDE_tom_brick_cost_l2024_202472

/-- The total cost for Tom's bricks -/
def total_cost (total_bricks : ℕ) (original_price : ℚ) (discount_percent : ℚ) : ℚ :=
  let discounted_bricks := total_bricks / 2
  let full_price_bricks := total_bricks - discounted_bricks
  let discounted_price := original_price * (1 - discount_percent)
  (discounted_bricks : ℚ) * discounted_price + (full_price_bricks : ℚ) * original_price

/-- Theorem stating that the total cost for Tom's bricks is $375 -/
theorem tom_brick_cost :
  total_cost 1000 (1/2) (1/2) = 375 := by
  sorry

end NUMINAMATH_CALUDE_tom_brick_cost_l2024_202472


namespace NUMINAMATH_CALUDE_inequality_proof_l2024_202497

theorem inequality_proof (x y z : ℝ) :
  -3/2 * (x^2 + y^2 + 2*z^2) ≤ 3*x*y + y*z + z*x ∧
  3*x*y + y*z + z*x ≤ (3 + Real.sqrt 13)/4 * (x^2 + y^2 + 2*z^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2024_202497


namespace NUMINAMATH_CALUDE_circle_radius_l2024_202416

theorem circle_radius (x y : ℝ) : 
  (∃ r, r > 0 ∧ ∀ x y, x^2 + y^2 - 4*x + 2*y + 2 = 0 ↔ (x - 2)^2 + (y + 1)^2 = r^2) →
  (∃ r, r > 0 ∧ ∀ x y, x^2 + y^2 - 4*x + 2*y + 2 = 0 ↔ (x - 2)^2 + (y + 1)^2 = r^2 ∧ r = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_l2024_202416


namespace NUMINAMATH_CALUDE_expression_bounds_l2024_202404

theorem expression_bounds (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 2) (hb : 0 ≤ b ∧ b ≤ 2) (hc : 0 ≤ c ∧ c ≤ 2) (hd : 0 ≤ d ∧ d ≤ 2) :
  4 * Real.sqrt 2 ≤ Real.sqrt (a^2 + (2-b)^2) + Real.sqrt (b^2 + (2-c)^2) + 
    Real.sqrt (c^2 + (2-d)^2) + Real.sqrt (d^2 + (2-a)^2) ∧
  Real.sqrt (a^2 + (2-b)^2) + Real.sqrt (b^2 + (2-c)^2) + 
    Real.sqrt (c^2 + (2-d)^2) + Real.sqrt (d^2 + (2-a)^2) ≤ 8 := by
  sorry


end NUMINAMATH_CALUDE_expression_bounds_l2024_202404


namespace NUMINAMATH_CALUDE_leftover_coin_value_l2024_202426

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 25 / 100

/-- The value of a dime in dollars -/
def dime_value : ℚ := 10 / 100

/-- The number of quarters in a complete roll -/
def quarters_per_roll : ℕ := 35

/-- The number of dimes in a complete roll -/
def dimes_per_roll : ℕ := 55

/-- James' quarters -/
def james_quarters : ℕ := 97

/-- James' dimes -/
def james_dimes : ℕ := 173

/-- Lindsay's quarters -/
def lindsay_quarters : ℕ := 141

/-- Lindsay's dimes -/
def lindsay_dimes : ℕ := 289

/-- The total number of quarters -/
def total_quarters : ℕ := james_quarters + lindsay_quarters

/-- The total number of dimes -/
def total_dimes : ℕ := james_dimes + lindsay_dimes

theorem leftover_coin_value :
  (total_quarters % quarters_per_roll : ℚ) * quarter_value +
  (total_dimes % dimes_per_roll : ℚ) * dime_value = 92 / 10 := by
  sorry

end NUMINAMATH_CALUDE_leftover_coin_value_l2024_202426


namespace NUMINAMATH_CALUDE_marbles_problem_l2024_202414

theorem marbles_problem (total : ℕ) (given_to_sister : ℕ) : 
  (given_to_sister = total / 6) →
  (given_to_sister = 9) →
  (total - (total / 2 + total / 6) = 18) :=
by sorry

end NUMINAMATH_CALUDE_marbles_problem_l2024_202414


namespace NUMINAMATH_CALUDE_number_of_male_students_l2024_202421

theorem number_of_male_students 
  (total_candidates : ℕ)
  (male_students : ℕ)
  (female_students : ℕ)
  (selected_male : ℕ)
  (selected_female : ℕ)
  (num_camps : ℕ)
  (total_schemes : ℕ) :
  total_candidates = 10 →
  male_students + female_students = total_candidates →
  male_students > female_students →
  selected_male = 2 →
  selected_female = 2 →
  num_camps = 3 →
  total_schemes = 3240 →
  (male_students.choose selected_male * female_students.choose selected_female * 
   (selected_male + selected_female).choose num_camps * num_camps.factorial = total_schemes) →
  male_students = 6 := by
sorry

end NUMINAMATH_CALUDE_number_of_male_students_l2024_202421


namespace NUMINAMATH_CALUDE_diagonal_division_ratio_equality_l2024_202465

/-- A point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A quadrilateral defined by four points -/
structure Quadrilateral :=
  (A B C D : Point)

/-- The orthocenter of a triangle -/
def orthocenter (A B C : Point) : Point :=
  sorry

/-- The ratio in which a line segment is divided by a point -/
def divisionRatio (A B P : Point) : ℝ :=
  sorry

/-- The intersection point of two line segments -/
def intersectionPoint (A B C D : Point) : Point :=
  sorry

/-- Theorem: In convex quadrilaterals ABCD and A'B'C'D', where A', B', C', D' are orthocenters
    of triangles BCD, CDA, DAB, ABC respectively, the corresponding diagonals are divided by 
    the points of intersection in the same ratio -/
theorem diagonal_division_ratio_equality 
  (ABCD : Quadrilateral) 
  (A' B' C' D' : Point) 
  (h_convex : sorry) -- Assume ABCD is convex
  (h_A' : A' = orthocenter ABCD.B ABCD.C ABCD.D)
  (h_B' : B' = orthocenter ABCD.C ABCD.D ABCD.A)
  (h_C' : C' = orthocenter ABCD.D ABCD.A ABCD.B)
  (h_D' : D' = orthocenter ABCD.A ABCD.B ABCD.C) :
  let P := intersectionPoint ABCD.A ABCD.C ABCD.B ABCD.D
  let P' := intersectionPoint A' C' B' D'
  divisionRatio ABCD.A ABCD.C P = divisionRatio A' C' P' ∧
  divisionRatio ABCD.B ABCD.D P = divisionRatio B' D' P' :=
sorry

end NUMINAMATH_CALUDE_diagonal_division_ratio_equality_l2024_202465


namespace NUMINAMATH_CALUDE_tims_change_l2024_202427

def initial_amount : ℚ := 1.50
def candy_cost : ℚ := 0.45
def chips_cost : ℚ := 0.65
def toy_cost : ℚ := 0.40
def discount_rate : ℚ := 0.10

def total_snacks_cost : ℚ := candy_cost + chips_cost
def discounted_snacks_cost : ℚ := total_snacks_cost * (1 - discount_rate)
def total_cost : ℚ := discounted_snacks_cost + toy_cost
def change : ℚ := initial_amount - total_cost

theorem tims_change : change = 0.11 := by sorry

end NUMINAMATH_CALUDE_tims_change_l2024_202427


namespace NUMINAMATH_CALUDE_no_roots_of_composition_if_no_roots_l2024_202412

/-- A quadratic polynomial -/
def QuadraticPolynomial (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- Theorem: If p(x) = x has no real roots, then p(p(x)) = x has no real roots -/
theorem no_roots_of_composition_if_no_roots (a b c : ℝ) :
  (∀ x : ℝ, QuadraticPolynomial a b c x ≠ x) →
  (∀ x : ℝ, QuadraticPolynomial a b c (QuadraticPolynomial a b c x) ≠ x) := by
  sorry


end NUMINAMATH_CALUDE_no_roots_of_composition_if_no_roots_l2024_202412


namespace NUMINAMATH_CALUDE_cara_seating_arrangements_l2024_202436

/-- The number of people at the circular table, including Cara -/
def total_people : ℕ := 7

/-- The number of Cara's friends -/
def num_friends : ℕ := 6

/-- Alex is one of Cara's friends -/
def alex_is_friend : Prop := true

/-- The number of different pairs Cara could be sitting between, where one must be Alex -/
def num_seating_arrangements : ℕ := 5

theorem cara_seating_arrangements :
  total_people = num_friends + 1 →
  alex_is_friend →
  num_seating_arrangements = num_friends - 1 :=
by
  sorry

end NUMINAMATH_CALUDE_cara_seating_arrangements_l2024_202436


namespace NUMINAMATH_CALUDE_negation_equivalence_l2024_202496

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - 4*x + 2 > 0) ↔ (∀ x : ℝ, x^2 - 4*x + 2 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2024_202496


namespace NUMINAMATH_CALUDE_twenty_player_tournament_games_l2024_202467

/-- Calculates the number of games in a chess tournament --/
def chess_tournament_games (n : ℕ) : ℕ :=
  n * (n - 1)

/-- Theorem: In a chess tournament with 20 players, where each player plays twice with every other player, 
    the total number of games played is 760. --/
theorem twenty_player_tournament_games : 
  chess_tournament_games 20 * 2 = 760 := by
  sorry

end NUMINAMATH_CALUDE_twenty_player_tournament_games_l2024_202467


namespace NUMINAMATH_CALUDE_prob_green_or_yellow_l2024_202487

/-- A cube with colored faces -/
structure ColoredCube where
  green_faces : ℕ
  yellow_faces : ℕ
  blue_faces : ℕ

/-- The probability of an event -/
def probability (favorable_outcomes : ℕ) (total_outcomes : ℕ) : ℚ :=
  favorable_outcomes / total_outcomes

/-- Theorem: Probability of rolling a green or yellow face -/
theorem prob_green_or_yellow (cube : ColoredCube) 
  (h1 : cube.green_faces = 3)
  (h2 : cube.yellow_faces = 2)
  (h3 : cube.blue_faces = 1) :
  probability (cube.green_faces + cube.yellow_faces) 
    (cube.green_faces + cube.yellow_faces + cube.blue_faces) = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_prob_green_or_yellow_l2024_202487


namespace NUMINAMATH_CALUDE_correct_card_to_disprove_jane_l2024_202486

-- Define the type for card sides
inductive CardSide
| Letter (c : Char)
| Number (n : Nat)

-- Define the structure for a card
structure Card where
  side1 : CardSide
  side2 : CardSide

-- Define the function to check if a number is odd
def isOdd (n : Nat) : Bool :=
  n % 2 = 1

-- Define the function to check if a character is a vowel
def isVowel (c : Char) : Bool :=
  c ∈ ['A', 'E', 'I', 'O', 'U']

-- Define Jane's statement as a function
def janesStatement (card : Card) : Bool :=
  match card.side1, card.side2 with
  | CardSide.Number n, CardSide.Letter c => 
      ¬(isOdd n ∧ isVowel c)
  | CardSide.Letter c, CardSide.Number n => 
      ¬(isOdd n ∧ isVowel c)
  | _, _ => true

-- Define the theorem
theorem correct_card_to_disprove_jane : 
  ∀ (cards : List Card),
  cards = [
    Card.mk (CardSide.Letter 'A') (CardSide.Number 0),
    Card.mk (CardSide.Letter 'S') (CardSide.Number 0),
    Card.mk (CardSide.Number 5) (CardSide.Letter ' '),
    Card.mk (CardSide.Number 8) (CardSide.Letter ' '),
    Card.mk (CardSide.Number 7) (CardSide.Letter ' ')
  ] →
  ∃ (card : Card),
  card ∈ cards ∧ 
  card.side1 = CardSide.Letter 'A' ∧
  (∃ (n : Nat), card.side2 = CardSide.Number n ∧ isOdd n) ∧
  (∀ (c : Card), c ∈ cards ∧ c ≠ card → janesStatement c) :=
by sorry


end NUMINAMATH_CALUDE_correct_card_to_disprove_jane_l2024_202486


namespace NUMINAMATH_CALUDE_bajazet_winning_strategy_l2024_202450

-- Define a polynomial of degree 4
def polynomial (a b c : ℝ) (x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + 1

-- State the theorem
theorem bajazet_winning_strategy :
  ∀ (a b c : ℝ), ∃ (x : ℝ), polynomial a b c x = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_bajazet_winning_strategy_l2024_202450


namespace NUMINAMATH_CALUDE_problem_solution_l2024_202411

/-- The probability that person A can solve the problem within half an hour -/
def prob_A : ℚ := 1/2

/-- The probability that person B can solve the problem within half an hour -/
def prob_B : ℚ := 1/3

/-- The probability that neither A nor B solves the problem -/
def prob_neither_solves : ℚ := (1 - prob_A) * (1 - prob_B)

/-- The probability that the problem is solved -/
def prob_problem_solved : ℚ := 1 - prob_neither_solves

theorem problem_solution :
  prob_neither_solves = 1/3 ∧ prob_problem_solved = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2024_202411


namespace NUMINAMATH_CALUDE_lunch_break_duration_l2024_202482

/-- Represents the painting scenario with Paula and her helpers --/
structure PaintingScenario where
  paula_rate : ℝ
  helpers_rate : ℝ
  lunch_break : ℝ

/-- Conditions of the painting scenario --/
def painting_conditions (s : PaintingScenario) : Prop :=
  -- Monday's work
  (9 - s.lunch_break) * (s.paula_rate + s.helpers_rate) = 0.4 ∧
  -- Tuesday's work
  (8 - s.lunch_break) * s.helpers_rate = 0.33 ∧
  -- Wednesday's work
  (12 - s.lunch_break) * s.paula_rate = 0.27

/-- The main theorem: lunch break duration is 420 minutes --/
theorem lunch_break_duration (s : PaintingScenario) :
  painting_conditions s → s.lunch_break * 60 = 420 := by
  sorry

end NUMINAMATH_CALUDE_lunch_break_duration_l2024_202482


namespace NUMINAMATH_CALUDE_determinant_special_matrix_l2024_202456

/-- The determinant of the matrix [[1, x, z], [1, x+z, z], [1, x, x+z]] is equal to xz - z^2 -/
theorem determinant_special_matrix (x z : ℝ) :
  Matrix.det !![1, x, z; 1, x + z, z; 1, x, x + z] = x * z - z^2 := by
  sorry

end NUMINAMATH_CALUDE_determinant_special_matrix_l2024_202456


namespace NUMINAMATH_CALUDE_sequence_convergence_l2024_202470

theorem sequence_convergence (a : ℕ → ℚ) :
  a 1 = 3 / 5 →
  (∀ n : ℕ, a (n + 1) = 2 - 1 / (a n)) →
  a 2018 = 4031 / 4029 := by
sorry

end NUMINAMATH_CALUDE_sequence_convergence_l2024_202470


namespace NUMINAMATH_CALUDE_value_of_c_l2024_202423

theorem value_of_c : 1996 * 19971997 - 1995 * 19961996 = 3995992 := by
  sorry

end NUMINAMATH_CALUDE_value_of_c_l2024_202423


namespace NUMINAMATH_CALUDE_weighted_average_is_38_5_l2024_202425

/-- Represents the marks in different subjects -/
structure Marks where
  mathematics : ℝ
  physics : ℝ
  chemistry : ℝ
  biology : ℝ

/-- Calculates the weighted average of Mathematics, Chemistry, and Biology marks -/
def weightedAverage (m : Marks) : ℝ :=
  0.4 * m.mathematics + 0.3 * m.chemistry + 0.3 * m.biology

/-- Theorem stating that under given conditions, the weighted average is 38.5 -/
theorem weighted_average_is_38_5 (m : Marks) :
  m.mathematics + m.physics + m.biology = 90 ∧
  m.chemistry = m.physics + 10 ∧
  m.biology = m.chemistry - 5 →
  weightedAverage m = 38.5 := by
  sorry

#eval weightedAverage { mathematics := 85, physics := 0, chemistry := 10, biology := 5 }

end NUMINAMATH_CALUDE_weighted_average_is_38_5_l2024_202425


namespace NUMINAMATH_CALUDE_garage_sale_items_count_l2024_202433

theorem garage_sale_items_count 
  (prices : Finset ℕ) 
  (radio_price : ℕ) 
  (h1 : radio_price ∈ prices) 
  (h2 : (prices.filter (λ x => x > radio_price)).card = 14) 
  (h3 : (prices.filter (λ x => x < radio_price)).card = 24) :
  prices.card = 39 :=
sorry

end NUMINAMATH_CALUDE_garage_sale_items_count_l2024_202433


namespace NUMINAMATH_CALUDE_parabola_equation_from_hyperbola_l2024_202403

/-- Given a hyperbola and a parabola with specific properties, prove the equation of the parabola -/
theorem parabola_equation_from_hyperbola (x y : ℝ) :
  (x^2 / 3 - y^2 = 1) →  -- Given hyperbola equation
  (∃ (p : ℝ), 
    (p > 0) ∧  -- p is positive for a right-opening parabola
    ((2 : ℝ) = p / 2) ∧  -- Focus of parabola is at (2, 0), which is (p/2, 0) in standard form
    (y^2 = 2 * p * x))  -- Standard form of parabola equation
  →
  y^2 = 8 * x  -- Conclusion: specific equation of the parabola
:= by sorry

end NUMINAMATH_CALUDE_parabola_equation_from_hyperbola_l2024_202403


namespace NUMINAMATH_CALUDE_Z_is_real_Z_is_pure_imaginary_Z_in_fourth_quadrant_l2024_202445

-- Define the complex number Z as a function of real number m
def Z (m : ℝ) : ℂ := Complex.mk (m^2 + 5*m + 6) (m^2 - 2*m - 15)

-- Theorem 1: Z is real iff m = -3 or m = 5
theorem Z_is_real (m : ℝ) : (Z m).im = 0 ↔ m = -3 ∨ m = 5 := by sorry

-- Theorem 2: Z is pure imaginary iff m = -2
theorem Z_is_pure_imaginary (m : ℝ) : (Z m).re = 0 ↔ m = -2 := by sorry

-- Theorem 3: Z is in the fourth quadrant iff -2 < m < 5
theorem Z_in_fourth_quadrant (m : ℝ) : 
  ((Z m).re > 0 ∧ (Z m).im < 0) ↔ -2 < m ∧ m < 5 := by sorry

end NUMINAMATH_CALUDE_Z_is_real_Z_is_pure_imaginary_Z_in_fourth_quadrant_l2024_202445


namespace NUMINAMATH_CALUDE_p_and_q_true_l2024_202417

theorem p_and_q_true (P Q : Prop) (h : ¬(P ∧ Q) = False) : P ∧ Q :=
sorry

end NUMINAMATH_CALUDE_p_and_q_true_l2024_202417


namespace NUMINAMATH_CALUDE_inequality_holds_iff_b_greater_than_one_l2024_202471

theorem inequality_holds_iff_b_greater_than_one (b : ℝ) (h : b > 0) :
  (∃ x : ℝ, |x - 2| + |x - 1| < b) ↔ b > 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_b_greater_than_one_l2024_202471


namespace NUMINAMATH_CALUDE_fraction_simplification_l2024_202488

theorem fraction_simplification (x y : ℝ) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (hxy : x^2 - 1/y ≠ 0) 
  (hyx : y^2 - 1/x ≠ 0) : 
  (x^2 - 1/y) / (y^2 - 1/x) = x * (x^2*y - 1) / (y * (y^2*x - 1)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2024_202488


namespace NUMINAMATH_CALUDE_current_rate_calculation_l2024_202400

/-- Represents the speed of a man rowing in different conditions -/
structure RowingSpeed where
  downstream : ℝ
  upstream : ℝ
  stillWater : ℝ

/-- Calculates the rate of the current given rowing speeds -/
def currentRate (speed : RowingSpeed) : ℝ :=
  speed.downstream - speed.stillWater

theorem current_rate_calculation (speed : RowingSpeed) 
  (h1 : speed.downstream = 24)
  (h2 : speed.upstream = 7)
  (h3 : speed.stillWater = 15.5) :
  currentRate speed = 8.5 := by
  sorry

#eval currentRate { downstream := 24, upstream := 7, stillWater := 15.5 }

end NUMINAMATH_CALUDE_current_rate_calculation_l2024_202400


namespace NUMINAMATH_CALUDE_f_2013_pi_third_l2024_202418

open Real

noncomputable def f₀ (x : ℝ) : ℝ := sin x - cos x

noncomputable def f (n : ℕ) (x : ℝ) : ℝ := 
  match n with
  | 0 => f₀ x
  | n + 1 => deriv (f n) x

theorem f_2013_pi_third : f 2013 (π/3) = (1 + Real.sqrt 3) / 2 := by sorry

end NUMINAMATH_CALUDE_f_2013_pi_third_l2024_202418


namespace NUMINAMATH_CALUDE_arithmetic_sequence_with_geometric_mean_l2024_202462

/-- An arithmetic sequence with common difference 2 -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

/-- The 4th term is the geometric mean of the 2nd and 5th terms -/
def geometric_mean_condition (a : ℕ → ℝ) : Prop :=
  a 4 ^ 2 = a 2 * a 5

/-- Main theorem: If a is an arithmetic sequence with common difference 2
    and the 4th term is the geometric mean of the 2nd and 5th terms,
    then the 2nd term is -8 -/
theorem arithmetic_sequence_with_geometric_mean
  (a : ℕ → ℝ)
  (h1 : arithmetic_sequence a)
  (h2 : geometric_mean_condition a) :
  a 2 = -8 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_with_geometric_mean_l2024_202462


namespace NUMINAMATH_CALUDE_five_students_three_villages_l2024_202466

/-- The number of ways to assign n students to m villages with at least one student per village -/
def assignmentCount (n m : ℕ) : ℕ := sorry

/-- The number of ways to assign 5 students to 3 villages with at least one student per village -/
theorem five_students_three_villages : assignmentCount 5 3 = 150 := by sorry

end NUMINAMATH_CALUDE_five_students_three_villages_l2024_202466


namespace NUMINAMATH_CALUDE_ancient_chinese_math_problem_l2024_202448

theorem ancient_chinese_math_problem (people : ℕ) (price : ℕ) : 
  (8 * people - price = 3) →
  (price - 7 * people = 4) →
  people = 7 := by
  sorry

end NUMINAMATH_CALUDE_ancient_chinese_math_problem_l2024_202448


namespace NUMINAMATH_CALUDE_purple_chips_count_l2024_202476

/-- Represents the number of chips of each color selected -/
structure ChipSelection where
  blue : ℕ
  green : ℕ
  purple : ℕ
  red : ℕ

/-- The theorem stating the number of purple chips selected -/
theorem purple_chips_count 
  (x : ℕ) 
  (h1 : 5 < x) 
  (h2 : x < 11) 
  (selection : ChipSelection) 
  (h3 : 1^selection.blue * 5^selection.green * x^selection.purple * 11^selection.red = 28160) :
  selection.purple = 2 :=
sorry

end NUMINAMATH_CALUDE_purple_chips_count_l2024_202476


namespace NUMINAMATH_CALUDE_percentage_calculation_l2024_202424

theorem percentage_calculation (P : ℝ) : 
  (0.15 * 0.30 * (P / 100) * 4400 = 99) → P = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l2024_202424


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l2024_202449

theorem polynomial_evaluation (w x y z : ℝ) 
  (eq1 : w + x + y + z = 5)
  (eq2 : 2*w + 4*x + 8*y + 16*z = 7)
  (eq3 : 3*w + 9*x + 27*y + 81*z = 11)
  (eq4 : 4*w + 16*x + 64*y + 256*z = 1) :
  5*w + 25*x + 125*y + 625*z = -60 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l2024_202449


namespace NUMINAMATH_CALUDE_max_speed_theorem_l2024_202408

/-- Represents a pair of observed values (speed, defective products) -/
structure Observation where
  speed : ℝ
  defects : ℝ

/-- The regression line equation -/
def regression_line (slope : ℝ) (intercept : ℝ) (x : ℝ) : ℝ :=
  slope * x + intercept

/-- Theorem: Maximum speed given observations and max defects -/
theorem max_speed_theorem (observations : List Observation) 
  (max_defects : ℝ) (slope : ℝ) (intercept : ℝ) :
  observations = [
    ⟨8, 5⟩, ⟨12, 8⟩, ⟨14, 9⟩, ⟨16, 11⟩
  ] →
  max_defects = 10 →
  slope = 51 / 70 →
  intercept = -6 / 7 →
  (∀ x, regression_line slope intercept x ≤ max_defects → x ≤ 14) ∧
  regression_line slope intercept 14 ≤ max_defects :=
by sorry

end NUMINAMATH_CALUDE_max_speed_theorem_l2024_202408


namespace NUMINAMATH_CALUDE_whitney_max_sets_l2024_202442

/-- Represents the number of items Whitney has -/
structure Inventory where
  tshirts : ℕ
  buttons : ℕ
  stickers : ℕ

/-- Represents the composition of each set -/
structure SetComposition where
  tshirts : ℕ
  buttons : ℕ
  stickers : ℕ

def max_sets (inv : Inventory) (comp : SetComposition) : ℕ :=
  min (inv.tshirts / comp.tshirts)
      (min (inv.buttons / comp.buttons) (inv.stickers / comp.stickers))

/-- Theorem stating that the maximum number of sets Whitney can make is 5 -/
theorem whitney_max_sets :
  let inv : Inventory := { tshirts := 5, buttons := 24, stickers := 12 }
  let comp : SetComposition := { tshirts := 1, buttons := 2, stickers := 1 }
  max_sets inv comp = 5 := by
  sorry

end NUMINAMATH_CALUDE_whitney_max_sets_l2024_202442


namespace NUMINAMATH_CALUDE_return_speed_calculation_l2024_202446

/-- Given a round trip where the return speed is twice the outbound speed,
    prove that the return speed is 15 km/h when the total distance is 60 km
    and the total travel time is 6 hours. -/
theorem return_speed_calculation (distance : ℝ) (total_time : ℝ) (outbound_speed : ℝ) :
  distance = 60 →
  total_time = 6 →
  outbound_speed > 0 →
  distance / (2 * outbound_speed) + distance / (2 * (2 * outbound_speed)) = total_time →
  2 * outbound_speed = 15 := by
  sorry

end NUMINAMATH_CALUDE_return_speed_calculation_l2024_202446


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l2024_202419

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 - x + 3 > 0) ↔ (∃ x : ℝ, x^2 - x + 3 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l2024_202419


namespace NUMINAMATH_CALUDE_correct_relative_pronouns_l2024_202413

/-- A type representing relative pronouns -/
inductive RelativePronoun
  | What
  | Where
  | That
  | Which

/-- A function that checks if a relative pronoun introduces a defining clause without an antecedent -/
def introduces_defining_clause_without_antecedent (rp : RelativePronoun) : Prop :=
  match rp with
  | RelativePronoun.What => True
  | _ => False

/-- A function that checks if a relative pronoun introduces a clause describing a location or circumstance -/
def introduces_location_clause (rp : RelativePronoun) : Prop :=
  match rp with
  | RelativePronoun.Where => True
  | _ => False

theorem correct_relative_pronouns :
  ∃ (rp1 rp2 : RelativePronoun),
    introduces_defining_clause_without_antecedent rp1 ∧
    introduces_location_clause rp2 ∧
    rp1 = RelativePronoun.What ∧
    rp2 = RelativePronoun.Where :=
by
  sorry

end NUMINAMATH_CALUDE_correct_relative_pronouns_l2024_202413


namespace NUMINAMATH_CALUDE_sector_perimeter_l2024_202451

theorem sector_perimeter (r c θ : ℝ) (hr : r = 10) (hc : c = 10) (hθ : θ = 120 * π / 180) :
  r * θ + c = 20 * π / 3 + 10 := by
  sorry

end NUMINAMATH_CALUDE_sector_perimeter_l2024_202451


namespace NUMINAMATH_CALUDE_quadratic_real_roots_k_range_l2024_202461

theorem quadratic_real_roots_k_range (k : ℝ) : 
  (∃ x : ℝ, 2 * x^2 + 4 * x + k - 1 = 0) → k ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_k_range_l2024_202461


namespace NUMINAMATH_CALUDE_problem_solution_l2024_202443

theorem problem_solution (a b c d : ℝ) 
  (h1 : a < b ∧ b < d)
  (h2 : ∀ x, (x - a) * (x - b) * (x - d) / (x - c) ≥ 0 ↔ x ≤ -7 ∨ (30 ≤ x ∧ x ≤ 32)) :
  a + 2*b + 3*c + 4*d = 160 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2024_202443


namespace NUMINAMATH_CALUDE_complex_inequality_l2024_202463

open Complex

theorem complex_inequality (x y : ℂ) (z : ℂ) 
  (h1 : abs x = 1) (h2 : abs y = 1)
  (h3 : π / 3 ≤ arg x - arg y) (h4 : arg x - arg y ≤ 5 * π / 3) :
  abs z + abs (z - x) + abs (z - y) ≥ abs (z * x - y) := by
  sorry

end NUMINAMATH_CALUDE_complex_inequality_l2024_202463


namespace NUMINAMATH_CALUDE_point_positions_l2024_202477

/-- Circle C is defined by the equation x^2 + y^2 - 2x + 4y - 4 = 0 --/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y - 4 = 0

/-- Point M has coordinates (2, -4) --/
def point_M : ℝ × ℝ := (2, -4)

/-- Point N has coordinates (-2, 1) --/
def point_N : ℝ × ℝ := (-2, 1)

/-- A point (x, y) is inside the circle if x^2 + y^2 - 2x + 4y - 4 < 0 --/
def inside_circle (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  x^2 + y^2 - 2*x + 4*y - 4 < 0

/-- A point (x, y) is outside the circle if x^2 + y^2 - 2x + 4y - 4 > 0 --/
def outside_circle (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  x^2 + y^2 - 2*x + 4*y - 4 > 0

theorem point_positions :
  inside_circle point_M ∧ outside_circle point_N :=
sorry

end NUMINAMATH_CALUDE_point_positions_l2024_202477


namespace NUMINAMATH_CALUDE_rectangle_to_cylinders_volume_ratio_l2024_202429

theorem rectangle_to_cylinders_volume_ratio :
  let rectangle_width : ℝ := 7
  let rectangle_length : ℝ := 10
  let cylinder1_radius : ℝ := rectangle_width / (2 * Real.pi)
  let cylinder1_height : ℝ := rectangle_length
  let cylinder1_volume : ℝ := Real.pi * cylinder1_radius^2 * cylinder1_height
  let cylinder2_radius : ℝ := rectangle_length / (2 * Real.pi)
  let cylinder2_height : ℝ := rectangle_width
  let cylinder2_volume : ℝ := Real.pi * cylinder2_radius^2 * cylinder2_height
  let larger_volume : ℝ := max cylinder1_volume cylinder2_volume
  let smaller_volume : ℝ := min cylinder1_volume cylinder2_volume
  larger_volume / smaller_volume = 10 / 7 := by sorry

end NUMINAMATH_CALUDE_rectangle_to_cylinders_volume_ratio_l2024_202429


namespace NUMINAMATH_CALUDE_larger_box_capacity_l2024_202437

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  height : ℝ
  width : ℝ
  length : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ :=
  d.height * d.width * d.length

/-- The capacity of clay a box can carry -/
def boxCapacity (volume : ℝ) (clayPerUnit : ℝ) : ℝ :=
  volume * clayPerUnit

theorem larger_box_capacity 
  (small_box : BoxDimensions)
  (small_box_clay : ℝ)
  (h_small_height : small_box.height = 1)
  (h_small_width : small_box.width = 2)
  (h_small_length : small_box.length = 4)
  (h_small_capacity : small_box_clay = 30) :
  let large_box : BoxDimensions := {
    height := 3 * small_box.height,
    width := 2 * small_box.width,
    length := 2 * small_box.length
  }
  let small_volume := boxVolume small_box
  let large_volume := boxVolume large_box
  let clay_per_unit := small_box_clay / small_volume
  boxCapacity large_volume clay_per_unit = 360 := by
sorry

end NUMINAMATH_CALUDE_larger_box_capacity_l2024_202437


namespace NUMINAMATH_CALUDE_max_value_of_sum_and_powers_l2024_202447

theorem max_value_of_sum_and_powers (x y z : ℝ) : 
  x ≥ 0 → y ≥ 0 → z ≥ 0 → x + y + z = 2 → 
  ∃ (max : ℝ), max = 2 ∧ ∀ (a b c : ℝ), a ≥ 0 → b ≥ 0 → c ≥ 0 → a + b + c = 2 → 
  a + b^3 + c^4 ≤ max := by
sorry

end NUMINAMATH_CALUDE_max_value_of_sum_and_powers_l2024_202447


namespace NUMINAMATH_CALUDE_farm_animals_difference_l2024_202481

theorem farm_animals_difference (initial_horses : ℕ) (initial_cows : ℕ) : 
  initial_horses = 4 * initial_cows →  -- Initial ratio condition
  (initial_horses - 15) / (initial_cows + 15) = 13 / 7 →  -- New ratio condition
  initial_horses - 15 - (initial_cows + 15) = 30 :=  -- Difference after transaction
by
  sorry

end NUMINAMATH_CALUDE_farm_animals_difference_l2024_202481


namespace NUMINAMATH_CALUDE_impossible_to_fill_board_l2024_202475

/-- Represents a piece on the board -/
inductive Piece
  | Regular
  | Special

/-- Represents the color of a square -/
inductive Color
  | White
  | Grey

/-- Represents the board configuration -/
structure Board :=
  (rows : Nat)
  (cols : Nat)
  (total_squares : Nat)
  (white_squares : Nat)
  (grey_squares : Nat)

/-- Represents the coverage of a piece -/
structure PieceCoverage :=
  (white : Nat)
  (grey : Nat)

/-- The board configuration -/
def puzzle_board : Board :=
  { rows := 5
  , cols := 8
  , total_squares := 40
  , white_squares := 20
  , grey_squares := 20 }

/-- The coverage of a regular piece -/
def regular_coverage : PieceCoverage :=
  { white := 2, grey := 2 }

/-- The coverage of the special piece -/
def special_coverage : PieceCoverage :=
  { white := 3, grey := 1 }

/-- The theorem to be proved -/
theorem impossible_to_fill_board : 
  ∀ (special_piece_count : Nat) (regular_piece_count : Nat),
    special_piece_count = 1 →
    regular_piece_count = 9 →
    ¬ (special_piece_count * special_coverage.white + regular_piece_count * regular_coverage.white = puzzle_board.white_squares ∧
       special_piece_count * special_coverage.grey + regular_piece_count * regular_coverage.grey = puzzle_board.grey_squares) :=
by sorry

end NUMINAMATH_CALUDE_impossible_to_fill_board_l2024_202475


namespace NUMINAMATH_CALUDE_reflection_about_x_axis_l2024_202432

theorem reflection_about_x_axis (a : ℝ) : 
  (∃ (A B : ℝ × ℝ), 
    A = (3, a) ∧ 
    B = (3, 4) ∧ 
    A.1 = B.1 ∧ 
    A.2 = -B.2) → 
  a = -4 := by sorry

end NUMINAMATH_CALUDE_reflection_about_x_axis_l2024_202432


namespace NUMINAMATH_CALUDE_hyperbola_triangle_l2024_202498

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x * y = 1

-- Define the branches of the hyperbola
def C₁ (x y : ℝ) : Prop := hyperbola x y ∧ x > 0
def C₂ (x y : ℝ) : Prop := hyperbola x y ∧ x < 0

-- Define a regular triangle
def regular_triangle (P Q R : ℝ × ℝ) : Prop :=
  let (px, py) := P
  let (qx, qy) := Q
  let (rx, ry) := R
  (px - qx)^2 + (py - qy)^2 = (qx - rx)^2 + (qy - ry)^2 ∧
  (qx - rx)^2 + (qy - ry)^2 = (rx - px)^2 + (ry - py)^2

-- Theorem statement
theorem hyperbola_triangle :
  ∀ (Q R : ℝ × ℝ),
  let P := (-1, -1)
  regular_triangle P Q R ∧
  C₂ P.1 P.2 ∧
  C₁ Q.1 Q.2 ∧
  C₁ R.1 R.2 →
  (¬(C₁ P.1 P.2 ∧ C₁ Q.1 Q.2 ∧ C₁ R.1 R.2) ∧
   ¬(C₂ P.1 P.2 ∧ C₂ Q.1 Q.2 ∧ C₂ R.1 R.2)) ∧
  Q = (2 - Real.sqrt 3, 2 + Real.sqrt 3) ∧
  R = (2 + Real.sqrt 3, 2 - Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_triangle_l2024_202498


namespace NUMINAMATH_CALUDE_smallest_integer_in_ratio_l2024_202415

theorem smallest_integer_in_ratio (a b c : ℕ) : 
  a > 0 → b > 0 → c > 0 →
  a + b + c = 60 →
  2 * a = 3 * b →
  3 * b = 5 * c →
  a = 6 := by
sorry

end NUMINAMATH_CALUDE_smallest_integer_in_ratio_l2024_202415


namespace NUMINAMATH_CALUDE_parabola_vertex_l2024_202406

/-- Given a quadratic function f(x) = -x^2 + cx + d where f(x) ≤ 0 has solutions [1,∞) and (-∞,-7],
    prove that the vertex of the parabola is (-3, 16) -/
theorem parabola_vertex (c d : ℝ) 
    (h1 : ∀ x ≥ 1, -x^2 + c*x + d ≤ 0)
    (h2 : ∀ x ≤ -7, -x^2 + c*x + d ≤ 0)
    (h3 : ∃ x > -7, -x^2 + c*x + d > 0)
    (h4 : ∃ x < 1, -x^2 + c*x + d > 0) :
    let f := fun x => -x^2 + c*x + d
    let vertex := (-3, 16)
    ∀ x, f x ≤ f vertex.1 ∧ f vertex.1 = vertex.2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2024_202406


namespace NUMINAMATH_CALUDE_P_intersect_Q_equals_target_l2024_202485

-- Define the sets P and Q
def P : Set ℝ := {x | ∃ y, y = Real.log (2 - x)}
def Q : Set ℝ := {x | x^2 - 5*x + 4 ≤ 0}

-- State the theorem
theorem P_intersect_Q_equals_target : P ∩ Q = {x | 1 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_P_intersect_Q_equals_target_l2024_202485


namespace NUMINAMATH_CALUDE_quadratic_real_solutions_l2024_202473

theorem quadratic_real_solutions (a : ℝ) :
  (∃ x : ℝ, a * x^2 - 4*x - 1 = 0) ↔ a ≥ -4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_real_solutions_l2024_202473


namespace NUMINAMATH_CALUDE_cos_sin_power_eight_identity_l2024_202480

theorem cos_sin_power_eight_identity (α : ℝ) : 
  Real.cos α ^ 8 - Real.sin α ^ 8 = Real.cos (2 * α) * ((3 + Real.cos (4 * α)) / 4) := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_power_eight_identity_l2024_202480


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l2024_202441

theorem sum_of_reciprocals (x y : ℚ) 
  (h1 : x⁻¹ + y⁻¹ = 4)
  (h2 : x⁻¹ - y⁻¹ = 8) : 
  x + y = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l2024_202441


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2024_202452

def M : Set ℝ := {x | x ≥ 2}
def N : Set ℝ := {x | x^2 - 25 < 0}

theorem intersection_of_M_and_N : M ∩ N = {x | 2 ≤ x ∧ x < 5} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2024_202452


namespace NUMINAMATH_CALUDE_binary_representation_253_l2024_202440

def decimal_to_binary (n : ℕ) : List Bool :=
  sorry

def count_ones (binary : List Bool) : ℕ :=
  sorry

def count_zeros (binary : List Bool) : ℕ :=
  sorry

theorem binary_representation_253 :
  let binary := decimal_to_binary 253
  let y := count_ones binary
  let x := count_zeros binary
  y - x = 6 := by sorry

end NUMINAMATH_CALUDE_binary_representation_253_l2024_202440


namespace NUMINAMATH_CALUDE_corn_ears_per_stalk_l2024_202499

/-- The number of corn stalks -/
def num_stalks : ℕ := 108

/-- The number of kernels in half of the ears -/
def kernels_half1 : ℕ := 500

/-- The number of kernels in the other half of the ears -/
def kernels_half2 : ℕ := 600

/-- The total number of kernels -/
def total_kernels : ℕ := 237600

/-- The number of ears per stalk -/
def ears_per_stalk : ℕ := 4

theorem corn_ears_per_stalk :
  num_stalks * (ears_per_stalk / 2 * kernels_half1 + ears_per_stalk / 2 * kernels_half2) = total_kernels :=
by sorry

end NUMINAMATH_CALUDE_corn_ears_per_stalk_l2024_202499


namespace NUMINAMATH_CALUDE_union_equals_A_l2024_202455

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

-- Define set B
def B : Set ℝ := {-1, 0, 1, 2, 3}

-- Theorem statement
theorem union_equals_A : A ∪ B = A := by
  sorry

end NUMINAMATH_CALUDE_union_equals_A_l2024_202455


namespace NUMINAMATH_CALUDE_water_from_river_calculation_l2024_202439

/-- The amount of water Jacob collects from the river daily -/
def water_from_river : ℕ := 1700

/-- Jacob's water tank capacity in milliliters -/
def tank_capacity : ℕ := 50000

/-- Water collected from rain daily in milliliters -/
def water_from_rain : ℕ := 800

/-- Number of days to fill the tank -/
def days_to_fill : ℕ := 20

/-- Theorem stating that the amount of water Jacob collects from the river daily is 1700 milliliters -/
theorem water_from_river_calculation :
  water_from_river = (tank_capacity - water_from_rain * days_to_fill) / days_to_fill :=
by sorry

end NUMINAMATH_CALUDE_water_from_river_calculation_l2024_202439


namespace NUMINAMATH_CALUDE_initial_crayons_l2024_202478

theorem initial_crayons (taken_out : ℕ) (left : ℕ) : 
  taken_out = 3 → left = 4 → taken_out + left = 7 :=
by sorry

end NUMINAMATH_CALUDE_initial_crayons_l2024_202478


namespace NUMINAMATH_CALUDE_value_calculation_l2024_202453

theorem value_calculation (number : ℕ) (value : ℕ) : 
  number = 48 → value = (number / 4 + 15) → value = 27 := by sorry

end NUMINAMATH_CALUDE_value_calculation_l2024_202453


namespace NUMINAMATH_CALUDE_sqrt_relation_l2024_202493

theorem sqrt_relation (h : Real.sqrt 262.44 = 16.2) : Real.sqrt 2.6244 = 1.62 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_relation_l2024_202493


namespace NUMINAMATH_CALUDE_golden_state_team_points_l2024_202438

/-- The number of points earned by Draymond in the Golden State Team -/
def draymondPoints : ℕ := 12

/-- The total points earned by the Golden State Team -/
def totalTeamPoints : ℕ := 69

/-- The number of points earned by Kelly -/
def kellyPoints : ℕ := 9

theorem golden_state_team_points :
  ∃ (D : ℕ), 
    D = draymondPoints ∧
    D + 2*D + kellyPoints + 2*kellyPoints + D/2 = totalTeamPoints :=
by sorry

end NUMINAMATH_CALUDE_golden_state_team_points_l2024_202438


namespace NUMINAMATH_CALUDE_point_on_x_axis_with_distance_3_l2024_202420

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the distance from a point to the y-axis
def distToYAxis (p : Point2D) : ℝ := |p.x|

-- Theorem statement
theorem point_on_x_axis_with_distance_3 (P : Point2D) :
  P.y = 0 ∧ distToYAxis P = 3 → P = ⟨3, 0⟩ ∨ P = ⟨-3, 0⟩ := by
  sorry

end NUMINAMATH_CALUDE_point_on_x_axis_with_distance_3_l2024_202420


namespace NUMINAMATH_CALUDE_smallest_n_digits_l2024_202459

/-- Sum of digits function -/
def sum_of_digits (k : ℕ) : ℕ := sorry

/-- Theorem stating the number of digits in the smallest n satisfying the condition -/
theorem smallest_n_digits :
  ∃ n : ℕ,
    (∀ m : ℕ, m < n → sum_of_digits m - sum_of_digits (5 * m) ≠ 2013) ∧
    (sum_of_digits n - sum_of_digits (5 * n) = 2013) ∧
    (Nat.digits 10 n).length = 224 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_digits_l2024_202459


namespace NUMINAMATH_CALUDE_ellipse_tangent_quadrilateral_area_l2024_202431

/-- Given an ellipse with equation 9x^2 + 25y^2 = 225, 
    the area of the quadrilateral formed by tangents at the parameter endpoints is 62.5 -/
theorem ellipse_tangent_quadrilateral_area :
  let a : ℝ := 5  -- semi-major axis
  let b : ℝ := 3  -- semi-minor axis
  ∀ x y : ℝ, 9 * x^2 + 25 * y^2 = 225 →
  let area := 2 * a^3 / Real.sqrt (a^2 - b^2)
  area = 62.5 := by
sorry


end NUMINAMATH_CALUDE_ellipse_tangent_quadrilateral_area_l2024_202431


namespace NUMINAMATH_CALUDE_sin_difference_product_l2024_202401

theorem sin_difference_product (a b : ℝ) : 
  Real.sin (2 * a + b) - Real.sin (2 * a - b) = 2 * Real.cos (2 * a) * Real.sin b := by
  sorry

end NUMINAMATH_CALUDE_sin_difference_product_l2024_202401


namespace NUMINAMATH_CALUDE_fuel_cost_calculation_l2024_202494

theorem fuel_cost_calculation (original_cost : ℝ) (capacity_increase : ℝ) (price_increase : ℝ) : 
  original_cost = 200 → 
  capacity_increase = 2 → 
  price_increase = 1.2 → 
  original_cost * capacity_increase * price_increase = 480 := by
sorry

end NUMINAMATH_CALUDE_fuel_cost_calculation_l2024_202494


namespace NUMINAMATH_CALUDE_complex_multiplication_l2024_202469

theorem complex_multiplication (i : ℂ) (z₁ z₂ : ℂ) :
  i * i = -1 →
  z₁ = 1 + 2 * i →
  z₂ = -3 * i →
  z₁ * z₂ = 6 - 3 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_multiplication_l2024_202469


namespace NUMINAMATH_CALUDE_complex_combination_equality_l2024_202405

/-- Given complex numbers A, M, S, P, and Q, prove that their combination equals 6 - 5i -/
theorem complex_combination_equality (A M S P Q : ℂ) : 
  A = 5 - 4*I ∧ 
  M = -5 + 2*I ∧ 
  S = 2*I ∧ 
  P = 3 ∧ 
  Q = 1 + I → 
  A - M + S - P - Q = 6 - 5*I :=
by sorry

end NUMINAMATH_CALUDE_complex_combination_equality_l2024_202405


namespace NUMINAMATH_CALUDE_max_determinable_elements_l2024_202491

open Finset

theorem max_determinable_elements : ∀ (a : Fin 11 → ℕ) (b : Fin 9 → ℕ),
  (∀ i : Fin 11, a i ∈ range 12 \ {0}) →
  (∀ i j : Fin 11, i ≠ j → a i ≠ a j) →
  (∀ i : Fin 9, b i = a i + a (i + 2)) →
  (∃ (S : Finset (Fin 11)), S.card = 5 ∧ 
    (∀ (a' : Fin 11 → ℕ),
      (∀ i : Fin 11, a' i ∈ range 12 \ {0}) →
      (∀ i j : Fin 11, i ≠ j → a' i ≠ a' j) →
      (∀ i : Fin 9, b i = a' i + a' (i + 2)) →
      (∀ i ∈ S, a i = a' i))) ∧
  ¬(∃ (S : Finset (Fin 11)), S.card > 5 ∧ 
    (∀ (a' : Fin 11 → ℕ),
      (∀ i : Fin 11, a' i ∈ range 12 \ {0}) →
      (∀ i j : Fin 11, i ≠ j → a' i ≠ a' j) →
      (∀ i : Fin 9, b i = a' i + a' (i + 2)) →
      (∀ i ∈ S, a i = a' i))) := by
  sorry

end NUMINAMATH_CALUDE_max_determinable_elements_l2024_202491


namespace NUMINAMATH_CALUDE_candies_eaten_l2024_202468

theorem candies_eaten (initial : ℕ) (remaining : ℕ) (eaten : ℕ) : 
  initial = 23 → remaining = 7 → eaten = initial - remaining → eaten = 16 := by sorry

end NUMINAMATH_CALUDE_candies_eaten_l2024_202468


namespace NUMINAMATH_CALUDE_mixture_problem_l2024_202402

/-- A mixture problem involving milk and water ratios -/
theorem mixture_problem (x : ℝ) (h1 : x > 0) : 
  (4 * x) / x = 4 →                  -- Initial ratio of milk to water is 4:1
  (4 * x) / (x + 9) = 2 →            -- Final ratio after adding 9 litres of water is 2:1
  5 * x = 45 :=                      -- Initial volume of the mixture is 45 litres
by
  sorry

end NUMINAMATH_CALUDE_mixture_problem_l2024_202402


namespace NUMINAMATH_CALUDE_hexagon_problem_l2024_202490

/-- Regular hexagon with side length 3 -/
structure RegularHexagon :=
  (A B C D E F : ℝ × ℝ)
  (side_length : ℝ)
  (regular : side_length = 3)

/-- L is the intersection point of diagonals CE and DF -/
def L (h : RegularHexagon) : ℝ × ℝ := sorry

/-- K is defined such that LK = 3AB - AC -/
def K (h : RegularHexagon) : ℝ × ℝ := sorry

/-- Determine if a point is outside the hexagon -/
def is_outside (h : RegularHexagon) (p : ℝ × ℝ) : Prop := sorry

/-- Calculate the distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem hexagon_problem (h : RegularHexagon) :
  is_outside h (K h) ∧ distance (K h) h.C = 3 * Real.sqrt 7 := by sorry

end NUMINAMATH_CALUDE_hexagon_problem_l2024_202490


namespace NUMINAMATH_CALUDE_distribute_planets_l2024_202430

/-- The number of ways to distribute units among distinct objects --/
def distribute_units (total_units : ℕ) (earth_like : ℕ) (mars_like : ℕ) (earth_units : ℕ) (mars_units : ℕ) : ℕ :=
  sorry

theorem distribute_planets :
  distribute_units 15 7 8 3 1 = 2961 :=
sorry

end NUMINAMATH_CALUDE_distribute_planets_l2024_202430


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l2024_202489

theorem fractional_equation_solution :
  ∃ (x : ℝ), x ≠ 0 ∧ x ≠ 2 ∧ (2 / x - 1 / (x - 2) = 0) ∧ x = 4 :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l2024_202489


namespace NUMINAMATH_CALUDE_max_x_value_l2024_202410

theorem max_x_value (x y z : ℝ) 
  (sum_eq : x + y + z = 7) 
  (prod_sum_eq : x * y + x * z + y * z = 12) : 
  x ≤ (14 + 2 * Real.sqrt 46) / 6 := by
sorry

end NUMINAMATH_CALUDE_max_x_value_l2024_202410
