import Mathlib

namespace smallest_integer_satisfying_inequality_four_satisfies_inequality_four_is_smallest_integer_l3956_395640

theorem smallest_integer_satisfying_inequality :
  ∀ x : ℤ, (x : ℚ) / 4 + 3 / 7 > 4 / 3 → x ≥ 4 :=
by
  sorry

theorem four_satisfies_inequality :
  (4 : ℚ) / 4 + 3 / 7 > 4 / 3 :=
by
  sorry

theorem four_is_smallest_integer :
  ∀ x : ℤ, x < 4 → (x : ℚ) / 4 + 3 / 7 ≤ 4 / 3 :=
by
  sorry

end smallest_integer_satisfying_inequality_four_satisfies_inequality_four_is_smallest_integer_l3956_395640


namespace correct_calculation_l3956_395632

theorem correct_calculation : 
  (5 + (-6) = -1) ∧ 
  (1 / Real.sqrt 2 ≠ Real.sqrt 2) ∧ 
  (3 * (-2) ≠ 6) ∧ 
  (Real.sin (30 * π / 180) ≠ Real.sqrt 3 / 3) :=
by sorry

end correct_calculation_l3956_395632


namespace modulus_of_complex_number_l3956_395676

theorem modulus_of_complex_number : 
  let z : ℂ := (1 - I) / (2 * I + 1) * I
  (∃ (k : ℝ), z = k * I) → Complex.abs z = Real.sqrt 10 / 5 := by
  sorry

end modulus_of_complex_number_l3956_395676


namespace incorrect_statement_l3956_395627

def A (k : ℕ) : Set ℤ := {x : ℤ | ∃ n : ℤ, x = 4*n + k}

theorem incorrect_statement :
  ¬ (∀ a b : ℤ, (a + b) ∈ A 3 → (a ∈ A 1 ∧ b ∈ A 2)) :=
by sorry

end incorrect_statement_l3956_395627


namespace bridge_length_l3956_395682

/-- The length of the bridge given train crossing times and train length -/
theorem bridge_length
  (train_length : ℝ)
  (bridge_crossing_time : ℝ)
  (lamppost_crossing_time : ℝ)
  (h1 : train_length = 600)
  (h2 : bridge_crossing_time = 70)
  (h3 : lamppost_crossing_time = 20) :
  ∃ (bridge_length : ℝ), bridge_length = 1500 := by
  sorry


end bridge_length_l3956_395682


namespace class_average_l3956_395629

theorem class_average (total_students : Nat) (perfect_scores : Nat) (zero_scores : Nat) (rest_average : Nat) : 
  total_students = 20 →
  perfect_scores = 2 →
  zero_scores = 3 →
  rest_average = 40 →
  (perfect_scores * 100 + zero_scores * 0 + (total_students - perfect_scores - zero_scores) * rest_average) / total_students = 40 := by
sorry

end class_average_l3956_395629


namespace quarters_in_school_year_l3956_395615

/-- The number of quarters in a school year -/
def quarters_per_year : ℕ := 4

/-- The number of students in the art club -/
def students : ℕ := 15

/-- The number of artworks each student makes per quarter -/
def artworks_per_student_per_quarter : ℕ := 2

/-- The total number of artworks collected in two school years -/
def total_artworks : ℕ := 240

/-- Theorem stating that the number of quarters in a school year is 4 -/
theorem quarters_in_school_year : 
  quarters_per_year * 2 * students * artworks_per_student_per_quarter = total_artworks :=
by sorry

end quarters_in_school_year_l3956_395615


namespace sin_sum_equality_l3956_395622

theorem sin_sum_equality : 
  Real.sin (17 * π / 180) * Real.sin (223 * π / 180) + 
  Real.sin (253 * π / 180) * Real.sin (313 * π / 180) = 1 / 2 := by
  sorry

end sin_sum_equality_l3956_395622


namespace grunters_win_probability_l3956_395657

/-- The probability of winning a single game for the Grunters -/
def p : ℚ := 3/5

/-- The number of games played -/
def n : ℕ := 5

/-- The probability of winning all games -/
def win_all : ℚ := p^n

theorem grunters_win_probability : win_all = 243/3125 := by
  sorry

end grunters_win_probability_l3956_395657


namespace hyperbola_intersection_range_l3956_395600

-- Define the hyperbola C
def hyperbola_C (x y : ℝ) : Prop :=
  x^2 / 3 - y^2 = 1

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x + Real.sqrt 2

-- Define the condition for intersection points
def distinct_intersection (k : ℝ) : Prop :=
  ∀ x₁ y₁ x₂ y₂, x₁ ≠ x₂ →
    hyperbola_C x₁ y₁ ∧ hyperbola_C x₂ y₂ ∧
    line_l k x₁ y₁ ∧ line_l k x₂ y₂

-- Define the dot product condition
def dot_product_condition (k : ℝ) : Prop :=
  ∀ x₁ y₁ x₂ y₂,
    hyperbola_C x₁ y₁ ∧ hyperbola_C x₂ y₂ ∧
    line_l k x₁ y₁ ∧ line_l k x₂ y₂ →
    x₁ * x₂ + y₁ * y₂ > 0

-- Main theorem
theorem hyperbola_intersection_range :
  ∀ k : ℝ, distinct_intersection k ∧ dot_product_condition k ↔
    (k > -1 ∧ k < -Real.sqrt 3 / 3) ∨ (k > Real.sqrt 3 / 3 ∧ k < 1) :=
sorry

end hyperbola_intersection_range_l3956_395600


namespace saline_solution_concentration_l3956_395602

/-- Proves that mixing 100 kg of 30% saline solution with 200 kg of pure water
    results in a final saline solution with a concentration of 10%. -/
theorem saline_solution_concentration
  (initial_solution_weight : ℝ)
  (initial_concentration : ℝ)
  (pure_water_weight : ℝ)
  (h1 : initial_solution_weight = 100)
  (h2 : initial_concentration = 0.3)
  (h3 : pure_water_weight = 200) :
  let salt_weight := initial_solution_weight * initial_concentration
  let total_weight := initial_solution_weight + pure_water_weight
  let final_concentration := salt_weight / total_weight
  final_concentration = 0.1 := by
sorry

end saline_solution_concentration_l3956_395602


namespace ABABABA_probability_l3956_395625

/-- The number of tiles marked A -/
def num_A : ℕ := 4

/-- The number of tiles marked B -/
def num_B : ℕ := 3

/-- The total number of tiles -/
def total_tiles : ℕ := num_A + num_B

/-- The number of favorable arrangements (ABABABA) -/
def favorable_arrangements : ℕ := 1

/-- The probability of the specific arrangement ABABABA -/
def probability_ABABABA : ℚ := favorable_arrangements / (total_tiles.choose num_A)

theorem ABABABA_probability : probability_ABABABA = 1 / 35 := by
  sorry

end ABABABA_probability_l3956_395625


namespace gake_uses_fewer_boards_l3956_395665

/-- Represents the width of a character in centimeters -/
def char_width : ℕ := 9

/-- Represents the width of a board in centimeters -/
def board_width : ℕ := 5

/-- Calculates the number of boards needed for a given total width -/
def boards_needed (total_width : ℕ) : ℕ :=
  (total_width + board_width - 1) / board_width

/-- Represents Tom's message -/
def tom_message : String := "MMO"

/-- Represents Gake's message -/
def gake_message : String := "2020"

/-- Calculates the total width needed for a message -/
def message_width (msg : String) : ℕ :=
  msg.length * char_width

theorem gake_uses_fewer_boards :
  boards_needed (message_width gake_message) < boards_needed (message_width tom_message) := by
  sorry

#eval boards_needed (message_width tom_message)
#eval boards_needed (message_width gake_message)

end gake_uses_fewer_boards_l3956_395665


namespace first_oil_price_l3956_395645

/-- Given two oils mixed together, prove the price of the first oil. -/
theorem first_oil_price 
  (first_oil_volume : ℝ) 
  (second_oil_volume : ℝ) 
  (second_oil_price : ℝ) 
  (mixture_price : ℝ)
  (h1 : first_oil_volume = 10)
  (h2 : second_oil_volume = 5)
  (h3 : second_oil_price = 66)
  (h4 : mixture_price = 58) :
  ∃ (first_oil_price : ℝ), 
    first_oil_price = 54 ∧ 
    first_oil_price * first_oil_volume + second_oil_price * second_oil_volume = 
      mixture_price * (first_oil_volume + second_oil_volume) := by
  sorry

end first_oil_price_l3956_395645


namespace base_height_example_l3956_395618

/-- Given a sculpture height in feet and inches, and a total height of sculpture and base,
    calculate the height of the base in feet. -/
def base_height (sculpture_feet : ℕ) (sculpture_inches : ℕ) (total_height : ℚ) : ℚ :=
  total_height - (sculpture_feet : ℚ) - ((sculpture_inches : ℚ) / 12)

/-- Theorem stating that for a sculpture of 2 feet 10 inches and a total height of 3.5 feet,
    the base height is 2/3 feet. -/
theorem base_height_example : base_height 2 10 (7/2) = 2/3 := by
  sorry

end base_height_example_l3956_395618


namespace battle_station_staffing_l3956_395617

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def permutations (n k : ℕ) : ℕ := 
  factorial n / factorial (n - k)

theorem battle_station_staffing :
  permutations 15 5 = 360360 := by
  sorry

end battle_station_staffing_l3956_395617


namespace tan_sixty_minus_reciprocal_tan_thirty_equals_zero_l3956_395654

theorem tan_sixty_minus_reciprocal_tan_thirty_equals_zero :
  Real.tan (60 * π / 180) - (1 / Real.tan (30 * π / 180)) = 0 := by
  sorry

end tan_sixty_minus_reciprocal_tan_thirty_equals_zero_l3956_395654


namespace binomial_coefficient_seven_four_l3956_395634

theorem binomial_coefficient_seven_four : (7 : ℕ).choose 4 = 35 := by
  sorry

end binomial_coefficient_seven_four_l3956_395634


namespace num_mc_questions_is_two_l3956_395688

/-- The number of true-false questions in the quiz -/
def num_tf : ℕ := 4

/-- The number of answer choices for each multiple-choice question -/
def num_mc_choices : ℕ := 4

/-- The total number of ways to write the answer key -/
def total_ways : ℕ := 224

/-- The number of ways to answer the true-false questions, excluding all-same answers -/
def tf_ways : ℕ := 2^num_tf - 2

/-- Theorem stating that the number of multiple-choice questions is 2 -/
theorem num_mc_questions_is_two :
  ∃ (n : ℕ), tf_ways * (num_mc_choices^n) = total_ways ∧ n = 2 := by
  sorry

end num_mc_questions_is_two_l3956_395688


namespace steve_earnings_l3956_395680

/-- Calculates an author's earnings from book sales after agent's commission --/
def author_earnings (copies_sold : ℕ) (price_per_copy : ℚ) (agent_commission_rate : ℚ) : ℚ :=
  let total_revenue := copies_sold * price_per_copy
  let agent_commission := total_revenue * agent_commission_rate
  total_revenue - agent_commission

/-- Proves that given the specified conditions, the author's earnings are $1,800,000 --/
theorem steve_earnings :
  author_earnings 1000000 2 (1/10) = 1800000 := by
  sorry

end steve_earnings_l3956_395680


namespace find_y_l3956_395633

theorem find_y (x : ℝ) (y : ℝ) (h1 : 1.5 * x = 0.3 * y) (h2 : x = 20) : y = 100 := by
  sorry

end find_y_l3956_395633


namespace simplified_sqrt_expression_l3956_395677

theorem simplified_sqrt_expression (x : ℝ) : 
  Real.sqrt (9 * x^4 + 3 * x^2) = Real.sqrt 3 * |x| * Real.sqrt (3 * x^2 + 1) := by
  sorry

end simplified_sqrt_expression_l3956_395677


namespace max_q_minus_r_for_852_l3956_395693

theorem max_q_minus_r_for_852 :
  ∃ (q r : ℕ), 
    q > 0 ∧ r > 0 ∧ 
    852 = 21 * q + r ∧
    ∀ (q' r' : ℕ), q' > 0 → r' > 0 → 852 = 21 * q' + r' → q' - r' ≤ q - r ∧
    q - r = 28 :=
sorry

end max_q_minus_r_for_852_l3956_395693


namespace isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l3956_395673

/-- An isosceles triangle with side lengths 4 and 9 has a perimeter of 22. -/
theorem isosceles_triangle_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun a b c =>
    (a = 4 ∨ a = 9) ∧ (b = 4 ∨ b = 9) ∧ (c = 4 ∨ c = 9) ∧  -- Side lengths are 4 or 9
    (a = b ∨ b = c ∨ a = c) ∧                              -- Isosceles condition
    (a + b > c ∧ b + c > a ∧ a + c > b) →                  -- Triangle inequality
    a + b + c = 22                                         -- Perimeter is 22

/-- Proof of the theorem -/
theorem isosceles_triangle_perimeter_proof : ∃ a b c, isosceles_triangle_perimeter a b c :=
  sorry

end isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l3956_395673


namespace opposite_of_negative_seven_l3956_395635

theorem opposite_of_negative_seven :
  ∀ x : ℤ, x + (-7) = 0 → x = 7 :=
by sorry

end opposite_of_negative_seven_l3956_395635


namespace max_cube_hemisphere_ratio_l3956_395671

/-- The maximum ratio of the volume of a cube inscribed in a hemisphere to the volume of the hemisphere -/
theorem max_cube_hemisphere_ratio : 
  let r := Real.sqrt 6 / (3 * Real.pi)
  ∃ (R a : ℝ), R > 0 ∧ a > 0 ∧
  (a^2 + (Real.sqrt 2 * a / 2)^2 = R^2) ∧
  (∀ (b : ℝ), b > 0 → b^2 + (Real.sqrt 2 * b / 2)^2 ≤ R^2 → 
    b^3 / ((2/3) * Real.pi * R^3) ≤ r) ∧
  a^3 / ((2/3) * Real.pi * R^3) = r :=
sorry

end max_cube_hemisphere_ratio_l3956_395671


namespace forty_bees_honey_l3956_395656

/-- The amount of honey (in grams) produced by one honey bee in 40 days -/
def honey_per_bee : ℕ := 1

/-- The number of honey bees -/
def num_bees : ℕ := 40

/-- The amount of honey (in grams) produced by a group of honey bees in 40 days -/
def total_honey (bees : ℕ) : ℕ := bees * honey_per_bee

/-- Theorem stating that 40 honey bees produce 40 grams of honey in 40 days -/
theorem forty_bees_honey : total_honey num_bees = 40 := by
  sorry

end forty_bees_honey_l3956_395656


namespace fraction_1800_1809_is_7_30_l3956_395686

/-- The number of states that joined the union from 1800 to 1809 -/
def states_1800_1809 : ℕ := 7

/-- The total number of states considered (first 30 states) -/
def total_states : ℕ := 30

/-- The fraction of states that joined from 1800 to 1809 out of the first 30 states -/
def fraction_1800_1809 : ℚ := states_1800_1809 / total_states

theorem fraction_1800_1809_is_7_30 : fraction_1800_1809 = 7 / 30 := by
  sorry

end fraction_1800_1809_is_7_30_l3956_395686


namespace sin_negative_1740_degrees_l3956_395603

theorem sin_negative_1740_degrees : Real.sin ((-1740 : ℝ) * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end sin_negative_1740_degrees_l3956_395603


namespace arithmetic_sequence_problem_l3956_395623

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : a 1 + 3 * a 8 + a 15 = 60) : 
  2 * a 9 - a 10 = 12 := by
  sorry

end arithmetic_sequence_problem_l3956_395623


namespace smallest_multiple_l3956_395613

theorem smallest_multiple (n : ℕ) : n = 204 ↔ 
  n > 0 ∧ 
  17 ∣ n ∧ 
  n % 43 = 11 ∧ 
  ∀ m : ℕ, m > 0 → 17 ∣ m → m % 43 = 11 → n ≤ m :=
by sorry

end smallest_multiple_l3956_395613


namespace inequality_solution_l3956_395604

theorem inequality_solution (a b : ℝ) (h1 : ∀ x, (2*a - b)*x + a - 5*b > 0 ↔ x < 10/7) :
  ∀ x, a*x + b > 0 ↔ x < -3/5 :=
sorry

end inequality_solution_l3956_395604


namespace expansion_coefficient_l3956_395687

theorem expansion_coefficient (n : ℕ) : 
  ((-2:ℤ)^n + n * (-2:ℤ)^(n-1) = -128) → n = 6 := by
sorry

end expansion_coefficient_l3956_395687


namespace exam_maximum_marks_l3956_395643

/-- 
Given an exam where:
1. 40% of the maximum marks are required to pass
2. A student got 40 marks
3. The student failed by 40 marks
Prove that the maximum marks for the exam are 200.
-/
theorem exam_maximum_marks :
  ∀ (max_marks : ℕ) (pass_percentage : ℚ) (student_marks : ℕ) (fail_margin : ℕ),
    pass_percentage = 40 / 100 →
    student_marks = 40 →
    fail_margin = 40 →
    (pass_percentage * max_marks : ℚ) = student_marks + fail_margin →
    max_marks = 200 := by
  sorry

end exam_maximum_marks_l3956_395643


namespace toy_piles_total_l3956_395681

theorem toy_piles_total (small_pile large_pile : ℕ) : 
  large_pile = 2 * small_pile → 
  large_pile = 80 → 
  small_pile + large_pile = 120 :=
by sorry

end toy_piles_total_l3956_395681


namespace token_game_ends_in_37_rounds_l3956_395679

/-- Represents a player in the token game -/
inductive Player : Type
  | A
  | B
  | C

/-- Represents the state of the game at any point -/
structure GameState :=
  (tokens : Player → Nat)
  (round : Nat)

/-- Determines if a player's tokens are divisible by 5 -/
def isDivisibleByFive (n : Nat) : Bool :=
  n % 5 = 0

/-- Determines the player with the most tokens -/
def playerWithMostTokens (state : GameState) : Player :=
  sorry

/-- Applies the rules of a single round to the game state -/
def applyRound (state : GameState) : GameState :=
  sorry

/-- Checks if the game has ended (i.e., a player has run out of tokens) -/
def isGameOver (state : GameState) : Bool :=
  sorry

/-- The initial state of the game -/
def initialState : GameState :=
  { tokens := λ p => match p with
    | Player.A => 17
    | Player.B => 15
    | Player.C => 14,
    round := 0 }

/-- The final state of the game -/
def finalState : GameState :=
  sorry

theorem token_game_ends_in_37_rounds :
  finalState.round = 37 ∧ isGameOver finalState :=
  sorry

end token_game_ends_in_37_rounds_l3956_395679


namespace circle_area_circumference_ratio_l3956_395669

theorem circle_area_circumference_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (π * r₁^2) / (π * r₂^2) = 16 / 25 →
  (2 * π * r₁) / (2 * π * r₂) = 4 / 5 := by
sorry

end circle_area_circumference_ratio_l3956_395669


namespace find_x_in_set_l3956_395678

theorem find_x_in_set (s : Finset ℝ) (x : ℝ) : 
  s = {8, 14, 20, 7, x, 16} →
  (Finset.sum s id) / (Finset.card s : ℝ) = 12 →
  x = 7 := by
sorry

end find_x_in_set_l3956_395678


namespace inequality_solution_l3956_395639

theorem inequality_solution (a : ℝ) :
  (a < 1/2 → ∀ x, x^2 - x + a - a^2 < 0 ↔ a < x ∧ x < 1-a) ∧
  (a > 1/2 → ∀ x, x^2 - x + a - a^2 < 0 ↔ 1-a < x ∧ x < a) ∧
  (a = 1/2 → ∀ x, ¬(x^2 - x + a - a^2 < 0)) := by
  sorry

end inequality_solution_l3956_395639


namespace min_value_expression_l3956_395659

theorem min_value_expression (x y z : ℝ) (hx : x > 1) (hy : y > 1) (hz : z > 1) :
  (x^2 / (y - 1) + y^2 / (z - 1) + z^2 / (x - 1)) ≥ 12 :=
by sorry

end min_value_expression_l3956_395659


namespace cookies_given_to_friend_l3956_395675

theorem cookies_given_to_friend (initial_cookies : ℕ) (eaten_cookies : ℕ) (remaining_cookies : ℕ) : 
  initial_cookies = 36 →
  eaten_cookies = 10 →
  remaining_cookies = 12 →
  initial_cookies - eaten_cookies - remaining_cookies = 14 := by
sorry

end cookies_given_to_friend_l3956_395675


namespace chessboard_circle_area_ratio_l3956_395695

/-- Represents a square chessboard -/
structure Chessboard where
  side_length : ℝ
  dimensions : ℕ × ℕ

/-- Represents a circle placed on the chessboard -/
structure PlacedCircle where
  radius : ℝ

/-- Calculates the sum of areas within the circle for intersected squares -/
def S₁ (board : Chessboard) (circle : PlacedCircle) : ℝ := sorry

/-- Calculates the sum of areas outside the circle for intersected squares -/
def S₂ (board : Chessboard) (circle : PlacedCircle) : ℝ := sorry

/-- The main theorem to be proved -/
theorem chessboard_circle_area_ratio
  (board : Chessboard)
  (circle : PlacedCircle)
  (h_board_side : board.side_length = 8)
  (h_board_dim : board.dimensions = (8, 8))
  (h_circle_radius : circle.radius = 4) :
  Int.floor (S₁ board circle / S₂ board circle) = 3 := by sorry

end chessboard_circle_area_ratio_l3956_395695


namespace min_dot_product_l3956_395660

def OA : ℝ × ℝ := (2, 2)
def OB : ℝ × ℝ := (4, 1)

def AP (x : ℝ) : ℝ × ℝ := (x - OA.1, -OA.2)
def BP (x : ℝ) : ℝ × ℝ := (x - OB.1, -OB.2)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem min_dot_product :
  ∃ (x : ℝ), ∀ (y : ℝ),
    dot_product (AP x) (BP x) ≤ dot_product (AP y) (BP y) ∧
    x = 3 :=
sorry

end min_dot_product_l3956_395660


namespace meaningful_fraction_range_l3956_395647

theorem meaningful_fraction_range (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 2)) ↔ x ≠ 2 := by
  sorry

end meaningful_fraction_range_l3956_395647


namespace circle_area_isosceles_triangle_l3956_395694

/-- The area of a circle circumscribing an isosceles triangle -/
theorem circle_area_isosceles_triangle (a b : ℝ) (h1 : a = 4) (h2 : b = 3) :
  let r := Real.sqrt ((a^2 / 4 + b^2 / 16))
  π * r^2 = 5.6875 * π := by
sorry

end circle_area_isosceles_triangle_l3956_395694


namespace area_between_curves_l3956_395641

theorem area_between_curves : 
  let f (x : ℝ) := Real.sqrt x
  let g (x : ℝ) := x^2
  ∫ x in (0 : ℝ)..1, (f x - g x) = (1 : ℝ) / 3 := by sorry

end area_between_curves_l3956_395641


namespace carson_clawed_39_times_l3956_395628

/-- The number of times Carson gets clawed in the zoo enclosure. -/
def total_claws (num_wombats : ℕ) (num_rheas : ℕ) (wombat_claws : ℕ) (rhea_claws : ℕ) : ℕ :=
  num_wombats * wombat_claws + num_rheas * rhea_claws

/-- Theorem stating that Carson gets clawed 39 times given the specific conditions. -/
theorem carson_clawed_39_times :
  total_claws 9 3 4 1 = 39 := by
  sorry

end carson_clawed_39_times_l3956_395628


namespace geometric_sequence_ratio_l3956_395689

/-- A geometric sequence with positive terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

/-- The first, third, and second terms form an arithmetic sequence -/
def arithmetic_condition (a : ℕ → ℝ) : Prop :=
  2 * ((1 / 2) * a 3) = a 1 + 2 * a 2

/-- All terms in the sequence are positive -/
def positive_terms (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n > 0

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  geometric_sequence a →
  positive_terms a →
  arithmetic_condition a →
  (a 9 + a 10) / (a 7 + a 8) = 3 + 2 * Real.sqrt 2 :=
sorry

end geometric_sequence_ratio_l3956_395689


namespace arithmetic_sequence_sum_ratio_l3956_395637

theorem arithmetic_sequence_sum_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, S n = (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1))) →  -- Definition of S_n
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- Arithmetic sequence condition
  S 3 / S 6 = 1 / 3 →
  S 6 / S 12 = 3 / 10 := by
sorry

end arithmetic_sequence_sum_ratio_l3956_395637


namespace solution_set_inequality_1_solution_set_inequality_2_l3956_395642

-- Part 1
theorem solution_set_inequality_1 : 
  {x : ℝ | (2 * x) / (x - 2) ≤ 1} = Set.Ici (-2) ∩ Set.Iio 2 := by sorry

-- Part 2
theorem solution_set_inequality_2 (a : ℝ) (ha : a > 0) :
  {x : ℝ | a * x^2 + 2 * x + 1 > 0} = 
    if a = 1 then 
      {x : ℝ | x ≠ -1}
    else if a > 1 then 
      Set.univ
    else 
      Set.Iic ((- 1 - Real.sqrt (1 - a)) / a) ∪ Set.Ioi ((- 1 + Real.sqrt (1 - a)) / a) := by sorry

end solution_set_inequality_1_solution_set_inequality_2_l3956_395642


namespace carol_rectangle_width_l3956_395668

/-- Given two rectangles with equal area, where one has a length of 5 inches
    and the other has dimensions of 3 inches by 40 inches,
    prove that the width of the first rectangle is 24 inches. -/
theorem carol_rectangle_width
  (length_carol : ℝ)
  (width_carol : ℝ)
  (length_jordan : ℝ)
  (width_jordan : ℝ)
  (h1 : length_carol = 5)
  (h2 : length_jordan = 3)
  (h3 : width_jordan = 40)
  (h4 : length_carol * width_carol = length_jordan * width_jordan) :
  width_carol = 24 :=
by sorry

end carol_rectangle_width_l3956_395668


namespace lukes_remaining_money_l3956_395690

/-- Converts an octal number to decimal --/
def octal_to_decimal (n : ℕ) : ℕ := sorry

/-- Calculates the remaining money after buying a ticket --/
def remaining_money (savings : ℕ) (ticket_cost : ℕ) : ℕ :=
  octal_to_decimal savings - ticket_cost

theorem lukes_remaining_money :
  remaining_money 0o5555 1200 = 1725 := by sorry

end lukes_remaining_money_l3956_395690


namespace min_value_of_function_min_value_is_three_l3956_395619

theorem min_value_of_function (x : ℝ) (h : x > 1) : x + 1 / (x - 1) ≥ 3 := by
  sorry

theorem min_value_is_three : ∃ (x : ℝ), x > 1 ∧ x + 1 / (x - 1) = 3 := by
  sorry

end min_value_of_function_min_value_is_three_l3956_395619


namespace coin_flip_expected_value_l3956_395662

def penny : ℚ := 1
def fifty_cent : ℚ := 50
def dime : ℚ := 10
def quarter : ℚ := 25

def coin_probability : ℚ := 1 / 2

def expected_value : ℚ := 
  coin_probability * penny + 
  coin_probability * fifty_cent + 
  coin_probability * dime + 
  coin_probability * quarter

theorem coin_flip_expected_value : expected_value = 43 := by
  sorry

end coin_flip_expected_value_l3956_395662


namespace book_page_increase_l3956_395607

/-- Represents a book with chapters that increase in page count -/
structure Book where
  total_pages : ℕ
  num_chapters : ℕ
  first_chapter_pages : ℕ
  page_increase : ℕ

/-- Calculates the total pages in a book based on its structure -/
def calculate_total_pages (b : Book) : ℕ :=
  b.first_chapter_pages * b.num_chapters + 
  (b.num_chapters * (b.num_chapters - 1) * b.page_increase) / 2

/-- Theorem stating the page increase for the given book specifications -/
theorem book_page_increase (b : Book) 
  (h1 : b.total_pages = 95)
  (h2 : b.num_chapters = 5)
  (h3 : b.first_chapter_pages = 13)
  (h4 : calculate_total_pages b = b.total_pages) :
  b.page_increase = 3 := by
  sorry

#eval calculate_total_pages { total_pages := 95, num_chapters := 5, first_chapter_pages := 13, page_increase := 3 }

end book_page_increase_l3956_395607


namespace hillshire_population_l3956_395653

theorem hillshire_population (num_cities : ℕ) (avg_lower : ℕ) (avg_upper : ℕ) :
  num_cities = 25 →
  avg_lower = 5000 →
  avg_upper = 5500 →
  (num_cities : ℝ) * ((avg_lower : ℝ) + (avg_upper : ℝ)) / 2 = 131250 :=
by sorry

end hillshire_population_l3956_395653


namespace mijeong_box_volume_l3956_395621

/-- The volume of a cuboid with given base area and height -/
def cuboid_volume (base_area : ℝ) (height : ℝ) : ℝ :=
  base_area * height

/-- Theorem: The volume of Mijeong's cuboid box -/
theorem mijeong_box_volume :
  cuboid_volume 14 13 = 182 := by
  sorry

end mijeong_box_volume_l3956_395621


namespace probability_x_squared_gt_one_l3956_395692

-- Define the interval
def interval : Set ℝ := Set.Icc (-2) 2

-- Define the event (x^2 > 1)
def event (x : ℝ) : Prop := x^2 > 1

-- Define the measure of the interval
def intervalMeasure : ℝ := 4

-- Define the measure of the event within the interval
def eventMeasure : ℝ := 2

-- State the theorem
theorem probability_x_squared_gt_one :
  (eventMeasure / intervalMeasure : ℝ) = 1/2 := by sorry

end probability_x_squared_gt_one_l3956_395692


namespace chicken_ratio_problem_l3956_395620

/-- Given the following conditions:
    - Wendi initially has 4 chickens
    - She increases the number of chickens by a ratio r
    - One chicken is eaten by a neighbor's dog
    - Wendi finds and brings home 6 more chickens
    - The final number of chickens is 13
    Prove that the ratio r is equal to 2 -/
theorem chicken_ratio_problem (r : ℚ) : 
  (4 * r - 1 + 6 : ℚ) = 13 → r = 2 := by sorry

end chicken_ratio_problem_l3956_395620


namespace max_both_writers_and_editors_is_13_l3956_395616

/-- Conference attendee information -/
structure ConferenceData where
  total : Nat
  writers : Nat
  editors : Nat
  both : Nat
  neither : Nat
  editors_gt_38 : editors > 38
  neither_eq_2both : neither = 2 * both
  total_sum : total = writers + editors - both + neither

/-- The maximum number of people who can be both writers and editors -/
def max_both_writers_and_editors (data : ConferenceData) : Nat :=
  13

/-- Theorem stating that 13 is the maximum number of people who can be both writers and editors -/
theorem max_both_writers_and_editors_is_13 (data : ConferenceData) 
  (h : data.total = 110 ∧ data.writers = 45) :
  max_both_writers_and_editors data = 13 := by
  sorry

#check max_both_writers_and_editors_is_13

end max_both_writers_and_editors_is_13_l3956_395616


namespace arithmetic_expression_equality_l3956_395670

theorem arithmetic_expression_equality : (2 + 3^2) * 4 - 6 / 3 + 5^2 = 67 := by
  sorry

end arithmetic_expression_equality_l3956_395670


namespace pool_area_is_30_l3956_395610

/-- The surface area of a rectangular pool -/
def pool_surface_area (width : ℝ) (length : ℝ) : ℝ := width * length

/-- Theorem: The surface area of a rectangular pool with width 3 meters and length 10 meters is 30 square meters -/
theorem pool_area_is_30 : pool_surface_area 3 10 = 30 := by
  sorry

end pool_area_is_30_l3956_395610


namespace oranges_per_glass_l3956_395631

/-- Proves that the number of oranges per glass is 2, given 12 oranges used for 6 glasses of juice -/
theorem oranges_per_glass (total_oranges : ℕ) (total_glasses : ℕ) 
  (h1 : total_oranges = 12) (h2 : total_glasses = 6) :
  total_oranges / total_glasses = 2 := by
  sorry

end oranges_per_glass_l3956_395631


namespace car_tank_size_l3956_395674

/-- Calculates the size of a car's gas tank given the advertised mileage, actual miles driven, and the difference between advertised and actual mileage. -/
theorem car_tank_size 
  (advertised_mileage : ℝ) 
  (miles_driven : ℝ) 
  (mileage_difference : ℝ) : 
  advertised_mileage = 35 →
  miles_driven = 372 →
  mileage_difference = 4 →
  miles_driven / (advertised_mileage - mileage_difference) = 12 := by
    sorry

#check car_tank_size

end car_tank_size_l3956_395674


namespace quadratic_inequality_solution_sets_l3956_395638

theorem quadratic_inequality_solution_sets (a : ℝ) :
  let S := {x : ℝ | x^2 + (a + 2) * x + 2 * a < 0}
  (a < 2 → S = {x : ℝ | -2 < x ∧ x < -a}) ∧
  (a = 2 → S = ∅) ∧
  (a > 2 → S = {x : ℝ | -a < x ∧ x < -2}) := by
sorry

end quadratic_inequality_solution_sets_l3956_395638


namespace store_purchase_divisibility_l3956_395608

theorem store_purchase_divisibility (m n k : ℕ) :
  ∃ p : ℕ, 3 * m + 4 * n + 5 * k = 11 * p →
  ∃ q : ℕ, 9 * m + n + 4 * k = 11 * q :=
by sorry

end store_purchase_divisibility_l3956_395608


namespace largest_c_for_function_range_l3956_395611

theorem largest_c_for_function_range (f : ℝ → ℝ) (c : ℝ) :
  (∀ x, f x = x^2 - 6*x + c) →
  (∃ x, f x = 4) →
  c ≤ 13 ∧ 
  (∀ d > 13, ¬∃ x, x^2 - 6*x + d = 4) :=
by sorry

end largest_c_for_function_range_l3956_395611


namespace composition_difference_l3956_395697

/-- Given two functions f and g, prove that their composition difference
    f(g(x)) - g(f(x)) equals 6x^2 - 12x + 9 for all real x. -/
theorem composition_difference (x : ℝ) : 
  let f (x : ℝ) := 3 * x^2 - 6 * x + 1
  let g (x : ℝ) := 2 * x - 1
  f (g x) - g (f x) = 6 * x^2 - 12 * x + 9 := by
  sorry

end composition_difference_l3956_395697


namespace percent_of_percent_l3956_395696

theorem percent_of_percent (y : ℝ) (hy : y ≠ 0) :
  (0.6 * 0.3 * y) / y = 0.18 := by sorry

end percent_of_percent_l3956_395696


namespace first_number_100th_group_l3956_395698

/-- The sequence term at position n -/
def sequenceTerm (n : ℕ) : ℕ := 3^(n - 1)

/-- The sum of the first n natural numbers -/
def triangularNumber (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The position of the first number in the nth group -/
def firstNumberPosition (n : ℕ) : ℕ := triangularNumber (n - 1) + 1

/-- The first number in the nth group -/
def firstNumberInGroup (n : ℕ) : ℕ := sequenceTerm (firstNumberPosition n)

theorem first_number_100th_group :
  firstNumberInGroup 100 = 3^4950 := by sorry

end first_number_100th_group_l3956_395698


namespace infinite_powers_of_two_l3956_395664

/-- A sequence of natural numbers where each term is the sum of the previous term and its last digit -/
def LastDigitSequence (a₁ : ℕ) : ℕ → ℕ
  | 0 => a₁
  | n + 1 => LastDigitSequence a₁ n + (LastDigitSequence a₁ n % 10)

/-- The theorem stating that the LastDigitSequence contains infinitely many powers of 2 -/
theorem infinite_powers_of_two (a₁ : ℕ) (h : a₁ % 5 ≠ 0) :
  ∀ N : ℕ, ∃ k : ℕ, k > N ∧ ∃ m : ℕ, LastDigitSequence a₁ k = 2^m :=
sorry

end infinite_powers_of_two_l3956_395664


namespace circumcircle_area_l3956_395651

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  -- Right-angled at A
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0 ∧
  -- AB = 6
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = 36 ∧
  -- AC = 8
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 64

-- Define the circumcircle of the triangle
def Circumcircle (A B C : ℝ × ℝ) : ℝ → Prop :=
  λ r => ∃ (center : ℝ × ℝ),
    (center.1 - A.1)^2 + (center.2 - A.2)^2 = r^2 ∧
    (center.1 - B.1)^2 + (center.2 - B.2)^2 = r^2 ∧
    (center.1 - C.1)^2 + (center.2 - C.2)^2 = r^2

-- Theorem statement
theorem circumcircle_area (A B C : ℝ × ℝ) :
  Triangle A B C →
  ∃ r, Circumcircle A B C r ∧ π * r^2 = 25 * π :=
by sorry

end circumcircle_area_l3956_395651


namespace problem_solution_l3956_395658

-- Define the complex square root function
noncomputable def complexSqrt (x : ℂ) : ℂ := sorry

-- Define the statements
def statement_I : Prop :=
  complexSqrt (-4) * complexSqrt (-16) = complexSqrt ((-4) * (-16))

def statement_II : Prop :=
  complexSqrt ((-4) * (-16)) = Real.sqrt 64

def statement_III : Prop :=
  Real.sqrt 64 = 8

-- Theorem to prove
theorem problem_solution :
  (¬statement_I ∧ statement_II ∧ statement_III) := by sorry

end problem_solution_l3956_395658


namespace total_weight_problem_l3956_395614

/-- The total weight problem -/
theorem total_weight_problem (a b c d : ℕ) 
  (h1 : a + b = 250)
  (h2 : b + c = 235)
  (h3 : c + d = 260)
  (h4 : a + d = 275) :
  a + b + c + d = 510 := by
  sorry

end total_weight_problem_l3956_395614


namespace power_sum_of_i_l3956_395650

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem power_sum_of_i : i^23 + i^223 = -2*i := by sorry

end power_sum_of_i_l3956_395650


namespace max_product_constraint_l3956_395644

theorem max_product_constraint (a b : ℝ) : 
  a > 0 → b > 0 → a + b = 2 → (∀ x y : ℝ, x > 0 → y > 0 → x + y = 2 → a * b ≥ x * y) → a * b = 1 :=
by sorry

end max_product_constraint_l3956_395644


namespace ordering_of_a_ab_ab_squared_l3956_395630

theorem ordering_of_a_ab_ab_squared (a b : ℝ) (ha : a < 0) (hb : b < -1) :
  a * b > a ∧ a > a * b^2 := by
  sorry

end ordering_of_a_ab_ab_squared_l3956_395630


namespace income_mean_difference_l3956_395667

/-- The number of families in the dataset -/
def num_families : ℕ := 500

/-- The correct maximum income -/
def correct_max_income : ℕ := 120000

/-- The incorrect maximum income -/
def incorrect_max_income : ℕ := 1200000

/-- The sum of all incomes excluding the maximum -/
def T : ℕ := sorry

theorem income_mean_difference :
  (T + incorrect_max_income) / num_families - (T + correct_max_income) / num_families = 2160 :=
sorry

end income_mean_difference_l3956_395667


namespace intersection_nonempty_implies_a_less_than_three_l3956_395624

def A : Set ℝ := {x | 3 + 2*x - x^2 ≥ 0}
def B (a : ℝ) : Set ℝ := {x | x > a}

theorem intersection_nonempty_implies_a_less_than_three (a : ℝ) :
  (A ∩ B a).Nonempty → a < 3 := by
  sorry

end intersection_nonempty_implies_a_less_than_three_l3956_395624


namespace k_range_l3956_395661

-- Define the propositions p and q
def p (x k : ℝ) : Prop := x ≥ k
def q (x : ℝ) : Prop := 3 / (x + 1) < 1

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (k : ℝ) : Prop :=
  (∀ x, p x k → q x) ∧ ¬(∀ x, q x → p x k)

-- Theorem statement
theorem k_range :
  ∀ k : ℝ, sufficient_not_necessary k ↔ k > 2 :=
sorry

end k_range_l3956_395661


namespace solution_set_f_less_than_one_range_of_a_l3956_395672

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 1|

-- Theorem for the solution set of f(x) < 1
theorem solution_set_f_less_than_one :
  {x : ℝ | f x < 1} = {x : ℝ | -3 < x ∧ x < 1/3} := by sorry

-- Theorem for the range of a
theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, f x ≤ a - a^2/2 + 5/2) → -2 ≤ a ∧ a ≤ 4 := by sorry

end solution_set_f_less_than_one_range_of_a_l3956_395672


namespace trip_length_calculation_l3956_395609

theorem trip_length_calculation (total : ℚ) 
  (h1 : total / 4 + 16 + total / 6 = total) : total = 192 / 7 := by
  sorry

end trip_length_calculation_l3956_395609


namespace masha_meeting_time_l3956_395606

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Represents the scenario of Masha's journey home -/
structure MashaJourney where
  usual_end_time : Time
  usual_arrival_time : Time
  early_end_time : Time
  early_arrival_time : Time
  meeting_time : Time

/-- Calculate the time difference in minutes between two Time values -/
def time_diff_minutes (t1 t2 : Time) : ℤ :=
  (t1.hours - t2.hours) * 60 + (t1.minutes - t2.minutes)

/-- The main theorem to prove -/
theorem masha_meeting_time (journey : MashaJourney) : 
  journey.usual_end_time = ⟨13, 0, by norm_num⟩ →
  journey.early_end_time = ⟨12, 0, by norm_num⟩ →
  time_diff_minutes journey.usual_arrival_time journey.early_arrival_time = 12 →
  journey.meeting_time = ⟨12, 54, by norm_num⟩ := by
  sorry

end masha_meeting_time_l3956_395606


namespace intersection_of_A_and_B_l3956_395636

def A : Set ℤ := {0, 1, 2, 3, 4, 5}
def B : Set ℤ := {-1, 0, 1, 6}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by
  sorry

end intersection_of_A_and_B_l3956_395636


namespace intersection_of_A_and_B_l3956_395666

def A : Set ℤ := {-1, 1, 2}
def B : Set ℤ := {x | x^2 = x}

theorem intersection_of_A_and_B : A ∩ B = {1} := by sorry

end intersection_of_A_and_B_l3956_395666


namespace cyclist_trip_distance_l3956_395691

/-- Represents a cyclist's trip with consistent speed -/
structure CyclistTrip where
  speed : ℝ
  time : ℝ
  distance : ℝ

/-- The trip satisfies the given conditions -/
def satisfiesConditions (trip : CyclistTrip) : Prop :=
  trip.distance = trip.speed * trip.time ∧
  trip.distance = (trip.speed + 1) * (2/3 * trip.time) ∧
  trip.distance = (trip.speed - 1) * (trip.time + 1)

/-- The theorem stating that the distance is 2 miles -/
theorem cyclist_trip_distance : 
  ∀ (trip : CyclistTrip), satisfiesConditions trip → trip.distance = 2 :=
by
  sorry

end cyclist_trip_distance_l3956_395691


namespace coins_taken_out_l3956_395612

/-- The number of coins Tina put in during the first hour -/
def first_hour_coins : ℕ := 20

/-- The number of coins Tina put in during each of the second and third hours -/
def second_third_hour_coins : ℕ := 30

/-- The number of coins Tina put in during the fourth hour -/
def fourth_hour_coins : ℕ := 40

/-- The number of coins left in the jar after the fifth hour -/
def coins_left : ℕ := 100

/-- The total number of coins Tina put in the jar -/
def total_coins_in : ℕ := first_hour_coins + 2 * second_third_hour_coins + fourth_hour_coins

/-- Theorem: The number of coins Tina's mother took out is equal to the total number of coins Tina put in minus the number of coins left in the jar after the fifth hour -/
theorem coins_taken_out : total_coins_in - coins_left = 20 := by
  sorry

end coins_taken_out_l3956_395612


namespace public_foundation_share_l3956_395605

/-- Represents the distribution of charitable funds by a private company. -/
structure CharityFunds where
  X : ℝ  -- Total amount raised
  Y : ℝ  -- Percentage donated to public foundation
  Z : ℕ  -- Number of organizations in public foundation
  W : ℕ  -- Number of local non-profit groups
  A : ℝ  -- Amount received by each local non-profit group
  B : ℝ  -- Amount received by special project
  h1 : Y > 0 ∧ Y ≤ 100  -- Ensure Y is a valid percentage
  h2 : Z > 0  -- Ensure there's at least one organization in the public foundation
  h3 : W > 0  -- Ensure there's at least one local non-profit group
  h4 : X > 0  -- Ensure a positive amount is raised
  h5 : B = (1/3) * X * (1 - Y/100)  -- Amount received by special project
  h6 : A = (2/3) * X * (1 - Y/100) / W  -- Amount received by each local non-profit group

/-- Theorem stating the amount received by each organization in the public foundation. -/
theorem public_foundation_share (cf : CharityFunds) :
  (cf.Y / 100) * cf.X / cf.Z = (cf.Y / 100) * cf.X / cf.Z :=
by sorry

end public_foundation_share_l3956_395605


namespace sqrt_6_simplest_l3956_395652

def is_simplest_sqrt (x : ℝ) : Prop :=
  ∀ y : ℝ, y > 0 → x = Real.sqrt y → ¬∃ a b : ℝ, a > 0 ∧ b > 1 ∧ y = a * b^2

theorem sqrt_6_simplest :
  is_simplest_sqrt (Real.sqrt 6) ∧
  ¬is_simplest_sqrt (Real.sqrt 8) ∧
  ¬is_simplest_sqrt (Real.sqrt (1/3)) ∧
  ¬is_simplest_sqrt (Real.sqrt 4) :=
sorry

end sqrt_6_simplest_l3956_395652


namespace number_in_scientific_notation_l3956_395684

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The number to be expressed in scientific notation -/
def number : ℕ := 21600

/-- Theorem stating that 21,600 in scientific notation is 2.16 × 10^4 -/
theorem number_in_scientific_notation :
  ∃ (sn : ScientificNotation), (sn.coefficient * (10 : ℝ) ^ sn.exponent = number) ∧
    (sn.coefficient = 2.16 ∧ sn.exponent = 4) :=
sorry

end number_in_scientific_notation_l3956_395684


namespace polygon_angles_l3956_395646

theorem polygon_angles (n : ℕ) (h : n > 2) :
  (180 * (n - 2) : ℝ) = 3 * 360 →
  n = 8 ∧ (180 * (n - 2) : ℝ) = 1080 := by
  sorry

end polygon_angles_l3956_395646


namespace trapezoid_area_is_correct_l3956_395648

/-- The area of a trapezoid bounded by y = 2x, y = 6, y = 3, and the y-axis -/
def trapezoidArea : ℝ := 6.75

/-- The line y = 2x -/
def line1 (x : ℝ) : ℝ := 2 * x

/-- The line y = 6 -/
def line2 : ℝ := 6

/-- The line y = 3 -/
def line3 : ℝ := 3

/-- The y-axis (x = 0) -/
def yAxis : ℝ := 0

theorem trapezoid_area_is_correct :
  trapezoidArea = 6.75 := by sorry

end trapezoid_area_is_correct_l3956_395648


namespace overall_loss_percentage_l3956_395655

def purchase_prices : List ℝ := [600, 800, 1000, 1200, 1400]
def selling_prices : List ℝ := [550, 750, 1100, 1000, 1350]

theorem overall_loss_percentage :
  let total_cost_price := purchase_prices.sum
  let total_selling_price := selling_prices.sum
  let loss := total_cost_price - total_selling_price
  let loss_percentage := (loss / total_cost_price) * 100
  loss_percentage = 5 := by sorry

end overall_loss_percentage_l3956_395655


namespace group_size_calculation_l3956_395649

theorem group_size_calculation (n : ℕ) : 
  (n * 15 + 37 = 17 * (n + 1)) → n = 10 := by
  sorry

end group_size_calculation_l3956_395649


namespace jelly_bean_probability_l3956_395626

theorem jelly_bean_probability (p_red p_orange : ℝ) 
  (h_red : p_red = 0.25)
  (h_orange : p_orange = 0.35)
  (h_sum : p_red + p_orange + (p_yellow + p_green) = 1) :
  p_yellow + p_green = 0.40 := by
  sorry

end jelly_bean_probability_l3956_395626


namespace centrally_symmetric_multiple_symmetry_axes_l3956_395601

/-- A polygon in a 2D plane. -/
structure Polygon where
  -- Add necessary fields for a polygon

/-- Represents a line in a 2D plane. -/
structure Line where
  -- Add necessary fields for a line

/-- Predicate to check if a polygon is centrally symmetric. -/
def is_centrally_symmetric (p : Polygon) : Prop :=
  sorry

/-- Predicate to check if a line is a symmetry axis of a polygon. -/
def is_symmetry_axis (l : Line) (p : Polygon) : Prop :=
  sorry

/-- The number of symmetry axes a polygon has. -/
def num_symmetry_axes (p : Polygon) : Nat :=
  sorry

/-- Theorem: A centrally symmetric polygon with at least one symmetry axis must have more than one symmetry axis. -/
theorem centrally_symmetric_multiple_symmetry_axes (p : Polygon) :
  is_centrally_symmetric p → (∃ l : Line, is_symmetry_axis l p) → num_symmetry_axes p > 1 :=
by sorry

end centrally_symmetric_multiple_symmetry_axes_l3956_395601


namespace average_of_remaining_digits_l3956_395685

theorem average_of_remaining_digits
  (total_count : Nat)
  (total_avg : ℚ)
  (subset_count : Nat)
  (subset_avg : ℚ)
  (h_total_count : total_count = 10)
  (h_total_avg : total_avg = 80)
  (h_subset_count : subset_count = 6)
  (h_subset_avg : subset_avg = 58)
  : (total_count * total_avg - subset_count * subset_avg) / (total_count - subset_count) = 113 := by
  sorry

end average_of_remaining_digits_l3956_395685


namespace reflection_line_equation_l3956_395663

-- Define the points of the original triangle
def P : ℝ × ℝ := (3, 2)
def Q : ℝ × ℝ := (8, 7)
def R : ℝ × ℝ := (6, -4)

-- Define the points of the reflected triangle
def P' : ℝ × ℝ := (-5, 2)
def Q' : ℝ × ℝ := (-10, 7)
def R' : ℝ × ℝ := (-8, -4)

-- Define the reflection line
def M : ℝ → Prop := λ x => x = -1

theorem reflection_line_equation :
  (∀ (x y : ℝ), (x, y) = P ∨ (x, y) = Q ∨ (x, y) = R →
    ∃ (x' : ℝ), M x' ∧ x' = (x + P'.1) / 2) ∧
  (∀ (x y : ℝ), (x, y) = P' ∨ (x, y) = Q' ∨ (x, y) = R' →
    ∃ (x' : ℝ), M x' ∧ x' = (x + P.1) / 2) :=
sorry

end reflection_line_equation_l3956_395663


namespace max_score_theorem_l3956_395683

/-- Represents a pile of stones -/
structure Pile :=
  (stones : ℕ)

/-- Represents the game state -/
structure GameState :=
  (piles : List Pile)
  (score : ℕ)

/-- Defines a move in the game -/
def move (state : GameState) (i j : ℕ) : GameState :=
  sorry

/-- Checks if the game is over (all stones removed) -/
def isGameOver (state : GameState) : Bool :=
  sorry

/-- Calculates the maximum score achievable from a given state -/
def maxScore (state : GameState) : ℕ :=
  sorry

/-- The main theorem stating the maximum achievable score -/
theorem max_score_theorem :
  let initialState : GameState := ⟨List.replicate 100 ⟨400⟩, 0⟩
  maxScore initialState = 3920000 := by
  sorry

end max_score_theorem_l3956_395683


namespace algebraic_simplification_l3956_395699

theorem algebraic_simplification (a b : ℝ) : 
  (a^3 * b^4)^2 / (a * b^2)^3 = a^3 * b^2 ∧ 
  (-a^2)^3 * a^2 + a^8 = 0 := by
  sorry

end algebraic_simplification_l3956_395699
