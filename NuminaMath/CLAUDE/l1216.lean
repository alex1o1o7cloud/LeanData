import Mathlib

namespace NUMINAMATH_CALUDE_infinite_solutions_equation_l1216_121603

theorem infinite_solutions_equation :
  ∀ n : ℕ+, ∃ a b c : ℕ+,
    (a : ℝ) ^ 2 + (b : ℝ) ^ 5 = (c : ℝ) ^ 3 :=
by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_equation_l1216_121603


namespace NUMINAMATH_CALUDE_cosine_amplitude_and_shift_l1216_121655

/-- Given a cosine function that oscillates between 5 and 1, prove its amplitude and vertical shift. -/
theorem cosine_amplitude_and_shift (a b c d : ℝ) : 
  (∀ x, 1 ≤ a * Real.cos (b * x + c) + d ∧ a * Real.cos (b * x + c) + d ≤ 5) →
  a = 2 ∧ d = 3 := by
  sorry

end NUMINAMATH_CALUDE_cosine_amplitude_and_shift_l1216_121655


namespace NUMINAMATH_CALUDE_officer_selection_count_l1216_121645

def club_members : ℕ := 12
def officer_positions : ℕ := 5

theorem officer_selection_count :
  (club_members.factorial) / ((club_members - officer_positions).factorial) = 95040 := by
  sorry

end NUMINAMATH_CALUDE_officer_selection_count_l1216_121645


namespace NUMINAMATH_CALUDE_arithmetic_sequences_difference_l1216_121610

def arithmetic_sum (a₁ : ℕ) (aₙ : ℕ) : ℕ :=
  let n := aₙ - a₁ + 1
  n * (a₁ + aₙ) / 2

theorem arithmetic_sequences_difference : 
  arithmetic_sum 2001 2093 - arithmetic_sum 201 293 - arithmetic_sum 1 93 = 165044 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequences_difference_l1216_121610


namespace NUMINAMATH_CALUDE_intersection_A_B_l1216_121658

-- Define set A
def A : Set ℝ := {x | 1 < x - 1 ∧ x - 1 ≤ 3}

-- Define set B
def B : Set ℝ := {2, 3, 4}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {3, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1216_121658


namespace NUMINAMATH_CALUDE_stratified_sample_properties_l1216_121632

/-- Represents the grades of parts in a batch -/
inductive Grade
  | First
  | Second
  | Third

/-- Structure representing a batch of parts -/
structure Batch :=
  (first : ℕ)
  (second : ℕ)
  (third : ℕ)

/-- Structure representing a sample drawn from a batch -/
structure Sample :=
  (first : ℕ)
  (second : ℕ)
  (third : ℕ)

/-- Function to check if a sample is valid for a given batch -/
def isValidSample (b : Batch) (s : Sample) : Prop :=
  s.first + s.second + s.third = 20 ∧
  s.first ≤ b.first ∧
  s.second ≤ b.second ∧
  s.third ≤ b.third

/-- Theorem stating the properties of the stratified sample -/
theorem stratified_sample_properties (b : Batch) (s : Sample) :
  b.first = 24 →
  b.second = 36 →
  s.third = 10 →
  isValidSample b s →
  b.third = 60 ∧ s.second = 6 := by sorry

end NUMINAMATH_CALUDE_stratified_sample_properties_l1216_121632


namespace NUMINAMATH_CALUDE_no_rain_probability_l1216_121657

theorem no_rain_probability (p : ℚ) (h : p = 2/3) :
  (1 - p)^4 = 1/81 := by sorry

end NUMINAMATH_CALUDE_no_rain_probability_l1216_121657


namespace NUMINAMATH_CALUDE_prime_factor_count_l1216_121690

theorem prime_factor_count (p : ℕ) : 
  (26 : ℕ) + p + (2 : ℕ) = (33 : ℕ) → p = (5 : ℕ) := by
  sorry

#check prime_factor_count

end NUMINAMATH_CALUDE_prime_factor_count_l1216_121690


namespace NUMINAMATH_CALUDE_turtle_difference_is_nine_l1216_121630

/-- Given the number of turtles Kristen has, calculate the difference between Trey's and Kristen's turtle counts. -/
def turtle_difference (kristen_turtles : ℕ) : ℕ :=
  let kris_turtles := kristen_turtles / 4
  let trey_turtles := 7 * kris_turtles
  trey_turtles - kristen_turtles

/-- Theorem stating that the difference between Trey's and Kristen's turtle counts is 9 when Kristen has 12 turtles. -/
theorem turtle_difference_is_nine :
  turtle_difference 12 = 9 := by
  sorry

end NUMINAMATH_CALUDE_turtle_difference_is_nine_l1216_121630


namespace NUMINAMATH_CALUDE_chess_competition_participants_l1216_121649

def is_valid_num_high_school_students (n : ℕ) : Prop :=
  let total_players := n + 2
  let total_games := total_players * (total_players - 1) / 2
  let remaining_points := total_games - 8
  remaining_points % n = 0

theorem chess_competition_participants : 
  ∀ n : ℕ, n > 2 → (is_valid_num_high_school_students n ↔ n = 7 ∨ n = 14) :=
by sorry

end NUMINAMATH_CALUDE_chess_competition_participants_l1216_121649


namespace NUMINAMATH_CALUDE_book_sale_profit_l1216_121650

/-- Calculates the difference between total selling price (including tax) and total purchase price (after discount) for books --/
theorem book_sale_profit (num_books : ℕ) (original_price discount_rate desired_price tax_rate : ℚ) : 
  num_books = 15 → 
  original_price = 11 → 
  discount_rate = 1/5 → 
  desired_price = 25 → 
  tax_rate = 1/10 → 
  (num_books * (desired_price * (1 + tax_rate))) - (num_books * (original_price * (1 - discount_rate))) = 280.5 := by
sorry

end NUMINAMATH_CALUDE_book_sale_profit_l1216_121650


namespace NUMINAMATH_CALUDE_zoo_animal_count_l1216_121612

/-- Given a zoo with penguins and polar bears, calculate the total number of animals -/
theorem zoo_animal_count (num_penguins : ℕ) (h1 : num_penguins = 21) 
  (h2 : ∃ (num_polar_bears : ℕ), num_polar_bears = 2 * num_penguins) : 
  ∃ (total_animals : ℕ), total_animals = num_penguins + 2 * num_penguins :=
by sorry

end NUMINAMATH_CALUDE_zoo_animal_count_l1216_121612


namespace NUMINAMATH_CALUDE_corresponding_sides_of_congruent_triangles_are_equal_l1216_121600

-- Define a triangle as a structure with three points
structure Triangle (α : Type*) :=
  (A B C : α)

-- Define congruence for triangles
def CongruentTriangles {α : Type*} (T1 T2 : Triangle α) : Prop :=
  sorry

-- Define the concept of corresponding sides
def CorrespondingSides {α : Type*} (T1 T2 : Triangle α) (s1 s2 : α × α) : Prop :=
  sorry

-- Define equality of sides
def EqualSides {α : Type*} (s1 s2 : α × α) : Prop :=
  sorry

-- Theorem: Corresponding sides of congruent triangles are equal
theorem corresponding_sides_of_congruent_triangles_are_equal
  {α : Type*} (T1 T2 : Triangle α) :
  CongruentTriangles T1 T2 →
  ∀ s1 s2, CorrespondingSides T1 T2 s1 s2 → EqualSides s1 s2 :=
by
  sorry

end NUMINAMATH_CALUDE_corresponding_sides_of_congruent_triangles_are_equal_l1216_121600


namespace NUMINAMATH_CALUDE_class_composition_l1216_121687

theorem class_composition (total_students : ℕ) (boy_ratio girl_ratio : ℕ) 
  (h1 : total_students = 56)
  (h2 : boy_ratio = 3)
  (h3 : girl_ratio = 4) :
  (boy_ratio : ℚ) / (boy_ratio + girl_ratio) * 100 = 42.86 ∧ 
  (girl_ratio * total_students) / (boy_ratio + girl_ratio) = 32 := by
sorry

end NUMINAMATH_CALUDE_class_composition_l1216_121687


namespace NUMINAMATH_CALUDE_min_cells_in_square_sheet_exists_min_square_sheet_l1216_121611

/-- Represents a rectangular shape with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a square shape with side length -/
structure Square where
  side : ℕ

/-- The ship cut out from the paper -/
def ship : Rectangle :=
  { width := 10, height := 11 }

/-- Theorem: The minimum number of cells in the square sheet of paper is 121 -/
theorem min_cells_in_square_sheet : 
  ∀ (s : Square), s.side ≥ max ship.width ship.height → s.side * s.side ≥ 121 :=
by
  sorry

/-- Corollary: There exists a square sheet with exactly 121 cells that can fit the ship -/
theorem exists_min_square_sheet :
  ∃ (s : Square), s.side * s.side = 121 ∧ s.side ≥ max ship.width ship.height :=
by
  sorry

end NUMINAMATH_CALUDE_min_cells_in_square_sheet_exists_min_square_sheet_l1216_121611


namespace NUMINAMATH_CALUDE_min_value_theorem_l1216_121660

theorem min_value_theorem (a : ℝ) (x₁ x₂ : ℝ) 
  (h_a : a > 0)
  (h_x : ∀ x, x^2 - 4*a*x + 3*a^2 < 0 ↔ x ∈ (Set.Ioo x₁ x₂)) :
  ∃ (min_val : ℝ), 
    (∀ y, x₁ + x₂ + a / (x₁ * x₂) ≥ y) ∧ 
    (x₁ + x₂ + a / (x₁ * x₂) = y ↔ y = 4 * Real.sqrt 3 / 3) :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1216_121660


namespace NUMINAMATH_CALUDE_allocation_methods_l1216_121606

/-- Represents the number of students --/
def num_students : ℕ := 5

/-- Represents the number of villages --/
def num_villages : ℕ := 3

/-- Represents the number of entities to be allocated (treating A and B as one entity) --/
def num_entities : ℕ := 4

/-- The number of ways to divide num_entities into num_villages non-empty groups --/
def ways_to_divide : ℕ := Nat.choose num_entities (num_villages - 1)

/-- The number of ways to arrange num_villages groups into num_villages villages --/
def ways_to_arrange : ℕ := Nat.factorial num_villages

/-- Theorem stating the total number of allocation methods --/
theorem allocation_methods :
  ways_to_divide * ways_to_arrange = 36 := by sorry

end NUMINAMATH_CALUDE_allocation_methods_l1216_121606


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l1216_121697

def balanced_about_2 (a b : ℝ) : Prop := a + b = 2

theorem problem_1 : balanced_about_2 3 (-1) := by sorry

theorem problem_2 (x : ℝ) : balanced_about_2 (x - 3) (5 - x) := by sorry

def a (x : ℝ) : ℝ := 2 * x^2 - 3 * (x^2 + x) + 4
def b (x : ℝ) : ℝ := 2 * x - (3 * x - (4 * x + x^2) - 2)

theorem problem_3 : ∀ x : ℝ, a x + b x ≠ 2 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l1216_121697


namespace NUMINAMATH_CALUDE_circle_area_is_6pi_l1216_121643

/-- The equation of the circle C -/
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 - 2*a*x - 2*y + 2 = 0

/-- The equation of the line -/
def line_equation (x y a : ℝ) : Prop :=
  y = a*x

/-- Definition of an equilateral triangle -/
def is_equilateral_triangle (A B C : ℝ × ℝ) : Prop :=
  let d (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d A B = d B C ∧ d B C = d C A

/-- The main theorem -/
theorem circle_area_is_6pi (a : ℝ) (A B : ℝ × ℝ) (C : ℝ × ℝ) :
  (∃ x y, circle_equation x y a ∧ line_equation x y a) →
  is_equilateral_triangle A B C →
  (∃ r, π * r^2 = 6 * π) :=
sorry

end NUMINAMATH_CALUDE_circle_area_is_6pi_l1216_121643


namespace NUMINAMATH_CALUDE_curve_is_semicircle_l1216_121613

-- Define the curve
def curve (x y : ℝ) : Prop := x - 1 = Real.sqrt (1 - (y - 1)^2)

-- Define a semicircle
def semicircle (center : ℝ × ℝ) (radius : ℝ) (x y : ℝ) : Prop :=
  (x - center.1)^2 + (y - center.2)^2 = radius^2 ∧ x ≥ center.1

theorem curve_is_semicircle :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    ∀ (x y : ℝ), curve x y ↔ semicircle center radius x y :=
sorry

end NUMINAMATH_CALUDE_curve_is_semicircle_l1216_121613


namespace NUMINAMATH_CALUDE_correct_factorization_l1216_121629

theorem correct_factorization (a b : ℝ) : a^2 - 4*a*b + 4*b^2 = (a - 2*b)^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_factorization_l1216_121629


namespace NUMINAMATH_CALUDE_find_a_value_l1216_121695

theorem find_a_value (x a : ℝ) (h : x = -1 ∧ -2 * (x - a) = 4) : a = 1 := by
  sorry

end NUMINAMATH_CALUDE_find_a_value_l1216_121695


namespace NUMINAMATH_CALUDE_tangent_line_y_intercept_l1216_121686

/-- The y-coordinate of the intersection point between the tangent line
    to the curve y = x^3 + 11 at point P(1, 12) and the y-axis is 9. -/
theorem tangent_line_y_intercept : 
  let f (x : ℝ) := x^3 + 11
  let P : ℝ × ℝ := (1, 12)
  let m : ℝ := deriv f 1
  let tangent_line (x : ℝ) := m * (x - P.1) + P.2
  tangent_line 0 = 9 := by sorry

end NUMINAMATH_CALUDE_tangent_line_y_intercept_l1216_121686


namespace NUMINAMATH_CALUDE_simplify_expression_l1216_121646

theorem simplify_expression (x : ℝ) : 3 * x - 5 * x^2 + 7 + (2 - 3 * x + 5 * x^2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1216_121646


namespace NUMINAMATH_CALUDE_range_of_a_l1216_121608

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 1 then x * Real.log x - a * x^2 else a^x

theorem range_of_a (a : ℝ) :
  (∀ x y, x < y → f a x > f a y) →
  a ∈ Set.Icc (1/2 : ℝ) 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1216_121608


namespace NUMINAMATH_CALUDE_factorial_division_l1216_121633

theorem factorial_division : 8 / 3 = 6720 :=
by
  -- Define 8! as given in the problem
  have h1 : 8 = 40320 := by sorry
  
  -- Define 3! (not given in the problem, but necessary for the proof)
  have h2 : 3 = 6 := by sorry
  
  -- Prove that 8! ÷ 3! = 6720
  sorry

end NUMINAMATH_CALUDE_factorial_division_l1216_121633


namespace NUMINAMATH_CALUDE_pyramid_height_theorem_l1216_121642

/-- Represents a right octagonal pyramid -/
structure RightOctagonalPyramid where
  /-- The distance from the apex to the larger section -/
  height : ℝ
  /-- The area of the larger section -/
  larger_section_area : ℝ
  /-- The area of the smaller section -/
  smaller_section_area : ℝ
  /-- The distance between the two sections -/
  section_distance : ℝ

/-- Theorem stating the relationship between the sections and the height of the pyramid -/
theorem pyramid_height_theorem (pyramid : RightOctagonalPyramid) 
    (h1 : pyramid.larger_section_area = 810)
    (h2 : pyramid.smaller_section_area = 360)
    (h3 : pyramid.section_distance = 10) : 
    pyramid.height = 30 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_height_theorem_l1216_121642


namespace NUMINAMATH_CALUDE_inequality_proof_l1216_121623

theorem inequality_proof (a b c : ℝ) (h : a * b < 0) :
  a^2 + b^2 + c^2 > 2*a*b + 2*b*c + 2*c*a := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1216_121623


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l1216_121609

/-- Proves that the interest rate A lends to B is 10% given the specified conditions -/
theorem interest_rate_calculation (principal : ℝ) (rate_c : ℝ) (time : ℝ) (b_gain : ℝ) 
  (h1 : principal = 3500)
  (h2 : rate_c = 0.115)
  (h3 : time = 3)
  (h4 : b_gain = 157.5)
  (h5 : b_gain = principal * rate_c * time - principal * rate_a * time) : 
  rate_a = 0.1 := by
  sorry

#check interest_rate_calculation

end NUMINAMATH_CALUDE_interest_rate_calculation_l1216_121609


namespace NUMINAMATH_CALUDE_virginia_eggs_remaining_l1216_121624

theorem virginia_eggs_remaining (initial_eggs : ℕ) (eggs_taken : ℕ) : 
  initial_eggs = 200 → eggs_taken = 37 → initial_eggs - eggs_taken = 163 :=
by
  sorry

end NUMINAMATH_CALUDE_virginia_eggs_remaining_l1216_121624


namespace NUMINAMATH_CALUDE_distance_between_vertices_l1216_121622

/-- The distance between the vertices of two quadratic functions -/
theorem distance_between_vertices (a b c d e f : ℝ) :
  let f1 := fun x : ℝ => x^2 + a*x + b
  let f2 := fun x : ℝ => x^2 + d*x + e
  let vertex1 := (-a/2, f1 (-a/2))
  let vertex2 := (-d/2, f2 (-d/2))
  a = -4 ∧ b = 7 ∧ d = 6 ∧ e = 20 →
  Real.sqrt ((vertex1.1 - vertex2.1)^2 + (vertex1.2 - vertex2.2)^2) = Real.sqrt 89 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_vertices_l1216_121622


namespace NUMINAMATH_CALUDE_perfect_square_condition_l1216_121621

theorem perfect_square_condition (y : ℕ) :
  (∃ x : ℕ, y^2 + 3^y = x^2) ↔ y = 1 ∨ y = 3 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l1216_121621


namespace NUMINAMATH_CALUDE_player_a_winning_strategy_l1216_121671

/-- Represents a point on the chessboard -/
structure Point where
  x : Int
  y : Int

/-- Defines the chessboard -/
def is_on_board (p : Point) : Prop :=
  abs p.x ≤ 2019 ∧ abs p.y ≤ 2019 ∧ abs p.x + abs p.y < 4038

/-- Defines a boundary point -/
def is_boundary_point (p : Point) : Prop :=
  abs p.x = 2019 ∨ abs p.y = 2019

/-- Defines adjacent points -/
def are_adjacent (p1 p2 : Point) : Prop :=
  abs (p1.x - p2.x) + abs (p1.y - p2.y) = 1

/-- Represents the state of the game -/
structure GameState where
  piece_position : Point
  removed_points : Set Point

/-- Player A's move -/
def player_a_move (state : GameState) : GameState :=
  sorry

/-- Player B's move -/
def player_b_move (state : GameState) : GameState :=
  sorry

/-- Theorem stating that Player A has a winning strategy -/
theorem player_a_winning_strategy :
  ∃ (strategy : GameState → GameState),
    ∀ (initial_state : GameState),
      initial_state.piece_position = ⟨0, 0⟩ →
      ∀ (n : ℕ),
        let final_state := (strategy ∘ player_b_move)^[n] initial_state
        ∀ (p : Point), is_boundary_point p → p ∈ final_state.removed_points :=
  sorry

end NUMINAMATH_CALUDE_player_a_winning_strategy_l1216_121671


namespace NUMINAMATH_CALUDE_tournament_probability_l1216_121664

/-- The number of teams in the tournament -/
def num_teams : ℕ := 30

/-- The total number of games played in the tournament -/
def total_games : ℕ := num_teams.choose 2

/-- The probability of a team winning any given game -/
def win_probability : ℚ := 1/2

/-- The probability that no two teams win the same number of games -/
noncomputable def unique_wins_probability : ℚ := (num_teams.factorial : ℚ) / 2^total_games

theorem tournament_probability :
  ∃ (m : ℕ), Odd m ∧ unique_wins_probability = 1 / (2^409 * m) :=
sorry

end NUMINAMATH_CALUDE_tournament_probability_l1216_121664


namespace NUMINAMATH_CALUDE_consecutive_integers_square_difference_l1216_121675

theorem consecutive_integers_square_difference (n : ℕ) : 
  n > 0 ∧ n + (n + 1) < 100 → (n + 1)^2 - n^2 = 79 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_square_difference_l1216_121675


namespace NUMINAMATH_CALUDE_ellipse_proof_l1216_121672

-- Define the given ellipse
def given_ellipse (x y : ℝ) : Prop := 3 * x^2 + 8 * y^2 = 24

-- Define the point that the new ellipse should pass through
def point : ℝ × ℝ := (3, -2)

-- Define the new ellipse
def new_ellipse (x y : ℝ) : Prop := x^2 / 15 + y^2 / 10 = 1

-- Theorem statement
theorem ellipse_proof :
  (∃ (c : ℝ), c > 0 ∧
    (∀ (x y : ℝ), given_ellipse x y ↔ x^2 / (c^2 + 5) + y^2 / c^2 = 1)) →
  (new_ellipse point.1 point.2) ∧
  (∃ (c : ℝ), c > 0 ∧
    (∀ (x y : ℝ), new_ellipse x y ↔ x^2 / (c^2 + 5) + y^2 / c^2 = 1)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_proof_l1216_121672


namespace NUMINAMATH_CALUDE_molecular_weight_of_one_mole_l1216_121698

/-- The molecular weight of aluminum sulfide for a given number of moles -/
def molecular_weight (moles : ℝ) : ℝ := sorry

/-- The number of moles used in the given condition -/
def given_moles : ℝ := 4

/-- The molecular weight for the given number of moles -/
def given_weight : ℝ := 600

/-- Theorem: The molecular weight of one mole of aluminum sulfide is 150 g/mol -/
theorem molecular_weight_of_one_mole : 
  molecular_weight 1 = 150 := by sorry

end NUMINAMATH_CALUDE_molecular_weight_of_one_mole_l1216_121698


namespace NUMINAMATH_CALUDE_rational_solutions_quadratic_l1216_121618

theorem rational_solutions_quadratic (k : ℕ+) : 
  (∃ x : ℚ, k * x^2 + 24 * x + k = 0) ↔ (k = 8 ∨ k = 12) := by
sorry

end NUMINAMATH_CALUDE_rational_solutions_quadratic_l1216_121618


namespace NUMINAMATH_CALUDE_chucks_team_score_l1216_121638

theorem chucks_team_score (yellow_team_score lead : ℕ) 
  (h1 : yellow_team_score = 55)
  (h2 : lead = 17) :
  yellow_team_score + lead = 72 := by
  sorry

end NUMINAMATH_CALUDE_chucks_team_score_l1216_121638


namespace NUMINAMATH_CALUDE_base_conversion_725_9_l1216_121628

def base_9_to_base_3 (n : Nat) : Nat :=
  -- Definition of conversion from base 9 to base 3
  sorry

theorem base_conversion_725_9 :
  base_9_to_base_3 725 = 210212 :=
sorry

end NUMINAMATH_CALUDE_base_conversion_725_9_l1216_121628


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1216_121651

theorem polynomial_factorization (a b : ℝ) : a^3*b - 9*a*b = a*b*(a+3)*(a-3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1216_121651


namespace NUMINAMATH_CALUDE_sin_alpha_value_l1216_121625

theorem sin_alpha_value (α β : ℝ) (h_acute : 0 < α ∧ α < π / 2)
  (h1 : 2 * Real.tan (π - α) - 3 * Real.cos (π / 2 + β) + 5 = 0)
  (h2 : Real.tan (π + α) + 6 * Real.sin (π + β) = 1) :
  Real.sin α = 3 * Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l1216_121625


namespace NUMINAMATH_CALUDE_complex_power_difference_l1216_121619

/-- Given that i^2 = -1, prove that (1+2i)^24 - (1-2i)^24 = 0 -/
theorem complex_power_difference (i : ℂ) (h : i^2 = -1) :
  (1 + 2*i)^24 - (1 - 2*i)^24 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_difference_l1216_121619


namespace NUMINAMATH_CALUDE_egyptian_fraction_proof_l1216_121688

theorem egyptian_fraction_proof :
  ∃! (b₂ b₃ b₅ b₆ b₇ b₈ : ℕ),
    (3 : ℚ) / 8 = b₂ / 2 + b₃ / 6 + b₅ / 120 + b₆ / 720 + b₇ / 5040 + b₈ / 40320 ∧
    b₂ < 2 ∧ b₃ < 3 ∧ b₅ < 5 ∧ b₆ < 6 ∧ b₇ < 7 ∧ b₈ < 8 ∧
    b₂ + b₃ + b₅ + b₆ + b₇ + b₈ = 12 :=
by sorry

end NUMINAMATH_CALUDE_egyptian_fraction_proof_l1216_121688


namespace NUMINAMATH_CALUDE_f_properties_l1216_121614

noncomputable def f (x : ℝ) := Real.exp (-x) * Real.sin x

theorem f_properties :
  let a := -Real.pi
  let b := Real.pi
  ∃ (max_val min_val : ℝ),
    (∀ x ∈ Set.Icc a b, f x ≤ max_val) ∧
    (∃ x ∈ Set.Icc a b, f x = max_val) ∧
    (∀ x ∈ Set.Icc a b, min_val ≤ f x) ∧
    (∃ x ∈ Set.Icc a b, f x = min_val) ∧
    (StrictMonoOn f (Set.Ioo (-3*Real.pi/4) (Real.pi/4))) ∧
    (StrictAntiOn f (Set.Ioc a (-3*Real.pi/4))) ∧
    (StrictAntiOn f (Set.Ico (Real.pi/4) b)) ∧
    max_val = (Real.sqrt 2 / 2) * Real.exp (-Real.pi/4) ∧
    min_val = -(Real.sqrt 2 / 2) * Real.exp (3*Real.pi/4) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1216_121614


namespace NUMINAMATH_CALUDE_factorization_x_squared_plus_2x_l1216_121631

theorem factorization_x_squared_plus_2x (x : ℝ) : x^2 + 2*x = x*(x+2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x_squared_plus_2x_l1216_121631


namespace NUMINAMATH_CALUDE_jake_not_dropping_coffee_l1216_121636

theorem jake_not_dropping_coffee (trip_probability : ℝ) (drop_given_trip_probability : ℝ) :
  trip_probability = 0.4 →
  drop_given_trip_probability = 0.25 →
  1 - trip_probability * drop_given_trip_probability = 0.9 := by
sorry

end NUMINAMATH_CALUDE_jake_not_dropping_coffee_l1216_121636


namespace NUMINAMATH_CALUDE_puppies_count_l1216_121692

/-- Calculates the number of puppies given the total food needed, mom's food consumption, and puppies' food consumption. -/
def number_of_puppies (total_food : ℚ) (mom_meal : ℚ) (mom_meals_per_day : ℕ) (puppy_meal : ℚ) (puppy_meals_per_day : ℕ) (days : ℕ) : ℕ :=
  let mom_food := mom_meal * mom_meals_per_day * days
  let puppy_food := total_food - mom_food
  let puppy_food_per_puppy := puppy_meal * puppy_meals_per_day * days
  (puppy_food / puppy_food_per_puppy).num.toNat

/-- Theorem stating that the number of puppies is 5 given the specified conditions. -/
theorem puppies_count : number_of_puppies 57 (3/2) 3 (1/2) 2 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_puppies_count_l1216_121692


namespace NUMINAMATH_CALUDE_sum_of_squares_l1216_121699

theorem sum_of_squares (a b c : ℝ) (h : a + 19 = b + 9 ∧ b + 9 = c + 8) :
  (a - b)^2 + (b - c)^2 + (c - a)^2 = 222 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1216_121699


namespace NUMINAMATH_CALUDE_soda_discount_percentage_l1216_121663

/-- The discount percentage for soda cans purchased in 24-can cases -/
def discount_percentage (regular_price : ℚ) (discounted_price : ℚ) : ℚ :=
  (1 - discounted_price / (100 * regular_price)) * 100

/-- Theorem stating that the discount percentage is 15% -/
theorem soda_discount_percentage :
  let regular_price : ℚ := 40 / 100  -- $0.40 per can
  let discounted_price : ℚ := 34     -- $34 for 100 cans
  discount_percentage regular_price discounted_price = 15 := by
  sorry

#eval discount_percentage (40/100) 34

end NUMINAMATH_CALUDE_soda_discount_percentage_l1216_121663


namespace NUMINAMATH_CALUDE_triangle_isosceles_condition_l1216_121693

theorem triangle_isosceles_condition (a b c : ℝ) (A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- Positive side lengths
  0 < A ∧ 0 < B ∧ 0 < C →  -- Positive angles
  A + B + C = π →          -- Sum of angles in a triangle
  a * Real.cos B + b * Real.cos C + c * Real.cos A = (a + b + c) / 2 →
  (a = b ∨ b = c ∨ c = a) :=
by sorry

end NUMINAMATH_CALUDE_triangle_isosceles_condition_l1216_121693


namespace NUMINAMATH_CALUDE_total_cost_four_games_l1216_121607

def batman_price : ℝ := 13.60
def superman_price : ℝ := 5.06
def batman_discount : ℝ := 0.10
def superman_discount : ℝ := 0.05
def sales_tax : ℝ := 0.08
def owned_game1 : ℝ := 7.25
def owned_game2 : ℝ := 12.50

theorem total_cost_four_games :
  let batman_discounted := batman_price * (1 - batman_discount)
  let superman_discounted := superman_price * (1 - superman_discount)
  let batman_with_tax := batman_discounted * (1 + sales_tax)
  let superman_with_tax := superman_discounted * (1 + sales_tax)
  let total_cost := batman_with_tax + superman_with_tax + owned_game1 + owned_game2
  total_cost = 38.16 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_four_games_l1216_121607


namespace NUMINAMATH_CALUDE_instrument_probability_l1216_121656

/-- The probability of selecting a cello and a viola made from the same tree -/
theorem instrument_probability (total_cellos : ℕ) (total_violas : ℕ) (same_tree_pairs : ℕ) :
  total_cellos = 800 →
  total_violas = 600 →
  same_tree_pairs = 100 →
  (same_tree_pairs : ℚ) / (total_cellos * total_violas) = 1 / 4800 := by
  sorry

end NUMINAMATH_CALUDE_instrument_probability_l1216_121656


namespace NUMINAMATH_CALUDE_factorial_square_root_squared_l1216_121661

theorem factorial_square_root_squared : (((4 * 3 * 2 * 1) * (3 * 2 * 1) : ℕ).sqrt ^ 2 : ℝ) = 144 := by
  sorry

end NUMINAMATH_CALUDE_factorial_square_root_squared_l1216_121661


namespace NUMINAMATH_CALUDE_final_cafeteria_count_l1216_121620

def total_students : ℕ := 300

def initial_cafeteria : ℕ := (2 * total_students) / 5
def initial_outside : ℕ := (3 * total_students) / 10
def initial_classroom : ℕ := total_students - initial_cafeteria - initial_outside

def outside_to_cafeteria : ℕ := (40 * initial_outside) / 100
def cafeteria_to_outside : ℕ := 5
def classroom_to_cafeteria : ℕ := (15 * initial_classroom + 50) / 100  -- Rounded up
def outside_to_classroom : ℕ := 2

theorem final_cafeteria_count :
  initial_cafeteria + outside_to_cafeteria - cafeteria_to_outside + classroom_to_cafeteria = 165 :=
sorry

end NUMINAMATH_CALUDE_final_cafeteria_count_l1216_121620


namespace NUMINAMATH_CALUDE_abs_diff_geq_sum_abs_iff_product_nonpositive_l1216_121683

theorem abs_diff_geq_sum_abs_iff_product_nonpositive (a b : ℝ) :
  a * b ≤ 0 ↔ |a - b| ≥ |a| + |b| := by
  sorry

end NUMINAMATH_CALUDE_abs_diff_geq_sum_abs_iff_product_nonpositive_l1216_121683


namespace NUMINAMATH_CALUDE_line_intersection_x_axis_l1216_121644

/-- Given a line passing through (2, 6) and (5, c) that intersects the x-axis at (d, 0), prove that d = -16 -/
theorem line_intersection_x_axis (c : ℝ) (d : ℝ) : 
  (∃ (m : ℝ), (6 - 0) = m * (2 - d) ∧ (c - 6) = m * (5 - 2)) → d = -16 := by
  sorry

end NUMINAMATH_CALUDE_line_intersection_x_axis_l1216_121644


namespace NUMINAMATH_CALUDE_rulers_in_drawer_l1216_121694

/-- The number of rulers remaining in a drawer after some are removed -/
def rulers_remaining (initial : ℕ) (removed : ℕ) : ℕ :=
  initial - removed

/-- Theorem: Given 46 initial rulers and 25 removed, 21 rulers remain -/
theorem rulers_in_drawer : rulers_remaining 46 25 = 21 := by
  sorry

end NUMINAMATH_CALUDE_rulers_in_drawer_l1216_121694


namespace NUMINAMATH_CALUDE_percentage_knives_after_trade_l1216_121605

/-- Represents Carolyn's silverware set -/
structure Silverware :=
  (knives : ℕ)
  (forks : ℕ)
  (spoons : ℕ)

/-- Calculates the total number of silverware pieces -/
def total_silverware (s : Silverware) : ℕ :=
  s.knives + s.forks + s.spoons

/-- Calculates the percentage of knives in the silverware set -/
def percentage_knives (s : Silverware) : ℚ :=
  (s.knives : ℚ) / (total_silverware s : ℚ) * 100

/-- The initial silverware set -/
def initial_set : Silverware :=
  { knives := 6
  , forks := 12
  , spoons := 6 * 3 }

/-- The silverware set after the trade -/
def after_trade_set : Silverware :=
  { knives := initial_set.knives + 10
  , forks := initial_set.forks
  , spoons := initial_set.spoons - 6 }

theorem percentage_knives_after_trade :
  percentage_knives after_trade_set = 40 := by
  sorry


end NUMINAMATH_CALUDE_percentage_knives_after_trade_l1216_121605


namespace NUMINAMATH_CALUDE_total_salaries_proof_l1216_121696

/-- Proves that given the conditions of A and B's salaries and spending,
    their total salaries amount to $5000 -/
theorem total_salaries_proof (A_salary B_salary : ℝ) : 
  A_salary = 3750 →
  A_salary * 0.05 = B_salary * 0.15 →
  A_salary + B_salary = 5000 := by
  sorry

end NUMINAMATH_CALUDE_total_salaries_proof_l1216_121696


namespace NUMINAMATH_CALUDE_part1_solution_set_part2_m_range_l1216_121639

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |x + m| + |2 * x - 1|

-- Part 1
theorem part1_solution_set :
  {x : ℝ | f 1 x ≥ 3} = {x : ℝ | x ≤ -1 ∨ x ≥ 1} :=
sorry

-- Part 2
theorem part2_m_range (m : ℝ) (hm : m > 0) :
  (∀ x ∈ Set.Icc m (2 * m^2), (1/2) * f m x ≤ |x + 1|) →
  1/2 < m ∧ m ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_part1_solution_set_part2_m_range_l1216_121639


namespace NUMINAMATH_CALUDE_complementary_events_l1216_121676

/-- Represents the color of a ball -/
inductive Color
| Red
| White

/-- Represents the outcome of drawing two balls -/
structure Draw where
  first : Color
  second : Color

/-- The set of all possible draws -/
def all_draws : Finset Draw :=
  sorry

/-- The event "Exactly no red ball" -/
def exactly_no_red (draw : Draw) : Prop :=
  draw.first = Color.White ∧ draw.second = Color.White

/-- The event "At most 1 white ball" -/
def at_most_one_white (draw : Draw) : Prop :=
  draw.first = Color.Red ∨ draw.second = Color.Red

/-- Theorem stating that "Exactly no red ball" and "At most 1 white ball" are complementary events -/
theorem complementary_events :
  ∀ (draw : Draw), draw ∈ all_draws → (exactly_no_red draw ↔ ¬at_most_one_white draw) :=
sorry

end NUMINAMATH_CALUDE_complementary_events_l1216_121676


namespace NUMINAMATH_CALUDE_office_gender_ratio_l1216_121634

/-- Given an office with 60 employees, if a meeting of 4 men and 6 women
    reduces the number of women on the office floor by 20%,
    then the ratio of men to women in the office is 1:1. -/
theorem office_gender_ratio
  (total_employees : ℕ)
  (meeting_men : ℕ)
  (meeting_women : ℕ)
  (women_reduction_percent : ℚ)
  (h1 : total_employees = 60)
  (h2 : meeting_men = 4)
  (h3 : meeting_women = 6)
  (h4 : women_reduction_percent = 1/5)
  : (total_employees / 2 : ℚ) = (total_employees - (total_employees / 2) : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_office_gender_ratio_l1216_121634


namespace NUMINAMATH_CALUDE_pelicans_in_shark_bite_cove_l1216_121681

/-- The number of Pelicans remaining in Shark Bite Cove after some have moved to Pelican Bay -/
def remaining_pelicans (initial_pelicans : ℕ) : ℕ :=
  initial_pelicans - initial_pelicans / 3

/-- The theorem stating the number of remaining Pelicans in Shark Bite Cove -/
theorem pelicans_in_shark_bite_cove :
  ∃ (initial_pelicans : ℕ),
    (2 * initial_pelicans = 60) ∧
    (remaining_pelicans initial_pelicans = 20) := by
  sorry

end NUMINAMATH_CALUDE_pelicans_in_shark_bite_cove_l1216_121681


namespace NUMINAMATH_CALUDE_min_value_theorem_l1216_121667

theorem min_value_theorem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 3) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 3 → 1/x + 4/(5+y) ≥ 9/8) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 3 ∧ 1/x + 4/(5+y) = 9/8) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1216_121667


namespace NUMINAMATH_CALUDE_original_average_age_l1216_121627

/-- Proves that the original average age of a class was 40 years, given the conditions of the problem -/
theorem original_average_age (original_count : ℕ) (new_count : ℕ) (new_avg_age : ℕ) (avg_decrease : ℕ) :
  original_count = 10 →
  new_count = 10 →
  new_avg_age = 32 →
  avg_decrease = 4 →
  ∃ (original_avg_age : ℕ),
    (original_avg_age * original_count + new_avg_age * new_count) / (original_count + new_count) 
    = original_avg_age - avg_decrease ∧
    original_avg_age = 40 :=
by sorry

end NUMINAMATH_CALUDE_original_average_age_l1216_121627


namespace NUMINAMATH_CALUDE_bus_driver_distance_to_destination_l1216_121674

theorem bus_driver_distance_to_destination :
  ∀ (distance_to_destination : ℝ),
    (distance_to_destination / 30 + (distance_to_destination + 10) / 30 + 2 = 6) →
    distance_to_destination = 55 := by
  sorry

end NUMINAMATH_CALUDE_bus_driver_distance_to_destination_l1216_121674


namespace NUMINAMATH_CALUDE_inscribed_cylinder_radius_l1216_121635

/-- 
A right circular cylinder is inscribed in a right circular cone.
The cylinder's diameter equals its height.
The cone has a diameter of 8 and an altitude of 10.
The axes of the cylinder and the cone coincide.
-/
theorem inscribed_cylinder_radius (r : ℝ) : r = 20 / 9 :=
  let cone_diameter := 8
  let cone_altitude := 10
  let cylinder_height := 2 * r
  -- The cylinder's diameter equals its height
  have h1 : cylinder_height = 2 * r := rfl
  -- The cone has a diameter of 8 and an altitude of 10
  have h2 : cone_diameter = 8 := rfl
  have h3 : cone_altitude = 10 := rfl
  -- The axes of the cylinder and the cone coincide (implicit in the problem setup)
  sorry


end NUMINAMATH_CALUDE_inscribed_cylinder_radius_l1216_121635


namespace NUMINAMATH_CALUDE_cookies_given_to_cousin_l1216_121678

theorem cookies_given_to_cousin (initial_boxes : ℕ) (brother_boxes : ℕ) (sister_boxes : ℕ) (self_boxes : ℕ) :
  initial_boxes = 45 →
  brother_boxes = 12 →
  sister_boxes = 9 →
  self_boxes = 17 →
  initial_boxes - brother_boxes - sister_boxes - self_boxes = 7 :=
by sorry

end NUMINAMATH_CALUDE_cookies_given_to_cousin_l1216_121678


namespace NUMINAMATH_CALUDE_pencils_per_row_l1216_121640

theorem pencils_per_row (packs : ℕ) (pencils_per_pack : ℕ) (rows : ℕ) 
  (h1 : packs = 35) 
  (h2 : pencils_per_pack = 4) 
  (h3 : rows = 70) : 
  (packs * pencils_per_pack) / rows = 2 := by
  sorry

end NUMINAMATH_CALUDE_pencils_per_row_l1216_121640


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_condition_l1216_121679

def sum_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def satisfies_condition (n : ℕ) : Prop :=
  is_perfect_square (sum_squares n * (sum_squares (3 * n) - sum_squares n))

theorem smallest_n_satisfying_condition :
  (∀ m : ℕ, 10 ≤ m ∧ m < 71 → ¬ satisfies_condition m) ∧
  satisfies_condition 71 := by sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_condition_l1216_121679


namespace NUMINAMATH_CALUDE_crane_flock_size_l1216_121626

theorem crane_flock_size (duck_flock_size : ℕ) (total_birds : ℕ) (h1 : duck_flock_size = 13) (h2 : total_birds = 221) (h3 : total_birds % duck_flock_size = 0) :
  ∃ (crane_flock_size : ℕ), crane_flock_size = total_birds ∧ total_birds % crane_flock_size = 0 := by
  sorry

end NUMINAMATH_CALUDE_crane_flock_size_l1216_121626


namespace NUMINAMATH_CALUDE_slower_whale_length_is_45_l1216_121682

/-- The length of a slower whale given the speeds of two whales and the time for the faster to cross the slower -/
def slower_whale_length (faster_speed slower_speed crossing_time : ℝ) : ℝ :=
  (faster_speed - slower_speed) * crossing_time

/-- Theorem stating that the length of the slower whale is 45 meters given the problem conditions -/
theorem slower_whale_length_is_45 :
  slower_whale_length 18 15 15 = 45 := by
  sorry

end NUMINAMATH_CALUDE_slower_whale_length_is_45_l1216_121682


namespace NUMINAMATH_CALUDE_solution_set_for_f_l1216_121689

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem solution_set_for_f
  (a : ℝ)
  (h1 : a > 0)
  (h2 : a ≠ 1)
  (h3 : f a (2/a) > f a (3/a)) :
  ∀ x, f a (1 - 1/x) > 1 ↔ 1 < x ∧ x < 1/(1-a) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_for_f_l1216_121689


namespace NUMINAMATH_CALUDE_impossibility_of_1x3_rectangle_l1216_121654

/-- Represents a cell on the grid -/
structure Cell :=
  (x : Fin 8)
  (y : Fin 8)

/-- Represents a 1x2 rectangle on the grid -/
structure Rectangle :=
  (topLeft : Cell)
  (isVertical : Bool)

/-- Checks if a cell is covered by a rectangle -/
def isCovered (c : Cell) (r : Rectangle) : Prop :=
  (c.x = r.topLeft.x ∧ c.y = r.topLeft.y) ∨
  (r.isVertical ∧ c.x = r.topLeft.x ∧ c.y = r.topLeft.y + 1) ∨
  (¬r.isVertical ∧ c.x = r.topLeft.x + 1 ∧ c.y = r.topLeft.y)

/-- Checks if three consecutive cells form a 1x3 rectangle -/
def is1x3Rectangle (c1 c2 c3 : Cell) : Prop :=
  (c1.x = c2.x ∧ c2.x = c3.x ∧ c2.y = c1.y + 1 ∧ c3.y = c2.y + 1) ∨
  (c1.y = c2.y ∧ c2.y = c3.y ∧ c2.x = c1.x + 1 ∧ c3.x = c2.x + 1)

/-- The main theorem -/
theorem impossibility_of_1x3_rectangle :
  ∃ (configuration : Finset Rectangle),
    configuration.card = 12 ∧
    (∀ c1 c2 c3 : Cell,
      is1x3Rectangle c1 c2 c3 →
      ∃ r ∈ configuration, isCovered c1 r ∨ isCovered c2 r ∨ isCovered c3 r) :=
by
  sorry

end NUMINAMATH_CALUDE_impossibility_of_1x3_rectangle_l1216_121654


namespace NUMINAMATH_CALUDE_triangle_angle_ratio_l1216_121669

theorem triangle_angle_ratio (a b c : ℝ) (h_sum : a + b + c = 180)
  (h_ratio : ∃ (x : ℝ), a = 4*x ∧ b = 5*x ∧ c = 9*x) (h_smallest : min a (min b c) > 40) :
  max a (max b c) = 90 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_ratio_l1216_121669


namespace NUMINAMATH_CALUDE_part_I_part_II_l1216_121602

-- Define propositions p and q
def p (a : ℝ) : Prop := a > 0
def q (a : ℝ) : Prop := ∀ x : ℝ, x ∈ Set.Icc (-2) (-1) → a - x ≥ 0

-- Part I
theorem part_I (a : ℝ) (hq : q a) : a ∈ {a : ℝ | a ≥ -1} := by sorry

-- Part II
theorem part_II (a : ℝ) (h_or : p a ∨ q a) (h_not_and : ¬(p a ∧ q a)) : 
  a ∈ Set.Icc (-1) 0 := by sorry

end NUMINAMATH_CALUDE_part_I_part_II_l1216_121602


namespace NUMINAMATH_CALUDE_sum_of_abs_roots_l1216_121691

theorem sum_of_abs_roots : ∃ (r₁ r₂ r₃ r₄ : ℝ),
  (∀ x : ℝ, x^4 - 4*x^3 - 4*x^2 + 16*x - 8 = 0 ↔ (x = r₁ ∨ x = r₂ ∨ x = r₃ ∨ x = r₄)) ∧
  |r₁| + |r₂| + |r₃| + |r₄| = 2 + 2 * Real.sqrt 2 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_abs_roots_l1216_121691


namespace NUMINAMATH_CALUDE_am_gm_inequality_application_l1216_121668

theorem am_gm_inequality_application (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  (x + y^3) * (x^3 + y) ≥ 4 * x^2 * y^2 ∧
  ((x + y^3) * (x^3 + y) = 4 * x^2 * y^2 ↔ (x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 1)) :=
by sorry

end NUMINAMATH_CALUDE_am_gm_inequality_application_l1216_121668


namespace NUMINAMATH_CALUDE_pi_over_two_not_fraction_l1216_121670

-- Define what a fraction is
def is_fraction (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

-- State the theorem
theorem pi_over_two_not_fraction : ¬ is_fraction (π / 2) := by
  sorry

end NUMINAMATH_CALUDE_pi_over_two_not_fraction_l1216_121670


namespace NUMINAMATH_CALUDE_stating_lens_screen_distance_l1216_121673

/-- Represents the focal length of a thin lens in centimeters -/
def focal_length : ℝ := 150

/-- Represents the distance the screen is moved in centimeters -/
def screen_movement : ℝ := 40

/-- Represents the possible initial distances from the lens to the screen in centimeters -/
def initial_distances : Set ℝ := {130, 170}

/-- 
Theorem stating that given a thin lens with focal length of 150 cm and a screen
that produces the same diameter spot when moved 40 cm, the initial distance
from the lens to the screen is either 130 cm or 170 cm.
-/
theorem lens_screen_distance 
  (s : ℝ) 
  (h1 : s ∈ initial_distances) 
  (h2 : s = focal_length + screen_movement / 2 ∨ s = focal_length - screen_movement / 2) : 
  s ∈ initial_distances :=
sorry

end NUMINAMATH_CALUDE_stating_lens_screen_distance_l1216_121673


namespace NUMINAMATH_CALUDE_circle_square_area_difference_l1216_121617

/-- The difference between the areas of the non-overlapping portions of a circle and a square -/
theorem circle_square_area_difference (r c s : ℝ) (h1 : r = 3) (h2 : s = 2) : 
  (π * r^2 - s^2) = 9 * π - 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_square_area_difference_l1216_121617


namespace NUMINAMATH_CALUDE_circle_lines_theorem_l1216_121616

/-- The number of points on the circle -/
def n : ℕ := 5

/-- The total number of lines between any two points -/
def total_lines (m : ℕ) : ℕ := m * (m - 1) / 2

/-- The number of lines between immediate neighbors -/
def neighbor_lines (m : ℕ) : ℕ := m

/-- The number of valid lines (excluding immediate neighbors) -/
def valid_lines (m : ℕ) : ℕ := total_lines m - neighbor_lines m

theorem circle_lines_theorem : valid_lines n = 5 := by sorry

end NUMINAMATH_CALUDE_circle_lines_theorem_l1216_121616


namespace NUMINAMATH_CALUDE_trig_equation_solution_l1216_121684

noncomputable def solve_trig_equation (x : ℝ) : Prop :=
  (1 - Real.sin (2 * x) ≠ 0) ∧ 
  (1 - Real.tan x ≠ 0) ∧ 
  (Real.cos x ≠ 0) ∧
  ((1 + Real.sin (2 * x)) / (1 - Real.sin (2 * x)) + 
   2 * ((1 + Real.tan x) / (1 - Real.tan x)) - 3 = 0)

theorem trig_equation_solution :
  ∀ x : ℝ, solve_trig_equation x ↔ 
    (∃ k : ℤ, x = k * Real.pi) ∨
    (∃ n : ℤ, x = Real.arctan 2 + n * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_trig_equation_solution_l1216_121684


namespace NUMINAMATH_CALUDE_x_power_4374_minus_reciprocal_l1216_121648

theorem x_power_4374_minus_reciprocal (x : ℂ) : 
  x - (1 / x) = -Complex.I * Real.sqrt 6 → x^4374 - (1 / x^4374) = 0 := by
  sorry

end NUMINAMATH_CALUDE_x_power_4374_minus_reciprocal_l1216_121648


namespace NUMINAMATH_CALUDE_circle_and_tangent_lines_l1216_121601

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  ∃ (m : ℝ), (x - m)^2 + (y - 3*m)^2 = (1 - m)^2 + (6 - 3*m)^2 ∧
             (x - m)^2 + (y - 3*m)^2 = (-2 - m)^2 + (3 - 3*m)^2

-- Define the line 3x-y=0
def center_line (x y : ℝ) : Prop := 3*x - y = 0

-- Define the point P
def point_P : ℝ × ℝ := (4, 1)

-- Theorem statement
theorem circle_and_tangent_lines :
  ∃ (x₀ y₀ r : ℝ),
    (∀ x y, circle_C x y ↔ (x - x₀)^2 + (y - y₀)^2 = r^2) ∧
    center_line x₀ y₀ ∧
    ((x₀ - 1)^2 + (y₀ - 3)^2 = 9) ∧
    (∀ x y, (5*x - 12*y - 8 = 0 ∨ x = 4) →
      ((x - x₀)^2 + (y - y₀)^2 = r^2 ∧
       ((x - 4)^2 + (y - 1)^2) * r^2 = ((x - x₀)*(4 - x₀) + (y - y₀)*(1 - y₀))^2)) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_tangent_lines_l1216_121601


namespace NUMINAMATH_CALUDE_sufficient_condition_range_l1216_121685

theorem sufficient_condition_range (p : ℝ) : 
  (∀ x : ℝ, 4*x + p < 0 → x^2 - x - 2 > 0) ∧ 
  (∃ x : ℝ, x^2 - x - 2 > 0 ∧ 4*x + p ≥ 0) →
  p ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_sufficient_condition_range_l1216_121685


namespace NUMINAMATH_CALUDE_no_integer_roots_l1216_121659

theorem no_integer_roots : ¬ ∃ (x : ℤ), x^3 - 4*x^2 - 14*x + 28 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_roots_l1216_121659


namespace NUMINAMATH_CALUDE_largest_difference_l1216_121653

def A : ℕ := 3 * 2010^2011
def B : ℕ := 2010^2011
def C : ℕ := 2009 * 2010^2010
def D : ℕ := 3 * 2010^2010
def E : ℕ := 2010^2010
def F : ℕ := 2010^2009

theorem largest_difference : 
  (A - B) > (B - C) ∧ 
  (A - B) > (C - D) ∧ 
  (A - B) > (D - E) ∧ 
  (A - B) > (E - F) :=
sorry

end NUMINAMATH_CALUDE_largest_difference_l1216_121653


namespace NUMINAMATH_CALUDE_base4_sequence_implies_bcd_52_l1216_121652

/-- Represents a digit in base-4 --/
inductive Base4Digit
| A
| B
| C
| D

/-- Converts a Base4Digit to its numerical value --/
def base4DigitToInt (d : Base4Digit) : Nat :=
  match d with
  | Base4Digit.A => 0
  | Base4Digit.B => 1
  | Base4Digit.C => 2
  | Base4Digit.D => 3

/-- Represents a three-digit number in base-4 --/
structure Base4Number :=
  (hundreds : Base4Digit)
  (tens : Base4Digit)
  (ones : Base4Digit)

/-- Converts a Base4Number to its base-10 representation --/
def toBase10 (n : Base4Number) : Nat :=
  4 * 4 * (base4DigitToInt n.hundreds) + 4 * (base4DigitToInt n.tens) + (base4DigitToInt n.ones)

theorem base4_sequence_implies_bcd_52 
  (n1 n2 n3 : Base4Number)
  (h1 : toBase10 n2 = toBase10 n1 + 1)
  (h2 : toBase10 n3 = toBase10 n2 + 1)
  (h3 : n1.hundreds = n2.hundreds ∧ n1.tens = n2.tens)
  (h4 : n2.hundreds = n3.hundreds ∧ n3.tens = Base4Digit.C)
  (h5 : n1.hundreds = Base4Digit.A ∧ n2.hundreds = Base4Digit.A ∧ n3.hundreds = Base4Digit.A)
  (h6 : n1.tens = Base4Digit.B ∧ n2.tens = Base4Digit.B)
  (h7 : n1.ones = Base4Digit.C ∧ n2.ones = Base4Digit.D ∧ n3.ones = Base4Digit.A) :
  toBase10 { hundreds := Base4Digit.B, tens := Base4Digit.C, ones := Base4Digit.D } = 52 := by
  sorry

end NUMINAMATH_CALUDE_base4_sequence_implies_bcd_52_l1216_121652


namespace NUMINAMATH_CALUDE_max_value_quadratic_expression_l1216_121604

theorem max_value_quadratic_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 - 2*x*y + 3*y^2 = 9) : 
  ∀ a b : ℝ, a > 0 → b > 0 → a^2 - 2*a*b + 3*b^2 = 9 → 
  x^2 + 2*x*y + 3*y^2 ≤ 18 + 9*Real.sqrt 3 ∧ 
  ∃ c d : ℝ, c > 0 ∧ d > 0 ∧ c^2 - 2*c*d + 3*d^2 = 9 ∧ 
  c^2 + 2*c*d + 3*d^2 = 18 + 9*Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_quadratic_expression_l1216_121604


namespace NUMINAMATH_CALUDE_satisfying_digits_characterization_l1216_121641

/-- A digit is a natural number less than 10. -/
def Digit : Type := { d : ℕ // d < 10 }

/-- The set of digits that satisfy the given property. -/
def SatisfyingDigits : Set Digit :=
  { z : Digit | ∀ k : ℕ, k ≥ 1 → ∃ n : ℕ, n ≥ 1 ∧ n^9 % 10^k = z.val^k % 10^k }

/-- The theorem stating that the satisfying digits are exactly 0, 1, 5, and 6. -/
theorem satisfying_digits_characterization :
  SatisfyingDigits = {⟨0, by norm_num⟩, ⟨1, by norm_num⟩, ⟨5, by norm_num⟩, ⟨6, by norm_num⟩} :=
by sorry

end NUMINAMATH_CALUDE_satisfying_digits_characterization_l1216_121641


namespace NUMINAMATH_CALUDE_circle_area_ratio_l1216_121665

theorem circle_area_ratio (A B : Real) (rA rB : ℝ) (hA : A = 2 * π * rA) (hB : B = 2 * π * rB)
  (h_arc : (60 / 360) * A = (40 / 360) * B) :
  π * rA^2 / (π * rB^2) = 4 / 9 := by
sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l1216_121665


namespace NUMINAMATH_CALUDE_sixth_number_divisible_by_45_and_6_l1216_121637

/-- The least common multiple of 45 and 6 -/
def lcm_45_6 : ℕ := 90

/-- The first multiple of lcm_45_6 greater than 190 -/
def first_multiple : ℕ := 270

/-- The ending number we want to prove -/
def ending_number : ℕ := 720

/-- The theorem to prove -/
theorem sixth_number_divisible_by_45_and_6 : 
  ending_number = first_multiple + 5 * lcm_45_6 ∧ 
  ending_number % 45 = 0 ∧ 
  ending_number % 6 = 0 ∧
  ∀ n : ℕ, first_multiple ≤ n ∧ n < ending_number ∧ n % 45 = 0 ∧ n % 6 = 0 → 
    ∃ k : ℕ, k < 6 ∧ n = first_multiple + k * lcm_45_6 :=
by sorry

end NUMINAMATH_CALUDE_sixth_number_divisible_by_45_and_6_l1216_121637


namespace NUMINAMATH_CALUDE_min_value_of_function_min_value_achievable_l1216_121662

theorem min_value_of_function (x : ℝ) : 3 * x^2 + 6 / (x^2 + 1) ≥ 6 * Real.sqrt 2 - 3 := by
  sorry

theorem min_value_achievable : ∃ x : ℝ, 3 * x^2 + 6 / (x^2 + 1) = 6 * Real.sqrt 2 - 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_min_value_achievable_l1216_121662


namespace NUMINAMATH_CALUDE_function_symmetry_l1216_121677

/-- Given a function f(x) = ax^4 - bx^2 + c - 1 where a, b, and c are real numbers,
    if f(2) = -1, then f(-2) = -1 -/
theorem function_symmetry (a b c : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x^4 - b * x^2 + c - 1
  f 2 = -1 → f (-2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_function_symmetry_l1216_121677


namespace NUMINAMATH_CALUDE_decimal_expansion_prime_modulo_l1216_121647

theorem decimal_expansion_prime_modulo
  (p : ℕ) (r : ℕ) (hp : p.Prime) (hp_gt_5 : p > 5)
  (hr : ∃ (a : ℕ → ℕ), (∀ i, a i < 10) ∧
    (1 : ℚ) / p = ∑' i, (a i : ℚ) / (10 ^ (i + 1)) - ∑' i, (a (i % r) : ℚ) / (10 ^ (i + r + 1)))
  : 10 ^ r ≡ 1 [MOD p] :=
sorry

end NUMINAMATH_CALUDE_decimal_expansion_prime_modulo_l1216_121647


namespace NUMINAMATH_CALUDE_rectangle_coverage_l1216_121615

/-- An L-shaped figure made of 4 unit squares -/
structure LShape :=
  (size : Nat)
  (h_size : size = 4)

/-- Represents a rectangle with dimensions m × n -/
structure Rectangle (m n : Nat) :=
  (width : Nat)
  (height : Nat)
  (h_width : width = m)
  (h_height : height = n)
  (h_positive : m > 1 ∧ n > 1)

/-- Predicate to check if a number is a multiple of 8 -/
def IsMultipleOf8 (n : Nat) : Prop := ∃ k, n = 8 * k

/-- Predicate to check if a rectangle can be covered by L-shaped figures -/
def CanBeCovered (r : Rectangle m n) (l : LShape) : Prop :=
  ∃ (arrangement : Nat), True  -- We don't define the specific arrangement here

/-- The main theorem -/
theorem rectangle_coverage (m n : Nat) (r : Rectangle m n) (l : LShape) :
  (CanBeCovered r l) ↔ (IsMultipleOf8 (m * n)) :=
sorry

end NUMINAMATH_CALUDE_rectangle_coverage_l1216_121615


namespace NUMINAMATH_CALUDE_bob_oyster_shucking_l1216_121680

/-- The number of oysters Bob can shuck in 2 hours -/
def oysters_in_two_hours : ℕ :=
  let oysters_per_five_minutes : ℕ := 10
  let minutes_per_hour : ℕ := 60
  let hours : ℕ := 2
  let total_minutes : ℕ := hours * minutes_per_hour
  (oysters_per_five_minutes * total_minutes) / 5

theorem bob_oyster_shucking :
  oysters_in_two_hours = 240 :=
by sorry

end NUMINAMATH_CALUDE_bob_oyster_shucking_l1216_121680


namespace NUMINAMATH_CALUDE_planes_parallel_to_same_plane_are_parallel_l1216_121666

-- Define a type for points in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a type for planes in 3D space
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Define parallelism between planes
def parallel (p1 p2 : Plane3D) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ p1.a = k * p2.a ∧ p1.b = k * p2.b ∧ p1.c = k * p2.c

-- Theorem statement
theorem planes_parallel_to_same_plane_are_parallel (p1 p2 p3 : Plane3D) :
  parallel p1 p3 → parallel p2 p3 → parallel p1 p2 := by
  sorry


end NUMINAMATH_CALUDE_planes_parallel_to_same_plane_are_parallel_l1216_121666
