import Mathlib

namespace NUMINAMATH_CALUDE_fraction_equality_l3492_349241

theorem fraction_equality : 48 / (7 - 3/8 + 4/9) = 3456 / 509 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3492_349241


namespace NUMINAMATH_CALUDE_total_pears_picked_l3492_349280

theorem total_pears_picked (sara_pears sally_pears : ℕ) 
  (h1 : sara_pears = 45)
  (h2 : sally_pears = 11) :
  sara_pears + sally_pears = 56 := by
  sorry

end NUMINAMATH_CALUDE_total_pears_picked_l3492_349280


namespace NUMINAMATH_CALUDE_savings_calculation_l3492_349236

theorem savings_calculation (income : ℕ) (ratio_income : ℕ) (ratio_expenditure : ℕ) 
  (h1 : income = 20000)
  (h2 : ratio_income = 4)
  (h3 : ratio_expenditure = 3) :
  income - (income * ratio_expenditure / ratio_income) = 5000 := by
  sorry

end NUMINAMATH_CALUDE_savings_calculation_l3492_349236


namespace NUMINAMATH_CALUDE_files_remaining_l3492_349293

theorem files_remaining (initial_music : ℕ) (initial_video : ℕ) (deleted : ℕ) : 
  initial_music = 4 → initial_video = 21 → deleted = 23 → 
  initial_music + initial_video - deleted = 2 := by
  sorry

end NUMINAMATH_CALUDE_files_remaining_l3492_349293


namespace NUMINAMATH_CALUDE_fifty_seventh_digit_of_1_13_l3492_349282

def decimal_rep_1_13 : List Nat := [0, 7, 6, 9, 2, 3]

theorem fifty_seventh_digit_of_1_13 : 
  (decimal_rep_1_13[(57 - 1) % decimal_rep_1_13.length] = 6) := by
  sorry

end NUMINAMATH_CALUDE_fifty_seventh_digit_of_1_13_l3492_349282


namespace NUMINAMATH_CALUDE_x_value_when_y_is_half_l3492_349258

theorem x_value_when_y_is_half :
  ∀ x y : ℝ, y = 1 / (4 * x + 2) → y = 1 / 2 → x = 0 := by
sorry

end NUMINAMATH_CALUDE_x_value_when_y_is_half_l3492_349258


namespace NUMINAMATH_CALUDE_prep_school_cost_per_semester_l3492_349227

/-- The cost per semester for John's son's prep school -/
def cost_per_semester (total_cost : ℕ) (years : ℕ) (semesters_per_year : ℕ) : ℕ :=
  total_cost / (years * semesters_per_year)

/-- Proof that the cost per semester is $20,000 -/
theorem prep_school_cost_per_semester :
  cost_per_semester 520000 13 2 = 20000 := by
  sorry

end NUMINAMATH_CALUDE_prep_school_cost_per_semester_l3492_349227


namespace NUMINAMATH_CALUDE_divisible_by_eleven_l3492_349214

/-- A seven-digit number in the form 8n46325 where n is a single digit -/
def sevenDigitNumber (n : ℕ) : ℕ := 8000000 + 1000000*n + 46325

/-- Predicate to check if a natural number is a single digit -/
def isSingleDigit (n : ℕ) : Prop := n < 10

theorem divisible_by_eleven (n : ℕ) : 
  isSingleDigit n → (sevenDigitNumber n) % 11 = 0 → n = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_eleven_l3492_349214


namespace NUMINAMATH_CALUDE_max_y_value_l3492_349275

theorem max_y_value (x y : ℤ) (h : x * y + 3 * x + 2 * y = -2) : y ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_max_y_value_l3492_349275


namespace NUMINAMATH_CALUDE_orchids_cut_l3492_349277

theorem orchids_cut (initial_roses initial_orchids final_roses final_orchids : ℕ) 
  (h1 : initial_roses = 16)
  (h2 : initial_orchids = 3)
  (h3 : final_roses = 13)
  (h4 : final_orchids = 7) :
  final_orchids - initial_orchids = 4 := by
  sorry

end NUMINAMATH_CALUDE_orchids_cut_l3492_349277


namespace NUMINAMATH_CALUDE_sin_beta_value_l3492_349257

theorem sin_beta_value (α β : Real) 
  (h : Real.sin (α - β) * Real.cos α - Real.cos (α - β) * Real.sin α = 3/5) : 
  Real.sin β = -3/5 := by
sorry

end NUMINAMATH_CALUDE_sin_beta_value_l3492_349257


namespace NUMINAMATH_CALUDE_solution_range_l3492_349231

theorem solution_range (x : ℝ) : 
  x > 9 → 
  Real.sqrt (x - 5 * Real.sqrt (x - 9)) + 3 = Real.sqrt (x + 5 * Real.sqrt (x - 9)) - 3 → 
  x ≥ 20.80 := by
  sorry

end NUMINAMATH_CALUDE_solution_range_l3492_349231


namespace NUMINAMATH_CALUDE_cards_given_away_ben_cards_given_away_l3492_349208

theorem cards_given_away (basketball_boxes : Nat) (basketball_cards_per_box : Nat)
                         (baseball_boxes : Nat) (baseball_cards_per_box : Nat)
                         (cards_left : Nat) : Nat :=
  let total_cards := basketball_boxes * basketball_cards_per_box + 
                     baseball_boxes * baseball_cards_per_box
  total_cards - cards_left

theorem ben_cards_given_away : 
  cards_given_away 4 10 5 8 22 = 58 := by
  sorry

end NUMINAMATH_CALUDE_cards_given_away_ben_cards_given_away_l3492_349208


namespace NUMINAMATH_CALUDE_trumpet_cost_l3492_349215

/-- The cost of Mike's trumpet, given the net amount spent and the amount received for selling a song book. -/
theorem trumpet_cost (net_spent : ℝ) (song_book_sold : ℝ) (h1 : net_spent = 139.32) (h2 : song_book_sold = 5.84) :
  net_spent + song_book_sold = 145.16 := by
  sorry

end NUMINAMATH_CALUDE_trumpet_cost_l3492_349215


namespace NUMINAMATH_CALUDE_line_slope_through_circle_l3492_349271

/-- Given a line passing through (0,√5) and intersecting the circle x^2 + y^2 = 16 at points A and B,
    if a point P on the circle satisfies OP = OA + OB, then the slope of the line is ±1/2. -/
theorem line_slope_through_circle (A B P : ℝ × ℝ) : 
  let O : ℝ × ℝ := (0, 0)
  let line := {(x, y) : ℝ × ℝ | ∃ (k : ℝ), y - Real.sqrt 5 = k * x}
  (0, Real.sqrt 5) ∈ line ∧ 
  A ∈ line ∧ 
  B ∈ line ∧
  A.1^2 + A.2^2 = 16 ∧
  B.1^2 + B.2^2 = 16 ∧
  P.1^2 + P.2^2 = 16 ∧
  (P.1 - O.1, P.2 - O.2) = (A.1 - O.1, A.2 - O.2) + (B.1 - O.1, B.2 - O.2) →
  ∃ (k : ℝ), k = 1/2 ∨ k = -1/2 ∧ ∀ (x y : ℝ), (x, y) ∈ line ↔ y - Real.sqrt 5 = k * x :=
by sorry

end NUMINAMATH_CALUDE_line_slope_through_circle_l3492_349271


namespace NUMINAMATH_CALUDE_factorization_p1_factorization_p2_l3492_349233

-- Define the polynomials
def p1 (x : ℝ) : ℝ := (x^2 - 2*x - 1) * (x^2 - 2*x + 3) + 4
def p2 (x : ℝ) : ℝ := (x^2 + 6*x) * (x^2 + 6*x + 18) + 81

-- State the theorems
theorem factorization_p1 : ∀ x : ℝ, p1 x = (x - 1)^4 := by sorry

theorem factorization_p2 : ∀ x : ℝ, p2 x = (x + 3)^4 := by sorry

end NUMINAMATH_CALUDE_factorization_p1_factorization_p2_l3492_349233


namespace NUMINAMATH_CALUDE_inverse_square_difference_l3492_349245

theorem inverse_square_difference (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x^2 - y^2 = x*y) : 
  1/x^2 - 1/y^2 = -1/(x*y) := by
  sorry

end NUMINAMATH_CALUDE_inverse_square_difference_l3492_349245


namespace NUMINAMATH_CALUDE_red_ball_probability_three_drawers_l3492_349269

/-- Represents the contents of a drawer --/
structure Drawer where
  red_balls : ℕ
  white_balls : ℕ

/-- Calculates the probability of drawing a red ball from a drawer --/
def red_ball_probability (d : Drawer) : ℚ :=
  d.red_balls / (d.red_balls + d.white_balls)

/-- The probability of randomly selecting each drawer --/
def drawer_selection_probability : ℚ := 1 / 3

theorem red_ball_probability_three_drawers 
  (left middle right : Drawer)
  (h_left : left = ⟨0, 5⟩)
  (h_middle : middle = ⟨1, 1⟩)
  (h_right : right = ⟨2, 1⟩) :
  drawer_selection_probability * red_ball_probability middle +
  drawer_selection_probability * red_ball_probability right = 7 / 18 := by
  sorry

end NUMINAMATH_CALUDE_red_ball_probability_three_drawers_l3492_349269


namespace NUMINAMATH_CALUDE_sequence_inequality_l3492_349230

theorem sequence_inequality (a : ℕ → ℕ) 
  (h0 : ∀ n, a n > 0)
  (h1 : a 1 > a 0)
  (h2 : ∀ n ∈ Finset.range 99, a (n + 2) = 3 * a (n + 1) - 2 * a n) :
  a 100 > 2^99 := by
  sorry

end NUMINAMATH_CALUDE_sequence_inequality_l3492_349230


namespace NUMINAMATH_CALUDE_clock_rings_107_times_in_january_l3492_349281

/-- Calculates the number of times a clock rings in January -/
def clock_rings_in_january (ring_interval : ℕ) (days_in_january : ℕ) : ℕ :=
  let hours_in_january := days_in_january * 24
  (hours_in_january / ring_interval) + 1

/-- Theorem: A clock that rings every 7 hours will ring 107 times in January -/
theorem clock_rings_107_times_in_january :
  clock_rings_in_january 7 31 = 107 := by
  sorry

end NUMINAMATH_CALUDE_clock_rings_107_times_in_january_l3492_349281


namespace NUMINAMATH_CALUDE_rabbit_average_distance_l3492_349285

theorem rabbit_average_distance (side_length : ℝ) (diagonal_distance : ℝ) (perpendicular_distance : ℝ) : 
  side_length = 12 →
  diagonal_distance = 8.4 →
  perpendicular_distance = 3 →
  let diagonal := side_length * Real.sqrt 2
  let fraction := diagonal_distance / diagonal
  let x := fraction * side_length + perpendicular_distance
  let y := fraction * side_length
  let dist_left := x
  let dist_bottom := y
  let dist_right := side_length - x
  let dist_top := side_length - y
  (dist_left + dist_bottom + dist_right + dist_top) / 4 = 6 := by sorry

end NUMINAMATH_CALUDE_rabbit_average_distance_l3492_349285


namespace NUMINAMATH_CALUDE_course_length_l3492_349247

/-- Represents the time taken by Team B to complete the course -/
def team_b_time : ℝ := 15

/-- Represents the speed of Team B in miles per hour -/
def team_b_speed : ℝ := 20

/-- Represents the difference in completion time between Team A and Team B -/
def time_difference : ℝ := 3

/-- Represents the difference in speed between Team A and Team B -/
def speed_difference : ℝ := 5

/-- Theorem stating that the course length is 300 miles -/
theorem course_length : 
  team_b_speed * team_b_time = 300 :=
sorry

end NUMINAMATH_CALUDE_course_length_l3492_349247


namespace NUMINAMATH_CALUDE_geometric_mean_exponent_sum_l3492_349254

theorem geometric_mean_exponent_sum (a b : ℝ) : 
  a > 0 → b > 0 → (Real.sqrt 3)^2 = 3^a * 3^b → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_exponent_sum_l3492_349254


namespace NUMINAMATH_CALUDE_businessmen_drink_neither_l3492_349284

theorem businessmen_drink_neither (total : ℕ) (coffee : ℕ) (tea : ℕ) (both : ℕ) 
  (h1 : total = 30)
  (h2 : coffee = 15)
  (h3 : tea = 13)
  (h4 : both = 8) :
  total - (coffee + tea - both) = 10 :=
by sorry

end NUMINAMATH_CALUDE_businessmen_drink_neither_l3492_349284


namespace NUMINAMATH_CALUDE_sin_120_degrees_l3492_349211

theorem sin_120_degrees : Real.sin (120 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_120_degrees_l3492_349211


namespace NUMINAMATH_CALUDE_cost_of_dozen_pens_l3492_349243

/-- Given the cost of 3 pens and 5 pencils is Rs. 200, and the cost ratio of one pen to one pencil
    is 5:1, prove that the cost of one dozen pens is Rs. 600. -/
theorem cost_of_dozen_pens (pen_cost pencil_cost : ℚ) : 
  3 * pen_cost + 5 * pencil_cost = 200 →
  pen_cost = 5 * pencil_cost →
  12 * pen_cost = 600 := by
sorry

end NUMINAMATH_CALUDE_cost_of_dozen_pens_l3492_349243


namespace NUMINAMATH_CALUDE_sibling_ages_equations_l3492_349291

/-- Represents the ages of two siblings -/
structure SiblingAges where
  x : ℕ  -- Age of the older brother
  y : ℕ  -- Age of the younger sister

/-- The conditions for the sibling ages problem -/
def SiblingAgesProblem (ages : SiblingAges) : Prop :=
  (ages.x = 4 * ages.y) ∧ 
  (ages.x + 3 = 3 * (ages.y + 3))

/-- The theorem stating that the given system of equations is correct -/
theorem sibling_ages_equations (ages : SiblingAges) :
  SiblingAgesProblem ages ↔ 
  (ages.x + 3 = 3 * (ages.y + 3)) ∧ (ages.x = 4 * ages.y) :=
sorry

end NUMINAMATH_CALUDE_sibling_ages_equations_l3492_349291


namespace NUMINAMATH_CALUDE_root_difference_implies_k_value_l3492_349234

theorem root_difference_implies_k_value (k : ℝ) :
  (∀ x y : ℝ, x^2 + k*x + 10 = 0 ∧ y^2 - k*y + 10 = 0 ∧ y = x + 3) →
  k = 3 := by
sorry

end NUMINAMATH_CALUDE_root_difference_implies_k_value_l3492_349234


namespace NUMINAMATH_CALUDE_inverse_proportion_k_value_l3492_349202

theorem inverse_proportion_k_value (y : ℝ → ℝ) (k : ℝ) :
  (∀ x, x ≠ 0 → y x = k / x) →  -- Inverse proportion function
  y 3 = 2 →                     -- Passes through (3, 2)
  k = 6 :=                      -- Prove k = 6
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_k_value_l3492_349202


namespace NUMINAMATH_CALUDE_function_range_l3492_349209

-- Define the function
def f (x : ℝ) := x^2 - 2*x

-- Define the domain
def domain : Set ℝ := { x | -1 ≤ x ∧ x ≤ 3 }

-- State the theorem
theorem function_range : 
  { y | ∃ x ∈ domain, f x = y } = { y | -1 ≤ y ∧ y ≤ 3 } := by sorry

end NUMINAMATH_CALUDE_function_range_l3492_349209


namespace NUMINAMATH_CALUDE_expand_expression_l3492_349276

theorem expand_expression (x y z : ℝ) : 
  (2 * x + 5) * (3 * y + 4 * z + 15) = 6 * x * y + 8 * x * z + 30 * x + 15 * y + 20 * z + 75 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3492_349276


namespace NUMINAMATH_CALUDE_problem_solution_l3492_349219

-- Define the set M
def M : Set ℝ := {x | x^2 - x - 2 < 0}

-- Define the set N
def N (a b : ℝ) : Set ℝ := {x | a < x ∧ x < b}

theorem problem_solution :
  -- Part 1: M = (-1, 2)
  M = Set.Ioo (-1) 2 ∧
  -- Part 2: If M ⊇ N, then the minimum value of a is -1
  (∀ a b : ℝ, M ⊇ N a b → a ≥ -1) ∧
  (∃ a₀ : ℝ, a₀ = -1 ∧ ∃ b : ℝ, M ⊇ N a₀ b) ∧
  -- Part 3: If M ∩ N = M, then b ∈ [2, +∞)
  (∀ a b : ℝ, M ∩ N a b = M → b ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3492_349219


namespace NUMINAMATH_CALUDE_pension_calculation_l3492_349218

-- Define the pension function
noncomputable def pension (k : ℝ) (x : ℝ) : ℝ := k * Real.sqrt x

-- Define the problem parameters
variable (c d r s : ℝ)

-- State the theorem
theorem pension_calculation (h1 : d ≠ c) 
                            (h2 : pension k x - pension k (x - c) = r) 
                            (h3 : pension k x - pension k (x - d) = s) : 
  pension k x = (r^2 - s^2) / (2 * (r - s)) := by
  sorry

end NUMINAMATH_CALUDE_pension_calculation_l3492_349218


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l3492_349246

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_third_term
  (a : ℕ → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_roots : a 1 * a 5 = 9 ∧ a 1 + a 5 = 12) :
  a 3 = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l3492_349246


namespace NUMINAMATH_CALUDE_single_digit_between_zero_and_two_l3492_349226

theorem single_digit_between_zero_and_two : 
  ∃! n : ℕ, n < 10 ∧ 0 < n ∧ n < 2 :=
by sorry

end NUMINAMATH_CALUDE_single_digit_between_zero_and_two_l3492_349226


namespace NUMINAMATH_CALUDE_equal_area_dividing_line_slope_l3492_349259

theorem equal_area_dividing_line_slope (r : ℝ) (c1 c2 p : ℝ × ℝ) (m : ℝ) : 
  r = 4 ∧ 
  c1 = (0, 20) ∧ 
  c2 = (6, 12) ∧ 
  p = (4, 0) ∧
  (∀ (x y : ℝ), y = m * (x - p.1) + p.2) ∧
  (∀ (x y : ℝ), (x - c1.1)^2 + (y - c1.2)^2 = r^2 → 
    (m * x - y + (p.2 - m * p.1))^2 / (m^2 + 1) = 
    (m * c2.1 - c2.2 + (p.2 - m * p.1))^2 / (m^2 + 1)) →
  |m| = 4/3 := by
sorry

end NUMINAMATH_CALUDE_equal_area_dividing_line_slope_l3492_349259


namespace NUMINAMATH_CALUDE_equation_solution_l3492_349264

theorem equation_solution : 
  ∃! y : ℚ, (7 * y - 2) / (y + 4) - 5 / (y + 4) = 2 / (y + 4) ∧ y = 9 / 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3492_349264


namespace NUMINAMATH_CALUDE_binomial_10_2_l3492_349217

theorem binomial_10_2 : (Nat.choose 10 2) = 45 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_2_l3492_349217


namespace NUMINAMATH_CALUDE_quarter_count_l3492_349279

theorem quarter_count (total : ℕ) (quarters : ℕ) (dimes : ℕ) : 
  total = 77 →
  total = quarters + dimes →
  total - quarters = 48 →
  quarters = 29 := by
sorry

end NUMINAMATH_CALUDE_quarter_count_l3492_349279


namespace NUMINAMATH_CALUDE_cylinder_volume_on_sphere_l3492_349268

theorem cylinder_volume_on_sphere (h : ℝ) (d : ℝ) : 
  h = 1 → d = 2 → 
  let r := Real.sqrt (1^2 - (d/2)^2)
  (π * r^2 * h) = (3*π)/4 := by
sorry

end NUMINAMATH_CALUDE_cylinder_volume_on_sphere_l3492_349268


namespace NUMINAMATH_CALUDE_first_player_wins_l3492_349240

/-- Represents the state of the game -/
structure GameState :=
  (bags : Fin 2008 → ℕ)

/-- The game rules -/
def gameRules (state : GameState) (bagNumber : Fin 2008) (frogsLeft : ℕ) : GameState :=
  { bags := λ i => if i < bagNumber then state.bags i
                   else if i = bagNumber then frogsLeft
                   else min (state.bags i) frogsLeft }

/-- Initial game state -/
def initialState : GameState :=
  { bags := λ _ => 2008 }

/-- Checks if the game is over (only one frog left in bag 1) -/
def isGameOver (state : GameState) : Prop :=
  state.bags 1 = 1 ∧ ∀ i > 1, state.bags i ≤ 1

/-- Theorem: The first player has a winning strategy -/
theorem first_player_wins :
  ∃ (strategy : GameState → Fin 2008 × ℕ),
    ∀ (opponent_move : Fin 2008 × ℕ),
      let (bag, frogs) := strategy initialState
      let state1 := gameRules initialState bag frogs
      let (opponentBag, opponentFrogs) := opponent_move
      let state2 := gameRules state1 opponentBag opponentFrogs
      ¬isGameOver state2 →
        ∃ (next_move : Fin 2008 × ℕ),
          let (nextBag, nextFrogs) := next_move
          let state3 := gameRules state2 nextBag nextFrogs
          isGameOver state3 :=
sorry


end NUMINAMATH_CALUDE_first_player_wins_l3492_349240


namespace NUMINAMATH_CALUDE_parabola_equation_l3492_349206

/-- Represents a parabola with equation y^2 = 2px -/
structure Parabola where
  p : ℝ
  h_pos : p > 0

/-- A point on a parabola -/
structure ParabolaPoint (para : Parabola) where
  x : ℝ
  y : ℝ
  h_on_parabola : y^2 = 2 * para.p * x

theorem parabola_equation (para : Parabola) 
  (point : ParabolaPoint para)
  (h_ordinate : point.y = -4 * Real.sqrt 2)
  (h_distance : point.x + para.p / 2 = 6) :
  para.p = 4 ∨ para.p = 8 :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l3492_349206


namespace NUMINAMATH_CALUDE_max_area_difference_l3492_349265

/-- A rectangle with integer dimensions and perimeter 160 cm -/
structure Rectangle where
  length : ℕ
  width : ℕ
  perimeter_constraint : length + width = 80

/-- The area of a rectangle -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- The theorem stating the maximum difference between areas of two rectangles -/
theorem max_area_difference :
  ∃ (r1 r2 : Rectangle), ∀ (s1 s2 : Rectangle),
    area r1 - area r2 ≥ area s1 - area s2 ∧
    area r1 - area r2 = 1521 := by
  sorry


end NUMINAMATH_CALUDE_max_area_difference_l3492_349265


namespace NUMINAMATH_CALUDE_input_is_input_statement_l3492_349237

-- Define the type for programming language statements
inductive Statement
  | Print
  | Input
  | If
  | Let

-- Define properties for different types of statements
def isPrintStatement (s : Statement) : Prop :=
  s = Statement.Print

def isInputStatement (s : Statement) : Prop :=
  s = Statement.Input

def isConditionalStatement (s : Statement) : Prop :=
  s = Statement.If

theorem input_is_input_statement :
  isPrintStatement Statement.Print →
  isInputStatement Statement.Input →
  isConditionalStatement Statement.If →
  isInputStatement Statement.Input :=
by
  sorry

end NUMINAMATH_CALUDE_input_is_input_statement_l3492_349237


namespace NUMINAMATH_CALUDE_sum_exterior_angles_pentagon_sum_exterior_angles_pentagon_is_360_l3492_349296

/-- The sum of exterior angles of a pentagon is 360 degrees -/
theorem sum_exterior_angles_pentagon : ℝ :=
  360

/-- A pentagon is a polygon with 5 sides -/
def Pentagon : Type := Unit

/-- The sum of exterior angles of a polygon -/
def sum_exterior_angles (p : Pentagon) : ℝ := 360

theorem sum_exterior_angles_pentagon_is_360 (p : Pentagon) :
  sum_exterior_angles p = 360 := by sorry

end NUMINAMATH_CALUDE_sum_exterior_angles_pentagon_sum_exterior_angles_pentagon_is_360_l3492_349296


namespace NUMINAMATH_CALUDE_equation_solution_l3492_349286

theorem equation_solution (x y : ℝ) : y^2 = 4*y - Real.sqrt (x - 3) - 4 → x + 2*y = 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3492_349286


namespace NUMINAMATH_CALUDE_triangle_arithmetic_sides_tangent_product_l3492_349289

/-- 
For a triangle with sides forming an arithmetic sequence, 
the product of 3 and the tangents of half the smallest and largest angles equals 1.
-/
theorem triangle_arithmetic_sides_tangent_product (a b c : ℝ) (α β γ : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →  -- sides are positive
  α > 0 ∧ β > 0 ∧ γ > 0 →  -- angles are positive
  α + β + γ = π →  -- sum of angles in a triangle
  a + c = 2 * b →  -- arithmetic sequence condition
  α ≤ β ∧ β ≤ γ →  -- α is smallest, γ is largest
  3 * Real.tan (α / 2) * Real.tan (γ / 2) = 1 := by
sorry

end NUMINAMATH_CALUDE_triangle_arithmetic_sides_tangent_product_l3492_349289


namespace NUMINAMATH_CALUDE_star_arrangement_count_l3492_349204

/-- The number of distinct arrangements of 12 objects on a regular six-pointed star -/
def star_arrangements : ℕ := 479001600

/-- The number of rotational and reflectional symmetries of a regular six-pointed star -/
def star_symmetries : ℕ := 12

/-- The total number of ways to arrange 12 objects in 12 positions -/
def total_arrangements : ℕ := Nat.factorial 12

theorem star_arrangement_count : 
  star_arrangements = total_arrangements / star_symmetries := by
  sorry

end NUMINAMATH_CALUDE_star_arrangement_count_l3492_349204


namespace NUMINAMATH_CALUDE_alpha_value_l3492_349228

theorem alpha_value (f : ℝ → ℝ) (α : ℝ) 
  (h1 : ∀ x, f x = 4 / (1 - x)) 
  (h2 : f α = 2) : 
  α = -1 := by
sorry

end NUMINAMATH_CALUDE_alpha_value_l3492_349228


namespace NUMINAMATH_CALUDE_garden_perimeter_l3492_349260

/-- 
Given a rectangular garden with one side of length 10 feet and an area of 80 square feet,
prove that the perimeter of the garden is 36 feet.
-/
theorem garden_perimeter : ∀ (width : ℝ), 
  width > 0 →
  10 * width = 80 →
  2 * (10 + width) = 36 := by
  sorry


end NUMINAMATH_CALUDE_garden_perimeter_l3492_349260


namespace NUMINAMATH_CALUDE_c_range_theorem_l3492_349253

-- Define the rectangular prism
def rectangular_prism (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0

-- Define the condition a + b - c = 1
def sum_condition (a b c : ℝ) : Prop :=
  a + b - c = 1

-- Define the condition that the length of the diagonal is 1
def diagonal_condition (a b c : ℝ) : Prop :=
  a^2 + b^2 + c^2 = 1

-- Define the condition a ≠ b
def not_equal_condition (a b : ℝ) : Prop :=
  a ≠ b

-- Theorem statement
theorem c_range_theorem (a b c : ℝ) :
  rectangular_prism a b c →
  sum_condition a b c →
  diagonal_condition a b c →
  not_equal_condition a b →
  0 < c ∧ c < 1/3 := by
  sorry

end NUMINAMATH_CALUDE_c_range_theorem_l3492_349253


namespace NUMINAMATH_CALUDE_smallest_number_l3492_349203

theorem smallest_number (S : Set ℤ) (h : S = {-4, -2, 0, 1}) : 
  ∃ m ∈ S, ∀ n ∈ S, m ≤ n ∧ m = -4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l3492_349203


namespace NUMINAMATH_CALUDE_polynomial_solutions_l3492_349201

def f (a b c d : ℝ) (x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

theorem polynomial_solutions (a b c d : ℝ) :
  (f a b c d 4 = 102) →
  (f a b c d 3 = 102) →
  (f a b c d (-3) = 102) →
  (f a b c d (-4) = 102) →
  ({x : ℝ | f a b c d x = 246} = {0, 5, -5}) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_solutions_l3492_349201


namespace NUMINAMATH_CALUDE_min_marked_cells_l3492_349255

/-- Represents a board with dimensions m × n -/
structure Board (m n : ℕ) where
  cells : Fin m → Fin n → Bool

/-- Represents an L-shaped piece -/
inductive LPiece
| mk : Fin 2 → Fin 2 → LPiece

/-- Checks if an L-piece placed at (i, j) touches a marked cell -/
def touchesMarked (b : Board m n) (p : LPiece) (i : Fin m) (j : Fin n) : Prop :=
  sorry

/-- Checks if a marking strategy ensures all L-piece placements touch a marked cell -/
def validMarking (b : Board m n) : Prop :=
  ∀ (p : LPiece) (i : Fin m) (j : Fin n), touchesMarked b p i j

/-- Counts the number of marked cells on a board -/
def countMarked (b : Board m n) : ℕ :=
  sorry

/-- Theorem stating that 50 is the smallest number of marked cells required -/
theorem min_marked_cells :
  (∃ (b : Board 10 11), validMarking b ∧ countMarked b = 50) ∧
  (∀ (b : Board 10 11), validMarking b → countMarked b ≥ 50) :=
sorry

end NUMINAMATH_CALUDE_min_marked_cells_l3492_349255


namespace NUMINAMATH_CALUDE_age_ratio_in_two_years_l3492_349278

theorem age_ratio_in_two_years (son_age : ℕ) (man_age : ℕ) : 
  son_age = 14 →
  man_age = son_age + 16 →
  ∃ k : ℕ, (man_age + 2) = k * (son_age + 2) →
  (man_age + 2) / (son_age + 2) = 2 := by
sorry

end NUMINAMATH_CALUDE_age_ratio_in_two_years_l3492_349278


namespace NUMINAMATH_CALUDE_frank_breakfast_shopping_cost_l3492_349220

/-- Calculates the total cost of Frank's breakfast shopping --/
def breakfast_shopping_cost (bun_price : ℚ) (bun_quantity : ℕ) (milk_price : ℚ) (milk_quantity : ℕ) (egg_price_multiplier : ℕ) : ℚ :=
  let bun_cost := bun_price * bun_quantity
  let milk_cost := milk_price * milk_quantity
  let egg_cost := milk_price * egg_price_multiplier
  bun_cost + milk_cost + egg_cost

/-- Theorem: The total cost of Frank's breakfast shopping is $11.00 --/
theorem frank_breakfast_shopping_cost :
  breakfast_shopping_cost 0.1 10 2 2 3 = 11 := by
  sorry

end NUMINAMATH_CALUDE_frank_breakfast_shopping_cost_l3492_349220


namespace NUMINAMATH_CALUDE_max_sum_xy_l3492_349261

theorem max_sum_xy (x y a b : ℝ) (hx : x > 0) (hy : y > 0) 
  (ha : 0 ≤ a ∧ a ≤ x) (hb : 0 ≤ b ∧ b ≤ y)
  (h1 : a^2 + y^2 = 2) (h2 : b^2 + x^2 = 1) (h3 : a*x + b*y = 1) :
  x + y ≤ Real.sqrt 5 ∧ ∃ x y, x + y = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_max_sum_xy_l3492_349261


namespace NUMINAMATH_CALUDE_triangle_area_l3492_349267

/-- Given a triangle with perimeter 48 and inradius 2.5, its area is 60 -/
theorem triangle_area (P : ℝ) (r : ℝ) (A : ℝ) 
    (h1 : P = 48) 
    (h2 : r = 2.5) 
    (h3 : A = r * P / 2) : A = 60 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3492_349267


namespace NUMINAMATH_CALUDE_objective_function_range_l3492_349283

/-- The objective function z in terms of x and y -/
def z (x y : ℝ) : ℝ := 3 * x + 2 * y

/-- The constraint function s in terms of x and y -/
def s (x y : ℝ) : ℝ := x + y

theorem objective_function_range :
  ∀ x y : ℝ, x ≥ 0 → y ≥ 0 → 3 ≤ s x y → s x y ≤ 5 →
  9 ≤ z x y ∧ z x y ≤ 15 :=
sorry

end NUMINAMATH_CALUDE_objective_function_range_l3492_349283


namespace NUMINAMATH_CALUDE_max_close_interval_length_l3492_349298

-- Define the functions m and n
def m (x : ℝ) : ℝ := x^2 - 3*x + 4
def n (x : ℝ) : ℝ := 2*x - 3

-- Define the property of being close functions on an interval
def close_functions (f g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → |f x - g x| ≤ 1

-- State the theorem
theorem max_close_interval_length :
  ∃ (a b : ℝ), close_functions m n a b ∧ 
  ∀ (c d : ℝ), close_functions m n c d → d - c ≤ b - a :=
by sorry

end NUMINAMATH_CALUDE_max_close_interval_length_l3492_349298


namespace NUMINAMATH_CALUDE_hallway_width_proof_l3492_349232

/-- Proves that the width of a hallway is 4 feet given the specified conditions -/
theorem hallway_width_proof (total_area : Real) (central_length : Real) (central_width : Real) (hallway_length : Real) :
  total_area = 124 ∧ 
  central_length = 10 ∧ 
  central_width = 10 ∧ 
  hallway_length = 6 → 
  (total_area - central_length * central_width) / hallway_length = 4 := by
sorry

end NUMINAMATH_CALUDE_hallway_width_proof_l3492_349232


namespace NUMINAMATH_CALUDE_mean_squares_sum_l3492_349242

theorem mean_squares_sum (x y z : ℝ) : 
  (x + y + z) / 3 = 10 →
  (x * y * z) ^ (1/3 : ℝ) = 6 →
  3 / (1/x + 1/y + 1/z) = 4 →
  x^2 + y^2 + z^2 = 576 := by
sorry

end NUMINAMATH_CALUDE_mean_squares_sum_l3492_349242


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3492_349266

/-- An arithmetic sequence with its sum sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The arithmetic sequence
  S : ℕ → ℝ  -- The sum sequence
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n : ℕ, S n = (n : ℝ) * (a 1 + a n) / 2

/-- The problem statement -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence)
    (h1 : seq.a 3 = 5)
    (h2 : seq.S 6 = 42) :
  seq.S 9 = 117 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3492_349266


namespace NUMINAMATH_CALUDE_unique_solution_f_f_x_eq_27_l3492_349249

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 27

-- State the theorem
theorem unique_solution_f_f_x_eq_27 :
  ∃! x : ℝ, x ∈ Set.Icc (-3 : ℝ) 5 ∧ f (f x) = 27 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_f_f_x_eq_27_l3492_349249


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_point_l3492_349270

theorem hyperbola_asymptote_point (a : ℝ) (h1 : a > 0) : 
  (∃ (x y : ℝ), x^2/4 - y^2/a = 1 ∧ 
   (y = (Real.sqrt a / 2) * x ∨ y = -(Real.sqrt a / 2) * x) ∧
   x = 2 ∧ y = Real.sqrt 3) → 
  a = 3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_point_l3492_349270


namespace NUMINAMATH_CALUDE_min_value_theorem_l3492_349294

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^2 + b^2 + c^2 + 1 / (a + b + c)^3 ≥ 2 * 3^(2/5) / 3 + 3^(1/3) :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3492_349294


namespace NUMINAMATH_CALUDE_hash_two_three_l3492_349295

-- Define the # operation
def hash (a b : ℕ) : ℕ := a * b - b + b ^ 2

-- Theorem to prove
theorem hash_two_three : hash 2 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_hash_two_three_l3492_349295


namespace NUMINAMATH_CALUDE_equation_solution_l3492_349212

theorem equation_solution : ∃! x : ℝ, (2 / (x + 3) + 3 * x / (x + 3) - 4 / (x + 3) = 4) ∧ x = -14 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3492_349212


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3492_349207

theorem polynomial_factorization (x : ℝ) : x^4 - 64 = (x^2 - 8) * (x^2 + 8) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3492_349207


namespace NUMINAMATH_CALUDE_periodic_function_zeros_l3492_349263

/-- A function f: ℝ → ℝ that is periodic with period 5 and defined as x^2 - 2^x on (-1, 4] -/
def periodic_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (x - 5)) ∧ 
  (∀ x, -1 < x ∧ x ≤ 4 → f x = x^2 - 2^x)

/-- The number of zeros of f on an interval -/
def num_zeros (f : ℝ → ℝ) (a b : ℝ) : ℕ :=
  sorry

theorem periodic_function_zeros (f : ℝ → ℝ) (h : periodic_function f) :
  num_zeros f 0 2013 = 1207 :=
sorry

end NUMINAMATH_CALUDE_periodic_function_zeros_l3492_349263


namespace NUMINAMATH_CALUDE_germination_rate_1000_estimated_probability_close_germinable_seeds_weight_l3492_349287

/-- Represents the germination data for a batch of seeds -/
structure GerminationData where
  seeds : ℕ
  germinations : ℕ

/-- The germination experiment data -/
def experimentData : List GerminationData := [
  ⟨100, 94⟩,
  ⟨500, 442⟩,
  ⟨800, 728⟩,
  ⟨1000, 902⟩,
  ⟨2000, 1798⟩,
  ⟨5000, 4505⟩
]

/-- Calculates the germination rate for a given GerminationData -/
def germinationRate (data : GerminationData) : ℚ :=
  data.germinations / data.seeds

/-- Theorem stating the germination rate for 1000 seeds -/
theorem germination_rate_1000 :
  ∃ data ∈ experimentData, data.seeds = 1000 ∧ germinationRate data = 902 / 1000 := by sorry

/-- Estimated germination probability -/
def estimatedProbability : ℚ := 9 / 10

/-- Theorem stating the estimated germination probability is close to actual rates -/
theorem estimated_probability_close :
  ∀ data ∈ experimentData, abs (germinationRate data - estimatedProbability) < 1 / 10 := by sorry

/-- Theorem calculating the weight of germinable seeds in 10 kg -/
theorem germinable_seeds_weight (totalWeight : ℚ) :
  totalWeight * estimatedProbability = 9 / 10 * totalWeight := by sorry

end NUMINAMATH_CALUDE_germination_rate_1000_estimated_probability_close_germinable_seeds_weight_l3492_349287


namespace NUMINAMATH_CALUDE_burger_meal_cost_l3492_349256

theorem burger_meal_cost (burger_cost soda_cost : ℝ) : 
  soda_cost = (1/3) * burger_cost →
  burger_cost + soda_cost + 2 * (burger_cost + soda_cost) = 24 →
  burger_cost = 6 := by
sorry

end NUMINAMATH_CALUDE_burger_meal_cost_l3492_349256


namespace NUMINAMATH_CALUDE_wheel_configuration_theorem_l3492_349248

/-- Represents a configuration of wheels with spokes -/
structure WheelConfiguration where
  num_wheels : ℕ
  max_spokes_per_wheel : ℕ
  total_visible_spokes : ℕ

/-- Checks if a given wheel configuration is possible -/
def is_possible_configuration (config : WheelConfiguration) : Prop :=
  config.num_wheels * config.max_spokes_per_wheel ≥ config.total_visible_spokes

/-- Theorem stating the possibility of 3 wheels and impossibility of 2 wheels -/
theorem wheel_configuration_theorem :
  let config_3 : WheelConfiguration := ⟨3, 3, 7⟩
  let config_2 : WheelConfiguration := ⟨2, 3, 7⟩
  is_possible_configuration config_3 ∧ ¬is_possible_configuration config_2 := by
  sorry

#check wheel_configuration_theorem

end NUMINAMATH_CALUDE_wheel_configuration_theorem_l3492_349248


namespace NUMINAMATH_CALUDE_inscribed_circle_length_equals_arc_length_l3492_349244

/-- Given a circular arc of 120° with radius R and an inscribed circle with radius r 
    tangent to the arc and the tangent lines drawn at the arc's endpoints, 
    the circumference of the inscribed circle (2πr) is equal to the length of the original 120° arc. -/
theorem inscribed_circle_length_equals_arc_length (R r : ℝ) : 
  R > 0 → r > 0 → r = R / 2 → 2 * π * r = 2 * π * R * (1/3) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_length_equals_arc_length_l3492_349244


namespace NUMINAMATH_CALUDE_units_digit_of_n_l3492_349210

theorem units_digit_of_n (m n : ℕ) : 
  m * n = 31^6 ∧ m % 10 = 3 → n % 10 = 7 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_n_l3492_349210


namespace NUMINAMATH_CALUDE_coefficient_x3y2_eq_neg_ten_l3492_349297

/-- The coefficient of x^3 * y^2 in the expansion of (x^2 - x + y)^5 -/
def coefficient_x3y2 : ℤ :=
  (-1) * (Nat.choose 5 3)

theorem coefficient_x3y2_eq_neg_ten : coefficient_x3y2 = -10 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x3y2_eq_neg_ten_l3492_349297


namespace NUMINAMATH_CALUDE_unsold_tomatoes_l3492_349274

def total_harvested : ℝ := 245.5
def sold_to_maxwell : ℝ := 125.5
def sold_to_wilson : ℝ := 78

theorem unsold_tomatoes : 
  total_harvested - (sold_to_maxwell + sold_to_wilson) = 42 := by
  sorry

end NUMINAMATH_CALUDE_unsold_tomatoes_l3492_349274


namespace NUMINAMATH_CALUDE_polygon_interior_exterior_angle_sum_l3492_349250

theorem polygon_interior_exterior_angle_sum (n : ℕ) : 
  (n ≥ 3) → (((n - 2) * 180 = 2 * 360) ↔ n = 6) := by
  sorry

end NUMINAMATH_CALUDE_polygon_interior_exterior_angle_sum_l3492_349250


namespace NUMINAMATH_CALUDE_min_value_problem_l3492_349222

theorem min_value_problem (a b c d e f g h : ℝ) 
  (h1 : a * b * c * d = 16) 
  (h2 : e * f * g * h = 1) : 
  (a * f)^2 + (b * e)^2 + (c * h)^2 + (d * g)^2 ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l3492_349222


namespace NUMINAMATH_CALUDE_winnie_balloons_distribution_l3492_349200

/-- Represents the number of balloons Winnie has left after distribution -/
def balloonsLeft (red white green chartreuse friends : ℕ) : ℕ :=
  (red + white + green + chartreuse) % friends

/-- Proves that Winnie has no balloons left after distribution -/
theorem winnie_balloons_distribution 
  (red : ℕ) (white : ℕ) (green : ℕ) (chartreuse : ℕ) (friends : ℕ)
  (h_red : red = 24)
  (h_white : white = 36)
  (h_green : green = 70)
  (h_chartreuse : chartreuse = 90)
  (h_friends : friends = 10) :
  balloonsLeft red white green chartreuse friends = 0 := by
  sorry

#eval balloonsLeft 24 36 70 90 10

end NUMINAMATH_CALUDE_winnie_balloons_distribution_l3492_349200


namespace NUMINAMATH_CALUDE_marble_probability_l3492_349272

theorem marble_probability (total : ℕ) (p_white p_green : ℚ) :
  p_white = 1/4 →
  p_green = 2/7 →
  p_white + p_green + (1 - p_white - p_green) = 1 →
  1 - p_white - p_green = 13/28 :=
by
  sorry

end NUMINAMATH_CALUDE_marble_probability_l3492_349272


namespace NUMINAMATH_CALUDE_right_triangle_area_l3492_349225

theorem right_triangle_area (base height hypotenuse : ℝ) :
  base = 12 →
  hypotenuse = 13 →
  base^2 + height^2 = hypotenuse^2 →
  (1/2) * base * height = 30 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3492_349225


namespace NUMINAMATH_CALUDE_polynomial_nonzero_coeffs_l3492_349223

/-- A polynomial has at least n+1 nonzero coefficients if its degree is at least n -/
def HasAtLeastNPlusOneNonzeroCoeffs (p : Polynomial ℝ) (n : ℕ) : Prop :=
  (Finset.filter (· ≠ 0) p.support).card ≥ n + 1

/-- The main theorem statement -/
theorem polynomial_nonzero_coeffs
  (a : ℝ) (k : ℕ) (Q : Polynomial ℝ) 
  (ha : a ≠ 0) (hQ : Q ≠ 0) :
  let W := (Polynomial.X - Polynomial.C a)^k * Q
  HasAtLeastNPlusOneNonzeroCoeffs W k := by
sorry

end NUMINAMATH_CALUDE_polynomial_nonzero_coeffs_l3492_349223


namespace NUMINAMATH_CALUDE_added_value_proof_l3492_349262

theorem added_value_proof (N V : ℚ) : 
  N = 1280 → (N + V) / 125 = 7392 / 462 → V = 720 := by sorry

end NUMINAMATH_CALUDE_added_value_proof_l3492_349262


namespace NUMINAMATH_CALUDE_coefficient_x4_is_80_l3492_349252

/-- The coefficient of x^4 in the expansion of (4x^2-2x+1)(2x+1)^5 -/
def coefficient_x4 : ℕ :=
  -- Define the coefficient here
  sorry

/-- Theorem stating that the coefficient of x^4 is 80 -/
theorem coefficient_x4_is_80 : coefficient_x4 = 80 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x4_is_80_l3492_349252


namespace NUMINAMATH_CALUDE_simplify_expression_l3492_349239

variable (R : Type*) [Ring R]
variable (a b c : R)

theorem simplify_expression :
  (12 * a + 35 * b + 17 * c) + (13 * a - 15 * b + 8 * c) - (8 * a + 28 * b - 25 * c) =
  17 * a - 8 * b + 50 * c := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3492_349239


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3492_349290

/-- The sum of the first n terms of a geometric sequence -/
def geometric_sum (a₀ r : ℚ) (n : ℕ) : ℚ :=
  a₀ * (1 - r^n) / (1 - r)

/-- The sum of the first five terms of the specific geometric sequence -/
def specific_sum : ℚ := geometric_sum (1/3) (1/3) 5

theorem geometric_sequence_sum :
  specific_sum = 121/243 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3492_349290


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3492_349229

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  a 1 + a 4 + a 10 + a 16 + a 19 = 150 →
  a 20 - a 26 + a 16 = 30 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3492_349229


namespace NUMINAMATH_CALUDE_binomial_8_4_l3492_349216

theorem binomial_8_4 : (8 : ℕ).choose 4 = 70 := by sorry

end NUMINAMATH_CALUDE_binomial_8_4_l3492_349216


namespace NUMINAMATH_CALUDE_no_real_solutions_existence_of_zero_product_ellipses_same_foci_l3492_349213

-- Statement 1
theorem no_real_solutions : ∀ x : ℝ, x^2 - 3*x + 3 ≠ 0 := by sorry

-- Statement 2
theorem existence_of_zero_product : ∃ x y : ℝ, x * y = 0 ∧ x ≠ 0 ∧ y ≠ 0 := by sorry

-- Statement 3
def ellipse1 (x y : ℝ) : Prop := x^2/25 + y^2/9 = 1

def ellipse2 (k x y : ℝ) : Prop := x^2/(25-k) + y^2/(9-k) = 1

def has_same_foci (k : ℝ) : Prop :=
  ∃ a b : ℝ, (∀ x y : ℝ, ellipse1 x y ↔ (x-a)^2/25 + (x+a)^2/25 + y^2/9 = 1) ∧
             (∀ x y : ℝ, ellipse2 k x y ↔ (x-b)^2/(25-k) + (x+b)^2/(25-k) + y^2/(9-k) = 1) ∧
             a = b

theorem ellipses_same_foci : ∀ k : ℝ, 9 < k → k < 25 → has_same_foci k := by sorry

end NUMINAMATH_CALUDE_no_real_solutions_existence_of_zero_product_ellipses_same_foci_l3492_349213


namespace NUMINAMATH_CALUDE_unique_monic_polynomial_l3492_349205

/-- A monic polynomial of degree 3 satisfying f(0) = 3 and f(2) = 19 -/
def f : ℝ → ℝ :=
  fun x ↦ x^3 + x^2 + 2*x + 3

/-- Theorem stating that f is the unique monic polynomial of degree 3 satisfying the given conditions -/
theorem unique_monic_polynomial :
  (∀ x, f x = x^3 + x^2 + 2*x + 3) ∧
  (∀ p : ℝ → ℝ, (∃ a b c : ℝ, ∀ x, p x = x^3 + a*x^2 + b*x + c) →
    p 0 = 3 → p 2 = 19 → p = f) := by
  sorry

end NUMINAMATH_CALUDE_unique_monic_polynomial_l3492_349205


namespace NUMINAMATH_CALUDE_rhombus_area_l3492_349235

/-- The area of a rhombus with side length 2 and an angle of 45 degrees between adjacent sides is 2√2 -/
theorem rhombus_area (s : ℝ) (θ : ℝ) (h1 : s = 2) (h2 : θ = π / 4) :
  s * s * Real.sin θ = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l3492_349235


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l3492_349288

theorem least_addition_for_divisibility : 
  ∃ (x : ℕ), x > 0 ∧ (1049 + x) % 25 = 0 ∧ ∀ (y : ℕ), y > 0 ∧ (1049 + y) % 25 = 0 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l3492_349288


namespace NUMINAMATH_CALUDE_exponent_unchanged_l3492_349292

/-- Represents a term in an algebraic expression -/
structure Term where
  coefficient : ℝ
  letter : Char
  exponent : ℕ

/-- Combines two like terms -/
def combineLikeTerms (t1 t2 : Term) : Term :=
  { coefficient := t1.coefficient + t2.coefficient,
    letter := t1.letter,
    exponent := t1.exponent }

/-- Theorem stating that the exponent remains unchanged when combining like terms -/
theorem exponent_unchanged (t1 t2 : Term) (h : t1.letter = t2.letter) :
  (combineLikeTerms t1 t2).exponent = t1.exponent :=
by sorry

end NUMINAMATH_CALUDE_exponent_unchanged_l3492_349292


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l3492_349273

/-- Given a line L1 with equation 2x + y - 1 = 0 and a point P (-1, 2),
    prove that the line L2 passing through P and perpendicular to L1
    has the equation x - 2y + 5 = 0 -/
theorem perpendicular_line_equation (L1 : Set (ℝ × ℝ)) (P : ℝ × ℝ) :
  L1 = {(x, y) | 2 * x + y - 1 = 0} →
  P = (-1, 2) →
  ∃ L2 : Set (ℝ × ℝ),
    (P ∈ L2) ∧
    (∀ (p q : ℝ × ℝ), p ∈ L1 → q ∈ L1 → p ≠ q →
      ∀ (r s : ℝ × ℝ), r ∈ L2 → s ∈ L2 → r ≠ s →
        (p.1 - q.1) * (r.1 - s.1) + (p.2 - q.2) * (r.2 - s.2) = 0) ∧
    L2 = {(x, y) | x - 2 * y + 5 = 0} :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l3492_349273


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3492_349251

-- Define the inequality
def inequality (x m : ℝ) : Prop := x^2 - (2*m-1)*x + m^2 - m > 0

-- Define the solution set
def solution_set (m : ℝ) : Set ℝ := {x | x < m-1 ∨ x > m}

-- Theorem statement
theorem inequality_solution_set (m : ℝ) : 
  {x : ℝ | inequality x m} = solution_set m := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3492_349251


namespace NUMINAMATH_CALUDE_f_monotonic_increasing_interval_l3492_349299

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 1)

theorem f_monotonic_increasing_interval :
  ∀ x y, 1 < x ∧ x < y → f x < f y :=
by sorry

end NUMINAMATH_CALUDE_f_monotonic_increasing_interval_l3492_349299


namespace NUMINAMATH_CALUDE_fraction_sum_bound_l3492_349238

theorem fraction_sum_bound (a b c : ℕ+) (h : (a : ℝ)⁻¹ + (b : ℝ)⁻¹ + (c : ℝ)⁻¹ < 1) :
  (a : ℝ)⁻¹ + (b : ℝ)⁻¹ + (c : ℝ)⁻¹ ≤ 41 / 42 ∧
  ∃ (x y z : ℕ+), (x : ℝ)⁻¹ + (y : ℝ)⁻¹ + (z : ℝ)⁻¹ = 41 / 42 :=
sorry

#check fraction_sum_bound

end NUMINAMATH_CALUDE_fraction_sum_bound_l3492_349238


namespace NUMINAMATH_CALUDE_susan_bob_cat_difference_l3492_349224

/-- Proves that Susan has 8 more cats than Bob after all exchanges -/
theorem susan_bob_cat_difference :
  let susan_initial : ℕ := 21
  let bob_initial : ℕ := 3
  let susan_received : ℕ := 5
  let bob_received : ℕ := 7
  let susan_gave : ℕ := 4
  let susan_final := susan_initial + susan_received - susan_gave
  let bob_final := bob_initial + bob_received + susan_gave
  susan_final - bob_final = 8 := by
  sorry

end NUMINAMATH_CALUDE_susan_bob_cat_difference_l3492_349224


namespace NUMINAMATH_CALUDE_total_distance_driven_l3492_349221

theorem total_distance_driven (initial_speed initial_time : ℝ) : 
  initial_speed = 30 ∧ initial_time = 0.5 →
  (initial_speed * initial_time) + (2 * initial_speed * (2 * initial_time)) = 75 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_driven_l3492_349221
