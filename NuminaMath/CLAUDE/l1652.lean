import Mathlib

namespace q_investment_proof_l1652_165233

/-- Calculates the investment of Q given the investment of P, total profit, and Q's profit share --/
def calculate_q_investment (p_investment : ℚ) (total_profit : ℚ) (q_profit_share : ℚ) : ℚ :=
  (p_investment * q_profit_share) / (total_profit - q_profit_share)

theorem q_investment_proof (p_investment : ℚ) (total_profit : ℚ) (q_profit_share : ℚ) 
  (h1 : p_investment = 54000)
  (h2 : total_profit = 18000)
  (h3 : q_profit_share = 6001.89) :
  calculate_q_investment p_investment total_profit q_profit_share = 27010 := by
  sorry

#eval calculate_q_investment 54000 18000 6001.89

end q_investment_proof_l1652_165233


namespace ball_distribution_with_constraint_l1652_165201

theorem ball_distribution_with_constraint (n : ℕ) (k : ℕ) (m : ℕ) :
  n = 5 → k = 3 → m = 2 →
  (n.pow k : ℕ) - (k - 1).pow n - n * (k - 1).pow (n - 1) = 131 :=
sorry

end ball_distribution_with_constraint_l1652_165201


namespace fraction_evaluation_l1652_165249

theorem fraction_evaluation (x y : ℝ) (h : x ≠ y) :
  (x^6 - y^6) / (x^3 - y^3) = x^3 + y^3 := by
sorry

end fraction_evaluation_l1652_165249


namespace ceiling_sqrt_250_l1652_165207

theorem ceiling_sqrt_250 : ⌈Real.sqrt 250⌉ = 16 := by sorry

end ceiling_sqrt_250_l1652_165207


namespace quadratic_inequality_and_square_property_l1652_165235

theorem quadratic_inequality_and_square_property : 
  (¬∃ x : ℝ, x^2 - x + 2 < 0) ∧ (∀ x ∈ Set.Icc 1 2, x^2 ≥ 1) := by
  sorry

end quadratic_inequality_and_square_property_l1652_165235


namespace cherry_sweets_count_l1652_165241

/-- The number of cherry-flavored sweets initially in the packet -/
def initial_cherry : ℕ := 30

/-- The number of strawberry-flavored sweets initially in the packet -/
def initial_strawberry : ℕ := 40

/-- The number of pineapple-flavored sweets initially in the packet -/
def initial_pineapple : ℕ := 50

/-- The number of cherry-flavored sweets Aaron gives to his friend -/
def given_away : ℕ := 5

/-- The total number of sweets left in the packet after Aaron's actions -/
def remaining_total : ℕ := 55

theorem cherry_sweets_count :
  initial_cherry = 30 ∧
  (initial_cherry / 2 - given_away) + (initial_strawberry / 2) + (initial_pineapple / 2) = remaining_total :=
by sorry

end cherry_sweets_count_l1652_165241


namespace comic_book_pages_l1652_165203

/-- Given that Trevor drew 220 pages in total over three months,
    and the third month's issue was four pages longer than the others,
    prove that the first issue had 72 pages. -/
theorem comic_book_pages :
  ∀ (x : ℕ),
  (x + x + (x + 4) = 220) →
  (x = 72) :=
by sorry

end comic_book_pages_l1652_165203


namespace evaluate_polynomial_l1652_165227

theorem evaluate_polynomial (y : ℝ) (h : y = 2) : y^4 + y^3 + y^2 + y + 1 = 31 := by
  sorry

end evaluate_polynomial_l1652_165227


namespace james_total_distance_l1652_165225

/-- Calculates the total distance driven given a series of driving segments -/
def total_distance (speeds : List ℝ) (times : List ℝ) : ℝ :=
  List.sum (List.zipWith (· * ·) speeds times)

/-- The total distance James drove under the given conditions -/
theorem james_total_distance :
  let speeds : List ℝ := [30, 60, 75, 60, 70]
  let times : List ℝ := [0.5, 0.75, 1.5, 2, 4]
  total_distance speeds times = 572.5 := by
  sorry

#check james_total_distance

end james_total_distance_l1652_165225


namespace johns_money_l1652_165280

/-- Given three people with a total of $67, where one has $5 less than the second,
    and the third has 4 times more than the second, prove that the third person has $48. -/
theorem johns_money (total : ℕ) (alis_money nadas_money johns_money : ℕ) : 
  total = 67 →
  alis_money = nadas_money - 5 →
  johns_money = 4 * nadas_money →
  alis_money + nadas_money + johns_money = total →
  johns_money = 48 := by
  sorry

end johns_money_l1652_165280


namespace ratio_q_p_l1652_165270

def total_slips : ℕ := 50
def distinct_numbers : ℕ := 10
def slips_per_number : ℕ := 5
def drawn_slips : ℕ := 5

def p : ℚ := (distinct_numbers : ℚ) / (Nat.choose total_slips drawn_slips : ℚ)

def q : ℚ := (Nat.choose distinct_numbers 2 * Nat.choose slips_per_number 2 * Nat.choose slips_per_number 2 * (distinct_numbers - 2) : ℚ) / (Nat.choose total_slips drawn_slips : ℚ)

theorem ratio_q_p : q / p = 360 := by sorry

end ratio_q_p_l1652_165270


namespace shortest_piece_length_l1652_165282

theorem shortest_piece_length (total_length : ℝ) (piece1 piece2 piece3 : ℝ) : 
  total_length = 138 →
  piece1 + piece2 + piece3 = total_length →
  piece1 = 2 * piece2 →
  piece2 = 3 * piece3 →
  piece3 = 13.8 := by
sorry

end shortest_piece_length_l1652_165282


namespace fixed_point_of_function_l1652_165278

theorem fixed_point_of_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f := fun x : ℝ => a^(1 - x) - 2
  f 1 = -1 := by sorry

end fixed_point_of_function_l1652_165278


namespace farm_bulls_count_l1652_165217

/-- Given a farm with cattle and a cow-to-bull ratio, calculates the number of bulls -/
def calculate_bulls (total_cattle : ℕ) (cow_ratio : ℕ) (bull_ratio : ℕ) : ℕ :=
  (total_cattle * bull_ratio) / (cow_ratio + bull_ratio)

/-- Theorem: On a farm with 555 cattle and a cow-to-bull ratio of 10:27, there are 405 bulls -/
theorem farm_bulls_count : calculate_bulls 555 10 27 = 405 := by
  sorry

end farm_bulls_count_l1652_165217


namespace quadratic_equation_root_l1652_165268

theorem quadratic_equation_root (k : ℝ) : 
  (∃ x : ℂ, x^2 + 4*x + k = 0 ∧ x = -2 + 3*Complex.I) → k = 13 := by
  sorry

end quadratic_equation_root_l1652_165268


namespace second_round_score_l1652_165204

/-- Represents the number of darts thrown in each round -/
def darts_per_round : ℕ := 8

/-- Represents the minimum points per dart -/
def min_points_per_dart : ℕ := 3

/-- Represents the maximum points per dart -/
def max_points_per_dart : ℕ := 9

/-- Represents the points scored in the first round -/
def first_round_points : ℕ := 24

/-- Represents the ratio of points scored in the second round compared to the first round -/
def second_round_ratio : ℚ := 2

/-- Represents the ratio of points scored in the third round compared to the second round -/
def third_round_ratio : ℚ := (3/2 : ℚ)

/-- Theorem stating that Misha scored 48 points in the second round -/
theorem second_round_score : 
  first_round_points * second_round_ratio = 48 := by sorry

end second_round_score_l1652_165204


namespace equal_money_days_l1652_165210

/-- The daily interest rate when leaving money with mother -/
def mother_rate : ℕ := 300

/-- The daily interest rate when leaving money with father -/
def father_rate : ℕ := 500

/-- The initial amount Kyu-won gave to her mother -/
def kyu_won_initial : ℕ := 8000

/-- The initial amount Seok-gi left with his father -/
def seok_gi_initial : ℕ := 5000

/-- The number of days needed for Kyu-won and Seok-gi to have the same amount of money -/
def days_needed : ℕ := 15

theorem equal_money_days :
  kyu_won_initial + mother_rate * days_needed = seok_gi_initial + father_rate * days_needed :=
by sorry

end equal_money_days_l1652_165210


namespace last_week_viewers_correct_l1652_165283

/-- The number of people who watched the baseball games last week -/
def last_week_viewers : ℕ := 200

/-- The number of people who watched the second game this week -/
def second_game_viewers : ℕ := 80

/-- The number of people who watched the first game this week -/
def first_game_viewers : ℕ := second_game_viewers - 20

/-- The number of people who watched the third game this week -/
def third_game_viewers : ℕ := second_game_viewers + 15

/-- The total number of people who watched the games this week -/
def this_week_total : ℕ := first_game_viewers + second_game_viewers + third_game_viewers

/-- The difference in viewers between this week and last week -/
def viewer_difference : ℕ := 35

theorem last_week_viewers_correct : 
  last_week_viewers = this_week_total - viewer_difference := by
  sorry

end last_week_viewers_correct_l1652_165283


namespace x_value_when_y_is_two_l1652_165240

theorem x_value_when_y_is_two (x y : ℚ) : 
  y = 1 / (4 * x + 2) → y = 2 → x = -3/8 := by
sorry

end x_value_when_y_is_two_l1652_165240


namespace rectangular_solid_volume_l1652_165218

/-- Given a rectangular solid with side areas 3, 5, and 15 sharing a common vertex,
    its volume is 15. -/
theorem rectangular_solid_volume (a b c : ℝ) (h1 : a * b = 3) (h2 : a * c = 5) (h3 : b * c = 15) :
  a * b * c = 15 := by
  sorry

end rectangular_solid_volume_l1652_165218


namespace min_dominoes_to_win_viktors_winning_strategy_l1652_165277

/-- Represents a square board -/
structure Board :=
  (size : ℕ)

/-- Represents a domino placement on the board -/
structure DominoPlacement :=
  (board : Board)
  (num_dominoes : ℕ)

/-- Theorem: The minimum number of dominoes Viktor needs to fix to win -/
theorem min_dominoes_to_win (b : Board) (d : DominoPlacement) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem viktors_winning_strategy (b : Board) (d : DominoPlacement) : 
  b.size = 2022 → d.board = b → d.num_dominoes = 2022 * 2022 / 2 → 
  min_dominoes_to_win b d = 1011^2 :=
sorry

end min_dominoes_to_win_viktors_winning_strategy_l1652_165277


namespace larger_number_problem_l1652_165279

theorem larger_number_problem (L S : ℕ) (h1 : L > S) (h2 : L - S = 1355) (h3 : L = 6 * S + 15) : L = 1623 := by
  sorry

end larger_number_problem_l1652_165279


namespace cos_72_minus_cos_144_l1652_165232

/-- Proves that the difference between cosine of 72 degrees and cosine of 144 degrees is 1/2 -/
theorem cos_72_minus_cos_144 : Real.cos (72 * π / 180) - Real.cos (144 * π / 180) = 1/2 := by
  sorry

end cos_72_minus_cos_144_l1652_165232


namespace f_monotone_decreasing_l1652_165265

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + m * x + 3

-- State the theorem
theorem f_monotone_decreasing (m : ℝ) :
  (∀ x : ℝ, f m x = f m (-x)) →  -- f is even
  ∀ x : ℝ, x > 0 → ∀ y : ℝ, y > x → f m y < f m x :=
by sorry

end f_monotone_decreasing_l1652_165265


namespace smallest_next_divisor_after_523_l1652_165215

def is_5digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

theorem smallest_next_divisor_after_523 (m : ℕ) (h1 : is_5digit m) (h2 : Even m) (h3 : m % 523 = 0) :
  ∃ (d : ℕ), d ∣ m ∧ d > 523 ∧ (∀ (x : ℕ), x ∣ m → x > 523 → x ≥ d) ∧ d = 524 :=
sorry

end smallest_next_divisor_after_523_l1652_165215


namespace x_plus_y_equals_483_l1652_165223

theorem x_plus_y_equals_483 (x y : ℝ) : 
  x = 300 * (1 - 0.3) → 
  y = x * (1 + 0.3) → 
  x + y = 483 := by
sorry

end x_plus_y_equals_483_l1652_165223


namespace foal_count_l1652_165298

def animal_count : ℕ := 11
def leg_count : ℕ := 30
def turkey_legs : ℕ := 2
def foal_legs : ℕ := 4

theorem foal_count (t f : ℕ) : 
  t + f = animal_count → 
  turkey_legs * t + foal_legs * f = leg_count → 
  f = 4 :=
by
  sorry

end foal_count_l1652_165298


namespace tan_sum_of_roots_l1652_165255

theorem tan_sum_of_roots (α β : ℝ) : 
  (∃ x y : ℝ, x^2 - 3*x + 2 = 0 ∧ y^2 - 3*y + 2 = 0 ∧ x = Real.tan α ∧ y = Real.tan β) →
  Real.tan (α + β) = -3 := by
sorry

end tan_sum_of_roots_l1652_165255


namespace hospital_bill_ambulance_cost_l1652_165289

theorem hospital_bill_ambulance_cost 
  (total_bill : ℝ)
  (medication_percentage : ℝ)
  (overnight_percentage : ℝ)
  (food_cost : ℝ)
  (h1 : total_bill = 5000)
  (h2 : medication_percentage = 0.5)
  (h3 : overnight_percentage = 0.25)
  (h4 : food_cost = 175) :
  let medication_cost := medication_percentage * total_bill
  let remaining_after_medication := total_bill - medication_cost
  let overnight_cost := overnight_percentage * remaining_after_medication
  let remaining_after_overnight := remaining_after_medication - overnight_cost
  let ambulance_cost := remaining_after_overnight - food_cost
  ambulance_cost = 1700 := by sorry

end hospital_bill_ambulance_cost_l1652_165289


namespace problem_statement_l1652_165263

/-- The repeating decimal 0.8̄ -/
def repeating_decimal : ℚ := 8/9

/-- The problem statement -/
theorem problem_statement : 2 - repeating_decimal = 10/9 := by
  sorry

end problem_statement_l1652_165263


namespace democrat_ratio_l1652_165213

theorem democrat_ratio (total_participants : ℕ) (female_democrats : ℕ) 
  (h1 : total_participants = 810)
  (h2 : female_democrats = 135)
  (h3 : female_democrats * 2 ≤ total_participants)
  (h4 : total_participants / 3 = female_democrats + (total_participants - female_democrats * 2) / 4) :
  (total_participants - female_democrats * 2) / 4 = (total_participants - female_democrats * 2) / 4 :=
by sorry

#check democrat_ratio

end democrat_ratio_l1652_165213


namespace max_cubes_for_given_prism_l1652_165275

/-- Represents the dimensions and properties of a wooden rectangular prism --/
structure WoodenPrism where
  totalSurfaceArea : ℝ
  cubeSurfaceArea : ℝ
  wastePerCut : ℝ

/-- Calculates the maximum number of cubes that can be sawed from the prism --/
def maxCubes (prism : WoodenPrism) : ℕ :=
  sorry

/-- Theorem stating the maximum number of cubes for the given problem --/
theorem max_cubes_for_given_prism :
  let prism : WoodenPrism := {
    totalSurfaceArea := 2448,
    cubeSurfaceArea := 216,
    wastePerCut := 0.2
  }
  maxCubes prism = 15 := by
  sorry

end max_cubes_for_given_prism_l1652_165275


namespace cone_surface_area_ratio_l1652_165257

/-- For a cone whose lateral surface unfolds into a sector with a central angle of 120° and radius 1,
    the ratio of its surface area to its lateral surface area is 4:3 -/
theorem cone_surface_area_ratio :
  let sector_angle : Real := 120 * π / 180
  let sector_radius : Real := 1
  let lateral_surface_area : Real := π * sector_radius^2 * (sector_angle / (2 * π))
  let base_radius : Real := sector_radius * sector_angle / (2 * π)
  let base_area : Real := π * base_radius^2
  let surface_area : Real := lateral_surface_area + base_area
  (surface_area / lateral_surface_area) = 4/3 := by
sorry

end cone_surface_area_ratio_l1652_165257


namespace f_of_4_equals_15_l1652_165299

/-- A function f(x) = cx^2 + dx + 3 satisfying f(1) = 3 and f(2) = 5 -/
def f (c d : ℝ) (x : ℝ) : ℝ := c * x^2 + d * x + 3

/-- The theorem stating that f(4) = 15 given the conditions -/
theorem f_of_4_equals_15 (c d : ℝ) :
  f c d 1 = 3 → f c d 2 = 5 → f c d 4 = 15 := by
  sorry

#check f_of_4_equals_15

end f_of_4_equals_15_l1652_165299


namespace intersection_A_complement_B_l1652_165200

-- Define the universal set U
def U : Set Nat := {0, 1, 2, 3, 4, 5}

-- Define set A
def A : Set Nat := {0, 2, 4}

-- Define set B
def B : Set Nat := {0, 5}

-- Theorem statement
theorem intersection_A_complement_B :
  A ∩ (U \ B) = {2, 4} := by sorry

end intersection_A_complement_B_l1652_165200


namespace birds_and_storks_l1652_165295

theorem birds_and_storks (initial_birds : ℕ) (storks : ℕ) (additional_birds : ℕ) : 
  initial_birds = 3 → storks = 4 → additional_birds = 2 →
  (initial_birds + additional_birds) - storks = 1 := by
  sorry

end birds_and_storks_l1652_165295


namespace pentagon_cannot_tessellate_l1652_165292

/-- A regular polygon can tessellate a plane if its internal angle divides 360° evenly -/
def can_tessellate (internal_angle : ℝ) : Prop :=
  ∃ n : ℕ, n * internal_angle = 360

/-- The internal angle of a regular pentagon is 108° -/
def pentagon_internal_angle : ℝ := 108

/-- Theorem: A regular pentagon cannot tessellate a plane by itself -/
theorem pentagon_cannot_tessellate :
  ¬(can_tessellate pentagon_internal_angle) :=
sorry

end pentagon_cannot_tessellate_l1652_165292


namespace equation1_representation_equation2_representation_l1652_165288

-- Define the equations
def equation1 (x y : ℝ) : Prop := 4 * x^2 + 8 * y^2 + 8 * y * |y| = 1
def equation2 (x y : ℝ) : Prop := 2 * x^2 - 4 * x + 2 + 2 * (x - 1) * |x - 1| + 8 * y^2 - 8 * y * |y| = 1

-- Define the regions for equation1
def upper_ellipse (x y : ℝ) : Prop := y ≥ 0 ∧ 4 * x^2 + 16 * y^2 = 1
def vertical_lines (x y : ℝ) : Prop := y < 0 ∧ (x = 1/2 ∨ x = -1/2)

-- Define the regions for equation2
def elliptic_part (x y : ℝ) : Prop := x ≥ 1 ∧ 4 * (x - 1)^2 + 16 * y^2 = 1
def vertical_section (x y : ℝ) : Prop := x < 1 ∧ y = -1/4

-- Theorem statements
theorem equation1_representation :
  ∀ x y : ℝ, equation1 x y ↔ (upper_ellipse x y ∨ vertical_lines x y) :=
sorry

theorem equation2_representation :
  ∀ x y : ℝ, equation2 x y ↔ (elliptic_part x y ∨ vertical_section x y) :=
sorry

end equation1_representation_equation2_representation_l1652_165288


namespace slope_determines_y_coordinate_l1652_165237

/-- Given two points P and Q, if the slope of the line passing through them is 1/4,
    then the y-coordinate of Q is -3. -/
theorem slope_determines_y_coordinate 
  (x_P y_P x_Q : ℝ) (slope : ℝ) :
  x_P = -3 →
  y_P = -5 →
  x_Q = 5 →
  slope = 1/4 →
  (y_Q - y_P) / (x_Q - x_P) = slope →
  y_Q = -3 :=
by sorry

end slope_determines_y_coordinate_l1652_165237


namespace wreath_distribution_l1652_165256

/-- The number of wreaths each Greek initially had -/
def wreaths_per_greek (m : ℕ) : ℕ := 4 * m

/-- The number of Greeks -/
def num_greeks : ℕ := 3

/-- The number of Muses -/
def num_muses : ℕ := 9

/-- The total number of people (Greeks and Muses) -/
def total_people : ℕ := num_greeks + num_muses

theorem wreath_distribution (m : ℕ) (h : m > 0) :
  ∃ (initial_wreaths : ℕ),
    initial_wreaths = wreaths_per_greek m ∧
    (initial_wreaths * num_greeks) % total_people = 0 ∧
    ∀ (final_wreaths : ℕ),
      final_wreaths * total_people = initial_wreaths * num_greeks →
      final_wreaths = m :=
by sorry

end wreath_distribution_l1652_165256


namespace solution_system_equations_l1652_165264

theorem solution_system_equations :
  ∀ x y : ℝ, x > 0 ∧ y > 0 →
  (x - 3 * Real.sqrt (x * y) - 2 * Real.sqrt (x / y) = 0 ∧
   x^2 * y^2 + x^4 = 82) →
  ((x = 3 ∧ y = 1/3) ∨ (x = Real.rpow 66 (1/4) ∧ y = 4 / Real.rpow 66 (1/4))) :=
by sorry

end solution_system_equations_l1652_165264


namespace hyperbola_vertex_distance_l1652_165291

/-- The distance between the vertices of the hyperbola x^2/144 - y^2/49 = 1 is 24 -/
theorem hyperbola_vertex_distance : 
  let a : ℝ := Real.sqrt 144
  let hyperbola := fun (x y : ℝ) ↦ x^2 / 144 - y^2 / 49 = 1
  2 * a = 24 := by sorry

end hyperbola_vertex_distance_l1652_165291


namespace arithmetic_progression_pairs_l1652_165221

-- Define what it means for four numbers to be in arithmetic progression
def is_arithmetic_progression (x y z w : ℝ) : Prop :=
  ∃ d : ℝ, y = x + d ∧ z = y + d ∧ w = z + d

-- State the theorem
theorem arithmetic_progression_pairs :
  ∀ a b : ℝ, is_arithmetic_progression 10 a b (a * b) ↔ 
  ((a = 4 ∧ b = -2) ∨ (a = 2.5 ∧ b = -5)) :=
by sorry

end arithmetic_progression_pairs_l1652_165221


namespace percentage_problem_l1652_165243

theorem percentage_problem : (45 * 7) / 900 * 100 = 35 := by
  sorry

end percentage_problem_l1652_165243


namespace tangent_line_x_intercept_l1652_165281

noncomputable def f (x : ℝ) : ℝ := Real.exp x

theorem tangent_line_x_intercept :
  let point : ℝ × ℝ := (2, Real.exp 2)
  let slope : ℝ := (deriv f) point.1
  let tangent_line (x : ℝ) : ℝ := slope * (x - point.1) + point.2
  (tangent_line 1 = 0) ∧ (∀ x : ℝ, x ≠ 1 → tangent_line x ≠ 0) :=
by
  sorry

#check tangent_line_x_intercept

end tangent_line_x_intercept_l1652_165281


namespace train_speed_calculation_l1652_165216

theorem train_speed_calculation (train_length : Real) (crossing_time : Real) : 
  train_length = 133.33333333333334 →
  crossing_time = 8 →
  (train_length / 1000) / (crossing_time / 3600) = 60 := by
  sorry

end train_speed_calculation_l1652_165216


namespace square_difference_fifty_fortynine_l1652_165209

theorem square_difference_fifty_fortynine : 50^2 - 49^2 = 99 := by
  sorry

end square_difference_fifty_fortynine_l1652_165209


namespace circle_properties_l1652_165290

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (2, 0)

/-- The radius of the circle -/
def circle_radius : ℝ := 2

/-- Theorem stating that the given equation describes a circle with the specified center and radius -/
theorem circle_properties :
  ∀ (x y : ℝ), circle_equation x y ↔ (x - circle_center.1)^2 + (y - circle_center.2)^2 = circle_radius^2 :=
by sorry

end circle_properties_l1652_165290


namespace ceiling_negative_sqrt_64_over_9_l1652_165252

theorem ceiling_negative_sqrt_64_over_9 : ⌈-Real.sqrt (64/9)⌉ = -2 := by
  sorry

end ceiling_negative_sqrt_64_over_9_l1652_165252


namespace original_number_proof_l1652_165297

theorem original_number_proof :
  ∀ x : ℕ,
  x < 10 →
  (x + 10) * ((x + 10) / x) = 72 →
  x = 2 :=
by
  sorry

end original_number_proof_l1652_165297


namespace min_value_a_plus_2b_min_value_is_2_sqrt_2_min_value_equality_l1652_165205

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (a + 1)⁻¹ + (b + 1)⁻¹ = 1) : 
  ∀ x y, x > 0 → y > 0 → (x + 1)⁻¹ + (y + 1)⁻¹ = 1 → a + 2 * b ≤ x + 2 * y :=
by
  sorry

theorem min_value_is_2_sqrt_2 (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (a + 1)⁻¹ + (b + 1)⁻¹ = 1) : 
  a + 2 * b ≥ 2 * Real.sqrt 2 :=
by
  sorry

theorem min_value_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (a + 1)⁻¹ + (b + 1)⁻¹ = 1) : 
  (a + 2 * b = 2 * Real.sqrt 2) ↔ (a + 1 = Real.sqrt 2 * (b + 1)) :=
by
  sorry

end min_value_a_plus_2b_min_value_is_2_sqrt_2_min_value_equality_l1652_165205


namespace percentage_of_fraction_equals_value_l1652_165261

theorem percentage_of_fraction_equals_value : 
  let number : ℝ := 70.58823529411765
  let fraction : ℝ := 3 / 5
  let percentage : ℝ := 85 / 100
  percentage * (fraction * number) = 36 := by
sorry

end percentage_of_fraction_equals_value_l1652_165261


namespace union_of_sets_l1652_165228

theorem union_of_sets : 
  let A : Set ℕ := {1, 2}
  let B : Set ℕ := {2, 3, 4}
  A ∪ B = {1, 2, 3, 4} := by
  sorry

end union_of_sets_l1652_165228


namespace fraction_of_ivys_collectors_dolls_l1652_165234

/-- The number of dolls Dina has -/
def dinas_dolls : ℕ := 60

/-- The number of collectors edition dolls Ivy has -/
def ivys_collectors_dolls : ℕ := 20

/-- The number of dolls Ivy has -/
def ivys_dolls : ℕ := dinas_dolls / 2

theorem fraction_of_ivys_collectors_dolls : 
  (ivys_collectors_dolls : ℚ) / (ivys_dolls : ℚ) = 2 / 3 := by sorry

end fraction_of_ivys_collectors_dolls_l1652_165234


namespace fire_water_requirement_l1652_165250

theorem fire_water_requirement 
  (flow_rate : ℝ) 
  (num_firefighters : ℕ) 
  (time_taken : ℝ) 
  (h1 : flow_rate = 20) 
  (h2 : num_firefighters = 5) 
  (h3 : time_taken = 40) : 
  flow_rate * num_firefighters * time_taken = 4000 :=
by
  sorry

end fire_water_requirement_l1652_165250


namespace betty_beads_l1652_165260

/-- Given that Betty has 3 red beads for every 2 blue beads and she has 20 blue beads,
    prove that Betty has 30 red beads. -/
theorem betty_beads (red_blue_ratio : ℚ) (blue_beads : ℕ) (red_beads : ℕ) : 
  red_blue_ratio = 3 / 2 →
  blue_beads = 20 →
  red_beads = red_blue_ratio * blue_beads →
  red_beads = 30 := by
sorry

end betty_beads_l1652_165260


namespace complex_modulus_example_l1652_165259

theorem complex_modulus_example : Complex.abs (2 - (5/6) * Complex.I) = 13/6 := by
  sorry

end complex_modulus_example_l1652_165259


namespace banana_change_l1652_165276

/-- Calculates the change received when buying bananas -/
theorem banana_change (num_bananas : ℕ) (cost_per_banana : ℚ) (amount_paid : ℚ) :
  num_bananas = 5 →
  cost_per_banana = 30 / 100 →
  amount_paid = 10 →
  amount_paid - (num_bananas : ℚ) * cost_per_banana = 17 / 2 :=
by sorry

end banana_change_l1652_165276


namespace power_equation_solution_l1652_165229

theorem power_equation_solution : ∃ x : ℤ, 5^3 - 7 = 6^2 + x ∧ x = 82 := by
  sorry

end power_equation_solution_l1652_165229


namespace john_overall_loss_l1652_165274

def grinder_cost : ℝ := 15000
def mobile_cost : ℝ := 8000
def bicycle_cost : ℝ := 12000
def laptop_cost : ℝ := 25000

def grinder_loss_percent : ℝ := 0.02
def mobile_profit_percent : ℝ := 0.10
def bicycle_profit_percent : ℝ := 0.15
def laptop_loss_percent : ℝ := 0.08

def total_cost : ℝ := grinder_cost + mobile_cost + bicycle_cost + laptop_cost

def grinder_sale : ℝ := grinder_cost * (1 - grinder_loss_percent)
def mobile_sale : ℝ := mobile_cost * (1 + mobile_profit_percent)
def bicycle_sale : ℝ := bicycle_cost * (1 + bicycle_profit_percent)
def laptop_sale : ℝ := laptop_cost * (1 - laptop_loss_percent)

def total_sale : ℝ := grinder_sale + mobile_sale + bicycle_sale + laptop_sale

theorem john_overall_loss : total_sale - total_cost = -700 := by sorry

end john_overall_loss_l1652_165274


namespace pradeep_marks_l1652_165222

theorem pradeep_marks (total_marks : ℕ) (pass_percentage : ℚ) (fail_margin : ℕ) (obtained_marks : ℕ) : 
  total_marks = 550 →
  pass_percentage = 40 / 100 →
  obtained_marks = (pass_percentage * total_marks).floor - fail_margin →
  obtained_marks = 200 := by
sorry

end pradeep_marks_l1652_165222


namespace functional_equation_proof_l1652_165267

open Real

theorem functional_equation_proof (f g : ℝ → ℝ) : 
  (∀ x y : ℝ, sin x + cos y = f x + f y + g x - g y) ↔ 
  (∃ c : ℝ, ∀ x : ℝ, f x = (sin x + cos x) / 2 ∧ g x = (sin x - cos x) / 2 + c) :=
sorry

end functional_equation_proof_l1652_165267


namespace min_value_of_function_equality_condition_l1652_165212

theorem min_value_of_function (x : ℝ) (h : x > -1) :
  2 * x + 1 / (x + 1) ≥ 2 * Real.sqrt 2 - 2 :=
by sorry

theorem equality_condition (x : ℝ) (h : x > -1) :
  2 * x + 1 / (x + 1) = 2 * Real.sqrt 2 - 2 ↔ x = Real.sqrt 2 / 2 - 1 :=
by sorry

end min_value_of_function_equality_condition_l1652_165212


namespace tunnel_length_specific_tunnel_length_l1652_165242

/-- The length of a tunnel given train and time information -/
theorem tunnel_length (train_length : ℝ) (time_diff : ℝ) (train_speed : ℝ) : ℝ :=
  let tunnel_length := train_speed * time_diff / 60
  by
    -- Proof goes here
    sorry

/-- The specific tunnel length for the given problem -/
theorem specific_tunnel_length : 
  tunnel_length 2 4 30 = 2 := by sorry

end tunnel_length_specific_tunnel_length_l1652_165242


namespace E_equality_condition_l1652_165284

/-- Definition of the function E --/
def E (a b c : ℚ) : ℚ := a * b^2 + b * c + c

/-- Theorem stating the equality condition for E(a,3,2) and E(a,5,3) --/
theorem E_equality_condition :
  ∀ a : ℚ, E a 3 2 = E a 5 3 ↔ a = -5/8 := by sorry

end E_equality_condition_l1652_165284


namespace homework_problem_count_l1652_165220

theorem homework_problem_count 
  (math_pages : ℕ) 
  (reading_pages : ℕ) 
  (problems_per_page : ℕ) 
  (h1 : math_pages = 6) 
  (h2 : reading_pages = 4) 
  (h3 : problems_per_page = 3) : 
  (math_pages + reading_pages) * problems_per_page = 30 := by
  sorry

end homework_problem_count_l1652_165220


namespace father_son_age_ratio_l1652_165296

/-- Proves that the ratio of father's age to son's age is 4:1 given the conditions -/
theorem father_son_age_ratio :
  ∀ (father_age son_age : ℕ),
    father_age = 64 →
    son_age = 16 →
    father_age - 10 + son_age - 10 = 60 →
    father_age / son_age = 4 := by
  sorry

end father_son_age_ratio_l1652_165296


namespace binary_253_property_l1652_165231

/-- Represents a binary number as a list of bits (0 or 1) -/
def BinaryRepresentation := List Nat

/-- Converts a natural number to its binary representation -/
def toBinary (n : Nat) : BinaryRepresentation :=
  sorry

/-- Counts the number of zeros in a binary representation -/
def countZeros (bin : BinaryRepresentation) : Nat :=
  sorry

/-- Counts the number of ones in a binary representation -/
def countOnes (bin : BinaryRepresentation) : Nat :=
  sorry

theorem binary_253_property :
  let bin := toBinary 253
  let a := countZeros bin
  let b := countOnes bin
  2 * b - a = 13 := by sorry

end binary_253_property_l1652_165231


namespace pattern_equation_l1652_165262

theorem pattern_equation (n : ℕ+) : n * (n + 2) + 1 = (n + 1)^2 := by
  sorry

end pattern_equation_l1652_165262


namespace triangle_inequality_l1652_165269

theorem triangle_inequality (a b c : ℝ) : |a - c| ≤ |a - b| + |b - c| := by
  sorry

end triangle_inequality_l1652_165269


namespace fraction_less_than_one_necessary_not_sufficient_l1652_165271

theorem fraction_less_than_one_necessary_not_sufficient (a : ℝ) :
  (∀ a, a > 1 → 1 / a < 1) ∧ (∃ a, 1 / a < 1 ∧ a ≤ 1) := by
  sorry

end fraction_less_than_one_necessary_not_sufficient_l1652_165271


namespace rectangle_k_value_l1652_165273

-- Define the rectangle ABCD
structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

-- Define the properties of the rectangle
def isValidRectangle (rect : Rectangle) : Prop :=
  rect.A.1 = -3 ∧ 
  rect.A.2 = 1 ∧
  rect.B.1 = 4 ∧
  rect.D.2 = rect.A.2 + (rect.B.1 - rect.A.1)

-- Define the area of the rectangle
def rectangleArea (rect : Rectangle) : ℝ :=
  (rect.B.1 - rect.A.1) * (rect.D.2 - rect.A.2)

-- Theorem statement
theorem rectangle_k_value (rect : Rectangle) (k : ℝ) :
  isValidRectangle rect →
  rectangleArea rect = 70 →
  k > 0 →
  rect.D.2 = k →
  k = 11 := by
  sorry


end rectangle_k_value_l1652_165273


namespace frequency_distribution_necessary_sufficient_l1652_165226

/-- Represents a sample of data -/
structure Sample (α : Type) where
  data : List α

/-- Represents a frequency distribution of a sample -/
structure FrequencyDistribution (α : Type) where
  ranges : List (α × α)
  counts : List Nat

/-- Represents the proportion of data points falling within a range -/
def proportion {α : Type} (s : Sample α) (range : α × α) : ℝ := sorry

/-- Main theorem: The frequency distribution is necessary and sufficient to determine
    the proportion of data points falling within any range in a sample -/
theorem frequency_distribution_necessary_sufficient
  {α : Type} [LinearOrder α] (s : Sample α) :
  ∃ (fd : FrequencyDistribution α),
    (∀ (range : α × α), ∃ (p : ℝ), proportion s range = p) ↔
    (∀ (range : α × α), ∃ (count : Nat), count ∈ fd.counts) :=
  sorry

end frequency_distribution_necessary_sufficient_l1652_165226


namespace pizza_distribution_l1652_165251

theorem pizza_distribution (num_students : ℕ) (pieces_per_pizza : ℕ) (total_pieces : ℕ) :
  num_students = 10 →
  pieces_per_pizza = 6 →
  total_pieces = 1200 →
  (total_pieces / pieces_per_pizza) / num_students = 20 :=
by sorry

end pizza_distribution_l1652_165251


namespace initial_money_calculation_l1652_165285

theorem initial_money_calculation (x : ℤ) : 
  ((x + 9) - 19 = 35) → (x = 45) := by
  sorry

end initial_money_calculation_l1652_165285


namespace largest_common_divisor_462_330_l1652_165248

theorem largest_common_divisor_462_330 : Nat.gcd 462 330 = 66 := by
  sorry

end largest_common_divisor_462_330_l1652_165248


namespace problem_solution_l1652_165208

theorem problem_solution (a b d : ℤ) 
  (h1 : a + b = d) 
  (h2 : b + d = 7) 
  (h3 : d = 4) : 
  a = 1 := by
sorry

end problem_solution_l1652_165208


namespace train_probabilities_l1652_165254

/-- Three independent events with given probabilities -/
structure ThreeIndependentEvents where
  p1 : ℝ
  p2 : ℝ
  p3 : ℝ
  p1_in_range : 0 ≤ p1 ∧ p1 ≤ 1
  p2_in_range : 0 ≤ p2 ∧ p2 ≤ 1
  p3_in_range : 0 ≤ p3 ∧ p3 ≤ 1

/-- The probability of exactly two events occurring -/
def prob_exactly_two (e : ThreeIndependentEvents) : ℝ :=
  e.p1 * e.p2 * (1 - e.p3) + e.p1 * (1 - e.p2) * e.p3 + (1 - e.p1) * e.p2 * e.p3

/-- The probability of at least one event occurring -/
def prob_at_least_one (e : ThreeIndependentEvents) : ℝ :=
  1 - (1 - e.p1) * (1 - e.p2) * (1 - e.p3)

/-- Theorem stating the probabilities for the given scenario -/
theorem train_probabilities (e : ThreeIndependentEvents) 
  (h1 : e.p1 = 0.8) (h2 : e.p2 = 0.7) (h3 : e.p3 = 0.9) : 
  prob_exactly_two e = 0.398 ∧ prob_at_least_one e = 0.994 := by
  sorry

end train_probabilities_l1652_165254


namespace parabola_intercepts_sum_l1652_165293

-- Define the parabola
def parabola (x : ℝ) : ℝ := 3 * x^2 - 9 * x + 4

-- Define the y-intercept
def y_intercept (d : ℝ) : Prop := parabola 0 = d

-- Define the x-intercepts
def x_intercepts (e f : ℝ) : Prop := parabola e = 0 ∧ parabola f = 0 ∧ e ≠ f

theorem parabola_intercepts_sum (d e f : ℝ) :
  y_intercept d → x_intercepts e f → d + e + f = 7 := by
  sorry

end parabola_intercepts_sum_l1652_165293


namespace max_value_S_l1652_165202

theorem max_value_S (x y z w : Real) 
  (hx : x ∈ Set.Icc 0 1) 
  (hy : y ∈ Set.Icc 0 1) 
  (hz : z ∈ Set.Icc 0 1) 
  (hw : w ∈ Set.Icc 0 1) : 
  (x^2*y + y^2*z + z^2*w + w^2*x - x*y^2 - y*z^2 - z*w^2 - w*x^2) ≤ 8/27 := by
  sorry

end max_value_S_l1652_165202


namespace correct_calculation_l1652_165287

theorem correct_calculation (x : ℤ) : x - 32 = 33 → x + 32 = 97 := by
  sorry

end correct_calculation_l1652_165287


namespace min_value_trig_expression_l1652_165244

theorem min_value_trig_expression (θ : Real) (h : 0 < θ ∧ θ < π/2) :
  ∃ (min_val : Real), min_val = 3*Real.sqrt 2 + Real.sqrt 3 ∧
  ∀ θ', 0 < θ' ∧ θ' < π/2 →
    3 * Real.sin θ' + 2 / Real.cos θ' + Real.sqrt 3 * (Real.cos θ' / Real.sin θ') ≥ min_val :=
by sorry

end min_value_trig_expression_l1652_165244


namespace triangle_angle_proof_l1652_165239

theorem triangle_angle_proof (a b c : ℝ) (A : ℝ) (S : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  S = (1/2) * b * c * Real.sin A →
  b^2 + c^2 = (1/3) * a^2 + (4 * Real.sqrt 3 / 3) * S →
  A = π / 6 := by
  sorry

end triangle_angle_proof_l1652_165239


namespace point_330_ratio_l1652_165286

/-- A point on the terminal side of a 330° angle, excluding the origin -/
structure Point330 where
  x : ℝ
  y : ℝ
  nonzero : x ≠ 0 ∨ y ≠ 0
  on_terminal_side : y / x = Real.tan (330 * π / 180)

/-- The ratio y/x for a point on the terminal side of a 330° angle is -√3/3 -/
theorem point_330_ratio (P : Point330) : P.y / P.x = -Real.sqrt 3 / 3 := by
  sorry

end point_330_ratio_l1652_165286


namespace cubic_fraction_equals_three_l1652_165238

theorem cubic_fraction_equals_three (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_zero : a + b + c = 0) (sum_prod_nonzero : a * b + a * c + b * c ≠ 0) :
  (a^3 + b^3 + c^3) / (a * b * c * (a * b + a * c + b * c)) = 3 := by
  sorry

end cubic_fraction_equals_three_l1652_165238


namespace flower_garden_mystery_l1652_165294

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents the arrangement of digits in the problem -/
structure Arrangement where
  garden : Fin 10000
  love : Fin 100
  unknown : Fin 100

/-- The conditions of the problem -/
def problem_conditions (a : Arrangement) : Prop :=
  ∃ (flower : Digit),
    a.garden + 6 = 85613 ∧
    a.love = 41 + a.unknown ∧
    a.garden.val = flower.val * 1000 + 9 * 100 + flower.val * 10 + 3

/-- The main theorem: proving that "花园探秘" equals 9713 -/
theorem flower_garden_mystery (a : Arrangement) 
  (h : problem_conditions a) : a.garden = 9713 := by
  sorry


end flower_garden_mystery_l1652_165294


namespace jessica_exam_progress_l1652_165266

/-- Represents the exam parameters and Jessica's progress -/
structure ExamProgress where
  total_time : ℕ  -- Total time for the exam in minutes
  total_questions : ℕ  -- Total number of questions in the exam
  time_used : ℕ  -- Time used so far in minutes
  time_remaining : ℕ  -- Time remaining when exam is finished

/-- Represents that it's impossible to determine the exact number of questions answered -/
def cannot_determine_questions_answered (ep : ExamProgress) : Prop :=
  ∀ (questions_answered : ℕ), 
    questions_answered ≤ ep.total_questions → 
    ∃ (other_answered : ℕ), 
      other_answered ≠ questions_answered ∧ 
      other_answered ≤ ep.total_questions

/-- Theorem stating that given the exam conditions, it's impossible to determine
    the exact number of questions Jessica has answered so far -/
theorem jessica_exam_progress : 
  let ep : ExamProgress := {
    total_time := 60,
    total_questions := 80,
    time_used := 12,
    time_remaining := 0
  }
  cannot_determine_questions_answered ep :=
by
  sorry

end jessica_exam_progress_l1652_165266


namespace cube_root_equation_solution_l1652_165247

theorem cube_root_equation_solution :
  ∃! x : ℝ, (5 - x / 3) ^ (1/3 : ℝ) = -4 :=
by
  -- Proof goes here
  sorry

end cube_root_equation_solution_l1652_165247


namespace sum_distinct_prime_divisors_1800_l1652_165272

def sum_of_distinct_prime_divisors (n : Nat) : Nat :=
  (Nat.factors n).toFinset.sum id

theorem sum_distinct_prime_divisors_1800 :
  sum_of_distinct_prime_divisors 1800 = 10 := by
  sorry

end sum_distinct_prime_divisors_1800_l1652_165272


namespace function_composition_equality_l1652_165258

/-- Given real numbers a, b, c, d, k where k ≠ 0, and functions f and g defined as
    f(x) = ax + b and g(x) = k(cx + d), this theorem states that f(g(x)) = g(f(x))
    if and only if b(1 - kc) = k(d(1 - a)). -/
theorem function_composition_equality
  (a b c d k : ℝ)
  (hk : k ≠ 0)
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (hf : ∀ x, f x = a * x + b)
  (hg : ∀ x, g x = k * (c * x + d)) :
  (∀ x, f (g x) = g (f x)) ↔ b * (1 - k * c) = k * (d * (1 - a)) :=
by sorry

end function_composition_equality_l1652_165258


namespace ball_hits_ground_l1652_165245

def ball_height (t : ℝ) : ℝ := -18 * t^2 + 30 * t + 60

theorem ball_hits_ground :
  ∃ t : ℝ, t > 0 ∧ ball_height t = 0 ∧ t = (5 + Real.sqrt 145) / 6 :=
sorry

end ball_hits_ground_l1652_165245


namespace tangent_sum_simplification_l1652_165236

theorem tangent_sum_simplification :
  (Real.tan (20 * π / 180) + Real.tan (30 * π / 180) + 
   Real.tan (80 * π / 180) + Real.tan (70 * π / 180)) / 
  Real.cos (40 * π / 180) = 
  (4 + 2 * (1 / Real.cos (40 * π / 180))) / Real.sqrt 3 := by
  sorry

end tangent_sum_simplification_l1652_165236


namespace arithmetic_geometric_sequence_properties_l1652_165246

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℚ := 3 * n - 1

-- Define the geometric sequence b_n
def b (n : ℕ) : ℚ := 2 * (4 ^ (n - 1))

-- Define S_n (sum of first n terms of a_n)
def S (n : ℕ) : ℚ := (3 / 2) * n^2 + (1 / 2) * n

-- Define T_n (sum of first n terms of b_n)
def T (n : ℕ) : ℚ := (2 / 3) * (4^n - 1)

theorem arithmetic_geometric_sequence_properties :
  (∀ n : ℕ, n ≥ 1 → S n = (3 / 2) * n^2 + (1 / 2) * n) ∧
  (b 1 = a 1) ∧
  (b 2 = a 3) →
  (∀ n : ℕ, n ≥ 1 → a n = 3 * n - 1) ∧
  (∀ n : ℕ, n ≥ 1 → T n = (2 / 3) * (4^n - 1)) :=
by sorry

end arithmetic_geometric_sequence_properties_l1652_165246


namespace complementary_angles_difference_l1652_165230

theorem complementary_angles_difference (a b : ℝ) : 
  a + b = 90 →  -- complementary angles
  a = 3 * b →   -- ratio of 3:1
  |a - b| = 45  -- positive difference
  := by sorry

end complementary_angles_difference_l1652_165230


namespace domain_of_f_l1652_165214

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(2x+1)
def domain_f_2x_plus_1 : Set ℝ := Set.Ioo (-2) 0

-- Theorem statement
theorem domain_of_f (x : ℝ) : 
  (∀ y ∈ domain_f_2x_plus_1, ∃ x, y = 2*x + 1) →
  (Set.Ioo (-3) 1).Nonempty →
  x ∈ Set.Ioo (-3) 1 ↔ f x ≠ 0 :=
sorry

end domain_of_f_l1652_165214


namespace smallest_surface_area_l1652_165206

def cube_surface_area (side : ℝ) : ℝ := 6 * side^2

def min_combined_surface_area (side1 side2 side3 : ℝ) : ℝ :=
  cube_surface_area side1 + cube_surface_area side2 + cube_surface_area side3 -
  (2 * side1^2 + 2 * side2^2 + 2 * side3^2)

theorem smallest_surface_area :
  min_combined_surface_area 3 5 8 = 502 := by
  sorry

end smallest_surface_area_l1652_165206


namespace unique_a_for_nonnegative_f_l1652_165219

theorem unique_a_for_nonnegative_f :
  ∃! a : ℝ, a > 0 ∧ ∀ x : ℝ, x > 0 → x^2 * (Real.log x - a) + a ≥ 0 ∧ a = 1/2 := by
  sorry

end unique_a_for_nonnegative_f_l1652_165219


namespace intersection_set_equality_l1652_165224

theorem intersection_set_equality : 
  let S := {α : ℝ | ∃ k : ℤ, α = k * π / 2 - π / 5} ∩ {α : ℝ | -π < α ∧ α < π}
  S = {-π/5, -7*π/10, 3*π/10, 4*π/5} := by sorry

end intersection_set_equality_l1652_165224


namespace ellipse_properties_l1652_165211

/-- Properties of an ellipse with equation x²/4 + y²/2 = 1 -/
theorem ellipse_properties :
  let a := 2  -- semi-major axis
  let b := Real.sqrt 2  -- semi-minor axis
  let c := Real.sqrt (a^2 - b^2)  -- focal distance / 2
  let e := c / a  -- eccentricity
  (∀ x y, x^2/4 + y^2/2 = 1 →
    (2*a = 4 ∧  -- length of major axis
     2*c = 2*Real.sqrt 2 ∧  -- focal distance
     e = Real.sqrt 2 / 2))  -- eccentricity
  := by sorry

end ellipse_properties_l1652_165211


namespace interest_rate_problem_l1652_165253

/-- Given a total sum and a second part, calculates the interest rate of the first part
    such that the interest on the first part for 8 years equals the interest on the second part for 3 years at 5% --/
def calculate_interest_rate (total_sum : ℚ) (second_part : ℚ) : ℚ :=
  let first_part := total_sum - second_part
  let second_part_interest := second_part * 5 * 3 / 100
  second_part_interest * 100 / (first_part * 8)

theorem interest_rate_problem (total_sum : ℚ) (second_part : ℚ) 
  (h1 : total_sum = 2769)
  (h2 : second_part = 1704) :
  calculate_interest_rate total_sum second_part = 3 := by
  sorry

end interest_rate_problem_l1652_165253
