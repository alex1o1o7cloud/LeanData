import Mathlib

namespace complex_magnitude_example_l2233_223314

theorem complex_magnitude_example : Complex.abs (-3 + (8/5) * Complex.I) = 17/5 := by
  sorry

end complex_magnitude_example_l2233_223314


namespace angle_identity_l2233_223321

theorem angle_identity (α : Real) (h1 : 0 ≤ α) (h2 : α < 2 * Real.pi) 
  (h3 : ∃ (x y : Real), x = Real.sin (215 * Real.pi / 180) ∧ 
                        y = Real.cos (215 * Real.pi / 180) ∧ 
                        x = Real.sin α ∧ 
                        y = Real.cos α) : 
  α = 235 * Real.pi / 180 := by
sorry

end angle_identity_l2233_223321


namespace smallest_n_with_seven_l2233_223373

/-- Check if a natural number contains the digit 7 -/
def containsSeven (n : ℕ) : Prop :=
  ∃ k m : ℕ, n = 10 * k + 7 + 10 * m

/-- The smallest natural number n such that both n^2 and (n+1)^2 contain the digit 7 -/
theorem smallest_n_with_seven : ∀ n : ℕ, n < 26 →
  ¬(containsSeven (n^2) ∧ containsSeven ((n+1)^2)) ∧
  (containsSeven (26^2) ∧ containsSeven (27^2)) :=
by sorry

end smallest_n_with_seven_l2233_223373


namespace division_error_problem_l2233_223389

theorem division_error_problem (x : ℝ) (y : ℝ) (h : y > 0) :
  (abs (5 * x - x / y) / (5 * x)) * 100 = 98 → y = 10 := by
  sorry

end division_error_problem_l2233_223389


namespace average_speed_first_half_l2233_223377

theorem average_speed_first_half (total_distance : ℝ) (total_avg_speed : ℝ) : 
  total_distance = 640 →
  total_avg_speed = 40 →
  let first_half_distance := total_distance / 2
  let second_half_distance := total_distance / 2
  let first_half_time := first_half_distance / (first_half_distance / (total_distance / (4 * total_avg_speed)))
  let second_half_time := 3 * first_half_time
  first_half_distance / first_half_time = 80 := by
  sorry

end average_speed_first_half_l2233_223377


namespace geometric_sequence_ratio_l2233_223391

/-- Given a geometric sequence with positive terms where (a_3, 1/2*a_5, a_4) form an arithmetic sequence,
    prove that (a_3 + a_5) / (a_4 + a_6) = (√5 - 1) / 2 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (h_positive : ∀ n, a n > 0)
  (h_geometric : ∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = a n * q)
  (h_arithmetic : a 3 + a 4 = a 5) :
  (a 3 + a 5) / (a 4 + a 6) = (Real.sqrt 5 - 1) / 2 := by
  sorry

end geometric_sequence_ratio_l2233_223391


namespace map_segment_to_yards_l2233_223363

/-- Converts a length in inches on a map to yards in reality, given a scale --/
def map_length_to_yards (map_length : ℚ) (scale : ℚ) : ℚ :=
  (map_length * scale) / 3

/-- The scale of the map (feet per inch) --/
def map_scale : ℚ := 500

/-- The length of the line segment on the map (in inches) --/
def line_segment_length : ℚ := 6.25

/-- Theorem: The 6.25-inch line segment on the map represents 1041 2/3 yards in reality --/
theorem map_segment_to_yards :
  map_length_to_yards line_segment_length map_scale = 1041 + 2/3 := by
  sorry

end map_segment_to_yards_l2233_223363


namespace mika_initial_stickers_l2233_223357

/-- Represents the number of stickers Mika has at different stages --/
structure StickerCount where
  initial : ℕ
  after_buying : ℕ
  after_birthday : ℕ
  after_giving : ℕ
  after_decorating : ℕ
  final : ℕ

/-- Defines the sticker transactions Mika goes through --/
def sticker_transactions (s : StickerCount) : Prop :=
  s.after_buying = s.initial + 26 ∧
  s.after_birthday = s.after_buying + 20 ∧
  s.after_giving = s.after_birthday - 6 ∧
  s.after_decorating = s.after_giving - 58 ∧
  s.final = s.after_decorating ∧
  s.final = 2

/-- Theorem stating that Mika initially had 20 stickers --/
theorem mika_initial_stickers :
  ∃ (s : StickerCount), sticker_transactions s ∧ s.initial = 20 := by
  sorry

end mika_initial_stickers_l2233_223357


namespace overall_length_is_13_l2233_223316

/-- The length of each ruler in centimeters -/
def ruler_length : ℝ := 10

/-- The mark on the first ruler that aligns with the second ruler -/
def align_mark1 : ℝ := 3

/-- The mark on the second ruler that aligns with the first ruler -/
def align_mark2 : ℝ := 4

/-- The overall length when the rulers are aligned as described -/
def L : ℝ := ruler_length + (ruler_length - align_mark2) - (align_mark2 - align_mark1)

theorem overall_length_is_13 : L = 13 := by
  sorry

end overall_length_is_13_l2233_223316


namespace triangle_properties_triangle_max_area_l2233_223371

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The sides form an arithmetic sequence -/
def isArithmeticSequence (t : Triangle) : Prop :=
  2 * t.b = t.a + t.c

/-- Vectors (3, sin B) and (2, sin C) are collinear -/
def areVectorsCollinear (t : Triangle) : Prop :=
  3 * Real.sin t.C = 2 * Real.sin t.B

/-- The product of sides a and c is 8 -/
def hasSideProduct8 (t : Triangle) : Prop :=
  t.a * t.c = 8

theorem triangle_properties (t : Triangle) 
  (h1 : isArithmeticSequence t) 
  (h2 : areVectorsCollinear t) :
  Real.cos t.A = -1/4 := by sorry

theorem triangle_max_area (t : Triangle) 
  (h1 : isArithmeticSequence t)
  (h2 : hasSideProduct8 t) :
  ∃ (S : ℝ), S = 2 * Real.sqrt 3 ∧ 
  ∀ (area : ℝ), area ≤ S := by sorry

end triangle_properties_triangle_max_area_l2233_223371


namespace sin_30_minus_one_plus_pi_to_zero_l2233_223365

theorem sin_30_minus_one_plus_pi_to_zero (h1 : Real.sin (30 * π / 180) = 1 / 2) 
  (h2 : ∀ x : ℝ, x ^ (0 : ℝ) = 1) : 
  Real.sin (30 * π / 180) - (1 + π) ^ (0 : ℝ) = -1 / 2 := by
  sorry

end sin_30_minus_one_plus_pi_to_zero_l2233_223365


namespace total_selections_exactly_three_girls_at_most_three_girls_both_boys_and_girls_l2233_223355

-- Define the number of boys and girls
def num_boys : ℕ := 8
def num_girls : ℕ := 5
def total_people : ℕ := num_boys + num_girls
def selection_size : ℕ := 6

-- (1) Total number of ways to select 6 people
theorem total_selections : Nat.choose total_people selection_size = 1716 := by sorry

-- (2) Number of ways to select exactly 3 girls
theorem exactly_three_girls : 
  Nat.choose num_girls 3 * Nat.choose num_boys 3 = 560 := by sorry

-- (3) Number of ways to select at most 3 girls
theorem at_most_three_girls : 
  Nat.choose num_boys 6 + 
  Nat.choose num_boys 5 * Nat.choose num_girls 1 + 
  Nat.choose num_boys 4 * Nat.choose num_girls 2 + 
  Nat.choose num_boys 3 * Nat.choose num_girls 3 = 1568 := by sorry

-- (4) Number of ways to select both boys and girls
theorem both_boys_and_girls : 
  Nat.choose total_people selection_size - Nat.choose num_boys selection_size = 1688 := by sorry

end total_selections_exactly_three_girls_at_most_three_girls_both_boys_and_girls_l2233_223355


namespace inequality_solution_set_l2233_223372

theorem inequality_solution_set (x : ℝ) : 
  (3 * x^2) / (1 - (3*x + 1)^(1/3))^2 ≤ x + 2 + (3*x + 1)^(1/3) → 
  -2/3 ≤ x ∧ x < 0 := by
sorry

end inequality_solution_set_l2233_223372


namespace geometric_sequence_a7_l2233_223331

/-- A geometric sequence with a_3 = 16 and a_5 = 4 has a_7 = 1 -/
theorem geometric_sequence_a7 (a : ℕ → ℝ) : 
  (∀ n : ℕ, ∃ r : ℝ, a (n + 1) = a n * r) →  -- geometric sequence condition
  a 3 = 16 →                                 -- given a_3 = 16
  a 5 = 4 →                                  -- given a_5 = 4
  a 7 = 1 :=                                 -- to prove a_7 = 1
by
  sorry


end geometric_sequence_a7_l2233_223331


namespace negative_two_squared_minus_zero_power_six_m_divided_by_two_m_l2233_223311

-- First problem
theorem negative_two_squared_minus_zero_power : ((-2 : ℤ)^2) - ((-2 : ℤ)^0) = 3 := by sorry

-- Second problem
theorem six_m_divided_by_two_m (m : ℝ) (hm : m ≠ 0) : (6 * m) / (2 * m) = 3 := by sorry

end negative_two_squared_minus_zero_power_six_m_divided_by_two_m_l2233_223311


namespace equation1_solutions_equation2_solution_l2233_223396

-- Define the equations
def equation1 (x : ℝ) : Prop := x^2 - 4*x - 6 = 0
def equation2 (x : ℝ) : Prop := x/(x-1) - 1 = 3/(x^2-1)

-- Theorem for the first equation
theorem equation1_solutions :
  ∃ x1 x2 : ℝ, 
    (x1 = 2 + Real.sqrt 10 ∧ equation1 x1) ∧
    (x2 = 2 - Real.sqrt 10 ∧ equation1 x2) :=
sorry

-- Theorem for the second equation
theorem equation2_solution :
  ∃ x : ℝ, x = 2 ∧ equation2 x :=
sorry

end equation1_solutions_equation2_solution_l2233_223396


namespace complex_sum_problem_l2233_223362

theorem complex_sum_problem (a b c d e f : ℝ) : 
  b = 3 → 
  e = -a - c → 
  (a + b * Complex.I) + (c + d * Complex.I) + (e + f * Complex.I) = 2 * Complex.I → 
  d + f = -1 := by
  sorry

end complex_sum_problem_l2233_223362


namespace cube_root_equivalence_l2233_223349

theorem cube_root_equivalence (x : ℝ) (hx : x > 0) : 
  (x^2 * x^(1/2))^(1/3) = x^(5/6) := by sorry

end cube_root_equivalence_l2233_223349


namespace sequence_general_term_l2233_223310

theorem sequence_general_term 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (h : ∀ n, S n = 2 * a n - 3) : 
  ∀ n, a n = 3 * 2^(n - 1) := by
sorry

end sequence_general_term_l2233_223310


namespace base_salary_calculation_l2233_223323

/-- Proves that the base salary in the second option is $1600 given the conditions of the problem. -/
theorem base_salary_calculation (monthly_salary : ℝ) (commission_rate : ℝ) (equal_sales : ℝ) 
  (h1 : monthly_salary = 1800)
  (h2 : commission_rate = 0.04)
  (h3 : equal_sales = 5000)
  (h4 : ∃ (base_salary : ℝ), base_salary + commission_rate * equal_sales = monthly_salary) :
  ∃ (base_salary : ℝ), base_salary = 1600 ∧ base_salary + commission_rate * equal_sales = monthly_salary :=
by sorry

end base_salary_calculation_l2233_223323


namespace magazine_subscription_cost_l2233_223322

theorem magazine_subscription_cost (reduction_percentage : ℝ) (reduction_amount : ℝ) (original_cost : ℝ) : 
  reduction_percentage = 0.30 → 
  reduction_amount = 588 → 
  reduction_percentage * original_cost = reduction_amount → 
  original_cost = 1960 := by
  sorry

end magazine_subscription_cost_l2233_223322


namespace additional_male_workers_l2233_223347

theorem additional_male_workers (initial_female_percent : ℚ) 
                                (final_female_percent : ℚ) 
                                (final_total : ℕ) : ℕ :=
  let initial_female_percent := 60 / 100
  let final_female_percent := 55 / 100
  let final_total := 312
  26

#check additional_male_workers

end additional_male_workers_l2233_223347


namespace no_winning_strategy_for_tony_l2233_223353

/-- Represents the state of the Ring Mafia game -/
structure GameState where
  total_counters : ℕ
  mafia_counters : ℕ
  town_counters : ℕ

/-- Defines a valid initial state for the Ring Mafia game -/
def valid_initial_state (state : GameState) : Prop :=
  state.total_counters ≥ 3 ∧
  state.total_counters % 2 = 1 ∧
  state.mafia_counters = (state.total_counters - 1) / 3 ∧
  state.town_counters = 2 * (state.total_counters - 1) / 3 ∧
  state.mafia_counters + state.town_counters = state.total_counters

/-- Defines a winning state for Tony -/
def tony_wins (state : GameState) : Prop :=
  state.town_counters > 0 ∧ state.mafia_counters = 0

/-- Represents a strategy for Tony -/
def TonyStrategy := GameState → Set ℕ

/-- Defines the concept of a winning strategy for Tony -/
def winning_strategy (strategy : TonyStrategy) : Prop :=
  ∀ (initial_state : GameState),
    valid_initial_state initial_state →
    ∃ (final_state : GameState),
      tony_wins final_state

/-- The main theorem: Tony does not have a winning strategy -/
theorem no_winning_strategy_for_tony :
  ¬∃ (strategy : TonyStrategy), winning_strategy strategy :=
sorry

end no_winning_strategy_for_tony_l2233_223353


namespace m_eq_two_iff_z_on_y_eq_x_l2233_223350

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := 1 + ((-1 + m) * Complex.I)

-- Define the condition for a point to lie on the line y = x
def lies_on_y_eq_x (z : ℂ) : Prop := z.im = z.re

-- State the theorem
theorem m_eq_two_iff_z_on_y_eq_x :
  ∀ m : ℝ, (m = 2) ↔ lies_on_y_eq_x (z m) :=
by sorry

end m_eq_two_iff_z_on_y_eq_x_l2233_223350


namespace hyperbola_standard_equation_l2233_223300

/-- A hyperbola with given asymptotes and a point it passes through -/
structure Hyperbola where
  /-- The slope of the asymptotes -/
  asymptote_slope : ℝ
  /-- A point that the hyperbola passes through -/
  point : ℝ × ℝ

/-- The standard form of a hyperbola equation -/
def standard_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  y^2 / 6 - x^2 / 2 = 1

/-- Theorem stating that a hyperbola with the given properties has the specified standard equation -/
theorem hyperbola_standard_equation (h : Hyperbola) 
    (h_slope : h.asymptote_slope = Real.sqrt 3)
    (h_point : h.point = (-1, 3)) :
    ∀ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | standard_equation h p.1 p.2} ↔ 
    (∃ t : ℝ, y = h.asymptote_slope * x + t ∨ y = -h.asymptote_slope * x + t) ∧
    (x = h.point.1 ∧ y = h.point.2) :=
  sorry

end hyperbola_standard_equation_l2233_223300


namespace inequality_proof_l2233_223390

theorem inequality_proof (x y : ℝ) : 
  -1/2 ≤ (x + y) * (1 - x * y) / ((1 + x^2) * (1 + y^2)) ∧ 
  (x + y) * (1 - x * y) / ((1 + x^2) * (1 + y^2)) ≤ 1/2 :=
by
  sorry

end inequality_proof_l2233_223390


namespace probability_all_red_in_hat_l2233_223382

/-- Represents the outcome of drawing chips from a hat -/
inductive DrawOutcome
  | AllRed
  | TwoGreen

/-- The probability of drawing all red chips before two green chips -/
def probability_all_red (total_chips : ℕ) (red_chips : ℕ) (green_chips : ℕ) : ℚ :=
  sorry

/-- The main theorem stating the probability of drawing all red chips -/
theorem probability_all_red_in_hat :
  probability_all_red 7 4 3 = 1/7 :=
sorry

end probability_all_red_in_hat_l2233_223382


namespace coin_problem_l2233_223346

theorem coin_problem (total_coins : ℕ) (total_value : ℚ) : 
  total_coins = 32 ∧ 
  total_value = 47/10 →
  ∃ (quarters dimes : ℕ), 
    quarters + dimes = total_coins ∧ 
    (1/4 : ℚ) * quarters + (1/10 : ℚ) * dimes = total_value ∧
    quarters = 10 := by sorry

end coin_problem_l2233_223346


namespace system_of_inequalities_solution_l2233_223308

theorem system_of_inequalities_solution (x : ℝ) : 
  (5 / (x + 3) ≥ 1 ∧ x^2 + x - 2 ≥ 0) ↔ ((-3 < x ∧ x ≤ -2) ∨ (1 ≤ x ∧ x ≤ 2)) :=
by sorry

end system_of_inequalities_solution_l2233_223308


namespace line_equation_sum_l2233_223303

/-- Given a line with slope -4 passing through the point (5, 2), 
    prove that if its equation is of the form y = mx + b, then m + b = 18 -/
theorem line_equation_sum (m b : ℝ) : 
  m = -4 → 
  2 = m * 5 + b → 
  m + b = 18 :=
by sorry

end line_equation_sum_l2233_223303


namespace candle_burn_time_l2233_223329

/-- Given that a candle lasts 8 nights and burning it for 2 hours a night uses 6 candles over 24 nights,
    prove that Carmen burns the candle for 1 hour every night in the first scenario. -/
theorem candle_burn_time (candle_duration : ℕ) (nights_per_candle : ℕ) (burn_time_second_scenario : ℕ) 
  (candles_used : ℕ) (total_nights : ℕ) :
  candle_duration = 8 ∧ 
  nights_per_candle = 8 ∧
  burn_time_second_scenario = 2 ∧
  candles_used = 6 ∧
  total_nights = 24 →
  ∃ (burn_time_first_scenario : ℕ), burn_time_first_scenario = 1 :=
by sorry

end candle_burn_time_l2233_223329


namespace tape_overlap_l2233_223352

theorem tape_overlap (tape_length : ℕ) (total_length : ℕ) (h1 : tape_length = 275) (h2 : total_length = 512) :
  2 * tape_length - total_length = 38 := by
  sorry

end tape_overlap_l2233_223352


namespace largest_number_l2233_223376

theorem largest_number (a b c d : ℝ) (h1 : a = Real.sqrt 5) (h2 : b = -1.6) (h3 : c = 0) (h4 : d = 2) :
  max a (max b (max c d)) = a :=
sorry

end largest_number_l2233_223376


namespace product_from_gcd_lcm_l2233_223397

theorem product_from_gcd_lcm (a b : ℤ) : 
  Int.gcd a b = 8 → Int.lcm a b = 48 → a * b = 384 := by
  sorry

end product_from_gcd_lcm_l2233_223397


namespace walking_speed_problem_l2233_223320

theorem walking_speed_problem (x : ℝ) :
  let james_speed := x^2 - 13*x - 30
  let jane_distance := x^2 - 5*x - 66
  let jane_time := x + 6
  let jane_speed := jane_distance / jane_time
  james_speed = jane_speed → james_speed = -4 + 2 * Real.sqrt 17 :=
by sorry

end walking_speed_problem_l2233_223320


namespace product_of_three_consecutive_cubes_divisible_by_504_l2233_223301

theorem product_of_three_consecutive_cubes_divisible_by_504 (a : ℕ) :
  ∃ k : ℕ, (a^3 - 1) * a^3 * (a^3 + 1) = 504 * k :=
by sorry

end product_of_three_consecutive_cubes_divisible_by_504_l2233_223301


namespace test_score_properties_l2233_223345

/-- A test with multiple-choice questions. -/
structure Test where
  num_questions : ℕ
  correct_points : ℕ
  incorrect_points : ℕ
  max_score : ℕ
  prob_correct : ℝ

/-- Calculate the expected score for a given test. -/
def expected_score (t : Test) : ℝ :=
  t.num_questions * (t.correct_points * t.prob_correct + t.incorrect_points * (1 - t.prob_correct))

/-- Calculate the variance of scores for a given test. -/
def score_variance (t : Test) : ℝ :=
  t.num_questions * (t.correct_points^2 * t.prob_correct + t.incorrect_points^2 * (1 - t.prob_correct) - 
    (t.correct_points * t.prob_correct + t.incorrect_points * (1 - t.prob_correct))^2)

/-- Theorem stating the expected score and variance for the given test conditions. -/
theorem test_score_properties :
  ∃ (t : Test),
    t.num_questions = 25 ∧
    t.correct_points = 4 ∧
    t.incorrect_points = 0 ∧
    t.max_score = 100 ∧
    t.prob_correct = 0.8 ∧
    expected_score t = 80 ∧
    score_variance t = 64 := by
  sorry

end test_score_properties_l2233_223345


namespace polyhedron_edge_length_bound_l2233_223378

/-- A polyhedron is represented as a set of points in ℝ³. -/
def Polyhedron : Type := Set (ℝ × ℝ × ℝ)

/-- The edges of a polyhedron. -/
def edges (P : Polyhedron) : Set (Set (ℝ × ℝ × ℝ)) := sorry

/-- The length of an edge. -/
def edgeLength (e : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

/-- The sum of all edge lengths in a polyhedron. -/
def sumEdgeLengths (P : Polyhedron) : ℝ := sorry

/-- The distance between two points in ℝ³. -/
def distance (p q : ℝ × ℝ × ℝ) : ℝ := sorry

/-- The maximum distance between any two points in a polyhedron. -/
def maxDistance (P : Polyhedron) : ℝ := sorry

/-- Theorem: The sum of edge lengths is at least 3 times the maximum distance. -/
theorem polyhedron_edge_length_bound (P : Polyhedron) :
  sumEdgeLengths P ≥ 3 * maxDistance P := by sorry

end polyhedron_edge_length_bound_l2233_223378


namespace angle_y_measure_l2233_223304

/-- An isosceles triangle with specific angle relationships -/
structure IsoscelesTriangle where
  -- Angle X in degrees
  x : ℝ
  -- Triangle sum theorem
  sum_theorem : x + 3*x + 3*x = 180
  -- Non-negativity of angles
  x_nonneg : x ≥ 0

/-- The measure of angle Y in the isosceles triangle is 540/7 degrees -/
theorem angle_y_measure (t : IsoscelesTriangle) : 3 * t.x = 540 / 7 := by
  sorry

end angle_y_measure_l2233_223304


namespace hash_seven_three_l2233_223326

/-- The # operation on real numbers -/
noncomputable def hash (x y : ℝ) : ℝ :=
  sorry

/-- The first condition: x # 0 = x -/
axiom hash_zero (x : ℝ) : hash x 0 = x

/-- The second condition: x # y = y # x -/
axiom hash_comm (x y : ℝ) : hash x y = hash y x

/-- The third condition: (x + 1) # y = (x # y) + 2y + 1 -/
axiom hash_succ (x y : ℝ) : hash (x + 1) y = hash x y + 2 * y + 1

/-- The main theorem: 7 # 3 = 52 -/
theorem hash_seven_three : hash 7 3 = 52 := by
  sorry

end hash_seven_three_l2233_223326


namespace M_on_x_axis_M_parallel_to_x_axis_M_distance_from_y_axis_l2233_223339

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given a real number m, construct a point M with coordinates (m-1, 2m+3) -/
def M (m : ℝ) : Point := ⟨m - 1, 2 * m + 3⟩

/-- N is a fixed point with coordinates (5, -1) -/
def N : Point := ⟨5, -1⟩

theorem M_on_x_axis (m : ℝ) : 
  M m = ⟨-5/2, 0⟩ ↔ (M m).y = 0 := by sorry

theorem M_parallel_to_x_axis (m : ℝ) :
  M m = ⟨-3, -1⟩ ↔ (M m).y = N.y := by sorry

theorem M_distance_from_y_axis (m : ℝ) :
  (M m = ⟨2, 9⟩ ∨ M m = ⟨-2, 1⟩) ↔ |(M m).x| = 2 := by sorry

end M_on_x_axis_M_parallel_to_x_axis_M_distance_from_y_axis_l2233_223339


namespace range_of_a_l2233_223342

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 2^x - 2 > a^2 - 3*a) → a ∈ Set.Icc 1 2 := by
  sorry

end range_of_a_l2233_223342


namespace triangle_abc_properties_l2233_223359

-- Define the triangle ABC
theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  -- Given conditions
  a + b = 11 →
  c = 7 →
  Real.cos A = -1/7 →
  -- Conclusions to prove
  a = 8 ∧
  Real.sin C = Real.sqrt 3 / 2 ∧
  (1/2 : ℝ) * a * b * Real.sin C = 6 * Real.sqrt 3 :=
by sorry


end triangle_abc_properties_l2233_223359


namespace prime_divisor_problem_l2233_223309

theorem prime_divisor_problem (n : ℕ) : 
  (∃ p : ℕ, Nat.Prime p ∧ p = Nat.sqrt n ∧ ∀ q : ℕ, Nat.Prime q ∧ q ∣ n → q ≤ p) ∧
  (∃ p : ℕ, Nat.Prime p ∧ p = Nat.sqrt (n + 72) ∧ ∀ q : ℕ, Nat.Prime q ∧ q ∣ (n + 72) → q ≤ p) →
  n = 49 ∨ n = 289 :=
by sorry

end prime_divisor_problem_l2233_223309


namespace point_x_coordinate_l2233_223340

/-- Represents a point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a straight line in the xy-plane -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Checks if a point lies on a given line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.yIntercept

theorem point_x_coordinate 
  (l : Line) 
  (p : Point) 
  (h1 : l.slope = 3.8666666666666667)
  (h2 : l.yIntercept = 20)
  (h3 : p.y = 600)
  (h4 : pointOnLine p l) :
  p.x = 150 :=
sorry

end point_x_coordinate_l2233_223340


namespace student_selection_sequences_l2233_223325

theorem student_selection_sequences (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 5) :
  Nat.descFactorial n k = 30240 := by
  sorry

end student_selection_sequences_l2233_223325


namespace first_player_wins_first_player_wins_modified_l2233_223385

/-- Represents the state of the game with two piles of stones -/
structure GameState :=
  (pile1 : Nat)
  (pile2 : Nat)

/-- Represents a move in the game -/
inductive Move
  | TakeFromFirst
  | TakeFromSecond
  | TakeFromBoth
  | TransferToSecond

/-- Defines if a move is valid for a given game state -/
def isValidMove (state : GameState) (move : Move) : Bool :=
  match move with
  | Move.TakeFromFirst => state.pile1 > 0
  | Move.TakeFromSecond => state.pile2 > 0
  | Move.TakeFromBoth => state.pile1 > 0 && state.pile2 > 0
  | Move.TransferToSecond => state.pile1 > 0

/-- Applies a move to a game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.TakeFromFirst => ⟨state.pile1 - 1, state.pile2⟩
  | Move.TakeFromSecond => ⟨state.pile1, state.pile2 - 1⟩
  | Move.TakeFromBoth => ⟨state.pile1 - 1, state.pile2 - 1⟩
  | Move.TransferToSecond => ⟨state.pile1 - 1, state.pile2 + 1⟩

/-- Determines if the game is over (no valid moves left) -/
def isGameOver (state : GameState) : Bool :=
  state.pile1 = 0 && state.pile2 = 0

/-- Theorem: The first player has a winning strategy in the two-pile stone game -/
theorem first_player_wins (initialState : GameState) 
  (h : initialState = ⟨7, 7⟩) : 
  ∃ (strategy : GameState → Move), 
    ∀ (opponentMove : Move), 
      isValidMove initialState (strategy initialState) ∧ 
      ¬isGameOver (applyMove initialState (strategy initialState)) ∧
      isGameOver (applyMove (applyMove initialState (strategy initialState)) opponentMove) :=
sorry

/-- Theorem: The first player has a winning strategy in the modified two-pile stone game -/
theorem first_player_wins_modified (initialState : GameState) 
  (h : initialState = ⟨7, 7⟩) : 
  ∃ (strategy : GameState → Move), 
    ∀ (opponentMove : Move), 
      isValidMove initialState (strategy initialState) ∧ 
      ¬isGameOver (applyMove initialState (strategy initialState)) ∧
      isGameOver (applyMove (applyMove initialState (strategy initialState)) opponentMove) :=
sorry

end first_player_wins_first_player_wins_modified_l2233_223385


namespace exists_column_with_many_zeros_l2233_223313

/-- Represents a row in the grid -/
def Row := Fin 6 → Fin 2

/-- The grid -/
def Grid (n : ℕ) := Fin n → Row

/-- Condition: integers in each row are distinct -/
def distinct_rows (g : Grid n) : Prop :=
  ∀ i j, i ≠ j → g i ≠ g j

/-- Condition: for any two rows, their element-wise product exists as a row -/
def product_exists (g : Grid n) : Prop :=
  ∀ i j, ∃ k, ∀ m, g k m = (g i m * g j m : Fin 2)

/-- Count of 0s in a column -/
def zero_count (g : Grid n) (col : Fin 6) : ℕ :=
  (Finset.filter (λ i => g i col = 0) Finset.univ).card

/-- Main theorem -/
theorem exists_column_with_many_zeros (n : ℕ) (hn : n ≥ 2) (g : Grid n)
  (h_distinct : distinct_rows g) (h_product : product_exists g) :
  ∃ col, zero_count g col ≥ n / 2 := by
  sorry

end exists_column_with_many_zeros_l2233_223313


namespace evening_screen_time_l2233_223367

-- Define the total recommended screen time in hours
def total_screen_time_hours : ℕ := 2

-- Define the screen time already used in minutes
def morning_screen_time : ℕ := 45

-- Define the function to calculate remaining screen time
def remaining_screen_time (total_hours : ℕ) (used_minutes : ℕ) : ℕ :=
  total_hours * 60 - used_minutes

-- Theorem statement
theorem evening_screen_time :
  remaining_screen_time total_screen_time_hours morning_screen_time = 75 := by
  sorry

end evening_screen_time_l2233_223367


namespace scientific_notation_exponent_l2233_223307

theorem scientific_notation_exponent (n : ℤ) : 12368000 = 1.2368 * (10 : ℝ) ^ n → n = 7 := by
  sorry

end scientific_notation_exponent_l2233_223307


namespace binary_101101_equals_octal_265_l2233_223332

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.foldr (fun b acc => 2 * acc + if b then 1 else 0) 0

/-- Converts a decimal number to its octal representation -/
def decimal_to_octal (n : Nat) : List Nat :=
  if n < 8 then [n]
  else (n % 8) :: decimal_to_octal (n / 8)

theorem binary_101101_equals_octal_265 :
  let binary : List Bool := [true, false, true, true, false, true]
  let decimal : Nat := binary_to_decimal binary
  let octal : List Nat := decimal_to_octal decimal
  octal = [5, 6, 2] := by sorry

end binary_101101_equals_octal_265_l2233_223332


namespace total_pencils_after_operations_l2233_223341

/-- 
Given:
- There are initially 43 pencils in a drawer
- There are initially 19 pencils on a desk
- 16 pencils are added to the desk
- 7 pencils are removed from the desk

Prove that the total number of pencils after these operations is 71.
-/
theorem total_pencils_after_operations : 
  ∀ (drawer_initial desk_initial added removed : ℕ),
    drawer_initial = 43 →
    desk_initial = 19 →
    added = 16 →
    removed = 7 →
    drawer_initial + (desk_initial + added - removed) = 71 :=
by
  sorry

end total_pencils_after_operations_l2233_223341


namespace trig_inequality_l2233_223327

theorem trig_inequality : 
  let a := Real.sin (Real.cos (2016 * π / 180))
  let b := Real.sin (Real.sin (2016 * π / 180))
  let c := Real.cos (Real.sin (2016 * π / 180))
  let d := Real.cos (Real.cos (2016 * π / 180))
  c > d ∧ d > b ∧ b > a := by sorry

end trig_inequality_l2233_223327


namespace subset_implies_a_geq_three_l2233_223335

def A : Set ℝ := {x | |x - 2| < 1}
def B (a : ℝ) : Set ℝ := {y | ∃ x, y = -x^2 + a}

theorem subset_implies_a_geq_three (a : ℝ) (h : A ⊆ B a) : a ≥ 3 := by
  sorry

end subset_implies_a_geq_three_l2233_223335


namespace jeans_savings_l2233_223399

/-- Calculates the total amount saved on a purchase with multiple discounts and taxes -/
def calculateSavings (originalPrice : ℝ) (saleDiscount : ℝ) (couponDiscount : ℝ) 
                     (creditCardDiscount : ℝ) (rebateDiscount : ℝ) (salesTax : ℝ) : ℝ :=
  let priceAfterSale := originalPrice * (1 - saleDiscount)
  let priceAfterCoupon := priceAfterSale - couponDiscount
  let priceAfterCreditCard := priceAfterCoupon * (1 - creditCardDiscount)
  let priceBeforeRebate := priceAfterCreditCard
  let taxAmount := priceBeforeRebate * salesTax
  let finalPrice := (priceBeforeRebate - priceBeforeRebate * rebateDiscount) + taxAmount
  originalPrice - finalPrice

theorem jeans_savings :
  calculateSavings 125 0.20 10 0.10 0.05 0.08 = 41.57 := by
  sorry

end jeans_savings_l2233_223399


namespace trigonometric_identity_l2233_223356

theorem trigonometric_identity (θ : ℝ) (h : Real.sin (3 * π / 2 + θ) = 1 / 4) :
  (Real.cos (π + θ)) / (Real.cos θ * (Real.cos (π + θ) - 1)) +
  (Real.cos (θ - 2 * π)) / (Real.cos (θ + 2 * π) * Real.cos (θ + π) + Real.cos (-θ)) = 32 / 15 := by
  sorry

end trigonometric_identity_l2233_223356


namespace point_order_on_line_l2233_223336

/-- Proves that for points (-3, y₁), (1, y₂), (-1, y₃) lying on the line y = 3x - b, 
    the relationship y₁ < y₃ < y₂ holds. -/
theorem point_order_on_line (b y₁ y₂ y₃ : ℝ) 
  (h₁ : y₁ = 3 * (-3) - b)
  (h₂ : y₂ = 3 * 1 - b)
  (h₃ : y₃ = 3 * (-1) - b) :
  y₁ < y₃ ∧ y₃ < y₂ := by
  sorry

end point_order_on_line_l2233_223336


namespace wooden_box_height_is_6_meters_l2233_223383

def wooden_box_length : ℝ := 8
def wooden_box_width : ℝ := 10
def small_box_length : ℝ := 0.04
def small_box_width : ℝ := 0.05
def small_box_height : ℝ := 0.06
def max_small_boxes : ℕ := 4000000

theorem wooden_box_height_is_6_meters :
  let small_box_volume := small_box_length * small_box_width * small_box_height
  let total_volume := small_box_volume * max_small_boxes
  let wooden_box_height := total_volume / (wooden_box_length * wooden_box_width)
  wooden_box_height = 6 := by sorry

end wooden_box_height_is_6_meters_l2233_223383


namespace min_distance_to_plane_l2233_223348

theorem min_distance_to_plane (x y z : ℝ) :
  x + 2*y + 3*z = 1 →
  x^2 + y^2 + z^2 ≥ 1/14 :=
by sorry

end min_distance_to_plane_l2233_223348


namespace expression_simplification_l2233_223317

theorem expression_simplification 
  (a b c d x y : ℝ) 
  (h : c * x ≠ d * y) : 
  (c * x * (b^2 * x^2 - 4 * b^2 * y^2 + a^2 * y^2) - 
   d * y * (b^2 * x^2 - 2 * a^2 * x^2 - 3 * a^2 * y^2)) / 
  (c * x - d * y) = 
  b^2 * x^2 + a^2 * y^2 := by
  sorry

end expression_simplification_l2233_223317


namespace scientific_notation_of_8790000_l2233_223369

theorem scientific_notation_of_8790000 :
  8790000 = 8.79 * (10 ^ 6) := by
  sorry

end scientific_notation_of_8790000_l2233_223369


namespace classroom_notebooks_l2233_223334

theorem classroom_notebooks (total_students : ℕ) 
  (notebooks_group1 : ℕ) (notebooks_group2 : ℕ) : 
  total_students = 28 →
  notebooks_group1 = 5 →
  notebooks_group2 = 3 →
  (total_students / 2 * notebooks_group1 + total_students / 2 * notebooks_group2) = 112 := by
  sorry

end classroom_notebooks_l2233_223334


namespace navigation_time_is_21_days_l2233_223398

/-- Represents the timeline of a cargo shipment from Shanghai to Vancouver --/
structure CargoShipment where
  /-- Number of days for the ship to navigate from Shanghai to Vancouver --/
  navigationDays : ℕ
  /-- Number of days for customs and regulatory processes in Vancouver --/
  customsDays : ℕ
  /-- Number of days from port to warehouse --/
  portToWarehouseDays : ℕ
  /-- Number of days since the ship departed --/
  daysSinceDeparture : ℕ
  /-- Number of days until expected arrival at the warehouse --/
  daysUntilArrival : ℕ

/-- The theorem stating that the navigation time is 21 days --/
theorem navigation_time_is_21_days (shipment : CargoShipment)
  (h1 : shipment.customsDays = 4)
  (h2 : shipment.portToWarehouseDays = 7)
  (h3 : shipment.daysSinceDeparture = 30)
  (h4 : shipment.daysUntilArrival = 2)
  (h5 : shipment.navigationDays + shipment.customsDays + shipment.portToWarehouseDays =
        shipment.daysSinceDeparture + shipment.daysUntilArrival) :
  shipment.navigationDays = 21 := by
  sorry

end navigation_time_is_21_days_l2233_223398


namespace f_properties_l2233_223305

/-- The function f(x) = a*ln(x) + b/x + c/(x^2) -/
noncomputable def f (a b c x : ℝ) : ℝ := a * Real.log x + b / x + c / (x^2)

/-- The statement that f has both a maximum and a minimum value -/
def has_max_and_min (f : ℝ → ℝ) : Prop := ∃ (x_max x_min : ℝ), ∀ x, f x ≤ f x_max ∧ f x_min ≤ f x

theorem f_properties (a b c : ℝ) (ha : a ≠ 0) 
  (h_max_min : has_max_and_min (f a b c)) :
  ab > 0 ∧ b^2 + 8*a*c > 0 ∧ a*c < 0 := by sorry

end f_properties_l2233_223305


namespace sum_distances_bound_l2233_223338

/-- A convex quadrilateral with side lengths p, q, r, s, where p ≤ q ≤ r ≤ s -/
structure ConvexQuadrilateral where
  p : ℝ
  q : ℝ
  r : ℝ
  s : ℝ
  p_le_q : p ≤ q
  q_le_r : q ≤ r
  r_le_s : r ≤ s
  convex : True  -- Assuming convexity without formal definition

/-- The sum of distances from an interior point to each side of the quadrilateral -/
def sum_distances (quad : ConvexQuadrilateral) (P : ℝ × ℝ) : ℝ :=
  sorry  -- Definition of the sum of distances

/-- Theorem: The sum of distances from any interior point to each side 
    is less than or equal to 3 times the sum of all side lengths -/
theorem sum_distances_bound (quad : ConvexQuadrilateral) (P : ℝ × ℝ) :
  sum_distances quad P ≤ 3 * (quad.p + quad.q + quad.r + quad.s) :=
sorry

end sum_distances_bound_l2233_223338


namespace probability_five_heads_ten_coins_l2233_223361

theorem probability_five_heads_ten_coins : 
  let n : ℕ := 10  -- total number of coins
  let k : ℕ := 5   -- number of heads we're looking for
  let p : ℚ := 1/2 -- probability of getting heads on a single coin flip
  Nat.choose n k * p^k * (1-p)^(n-k) = 63/256 := by
  sorry

end probability_five_heads_ten_coins_l2233_223361


namespace min_value_theorem_min_value_achieved_l2233_223330

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + 2*y + 3*z = 1) :
  1/(x+2*y) + 4/(2*y+3*z) + 9/(3*z+x) ≥ 18 := by
sorry

theorem min_value_achieved (ε : ℝ) (hε : ε > 0) : 
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ x + 2*y + 3*z = 1 ∧ 
  1/(x+2*y) + 4/(2*y+3*z) + 9/(3*z+x) < 18 + ε := by
sorry

end min_value_theorem_min_value_achieved_l2233_223330


namespace birthday_cake_is_tradition_l2233_223393

/-- Represents different types of office practices -/
inductive OfficePractice
  | Tradition
  | Balance
  | Concern
  | Relationship

/-- Represents the office birthday cake practice -/
def birthdayCakePractice : OfficePractice := OfficePractice.Tradition

/-- Theorem stating that the office birthday cake practice is a tradition -/
theorem birthday_cake_is_tradition : 
  birthdayCakePractice = OfficePractice.Tradition := by sorry


end birthday_cake_is_tradition_l2233_223393


namespace translation_down_3_units_l2233_223368

def f (x : ℝ) : ℝ := 3 * x + 2

def g (x : ℝ) : ℝ := 3 * x - 1

def vertical_translation (h : ℝ → ℝ) (d : ℝ) : ℝ → ℝ := fun x ↦ h x - d

theorem translation_down_3_units :
  vertical_translation f 3 = g := by sorry

end translation_down_3_units_l2233_223368


namespace expression_simplification_and_evaluation_l2233_223370

theorem expression_simplification_and_evaluation :
  let x : ℤ := -1
  let original_expression := (x * (x + 1)) - ((x + 2) * (2 - x)) - (2 * (x + 2)^2)
  let simplified_expression := -2 * x^2 - 9 * x - 12
  original_expression = simplified_expression ∧ simplified_expression = -5 := by
  sorry

end expression_simplification_and_evaluation_l2233_223370


namespace hexagonal_prism_lateral_area_l2233_223302

/-- The lateral surface area of a hexagonal prism with regular hexagon base -/
def lateralSurfaceArea (baseSideLength : ℝ) (lateralEdgeLength : ℝ) : ℝ :=
  6 * baseSideLength * lateralEdgeLength

/-- Theorem: The lateral surface area of a hexagonal prism with base side length 3 and lateral edge length 4 is 72 -/
theorem hexagonal_prism_lateral_area :
  lateralSurfaceArea 3 4 = 72 := by
  sorry

end hexagonal_prism_lateral_area_l2233_223302


namespace quadratic_root_implies_k_zero_l2233_223394

/-- Given a quadratic equation (k-1)x^2 + x - k^2 = 0 with a root x = 1, prove that k = 0 -/
theorem quadratic_root_implies_k_zero (k : ℝ) : 
  ((k - 1) * 1^2 + 1 - k^2 = 0) → k = 0 := by
  sorry

end quadratic_root_implies_k_zero_l2233_223394


namespace union_necessary_not_sufficient_for_intersection_l2233_223374

theorem union_necessary_not_sufficient_for_intersection (A B : Set α) :
  (∀ x, x ∈ A ∩ B → x ∈ A ∪ B) ∧
  (∃ x, x ∈ A ∪ B ∧ x ∉ A ∩ B) :=
sorry

end union_necessary_not_sufficient_for_intersection_l2233_223374


namespace rectangle_area_l2233_223312

/-- The area of a rectangle with length 4 cm and width 2 cm is 8 cm² -/
theorem rectangle_area : 
  ∀ (length width area : ℝ),
  length = 4 →
  width = 2 →
  area = length * width →
  area = 8 := by
sorry

end rectangle_area_l2233_223312


namespace sqrt_x_plus_inv_sqrt_x_l2233_223354

theorem sqrt_x_plus_inv_sqrt_x (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50) :
  Real.sqrt x + 1 / Real.sqrt x = 2 * Real.sqrt 13 := by
  sorry

end sqrt_x_plus_inv_sqrt_x_l2233_223354


namespace field_trip_bus_capacity_l2233_223387

theorem field_trip_bus_capacity 
  (total_vehicles : Nat) 
  (people_per_van : Nat) 
  (total_people : Nat) 
  (num_vans : Nat) 
  (num_buses : Nat) 
  (h1 : total_vehicles = num_vans + num_buses)
  (h2 : num_vans = 2)
  (h3 : num_buses = 3)
  (h4 : people_per_van = 8)
  (h5 : total_people = 76) :
  (total_people - num_vans * people_per_van) / num_buses = 20 := by
sorry

end field_trip_bus_capacity_l2233_223387


namespace polynomial_value_l2233_223388

theorem polynomial_value (x y : ℝ) (h : 2 * x^2 + 3 * y + 7 = 8) :
  -2 * x^2 - 3 * y + 10 = 9 := by
sorry

end polynomial_value_l2233_223388


namespace second_trip_crates_parameters_are_valid_l2233_223324

/-- The number of crates carried in the second trip of a trailer -/
def crates_in_second_trip (total_crates : ℕ) (min_crate_weight : ℕ) (max_trip_weight : ℕ) : ℕ :=
  total_crates - (max_trip_weight / min_crate_weight)

/-- Theorem stating that given the specified conditions, the trailer carries 7 crates in the second trip -/
theorem second_trip_crates :
  crates_in_second_trip 12 120 600 = 7 := by
  sorry

/-- Checks if the given parameters satisfy the problem conditions -/
def valid_parameters (total_crates : ℕ) (min_crate_weight : ℕ) (max_trip_weight : ℕ) : Prop :=
  total_crates > 0 ∧
  min_crate_weight > 0 ∧
  max_trip_weight > 0 ∧
  min_crate_weight * total_crates > max_trip_weight

/-- Theorem stating that the given parameters satisfy the problem conditions -/
theorem parameters_are_valid :
  valid_parameters 12 120 600 := by
  sorry

end second_trip_crates_parameters_are_valid_l2233_223324


namespace floor_sqrt_200_l2233_223358

theorem floor_sqrt_200 : ⌊Real.sqrt 200⌋ = 14 := by sorry

end floor_sqrt_200_l2233_223358


namespace max_value_ab_l2233_223318

theorem max_value_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : Real.sqrt 3 = Real.sqrt (3^a * 3^b)) : 
  (∀ x y : ℝ, x > 0 → y > 0 → x * y ≤ (1/4 : ℝ)) → a * b ≤ (1/4 : ℝ) := by
  sorry

end max_value_ab_l2233_223318


namespace car_truck_sales_l2233_223333

theorem car_truck_sales (total_vehicles : ℕ) (car_truck_difference : ℕ) : 
  total_vehicles = 69 → car_truck_difference = 27 → 
  ∃ (trucks : ℕ), trucks = 21 ∧ trucks + (trucks + car_truck_difference) = total_vehicles := by
sorry

end car_truck_sales_l2233_223333


namespace total_cost_over_two_years_l2233_223381

/-- Represents the number of games attended and their types -/
structure GameAttendance where
  home : Nat
  away : Nat
  homePlayoff : Nat
  awayPlayoff : Nat

/-- Represents the ticket prices for different game types -/
structure TicketPrices where
  home : Nat
  away : Nat
  homePlayoff : Nat
  awayPlayoff : Nat

/-- Calculates the total cost for a given year -/
def calculateYearlyCost (attendance : GameAttendance) (prices : TicketPrices) : Nat :=
  attendance.home * prices.home +
  attendance.away * prices.away +
  attendance.homePlayoff * prices.homePlayoff +
  attendance.awayPlayoff * prices.awayPlayoff

/-- Theorem stating the total cost over two years -/
theorem total_cost_over_two_years
  (prices : TicketPrices)
  (thisYear : GameAttendance)
  (lastYear : GameAttendance)
  (h1 : prices.home = 60)
  (h2 : prices.away = 75)
  (h3 : prices.homePlayoff = 120)
  (h4 : prices.awayPlayoff = 100)
  (h5 : thisYear.home = 2)
  (h6 : thisYear.away = 2)
  (h7 : thisYear.homePlayoff = 1)
  (h8 : thisYear.awayPlayoff = 0)
  (h9 : lastYear.home = 6)
  (h10 : lastYear.away = 3)
  (h11 : lastYear.homePlayoff = 1)
  (h12 : lastYear.awayPlayoff = 1) :
  calculateYearlyCost thisYear prices + calculateYearlyCost lastYear prices = 1195 := by
  sorry

end total_cost_over_two_years_l2233_223381


namespace magic_numbers_theorem_l2233_223360

/-- Represents the numbers chosen by three people -/
structure Numbers where
  ana : ℕ
  beto : ℕ
  caio : ℕ

/-- Performs one round of exchange -/
def exchange (n : Numbers) : Numbers :=
  { ana := n.beto + n.caio
  , beto := n.ana + n.caio
  , caio := n.ana + n.beto }

/-- The theorem to prove -/
theorem magic_numbers_theorem (initial : Numbers) :
  1 ≤ initial.ana ∧ initial.ana ≤ 50 ∧
  1 ≤ initial.beto ∧ initial.beto ≤ 50 ∧
  1 ≤ initial.caio ∧ initial.caio ≤ 50 →
  let second := exchange initial
  let final := exchange second
  final.ana = 104 ∧ final.beto = 123 ∧ final.caio = 137 →
  initial.ana = 13 ∧ initial.beto = 32 ∧ initial.caio = 46 :=
by
  sorry


end magic_numbers_theorem_l2233_223360


namespace sqrt_16_equals_4_l2233_223319

theorem sqrt_16_equals_4 : Real.sqrt 16 = 4 := by
  sorry

end sqrt_16_equals_4_l2233_223319


namespace magician_earnings_calculation_l2233_223344

/-- The amount of money earned by a magician selling card decks -/
def magician_earnings (price_per_deck : ℕ) (initial_decks : ℕ) (final_decks : ℕ) : ℕ :=
  (initial_decks - final_decks) * price_per_deck

/-- Theorem: The magician earns $56 -/
theorem magician_earnings_calculation :
  magician_earnings 7 16 8 = 56 := by
  sorry

end magician_earnings_calculation_l2233_223344


namespace jason_climbing_speed_l2233_223379

/-- Given that Matt climbs at 6 feet per minute and Jason is 42 feet higher than Matt after 7 minutes,
    prove that Jason's climbing speed is 12 feet per minute. -/
theorem jason_climbing_speed (matt_speed : ℝ) (time : ℝ) (height_difference : ℝ) :
  matt_speed = 6 →
  time = 7 →
  height_difference = 42 →
  (time * matt_speed + height_difference) / time = 12 := by
sorry

end jason_climbing_speed_l2233_223379


namespace average_chocolate_pieces_per_cookie_l2233_223375

theorem average_chocolate_pieces_per_cookie 
  (num_cookies : ℕ) 
  (num_choc_chips : ℕ) 
  (num_mms : ℕ) 
  (h1 : num_cookies = 48) 
  (h2 : num_choc_chips = 108) 
  (h3 : num_mms = num_choc_chips / 3) : 
  (num_choc_chips + num_mms) / num_cookies = 3 := by
  sorry

end average_chocolate_pieces_per_cookie_l2233_223375


namespace sequence_properties_l2233_223351

/-- Sequence of integers defined by a recursive formula -/
def a : ℕ → ℕ
  | 0 => 4
  | 1 => 11
  | (n + 2) => 3 * a (n + 1) - a n

/-- Theorem stating the properties of the sequence -/
theorem sequence_properties :
  ∀ n : ℕ,
    a (n + 1) > a n ∧
    Nat.gcd (a n) (a (n + 1)) = 1 ∧
    (a n ∣ a (n + 1)^2 - 5) ∧
    (a (n + 1) ∣ a n^2 - 5) :=
by sorry

end sequence_properties_l2233_223351


namespace product_sale_result_l2233_223343

def cost_price : ℝ := 100
def markup_percentage : ℝ := 0.2
def discount_percentage : ℝ := 0.2
def final_selling_price : ℝ := 96

theorem product_sale_result :
  let initial_price := cost_price * (1 + markup_percentage)
  let discounted_price := initial_price * (1 - discount_percentage)
  discounted_price = final_selling_price ∧ 
  cost_price - final_selling_price = 4 := by
sorry

end product_sale_result_l2233_223343


namespace smallest_n_complex_equality_l2233_223386

theorem smallest_n_complex_equality (n : ℕ) (a b c : ℝ) :
  (n > 0) →
  (a > 0) →
  (b > 0) →
  (c > 0) →
  (∀ k : ℕ, k > 0 ∧ k < n → ¬ ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ (x + y*I + z*I)^k = (x - y*I - z*I)^k) →
  ((a + b*I + c*I)^n = (a - b*I - c*I)^n) →
  ((b + c) / a = Real.sqrt (12 / 5)) :=
by sorry

end smallest_n_complex_equality_l2233_223386


namespace reciprocal_of_negative_one_third_l2233_223395

theorem reciprocal_of_negative_one_third :
  ∀ x : ℚ, x * (-1/3) = 1 → x = -3 := by
  sorry

end reciprocal_of_negative_one_third_l2233_223395


namespace equation_solutions_l2233_223380

theorem equation_solutions : 
  (∃ x₁ x₂ : ℝ, x₁ = 4 ∧ x₂ = 7 ∧ 
    ∀ x : ℝ, 3 * (x - 4) = (x - 4)^2 ↔ x = x₁ ∨ x = x₂) ∧
  (∃ y₁ y₂ : ℝ, y₁ = (-1 + Real.sqrt 10) / 3 ∧ y₂ = (-1 - Real.sqrt 10) / 3 ∧ 
    ∀ x : ℝ, 3 * x^2 + 2 * x - 3 = 0 ↔ x = y₁ ∨ x = y₂) :=
by sorry

end equation_solutions_l2233_223380


namespace unique_positive_solution_l2233_223366

open Real

theorem unique_positive_solution :
  ∃! x : ℝ, x > 0 ∧ sin (arccos (tan (arcsin x))) = x :=
by sorry

end unique_positive_solution_l2233_223366


namespace max_plus_min_equals_zero_l2233_223337

def f (x : ℝ) := x^3 - 3*x

theorem max_plus_min_equals_zero :
  ∀ m n : ℝ,
  (∀ x : ℝ, f x ≤ m) →
  (∃ x : ℝ, f x = m) →
  (∀ x : ℝ, n ≤ f x) →
  (∃ x : ℝ, f x = n) →
  m + n = 0 :=
by sorry

end max_plus_min_equals_zero_l2233_223337


namespace negation_equivalence_l2233_223328

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x < 0 ∧ x^2 - 2*x > 0) ↔ (∀ x : ℝ, x < 0 → x^2 - 2*x ≤ 0) :=
by sorry

end negation_equivalence_l2233_223328


namespace correct_calculation_l2233_223392

theorem correct_calculation (x : ℝ) : 3 * x - 5 = 103 → x / 3 - 5 = 7 := by
  sorry

end correct_calculation_l2233_223392


namespace double_scientific_notation_l2233_223306

theorem double_scientific_notation : 
  let x : ℝ := 1.2 * (10 ^ 6)
  2 * x = 2.4 * (10 ^ 6) := by sorry

end double_scientific_notation_l2233_223306


namespace rectangle_area_equals_perimeter_l2233_223315

theorem rectangle_area_equals_perimeter (x : ℝ) : 
  (4 * x) * (x + 4) = 2 * (4 * x) + 2 * (x + 4) → x = 1/2 := by
  sorry

end rectangle_area_equals_perimeter_l2233_223315


namespace max_span_sum_of_digits_div_by_8_l2233_223384

/-- Sum of digits of a positive integer -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem: Maximum span between numbers with sum of digits divisible by 8 -/
theorem max_span_sum_of_digits_div_by_8 (m : ℕ) (h1 : m > 0) (h2 : sumOfDigits m % 8 = 0) :
  ∃ (n : ℕ), n = 15 ∧
    sumOfDigits (m + n) % 8 = 0 ∧
    ∀ k : ℕ, 1 ≤ k → k < n → sumOfDigits (m + k) % 8 ≠ 0 ∧
    ∀ n' : ℕ, n' > n →
      ¬(sumOfDigits (m + n') % 8 = 0 ∧
        ∀ k : ℕ, 1 ≤ k → k < n' → sumOfDigits (m + k) % 8 ≠ 0) :=
by sorry

end max_span_sum_of_digits_div_by_8_l2233_223384


namespace cost_of_dozen_rolls_l2233_223364

/-- The cost of a dozen rolls given the total spent and number of rolls purchased -/
theorem cost_of_dozen_rolls (total_spent : ℚ) (total_rolls : ℕ) (h1 : total_spent = 15) (h2 : total_rolls = 36) : 
  total_spent / (total_rolls / 12 : ℚ) = 5 := by
  sorry

end cost_of_dozen_rolls_l2233_223364
