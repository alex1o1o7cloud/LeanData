import Mathlib

namespace NUMINAMATH_CALUDE_inequality_equivalence_l3786_378659

theorem inequality_equivalence (x : ℝ) : 
  (3 / (5 - 3 * x) > 1) ↔ (2 / 3 < x ∧ x < 5 / 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3786_378659


namespace NUMINAMATH_CALUDE_square_of_difference_l3786_378678

theorem square_of_difference (x : ℝ) : (x - 1)^2 = x^2 + 1 - 2*x := by
  sorry

end NUMINAMATH_CALUDE_square_of_difference_l3786_378678


namespace NUMINAMATH_CALUDE_rahim_book_purchase_l3786_378658

/-- The amount Rahim paid for books from the first shop -/
def amount_first_shop (books_first_shop : ℕ) (books_second_shop : ℕ) (price_second_shop : ℚ) (average_price : ℚ) : ℚ :=
  (average_price * (books_first_shop + books_second_shop : ℚ)) - price_second_shop

/-- Theorem stating the amount Rahim paid for books from the first shop -/
theorem rahim_book_purchase :
  amount_first_shop 65 50 920 (18088695652173913 / 1000000000000000) = 1160 := by
  sorry

end NUMINAMATH_CALUDE_rahim_book_purchase_l3786_378658


namespace NUMINAMATH_CALUDE_stamp_collection_fraction_l3786_378690

/-- Given the stamp collection scenario, prove that KJ has half the stamps of AJ -/
theorem stamp_collection_fraction :
  ∀ (cj kj aj : ℕ) (f : ℚ),
  -- CJ has 5 more than twice the number of stamps that KJ has
  cj = 2 * kj + 5 →
  -- KJ has a certain fraction of the number of stamps AJ has
  kj = f * aj →
  -- The three boys have 930 stamps in total
  cj + kj + aj = 930 →
  -- AJ has 370 stamps
  aj = 370 →
  -- The fraction of stamps KJ has compared to AJ is 1/2
  f = 1/2 := by
sorry


end NUMINAMATH_CALUDE_stamp_collection_fraction_l3786_378690


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l3786_378663

def U : Set ℕ := { x | (x - 1) / (5 - x) > 0 ∧ x > 0 }

def A : Set ℕ := {2, 3}

theorem complement_of_A_in_U : Set.compl A ∩ U = {4} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l3786_378663


namespace NUMINAMATH_CALUDE_chairs_count_l3786_378680

/-- The number of chairs bought for the entire house -/
def total_chairs (living_room kitchen dining_room outdoor_patio : ℕ) : ℕ :=
  living_room + kitchen + dining_room + outdoor_patio

/-- Theorem stating that the total number of chairs is 29 -/
theorem chairs_count :
  total_chairs 3 6 8 12 = 29 := by
  sorry

end NUMINAMATH_CALUDE_chairs_count_l3786_378680


namespace NUMINAMATH_CALUDE_calculation_proof_equation_no_solution_l3786_378608

-- Part 1
theorem calculation_proof : (Real.sqrt 12 - 3 * Real.sqrt (1/3)) / Real.sqrt 3 = 1 := by
  sorry

-- Part 2
theorem equation_no_solution :
  ∀ x : ℝ, x ≠ 1 ∧ x ≠ -1 → (x - 1) / (x + 1) + 4 / (x^2 - 1) ≠ (x + 1) / (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_equation_no_solution_l3786_378608


namespace NUMINAMATH_CALUDE_parabola_theorem_l3786_378674

/-- Parabola C defined by x²=4y -/
def parabola_C (x y : ℝ) : Prop := x^2 = 4*y

/-- Point P on parabola C -/
def point_P : ℝ × ℝ := (2, 1)

/-- Focus F of parabola C -/
def focus_F : ℝ × ℝ := (0, 1)

/-- Point H where the axis of the parabola intersects the y-axis -/
def point_H : ℝ × ℝ := (0, -1)

/-- Line l passing through focus F and intersecting parabola C at points A and B -/
def line_l (x y : ℝ) : Prop := ∃ (k b : ℝ), y = k*x + b ∧ (0 = k*0 + b - 1)

/-- Points A and B on parabola C -/
def points_AB (A B : ℝ × ℝ) : Prop :=
  parabola_C A.1 A.2 ∧ parabola_C B.1 B.2 ∧
  line_l A.1 A.2 ∧ line_l B.1 B.2

/-- AB is perpendicular to HB -/
def AB_perp_HB (A B : ℝ × ℝ) : Prop :=
  (A.2 - B.2) * (B.1 - point_H.1) = -(A.1 - B.1) * (B.2 - point_H.2)

/-- Main theorem: |AF| - |BF| = 4 -/
theorem parabola_theorem (A B : ℝ × ℝ) :
  points_AB A B → AB_perp_HB A B →
  Real.sqrt ((A.1 - focus_F.1)^2 + (A.2 - focus_F.2)^2) -
  Real.sqrt ((B.1 - focus_F.1)^2 + (B.2 - focus_F.2)^2) = 4 :=
by sorry

end NUMINAMATH_CALUDE_parabola_theorem_l3786_378674


namespace NUMINAMATH_CALUDE_hcl_moles_combined_l3786_378664

/-- The number of moles of HCl combined to produce a given amount of NH4Cl -/
theorem hcl_moles_combined 
  (nh3_moles : ℝ) 
  (nh4cl_grams : ℝ) 
  (nh4cl_molar_mass : ℝ) 
  (h1 : nh3_moles = 3)
  (h2 : nh4cl_grams = 159)
  (h3 : nh4cl_molar_mass = 53.50) :
  ∃ hcl_moles : ℝ, abs (hcl_moles - (nh4cl_grams / nh4cl_molar_mass)) < 0.001 :=
by
  sorry

#check hcl_moles_combined

end NUMINAMATH_CALUDE_hcl_moles_combined_l3786_378664


namespace NUMINAMATH_CALUDE_quadratic_function_inequality_l3786_378676

theorem quadratic_function_inequality (a : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : 0 < a) (h₂ : a < 3) (h₃ : x₁ < x₂) (h₄ : x₁ + x₂ ≠ 1 - a) :
  let f := fun x => a * x^2 + 2 * a * x + 4
  f x₁ < f x₂ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_inequality_l3786_378676


namespace NUMINAMATH_CALUDE_calculator_to_protractor_equivalence_l3786_378600

/-- Exchange rates at a math conference -/
structure ExchangeRates where
  calculator_to_ruler : ℚ
  ruler_to_compass : ℚ
  compass_to_protractor : ℚ

/-- The exchange rates given in the problem -/
def conference_rates : ExchangeRates where
  calculator_to_ruler := 100
  ruler_to_compass := 3/1
  compass_to_protractor := 2/1

/-- Theorem stating the equivalence between calculators and protractors -/
theorem calculator_to_protractor_equivalence (rates : ExchangeRates) :
  rates.calculator_to_ruler * rates.ruler_to_compass * rates.compass_to_protractor = 600 → 
  rates = conference_rates :=
sorry

#check calculator_to_protractor_equivalence

end NUMINAMATH_CALUDE_calculator_to_protractor_equivalence_l3786_378600


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l3786_378609

theorem system_of_equations_solution (x y m : ℝ) : 
  2 * x + y = 4 → 
  x + 2 * y = m → 
  x + y = 1 → 
  m = -1 := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l3786_378609


namespace NUMINAMATH_CALUDE_parallel_lines_distance_l3786_378668

/-- Two parallel lines with a specified distance between them -/
structure ParallelLines where
  -- First line equation: 3x - y + 3 = 0
  l₁ : ℝ → ℝ → Prop
  l₁_def : l₁ = fun x y ↦ 3 * x - y + 3 = 0
  -- Second line equation: 3x - y + C = 0
  l₂ : ℝ → ℝ → Prop
  C : ℝ
  l₂_def : l₂ = fun x y ↦ 3 * x - y + C = 0
  -- Distance between the lines is √10
  distance : ℝ
  distance_def : distance = Real.sqrt 10

/-- The main theorem stating the possible values of C -/
theorem parallel_lines_distance (pl : ParallelLines) : pl.C = 13 ∨ pl.C = -7 := by
  sorry


end NUMINAMATH_CALUDE_parallel_lines_distance_l3786_378668


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l3786_378673

theorem geometric_sequence_fifth_term 
  (a : ℕ) (r : ℕ) (h1 : a = 4) (h2 : a * r^3 = 324) : a * r^4 = 324 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l3786_378673


namespace NUMINAMATH_CALUDE_multiple_birth_statistics_l3786_378627

theorem multiple_birth_statistics (total_babies : ℕ) 
  (h_total : total_babies = 1200) 
  (twins triplets quintuplets : ℕ) 
  (h_twins : twins = 3 * triplets) 
  (h_triplets : triplets = 2 * quintuplets) 
  (h_sum : 2 * twins + 3 * triplets + 5 * quintuplets = total_babies) : 
  5 * quintuplets = 260 := by
  sorry

end NUMINAMATH_CALUDE_multiple_birth_statistics_l3786_378627


namespace NUMINAMATH_CALUDE_sum_m_n_in_interval_l3786_378616

-- Define the function f(x)
def f (x : ℝ) : ℝ := -x^2 + 4*x

-- Define the theorem
theorem sum_m_n_in_interval (m n : ℝ) :
  (∀ x ∈ Set.Icc m n, f x ∈ Set.Icc (-5) 4) →
  (∀ y ∈ Set.Icc (-5) 4, ∃ x ∈ Set.Icc m n, f x = y) →
  m + n ∈ Set.Icc 1 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_m_n_in_interval_l3786_378616


namespace NUMINAMATH_CALUDE_abs_neg_one_third_eq_one_third_l3786_378692

theorem abs_neg_one_third_eq_one_third : |(-1/3 : ℚ)| = 1/3 := by sorry

end NUMINAMATH_CALUDE_abs_neg_one_third_eq_one_third_l3786_378692


namespace NUMINAMATH_CALUDE_factors_of_M_l3786_378651

/-- The number of natural-number factors of M, where M = 2^2 · 3^3 · 5^2 · 7^1 -/
def number_of_factors (M : ℕ) : ℕ :=
  if M = 2^2 * 3^3 * 5^2 * 7^1 then 72 else 0

/-- Theorem stating that the number of natural-number factors of M is 72 -/
theorem factors_of_M :
  number_of_factors (2^2 * 3^3 * 5^2 * 7^1) = 72 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_M_l3786_378651


namespace NUMINAMATH_CALUDE_standard_pairs_parity_l3786_378652

/-- Represents a color of a square on the chessboard -/
inductive Color
| Red
| Blue

/-- Represents a chessboard of size m × n -/
structure Chessboard (m n : ℕ) where
  cells : Fin m → Fin n → Color
  m_ge_three : m ≥ 3
  n_ge_three : n ≥ 3

/-- Counts the number of blue cells on the border of the chessboard (excluding corners) -/
def countBlueBorderCells (board : Chessboard m n) : ℕ :=
  sorry

/-- Counts the number of "standard pairs" on the chessboard -/
def countStandardPairs (board : Chessboard m n) : ℕ :=
  sorry

/-- Theorem stating the relationship between the number of standard pairs and blue border cells -/
theorem standard_pairs_parity (m n : ℕ) (board : Chessboard m n) :
  Odd (countStandardPairs board) ↔ Odd (countBlueBorderCells board) :=
sorry

end NUMINAMATH_CALUDE_standard_pairs_parity_l3786_378652


namespace NUMINAMATH_CALUDE_tangent_parallel_points_l3786_378657

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 + 1

-- Theorem statement
theorem tangent_parallel_points :
  ∀ x y : ℝ, (y = f x ∧ f' x = 4) ↔ (x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = -4) :=
by sorry

end NUMINAMATH_CALUDE_tangent_parallel_points_l3786_378657


namespace NUMINAMATH_CALUDE_q_duration_is_nine_l3786_378650

/-- Investment and profit ratios for partners P and Q -/
structure PartnershipRatios where
  investment_ratio_p : ℕ
  investment_ratio_q : ℕ
  profit_ratio_p : ℕ
  profit_ratio_q : ℕ

/-- Calculate the investment duration of partner Q given the ratios and P's duration -/
def calculate_q_duration (ratios : PartnershipRatios) (p_duration : ℕ) : ℕ :=
  (ratios.investment_ratio_p * p_duration * ratios.profit_ratio_q) / 
  (ratios.investment_ratio_q * ratios.profit_ratio_p)

/-- Theorem stating that Q's investment duration is 9 months given the specified ratios and P's duration -/
theorem q_duration_is_nine :
  let ratios : PartnershipRatios := {
    investment_ratio_p := 7,
    investment_ratio_q := 5,
    profit_ratio_p := 7,
    profit_ratio_q := 9
  }
  let p_duration := 5
  calculate_q_duration ratios p_duration = 9 := by
  sorry

end NUMINAMATH_CALUDE_q_duration_is_nine_l3786_378650


namespace NUMINAMATH_CALUDE_max_six_yuan_items_proof_l3786_378617

/-- The maximum number of 6-yuan items that can be bought given the conditions -/
def max_six_yuan_items : ℕ := 7

theorem max_six_yuan_items_proof :
  ∀ (x y z : ℕ),
    6 * x + 4 * y + 2 * z = 60 →
    x + y + z = 16 →
    x ≤ max_six_yuan_items :=
by
  sorry

#check max_six_yuan_items_proof

end NUMINAMATH_CALUDE_max_six_yuan_items_proof_l3786_378617


namespace NUMINAMATH_CALUDE_zero_of_function_l3786_378682

/-- Given a function f(x) = m + (1/3)^x where f(-2) = 0, prove that m = -9 -/
theorem zero_of_function (m : ℝ) : 
  (let f := fun x : ℝ => m + (1/3)^x; f (-2) = 0) → m = -9 := by
  sorry

end NUMINAMATH_CALUDE_zero_of_function_l3786_378682


namespace NUMINAMATH_CALUDE_expression_simplification_l3786_378670

theorem expression_simplification (p : ℝ) : 
  ((7 * p + 3) - 3 * p * 5) * 4 + (5 - 2 / 4) * (8 * p - 12) = 4 * p - 42 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3786_378670


namespace NUMINAMATH_CALUDE_painting_rate_calculation_l3786_378619

theorem painting_rate_calculation (room_length room_width room_height : ℝ)
  (door_width door_height : ℝ) (num_doors : ℕ)
  (window1_width window1_height : ℝ) (num_window1 : ℕ)
  (window2_width window2_height : ℝ) (num_window2 : ℕ)
  (total_cost : ℝ) :
  room_length = 10 ∧ room_width = 7 ∧ room_height = 5 ∧
  door_width = 1 ∧ door_height = 3 ∧ num_doors = 2 ∧
  window1_width = 2 ∧ window1_height = 1.5 ∧ num_window1 = 1 ∧
  window2_width = 1 ∧ window2_height = 1.5 ∧ num_window2 = 2 ∧
  total_cost = 474 →
  (total_cost / (2 * (room_length * room_height + room_width * room_height) -
    (num_doors * door_width * door_height +
     num_window1 * window1_width * window1_height +
     num_window2 * window2_width * window2_height))) = 3 := by
  sorry

end NUMINAMATH_CALUDE_painting_rate_calculation_l3786_378619


namespace NUMINAMATH_CALUDE_pastry_eating_time_l3786_378628

/-- The time it takes for two people to eat a certain number of pastries together -/
def eating_time (quick_rate : ℚ) (slow_rate : ℚ) (total_pastries : ℚ) : ℚ :=
  total_pastries / (quick_rate + slow_rate)

/-- Theorem stating the time it takes Miss Quick and Miss Slow to eat 5 pastries together -/
theorem pastry_eating_time :
  let quick_rate : ℚ := 1 / 15
  let slow_rate : ℚ := 1 / 25
  let total_pastries : ℚ := 5
  eating_time quick_rate slow_rate total_pastries = 375 / 8 := by
sorry

end NUMINAMATH_CALUDE_pastry_eating_time_l3786_378628


namespace NUMINAMATH_CALUDE_union_complement_equal_l3786_378681

def U : Finset ℕ := {1,2,3,4,5,6}
def M : Finset ℕ := {1,3,4}
def N : Finset ℕ := {3,5,6}

theorem union_complement_equal : M ∪ (U \ N) = {1,2,3,4} := by
  sorry

end NUMINAMATH_CALUDE_union_complement_equal_l3786_378681


namespace NUMINAMATH_CALUDE_power_sum_negative_two_l3786_378601

theorem power_sum_negative_two : (-2)^2002 + (-2)^2003 = -2^2002 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_negative_two_l3786_378601


namespace NUMINAMATH_CALUDE_mailing_cost_formula_l3786_378606

/-- The cost function for mailing a package -/
noncomputable def mailing_cost (W : ℝ) : ℝ := 8 * ⌈W / 2⌉

/-- Theorem stating the correct formula for the mailing cost -/
theorem mailing_cost_formula (W : ℝ) : 
  mailing_cost W = 8 * ⌈W / 2⌉ := by
  sorry

#check mailing_cost_formula

end NUMINAMATH_CALUDE_mailing_cost_formula_l3786_378606


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_equality_l3786_378695

theorem consecutive_integers_sum_equality (x : ℤ) (h : x = 25) :
  (x + (x + 1) + (x + 2) + (x + 3) + (x + 4) + (x + 5)) =
  ((x + 6) + (x + 7) + (x + 8) + (x + 9) + (x + 10)) := by
  sorry

#check consecutive_integers_sum_equality

end NUMINAMATH_CALUDE_consecutive_integers_sum_equality_l3786_378695


namespace NUMINAMATH_CALUDE_stream_speed_l3786_378689

/-- Given a boat that travels downstream and upstream, calculate the speed of the stream -/
theorem stream_speed (downstream_distance upstream_distance : ℝ) 
                     (downstream_time upstream_time : ℝ) 
                     (h1 : downstream_distance = 72)
                     (h2 : upstream_distance = 30)
                     (h3 : downstream_time = 3)
                     (h4 : upstream_time = 3) :
  ∃ (boat_speed stream_speed : ℝ),
    downstream_distance = (boat_speed + stream_speed) * downstream_time ∧
    upstream_distance = (boat_speed - stream_speed) * upstream_time ∧
    stream_speed = 7 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l3786_378689


namespace NUMINAMATH_CALUDE_fair_admission_collection_l3786_378683

theorem fair_admission_collection :
  let child_fee : ℚ := 3/2  -- $1.50 as a rational number
  let adult_fee : ℚ := 4    -- $4.00 as a rational number
  let total_people : ℕ := 2200
  let num_children : ℕ := 700
  let num_adults : ℕ := 1500
  
  (num_children : ℚ) * child_fee + (num_adults : ℚ) * adult_fee = 7050
  := by sorry

end NUMINAMATH_CALUDE_fair_admission_collection_l3786_378683


namespace NUMINAMATH_CALUDE_abs_plus_one_positive_l3786_378698

theorem abs_plus_one_positive (a : ℚ) : 0 < |a| + 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_plus_one_positive_l3786_378698


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3786_378602

/-- Given an arithmetic sequence {a_n} where a_1 + a_5 + a_9 = 6, prove that a_2 + a_8 = 4 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) →  -- arithmetic sequence condition
  (a 1 + a 5 + a 9 = 6) →                           -- given condition
  (a 2 + a 8 = 4) :=                                -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3786_378602


namespace NUMINAMATH_CALUDE_lowest_unique_score_l3786_378694

/-- Represents the scoring system for the national Mathematics Competition. -/
structure ScoringSystem where
  totalProblems : ℕ
  correctPoints : ℕ
  wrongPoints : ℕ
  baseScore : ℕ

/-- Calculates the score based on the number of correct and wrong answers. -/
def calculateScore (system : ScoringSystem) (correct : ℕ) (wrong : ℕ) : ℕ :=
  system.baseScore + system.correctPoints * correct - system.wrongPoints * wrong

/-- Checks if the number of correct answers can be uniquely determined from the score. -/
def isUniqueDetermination (system : ScoringSystem) (score : ℕ) : Prop :=
  ∃! correct : ℕ, ∃ wrong : ℕ,
    correct + wrong ≤ system.totalProblems ∧
    calculateScore system correct wrong = score

/-- The theorem stating that 105 is the lowest score above 100 for which
    the number of correctly solved problems can be uniquely determined. -/
theorem lowest_unique_score (system : ScoringSystem)
    (h1 : system.totalProblems = 40)
    (h2 : system.correctPoints = 5)
    (h3 : system.wrongPoints = 1)
    (h4 : system.baseScore = 40) :
    (∀ s, 100 < s → s < 105 → ¬ isUniqueDetermination system s) ∧
    isUniqueDetermination system 105 := by
  sorry

end NUMINAMATH_CALUDE_lowest_unique_score_l3786_378694


namespace NUMINAMATH_CALUDE_crayon_box_problem_l3786_378633

theorem crayon_box_problem (C R B G Y P U : ℝ) : 
  R + B + G + Y + P + U = C →
  R = 12 →
  B = 8 →
  G = (3/4) * B →
  Y = 0.15 * C →
  P = U →
  P = 0.425 * C - 13 := by
sorry

end NUMINAMATH_CALUDE_crayon_box_problem_l3786_378633


namespace NUMINAMATH_CALUDE_tangent_line_at_A_l3786_378632

/-- The function f(x) = -x^3 + 3x --/
def f (x : ℝ) := -x^3 + 3*x

/-- The derivative of f(x) --/
def f' (x : ℝ) := -3*x^2 + 3

/-- Point A --/
def A : ℝ × ℝ := (2, -2)

/-- Equation of a line passing through A with slope m --/
def line_eq (m : ℝ) (x : ℝ) : ℝ := m*(x - A.1) + A.2

/-- Theorem: The tangent line to f(x) at A is either y = -2 or 9x + y - 16 = 0 --/
theorem tangent_line_at_A : 
  (∃ x y, line_eq (f' A.1) x = y ∧ 9*x + y - 16 = 0) ∨
  (∀ x, line_eq (f' A.1) x = -2) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_A_l3786_378632


namespace NUMINAMATH_CALUDE_paving_rate_per_square_metre_l3786_378610

/-- Given a room with length 5.5 m and width 3.75 m, and a total paving cost of Rs. 28875,
    the rate of paving per square metre is Rs. 1400. -/
theorem paving_rate_per_square_metre 
  (length : ℝ) 
  (width : ℝ) 
  (total_cost : ℝ) 
  (h1 : length = 5.5)
  (h2 : width = 3.75)
  (h3 : total_cost = 28875) :
  total_cost / (length * width) = 1400 := by
sorry


end NUMINAMATH_CALUDE_paving_rate_per_square_metre_l3786_378610


namespace NUMINAMATH_CALUDE_polygon_sides_l3786_378630

theorem polygon_sides (d : ℕ) (v : ℕ) : d = 77 ∧ v = 1 → ∃ n : ℕ, n * (n - 3) / 2 = d ∧ n + v = 15 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l3786_378630


namespace NUMINAMATH_CALUDE_min_distance_circle_ellipse_l3786_378618

/-- The minimum distance between a point on a unit circle centered at the origin
    and a point on an ellipse centered at (-1, 0) with semi-major axis 3 and semi-minor axis 5 -/
theorem min_distance_circle_ellipse :
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 1}
  let ellipse := {(x, y) : ℝ × ℝ | ((x + 1)^2 / 9) + (y^2 / 25) = 1}
  ∃ d : ℝ, d = Real.sqrt 14 - 1 ∧
    ∀ (a : ℝ × ℝ) (b : ℝ × ℝ), a ∈ circle → b ∈ ellipse →
      Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) ≥ d :=
by sorry

end NUMINAMATH_CALUDE_min_distance_circle_ellipse_l3786_378618


namespace NUMINAMATH_CALUDE_cell_population_growth_l3786_378667

/-- Represents the number of cells in the population after n hours -/
def cell_count (n : ℕ) : ℕ :=
  2^(n-1) + 4

/-- The rule for cell population growth -/
def cell_growth_rule (prev : ℕ) : ℕ :=
  2 * (prev - 2)

theorem cell_population_growth (n : ℕ) :
  n > 0 →
  cell_count 1 = 5 →
  (∀ k, k ≥ 1 → cell_count (k + 1) = cell_growth_rule (cell_count k)) →
  cell_count n = 2^(n-1) + 4 :=
by
  sorry

#check cell_population_growth

end NUMINAMATH_CALUDE_cell_population_growth_l3786_378667


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l3786_378612

/-- Given a point M and two lines, this theorem proves that the second line passes through M and is perpendicular to the first line. -/
theorem perpendicular_line_through_point (x₀ y₀ a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) :
  (a₁ * x₀ + b₁ * y₀ + c₁ ≠ 0) →  -- M is not on the first line
  (a₁ * b₂ = -a₂ * b₁) →          -- Lines are perpendicular
  (a₂ * x₀ + b₂ * y₀ + c₂ = 0) →  -- Second line passes through M
  ∃ (k : ℝ), k ≠ 0 ∧ a₂ = k * 4 ∧ b₂ = k * 3 ∧ c₂ = k * (-13) ∧
             a₁ = k * 3 ∧ b₁ = k * (-4) ∧ c₁ = k * 6 ∧
             x₀ = 4 ∧ y₀ = -1 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l3786_378612


namespace NUMINAMATH_CALUDE_square_of_99_l3786_378604

theorem square_of_99 : 99 ^ 2 = 9801 := by
  sorry

end NUMINAMATH_CALUDE_square_of_99_l3786_378604


namespace NUMINAMATH_CALUDE_daniels_purchase_worth_l3786_378625

/-- The total worth of Daniel's purchases -/
def total_worth (taxable_purchase : ℝ) (tax_free_items : ℝ) : ℝ :=
  taxable_purchase + tax_free_items

/-- The amount of sales tax paid on taxable purchases -/
def sales_tax (taxable_purchase : ℝ) (tax_rate : ℝ) : ℝ :=
  taxable_purchase * tax_rate

theorem daniels_purchase_worth :
  ∃ (taxable_purchase : ℝ),
    sales_tax taxable_purchase 0.05 = 0.30 ∧
    total_worth taxable_purchase 18.7 = 24.7 := by
  sorry

end NUMINAMATH_CALUDE_daniels_purchase_worth_l3786_378625


namespace NUMINAMATH_CALUDE_amy_cupcakes_l3786_378622

theorem amy_cupcakes (todd_ate : ℕ) (packages : ℕ) (cupcakes_per_package : ℕ) 
  (h1 : todd_ate = 5)
  (h2 : packages = 9)
  (h3 : cupcakes_per_package = 5) :
  todd_ate + packages * cupcakes_per_package = 50 := by
  sorry

end NUMINAMATH_CALUDE_amy_cupcakes_l3786_378622


namespace NUMINAMATH_CALUDE_c_share_of_profit_l3786_378613

/-- Calculates the share of profit for a partner in a business partnership --/
def calculate_share_of_profit (investment : ℕ) (total_investment : ℕ) (total_profit : ℕ) : ℕ :=
  (investment * total_profit) / total_investment

theorem c_share_of_profit (investment_a investment_b investment_c total_profit : ℕ) 
  (h1 : investment_a = 27000)
  (h2 : investment_b = 72000)
  (h3 : investment_c = 81000)
  (h4 : total_profit = 80000) :
  calculate_share_of_profit investment_c (investment_a + investment_b + investment_c) total_profit = 36000 :=
by
  sorry

#eval calculate_share_of_profit 81000 (27000 + 72000 + 81000) 80000

end NUMINAMATH_CALUDE_c_share_of_profit_l3786_378613


namespace NUMINAMATH_CALUDE_two_talents_count_l3786_378677

def num_students : ℕ := 120

def num_cant_sing : ℕ := 50
def num_cant_dance : ℕ := 75
def num_cant_act : ℕ := 35

def num_can_sing : ℕ := num_students - num_cant_sing
def num_can_dance : ℕ := num_students - num_cant_dance
def num_can_act : ℕ := num_students - num_cant_act

theorem two_talents_count :
  ∀ (x : ℕ),
    x ≤ num_students →
    (num_can_sing + num_can_dance + num_can_act) - (num_students - x) = 80 + x →
    x = 0 →
    (num_can_sing + num_can_dance + num_can_act) - num_students = 80 :=
by sorry

end NUMINAMATH_CALUDE_two_talents_count_l3786_378677


namespace NUMINAMATH_CALUDE_sixth_student_stickers_l3786_378672

def sticker_sequence (n : ℕ) : ℕ :=
  29 + 6 * (n - 1)

theorem sixth_student_stickers : sticker_sequence 6 = 59 := by
  sorry

end NUMINAMATH_CALUDE_sixth_student_stickers_l3786_378672


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3786_378666

-- Problem 1
theorem problem_1 (a : ℝ) : (-a^2)^3 + (-2*a^3)^2 - a^3 * a^2 = 3*a^6 - a^5 := by
  sorry

-- Problem 2
theorem problem_2 (x y : ℝ) : ((x + 2*y) * (x - 2*y) + 4*(x - y)^2) + 6*x = 5*x^2 - 8*x*y + 6*x := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3786_378666


namespace NUMINAMATH_CALUDE_two_books_different_genres_l3786_378697

theorem two_books_different_genres (n : ℕ) (h : n = 4) : 
  (n.choose 2) * n * n = 96 :=
by sorry

end NUMINAMATH_CALUDE_two_books_different_genres_l3786_378697


namespace NUMINAMATH_CALUDE_scientific_notation_of_nanometer_l3786_378644

def nanometer : ℝ := 0.000000001

theorem scientific_notation_of_nanometer :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ nanometer = a * (10 : ℝ) ^ n ∧ a = 1 ∧ n = -9 :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_nanometer_l3786_378644


namespace NUMINAMATH_CALUDE_gcd_lcm_8951_4267_l3786_378686

theorem gcd_lcm_8951_4267 : 
  (Nat.gcd 8951 4267 = 1) ∧ 
  (Nat.lcm 8951 4267 = 38212917) := by
sorry

end NUMINAMATH_CALUDE_gcd_lcm_8951_4267_l3786_378686


namespace NUMINAMATH_CALUDE_binomial_floor_divisibility_l3786_378620

theorem binomial_floor_divisibility (p n : ℕ) (hp : Prime p) (hn : n ≥ p) :
  p ∣ (Nat.choose n p - n / p) := by
  sorry

end NUMINAMATH_CALUDE_binomial_floor_divisibility_l3786_378620


namespace NUMINAMATH_CALUDE_survey_total_students_l3786_378643

theorem survey_total_students : 
  let mac_preference : ℕ := 60
  let both_preference : ℕ := mac_preference / 3
  let no_preference : ℕ := 90
  let windows_preference : ℕ := 40
  mac_preference + both_preference + no_preference + windows_preference = 210 := by
sorry

end NUMINAMATH_CALUDE_survey_total_students_l3786_378643


namespace NUMINAMATH_CALUDE_probability_at_least_one_defective_l3786_378646

theorem probability_at_least_one_defective (total : ℕ) (defective : ℕ) : 
  total = 20 → defective = 4 → 
  (1 - (total - defective) * (total - defective - 1) / (total * (total - 1))) = 7 / 19 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_defective_l3786_378646


namespace NUMINAMATH_CALUDE_painting_cost_after_modification_l3786_378634

/-- Represents the dimensions of a rectangular surface -/
structure Dimensions where
  length : Float
  width : Float

/-- Calculates the area of a rectangular surface -/
def area (d : Dimensions) : Float :=
  d.length * d.width

/-- Represents a room with walls, windows, and doors -/
structure Room where
  walls : List Dimensions
  windows : List Dimensions
  doors : List Dimensions

/-- Calculates the total wall area of a room -/
def totalWallArea (r : Room) : Float :=
  r.walls.map area |> List.sum

/-- Calculates the total area of openings (windows and doors) in a room -/
def totalOpeningArea (r : Room) : Float :=
  (r.windows.map area |> List.sum) + (r.doors.map area |> List.sum)

/-- Calculates the net paintable area of a room -/
def netPaintableArea (r : Room) : Float :=
  totalWallArea r - totalOpeningArea r

/-- Calculates the cost to paint a room given the cost per square foot -/
def paintingCost (r : Room) (costPerSqFt : Float) : Float :=
  netPaintableArea r * costPerSqFt

/-- Increases the dimensions of a room by a given factor -/
def increaseRoomSize (r : Room) (factor : Float) : Room :=
  { walls := r.walls.map fun d => { length := d.length * factor, width := d.width * factor },
    windows := r.windows,
    doors := r.doors }

/-- Adds additional windows and doors to a room -/
def addOpenings (r : Room) (additionalWindows : List Dimensions) (additionalDoors : List Dimensions) : Room :=
  { walls := r.walls,
    windows := r.windows ++ additionalWindows,
    doors := r.doors ++ additionalDoors }

theorem painting_cost_after_modification (originalRoom : Room) (costPerSqFt : Float) : 
  let modifiedRoom := addOpenings (increaseRoomSize originalRoom 1.5) 
                        [⟨3, 4⟩, ⟨3, 4⟩] [⟨3, 7⟩]
  paintingCost modifiedRoom costPerSqFt = 1732.50 :=
by
  sorry

#check painting_cost_after_modification

end NUMINAMATH_CALUDE_painting_cost_after_modification_l3786_378634


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3786_378615

theorem sufficient_not_necessary_condition :
  (∀ x : ℝ, 1 < x ∧ x < 2 → x < 2) ∧
  (∃ x : ℝ, x < 2 ∧ ¬(1 < x ∧ x < 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3786_378615


namespace NUMINAMATH_CALUDE_rachels_homework_l3786_378656

/-- Rachel's homework problem -/
theorem rachels_homework (reading_homework : ℕ) (math_homework : ℕ) : 
  reading_homework = 2 → math_homework = reading_homework + 7 → math_homework = 9 :=
by
  sorry


end NUMINAMATH_CALUDE_rachels_homework_l3786_378656


namespace NUMINAMATH_CALUDE_min_value_theorem_l3786_378637

theorem min_value_theorem (a b : ℝ) (h1 : a + 2*b = 2) (h2 : a > 1) (h3 : b > 0) :
  (∀ x y : ℝ, x > 1 ∧ y > 0 ∧ x + 2*y = 2 → 2/(x-1) + 1/y ≥ 2/(a-1) + 1/b) ∧
  2/(a-1) + 1/b = 8 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3786_378637


namespace NUMINAMATH_CALUDE_regular_hexagon_dimensions_l3786_378626

/-- Regular hexagon with given area and side lengths -/
structure RegularHexagon where
  area : ℝ
  x : ℝ
  y : ℝ
  area_eq : area = 54 * Real.sqrt 3
  side_length : x > 0
  diagonal_length : y > 0

/-- Theorem: For a regular hexagon with area 54√3 cm², if AB = x cm and AC = y√3 cm, then x = 6 and y = 6 -/
theorem regular_hexagon_dimensions (h : RegularHexagon) : h.x = 6 ∧ h.y = 6 := by
  sorry

end NUMINAMATH_CALUDE_regular_hexagon_dimensions_l3786_378626


namespace NUMINAMATH_CALUDE_min_value_of_squared_ratios_l3786_378631

theorem min_value_of_squared_ratios (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / b)^2 + (b / c)^2 + (c / a)^2 ≥ 3 ∧
  ((a / b)^2 + (b / c)^2 + (c / a)^2 = 3 ↔ a = b ∧ b = c) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_squared_ratios_l3786_378631


namespace NUMINAMATH_CALUDE_total_assembly_time_l3786_378696

def chairs : ℕ := 7
def tables : ℕ := 3
def bookshelves : ℕ := 2
def lamps : ℕ := 4

def chair_time : ℕ := 4
def table_time : ℕ := 8
def bookshelf_time : ℕ := 12
def lamp_time : ℕ := 2

theorem total_assembly_time :
  chairs * chair_time + tables * table_time + bookshelves * bookshelf_time + lamps * lamp_time = 84 := by
  sorry

end NUMINAMATH_CALUDE_total_assembly_time_l3786_378696


namespace NUMINAMATH_CALUDE_roots_are_irrational_l3786_378665

-- Define the quadratic equation
def quadratic (k : ℝ) (x : ℝ) : ℝ := x^2 - 4*k*x + 3*k^2 - 2

-- Define the discriminant of the quadratic equation
def discriminant (k : ℝ) : ℝ := (4*k)^2 - 4*(3*k^2 - 2)

-- Theorem statement
theorem roots_are_irrational (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ quadratic k x = 0 ∧ quadratic k y = 0 ∧ x * y = 10) →
  ∃ x y : ℝ, x ≠ y ∧ quadratic k x = 0 ∧ quadratic k y = 0 ∧ ¬(∃ q : ℚ, x = ↑q) ∧ ¬(∃ q : ℚ, y = ↑q) :=
by sorry

end NUMINAMATH_CALUDE_roots_are_irrational_l3786_378665


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3786_378687

theorem inequality_solution_set :
  {x : ℝ | 1 + x > 6 - 4 * x} = {x : ℝ | x > 1} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3786_378687


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l3786_378655

/-- Represents the repeating decimal 0.35247̄ -/
def repeating_decimal : ℚ :=
  35247 / 100000 + (247 / 100000) / (1 - 1 / 1000)

/-- The fraction we want to prove equality with -/
def target_fraction : ℚ := 3518950 / 999900

/-- Theorem stating that the repeating decimal equals the target fraction -/
theorem repeating_decimal_equals_fraction :
  repeating_decimal = target_fraction := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l3786_378655


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_product_l3786_378641

/-- Given that -9, a, -1 form an arithmetic sequence and 
    -9, m, b, n, -1 form a geometric sequence, prove that ab = 15 -/
theorem arithmetic_geometric_sequence_product (a m b n : ℝ) : 
  ((-9 + (-1)) / 2 = a) →  -- arithmetic sequence condition
  ((-1 / -9) ^ (1/4) = (-1 / -9) ^ (1/4)) →  -- geometric sequence condition
  a * b = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_product_l3786_378641


namespace NUMINAMATH_CALUDE_symmetrical_point_y_axis_l3786_378679

/-- Represents a point in the 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the y-axis -/
def reflect_y_axis (p : Point) : Point :=
  ⟨-p.x, p.y⟩

theorem symmetrical_point_y_axis :
  let A : Point := ⟨-1, 2⟩
  reflect_y_axis A = ⟨1, 2⟩ := by
  sorry

end NUMINAMATH_CALUDE_symmetrical_point_y_axis_l3786_378679


namespace NUMINAMATH_CALUDE_sequence_problem_l3786_378684

/-- Given a sequence {a_n} with sum S_n = kn^2 + n and a_10 = 39, prove a_100 = 399 -/
theorem sequence_problem (k : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, S n = k * n^2 + n) →
  a 10 = 39 →
  a 100 = 399 := by
sorry

end NUMINAMATH_CALUDE_sequence_problem_l3786_378684


namespace NUMINAMATH_CALUDE_letters_in_mailboxes_l3786_378691

theorem letters_in_mailboxes :
  (number_of_ways : ℕ) →
  (number_of_letters : ℕ) →
  (number_of_mailboxes : ℕ) →
  (number_of_letters = 4) →
  (number_of_mailboxes = 3) →
  (number_of_ways = number_of_mailboxes ^ number_of_letters) :=
by sorry

end NUMINAMATH_CALUDE_letters_in_mailboxes_l3786_378691


namespace NUMINAMATH_CALUDE_consecutive_even_numbers_divisible_by_eight_l3786_378635

theorem consecutive_even_numbers_divisible_by_eight (n : ℤ) : 
  ∃ k : ℤ, 4 * n * (n + 1) = 8 * k := by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_numbers_divisible_by_eight_l3786_378635


namespace NUMINAMATH_CALUDE_total_balls_is_seven_l3786_378649

/-- The number of balls in the first box -/
def box1_balls : ℕ := 3

/-- The number of balls in the second box -/
def box2_balls : ℕ := 4

/-- The total number of balls in both boxes -/
def total_balls : ℕ := box1_balls + box2_balls

/-- Theorem stating that the total number of balls is 7 -/
theorem total_balls_is_seven : total_balls = 7 := by
  sorry

end NUMINAMATH_CALUDE_total_balls_is_seven_l3786_378649


namespace NUMINAMATH_CALUDE_parallel_vectors_l3786_378688

/-- Given two vectors AB and CD in R², if AB is parallel to CD and AB = (6,1) and CD = (x,-3), then x = -18 -/
theorem parallel_vectors (AB CD : ℝ × ℝ) (x : ℝ) 
  (h1 : AB = (6, 1)) 
  (h2 : CD = (x, -3)) 
  (h3 : ∃ (k : ℝ), k ≠ 0 ∧ AB = k • CD) : 
  x = -18 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_l3786_378688


namespace NUMINAMATH_CALUDE_plant_arrangement_count_l3786_378614

/-- Represents the number of basil plants -/
def num_basil : ℕ := 4

/-- Represents the number of tomato plants -/
def num_tomato : ℕ := 4

/-- Calculates the factorial of a natural number -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Theorem stating the number of ways to arrange the plants -/
theorem plant_arrangement_count : 
  factorial (num_basil + 1) * factorial num_tomato = 2880 := by
  sorry

end NUMINAMATH_CALUDE_plant_arrangement_count_l3786_378614


namespace NUMINAMATH_CALUDE_oliver_new_cards_l3786_378662

/-- Calculates the number of new baseball cards Oliver had -/
def new_cards (cards_per_page : ℕ) (total_pages : ℕ) (old_cards : ℕ) : ℕ :=
  cards_per_page * total_pages - old_cards

/-- Proves that Oliver had 2 new baseball cards -/
theorem oliver_new_cards : new_cards 3 4 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_oliver_new_cards_l3786_378662


namespace NUMINAMATH_CALUDE_unique_solution_equation_l3786_378669

theorem unique_solution_equation : ∃! x : ℝ, 3 * x - 8 - 2 = x := by sorry

end NUMINAMATH_CALUDE_unique_solution_equation_l3786_378669


namespace NUMINAMATH_CALUDE_area_BQW_is_48_l3786_378638

/-- Rectangle ABCD with specific measurements and areas -/
structure Rectangle where
  AB : ℝ
  AZ : ℝ
  WC : ℝ
  area_ZWCD : ℝ
  h : AB = 16
  h' : AZ = 8
  h'' : WC = 8
  h''' : area_ZWCD = 160

/-- The area of triangle BQW in the given rectangle -/
def area_BQW (r : Rectangle) : ℝ := 48

/-- Theorem stating that the area of triangle BQW is 48 square units -/
theorem area_BQW_is_48 (r : Rectangle) : area_BQW r = 48 := by
  sorry

end NUMINAMATH_CALUDE_area_BQW_is_48_l3786_378638


namespace NUMINAMATH_CALUDE_perfect_square_floor_equality_l3786_378648

theorem perfect_square_floor_equality (n : ℕ+) :
  ⌊2 * Real.sqrt n⌋ = ⌊Real.sqrt (n - 1) + Real.sqrt (n + 1)⌋ + 1 ↔ ∃ m : ℕ+, n = m^2 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_floor_equality_l3786_378648


namespace NUMINAMATH_CALUDE_min_value_a_k_l3786_378661

/-- A positive arithmetic sequence satisfying the given condition -/
def PositiveArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧
  (∃ d, ∀ n, a (n + 1) = a n + d) ∧
  (∀ k : ℕ, k ≥ 2 → 1 / a 1 + 4 / a (2 * k - 1) ≤ 1)

/-- The theorem stating the minimum value of a_k -/
theorem min_value_a_k (a : ℕ → ℝ) (h : PositiveArithmeticSequence a) :
    ∀ k : ℕ, k ≥ 2 → a k ≥ 9/2 :=
  sorry

end NUMINAMATH_CALUDE_min_value_a_k_l3786_378661


namespace NUMINAMATH_CALUDE_min_value_expression_l3786_378671

theorem min_value_expression (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
  (5 * r) / (3 * p + 2 * q) + (5 * p) / (2 * q + 3 * r) + (2 * q) / (p + r) ≥ 151 / 18 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3786_378671


namespace NUMINAMATH_CALUDE_a_values_l3786_378603

def A (a : ℝ) : Set ℝ := {1, 2, a^2 - 3*a - 1}
def B : Set ℝ := {1, 3}

theorem a_values (a : ℝ) : (A a ∩ B = {1, 3}) → (a = -1 ∨ a = 4) := by
  sorry

end NUMINAMATH_CALUDE_a_values_l3786_378603


namespace NUMINAMATH_CALUDE_horner_method_v₄_l3786_378605

-- Define the polynomial coefficients
def a₀ : ℝ := 12
def a₁ : ℝ := 35
def a₂ : ℝ := -8
def a₃ : ℝ := 79
def a₄ : ℝ := 6
def a₅ : ℝ := 5
def a₆ : ℝ := 3

-- Define x
def x : ℝ := -4

-- Define v₄ using Horner's method
def v₄ : ℝ := ((((a₆ * x + a₅) * x + a₄) * x + a₃) * x + a₂) * x + a₁

-- Theorem statement
theorem horner_method_v₄ : v₄ = 220 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_v₄_l3786_378605


namespace NUMINAMATH_CALUDE_geometric_sequence_ak_l3786_378623

/-- Given a geometric sequence {a_n} with sum S_n = k * 2^n - 3, prove a_k = 12 -/
theorem geometric_sequence_ak (k : ℕ) (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, S n = k * 2^n - 3) →
  (∀ n, S (n + 1) - S n = a (n + 1)) →
  (∀ n, a (n + 1) = 2 * a n) →
  a k = 12 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_ak_l3786_378623


namespace NUMINAMATH_CALUDE_abes_age_l3786_378654

theorem abes_age :
  ∀ (present_age : ℕ), 
    (present_age + (present_age - 7) = 37) → 
    present_age = 22 := by
  sorry

end NUMINAMATH_CALUDE_abes_age_l3786_378654


namespace NUMINAMATH_CALUDE_fraction_inequality_l3786_378640

theorem fraction_inequality (a b m : ℝ) (ha : 0 < a) (hb : 0 < b) (hm : 0 < m) (hab : a < b) :
  (b + m) / (a + m) < b / a := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l3786_378640


namespace NUMINAMATH_CALUDE_jump_rope_median_and_mode_l3786_378611

def jump_rope_scores : List ℕ := [129, 130, 130, 130, 132, 132, 135, 135, 137, 137]

def median (scores : List ℕ) : ℚ := sorry

def mode (scores : List ℕ) : ℕ := sorry

theorem jump_rope_median_and_mode :
  median jump_rope_scores = 132 ∧ mode jump_rope_scores = 130 := by sorry

end NUMINAMATH_CALUDE_jump_rope_median_and_mode_l3786_378611


namespace NUMINAMATH_CALUDE_pebble_collection_sum_l3786_378639

def geometric_sum (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

theorem pebble_collection_sum : geometric_sum 2 2 10 = 2046 := by
  sorry

end NUMINAMATH_CALUDE_pebble_collection_sum_l3786_378639


namespace NUMINAMATH_CALUDE_probability_same_color_is_19_39_l3786_378645

def num_green_balls : ℕ := 5
def num_white_balls : ℕ := 8

def total_balls : ℕ := num_green_balls + num_white_balls

def probability_same_color : ℚ :=
  (Nat.choose num_green_balls 2 + Nat.choose num_white_balls 2) / Nat.choose total_balls 2

theorem probability_same_color_is_19_39 :
  probability_same_color = 19 / 39 := by
  sorry

end NUMINAMATH_CALUDE_probability_same_color_is_19_39_l3786_378645


namespace NUMINAMATH_CALUDE_ladder_problem_l3786_378624

theorem ladder_problem (ladder_length height : ℝ) 
  (h1 : ladder_length = 13)
  (h2 : height = 12) :
  ∃ (base : ℝ), base^2 + height^2 = ladder_length^2 ∧ base = 5 := by
  sorry

end NUMINAMATH_CALUDE_ladder_problem_l3786_378624


namespace NUMINAMATH_CALUDE_M_subset_N_l3786_378653

-- Define the set M
def M : Set ℝ := {x | ∃ k : ℤ, x = (k / 2 : ℝ) * 180 + 45}

-- Define the set N
def N : Set ℝ := {x | ∃ k : ℤ, x = (k / 4 : ℝ) * 180 + 45}

-- Theorem statement
theorem M_subset_N : M ⊆ N := by
  sorry

end NUMINAMATH_CALUDE_M_subset_N_l3786_378653


namespace NUMINAMATH_CALUDE_f_property_l3786_378693

def f (x : ℝ) : ℝ := x * |x|

theorem f_property : ∀ x : ℝ, f (Real.sqrt 2 * x) = 2 * f x := by
  sorry

end NUMINAMATH_CALUDE_f_property_l3786_378693


namespace NUMINAMATH_CALUDE_sum_of_tenth_set_l3786_378636

/-- Calculates the sum of the first n triangular numbers -/
def sumOfTriangularNumbers (n : ℕ) : ℕ := n * (n + 1) * (n + 2) / 6

/-- Calculates the first element of the nth set -/
def firstElementOfSet (n : ℕ) : ℕ := sumOfTriangularNumbers (n - 1) + 1

/-- Calculates the number of elements in the nth set -/
def numberOfElementsInSet (n : ℕ) : ℕ := n + 2 * (n - 1)

/-- Calculates the last element of the nth set -/
def lastElementOfSet (n : ℕ) : ℕ := firstElementOfSet n + numberOfElementsInSet n - 1

/-- Calculates the sum of elements in the nth set -/
def sumOfSet (n : ℕ) : ℕ := 
  (numberOfElementsInSet n * (firstElementOfSet n + lastElementOfSet n)) / 2

theorem sum_of_tenth_set : sumOfSet 10 = 5026 := by sorry

end NUMINAMATH_CALUDE_sum_of_tenth_set_l3786_378636


namespace NUMINAMATH_CALUDE_treehouse_paint_calculation_l3786_378685

/-- The amount of paint needed for a treehouse project, including paint loss. -/
def total_paint_needed (white_paint green_paint brown_paint blue_paint : Real)
  (paint_loss_percentage : Real) (oz_to_liter_conversion : Real) : Real :=
  let total_oz := white_paint + green_paint + brown_paint + blue_paint
  let total_oz_with_loss := total_oz * (1 + paint_loss_percentage)
  total_oz_with_loss * oz_to_liter_conversion

/-- Theorem stating the total amount of paint needed is approximately 2.635 liters. -/
theorem treehouse_paint_calculation :
  let white_paint := 20
  let green_paint := 15
  let brown_paint := 34
  let blue_paint := 12
  let paint_loss_percentage := 0.1
  let oz_to_liter_conversion := 0.0295735
  ∃ ε > 0, |total_paint_needed white_paint green_paint brown_paint blue_paint
    paint_loss_percentage oz_to_liter_conversion - 2.635| < ε :=
by sorry

end NUMINAMATH_CALUDE_treehouse_paint_calculation_l3786_378685


namespace NUMINAMATH_CALUDE_sum_of_three_circles_l3786_378621

theorem sum_of_three_circles (square circle : ℝ) 
  (eq1 : 3 * square + 2 * circle = 27)
  (eq2 : 2 * square + 3 * circle = 25) : 
  3 * circle = 12.6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_circles_l3786_378621


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l3786_378642

-- Define the universal set U
def U : Set ℝ := {x : ℝ | -Real.sqrt 3 < x}

-- Define set A
def A : Set ℝ := {x : ℝ | 1 < 4 - x^2 ∧ 4 - x^2 ≤ 2}

-- State the theorem
theorem complement_of_A_in_U :
  (U \ A) = {x : ℝ | (-Real.sqrt 3 < x ∧ x < -Real.sqrt 2) ∨ (Real.sqrt 2 < x)} :=
by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l3786_378642


namespace NUMINAMATH_CALUDE_inequality_solution_l3786_378660

theorem inequality_solution (x : ℝ) :
  (2 ≤ |3*x - 6| ∧ |3*x - 6| ≤ 12) ↔ (x ∈ Set.Icc (-2) (4/3) ∪ Set.Icc (8/3) 6) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3786_378660


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3786_378629

theorem necessary_but_not_sufficient :
  ∀ a : ℝ,
  (∀ x : ℝ, x^2 + 1 > a) →
  (∃ b : ℝ, b > 0 ∧ b ≠ 1 ∧ (∀ x y : ℝ, x < y → b^x > b^y)) ∧
  (∃ c : ℝ, (∀ x : ℝ, x^2 + 1 > c) ∧ 
   ¬(∀ x y : ℝ, x < y → c^x > c^y)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3786_378629


namespace NUMINAMATH_CALUDE_origin_outside_circle_l3786_378607

theorem origin_outside_circle (a : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + 2*y + a - 2 = 0 → (x^2 + y^2 > 0)) ↔ (2 < a ∧ a < 3) :=
by sorry

end NUMINAMATH_CALUDE_origin_outside_circle_l3786_378607


namespace NUMINAMATH_CALUDE_pyramid_face_area_l3786_378699

theorem pyramid_face_area (base_edge : ℝ) (lateral_edge : ℝ) 
  (h_base : base_edge = 8) 
  (h_lateral : lateral_edge = 7) : 
  4 * (1/2 * base_edge * Real.sqrt (lateral_edge^2 - (base_edge/2)^2)) = 16 * Real.sqrt 33 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_face_area_l3786_378699


namespace NUMINAMATH_CALUDE_coordinates_of_B_l3786_378675

/-- Given a line segment AB parallel to the y-axis, with A(1, -2) and AB = 8,
    the coordinates of B are either (1, -10) or (1, 6). -/
theorem coordinates_of_B (A B : ℝ × ℝ) : 
  A = (1, -2) →
  (B.1 = A.1) →
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 8 →
  (B = (1, -10) ∨ B = (1, 6)) :=
by sorry

end NUMINAMATH_CALUDE_coordinates_of_B_l3786_378675


namespace NUMINAMATH_CALUDE_impossibleTable_l3786_378647

/-- Represents a digit from 1 to 9 -/
def Digit := Fin 9

/-- Represents a 10x10 table of digits -/
def Table := Fin 10 → Fin 10 → Digit

/-- Converts a sequence of 10 digits to a natural number -/
def toNumber (seq : Fin 10 → Digit) : ℕ := sorry

/-- The main theorem stating the impossibility of constructing the required table -/
theorem impossibleTable : ¬ ∃ (t : Table),
  (∀ i : Fin 10, toNumber (λ j => t i j) > toNumber (λ k => t k k)) ∧
  (∀ j : Fin 10, toNumber (λ k => t k k) > toNumber (λ i => t i j)) := by
  sorry


end NUMINAMATH_CALUDE_impossibleTable_l3786_378647
