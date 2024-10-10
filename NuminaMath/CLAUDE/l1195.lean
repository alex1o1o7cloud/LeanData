import Mathlib

namespace polynomial_roots_l1195_119589

theorem polynomial_roots : 
  let p (x : ℝ) := 10*x^4 - 55*x^3 + 96*x^2 - 55*x + 10
  ∀ x : ℝ, p x = 0 ↔ (x = 2 ∨ x = 1/2 ∨ x = 1) :=
by sorry

end polynomial_roots_l1195_119589


namespace complex_absolute_value_l1195_119595

theorem complex_absolute_value (ω : ℂ) (h : ω = 7 + 4 * Complex.I) :
  Complex.abs (ω^2 + 10*ω + 88) = Real.sqrt 313 * 13 := by
  sorry

end complex_absolute_value_l1195_119595


namespace necessary_condition_for_124_l1195_119553

/-- A line in the form y = (m/n)x - 1/n -/
structure Line where
  m : ℝ
  n : ℝ
  n_nonzero : n ≠ 0

/-- Predicate for a line passing through the first, second, and fourth quadrants -/
def passes_through_124 (l : Line) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ x₄ y₄ : ℝ),
    x₁ > 0 ∧ y₁ > 0 ∧  -- First quadrant
    x₂ < 0 ∧ y₂ > 0 ∧  -- Second quadrant
    x₄ > 0 ∧ y₄ < 0 ∧  -- Fourth quadrant
    y₁ = (l.m / l.n) * x₁ - 1 / l.n ∧
    y₂ = (l.m / l.n) * x₂ - 1 / l.n ∧
    y₄ = (l.m / l.n) * x₄ - 1 / l.n

/-- Theorem stating the necessary condition -/
theorem necessary_condition_for_124 (l : Line) :
  passes_through_124 l → l.m > 0 ∧ l.n < 0 :=
by sorry

end necessary_condition_for_124_l1195_119553


namespace complex_product_real_l1195_119530

theorem complex_product_real (a : ℝ) : 
  let z₁ : ℂ := 3 + a * Complex.I
  let z₂ : ℂ := a - 3 * Complex.I
  (z₁ * z₂).im = 0 ↔ a = 3 ∨ a = -3 := by
sorry

end complex_product_real_l1195_119530


namespace negative_fraction_comparison_l1195_119551

theorem negative_fraction_comparison : -3/4 > -4/5 := by
  sorry

end negative_fraction_comparison_l1195_119551


namespace initial_workers_correct_l1195_119540

/-- Represents the initial number of workers -/
def initial_workers : ℕ := 120

/-- Represents the total number of days to complete the wall -/
def total_days : ℕ := 50

/-- Represents the number of days after which progress is measured -/
def progress_days : ℕ := 25

/-- Represents the fraction of work completed after progress_days -/
def work_completed : ℚ := 2/5

/-- Represents the additional workers needed to complete on time -/
def additional_workers : ℕ := 30

/-- Proves that the initial number of workers is correct given the conditions -/
theorem initial_workers_correct :
  initial_workers * total_days = (initial_workers + additional_workers) * 
    (total_days * work_completed + progress_days * (1 - work_completed)) :=
by sorry

end initial_workers_correct_l1195_119540


namespace system_solution_value_l1195_119563

theorem system_solution_value (x y m : ℝ) : 
  (2 * x + 6 * y = 25) →
  (6 * x + 2 * y = -11) →
  (x - y = m - 1) →
  m = -8 := by
  sorry

end system_solution_value_l1195_119563


namespace arithmetic_calculations_l1195_119587

theorem arithmetic_calculations : 
  ((-8) + 10 - 2 + (-1) = -1) ∧ 
  (12 - 7 * (-4) + 8 / (-2) = 36) ∧ 
  ((1/2 + 1/3 - 1/6) / (-1/18) = -12) ∧ 
  (-1^4 - (1 + 0.5) * (1/3) * (-4)^2 = -33/32) := by
  sorry

#eval (-8) + 10 - 2 + (-1)
#eval 12 - 7 * (-4) + 8 / (-2)
#eval (1/2 + 1/3 - 1/6) / (-1/18)
#eval -1^4 - (1 + 0.5) * (1/3) * (-4)^2

end arithmetic_calculations_l1195_119587


namespace sphere_to_hemisphere_volume_ratio_l1195_119513

/-- The ratio of the volume of a sphere to the volume of a hemisphere -/
theorem sphere_to_hemisphere_volume_ratio (r : ℝ) (r_pos : r > 0) :
  (4 / 3 * Real.pi * r^3) / (1 / 2 * 4 / 3 * Real.pi * (3 * r)^3) = 1 / 13.5 := by
  sorry

#check sphere_to_hemisphere_volume_ratio

end sphere_to_hemisphere_volume_ratio_l1195_119513


namespace sqrt_equation_solution_l1195_119562

theorem sqrt_equation_solution (x : ℝ) :
  (3 * x - 2 > 0) →
  (Real.sqrt (3 * x - 2) + 9 / Real.sqrt (3 * x - 2) = 6) ↔
  (x = 11 / 3) :=
by sorry

end sqrt_equation_solution_l1195_119562


namespace arithmetic_sequence_eighth_term_l1195_119525

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_eighth_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 4 + a 6 + a 8 + a 10 + a 12 = 120) :
  a 8 = 24 :=
sorry

end arithmetic_sequence_eighth_term_l1195_119525


namespace student_distribution_l1195_119533

/-- The number of ways to distribute n students between two cities --/
def distribute (n : ℕ) (min1 min2 : ℕ) : ℕ :=
  (Finset.range (n - min1 - min2 + 1)).sum (λ k => Nat.choose n (min1 + k))

/-- The theorem stating the number of arrangements for 6 students --/
theorem student_distribution : distribute 6 2 3 = 35 := by
  sorry

end student_distribution_l1195_119533


namespace smallest_n_no_sum_of_powers_is_square_l1195_119591

theorem smallest_n_no_sum_of_powers_is_square : ∃ (n : ℕ), n > 1 ∧
  (∀ (m k : ℕ), ¬∃ (a : ℕ), n^m + n^k = a^2) ∧
  (∀ (n' : ℕ), 1 < n' ∧ n' < n →
    ∃ (m k a : ℕ), n'^m + n'^k = a^2) :=
by
  -- The proof goes here
  sorry

end smallest_n_no_sum_of_powers_is_square_l1195_119591


namespace percentage_of_number_l1195_119545

theorem percentage_of_number (x : ℝ) (y : ℝ) (z : ℝ) : 
  x = (y / 100) * z → x = 120 ∧ y = 150 ∧ z = 80 :=
by sorry

end percentage_of_number_l1195_119545


namespace project_hours_difference_l1195_119537

/-- 
Given a project where:
- The total hours charged is 180
- Pat charged twice as much time as Kate
- Pat charged 1/3 as much time as Mark

Prove that Mark charged 100 more hours than Kate.
-/
theorem project_hours_difference (kate : ℝ) (pat : ℝ) (mark : ℝ) 
  (h1 : kate + pat + mark = 180)
  (h2 : pat = 2 * kate)
  (h3 : pat = (1/3) * mark) : 
  mark - kate = 100 := by
  sorry

end project_hours_difference_l1195_119537


namespace bug_probability_after_10_moves_l1195_119546

/-- Probability of the bug being at the starting vertex after n moves -/
def Q : ℕ → ℚ
  | 0 => 1
  | n+1 => (1/3) * (1 - Q n)

/-- The probability of the bug returning to its starting vertex on a square after 10 moves is 3431/19683 -/
theorem bug_probability_after_10_moves :
  Q 10 = 3431 / 19683 := by
  sorry

end bug_probability_after_10_moves_l1195_119546


namespace proposition_1_proposition_2_not_always_true_proposition_3_proposition_4_l1195_119580

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicularLP : Line → Plane → Prop)
variable (perpendicularPP : Plane → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (subset : Line → Plane → Prop)

-- Define the lines and planes
variable (a b : Line)
variable (α β : Plane)

-- Assume the lines and planes are distinct
variable (h_distinct_lines : a ≠ b)
variable (h_distinct_planes : α ≠ β)

-- Proposition 1
theorem proposition_1 : 
  perpendicular a b → perpendicularLP a α → ¬contains α b → parallel b α :=
sorry

-- Proposition 2 (not necessarily true)
theorem proposition_2_not_always_true : 
  ¬(∀ (a : Line) (α β : Plane), parallel a α → perpendicularPP α β → perpendicularLP a β) :=
sorry

-- Proposition 3
theorem proposition_3 : 
  perpendicularLP a β → perpendicularPP α β → (parallel a α ∨ subset a α) :=
sorry

-- Proposition 4
theorem proposition_4 : 
  perpendicular a b → perpendicularLP a α → perpendicularLP b β → perpendicularPP α β :=
sorry

end proposition_1_proposition_2_not_always_true_proposition_3_proposition_4_l1195_119580


namespace quadratic_inequality_l1195_119527

theorem quadratic_inequality (a b c : ℝ) 
  (h : ∀ x, a * x^2 + b * x + c < 0) : 
  b / a < c / a + 1 := by
  sorry

end quadratic_inequality_l1195_119527


namespace nested_root_equality_l1195_119578

theorem nested_root_equality (a b c : ℕ) (h1 : a > 1) (h2 : b > 1) (h3 : c > 1) :
  (∀ N : ℝ, N ≠ 1 → (N^(1/a + 1/(a*b) + 1/(a*b*c)) = N^(15/24))) →
  c = 8 := by
sorry

end nested_root_equality_l1195_119578


namespace kyuhyung_cards_l1195_119550

/-- The number of cards in Kyuhyung's possession -/
def total_cards : ℕ := 103

/-- The side length of the square arrangement -/
def side_length : ℕ := 10

/-- The number of cards left over after forming the square -/
def leftover_cards : ℕ := 3

/-- The number of additional cards needed to fill the outer perimeter -/
def perimeter_cards : ℕ := 44

theorem kyuhyung_cards :
  total_cards = side_length^2 + leftover_cards ∧
  (side_length + 2)^2 - side_length^2 = perimeter_cards :=
by sorry

end kyuhyung_cards_l1195_119550


namespace cocktail_theorem_l1195_119567

def cocktail_proof (initial_volume : ℝ) (jasmine_percent : ℝ) (rose_percent : ℝ) (mint_percent : ℝ)
  (added_jasmine : ℝ) (added_rose : ℝ) (added_mint : ℝ) (added_plain : ℝ) : Prop :=
  let initial_jasmine := initial_volume * jasmine_percent
  let initial_rose := initial_volume * rose_percent
  let initial_mint := initial_volume * mint_percent
  let new_jasmine := initial_jasmine + added_jasmine
  let new_rose := initial_rose + added_rose
  let new_mint := initial_mint + added_mint
  let new_volume := initial_volume + added_jasmine + added_rose + added_mint + added_plain
  let new_percent := (new_jasmine + new_rose + new_mint) / new_volume * 100
  new_percent = 21.91

theorem cocktail_theorem :
  cocktail_proof 150 0.03 0.05 0.02 12 9 3 4 := by
  sorry

end cocktail_theorem_l1195_119567


namespace angela_insects_l1195_119542

theorem angela_insects (dean_insects : ℕ) (jacob_insects : ℕ) (angela_insects : ℕ)
  (h1 : dean_insects = 30)
  (h2 : jacob_insects = 5 * dean_insects)
  (h3 : angela_insects = jacob_insects / 2) :
  angela_insects = 75 := by
  sorry

end angela_insects_l1195_119542


namespace complex_fraction_equality_l1195_119524

theorem complex_fraction_equality (a : ℝ) : (a + Complex.I) / (2 - Complex.I) = 1 + Complex.I → a = 3 := by
  sorry

end complex_fraction_equality_l1195_119524


namespace greatest_a_divisible_by_three_l1195_119560

theorem greatest_a_divisible_by_three : 
  ∀ a : ℕ, 
    a < 10 → 
    (168 * 10000 + a * 100 + 26) % 3 = 0 → 
    a ≤ 7 :=
by sorry

end greatest_a_divisible_by_three_l1195_119560


namespace problem_solution_l1195_119535

def three_digit_number (x y z : ℕ) : ℕ := 100 * x + 10 * y + z

theorem problem_solution (a b : ℕ) : 
  (three_digit_number 5 b 9) - (three_digit_number 2 a 3) = 326 →
  (three_digit_number 5 6 9) % 9 = 0 →
  a + b = 6 := by
  sorry

end problem_solution_l1195_119535


namespace min_games_for_prediction_l1195_119597

/-- Represents the chess tournament setup -/
structure ChessTournament where
  white_rook : ℕ  -- number of students from "White Rook" school
  black_elephant : ℕ  -- number of students from "Black Elephant" school
  total_games : ℕ  -- total number of games to be played

/-- Checks if the tournament setup is valid -/
def is_valid_tournament (t : ChessTournament) : Prop :=
  t.white_rook * t.black_elephant = t.total_games

/-- Represents the state of the tournament after some games -/
structure TournamentState where
  tournament : ChessTournament
  games_played : ℕ

/-- Checks if Sasha can predict a participant in the next game -/
def can_predict_participant (state : TournamentState) : Prop :=
  state.games_played ≥ state.tournament.total_games - state.tournament.black_elephant

/-- The main theorem to be proved -/
theorem min_games_for_prediction (t : ChessTournament) 
    (h_valid : is_valid_tournament t) 
    (h_white : t.white_rook = 15) 
    (h_black : t.black_elephant = 20) : 
    ∀ n : ℕ, can_predict_participant ⟨t, n⟩ ↔ n ≥ 280 := by
  sorry

end min_games_for_prediction_l1195_119597


namespace dot_product_on_trajectory_l1195_119583

/-- The trajectory E in the xy-plane -/
def TrajectoryE (x y : ℝ) : Prop :=
  |((x + 2)^2 + y^2).sqrt - ((x - 2)^2 + y^2).sqrt| = 2

/-- Point A -/
def A : ℝ × ℝ := (-2, 0)

/-- Point B -/
def B : ℝ × ℝ := (2, 0)

/-- Theorem stating that for any point C on trajectory E with BC perpendicular to x-axis,
    the dot product of AC and BC equals 9 -/
theorem dot_product_on_trajectory (C : ℝ × ℝ) (hC : TrajectoryE C.1 C.2)
  (hPerp : C.1 = B.1) : (C.1 - A.1) * (C.1 - B.1) + (C.2 - A.2) * (C.2 - B.2) = 9 := by
  sorry

end dot_product_on_trajectory_l1195_119583


namespace skyline_hospital_quadruplets_l1195_119519

theorem skyline_hospital_quadruplets :
  ∀ (twins triplets quads : ℕ),
    triplets = 5 * quads →
    twins = 3 * triplets →
    2 * twins + 3 * triplets + 4 * quads = 1200 →
    4 * quads = 98 :=
by
  sorry

end skyline_hospital_quadruplets_l1195_119519


namespace work_problem_underdetermined_l1195_119592

-- Define the work rate of one man and one woman
variable (m w : ℝ)

-- Define the unknown number of men
variable (x : ℝ)

-- Condition 1: x men or 12 women can do the work in 20 days
def condition1 : Prop := x * m * 20 = 12 * w * 20

-- Condition 2: 6 men and 11 women can do the work in 12 days
def condition2 : Prop := (6 * m + 11 * w) * 12 = 1

-- Theorem: The conditions are insufficient to uniquely determine x
theorem work_problem_underdetermined :
  ∃ (m1 w1 x1 : ℝ) (m2 w2 x2 : ℝ),
    condition1 m1 w1 x1 ∧ condition2 m1 w1 ∧
    condition1 m2 w2 x2 ∧ condition2 m2 w2 ∧
    x1 ≠ x2 :=
sorry

end work_problem_underdetermined_l1195_119592


namespace math_preference_gender_related_l1195_119539

/-- Represents the survey data and critical value for the chi-square test -/
structure SurveyData where
  total_students : Nat
  male_percentage : Rat
  total_math_liking : Nat
  female_math_liking : Nat
  critical_value : Rat

/-- Calculates the chi-square statistic for the given survey data -/
def calculate_chi_square (data : SurveyData) : Rat :=
  sorry

/-- Theorem stating that the calculated chi-square value exceeds the critical value -/
theorem math_preference_gender_related (data : SurveyData) :
  data.total_students = 100 ∧
  data.male_percentage = 55/100 ∧
  data.total_math_liking = 40 ∧
  data.female_math_liking = 20 ∧
  data.critical_value = 7879/1000 →
  calculate_chi_square data > data.critical_value :=
sorry

end math_preference_gender_related_l1195_119539


namespace rhombus_perimeter_l1195_119522

/-- The perimeter of a rhombus inscribed in a rectangle --/
theorem rhombus_perimeter (w l : ℝ) (hw : w = 20) (hl : l = 25) :
  let s := Real.sqrt (w^2 / 4 + l^2 / 4)
  let perimeter := 4 * s
  ∃ ε > 0, abs (perimeter - 64.04) < ε := by
  sorry

end rhombus_perimeter_l1195_119522


namespace max_consecutive_integers_sum_largest_n_sum_less_than_1000_max_consecutive_integers_1000_l1195_119503

theorem max_consecutive_integers_sum (n : ℕ) : n ≤ 44 ↔ n * (n + 1) ≤ 2000 := by
  sorry

theorem largest_n_sum_less_than_1000 : ∀ k > 44, k * (k + 1) > 2000 := by
  sorry

theorem max_consecutive_integers_1000 : 
  (∀ m ≤ 44, m * (m + 1) ≤ 2000) ∧
  (∀ k > 44, k * (k + 1) > 2000) := by
  sorry

end max_consecutive_integers_sum_largest_n_sum_less_than_1000_max_consecutive_integers_1000_l1195_119503


namespace rose_additional_money_l1195_119570

def paintbrush_cost : ℚ := 2.4
def paints_cost : ℚ := 9.2
def easel_cost : ℚ := 6.5
def rose_money : ℚ := 7.1

theorem rose_additional_money :
  paintbrush_cost + paints_cost + easel_cost - rose_money = 11 := by sorry

end rose_additional_money_l1195_119570


namespace dave_tickets_l1195_119573

/-- The number of tickets Dave spent on the stuffed tiger -/
def tickets_spent : ℕ := 43

/-- The number of tickets Dave had left after the purchase -/
def tickets_left : ℕ := 55

/-- The initial number of tickets Dave had -/
def initial_tickets : ℕ := tickets_spent + tickets_left

theorem dave_tickets : initial_tickets = 98 := by sorry

end dave_tickets_l1195_119573


namespace complex_reciprocal_sum_l1195_119505

theorem complex_reciprocal_sum (z w : ℂ) 
  (hz : Complex.abs z = 2)
  (hw : Complex.abs w = 4)
  (hzw : Complex.abs (z + w) = 5) :
  Complex.abs (1 / z + 1 / w) = 5 / 8 := by sorry

end complex_reciprocal_sum_l1195_119505


namespace bill_processing_error_l1195_119584

theorem bill_processing_error (x y : ℕ) : 
  10 ≤ x ∧ x ≤ 99 ∧ 10 ≤ y ∧ y ≤ 99 →
  100 * y + x - (100 * x + y) = 2970 →
  y = x + 30 ∧ 10 ≤ x ∧ x ≤ 69 :=
by sorry

end bill_processing_error_l1195_119584


namespace triangle_problem_l1195_119576

noncomputable def f (x φ : Real) : Real := 2 * Real.sin x * (Real.cos (φ / 2))^2 + Real.cos x * Real.sin φ - Real.sin x

theorem triangle_problem (φ : Real) (A B C : Real) (a b c : Real) :
  (0 < φ) ∧ (φ < Real.pi) ∧
  (∀ x, f x φ ≥ f Real.pi φ) ∧
  (a = 1) ∧ (b = Real.sqrt 2) ∧
  (f A φ = Real.sqrt 3 / 2) ∧
  (a / Real.sin A = b / Real.sin B) ∧
  (A + B + C = Real.pi) →
  (φ = Real.pi / 2) ∧
  (∀ x, f x φ = Real.cos x) ∧
  (C = 7 * Real.pi / 12) := by
  sorry

end triangle_problem_l1195_119576


namespace negation_of_existence_original_proposition_negation_l1195_119507

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x > 1, p x) ↔ ∀ x > 1, ¬ p x := by sorry

theorem original_proposition_negation :
  (¬ ∃ x > 1, 3*x + 1 > 5) ↔ (∀ x > 1, 3*x + 1 ≤ 5) := by sorry

end negation_of_existence_original_proposition_negation_l1195_119507


namespace solution_set_range_of_a_l1195_119566

-- Part 1
def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

theorem solution_set (x : ℝ) : f x ≥ 2 ↔ x ≤ 1/2 ∨ x ≥ 5/2 := by sorry

-- Part 2
def g (a x : ℝ) : ℝ := |x - 1| + |x - a|

theorem range_of_a (a : ℝ) : 
  (a > 1 ∧ ∀ x, g a x + |x - 1| ≥ 1) ↔ a ≥ 2 := by sorry

end solution_set_range_of_a_l1195_119566


namespace cube_sum_problem_l1195_119559

theorem cube_sum_problem (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) :
  x^3 + y^3 = 65 := by
  sorry

end cube_sum_problem_l1195_119559


namespace arithmetic_sequence_index_l1195_119581

/-- Given an arithmetic sequence {a_n} with first term a₁ = 1 and common difference d = 5,
    prove that if a_n = 2016, then n = 404. -/
theorem arithmetic_sequence_index (a : ℕ → ℝ) (n : ℕ) 
  (h1 : a 1 = 1)
  (h2 : ∀ k : ℕ, a (k + 1) - a k = 5)
  (h3 : a n = 2016) :
  n = 404 :=
sorry

end arithmetic_sequence_index_l1195_119581


namespace stock_worth_calculation_l1195_119596

/-- Calculates the total worth of a stock given specific selling conditions and overall loss --/
theorem stock_worth_calculation (stock_value : ℝ) : 
  (0.2 * stock_value * 1.2 + 0.8 * stock_value * 0.9) - stock_value = -500 → 
  stock_value = 12500 := by
  sorry

#check stock_worth_calculation

end stock_worth_calculation_l1195_119596


namespace pastries_made_pastries_made_correct_l1195_119536

/-- Given information about Baker's cakes and pastries -/
structure BakerInfo where
  cakes_made : ℕ
  cakes_sold : ℕ
  pastries_sold : ℕ
  cakes_pastries_diff : ℕ
  h1 : cakes_made = 157
  h2 : cakes_sold = 158
  h3 : pastries_sold = 147
  h4 : cakes_sold - pastries_sold = cakes_pastries_diff
  h5 : cakes_pastries_diff = 11

/-- Theorem stating the number of pastries Baker made -/
theorem pastries_made (info : BakerInfo) : ℕ := by
  sorry

#check @pastries_made

/-- The actual number of pastries Baker made -/
def actual_pastries_made : ℕ := 146

/-- Theorem proving that the calculated number of pastries matches the actual number -/
theorem pastries_made_correct (info : BakerInfo) : pastries_made info = actual_pastries_made := by
  sorry

end pastries_made_pastries_made_correct_l1195_119536


namespace probability_of_triangle_in_decagon_l1195_119582

/-- A regular decagon is a 10-sided polygon with all sides and angles equal -/
structure RegularDecagon where
  -- No specific properties needed for this problem

/-- The number of diagonals in a regular decagon -/
def num_diagonals : ℕ := 35

/-- The number of ways to choose 3 diagonals from the total number of diagonals -/
def total_diagonal_choices : ℕ := Nat.choose num_diagonals 3

/-- The number of ways to choose 4 points from 10 points -/
def four_point_choices : ℕ := Nat.choose 10 4

/-- The number of ways to choose 3 points out of 4 points -/
def three_out_of_four : ℕ := Nat.choose 4 3

/-- The number of triangle-forming sets of diagonals -/
def triangle_forming_sets : ℕ := four_point_choices * three_out_of_four

theorem probability_of_triangle_in_decagon (d : RegularDecagon) :
  (triangle_forming_sets : ℚ) / total_diagonal_choices = 840 / 6545 := by
  sorry

end probability_of_triangle_in_decagon_l1195_119582


namespace real_estate_calendar_problem_l1195_119515

/-- Proves that given the conditions of the real estate problem, the number of calendars ordered is 200 -/
theorem real_estate_calendar_problem :
  ∀ (calendar_cost date_book_cost : ℚ) (total_items : ℕ) (total_spent : ℚ) (calendars date_books : ℕ),
    calendar_cost = 3/4 →
    date_book_cost = 1/2 →
    total_items = 500 →
    total_spent = 300 →
    calendars + date_books = total_items →
    calendar_cost * calendars + date_book_cost * date_books = total_spent →
    calendars = 200 := by
  sorry

end real_estate_calendar_problem_l1195_119515


namespace snack_expenditure_l1195_119585

theorem snack_expenditure (initial_amount : ℕ) (computer_accessories : ℕ) (remaining_amount : ℕ) 
  (h1 : initial_amount = 48)
  (h2 : computer_accessories = 12)
  (h3 : remaining_amount = initial_amount / 2 + 4) :
  initial_amount - computer_accessories - remaining_amount = 8 := by
  sorry

end snack_expenditure_l1195_119585


namespace family_ages_theorem_l1195_119534

/-- Represents the ages and birth times of a father and his two children -/
structure FamilyAges where
  fatherCurrentAge : ℝ
  sonAgeFiveYearsAgo : ℝ
  daughterAgeFiveYearsAgo : ℝ
  sonCurrentAge : ℝ
  daughterCurrentAge : ℝ
  fatherAgeAtSonBirth : ℝ
  fatherAgeAtDaughterBirth : ℝ

/-- Theorem about the ages in a family based on given conditions -/
theorem family_ages_theorem (f : FamilyAges)
    (h1 : f.fatherCurrentAge = 38)
    (h2 : f.sonAgeFiveYearsAgo = 7)
    (h3 : f.daughterAgeFiveYearsAgo = f.sonAgeFiveYearsAgo / 2)
    (h4 : f.sonCurrentAge = f.sonAgeFiveYearsAgo + 5)
    (h5 : f.daughterCurrentAge = f.daughterAgeFiveYearsAgo + 5)
    (h6 : f.fatherAgeAtSonBirth = f.fatherCurrentAge - f.sonCurrentAge)
    (h7 : f.fatherAgeAtDaughterBirth = f.fatherCurrentAge - f.daughterCurrentAge) :
    f.sonCurrentAge = 12 ∧
    f.daughterCurrentAge = 8.5 ∧
    f.fatherAgeAtSonBirth = 26 ∧
    f.fatherAgeAtDaughterBirth = 29.5 := by
  sorry


end family_ages_theorem_l1195_119534


namespace hip_size_conversion_l1195_119506

/-- Converts inches to millimeters given the conversion factors -/
def inches_to_mm (inches_per_foot : ℚ) (mm_per_foot : ℚ) (inches : ℚ) : ℚ :=
  inches * (mm_per_foot / inches_per_foot)

/-- Proves that 42 inches is equivalent to 1067.5 millimeters -/
theorem hip_size_conversion (inches_per_foot mm_per_foot : ℚ) 
  (h1 : inches_per_foot = 12)
  (h2 : mm_per_foot = 305) : 
  inches_to_mm inches_per_foot mm_per_foot 42 = 1067.5 := by
  sorry

#eval inches_to_mm 12 305 42

end hip_size_conversion_l1195_119506


namespace word_problems_count_l1195_119504

theorem word_problems_count (total_questions : ℕ) 
                             (addition_subtraction_problems : ℕ) 
                             (steve_answered : ℕ) : 
  total_questions = 45 →
  addition_subtraction_problems = 28 →
  steve_answered = 38 →
  total_questions - steve_answered = 7 →
  total_questions - addition_subtraction_problems = 17 := by
  sorry

end word_problems_count_l1195_119504


namespace square_sum_equals_sixteen_l1195_119599

theorem square_sum_equals_sixteen (x y : ℝ) (h1 : (x + y)^2 = 4) (h2 : x * y = -6) :
  x^2 + y^2 = 16 := by
  sorry

end square_sum_equals_sixteen_l1195_119599


namespace rectangle_diagonal_l1195_119552

/-- The length of the diagonal of a rectangle with length 20√5 and width 10√3 is 10√23 -/
theorem rectangle_diagonal (length width diagonal : ℝ) 
  (h_length : length = 20 * Real.sqrt 5)
  (h_width : width = 10 * Real.sqrt 3)
  (h_diagonal : diagonal^2 = length^2 + width^2) : 
  diagonal = 10 * Real.sqrt 23 := by
  sorry

end rectangle_diagonal_l1195_119552


namespace jack_needs_five_rocks_l1195_119574

-- Define the weights and rock weight
def jack_weight : ℕ := 60
def anna_weight : ℕ := 40
def rock_weight : ℕ := 4

-- Define the function to calculate the number of rocks
def num_rocks (jack_w anna_w rock_w : ℕ) : ℕ :=
  (jack_w - anna_w) / rock_w

-- Theorem statement
theorem jack_needs_five_rocks :
  num_rocks jack_weight anna_weight rock_weight = 5 := by
  sorry

end jack_needs_five_rocks_l1195_119574


namespace non_black_cows_l1195_119528

theorem non_black_cows (total : ℕ) (black : ℕ) (h1 : total = 18) (h2 : black = total / 2 + 5) :
  total - black = 4 := by
sorry

end non_black_cows_l1195_119528


namespace altitude_df_length_l1195_119510

/-- Represents a parallelogram ABCD with altitudes DE and DF -/
structure Parallelogram where
  /-- Length of side DC -/
  dc : ℝ
  /-- Length of segment EB -/
  eb : ℝ
  /-- Length of altitude DE -/
  de : ℝ
  /-- Ensures dc is positive -/
  dc_pos : dc > 0
  /-- Ensures eb is positive -/
  eb_pos : eb > 0
  /-- Ensures de is positive -/
  de_pos : de > 0
  /-- Ensures eb is less than dc (as EB is part of AB which is equal to DC) -/
  eb_lt_dc : eb < dc

/-- Theorem stating that under the given conditions, DF = 5 -/
theorem altitude_df_length (p : Parallelogram) (h1 : p.dc = 15) (h2 : p.eb = 3) (h3 : p.de = 5) :
  ∃ df : ℝ, df = 5 ∧ df > 0 := by
  sorry

end altitude_df_length_l1195_119510


namespace cube_volume_from_surface_area_l1195_119548

theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), s > 0 → 6 * s^2 = 864 → s^3 = 1728 :=
by
  sorry

end cube_volume_from_surface_area_l1195_119548


namespace not_closed_under_addition_l1195_119512

-- Define a "good set" S
def GoodSet (S : Set ℤ) : Prop :=
  ∀ a b : ℤ, (a^2 - b^2) ∈ S

-- Theorem statement
theorem not_closed_under_addition
  (S : Set ℤ) (hS : S.Nonempty) (hGood : GoodSet S) :
  ¬ (∀ x y : ℤ, x ∈ S → y ∈ S → (x + y) ∈ S) :=
sorry

end not_closed_under_addition_l1195_119512


namespace fourth_root_l1195_119588

/-- The polynomial function defined by the given coefficients -/
def f (b c x : ℝ) : ℝ := b*x^4 + (b + 3*c)*x^3 + (c - 4*b)*x^2 + (19 - b)*x - 2

theorem fourth_root (b c : ℝ) 
  (h1 : f b c (-3) = 0)
  (h2 : f b c 4 = 0)
  (h3 : f b c 2 = 0) :
  ∃ x, x ≠ -3 ∧ x ≠ 4 ∧ x ≠ 2 ∧ f b c x = 0 ∧ x = 1 := by
  sorry

end fourth_root_l1195_119588


namespace complex_number_representation_l1195_119502

theorem complex_number_representation : ∃ (z : ℂ), z = 1 + 2*I := by sorry

end complex_number_representation_l1195_119502


namespace piggy_bank_problem_l1195_119556

def arithmetic_sum (a₁ n : ℕ) : ℕ := n * (a₁ + (a₁ + n - 1)) / 2

theorem piggy_bank_problem (initial_amount final_amount : ℕ) : 
  final_amount = 1478 →
  arithmetic_sum 1 52 = 1378 →
  initial_amount = final_amount - arithmetic_sum 1 52 →
  initial_amount = 100 := by
  sorry

end piggy_bank_problem_l1195_119556


namespace range_of_2a_minus_b_l1195_119568

theorem range_of_2a_minus_b (a b : ℝ) (h : -1 < a ∧ a < b ∧ b < 2) :
  ∀ x, x ∈ Set.Ioo (-4 : ℝ) 2 ↔ ∃ a b, -1 < a ∧ a < b ∧ b < 2 ∧ x = 2*a - b :=
by sorry

end range_of_2a_minus_b_l1195_119568


namespace triangle_base_length_l1195_119543

/-- The length of the base of a triangle with altitude 12 cm and area equal to a square with side 6 cm is 6 cm. -/
theorem triangle_base_length (base altitude : ℝ) (h1 : altitude = 12) 
  (h2 : (base * altitude) / 2 = 6 * 6) : base = 6 := by
  sorry

end triangle_base_length_l1195_119543


namespace min_cubes_for_specific_box_l1195_119541

/-- The minimum number of cubes required to build a box -/
def min_cubes (length width height cube_size : ℕ) : ℕ :=
  (length * width * height) / (cube_size ^ 3)

/-- Theorem: The minimum number of 3 cm³ cubes required to build a 9 cm × 12 cm × 3 cm box is 108 -/
theorem min_cubes_for_specific_box :
  min_cubes 9 12 3 3 = 108 := by
  sorry

end min_cubes_for_specific_box_l1195_119541


namespace solve_complex_equation_l1195_119516

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation
def equation (z : ℂ) : Prop := z * (1 + i) = 2 + i

-- Theorem statement
theorem solve_complex_equation :
  ∀ z : ℂ, equation z → z = (3/2 : ℝ) - (1/2 : ℝ) * i :=
by
  sorry

end solve_complex_equation_l1195_119516


namespace hair_cut_total_l1195_119565

theorem hair_cut_total : 
  let monday : ℚ := 38 / 100
  let tuesday : ℚ := 1 / 2
  let wednesday : ℚ := 1 / 4
  let thursday : ℚ := 87 / 100
  monday + tuesday + wednesday + thursday = 2 := by sorry

end hair_cut_total_l1195_119565


namespace zero_subset_X_l1195_119500

def X : Set ℝ := {x | x > -1}

theorem zero_subset_X : {0} ⊆ X := by
  sorry

end zero_subset_X_l1195_119500


namespace decoration_sets_count_l1195_119575

/-- Represents a decoration set with balloons and ribbons -/
structure DecorationSet where
  balloons : ℕ
  ribbons : ℕ

/-- The cost of a decoration set -/
def cost (set : DecorationSet) : ℕ := 4 * set.balloons + 6 * set.ribbons

/-- Predicate for valid decoration sets -/
def isValid (set : DecorationSet) : Prop :=
  cost set = 120 ∧ Even set.balloons

theorem decoration_sets_count :
  ∃! (sets : Finset DecorationSet), 
    (∀ s ∈ sets, isValid s) ∧ 
    (∀ s, isValid s → s ∈ sets) ∧
    Finset.card sets = 2 := by
  sorry

end decoration_sets_count_l1195_119575


namespace point_below_line_l1195_119571

/-- A point P(a, 3) is below the line 2x - y = 3 if and only if a < 3 -/
theorem point_below_line (a : ℝ) : 
  (2 * a - 3 < 3) ↔ (a < 3) := by sorry

end point_below_line_l1195_119571


namespace framed_painting_ratio_l1195_119511

theorem framed_painting_ratio : 
  ∀ (y : ℝ),
  y > 0 →
  (20 + 2*y) * (30 + 6*y) = 2 * 20 * 30 →
  (min (20 + 2*y) (30 + 6*y)) / (max (20 + 2*y) (30 + 6*y)) = 4/7 :=
by sorry

end framed_painting_ratio_l1195_119511


namespace cherry_tree_leaves_l1195_119579

/-- The number of cherry trees originally planned to be planted -/
def originalTreeCount : ℕ := 7

/-- The actual number of cherry trees planted -/
def actualTreeCount : ℕ := 2 * originalTreeCount

/-- The number of leaves dropped by each tree during fall -/
def leavesPerTree : ℕ := 100

/-- The total number of leaves falling from all cherry trees -/
def totalLeaves : ℕ := actualTreeCount * leavesPerTree

theorem cherry_tree_leaves :
  totalLeaves = 1400 := by sorry

end cherry_tree_leaves_l1195_119579


namespace point_on_intersection_line_l1195_119598

-- Define the sets and points
variable (α β m n l : Set Point)
variable (P : Point)

-- State the theorem
theorem point_on_intersection_line
  (h1 : α ∩ β = l)
  (h2 : m ⊆ α)
  (h3 : n ⊆ β)
  (h4 : m ∩ n = {P}) :
  P ∈ l := by
sorry

end point_on_intersection_line_l1195_119598


namespace f_properties_l1195_119561

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (x - a) - Real.log (x + a)

theorem f_properties :
  (∀ x > -1/2, f (1/2) x < f (1/2) (1/2)) ∧ 
  (∀ x > 1/2, f (1/2) x > f (1/2) (1/2)) ∧
  (f (1/2) (1/2) = 1) ∧
  (∀ a ≤ 1, ∀ x > -a, f a x > 0) := by sorry

end f_properties_l1195_119561


namespace sqrt_three_times_sqrt_twelve_l1195_119555

theorem sqrt_three_times_sqrt_twelve : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_three_times_sqrt_twelve_l1195_119555


namespace license_plate_count_is_9360_l1195_119523

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possibilities for the second character (letter or digit) -/
def num_second_char : ℕ := num_letters + num_digits

/-- The number of ways to design a 4-character license plate with the given conditions -/
def license_plate_count : ℕ := num_letters * num_second_char * 1 * num_digits

theorem license_plate_count_is_9360 : license_plate_count = 9360 := by
  sorry

end license_plate_count_is_9360_l1195_119523


namespace volume_of_removed_tetrahedra_l1195_119514

-- Define the cube
def cube_side_length : ℝ := 2

-- Define the number of segments per edge
def segments_per_edge : ℕ := 3

-- Define the number of corners (tetrahedra)
def num_corners : ℕ := 8

-- Theorem statement
theorem volume_of_removed_tetrahedra :
  let segment_length : ℝ := cube_side_length / segments_per_edge
  let base_area : ℝ := (1 / 2) * segment_length^2
  let tetrahedron_height : ℝ := segment_length
  let tetrahedron_volume : ℝ := (1 / 3) * base_area * tetrahedron_height
  let total_volume : ℝ := num_corners * tetrahedron_volume
  total_volume = 32 / 81 := by sorry

end volume_of_removed_tetrahedra_l1195_119514


namespace zoe_pool_cleaning_earnings_l1195_119593

/-- Represents Zoe's earnings and babysitting frequencies -/
structure ZoeEarnings where
  total : ℕ
  zachary_earnings : ℕ
  julie_freq : ℕ
  zachary_freq : ℕ
  chloe_freq : ℕ

/-- Calculates Zoe's earnings from pool cleaning -/
def pool_cleaning_earnings (e : ZoeEarnings) : ℕ :=
  e.total - (e.zachary_earnings * (1 + 3 + 5))

/-- Theorem stating that Zoe's pool cleaning earnings are $2,600 -/
theorem zoe_pool_cleaning_earnings :
  ∀ e : ZoeEarnings,
    e.total = 8000 ∧
    e.zachary_earnings = 600 ∧
    e.julie_freq = 3 * e.zachary_freq ∧
    e.zachary_freq * 5 = e.chloe_freq →
    pool_cleaning_earnings e = 2600 :=
by
  sorry


end zoe_pool_cleaning_earnings_l1195_119593


namespace meaningful_range_l1195_119508

def is_meaningful (x : ℝ) : Prop :=
  x + 1 ≥ 0 ∧ x ≠ 0

theorem meaningful_range :
  ∀ x : ℝ, is_meaningful x ↔ x ≥ -1 ∧ x ≠ 0 :=
by sorry

end meaningful_range_l1195_119508


namespace smallest_positive_m_squared_l1195_119554

/-- Definition of circle w₁ -/
def w₁ (x y : ℝ) : Prop := x^2 + y^2 + 10*x - 24*y - 87 = 0

/-- Definition of circle w₂ -/
def w₂ (x y : ℝ) : Prop := x^2 + y^2 - 10*x - 24*y + 153 = 0

/-- Definition of a line y = ax -/
def line (a x y : ℝ) : Prop := y = a * x

/-- Definition of external tangency to w₂ -/
def externally_tangent_w₂ (x y r : ℝ) : Prop :=
  (x - 5)^2 + (y - 12)^2 = (r + 4)^2

/-- Definition of internal tangency to w₁ -/
def internally_tangent_w₁ (x y r : ℝ) : Prop :=
  (x + 5)^2 + (y - 12)^2 = (16 - r)^2

/-- The main theorem -/
theorem smallest_positive_m_squared (m : ℝ) : 
  (∀ a : ℝ, a > 0 → (∃ x y r : ℝ, line a x y ∧ externally_tangent_w₂ x y r ∧ internally_tangent_w₁ x y r) → m ≤ a) ∧
  (∃ x y r : ℝ, line m x y ∧ externally_tangent_w₂ x y r ∧ internally_tangent_w₁ x y r) →
  m^2 = 69/100 :=
sorry

end smallest_positive_m_squared_l1195_119554


namespace sophie_savings_l1195_119572

/-- Represents the amount of money saved in a year by not buying dryer sheets -/
def money_saved (loads_per_week : ℕ) (sheets_per_load : ℕ) (sheets_per_box : ℕ) (cost_per_box : ℚ) (weeks_per_year : ℕ) : ℚ :=
  let sheets_per_year := loads_per_week * sheets_per_load * weeks_per_year
  let boxes_per_year := (sheets_per_year + sheets_per_box - 1) / sheets_per_box
  boxes_per_year * cost_per_box

/-- The amount of money Sophie saves in a year by not buying dryer sheets is $11.00 -/
theorem sophie_savings : 
  money_saved 4 1 104 (11/2) 52 = 11 := by
  sorry

end sophie_savings_l1195_119572


namespace expected_red_lights_proof_l1195_119590

/-- The number of intersections with traffic lights -/
def num_intersections : ℕ := 3

/-- The probability of encountering a red light at each intersection -/
def red_light_probability : ℝ := 0.3

/-- The events of encountering a red light at each intersection are independent -/
axiom events_independent : True

/-- The expected number of red lights encountered -/
def expected_red_lights : ℝ := num_intersections * red_light_probability

theorem expected_red_lights_proof :
  expected_red_lights = 0.9 :=
by sorry

end expected_red_lights_proof_l1195_119590


namespace zeta_sum_seventh_power_l1195_119564

theorem zeta_sum_seventh_power (ζ₁ ζ₂ ζ₃ : ℂ) 
  (h1 : ζ₁ + ζ₂ + ζ₃ = 2)
  (h2 : ζ₁^2 + ζ₂^2 + ζ₃^2 = 6)
  (h3 : ζ₁^3 + ζ₂^3 + ζ₃^3 = 8) :
  ζ₁^7 + ζ₂^7 + ζ₃^7 = 58 := by
  sorry

end zeta_sum_seventh_power_l1195_119564


namespace people_in_line_l1195_119549

theorem people_in_line (people_between : ℕ) (h : people_between = 5) : 
  people_between + 2 = 7 := by
  sorry

end people_in_line_l1195_119549


namespace platform_length_l1195_119569

/-- The length of a platform given train crossing times -/
theorem platform_length (train_length : ℝ) (platform_time : ℝ) (pole_time : ℝ) :
  train_length = 300 →
  platform_time = 39 →
  pole_time = 18 →
  ∃ platform_length : ℝ,
    platform_length = 350 ∧
    (train_length + platform_length) / platform_time = train_length / pole_time :=
by sorry

end platform_length_l1195_119569


namespace original_salary_proof_l1195_119517

/-- Given a 6% raise resulting in a new salary of $530, prove that the original salary was $500. -/
theorem original_salary_proof (original_salary : ℝ) : 
  original_salary * 1.06 = 530 → original_salary = 500 := by
  sorry

end original_salary_proof_l1195_119517


namespace office_paper_sheets_per_pack_l1195_119509

/-- The number of sheets in each pack of printer paper -/
def sheets_per_pack (total_packs : ℕ) (documents_per_day : ℕ) (days_lasted : ℕ) : ℕ :=
  (documents_per_day * days_lasted) / total_packs

/-- Theorem stating the number of sheets in each pack of printer paper -/
theorem office_paper_sheets_per_pack :
  sheets_per_pack 2 80 6 = 240 := by
  sorry

end office_paper_sheets_per_pack_l1195_119509


namespace z_in_first_quadrant_l1195_119586

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- The complex number z -/
def z : ℂ := i * (2 - i)

/-- A complex number is in the first quadrant if its real part is positive and its imaginary part is positive -/
def is_in_first_quadrant (c : ℂ) : Prop := 0 < c.re ∧ 0 < c.im

/-- Theorem: z is in the first quadrant -/
theorem z_in_first_quadrant : is_in_first_quadrant z := by
  sorry

end z_in_first_quadrant_l1195_119586


namespace giannas_savings_l1195_119594

/-- Gianna's savings calculation --/
theorem giannas_savings (daily_savings : ℕ) (days_in_year : ℕ) (total_savings : ℕ) :
  daily_savings = 39 →
  days_in_year = 365 →
  total_savings = daily_savings * days_in_year →
  total_savings = 14235 := by
  sorry

end giannas_savings_l1195_119594


namespace intersection_complement_equals_l1195_119557

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define set A
def A : Set Nat := {1, 3, 6}

-- Define set B
def B : Set Nat := {2, 3, 4}

-- Theorem to prove
theorem intersection_complement_equals : A ∩ (U \ B) = {1, 6} := by
  sorry

end intersection_complement_equals_l1195_119557


namespace remaining_amount_is_14_90_l1195_119521

-- Define the initial amount and item costs
def initial_amount : ℚ := 78
def kite_cost : ℚ := 8
def frisbee_cost : ℚ := 9
def roller_skates_cost : ℚ := 15
def roller_skates_discount : ℚ := 0.1
def lego_cost : ℚ := 25
def lego_coupon : ℚ := 5
def puzzle_cost : ℚ := 12
def puzzle_tax_rate : ℚ := 0.05

-- Define the function to calculate the remaining amount
def remaining_amount : ℚ :=
  initial_amount -
  (kite_cost +
   frisbee_cost +
   (roller_skates_cost * (1 - roller_skates_discount)) +
   (lego_cost - lego_coupon) +
   (puzzle_cost * (1 + puzzle_tax_rate)))

-- Theorem stating that the remaining amount is $14.90
theorem remaining_amount_is_14_90 :
  remaining_amount = 14.90 := by sorry

end remaining_amount_is_14_90_l1195_119521


namespace limes_given_to_sara_l1195_119529

/-- Given that Dan initially picked some limes and gave some to Sara, 
    prove that the number of limes Dan gave to Sara is equal to 
    the difference between his initial and final number of limes. -/
theorem limes_given_to_sara 
  (initial_limes : ℕ) 
  (final_limes : ℕ) 
  (h1 : initial_limes = 9)
  (h2 : final_limes = 5) :
  initial_limes - final_limes = 4 := by
  sorry

end limes_given_to_sara_l1195_119529


namespace arithmetic_sequence_property_l1195_119531

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum : a 4 + a 10 + a 16 = 30) :
  a 18 - 2 * a 14 = -10 :=
sorry

end arithmetic_sequence_property_l1195_119531


namespace least_sum_exponents_500_l1195_119544

/-- Represents a sum of distinct powers of 2 -/
def DistinctPowersOfTwo := List Nat

/-- Checks if a list of natural numbers represents distinct powers of 2 -/
def isDistinctPowersOfTwo (l : List Nat) : Prop :=
  l.Nodup ∧ ∀ n ∈ l, ∃ k, n = 2^k

/-- Computes the sum of a list of natural numbers -/
def sumList (l : List Nat) : Nat :=
  l.foldl (·+·) 0

/-- Computes the sum of exponents for a list of powers of 2 -/
def sumExponents (l : DistinctPowersOfTwo) : Nat :=
  sumList (l.map (fun n => (Nat.log n 2)))

/-- The main theorem to be proved -/
theorem least_sum_exponents_500 :
  (∃ (l : DistinctPowersOfTwo),
    isDistinctPowersOfTwo l ∧
    l.length ≥ 2 ∧
    sumList l = 500 ∧
    (∀ (m : DistinctPowersOfTwo),
      isDistinctPowersOfTwo m → m.length ≥ 2 → sumList m = 500 →
      sumExponents l ≤ sumExponents m)) ∧
  (∃ (l : DistinctPowersOfTwo),
    isDistinctPowersOfTwo l ∧
    l.length ≥ 2 ∧
    sumList l = 500 ∧
    sumExponents l = 30) :=
by sorry

end least_sum_exponents_500_l1195_119544


namespace quadratic_roots_nature_l1195_119538

/-- The nature of roots of a quadratic equation based on parameters a and b -/
theorem quadratic_roots_nature (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  let f : ℝ → ℝ := λ x => (a^2 + b^2) * x^2 + 4 * a * b * x + 2 * a * b
  (a = b → (∃! x, f x = 0)) ∧
  (a ≠ b → a * b > 0 → ∀ x, f x ≠ 0) ∧
  (a ≠ b → a * b < 0 → ∃ x y, x ≠ y ∧ f x = 0 ∧ f y = 0) :=
by sorry


end quadratic_roots_nature_l1195_119538


namespace y_intercept_of_parallel_line_l1195_119518

-- Define a line type
structure Line where
  slope : ℝ
  y_intercept : ℝ

-- Define a point type
structure Point where
  x : ℝ
  y : ℝ

-- Define parallel lines
def parallel (l1 l2 : Line) : Prop := l1.slope = l2.slope

-- Define a line passing through a point
def passes_through (l : Line) (p : Point) : Prop :=
  p.y = l.slope * p.x + l.y_intercept

-- Define the given line
def given_line : Line := { slope := 2, y_intercept := 4 }

-- Define the point that line b passes through
def given_point : Point := { x := 3, y := 7 }

-- Theorem statement
theorem y_intercept_of_parallel_line :
  ∃ (b : Line),
    parallel b given_line ∧
    passes_through b given_point ∧
    b.y_intercept = 1 := by sorry

end y_intercept_of_parallel_line_l1195_119518


namespace smallest_n_for_floor_equation_l1195_119577

theorem smallest_n_for_floor_equation : 
  ∀ n : ℕ, n < 7 → ¬∃ x : ℤ, ⌊(10 : ℝ)^n / x⌋ = 2006 ∧ 
  ∃ x : ℤ, ⌊(10 : ℝ)^7 / x⌋ = 2006 :=
sorry

end smallest_n_for_floor_equation_l1195_119577


namespace min_value_not_neg_half_l1195_119520

open Real

theorem min_value_not_neg_half (g : ℝ → ℝ) :
  (∀ x, g x = -Real.sqrt 3 * Real.sin (2 * x) + 1) →
  ¬(∃ x ∈ Set.Icc (π / 6) (π / 2), ∀ y ∈ Set.Icc (π / 6) (π / 2), g x ≤ g y ∧ g x = -1/2) :=
by sorry

end min_value_not_neg_half_l1195_119520


namespace students_playing_both_sports_l1195_119558

/-- Given a college with students playing cricket or basketball, 
    this theorem proves the number of students playing both sports. -/
theorem students_playing_both_sports 
  (total : ℕ) 
  (cricket : ℕ) 
  (basketball : ℕ) 
  (h1 : total = 880) 
  (h2 : cricket = 500) 
  (h3 : basketball = 600) : 
  cricket + basketball - total = 220 := by
  sorry


end students_playing_both_sports_l1195_119558


namespace product_digits_l1195_119501

def a : ℕ := 7123456789
def b : ℕ := 23567891234

theorem product_digits : (String.length (toString (a * b))) = 21 := by
  sorry

end product_digits_l1195_119501


namespace candy_bar_cost_l1195_119532

/-- The cost of a candy bar given initial amount and change --/
theorem candy_bar_cost (initial_amount change : ℕ) (h1 : initial_amount = 50) (h2 : change = 5) :
  initial_amount - change = 45 :=
by
  sorry

end candy_bar_cost_l1195_119532


namespace smallest_three_digit_prime_with_composite_reverse_l1195_119547

/-- A function that reverses the digits of a three-digit number -/
def reverseDigits (n : Nat) : Nat :=
  (n % 10) * 100 + ((n / 10) % 10) * 10 + (n / 100)

/-- A predicate that checks if a number is prime -/
def isPrime (n : Nat) : Prop :=
  n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

/-- A predicate that checks if a number is composite -/
def isComposite (n : Nat) : Prop :=
  n > 1 ∧ ∃ d : Nat, d > 1 ∧ d < n ∧ n % d = 0

theorem smallest_three_digit_prime_with_composite_reverse :
  ∃ (p : Nat),
    p = 103 ∧
    isPrime p ∧
    100 ≤ p ∧ p < 1000 ∧
    isComposite (reverseDigits p) ∧
    ∀ (q : Nat),
      isPrime q ∧
      100 ≤ q ∧ q < p →
      ¬(isComposite (reverseDigits q)) :=
by sorry

end smallest_three_digit_prime_with_composite_reverse_l1195_119547


namespace parallel_planes_lines_l1195_119526

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes
variable (plane_parallel : Plane → Plane → Prop)

-- Define the contained relation for lines and planes
variable (line_in_plane : Line → Plane → Prop)

-- Define the parallel relation for lines
variable (line_parallel : Line → Line → Prop)

-- Define the skew relation for lines
variable (line_skew : Line → Line → Prop)

-- Define the intersection relation for lines
variable (line_intersect : Line → Line → Prop)

-- Theorem statement
theorem parallel_planes_lines
  (α β : Plane) (a b : Line)
  (h_parallel : plane_parallel α β)
  (h_a_in_α : line_in_plane a α)
  (h_b_in_β : line_in_plane b β) :
  (¬ line_intersect a b) ∧
  (line_parallel a b ∨ line_skew a b) :=
sorry

end parallel_planes_lines_l1195_119526
