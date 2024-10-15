import Mathlib

namespace NUMINAMATH_CALUDE_decimal_point_problem_l1671_167194

theorem decimal_point_problem :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧
  x + y = 13.5927 ∧
  y = 10 * x ∧
  x = 1.2357 ∧ y = 12.357 := by
sorry

end NUMINAMATH_CALUDE_decimal_point_problem_l1671_167194


namespace NUMINAMATH_CALUDE_hugo_first_roll_7_given_win_l1671_167176

-- Define the number of players
def num_players : ℕ := 5

-- Define the number of sides on the first die
def first_die_sides : ℕ := 8

-- Define the number of sides on the subsequent die
def subsequent_die_sides : ℕ := 10

-- Define Hugo's winning probability
def hugo_win_prob : ℚ := 1 / num_players

-- Define the probability of rolling a 7 on the first die
def prob_roll_7 : ℚ := 1 / first_die_sides

-- Define the event that Hugo wins given his first roll was 7
def hugo_win_given_7 : ℚ := 961 / 2560

-- Theorem to prove
theorem hugo_first_roll_7_given_win (num_players : ℕ) (first_die_sides : ℕ) 
  (subsequent_die_sides : ℕ) (hugo_win_prob : ℚ) (prob_roll_7 : ℚ) 
  (hugo_win_given_7 : ℚ) :
  num_players = 5 → 
  first_die_sides = 8 → 
  subsequent_die_sides = 10 → 
  hugo_win_prob = 1 / 5 → 
  prob_roll_7 = 1 / 8 → 
  hugo_win_given_7 = 961 / 2560 → 
  (prob_roll_7 * hugo_win_given_7) / hugo_win_prob = 961 / 2048 := by
  sorry


end NUMINAMATH_CALUDE_hugo_first_roll_7_given_win_l1671_167176


namespace NUMINAMATH_CALUDE_earlier_movie_savings_l1671_167103

/-- Calculates the savings when attending an earlier movie with discounts -/
def calculate_savings (evening_ticket_cost : ℝ) (food_combo_cost : ℝ) 
  (ticket_discount_percent : ℝ) (food_discount_percent : ℝ) : ℝ :=
  (evening_ticket_cost * ticket_discount_percent) + 
  (food_combo_cost * food_discount_percent)

/-- Proves that the savings for the earlier movie is $7 -/
theorem earlier_movie_savings :
  let evening_ticket_cost : ℝ := 10
  let food_combo_cost : ℝ := 10
  let ticket_discount_percent : ℝ := 0.2
  let food_discount_percent : ℝ := 0.5
  calculate_savings evening_ticket_cost food_combo_cost 
    ticket_discount_percent food_discount_percent = 7 := by
  sorry

end NUMINAMATH_CALUDE_earlier_movie_savings_l1671_167103


namespace NUMINAMATH_CALUDE_line_parallel_perpendicular_implies_planes_perpendicular_l1671_167138

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (planes_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_parallel_perpendicular_implies_planes_perpendicular
  (l : Line) (α β : Plane) :
  parallel l α → perpendicular l β → planes_perpendicular α β :=
by sorry

end NUMINAMATH_CALUDE_line_parallel_perpendicular_implies_planes_perpendicular_l1671_167138


namespace NUMINAMATH_CALUDE_slope_intercept_sum_l1671_167142

/-- Given points A, B, C, and D in a Cartesian plane, where D is the midpoint of AB,
    prove that the sum of the slope and y-intercept of the line passing through C and D is 3.6 -/
theorem slope_intercept_sum (A B C D : ℝ × ℝ) : 
  A = (0, 8) →
  B = (0, 0) →
  C = (10, 0) →
  D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  let slope := (D.2 - C.2) / (D.1 - C.1)
  let y_intercept := D.2
  slope + y_intercept = 3.6 := by
sorry


end NUMINAMATH_CALUDE_slope_intercept_sum_l1671_167142


namespace NUMINAMATH_CALUDE_max_value_equality_l1671_167128

theorem max_value_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b)^2 / (a^2 + 2*a*b + b^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_equality_l1671_167128


namespace NUMINAMATH_CALUDE_real_y_condition_l1671_167167

theorem real_y_condition (x y : ℝ) : 
  (9 * y^2 - 6 * x * y + 2 * x + 7 = 0) → 
  (∃ (y : ℝ), 9 * y^2 - 6 * x * y + 2 * x + 7 = 0) ↔ (x ≤ -2 ∨ x ≥ 7) :=
by sorry

end NUMINAMATH_CALUDE_real_y_condition_l1671_167167


namespace NUMINAMATH_CALUDE_batsman_average_increase_l1671_167143

/-- Represents a batsman's performance over a series of innings -/
structure BatsmanPerformance where
  innings : ℕ
  totalRuns : ℕ
  neverNotOut : Bool

/-- Calculates the average runs per innings -/
def average (perf : BatsmanPerformance) : ℚ :=
  perf.totalRuns / perf.innings

/-- Represents the change in a batsman's performance after an additional innings -/
structure PerformanceChange where
  before : BatsmanPerformance
  runsScored : ℕ
  newAverage : ℚ

/-- Calculates the increase in average after an additional innings -/
def averageIncrease (change : PerformanceChange) : ℚ :=
  change.newAverage - average change.before

theorem batsman_average_increase :
  ∀ (perf : BatsmanPerformance) (change : PerformanceChange),
    perf.innings = 11 →
    perf.neverNotOut = true →
    change.before = perf →
    change.runsScored = 60 →
    change.newAverage = 38 →
    averageIncrease change = 2 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_increase_l1671_167143


namespace NUMINAMATH_CALUDE_chocolate_problem_l1671_167154

theorem chocolate_problem (cost_price selling_price : ℝ) 
  (h1 : cost_price * 81 = selling_price * 45)
  (h2 : (selling_price - cost_price) / cost_price = 0.8) :
  81 = 81 := by sorry

end NUMINAMATH_CALUDE_chocolate_problem_l1671_167154


namespace NUMINAMATH_CALUDE_sum_of_cubes_l1671_167169

theorem sum_of_cubes (a b c : ℕ+) :
  (a + b + c)^3 - a^3 - b^3 - c^3 = 294 →
  a + b + c = 8 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l1671_167169


namespace NUMINAMATH_CALUDE_translation_problem_l1671_167144

def complex_translation (z w : ℂ) : ℂ := z + w

theorem translation_problem (t : ℂ → ℂ) (h : t (1 + 3*I) = 4 + 7*I) :
  t (2 + 6*I) = 5 + 10*I :=
by
  sorry

end NUMINAMATH_CALUDE_translation_problem_l1671_167144


namespace NUMINAMATH_CALUDE_complex_counterexample_l1671_167186

theorem complex_counterexample : ∃ z₁ z₂ : ℂ, (Complex.abs z₁ = Complex.abs z₂) ∧ (z₁^2 ≠ z₂^2) := by
  sorry

end NUMINAMATH_CALUDE_complex_counterexample_l1671_167186


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l1671_167102

theorem tan_alpha_plus_pi_fourth (α : Real) 
  (h : 6 * Real.sin α * Real.cos α = 1 + Real.cos (2 * α)) : 
  Real.tan (α + π/4) = 2 ∨ Real.tan (α + π/4) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l1671_167102


namespace NUMINAMATH_CALUDE_sandro_has_three_sons_l1671_167126

/-- Represents the number of sons Sandro has -/
def num_sons : ℕ := 3

/-- Represents the number of daughters Sandro has -/
def num_daughters : ℕ := 6 * num_sons

/-- The total number of children Sandro has -/
def total_children : ℕ := 21

/-- Theorem stating that Sandro has 3 sons, given the conditions -/
theorem sandro_has_three_sons : 
  (num_daughters = 6 * num_sons) ∧ 
  (num_sons + num_daughters = total_children) → 
  num_sons = 3 := by
  sorry

end NUMINAMATH_CALUDE_sandro_has_three_sons_l1671_167126


namespace NUMINAMATH_CALUDE_peach_count_difference_l1671_167105

/-- The number of red peaches in the basket -/
def red_peaches : ℕ := 17

/-- The number of green peaches in the basket -/
def green_peaches : ℕ := 16

/-- The difference between the number of red peaches and green peaches -/
def peach_difference : ℕ := red_peaches - green_peaches

theorem peach_count_difference : peach_difference = 1 := by
  sorry

end NUMINAMATH_CALUDE_peach_count_difference_l1671_167105


namespace NUMINAMATH_CALUDE_average_salary_problem_l1671_167197

/-- The average monthly salary problem -/
theorem average_salary_problem (initial_average : ℚ) (old_supervisor_salary : ℚ) 
  (new_supervisor_salary : ℚ) (num_workers : ℕ) (total_people : ℕ) 
  (h1 : initial_average = 430)
  (h2 : old_supervisor_salary = 870)
  (h3 : new_supervisor_salary = 780)
  (h4 : num_workers = 8)
  (h5 : total_people = num_workers + 1) :
  let total_initial_salary := initial_average * total_people
  let workers_salary := total_initial_salary - old_supervisor_salary
  let new_total_salary := workers_salary + new_supervisor_salary
  let new_average_salary := new_total_salary / total_people
  new_average_salary = 420 := by sorry

end NUMINAMATH_CALUDE_average_salary_problem_l1671_167197


namespace NUMINAMATH_CALUDE_n_has_nine_digits_l1671_167181

/-- The smallest positive integer satisfying the given conditions -/
def n : ℕ := sorry

/-- n is divisible by 30 -/
axiom n_div_30 : 30 ∣ n

/-- n^2 is a perfect cube -/
axiom n_sq_cube : ∃ k : ℕ, n^2 = k^3

/-- n^3 is a perfect square -/
axiom n_cube_square : ∃ k : ℕ, n^3 = k^2

/-- n is the smallest positive integer satisfying the conditions -/
axiom n_smallest : ∀ m : ℕ, m > 0 → (30 ∣ m) → (∃ k : ℕ, m^2 = k^3) → (∃ k : ℕ, m^3 = k^2) → m ≥ n

/-- The number of digits in n -/
def digits (x : ℕ) : ℕ := sorry

theorem n_has_nine_digits : digits n = 9 := by sorry

end NUMINAMATH_CALUDE_n_has_nine_digits_l1671_167181


namespace NUMINAMATH_CALUDE_min_value_quadratic_ratio_l1671_167100

theorem min_value_quadratic_ratio (a b : ℝ) (h1 : b > 0) 
  (h2 : b^2 - 4*a = 0) : 
  (∀ x : ℝ, (a*x^2 + b*x + 1) / b ≥ 2) ∧ 
  (∃ x : ℝ, (a*x^2 + b*x + 1) / b = 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_ratio_l1671_167100


namespace NUMINAMATH_CALUDE_julie_total_earnings_l1671_167120

/-- Calculates Julie's total earnings for September and October based on her landscaping business rates and hours worked. -/
theorem julie_total_earnings (
  -- September hours
  small_lawn_sept : ℕ) (large_lawn_sept : ℕ) (simple_garden_sept : ℕ) (complex_garden_sept : ℕ)
  (small_tree_sept : ℕ) (large_tree_sept : ℕ) (mulch_sept : ℕ)
  -- Rates
  (small_lawn_rate : ℕ) (large_lawn_rate : ℕ) (simple_garden_rate : ℕ) (complex_garden_rate : ℕ)
  (small_tree_rate : ℕ) (large_tree_rate : ℕ) (mulch_rate : ℕ)
  -- Given conditions
  (h1 : small_lawn_sept = 10) (h2 : large_lawn_sept = 15) (h3 : simple_garden_sept = 2)
  (h4 : complex_garden_sept = 1) (h5 : small_tree_sept = 5) (h6 : large_tree_sept = 5)
  (h7 : mulch_sept = 5)
  (h8 : small_lawn_rate = 4) (h9 : large_lawn_rate = 6) (h10 : simple_garden_rate = 8)
  (h11 : complex_garden_rate = 10) (h12 : small_tree_rate = 10) (h13 : large_tree_rate = 15)
  (h14 : mulch_rate = 12) :
  -- Theorem statement
  (small_lawn_rate * small_lawn_sept + large_lawn_rate * large_lawn_sept +
   simple_garden_rate * simple_garden_sept + complex_garden_rate * complex_garden_sept +
   small_tree_rate * small_tree_sept + large_tree_rate * large_tree_sept +
   mulch_rate * mulch_sept) +
  ((small_lawn_rate * small_lawn_sept + large_lawn_rate * large_lawn_sept +
    simple_garden_rate * simple_garden_sept + complex_garden_rate * complex_garden_sept +
    small_tree_rate * small_tree_sept + large_tree_rate * large_tree_sept +
    mulch_rate * mulch_sept) * 3 / 2) = 8525/10 := by
  sorry

end NUMINAMATH_CALUDE_julie_total_earnings_l1671_167120


namespace NUMINAMATH_CALUDE_fraction_simplification_l1671_167121

theorem fraction_simplification (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a ≠ b) :
  (a^3 - b^3) / (a * b) - (a * b - c * b) / (a * b - a^2) = (a^2 + a * b + b^2 + a * c) / (a * b) :=
sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1671_167121


namespace NUMINAMATH_CALUDE_max_marbles_for_score_l1671_167164

/-- Represents the size of a marble -/
inductive MarbleSize
| Small
| Medium
| Large

/-- Represents a hole with its score -/
structure Hole :=
  (number : Nat)
  (score : Nat)

/-- Represents the game setup -/
structure GameSetup :=
  (holes : List Hole)
  (maxMarbles : Nat)
  (totalScore : Nat)

/-- Checks if a marble can go through a hole -/
def canGoThrough (size : MarbleSize) (hole : Hole) : Bool :=
  match size with
  | MarbleSize.Small => true
  | MarbleSize.Medium => hole.number ≥ 3
  | MarbleSize.Large => hole.number = 5

/-- Represents a valid game configuration -/
structure GameConfig :=
  (smallMarbles : List Hole)
  (mediumMarbles : List Hole)
  (largeMarbles : List Hole)

/-- Calculates the total score for a game configuration -/
def totalScore (config : GameConfig) : Nat :=
  (config.smallMarbles.map (·.score)).sum +
  (config.mediumMarbles.map (·.score)).sum +
  (config.largeMarbles.map (·.score)).sum

/-- Calculates the total number of marbles used in a game configuration -/
def totalMarbles (config : GameConfig) : Nat :=
  config.smallMarbles.length +
  config.mediumMarbles.length +
  config.largeMarbles.length

/-- The main theorem to prove -/
theorem max_marbles_for_score (setup : GameSetup) :
  (∃ (config : GameConfig),
    totalScore config = setup.totalScore ∧
    totalMarbles config ≤ setup.maxMarbles ∧
    (∀ (other : GameConfig),
      totalScore other = setup.totalScore →
      totalMarbles other ≤ totalMarbles config)) →
  (∃ (maxConfig : GameConfig),
    totalScore maxConfig = setup.totalScore ∧
    totalMarbles maxConfig = 14 ∧
    (∀ (other : GameConfig),
      totalScore other = setup.totalScore →
      totalMarbles other ≤ 14)) :=
by sorry

end NUMINAMATH_CALUDE_max_marbles_for_score_l1671_167164


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l1671_167115

theorem fraction_equals_zero (x : ℝ) : 
  (|x| - 2) / (x - 2) = 0 ∧ x - 2 ≠ 0 ↔ x = -2 := by
sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l1671_167115


namespace NUMINAMATH_CALUDE_outfit_count_l1671_167137

/-- The number of red shirts -/
def red_shirts : ℕ := 7

/-- The number of blue shirts -/
def blue_shirts : ℕ := 7

/-- The number of pairs of pants -/
def pants : ℕ := 10

/-- The number of green hats -/
def green_hats : ℕ := 9

/-- The number of red hats -/
def red_hats : ℕ := 9

/-- Each piece of clothing is distinct -/
axiom distinct_clothing : red_shirts + blue_shirts + pants + green_hats + red_hats = red_shirts + blue_shirts + pants + green_hats + red_hats

/-- The number of outfits where the shirt and hat are never the same color -/
def num_outfits : ℕ := red_shirts * pants * green_hats + blue_shirts * pants * red_hats

theorem outfit_count : num_outfits = 1260 := by
  sorry

end NUMINAMATH_CALUDE_outfit_count_l1671_167137


namespace NUMINAMATH_CALUDE_plane_parallel_criterion_l1671_167165

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation between planes
variable (plane_parallel : Plane → Plane → Prop)

-- Define the parallel relation between a line and a plane
variable (line_parallel_plane : Line → Plane → Prop)

-- Define the relation of a line being in a plane
variable (line_in_plane : Line → Plane → Prop)

-- Theorem statement
theorem plane_parallel_criterion
  (α β : Plane)
  (h : ∀ l : Line, line_in_plane l α → line_parallel_plane l β) :
  plane_parallel α β :=
sorry

end NUMINAMATH_CALUDE_plane_parallel_criterion_l1671_167165


namespace NUMINAMATH_CALUDE_simplify_fraction_l1671_167180

theorem simplify_fraction : (4^4 + 4^2) / (4^3 - 4) = 17 / 3 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1671_167180


namespace NUMINAMATH_CALUDE_vector_magnitude_l1671_167125

/-- Given vectors a and b, where a is perpendicular to (2a - b), prove that the magnitude of b is 2√10 -/
theorem vector_magnitude (a b : ℝ × ℝ) (h : a = (1, 2)) (h' : b.1 = -2)
  (h'' : a.1 * (2 * a.1 - b.1) + a.2 * (2 * a.2 - b.2) = 0) :
  Real.sqrt (b.1^2 + b.2^2) = 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l1671_167125


namespace NUMINAMATH_CALUDE_max_z_value_l1671_167157

theorem max_z_value (x y z : ℝ) (sum_eq : x + y + z = 9) (prod_eq : x*y + y*z + z*x = 24) :
  z ≤ 5 := by
sorry

end NUMINAMATH_CALUDE_max_z_value_l1671_167157


namespace NUMINAMATH_CALUDE_decreasing_f_implies_a_range_l1671_167139

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (3*a - 1)*x + 4*a else Real.log x / Real.log a

-- State the theorem
theorem decreasing_f_implies_a_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x > f a y) →
  a ∈ Set.Icc (1/7 : ℝ) (1/3 : ℝ) ∧ a ≠ 1/3 := by
  sorry


end NUMINAMATH_CALUDE_decreasing_f_implies_a_range_l1671_167139


namespace NUMINAMATH_CALUDE_line_slope_and_intercept_l1671_167155

theorem line_slope_and_intercept :
  ∀ (k b : ℝ),
  (∀ x y : ℝ, 3 * x + 2 * y + 6 = 0 ↔ y = k * x + b) →
  k = -3/2 ∧ b = -3 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_and_intercept_l1671_167155


namespace NUMINAMATH_CALUDE_tangent_line_at_one_max_value_min_value_l1671_167119

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 6*x + 5

-- Define the interval
def interval : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- Theorem for the tangent line at x = 1
theorem tangent_line_at_one :
  ∃ (m b : ℝ), ∀ x y, y = m*x + b ↔ 3*x - y - 3 = 0 :=
sorry

-- Theorem for the maximum value
theorem max_value :
  ∃ x ∈ interval, f x = 5 + 4 * Real.sqrt 2 ∧ ∀ y ∈ interval, f y ≤ f x :=
sorry

-- Theorem for the minimum value
theorem min_value :
  ∃ x ∈ interval, f x = 5 - 4 * Real.sqrt 2 ∧ ∀ y ∈ interval, f y ≥ f x :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_max_value_min_value_l1671_167119


namespace NUMINAMATH_CALUDE_curve_transformation_l1671_167146

theorem curve_transformation (x : ℝ) : 
  Real.sin (2 * x + 2 * Real.pi / 3) = Real.cos (2 * (x - Real.pi / 12)) := by
  sorry

end NUMINAMATH_CALUDE_curve_transformation_l1671_167146


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1671_167127

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, ax^2 + b*x + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) → a - b = -10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1671_167127


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l1671_167111

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_general_term
  (a : ℕ → ℝ)
  (d : ℝ)
  (h1 : arithmetic_sequence a d)
  (h2 : d = 3)
  (h3 : a 4 = 14) :
  ∀ n, a n = 3 * n + 2 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l1671_167111


namespace NUMINAMATH_CALUDE_subcommittees_count_l1671_167108

def planning_committee_size : ℕ := 10
def teacher_count : ℕ := 4
def subcommittee_size : ℕ := 4

/-- The number of distinct subcommittees with at least one teacher -/
def subcommittees_with_teacher : ℕ :=
  Nat.choose planning_committee_size subcommittee_size -
  Nat.choose (planning_committee_size - teacher_count) subcommittee_size

theorem subcommittees_count :
  subcommittees_with_teacher = 195 :=
sorry

end NUMINAMATH_CALUDE_subcommittees_count_l1671_167108


namespace NUMINAMATH_CALUDE_race_head_start_l1671_167187

theorem race_head_start (L : ℝ) (Va Vb : ℝ) (h : Va = (51 / 44) * Vb) :
  let H := L * (7 / 51)
  (L / Va) = ((L - H) / Vb) := by
  sorry

end NUMINAMATH_CALUDE_race_head_start_l1671_167187


namespace NUMINAMATH_CALUDE_equation_solution_l1671_167131

theorem equation_solution : ∃ x : ℝ, (17.28 / x) / (3.6 * 0.2) = 2 ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1671_167131


namespace NUMINAMATH_CALUDE_sin_cos_identity_l1671_167152

theorem sin_cos_identity : 
  Real.sin (75 * π / 180) * Real.cos (30 * π / 180) - 
  Real.sin (15 * π / 180) * Real.sin (150 * π / 180) = 
  Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l1671_167152


namespace NUMINAMATH_CALUDE_half_dollars_in_tip_jar_l1671_167114

def nickel_value : ℚ := 5 / 100
def dime_value : ℚ := 10 / 100
def half_dollar_value : ℚ := 50 / 100

def nickels_shining : ℕ := 3
def dimes_shining : ℕ := 13
def dimes_tip : ℕ := 7
def total_amount : ℚ := 665 / 100

theorem half_dollars_in_tip_jar :
  ∃ (half_dollars : ℕ),
    (nickels_shining : ℚ) * nickel_value +
    (dimes_shining : ℚ) * dime_value +
    (dimes_tip : ℚ) * dime_value +
    (half_dollars : ℚ) * half_dollar_value = total_amount ∧
    half_dollars = 9 :=
by sorry

end NUMINAMATH_CALUDE_half_dollars_in_tip_jar_l1671_167114


namespace NUMINAMATH_CALUDE_circle_existence_condition_l1671_167104

theorem circle_existence_condition (x y c : ℝ) : 
  (∃ h k r, (x - h)^2 + (y - k)^2 = r^2 ∧ r > 0) ↔ 
  (x^2 + y^2 + 4*x - 2*y - 5*c = 0 → c > -1) :=
sorry

end NUMINAMATH_CALUDE_circle_existence_condition_l1671_167104


namespace NUMINAMATH_CALUDE_hockey_players_l1671_167160

theorem hockey_players (n : ℕ) : 
  n < 30 ∧ 
  2 ∣ n ∧ 
  4 ∣ n ∧ 
  7 ∣ n → 
  n / 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_hockey_players_l1671_167160


namespace NUMINAMATH_CALUDE_product_of_four_is_perfect_square_l1671_167109

theorem product_of_four_is_perfect_square 
  (nums : Finset ℕ) 
  (h_card : nums.card = 48) 
  (h_primes : (nums.prod id).factorization.support.card = 10) : 
  ∃ (subset : Finset ℕ), subset ⊆ nums ∧ subset.card = 4 ∧ 
  ∃ (m : ℕ), (subset.prod id) = m^2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_four_is_perfect_square_l1671_167109


namespace NUMINAMATH_CALUDE_smallest_natural_with_last_four_digits_l1671_167178

theorem smallest_natural_with_last_four_digits : ∃ (N : ℕ), 
  (∀ (k : ℕ), k < N → ¬(47 * k ≡ 1969 [ZMOD 10000])) ∧ 
  (47 * N ≡ 1969 [ZMOD 10000]) := by
  sorry

end NUMINAMATH_CALUDE_smallest_natural_with_last_four_digits_l1671_167178


namespace NUMINAMATH_CALUDE_salary_before_raise_l1671_167184

theorem salary_before_raise (new_salary : ℝ) (increase_percentage : ℝ) 
  (h1 : new_salary = 70)
  (h2 : increase_percentage = 0.40) :
  let original_salary := new_salary / (1 + increase_percentage)
  original_salary = 50 := by
sorry

end NUMINAMATH_CALUDE_salary_before_raise_l1671_167184


namespace NUMINAMATH_CALUDE_max_value_sum_of_square_roots_l1671_167141

theorem max_value_sum_of_square_roots (x : ℝ) (h : x ∈ Set.Icc (-36) 36) :
  Real.sqrt (36 + x) + Real.sqrt (36 - x) ≤ 12 ∧
  ∃ y ∈ Set.Icc (-36) 36, Real.sqrt (36 + y) + Real.sqrt (36 - y) = 12 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_of_square_roots_l1671_167141


namespace NUMINAMATH_CALUDE_quadratic_max_value_l1671_167110

/-- The quadratic function f(x) = -x^2 - 2x + 3 -/
def f (x : ℝ) : ℝ := -x^2 - 2*x + 3

theorem quadratic_max_value (a : ℝ) :
  (∀ x ∈ Set.Icc a 2, f x ≤ 15/4) ∧ 
  (∃ x ∈ Set.Icc a 2, f x = 15/4) →
  a = -1/2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l1671_167110


namespace NUMINAMATH_CALUDE_family_hard_shell_tacos_l1671_167159

/-- The number of hard shell tacos bought by a family -/
def hard_shell_tacos : ℕ := sorry

/-- The price of a soft taco in dollars -/
def soft_taco_price : ℕ := 2

/-- The price of a hard shell taco in dollars -/
def hard_shell_taco_price : ℕ := 5

/-- The number of soft tacos bought by the family -/
def family_soft_tacos : ℕ := 3

/-- The number of additional customers -/
def additional_customers : ℕ := 10

/-- The number of soft tacos bought by each additional customer -/
def soft_tacos_per_customer : ℕ := 2

/-- The total revenue in dollars -/
def total_revenue : ℕ := 66

theorem family_hard_shell_tacos :
  hard_shell_tacos = 4 :=
by sorry

end NUMINAMATH_CALUDE_family_hard_shell_tacos_l1671_167159


namespace NUMINAMATH_CALUDE_expression_value_l1671_167192

theorem expression_value (x y : ℝ) (hx : x = 3) (hy : y = 4) :
  (x^3 + 3*y^3) / 9 = 24.33 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1671_167192


namespace NUMINAMATH_CALUDE_remaining_money_correct_l1671_167117

structure Currency where
  usd : ℚ
  eur : ℚ
  gbp : ℚ

def initial_amount : Currency := ⟨5.10, 8.75, 10.30⟩

def spend_usd (amount : ℚ) (c : Currency) : Currency :=
  ⟨c.usd - amount, c.eur, c.gbp⟩

def spend_eur (amount : ℚ) (c : Currency) : Currency :=
  ⟨c.usd, c.eur - amount, c.gbp⟩

def exchange_gbp_to_eur (gbp_amount : ℚ) (eur_amount : ℚ) (c : Currency) : Currency :=
  ⟨c.usd, c.eur + eur_amount, c.gbp - gbp_amount⟩

def final_amount : Currency :=
  initial_amount
  |> spend_usd 1.05
  |> spend_usd 2.00
  |> spend_eur 3.25
  |> exchange_gbp_to_eur 5.00 5.60

theorem remaining_money_correct :
  final_amount.usd = 2.05 ∧
  final_amount.eur = 11.10 ∧
  final_amount.gbp = 5.30 := by
  sorry


end NUMINAMATH_CALUDE_remaining_money_correct_l1671_167117


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1671_167151

-- Define the number of diagonals in a regular polygon
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- Theorem statement
theorem regular_polygon_sides :
  ∀ n : ℕ, n ≥ 3 →
  (num_diagonals n + 2 * n = n^2) → n = 3 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1671_167151


namespace NUMINAMATH_CALUDE_simplify_expression_l1671_167124

theorem simplify_expression :
  81 * ((5 + 1/3) - (3 + 1/4)) / ((4 + 1/2) + (2 + 2/5)) = 225/92 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1671_167124


namespace NUMINAMATH_CALUDE_twelve_point_polygons_l1671_167188

/-- The number of distinct convex polygons with 3 or more sides that can be formed from n points on a circle -/
def convex_polygons (n : ℕ) : ℕ :=
  2^n - 1 - n - (n.choose 2)

theorem twelve_point_polygons :
  convex_polygons 12 = 4017 := by
  sorry

end NUMINAMATH_CALUDE_twelve_point_polygons_l1671_167188


namespace NUMINAMATH_CALUDE_roger_tray_capacity_l1671_167175

/-- The number of trays Roger can carry at a time -/
def trays_per_trip : ℕ := sorry

/-- The number of trips Roger made -/
def num_trips : ℕ := 3

/-- The number of trays Roger picked up from the first table -/
def trays_table1 : ℕ := 10

/-- The number of trays Roger picked up from the second table -/
def trays_table2 : ℕ := 2

/-- The total number of trays Roger picked up -/
def total_trays : ℕ := trays_table1 + trays_table2

theorem roger_tray_capacity :
  trays_per_trip * num_trips = total_trays ∧ trays_per_trip = 4 := by
  sorry

end NUMINAMATH_CALUDE_roger_tray_capacity_l1671_167175


namespace NUMINAMATH_CALUDE_car_sale_percentage_l1671_167185

theorem car_sale_percentage (P x : ℝ) : 
  P - 2500 = 30000 →
  x / 100 * P = 30000 - 4000 →
  x = 80 := by
sorry

end NUMINAMATH_CALUDE_car_sale_percentage_l1671_167185


namespace NUMINAMATH_CALUDE_count_seven_digit_phone_numbers_l1671_167130

/-- The number of different seven-digit phone numbers where the first digit cannot be zero -/
def sevenDigitPhoneNumbers : ℕ := 9 * (10 ^ 6)

/-- Theorem stating that the number of different seven-digit phone numbers
    where the first digit cannot be zero is equal to 9 * 10^6 -/
theorem count_seven_digit_phone_numbers :
  sevenDigitPhoneNumbers = 9 * (10 ^ 6) := by
  sorry

end NUMINAMATH_CALUDE_count_seven_digit_phone_numbers_l1671_167130


namespace NUMINAMATH_CALUDE_simple_interest_problem_l1671_167191

theorem simple_interest_problem (P R : ℝ) (h1 : P > 0) (h2 : R > 0) :
  (P * (R + 5) * 10) / 100 = (P * R * 10) / 100 + 150 →
  P = 300 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l1671_167191


namespace NUMINAMATH_CALUDE_twelfth_odd_multiple_of_five_l1671_167183

theorem twelfth_odd_multiple_of_five : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 2 = 1 ∧ 
  n % 5 = 0 ∧
  (∃ k : ℕ, k = 12 ∧ 
    n = (Finset.filter (λ x => x % 2 = 1 ∧ x % 5 = 0) (Finset.range n)).card) ∧
  n = 115 := by
sorry

end NUMINAMATH_CALUDE_twelfth_odd_multiple_of_five_l1671_167183


namespace NUMINAMATH_CALUDE_arithmetic_and_geometric_sequences_l1671_167123

/-- Arithmetic sequence sum -/
def arithmetic_sum (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * a₁ + (n * (n - 1) * d) / 2

/-- Geometric sequence term -/
def geometric_term (a₁ : ℤ) (q : ℤ) (n : ℕ) : ℤ :=
  a₁ * q ^ (n - 1)

theorem arithmetic_and_geometric_sequences :
  (arithmetic_sum (-2) 4 8 = 96) ∧
  (geometric_term 1 3 7 = 729) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_and_geometric_sequences_l1671_167123


namespace NUMINAMATH_CALUDE_exists_unique_N_l1671_167107

theorem exists_unique_N : ∃ N : ℤ, N = 1719 ∧
  ∀ a b : ℤ, (N / 2 - a = b - N / 2) →
    ((∃ m n : ℕ+, a = 19 * m + 85 * n) ∨ (∃ m n : ℕ+, b = 19 * m + 85 * n)) ∧
    ¬((∃ m n : ℕ+, a = 19 * m + 85 * n) ∧ (∃ m n : ℕ+, b = 19 * m + 85 * n)) :=
by sorry

end NUMINAMATH_CALUDE_exists_unique_N_l1671_167107


namespace NUMINAMATH_CALUDE_quadratic_contradiction_l1671_167170

theorem quadratic_contradiction : ¬ ∃ (a b c : ℝ), 
  ((a > 0 ∧ b > 0 ∧ c < 0) ∨ (a > 0 ∧ b < 0 ∧ c > 0) ∨ (a < 0 ∧ b > 0 ∧ c > 0)) ∧
  ((a < 0 ∧ b < 0 ∧ c > 0) ∨ (a < 0 ∧ b > 0 ∧ c < 0) ∨ (a > 0 ∧ b < 0 ∧ c < 0)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_contradiction_l1671_167170


namespace NUMINAMATH_CALUDE_single_female_fraction_l1671_167112

theorem single_female_fraction (total : ℕ) (h1 : total > 0) :
  let male_percent : ℚ := 70 / 100
  let married_percent : ℚ := 30 / 100
  let male_married_fraction : ℚ := 1 / 7
  let male_count := (male_percent * total).floor
  let female_count := total - male_count
  let married_count := (married_percent * total).floor
  let male_married_count := (male_married_fraction * male_count).floor
  let female_married_count := married_count - male_married_count
  let single_female_count := female_count - female_married_count
  (single_female_count : ℚ) / female_count = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_single_female_fraction_l1671_167112


namespace NUMINAMATH_CALUDE_expand_and_subtract_fraction_division_l1671_167196

-- Part 1
theorem expand_and_subtract (m n : ℝ) :
  (2*m + 3*n)^2 - (2*m + n)*(2*m - n) = 12*m*n + 10*n^2 := by sorry

-- Part 2
theorem fraction_division (x y : ℝ) (hx : x ≠ 0) (hxy : x ≠ y) :
  (x - y) / x / (x + (y^2 - 2*x*y) / x) = 1 / (x - y) := by sorry

end NUMINAMATH_CALUDE_expand_and_subtract_fraction_division_l1671_167196


namespace NUMINAMATH_CALUDE_solve_composite_function_equation_l1671_167136

theorem solve_composite_function_equation (a : ℝ) 
  (h : ℝ → ℝ) (f : ℝ → ℝ) (g : ℝ → ℝ)
  (h_def : ∀ x, h x = x + 2)
  (f_def : ∀ x, f x = 2 * x + 3)
  (g_def : ∀ x, g x = x^2 - 5)
  (a_pos : a > 0)
  (eq : h (f (g a)) = 12) :
  a = Real.sqrt (17 / 2) := by
sorry

end NUMINAMATH_CALUDE_solve_composite_function_equation_l1671_167136


namespace NUMINAMATH_CALUDE_sum_x_2y_equals_5_l1671_167156

theorem sum_x_2y_equals_5 (x y : ℕ+) 
  (h : x^3 + 3*x^2*y + 8*x*y^2 + 6*y^3 = 87) : 
  x + 2*y = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_x_2y_equals_5_l1671_167156


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l1671_167118

/-- A polynomial is monic if its leading coefficient is 1. -/
def Monic (p : Polynomial ℤ) : Prop :=
  p.leadingCoeff = 1

/-- A polynomial is non-constant if its degree is greater than 0. -/
def NonConstant (p : Polynomial ℤ) : Prop :=
  p.degree > 0

/-- P(n) divides Q(n) in ℤ -/
def DividesAtInteger (P Q : Polynomial ℤ) (n : ℤ) : Prop :=
  ∃ k : ℤ, Q.eval n = k * P.eval n

/-- There are infinitely many integers n such that P(n) divides Q(n) in ℤ -/
def InfinitelyManyDivisions (P Q : Polynomial ℤ) : Prop :=
  ∀ m : ℕ, ∃ n : ℤ, n > m ∧ DividesAtInteger P Q n

theorem polynomial_division_theorem (P Q : Polynomial ℤ) 
  (h_monic_P : Monic P) (h_monic_Q : Monic Q)
  (h_non_const_P : NonConstant P) (h_non_const_Q : NonConstant Q)
  (h_infinite_divisions : InfinitelyManyDivisions P Q) :
  P ∣ Q :=
sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l1671_167118


namespace NUMINAMATH_CALUDE_base7_to_base10_conversion_l1671_167199

/-- Converts a base 7 number to base 10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The given number in base 7 --/
def base7Number : List Nat := [4, 3, 6, 2, 5]

/-- Theorem: The base 10 equivalent of 52634₇ is 13010 --/
theorem base7_to_base10_conversion :
  base7ToBase10 base7Number = 13010 := by
  sorry

end NUMINAMATH_CALUDE_base7_to_base10_conversion_l1671_167199


namespace NUMINAMATH_CALUDE_parallel_vectors_solution_perpendicular_vectors_solution_l1671_167161

-- Define the vectors
def a (x : ℝ) : ℝ × ℝ := (1, 2*x)
def b (x : ℝ) : ℝ × ℝ := (x, 3)
def c : ℝ × ℝ := (-2, 0)

-- Question 1: Parallel vectors condition
def parallel_condition (x : ℝ) : Prop :=
  (1 + 2*x) * 4*x = 4*(2*x + 6)

-- Question 2: Perpendicular vectors condition
def perpendicular_condition (x : ℝ) : Prop :=
  8*x^2 + 32*x + 4 = 0

-- Theorem for parallel vectors
theorem parallel_vectors_solution :
  ∃ x₁ x₂ : ℝ, x₁ = 2 ∧ x₂ = -3/2 ∧ parallel_condition x₁ ∧ parallel_condition x₂ :=
sorry

-- Theorem for perpendicular vectors
theorem perpendicular_vectors_solution :
  ∃ x₁ x₂ : ℝ, x₁ = (-4 + Real.sqrt 14)/2 ∧ x₂ = (-4 - Real.sqrt 14)/2 ∧
  perpendicular_condition x₁ ∧ perpendicular_condition x₂ :=
sorry

end NUMINAMATH_CALUDE_parallel_vectors_solution_perpendicular_vectors_solution_l1671_167161


namespace NUMINAMATH_CALUDE_evaluate_expression_l1671_167116

theorem evaluate_expression : 3 - 6 * (7 - 2^3)^2 = -3 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1671_167116


namespace NUMINAMATH_CALUDE_club_participation_theorem_l1671_167101

universe u

def club_participation (n : ℕ) : Prop :=
  ∃ (U A B C : Finset ℕ),
    Finset.card U = 40 ∧
    Finset.card A = 22 ∧
    Finset.card B = 16 ∧
    Finset.card C = 20 ∧
    Finset.card (A ∩ B) = 8 ∧
    Finset.card (B ∩ C) = 6 ∧
    Finset.card (A ∩ C) = 10 ∧
    Finset.card (A ∩ B ∩ C) = 2 ∧
    Finset.card (A \ (B ∪ C) ∪ B \ (A ∪ C) ∪ C \ (A ∪ B)) = 16 ∧
    Finset.card (U \ (A ∪ B ∪ C)) = 4

theorem club_participation_theorem : club_participation 40 := by
  sorry

end NUMINAMATH_CALUDE_club_participation_theorem_l1671_167101


namespace NUMINAMATH_CALUDE_adjacent_pair_properties_l1671_167134

/-- Definition of "adjacent number pairs" -/
def adjacent_pair (m n : ℚ) : Prop :=
  m / 2 + n / 5 = (m + n) / 7

theorem adjacent_pair_properties :
  ∃ (m n : ℚ),
    /- Part 1 -/
    (adjacent_pair 2 n → n = -25 / 2) ∧
    /- Part 2① -/
    (adjacent_pair m n → m = -4 * n / 25) ∧
    /- Part 2② -/
    (adjacent_pair m n ∧ 25 * m + n = 6 → m = 8 / 25 ∧ n = -2) := by
  sorry

end NUMINAMATH_CALUDE_adjacent_pair_properties_l1671_167134


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1671_167177

theorem polynomial_divisibility (m n : ℕ) :
  ∃ q : Polynomial ℚ, x^(3*m+2) + (-x^2 - 1)^(3*n+1) + 1 = (x^2 + x + 1) * q := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1671_167177


namespace NUMINAMATH_CALUDE_product_of_two_numbers_l1671_167190

theorem product_of_two_numbers (x y : ℝ) (h1 : x - y = 12) (h2 : x^2 + y^2 = 340) : 
  x * y = 97.9450625 := by
sorry

end NUMINAMATH_CALUDE_product_of_two_numbers_l1671_167190


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l1671_167158

theorem intersection_complement_equality (U A B : Set Nat) : 
  U = {1, 2, 3, 4, 5} → 
  A = {1, 3} → 
  B = {2, 5} → 
  A ∩ (U \ B) = {1, 3} := by
sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l1671_167158


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1671_167182

theorem complex_equation_solution :
  ∀ (z : ℂ), (2 + Complex.I) * z = 5 * Complex.I → z = 1 + 2 * Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1671_167182


namespace NUMINAMATH_CALUDE_f_at_two_l1671_167172

noncomputable section

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- f' is the derivative of f
axiom is_derivative : ∀ x, deriv f x = f' x

-- f(x) = 2xf'(2) + ln(x-1)
axiom f_def : ∀ x, f x = 2 * x * (f' 2) + Real.log (x - 1)

theorem f_at_two : f 2 = -4 := by sorry

end NUMINAMATH_CALUDE_f_at_two_l1671_167172


namespace NUMINAMATH_CALUDE_train_speed_l1671_167198

/- Define the train length in meters -/
def train_length : ℝ := 160

/- Define the time taken to pass in seconds -/
def passing_time : ℝ := 8

/- Define the conversion factor from m/s to km/h -/
def ms_to_kmh : ℝ := 3.6

/- Theorem statement -/
theorem train_speed : 
  (train_length / passing_time) * ms_to_kmh = 72 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l1671_167198


namespace NUMINAMATH_CALUDE_saramago_readers_l1671_167129

theorem saramago_readers (total_workers : ℕ) (kureishi_readers : ℚ) 
  (both_readers : ℕ) (s : ℚ) : 
  total_workers = 40 →
  kureishi_readers = 5/8 →
  both_readers = 2 →
  (s * total_workers - both_readers - 1 : ℚ) = 
    (total_workers * (1 - kureishi_readers - s) : ℚ) →
  s = 9/40 := by sorry

end NUMINAMATH_CALUDE_saramago_readers_l1671_167129


namespace NUMINAMATH_CALUDE_M_lower_bound_l1671_167179

theorem M_lower_bound (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) :
  (1/a - 1) * (1/b - 1) * (1/c - 1) ≥ 8 := by
sorry

end NUMINAMATH_CALUDE_M_lower_bound_l1671_167179


namespace NUMINAMATH_CALUDE_complex_square_l1671_167149

/-- Given that i^2 = -1, prove that (3 - 4i)^2 = 5 - 24i -/
theorem complex_square (i : ℂ) (h : i^2 = -1) : (3 - 4*i)^2 = 5 - 24*i := by
  sorry

end NUMINAMATH_CALUDE_complex_square_l1671_167149


namespace NUMINAMATH_CALUDE_find_number_l1671_167147

theorem find_number : ∃! x : ℝ, 7 * x + 37 = 100 ∧ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l1671_167147


namespace NUMINAMATH_CALUDE_opposite_of_negative_eight_l1671_167193

-- Define the concept of opposite
def opposite (x : Int) : Int := -x

-- State the theorem
theorem opposite_of_negative_eight :
  opposite (-8) = 8 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_eight_l1671_167193


namespace NUMINAMATH_CALUDE_tan_2alpha_values_l1671_167189

theorem tan_2alpha_values (α : ℝ) (h : 2 * Real.sin (2 * α) = 1 + Real.cos (2 * α)) :
  Real.tan (2 * α) = 4/3 ∨ Real.tan (2 * α) = 0 := by
  sorry

end NUMINAMATH_CALUDE_tan_2alpha_values_l1671_167189


namespace NUMINAMATH_CALUDE_simplify_fraction_l1671_167195

theorem simplify_fraction : (121 / 9801 : ℚ) * 22 = 22 / 81 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1671_167195


namespace NUMINAMATH_CALUDE_nathaniel_win_probability_l1671_167122

-- Define the game state
structure GameState where
  tally : ℕ
  currentPlayer : Bool  -- True for Nathaniel, False for Obediah

-- Define the probability of winning for a given game state
def winProbability (state : GameState) : ℚ :=
  sorry

-- Define the theorem
theorem nathaniel_win_probability :
  winProbability ⟨0, true⟩ = 5/11 := by sorry

end NUMINAMATH_CALUDE_nathaniel_win_probability_l1671_167122


namespace NUMINAMATH_CALUDE_tea_mixture_price_l1671_167153

theorem tea_mixture_price (price1 price2 price3 mixture_price : ℝ) 
  (h1 : price1 = 126)
  (h2 : price3 = 175.5)
  (h3 : mixture_price = 153)
  (h4 : price1 + price2 + 2 * price3 = 4 * mixture_price) :
  price2 = 135 := by
  sorry

end NUMINAMATH_CALUDE_tea_mixture_price_l1671_167153


namespace NUMINAMATH_CALUDE_sequence_formula_and_sum_bound_l1671_167145

def S (n : ℕ) : ℚ := 3/2 * n^2 - 1/2 * n

def a (n : ℕ+) : ℚ := 3 * n - 2

def T (n : ℕ+) : ℚ := 1 - 1 / (3 * n + 1)

theorem sequence_formula_and_sum_bound :
  (∀ n : ℕ+, a n = S n - S (n-1)) ∧
  (∃ m : ℕ+, (∀ n : ℕ+, T n < m / 20) ∧
             (∀ k : ℕ+, k < m → ∃ n : ℕ+, T n ≥ k / 20)) :=
sorry

end NUMINAMATH_CALUDE_sequence_formula_and_sum_bound_l1671_167145


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1671_167113

theorem sqrt_equation_solution (x y : ℝ) :
  Real.sqrt (x^2 + y^2 - 1) = x + y - 1 ↔ (x = 1 ∧ y ≥ 0) ∨ (y = 1 ∧ x ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1671_167113


namespace NUMINAMATH_CALUDE_amount_added_to_doubled_number_l1671_167166

theorem amount_added_to_doubled_number (original : ℝ) (total : ℝ) (h1 : original = 6.0) (h2 : 2 * original + (total - 2 * original) = 17) : 
  total - 2 * original = 5.0 := by
  sorry

end NUMINAMATH_CALUDE_amount_added_to_doubled_number_l1671_167166


namespace NUMINAMATH_CALUDE_quadratic_roots_and_m_value_l1671_167174

-- Define the quadratic equation
def quadratic (m : ℝ) (x : ℝ) : ℝ := x^2 + (2-m)*x + (1-m)

-- Theorem statement
theorem quadratic_roots_and_m_value (m : ℝ) :
  (∀ x : ℝ, ∃ y z : ℝ, y ≠ z ∧ quadratic m y = 0 ∧ quadratic m z = 0) ∧
  (m < 0 → (∃ y z : ℝ, y ≠ z ∧ quadratic m y = 0 ∧ quadratic m z = 0 ∧ |y - z| = 3) → m = -3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_and_m_value_l1671_167174


namespace NUMINAMATH_CALUDE_water_volume_ratio_in_cone_l1671_167168

/-- Theorem: Volume ratio of water in a cone filled to 2/3 of its height -/
theorem water_volume_ratio_in_cone (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let water_height : ℝ := 2 / 3 * h
  let water_radius : ℝ := 2 / 3 * r
  let cone_volume : ℝ := (1 / 3) * π * r^2 * h
  let water_volume : ℝ := (1 / 3) * π * water_radius^2 * water_height
  water_volume / cone_volume = 8 / 27 :=
by sorry

end NUMINAMATH_CALUDE_water_volume_ratio_in_cone_l1671_167168


namespace NUMINAMATH_CALUDE_chocolate_mixture_percentage_l1671_167162

theorem chocolate_mixture_percentage (initial_amount : ℝ) (initial_percentage : ℝ) 
  (added_amount : ℝ) (desired_percentage : ℝ) : 
  initial_amount = 220 →
  initial_percentage = 0.5 →
  added_amount = 220 →
  desired_percentage = 0.75 →
  (initial_amount * initial_percentage + added_amount) / (initial_amount + added_amount) = desired_percentage :=
by sorry

end NUMINAMATH_CALUDE_chocolate_mixture_percentage_l1671_167162


namespace NUMINAMATH_CALUDE_factor_expression_l1671_167173

theorem factor_expression (x : ℝ) : 75 * x^2 + 50 * x = 25 * x * (3 * x + 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1671_167173


namespace NUMINAMATH_CALUDE_three_conclusions_correct_l1671_167150

-- Define the "heap" for natural numbers
def heap (r : Nat) : Set Nat := {n : Nat | ∃ k : Nat, n = 3 * k + r}

-- Define the four conclusions
def conclusion1 : Prop := 2011 ∈ heap 1
def conclusion2 : Prop := ∀ a b : Nat, a ∈ heap 1 → b ∈ heap 2 → (a + b) ∈ heap 0
def conclusion3 : Prop := (heap 0) ∪ (heap 1) ∪ (heap 2) = Set.univ
def conclusion4 : Prop := ∀ r : Fin 3, ∀ a b : Nat, a ∈ heap r → b ∈ heap r → (a - b) ∉ heap r

-- Theorem stating that exactly 3 out of 4 conclusions are correct
theorem three_conclusions_correct :
  (conclusion1 ∧ conclusion2 ∧ conclusion3 ∧ ¬conclusion4) ∨
  (conclusion1 ∧ conclusion2 ∧ ¬conclusion3 ∧ conclusion4) ∨
  (conclusion1 ∧ ¬conclusion2 ∧ conclusion3 ∧ conclusion4) ∨
  (¬conclusion1 ∧ conclusion2 ∧ conclusion3 ∧ conclusion4) :=
sorry

end NUMINAMATH_CALUDE_three_conclusions_correct_l1671_167150


namespace NUMINAMATH_CALUDE_single_windows_upstairs_correct_number_of_single_windows_l1671_167132

theorem single_windows_upstairs 
  (double_windows : ℕ) 
  (panels_per_double : ℕ) 
  (panels_per_single : ℕ) 
  (total_panels : ℕ) : ℕ :=
  let downstairs_panels := double_windows * panels_per_double
  let upstairs_panels := total_panels - downstairs_panels
  upstairs_panels / panels_per_single

theorem correct_number_of_single_windows :
  single_windows_upstairs 6 4 4 80 = 14 := by
  sorry

end NUMINAMATH_CALUDE_single_windows_upstairs_correct_number_of_single_windows_l1671_167132


namespace NUMINAMATH_CALUDE_latin_speakers_l1671_167140

/-- In a group of people, given the total number, the number of French speakers,
    the number of people speaking neither Latin nor French, and the number of people
    speaking both Latin and French, we can determine the number of Latin speakers. -/
theorem latin_speakers (total : ℕ) (french : ℕ) (neither : ℕ) (both : ℕ) :
  total = 25 →
  french = 15 →
  neither = 6 →
  both = 9 →
  ∃ latin : ℕ, latin = 13 ∧ latin + french - both = total - neither :=
by sorry

end NUMINAMATH_CALUDE_latin_speakers_l1671_167140


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_angle_l1671_167135

/-- The angle between asymptotes of a hyperbola -/
theorem hyperbola_asymptote_angle (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := (2 * Real.sqrt 3) / 3
  let hyperbola := fun (x y : ℝ) ↦ x^2 / a^2 - y^2 / b^2 = 1
  hyperbola x y ∧ e = Real.sqrt (1 + b^2 / a^2) →
  ∃ θ : ℝ, θ = π / 3 ∧ θ = 2 * Real.arctan (b / a) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_angle_l1671_167135


namespace NUMINAMATH_CALUDE_prob_king_ace_ten_l1671_167133

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Number of Kings in a standard deck -/
def NumKings : ℕ := 4

/-- Number of Aces in a standard deck -/
def NumAces : ℕ := 4

/-- Number of Tens in a standard deck -/
def NumTens : ℕ := 4

/-- Probability of drawing a King, then an Ace, then a 10 from a standard deck -/
theorem prob_king_ace_ten (deck : ℕ) (kings aces tens : ℕ) : 
  deck = StandardDeck → kings = NumKings → aces = NumAces → tens = NumTens →
  (kings : ℚ) / deck * aces / (deck - 1) * tens / (deck - 2) = 8 / 16575 := by
sorry

end NUMINAMATH_CALUDE_prob_king_ace_ten_l1671_167133


namespace NUMINAMATH_CALUDE_solution_set_is_circle_minus_point_l1671_167171

theorem solution_set_is_circle_minus_point :
  ∀ (x y a : ℝ),
  (a * x + y = 2 * a + 3 ∧ x - a * y = a + 4) ↔
  ((x - 3)^2 + (y - 1)^2 = 5 ∧ (x, y) ≠ (2, -1)) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_is_circle_minus_point_l1671_167171


namespace NUMINAMATH_CALUDE_problem_statement_l1671_167106

theorem problem_statement (a b x y : ℝ) 
  (h1 : a + b = 2) 
  (h2 : x + y = 2) 
  (h3 : a * x + b * y = 5) : 
  (a^2 + b^2) * x * y + a * b * (x^2 + y^2) = -5 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1671_167106


namespace NUMINAMATH_CALUDE_add_minutes_theorem_l1671_167148

/-- Represents a date and time -/
structure DateTime where
  year : ℕ
  month : ℕ
  day : ℕ
  hour : ℕ
  minute : ℕ

/-- Adds minutes to a DateTime -/
def addMinutes (dt : DateTime) (minutes : ℕ) : DateTime :=
  sorry

/-- The starting DateTime -/
def startTime : DateTime :=
  { year := 2020, month := 2, day := 1, hour := 18, minute := 0 }

/-- The number of minutes to add -/
def minutesToAdd : ℕ := 3457

/-- The expected end DateTime -/
def endTime : DateTime :=
  { year := 2020, month := 2, day := 4, hour := 3, minute := 37 }

/-- Theorem stating that adding minutesToAdd to startTime results in endTime -/
theorem add_minutes_theorem : addMinutes startTime minutesToAdd = endTime :=
  sorry

end NUMINAMATH_CALUDE_add_minutes_theorem_l1671_167148


namespace NUMINAMATH_CALUDE_prob_four_suits_in_five_draws_l1671_167163

/-- Represents a standard deck of 52 cards -/
def StandardDeck : Type := Unit

/-- Represents the number of suits in a standard deck -/
def numSuits : ℕ := 4

/-- Represents the number of cards drawn -/
def numDraws : ℕ := 5

/-- Represents the probability of drawing a card from a particular suit -/
def probSuitDraw : ℚ := 1 / 4

/-- The probability of drawing 4 cards representing each of the 4 suits 
    when drawing 5 cards with replacement from a standard 52-card deck -/
theorem prob_four_suits_in_five_draws (deck : StandardDeck) : 
  (3 : ℚ) / 32 = probSuitDraw^3 * (1 - probSuitDraw) * (2 - probSuitDraw) * (3 - probSuitDraw) / 6 :=
sorry

end NUMINAMATH_CALUDE_prob_four_suits_in_five_draws_l1671_167163
