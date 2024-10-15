import Mathlib

namespace NUMINAMATH_CALUDE_problem_solution_l3783_378378

theorem problem_solution (a b : ℤ) (h1 : a > b) (h2 : b > 0) (h3 : a + b + a * b = 119) : a = 59 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3783_378378


namespace NUMINAMATH_CALUDE_half_abs_diff_squares_21_19_l3783_378387

theorem half_abs_diff_squares_21_19 : (1 / 2 : ℝ) * |21^2 - 19^2| = 40 := by sorry

end NUMINAMATH_CALUDE_half_abs_diff_squares_21_19_l3783_378387


namespace NUMINAMATH_CALUDE_scooter_only_owners_l3783_378316

theorem scooter_only_owners (total : ℕ) (scooter : ℕ) (bike : ℕ) 
  (h1 : total = 450) 
  (h2 : scooter = 380) 
  (h3 : bike = 120) : 
  scooter - (scooter + bike - total) = 330 := by
  sorry

end NUMINAMATH_CALUDE_scooter_only_owners_l3783_378316


namespace NUMINAMATH_CALUDE_small_ring_rotation_l3783_378337

theorem small_ring_rotation (r₁ r₂ : ℝ) (h₁ : r₁ = 1) (h₂ : r₂ = 4) :
  (2 * r₂ * Real.pi - 2 * r₁ * Real.pi) / (2 * r₁ * Real.pi) = 3 := by
  sorry

end NUMINAMATH_CALUDE_small_ring_rotation_l3783_378337


namespace NUMINAMATH_CALUDE_sin_alpha_minus_pi_third_l3783_378303

theorem sin_alpha_minus_pi_third (α : ℝ) (h : Real.cos (α + π/6) = -1/3) : 
  Real.sin (α - π/3) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_minus_pi_third_l3783_378303


namespace NUMINAMATH_CALUDE_sin_negative_ten_pi_thirds_equals_sqrt_three_halves_l3783_378383

theorem sin_negative_ten_pi_thirds_equals_sqrt_three_halves :
  Real.sin (-10 * Real.pi / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_ten_pi_thirds_equals_sqrt_three_halves_l3783_378383


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l3783_378345

theorem quadratic_roots_property :
  ∀ x₁ x₂ : ℝ,
  x₁^2 - 3*x₁ - 4 = 0 →
  x₂^2 - 3*x₂ - 4 = 0 →
  x₁ ≠ x₂ →
  x₁*x₂ - x₁ - x₂ = -7 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l3783_378345


namespace NUMINAMATH_CALUDE_equation_system_solution_l3783_378350

theorem equation_system_solution (x y z : ℝ) 
  (eq1 : 3 * x - 4 * y - 2 * z = 0)
  (eq2 : x + 2 * y - 7 * z = 0)
  (z_nonzero : z ≠ 0) :
  (x^2 - 2*x*y) / (y^2 + 4*z^2) = -0.252 := by
  sorry

end NUMINAMATH_CALUDE_equation_system_solution_l3783_378350


namespace NUMINAMATH_CALUDE_perfect_cube_divisibility_l3783_378328

theorem perfect_cube_divisibility (a b : ℕ) (h1 : 0 < b) (h2 : b < a) 
  (h3 : (a^3 + b^3 + a*b) % (a*b*(a-b)) = 0) : 
  ∃ (k : ℕ), a * b = k^3 :=
sorry

end NUMINAMATH_CALUDE_perfect_cube_divisibility_l3783_378328


namespace NUMINAMATH_CALUDE_uncle_fyodor_cannot_always_win_l3783_378362

/-- Represents a sandwich with sausage and cheese -/
structure Sandwich :=
  (hasSausage : Bool)

/-- Represents the state of the game -/
structure GameState :=
  (sandwiches : List Sandwich)
  (turn : Nat)

/-- Uncle Fyodor's move: eat one sandwich from either end -/
def uncleFyodorMove (state : GameState) : GameState :=
  sorry

/-- Matroskin's move: remove sausage from one sandwich or do nothing -/
def matroskinMove (state : GameState) : GameState :=
  sorry

/-- Play the game until all sandwiches are eaten -/
def playGame (initialState : GameState) : GameState :=
  sorry

/-- Check if Uncle Fyodor wins (last sandwich eaten contains sausage) -/
def uncleFyodorWins (finalState : GameState) : Bool :=
  sorry

/-- Theorem: There exists a natural number N for which Uncle Fyodor cannot guarantee a win -/
theorem uncle_fyodor_cannot_always_win :
  ∃ N : Nat, ∀ uncleFyodorStrategy : GameState → GameState,
    ∃ matroskinStrategy : GameState → GameState,
      let initialState := GameState.mk (List.replicate N (Sandwich.mk true)) 0
      ¬(uncleFyodorWins (playGame initialState)) :=
by
  sorry

end NUMINAMATH_CALUDE_uncle_fyodor_cannot_always_win_l3783_378362


namespace NUMINAMATH_CALUDE_equation_solution_l3783_378397

theorem equation_solution :
  let f : ℝ → ℝ := λ x => (x^2 + 1)^2 - 4*(x^2 + 1) - 12
  ∀ x : ℝ, f x = 0 ↔ x = Real.sqrt 5 ∨ x = -Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3783_378397


namespace NUMINAMATH_CALUDE_exactly_one_black_and_exactly_two_red_mutually_exclusive_but_not_complementary_l3783_378363

/-- Represents the color of a ball -/
inductive Color
| Red
| Black

/-- Represents the outcome of drawing two balls -/
structure Outcome :=
  (first second : Color)

/-- The sample space of all possible outcomes when drawing two balls -/
def sampleSpace : Finset Outcome :=
  sorry

/-- The event of having exactly one black ball -/
def exactlyOneBlack (outcome : Outcome) : Prop :=
  (outcome.first = Color.Black ∧ outcome.second = Color.Red) ∨
  (outcome.first = Color.Red ∧ outcome.second = Color.Black)

/-- The event of having exactly two red balls -/
def exactlyTwoRed (outcome : Outcome) : Prop :=
  outcome.first = Color.Red ∧ outcome.second = Color.Red

/-- Two events are mutually exclusive if they cannot occur simultaneously -/
def mutuallyExclusive (e1 e2 : Outcome → Prop) : Prop :=
  ∀ outcome, ¬(e1 outcome ∧ e2 outcome)

/-- Two events are complementary if one of them always occurs -/
def complementary (e1 e2 : Outcome → Prop) : Prop :=
  ∀ outcome, e1 outcome ∨ e2 outcome

theorem exactly_one_black_and_exactly_two_red_mutually_exclusive_but_not_complementary :
  mutuallyExclusive exactlyOneBlack exactlyTwoRed ∧
  ¬complementary exactlyOneBlack exactlyTwoRed :=
sorry

end NUMINAMATH_CALUDE_exactly_one_black_and_exactly_two_red_mutually_exclusive_but_not_complementary_l3783_378363


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l3783_378367

/-- Proves that the repeating decimal 7.832̅ is equal to the fraction 70/9 -/
theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), x = 7 + 832 / 999 ∧ x = 70 / 9 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l3783_378367


namespace NUMINAMATH_CALUDE_limit_of_sequence_l3783_378310

def a (n : ℕ) : ℚ := (2 * n - 5 : ℚ) / (3 * n + 1)

theorem limit_of_sequence : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - 2/3| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_of_sequence_l3783_378310


namespace NUMINAMATH_CALUDE_vessel_base_length_vessel_problem_solution_l3783_378344

/-- Given a cube immersed in a rectangular vessel, calculates the length of the vessel's base. -/
theorem vessel_base_length (cube_edge : ℝ) (vessel_width : ℝ) (water_rise : ℝ) : ℝ :=
  let cube_volume := cube_edge^3
  let vessel_length := cube_volume / (vessel_width * water_rise)
  vessel_length

/-- Proves that for a 15 cm cube in a vessel of width 15 cm causing 11.25 cm water rise, 
    the vessel's base length is 20 cm. -/
theorem vessel_problem_solution : 
  vessel_base_length 15 15 11.25 = 20 := by
  sorry

end NUMINAMATH_CALUDE_vessel_base_length_vessel_problem_solution_l3783_378344


namespace NUMINAMATH_CALUDE_workshop_workers_count_l3783_378335

/-- Proves that the total number of workers in a workshop is 49 given specific salary conditions. -/
theorem workshop_workers_count :
  let average_salary : ℕ := 8000
  let technician_salary : ℕ := 20000
  let other_salary : ℕ := 6000
  let technician_count : ℕ := 7
  ∃ (total_workers : ℕ) (other_workers : ℕ),
    total_workers = technician_count + other_workers ∧
    total_workers * average_salary = technician_count * technician_salary + other_workers * other_salary ∧
    total_workers = 49 := by
  sorry

#check workshop_workers_count

end NUMINAMATH_CALUDE_workshop_workers_count_l3783_378335


namespace NUMINAMATH_CALUDE_square_and_reciprocal_square_l3783_378352

theorem square_and_reciprocal_square (n : ℝ) (h : n + 1/n = 10) : n^2 + 1/n^2 + 6 = 104 := by
  sorry

end NUMINAMATH_CALUDE_square_and_reciprocal_square_l3783_378352


namespace NUMINAMATH_CALUDE_function_passes_through_point_l3783_378365

/-- Given a > 0 and a ≠ 1, the function f(x) = a^(x-1) + 3 passes through the point (1, 4) -/
theorem function_passes_through_point (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 1) + 3
  f 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l3783_378365


namespace NUMINAMATH_CALUDE_gift_wrap_sales_l3783_378364

theorem gift_wrap_sales (total_goal : ℕ) (grandmother_sales uncle_sales neighbor_sales : ℕ) : 
  total_goal = 45 ∧ 
  grandmother_sales = 1 ∧ 
  uncle_sales = 10 ∧ 
  neighbor_sales = 6 → 
  total_goal - (grandmother_sales + uncle_sales + neighbor_sales) = 28 := by
  sorry

end NUMINAMATH_CALUDE_gift_wrap_sales_l3783_378364


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3783_378327

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (∀ n : ℕ, a (n + 1) < a n) →
  a 2 * a 8 = 6 →
  a 4 + a 6 = 5 →
  a 3 / a 7 = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3783_378327


namespace NUMINAMATH_CALUDE_triangle_third_side_length_l3783_378329

theorem triangle_third_side_length (a b : ℝ) (ha : a = 6.31) (hb : b = 0.82) :
  ∃! c : ℕ, (c : ℝ) > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b ∧ c = 6 :=
by sorry

end NUMINAMATH_CALUDE_triangle_third_side_length_l3783_378329


namespace NUMINAMATH_CALUDE_simplify_expression_l3783_378340

theorem simplify_expression (a b c : ℝ) (h : a + b + c = 0) :
  a * (1 / b + 1 / c) + b * (1 / c + 1 / a) + c * (1 / a + 1 / b) + 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3783_378340


namespace NUMINAMATH_CALUDE_cycle_loss_percentage_l3783_378375

/-- Calculates the percentage of loss given the cost price and selling price -/
def percentage_loss (cost_price selling_price : ℚ) : ℚ :=
  ((cost_price - selling_price) / cost_price) * 100

/-- Theorem: The percentage of loss for a cycle with cost price 1200 and selling price 1020 is 15% -/
theorem cycle_loss_percentage :
  let cost_price : ℚ := 1200
  let selling_price : ℚ := 1020
  percentage_loss cost_price selling_price = 15 := by
  sorry

#eval percentage_loss 1200 1020

end NUMINAMATH_CALUDE_cycle_loss_percentage_l3783_378375


namespace NUMINAMATH_CALUDE_victors_weekly_earnings_l3783_378360

/-- Calculates the total earnings for a week given an hourly wage and hours worked each day -/
def weeklyEarnings (hourlyWage : ℕ) (hoursWorked : List ℕ) : ℕ :=
  hourlyWage * (hoursWorked.sum)

/-- Theorem: Victor's weekly earnings -/
theorem victors_weekly_earnings :
  let hourlyWage : ℕ := 12
  let hoursWorked : List ℕ := [5, 6, 7, 4, 8]
  weeklyEarnings hourlyWage hoursWorked = 360 := by
  sorry

end NUMINAMATH_CALUDE_victors_weekly_earnings_l3783_378360


namespace NUMINAMATH_CALUDE_odd_prime_fifth_power_difference_l3783_378356

theorem odd_prime_fifth_power_difference (p : ℕ) (hp : Nat.Prime p) (hp_odd : Odd p) 
  (hx : ∃ (x y : ℤ), (x : ℝ)^5 - (y : ℝ)^5 = p) :
  ∃ (v : ℤ), Odd v ∧ Real.sqrt ((4 * p + 1 : ℝ) / 5) = ((v^2 : ℝ) + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_odd_prime_fifth_power_difference_l3783_378356


namespace NUMINAMATH_CALUDE_rectangle_ratio_l3783_378347

/-- Given two rectangles A and B with sides (a, b) and (c, d) respectively,
    if the ratio of their areas is 0.16 and a/c = b/d,
    then a/c = b/d = 0.4 -/
theorem rectangle_ratio (a b c d : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0)
  (h5 : (a * b) / (c * d) = 0.16) (h6 : a / c = b / d) :
  a / c = 0.4 ∧ b / d = 0.4 := by
sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l3783_378347


namespace NUMINAMATH_CALUDE_m_range_l3783_378319

def P (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ < 0 ∧ x₂ < 0 ∧ x₁^2 + m*x₁ + 1 = 0 ∧ x₂^2 + m*x₂ + 1 = 0

def Q (m : ℝ) : Prop :=
  ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

theorem m_range (m : ℝ) :
  (P m ∨ Q m) ∧ ¬(P m ∧ Q m) → m ∈ Set.Ioo 1 2 ∪ Set.Ici 3 :=
by sorry

end NUMINAMATH_CALUDE_m_range_l3783_378319


namespace NUMINAMATH_CALUDE_train_distance_time_relation_l3783_378379

/-- The distance-time relationship for a train journey -/
theorem train_distance_time_relation 
  (initial_distance : ℝ) 
  (speed : ℝ) 
  (t : ℝ) 
  (h1 : initial_distance = 3) 
  (h2 : speed = 120) 
  (h3 : t ≥ 0) : 
  ∃ s : ℝ, s = initial_distance + speed * t :=
sorry

end NUMINAMATH_CALUDE_train_distance_time_relation_l3783_378379


namespace NUMINAMATH_CALUDE_largest_integer_solution_l3783_378353

theorem largest_integer_solution (m : ℤ) : (2 * m + 7 ≤ 3) → m ≤ -2 ∧ ∀ k : ℤ, (2 * k + 7 ≤ 3) → k ≤ m := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_solution_l3783_378353


namespace NUMINAMATH_CALUDE_min_correct_answers_for_score_l3783_378385

/-- Given a math test with the following conditions:
  * There are 16 total questions
  * 6 points are awarded for each correct answer
  * 2 points are deducted for each wrong answer
  * No points are deducted for unanswered questions
  * The student did not answer one question
  * The goal is to score more than 60 points

  This theorem proves that the minimum number of correct answers needed is 12. -/
theorem min_correct_answers_for_score (total_questions : ℕ) (correct_points : ℕ) (wrong_points : ℕ) 
  (unanswered : ℕ) (target_score : ℕ) : 
  total_questions = 16 → 
  correct_points = 6 → 
  wrong_points = 2 → 
  unanswered = 1 → 
  target_score = 60 → 
  ∃ (min_correct : ℕ), 
    (∀ (x : ℕ), x ≥ min_correct → 
      x * correct_points - (total_questions - unanswered - x) * wrong_points > target_score) ∧ 
    (∀ (y : ℕ), y < min_correct → 
      y * correct_points - (total_questions - unanswered - y) * wrong_points ≤ target_score) ∧
    min_correct = 12 := by
  sorry

end NUMINAMATH_CALUDE_min_correct_answers_for_score_l3783_378385


namespace NUMINAMATH_CALUDE_function_equals_square_l3783_378320

-- Define the property that f has the same number of intersections as x^2 with any line
def SameIntersections (f : ℝ → ℝ) : Prop :=
  ∀ (m c : ℝ), (∃ (x : ℝ), f x = m * x + c) ↔ (∃ (x : ℝ), x^2 = m * x + c)

-- State the theorem
theorem function_equals_square (f : ℝ → ℝ) (h : SameIntersections f) : 
  ∀ x : ℝ, f x = x^2 := by
sorry

end NUMINAMATH_CALUDE_function_equals_square_l3783_378320


namespace NUMINAMATH_CALUDE_parabola_directrix_l3783_378390

/-- Given a parabola y^2 = 2px where p > 0 that passes through the point (1, 1/2),
    its directrix has the equation x = -1/16 -/
theorem parabola_directrix (p : ℝ) (h1 : p > 0) :
  (∀ x y : ℝ, y^2 = 2*p*x) →
  ((1 : ℝ)^2 = 2*p*(1/2 : ℝ)^2) →
  (∃ k : ℝ, ∀ x : ℝ, x = k ↔ x = -1/16) := by
  sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3783_378390


namespace NUMINAMATH_CALUDE_geometric_sum_remainder_main_theorem_l3783_378377

theorem geometric_sum_remainder (n : ℕ) (a r : ℤ) (m : ℕ) (h : m > 0) :
  (a * (r^n - 1) / (r - 1)) % m = ((a * (r^n % m - 1)) / (r - 1)) % m :=
sorry

theorem main_theorem :
  (((3^1005 - 1) / 2) : ℤ) % 500 = 121 :=
sorry

end NUMINAMATH_CALUDE_geometric_sum_remainder_main_theorem_l3783_378377


namespace NUMINAMATH_CALUDE_white_square_arc_length_bound_l3783_378321

/-- Represents a circle on a chessboard --/
structure ChessboardCircle where
  center : ℝ × ℝ
  radius : ℝ
  encloses_white_square : Bool

/-- Represents the portion of a circle's circumference passing through white squares --/
def white_square_arc_length (c : ChessboardCircle) : ℝ := sorry

/-- The theorem to be proved --/
theorem white_square_arc_length_bound 
  (c : ChessboardCircle) 
  (h1 : c.radius = 1) 
  (h2 : c.encloses_white_square = true) : 
  white_square_arc_length c ≤ (1/3) * (2 * Real.pi * c.radius) := by
  sorry

end NUMINAMATH_CALUDE_white_square_arc_length_bound_l3783_378321


namespace NUMINAMATH_CALUDE_a4_plus_b4_equals_228_l3783_378342

theorem a4_plus_b4_equals_228 (a b : ℝ) 
  (h1 : (a^2 - b^2)^2 = 100) 
  (h2 : (a^3 * b^3) = 512) : 
  a^4 + b^4 = 228 := by
sorry

end NUMINAMATH_CALUDE_a4_plus_b4_equals_228_l3783_378342


namespace NUMINAMATH_CALUDE_constant_function_proof_l3783_378392

def IsFunctionalRelation (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (2 * x)

theorem constant_function_proof (f : ℝ → ℝ) 
  (h1 : Continuous f) 
  (h2 : IsFunctionalRelation f) : 
  ∀ x : ℝ, f x = f 0 := by
  sorry

end NUMINAMATH_CALUDE_constant_function_proof_l3783_378392


namespace NUMINAMATH_CALUDE_scientific_notation_of_nine_billion_l3783_378384

theorem scientific_notation_of_nine_billion :
  9000000000 = 9 * (10 : ℝ)^9 :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_nine_billion_l3783_378384


namespace NUMINAMATH_CALUDE_science_class_ends_at_350pm_l3783_378341

-- Define the start time and class durations
def school_start_time : Nat := 12 * 60  -- 12:00 pm in minutes
def maths_duration : Nat := 45
def history_duration : Nat := 75  -- 1 hour and 15 minutes
def geography_duration : Nat := 30
def science_duration : Nat := 50
def break_duration : Nat := 10

-- Define a function to calculate the end time of Science class
def science_class_end_time : Nat :=
  school_start_time +
  maths_duration + break_duration +
  history_duration + break_duration +
  geography_duration + break_duration +
  science_duration

-- Convert minutes to hours and minutes
def minutes_to_time (minutes : Nat) : String :=
  let hours := minutes / 60
  let mins := minutes % 60
  s!"{hours}:{mins}"

-- Theorem to prove
theorem science_class_ends_at_350pm :
  minutes_to_time science_class_end_time = "3:50" :=
by sorry

end NUMINAMATH_CALUDE_science_class_ends_at_350pm_l3783_378341


namespace NUMINAMATH_CALUDE_min_fraction_sum_l3783_378380

def digits : Finset ℕ := {1, 2, 3, 4, 5, 6}

theorem min_fraction_sum (W X Y Z : ℕ) 
  (h_distinct : W ≠ X ∧ W ≠ Y ∧ W ≠ Z ∧ X ≠ Y ∧ X ≠ Z ∧ Y ≠ Z)
  (h_in_set : W ∈ digits ∧ X ∈ digits ∧ Y ∈ digits ∧ Z ∈ digits) :
  (W : ℚ) / X + (Y : ℚ) / Z ≥ 17 / 30 :=
sorry

end NUMINAMATH_CALUDE_min_fraction_sum_l3783_378380


namespace NUMINAMATH_CALUDE_number_equation_l3783_378309

theorem number_equation (x : ℝ) : (9 * x) / 3 = 27 ↔ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l3783_378309


namespace NUMINAMATH_CALUDE_greatest_x_cube_less_than_2000_l3783_378361

theorem greatest_x_cube_less_than_2000 :
  ∃ (x : ℕ), x > 0 ∧ ∃ (k : ℕ), x = 5 * k ∧ x^3 < 2000 ∧
  ∀ (y : ℕ), y > 0 → (∃ (m : ℕ), y = 5 * m) → y^3 < 2000 → y ≤ x :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_greatest_x_cube_less_than_2000_l3783_378361


namespace NUMINAMATH_CALUDE_tangent_line_sum_l3783_378307

/-- Given a function f: ℝ → ℝ with a tangent line at x = 1 defined by 2x - y + 1 = 0,
    prove that f(1) + f'(1) = 5 -/
theorem tangent_line_sum (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (h_tangent : ∀ x y, y = f x → (x = 1 → 2*x - y + 1 = 0)) : 
    f 1 + (deriv f) 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_sum_l3783_378307


namespace NUMINAMATH_CALUDE_distance_sum_bounds_l3783_378357

/-- Given points A, B, C in a 2D plane and a point P satisfying x^2 + y^2 ≤ 4,
    the sum of squared distances from P to A, B, and C is between 72 and 88. -/
theorem distance_sum_bounds (x y : ℝ) :
  x^2 + y^2 ≤ 4 →
  72 ≤ ((x + 2)^2 + (y - 2)^2) + ((x + 2)^2 + (y - 6)^2) + ((x - 4)^2 + (y + 2)^2) ∧
  ((x + 2)^2 + (y - 2)^2) + ((x + 2)^2 + (y - 6)^2) + ((x - 4)^2 + (y + 2)^2) ≤ 88 :=
by sorry

end NUMINAMATH_CALUDE_distance_sum_bounds_l3783_378357


namespace NUMINAMATH_CALUDE_defective_units_shipped_l3783_378330

theorem defective_units_shipped (total_units : ℝ) (defective_rate : ℝ) (shipped_rate : ℝ)
  (h1 : defective_rate = 0.04)
  (h2 : shipped_rate = 0.04) :
  (defective_rate * shipped_rate * total_units) / total_units = 0.0016 := by
  sorry

end NUMINAMATH_CALUDE_defective_units_shipped_l3783_378330


namespace NUMINAMATH_CALUDE_rectangle_side_difference_l3783_378355

theorem rectangle_side_difference (p d : ℝ) (h_positive : p > 0 ∧ d > 0) :
  ∃ (x y : ℝ), 
    x > 0 ∧ y > 0 ∧
    x = 2 * y ∧
    2 * (x + y) = p ∧
    x^2 + y^2 = d^2 ∧
    x - y = p / 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_side_difference_l3783_378355


namespace NUMINAMATH_CALUDE_expression_nonnegative_l3783_378346

theorem expression_nonnegative (x : ℝ) : 
  (x - 20*x^2 + 100*x^3) / (16 - 2*x^3) ≥ 0 ↔ x ∈ Set.Ici 0 ∩ Set.Iio 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_nonnegative_l3783_378346


namespace NUMINAMATH_CALUDE_abs_x_minus_log_y_equals_x_plus_2log_y_implies_x_0_y_1_l3783_378304

theorem abs_x_minus_log_y_equals_x_plus_2log_y_implies_x_0_y_1
  (x : ℝ) (y : ℝ) (h : y > 0) :
  |x - Real.log y| = x + 2 * Real.log y → x = 0 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_x_minus_log_y_equals_x_plus_2log_y_implies_x_0_y_1_l3783_378304


namespace NUMINAMATH_CALUDE_exponent_multiplication_l3783_378313

theorem exponent_multiplication (x : ℝ) : x^3 * x^2 = x^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l3783_378313


namespace NUMINAMATH_CALUDE_triangle_area_with_given_conditions_l3783_378351

noncomputable def triangle_area (r : ℝ) (R : ℝ) (A B C : ℝ) : ℝ :=
  4 * r * R * Real.sin A

theorem triangle_area_with_given_conditions (r R A B C : ℝ) 
  (h_inradius : r = 4)
  (h_circumradius : R = 9)
  (h_angle_condition : 2 * Real.cos A = Real.cos B + Real.cos C) :
  triangle_area r R A B C = 8 * Real.sqrt 181 := by
  sorry

#check triangle_area_with_given_conditions

end NUMINAMATH_CALUDE_triangle_area_with_given_conditions_l3783_378351


namespace NUMINAMATH_CALUDE_rectangular_field_with_pond_l3783_378376

theorem rectangular_field_with_pond (w l : ℝ) : 
  l = 2 * w →                 -- length is double the width
  36 = (1/8) * (l * w) →      -- pond area (6^2) is 1/8 of field area
  l = 24 := by               -- length of the field is 24 meters
sorry

end NUMINAMATH_CALUDE_rectangular_field_with_pond_l3783_378376


namespace NUMINAMATH_CALUDE_w_squared_equals_one_fourth_l3783_378338

theorem w_squared_equals_one_fourth (w : ℝ) (h : 13 = 13 * w / (1 - w)) : w^2 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_w_squared_equals_one_fourth_l3783_378338


namespace NUMINAMATH_CALUDE_optimal_price_reduction_l3783_378399

/-- Represents the price reduction and sales model of a shopping mall. -/
structure MallSalesModel where
  initial_cost : ℝ
  initial_price : ℝ
  initial_sales : ℝ
  sales_increase_rate : ℝ

/-- Calculates the daily profit given a price reduction. -/
def daily_profit (model : MallSalesModel) (price_reduction : ℝ) : ℝ :=
  let new_sales := model.initial_sales + model.sales_increase_rate * price_reduction
  let new_profit_per_item := (model.initial_price - model.initial_cost) - price_reduction
  new_sales * new_profit_per_item

/-- Theorem stating that a price reduction of 30 yuan results in a daily profit of 3600 yuan. -/
theorem optimal_price_reduction (model : MallSalesModel) 
  (h1 : model.initial_cost = 220)
  (h2 : model.initial_price = 280)
  (h3 : model.initial_sales = 30)
  (h4 : model.sales_increase_rate = 3) :
  daily_profit model 30 = 3600 := by
  sorry

#check optimal_price_reduction

end NUMINAMATH_CALUDE_optimal_price_reduction_l3783_378399


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l3783_378334

/-- Given a line passing through points (1, 3) and (-3, -1), 
    prove that the sum of its slope and y-intercept is 3. -/
theorem line_slope_intercept_sum : 
  ∀ (m b : ℝ), 
  (3 = m * 1 + b) →  -- Point (1, 3) satisfies the line equation
  (-1 = m * (-3) + b) →  -- Point (-3, -1) satisfies the line equation
  m + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l3783_378334


namespace NUMINAMATH_CALUDE_class_size_calculation_l3783_378370

theorem class_size_calculation (total_average : ℝ) (excluded_average : ℝ) (remaining_average : ℝ) (excluded_count : ℕ) :
  total_average = 72 →
  excluded_average = 40 →
  remaining_average = 92 →
  excluded_count = 5 →
  ∃ (total_count : ℕ),
    (total_count : ℝ) * total_average = 
      (total_count - excluded_count : ℝ) * remaining_average + (excluded_count : ℝ) * excluded_average ∧
    total_count = 13 :=
by sorry

end NUMINAMATH_CALUDE_class_size_calculation_l3783_378370


namespace NUMINAMATH_CALUDE_factors_of_product_l3783_378371

/-- A function that returns the number of factors of a natural number -/
def num_factors (n : ℕ) : ℕ := sorry

/-- A function that returns the number of factors of n^k for a natural number n and exponent k -/
def num_factors_power (n k : ℕ) : ℕ := sorry

theorem factors_of_product (a b c : ℕ) : 
  a ≠ b → b ≠ c → a ≠ c →
  num_factors a = 3 →
  num_factors b = 3 →
  num_factors c = 4 →
  num_factors_power a 3 * num_factors_power b 4 * num_factors_power c 5 = 1008 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_product_l3783_378371


namespace NUMINAMATH_CALUDE_some_number_solution_l3783_378388

theorem some_number_solution :
  ∃ x : ℝ, x * 13.26 + x * 9.43 + x * 77.31 = 470 ∧ x = 4.7 := by
  sorry

end NUMINAMATH_CALUDE_some_number_solution_l3783_378388


namespace NUMINAMATH_CALUDE_smallest_four_digit_mod_4_3_l3783_378318

theorem smallest_four_digit_mod_4_3 :
  ∀ n : ℕ, 1000 ≤ n ∧ n ≡ 3 [ZMOD 4] → 1003 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_mod_4_3_l3783_378318


namespace NUMINAMATH_CALUDE_rectangle_with_hole_area_formula_l3783_378312

/-- The area of a rectangle with a rectangular hole -/
def rectangle_with_hole_area (x : ℝ) : ℝ :=
  let large_length : ℝ := 2 * x + 8
  let large_width : ℝ := x + 6
  let hole_length : ℝ := 3 * x - 4
  let hole_width : ℝ := x - 3
  (large_length * large_width) - (hole_length * hole_width)

/-- Theorem: The area of the rectangle with a hole is equal to -x^2 + 33x + 36 -/
theorem rectangle_with_hole_area_formula (x : ℝ) :
  rectangle_with_hole_area x = -x^2 + 33*x + 36 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_with_hole_area_formula_l3783_378312


namespace NUMINAMATH_CALUDE_equation_solution_l3783_378398

theorem equation_solution (x : ℝ) : 2 * x^2 + 9 = (4 - x)^2 ↔ x = 4 + Real.sqrt 23 ∨ x = 4 - Real.sqrt 23 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3783_378398


namespace NUMINAMATH_CALUDE_amaya_total_marks_l3783_378358

/-- Represents the marks scored in different subjects -/
structure Marks where
  arts : ℕ
  maths : ℕ
  music : ℕ
  social_studies : ℕ

/-- Calculates the total marks across all subjects -/
def total_marks (m : Marks) : ℕ :=
  m.arts + m.maths + m.music + m.social_studies

/-- Theorem stating the total marks Amaya scored given the conditions -/
theorem amaya_total_marks :
  ∀ (m : Marks),
    m.arts - m.maths = 20 →
    m.social_studies > m.music →
    m.music = 70 →
    m.maths = (9 * m.arts) / 10 →
    m.social_studies - m.music = 10 →
    total_marks m = 530 := by
  sorry

#check amaya_total_marks

end NUMINAMATH_CALUDE_amaya_total_marks_l3783_378358


namespace NUMINAMATH_CALUDE_sin_cos_15_product_l3783_378386

theorem sin_cos_15_product : 
  (Real.sin (15 * π / 180) + Real.cos (15 * π / 180)) * 
  (Real.sin (15 * π / 180) - Real.cos (15 * π / 180)) = 
  -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_15_product_l3783_378386


namespace NUMINAMATH_CALUDE_angle_CDB_is_15_l3783_378349

/-- A triangle that shares a side with a rectangle -/
structure TriangleWithRectangle where
  /-- The length of the shared side -/
  side : ℝ
  /-- The triangle is equilateral -/
  equilateral : True
  /-- The adjacent side of the rectangle is perpendicular to the shared side -/
  perpendicular : True
  /-- The adjacent side of the rectangle is twice the length of the shared side -/
  adjacent_side : ℝ := 2 * side

/-- The measure of angle CDB in degrees -/
def angle_CDB (t : TriangleWithRectangle) : ℝ := 15

/-- Theorem: The measure of angle CDB is 15 degrees -/
theorem angle_CDB_is_15 (t : TriangleWithRectangle) : angle_CDB t = 15 := by sorry

end NUMINAMATH_CALUDE_angle_CDB_is_15_l3783_378349


namespace NUMINAMATH_CALUDE_cubic_inequality_l3783_378311

theorem cubic_inequality (x : ℝ) :
  x^3 - 12*x^2 + 47*x - 60 < 0 ↔ 3 < x ∧ x < 5 := by sorry

end NUMINAMATH_CALUDE_cubic_inequality_l3783_378311


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l3783_378374

-- Define the radius of the cylinder
def cylinder_radius : ℝ := 2

-- Define the ratio between major and minor axes
def axis_ratio : ℝ := 1.25

-- Theorem statement
theorem ellipse_major_axis_length :
  let minor_axis : ℝ := 2 * cylinder_radius
  let major_axis : ℝ := minor_axis * axis_ratio
  major_axis = 5 := by sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l3783_378374


namespace NUMINAMATH_CALUDE_james_spent_six_l3783_378381

/-- Calculates the total amount spent given the cost of milk, cost of bananas, and sales tax rate. -/
def total_spent (milk_cost banana_cost tax_rate : ℚ) : ℚ :=
  let subtotal := milk_cost + banana_cost
  let tax := subtotal * tax_rate
  subtotal + tax

/-- Proves that James spent $6 given the costs and tax rate. -/
theorem james_spent_six :
  let milk_cost : ℚ := 3
  let banana_cost : ℚ := 2
  let tax_rate : ℚ := 1/5
  total_spent milk_cost banana_cost tax_rate = 6 := by
  sorry

#eval total_spent 3 2 (1/5)

end NUMINAMATH_CALUDE_james_spent_six_l3783_378381


namespace NUMINAMATH_CALUDE_quadratic_root_value_l3783_378368

theorem quadratic_root_value (m : ℝ) : 
  ((m - 2) * 1^2 + 4 * 1 - m^2 = 0) ∧ (m ≠ 2) → m = -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l3783_378368


namespace NUMINAMATH_CALUDE_condition_p_necessary_not_sufficient_for_q_l3783_378382

theorem condition_p_necessary_not_sufficient_for_q :
  (∀ x : ℝ, |x + 1| ≤ 1 → (x - 1) * (x + 2) ≤ 0) ∧
  (∃ x : ℝ, (x - 1) * (x + 2) ≤ 0 ∧ |x + 1| > 1) := by
  sorry

end NUMINAMATH_CALUDE_condition_p_necessary_not_sufficient_for_q_l3783_378382


namespace NUMINAMATH_CALUDE_homework_time_difference_l3783_378394

/-- Proves that the difference in time taken by Sarah and Samuel to finish their homework is 48 minutes -/
theorem homework_time_difference (samuel_time sarah_time_hours : ℝ) : 
  samuel_time = 30 → 
  sarah_time_hours = 1.3 → 
  sarah_time_hours * 60 - samuel_time = 48 := by
sorry

end NUMINAMATH_CALUDE_homework_time_difference_l3783_378394


namespace NUMINAMATH_CALUDE_puzzle_ratio_is_three_to_one_l3783_378333

/-- Given a total puzzle-solving time, warm-up time, and number of additional puzzles,
    calculates the ratio of time spent on each additional puzzle to the warm-up time. -/
def puzzle_time_ratio (total_time warm_up_time : ℕ) (num_puzzles : ℕ) : ℚ :=
  let remaining_time := total_time - warm_up_time
  let time_per_puzzle := remaining_time / num_puzzles
  (time_per_puzzle : ℚ) / warm_up_time

/-- Proves that for the given conditions, the ratio of time spent on each additional puzzle
    to the warm-up puzzle is 3:1. -/
theorem puzzle_ratio_is_three_to_one :
  puzzle_time_ratio 70 10 2 = 3 / 1 := by
  sorry

end NUMINAMATH_CALUDE_puzzle_ratio_is_three_to_one_l3783_378333


namespace NUMINAMATH_CALUDE_root_product_value_l3783_378331

theorem root_product_value (m n : ℝ) : 
  m^2 - 3*m - 2 = 0 → 
  n^2 - 3*n - 2 = 0 → 
  (7*m^2 - 21*m - 3)*(3*n^2 - 9*n + 5) = 121 := by
sorry

end NUMINAMATH_CALUDE_root_product_value_l3783_378331


namespace NUMINAMATH_CALUDE_product_remainder_l3783_378395

theorem product_remainder (a b c d : ℕ) (ha : a = 1492) (hb : b = 1776) (hc : c = 1812) (hd : d = 1996) :
  (a * b * c * d) % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_l3783_378395


namespace NUMINAMATH_CALUDE_average_difference_due_to_input_error_l3783_378348

theorem average_difference_due_to_input_error :
  ∀ (data_points : ℕ) (incorrect_value : ℝ) (correct_value : ℝ),
    data_points = 30 →
    incorrect_value = 105 →
    correct_value = 15 →
    (incorrect_value - correct_value) / data_points = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_average_difference_due_to_input_error_l3783_378348


namespace NUMINAMATH_CALUDE_value_of_expression_l3783_378354

theorem value_of_expression (x y : ℤ) (hx : x = 12) (hy : y = 18) :
  3 * (x - y) * (x + y) = -540 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l3783_378354


namespace NUMINAMATH_CALUDE_split_bill_proof_l3783_378373

def num_friends : ℕ := 5
def num_hamburgers : ℕ := 5
def price_hamburger : ℚ := 3
def num_fries : ℕ := 4
def price_fries : ℚ := 1.20
def num_soda : ℕ := 5
def price_soda : ℚ := 0.50
def price_spaghetti : ℚ := 2.70

theorem split_bill_proof :
  let total_bill := num_hamburgers * price_hamburger +
                    num_fries * price_fries +
                    num_soda * price_soda +
                    price_spaghetti
  (total_bill / num_friends : ℚ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_split_bill_proof_l3783_378373


namespace NUMINAMATH_CALUDE_equal_pairs_infinity_l3783_378393

def infinite_sequence (a : ℤ → ℝ) : Prop :=
  ∀ n : ℤ, a n = (1/4) * (a (n-1) + a (n+1))

theorem equal_pairs_infinity (a : ℤ → ℝ) :
  infinite_sequence a →
  (∃ i j : ℤ, i ≠ j ∧ a i = a j) →
  ∃ f : ℕ → (ℤ × ℤ), (∀ n : ℕ, (f n).1 ≠ (f n).2 ∧ a (f n).1 = a (f n).2) ∧
                      (∀ m n : ℕ, m ≠ n → f m ≠ f n) :=
by sorry

end NUMINAMATH_CALUDE_equal_pairs_infinity_l3783_378393


namespace NUMINAMATH_CALUDE_nickel_dime_difference_l3783_378308

/-- The value of one dollar in cents -/
def dollar : ℕ := 100

/-- The value of a nickel in cents -/
def nickel : ℕ := 5

/-- The value of a dime in cents -/
def dime : ℕ := 10

/-- The number of coins needed to make one dollar using only coins of a given value -/
def coinsNeeded (coinValue : ℕ) : ℕ := dollar / coinValue

theorem nickel_dime_difference :
  coinsNeeded nickel - coinsNeeded dime = 10 := by sorry

end NUMINAMATH_CALUDE_nickel_dime_difference_l3783_378308


namespace NUMINAMATH_CALUDE_difference_of_squares_l3783_378343

theorem difference_of_squares (x y : ℝ) : (x + 2*y) * (-2*y + x) = x^2 - 4*y^2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3783_378343


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3783_378339

-- Problem 1
theorem problem_1 : (π - 1)^0 - Real.sqrt 8 + |(- 2) * Real.sqrt 2| = 1 := by
  sorry

-- Problem 2
theorem problem_2 : ∀ x : ℝ, 3 * x - 2 > x + 4 ↔ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3783_378339


namespace NUMINAMATH_CALUDE_sum_of_cubes_difference_l3783_378324

theorem sum_of_cubes_difference (d e f : ℕ+) :
  (d + e + f : ℕ)^3 - d^3 - e^3 - f^3 = 300 → d + e + f = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_difference_l3783_378324


namespace NUMINAMATH_CALUDE_quadratic_always_positive_implies_m_greater_than_one_l3783_378302

/-- Theorem: If for all real x, x^2 - 2x + m > 0 is true, then m > 1 -/
theorem quadratic_always_positive_implies_m_greater_than_one (m : ℝ) :
  (∀ x : ℝ, x^2 - 2*x + m > 0) → m > 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_implies_m_greater_than_one_l3783_378302


namespace NUMINAMATH_CALUDE_combined_population_l3783_378322

def wellington_population : ℕ := 900

def port_perry_population : ℕ := 7 * wellington_population

def lazy_harbor_population : ℕ := 2 * wellington_population + 600

def newbridge_population : ℕ := 3 * (port_perry_population - wellington_population)

theorem combined_population :
  port_perry_population + lazy_harbor_population + newbridge_population = 24900 := by
  sorry

end NUMINAMATH_CALUDE_combined_population_l3783_378322


namespace NUMINAMATH_CALUDE_total_gross_profit_calculation_l3783_378315

/-- Represents the sales prices and costs for an item over three months -/
structure ItemData :=
  (sales_prices : Fin 3 → ℕ)
  (costs : Fin 3 → ℕ)
  (gross_profit_percentage : ℕ)

/-- Calculates the gross profit for an item in a given month -/
def gross_profit (item : ItemData) (month : Fin 3) : ℕ :=
  item.sales_prices month - item.costs month

/-- Calculates the total gross profit for an item over three months -/
def total_gross_profit (item : ItemData) : ℕ :=
  (gross_profit item 0) + (gross_profit item 1) + (gross_profit item 2)

/-- The main theorem to prove -/
theorem total_gross_profit_calculation 
  (item_a item_b item_c item_d : ItemData)
  (ha : item_a.sales_prices = ![44, 47, 50])
  (hac : item_a.costs = ![20, 22, 25])
  (hap : item_a.gross_profit_percentage = 120)
  (hb : item_b.sales_prices = ![60, 63, 65])
  (hbc : item_b.costs = ![30, 33, 35])
  (hbp : item_b.gross_profit_percentage = 150)
  (hc : item_c.sales_prices = ![80, 83, 85])
  (hcc : item_c.costs = ![40, 42, 45])
  (hcp : item_c.gross_profit_percentage = 100)
  (hd : item_d.sales_prices = ![100, 103, 105])
  (hdc : item_d.costs = ![50, 52, 55])
  (hdp : item_d.gross_profit_percentage = 130) :
  total_gross_profit item_a + total_gross_profit item_b + 
  total_gross_profit item_c + total_gross_profit item_d = 436 := by
  sorry

end NUMINAMATH_CALUDE_total_gross_profit_calculation_l3783_378315


namespace NUMINAMATH_CALUDE_equation_solution_l3783_378326

theorem equation_solution (x y : ℚ) 
  (h : x^2 - 2*y - Real.sqrt 2*y = 17 - 4*Real.sqrt 2) : 
  2*x + y = 14 ∨ 2*x + y = -6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3783_378326


namespace NUMINAMATH_CALUDE_number_puzzle_l3783_378323

theorem number_puzzle (y : ℝ) (h : y ≠ 0) : y = (1 / y) * (-y) + 5 → y = 4 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l3783_378323


namespace NUMINAMATH_CALUDE_fraction_equality_l3783_378391

theorem fraction_equality (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a - 4*b ≠ 0) (h4 : 4*a - b ≠ 0)
  (h5 : (4*a + b) / (a - 4*b) = 3) : (a + 4*b) / (4*a - b) = 9/53 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3783_378391


namespace NUMINAMATH_CALUDE_xoz_symmetry_of_M_l3783_378325

/-- Defines a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Defines the xoz plane symmetry operation -/
def xozPlaneSymmetry (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := p.z }

/-- Theorem: The symmetric point of M(5, 1, -2) with respect to the xoz plane is (5, -1, -2) -/
theorem xoz_symmetry_of_M :
  let M : Point3D := { x := 5, y := 1, z := -2 }
  xozPlaneSymmetry M = { x := 5, y := -1, z := -2 } := by
  sorry

end NUMINAMATH_CALUDE_xoz_symmetry_of_M_l3783_378325


namespace NUMINAMATH_CALUDE_optimal_renovation_solution_l3783_378396

/-- Represents a renovation team -/
structure Team where
  dailyRate : ℕ
  daysAlone : ℕ

/-- The renovation scenario -/
structure RenovationScenario where
  teamA : Team
  teamB : Team
  jointDays : ℕ
  jointCost : ℕ
  mixedDaysA : ℕ
  mixedDaysB : ℕ
  mixedCost : ℕ

/-- Theorem stating the optimal solution for the renovation scenario -/
theorem optimal_renovation_solution (scenario : RenovationScenario) 
  (h1 : scenario.jointDays * (scenario.teamA.dailyRate + scenario.teamB.dailyRate) = scenario.jointCost)
  (h2 : scenario.mixedDaysA * scenario.teamA.dailyRate + scenario.mixedDaysB * scenario.teamB.dailyRate = scenario.mixedCost)
  (h3 : scenario.teamA.daysAlone = 12)
  (h4 : scenario.teamB.daysAlone = 24)
  (h5 : scenario.jointDays = 8)
  (h6 : scenario.jointCost = 3520)
  (h7 : scenario.mixedDaysA = 6)
  (h8 : scenario.mixedDaysB = 12)
  (h9 : scenario.mixedCost = 3480) :
  scenario.teamA.dailyRate = 300 ∧ 
  scenario.teamB.dailyRate = 140 ∧ 
  scenario.teamB.daysAlone * scenario.teamB.dailyRate < scenario.teamA.daysAlone * scenario.teamA.dailyRate :=
by sorry

end NUMINAMATH_CALUDE_optimal_renovation_solution_l3783_378396


namespace NUMINAMATH_CALUDE_union_M_N_l3783_378306

def M : Set ℕ := {0, 1, 2}

def N : Set ℕ := {x | ∃ a ∈ M, x = 2 * a}

theorem union_M_N : M ∪ N = {0, 1, 2, 4} := by sorry

end NUMINAMATH_CALUDE_union_M_N_l3783_378306


namespace NUMINAMATH_CALUDE_stating_max_bulbs_on_theorem_l3783_378359

/-- Represents the maximum number of bulbs that can be turned on in an n × n grid -/
def maxBulbsOn (n : ℕ) : ℕ :=
  if n % 2 = 0 then
    n^2 / 2
  else
    (n^2 - 1) / 2

/-- 
Theorem stating the maximum number of bulbs that can be turned on in an n × n grid,
given the constraints of the problem.
-/
theorem max_bulbs_on_theorem (n : ℕ) :
  ∀ (pressed : Finset (ℕ × ℕ)),
    (∀ (i j : ℕ), (i, j) ∈ pressed → i < n ∧ j < n) →
    (∀ (i j k l : ℕ), (i, j) ∈ pressed → (k, l) ∈ pressed → (i = k ∨ j = l) → i = k ∧ j = l) →
    (∃ (final_state : Finset (ℕ × ℕ)),
      (∀ (i j : ℕ), (i, j) ∈ final_state → i < n ∧ j < n) ∧
      final_state.card ≤ maxBulbsOn n) :=
by
  sorry

#check max_bulbs_on_theorem

end NUMINAMATH_CALUDE_stating_max_bulbs_on_theorem_l3783_378359


namespace NUMINAMATH_CALUDE_carter_performs_30_nights_l3783_378317

/-- The number of nights Carter performs, given his drum stick usage pattern --/
def carter_performance_nights (sticks_per_show : ℕ) (sticks_tossed : ℕ) (total_sticks : ℕ) : ℕ :=
  total_sticks / (sticks_per_show + sticks_tossed)

/-- Theorem stating that Carter performs for 30 nights under the given conditions --/
theorem carter_performs_30_nights :
  carter_performance_nights 5 6 330 = 30 := by
  sorry

end NUMINAMATH_CALUDE_carter_performs_30_nights_l3783_378317


namespace NUMINAMATH_CALUDE_grid_rectangles_l3783_378372

theorem grid_rectangles (h : ℕ) (v : ℕ) (h_eq : h = 5) (v_eq : v = 6) :
  (h.choose 2) * (v.choose 2) = 150 := by
  sorry

end NUMINAMATH_CALUDE_grid_rectangles_l3783_378372


namespace NUMINAMATH_CALUDE_banana_orange_equivalence_l3783_378314

/-- The cost relationships between fruits at Minnie's Orchard -/
structure FruitCosts where
  banana_pear : ℚ  -- ratio of bananas to pears
  pear_apple : ℚ   -- ratio of pears to apples
  apple_orange : ℚ -- ratio of apples to oranges

/-- The number of oranges equivalent in cost to a given number of bananas -/
def bananas_to_oranges (costs : FruitCosts) (num_bananas : ℚ) : ℚ :=
  num_bananas * costs.banana_pear * costs.pear_apple * costs.apple_orange

/-- Theorem stating that 80 bananas are equivalent in cost to 18 oranges -/
theorem banana_orange_equivalence (costs : FruitCosts) 
  (h1 : costs.banana_pear = 4/5)
  (h2 : costs.pear_apple = 3/8)
  (h3 : costs.apple_orange = 9/12) :
  bananas_to_oranges costs 80 = 18 := by
  sorry

#eval bananas_to_oranges ⟨4/5, 3/8, 9/12⟩ 80

end NUMINAMATH_CALUDE_banana_orange_equivalence_l3783_378314


namespace NUMINAMATH_CALUDE_city_pairing_equality_l3783_378305

/-- The number of ways to form r pairs in City A -/
def A (n r : ℕ) : ℕ := sorry

/-- The number of ways to form r pairs in City B -/
def B (n r : ℕ) : ℕ := sorry

/-- Girls in City B know a specific number of boys -/
def girls_know_boys (i : ℕ) : ℕ := 2 * i - 1

theorem city_pairing_equality (n : ℕ) (hn : n ≥ 1) :
  ∀ r : ℕ, 1 ≤ r ∧ r ≤ n → A n r = B n r := by
  sorry

end NUMINAMATH_CALUDE_city_pairing_equality_l3783_378305


namespace NUMINAMATH_CALUDE_proposition_implication_l3783_378366

theorem proposition_implication (p q : Prop) 
  (h1 : ¬p) 
  (h2 : p ∨ q) : 
  q := by sorry

end NUMINAMATH_CALUDE_proposition_implication_l3783_378366


namespace NUMINAMATH_CALUDE_locus_of_intersection_point_l3783_378369

/-- The locus of the intersection point of two rotating lines in a triangle --/
theorem locus_of_intersection_point (d e : ℝ) (h1 : d ≠ 0) (h2 : e ≠ 0) :
  ∃ (f : ℝ → ℝ × ℝ),
    (∀ t, ∃ (m : ℝ),
      (f t).1 = -2 * e / m ∧
      (f t).2 = -m * d) ∧
    (∀ x y, (x, y) ∈ Set.range f ↔ x * y = d * e) :=
sorry

end NUMINAMATH_CALUDE_locus_of_intersection_point_l3783_378369


namespace NUMINAMATH_CALUDE_smallest_n_with_constant_term_l3783_378332

theorem smallest_n_with_constant_term :
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (k : ℕ), k > 0 ∧ k < n → ¬∃ (r : ℕ), 2*k = 5*r) ∧
  (∃ (r : ℕ), 2*n = 5*r) ∧
  n = 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_with_constant_term_l3783_378332


namespace NUMINAMATH_CALUDE_pants_bought_with_tshirts_l3783_378336

/-- Given the price relationships of pants and t-shirts, prove that 1 pant was bought with 6 t-shirts -/
theorem pants_bought_with_tshirts (x : ℚ) :
  (∃ (p t : ℚ), p > 0 ∧ t > 0 ∧ 
    x * p + 6 * t = 750 ∧
    p + 12 * t = 750 ∧
    8 * t = 400) →
  x = 1 := by
sorry

end NUMINAMATH_CALUDE_pants_bought_with_tshirts_l3783_378336


namespace NUMINAMATH_CALUDE_fran_required_speed_l3783_378301

/-- Represents a bike ride with total time, break time, and average speed -/
structure BikeRide where
  totalTime : ℝ
  breakTime : ℝ
  avgSpeed : ℝ

/-- Calculates the distance traveled given a BikeRide -/
def distanceTraveled (ride : BikeRide) : ℝ :=
  ride.avgSpeed * (ride.totalTime - ride.breakTime)

theorem fran_required_speed (joann fran : BikeRide)
    (h1 : joann.totalTime = 4)
    (h2 : joann.breakTime = 1)
    (h3 : joann.avgSpeed = 10)
    (h4 : fran.totalTime = 3)
    (h5 : fran.breakTime = 0.5)
    (h6 : distanceTraveled joann = distanceTraveled fran) :
    fran.avgSpeed = 12 := by
  sorry

#check fran_required_speed

end NUMINAMATH_CALUDE_fran_required_speed_l3783_378301


namespace NUMINAMATH_CALUDE_pet_store_combinations_l3783_378389

def num_puppies : ℕ := 12
def num_kittens : ℕ := 10
def num_hamsters : ℕ := 5
def num_birds : ℕ := 3
def num_people : ℕ := 4

def ways_to_choose_pets : ℕ := num_puppies * num_kittens * num_hamsters * num_birds

def permutations_of_choices : ℕ := Nat.factorial num_people

theorem pet_store_combinations : 
  ways_to_choose_pets * permutations_of_choices = 43200 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_combinations_l3783_378389


namespace NUMINAMATH_CALUDE_fourth_section_size_l3783_378300

/-- The number of students in the fourth section of a chemistry class -/
def fourth_section_students : ℕ :=
  -- We'll define this later in the theorem
  42

/-- Represents the data for a chemistry class section -/
structure Section where
  students : ℕ
  mean_marks : ℚ

/-- Calculates the total marks for a section -/
def total_marks (s : Section) : ℚ :=
  s.students * s.mean_marks

/-- Represents the data for all sections of the chemistry class -/
structure ChemistryClass where
  section1 : Section
  section2 : Section
  section3 : Section
  section4 : Section
  overall_average : ℚ

theorem fourth_section_size (c : ChemistryClass) :
  c.section1.students = 65 →
  c.section2.students = 35 →
  c.section3.students = 45 →
  c.section1.mean_marks = 50 →
  c.section2.mean_marks = 60 →
  c.section3.mean_marks = 55 →
  c.section4.mean_marks = 45 →
  c.overall_average = 51.95 →
  c.section4.students = fourth_section_students :=
by
  sorry

#eval fourth_section_students

end NUMINAMATH_CALUDE_fourth_section_size_l3783_378300
