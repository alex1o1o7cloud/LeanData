import Mathlib

namespace fraction_simplification_l1773_177324

theorem fraction_simplification (x : ℝ) : (2*x - 3) / 4 + (4*x + 5) / 3 = (22*x + 11) / 12 := by
  sorry

end fraction_simplification_l1773_177324


namespace inequality_proof_l1773_177387

theorem inequality_proof (a b : ℝ) (h1 : a < 1) (h2 : b < 1) (h3 : a + b ≥ 0.5) :
  (1 - a) * (1 - b) ≤ 9/16 := by
  sorry

end inequality_proof_l1773_177387


namespace quadratic_inequality_theorem_l1773_177351

theorem quadratic_inequality_theorem (a : ℝ) :
  (∀ m > a, ∀ x : ℝ, x^2 + 2*x + m > 0) → a = 1 :=
by sorry

end quadratic_inequality_theorem_l1773_177351


namespace one_thirds_in_nine_halves_l1773_177304

theorem one_thirds_in_nine_halves :
  (9 / 2) / (1 / 3) = 27 / 2 := by
  sorry

end one_thirds_in_nine_halves_l1773_177304


namespace hancho_milk_consumption_l1773_177320

theorem hancho_milk_consumption (total_milk : Real) (yeseul_milk : Real) (gayoung_extra : Real) (remaining_milk : Real) :
  total_milk = 1 →
  yeseul_milk = 0.1 →
  gayoung_extra = 0.2 →
  remaining_milk = 0.3 →
  total_milk - yeseul_milk - (yeseul_milk + gayoung_extra) - remaining_milk = 0.3 := by
  sorry

#check hancho_milk_consumption

end hancho_milk_consumption_l1773_177320


namespace total_time_to_grandmaster_l1773_177342

/-- Time spent on learning basic chess rules (in hours) -/
def basic_rules : ℝ := 2

/-- Factor for intermediate level time compared to basic rules -/
def intermediate_factor : ℝ := 75

/-- Factor for expert level time compared to combined basic and intermediate -/
def expert_factor : ℝ := 50

/-- Factor for master level time compared to expert level -/
def master_factor : ℝ := 30

/-- Percentage of intermediate level time spent on endgame exercises -/
def endgame_percentage : ℝ := 0.25

/-- Factor for middle game study compared to endgame exercises -/
def middle_game_factor : ℝ := 2

/-- Percentage of expert level time spent on mentoring -/
def mentoring_percentage : ℝ := 0.5

/-- Theorem: The total time James spent to become a chess grandmaster -/
theorem total_time_to_grandmaster :
  let intermediate := basic_rules * intermediate_factor
  let expert := expert_factor * (basic_rules + intermediate)
  let master := master_factor * expert
  let endgame := endgame_percentage * intermediate
  let middle_game := middle_game_factor * endgame
  let mentoring := mentoring_percentage * expert
  basic_rules + intermediate + expert + master + endgame + middle_game + mentoring = 235664.5 := by
sorry

end total_time_to_grandmaster_l1773_177342


namespace rectangle_longer_side_l1773_177332

theorem rectangle_longer_side (a : ℝ) (h1 : a > 0) : 
  (a * (0.8 * a) = 81 / 20) → a = 2.25 := by
  sorry

end rectangle_longer_side_l1773_177332


namespace correct_equation_after_digit_move_l1773_177330

theorem correct_equation_after_digit_move : 101 - 10^2 = 1 := by
  sorry

end correct_equation_after_digit_move_l1773_177330


namespace exponent_division_l1773_177399

theorem exponent_division (a : ℝ) (m n : ℕ) (h : m > n) :
  a^m / a^n = a^(m - n) := by
  sorry

end exponent_division_l1773_177399


namespace midpoint_count_l1773_177377

theorem midpoint_count (n : ℕ) (h : n ≥ 2) :
  ∃ N : ℕ, (2 * n - 3 ≤ N) ∧ (N ≤ n * (n - 1) / 2) := by
  sorry

end midpoint_count_l1773_177377


namespace calculate_total_cost_l1773_177386

/-- The cost of a single movie ticket in dollars -/
def movie_ticket_cost : ℕ := 30

/-- The number of movie tickets -/
def num_movie_tickets : ℕ := 8

/-- The number of football game tickets -/
def num_football_tickets : ℕ := 5

/-- The total cost of buying movie tickets and football game tickets -/
def total_cost : ℕ := 840

/-- Theorem stating the total cost of buying movie and football game tickets -/
theorem calculate_total_cost :
  (num_movie_tickets * movie_ticket_cost) + 
  (num_football_tickets * (num_movie_tickets * movie_ticket_cost / 2)) = total_cost :=
by sorry

end calculate_total_cost_l1773_177386


namespace simplify_nested_sqrt_l1773_177384

theorem simplify_nested_sqrt (a : ℝ) (ha : a ≥ 0) :
  Real.sqrt (Real.sqrt (a^(1/2)) * Real.sqrt (Real.sqrt (a^(1/2)) * Real.sqrt a)) = a^(1/2) := by
  sorry

end simplify_nested_sqrt_l1773_177384


namespace skew_to_common_line_relationships_l1773_177397

-- Define the concept of a line in 3D space
structure Line3D where
  -- You might represent a line using a point and a direction vector
  -- or any other suitable representation
  -- This is just a placeholder structure

-- Define the concept of skew lines
def are_skew (l1 l2 : Line3D) : Prop :=
  -- Two lines are skew if they are not parallel and do not intersect
  sorry

-- Define the possible positional relationships
inductive PositionalRelationship
  | Parallel
  | Intersecting
  | Skew

-- Theorem statement
theorem skew_to_common_line_relationships 
  (a b l : Line3D) 
  (ha : are_skew a l) 
  (hb : are_skew b l) : 
  ∃ (r : PositionalRelationship), 
    (r = PositionalRelationship.Parallel) ∨ 
    (r = PositionalRelationship.Intersecting) ∨ 
    (r = PositionalRelationship.Skew) :=
sorry

end skew_to_common_line_relationships_l1773_177397


namespace degree_three_polynomial_l1773_177383

-- Define the polynomials f and g
def f (x : ℝ) : ℝ := 2 - 15*x + 4*x^2 - 3*x^3 + 6*x^4
def g (x : ℝ) : ℝ := 4 - 3*x + x^2 - 7*x^3 + 10*x^4

-- Define the combined polynomial h
def h (c : ℝ) (x : ℝ) : ℝ := f x + c * g x

-- Theorem statement
theorem degree_three_polynomial :
  ∃ c : ℝ, (∀ x : ℝ, h c x = 2 + (-15 - 3*c)*x + (4 + c)*x^2 + (-3 - 7*c)*x^3) ∧ 
  (-3 - 7*c ≠ 0) ∧ (6 + 10*c = 0) :=
by sorry

end degree_three_polynomial_l1773_177383


namespace factorization_equality_l1773_177343

theorem factorization_equality (a b : ℝ) : 4 * a^2 * b - b = b * (2*a + 1) * (2*a - 1) := by
  sorry

end factorization_equality_l1773_177343


namespace area_ABC_is_72_l1773_177352

-- Define the points X, Y, and Z
def X : ℝ × ℝ := (6, 0)
def Y : ℝ × ℝ := (8, 4)
def Z : ℝ × ℝ := (10, 0)

-- Define the area of a triangle given its vertices
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  0.5 * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

-- Theorem statement
theorem area_ABC_is_72 :
  ∃ (A B C : ℝ × ℝ),
    triangleArea X Y Z = 0.1111111111111111 * triangleArea A B C ∧
    triangleArea A B C = 72 := by
  sorry

end area_ABC_is_72_l1773_177352


namespace largest_number_below_threshold_l1773_177308

def numbers : List ℚ := [14/10, 9/10, 12/10, 5/10, 13/10]
def threshold : ℚ := 11/10

theorem largest_number_below_threshold :
  (numbers.filter (λ x => x ≤ threshold)).maximum? = some (9/10) := by
  sorry

end largest_number_below_threshold_l1773_177308


namespace function_value_at_five_l1773_177367

def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b

theorem function_value_at_five (a b : ℝ) (h1 : f a b 1 = 3) (h2 : f a b 8 = 10) : f a b 5 = 6 := by
  sorry

end function_value_at_five_l1773_177367


namespace T_bounds_not_in_T_l1773_177345

-- Define the set T
def T : Set ℝ := {y | ∃ x : ℝ, x ≠ 1 ∧ y = (3*x + 4)/(x - 1)}

-- State the theorem
theorem T_bounds_not_in_T :
  (∃ M : ℝ, IsLUB T M ∧ M = 3) ∧
  (∀ m : ℝ, ¬IsGLB T m) ∧
  3 ∉ T ∧
  (∀ y : ℝ, y ∈ T → y < 3) :=
sorry

end T_bounds_not_in_T_l1773_177345


namespace circle_equation_l1773_177354

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the tangent line
def tangentLine (x y : ℝ) : Prop := 3 * x + 4 * y - 14 = 0

-- Define the line on which the center lies
def centerLine (x y : ℝ) : Prop := x + y - 11 = 0

-- Define the point of tangency
def tangentPoint : ℝ × ℝ := (2, 2)

-- Theorem statement
theorem circle_equation (C : Circle) :
  (tangentLine tangentPoint.1 tangentPoint.2) ∧
  (centerLine C.center.1 C.center.2) →
  ∀ (x y : ℝ), (x - C.center.1)^2 + (y - C.center.2)^2 = C.radius^2 ↔
  (x - 5)^2 + (y - 6)^2 = 25 :=
sorry

end circle_equation_l1773_177354


namespace car_distance_ratio_l1773_177322

/-- Represents a car with its speed and travel time -/
structure Car where
  speed : ℝ
  time : ℝ

/-- Calculates the distance traveled by a car -/
def distance (car : Car) : ℝ := car.speed * car.time

/-- Theorem: The ratio of distances covered by Car A and Car B is 3:1 -/
theorem car_distance_ratio :
  let car_a : Car := { speed := 50, time := 6 }
  let car_b : Car := { speed := 100, time := 1 }
  (distance car_a) / (distance car_b) = 3 := by
  sorry


end car_distance_ratio_l1773_177322


namespace range_of_a_l1773_177370

theorem range_of_a (x y a : ℝ) : 
  (77 * a = x + y) →
  (Real.sqrt (abs a) = Real.sqrt (x * y)) →
  (a ≤ -4 ∨ a ≥ 4) :=
by sorry

end range_of_a_l1773_177370


namespace jeff_performance_time_per_point_l1773_177369

/-- Represents a tennis player's performance -/
structure TennisPerformance where
  playTime : ℕ  -- play time in hours
  pointsPerMatch : ℕ  -- points needed to win a match
  gamesWon : ℕ  -- number of games won

/-- Calculates the time it takes to score a point in minutes -/
def timePerPoint (perf : TennisPerformance) : ℚ :=
  (perf.playTime * 60) / (perf.pointsPerMatch * perf.gamesWon)

/-- Theorem stating that for the given performance, it takes 5 minutes to score a point -/
theorem jeff_performance_time_per_point :
  let jeff : TennisPerformance := ⟨2, 8, 3⟩
  timePerPoint jeff = 5 := by sorry

end jeff_performance_time_per_point_l1773_177369


namespace work_completion_time_l1773_177363

theorem work_completion_time (x_time y_worked x_remaining : ℕ) (h1 : x_time = 20) (h2 : y_worked = 9) (h3 : x_remaining = 8) :
  ∃ (y_time : ℕ), y_time = 15 ∧ 
  (y_worked : ℚ) / y_time + x_remaining / x_time = 1 :=
by sorry

end work_completion_time_l1773_177363


namespace mean_equality_implies_y_equals_four_l1773_177323

/-- Given that the mean of 8, 15, and 21 is equal to the mean of 16, 24, and y, prove that y = 4 -/
theorem mean_equality_implies_y_equals_four :
  (((8 + 15 + 21) / 3) = ((16 + 24 + y) / 3)) → y = 4 :=
by sorry

end mean_equality_implies_y_equals_four_l1773_177323


namespace binomial_coefficient_equality_l1773_177344

theorem binomial_coefficient_equality (x : ℕ+) : 
  (Nat.choose 11 (2 * x.val - 1) = Nat.choose 11 x.val) → (x = 1 ∨ x = 4) :=
by sorry

end binomial_coefficient_equality_l1773_177344


namespace cyclic_quadrilateral_property_l1773_177306

-- Define a structure for a cyclic quadrilateral
structure CyclicQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  h_a : ℝ
  h_b : ℝ
  h_c : ℝ
  h_d : ℝ
  a_positive : a > 0
  b_positive : b > 0
  c_positive : c > 0
  d_positive : d > 0
  h_a_positive : h_a > 0
  h_b_positive : h_b > 0
  h_c_positive : h_c > 0
  h_d_positive : h_d > 0
  is_cyclic : True  -- Placeholder for the cyclic property
  center_inside : True  -- Placeholder for the center being inside the quadrilateral

-- State the theorem
theorem cyclic_quadrilateral_property (q : CyclicQuadrilateral) :
  q.a * q.h_c + q.c * q.h_a = q.b * q.h_d + q.d * q.h_b := by
  sorry

end cyclic_quadrilateral_property_l1773_177306


namespace number_line_problem_l1773_177374

/-- Given a number line with equally spaced markings, prove that if the starting point is 2,
    the ending point is 34, and there are 8 equal steps between them,
    then the point z reached after 6 steps from 2 is 26. -/
theorem number_line_problem (start end_ : ℝ) (total_steps : ℕ) (steps_to_z : ℕ) :
  start = 2 →
  end_ = 34 →
  total_steps = 8 →
  steps_to_z = 6 →
  let step_length := (end_ - start) / total_steps
  start + steps_to_z * step_length = 26 := by
  sorry

end number_line_problem_l1773_177374


namespace equation_solutions_l1773_177328

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 1 + Real.sqrt 5 ∧ x₂ = 1 - Real.sqrt 5 ∧
    x₁^2 - 2*x₁ - 4 = 0 ∧ x₂^2 - 2*x₂ - 4 = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ = -1 ∧ y₂ = 2 ∧
    y₁*(y₁-2) + y₁ - 2 = 0 ∧ y₂*(y₂-2) + y₂ - 2 = 0) :=
by sorry

end equation_solutions_l1773_177328


namespace decrement_calculation_l1773_177362

theorem decrement_calculation (n : ℕ) (original_mean new_mean : ℚ) 
  (h1 : n = 50)
  (h2 : original_mean = 200)
  (h3 : new_mean = 191) :
  (n : ℚ) * original_mean - n * new_mean = n * 9 := by
  sorry

end decrement_calculation_l1773_177362


namespace pizza_problem_l1773_177317

/-- Calculates the number of pizza slices left per person given the initial number of slices and the number of slices eaten. -/
def slices_left_per_person (small_pizza_slices large_pizza_slices eaten_per_person : ℕ) : ℕ :=
  let total_slices := small_pizza_slices + large_pizza_slices
  let total_eaten := 2 * eaten_per_person
  let slices_left := total_slices - total_eaten
  slices_left / 2

/-- Theorem stating that given the specific conditions of the problem, the number of slices left per person is 2. -/
theorem pizza_problem : slices_left_per_person 8 14 9 = 2 := by
  sorry

end pizza_problem_l1773_177317


namespace binomial_150_149_l1773_177355

theorem binomial_150_149 : Nat.choose 150 149 = 150 := by
  sorry

end binomial_150_149_l1773_177355


namespace largest_increase_2003_2004_l1773_177361

def students : ℕ → ℕ
  | 2002 => 70
  | 2003 => 77
  | 2004 => 85
  | 2005 => 89
  | 2006 => 95
  | 2007 => 104
  | 2008 => 112
  | _ => 0

def percentage_increase (year1 year2 : ℕ) : ℚ :=
  (students year2 - students year1 : ℚ) / students year1 * 100

def is_largest_increase (year1 year2 : ℕ) : Prop :=
  ∀ y1 y2, y1 ≥ 2002 ∧ y2 ≤ 2008 ∧ y2 = y1 + 1 →
    percentage_increase year1 year2 ≥ percentage_increase y1 y2

theorem largest_increase_2003_2004 :
  is_largest_increase 2003 2004 :=
sorry

end largest_increase_2003_2004_l1773_177361


namespace programmers_typing_speed_l1773_177339

/-- The number of programmers --/
def num_programmers : ℕ := 10

/-- The number of lines typed in 60 minutes --/
def lines_in_60_min : ℕ := 60

/-- The duration in minutes for which we want to calculate the lines typed --/
def target_duration : ℕ := 10

/-- Theorem stating that the programmers can type 100 lines in 10 minutes --/
theorem programmers_typing_speed :
  (num_programmers * lines_in_60_min * target_duration) / 60 = 100 := by
  sorry

end programmers_typing_speed_l1773_177339


namespace arithmetic_sequence_common_difference_l1773_177364

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ) (d : ℝ)
  (h_arithmetic : arithmetic_sequence a d)
  (h_mean_1_2 : (a 1 + a 2) / 2 = 1)
  (h_mean_2_3 : (a 2 + a 3) / 2 = 2) :
  d = 1 := by
sorry

end arithmetic_sequence_common_difference_l1773_177364


namespace salary_percentage_decrease_l1773_177347

/-- Calculates the percentage decrease in salary after an initial increase -/
theorem salary_percentage_decrease 
  (initial_salary : ℝ) 
  (increase_percentage : ℝ) 
  (final_salary : ℝ) 
  (h1 : initial_salary = 6000)
  (h2 : increase_percentage = 10)
  (h3 : final_salary = 6270) :
  let increased_salary := initial_salary * (1 + increase_percentage / 100)
  let decrease_percentage := (increased_salary - final_salary) / increased_salary * 100
  decrease_percentage = 5 := by sorry

end salary_percentage_decrease_l1773_177347


namespace n_gon_division_l1773_177376

/-- The number of parts into which the diagonals of an n-gon divide it, 
    given that no three diagonals intersect at one point. -/
def numberOfParts (n : ℕ) : ℚ :=
  1 + (n * (n - 3) / 2) + (n * (n - 1) * (n - 2) * (n - 3) / 24)

/-- Theorem stating that the number of parts into which the diagonals of an n-gon divide it,
    given that no three diagonals intersect at one point, is equal to the formula. -/
theorem n_gon_division (n : ℕ) (h : n ≥ 3) : 
  numberOfParts n = 1 + (n * (n - 3) / 2) + (n * (n - 1) * (n - 2) * (n - 3) / 24) := by
  sorry

end n_gon_division_l1773_177376


namespace sum_and_count_30_to_40_l1773_177315

def sum_range (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

def count_even_in_range (a b : ℕ) : ℕ := (b - a) / 2 + 1

theorem sum_and_count_30_to_40 : 
  sum_range 30 40 + count_even_in_range 30 40 = 391 := by
  sorry

end sum_and_count_30_to_40_l1773_177315


namespace profit_calculation_l1773_177329

theorem profit_calculation (P Q R : ℚ) (profit_R : ℚ) :
  4 * P = 6 * Q ∧ 6 * Q = 10 * R ∧ profit_R = 840 →
  (P + Q + R) * (profit_R / R) = 4340 := by
sorry

end profit_calculation_l1773_177329


namespace arithmetic_sequence_sum_16_l1773_177336

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  d : ℤ      -- Common difference
  first : a 1 = a 1  -- First term (tautology to define a₁)
  arithmetic : ∀ n : ℕ, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  n * (2 * seq.a 1 + (n - 1) * seq.d) / 2

theorem arithmetic_sequence_sum_16 (seq : ArithmeticSequence) 
  (h₁ : seq.a 12 = -8)
  (h₂ : S seq 9 = -9) :
  S seq 16 = -72 := by
  sorry

end arithmetic_sequence_sum_16_l1773_177336


namespace sam_watermelons_l1773_177396

theorem sam_watermelons (grown : ℕ) (eaten : ℕ) (h1 : grown = 4) (h2 : eaten = 3) :
  grown - eaten = 1 := by
  sorry

end sam_watermelons_l1773_177396


namespace range_of_f_l1773_177375

def f (x : ℝ) : ℝ := x^2 + 2*x - 1

theorem range_of_f :
  let S := {y | ∃ x ∈ Set.Icc (-2 : ℝ) 2, f x = y}
  S = Set.Icc (-2 : ℝ) 7 := by sorry

end range_of_f_l1773_177375


namespace students_playing_sports_l1773_177365

theorem students_playing_sports (basketball cricket both : ℕ) 
  (hb : basketball = 7)
  (hc : cricket = 8)
  (hboth : both = 3) :
  basketball + cricket - both = 12 := by
  sorry

end students_playing_sports_l1773_177365


namespace ab_plus_cd_value_l1773_177310

theorem ab_plus_cd_value (a b c d : ℝ) 
  (eq1 : a + b + c = 5)
  (eq2 : a + b + d = 1)
  (eq3 : a + c + d = 16)
  (eq4 : b + c + d = 9) :
  a * b + c * d = 734 / 9 := by
  sorry

end ab_plus_cd_value_l1773_177310


namespace orange_selling_loss_l1773_177321

/-- Calculates the percentage loss when selling oranges at a given rate per rupee,
    given the rate that would result in a 44% gain. -/
def calculate_loss_percentage (loss_rate : ℚ) (gain_rate : ℚ) (gain_percentage : ℚ) : ℚ :=
  let cost_price := 1 / (gain_rate * (1 + gain_percentage))
  let loss := cost_price - 1 / loss_rate
  (loss / cost_price) * 100

/-- The percentage loss when selling oranges at 36 per rupee is approximately 4.17%,
    given that selling at 24 per rupee results in a 44% gain. -/
theorem orange_selling_loss : 
  let loss_rate : ℚ := 36
  let gain_rate : ℚ := 24
  let gain_percentage : ℚ := 44 / 100
  let calculated_loss := calculate_loss_percentage loss_rate gain_rate gain_percentage
  ∃ ε > 0, abs (calculated_loss - 4.17) < ε ∧ ε < 0.01 :=
sorry

end orange_selling_loss_l1773_177321


namespace system_of_equations_solutions_l1773_177382

theorem system_of_equations_solutions :
  let solutions : List (ℂ × ℂ × ℂ) := [
    (0, 0, 0),
    (2/3, -1/3, -1/3),
    (1/3, 1/3, 1/3),
    (1, 0, 0),
    (2/3, (1 + Complex.I * Real.sqrt 3) / 6, (1 - Complex.I * Real.sqrt 3) / 6),
    (2/3, (1 - Complex.I * Real.sqrt 3) / 6, (1 + Complex.I * Real.sqrt 3) / 6),
    (1/3, (1 + Complex.I * Real.sqrt 3) / 6, (1 - Complex.I * Real.sqrt 3) / 6),
    (1/3, (1 - Complex.I * Real.sqrt 3) / 6, (1 + Complex.I * Real.sqrt 3) / 6)
  ]
  ∀ x y z : ℂ,
    (x^2 + 2*y*z = x ∧ y^2 + 2*z*x = z ∧ z^2 + 2*x*y = y) ↔ (x, y, z) ∈ solutions := by
  sorry

end system_of_equations_solutions_l1773_177382


namespace line_of_sight_condition_l1773_177372

-- Define the curve C
def C (x : ℝ) : ℝ := 2 * x^2

-- Define point A
def A : ℝ × ℝ := (0, -2)

-- Define point B
def B (a : ℝ) : ℝ × ℝ := (3, a)

-- Define the condition for line of sight not being blocked
def lineOfSightNotBlocked (a : ℝ) : Prop :=
  ∀ x : ℝ, 0 < x → x < 3 → 
    (A.2 + (B a).2 - A.2) / 3 * x + A.2 > C x

-- State the theorem
theorem line_of_sight_condition :
  ∀ a : ℝ, lineOfSightNotBlocked a ↔ a < 10 := by sorry

end line_of_sight_condition_l1773_177372


namespace order_of_exponential_expressions_l1773_177373

theorem order_of_exponential_expressions :
  let a := Real.exp (2 * Real.log 3 * Real.log 2)
  let b := Real.exp (3 * Real.log 2 * Real.log 3)
  let c := Real.exp (Real.log 5 * Real.log 5)
  a < b ∧ b < c := by
  sorry

end order_of_exponential_expressions_l1773_177373


namespace circle_transform_prime_impossibility_l1773_177314

/-- Represents the transformation of four numbers on a circle -/
def circle_transform (a b c d : ℤ) : (ℤ × ℤ × ℤ × ℤ) :=
  (a - b, b - c, c - d, d - a)

/-- Applies the circle transformation n times -/
def iterate_transform (n : ℕ) (a b c d : ℤ) : (ℤ × ℤ × ℤ × ℤ) :=
  match n with
  | 0 => (a, b, c, d)
  | n + 1 =>
    let (a', b', c', d') := iterate_transform n a b c d
    circle_transform a' b' c' d'

/-- Checks if a number is prime -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem circle_transform_prime_impossibility :
  ∀ a b c d : ℤ,
  let (a', b', c', d') := iterate_transform 1996 a b c d
  ¬(is_prime (|b' * c' - a' * d'|.natAbs) ∧
    is_prime (|a' * c' - b' * d'|.natAbs) ∧
    is_prime (|a' * b' - c' * d'|.natAbs)) := by
  sorry

end circle_transform_prime_impossibility_l1773_177314


namespace product_of_base8_digits_7432_l1773_177300

/-- Converts a natural number from base 10 to base 8 --/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- Calculates the product of a list of natural numbers --/
def productOfList (l : List ℕ) : ℕ :=
  sorry

/-- The product of the digits in the base 8 representation of 7432₁₀ is 192 --/
theorem product_of_base8_digits_7432 :
  productOfList (toBase8 7432) = 192 := by
  sorry

end product_of_base8_digits_7432_l1773_177300


namespace forester_tree_planting_l1773_177327

theorem forester_tree_planting (initial_trees : ℕ) (monday_multiplier : ℕ) (tuesday_fraction : ℚ) : 
  initial_trees = 30 →
  monday_multiplier = 3 →
  tuesday_fraction = 1/3 →
  (monday_multiplier * initial_trees - initial_trees) + 
  (tuesday_fraction * (monday_multiplier * initial_trees - initial_trees)) = 80 := by
sorry

end forester_tree_planting_l1773_177327


namespace min_value_theorem_l1773_177309

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 1 / (x + 3) + 1 / (y + 3) = 1 / 4) :
  ∀ a b : ℝ, a > 0 ∧ b > 0 ∧ 1 / (a + 3) + 1 / (b + 3) = 1 / 4 →
  3 * x + 4 * y ≤ 3 * a + 4 * b ∧
  ∃ c d : ℝ, c > 0 ∧ d > 0 ∧
    1 / (c + 3) + 1 / (d + 3) = 1 / 4 ∧
    3 * c + 4 * d = 21 :=
by sorry

end min_value_theorem_l1773_177309


namespace card_number_factorization_l1773_177305

/-- Represents a set of 90 cards with 10 each of digits 1 through 9 -/
def CardSet := Finset (Fin 9)

/-- Predicate to check if a number can be formed from the given card set -/
def canBeFormedFromCards (n : ℕ) (cards : CardSet) : Prop := sorry

/-- Predicate to check if a number can be factored into four natural factors each greater than one -/
def hasEligibleFactorization (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), a > 1 ∧ b > 1 ∧ c > 1 ∧ d > 1 ∧ n = a * b * c * d

/-- Main theorem statement -/
theorem card_number_factorization (cards : CardSet) (A B : ℕ) :
  (canBeFormedFromCards A cards) →
  (canBeFormedFromCards B cards) →
  B = 3 * A →
  A > 0 →
  (hasEligibleFactorization A ∨ hasEligibleFactorization B) := by sorry

end card_number_factorization_l1773_177305


namespace area_ratio_rectangle_square_l1773_177318

/-- Given a square S and a rectangle R where:
    - The longer side of R is 20% more than a side of S
    - The shorter side of R is 15% less than a side of S
    Prove that the ratio of the area of R to the area of S is 51/50 -/
theorem area_ratio_rectangle_square (S : Real) (R : Real × Real) : 
  R.1 = 1.2 * S ∧ R.2 = 0.85 * S → 
  (R.1 * R.2) / (S * S) = 51 / 50 := by
sorry

end area_ratio_rectangle_square_l1773_177318


namespace imaginary_part_of_complex_fraction_l1773_177303

theorem imaginary_part_of_complex_fraction : 
  Complex.im (Complex.I^3 / (2 * Complex.I - 1)) = 1/5 := by sorry

end imaginary_part_of_complex_fraction_l1773_177303


namespace isosceles_right_triangle_area_l1773_177392

/-- The area of an isosceles right triangle with hypotenuse 6√2 is 18 -/
theorem isosceles_right_triangle_area (h : ℝ) (A : ℝ) : 
  h = 6 * Real.sqrt 2 →  -- hypotenuse length
  A = (h^2) / 4 →        -- area formula for isosceles right triangle
  A = 18 := by
sorry

end isosceles_right_triangle_area_l1773_177392


namespace expression_simplification_l1773_177368

theorem expression_simplification (m : ℝ) (h : m = 2) : 
  (m^2 / (1 - m^2)) * (1 - 1/m) = -2/3 := by
  sorry

end expression_simplification_l1773_177368


namespace so3_required_moles_l1773_177349

/-- Represents a chemical species in a reaction -/
inductive Species
| SO3
| H2O
| H2SO4

/-- Represents the stoichiometric coefficient of a species in a reaction -/
def stoich_coeff (s : Species) : ℚ :=
  match s with
  | Species.SO3 => 1
  | Species.H2O => 1
  | Species.H2SO4 => 1

/-- The amount of H2O available in moles -/
def h2o_available : ℚ := 2

/-- The amount of H2SO4 to be formed in moles -/
def h2so4_formed : ℚ := 2

/-- Theorem: The number of moles of SO3 required is 2 -/
theorem so3_required_moles : 
  let so3_moles := h2so4_formed / stoich_coeff Species.H2SO4 * stoich_coeff Species.SO3
  so3_moles = 2 := by sorry

end so3_required_moles_l1773_177349


namespace marble_distribution_l1773_177380

theorem marble_distribution (a : ℕ) : 
  let angela := a
  let brian := 2 * angela
  let caden := 3 * brian
  let daryl := 5 * caden
  angela + brian + caden + daryl = 78 → a = 2 := by
  sorry

end marble_distribution_l1773_177380


namespace geometric_sum_first_eight_terms_l1773_177350

/-- The sum of the first n terms of a geometric sequence with first term a and common ratio r -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The first term of the geometric sequence -/
def a : ℚ := 1/3

/-- The common ratio of the geometric sequence -/
def r : ℚ := 1/3

/-- The number of terms to sum -/
def n : ℕ := 8

theorem geometric_sum_first_eight_terms :
  geometric_sum a r n = 3280/6561 := by
  sorry

end geometric_sum_first_eight_terms_l1773_177350


namespace square_difference_1001_999_l1773_177381

theorem square_difference_1001_999 : 1001^2 - 999^2 = 4000 := by
  sorry

end square_difference_1001_999_l1773_177381


namespace equation_solutions_l1773_177307

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = (-3 + Real.sqrt 13) / 2 ∧ x₂ = (-3 - Real.sqrt 13) / 2 ∧
    x₁^2 + 3*x₁ - 1 = 0 ∧ x₂^2 + 3*x₂ - 1 = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ = 2 ∧ y₂ = 4 ∧
    (y₁ - 2)^2 = 2*(y₁ - 2) ∧ (y₂ - 2)^2 = 2*(y₂ - 2)) :=
by sorry

end equation_solutions_l1773_177307


namespace allowance_percentage_increase_l1773_177359

def middle_school_allowance : ℕ := 8 + 2

def senior_year_allowance : ℕ := 2 * middle_school_allowance + 5

def allowance_increase : ℕ := senior_year_allowance - middle_school_allowance

def percentage_increase : ℚ := (allowance_increase : ℚ) / (middle_school_allowance : ℚ) * 100

theorem allowance_percentage_increase :
  percentage_increase = 150 := by sorry

end allowance_percentage_increase_l1773_177359


namespace min_cost_at_one_l1773_177360

/-- Represents the transportation problem for mangoes between supermarkets and destinations -/
structure MangoTransportation where
  supermarket_A_stock : ℝ
  supermarket_B_stock : ℝ
  destination_X_demand : ℝ
  destination_Y_demand : ℝ
  cost_A_to_X : ℝ
  cost_A_to_Y : ℝ
  cost_B_to_X : ℝ
  cost_B_to_Y : ℝ

/-- Calculates the total transportation cost given the amount transported from A to X -/
def total_cost (mt : MangoTransportation) (x : ℝ) : ℝ :=
  mt.cost_A_to_X * x + 
  mt.cost_A_to_Y * (mt.supermarket_A_stock - x) + 
  mt.cost_B_to_X * (mt.destination_X_demand - x) + 
  mt.cost_B_to_Y * (x - 1)

/-- Theorem stating that the minimum transportation cost occurs when x = 1 -/
theorem min_cost_at_one (mt : MangoTransportation) 
  (h1 : mt.supermarket_A_stock = 15)
  (h2 : mt.supermarket_B_stock = 15)
  (h3 : mt.destination_X_demand = 16)
  (h4 : mt.destination_Y_demand = 14)
  (h5 : mt.cost_A_to_X = 50)
  (h6 : mt.cost_A_to_Y = 30)
  (h7 : mt.cost_B_to_X = 60)
  (h8 : mt.cost_B_to_Y = 45)
  (h9 : ∀ x, 1 ≤ x ∧ x ≤ 15 → total_cost mt 1 ≤ total_cost mt x) :
  ∃ (min_x : ℝ), min_x = 1 ∧ 
    ∀ x, 1 ≤ x ∧ x ≤ 15 → total_cost mt min_x ≤ total_cost mt x :=
  sorry

end min_cost_at_one_l1773_177360


namespace intersection_of_M_and_N_l1773_177393

open Set

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x, y = x^2}
def N : Set ℝ := {y | ∃ x, y = x}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {y | 0 ≤ y} := by sorry

end intersection_of_M_and_N_l1773_177393


namespace sin_absolute_value_condition_l1773_177337

theorem sin_absolute_value_condition (α : ℝ) :
  (|Real.sin α| = -Real.sin α) ↔ ∃ k : ℤ, α ∈ Set.Icc ((2 * k - 1) * Real.pi) (2 * k * Real.pi) := by
  sorry

end sin_absolute_value_condition_l1773_177337


namespace triangle_side_lengths_l1773_177335

/-- Given a triangle ABC with sides a, b, and c, prove that if a = b + 1, b = c + 1, 
    and the perimeter is 21, then a = 8, b = 7, and c = 6. -/
theorem triangle_side_lengths 
  (a b c : ℝ) 
  (h1 : a = b + 1) 
  (h2 : b = c + 1) 
  (h3 : a + b + c = 21) : 
  a = 8 ∧ b = 7 ∧ c = 6 := by
  sorry


end triangle_side_lengths_l1773_177335


namespace diamond_inequality_exists_l1773_177325

/-- Definition of the diamond operation -/
def diamond (f : ℝ → ℝ) (x y : ℝ) : ℝ := |f x - f y|

/-- The function f(x) = 3x -/
def f (x : ℝ) : ℝ := 3 * x

/-- Theorem stating that 3(x ◊ y) ≠ (3x) ◊ (3y) for some x and y -/
theorem diamond_inequality_exists : ∃ x y : ℝ, 3 * (diamond f x y) ≠ diamond f (3 * x) (3 * y) := by
  sorry

end diamond_inequality_exists_l1773_177325


namespace street_length_calculation_l1773_177326

/-- Proves that given a speed of 5.31 km/h and a time of 8 minutes, the distance traveled is 708 meters. -/
theorem street_length_calculation (speed : ℝ) (time : ℝ) : 
  speed = 5.31 → time = 8 → speed * time * (1000 / 60) = 708 :=
by sorry

end street_length_calculation_l1773_177326


namespace smallest_value_expression_l1773_177371

theorem smallest_value_expression (x : ℝ) (h : x = -3) :
  let a := x^2 - 3
  let b := (x - 3)^2
  let c := x^2
  let d := (x + 3)^2
  let e := x^2 + 3
  d ≤ a ∧ d ≤ b ∧ d ≤ c ∧ d ≤ e :=
by sorry

end smallest_value_expression_l1773_177371


namespace stationery_costs_l1773_177311

theorem stationery_costs : ∃ (x y z : ℕ+), 
  (x : ℤ) % 2 = 0 ∧
  x + 3*y + 2*z = 98 ∧
  3*x + y = 5*z - 36 ∧
  x = 4 ∧ y = 22 ∧ z = 14 := by
sorry

end stationery_costs_l1773_177311


namespace tangent_line_at_one_l1773_177341

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem tangent_line_at_one :
  ∃ (m b : ℝ), ∀ x y : ℝ,
    y = m * x + b ∧
    (∃ h : x > 0, y = f x) →
    (x = 1 → y = f 1) ∧
    (∀ ε > 0, ∃ δ > 0, ∀ x', 0 < |x' - 1| ∧ |x' - 1| < δ →
      |y - (f 1 + (x' - 1) * ((f x' - f 1) / (x' - 1)))| / |x' - 1| < ε) →
    m = 1 ∧ b = -1 :=
sorry

end tangent_line_at_one_l1773_177341


namespace function_condition_implies_b_bound_l1773_177333

theorem function_condition_implies_b_bound (b : ℝ) :
  (∃ x ∈ Set.Icc (1/2 : ℝ) 2, 
    Real.exp x * (x - b) + x * (Real.exp x * (x - b + 2)) > 0) →
  b < 8/3 := by
  sorry

end function_condition_implies_b_bound_l1773_177333


namespace carries_work_hours_l1773_177366

/-- Proves that Carrie worked 2 hours each day to earn a profit of $122 -/
theorem carries_work_hours 
  (days : ℕ) 
  (hourly_rate : ℚ) 
  (supply_cost : ℚ) 
  (profit : ℚ) 
  (h : ℚ)
  (h_days : days = 4)
  (h_rate : hourly_rate = 22)
  (h_cost : supply_cost = 54)
  (h_profit : profit = 122)
  (h_equation : profit = days * hourly_rate * h - supply_cost) : 
  h = 2 := by
  sorry

end carries_work_hours_l1773_177366


namespace swimming_club_girls_l1773_177338

theorem swimming_club_girls (total : ℕ) (present : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 30 →
  present = 18 →
  boys + girls = total →
  boys / 3 + girls = present →
  girls = 12 := by
sorry

end swimming_club_girls_l1773_177338


namespace car_wash_earnings_l1773_177340

theorem car_wash_earnings :
  ∀ (total lisa tommy : ℝ),
    lisa = total / 2 →
    tommy = lisa / 2 →
    lisa = tommy + 15 →
    total = 60 :=
by
  sorry

end car_wash_earnings_l1773_177340


namespace modulus_of_complex_product_l1773_177313

theorem modulus_of_complex_product : Complex.abs ((3 - 4 * Complex.I) * Complex.I) = 5 := by
  sorry

end modulus_of_complex_product_l1773_177313


namespace tangent_line_and_monotonicity_l1773_177319

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 3*x - 9

-- Define the derivative of f(x)
def f_prime (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + 3

theorem tangent_line_and_monotonicity (a : ℝ) :
  (f_prime a (-3) = 0 →
    ∃ m b : ℝ, ∀ x y : ℝ, y = f a x → (y = m*x + b ↔ x = 0 ∨ y - f a 0 = m*(x - 0))) ∧
  ((∀ x : ℝ, x ∈ Set.Icc 1 2 → f_prime a x ≤ 0) →
    a ≤ -15/4) := by sorry

end tangent_line_and_monotonicity_l1773_177319


namespace first_discount_percentage_l1773_177356

theorem first_discount_percentage (original_price : ℝ) (final_price : ℝ) (second_discount : ℝ) :
  original_price = 510 →
  final_price = 381.48 →
  second_discount = 15 →
  ∃ (first_discount : ℝ),
    first_discount = 12 ∧
    final_price = original_price * (1 - first_discount / 100) * (1 - second_discount / 100) :=
by
  sorry

end first_discount_percentage_l1773_177356


namespace rationalize_denominator_l1773_177331

theorem rationalize_denominator : 
  (Real.sqrt 12 + Real.sqrt 5) / (Real.sqrt 3 + Real.sqrt 5) = (Real.sqrt 15 - 1) / 2 := by
sorry

end rationalize_denominator_l1773_177331


namespace fraction_simplification_l1773_177388

theorem fraction_simplification : (5 * 7) / 10 = 3.5 := by
  sorry

end fraction_simplification_l1773_177388


namespace first_book_price_l1773_177302

/-- Given 41 books arranged in increasing price order with a $3 difference between adjacent books,
    if the sum of the prices of the first and last books is $246,
    then the price of the first book is $63. -/
theorem first_book_price (n : ℕ) (price_diff : ℝ) (total_sum : ℝ) :
  n = 41 →
  price_diff = 3 →
  total_sum = 246 →
  ∃ (first_price : ℝ),
    first_price + (first_price + price_diff * (n - 1)) = total_sum ∧
    first_price = 63 := by
  sorry

end first_book_price_l1773_177302


namespace circle_ratio_after_increase_l1773_177391

/-- Given a circle with original radius r₀ > 0, prove that when the radius is
    increased by 50%, the ratio of the new circumference to the new area
    is equal to 4 / (3r₀). -/
theorem circle_ratio_after_increase (r₀ : ℝ) (h : r₀ > 0) :
  let new_radius := 1.5 * r₀
  let new_circumference := 2 * Real.pi * new_radius
  let new_area := Real.pi * new_radius ^ 2
  new_circumference / new_area = 4 / (3 * r₀) := by
sorry

end circle_ratio_after_increase_l1773_177391


namespace missing_entry_is_L_l1773_177353

/-- Represents the possible entries in the table -/
inductive TableEntry
| W
| Q
| L

/-- Represents a position in the 3x3 table -/
structure Position :=
  (row : Fin 3)
  (col : Fin 3)

/-- Represents the 3x3 table -/
def Table := Position → TableEntry

/-- The given table with known entries -/
def givenTable : Table :=
  fun pos => match pos with
  | ⟨0, 0⟩ => TableEntry.W
  | ⟨0, 2⟩ => TableEntry.Q
  | ⟨1, 0⟩ => TableEntry.L
  | ⟨1, 1⟩ => TableEntry.Q
  | ⟨1, 2⟩ => TableEntry.W
  | ⟨2, 0⟩ => TableEntry.Q
  | ⟨2, 1⟩ => TableEntry.W
  | ⟨2, 2⟩ => TableEntry.L
  | _ => TableEntry.W  -- Default value for unknown positions

theorem missing_entry_is_L :
  givenTable ⟨0, 1⟩ = TableEntry.L :=
sorry

end missing_entry_is_L_l1773_177353


namespace tenth_equation_right_side_l1773_177346

/-- The sum of the first n natural numbers -/
def sum_of_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sum of cubes of the first n natural numbers -/
def sum_of_cubes (n : ℕ) : ℕ := (sum_of_n n) ^ 2

theorem tenth_equation_right_side :
  sum_of_cubes 10 = 55^2 := by sorry

end tenth_equation_right_side_l1773_177346


namespace prob_less_than_8_ring_l1773_177316

def prob_10_ring : ℝ := 0.3
def prob_9_ring : ℝ := 0.3
def prob_8_ring : ℝ := 0.2

theorem prob_less_than_8_ring :
  1 - (prob_10_ring + prob_9_ring + prob_8_ring) = 0.2 := by
  sorry

end prob_less_than_8_ring_l1773_177316


namespace adam_has_more_apples_l1773_177358

/-- The number of apples Adam has -/
def adam_apples : ℕ := 14

/-- The number of apples Jackie has -/
def jackie_apples : ℕ := 9

/-- The difference in apples between Adam and Jackie -/
def apple_difference : ℕ := adam_apples - jackie_apples

theorem adam_has_more_apples : apple_difference = 5 := by
  sorry

end adam_has_more_apples_l1773_177358


namespace max_lilacs_purchase_lilac_purchase_proof_l1773_177385

theorem max_lilacs_purchase (cost_per_lilac : ℕ) (max_total_cost : ℕ) : ℕ :=
  let max_lilacs := max_total_cost / cost_per_lilac
  if max_lilacs * cost_per_lilac > max_total_cost then
    max_lilacs - 1
  else
    max_lilacs

theorem lilac_purchase_proof :
  max_lilacs_purchase 6 5000 = 833 :=
by sorry

end max_lilacs_purchase_lilac_purchase_proof_l1773_177385


namespace inequality_proof_l1773_177334

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a * b + b * c + c * a = 1) : 
  1 / (a^2 + 1) + 1 / (b^2 + 1) + 1 / (c^2 + 1) ≤ 9/4 := by
  sorry

end inequality_proof_l1773_177334


namespace cost_price_calculation_l1773_177398

/-- Proves that the cost price of an article is $975, given that it was sold at $1170 with a 20% profit. -/
theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) :
  selling_price = 1170 →
  profit_percentage = 20 →
  selling_price = (100 + profit_percentage) / 100 * 975 :=
by sorry

end cost_price_calculation_l1773_177398


namespace school_bus_distance_l1773_177390

/-- Calculates the total distance traveled by a school bus under specific conditions -/
theorem school_bus_distance : 
  let initial_velocity := 0
  let acceleration := 2
  let acceleration_time := 30
  let constant_speed_time := 20 * 60
  let deceleration := 1
  let final_velocity := acceleration * acceleration_time
  let distance_constant_speed := final_velocity * constant_speed_time
  let distance_deceleration := final_velocity^2 / (2 * deceleration)
  distance_constant_speed + distance_deceleration = 73800 := by
  sorry

end school_bus_distance_l1773_177390


namespace circle_equation_from_conditions_l1773_177348

/-- The equation of a circle given specific conditions -/
theorem circle_equation_from_conditions :
  ∀ (M : ℝ × ℝ),
  (2 * M.1 + M.2 - 1 = 0) →  -- M lies on the line 2x + y - 1 = 0
  (∃ (r : ℝ), r > 0 ∧
    ((M.1 - 3)^2 + M.2^2 = r^2) ∧  -- (3,0) is on the circle
    ((M.1 - 0)^2 + (M.2 - 1)^2 = r^2)) →  -- (0,1) is on the circle
  (∀ (x y : ℝ), (x - 1)^2 + (y + 1)^2 = 5 ↔
    ((x - M.1)^2 + (y - M.2)^2 = ((M.1 - 3)^2 + M.2^2))) :=
by sorry

end circle_equation_from_conditions_l1773_177348


namespace divisible_by_240_l1773_177389

-- Define a prime number p that is greater than or equal to 7
def p : ℕ := sorry

-- Axiom: p is prime
axiom p_prime : Nat.Prime p

-- Axiom: p is greater than or equal to 7
axiom p_ge_7 : p ≥ 7

-- Theorem to prove
theorem divisible_by_240 : 240 ∣ p^4 - 1 := by
  sorry

end divisible_by_240_l1773_177389


namespace ken_summit_time_l1773_177357

/-- Represents the climbing scenario of Sari and Ken -/
structure ClimbingScenario where
  sari_start_time : ℕ  -- in hours after midnight
  ken_start_time : ℕ   -- in hours after midnight
  initial_distance : ℝ  -- distance Sari is ahead when Ken starts
  ken_pace : ℝ          -- Ken's climbing pace in meters per hour
  final_distance : ℝ    -- distance Sari is behind when Ken reaches summit

/-- The time it takes Ken to reach the summit -/
def time_to_summit (scenario : ClimbingScenario) : ℝ :=
  sorry

/-- Theorem stating that Ken reaches the summit 5 hours after starting -/
theorem ken_summit_time (scenario : ClimbingScenario) 
  (h1 : scenario.sari_start_time = 8)
  (h2 : scenario.ken_start_time = 10)
  (h3 : scenario.initial_distance = 700)
  (h4 : scenario.ken_pace = 500)
  (h5 : scenario.final_distance = 50) :
  time_to_summit scenario = 5 :=
sorry

end ken_summit_time_l1773_177357


namespace frac_5_13_150th_digit_l1773_177394

def decimal_expansion (n d : ℕ) : List ℕ := sorry

def nth_digit_after_decimal (n d : ℕ) (k : ℕ) : ℕ := sorry

theorem frac_5_13_150th_digit :
  nth_digit_after_decimal 5 13 150 = 5 := by sorry

end frac_5_13_150th_digit_l1773_177394


namespace problem_statement_l1773_177378

theorem problem_statement (a b c : Int) (h1 : a = -2) (h2 : b = 3) (h3 : c = -4) :
  a - (b - c) = -9 := by
  sorry

end problem_statement_l1773_177378


namespace unique_x_with_three_prime_divisors_including_31_l1773_177395

theorem unique_x_with_three_prime_divisors_including_31 :
  ∀ (x n : ℕ),
    x = 8^n - 1 →
    (∃ (p q : ℕ), Prime p ∧ Prime q ∧ p ≠ q ∧ p ≠ 31 ∧ q ≠ 31 ∧ x = 31 * p * q) →
    (∀ (r : ℕ), Prime r ∧ r ∣ x → r = 31 ∨ r = p ∨ r = q) →
    x = 32767 :=
by sorry

end unique_x_with_three_prime_divisors_including_31_l1773_177395


namespace square_perimeter_product_l1773_177312

theorem square_perimeter_product (x y : ℝ) (h1 : x^2 + y^2 = 130) (h2 : x^2 - y^2 = 58) :
  (4*x) * (4*y) = 96 * Real.sqrt 94 := by
  sorry

end square_perimeter_product_l1773_177312


namespace geometric_sequence_problem_l1773_177301

theorem geometric_sequence_problem (a b c d e : ℕ) : 
  (2 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < 100) →
  Nat.gcd a e = 1 →
  (∃ (r : ℚ), r > 1 ∧ b = a * r ∧ c = a * r^2 ∧ d = a * r^3 ∧ e = a * r^4) →
  c = 36 := by
  sorry

end geometric_sequence_problem_l1773_177301


namespace geometric_series_comparison_l1773_177379

theorem geometric_series_comparison (a₁ : ℚ) (r₁ r₂ : ℚ) :
  a₁ = 5/12 →
  r₁ = 3/4 →
  r₂ < 1 →
  r₂ > 0 →
  a₁ / (1 - r₁) > a₁ / (1 - r₂) →
  r₂ = 5/6 ∧ a₁ / (1 - r₁) = 5/3 ∧ a₁ / (1 - r₂) = 5/2 :=
by sorry

end geometric_series_comparison_l1773_177379
