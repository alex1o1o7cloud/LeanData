import Mathlib

namespace NUMINAMATH_CALUDE_cube_sum_inequality_l2689_268995

theorem cube_sum_inequality (x y z : ℝ) (h : x + y + z = 0) :
  6 * (x^3 + y^3 + z^3)^2 ≤ (x^2 + y^2 + z^2)^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_inequality_l2689_268995


namespace NUMINAMATH_CALUDE_voting_scenario_theorem_l2689_268947

/-- Represents the voting scenario in a certain city -/
structure VotingScenario where
  total_voters : ℝ
  dem_percent : ℝ
  rep_percent : ℝ
  dem_for_A_percent : ℝ
  total_for_A_percent : ℝ
  rep_for_A_percent : ℝ

/-- The theorem statement for the voting scenario problem -/
theorem voting_scenario_theorem (v : VotingScenario) :
  v.dem_percent = 0.6 ∧
  v.rep_percent = 0.4 ∧
  v.dem_for_A_percent = 0.75 ∧
  v.total_for_A_percent = 0.57 →
  v.rep_for_A_percent = 0.3 := by
  sorry


end NUMINAMATH_CALUDE_voting_scenario_theorem_l2689_268947


namespace NUMINAMATH_CALUDE_sum_of_fifth_powers_l2689_268948

theorem sum_of_fifth_powers (a b c : ℝ) 
  (sum_condition : a + b + c = 1)
  (sum_squares_condition : a^2 + b^2 + c^2 = 3)
  (sum_cubes_condition : a^3 + b^3 + c^3 = 4) :
  a^5 + b^5 + c^5 = 11/3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_fifth_powers_l2689_268948


namespace NUMINAMATH_CALUDE_claire_balloons_l2689_268952

/-- The number of balloons Claire has at the end of the fair --/
def final_balloon_count (initial : ℕ) (given_to_girl : ℕ) (floated_away : ℕ) (given_away : ℕ) (taken_from_coworker : ℕ) : ℕ :=
  initial - given_to_girl - floated_away - given_away + taken_from_coworker

/-- Theorem stating that Claire ends up with 39 balloons --/
theorem claire_balloons : 
  final_balloon_count 50 1 12 9 11 = 39 := by
  sorry

end NUMINAMATH_CALUDE_claire_balloons_l2689_268952


namespace NUMINAMATH_CALUDE_percentage_with_no_conditions_is_7_5_l2689_268944

/-- Represents the survey data of teachers' health conditions -/
structure SurveyData where
  total : ℕ
  highBP : ℕ
  heartTrouble : ℕ
  cholesterol : ℕ
  highBP_heartTrouble : ℕ
  heartTrouble_cholesterol : ℕ
  cholesterol_highBP : ℕ
  all_three : ℕ

/-- Calculates the percentage of teachers with none of the conditions -/
def percentageWithNoConditions (data : SurveyData) : ℚ :=
  let withAtLeastOne := data.highBP + data.heartTrouble + data.cholesterol
    - data.highBP_heartTrouble - data.heartTrouble_cholesterol - data.cholesterol_highBP
    + data.all_three
  let withNone := data.total - withAtLeastOne
  (withNone : ℚ) / (data.total : ℚ) * 100

/-- The survey data from the problem -/
def surveyData : SurveyData := {
  total := 200,
  highBP := 110,
  heartTrouble := 80,
  cholesterol := 50,
  highBP_heartTrouble := 30,
  heartTrouble_cholesterol := 20,
  cholesterol_highBP := 10,
  all_three := 5
}

/-- Theorem stating that the percentage of teachers with none of the conditions is 7.5% -/
theorem percentage_with_no_conditions_is_7_5 :
  percentageWithNoConditions surveyData = 15/2 := by
  sorry

end NUMINAMATH_CALUDE_percentage_with_no_conditions_is_7_5_l2689_268944


namespace NUMINAMATH_CALUDE_family_size_l2689_268984

theorem family_size (boys girls : ℕ) 
  (sister_condition : boys = girls - 1)
  (brother_condition : girls = 2 * (boys - 1)) : 
  boys + girls = 7 := by
sorry

end NUMINAMATH_CALUDE_family_size_l2689_268984


namespace NUMINAMATH_CALUDE_vampire_conversion_theorem_l2689_268980

/-- The number of people each vampire turns into vampires per night. -/
def vampire_conversion_rate : ℕ → Prop := λ x =>
  let initial_population : ℕ := 300
  let initial_vampires : ℕ := 2
  let nights : ℕ := 2
  let final_vampires : ℕ := 72
  
  -- After first night: initial_vampires + (initial_vampires * x)
  -- After second night: (initial_vampires + (initial_vampires * x)) + 
  --                     (initial_vampires + (initial_vampires * x)) * x
  
  (initial_vampires + (initial_vampires * x)) + 
  (initial_vampires + (initial_vampires * x)) * x = final_vampires

theorem vampire_conversion_theorem : vampire_conversion_rate 5 := by
  sorry

end NUMINAMATH_CALUDE_vampire_conversion_theorem_l2689_268980


namespace NUMINAMATH_CALUDE_total_files_deleted_l2689_268958

def initial_files : ℕ := 24
def final_files : ℕ := 21

def deletions : List ℕ := [5, 10]
def additions : List ℕ := [7, 5]

theorem total_files_deleted :
  (initial_files + additions.sum - deletions.sum = final_files) →
  deletions.sum = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_files_deleted_l2689_268958


namespace NUMINAMATH_CALUDE_vector_equality_l2689_268906

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define points in the vector space
variable (A B C M O : V)

-- Define vectors as differences between points
def vec (P Q : V) : V := Q - P

-- State the theorem
theorem vector_equality :
  (vec A B + vec M B) + (vec B O + vec B C) + vec O M = vec A C :=
by sorry

end NUMINAMATH_CALUDE_vector_equality_l2689_268906


namespace NUMINAMATH_CALUDE_natural_number_equation_solutions_l2689_268959

theorem natural_number_equation_solutions :
  ∀ (a b c d : ℕ), 
    a * b = c + d ∧ a + b = c * d →
    ((a = 2 ∧ b = 2 ∧ c = 2 ∧ d = 2) ∨
     (a = 2 ∧ b = 3 ∧ c = 5 ∧ d = 1) ∨
     (a = 3 ∧ b = 2 ∧ c = 5 ∧ d = 1) ∨
     (a = 2 ∧ b = 2 ∧ c = 1 ∧ d = 5) ∧
     (a = 3 ∧ b = 2 ∧ c = 1 ∧ d = 5) ∨
     (a = 2 ∧ b = 3 ∧ c = 1 ∧ d = 5)) :=
by sorry

end NUMINAMATH_CALUDE_natural_number_equation_solutions_l2689_268959


namespace NUMINAMATH_CALUDE_group_size_calculation_l2689_268918

theorem group_size_calculation (n : ℕ) : 
  (n * n = 5929) → n = 77 := by
  sorry

end NUMINAMATH_CALUDE_group_size_calculation_l2689_268918


namespace NUMINAMATH_CALUDE_turtle_count_l2689_268931

/-- Represents the number of turtles in the lake -/
def total_turtles : ℕ := 100

/-- Percentage of female turtles -/
def female_percentage : ℚ := 60 / 100

/-- Percentage of male turtles with stripes -/
def male_striped_percentage : ℚ := 25 / 100

/-- Number of baby male turtles with stripes -/
def baby_striped_males : ℕ := 4

/-- Percentage of adult male turtles with stripes -/
def adult_striped_percentage : ℚ := 60 / 100

theorem turtle_count :
  total_turtles = 100 :=
sorry

end NUMINAMATH_CALUDE_turtle_count_l2689_268931


namespace NUMINAMATH_CALUDE_cubic_equation_root_l2689_268974

theorem cubic_equation_root (c d : ℚ) : 
  (3 + Real.sqrt 5)^3 + c * (3 + Real.sqrt 5)^2 + d * (3 + Real.sqrt 5) + 15 = 0 → 
  d = -37/2 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_root_l2689_268974


namespace NUMINAMATH_CALUDE_iron_bucket_area_l2689_268925

/-- The area of iron sheet needed for a rectangular bucket -/
def bucket_area (length width height : ℝ) : ℝ :=
  length * width + 2 * (length * height + width * height)

/-- Theorem: The area of iron sheet needed for the specified bucket is 1.24 square meters -/
theorem iron_bucket_area :
  let length : ℝ := 0.4
  let width : ℝ := 0.3
  let height : ℝ := 0.8
  bucket_area length width height = 1.24 := by
  sorry


end NUMINAMATH_CALUDE_iron_bucket_area_l2689_268925


namespace NUMINAMATH_CALUDE_base10_to_base5_453_l2689_268994

-- Define a function to convert from base 10 to base 5
def toBase5 (n : ℕ) : List ℕ :=
  sorry

-- Theorem stating that 453 in base 10 is equal to 3303 in base 5
theorem base10_to_base5_453 : toBase5 453 = [3, 3, 0, 3] :=
  sorry

end NUMINAMATH_CALUDE_base10_to_base5_453_l2689_268994


namespace NUMINAMATH_CALUDE_ball_placement_theorem_l2689_268936

/-- Converts a natural number to its base 7 representation --/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

/-- Sums the digits in a list --/
def sumDigits (digits : List ℕ) : ℕ :=
  sorry

/-- Represents the ball placement process up to n steps --/
def ballPlacement (n : ℕ) : ℕ :=
  sorry

theorem ball_placement_theorem :
  ballPlacement 1729 = sumDigits (toBase7 1729) :=
sorry

end NUMINAMATH_CALUDE_ball_placement_theorem_l2689_268936


namespace NUMINAMATH_CALUDE_number_count_proof_l2689_268968

/-- Given a set of numbers with specific average properties, prove that the total count is 8 -/
theorem number_count_proof (n : ℕ) (S : ℝ) (S₅ : ℝ) (S₃ : ℝ) : 
  S / n = 20 →
  S₅ / 5 = 12 →
  S₃ / 3 = 33.333333333333336 →
  S = S₅ + S₃ →
  n = 8 := by
sorry

end NUMINAMATH_CALUDE_number_count_proof_l2689_268968


namespace NUMINAMATH_CALUDE_sequence_property_l2689_268989

def arithmetic_sequence (a b c : ℝ) : Prop :=
  b - a = c - b ∧ a > 0 ∧ b > 0 ∧ c > 0

def geometric_sequence (a b c : ℝ) : Prop :=
  b / a = c / b ∧ a ≠ 0 ∧ b ≠ 0

def general_term (n : ℕ) : ℝ := 2^(n - 1)

theorem sequence_property :
  ∀ a b c : ℝ,
    arithmetic_sequence a b c →
    a + b + c = 6 →
    geometric_sequence (a + 3) (b + 6) (c + 13) →
    (∀ n : ℕ, n ≥ 3 → general_term n = (a + 3) * 2^(n - 3)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_property_l2689_268989


namespace NUMINAMATH_CALUDE_total_score_is_210_l2689_268946

/-- Represents the test scores of three students -/
structure TestScores where
  total_questions : ℕ
  marks_per_question : ℕ
  jose_wrong_questions : ℕ
  meghan_diff : ℕ
  jose_alisson_diff : ℕ

/-- Calculates the total score for three students given their test performance -/
def calculate_total_score (scores : TestScores) : ℕ :=
  let total_marks := scores.total_questions * scores.marks_per_question
  let jose_score := total_marks - (scores.jose_wrong_questions * scores.marks_per_question)
  let meghan_score := jose_score - scores.meghan_diff
  let alisson_score := jose_score - scores.jose_alisson_diff
  jose_score + meghan_score + alisson_score

/-- Theorem stating that the total score for the three students is 210 marks -/
theorem total_score_is_210 (scores : TestScores) 
  (h1 : scores.total_questions = 50)
  (h2 : scores.marks_per_question = 2)
  (h3 : scores.jose_wrong_questions = 5)
  (h4 : scores.meghan_diff = 20)
  (h5 : scores.jose_alisson_diff = 40) :
  calculate_total_score scores = 210 := by
  sorry

end NUMINAMATH_CALUDE_total_score_is_210_l2689_268946


namespace NUMINAMATH_CALUDE_salary_change_percentage_l2689_268945

theorem salary_change_percentage (x : ℝ) : 
  (1 - x / 100) * (1 + x / 100) = 64 / 100 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_salary_change_percentage_l2689_268945


namespace NUMINAMATH_CALUDE_exists_multiple_with_digit_sum_l2689_268939

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: There exists a natural number that is a multiple of 2015 and whose sum of digits equals 2015 -/
theorem exists_multiple_with_digit_sum :
  ∃ (n : ℕ), (n % 2015 = 0) ∧ (sum_of_digits n = 2015) := by sorry

end NUMINAMATH_CALUDE_exists_multiple_with_digit_sum_l2689_268939


namespace NUMINAMATH_CALUDE_jared_yearly_income_l2689_268987

/-- Calculates the yearly income of a degree holder after one year of employment --/
def yearly_income_after_one_year (diploma_salary : ℝ) : ℝ :=
  let degree_salary := 3 * diploma_salary
  let annual_salary := 12 * degree_salary
  let salary_after_raise := annual_salary * 1.05
  salary_after_raise * 1.05

/-- Theorem stating that Jared's yearly income after one year is $158760 --/
theorem jared_yearly_income :
  yearly_income_after_one_year 4000 = 158760 := by
  sorry

#eval yearly_income_after_one_year 4000

end NUMINAMATH_CALUDE_jared_yearly_income_l2689_268987


namespace NUMINAMATH_CALUDE_change_in_expression_l2689_268935

theorem change_in_expression (x b : ℝ) (h : b > 0) :
  let f := fun t : ℝ => t^2 - 5*t + 2
  f (x + b) - f x = 2*b*x + b^2 - 5*b :=
by sorry

end NUMINAMATH_CALUDE_change_in_expression_l2689_268935


namespace NUMINAMATH_CALUDE_linear_expr_pythagorean_relation_l2689_268993

-- Define linear expressions
def LinearExpr (α : Type*) [Ring α] := α → α

-- Theorem statement
theorem linear_expr_pythagorean_relation
  {α : Type*} [Field α]
  (A B C : LinearExpr α)
  (h : ∀ x, (A x)^2 + (B x)^2 = (C x)^2) :
  ∃ (k₁ k₂ : α), ∀ x, A x = k₁ * (C x) ∧ B x = k₂ * (C x) := by
  sorry

end NUMINAMATH_CALUDE_linear_expr_pythagorean_relation_l2689_268993


namespace NUMINAMATH_CALUDE_peace_numbers_examples_l2689_268973

/-- Two numbers are peace numbers about 3 if their sum is 3 -/
def PeaceNumbersAbout3 (a b : ℝ) : Prop := a + b = 3

theorem peace_numbers_examples :
  (PeaceNumbersAbout3 4 (-1)) ∧
  (∀ x : ℝ, PeaceNumbersAbout3 (8 - x) (-5 + x)) ∧
  (∀ x : ℝ, PeaceNumbersAbout3 (x^2 - 4*x - 1) (x^2 - 2*(x^2 - 2*x - 2))) ∧
  (∀ k : ℕ, (∃ x : ℕ, x > 0 ∧ PeaceNumbersAbout3 (k * x + 1) (x - 2)) ↔ (k = 1 ∨ k = 3)) :=
by sorry

end NUMINAMATH_CALUDE_peace_numbers_examples_l2689_268973


namespace NUMINAMATH_CALUDE_kat_weekly_training_hours_l2689_268962

/-- Represents Kat's weekly training schedule --/
structure TrainingSchedule where
  strength_sessions : ℕ
  strength_hours_per_session : ℝ
  boxing_sessions : ℕ
  boxing_hours_per_session : ℝ
  cardio_sessions : ℕ
  cardio_hours_per_session : ℝ
  flexibility_sessions : ℕ
  flexibility_hours_per_session : ℝ
  interval_sessions : ℕ
  interval_hours_per_session : ℝ

/-- Calculates the total weekly training hours --/
def total_weekly_hours (schedule : TrainingSchedule) : ℝ :=
  schedule.strength_sessions * schedule.strength_hours_per_session +
  schedule.boxing_sessions * schedule.boxing_hours_per_session +
  schedule.cardio_sessions * schedule.cardio_hours_per_session +
  schedule.flexibility_sessions * schedule.flexibility_hours_per_session +
  schedule.interval_sessions * schedule.interval_hours_per_session

/-- Kat's actual training schedule --/
def kat_schedule : TrainingSchedule := {
  strength_sessions := 3
  strength_hours_per_session := 1
  boxing_sessions := 4
  boxing_hours_per_session := 1.5
  cardio_sessions := 2
  cardio_hours_per_session := 0.5
  flexibility_sessions := 1
  flexibility_hours_per_session := 0.75
  interval_sessions := 1
  interval_hours_per_session := 1.25
}

/-- Theorem stating that Kat's total weekly training time is 12 hours --/
theorem kat_weekly_training_hours :
  total_weekly_hours kat_schedule = 12 := by sorry

end NUMINAMATH_CALUDE_kat_weekly_training_hours_l2689_268962


namespace NUMINAMATH_CALUDE_odd_function_zero_condition_l2689_268990

-- Define a real-valued function
def RealFunction := ℝ → ℝ

-- Define what it means for a function to be odd
def IsOdd (f : RealFunction) : Prop := ∀ x : ℝ, f (-x) = -f x

-- State the theorem
theorem odd_function_zero_condition :
  (∀ f : RealFunction, IsOdd f → f 0 = 0) ∧
  (∃ f : RealFunction, f 0 = 0 ∧ ¬IsOdd f) :=
sorry

end NUMINAMATH_CALUDE_odd_function_zero_condition_l2689_268990


namespace NUMINAMATH_CALUDE_x_squared_mod_24_l2689_268976

theorem x_squared_mod_24 (x : ℤ) 
  (h1 : 6 * x ≡ 12 [ZMOD 24])
  (h2 : 4 * x ≡ 20 [ZMOD 24]) : 
  x^2 ≡ 12 [ZMOD 24] := by
sorry

end NUMINAMATH_CALUDE_x_squared_mod_24_l2689_268976


namespace NUMINAMATH_CALUDE_concrete_density_l2689_268975

/-- Concrete density problem -/
theorem concrete_density (num_homes : ℕ) (length width height : ℝ) (cost_per_pound : ℝ) (total_cost : ℝ)
  (h1 : num_homes = 3)
  (h2 : length = 100)
  (h3 : width = 100)
  (h4 : height = 0.5)
  (h5 : cost_per_pound = 0.02)
  (h6 : total_cost = 45000) :
  (total_cost / cost_per_pound) / (num_homes * length * width * height) = 150 := by
  sorry

end NUMINAMATH_CALUDE_concrete_density_l2689_268975


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_tangent_l2689_268957

/-- An ellipse and a hyperbola are tangent if and only if m = 8/9 -/
theorem ellipse_hyperbola_tangent (m : ℝ) : 
  (∃ x y : ℝ, x^2 + 9*y^2 = 9 ∧ x^2 - m*(y+3)^2 = 1 ∧ 
   ∀ x' y' : ℝ, x'^2 + 9*y'^2 = 9 ∧ x'^2 - m*(y'+3)^2 = 1 → (x', y') = (x, y)) ↔ 
  m = 8/9 := by
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_tangent_l2689_268957


namespace NUMINAMATH_CALUDE_fraction_problem_l2689_268917

theorem fraction_problem (m n p q : ℚ) 
  (h1 : m / n = 20)
  (h2 : p / n = 5)
  (h3 : p / q = 1 / 15) :
  m / q = 4 / 15 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l2689_268917


namespace NUMINAMATH_CALUDE_rectangular_solid_volume_l2689_268920

theorem rectangular_solid_volume (a b c : ℝ) (h1 : a * b = 18) (h2 : b * c = 50) (h3 : a * c = 45) :
  a * b * c = 150 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_volume_l2689_268920


namespace NUMINAMATH_CALUDE_borrowing_interest_rate_l2689_268951

/-- Proves that the interest rate at which a person borrowed money is 4% per annum
    given the specified conditions. -/
theorem borrowing_interest_rate : 
  ∀ (principal : ℝ) (borrowing_time : ℝ) (lending_rate : ℝ) (lending_time : ℝ) (yearly_gain : ℝ),
  principal = 5000 →
  borrowing_time = 2 →
  lending_rate = 0.06 →
  lending_time = 2 →
  yearly_gain = 100 →
  ∃ (borrowing_rate : ℝ),
    borrowing_rate = 0.04 ∧
    principal * lending_rate * lending_time - 
    principal * borrowing_rate * borrowing_time = 
    yearly_gain * borrowing_time :=
by sorry

end NUMINAMATH_CALUDE_borrowing_interest_rate_l2689_268951


namespace NUMINAMATH_CALUDE_profit_maximized_at_optimal_reduction_optimal_reduction_is_five_profit_function_correct_l2689_268940

/-- Profit function for a product with given initial conditions -/
def profit_function (x : ℝ) : ℝ := -x^2 + 10*x + 600

/-- The price reduction that maximizes profit -/
def optimal_reduction : ℝ := 5

theorem profit_maximized_at_optimal_reduction :
  ∀ x : ℝ, profit_function x ≤ profit_function optimal_reduction :=
sorry

theorem optimal_reduction_is_five :
  optimal_reduction = 5 :=
sorry

theorem profit_function_correct (x : ℝ) :
  profit_function x = (100 - 70 - x) * (20 + x) :=
sorry

end NUMINAMATH_CALUDE_profit_maximized_at_optimal_reduction_optimal_reduction_is_five_profit_function_correct_l2689_268940


namespace NUMINAMATH_CALUDE_math_paths_count_l2689_268919

/-- Represents the number of boundary "M"s in the grid -/
def boundary_Ms : ℕ := 4

/-- Represents the number of possible moves from one letter to the next -/
def moves_per_step : ℕ := 2

/-- Represents the number of steps in the word "MATH" -/
def word_length : ℕ := 3

/-- Calculates the number of paths for a single starting "M" -/
def paths_per_M : ℕ := moves_per_step ^ word_length

/-- Theorem stating that the total number of paths spelling "MATH" is 32 -/
theorem math_paths_count : boundary_Ms * paths_per_M = 32 := by
  sorry

end NUMINAMATH_CALUDE_math_paths_count_l2689_268919


namespace NUMINAMATH_CALUDE_pet_shop_legs_l2689_268902

/-- The total number of legs in a pet shop with birds, dogs, snakes, and spiders -/
def total_legs (num_birds num_dogs num_snakes num_spiders : ℕ) 
               (bird_legs dog_legs snake_legs spider_legs : ℕ) : ℕ :=
  num_birds * bird_legs + num_dogs * dog_legs + num_snakes * snake_legs + num_spiders * spider_legs

/-- Theorem stating that the total number of legs in the given pet shop scenario is 34 -/
theorem pet_shop_legs : 
  total_legs 3 5 4 1 2 4 0 8 = 34 := by
  sorry

end NUMINAMATH_CALUDE_pet_shop_legs_l2689_268902


namespace NUMINAMATH_CALUDE_exp_sum_greater_than_two_l2689_268963

theorem exp_sum_greater_than_two (a b : ℝ) (h1 : a ≠ b) (h2 : a * Real.exp b - b * Real.exp a = Real.exp a - Real.exp b) : 
  Real.exp a + Real.exp b > 2 := by
  sorry

end NUMINAMATH_CALUDE_exp_sum_greater_than_two_l2689_268963


namespace NUMINAMATH_CALUDE_stock_exchange_problem_l2689_268916

theorem stock_exchange_problem (total_stocks : ℕ) 
  (h_total : total_stocks = 1980) 
  (H L : ℕ) 
  (h_relation : H = L + L / 5) 
  (h_sum : H + L = total_stocks) : 
  H = 1080 := by
sorry

end NUMINAMATH_CALUDE_stock_exchange_problem_l2689_268916


namespace NUMINAMATH_CALUDE_polynomial_composition_l2689_268998

theorem polynomial_composition (f g : ℝ → ℝ) :
  (∀ x, f x = x^2) →
  (∃ a b c : ℝ, ∀ x, g x = a * x^2 + b * x + c) →
  (∀ x, f (g x) = 9 * x^2 - 6 * x + 1) →
  (∀ x, g x = 3 * x - 1) ∨ (∀ x, g x = -3 * x + 1) := by
sorry

end NUMINAMATH_CALUDE_polynomial_composition_l2689_268998


namespace NUMINAMATH_CALUDE_complex_magnitude_squared_l2689_268941

theorem complex_magnitude_squared (z : ℂ) (h : z + Complex.abs z = 6 + 10 * Complex.I) : 
  Complex.abs z ^ 2 = 1156 / 9 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_squared_l2689_268941


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l2689_268921

theorem triangle_angle_measure (a b c : ℝ) (A B C : ℝ) :
  (0 < a) → (0 < b) → (0 < c) →
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  (a * Real.cos B - b * Real.cos A = c) →
  (C = π / 5) →
  (B = 3 * π / 10) := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l2689_268921


namespace NUMINAMATH_CALUDE_tan_2x_geq_1_solution_set_l2689_268914

theorem tan_2x_geq_1_solution_set :
  {x : ℝ | Real.tan (2 * x) ≥ 1} = {x : ℝ | ∃ k : ℤ, k * Real.pi / 2 + Real.pi / 8 ≤ x ∧ x < k * Real.pi / 2 + Real.pi / 4} :=
by sorry

end NUMINAMATH_CALUDE_tan_2x_geq_1_solution_set_l2689_268914


namespace NUMINAMATH_CALUDE_prime_sequence_recurrence_relation_l2689_268949

theorem prime_sequence_recurrence_relation 
  (p : ℕ → ℕ) 
  (k : ℤ) 
  (h_prime : ∀ n, Nat.Prime (p n)) 
  (h_recurrence : ∀ n, p (n + 2) = p (n + 1) + p n + k) : 
  (∃ (prime : ℕ) (h_prime : Nat.Prime prime), 
    (∀ n, p n = prime) ∧ k = -prime) := by
  sorry

end NUMINAMATH_CALUDE_prime_sequence_recurrence_relation_l2689_268949


namespace NUMINAMATH_CALUDE_gerald_expense_l2689_268903

/-- Represents Gerald's baseball supplies expense situation -/
structure BaseballExpenses where
  season_length : ℕ
  saving_months : ℕ
  chore_price : ℕ
  chores_per_month : ℕ

/-- Calculates the monthly expense for baseball supplies -/
def monthly_expense (e : BaseballExpenses) : ℕ :=
  (e.saving_months * e.chores_per_month * e.chore_price) / e.season_length

/-- Theorem: Given Gerald's specific situation, his monthly expense is $100 -/
theorem gerald_expense :
  let e : BaseballExpenses := {
    season_length := 4,
    saving_months := 8,
    chore_price := 10,
    chores_per_month := 5
  }
  monthly_expense e = 100 := by sorry

end NUMINAMATH_CALUDE_gerald_expense_l2689_268903


namespace NUMINAMATH_CALUDE_delivery_speed_l2689_268953

/-- Given the conditions of the delivery problem, prove that the required average speed is 30 km/h -/
theorem delivery_speed (d : ℝ) (t : ℝ) (v : ℝ) : 
  (d / 60 = t - 1/4) →  -- Condition for moderate traffic
  (d / 20 = t + 1/4) →  -- Condition for traffic jams
  (d / v = 1/2) →       -- Condition for arriving exactly at 18:00
  v = 30 := by
  sorry

end NUMINAMATH_CALUDE_delivery_speed_l2689_268953


namespace NUMINAMATH_CALUDE_alligator_journey_time_l2689_268926

/-- The additional time taken for the return journey of alligators -/
def additional_time (initial_time : ℕ) (total_alligators : ℕ) (total_time : ℕ) : ℕ :=
  (total_time - initial_time) / total_alligators - initial_time

/-- Theorem stating that the additional time for the return journey is 2 hours -/
theorem alligator_journey_time : additional_time 4 7 46 = 2 := by
  sorry

end NUMINAMATH_CALUDE_alligator_journey_time_l2689_268926


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l2689_268971

theorem sqrt_expression_equality : 3 * Real.sqrt 12 / (3 * Real.sqrt (1/3)) - 2 * Real.sqrt 3 = 6 - 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l2689_268971


namespace NUMINAMATH_CALUDE_wire_length_proof_l2689_268924

theorem wire_length_proof (side_length : ℝ) (total_area : ℝ) (original_length : ℝ) : 
  side_length = 2 →
  total_area = 92 →
  original_length = (total_area / (side_length ^ 2)) * (4 * side_length) →
  original_length = 184 := by
  sorry

#check wire_length_proof

end NUMINAMATH_CALUDE_wire_length_proof_l2689_268924


namespace NUMINAMATH_CALUDE_function_properties_l2689_268997

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The domain of a function -/
def Domain (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → f x ≠ 0

/-- The quadratic function we're considering -/
def f (m n : ℝ) (x : ℝ) : ℝ :=
  m * x^2 + n * x + 3 * m + n

/-- The theorem stating the properties of the function and its maximum value -/
theorem function_properties :
  ∃ (m n : ℝ),
    EvenFunction (f m n) ∧
    Domain (f m n) (m - 1) (2 * m) ∧
    m = 1/3 ∧
    n = 0 ∧
    (∀ x, m - 1 ≤ x ∧ x ≤ 2 * m → f m n x ≤ 31/27) :=
sorry

end NUMINAMATH_CALUDE_function_properties_l2689_268997


namespace NUMINAMATH_CALUDE_alex_amount_l2689_268907

def total : ℚ := 972.45
def sam : ℚ := 325.67
def erica : ℚ := 214.29

theorem alex_amount : total - (sam + erica) = 432.49 := by
  sorry

end NUMINAMATH_CALUDE_alex_amount_l2689_268907


namespace NUMINAMATH_CALUDE_g_zeros_count_l2689_268910

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (x + a)

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a (x - a) - x^2

theorem g_zeros_count (a : ℝ) :
  (∀ x, g a x ≠ 0) ∨
  (∃! x, g a x = 0) ∨
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ g a x₁ = 0 ∧ g a x₂ = 0 ∧ ∀ x, g a x = 0 → x = x₁ ∨ x = x₂) ∨
  (∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ g a x₁ = 0 ∧ g a x₂ = 0 ∧ g a x₃ = 0 ∧
    ∀ x, g a x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃) :=
by sorry

end NUMINAMATH_CALUDE_g_zeros_count_l2689_268910


namespace NUMINAMATH_CALUDE_jane_reading_period_l2689_268908

/-- Represents Jane's reading habits and total pages read --/
structure ReadingHabit where
  morning_pages : ℕ
  evening_pages : ℕ
  total_pages : ℕ

/-- Calculates the number of days Jane reads based on her reading habit --/
def calculate_reading_days (habit : ReadingHabit) : ℚ :=
  habit.total_pages / (habit.morning_pages + habit.evening_pages)

/-- Theorem stating that Jane reads for 7 days --/
theorem jane_reading_period (habit : ReadingHabit) 
  (h1 : habit.morning_pages = 5)
  (h2 : habit.evening_pages = 10)
  (h3 : habit.total_pages = 105) :
  calculate_reading_days habit = 7 := by
  sorry


end NUMINAMATH_CALUDE_jane_reading_period_l2689_268908


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2689_268996

theorem geometric_sequence_sum (a r : ℝ) (h1 : a + a*r + a*r^2 = 13) (h2 : a * (1 - r^7) / (1 - r) = 183) : 
  ∃ (ε : ℝ), abs (a + a*r + a*r^2 + a*r^3 + a*r^4 - 75.764) < ε ∧ ε > 0 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2689_268996


namespace NUMINAMATH_CALUDE_function_properties_l2689_268923

-- Define the function f(x)
noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  Real.sqrt 3 * Real.sin x * Real.cos x + Real.cos x ^ 2 + a

-- Define the theorem
theorem function_properties (a : ℝ) :
  -- Condition: x ∈ [-π/6, π/3]
  (∀ x, -π/6 ≤ x ∧ x ≤ π/3 →
    -- Condition: sum of max and min values is 3/2
    (⨆ x, f x a) + (⨅ x, f x a) = 3/2) →
  -- 1. Smallest positive period is π
  (∀ x, f (x + π) a = f x a) ∧
  (∀ T, T > 0 ∧ (∀ x, f (x + T) a = f x a) → T ≥ π) ∧
  -- 2. Interval of monotonic decrease
  (∀ k : ℤ, ∀ x y, k * π + π/6 ≤ x ∧ x ≤ y ∧ y ≤ k * π + 2*π/3 →
    f y a ≤ f x a) ∧
  -- 3. Solution set of f(x) > 1
  (∀ x, 0 < x ∧ x < π/3 ↔ f x a > 1) :=
sorry

end NUMINAMATH_CALUDE_function_properties_l2689_268923


namespace NUMINAMATH_CALUDE_solution_set_inequalities_l2689_268991

theorem solution_set_inequalities :
  {x : ℝ | x - 2 > 1 ∧ x < 4} = {x : ℝ | 3 < x ∧ x < 4} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequalities_l2689_268991


namespace NUMINAMATH_CALUDE_inequality_proof_l2689_268938

theorem inequality_proof (x y : ℝ) : 
  |((x + y) * (1 - x * y)) / ((1 + x^2) * (1 + y^2))| ≤ (1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2689_268938


namespace NUMINAMATH_CALUDE_largest_band_size_l2689_268982

/-- Represents a rectangular band formation --/
structure BandFormation where
  rows : ℕ
  membersPerRow : ℕ

/-- The total number of members in a formation --/
def totalMembers (f : BandFormation) : ℕ := f.rows * f.membersPerRow

/-- The condition that the band has less than 150 members --/
def lessThan150 (f : BandFormation) : Prop := totalMembers f < 150

/-- The condition that there are 3 members left over in the original formation --/
def hasThreeLeftOver (f : BandFormation) (totalBandMembers : ℕ) : Prop :=
  totalMembers f + 3 = totalBandMembers

/-- The new formation with 2 more members per row and 3 fewer rows --/
def newFormation (f : BandFormation) : BandFormation :=
  { rows := f.rows - 3, membersPerRow := f.membersPerRow + 2 }

/-- The condition that the new formation fits all members exactly --/
def newFormationFitsExactly (f : BandFormation) (totalBandMembers : ℕ) : Prop :=
  totalMembers (newFormation f) = totalBandMembers

/-- The theorem stating that the largest possible number of band members is 108 --/
theorem largest_band_size :
  ∃ (f : BandFormation) (totalBandMembers : ℕ),
    lessThan150 f ∧
    hasThreeLeftOver f totalBandMembers ∧
    newFormationFitsExactly f totalBandMembers ∧
    totalBandMembers = 108 ∧
    (∀ (g : BandFormation) (m : ℕ),
      lessThan150 g →
      hasThreeLeftOver g m →
      newFormationFitsExactly g m →
      m ≤ 108) :=
  sorry


end NUMINAMATH_CALUDE_largest_band_size_l2689_268982


namespace NUMINAMATH_CALUDE_tan_value_for_given_conditions_l2689_268983

theorem tan_value_for_given_conditions (α : Real) 
  (h1 : α ∈ Set.Ioo 0 (Real.pi / 2))
  (h2 : Real.sin α ^ 2 + Real.cos (2 * α) = 3 / 4) : 
  Real.tan α = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_value_for_given_conditions_l2689_268983


namespace NUMINAMATH_CALUDE_pencil_distribution_l2689_268911

theorem pencil_distribution (total : ℕ) (h1 : total = 8 * 6 + 4) : 
  total / 4 = 13 := by
sorry

end NUMINAMATH_CALUDE_pencil_distribution_l2689_268911


namespace NUMINAMATH_CALUDE_sum_distinct_remainders_divided_by_13_l2689_268930

def distinct_remainders (n : ℕ) : Finset ℕ :=
  (Finset.range n).image (λ i => (i + 1)^2 % 13)

theorem sum_distinct_remainders_divided_by_13 :
  (Finset.sum (distinct_remainders 12) id) / 13 = 3 :=
sorry

end NUMINAMATH_CALUDE_sum_distinct_remainders_divided_by_13_l2689_268930


namespace NUMINAMATH_CALUDE_complement_not_always_greater_l2689_268981

def complement (θ : ℝ) : ℝ := 90 - θ

theorem complement_not_always_greater : ∃ θ : ℝ, complement θ ≤ θ := by
  sorry

end NUMINAMATH_CALUDE_complement_not_always_greater_l2689_268981


namespace NUMINAMATH_CALUDE_fraction_value_l2689_268915

theorem fraction_value (x y : ℝ) (h1 : 2 < (x - y) / (x + y)) 
  (h2 : (x - y) / (x + y) < 5) (h3 : ∃ (n : ℤ), x / y = n) : x / y = -2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l2689_268915


namespace NUMINAMATH_CALUDE_min_value_problem_l2689_268900

theorem min_value_problem (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (h_sum : x^2 + y^2 = 4) :
  ∃ m : ℝ, m = -8 * Real.sqrt 2 ∧ ∀ z : ℝ, z = x * y - 4 * (x + y) - 2 → z ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l2689_268900


namespace NUMINAMATH_CALUDE_nested_sqrt_value_l2689_268928

theorem nested_sqrt_value : 
  ∃ x : ℝ, x = Real.sqrt (3 - x) ∧ x = (-1 + Real.sqrt 13) / 2 := by
sorry

end NUMINAMATH_CALUDE_nested_sqrt_value_l2689_268928


namespace NUMINAMATH_CALUDE_sixteen_seats_painting_ways_l2689_268912

def paintingWays (n : ℕ) : ℕ := 
  let rec a : ℕ → ℕ
    | 0 => 1
    | 1 => 1
    | i + 1 => (List.range ((i + 1) / 2 + 1)).foldl (λ sum j => sum + a (i - 2 * j)) 0
  2 * a n

theorem sixteen_seats_painting_ways :
  paintingWays 16 = 1686 := by sorry

end NUMINAMATH_CALUDE_sixteen_seats_painting_ways_l2689_268912


namespace NUMINAMATH_CALUDE_equal_remainders_theorem_l2689_268954

theorem equal_remainders_theorem (p : ℕ) (x : ℕ) (h_prime : Nat.Prime p) (h_pos : x > 0) :
  (∃ r : ℕ, x % p = r ∧ p^2 % x = r) →
  ((x = p ∧ p % x = 0) ∨ (x = p^2 ∧ p^2 % x = 0) ∨ (x = p + 1 ∧ p^2 % x = 1)) :=
sorry

end NUMINAMATH_CALUDE_equal_remainders_theorem_l2689_268954


namespace NUMINAMATH_CALUDE_even_odd_solution_l2689_268986

theorem even_odd_solution (m n p q : ℤ) 
  (h_m_odd : Odd m)
  (h_n_even : Even n)
  (h_eq1 : p - 1998*q = n)
  (h_eq2 : 1999*p + 3*q = m) :
  Even p ∧ Odd q := by
sorry

end NUMINAMATH_CALUDE_even_odd_solution_l2689_268986


namespace NUMINAMATH_CALUDE_birthday_candles_sharing_l2689_268905

theorem birthday_candles_sharing (ambika_candles : ℕ) (aniyah_multiplier : ℕ) : 
  ambika_candles = 4 →
  aniyah_multiplier = 6 →
  ((ambika_candles + aniyah_multiplier * ambika_candles) / 2 : ℕ) = 14 :=
by sorry

end NUMINAMATH_CALUDE_birthday_candles_sharing_l2689_268905


namespace NUMINAMATH_CALUDE_absolute_value_sum_zero_implies_value_l2689_268937

theorem absolute_value_sum_zero_implies_value (x y : ℝ) :
  |x - 4| + |5 + y| = 0 → 2*x + 3*y = -7 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sum_zero_implies_value_l2689_268937


namespace NUMINAMATH_CALUDE_smallest_k_for_zero_difference_l2689_268927

def u (n : ℕ) := n^4 + n^2 + n

def Δ : (ℕ → ℕ) → (ℕ → ℕ)
  | f => fun n => f (n + 1) - f n

def iteratedΔ : ℕ → (ℕ → ℕ) → (ℕ → ℕ)
  | 0 => id
  | k + 1 => Δ ∘ iteratedΔ k

theorem smallest_k_for_zero_difference :
  ∃ k, k = 5 ∧ 
    (∀ n, iteratedΔ k u n = 0) ∧
    (∀ j < k, ∃ n, iteratedΔ j u n ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_smallest_k_for_zero_difference_l2689_268927


namespace NUMINAMATH_CALUDE_wage_difference_l2689_268942

/-- The total pay for the research project -/
def total_pay : ℝ := 360

/-- Candidate P's hourly wage -/
def wage_p : ℝ := 18

/-- Candidate Q's hourly wage -/
def wage_q : ℝ := 12

/-- The number of hours candidate P needs to complete the job -/
def hours_p : ℝ := 20

/-- The number of hours candidate Q needs to complete the job -/
def hours_q : ℝ := 30

theorem wage_difference : 
  (wage_p = 1.5 * wage_q) ∧ 
  (hours_q = hours_p + 10) ∧ 
  (wage_p * hours_p = total_pay) ∧ 
  (wage_q * hours_q = total_pay) → 
  wage_p - wage_q = 6 := by
  sorry

end NUMINAMATH_CALUDE_wage_difference_l2689_268942


namespace NUMINAMATH_CALUDE_non_positive_sequence_l2689_268992

theorem non_positive_sequence (n : ℕ) (a : ℕ → ℝ) 
  (h0 : a 0 = 0) 
  (hn : a n = 0) 
  (h_ineq : ∀ k : ℕ, k ∈ Finset.range (n - 1) → a k - 2 * a (k + 1) + a (k + 2) ≥ 0) :
  ∀ i : ℕ, i ≤ n → a i ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_non_positive_sequence_l2689_268992


namespace NUMINAMATH_CALUDE_chord_convex_quadrilateral_probability_l2689_268999

/-- Given six points on a circle, the probability that four randomly chosen chords
    form a convex quadrilateral is 1/91. -/
theorem chord_convex_quadrilateral_probability (n : ℕ) (h : n = 6) :
  (Nat.choose n 4 : ℚ) / (Nat.choose (Nat.choose n 2) 4) = 1 / 91 :=
sorry

end NUMINAMATH_CALUDE_chord_convex_quadrilateral_probability_l2689_268999


namespace NUMINAMATH_CALUDE_annual_interest_income_l2689_268909

/-- Calculates the annual interest income from two municipal bonds -/
theorem annual_interest_income
  (total_investment : ℝ)
  (rate1 rate2 : ℝ)
  (investment1 : ℝ)
  (h1 : total_investment = 32000)
  (h2 : rate1 = 0.0575)
  (h3 : rate2 = 0.0625)
  (h4 : investment1 = 20000)
  (h5 : investment1 < total_investment) :
  investment1 * rate1 + (total_investment - investment1) * rate2 = 1900 := by
  sorry

end NUMINAMATH_CALUDE_annual_interest_income_l2689_268909


namespace NUMINAMATH_CALUDE_equation_solution_l2689_268904

theorem equation_solution : ∃! x : ℝ, (128 : ℝ)^(x - 1) / (16 : ℝ)^(x - 1) = (64 : ℝ)^(3 * x) ∧ x = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2689_268904


namespace NUMINAMATH_CALUDE_z_value_theorem_l2689_268961

theorem z_value_theorem (x y z k : ℝ) (h : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ k ≠ 0) 
  (eq : 1/x - 1/y = k * 1/z) : z = x*y / (k*(y-x)) := by
  sorry

end NUMINAMATH_CALUDE_z_value_theorem_l2689_268961


namespace NUMINAMATH_CALUDE_find_x_l2689_268988

theorem find_x (y : ℝ) (x : ℝ) (h1 : (12 : ℝ)^3 * 6^3 / x = y) (h2 : y = 864) : x = 432 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l2689_268988


namespace NUMINAMATH_CALUDE_half_AB_equals_2_1_l2689_268979

def MA : ℝ × ℝ := (-2, 4)
def MB : ℝ × ℝ := (2, 6)

theorem half_AB_equals_2_1 : (1 / 2 : ℝ) • (MB - MA) = (2, 1) := by sorry

end NUMINAMATH_CALUDE_half_AB_equals_2_1_l2689_268979


namespace NUMINAMATH_CALUDE_symmetric_point_wrt_origin_l2689_268977

/-- Given a point (2, -2), its symmetric point with respect to the origin has coordinates (-2, 2) -/
theorem symmetric_point_wrt_origin :
  let original_point : ℝ × ℝ := (2, -2)
  let symmetric_point : ℝ × ℝ := (-2, 2)
  (∀ (x y : ℝ), (x, y) = original_point → (-x, -y) = symmetric_point) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_point_wrt_origin_l2689_268977


namespace NUMINAMATH_CALUDE_monotonic_increasing_sine_cosine_function_l2689_268956

theorem monotonic_increasing_sine_cosine_function (a : ℝ) :
  (∀ x ∈ Set.Icc 0 (π / 4), Monotone (fun x => a * Real.sin x + Real.cos x)) ↔ a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_increasing_sine_cosine_function_l2689_268956


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l2689_268965

theorem complex_expression_simplification (x y : ℝ) :
  let i : ℂ := Complex.I
  (x^2 + i*y)^3 * (x^2 - i*y)^3 = x^12 - 9*x^8*y^2 - 9*x^4*y^4 - y^6 :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l2689_268965


namespace NUMINAMATH_CALUDE_m_range_l2689_268950

def y₁ (m x : ℝ) : ℝ := m * (x - 2 * m) * (x + m + 2)
def y₂ (x : ℝ) : ℝ := x - 1

theorem m_range :
  (∀ m : ℝ,
    (∀ x : ℝ, y₁ m x < 0 ∨ y₂ x < 0) ∧
    (∃ x : ℝ, x < -3 ∧ y₁ m x * y₂ x < 0)) ↔
  (∀ m : ℝ, -4 < m ∧ m < -3/2) :=
sorry

end NUMINAMATH_CALUDE_m_range_l2689_268950


namespace NUMINAMATH_CALUDE_expression_evaluation_l2689_268922

theorem expression_evaluation (x : ℝ) (h1 : x^5 + 1 ≠ 0) (h2 : x^5 - 1 ≠ 0) :
  let expr := (((x^2 - 2*x + 2)^2 * (x^3 - x^2 + 1)^2) / (x^5 + 1)^2)^2 *
               (((x^2 + 2*x + 2)^2 * (x^3 + x^2 + 1)^2) / (x^5 - 1)^2)^2
  expr = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2689_268922


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l2689_268943

theorem arithmetic_mean_of_fractions (x a : ℝ) (hx : x ≠ 0) (hxa : x^2 ≠ a) :
  ((x^2 + a) / x^2 + (x^2 - a) / x^2) / 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l2689_268943


namespace NUMINAMATH_CALUDE_painting_price_increase_l2689_268901

theorem painting_price_increase (x : ℝ) : 
  (1 + x / 100) * (1 - 15 / 100) = 93.5 / 100 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_painting_price_increase_l2689_268901


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2689_268978

theorem quadratic_inequality_solution (x : ℝ) :
  3 * x^2 - 5 * x - 2 < 0 ↔ -1/3 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2689_268978


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2689_268933

theorem inequality_solution_set (x : ℝ) : 1 - 3 * (x - 1) < x ↔ x > 1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2689_268933


namespace NUMINAMATH_CALUDE_x_equals_y_squared_plus_two_y_minus_one_l2689_268967

theorem x_equals_y_squared_plus_two_y_minus_one (x y : ℝ) :
  x / (x - 1) = (y^2 + 2*y - 1) / (y^2 + 2*y - 2) → x = y^2 + 2*y - 1 := by
  sorry

end NUMINAMATH_CALUDE_x_equals_y_squared_plus_two_y_minus_one_l2689_268967


namespace NUMINAMATH_CALUDE_sin_four_thirds_pi_l2689_268972

theorem sin_four_thirds_pi : Real.sin (4 / 3 * Real.pi) = -(Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_sin_four_thirds_pi_l2689_268972


namespace NUMINAMATH_CALUDE_pizza_toppings_combinations_l2689_268970

theorem pizza_toppings_combinations : Nat.choose 7 3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_combinations_l2689_268970


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2689_268969

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  (3 * a 3 ^ 2 - 11 * a 3 + 9 = 0) →
  (3 * a 9 ^ 2 - 11 * a 9 + 9 = 0) →
  (a 5 * a 6 * a 7 = 3 * Real.sqrt 3 ∨ a 5 * a 6 * a 7 = -3 * Real.sqrt 3) :=
by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_product_l2689_268969


namespace NUMINAMATH_CALUDE_intersection_point_l₁_l₂_l2689_268966

/-- The intersection point of two lines in 2D space -/
structure IntersectionPoint (l₁ l₂ : ℝ → ℝ → Prop) where
  x : ℝ
  y : ℝ
  on_l₁ : l₁ x y
  on_l₂ : l₂ x y
  unique : ∀ x' y', l₁ x' y' → l₂ x' y' → x' = x ∧ y' = y

/-- Line l₁: 2x - y - 10 = 0 -/
def l₁ (x y : ℝ) : Prop := 2 * x - y - 10 = 0

/-- Line l₂: 3x + 4y - 4 = 0 -/
def l₂ (x y : ℝ) : Prop := 3 * x + 4 * y - 4 = 0

/-- The intersection point of l₁ and l₂ is (4, -2) -/
theorem intersection_point_l₁_l₂ : IntersectionPoint l₁ l₂ where
  x := 4
  y := -2
  on_l₁ := by sorry
  on_l₂ := by sorry
  unique := by sorry

end NUMINAMATH_CALUDE_intersection_point_l₁_l₂_l2689_268966


namespace NUMINAMATH_CALUDE_ball_probability_pairs_l2689_268934

theorem ball_probability_pairs : 
  ∃! k : ℕ, ∃ S : Finset (ℕ × ℕ),
    (∀ (m n : ℕ), (m, n) ∈ S ↔ 
      (m > n ∧ n ≥ 4 ∧ m + n ≤ 40 ∧ (m - n)^2 = m + n)) ∧
    S.card = k ∧ k = 3 := by sorry

end NUMINAMATH_CALUDE_ball_probability_pairs_l2689_268934


namespace NUMINAMATH_CALUDE_bird_weight_equations_l2689_268985

/-- Represents the weight of birds in jin -/
structure BirdWeight where
  sparrow : ℝ
  swallow : ℝ

/-- The total weight of 5 sparrows and 6 swallows is 1 jin -/
def total_weight (w : BirdWeight) : Prop :=
  5 * w.sparrow + 6 * w.swallow = 1

/-- Sparrows are heavier than swallows -/
def sparrow_heavier (w : BirdWeight) : Prop :=
  w.sparrow > w.swallow

/-- Exchanging one sparrow with one swallow doesn't change the total weight -/
def exchange_weight (w : BirdWeight) : Prop :=
  4 * w.sparrow + 7 * w.swallow = 5 * w.swallow + w.sparrow

/-- The system of equations correctly represents the bird weight problem -/
theorem bird_weight_equations (w : BirdWeight) 
  (h1 : total_weight w) 
  (h2 : sparrow_heavier w) 
  (h3 : exchange_weight w) : 
  5 * w.sparrow + 6 * w.swallow = 1 ∧ 3 * w.sparrow = -2 * w.swallow := by
  sorry

end NUMINAMATH_CALUDE_bird_weight_equations_l2689_268985


namespace NUMINAMATH_CALUDE_period_of_symmetric_function_l2689_268964

/-- A function f is symmetric about a point c if f(c + x) = f(c - x) for all x -/
def SymmetricAbout (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, f (c + x) = f (c - x)

/-- A real number p is a period of a function f if f(x + p) = f(x) for all x -/
def IsPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem period_of_symmetric_function (f : ℝ → ℝ) (a b : ℝ) 
    (h1 : SymmetricAbout (fun x ↦ f (2 * x)) (a / 2)) 
    (h2 : SymmetricAbout (fun x ↦ f (2 * x)) (b / 2)) 
    (h3 : b > a) : 
    IsPeriod f (4 * (b - a)) := by
  sorry

end NUMINAMATH_CALUDE_period_of_symmetric_function_l2689_268964


namespace NUMINAMATH_CALUDE_congruence_solution_l2689_268932

theorem congruence_solution (n : ℤ) : (13 * n) % 47 = 8 ↔ n % 47 = 4 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l2689_268932


namespace NUMINAMATH_CALUDE_solve_linear_equation_l2689_268929

theorem solve_linear_equation :
  ∃ x : ℚ, 3*x - 5*x + 9*x + 4 = 289 ∧ x = 285/7 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l2689_268929


namespace NUMINAMATH_CALUDE_triangular_bipyramid_existence_condition_l2689_268955

/-- A triangular bipyramid with four edges of length 1 and two edges of length x -/
structure TriangularBipyramid (x : ℝ) :=
  (edge_length_1 : ℝ := 1)
  (edge_length_x : ℝ := x)
  (num_edges_1 : ℕ := 4)
  (num_edges_x : ℕ := 2)

/-- The existence condition for a triangular bipyramid -/
def exists_triangular_bipyramid (x : ℝ) : Prop :=
  0 < x ∧ x < (Real.sqrt 6 + Real.sqrt 2) / 2

/-- Theorem stating the range of x for which a triangular bipyramid can exist -/
theorem triangular_bipyramid_existence_condition (x : ℝ) :
  (∃ t : TriangularBipyramid x, True) ↔ exists_triangular_bipyramid x :=
sorry

end NUMINAMATH_CALUDE_triangular_bipyramid_existence_condition_l2689_268955


namespace NUMINAMATH_CALUDE_largest_number_l2689_268913

def numbers : List ℝ := [0.988, 0.9808, 0.989, 0.9809, 0.998]

theorem largest_number (n : ℝ) (hn : n ∈ numbers) : n ≤ 0.998 :=
by sorry

end NUMINAMATH_CALUDE_largest_number_l2689_268913


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l2689_268960

theorem simplify_sqrt_expression (y : ℝ) (h : y ≠ 0) :
  Real.sqrt (4 + ((y^6 - 4) / (3 * y^3))^2) = (Real.sqrt (y^12 + 28 * y^6 + 16)) / (3 * y^3) :=
by sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l2689_268960
