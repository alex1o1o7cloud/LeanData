import Mathlib

namespace expression_evaluation_l2416_241648

theorem expression_evaluation : (-4)^6 / 4^4 + 2^5 * 5 - 7^2 = 127 := by
  sorry

end expression_evaluation_l2416_241648


namespace kathy_happy_probability_kathy_probability_sum_l2416_241615

def total_cards : ℕ := 10
def cards_laid_out : ℕ := 5
def red_cards : ℕ := 5
def green_cards : ℕ := 5

def happy_configurations : ℕ := 62
def total_configurations : ℕ := 30240

def probability_numerator : ℕ := 31
def probability_denominator : ℕ := 15120

theorem kathy_happy_probability :
  (happy_configurations : ℚ) / total_configurations = probability_numerator / probability_denominator :=
sorry

theorem kathy_probability_sum :
  probability_numerator + probability_denominator = 15151 :=
sorry

end kathy_happy_probability_kathy_probability_sum_l2416_241615


namespace company_employees_l2416_241619

/-- If a company has 15% more employees in December than in January,
    and it has 500 employees in December, then it had 435 employees in January. -/
theorem company_employees (january_employees : ℕ) (december_employees : ℕ) :
  december_employees = 500 →
  december_employees = january_employees + (january_employees * 15 / 100) →
  january_employees = 435 := by
sorry

end company_employees_l2416_241619


namespace regression_coefficient_nonzero_l2416_241625

/-- Represents a regression line for two variables with a linear relationship -/
structure RegressionLine where
  a : ℝ
  b : ℝ

/-- Theorem: The regression coefficient b in a regression line y = a + bx 
    for two variables with a linear relationship cannot be equal to 0 -/
theorem regression_coefficient_nonzero (line : RegressionLine) : 
  line.b ≠ 0 := by
  sorry

end regression_coefficient_nonzero_l2416_241625


namespace seating_arrangements_count_l2416_241627

/-- The number of ways to seat n people in a row -/
def totalArrangements (n : ℕ) : ℕ := n.factorial

/-- The number of ways to seat n people in a row where m specific people sit together -/
def arrangementsWithGroupTogether (n m : ℕ) : ℕ := 
  (n - m + 1).factorial * m.factorial

/-- The number of ways to seat 10 people in a row where 4 specific people cannot sit in 4 consecutive seats -/
def seatingArrangements : ℕ := 
  totalArrangements 10 - arrangementsWithGroupTogether 10 4

theorem seating_arrangements_count : seatingArrangements = 3507840 := by
  sorry

end seating_arrangements_count_l2416_241627


namespace a_equals_one_sufficient_not_necessary_for_abs_a_equals_one_l2416_241665

theorem a_equals_one_sufficient_not_necessary_for_abs_a_equals_one :
  ∃ (a : ℝ), (a = 1 → |a| = 1) ∧ (|a| = 1 → ¬(a = 1 ↔ |a| = 1)) := by
  sorry

end a_equals_one_sufficient_not_necessary_for_abs_a_equals_one_l2416_241665


namespace pipe_fill_rate_l2416_241660

theorem pipe_fill_rate (slow_time fast_time combined_time : ℝ) 
  (h1 : slow_time = 160)
  (h2 : combined_time = 40)
  (h3 : slow_time > 0)
  (h4 : fast_time > 0)
  (h5 : combined_time > 0)
  (h6 : 1 / combined_time = 1 / fast_time + 1 / slow_time) :
  fast_time = slow_time / 3 :=
sorry

end pipe_fill_rate_l2416_241660


namespace cubic_root_complex_coefficients_l2416_241606

theorem cubic_root_complex_coefficients :
  ∀ (a b : ℝ),
  (∃ (x : ℂ), x^3 + a*x^2 + 2*x + b = 0 ∧ x = Complex.mk 2 (-3)) →
  a = -5/4 ∧ b = 143/4 := by
sorry

end cubic_root_complex_coefficients_l2416_241606


namespace derek_dogs_at_16_l2416_241658

/-- Represents Derek's possessions at different ages --/
structure DereksPossessions where
  dogs_at_6 : ℕ
  cars_at_6 : ℕ
  dogs_at_16 : ℕ
  cars_at_16 : ℕ

/-- Theorem stating the number of dogs Derek has at age 16 --/
theorem derek_dogs_at_16 (d : DereksPossessions) 
  (h1 : d.dogs_at_6 = 3 * d.cars_at_6)
  (h2 : d.dogs_at_6 = 90)
  (h3 : d.cars_at_16 = d.cars_at_6 + 210)
  (h4 : d.cars_at_16 = 2 * d.dogs_at_16) :
  d.dogs_at_16 = 120 := by
  sorry

#check derek_dogs_at_16

end derek_dogs_at_16_l2416_241658


namespace basketball_win_percentage_l2416_241644

theorem basketball_win_percentage (games_played : ℕ) (games_won : ℕ) (games_left : ℕ) 
  (target_percentage : ℚ) (h1 : games_played = 50) (h2 : games_won = 35) 
  (h3 : games_left = 25) (h4 : target_percentage = 64 / 100) : 
  ∃ (additional_wins : ℕ), 
    (games_won + additional_wins) / (games_played + games_left : ℚ) = target_percentage ∧ 
    additional_wins = 13 := by
  sorry

end basketball_win_percentage_l2416_241644


namespace cubes_not_touching_foil_count_l2416_241601

/-- Represents the dimensions of a rectangular prism -/
structure PrismDimensions where
  width : ℕ
  length : ℕ
  height : ℕ

/-- Calculates the number of cubes not touching tin foil in a prism -/
def cubes_not_touching_foil (outer_width : ℕ) (inner : PrismDimensions) : ℕ :=
  inner.length * inner.width * inner.height

/-- Theorem stating the number of cubes not touching tin foil -/
theorem cubes_not_touching_foil_count 
  (outer_width : ℕ) 
  (inner : PrismDimensions) 
  (h1 : outer_width = 10)
  (h2 : inner.width = 2 * inner.length)
  (h3 : inner.width = 2 * inner.height)
  (h4 : inner.width = outer_width - 4) :
  cubes_not_touching_foil outer_width inner = 54 := by
  sorry

#eval cubes_not_touching_foil 10 ⟨6, 3, 3⟩

end cubes_not_touching_foil_count_l2416_241601


namespace two_solutions_characterization_l2416_241653

def has_two_solutions (a : ℕ) : Prop :=
  ∃ x y : ℕ, x < 2007 ∧ y < 2007 ∧ x ≠ y ∧
  x^2 + a ≡ 0 [ZMOD 2007] ∧ y^2 + a ≡ 0 [ZMOD 2007] ∧
  ∀ z : ℕ, z < 2007 → z^2 + a ≡ 0 [ZMOD 2007] → (z = x ∨ z = y)

theorem two_solutions_characterization :
  {a : ℕ | a < 2007 ∧ has_two_solutions a} = {446, 1115, 1784} := by sorry

end two_solutions_characterization_l2416_241653


namespace find_k_l2416_241610

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x^2 - 3 * x + 5
def g (k : ℝ) (x : ℝ) : ℝ := 2 * x^2 - k * x + 7

-- State the theorem
theorem find_k : ∃ k : ℝ, f 5 - g k 5 = 40 ∧ k = 1.4 := by sorry

end find_k_l2416_241610


namespace system_solution_l2416_241652

-- Define the system of equations
def system (x y : ℝ) : Prop :=
  3 * x + y = 5 ∧ x + 3 * y = 7

-- State the theorem
theorem system_solution :
  ∃! (x y : ℝ), system x y ∧ x = 1 ∧ y = 2 := by
  sorry

end system_solution_l2416_241652


namespace smallest_stairs_l2416_241609

theorem smallest_stairs (n : ℕ) : 
  n > 10 ∧ 
  n % 6 = 4 ∧ 
  n % 7 = 3 → 
  n ≥ 52 ∧ 
  ∃ (m : ℕ), m > 10 ∧ m % 6 = 4 ∧ m % 7 = 3 ∧ m = 52 := by
  sorry

end smallest_stairs_l2416_241609


namespace cube_sum_divisible_implies_product_divisible_l2416_241684

theorem cube_sum_divisible_implies_product_divisible (a b c : ℤ) :
  7 ∣ (a^3 + b^3 + c^3) → 7 ∣ (a * b * c) := by
sorry

end cube_sum_divisible_implies_product_divisible_l2416_241684


namespace classroom_boys_count_l2416_241611

/-- Represents the number of desks with one boy and one girl -/
def x : ℕ := 2

/-- The number of desks with two girls -/
def desks_two_girls : ℕ := 2 * x

/-- The number of desks with two boys -/
def desks_two_boys : ℕ := 2 * desks_two_girls

/-- The total number of girls in the classroom -/
def total_girls : ℕ := 10

/-- The total number of boys in the classroom -/
def total_boys : ℕ := 2 * desks_two_boys + x

theorem classroom_boys_count :
  total_girls = 5 * x ∧ total_boys = 18 := by
  sorry

#check classroom_boys_count

end classroom_boys_count_l2416_241611


namespace smallest_N_satisfying_condition_l2416_241697

def P (N : ℕ) : ℚ := (4 * N + 2) / (5 * N + 1)

theorem smallest_N_satisfying_condition :
  ∃ (N : ℕ), N > 0 ∧ N % 5 = 0 ∧ P N < 321 / 400 ∧
  ∀ (M : ℕ), M > 0 → M % 5 = 0 → P M < 321 / 400 → N ≤ M ∧
  N = 480 := by
  sorry

end smallest_N_satisfying_condition_l2416_241697


namespace cubic_function_extreme_value_l2416_241618

/-- Given a cubic function f(x) = ax³ + bx + c that reaches an extreme value of c-6 at x=2,
    prove that a = 3/8 and b = -9/2 -/
theorem cubic_function_extreme_value (a b c : ℝ) :
  (∀ x, ∃ f : ℝ → ℝ, f x = a * x^3 + b * x + c) →
  (∃ f : ℝ → ℝ, f 2 = c - 6 ∧ ∀ x, f x ≤ f 2 ∨ f x ≥ f 2) →
  a = 3/8 ∧ b = -9/2 := by
  sorry

end cubic_function_extreme_value_l2416_241618


namespace truck_driver_net_pay_rate_l2416_241634

/-- Calculates the net rate of pay for a truck driver given specific conditions --/
theorem truck_driver_net_pay_rate
  (travel_time : ℝ)
  (speed : ℝ)
  (fuel_efficiency : ℝ)
  (pay_rate : ℝ)
  (diesel_cost : ℝ)
  (h_travel_time : travel_time = 3)
  (h_speed : speed = 50)
  (h_fuel_efficiency : fuel_efficiency = 25)
  (h_pay_rate : pay_rate = 0.60)
  (h_diesel_cost : diesel_cost = 2.50)
  : (pay_rate * speed * travel_time - (speed * travel_time / fuel_efficiency) * diesel_cost) / travel_time = 25 := by
  sorry

end truck_driver_net_pay_rate_l2416_241634


namespace water_flow_difference_l2416_241630

/-- Given a water flow restrictor problem, prove the difference between 0.6 times
    the original flow rate and the reduced flow rate. -/
theorem water_flow_difference (original_rate reduced_rate : ℝ) 
    (h1 : original_rate = 5)
    (h2 : reduced_rate = 2) :
  0.6 * original_rate - reduced_rate = 1 := by
  sorry

end water_flow_difference_l2416_241630


namespace algebraic_expression_value_l2416_241641

theorem algebraic_expression_value (a b : ℝ) (h : 2 * a - b = 5) :
  4 * a - 2 * b + 7 = 17 := by
  sorry

end algebraic_expression_value_l2416_241641


namespace inequality_proof_l2416_241674

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := 2 * (a + 1) * Real.log x - a * x

def g (x : ℝ) : ℝ := (1 / 2) * x^2 - x

theorem inequality_proof (a : ℝ) (x₁ x₂ : ℝ) 
  (ha : -1 < a ∧ a < 7) 
  (hx₁ : x₁ > 1) 
  (hx₂ : x₂ > 1) 
  (hne : x₁ ≠ x₂) : 
  (f a x₁ - f a x₂) / (g x₁ - g x₂) > -1 := by
  sorry

end

end inequality_proof_l2416_241674


namespace pool_filling_time_l2416_241626

theorem pool_filling_time (a b c d : ℝ) 
  (h1 : a + b = 1/2)
  (h2 : b + c = 1/3)
  (h3 : c + d = 1/4) :
  a + d = 5/12 :=
sorry

end pool_filling_time_l2416_241626


namespace ratio_problem_l2416_241643

theorem ratio_problem (a b c : ℝ) 
  (h1 : b / c = 1 / 5)
  (h2 : a / c = 1 / 7.5) :
  a / b = 2 / 3 := by
  sorry

end ratio_problem_l2416_241643


namespace variableCostIncrease_is_ten_percent_l2416_241664

/-- Represents the annual breeding cost model for a certain breeder -/
structure BreedingCost where
  fixedCost : ℝ
  initialVariableCost : ℝ
  variableCostIncrease : ℝ

/-- Calculates the total breeding cost for a given year -/
def totalCost (model : BreedingCost) (year : ℕ) : ℝ :=
  model.fixedCost + model.initialVariableCost * (1 + model.variableCostIncrease) ^ (year - 1)

/-- Theorem stating that the percentage increase in variable costs is 10% -/
theorem variableCostIncrease_is_ten_percent (model : BreedingCost) :
  model.fixedCost = 40000 →
  model.initialVariableCost = 26000 →
  totalCost model 3 = 71460 →
  model.variableCostIncrease = 0.1 := by
  sorry


end variableCostIncrease_is_ten_percent_l2416_241664


namespace sum_of_five_consecutive_integers_l2416_241600

theorem sum_of_five_consecutive_integers (n : ℤ) : 
  (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 5 * n + 20 := by
  sorry

end sum_of_five_consecutive_integers_l2416_241600


namespace family_size_l2416_241691

/-- Represents the number of slices per tomato -/
def slices_per_tomato : ℕ := 8

/-- Represents the number of slices needed for one person's meal -/
def slices_per_meal : ℕ := 20

/-- Represents the number of tomatoes Thelma needs -/
def total_tomatoes : ℕ := 20

/-- Theorem: Given the conditions, the family has 8 people -/
theorem family_size :
  (total_tomatoes * slices_per_tomato) / slices_per_meal = 8 := by
  sorry

end family_size_l2416_241691


namespace sequence_inequality_l2416_241671

theorem sequence_inequality (a : ℕ → ℕ) 
  (h1 : a 2 > a 1) 
  (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 2) = 3 * a (n + 1) - 2 * a n) : 
  a 2021 > 2^2019 := by
  sorry

end sequence_inequality_l2416_241671


namespace equation_solution_exists_l2416_241676

theorem equation_solution_exists : ∃ x : ℝ, -x^3 + 555^3 = x^2 - x * 555 + 555^2 := by
  sorry

end equation_solution_exists_l2416_241676


namespace work_completion_time_l2416_241687

/-- The number of days it takes to complete a task when two people work together -/
def combined_work_time (john_rate : ℚ) (rose_rate : ℚ) : ℚ :=
  1 / (john_rate + rose_rate)

/-- Theorem: John and Rose complete the work in 8 days when working together -/
theorem work_completion_time :
  let john_rate : ℚ := 1 / 10
  let rose_rate : ℚ := 1 / 40
  combined_work_time john_rate rose_rate = 8 := by sorry

end work_completion_time_l2416_241687


namespace existence_and_not_forall_l2416_241672

theorem existence_and_not_forall :
  (∃ x₀ : ℝ, x₀ - 2 > 0) ∧ ¬(∀ x : ℝ, 2^x > x^2) := by
  sorry

end existence_and_not_forall_l2416_241672


namespace chocolate_candy_cost_l2416_241649

/-- The cost of purchasing a given number of chocolate candies, given the cost and quantity of a box. -/
theorem chocolate_candy_cost (box_quantity : ℕ) (box_cost : ℚ) (total_quantity : ℕ) : 
  (total_quantity / box_quantity : ℚ) * box_cost = 72 :=
by
  sorry

#check chocolate_candy_cost 40 8 360

end chocolate_candy_cost_l2416_241649


namespace douglas_county_y_percentage_l2416_241679

-- Define the ratio of voters in county X to county Y
def voter_ratio : ℚ := 2 / 1

-- Define the percentage of votes Douglas won in both counties combined
def total_vote_percentage : ℚ := 60 / 100

-- Define the percentage of votes Douglas won in county X
def county_x_percentage : ℚ := 72 / 100

-- Theorem to prove
theorem douglas_county_y_percentage :
  let total_voters := 3 -- represents the sum of parts in the ratio (2 + 1)
  let county_x_voters := 2 -- represents the larger part of the ratio
  let county_y_voters := 1 -- represents the smaller part of the ratio
  let total_douglas_votes := total_vote_percentage * total_voters
  let county_x_douglas_votes := county_x_percentage * county_x_voters
  let county_y_douglas_votes := total_douglas_votes - county_x_douglas_votes
  county_y_douglas_votes / county_y_voters = 36 / 100 :=
by sorry

end douglas_county_y_percentage_l2416_241679


namespace solve_for_y_l2416_241685

theorem solve_for_y (x y : ℝ) (h1 : x^2 + x + 4 = y - 4) (h2 : x = 3) : y = 20 := by
  sorry

end solve_for_y_l2416_241685


namespace problem_statement_l2416_241695

theorem problem_statement (x : ℝ) : 
  (0.4 * 60 = (4/5) * x + 4) → x = 25 := by
sorry

end problem_statement_l2416_241695


namespace finsler_hadwiger_inequality_l2416_241673

/-- The Finsler-Hadwiger inequality for triangles -/
theorem finsler_hadwiger_inequality (a b c : ℝ) (S : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) (h_area : S > 0) :
  a^2 + b^2 + c^2 - (a-b)^2 - (b-c)^2 - (c-a)^2 ≥ 4 * Real.sqrt 3 * S := by
  sorry

end finsler_hadwiger_inequality_l2416_241673


namespace tallest_building_height_l2416_241632

theorem tallest_building_height :
  ∀ (h1 h2 h3 h4 : ℝ),
    h2 = h1 / 2 →
    h3 = h2 / 2 →
    h4 = h3 / 5 →
    h1 + h2 + h3 + h4 = 180 →
    h1 = 100 := by
  sorry

end tallest_building_height_l2416_241632


namespace james_writing_hours_l2416_241693

/-- Calculates the number of hours James spends writing per week -/
def writing_hours_per_week (pages_per_hour : ℕ) (pages_per_person_per_day : ℕ) (num_people : ℕ) (days_per_week : ℕ) : ℕ :=
  (pages_per_person_per_day * num_people * days_per_week) / pages_per_hour

theorem james_writing_hours (pages_per_hour : ℕ) (pages_per_person_per_day : ℕ) (num_people : ℕ) (days_per_week : ℕ)
  (h1 : pages_per_hour = 10)
  (h2 : pages_per_person_per_day = 5)
  (h3 : num_people = 2)
  (h4 : days_per_week = 7) :
  writing_hours_per_week pages_per_hour pages_per_person_per_day num_people days_per_week = 7 := by
  sorry

end james_writing_hours_l2416_241693


namespace conic_section_eccentricity_l2416_241683

/-- The eccentricity of the conic section defined by 10x - 2xy - 2y + 1 = 0 is √2 -/
theorem conic_section_eccentricity :
  let P : ℝ × ℝ → Prop := λ (x, y) ↦ 10 * x - 2 * x * y - 2 * y + 1 = 0
  ∃ e : ℝ, e = Real.sqrt 2 ∧
    ∀ (x y : ℝ), P (x, y) →
      (Real.sqrt ((x - 2)^2 + (y - 2)^2)) / (|x - y + 3| / Real.sqrt 2) = e :=
by sorry

end conic_section_eccentricity_l2416_241683


namespace letter_F_transformation_l2416_241605

-- Define the position of the letter F
structure LetterFPosition where
  base : (ℝ × ℝ) -- Base endpoint
  top : (ℝ × ℝ)  -- Top endpoint

-- Define the transformations
def reflectXAxis (p : LetterFPosition) : LetterFPosition :=
  { base := (p.base.1, -p.base.2), top := (p.top.1, -p.top.2) }

def rotateCounterClockwise90 (p : LetterFPosition) : LetterFPosition :=
  { base := (-p.base.2, p.base.1), top := (-p.top.2, p.top.1) }

def rotate180 (p : LetterFPosition) : LetterFPosition :=
  { base := (-p.base.1, -p.base.2), top := (-p.top.1, -p.top.2) }

def reflectYAxis (p : LetterFPosition) : LetterFPosition :=
  { base := (-p.base.1, p.base.2), top := (-p.top.1, p.top.2) }

-- Define the initial position
def initialPosition : LetterFPosition :=
  { base := (0, -1), top := (1, 0) }

-- Define the final position
def finalPosition : LetterFPosition :=
  { base := (1, 0), top := (0, 1) }

-- Theorem statement
theorem letter_F_transformation :
  (reflectYAxis ∘ rotate180 ∘ rotateCounterClockwise90 ∘ reflectXAxis) initialPosition = finalPosition := by
  sorry

end letter_F_transformation_l2416_241605


namespace AgOH_formation_l2416_241670

-- Define the chemical reaction
structure Reaction where
  reactant1 : String
  reactant2 : String
  product1 : String
  product2 : String
  ratio : ℚ

-- Define the initial conditions
def initial_AgNO3 : ℚ := 3
def initial_NaOH : ℚ := 3

-- Define the reaction
def silver_hydroxide_reaction : Reaction := {
  reactant1 := "AgNO3"
  reactant2 := "NaOH"
  product1 := "AgOH"
  product2 := "NaNO3"
  ratio := 1
}

-- Theorem statement
theorem AgOH_formation (r : Reaction) (h1 : r = silver_hydroxide_reaction) 
  (h2 : initial_AgNO3 = initial_NaOH) : 
  let moles_AgOH := min initial_AgNO3 initial_NaOH
  moles_AgOH = 3 := by sorry

end AgOH_formation_l2416_241670


namespace negation_equivalence_l2416_241616

theorem negation_equivalence : 
  (¬ ∃ x : ℝ, x^2 + 2*x + 1 < 0) ↔ (∀ x : ℝ, x^2 + 2*x + 1 ≥ 0) := by
  sorry

end negation_equivalence_l2416_241616


namespace mingming_calculation_correction_l2416_241662

theorem mingming_calculation_correction : 
  (-4 - 2/3) - (1 + 5/6) - (-18 - 1/2) + (-13 - 3/4) = -7/4 := by sorry

end mingming_calculation_correction_l2416_241662


namespace even_function_derivative_is_odd_l2416_241690

-- Define a function f on the real numbers
variable (f : ℝ → ℝ)

-- Define the derivative of f as g
variable (g : ℝ → ℝ)

-- State the theorem
theorem even_function_derivative_is_odd 
  (h1 : ∀ x, f (-x) = f x)  -- f is an even function
  (h2 : ∀ x, HasDerivAt f (g x) x) -- g is the derivative of f
  : ∀ x, g (-x) = -g x := by sorry

end even_function_derivative_is_odd_l2416_241690


namespace rectangle_perimeter_theorem_l2416_241675

theorem rectangle_perimeter_theorem (a b : ℕ) : 
  a ≠ b →                 -- non-square condition
  a * b = 4 * (2 * a + 2 * b) →  -- area equals four times perimeter
  2 * (a + b) = 66 :=     -- perimeter is 66
by
  sorry

end rectangle_perimeter_theorem_l2416_241675


namespace art_club_collection_l2416_241623

/-- Calculates the total number of artworks collected by an art club over multiple school years. -/
def total_artworks (students : ℕ) (artworks_per_student_per_quarter : ℕ) (quarters_per_year : ℕ) (years : ℕ) : ℕ :=
  students * artworks_per_student_per_quarter * quarters_per_year * years

/-- Proves that an art club with 15 students, each making 2 artworks per quarter, 
    collects 240 artworks in 2 school years with 4 quarters per year. -/
theorem art_club_collection : total_artworks 15 2 4 2 = 240 := by
  sorry

end art_club_collection_l2416_241623


namespace more_subsets_gt_2009_l2416_241654

def M : Finset ℕ := {1, 2, 3, 4, 6, 8, 12, 16, 24, 48}

def product (s : Finset ℕ) : ℕ := s.prod id

def subsets_gt_2009 : Finset (Finset ℕ) :=
  M.powerset.filter (λ s => s.card = 4 ∧ product s > 2009)

def subsets_lt_2009 : Finset (Finset ℕ) :=
  M.powerset.filter (λ s => s.card = 4 ∧ product s < 2009)

theorem more_subsets_gt_2009 : subsets_gt_2009.card > subsets_lt_2009.card := by
  sorry

end more_subsets_gt_2009_l2416_241654


namespace thursday_beef_sales_l2416_241692

/-- Given a store's beef sales over three days, prove that Thursday's sales were 210 pounds -/
theorem thursday_beef_sales (x : ℝ) : 
  (x + 2*x + 150) / 3 = 260 → x = 210 := by sorry

end thursday_beef_sales_l2416_241692


namespace expansion_coefficient_a_l2416_241694

theorem expansion_coefficient_a (a a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (x + 1)^5 = a + a₁*(x - 1) + a₂*(x - 1)^2 + a₃*(x - 1)^3 + a₄*(x - 1)^4 + a₅*(x - 1)^5) →
  a = 32 := by
sorry

end expansion_coefficient_a_l2416_241694


namespace parking_spaces_available_l2416_241681

theorem parking_spaces_available (front_spaces back_spaces parked_cars : ℕ) :
  front_spaces = 52 →
  back_spaces = 38 →
  parked_cars = 39 →
  (front_spaces + back_spaces) - (parked_cars + back_spaces / 2) = 32 := by
  sorry

end parking_spaces_available_l2416_241681


namespace movies_to_watch_l2416_241698

/-- Given a series with 17 movies, if 7 movies have been watched,
    then the number of movies still to watch is 10. -/
theorem movies_to_watch (total_movies : ℕ) (watched_movies : ℕ) : 
  total_movies = 17 → watched_movies = 7 → total_movies - watched_movies = 10 := by
  sorry

end movies_to_watch_l2416_241698


namespace fn_solution_l2416_241620

-- Define the sequence of functions
def f : ℕ → ℝ → ℝ
| 0, x => |x|
| n + 1, x => |f n x - 2|

-- Define the set of solutions
def solution_set (n : ℕ) : Set ℝ :=
  {x | ∃ k : ℤ, x = 2*k + 1 ∨ x = -(2*k + 1) ∧ |2*k + 1| ≤ 2*n + 1}

-- Theorem statement
theorem fn_solution (n : ℕ+) :
  {x : ℝ | f n x = 1} = solution_set n :=
sorry

end fn_solution_l2416_241620


namespace backpack_store_theorem_l2416_241646

/-- Represents the backpack types --/
inductive BackpackType
| A
| B

/-- Represents a purchasing plan --/
structure PurchasePlan where
  typeA : ℕ
  typeB : ℕ

/-- Represents the backpack pricing and inventory --/
structure BackpackStore where
  sellingPriceA : ℕ
  sellingPriceB : ℕ
  costPriceA : ℕ
  costPriceB : ℕ
  inventory : PurchasePlan
  givenAwayA : ℕ
  givenAwayB : ℕ

/-- The main theorem to prove --/
theorem backpack_store_theorem (store : BackpackStore) : 
  (store.sellingPriceA = store.sellingPriceB + 12) →
  (2 * store.sellingPriceA + 3 * store.sellingPriceB = 264) →
  (store.inventory.typeA + store.inventory.typeB = 100) →
  (store.costPriceA * store.inventory.typeA + store.costPriceB * store.inventory.typeB ≤ 4550) →
  (store.inventory.typeA > 52) →
  (store.costPriceA = 50) →
  (store.costPriceB = 40) →
  (store.givenAwayA + store.givenAwayB = 5) →
  (store.sellingPriceA * (store.inventory.typeA - store.givenAwayA) + 
   store.sellingPriceB * (store.inventory.typeB - store.givenAwayB) - 
   store.costPriceA * store.inventory.typeA - 
   store.costPriceB * store.inventory.typeB = 658) →
  (store.sellingPriceA = 60 ∧ store.sellingPriceB = 48) ∧
  ((store.inventory.typeA = 53 ∧ store.inventory.typeB = 47) ∨
   (store.inventory.typeA = 54 ∧ store.inventory.typeB = 46) ∨
   (store.inventory.typeA = 55 ∧ store.inventory.typeB = 45)) ∧
  (store.givenAwayA = 1 ∧ store.givenAwayB = 4) :=
by sorry


end backpack_store_theorem_l2416_241646


namespace no_integer_solutions_l2416_241659

theorem no_integer_solutions :
  ¬ ∃ (x y : ℤ), 19 * x^3 - 84 * y^2 = 1984 := by
sorry

end no_integer_solutions_l2416_241659


namespace no_valid_arrangement_l2416_241696

/-- A sequence of natural numbers from 1 to 10 -/
def Sequence := Fin 10 → ℕ

/-- Predicate to check if a sequence satisfies the integer percentage difference property -/
def IntegerPercentageDifference (s : Sequence) : Prop :=
  ∀ i : Fin 9, ∃ k : ℤ,
    s (i.succ) = s i + (s i * k) / 100 ∨
    s (i.succ) = s i - (s i * k) / 100

/-- Predicate to check if a sequence contains all numbers from 1 to 10 -/
def ContainsAllNumbers (s : Sequence) : Prop :=
  ∀ n : Fin 10, ∃ i : Fin 10, s i = n.val + 1

/-- Theorem stating that it's impossible to arrange numbers 1 to 10 with the given property -/
theorem no_valid_arrangement :
  ¬ ∃ s : Sequence, IntegerPercentageDifference s ∧ ContainsAllNumbers s := by
  sorry

end no_valid_arrangement_l2416_241696


namespace last_car_speed_l2416_241628

theorem last_car_speed (n : ℕ) (first_speed last_speed : ℕ) : 
  n = 31 ∧ 
  first_speed = 61 ∧ 
  last_speed = 91 ∧ 
  last_speed - first_speed + 1 = n → 
  first_speed + ((n + 1) / 2 - 1) = 76 :=
by sorry

end last_car_speed_l2416_241628


namespace seating_arrangement_l2416_241640

/-- Given a seating arrangement where each row seats either 6 or 9 people,
    and 57 people are to be seated with all seats occupied,
    prove that there is exactly 1 row seating 9 people. -/
theorem seating_arrangement (x y : ℕ) : 
  9 * x + 6 * y = 57 → 
  x + y > 0 →
  x = 1 := by
sorry

end seating_arrangement_l2416_241640


namespace average_speed_calculation_l2416_241613

/-- Calculates the average speed of a car trip given odometer readings and time taken -/
theorem average_speed_calculation
  (initial_reading : ℝ)
  (lunch_reading : ℝ)
  (final_reading : ℝ)
  (total_time : ℝ)
  (h1 : initial_reading < lunch_reading)
  (h2 : lunch_reading < final_reading)
  (h3 : total_time > 0) :
  let total_distance := final_reading - initial_reading
  (total_distance / total_time) = (final_reading - initial_reading) / total_time :=
by sorry

end average_speed_calculation_l2416_241613


namespace theater_seat_increment_l2416_241677

/-- Represents a theater with a specific seating arrangement -/
structure Theater where
  num_rows : ℕ
  first_row_seats : ℕ
  last_row_seats : ℕ
  total_seats : ℕ

/-- 
  Given a theater with 23 rows, where the first row has 14 seats, 
  the last row has 56 seats, and the total number of seats is 770, 
  prove that the number of additional seats in each row compared 
  to the previous row is 2.
-/
theorem theater_seat_increment (t : Theater) 
  (h1 : t.num_rows = 23)
  (h2 : t.first_row_seats = 14)
  (h3 : t.last_row_seats = 56)
  (h4 : t.total_seats = 770) : 
  (t.last_row_seats - t.first_row_seats) / (t.num_rows - 1) = 2 := by
  sorry


end theater_seat_increment_l2416_241677


namespace right_triangle_area_l2416_241651

theorem right_triangle_area (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : c = 13) (h3 : a = 5) : (1/2) * a * b = 30 := by
  sorry

end right_triangle_area_l2416_241651


namespace polynomial_identity_l2416_241622

/-- Given a polynomial function f such that f(x^2 + 1) = x^4 + 4x^2 for all x,
    prove that f(x^2 - 1) = x^4 - 4 for all x. -/
theorem polynomial_identity (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x^2 + 1) = x^4 + 4*x^2) :
  ∀ x : ℝ, f (x^2 - 1) = x^4 - 4 := by
sorry

end polynomial_identity_l2416_241622


namespace floor_painting_two_solutions_l2416_241617

/-- The number of ordered pairs (a, b) satisfying the floor painting conditions -/
def floor_painting_solutions : Nat :=
  (Finset.filter (fun p : Nat × Nat =>
    let a := p.1
    let b := p.2
    b > a ∧ (a - 4) * (b - 4) = 2 * a * b / 3
  ) (Finset.product (Finset.range 100) (Finset.range 100))).card

/-- There are exactly two solutions to the floor painting problem -/
theorem floor_painting_two_solutions : floor_painting_solutions = 2 := by
  sorry

end floor_painting_two_solutions_l2416_241617


namespace initial_birds_on_fence_l2416_241602

theorem initial_birds_on_fence (initial_birds additional_birds total_birds : ℕ) :
  initial_birds + additional_birds = total_birds →
  additional_birds = 4 →
  total_birds = 6 →
  initial_birds = 2 := by
sorry

end initial_birds_on_fence_l2416_241602


namespace gcd_10010_15015_l2416_241624

theorem gcd_10010_15015 : Nat.gcd 10010 15015 = 5005 := by
  sorry

end gcd_10010_15015_l2416_241624


namespace remaining_concert_time_l2416_241657

def concert_duration : ℕ := 165 -- 2 hours and 45 minutes in minutes
def intermission1 : ℕ := 12
def intermission2 : ℕ := 10
def intermission3 : ℕ := 8
def regular_song_duration : ℕ := 4
def ballad_duration : ℕ := 7
def medley_duration : ℕ := 15
def num_regular_songs : ℕ := 15
def num_ballads : ℕ := 6

theorem remaining_concert_time : 
  concert_duration - 
  (intermission1 + intermission2 + intermission3 + 
   num_regular_songs * regular_song_duration + 
   num_ballads * ballad_duration + 
   medley_duration) = 18 := by sorry

end remaining_concert_time_l2416_241657


namespace simplify_expression_l2416_241642

theorem simplify_expression (x y : ℝ) : 
  (15 * x + 45 * y) + (20 * x + 58 * y) - (18 * x + 75 * y) = 17 * x + 28 * y := by
  sorry

end simplify_expression_l2416_241642


namespace chessboard_tromino_coverage_l2416_241647

/-- Represents a chessboard with alternating colors and black corners -/
structure Chessboard (n : ℕ) :=
  (is_odd : n % 2 = 1)
  (ge_seven : n ≥ 7)

/-- Calculates the number of black squares on the chessboard -/
def black_squares (board : Chessboard n) : ℕ :=
  (n^2 + 1) / 2

/-- Calculates the minimum number of trominos needed -/
def min_trominos (board : Chessboard n) : ℕ :=
  (n^2 + 1) / 6

theorem chessboard_tromino_coverage (n : ℕ) (board : Chessboard n) :
  (black_squares board) % 3 = 0 ∧
  min_trominos board = (n^2 + 1) / 6 :=
sorry

end chessboard_tromino_coverage_l2416_241647


namespace binomial_square_coefficient_l2416_241645

theorem binomial_square_coefficient (b : ℚ) : 
  (∃ t u : ℚ, ∀ x, bx^2 + 18*x + 16 = (t*x + u)^2) → b = 81/16 := by
sorry

end binomial_square_coefficient_l2416_241645


namespace range_of_x_l2416_241682

def p (x : ℝ) : Prop := x^2 - 5*x + 6 ≥ 0

def q (x : ℝ) : Prop := 0 < x ∧ x < 4

theorem range_of_x (x : ℝ) (h1 : p x ∨ q x) (h2 : ¬q x) :
  x ≤ 0 ∨ x ≥ 4 := by
  sorry

end range_of_x_l2416_241682


namespace a_equals_five_l2416_241689

/-- Given the equation 632 - A9B = 41, where A and B are single digits, prove that A must equal 5. -/
theorem a_equals_five (A B : ℕ) (h1 : A ≤ 9) (h2 : B ≤ 9) (h3 : 632 - (100 * A + 10 * B) = 41) : A = 5 := by
  sorry

end a_equals_five_l2416_241689


namespace functional_equation_solution_l2416_241688

-- Define the function type
def RealFunction := ℝ → ℝ

-- State the theorem
theorem functional_equation_solution (f : RealFunction)
  (h : ∀ x y : ℝ, f (f x + y) = 2 * x + f (f y - x)) :
  ∃ c : ℝ, ∀ x : ℝ, f x = x + c :=
sorry

end functional_equation_solution_l2416_241688


namespace files_remaining_l2416_241669

theorem files_remaining (music_files video_files deleted_files : ℕ) 
  (h1 : music_files = 13)
  (h2 : video_files = 30)
  (h3 : deleted_files = 10) :
  music_files + video_files - deleted_files = 33 := by
  sorry

end files_remaining_l2416_241669


namespace sum_of_four_primes_divisible_by_60_l2416_241699

theorem sum_of_four_primes_divisible_by_60 
  (p q r s : ℕ) 
  (hp : Nat.Prime p) 
  (hq : Nat.Prime q) 
  (hr : Nat.Prime r) 
  (hs : Nat.Prime s) 
  (h_order : 5 < p ∧ p < q ∧ q < r ∧ r < s ∧ s < p + 10) : 
  60 ∣ (p + q + r + s) := by
sorry

end sum_of_four_primes_divisible_by_60_l2416_241699


namespace opposite_numbers_sum_property_l2416_241655

theorem opposite_numbers_sum_property (a b : ℝ) : 
  (∃ k : ℝ, a = k ∧ b = -k) → -5 * (a + b) = 0 := by
sorry

end opposite_numbers_sum_property_l2416_241655


namespace desired_average_sale_l2416_241666

theorem desired_average_sale (sales : List ℕ) (desired_sixth : ℕ) : 
  sales = [6235, 6927, 6855, 7230, 6562] → 
  desired_sixth = 5191 → 
  (sales.sum + desired_sixth) / 6 = 6500 := by
  sorry

end desired_average_sale_l2416_241666


namespace exists_valid_strategy_l2416_241639

/-- Represents the problem of the father and two sons visiting their grandmother --/
structure VisitProblem where
  distance : ℝ
  scooter_speed_alone : ℝ
  scooter_speed_with_passenger : ℝ
  walking_speed : ℝ

/-- Defines the specific problem instance --/
def problem : VisitProblem :=
  { distance := 33
  , scooter_speed_alone := 25
  , scooter_speed_with_passenger := 20
  , walking_speed := 5
  }

/-- Represents a solution strategy for the visit problem --/
structure Strategy where
  (p : VisitProblem)
  travel_time : ℝ

/-- Predicate to check if a strategy is valid --/
def is_valid_strategy (s : Strategy) : Prop :=
  s.travel_time ≤ 3 ∧
  ∃ (t1 t2 t3 : ℝ),
    t1 ≤ s.travel_time ∧
    t2 ≤ s.travel_time ∧
    t3 ≤ s.travel_time ∧
    s.p.distance / s.p.walking_speed ≤ t1 ∧
    s.p.distance / s.p.walking_speed ≤ t2 ∧
    s.p.distance / s.p.scooter_speed_alone ≤ t3

/-- Theorem stating that there exists a valid strategy for the given problem --/
theorem exists_valid_strategy :
  ∃ (s : Strategy), s.p = problem ∧ is_valid_strategy s :=
sorry


end exists_valid_strategy_l2416_241639


namespace odd_integer_divisibility_l2416_241608

theorem odd_integer_divisibility (n : Int) (h : Odd n) :
  ∀ k : Nat, 2 ∣ k * (n - k) :=
sorry

end odd_integer_divisibility_l2416_241608


namespace josiah_cookie_expense_l2416_241680

/-- The amount Josiah spent on cookies in March -/
def cookie_expense : ℕ → ℕ → ℕ → ℕ
| cookies_per_day, cookie_cost, days_in_march =>
  cookies_per_day * cookie_cost * days_in_march

/-- Theorem: Josiah spent 992 dollars on cookies in March -/
theorem josiah_cookie_expense :
  cookie_expense 2 16 31 = 992 := by
  sorry

end josiah_cookie_expense_l2416_241680


namespace angle_relation_l2416_241661

theorem angle_relation (α β : Real) : 
  π / 2 < α ∧ α < π ∧
  π / 2 < β ∧ β < π ∧
  (1 - Real.cos (2 * α)) * (1 + Real.sin β) = Real.sin (2 * α) * Real.cos β →
  2 * α + β = 5 * π / 2 := by
sorry

end angle_relation_l2416_241661


namespace complex_power_four_l2416_241631

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- Theorem: (1+i)^4 = -4 -/
theorem complex_power_four : (1 + i)^4 = -4 := by sorry

end complex_power_four_l2416_241631


namespace apple_distribution_l2416_241629

theorem apple_distribution (total_apples : ℕ) (additional_people : ℕ) (apple_reduction : ℕ) :
  total_apples = 10000 →
  additional_people = 100 →
  apple_reduction = 15 →
  ∃ X : ℕ,
    (X * (total_apples / X) = total_apples) ∧
    ((X + additional_people) * (total_apples / X - apple_reduction) = total_apples) ∧
    X = 213 := by
  sorry

end apple_distribution_l2416_241629


namespace weight_replacement_l2416_241635

theorem weight_replacement (initial_total : ℝ) (replaced_weight : ℝ) :
  (∃ (average_increase : ℝ),
    average_increase = 1.5 ∧
    4 * (initial_total / 4 + average_increase) = initial_total - replaced_weight + 71) →
  replaced_weight = 65 := by
  sorry

end weight_replacement_l2416_241635


namespace lg_100_equals_2_l2416_241667

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem lg_100_equals_2 : lg 100 = 2 := by
  sorry

end lg_100_equals_2_l2416_241667


namespace car_speed_problem_l2416_241621

/-- Given a car's speed in the first hour and its average speed over two hours,
    calculate its speed in the second hour. -/
def second_hour_speed (first_hour_speed average_speed : ℝ) : ℝ :=
  2 * average_speed - first_hour_speed

/-- Theorem stating that if a car travels at 90 km/h for the first hour
    and has an average speed of 60 km/h over two hours,
    its speed in the second hour must be 30 km/h. -/
theorem car_speed_problem :
  second_hour_speed 90 60 = 30 := by
  sorry

#eval second_hour_speed 90 60

end car_speed_problem_l2416_241621


namespace fraction_product_simplification_l2416_241604

theorem fraction_product_simplification :
  (2 / 3) * (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 1 / 4 := by
  sorry

end fraction_product_simplification_l2416_241604


namespace total_legs_in_park_l2416_241607

/-- The total number of legs in a park with various animals, some with missing legs -/
def total_legs : ℕ :=
  let num_dogs : ℕ := 109
  let num_cats : ℕ := 37
  let num_birds : ℕ := 52
  let num_spiders : ℕ := 19
  let dogs_missing_legs : ℕ := 4
  let cats_missing_legs : ℕ := 3
  let spiders_missing_legs : ℕ := 2
  let dog_legs : ℕ := 4
  let cat_legs : ℕ := 4
  let bird_legs : ℕ := 2
  let spider_legs : ℕ := 8
  (num_dogs * dog_legs - dogs_missing_legs) +
  (num_cats * cat_legs - cats_missing_legs) +
  (num_birds * bird_legs) +
  (num_spiders * spider_legs - 2 * spiders_missing_legs)

theorem total_legs_in_park : total_legs = 829 := by
  sorry

end total_legs_in_park_l2416_241607


namespace octal_calculation_l2416_241612

/-- Represents a number in base 8 --/
def OctalNumber := ℕ

/-- Addition operation for octal numbers --/
def octal_add (a b : OctalNumber) : OctalNumber :=
  sorry

/-- Subtraction operation for octal numbers --/
def octal_sub (a b : OctalNumber) : OctalNumber :=
  sorry

/-- Conversion from decimal to octal --/
def to_octal (n : ℕ) : OctalNumber :=
  sorry

/-- Theorem: ($451_8 + 162_8) - 123_8 = 510_8$ in base 8 --/
theorem octal_calculation : 
  octal_sub (octal_add (to_octal 451) (to_octal 162)) (to_octal 123) = to_octal 510 :=
by sorry

end octal_calculation_l2416_241612


namespace fill_time_correct_l2416_241636

/-- The time in seconds for eight faucets to fill a 30-gallon tub, given that four faucets can fill a 120-gallon tub in 8 minutes -/
def fill_time : ℝ := 60

/-- The volume of the large tub in gallons -/
def large_tub_volume : ℝ := 120

/-- The volume of the small tub in gallons -/
def small_tub_volume : ℝ := 30

/-- The time in minutes for four faucets to fill the large tub -/
def large_tub_fill_time : ℝ := 8

/-- The number of faucets used to fill the large tub -/
def large_tub_faucets : ℕ := 4

/-- The number of faucets used to fill the small tub -/
def small_tub_faucets : ℕ := 8

/-- Conversion factor from minutes to seconds -/
def minutes_to_seconds : ℝ := 60

theorem fill_time_correct : fill_time = 
  (small_tub_volume / large_tub_volume) * 
  (large_tub_faucets / small_tub_faucets) * 
  large_tub_fill_time * 
  minutes_to_seconds := by
  sorry

end fill_time_correct_l2416_241636


namespace third_number_in_second_set_l2416_241650

theorem third_number_in_second_set (x y : ℝ) : 
  (28 + x + 42 + 78 + 104) / 5 = 90 →
  (128 + 255 + y + 1023 + x) / 5 = 423 →
  y = 511 := by sorry

end third_number_in_second_set_l2416_241650


namespace parallel_lines_condition_l2416_241686

/-- Two lines are parallel but not coincident -/
def parallel_not_coincident (a : ℝ) : Prop :=
  (a * 3 - 3 * (a - 1) = 0) ∧ (a * (a - 7) - 3 * (3 * a) ≠ 0)

/-- The condition a = 3 or a = -2 -/
def condition (a : ℝ) : Prop := a = 3 ∨ a = -2

theorem parallel_lines_condition :
  (∀ a : ℝ, parallel_not_coincident a → condition a) ∧
  ¬(∀ a : ℝ, condition a → parallel_not_coincident a) :=
sorry

end parallel_lines_condition_l2416_241686


namespace fred_gave_156_sheets_l2416_241656

/-- The number of sheets Fred gave to Charles -/
def sheets_given_to_charles (initial_sheets : ℕ) (received_sheets : ℕ) (final_sheets : ℕ) : ℕ :=
  initial_sheets + received_sheets - final_sheets

/-- Theorem stating that Fred gave 156 sheets to Charles -/
theorem fred_gave_156_sheets :
  sheets_given_to_charles 212 307 363 = 156 := by
  sorry

end fred_gave_156_sheets_l2416_241656


namespace lcm_nine_six_l2416_241637

theorem lcm_nine_six : Nat.lcm 9 6 = 18 := by
  sorry

end lcm_nine_six_l2416_241637


namespace arithmetic_sequence_third_term_l2416_241603

/-- Given an arithmetic sequence {a_n} where a_2 + a_4 = 5, prove that a_3 = 5/2 -/
theorem arithmetic_sequence_third_term 
  (a : ℕ → ℚ) -- a is a function from natural numbers to rationals
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) -- arithmetic sequence condition
  (h_sum : a 2 + a 4 = 5) -- given condition
  : a 3 = 5/2 := by
  sorry

end arithmetic_sequence_third_term_l2416_241603


namespace parallelogram_base_length_l2416_241614

-- Define the properties of the parallelogram
def parallelogram_area : ℝ := 200
def parallelogram_height : ℝ := 20

-- Theorem statement
theorem parallelogram_base_length :
  ∃ (base : ℝ), base * parallelogram_height = parallelogram_area ∧ base = 10 :=
by
  sorry

end parallelogram_base_length_l2416_241614


namespace pens_left_in_jar_l2416_241668

/-- Represents the number of pens of each color in the jar -/
structure PenCount where
  blue : ℕ
  black : ℕ
  red : ℕ
  green : ℕ
  purple : ℕ

def initial_pens : PenCount :=
  { blue := 15, black := 27, red := 12, green := 10, purple := 8 }

def removed_pens : PenCount :=
  { blue := 8, black := 9, red := 3, green := 5, purple := 6 }

def remaining_pens (initial : PenCount) (removed : PenCount) : PenCount :=
  { blue := initial.blue - removed.blue,
    black := initial.black - removed.black,
    red := initial.red - removed.red,
    green := initial.green - removed.green,
    purple := initial.purple - removed.purple }

def total_pens (pens : PenCount) : ℕ :=
  pens.blue + pens.black + pens.red + pens.green + pens.purple

theorem pens_left_in_jar :
  total_pens (remaining_pens initial_pens removed_pens) = 41 := by
  sorry

end pens_left_in_jar_l2416_241668


namespace plane_cost_calculation_l2416_241678

/-- The cost of taking a plane to the Island of Mysteries --/
def plane_cost : ℕ := 600

/-- The cost of taking a boat to the Island of Mysteries --/
def boat_cost : ℕ := 254

/-- The amount saved by taking the boat instead of the plane --/
def savings : ℕ := 346

/-- Theorem stating that the plane cost is equal to the boat cost plus the savings --/
theorem plane_cost_calculation : plane_cost = boat_cost + savings := by
  sorry

end plane_cost_calculation_l2416_241678


namespace problem_1_l2416_241638

theorem problem_1 (α : Real) (h : 2 * Real.sin α - Real.cos α = 0) : 
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) + 
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = -10/3 := by
sorry

end problem_1_l2416_241638


namespace complex_fraction_equality_l2416_241663

theorem complex_fraction_equality : Complex.I * 2 / (1 + Complex.I) = 1 + Complex.I := by
  sorry

end complex_fraction_equality_l2416_241663


namespace real_estate_investment_l2416_241633

theorem real_estate_investment
  (total_investment : ℝ)
  (real_estate_ratio : ℝ)
  (h1 : total_investment = 250000)
  (h2 : real_estate_ratio = 3)
  : real_estate_ratio * (total_investment / (1 + real_estate_ratio)) = 187500 :=
by
  sorry

end real_estate_investment_l2416_241633
