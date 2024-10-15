import Mathlib

namespace NUMINAMATH_CALUDE_metro_earnings_l4127_412734

/-- Calculates the earnings from ticket sales over a given period of time. -/
def calculate_earnings (ticket_cost : ℕ) (tickets_per_minute : ℕ) (minutes : ℕ) : ℕ :=
  ticket_cost * tickets_per_minute * minutes

/-- Proves that the earnings from ticket sales in 6 minutes is $90,
    given the ticket cost and average tickets sold per minute. -/
theorem metro_earnings :
  calculate_earnings 3 5 6 = 90 := by
  sorry

end NUMINAMATH_CALUDE_metro_earnings_l4127_412734


namespace NUMINAMATH_CALUDE_bridge_length_l4127_412768

/-- The length of a bridge given train parameters and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 110 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600) * crossing_time - train_length = 265 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l4127_412768


namespace NUMINAMATH_CALUDE_thalassa_population_2050_l4127_412784

/-- The population growth factor for Thalassa every 30 years -/
def growth_factor : ℕ := 3

/-- The initial population of Thalassa in 1990 -/
def initial_population : ℕ := 300

/-- The number of 30-year periods between 1990 and 2050 -/
def num_periods : ℕ := 2

/-- The population of Thalassa in 2050 -/
def population_2050 : ℕ := initial_population * growth_factor ^ num_periods

theorem thalassa_population_2050 : population_2050 = 2700 := by
  sorry

end NUMINAMATH_CALUDE_thalassa_population_2050_l4127_412784


namespace NUMINAMATH_CALUDE_prime_equation_solutions_l4127_412777

theorem prime_equation_solutions (p : ℕ) (hp : Prime p) (hp_mod : p % 8 = 3) :
  ∀ (x y : ℚ), p^2 * x^4 - 6*p*x^2 + 1 = y^2 ↔ (x = 0 ∧ (y = 1 ∨ y = -1)) :=
by sorry

end NUMINAMATH_CALUDE_prime_equation_solutions_l4127_412777


namespace NUMINAMATH_CALUDE_book_selling_price_l4127_412765

theorem book_selling_price (CP : ℝ) : 
  (0.9 * CP = CP - 0.1 * CP) →  -- 10% loss condition
  (1.1 * CP = 990) →            -- 10% gain condition
  (0.9 * CP = 810) :=           -- Original selling price
by sorry

end NUMINAMATH_CALUDE_book_selling_price_l4127_412765


namespace NUMINAMATH_CALUDE_simple_interest_rate_l4127_412735

/-- Simple interest calculation --/
theorem simple_interest_rate (P : ℝ) (t : ℝ) (I : ℝ) : 
  P = 450 →
  t = 8 →
  I = P - 306 →
  I = P * (4 / 100) * t :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_rate_l4127_412735


namespace NUMINAMATH_CALUDE_expression_evaluation_l4127_412751

theorem expression_evaluation : 4 * (9 - 6) / 2 - 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4127_412751


namespace NUMINAMATH_CALUDE_complex_magnitude_equality_l4127_412744

theorem complex_magnitude_equality (t : ℝ) : 
  t > 0 → (Complex.abs (-5 + t * Complex.I) = 3 * Real.sqrt 6 ↔ t = Real.sqrt 29) := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equality_l4127_412744


namespace NUMINAMATH_CALUDE_exact_four_white_probability_l4127_412711

-- Define the number of balls
def n : ℕ := 8

-- Define the probability of a ball being white (or black)
def p : ℚ := 1/2

-- Define the number of white balls we're interested in
def k : ℕ := 4

-- State the theorem
theorem exact_four_white_probability :
  (n.choose k) * p^k * (1 - p)^(n - k) = 35/128 := by
  sorry

end NUMINAMATH_CALUDE_exact_four_white_probability_l4127_412711


namespace NUMINAMATH_CALUDE_parallel_vectors_angle_l4127_412790

theorem parallel_vectors_angle (α : ℝ) 
  (h_acute : 0 < α ∧ α < π / 2)
  (h_parallel : (3/2, Real.sin α) = (Real.cos α * k, 1/3 * k) → k ≠ 0) :
  α = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_angle_l4127_412790


namespace NUMINAMATH_CALUDE_cubic_identity_l4127_412718

theorem cubic_identity (a b c : ℝ) :
  a^3 * (b^3 - c^3) + b^3 * (c^3 - a^3) + c^3 * (a^3 - b^3) =
  (a - b) * (b - c) * (c - a) * (a^2 + b^2 + c^2 + a*b + b*c + c*a) := by
  sorry

end NUMINAMATH_CALUDE_cubic_identity_l4127_412718


namespace NUMINAMATH_CALUDE_van_distance_theorem_l4127_412708

theorem van_distance_theorem (initial_time : ℝ) (speed : ℝ) :
  initial_time = 6 →
  speed = 30 →
  (3 / 2 : ℝ) * initial_time * speed = 270 :=
by
  sorry

end NUMINAMATH_CALUDE_van_distance_theorem_l4127_412708


namespace NUMINAMATH_CALUDE_sculpture_height_l4127_412729

/-- Converts feet to inches -/
def feet_to_inches (feet : ℝ) : ℝ := feet * 12

theorem sculpture_height :
  let base_height : ℝ := 2
  let total_height_feet : ℝ := 3
  let total_height_inches : ℝ := feet_to_inches total_height_feet
  let sculpture_height : ℝ := total_height_inches - base_height
  sculpture_height = 34 := by
  sorry

end NUMINAMATH_CALUDE_sculpture_height_l4127_412729


namespace NUMINAMATH_CALUDE_optimal_sampling_methods_l4127_412700

/-- Represents different sampling methods --/
inductive SamplingMethod
  | Random
  | Systematic
  | Stratified

/-- Represents income levels --/
inductive IncomeLevel
  | High
  | Middle
  | Low

/-- Represents a community with families of different income levels --/
structure Community where
  totalFamilies : ℕ
  highIncomeFamilies : ℕ
  middleIncomeFamilies : ℕ
  lowIncomeFamilies : ℕ

/-- Represents a group of volleyball players --/
structure VolleyballTeam where
  totalPlayers : ℕ

/-- Determines the optimal sampling method for a given community and sample size --/
def optimalSamplingMethodForCommunity (c : Community) (sampleSize : ℕ) : SamplingMethod :=
  sorry

/-- Determines the optimal sampling method for a volleyball team and selection size --/
def optimalSamplingMethodForTeam (t : VolleyballTeam) (selectionSize : ℕ) : SamplingMethod :=
  sorry

/-- The main theorem stating the optimal sampling methods for the given scenarios --/
theorem optimal_sampling_methods 
  (community : Community)
  (team : VolleyballTeam)
  (h1 : community.totalFamilies = 400)
  (h2 : community.highIncomeFamilies = 120)
  (h3 : community.middleIncomeFamilies = 180)
  (h4 : community.lowIncomeFamilies = 100)
  (h5 : team.totalPlayers = 12) :
  (optimalSamplingMethodForCommunity community 100 = SamplingMethod.Stratified) ∧
  (optimalSamplingMethodForTeam team 3 = SamplingMethod.Random) :=
sorry

end NUMINAMATH_CALUDE_optimal_sampling_methods_l4127_412700


namespace NUMINAMATH_CALUDE_cook_remaining_potatoes_l4127_412733

/-- Given a chef needs to cook potatoes with the following conditions:
  * The total number of potatoes to cook
  * The number of potatoes already cooked
  * The time it takes to cook each potato
  This function calculates the time required to cook the remaining potatoes. -/
def time_to_cook_remaining (total : ℕ) (cooked : ℕ) (time_per_potato : ℕ) : ℕ :=
  (total - cooked) * time_per_potato

/-- Theorem stating that it takes 36 minutes to cook the remaining potatoes. -/
theorem cook_remaining_potatoes :
  time_to_cook_remaining 12 6 6 = 36 := by
  sorry

end NUMINAMATH_CALUDE_cook_remaining_potatoes_l4127_412733


namespace NUMINAMATH_CALUDE_student_teacher_ratio_l4127_412742

/-- Proves that the current ratio of students to teachers is 50:1 given the problem conditions -/
theorem student_teacher_ratio 
  (current_teachers : ℕ) 
  (current_students : ℕ) 
  (h1 : current_teachers = 3)
  (h2 : (current_students + 50) / (current_teachers + 5) = 25) :
  current_students / current_teachers = 50 := by
sorry

end NUMINAMATH_CALUDE_student_teacher_ratio_l4127_412742


namespace NUMINAMATH_CALUDE_simplify_expression_l4127_412787

theorem simplify_expression :
  4 * (12 / 9) * (36 / -45) = -12 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l4127_412787


namespace NUMINAMATH_CALUDE_solid_with_rectangular_views_is_cuboid_l4127_412719

/-- A solid is a three-dimensional geometric object. -/
structure Solid :=
  (shape : Type)

/-- A view is a two-dimensional projection of a solid. -/
inductive View
  | Front
  | Top
  | Side

/-- A rectangle is a quadrilateral with four right angles. -/
structure Rectangle :=
  (width : ℝ)
  (height : ℝ)

/-- A cuboid is a three-dimensional solid with six rectangular faces. -/
structure Cuboid :=
  (length : ℝ)
  (width : ℝ)
  (height : ℝ)

/-- The projection of a solid onto a view. -/
def projection (s : Solid) (v : View) : Type :=
  sorry

/-- Theorem: If a solid's three views are all rectangles, then the solid is a cuboid. -/
theorem solid_with_rectangular_views_is_cuboid (s : Solid) :
  (∀ v : View, projection s v = Rectangle) → (s.shape = Cuboid) :=
sorry

end NUMINAMATH_CALUDE_solid_with_rectangular_views_is_cuboid_l4127_412719


namespace NUMINAMATH_CALUDE_certain_number_is_six_l4127_412702

theorem certain_number_is_six : ∃ x : ℝ, 7 * x - 6 = 4 * x + 12 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_is_six_l4127_412702


namespace NUMINAMATH_CALUDE_franklin_valentines_l4127_412713

/-- The number of Valentines Mrs. Franklin gave to her students -/
def valentines_given (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

/-- Theorem stating that Mrs. Franklin gave 42 Valentines to her students -/
theorem franklin_valentines : valentines_given 58 16 = 42 := by
  sorry

end NUMINAMATH_CALUDE_franklin_valentines_l4127_412713


namespace NUMINAMATH_CALUDE_sum_of_two_arithmetic_sequences_l4127_412728

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : List ℕ :=
  List.range n |>.map (λ i => a₁ + i * d)

theorem sum_of_two_arithmetic_sequences :
  let seq1 := arithmetic_sequence 1 10 5
  let seq2 := arithmetic_sequence 9 10 5
  List.sum seq1 + List.sum seq2 = 250 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_arithmetic_sequences_l4127_412728


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l4127_412759

theorem quadratic_equation_solution :
  let a : ℚ := -2
  let b : ℚ := 1
  let c : ℚ := 3
  let x₁ : ℚ := -1
  let x₂ : ℚ := 3/2
  (a * x₁^2 + b * x₁ + c = 0) ∧ (a * x₂^2 + b * x₂ + c = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l4127_412759


namespace NUMINAMATH_CALUDE_min_participants_l4127_412727

/-- Represents a participant in the race -/
structure Participant where
  name : String
  position : Nat

/-- Represents the race with its participants -/
structure Race where
  participants : List Participant

/-- Checks if the race satisfies the given conditions -/
def satisfiesConditions (race : Race) : Prop :=
  ∃ (andrei dima lenya : Participant),
    andrei ∈ race.participants ∧
    dima ∈ race.participants ∧
    lenya ∈ race.participants ∧
    (∀ p1 p2 : Participant, p1 ∈ race.participants → p2 ∈ race.participants → p1 ≠ p2 → p1.position ≠ p2.position) ∧
    (2 * (andrei.position - 1) = race.participants.length - andrei.position) ∧
    (3 * (dima.position - 1) = race.participants.length - dima.position) ∧
    (4 * (lenya.position - 1) = race.participants.length - lenya.position)

/-- The theorem stating the minimum number of participants -/
theorem min_participants : ∀ race : Race, satisfiesConditions race → race.participants.length ≥ 61 := by
  sorry

end NUMINAMATH_CALUDE_min_participants_l4127_412727


namespace NUMINAMATH_CALUDE_dividend_percentage_calculation_l4127_412792

/-- Calculates the dividend percentage given investment details --/
theorem dividend_percentage_calculation
  (investment : ℝ)
  (share_face_value : ℝ)
  (premium_rate : ℝ)
  (total_dividend : ℝ)
  (h1 : investment = 14400)
  (h2 : share_face_value = 100)
  (h3 : premium_rate = 0.2)
  (h4 : total_dividend = 600) :
  let share_cost := share_face_value * (1 + premium_rate)
  let num_shares := investment / share_cost
  let dividend_per_share := total_dividend / num_shares
  let dividend_percentage := (dividend_per_share / share_face_value) * 100
  dividend_percentage = 5 := by
sorry


end NUMINAMATH_CALUDE_dividend_percentage_calculation_l4127_412792


namespace NUMINAMATH_CALUDE_first_quarter_profit_determination_l4127_412798

/-- Represents the quarterly profits of a store in dollars. -/
structure QuarterlyProfits where
  first : ℕ
  third : ℕ
  fourth : ℕ

/-- Calculates the annual profit given quarterly profits. -/
def annualProfit (q : QuarterlyProfits) : ℕ :=
  q.first + q.third + q.fourth

/-- Theorem stating that given the annual profit and profits from the third and fourth quarters,
    the first quarter profit can be determined. -/
theorem first_quarter_profit_determination
  (annual_profit : ℕ)
  (third_quarter : ℕ)
  (fourth_quarter : ℕ)
  (h1 : third_quarter = 3000)
  (h2 : fourth_quarter = 2000)
  (h3 : annual_profit = 8000)
  (h4 : ∃ q : QuarterlyProfits, q.third = third_quarter ∧ q.fourth = fourth_quarter ∧ annualProfit q = annual_profit) :
  ∃ q : QuarterlyProfits, q.first = 3000 ∧ q.third = third_quarter ∧ q.fourth = fourth_quarter ∧ annualProfit q = annual_profit :=
by sorry

end NUMINAMATH_CALUDE_first_quarter_profit_determination_l4127_412798


namespace NUMINAMATH_CALUDE_lawn_mowing_problem_l4127_412732

-- Define the rates and time
def mary_rate : ℚ := 1 / 3
def tom_rate : ℚ := 1 / 5
def work_time : ℚ := 3 / 2

-- Define the theorem
theorem lawn_mowing_problem :
  let combined_rate := mary_rate + tom_rate
  let mowed_fraction := work_time * combined_rate
  1 - mowed_fraction = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_lawn_mowing_problem_l4127_412732


namespace NUMINAMATH_CALUDE_overtime_hours_calculation_l4127_412709

/-- Calculates the number of overtime hours worked by an employee given their gross pay and pay rates. -/
theorem overtime_hours_calculation (regular_rate overtime_rate gross_pay : ℚ) : 
  regular_rate = 11.25 →
  overtime_rate = 16 →
  gross_pay = 622 →
  ∃ (overtime_hours : ℕ), 
    overtime_hours = 11 ∧ 
    gross_pay = (40 * regular_rate) + (overtime_hours : ℚ) * overtime_rate :=
by sorry

end NUMINAMATH_CALUDE_overtime_hours_calculation_l4127_412709


namespace NUMINAMATH_CALUDE_parabola_shift_l4127_412762

/-- The original parabola function -/
def original_parabola (x : ℝ) : ℝ := x^2 - 6*x + 5

/-- The shifted parabola function -/
def shifted_parabola (x : ℝ) : ℝ := (x-4)^2 - 2

/-- Theorem stating that the shifted parabola is equivalent to 
    shifting the original parabola 1 unit right and 2 units up -/
theorem parabola_shift :
  ∀ x : ℝ, shifted_parabola x = original_parabola (x - 1) + 2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_shift_l4127_412762


namespace NUMINAMATH_CALUDE_constant_term_expansion_l4127_412778

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the function for the expansion
def expansion_term (x : ℝ) (r : ℕ) : ℝ :=
  (-1)^r * binomial 16 r * x^(16 - 4*r/3)

-- State the theorem
theorem constant_term_expansion :
  expansion_term 1 12 = 1820 :=
sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l4127_412778


namespace NUMINAMATH_CALUDE_derivative_f_at_zero_l4127_412707

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then 3^(x^2 * Real.sin (2/x)) - 1 + 2*x else 0

theorem derivative_f_at_zero : 
  deriv f 0 = -2 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_zero_l4127_412707


namespace NUMINAMATH_CALUDE_pentagon_perimeter_is_40_l4127_412701

/-- An irregular pentagon with given side lengths -/
structure IrregularPentagon where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  side5 : ℝ

/-- The perimeter of an irregular pentagon -/
def perimeter (p : IrregularPentagon) : ℝ :=
  p.side1 + p.side2 + p.side3 + p.side4 + p.side5

/-- Theorem: The perimeter of the given irregular pentagon is 40 -/
theorem pentagon_perimeter_is_40 :
  let p : IrregularPentagon := {
    side1 := 6,
    side2 := 7,
    side3 := 8,
    side4 := 9,
    side5 := 10
  }
  perimeter p = 40 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_perimeter_is_40_l4127_412701


namespace NUMINAMATH_CALUDE_brendan_taxes_l4127_412705

/-- Calculates the taxes paid by a waiter named Brendan based on his work schedule and income. -/
theorem brendan_taxes : 
  let hourly_wage : ℚ := 6
  let shifts_8hour : ℕ := 2
  let shifts_12hour : ℕ := 1
  let hourly_tips : ℚ := 12
  let tax_rate : ℚ := 1/5
  let reported_tips_fraction : ℚ := 1/3
  
  let total_hours : ℕ := shifts_8hour * 8 + shifts_12hour * 12
  let wage_income : ℚ := hourly_wage * total_hours
  let total_tips : ℚ := hourly_tips * total_hours
  let reported_tips : ℚ := total_tips * reported_tips_fraction
  let reported_income : ℚ := wage_income + reported_tips
  let taxes_paid : ℚ := reported_income * tax_rate

  taxes_paid = 56 := by sorry

end NUMINAMATH_CALUDE_brendan_taxes_l4127_412705


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_factor_difference_of_cubes_l4127_412722

-- Theorem 1
theorem perfect_square_trinomial (m : ℝ) : m^2 - 10*m + 25 = (m - 5)^2 := by
  sorry

-- Theorem 2
theorem factor_difference_of_cubes (a b : ℝ) : a^3*b - a*b = a*b*(a + 1)*(a - 1) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_factor_difference_of_cubes_l4127_412722


namespace NUMINAMATH_CALUDE_joe_fruit_probability_l4127_412724

def num_fruit_types : ℕ := 4
def num_meals : ℕ := 3

def probability_same_fruit : ℚ := (1 / num_fruit_types) ^ num_meals * num_fruit_types

theorem joe_fruit_probability :
  1 - probability_same_fruit = 15/16 := by sorry

end NUMINAMATH_CALUDE_joe_fruit_probability_l4127_412724


namespace NUMINAMATH_CALUDE_guppies_count_l4127_412717

/-- The number of Goldfish -/
def num_goldfish : ℕ := 2

/-- The amount of food each Goldfish gets (in teaspoons) -/
def goldfish_food : ℚ := 1

/-- The number of Swordtails -/
def num_swordtails : ℕ := 3

/-- The amount of food each Swordtail gets (in teaspoons) -/
def swordtail_food : ℚ := 2

/-- The amount of food each Guppy gets (in teaspoons) -/
def guppy_food : ℚ := 1/2

/-- The total amount of food given to all fish (in teaspoons) -/
def total_food : ℚ := 12

/-- The number of Guppies Layla has -/
def num_guppies : ℕ := 8

theorem guppies_count :
  (num_goldfish : ℚ) * goldfish_food +
  (num_swordtails : ℚ) * swordtail_food +
  (num_guppies : ℚ) * guppy_food = total_food :=
by sorry

end NUMINAMATH_CALUDE_guppies_count_l4127_412717


namespace NUMINAMATH_CALUDE_determine_back_iff_conditions_met_l4127_412757

/-- Represents a card with two sides -/
structure Card where
  side1 : Nat
  side2 : Nat

/-- Checks if a number appears on a card -/
def numberOnCard (c : Card) (n : Nat) : Prop :=
  c.side1 = n ∨ c.side2 = n

/-- Represents the deck of n cards -/
def deck (n : Nat) : List Card :=
  List.range n |>.map (λ i => ⟨i, i + 1⟩)

/-- Represents the cards seen so far -/
def SeenCards := List Nat

/-- Determines if the back of the last card can be identified -/
def canDetermineBack (n : Nat) (k : Nat) (seen : SeenCards) : Prop :=
  (k = 0 ∨ k = n) ∨
  (0 < k ∧ k < n ∧
    (seen.count (k + 1) = 2 ∨
     (∃ j, 1 ≤ j ∧ j ≤ n - k - 1 ∧
       (∀ i, k + 1 ≤ i ∧ i ≤ k + j → seen.count i ≥ 1) ∧
       (if k + j + 1 = n then seen.count n ≥ 1 else seen.count (k + j + 1) = 2)) ∨
     seen.count (k - 1) = 2 ∨
     (∃ j, 1 ≤ j ∧ j ≤ k - 1 ∧
       (∀ i, k - j ≤ i ∧ i ≤ k - 1 → seen.count i ≥ 1) ∧
       (if k - j - 1 = 0 then seen.count 0 ≥ 1 else seen.count (k - j - 1) = 2))))

/-- The main theorem to be proved -/
theorem determine_back_iff_conditions_met (n : Nat) (k : Nat) (seen : SeenCards) :
  canDetermineBack n k seen ↔
  (∀ (lastCard : Card),
    numberOnCard lastCard k →
    lastCard ∈ deck n →
    ∃! backNumber, numberOnCard lastCard backNumber ∧ backNumber ≠ k) := by
  sorry

end NUMINAMATH_CALUDE_determine_back_iff_conditions_met_l4127_412757


namespace NUMINAMATH_CALUDE_two_digit_number_problem_l4127_412761

theorem two_digit_number_problem :
  ∀ (x y : ℕ),
    x < 10 ∧ y < 10 ∧  -- Ensures x and y are single digits
    y = x + 2 ∧        -- Unit's digit exceeds 10's digit by 2
    (10 * x + y) * (x + y) = 144  -- Product condition
    → 10 * x + y = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_problem_l4127_412761


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l4127_412763

theorem quadratic_real_roots_condition (k : ℝ) : 
  (∃ x : ℝ, k * x^2 - 2 * x + 3 = 0) ↔ (k ≤ 1/3 ∧ k ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l4127_412763


namespace NUMINAMATH_CALUDE_red_candy_count_l4127_412770

theorem red_candy_count (total : ℕ) (blue : ℕ) (h1 : total = 3409) (h2 : blue = 3264) :
  total - blue = 145 := by
  sorry

end NUMINAMATH_CALUDE_red_candy_count_l4127_412770


namespace NUMINAMATH_CALUDE_scientific_notation_of_small_number_l4127_412764

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  significand : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |significand| ∧ |significand| < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_small_number :
  toScientificNotation 0.00000002 = ScientificNotation.mk 2 (-8) sorry := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_small_number_l4127_412764


namespace NUMINAMATH_CALUDE_train_crossing_time_l4127_412769

/-- Proves that a train 40 meters long, traveling at 144 km/hr, will take 1 second to cross an electric pole. -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 40 →
  train_speed_kmh = 144 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 1 := by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l4127_412769


namespace NUMINAMATH_CALUDE_journey_distance_is_correct_l4127_412703

/-- Represents the cab fare structure and journey details -/
structure CabJourney where
  baseFare : ℝ
  peakRateFirst2Miles : ℝ
  peakRateAfter2Miles : ℝ
  toll1 : ℝ
  toll2 : ℝ
  tipPercentage : ℝ
  totalPaid : ℝ

/-- Calculates the distance of the journey based on the given fare structure and total paid -/
def calculateDistance (journey : CabJourney) : ℝ :=
  sorry

/-- Theorem stating that the calculated distance for the given journey is 6.58 miles -/
theorem journey_distance_is_correct (journey : CabJourney) 
  (h1 : journey.baseFare = 3)
  (h2 : journey.peakRateFirst2Miles = 5)
  (h3 : journey.peakRateAfter2Miles = 4)
  (h4 : journey.toll1 = 1.5)
  (h5 : journey.toll2 = 2.5)
  (h6 : journey.tipPercentage = 0.15)
  (h7 : journey.totalPaid = 39.57) :
  calculateDistance journey = 6.58 := by
  sorry

end NUMINAMATH_CALUDE_journey_distance_is_correct_l4127_412703


namespace NUMINAMATH_CALUDE_weight_of_new_person_l4127_412746

/-- Given a group of 8 people, if replacing a 65 kg person with a new person
    increases the average weight by 1.5 kg, then the weight of the new person is 77 kg. -/
theorem weight_of_new_person
  (initial_count : ℕ)
  (weight_replaced : ℝ)
  (avg_increase : ℝ)
  (h_count : initial_count = 8)
  (h_replaced : weight_replaced = 65)
  (h_increase : avg_increase = 1.5) :
  weight_replaced + initial_count * avg_increase = 77 :=
by sorry

end NUMINAMATH_CALUDE_weight_of_new_person_l4127_412746


namespace NUMINAMATH_CALUDE_orchard_sections_count_l4127_412756

/-- The number of sacks harvested from each section daily -/
def sacks_per_section : ℕ := 45

/-- The total number of sacks harvested daily -/
def total_sacks : ℕ := 360

/-- The number of sections in the orchard -/
def num_sections : ℕ := total_sacks / sacks_per_section

theorem orchard_sections_count :
  num_sections = 8 :=
sorry

end NUMINAMATH_CALUDE_orchard_sections_count_l4127_412756


namespace NUMINAMATH_CALUDE_min_c_value_l4127_412725

theorem min_c_value (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_order : a < b ∧ b < c) (h_sum : a + b + c = 2010)
  (h_unique : ∃! (x y : ℝ), 3 * x + y = 3005 ∧ y = |x - a| + |x - 2*b| + |x - c|) :
  c ≥ 1014 :=
sorry

end NUMINAMATH_CALUDE_min_c_value_l4127_412725


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l4127_412793

theorem polynomial_divisibility (r : ℝ) :
  (∃ s : ℝ, 10 * X^3 - 5 * X^2 - 52 * X + 60 = 10 * (X - r)^2 * (X - s)) →
  r = -3/2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l4127_412793


namespace NUMINAMATH_CALUDE_sum_of_digits_of_B_is_seven_l4127_412749

-- Define the function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Define A as the sum of digits of 4444^4144
def A : ℕ := sumOfDigits (4444^4144)

-- Define B as the sum of digits of A
def B : ℕ := sumOfDigits A

-- Theorem to prove
theorem sum_of_digits_of_B_is_seven : sumOfDigits B = 7 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_B_is_seven_l4127_412749


namespace NUMINAMATH_CALUDE_pc_length_l4127_412780

/-- A convex quadrilateral with specific properties -/
structure SpecialQuadrilateral where
  -- Points of the quadrilateral
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  -- Point P on diagonal AC
  P : ℝ × ℝ
  -- Convexity condition (simplified)
  convex : True
  -- CD perpendicular to AC
  cd_perp_ac : (C.1 - D.1) * (C.1 - A.1) + (C.2 - D.2) * (C.2 - A.2) = 0
  -- AB perpendicular to BD
  ab_perp_bd : (A.1 - B.1) * (B.1 - D.1) + (A.2 - B.2) * (B.2 - D.2) = 0
  -- CD length
  cd_length : Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 72
  -- AB length
  ab_length : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 35
  -- P on AC
  p_on_ac : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (A.1 + t * (C.1 - A.1), A.2 + t * (C.2 - A.2))
  -- BP perpendicular to AD
  bp_perp_ad : (B.1 - P.1) * (A.1 - D.1) + (B.2 - P.2) * (A.2 - D.2) = 0
  -- AP length
  ap_length : Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2) = 15

/-- The main theorem stating that PC = 72.5 in the special quadrilateral -/
theorem pc_length (q : SpecialQuadrilateral) : 
  Real.sqrt ((q.P.1 - q.C.1)^2 + (q.P.2 - q.C.2)^2) = 72.5 := by
  sorry

end NUMINAMATH_CALUDE_pc_length_l4127_412780


namespace NUMINAMATH_CALUDE_complement_of_union_l4127_412789

-- Define the sets A and B
def A : Set ℝ := {x | x < 0}
def B : Set ℝ := {x | x ≥ 2}

-- Define the set C as the complement of A ∪ B in ℝ
def C : Set ℝ := (A ∪ B)ᶜ

-- Theorem statement
theorem complement_of_union :
  C = {x : ℝ | 0 ≤ x ∧ x < 2} :=
sorry

end NUMINAMATH_CALUDE_complement_of_union_l4127_412789


namespace NUMINAMATH_CALUDE_f_of_one_eq_six_l4127_412754

def f (x : ℝ) : ℝ := x^2 + 2*x + 3

theorem f_of_one_eq_six : f 1 = 6 := by
  sorry

end NUMINAMATH_CALUDE_f_of_one_eq_six_l4127_412754


namespace NUMINAMATH_CALUDE_d_share_is_750_l4127_412799

/-- Represents the share of money for each person -/
structure Share :=
  (amount : ℝ)

/-- Represents the distribution of money among 5 people -/
structure Distribution :=
  (a b c d e : Share)

/-- The total amount of money to be distributed -/
def total_amount (dist : Distribution) : ℝ :=
  dist.a.amount + dist.b.amount + dist.c.amount + dist.d.amount + dist.e.amount

/-- The condition that the distribution follows the proportion 5 : 2 : 4 : 3 : 1 -/
def proportional_distribution (dist : Distribution) : Prop :=
  5 * dist.b.amount = 2 * dist.a.amount ∧
  5 * dist.c.amount = 4 * dist.a.amount ∧
  5 * dist.d.amount = 3 * dist.a.amount ∧
  5 * dist.e.amount = 1 * dist.a.amount

/-- The condition that the combined share of A and C is 3/5 of the total amount -/
def combined_share_condition (dist : Distribution) : Prop :=
  dist.a.amount + dist.c.amount = 3/5 * total_amount dist

/-- The condition that E gets $250 less than B -/
def e_less_than_b_condition (dist : Distribution) : Prop :=
  dist.b.amount - dist.e.amount = 250

theorem d_share_is_750 (dist : Distribution) 
  (h1 : proportional_distribution dist)
  (h2 : combined_share_condition dist)
  (h3 : e_less_than_b_condition dist) :
  dist.d.amount = 750 := by
  sorry

end NUMINAMATH_CALUDE_d_share_is_750_l4127_412799


namespace NUMINAMATH_CALUDE_prop_2_prop_4_prop_1_counter_prop_3_counter_l4127_412739

-- Define basic geometric concepts
def Line : Type := sorry
def Plane : Type := sorry
def Point : Type := sorry

-- Define geometric relations
def parallel (a b : Plane) : Prop := sorry
def perpendicular (a b : Plane) : Prop := sorry
def perpendicular_line_plane (l : Line) (p : Plane) : Prop := sorry
def line_in_plane (l : Line) (p : Plane) : Prop := sorry
def line_of_intersection (p1 p2 : Plane) : Line := sorry

-- Proposition 2
theorem prop_2 (p1 p2 : Plane) (l : Line) : 
  perpendicular_line_plane l p1 → line_in_plane l p2 → perpendicular p1 p2 := by sorry

-- Proposition 4
theorem prop_4 (p1 p2 : Plane) (l : Line) :
  perpendicular p1 p2 → 
  line_in_plane l p1 → 
  ¬perpendicular_line_plane l (line_of_intersection p1 p2) → 
  ¬perpendicular_line_plane l p2 := by sorry

-- Proposition 1 (counterexample)
theorem prop_1_counter : ∃ (p1 p2 p3 : Plane) (l1 l2 : Line),
  line_in_plane l1 p1 ∧ line_in_plane l2 p1 ∧
  parallel p2 p1 ∧ parallel p3 p1 ∧
  ¬parallel p2 p3 := by sorry

-- Proposition 3 (counterexample)
theorem prop_3_counter : ∃ (l1 l2 l3 : Line),
  perpendicular_line_plane l1 l3 ∧ 
  perpendicular_line_plane l2 l3 ∧
  ¬parallel l1 l2 := by sorry

end NUMINAMATH_CALUDE_prop_2_prop_4_prop_1_counter_prop_3_counter_l4127_412739


namespace NUMINAMATH_CALUDE_largest_whole_number_nine_times_less_than_200_l4127_412743

theorem largest_whole_number_nine_times_less_than_200 :
  ∀ x : ℕ, x ≤ 22 ↔ 9 * x < 200 :=
by sorry

end NUMINAMATH_CALUDE_largest_whole_number_nine_times_less_than_200_l4127_412743


namespace NUMINAMATH_CALUDE_equal_area_necessary_not_sufficient_l4127_412783

-- Define a triangle type
structure Triangle where
  -- You might add more specific properties here, but for this problem we only need area
  area : ℝ

-- Define congruence for triangles
def congruent (t1 t2 : Triangle) : Prop := sorry

-- Theorem statement
theorem equal_area_necessary_not_sufficient :
  (∀ t1 t2 : Triangle, congruent t1 t2 → t1.area = t2.area) ∧
  (∃ t1 t2 : Triangle, t1.area = t2.area ∧ ¬congruent t1 t2) := by
  sorry

end NUMINAMATH_CALUDE_equal_area_necessary_not_sufficient_l4127_412783


namespace NUMINAMATH_CALUDE_find_w_l4127_412720

theorem find_w : ∃ w : ℝ, ((2^5 : ℝ) * (9^2)) / ((8^2) * w) = 0.16666666666666666 ∧ w = 243 := by
  sorry

end NUMINAMATH_CALUDE_find_w_l4127_412720


namespace NUMINAMATH_CALUDE_one_solution_less_than_two_l4127_412730

def f (x : ℝ) : ℝ := x^8 + 6*x^7 + 14*x^6 + 1429*x^5 - 1279*x^4

theorem one_solution_less_than_two :
  ∃! x : ℝ, 0 < x ∧ x < 2 ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_one_solution_less_than_two_l4127_412730


namespace NUMINAMATH_CALUDE_coin_toss_is_classical_model_l4127_412767

structure Experiment where
  name : String
  is_finite : Bool
  is_equiprobable : Bool

def is_classical_probability_model (e : Experiment) : Prop :=
  e.is_finite ∧ e.is_equiprobable

def seed_germination : Experiment :=
  { name := "Seed germination",
    is_finite := true,
    is_equiprobable := false }

def product_measurement : Experiment :=
  { name := "Product measurement",
    is_finite := false,
    is_equiprobable := false }

def coin_toss : Experiment :=
  { name := "Coin toss",
    is_finite := true,
    is_equiprobable := true }

def target_shooting : Experiment :=
  { name := "Target shooting",
    is_finite := true,
    is_equiprobable := false }

theorem coin_toss_is_classical_model :
  is_classical_probability_model coin_toss ∧
  ¬is_classical_probability_model seed_germination ∧
  ¬is_classical_probability_model product_measurement ∧
  ¬is_classical_probability_model target_shooting :=
by sorry

end NUMINAMATH_CALUDE_coin_toss_is_classical_model_l4127_412767


namespace NUMINAMATH_CALUDE_total_profit_is_23200_l4127_412715

/-- Represents the business investment scenario -/
structure BusinessInvestment where
  b_investment : ℝ
  b_period : ℝ
  a_investment : ℝ := 3 * b_investment
  a_period : ℝ := 2 * b_period
  c_investment : ℝ := 2 * b_investment
  c_period : ℝ := 0.5 * b_period
  a_rate : ℝ := 0.10
  b_rate : ℝ := 0.15
  c_rate : ℝ := 0.12
  b_profit : ℝ := 4000

/-- Calculates the total profit for the business investment -/
def total_profit (bi : BusinessInvestment) : ℝ :=
  bi.a_investment * bi.a_period * bi.a_rate +
  bi.b_investment * bi.b_period * bi.b_rate +
  bi.c_investment * bi.c_period * bi.c_rate

/-- Theorem stating that the total profit is 23200 -/
theorem total_profit_is_23200 (bi : BusinessInvestment) :
  total_profit bi = 23200 := by
  sorry

end NUMINAMATH_CALUDE_total_profit_is_23200_l4127_412715


namespace NUMINAMATH_CALUDE_a_formula_S_min_l4127_412736

-- Define the sequence and its sum
def S (n : ℕ) : ℤ := n^2 - 48*n

def a : ℕ → ℤ
  | 0 => 0  -- We define a₀ = 0 to make a total function
  | n + 1 => S (n + 1) - S n

-- Theorem for the general formula of a_n
theorem a_formula (n : ℕ) : a (n + 1) = 2 * (n + 1) - 49 := by sorry

-- Theorem for the minimum value of S_n
theorem S_min : ∃ n : ℕ, S n = -576 ∧ ∀ m : ℕ, S m ≥ -576 := by sorry

end NUMINAMATH_CALUDE_a_formula_S_min_l4127_412736


namespace NUMINAMATH_CALUDE_ticket_sales_difference_l4127_412774

/-- Proves the difference in ticket sales given ticket prices and total sales -/
theorem ticket_sales_difference (student_price non_student_price : ℕ) 
  (total_sales total_tickets : ℕ) : 
  student_price = 6 →
  non_student_price = 9 →
  total_sales = 10500 →
  total_tickets = 1700 →
  ∃ (student_tickets non_student_tickets : ℕ),
    student_tickets + non_student_tickets = total_tickets ∧
    student_price * student_tickets + non_student_price * non_student_tickets = total_sales ∧
    student_tickets - non_student_tickets = 1500 :=
by sorry

end NUMINAMATH_CALUDE_ticket_sales_difference_l4127_412774


namespace NUMINAMATH_CALUDE_max_tiles_on_floor_l4127_412791

/-- Represents the dimensions of a rectangle -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the maximum number of tiles that can be placed in a checkerboard pattern -/
def maxTiles (floor : Dimensions) (tile : Dimensions) : ℕ :=
  let lengthTiles := floor.length / tile.length
  let widthTiles := floor.width / tile.width
  (lengthTiles / 2) * (widthTiles / 2)

/-- Theorem stating the maximum number of tiles that can be placed on the given floor -/
theorem max_tiles_on_floor :
  let floor := Dimensions.mk 280 240
  let tile := Dimensions.mk 40 28
  maxTiles floor tile = 12 := by
  sorry

end NUMINAMATH_CALUDE_max_tiles_on_floor_l4127_412791


namespace NUMINAMATH_CALUDE_inequality_system_solution_l4127_412747

theorem inequality_system_solution :
  ∀ x : ℝ, (2 * x + 1 < 3 * x - 2 ∧ 3 * (x - 2) - x ≤ 4) ↔ (3 < x ∧ x ≤ 5) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l4127_412747


namespace NUMINAMATH_CALUDE_sum_first_four_terms_l4127_412737

def a (n : ℕ) : ℤ := (-1)^n * (3*n - 2)

theorem sum_first_four_terms : 
  (a 1) + (a 2) + (a 3) + (a 4) = 6 := by
sorry

end NUMINAMATH_CALUDE_sum_first_four_terms_l4127_412737


namespace NUMINAMATH_CALUDE_sufficient_condition_implies_a_range_l4127_412772

/-- Proposition p: A real number x satisfies the given inequalities -/
def p (x : ℝ) : Prop := 2 < x ∧ x < 3

/-- Proposition q: A real number x satisfies the given inequality -/
def q (x a : ℝ) : Prop := 2 * x^2 - 9 * x + a < 0

/-- Theorem stating that if p is a sufficient condition for q, then 7 ≤ a ≤ 8 -/
theorem sufficient_condition_implies_a_range (a : ℝ) : 
  (∀ x, p x → q x a) → 7 ≤ a ∧ a ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_implies_a_range_l4127_412772


namespace NUMINAMATH_CALUDE_outfit_combinations_l4127_412745

/-- The number of shirts available -/
def num_shirts : ℕ := 8

/-- The number of pairs of pants available -/
def num_pants : ℕ := 5

/-- The number of ties available -/
def num_ties : ℕ := 6

/-- The number of jackets available -/
def num_jackets : ℕ := 2

/-- The number of different outfits that can be created -/
def num_outfits : ℕ := num_shirts * num_pants * (num_ties + 1) * (num_jackets + 1)

/-- Theorem stating that the number of different outfits is 840 -/
theorem outfit_combinations : num_outfits = 840 := by
  sorry

end NUMINAMATH_CALUDE_outfit_combinations_l4127_412745


namespace NUMINAMATH_CALUDE_degree_not_determined_by_A_P_l4127_412796

/-- A characteristic associated with a polynomial -/
def A_P (P : Polynomial ℝ) : Set ℝ := sorry

/-- Theorem stating that the degree of a polynomial cannot be uniquely determined from A_P -/
theorem degree_not_determined_by_A_P :
  ∃ (P1 P2 : Polynomial ℝ), A_P P1 = A_P P2 ∧ P1.degree ≠ P2.degree := by
  sorry

end NUMINAMATH_CALUDE_degree_not_determined_by_A_P_l4127_412796


namespace NUMINAMATH_CALUDE_mode_is_180_l4127_412773

/-- Represents the electricity consumption data for households -/
structure ElectricityData where
  consumption : List Nat
  frequency : List Nat
  total_households : Nat

/-- Calculates the mode of a list of numbers -/
def mode (data : ElectricityData) : Nat :=
  let paired_data := data.consumption.zip data.frequency
  let max_frequency := paired_data.map Prod.snd |>.maximum?
  match max_frequency with
  | none => 0  -- Default value if the list is empty
  | some max => 
      (paired_data.filter (fun p => p.2 = max)).map Prod.fst |>.head!

/-- The electricity consumption survey data -/
def survey_data : ElectricityData := {
  consumption := [120, 140, 160, 180, 200],
  frequency := [5, 5, 3, 6, 1],
  total_households := 20
}

theorem mode_is_180 : mode survey_data = 180 := by
  sorry

end NUMINAMATH_CALUDE_mode_is_180_l4127_412773


namespace NUMINAMATH_CALUDE_xy_value_l4127_412771

theorem xy_value (x y : ℝ) (h1 : x - y = 6) (h2 : x^2 - y^2 = 20) : x * y = -56/9 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l4127_412771


namespace NUMINAMATH_CALUDE_french_fries_cooking_time_l4127_412712

/-- Calculates the remaining cooking time in seconds -/
def remaining_cooking_time (recommended_minutes : ℕ) (actual_seconds : ℕ) : ℕ :=
  recommended_minutes * 60 - actual_seconds

/-- Theorem: Given the recommended cooking time of 5 minutes and an actual cooking time of 45 seconds, the remaining cooking time is 255 seconds -/
theorem french_fries_cooking_time : remaining_cooking_time 5 45 = 255 := by
  sorry

end NUMINAMATH_CALUDE_french_fries_cooking_time_l4127_412712


namespace NUMINAMATH_CALUDE_deepak_age_l4127_412748

theorem deepak_age (arun_age deepak_age : ℕ) : 
  (arun_age : ℚ) / deepak_age = 2 / 5 →
  arun_age + 10 = 30 →
  deepak_age = 50 := by
sorry

end NUMINAMATH_CALUDE_deepak_age_l4127_412748


namespace NUMINAMATH_CALUDE_inequality_proof_l4127_412704

theorem inequality_proof (a b c A α : Real) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hA : a + b + c = A) (hA1 : A ≤ 1) (hα : α > 0) : 
  (1/a - a)^α + (1/b - b)^α + (1/c - c)^α ≥ 3*(3/A - A/3)^α := by
  sorry

#check inequality_proof

end NUMINAMATH_CALUDE_inequality_proof_l4127_412704


namespace NUMINAMATH_CALUDE_zoo_trip_bus_capacity_l4127_412750

/-- Given a school trip to the zoo with the following conditions:
  * total_students: The total number of students on the trip
  * num_buses: The number of buses used for transportation
  * students_in_cars: The number of students who traveled in cars
  * students_per_bus: The number of students in each bus

  This theorem proves that when total_students = 396, num_buses = 7, and students_in_cars = 4,
  the number of students in each bus (students_per_bus) is equal to 56. -/
theorem zoo_trip_bus_capacity 
  (total_students : ℕ) 
  (num_buses : ℕ) 
  (students_in_cars : ℕ) 
  (students_per_bus : ℕ) 
  (h1 : total_students = 396) 
  (h2 : num_buses = 7) 
  (h3 : students_in_cars = 4) 
  (h4 : students_per_bus * num_buses + students_in_cars = total_students) :
  students_per_bus = 56 := by
  sorry

end NUMINAMATH_CALUDE_zoo_trip_bus_capacity_l4127_412750


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l4127_412755

theorem sqrt_equation_solution :
  ∃! z : ℝ, Real.sqrt (10 + 3 * z) = 8 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l4127_412755


namespace NUMINAMATH_CALUDE_range_of_m_l4127_412753

-- Define the function y in terms of x, k, and m
def y (x k m : ℝ) : ℝ := k * x - k + m

-- State the theorem
theorem range_of_m (k m : ℝ) : 
  (∃ x, y x k m = 3 ∧ x = -2) →  -- When x = -2, y = 3
  (k ≠ 0) →  -- k is non-zero (implied by direct proportionality)
  (k < 0) →  -- Slope is negative (passes through 2nd, 3rd, and 4th quadrants)
  (-k + m < 0) →  -- y-intercept is negative (passes through 2nd, 3rd, and 4th quadrants)
  m < -3/2 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l4127_412753


namespace NUMINAMATH_CALUDE_salary_changes_l4127_412710

def initial_salary : ℝ := 1800

def may_raise : ℝ := 0.30
def june_cut : ℝ := 0.25
def july_increase : ℝ := 0.10

def final_salary : ℝ := initial_salary * (1 + july_increase)

theorem salary_changes :
  final_salary = 1980 := by sorry

end NUMINAMATH_CALUDE_salary_changes_l4127_412710


namespace NUMINAMATH_CALUDE_min_additional_marbles_correct_l4127_412741

/-- The number of friends Lisa has -/
def num_friends : ℕ := 10

/-- The initial number of marbles Lisa has -/
def initial_marbles : ℕ := 34

/-- The sum of the first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The minimum number of additional marbles needed -/
def min_additional_marbles : ℕ := sum_first_n num_friends - initial_marbles

theorem min_additional_marbles_correct :
  min_additional_marbles = 21 ∧
  sum_first_n num_friends ≥ initial_marbles + min_additional_marbles ∧
  ∀ k : ℕ, k < min_additional_marbles →
    sum_first_n num_friends > initial_marbles + k :=
by sorry

end NUMINAMATH_CALUDE_min_additional_marbles_correct_l4127_412741


namespace NUMINAMATH_CALUDE_four_people_handshakes_l4127_412795

/-- The number of handshakes in a group where each person shakes hands with every other person exactly once -/
def num_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a group of 4 people, where each person shakes hands with every other person exactly once, the total number of handshakes is 6. -/
theorem four_people_handshakes : num_handshakes 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_four_people_handshakes_l4127_412795


namespace NUMINAMATH_CALUDE_prob_xavier_yvonne_not_zelda_l4127_412782

-- Define the difficulty factors and probabilities
variable (a b c : ℝ)
variable (p_xavier : ℝ := (1/3)^a)
variable (p_yvonne : ℝ := (1/2)^b)
variable (p_zelda : ℝ := (5/8)^c)

-- Define the theorem
theorem prob_xavier_yvonne_not_zelda :
  p_xavier * p_yvonne * (1 - p_zelda) = (1/16) * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_prob_xavier_yvonne_not_zelda_l4127_412782


namespace NUMINAMATH_CALUDE_supplement_quadruple_complement_30_l4127_412779

/-- The degree measure of the supplement of the quadruple of the complement of a 30-degree angle is 120 degrees. -/
theorem supplement_quadruple_complement_30 : 
  let initial_angle : ℝ := 30
  let complement := 90 - initial_angle
  let quadruple := 4 * complement
  let supplement := if quadruple ≤ 180 then 180 - quadruple else 360 - quadruple
  supplement = 120 := by sorry

end NUMINAMATH_CALUDE_supplement_quadruple_complement_30_l4127_412779


namespace NUMINAMATH_CALUDE_isabellas_original_hair_length_l4127_412758

/-- The length of Isabella's hair before the haircut -/
def original_length : ℝ := sorry

/-- The length of Isabella's hair after the haircut -/
def after_haircut_length : ℝ := 9

/-- The length of hair that was cut off -/
def cut_length : ℝ := 9

/-- Theorem stating that Isabella's original hair length was 18 inches -/
theorem isabellas_original_hair_length :
  original_length = after_haircut_length + cut_length ∧ original_length = 18 := by
  sorry

end NUMINAMATH_CALUDE_isabellas_original_hair_length_l4127_412758


namespace NUMINAMATH_CALUDE_inscribed_circle_diameter_l4127_412740

theorem inscribed_circle_diameter (DE DF EF : ℝ) (h1 : DE = 10) (h2 : DF = 5) (h3 : EF = 9) :
  let s := (DE + DF + EF) / 2
  let area := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  let diameter := 2 * area / s
  diameter = Real.sqrt 14 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_diameter_l4127_412740


namespace NUMINAMATH_CALUDE_complex_expression_evaluation_l4127_412726

theorem complex_expression_evaluation :
  - Real.sqrt 3 * Real.sqrt 6 + abs (1 - Real.sqrt 2) - (1/3)⁻¹ = -4 * Real.sqrt 2 - 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_evaluation_l4127_412726


namespace NUMINAMATH_CALUDE_meaningful_fraction_l4127_412706

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 3)) ↔ x ≠ 3 := by sorry

end NUMINAMATH_CALUDE_meaningful_fraction_l4127_412706


namespace NUMINAMATH_CALUDE_brick_surface_area_l4127_412794

/-- The surface area of a rectangular prism. -/
def surface_area (length width height : ℝ) : ℝ :=
  2 * (length * width + length * height + width * height)

/-- Theorem: The surface area of a 10 cm x 4 cm x 3 cm brick is 164 cm². -/
theorem brick_surface_area :
  surface_area 10 4 3 = 164 := by
  sorry

end NUMINAMATH_CALUDE_brick_surface_area_l4127_412794


namespace NUMINAMATH_CALUDE_ryan_fundraising_goal_l4127_412714

/-- The total amount Ryan wants to raise for his business -/
def total_amount (avg_funding : ℕ) (num_people : ℕ) (existing_funds : ℕ) : ℕ :=
  avg_funding * num_people + existing_funds

/-- Proof that Ryan wants to raise $1000 for his business -/
theorem ryan_fundraising_goal :
  let avg_funding : ℕ := 10
  let num_people : ℕ := 80
  let existing_funds : ℕ := 200
  total_amount avg_funding num_people existing_funds = 1000 := by
  sorry

end NUMINAMATH_CALUDE_ryan_fundraising_goal_l4127_412714


namespace NUMINAMATH_CALUDE_dallas_age_l4127_412775

theorem dallas_age (dexter_age : ℕ) (darcy_age : ℕ) (dallas_age_last_year : ℕ) :
  dexter_age = 8 →
  darcy_age = 2 * dexter_age →
  dallas_age_last_year = 3 * (darcy_age - 1) →
  dallas_age_last_year + 1 = 46 :=
by
  sorry

end NUMINAMATH_CALUDE_dallas_age_l4127_412775


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_intersection_equality_range_l4127_412766

-- Define the sets A and B
def A : Set ℝ := {x | (x - 5) / (x - 2) ≤ 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - 2*a*x + a^2 - 1 < 0}

-- Theorem for part 1
theorem complement_intersection_theorem :
  (Set.univ \ A) ∩ (Set.univ \ B 2) = {x | x ≤ 1 ∨ x > 5} := by sorry

-- Theorem for part 2
theorem intersection_equality_range :
  {a : ℝ | A ∩ B a = B a} = {a : ℝ | 3 ≤ a ∧ a ≤ 4} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_intersection_equality_range_l4127_412766


namespace NUMINAMATH_CALUDE_jerusha_earned_68_l4127_412731

/-- Jerusha's earnings given Lottie's earnings and their total earnings -/
def jerushas_earnings (lotties_earnings : ℚ) (total_earnings : ℚ) : ℚ :=
  4 * lotties_earnings

theorem jerusha_earned_68 :
  ∃ (lotties_earnings : ℚ),
    jerushas_earnings lotties_earnings 85 = 68 ∧
    lotties_earnings + jerushas_earnings lotties_earnings 85 = 85 := by
  sorry

end NUMINAMATH_CALUDE_jerusha_earned_68_l4127_412731


namespace NUMINAMATH_CALUDE_probability_same_team_l4127_412788

/-- The probability of two volunteers joining the same team out of three teams -/
theorem probability_same_team (num_teams : ℕ) (num_volunteers : ℕ) : 
  num_teams = 3 → num_volunteers = 2 → 
  (num_teams.choose num_volunteers : ℚ) / (num_teams ^ num_volunteers : ℚ) = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_probability_same_team_l4127_412788


namespace NUMINAMATH_CALUDE_clock_in_probability_l4127_412716

/-- The probability of an employee clocking in on time given a total time window and valid clock-in time -/
theorem clock_in_probability (total_window : ℕ) (valid_time : ℕ) 
  (h1 : total_window = 40) 
  (h2 : valid_time = 15) : 
  (valid_time : ℚ) / total_window = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_clock_in_probability_l4127_412716


namespace NUMINAMATH_CALUDE_inequality_range_l4127_412785

theorem inequality_range (a : ℝ) (h : 0 ≤ a ∧ a ≤ 4) :
  ∀ x : ℝ, x^2 + a*x > 4*x + a - 3 ↔ x < -1 ∨ x > 3 := by sorry

end NUMINAMATH_CALUDE_inequality_range_l4127_412785


namespace NUMINAMATH_CALUDE_calculation_proof_l4127_412786

theorem calculation_proof : ((-2)^2 : ℝ) + Real.sqrt 16 - 2 * Real.sin (π / 6) + (2023 - Real.pi)^0 = 8 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l4127_412786


namespace NUMINAMATH_CALUDE_emily_max_servings_l4127_412797

/-- Represents the recipe and available ingredients for a fruit smoothie --/
structure SmoothieIngredients where
  recipe_bananas : ℕ
  recipe_strawberries : ℕ
  recipe_yogurt : ℕ
  available_bananas : ℕ
  available_strawberries : ℕ
  available_yogurt : ℕ

/-- Calculates the maximum number of servings that can be made --/
def max_servings (ingredients : SmoothieIngredients) : ℕ :=
  min
    (ingredients.available_bananas * 3 / ingredients.recipe_bananas)
    (min
      (ingredients.available_strawberries * 3 / ingredients.recipe_strawberries)
      (ingredients.available_yogurt * 3 / ingredients.recipe_yogurt))

/-- Theorem stating that Emily can make at most 6 servings --/
theorem emily_max_servings :
  let emily_ingredients : SmoothieIngredients := {
    recipe_bananas := 2,
    recipe_strawberries := 1,
    recipe_yogurt := 2,
    available_bananas := 4,
    available_strawberries := 3,
    available_yogurt := 6
  }
  max_servings emily_ingredients = 6 := by
  sorry

end NUMINAMATH_CALUDE_emily_max_servings_l4127_412797


namespace NUMINAMATH_CALUDE_strawberry_charity_donation_is_correct_l4127_412781

/-- The amount of money donated to charity from strawberry jam sales -/
def strawberry_charity_donation : ℚ :=
let betty_strawberries : ℕ := 25
let matthew_strawberries : ℕ := betty_strawberries + 30
let natalie_strawberries : ℕ := matthew_strawberries / 3
let emily_strawberries : ℕ := natalie_strawberries / 2
let ethan_strawberries : ℕ := natalie_strawberries * 2
let total_strawberries : ℕ := betty_strawberries + matthew_strawberries + natalie_strawberries + emily_strawberries + ethan_strawberries
let strawberries_per_jar : ℕ := 12
let jars_made : ℕ := total_strawberries / strawberries_per_jar
let price_per_jar : ℚ := 6
let total_revenue : ℚ := (jars_made : ℚ) * price_per_jar
let donation_percentage : ℚ := 40 / 100
donation_percentage * total_revenue

theorem strawberry_charity_donation_is_correct :
  strawberry_charity_donation = 26.4 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_charity_donation_is_correct_l4127_412781


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_range_l4127_412723

/-- The eccentricity of an ellipse with a perpendicular bisector through a point on the ellipse --/
theorem ellipse_eccentricity_range (a b : ℝ) (h_pos : 0 < b ∧ b < a) :
  let e := Real.sqrt (1 - b^2 / a^2)
  ∃ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1 ∧ 
    x^2 + y^2 = (a^2 - b^2)) → 
    Real.sqrt 2 / 2 ≤ e ∧ e < 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_range_l4127_412723


namespace NUMINAMATH_CALUDE_unique_box_dimensions_l4127_412776

theorem unique_box_dimensions : ∃! (a b c : ℕ+), 
  (a ≥ b) ∧ (b ≥ c) ∧ 
  (a.val * b.val * c.val = 2 * (a.val * b.val + a.val * c.val + b.val * c.val)) := by
  sorry

end NUMINAMATH_CALUDE_unique_box_dimensions_l4127_412776


namespace NUMINAMATH_CALUDE_band_member_earnings_l4127_412760

theorem band_member_earnings 
  (attendees : ℕ) 
  (band_share : ℚ) 
  (ticket_price : ℕ) 
  (band_members : ℕ) 
  (h1 : attendees = 500) 
  (h2 : band_share = 70 / 100) 
  (h3 : ticket_price = 30) 
  (h4 : band_members = 4) : 
  (attendees * ticket_price * band_share) / band_members = 2625 := by
sorry

end NUMINAMATH_CALUDE_band_member_earnings_l4127_412760


namespace NUMINAMATH_CALUDE_line_intersects_y_axis_l4127_412721

/-- A line passing through two points (1, 7) and (3, 11) -/
def line (x : ℝ) : ℝ := 2 * x + 5

/-- The y-axis is defined as the set of points with x-coordinate equal to 0 -/
def y_axis (x : ℝ) : Prop := x = 0

theorem line_intersects_y_axis :
  ∃ y : ℝ, y_axis 0 ∧ line 0 = y ∧ y = 5 := by sorry

end NUMINAMATH_CALUDE_line_intersects_y_axis_l4127_412721


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l4127_412752

/-- A quadratic function with given vertex and y-intercept -/
def quadratic_function (x : ℝ) : ℝ := 2 * (x - 2)^2 - 4

theorem quadratic_function_properties :
  (∀ x, quadratic_function x = 2 * (x - 2)^2 - 4) ∧
  (quadratic_function 2 = -4) ∧
  (quadratic_function 0 = 4) ∧
  (quadratic_function 3 ≠ 5) :=
sorry

#check quadratic_function_properties

end NUMINAMATH_CALUDE_quadratic_function_properties_l4127_412752


namespace NUMINAMATH_CALUDE_sum_of_digits_of_n_is_nine_l4127_412738

/-- Two distinct digits -/
def distinct_digits (d e : Nat) : Prop :=
  d ≠ e ∧ d < 10 ∧ e < 10

/-- Sum of digits is prime -/
def sum_is_prime (d e : Nat) : Prop :=
  Nat.Prime (d + e)

/-- k is prime and greater than both d and e -/
def k_is_valid_prime (d e k : Nat) : Prop :=
  Nat.Prime k ∧ k > d ∧ k > e

/-- n is the product of d, e, and k -/
def n_is_product (n d e k : Nat) : Prop :=
  n = d * e * k

/-- k is related to d and e -/
def k_relation (d e k : Nat) : Prop :=
  k = 10 * d + e

/-- n is the largest such product -/
def n_is_largest (n : Nat) : Prop :=
  ∀ m d e k, distinct_digits d e → sum_is_prime d e → k_is_valid_prime d e k →
    k_relation d e k → n_is_product m d e k → m ≤ n

/-- n is the smallest multiple of k -/
def n_is_smallest_multiple (n k : Nat) : Prop :=
  k ∣ n ∧ ∀ m, m < n → ¬(k ∣ m)

/-- Sum of digits of a number -/
def sum_of_digits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_of_n_is_nine :
  ∃ n d e k, distinct_digits d e ∧ sum_is_prime d e ∧ k_is_valid_prime d e k ∧
    k_relation d e k ∧ n_is_product n d e k ∧ n_is_largest n ∧
    n_is_smallest_multiple n k ∧ sum_of_digits n = 9 :=
sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_n_is_nine_l4127_412738
