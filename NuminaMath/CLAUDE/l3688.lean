import Mathlib

namespace NUMINAMATH_CALUDE_theft_culprits_l3688_368861

-- Define the guilt status of each person
variable (E F G : Prop)

-- E represents "Elise is guilty"
-- F represents "Fred is guilty"
-- G represents "Gaétan is guilty"

-- Define the given conditions
axiom cond1 : ¬G → F
axiom cond2 : ¬E → G
axiom cond3 : G → E
axiom cond4 : E → ¬F

-- Theorem to prove
theorem theft_culprits : E ∧ G ∧ ¬F := by
  sorry

end NUMINAMATH_CALUDE_theft_culprits_l3688_368861


namespace NUMINAMATH_CALUDE_red_star_wins_l3688_368818

theorem red_star_wins (total_matches : ℕ) (total_points : ℕ) 
  (h1 : total_matches = 9)
  (h2 : total_points = 23)
  (h3 : ∀ (wins draws : ℕ), wins + draws = total_matches → 3 * wins + draws = total_points) :
  ∃ (wins draws : ℕ), wins = 7 ∧ draws = 2 := by
  sorry

end NUMINAMATH_CALUDE_red_star_wins_l3688_368818


namespace NUMINAMATH_CALUDE_min_value_a_a_range_l3688_368885

-- Define the function f
def f (x a : ℝ) : ℝ := |x + a| + |x - 3|

-- Theorem 1
theorem min_value_a (a : ℝ) :
  (∀ x, f x a ≥ 5) ∧ (∃ x, f x a = 5) → a = 2 ∨ a = -8 := by
  sorry

-- Theorem 2
theorem a_range (a : ℝ) :
  (∀ x, -1 ≤ x → x ≤ 0 → f x a ≤ |x - 4|) → 0 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_a_a_range_l3688_368885


namespace NUMINAMATH_CALUDE_franklin_students_count_l3688_368853

/-- The number of Valentines Mrs. Franklin already has -/
def valentines_owned : ℝ := 58.0

/-- The number of additional Valentines Mrs. Franklin needs -/
def valentines_needed : ℝ := 16.0

/-- The number of students Mrs. Franklin has -/
def number_of_students : ℝ := valentines_owned + valentines_needed

theorem franklin_students_count : number_of_students = 74.0 := by
  sorry

end NUMINAMATH_CALUDE_franklin_students_count_l3688_368853


namespace NUMINAMATH_CALUDE_problem_solution_l3688_368844

theorem problem_solution (a b : ℝ) (h1 : a * b = 7) (h2 : a - b = 5) :
  a^2 - 6*a*b + b^2 = -3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3688_368844


namespace NUMINAMATH_CALUDE_kyle_monthly_income_l3688_368891

def rent : ℕ := 1250
def utilities : ℕ := 150
def retirement_savings : ℕ := 400
def groceries_eating_out : ℕ := 300
def insurance : ℕ := 200
def miscellaneous : ℕ := 200
def car_payment : ℕ := 350
def gas_maintenance : ℕ := 350

def total_expenses : ℕ := rent + utilities + retirement_savings + groceries_eating_out + insurance + miscellaneous + car_payment + gas_maintenance

theorem kyle_monthly_income : total_expenses = 3200 := by
  sorry

end NUMINAMATH_CALUDE_kyle_monthly_income_l3688_368891


namespace NUMINAMATH_CALUDE_alex_candles_used_l3688_368801

/-- The number of candles Alex used -/
def candles_used (initial : ℕ) (remaining : ℕ) : ℕ := initial - remaining

/-- Theorem stating that Alex used 32 candles -/
theorem alex_candles_used :
  let initial : ℕ := 44
  let remaining : ℕ := 12
  candles_used initial remaining = 32 := by
  sorry

end NUMINAMATH_CALUDE_alex_candles_used_l3688_368801


namespace NUMINAMATH_CALUDE_total_driving_time_bound_l3688_368849

/-- Represents the driving scenario with given distances and speeds -/
structure DrivingScenario where
  distance_first : ℝ
  time_first : ℝ
  distance_second : ℝ
  distance_third : ℝ
  distance_fourth : ℝ
  speed_second : ℝ
  speed_third : ℝ
  speed_fourth : ℝ

/-- The total driving time is less than or equal to 10 hours -/
theorem total_driving_time_bound (scenario : DrivingScenario) 
  (h1 : scenario.distance_first = 120)
  (h2 : scenario.time_first = 3)
  (h3 : scenario.distance_second = 60)
  (h4 : scenario.distance_third = 90)
  (h5 : scenario.distance_fourth = 200)
  (h6 : scenario.speed_second > 0)
  (h7 : scenario.speed_third > 0)
  (h8 : scenario.speed_fourth > 0) :
  scenario.time_first + 
  scenario.distance_second / scenario.speed_second + 
  scenario.distance_third / scenario.speed_third + 
  scenario.distance_fourth / scenario.speed_fourth ≤ 10 := by
  sorry

end NUMINAMATH_CALUDE_total_driving_time_bound_l3688_368849


namespace NUMINAMATH_CALUDE_palindrome_product_sum_l3688_368830

/-- A number is a three-digit palindrome if it's between 100 and 999 (inclusive) and reads the same forwards and backwards. -/
def IsThreeDigitPalindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n / 100 = n % 10) ∧ ((n / 10) % 10 = (n % 100) / 10)

theorem palindrome_product_sum (a b : ℕ) : 
  IsThreeDigitPalindrome a → IsThreeDigitPalindrome b → a * b = 334491 → a + b = 1324 := by
  sorry

end NUMINAMATH_CALUDE_palindrome_product_sum_l3688_368830


namespace NUMINAMATH_CALUDE_parallel_planes_imply_parallel_lines_l3688_368874

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation
variable (parallel : Plane → Plane → Prop)

-- Define the subset relation for a line being in a plane
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem parallel_planes_imply_parallel_lines 
  (a b : Line) (α β : Plane) 
  (ha : subset a α) (hb : subset b α) :
  parallel α β → (parallel α β ∧ parallel α β) := by
  sorry

end NUMINAMATH_CALUDE_parallel_planes_imply_parallel_lines_l3688_368874


namespace NUMINAMATH_CALUDE_johnson_family_reunion_l3688_368850

theorem johnson_family_reunion (children : ℕ) (adults : ℕ) (blue_adults : ℕ) : 
  children = 45 →
  adults = children / 3 →
  blue_adults = adults / 3 →
  adults - blue_adults = 10 := by
sorry

end NUMINAMATH_CALUDE_johnson_family_reunion_l3688_368850


namespace NUMINAMATH_CALUDE_divisors_of_m_squared_count_specific_divisors_l3688_368811

def m : ℕ := 2^40 * 5^24

theorem divisors_of_m_squared (d : ℕ) : 
  (d ∣ m^2) ∧ (d < m) ∧ ¬(d ∣ m) ↔ d ∈ Finset.filter (λ x => (x ∣ m^2) ∧ (x < m) ∧ ¬(x ∣ m)) (Finset.range (m + 1)) :=
sorry

theorem count_specific_divisors : 
  Finset.card (Finset.filter (λ x => (x ∣ m^2) ∧ (x < m) ∧ ¬(x ∣ m)) (Finset.range (m + 1))) = 959 :=
sorry

end NUMINAMATH_CALUDE_divisors_of_m_squared_count_specific_divisors_l3688_368811


namespace NUMINAMATH_CALUDE_jeremy_gives_two_watermelons_l3688_368880

/-- The number of watermelons Jeremy gives to his dad each week. -/
def watermelons_given_to_dad (total_watermelons : ℕ) (weeks_lasted : ℕ) (eaten_per_week : ℕ) : ℕ :=
  (total_watermelons / weeks_lasted) - eaten_per_week

/-- Theorem stating that Jeremy gives 2 watermelons to his dad each week. -/
theorem jeremy_gives_two_watermelons :
  watermelons_given_to_dad 30 6 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_jeremy_gives_two_watermelons_l3688_368880


namespace NUMINAMATH_CALUDE_distance_inequality_l3688_368870

theorem distance_inequality (a : ℝ) :
  (abs (a - 1) < 3) → (-2 < a ∧ a < 4) := by
  sorry

end NUMINAMATH_CALUDE_distance_inequality_l3688_368870


namespace NUMINAMATH_CALUDE_kara_book_count_l3688_368873

/-- The number of books read by each person in the Book Tournament --/
structure BookCount where
  candice : ℕ
  amanda : ℕ
  kara : ℕ
  patricia : ℕ

/-- The conditions of the Book Tournament --/
def BookTournament (bc : BookCount) : Prop :=
  bc.candice = 18 ∧
  bc.candice = 3 * bc.amanda ∧
  bc.kara = bc.amanda / 2

theorem kara_book_count (bc : BookCount) (h : BookTournament bc) : bc.kara = 3 := by
  sorry

end NUMINAMATH_CALUDE_kara_book_count_l3688_368873


namespace NUMINAMATH_CALUDE_initial_average_customers_l3688_368840

theorem initial_average_customers (x : ℕ) (today_customers : ℕ) (new_average : ℕ) 
  (h1 : x = 1)
  (h2 : today_customers = 120)
  (h3 : new_average = 90)
  : ∃ initial_average : ℕ, initial_average = 60 ∧ 
    (initial_average * x + today_customers) / (x + 1) = new_average :=
by
  sorry

end NUMINAMATH_CALUDE_initial_average_customers_l3688_368840


namespace NUMINAMATH_CALUDE_pierre_cake_consumption_l3688_368827

theorem pierre_cake_consumption (cake_weight : ℝ) (num_parts : ℕ) 
  (h1 : cake_weight = 400)
  (h2 : num_parts = 8)
  (h3 : num_parts > 0) :
  let part_weight := cake_weight / num_parts
  let nathalie_ate := part_weight
  let pierre_ate := 2 * nathalie_ate
  pierre_ate = 100 := by
  sorry

end NUMINAMATH_CALUDE_pierre_cake_consumption_l3688_368827


namespace NUMINAMATH_CALUDE_sum_of_powers_l3688_368882

theorem sum_of_powers : (-3)^4 + (-3)^2 + (-3)^0 + 3^0 + 3^2 + 3^4 = 182 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_l3688_368882


namespace NUMINAMATH_CALUDE_green_ball_probability_l3688_368816

-- Define the containers and their contents
def containerA : ℕ × ℕ := (5, 7)  -- (red, green)
def containerB : ℕ × ℕ := (8, 6)
def containerC : ℕ × ℕ := (3, 9)

-- Define the probability of selecting each container
def containerProb : ℚ := 1 / 3

-- Define the probability of selecting a green ball from each container
def greenProbA : ℚ := containerA.2 / (containerA.1 + containerA.2)
def greenProbB : ℚ := containerB.2 / (containerB.1 + containerB.2)
def greenProbC : ℚ := containerC.2 / (containerC.1 + containerC.2)

-- Theorem: The probability of selecting a green ball is 127/252
theorem green_ball_probability : 
  containerProb * greenProbA + containerProb * greenProbB + containerProb * greenProbC = 127 / 252 := by
  sorry


end NUMINAMATH_CALUDE_green_ball_probability_l3688_368816


namespace NUMINAMATH_CALUDE_rationalize_denominator_cube_roots_l3688_368804

theorem rationalize_denominator_cube_roots :
  let x := (3 : ℝ)^(1/3)
  let y := (2 : ℝ)^(1/3)
  1 / (x - y) = x^2 + x*y + y^2 :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_cube_roots_l3688_368804


namespace NUMINAMATH_CALUDE_f_and_g_properties_l3688_368803

/-- Given a function f and constants a and b -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + x^2 + b * x

/-- The derivative of f -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * x + b

/-- Function g defined as the sum of f and its derivative -/
def g (a b : ℝ) (x : ℝ) : ℝ := f a b x + f' a b x

/-- g is an odd function -/
axiom g_odd (a b : ℝ) : ∀ x, g a b (-x) = -(g a b x)

theorem f_and_g_properties :
  ∃ (a b : ℝ),
    (∀ x, f a b x = -1/3 * x^3 + x^2) ∧
    (∀ x ∈ Set.Icc 1 2, g a b x ≤ 4 * Real.sqrt 2 / 3) ∧
    (∀ x ∈ Set.Icc 1 2, g a b x ≥ 4 / 3) ∧
    (g a b (Real.sqrt 2) = 4 * Real.sqrt 2 / 3) ∧
    (g a b 2 = 4 / 3) := by sorry

end NUMINAMATH_CALUDE_f_and_g_properties_l3688_368803


namespace NUMINAMATH_CALUDE_probability_no_more_than_one_complaint_probability_two_complaints_in_two_months_l3688_368856

-- Define the probabilities for complaints in a single month
def p_zero_complaints : ℝ := 0.3
def p_one_complaint : ℝ := 0.5
def p_two_complaints : ℝ := 0.2

-- Theorem for part (I)
theorem probability_no_more_than_one_complaint :
  p_zero_complaints + p_one_complaint = 0.8 := by sorry

-- Theorem for part (II)
theorem probability_two_complaints_in_two_months :
  let p_two_total := p_zero_complaints * p_two_complaints +
                     p_two_complaints * p_zero_complaints +
                     p_one_complaint * p_one_complaint
  p_two_total = 0.37 := by sorry

end NUMINAMATH_CALUDE_probability_no_more_than_one_complaint_probability_two_complaints_in_two_months_l3688_368856


namespace NUMINAMATH_CALUDE_incorrect_observation_value_l3688_368837

theorem incorrect_observation_value
  (n : ℕ)
  (initial_mean correct_mean correct_value : ℚ)
  (h_n : n = 50)
  (h_initial_mean : initial_mean = 36)
  (h_correct_mean : correct_mean = 365/10)
  (h_correct_value : correct_value = 45)
  : (n : ℚ) * initial_mean + correct_value - ((n : ℚ) * correct_mean) = 20 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_observation_value_l3688_368837


namespace NUMINAMATH_CALUDE_angle_between_vectors_l3688_368823

/-- Given vectors a and b in ℝ², prove that the angle between them is π -/
theorem angle_between_vectors (a b : ℝ × ℝ) : 
  a + 2 • b = (2, -4) → 
  3 • a - b = (-8, 16) → 
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) = π :=
by sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l3688_368823


namespace NUMINAMATH_CALUDE_function_extrema_m_range_l3688_368852

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + m*x^2 + (m + 6)*x + 1

-- State the theorem
theorem function_extrema_m_range (m : ℝ) :
  (∃ (x_max x_min : ℝ), ∀ (x : ℝ), f m x ≤ f m x_max ∧ f m x_min ≤ f m x) →
  m < -3 ∨ m > 6 :=
sorry

end NUMINAMATH_CALUDE_function_extrema_m_range_l3688_368852


namespace NUMINAMATH_CALUDE_largest_possible_reflections_l3688_368841

/-- Represents the angle of reflection at each point -/
def reflection_angle (n : ℕ) : ℝ := 15 * n

/-- The condition for the beam to hit perpendicularly and retrace its path -/
def valid_reflection (n : ℕ) : Prop := reflection_angle n ≤ 90

theorem largest_possible_reflections : ∃ (max_n : ℕ), 
  (∀ n : ℕ, valid_reflection n → n ≤ max_n) ∧ 
  valid_reflection max_n ∧ 
  max_n = 6 :=
sorry

end NUMINAMATH_CALUDE_largest_possible_reflections_l3688_368841


namespace NUMINAMATH_CALUDE_complex_arithmetic_equality_l3688_368886

theorem complex_arithmetic_equality : -6 / 2 + (1/3 - 3/4) * 12 + (-3)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equality_l3688_368886


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3688_368859

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (3 + Real.sqrt x) = 4 → x = 169 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3688_368859


namespace NUMINAMATH_CALUDE_expression_value_at_five_l3688_368834

theorem expression_value_at_five : 
  let x : ℝ := 5
  (x^3 - 4*x^2 + 3*x) / (x - 3) = 20 := by sorry

end NUMINAMATH_CALUDE_expression_value_at_five_l3688_368834


namespace NUMINAMATH_CALUDE_income_ratio_proof_l3688_368813

/-- Proves that the ratio of A's monthly income to B's monthly income is 2.5:1 -/
theorem income_ratio_proof (c_monthly_income b_monthly_income a_annual_income : ℝ) 
  (h1 : c_monthly_income = 14000)
  (h2 : b_monthly_income = c_monthly_income * 1.12)
  (h3 : a_annual_income = 470400) : 
  (a_annual_income / 12) / b_monthly_income = 2.5 := by
  sorry

#check income_ratio_proof

end NUMINAMATH_CALUDE_income_ratio_proof_l3688_368813


namespace NUMINAMATH_CALUDE_proportional_segments_l3688_368892

theorem proportional_segments (a b c d : ℝ) : 
  a / b = c / d → a = 2 → b = 4 → c = 3 → d = 6 := by
  sorry

end NUMINAMATH_CALUDE_proportional_segments_l3688_368892


namespace NUMINAMATH_CALUDE_geometric_proportion_conclusion_l3688_368868

/-- A set of four real numbers forms a geometric proportion in any order -/
def GeometricProportionAnyOrder (a b c d : ℝ) : Prop :=
  (a / b = c / d ∧ a / b = d / c) ∧
  (a / c = b / d ∧ a / c = d / b) ∧
  (a / d = b / c ∧ a / d = c / b)

/-- The conclusion about four numbers forming a geometric proportion in any order -/
theorem geometric_proportion_conclusion (a b c d : ℝ) 
  (h : GeometricProportionAnyOrder a b c d) :
  (a = b ∧ b = c ∧ c = d) ∨ 
  (|a| = |b| ∧ |b| = |c| ∧ |c| = |d| ∧ 
   ((a > 0 ∧ b > 0 ∧ c < 0 ∧ d < 0) ∨
    (a > 0 ∧ c > 0 ∧ b < 0 ∧ d < 0) ∨
    (a > 0 ∧ d > 0 ∧ b < 0 ∧ c < 0) ∨
    (b > 0 ∧ c > 0 ∧ a < 0 ∧ d < 0) ∨
    (b > 0 ∧ d > 0 ∧ a < 0 ∧ c < 0) ∨
    (c > 0 ∧ d > 0 ∧ a < 0 ∧ b < 0))) :=
by sorry

end NUMINAMATH_CALUDE_geometric_proportion_conclusion_l3688_368868


namespace NUMINAMATH_CALUDE_min_cards_to_draw_l3688_368812

/- Define the number of suits in a deck -/
def num_suits : ℕ := 4

/- Define the number of cards in each suit -/
def cards_per_suit : ℕ := 13

/- Define the number of cards needed in the same suit -/
def cards_needed_same_suit : ℕ := 4

/- Define the number of jokers in the deck -/
def num_jokers : ℕ := 2

/- Theorem: The minimum number of cards to draw to ensure 4 of the same suit is 15 -/
theorem min_cards_to_draw : 
  (num_suits - 1) * (cards_needed_same_suit - 1) + cards_needed_same_suit + num_jokers = 15 := by
  sorry

end NUMINAMATH_CALUDE_min_cards_to_draw_l3688_368812


namespace NUMINAMATH_CALUDE_base_conversion_subtraction_l3688_368822

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun d acc => d + b * acc) 0

/-- The problem statement -/
theorem base_conversion_subtraction :
  let base_7_num := [3, 0, 1, 2, 5]  -- 52103 in base 7 (least significant digit first)
  let base_5_num := [0, 2, 1, 3, 4]  -- 43120 in base 5 (least significant digit first)
  to_base_10 base_7_num 7 - to_base_10 base_5_num 5 = 9833 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_subtraction_l3688_368822


namespace NUMINAMATH_CALUDE_village_population_equality_l3688_368847

/-- The number of years it takes for two villages' populations to be equal -/
def years_until_equal_population (x_initial : ℕ) (x_decrease : ℕ) (y_initial : ℕ) (y_increase : ℕ) : ℕ :=
  (x_initial - y_initial) / (y_increase + x_decrease)

theorem village_population_equality :
  years_until_equal_population 78000 1200 42000 800 = 18 := by
  sorry

end NUMINAMATH_CALUDE_village_population_equality_l3688_368847


namespace NUMINAMATH_CALUDE_jen_jam_consumption_l3688_368895

theorem jen_jam_consumption (total_jam : ℚ) : 
  let lunch_consumption := (1 : ℚ) / 3
  let after_lunch := total_jam - lunch_consumption * total_jam
  let after_dinner := (4 : ℚ) / 7 * total_jam
  let dinner_consumption := (after_lunch - after_dinner) / after_lunch
  dinner_consumption = (1 : ℚ) / 7 := by sorry

end NUMINAMATH_CALUDE_jen_jam_consumption_l3688_368895


namespace NUMINAMATH_CALUDE_lcm_problem_l3688_368817

theorem lcm_problem (a b : ℕ+) (h_product : a * b = 18750) (h_hcf : Nat.gcd a b = 25) :
  Nat.lcm a b = 750 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l3688_368817


namespace NUMINAMATH_CALUDE_solve_for_m_l3688_368872

theorem solve_for_m (x m : ℝ) (h1 : 3 * x - 2 * m = 4) (h2 : x = m) : m = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_m_l3688_368872


namespace NUMINAMATH_CALUDE_tim_and_tina_same_age_l3688_368867

def tim_age_condition (x : ℕ) : Prop := x + 2 = 2 * (x - 2)

def tina_age_condition (y : ℕ) : Prop := y + 3 = 3 * (y - 3)

theorem tim_and_tina_same_age :
  ∃ (x y : ℕ), tim_age_condition x ∧ tina_age_condition y ∧ x = y :=
by
  sorry

end NUMINAMATH_CALUDE_tim_and_tina_same_age_l3688_368867


namespace NUMINAMATH_CALUDE_scientific_notation_equality_l3688_368842

/-- Proves that 448000 is equal to 4.48 * 10^5 in scientific notation -/
theorem scientific_notation_equality : 448000 = 4.48 * (10 ^ 5) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equality_l3688_368842


namespace NUMINAMATH_CALUDE_distinct_primes_dividing_P_l3688_368888

def P : ℕ := (List.range 10).foldl (· * ·) 1

theorem distinct_primes_dividing_P :
  (Finset.filter (fun p => Nat.Prime p ∧ P % p = 0) (Finset.range 11)).card = 4 := by
  sorry

end NUMINAMATH_CALUDE_distinct_primes_dividing_P_l3688_368888


namespace NUMINAMATH_CALUDE_molly_gift_cost_per_package_l3688_368821

/-- The cost per package for Molly's Christmas gifts --/
def cost_per_package (total_relatives : ℕ) (total_cost : ℚ) : ℚ :=
  total_cost / total_relatives

/-- Theorem: The cost per package for Molly's Christmas gifts is $5 --/
theorem molly_gift_cost_per_package :
  let total_relatives : ℕ := 14
  let total_cost : ℚ := 70
  cost_per_package total_relatives total_cost = 5 := by
  sorry


end NUMINAMATH_CALUDE_molly_gift_cost_per_package_l3688_368821


namespace NUMINAMATH_CALUDE_book_has_fifty_pages_l3688_368819

/-- Calculates the number of pages in a book based on reading speed and book structure -/
def book_pages (sentences_per_hour : ℕ) (paragraphs_per_page : ℕ) (sentences_per_paragraph : ℕ) (total_reading_hours : ℕ) : ℕ :=
  (sentences_per_hour * total_reading_hours) / (sentences_per_paragraph * paragraphs_per_page)

/-- Theorem stating that given the specific conditions, the book has 50 pages -/
theorem book_has_fifty_pages :
  book_pages 200 20 10 50 = 50 := by
  sorry

end NUMINAMATH_CALUDE_book_has_fifty_pages_l3688_368819


namespace NUMINAMATH_CALUDE_tan_fifteen_thirty_product_l3688_368829

theorem tan_fifteen_thirty_product : (1 + Real.tan (15 * π / 180)) * (1 + Real.tan (30 * π / 180)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_fifteen_thirty_product_l3688_368829


namespace NUMINAMATH_CALUDE_price_difference_proof_l3688_368839

def original_price : ℝ := 120.00
def tax_rate : ℝ := 0.07
def discount_rate : ℝ := 0.25

def amy_total : ℝ := original_price * (1 + tax_rate) * (1 - discount_rate)
def bob_total : ℝ := original_price * (1 - discount_rate) * (1 + tax_rate)
def carla_total : ℝ := original_price * (1 + tax_rate) * (1 - discount_rate) * (1 + tax_rate)

theorem price_difference_proof :
  carla_total - amy_total = 6.744 ∧ carla_total - bob_total = 6.744 :=
by sorry

end NUMINAMATH_CALUDE_price_difference_proof_l3688_368839


namespace NUMINAMATH_CALUDE_quadratic_root_range_l3688_368832

theorem quadratic_root_range (a : ℝ) :
  (∃ x y : ℝ, x^2 + (a^2 - 1)*x + a - 2 = 0 ∧ x > 1 ∧ y < 1 ∧ y^2 + (a^2 - 1)*y + a - 2 = 0) →
  a ∈ Set.Ioo (-2 : ℝ) 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l3688_368832


namespace NUMINAMATH_CALUDE_quotient_calculation_l3688_368854

theorem quotient_calculation (divisor dividend remainder quotient : ℕ) : 
  divisor = 17 → dividend = 76 → remainder = 8 → quotient = 4 →
  dividend = divisor * quotient + remainder :=
by
  sorry

end NUMINAMATH_CALUDE_quotient_calculation_l3688_368854


namespace NUMINAMATH_CALUDE_seating_arrangements_l3688_368835

/-- The number of seating arrangements for four students and two teachers under different conditions. -/
theorem seating_arrangements (n_students : Nat) (n_teachers : Nat) : n_students = 4 ∧ n_teachers = 2 →
  (∃ (arrangements_middle : Nat), arrangements_middle = 48) ∧
  (∃ (arrangements_together : Nat), arrangements_together = 144) ∧
  (∃ (arrangements_separate : Nat), arrangements_separate = 144) := by
  sorry

#check seating_arrangements

end NUMINAMATH_CALUDE_seating_arrangements_l3688_368835


namespace NUMINAMATH_CALUDE_complex_power_sum_l3688_368875

theorem complex_power_sum (z : ℂ) (h : z + 1/z = 2 * Real.cos (5 * Real.pi / 180)) :
  z^100 + 1/z^100 = -2 * Real.cos (40 * Real.pi / 180) := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l3688_368875


namespace NUMINAMATH_CALUDE_road_repair_group_size_l3688_368857

/-- The number of persons in the first group -/
def first_group_size : ℕ := 39

/-- The number of days the first group works -/
def first_group_days : ℕ := 12

/-- The number of hours per day the first group works -/
def first_group_hours_per_day : ℕ := 10

/-- The number of persons in the second group -/
def second_group_size : ℕ := 30

/-- The number of days the second group works -/
def second_group_days : ℕ := 26

/-- The number of hours per day the second group works -/
def second_group_hours_per_day : ℕ := 6

theorem road_repair_group_size :
  first_group_size * first_group_days * first_group_hours_per_day =
  second_group_size * second_group_days * second_group_hours_per_day :=
by sorry

end NUMINAMATH_CALUDE_road_repair_group_size_l3688_368857


namespace NUMINAMATH_CALUDE_new_car_distance_l3688_368810

theorem new_car_distance (old_car_speed : ℝ) (old_car_distance : ℝ) (new_car_speed : ℝ) : 
  old_car_distance = 180 →
  new_car_speed = old_car_speed * 1.15 →
  new_car_speed * (old_car_distance / old_car_speed) = 207 :=
by sorry

end NUMINAMATH_CALUDE_new_car_distance_l3688_368810


namespace NUMINAMATH_CALUDE_tangent_product_l3688_368825

theorem tangent_product (α β : ℝ) (h : 2 * Real.cos (2 * α + β) + 3 * Real.cos β = 0) :
  Real.tan (α + β) * Real.tan α = -5 := by sorry

end NUMINAMATH_CALUDE_tangent_product_l3688_368825


namespace NUMINAMATH_CALUDE_sarahs_bowling_score_l3688_368808

/-- Sarah's bowling score problem -/
theorem sarahs_bowling_score :
  ∀ (sarah_score greg_score : ℕ),
  sarah_score = greg_score + 50 →
  (sarah_score + greg_score) / 2 = 110 →
  sarah_score = 135 :=
by
  sorry

end NUMINAMATH_CALUDE_sarahs_bowling_score_l3688_368808


namespace NUMINAMATH_CALUDE_total_eggs_supplied_weekly_l3688_368833

/-- The number of eggs in a dozen -/
def eggs_per_dozen : ℕ := 12

/-- The number of dozens supplied to Store A daily -/
def dozens_to_store_A : ℕ := 5

/-- The number of eggs supplied to Store B daily -/
def eggs_to_store_B : ℕ := 30

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- Theorem: The total number of eggs supplied to both stores in a week is 630 -/
theorem total_eggs_supplied_weekly : 
  (dozens_to_store_A * eggs_per_dozen + eggs_to_store_B) * days_in_week = 630 := by
sorry

end NUMINAMATH_CALUDE_total_eggs_supplied_weekly_l3688_368833


namespace NUMINAMATH_CALUDE_count_pairs_satisfying_inequality_l3688_368855

def S : Finset ℤ := {-3, -2, -1, 0, 1, 2, 3}

theorem count_pairs_satisfying_inequality :
  (Finset.filter (fun p : ℤ × ℤ => 
    p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 ≠ p.2 ∧ p.2^2 < (5/4) * p.1^2)
    (S.product S)).card = 18 := by
  sorry

end NUMINAMATH_CALUDE_count_pairs_satisfying_inequality_l3688_368855


namespace NUMINAMATH_CALUDE_fraction_product_equals_reciprocal_of_2835_l3688_368846

theorem fraction_product_equals_reciprocal_of_2835 :
  (1 / 3 : ℚ)^4 * (1 / 5 : ℚ) * (1 / 7 : ℚ) = 1 / 2835 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_equals_reciprocal_of_2835_l3688_368846


namespace NUMINAMATH_CALUDE_marble_redistribution_l3688_368877

theorem marble_redistribution (dilan martha phillip veronica : ℕ) 
  (h1 : dilan = 14)
  (h2 : martha = 20)
  (h3 : phillip = 19)
  (h4 : veronica = 7) :
  (dilan + martha + phillip + veronica) / 4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_marble_redistribution_l3688_368877


namespace NUMINAMATH_CALUDE_solve_for_x_l3688_368843

theorem solve_for_x (x : ℝ) : 
  let M := 2*x - 2
  let N := 2*x + 3
  2*M - N = 1 → x = 4 := by
sorry

end NUMINAMATH_CALUDE_solve_for_x_l3688_368843


namespace NUMINAMATH_CALUDE_smallest_sum_arithmetic_geometric_sequence_l3688_368848

theorem smallest_sum_arithmetic_geometric_sequence 
  (A B C D : ℕ+) 
  (arith_seq : ∃ d : ℤ, (C : ℤ) - (B : ℤ) = d ∧ (B : ℤ) - (A : ℤ) = d)
  (geo_seq : ∃ r : ℚ, (C : ℚ) / (B : ℚ) = r ∧ (D : ℚ) / (C : ℚ) = r)
  (ratio : (C : ℚ) / (B : ℚ) = 7 / 4) :
  (A : ℕ) + B + C + D ≥ 97 ∧ ∃ A' B' C' D' : ℕ+, 
    (∃ d : ℤ, (C' : ℤ) - (B' : ℤ) = d ∧ (B' : ℤ) - (A' : ℤ) = d) ∧
    (∃ r : ℚ, (C' : ℚ) / (B' : ℚ) = r ∧ (D' : ℚ) / (C' : ℚ) = r) ∧
    (C' : ℚ) / (B' : ℚ) = 7 / 4 ∧
    (A' : ℕ) + B' + C' + D' = 97 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_arithmetic_geometric_sequence_l3688_368848


namespace NUMINAMATH_CALUDE_dandelion_puffs_distribution_l3688_368802

theorem dandelion_puffs_distribution (total : ℕ) (given_away : ℕ) (friends : ℕ) 
  (h_total : total = 40)
  (h_given_away : given_away = 3 + 3 + 5 + 2)
  (h_friends : friends = 3)
  (h_positive : friends > 0) :
  (total - given_away) / friends = 9 :=
sorry

end NUMINAMATH_CALUDE_dandelion_puffs_distribution_l3688_368802


namespace NUMINAMATH_CALUDE_playground_area_l3688_368828

/-- Given a rectangular landscape with specific dimensions and a playground, 
    prove that the playground area is 3200 square meters. -/
theorem playground_area (length breadth : ℝ) (playground_area : ℝ) : 
  breadth = 8 * length →
  breadth = 480 →
  playground_area = (1 / 9) * (length * breadth) →
  playground_area = 3200 := by
  sorry

end NUMINAMATH_CALUDE_playground_area_l3688_368828


namespace NUMINAMATH_CALUDE_lcm_one_to_ten_l3688_368897

theorem lcm_one_to_ten : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 := by
  sorry

#eval Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10))))))))

end NUMINAMATH_CALUDE_lcm_one_to_ten_l3688_368897


namespace NUMINAMATH_CALUDE_park_diameter_is_40_l3688_368887

/-- Represents the circular park with its components -/
structure CircularPark where
  pond_diameter : ℝ
  garden_width : ℝ
  path_width : ℝ

/-- Calculates the diameter of the outer boundary of the jogging path -/
def outer_boundary_diameter (park : CircularPark) : ℝ :=
  park.pond_diameter + 2 * (park.garden_width + park.path_width)

/-- Theorem stating that for the given park dimensions, the outer boundary diameter is 40 feet -/
theorem park_diameter_is_40 :
  let park : CircularPark := {
    pond_diameter := 12,
    garden_width := 10,
    path_width := 4
  }
  outer_boundary_diameter park = 40 := by sorry

end NUMINAMATH_CALUDE_park_diameter_is_40_l3688_368887


namespace NUMINAMATH_CALUDE_unique_valid_number_l3688_368858

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧
  n % 10 = (n / 10) % 10 ∧
  (n / 100) % 10 = n / 1000 ∧
  ∃ k : ℕ, n = k * k

theorem unique_valid_number :
  ∃! n : ℕ, is_valid_number n ∧ n = 7744 :=
sorry

end NUMINAMATH_CALUDE_unique_valid_number_l3688_368858


namespace NUMINAMATH_CALUDE_degrees_to_radians_210_l3688_368876

theorem degrees_to_radians_210 : 
  (210 : ℝ) * (π / 180) = (7 * π) / 6 := by
  sorry

end NUMINAMATH_CALUDE_degrees_to_radians_210_l3688_368876


namespace NUMINAMATH_CALUDE_lunch_percentage_theorem_l3688_368826

theorem lunch_percentage_theorem (total : ℕ) (boy_ratio girl_ratio : ℕ) 
  (boy_lunch_percent girl_lunch_percent : ℚ) :
  boy_ratio + girl_ratio > 0 →
  boy_lunch_percent ≥ 0 →
  boy_lunch_percent ≤ 1 →
  girl_lunch_percent ≥ 0 →
  girl_lunch_percent ≤ 1 →
  boy_ratio = 6 →
  girl_ratio = 4 →
  boy_lunch_percent = 6/10 →
  girl_lunch_percent = 4/10 →
  (((boy_ratio * boy_lunch_percent + girl_ratio * girl_lunch_percent) / 
    (boy_ratio + girl_ratio)) : ℚ) = 52/100 := by
  sorry

end NUMINAMATH_CALUDE_lunch_percentage_theorem_l3688_368826


namespace NUMINAMATH_CALUDE_smallest_f_one_l3688_368807

/-- A cubic polynomial f(x) with specific properties -/
noncomputable def f (r s : ℝ) (x : ℝ) : ℝ := (x - r) * (x - s) * (x - (r + s) / 2)

/-- The theorem stating the smallest value of f(1) -/
theorem smallest_f_one (r s : ℝ) :
  (r ≠ s) →
  (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    f r s (f r s x₁) = 0 ∧ f r s (f r s x₂) = 0 ∧ f r s (f r s x₃) = 0) →
  (∀ (x : ℝ), f r s (f r s x) = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃) →
  (∀ (r' s' : ℝ), r' ≠ s' → f r' s' 1 ≥ 3/8) ∧
  (∃ (r₀ s₀ : ℝ), r₀ ≠ s₀ ∧ f r₀ s₀ 1 = 3/8) := by
  sorry


end NUMINAMATH_CALUDE_smallest_f_one_l3688_368807


namespace NUMINAMATH_CALUDE_concert_stay_probability_l3688_368809

/-- The probability that at least 4 people stay for an entire concert, given the conditions. -/
theorem concert_stay_probability (total : ℕ) (certain : ℕ) (uncertain : ℕ) (p : ℚ) : 
  total = 8 →
  certain = 5 →
  uncertain = 3 →
  p = 1/3 →
  ∃ (prob : ℚ), prob = 19/27 ∧ 
    prob = (uncertain.choose 1 * p * (1-p)^2 + 
            uncertain.choose 2 * p^2 * (1-p) + 
            uncertain.choose 3 * p^3) := by
  sorry


end NUMINAMATH_CALUDE_concert_stay_probability_l3688_368809


namespace NUMINAMATH_CALUDE_meat_for_forty_burgers_l3688_368836

/-- The number of pounds of meat needed to make a given number of hamburgers -/
def meatNeeded (initialPounds : ℚ) (initialBurgers : ℕ) (targetBurgers : ℕ) : ℚ :=
  (initialPounds / initialBurgers) * targetBurgers

/-- Theorem stating that 20 pounds of meat are needed for 40 hamburgers
    given that 5 pounds of meat make 10 hamburgers -/
theorem meat_for_forty_burgers :
  meatNeeded 5 10 40 = 20 := by
  sorry

#eval meatNeeded 5 10 40

end NUMINAMATH_CALUDE_meat_for_forty_burgers_l3688_368836


namespace NUMINAMATH_CALUDE_xiao_tian_hat_l3688_368831

-- Define the type for hat numbers
inductive HatNumber
  | one
  | two
  | three
  | four
  | five

-- Define the type for people
inductive Person
  | xiaoWang
  | xiaoKong
  | xiaoTian
  | xiaoYan
  | xiaoWei

-- Define the function that assigns hat numbers to people
def hatAssignment : Person → HatNumber := sorry

-- Define the function that determines if one person can see another's hat
def canSee : Person → Person → Bool := sorry

-- State the theorem
theorem xiao_tian_hat :
  (∀ p, ¬canSee Person.xiaoWang p) →
  (∃! p, canSee Person.xiaoKong p ∧ hatAssignment p = HatNumber.four) →
  (∃ p, canSee Person.xiaoTian p ∧ hatAssignment p = HatNumber.one) →
  (¬∃ p, canSee Person.xiaoTian p ∧ hatAssignment p = HatNumber.three) →
  (∃ p₁ p₂ p₃, canSee Person.xiaoYan p₁ ∧ canSee Person.xiaoYan p₂ ∧ canSee Person.xiaoYan p₃ ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃) →
  (¬∃ p, canSee Person.xiaoYan p ∧ hatAssignment p = HatNumber.three) →
  (∃ p₁ p₂, canSee Person.xiaoWei p₁ ∧ canSee Person.xiaoWei p₂ ∧
    hatAssignment p₁ = HatNumber.three ∧ hatAssignment p₂ = HatNumber.two) →
  (∀ p₁ p₂, p₁ ≠ p₂ → hatAssignment p₁ ≠ hatAssignment p₂) →
  hatAssignment Person.xiaoTian = HatNumber.two :=
sorry

end NUMINAMATH_CALUDE_xiao_tian_hat_l3688_368831


namespace NUMINAMATH_CALUDE_add_3_15_base6_l3688_368871

/-- Represents a number in base 6 --/
def Base6 := Nat

/-- Converts a base 6 number to its decimal representation --/
def toDecimal (n : Base6) : Nat :=
  sorry

/-- Converts a decimal number to its base 6 representation --/
def toBase6 (n : Nat) : Base6 :=
  sorry

/-- Addition in base 6 --/
def addBase6 (a b : Base6) : Base6 :=
  toBase6 (toDecimal a + toDecimal b)

theorem add_3_15_base6 :
  addBase6 (toBase6 3) (toBase6 15) = toBase6 22 := by
  sorry

end NUMINAMATH_CALUDE_add_3_15_base6_l3688_368871


namespace NUMINAMATH_CALUDE_triangle_problem_l3688_368894

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_problem (t : Triangle) 
  (h1 : t.a = 1) 
  (h2 : t.b = 2 * Real.sqrt 3) 
  (h3 : t.B - t.A = π / 6) 
  (h4 : t.A + t.B + t.C = π) -- Triangle angle sum
  (h5 : t.a / Real.sin t.A = t.b / Real.sin t.B) -- Law of sines
  (h6 : t.b / Real.sin t.B = t.c / Real.sin t.C) -- Law of sines
  : Real.sin t.A = Real.sqrt 7 / 14 ∧ t.c = (11 / 7) * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l3688_368894


namespace NUMINAMATH_CALUDE_base8_to_base10_conversion_l3688_368898

/-- Converts a base 8 number to base 10 -/
def base8ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

/-- The base 8 representation of 246₈ -/
def base8Number : List Nat := [6, 4, 2]

theorem base8_to_base10_conversion :
  base8ToBase10 base8Number = 166 := by
  sorry

end NUMINAMATH_CALUDE_base8_to_base10_conversion_l3688_368898


namespace NUMINAMATH_CALUDE_inscribed_circle_diameter_l3688_368889

/-- The diameter of the inscribed circle in a triangle with side lengths 13, 14, and 15 is 8 -/
theorem inscribed_circle_diameter (DE DF EF : ℝ) (h1 : DE = 13) (h2 : DF = 14) (h3 : EF = 15) :
  let s := (DE + DF + EF) / 2
  let A := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  2 * A / s = 8 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_diameter_l3688_368889


namespace NUMINAMATH_CALUDE_math_voters_l3688_368878

theorem math_voters (total_students : ℕ) (math_percentage : ℚ) : 
  total_students = 480 → math_percentage = 40 / 100 →
  (math_percentage * total_students.cast) = 192 := by
sorry

end NUMINAMATH_CALUDE_math_voters_l3688_368878


namespace NUMINAMATH_CALUDE_dividend_calculation_l3688_368805

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 17)
  (h2 : quotient = 9)
  (h3 : remainder = 9) :
  divisor * quotient + remainder = 162 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l3688_368805


namespace NUMINAMATH_CALUDE_apple_juice_cost_l3688_368890

def orange_juice_cost : ℚ := 70 / 100
def total_bottles : ℕ := 70
def total_cost : ℚ := 4620 / 100
def orange_juice_bottles : ℕ := 42

theorem apple_juice_cost :
  let apple_juice_bottles : ℕ := total_bottles - orange_juice_bottles
  let apple_juice_total_cost : ℚ := total_cost - (orange_juice_cost * orange_juice_bottles)
  apple_juice_total_cost / apple_juice_bottles = 60 / 100 :=
by sorry

end NUMINAMATH_CALUDE_apple_juice_cost_l3688_368890


namespace NUMINAMATH_CALUDE_projections_on_concentric_circles_imply_parallelogram_l3688_368862

/-- A point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A circle in 2D space -/
structure Circle :=
  (center : Point)
  (radius : ℝ)

/-- A quadrilateral in 2D space -/
structure Quadrilateral :=
  (a b c d : Point)

/-- Projection of a point onto a line segment -/
def project (p : Point) (a b : Point) : Point :=
  sorry

/-- Check if four points form an inscribed quadrilateral in a circle -/
def is_inscribed (q : Quadrilateral) (c : Circle) : Prop :=
  sorry

/-- Check if a quadrilateral is a parallelogram -/
def is_parallelogram (q : Quadrilateral) : Prop :=
  sorry

/-- Main theorem -/
theorem projections_on_concentric_circles_imply_parallelogram 
  (q : Quadrilateral) (p1 p2 : Point) (c1 c2 : Circle) :
  c1.center = c2.center →
  c1.radius ≠ c2.radius →
  is_inscribed (Quadrilateral.mk 
    (project p1 q.a q.b) (project p1 q.b q.c) 
    (project p1 q.c q.d) (project p1 q.d q.a)) c1 →
  is_inscribed (Quadrilateral.mk 
    (project p2 q.a q.b) (project p2 q.b q.c) 
    (project p2 q.c q.d) (project p2 q.d q.a)) c2 →
  is_parallelogram q :=
sorry

end NUMINAMATH_CALUDE_projections_on_concentric_circles_imply_parallelogram_l3688_368862


namespace NUMINAMATH_CALUDE_foci_coincide_l3688_368879

/-- The value of m for which the foci of the given hyperbola and ellipse coincide -/
theorem foci_coincide (m : ℝ) : 
  (∀ x y : ℝ, y^2/2 - x^2/m = 1 ↔ (y^2/2 = 1 + x^2/m)) ∧ 
  (∀ x y : ℝ, x^2/4 + y^2/9 = 1) ∧
  (∃ c : ℝ, c^2 = 2 + m ∧ c^2 = 5) →
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_foci_coincide_l3688_368879


namespace NUMINAMATH_CALUDE_width_covered_formula_l3688_368866

/-- The width covered by n asbestos tiles -/
def width_covered (n : ℕ+) : ℝ :=
  let tile_width : ℝ := 60
  let overlap : ℝ := 10
  (n : ℝ) * (tile_width - overlap) + overlap

/-- Theorem: The width covered by n asbestos tiles is (50n + 10) cm -/
theorem width_covered_formula (n : ℕ+) :
  width_covered n = 50 * (n : ℝ) + 10 := by
  sorry

end NUMINAMATH_CALUDE_width_covered_formula_l3688_368866


namespace NUMINAMATH_CALUDE_bamboo_nine_sections_l3688_368883

theorem bamboo_nine_sections (a : ℕ → ℚ) (d : ℚ) :
  (∀ n : ℕ, n ≥ 1 ∧ n ≤ 9 → a n = a 1 + (n - 1) * d) →
  a 1 + a 2 + a 3 + a 4 = 3 →
  a 7 + a 8 + a 9 = 4 →
  a 1 = 13 / 22 :=
sorry

end NUMINAMATH_CALUDE_bamboo_nine_sections_l3688_368883


namespace NUMINAMATH_CALUDE_factorial_of_factorial_divided_by_factorial_l3688_368860

theorem factorial_of_factorial_divided_by_factorial :
  (Nat.factorial (Nat.factorial 4)) / (Nat.factorial 4) = Nat.factorial 23 := by
  sorry

end NUMINAMATH_CALUDE_factorial_of_factorial_divided_by_factorial_l3688_368860


namespace NUMINAMATH_CALUDE_cubic_function_continuous_l3688_368864

-- Define the function f(x) = x^3
def f (x : ℝ) : ℝ := x^3

-- State the theorem that f is continuous for all real x
theorem cubic_function_continuous :
  ∀ x : ℝ, ContinuousAt f x :=
by
  sorry

end NUMINAMATH_CALUDE_cubic_function_continuous_l3688_368864


namespace NUMINAMATH_CALUDE_inequality_proof_l3688_368863

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (b + c - a)^2 / ((b + c)^2 + a^2) +
  (c + a - b)^2 / ((c + a)^2 + b^2) +
  (a + b - c)^2 / ((a + b)^2 + c^2) ≥ 3/5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3688_368863


namespace NUMINAMATH_CALUDE_two_sessions_scientific_notation_l3688_368869

theorem two_sessions_scientific_notation :
  78200000000 = 7.82 * (10 : ℝ)^10 := by sorry

end NUMINAMATH_CALUDE_two_sessions_scientific_notation_l3688_368869


namespace NUMINAMATH_CALUDE_tempo_insured_fraction_l3688_368838

/-- Represents the insurance details of a tempo --/
structure TempoInsurance where
  premium_rate : Rat
  premium_amount : Rat
  original_value : Rat

/-- Calculates the fraction of the original value that is insured --/
def insured_fraction (insurance : TempoInsurance) : Rat :=
  (insurance.premium_amount / insurance.premium_rate) / insurance.original_value

/-- Theorem stating that for the given insurance details, the insured fraction is 5/7 --/
theorem tempo_insured_fraction :
  let insurance : TempoInsurance := {
    premium_rate := 3 / 100,
    premium_amount := 300,
    original_value := 14000
  }
  insured_fraction insurance = 5 / 7 := by sorry

end NUMINAMATH_CALUDE_tempo_insured_fraction_l3688_368838


namespace NUMINAMATH_CALUDE_sum_first_seven_primes_mod_eighth_prime_l3688_368845

def first_seven_primes : List Nat := [2, 3, 5, 7, 11, 13, 17]
def eighth_prime : Nat := 19

theorem sum_first_seven_primes_mod_eighth_prime :
  (first_seven_primes.sum % eighth_prime) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_seven_primes_mod_eighth_prime_l3688_368845


namespace NUMINAMATH_CALUDE_village_blocks_l3688_368820

/-- The number of blocks in a village, given the number of children per block and the total number of children. -/
def number_of_blocks (children_per_block : ℕ) (total_children : ℕ) : ℕ :=
  total_children / children_per_block

/-- Theorem: Given 6 children per block and 54 total children, there are 9 blocks in the village. -/
theorem village_blocks :
  number_of_blocks 6 54 = 9 := by
  sorry

end NUMINAMATH_CALUDE_village_blocks_l3688_368820


namespace NUMINAMATH_CALUDE_shopping_tax_rate_l3688_368884

def shopping_problem (clothing_percent : ℝ) (food_percent : ℝ) (other_percent : ℝ) 
                     (other_tax_rate : ℝ) (total_tax_rate : ℝ) : Prop :=
  clothing_percent + food_percent + other_percent = 100 ∧
  clothing_percent = 50 ∧
  food_percent = 20 ∧
  other_percent = 30 ∧
  other_tax_rate = 10 ∧
  total_tax_rate = 5 ∧
  ∃ (clothing_tax_rate : ℝ),
    clothing_tax_rate * clothing_percent + other_tax_rate * other_percent = 
    total_tax_rate * 100 ∧
    clothing_tax_rate = 4

theorem shopping_tax_rate :
  ∀ (clothing_percent food_percent other_percent other_tax_rate total_tax_rate : ℝ),
  shopping_problem clothing_percent food_percent other_percent other_tax_rate total_tax_rate →
  ∃ (clothing_tax_rate : ℝ), clothing_tax_rate = 4 :=
by
  sorry

#check shopping_tax_rate

end NUMINAMATH_CALUDE_shopping_tax_rate_l3688_368884


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_l3688_368893

theorem smallest_four_digit_divisible_by_53 :
  ∃ n : ℕ, n = 1007 ∧ 
  (∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ 53 ∣ m → n ≤ m) ∧
  1000 ≤ n ∧ n < 10000 ∧ 53 ∣ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_l3688_368893


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_4_l3688_368824

-- Define the motion equation
def s (t : ℝ) : ℝ := 1 - t + t^2

-- Define the instantaneous velocity (derivative of s)
def v (t : ℝ) : ℝ := 2 * t - 1

-- Theorem statement
theorem instantaneous_velocity_at_4 : v 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_4_l3688_368824


namespace NUMINAMATH_CALUDE_range_of_f_l3688_368899

noncomputable def f (x : ℝ) : ℝ := 3 * (x - 2)

theorem range_of_f :
  Set.range f = {y : ℝ | y < -21 ∨ y > -21} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l3688_368899


namespace NUMINAMATH_CALUDE_equation_solution_l3688_368881

theorem equation_solution : ∃ x : ℝ, x ≠ 0 ∧ (1 / x + (3 / x) / (6 / x) = 1) ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3688_368881


namespace NUMINAMATH_CALUDE_family_income_problem_l3688_368806

/-- The number of initial earning members in a family -/
def initial_members : ℕ := 4

/-- The initial average monthly income -/
def initial_average : ℚ := 735

/-- The new average monthly income after one member's death -/
def new_average : ℚ := 650

/-- The income of the deceased member -/
def deceased_income : ℚ := 990

theorem family_income_problem :
  initial_members * initial_average - (initial_members - 1) * new_average = deceased_income :=
by sorry

end NUMINAMATH_CALUDE_family_income_problem_l3688_368806


namespace NUMINAMATH_CALUDE_angle_measure_in_special_pentagon_l3688_368815

/-- Given a pentagon PQRST where ∠P ≅ ∠R ≅ ∠T and ∠Q is supplementary to ∠S,
    the measure of ∠T is 120°. -/
theorem angle_measure_in_special_pentagon (P Q R S T : ℝ) : 
  P + Q + R + S + T = 540 →  -- Sum of angles in a pentagon
  Q + S = 180 →              -- ∠Q and ∠S are supplementary
  P = T ∧ R = T →            -- ∠P ≅ ∠R ≅ ∠T
  T = 120 := by
sorry

end NUMINAMATH_CALUDE_angle_measure_in_special_pentagon_l3688_368815


namespace NUMINAMATH_CALUDE_salary_change_result_l3688_368800

def initial_salary : ℝ := 2500

def raise_percentage : ℝ := 0.10

def cut_percentage : ℝ := 0.25

def final_salary : ℝ := initial_salary * (1 + raise_percentage) * (1 - cut_percentage)

theorem salary_change_result :
  final_salary = 2062.5 := by sorry

end NUMINAMATH_CALUDE_salary_change_result_l3688_368800


namespace NUMINAMATH_CALUDE_integral_x_cos_x_over_sin_cubed_x_l3688_368814

open Real

theorem integral_x_cos_x_over_sin_cubed_x (x : ℝ) :
  deriv (fun x => - (x + cos x * sin x) / (2 * sin x ^ 2)) x = 
    x * cos x / sin x ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_integral_x_cos_x_over_sin_cubed_x_l3688_368814


namespace NUMINAMATH_CALUDE_systematic_sampling_third_event_l3688_368896

/-- Given a total of 960 students, selecting every 30th student starting from
    student number 30, the number of selected students in the interval [701, 960] is 9. -/
theorem systematic_sampling_third_event (total_students : Nat) (selection_interval : Nat) 
    (first_selected : Nat) (event_start : Nat) (event_end : Nat) : Nat :=
  have h1 : total_students = 960 := by sorry
  have h2 : selection_interval = 30 := by sorry
  have h3 : first_selected = 30 := by sorry
  have h4 : event_start = 701 := by sorry
  have h5 : event_end = 960 := by sorry
  9

#check systematic_sampling_third_event

end NUMINAMATH_CALUDE_systematic_sampling_third_event_l3688_368896


namespace NUMINAMATH_CALUDE_magnitude_of_complex_number_l3688_368851

theorem magnitude_of_complex_number (z : ℂ) : z = (4 - 2*I) / (1 + I) → Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_number_l3688_368851


namespace NUMINAMATH_CALUDE_table_price_is_56_l3688_368865

/-- The price of a chair in dollars -/
def chair_price : ℝ := sorry

/-- The price of a table in dollars -/
def table_price : ℝ := sorry

/-- The condition that the price of 2 chairs and 1 table is 60% of the price of some chairs and 2 tables -/
axiom price_ratio : ∃ x : ℝ, 2 * chair_price + table_price = 0.6 * (x * chair_price + 2 * table_price)

/-- The condition that the price of 1 table and 1 chair is $64 -/
axiom total_price : chair_price + table_price = 64

/-- Theorem stating that the price of 1 table is $56 -/
theorem table_price_is_56 : table_price = 56 := by sorry

end NUMINAMATH_CALUDE_table_price_is_56_l3688_368865
