import Mathlib

namespace range_of_a_l4002_400217

-- Define the linear function
def f (a x : ℝ) : ℝ := (2 + a) * x + (5 - a)

-- Define the condition for the graph to pass through the first, second, and third quadrants
def passes_through_123_quadrants (a : ℝ) : Prop :=
  (2 + a > 0) ∧ (5 - a > 0)

-- Theorem statement
theorem range_of_a (a : ℝ) :
  passes_through_123_quadrants a → -2 < a ∧ a < 5 := by
  sorry

end range_of_a_l4002_400217


namespace karen_cookies_to_grandparents_l4002_400207

/-- The number of cookies Karen gave to her grandparents -/
def cookies_to_grandparents (total_cookies class_size cookies_per_student cookies_for_self : ℕ) : ℕ :=
  total_cookies - (cookies_for_self + class_size * cookies_per_student)

/-- Theorem stating the number of cookies Karen gave to her grandparents -/
theorem karen_cookies_to_grandparents :
  cookies_to_grandparents 50 16 2 10 = 8 := by
  sorry

end karen_cookies_to_grandparents_l4002_400207


namespace arithmetic_sequence_proof_l4002_400245

/-- Given a sequence of real numbers satisfying the condition
    |a_m + a_n - a_(m+n)| ≤ 1 / (m + n) for all m and n,
    prove that the sequence is arithmetic with a_k = k * a_1 for all k. -/
theorem arithmetic_sequence_proof (a : ℕ → ℝ) 
    (h : ∀ m n : ℕ, |a m + a n - a (m + n)| ≤ 1 / (m + n)) :
  ∀ k : ℕ, a k = k * a 1 := by
  sorry

end arithmetic_sequence_proof_l4002_400245


namespace set_A_determination_l4002_400275

def U : Set ℕ := {0, 1, 2, 4}

theorem set_A_determination (A : Set ℕ) (h : (U \ A) = {1, 2}) : A = {0, 4} := by
  sorry

end set_A_determination_l4002_400275


namespace not_sufficient_nor_necessary_l4002_400298

theorem not_sufficient_nor_necessary (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ¬(∀ a b : ℝ, a > b → (1 / a) < (1 / b)) ∧
  ¬(∀ a b : ℝ, (1 / a) < (1 / b) → a > b) :=
sorry

end not_sufficient_nor_necessary_l4002_400298


namespace probability_three_black_face_cards_l4002_400272

theorem probability_three_black_face_cards (total_cards : ℕ) (drawn_cards : ℕ) 
  (black_face_cards : ℕ) (non_black_face_cards : ℕ) :
  total_cards = 36 →
  drawn_cards = 6 →
  black_face_cards = 8 →
  non_black_face_cards = 28 →
  (Nat.choose black_face_cards 3 * Nat.choose non_black_face_cards 3) / 
  Nat.choose total_cards drawn_cards = 11466 / 121737 := by
  sorry

end probability_three_black_face_cards_l4002_400272


namespace optimal_chip_purchase_l4002_400239

/-- Represents the purchase of chips with given constraints -/
structure ChipPurchase where
  priceA : ℕ  -- Unit price of type A chips
  priceB : ℕ  -- Unit price of type B chips
  quantityA : ℕ  -- Quantity of type A chips
  quantityB : ℕ  -- Quantity of type B chips
  total_cost : ℕ  -- Total cost of the purchase

/-- Theorem stating the optimal purchase and minimum cost -/
theorem optimal_chip_purchase :
  ∃ (purchase : ChipPurchase),
    -- Conditions
    purchase.priceB = purchase.priceA + 9 ∧
    purchase.quantityA * purchase.priceA = 3120 ∧
    purchase.quantityB * purchase.priceB = 4200 ∧
    purchase.quantityA = purchase.quantityB ∧
    purchase.quantityA + purchase.quantityB = 200 ∧
    4 * purchase.quantityA ≥ purchase.quantityB ∧
    3 * purchase.quantityA ≤ purchase.quantityB ∧
    -- Correct answer
    purchase.priceA = 26 ∧
    purchase.priceB = 35 ∧
    purchase.quantityA = 50 ∧
    purchase.quantityB = 150 ∧
    purchase.total_cost = 6550 ∧
    -- Minimum cost property
    (∀ (other : ChipPurchase),
      other.priceB = other.priceA + 9 →
      other.quantityA + other.quantityB = 200 →
      4 * other.quantityA ≥ other.quantityB →
      3 * other.quantityA ≤ other.quantityB →
      other.total_cost ≥ purchase.total_cost) :=
by
  sorry


end optimal_chip_purchase_l4002_400239


namespace fraction_simplification_l4002_400249

theorem fraction_simplification : (4 / 252 : ℚ) + (17 / 36 : ℚ) = 41 / 84 := by sorry

end fraction_simplification_l4002_400249


namespace polygon_exterior_angles_l4002_400210

theorem polygon_exterior_angles (n : ℕ) (exterior_angle : ℝ) : 
  (n : ℝ) * exterior_angle = 360 → exterior_angle = 45 → n = 8 := by
  sorry

end polygon_exterior_angles_l4002_400210


namespace sum_of_x_solutions_is_zero_l4002_400238

theorem sum_of_x_solutions_is_zero (y : ℝ) (h1 : y = 9) (h2 : ∃ x : ℝ, x^2 + y^2 = 169) :
  ∃ x₁ x₂ : ℝ, x₁^2 + y^2 = 169 ∧ x₂^2 + y^2 = 169 ∧ x₁ + x₂ = 0 := by
sorry

end sum_of_x_solutions_is_zero_l4002_400238


namespace vasily_expected_salary_l4002_400280

/-- Represents the salary distribution for graduates --/
structure SalaryDistribution where
  high : ℝ  -- Salary for 1/5 of graduates
  medium : ℝ  -- Salary for 1/10 of graduates
  low : ℝ  -- Salary for 1/20 of graduates
  default : ℝ  -- Salary for the rest

/-- Represents the given conditions of the problem --/
structure ProblemConditions where
  total_students : ℕ
  successful_graduates : ℕ
  non_graduate_salary : ℝ
  graduate_salary_dist : SalaryDistribution
  education_duration : ℕ

def expected_salary (conditions : ProblemConditions) : ℝ :=
  sorry

theorem vasily_expected_salary 
  (conditions : ProblemConditions)
  (h1 : conditions.total_students = 300)
  (h2 : conditions.successful_graduates = 270)
  (h3 : conditions.non_graduate_salary = 25000)
  (h4 : conditions.graduate_salary_dist.high = 60000)
  (h5 : conditions.graduate_salary_dist.medium = 80000)
  (h6 : conditions.graduate_salary_dist.low = 25000)
  (h7 : conditions.graduate_salary_dist.default = 40000)
  (h8 : conditions.education_duration = 4) :
  expected_salary conditions = 45025 :=
sorry

end vasily_expected_salary_l4002_400280


namespace triangle_side_length_l4002_400273

theorem triangle_side_length (A B C : ℝ) (angleA angleB : ℝ) (sideAC : ℝ) :
  angleA = π / 4 →
  angleB = 5 * π / 12 →
  sideAC = 6 →
  ∃ (sideBC : ℝ), sideBC = 6 * (Real.sqrt 3 - 1) := by
  sorry

end triangle_side_length_l4002_400273


namespace f_difference_at_three_l4002_400282

-- Define the function f
def f (x : ℝ) : ℝ := x^5 + 3*x^3 + 2*x^2 + 7*x

-- Theorem statement
theorem f_difference_at_three : f 3 - f (-3) = 690 := by
  sorry

end f_difference_at_three_l4002_400282


namespace f_has_root_in_interval_l4002_400257

def f (x : ℝ) := x^3 - 3*x - 3

theorem f_has_root_in_interval :
  ∃ c ∈ Set.Ioo 2 3, f c = 0 :=
by
  sorry

end f_has_root_in_interval_l4002_400257


namespace days_without_calls_l4002_400286

/-- The number of days in the year -/
def total_days : ℕ := 365

/-- The calling frequencies of the three grandchildren -/
def call_frequencies : List ℕ := [4, 6, 8]

/-- Calculate the number of days with at least one call -/
def days_with_calls (frequencies : List ℕ) (total : ℕ) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem days_without_calls (frequencies : List ℕ) (total : ℕ) :
  frequencies = call_frequencies → total = total_days →
  total - days_with_calls frequencies total = 244 :=
by sorry

end days_without_calls_l4002_400286


namespace complex_fraction_equality_l4002_400228

theorem complex_fraction_equality : ∃ (i : ℂ), i^2 = -1 ∧ (1 - Real.sqrt 2 * i) / (Real.sqrt 2 + 1) = -i := by
  sorry

end complex_fraction_equality_l4002_400228


namespace profit_division_ratio_l4002_400295

/-- Represents the capital contribution and duration for a business partner -/
structure Contribution where
  capital : ℕ
  months : ℕ

/-- Calculates the total capital contribution over time -/
def totalContribution (c : Contribution) : ℕ := c.capital * c.months

/-- Simplifies a ratio by dividing both numbers by their GCD -/
def simplifyRatio (a b : ℕ) : ℕ × ℕ :=
  let gcd := Nat.gcd a b
  (a / gcd, b / gcd)

/-- The main theorem stating the profit division ratio -/
theorem profit_division_ratio 
  (a_initial : ℕ) 
  (b_capital : ℕ) 
  (total_months : ℕ) 
  (b_join_month : ℕ) 
  (h1 : a_initial = 3500)
  (h2 : b_capital = 31500)
  (h3 : total_months = 12)
  (h4 : b_join_month = 10) :
  simplifyRatio 
    (totalContribution { capital := a_initial, months := total_months })
    (totalContribution { capital := b_capital, months := total_months - b_join_month }) 
  = (2, 3) := by
  sorry

end profit_division_ratio_l4002_400295


namespace system_of_equations_solutions_l4002_400288

theorem system_of_equations_solutions :
  -- First system of equations
  (∃ x y : ℝ, 2 * x - 3 * y = -2 ∧ 5 * x + 3 * y = 37 ∧ x = 5 ∧ y = 4) ∧
  -- Second system of equations
  (∃ x y : ℝ, 3 * x + 2 * y = 5 ∧ 4 * x - y = 3 ∧ x = 1 ∧ y = 1) :=
by sorry

end system_of_equations_solutions_l4002_400288


namespace wheel_radii_theorem_l4002_400215

/-- Given two wheels A and B with radii R and r respectively, 
    if the ratio of their rotational speeds is 4:5 and 
    the distance between their centers is 9, 
    then R = 2.5 and r = 2. -/
theorem wheel_radii_theorem (R r : ℝ) : 
  (4 : ℝ) / 5 = 1200 / 1500 →  -- ratio of rotational speeds
  2 * (R + r) = 9 →            -- distance between centers
  R = 2.5 ∧ r = 2 := by
  sorry


end wheel_radii_theorem_l4002_400215


namespace magic_8_ball_probability_l4002_400284

/-- The probability of getting exactly 3 successes in 7 independent trials,
    where each trial has a success probability of 3/8. -/
theorem magic_8_ball_probability : 
  (Nat.choose 7 3 : ℚ) * (3/8)^3 * (5/8)^4 = 590625/2097152 := by
  sorry

end magic_8_ball_probability_l4002_400284


namespace average_equals_seven_implies_x_equals_twelve_l4002_400232

theorem average_equals_seven_implies_x_equals_twelve :
  let numbers : List ℝ := [1, 2, 4, 5, 6, 9, 9, 10, 12, x]
  (List.sum numbers) / (List.length numbers) = 7 →
  x = 12 :=
by
  sorry

end average_equals_seven_implies_x_equals_twelve_l4002_400232


namespace divisibility_implies_value_l4002_400276

-- Define the polynomials
def f (x : ℝ) : ℝ := x^4 + 4*x^3 + 6*x^2 + 4*x + 1
def g (x p q r s : ℝ) : ℝ := x^5 + 5*x^4 + 10*p*x^3 + 10*q*x^2 + 5*r*x + s

-- State the theorem
theorem divisibility_implies_value (p q r s : ℝ) :
  (∃ h : ℝ → ℝ, ∀ x, g x p q r s = f x * h x) →
  (p + q) * r = -1.5 := by
  sorry

end divisibility_implies_value_l4002_400276


namespace loan_interest_time_l4002_400281

/-- 
Given:
- A loan of 1000 at 3% per year
- A loan of 1400 at 5% per year
- The total interest is 350

Prove that the number of years required for the total interest to reach 350 is 3.5
-/
theorem loan_interest_time (principal1 principal2 rate1 rate2 total_interest : ℝ) 
  (h1 : principal1 = 1000)
  (h2 : principal2 = 1400)
  (h3 : rate1 = 0.03)
  (h4 : rate2 = 0.05)
  (h5 : total_interest = 350) :
  (total_interest / (principal1 * rate1 + principal2 * rate2)) = 3.5 := by
  sorry

end loan_interest_time_l4002_400281


namespace remaining_marbles_l4002_400226

def initial_marbles : ℕ := 64
def marbles_given : ℕ := 14

theorem remaining_marbles :
  initial_marbles - marbles_given = 50 :=
by sorry

end remaining_marbles_l4002_400226


namespace cost_AB_flight_l4002_400270

-- Define the distances
def distance_AC : ℝ := 3000
def distance_AB : ℝ := 3250

-- Define the cost structure
def bus_cost_per_km : ℝ := 0.15
def plane_cost_per_km : ℝ := 0.10
def plane_booking_fee : ℝ := 100

-- Define the function to calculate flight cost
def flight_cost (distance : ℝ) : ℝ :=
  distance * plane_cost_per_km + plane_booking_fee

-- Theorem to prove
theorem cost_AB_flight : flight_cost distance_AB = 425 := by
  sorry

end cost_AB_flight_l4002_400270


namespace circumscribed_sphere_surface_area_l4002_400214

/-- Given a rectangular tank with length 3√3, width 1, and height 2√2,
    the surface area of its circumscribed sphere is 36π. -/
theorem circumscribed_sphere_surface_area 
  (length : ℝ) (width : ℝ) (height : ℝ)
  (h_length : length = 3 * Real.sqrt 3)
  (h_width : width = 1)
  (h_height : height = 2 * Real.sqrt 2) :
  4 * Real.pi * ((length^2 + width^2 + height^2) / 4) = 36 * Real.pi := by
sorry

end circumscribed_sphere_surface_area_l4002_400214


namespace ten_consecutive_composites_l4002_400231

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

/-- A function that checks if a number is composite -/
def isComposite (n : ℕ) : Prop := n > 1 ∧ ¬(isPrime n)

theorem ten_consecutive_composites :
  ∃ (start : ℕ), 
    start + 9 < 500 ∧
    (∀ i : ℕ, i ∈ Finset.range 10 → isComposite (start + i)) ∧
    start + 9 = 489 := by
  sorry

end ten_consecutive_composites_l4002_400231


namespace no_solution_exists_l4002_400299

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem no_solution_exists : ¬ ∃ n : ℕ, n * sum_of_digits n = 20222022 := by
  sorry

end no_solution_exists_l4002_400299


namespace tan_alpha_value_l4002_400292

theorem tan_alpha_value (α : Real)
  (h1 : α ∈ Set.Ioo (π / 2) π)
  (h2 : Real.sqrt ((1 + Real.sin α) / (1 - Real.sin α)) = (1 - 2 * Real.cos α) / (2 * Real.sin (α / 2)^2 - 1)) :
  Real.tan α = -2 := by
  sorry

end tan_alpha_value_l4002_400292


namespace quadratic_inequality_solution_set_l4002_400209

theorem quadratic_inequality_solution_set (a : ℝ) :
  let solution_set := {x : ℝ | a * x^2 - (2*a - 1) * x + (a - 1) < 0}
  if a > 0 then
    solution_set = {x : ℝ | (a - 1) / a < x ∧ x < 1}
  else if a = 0 then
    solution_set = {x : ℝ | x < 1}
  else
    solution_set = {x : ℝ | x > (a - 1) / a ∨ x < 1} :=
by sorry

end quadratic_inequality_solution_set_l4002_400209


namespace may_day_travelers_l4002_400240

def scientific_notation (n : ℕ) (c : ℝ) (e : ℤ) : Prop :=
  (1 ≤ c) ∧ (c < 10) ∧ (n = c * (10 : ℝ) ^ e)

theorem may_day_travelers :
  scientific_notation 213000000 2.13 8 :=
by sorry

end may_day_travelers_l4002_400240


namespace steel_making_experiment_l4002_400285

/-- The 0.618 method calculation for steel-making experiment --/
theorem steel_making_experiment (lower upper : ℝ) (h1 : lower = 500) (h2 : upper = 1000) :
  lower + (upper - lower) * 0.618 = 809 :=
by sorry

end steel_making_experiment_l4002_400285


namespace original_savings_l4002_400277

def lindas_savings : ℚ → Prop :=
  λ s => (1 / 4 : ℚ) * s = 450

theorem original_savings : ∃ s : ℚ, lindas_savings s ∧ s = 1800 :=
  sorry

end original_savings_l4002_400277


namespace quadratic_two_real_roots_l4002_400266

theorem quadratic_two_real_roots (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - x + k + 1 = 0 ∧ y^2 - y + k + 1 = 0) → k ≤ -3/4 := by
  sorry

end quadratic_two_real_roots_l4002_400266


namespace lcm_of_5_6_8_21_l4002_400268

theorem lcm_of_5_6_8_21 : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 8 21)) = 840 := by sorry

end lcm_of_5_6_8_21_l4002_400268


namespace runner_problem_l4002_400296

theorem runner_problem (v : ℝ) (h : v > 0) : 
  (40 / v = 20 / v + 11) → (40 / (v / 2) = 22) :=
by
  sorry

end runner_problem_l4002_400296


namespace typist_salary_calculation_l4002_400267

theorem typist_salary_calculation (original_salary : ℝ) (raise_percentage : ℝ) (reduction_percentage : ℝ) : 
  original_salary = 2000 ∧ 
  raise_percentage = 10 ∧ 
  reduction_percentage = 5 → 
  original_salary * (1 + raise_percentage / 100) * (1 - reduction_percentage / 100) = 2090 :=
by sorry

end typist_salary_calculation_l4002_400267


namespace isabella_currency_exchange_l4002_400254

theorem isabella_currency_exchange :
  ∃ d : ℕ+, 
    (10 : ℚ) / 7 * d.val - 60 = d.val ∧ 
    (d.val / 100 + (d.val / 10) % 10 + d.val % 10 = 5) := by
  sorry

end isabella_currency_exchange_l4002_400254


namespace exp_addition_property_l4002_400246

-- Define the function f(x) = e^x
noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- State the theorem
theorem exp_addition_property (x y : ℝ) : f (x + y) = f x * f y := by sorry

end exp_addition_property_l4002_400246


namespace sum_of_fractions_l4002_400229

theorem sum_of_fractions : 
  (1 / 10 : ℚ) + (2 / 10 : ℚ) + (3 / 10 : ℚ) + (4 / 10 : ℚ) + (5 / 10 : ℚ) + 
  (6 / 10 : ℚ) + (7 / 10 : ℚ) + (8 / 10 : ℚ) + (10 / 10 : ℚ) + (60 / 10 : ℚ) = 
  (106 : ℚ) / 10 := by
  sorry

#eval (106 : ℚ) / 10  -- This should evaluate to 10.6

end sum_of_fractions_l4002_400229


namespace line_properties_l4002_400230

def line_equation (x y : ℝ) : Prop := 3 * y = 4 * x - 9

theorem line_properties :
  (∃ m : ℝ, m = 4/3 ∧ ∀ x y : ℝ, line_equation x y → y = m * x + (-3)) ∧
  line_equation 3 1 :=
sorry

end line_properties_l4002_400230


namespace fraction_simplification_l4002_400263

theorem fraction_simplification :
  1 / (1 / (1/3)^1 + 1 / (1/3)^2 + 1 / (1/3)^3 + 1 / (1/3)^4) = 1 / 120 := by
  sorry

end fraction_simplification_l4002_400263


namespace equation_solutions_l4002_400297

theorem equation_solutions : 
  (∃ (s : Set ℝ), s = {0, 3} ∧ ∀ x ∈ s, 4 * x^2 = 12 * x) ∧
  (∃ (t : Set ℝ), t = {-3, -1} ∧ ∀ x ∈ t, x^2 + 4 * x + 3 = 0) :=
by sorry

end equation_solutions_l4002_400297


namespace triangle_abc_properties_l4002_400241

theorem triangle_abc_properties (A B C : Real) (a b c : Real) (S : Real) :
  -- Given conditions
  (3 * Real.cos B * Real.cos C + 2 = 3 * Real.sin B * Real.sin C + 2 * Real.cos (2 * A)) →
  (S = 5 * Real.sqrt 3) →
  (b = 5) →
  -- Triangle inequality and positive side lengths
  (a + b > c ∧ b + c > a ∧ c + a > b) →
  (a > 0 ∧ b > 0 ∧ c > 0) →
  -- Angle sum in a triangle
  (A + B + C = Real.pi) →
  -- Area formula
  (S = 1/2 * b * c * Real.sin A) →
  -- Conclusions
  (A = Real.pi / 3 ∧ Real.sin B * Real.sin C = 5 / 7) := by
  sorry


end triangle_abc_properties_l4002_400241


namespace inverse_sqrt_problem_l4002_400243

-- Define the relationship between x and y
def inverse_sqrt_relation (x y : ℝ) (k : ℝ) : Prop :=
  y * Real.sqrt x = k

-- Define the theorem
theorem inverse_sqrt_problem (x y k : ℝ) :
  inverse_sqrt_relation x y k →
  inverse_sqrt_relation 2 4 k →
  y = 1 →
  x = 32 := by sorry

end inverse_sqrt_problem_l4002_400243


namespace stream_speed_l4002_400224

/-- Proves that the speed of a stream is 3 km/h given specific rowing conditions. -/
theorem stream_speed (downstream_distance : ℝ) (upstream_distance : ℝ) (trip_time : ℝ) 
  (h1 : downstream_distance = 75)
  (h2 : upstream_distance = 45)
  (h3 : trip_time = 5) :
  ∃ (boat_speed stream_speed : ℝ),
    downstream_distance = trip_time * (boat_speed + stream_speed) ∧
    upstream_distance = trip_time * (boat_speed - stream_speed) ∧
    stream_speed = 3 := by
  sorry

end stream_speed_l4002_400224


namespace new_capacity_is_250_l4002_400205

/-- Calculates the new combined total lifting capacity after improvements -/
def new_total_capacity (initial_clean_jerk : ℝ) (initial_snatch : ℝ) : ℝ :=
  (2 * initial_clean_jerk) + (initial_snatch + 0.8 * initial_snatch)

/-- Theorem stating that given the initial capacities and improvements, 
    the new total capacity is 250 kg -/
theorem new_capacity_is_250 :
  new_total_capacity 80 50 = 250 := by
  sorry

end new_capacity_is_250_l4002_400205


namespace fraction_calculation_l4002_400221

theorem fraction_calculation : 
  (3/7 + 2/3) / (5/12 + 1/4) = 23/14 := by
  sorry

end fraction_calculation_l4002_400221


namespace cheryl_same_color_probability_l4002_400227

/-- Represents the number of marbles of each color -/
def marbles_per_color : ℕ := 3

/-- Represents the number of colors -/
def num_colors : ℕ := 3

/-- Represents the total number of marbles -/
def total_marbles : ℕ := marbles_per_color * num_colors

/-- Represents the number of marbles each person draws -/
def marbles_drawn : ℕ := 3

/-- Calculates the probability of Cheryl getting 3 marbles of the same color -/
theorem cheryl_same_color_probability :
  (num_colors * (Nat.choose (total_marbles - 2 * marbles_drawn) marbles_drawn)) /
  (Nat.choose total_marbles marbles_drawn * Nat.choose (total_marbles - marbles_drawn) marbles_drawn) = 1 / 28 :=
by sorry

end cheryl_same_color_probability_l4002_400227


namespace min_distance_to_line_l4002_400200

/-- The curve function f(x) = x^2 - ln x -/
noncomputable def f (x : ℝ) : ℝ := x^2 - Real.log x

/-- The line function g(x) = x - 2 -/
def g (x : ℝ) : ℝ := x - 2

/-- A point P on the curve -/
structure PointOnCurve where
  x : ℝ
  y : ℝ
  h : y = f x

/-- Theorem: The minimum distance from any point on the curve to the line is 1 -/
theorem min_distance_to_line (P : PointOnCurve) : 
  ∃ (d : ℝ), d = 1 ∧ ∀ (Q : ℝ × ℝ), Q.2 = g Q.1 → Real.sqrt ((P.x - Q.1)^2 + (P.y - Q.2)^2) ≥ d :=
sorry

end min_distance_to_line_l4002_400200


namespace symmetric_points_sum_l4002_400251

/-- A point in a 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of symmetry with respect to x-axis --/
def symmetricXAxis (p q : Point) : Prop :=
  p.x = q.x ∧ p.y = -q.y

/-- Theorem: If A(3,a) is symmetric to B(b,4) with respect to x-axis, then a + b = -1 --/
theorem symmetric_points_sum (a b : ℝ) : 
  let A : Point := ⟨3, a⟩
  let B : Point := ⟨b, 4⟩
  symmetricXAxis A B → a + b = -1 := by
  sorry

end symmetric_points_sum_l4002_400251


namespace marys_income_percentage_l4002_400225

theorem marys_income_percentage (juan tim mary : ℝ) 
  (h1 : tim = juan * 0.9) 
  (h2 : mary = tim * 1.6) : 
  mary = juan * 1.44 := by
sorry

end marys_income_percentage_l4002_400225


namespace max_distance_on_curve_l4002_400234

/-- The maximum distance between a point on the curve y² = 4 - 2x² and the point (0, -√2) -/
theorem max_distance_on_curve : ∃ (max_dist : ℝ),
  max_dist = 2 + Real.sqrt 2 ∧
  ∀ (x y : ℝ),
    y^2 = 4 - 2*x^2 →
    Real.sqrt ((x - 0)^2 + (y - (-Real.sqrt 2))^2) ≤ max_dist :=
by sorry

end max_distance_on_curve_l4002_400234


namespace player_b_winning_strategy_l4002_400269

/-- Represents the game state with two players on a line -/
structure GameState where
  L : ℕ+  -- Distance between initial positions (positive integer)
  a : ℕ+  -- Move distance for player A (positive integer)
  b : ℕ+  -- Move distance for player B (positive integer)
  h : a < b  -- Condition that a is less than b

/-- Winning condition for player B -/
def winning_condition (g : GameState) : Prop :=
  g.b = 2 * g.a ∧ ∃ k : ℕ, g.L = k * g.a

/-- Theorem stating the necessary and sufficient conditions for player B to have a winning strategy -/
theorem player_b_winning_strategy (g : GameState) :
  winning_condition g ↔ ∃ (strategy : Unit), True  -- Replace True with actual strategy type when implementing
:= by sorry

end player_b_winning_strategy_l4002_400269


namespace soccer_campers_count_l4002_400201

/-- The number of soccer campers at a summer sports camp -/
def soccer_campers (total : ℕ) (basketball : ℕ) (football : ℕ) : ℕ :=
  total - (basketball + football)

/-- Theorem stating the number of soccer campers given the conditions -/
theorem soccer_campers_count :
  soccer_campers 88 24 32 = 32 := by
  sorry

end soccer_campers_count_l4002_400201


namespace problem_solution_l4002_400264

theorem problem_solution (p q : ℚ) (h1 : 3 / p = 6) (h2 : 3 / q = 18) : p - q = 1/3 := by
  sorry

end problem_solution_l4002_400264


namespace max_chocolates_buyable_l4002_400287

def total_money : ℚ := 24.50
def chocolate_price : ℚ := 2.20

theorem max_chocolates_buyable : 
  ⌊total_money / chocolate_price⌋ = 11 := by sorry

end max_chocolates_buyable_l4002_400287


namespace linear_function_proof_l4002_400247

-- Define the linear function
def linear_function (k b : ℝ) (x : ℝ) : ℝ := k * x + b

-- Define the theorem
theorem linear_function_proof :
  ∃ (k b : ℝ),
    (linear_function k b 1 = 5) ∧
    (linear_function k b (-1) = -1) ∧
    (∀ (x : ℝ), linear_function k b x = 3 * x + 2) ∧
    (linear_function k b 2 = 8) := by
  sorry


end linear_function_proof_l4002_400247


namespace exact_time_proof_l4002_400219

def minutes_after_3 (h m : ℕ) : ℝ := 60 * (h - 3 : ℝ) + m

def minute_hand_position (t : ℝ) : ℝ := 6 * t

def hour_hand_position (t : ℝ) : ℝ := 90 + 0.5 * t

theorem exact_time_proof :
  ∃ (h m : ℕ), h = 3 ∧ m < 60 ∧
  let t := minutes_after_3 h m
  abs (minute_hand_position (t + 5) - hour_hand_position (t - 4)) = 178 ∧
  h = 3 ∧ m = 43 := by
  sorry

end exact_time_proof_l4002_400219


namespace rational_sqrt5_zero_quadratic_roots_sum_difference_l4002_400274

theorem rational_sqrt5_zero (a b : ℚ) (h : a + b * Real.sqrt 5 = 0) : a = 0 ∧ b = 0 := by
  sorry

theorem quadratic_roots_sum_difference (k : ℝ) :
  k ≠ 0 →
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    4 * k * x₁^2 - 4 * k * x₁ + k + 1 = 0 ∧
    4 * k * x₂^2 - 4 * k * x₂ + k + 1 = 0 ∧
    x₁^2 + x₂^2 - 2 * x₁ * x₂ = 1/2) →
  k = -2 := by
  sorry

end rational_sqrt5_zero_quadratic_roots_sum_difference_l4002_400274


namespace largest_inscribed_circle_radius_l4002_400223

-- Define the quadrilateral ABCD
def Quadrilateral (A B C D : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let CD := Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2)
  let DA := Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2)
  AB = 13 ∧ BC = 10 ∧ CD = 8 ∧ DA = 11

-- Define the inscribed circle
def InscribedCircle (A B C D : ℝ × ℝ) (O : ℝ × ℝ) (r : ℝ) : Prop :=
  Quadrilateral A B C D ∧
  ∀ P : ℝ × ℝ, (P ∈ Set.range (fun t => (t * B.1 + (1 - t) * A.1, t * B.2 + (1 - t) * A.2)) ∨
                 P ∈ Set.range (fun t => (t * C.1 + (1 - t) * B.1, t * C.2 + (1 - t) * B.2)) ∨
                 P ∈ Set.range (fun t => (t * D.1 + (1 - t) * C.1, t * D.2 + (1 - t) * C.2)) ∨
                 P ∈ Set.range (fun t => (t * A.1 + (1 - t) * D.1, t * A.2 + (1 - t) * D.2))) →
                Real.sqrt ((P.1 - O.1)^2 + (P.2 - O.2)^2) ≥ r

-- Theorem statement
theorem largest_inscribed_circle_radius :
  ∀ A B C D O : ℝ × ℝ,
  ∀ r : ℝ,
  InscribedCircle A B C D O r →
  r ≤ 2 * Real.sqrt 6 :=
sorry

end largest_inscribed_circle_radius_l4002_400223


namespace factorial_sum_equality_l4002_400233

theorem factorial_sum_equality : 7 * Nat.factorial 7 + 6 * Nat.factorial 6 + 2 * Nat.factorial 5 + Nat.factorial 5 = 39960 := by
  sorry

end factorial_sum_equality_l4002_400233


namespace problem_statement_l4002_400212

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : (a+2)*(b+2) = 18) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ (x+2)*(y+2) = 18 ∧ 3/(x+2) + 3/(y+2) < 3/(a+2) + 3/(b+2)) ∨
  (3/(a+2) + 3/(b+2) = Real.sqrt 2) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → (x+2)*(y+2) = 18 → 2*x + y ≥ 6) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → (x+2)*(y+2) = 18 → (x+1)*y ≤ 8) := by
  sorry

end problem_statement_l4002_400212


namespace gcf_lcm_sum_4_8_l4002_400237

theorem gcf_lcm_sum_4_8 : Nat.gcd 4 8 + Nat.lcm 4 8 = 12 := by
  sorry

end gcf_lcm_sum_4_8_l4002_400237


namespace candy_necklace_problem_l4002_400293

/-- Candy necklace problem -/
theorem candy_necklace_problem 
  (pieces_per_necklace : ℕ) 
  (pieces_per_block : ℕ) 
  (blocks_used : ℕ) 
  (h1 : pieces_per_necklace = 10)
  (h2 : pieces_per_block = 30)
  (h3 : blocks_used = 3)
  : (blocks_used * pieces_per_block) / pieces_per_necklace - 1 = 8 := by
  sorry

#check candy_necklace_problem

end candy_necklace_problem_l4002_400293


namespace cube_plus_reciprocal_cube_l4002_400248

theorem cube_plus_reciprocal_cube (r : ℝ) (h : (r + 1/r)^2 = 3) : 
  r^3 + 1/r^3 = 0 := by sorry

end cube_plus_reciprocal_cube_l4002_400248


namespace sum_of_five_consecutive_even_integers_l4002_400289

theorem sum_of_five_consecutive_even_integers (m : ℤ) : 
  (m + (m + 2) + (m + 4) + (m + 6) + (m + 8) = 5 * m + 20) := by
  sorry

end sum_of_five_consecutive_even_integers_l4002_400289


namespace hotel_stay_cost_l4002_400291

/-- The total cost for a group staying at a hotel. -/
def total_cost (cost_per_night_per_person : ℕ) (num_people : ℕ) (num_nights : ℕ) : ℕ :=
  cost_per_night_per_person * num_people * num_nights

/-- Theorem: The total cost for 3 people staying 3 nights at $40 per night per person is $360. -/
theorem hotel_stay_cost : total_cost 40 3 3 = 360 := by
  sorry

end hotel_stay_cost_l4002_400291


namespace joans_balloons_l4002_400202

theorem joans_balloons (initial_balloons final_balloons : ℕ) 
  (h1 : initial_balloons = 8)
  (h2 : final_balloons = 10) :
  final_balloons - initial_balloons = 2 := by
  sorry

end joans_balloons_l4002_400202


namespace optimal_pricing_strategy_l4002_400261

/-- Represents the pricing strategy of a merchant -/
structure MerchantPricing where
  list_price : ℝ
  purchase_discount : ℝ
  marked_price : ℝ
  sale_discount : ℝ
  profit_margin : ℝ

/-- Calculates the purchase price given the list price and purchase discount -/
def purchase_price (m : MerchantPricing) : ℝ :=
  m.list_price * (1 - m.purchase_discount)

/-- Calculates the selling price given the marked price and sale discount -/
def selling_price (m : MerchantPricing) : ℝ :=
  m.marked_price * (1 - m.sale_discount)

/-- Calculates the profit given the selling price and purchase price -/
def profit (m : MerchantPricing) : ℝ :=
  selling_price m - purchase_price m

/-- Theorem stating the optimal pricing strategy for the merchant -/
theorem optimal_pricing_strategy (m : MerchantPricing) 
  (h1 : m.purchase_discount = 0.25)
  (h2 : m.sale_discount = 0.20)
  (h3 : m.profit_margin = 0.25)
  (h4 : profit m = m.profit_margin * selling_price m) :
  m.marked_price = 1.25 * m.list_price := by
  sorry

end optimal_pricing_strategy_l4002_400261


namespace road_repair_groups_equivalent_l4002_400220

/-- The number of persons in the second group repairing the road -/
def second_group_size : ℕ := 30

/-- The number of persons in the first group -/
def first_group_size : ℕ := 39

/-- The number of days the first group works -/
def first_group_days : ℕ := 12

/-- The number of hours per day the first group works -/
def first_group_hours_per_day : ℕ := 10

/-- The number of days the second group works -/
def second_group_days : ℕ := 26

/-- The number of hours per day the second group works -/
def second_group_hours_per_day : ℕ := 6

/-- The total man-hours required to complete the road repair -/
def total_man_hours : ℕ := first_group_size * first_group_days * first_group_hours_per_day

theorem road_repair_groups_equivalent :
  second_group_size * second_group_days * second_group_hours_per_day = total_man_hours :=
by sorry

end road_repair_groups_equivalent_l4002_400220


namespace parallel_vectors_x_value_l4002_400262

theorem parallel_vectors_x_value (x : ℝ) :
  let a : ℝ × ℝ := (2*x + 1, 3)
  let b : ℝ × ℝ := (2 - x, 1)
  (∃ (k : ℝ), a.1 = k * b.1 ∧ a.2 = k * b.2) → x = 1 := by
  sorry

end parallel_vectors_x_value_l4002_400262


namespace abs_inequality_equivalence_l4002_400256

theorem abs_inequality_equivalence (x : ℝ) : 
  (2 ≤ |x - 3| ∧ |x - 3| ≤ 8) ↔ ((-5 ≤ x ∧ x ≤ 1) ∨ (5 ≤ x ∧ x ≤ 11)) :=
by sorry

end abs_inequality_equivalence_l4002_400256


namespace tree_spacing_l4002_400216

theorem tree_spacing (yard_length : ℝ) (num_trees : ℕ) :
  yard_length = 350 →
  num_trees = 26 →
  yard_length / (num_trees - 1) = 14 :=
by sorry

end tree_spacing_l4002_400216


namespace probability_expired_20_2_l4002_400206

/-- The probability of selecting an expired bottle from a set of bottles -/
def probability_expired (total : ℕ) (expired : ℕ) : ℚ :=
  (expired : ℚ) / (total : ℚ)

/-- Theorem: The probability of selecting an expired bottle from 20 bottles, where 2 are expired, is 1/10 -/
theorem probability_expired_20_2 :
  probability_expired 20 2 = 1 / 10 := by
  sorry

end probability_expired_20_2_l4002_400206


namespace expression_simplification_l4002_400235

theorem expression_simplification (x y : ℚ) (hx : x = 4) (hy : y = -1/4) :
  ((x + y) * (3 * x - y) + y^2) / (-x) = -23/2 := by
  sorry

end expression_simplification_l4002_400235


namespace taylor_family_reunion_l4002_400211

theorem taylor_family_reunion (kids : ℕ) (tables : ℕ) (people_per_table : ℕ) (adults : ℕ) : 
  kids = 45 → tables = 14 → people_per_table = 12 → 
  adults = tables * people_per_table - kids → adults = 123 :=
by
  sorry

end taylor_family_reunion_l4002_400211


namespace prob_queen_then_diamond_correct_l4002_400252

/-- The probability of drawing a Queen first and a diamond second from a standard 52-card deck, without replacement -/
def prob_queen_then_diamond : ℚ := 18 / 221

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of Queens in a standard deck -/
def num_queens : ℕ := 4

/-- The number of diamond cards in a standard deck -/
def num_diamonds : ℕ := 13

/-- The number of non-diamond Queens in a standard deck -/
def num_non_diamond_queens : ℕ := 3

theorem prob_queen_then_diamond_correct :
  prob_queen_then_diamond = 
    (1 / deck_size * num_diamonds / (deck_size - 1)) + 
    (num_non_diamond_queens / deck_size * num_diamonds / (deck_size - 1)) := by
  sorry

end prob_queen_then_diamond_correct_l4002_400252


namespace relay_team_orders_l4002_400271

/-- The number of permutations of n distinct objects -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- There are 5 team members -/
def team_size : ℕ := 5

/-- Lara always runs the last lap, so we need to arrange the other 4 runners -/
def runners_to_arrange : ℕ := team_size - 1

theorem relay_team_orders : permutations runners_to_arrange = 24 := by
  sorry

end relay_team_orders_l4002_400271


namespace inscribed_circle_segment_ratio_l4002_400259

-- Define the triangle and circle
def Triangle (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

def InscribedCircle (t : Triangle a b c) := 
  ∃ (r : ℝ), r > 0 ∧ 
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
  x + y = a ∧ y + z = b ∧ z + x = c ∧
  x + y + z = (a + b + c) / 2

-- Define the theorem
theorem inscribed_circle_segment_ratio 
  (t : Triangle 10 15 19) 
  (c : InscribedCircle t) :
  ∃ (r s : ℝ), r > 0 ∧ s > 0 ∧ r < s ∧ r + s = 10 ∧ r / s = 3 / 7 :=
sorry

end inscribed_circle_segment_ratio_l4002_400259


namespace sum_of_max_min_g_l4002_400294

-- Define the function g(x)
def g (x : ℝ) : ℝ := |x - 3| + |x - 6| - |3*x - 9|

-- Define the domain of x
def domain (x : ℝ) : Prop := 3 ≤ x ∧ x ≤ 12

-- Theorem statement
theorem sum_of_max_min_g :
  ∃ (max min : ℝ), 
    (∀ x, domain x → g x ≤ max) ∧
    (∃ x, domain x ∧ g x = max) ∧
    (∀ x, domain x → min ≤ g x) ∧
    (∃ x, domain x ∧ g x = min) ∧
    max + min = -6 :=
sorry

end sum_of_max_min_g_l4002_400294


namespace max_value_of_sum_products_l4002_400260

theorem max_value_of_sum_products (a b c d : ℕ) : 
  ({a, b, c, d} : Finset ℕ) = {1, 2, 4, 5} →
  a * b + b * c + c * d + d * a ≤ 36 :=
by
  sorry

end max_value_of_sum_products_l4002_400260


namespace simplify_and_evaluate_l4002_400236

theorem simplify_and_evaluate (x : ℝ) (h : x = 4) :
  (x - 1 - 3 / (x + 1)) / ((x^2 + 2*x) / (x + 1)) = 1/2 := by
  sorry

end simplify_and_evaluate_l4002_400236


namespace ellipse_equation_l4002_400208

/-- The standard equation of an ellipse with given eccentricity and major axis length -/
theorem ellipse_equation (e : ℝ) (major_axis : ℝ) (h_e : e = 1 / 2) (h_major : major_axis = 4) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a > b ∧
  (∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1) ↔ (x^2 / 4 + y^2 / 3 = 1)) :=
by sorry

end ellipse_equation_l4002_400208


namespace vector_subtraction_l4002_400244

def a : Fin 3 → ℝ := ![(-5 : ℝ), 1, 3]
def b : Fin 3 → ℝ := ![(3 : ℝ), -1, 2]

theorem vector_subtraction :
  a - 2 • b = ![(-11 : ℝ), 3, -1] := by sorry

end vector_subtraction_l4002_400244


namespace hundreds_digit_of_8_pow_1234_l4002_400218

-- Define a function to get the last three digits of 8^n
def lastThreeDigits (n : ℕ) : ℕ := 8^n % 1000

-- Define the cycle length of the last three digits of 8^n
def cycleLengthOf8 : ℕ := 20

-- Theorem statement
theorem hundreds_digit_of_8_pow_1234 :
  (lastThreeDigits 1234) / 100 = 1 :=
sorry

end hundreds_digit_of_8_pow_1234_l4002_400218


namespace student_takehome_pay_l4002_400250

/-- Calculates the take-home pay for a well-performing student at a fast-food chain --/
def takehomePay (baseSalary bonus taxRate : ℚ) : ℚ :=
  let totalEarnings := baseSalary + bonus
  let incomeTax := totalEarnings * taxRate
  totalEarnings - incomeTax

/-- Theorem: The take-home pay for a well-performing student is 26,100 rubles --/
theorem student_takehome_pay :
  takehomePay 25000 5000 (13/100) = 26100 := by
  sorry

#eval takehomePay 25000 5000 (13/100)

end student_takehome_pay_l4002_400250


namespace olivia_spent_l4002_400279

/-- Calculates the amount spent given initial amount, amount collected, and amount left after shopping. -/
def amount_spent (initial : ℕ) (collected : ℕ) (left : ℕ) : ℕ :=
  initial + collected - left

/-- Proves that Olivia spent 89 dollars given the problem conditions. -/
theorem olivia_spent (initial : ℕ) (collected : ℕ) (left : ℕ)
  (h1 : initial = 100)
  (h2 : collected = 148)
  (h3 : left = 159) :
  amount_spent initial collected left = 89 := by
  sorry

#eval amount_spent 100 148 159

end olivia_spent_l4002_400279


namespace total_peppers_calculation_l4002_400242

/-- The amount of green peppers bought by Dale's Vegetarian Restaurant in pounds -/
def green_peppers : ℝ := 2.8333333333333335

/-- The amount of red peppers bought by Dale's Vegetarian Restaurant in pounds -/
def red_peppers : ℝ := 2.8333333333333335

/-- The total amount of peppers bought by Dale's Vegetarian Restaurant in pounds -/
def total_peppers : ℝ := green_peppers + red_peppers

theorem total_peppers_calculation :
  total_peppers = 5.666666666666667 := by sorry

end total_peppers_calculation_l4002_400242


namespace largest_n_for_factorization_l4002_400283

/-- 
Theorem: The largest value of n for which 2x^2 + nx + 50 can be factored 
as the product of two linear factors with integer coefficients is 101.
-/
theorem largest_n_for_factorization : 
  (∃ (n : ℤ), ∀ (m : ℤ), 
    (∃ (a b : ℤ), 2 * X^2 + n * X + 50 = (2 * X + a) * (X + b)) → 
    m ≤ n) ∧ 
  (∃ (a b : ℤ), 2 * X^2 + 101 * X + 50 = (2 * X + a) * (X + b)) :=
by sorry


end largest_n_for_factorization_l4002_400283


namespace line_bisects_circle_l4002_400204

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 4*x - 4*y - 8 = 0

/-- The equation of the line -/
def line_equation (x y b : ℝ) : Prop :=
  y = x + b

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (-2, 2)

/-- The line bisects the circumference of the circle if it passes through the center -/
def bisects_circle (b : ℝ) : Prop :=
  let (cx, cy) := circle_center
  line_equation cx cy b

theorem line_bisects_circle (b : ℝ) :
  bisects_circle b → b = 4 := by sorry

end line_bisects_circle_l4002_400204


namespace log_sum_equality_l4002_400258

theorem log_sum_equality : 2 * Real.log 10 / Real.log 5 + Real.log 0.25 / Real.log 5 = 2 := by
  sorry

end log_sum_equality_l4002_400258


namespace sales_price_calculation_l4002_400253

theorem sales_price_calculation (C S : ℝ) 
  (h1 : 1.7 * C = 51)  -- Gross profit is 170% of cost and equals $51
  (h2 : S = C + 1.7 * C)  -- Sales price is cost plus gross profit
  : S = 81 := by
  sorry

end sales_price_calculation_l4002_400253


namespace book_pages_count_l4002_400213

/-- The total number of pages in Isabella's book -/
def total_pages : ℕ := 288

/-- The number of days Isabella took to read the book -/
def total_days : ℕ := 8

/-- The average number of pages Isabella read daily for the first four days -/
def first_four_days_avg : ℕ := 28

/-- The average number of pages Isabella read daily for the next three days -/
def next_three_days_avg : ℕ := 52

/-- The number of pages Isabella read on the final day -/
def final_day_pages : ℕ := 20

/-- Theorem stating that the total number of pages in the book is 288 -/
theorem book_pages_count : 
  (4 * first_four_days_avg + 3 * next_three_days_avg + final_day_pages = total_pages) ∧
  (total_days = 8) := by
  sorry

end book_pages_count_l4002_400213


namespace solve_for_a_l4002_400265

theorem solve_for_a : 
  ∀ a : ℝ, (∃ x : ℝ, x = 3 ∧ a * x - 5 = x + 1) → a = 3 :=
by
  sorry

end solve_for_a_l4002_400265


namespace largest_integer_for_negative_quadratic_six_satisfies_inequality_seven_does_not_satisfy_inequality_l4002_400278

theorem largest_integer_for_negative_quadratic : 
  ∀ x : ℤ, x^2 - 11*x + 28 < 0 → x ≤ 6 :=
by sorry

theorem six_satisfies_inequality : 
  (6 : ℤ)^2 - 11*6 + 28 < 0 :=
by sorry

theorem seven_does_not_satisfy_inequality : 
  (7 : ℤ)^2 - 11*7 + 28 ≥ 0 :=
by sorry

end largest_integer_for_negative_quadratic_six_satisfies_inequality_seven_does_not_satisfy_inequality_l4002_400278


namespace arithmetic_sequence_max_product_l4002_400222

theorem arithmetic_sequence_max_product (a : ℕ → ℝ) (d : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  a 8 + a 9 + a 10 = 24 →           -- given sum condition
  ∃ m : ℝ, m = 2 ∧ ∀ d' : ℝ, a 1 * d' ≤ m := by
  sorry

end arithmetic_sequence_max_product_l4002_400222


namespace distribute_5_3_l4002_400255

/-- The number of ways to distribute n distinct objects into k distinct boxes,
    where each box must contain at least one object. -/
def distribute (n k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 150 ways to distribute 5 distinct objects into 3 distinct boxes,
    where each box must contain at least one object. -/
theorem distribute_5_3 : distribute 5 3 = 150 := by
  sorry

end distribute_5_3_l4002_400255


namespace pentagon_angle_measure_l4002_400203

theorem pentagon_angle_measure (Q R S T U : ℝ) :
  R = 120 ∧ S = 94 ∧ T = 115 ∧ U = 101 →
  Q + R + S + T + U = 540 →
  Q = 110 := by
sorry

end pentagon_angle_measure_l4002_400203


namespace intersection_of_A_and_B_l4002_400290

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by sorry

end intersection_of_A_and_B_l4002_400290
