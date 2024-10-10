import Mathlib

namespace simplest_quadratic_radical_l3510_351024

def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∀ y : ℝ, y ≠ x → (∃ a b : ℚ, y = a * (b ^ (1/2 : ℝ))) → (∃ c d : ℚ, x = c * (d ^ (1/2 : ℝ))) → False

theorem simplest_quadratic_radical :
  is_simplest_quadratic_radical (26 ^ (1/2 : ℝ)) ∧
  ¬is_simplest_quadratic_radical (8 ^ (1/2 : ℝ)) ∧
  ¬is_simplest_quadratic_radical ((1/3 : ℝ) ^ (1/2 : ℝ)) ∧
  ¬is_simplest_quadratic_radical (2 / (6 ^ (1/2 : ℝ))) :=
by sorry

end simplest_quadratic_radical_l3510_351024


namespace max_black_balls_proof_l3510_351040

/-- The total number of balls -/
def total_balls : ℕ := 8

/-- The number of ways to select 2 red balls and 1 black ball -/
def selection_ways : ℕ := 30

/-- Calculates the number of ways to select 2 red balls and 1 black ball -/
def calc_selection_ways (red : ℕ) : ℕ :=
  Nat.choose red 2 * Nat.choose (total_balls - red) 1

/-- Checks if a given number of red balls satisfies the selection condition -/
def satisfies_condition (red : ℕ) : Prop :=
  calc_selection_ways red = selection_ways

/-- The maximum number of black balls -/
def max_black_balls : ℕ := 3

theorem max_black_balls_proof :
  ∃ (red : ℕ), satisfies_condition red ∧
  ∀ (x : ℕ), satisfies_condition x → (total_balls - x ≤ max_black_balls) :=
sorry

end max_black_balls_proof_l3510_351040


namespace expression_simplification_and_evaluation_l3510_351088

theorem expression_simplification_and_evaluation :
  let a : ℚ := -2
  let b : ℚ := 1/3
  2 * (a^2 * b - 2 * a * b) - 3 * (a^2 * b - 3 * a * b) + a^2 * b = -10/3 := by
  sorry

end expression_simplification_and_evaluation_l3510_351088


namespace valentines_calculation_l3510_351055

/-- The number of Valentines Mrs. Franklin initially had -/
def initial_valentines : ℕ := 58

/-- The number of Valentines Mrs. Franklin gave to her students -/
def given_valentines : ℕ := 42

/-- The number of Valentines Mrs. Franklin has now -/
def remaining_valentines : ℕ := initial_valentines - given_valentines

theorem valentines_calculation : remaining_valentines = 16 := by
  sorry

end valentines_calculation_l3510_351055


namespace bicycle_final_price_l3510_351025

/-- The final selling price of a bicycle given initial cost and profit margins -/
theorem bicycle_final_price (a_cost : ℝ) (a_profit_percent : ℝ) (b_profit_percent : ℝ) : 
  a_cost = 112.5 → 
  a_profit_percent = 60 → 
  b_profit_percent = 25 → 
  a_cost * (1 + a_profit_percent / 100) * (1 + b_profit_percent / 100) = 225 := by
sorry

end bicycle_final_price_l3510_351025


namespace hike_time_calculation_l3510_351017

theorem hike_time_calculation (distance : ℝ) (pace_up : ℝ) (pace_down : ℝ) 
  (h1 : distance = 12)
  (h2 : pace_up = 4)
  (h3 : pace_down = 6) :
  distance / pace_up + distance / pace_down = 5 := by
  sorry

end hike_time_calculation_l3510_351017


namespace min_distance_parabola_to_line_l3510_351062

/-- The minimum distance from a point on the parabola y^2 = 4x to the line 3x + 4y + 15 = 0 -/
theorem min_distance_parabola_to_line :
  let parabola := {P : ℝ × ℝ | P.2^2 = 4 * P.1}
  let line := {P : ℝ × ℝ | 3 * P.1 + 4 * P.2 + 15 = 0}
  (∃ (d : ℝ), d > 0 ∧
    (∀ P ∈ parabola, ∀ Q ∈ line, d ≤ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)) ∧
    (∃ P ∈ parabola, ∃ Q ∈ line, d = Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2))) ∧
  (∀ d' : ℝ, (∀ P ∈ parabola, ∀ Q ∈ line, d' ≤ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)) →
    d' ≥ 29/15) :=
by sorry


end min_distance_parabola_to_line_l3510_351062


namespace difference_median_mode_l3510_351070

def data : List ℕ := [36, 37, 37, 38, 40, 40, 40, 41, 42, 43, 54, 55, 57, 59, 61, 61, 65, 68, 69]

def mode (l : List ℕ) : ℕ := sorry

def median (l : List ℕ) : ℝ := sorry

theorem difference_median_mode : 
  |median data - mode data| = 2 := by sorry

end difference_median_mode_l3510_351070


namespace vegan_soy_free_dishes_l3510_351098

theorem vegan_soy_free_dishes 
  (total_dishes : ℕ) 
  (vegan_ratio : ℚ) 
  (soy_ratio : ℚ) 
  (h1 : vegan_ratio = 1 / 3) 
  (h2 : soy_ratio = 5 / 6) : 
  ↑total_dishes * vegan_ratio * (1 - soy_ratio) = ↑total_dishes * (1 / 18) := by
sorry

end vegan_soy_free_dishes_l3510_351098


namespace farmer_brown_sheep_l3510_351035

/-- The number of chickens Farmer Brown fed -/
def num_chickens : ℕ := 7

/-- The total number of legs among all animals Farmer Brown fed -/
def total_legs : ℕ := 34

/-- The number of legs a chicken has -/
def chicken_legs : ℕ := 2

/-- The number of legs a sheep has -/
def sheep_legs : ℕ := 4

/-- The number of sheep Farmer Brown fed -/
def num_sheep : ℕ := (total_legs - num_chickens * chicken_legs) / sheep_legs

theorem farmer_brown_sheep : num_sheep = 5 := by
  sorry

end farmer_brown_sheep_l3510_351035


namespace shiela_paintings_distribution_l3510_351037

/-- Given Shiela has 18 paintings and each relative gets 9 paintings, 
    prove that she is giving paintings to 2 relatives. -/
theorem shiela_paintings_distribution (total_paintings : ℕ) (paintings_per_relative : ℕ) 
  (h1 : total_paintings = 18) 
  (h2 : paintings_per_relative = 9) : 
  total_paintings / paintings_per_relative = 2 := by
  sorry

end shiela_paintings_distribution_l3510_351037


namespace unique_solution_absolute_value_equation_l3510_351069

theorem unique_solution_absolute_value_equation :
  ∃! y : ℝ, |y - 25| + |y - 15| = |2*y - 40| :=
by
  sorry

end unique_solution_absolute_value_equation_l3510_351069


namespace triangle_area_l3510_351047

/-- The area of a triangle with sides 9, 40, and 41 is 180 square units. -/
theorem triangle_area : ∀ (a b c : ℝ), a = 9 ∧ b = 40 ∧ c = 41 →
  (a * a + b * b = c * c) → (1/2 : ℝ) * a * b = 180 := by
  sorry

end triangle_area_l3510_351047


namespace equation_solutions_l3510_351086

theorem equation_solutions :
  (∃ x : ℝ, 4 * x + x = 19.5 ∧ x = 3.9) ∧
  (∃ x : ℝ, 26.4 - 3 * x = 14.4 ∧ x = 4) ∧
  (∃ x : ℝ, 2 * x - 0.5 * 2 = 0.8 ∧ x = 0.9) := by
  sorry

end equation_solutions_l3510_351086


namespace smallest_positive_integer_satisfying_conditions_l3510_351056

theorem smallest_positive_integer_satisfying_conditions (a : ℕ) :
  (∀ b : ℕ, b > 0 ∧ b % 3 = 1 ∧ 5 ∣ b → a ≤ b) ∧
  a > 0 ∧ a % 3 = 1 ∧ 5 ∣ a →
  a = 10 := by
sorry

end smallest_positive_integer_satisfying_conditions_l3510_351056


namespace greatest_k_value_l3510_351007

theorem greatest_k_value (k : ℝ) : 
  (∃ x y : ℝ, x^2 + k*x + 8 = 0 ∧ y^2 + k*y + 8 = 0 ∧ |x - y| = Real.sqrt 72) →
  k ≤ 2 * Real.sqrt 26 :=
sorry

end greatest_k_value_l3510_351007


namespace grandmother_truth_lies_consistent_solution_l3510_351094

-- Define the type for grandmothers
inductive Grandmother
| Emilia
| Leonie
| Gabrielle

-- Define a function to represent the number of grandchildren for each grandmother
def grandchildren : Grandmother → ℕ
| Grandmother.Emilia => 8
| Grandmother.Leonie => 7
| Grandmother.Gabrielle => 10

-- Define a function to represent the statements made by each grandmother
def statements : Grandmother → List (Grandmother → Bool)
| Grandmother.Emilia => [
    fun g => grandchildren g = 7,
    fun g => grandchildren g = 8,
    fun g => grandchildren Grandmother.Gabrielle = 10
  ]
| Grandmother.Leonie => [
    fun g => grandchildren Grandmother.Emilia = 8,
    fun g => grandchildren g = 6,
    fun g => grandchildren g = 7
  ]
| Grandmother.Gabrielle => [
    fun g => grandchildren Grandmother.Emilia = 7,
    fun g => grandchildren g = 9,
    fun g => grandchildren g = 10
  ]

-- Define a function to count true statements for each grandmother
def countTrueStatements (g : Grandmother) : ℕ :=
  (statements g).filter (fun s => s g) |>.length

-- Theorem stating that each grandmother tells the truth twice and lies once
theorem grandmother_truth_lies :
  ∀ g : Grandmother, countTrueStatements g = 2 :=
sorry

-- Main theorem proving the consistency of the solution
theorem consistent_solution :
  (grandchildren Grandmother.Emilia = 8) ∧
  (grandchildren Grandmother.Leonie = 7) ∧
  (grandchildren Grandmother.Gabrielle = 10) :=
sorry

end grandmother_truth_lies_consistent_solution_l3510_351094


namespace inequality_and_optimality_l3510_351012

theorem inequality_and_optimality :
  (∀ (x y : ℝ), x > 0 → y > 0 → (x + y)^5 ≥ 12 * x * y * (x^3 + y^3)) ∧
  (∀ (K : ℝ), K > 12 → ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ (x + y)^5 < K * x * y * (x^3 + y^3)) :=
by sorry

end inequality_and_optimality_l3510_351012


namespace initial_number_of_persons_l3510_351067

theorem initial_number_of_persons (avg_weight_increase : ℝ) (old_weight : ℝ) (new_weight : ℝ) :
  avg_weight_increase = 1.5 →
  old_weight = 65 →
  new_weight = 78.5 →
  (new_weight - old_weight) / avg_weight_increase = 9 :=
by
  sorry

end initial_number_of_persons_l3510_351067


namespace equation_solver_l3510_351079

theorem equation_solver (m n : ℕ) : 
  ((1^(m+1))/(5^(m+1))) * ((1^n)/(4^n)) = 1/(2*(10^35)) ∧ m = 34 → n = 18 := by
  sorry

end equation_solver_l3510_351079


namespace typing_difference_is_1200_l3510_351042

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Micah's typing speed in words per minute -/
def micah_speed : ℕ := 20

/-- Isaiah's typing speed in words per minute -/
def isaiah_speed : ℕ := 40

/-- The difference in words typed per hour between Isaiah and Micah -/
def typing_difference : ℕ := isaiah_speed * minutes_per_hour - micah_speed * minutes_per_hour

theorem typing_difference_is_1200 : typing_difference = 1200 := by
  sorry

end typing_difference_is_1200_l3510_351042


namespace f_derivative_l3510_351033

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x + 3

-- State the theorem
theorem f_derivative : 
  ∀ x : ℝ, deriv f x = 2 := by sorry

end f_derivative_l3510_351033


namespace supply_duration_with_three_leaks_l3510_351089

/-- Represents a water tank with its supply duration and leak information -/
structure WaterTank where
  normalDuration : ℕ  -- Duration in days without leaks
  singleLeakDuration : ℕ  -- Duration in days with a single leak
  singleLeakRate : ℕ  -- Rate of the single leak in liters per day
  leakRates : List ℕ  -- List of leak rates for multiple leaks

/-- Calculates the duration of water supply given multiple leaks -/
def supplyDurationWithLeaks (tank : WaterTank) : ℕ :=
  -- Implementation details omitted
  sorry

/-- Theorem stating the correct supply duration for the given scenario -/
theorem supply_duration_with_three_leaks 
  (tank : WaterTank) 
  (h1 : tank.normalDuration = 60)
  (h2 : tank.singleLeakDuration = 45)
  (h3 : tank.singleLeakRate = 10)
  (h4 : tank.leakRates = [10, 15, 20]) :
  supplyDurationWithLeaks tank = 24 := by
  sorry

end supply_duration_with_three_leaks_l3510_351089


namespace bernoulli_prob_zero_success_l3510_351003

/-- The number of Bernoulli trials -/
def n : ℕ := 7

/-- The probability of success in each trial -/
def p : ℚ := 2/7

/-- The probability of failure in each trial -/
def q : ℚ := 1 - p

/-- The number of successes we're interested in -/
def k : ℕ := 0

/-- Theorem: The probability of 0 successes in 7 Bernoulli trials 
    with success probability 2/7 is (5/7)^7 -/
theorem bernoulli_prob_zero_success : 
  (n.choose k) * p^k * q^(n-k) = (5/7)^7 := by
  sorry

end bernoulli_prob_zero_success_l3510_351003


namespace car_value_after_depreciation_l3510_351061

def initial_value : ℝ := 10000

def depreciation_rates : List ℝ := [0.20, 0.15, 0.10, 0.08, 0.05]

def calculate_value (initial : ℝ) (rates : List ℝ) : ℝ :=
  rates.foldl (fun acc rate => acc * (1 - rate)) initial

theorem car_value_after_depreciation :
  calculate_value initial_value depreciation_rates = 5348.88 := by
  sorry

end car_value_after_depreciation_l3510_351061


namespace ball_probability_l3510_351016

theorem ball_probability (m n : ℕ) : 
  (∃ (total : ℕ), total = m + 8 + n ∧ 
   (8 : ℚ) / total = (m + n : ℚ) / total) → 
  m + n = 8 := by
  sorry

end ball_probability_l3510_351016


namespace reciprocal_of_2022_l3510_351011

theorem reciprocal_of_2022 : (2022⁻¹ : ℝ) = 1 / 2022 := by sorry

end reciprocal_of_2022_l3510_351011


namespace p_sufficient_not_necessary_for_q_l3510_351054

open Real

-- Define the property p
def property_p (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + π) = -f x

-- Define the property q
def property_q (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2*π) = f x

-- Theorem: p is sufficient but not necessary for q
theorem p_sufficient_not_necessary_for_q :
  (∀ f : ℝ → ℝ, property_p f → property_q f) ∧
  (∃ f : ℝ → ℝ, property_q f ∧ ¬property_p f) :=
sorry

end p_sufficient_not_necessary_for_q_l3510_351054


namespace system_solution_unique_l3510_351071

theorem system_solution_unique : 
  ∃! (x y : ℝ), (2 * x - y = 5) ∧ (3 * x + 2 * y = -3) :=
by
  sorry

end system_solution_unique_l3510_351071


namespace intersection_complement_equality_l3510_351076

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3, 4}

theorem intersection_complement_equality : M ∩ (U \ N) = {1} := by
  sorry

end intersection_complement_equality_l3510_351076


namespace sixth_term_value_l3510_351005

def sequence_rule (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → a n = (a (n-1) + a (n+1)) / 3

theorem sixth_term_value (a : ℕ → ℕ) :
  sequence_rule a →
  a 2 = 7 →
  a 3 = 20 →
  a 6 = 364 := by
  sorry

end sixth_term_value_l3510_351005


namespace prime_between_squares_l3510_351023

theorem prime_between_squares : ∃! p : ℕ, 
  Nat.Prime p ∧ 
  ∃ a : ℕ, p = a^2 + 5 ∧ p = (a+1)^2 - 8 :=
by
  sorry

end prime_between_squares_l3510_351023


namespace length_BI_approx_l3510_351031

/-- Triangle ABC with given side lengths --/
structure Triangle where
  ab : ℝ
  ac : ℝ
  bc : ℝ

/-- The incenter of a triangle --/
def Incenter (t : Triangle) : Point := sorry

/-- The distance between two points --/
def distance (p q : Point) : ℝ := sorry

/-- The given triangle --/
def triangle_ABC : Triangle := { ab := 31, ac := 29, bc := 30 }

/-- Theorem: The length of BI in the given triangle is approximately 17.22 --/
theorem length_BI_approx (ε : ℝ) (h : ε > 0) : 
  ∃ (B I : Point), I = Incenter triangle_ABC ∧ 
    |distance B I - 17.22| < ε := by sorry

end length_BI_approx_l3510_351031


namespace ratio_equality_l3510_351039

theorem ratio_equality (x y z m n k a b c : ℝ) 
  (h : x / (m * (n * b + k * c - m * a)) = 
       y / (n * (k * c + m * a - n * b)) ∧
       y / (n * (k * c + m * a - n * b)) = 
       z / (k * (m * a + n * b - k * c))) :
  m / (x * (b * y + c * z - a * x)) = 
  n / (y * (c * z + a * x - b * y)) ∧
  n / (y * (c * z + a * x - b * y)) = 
  k / (z * (a * x + b * y - c * z)) :=
by sorry

end ratio_equality_l3510_351039


namespace square_difference_l3510_351002

theorem square_difference : (50 : ℕ)^2 - (49 : ℕ)^2 = 99 := by
  sorry

end square_difference_l3510_351002


namespace line_x_intercept_l3510_351072

/-- Given a line with slope 3/4 passing through (-12, -39), prove its x-intercept is 40 -/
theorem line_x_intercept :
  let m : ℚ := 3/4  -- slope
  let x₀ : ℤ := -12
  let y₀ : ℤ := -39
  let b : ℚ := y₀ - m * x₀  -- y-intercept
  let x_intercept : ℚ := -b / m  -- x-coordinate where y = 0
  x_intercept = 40 := by
sorry

end line_x_intercept_l3510_351072


namespace cricket_bat_selling_price_l3510_351073

theorem cricket_bat_selling_price 
  (profit : ℝ) 
  (profit_percentage : ℝ) 
  (selling_price : ℝ) : 
  profit = 300 → 
  profit_percentage = 50 → 
  selling_price = profit * (100 / profit_percentage) + profit → 
  selling_price = 900 := by
sorry

end cricket_bat_selling_price_l3510_351073


namespace tangent_line_inverse_l3510_351095

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the inverse function of f
variable (f_inv : ℝ → ℝ)

-- State that f_inv is the inverse of f
variable (h_inverse : Function.RightInverse f_inv f ∧ Function.LeftInverse f_inv f)

-- Define a point x
variable (x : ℝ)

-- Define the tangent line to f at (x, f(x))
def tangent_line_f (t : ℝ) : ℝ := 2 * t - 3

-- State the theorem
theorem tangent_line_inverse (h_tangent : ∀ t, f x = tangent_line_f t - t) :
  ∀ t, x = t - 2 * (f_inv t) - 3 := by sorry

end tangent_line_inverse_l3510_351095


namespace total_molecular_weight_l3510_351077

-- Define molecular weights of elements
def mw_C : ℝ := 12.01
def mw_H : ℝ := 1.008
def mw_O : ℝ := 16.00
def mw_Na : ℝ := 22.99

-- Define composition of compounds
def acetic_acid_C : ℕ := 2
def acetic_acid_H : ℕ := 4
def acetic_acid_O : ℕ := 2

def sodium_hydroxide_Na : ℕ := 1
def sodium_hydroxide_O : ℕ := 1
def sodium_hydroxide_H : ℕ := 1

-- Define number of moles
def moles_acetic_acid : ℝ := 7
def moles_sodium_hydroxide : ℝ := 10

-- Theorem statement
theorem total_molecular_weight :
  let mw_acetic_acid := acetic_acid_C * mw_C + acetic_acid_H * mw_H + acetic_acid_O * mw_O
  let mw_sodium_hydroxide := sodium_hydroxide_Na * mw_Na + sodium_hydroxide_O * mw_O + sodium_hydroxide_H * mw_H
  let total_weight := moles_acetic_acid * mw_acetic_acid + moles_sodium_hydroxide * mw_sodium_hydroxide
  total_weight = 820.344 := by
  sorry

end total_molecular_weight_l3510_351077


namespace exactly_one_hit_probability_l3510_351057

theorem exactly_one_hit_probability (p : ℝ) (h : p = 0.6) :
  p * (1 - p) + (1 - p) * p = 0.48 := by
  sorry

end exactly_one_hit_probability_l3510_351057


namespace contrapositive_divisibility_l3510_351032

theorem contrapositive_divisibility (n : ℤ) : 
  (∀ m : ℤ, m % 6 = 0 → m % 2 = 0) ↔ 
  (∀ k : ℤ, k % 2 ≠ 0 → k % 6 ≠ 0) :=
by sorry

end contrapositive_divisibility_l3510_351032


namespace min_buses_for_535_students_l3510_351059

/-- The minimum number of buses needed to transport a given number of students -/
def min_buses (capacity : ℕ) (students : ℕ) : ℕ :=
  (students + capacity - 1) / capacity

/-- Theorem: Given a bus capacity of 45 students and 535 students to transport,
    the minimum number of buses needed is 12 -/
theorem min_buses_for_535_students :
  min_buses 45 535 = 12 := by
  sorry

end min_buses_for_535_students_l3510_351059


namespace roots_differ_by_one_l3510_351058

theorem roots_differ_by_one (a : ℝ) : 
  (∃ x y : ℝ, x^2 - a*x + 1 = 0 ∧ y^2 - a*y + 1 = 0 ∧ y - x = 1) → a = Real.sqrt 5 := by
  sorry

end roots_differ_by_one_l3510_351058


namespace problem_statement_l3510_351091

-- Define proposition p
def p : Prop := ∃ x : ℝ, Real.exp x = 0.1

-- Define the perpendicularity condition for two lines
def perpendicular (a : ℝ) : Prop :=
  ∀ x y : ℝ, (x - a * y = 0) → (2 * x + a * y - 1 = 0) → 
  (1 / a) * (-2 / a) = -1

-- Define proposition q
def q : Prop := ∀ a : ℝ, perpendicular a ↔ a = Real.sqrt 2

-- The theorem to be proved
theorem problem_statement : p ∧ ¬q := by
  sorry

end problem_statement_l3510_351091


namespace take_home_pay_l3510_351000

def annual_salary : ℝ := 40000
def tax_rate : ℝ := 0.20
def healthcare_rate : ℝ := 0.10
def union_dues : ℝ := 800

theorem take_home_pay :
  annual_salary * (1 - tax_rate - healthcare_rate) - union_dues = 27200 := by
  sorry

end take_home_pay_l3510_351000


namespace shirt_markup_price_l3510_351008

/-- Given a wholesale price of a shirt, prove that the initial price after 80% markup is $27 -/
theorem shirt_markup_price (P : ℝ) 
  (h1 : 1.80 * P = 1.80 * P) -- Initial price after 80% markup
  (h2 : 2.00 * P = 2.00 * P) -- Price for 100% markup
  (h3 : 2.00 * P - 1.80 * P = 3) -- Difference between 100% and 80% markup is $3
  : 1.80 * P = 27 := by
  sorry

end shirt_markup_price_l3510_351008


namespace harvest_rent_proof_l3510_351029

/-- The total rent paid during the harvest season. -/
def total_rent (weekly_rent : ℕ) (weeks : ℕ) : ℕ :=
  weekly_rent * weeks

/-- Proof that the total rent paid during the harvest season is $527,292. -/
theorem harvest_rent_proof :
  total_rent 388 1359 = 527292 := by
  sorry

end harvest_rent_proof_l3510_351029


namespace angle_between_vectors_l3510_351045

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define the vectors a and b
variable (a b : V)

-- State the theorem
theorem angle_between_vectors
  (h1 : ‖a‖ = 2)
  (h2 : ‖b‖ = (1/3) * ‖a‖)
  (h3 : ‖a - (1/2) • b‖ = Real.sqrt 43 / 3) :
  Real.arccos (inner a b / (‖a‖ * ‖b‖)) = (2 * Real.pi) / 3 := by
sorry

end angle_between_vectors_l3510_351045


namespace debate_team_group_size_l3510_351014

/-- Proves that the number of students in each group is 9,
    given the number of boys, girls, and total groups. -/
theorem debate_team_group_size
  (boys : ℕ)
  (girls : ℕ)
  (total_groups : ℕ)
  (h1 : boys = 26)
  (h2 : girls = 46)
  (h3 : total_groups = 8) :
  (boys + girls) / total_groups = 9 :=
by sorry

end debate_team_group_size_l3510_351014


namespace ball_radius_for_given_hole_l3510_351063

/-- The radius of a spherical ball that leaves a circular hole with given dimensions -/
def ball_radius (hole_diameter : ℝ) (hole_depth : ℝ) : ℝ :=
  13

/-- Theorem stating that a ball leaving a hole with diameter 24 cm and depth 8 cm has a radius of 13 cm -/
theorem ball_radius_for_given_hole : ball_radius 24 8 = 13 := by
  sorry

end ball_radius_for_given_hole_l3510_351063


namespace valid_m_set_l3510_351004

def is_valid_m (m : ℕ) : Prop :=
  ∃ n : ℕ, n > 0 ∧ 
    ∃ k : ℕ, m * n = k * k ∧
    ∃ p : ℕ, Nat.Prime p ∧ m - n = p

theorem valid_m_set :
  {m : ℕ | 1000 ≤ m ∧ m ≤ 2021 ∧ is_valid_m m} =
  {1156, 1296, 1369, 1600, 1764} :=
by sorry

end valid_m_set_l3510_351004


namespace sams_adventure_books_l3510_351083

/-- The number of adventure books Sam bought at the school's book fair -/
def adventure_books : ℕ := sorry

/-- The number of mystery books Sam bought -/
def mystery_books : ℕ := 17

/-- The total number of books Sam bought -/
def total_books : ℕ := 30

theorem sams_adventure_books :
  adventure_books = total_books - mystery_books ∧ adventure_books = 13 := by sorry

end sams_adventure_books_l3510_351083


namespace current_gas_in_car_l3510_351046

/-- Represents the fuel efficiency of a car in miles per gallon -/
def fuel_efficiency : ℝ := 20

/-- Represents the total distance to be traveled in miles -/
def total_distance : ℝ := 1200

/-- Represents the additional gallons of gas needed for the trip -/
def additional_gas_needed : ℝ := 52

/-- Theorem stating the current amount of gas in the car -/
theorem current_gas_in_car : 
  (total_distance / fuel_efficiency) - additional_gas_needed = 8 := by
  sorry

end current_gas_in_car_l3510_351046


namespace arithmetic_sequence_solution_l3510_351053

theorem arithmetic_sequence_solution :
  let a₁ : ℚ := 3/4
  let a₂ : ℚ → ℚ := λ x => x - 2
  let a₃ : ℚ → ℚ := λ x => 5*x
  ∀ x : ℚ, (a₂ x - a₁ = a₃ x - a₂ x) → x = -19/12 := by
  sorry

end arithmetic_sequence_solution_l3510_351053


namespace max_points_2079_l3510_351075

def points (x : ℕ) : ℕ :=
  (if x % 3 = 0 then 3 else 0) +
  (if x % 5 = 0 then 5 else 0) +
  (if x % 7 = 0 then 7 else 0) +
  (if x % 9 = 0 then 9 else 0) +
  (if x % 11 = 0 then 11 else 0)

theorem max_points_2079 :
  ∀ x : ℕ, 2017 ≤ x → x ≤ 2117 → points x ≤ points 2079 :=
by
  sorry

end max_points_2079_l3510_351075


namespace inequality_proof_l3510_351009

theorem inequality_proof (a b c : ℝ) 
  (ha : a = 1 / 10)
  (hb : b = Real.sin 1 / (9 + Real.cos 1))
  (hc : c = (Real.exp (1 / 10)) - 1) :
  b < a ∧ a < c :=
sorry

end inequality_proof_l3510_351009


namespace smallest_class_size_l3510_351044

theorem smallest_class_size (n : ℕ) (scores : Fin n → ℕ) : 
  (∀ i, scores i ≥ 50) →   -- Each student scored at least 50
  (∃ s : Finset (Fin n), s.card = 4 ∧ ∀ i ∈ s, scores i = 80) →   -- Four students achieved the maximum score
  (Finset.sum Finset.univ scores) / n = 65 →   -- The average score was 65
  n ≥ 8 ∧ 
  (∃ scores_8 : Fin 8 → ℕ, 
    (∀ i, scores_8 i ≥ 50) ∧ 
    (∃ s : Finset (Fin 8), s.card = 4 ∧ ∀ i ∈ s, scores_8 i = 80) ∧ 
    (Finset.sum Finset.univ scores_8) / 8 = 65) :=
by sorry

end smallest_class_size_l3510_351044


namespace max_cookies_without_ingredients_l3510_351030

theorem max_cookies_without_ingredients (total_cookies : ℕ) 
  (peanut_cookies : ℕ) (choc_cookies : ℕ) (almond_cookies : ℕ) (raisin_cookies : ℕ) : 
  total_cookies = 60 →
  peanut_cookies ≥ 20 →
  choc_cookies ≥ 15 →
  almond_cookies ≥ 12 →
  raisin_cookies ≥ 7 →
  ∃ (plain_cookies : ℕ), plain_cookies ≤ 6 ∧ 
    plain_cookies + peanut_cookies + choc_cookies + almond_cookies + raisin_cookies ≥ total_cookies := by
  sorry

end max_cookies_without_ingredients_l3510_351030


namespace parabola_vertex_not_in_second_quadrant_l3510_351090

/-- The vertex of the parabola y = 4x^2 - 4(a+1)x + a cannot lie in the second quadrant for any real value of a. -/
theorem parabola_vertex_not_in_second_quadrant (a : ℝ) : 
  let f (x : ℝ) := 4 * x^2 - 4 * (a + 1) * x + a
  let vertex_x := (a + 1) / 2
  let vertex_y := f vertex_x
  ¬(vertex_x < 0 ∧ vertex_y > 0) := by
sorry

end parabola_vertex_not_in_second_quadrant_l3510_351090


namespace pizza_consumption_order_l3510_351093

-- Define the siblings
inductive Sibling : Type
| Alex : Sibling
| Beth : Sibling
| Cyril : Sibling
| Daria : Sibling
| Ed : Sibling

-- Define a function to represent the fraction of pizza eaten by each sibling
def pizza_fraction (s : Sibling) : ℚ :=
  match s with
  | Sibling.Alex => 1/6
  | Sibling.Beth => 1/4
  | Sibling.Cyril => 1/3
  | Sibling.Daria => 1/8
  | Sibling.Ed => 1 - (1/6 + 1/4 + 1/3 + 1/8)

-- Define the theorem
theorem pizza_consumption_order :
  ∃ (l : List Sibling),
    l = [Sibling.Cyril, Sibling.Beth, Sibling.Alex, Sibling.Daria, Sibling.Ed] ∧
    ∀ (i j : Nat), i < j → j < l.length →
      pizza_fraction (l.get ⟨i, by sorry⟩) ≥ pizza_fraction (l.get ⟨j, by sorry⟩) :=
by sorry

end pizza_consumption_order_l3510_351093


namespace used_car_seller_problem_l3510_351052

theorem used_car_seller_problem (num_clients : ℕ) (cars_per_client : ℕ) (selections_per_car : ℕ) : 
  num_clients = 18 → cars_per_client = 3 → selections_per_car = 3 →
  num_clients * cars_per_client / selections_per_car = 18 := by
sorry

end used_car_seller_problem_l3510_351052


namespace walking_distance_l3510_351085

theorem walking_distance (x y : ℝ) 
  (h1 : x / 4 + y / 3 + y / 6 + x / 4 = 5) : x + y = 10 ∧ 2 * (x + y) = 20 := by
  sorry

end walking_distance_l3510_351085


namespace sum_of_squares_of_roots_l3510_351097

/-- Given a cubic equation x√x - 9x + 8√x - 2 = 0 with all roots real and positive,
    the sum of the squares of its roots is 65. -/
theorem sum_of_squares_of_roots : ∃ (r s t : ℝ), 
  (∀ x : ℝ, x > 0 → (x * Real.sqrt x - 9*x + 8*Real.sqrt x - 2 = 0) ↔ (x = r ∨ x = s ∨ x = t)) →
  r > 0 ∧ s > 0 ∧ t > 0 →
  r^2 + s^2 + t^2 = 65 := by
sorry

end sum_of_squares_of_roots_l3510_351097


namespace parallel_vectors_m_value_l3510_351010

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ → ℝ × ℝ := λ m ↦ (2 + m, 3 - m)
  let c : ℝ → ℝ × ℝ := λ m ↦ (3 * m, 1)
  ∀ m : ℝ, (∃ k : ℝ, a = k • (c m - b m)) → m = 2/3 := by
  sorry

end parallel_vectors_m_value_l3510_351010


namespace four_digit_square_palindromes_l3510_351087

theorem four_digit_square_palindromes :
  (∃! (s : Finset Nat), 
    ∀ n, n ∈ s ↔ 
      32 ≤ n ∧ n ≤ 99 ∧ 
      1000 ≤ n^2 ∧ n^2 ≤ 9999 ∧ 
      (∃ a b : Nat, n^2 = a * 1000 + b * 100 + b * 10 + a)) ∧ 
  (∃ s : Finset Nat, 
    (∀ n, n ∈ s ↔ 
      32 ≤ n ∧ n ≤ 99 ∧ 
      1000 ≤ n^2 ∧ n^2 ≤ 9999 ∧ 
      (∃ a b : Nat, n^2 = a * 1000 + b * 100 + b * 10 + a)) ∧ 
    s.card = 2) :=
by sorry

end four_digit_square_palindromes_l3510_351087


namespace parallel_lines_distance_l3510_351074

/-- Two parallel lines in the plane -/
structure ParallelLines :=
  (a : ℝ)
  (l₁ : ℝ → ℝ → Prop)
  (l₂ : ℝ → ℝ → Prop)
  (h_l₁ : ∀ x y, l₁ x y ↔ x + (a - 1) * y + 2 = 0)
  (h_l₂ : ∀ x y, l₂ x y ↔ a * x + 2 * y + 1 = 0)
  (h_parallel : ∀ x₁ y₁ x₂ y₂, l₁ x₁ y₁ → l₂ x₂ y₂ → (x₁ - x₂) * 2 = (y₁ - y₂) * (1 - a))

/-- Distance between two lines -/
def distance (l₁ l₂ : ℝ → ℝ → Prop) : ℝ := sorry

/-- Theorem: If the distance between two parallel lines is 3√5/5, then a = -1 -/
theorem parallel_lines_distance (lines : ParallelLines) :
  distance lines.l₁ lines.l₂ = 3 * Real.sqrt 5 / 5 → lines.a = -1 := by sorry

end parallel_lines_distance_l3510_351074


namespace parallel_lines_a_value_l3510_351006

/-- Two lines in the plane, parameterized by a real number a -/
structure Lines (a : ℝ) where
  l₁ : ℝ → ℝ → Prop := λ x y => x + 2*a*y - 1 = 0
  l₂ : ℝ → ℝ → Prop := λ x y => (a + 1)*x - a*y = 0

/-- The condition for two lines to be parallel -/
def parallel (a : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ (1 : ℝ) / (2*a) = k * ((a + 1) / (-a))

theorem parallel_lines_a_value (a : ℝ) :
  parallel a → a = -3/2 ∨ a = 0 := by
  sorry

end parallel_lines_a_value_l3510_351006


namespace y_minus_x_value_l3510_351051

theorem y_minus_x_value (x y : ℚ) (h1 : x + y = 8) (h2 : y - 3 * x = 7) : 
  y - x = (15 : ℚ) / 2 := by sorry

end y_minus_x_value_l3510_351051


namespace solution_set_characterization_l3510_351028

-- Define the properties of the function f
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def increasing_on_positive (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → f x < f y

-- State the theorem
theorem solution_set_characterization
  (f : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_incr : increasing_on_positive f)
  (h_zero : f (-3) = 0) :
  {x : ℝ | x * f x < 0} = Set.Ioo (-3) 0 :=
sorry

end solution_set_characterization_l3510_351028


namespace total_cost_calculation_l3510_351018

/-- The cost of mangos per kg -/
def mango_cost : ℝ := sorry

/-- The cost of rice per kg -/
def rice_cost : ℝ := sorry

/-- The cost of flour per kg -/
def flour_cost : ℝ := 22

theorem total_cost_calculation :
  (10 * mango_cost = 24 * rice_cost) →
  (6 * flour_cost = 2 * rice_cost) →
  (4 * mango_cost + 3 * rice_cost + 5 * flour_cost = 941.6) :=
by sorry

end total_cost_calculation_l3510_351018


namespace plane_equation_proof_l3510_351001

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space defined by parametric equations -/
structure Line3D where
  t : ℝ → Point3D

/-- A plane in 3D space defined by the equation Ax + By + Cz + D = 0 -/
structure Plane where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- Check if a point lies on a plane -/
def pointOnPlane (p : Point3D) (plane : Plane) : Prop :=
  plane.A * p.x + plane.B * p.y + plane.C * p.z + plane.D = 0

/-- Check if a line is contained in a plane -/
def lineInPlane (l : Line3D) (plane : Plane) : Prop :=
  ∀ t, pointOnPlane (l.t t) plane

/-- The specific point given in the problem -/
def givenPoint : Point3D :=
  { x := 1, y := -3, z := 6 }

/-- The specific line given in the problem -/
def givenLine : Line3D :=
  { t := λ t => { x := 4*t + 2, y := -t - 1, z := 2*t + 3 } }

/-- The plane we need to prove -/
def resultPlane : Plane :=
  { A := 1, B := -18, C := -7, D := -13 }

theorem plane_equation_proof :
  (pointOnPlane givenPoint resultPlane) ∧
  (lineInPlane givenLine resultPlane) ∧
  (resultPlane.A > 0) ∧
  (Nat.gcd (Nat.gcd (Int.natAbs resultPlane.A) (Int.natAbs resultPlane.B))
           (Nat.gcd (Int.natAbs resultPlane.C) (Int.natAbs resultPlane.D)) = 1) :=
by sorry

end plane_equation_proof_l3510_351001


namespace perfect_apples_count_l3510_351034

/-- Represents the number of perfect apples in a batch with given conditions -/
def number_of_perfect_apples (total_apples : ℕ) 
  (small_ratio medium_ratio large_ratio : ℚ)
  (unripe_ratio partly_ripe_ratio fully_ripe_ratio : ℚ) : ℕ :=
  22

/-- Theorem stating the number of perfect apples under given conditions -/
theorem perfect_apples_count : 
  number_of_perfect_apples 60 (1/4) (1/2) (1/4) (1/3) (1/6) (1/2) = 22 := by
  sorry

end perfect_apples_count_l3510_351034


namespace debelyn_gave_two_dolls_l3510_351048

/-- Represents the number of dolls each person has --/
structure DollCount where
  debelyn_initial : ℕ
  christel_initial : ℕ
  christel_to_andrena : ℕ
  debelyn_to_andrena : ℕ

/-- The conditions of the problem --/
def problem_conditions (d : DollCount) : Prop :=
  d.debelyn_initial = 20 ∧
  d.christel_initial = 24 ∧
  d.christel_to_andrena = 5 ∧
  d.debelyn_initial - d.debelyn_to_andrena + 3 = d.christel_initial - d.christel_to_andrena + 2

theorem debelyn_gave_two_dolls (d : DollCount) 
  (h : problem_conditions d) : d.debelyn_to_andrena = 2 := by
  sorry

#check debelyn_gave_two_dolls

end debelyn_gave_two_dolls_l3510_351048


namespace fib_75_mod_9_l3510_351043

def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

theorem fib_75_mod_9 : fib 74 % 9 = 2 := by
  sorry

end fib_75_mod_9_l3510_351043


namespace sum_of_207_instances_of_33_difference_25_instances_of_112_from_3000_difference_product_and_sum_of_12_and_13_l3510_351026

-- Question 1
theorem sum_of_207_instances_of_33 : (Finset.range 207).sum (λ _ => 33) = 6831 := by sorry

-- Question 2
theorem difference_25_instances_of_112_from_3000 : 3000 - 25 * 112 = 200 := by sorry

-- Question 3
theorem difference_product_and_sum_of_12_and_13 : 12 * 13 - (12 + 13) = 131 := by sorry

end sum_of_207_instances_of_33_difference_25_instances_of_112_from_3000_difference_product_and_sum_of_12_and_13_l3510_351026


namespace total_owed_is_790_l3510_351065

/-- Calculates the total amount owed for three overdue bills -/
def total_amount_owed (bill1_principal : ℝ) (bill1_interest_rate : ℝ) (bill1_months : ℕ)
                      (bill2_principal : ℝ) (bill2_late_fee : ℝ) (bill2_months : ℕ)
                      (bill3_first_month_fee : ℝ) (bill3_months : ℕ) : ℝ :=
  let bill1_total := bill1_principal * (1 + bill1_interest_rate * bill1_months)
  let bill2_total := bill2_principal + bill2_late_fee * bill2_months
  let bill3_total := bill3_first_month_fee * (1 + (bill3_months - 1) * 2)
  bill1_total + bill2_total + bill3_total

/-- Theorem stating the total amount owed is $790 given the specific bill conditions -/
theorem total_owed_is_790 :
  total_amount_owed 200 0.1 2 130 50 6 40 2 = 790 := by
  sorry

end total_owed_is_790_l3510_351065


namespace intersection_when_a_is_2_subset_range_l3510_351041

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x^2 - 3*(a+1)*x + 2*(3*a+1) < 0}
def B (a : ℝ) : Set ℝ := {x | (x-2*a) / (x-(a^2+1)) < 0}

-- Theorem for part (1)
theorem intersection_when_a_is_2 : A 2 ∩ B 2 = Set.Ioo 4 5 := by sorry

-- Theorem for part (2)
theorem subset_range : {a : ℝ | B a ⊆ A a} = Set.Icc (-1) 3 := by sorry

end intersection_when_a_is_2_subset_range_l3510_351041


namespace equation_solution_l3510_351019

theorem equation_solution : ∃ (z : ℂ), 
  (z - 4)^6 + (z - 6)^6 = 32 ∧ 
  (z = 5 + Complex.I * Real.sqrt 3 ∨ z = 5 - Complex.I * Real.sqrt 3) := by
  sorry

end equation_solution_l3510_351019


namespace zero_overtime_accidents_l3510_351080

/-- Represents the linear relationship between overtime hours and accidents -/
structure AccidentModel where
  slope : ℝ
  intercept : ℝ

/-- Calculates the expected number of accidents for a given number of overtime hours -/
def expected_accidents (model : AccidentModel) (hours : ℝ) : ℝ :=
  model.slope * hours + model.intercept

/-- Theorem stating the expected number of accidents when no overtime is logged -/
theorem zero_overtime_accidents 
  (model : AccidentModel)
  (h1 : expected_accidents model 1000 = 8)
  (h2 : expected_accidents model 400 = 5) :
  expected_accidents model 0 = 3 := by
  sorry

end zero_overtime_accidents_l3510_351080


namespace cos_75_cos_15_minus_sin_75_sin_195_l3510_351020

theorem cos_75_cos_15_minus_sin_75_sin_195 : 
  Real.cos (75 * π / 180) * Real.cos (15 * π / 180) - 
  Real.sin (75 * π / 180) * Real.sin (195 * π / 180) = 1/2 := by
  sorry

end cos_75_cos_15_minus_sin_75_sin_195_l3510_351020


namespace train_length_l3510_351050

/-- Given a train that crosses a platform in a certain time and a signal pole in another time,
    this theorem calculates the length of the train. -/
theorem train_length
  (platform_length : ℝ)
  (platform_time : ℝ)
  (pole_time : ℝ)
  (h1 : platform_length = 400)
  (h2 : platform_time = 42)
  (h3 : pole_time = 18) :
  ∃ (train_length : ℝ),
    train_length = 300 ∧
    train_length * (1 / pole_time) * platform_time = train_length + platform_length :=
by sorry

end train_length_l3510_351050


namespace charcoal_drawings_count_l3510_351099

theorem charcoal_drawings_count (total : ℕ) (colored_pencil : ℕ) (blending_marker : ℕ) 
  (h1 : total = 25)
  (h2 : colored_pencil = 14)
  (h3 : blending_marker = 7) :
  total - (colored_pencil + blending_marker) = 4 := by
  sorry

end charcoal_drawings_count_l3510_351099


namespace fifteen_percent_of_thousand_is_150_l3510_351082

theorem fifteen_percent_of_thousand_is_150 :
  (15 / 100) * 1000 = 150 := by
  sorry

end fifteen_percent_of_thousand_is_150_l3510_351082


namespace optimal_loquat_variety_l3510_351084

/-- Represents a variety of loquat trees -/
structure LoquatVariety where
  name : String
  average_yield : ℝ
  variance : ℝ

/-- Determines if one variety is better than another based on yield and stability -/
def is_better (v1 v2 : LoquatVariety) : Prop :=
  (v1.average_yield > v2.average_yield) ∨ 
  (v1.average_yield = v2.average_yield ∧ v1.variance < v2.variance)

/-- Determines if a variety is the best among a list of varieties -/
def is_best (v : LoquatVariety) (vs : List LoquatVariety) : Prop :=
  ∀ v' ∈ vs, v ≠ v' → is_better v v'

theorem optimal_loquat_variety (A B C : LoquatVariety)
  (hA : A = { name := "A", average_yield := 42, variance := 1.8 })
  (hB : B = { name := "B", average_yield := 45, variance := 23 })
  (hC : C = { name := "C", average_yield := 45, variance := 1.8 }) :
  is_best C [A, B, C] := by
  sorry

#check optimal_loquat_variety

end optimal_loquat_variety_l3510_351084


namespace gcd_228_1995_l3510_351038

theorem gcd_228_1995 : Nat.gcd 228 1995 = 21 := by
  sorry

end gcd_228_1995_l3510_351038


namespace grant_baseball_gear_sale_total_l3510_351096

def baseball_cards_price : ℝ := 25
def baseball_bat_price : ℝ := 10
def baseball_glove_original_price : ℝ := 30
def baseball_glove_discount : ℝ := 0.20
def baseball_cleats_original_price : ℝ := 10
def usd_to_eur_rate : ℝ := 0.85
def baseball_cleats_discount : ℝ := 0.15

theorem grant_baseball_gear_sale_total :
  let baseball_glove_sale_price := baseball_glove_original_price * (1 - baseball_glove_discount)
  let cleats_eur_price := baseball_cleats_original_price * usd_to_eur_rate
  let cleats_discounted_price := baseball_cleats_original_price * (1 - baseball_cleats_discount)
  baseball_cards_price + baseball_bat_price + baseball_glove_sale_price + cleats_eur_price + cleats_discounted_price = 76 := by
  sorry

end grant_baseball_gear_sale_total_l3510_351096


namespace leo_weight_l3510_351068

/-- Given the weights of Leo (L), Kendra (K), and Ethan (E) satisfying the following conditions:
    1. L + K + E = 210
    2. L + 10 = 1.5K
    3. L + 10 = 0.75E
    We prove that Leo's weight (L) is approximately 63.33 pounds. -/
theorem leo_weight (L K E : ℝ) 
    (h1 : L + K + E = 210)
    (h2 : L + 10 = 1.5 * K)
    (h3 : L + 10 = 0.75 * E) : 
    ∃ ε > 0, |L - 63.33| < ε := by
  sorry

end leo_weight_l3510_351068


namespace cube_sum_over_product_is_18_l3510_351064

theorem cube_sum_over_product_is_18 
  (x y z : ℂ) 
  (nonzero_x : x ≠ 0) 
  (nonzero_y : y ≠ 0) 
  (nonzero_z : z ≠ 0) 
  (sum_30 : x + y + z = 30) 
  (squared_diff_sum : (x - y)^2 + (x - z)^2 + (y - z)^2 + x*y*z = 2*x*y*z) : 
  (x^3 + y^3 + z^3) / (x*y*z) = 18 := by
sorry

end cube_sum_over_product_is_18_l3510_351064


namespace expression_evaluation_l3510_351022

theorem expression_evaluation (a b : ℤ) (h1 : a = 2) (h2 : b = -1) :
  ((2*a + 3*b) * (2*a - 3*b) - (2*a - b)^2 - 3*a*b) / (-b) = -12 := by
  sorry

end expression_evaluation_l3510_351022


namespace tenth_finger_number_l3510_351049

-- Define the function g based on the graph points
def g : ℕ → ℕ
| 0 => 5
| 1 => 0
| 2 => 4
| 3 => 8
| 4 => 3
| 5 => 7
| 6 => 2
| 7 => 6
| 8 => 1
| 9 => 5
| n => n  -- Default case for numbers not explicitly defined

-- Define a function that applies g n times to an initial value
def apply_g_n_times (n : ℕ) (initial : ℕ) : ℕ :=
  match n with
  | 0 => initial
  | k + 1 => g (apply_g_n_times k initial)

-- Theorem statement
theorem tenth_finger_number : apply_g_n_times 10 4 = 4 := by
  sorry

end tenth_finger_number_l3510_351049


namespace average_age_of_friends_l3510_351066

theorem average_age_of_friends (age1 age2 age3 : ℕ) : 
  age1 = 40 →
  age2 = 30 →
  age3 = age1 + 10 →
  (age1 + age2 + age3) / 3 = 40 := by
sorry

end average_age_of_friends_l3510_351066


namespace coefficient_c_nonzero_l3510_351015

/-- A polynomial of degree 4 with four distinct roots, one of which is 0 -/
structure QuarticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  has_four_distinct_roots : ∃ (p q r : ℝ), p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0 ∧
    ∀ x, x^4 + a*x^3 + b*x^2 + c*x + d = x*(x-p)*(x-q)*(x-r)
  zero_is_root : d = 0

theorem coefficient_c_nonzero (Q : QuarticPolynomial) : Q.c ≠ 0 := by
  sorry

end coefficient_c_nonzero_l3510_351015


namespace john_annual_cost_l3510_351021

def epipen_cost : ℝ := 500
def insurance_coverage : ℝ := 0.75
def replacements_per_year : ℕ := 2

def annual_cost : ℝ :=
  replacements_per_year * (epipen_cost * (1 - insurance_coverage))

theorem john_annual_cost : annual_cost = 250 := by
  sorry

end john_annual_cost_l3510_351021


namespace square_root_equation_l3510_351027

theorem square_root_equation (x : ℝ) : Real.sqrt (5 * x - 1) = 3 → x = 2 := by
  sorry

end square_root_equation_l3510_351027


namespace parabola_equation_l3510_351060

-- Define the parabola and its properties
structure Parabola where
  focus : ℝ × ℝ
  vertex : ℝ × ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the theorem
theorem parabola_equation (p : Parabola) :
  p.vertex = (0, 0) →
  p.focus.1 > 0 →
  p.focus.2 = 0 →
  (p.A.1 - p.focus.1, p.A.2 - p.focus.2) +
  (p.B.1 - p.focus.1, p.B.2 - p.focus.2) +
  (p.C.1 - p.focus.1, p.C.2 - p.focus.2) = (0, 0) →
  Real.sqrt ((p.A.1 - p.focus.1)^2 + (p.A.2 - p.focus.2)^2) +
  Real.sqrt ((p.B.1 - p.focus.1)^2 + (p.B.2 - p.focus.2)^2) +
  Real.sqrt ((p.C.1 - p.focus.1)^2 + (p.C.2 - p.focus.2)^2) = 6 →
  ∀ (x y : ℝ), (x, y) ∈ {(x, y) | y^2 = 8*x} ↔
    Real.sqrt ((x - p.focus.1)^2 + y^2) = x + p.focus.1 :=
by sorry

end parabola_equation_l3510_351060


namespace f_is_quadratic_l3510_351081

/-- A function f : ℝ → ℝ is quadratic if there exist real numbers a, b, and c
    with a ≠ 0 such that f(x) = ax^2 + bx + c for all x ∈ ℝ. -/
def IsQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function f(x) = (x + 1)^2 - 5 -/
def f (x : ℝ) : ℝ := (x + 1)^2 - 5

/-- Theorem: The function f(x) = (x + 1)^2 - 5 is a quadratic function -/
theorem f_is_quadratic : IsQuadratic f := by
  sorry


end f_is_quadratic_l3510_351081


namespace extremal_points_imply_a_gt_one_and_sum_gt_two_l3510_351036

open Real

/-- The function f(x) with parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := exp x - (1/2) * x^2 - a * x

/-- The derivative of f(x) -/
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := exp x - x - a

theorem extremal_points_imply_a_gt_one_and_sum_gt_two
  (a : ℝ)
  (x₁ x₂ : ℝ)
  (h₁ : f' a x₁ = 0)
  (h₂ : f' a x₂ = 0)
  (h₃ : x₁ ≠ x₂)
  : a > 1 ∧ f a x₁ + f a x₂ > 2 := by
  sorry

end extremal_points_imply_a_gt_one_and_sum_gt_two_l3510_351036


namespace f_negative_nine_halves_l3510_351078

def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

def periodic_2 (f : ℝ → ℝ) := ∀ x, f (x + 2) = f x

def f_on_unit_interval (f : ℝ → ℝ) := ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2 * x * (1 - x)

theorem f_negative_nine_halves 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_periodic : periodic_2 f) 
  (h_unit_interval : f_on_unit_interval f) : 
  f (-9/2) = -1/2 := by
sorry

end f_negative_nine_halves_l3510_351078


namespace nine_valid_sets_l3510_351092

def count_valid_sets : ℕ → Prop :=
  λ n => ∃ S : Finset (ℕ × ℕ × ℕ),
    (∀ (a b c : ℕ), (a, b, c) ∈ S ↔ 
      (Nat.gcd a b = 4 ∧ 
       Nat.lcm a c = 100 ∧ 
       Nat.lcm b c = 100 ∧ 
       a ≤ b)) ∧
    S.card = n

theorem nine_valid_sets : count_valid_sets 9 := by sorry

end nine_valid_sets_l3510_351092


namespace equation_solution_l3510_351013

theorem equation_solution (x : ℝ) : 
  2 - 1 / (3 - x) = 1 / (2 + x) → 
  x = (1 + Real.sqrt 15) / 2 ∨ x = (1 - Real.sqrt 15) / 2 := by
  sorry

end equation_solution_l3510_351013
