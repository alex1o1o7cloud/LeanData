import Mathlib

namespace division_remainder_seventeen_by_two_l1093_109377

theorem division_remainder_seventeen_by_two :
  ∃ (q : ℕ), 17 = 2 * q + 1 := by
  sorry

end division_remainder_seventeen_by_two_l1093_109377


namespace quadratic_equation_roots_l1093_109398

theorem quadratic_equation_roots : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  ((x₁ + 1) * (x₁ - 1) = 2 * x₁ + 3) ∧ ((x₂ + 1) * (x₂ - 1) = 2 * x₂ + 3) := by
  sorry

end quadratic_equation_roots_l1093_109398


namespace common_roots_product_l1093_109329

-- Define the two polynomial functions
def f (x : ℝ) : ℝ := x^3 + 3*x + 20
def g (x : ℝ) : ℝ := x^3 + 4*x^2 + 80

-- Define the property of having common roots
def has_common_roots (p q : ℝ → ℝ) : Prop :=
  ∃ (x y : ℝ), x ≠ y ∧ p x = 0 ∧ p y = 0 ∧ q x = 0 ∧ q y = 0

-- Theorem statement
theorem common_roots_product :
  has_common_roots f g →
  ∃ (x y : ℝ), x ≠ y ∧ f x = 0 ∧ f y = 0 ∧ g x = 0 ∧ g y = 0 ∧ x * y = 20 :=
by sorry

end common_roots_product_l1093_109329


namespace total_monthly_wages_after_new_hires_l1093_109390

/-- Calculate total monthly wages after new hires -/
theorem total_monthly_wages_after_new_hires 
  (initial_employees : ℕ) 
  (hourly_wage : ℕ) 
  (hours_per_day : ℕ) 
  (days_per_week : ℕ) 
  (weeks_per_month : ℕ) 
  (new_employees : ℕ) 
  (h1 : initial_employees = 500)
  (h2 : hourly_wage = 12)
  (h3 : hours_per_day = 10)
  (h4 : days_per_week = 5)
  (h5 : weeks_per_month = 4)
  (h6 : new_employees = 200) : 
  (initial_employees + new_employees) * 
  (hourly_wage * hours_per_day * days_per_week * weeks_per_month) = 1680000 := by
  sorry

#eval 700 * (12 * 10 * 5 * 4)  -- Should output 1680000

end total_monthly_wages_after_new_hires_l1093_109390


namespace population_growth_l1093_109386

theorem population_growth (P : ℝ) : 
  P * 1.1 * 1.2 = 1320 → P = 1000 := by
  sorry

end population_growth_l1093_109386


namespace like_terms_imply_sum_six_l1093_109336

/-- Two monomials are like terms if they have the same variables with the same exponents -/
def like_terms (term1 term2 : ℕ → ℕ → ℚ) : Prop :=
  ∀ (x y : ℕ), term1 x y ≠ 0 ∧ term2 x y ≠ 0 → x = 2 ∧ y = 3

/-- The first monomial 5a^m * b^3 -/
def term1 (m : ℕ) (x y : ℕ) : ℚ :=
  if x = m ∧ y = 3 then 5 else 0

/-- The second monomial -4a^2 * b^(n-1) -/
def term2 (n : ℕ) (x y : ℕ) : ℚ :=
  if x = 2 ∧ y = n - 1 then -4 else 0

theorem like_terms_imply_sum_six (m n : ℕ) :
  like_terms (term1 m) (term2 n) → m + n = 6 := by
  sorry

end like_terms_imply_sum_six_l1093_109336


namespace parabola_intersection_theorem_l1093_109374

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def line_through_focus (m : ℝ) (x y : ℝ) : Prop := x = m*y + 1

-- Define the intersection points
def intersection_points (m : ℝ) : ℝ × ℝ × ℝ × ℝ := sorry

-- Define the condition |MD| = 2|NF|
def length_condition (M N : ℝ × ℝ) : Prop := 
  let (x₁, y₁) := M
  let (x₂, y₂) := N
  x₁ = 2*x₂ + 2

-- Main theorem
theorem parabola_intersection_theorem (m : ℝ) : 
  let (x₁, y₁, x₂, y₂) := intersection_points m
  let M := (x₁, y₁)
  let N := (x₂, y₂)
  parabola x₁ y₁ ∧ 
  parabola x₂ y₂ ∧
  line_through_focus m x₁ y₁ ∧
  line_through_focus m x₂ y₂ ∧
  length_condition M N →
  Real.sqrt ((x₁ - 1)^2 + y₁^2) = 2 + Real.sqrt 3 := by
sorry

end parabola_intersection_theorem_l1093_109374


namespace factorial_equation_solution_l1093_109363

theorem factorial_equation_solution :
  ∃! N : ℕ, (6 : ℕ).factorial * (11 : ℕ).factorial = 20 * N.factorial :=
by sorry

end factorial_equation_solution_l1093_109363


namespace fill_time_with_leak_l1093_109338

/-- Time to fill the cistern without a leak (in hours) -/
def fill_time : ℝ := 8

/-- Time to empty the full cistern through the leak (in hours) -/
def empty_time : ℝ := 24

/-- Theorem: The time to fill the cistern with a leak is 12 hours -/
theorem fill_time_with_leak : 
  (1 / fill_time - 1 / empty_time)⁻¹ = 12 := by sorry

end fill_time_with_leak_l1093_109338


namespace modulus_of_complex_fraction_l1093_109301

theorem modulus_of_complex_fraction : 
  let i : ℂ := Complex.I
  let z : ℂ := (2 + i) / i
  Complex.abs z = Real.sqrt 5 := by
sorry

end modulus_of_complex_fraction_l1093_109301


namespace original_alcohol_percentage_l1093_109339

/-- Proves that given a 15-liter mixture of alcohol and water, if adding 3 liters of water
    results in a new mixture with 20.833333333333336% alcohol, then the original mixture
    contained 25% alcohol. -/
theorem original_alcohol_percentage
  (original_volume : ℝ)
  (added_water : ℝ)
  (new_alcohol_percentage : ℝ)
  (h1 : original_volume = 15)
  (h2 : added_water = 3)
  (h3 : new_alcohol_percentage = 20.833333333333336)
  : ∃ (original_alcohol_percentage : ℝ),
    original_alcohol_percentage = 25 ∧
    (original_alcohol_percentage / 100) * original_volume =
    (new_alcohol_percentage / 100) * (original_volume + added_water) :=
by sorry

end original_alcohol_percentage_l1093_109339


namespace jerry_money_duration_l1093_109354

/-- The number of weeks Jerry's money will last -/
def weeks_money_lasts (lawn_mowing_earnings weed_eating_earnings weekly_spending : ℕ) : ℕ :=
  (lawn_mowing_earnings + weed_eating_earnings) / weekly_spending

/-- Theorem: Given Jerry's earnings and weekly spending, his money will last 9 weeks -/
theorem jerry_money_duration :
  weeks_money_lasts 14 31 5 = 9 := by
  sorry

end jerry_money_duration_l1093_109354


namespace interest_rate_difference_l1093_109379

/-- Proves that the difference in interest rates is 3% given the problem conditions -/
theorem interest_rate_difference
  (principal : ℝ)
  (time : ℝ)
  (interest_difference : ℝ)
  (h_principal : principal = 5000)
  (h_time : time = 2)
  (h_interest_diff : interest_difference = 300)
  : ∃ (r dr : ℝ),
    principal * (r + dr) / 100 * time - principal * r / 100 * time = interest_difference ∧
    dr = 3 := by
  sorry

#check interest_rate_difference

end interest_rate_difference_l1093_109379


namespace largest_integer_satisfying_inequality_l1093_109361

theorem largest_integer_satisfying_inequality :
  ∀ n : ℕ, n^200 < 5^300 ↔ n ≤ 11 :=
by sorry

end largest_integer_satisfying_inequality_l1093_109361


namespace square_free_divisibility_l1093_109395

theorem square_free_divisibility (n : ℕ) (h1 : n > 1) (h2 : Squarefree n) :
  ∃ (p m : ℕ), Prime p ∧ p ∣ n ∧ n ∣ p^2 + p * m^p :=
sorry

end square_free_divisibility_l1093_109395


namespace binomial_probability_theorem_l1093_109368

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- The probability of a binomial random variable being greater than or equal to k -/
noncomputable def prob_ge (X : BinomialRV) (k : ℕ) : ℝ := sorry

/-- The theorem statement -/
theorem binomial_probability_theorem (ξ η : BinomialRV) 
  (h_ξ : ξ.n = 2) (h_η : η.n = 4) (h_p : ξ.p = η.p) 
  (h_prob : prob_ge ξ 1 = 5/9) : 
  prob_ge η 2 = 11/27 := by sorry

end binomial_probability_theorem_l1093_109368


namespace not_valid_base_5_l1093_109362

/-- Given a base k and a sequence of digits, determines if it's a valid representation in that base -/
def is_valid_base_k_number (k : ℕ) (digits : List ℕ) : Prop :=
  ∀ d ∈ digits, d < k

/-- The theorem states that 32501 is not a valid base-5 number -/
theorem not_valid_base_5 :
  ¬ (is_valid_base_k_number 5 [3, 2, 5, 0, 1]) :=
sorry

end not_valid_base_5_l1093_109362


namespace tangent_line_implies_b_minus_a_zero_l1093_109383

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + Real.log x

theorem tangent_line_implies_b_minus_a_zero (a b : ℝ) :
  (∀ x, f a b x = a * x^2 + b * x + Real.log x) →
  (∃ m c, ∀ x, m * x + c = 4 * x - 2 ∧ f a b 1 = m * 1 + c) →
  b - a = 0 := by
  sorry

end tangent_line_implies_b_minus_a_zero_l1093_109383


namespace point_on_hyperbola_l1093_109328

/-- A point (x, y) lies on the hyperbola y = -6/x if and only if xy = -6 -/
def lies_on_hyperbola (x y : ℝ) : Prop := x * y = -6

/-- The point (3, -2) lies on the hyperbola y = -6/x -/
theorem point_on_hyperbola : lies_on_hyperbola 3 (-2) := by
  sorry

end point_on_hyperbola_l1093_109328


namespace matrix_power_2018_l1093_109388

def A : Matrix (Fin 2) (Fin 2) ℕ := !![1, 0; 1, 1]

theorem matrix_power_2018 :
  A ^ 2018 = !![1, 0; 2018, 1] := by sorry

end matrix_power_2018_l1093_109388


namespace at_least_one_nonnegative_l1093_109347

theorem at_least_one_nonnegative (a b c d e f g h : ℝ) :
  (ac + bd ≥ 0) ∨ (ae + bf ≥ 0) ∨ (ag + bh ≥ 0) ∨ 
  (ce + df ≥ 0) ∨ (cg + dh ≥ 0) ∨ (eg + fh ≥ 0) :=
by sorry

end at_least_one_nonnegative_l1093_109347


namespace distribution_schemes_l1093_109394

def math_teachers : ℕ := 3
def chinese_teachers : ℕ := 6
def schools : ℕ := 3
def math_teachers_per_school : ℕ := 1
def chinese_teachers_per_school : ℕ := 2

theorem distribution_schemes :
  (math_teachers.factorial) *
  (chinese_teachers.choose chinese_teachers_per_school) *
  ((chinese_teachers - chinese_teachers_per_school).choose chinese_teachers_per_school) = 540 :=
sorry

end distribution_schemes_l1093_109394


namespace sequence_relation_l1093_109358

def x : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 4 * x (n + 1) - x n

def y : ℕ → ℤ
  | 0 => 1
  | 1 => 2
  | (n + 2) => 4 * y (n + 1) - y n

theorem sequence_relation (n : ℕ) : (y n)^2 = 3 * (x n)^2 + 1 := by
  sorry

end sequence_relation_l1093_109358


namespace downstream_speed_calculation_l1093_109325

/-- Represents the speed of a man rowing in different conditions -/
structure RowingSpeed where
  upstream : ℝ
  stillWater : ℝ

/-- Calculates the downstream speed given upstream and still water speeds -/
def downstreamSpeed (s : RowingSpeed) : ℝ :=
  2 * s.stillWater - s.upstream

theorem downstream_speed_calculation (s : RowingSpeed)
  (h1 : s.upstream = 25)
  (h2 : s.stillWater = 31) :
  downstreamSpeed s = 37 := by sorry

end downstream_speed_calculation_l1093_109325


namespace postage_cost_for_625_ounces_l1093_109366

/-- Calculates the postage cost for a letter -/
def postage_cost (weight : ℚ) (base_rate : ℚ) (additional_rate : ℚ) : ℚ :=
  let additional_weight := (weight - 1).ceil
  base_rate + additional_weight * additional_rate

theorem postage_cost_for_625_ounces :
  postage_cost (6.25 : ℚ) (0.50 : ℚ) (0.30 : ℚ) = (2.30 : ℚ) := by
  sorry

end postage_cost_for_625_ounces_l1093_109366


namespace football_field_fertilizer_l1093_109364

/-- Proves that the total amount of fertilizer spread across a football field is 800 pounds,
    given the field's area and a known fertilizer distribution over a portion of the field. -/
theorem football_field_fertilizer (total_area : ℝ) (partial_area : ℝ) (partial_fertilizer : ℝ) :
  total_area = 9600 →
  partial_area = 3600 →
  partial_fertilizer = 300 →
  (partial_fertilizer / partial_area) * total_area = 800 := by
  sorry

end football_field_fertilizer_l1093_109364


namespace imaginary_power_2011_l1093_109392

theorem imaginary_power_2011 (i : ℂ) (h : i^2 = -1) : i^2011 = -i := by
  sorry

end imaginary_power_2011_l1093_109392


namespace fence_cost_per_foot_l1093_109391

/-- Proves that for a square plot with an area of 36 sq ft and a total fencing cost of Rs. 1392, the price per foot of fencing is Rs. 58. -/
theorem fence_cost_per_foot (plot_area : ℝ) (total_cost : ℝ) (h1 : plot_area = 36) (h2 : total_cost = 1392) :
  let side_length : ℝ := Real.sqrt plot_area
  let perimeter : ℝ := 4 * side_length
  let cost_per_foot : ℝ := total_cost / perimeter
  cost_per_foot = 58 := by
  sorry

end fence_cost_per_foot_l1093_109391


namespace sum_of_integers_l1093_109371

theorem sum_of_integers (a b c d e : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e ∧
  (6 - a) * (6 - b) * (6 - c) * (6 - d) * (6 - e) = 45 →
  a + b + c + d + e = 25 := by
sorry

end sum_of_integers_l1093_109371


namespace mcpherson_rent_contribution_l1093_109376

/-- Calculates the amount Mr. McPherson needs to raise for rent -/
theorem mcpherson_rent_contribution 
  (total_rent : ℕ) 
  (mrs_mcpherson_percentage : ℚ) 
  (h1 : total_rent = 1200)
  (h2 : mrs_mcpherson_percentage = 30 / 100) : 
  total_rent - (mrs_mcpherson_percentage * total_rent).floor = 840 := by
sorry

end mcpherson_rent_contribution_l1093_109376


namespace fixed_point_theorem_l1093_109365

-- Define the parabola E: y^2 = 4x
def E : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 4 * p.1}

-- Define the directrix l: x = -1
def l : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = -1}

-- Define the focus F(1, 0)
def F : ℝ × ℝ := (1, 0)

-- Define a function to reflect a point across the x-axis
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

-- Main theorem
theorem fixed_point_theorem (A B : ℝ × ℝ) (h_A : A ∈ E) (h_B : B ∈ E) 
  (h_line : ∃ k : ℝ, A.2 - F.2 = k * (A.1 - F.1) ∧ B.2 - F.2 = k * (B.1 - F.1)) :
  ∃ t : ℝ, reflect_x A + t • (B - reflect_x A) = (-1, 0) :=
sorry

end fixed_point_theorem_l1093_109365


namespace line_not_in_second_quadrant_l1093_109304

/-- The line l with equation (a-2)y = (3a-1)x - 1 does not pass through the second quadrant
    if and only if a ∈ [2, +∞) -/
theorem line_not_in_second_quadrant (a : ℝ) :
  (∀ x y : ℝ, (a - 2) * y = (3 * a - 1) * x - 1 → ¬(x < 0 ∧ y > 0)) ↔ a ≥ 2 := by
  sorry

end line_not_in_second_quadrant_l1093_109304


namespace inscribed_rectangle_area_l1093_109385

-- Define the square
def square_area : ℝ := 24

-- Define the rectangle's side ratio
def rectangle_ratio : ℝ := 3

-- Theorem statement
theorem inscribed_rectangle_area :
  ∃ (x y : ℝ),
    x > 0 ∧ y > 0 ∧
    y = rectangle_ratio * x ∧
    x * y = 18 ∧
    x^2 + y^2 = square_area := by
  sorry

end inscribed_rectangle_area_l1093_109385


namespace average_of_c_and_d_l1093_109373

theorem average_of_c_and_d (c d : ℝ) : 
  (4 + 6 + 9 + c + d) / 5 = 18 → (c + d) / 2 = 35.5 := by
  sorry

end average_of_c_and_d_l1093_109373


namespace fifth_term_geometric_sequence_l1093_109330

def geometric_sequence (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r^(n - 1)

theorem fifth_term_geometric_sequence :
  let a₁ : ℚ := 2
  let a₂ : ℚ := 1/4
  let r : ℚ := a₂ / a₁
  geometric_sequence a₁ r 5 = 1/2048 := by
sorry

end fifth_term_geometric_sequence_l1093_109330


namespace largest_number_value_l1093_109317

theorem largest_number_value (a b c : ℝ) 
  (h_order : a < b ∧ b < c)
  (h_sum : a + b + c = 100)
  (h_diff_large : c - b = 10)
  (h_diff_small : b - a = 5) :
  c = 41.67 := by
sorry

end largest_number_value_l1093_109317


namespace count_specific_divisors_l1093_109387

/-- The number of positive integer divisors of 2016^2016 that are divisible by exactly 2016 positive integers -/
def divisors_with_2016_divisors : ℕ :=
  let base := 2016
  let exponent := 2016
  let target_divisors := 2016
  -- Definition of the function to count the divisors
  sorry

/-- The main theorem stating that the number of such divisors is 126 -/
theorem count_specific_divisors :
  divisors_with_2016_divisors = 126 := by sorry

end count_specific_divisors_l1093_109387


namespace cows_ran_away_after_10_days_l1093_109342

/-- The number of days that passed before cows ran away -/
def days_before_cows_ran_away (initial_cows : ℕ) (initial_duration : ℕ) (cows_ran_away : ℕ) : ℕ :=
  (initial_cows * initial_duration - (initial_cows - cows_ran_away) * initial_duration) / initial_cows

theorem cows_ran_away_after_10_days :
  days_before_cows_ran_away 1000 50 200 = 10 := by
  sorry

end cows_ran_away_after_10_days_l1093_109342


namespace total_students_l1093_109399

/-- The number of students who went to the movie -/
def M : ℕ := 10

/-- The number of students who went to the picnic -/
def P : ℕ := 20

/-- The number of students who played games -/
def G : ℕ := 5

/-- The number of students who went to both the movie and the picnic -/
def MP : ℕ := 4

/-- The number of students who went to both the movie and games -/
def MG : ℕ := 2

/-- The number of students who went to both the picnic and games -/
def PG : ℕ := 0

/-- The number of students who participated in all three activities -/
def MPG : ℕ := 2

/-- The total number of students -/
def T : ℕ := M + P + G - MP - MG - PG + MPG

theorem total_students : T = 31 := by
  sorry

end total_students_l1093_109399


namespace circle_C_equation_l1093_109324

/-- A circle C with center on the x-axis passing through points A(-1,1) and B(1,3) -/
structure CircleC where
  center : ℝ × ℝ
  center_on_x_axis : center.2 = 0
  passes_through_A : (center.1 + 1)^2 + (center.2 - 1)^2 = (center.1 - 1)^2 + (center.2 - 3)^2

/-- The equation of circle C is (x-2)²+y²=10 -/
theorem circle_C_equation (C : CircleC) :
  ∀ (x y : ℝ), (x - 2)^2 + y^2 = 10 ↔ (x - C.center.1)^2 + (y - C.center.2)^2 = (C.center.1 + 1)^2 + (C.center.2 - 1)^2 :=
by sorry

end circle_C_equation_l1093_109324


namespace extremum_values_l1093_109356

def f (a b x : ℝ) : ℝ := x^3 - a*x^2 - b*x + a^2

theorem extremum_values (a b : ℝ) : 
  (∀ x, f a b x ≤ f a b 1) ∨ (∀ x, f a b x ≥ f a b 1) →
  f a b 1 = 10 →
  a = -4 ∧ b = 11 := by
sorry

end extremum_values_l1093_109356


namespace orthogonal_vectors_l1093_109308

theorem orthogonal_vectors (x y : ℝ) : 
  (3 * x + 4 * (-2) = 0 ∧ 3 * 1 + 4 * y = 0) ↔ (x = 8/3 ∧ y = -3/4) := by
  sorry

end orthogonal_vectors_l1093_109308


namespace opposite_number_l1093_109384

theorem opposite_number (a : ℝ) : -a = -2023 → a = 2023 := by
  sorry

end opposite_number_l1093_109384


namespace only_striped_has_eight_legs_l1093_109307

/-- Represents the color of an octopus -/
inductive OctopusColor
  | Green
  | DarkBlue
  | Purple
  | Striped

/-- Represents an octopus with its color and number of legs -/
structure Octopus where
  color : OctopusColor
  legs : ℕ

/-- Determines if an octopus tells the truth based on its number of legs -/
def tellsTruth (o : Octopus) : Prop :=
  o.legs % 2 = 0

/-- Represents the statements made by each octopus -/
def greenStatement (green darkBlue : Octopus) : Prop :=
  green.legs = 8 ∧ darkBlue.legs = 6

def darkBlueStatement (darkBlue green : Octopus) : Prop :=
  darkBlue.legs = 8 ∧ green.legs = 7

def purpleStatement (darkBlue purple : Octopus) : Prop :=
  darkBlue.legs = 8 ∧ purple.legs = 9

def stripedStatement (green darkBlue purple striped : Octopus) : Prop :=
  green.legs ≠ 8 ∧ darkBlue.legs ≠ 8 ∧ purple.legs ≠ 8 ∧ striped.legs = 8

/-- The main theorem stating that only the striped octopus has 8 legs -/
theorem only_striped_has_eight_legs
  (green darkBlue purple striped : Octopus)
  (h_green : green.color = OctopusColor.Green)
  (h_darkBlue : darkBlue.color = OctopusColor.DarkBlue)
  (h_purple : purple.color = OctopusColor.Purple)
  (h_striped : striped.color = OctopusColor.Striped)
  (h_greenStatement : tellsTruth green = greenStatement green darkBlue)
  (h_darkBlueStatement : tellsTruth darkBlue = darkBlueStatement darkBlue green)
  (h_purpleStatement : tellsTruth purple = purpleStatement darkBlue purple)
  (h_stripedStatement : tellsTruth striped = stripedStatement green darkBlue purple striped) :
  striped.legs = 8 ∧ green.legs ≠ 8 ∧ darkBlue.legs ≠ 8 ∧ purple.legs ≠ 8 :=
sorry

end only_striped_has_eight_legs_l1093_109307


namespace laptop_repairs_count_l1093_109370

def phone_repair_cost : ℕ := 11
def laptop_repair_cost : ℕ := 15
def computer_repair_cost : ℕ := 18
def phone_repairs : ℕ := 5
def computer_repairs : ℕ := 2
def total_earnings : ℕ := 121

theorem laptop_repairs_count :
  ∃ (laptop_repairs : ℕ),
    phone_repair_cost * phone_repairs +
    laptop_repair_cost * laptop_repairs +
    computer_repair_cost * computer_repairs = total_earnings ∧
    laptop_repairs = 2 := by
  sorry

end laptop_repairs_count_l1093_109370


namespace sector_central_angle_l1093_109305

/-- Given a circular sector with arc length 4 and area 4, prove that its central angle in radians is 2. -/
theorem sector_central_angle (arc_length area : ℝ) (h1 : arc_length = 4) (h2 : area = 4) :
  let r := 2 * area / arc_length
  2 * area / (r ^ 2) = 2 := by sorry

end sector_central_angle_l1093_109305


namespace sinusoidal_amplitude_l1093_109397

/-- Given a sinusoidal function f(x) = a * sin(bx + c) + d with positive constants a, b, c, d,
    if the maximum value of f is 3 and the minimum value is -1, then a = 2 -/
theorem sinusoidal_amplitude (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  let f : ℝ → ℝ := λ x ↦ a * Real.sin (b * x + c) + d
  (∀ x, f x ≤ 3) ∧ (∀ x, f x ≥ -1) ∧ (∃ x y, f x = 3 ∧ f y = -1) → a = 2 := by
  sorry

end sinusoidal_amplitude_l1093_109397


namespace tangent_line_perpendicular_main_theorem_l1093_109332

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + x - 1

-- Define the derivative of f(x)
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*a*x + 1

-- Theorem statement
theorem tangent_line_perpendicular (a : ℝ) : 
  (f' a 1 = 2) → a = 1 := by
  sorry

-- Main theorem
theorem main_theorem (a : ℝ) : 
  (∃ (k : ℝ), f' a 1 = k ∧ k * (-1/2) = -1) → a = 1 := by
  sorry

end tangent_line_perpendicular_main_theorem_l1093_109332


namespace equation_one_solution_l1093_109348

theorem equation_one_solution :
  ∀ x : ℝ, x^4 - x^2 - 6 = 0 ↔ x = Real.sqrt 3 ∨ x = -Real.sqrt 3 := by
sorry

end equation_one_solution_l1093_109348


namespace max_symmetry_axes_is_2k_l1093_109343

/-- The maximum number of axes of symmetry for the union of k line segments on a plane -/
def max_symmetry_axes (k : ℕ) : ℕ := 2 * k

/-- Theorem: The maximum number of axes of symmetry for the union of k line segments on a plane is 2k -/
theorem max_symmetry_axes_is_2k (k : ℕ) :
  max_symmetry_axes k = 2 * k :=
by sorry

end max_symmetry_axes_is_2k_l1093_109343


namespace part_one_part_two_l1093_109316

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part I
theorem part_one (a : ℝ) : 
  (∀ x, f a x ≥ 3 ↔ x ≤ 1 ∨ x ≥ 5) → a = 2 :=
sorry

-- Part II
theorem part_two (m : ℝ) :
  (∀ x, f 2 x + f 2 (x + 4) ≥ m) → m ≤ 4 :=
sorry

end part_one_part_two_l1093_109316


namespace specific_rectangle_perimeter_l1093_109303

/-- Represents a rectangle with two internal segments --/
structure CutRectangle where
  AD : ℝ  -- Length of side AD
  AB : ℝ  -- Length of side AB
  EF : ℝ  -- Length of internal segment EF
  GH : ℝ  -- Length of internal segment GH

/-- Calculates the total perimeter of the two shapes formed by cutting the rectangle --/
def totalPerimeter (r : CutRectangle) : ℝ :=
  2 * (r.AD + r.AB + r.EF + r.GH)

/-- Theorem stating that for a specific rectangle, the total perimeter is 40 --/
theorem specific_rectangle_perimeter :
  ∃ (r : CutRectangle), r.AD = 10 ∧ r.AB = 6 ∧ r.EF = 2 ∧ r.GH = 2 ∧ totalPerimeter r = 40 := by
  sorry

end specific_rectangle_perimeter_l1093_109303


namespace inequality_proof_l1093_109335

theorem inequality_proof (x y z : ℝ) 
  (sum_zero : x + y + z = 0) 
  (abs_sum_leq_one : |x| + |y| + |z| ≤ 1) : 
  x + y/3 + z/5 ≤ 2/5 := by
sorry

end inequality_proof_l1093_109335


namespace houses_painted_in_three_hours_l1093_109369

/-- The number of houses that can be painted in a given time -/
def houses_painted (minutes_per_house : ℕ) (total_hours : ℕ) : ℕ :=
  (total_hours * 60) / minutes_per_house

/-- Theorem: Given it takes 20 minutes to paint a house, 
    the number of houses that can be painted in 3 hours is 9 -/
theorem houses_painted_in_three_hours :
  houses_painted 20 3 = 9 := by
  sorry

end houses_painted_in_three_hours_l1093_109369


namespace sum_of_specific_numbers_l1093_109346

theorem sum_of_specific_numbers :
  36 + 17 + 32 + 54 + 28 + 3 = 170 := by
  sorry

end sum_of_specific_numbers_l1093_109346


namespace smallest_bench_arrangement_l1093_109367

theorem smallest_bench_arrangement (n : ℕ) : 
  (∃ k : ℕ, 8 * n = 10 * k) ∧ 
  (n % 8 = 0) ∧ (n % 10 = 0) ∧
  (∀ m : ℕ, m < n → ¬((∃ k : ℕ, 8 * m = 10 * k) ∧ (m % 8 = 0) ∧ (m % 10 = 0))) →
  n = 20 := by
sorry

end smallest_bench_arrangement_l1093_109367


namespace nancy_mexican_antacids_l1093_109322

/-- Represents the number of antacids Nancy takes per day when eating Mexican food -/
def mexican_antacids : ℕ := sorry

/-- Represents the number of antacids Nancy takes per day when eating Indian food -/
def indian_antacids : ℕ := 3

/-- Represents the number of antacids Nancy takes per day when eating other food -/
def other_antacids : ℕ := 1

/-- Represents the number of times Nancy eats Indian food per week -/
def indian_meals_per_week : ℕ := 3

/-- Represents the number of times Nancy eats Mexican food per week -/
def mexican_meals_per_week : ℕ := 2

/-- Represents the number of antacids Nancy takes per month -/
def antacids_per_month : ℕ := 60

/-- Represents the number of weeks in a month (approximated) -/
def weeks_per_month : ℕ := 4

theorem nancy_mexican_antacids : 
  mexican_antacids = 2 :=
by sorry

end nancy_mexican_antacids_l1093_109322


namespace job_total_amount_l1093_109309

/-- Calculates the total amount earned for a job given the time taken by two workers and one worker's share. -/
theorem job_total_amount 
  (rahul_days : ℚ) 
  (rajesh_days : ℚ) 
  (rahul_share : ℚ) : 
  rahul_days = 3 → 
  rajesh_days = 2 → 
  rahul_share = 142 → 
  ∃ (total_amount : ℚ), total_amount = 355 := by
  sorry

#check job_total_amount

end job_total_amount_l1093_109309


namespace angle_ratios_l1093_109352

theorem angle_ratios (α : Real) (h : Real.tan α = -3/4) :
  (Real.cos (π/2 + α) * Real.sin (-π - α)) / (Real.cos (11*π/2 - α) * Real.sin (9*π/2 + α)) = -3/4 ∧
  2 + Real.sin α * Real.cos α - (Real.cos α)^2 = 22/25 := by
  sorry

end angle_ratios_l1093_109352


namespace italian_sausage_length_l1093_109327

/-- The length of an Italian sausage in inches -/
def sausage_length : ℚ := 12 * (2 / 3)

/-- Theorem: The length of the Italian sausage is 8 inches -/
theorem italian_sausage_length : sausage_length = 8 := by
  sorry

end italian_sausage_length_l1093_109327


namespace three_digit_numbers_satisfying_condition_l1093_109310

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def abc_to_number (a b c : ℕ) : ℕ :=
  100 * a + 10 * b + c

def bca_to_number (a b c : ℕ) : ℕ :=
  100 * b + 10 * c + a

def cab_to_number (a b c : ℕ) : ℕ :=
  100 * c + 10 * a + b

def satisfies_condition (n : ℕ) : Prop :=
  ∃ a b c : ℕ,
    n = abc_to_number a b c ∧
    2 * n = bca_to_number a b c + cab_to_number a b c

theorem three_digit_numbers_satisfying_condition :
  {n : ℕ | is_three_digit_number n ∧ satisfies_condition n} = {481, 518, 592, 629} :=
sorry

end three_digit_numbers_satisfying_condition_l1093_109310


namespace vector_parallel_implies_m_value_l1093_109380

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 * b.2 = k * a.2 * b.1

theorem vector_parallel_implies_m_value :
  ∀ m : ℝ,
  let a : ℝ × ℝ := (2, 1 + m)
  let b : ℝ × ℝ := (3, m)
  parallel a b → m = -3 := by
  sorry

end vector_parallel_implies_m_value_l1093_109380


namespace aisha_head_fraction_l1093_109314

/-- Miss Aisha's height measurements -/
structure AishaHeight where
  total : ℝ
  legs : ℝ
  rest : ℝ
  head : ℝ

/-- Properties of Miss Aisha's height -/
def aisha_properties (h : AishaHeight) : Prop :=
  h.total = 60 ∧
  h.legs = (1/3) * h.total ∧
  h.rest = 25 ∧
  h.head = h.total - (h.legs + h.rest)

/-- Theorem: Miss Aisha's head is 1/4 of her total height -/
theorem aisha_head_fraction (h : AishaHeight) 
  (hprops : aisha_properties h) : h.head / h.total = 1/4 := by
  sorry

end aisha_head_fraction_l1093_109314


namespace baking_distribution_problem_l1093_109393

/-- Calculates the number of leftover items when distributing a total number of items into containers of a specific capacity -/
def leftovers (total : ℕ) (capacity : ℕ) : ℕ :=
  total % capacity

/-- Represents the baking and distribution problem -/
theorem baking_distribution_problem 
  (gingerbread_batches : ℕ) (gingerbread_per_batch : ℕ) (gingerbread_per_jar : ℕ)
  (sugar_batches : ℕ) (sugar_per_batch : ℕ) (sugar_per_box : ℕ)
  (tart_batches : ℕ) (tarts_per_batch : ℕ) (tarts_per_box : ℕ)
  (h_gingerbread : gingerbread_batches = 3 ∧ gingerbread_per_batch = 47 ∧ gingerbread_per_jar = 6)
  (h_sugar : sugar_batches = 2 ∧ sugar_per_batch = 78 ∧ sugar_per_box = 9)
  (h_tart : tart_batches = 4 ∧ tarts_per_batch = 36 ∧ tarts_per_box = 4) :
  leftovers (gingerbread_batches * gingerbread_per_batch) gingerbread_per_jar = 3 ∧
  leftovers (sugar_batches * sugar_per_batch) sugar_per_box = 3 ∧
  leftovers (tart_batches * tarts_per_batch) tarts_per_box = 0 :=
by
  sorry

end baking_distribution_problem_l1093_109393


namespace limit_of_ratio_l1093_109326

def arithmetic_sequence (n : ℕ) : ℝ := 2 * n - 1

def sum_of_terms (n : ℕ) : ℝ := n^2

theorem limit_of_ratio :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |sum_of_terms n / (arithmetic_sequence n)^2 - 1/4| < ε :=
by sorry

end limit_of_ratio_l1093_109326


namespace polynomial_factorization_l1093_109351

theorem polynomial_factorization (z : ℂ) : z^6 - 64*z^2 = z^2 * (z^2 - 8) * (z^2 + 8) := by
  sorry

end polynomial_factorization_l1093_109351


namespace smallest_winning_points_l1093_109375

/-- Represents the possible placings in a race -/
inductive Placing
| First
| Second
| Third
| Other

/-- Calculates the points for a given placing -/
def points_for_placing (p : Placing) : ℕ :=
  match p with
  | Placing.First => 7
  | Placing.Second => 4
  | Placing.Third => 2
  | Placing.Other => 0

/-- Calculates the total points for a list of placings -/
def total_points (placings : List Placing) : ℕ :=
  placings.map points_for_placing |>.sum

/-- Represents the results of four races -/
def RaceResults := List Placing

/-- Checks if a given point total guarantees winning -/
def guarantees_win (points : ℕ) : Prop :=
  ∀ (other_results : RaceResults), 
    other_results.length = 4 → total_points other_results < points

theorem smallest_winning_points : 
  (guarantees_win 25) ∧ (∀ p : ℕ, p < 25 → ¬guarantees_win p) := by
  sorry

end smallest_winning_points_l1093_109375


namespace adelaide_ducks_main_theorem_l1093_109337

/-- Proves that Adelaide bought 30 ducks given the conditions of the problem -/
theorem adelaide_ducks : ℕ → ℕ → ℕ → Prop :=
  fun adelaide ephraim kolton =>
    adelaide = 2 * ephraim ∧
    ephraim = kolton - 45 ∧
    (adelaide + ephraim + kolton) / 3 = 35 →
    adelaide = 30

/-- Main theorem statement -/
theorem main_theorem : ∃ (a e k : ℕ), adelaide_ducks a e k :=
  sorry

end adelaide_ducks_main_theorem_l1093_109337


namespace quadrilateral_area_l1093_109372

/-- The area of a quadrilateral given its four sides and the angle between diagonals -/
theorem quadrilateral_area (a b c d ω : ℝ) (h_pos : 0 < ω ∧ ω < π) :
  ∃ t : ℝ, t = (1/4) * (b^2 + d^2 - a^2 - c^2) * Real.tan ω :=
by sorry

end quadrilateral_area_l1093_109372


namespace std_dev_and_range_invariance_l1093_109323

variable {n : ℕ} (c : ℝ)
variable (X Y : Fin n → ℝ)

def add_constant (X : Fin n → ℝ) (c : ℝ) : Fin n → ℝ :=
  fun i => X i + c

def sample_std_dev (X : Fin n → ℝ) : ℝ := sorry

def sample_range (X : Fin n → ℝ) : ℝ := sorry

theorem std_dev_and_range_invariance
  (h_nonzero : c ≠ 0)
  (h_Y : Y = add_constant X c) :
  sample_std_dev X = sample_std_dev Y ∧
  sample_range X = sample_range Y := by sorry

end std_dev_and_range_invariance_l1093_109323


namespace cube_root_not_always_two_l1093_109381

theorem cube_root_not_always_two (x : ℝ) (h : x^2 = 64) : 
  ∃ y, y^3 = x ∧ y ≠ 2 :=
by sorry

end cube_root_not_always_two_l1093_109381


namespace return_flight_theorem_l1093_109349

/-- Represents a direction in degrees relative to a cardinal direction -/
structure Direction where
  angle : ℝ
  cardinal : String
  relative : String

/-- Represents a flight path -/
structure FlightPath where
  distance : ℝ
  direction : Direction

/-- Returns the opposite direction for a given flight path -/
def oppositeDirection (fp : FlightPath) : Direction :=
  { angle := fp.direction.angle,
    cardinal := if fp.direction.cardinal = "east" then "west" else "east",
    relative := if fp.direction.relative = "south" then "north" else "south" }

theorem return_flight_theorem (outbound : FlightPath) 
  (h1 : outbound.distance = 1200)
  (h2 : outbound.direction.angle = 30)
  (h3 : outbound.direction.cardinal = "east")
  (h4 : outbound.direction.relative = "south") :
  ∃ (inbound : FlightPath),
    inbound.distance = outbound.distance ∧
    inbound.direction = oppositeDirection outbound :=
  sorry

end return_flight_theorem_l1093_109349


namespace problem_solution_l1093_109359

theorem problem_solution (m n : ℝ) : 
  (Real.sqrt (1 - m))^2 + |n + 2| = 0 → m - n = 3 := by
sorry

end problem_solution_l1093_109359


namespace meaningful_condition_l1093_109313

def is_meaningful (x : ℝ) : Prop :=
  x > -1 ∧ x ≠ 1

theorem meaningful_condition (x : ℝ) : 
  (∃ y : ℝ, y = Real.sqrt (x + 1) / (x - 1)) ↔ is_meaningful x :=
sorry

end meaningful_condition_l1093_109313


namespace derived_function_coefficients_target_point_coords_two_base_points_and_distance_range_l1093_109334

/-- Definition of a derived function -/
def is_derived_function (a b c : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ ≠ 0 ∧ x₂ ≠ 0 ∧
  (a * x₁ + b = c / x₂) ∧
  (x₁ = -x₂)

/-- Part 1: Derived function coefficients -/
theorem derived_function_coefficients :
  is_derived_function 2 4 5 := by sorry

/-- Part 2: Target point coordinates -/
theorem target_point_coords (b c : ℝ) :
  is_derived_function 1 b c →
  (∃ (x : ℝ), x^2 + b*x + c = 0) →
  (1 + b = -c) →
  (∃ (x y : ℝ), x = -1 ∧ y = -1 ∧ y = c / x) := by sorry

/-- Part 3: Existence of two base points and their distance range -/
theorem two_base_points_and_distance_range (a b : ℝ) :
  a > b ∧ b > 0 →
  is_derived_function a (2*b) (-2) →
  (∃ (x : ℝ), a*x^2 + 2*b*x - 2 = 6) →
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ a*x₁ + 2*b = a*x₁^2 + 2*b*x₁ - 2 ∧ a*x₂ + 2*b = a*x₂^2 + 2*b*x₂ - 2) ∧
  (∃ (x₁ x₂ : ℝ), 2 < |x₁ - x₂| ∧ |x₁ - x₂| < 2 * Real.sqrt 3) := by sorry

end derived_function_coefficients_target_point_coords_two_base_points_and_distance_range_l1093_109334


namespace calculator_result_l1093_109315

def calculator_operation (n : ℕ) : ℕ :=
  let doubled := n * 2
  let swapped := (doubled % 10) * 10 + (doubled / 10)
  swapped + 2

def is_valid_input (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 49

theorem calculator_result :
  (∃ n : ℕ, is_valid_input n ∧ calculator_operation n = 44) ∧
  (¬ ∃ n : ℕ, is_valid_input n ∧ calculator_operation n = 43) ∧
  (¬ ∃ n : ℕ, is_valid_input n ∧ calculator_operation n = 42) ∧
  (¬ ∃ n : ℕ, is_valid_input n ∧ calculator_operation n = 41) :=
sorry

end calculator_result_l1093_109315


namespace rectangle_area_l1093_109306

/-- Given a rectangle where the length is thrice the breadth and the perimeter is 96,
    prove that its area is 432. -/
theorem rectangle_area (breadth : ℝ) (length : ℝ) : 
  length = 3 * breadth → 
  2 * (length + breadth) = 96 → 
  length * breadth = 432 := by
  sorry

end rectangle_area_l1093_109306


namespace average_salary_before_manager_l1093_109302

/-- Proves that the average salary of employees is 1500 given the conditions -/
theorem average_salary_before_manager (num_employees : ℕ) (manager_salary : ℕ) (avg_increase : ℕ) :
  num_employees = 20 →
  manager_salary = 12000 →
  avg_increase = 500 →
  (∃ (avg_salary : ℕ),
    (num_employees + 1) * (avg_salary + avg_increase) = num_employees * avg_salary + manager_salary ∧
    avg_salary = 1500) :=
by sorry

end average_salary_before_manager_l1093_109302


namespace line_passes_through_points_l1093_109345

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space using the general form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point lies on a given line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The two given points -/
def p1 : Point := ⟨-1, 1⟩
def p2 : Point := ⟨3, 9⟩

/-- The line we want to prove passes through the given points -/
def line : Line := ⟨2, -1, 3⟩

/-- Theorem stating that the given line passes through both points -/
theorem line_passes_through_points : 
  p1.liesOn line ∧ p2.liesOn line := by sorry

end line_passes_through_points_l1093_109345


namespace even_four_digit_count_is_336_l1093_109340

/-- A function that counts the number of even integers between 4000 and 8000 with four different digits -/
def count_even_four_digit_numbers : ℕ :=
  336

/-- Theorem stating that the count of even integers between 4000 and 8000 with four different digits is 336 -/
theorem even_four_digit_count_is_336 : count_even_four_digit_numbers = 336 := by
  sorry

end even_four_digit_count_is_336_l1093_109340


namespace quadratic_counterexample_l1093_109318

theorem quadratic_counterexample :
  ∃ m : ℝ, m < -2 ∧ ∀ x : ℝ, x^2 + m*x + 4 ≠ 0 :=
by sorry

end quadratic_counterexample_l1093_109318


namespace joined_hexagon_triangle_edges_l1093_109355

/-- A regular polygon with n sides and side length 1 -/
structure RegularPolygon where
  sides : ℕ
  sideLength : ℝ
  regularHexagon : sides = 6 → sideLength = 1
  regularTriangle : sides = 3 → sideLength = 1

/-- The number of edges in a shape formed by joining a regular hexagon and a regular triangle -/
def joinedEdges (hexagon triangle : RegularPolygon) : ℕ :=
  hexagon.sides + triangle.sides - 3

theorem joined_hexagon_triangle_edges :
  ∀ (hexagon triangle : RegularPolygon),
  hexagon.sides = 6 ∧ 
  triangle.sides = 3 ∧ 
  hexagon.sideLength = 1 ∧ 
  triangle.sideLength = 1 →
  joinedEdges hexagon triangle = 5 := by
  sorry

end joined_hexagon_triangle_edges_l1093_109355


namespace simplify_expression_l1093_109311

theorem simplify_expression (x : ℝ) (h : x ≥ 0) : 
  (1/2 * x^(1/2))^4 = 1/16 * x^2 := by sorry

end simplify_expression_l1093_109311


namespace distance_to_big_rock_big_rock_distance_l1093_109300

/-- The distance to Big Rock given the rower's speed, river current, and round trip time -/
theorem distance_to_big_rock (v : ℝ) (c : ℝ) (t : ℝ) : 
  v > c ∧ v > 0 ∧ c > 0 ∧ t > 0 → 
  (v + c)⁻¹ * d + (v - c)⁻¹ * d = t → 
  d = (t * v^2 - t * c^2) / (2 * v) :=
by sorry

/-- The specific case for the given problem -/
theorem big_rock_distance : 
  let v := 6 -- rower's speed in still water
  let c := 1 -- river current speed
  let t := 1 -- total time for round trip
  let d := (t * v^2 - t * c^2) / (2 * v) -- distance to Big Rock
  d = 35 / 12 :=
by sorry

end distance_to_big_rock_big_rock_distance_l1093_109300


namespace function_range_theorem_l1093_109378

open Real

theorem function_range_theorem (f : ℝ → ℝ) 
  (h1 : ∀ x > 0, 9 * f x < x * (deriv f x) ∧ x * (deriv f x) < 10 * f x)
  (h2 : ∀ x > 0, f x > 0) :
  2^9 < f 2 / f 1 ∧ f 2 / f 1 < 2^10 := by
sorry

end function_range_theorem_l1093_109378


namespace arithmetic_sequence_problem_l1093_109396

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a where a₃ = 16 and a₉ = 80,
    prove that a₆ = 48. -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : is_arithmetic_sequence a)
    (h_3 : a 3 = 16)
    (h_9 : a 9 = 80) : 
  a 6 = 48 := by
sorry

end arithmetic_sequence_problem_l1093_109396


namespace cookie_difference_l1093_109341

/-- Proves that the difference between the number of cookies in 8 boxes and 9 bags is 33,
    given that each bag contains 7 cookies and each box contains 12 cookies. -/
theorem cookie_difference :
  let cookies_per_bag : ℕ := 7
  let cookies_per_box : ℕ := 12
  let num_boxes : ℕ := 8
  let num_bags : ℕ := 9
  (num_boxes * cookies_per_box) - (num_bags * cookies_per_bag) = 33 := by
  sorry

end cookie_difference_l1093_109341


namespace custom_mul_one_neg_three_l1093_109350

-- Define the custom multiplication operation
def custom_mul (a b : ℤ) : ℤ := a^2 + 2*a*b - b^2

-- Theorem statement
theorem custom_mul_one_neg_three :
  custom_mul 1 (-3) = -14 :=
by
  sorry

end custom_mul_one_neg_three_l1093_109350


namespace residue_negative_999_mod_25_l1093_109331

theorem residue_negative_999_mod_25 : Int.mod (-999) 25 = 1 := by
  sorry

end residue_negative_999_mod_25_l1093_109331


namespace james_run_calories_l1093_109353

/-- Calculates the calories burned per minute during James' run -/
def caloriesBurnedPerMinute (bagsEaten : ℕ) (ouncesPerBag : ℕ) (caloriesPerOunce : ℕ) 
  (runDuration : ℕ) (excessCalories : ℕ) : ℕ :=
  let totalOunces := bagsEaten * ouncesPerBag
  let totalCaloriesConsumed := totalOunces * caloriesPerOunce
  let caloriesBurned := totalCaloriesConsumed - excessCalories
  caloriesBurned / runDuration

theorem james_run_calories : 
  caloriesBurnedPerMinute 3 2 150 40 420 = 12 := by
  sorry

end james_run_calories_l1093_109353


namespace total_teachers_is_182_l1093_109357

/-- Represents the number of teachers in different categories and sampling information -/
structure TeacherInfo where
  senior_teachers : ℕ
  intermediate_teachers : ℕ
  total_sampled : ℕ
  other_sampled : ℕ

/-- Calculates the total number of teachers in the school -/
def total_teachers (info : TeacherInfo) : ℕ :=
  let senior_intermediate := info.senior_teachers + info.intermediate_teachers
  let senior_intermediate_sampled := info.total_sampled - info.other_sampled
  (senior_intermediate * info.total_sampled) / senior_intermediate_sampled

/-- Theorem stating that the total number of teachers is 182 -/
theorem total_teachers_is_182 (info : TeacherInfo) 
    (h1 : info.senior_teachers = 26)
    (h2 : info.intermediate_teachers = 104)
    (h3 : info.total_sampled = 56)
    (h4 : info.other_sampled = 16) :
    total_teachers info = 182 := by
  sorry

#eval total_teachers { senior_teachers := 26, intermediate_teachers := 104, total_sampled := 56, other_sampled := 16 }

end total_teachers_is_182_l1093_109357


namespace sets_equality_l1093_109382

-- Define the sets M, N, and P
def M : Set ℝ := {x | ∃ n : ℤ, x = n}
def N : Set ℝ := {x | ∃ n : ℤ, x = n / 2}
def P : Set ℝ := {x | ∃ n : ℤ, x = n + 1/2}

-- Theorem statement
theorem sets_equality : N = M ∪ P := by sorry

end sets_equality_l1093_109382


namespace relationship_abc_l1093_109333

theorem relationship_abc (a b c : ℝ) (ha : a = Real.exp 0.3) (hb : b = 0.9^2) (hc : c = Real.log 0.9) :
  c < b ∧ b < a := by
  sorry

end relationship_abc_l1093_109333


namespace p_sufficient_not_necessary_l1093_109344

/-- Proposition p: x = 1 and y = 1 -/
def p (x y : ℝ) : Prop := x = 1 ∧ y = 1

/-- Proposition q: x + y = 2 -/
def q (x y : ℝ) : Prop := x + y = 2

/-- p is a sufficient but not necessary condition for q -/
theorem p_sufficient_not_necessary :
  (∀ x y : ℝ, p x y → q x y) ∧
  (∃ x y : ℝ, q x y ∧ ¬p x y) :=
sorry

end p_sufficient_not_necessary_l1093_109344


namespace distribute_five_to_two_nonempty_l1093_109360

theorem distribute_five_to_two_nonempty (n : Nat) (k : Nat) : 
  n = 5 → k = 2 → (Finset.sum (Finset.range (n - 1)) (λ i => Nat.choose n (i + 1) * 2)) = 30 := by
  sorry

end distribute_five_to_two_nonempty_l1093_109360


namespace angle_sum_around_point_l1093_109319

theorem angle_sum_around_point (y : ℝ) : 
  y + y + 140 = 360 → y = 110 := by sorry

end angle_sum_around_point_l1093_109319


namespace friend_payment_amount_l1093_109389

/-- The cost per item for each food item --/
def hamburger_cost : ℚ := 3
def fries_cost : ℚ := 6/5  -- 1.20 as a rational number
def soda_cost : ℚ := 1/2
def spaghetti_cost : ℚ := 27/10
def milkshake_cost : ℚ := 5/2
def nuggets_cost : ℚ := 7/2

/-- The number of each item ordered --/
def hamburger_count : ℕ := 5
def fries_count : ℕ := 4
def soda_count : ℕ := 5
def spaghetti_count : ℕ := 1
def milkshake_count : ℕ := 3
def nuggets_count : ℕ := 2

/-- The discount percentage as a rational number --/
def discount_percent : ℚ := 1/10

/-- The percentage of the bill paid by the birthday friend --/
def birthday_friend_percent : ℚ := 3/10

/-- The number of friends splitting the remaining bill --/
def remaining_friends : ℕ := 4

/-- The theorem stating that each remaining friend will pay $6.22 --/
theorem friend_payment_amount : 
  let total_bill := hamburger_cost * hamburger_count + 
                    fries_cost * fries_count +
                    soda_cost * soda_count +
                    spaghetti_cost * spaghetti_count +
                    milkshake_cost * milkshake_count +
                    nuggets_cost * nuggets_count
  let discounted_bill := total_bill * (1 - discount_percent)
  let remaining_bill := discounted_bill * (1 - birthday_friend_percent)
  remaining_bill / remaining_friends = 311/50  -- 6.22 as a rational number
  := by sorry

end friend_payment_amount_l1093_109389


namespace x_cubed_term_is_seventh_l1093_109312

/-- The exponent of the binomial expansion -/
def n : ℕ := 16

/-- The general term of the expansion -/
def T (r : ℕ) : ℚ → ℚ := λ x => 2^r * Nat.choose n r * x^(8 - 5/6 * r)

/-- The index of the term containing x^3 -/
def r : ℕ := 6

theorem x_cubed_term_is_seventh :
  T r = T 6 ∧ 8 - 5/6 * r = 3 ∧ r + 1 = 7 := by
  sorry

end x_cubed_term_is_seventh_l1093_109312


namespace employment_percentage_l1093_109321

theorem employment_percentage (total_population : ℝ) 
  (employed_males_percentage : ℝ) (employed_females_percentage : ℝ) :
  employed_males_percentage = 48 →
  employed_females_percentage = 20 →
  (employed_males_percentage / (100 - employed_females_percentage)) * 100 = 60 :=
by
  sorry

end employment_percentage_l1093_109321


namespace problem_statement_l1093_109320

theorem problem_statement (x y : ℝ) (h1 : x + y = 2) (h2 : x * y = 4) :
  x^2 * y^3 + y^2 * x^3 = 0 := by
  sorry

end problem_statement_l1093_109320
