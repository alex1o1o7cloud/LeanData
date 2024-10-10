import Mathlib

namespace largest_integer_problem_l3990_399031

theorem largest_integer_problem (a b c d : ℤ) : 
  a < b ∧ b < c ∧ c < d →  -- Four different integers
  (a + b + c + d) / 4 = 76 →  -- Average is 76
  a ≥ 37 →  -- Smallest integer is at least 37
  d ≤ 190 :=  -- Largest integer is at most 190
by sorry

end largest_integer_problem_l3990_399031


namespace axis_of_symmetry_is_neg_two_l3990_399025

/-- A quadratic function with given coordinate values -/
structure QuadraticFunction where
  f : ℝ → ℝ
  coords : List (ℝ × ℝ)
  is_quadratic : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

/-- The axis of symmetry of a quadratic function -/
def axis_of_symmetry (qf : QuadraticFunction) : ℝ := sorry

/-- The given quadratic function from the problem -/
def given_function : QuadraticFunction where
  f := sorry
  coords := [(-3, -3), (-2, -2), (-1, -3), (0, -6), (1, -11)]
  is_quadratic := sorry

/-- Theorem stating that the axis of symmetry of the given function is -2 -/
theorem axis_of_symmetry_is_neg_two :
  axis_of_symmetry given_function = -2 := by sorry

end axis_of_symmetry_is_neg_two_l3990_399025


namespace largest_number_from_digits_l3990_399024

def digits : List ℕ := [1, 7, 0]

def formNumber (d : List ℕ) : ℕ :=
  d.foldl (fun acc x => acc * 10 + x) 0

def isPermutation (l1 l2 : List ℕ) : Prop :=
  l1.length = l2.length ∧ ∀ x, x ∈ l1 ↔ x ∈ l2

theorem largest_number_from_digits : 
  ∀ p : List ℕ, isPermutation digits p → formNumber p ≤ 710 :=
sorry

end largest_number_from_digits_l3990_399024


namespace max_value_x_y3_z4_l3990_399015

theorem max_value_x_y3_z4 (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 1) :
  x + y^3 + z^4 ≤ 1 ∧ ∃ (x' y' z' : ℝ), x' ≥ 0 ∧ y' ≥ 0 ∧ z' ≥ 0 ∧ x' + y' + z' = 1 ∧ x' + y'^3 + z'^4 = 1 :=
by sorry

end max_value_x_y3_z4_l3990_399015


namespace tara_quarters_l3990_399021

theorem tara_quarters : ∃! q : ℕ, 
  8 < q ∧ q < 80 ∧ 
  q % 4 = 2 ∧
  q % 6 = 2 ∧
  q % 8 = 2 ∧
  q = 26 := by sorry

end tara_quarters_l3990_399021


namespace E27D6_divisibility_l3990_399034

/-- A number in the form E27D6 where E and D are single digits -/
def E27D6 (E D : ℕ) : ℕ := E * 10000 + 27000 + D * 10 + 6

/-- Predicate to check if a number is a single digit -/
def is_single_digit (n : ℕ) : Prop := n < 10

theorem E27D6_divisibility (E D : ℕ) :
  is_single_digit E →
  is_single_digit D →
  E27D6 E D % 8 = 0 →
  ∃ (sum : ℕ), sum = D + E ∧ 1 ≤ sum ∧ sum ≤ 10 :=
sorry

end E27D6_divisibility_l3990_399034


namespace remainder_73_power_73_plus_73_mod_137_l3990_399049

theorem remainder_73_power_73_plus_73_mod_137 :
  ∃ k : ℤ, 73^73 + 73 = 137 * k + 9 :=
by
  sorry

end remainder_73_power_73_plus_73_mod_137_l3990_399049


namespace bowling_ball_weight_l3990_399013

theorem bowling_ball_weight :
  ∀ (bowling_ball_weight canoe_weight : ℝ),
    (9 * bowling_ball_weight = 5 * canoe_weight) →
    (4 * canoe_weight = 120) →
    bowling_ball_weight = 50 / 3 := by
  sorry

end bowling_ball_weight_l3990_399013


namespace no_real_roots_l3990_399081

theorem no_real_roots : ∀ x : ℝ, 4 * x^2 - 5 * x + 2 ≠ 0 := by
  sorry

end no_real_roots_l3990_399081


namespace mashas_juice_theorem_l3990_399088

/-- Represents Masha's juice drinking process over 3 days -/
def mashas_juice_process (x : ℝ) : Prop :=
  let day1_juice := x - 1
  let day2_juice := (day1_juice^2) / x
  let day3_juice := (day2_juice^2) / x
  let final_juice := (day3_juice^2) / x
  let final_water := x - final_juice
  (final_water = final_juice + 1.5) ∧ (x > 1)

/-- The theorem stating the result of Masha's juice drinking process -/
theorem mashas_juice_theorem :
  ∀ x : ℝ, mashas_juice_process x ↔ (x = 2 ∧ (2 - ((2 - 1)^3) / 2^2 = 1.75)) :=
by sorry

end mashas_juice_theorem_l3990_399088


namespace parabola_line_intersection_l3990_399004

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 2px -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Represents a line with slope m -/
structure Line where
  m : ℝ

theorem parabola_line_intersection
  (para : Parabola)
  (line : Line)
  (A B : Point)
  (h_line_slope : line.m = 2 * Real.sqrt 2)
  (h_on_parabola_A : A.y ^ 2 = 2 * para.p * A.x)
  (h_on_parabola_B : B.y ^ 2 = 2 * para.p * B.x)
  (h_on_line_A : A.y = line.m * (A.x - para.p / 2))
  (h_on_line_B : B.y = line.m * (B.x - para.p / 2))
  (h_x_order : A.x < B.x)
  (h_distance : Real.sqrt ((B.x - A.x) ^ 2 + (B.y - A.y) ^ 2) = 9) :
  (∃ (C : Point),
    C.y ^ 2 = 8 * C.x ∧
    (C.x = A.x + 0 * (B.x - A.x) ∧ C.y = A.y + 0 * (B.y - A.y) ∨
     C.x = A.x + 2 * (B.x - A.x) ∧ C.y = A.y + 2 * (B.y - A.y))) :=
by sorry

end parabola_line_intersection_l3990_399004


namespace arccos_one_half_equals_pi_third_l3990_399078

theorem arccos_one_half_equals_pi_third : Real.arccos (1/2) = π/3 := by
  sorry

end arccos_one_half_equals_pi_third_l3990_399078


namespace rented_cars_at_3600_optimal_rent_max_monthly_revenue_l3990_399092

/-- Represents the rental company's car fleet and pricing model. -/
structure RentalCompany where
  totalCars : ℕ := 100
  initialRent : ℕ := 3000
  rentIncrease : ℕ := 50
  maintenanceCostRented : ℕ := 150
  maintenanceCostUnrented : ℕ := 50

/-- Calculates the number of rented cars given a specific rent. -/
def rentedCars (company : RentalCompany) (rent : ℕ) : ℕ :=
  company.totalCars - (rent - company.initialRent) / company.rentIncrease

/-- Calculates the monthly revenue for the rental company. -/
def monthlyRevenue (company : RentalCompany) (rent : ℕ) : ℕ :=
  let rented := rentedCars company rent
  rented * (rent - company.maintenanceCostRented) -
    (company.totalCars - rented) * company.maintenanceCostUnrented

/-- Theorem stating the correct number of rented cars at 3600 yuan rent. -/
theorem rented_cars_at_3600 (company : RentalCompany) :
    rentedCars company 3600 = 88 := by sorry

/-- Theorem stating the optimal rent that maximizes revenue. -/
theorem optimal_rent (company : RentalCompany) :
    ∃ (optimalRent : ℕ), optimalRent = 4050 ∧
    ∀ (rent : ℕ), monthlyRevenue company rent ≤ monthlyRevenue company optimalRent := by sorry

/-- Theorem stating the maximum monthly revenue. -/
theorem max_monthly_revenue (company : RentalCompany) :
    ∃ (maxRevenue : ℕ), maxRevenue = 307050 ∧
    ∀ (rent : ℕ), monthlyRevenue company rent ≤ maxRevenue := by sorry

end rented_cars_at_3600_optimal_rent_max_monthly_revenue_l3990_399092


namespace sum_of_roots_equals_four_l3990_399017

theorem sum_of_roots_equals_four :
  let f (x : ℝ) := (x^3 - 2*x^2 - 8*x) / (x + 2)
  (∃ a b : ℝ, (f a = 5 ∧ f b = 5 ∧ a ≠ b) ∧ a + b = 4) :=
by sorry

end sum_of_roots_equals_four_l3990_399017


namespace linglings_spending_l3990_399057

theorem linglings_spending (x : ℝ) 
  (h1 : 720 = (1 - 1/3) * ((1 - 2/5) * x + 240)) : 
  ∃ (spent : ℝ), spent = 2/5 * x ∧ 
  720 = (1 - 1/3) * ((x - spent) + 240) := by
  sorry

end linglings_spending_l3990_399057


namespace sum_x_y_equals_five_l3990_399058

theorem sum_x_y_equals_five (x y : ℝ) 
  (eq1 : x + 3*y = 12) 
  (eq2 : 3*x + y = 8) : 
  x + y = 5 := by
sorry

end sum_x_y_equals_five_l3990_399058


namespace negation_of_universal_conditional_l3990_399001

theorem negation_of_universal_conditional (P : ℝ → Prop) :
  (¬∀ x : ℝ, x ≥ 2 → P x) ↔ (∃ x : ℝ, x < 2 ∧ ¬P x) :=
by sorry

end negation_of_universal_conditional_l3990_399001


namespace investment_value_l3990_399083

theorem investment_value (x : ℝ) : 
  (0.07 * 500 + 0.23 * x = 0.19 * (500 + x)) → x = 1500 := by
  sorry

end investment_value_l3990_399083


namespace initial_tax_rate_calculation_l3990_399035

/-- Proves that given an annual income of $36,000, if lowering the tax rate to 32% 
    results in a savings of $5,040, then the initial tax rate was 46%. -/
theorem initial_tax_rate_calculation 
  (annual_income : ℝ) 
  (new_tax_rate : ℝ) 
  (savings : ℝ) 
  (h1 : annual_income = 36000)
  (h2 : new_tax_rate = 32)
  (h3 : savings = 5040) :
  ∃ (initial_tax_rate : ℝ), 
    initial_tax_rate = 46 ∧ 
    (initial_tax_rate / 100 * annual_income) - (new_tax_rate / 100 * annual_income) = savings :=
by sorry

end initial_tax_rate_calculation_l3990_399035


namespace investment_value_after_eight_years_l3990_399008

/-- Calculates the final value of an investment under simple interest -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Proves that given the conditions, the investment value after 8 years is $660 -/
theorem investment_value_after_eight_years 
  (P : ℝ) -- Initial investment (principal)
  (h1 : simple_interest P 0.04 3 = 560) -- Value after 3 years
  : simple_interest P 0.04 8 = 660 := by
  sorry

#check investment_value_after_eight_years

end investment_value_after_eight_years_l3990_399008


namespace intersection_A_B_intersection_A_C_condition_l3990_399044

-- Define the sets A, B, and C
def A : Set ℝ := {y | ∃ x, 1 ≤ x ∧ x ≤ 2 ∧ y = 2^x}
def B : Set ℝ := {x | 0 < Real.log x ∧ Real.log x < 1}
def C (t : ℝ) : Set ℝ := {x | t + 1 < x ∧ x < 2*t}

-- Theorem 1: Intersection of A and B
theorem intersection_A_B : A ∩ B = {x | 2 ≤ x ∧ x < Real.exp 1} := by sorry

-- Theorem 2: Condition for t when A ∩ C = C
theorem intersection_A_C_condition (t : ℝ) : A ∩ C t = C t → t ≤ 2 := by sorry

end intersection_A_B_intersection_A_C_condition_l3990_399044


namespace arithmetic_mean_problem_l3990_399068

theorem arithmetic_mean_problem (a₁ a₂ a₃ a₄ a₅ a₆ A : ℝ) 
  (h1 : (a₁ + a₂ + a₃ + a₄ + a₅ + a₆) / 6 = A)
  (h2 : (a₁ + a₂ + a₃ + a₄) / 4 = A + 10)
  (h3 : (a₃ + a₄ + a₅ + a₆) / 4 = A - 7) :
  (a₁ + a₂ + a₅ + a₆) / 4 = A - 3 := by
  sorry

end arithmetic_mean_problem_l3990_399068


namespace triangle_radii_relation_l3990_399030

/-- Given a triangle with side lengths a, b, c, semi-perimeter p, area S, and circumradius R,
    prove the relationship between the inradius τ, exradii τa, τb, τc, and other triangle properties. -/
theorem triangle_radii_relation
  (a b c p S R τ τa τb τc : ℝ)
  (h1 : S = τ * p)
  (h2 : S = τa * (p - a))
  (h3 : S = τb * (p - b))
  (h4 : S = τc * (p - c))
  (h5 : a * b * c / S = 4 * R) :
  1 / τ^3 - 1 / τa^3 - 1 / τb^3 - 1 / τc^3 = 12 * R / S^2 := by
  sorry


end triangle_radii_relation_l3990_399030


namespace zero_point_existence_not_necessary_l3990_399073

def f (a : ℝ) (x : ℝ) : ℝ := a * x + 3

theorem zero_point_existence (a : ℝ) (h : a > 2) :
  ∃ x ∈ Set.Icc (-1 : ℝ) 0, f a x = 0 :=
sorry

theorem not_necessary (a : ℝ) :
  (∃ x ∈ Set.Icc (-1 : ℝ) 0, f a x = 0) → a > 2 → False :=
sorry

end zero_point_existence_not_necessary_l3990_399073


namespace sin_315_degrees_l3990_399020

theorem sin_315_degrees : Real.sin (315 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end sin_315_degrees_l3990_399020


namespace difference_of_squares_l3990_399096

theorem difference_of_squares : (535 : ℕ)^2 - (465 : ℕ)^2 = 70000 := by
  sorry

end difference_of_squares_l3990_399096


namespace candy_bar_cost_l3990_399054

/-- Given that Dan had $4 at the start and $3 left after buying a candy bar,
    prove that the candy bar cost $1. -/
theorem candy_bar_cost (initial_amount : ℕ) (remaining_amount : ℕ) :
  initial_amount = 4 →
  remaining_amount = 3 →
  initial_amount - remaining_amount = 1 :=
by sorry

end candy_bar_cost_l3990_399054


namespace sheelas_monthly_income_l3990_399053

/-- Given that Sheela's deposit is 20% of her monthly income, 
    prove that her monthly income is Rs. 25000 -/
theorem sheelas_monthly_income 
  (deposit : ℝ) 
  (deposit_percentage : ℝ) 
  (h1 : deposit = 5000)
  (h2 : deposit_percentage = 0.20)
  (h3 : deposit = deposit_percentage * sheelas_income) : 
  sheelas_income = 25000 :=
by
  sorry

#check sheelas_monthly_income

end sheelas_monthly_income_l3990_399053


namespace remainder_r_15_minus_1_l3990_399055

theorem remainder_r_15_minus_1 (r : ℤ) : (r^15 - 1) % (r + 1) = -2 := by
  sorry

end remainder_r_15_minus_1_l3990_399055


namespace min_reciprocal_sum_l3990_399079

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 10) :
  (1 / x + 1 / y) ≥ 2 / 5 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ x + y = 10 ∧ 1 / x + 1 / y = 2 / 5 :=
sorry

end min_reciprocal_sum_l3990_399079


namespace trigonometric_identities_l3990_399067

theorem trigonometric_identities (α : Real) (h : Real.tan α = 2) :
  (Real.sin α - 3 * Real.cos α) / (Real.sin α + Real.cos α) = -1/3 ∧
  2 * Real.sin α ^ 2 - Real.sin α * Real.cos α + Real.cos α ^ 2 = 7/5 := by
  sorry

end trigonometric_identities_l3990_399067


namespace find_x_l3990_399077

theorem find_x : ∃ x : ℝ,
  (24 + 35 + 58) / 3 = ((19 + 51 + x) / 3) + 6 → x = 29 :=
by sorry

end find_x_l3990_399077


namespace town_budget_theorem_l3990_399023

/-- Represents the town's budget allocation problem -/
def TownBudget (total : ℝ) (policing_fraction : ℝ) (education : ℝ) : Prop :=
  let policing := total * policing_fraction
  let remaining := total - policing - education
  remaining = 4

/-- The theorem statement for the town's budget allocation problem -/
theorem town_budget_theorem :
  TownBudget 32 0.5 12 := by
  sorry

end town_budget_theorem_l3990_399023


namespace system_solutions_l3990_399050

def system (x y z : ℚ) : Prop :=
  x^2 + 2*y*z = x ∧ y^2 + 2*z*x = y ∧ z^2 + 2*x*y = z

def solutions : List (ℚ × ℚ × ℚ) :=
  [(0, 0, 0), (1/3, 1/3, 1/3), (1, 0, 0), (0, 1, 0), (0, 0, 1),
   (2/3, -1/3, -1/3), (-1/3, 2/3, -1/3), (-1/3, -1/3, 2/3)]

theorem system_solutions :
  ∀ x y z : ℚ, system x y z ↔ (x, y, z) ∈ solutions := by sorry

end system_solutions_l3990_399050


namespace jebb_take_home_pay_l3990_399075

/-- Calculates the take-home pay after tax deduction -/
def takeHomePay (totalPay : ℝ) (taxRate : ℝ) : ℝ :=
  totalPay * (1 - taxRate)

/-- Proves that Jebb's take-home pay is $585 -/
theorem jebb_take_home_pay :
  let totalPay : ℝ := 650
  let taxRate : ℝ := 0.1
  takeHomePay totalPay taxRate = 585 := by
  sorry

end jebb_take_home_pay_l3990_399075


namespace michaels_blocks_l3990_399006

/-- Given that Michael has some blocks stored in boxes, prove that the total number of blocks is 16 -/
theorem michaels_blocks (num_boxes : ℕ) (blocks_per_box : ℕ) (h1 : num_boxes = 8) (h2 : blocks_per_box = 2) :
  num_boxes * blocks_per_box = 16 := by
  sorry

end michaels_blocks_l3990_399006


namespace binomial_10_choose_3_l3990_399059

theorem binomial_10_choose_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_choose_3_l3990_399059


namespace bench_seating_theorem_l3990_399048

/-- The number of ways to arrange people on a bench with empty seats -/
def seating_arrangements (total_seats : ℕ) (people : ℕ) (adjacent_empty : ℕ) : ℕ :=
  (people.factorial) * (people + 1).factorial / 2

/-- Theorem: There are 480 ways to arrange 4 people on a bench with 7 seats,
    such that exactly 2 of the 3 empty seats are adjacent -/
theorem bench_seating_theorem :
  seating_arrangements 7 4 2 = 480 := by
  sorry

#eval seating_arrangements 7 4 2

end bench_seating_theorem_l3990_399048


namespace johns_leftover_earnings_l3990_399051

/-- Proves that given John spent 40% of his earnings on rent and 30% less than that on a dishwasher, he had 32% of his earnings left over. -/
theorem johns_leftover_earnings : 
  ∀ (total_earnings : ℝ) (rent_percent : ℝ) (dishwasher_percent : ℝ),
    rent_percent = 40 →
    dishwasher_percent = rent_percent - (0.3 * rent_percent) →
    100 - (rent_percent + dishwasher_percent) = 32 := by
  sorry

end johns_leftover_earnings_l3990_399051


namespace power_mod_thousand_l3990_399070

theorem power_mod_thousand : 7^27 % 1000 = 543 := by sorry

end power_mod_thousand_l3990_399070


namespace problem_1_l3990_399072

theorem problem_1 : 40 + (1/6 - 2/3 + 3/4) * 12 = 43 := by
  sorry

end problem_1_l3990_399072


namespace wooden_strip_sawing_time_l3990_399056

theorem wooden_strip_sawing_time 
  (initial_length : ℝ) 
  (initial_sections : ℕ) 
  (initial_time : ℝ) 
  (final_sections : ℕ) 
  (h1 : initial_length = 12) 
  (h2 : initial_sections = 4) 
  (h3 : initial_time = 12) 
  (h4 : final_sections = 8) : 
  (initial_time / (initial_sections - 1)) * (final_sections - 1) = 28 := by
sorry

end wooden_strip_sawing_time_l3990_399056


namespace car_ac_price_difference_l3990_399039

/-- Given that the price of a car and AC are in the ratio 3:2, and the AC costs $1500,
    prove that the car costs $750 more than the AC. -/
theorem car_ac_price_difference :
  ∀ (car_price ac_price : ℕ),
    car_price / ac_price = 3 / 2 →
    ac_price = 1500 →
    car_price - ac_price = 750 :=
by
  sorry

end car_ac_price_difference_l3990_399039


namespace major_premise_incorrect_l3990_399010

-- Define a differentiable function
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- Define a point x₀
variable (x₀ : ℝ)

-- State that f'(x₀) = 0
variable (h : deriv f x₀ = 0)

-- Define what it means for x₀ to be an extremum point
def is_extremum_point (f : ℝ → ℝ) (x₀ : ℝ) :=
  ∀ x, f x ≤ f x₀ ∨ ∀ x, f x ≥ f x₀

-- Theorem stating that the major premise is false
theorem major_premise_incorrect :
  ¬(∀ f : ℝ → ℝ, ∀ x₀ : ℝ, Differentiable ℝ f → deriv f x₀ = 0 → is_extremum_point f x₀) :=
sorry

end major_premise_incorrect_l3990_399010


namespace jacob_walking_distance_l3990_399066

/-- Calculates the distance traveled given a constant rate and time --/
def distance (rate : ℝ) (time : ℝ) : ℝ := rate * time

/-- Theorem: Jacob walks 8 miles in 2 hours at a rate of 4 miles per hour --/
theorem jacob_walking_distance :
  let rate : ℝ := 4
  let time : ℝ := 2
  distance rate time = 8 := by
  sorry

end jacob_walking_distance_l3990_399066


namespace circle_diameter_from_area_l3990_399094

theorem circle_diameter_from_area :
  ∀ (r d : ℝ),
  r > 0 →
  d = 2 * r →
  π * r^2 = 225 * π →
  d = 30 :=
by
  sorry

end circle_diameter_from_area_l3990_399094


namespace expression_simplification_l3990_399009

theorem expression_simplification (y : ℝ) : 
  4 * y - 2 * y^2 + 3 - (8 - 4 * y + y^2) = 8 * y - 3 * y^2 - 5 := by
  sorry

end expression_simplification_l3990_399009


namespace frosting_per_cake_l3990_399018

/-- The number of cans of frosting needed to frost a single cake -/
def cans_per_cake (cakes_per_day : ℕ) (days : ℕ) (cakes_eaten : ℕ) (total_cans : ℕ) : ℚ :=
  total_cans / (cakes_per_day * days - cakes_eaten)

/-- Theorem stating that given Sara's baking schedule and frosting needs, 
    it takes 2 cans of frosting to frost a single cake -/
theorem frosting_per_cake : 
  cans_per_cake 10 5 12 76 = 2 := by
  sorry

end frosting_per_cake_l3990_399018


namespace no_prime_sum_10003_l3990_399069

theorem no_prime_sum_10003 : ¬∃ (p q : ℕ), Prime p ∧ Prime q ∧ p + q = 10003 := by
  sorry

end no_prime_sum_10003_l3990_399069


namespace chords_from_eight_points_l3990_399074

/-- The number of chords that can be drawn from n points on a circle's circumference -/
def num_chords (n : ℕ) : ℕ := n.choose 2

/-- Theorem: The number of chords from 8 points on a circle's circumference is 28 -/
theorem chords_from_eight_points : num_chords 8 = 28 := by
  sorry

end chords_from_eight_points_l3990_399074


namespace first_day_of_month_is_sunday_l3990_399090

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define a function to get the day of the week for a given day of the month
def dayOfWeek (dayOfMonth : Nat) : DayOfWeek := sorry

-- Theorem statement
theorem first_day_of_month_is_sunday 
  (h : dayOfWeek 18 = DayOfWeek.Wednesday) : 
  dayOfWeek 1 = DayOfWeek.Sunday := by
  sorry

end first_day_of_month_is_sunday_l3990_399090


namespace apple_ratio_is_half_l3990_399029

/-- The number of apples Anna ate on Tuesday -/
def tuesday_apples : ℕ := 4

/-- The number of apples Anna ate on Wednesday -/
def wednesday_apples : ℕ := 2 * tuesday_apples

/-- The total number of apples Anna ate over the three days -/
def total_apples : ℕ := 14

/-- The number of apples Anna ate on Thursday -/
def thursday_apples : ℕ := total_apples - tuesday_apples - wednesday_apples

/-- The ratio of apples eaten on Thursday to Tuesday -/
def thursday_to_tuesday_ratio : ℚ := thursday_apples / tuesday_apples

theorem apple_ratio_is_half : thursday_to_tuesday_ratio = 1/2 := by
  sorry

end apple_ratio_is_half_l3990_399029


namespace inequality_not_true_l3990_399040

theorem inequality_not_true (x y : ℝ) (h : x > y) : ¬(-3*x + 6 > -3*y + 6) := by
  sorry

end inequality_not_true_l3990_399040


namespace current_speed_calculation_l3990_399027

-- Define the given conditions
def downstream_distance : ℝ := 96
def downstream_time : ℝ := 8
def upstream_distance : ℝ := 8
def upstream_time : ℝ := 2

-- Define the speed of the current
def current_speed : ℝ := 4

-- Theorem statement
theorem current_speed_calculation :
  let boat_speed := (downstream_distance / downstream_time + upstream_distance / upstream_time) / 2
  current_speed = boat_speed - upstream_distance / upstream_time :=
by sorry

end current_speed_calculation_l3990_399027


namespace num_triangles_in_circle_l3990_399082

/-- The number of points on the circle -/
def n : ℕ := 9

/-- The number of chords -/
def num_chords : ℕ := n.choose 2

/-- The number of intersection points inside the circle -/
def num_intersections : ℕ := n.choose 4

/-- Theorem: The number of triangles formed by intersection points of chords inside a circle -/
theorem num_triangles_in_circle (n : ℕ) (h : n = 9) : 
  (num_intersections.choose 3) = 315500 :=
sorry

end num_triangles_in_circle_l3990_399082


namespace quadratic_coefficient_of_equation_l3990_399033

theorem quadratic_coefficient_of_equation (x : ℝ) : 
  (2*x + 1) * (3*x - 2) = x^2 + 2 → 
  ∃ a b c : ℝ, a = 5 ∧ a*x^2 + b*x + c = 0 :=
by sorry

end quadratic_coefficient_of_equation_l3990_399033


namespace cube_root_function_l3990_399065

theorem cube_root_function (k : ℝ) :
  (∃ y : ℝ, y = k * 64^(1/3) ∧ y = 4 * Real.sqrt 3) →
  k * 8^(1/3) = 2 * Real.sqrt 3 := by
  sorry

end cube_root_function_l3990_399065


namespace figure_100_squares_l3990_399032

def figure_squares (n : ℕ) : ℕ :=
  3 * n^2 + 3 * n + 1

theorem figure_100_squares :
  figure_squares 0 = 1 ∧
  figure_squares 1 = 7 ∧
  figure_squares 2 = 19 ∧
  figure_squares 3 = 37 →
  figure_squares 100 = 30301 := by
sorry

end figure_100_squares_l3990_399032


namespace expected_draws_eq_sixteen_thirds_l3990_399016

/-- The number of red balls in the bag -/
def num_red : ℕ := 2

/-- The number of black balls in the bag -/
def num_black : ℕ := 5

/-- The total number of balls in the bag -/
def total_balls : ℕ := num_red + num_black

/-- The set of possible numbers of draws -/
def possible_draws : Finset ℕ := Finset.range (total_balls + 1) \ Finset.range num_red

/-- The probability of drawing a specific number of balls -/
noncomputable def prob_draw (n : ℕ) : ℚ :=
  if n ∈ possible_draws then
    -- This is a placeholder for the actual probability calculation
    1 / possible_draws.card
  else
    0

/-- The expected number of draws -/
noncomputable def expected_draws : ℚ :=
  Finset.sum possible_draws (λ n => n * prob_draw n)

theorem expected_draws_eq_sixteen_thirds :
  expected_draws = 16 / 3 := by sorry

end expected_draws_eq_sixteen_thirds_l3990_399016


namespace count_arrangements_l3990_399085

/-- Represents the number of students --/
def totalStudents : ℕ := 6

/-- Represents the number of boys --/
def numBoys : ℕ := 3

/-- Represents the number of girls --/
def numGirls : ℕ := 3

/-- Represents whether girls are allowed at the ends --/
def girlsAtEnds : Prop := False

/-- Represents whether girls A and B can stand next to girl C --/
def girlsABNextToC : Prop := False

/-- The number of valid arrangements --/
def validArrangements : ℕ := 72

/-- Theorem stating the number of valid arrangements --/
theorem count_arrangements :
  (totalStudents = numBoys + numGirls) →
  (numBoys = 3) →
  (numGirls = 3) →
  girlsAtEnds = False →
  girlsABNextToC = False →
  validArrangements = 72 := by
  sorry

end count_arrangements_l3990_399085


namespace field_length_width_ratio_l3990_399014

/-- Proves that for a rectangular field with a square pond, given specific conditions, the ratio of length to width is 2:1 -/
theorem field_length_width_ratio (field_length field_width pond_side : ℝ) : 
  field_length = 28 →
  pond_side = 7 →
  field_length * field_width = 8 * pond_side * pond_side →
  field_length / field_width = 2 := by
  sorry

#check field_length_width_ratio

end field_length_width_ratio_l3990_399014


namespace area_outside_rectangle_in_square_l3990_399047

/-- The area of the region outside a rectangle contained within a square -/
theorem area_outside_rectangle_in_square (square_side : ℝ) (rect_length rect_width : ℝ) : 
  square_side = 8 ∧ rect_length = 4 ∧ rect_width = 2 →
  square_side^2 - rect_length * rect_width = 56 := by
sorry


end area_outside_rectangle_in_square_l3990_399047


namespace f_monotone_increasing_f_extreme_values_l3990_399045

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 2*x^2 - 4*x + 2

-- Theorem for monotonically increasing intervals
theorem f_monotone_increasing :
  (∀ x y, x < y ∧ x < -2/3 → f x < f y) ∧
  (∀ x y, 2 < x ∧ x < y → f x < f y) :=
sorry

-- Theorem for extreme values on [-1, 3]
theorem f_extreme_values :
  (∀ x, -1 ≤ x ∧ x ≤ 3 → -6 ≤ f x ∧ f x ≤ 94/27) ∧
  (∃ x, -1 ≤ x ∧ x ≤ 3 ∧ f x = -6) ∧
  (∃ x, -1 ≤ x ∧ x ≤ 3 ∧ f x = 94/27) :=
sorry

end f_monotone_increasing_f_extreme_values_l3990_399045


namespace negation_equivalence_l3990_399003

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - x + 2 > 0) ↔ (∀ x : ℝ, x^2 - x + 2 ≤ 0) := by
  sorry

end negation_equivalence_l3990_399003


namespace framed_painting_ratio_l3990_399080

theorem framed_painting_ratio : 
  ∀ (x : ℝ),
  x > 0 →
  (30 + 2*x) * (20 + 4*x) = 1500 →
  (min (30 + 2*x) (20 + 4*x)) / (max (30 + 2*x) (20 + 4*x)) = 4/5 := by
  sorry

end framed_painting_ratio_l3990_399080


namespace set_equality_l3990_399042

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}
def M : Set Nat := {3, 4, 5}
def N : Set Nat := {1, 3, 6}

theorem set_equality : (U \ M) ∩ (U \ N) = {2, 7} := by sorry

end set_equality_l3990_399042


namespace exponent_problem_l3990_399084

theorem exponent_problem (a : ℝ) (x y : ℝ) (h1 : a^x = 2) (h2 : a^y = 3) : a^(2*x + y) = 12 := by
  sorry

end exponent_problem_l3990_399084


namespace max_garden_area_l3990_399000

/-- Represents the dimensions of a rectangular garden. -/
structure GardenDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular garden given its dimensions. -/
def area (d : GardenDimensions) : ℝ := d.length * d.width

/-- Calculates the perimeter of a rectangular garden given its dimensions. -/
def perimeter (d : GardenDimensions) : ℝ := 2 * (d.length + d.width)

/-- Theorem: The maximum area of a rectangular garden with 320 feet of fencing
    and length no less than 100 feet is 6000 square feet, achieved when
    the length is 100 feet and the width is 60 feet. -/
theorem max_garden_area :
  ∃ (d : GardenDimensions),
    perimeter d = 320 ∧
    d.length ≥ 100 ∧
    area d = 6000 ∧
    (∀ (d' : GardenDimensions), perimeter d' = 320 ∧ d'.length ≥ 100 → area d' ≤ area d) :=
by sorry

end max_garden_area_l3990_399000


namespace students_in_band_or_sports_l3990_399043

theorem students_in_band_or_sports 
  (total : ℕ) 
  (band : ℕ) 
  (sports : ℕ) 
  (both : ℕ) 
  (h_total : total = 320)
  (h_band : band = 85)
  (h_sports : sports = 200)
  (h_both : both = 60) :
  band + sports - both = 225 := by
  sorry

end students_in_band_or_sports_l3990_399043


namespace sequence_correct_l3990_399038

def sequence_formula (n : ℕ) : ℤ := (-1)^(n+1) * 2^(n-1)

theorem sequence_correct : ∀ n : ℕ, n ≥ 1 ∧ n ≤ 6 →
  match n with
  | 1 => sequence_formula n = 1
  | 2 => sequence_formula n = -2
  | 3 => sequence_formula n = 4
  | 4 => sequence_formula n = -8
  | 5 => sequence_formula n = 16
  | 6 => sequence_formula n = -32
  | _ => True
  :=
by
  sorry

end sequence_correct_l3990_399038


namespace solution_to_equation_l3990_399099

theorem solution_to_equation (z : ℝ) : 
  (z^2 - 5*z + 6)/(z-2) + (5*z^2 + 11*z - 32)/(5*z - 16) = 1 ↔ z = 1 :=
by sorry

end solution_to_equation_l3990_399099


namespace B_spend_percent_is_85_percent_l3990_399026

def total_salary : ℝ := 7000
def A_salary : ℝ := 5250
def A_spend_percent : ℝ := 0.95

def B_salary : ℝ := total_salary - A_salary
def A_savings : ℝ := A_salary * (1 - A_spend_percent)

theorem B_spend_percent_is_85_percent :
  ∃ (B_spend_percent : ℝ),
    B_spend_percent = 0.85 ∧
    A_savings = B_salary * (1 - B_spend_percent) := by
  sorry

end B_spend_percent_is_85_percent_l3990_399026


namespace trigonometric_identities_l3990_399046

theorem trigonometric_identities (α : Real) (h : Real.tan α = 3) :
  (4 * Real.sin α - Real.cos α) / (3 * Real.sin α + 5 * Real.cos α) = 11 / 14 ∧
  Real.sin α * Real.cos α = 3 / 10 := by
  sorry

end trigonometric_identities_l3990_399046


namespace borrowed_sheets_theorem_l3990_399098

/-- Represents a notebook with double-sided pages -/
structure Notebook where
  total_sheets : ℕ
  total_pages : ℕ
  pages_per_sheet : ℕ
  h_pages_per_sheet : pages_per_sheet = 2

/-- Calculates the average of remaining page numbers after borrowing sheets -/
def average_remaining_pages (nb : Notebook) (borrowed_sheets : ℕ) : ℚ :=
  let remaining_pages := nb.total_pages - borrowed_sheets * nb.pages_per_sheet
  let sum_remaining := (nb.total_pages * (nb.total_pages + 1) / 2) -
    (borrowed_sheets * nb.pages_per_sheet * (borrowed_sheets * nb.pages_per_sheet + 1) / 2)
  sum_remaining / remaining_pages

/-- Theorem stating that borrowing 12 sheets results in an average of 23 for remaining pages -/
theorem borrowed_sheets_theorem (nb : Notebook)
    (h_total_sheets : nb.total_sheets = 32)
    (h_total_pages : nb.total_pages = 64)
    (borrowed_sheets : ℕ)
    (h_borrowed : borrowed_sheets = 12) :
    average_remaining_pages nb borrowed_sheets = 23 := by
  sorry

end borrowed_sheets_theorem_l3990_399098


namespace fourth_root_squared_l3990_399061

theorem fourth_root_squared (y : ℝ) : (y^(1/4))^2 = 81 → y = 81 := by sorry

end fourth_root_squared_l3990_399061


namespace plane_equation_correct_l3990_399060

/-- The equation of a plane given the foot of the perpendicular from the origin -/
def plane_equation (foot : ℝ × ℝ × ℝ) : ℤ × ℤ × ℤ × ℤ := sorry

/-- Check if the given coefficients satisfy the required conditions -/
def valid_coefficients (coeffs : ℤ × ℤ × ℤ × ℤ) : Prop :=
  let (A, B, C, D) := coeffs
  A > 0 ∧ Nat.gcd (Nat.gcd (Int.natAbs A) (Int.natAbs B)) (Nat.gcd (Int.natAbs C) (Int.natAbs D)) = 1

theorem plane_equation_correct (foot : ℝ × ℝ × ℝ) :
  foot = (10, -2, 1) →
  plane_equation foot = (10, -2, 1, -105) ∧
  valid_coefficients (plane_equation foot) := by
  sorry

end plane_equation_correct_l3990_399060


namespace dinner_cost_calculation_l3990_399019

/-- The total cost of dinner for Bret and his co-workers -/
def dinner_cost (num_people : ℕ) (main_meal_cost : ℚ) (num_appetizers : ℕ) (appetizer_cost : ℚ) (tip_percentage : ℚ) (rush_order_fee : ℚ) : ℚ :=
  let subtotal := num_people * main_meal_cost + num_appetizers * appetizer_cost
  let tip := tip_percentage * subtotal
  subtotal + tip + rush_order_fee

/-- Theorem stating the total cost of dinner -/
theorem dinner_cost_calculation :
  dinner_cost 4 12 2 6 (1/5) 5 = 77 :=
by sorry

end dinner_cost_calculation_l3990_399019


namespace problem_solution_l3990_399064

theorem problem_solution :
  (∀ x : ℝ, x^2 + x + 2 ≥ 0) ∧
  (∀ x y : ℝ, x * y = ((x + y) / 2)^2 ↔ x = y) ∧
  (∃ p q : Prop, ¬(p ∧ q) ∧ ¬(¬p ∧ ¬q)) ∧
  (∀ A B C : ℝ, ∀ sinA sinB : ℝ, 
    sinA = Real.sin A ∧ sinB = Real.sin B →
    sinA > sinB → A > B) :=
by sorry

end problem_solution_l3990_399064


namespace e1_e2_form_basis_l3990_399028

def e1 : ℝ × ℝ := (-1, 2)
def e2 : ℝ × ℝ := (5, 7)

def is_non_collinear (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 - v.2 * w.1 ≠ 0

def forms_basis (v w : ℝ × ℝ) : Prop :=
  is_non_collinear v w

theorem e1_e2_form_basis : forms_basis e1 e2 := by
  sorry

end e1_e2_form_basis_l3990_399028


namespace arithmetic_sequence_vertex_l3990_399012

/-- Given that a, b, c, d form an arithmetic sequence and (a, d) is the vertex of f(x) = x^2 - 2x,
    prove that b + c = 0 -/
theorem arithmetic_sequence_vertex (a b c d : ℝ) : 
  (∃ r : ℝ, b = a + r ∧ c = b + r ∧ d = c + r) →  -- arithmetic sequence condition
  (a = 1 ∧ d = -1) →                              -- vertex condition
  b + c = 0 := by
sorry

end arithmetic_sequence_vertex_l3990_399012


namespace trapezoid_segment_length_squared_l3990_399091

/-- Represents a trapezoid with the given properties -/
structure Trapezoid where
  b : ℝ  -- length of the shorter base
  h : ℝ  -- height of the trapezoid
  midline_ratio : (b + 75) / (b + 25) = 3 / 2  -- ratio condition for the midline
  x : ℝ  -- length of the segment dividing the trapezoid into two equal areas
  equal_area_condition : x = 125 * (100 / (x - 75)) - 75

/-- The main theorem about the trapezoid -/
theorem trapezoid_segment_length_squared (t : Trapezoid) :
  ⌊(t.x^2) / 100⌋ = 181 := by
  sorry

end trapezoid_segment_length_squared_l3990_399091


namespace expression_simplification_l3990_399007

theorem expression_simplification (a : ℝ) (h1 : a ≠ 0) (h2 : a ≠ -1) :
  (2*a + 2) / a / (4 / a^2) - a / (a + 1) = (a^3 + 2*a^2 - a) / (2*a + 2) :=
by sorry

end expression_simplification_l3990_399007


namespace base_five_to_decimal_l3990_399076

/-- Converts a list of digits in a given base to its decimal representation. -/
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base ^ (digits.length - 1 - i)) 0

/-- The decimal representation of 3412 in base 5 is 482. -/
theorem base_five_to_decimal : to_decimal [3, 4, 1, 2] 5 = 482 := by sorry

end base_five_to_decimal_l3990_399076


namespace division_remainder_problem_l3990_399089

theorem division_remainder_problem (L S R : ℕ) : 
  L - S = 1365 → 
  L = 1575 → 
  L = 7 * S + R → 
  R = 105 := by
  sorry

end division_remainder_problem_l3990_399089


namespace least_subtraction_for_divisibility_l3990_399093

theorem least_subtraction_for_divisibility :
  ∃ (n : ℕ), n ≤ 7 ∧
  12 ∣ (652543 - n) ∧
  ∀ (m : ℕ), m < n → ¬(12 ∣ (652543 - m)) :=
sorry

end least_subtraction_for_divisibility_l3990_399093


namespace coefficient_of_x_is_negative_five_l3990_399022

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℤ := sorry

-- Define the general term of the expansion
def generalTerm (r : ℕ) : ℤ := (-1)^r * binomial 5 r

-- Define the exponent of x in the general term
def exponent (r : ℕ) : ℚ := (5 - 3*r) / 2

theorem coefficient_of_x_is_negative_five :
  ∃ (r : ℕ), exponent r = 1 ∧ generalTerm r = -5 := by sorry

end coefficient_of_x_is_negative_five_l3990_399022


namespace employed_females_percentage_is_16_percent_l3990_399002

/-- Represents an age group in Town X -/
inductive AgeGroup
  | Young    -- 18-34
  | Middle   -- 35-54
  | Senior   -- 55+

/-- The percentage of employed population in each age group -/
def employed_percentage : ℝ := 64

/-- The percentage of employed males in each age group -/
def employed_males_percentage : ℝ := 48

/-- The percentage of employed females in each age group -/
def employed_females_percentage : ℝ := employed_percentage - employed_males_percentage

theorem employed_females_percentage_is_16_percent :
  employed_females_percentage = 16 := by sorry

end employed_females_percentage_is_16_percent_l3990_399002


namespace school_problem_solution_l3990_399087

/-- Represents the number of students in each class of a school -/
structure School where
  class1 : ℕ
  class2 : ℕ
  class3 : ℕ
  class4 : ℕ
  class5 : ℕ

/-- The conditions of the school problem -/
def SchoolProblem (s : School) : Prop :=
  s.class1 = 23 ∧
  s.class2 < s.class1 ∧
  s.class3 < s.class2 ∧
  s.class4 < s.class3 ∧
  s.class5 < s.class4 ∧
  s.class1 + s.class2 + s.class3 + s.class4 + s.class5 = 95 ∧
  ∃ (x : ℕ), 
    s.class2 = s.class1 - x ∧
    s.class3 = s.class2 - x ∧
    s.class4 = s.class3 - x ∧
    s.class5 = s.class4 - x

theorem school_problem_solution (s : School) (h : SchoolProblem s) :
  ∃ (x : ℕ), x = 2 ∧
    s.class2 = s.class1 - x ∧
    s.class3 = s.class2 - x ∧
    s.class4 = s.class3 - x ∧
    s.class5 = s.class4 - x :=
  sorry

end school_problem_solution_l3990_399087


namespace shopkeeper_profit_l3990_399063

/-- Calculates the actual percent profit when a shopkeeper labels an item's price
    to earn a specified profit percentage and then offers a discount. -/
def actualPercentProfit (labeledProfitPercent : ℝ) (discountPercent : ℝ) : ℝ :=
  let labeledPrice := 1 + labeledProfitPercent
  let sellingPrice := labeledPrice * (1 - discountPercent)
  (sellingPrice - 1) * 100

/-- Proves that when a shopkeeper labels an item's price to earn a 30% profit
    on the cost price and then offers a 10% discount on the labeled price,
    the actual percent profit earned is 17%. -/
theorem shopkeeper_profit :
  actualPercentProfit 0.3 0.1 = 17 :=
by sorry

end shopkeeper_profit_l3990_399063


namespace arithmetic_number_difference_l3990_399011

/-- A 3-digit number with distinct digits forming an arithmetic sequence --/
def ArithmeticNumber (n : ℕ) : Prop :=
  ∃ a b c : ℕ,
    n = 100 * a + 10 * b + c ∧
    0 ≤ a ∧ a ≤ 9 ∧
    0 ≤ b ∧ b ≤ 9 ∧
    0 ≤ c ∧ c ≤ 9 ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    b - a = c - b

theorem arithmetic_number_difference : 
  (∃ max min : ℕ, 
    ArithmeticNumber max ∧ 
    ArithmeticNumber min ∧
    (∀ n : ℕ, ArithmeticNumber n → min ≤ n ∧ n ≤ max) ∧
    max - min = 864) :=
by sorry

end arithmetic_number_difference_l3990_399011


namespace original_fraction_l3990_399086

theorem original_fraction (x y : ℚ) :
  (x > 0) →
  (y > 0) →
  ((1.2 * x) / (0.75 * y) = 2 / 15) →
  (x / y = 1 / 12) := by
sorry

end original_fraction_l3990_399086


namespace sum_of_roots_l3990_399037

theorem sum_of_roots (m n : ℝ) : 
  m^2 - 3*m - 2 = 0 → n^2 - 3*n - 2 = 0 → m + n = 3 := by
  sorry

end sum_of_roots_l3990_399037


namespace johns_starting_elevation_l3990_399062

def starting_elevation (rate : ℝ) (time : ℝ) (final_elevation : ℝ) : ℝ :=
  final_elevation + rate * time

theorem johns_starting_elevation :
  starting_elevation 10 5 350 = 400 := by sorry

end johns_starting_elevation_l3990_399062


namespace cards_per_page_l3990_399052

theorem cards_per_page (new_cards old_cards pages : ℕ) 
  (h1 : new_cards = 8)
  (h2 : old_cards = 10)
  (h3 : pages = 6) :
  (new_cards + old_cards) / pages = 3 := by
  sorry

end cards_per_page_l3990_399052


namespace right_to_left_equiv_ordinary_l3990_399095

-- Define a function to represent the right-to-left evaluation
def rightToLeftEval (a b c d : ℝ) : ℝ := a * (b + (c - d))

-- Define a function to represent the ordinary algebraic notation
def ordinaryNotation (a b c d : ℝ) : ℝ := a * (b + c - d)

-- Theorem statement
theorem right_to_left_equiv_ordinary (a b c d : ℝ) :
  rightToLeftEval a b c d = ordinaryNotation a b c d := by
  sorry

end right_to_left_equiv_ordinary_l3990_399095


namespace min_F_beautiful_pair_l3990_399071

def is_beautiful_pair (p q : ℕ) : Prop :=
  ∃ x y : ℕ,
    1 ≤ x ∧ x ≤ 4 ∧
    1 ≤ y ∧ y ≤ 5 ∧
    p = 21 * x + y ∧
    q = 52 + y ∧
    (10 * y + x + 6 * y) % 13 = 0

def F (p q : ℕ) : ℕ :=
  let tens_p := p / 10
  let units_p := p % 10
  let tens_q := q / 10
  let units_q := q % 10
  10 * tens_p + units_q +
  10 * tens_p + units_p +
  10 * units_p + units_q +
  10 * units_p + tens_q

theorem min_F_beautiful_pair :
  ∀ p q : ℕ,
    is_beautiful_pair p q →
    F p q ≥ 156 :=
sorry

end min_F_beautiful_pair_l3990_399071


namespace factorization_proof_l3990_399005

theorem factorization_proof (x y : ℝ) : x^2 * y - x * y^2 = x * y * (x - y) := by
  sorry

end factorization_proof_l3990_399005


namespace range_of_a_l3990_399097

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x + a| > 2) ↔ (a < -1 ∨ a > 3) := by sorry

end range_of_a_l3990_399097


namespace tan_two_alpha_l3990_399036

theorem tan_two_alpha (α : Real) 
  (h : (Real.sin (Real.pi - α) + Real.sin (Real.pi / 2 - α)) / (Real.sin α - Real.cos α) = 1 / 2) : 
  Real.tan (2 * α) = 3 / 4 := by
  sorry

end tan_two_alpha_l3990_399036


namespace triangle_and_line_properties_l3990_399041

-- Define the points
def A : ℝ × ℝ := (4, 1)
def B : ℝ × ℝ := (-1, -6)
def C : ℝ × ℝ := (-3, 2)

-- Define the triangular region D
def D : Set (ℝ × ℝ) := {(x, y) | 7*x - 5*y - 23 ≤ 0 ∧ x + 7*y - 11 ≤ 0 ∧ 4*x + y + 10 ≥ 0}

-- Define the line 4x - 3y - a = 0
def line (a : ℝ) : Set (ℝ × ℝ) := {(x, y) | 4*x - 3*y - a = 0}

-- Theorem statement
theorem triangle_and_line_properties :
  -- B and C are on opposite sides of the line 4x - 3y - a = 0
  ∀ a : ℝ, (4 * B.1 - 3 * B.2 - a) * (4 * C.1 - 3 * C.2 - a) < 0 →
  -- The system of inequalities correctly represents region D
  (∀ p : ℝ × ℝ, p ∈ D ↔ 7*p.1 - 5*p.2 - 23 ≤ 0 ∧ p.1 + 7*p.2 - 11 ≤ 0 ∧ 4*p.1 + p.2 + 10 ≥ 0) ∧
  -- The range of values for a is (-18, 14)
  (∀ a : ℝ, (4 * B.1 - 3 * B.2 - a) * (4 * C.1 - 3 * C.2 - a) < 0 ↔ -18 < a ∧ a < 14) :=
sorry

end triangle_and_line_properties_l3990_399041
