import Mathlib

namespace toys_sold_second_week_is_26_l502_50216

/-- The number of toys sold in the second week at an online toy store. -/
def toys_sold_second_week (initial_stock : ℕ) (sold_first_week : ℕ) (toys_left : ℕ) : ℕ :=
  initial_stock - sold_first_week - toys_left

/-- Theorem stating that 26 toys were sold in the second week. -/
theorem toys_sold_second_week_is_26 :
  toys_sold_second_week 83 38 19 = 26 := by
  sorry

#eval toys_sold_second_week 83 38 19

end toys_sold_second_week_is_26_l502_50216


namespace interval_length_for_inequality_l502_50280

theorem interval_length_for_inequality : ∃ (a b : ℚ),
  (∀ x : ℝ, |5 * x^2 - 2/5| ≤ |x - 8| ↔ a ≤ x ∧ x ≤ b) ∧
  b - a = 13/5 :=
by sorry

end interval_length_for_inequality_l502_50280


namespace salesman_commission_percentage_l502_50203

/-- Proves that the flat commission percentage in the previous scheme is 5% --/
theorem salesman_commission_percentage :
  ∀ (previous_commission_percentage : ℝ),
    -- New scheme: Rs. 1000 fixed salary + 2.5% commission on sales exceeding Rs. 4,000
    let new_scheme_fixed_salary : ℝ := 1000
    let new_scheme_commission_rate : ℝ := 2.5 / 100
    let sales_threshold : ℝ := 4000
    -- Total sales
    let total_sales : ℝ := 12000
    -- Calculate new scheme remuneration
    let new_scheme_commission : ℝ := new_scheme_commission_rate * (total_sales - sales_threshold)
    let new_scheme_remuneration : ℝ := new_scheme_fixed_salary + new_scheme_commission
    -- Previous scheme remuneration
    let previous_scheme_remuneration : ℝ := previous_commission_percentage / 100 * total_sales
    -- New scheme remuneration is Rs. 600 more than the previous scheme
    new_scheme_remuneration = previous_scheme_remuneration + 600
    →
    previous_commission_percentage = 5 := by
  sorry

end salesman_commission_percentage_l502_50203


namespace lcm_of_12_and_16_l502_50228

theorem lcm_of_12_and_16 :
  let n : ℕ := 12
  let m : ℕ := 16
  let gcf : ℕ := 4
  Nat.gcd n m = gcf →
  Nat.lcm n m = 48 := by
sorry

end lcm_of_12_and_16_l502_50228


namespace arctan_gt_arcsin_iff_in_open_interval_l502_50242

theorem arctan_gt_arcsin_iff_in_open_interval (x : ℝ) :
  Real.arctan x > Real.arcsin x ↔ x ∈ Set.Ioo (-1 : ℝ) 0 :=
by
  sorry

end arctan_gt_arcsin_iff_in_open_interval_l502_50242


namespace fraction_subtraction_property_l502_50240

theorem fraction_subtraction_property (a b n : ℕ) (h1 : b > a) (h2 : a > 0) 
  (h3 : ∀ k : ℕ, k > 0 → (1 : ℚ) / k ≤ a / b → k ≥ n) : a * n - b < a := by
  sorry

end fraction_subtraction_property_l502_50240


namespace quadratic_function_m_value_l502_50287

/-- A quadratic function of the form y = 3x^2 + 2(m-1)x + n -/
def quadratic_function (m n : ℝ) (x : ℝ) : ℝ := 3 * x^2 + 2 * (m - 1) * x + n

/-- The derivative of the quadratic function -/
def quadratic_derivative (m : ℝ) (x : ℝ) : ℝ := 6 * x + 2 * (m - 1)

theorem quadratic_function_m_value (m n : ℝ) :
  (∀ x < 1, quadratic_derivative m x < 0) →
  (∀ x ≥ 1, quadratic_derivative m x ≥ 0) →
  m = -2 := by sorry

end quadratic_function_m_value_l502_50287


namespace total_cost_is_correct_l502_50234

/-- Calculates the total cost of purchasing a puppy and related items. -/
def total_cost (puppy_cost : ℚ) (food_consumption_per_day : ℚ) (food_duration_weeks : ℕ) 
  (food_cost_per_bag : ℚ) (food_amount_per_bag : ℚ) (leash_cost : ℚ) (collar_cost : ℚ) 
  (dog_bed_cost : ℚ) (sales_tax_rate : ℚ) : ℚ :=
  let food_total_consumption := food_consumption_per_day * (food_duration_weeks * 7)
  let food_bags_needed := (food_total_consumption / food_amount_per_bag).ceil
  let food_cost := food_bags_needed * food_cost_per_bag
  let collar_discounted := collar_cost * (1 - 0.1)
  let taxable_items_cost := leash_cost + collar_discounted + dog_bed_cost
  let tax_amount := taxable_items_cost * sales_tax_rate
  puppy_cost + food_cost + taxable_items_cost + tax_amount

/-- Theorem stating that the total cost is $211.85 given the specified conditions. -/
theorem total_cost_is_correct : 
  total_cost 150 (1/3) 6 2 (7/2) 15 12 25 (6/100) = 21185/100 := by
  sorry


end total_cost_is_correct_l502_50234


namespace toys_sold_l502_50221

def initial_toys : ℕ := 7
def remaining_toys : ℕ := 4

theorem toys_sold : initial_toys - remaining_toys = 3 := by
  sorry

end toys_sold_l502_50221


namespace tourists_scientific_notation_l502_50281

-- Define the number of tourists
def tourists : ℝ := 4.55e9

-- Theorem statement
theorem tourists_scientific_notation :
  tourists = 4.55 * (10 : ℝ) ^ 9 :=
by sorry

end tourists_scientific_notation_l502_50281


namespace solve_system_l502_50254

theorem solve_system (p q : ℚ) 
  (eq1 : 2 * p + 5 * q = 10) 
  (eq2 : 5 * p + 2 * q = 20) : 
  q = 10 / 21 := by
  sorry

end solve_system_l502_50254


namespace certain_number_proof_l502_50251

theorem certain_number_proof (x y : ℕ) : 
  x + y = 24 → 
  x = 11 → 
  x ≤ y → 
  7 * x + 5 * y = 142 := by
sorry

end certain_number_proof_l502_50251


namespace absolute_value_equation_solution_l502_50241

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |2 * x - 6| = 3 * x + 1 := by
  sorry

end absolute_value_equation_solution_l502_50241


namespace solve_inequality_range_of_a_l502_50218

-- Part 1
def inequality_solution_set (x : ℝ) : Prop :=
  x^2 - 5*x + 4 > 0

theorem solve_inequality :
  ∀ x : ℝ, inequality_solution_set x ↔ (x < 1 ∨ x > 4) :=
sorry

-- Part 2
def always_positive (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + a*x + 4 > 0

theorem range_of_a :
  ∀ a : ℝ, always_positive a ↔ -4 < a ∧ a < 4 :=
sorry

end solve_inequality_range_of_a_l502_50218


namespace complement_A_intersect_B_l502_50258

-- Define the universal set U
def U : Set ℕ := {1, 3, 5, 7, 9}

-- Define set A
def A : Set ℕ := {1, 5, 9}

-- Define set B
def B : Set ℕ := {3, 7, 9}

-- Theorem statement
theorem complement_A_intersect_B : (Aᶜ ∩ B) = {3, 7} := by
  sorry

end complement_A_intersect_B_l502_50258


namespace partner_p_investment_time_l502_50299

/-- The investment and profit scenario for two partners -/
structure InvestmentScenario where
  /-- The ratio of investments for partners p and q -/
  investment_ratio : Rat × Rat
  /-- The ratio of profits for partners p and q -/
  profit_ratio : Rat × Rat
  /-- The number of months partner q invested -/
  q_months : ℕ

/-- The theorem stating the investment time for partner p -/
theorem partner_p_investment_time (scenario : InvestmentScenario) 
  (h1 : scenario.investment_ratio = (7, 5))
  (h2 : scenario.profit_ratio = (7, 13))
  (h3 : scenario.q_months = 13) :
  ∃ (p_months : ℕ), p_months = 7 ∧ 
  (scenario.investment_ratio.1 * p_months) / (scenario.investment_ratio.2 * scenario.q_months) = 
  scenario.profit_ratio.1 / scenario.profit_ratio.2 :=
sorry

end partner_p_investment_time_l502_50299


namespace geometric_sequence_fourth_term_l502_50290

/-- Given a geometric sequence with first term a and common ratio r,
    prove that if the first three terms are of the form a, 3a+3, 6a+6,
    then the fourth term is -24. -/
theorem geometric_sequence_fourth_term
  (a r : ℝ) -- a is the first term, r is the common ratio
  (h1 : (3*a + 3) = a * r) -- second term = first term * r
  (h2 : (6*a + 6) = (3*a + 3) * r) -- third term = second term * r
  : a * r^3 = -24 := by
  sorry


end geometric_sequence_fourth_term_l502_50290


namespace arc_length_for_given_circle_l502_50260

/-- Given a circle with radius 2 and a central angle of 2 radians, 
    the corresponding arc length is 4. -/
theorem arc_length_for_given_circle : 
  ∀ (r θ l : ℝ), r = 2 → θ = 2 → l = r * θ → l = 4 :=
by sorry

end arc_length_for_given_circle_l502_50260


namespace N_subset_M_l502_50273

-- Define the sets M and N
def M : Set ℝ := Set.univ
def N : Set ℝ := {y : ℝ | ∃ x : ℝ, y = -x^2}

-- State the theorem
theorem N_subset_M : N ⊆ M := by sorry

end N_subset_M_l502_50273


namespace condition_relationship_l502_50275

theorem condition_relationship (x : ℝ) :
  (1 / x > 1 → x < 1) ∧ ¬(x < 1 → 1 / x > 1) := by
  sorry

end condition_relationship_l502_50275


namespace min_a_over_x_l502_50252

theorem min_a_over_x (a x y : ℕ) (ha : a > 100) (hx : x > 100) (hy : y > 100)
  (h : y^2 - 1 = a^2 * (x^2 - 1)) :
  ∀ k : ℚ, (k : ℝ) = a / x → k ≥ 2 ∧ ∃ a₀ x₀ y₀ : ℕ,
    a₀ > 100 ∧ x₀ > 100 ∧ y₀ > 100 ∧
    y₀^2 - 1 = a₀^2 * (x₀^2 - 1) ∧
    (a₀ : ℝ) / x₀ = 2 :=
by sorry

end min_a_over_x_l502_50252


namespace octal_123_equals_decimal_83_l502_50229

/-- Converts an octal number to decimal --/
def octal_to_decimal (d₂ d₁ d₀ : ℕ) : ℕ :=
  d₂ * 8^2 + d₁ * 8^1 + d₀ * 8^0

/-- Proves that the octal number 123₈ is equal to the decimal number 83 --/
theorem octal_123_equals_decimal_83 : octal_to_decimal 1 2 3 = 83 := by
  sorry

end octal_123_equals_decimal_83_l502_50229


namespace machine_production_in_10_seconds_l502_50227

/-- A machine that produces items at a constant rate -/
structure Machine where
  items_per_minute : ℕ

/-- Calculate the number of items produced in a given number of seconds -/
def items_produced (m : Machine) (seconds : ℕ) : ℚ :=
  (m.items_per_minute : ℚ) * (seconds : ℚ) / 60

theorem machine_production_in_10_seconds (m : Machine) 
  (h : m.items_per_minute = 150) : 
  items_produced m 10 = 25 := by
  sorry

#eval items_produced ⟨150⟩ 10

end machine_production_in_10_seconds_l502_50227


namespace calculation_proof_l502_50269

theorem calculation_proof : (Real.sqrt 5 - 1)^0 + 3⁻¹ - |-(1/3)| = 1 := by
  sorry

end calculation_proof_l502_50269


namespace arnel_kept_pencils_l502_50272

/-- Calculates the number of pencils Arnel kept given the problem conditions -/
def pencils_kept (num_boxes : ℕ) (num_friends : ℕ) (pencils_per_friend : ℕ) (pencils_left_per_box : ℕ) : ℕ :=
  let total_pencils := num_boxes * (pencils_per_friend * num_friends / num_boxes + pencils_left_per_box)
  total_pencils - pencils_per_friend * num_friends

/-- Theorem stating that Arnel kept 50 pencils under the given conditions -/
theorem arnel_kept_pencils :
  pencils_kept 10 5 8 5 = 50 := by
  sorry

end arnel_kept_pencils_l502_50272


namespace basketball_probabilities_l502_50283

/-- Probability of A making a shot -/
def prob_A_makes : ℝ := 0.8

/-- Probability of B missing a shot -/
def prob_B_misses : ℝ := 0.1

/-- Probability of B making a shot -/
def prob_B_makes : ℝ := 1 - prob_B_misses

theorem basketball_probabilities :
  (prob_A_makes * prob_B_makes = 0.72) ∧
  (prob_A_makes * (1 - prob_B_makes) + (1 - prob_A_makes) * prob_B_makes = 0.26) := by
  sorry

end basketball_probabilities_l502_50283


namespace steve_total_cost_is_23_56_l502_50267

/-- Calculates the total cost of Steve's DVD purchase --/
def steveTotalCost (mikeDVDPrice baseShippingRate salesTaxRate discountRate : ℚ) : ℚ :=
  let steveDVDPrice := 2 * mikeDVDPrice
  let otherDVDPrice := 7
  let subtotalBeforePromo := otherDVDPrice + otherDVDPrice
  let shippingCost := baseShippingRate * subtotalBeforePromo
  let subtotalWithShipping := subtotalBeforePromo + shippingCost
  let salesTax := salesTaxRate * subtotalWithShipping
  let subtotalWithTax := subtotalWithShipping + salesTax
  let discount := discountRate * subtotalWithTax
  subtotalWithTax - discount

/-- Theorem stating that Steve's total cost is $23.56 --/
theorem steve_total_cost_is_23_56 :
  steveTotalCost 5 0.8 0.1 0.15 = 23.56 := by
  sorry

end steve_total_cost_is_23_56_l502_50267


namespace digit_count_l502_50238

theorem digit_count (n : ℕ) 
  (h1 : (n : ℚ) * 18 = n * 18) 
  (h2 : 4 * 8 = 32) 
  (h3 : 5 * 26 = 130) 
  (h4 : n * 18 = 32 + 130) : n = 9 := by
  sorry

end digit_count_l502_50238


namespace compound_interest_problem_l502_50282

-- Define the compound interest function
def compound_interest (P r : ℝ) (n : ℕ) : ℝ := P * (1 + r) ^ n

-- State the theorem
theorem compound_interest_problem :
  ∃ (P r : ℝ), 
    compound_interest P r 2 = 8800 ∧
    compound_interest P r 3 = 9261 ∧
    abs (P - 7945.67) < 0.01 := by
  sorry

end compound_interest_problem_l502_50282


namespace sum_of_repeating_decimals_three_and_six_l502_50265

/-- Represents a repeating decimal with a single repeating digit -/
def RepeatingDecimal (d : Nat) : ℚ := d / 9

/-- The sum of the repeating decimals 0.3333... and 0.6666... is equal to 1 -/
theorem sum_of_repeating_decimals_three_and_six :
  RepeatingDecimal 3 + RepeatingDecimal 6 = 1 := by
  sorry

end sum_of_repeating_decimals_three_and_six_l502_50265


namespace prob_B_wins_match_value_l502_50271

/-- The probability of player B winning a single game -/
def p_B : ℝ := 0.4

/-- The probability of player A winning a single game -/
def p_A : ℝ := 1 - p_B

/-- The probability of player B winning a best-of-three billiards match -/
def prob_B_wins_match : ℝ := p_B^2 + 2 * p_B^2 * p_A

theorem prob_B_wins_match_value :
  prob_B_wins_match = 0.352 := by sorry

end prob_B_wins_match_value_l502_50271


namespace function_value_theorem_l502_50209

/-- Given functions f and g, prove that f(2) = 2 under certain conditions -/
theorem function_value_theorem (a b c : ℝ) (h_abc : a * b * c ≠ 0) :
  let f := fun (x : ℝ) ↦ a * x^2 + b * Real.cos x
  let g := fun (x : ℝ) ↦ c * Real.sin x
  (f 2 + g 2 = 3) → (f (-2) + g (-2) = 1) → f 2 = 2 :=
by
  sorry


end function_value_theorem_l502_50209


namespace apples_in_basket_l502_50263

def apples_remaining (initial : ℕ) (ricki_removes : ℕ) : ℕ :=
  initial - (ricki_removes + 2 * ricki_removes)

theorem apples_in_basket (initial : ℕ) (ricki_removes : ℕ) 
  (h1 : initial = 74) (h2 : ricki_removes = 14) : 
  apples_remaining initial ricki_removes = 32 := by
  sorry

end apples_in_basket_l502_50263


namespace infinite_geometric_series_sum_l502_50288

/-- The sum of an infinite geometric series with first term 5/3 and common ratio -9/20 is 100/87 -/
theorem infinite_geometric_series_sum :
  let a : ℚ := 5/3
  let r : ℚ := -9/20
  let S := a / (1 - r)
  S = 100/87 := by sorry

end infinite_geometric_series_sum_l502_50288


namespace sqrt_expressions_l502_50222

theorem sqrt_expressions :
  (∀ x y z : ℝ, x = 27 ∧ y = 1/3 ∧ z = 3 → 
    Real.sqrt x - Real.sqrt y + Real.sqrt z = (11 * Real.sqrt 3) / 3) ∧
  (∀ a b c : ℝ, a = 32 ∧ b = 18 ∧ c = 2 → 
    (Real.sqrt a + Real.sqrt b) / Real.sqrt c - 8 = -1) :=
by sorry

end sqrt_expressions_l502_50222


namespace jelly_cost_l502_50204

theorem jelly_cost (N C J : ℕ) (h1 : N > 1) (h2 : 3 * N * C + 6 * N * J = 312) : 
  (6 * N * J : ℚ) / 100 = 0.72 := by sorry

end jelly_cost_l502_50204


namespace solution_implication_l502_50285

theorem solution_implication (m n : ℝ) : 
  (2 * m + n = 8 ∧ 2 * n - m = 1) → 
  Real.sqrt (2 * m - n) = 2 := by
sorry

end solution_implication_l502_50285


namespace max_partition_product_l502_50214

def partition_product (p : List Nat) : Nat :=
  p.prod

def is_valid_partition (p : List Nat) : Prop :=
  p.sum = 25 ∧ p.all (· > 0) ∧ p.length ≤ 25

theorem max_partition_product :
  ∃ (max_p : List Nat), 
    is_valid_partition max_p ∧ 
    partition_product max_p = 8748 ∧
    ∀ (p : List Nat), is_valid_partition p → partition_product p ≤ 8748 := by
  sorry

end max_partition_product_l502_50214


namespace expression_value_l502_50259

theorem expression_value : 3 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 2400 := by
  sorry

end expression_value_l502_50259


namespace direct_proportion_implies_m_eq_two_l502_50250

/-- A function y of x is a direct proportion if it can be written as y = kx where k is a non-zero constant -/
def is_direct_proportion (y : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x, y x = k * x

/-- Given y = (m^2 + 2m)x^(m^2 - 3), if y is a direct proportion function of x, then m = 2 -/
theorem direct_proportion_implies_m_eq_two (m : ℝ) :
  is_direct_proportion (fun x => (m^2 + 2*m) * x^(m^2 - 3)) → m = 2 := by
  sorry

end direct_proportion_implies_m_eq_two_l502_50250


namespace mass_of_man_is_72_l502_50278

/-- The density of water in kg/m³ -/
def water_density : ℝ := 1000

/-- Calculates the mass of a man based on boat dimensions and sinking depth -/
def mass_of_man (boat_length boat_breadth sinking_depth : ℝ) : ℝ :=
  water_density * boat_length * boat_breadth * sinking_depth

/-- Theorem stating that the mass of the man is 72 kg given the boat's dimensions and sinking depth -/
theorem mass_of_man_is_72 :
  mass_of_man 3 2 0.012 = 72 := by sorry

end mass_of_man_is_72_l502_50278


namespace cloth_selling_price_l502_50262

/-- Calculates the total selling price of cloth given the quantity, profit per meter, and cost price per meter. -/
theorem cloth_selling_price 
  (quantity : ℕ) 
  (profit_per_meter : ℕ) 
  (cost_price_per_meter : ℕ) :
  quantity = 85 →
  profit_per_meter = 35 →
  cost_price_per_meter = 70 →
  quantity * (profit_per_meter + cost_price_per_meter) = 8925 := by
  sorry

#check cloth_selling_price

end cloth_selling_price_l502_50262


namespace parabola_point_comparison_l502_50255

/-- Proves that for a downward-opening parabola passing through (-1, y₁) and (4, y₂), y₁ > y₂ -/
theorem parabola_point_comparison (a c y₁ y₂ : ℝ) 
  (h_a : a < 0)
  (h_y₁ : y₁ = a * (-1 - 1)^2 + c)
  (h_y₂ : y₂ = a * (4 - 1)^2 + c) :
  y₁ > y₂ := by
  sorry

end parabola_point_comparison_l502_50255


namespace most_suitable_student_l502_50239

-- Define the students
inductive Student : Type
| A : Student
| B : Student
| C : Student
| D : Student

-- Define the average score and variances
def average_score : ℝ := 180

def variance (s : Student) : ℝ :=
  match s with
  | Student.A => 65
  | Student.B => 56.5
  | Student.C => 53
  | Student.D => 50.5

-- Define the suitability criterion
def more_suitable (s1 s2 : Student) : Prop :=
  variance s1 < variance s2

-- Theorem statement
theorem most_suitable_student :
  ∀ s : Student, s ≠ Student.D → more_suitable Student.D s :=
sorry

end most_suitable_student_l502_50239


namespace cos_arcsin_three_fifths_l502_50233

theorem cos_arcsin_three_fifths : 
  Real.cos (Real.arcsin (3/5)) = 4/5 := by sorry

end cos_arcsin_three_fifths_l502_50233


namespace investment_problem_l502_50237

/-- Investment problem -/
theorem investment_problem 
  (x_investment : ℕ) 
  (y_investment : ℕ) 
  (z_join_time : ℕ) 
  (total_profit : ℕ) 
  (z_profit_share : ℕ) 
  (h1 : x_investment = 36000)
  (h2 : y_investment = 42000)
  (h3 : z_join_time = 4)
  (h4 : total_profit = 13860)
  (h5 : z_profit_share = 4032) :
  ∃ z_investment : ℕ, z_investment = 52000 ∧ 
    (x_investment * 12 + y_investment * 12) * z_profit_share = 
    z_investment * (12 - z_join_time) * (total_profit - z_profit_share) :=
sorry

end investment_problem_l502_50237


namespace number_less_than_opposite_l502_50206

theorem number_less_than_opposite (x : ℝ) : x = -x + (-4) ↔ x + 4 = -x := by sorry

end number_less_than_opposite_l502_50206


namespace constant_term_of_expansion_l502_50256

theorem constant_term_of_expansion (x : ℝ) (x_pos : x > 0) :
  ∃ (c : ℝ), (∀ ε > 0, ∃ δ > 0, ∀ y, |y - x| < δ → |((y^(1/2) - 2/y)^3 - (x^(1/2) - 2/x)^3) - c| < ε) ∧ c = -6 :=
sorry

end constant_term_of_expansion_l502_50256


namespace red_light_runners_estimate_l502_50208

/-- Represents the result of a survey on traffic law compliance -/
structure SurveyResult where
  total_students : ℕ
  yes_answers : ℕ
  id_range : Finset ℕ
  odd_ids : Finset ℕ

/-- Calculates the estimated number of students who have run a red light -/
def estimate_red_light_runners (result : SurveyResult) : ℕ :=
  2 * (result.yes_answers - result.odd_ids.card / 2)

/-- Theorem stating the estimated number of red light runners based on the survey -/
theorem red_light_runners_estimate 
  (result : SurveyResult)
  (h1 : result.total_students = 300)
  (h2 : result.yes_answers = 90)
  (h3 : result.id_range = Finset.range 300)
  (h4 : result.odd_ids = result.id_range.filter (fun n => n % 2 = 1)) :
  estimate_red_light_runners result = 30 := by
  sorry

end red_light_runners_estimate_l502_50208


namespace inequality_proof_l502_50213

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) : 
  1 / (a - b) + 1 / (b - c) + 4 / (c - a) ≥ 0 :=
by sorry

end inequality_proof_l502_50213


namespace harmonic_sum_denominator_not_div_by_five_l502_50257

/-- The sum of reciprocals from 1 to n -/
def harmonic_sum (n : ℕ+) : ℚ :=
  Finset.sum (Finset.range n) (λ m => 1 / (m + 1 : ℚ))

/-- The set of positive integers n for which 5 does not divide the denominator
    of the harmonic sum when expressed in lowest terms -/
def D : Set ℕ+ :=
  {n | ¬ (5 ∣ (harmonic_sum n).den)}

/-- The theorem stating that D is exactly the given set -/
theorem harmonic_sum_denominator_not_div_by_five :
  D = {1, 2, 3, 4, 20, 21, 22, 23, 24, 100, 101, 102, 103, 104, 120, 121, 122, 123, 124} := by
  sorry


end harmonic_sum_denominator_not_div_by_five_l502_50257


namespace correct_equation_l502_50279

theorem correct_equation (x : ℝ) : 
  (550 + x) + (460 + x) + (359 + x) + (340 + x) = 2012 + x ↔ x = 75.75 := by
  sorry

end correct_equation_l502_50279


namespace jenny_toy_spending_l502_50264

/-- Proves that Jenny spent $200 on toys for the cat in the first year. -/
theorem jenny_toy_spending (adoption_fee vet_cost monthly_food_cost total_months jenny_total_spent : ℕ) 
  (h1 : adoption_fee = 50)
  (h2 : vet_cost = 500)
  (h3 : monthly_food_cost = 25)
  (h4 : total_months = 12)
  (h5 : jenny_total_spent = 625) : 
  jenny_total_spent - (adoption_fee + vet_cost + monthly_food_cost * total_months) / 2 = 200 := by
  sorry

end jenny_toy_spending_l502_50264


namespace original_number_l502_50215

theorem original_number (final_number : ℝ) (increase_percentage : ℝ) 
  (h1 : final_number = 210)
  (h2 : increase_percentage = 0.40) : 
  final_number = (1 + increase_percentage) * 150 := by
  sorry

end original_number_l502_50215


namespace car_travel_time_ratio_l502_50212

/-- Proves that the ratio of time taken at 70 km/h to the original time is 3:2 -/
theorem car_travel_time_ratio :
  let distance : ℝ := 630
  let original_time : ℝ := 6
  let new_speed : ℝ := 70
  let new_time : ℝ := distance / new_speed
  new_time / original_time = 3 / 2 := by sorry

end car_travel_time_ratio_l502_50212


namespace halloween_trick_or_treat_l502_50205

theorem halloween_trick_or_treat (duration : ℕ) (houses_per_hour : ℕ) (treats_per_house : ℕ) (total_treats : ℕ) :
  duration = 4 →
  houses_per_hour = 5 →
  treats_per_house = 3 →
  total_treats = 180 →
  total_treats / (duration * houses_per_hour * treats_per_house) = 3 := by
sorry


end halloween_trick_or_treat_l502_50205


namespace P_plus_Q_equals_46_l502_50276

theorem P_plus_Q_equals_46 (P Q : ℝ) :
  (∀ x : ℝ, x ≠ 3 → P / (x - 3) + Q * (x - 2) = (-5 * x^2 + 20 * x + 36) / (x - 3)) →
  P + Q = 46 := by
sorry

end P_plus_Q_equals_46_l502_50276


namespace square_sum_given_conditions_l502_50248

theorem square_sum_given_conditions (x y : ℝ) 
  (h1 : (x + y)^2 = 9) 
  (h2 : x * y = -6) : 
  x^2 + y^2 = 21 := by
sorry

end square_sum_given_conditions_l502_50248


namespace hannah_easter_eggs_l502_50220

theorem hannah_easter_eggs :
  ∀ (total helen hannah : ℕ),
  total = 63 →
  hannah = 2 * helen →
  total = helen + hannah →
  hannah = 42 := by
sorry

end hannah_easter_eggs_l502_50220


namespace prob_red_then_black_standard_deck_l502_50249

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (red_cards : Nat)
  (black_cards : Nat)

/-- Definition of a standard 52-card deck -/
def standard_deck : Deck :=
  { total_cards := 52,
    red_cards := 26,
    black_cards := 26 }

/-- Probability of drawing a red card first and a black card second -/
def prob_red_then_black (d : Deck) : ℚ :=
  (d.red_cards : ℚ) / d.total_cards * (d.black_cards : ℚ) / (d.total_cards - 1)

/-- Theorem stating the probability for a standard deck -/
theorem prob_red_then_black_standard_deck :
  prob_red_then_black standard_deck = 13 / 51 := by
  sorry

end prob_red_then_black_standard_deck_l502_50249


namespace isosceles_similar_triangle_perimeter_l502_50293

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  positive_a : a > 0
  positive_b : b > 0
  positive_c : c > 0
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- Defines similarity between two triangles -/
def similar (t1 t2 : Triangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ 
    t2.a = k * t1.a ∧
    t2.b = k * t1.b ∧
    t2.c = k * t1.c

/-- Calculates the perimeter of a triangle -/
def perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

theorem isosceles_similar_triangle_perimeter :
  ∀ (t1 t2 : Triangle),
    t1.a = 15 ∧ t1.b = 30 ∧ t1.c = 30 →
    similar t1 t2 →
    t2.a = 75 →
    perimeter t2 = 375 := by
  sorry

end isosceles_similar_triangle_perimeter_l502_50293


namespace fraction_sum_equality_l502_50277

theorem fraction_sum_equality (a b c : ℝ) (n : ℕ) 
  (h1 : 1/a + 1/b + 1/c = 1/(a + b + c)) 
  (h2 : Odd n) 
  (h3 : n > 0) : 
  1/a^n + 1/b^n + 1/c^n = 1/(a^n + b^n + c^n) :=
by sorry

end fraction_sum_equality_l502_50277


namespace jonas_sequence_l502_50297

/-- Sequence of positive multiples of 13 in ascending order -/
def multiples_of_13 : ℕ → ℕ := λ n => 13 * (n + 1)

/-- The nth digit in the sequence of multiples of 13 -/
def nth_digit (n : ℕ) : ℕ := sorry

/-- Whether a number appears in the sequence of multiples of 13 -/
def appears_in_sequence (m : ℕ) : Prop := ∃ k : ℕ, multiples_of_13 k = m

theorem jonas_sequence :
  (nth_digit 2019 = 8) ∧ appears_in_sequence 2019 := by sorry

end jonas_sequence_l502_50297


namespace quadratic_inequality_coefficient_sum_l502_50217

/-- Given a quadratic inequality ax^2 - bx + 2 < 0 with solution set {x | 1 < x < 2},
    prove that the sum of coefficients a + b equals 4. -/
theorem quadratic_inequality_coefficient_sum (a b : ℝ) : 
  (∀ x, (1 < x ∧ x < 2) ↔ (a * x^2 - b * x + 2 < 0)) → 
  a + b = 4 := by
sorry

end quadratic_inequality_coefficient_sum_l502_50217


namespace books_on_shelf_l502_50232

def initial_books : ℕ := 38
def marta_removes : ℕ := 10
def tom_removes : ℕ := 5
def tom_adds : ℕ := 12

theorem books_on_shelf : 
  initial_books - marta_removes - tom_removes + tom_adds = 35 := by
  sorry

end books_on_shelf_l502_50232


namespace three_solutions_l502_50286

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- The number of positive integers n satisfying n + S(n) + S(S(n)) = 2500 -/
def count_solutions : ℕ := sorry

/-- Theorem stating that there are exactly 3 solutions -/
theorem three_solutions : count_solutions = 3 := by sorry

end three_solutions_l502_50286


namespace harris_dog_carrot_cost_l502_50211

/-- The annual cost of carrots for Harris's dog -/
def annual_carrot_cost (carrots_per_day : ℕ) (carrots_per_bag : ℕ) (cost_per_bag : ℚ) (days_per_year : ℕ) : ℚ :=
  (days_per_year * carrots_per_day / carrots_per_bag) * cost_per_bag

/-- Theorem stating the annual cost of carrots for Harris's dog -/
theorem harris_dog_carrot_cost :
  annual_carrot_cost 1 5 2 365 = 146 := by
  sorry

end harris_dog_carrot_cost_l502_50211


namespace gcd_bound_for_special_fraction_l502_50289

theorem gcd_bound_for_special_fraction (a b : ℕ+) 
  (h : ∃ (k : ℤ), (a.1 + 1 : ℚ) / b.1 + (b.1 + 1 : ℚ) / a.1 = k) : 
  Nat.gcd a.1 b.1 ≤ Real.sqrt (a.1 + b.1) := by
  sorry

end gcd_bound_for_special_fraction_l502_50289


namespace linda_egg_ratio_l502_50226

theorem linda_egg_ratio : 
  ∀ (total_eggs : ℕ) (brown_eggs : ℕ) (white_eggs : ℕ),
  total_eggs = 12 →
  brown_eggs = 5 →
  total_eggs = brown_eggs + white_eggs →
  (white_eggs : ℚ) / (brown_eggs : ℚ) = 7 / 5 := by
sorry

end linda_egg_ratio_l502_50226


namespace exists_uncovered_vertices_l502_50244

/-- A regular polygon with 2n vertices -/
structure RegularPolygon (n : ℕ) :=
  (vertices : Fin (2*n) → ℝ × ℝ)

/-- A pattern is a subset of n vertices of a 2n-gon -/
def Pattern (n : ℕ) := Finset (Fin (2*n))

/-- Rotation of a pattern by k positions -/
def rotate (n : ℕ) (p : Pattern n) (k : ℕ) : Pattern n :=
  sorry

/-- The set of vertices covered by 100 rotations of a pattern -/
def coveredVertices (n : ℕ) (p : Pattern n) : Finset (Fin (2*n)) :=
  sorry

/-- Theorem stating that there exists a 2n-gon and a pattern such that
    100 rotations do not cover all vertices -/
theorem exists_uncovered_vertices :
  ∃ (n : ℕ) (p : Pattern n), (coveredVertices n p).card < 2*n :=
sorry

end exists_uncovered_vertices_l502_50244


namespace point_coordinates_in_second_quadrant_l502_50236

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the second quadrant
def second_quadrant (p : Point) : Prop :=
  p.1 < 0 ∧ p.2 > 0

-- Define distance to x-axis
def distance_to_x_axis (p : Point) : ℝ :=
  |p.2|

-- Define distance to y-axis
def distance_to_y_axis (p : Point) : ℝ :=
  |p.1|

-- Theorem statement
theorem point_coordinates_in_second_quadrant (M : Point) 
  (h1 : second_quadrant M)
  (h2 : distance_to_x_axis M = 1)
  (h3 : distance_to_y_axis M = 2) :
  M = (-2, 1) :=
sorry

end point_coordinates_in_second_quadrant_l502_50236


namespace intersection_P_complement_Q_l502_50296

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def P : Set ℕ := {1, 2, 3, 4}
def Q : Set ℕ := {3, 4, 5}

theorem intersection_P_complement_Q : P ∩ (U \ Q) = {1, 2} := by sorry

end intersection_P_complement_Q_l502_50296


namespace solve_for_a_l502_50200

theorem solve_for_a : ∃ a : ℝ, 
  (∀ x y : ℝ, x = 1 ∧ y = -3 → a * x - y = 1) ∧ a = -2 := by
  sorry

end solve_for_a_l502_50200


namespace basketball_league_games_l502_50266

/-- The number of games played in a league --/
def total_games (n : ℕ) (g : ℕ) : ℕ :=
  n * (n - 1) * g / 2

/-- Theorem: In a league with 10 teams, where each team plays 5 games with every other team, 
    the total number of games played is 225. --/
theorem basketball_league_games : total_games 10 5 = 225 := by
  sorry

end basketball_league_games_l502_50266


namespace angle_bisector_sum_l502_50261

/-- Triangle with vertices P, Q, R -/
structure Triangle where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ

/-- Angle bisector equation coefficients -/
structure AngleBisectorEq where
  a : ℝ
  c : ℝ

/-- Theorem: For the given triangle, the angle bisector equation of ∠P has a + c = 89 -/
theorem angle_bisector_sum (t : Triangle) (eq : AngleBisectorEq) : 
  t.P = (-8, 5) → t.Q = (-15, -19) → t.R = (1, -7) → 
  (∃ (x y : ℝ), eq.a * x + 2 * y + eq.c = 0) →
  eq.a + eq.c = 89 := by
  sorry

end angle_bisector_sum_l502_50261


namespace scientific_notation_of_0_000815_l502_50284

def scientific_notation (n : ℝ) (coefficient : ℝ) (exponent : ℤ) : Prop :=
  1 ≤ coefficient ∧ coefficient < 10 ∧ n = coefficient * (10 : ℝ) ^ exponent

theorem scientific_notation_of_0_000815 :
  scientific_notation 0.000815 8.15 (-4) := by
  sorry

end scientific_notation_of_0_000815_l502_50284


namespace vector_form_equiv_line_equation_l502_50247

/-- The line equation y = 2x + 5 -/
def line_equation (x y : ℝ) : Prop := y = 2 * x + 5

/-- The vector form of the line -/
def vector_form (r k t x y : ℝ) : Prop :=
  x = r + 3 * t ∧ y = -3 + k * t

/-- Theorem stating that the vector form represents the line y = 2x + 5 
    if and only if r = -4 and k = 6 -/
theorem vector_form_equiv_line_equation :
  ∀ r k : ℝ, (∀ t x y : ℝ, vector_form r k t x y → line_equation x y) ∧
             (∀ x y : ℝ, line_equation x y → ∃ t : ℝ, vector_form r k t x y) ↔
  r = -4 ∧ k = 6 := by
  sorry

end vector_form_equiv_line_equation_l502_50247


namespace sum_of_digits_l502_50246

/-- Given three-digit numbers of the form 4a5 and 9b2, where a and b are single digits,
    if 4a5 + 457 = 9b2 and 9b2 is divisible by 11, then a + b = 4 -/
theorem sum_of_digits (a b : ℕ) : 
  (a < 10) →
  (b < 10) →
  (400 + 10 * a + 5 + 457 = 900 + 10 * b + 2) →
  (900 + 10 * b + 2) % 11 = 0 →
  a + b = 4 := by
sorry

end sum_of_digits_l502_50246


namespace geometric_series_proof_l502_50210

def geometric_series (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_proof 
  (a : ℚ) 
  (h_a : a = 4/7) 
  (r : ℚ) 
  (h_r : r = 4/7) :
  r = 4/7 ∧ geometric_series a r 3 = 372/343 := by
  sorry

end geometric_series_proof_l502_50210


namespace midpoint_x_coordinate_sum_l502_50207

theorem midpoint_x_coordinate_sum (a b c : ℝ) (h : a + b + c = 12) :
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 12 := by
  sorry

end midpoint_x_coordinate_sum_l502_50207


namespace ducks_arrived_later_l502_50219

theorem ducks_arrived_later (initial_ducks : ℕ) (initial_geese : ℕ) (final_ducks : ℕ) (final_geese : ℕ) : 
  initial_ducks = 25 →
  initial_geese = 2 * initial_ducks - 10 →
  final_geese = initial_geese - (15 - 5) →
  final_geese = final_ducks + 1 →
  final_ducks - initial_ducks = 4 :=
by sorry

end ducks_arrived_later_l502_50219


namespace max_sections_five_lines_l502_50235

/-- The number of sections created by n line segments in a rectangle,
    where each new line intersects all previous lines -/
def num_sections (n : ℕ) : ℕ :=
  1 + (List.range n).sum

/-- The property that each new line intersects all previous lines -/
def intersects_all_previous (n : ℕ) : Prop :=
  ∀ k, k < n → num_sections k < num_sections (k + 1)

theorem max_sections_five_lines :
  intersects_all_previous 5 →
  num_sections 5 = 16 :=
by sorry

end max_sections_five_lines_l502_50235


namespace y_coordinate_difference_zero_l502_50201

/-- Given two points (m, n) and (m + 3, n + q) on the line x = (y / 7) - (2 / 5),
    the difference between their y-coordinates is 0. -/
theorem y_coordinate_difference_zero
  (m n q : ℚ) : 
  (m = n / 7 - 2 / 5) →
  (m + 3 = (n + q) / 7 - 2 / 5) →
  q = 0 :=
by sorry

end y_coordinate_difference_zero_l502_50201


namespace circle_equation_k_value_l502_50298

theorem circle_equation_k_value :
  ∀ k : ℝ,
  (∀ x y : ℝ, x^2 + 8*x + y^2 + 10*y - k = 0 ↔ (x + 4)^2 + (y + 5)^2 = 100) →
  k = 59 :=
by
  sorry

end circle_equation_k_value_l502_50298


namespace system_of_equations_solution_system_of_inequalities_solution_l502_50230

-- System of equations
theorem system_of_equations_solution :
  ∃! (x y : ℝ), x + 2*y = 7 ∧ 3*x + y = 6 ∧ x = 1 ∧ y = 3 := by sorry

-- System of inequalities
theorem system_of_inequalities_solution :
  ∀ x : ℝ, (2*(x - 1) + 1 > -3 ∧ x - 1 ≤ (1 + x) / 3) ↔ (-1 < x ∧ x ≤ 2) := by sorry

end system_of_equations_solution_system_of_inequalities_solution_l502_50230


namespace solution_x_fourth_plus_81_l502_50224

theorem solution_x_fourth_plus_81 :
  let solutions : List ℂ := [
    Complex.mk ((3 * Real.sqrt 2) / 2) ((3 * Real.sqrt 2) / 2),
    Complex.mk (-(3 * Real.sqrt 2) / 2) (-(3 * Real.sqrt 2) / 2),
    Complex.mk (-(3 * Real.sqrt 2) / 2) ((3 * Real.sqrt 2) / 2),
    Complex.mk ((3 * Real.sqrt 2) / 2) (-(3 * Real.sqrt 2) / 2)
  ]
  ∀ z : ℂ, z^4 + 81 = 0 ↔ z ∈ solutions := by
sorry

end solution_x_fourth_plus_81_l502_50224


namespace cube_sum_implies_sum_bound_l502_50231

theorem cube_sum_implies_sum_bound (p q : ℝ) (h : p^3 + q^3 = 2) : p + q ≤ 2 := by
  sorry

end cube_sum_implies_sum_bound_l502_50231


namespace same_color_probability_l502_50292

def total_plates : ℕ := 11
def red_plates : ℕ := 6
def green_plates : ℕ := 5
def plates_selected : ℕ := 3

theorem same_color_probability :
  (Nat.choose red_plates plates_selected + Nat.choose green_plates plates_selected) /
  Nat.choose total_plates plates_selected = 2 / 11 := by
  sorry

end same_color_probability_l502_50292


namespace sqrt_fraction_equivalence_l502_50268

theorem sqrt_fraction_equivalence (x : ℝ) (h : x < -2) :
  Real.sqrt (x / (1 + (x + 1) / (x + 2))) = -x := by sorry

end sqrt_fraction_equivalence_l502_50268


namespace total_travel_time_l502_50294

def station_distance : ℕ := 2 -- hours
def break_time : ℕ := 30 -- minutes

theorem total_travel_time :
  let travel_time_between_stations := station_distance * 60 -- convert hours to minutes
  let total_travel_time := 2 * travel_time_between_stations + break_time
  total_travel_time = 270 := by
sorry

end total_travel_time_l502_50294


namespace ferry_time_difference_l502_50225

/-- Represents the properties of a ferry --/
structure Ferry where
  speed : ℝ
  time : ℝ
  distance : ℝ

/-- The problem setup for the two ferries --/
def ferryProblem : Prop :=
  ∃ (P Q : Ferry),
    P.speed = 6 ∧
    P.time = 3 ∧
    P.distance = P.speed * P.time ∧
    Q.distance = 3 * P.distance ∧
    Q.speed = P.speed + 3 ∧
    Q.time = Q.distance / Q.speed ∧
    Q.time - P.time = 3

/-- The main theorem to be proved --/
theorem ferry_time_difference : ferryProblem :=
sorry

end ferry_time_difference_l502_50225


namespace elois_banana_bread_l502_50245

/-- Represents the number of loaves of banana bread Elois made on Monday -/
def monday_loaves : ℕ := sorry

/-- Represents the number of loaves of banana bread Elois made on Tuesday -/
def tuesday_loaves : ℕ := 2 * monday_loaves

/-- Represents the total number of bananas used for both days -/
def total_bananas : ℕ := 36

/-- Represents the number of bananas needed to make one loaf of banana bread -/
def bananas_per_loaf : ℕ := 4

theorem elois_banana_bread :
  monday_loaves = 3 ∧
  tuesday_loaves = 2 * monday_loaves ∧
  total_bananas = bananas_per_loaf * (monday_loaves + tuesday_loaves) :=
sorry

end elois_banana_bread_l502_50245


namespace fourth_root_equality_l502_50253

theorem fourth_root_equality (x : ℝ) (hx : x > 0) : 
  (x * (x^3)^(1/4))^(1/4) = x^(7/16) := by sorry

end fourth_root_equality_l502_50253


namespace vector_sum_closed_polygon_l502_50202

variable {V : Type*} [AddCommGroup V]

/-- Given vectors AB, CF, BC, and FA in a vector space V, 
    their sum is equal to the zero vector. -/
theorem vector_sum_closed_polygon (AB CF BC FA : V) :
  AB + CF + BC + FA = (0 : V) := by
  sorry

end vector_sum_closed_polygon_l502_50202


namespace hyperbola_equation_l502_50243

/-- Given a hyperbola with the following properties:
  1. Its equation is of the form x²/a² - y²/b² = 1 where a > 0 and b > 0
  2. It has an asymptote parallel to the line x + 2y + 5 = 0
  3. One of its foci lies on the line x + 2y + 5 = 0
  Prove that its equation is x²/20 - y²/5 = 1 -/
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : ∃ k, k ≠ 0 ∧ b / a = 1 / 2 * k) -- Asymptote parallel condition
  (h4 : ∃ x y, x + 2*y + 5 = 0 ∧ (x - a)^2 / a^2 + y^2 / b^2 = 1) -- Focus on line condition
  : a^2 = 20 ∧ b^2 = 5 := by
  sorry

end hyperbola_equation_l502_50243


namespace rook_placement_impossibility_l502_50295

theorem rook_placement_impossibility :
  ∀ (r b g : ℕ),
  r + b + g = 50 →
  2 * r ≤ b →
  2 * b ≤ g →
  2 * g ≤ r →
  False :=
by sorry

end rook_placement_impossibility_l502_50295


namespace tank_capacity_ratio_l502_50270

theorem tank_capacity_ratio : 
  let h_a : ℝ := 8
  let c_a : ℝ := 8
  let h_b : ℝ := 8
  let c_b : ℝ := 10
  let r_a : ℝ := c_a / (2 * Real.pi)
  let r_b : ℝ := c_b / (2 * Real.pi)
  let v_a : ℝ := Real.pi * r_a^2 * h_a
  let v_b : ℝ := Real.pi * r_b^2 * h_b
  v_a / v_b = 0.64
  := by sorry

end tank_capacity_ratio_l502_50270


namespace decreasing_quadratic_range_l502_50223

/-- A quadratic function f(x) with parameter a. -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

/-- The theorem stating that if f(x) is decreasing on (-∞, 4], then a ≤ -3. -/
theorem decreasing_quadratic_range (a : ℝ) :
  (∀ x ≤ 4, ∀ y ≤ 4, x < y → f a x > f a y) →
  a ≤ -3 :=
sorry

end decreasing_quadratic_range_l502_50223


namespace unique_valid_number_l502_50274

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n ≤ 9999 ∧
  (∀ i : ℕ, i < 3 → (n / 10^i % 10 + n / 10^(i+1) % 10) ≤ 2) ∧
  (∀ i : ℕ, i < 2 → (n / 10^i % 10 + n / 10^(i+1) % 10 + n / 10^(i+2) % 10) ≥ 3)

theorem unique_valid_number :
  ∃! n : ℕ, is_valid_number n :=
sorry

end unique_valid_number_l502_50274


namespace amount_with_r_l502_50291

/-- Given a total amount shared among three parties where one party has
    two-thirds of the combined amount of the other two, this function
    calculates the amount held by the third party. -/
def calculate_third_party_amount (total : ℚ) : ℚ :=
  (2 / 3) * (3 / 5) * total

/-- Theorem stating that given the problem conditions, 
    the amount held by r is 3200. -/
theorem amount_with_r (total : ℚ) (h_total : total = 8000) :
  calculate_third_party_amount total = 3200 := by
  sorry

#eval calculate_third_party_amount 8000

end amount_with_r_l502_50291
