import Mathlib

namespace NUMINAMATH_CALUDE_two_tower_100_gt_3_three_tower_100_gt_three_tower_99_three_tower_100_gt_four_tower_99_l1092_109299

-- Define a function to represent the power tower
def powerTower (base : ℕ) (height : ℕ) : ℕ :=
  match height with
  | 0 => 1
  | n + 1 => base ^ (powerTower base n)

-- Theorem 1
theorem two_tower_100_gt_3 : powerTower 2 100 > 3 := by sorry

-- Theorem 2
theorem three_tower_100_gt_three_tower_99 : powerTower 3 100 > powerTower 3 99 := by sorry

-- Theorem 3
theorem three_tower_100_gt_four_tower_99 : powerTower 3 100 > powerTower 4 99 := by sorry

end NUMINAMATH_CALUDE_two_tower_100_gt_3_three_tower_100_gt_three_tower_99_three_tower_100_gt_four_tower_99_l1092_109299


namespace NUMINAMATH_CALUDE_tailor_cuts_difference_l1092_109222

theorem tailor_cuts_difference : 
  (7/8 + 11/12) - (5/6 + 3/4) = 5/24 := by
  sorry

end NUMINAMATH_CALUDE_tailor_cuts_difference_l1092_109222


namespace NUMINAMATH_CALUDE_set_equality_implies_difference_l1092_109233

theorem set_equality_implies_difference (a b : ℝ) : 
  ({1, a + b, a} : Set ℝ) = {0, b / a, b} → b - a = 2 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_difference_l1092_109233


namespace NUMINAMATH_CALUDE_unique_function_satisfying_conditions_l1092_109276

theorem unique_function_satisfying_conditions
  (f : ℕ → ℕ → ℕ)
  (h1 : ∀ a b c : ℕ, f (Nat.gcd a b) c = Nat.gcd a (f c b))
  (h2 : ∀ a : ℕ, f a a ≥ a) :
  ∀ a b : ℕ, f a b = Nat.gcd a b :=
by sorry

end NUMINAMATH_CALUDE_unique_function_satisfying_conditions_l1092_109276


namespace NUMINAMATH_CALUDE_evenOnesTableCountTheorem_l1092_109238

/-- The number of ways to fill an m × n table with zeros and ones,
    such that there is an even number of ones in every row and every column -/
def evenOnesTableCount (m n : ℕ) : ℕ :=
  2^((m-1)*(n-1))

/-- Theorem: The number of ways to fill an m × n table with zeros and ones,
    such that there is an even number of ones in every row and every column,
    is equal to 2^((m-1)(n-1)) -/
theorem evenOnesTableCountTheorem (m n : ℕ) :
  evenOnesTableCount m n = 2^((m-1)*(n-1)) := by
  sorry


end NUMINAMATH_CALUDE_evenOnesTableCountTheorem_l1092_109238


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1092_109201

theorem sufficient_not_necessary_condition :
  (∃ x : ℝ, x^2 + x = 0 ∧ x ≠ -1) ∧
  (∀ x : ℝ, x = -1 → x^2 + x = 0) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1092_109201


namespace NUMINAMATH_CALUDE_pipe_A_fill_time_l1092_109246

-- Define the time it takes for Pipe A to fill the tank
variable (A : ℝ)

-- Define the time it takes for Pipe B to empty the tank
def B : ℝ := 24

-- Define the total time to fill the tank when both pipes are used
def total_time : ℝ := 30

-- Define the time Pipe B is open
def B_open_time : ℝ := 24

-- Define the theorem
theorem pipe_A_fill_time :
  (1 / A - 1 / B) * B_open_time + (1 / A) * (total_time - B_open_time) = 1 →
  A = 15 := by
sorry

end NUMINAMATH_CALUDE_pipe_A_fill_time_l1092_109246


namespace NUMINAMATH_CALUDE_complex_sum_problem_l1092_109229

/-- Given three complex numbers and conditions, prove that s+u = -1 -/
theorem complex_sum_problem (p q r s t u : ℝ) : 
  q = 5 →
  t = -p - r →
  (p + q * I) + (r + s * I) + (t + u * I) = 4 * I →
  s + u = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_problem_l1092_109229


namespace NUMINAMATH_CALUDE_beaker_liquid_distribution_l1092_109236

/-- Proves that if 5 ml of liquid is removed from a beaker and 35 ml remains, 
    then the original amount of liquid would have been 8 ml per cup if equally 
    distributed among 5 cups. -/
theorem beaker_liquid_distribution (initial_volume : ℝ) : 
  initial_volume - 5 = 35 → initial_volume / 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_beaker_liquid_distribution_l1092_109236


namespace NUMINAMATH_CALUDE_q_divisibility_q_values_q_cubic_form_q_10_expression_l1092_109279

/-- A cubic polynomial q(x) such that [q(x)]^2 - x is divisible by (x - 2)(x + 2)(x - 5)(x - 7) -/
def q (x : ℝ) : ℝ := sorry

theorem q_divisibility (x : ℝ) : 
  ∃ k : ℝ, q x ^ 2 - x = k * ((x - 2) * (x + 2) * (x - 5) * (x - 7)) := sorry

theorem q_values : 
  q 2 = Real.sqrt 2 ∧ q (-2) = -Real.sqrt 2 ∧ q 5 = Real.sqrt 5 ∧ q 7 = Real.sqrt 7 := sorry

theorem q_cubic_form : 
  ∃ a b c d : ℝ, ∀ x : ℝ, q x = a * x^3 + b * x^2 + c * x + d := sorry

theorem q_10_expression (a b c d : ℝ) 
  (h : ∀ x : ℝ, q x = a * x^3 + b * x^2 + c * x + d) : 
  q 10 = 1000 * a + 100 * b + 10 * c + d := sorry

end NUMINAMATH_CALUDE_q_divisibility_q_values_q_cubic_form_q_10_expression_l1092_109279


namespace NUMINAMATH_CALUDE_general_laborer_pay_general_laborer_pay_is_90_l1092_109295

/-- The daily pay for general laborers given the following conditions:
  - There are 35 people hired in total
  - The total payroll is 3950 dollars
  - 19 of the hired people are general laborers
  - Heavy equipment operators are paid 140 dollars per day
-/
theorem general_laborer_pay (total_hired : ℕ) (total_payroll : ℕ) 
  (num_laborers : ℕ) (operator_pay : ℕ) : ℕ :=
  let num_operators := total_hired - num_laborers
  let operator_total_pay := num_operators * operator_pay
  let laborer_total_pay := total_payroll - operator_total_pay
  laborer_total_pay / num_laborers

/-- Proof that the daily pay for general laborers is 90 dollars -/
theorem general_laborer_pay_is_90 : 
  general_laborer_pay 35 3950 19 140 = 90 := by
  sorry

end NUMINAMATH_CALUDE_general_laborer_pay_general_laborer_pay_is_90_l1092_109295


namespace NUMINAMATH_CALUDE_hannahs_total_spend_l1092_109221

/-- The total cost of Hannah's purchase --/
def total_cost (num_sweatshirts num_tshirts sweatshirt_price tshirt_price : ℕ) : ℕ :=
  num_sweatshirts * sweatshirt_price + num_tshirts * tshirt_price

/-- Theorem stating that Hannah's total spend is $65 --/
theorem hannahs_total_spend :
  total_cost 3 2 15 10 = 65 := by
  sorry

end NUMINAMATH_CALUDE_hannahs_total_spend_l1092_109221


namespace NUMINAMATH_CALUDE_functions_same_domain_range_not_necessarily_equal_l1092_109258

theorem functions_same_domain_range_not_necessarily_equal :
  ∃ (A B : Type) (f g : A → B), (∀ x : A, ∃ y : B, f x = y ∧ g x = y) ∧ f ≠ g :=
sorry

end NUMINAMATH_CALUDE_functions_same_domain_range_not_necessarily_equal_l1092_109258


namespace NUMINAMATH_CALUDE_lagoon_island_male_alligators_l1092_109268

/-- Represents the population of alligators on Lagoon island -/
structure AlligatorPopulation where
  total : ℕ
  males : ℕ
  females : ℕ
  juvenileFemales : ℕ
  adultFemales : ℕ

/-- Conditions for the Lagoon island alligator population -/
def lagoonIslandConditions (pop : AlligatorPopulation) : Prop :=
  pop.males = pop.females ∧
  pop.females = pop.juvenileFemales + pop.adultFemales ∧
  pop.juvenileFemales = (2 * pop.females) / 5 ∧
  pop.adultFemales = 15

theorem lagoon_island_male_alligators (pop : AlligatorPopulation) 
  (h : lagoonIslandConditions pop) : 
  pop.males = pop.adultFemales / (3 : ℚ) / (10 : ℚ) := by
  sorry

#check lagoon_island_male_alligators

end NUMINAMATH_CALUDE_lagoon_island_male_alligators_l1092_109268


namespace NUMINAMATH_CALUDE_darla_total_cost_l1092_109289

/-- The total cost of electricity given the rate, usage, and late fee. -/
def total_cost (rate : ℝ) (usage : ℝ) (late_fee : ℝ) : ℝ :=
  rate * usage + late_fee

/-- Proof that Darla's total cost is $1350 -/
theorem darla_total_cost : 
  total_cost 4 300 150 = 1350 := by
  sorry

end NUMINAMATH_CALUDE_darla_total_cost_l1092_109289


namespace NUMINAMATH_CALUDE_gift_cost_equation_l1092_109227

/-- Represents the cost equation for Xiaofen's gift purchase -/
theorem gift_cost_equation (x : ℝ) : 
  (15 : ℝ) * (x + 2 * 20) = 900 ↔ 
  (∃ (total_cost num_gifts num_lollipops_per_gift lollipop_cost : ℝ),
    total_cost = 900 ∧
    num_gifts = 15 ∧
    num_lollipops_per_gift = 2 ∧
    lollipop_cost = 20 ∧
    total_cost = num_gifts * (x + num_lollipops_per_gift * lollipop_cost)) :=
by sorry

end NUMINAMATH_CALUDE_gift_cost_equation_l1092_109227


namespace NUMINAMATH_CALUDE_unique_integer_value_l1092_109248

theorem unique_integer_value : ∃! x : ℤ, 
  1 < x ∧ x < 9 ∧ 
  2 < x ∧ x < 15 ∧ 
  -1 < x ∧ x < 7 ∧ 
  0 < x ∧ x < 4 ∧ 
  x + 1 < 5 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_integer_value_l1092_109248


namespace NUMINAMATH_CALUDE_paper_folding_l1092_109240

theorem paper_folding (s : ℝ) (x : ℝ) :
  s^2 = 18 →
  (1/2) * x^2 = 18 - x^2 →
  ∃ d : ℝ, d = 2 * Real.sqrt 6 ∧ d^2 = 2 * x^2 :=
by sorry

end NUMINAMATH_CALUDE_paper_folding_l1092_109240


namespace NUMINAMATH_CALUDE_rectangle_area_excluding_hole_l1092_109242

variable (x : ℝ)

def large_rectangle_length : ℝ := 2 * x + 4
def large_rectangle_width : ℝ := x + 7
def hole_length : ℝ := x + 2
def hole_width : ℝ := 3 * x - 5

theorem rectangle_area_excluding_hole (h : x > 5/3) :
  large_rectangle_length x * large_rectangle_width x - hole_length x * hole_width x = -x^2 + 17*x + 38 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_excluding_hole_l1092_109242


namespace NUMINAMATH_CALUDE_project_hours_ratio_l1092_109259

/-- Represents the hours charged by Kate -/
def kate_hours : ℕ := sorry

/-- Represents the hours charged by Pat -/
def pat_hours : ℕ := 2 * kate_hours

/-- Represents the hours charged by Mark -/
def mark_hours : ℕ := kate_hours + 110

/-- The total hours charged by all three -/
def total_hours : ℕ := 198

theorem project_hours_ratio :
  pat_hours + kate_hours + mark_hours = total_hours ∧
  pat_hours.gcd mark_hours = pat_hours ∧
  (pat_hours / pat_hours.gcd mark_hours) = 1 ∧
  (mark_hours / pat_hours.gcd mark_hours) = 3 :=
sorry

end NUMINAMATH_CALUDE_project_hours_ratio_l1092_109259


namespace NUMINAMATH_CALUDE_solution_replacement_fraction_l1092_109280

theorem solution_replacement_fraction 
  (initial_concentration : Real)
  (replacement_concentration : Real)
  (final_concentration : Real)
  (h1 : initial_concentration = 0.40)
  (h2 : replacement_concentration = 0.25)
  (h3 : final_concentration = 0.35)
  : ∃ x : Real, x = 1/3 ∧ 
    final_concentration * 1 = 
    (initial_concentration * (1 - x)) + (replacement_concentration * x) := by
  sorry

end NUMINAMATH_CALUDE_solution_replacement_fraction_l1092_109280


namespace NUMINAMATH_CALUDE_exclusive_proposition_range_l1092_109264

-- Define the propositions p and q as functions of m
def p (m : ℝ) : Prop := ∃ (h k r : ℝ), ∀ (x y : ℝ), x^2 + y^2 - x + y + m = 0 ↔ (x - h)^2 + (y - k)^2 = r^2

def q (m : ℝ) : Prop := ∀ x : ℝ, m * x^2 - 4 * x + m > 0

-- Define the range of m
def m_range (m : ℝ) : Prop := m < 1/2 ∨ m > 2

-- Theorem statement
theorem exclusive_proposition_range :
  ∀ m : ℝ, (p m ∧ ¬q m) ∨ (¬p m ∧ q m) → m_range m :=
by sorry

end NUMINAMATH_CALUDE_exclusive_proposition_range_l1092_109264


namespace NUMINAMATH_CALUDE_inequality_problem_l1092_109256

theorem inequality_problem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : 1/a + 4/b + 9/c ≤ 36/(a + b + c)) : 
  (2*b + 3*c)/(a + b + c) = 13/6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_problem_l1092_109256


namespace NUMINAMATH_CALUDE_maria_spent_60_dollars_l1092_109249

def flower_cost : ℕ := 6
def roses_bought : ℕ := 7
def daisies_bought : ℕ := 3

theorem maria_spent_60_dollars : 
  (roses_bought + daisies_bought) * flower_cost = 60 := by
  sorry

end NUMINAMATH_CALUDE_maria_spent_60_dollars_l1092_109249


namespace NUMINAMATH_CALUDE_abc_product_l1092_109252

theorem abc_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a * (b + c) = 156) (h2 : b * (c + a) = 168) (h3 : c * (a + b) = 176) :
  a * b * c = 754 := by
sorry

end NUMINAMATH_CALUDE_abc_product_l1092_109252


namespace NUMINAMATH_CALUDE_partition_existence_l1092_109209

theorem partition_existence (p q : ℕ+) (h_coprime : Nat.Coprime p q) (h_neq : p ≠ q) :
  (∃ (A B C : Set ℕ+),
    (∀ z : ℕ+, (z ∈ A ∧ z + p ∈ B ∧ z + q ∈ C) ∨
               (z ∈ B ∧ z + p ∈ C ∧ z + q ∈ A) ∨
               (z ∈ C ∧ z + p ∈ A ∧ z + q ∈ B)) ∧
    (A ∪ B ∪ C = Set.univ) ∧
    (A ∩ B = ∅) ∧ (B ∩ C = ∅) ∧ (C ∩ A = ∅)) ↔
  (3 ∣ p + q) :=
sorry

end NUMINAMATH_CALUDE_partition_existence_l1092_109209


namespace NUMINAMATH_CALUDE_initial_weasels_count_l1092_109272

/-- Represents the number of weasels caught by one fox in one week -/
def weasels_per_fox_per_week : ℕ := 4

/-- Represents the number of rabbits caught by one fox in one week -/
def rabbits_per_fox_per_week : ℕ := 2

/-- Represents the number of foxes -/
def num_foxes : ℕ := 3

/-- Represents the number of weeks the foxes hunt -/
def num_weeks : ℕ := 3

/-- Represents the initial number of rabbits -/
def initial_rabbits : ℕ := 50

/-- Represents the number of rabbits and weasels left after hunting -/
def remaining_animals : ℕ := 96

/-- Theorem stating that the initial number of weasels is 100 -/
theorem initial_weasels_count : 
  ∃ (initial_weasels : ℕ), 
    initial_weasels = 100 ∧
    initial_weasels + initial_rabbits = 
      remaining_animals + 
      (weasels_per_fox_per_week * num_foxes * num_weeks) + 
      (rabbits_per_fox_per_week * num_foxes * num_weeks) := by
  sorry

end NUMINAMATH_CALUDE_initial_weasels_count_l1092_109272


namespace NUMINAMATH_CALUDE_largest_expression_l1092_109243

theorem largest_expression : 
  let a := 3 + 1 + 0 + 5
  let b := 3 * 1 + 0 + 5
  let c := 3 + 1 * 0 + 5
  let d := 3 * 1 + 0 * 5
  let e := 3 * 1 + 0 * 5 * 3
  (a > b) ∧ (a > c) ∧ (a > d) ∧ (a > e) := by
  sorry

end NUMINAMATH_CALUDE_largest_expression_l1092_109243


namespace NUMINAMATH_CALUDE_machine_output_for_68_l1092_109203

def number_machine (x : ℕ) : ℕ := x + 15 - 6

theorem machine_output_for_68 : number_machine 68 = 77 := by
  sorry

end NUMINAMATH_CALUDE_machine_output_for_68_l1092_109203


namespace NUMINAMATH_CALUDE_water_pumped_in_30_min_l1092_109286

/-- 
Given a pump that operates at a rate of 540 gallons per hour, 
this theorem proves that the volume of water pumped in 30 minutes is 270 gallons.
-/
theorem water_pumped_in_30_min (pump_rate : ℝ) (time : ℝ) : 
  pump_rate = 540 → time = 0.5 → pump_rate * time = 270 := by
  sorry

#check water_pumped_in_30_min

end NUMINAMATH_CALUDE_water_pumped_in_30_min_l1092_109286


namespace NUMINAMATH_CALUDE_twenty_six_billion_scientific_notation_l1092_109262

/-- Scientific notation representation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem twenty_six_billion_scientific_notation :
  toScientificNotation (26 * 10^9) = ScientificNotation.mk 2.6 9 sorry := by
  sorry

end NUMINAMATH_CALUDE_twenty_six_billion_scientific_notation_l1092_109262


namespace NUMINAMATH_CALUDE_wechat_group_size_l1092_109284

theorem wechat_group_size :
  ∃ (n : ℕ), n > 0 ∧ n * (n - 1) / 2 = 72 ∧ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_wechat_group_size_l1092_109284


namespace NUMINAMATH_CALUDE_cosine_sum_specific_angles_l1092_109224

theorem cosine_sum_specific_angles : 
  Real.cos (π/3) * Real.cos (π/6) - Real.sin (π/3) * Real.sin (π/6) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_specific_angles_l1092_109224


namespace NUMINAMATH_CALUDE_extreme_value_zero_not_necessary_nor_sufficient_l1092_109218

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Define the property of having an extreme value at a point
def has_extreme_value_at (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ ε > 0, ∀ x ∈ Set.Ioo (x₀ - ε) (x₀ + ε), f x₀ ≤ f x ∨ f x ≤ f x₀

-- State the theorem
theorem extreme_value_zero_not_necessary_nor_sufficient :
  ∃ (f : ℝ → ℝ) (x₀ : ℝ), Differentiable ℝ f ∧
    (has_extreme_value_at f x₀ ∧ f x₀ ≠ 0) ∧
    (f x₀ = 0 ∧ ¬(has_extreme_value_at f x₀)) :=
  sorry

end NUMINAMATH_CALUDE_extreme_value_zero_not_necessary_nor_sufficient_l1092_109218


namespace NUMINAMATH_CALUDE_unique_solution_of_equation_l1092_109251

theorem unique_solution_of_equation : 
  ∃! x : ℝ, (x^3 + 2*x^2) / (x^2 + 3*x + 2) + x = -6 ∧ x ≠ -2 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_of_equation_l1092_109251


namespace NUMINAMATH_CALUDE_problem_solution_l1092_109215

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- State the theorem
theorem problem_solution (a : ℝ) (m : ℝ) : 
  (∀ x : ℝ, f a x ≤ 3 ↔ x ∈ Set.Icc (-6) 0) →
  (a = -3 ∧ 
   (∀ x : ℝ, f a x + f a (x + 5) ≥ 2 * m) → m ≤ 5/2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1092_109215


namespace NUMINAMATH_CALUDE_max_coins_distribution_l1092_109216

theorem max_coins_distribution (n : ℕ) : 
  n < 150 ∧ 
  ∃ k : ℕ, n = 13 * k + 3 →
  n ≤ 146 :=
by sorry

end NUMINAMATH_CALUDE_max_coins_distribution_l1092_109216


namespace NUMINAMATH_CALUDE_hexacontagon_triangles_l1092_109235

/-- The number of sides in a regular hexacontagon -/
def n : ℕ := 60

/-- The number of triangles that can be formed using the vertices of a regular hexacontagon,
    without using any three consecutive vertices -/
def num_triangles : ℕ := Nat.choose n 3 - n

theorem hexacontagon_triangles : num_triangles = 34160 := by
  sorry

end NUMINAMATH_CALUDE_hexacontagon_triangles_l1092_109235


namespace NUMINAMATH_CALUDE_ab_one_sufficient_not_necessary_l1092_109220

theorem ab_one_sufficient_not_necessary (a b : ℝ) : 
  (a * b = 1 → a^2 + b^2 ≥ 2) ∧ 
  ∃ a b : ℝ, a^2 + b^2 ≥ 2 ∧ a * b ≠ 1 :=
sorry

end NUMINAMATH_CALUDE_ab_one_sufficient_not_necessary_l1092_109220


namespace NUMINAMATH_CALUDE_joshua_total_cars_l1092_109281

/-- The total number of toy cars Joshua has -/
def total_cars (box1 box2 box3 : ℕ) : ℕ := box1 + box2 + box3

/-- Theorem: Joshua has 71 toy cars in total -/
theorem joshua_total_cars :
  total_cars 21 31 19 = 71 := by
  sorry

end NUMINAMATH_CALUDE_joshua_total_cars_l1092_109281


namespace NUMINAMATH_CALUDE_stone_piles_sum_l1092_109232

/-- Represents the number of stones in each pile -/
structure StonePiles where
  pile1 : ℕ
  pile2 : ℕ
  pile3 : ℕ
  pile4 : ℕ
  pile5 : ℕ

/-- Conditions for the stone piles problem -/
def validStonePiles (p : StonePiles) : Prop :=
  p.pile5 = 6 * p.pile3 ∧
  p.pile2 = 2 * (p.pile3 + p.pile5) ∧
  p.pile1 = p.pile5 / 3 ∧
  p.pile1 = p.pile4 - 10 ∧
  p.pile4 = p.pile2 / 2

theorem stone_piles_sum (p : StonePiles) (h : validStonePiles p) :
  p.pile1 + p.pile2 + p.pile3 + p.pile4 + p.pile5 = 60 := by
  sorry

#check stone_piles_sum

end NUMINAMATH_CALUDE_stone_piles_sum_l1092_109232


namespace NUMINAMATH_CALUDE_expression_value_l1092_109294

theorem expression_value (m : ℝ) (h : 1 / (m - 2) = 1) : 2 / (m - 2) - m + 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1092_109294


namespace NUMINAMATH_CALUDE_gas_diffusion_rate_and_molar_mass_l1092_109205

theorem gas_diffusion_rate_and_molar_mass 
  (r_unknown r_O2 : ℝ) 
  (M_unknown M_O2 : ℝ) 
  (h1 : r_unknown / r_O2 = 1 / 3) 
  (h2 : r_unknown / r_O2 = Real.sqrt (M_O2 / M_unknown)) :
  M_unknown = 9 * M_O2 := by
  sorry

end NUMINAMATH_CALUDE_gas_diffusion_rate_and_molar_mass_l1092_109205


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1092_109228

theorem solution_set_inequality (x : ℝ) : 
  (x + 3) * (1 - x) ≥ 0 ↔ -3 ≤ x ∧ x ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1092_109228


namespace NUMINAMATH_CALUDE_profit_ratio_theorem_l1092_109274

/-- Represents the investment and profit information for a partner -/
structure Partner where
  investment : ℕ
  months : ℕ

/-- Calculates the profit factor for a partner -/
def profitFactor (p : Partner) : ℕ := p.investment * p.months

/-- Theorem stating the profit ratio of two partners given their investments and time periods -/
theorem profit_ratio_theorem (p q : Partner) 
  (h_investment_ratio : p.investment * 5 = q.investment * 7)
  (h_p_months : p.months = 5)
  (h_q_months : q.months = 9) :
  profitFactor p * 9 = profitFactor q * 7 := by
  sorry

#check profit_ratio_theorem

end NUMINAMATH_CALUDE_profit_ratio_theorem_l1092_109274


namespace NUMINAMATH_CALUDE_emily_meal_combinations_l1092_109234

/-- The number of protein options available --/
def num_proteins : ℕ := 4

/-- The number of side options available --/
def num_sides : ℕ := 5

/-- The number of dessert options available --/
def num_desserts : ℕ := 5

/-- The number of sides Emily must choose --/
def sides_to_choose : ℕ := 3

/-- The total number of different meal combinations Emily can choose --/
def total_meals : ℕ := num_proteins * (num_sides.choose sides_to_choose) * num_desserts

theorem emily_meal_combinations :
  total_meals = 200 :=
sorry

end NUMINAMATH_CALUDE_emily_meal_combinations_l1092_109234


namespace NUMINAMATH_CALUDE_equation_unique_solution_l1092_109297

theorem equation_unique_solution :
  ∃! x : ℝ, (3 * x^2 - 18 * x) / (x^2 - 6 * x) = x^2 - 4 * x + 3 ∧ x ≠ 0 ∧ x ≠ 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_unique_solution_l1092_109297


namespace NUMINAMATH_CALUDE_high_school_sampling_l1092_109273

/-- Represents a stratified sampling scenario in a high school -/
structure StratifiedSampling where
  total_students : ℕ
  freshmen : ℕ
  sampled_freshmen : ℕ

/-- Calculates the total number of students to be sampled in a stratified sampling scenario -/
def total_sampled (s : StratifiedSampling) : ℚ :=
  (s.total_students : ℚ) * s.sampled_freshmen / s.freshmen

/-- Theorem stating that for the given high school scenario, the total number of students
    to be sampled is 80 -/
theorem high_school_sampling :
  let s : StratifiedSampling := { total_students := 2400, freshmen := 600, sampled_freshmen := 20 }
  total_sampled s = 80 := by
  sorry


end NUMINAMATH_CALUDE_high_school_sampling_l1092_109273


namespace NUMINAMATH_CALUDE_six_meter_logs_more_efficient_l1092_109271

/-- Represents the number of pieces obtained from a log of given length -/
def pieces_from_log (log_length : ℕ) : ℕ := log_length

/-- Represents the number of cuts needed to divide a log into 1-meter pieces -/
def cuts_for_log (log_length : ℕ) : ℕ := log_length - 1

/-- Represents the efficiency of cutting a log, measured as pieces per cut -/
def cutting_efficiency (log_length : ℕ) : ℚ :=
  (pieces_from_log log_length : ℚ) / (cuts_for_log log_length : ℚ)

/-- Theorem stating that 6-meter logs are more efficient to cut than 8-meter logs -/
theorem six_meter_logs_more_efficient :
  cutting_efficiency 6 > cutting_efficiency 8 := by
  sorry

end NUMINAMATH_CALUDE_six_meter_logs_more_efficient_l1092_109271


namespace NUMINAMATH_CALUDE_cosine_law_acute_triangle_condition_l1092_109225

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  -- Assume the triangle is valid
  valid : a > 0 ∧ b > 0 ∧ c > 0 ∧ A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π

-- Theorem 1: c = a cos B + b cos A
theorem cosine_law (t : Triangle) : t.c = t.a * Real.cos t.B + t.b * Real.cos t.A := by
  sorry

-- Theorem 2: If a³ + b³ = c³, then a² + b² > c²
theorem acute_triangle_condition (t : Triangle) : 
  t.a^3 + t.b^3 = t.c^3 → t.a^2 + t.b^2 > t.c^2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_law_acute_triangle_condition_l1092_109225


namespace NUMINAMATH_CALUDE_floor_minus_self_unique_solution_l1092_109231

theorem floor_minus_self_unique_solution :
  ∃! s : ℝ, ⌊s⌋ - s = -10.3 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_floor_minus_self_unique_solution_l1092_109231


namespace NUMINAMATH_CALUDE_no_common_terms_except_one_l1092_109207

def x : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | (n + 2) => x (n + 1) + 2 * x n

def y : ℕ → ℤ
  | 0 => 1
  | 1 => 7
  | (n + 2) => 2 * y (n + 1) + 3 * y n

theorem no_common_terms_except_one : ∀ n : ℕ, x n = y n → x n = 1 := by sorry

end NUMINAMATH_CALUDE_no_common_terms_except_one_l1092_109207


namespace NUMINAMATH_CALUDE_saturday_zoo_visitors_l1092_109204

theorem saturday_zoo_visitors (friday_visitors : ℕ) (saturday_multiplier : ℕ) : 
  friday_visitors = 1250 →
  saturday_multiplier = 3 →
  friday_visitors * saturday_multiplier = 3750 :=
by
  sorry

end NUMINAMATH_CALUDE_saturday_zoo_visitors_l1092_109204


namespace NUMINAMATH_CALUDE_inscribed_square_area_l1092_109287

/-- The area of a square inscribed in a right triangle with hypotenuse 100 units and one leg 35 units -/
theorem inscribed_square_area (h : ℝ) (l : ℝ) (s : ℝ) 
  (hyp : h = 100)  -- hypotenuse length
  (leg : l = 35)   -- one leg length
  (square : s^2 = l * (h - l)) : -- s is the side length of the inscribed square
  s^2 = 2275 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l1092_109287


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l1092_109277

theorem fraction_sum_equality (a b c : ℝ) 
  (h : a / (36 - a) + b / (49 - b) + c / (81 - c) = 9) :
  6 / (36 - a) + 7 / (49 - b) + 9 / (81 - c) = 5047 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l1092_109277


namespace NUMINAMATH_CALUDE_remainder_equality_l1092_109275

theorem remainder_equality (A A' D S S' s s' : ℕ) 
  (h1 : A > A')
  (h2 : S = A % D)
  (h3 : S' = A' % D)
  (h4 : s = (A + A') % D)
  (h5 : s' = (S + S') % D) :
  s = s' :=
sorry

end NUMINAMATH_CALUDE_remainder_equality_l1092_109275


namespace NUMINAMATH_CALUDE_cube_inequality_l1092_109223

theorem cube_inequality (a b : ℝ) (h : a > b) : a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_inequality_l1092_109223


namespace NUMINAMATH_CALUDE_d_negative_iff_b_decreasing_l1092_109255

/-- An arithmetic sequence with common difference d -/
def arithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

/-- The sequence b_n defined as 2^(a_n) -/
def bSequence (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n, b n = 2^(a n)

/-- A decreasing sequence -/
def isDecreasing (b : ℕ → ℝ) : Prop :=
  ∀ n, b (n + 1) < b n

theorem d_negative_iff_b_decreasing
  (a : ℕ → ℝ) (b : ℕ → ℝ) (d : ℝ)
  (h1 : arithmeticSequence a d)
  (h2 : bSequence a b) :
  d < 0 ↔ isDecreasing b :=
sorry

end NUMINAMATH_CALUDE_d_negative_iff_b_decreasing_l1092_109255


namespace NUMINAMATH_CALUDE_intersection_M_N_l1092_109230

def M : Set ℝ := {x : ℝ | x^2 + 3*x = 0}
def N : Set ℝ := {3, 0}

theorem intersection_M_N : M ∩ N = {0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1092_109230


namespace NUMINAMATH_CALUDE_greatest_M_inequality_l1092_109254

theorem greatest_M_inequality (x y z : ℝ) : 
  ∃ (M : ℝ), M = 2/3 ∧ 
  (∀ (N : ℝ), (∀ (a b c : ℝ), a^4 + b^4 + c^4 + a*b*c*(a + b + c) ≥ N*(a*b + b*c + c*a)^2) → N ≤ M) ∧
  x^4 + y^4 + z^4 + x*y*z*(x + y + z) ≥ M*(x*y + y*z + z*x)^2 := by
  sorry

end NUMINAMATH_CALUDE_greatest_M_inequality_l1092_109254


namespace NUMINAMATH_CALUDE_polynomial_value_for_special_x_l1092_109257

theorem polynomial_value_for_special_x :
  let x : ℝ := 1 / (2 - Real.sqrt 3)
  x^6 - 2 * Real.sqrt 3 * x^5 - x^4 + x^3 - 4 * x^2 + 2 * x - Real.sqrt 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_for_special_x_l1092_109257


namespace NUMINAMATH_CALUDE_tom_car_distribution_l1092_109210

theorem tom_car_distribution (total_packages : ℕ) (cars_per_package : ℕ) (num_nephews : ℕ) (cars_remaining : ℕ) :
  total_packages = 10 →
  cars_per_package = 5 →
  num_nephews = 2 →
  cars_remaining = 30 →
  (total_packages * cars_per_package - cars_remaining) / (num_nephews * (total_packages * cars_per_package)) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_tom_car_distribution_l1092_109210


namespace NUMINAMATH_CALUDE_inscribed_circles_radii_sum_l1092_109202

theorem inscribed_circles_radii_sum (D : ℝ) (r₁ r₂ : ℝ) : 
  D = 23 → r₁ > 0 → r₂ > 0 → r₁ + r₂ = D / 2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circles_radii_sum_l1092_109202


namespace NUMINAMATH_CALUDE_prism_square_intersection_angle_l1092_109226

theorem prism_square_intersection_angle (d : ℝ) (h : d > 0) : 
  let rhombus_acute_angle : ℝ := 60 * π / 180
  let rhombus_diagonal : ℝ := d * Real.sqrt 3
  let intersection_angle : ℝ := Real.arccos (Real.sqrt 3 / 3)
  intersection_angle = Real.arccos (d / rhombus_diagonal) :=
by sorry

end NUMINAMATH_CALUDE_prism_square_intersection_angle_l1092_109226


namespace NUMINAMATH_CALUDE_probability_factor_less_than_10_l1092_109217

def factors_of_90 : Finset ℕ := {1, 2, 3, 5, 6, 9, 10, 15, 18, 30, 45, 90}

def factors_less_than_10 : Finset ℕ := {1, 2, 3, 5, 6, 9}

theorem probability_factor_less_than_10 : 
  (factors_less_than_10.card : ℚ) / factors_of_90.card = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_probability_factor_less_than_10_l1092_109217


namespace NUMINAMATH_CALUDE_line_not_in_second_quadrant_l1092_109290

/-- A line does not pass through the second quadrant if and only if
    its slope is non-negative and its y-intercept is non-positive -/
def not_in_second_quadrant (a b c : ℝ) : Prop :=
  a ≥ 0 ∧ c ≤ 0

theorem line_not_in_second_quadrant (t : ℝ) :
  not_in_second_quadrant (2*t - 3) 2 t → 0 ≤ t ∧ t ≤ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_line_not_in_second_quadrant_l1092_109290


namespace NUMINAMATH_CALUDE_cistern_wet_surface_area_l1092_109219

/-- Calculates the total wet surface area of a rectangular cistern -/
def total_wet_surface_area (length width depth : ℝ) : ℝ :=
  length * width + 2 * (length * depth) + 2 * (width * depth)

/-- Theorem stating the total wet surface area of the given cistern -/
theorem cistern_wet_surface_area :
  total_wet_surface_area 4 8 1.25 = 62 := by
  sorry

end NUMINAMATH_CALUDE_cistern_wet_surface_area_l1092_109219


namespace NUMINAMATH_CALUDE_tiffany_bags_l1092_109208

/-- The number of bags found the next day -/
def bags_found (initial_bags total_bags : ℕ) : ℕ := total_bags - initial_bags

/-- Proof that Tiffany found 8 bags the next day -/
theorem tiffany_bags : bags_found 4 12 = 8 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_bags_l1092_109208


namespace NUMINAMATH_CALUDE_job_completion_time_l1092_109293

/-- Calculates the remaining days to complete a job given initial and additional workers -/
def remaining_days (initial_workers : ℕ) (initial_days : ℕ) (days_worked : ℕ) (additional_workers : ℕ) : ℚ :=
  let total_work := initial_workers * initial_days
  let work_done := initial_workers * days_worked
  let remaining_work := total_work - work_done
  let total_workers := initial_workers + additional_workers
  remaining_work / total_workers

theorem job_completion_time : remaining_days 6 8 3 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l1092_109293


namespace NUMINAMATH_CALUDE_quadratic_inequality_solutions_l1092_109269

theorem quadratic_inequality_solutions (d : ℝ) :
  d > 0 →
  (∃ x : ℝ, x^2 - 8*x + d < 0) ↔ d < 16 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solutions_l1092_109269


namespace NUMINAMATH_CALUDE_base_conversion_product_l1092_109282

/-- Converts a number from base b to base 10 --/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * b^i) 0

/-- The problem statement --/
theorem base_conversion_product : 
  let numerator1 := to_base_10 [2, 6, 2] 8
  let denominator1 := to_base_10 [1, 3] 4
  let numerator2 := to_base_10 [1, 4, 4] 7
  let denominator2 := to_base_10 [2, 4] 5
  (numerator1 * numerator2) / (denominator1 * denominator2) = 147 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_product_l1092_109282


namespace NUMINAMATH_CALUDE_initial_courses_is_three_l1092_109260

/-- Represents the wall construction problem -/
def WallProblem (initial_courses : ℕ) : Prop :=
  let bricks_per_course : ℕ := 400
  let added_courses : ℕ := 2
  let removed_bricks : ℕ := 200
  let total_bricks : ℕ := 1800
  (initial_courses * bricks_per_course) + (added_courses * bricks_per_course) - removed_bricks = total_bricks

/-- Theorem stating that the initial number of courses is 3 -/
theorem initial_courses_is_three : WallProblem 3 := by
  sorry

end NUMINAMATH_CALUDE_initial_courses_is_three_l1092_109260


namespace NUMINAMATH_CALUDE_dataset_mode_l1092_109298

def dataset : List ℕ := [24, 23, 24, 25, 22]

def mode (l : List ℕ) : ℕ :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem dataset_mode : mode dataset = 24 := by
  sorry

end NUMINAMATH_CALUDE_dataset_mode_l1092_109298


namespace NUMINAMATH_CALUDE_book_reading_time_l1092_109263

theorem book_reading_time (total_pages : ℕ) (planned_days : ℕ) (extra_pages : ℕ) 
    (h1 : total_pages = 960)
    (h2 : planned_days = 20)
    (h3 : extra_pages = 12) :
  (total_pages : ℚ) / ((total_pages / planned_days + extra_pages) : ℚ) = 16 := by
  sorry

end NUMINAMATH_CALUDE_book_reading_time_l1092_109263


namespace NUMINAMATH_CALUDE_tan_three_implies_sum_l1092_109266

theorem tan_three_implies_sum (θ : Real) (h : Real.tan θ = 3) :
  (1 - Real.cos θ) / Real.sin θ + Real.sin θ / (1 + Real.cos θ) = 2 * (Real.sqrt 10 - 1) / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_three_implies_sum_l1092_109266


namespace NUMINAMATH_CALUDE_f_properties_l1092_109241

noncomputable def f (x : ℝ) : ℝ := Real.cos (Real.pi / 2 - x) * Real.cos x + Real.sqrt 3 * Real.sin x ^ 2

theorem f_properties :
  -- Smallest positive period is π
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧ 
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T')) ∧
  -- Monotonically decreasing in [5π/12 + kπ, 11π/12 + kπ]
  (∀ (k : ℤ), StrictMonoOn f (Set.Icc (5 * Real.pi / 12 + k * Real.pi) (11 * Real.pi / 12 + k * Real.pi))) ∧
  -- Minimum and maximum values on [π/6, π/2]
  (∃ (x_min x_max : ℝ), x_min ∈ Set.Icc (Real.pi / 6) (Real.pi / 2) ∧
                        x_max ∈ Set.Icc (Real.pi / 6) (Real.pi / 2) ∧
                        (∀ (x : ℝ), x ∈ Set.Icc (Real.pi / 6) (Real.pi / 2) → 
                          f x_min ≤ f x ∧ f x ≤ f x_max) ∧
                        f x_min = Real.sqrt 3 / 2 ∧
                        f x_max = Real.sqrt 3 / 2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1092_109241


namespace NUMINAMATH_CALUDE_geometric_series_sum_l1092_109244

theorem geometric_series_sum : ∀ (a r : ℝ), 
  a = 2 → r = 2/3 → |r| < 1 → 
  ∑' n, a * r^n = 6 := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l1092_109244


namespace NUMINAMATH_CALUDE_binomial_five_one_l1092_109237

theorem binomial_five_one : (5 : ℕ).choose 1 = 5 := by sorry

end NUMINAMATH_CALUDE_binomial_five_one_l1092_109237


namespace NUMINAMATH_CALUDE_shipment_arrival_time_l1092_109296

/-- Calculates the number of days until a shipment arrives at a warehouse -/
def daysUntilArrival (daysSinceDeparture : ℕ) (navigationDays : ℕ) (customsDays : ℕ) (warehouseArrivalDay : ℕ) : ℕ :=
  let daysInPort := daysSinceDeparture - navigationDays
  let daysAfterCustoms := daysInPort - customsDays
  warehouseArrivalDay - daysAfterCustoms

theorem shipment_arrival_time :
  daysUntilArrival 30 21 4 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_shipment_arrival_time_l1092_109296


namespace NUMINAMATH_CALUDE_tenth_ring_squares_l1092_109288

/-- The number of unit squares in the nth ring around a 3x3 center block -/
def ring_squares (n : ℕ) : ℕ := 8 * n + 8

/-- The 10th ring around a 3x3 center block contains 88 unit squares -/
theorem tenth_ring_squares : ring_squares 10 = 88 := by sorry

end NUMINAMATH_CALUDE_tenth_ring_squares_l1092_109288


namespace NUMINAMATH_CALUDE_yogurt_expiration_probability_l1092_109239

def total_boxes : ℕ := 6
def expired_boxes : ℕ := 2
def selected_boxes : ℕ := 2

def probability_at_least_one_expired : ℚ := 3/5

theorem yogurt_expiration_probability :
  (Nat.choose total_boxes selected_boxes - Nat.choose (total_boxes - expired_boxes) selected_boxes) /
  Nat.choose total_boxes selected_boxes = probability_at_least_one_expired :=
sorry

end NUMINAMATH_CALUDE_yogurt_expiration_probability_l1092_109239


namespace NUMINAMATH_CALUDE_prob_even_sum_first_15_primes_l1092_109292

/-- The number of prime numbers we're considering -/
def n : ℕ := 15

/-- The number of prime numbers we're selecting -/
def k : ℕ := 5

/-- The number of odd primes among the first n primes -/
def odd_primes : ℕ := n - 1

/-- The probability of selecting k primes from n primes such that their sum is even -/
def prob_even_sum (n k odd_primes : ℕ) : ℚ :=
  (Nat.choose odd_primes k + Nat.choose odd_primes (k - 3)) / Nat.choose n k

theorem prob_even_sum_first_15_primes : 
  prob_even_sum n k odd_primes = 2093 / 3003 :=
sorry

end NUMINAMATH_CALUDE_prob_even_sum_first_15_primes_l1092_109292


namespace NUMINAMATH_CALUDE_power_of_two_plus_one_l1092_109212

theorem power_of_two_plus_one (b m n : ℕ) 
  (hb : b > 1) 
  (hm : m > 0) 
  (hn : n > 0) 
  (hmn : m ≠ n) 
  (h_prime_divisors : ∀ p : ℕ, Prime p → (p ∣ (b^m - 1) ↔ p ∣ (b^n - 1))) :
  ∃ k : ℕ, b + 1 = 2^k :=
by sorry

end NUMINAMATH_CALUDE_power_of_two_plus_one_l1092_109212


namespace NUMINAMATH_CALUDE_equation_solution_l1092_109283

theorem equation_solution (y : ℝ) : 
  Real.sqrt (4 + Real.sqrt (3 * y - 2)) = Real.sqrt 12 → y = 22 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l1092_109283


namespace NUMINAMATH_CALUDE_impossible_sequence_is_invalid_l1092_109200

/-- Represents a sequence of letters --/
def Sequence := List Nat

/-- Checks if a sequence is valid according to the letter printing process --/
def is_valid_sequence (s : Sequence) : Prop :=
  ∀ i j, i < j → (s.indexOf i < s.indexOf j → ∀ k, i < k ∧ k < j → s.indexOf k < s.indexOf j)

/-- The impossible sequence --/
def impossible_sequence : Sequence := [4, 5, 2, 3, 1]

/-- Theorem stating that the impossible sequence is indeed impossible --/
theorem impossible_sequence_is_invalid : 
  ¬ is_valid_sequence impossible_sequence := by sorry

end NUMINAMATH_CALUDE_impossible_sequence_is_invalid_l1092_109200


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l1092_109265

/-- The line equation parameterized by k -/
def line_equation (x y k : ℝ) : Prop :=
  (1 + 4*k)*x - (2 - 3*k)*y + (2 - 3*k) = 0

/-- The fixed point that the line passes through -/
def fixed_point : ℝ × ℝ := (0, 1)

/-- Theorem stating that the fixed point is the unique point that satisfies the line equation for all k -/
theorem fixed_point_on_line :
  ∀ (k : ℝ), line_equation (fixed_point.1) (fixed_point.2) k ∧
  ∀ (x y : ℝ), (∀ (k : ℝ), line_equation x y k) → (x, y) = fixed_point :=
by sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l1092_109265


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l1092_109291

-- Define the inverse proportionality relationship
def inverse_proportional (α β : ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ α * β = k

-- State the theorem
theorem inverse_proportion_problem (α₁ α₂ β₁ β₂ : ℝ) :
  inverse_proportional α₁ β₁ →
  α₁ = 2 →
  β₁ = 5 →
  β₂ = -10 →
  inverse_proportional α₂ β₂ →
  α₂ = -1 :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l1092_109291


namespace NUMINAMATH_CALUDE_characterization_of_special_numbers_l1092_109278

-- Define a structure for numbers of the form p^n - 1
structure PrimeExponentMinusOne where
  p : Nat
  n : Nat
  isPrime : Nat.Prime p

-- Define a predicate for numbers whose all divisors are of the form p^n - 1
def allDivisorsArePrimeExponentMinusOne (m : Nat) : Prop :=
  ∀ d : Nat, d ∣ m → ∃ (p n : Nat), Nat.Prime p ∧ d = p^n - 1

-- Main theorem
theorem characterization_of_special_numbers (m : Nat) 
  (h1 : ∃ (p n : Nat), Nat.Prime p ∧ m = p^n - 1)
  (h2 : allDivisorsArePrimeExponentMinusOne m) :
  (∃ k : Nat, m = 2^k - 1 ∧ Nat.Prime m) ∨ m ∣ 48 :=
sorry

end NUMINAMATH_CALUDE_characterization_of_special_numbers_l1092_109278


namespace NUMINAMATH_CALUDE_inequality_theorem_l1092_109253

theorem inequality_theorem (a b c d : ℝ) 
  (h1 : a < b) (h2 : b < 0) (h3 : c < d) (h4 : d < 0) : 
  d / a < c / a := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l1092_109253


namespace NUMINAMATH_CALUDE_divisor_problem_l1092_109267

theorem divisor_problem (x : ℕ) : x > 0 ∧ x ∣ 1058 ∧ ∀ y, 0 < y ∧ y < x → ¬(y ∣ 1058) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l1092_109267


namespace NUMINAMATH_CALUDE_base7_321_equals_base10_162_l1092_109213

def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

theorem base7_321_equals_base10_162 :
  base7ToBase10 [1, 2, 3] = 162 := by
  sorry

end NUMINAMATH_CALUDE_base7_321_equals_base10_162_l1092_109213


namespace NUMINAMATH_CALUDE_salt_production_increase_l1092_109285

/-- Proves that given an initial production of 1000 tonnes in January and an average
    monthly production of 1550 tonnes for the year, the constant monthly increase
    in production from February to December is 100 tonnes. -/
theorem salt_production_increase (initial_production : ℕ) (average_production : ℕ) 
  (monthly_increase : ℕ) (h1 : initial_production = 1000) 
  (h2 : average_production = 1550) :
  (monthly_increase = 100 ∧ 
   (12 * initial_production + (monthly_increase * 11 * 12 / 2) = 12 * average_production)) := by
  sorry

end NUMINAMATH_CALUDE_salt_production_increase_l1092_109285


namespace NUMINAMATH_CALUDE_zoe_songs_total_l1092_109214

theorem zoe_songs_total (country_albums : ℕ) (pop_albums : ℕ) (songs_per_album : ℕ) : 
  country_albums = 3 → pop_albums = 5 → songs_per_album = 3 →
  (country_albums + pop_albums) * songs_per_album = 24 := by
sorry

end NUMINAMATH_CALUDE_zoe_songs_total_l1092_109214


namespace NUMINAMATH_CALUDE_number_five_less_than_one_l1092_109211

theorem number_five_less_than_one : (1 : ℤ) - 5 = -4 := by sorry

end NUMINAMATH_CALUDE_number_five_less_than_one_l1092_109211


namespace NUMINAMATH_CALUDE_tire_repair_tax_l1092_109247

theorem tire_repair_tax (repair_cost : ℚ) (num_tires : ℕ) (final_cost : ℚ) :
  repair_cost = 7 →
  num_tires = 4 →
  final_cost = 30 →
  (final_cost - (repair_cost * num_tires)) / num_tires = (1/2 : ℚ) := by
sorry

end NUMINAMATH_CALUDE_tire_repair_tax_l1092_109247


namespace NUMINAMATH_CALUDE_expression_equals_zero_l1092_109270

theorem expression_equals_zero (x : ℝ) : 
  (((x^2 - 2*x + 1) / (x^2 - 1)) / ((x - 1) / (x^2 + x))) - x = 0 :=
by sorry

end NUMINAMATH_CALUDE_expression_equals_zero_l1092_109270


namespace NUMINAMATH_CALUDE_tim_prank_combinations_l1092_109206

theorem tim_prank_combinations :
  let day1_choices : ℕ := 1
  let day2_choices : ℕ := 2
  let day3_choices : ℕ := 6
  let day4_choices : ℕ := 5
  let day5_choices : ℕ := 1
  day1_choices * day2_choices * day3_choices * day4_choices * day5_choices = 60 :=
by sorry

end NUMINAMATH_CALUDE_tim_prank_combinations_l1092_109206


namespace NUMINAMATH_CALUDE_sequence_general_term_l1092_109250

theorem sequence_general_term (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, n ≥ 1 → S n = 4 * a n - 3) →
  (∀ n : ℕ, n ≥ 1 → a n = (4/3)^(n-1)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l1092_109250


namespace NUMINAMATH_CALUDE_tangent_product_inequality_l1092_109261

theorem tangent_product_inequality (a b c : ℝ) (α β : ℝ) :
  a + b < 3 * c →
  Real.tan (α / 2) * Real.tan (β / 2) = (a + b - c) / (a + b + c) →
  Real.tan (α / 2) * Real.tan (β / 2) < 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_product_inequality_l1092_109261


namespace NUMINAMATH_CALUDE_recurring_decimal_sum_l1092_109245

theorem recurring_decimal_sum : 
  (1 / 3 : ℚ) + (4 / 99 : ℚ) + (5 / 999 : ℚ) = 42 / 111 := by sorry

end NUMINAMATH_CALUDE_recurring_decimal_sum_l1092_109245
