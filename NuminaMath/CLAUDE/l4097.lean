import Mathlib

namespace NUMINAMATH_CALUDE_cos_alpha_on_unit_circle_l4097_409795

theorem cos_alpha_on_unit_circle (α : Real) :
  let P : ℝ × ℝ := (-Real.sqrt 3 / 2, -1 / 2)
  (P.1^2 + P.2^2 = 1) →  -- Point P is on the unit circle
  (∃ t : ℝ, t > 0 ∧ t * (Real.cos α) = P.1 ∧ t * (Real.sin α) = P.2) →  -- P is on the terminal side of α
  Real.cos α = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_on_unit_circle_l4097_409795


namespace NUMINAMATH_CALUDE_mandy_toys_count_l4097_409789

theorem mandy_toys_count (mandy anna amanda : ℕ) 
  (h1 : anna = 3 * mandy)
  (h2 : amanda = anna + 2)
  (h3 : mandy + anna + amanda = 142) :
  mandy = 20 := by
sorry

end NUMINAMATH_CALUDE_mandy_toys_count_l4097_409789


namespace NUMINAMATH_CALUDE_flour_qualification_l4097_409767

def is_qualified (weight : ℝ) : Prop :=
  24.75 ≤ weight ∧ weight ≤ 25.25

theorem flour_qualification (weight : ℝ) :
  weight = 24.80 → is_qualified weight :=
by
  sorry

#check flour_qualification

end NUMINAMATH_CALUDE_flour_qualification_l4097_409767


namespace NUMINAMATH_CALUDE_negation_equivalence_l4097_409768

-- Define the original proposition
def original_prop (a b : ℝ) : Prop :=
  a^2 + b^2 = 0 → a = 0 ∧ b = 0

-- Define the negation we want to prove
def negation (a b : ℝ) : Prop :=
  a^2 + b^2 = 0 → ¬(a = 0 ∧ b = 0)

-- Theorem stating that the negation of the original proposition
-- is equivalent to our defined negation
theorem negation_equivalence :
  (¬ ∀ a b : ℝ, original_prop a b) ↔ (∀ a b : ℝ, negation a b) :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l4097_409768


namespace NUMINAMATH_CALUDE_function_inequality_condition_l4097_409718

theorem function_inequality_condition (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x, f x = x^2 - 4*x + 3) →
  a > 0 →
  b > 0 →
  (∀ x, |x + 2| < b → |f x + 1| < a) ↔
  b ≤ Real.sqrt a := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_condition_l4097_409718


namespace NUMINAMATH_CALUDE_choir_members_count_l4097_409754

theorem choir_members_count : ∃! n : ℕ, 200 ≤ n ∧ n ≤ 300 ∧ n % 10 = 6 ∧ n % 11 = 6 := by
  sorry

end NUMINAMATH_CALUDE_choir_members_count_l4097_409754


namespace NUMINAMATH_CALUDE_total_profit_is_840_l4097_409798

/-- Represents the investment and profit details of a business partnership --/
structure BusinessPartnership where
  initial_investment_A : ℕ
  initial_investment_B : ℕ
  withdrawal_A : ℕ
  addition_B : ℕ
  months_before_change : ℕ
  total_months : ℕ
  profit_share_A : ℕ

/-- Calculates the total profit given the business partnership details --/
def calculate_total_profit (bp : BusinessPartnership) : ℕ :=
  sorry

/-- Theorem stating that given the specific investment pattern, if A's profit share is 320,
    then the total profit is 840 --/
theorem total_profit_is_840 (bp : BusinessPartnership)
  (h1 : bp.initial_investment_A = 3000)
  (h2 : bp.initial_investment_B = 4000)
  (h3 : bp.withdrawal_A = 1000)
  (h4 : bp.addition_B = 1000)
  (h5 : bp.months_before_change = 8)
  (h6 : bp.total_months = 12)
  (h7 : bp.profit_share_A = 320) :
  calculate_total_profit bp = 840 :=
  sorry

end NUMINAMATH_CALUDE_total_profit_is_840_l4097_409798


namespace NUMINAMATH_CALUDE_optimal_purchasing_plan_l4097_409723

theorem optimal_purchasing_plan :
  let total_price : ℝ := 12
  let bulb_cost : ℝ := 30
  let motor_cost : ℝ := 45
  let total_items : ℕ := 90
  let bulb_price : ℝ := 3
  let motor_price : ℝ := 9
  let optimal_bulbs : ℕ := 30
  let optimal_motors : ℕ := 60
  let optimal_cost : ℝ := 630

  (∀ x y : ℕ, 
    x = 2 * y → 
    x * bulb_price = bulb_cost ∧ 
    y * motor_price = motor_cost) ∧
  
  (∀ m : ℕ,
    m ≤ total_items ∧
    m ≤ (total_items - m) / 2 →
    3 * m + 9 * (total_items - m) ≥ optimal_cost) ∧
  
  optimal_bulbs * bulb_price + optimal_motors * motor_price = optimal_cost :=
by sorry

end NUMINAMATH_CALUDE_optimal_purchasing_plan_l4097_409723


namespace NUMINAMATH_CALUDE_f_negative_pi_third_l4097_409780

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^x / (2^x + 1) + a * x + Real.cos (2 * x)

theorem f_negative_pi_third (a : ℝ) : 
  f a (π / 3) = 2 → f a (-π / 3) = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_pi_third_l4097_409780


namespace NUMINAMATH_CALUDE_exists_set_with_square_diff_divides_product_l4097_409792

theorem exists_set_with_square_diff_divides_product (n : ℕ+) :
  ∃ (S : Finset ℕ+), 
    S.card = n ∧ 
    ∀ (a b : ℕ+), a ∈ S → b ∈ S → (a - b)^2 ∣ (a * b) :=
sorry

end NUMINAMATH_CALUDE_exists_set_with_square_diff_divides_product_l4097_409792


namespace NUMINAMATH_CALUDE_minimum_packages_to_breakeven_l4097_409745

def bike_cost : ℕ := 1200
def earning_per_package : ℕ := 15
def maintenance_cost : ℕ := 5

theorem minimum_packages_to_breakeven :
  ∃ n : ℕ, n * (earning_per_package - maintenance_cost) ≥ bike_cost ∧
  ∀ m : ℕ, m * (earning_per_package - maintenance_cost) ≥ bike_cost → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_minimum_packages_to_breakeven_l4097_409745


namespace NUMINAMATH_CALUDE_marriage_age_proof_l4097_409788

/-- The average age of a husband and wife at the time of their marriage -/
def average_age_at_marriage : ℝ := 23

/-- The number of years passed since the marriage -/
def years_passed : ℕ := 5

/-- The age of the child -/
def child_age : ℕ := 1

/-- The current average age of the family -/
def current_family_average_age : ℝ := 19

/-- The number of people in the family -/
def family_size : ℕ := 3

theorem marriage_age_proof :
  average_age_at_marriage = 23 :=
by
  sorry

#check marriage_age_proof

end NUMINAMATH_CALUDE_marriage_age_proof_l4097_409788


namespace NUMINAMATH_CALUDE_gauss_candy_remaining_l4097_409790

/-- The number of lollipops that remain after packaging -/
def remaining_lollipops (total : ℕ) (per_package : ℕ) : ℕ :=
  total % per_package

/-- Theorem stating the number of remaining lollipops for the Gauss Candy Company problem -/
theorem gauss_candy_remaining : remaining_lollipops 8362 12 = 10 := by
  sorry

end NUMINAMATH_CALUDE_gauss_candy_remaining_l4097_409790


namespace NUMINAMATH_CALUDE_trail_length_proof_l4097_409747

/-- The total length of a trail where two friends walk from opposite ends, with one friend 20% faster than the other, and the faster friend walks 12 km when they meet. -/
def trail_length : ℝ := 22

/-- The distance walked by the faster friend when they meet. -/
def faster_friend_distance : ℝ := 12

/-- The ratio of the faster friend's speed to the slower friend's speed. -/
def speed_ratio : ℝ := 1.2

theorem trail_length_proof :
  ∃ (v : ℝ), v > 0 ∧
    trail_length = faster_friend_distance + v * (faster_friend_distance / (speed_ratio * v)) :=
by sorry

end NUMINAMATH_CALUDE_trail_length_proof_l4097_409747


namespace NUMINAMATH_CALUDE_final_red_probability_l4097_409782

-- Define the contents of each bag
def bagA : ℕ × ℕ := (5, 3)  -- (white, black)
def bagB : ℕ × ℕ := (4, 6)  -- (red, green)
def bagC : ℕ × ℕ := (3, 4)  -- (red, green)

-- Define the probability of drawing a specific marble from a bag
def probDraw (color : ℕ) (bag : ℕ × ℕ) : ℚ :=
  color / (bag.1 + bag.2)

-- Define the probability of the final marble being red
def probFinalRed : ℚ :=
  let probWhiteA := probDraw bagA.1 bagA
  let probBlackA := probDraw bagA.2 bagA
  let probGreenB := probDraw bagB.2 bagB
  let probRedB := probDraw bagB.1 bagB
  let probGreenC := probDraw bagC.2 bagC
  let probRedC := probDraw bagC.1 bagC
  probWhiteA * probGreenB * probRedB + probBlackA * probGreenC * probRedC

-- Theorem statement
theorem final_red_probability : probFinalRed = 79 / 980 := by
  sorry

end NUMINAMATH_CALUDE_final_red_probability_l4097_409782


namespace NUMINAMATH_CALUDE_sandys_sum_attempt_l4097_409704

/-- Sandy's sum attempt problem -/
theorem sandys_sum_attempt :
  ∀ (correct_marks incorrect_marks total_marks correct_sums : ℕ),
    correct_marks = 3 →
    incorrect_marks = 2 →
    total_marks = 45 →
    correct_sums = 21 →
    ∃ (total_sums : ℕ),
      total_sums = correct_sums + (total_marks - correct_marks * correct_sums) / incorrect_marks ∧
      total_sums = 30 :=
by sorry

end NUMINAMATH_CALUDE_sandys_sum_attempt_l4097_409704


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l4097_409793

theorem inverse_variation_problem (a b : ℝ) (k : ℝ) :
  (∀ a b, a^3 * b^2 = k) →  -- a^3 varies inversely with b^2
  (4^3 * 2^2 = k) →         -- a = 4 when b = 2
  (a^3 * 8^2 = k) →         -- condition for b = 8
  a = 4^(1/3) :=            -- prove that a = 4^(1/3) when b = 8
by sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l4097_409793


namespace NUMINAMATH_CALUDE_cars_meeting_time_l4097_409761

/-- Given a scenario with two cars and a train, calculate the time for the cars to meet -/
theorem cars_meeting_time (train_length : ℝ) (train_speed_kmh : ℝ) (time_between_encounters : ℝ)
  (time_pass_A : ℝ) (time_pass_B : ℝ)
  (h1 : train_length = 180)
  (h2 : train_speed_kmh = 60)
  (h3 : time_between_encounters = 5)
  (h4 : time_pass_A = 30 / 60)
  (h5 : time_pass_B = 6 / 60) :
  let train_speed := train_speed_kmh * 1000 / 3600
  let car_A_speed := train_speed - train_length / time_pass_A
  let car_B_speed := train_length / time_pass_B - train_speed
  let distance := time_between_encounters * (train_speed - car_A_speed)
  distance / (car_A_speed + car_B_speed) = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_cars_meeting_time_l4097_409761


namespace NUMINAMATH_CALUDE_oneBlack_twoWhite_mutually_exclusive_not_contradictory_l4097_409759

/-- Represents the color of a ball -/
inductive BallColor
| Black
| White

/-- Represents the outcome of drawing two balls -/
structure DrawOutcome :=
  (first second : BallColor)

/-- The bag containing 2 black balls and 2 white balls -/
def bag : Multiset BallColor :=
  2 • {BallColor.Black} + 2 • {BallColor.White}

/-- The event of drawing exactly one black ball -/
def oneBlack (outcome : DrawOutcome) : Prop :=
  (outcome.first = BallColor.Black ∧ outcome.second = BallColor.White) ∨
  (outcome.first = BallColor.White ∧ outcome.second = BallColor.Black)

/-- The event of drawing exactly two white balls -/
def twoWhite (outcome : DrawOutcome) : Prop :=
  outcome.first = BallColor.White ∧ outcome.second = BallColor.White

/-- The theorem stating that oneBlack and twoWhite are mutually exclusive but not contradictory -/
theorem oneBlack_twoWhite_mutually_exclusive_not_contradictory :
  (∀ outcome : DrawOutcome, ¬(oneBlack outcome ∧ twoWhite outcome)) ∧
  (∃ outcome : DrawOutcome, ¬oneBlack outcome ∧ ¬twoWhite outcome) :=
sorry

end NUMINAMATH_CALUDE_oneBlack_twoWhite_mutually_exclusive_not_contradictory_l4097_409759


namespace NUMINAMATH_CALUDE_cloth_sale_commission_calculation_l4097_409779

/-- Calculates the worth of cloth sold given the commission rate and commission amount. -/
def worth_of_cloth_sold (commission_rate : ℚ) (commission : ℚ) : ℚ :=
  commission * (100 / commission_rate)

/-- Theorem stating that for a 4% commission rate and Rs. 12.50 commission, 
    the worth of cloth sold is Rs. 312.50 -/
theorem cloth_sale_commission_calculation :
  worth_of_cloth_sold (4 : ℚ) (25/2 : ℚ) = (625/2 : ℚ) := by
  sorry

#eval worth_of_cloth_sold (4 : ℚ) (25/2 : ℚ)

end NUMINAMATH_CALUDE_cloth_sale_commission_calculation_l4097_409779


namespace NUMINAMATH_CALUDE_parallel_vectors_condition_l4097_409753

/-- Given vectors a and b, where a is parallel to their sum, prove that the y-component of b is -3. -/
theorem parallel_vectors_condition (a b : ℝ × ℝ) : 
  a = (-1, 1) → 
  b.1 = 3 → 
  ∃ (k : ℝ), k • a = (a + b) → 
  b.2 = -3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_condition_l4097_409753


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l4097_409714

theorem solve_exponential_equation :
  ∃ x : ℝ, (8 : ℝ) ^ (4 * x - 6) = (1 / 2 : ℝ) ^ (x + 5) ∧ x = 1 :=
by sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l4097_409714


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l4097_409749

theorem partial_fraction_decomposition (x : ℝ) (h1 : x ≠ 4) (h2 : x ≠ 7) :
  6 * x / ((x - 7) * (x - 4)^2) = 
    14/3 / (x - 7) + 26/33 / (x - 4) + (-8) / (x - 4)^2 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l4097_409749


namespace NUMINAMATH_CALUDE_prob_different_topics_is_five_sixths_l4097_409752

/-- The number of essay topics -/
def num_topics : ℕ := 6

/-- The probability that two students select different topics -/
def prob_different_topics : ℚ := 5/6

/-- Theorem stating that the probability of two students selecting different topics
    from 6 available topics is 5/6 -/
theorem prob_different_topics_is_five_sixths :
  prob_different_topics = 5/6 := by sorry

end NUMINAMATH_CALUDE_prob_different_topics_is_five_sixths_l4097_409752


namespace NUMINAMATH_CALUDE_B_elements_l4097_409741

def B : Set ℤ := {x : ℤ | -3 < 2 * x - 1 ∧ 2 * x - 1 < 3}

theorem B_elements : B = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_B_elements_l4097_409741


namespace NUMINAMATH_CALUDE_hat_and_glasses_probability_l4097_409786

theorem hat_and_glasses_probability
  (total_hats : ℕ)
  (total_glasses : ℕ)
  (prob_hat_given_glasses : ℚ)
  (h1 : total_hats = 60)
  (h2 : total_glasses = 40)
  (h3 : prob_hat_given_glasses = 1 / 4) :
  (total_glasses : ℚ) * prob_hat_given_glasses / total_hats = 1 / 6 :=
by sorry

end NUMINAMATH_CALUDE_hat_and_glasses_probability_l4097_409786


namespace NUMINAMATH_CALUDE_f_max_value_l4097_409765

/-- The quadratic function f(x) = -3x^2 + 18x - 5 -/
def f (x : ℝ) : ℝ := -3 * x^2 + 18 * x - 5

/-- The maximum value of f(x) is 22 -/
theorem f_max_value : ∃ (M : ℝ), M = 22 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_f_max_value_l4097_409765


namespace NUMINAMATH_CALUDE_percentage_of_male_employees_l4097_409738

theorem percentage_of_male_employees (total_employees : ℕ) 
  (males_below_50 : ℕ) (h1 : total_employees = 6400) 
  (h2 : males_below_50 = 3120) : 
  (males_below_50 : ℚ) / (0.75 * total_employees) = 65 / 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_male_employees_l4097_409738


namespace NUMINAMATH_CALUDE_fog_sum_l4097_409727

theorem fog_sum (f o g : Nat) : 
  f < 10 → o < 10 → g < 10 →
  (100 * f + 10 * o + g) * 4 = 1464 →
  f + o + g = 15 := by
sorry

end NUMINAMATH_CALUDE_fog_sum_l4097_409727


namespace NUMINAMATH_CALUDE_abs_five_implies_plus_minus_five_l4097_409763

theorem abs_five_implies_plus_minus_five (a : ℝ) : |a| = 5 → a = 5 ∨ a = -5 := by
  sorry

end NUMINAMATH_CALUDE_abs_five_implies_plus_minus_five_l4097_409763


namespace NUMINAMATH_CALUDE_thursday_tea_consumption_l4097_409784

/-- Represents the relationship between hours grading and liters of tea consumed -/
structure TeaGrading where
  hours : ℝ
  liters : ℝ
  inv_prop : hours * liters = hours * liters

/-- The constant of proportionality derived from Wednesday's data -/
def wednesday_constant : ℝ := 5 * 4

/-- Theorem stating that given Wednesday's data and Thursday's hours, the teacher drinks 2.5 liters of tea on Thursday -/
theorem thursday_tea_consumption (wednesday : TeaGrading) (thursday : TeaGrading) 
    (h_wednesday : wednesday.hours = 5 ∧ wednesday.liters = 4)
    (h_thursday : thursday.hours = 8)
    (h_constant : wednesday.hours * wednesday.liters = thursday.hours * thursday.liters) :
    thursday.liters = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_thursday_tea_consumption_l4097_409784


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l4097_409730

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0) → 0 ≤ a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l4097_409730


namespace NUMINAMATH_CALUDE_percentage_relation_l4097_409740

theorem percentage_relation (x a b : ℝ) (h1 : a = 0.06 * x) (h2 : b = 0.3 * x) :
  a = 0.2 * b := by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l4097_409740


namespace NUMINAMATH_CALUDE_euler_family_mean_age_l4097_409750

def euler_family_ages : List ℝ := [12, 12, 12, 12, 9, 9, 15, 17]

theorem euler_family_mean_age : 
  (euler_family_ages.sum / euler_family_ages.length : ℝ) = 12.25 := by
  sorry

end NUMINAMATH_CALUDE_euler_family_mean_age_l4097_409750


namespace NUMINAMATH_CALUDE_quadratic_transformation_l4097_409732

/-- Given a quadratic function ax² + bx + c that can be expressed as 5(x - 5)² - 3,
    prove that when 4ax² + 4bx + 4c is expressed as n(x - h)² + k, h = 5. -/
theorem quadratic_transformation (a b c : ℝ) : 
  (∃ x, ax^2 + b*x + c = 5*(x - 5)^2 - 3) → 
  (∃ n k, ∀ x, 4*a*x^2 + 4*b*x + 4*c = n*(x - 5)^2 + k) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l4097_409732


namespace NUMINAMATH_CALUDE_bea_highest_profit_l4097_409764

/-- Represents a lemonade seller with their sales information -/
structure LemonadeSeller where
  name : String
  price : ℕ
  soldGlasses : ℕ
  variableCost : ℕ

/-- Calculates the profit for a lemonade seller -/
def calculateProfit (seller : LemonadeSeller) : ℕ :=
  seller.price * seller.soldGlasses - seller.variableCost * seller.soldGlasses

/-- Theorem stating that Bea makes the most profit -/
theorem bea_highest_profit (bea dawn carla : LemonadeSeller)
  (h_bea : bea = { name := "Bea", price := 25, soldGlasses := 10, variableCost := 10 })
  (h_dawn : dawn = { name := "Dawn", price := 28, soldGlasses := 8, variableCost := 12 })
  (h_carla : carla = { name := "Carla", price := 35, soldGlasses := 6, variableCost := 15 }) :
  calculateProfit bea ≥ calculateProfit dawn ∧ calculateProfit bea ≥ calculateProfit carla :=
by sorry

end NUMINAMATH_CALUDE_bea_highest_profit_l4097_409764


namespace NUMINAMATH_CALUDE_num_rna_molecules_l4097_409796

/-- Represents the number of possible bases for each position in an RNA molecule -/
def num_bases : ℕ := 4

/-- Represents the length of the RNA molecule -/
def rna_length : ℕ := 100

/-- Theorem stating that the number of unique RNA molecules is 4^100 -/
theorem num_rna_molecules : (num_bases : ℕ) ^ rna_length = 4 ^ 100 := by
  sorry

end NUMINAMATH_CALUDE_num_rna_molecules_l4097_409796


namespace NUMINAMATH_CALUDE_min_type_A_costumes_l4097_409794

-- Define the cost of type B costumes
def cost_B : ℝ := 120

-- Define the cost of type A costumes
def cost_A : ℝ := cost_B + 30

-- Define the total number of costumes
def total_costumes : ℕ := 20

-- Define the minimum total cost
def min_total_cost : ℝ := 2800

-- Theorem statement
theorem min_type_A_costumes :
  ∀ m : ℕ,
  (m : ℝ) * cost_A + (total_costumes - m : ℝ) * cost_B ≥ min_total_cost →
  m ≥ 14 :=
by sorry

end NUMINAMATH_CALUDE_min_type_A_costumes_l4097_409794


namespace NUMINAMATH_CALUDE_jackson_earnings_l4097_409762

def usd_per_hour : ℝ := 5
def gbp_per_hour : ℝ := 3
def jpy_per_hour : ℝ := 400

def vacuum_hours : ℝ := 2 * 2
def dishes_hours : ℝ := 0.5
def bathroom_hours : ℝ := 0.5 * 3

def gbp_to_usd : ℝ := 1.35
def jpy_to_usd : ℝ := 0.009

theorem jackson_earnings : 
  (vacuum_hours * usd_per_hour) + 
  (dishes_hours * gbp_per_hour * gbp_to_usd) + 
  (bathroom_hours * jpy_per_hour * jpy_to_usd) = 27.425 := by
sorry

end NUMINAMATH_CALUDE_jackson_earnings_l4097_409762


namespace NUMINAMATH_CALUDE_fraction_simplification_l4097_409744

theorem fraction_simplification : 
  ((3^2010)^2 - (3^2008)^2) / ((3^2009)^2 - (3^2007)^2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l4097_409744


namespace NUMINAMATH_CALUDE_afternoon_snowfall_l4097_409769

theorem afternoon_snowfall (total : ℝ) (morning : ℝ) (afternoon : ℝ)
  (h1 : total = 0.625)
  (h2 : morning = 0.125)
  (h3 : total = morning + afternoon) :
  afternoon = 0.500 := by
sorry

end NUMINAMATH_CALUDE_afternoon_snowfall_l4097_409769


namespace NUMINAMATH_CALUDE_exists_500g_bag_of_salt_l4097_409743

/-- A bag of salt is represented as a positive real number indicating its weight in grams. -/
def BagOfSalt : Type := { w : ℝ // w > 0 }

/-- Theorem: There exists a bag of salt that weighs 500 grams. -/
theorem exists_500g_bag_of_salt : ∃ (bag : BagOfSalt), bag.val = 500 := by
  sorry

end NUMINAMATH_CALUDE_exists_500g_bag_of_salt_l4097_409743


namespace NUMINAMATH_CALUDE_rug_overlap_problem_l4097_409726

/-- Given three rugs with total area A, overlapped to cover floor area F,
    with S2 area covered by exactly two layers, prove the area S3
    covered by three layers. -/
theorem rug_overlap_problem (A F S2 : ℝ) (hA : A = 200) (hF : F = 138) (hS2 : S2 = 24) :
  ∃ S1 S3 : ℝ,
    S1 + S2 + S3 = F ∧
    S1 + 2 * S2 + 3 * S3 = A ∧
    S3 = 19 := by
  sorry

end NUMINAMATH_CALUDE_rug_overlap_problem_l4097_409726


namespace NUMINAMATH_CALUDE_prime_factor_sum_l4097_409725

theorem prime_factor_sum (w x y z : ℕ) : 
  2^w * 3^x * 5^y * 7^z = 2450 → 3*w + 2*x + 7*y + 5*z = 27 := by
  sorry

end NUMINAMATH_CALUDE_prime_factor_sum_l4097_409725


namespace NUMINAMATH_CALUDE_increasing_function_sum_implication_l4097_409734

theorem increasing_function_sum_implication (f : ℝ → ℝ) (a b : ℝ) 
  (h_increasing : ∀ x y : ℝ, x < y → f x < f y) :
  f a + f b > f (-a) + f (-b) → a + b > 0 := by
  sorry

end NUMINAMATH_CALUDE_increasing_function_sum_implication_l4097_409734


namespace NUMINAMATH_CALUDE_construct_3x3x3_cube_l4097_409781

/-- Represents a 3D piece with given dimensions -/
structure Piece where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents the collection of pieces available for construction -/
structure PieceSet where
  large_pieces : List Piece
  small_pieces : List Piece

/-- Represents a 3D cube -/
structure Cube where
  side_length : ℕ

/-- Checks if a set of pieces can construct the given cube -/
def can_construct_cube (pieces : PieceSet) (cube : Cube) : Prop :=
  -- The actual implementation would involve complex logic to check if the pieces can form the cube
  sorry

/-- The main theorem stating that the given set of pieces can construct a 3x3x3 cube -/
theorem construct_3x3x3_cube : 
  let pieces : PieceSet := {
    large_pieces := List.replicate 6 { length := 1, width := 2, height := 2 },
    small_pieces := List.replicate 3 { length := 1, width := 1, height := 1 }
  }
  let target_cube : Cube := { side_length := 3 }
  can_construct_cube pieces target_cube := by
  sorry


end NUMINAMATH_CALUDE_construct_3x3x3_cube_l4097_409781


namespace NUMINAMATH_CALUDE_mathathon_problem_count_l4097_409733

theorem mathathon_problem_count (rounds : Nat) (problems_per_round : Nat) : 
  rounds = 7 → problems_per_round = 3 → rounds * problems_per_round = 21 := by
  sorry

end NUMINAMATH_CALUDE_mathathon_problem_count_l4097_409733


namespace NUMINAMATH_CALUDE_noah_large_paintings_l4097_409706

/-- Represents the number of large paintings sold last month -/
def L : ℕ := sorry

/-- Price of a large painting -/
def large_price : ℕ := 60

/-- Price of a small painting -/
def small_price : ℕ := 30

/-- Number of small paintings sold last month -/
def small_paintings_last_month : ℕ := 4

/-- Total sales this month -/
def sales_this_month : ℕ := 1200

/-- Theorem stating that Noah sold 8 large paintings last month -/
theorem noah_large_paintings : L = 8 := by
  sorry

end NUMINAMATH_CALUDE_noah_large_paintings_l4097_409706


namespace NUMINAMATH_CALUDE_cos_negative_45_degrees_l4097_409776

theorem cos_negative_45_degrees : Real.cos (-(45 * π / 180)) = 1 / Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_cos_negative_45_degrees_l4097_409776


namespace NUMINAMATH_CALUDE_bananas_cantaloupe_cost_l4097_409724

-- Define variables for the prices of each item
variable (a : ℚ) -- Price of a sack of apples
variable (b : ℚ) -- Price of a bunch of bananas
variable (c : ℚ) -- Price of a cantaloupe
variable (d : ℚ) -- Price of a carton of dates
variable (h : ℚ) -- Price of a jar of honey

-- Define the conditions
axiom total_cost : a + b + c + d + h = 30
axiom dates_cost : d = 4 * a
axiom cantaloupe_cost : c = 2 * a - b

-- Theorem to prove
theorem bananas_cantaloupe_cost : b + c = 50 / 7 := by
  sorry

end NUMINAMATH_CALUDE_bananas_cantaloupe_cost_l4097_409724


namespace NUMINAMATH_CALUDE_chef_apples_l4097_409758

theorem chef_apples (apples_left apples_used : ℕ) 
  (h1 : apples_left = 2) 
  (h2 : apples_used = 41) : 
  apples_left + apples_used = 43 := by
  sorry

end NUMINAMATH_CALUDE_chef_apples_l4097_409758


namespace NUMINAMATH_CALUDE_factor_expression_l4097_409705

theorem factor_expression (x : ℝ) : 3 * x^2 - 75 = 3 * (x + 5) * (x - 5) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l4097_409705


namespace NUMINAMATH_CALUDE_max_value_of_expression_l4097_409720

/-- Given that a, b, and c are distinct elements from the set {1, 2, 4},
    the maximum value of (a / 2) / (b / c) is 8. -/
theorem max_value_of_expression (a b c : ℕ) : 
  a ∈ ({1, 2, 4} : Set ℕ) →
  b ∈ ({1, 2, 4} : Set ℕ) →
  c ∈ ({1, 2, 4} : Set ℕ) →
  a ≠ b → b ≠ c → a ≠ c →
  (a / 2 : ℚ) / (b / c : ℚ) ≤ 8 ∧ 
  ∃ (x y z : ℕ), x ∈ ({1, 2, 4} : Set ℕ) ∧ 
                 y ∈ ({1, 2, 4} : Set ℕ) ∧ 
                 z ∈ ({1, 2, 4} : Set ℕ) ∧ 
                 x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
                 (x / 2 : ℚ) / (y / z : ℚ) = 8 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l4097_409720


namespace NUMINAMATH_CALUDE_no_arithmetic_mean_l4097_409716

theorem no_arithmetic_mean (f1 f2 f3 : ℚ) : 
  f1 = 5/8 ∧ f2 = 9/12 ∧ f3 = 7/10 →
  (f1 ≠ (f2 + f3) / 2) ∧ (f2 ≠ (f1 + f3) / 2) ∧ (f3 ≠ (f1 + f2) / 2) := by
  sorry

#check no_arithmetic_mean

end NUMINAMATH_CALUDE_no_arithmetic_mean_l4097_409716


namespace NUMINAMATH_CALUDE_circle_with_diameter_OC_l4097_409778

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 6)^2 + (y - 8)^2 = 4

-- Define the origin O
def origin : ℝ × ℝ := (0, 0)

-- Define the center of circle C
def center_C : ℝ × ℝ := (6, 8)

-- Define the equation of the circle with diameter OC
def circle_OC (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 25

-- Theorem statement
theorem circle_with_diameter_OC :
  ∀ x y : ℝ, circle_C x y → circle_OC x y :=
sorry

end NUMINAMATH_CALUDE_circle_with_diameter_OC_l4097_409778


namespace NUMINAMATH_CALUDE_chocolate_candy_cost_difference_l4097_409755

/-- Calculates the cost difference between chocolates and candy bars --/
theorem chocolate_candy_cost_difference :
  let initial_money : ℚ := 50
  let candy_price : ℚ := 4
  let candy_discount_rate : ℚ := 0.2
  let candy_discount_threshold : ℕ := 3
  let candy_quantity : ℕ := 5
  let chocolate_price : ℚ := 6
  let chocolate_tax_rate : ℚ := 0.05
  let chocolate_quantity : ℕ := 4

  let candy_cost : ℚ := if candy_quantity ≥ candy_discount_threshold
    then candy_quantity * candy_price * (1 - candy_discount_rate)
    else candy_quantity * candy_price

  let chocolate_cost : ℚ := chocolate_quantity * chocolate_price * (1 + chocolate_tax_rate)

  chocolate_cost - candy_cost = 9.2 :=
by
  sorry


end NUMINAMATH_CALUDE_chocolate_candy_cost_difference_l4097_409755


namespace NUMINAMATH_CALUDE_no_solution_implies_b_bounded_l4097_409791

theorem no_solution_implies_b_bounded (a b : ℝ) :
  (∀ x : ℝ, ¬(a * Real.cos x + b * Real.cos (3 * x) > 1)) →
  |b| ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_b_bounded_l4097_409791


namespace NUMINAMATH_CALUDE_square_fold_visible_area_l4097_409739

theorem square_fold_visible_area (side_length : ℝ) (ao_length : ℝ) : 
  side_length = 1 → ao_length = 1/3 → 
  (visible_area : ℝ) = side_length * ao_length :=
by sorry

end NUMINAMATH_CALUDE_square_fold_visible_area_l4097_409739


namespace NUMINAMATH_CALUDE_quadratic_sum_l4097_409777

/-- A quadratic function with vertex at (2, 8) and passing through (0, 0) -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_sum (a b c : ℝ) :
  (∀ x, f a b c x = a * x^2 + b * x + c) →
  f a b c 2 = 8 →
  (∀ x, f a b c x ≤ f a b c 2) →
  f a b c 0 = 0 →
  a + b + 2*c = 6 := by sorry

end NUMINAMATH_CALUDE_quadratic_sum_l4097_409777


namespace NUMINAMATH_CALUDE_specific_participants_match_probability_l4097_409712

/-- The number of participants in the tournament -/
def n : ℕ := 26

/-- The probability that two specific participants will play against each other -/
def probability : ℚ := 1 / 13

/-- Theorem stating the probability of two specific participants playing against each other -/
theorem specific_participants_match_probability :
  (n - 1 : ℚ) / (n * (n - 1) / 2) = probability := by sorry

end NUMINAMATH_CALUDE_specific_participants_match_probability_l4097_409712


namespace NUMINAMATH_CALUDE_polynomial_identity_l4097_409719

theorem polynomial_identity (a b c : ℝ) :
  (a + b) * (b + c) * (c + a) = (a + b + c) * (a * b + b * c + c * a) - 2 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l4097_409719


namespace NUMINAMATH_CALUDE_extremum_of_f_under_constraint_l4097_409707

-- Define the function f
def f (x y : ℝ) : ℝ := x^2 + y^2 - 2*x - y

-- Define the constraint function φ
def φ (x y : ℝ) : ℝ := x + y - 1

-- State the theorem
theorem extremum_of_f_under_constraint :
  ∃ (x y : ℝ),
    φ x y = 0 ∧
    (∀ (x' y' : ℝ), φ x' y' = 0 → f x' y' ≥ f x y) ∧
    x = 3/4 ∧ y = 1/4 ∧ f x y = -9/8 :=
sorry

end NUMINAMATH_CALUDE_extremum_of_f_under_constraint_l4097_409707


namespace NUMINAMATH_CALUDE_mans_current_age_l4097_409735

/-- Given a man and his son, where the man is thrice as old as his son now,
    and after 12 years he will be twice as old as his son,
    prove that the man's current age is 36 years. -/
theorem mans_current_age (man_age son_age : ℕ) : 
  man_age = 3 * son_age →
  man_age + 12 = 2 * (son_age + 12) →
  man_age = 36 := by
  sorry

end NUMINAMATH_CALUDE_mans_current_age_l4097_409735


namespace NUMINAMATH_CALUDE_right_triangle_leg_sum_bound_l4097_409766

theorem right_triangle_leg_sum_bound (a b c : ℝ) (h : a^2 + b^2 = c^2) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a + b ≤ Real.sqrt 2 * c := by
sorry

end NUMINAMATH_CALUDE_right_triangle_leg_sum_bound_l4097_409766


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2019th_term_l4097_409774

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_2019th_term
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum_5 : (a 1 + a 2 + a 3 + a 4 + a 5) = 15)
  (h_6th_term : a 6 = 6) :
  a 2019 = 2019 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2019th_term_l4097_409774


namespace NUMINAMATH_CALUDE_problem_solution_l4097_409785

noncomputable section

def m (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin (x / 3), Real.cos (x / 3))
def n (x : ℝ) : ℝ × ℝ := (Real.cos (x / 3), Real.cos (x / 3))
def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

def a : ℝ := 2

variable (A B C : ℝ)
variable (b c : ℝ)

axiom triangle_condition : (2 * a - b) * Real.cos C = c * Real.cos B
axiom f_of_A : f A = 3 / 2

theorem problem_solution :
  (∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x ∧ ∀ S : ℝ, S > 0 ∧ (∀ x : ℝ, f (x + S) = f x) → T ≤ S) ∧
  (∃ k : ℤ, ∀ x : ℝ, f ((-π/4 + 3*π/2*↑k) + x) = f ((-π/4 + 3*π/2*↑k) - x)) ∧
  c = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4097_409785


namespace NUMINAMATH_CALUDE_percentage_saved_approx_11_percent_l4097_409708

def original_price : ℝ := 30
def amount_saved : ℝ := 3
def amount_spent : ℝ := 24

theorem percentage_saved_approx_11_percent :
  let actual_price := amount_spent + amount_saved
  let percentage_saved := (amount_saved / actual_price) * 100
  ∃ ε > 0, abs (percentage_saved - 11) < ε :=
by sorry

end NUMINAMATH_CALUDE_percentage_saved_approx_11_percent_l4097_409708


namespace NUMINAMATH_CALUDE_sunshine_orchard_pumpkins_l4097_409701

/-- The number of pumpkins at Moonglow Orchard -/
def moonglow_pumpkins : ℕ := 14

/-- The number of pumpkins at Sunshine Orchard -/
def sunshine_pumpkins : ℕ := 3 * moonglow_pumpkins + 12

theorem sunshine_orchard_pumpkins : sunshine_pumpkins = 54 := by
  sorry

end NUMINAMATH_CALUDE_sunshine_orchard_pumpkins_l4097_409701


namespace NUMINAMATH_CALUDE_unique_prime_pair_sum_59_l4097_409721

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

theorem unique_prime_pair_sum_59 :
  ∃! (p q : ℕ), is_prime p ∧ is_prime q ∧ p ≠ q ∧ p + q = 59 :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_pair_sum_59_l4097_409721


namespace NUMINAMATH_CALUDE_number_of_subsets_l4097_409746

/-- For a finite set with n elements, the number of subsets is 2^n -/
theorem number_of_subsets (S : Type*) [Fintype S] : 
  Finset.card (Finset.powerset (Finset.univ : Finset S)) = 2 ^ Fintype.card S := by
  sorry

end NUMINAMATH_CALUDE_number_of_subsets_l4097_409746


namespace NUMINAMATH_CALUDE_music_class_size_l4097_409728

theorem music_class_size :
  ∃! n : ℕ, 0 < n ∧ n < 50 ∧ n % 8 = 5 ∧ n % 6 = 2 ∧ n = 29 := by
  sorry

end NUMINAMATH_CALUDE_music_class_size_l4097_409728


namespace NUMINAMATH_CALUDE_equal_length_different_turns_l4097_409770

/-- Represents a point in the triangular grid -/
structure Point where
  x : ℤ
  y : ℤ

/-- Represents a direction in the triangular grid -/
inductive Direction
  | Up
  | UpRight
  | DownRight
  | Down
  | DownLeft
  | UpLeft

/-- Represents a route in the triangular grid -/
structure Route where
  start : Point
  steps : List Direction
  leftTurns : ℕ

/-- Calculates the length of a route -/
def routeLength (r : Route) : ℕ := r.steps.length

/-- Theorem: There exist two routes in a triangular grid with different numbers of left turns but equal length -/
theorem equal_length_different_turns :
  ∃ (start finish : Point) (route1 route2 : Route),
    route1.start = start ∧
    route2.start = start ∧
    (routeLength route1 = routeLength route2) ∧
    route1.leftTurns = 4 ∧
    route2.leftTurns = 1 :=
  sorry

end NUMINAMATH_CALUDE_equal_length_different_turns_l4097_409770


namespace NUMINAMATH_CALUDE_range_of_m_l4097_409731

def p (m : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 + m*x₀ + 2*m - 3 < 0

def q (m : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - m ≤ 0

theorem range_of_m (m : ℝ) :
  (p m ∨ q m) ∧ ¬(p m ∧ q m) → m < 2 ∨ (4 ≤ m ∧ m ≤ 6) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l4097_409731


namespace NUMINAMATH_CALUDE_specific_pyramid_height_l4097_409713

/-- Represents a right pyramid with a rectangular base -/
structure RightPyramid where
  basePerimeter : ℝ
  baseLength : ℝ
  baseBreadth : ℝ
  apexToVertexDistance : ℝ

/-- Calculates the height of a right pyramid -/
def pyramidHeight (p : RightPyramid) : ℝ :=
  sorry

/-- Theorem stating the height of the specific pyramid -/
theorem specific_pyramid_height :
  let p : RightPyramid := {
    basePerimeter := 40,
    baseLength := 40 / 3,
    baseBreadth := 20 / 3,
    apexToVertexDistance := 15
  }
  pyramidHeight p = 10 * Real.sqrt 19 / 3 := by
  sorry

end NUMINAMATH_CALUDE_specific_pyramid_height_l4097_409713


namespace NUMINAMATH_CALUDE_quadratic_roots_expression_l4097_409722

theorem quadratic_roots_expression (x₁ x₂ : ℝ) : 
  x₁^2 - x₁ - 2 = 0 → x₂^2 - x₂ - 2 = 0 → (1 + x₁) + x₂ * (1 - x₁) = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_expression_l4097_409722


namespace NUMINAMATH_CALUDE_angle_range_l4097_409717

theorem angle_range (α : Real) 
  (h1 : α > 0 ∧ α < 2 * Real.pi) 
  (h2 : Real.sin α > 0) 
  (h3 : Real.cos α < 0) : 
  α > Real.pi / 2 ∧ α < Real.pi :=
by sorry

end NUMINAMATH_CALUDE_angle_range_l4097_409717


namespace NUMINAMATH_CALUDE_product_sequence_equals_32_l4097_409711

theorem product_sequence_equals_32 : 
  (1/4 : ℚ) * 8 * (1/16 : ℚ) * 32 * (1/64 : ℚ) * 128 * (1/256 : ℚ) * 512 * (1/1024 : ℚ) * 2048 = 32 := by
  sorry

end NUMINAMATH_CALUDE_product_sequence_equals_32_l4097_409711


namespace NUMINAMATH_CALUDE_camel_count_l4097_409736

/-- The cost of an elephant in rupees -/
def elephant_cost : ℚ := 12000

/-- The cost of an ox in rupees -/
def ox_cost : ℚ := 8000

/-- The cost of a horse in rupees -/
def horse_cost : ℚ := 2000

/-- The cost of a camel in rupees -/
def camel_cost : ℚ := 4800

/-- The number of camels -/
def num_camels : ℚ := 10

theorem camel_count :
  (6 * ox_cost = 4 * elephant_cost) →
  (16 * horse_cost = 4 * ox_cost) →
  (24 * horse_cost = num_camels * camel_cost) →
  (10 * elephant_cost = 120000) →
  num_camels = 10 := by
  sorry

end NUMINAMATH_CALUDE_camel_count_l4097_409736


namespace NUMINAMATH_CALUDE_sector_central_angle_l4097_409775

theorem sector_central_angle (circumference area : ℝ) (h1 : circumference = 4) (h2 : area = 1) :
  let r := (4 - circumference) / 2
  let l := circumference - 2 * r
  l / r = 2 := by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l4097_409775


namespace NUMINAMATH_CALUDE_blue_balls_count_l4097_409772

theorem blue_balls_count (B : ℕ) : 
  (6 : ℚ) * 5 / ((8 + B) * (7 + B)) = 0.19230769230769232 → B = 5 := by
  sorry

end NUMINAMATH_CALUDE_blue_balls_count_l4097_409772


namespace NUMINAMATH_CALUDE_binomial_square_constant_l4097_409799

theorem binomial_square_constant (c : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 150*x + c = (x + a)^2) → c = 5625 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_constant_l4097_409799


namespace NUMINAMATH_CALUDE_det_scalar_mult_l4097_409703

def A : Matrix (Fin 2) (Fin 2) ℝ := !![5, -2; 4, 3]
def k : ℝ := 3

theorem det_scalar_mult :
  Matrix.det (k • A) = 207 := by sorry

end NUMINAMATH_CALUDE_det_scalar_mult_l4097_409703


namespace NUMINAMATH_CALUDE_rectangular_box_volume_l4097_409710

theorem rectangular_box_volume (l w h : ℝ) 
  (area1 : l * w = 30)
  (area2 : w * h = 20)
  (area3 : l * h = 12) :
  l * w * h = 60 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l4097_409710


namespace NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l4097_409773

theorem greatest_divisor_four_consecutive_integers :
  ∃ (k : ℕ), k > 0 ∧ 
  (∀ (n : ℕ), n > 0 → (k ∣ n * (n + 1) * (n + 2) * (n + 3))) ∧
  (∀ (m : ℕ), m > k → ∃ (n : ℕ), n > 0 ∧ ¬(m ∣ n * (n + 1) * (n + 2) * (n + 3))) :=
by
  use 24
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l4097_409773


namespace NUMINAMATH_CALUDE_slope_range_l4097_409756

theorem slope_range (m : ℝ) : ((8 - m) / (m - 5) > 1) → (5 < m ∧ m < 13/2) := by
  sorry

end NUMINAMATH_CALUDE_slope_range_l4097_409756


namespace NUMINAMATH_CALUDE_minimum_value_implies_a_l4097_409751

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x + 3

theorem minimum_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, f a x ≥ 4) ∧
  (∃ x ∈ Set.Icc 1 2, f a x = 4) →
  a = Real.exp 1 - 1 := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_implies_a_l4097_409751


namespace NUMINAMATH_CALUDE_product_of_one_plus_roots_l4097_409742

theorem product_of_one_plus_roots (p q r : ℝ) : 
  (p^3 - 15*p^2 + 25*p - 10 = 0) →
  (q^3 - 15*q^2 + 25*q - 10 = 0) →
  (r^3 - 15*r^2 + 25*r - 10 = 0) →
  (1 + p) * (1 + q) * (1 + r) = 51 := by
sorry

end NUMINAMATH_CALUDE_product_of_one_plus_roots_l4097_409742


namespace NUMINAMATH_CALUDE_product_equals_two_l4097_409748

theorem product_equals_two : 10 * (1/5) * 4 * (1/16) * (1/2) * 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_two_l4097_409748


namespace NUMINAMATH_CALUDE_triangle_sine_inequality_l4097_409702

theorem triangle_sine_inequality (A B C : ℝ) (h : A + B + C = π) :
  Real.sin A * Real.sin (A/2) + Real.sin B * Real.sin (B/2) + Real.sin C * Real.sin (C/2) ≤ 4 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_inequality_l4097_409702


namespace NUMINAMATH_CALUDE_favorite_movies_total_length_l4097_409737

theorem favorite_movies_total_length : 
  ∀ (michael joyce nikki ryn sam alex : ℝ),
    nikki = 30 →
    michael = nikki / 3 →
    joyce = michael + 2 →
    ryn = nikki * (4/5) →
    sam = joyce * 1.5 →
    alex = 2 * (min michael (min joyce (min nikki (min ryn sam)))) →
    michael + joyce + nikki + ryn + sam + alex = 114 :=
by
  sorry

end NUMINAMATH_CALUDE_favorite_movies_total_length_l4097_409737


namespace NUMINAMATH_CALUDE_smallest_five_digit_multiple_l4097_409709

theorem smallest_five_digit_multiple : ∃ n : ℕ,
  (n ≥ 10000 ∧ n < 100000) ∧  -- 5-digit number
  (15 ∣ n) ∧ (32 ∣ n) ∧ (9 ∣ n) ∧ (5 ∣ n) ∧ (54 ∣ n) ∧
  (∀ m : ℕ, m ≥ 10000 ∧ m < 100000 ∧ 
    (15 ∣ m) ∧ (32 ∣ m) ∧ (9 ∣ m) ∧ (5 ∣ m) ∧ (54 ∣ m) → n ≤ m) ∧
  n = 17280 :=
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_multiple_l4097_409709


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l4097_409700

theorem algebraic_expression_equality (x y : ℝ) (h : 2*x - 3*y = 1) : 
  6*y - 4*x + 8 = 6 := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l4097_409700


namespace NUMINAMATH_CALUDE_oldest_child_age_l4097_409783

theorem oldest_child_age (a b c : ℕ) (h1 : a = 6) (h2 : b = 8) 
  (h3 : (a + b + c) / 3 = 9) : c = 13 := by
  sorry

end NUMINAMATH_CALUDE_oldest_child_age_l4097_409783


namespace NUMINAMATH_CALUDE_volunteer_assignment_count_l4097_409715

theorem volunteer_assignment_count : 
  (volunteers : ℕ) → 
  (pavilions : ℕ) → 
  volunteers = 5 → 
  pavilions = 4 → 
  (arrangements : ℕ) → 
  arrangements = 240 := by sorry

end NUMINAMATH_CALUDE_volunteer_assignment_count_l4097_409715


namespace NUMINAMATH_CALUDE_luke_used_eight_stickers_l4097_409760

/-- The number of stickers Luke used to decorate the greeting card -/
def stickers_used_for_card (initial_stickers bought_stickers birthday_stickers : ℕ)
  (given_to_sister remaining_stickers : ℕ) : ℕ :=
  initial_stickers + bought_stickers + birthday_stickers - given_to_sister - remaining_stickers

/-- Theorem stating that Luke used 8 stickers to decorate the greeting card -/
theorem luke_used_eight_stickers :
  stickers_used_for_card 20 12 20 5 39 = 8 := by
  sorry

end NUMINAMATH_CALUDE_luke_used_eight_stickers_l4097_409760


namespace NUMINAMATH_CALUDE_wage_increase_l4097_409757

theorem wage_increase (original_wage : ℝ) (increase_percentage : ℝ) 
  (h1 : original_wage = 60)
  (h2 : increase_percentage = 20) : 
  original_wage * (1 + increase_percentage / 100) = 72 := by
  sorry

end NUMINAMATH_CALUDE_wage_increase_l4097_409757


namespace NUMINAMATH_CALUDE_sports_club_overlap_l4097_409771

theorem sports_club_overlap (total : ℕ) (badminton tennis neither : ℕ) 
  (h_total : total = 30)
  (h_badminton : badminton = 17)
  (h_tennis : tennis = 17)
  (h_neither : neither = 2)
  (h_sum : total = badminton + tennis - (total - neither)) :
  badminton + tennis - (total - neither) = 6 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_overlap_l4097_409771


namespace NUMINAMATH_CALUDE_next_but_one_perfect_square_l4097_409787

theorem next_but_one_perfect_square (x : ℕ) (h : ∃ k : ℕ, x = k^2) :
  ∃ n : ℕ, n > x ∧ (∃ m : ℕ, m > x ∧ m < n ∧ ∃ k : ℕ, m = k^2) ∧ n = x + 4 * Int.sqrt x + 4 :=
sorry

end NUMINAMATH_CALUDE_next_but_one_perfect_square_l4097_409787


namespace NUMINAMATH_CALUDE_nucleic_acid_test_is_comprehensive_l4097_409797

/-- Represents a survey method -/
inductive SurveyMethod
| BallpointPenRefills
| FoodProducts
| CarCrashResistance
| NucleicAcidTest

/-- Predicate to determine if a survey method destroys its subjects -/
def destroysSubjects (method : SurveyMethod) : Prop :=
  match method with
  | SurveyMethod.BallpointPenRefills => true
  | SurveyMethod.FoodProducts => true
  | SurveyMethod.CarCrashResistance => true
  | SurveyMethod.NucleicAcidTest => false

/-- Definition of a comprehensive survey -/
def isComprehensiveSurvey (method : SurveyMethod) : Prop :=
  ¬(destroysSubjects method)

/-- Theorem: Nucleic Acid Test is suitable for a comprehensive survey -/
theorem nucleic_acid_test_is_comprehensive :
  isComprehensiveSurvey SurveyMethod.NucleicAcidTest :=
by
  sorry

#check nucleic_acid_test_is_comprehensive

end NUMINAMATH_CALUDE_nucleic_acid_test_is_comprehensive_l4097_409797


namespace NUMINAMATH_CALUDE_games_won_is_fifteen_l4097_409729

/-- Represents the number of baseball games played by Dan's high school team. -/
def total_games : ℕ := 18

/-- Represents the number of games lost by Dan's high school team. -/
def games_lost : ℕ := 3

/-- Theorem stating that the number of games won is 15. -/
theorem games_won_is_fifteen : total_games - games_lost = 15 := by
  sorry

end NUMINAMATH_CALUDE_games_won_is_fifteen_l4097_409729
