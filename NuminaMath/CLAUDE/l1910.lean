import Mathlib

namespace NUMINAMATH_CALUDE_sin_50_plus_sqrt3_tan_10_equals_1_l1910_191099

theorem sin_50_plus_sqrt3_tan_10_equals_1 :
  Real.sin (50 * π / 180) * (1 + Real.sqrt 3 * Real.tan (10 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_50_plus_sqrt3_tan_10_equals_1_l1910_191099


namespace NUMINAMATH_CALUDE_phd_time_calculation_l1910_191096

/-- Calculates the total time John spent on his PhD --/
def total_phd_time (acclimation_time : ℝ) (basics_time : ℝ) (research_multiplier : ℝ) 
  (sabbatical_time : ℝ) (dissertation_fraction : ℝ) (conference_time : ℝ) : ℝ :=
  let research_time := basics_time * (1 + research_multiplier) + sabbatical_time
  let dissertation_time := acclimation_time * dissertation_fraction + conference_time
  acclimation_time + basics_time + research_time + dissertation_time

theorem phd_time_calculation :
  total_phd_time 1 2 0.75 0.5 0.5 0.25 = 7.75 := by
  sorry

end NUMINAMATH_CALUDE_phd_time_calculation_l1910_191096


namespace NUMINAMATH_CALUDE_vector_perpendicular_l1910_191011

def problem1 (p q : ℝ × ℝ) : Prop :=
  p = (1, 2) ∧ 
  ∃ m : ℝ, q = (m, 1) ∧ 
  p.1 * q.1 + p.2 * q.2 = 0 →
  ‖q‖ = Real.sqrt 5

theorem vector_perpendicular : problem1 (1, 2) (-2, 1) := by sorry

end NUMINAMATH_CALUDE_vector_perpendicular_l1910_191011


namespace NUMINAMATH_CALUDE_max_value_ab_l1910_191005

theorem max_value_ab (a b : ℝ) : 
  (∀ x : ℝ, Real.exp (x + 1) ≥ a * x + b) → 
  a * b ≤ (1/2) * Real.exp 3 := by
sorry

end NUMINAMATH_CALUDE_max_value_ab_l1910_191005


namespace NUMINAMATH_CALUDE_sandcastle_height_difference_l1910_191028

/-- The height difference between Janet's sandcastle and her sister's sandcastle -/
def height_difference : ℝ := 1.333333333333333

/-- Janet's sandcastle height in feet -/
def janet_height : ℝ := 3.6666666666666665

/-- Janet's sister's sandcastle height in feet -/
def sister_height : ℝ := 2.3333333333333335

/-- Theorem stating that the height difference between Janet's sandcastle and her sister's sandcastle
    is equal to Janet's sandcastle height minus her sister's sandcastle height -/
theorem sandcastle_height_difference :
  height_difference = janet_height - sister_height := by
  sorry

end NUMINAMATH_CALUDE_sandcastle_height_difference_l1910_191028


namespace NUMINAMATH_CALUDE_smallest_number_l1910_191084

theorem smallest_number (S : Set ℕ) (h : S = {10, 11, 12, 13, 14}) : 
  ∃ n ∈ S, ∀ m ∈ S, n ≤ m ∧ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l1910_191084


namespace NUMINAMATH_CALUDE_noodle_portions_l1910_191063

-- Define the variables
def total_spent : ℕ := 3000
def total_portions : ℕ := 170
def price_mixed : ℕ := 15
def price_beef : ℕ := 20

-- Define the theorem
theorem noodle_portions :
  ∃ (mixed beef : ℕ),
    mixed + beef = total_portions ∧
    price_mixed * mixed + price_beef * beef = total_spent ∧
    mixed = 80 ∧
    beef = 90 := by
  sorry

end NUMINAMATH_CALUDE_noodle_portions_l1910_191063


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l1910_191076

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1: Solution set of f(x) ≥ 6 when a = 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2} :=
sorry

-- Part 2: Range of a for which f(x) > -a for all x
theorem range_of_a_part2 :
  {a : ℝ | ∀ x, f a x > -a} = {a : ℝ | a > -3/2} :=
sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l1910_191076


namespace NUMINAMATH_CALUDE_binomial_square_constant_l1910_191038

theorem binomial_square_constant (c : ℝ) : 
  (∃ a b : ℝ, ∀ x, 9*x^2 + 30*x + c = (a*x + b)^2) → c = 25 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_constant_l1910_191038


namespace NUMINAMATH_CALUDE_unique_valid_config_l1910_191039

/-- Represents a fence configuration --/
structure FenceConfig where
  max_length : Nat
  num_max : Nat
  num_minus_one : Nat
  num_minus_two : Nat
  num_minus_three : Nat

/-- Checks if a fence configuration is valid --/
def is_valid_config (config : FenceConfig) : Prop :=
  config.num_max + config.num_minus_one + config.num_minus_two + config.num_minus_three = 16 ∧
  config.num_max * config.max_length +
  config.num_minus_one * (config.max_length - 1) +
  config.num_minus_two * (config.max_length - 2) +
  config.num_minus_three * (config.max_length - 3) = 297 ∧
  config.num_max = 8

/-- The unique valid fence configuration --/
def unique_config : FenceConfig :=
  { max_length := 20
  , num_max := 8
  , num_minus_one := 0
  , num_minus_two := 7
  , num_minus_three := 1
  }

/-- Theorem stating that the unique_config is the only valid configuration --/
theorem unique_valid_config :
  is_valid_config unique_config ∧
  (∀ config : FenceConfig, is_valid_config config → config = unique_config) := by
  sorry


end NUMINAMATH_CALUDE_unique_valid_config_l1910_191039


namespace NUMINAMATH_CALUDE_pencil_distribution_result_l1910_191036

/-- Represents the pencil distribution problem --/
structure PencilDistribution where
  gloria_initial : ℕ
  lisa_initial : ℕ
  tim_initial : ℕ

/-- Calculates the final pencil counts after Lisa's distribution --/
def final_counts (pd : PencilDistribution) : ℕ × ℕ × ℕ :=
  let lisa_half := pd.lisa_initial / 2
  (pd.gloria_initial + lisa_half, 0, pd.tim_initial + lisa_half)

/-- Theorem stating the final pencil counts after distribution --/
theorem pencil_distribution_result (pd : PencilDistribution)
  (h1 : pd.gloria_initial = 2500)
  (h2 : pd.lisa_initial = 75800)
  (h3 : pd.tim_initial = 1950) :
  final_counts pd = (40400, 0, 39850) := by
  sorry

end NUMINAMATH_CALUDE_pencil_distribution_result_l1910_191036


namespace NUMINAMATH_CALUDE_triangle_area_is_24_l1910_191021

-- Define the points
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (6, 0)
def C : ℝ × ℝ := (0, 8)

-- Define the equation
def satisfies_equation (p : ℝ × ℝ) : Prop :=
  |4 * p.1| + |3 * p.2| + |24 - 4 * p.1 - 3 * p.2| = 24

-- Theorem statement
theorem triangle_area_is_24 :
  satisfies_equation A ∧ satisfies_equation B ∧ satisfies_equation C →
  (1/2 : ℝ) * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)| = 24 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_is_24_l1910_191021


namespace NUMINAMATH_CALUDE_animals_per_aquarium_l1910_191050

/-- Given that Tyler has 56 saltwater aquariums and 2184 saltwater animals,
    prove that there are 39 animals in each saltwater aquarium. -/
theorem animals_per_aquarium (saltwater_aquariums : ℕ) (saltwater_animals : ℕ)
    (h1 : saltwater_aquariums = 56)
    (h2 : saltwater_animals = 2184) :
    saltwater_animals / saltwater_aquariums = 39 := by
  sorry

end NUMINAMATH_CALUDE_animals_per_aquarium_l1910_191050


namespace NUMINAMATH_CALUDE_ellipse_region_area_l1910_191085

/-- The area of the region formed by all points on ellipses passing through (√3, 1) where y ≥ 1 -/
theorem ellipse_region_area :
  ∀ a b : ℝ,
  a ≥ b ∧ b > 0 →
  (3 / a^2) + (1 / b^2) = 1 →
  (∃ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ∧ y ≥ 1) →
  (∃ area : ℝ, area = 4 * Real.pi / 3 - Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_region_area_l1910_191085


namespace NUMINAMATH_CALUDE_sum_and_product_reciprocal_sum_cube_surface_area_probability_white_ball_equilateral_triangle_area_l1910_191071

-- Problem 1
theorem sum_and_product_reciprocal_sum (x y : ℝ) (h1 : x + y = 40) (h2 : x * y = 20) :
  1 / x + 1 / y = 2 := by sorry

-- Problem 2
theorem cube_surface_area (a : ℝ) :
  6 * (a + 1)^2 = 54 := by sorry

-- Problem 3
theorem probability_white_ball (b : ℝ) (c : ℝ) :
  (b - 4) / (2 * b + 42) = c / 6 := by sorry

-- Problem 4
theorem equilateral_triangle_area (c d : ℝ) :
  d * Real.sqrt 3 = (Real.sqrt 3 / 4) * c^2 := by sorry

end NUMINAMATH_CALUDE_sum_and_product_reciprocal_sum_cube_surface_area_probability_white_ball_equilateral_triangle_area_l1910_191071


namespace NUMINAMATH_CALUDE_inverse_proportion_ratio_l1910_191001

/-- Given that x is inversely proportional to y, this function represents their relationship -/
def inverse_proportion (x y : ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ x * y = k

theorem inverse_proportion_ratio 
  (x₁ x₂ y₁ y₂ : ℝ) 
  (hx₁ : x₁ ≠ 0) (hx₂ : x₂ ≠ 0) (hy₁ : y₁ ≠ 0) (hy₂ : y₂ ≠ 0)
  (hxy₁ : inverse_proportion x₁ y₁)
  (hxy₂ : inverse_proportion x₂ y₂)
  (hx_ratio : x₁ / x₂ = 3 / 4) :
  y₁ / y₂ = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_ratio_l1910_191001


namespace NUMINAMATH_CALUDE_incorrect_operation_l1910_191066

theorem incorrect_operation : 
  (5 - (-2) = 7) ∧ 
  (-9 / (-3) = 3) ∧ 
  (-4 * (-5) = 20) ∧ 
  (-5 + 3 ≠ 8) := by
sorry

end NUMINAMATH_CALUDE_incorrect_operation_l1910_191066


namespace NUMINAMATH_CALUDE_characterization_of_M_inequality_for_M_elements_l1910_191034

-- Define the set M
def M : Set ℝ := {x : ℝ | |2*x - 1| < 1}

-- Theorem 1: Characterization of set M
theorem characterization_of_M : M = {x : ℝ | 0 < x ∧ x < 1} := by sorry

-- Theorem 2: Inequality for elements in M
theorem inequality_for_M_elements (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) :
  a * b + 1 > a + b := by sorry

end NUMINAMATH_CALUDE_characterization_of_M_inequality_for_M_elements_l1910_191034


namespace NUMINAMATH_CALUDE_pizza_consumption_order_l1910_191079

-- Define the fractions of pizza eaten by each friend
def samuel_fraction : ℚ := 1/6
def teresa_fraction : ℚ := 2/5
def uma_fraction : ℚ := 1/4

-- Define the amount of pizza eaten by Victor
def victor_fraction : ℚ := 1 - (samuel_fraction + teresa_fraction + uma_fraction)

-- Define a function to compare two fractions
def eats_more (a b : ℚ) : Prop := a > b

-- Theorem stating the order of pizza consumption
theorem pizza_consumption_order :
  eats_more teresa_fraction uma_fraction ∧
  eats_more uma_fraction victor_fraction ∧
  eats_more victor_fraction samuel_fraction :=
sorry

end NUMINAMATH_CALUDE_pizza_consumption_order_l1910_191079


namespace NUMINAMATH_CALUDE_lcm_48_90_l1910_191074

theorem lcm_48_90 : Nat.lcm 48 90 = 720 := by
  sorry

end NUMINAMATH_CALUDE_lcm_48_90_l1910_191074


namespace NUMINAMATH_CALUDE_not_necessary_not_sufficient_l1910_191082

theorem not_necessary_not_sufficient (a : ℝ) : 
  ¬(∀ a, a < 2 → a^2 < 2*a) ∧ ¬(∀ a, a^2 < 2*a → a < 2) := by
  sorry

end NUMINAMATH_CALUDE_not_necessary_not_sufficient_l1910_191082


namespace NUMINAMATH_CALUDE_one_third_percent_of_180_l1910_191054

theorem one_third_percent_of_180 : (1 / 3 : ℚ) / 100 * 180 = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_one_third_percent_of_180_l1910_191054


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l1910_191059

theorem quadratic_discriminant : 
  let a : ℝ := 1
  let b : ℝ := -7
  let c : ℝ := 4
  (b^2 - 4*a*c) = 33 := by
sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l1910_191059


namespace NUMINAMATH_CALUDE_root_interval_sum_l1910_191037

def f (x : ℝ) := x^3 - x + 1

theorem root_interval_sum (a b : ℤ) : 
  (∃ x : ℝ, a < x ∧ x < b ∧ f x = 0) →
  b - a = 1 →
  a + b = -3 := by
sorry

end NUMINAMATH_CALUDE_root_interval_sum_l1910_191037


namespace NUMINAMATH_CALUDE_fruit_purchase_cost_l1910_191000

theorem fruit_purchase_cost (strawberry_price : ℝ) (cherry_price : ℝ) (blueberry_price : ℝ)
  (strawberry_amount : ℝ) (cherry_amount : ℝ) (blueberry_amount : ℝ)
  (blueberry_discount : ℝ) (bag_fee : ℝ) :
  strawberry_price = 2.20 →
  cherry_price = 6 * strawberry_price →
  blueberry_price = cherry_price / 2 →
  strawberry_amount = 3 →
  cherry_amount = 4.5 →
  blueberry_amount = 6.2 →
  blueberry_discount = 0.15 →
  bag_fee = 0.75 →
  strawberry_price * strawberry_amount +
  cherry_price * cherry_amount +
  blueberry_price * blueberry_amount * (1 - blueberry_discount) +
  bag_fee = 101.53 := by
sorry

end NUMINAMATH_CALUDE_fruit_purchase_cost_l1910_191000


namespace NUMINAMATH_CALUDE_bernoulli_inequality_l1910_191027

theorem bernoulli_inequality (x : ℝ) (n : ℕ) 
  (h1 : x > -1) (h2 : x ≠ 0) (h3 : n > 1) : 
  (1 + x)^n > 1 + n * x := by
  sorry

end NUMINAMATH_CALUDE_bernoulli_inequality_l1910_191027


namespace NUMINAMATH_CALUDE_max_profit_pork_zongzi_l1910_191053

/-- Represents the wholesale and retail prices of zongzi -/
structure ZongziPrices where
  porkWholesale : ℝ
  redBeanWholesale : ℝ
  porkRetail : ℝ

/-- Represents the daily sales and profit of pork zongzi -/
structure PorkZongziSales where
  price : ℝ
  quantity : ℝ
  profit : ℝ

/-- The conditions given in the problem -/
def zongziConditions (z : ZongziPrices) : Prop :=
  z.porkWholesale = z.redBeanWholesale + 10 ∧
  z.porkWholesale + 2 * z.redBeanWholesale = 100

/-- The relationship between price and quantity sold for pork zongzi -/
def porkZongziDemand (basePrice baseQuantity : ℝ) (z : ZongziPrices) (s : PorkZongziSales) : Prop :=
  s.quantity = baseQuantity - 2 * (s.price - basePrice)

/-- The profit function for pork zongzi -/
def porkZongziProfit (z : ZongziPrices) (s : PorkZongziSales) : Prop :=
  s.profit = (s.price - z.porkWholesale) * s.quantity

/-- The main theorem stating the maximum profit -/
theorem max_profit_pork_zongzi (z : ZongziPrices) (s : PorkZongziSales) :
  zongziConditions z →
  porkZongziDemand 50 100 z s →
  porkZongziProfit z s →
  ∃ maxProfit : ℝ, maxProfit = 1800 ∧ ∀ s', porkZongziProfit z s' → s'.profit ≤ maxProfit :=
sorry

end NUMINAMATH_CALUDE_max_profit_pork_zongzi_l1910_191053


namespace NUMINAMATH_CALUDE_wendys_washing_machine_capacity_l1910_191048

-- Define the number of shirts
def shirts : ℕ := 39

-- Define the number of sweaters
def sweaters : ℕ := 33

-- Define the number of loads
def loads : ℕ := 9

-- Define the function to calculate the washing machine capacity
def washing_machine_capacity (s : ℕ) (w : ℕ) (l : ℕ) : ℕ :=
  (s + w) / l

-- Theorem statement
theorem wendys_washing_machine_capacity :
  washing_machine_capacity shirts sweaters loads = 8 := by
  sorry

end NUMINAMATH_CALUDE_wendys_washing_machine_capacity_l1910_191048


namespace NUMINAMATH_CALUDE_current_speed_l1910_191026

/-- Calculates the speed of the current given the rowing speed in still water and the time taken to cover a distance downstream. -/
theorem current_speed (rowing_speed : ℝ) (distance : ℝ) (time : ℝ) : 
  rowing_speed = 30 →
  distance = 100 →
  time = 9.99920006399488 →
  (distance / time) * 3.6 - rowing_speed = 6 := by
  sorry

#eval (100 / 9.99920006399488) * 3.6 - 30

end NUMINAMATH_CALUDE_current_speed_l1910_191026


namespace NUMINAMATH_CALUDE_fiona_earnings_l1910_191069

-- Define the time worked each day in hours
def monday_hours : ℝ := 1.5
def tuesday_hours : ℝ := 1.25
def wednesday_hours : ℝ := 3.1667
def thursday_hours : ℝ := 0.75

-- Define the hourly rate
def hourly_rate : ℝ := 4

-- Define the total hours worked
def total_hours : ℝ := monday_hours + tuesday_hours + wednesday_hours + thursday_hours

-- Define the weekly earnings
def weekly_earnings : ℝ := total_hours * hourly_rate

-- Theorem statement
theorem fiona_earnings : 
  ∃ ε > 0, |weekly_earnings - 26.67| < ε :=
sorry

end NUMINAMATH_CALUDE_fiona_earnings_l1910_191069


namespace NUMINAMATH_CALUDE_min_output_no_loss_l1910_191067

-- Define the total cost function
def total_cost (x : ℝ) : ℝ := 3000 + 20 * x - 0.1 * x^2

-- Define the sales revenue function
def sales_revenue (x : ℝ) : ℝ := 25 * x

-- Define the domain constraint
def in_domain (x : ℝ) : Prop := 0 < x ∧ x < 240

-- Theorem statement
theorem min_output_no_loss :
  ∃ (x_min : ℝ), x_min = 150 ∧
  in_domain x_min ∧
  (∀ x : ℝ, in_domain x → sales_revenue x ≥ total_cost x → x ≥ x_min) :=
sorry

end NUMINAMATH_CALUDE_min_output_no_loss_l1910_191067


namespace NUMINAMATH_CALUDE_intersection_M_N_l1910_191009

def M : Set ℝ := {-3, 1, 3}
def N : Set ℝ := {x | x^2 - 3*x - 4 < 0}

theorem intersection_M_N : M ∩ N = {1, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1910_191009


namespace NUMINAMATH_CALUDE_complex_square_sum_l1910_191073

theorem complex_square_sum (a b : ℝ) (i : ℂ) : 
  i * i = -1 → (2 - i)^2 = a + b * i^3 → a + b = 7 := by
  sorry

end NUMINAMATH_CALUDE_complex_square_sum_l1910_191073


namespace NUMINAMATH_CALUDE_red_packet_probability_l1910_191098

def red_packet_amounts : List ℝ := [1.49, 1.31, 2.19, 3.40, 0.61]

def total_amount : ℝ := 9

def num_people : ℕ := 5

def threshold : ℝ := 4

def probability_ab_sum_ge_threshold (amounts : List ℝ) (total : ℝ) (n : ℕ) (t : ℝ) : ℚ :=
  sorry

theorem red_packet_probability :
  probability_ab_sum_ge_threshold red_packet_amounts total_amount num_people threshold = 2/5 :=
sorry

end NUMINAMATH_CALUDE_red_packet_probability_l1910_191098


namespace NUMINAMATH_CALUDE_left_of_origin_abs_value_l1910_191006

theorem left_of_origin_abs_value (a : ℝ) : 
  (a < 0) → (|a| = 4.5) → (a = -4.5) := by sorry

end NUMINAMATH_CALUDE_left_of_origin_abs_value_l1910_191006


namespace NUMINAMATH_CALUDE_vidyas_age_multiple_l1910_191044

theorem vidyas_age_multiple (vidya_age mother_age : ℕ) (h1 : vidya_age = 13) (h2 : mother_age = 44) :
  ∃ m : ℕ, m * vidya_age + 5 = mother_age ∧ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_vidyas_age_multiple_l1910_191044


namespace NUMINAMATH_CALUDE_point_inside_circle_implies_a_range_l1910_191030

/-- The circle with equation (x-a)^2 + (y+a)^2 = 4 -/
def Circle (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - a)^2 + (p.2 + a)^2 = 4}

/-- A point is inside the circle if its distance from the center is less than the radius -/
def IsInside (p : ℝ × ℝ) (a : ℝ) : Prop :=
  (p.1 - a)^2 + (p.2 + a)^2 < 4

/-- The theorem stating that if P(1,1) is inside the circle, then -1 < a < 1 -/
theorem point_inside_circle_implies_a_range :
  ∀ a : ℝ, IsInside (1, 1) a → -1 < a ∧ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_point_inside_circle_implies_a_range_l1910_191030


namespace NUMINAMATH_CALUDE_cafeteria_pies_l1910_191083

theorem cafeteria_pies (total_apples : Real) (handed_out : Real) (apples_per_pie : Real) 
  (h1 : total_apples = 135.5)
  (h2 : handed_out = 89.75)
  (h3 : apples_per_pie = 5.25) :
  ⌊(total_apples - handed_out) / apples_per_pie⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_pies_l1910_191083


namespace NUMINAMATH_CALUDE_sofia_survey_l1910_191062

theorem sofia_survey (mashed_potatoes bacon : ℕ) 
  (h1 : mashed_potatoes = 185) 
  (h2 : bacon = 125) : 
  mashed_potatoes + bacon = 310 := by
sorry

end NUMINAMATH_CALUDE_sofia_survey_l1910_191062


namespace NUMINAMATH_CALUDE_parabola_shift_sum_l1910_191075

/-- Represents a parabola of the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift (p : Parabola) (h v : ℝ) : Parabola :=
  { a := p.a
  , b := -2 * p.a * h + p.b
  , c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_shift_sum (p : Parabola) :
  (shift (shift p 1 0) 0 2) = { a := 1, b := -4, c := 5 } →
  p.a + p.b + p.c = 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_sum_l1910_191075


namespace NUMINAMATH_CALUDE_probability_both_truth_l1910_191019

theorem probability_both_truth (prob_A prob_B : ℝ) 
  (h1 : prob_A = 0.8) (h2 : prob_B = 0.6) :
  prob_A * prob_B = 0.48 := by
sorry

end NUMINAMATH_CALUDE_probability_both_truth_l1910_191019


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_inequality_l1910_191064

def a (n : ℕ+) : ℝ := 3 * 2^(n.val - 1)

def S (n : ℕ+) : ℝ := 3 * (2^n.val - 1)

theorem geometric_sequence_sum_inequality {k : ℝ} :
  (∀ n : ℕ+, a (n + 1) + a n = 9 * 2^(n.val - 1)) →
  (∀ n : ℕ+, S n > k * a n - 2) →
  k < 5/3 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_inequality_l1910_191064


namespace NUMINAMATH_CALUDE_expression_equals_24_l1910_191047

def arithmetic_expression (a b c d : ℕ) : Prop :=
  ∃ (e : ℕ → ℕ → ℕ → ℕ → ℕ), e a b c d = 24

theorem expression_equals_24 : arithmetic_expression 8 8 8 10 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_24_l1910_191047


namespace NUMINAMATH_CALUDE_z_relation_to_x_minus_2y_l1910_191018

theorem z_relation_to_x_minus_2y (x y z : ℝ) 
  (h1 : x > y) (h2 : y > 1) (h3 : z = (x + 3) - 2 * (y - 5)) :
  z = x - 2 * y + 13 := by
  sorry

end NUMINAMATH_CALUDE_z_relation_to_x_minus_2y_l1910_191018


namespace NUMINAMATH_CALUDE_inscribed_triangle_area_l1910_191033

/-- The area of a triangle inscribed in a circle, where the triangle's vertices
    divide the circle into three arcs of lengths 4, 5, and 7. -/
theorem inscribed_triangle_area : ∃ (A : ℝ), 
  (∀ (r : ℝ), r > 0 → r = 8 / Real.pi → 
    ∃ (θ₁ θ₂ θ₃ : ℝ), 
      θ₁ > 0 ∧ θ₂ > 0 ∧ θ₃ > 0 ∧
      4 * r = 4 * θ₁ ∧
      5 * r = 5 * θ₂ ∧
      7 * r = 7 * θ₃ ∧
      θ₁ + θ₂ + θ₃ = 2 * Real.pi ∧
      A = (1/2) * r^2 * (Real.sin (2*θ₁) + Real.sin (2*(θ₁+θ₂)) + Real.sin (2*(θ₁+θ₂+θ₃)))) ∧
  A = 147.6144 / Real.pi^2 := by
sorry

end NUMINAMATH_CALUDE_inscribed_triangle_area_l1910_191033


namespace NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l1910_191080

-- Define the plane and lines
variable (α : Plane)
variable (m n : Line)

-- Define the perpendicular and parallel relations
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel (l1 l2 : Line) : Prop := sorry

-- State the theorem
theorem lines_perpendicular_to_plane_are_parallel :
  perpendicular m α → perpendicular n α → parallel m n := by sorry

end NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l1910_191080


namespace NUMINAMATH_CALUDE_square_area_unchanged_l1910_191093

theorem square_area_unchanged (k : ℝ) : k > 0 → k^2 = 1 → k = 1 := by sorry

end NUMINAMATH_CALUDE_square_area_unchanged_l1910_191093


namespace NUMINAMATH_CALUDE_parallel_plane_intersection_lines_parallel_l1910_191008

-- Define the concept of a plane
variable (Plane : Type)

-- Define the concept of a line
variable (Line : Type)

-- Define the parallel relation between planes
variable (parallel_planes : Plane → Plane → Prop)

-- Define the intersection relation between a plane and a line
variable (intersects : Plane → Plane → Line → Prop)

-- Theorem statement
theorem parallel_plane_intersection_lines_parallel 
  (P1 P2 P3 : Plane) (l1 l2 : Line) :
  parallel_planes P1 P2 →
  intersects P3 P1 l1 →
  intersects P3 P2 l2 →
  -- Conclusion: l1 and l2 are parallel
  parallel_planes P1 P2 := by sorry

end NUMINAMATH_CALUDE_parallel_plane_intersection_lines_parallel_l1910_191008


namespace NUMINAMATH_CALUDE_power_division_equivalence_l1910_191002

theorem power_division_equivalence : 8^15 / 64^5 = 32768 := by
  have h1 : 8 = 2^3 := by sorry
  have h2 : 64 = 2^6 := by sorry
  sorry

end NUMINAMATH_CALUDE_power_division_equivalence_l1910_191002


namespace NUMINAMATH_CALUDE_negation_of_existence_square_leq_power_of_two_negation_l1910_191057

theorem negation_of_existence (p : Nat → Prop) :
  (¬ ∃ n : Nat, p n) ↔ (∀ n : Nat, ¬ p n) := by sorry

theorem square_leq_power_of_two_negation :
  (¬ ∃ n : Nat, n^2 > 2^n) ↔ (∀ n : Nat, n^2 ≤ 2^n) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_square_leq_power_of_two_negation_l1910_191057


namespace NUMINAMATH_CALUDE_widget_purchase_problem_l1910_191065

theorem widget_purchase_problem (C W : ℚ) 
  (h1 : 8 * (C - 1.25) = 16.67)
  (h2 : 16.67 / C = W) : 
  W = 5 := by
sorry

end NUMINAMATH_CALUDE_widget_purchase_problem_l1910_191065


namespace NUMINAMATH_CALUDE_real_part_of_complex_fraction_l1910_191024

theorem real_part_of_complex_fraction (i : ℂ) (h : i^2 = -1) : 
  Complex.re ((1 - 2*i) / (2 + i^5)) = 0 := by sorry

end NUMINAMATH_CALUDE_real_part_of_complex_fraction_l1910_191024


namespace NUMINAMATH_CALUDE_all_descendants_have_no_daughters_l1910_191041

/-- Represents Bertha's family tree -/
structure BerthaFamily where
  daughters : ℕ
  granddaughters : ℕ
  great_granddaughters : ℕ

/-- The number of Bertha's daughters who have daughters -/
def daughters_with_daughters (f : BerthaFamily) : ℕ := f.granddaughters / 5

/-- The number of Bertha's descendants who have no daughters -/
def descendants_without_daughters (f : BerthaFamily) : ℕ :=
  f.daughters + f.granddaughters

theorem all_descendants_have_no_daughters (f : BerthaFamily) :
  f.daughters = 8 →
  f.daughters + f.granddaughters + f.great_granddaughters = 48 →
  f.great_granddaughters = 0 →
  daughters_with_daughters f * 5 = f.granddaughters →
  descendants_without_daughters f = f.daughters + f.granddaughters + f.great_granddaughters :=
by sorry

end NUMINAMATH_CALUDE_all_descendants_have_no_daughters_l1910_191041


namespace NUMINAMATH_CALUDE_rectangular_prism_problem_l1910_191015

/-- Represents a rectangular prism with dimensions a, b, and c. -/
structure RectangularPrism where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Calculates the total number of faces of unit cubes in the prism. -/
def totalFaces (p : RectangularPrism) : ℕ := 6 * p.a * p.b * p.c

/-- Calculates the number of red faces in the prism. -/
def redFaces (p : RectangularPrism) : ℕ := 2 * (p.a * p.b + p.b * p.c + p.a * p.c)

/-- Theorem stating the conditions and result for the rectangular prism problem. -/
theorem rectangular_prism_problem (p : RectangularPrism) :
  p.a + p.b + p.c = 12 →
  3 * redFaces p = totalFaces p →
  p.a = 3 ∧ p.b = 4 ∧ p.c = 5 := by
  sorry


end NUMINAMATH_CALUDE_rectangular_prism_problem_l1910_191015


namespace NUMINAMATH_CALUDE_inverse_equals_one_implies_a_equals_one_l1910_191061

theorem inverse_equals_one_implies_a_equals_one (a : ℝ) (h : a ≠ 0) :
  a⁻¹ = (-1)^0 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_equals_one_implies_a_equals_one_l1910_191061


namespace NUMINAMATH_CALUDE_calculate_expression_l1910_191020

theorem calculate_expression : ((15^10 / 15^9)^3 * 5^3) / 3^3 = 15625 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1910_191020


namespace NUMINAMATH_CALUDE_business_school_majors_l1910_191051

theorem business_school_majors (p q r s : ℕ) (h1 : p * q * r * s = 1365) (h2 : p = 3) :
  q * r * s = 455 := by
  sorry

end NUMINAMATH_CALUDE_business_school_majors_l1910_191051


namespace NUMINAMATH_CALUDE_juice_bar_group_size_l1910_191087

theorem juice_bar_group_size :
  let total_spent : ℕ := 94
  let mango_price : ℕ := 5
  let pineapple_price : ℕ := 6
  let pineapple_spent : ℕ := 54
  let mango_spent : ℕ := total_spent - pineapple_spent
  let mango_people : ℕ := mango_spent / mango_price
  let pineapple_people : ℕ := pineapple_spent / pineapple_price
  mango_people + pineapple_people = 17 :=
by sorry

end NUMINAMATH_CALUDE_juice_bar_group_size_l1910_191087


namespace NUMINAMATH_CALUDE_point_side_line_range_l1910_191007

/-- Given that the points (3,-1) and (-4,-3) are on the same side of the line 3x-2y+a=0,
    prove that the range of values for a is (-∞,-11) ∪ (6,+∞). -/
theorem point_side_line_range (a : ℝ) : 
  (3 * 3 - 2 * (-1) + a) * (3 * (-4) - 2 * (-3) + a) > 0 ↔ 
  a ∈ Set.Iio (-11) ∪ Set.Ioi 6 :=
sorry

end NUMINAMATH_CALUDE_point_side_line_range_l1910_191007


namespace NUMINAMATH_CALUDE_reactions_not_usable_in_primary_cell_l1910_191003

-- Define the types of reactions
inductive ReactionType
| Neutralization
| Redox
| Endothermic

-- Define a structure for chemical reactions
structure ChemicalReaction where
  id : Nat
  reactionType : ReactionType
  isExothermic : Bool

-- Define the condition for a reaction to be used in a primary cell
def canBeUsedInPrimaryCell (reaction : ChemicalReaction) : Prop :=
  reaction.reactionType = ReactionType.Redox ∧ reaction.isExothermic

-- Define the given reactions
def reaction1 : ChemicalReaction :=
  { id := 1, reactionType := ReactionType.Neutralization, isExothermic := true }

def reaction2 : ChemicalReaction :=
  { id := 2, reactionType := ReactionType.Redox, isExothermic := true }

def reaction3 : ChemicalReaction :=
  { id := 3, reactionType := ReactionType.Redox, isExothermic := true }

def reaction4 : ChemicalReaction :=
  { id := 4, reactionType := ReactionType.Endothermic, isExothermic := false }

-- Theorem to prove
theorem reactions_not_usable_in_primary_cell :
  ¬(canBeUsedInPrimaryCell reaction1) ∧ ¬(canBeUsedInPrimaryCell reaction4) :=
sorry

end NUMINAMATH_CALUDE_reactions_not_usable_in_primary_cell_l1910_191003


namespace NUMINAMATH_CALUDE_clive_olive_money_l1910_191070

/-- Proves that Clive has $10.00 to spend on olives given the problem conditions -/
theorem clive_olive_money : 
  -- Define the given conditions
  let olives_needed : ℕ := 80
  let olives_per_jar : ℕ := 20
  let cost_per_jar : ℚ := 3/2  -- $1.50 represented as a rational number
  let change : ℚ := 4  -- $4.00 change

  -- Calculate the number of jars needed
  let jars_needed : ℕ := olives_needed / olives_per_jar

  -- Calculate the total cost of olives
  let total_cost : ℚ := jars_needed * cost_per_jar

  -- Define Clive's total money as the sum of total cost and change
  let clive_money : ℚ := total_cost + change

  -- Prove that Clive's total money is $10.00
  clive_money = 10 := by sorry

end NUMINAMATH_CALUDE_clive_olive_money_l1910_191070


namespace NUMINAMATH_CALUDE_budget_allocation_l1910_191023

/-- Given a family's budget allocation, calculate the fraction spent on eating out -/
theorem budget_allocation (budget_groceries : ℝ) (budget_total_food : ℝ) 
  (h1 : budget_groceries = 0.6) 
  (h2 : budget_total_food = 0.8) :
  budget_total_food - budget_groceries = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_budget_allocation_l1910_191023


namespace NUMINAMATH_CALUDE_practice_multiple_days_l1910_191056

/-- Given a person who practices a constant amount each day, and 20 days ago had half as much
    practice as they have currently, prove that it takes 40(M - 1) days to reach M times
    their current practice. -/
theorem practice_multiple_days (d : ℝ) (P : ℝ) (M : ℝ) :
  (P / 2 + 20 * d = P) →  -- 20 days ago, had half as much practice
  (P = 40 * d) →          -- Current practice
  (∃ D : ℝ, D * d = M * P - P ∧ D = 40 * (M - 1)) :=
by sorry

end NUMINAMATH_CALUDE_practice_multiple_days_l1910_191056


namespace NUMINAMATH_CALUDE_blood_type_sample_size_l1910_191013

/-- Given a population of students with known blood types, calculate the number of students
    with a specific blood type that should be drawn in a stratified sample. -/
theorem blood_type_sample_size (total_students sample_size blood_type_O : ℕ)
    (h1 : total_students = 500)
    (h2 : blood_type_O = 200)
    (h3 : sample_size = 40) :
    (blood_type_O : ℚ) / total_students * sample_size = 16 := by
  sorry


end NUMINAMATH_CALUDE_blood_type_sample_size_l1910_191013


namespace NUMINAMATH_CALUDE_largest_of_six_consecutive_odds_l1910_191017

theorem largest_of_six_consecutive_odds (a : ℕ) (h1 : a > 0) 
  (h2 : a % 2 = 1) 
  (h3 : (a * (a + 2) * (a + 4) * (a + 6) * (a + 8) * (a + 10) = 135135)) : 
  a + 10 = 13 := by
  sorry

end NUMINAMATH_CALUDE_largest_of_six_consecutive_odds_l1910_191017


namespace NUMINAMATH_CALUDE_unique_solution_exists_l1910_191094

/-- A function satisfying the given functional equation for a constant k -/
def SatisfiesEquation (f : ℝ → ℝ) (k : ℝ) : Prop :=
  ∀ x y : ℝ, f (x + f y) = x + y + k

/-- The theorem stating the uniqueness and form of the solution -/
theorem unique_solution_exists (k : ℝ) :
  ∃! f : ℝ → ℝ, SatisfiesEquation f k ∧ ∀ x : ℝ, f x = x - k :=
sorry

end NUMINAMATH_CALUDE_unique_solution_exists_l1910_191094


namespace NUMINAMATH_CALUDE_solution_mixture_l1910_191029

/-- Proves that 112 ounces of Solution B is needed to create a 140-ounce mixture
    that is 80% salt when combined with Solution A (40% salt) --/
theorem solution_mixture (solution_a_salt_percentage : ℝ) (solution_b_salt_percentage : ℝ)
  (total_mixture_ounces : ℝ) (target_salt_percentage : ℝ) :
  solution_a_salt_percentage = 0.4 →
  solution_b_salt_percentage = 0.9 →
  total_mixture_ounces = 140 →
  target_salt_percentage = 0.8 →
  ∃ (solution_b_ounces : ℝ),
    solution_b_ounces = 112 ∧
    solution_b_ounces + (total_mixture_ounces - solution_b_ounces) = total_mixture_ounces ∧
    solution_a_salt_percentage * (total_mixture_ounces - solution_b_ounces) +
      solution_b_salt_percentage * solution_b_ounces =
      target_salt_percentage * total_mixture_ounces :=
by sorry


end NUMINAMATH_CALUDE_solution_mixture_l1910_191029


namespace NUMINAMATH_CALUDE_one_pair_probability_l1910_191090

-- Define the total number of socks
def total_socks : ℕ := 12

-- Define the number of colors
def num_colors : ℕ := 4

-- Define the number of socks per color
def socks_per_color : ℕ := 3

-- Define the number of socks drawn
def socks_drawn : ℕ := 5

-- Define the probability of drawing exactly one pair of socks with the same color
def prob_one_pair : ℚ := 9/22

-- Theorem statement
theorem one_pair_probability :
  (total_socks = num_colors * socks_per_color) →
  (socks_drawn = 5) →
  (prob_one_pair = 9/22) := by
  sorry

end NUMINAMATH_CALUDE_one_pair_probability_l1910_191090


namespace NUMINAMATH_CALUDE_determinant_equality_l1910_191040

theorem determinant_equality (a b c d : ℝ) :
  Matrix.det ![![a, b], ![c, d]] = 5 →
  Matrix.det ![![a - c, b - d], ![c, d]] = 5 := by
  sorry

end NUMINAMATH_CALUDE_determinant_equality_l1910_191040


namespace NUMINAMATH_CALUDE_chess_competition_probabilities_l1910_191095

/-- Scoring system for the chess competition -/
structure ScoringSystem where
  win : Nat
  lose : Nat
  draw : Nat

/-- Probabilities for player A in a single game -/
structure PlayerProbabilities where
  win : Real
  lose : Real
  draw : Real

/-- Function to calculate the probability of player A scoring exactly 2 points in two games -/
def prob_A_scores_2 (s : ScoringSystem) (p : PlayerProbabilities) : Real :=
  sorry

/-- Function to calculate the probability of player B scoring at least 2 points in two games -/
def prob_B_scores_at_least_2 (s : ScoringSystem) (p : PlayerProbabilities) : Real :=
  sorry

theorem chess_competition_probabilities 
  (s : ScoringSystem) 
  (p : PlayerProbabilities) 
  (h1 : s.win = 2 ∧ s.lose = 0 ∧ s.draw = 1)
  (h2 : p.win = 0.5 ∧ p.lose = 0.3 ∧ p.draw = 0.2)
  (h3 : p.win + p.lose + p.draw = 1) :
  prob_A_scores_2 s p = 0.34 ∧ prob_B_scores_at_least_2 s p = 0.55 := by
  sorry

end NUMINAMATH_CALUDE_chess_competition_probabilities_l1910_191095


namespace NUMINAMATH_CALUDE_pigeonhole_multiples_of_five_l1910_191004

theorem pigeonhole_multiples_of_five (n : ℕ) (h : n = 200) : 
  ∀ (S : Finset ℕ), S ⊆ Finset.range n → S.card ≥ 82 → 
  ∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ (a + b) % 5 = 0 :=
by sorry

end NUMINAMATH_CALUDE_pigeonhole_multiples_of_five_l1910_191004


namespace NUMINAMATH_CALUDE_unique_base_for_315_l1910_191049

theorem unique_base_for_315 :
  ∃! b : ℕ, b ≥ 2 ∧ b^4 ≤ 315 ∧ 315 < b^5 :=
by sorry

end NUMINAMATH_CALUDE_unique_base_for_315_l1910_191049


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1910_191014

theorem inequality_system_solution (x : ℝ) :
  x > -6 - 2*x ∧ x ≤ (3 + x) / 4 → -2 < x ∧ x ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1910_191014


namespace NUMINAMATH_CALUDE_complex_abs_calculation_l1910_191045

def z : ℂ := 7 + 3 * Complex.I

theorem complex_abs_calculation : Complex.abs (z^2 + 4*z + 40) = 54 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_abs_calculation_l1910_191045


namespace NUMINAMATH_CALUDE_subtracted_number_l1910_191091

theorem subtracted_number (x y : ℤ) : x = 125 ∧ 2 * x - y = 112 → y = 138 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_number_l1910_191091


namespace NUMINAMATH_CALUDE_correct_answer_is_105_l1910_191058

theorem correct_answer_is_105 (x : ℕ) : 
  (x - 5 = 95) → (x + 5 = 105) :=
by
  sorry

end NUMINAMATH_CALUDE_correct_answer_is_105_l1910_191058


namespace NUMINAMATH_CALUDE_scalene_triangle_gp_ratio_bounds_l1910_191092

/-- A scalene triangle with sides in geometric progression -/
structure ScaleneTriangleGP where
  -- The first side of the triangle
  a : ℝ
  -- The common ratio of the geometric progression
  q : ℝ
  -- Ensure the triangle is scalene and sides are positive
  h_scalene : a ≠ a * q ∧ a * q ≠ a * q^2 ∧ a ≠ a * q^2 ∧ a > 0 ∧ q > 0

/-- The common ratio of a scalene triangle with sides in geometric progression
    must be between (1 - √5)/2 and (1 + √5)/2 -/
theorem scalene_triangle_gp_ratio_bounds (t : ScaleneTriangleGP) :
  (1 - Real.sqrt 5) / 2 < t.q ∧ t.q < (1 + Real.sqrt 5) / 2 :=
by sorry

end NUMINAMATH_CALUDE_scalene_triangle_gp_ratio_bounds_l1910_191092


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l1910_191042

theorem fraction_equals_zero (x : ℝ) : 
  (x^2 - 4) / (x - 2) = 0 ∧ x ≠ 2 → x = -2 :=
by sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l1910_191042


namespace NUMINAMATH_CALUDE_newton_basketball_league_members_l1910_191025

theorem newton_basketball_league_members :
  let headband_cost : ℕ := 3
  let jersey_cost : ℕ := headband_cost + 7
  let items_per_member : ℕ := 2  -- 2 headbands and 2 jerseys
  let total_cost : ℕ := 2700
  (total_cost = (headband_cost * items_per_member + jersey_cost * items_per_member) * 103) :=
by sorry

end NUMINAMATH_CALUDE_newton_basketball_league_members_l1910_191025


namespace NUMINAMATH_CALUDE_participation_and_optimality_l1910_191072

/-- Represents a company in country A --/
structure Company where
  investmentCost : ℝ
  successProbability : ℝ
  potentialRevenue : ℝ

/-- Conditions for the problem --/
axiom probability_bounds {α : ℝ} : 0 < α ∧ α < 1

/-- Expected income when both companies participate --/
def expectedIncomeBoth (c : Company) : ℝ :=
  c.successProbability * (1 - c.successProbability) * c.potentialRevenue +
  0.5 * c.successProbability^2 * c.potentialRevenue

/-- Expected income when only one company participates --/
def expectedIncomeOne (c : Company) : ℝ :=
  c.successProbability * c.potentialRevenue

/-- Condition for a company to participate --/
def willParticipate (c : Company) : Prop :=
  expectedIncomeBoth c - c.investmentCost ≥ 0

/-- Social welfare as total profit of both companies --/
def socialWelfare (c1 c2 : Company) : ℝ :=
  2 * (expectedIncomeBoth c1 - c1.investmentCost)

/-- Theorem stating both companies will participate and it's not socially optimal --/
theorem participation_and_optimality (c1 c2 : Company)
  (h1 : c1.potentialRevenue = 24 ∧ c1.successProbability = 0.5 ∧ c1.investmentCost = 7)
  (h2 : c2.potentialRevenue = 24 ∧ c2.successProbability = 0.5 ∧ c2.investmentCost = 7) :
  willParticipate c1 ∧ willParticipate c2 ∧
  socialWelfare c1 c2 < expectedIncomeOne c1 - c1.investmentCost := by
  sorry

end NUMINAMATH_CALUDE_participation_and_optimality_l1910_191072


namespace NUMINAMATH_CALUDE_min_value_of_function_l1910_191060

theorem min_value_of_function (x : ℝ) (h : 0 < x ∧ x < 1) :
  ∃ y : ℝ, y = 9 ∧ ∀ z : ℝ, (4 / x + 1 / (1 - x)) ≥ y := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_l1910_191060


namespace NUMINAMATH_CALUDE_probability_ratio_l1910_191046

/-- The number of slips in the hat -/
def total_slips : ℕ := 50

/-- The number of distinct numbers on the slips -/
def distinct_numbers : ℕ := 10

/-- The number of slips drawn -/
def drawn_slips : ℕ := 5

/-- The number of slips for each number -/
def slips_per_number : ℕ := 5

/-- The probability of drawing 5 slips with the same number -/
def p : ℚ := (distinct_numbers : ℚ) / (Nat.choose total_slips drawn_slips : ℚ)

/-- The probability of drawing 2 slips with one number and 3 slips with a different number -/
def q : ℚ := (Nat.choose distinct_numbers 2 * Nat.choose slips_per_number 2 * Nat.choose slips_per_number 3 : ℚ) / (Nat.choose total_slips drawn_slips : ℚ)

theorem probability_ratio :
  q / p = 450 := by sorry

end NUMINAMATH_CALUDE_probability_ratio_l1910_191046


namespace NUMINAMATH_CALUDE_no_resident_claims_to_be_liar_l1910_191012

-- Define the types of residents on the island
inductive Resident
| Knight
| Liar

-- Define the statement made by a resident
def makes_statement (r : Resident) : Prop :=
  match r with
  | Resident.Knight => True   -- Knights always tell the truth
  | Resident.Liar => False    -- Liars always lie

-- Define the statement "I am a liar"
def claims_to_be_liar (r : Resident) : Prop :=
  makes_statement r = (r = Resident.Liar)

-- Theorem: No resident can claim to be a liar
theorem no_resident_claims_to_be_liar :
  ∀ r : Resident, ¬(claims_to_be_liar r) :=
by sorry

end NUMINAMATH_CALUDE_no_resident_claims_to_be_liar_l1910_191012


namespace NUMINAMATH_CALUDE_coplanar_condition_l1910_191081

open Vector

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]
variable (O P Q R S : V)

-- Define the coplanarity condition
def coplanar (P Q R S : V) : Prop :=
  ∃ (a b c d : ℝ), a • (P - O) + b • (Q - O) + c • (R - O) + d • (S - O) = 0 ∧ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∨ d ≠ 0)

-- State the theorem
theorem coplanar_condition (O P Q R S : V) :
  4 • (P - O) - 3 • (Q - O) + 6 • (R - O) + (-7) • (S - O) = 0 →
  coplanar O P Q R S :=
by sorry

end NUMINAMATH_CALUDE_coplanar_condition_l1910_191081


namespace NUMINAMATH_CALUDE_cube_root_1728_l1910_191032

theorem cube_root_1728 (a b : ℕ+) (h1 : (1728 : ℝ)^(1/3) = a * b^(1/3)) 
  (h2 : ∀ c d : ℕ+, (1728 : ℝ)^(1/3) = c * d^(1/3) → b ≤ d) : 
  a + b = 13 := by sorry

end NUMINAMATH_CALUDE_cube_root_1728_l1910_191032


namespace NUMINAMATH_CALUDE_oil_distribution_l1910_191088

theorem oil_distribution (total oil_A oil_B oil_C : ℕ) : 
  total = 3000 →
  oil_A = oil_B + 200 →
  oil_B = oil_C + 200 →
  total = oil_A + oil_B + oil_C →
  oil_B = 1000 := by
  sorry

end NUMINAMATH_CALUDE_oil_distribution_l1910_191088


namespace NUMINAMATH_CALUDE_linear_function_preserves_arithmetic_progression_l1910_191077

/-- A sequence (xₙ) is an arithmetic progression if there exists a constant d
    such that xₙ₊₁ = xₙ + d for all n. -/
def is_arithmetic_progression (x : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, x (n + 1) = x n + d

/-- A function f is linear if there exist constants k and b such that
    f(x) = kx + b for all x. -/
def is_linear_function (f : ℝ → ℝ) : Prop :=
  ∃ k b : ℝ, ∀ x : ℝ, f x = k * x + b

theorem linear_function_preserves_arithmetic_progression
  (f : ℝ → ℝ) (x : ℕ → ℝ)
  (hf : is_linear_function f)
  (hx : is_arithmetic_progression x) :
  is_arithmetic_progression (fun n ↦ f (x n)) :=
sorry

end NUMINAMATH_CALUDE_linear_function_preserves_arithmetic_progression_l1910_191077


namespace NUMINAMATH_CALUDE_min_envelopes_correct_l1910_191078

/-- The number of different flags -/
def num_flags : ℕ := 12

/-- The number of flags in each envelope -/
def flags_per_envelope : ℕ := 2

/-- The probability threshold for having a repeated flag -/
def probability_threshold : ℚ := 1/2

/-- Calculates the probability of all flags being different when opening k envelopes -/
def prob_all_different (k : ℕ) : ℚ :=
  (num_flags.descFactorial (k * flags_per_envelope)) / (num_flags ^ (k * flags_per_envelope))

/-- The minimum number of envelopes to open -/
def min_envelopes : ℕ := 3

theorem min_envelopes_correct :
  (∀ k < min_envelopes, prob_all_different k > probability_threshold) ∧
  (prob_all_different min_envelopes ≤ probability_threshold) :=
sorry

end NUMINAMATH_CALUDE_min_envelopes_correct_l1910_191078


namespace NUMINAMATH_CALUDE_parabola_vertex_position_l1910_191097

/-- A parabola with points P, Q, and M satisfying specific conditions -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  y₁ : ℝ
  y₂ : ℝ
  y₃ : ℝ
  m : ℝ
  h1 : y₁ = a * (-2)^2 + b * (-2) + c
  h2 : y₂ = a * 4^2 + b * 4 + c
  h3 : y₃ = a * m^2 + b * m + c
  h4 : 2 * a * m + b = 0
  h5 : y₃ ≥ y₂
  h6 : y₂ > y₁

theorem parabola_vertex_position (p : Parabola) : p.m > 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_position_l1910_191097


namespace NUMINAMATH_CALUDE_ellipse_intersection_sum_of_squares_l1910_191035

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Represents an ellipse in 2D space -/
structure Ellipse where
  a : ℝ
  b : ℝ

def Ellipse.standard : Ellipse := { a := 2, b := 1 }

/-- Check if a point is on the ellipse -/
def Ellipse.contains (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Check if a line intersects the ellipse -/
def Line.intersectsEllipse (l : Line) (e : Ellipse) : Prop :=
  ∃ p : Point, e.contains p ∧ p.y = l.slope * p.x + l.intercept

/-- Calculate the distance squared from origin to a point -/
def Point.distanceSquared (p : Point) : ℝ :=
  p.x^2 + p.y^2

/-- Theorem statement -/
theorem ellipse_intersection_sum_of_squares :
  ∀ (l : Line),
    l.slope = 1/2 ∨ l.slope = -1/2 →
    l.intercept ≠ 0 →
    l.intersectsEllipse Ellipse.standard →
    ∃ (p1 p2 : Point),
      Ellipse.standard.contains p1 ∧
      Ellipse.standard.contains p2 ∧
      p1 ≠ p2 ∧
      p1.y = l.slope * p1.x + l.intercept ∧
      p2.y = l.slope * p2.x + l.intercept ∧
      p1.distanceSquared + p2.distanceSquared = 5 :=
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_sum_of_squares_l1910_191035


namespace NUMINAMATH_CALUDE_current_velocity_l1910_191031

/-- Velocity of current given rowing speed and round trip time -/
theorem current_velocity (rowing_speed : ℝ) (distance : ℝ) (total_time : ℝ) :
  rowing_speed = 5 →
  distance = 2.4 →
  total_time = 1 →
  ∃ v : ℝ,
    v > 0 ∧
    (distance / (rowing_speed - v) + distance / (rowing_speed + v) = total_time) ∧
    v = 1 := by
  sorry

end NUMINAMATH_CALUDE_current_velocity_l1910_191031


namespace NUMINAMATH_CALUDE_smallest_sticker_collection_l1910_191022

theorem smallest_sticker_collection : ∃ (S : ℕ), 
  S > 2 ∧
  S % 4 = 2 ∧
  S % 6 = 2 ∧
  S % 9 = 2 ∧
  S % 10 = 2 ∧
  (∀ (T : ℕ), T > 2 → T % 4 = 2 → T % 6 = 2 → T % 9 = 2 → T % 10 = 2 → S ≤ T) ∧
  S = 182 := by
  sorry

end NUMINAMATH_CALUDE_smallest_sticker_collection_l1910_191022


namespace NUMINAMATH_CALUDE_shaded_area_of_square_shaded_percentage_l1910_191052

/-- The shaded area of a square with side length 6 units -/
theorem shaded_area_of_square (side_length : ℝ) (shaded_square : ℝ) (shaded_region : ℝ) (shaded_strip : ℝ) : 
  side_length = 6 →
  shaded_square = 2^2 →
  shaded_region = 5^2 - 3^2 →
  shaded_strip = 6 * 1 →
  shaded_square + shaded_region + shaded_strip = 26 := by
sorry

/-- The percentage of the square that is shaded -/
theorem shaded_percentage (total_area : ℝ) (shaded_area : ℝ) :
  total_area = 6^2 →
  shaded_area = 26 →
  (shaded_area / total_area) * 100 = 72.22 := by
sorry

end NUMINAMATH_CALUDE_shaded_area_of_square_shaded_percentage_l1910_191052


namespace NUMINAMATH_CALUDE_max_value_theorem_l1910_191055

theorem max_value_theorem (a b : ℝ) 
  (h : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |a * x + b| ≤ 1) :
  ∃ M : ℝ, M = 80 ∧ 
    (∀ a' b' : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |a' * x + b'| ≤ 1) → 
      |20 * a' + 14 * b'| + |20 * a' - 14 * b'| ≤ M) ∧
    (∃ a' b' : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |a' * x + b'| ≤ 1) ∧ 
      |20 * a' + 14 * b'| + |20 * a' - 14 * b'| = M) :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1910_191055


namespace NUMINAMATH_CALUDE_team_selection_count_l1910_191010

/-- The number of ways to select a team of 4 boys from 10 boys and 4 girls from 12 girls -/
def select_team : ℕ :=
  Nat.choose 10 4 * Nat.choose 12 4

/-- Theorem stating that the number of ways to select the team is 103950 -/
theorem team_selection_count : select_team = 103950 := by
  sorry

end NUMINAMATH_CALUDE_team_selection_count_l1910_191010


namespace NUMINAMATH_CALUDE_place_value_ratio_l1910_191089

def number : ℚ := 86304.2957

theorem place_value_ratio :
  let thousands_place_value : ℚ := 1000
  let tenths_place_value : ℚ := 0.1
  (thousands_place_value / tenths_place_value : ℚ) = 10000 := by
  sorry

end NUMINAMATH_CALUDE_place_value_ratio_l1910_191089


namespace NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l1910_191016

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_12th_term
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_3 : a 3 = 9)
  (h_6 : a 6 = 15) :
  a 12 = 27 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l1910_191016


namespace NUMINAMATH_CALUDE_solve_for_x_l1910_191043

theorem solve_for_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 14) : x = 11 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l1910_191043


namespace NUMINAMATH_CALUDE_initial_pens_l1910_191086

theorem initial_pens (initial : ℕ) (mike_gives : ℕ) (cindy_doubles : ℕ → ℕ) (sharon_takes : ℕ) (final : ℕ) : 
  mike_gives = 22 →
  cindy_doubles = (· * 2) →
  sharon_takes = 19 →
  final = 75 →
  cindy_doubles (initial + mike_gives) - sharon_takes = final →
  initial = 25 :=
by sorry

end NUMINAMATH_CALUDE_initial_pens_l1910_191086


namespace NUMINAMATH_CALUDE_income_p_is_3000_l1910_191068

/-- The monthly income of three people given their pairwise averages -/
def monthly_income (avg_pq avg_qr avg_pr : ℚ) : ℚ × ℚ × ℚ :=
  let p := 2 * (avg_pq + avg_pr - avg_qr)
  let q := 2 * (avg_pq + avg_qr - avg_pr)
  let r := 2 * (avg_qr + avg_pr - avg_pq)
  (p, q, r)

theorem income_p_is_3000 (avg_pq avg_qr avg_pr : ℚ) :
  avg_pq = 2050 → avg_qr = 5250 → avg_pr = 6200 →
  (monthly_income avg_pq avg_qr avg_pr).1 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_income_p_is_3000_l1910_191068
