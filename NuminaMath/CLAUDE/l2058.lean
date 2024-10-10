import Mathlib

namespace odd_sum_even_product_l2058_205828

theorem odd_sum_even_product (a b : ℤ) : 
  Odd (a + b) → Even (a * b) := by sorry

end odd_sum_even_product_l2058_205828


namespace pie_chart_percentage_central_angle_relation_l2058_205896

/-- Represents a part of a pie chart -/
structure PieChartPart where
  percentage : ℝ
  centralAngle : ℝ

/-- Theorem stating the relationship between percentage and central angle in a pie chart -/
theorem pie_chart_percentage_central_angle_relation (part : PieChartPart) :
  part.percentage = part.centralAngle / 360 := by
  sorry

end pie_chart_percentage_central_angle_relation_l2058_205896


namespace orchids_cut_count_l2058_205824

/-- Represents the number of flowers in the vase -/
structure FlowerCount where
  roses : ℕ
  orchids : ℕ

/-- Represents the ratio of cut flowers -/
structure CutRatio where
  roses : ℕ
  orchids : ℕ

def initial_count : FlowerCount := { roses := 16, orchids := 3 }
def final_count : FlowerCount := { roses := 31, orchids := 7 }
def cut_ratio : CutRatio := { roses := 5, orchids := 3 }

theorem orchids_cut_count (initial : FlowerCount) (final : FlowerCount) (ratio : CutRatio) :
  final.orchids - initial.orchids = 4 :=
sorry

end orchids_cut_count_l2058_205824


namespace sqrt_two_irrational_l2058_205857

theorem sqrt_two_irrational : Irrational (Real.sqrt 2) := by
  sorry

end sqrt_two_irrational_l2058_205857


namespace parabola_tangent_line_l2058_205852

theorem parabola_tangent_line (b c : ℝ) : 
  (∃ x y : ℝ, y = -2 * x^2 + b * x + c ∧ 
              y = x - 3 ∧ 
              x = 2 ∧ 
              y = -1) → 
  b + c = -2 :=
sorry


end parabola_tangent_line_l2058_205852


namespace johns_cloth_cost_l2058_205873

/-- The total cost of cloth purchased by John -/
def total_cost (length : ℝ) (price_per_metre : ℝ) : ℝ :=
  length * price_per_metre

/-- Theorem stating the total cost of John's cloth purchase -/
theorem johns_cloth_cost :
  let length : ℝ := 9.25
  let price_per_metre : ℝ := 46
  total_cost length price_per_metre = 425.50 := by
  sorry

end johns_cloth_cost_l2058_205873


namespace train_length_proof_l2058_205803

def train_problem (speed1 speed2 shorter_length clearing_time : ℝ) : Prop :=
  let relative_speed := (speed1 + speed2) * 1000 / 3600
  let total_distance := relative_speed * clearing_time
  let longer_length := total_distance - shorter_length
  longer_length = 164.9771230827526

theorem train_length_proof :
  train_problem 80 55 121 7.626056582140095 := by sorry

end train_length_proof_l2058_205803


namespace tax_rate_on_other_items_l2058_205850

-- Define the total amount spent (excluding taxes)
def total_amount : ℝ := 100

-- Define the percentages spent on each category
def clothing_percent : ℝ := 0.5
def food_percent : ℝ := 0.25
def other_percent : ℝ := 0.25

-- Define the tax rates
def clothing_tax_rate : ℝ := 0.1
def food_tax_rate : ℝ := 0
def total_tax_rate : ℝ := 0.1

-- Define the amounts spent on each category
def clothing_amount : ℝ := total_amount * clothing_percent
def food_amount : ℝ := total_amount * food_percent
def other_amount : ℝ := total_amount * other_percent

-- Define the tax paid on clothing
def clothing_tax : ℝ := clothing_amount * clothing_tax_rate

-- Define the total tax paid
def total_tax : ℝ := total_amount * total_tax_rate

-- Define the tax paid on other items
def other_tax : ℝ := total_tax - clothing_tax

-- Theorem to prove
theorem tax_rate_on_other_items :
  other_tax / other_amount = 0.2 := by sorry

end tax_rate_on_other_items_l2058_205850


namespace ruby_pizza_order_cost_l2058_205838

/-- Represents the cost of a pizza order --/
structure PizzaOrder where
  basePizzaCost : ℝ
  toppingCost : ℝ
  tipAmount : ℝ
  pepperoniToppings : ℕ
  sausageToppings : ℕ
  blackOliveMushroomToppings : ℕ
  numberOfPizzas : ℕ

/-- Calculates the total cost of a pizza order --/
def totalCost (order : PizzaOrder) : ℝ :=
  order.basePizzaCost * order.numberOfPizzas +
  order.toppingCost * (order.pepperoniToppings + order.sausageToppings + order.blackOliveMushroomToppings) +
  order.tipAmount

/-- Theorem stating that the total cost of Ruby's pizza order is $39.00 --/
theorem ruby_pizza_order_cost :
  let order : PizzaOrder := {
    basePizzaCost := 10,
    toppingCost := 1,
    tipAmount := 5,
    pepperoniToppings := 1,
    sausageToppings := 1,
    blackOliveMushroomToppings := 2,
    numberOfPizzas := 3
  }
  totalCost order = 39 := by
  sorry

end ruby_pizza_order_cost_l2058_205838


namespace xyz_value_l2058_205854

theorem xyz_value (a b c x y z : ℂ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (eq1 : a = (b + c) / (x - 3))
  (eq2 : b = (a + c) / (y - 3))
  (eq3 : c = (a + b) / (z - 3))
  (eq4 : x * y + x * z + y * z = 9)
  (eq5 : x + y + z = 6) :
  x * y * z = 14 := by
sorry

end xyz_value_l2058_205854


namespace ratio_equality_l2058_205835

theorem ratio_equality (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (h_eq : y / (x - z) = (x + y) / z ∧ (x + y) / z = x / y) : 
  x / y = 2 := by
sorry

end ratio_equality_l2058_205835


namespace square_even_implies_even_l2058_205866

theorem square_even_implies_even (a : ℤ) (h : Even (a ^ 2)) : Even a := by
  sorry

end square_even_implies_even_l2058_205866


namespace salary_increase_l2058_205877

theorem salary_increase (last_year_salary : ℝ) (last_year_savings_rate : ℝ) 
  (this_year_savings_rate : ℝ) (salary_increase_rate : ℝ) :
  last_year_savings_rate = 0.06 →
  this_year_savings_rate = 0.05 →
  this_year_savings_rate * (last_year_salary * (1 + salary_increase_rate)) = 
    last_year_savings_rate * last_year_salary →
  salary_increase_rate = 0.2 := by
  sorry

#check salary_increase

end salary_increase_l2058_205877


namespace delta_donuts_calculation_l2058_205839

def total_donuts : ℕ := 40
def gamma_donuts : ℕ := 8
def beta_donuts : ℕ := 3 * gamma_donuts

theorem delta_donuts_calculation :
  total_donuts - (beta_donuts + gamma_donuts) = 8 :=
by sorry

end delta_donuts_calculation_l2058_205839


namespace incorrect_inequality_l2058_205827

-- Define the conditions
variable (a b : ℝ)
variable (h1 : b < a)
variable (h2 : a < 0)

-- Define the theorem
theorem incorrect_inequality :
  ¬((1/2:ℝ)^b < (1/2:ℝ)^a) :=
by sorry

end incorrect_inequality_l2058_205827


namespace smallest_polygon_with_lighting_property_l2058_205871

/-- A polygon with n sides -/
structure Polygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- A point in the plane -/
def Point := ℝ × ℝ

/-- Predicate to check if a point is inside a polygon -/
def isInside (p : Point) (poly : Polygon n) : Prop := sorry

/-- Predicate to check if a point on a side of the polygon is lightened by a bulb -/
def isLightened (p : Point) (side : Fin n) (poly : Polygon n) (bulb : Point) : Prop := sorry

/-- Predicate to check if a polygon satisfies the lighting property -/
def hasLightingProperty (poly : Polygon n) : Prop :=
  ∃ bulb : Point, isInside bulb poly ∧
    ∀ side : Fin n, ∃ p : Point, ¬isLightened p side poly bulb

/-- Predicate to check if two bulbs light up the whole perimeter -/
def lightsWholePerimeter (poly : Polygon n) (bulb1 bulb2 : Point) : Prop :=
  ∀ side : Fin n, ∀ p : Point, isLightened p side poly bulb1 ∨ isLightened p side poly bulb2

theorem smallest_polygon_with_lighting_property :
  (∀ n < 6, ¬∃ poly : Polygon n, hasLightingProperty poly) ∧
  (∃ poly : Polygon 6, hasLightingProperty poly) ∧
  (∀ poly : Polygon 6, ∃ bulb1 bulb2 : Point, lightsWholePerimeter poly bulb1 bulb2) := by
  sorry

end smallest_polygon_with_lighting_property_l2058_205871


namespace target_annual_revenue_l2058_205800

/-- Calculates the target annual revenue for a shoe company given their current monthly sales and required monthly increase. -/
theorem target_annual_revenue
  (current_monthly_sales : ℕ)
  (required_monthly_increase : ℕ)
  (months_per_year : ℕ)
  (h1 : current_monthly_sales = 4000)
  (h2 : required_monthly_increase = 1000)
  (h3 : months_per_year = 12) :
  (current_monthly_sales + required_monthly_increase) * months_per_year = 60000 :=
by
  sorry

#check target_annual_revenue

end target_annual_revenue_l2058_205800


namespace intersection_of_sets_l2058_205830

theorem intersection_of_sets : 
  let A : Set ℤ := {0, 1, 2, 4}
  let B : Set ℤ := {-1, 0, 1, 3}
  A ∩ B = {0, 1} := by
sorry

end intersection_of_sets_l2058_205830


namespace fraction_to_decimal_l2058_205801

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by
  sorry

end fraction_to_decimal_l2058_205801


namespace max_distance_between_functions_l2058_205862

open Real

theorem max_distance_between_functions :
  let f (x : ℝ) := 2 * (sin (π / 4 + x))^2
  let g (x : ℝ) := Real.sqrt 3 * cos (2 * x)
  ∀ a : ℝ, |f a - g a| ≤ 3 ∧ ∃ b : ℝ, |f b - g b| = 3 :=
by sorry

end max_distance_between_functions_l2058_205862


namespace gcd_equality_l2058_205891

theorem gcd_equality (a b c : ℕ+) (h : Nat.gcd (a^2 - 1) (Nat.gcd (b^2 - 1) (c^2 - 1)) = 1) :
  Nat.gcd (a*b + c) (Nat.gcd (b*c + a) (c*a + b)) = Nat.gcd a (Nat.gcd b c) :=
sorry

end gcd_equality_l2058_205891


namespace constant_for_max_n_l2058_205876

theorem constant_for_max_n (c : ℝ) : 
  (∀ n : ℤ, c * n^2 ≤ 3600) ∧ 
  (∃ n : ℤ, n > 5 ∧ c * n^2 > 3600) ∧
  c * 5^2 ≤ 3600 →
  c = 144 := by sorry

end constant_for_max_n_l2058_205876


namespace max_value_of_product_l2058_205892

theorem max_value_of_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 2) :
  a^2 * b^3 * c^4 ≤ 143327232 / 386989855 :=
sorry

end max_value_of_product_l2058_205892


namespace quadratic_solution_sum_l2058_205859

theorem quadratic_solution_sum (a b : ℚ) : 
  (∃ x : ℂ, x = a + b * I ∧ 5 * x^2 - 2 * x + 17 = 0) →
  a + b^2 = 89/25 := by
sorry

end quadratic_solution_sum_l2058_205859


namespace prob_odd_sum_is_two_thirds_l2058_205819

/-- A type representing the cards labeled 0, 1, and 2 -/
inductive Card : Type
  | zero : Card
  | one : Card
  | two : Card

/-- A function to convert a Card to its numerical value -/
def cardValue : Card → ℕ
  | Card.zero => 0
  | Card.one => 1
  | Card.two => 2

/-- A predicate to check if the sum of two cards is odd -/
def isSumOdd (c1 c2 : Card) : Prop :=
  Odd (cardValue c1 + cardValue c2)

/-- The set of all possible card combinations -/
def allCombinations : Finset (Card × Card) :=
  sorry

/-- The set of card combinations with odd sum -/
def oddSumCombinations : Finset (Card × Card) :=
  sorry

/-- Theorem stating the probability of drawing two cards with odd sum is 2/3 -/
theorem prob_odd_sum_is_two_thirds :
    (Finset.card oddSumCombinations : ℚ) / (Finset.card allCombinations : ℚ) = 2 / 3 :=
  sorry

end prob_odd_sum_is_two_thirds_l2058_205819


namespace union_of_A_and_B_l2058_205816

def set_A : Set ℝ := {x | x^2 - x - 2 ≤ 0}
def set_B : Set ℝ := {x | x - 1 < 0}

theorem union_of_A_and_B : set_A ∪ set_B = {x : ℝ | x ≤ 2} := by sorry

end union_of_A_and_B_l2058_205816


namespace f_properties_l2058_205883

noncomputable def f : ℝ → ℝ := fun x => if x ≤ 0 then x^2 + 2*x else x^2 - 2*x

theorem f_properties :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x : ℝ, x > 0 → f x = x^2 - 2*x) ∧
  (StrictMonoOn f (Set.Ioo (-1) 0)) ∧
  (StrictMonoOn f (Set.Ioi 1)) ∧
  (StrictAntiOn f (Set.Iic (-1))) ∧
  (StrictAntiOn f (Set.Ioo 0 1)) ∧
  (Set.range f = Set.Ici (-1)) := by
sorry

end f_properties_l2058_205883


namespace emails_left_in_inbox_l2058_205821

theorem emails_left_in_inbox (initial_emails : ℕ) : 
  initial_emails = 400 → 
  (initial_emails / 2 - (initial_emails / 2 * 40 / 100) : ℕ) = 120 := by
  sorry

end emails_left_in_inbox_l2058_205821


namespace competition_max_robot_weight_l2058_205869

/-- The weight of the standard robot in pounds -/
def standard_robot_weight : ℝ := 100

/-- The weight of the battery in pounds -/
def battery_weight : ℝ := 20

/-- The minimum additional weight above the standard robot in pounds -/
def min_additional_weight : ℝ := 5

/-- The maximum weight multiplier -/
def max_weight_multiplier : ℝ := 2

/-- The maximum weight of a robot in the competition, including the battery -/
def max_robot_weight : ℝ := 250

theorem competition_max_robot_weight :
  max_robot_weight = 
    max_weight_multiplier * (standard_robot_weight + min_additional_weight + battery_weight) :=
by sorry

end competition_max_robot_weight_l2058_205869


namespace min_distance_sum_l2058_205860

-- Define the circles
def C₁ (x y : ℝ) : Prop := (x - 2)^2 + (y - 3)^2 = 1
def C₂ (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 9

-- Define the theorem
theorem min_distance_sum (M N P : ℝ × ℝ) :
  C₁ M.1 M.2 →
  C₂ N.1 N.2 →
  P.2 = 0 →
  ∃ (M' N' P' : ℝ × ℝ),
    C₁ M'.1 M'.2 ∧
    C₂ N'.1 N'.2 ∧
    P'.2 = 0 ∧
    Real.sqrt ((M.1 - P.1)^2 + (M.2 - P.2)^2) +
    Real.sqrt ((N.1 - P.1)^2 + (N.2 - P.2)^2) ≥
    Real.sqrt ((M'.1 - P'.1)^2 + (M'.2 - P'.2)^2) +
    Real.sqrt ((N'.1 - P'.1)^2 + (N'.2 - P'.2)^2) ∧
    Real.sqrt ((M'.1 - P'.1)^2 + (M'.2 - P'.2)^2) +
    Real.sqrt ((N'.1 - P'.1)^2 + (N'.2 - P'.2)^2) =
    5 * Real.sqrt 2 - 4 :=
by sorry

end min_distance_sum_l2058_205860


namespace quadratic_factorization_l2058_205834

theorem quadratic_factorization (x : ℝ) : 16 * x^2 - 56 * x + 49 = (4 * x - 7)^2 := by
  sorry

end quadratic_factorization_l2058_205834


namespace inverse_f_at_4_l2058_205868

noncomputable def f (x : ℝ) : ℝ := Real.sqrt x

def HasInverse (f : ℝ → ℝ) (g : ℝ → ℝ) : Prop :=
  ∀ x, f (g x) = x ∧ g (f x) = x

theorem inverse_f_at_4 (f_inv : ℝ → ℝ) (h : HasInverse f f_inv) : f_inv 4 = 16 := by
  sorry

end inverse_f_at_4_l2058_205868


namespace expected_value_of_twelve_sided_die_l2058_205815

/-- A fair 12-sided die with faces numbered from 1 to 12 -/
def twelve_sided_die : Finset ℕ := Finset.range 12

/-- The expected value of rolling the 12-sided die -/
def expected_value : ℚ := (Finset.sum twelve_sided_die (fun i => i + 1)) / 12

/-- Theorem: The expected value of rolling a fair 12-sided die with faces numbered from 1 to 12 is 6.5 -/
theorem expected_value_of_twelve_sided_die : expected_value = 13/2 := by
  sorry

end expected_value_of_twelve_sided_die_l2058_205815


namespace sandy_earnings_l2058_205867

/-- Calculates Sandy's earnings for a given day -/
def daily_earnings (hours : ℝ) (hourly_rate : ℝ) (with_best_friend : Bool) (longer_than_12_hours : Bool) : ℝ :=
  let base_wage := hours * hourly_rate
  let commission := if with_best_friend then base_wage * 0.1 else 0
  let bonus := if longer_than_12_hours then base_wage * 0.05 else 0
  let total_before_tax := base_wage + commission + bonus
  total_before_tax * 0.93  -- Apply 7% tax deduction

/-- Sandy's total earnings for Friday, Saturday, and Sunday -/
def total_earnings : ℝ :=
  let hourly_rate := 15
  let friday := daily_earnings 10 hourly_rate true false
  let saturday := daily_earnings 6 hourly_rate false false
  let sunday := daily_earnings 14 hourly_rate false true
  friday + saturday + sunday

/-- Theorem stating Sandy's total earnings -/
theorem sandy_earnings : total_earnings = 442.215 := by
  sorry

end sandy_earnings_l2058_205867


namespace alan_cd_purchase_cost_l2058_205849

/-- The price of a CD by "AVN" in dollars -/
def avnPrice : ℝ := 12

/-- The price of a CD by "The Dark" in dollars -/
def darkPrice : ℝ := 2 * avnPrice

/-- The cost of CDs by "The Dark" and "AVN" in dollars -/
def mainCost : ℝ := 2 * darkPrice + avnPrice

/-- The cost of 90s music CDs in dollars -/
def mixCost : ℝ := 0.4 * mainCost

/-- The total cost of Alan's purchase in dollars -/
def totalCost : ℝ := mainCost + mixCost

theorem alan_cd_purchase_cost :
  totalCost = 84 := by sorry

end alan_cd_purchase_cost_l2058_205849


namespace intersection_properties_l2058_205845

/-- A parabola intersecting a line and the x-axis -/
structure Intersection where
  a : ℝ
  b : ℝ
  c : ℝ
  k : ℝ
  haNonZero : a ≠ 0
  hIntersectLine : ∀ x, a * x^2 + b * x + c = k * x + 4 → (x = 1 ∨ x = 4)
  hIntersectXAxis : ∀ x, a * x^2 + b * x + c = 0 → (x = 0 ∨ ∃ y, y ≠ 0 ∧ a * y^2 + b * y + c = 0)

/-- The main theorem about the intersection -/
theorem intersection_properties (i : Intersection) :
  (∀ x, k * x + 4 = x + 4) ∧
  (∀ x, a * x^2 + b * x + c = -x^2 + 6 * x) ∧
  (∃ x, x > 0 ∧ -x^2 + 6 * x = 4 ∧
    2 * (1/2 * 1 * 5 + 1/2 * 3 * 3) = 1/2 * 6 * 4) := by
  sorry

end intersection_properties_l2058_205845


namespace third_root_of_cubic_l2058_205855

theorem third_root_of_cubic (a b : ℚ) : 
  (∀ x : ℚ, a * x^3 + (a + 3*b) * x^2 + (b - 4*a) * x + (10 - a) = 0 ↔ x = -1 ∨ x = 4 ∨ x = 11/5) :=
by sorry

end third_root_of_cubic_l2058_205855


namespace two_digit_divisible_by_72_l2058_205856

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def six_digit_number (x y : ℕ) : ℕ := 640000 + x * 1000 + 720 + y

theorem two_digit_divisible_by_72 :
  ∀ x y : ℕ, is_two_digit (10 * x + y) →
  (six_digit_number x y ∣ 72) →
  (x = 8 ∧ y = 0) ∨ (x = 9 ∧ y = 8) :=
sorry

end two_digit_divisible_by_72_l2058_205856


namespace line_circle_intersection_condition_l2058_205878

/-- The line y = x + b intersects the circle x^2 + y^2 = 1 -/
def line_intersects_circle (b : ℝ) : Prop :=
  ∃ x y : ℝ, y = x + b ∧ x^2 + y^2 = 1

/-- The condition 0 < b < 1 is necessary but not sufficient for the intersection -/
theorem line_circle_intersection_condition (b : ℝ) :
  line_intersects_circle b → 0 < b ∧ b < 1 ∧
  ¬(∀ b : ℝ, 0 < b ∧ b < 1 → line_intersects_circle b) :=
by sorry

end line_circle_intersection_condition_l2058_205878


namespace solve_y_l2058_205858

theorem solve_y (x y : ℝ) (h1 : x - y = 20) (h2 : x + y = 14) : y = -3 := by
  sorry

end solve_y_l2058_205858


namespace inverse_sum_mod_17_l2058_205861

theorem inverse_sum_mod_17 : 
  (∃ x y : ℤ, (7 * x) % 17 = 1 ∧ (7 * y) % 17 = x % 17 ∧ (x + y) % 17 = 13) := by
  sorry

end inverse_sum_mod_17_l2058_205861


namespace multiple_of_nine_three_odd_l2058_205890

theorem multiple_of_nine_three_odd (n : ℕ) :
  (∀ m : ℕ, 9 ∣ m → 3 ∣ m) →
  Odd n →
  9 ∣ n →
  3 ∣ n :=
by
  sorry

end multiple_of_nine_three_odd_l2058_205890


namespace fraction_equality_l2058_205804

theorem fraction_equality : (1/4 - 1/6) / (1/3 - 1/4) = 1 := by sorry

end fraction_equality_l2058_205804


namespace max_sphere_in_intersecting_cones_l2058_205822

/-- Represents a right circular cone --/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents the configuration of two intersecting cones --/
structure IntersectingCones where
  cone1 : Cone
  cone2 : Cone
  intersectionDistance : ℝ

/-- Calculates the maximum squared radius of a sphere fitting inside the intersecting cones --/
def maxSquaredSphereRadius (ic : IntersectingCones) : ℝ := sorry

/-- Theorem statement --/
theorem max_sphere_in_intersecting_cones :
  let ic : IntersectingCones := {
    cone1 := { baseRadius := 4, height := 10 },
    cone2 := { baseRadius := 4, height := 10 },
    intersectionDistance := 4
  }
  maxSquaredSphereRadius ic = 144 := by sorry

end max_sphere_in_intersecting_cones_l2058_205822


namespace negative_cube_inequality_l2058_205846

theorem negative_cube_inequality (a : ℝ) (h : a < 0) : a^3 ≠ (-a)^3 := by
  sorry

end negative_cube_inequality_l2058_205846


namespace largest_term_binomial_expansion_l2058_205832

theorem largest_term_binomial_expansion (n : ℕ) (x : ℝ) (h : n = 500 ∧ x = 0.1) :
  ∃ k : ℕ, k = 45 ∧
  ∀ j : ℕ, j ≤ n → (n.choose k) * x^k ≥ (n.choose j) * x^j :=
sorry

end largest_term_binomial_expansion_l2058_205832


namespace multiply_subtract_distribute_computation_result_l2058_205853

theorem multiply_subtract_distribute (a b c : ℕ) :
  a * c - b * c = (a - b) * c :=
by sorry

theorem computation_result : 72 * 1313 - 32 * 1313 = 52520 := by
  -- Proof goes here
  sorry

end multiply_subtract_distribute_computation_result_l2058_205853


namespace one_two_five_th_number_l2058_205813

def digit_sum (n : ℕ) : ℕ := sorry

def nth_number_with_digit_sum_5 (n : ℕ) : ℕ := sorry

theorem one_two_five_th_number : nth_number_with_digit_sum_5 125 = 41000 := by sorry

end one_two_five_th_number_l2058_205813


namespace carrot_count_l2058_205882

theorem carrot_count (olivia_carrots : ℕ) (mom_carrots : ℕ) : 
  olivia_carrots = 20 → mom_carrots = 14 → olivia_carrots + mom_carrots = 34 := by
sorry

end carrot_count_l2058_205882


namespace stadium_area_calculation_l2058_205817

/-- Calculates the total surface area of a rectangular stadium in square feet,
    given its dimensions in yards. -/
def stadium_surface_area (length_yd width_yd height_yd : ℕ) : ℕ :=
  let length := length_yd * 3
  let width := width_yd * 3
  let height := height_yd * 3
  2 * (length * width + length * height + width * height)

/-- Theorem stating that the surface area of a stadium with given dimensions is 110,968 sq ft. -/
theorem stadium_area_calculation :
  stadium_surface_area 62 48 30 = 110968 := by
  sorry

#eval stadium_surface_area 62 48 30

end stadium_area_calculation_l2058_205817


namespace initial_water_amount_l2058_205875

/-- Given a bucket of water, prove that the initial amount is 0.8 gallons when 0.2 gallons are poured out and 0.6 gallons remain. -/
theorem initial_water_amount (poured_out : ℝ) (remaining : ℝ) : poured_out = 0.2 → remaining = 0.6 → poured_out + remaining = 0.8 := by
  sorry

end initial_water_amount_l2058_205875


namespace remainder_theorem_l2058_205806

theorem remainder_theorem (P D Q R D' Q' R' : ℕ) 
  (h1 : P = Q * D + R) 
  (h2 : Q = 2 * D' * Q' + R') : 
  P % (2 * D * D') = D * R' + R := by
sorry

end remainder_theorem_l2058_205806


namespace total_study_hours_is_three_l2058_205863

-- Define the time spent on each subject in minutes
def science_time : ℕ := 60
def math_time : ℕ := 80
def literature_time : ℕ := 40

-- Define the total study time in minutes
def total_study_time : ℕ := science_time + math_time + literature_time

-- Define the conversion factor from minutes to hours
def minutes_per_hour : ℕ := 60

-- Theorem to prove
theorem total_study_hours_is_three :
  (total_study_time : ℚ) / minutes_per_hour = 3 := by sorry

end total_study_hours_is_three_l2058_205863


namespace part_one_part_two_l2058_205898

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x | 2*x - 4 ≥ x - 2}

-- Define set C with parameter a
def C (a : ℝ) : Set ℝ := {x | 2*x + a > 0}

-- Theorem for part (1)
theorem part_one :
  B = {x : ℝ | x ≥ 2} ∧
  (A ∩ B)ᶜ = {x : ℝ | x < 2 ∨ x ≥ 3} :=
sorry

-- Theorem for part (2)
theorem part_two :
  {a : ℝ | ∀ x, x ∈ B → x ∈ C a} = {a : ℝ | a > -4} :=
sorry

end part_one_part_two_l2058_205898


namespace tree_height_proof_l2058_205847

/-- Proves that a tree with a current height of 180 inches, which is 50% taller than its original height, had an original height of 10 feet. -/
theorem tree_height_proof (current_height : ℝ) (growth_factor : ℝ) (inches_per_foot : ℝ) :
  current_height = 180 ∧
  growth_factor = 1.5 ∧
  inches_per_foot = 12 →
  current_height / growth_factor / inches_per_foot = 10 := by
  sorry

end tree_height_proof_l2058_205847


namespace prob_three_spades_l2058_205812

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (num_suits : Nat)
  (cards_per_suit : Nat)
  (h_total : total_cards = num_suits * cards_per_suit)

/-- A standard deck of 52 cards with 4 suits and 13 cards per suit -/
def standard_deck : Deck :=
  { total_cards := 52,
    num_suits := 4,
    cards_per_suit := 13,
    h_total := rfl }

/-- The probability of drawing three spades in a row from a standard deck -/
theorem prob_three_spades (d : Deck) (h : d = standard_deck) :
  (d.cards_per_suit : ℚ) / d.total_cards *
  (d.cards_per_suit - 1) / (d.total_cards - 1) *
  (d.cards_per_suit - 2) / (d.total_cards - 2) = 33 / 2550 := by
  sorry

end prob_three_spades_l2058_205812


namespace sine_graph_transformation_l2058_205888

/-- Given two sine functions with different periods, prove that the graph of one
    can be obtained by transforming the graph of the other. -/
theorem sine_graph_transformation (x : ℝ) :
  let f (x : ℝ) := Real.sin (x + π/4)
  let g (x : ℝ) := Real.sin (3*x + π/4)
  ∃ (h : ℝ → ℝ), (∀ x, g x = f (h x)) ∧ (∀ x, h x = x/3) := by
  sorry

end sine_graph_transformation_l2058_205888


namespace power_two_greater_than_square_l2058_205805

theorem power_two_greater_than_square (n : ℕ) (h : n > 5) : 2^n > n^2 := by
  sorry

end power_two_greater_than_square_l2058_205805


namespace smallest_prime_after_six_nonprimes_l2058_205864

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

-- Define a function to check if a number is the start of six consecutive non-primes
def isSixConsecutiveNonPrimes (n : ℕ) : Prop :=
  ∀ k : ℕ, k ≥ n → k < n + 6 → ¬(isPrime k)

-- Theorem statement
theorem smallest_prime_after_six_nonprimes :
  ∃ p : ℕ, isPrime p ∧ 
    (∃ n : ℕ, isSixConsecutiveNonPrimes n ∧ p = n + 6) ∧
    (∀ q : ℕ, q < p → ¬(∃ m : ℕ, isSixConsecutiveNonPrimes m ∧ isPrime (m + 6) ∧ q = m + 6)) :=
  sorry

end smallest_prime_after_six_nonprimes_l2058_205864


namespace zero_point_implies_a_range_l2058_205844

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + 2

-- Define the theorem
theorem zero_point_implies_a_range (a : ℝ) : 
  (∃ x ∈ Set.Icc (-2) 1, g a x = 0) → 
  (a < 2) → 
  a ∈ Set.Icc (-3/2) 2 := by
sorry

end zero_point_implies_a_range_l2058_205844


namespace chipmunk_increase_l2058_205848

/-- Proves that the number of chipmunks increased by 50 given the initial counts, doubling of beavers, and total animal count. -/
theorem chipmunk_increase (initial_beavers initial_chipmunks total_animals : ℕ) 
  (h1 : initial_beavers = 20)
  (h2 : initial_chipmunks = 40)
  (h3 : total_animals = 130) :
  (total_animals - 2 * initial_beavers) - initial_chipmunks = 50 := by
  sorry

#check chipmunk_increase

end chipmunk_increase_l2058_205848


namespace largest_cosine_in_geometric_triangle_l2058_205823

/-- Given a triangle ABC where its sides form a geometric sequence with common ratio √2,
    the largest cosine value of its angles is -√2/4 -/
theorem largest_cosine_in_geometric_triangle :
  ∀ (a b c : ℝ),
  a > 0 →
  b = a * Real.sqrt 2 →
  c = b * Real.sqrt 2 →
  let cosA := (b^2 + c^2 - a^2) / (2*b*c)
  let cosB := (a^2 + c^2 - b^2) / (2*a*c)
  let cosC := (a^2 + b^2 - c^2) / (2*a*b)
  max cosA (max cosB cosC) = -(Real.sqrt 2) / 4 :=
by sorry

end largest_cosine_in_geometric_triangle_l2058_205823


namespace water_displaced_squared_value_l2058_205825

/-- The square of the volume of water displaced by a fully submerged cube in a cylindrical barrel -/
def water_displaced_squared (cube_side : ℝ) (barrel_radius : ℝ) (barrel_height : ℝ) : ℝ :=
  (cube_side ^ 3) ^ 2

/-- Theorem stating that the square of the volume of water displaced by a fully submerged cube
    with side length 7 feet in a cylindrical barrel with radius 5 feet and height 15 feet is 117649 cubic feet -/
theorem water_displaced_squared_value :
  water_displaced_squared 7 5 15 = 117649 := by
  sorry

end water_displaced_squared_value_l2058_205825


namespace people_who_got_off_l2058_205872

theorem people_who_got_off (initial_people : ℕ) (people_left : ℕ) : 
  initial_people = 48 → people_left = 31 → initial_people - people_left = 17 := by
  sorry

end people_who_got_off_l2058_205872


namespace max_reciprocal_sum_l2058_205810

theorem max_reciprocal_sum (x y z : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0)
  (hsum1 : x/y + y/z + z/x = 3) (hsum2 : x + y + z = 6) :
  ∃ (M : ℝ), ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 →
    a/b + b/c + c/a = 3 → a + b + c = 6 →
    y/x + z/y + x/z ≤ M ∧ M = 3 := by
  sorry

end max_reciprocal_sum_l2058_205810


namespace remainder_theorem_l2058_205818

theorem remainder_theorem (n : ℤ) (h : ∃ k : ℤ, n = 50 * k - 1) :
  (n^2 + 2*n + 3) % 50 = 2 := by
sorry

end remainder_theorem_l2058_205818


namespace halfway_fraction_l2058_205870

theorem halfway_fraction : 
  let a := (1 : ℚ) / 2
  let b := (3 : ℚ) / 4
  (a + b) / 2 = (5 : ℚ) / 8 := by sorry

end halfway_fraction_l2058_205870


namespace tenth_term_is_one_over_120_l2058_205837

def a (n : ℕ+) : ℚ := 1 / (n * (n + 2))

theorem tenth_term_is_one_over_120 : a 10 = 1 / 120 := by
  sorry

end tenth_term_is_one_over_120_l2058_205837


namespace other_number_value_l2058_205826

theorem other_number_value (x y : ℝ) : 
  y = 125 * 1.1 →
  x = y * 0.9 →
  x = 123.75 →
  y = 137.5 := by
sorry

end other_number_value_l2058_205826


namespace circle_area_not_covered_l2058_205895

theorem circle_area_not_covered (outer_diameter inner_diameter : ℝ) 
  (h1 : outer_diameter = 30) 
  (h2 : inner_diameter = 24) : 
  (outer_diameter^2 - inner_diameter^2) / outer_diameter^2 = 9 / 25 := by
  sorry

end circle_area_not_covered_l2058_205895


namespace gretchen_objects_l2058_205808

/-- The number of objects Gretchen can carry per trip -/
def objects_per_trip : ℕ := 3

/-- The number of trips Gretchen took -/
def number_of_trips : ℕ := 6

/-- The total number of objects Gretchen found underwater -/
def total_objects : ℕ := objects_per_trip * number_of_trips

theorem gretchen_objects : total_objects = 18 := by
  sorry

end gretchen_objects_l2058_205808


namespace reduced_price_is_30_l2058_205841

/-- Represents the price reduction of oil as a percentage -/
def price_reduction : ℝ := 0.20

/-- Represents the additional amount of oil that can be purchased after the price reduction -/
def additional_oil : ℝ := 10

/-- Represents the total cost in Rupees -/
def total_cost : ℝ := 1500

/-- Calculates the reduced price per kg of oil -/
def reduced_price_per_kg (original_price : ℝ) : ℝ :=
  original_price * (1 - price_reduction)

/-- Theorem stating that the reduced price per kg of oil is 30 Rupees -/
theorem reduced_price_is_30 :
  ∃ (original_price : ℝ) (original_quantity : ℝ),
    original_quantity > 0 ∧
    original_price > 0 ∧
    original_quantity * original_price = total_cost ∧
    (original_quantity + additional_oil) * reduced_price_per_kg original_price = total_cost ∧
    reduced_price_per_kg original_price = 30 :=
  sorry

end reduced_price_is_30_l2058_205841


namespace number_calculations_l2058_205811

/-- The number that is 17 more than 5 times X -/
def number_more_than_5x (x : ℝ) : ℝ := 5 * x + 17

/-- The number that is less than 5 times 22 by Y -/
def number_less_than_5_times_22 (y : ℝ) : ℝ := 22 * 5 - y

theorem number_calculations (x y : ℝ) : 
  (number_more_than_5x x = 5 * x + 17) ∧ 
  (number_less_than_5_times_22 y = 22 * 5 - y) :=
by sorry

end number_calculations_l2058_205811


namespace rent_increase_problem_l2058_205840

theorem rent_increase_problem (num_friends : ℕ) (original_rent : ℝ) (increase_percentage : ℝ) (new_mean : ℝ) : 
  num_friends = 4 →
  original_rent = 1400 →
  increase_percentage = 0.20 →
  new_mean = 870 →
  (num_friends * new_mean - original_rent * increase_percentage) / num_friends = 800 := by
sorry

end rent_increase_problem_l2058_205840


namespace prob_at_least_eight_sixes_l2058_205897

/-- The probability of rolling a six on a fair die -/
def prob_six : ℚ := 1/6

/-- The number of rolls -/
def num_rolls : ℕ := 10

/-- The minimum number of sixes required -/
def min_sixes : ℕ := 8

/-- Calculates the probability of rolling exactly k sixes in n rolls -/
def prob_exact_sixes (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (prob_six ^ k) * ((1 - prob_six) ^ (n - k))

/-- The probability of rolling at least 8 sixes in 10 rolls of a fair die -/
theorem prob_at_least_eight_sixes : 
  (prob_exact_sixes num_rolls min_sixes + 
   prob_exact_sixes num_rolls (min_sixes + 1) + 
   prob_exact_sixes num_rolls (min_sixes + 2)) = 3/15504 := by
  sorry

end prob_at_least_eight_sixes_l2058_205897


namespace complement_A_intersect_B_l2058_205881

open Set

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.log (x - 1)}
def B : Set ℝ := {x | -1 < x ∧ x < 2}

-- State the theorem
theorem complement_A_intersect_B :
  (Aᶜ ∩ B) = Ioo (-1 : ℝ) 1 ∪ {1} := by sorry

end complement_A_intersect_B_l2058_205881


namespace a_plus_b_equals_two_l2058_205886

theorem a_plus_b_equals_two (a b : ℝ) : 
  (∀ x y : ℝ, y = a + b / x) →
  (2 = a + b / 1) →
  (4 = a + b / 4) →
  a + b = 2 := by
sorry

end a_plus_b_equals_two_l2058_205886


namespace shop_owner_gain_l2058_205851

/-- Represents the problem of calculating the gain in terms of cloth meters for a shop owner. -/
theorem shop_owner_gain (total_meters : ℝ) (gain_percentage : ℝ) (gain_meters : ℝ) : 
  total_meters = 30 ∧ 
  gain_percentage = 50 / 100 → 
  gain_meters = 10 := by
  sorry


end shop_owner_gain_l2058_205851


namespace xyz_inequality_l2058_205899

theorem xyz_inequality : ∃ c : ℝ, ∀ x y z : ℝ, -|x*y*z| > c * (|x| + |y| + |z|) := by
  sorry

end xyz_inequality_l2058_205899


namespace problem_statement_l2058_205889

theorem problem_statement (x y P Q : ℝ) 
  (h1 : x + y = P)
  (h2 : x^2 + y^2 = Q)
  (h3 : x^3 + y^3 = P^2) :
  Q = 5 := by
  sorry

end problem_statement_l2058_205889


namespace elevator_descent_time_l2058_205807

/-- Represents the elevator descent problem -/
def elevator_descent (total_floors : ℕ) 
  (first_half_time : ℕ) 
  (mid_floor_time : ℕ) 
  (final_floor_time : ℕ) : Prop :=
  let first_half := total_floors / 2
  let mid_section := 5
  let final_section := 5
  let total_time := first_half_time + mid_floor_time * mid_section + final_floor_time * final_section
  total_floors = 20 ∧ 
  first_half_time = 15 ∧ 
  mid_floor_time = 5 ∧ 
  final_floor_time = 16 ∧ 
  total_time / 60 = 2

/-- Theorem stating that the elevator descent takes 2 hours -/
theorem elevator_descent_time : 
  elevator_descent 20 15 5 16 := by sorry

end elevator_descent_time_l2058_205807


namespace ch4_moles_formed_l2058_205814

/-- Represents the balanced chemical equation: Be2C + 4 H2O → 2 Be(OH)2 + 3 CH4 -/
structure ChemicalEquation where
  be2c_coeff : ℚ
  h2o_coeff : ℚ
  beoh2_coeff : ℚ
  ch4_coeff : ℚ

/-- Represents the available moles of reactants -/
structure AvailableReactants where
  be2c_moles : ℚ
  h2o_moles : ℚ

/-- Calculates the moles of CH4 formed based on the chemical equation and available reactants -/
def moles_ch4_formed (equation : ChemicalEquation) (reactants : AvailableReactants) : ℚ :=
  min
    (reactants.be2c_moles * equation.ch4_coeff / equation.be2c_coeff)
    (reactants.h2o_moles * equation.ch4_coeff / equation.h2o_coeff)

theorem ch4_moles_formed
  (equation : ChemicalEquation)
  (reactants : AvailableReactants)
  (h_equation : equation = ⟨1, 4, 2, 3⟩)
  (h_reactants : reactants = ⟨3, 12⟩) :
  moles_ch4_formed equation reactants = 9 := by
  sorry

end ch4_moles_formed_l2058_205814


namespace range_of_a_for_monotonic_f_l2058_205833

/-- The piecewise function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then 2 * x + a else x + 1

/-- The theorem stating the range of a for which f is monotonic -/
theorem range_of_a_for_monotonic_f :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ↔ a ≤ 1 :=
sorry

end range_of_a_for_monotonic_f_l2058_205833


namespace wayne_blocks_l2058_205802

/-- The number of blocks Wayne's father gave him -/
def blocks_given (initial final : ℕ) : ℕ := final - initial

/-- Proof that Wayne's father gave him 6 blocks -/
theorem wayne_blocks : blocks_given 9 15 = 6 := by
  sorry

end wayne_blocks_l2058_205802


namespace quadratic_perfect_square_l2058_205820

theorem quadratic_perfect_square (m : ℝ) :
  (∃ (a b : ℝ), ∀ x, (6*x^2 + 16*x + 3*m) / 6 = (a*x + b)^2) →
  m = 32/9 := by
sorry

end quadratic_perfect_square_l2058_205820


namespace square_diff_fourth_power_l2058_205894

theorem square_diff_fourth_power : (7^2 - 3^2)^4 = 2560000 := by
  sorry

end square_diff_fourth_power_l2058_205894


namespace same_color_probability_l2058_205880

def total_plates : ℕ := 11
def red_plates : ℕ := 6
def blue_plates : ℕ := 5
def selected_plates : ℕ := 3

theorem same_color_probability : 
  (Nat.choose red_plates selected_plates : ℚ) / (Nat.choose total_plates selected_plates) = 4 / 33 := by
sorry

end same_color_probability_l2058_205880


namespace fifteen_factorial_base_fifteen_zeros_l2058_205809

/-- The number of trailing zeros in n! when expressed in base b -/
def trailingZeros (n : ℕ) (b : ℕ) : ℕ :=
  sorry

theorem fifteen_factorial_base_fifteen_zeros :
  trailingZeros 15 15 = 3 :=
sorry

end fifteen_factorial_base_fifteen_zeros_l2058_205809


namespace product_of_powers_l2058_205884

theorem product_of_powers : 3^2 * 5^2 * 7 * 11^2 = 190575 := by
  sorry

end product_of_powers_l2058_205884


namespace tan_theta_value_l2058_205879

theorem tan_theta_value (θ : Real) 
  (h1 : Real.cos (θ / 2) = 4 / 5) 
  (h2 : Real.sin θ < 0) : 
  Real.tan θ = -24 / 7 := by
sorry

end tan_theta_value_l2058_205879


namespace marble_selection_probability_l2058_205885

theorem marble_selection_probability : 
  let total_marbles : ℕ := 12
  let marbles_per_color : ℕ := 3
  let colors : ℕ := 4
  let selected_marbles : ℕ := 4

  let total_ways : ℕ := Nat.choose total_marbles selected_marbles
  let favorable_ways : ℕ := marbles_per_color ^ colors

  (favorable_ways : ℚ) / total_ways = 9 / 55 :=
by sorry

end marble_selection_probability_l2058_205885


namespace embankment_construction_time_l2058_205842

theorem embankment_construction_time 
  (workers_initial : ℕ) 
  (days_initial : ℕ) 
  (embankments : ℕ) 
  (workers_new : ℕ) 
  (h1 : workers_initial = 75)
  (h2 : days_initial = 4)
  (h3 : embankments = 2)
  (h4 : workers_new = 60) :
  ∃ (days_new : ℕ), 
    (workers_initial * days_initial = workers_new * days_new) ∧ 
    (days_new = 5) := by
  sorry

end embankment_construction_time_l2058_205842


namespace max_value_of_C_l2058_205836

theorem max_value_of_C (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let a := 1 / y
  let b := y + 1 / x
  let C := min x (min a b)
  ∀ ε > 0, C ≤ Real.sqrt 2 + ε ∧ ∃ x' y' : ℝ, x' > 0 ∧ y' > 0 ∧
    let a' := 1 / y'
    let b' := y' + 1 / x'
    let C' := min x' (min a' b')
    C' > Real.sqrt 2 - ε :=
sorry

end max_value_of_C_l2058_205836


namespace binomial_remainder_l2058_205865

theorem binomial_remainder (x : ℕ) : x = 2000 → (1 - x)^1999 % 2000 = 1 := by
  sorry

end binomial_remainder_l2058_205865


namespace valid_grid_probability_l2058_205893

/-- Represents a 3x3 grid filled with numbers 1 to 9 --/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- Checks if a number is odd --/
def isOdd (n : Fin 9) : Bool :=
  n.val % 2 ≠ 0

/-- Checks if the sum of numbers in a row is odd --/
def isRowSumOdd (g : Grid) (row : Fin 3) : Bool :=
  isOdd (g row 0 + g row 1 + g row 2)

/-- Checks if the sum of numbers in a column is odd --/
def isColumnSumOdd (g : Grid) (col : Fin 3) : Bool :=
  isOdd (g 0 col + g 1 col + g 2 col)

/-- Checks if all rows and columns have odd sums --/
def isValidGrid (g : Grid) : Bool :=
  (∀ row, isRowSumOdd g row) ∧ (∀ col, isColumnSumOdd g col)

/-- The total number of possible 3x3 grids filled with numbers 1 to 9 --/
def totalGrids : Nat :=
  Nat.factorial 9

/-- The number of valid grids where all rows and columns have odd sums --/
def validGrids : Nat :=
  9

/-- The main theorem stating the probability of a valid grid --/
theorem valid_grid_probability :
  (validGrids : ℚ) / totalGrids = 1 / 14 :=
sorry

end valid_grid_probability_l2058_205893


namespace plane_line_relations_l2058_205829

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (parallel : Plane → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (lineparallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)

-- Axioms
axiom parallel_trans {α β γ : Plane} : parallel α β → parallel β γ → parallel α γ
axiom perpendicular_trans {l m : Line} {α β : Plane} : 
  perpendicular l α → perpendicular l β → perpendicular m α → perpendicular m β

-- Theorem
theorem plane_line_relations 
  (α β : Plane) (m n : Line) 
  (h_diff_planes : α ≠ β) 
  (h_diff_lines : m ≠ n) :
  (parallel α β ∧ contains α m → lineparallel m β) ∧
  (perpendicular n α ∧ perpendicular n β ∧ perpendicular m α → perpendicular m β) :=
by sorry

end plane_line_relations_l2058_205829


namespace factors_of_20160_l2058_205874

theorem factors_of_20160 : (Finset.filter (· ∣ 20160) (Finset.range 20161)).card = 72 := by
  sorry

end factors_of_20160_l2058_205874


namespace sufficient_condition_range_l2058_205887

theorem sufficient_condition_range (a : ℝ) : 
  (∀ x, x > a → (x - 1) / x > 0) → a ≥ 1 := by sorry

end sufficient_condition_range_l2058_205887


namespace system_solution_ratio_l2058_205843

theorem system_solution_ratio (k x y z : ℚ) : 
  x ≠ 0 → y ≠ 0 → z ≠ 0 →
  x + k * y + 4 * z = 0 →
  2 * x + k * y - 3 * z = 0 →
  x + 2 * y - 4 * z = 0 →
  x * z / (y * y) = 59 / 1024 := by
sorry

end system_solution_ratio_l2058_205843


namespace crate_height_is_16_feet_l2058_205831

/-- Represents the dimensions of a rectangular crate -/
structure CrateDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a cylindrical gas tank -/
structure GasTank where
  radius : ℝ

/-- Theorem stating that the height of the crate is 16 feet -/
theorem crate_height_is_16_feet (crate : CrateDimensions) (tank : GasTank) :
  crate.length = 12 ∧ crate.width = 16 ∧ tank.radius = 8 →
  crate.height = 16 := by
  sorry

#check crate_height_is_16_feet

end crate_height_is_16_feet_l2058_205831
