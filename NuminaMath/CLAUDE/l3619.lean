import Mathlib

namespace cloth_sale_meters_l3619_361957

/-- Proves that the number of meters of cloth sold is 40, given the conditions of the problem -/
theorem cloth_sale_meters : 
  -- C represents the cost price of 1 meter of cloth
  ∀ (C : ℝ), C > 0 →
  -- S represents the selling price of 1 meter of cloth
  ∀ (S : ℝ), S > C →
  -- The gain is the selling price of 10 meters
  let G := 10 * S
  -- The gain percentage is 1/3 (33.33333333333333%)
  let gain_percentage := (1 : ℝ) / 3
  -- M represents the number of meters sold
  ∃ (M : ℝ),
    -- The gain is equal to the gain percentage times the total cost
    G = gain_percentage * (M * C) ∧
    -- The selling price is the cost price plus the gain per meter
    S = C + G / M ∧
    -- The number of meters sold is 40
    M = 40 := by
  sorry

end cloth_sale_meters_l3619_361957


namespace optimal_planting_solution_l3619_361970

/-- Represents the planting problem with two types of flowers -/
structure PlantingProblem where
  costA3B4 : ℕ  -- Cost of 3 pots of A and 4 pots of B
  costA4B3 : ℕ  -- Cost of 4 pots of A and 3 pots of B
  totalPots : ℕ  -- Total number of pots to be planted
  survivalRateA : ℚ  -- Survival rate of type A flowers
  survivalRateB : ℚ  -- Survival rate of type B flowers
  maxReplacement : ℕ  -- Maximum number of pots to be replaced next year

/-- Represents the solution to the planting problem -/
structure PlantingSolution where
  costA : ℕ  -- Cost of each pot of type A flowers
  costB : ℕ  -- Cost of each pot of type B flowers
  potsA : ℕ  -- Number of pots of type A flowers to plant
  potsB : ℕ  -- Number of pots of type B flowers to plant
  totalCost : ℕ  -- Total cost of planting

/-- Theorem stating the optimal solution for the planting problem -/
theorem optimal_planting_solution (problem : PlantingProblem) 
  (h1 : problem.costA3B4 = 360)
  (h2 : problem.costA4B3 = 340)
  (h3 : problem.totalPots = 600)
  (h4 : problem.survivalRateA = 7/10)
  (h5 : problem.survivalRateB = 9/10)
  (h6 : problem.maxReplacement = 100) :
  ∃ (solution : PlantingSolution),
    solution.costA = 40 ∧
    solution.costB = 60 ∧
    solution.potsA = 200 ∧
    solution.potsB = 400 ∧
    solution.totalCost = 32000 ∧
    solution.potsA + solution.potsB = problem.totalPots ∧
    (1 - problem.survivalRateA) * solution.potsA + (1 - problem.survivalRateB) * solution.potsB ≤ problem.maxReplacement ∧
    ∀ (altSolution : PlantingSolution),
      altSolution.potsA + altSolution.potsB = problem.totalPots →
      (1 - problem.survivalRateA) * altSolution.potsA + (1 - problem.survivalRateB) * altSolution.potsB ≤ problem.maxReplacement →
      solution.totalCost ≤ altSolution.totalCost :=
by
  sorry


end optimal_planting_solution_l3619_361970


namespace trig_equation_solution_l3619_361955

theorem trig_equation_solution (t : ℝ) :
  4 * (Real.sin t * Real.cos t ^ 5 + Real.cos t * Real.sin t ^ 5) + Real.sin (2 * t) ^ 3 = 1 ↔
  ∃ k : ℤ, t = (-1) ^ k * (Real.pi / 12) + k * (Real.pi / 2) :=
by sorry

end trig_equation_solution_l3619_361955


namespace greatest_negative_root_of_equation_l3619_361998

open Real

theorem greatest_negative_root_of_equation :
  ∃ (x : ℝ), x = -7/6 ∧ 
  (sin (π * x) - cos (2 * π * x)) / ((sin (π * x) + 1)^2 + cos (π * x)^2) = 0 ∧
  (∀ y < 0, y > x → 
    (sin (π * y) - cos (2 * π * y)) / ((sin (π * y) + 1)^2 + cos (π * y)^2) ≠ 0) :=
sorry

end greatest_negative_root_of_equation_l3619_361998


namespace robin_gum_count_l3619_361919

/-- The number of packages of gum Robin has -/
def num_packages : ℕ := 25

/-- The number of pieces of gum in each package -/
def pieces_per_package : ℕ := 42

/-- The total number of pieces of gum Robin has -/
def total_pieces : ℕ := num_packages * pieces_per_package

theorem robin_gum_count : total_pieces = 1050 := by
  sorry

end robin_gum_count_l3619_361919


namespace right_handed_players_count_l3619_361923

theorem right_handed_players_count (total_players : ℕ) (throwers : ℕ) 
  (left_handed_percentage : ℚ) :
  total_players = 120 →
  throwers = 58 →
  left_handed_percentage = 40 / 100 →
  (total_players - throwers : ℚ) * left_handed_percentage = 24 →
  throwers + (total_players - throwers - 24) = 96 :=
by sorry

end right_handed_players_count_l3619_361923


namespace factor_expression_l3619_361900

theorem factor_expression (x : ℝ) : 75*x + 45 = 15*(5*x + 3) := by
  sorry

end factor_expression_l3619_361900


namespace ratio_difference_l3619_361954

theorem ratio_difference (a b c : ℝ) : 
  a / 3 = b / 5 ∧ b / 5 = c / 7 ∧ c = 56 → c - a = 32 := by
  sorry

end ratio_difference_l3619_361954


namespace system_solution_fractional_equation_solution_l3619_361977

-- System of equations
theorem system_solution :
  ∃ (x y : ℚ), 3 * x - 5 * y = 3 ∧ x / 2 - y / 3 = 1 ∧ x = 8 / 3 ∧ y = 1 := by sorry

-- Fractional equation
theorem fractional_equation_solution :
  ∃ (x : ℚ), x ≠ 1 ∧ x / (x - 1) + 1 = 3 / (2 * x - 2) ∧ x = 5 / 4 := by sorry

end system_solution_fractional_equation_solution_l3619_361977


namespace product_of_base6_digits_7891_l3619_361941

/-- The base 6 representation of a natural number -/
def toBase6 (n : ℕ) : List ℕ :=
  sorry

/-- The product of a list of natural numbers -/
def listProduct (l : List ℕ) : ℕ :=
  sorry

/-- Theorem: The product of the digits in the base 6 representation of 7891 is 0 -/
theorem product_of_base6_digits_7891 :
  listProduct (toBase6 7891) = 0 := by
  sorry

end product_of_base6_digits_7891_l3619_361941


namespace cookie_flour_weight_l3619_361946

/-- Given the conditions of Matt's cookie baking, prove that each bag of flour weighs 5 pounds -/
theorem cookie_flour_weight 
  (cookies_per_batch : ℕ) 
  (flour_per_batch : ℕ) 
  (num_bags : ℕ) 
  (cookies_eaten : ℕ) 
  (cookies_left : ℕ) 
  (h1 : cookies_per_batch = 12)
  (h2 : flour_per_batch = 2)
  (h3 : num_bags = 4)
  (h4 : cookies_eaten = 15)
  (h5 : cookies_left = 105) :
  (cookies_eaten + cookies_left) * flour_per_batch / (cookies_per_batch * num_bags) = 5 := by
  sorry

#check cookie_flour_weight

end cookie_flour_weight_l3619_361946


namespace relay_race_ratio_l3619_361958

/-- Relay race problem -/
theorem relay_race_ratio (mary susan jen tiffany : ℕ) : 
  susan = jen + 10 →
  jen = 30 →
  tiffany = mary - 7 →
  mary + susan + jen + tiffany = 223 →
  mary / susan = 2 := by
  sorry

end relay_race_ratio_l3619_361958


namespace additional_people_for_faster_mowing_l3619_361913

/-- Represents the number of people needed to mow a lawn in a given time -/
structure LawnMowing where
  people : ℕ
  hours : ℕ

/-- The work rate (people × hours) for mowing the lawn -/
def workRate (l : LawnMowing) : ℕ := l.people * l.hours

theorem additional_people_for_faster_mowing 
  (initial : LawnMowing) 
  (target : LawnMowing) 
  (h1 : initial.people = 4) 
  (h2 : initial.hours = 6) 
  (h3 : target.hours = 3) 
  (h4 : workRate initial = workRate target) : 
  target.people - initial.people = 4 := by
  sorry

end additional_people_for_faster_mowing_l3619_361913


namespace visible_sides_is_seventeen_l3619_361903

/-- Represents a polygon with a given number of sides. -/
structure Polygon where
  sides : Nat
  sides_positive : sides > 0

/-- The configuration of polygons in the problem. -/
def polygon_configuration : List Polygon :=
  [⟨4, by norm_num⟩, ⟨3, by norm_num⟩, ⟨5, by norm_num⟩, ⟨6, by norm_num⟩, ⟨7, by norm_num⟩]

/-- Calculates the number of visible sides in the configuration. -/
def visible_sides (config : List Polygon) : Nat :=
  (config.map (·.sides)).sum - 2 * (config.length - 1)

/-- Theorem stating that the number of visible sides in the given configuration is 17. -/
theorem visible_sides_is_seventeen :
  visible_sides polygon_configuration = 17 := by
  sorry

#eval visible_sides polygon_configuration

end visible_sides_is_seventeen_l3619_361903


namespace bobby_child_jumps_l3619_361985

/-- The number of jumps Bobby can do per minute as an adult -/
def adult_jumps : ℕ := 60

/-- The number of additional jumps Bobby can do as an adult compared to when he was a child -/
def additional_jumps : ℕ := 30

/-- The number of jumps Bobby could do per minute as a child -/
def child_jumps : ℕ := adult_jumps - additional_jumps

theorem bobby_child_jumps : child_jumps = 30 := by sorry

end bobby_child_jumps_l3619_361985


namespace corn_price_is_ten_cents_l3619_361997

/-- Represents the farmer's corn production and sales --/
structure CornFarmer where
  seeds_per_ear : ℕ
  seeds_per_bag : ℕ
  cost_per_bag : ℚ
  profit : ℚ
  ears_sold : ℕ

/-- Calculates the price per ear of corn --/
def price_per_ear (farmer : CornFarmer) : ℚ :=
  let total_seeds := farmer.ears_sold * farmer.seeds_per_ear
  let bags_needed := (total_seeds + farmer.seeds_per_bag - 1) / farmer.seeds_per_bag
  let seed_cost := bags_needed * farmer.cost_per_bag
  let total_revenue := farmer.profit + seed_cost
  total_revenue / farmer.ears_sold

/-- Theorem stating the price per ear of corn is $0.10 --/
theorem corn_price_is_ten_cents (farmer : CornFarmer) 
    (h1 : farmer.seeds_per_ear = 4)
    (h2 : farmer.seeds_per_bag = 100)
    (h3 : farmer.cost_per_bag = 1/2)
    (h4 : farmer.profit = 40)
    (h5 : farmer.ears_sold = 500) : 
  price_per_ear farmer = 1/10 := by
  sorry


end corn_price_is_ten_cents_l3619_361997


namespace am_length_l3619_361960

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a line
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

-- Define the problem setup
def problem_setup (c : Circle) (l1 l2 : Line) (M A B C : ℝ × ℝ) : Prop :=
  ∃ (BC BM : ℝ),
    -- l1 is tangent to c at A
    (∃ (t : ℝ), l1.point1 = c.center + t • (A - c.center) ∧ 
                l1.point2 = c.center + (t + 1) • (A - c.center) ∧
                ‖A - c.center‖ = c.radius) ∧
    -- l2 intersects c at B and C
    (∃ (t1 t2 : ℝ), l2.point1 + t1 • (l2.point2 - l2.point1) = B ∧
                    l2.point1 + t2 • (l2.point2 - l2.point1) = C ∧
                    ‖B - c.center‖ = c.radius ∧
                    ‖C - c.center‖ = c.radius) ∧
    -- BC = 7
    BC = 7 ∧
    -- BM = 9
    BM = 9

-- Theorem statement
theorem am_length (c : Circle) (l1 l2 : Line) (M A B C : ℝ × ℝ) 
  (h : problem_setup c l1 l2 M A B C) :
  ‖A - M‖ = 12 ∨ ‖A - M‖ = 3 * Real.sqrt 2 :=
sorry

end am_length_l3619_361960


namespace inclination_angle_of_line_l3619_361907

open Real

theorem inclination_angle_of_line (x y : ℝ) :
  let line_equation := x * tan (π / 3) + y + 2 = 0
  let inclination_angle := 2 * π / 3
  line_equation → ∃ α, α = inclination_angle ∧ tan α = -tan (π / 3) ∧ 0 ≤ α ∧ α < π :=
by
  sorry

end inclination_angle_of_line_l3619_361907


namespace cross_section_area_theorem_l3619_361952

/-- A rectangular parallelepiped inscribed in a sphere -/
structure InscribedParallelepiped where
  R : ℝ  -- radius of the sphere
  diagonal_inclination : ℝ  -- angle between diagonals and base plane
  diagonal_inclination_is_45 : diagonal_inclination = Real.pi / 4

/-- The cross-section plane of the parallelepiped -/
structure CrossSectionPlane (p : InscribedParallelepiped) where
  angle_with_diagonal : ℝ  -- angle between the plane and diagonal BD₁
  angle_is_arcsin_sqrt2_4 : angle_with_diagonal = Real.arcsin (Real.sqrt 2 / 4)

/-- The area of the cross-section -/
noncomputable def cross_section_area (p : InscribedParallelepiped) (plane : CrossSectionPlane p) : ℝ :=
  2 * p.R^2 * Real.sqrt 3 / 3

/-- Theorem stating that the area of the cross-section is (2R²√3)/3 -/
theorem cross_section_area_theorem (p : InscribedParallelepiped) (plane : CrossSectionPlane p) :
    cross_section_area p plane = 2 * p.R^2 * Real.sqrt 3 / 3 := by
  sorry

end cross_section_area_theorem_l3619_361952


namespace min_value_expression_l3619_361969

theorem min_value_expression (r s t : ℝ) 
  (h1 : 1 ≤ r) (h2 : r ≤ s) (h3 : s ≤ t) (h4 : t ≤ 4) :
  (r - 1)^2 + (s/r - 1)^2 + (t/s - 1)^2 + (4/t - 1)^2 ≥ 12 - 8 * Real.sqrt 2 := by
  sorry

end min_value_expression_l3619_361969


namespace first_year_exceeding_target_l3619_361953

def initial_investment : ℝ := 1.3
def annual_increase_rate : ℝ := 0.12
def target_investment : ℝ := 2.0
def start_year : ℕ := 2015

def investment (year : ℕ) : ℝ :=
  initial_investment * (1 + annual_increase_rate) ^ (year - start_year)

theorem first_year_exceeding_target :
  (∀ y < 2019, investment y ≤ target_investment) ∧
  investment 2019 > target_investment :=
sorry

end first_year_exceeding_target_l3619_361953


namespace expected_value_biased_coin_l3619_361988

/-- Expected value of winnings for a biased coin flip -/
theorem expected_value_biased_coin : 
  let prob_heads : ℚ := 2/5
  let prob_tails : ℚ := 3/5
  let win_heads : ℚ := 5
  let loss_tails : ℚ := 1
  prob_heads * win_heads - prob_tails * loss_tails = 7/5 := by
sorry

end expected_value_biased_coin_l3619_361988


namespace buy_one_get_one_free_promotion_l3619_361930

/-- Calculates the total number of items received in a "buy one get one free" promotion --/
def itemsReceived (itemCost : ℕ) (totalPaid : ℕ) : ℕ :=
  2 * (totalPaid / itemCost)

/-- Theorem: Given a "buy one get one free" promotion where each item costs $3
    and a total payment of $15, the number of items received is 10 --/
theorem buy_one_get_one_free_promotion (itemCost : ℕ) (totalPaid : ℕ) 
    (h1 : itemCost = 3) (h2 : totalPaid = 15) : 
    itemsReceived itemCost totalPaid = 10 := by
  sorry

#eval itemsReceived 3 15  -- Should output 10

end buy_one_get_one_free_promotion_l3619_361930


namespace john_zoo_snakes_l3619_361996

/-- The number of snakes John has in his zoo --/
def num_snakes : ℕ := 15

/-- The total number of animals in John's zoo --/
def total_animals : ℕ := 114

/-- Theorem stating that the number of snakes in John's zoo is correct --/
theorem john_zoo_snakes :
  (num_snakes : ℚ) +
  (2 * num_snakes : ℚ) +
  ((2 * num_snakes : ℚ) - 5) +
  ((2 * num_snakes : ℚ) - 5 + 8) +
  (1/3 * ((2 * num_snakes : ℚ) - 5 + 8)) = total_animals := by
  sorry

#check john_zoo_snakes

end john_zoo_snakes_l3619_361996


namespace absolute_value_equation_solution_product_absolute_value_equation_solution_product_holds_l3619_361947

theorem absolute_value_equation_solution_product : ℝ → Prop :=
  fun x ↦ (|2 * x - 14| - 5 = 1) → 
    ∃ y, (|2 * y - 14| - 5 = 1) ∧ x * y = 40 ∧ 
    ∀ z, (|2 * z - 14| - 5 = 1) → (z = x ∨ z = y)

-- Proof
theorem absolute_value_equation_solution_product_holds :
  ∃ a b : ℝ, absolute_value_equation_solution_product a ∧
             absolute_value_equation_solution_product b ∧
             a ≠ b :=
by
  sorry

end absolute_value_equation_solution_product_absolute_value_equation_solution_product_holds_l3619_361947


namespace ribbon_left_l3619_361917

theorem ribbon_left (num_gifts : ℕ) (ribbon_per_gift : ℚ) (total_ribbon : ℚ) :
  num_gifts = 8 →
  ribbon_per_gift = 3/2 →
  total_ribbon = 15 →
  total_ribbon - (num_gifts : ℚ) * ribbon_per_gift = 3 := by
sorry

end ribbon_left_l3619_361917


namespace min_variance_of_sample_l3619_361940

theorem min_variance_of_sample (x y : ℝ) : 
  (x + 1 + y + 5) / 4 = 2 → 
  ((x - 2)^2 + (1 - 2)^2 + (y - 2)^2 + (5 - 2)^2) / 4 ≥ 3 := by
  sorry

end min_variance_of_sample_l3619_361940


namespace circles_intersection_properties_l3619_361924

-- Define the circles
def circle_O1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0
def circle_O2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y = 0

-- Define the intersection points A and B
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define the line AB
def line_AB (x y : ℝ) : Prop := x - y = 0

-- Define the perpendicular bisector of AB
def perp_bisector_AB (x y : ℝ) : Prop := x + y - 1 = 0

-- Define a point P on circle O1
def P : ℝ × ℝ := sorry

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the distance from a point to a line
def distance_to_line (p : ℝ × ℝ) (l : (ℝ → ℝ → Prop)) : ℝ := sorry

-- Theorem statement
theorem circles_intersection_properties :
  (∀ x y, line_AB x y ↔ x = y) ∧
  (∀ x y, perp_bisector_AB x y ↔ x + y = 1) ∧
  (∃ P, circle_O1 P.1 P.2 ∧ 
    distance_to_line P line_AB = Real.sqrt 2 / 2 + 1) :=
sorry

end circles_intersection_properties_l3619_361924


namespace order_of_even_monotone_increasing_l3619_361956

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def monotone_increasing_on (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ S → y ∈ S → x ≤ y → f x ≤ f y

-- State the theorem
theorem order_of_even_monotone_increasing (heven : is_even f)
  (hmono : monotone_increasing_on f (Set.Ici 0)) :
  f (-Real.pi) > f 3 ∧ f 3 > f (-2) := by
  sorry

end order_of_even_monotone_increasing_l3619_361956


namespace factor_sum_l3619_361911

theorem factor_sum (P Q : ℝ) : 
  (∃ b c : ℝ, (X^2 + 3*X + 7) * (X^2 + b*X + c) = X^4 + P*X^2 + Q) → 
  P + Q = 50 := by
sorry

end factor_sum_l3619_361911


namespace equality_implies_product_equality_l3619_361967

theorem equality_implies_product_equality (a b c : ℝ) : a = b → a * c = b * c := by sorry

end equality_implies_product_equality_l3619_361967


namespace inequality_proof_l3619_361976

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (c + a))) + (1 / (c^3 * (a + b))) ≥ 3/2 :=
by sorry

end inequality_proof_l3619_361976


namespace train_speed_calculation_l3619_361986

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed_calculation (train_length bridge_length : ℝ) (time : ℝ) :
  train_length = 130 →
  bridge_length = 150 →
  time = 27.997760179185665 →
  ∃ (speed : ℝ), (abs (speed - 36.0036) < 0.0001 ∧ 
    speed = (train_length + bridge_length) / time * 3.6) := by
  sorry

end train_speed_calculation_l3619_361986


namespace line_graph_most_suitable_for_forest_data_l3619_361992

/-- Represents types of statistical graphs -/
inductive StatisticalGraph
| LineGraph
| BarChart
| PieChart
| ScatterPlot
| Histogram

/-- Represents characteristics of data and analysis requirements -/
structure DataCharacteristics where
  continuous : Bool
  timeSpan : ℕ
  decreasingTrend : Bool

/-- Determines the most suitable graph type for given data characteristics -/
def mostSuitableGraph (data : DataCharacteristics) : StatisticalGraph :=
  sorry

/-- Theorem stating that a line graph is the most suitable for the given forest area data -/
theorem line_graph_most_suitable_for_forest_data :
  let forestData : DataCharacteristics := {
    continuous := true,
    timeSpan := 20,
    decreasingTrend := true
  }
  mostSuitableGraph forestData = StatisticalGraph.LineGraph :=
sorry

end line_graph_most_suitable_for_forest_data_l3619_361992


namespace geometric_sequence_property_l3619_361942

/-- Given a geometric sequence of positive terms {a_n}, prove that if the sum of logarithms of certain terms equals 6, then the product of the first and fifteenth terms is 10000. -/
theorem geometric_sequence_property (a : ℕ → ℝ) (r : ℝ) :
  (∀ n, a n > 0) →  -- Sequence of positive terms
  (∀ n, a (n + 1) = a n * r) →  -- Geometric sequence property
  Real.log (a 3) + Real.log (a 8) + Real.log (a 13) = 6 →
  a 1 * a 15 = 10000 := by
sorry

end geometric_sequence_property_l3619_361942


namespace quadratic_root_sum_property_l3619_361909

theorem quadratic_root_sum_property (a b c : ℝ) (x₁ x₂ : ℝ) (p q r : ℝ) 
  (h1 : a ≠ 0)
  (h2 : a * x₁^2 + b * x₁ + c = 0)
  (h3 : a * x₂^2 + b * x₂ + c = 0)
  (h4 : p = x₁ + x₂)
  (h5 : q = x₁^2 + x₂^2)
  (h6 : r = x₁^3 + x₂^3) :
  a * r + b * q + c * p = 0 := by
  sorry

end quadratic_root_sum_property_l3619_361909


namespace min_buses_needed_l3619_361920

/-- The capacity of each bus -/
def bus_capacity : ℕ := 45

/-- The total number of students to be transported -/
def total_students : ℕ := 540

/-- The minimum number of buses needed -/
def min_buses : ℕ := 12

/-- Theorem: The minimum number of buses needed to transport all students is 12 -/
theorem min_buses_needed : 
  (∀ n : ℕ, n * bus_capacity ≥ total_students → n ≥ min_buses) ∧ 
  (min_buses * bus_capacity ≥ total_students) :=
sorry

end min_buses_needed_l3619_361920


namespace isosceles_triangle_area_isosceles_triangle_area_proof_l3619_361948

/-- The area of an isosceles triangle with two sides of length 13 and a base of 10 is 60 -/
theorem isosceles_triangle_area : ℝ → Prop :=
  fun area =>
    ∃ (x y z : ℝ),
      x = 13 ∧ y = 13 ∧ z = 10 ∧  -- Two sides are 13, base is 10
      x = y ∧                     -- Isosceles condition
      area = (z * (x ^ 2 - (z / 2) ^ 2).sqrt) / 2 ∧  -- Area formula
      area = 60

/-- Proof of the theorem -/
theorem isosceles_triangle_area_proof : isosceles_triangle_area 60 := by
  sorry

#check isosceles_triangle_area_proof

end isosceles_triangle_area_isosceles_triangle_area_proof_l3619_361948


namespace bananas_and_cantaloupe_cost_l3619_361962

/-- Represents the cost of various fruits -/
structure FruitCosts where
  apples : ℝ
  bananas : ℝ
  cantaloupe : ℝ
  dates : ℝ
  figs : ℝ

/-- The conditions of the fruit purchase problem -/
def fruitProblemConditions (costs : FruitCosts) : Prop :=
  costs.apples + costs.bananas + costs.cantaloupe + costs.dates + costs.figs = 30 ∧
  costs.dates = 3 * costs.apples ∧
  costs.cantaloupe = costs.apples - costs.bananas ∧
  costs.figs = costs.bananas

/-- The theorem stating that the cost of bananas and cantaloupe is 6 -/
theorem bananas_and_cantaloupe_cost (costs : FruitCosts) 
  (h : fruitProblemConditions costs) : 
  costs.bananas + costs.cantaloupe = 6 := by
  sorry


end bananas_and_cantaloupe_cost_l3619_361962


namespace quadratic_solution_property_l3619_361918

theorem quadratic_solution_property (k : ℝ) : 
  (∃ a b : ℝ, a ≠ b ∧ 
   3 * a^2 + 6 * a + k = 0 ∧ 
   3 * b^2 + 6 * b + k = 0 ∧
   |a - b| = (1/2) * (a^2 + b^2)) ↔ 
  (k = 0 ∨ k = 6) :=
sorry

end quadratic_solution_property_l3619_361918


namespace complex_square_equality_l3619_361978

theorem complex_square_equality (c d : ℕ+) :
  (↑c - Complex.I * ↑d) ^ 2 = 18 - 8 * Complex.I →
  ↑c - Complex.I * ↑d = 5 - Complex.I := by
sorry

end complex_square_equality_l3619_361978


namespace line_equation_through_point_with_slope_l3619_361943

/-- A line passing through (1, 0) with slope 3 has the equation 3x - y - 3 = 0 -/
theorem line_equation_through_point_with_slope (x y : ℝ) :
  (3 : ℝ) * x - y - 3 = 0 ↔ (y - 0 = 3 * (x - 1) ∧ (1, 0) ∈ {p : ℝ × ℝ | (3 : ℝ) * p.1 - p.2 - 3 = 0}) :=
by sorry

end line_equation_through_point_with_slope_l3619_361943


namespace geometric_sequence_ratio_l3619_361950

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_ratio
  (a : ℕ → ℝ)
  (h_geo : geometric_sequence a)
  (h_pos : ∀ n : ℕ, a n > 0)
  (h_cond : 2 * a 1 + a 2 = a 3) :
  (a 4 + a 5) / (a 3 + a 4) = 2 := by
  sorry

end geometric_sequence_ratio_l3619_361950


namespace purely_imaginary_complex_number_l3619_361912

theorem purely_imaginary_complex_number (a : ℝ) : 
  (∃ (z : ℂ), z = (a^2 + a - 2 : ℝ) + (a^2 - 3*a + 2 : ℝ)*I ∧ z.re = 0 ∧ z.im ≠ 0) → a = -2 := by
  sorry

end purely_imaginary_complex_number_l3619_361912


namespace steven_erasers_count_l3619_361904

/-- The number of skittles Steven has -/
def skittles : ℕ := 4502

/-- The number of groups the items are organized into -/
def groups : ℕ := 154

/-- The number of items in each group -/
def items_per_group : ℕ := 57

/-- The total number of items (skittles and erasers) -/
def total_items : ℕ := groups * items_per_group

/-- The number of erasers Steven has -/
def erasers : ℕ := total_items - skittles

theorem steven_erasers_count : erasers = 4276 := by
  sorry

end steven_erasers_count_l3619_361904


namespace negation_of_universal_proposition_l3619_361928

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x₀ : ℝ, x₀^3 - x₀^2 + 1 > 0) :=
by sorry

end negation_of_universal_proposition_l3619_361928


namespace absolute_value_sum_l3619_361984

theorem absolute_value_sum (a b c d : ℝ) 
  (h1 : a < b) (h2 : b < c) (h3 : c < d)
  (h4 : |a - b| = 3) (h5 : |b - c| = 4) (h6 : |c - d| = 5) :
  |a - d| = 12 := by
sorry

end absolute_value_sum_l3619_361984


namespace circle_center_sum_l3619_361966

/-- Given a circle with equation x^2 + y^2 = 4x - 6y + 9, prove that its center (h, k) satisfies h + k = -1 -/
theorem circle_center_sum (h k : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 4*x - 6*y + 9 ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 4*h + 6*k - 9)) → 
  h + k = -1 := by
sorry

end circle_center_sum_l3619_361966


namespace bus_capacity_l3619_361982

theorem bus_capacity :
  let left_seats : ℕ := 15
  let right_seats : ℕ := left_seats - 3
  let regular_seat_capacity : ℕ := 3
  let back_seat_capacity : ℕ := 12
  let total_regular_seats : ℕ := left_seats + right_seats
  let regular_seats_capacity : ℕ := total_regular_seats * regular_seat_capacity
  let total_capacity : ℕ := regular_seats_capacity + back_seat_capacity
  total_capacity = 93 := by
  sorry

end bus_capacity_l3619_361982


namespace cow_count_l3619_361922

/-- Represents the number of cows in a farm -/
def num_cows : ℕ := 40

/-- Represents the number of bags of husk consumed by a group of cows in 40 days -/
def group_consumption : ℕ := 40

/-- Represents the number of days it takes one cow to consume one bag of husk -/
def days_per_bag : ℕ := 40

/-- Represents the number of days over which the consumption is measured -/
def total_days : ℕ := 40

theorem cow_count :
  num_cows = group_consumption * days_per_bag / total_days :=
by sorry

end cow_count_l3619_361922


namespace simplify_fraction_product_l3619_361926

theorem simplify_fraction_product : (90 : ℚ) / 150 * 35 / 21 = 1 := by
  sorry

end simplify_fraction_product_l3619_361926


namespace negation_forall_positive_converse_product_zero_symmetry_implies_even_symmetry_shifted_functions_l3619_361916

-- Statement 1
theorem negation_forall_positive (P : ℝ → Prop) :
  (¬ ∀ x, P x) ↔ ∃ x, ¬(P x) :=
sorry

-- Statement 2
theorem converse_product_zero (a b : ℝ) :
  (a * b ≠ 0 → a ≠ 0 ∧ b ≠ 0) ↔ (a = 0 ∨ b = 0 → a * b = 0) :=
sorry

-- Statement 3
theorem symmetry_implies_even (f : ℝ → ℝ) :
  (∀ x, f (1 - x) = f (x - 1)) → (∀ x, f x = f (-x)) :=
sorry

-- Statement 4
theorem symmetry_shifted_functions (f : ℝ → ℝ) :
  ∀ x, f (x + 1) = f (-(x - 1)) :=
sorry

end negation_forall_positive_converse_product_zero_symmetry_implies_even_symmetry_shifted_functions_l3619_361916


namespace geometric_progression_fourth_term_l3619_361908

theorem geometric_progression_fourth_term 
  (a₁ a₂ a₃ : ℝ) 
  (h₁ : a₁ = 2^2) 
  (h₂ : a₂ = 2^(3/2)) 
  (h₃ : a₃ = 2) 
  (h_gp : ∃ r : ℝ, a₂ = a₁ * r ∧ a₃ = a₂ * r) : 
  ∃ a₄ : ℝ, a₄ = a₃ * (a₃ / a₂) ∧ a₄ = Real.sqrt 2 := by
  sorry

end geometric_progression_fourth_term_l3619_361908


namespace expression_simplification_l3619_361979

theorem expression_simplification (x : ℤ) 
  (h1 : x - 3 * (x - 2) ≥ 2) 
  (h2 : 4 * x - 2 < 5 * x - 1) : 
  (3 / (x - 1) - x - 1) / ((x - 2) / (x - 1)) = -2 := by
  sorry

end expression_simplification_l3619_361979


namespace function_inequality_l3619_361931

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the derivative condition
variable (h : ∀ x, deriv f x > deriv g x)

-- Define the theorem
theorem function_inequality (a x b : ℝ) (h_order : a < x ∧ x < b) :
  f x + g a > g x + f a := by sorry

end function_inequality_l3619_361931


namespace division_value_problem_l3619_361959

theorem division_value_problem (x : ℝ) : 
  (1376 / x) - 160 = 12 → x = 8 := by
  sorry

end division_value_problem_l3619_361959


namespace tamil_speakers_l3619_361981

theorem tamil_speakers (total_population : ℕ) (english_speakers : ℕ) (both_speakers : ℕ) (hindi_probability : ℚ) : 
  total_population = 1024 →
  english_speakers = 562 →
  both_speakers = 346 →
  hindi_probability = 0.0859375 →
  ∃ tamil_speakers : ℕ, tamil_speakers = 720 ∧ 
    tamil_speakers = total_population - (english_speakers + (total_population * hindi_probability).floor - both_speakers) :=
by
  sorry

end tamil_speakers_l3619_361981


namespace sqrt_equation_solution_l3619_361994

theorem sqrt_equation_solution (n : ℝ) : Real.sqrt (8 + n) = 9 → n = 73 := by
  sorry

end sqrt_equation_solution_l3619_361994


namespace eldest_child_age_l3619_361927

theorem eldest_child_age (y m e : ℕ) : 
  m = y + 3 →
  e = 3 * y →
  e = y + m + 2 →
  e = 15 :=
by
  sorry

end eldest_child_age_l3619_361927


namespace garden_border_perimeter_l3619_361975

/-- The total perimeter of Mrs. Hilt's garden border -/
theorem garden_border_perimeter :
  let num_rocks_a : ℝ := 125.0
  let circumference_a : ℝ := 0.5
  let num_rocks_b : ℝ := 64.0
  let circumference_b : ℝ := 0.7
  let total_perimeter : ℝ := num_rocks_a * circumference_a + num_rocks_b * circumference_b
  total_perimeter = 107.3 := by
sorry

end garden_border_perimeter_l3619_361975


namespace angle_inequalities_l3619_361915

theorem angle_inequalities (α β : Real) (h1 : π / 2 < α) (h2 : α < β) (h3 : β < π) :
  (π < α + β ∧ α + β < 2 * π) ∧
  (-π / 2 < α - β ∧ α - β < 0) ∧
  (1 / 2 < α / β ∧ α / β < 1) := by
  sorry

end angle_inequalities_l3619_361915


namespace discount_percentage_proof_l3619_361935

/-- Represents the discount percentage on bulk photocopy orders -/
def discount_percentage : ℝ := 25

/-- Represents the regular cost per photocopy in dollars -/
def regular_cost_per_copy : ℝ := 0.02

/-- Represents the number of copies in a bulk order -/
def bulk_order_size : ℕ := 160

/-- Represents the individual savings when placing a bulk order -/
def individual_savings : ℝ := 0.40

/-- Represents the total savings when two people place a bulk order together -/
def total_savings : ℝ := 2 * individual_savings

/-- Proves that the discount percentage is correct given the problem conditions -/
theorem discount_percentage_proof :
  discount_percentage = (total_savings / (regular_cost_per_copy * bulk_order_size)) * 100 :=
by sorry

end discount_percentage_proof_l3619_361935


namespace female_employees_count_l3619_361914

/-- Represents the number of employees in a company -/
structure Company where
  total_employees : ℕ
  female_managers : ℕ
  male_employees : ℕ
  female_employees : ℕ

/-- Conditions for the company -/
def company_conditions (c : Company) : Prop :=
  c.female_managers = 200 ∧
  c.total_employees * 2 = (c.female_managers + (c.male_employees * 2 / 5)) * 5 ∧
  c.total_employees = c.male_employees + c.female_employees

/-- Theorem stating that under the given conditions, the number of female employees is 500 -/
theorem female_employees_count (c : Company) :
  company_conditions c → c.female_employees = 500 := by
  sorry

end female_employees_count_l3619_361914


namespace coloring_perfect_square_difference_l3619_361925

/-- A coloring of integers using three colors -/
def Coloring := ℤ → Fin 3

/-- Theorem: For any coloring of integers using three colors, 
    there exist two distinct integers with the same color 
    whose difference is a perfect square -/
theorem coloring_perfect_square_difference (c : Coloring) : 
  ∃ (x y k : ℤ), x ≠ y ∧ c x = c y ∧ y - x = k^2 := by
  sorry

end coloring_perfect_square_difference_l3619_361925


namespace exam_results_l3619_361944

/-- Represents a student in the autonomous recruitment exam -/
structure Student where
  writtenProb : ℝ  -- Probability of passing the written exam
  oralProb : ℝ     -- Probability of passing the oral exam

/-- The autonomous recruitment exam setup -/
def ExamSetup : (Student × Student × Student) :=
  (⟨0.6, 0.5⟩, ⟨0.5, 0.6⟩, ⟨0.4, 0.75⟩)

/-- Calculates the probability of exactly one student passing the written exam -/
noncomputable def probExactlyOnePassWritten (setup : Student × Student × Student) : ℝ :=
  sorry

/-- Calculates the expected number of pre-admitted students -/
noncomputable def expectedPreAdmitted (setup : Student × Student × Student) : ℝ :=
  sorry

/-- Main theorem stating the results of the calculations -/
theorem exam_results :
  let setup := ExamSetup
  probExactlyOnePassWritten setup = 0.38 ∧
  expectedPreAdmitted setup = 0.9 := by
  sorry

end exam_results_l3619_361944


namespace unique_triple_lcm_gcd_l3619_361905

theorem unique_triple_lcm_gcd : 
  ∃! (x y z : ℕ+), 
    Nat.lcm x y = 100 ∧ 
    Nat.lcm x z = 450 ∧ 
    Nat.lcm y z = 1100 ∧ 
    Nat.gcd (Nat.gcd x y) z = 5 := by
  sorry

end unique_triple_lcm_gcd_l3619_361905


namespace mean_of_remaining_numbers_l3619_361921

def numbers : List ℝ := [1924, 2057, 2170, 2229, 2301, 2365]

theorem mean_of_remaining_numbers (subset : List ℝ) (h1 : subset ⊆ numbers) 
  (h2 : subset.length = 4) (h3 : (subset.sum / subset.length) = 2187.25) :
  let remaining := numbers.filter (fun x => x ∉ subset)
  (remaining.sum / remaining.length) = 2148.5 := by
sorry

end mean_of_remaining_numbers_l3619_361921


namespace second_company_visit_charge_l3619_361937

/-- Paul's Plumbing visit charge -/
def pauls_visit_charge : ℕ := 55

/-- Paul's Plumbing hourly labor charge -/
def pauls_hourly_charge : ℕ := 35

/-- Second company's hourly labor charge -/
def second_hourly_charge : ℕ := 30

/-- Number of labor hours -/
def labor_hours : ℕ := 4

/-- Second company's visit charge -/
def second_visit_charge : ℕ := 75

theorem second_company_visit_charge :
  pauls_visit_charge + labor_hours * pauls_hourly_charge =
  second_visit_charge + labor_hours * second_hourly_charge :=
by sorry

end second_company_visit_charge_l3619_361937


namespace sum_of_fractions_l3619_361974

theorem sum_of_fractions : (1 : ℚ) / 3 + 5 / 9 = 8 / 9 := by
  sorry

end sum_of_fractions_l3619_361974


namespace seven_valid_configurations_l3619_361983

/-- A polygon shape made of congruent squares -/
structure SquarePolygon where
  squares : ℕ
  shape : String

/-- Possible positions to attach an additional square -/
def AttachmentPositions : ℕ := 11

/-- A cube with one face missing requires this many squares -/
def CubeSquares : ℕ := 5

/-- The base cross-shaped polygon -/
def baseCross : SquarePolygon :=
  { squares := 6, shape := "cross" }

/-- Predicate for whether a polygon can form a cube with one face missing -/
def canFormCube (p : SquarePolygon) : Prop := sorry

/-- The number of valid configurations that can form a cube with one face missing -/
def validConfigurations : ℕ := 7

/-- Main theorem: There are exactly 7 valid configurations -/
theorem seven_valid_configurations :
  (∃ (configs : Finset SquarePolygon),
    configs.card = validConfigurations ∧
    (∀ p ∈ configs, p.squares = baseCross.squares + 1 ∧ canFormCube p) ∧
    (∀ p : SquarePolygon, p.squares = baseCross.squares + 1 →
      canFormCube p → p ∈ configs)) := by sorry

end seven_valid_configurations_l3619_361983


namespace solve_star_equation_l3619_361901

/-- Custom binary operation -/
def star (a b : ℝ) : ℝ := 2 * a * b - 3 * b - a

/-- Theorem statement -/
theorem solve_star_equation :
  ∀ y : ℝ, star 4 y = 80 → y = 84 / 5 := by
  sorry

end solve_star_equation_l3619_361901


namespace distance_between_foci_rectangular_hyperbola_l3619_361906

/-- The distance between the foci of a rectangular hyperbola -/
theorem distance_between_foci_rectangular_hyperbola (c : ℝ) :
  let hyperbola := {(x, y) : ℝ × ℝ | x * y = c^2}
  let foci := {(c, c), (-c, -c)}
  (Set.ncard foci = 2) →
  ∀ (f₁ f₂ : ℝ × ℝ), f₁ ∈ foci → f₂ ∈ foci → f₁ ≠ f₂ →
    Real.sqrt ((f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2) = 2 * Real.sqrt 2 * c :=
by sorry

end distance_between_foci_rectangular_hyperbola_l3619_361906


namespace server_data_requests_l3619_361929

/-- The number of data requests processed by a server in 24 hours -/
def data_requests_per_day (requests_per_minute : ℕ) : ℕ :=
  requests_per_minute * (24 * 60)

/-- Theorem stating that a server processing 15,000 data requests per minute
    will process 21,600,000 data requests in 24 hours -/
theorem server_data_requests :
  data_requests_per_day 15000 = 21600000 := by
  sorry

end server_data_requests_l3619_361929


namespace min_group_size_repunit_sum_l3619_361933

def is_repunit (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 0 ∧ n = (10^k - 1) / 9

theorem min_group_size_repunit_sum :
  ∃ m : ℕ, m > 1 ∧
    (∀ m' : ℕ, m' > 1 → m' < m →
      ¬∃ n k : ℕ, n > k ∧ k > 1 ∧
        is_repunit n ∧ is_repunit k ∧ n = k * m') ∧
    (∃ n k : ℕ, n > k ∧ k > 1 ∧
      is_repunit n ∧ is_repunit k ∧ n = k * m) ∧
  m = 101 :=
sorry

end min_group_size_repunit_sum_l3619_361933


namespace smallest_fraction_greater_than_three_fourths_l3619_361945

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem smallest_fraction_greater_than_three_fourths :
  ∃ (a b : ℕ), 
    is_two_digit a ∧ 
    is_two_digit b ∧ 
    (a : ℚ) / b > 3 / 4 ∧
    (∀ (c d : ℕ), is_two_digit c → is_two_digit d → (c : ℚ) / d > 3 / 4 → a ≤ c) ∧
    a = 73 :=
by sorry

end smallest_fraction_greater_than_three_fourths_l3619_361945


namespace range_of_a_for_decreasing_f_l3619_361989

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (a - 3) * x + 5 else 2 * a / x

-- State the theorem
theorem range_of_a_for_decreasing_f :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x > f a y) → 0 < a ∧ a ≤ 2 :=
by sorry

end range_of_a_for_decreasing_f_l3619_361989


namespace chess_draw_probability_l3619_361964

theorem chess_draw_probability (p_win p_not_lose : ℝ) 
  (h1 : p_win = 0.3) 
  (h2 : p_not_lose = 0.8) : 
  p_not_lose - p_win = 0.5 := by
sorry

end chess_draw_probability_l3619_361964


namespace trajectory_and_intersection_l3619_361973

-- Define the circles M and N
def circle_M (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1
def circle_N (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 9

-- Define the trajectory curve C
def curve_C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1 ∧ x ≠ -2

-- Define the line l passing through (-4,0) and tangent to circle M
def line_l (x y : ℝ) : Prop := ∃ k : ℝ, y = k * (x + 4) ∧ k^2 / (1 + k^2) = 1/9

-- Theorem statement
theorem trajectory_and_intersection :
  -- The trajectory of the center of circle P forms curve C
  (∀ x y : ℝ, (∃ r : ℝ, 0 < r ∧ r < 3 ∧
    (∀ x' y' : ℝ, (x' - x)^2 + (y' - y)^2 = r^2 →
      (circle_M x' y' → (x' - x)^2 + (y' - y)^2 = (1 + r)^2) ∧
      (circle_N x' y' → (x' - x)^2 + (y' - y)^2 = (3 - r)^2))
  ) → curve_C x y) ∧
  -- The line l intersects curve C at two points with distance 18/7
  (∀ x₁ y₁ x₂ y₂ : ℝ, curve_C x₁ y₁ ∧ curve_C x₂ y₂ ∧
    line_l x₁ y₁ ∧ line_l x₂ y₂ ∧ (x₁, y₁) ≠ (x₂, y₂) →
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = (18/7)^2) :=
by sorry

end trajectory_and_intersection_l3619_361973


namespace nate_cooking_for_eight_l3619_361999

/-- The number of scallops per pound -/
def scallops_per_pound : ℕ := 8

/-- The cost of scallops per pound in cents -/
def cost_per_pound : ℕ := 2400

/-- The number of scallops per person -/
def scallops_per_person : ℕ := 2

/-- The total cost of scallops Nate is spending in cents -/
def total_cost : ℕ := 4800

/-- The number of people Nate is cooking for -/
def number_of_people : ℕ := total_cost / cost_per_pound * scallops_per_pound / scallops_per_person

theorem nate_cooking_for_eight : number_of_people = 8 := by
  sorry

end nate_cooking_for_eight_l3619_361999


namespace acid_solution_volume_l3619_361965

/-- Given a volume of pure acid in a solution with a known concentration,
    calculate the total volume of the solution. -/
theorem acid_solution_volume (pure_acid : ℝ) (concentration : ℝ) 
    (h1 : pure_acid = 4.8)
    (h2 : concentration = 0.4) : 
    pure_acid / concentration = 12 := by
  sorry

end acid_solution_volume_l3619_361965


namespace container_capacity_l3619_361971

theorem container_capacity : 
  ∀ (C : ℝ), 
    C > 0 → 
    (0.40 * C + 28 = 0.75 * C) → 
    C = 80 :=
by
  sorry

end container_capacity_l3619_361971


namespace multiples_properties_l3619_361910

theorem multiples_properties (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 4 * k) 
  (hb : ∃ k : ℤ, b = 8 * k) : 
  (∃ k : ℤ, b = 4 * k) ∧ 
  (∃ k : ℤ, a - b = 4 * k) ∧ 
  (∃ k : ℤ, a + b = 2 * k) := by
sorry

end multiples_properties_l3619_361910


namespace polynomial_division_theorem_l3619_361951

-- Define the polynomials
def f (x : ℝ) : ℝ := 3*x^5 + 7*x^4 - 15*x^3 - 35*x^2 + 22*x + 24
def g (x : ℝ) : ℝ := x^3 + 5*x^2 - 4*x + 2
def r (x : ℝ) : ℝ := -258*x^2 + 186*x - 50

-- State the theorem
theorem polynomial_division_theorem :
  ∃ q : ℝ → ℝ, ∀ x, f x = g x * q x + r x ∧ (∀ x, r x = -258*x^2 + 186*x - 50) :=
sorry

end polynomial_division_theorem_l3619_361951


namespace planes_lines_false_implications_l3619_361949

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (line_parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

theorem planes_lines_false_implications 
  (α β : Plane) (l m : Line) :
  ∃ (α β : Plane) (l m : Line),
    α ≠ β ∧ l ≠ m ∧
    subset l α ∧ subset m β ∧
    ¬(¬(parallel α β) → ¬(line_parallel l m)) ∧
    ¬(perpendicular l m → plane_perpendicular α β) := by
  sorry

end planes_lines_false_implications_l3619_361949


namespace yoongi_age_l3619_361972

theorem yoongi_age (yoongi_age hoseok_age : ℕ) 
  (sum_of_ages : yoongi_age + hoseok_age = 16)
  (age_difference : yoongi_age = hoseok_age + 2) : 
  yoongi_age = 9 := by
sorry

end yoongi_age_l3619_361972


namespace cake_division_possible_l3619_361991

/-- Represents the different ways a cake can be divided -/
inductive CakePortion
  | Whole
  | Half
  | Third

/-- Represents the distribution of cakes to children -/
structure CakeDistribution where
  whole : Nat
  half : Nat
  third : Nat

/-- Calculates the total portion of cake for a given distribution -/
def totalPortion (d : CakeDistribution) : Rat :=
  d.whole + d.half / 2 + d.third / 3

theorem cake_division_possible : ∃ (d : CakeDistribution),
  -- Each child gets the same amount
  totalPortion d = 13 / 6 ∧
  -- The distribution uses exactly 13 cakes
  d.whole + d.half + d.third = 13 ∧
  -- The number of half cakes is even (so they can be paired)
  d.half % 2 = 0 ∧
  -- The number of third cakes is divisible by 3 (so they can be grouped)
  d.third % 3 = 0 :=
sorry

end cake_division_possible_l3619_361991


namespace ceiling_fraction_equality_l3619_361980

theorem ceiling_fraction_equality : 
  (⌈(23 : ℚ) / 9 - ⌈(35 : ℚ) / 23⌉⌉) / (⌈(35 : ℚ) / 9 + ⌈(9 : ℚ) * 23 / 35⌉⌉) = (1 : ℚ) / 10 := by
  sorry

end ceiling_fraction_equality_l3619_361980


namespace cheryl_walk_distance_l3619_361961

/-- Calculates the total distance walked by a person who walks at a constant speed
    for a given time in one direction and then returns along the same path. -/
def total_distance_walked (speed : ℝ) (time : ℝ) : ℝ :=
  2 * speed * time

/-- Theorem: Given a person walking at 2 miles per hour for 3 hours in one direction
    and then returning along the same path, the total distance walked is 12 miles. -/
theorem cheryl_walk_distance :
  total_distance_walked 2 3 = 12 := by
  sorry

#eval total_distance_walked 2 3

end cheryl_walk_distance_l3619_361961


namespace smallest_equivalent_angle_proof_l3619_361990

/-- The smallest positive angle in [0°, 360°) with the same terminal side as 2011° -/
def smallest_equivalent_angle : ℝ := 211

/-- Two angles have the same terminal side if they differ by a multiple of 360° -/
def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, α = β + k * 360

theorem smallest_equivalent_angle_proof :
  same_terminal_side smallest_equivalent_angle 2011 ∧
  smallest_equivalent_angle ≥ 0 ∧
  smallest_equivalent_angle < 360 ∧
  ∀ θ, 0 ≤ θ ∧ θ < 360 ∧ same_terminal_side θ 2011 → θ ≥ smallest_equivalent_angle := by
  sorry


end smallest_equivalent_angle_proof_l3619_361990


namespace complex_point_location_l3619_361936

theorem complex_point_location (z : ℂ) (h : z = 1 + I) :
  let w := 2 / z + z^2
  0 < w.re ∧ 0 < w.im :=
by sorry

end complex_point_location_l3619_361936


namespace largest_solution_of_quartic_l3619_361938

theorem largest_solution_of_quartic (x : ℝ) : 
  x^4 - 50*x^2 + 625 = 0 → x ≤ 5 ∧ ∃ y, y^4 - 50*y^2 + 625 = 0 ∧ y = 5 :=
by sorry

end largest_solution_of_quartic_l3619_361938


namespace nina_total_homework_l3619_361987

/-- Represents the number of homework assignments for a student -/
structure Homework where
  math : ℕ
  reading : ℕ

/-- Calculates the total number of homework assignments -/
def totalHomework (hw : Homework) : ℕ := hw.math + hw.reading

theorem nina_total_homework :
  let ruby : Homework := { math := 6, reading := 2 }
  let nina : Homework := { math := 4 * ruby.math, reading := 8 * ruby.reading }
  totalHomework nina = 40 := by
  sorry

end nina_total_homework_l3619_361987


namespace johns_number_is_55_l3619_361932

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_reversal (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  ones * 100 + tens * 10 + hundreds

theorem johns_number_is_55 :
  ∃! n : ℕ, is_three_digit n ∧
    321 ≤ digit_reversal (2 * n + 13) ∧
    digit_reversal (2 * n + 13) ≤ 325 ∧
    n = 55 :=
sorry

end johns_number_is_55_l3619_361932


namespace least_positive_angle_theorem_l3619_361963

/-- The least positive angle θ (in degrees) satisfying cos 10° = sin 15° + sin θ is 32.5° -/
theorem least_positive_angle_theorem : 
  ∃ θ : ℝ, θ > 0 ∧ θ = 32.5 ∧ 
  (∀ φ : ℝ, φ > 0 ∧ Real.cos (10 * π / 180) = Real.sin (15 * π / 180) + Real.sin (φ * π / 180) → θ ≤ φ) ∧
  Real.cos (10 * π / 180) = Real.sin (15 * π / 180) + Real.sin (θ * π / 180) := by
  sorry


end least_positive_angle_theorem_l3619_361963


namespace two_sarees_four_shirts_cost_l3619_361993

/-- The price of a single saree -/
def saree_price : ℝ := sorry

/-- The price of a single shirt -/
def shirt_price : ℝ := sorry

/-- The cost of 2 sarees and 4 shirts equals the cost of 1 saree and 6 shirts -/
axiom price_equality : 2 * saree_price + 4 * shirt_price = saree_price + 6 * shirt_price

/-- The price of 12 shirts is $2400 -/
axiom twelve_shirts_price : 12 * shirt_price = 2400

/-- The theorem stating that 2 sarees and 4 shirts cost $1600 -/
theorem two_sarees_four_shirts_cost : 2 * saree_price + 4 * shirt_price = 1600 := by sorry

end two_sarees_four_shirts_cost_l3619_361993


namespace min_sum_of_squares_l3619_361995

theorem min_sum_of_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 8) :
  ∃ (m : ℝ), m = 4 ∧ ∀ (a b c : ℝ), a^3 + b^3 + c^3 - 3*a*b*c = 8 → x^2 + y^2 + z^2 ≤ a^2 + b^2 + c^2 := by
  sorry

end min_sum_of_squares_l3619_361995


namespace min_value_expression_l3619_361934

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 3 * b = 4) :
  1 / (a + 1) + 3 / (b + 1) ≥ 2 ∧
  (1 / (a + 1) + 3 / (b + 1) = 2 ↔ a = 1 ∧ b = 1) :=
by sorry

end min_value_expression_l3619_361934


namespace product_equals_sum_implies_y_value_l3619_361939

theorem product_equals_sum_implies_y_value :
  ∀ y : ℚ, (2 * 3 * 5 * y = 2 + 3 + 5 + y) → y = 10 / 29 := by
  sorry

end product_equals_sum_implies_y_value_l3619_361939


namespace inequality_chain_l3619_361968

theorem inequality_chain (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  9 / (a + b + c) ≤ 2 / (a + b) + 2 / (b + c) + 2 / (c + a) ∧
  2 / (a + b) + 2 / (b + c) + 2 / (c + a) ≤ 1 / a + 1 / b + 1 / c :=
by sorry

end inequality_chain_l3619_361968


namespace product_42_sum_9_l3619_361902

theorem product_42_sum_9 (a b c : ℕ+) : 
  a * b * c = 42 → a + b = 9 → c = 3 := by
  sorry

end product_42_sum_9_l3619_361902
