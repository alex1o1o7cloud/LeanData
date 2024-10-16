import Mathlib

namespace NUMINAMATH_CALUDE_employee_count_proof_l3586_358666

/-- The number of employees in the room -/
def num_employees : ℕ := 25000

/-- The initial percentage of managers as a rational number -/
def initial_manager_percentage : ℚ := 99 / 100

/-- The final percentage of managers as a rational number -/
def final_manager_percentage : ℚ := 98 / 100

/-- The number of managers that leave the room -/
def managers_leaving : ℕ := 250

theorem employee_count_proof :
  (initial_manager_percentage * num_employees : ℚ) - managers_leaving = 
  final_manager_percentage * num_employees :=
by sorry

end NUMINAMATH_CALUDE_employee_count_proof_l3586_358666


namespace NUMINAMATH_CALUDE_quadratic_root_ratio_l3586_358635

/-- Given a quadratic equation ax^2 + bx + c = 0 where a ≠ 0 and c ≠ 0,
    if one root is 4 times the other root, then b^2 / (ac) = 25/4 -/
theorem quadratic_root_ratio (a b c : ℝ) (ha : a ≠ 0) (hc : c ≠ 0) :
  (∃ x y : ℝ, x = 4 * y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) →
  b^2 / (a * c) = 25 / 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_ratio_l3586_358635


namespace NUMINAMATH_CALUDE_car_sale_profit_l3586_358645

theorem car_sale_profit (original_price : ℝ) (h : original_price > 0) :
  let purchase_price := 0.80 * original_price
  let selling_price := 1.6000000000000001 * original_price
  let profit_percentage := (selling_price - purchase_price) / purchase_price * 100
  profit_percentage = 100.00000000000001 := by
sorry

end NUMINAMATH_CALUDE_car_sale_profit_l3586_358645


namespace NUMINAMATH_CALUDE_basis_transformation_l3586_358622

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem basis_transformation (OA OB OC : V) 
  (h : LinearIndependent ℝ ![OA, OB, OC]) 
  (h_span : Submodule.span ℝ {OA, OB, OC} = ⊤) :
  LinearIndependent ℝ ![OA + OB, OA - OB, OC] ∧ 
  Submodule.span ℝ {OA + OB, OA - OB, OC} = ⊤ := by
  sorry

end NUMINAMATH_CALUDE_basis_transformation_l3586_358622


namespace NUMINAMATH_CALUDE_cubic_quartic_system_solution_l3586_358699

theorem cubic_quartic_system_solution (x y : ℝ) 
  (h1 : x^3 + y^3 = 1) 
  (h2 : x^4 + y^4 = 1) : 
  (x = 0 ∧ y = 1) ∨ (x = 1 ∧ y = 0) := by
  sorry

end NUMINAMATH_CALUDE_cubic_quartic_system_solution_l3586_358699


namespace NUMINAMATH_CALUDE_current_velocity_proof_l3586_358653

/-- The velocity of the current in a river, given the following conditions:
  1. A man can row at 5 kmph in still water.
  2. It takes him 1 hour to row to a place and come back.
  3. The place is 2.4 km away. -/
def current_velocity : ℝ := 1

/-- The man's rowing speed in still water (in kmph) -/
def rowing_speed : ℝ := 5

/-- The distance to the destination (in km) -/
def distance : ℝ := 2.4

/-- The total time for the round trip (in hours) -/
def total_time : ℝ := 1

theorem current_velocity_proof :
  (distance / (rowing_speed + current_velocity) +
   distance / (rowing_speed - current_velocity) = total_time) ∧
  (current_velocity > 0) ∧
  (current_velocity < rowing_speed) := by
  sorry

end NUMINAMATH_CALUDE_current_velocity_proof_l3586_358653


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3586_358615

theorem trigonometric_identity : 
  - Real.sin (133 * π / 180) * Real.cos (197 * π / 180) - 
  Real.cos (47 * π / 180) * Real.cos (73 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3586_358615


namespace NUMINAMATH_CALUDE_ten_streets_intersections_l3586_358667

/-- The number of intersections created by n non-parallel streets -/
def intersections (n : ℕ) : ℕ := n.choose 2

/-- The theorem stating that 10 non-parallel streets create 45 intersections -/
theorem ten_streets_intersections : intersections 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_ten_streets_intersections_l3586_358667


namespace NUMINAMATH_CALUDE_inverse_f_at_150_l3586_358669

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^4 + 6

-- State the theorem
theorem inverse_f_at_150 :
  ∃ (y : ℝ), f y = 150 ∧ y = (48 : ℝ)^(1/4) :=
sorry

end NUMINAMATH_CALUDE_inverse_f_at_150_l3586_358669


namespace NUMINAMATH_CALUDE_minimum_bundle_price_l3586_358632

/-- Represents the cost and composition of a bundle --/
structure Bundle where
  water_bottle_cost : ℚ
  fruit_cost : ℚ
  snack_cost : ℚ
  water_bottles : ℕ
  fruits : ℕ
  snacks : ℕ

/-- Calculates the total cost of a bundle --/
def bundle_cost (b : Bundle) : ℚ :=
  b.water_bottle_cost * b.water_bottles +
  b.fruit_cost * b.fruits +
  b.snack_cost * b.snacks

/-- Represents the pricing strategy for bundles --/
structure PricingStrategy where
  regular_price : ℚ
  fifth_bundle_price : ℚ
  complimentary_snacks : ℕ

/-- Calculates the revenue from selling 5 bundles --/
def revenue_from_5_bundles (p : PricingStrategy) : ℚ :=
  4 * p.regular_price + p.fifth_bundle_price

/-- Theorem stating the minimum price for a bundle --/
theorem minimum_bundle_price (b : Bundle) (p : PricingStrategy) :
  b.water_bottle_cost = 0.5 ∧
  b.fruit_cost = 0.25 ∧
  b.snack_cost = 1 ∧
  b.water_bottles = 1 ∧
  b.fruits = 2 ∧
  b.snacks = 3 ∧
  p.fifth_bundle_price = 2 ∧
  p.complimentary_snacks = 1 →
  p.regular_price ≥ 4.75 ↔
  revenue_from_5_bundles p ≥ 5 * bundle_cost b + p.complimentary_snacks * b.snack_cost :=
by sorry

end NUMINAMATH_CALUDE_minimum_bundle_price_l3586_358632


namespace NUMINAMATH_CALUDE_negation_equivalence_l3586_358602

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x ∈ Set.Ici (0 : ℝ) ∧ x^3 + x < 0) ↔
  (∀ x : ℝ, x ∈ Set.Ici (0 : ℝ) → x^3 + x ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3586_358602


namespace NUMINAMATH_CALUDE_simplify_expression_l3586_358681

theorem simplify_expression (x y : ℝ) (hx : x ≠ 0) : 
  8 * x^3 * y / (2 * x)^2 = 2 * x * y :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l3586_358681


namespace NUMINAMATH_CALUDE_decimal_to_fraction_sum_l3586_358633

theorem decimal_to_fraction_sum (p q : ℕ+) : 
  (p : ℚ) / q = 504/1000 → 
  (∀ (a b : ℕ+), (a : ℚ) / b = p / q → a ≤ p ∧ b ≤ q) → 
  (p : ℕ) + q = 188 := by
sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_sum_l3586_358633


namespace NUMINAMATH_CALUDE_number_times_five_equals_hundred_l3586_358674

theorem number_times_five_equals_hundred :
  ∃ x : ℝ, 5 * x = 100 ∧ x = 20 := by
  sorry

end NUMINAMATH_CALUDE_number_times_five_equals_hundred_l3586_358674


namespace NUMINAMATH_CALUDE_candy_consumption_theorem_l3586_358690

/-- Represents the number of candies eaten by each person -/
structure CandyConsumption where
  andrey : ℕ
  boris : ℕ
  denis : ℕ

/-- Represents the relative eating rates of the three people -/
structure EatingRates where
  andrey_boris : ℚ  -- Ratio of Andrey's rate to Boris's rate
  andrey_denis : ℚ  -- Ratio of Andrey's rate to Denis's rate

theorem candy_consumption_theorem (rates : EatingRates) (total : ℕ) : 
  rates.andrey_boris = 4/3 → 
  rates.andrey_denis = 6/7 → 
  total = 70 → 
  ∃ (consumption : CandyConsumption), 
    consumption.andrey = 24 ∧ 
    consumption.boris = 18 ∧ 
    consumption.denis = 28 ∧
    consumption.andrey + consumption.boris + consumption.denis = total :=
by sorry

end NUMINAMATH_CALUDE_candy_consumption_theorem_l3586_358690


namespace NUMINAMATH_CALUDE_completing_square_transformation_l3586_358692

theorem completing_square_transformation (x : ℝ) :
  (x^2 - 2*x - 5 = 0) ↔ ((x - 1)^2 = 6) :=
by sorry

end NUMINAMATH_CALUDE_completing_square_transformation_l3586_358692


namespace NUMINAMATH_CALUDE_divisible_by_three_or_six_percentage_l3586_358678

theorem divisible_by_three_or_six_percentage (n : Nat) : 
  n = 200 → 
  (((Finset.filter (fun x => x % 3 = 0 ∨ x % 6 = 0) (Finset.range (n + 1))).card : ℚ) / n) * 100 = 33 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_three_or_six_percentage_l3586_358678


namespace NUMINAMATH_CALUDE_gcd_factorial_problem_l3586_358657

theorem gcd_factorial_problem : Nat.gcd (Nat.factorial 6) ((Nat.factorial 12) / (Nat.factorial 4)) = 240 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_problem_l3586_358657


namespace NUMINAMATH_CALUDE_vertex_of_quadratic_l3586_358677

/-- The quadratic function f(x) = (x-3)^2 + 1 -/
def f (x : ℝ) : ℝ := (x - 3)^2 + 1

/-- The vertex of the quadratic function f -/
def vertex : ℝ × ℝ := (3, 1)

/-- Theorem: The vertex of the quadratic function f(x) = (x-3)^2 + 1 is at the point (3,1) -/
theorem vertex_of_quadratic :
  (∀ x : ℝ, f x ≥ f (vertex.1)) ∧ f (vertex.1) = vertex.2 :=
sorry

end NUMINAMATH_CALUDE_vertex_of_quadratic_l3586_358677


namespace NUMINAMATH_CALUDE_debate_team_arrangements_l3586_358607

-- Define the number of students
def total_students : ℕ := 6

-- Define the number of team members
def team_size : ℕ := 4

-- Define the number of positions where student A can be placed
def positions_for_A : ℕ := 3

-- Define the number of remaining students after A is placed
def remaining_students : ℕ := total_students - 1

-- Define the number of remaining positions after A is placed
def remaining_positions : ℕ := team_size - 1

-- Theorem statement
theorem debate_team_arrangements :
  (positions_for_A * (remaining_students.factorial / (remaining_students - remaining_positions).factorial)) = 180 := by
  sorry

end NUMINAMATH_CALUDE_debate_team_arrangements_l3586_358607


namespace NUMINAMATH_CALUDE_only_one_and_four_are_propositions_l3586_358659

-- Define a type for the statements
inductive Statement
  | EmptySetSubset
  | GreaterThanImplication
  | IsThreeGreaterThanOne
  | NonIntersectingLinesParallel

-- Define a function to check if a statement is a proposition
def isProposition (s : Statement) : Prop :=
  match s with
  | Statement.EmptySetSubset => True
  | Statement.GreaterThanImplication => False
  | Statement.IsThreeGreaterThanOne => False
  | Statement.NonIntersectingLinesParallel => True

-- Theorem stating that only statements ① and ④ are propositions
theorem only_one_and_four_are_propositions :
  (∀ s : Statement, isProposition s ↔ (s = Statement.EmptySetSubset ∨ s = Statement.NonIntersectingLinesParallel)) :=
by sorry

end NUMINAMATH_CALUDE_only_one_and_four_are_propositions_l3586_358659


namespace NUMINAMATH_CALUDE_original_number_is_fifteen_l3586_358651

theorem original_number_is_fifteen : 
  ∃ x : ℝ, 3 * (2 * x + 5) = 105 ∧ x = 15 := by
  sorry

end NUMINAMATH_CALUDE_original_number_is_fifteen_l3586_358651


namespace NUMINAMATH_CALUDE_cost_of_traveling_specific_roads_l3586_358670

/-- Calculates the cost of traveling two intersecting roads on a rectangular lawn. -/
def cost_of_traveling_roads (lawn_length lawn_width road_width cost_per_sqm : ℝ) : ℝ :=
  let road1_area := road_width * lawn_width
  let road2_area := road_width * lawn_length
  let intersection_area := road_width * road_width
  let total_road_area := road1_area + road2_area - intersection_area
  total_road_area * cost_per_sqm

/-- Theorem stating the cost of traveling two intersecting roads on a specific rectangular lawn. -/
theorem cost_of_traveling_specific_roads : 
  cost_of_traveling_roads 80 50 10 3 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_traveling_specific_roads_l3586_358670


namespace NUMINAMATH_CALUDE_smaller_fraction_l3586_358650

theorem smaller_fraction (x y : ℝ) (sum_eq : x + y = 5/6) (prod_eq : x * y = 1/8) :
  min x y = (5 - Real.sqrt 7) / 12 := by sorry

end NUMINAMATH_CALUDE_smaller_fraction_l3586_358650


namespace NUMINAMATH_CALUDE_speed_above_limit_l3586_358697

/-- Proves that given a travel distance of 150 miles, a travel time of 2 hours,
    and a speed limit of 60 mph, the difference between the average speed
    and the speed limit is 15 mph. -/
theorem speed_above_limit (distance : ℝ) (time : ℝ) (speed_limit : ℝ) :
  distance = 150 ∧ time = 2 ∧ speed_limit = 60 →
  distance / time - speed_limit = 15 := by
  sorry

end NUMINAMATH_CALUDE_speed_above_limit_l3586_358697


namespace NUMINAMATH_CALUDE_complex_exponential_sum_l3586_358626

theorem complex_exponential_sum (θ φ : ℝ) :
  Complex.exp (Complex.I * θ) + Complex.exp (Complex.I * φ) = (1/2 : ℂ) + (1/3 : ℂ) * Complex.I →
  Complex.exp (-Complex.I * θ) + Complex.exp (-Complex.I * φ) = (1/2 : ℂ) - (1/3 : ℂ) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_exponential_sum_l3586_358626


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3586_358624

theorem quadratic_inequality_solution_set :
  ∀ x : ℝ, 3 * x^2 - 5 * x - 2 < 0 ↔ x ∈ Set.Ioo (-1/3 : ℝ) 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3586_358624


namespace NUMINAMATH_CALUDE_geometric_arithmetic_progression_sum_l3586_358642

theorem geometric_arithmetic_progression_sum : 
  ∃ (a b : ℝ), 
    3 < a ∧ a < b ∧ b < 9 ∧ 
    (∃ (r : ℝ), r > 0 ∧ a = 3 * r ∧ b = 3 * r^2) ∧ 
    (∃ (d : ℝ), b = a + d ∧ 9 = b + d) ∧ 
    a + b = 11.25 := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_progression_sum_l3586_358642


namespace NUMINAMATH_CALUDE_multiply_25_26_8_l3586_358600

theorem multiply_25_26_8 : 25 * 26 * 8 = 5200 := by
  sorry

end NUMINAMATH_CALUDE_multiply_25_26_8_l3586_358600


namespace NUMINAMATH_CALUDE_temperature_proof_l3586_358675

-- Define the temperatures for each day
def monday : ℝ := sorry
def tuesday : ℝ := sorry
def wednesday : ℝ := sorry
def thursday : ℝ := sorry
def friday : ℝ := 31

-- Define the conditions
theorem temperature_proof :
  (monday + tuesday + wednesday + thursday) / 4 = 48 →
  (tuesday + wednesday + thursday + friday) / 4 = 46 →
  friday = 31 →
  monday = 39 :=
by sorry

end NUMINAMATH_CALUDE_temperature_proof_l3586_358675


namespace NUMINAMATH_CALUDE_cos_alpha_value_l3586_358680

theorem cos_alpha_value (α : ℝ) (h1 : α ∈ Set.Ioo 0 (Real.pi / 2)) 
  (h2 : Real.cos (α + Real.pi / 3) = -2/3) : 
  Real.cos α = (Real.sqrt 15 - 2) / 6 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l3586_358680


namespace NUMINAMATH_CALUDE_hospital_workers_count_l3586_358627

/-- The number of other workers at the hospital -/
def num_other_workers : ℕ := 2

/-- The total number of workers at the hospital -/
def total_workers : ℕ := num_other_workers + 2

/-- The probability of selecting both John and David when choosing 2 workers randomly -/
def prob_select_john_and_david : ℚ := 1 / 6

theorem hospital_workers_count :
  (prob_select_john_and_david = 1 / (total_workers.choose 2)) →
  num_other_workers = 2 := by
sorry

#eval num_other_workers

end NUMINAMATH_CALUDE_hospital_workers_count_l3586_358627


namespace NUMINAMATH_CALUDE_writable_13121_not_writable_12131_l3586_358679

/-- A number that can be written on the blackboard -/
def Writable (n : ℕ) : Prop :=
  ∃ x y : ℕ, n + 1 = 2^x * 3^y

/-- The rule for writing new numbers on the blackboard -/
axiom write_rule {a b : ℕ} (ha : Writable a) (hb : Writable b) : Writable (a * b + a + b)

/-- 1 is initially on the blackboard -/
axiom writable_one : Writable 1

/-- 2 is initially on the blackboard -/
axiom writable_two : Writable 2

/-- Theorem: 13121 can be written on the blackboard -/
theorem writable_13121 : Writable 13121 :=
  sorry

/-- Theorem: 12131 cannot be written on the blackboard -/
theorem not_writable_12131 : ¬ Writable 12131 :=
  sorry

end NUMINAMATH_CALUDE_writable_13121_not_writable_12131_l3586_358679


namespace NUMINAMATH_CALUDE_equation_solution_l3586_358609

theorem equation_solution : 
  ∃! x : ℝ, (3*x - 2 ≥ 0) ∧ (Real.sqrt (3*x - 2) + 9 / Real.sqrt (3*x - 2) = 6) ∧ (x = 11/3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3586_358609


namespace NUMINAMATH_CALUDE_rectangle_area_theorem_l3586_358660

theorem rectangle_area_theorem : ∃ (x y : ℝ), 
  (x + 3) * (y - 1) = x * y ∧ 
  (x - 3) * (y + 2) = x * y ∧ 
  x * y = 36 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_theorem_l3586_358660


namespace NUMINAMATH_CALUDE_additional_people_calculation_l3586_358630

/-- Represents Carl's open house scenario -/
structure OpenHouse where
  confirmed_attendees : ℕ
  extravagant_bags : ℕ
  initial_average_bags : ℕ
  additional_bags_needed : ℕ

/-- Calculates the number of additional people Carl hopes will show up -/
def additional_people (oh : OpenHouse) : ℕ :=
  (oh.extravagant_bags + oh.initial_average_bags + oh.additional_bags_needed) - oh.confirmed_attendees

/-- Theorem stating that the number of additional people Carl hopes will show up
    is equal to the total number of gift bags minus the number of confirmed attendees -/
theorem additional_people_calculation (oh : OpenHouse) :
  additional_people oh = (oh.extravagant_bags + oh.initial_average_bags + oh.additional_bags_needed) - oh.confirmed_attendees :=
by
  sorry

#eval additional_people {
  confirmed_attendees := 50,
  extravagant_bags := 10,
  initial_average_bags := 20,
  additional_bags_needed := 60
}

end NUMINAMATH_CALUDE_additional_people_calculation_l3586_358630


namespace NUMINAMATH_CALUDE_repeating_decimal_fraction_sum_l3586_358686

theorem repeating_decimal_fraction_sum (a b : ℕ+) : 
  (a.val : ℚ) / b.val = 35 / 99 → 
  Nat.gcd a.val b.val = 1 → 
  a.val + b.val = 134 := by
sorry

end NUMINAMATH_CALUDE_repeating_decimal_fraction_sum_l3586_358686


namespace NUMINAMATH_CALUDE_cubic_sum_minus_product_l3586_358604

theorem cubic_sum_minus_product (a b c : ℝ) 
  (sum_eq : a + b + c = 12) 
  (sum_products_eq : a * b + a * c + b * c = 30) : 
  a^3 + b^3 + c^3 - 3*a*b*c = 648 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_minus_product_l3586_358604


namespace NUMINAMATH_CALUDE_set_intersection_theorem_l3586_358621

def A : Set ℝ := { x | x ≥ 1 }
def B : Set ℝ := { x | x^2 ≤ 4 }

theorem set_intersection_theorem :
  A ∩ B = { x : ℝ | 1 ≤ x ∧ x ≤ 2 } := by sorry

end NUMINAMATH_CALUDE_set_intersection_theorem_l3586_358621


namespace NUMINAMATH_CALUDE_expression_simplification_l3586_358672

theorem expression_simplification (a b : ℝ) (h1 : a = 1) (h2 : b = 2) :
  (a - b)^2 - a*(a - b) + (a + b)*(a - b) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3586_358672


namespace NUMINAMATH_CALUDE_system_solution_l3586_358619

theorem system_solution (u v : ℚ) 
  (eq1 : 3 * u - 4 * v = -14)
  (eq2 : 6 * u + 5 * v = 7) :
  2 * u - v = -63/13 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3586_358619


namespace NUMINAMATH_CALUDE_sum_of_r_p_x_is_negative_eleven_l3586_358617

def p (x : ℝ) : ℝ := |x| - 2

def r (x : ℝ) : ℝ := -|p x - 1|

def x_values : List ℝ := [-4, -3, -2, -1, 0, 1, 2, 3, 4]

theorem sum_of_r_p_x_is_negative_eleven :
  (x_values.map (λ x => r (p x))).sum = -11 := by sorry

end NUMINAMATH_CALUDE_sum_of_r_p_x_is_negative_eleven_l3586_358617


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l3586_358644

theorem polynomial_coefficient_sum : 
  ∀ A B C D : ℝ, 
  (∀ x : ℝ, (x + 3) * (4 * x^2 - 2 * x + 7) = A * x^3 + B * x^2 + C * x + D) → 
  A + B + C + D = 36 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l3586_358644


namespace NUMINAMATH_CALUDE_expression_evaluation_l3586_358693

theorem expression_evaluation :
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) * Real.sqrt (3^2 + 1^2) = 3280 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3586_358693


namespace NUMINAMATH_CALUDE_minimum_value_theorem_l3586_358610

theorem minimum_value_theorem (m n : ℝ) (a : ℝ) : 
  m > 0 → n > 0 → 
  (∃ x y : ℝ, y = a^(x+2) - 2 ∧ y = -n/m * x - 1/m) →
  1/m + 1/n ≥ 3 + 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_minimum_value_theorem_l3586_358610


namespace NUMINAMATH_CALUDE_no_primes_in_range_l3586_358695

theorem no_primes_in_range (n : ℕ) (h : n > 2) :
  ∀ k ∈ Set.Icc (n! + 3) (n! + 2*n), ¬ Nat.Prime k := by
  sorry

end NUMINAMATH_CALUDE_no_primes_in_range_l3586_358695


namespace NUMINAMATH_CALUDE_system_solution_correct_l3586_358664

theorem system_solution_correct (x y : ℝ) : 
  x = 2 ∧ y = 0 → (x - 2*y = 2 ∧ 2*x + y = 4) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_correct_l3586_358664


namespace NUMINAMATH_CALUDE_sector_central_angle_l3586_358636

/-- Given a circular sector with radius 2 and area 8, 
    the radian measure of its central angle is 4. -/
theorem sector_central_angle (r : ℝ) (area : ℝ) (angle : ℝ) : 
  r = 2 → area = 8 → area = (1/2) * r^2 * angle → angle = 4 := by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l3586_358636


namespace NUMINAMATH_CALUDE_base_number_proof_l3586_358648

theorem base_number_proof (x n : ℕ) (h1 : 4 * x^(2*n) = 4^26) (h2 : n = 25) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_base_number_proof_l3586_358648


namespace NUMINAMATH_CALUDE_no_real_solutions_l3586_358601

theorem no_real_solutions :
  ¬∃ (x : ℝ), x ≠ -9 ∧ (8*x^2 + 90*x + 2) / (3*x + 27) = 4*x + 2 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l3586_358601


namespace NUMINAMATH_CALUDE_quadratic_root_equation_l3586_358618

theorem quadratic_root_equation (x : ℝ) : 
  (∃ r : ℝ, x = (2 + r * Real.sqrt (4 - 4 * 3 * (-1))) / (2 * 3) ∧ r^2 = 1) →
  (3 * x^2 - 2 * x - 1 = 0) := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_equation_l3586_358618


namespace NUMINAMATH_CALUDE_coefficient_a2_equals_56_l3586_358638

/-- Given a polynomial equality, prove that the coefficient a₂ equals 56 -/
theorem coefficient_a2_equals_56 
  (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) : 
  (∀ x : ℝ, 1 + x + x^2 + x^3 + x^4 + x^5 + x^6 + x^7 
    = a₀ + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + a₅*(x-1)^5 + a₆*(x-1)^6 + a₇*(x-1)^7) 
  → a₂ = 56 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_a2_equals_56_l3586_358638


namespace NUMINAMATH_CALUDE_square_perimeter_from_area_l3586_358649

theorem square_perimeter_from_area (area : ℝ) (side : ℝ) (perimeter : ℝ) : 
  area = 400 → 
  area = side ^ 2 → 
  perimeter = 4 * side → 
  perimeter = 80 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_from_area_l3586_358649


namespace NUMINAMATH_CALUDE_family_gathering_handshakes_l3586_358676

/-- The number of handshakes at a family gathering --/
def total_handshakes (twin_sets : ℕ) (triplet_sets : ℕ) : ℕ :=
  let twin_count := twin_sets * 2
  let triplet_count := triplet_sets * 3
  let twin_handshakes := (twin_count * (twin_count - 2)) / 2
  let triplet_handshakes := (triplet_count * (triplet_count - 3)) / 2
  let twin_triplet_handshakes := twin_count * (triplet_count / 3) + triplet_count * (twin_count / 4)
  twin_handshakes + triplet_handshakes + twin_triplet_handshakes

/-- Theorem stating the total number of handshakes at the family gathering --/
theorem family_gathering_handshakes :
  total_handshakes 10 7 = 614 := by
  sorry

end NUMINAMATH_CALUDE_family_gathering_handshakes_l3586_358676


namespace NUMINAMATH_CALUDE_village_panic_percentage_l3586_358623

theorem village_panic_percentage (original : ℕ) (final : ℕ) : original = 7800 → final = 5265 → 
  (((original - original / 10) - final) / (original - original / 10) : ℚ) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_village_panic_percentage_l3586_358623


namespace NUMINAMATH_CALUDE_money_division_l3586_358640

/-- Proof that the total sum of money is $320 given the specified conditions -/
theorem money_division (a b c d : ℝ) : 
  (∀ (x : ℝ), b = 0.75 * x → c = 0.5 * x → d = 0.25 * x → a = x) →
  c = 64 →
  a + b + c + d = 320 := by
  sorry

end NUMINAMATH_CALUDE_money_division_l3586_358640


namespace NUMINAMATH_CALUDE_representatives_formula_l3586_358639

/-- Represents the number of representatives for a given number of students -/
def num_representatives (x : ℕ) : ℕ :=
  if x % 10 > 6 then
    x / 10 + 1
  else
    x / 10

/-- The greatest integer function -/
def floor (r : ℚ) : ℤ :=
  ⌊r⌋

theorem representatives_formula (x : ℕ) :
  (num_representatives x : ℤ) = floor ((x + 3 : ℚ) / 10) :=
sorry

end NUMINAMATH_CALUDE_representatives_formula_l3586_358639


namespace NUMINAMATH_CALUDE_microscope_magnification_factor_l3586_358656

/-- The magnification factor of an electron microscope, given the magnified image diameter and actual tissue diameter. -/
theorem microscope_magnification_factor 
  (magnified_diameter : ℝ) 
  (actual_diameter : ℝ) 
  (h1 : magnified_diameter = 2) 
  (h2 : actual_diameter = 0.002) : 
  magnified_diameter / actual_diameter = 1000 := by
sorry

end NUMINAMATH_CALUDE_microscope_magnification_factor_l3586_358656


namespace NUMINAMATH_CALUDE_rectangular_plot_length_breadth_difference_l3586_358628

theorem rectangular_plot_length_breadth_difference 
  (area length breadth : ℝ)
  (h1 : area = length * breadth)
  (h2 : area = 15 * breadth)
  (h3 : breadth = 5) :
  length - breadth = 10 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_length_breadth_difference_l3586_358628


namespace NUMINAMATH_CALUDE_no_positive_integer_pairs_l3586_358687

theorem no_positive_integer_pairs : 
  ¬∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x > y ∧ (x^2 : ℝ) + y^2 = x^4 := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_pairs_l3586_358687


namespace NUMINAMATH_CALUDE_two_and_three_digit_problem_l3586_358655

theorem two_and_three_digit_problem :
  ∃ (x y : ℕ), 
    10 ≤ x ∧ x < 100 ∧
    100 ≤ y ∧ y < 1000 ∧
    1000 * x + y = 7 * x * y ∧
    x + y = 1074 := by
  sorry

end NUMINAMATH_CALUDE_two_and_three_digit_problem_l3586_358655


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_sqrt_16_l3586_358663

theorem arithmetic_square_root_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_sqrt_16_l3586_358663


namespace NUMINAMATH_CALUDE_factorization_of_18x_squared_minus_8_l3586_358647

theorem factorization_of_18x_squared_minus_8 (x : ℝ) : 18 * x^2 - 8 = 2 * (3*x + 2) * (3*x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_18x_squared_minus_8_l3586_358647


namespace NUMINAMATH_CALUDE_brady_current_yards_l3586_358605

/-- The passing yards record in a season -/
def record : ℕ := 5999

/-- The number of games left in the season -/
def games_left : ℕ := 6

/-- The average passing yards needed per game to beat the record -/
def average_needed : ℕ := 300

/-- Tom Brady's current passing yards -/
def current_yards : ℕ := 4200

theorem brady_current_yards : 
  current_yards = record + 1 - (games_left * average_needed) :=
sorry

end NUMINAMATH_CALUDE_brady_current_yards_l3586_358605


namespace NUMINAMATH_CALUDE_tan_sum_of_roots_l3586_358691

theorem tan_sum_of_roots (α β : Real) : 
  (∃ x y : Real, x^2 + 6*x + 7 = 0 ∧ y^2 + 6*y + 7 = 0 ∧ x = Real.tan α ∧ y = Real.tan β) → 
  Real.tan (α + β) = 1 := by
sorry

end NUMINAMATH_CALUDE_tan_sum_of_roots_l3586_358691


namespace NUMINAMATH_CALUDE_tile_border_ratio_l3586_358620

theorem tile_border_ratio (t w : ℝ) (h : t > 0) (h' : w > 0) : 
  (900 * t^2) / ((30 * t + 30 * w)^2) = 81/100 → w/t = 1/9 := by
sorry

end NUMINAMATH_CALUDE_tile_border_ratio_l3586_358620


namespace NUMINAMATH_CALUDE_savings_percentage_l3586_358611

/-- Proves that given the conditions of the problem, the percentage of income saved in the first year is 20% --/
theorem savings_percentage (income : ℝ) (savings : ℝ) 
  (h1 : savings > 0)
  (h2 : income > savings)
  (h3 : (income - savings) + (1.2 * income - 2 * savings) = 2 * (income - savings)) :
  savings / income = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_savings_percentage_l3586_358611


namespace NUMINAMATH_CALUDE_frequency_in_interval_l3586_358668

def sample_capacity : ℕ := 100

def group_frequencies : List ℕ := [12, 13, 24, 15, 16, 13, 7]

def interval_sum : ℕ := 12 + 13 + 24 + 15

theorem frequency_in_interval :
  (interval_sum : ℚ) / sample_capacity = 0.64 := by sorry

end NUMINAMATH_CALUDE_frequency_in_interval_l3586_358668


namespace NUMINAMATH_CALUDE_balance_after_six_months_l3586_358658

/-- Calculates the balance after two quarters of compound interest -/
def balance_after_two_quarters (initial_deposit : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  let balance_after_first_quarter := initial_deposit * (1 + rate1)
  balance_after_first_quarter * (1 + rate2)

/-- Theorem stating the balance after two quarters with given initial deposit and interest rates -/
theorem balance_after_six_months 
  (initial_deposit : ℝ) 
  (rate1 : ℝ) 
  (rate2 : ℝ) 
  (h1 : initial_deposit = 5000)
  (h2 : rate1 = 0.07)
  (h3 : rate2 = 0.085) :
  balance_after_two_quarters initial_deposit rate1 rate2 = 5804.25 := by
  sorry

#eval balance_after_two_quarters 5000 0.07 0.085

end NUMINAMATH_CALUDE_balance_after_six_months_l3586_358658


namespace NUMINAMATH_CALUDE_hyperbola_other_asymptote_l3586_358612

/-- A hyperbola with given properties -/
structure Hyperbola where
  /-- One asymptote of the hyperbola -/
  asymptote1 : ℝ → ℝ
  /-- x-coordinate of the foci -/
  foci_x : ℝ
  /-- Condition that the first asymptote is y = 2x -/
  h_asymptote1 : ∀ x, asymptote1 x = 2 * x
  /-- Condition that the foci x-coordinate is 4 -/
  h_foci_x : foci_x = 4

/-- The other asymptote of the hyperbola -/
def other_asymptote (h : Hyperbola) : ℝ → ℝ :=
  fun x ↦ -2 * x + 16

/-- Theorem stating the equation of the other asymptote -/
theorem hyperbola_other_asymptote (h : Hyperbola) :
  other_asymptote h = fun x ↦ -2 * x + 16 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_other_asymptote_l3586_358612


namespace NUMINAMATH_CALUDE_expression_simplification_l3586_358637

theorem expression_simplification :
  (((3 + 4 + 6 + 7) / 3) + ((4 * 3 + 5 - 2) / 4)) = 125 / 12 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3586_358637


namespace NUMINAMATH_CALUDE_b_bounded_a_value_l3586_358689

/-- A quadratic function f(x) = ax^2 + bx + c with certain properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  bounded : ∀ x : ℝ, x ∈ Set.Icc (-1) 1 → |a * x^2 + b * x + c| ≤ 1

/-- The coefficient b of a QuadraticFunction is bounded by 1 -/
theorem b_bounded (f : QuadraticFunction) : |f.b| ≤ 1 := by sorry

/-- If f(0) = -1 and f(1) = 1, then a = 2 -/
theorem a_value (f : QuadraticFunction) 
  (h0 : f.c = -1) 
  (h1 : f.a + f.b + f.c = 1) : 
  f.a = 2 := by sorry

end NUMINAMATH_CALUDE_b_bounded_a_value_l3586_358689


namespace NUMINAMATH_CALUDE_arithmetic_simplification_l3586_358671

theorem arithmetic_simplification :
  -3 + (-9) + 10 - (-18) = 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_simplification_l3586_358671


namespace NUMINAMATH_CALUDE_molecular_weight_8_moles_Al2O3_l3586_358652

/-- The atomic weight of Aluminum in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of Aluminum atoms in one molecule of Al2O3 -/
def num_Al_atoms : ℕ := 2

/-- The number of Oxygen atoms in one molecule of Al2O3 -/
def num_O_atoms : ℕ := 3

/-- The number of moles of Al2O3 -/
def num_moles : ℕ := 8

/-- The molecular weight of Al2O3 in g/mol -/
def molecular_weight_Al2O3 : ℝ := 
  num_Al_atoms * atomic_weight_Al + num_O_atoms * atomic_weight_O

theorem molecular_weight_8_moles_Al2O3 : 
  num_moles * molecular_weight_Al2O3 = 815.68 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_8_moles_Al2O3_l3586_358652


namespace NUMINAMATH_CALUDE_complex_multiplication_l3586_358673

theorem complex_multiplication (i : ℂ) : i * i = -1 → i * (i + 1) = -1 + i := by sorry

end NUMINAMATH_CALUDE_complex_multiplication_l3586_358673


namespace NUMINAMATH_CALUDE_probability_female_wears_glasses_l3586_358613

/-- Given a class with female and male students, some wearing glasses, prove the probability of a randomly selected female student wearing glasses. -/
theorem probability_female_wears_glasses 
  (total_female : ℕ) 
  (total_male : ℕ) 
  (female_no_glasses : ℕ) 
  (male_with_glasses : ℕ) 
  (h1 : total_female = 18) 
  (h2 : total_male = 20) 
  (h3 : female_no_glasses = 8) 
  (h4 : male_with_glasses = 11) : 
  (total_female - female_no_glasses : ℚ) / total_female = 5 / 9 := by
  sorry

#check probability_female_wears_glasses

end NUMINAMATH_CALUDE_probability_female_wears_glasses_l3586_358613


namespace NUMINAMATH_CALUDE_average_age_after_leaving_l3586_358616

def initial_people : ℕ := 8
def initial_average_age : ℚ := 35
def leaving_person_age : ℕ := 22
def remaining_people : ℕ := initial_people - 1

theorem average_age_after_leaving :
  (initial_people * initial_average_age - leaving_person_age) / remaining_people = 258 / 7 := by
  sorry

end NUMINAMATH_CALUDE_average_age_after_leaving_l3586_358616


namespace NUMINAMATH_CALUDE_nabla_calculation_l3586_358603

def nabla (a b : ℕ) : ℕ := 3 + a^b

theorem nabla_calculation : nabla (nabla 2 3) 2 = 124 := by
  sorry

end NUMINAMATH_CALUDE_nabla_calculation_l3586_358603


namespace NUMINAMATH_CALUDE_apples_per_pie_l3586_358654

theorem apples_per_pie (total_apples : ℕ) (handed_out : ℕ) (num_pies : ℕ) :
  total_apples = 51 →
  handed_out = 41 →
  num_pies = 2 →
  (total_apples - handed_out) / num_pies = 5 := by
sorry

end NUMINAMATH_CALUDE_apples_per_pie_l3586_358654


namespace NUMINAMATH_CALUDE_jessie_weight_loss_l3586_358683

/-- Jessie's weight loss problem -/
theorem jessie_weight_loss (current_weight lost_weight : ℕ) 
  (h1 : current_weight = 34)
  (h2 : lost_weight = 35) : 
  current_weight + lost_weight = 69 := by
  sorry

end NUMINAMATH_CALUDE_jessie_weight_loss_l3586_358683


namespace NUMINAMATH_CALUDE_shopping_tax_percentage_l3586_358661

theorem shopping_tax_percentage (total : ℝ) (h_total_pos : total > 0) : 
  let clothing_percent : ℝ := 0.50
  let food_percent : ℝ := 0.20
  let other_percent : ℝ := 0.30
  let clothing_tax_rate : ℝ := 0.04
  let food_tax_rate : ℝ := 0
  let other_tax_rate : ℝ := 0.08
  let clothing_amount := clothing_percent * total
  let food_amount := food_percent * total
  let other_amount := other_percent * total
  let clothing_tax := clothing_tax_rate * clothing_amount
  let food_tax := food_tax_rate * food_amount
  let other_tax := other_tax_rate * other_amount
  let total_tax := clothing_tax + food_tax + other_tax
  total_tax / total = 0.0440 :=
by sorry

end NUMINAMATH_CALUDE_shopping_tax_percentage_l3586_358661


namespace NUMINAMATH_CALUDE_trapezoid_semicircle_perimeter_l3586_358631

/-- The perimeter of a shape composed of a trapezoid and a semicircle with given dimensions -/
theorem trapezoid_semicircle_perimeter 
  (trapezoid_short_base : ℝ) 
  (trapezoid_long_base : ℝ) 
  (trapezoid_side1 : ℝ) 
  (trapezoid_side2 : ℝ) 
  (semicircle_radius : ℝ) 
  (h1 : trapezoid_short_base = 5) 
  (h2 : trapezoid_long_base = 7) 
  (h3 : trapezoid_side1 = 3) 
  (h4 : trapezoid_side2 = 4) 
  (h5 : semicircle_radius = 3.1) 
  (h6 : trapezoid_long_base = 2 * semicircle_radius) : 
  trapezoid_short_base + trapezoid_side1 + trapezoid_side2 + π * semicircle_radius = 12 + π * 3.1 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_semicircle_perimeter_l3586_358631


namespace NUMINAMATH_CALUDE_digit_1234_is_4_l3586_358662

/-- The decimal number constructed by concatenating integers from 1 to 499 -/
def x : ℚ :=
  sorry

/-- The nth digit of a rational number -/
def nthDigit (q : ℚ) (n : ℕ) : ℕ :=
  sorry

theorem digit_1234_is_4 : nthDigit x 1234 = 4 := by
  sorry

end NUMINAMATH_CALUDE_digit_1234_is_4_l3586_358662


namespace NUMINAMATH_CALUDE_sum_remainder_l3586_358665

theorem sum_remainder (m : ℤ) : ((9 - m) + (m + 5)) % 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_l3586_358665


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3586_358694

theorem sqrt_equation_solution (x : ℝ) :
  (x > 2) → (Real.sqrt (7 * x) / Real.sqrt (4 * (x - 2)) = 3) → (x = 72 / 29) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3586_358694


namespace NUMINAMATH_CALUDE_derivative_of_f_l3586_358682

-- Define the function f
def f (x : ℝ) : ℝ := (x + 1) * (x^2 - x + 1)

-- State the theorem
theorem derivative_of_f : 
  deriv f = λ x => 3 * x^2 := by sorry

end NUMINAMATH_CALUDE_derivative_of_f_l3586_358682


namespace NUMINAMATH_CALUDE_digit_150_of_11_13_l3586_358608

/-- The decimal representation of 11/13 has a repeating sequence of 6 digits. -/
def decimal_rep_11_13 : List Nat := [8, 4, 6, 1, 5, 3]

/-- The 150th digit after the decimal point in the decimal representation of 11/13 is 3. -/
theorem digit_150_of_11_13 : 
  decimal_rep_11_13[150 % decimal_rep_11_13.length] = 3 := by sorry

end NUMINAMATH_CALUDE_digit_150_of_11_13_l3586_358608


namespace NUMINAMATH_CALUDE_max_wooden_pencils_l3586_358606

theorem max_wooden_pencils :
  ∀ (m w : ℕ),
  m + w = 72 →
  ∃ (p : ℕ), Nat.Prime p ∧ m = w + p →
  w ≤ 35 :=
by sorry

end NUMINAMATH_CALUDE_max_wooden_pencils_l3586_358606


namespace NUMINAMATH_CALUDE_car_speed_l3586_358614

/-- Given a car that travels 275 miles in 5 hours, its speed is 55 miles per hour. -/
theorem car_speed (distance : ℝ) (time : ℝ) (speed : ℝ) 
  (h1 : distance = 275) 
  (h2 : time = 5) 
  (h3 : speed = distance / time) : 
  speed = 55 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_l3586_358614


namespace NUMINAMATH_CALUDE_derivative_even_implies_b_zero_l3586_358643

/-- A cubic function with a constant term of 2 -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + 2

/-- The derivative of f -/
def f' (a b c : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

/-- A function is even if f(x) = f(-x) for all x -/
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)

theorem derivative_even_implies_b_zero (a b c : ℝ) :
  is_even (f' a b c) → b = 0 := by sorry

end NUMINAMATH_CALUDE_derivative_even_implies_b_zero_l3586_358643


namespace NUMINAMATH_CALUDE_max_value_z_l3586_358641

theorem max_value_z (x y : ℝ) (h1 : x - y + 1 ≤ 0) (h2 : x - 2*y ≤ 0) (h3 : x + 2*y - 2 ≤ 0) :
  ∀ z, z = x + y → z ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_z_l3586_358641


namespace NUMINAMATH_CALUDE_expression_simplification_l3586_358688

theorem expression_simplification (x y : ℝ) (hx : x = -1) (hy : y = 3) :
  5 * (3 * x^2 * y - 2 * x * y^2) - 2 * (3 * x^2 * y - 5 * x * y^2) = 27 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3586_358688


namespace NUMINAMATH_CALUDE_linear_transformation_mapping_l3586_358629

theorem linear_transformation_mapping (x : ℝ) :
  0 ≤ x ∧ x ≤ 1 → -1 ≤ 4 * x - 1 ∧ 4 * x - 1 ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_linear_transformation_mapping_l3586_358629


namespace NUMINAMATH_CALUDE_perfect_square_condition_l3586_358698

theorem perfect_square_condition (n : ℕ) : 
  (∃ (k : ℕ), 2^(n+1) * n = k^2) ↔ 
  (∃ (m : ℕ), n = 2 * m^2) ∨ 
  (∃ (k : ℕ), n = k^2 ∧ k % 2 = 1) := by
sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l3586_358698


namespace NUMINAMATH_CALUDE_square_side_length_average_l3586_358696

theorem square_side_length_average (a₁ a₂ a₃ a₄ : ℝ) 
  (h₁ : a₁ = 25) (h₂ : a₂ = 64) (h₃ : a₃ = 144) (h₄ : a₄ = 225) : 
  (Real.sqrt a₁ + Real.sqrt a₂ + Real.sqrt a₃ + Real.sqrt a₄) / 4 = 10 := by
sorry

end NUMINAMATH_CALUDE_square_side_length_average_l3586_358696


namespace NUMINAMATH_CALUDE_sum_in_base6_l3586_358685

/-- Converts a base 6 number to base 10 --/
def base6ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (6 ^ i)) 0

/-- Converts a base 10 number to base 6 --/
def base10ToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc
    else aux (m / 6) ((m % 6) :: acc)
  aux n []

/-- The main theorem --/
theorem sum_in_base6 :
  let a := base6ToBase10 [4, 3, 2, 1]  -- 1234₆
  let b := base6ToBase10 [4, 5, 6]     -- 654₆
  let c := base6ToBase10 [2, 1]        -- 12₆
  base10ToBase6 (a + b + c) = [4, 4, 3, 2] -- 2344₆
:= by sorry

end NUMINAMATH_CALUDE_sum_in_base6_l3586_358685


namespace NUMINAMATH_CALUDE_square_vertex_B_l3586_358684

/-- A square in a 2D Cartesian plane -/
structure Square where
  O : ℝ × ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Predicate to check if a point is in the fourth quadrant -/
def isInFourthQuadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

/-- Theorem: Given a square OABC with O(0,0) and A(4,3), and C in the fourth quadrant, B is at (7,-1) -/
theorem square_vertex_B (s : Square) : 
  s.O = (0, 0) → 
  s.A = (4, 3) → 
  isInFourthQuadrant s.C → 
  s.B = (7, -1) := by
  sorry


end NUMINAMATH_CALUDE_square_vertex_B_l3586_358684


namespace NUMINAMATH_CALUDE_andre_carl_speed_ratio_l3586_358646

/-- 
Given two runners, Carl and André, with the following conditions:
- Carl runs at a constant speed of x meters per second
- André runs at a constant speed of y meters per second
- André starts running 20 seconds after Carl
- André catches up to Carl after running for 10 seconds

Prove that the ratio of André's speed to Carl's speed is 3:1
-/
theorem andre_carl_speed_ratio 
  (x y : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (h_catchup : 10 * y = 30 * x) : 
  y / x = 3 := by
sorry

end NUMINAMATH_CALUDE_andre_carl_speed_ratio_l3586_358646


namespace NUMINAMATH_CALUDE_expression_undefined_at_13_l3586_358625

-- Define the numerator and denominator of the expression
def numerator (x : ℝ) : ℝ := 3 * x^3 - 5
def denominator (x : ℝ) : ℝ := x^2 - 26 * x + 169

-- Theorem stating that the expression is undefined when x = 13
theorem expression_undefined_at_13 : denominator 13 = 0 := by sorry

end NUMINAMATH_CALUDE_expression_undefined_at_13_l3586_358625


namespace NUMINAMATH_CALUDE_vet_donation_is_78_l3586_358634

/-- Represents the vet fees and adoption numbers for different animal types -/
structure AnimalAdoption where
  dog_fee : ℕ
  cat_fee : ℕ
  rabbit_fee : ℕ
  parrot_fee : ℕ
  dog_adoptions : ℕ
  cat_adoptions : ℕ
  rabbit_adoptions : ℕ
  parrot_adoptions : ℕ

/-- Calculates the total vet fees collected -/
def total_fees (a : AnimalAdoption) : ℕ :=
  a.dog_fee * a.dog_adoptions +
  a.cat_fee * a.cat_adoptions +
  a.rabbit_fee * a.rabbit_adoptions +
  a.parrot_fee * a.parrot_adoptions

/-- Calculates the amount donated by the vet -/
def vet_donation (a : AnimalAdoption) : ℕ :=
  (total_fees a + 1) / 3

/-- Theorem stating that the vet's donation is $78 given the specified conditions -/
theorem vet_donation_is_78 (a : AnimalAdoption) 
  (h1 : a.dog_fee = 15)
  (h2 : a.cat_fee = 13)
  (h3 : a.rabbit_fee = 10)
  (h4 : a.parrot_fee = 12)
  (h5 : a.dog_adoptions = 8)
  (h6 : a.cat_adoptions = 3)
  (h7 : a.rabbit_adoptions = 5)
  (h8 : a.parrot_adoptions = 2) :
  vet_donation a = 78 := by
  sorry


end NUMINAMATH_CALUDE_vet_donation_is_78_l3586_358634
