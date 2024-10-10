import Mathlib

namespace school_election_votes_l51_5137

theorem school_election_votes (total_votes : ℕ) 
  (h1 : 45 = (3 : ℕ) * total_votes / 8)
  (h2 : (1 : ℕ) * total_votes / 4 + (3 : ℕ) * total_votes / 8 ≤ total_votes) : 
  total_votes = 120 := by
sorry

end school_election_votes_l51_5137


namespace parallelogram_height_l51_5179

/-- Given a parallelogram with area 384 cm² and base 24 cm, its height is 16 cm -/
theorem parallelogram_height (area : ℝ) (base : ℝ) (height : ℝ) : 
  area = 384 ∧ base = 24 ∧ area = base * height → height = 16 := by
  sorry

end parallelogram_height_l51_5179


namespace lcm_three_integers_l51_5160

theorem lcm_three_integers (A₁ A₂ A₃ : ℤ) :
  let D := Int.gcd (A₁ * A₂) (Int.gcd (A₂ * A₃) (A₃ * A₁))
  Int.lcm A₁ (Int.lcm A₂ A₃) = (A₁ * A₂ * A₃) / D :=
by sorry

end lcm_three_integers_l51_5160


namespace functional_equation_solution_l51_5117

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x * f y / f (x * y)

/-- Theorem stating the possible forms of f -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : FunctionalEquation f) :
  (∀ x : ℝ, f x = 0) ∨ ∃ c : ℝ, c ≠ 0 ∧ ∀ x : ℝ, f x = c :=
sorry

end functional_equation_solution_l51_5117


namespace school_garbage_plan_l51_5196

/-- Represents a purchasing plan for warm reminder signs and garbage bins -/
structure PurchasePlan where
  signs : ℕ
  bins : ℕ

/-- Calculates the total cost of a purchasing plan given the prices -/
def totalCost (plan : PurchasePlan) (signPrice binPrice : ℕ) : ℕ :=
  plan.signs * signPrice + plan.bins * binPrice

theorem school_garbage_plan :
  ∃ (signPrice binPrice : ℕ) (bestPlan : PurchasePlan),
    -- Conditions
    (2 * signPrice + 3 * binPrice = 550) ∧
    (binPrice = 3 * signPrice) ∧
    (bestPlan.signs + bestPlan.bins = 100) ∧
    (bestPlan.bins ≥ 48) ∧
    (totalCost bestPlan signPrice binPrice ≤ 10000) ∧
    -- Conclusions
    (signPrice = 50) ∧
    (binPrice = 150) ∧
    (bestPlan.signs = 52) ∧
    (bestPlan.bins = 48) ∧
    (totalCost bestPlan signPrice binPrice = 9800) ∧
    (∀ (plan : PurchasePlan),
      (plan.signs + plan.bins = 100) →
      (plan.bins ≥ 48) →
      (totalCost plan signPrice binPrice ≤ 10000) →
      (totalCost plan signPrice binPrice ≥ totalCost bestPlan signPrice binPrice)) :=
by
  sorry

end school_garbage_plan_l51_5196


namespace proportional_function_ratio_l51_5139

/-- Proves that for a proportional function y = kx passing through the points (1, 3) and (a, b) where b ≠ 0, a/b = 1/3 -/
theorem proportional_function_ratio (k a b : ℝ) (h1 : b ≠ 0) (h2 : 3 = k * 1) (h3 : b = k * a) : a / b = 1 / 3 := by
  sorry

end proportional_function_ratio_l51_5139


namespace floor_abs_negative_real_l51_5191

theorem floor_abs_negative_real : ⌊|(-45.7 : ℝ)|⌋ = 45 := by sorry

end floor_abs_negative_real_l51_5191


namespace inequalities_proof_l51_5168

theorem inequalities_proof :
  (Real.log (Real.sqrt 2) < Real.sqrt 2 / 2) ∧
  (2 * Real.log (Real.sin (1/8) + Real.cos (1/8)) < 1/4) := by
  sorry

end inequalities_proof_l51_5168


namespace amanda_notebooks_problem_l51_5145

theorem amanda_notebooks_problem (initial_notebooks ordered_notebooks loss_percentage : ℕ) 
  (h1 : initial_notebooks = 65)
  (h2 : ordered_notebooks = 23)
  (h3 : loss_percentage = 15) : 
  initial_notebooks + ordered_notebooks - (((initial_notebooks + ordered_notebooks) * loss_percentage) / 100) = 75 := by
  sorry

end amanda_notebooks_problem_l51_5145


namespace profit_at_twenty_reduction_max_profit_at_fifteen_reduction_l51_5156

-- Define the profit function
def profit_function (x : ℝ) : ℝ := -2 * x^2 + 60 * x + 800

-- Theorem for part 1
theorem profit_at_twenty_reduction (x : ℝ) :
  x = 20 → profit_function x = 1200 := by sorry

-- Theorem for part 2
theorem max_profit_at_fifteen_reduction :
  ∃ (x : ℝ), x = 15 ∧ 
  profit_function x = 1250 ∧ 
  ∀ (y : ℝ), profit_function y ≤ profit_function x := by sorry

end profit_at_twenty_reduction_max_profit_at_fifteen_reduction_l51_5156


namespace max_large_planes_is_seven_l51_5105

/-- Calculates the maximum number of planes that can fit in a hangar -/
def max_planes (hangar_length : ℕ) (plane_length : ℕ) (safety_gap : ℕ) : ℕ :=
  (hangar_length) / (plane_length + safety_gap)

/-- Theorem: The maximum number of large planes in the hangar is 7 -/
theorem max_large_planes_is_seven :
  max_planes 900 110 10 = 7 := by
  sorry

#eval max_planes 900 110 10

end max_large_planes_is_seven_l51_5105


namespace sneaker_coupon_value_l51_5125

/-- Proves that the coupon value is $10 given the conditions of the sneaker purchase problem -/
theorem sneaker_coupon_value (original_price : ℝ) (membership_discount : ℝ) (final_price : ℝ)
  (h1 : original_price = 120)
  (h2 : membership_discount = 0.1)
  (h3 : final_price = 99) :
  ∃ (coupon_value : ℝ), 
    (1 - membership_discount) * (original_price - coupon_value) = final_price ∧
    coupon_value = 10 :=
by sorry

end sneaker_coupon_value_l51_5125


namespace monthly_parking_fee_l51_5115

/-- Proves that the monthly parking fee is $40 given the specified conditions -/
theorem monthly_parking_fee (weekly_fee : ℕ) (yearly_savings : ℕ) (weeks_per_year : ℕ) (months_per_year : ℕ) :
  weekly_fee = 10 →
  yearly_savings = 40 →
  weeks_per_year = 52 →
  months_per_year = 12 →
  ∃ (monthly_fee : ℕ), monthly_fee = 40 ∧ weeks_per_year * weekly_fee - months_per_year * monthly_fee = yearly_savings :=
by sorry

end monthly_parking_fee_l51_5115


namespace roots_of_f_minus_x_and_f_of_f_minus_x_l51_5135

def f (x : ℝ) : ℝ := x^2 - 2*x + 2

theorem roots_of_f_minus_x_and_f_of_f_minus_x :
  (∀ x : ℝ, f x - x = 0 ↔ x = 1 ∨ x = 2) ∧
  (∀ x : ℝ, f (f x) - x = 0 ↔ x = 1 ∨ x = 2) := by
  sorry

end roots_of_f_minus_x_and_f_of_f_minus_x_l51_5135


namespace tan_function_property_l51_5141

/-- Given a function y = a * tan(b * x) where a and b are positive constants,
    if the function passes through (π/4, 3) and has a period of 3π/2,
    then a * b = 2 * √3 -/
theorem tan_function_property (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a * Real.tan (b * (π / 4)) = 3) →
  (π / b = 3 * π / 2) →
  a * b = 2 * Real.sqrt 3 := by
  sorry

end tan_function_property_l51_5141


namespace problem_statement_l51_5187

theorem problem_statement (x y : ℝ) (h1 : x + y = 4) (h2 : x * y = -2) :
  x + (x^3 / y^2) + (y^3 / x^2) + y = 440 := by
  sorry

end problem_statement_l51_5187


namespace min_value_m_plus_2n_l51_5183

/-- The function f(x) = |x-a| where a is a real number -/
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

/-- The theorem stating the minimum value of m + 2n -/
theorem min_value_m_plus_2n (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (∀ x, f 2 x ≤ 1 ↔ 1 ≤ x ∧ x ≤ 3) →
  1/m + 1/(2*n) = 2 →
  ∀ k l, k > 0 → l > 0 → 1/k + 1/(2*l) = 2 → m + 2*n ≤ k + 2*l :=
by sorry

end min_value_m_plus_2n_l51_5183


namespace age_ratio_l51_5144

def kul_age : ℕ := 22
def saras_age : ℕ := 33

theorem age_ratio : 
  (saras_age : ℚ) / (kul_age : ℚ) = 3 / 2 := by sorry

end age_ratio_l51_5144


namespace complex_in_third_quadrant_l51_5165

def complex_number (x : ℝ) : ℂ := Complex.mk (x^2 - 6*x + 5) (x - 2)

def in_third_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im < 0

theorem complex_in_third_quadrant (x : ℝ) :
  in_third_quadrant (complex_number x) ↔ 1 < x ∧ x < 2 := by sorry

end complex_in_third_quadrant_l51_5165


namespace max_value_of_b_plus_c_l51_5166

/-- A cubic function f(x) = x³ + bx² + cx + d -/
def f (b c d : ℝ) (x : ℝ) : ℝ := x^3 + b*x^2 + c*x + d

/-- The derivative of f(x) -/
def f_deriv (b c : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*b*x + c

/-- f(x) is decreasing on the interval [-2, 2] -/
def is_decreasing_on_interval (b c d : ℝ) : Prop :=
  ∀ x ∈ Set.Icc (-2) 2, f_deriv b c x ≤ 0

theorem max_value_of_b_plus_c (b c d : ℝ) 
  (h : is_decreasing_on_interval b c d) : 
  b + c ≤ -12 := by
  sorry

end max_value_of_b_plus_c_l51_5166


namespace square_area_from_perimeter_l51_5158

/-- The area of a square with perimeter 40 feet is 100 square feet -/
theorem square_area_from_perimeter :
  ∀ (s : ℝ), s > 0 → 4 * s = 40 → s^2 = 100 := by
  sorry

end square_area_from_perimeter_l51_5158


namespace house_transactions_result_l51_5157

/-- Represents the state of cash and house ownership for both Mr. A and Mr. B -/
structure State where
  a_cash : Int
  b_cash : Int
  a_has_house : Bool

/-- Represents a transaction between Mr. A and Mr. B -/
inductive Transaction
  | sell_to_b (price : Int)
  | buy_from_b (price : Int)

def initial_state : State := {
  a_cash := 12000,
  b_cash := 13000,
  a_has_house := true
}

def apply_transaction (s : State) (t : Transaction) : State :=
  match t with
  | Transaction.sell_to_b price =>
      { a_cash := s.a_cash + price,
        b_cash := s.b_cash - price,
        a_has_house := false }
  | Transaction.buy_from_b price =>
      { a_cash := s.a_cash - price,
        b_cash := s.b_cash + price,
        a_has_house := true }

def transactions : List Transaction := [
  Transaction.sell_to_b 14000,
  Transaction.buy_from_b 11000,
  Transaction.sell_to_b 15000
]

def final_state : State :=
  transactions.foldl apply_transaction initial_state

theorem house_transactions_result :
  final_state.a_cash = 30000 ∧ final_state.b_cash = -5000 := by
  sorry

end house_transactions_result_l51_5157


namespace complement_of_M_in_U_l51_5138

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define the set M
def M : Set Nat := {1, 3, 5}

-- Theorem stating that the complement of M in U is {2, 4, 6}
theorem complement_of_M_in_U :
  (U \ M) = {2, 4, 6} := by sorry

end complement_of_M_in_U_l51_5138


namespace parabola_parameter_distance_l51_5143

/-- Parabola type representing y = ax^2 -/
structure Parabola where
  a : ℝ

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Function to calculate the distance from a point to the directrix of a parabola -/
noncomputable def distance_to_directrix (p : Parabola) (pt : Point) : ℝ :=
  if p.a > 0 then
    abs (pt.y + 1 / (4 * p.a))
  else
    abs (pt.y - 1 / (4 * p.a))

/-- Theorem stating the relationship between the parabola parameter and the distance to directrix -/
theorem parabola_parameter_distance (p : Parabola) :
  let m : Point := ⟨2, 1⟩
  distance_to_directrix p m = 2 →
  p.a = 1/4 ∨ p.a = -1/12 :=
sorry

end parabola_parameter_distance_l51_5143


namespace special_dog_food_ounces_per_pound_l51_5173

/-- Represents the number of ounces in a pound of special dog food -/
def ounces_per_pound : ℕ := 16

/-- Represents the number of days in a year -/
def days_in_year : ℕ := 365

/-- Represents the number of days the puppy eats 2 ounces per day -/
def initial_feeding_days : ℕ := 60

/-- Represents the number of ounces the puppy eats per day during the initial feeding period -/
def initial_feeding_ounces : ℕ := 2

/-- Represents the number of ounces the puppy eats per day after the initial feeding period -/
def later_feeding_ounces : ℕ := 4

/-- Represents the number of pounds in each bag of special dog food -/
def pounds_per_bag : ℕ := 5

/-- Represents the number of bags the family needs to buy -/
def bags_needed : ℕ := 17

theorem special_dog_food_ounces_per_pound :
  ounces_per_pound = 16 :=
by sorry

end special_dog_food_ounces_per_pound_l51_5173


namespace marble_fraction_after_doubling_red_l51_5109

theorem marble_fraction_after_doubling_red (total : ℚ) (h : total > 0) :
  let initial_blue := (3 / 5) * total
  let initial_red := total - initial_blue
  let new_red := 2 * initial_red
  let new_total := initial_blue + new_red
  new_red / new_total = 4 / 7 := by sorry

end marble_fraction_after_doubling_red_l51_5109


namespace x_plus_y_value_l51_5195

theorem x_plus_y_value (x y : ℝ) (hx : |x| = 5) (hy : |y| = 3) (hxy : x - y > 0) :
  x + y = 8 ∨ x + y = 2 := by
sorry

end x_plus_y_value_l51_5195


namespace negation_equivalence_l51_5140

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - x + 1 ≤ 0) ↔ (∀ x : ℝ, x^2 - x + 1 > 0) := by sorry

end negation_equivalence_l51_5140


namespace fraction_transformation_l51_5171

theorem fraction_transformation (d : ℚ) : 
  (2 : ℚ) / d ≠ 0 →
  (2 + 3 : ℚ) / (d + 3) = 1 / 3 →
  d = 12 := by
sorry

end fraction_transformation_l51_5171


namespace y_bounds_for_n_4_l51_5199

/-- The function y(t) = (n-1)t² - 10t + 10 -/
def y (n : ℕ) (t : ℝ) : ℝ := (n - 1) * t^2 - 10 * t + 10

/-- The theorem stating that for n = 4, y(t) is always between 0 and 30 for t in (0,4] -/
theorem y_bounds_for_n_4 :
  ∀ t : ℝ, t > 0 → t ≤ 4 → 0 < y 4 t ∧ y 4 t ≤ 30 := by sorry

end y_bounds_for_n_4_l51_5199


namespace vector_decomposition_l51_5188

/-- Given vectors in ℝ³ -/
def x : Fin 3 → ℝ := ![13, 2, 7]
def p : Fin 3 → ℝ := ![5, 1, 0]
def q : Fin 3 → ℝ := ![2, -1, 3]
def r : Fin 3 → ℝ := ![1, 0, -1]

/-- Theorem stating the decomposition of x in terms of p, q, and r -/
theorem vector_decomposition :
  x = fun i => 3 * p i + q i - 4 * r i := by
  sorry

end vector_decomposition_l51_5188


namespace square_fraction_count_l51_5190

theorem square_fraction_count : 
  ∃! (s : Finset Int), 
    (∀ n ∈ s, ∃ k : Int, (n : ℚ) / (25 - n) = k^2) ∧ 
    (∀ n ∉ s, ¬∃ k : Int, (n : ℚ) / (25 - n) = k^2) ∧ 
    s.card = 2 := by
  sorry

end square_fraction_count_l51_5190


namespace range_of_sum_of_reciprocals_l51_5175

theorem range_of_sum_of_reciprocals (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x + 4*y + 1/x + 1/y = 10) : 
  1 ≤ 1/x + 1/y ∧ 1/x + 1/y ≤ 9 := by
  sorry

end range_of_sum_of_reciprocals_l51_5175


namespace cone_base_circumference_l51_5149

/-- The circumference of the base of a right circular cone formed by gluing together
    the edges of a 180° sector cut from a circle with radius 6 inches is equal to 6π. -/
theorem cone_base_circumference (r : ℝ) (h : r = 6) : 
  (2 * π * r) / 2 = 6 * π := by
  sorry

end cone_base_circumference_l51_5149


namespace solution_set_when_m_is_one_inequality_holds_iff_m_in_range_l51_5110

-- Define the function f
def f (x m : ℝ) : ℝ := |x - m| - |x + 3*m|

-- Theorem for part I
theorem solution_set_when_m_is_one :
  {x : ℝ | f x 1 ≥ 1} = {x : ℝ | x ≤ -3/2} := by sorry

-- Theorem for part II
theorem inequality_holds_iff_m_in_range :
  (∀ (x t : ℝ), f x m < |2 + t| + |t - 1|) ↔ (0 < m ∧ m < 3/4) := by sorry

end solution_set_when_m_is_one_inequality_holds_iff_m_in_range_l51_5110


namespace sams_walking_speed_l51_5116

/-- Proves that Sam's walking speed is 5 miles per hour given the problem conditions -/
theorem sams_walking_speed (initial_distance : ℝ) (freds_speed : ℝ) (sams_distance : ℝ) : 
  initial_distance = 35 →
  freds_speed = 2 →
  sams_distance = 25 →
  (initial_distance - sams_distance) / freds_speed = sams_distance / 5 := by
  sorry

#check sams_walking_speed

end sams_walking_speed_l51_5116


namespace customers_who_tipped_l51_5126

theorem customers_who_tipped (initial_customers : ℕ) (additional_customers : ℕ) (non_tipping_customers : ℕ) : 
  initial_customers = 39 →
  additional_customers = 12 →
  non_tipping_customers = 49 →
  initial_customers + additional_customers - non_tipping_customers = 2 :=
by sorry

end customers_who_tipped_l51_5126


namespace optimal_solution_is_valid_and_unique_l51_5152

/-- Represents the solution for the tourist attraction problem -/
structure TouristAttractionSolution where
  small_car_cost : ℕ
  large_car_cost : ℕ
  small_car_trips : ℕ
  large_car_trips : ℕ

/-- Checks if a solution is valid for the tourist attraction problem -/
def is_valid_solution (s : TouristAttractionSolution) : Prop :=
  -- Total number of employees is 70
  4 * s.small_car_trips + 11 * s.large_car_trips = 70 ∧
  -- Small car cost is 5 more than large car cost
  s.small_car_cost = s.large_car_cost + 5 ∧
  -- Revenue difference between large and small car when fully loaded
  11 * s.large_car_cost - 4 * s.small_car_cost = 50 ∧
  -- Total cost does not exceed 5000
  70 * 60 + 4 * s.small_car_trips * s.small_car_cost + 
  11 * s.large_car_trips * s.large_car_cost ≤ 5000

/-- The optimal solution for the tourist attraction problem -/
def optimal_solution : TouristAttractionSolution :=
  { small_car_cost := 15
  , large_car_cost := 10
  , small_car_trips := 1
  , large_car_trips := 6 }

/-- Theorem stating that the optimal solution is valid and unique -/
theorem optimal_solution_is_valid_and_unique :
  is_valid_solution optimal_solution ∧
  ∀ s : TouristAttractionSolution, 
    is_valid_solution s → s = optimal_solution :=
sorry


end optimal_solution_is_valid_and_unique_l51_5152


namespace ellipse_semi_minor_axis_l51_5129

/-- Given an ellipse with specified center, focus, and endpoint of semi-major axis,
    prove that its semi-minor axis has length √3 -/
theorem ellipse_semi_minor_axis 
  (center : ℝ × ℝ) 
  (focus : ℝ × ℝ) 
  (semi_major_endpoint : ℝ × ℝ) 
  (h1 : center = (-3, 1)) 
  (h2 : focus = (-3, 0)) 
  (h3 : semi_major_endpoint = (-3, 3)) : 
  Real.sqrt ((center.2 - semi_major_endpoint.2)^2 - (center.2 - focus.2)^2) = Real.sqrt 3 := by
sorry

end ellipse_semi_minor_axis_l51_5129


namespace intersection_A_complement_B_l51_5122

def A : Set ℝ := {x | x ≥ -1}
def B : Set ℝ := {x | ∃ y, y = Real.log (x - 2)}

theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = Set.Icc (-1) 2 := by
  sorry

end intersection_A_complement_B_l51_5122


namespace songs_added_l51_5169

theorem songs_added (initial : ℕ) (deleted : ℕ) (final : ℕ) : 
  initial = 11 → deleted = 7 → final = 28 → final - (initial - deleted) = 24 :=
by sorry

end songs_added_l51_5169


namespace red_balls_count_l51_5124

theorem red_balls_count (total white green yellow purple : ℕ) (prob : ℚ) : 
  total = 60 ∧ 
  white = 22 ∧ 
  green = 10 ∧ 
  yellow = 7 ∧ 
  purple = 6 ∧ 
  prob = 65 / 100 ∧ 
  (white + green + yellow : ℚ) / total = prob →
  total - (white + green + yellow + purple) = 0 := by
  sorry

end red_balls_count_l51_5124


namespace range_of_a_l51_5121

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- Assume f' is the derivative of f
axiom is_derivative : ∀ x, HasDerivAt f (f' x) x

-- Condition: For all x ∈ ℝ, f'(x) < x
axiom f'_less_than_x : ∀ x, f' x < x

-- Condition: f(1-a) - f(a) ≤ 1/2 - a
axiom inequality_condition : ∀ a, f (1 - a) - f a ≤ 1/2 - a

-- Theorem: The range of values for a is a ≤ 1/2
theorem range_of_a : ∀ a, (∀ x, f (1 - x) - f x ≤ 1/2 - x) → a ≤ 1/2 :=
sorry

end range_of_a_l51_5121


namespace quadratic_roots_property_l51_5189

theorem quadratic_roots_property (m : ℝ) (r s : ℝ) : 
  (∀ x, x^2 - (m+1)*x + m = 0 ↔ x = r ∨ x = s) →
  |r + s - 2*r*s| = |1 - m| := by
  sorry

end quadratic_roots_property_l51_5189


namespace inscribed_circle_existence_l51_5127

-- Define a convex polygon type
structure ConvexPolygon where
  -- Add necessary fields (this is a simplified representation)
  vertices : List (ℝ × ℝ)
  is_convex : Bool

-- Define a function to represent the outward translation of polygon sides
def translate_sides (p : ConvexPolygon) (distance : ℝ) : ConvexPolygon :=
  sorry

-- Define a similarity relation between polygons
def is_similar (p1 p2 : ConvexPolygon) : Prop :=
  sorry

-- Define the property of having parallel and proportional sides
def has_parallel_proportional_sides (p1 p2 : ConvexPolygon) : Prop :=
  sorry

-- Define what it means for a circle to be inscribed in a polygon
def has_inscribed_circle (p : ConvexPolygon) : Prop :=
  sorry

-- The main theorem
theorem inscribed_circle_existence 
  (p : ConvexPolygon) 
  (h_convex : p.is_convex)
  (h_similar : is_similar p (translate_sides p 1))
  (h_parallel_prop : has_parallel_proportional_sides p (translate_sides p 1)) :
  has_inscribed_circle p :=
sorry

end inscribed_circle_existence_l51_5127


namespace fraction_meaningful_l51_5107

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = (2 * x - 6) / (x + 1)) ↔ x ≠ -1 :=
by sorry

end fraction_meaningful_l51_5107


namespace parabola_curve_intersection_l51_5184

/-- A parabola with equation y² = 4x and focus at (1, 0) -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- A curve with equation y = k/x where k > 0 -/
def Curve (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k / p.1 ∧ k > 0}

/-- The focus of the parabola -/
def Focus : ℝ × ℝ := (1, 0)

/-- A point P is perpendicular to the x-axis if its x-coordinate is 1 -/
def isPerpendicular (P : ℝ × ℝ) : Prop :=
  P.1 = 1

theorem parabola_curve_intersection (k : ℝ) :
  ∃ P : ℝ × ℝ, P ∈ Parabola ∧ P ∈ Curve k ∧ isPerpendicular P → k = 2 := by
  sorry

end parabola_curve_intersection_l51_5184


namespace race_lead_calculation_l51_5167

theorem race_lead_calculation (total_length max_remaining : ℕ) 
  (initial_together first_lead second_lead : ℕ) : 
  total_length = 5000 →
  max_remaining = 3890 →
  initial_together = 200 →
  first_lead = 300 →
  second_lead = 170 →
  (total_length - max_remaining - initial_together) - (first_lead - second_lead) = 780 :=
by sorry

end race_lead_calculation_l51_5167


namespace total_tax_percentage_l51_5153

/-- Calculate the total tax percentage given spending percentages, discounts, and tax rates --/
theorem total_tax_percentage
  (total_amount : ℝ)
  (clothing_percent : ℝ)
  (food_percent : ℝ)
  (electronics_percent : ℝ)
  (other_percent : ℝ)
  (clothing_discount : ℝ)
  (electronics_discount : ℝ)
  (clothing_tax : ℝ)
  (food_tax : ℝ)
  (electronics_tax : ℝ)
  (other_tax : ℝ)
  (h1 : clothing_percent = 0.4)
  (h2 : food_percent = 0.15)
  (h3 : electronics_percent = 0.25)
  (h4 : other_percent = 0.2)
  (h5 : clothing_discount = 0.1)
  (h6 : electronics_discount = 0.05)
  (h7 : clothing_tax = 0.04)
  (h8 : food_tax = 0)
  (h9 : electronics_tax = 0.06)
  (h10 : other_tax = 0.08)
  (h11 : total_amount > 0) :
  let clothing_amount := clothing_percent * total_amount
  let food_amount := food_percent * total_amount
  let electronics_amount := electronics_percent * total_amount
  let other_amount := other_percent * total_amount
  let discounted_clothing := clothing_amount * (1 - clothing_discount)
  let discounted_electronics := electronics_amount * (1 - electronics_discount)
  let total_tax := clothing_tax * discounted_clothing +
                   food_tax * food_amount +
                   electronics_tax * discounted_electronics +
                   other_tax * other_amount
  ∃ ε > 0, |total_tax / total_amount - 0.04465| < ε :=
by sorry

end total_tax_percentage_l51_5153


namespace problem_statement_l51_5104

-- Define the sets A and B
def A : Set ℝ := Set.Ioo (-2) 2
def B (a : ℝ) : Set ℝ := Set.Ioo a (1 - a)

-- State the theorem
theorem problem_statement (a : ℝ) (h : a < 0) :
  (A ∪ B a = B a → a ≤ -2) ∧
  (A ∩ B a = B a → a ≥ -1) := by
  sorry

end problem_statement_l51_5104


namespace no_solution_absolute_value_equation_l51_5120

theorem no_solution_absolute_value_equation :
  ¬∃ x : ℝ, |(-2 * x + 1)| + 4 = 0 := by
sorry

end no_solution_absolute_value_equation_l51_5120


namespace teams_of_four_from_seven_l51_5142

theorem teams_of_four_from_seven (n : ℕ) (k : ℕ) : n = 7 → k = 4 → Nat.choose n k = 35 := by
  sorry

end teams_of_four_from_seven_l51_5142


namespace exam_average_l51_5182

theorem exam_average (students_group1 : ℕ) (average1 : ℚ) 
  (students_group2 : ℕ) (average2 : ℚ) : 
  students_group1 = 15 → 
  average1 = 70/100 → 
  students_group2 = 10 → 
  average2 = 90/100 → 
  (students_group1 * average1 + students_group2 * average2) / (students_group1 + students_group2) = 78/100 := by
  sorry

end exam_average_l51_5182


namespace circle_inequality_m_range_l51_5178

theorem circle_inequality_m_range :
  ∀ m : ℝ,
  (∀ x y : ℝ, x^2 + (y - 1)^2 = 1 → x + y + m ≥ 0) ↔
  m > -1 :=
by sorry

end circle_inequality_m_range_l51_5178


namespace hyperbola_asymptote_l51_5123

-- Define the hyperbola and its properties
def Hyperbola (m : ℝ) : Prop :=
  m > 0 ∧ ∃ x y : ℝ, x^2 / m - y^2 = 1

-- Define the asymptotic line
def AsymptoticLine (x y : ℝ) : Prop :=
  x + 3 * y = 0

-- Theorem statement
theorem hyperbola_asymptote (m : ℝ) :
  Hyperbola m → (∃ x y : ℝ, AsymptoticLine x y) → m = 9 :=
sorry

end hyperbola_asymptote_l51_5123


namespace point_not_in_region_l51_5131

def plane_region (x y : ℝ) : Prop := 3*x + 2*y > 3

theorem point_not_in_region :
  ¬(plane_region 0 0) ∧
  (plane_region 1 1) ∧
  (plane_region 0 2) ∧
  (plane_region 2 0) := by
  sorry

end point_not_in_region_l51_5131


namespace matrix_equation_solution_l51_5118

theorem matrix_equation_solution :
  ∀ (M : Matrix (Fin 2) (Fin 2) ℝ),
  M^3 - 5 • M^2 + 6 • M = !![16, 8; 24, 12] →
  M = !![4, 2; 6, 3] := by
sorry

end matrix_equation_solution_l51_5118


namespace original_solution_concentration_l51_5136

/-- Proves that given the conditions, the original solution's concentration is 50% -/
theorem original_solution_concentration
  (replaced_portion : ℝ)
  (h_replaced : replaced_portion = 0.8181818181818182)
  (x : ℝ)
  (h_result : x / 100 * (1 - replaced_portion) + 30 / 100 * replaced_portion = 40 / 100) :
  x = 50 :=
sorry

end original_solution_concentration_l51_5136


namespace cost_price_is_640_l51_5101

/-- The cost price of an article given its selling price and profit percentage -/
def costPrice (sellingPrice : ℚ) (profitPercentage : ℚ) : ℚ :=
  sellingPrice / (1 + profitPercentage / 100)

/-- Theorem stating that the cost price is 640 given the conditions -/
theorem cost_price_is_640 (sellingPrice : ℚ) (profitPercentage : ℚ) 
  (h1 : sellingPrice = 800)
  (h2 : profitPercentage = 25) : 
  costPrice sellingPrice profitPercentage = 640 := by
  sorry

end cost_price_is_640_l51_5101


namespace triangle_inequality_variant_l51_5113

theorem triangle_inequality_variant (x y z : ℝ) :
  (|x| < |y - z| ∧ |y| < |z - x|) → |z| ≥ |x - y| := by
  sorry

end triangle_inequality_variant_l51_5113


namespace not_divisible_by_three_times_sum_of_products_l51_5161

theorem not_divisible_by_three_times_sum_of_products (x y z : ℕ+) :
  ¬ (3 * (x * y + y * z + z * x) ∣ x^2 + y^2 + z^2) := by
  sorry

end not_divisible_by_three_times_sum_of_products_l51_5161


namespace cylinder_in_sphere_volume_l51_5150

theorem cylinder_in_sphere_volume (r_sphere r_cylinder : ℝ) (h_sphere : r_sphere = 6) (h_cylinder : r_cylinder = 4) :
  let h_cylinder := 2 * (r_sphere ^ 2 - r_cylinder ^ 2).sqrt
  let v_sphere := (4 / 3) * π * r_sphere ^ 3
  let v_cylinder := π * r_cylinder ^ 2 * h_cylinder
  (v_sphere - v_cylinder) / π = 288 - 64 * Real.sqrt 5 := by
sorry

end cylinder_in_sphere_volume_l51_5150


namespace river_speed_proof_l51_5154

theorem river_speed_proof (rowing_speed : ℝ) (total_time : ℝ) (total_distance : ℝ) 
  (h1 : rowing_speed = 6)
  (h2 : total_time = 1)
  (h3 : total_distance = 5.76) :
  ∃ (river_speed : ℝ),
    river_speed = 1.2 ∧
    (total_distance / 2) / (rowing_speed - river_speed) +
    (total_distance / 2) / (rowing_speed + river_speed) = total_time :=
by sorry

end river_speed_proof_l51_5154


namespace spinner_final_direction_l51_5102

-- Define the possible directions
inductive Direction
| North
| East
| South
| West

-- Define the rotation type
inductive RotationType
| Clockwise
| Counterclockwise

-- Define a function to represent a rotation
def rotate (initial : Direction) (amount : Rat) (type : RotationType) : Direction :=
  sorry

-- Define the problem statement
theorem spinner_final_direction 
  (initial : Direction)
  (rotation1 : Rat)
  (type1 : RotationType)
  (rotation2 : Rat)
  (type2 : RotationType)
  (h1 : initial = Direction.South)
  (h2 : rotation1 = 19/4)
  (h3 : type1 = RotationType.Clockwise)
  (h4 : rotation2 = 13/2)
  (h5 : type2 = RotationType.Counterclockwise) :
  rotate (rotate initial rotation1 type1) rotation2 type2 = Direction.East :=
sorry

end spinner_final_direction_l51_5102


namespace divisibility_implies_equality_l51_5151

-- Define the divisibility relation
def divides (m n : ℕ) : Prop := ∃ k : ℕ, n = m * k

-- Define an infinite set of natural numbers
def InfiniteSet (S : Set ℕ) : Prop :=
  ∀ n : ℕ, ∃ m ∈ S, m > n

theorem divisibility_implies_equality (a b : ℕ) 
  (h : ∃ S : Set ℕ, InfiniteSet S ∧ ∀ n ∈ S, divides (a^n + b^n) (a^(n+1) + b^(n+1))) :
  a = b :=
sorry

end divisibility_implies_equality_l51_5151


namespace isosceles_right_triangle_shaded_area_l51_5194

theorem isosceles_right_triangle_shaded_area (leg_length : ℝ) (total_partitions : ℕ) (shaded_partitions : ℕ) : 
  leg_length = 12 →
  total_partitions = 36 →
  shaded_partitions = 15 →
  (shaded_partitions : ℝ) * (leg_length^2 / (2 * total_partitions : ℝ)) = 30 :=
by sorry

end isosceles_right_triangle_shaded_area_l51_5194


namespace scooter_purchase_price_l51_5180

/-- Proves that given the conditions of the scooter purchase, repair, sale, and profit,
    the original purchase price must be $4700. -/
theorem scooter_purchase_price (P : ℝ) : 
  P > 0 →
  5800 - (P + 600) = (9.433962264150944 / 100) * (P + 600) →
  P = 4700 := by
sorry

end scooter_purchase_price_l51_5180


namespace student_average_weight_l51_5192

theorem student_average_weight 
  (n : ℕ) 
  (teacher_weight : ℝ) 
  (weight_increase : ℝ) : 
  n = 24 → 
  teacher_weight = 45 → 
  weight_increase = 0.4 → 
  (n * 35 + teacher_weight) / (n + 1) = 35 + weight_increase :=
by sorry

end student_average_weight_l51_5192


namespace trajectory_is_parabola_l51_5148

/-- The trajectory of a point equidistant from a fixed point and a line is a parabola -/
theorem trajectory_is_parabola (M : ℝ × ℝ) :
  (∀ (x y : ℝ), M = (x, y) →
    dist M (0, -3) = |y - 3|) →
  ∃ (x y : ℝ), M = (x, y) ∧ x^2 = -12*y :=
sorry

end trajectory_is_parabola_l51_5148


namespace ball_count_after_50_moves_l51_5119

/-- Represents the state of the boxes --/
structure BoxState :=
  (A : ℕ)
  (B : ℕ)
  (C : ℕ)
  (D : ℕ)

/-- Performs one iteration of the ball-moving process --/
def moveOnce (state : BoxState) : BoxState :=
  sorry

/-- Performs n iterations of the ball-moving process --/
def moveNTimes (n : ℕ) (state : BoxState) : BoxState :=
  sorry

/-- The initial state of the boxes --/
def initialState : BoxState :=
  { A := 8, B := 6, C := 3, D := 1 }

theorem ball_count_after_50_moves :
  (moveNTimes 50 initialState).A = 6 := by
  sorry

end ball_count_after_50_moves_l51_5119


namespace multiplication_mistake_difference_l51_5103

theorem multiplication_mistake_difference : 
  let correct_multiplicand : Nat := 136
  let correct_multiplier : Nat := 43
  let mistaken_multiplier : Nat := 34
  (correct_multiplicand * correct_multiplier) - (correct_multiplicand * mistaken_multiplier) = 1224 := by
  sorry

end multiplication_mistake_difference_l51_5103


namespace inscribed_square_area_largest_inscribed_square_area_l51_5198

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 - 6*x + 7

/-- The side length of the inscribed square -/
noncomputable def s : ℝ := -1 + Real.sqrt 3

/-- The area of the inscribed square -/
noncomputable def area : ℝ := (2*s)^2

theorem inscribed_square_area :
  ∀ (a : ℝ), a > 0 →
  (∀ (x : ℝ), x ∈ Set.Icc (3 - a/2) (3 + a/2) → f x ≥ 0) →
  (f (3 - a/2) = 0 ∨ f (3 + a/2) = 0) →
  a ≤ 2*s :=
by sorry

theorem largest_inscribed_square_area :
  area = 16 - 8 * Real.sqrt 3 :=
by sorry

end inscribed_square_area_largest_inscribed_square_area_l51_5198


namespace joe_age_proof_l51_5176

theorem joe_age_proof (joe james : ℕ) : 
  joe = james + 10 →
  2 * (joe + 8) = 3 * (james + 8) →
  joe = 22 := by
sorry

end joe_age_proof_l51_5176


namespace intersection_sum_l51_5174

/-- Given two lines y = 2x + c and y = 4x + d intersecting at (3, 11), prove that c + d = 4 -/
theorem intersection_sum (c d : ℝ) 
  (h1 : 11 = 2 * 3 + c) 
  (h2 : 11 = 4 * 3 + d) : 
  c + d = 4 := by sorry

end intersection_sum_l51_5174


namespace line_passes_through_point_l51_5100

theorem line_passes_through_point (A B C : ℝ) :
  A - B + C = 0 →
  ∀ (x y : ℝ), A * x + B * y + C = 0 ↔ (x = 1 ∧ y = -1) :=
by sorry

end line_passes_through_point_l51_5100


namespace statement_is_universal_l51_5146

-- Define the concept of a line
def Line : Type := sorry

-- Define the concept of a plane
def Plane : Type := sorry

-- Define the property of two lines intersecting
def intersect (l1 l2 : Line) : Prop := sorry

-- Define the property of a plane passing through two lines
def passes_through (p : Plane) (l1 l2 : Line) : Prop := sorry

-- Define the statement as a proposition
def statement : Prop :=
  ∀ l1 l2 : Line, intersect l1 l2 → ∃! p : Plane, passes_through p l1 l2

-- Theorem to prove that the statement is a universal proposition
theorem statement_is_universal : 
  (∀ l1 l2 : Line, intersect l1 l2 → ∃! p : Plane, passes_through p l1 l2) ↔ statement :=
sorry

end statement_is_universal_l51_5146


namespace six_arts_competition_l51_5186

theorem six_arts_competition (a b c : ℕ) (h_abc : a > b ∧ b > c) :
  (∃ (x y z : ℕ),
    x + y + z = 6 ∧
    a * x + b * y + c * z = 26 ∧
    (∃ (p q r : ℕ),
      p + q + r = 6 ∧
      a * p + b * q + c * r = 11 ∧
      p = 1 ∧
      (∃ (u v w : ℕ),
        u + v + w = 6 ∧
        a * u + b * v + c * w = 11 ∧
        a + b + c = 8))) →
  (∃ (p q r : ℕ),
    p + q + r = 6 ∧
    a * p + b * q + c * r = 11 ∧
    p = 1 ∧
    r = 4) :=
by sorry

end six_arts_competition_l51_5186


namespace fourth_to_third_l51_5133

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point is in the fourth quadrant -/
def in_fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Predicate to check if a point is in the third quadrant -/
def in_third_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- Theorem stating that if P(a,b) is in the fourth quadrant, 
    then Q(-a,b-1) is in the third quadrant -/
theorem fourth_to_third (a b : ℝ) :
  in_fourth_quadrant ⟨a, b⟩ → in_third_quadrant ⟨-a, b-1⟩ := by
  sorry


end fourth_to_third_l51_5133


namespace time_after_1007_hours_l51_5164

def clock_add (current_time hours_elapsed : ℕ) : ℕ :=
  (current_time + hours_elapsed) % 12

theorem time_after_1007_hours :
  let current_time := 5
  let hours_elapsed := 1007
  clock_add current_time hours_elapsed = 4 := by
sorry

end time_after_1007_hours_l51_5164


namespace soda_cost_per_ounce_l51_5197

/-- The cost of soda per ounce, given initial money, remaining money, and amount bought. -/
def cost_per_ounce (initial_money remaining_money amount_bought : ℚ) : ℚ :=
  (initial_money - remaining_money) / amount_bought

/-- Theorem stating that the cost per ounce is $0.25 under given conditions. -/
theorem soda_cost_per_ounce :
  cost_per_ounce 2 0.5 6 = 0.25 := by
  sorry

end soda_cost_per_ounce_l51_5197


namespace complex_fraction_sum_l51_5177

theorem complex_fraction_sum (a b : ℝ) :
  (3 + b * Complex.I) / (1 - Complex.I) = a + b * Complex.I →
  a + b = 3 := by
sorry

end complex_fraction_sum_l51_5177


namespace differential_equation_solution_l51_5159

open Real

theorem differential_equation_solution 
  (y : ℝ → ℝ) 
  (C₁ C₂ : ℝ) 
  (h : ∀ x, y x = (C₁ + C₂ * x) * exp (3 * x) + exp x - 8 * x^2 * exp (3 * x)) :
  ∀ x, (deriv^[2] y) x - 6 * (deriv y) x + 9 * y x = 4 * exp x - 16 * exp (3 * x) := by
  sorry

end differential_equation_solution_l51_5159


namespace juan_saw_eight_pickup_trucks_l51_5106

/-- The number of pickup trucks Juan saw -/
def num_pickup_trucks : ℕ := sorry

/-- The total number of tires on all vehicles Juan saw -/
def total_tires : ℕ := 101

/-- The number of cars Juan saw -/
def num_cars : ℕ := 15

/-- The number of bicycles Juan saw -/
def num_bicycles : ℕ := 3

/-- The number of tricycles Juan saw -/
def num_tricycles : ℕ := 1

/-- The number of tires on a car -/
def tires_per_car : ℕ := 4

/-- The number of tires on a bicycle -/
def tires_per_bicycle : ℕ := 2

/-- The number of tires on a tricycle -/
def tires_per_tricycle : ℕ := 3

/-- The number of tires on a pickup truck -/
def tires_per_pickup : ℕ := 4

theorem juan_saw_eight_pickup_trucks : num_pickup_trucks = 8 := by
  sorry

end juan_saw_eight_pickup_trucks_l51_5106


namespace part_one_part_two_part_three_l51_5114

/-- Definition of equivalent rational number pair -/
def is_equivalent_pair (m n : ℚ) : Prop := m + n = m * n

/-- Part 1: Prove that (3, 3/2) is an equivalent rational number pair -/
theorem part_one : is_equivalent_pair 3 (3/2) := by sorry

/-- Part 2: If (x+1, 4) is an equivalent rational number pair, then x = 1/3 -/
theorem part_two (x : ℚ) : is_equivalent_pair (x + 1) 4 → x = 1/3 := by sorry

/-- Part 3: If (m, n) is an equivalent rational number pair, 
    then 12 - 6mn + 6m + 6n = 12 -/
theorem part_three (m n : ℚ) : 
  is_equivalent_pair m n → 12 - 6*m*n + 6*m + 6*n = 12 := by sorry

end part_one_part_two_part_three_l51_5114


namespace radio_cost_price_l51_5147

theorem radio_cost_price (selling_price : ℝ) (loss_percentage : ℝ) 
  (h1 : selling_price = 1275)
  (h2 : loss_percentage = 15) : 
  ∃ (cost_price : ℝ), 
    cost_price = 1500 ∧ 
    selling_price = cost_price * (1 - loss_percentage / 100) := by
sorry

end radio_cost_price_l51_5147


namespace swim_team_girls_count_l51_5163

theorem swim_team_girls_count :
  ∀ (boys girls coaches managers : ℕ),
  girls = 5 * boys →
  coaches = 4 →
  managers = 4 →
  boys + girls + coaches + managers = 104 →
  girls = 80 :=
by
  sorry

end swim_team_girls_count_l51_5163


namespace greatest_b_quadratic_inequality_l51_5130

theorem greatest_b_quadratic_inequality :
  ∃ b : ℝ, b^2 - 14*b + 45 ≤ 0 ∧
  ∀ x : ℝ, x^2 - 14*x + 45 ≤ 0 → x ≤ b ∧
  b = 9 :=
sorry

end greatest_b_quadratic_inequality_l51_5130


namespace line_perpendicular_to_plane_l51_5172

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (non_intersecting : Plane → Plane → Prop)

-- State the theorem
theorem line_perpendicular_to_plane
  (m n : Line) (α β : Plane)
  (different_lines : m ≠ n)
  (non_intersecting_planes : non_intersecting α β)
  (m_parallel_n : parallel m n)
  (n_perp_β : perpendicular n β) :
  perpendicular m β :=
sorry

end line_perpendicular_to_plane_l51_5172


namespace evaluate_expression_l51_5134

theorem evaluate_expression (c d : ℝ) (h : c^2 ≠ d^2) :
  (c^4 - d^4) / (2 * (c^2 - d^2)) = (c^2 + d^2) / 2 := by
  sorry

end evaluate_expression_l51_5134


namespace magnitude_comparison_l51_5181

theorem magnitude_comparison (a b c : ℝ) 
  (ha : a > 0) 
  (hbc : b * c > a^2) 
  (heq : a^2 - 2*a*b + c^2 = 0) : 
  b > c ∧ c > a :=
by sorry

end magnitude_comparison_l51_5181


namespace museum_group_time_l51_5155

/-- Proves that the time taken for each group to go through the museum is 24 minutes -/
theorem museum_group_time (total_students : ℕ) (num_groups : ℕ) (time_per_student : ℕ) : 
  total_students = 18 → num_groups = 3 → time_per_student = 4 → 
  (total_students / num_groups) * time_per_student = 24 := by
  sorry

end museum_group_time_l51_5155


namespace selina_pants_sold_l51_5162

/-- Represents the number of pants Selina sold -/
def pants_sold : ℕ := sorry

/-- The price of each pair of pants -/
def pants_price : ℕ := 5

/-- The price of each pair of shorts -/
def shorts_price : ℕ := 3

/-- The price of each shirt -/
def shirt_price : ℕ := 4

/-- The number of shorts Selina sold -/
def shorts_sold : ℕ := 5

/-- The number of shirts Selina sold -/
def shirts_sold : ℕ := 5

/-- The price of each new shirt Selina bought -/
def new_shirt_price : ℕ := 10

/-- The number of new shirts Selina bought -/
def new_shirts_bought : ℕ := 2

/-- The amount of money Selina left the store with -/
def money_left : ℕ := 30

theorem selina_pants_sold : 
  pants_sold * pants_price + 
  shorts_sold * shorts_price + 
  shirts_sold * shirt_price = 
  money_left + new_shirts_bought * new_shirt_price ∧ 
  pants_sold = 3 := by sorry

end selina_pants_sold_l51_5162


namespace video_game_time_l51_5128

/-- 
Proves that given the conditions of the problem, 
the time spent playing video games is 9 hours.
-/
theorem video_game_time 
  (study_rate : ℝ)  -- Rate at which grade increases per hour of studying
  (final_grade : ℝ)  -- Final grade achieved
  (study_ratio : ℝ)  -- Ratio of study time to gaming time
  (h_study_rate : study_rate = 15)  -- Grade increases by 15 points per hour of studying
  (h_final_grade : final_grade = 45)  -- Final grade is 45 points
  (h_study_ratio : study_ratio = 1/3)  -- Study time is 1/3 of gaming time
  : ∃ (game_time : ℝ), game_time = 9 := by
  sorry

end video_game_time_l51_5128


namespace parallel_lines_imply_a_eq_neg_two_l51_5108

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ b₁ m₂ b₂ : ℝ} :
  (∀ x y : ℝ, m₁ * x + b₁ = y ↔ m₂ * x + b₂ = y) ↔ m₁ = m₂

/-- Definition of line l₁ -/
def l₁ (x y : ℝ) : Prop := 2 * x - y - 1 = 0

/-- Definition of line l₂ -/
def l₂ (a x y : ℝ) : Prop := 2 * x + (a + 1) * y + 2 = 0

/-- Theorem: If l₁ is parallel to l₂, then a = -2 -/
theorem parallel_lines_imply_a_eq_neg_two :
  (∀ x y : ℝ, l₁ x y ↔ l₂ a x y) → a = -2 := by sorry

end parallel_lines_imply_a_eq_neg_two_l51_5108


namespace sqrt_meaningful_range_l51_5170

theorem sqrt_meaningful_range (a : ℝ) : (∃ (x : ℝ), x^2 = 2 + a) ↔ a ≥ -2 := by sorry

end sqrt_meaningful_range_l51_5170


namespace line_AB_equation_l51_5185

-- Define the circles
def circle1 (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := (x - 2.5)^2 + (y - 0.5)^2 = 2.5

-- Define point P
def P : ℝ × ℝ := (4, 1)

-- Define the line AB
def lineAB (x y : ℝ) : Prop := 3*x + y - 4 = 0

-- Theorem statement
theorem line_AB_equation :
  ∀ x y : ℝ,
  (circle1 x y ∧ circle2 x y) →
  lineAB x y :=
sorry

end line_AB_equation_l51_5185


namespace triangle_angle_c_l51_5111

theorem triangle_angle_c (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ A < π → 
  0 < B ∧ B < π → 
  0 < C ∧ C < π → 
  a > 0 → b > 0 → c > 0 →
  a = 2 →
  b + c = 2 * a →
  3 * Real.sin A = 5 * Real.sin B →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  c^2 = a^2 + b^2 - 2*a*b * Real.cos C →
  C = 2 * π / 3 := by
sorry

end triangle_angle_c_l51_5111


namespace collectible_figure_price_l51_5132

theorem collectible_figure_price (sneaker_cost lawn_count lawn_price job_hours job_rate figure_count : ℕ) 
  (h1 : sneaker_cost = 92)
  (h2 : lawn_count = 3)
  (h3 : lawn_price = 8)
  (h4 : job_hours = 10)
  (h5 : job_rate = 5)
  (h6 : figure_count = 2) :
  let lawn_earnings := lawn_count * lawn_price
  let job_earnings := job_hours * job_rate
  let total_earnings := lawn_earnings + job_earnings
  let remaining_amount := sneaker_cost - total_earnings
  (remaining_amount / figure_count : ℚ) = 9 := by
  sorry

end collectible_figure_price_l51_5132


namespace solve_candy_bar_problem_l51_5112

def candy_bar_problem (initial_amount : ℚ) (num_candy_bars : ℕ) (remaining_amount : ℚ) : Prop :=
  ∃ (price_per_bar : ℚ),
    initial_amount - num_candy_bars * price_per_bar = remaining_amount ∧
    price_per_bar > 0

theorem solve_candy_bar_problem :
  candy_bar_problem 4 10 1 → (4 : ℚ) - 1 = 3 :=
by
  sorry

end solve_candy_bar_problem_l51_5112


namespace constant_term_binomial_expansion_l51_5193

theorem constant_term_binomial_expansion (n : ℕ) (A B : ℕ) : 
  A = (4 : ℝ) ^ n →
  B = 2 ^ n →
  A + B = 72 →
  ∃ (r : ℕ), r = 1 ∧ 3 * (Nat.choose n r) = 9 :=
by sorry

end constant_term_binomial_expansion_l51_5193
