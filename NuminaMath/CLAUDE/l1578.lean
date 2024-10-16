import Mathlib

namespace NUMINAMATH_CALUDE_cos_2alpha_value_l1578_157805

theorem cos_2alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo (π / 2) π) 
  (h2 : Real.sin α + Real.cos α = 1 / 5) : 
  Real.cos (2 * α) = -7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_value_l1578_157805


namespace NUMINAMATH_CALUDE_heather_walking_distance_l1578_157844

/-- The total distance Heather walked at the county fair -/
theorem heather_walking_distance :
  let car_to_entrance : ℚ := 0.3333333333333333
  let entrance_to_rides : ℚ := 0.3333333333333333
  let rides_to_car : ℚ := 0.08333333333333333
  car_to_entrance + entrance_to_rides + rides_to_car = 0.75
:= by sorry

end NUMINAMATH_CALUDE_heather_walking_distance_l1578_157844


namespace NUMINAMATH_CALUDE_line_inclination_angle_l1578_157897

theorem line_inclination_angle (x1 y1 x2 y2 : ℝ) :
  x1 = 1 →
  y1 = 1 →
  x2 = 2 →
  y2 = 1 + Real.sqrt 3 →
  ∃ θ : ℝ, θ * (π / 180) = π / 3 ∧ Real.tan θ = (y2 - y1) / (x2 - x1) := by
  sorry

end NUMINAMATH_CALUDE_line_inclination_angle_l1578_157897


namespace NUMINAMATH_CALUDE_trig_identity_proof_l1578_157884

theorem trig_identity_proof :
  6 * Real.cos (10 * π / 180) * Real.cos (50 * π / 180) * Real.cos (70 * π / 180) +
  Real.sin (10 * π / 180) * Real.sin (50 * π / 180) * Real.sin (70 * π / 180) =
  6 * (1 + Real.sqrt 3) / 8 := by
sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l1578_157884


namespace NUMINAMATH_CALUDE_fair_distribution_theorem_l1578_157861

/-- Represents the outcome of a chess game -/
inductive GameOutcome
  | A_Win
  | B_Win

/-- Represents the state of the chess competition -/
structure ChessCompetition where
  total_games : Nat
  games_played : Nat
  a_wins : Nat
  prize_money : Nat
  deriving Repr

/-- Calculates the probability of player A winning the competition -/
def probability_a_wins (comp : ChessCompetition) : Rat :=
  sorry

/-- Calculates the fair distribution of prize money -/
def fair_distribution (comp : ChessCompetition) : Nat × Nat :=
  sorry

/-- Theorem stating the fair distribution of prize money -/
theorem fair_distribution_theorem (comp : ChessCompetition) 
  (h1 : comp.total_games = 7)
  (h2 : comp.games_played = 5)
  (h3 : comp.a_wins = 3)
  (h4 : comp.prize_money = 10000) :
  fair_distribution comp = (7500, 2500) :=
sorry

end NUMINAMATH_CALUDE_fair_distribution_theorem_l1578_157861


namespace NUMINAMATH_CALUDE_arithmetic_sum_example_l1578_157801

/-- Sum of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) : ℕ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem arithmetic_sum_example : arithmetic_sum 2 20 2 = 110 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sum_example_l1578_157801


namespace NUMINAMATH_CALUDE_baez_marbles_l1578_157877

theorem baez_marbles (p : ℝ) : 
  25 > 0 ∧ 0 ≤ p ∧ p ≤ 100 ∧ 2 * ((100 - p) / 100 * 25) = 60 → p = 20 :=
by sorry

end NUMINAMATH_CALUDE_baez_marbles_l1578_157877


namespace NUMINAMATH_CALUDE_system_solution_existence_l1578_157898

theorem system_solution_existence (a : ℝ) :
  (∃ (x y b : ℝ), y = x^2 - a ∧ x^2 + y^2 + 8*b^2 = 4*b*(y - x) + 1) ↔ 
  a ≥ -Real.sqrt 2 - 1/4 := by sorry

end NUMINAMATH_CALUDE_system_solution_existence_l1578_157898


namespace NUMINAMATH_CALUDE_rectangular_prism_surface_area_volume_l1578_157846

theorem rectangular_prism_surface_area_volume (x : ℝ) (h : x > 0) :
  let a := Real.log x
  let b := Real.exp (Real.log x)
  let c := x
  let surface_area := 2 * (a * b + b * c + c * a)
  let volume := a * b * c
  surface_area = 3 * volume → x = Real.exp 2 := by
sorry

end NUMINAMATH_CALUDE_rectangular_prism_surface_area_volume_l1578_157846


namespace NUMINAMATH_CALUDE_min_honey_amount_l1578_157820

theorem min_honey_amount (o h : ℝ) : 
  (o ≥ 8 + h / 3 ∧ o ≤ 3 * h) → h ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_honey_amount_l1578_157820


namespace NUMINAMATH_CALUDE_simplify_expression_constant_sum_l1578_157807

/-- Given expressions for A and B in terms of a and b -/
def A (a b : ℝ) : ℝ := 2 * a^2 + a * b - 2 * b - 1

/-- Given expressions for A and B in terms of a and b -/
def B (a b : ℝ) : ℝ := -a^2 + a * b - 2

/-- Theorem 1: Simplification of 3A - (2A - 2B) -/
theorem simplify_expression (a b : ℝ) :
  3 * A a b - (2 * A a b - 2 * B a b) = 3 * a * b - 2 * b - 5 := by sorry

/-- Theorem 2: Value of a when A + 2B is constant for any b -/
theorem constant_sum (a : ℝ) :
  (∀ b : ℝ, ∃ k : ℝ, A a b + 2 * B a b = k) → a = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_constant_sum_l1578_157807


namespace NUMINAMATH_CALUDE_points_on_parabola_l1578_157818

-- Define the sequence of points
def SequencePoints (x y : ℕ → ℝ) : Prop :=
  ∀ n, Real.sqrt ((x n)^2 + (y n)^2) - y n = 6

-- Define the parabola
def OnParabola (x y : ℝ) : Prop :=
  y = (x^2 / 12) - 3

-- Theorem statement
theorem points_on_parabola 
  (x y : ℕ → ℝ) 
  (h : SequencePoints x y) :
  ∀ n, OnParabola (x n) (y n) := by
sorry

end NUMINAMATH_CALUDE_points_on_parabola_l1578_157818


namespace NUMINAMATH_CALUDE_arithmetic_progression_quadratic_roots_l1578_157888

/-- Given non-zero real numbers a, b, c forming an arithmetic progression with b as the middle term,
    the quadratic equation ax^2 + 2√2bx + c = 0 has two distinct real roots. -/
theorem arithmetic_progression_quadratic_roots (a b c : ℝ) 
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h_arithmetic : ∃ d : ℝ, a = b - d ∧ c = b + d) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + 2 * Real.sqrt 2 * b * x₁ + c = 0 ∧
                a * x₂^2 + 2 * Real.sqrt 2 * b * x₂ + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_quadratic_roots_l1578_157888


namespace NUMINAMATH_CALUDE_sum_abc_equals_16_l1578_157882

theorem sum_abc_equals_16 (a b c : ℕ+) 
  (h1 : a * b + 2 * c + 3 = 47)
  (h2 : b * c + 2 * a + 3 = 47)
  (h3 : a * c + 2 * b + 3 = 47) :
  a + b + c = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_abc_equals_16_l1578_157882


namespace NUMINAMATH_CALUDE_wrapping_paper_area_l1578_157871

/-- The area of wrapping paper required to wrap a box on a pedestal -/
theorem wrapping_paper_area (w h p : ℝ) (hw : w > 0) (hh : h > 0) (hp : p > 0) :
  let paper_area := 4 * w * (p + h)
  paper_area = 4 * w * (p + h) :=
by sorry

end NUMINAMATH_CALUDE_wrapping_paper_area_l1578_157871


namespace NUMINAMATH_CALUDE_sara_remaining_money_l1578_157804

/-- Calculates the remaining money after a two-week pay period and a purchase -/
def remaining_money (hours_per_week : ℕ) (hourly_rate : ℚ) (purchase_cost : ℚ) : ℚ :=
  2 * (hours_per_week : ℚ) * hourly_rate - purchase_cost

/-- Proves that given the specified work conditions and purchase, the remaining money is $510 -/
theorem sara_remaining_money :
  remaining_money 40 (11.5) 410 = 510 := by
  sorry

end NUMINAMATH_CALUDE_sara_remaining_money_l1578_157804


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_attained_l1578_157816

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (3 * z) / (x + 2 * y) + (5 * x) / (2 * y + 3 * z) + (2 * y) / (3 * x + z) ≥ (3 : ℝ) / 4 :=
by sorry

theorem min_value_attained (ε : ℝ) (hε : ε > 0) :
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧
  (3 * z) / (x + 2 * y) + (5 * x) / (2 * y + 3 * z) + (2 * y) / (3 * x + z) < (3 : ℝ) / 4 + ε :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_attained_l1578_157816


namespace NUMINAMATH_CALUDE_geometric_sequence_a4_l1578_157869

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_a4 (a : ℕ → ℝ) :
  geometric_sequence a → a 2 = 4 → a 6 = 16 → a 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a4_l1578_157869


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l1578_157809

theorem unique_solution_quadratic (m : ℝ) : 
  (∃! x : ℝ, (x + 5) * (x + 2) = m + 3 * x) ↔ m = 6 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l1578_157809


namespace NUMINAMATH_CALUDE_river_flow_speed_l1578_157813

/-- Proves that the speed of river flow is 2 km/hr given the conditions of the boat journey -/
theorem river_flow_speed (distance : ℝ) (boat_speed : ℝ) (total_time : ℝ) :
  distance = 48 →
  boat_speed = 6 →
  total_time = 18 →
  ∃ (river_speed : ℝ),
    river_speed > 0 ∧
    (distance / (boat_speed - river_speed) + distance / (boat_speed + river_speed) = total_time) ∧
    river_speed = 2 := by
  sorry


end NUMINAMATH_CALUDE_river_flow_speed_l1578_157813


namespace NUMINAMATH_CALUDE_win_sector_area_l1578_157857

theorem win_sector_area (r : ℝ) (p : ℝ) (h1 : r = 7) (h2 : p = 3/8) :
  p * π * r^2 = 147 * π / 8 := by
  sorry

end NUMINAMATH_CALUDE_win_sector_area_l1578_157857


namespace NUMINAMATH_CALUDE_cone_volume_l1578_157814

/-- The volume of a cone with given slant height and lateral surface area -/
theorem cone_volume (l : ℝ) (lateral_area : ℝ) (h : l = 2) (h' : lateral_area = 2 * Real.pi) :
  ∃ (r : ℝ) (h : ℝ),
    r > 0 ∧ h > 0 ∧
    lateral_area = Real.pi * r * l ∧
    h^2 + r^2 = l^2 ∧
    (1/3 : ℝ) * Real.pi * r^2 * h = (Real.sqrt 3 * Real.pi) / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_cone_volume_l1578_157814


namespace NUMINAMATH_CALUDE_raisin_nut_cost_ratio_l1578_157886

theorem raisin_nut_cost_ratio :
  ∀ (r n : ℝ),
  r > 0 →
  n > 0 →
  (5 * r) / (5 * r + 4 * n) = 0.29411764705882354 →
  n / r = 3 :=
by sorry

end NUMINAMATH_CALUDE_raisin_nut_cost_ratio_l1578_157886


namespace NUMINAMATH_CALUDE_distance_at_time_l1578_157860

/-- Represents a right-angled triangle with given hypotenuse and leg lengths -/
structure RightTriangle where
  hypotenuse : ℝ
  leg : ℝ

/-- Represents a moving point with a given speed -/
structure MovingPoint where
  speed : ℝ

theorem distance_at_time (triangle : RightTriangle) (point1 point2 : MovingPoint) :
  triangle.hypotenuse = 85 →
  triangle.leg = 75 →
  point1.speed = 8.5 →
  point2.speed = 5 →
  ∃ t : ℝ, t = 4 ∧ 
    let d1 := triangle.hypotenuse - point1.speed * t
    let d2 := triangle.leg - point2.speed * t
    d1 * d1 + d2 * d2 = 26 * 26 :=
by sorry

end NUMINAMATH_CALUDE_distance_at_time_l1578_157860


namespace NUMINAMATH_CALUDE_fertilizer_weight_calculation_l1578_157810

/-- Calculates the total weight of fertilizers applied to a given area -/
theorem fertilizer_weight_calculation 
  (field_area : ℝ) 
  (fertilizer_a_rate : ℝ) 
  (fertilizer_a_area : ℝ) 
  (fertilizer_b_rate : ℝ) 
  (fertilizer_b_area : ℝ) 
  (area_to_fertilize : ℝ) : 
  field_area = 10800 ∧ 
  fertilizer_a_rate = 150 ∧ 
  fertilizer_a_area = 3000 ∧ 
  fertilizer_b_rate = 180 ∧ 
  fertilizer_b_area = 4000 ∧ 
  area_to_fertilize = 3600 → 
  (fertilizer_a_rate * area_to_fertilize / fertilizer_a_area) + 
  (fertilizer_b_rate * area_to_fertilize / fertilizer_b_area) = 342 := by
  sorry

#check fertilizer_weight_calculation

end NUMINAMATH_CALUDE_fertilizer_weight_calculation_l1578_157810


namespace NUMINAMATH_CALUDE_bianca_drawing_time_l1578_157832

theorem bianca_drawing_time (school_time home_time total_time : ℕ) : 
  school_time = 22 → total_time = 41 → home_time = total_time - school_time → home_time = 19 :=
by sorry

end NUMINAMATH_CALUDE_bianca_drawing_time_l1578_157832


namespace NUMINAMATH_CALUDE_triangle_trig_max_value_l1578_157835

theorem triangle_trig_max_value (A B C : ℝ) (h_sum : A + B + C = Real.pi) :
  (∀ A' B' C' : ℝ, A' + B' + C' = Real.pi →
    (Real.sin A * Real.cos B + Real.sin B * Real.cos C + Real.sin C * Real.cos A)^2 ≤
    (Real.sin A' * Real.cos B' + Real.sin B' * Real.cos C' + Real.sin C' * Real.cos A')^2) →
  (Real.sin A * Real.cos B + Real.sin B * Real.cos C + Real.sin C * Real.cos A)^2 = 27 / 16 :=
by sorry

end NUMINAMATH_CALUDE_triangle_trig_max_value_l1578_157835


namespace NUMINAMATH_CALUDE_candle_count_l1578_157872

def total_candles (bedroom_candles : ℕ) (additional_candles : ℕ) : ℕ :=
  bedroom_candles + (bedroom_candles / 2) + additional_candles

theorem candle_count : total_candles 20 20 = 50 := by
  sorry

end NUMINAMATH_CALUDE_candle_count_l1578_157872


namespace NUMINAMATH_CALUDE_school_capacity_l1578_157839

theorem school_capacity (total_classrooms : ℕ) 
  (classrooms_with_30_desks : ℕ) 
  (classrooms_with_25_desks : ℕ) 
  (desks_per_classroom_30 : ℕ) 
  (desks_per_classroom_25 : ℕ) : ℕ :=
  
  have h1 : total_classrooms = 15 := by sorry
  have h2 : classrooms_with_30_desks = total_classrooms / 3 := by sorry
  have h3 : classrooms_with_25_desks = total_classrooms - classrooms_with_30_desks := by sorry
  have h4 : desks_per_classroom_30 = 30 := by sorry
  have h5 : desks_per_classroom_25 = 25 := by sorry

  let total_desks := 
    classrooms_with_30_desks * desks_per_classroom_30 + 
    classrooms_with_25_desks * desks_per_classroom_25

  have h_result : total_desks = 400 := by sorry

  total_desks

end NUMINAMATH_CALUDE_school_capacity_l1578_157839


namespace NUMINAMATH_CALUDE_proper_subsets_without_two_eq_l1578_157893

def S : Set ℕ := {1, 2, 3, 4}

def proper_subsets_without_two : Set (Set ℕ) :=
  {A | A ⊂ S ∧ 2 ∉ A}

theorem proper_subsets_without_two_eq :
  proper_subsets_without_two = {∅, {1}, {3}, {4}, {1, 3}, {1, 4}, {3, 4}, {1, 3, 4}} := by
  sorry

end NUMINAMATH_CALUDE_proper_subsets_without_two_eq_l1578_157893


namespace NUMINAMATH_CALUDE_heartsuit_three_four_l1578_157895

-- Define the heartsuit operation
def heartsuit (x y : ℝ) : ℝ := 4 * x + 6 * y

-- Theorem statement
theorem heartsuit_three_four : heartsuit 3 4 = 36 := by
  sorry

end NUMINAMATH_CALUDE_heartsuit_three_four_l1578_157895


namespace NUMINAMATH_CALUDE_car_trip_distance_l1578_157890

theorem car_trip_distance (D : ℝ) 
  (h1 : D / 2 + D / 2 = D)  -- First stop at 1/2 of total distance
  (h2 : D / 2 - (D / 2) / 4 + (D / 2) / 4 = D / 2)  -- Second stop at 1/4 of remaining distance
  (h3 : D - D / 2 - (D / 2) / 4 = 105)  -- Remaining distance after second stop is 105 miles
  : D = 280 := by
sorry

end NUMINAMATH_CALUDE_car_trip_distance_l1578_157890


namespace NUMINAMATH_CALUDE_mindys_tax_rate_l1578_157856

theorem mindys_tax_rate 
  (morks_tax_rate : ℝ) 
  (mindys_income_multiplier : ℝ) 
  (combined_tax_rate : ℝ) 
  (h1 : morks_tax_rate = 0.45)
  (h2 : mindys_income_multiplier = 4)
  (h3 : combined_tax_rate = 0.21) :
  let mindys_tax_rate := 
    (combined_tax_rate * (1 + mindys_income_multiplier) - morks_tax_rate) / mindys_income_multiplier
  mindys_tax_rate = 0.15 := by
sorry

end NUMINAMATH_CALUDE_mindys_tax_rate_l1578_157856


namespace NUMINAMATH_CALUDE_pizza_expense_proof_l1578_157855

/-- Proves that given a total expense of $465 on pizzas in May (31 days),
    and assuming equal daily consumption, the daily expense on pizzas is $15. -/
theorem pizza_expense_proof (total_expense : ℕ) (days_in_may : ℕ) (daily_expense : ℕ) :
  total_expense = 465 →
  days_in_may = 31 →
  daily_expense * days_in_may = total_expense →
  daily_expense = 15 := by
sorry

end NUMINAMATH_CALUDE_pizza_expense_proof_l1578_157855


namespace NUMINAMATH_CALUDE_complex_square_expansion_l1578_157803

theorem complex_square_expansion (x y c : ℝ) : 
  (x + Complex.I * y + c)^2 = x^2 + c^2 - y^2 + 2*c*x + Complex.I * (2*x*y + 2*c*y) := by
  sorry

end NUMINAMATH_CALUDE_complex_square_expansion_l1578_157803


namespace NUMINAMATH_CALUDE_bananas_profit_theorem_l1578_157853

/-- The number of pounds of bananas purchased by the grocer -/
def bananas_purchased : ℝ := 84

/-- The purchase price in dollars for 3 pounds of bananas -/
def purchase_price : ℝ := 0.50

/-- The selling price in dollars for 4 pounds of bananas -/
def selling_price : ℝ := 1.00

/-- The total profit in dollars -/
def total_profit : ℝ := 7.00

/-- Theorem stating that the number of pounds of bananas purchased is correct -/
theorem bananas_profit_theorem :
  bananas_purchased * (selling_price / 4 - purchase_price / 3) = total_profit :=
by sorry

end NUMINAMATH_CALUDE_bananas_profit_theorem_l1578_157853


namespace NUMINAMATH_CALUDE_pie_eating_contest_l1578_157896

theorem pie_eating_contest (first_student second_student : ℚ) :
  first_student = 7/8 ∧ second_student = 5/6 →
  first_student - second_student = 1/24 := by
  sorry

end NUMINAMATH_CALUDE_pie_eating_contest_l1578_157896


namespace NUMINAMATH_CALUDE_glue_per_clipping_l1578_157822

theorem glue_per_clipping 
  (num_friends : ℕ) 
  (clippings_per_friend : ℕ) 
  (total_glue_drops : ℕ) : 
  num_friends = 7 → 
  clippings_per_friend = 3 → 
  total_glue_drops = 126 → 
  total_glue_drops / (num_friends * clippings_per_friend) = 6 := by
  sorry

end NUMINAMATH_CALUDE_glue_per_clipping_l1578_157822


namespace NUMINAMATH_CALUDE_cans_per_bag_l1578_157802

theorem cans_per_bag (total_bags : ℕ) (total_cans : ℕ) (h1 : total_bags = 9) (h2 : total_cans = 72) :
  total_cans / total_bags = 8 :=
by sorry

end NUMINAMATH_CALUDE_cans_per_bag_l1578_157802


namespace NUMINAMATH_CALUDE_optimal_purchase_plan_l1578_157840

/-- Represents the daily carrying capacity and cost of robots --/
structure Robot where
  capacity : ℕ  -- daily carrying capacity in tons
  cost : ℕ      -- cost in yuan

/-- Represents the purchase plan for robots --/
structure PurchasePlan where
  typeA : ℕ  -- number of type A robots
  typeB : ℕ  -- number of type B robots

/-- Calculates the total daily carrying capacity for a given purchase plan --/
def totalCapacity (a b : Robot) (plan : PurchasePlan) : ℕ :=
  plan.typeA * a.capacity + plan.typeB * b.capacity

/-- Calculates the total cost for a given purchase plan --/
def totalCost (a b : Robot) (plan : PurchasePlan) : ℕ :=
  plan.typeA * a.cost + plan.typeB * b.cost

/-- Theorem stating the optimal purchase plan --/
theorem optimal_purchase_plan (a b : Robot) :
  a.capacity = b.capacity + 20 →
  3 * a.capacity + 2 * b.capacity = 460 →
  a.cost = 30000 →
  b.cost = 20000 →
  (∀ plan : PurchasePlan, plan.typeA + plan.typeB = 20 →
    totalCapacity a b plan ≥ 1820 →
    totalCost a b plan ≥ 510000) ∧
  (∃ plan : PurchasePlan, plan.typeA = 11 ∧ plan.typeB = 9 ∧
    totalCapacity a b plan ≥ 1820 ∧
    totalCost a b plan = 510000) :=
by sorry

end NUMINAMATH_CALUDE_optimal_purchase_plan_l1578_157840


namespace NUMINAMATH_CALUDE_complement_of_A_union_B_l1578_157850

open Set

def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B : Set ℝ := {x | x ≥ 0}

theorem complement_of_A_union_B :
  (A ∪ B)ᶜ = {x : ℝ | x ≤ -1} :=
by sorry

end NUMINAMATH_CALUDE_complement_of_A_union_B_l1578_157850


namespace NUMINAMATH_CALUDE_sequence_problem_l1578_157833

theorem sequence_problem (a : ℕ → ℤ) (h1 : a 5 = 14) (h2 : ∀ n : ℕ, a (n + 1) - a n = n + 1) : a 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sequence_problem_l1578_157833


namespace NUMINAMATH_CALUDE_circle_area_from_circumference_l1578_157899

theorem circle_area_from_circumference : 
  ∀ (r : ℝ), 2 * π * r = 24 * π → π * r^2 = 144 * π := by
  sorry

end NUMINAMATH_CALUDE_circle_area_from_circumference_l1578_157899


namespace NUMINAMATH_CALUDE_expression_simplification_l1578_157894

theorem expression_simplification (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  ((x^3 + 1) / x * (y^3 + 1) / y) - ((x^3 - 1) / y * (y^3 - 1) / x) = 2*x^2 + 2*y^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1578_157894


namespace NUMINAMATH_CALUDE_stones_per_bracelet_l1578_157836

theorem stones_per_bracelet (total_stones : Float) (num_bracelets : Float) 
  (h1 : total_stones = 88.0)
  (h2 : num_bracelets = 8.0) :
  total_stones / num_bracelets = 11.0 := by
  sorry

end NUMINAMATH_CALUDE_stones_per_bracelet_l1578_157836


namespace NUMINAMATH_CALUDE_wolf_winning_strategy_wolf_wins_l1578_157883

/-- Represents a player in the game -/
inductive Player
| Wolf
| Hare

/-- Represents the state of the game board -/
structure GameState where
  number : Nat
  currentPlayer : Player

/-- Defines a valid move in the game -/
def isValidMove (n : Nat) (digit : Nat) : Prop :=
  digit > 0 ∧ digit ≤ 9 ∧ digit ≤ n

/-- Applies a move to the game state -/
def applyMove (state : GameState) (digit : Nat) : GameState :=
  { number := state.number - digit,
    currentPlayer := match state.currentPlayer with
      | Player.Wolf => Player.Hare
      | Player.Hare => Player.Wolf }

/-- Defines the winning condition -/
def isWinningState (state : GameState) : Prop :=
  state.number = 0

/-- Theorem: There exists a winning strategy for Wolf starting with 1234 -/
theorem wolf_winning_strategy :
  ∃ (strategy : GameState → Nat),
    (∀ (state : GameState), isValidMove state.number (strategy state)) →
    (∀ (state : GameState),
      state.currentPlayer = Player.Wolf →
      isWinningState (applyMove state (strategy state)) ∨
      ∃ (hareMove : Nat),
        isValidMove (applyMove state (strategy state)).number hareMove →
        isWinningState (applyMove (applyMove state (strategy state)) hareMove)) :=
sorry

/-- The initial game state -/
def initialState : GameState :=
  { number := 1234, currentPlayer := Player.Wolf }

/-- Corollary: Wolf wins the game starting from 1234 -/
theorem wolf_wins : ∃ (moves : List Nat), 
  isWinningState (moves.foldl applyMove initialState) ∧
  moves.length % 2 = 0 :=
sorry

end NUMINAMATH_CALUDE_wolf_winning_strategy_wolf_wins_l1578_157883


namespace NUMINAMATH_CALUDE_fish_count_l1578_157876

def billy_fish : ℕ := 10

def tony_fish (billy : ℕ) : ℕ := 3 * billy

def sarah_fish (tony : ℕ) : ℕ := tony + 5

def bobby_fish (sarah : ℕ) : ℕ := 2 * sarah

def total_fish (billy tony sarah bobby : ℕ) : ℕ := billy + tony + sarah + bobby

theorem fish_count :
  total_fish billy_fish 
             (tony_fish billy_fish) 
             (sarah_fish (tony_fish billy_fish)) 
             (bobby_fish (sarah_fish (tony_fish billy_fish))) = 145 := by
  sorry

end NUMINAMATH_CALUDE_fish_count_l1578_157876


namespace NUMINAMATH_CALUDE_sugar_for_frosting_l1578_157834

theorem sugar_for_frosting (total_sugar cake_sugar frosting_sugar : ℚ) : 
  total_sugar = 0.8 →
  cake_sugar = 0.2 →
  total_sugar = cake_sugar + frosting_sugar →
  frosting_sugar = 0.6 := by
sorry

end NUMINAMATH_CALUDE_sugar_for_frosting_l1578_157834


namespace NUMINAMATH_CALUDE_solution_to_equation_l1578_157887

theorem solution_to_equation : ∃! (x : ℝ), x ≠ 0 ∧ (7 * x)^4 = (14 * x)^3 ∧ x = 8/7 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l1578_157887


namespace NUMINAMATH_CALUDE_unique_right_triangle_18_l1578_157858

/-- Represents a triple of positive integers (a, b, c) that form a right triangle with perimeter 18. -/
structure RightTriangle18 where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  right_triangle : a^2 + b^2 = c^2
  perimeter_18 : a + b + c = 18

/-- There exists exactly one right triangle with integer side lengths and perimeter 18. -/
theorem unique_right_triangle_18 : ∃! t : RightTriangle18, True := by sorry

end NUMINAMATH_CALUDE_unique_right_triangle_18_l1578_157858


namespace NUMINAMATH_CALUDE_correct_average_l1578_157878

theorem correct_average (n : Nat) (incorrect_avg : ℚ) (incorrect_num : ℚ) (correct_num : ℚ) :
  n = 10 →
  incorrect_avg = 16 →
  incorrect_num = 26 →
  correct_num = 46 →
  (n : ℚ) * incorrect_avg + (correct_num - incorrect_num) = n * 18 :=
by sorry

end NUMINAMATH_CALUDE_correct_average_l1578_157878


namespace NUMINAMATH_CALUDE_calculation_proof_l1578_157827

theorem calculation_proof : 
  (5^(2/3) - 5^(3/2)) / 5^(1/2) = 60 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1578_157827


namespace NUMINAMATH_CALUDE_decagon_perimeter_decagon_perimeter_30_l1578_157881

/-- The perimeter of a regular decagon with side length 3 units is 30 units. -/
theorem decagon_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun (num_sides : ℝ) (side_length : ℝ) (perimeter : ℝ) =>
    num_sides = 10 ∧ side_length = 3 → perimeter = num_sides * side_length

/-- The theorem applied to our specific case. -/
theorem decagon_perimeter_30 : decagon_perimeter 10 3 30 := by
  sorry

end NUMINAMATH_CALUDE_decagon_perimeter_decagon_perimeter_30_l1578_157881


namespace NUMINAMATH_CALUDE_target_hit_probability_l1578_157852

theorem target_hit_probability 
  (prob_A : ℝ) 
  (prob_B : ℝ) 
  (h1 : prob_A = 1/2) 
  (h2 : prob_B = 1/3) : 
  1 - (1 - prob_A) * (1 - prob_B) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_target_hit_probability_l1578_157852


namespace NUMINAMATH_CALUDE_max_n_with_special_divisors_l1578_157870

theorem max_n_with_special_divisors (N : ℕ) : 
  (∃ (d : ℕ), d ∣ N ∧ d ≠ 1 ∧ d ≠ N ∧
   (∃ (a b : ℕ), a ∣ N ∧ b ∣ N ∧ a < b ∧
    (∀ (x : ℕ), x ∣ N → x < a ∨ x > b) ∧
    b = 21 * d)) →
  N ≤ 441 :=
sorry

end NUMINAMATH_CALUDE_max_n_with_special_divisors_l1578_157870


namespace NUMINAMATH_CALUDE_sum_of_fractions_inequality_l1578_157879

theorem sum_of_fractions_inequality (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq_one : a + b + c + d = 1) : 
  b * c * d / (1 - a)^2 + c * d * a / (1 - b)^2 + 
  d * a * b / (1 - c)^2 + a * b * c / (1 - d)^2 ≤ 1/9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_inequality_l1578_157879


namespace NUMINAMATH_CALUDE_christmas_play_volunteers_l1578_157866

theorem christmas_play_volunteers 
  (total_needed : ℕ) 
  (num_classes : ℕ) 
  (teachers_volunteered : ℕ) 
  (more_needed : ℕ) 
  (h1 : total_needed = 50) 
  (h2 : num_classes = 6) 
  (h3 : teachers_volunteered = 13) 
  (h4 : more_needed = 7) :
  (total_needed - teachers_volunteered - more_needed) / num_classes = 5 := by
  sorry

end NUMINAMATH_CALUDE_christmas_play_volunteers_l1578_157866


namespace NUMINAMATH_CALUDE_workshop_workers_l1578_157885

/-- The total number of workers in a workshop given specific salary conditions -/
theorem workshop_workers (average_salary : ℝ) (technician_salary : ℝ) (other_salary : ℝ) 
  (num_technicians : ℕ) :
  average_salary = 8000 →
  technician_salary = 12000 →
  other_salary = 6000 →
  num_technicians = 7 →
  ∃ (total_workers : ℕ), 
    (total_workers : ℝ) * average_salary = 
      (num_technicians : ℝ) * technician_salary + 
      ((total_workers - num_technicians) : ℝ) * other_salary ∧
    total_workers = 21 :=
by sorry

end NUMINAMATH_CALUDE_workshop_workers_l1578_157885


namespace NUMINAMATH_CALUDE_division_problem_l1578_157828

theorem division_problem (x y z : ℝ) (h1 : x / y = 3) (h2 : y / z = 5/2) : 
  z / x = 2/15 := by sorry

end NUMINAMATH_CALUDE_division_problem_l1578_157828


namespace NUMINAMATH_CALUDE_price_after_discounts_l1578_157808

/-- The original price of an article before discounts -/
def original_price : ℝ := 70.59

/-- The final price after discounts -/
def final_price : ℝ := 36

/-- The first discount rate -/
def discount1 : ℝ := 0.15

/-- The second discount rate -/
def discount2 : ℝ := 0.25

/-- The third discount rate -/
def discount3 : ℝ := 0.20

/-- Theorem stating that the original price results in the final price after applying the discounts -/
theorem price_after_discounts : 
  ∃ ε > 0, abs (original_price * (1 - discount1) * (1 - discount2) * (1 - discount3) - final_price) < ε :=
sorry

end NUMINAMATH_CALUDE_price_after_discounts_l1578_157808


namespace NUMINAMATH_CALUDE_rectangle_length_proof_l1578_157806

theorem rectangle_length_proof (b : ℝ) (h1 : b > 0) : 
  (2 * b - 5) * (b + 5) = 2 * b^2 + 75 → 2 * b = 40 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_proof_l1578_157806


namespace NUMINAMATH_CALUDE_rhombus_constructible_l1578_157819

/-- Represents a rhombus in 2D space -/
structure Rhombus where
  /-- Side length of the rhombus -/
  side : ℝ
  /-- Difference between the two diagonals -/
  diag_diff : ℝ
  /-- Assumption that side length is positive -/
  side_pos : side > 0
  /-- Assumption that diagonal difference is non-negative and less than twice the side length -/
  diag_diff_valid : 0 ≤ diag_diff ∧ diag_diff < 2 * side

/-- Theorem stating that a rhombus can be constructed given a side length and diagonal difference -/
theorem rhombus_constructible (a : ℝ) (d : ℝ) (h1 : a > 0) (h2 : 0 ≤ d ∧ d < 2 * a) :
  ∃ (r : Rhombus), r.side = a ∧ r.diag_diff = d :=
sorry

end NUMINAMATH_CALUDE_rhombus_constructible_l1578_157819


namespace NUMINAMATH_CALUDE_block_weight_difference_l1578_157859

/-- Given two blocks with different weights, prove the difference between their weights. -/
theorem block_weight_difference (yellow_weight green_weight : ℝ)
  (h1 : yellow_weight = 0.6)
  (h2 : green_weight = 0.4) :
  yellow_weight - green_weight = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_block_weight_difference_l1578_157859


namespace NUMINAMATH_CALUDE_polynomial_root_implies_coefficients_l1578_157838

theorem polynomial_root_implies_coefficients 
  (a b : ℝ) 
  (h : (Complex.I : ℂ) ^ 2 = -1) 
  (root : (2 : ℂ) - Complex.I ∈ {z : ℂ | z^3 + a*z^2 + b*z - 6 = 0}) : 
  a = -26/5 ∧ b = 49/5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_root_implies_coefficients_l1578_157838


namespace NUMINAMATH_CALUDE_polynomial_symmetry_l1578_157868

/-- A polynomial is symmetric with respect to a point if and only if it has a specific form. -/
theorem polynomial_symmetry (P : ℝ → ℝ) (a b : ℝ) :
  (∀ x, P (2*a - x) = 2*b - P x) ↔
  (∃ Q : ℝ → ℝ, ∀ x, P x = b + (x - a) * Q ((x - a)^2)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_symmetry_l1578_157868


namespace NUMINAMATH_CALUDE_monomial_coefficient_and_degree_l1578_157854

/-- Represents a monomial with coefficient and variables -/
structure Monomial where
  coeff : ℚ
  vars : List (Char × ℕ)

/-- Calculate the degree of a monomial -/
def monomialDegree (m : Monomial) : ℕ :=
  m.vars.foldl (fun acc (_, exp) => acc + exp) 0

/-- The monomial -2/3 * a * b^2 -/
def mono : Monomial :=
  { coeff := -2/3
  , vars := [('a', 1), ('b', 2)] }

theorem monomial_coefficient_and_degree :
  mono.coeff = -2/3 ∧ monomialDegree mono = 3 := by
  sorry

end NUMINAMATH_CALUDE_monomial_coefficient_and_degree_l1578_157854


namespace NUMINAMATH_CALUDE_point_on_y_axis_l1578_157863

/-- A point lies on the y-axis if and only if its x-coordinate is 0 -/
def lies_on_y_axis (x y : ℝ) : Prop := x = 0

/-- The theorem states that if the point (a+1, a-1) lies on the y-axis, then a = -1 -/
theorem point_on_y_axis (a : ℝ) : lies_on_y_axis (a + 1) (a - 1) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_point_on_y_axis_l1578_157863


namespace NUMINAMATH_CALUDE_bank_deposit_is_50_l1578_157817

def total_income : ℚ := 200

def provident_fund_ratio : ℚ := 1 / 16
def insurance_premium_ratio : ℚ := 1 / 15
def domestic_needs_ratio : ℚ := 5 / 7

def provident_fund : ℚ := provident_fund_ratio * total_income
def remaining_after_provident_fund : ℚ := total_income - provident_fund

def insurance_premium : ℚ := insurance_premium_ratio * remaining_after_provident_fund
def remaining_after_insurance : ℚ := remaining_after_provident_fund - insurance_premium

def domestic_needs : ℚ := domestic_needs_ratio * remaining_after_insurance
def bank_deposit : ℚ := remaining_after_insurance - domestic_needs

theorem bank_deposit_is_50 : bank_deposit = 50 := by
  sorry

end NUMINAMATH_CALUDE_bank_deposit_is_50_l1578_157817


namespace NUMINAMATH_CALUDE_equation_three_solutions_l1578_157865

theorem equation_three_solutions :
  ∃ (s : Finset ℝ), (s.card = 3) ∧ 
  (∀ x ∈ s, (x^2 - 6*x + 9) / (x - 1) - (3 - x) / (x^2 - 1) = 0) ∧
  (∀ y : ℝ, (y^2 - 6*y + 9) / (y - 1) - (3 - y) / (y^2 - 1) = 0 → y ∈ s) := by
  sorry

end NUMINAMATH_CALUDE_equation_three_solutions_l1578_157865


namespace NUMINAMATH_CALUDE_highest_probability_greater_than_two_l1578_157823

def fair_dice_probability (event : Finset Nat) : Rat :=
  (event.card : Rat) / 6

theorem highest_probability_greater_than_two :
  let less_than_two : Finset Nat := {1}
  let greater_than_two : Finset Nat := {3, 4, 5, 6}
  let even_numbers : Finset Nat := {2, 4, 6}
  fair_dice_probability greater_than_two > fair_dice_probability even_numbers ∧
  fair_dice_probability greater_than_two > fair_dice_probability less_than_two :=
by sorry

end NUMINAMATH_CALUDE_highest_probability_greater_than_two_l1578_157823


namespace NUMINAMATH_CALUDE_first_divisor_l1578_157849

theorem first_divisor (k : ℕ) (h1 : k > 0) (h2 : k % 5 = 2) (h3 : k % 6 = 5) (h4 : k % 7 = 3) (h5 : k < 42) :
  min 5 (min 6 7) = 5 :=
by sorry

end NUMINAMATH_CALUDE_first_divisor_l1578_157849


namespace NUMINAMATH_CALUDE_fuel_tank_capacity_l1578_157873

theorem fuel_tank_capacity : ∃ (C : ℝ), C = 204 ∧ C > 0 := by
  -- Define the ethanol content of fuels A and B
  let ethanol_A : ℝ := 0.12
  let ethanol_B : ℝ := 0.16

  -- Define the volume of fuel A added
  let volume_A : ℝ := 66

  -- Define the total ethanol volume in the full tank
  let total_ethanol : ℝ := 30

  -- The capacity C satisfies the equation:
  -- ethanol_A * volume_A + ethanol_B * (C - volume_A) = total_ethanol
  
  sorry

end NUMINAMATH_CALUDE_fuel_tank_capacity_l1578_157873


namespace NUMINAMATH_CALUDE_rows_per_wall_is_fifty_l1578_157892

/-- The number of bricks in a single row of each wall -/
def bricks_per_row : ℕ := 30

/-- The total number of bricks used for both walls -/
def total_bricks : ℕ := 3000

/-- The number of rows in each wall -/
def rows_per_wall : ℕ := total_bricks / (2 * bricks_per_row)

theorem rows_per_wall_is_fifty : rows_per_wall = 50 := by
  sorry

end NUMINAMATH_CALUDE_rows_per_wall_is_fifty_l1578_157892


namespace NUMINAMATH_CALUDE_sin_product_zero_l1578_157848

theorem sin_product_zero : Real.sin (12 * π / 180) * Real.sin (36 * π / 180) * Real.sin (60 * π / 180) * Real.sin (84 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_product_zero_l1578_157848


namespace NUMINAMATH_CALUDE_sqrt_205_between_14_and_15_l1578_157800

theorem sqrt_205_between_14_and_15 : 14 < Real.sqrt 205 ∧ Real.sqrt 205 < 15 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_205_between_14_and_15_l1578_157800


namespace NUMINAMATH_CALUDE_z_as_percentage_of_x_l1578_157880

theorem z_as_percentage_of_x (x y z : ℝ) 
  (h1 : 0.45 * z = 0.90 * y) 
  (h2 : y = 0.75 * x) : 
  z = 1.5 * x := by
sorry

end NUMINAMATH_CALUDE_z_as_percentage_of_x_l1578_157880


namespace NUMINAMATH_CALUDE_zeta_power_sum_l1578_157864

theorem zeta_power_sum (ζ₁ ζ₂ ζ₃ : ℂ) 
  (h1 : ζ₁ + ζ₂ + ζ₃ = 2)
  (h2 : ζ₁^2 + ζ₂^2 + ζ₃^2 = 6)
  (h3 : ζ₁^3 + ζ₂^3 + ζ₃^3 = 20) :
  ζ₁^5 + ζ₂^5 + ζ₃^5 = 54 := by
  sorry

end NUMINAMATH_CALUDE_zeta_power_sum_l1578_157864


namespace NUMINAMATH_CALUDE_pencil_multiple_l1578_157867

theorem pencil_multiple (reeta_pencils : ℕ) (total_pencils : ℕ) (anika_pencils : ℕ → ℕ) :
  reeta_pencils = 20 →
  total_pencils = 64 →
  (∀ M : ℕ, anika_pencils M = 20 * M + 4) →
  ∃ M : ℕ, M = 2 ∧ anika_pencils M + reeta_pencils = total_pencils :=
by sorry

end NUMINAMATH_CALUDE_pencil_multiple_l1578_157867


namespace NUMINAMATH_CALUDE_ratio_problem_l1578_157841

theorem ratio_problem (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 4 / 3)
  (hdb : d / b = 1 / 5) :
  a / c = 75 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l1578_157841


namespace NUMINAMATH_CALUDE_perpendicular_planes_l1578_157862

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perp_line_line : Line → Line → Prop)
variable (perp_line_plane : Line → Plane → Prop)
variable (perp_plane_plane : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_planes 
  (a b : Line) 
  (ξ ζ : Plane) 
  (diff_lines : a ≠ b) 
  (diff_planes : ξ ≠ ζ) 
  (h1 : perp_line_line a b) 
  (h2 : perp_line_plane a ξ) 
  (h3 : perp_line_plane b ζ) : 
  perp_plane_plane ξ ζ :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_l1578_157862


namespace NUMINAMATH_CALUDE_probability_green_is_25_56_l1578_157842

/-- Represents a container with red and green balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- The probability of selecting a green ball from the given containers -/
def probability_green (containers : List Container) : ℚ :=
  let total_containers := containers.length
  let prob_green_per_container := containers.map (fun c => c.green / (c.red + c.green))
  (prob_green_per_container.sum) / total_containers

/-- The containers as described in the problem -/
def problem_containers : List Container :=
  [⟨8, 4⟩, ⟨2, 5⟩, ⟨2, 5⟩, ⟨4, 4⟩]

/-- The theorem stating the probability of selecting a green ball -/
theorem probability_green_is_25_56 :
  probability_green problem_containers = 25 / 56 := by
  sorry


end NUMINAMATH_CALUDE_probability_green_is_25_56_l1578_157842


namespace NUMINAMATH_CALUDE_fruit_combinations_l1578_157824

theorem fruit_combinations (n r : ℕ) (h1 : n = 5) (h2 : r = 2) :
  (n + r - 1).choose r = 15 := by
sorry

end NUMINAMATH_CALUDE_fruit_combinations_l1578_157824


namespace NUMINAMATH_CALUDE_product_of_divisors_of_product_of_divisors_of_2005_l1578_157851

def divisors (n : ℕ) : Finset ℕ := sorry

def divisor_product (n : ℕ) : ℕ := (divisors n).prod id

theorem product_of_divisors_of_product_of_divisors_of_2005 :
  divisor_product (divisor_product 2005) = 2005^9 := by sorry

end NUMINAMATH_CALUDE_product_of_divisors_of_product_of_divisors_of_2005_l1578_157851


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l1578_157843

theorem triangle_area_theorem (x : ℝ) (h1 : x > 0) : 
  (1/2 : ℝ) * x * (3*x) = 72 → x = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_theorem_l1578_157843


namespace NUMINAMATH_CALUDE_cutlery_added_l1578_157837

def initial_forks : ℕ := 6

def initial_knives (forks : ℕ) : ℕ := forks + 9

def initial_spoons (knives : ℕ) : ℕ := 2 * knives

def initial_teaspoons (forks : ℕ) : ℕ := forks / 2

def total_initial_cutlery (forks knives spoons teaspoons : ℕ) : ℕ :=
  forks + knives + spoons + teaspoons

def final_total_cutlery : ℕ := 62

theorem cutlery_added :
  final_total_cutlery - total_initial_cutlery initial_forks
    (initial_knives initial_forks)
    (initial_spoons (initial_knives initial_forks))
    (initial_teaspoons initial_forks) = 8 := by
  sorry

end NUMINAMATH_CALUDE_cutlery_added_l1578_157837


namespace NUMINAMATH_CALUDE_parabola_directrix_parameter_l1578_157811

/-- Given a parabola with equation x² = ay and directrix y = 1, prove that a = -4 -/
theorem parabola_directrix_parameter (a : ℝ) : 
  (∀ x y : ℝ, x^2 = a*y) →  -- Parabola equation
  (1 = -a/4) →              -- Relation between 'a' and directrix
  a = -4 := by
sorry

end NUMINAMATH_CALUDE_parabola_directrix_parameter_l1578_157811


namespace NUMINAMATH_CALUDE_special_list_median_l1578_157821

/-- The sum of the first n positive integers -/
def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The list of integers where each n appears n times for 1 ≤ n ≤ 150 -/
def special_list : List ℕ := sorry

/-- The median of a list is the middle value when the list is sorted -/
def median (l : List ℕ) : ℕ := sorry

theorem special_list_median :
  median special_list = 107 := by sorry

end NUMINAMATH_CALUDE_special_list_median_l1578_157821


namespace NUMINAMATH_CALUDE_double_age_in_two_years_l1578_157829

/-- The number of years until a man's age is twice his son's age -/
def years_until_double_age (son_age : ℕ) (age_difference : ℕ) : ℕ :=
  let man_age := son_age + age_difference
  (man_age - 2 * son_age) / (2 - 1)

/-- Theorem: Given the son's age is 35 and the age difference is 37, 
    the number of years until the man's age is twice his son's age is 2 -/
theorem double_age_in_two_years (son_age : ℕ) (age_difference : ℕ) 
  (h1 : son_age = 35) (h2 : age_difference = 37) : 
  years_until_double_age son_age age_difference = 2 := by
  sorry

end NUMINAMATH_CALUDE_double_age_in_two_years_l1578_157829


namespace NUMINAMATH_CALUDE_symmetry_wrt_y_axis_l1578_157874

/-- Given two real numbers a and b such that lg a + lg b = 0, a ≠ 1, and b ≠ 1,
    the functions f(x) = a^x and g(x) = b^x are symmetric with respect to the y-axis. -/
theorem symmetry_wrt_y_axis (a b : ℝ) (ha : a ≠ 1) (hb : b ≠ 1) 
    (h : Real.log a + Real.log b = 0) :
  ∀ x : ℝ, a^(-x) = b^x := by
  sorry

end NUMINAMATH_CALUDE_symmetry_wrt_y_axis_l1578_157874


namespace NUMINAMATH_CALUDE_divide_by_seven_l1578_157891

theorem divide_by_seven (x : ℚ) (h : x = 5/2) : x / 7 = 5/14 := by
  sorry

end NUMINAMATH_CALUDE_divide_by_seven_l1578_157891


namespace NUMINAMATH_CALUDE_train_departure_time_l1578_157826

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Calculates the difference between two times in minutes -/
def timeDiff (t1 t2 : Time) : Nat :=
  (t1.hours * 60 + t1.minutes) - (t2.hours * 60 + t2.minutes)

theorem train_departure_time 
  (arrival : Time)
  (journey_duration : Nat)
  (h_arrival : arrival.hours = 10 ∧ arrival.minutes = 0)
  (h_duration : journey_duration = 15) :
  ∃ (departure : Time), 
    timeDiff arrival departure = journey_duration ∧ 
    departure.hours = 9 ∧ 
    departure.minutes = 45 := by
  sorry

end NUMINAMATH_CALUDE_train_departure_time_l1578_157826


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1578_157825

theorem polynomial_divisibility (A B : ℝ) : 
  (∀ x : ℂ, x^2 + x + 1 = 0 → x^103 + A*x^2 + B = 0) → 
  A + B = 2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1578_157825


namespace NUMINAMATH_CALUDE_line_equation_with_given_area_l1578_157815

/-- Given a line passing through points (a, 0) and (b, 0) where b > a, 
    cutting a triangular region from the first quadrant with area S,
    prove that the equation of the line is 0 = -2Sx + (b-a)^2y + 2Sa - 2Sb -/
theorem line_equation_with_given_area (a b S : ℝ) (h1 : b > a) (h2 : S > 0) :
  ∃ (f : ℝ → ℝ), 
    (∀ x, f x = 0 ↔ -2 * S * x + (b - a)^2 * x + 2 * S * a - 2 * S * b = 0) ∧
    f a = 0 ∧ 
    f b = 0 ∧
    (∃ k, k > 0 ∧ f k > 0 ∧ (k - 0) * (b - a) / 2 = S) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_with_given_area_l1578_157815


namespace NUMINAMATH_CALUDE_largest_consecutive_sum_28_l1578_157830

/-- The sum of n consecutive positive integers starting from a -/
def sumConsecutive (n : ℕ) (a : ℕ) : ℕ := n * (2 * a + n - 1) / 2

/-- Predicate to check if a sequence of n consecutive integers starting from a sums to 28 -/
def isValidSequence (n : ℕ) (a : ℕ) : Prop :=
  a > 0 ∧ sumConsecutive n a = 28

theorem largest_consecutive_sum_28 :
  (∃ a : ℕ, isValidSequence 7 a) ∧
  (∀ n : ℕ, n > 7 → ¬∃ a : ℕ, isValidSequence n a) :=
sorry

end NUMINAMATH_CALUDE_largest_consecutive_sum_28_l1578_157830


namespace NUMINAMATH_CALUDE_identical_sequences_l1578_157812

/-- Given two sequences of n real numbers where the first is strictly increasing,
    and their element-wise sum is strictly increasing,
    prove that the sequences are identical. -/
theorem identical_sequences
  (n : ℕ)
  (a b : Fin n → ℝ)
  (h_a_increasing : ∀ i j : Fin n, i < j → a i < a j)
  (h_sum_increasing : ∀ i j : Fin n, i < j → a i + b i < a j + b j) :
  a = b :=
sorry

end NUMINAMATH_CALUDE_identical_sequences_l1578_157812


namespace NUMINAMATH_CALUDE_cycle_gain_percent_l1578_157845

/-- Calculates the gain percent given the cost price and selling price -/
def gain_percent (cost_price selling_price : ℚ) : ℚ :=
  ((selling_price - cost_price) / cost_price) * 100

/-- Theorem: The gain percent is 8% when a cycle is bought for Rs. 1000 and sold for Rs. 1080 -/
theorem cycle_gain_percent :
  gain_percent 1000 1080 = 8 := by
  sorry

end NUMINAMATH_CALUDE_cycle_gain_percent_l1578_157845


namespace NUMINAMATH_CALUDE_trig_identities_l1578_157889

theorem trig_identities (x : Real) 
  (h1 : 0 < x) (h2 : x < Real.pi) 
  (h3 : Real.sin x + Real.cos x = 7/13) : 
  (Real.sin x * Real.cos x = -60/169) ∧ 
  ((5 * Real.sin x + 4 * Real.cos x) / (15 * Real.sin x - 7 * Real.cos x) = 8/43) := by
  sorry

end NUMINAMATH_CALUDE_trig_identities_l1578_157889


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l1578_157847

theorem unique_solution_quadratic (q : ℝ) (hq : q ≠ 0) :
  (∃! x, q * x^2 - 8 * x + 2 = 0) ↔ q = 8 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l1578_157847


namespace NUMINAMATH_CALUDE_cone_height_l1578_157831

/-- Given a cone with slant height 13 cm and lateral area 65π cm², prove its height is 12 cm -/
theorem cone_height (s : ℝ) (l : ℝ) (h : ℝ) : 
  s = 13 → l = 65 * Real.pi → l = Real.pi * s * (l / (Real.pi * s)) → h^2 + (l / (Real.pi * s))^2 = s^2 → h = 12 := by
  sorry

end NUMINAMATH_CALUDE_cone_height_l1578_157831


namespace NUMINAMATH_CALUDE_imaginary_power_2019_l1578_157875

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_2019 : i^2019 = -i := by sorry

end NUMINAMATH_CALUDE_imaginary_power_2019_l1578_157875
