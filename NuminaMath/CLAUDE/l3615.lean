import Mathlib

namespace unique_a_sqrt_2_l3615_361512

-- Define the set of options
def options : Set ℝ := {Real.sqrt (2/3), Real.sqrt 3, Real.sqrt 8, Real.sqrt 12}

-- Define the property of being expressible as a * √2
def is_a_sqrt_2 (x : ℝ) : Prop := ∃ (a : ℚ), x = a * Real.sqrt 2

-- Theorem statement
theorem unique_a_sqrt_2 : ∃! (x : ℝ), x ∈ options ∧ is_a_sqrt_2 x :=
sorry

end unique_a_sqrt_2_l3615_361512


namespace net_profit_calculation_l3615_361528

/-- Given the purchase price, overhead percentage, and markup, calculate the net profit --/
def calculate_net_profit (purchase_price overhead_percentage markup : ℝ) : ℝ :=
  let overhead := purchase_price * overhead_percentage
  markup - overhead

/-- Theorem stating that given the specified conditions, the net profit is $27.60 --/
theorem net_profit_calculation :
  let purchase_price : ℝ := 48
  let overhead_percentage : ℝ := 0.05
  let markup : ℝ := 30
  calculate_net_profit purchase_price overhead_percentage markup = 27.60 := by
  sorry

#eval calculate_net_profit 48 0.05 30

end net_profit_calculation_l3615_361528


namespace custom_op_value_l3615_361533

-- Define the custom operation *
def custom_op (a b : ℚ) : ℚ := 1 / a + 1 / b

-- Theorem statement
theorem custom_op_value (a b : ℚ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (sum_eq : a + b = 12) (prod_eq : a * b = 32) : 
  custom_op a b = 3 / 8 := by
  sorry

end custom_op_value_l3615_361533


namespace florist_roses_problem_l3615_361576

/-- Proves that the initial number of roses was 37 given the conditions of the problem. -/
theorem florist_roses_problem (initial_roses : ℕ) : 
  (initial_roses - 16 + 19 = 40) → initial_roses = 37 := by
  sorry

end florist_roses_problem_l3615_361576


namespace min_socks_for_ten_pairs_l3615_361538

/-- Represents the number of socks of each color in the drawer -/
structure SockDrawer :=
  (red : ℕ)
  (green : ℕ)
  (blue : ℕ)
  (black : ℕ)

/-- Calculates the minimum number of socks to draw to guarantee a certain number of pairs -/
def minSocksForPairs (drawer : SockDrawer) (numPairs : ℕ) : ℕ :=
  4 + 1 + 2 * (numPairs - 1)

/-- Theorem stating the minimum number of socks to draw for 10 pairs -/
theorem min_socks_for_ten_pairs (drawer : SockDrawer) 
  (h_red : drawer.red = 100)
  (h_green : drawer.green = 80)
  (h_blue : drawer.blue = 60)
  (h_black : drawer.black = 40) :
  minSocksForPairs drawer 10 = 23 := by
  sorry

#eval minSocksForPairs ⟨100, 80, 60, 40⟩ 10

end min_socks_for_ten_pairs_l3615_361538


namespace polygon_sides_from_triangles_l3615_361542

/-- Represents a polygon with n sides -/
structure Polygon (n : ℕ) where
  -- Add any necessary fields here

/-- Represents a point on a side of a polygon -/
structure PointOnSide (p : Polygon n) where
  -- Add any necessary fields here

/-- The number of triangles formed when connecting a point on a side to all vertices -/
def numTriangles (p : Polygon n) (point : PointOnSide p) : ℕ :=
  n - 1

theorem polygon_sides_from_triangles
  (p : Polygon n) (point : PointOnSide p)
  (h : numTriangles p point = 8) :
  n = 9 := by
  sorry

end polygon_sides_from_triangles_l3615_361542


namespace polar_coordinates_of_point_M_l3615_361543

theorem polar_coordinates_of_point_M : 
  let x : ℝ := -1
  let y : ℝ := Real.sqrt 3
  let ρ : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := Real.arccos (x / ρ)
  (ρ = 2) ∧ (θ = 2 * Real.pi / 3) := by sorry

end polar_coordinates_of_point_M_l3615_361543


namespace simplify_expression_l3615_361513

theorem simplify_expression (a b : ℝ) : 120*a - 55*a + 33*b - 7*b = 65*a + 26*b := by
  sorry

end simplify_expression_l3615_361513


namespace fraction_of_fraction_one_eighth_of_one_third_l3615_361585

theorem fraction_of_fraction (a b c d : ℚ) (hb : b ≠ 0) (hd : d ≠ 0) :
  (a / b) / (c / d) = (a * d) / (b * c) := by sorry

theorem one_eighth_of_one_third :
  (1 / 8 : ℚ) / (1 / 3 : ℚ) = 3 / 8 := by sorry

end fraction_of_fraction_one_eighth_of_one_third_l3615_361585


namespace flowers_in_basket_is_four_l3615_361514

/-- The number of flowers in each basket after planting, growth, and distribution -/
def flowers_per_basket (daughters : ℕ) (flowers_per_daughter : ℕ) (new_flowers : ℕ) (dead_flowers : ℕ) (num_baskets : ℕ) : ℕ :=
  let initial_flowers := daughters * flowers_per_daughter
  let total_flowers := initial_flowers + new_flowers
  let remaining_flowers := total_flowers - dead_flowers
  remaining_flowers / num_baskets

/-- Theorem stating that under the given conditions, each basket will contain 4 flowers -/
theorem flowers_in_basket_is_four :
  flowers_per_basket 2 5 20 10 5 = 4 := by
  sorry

end flowers_in_basket_is_four_l3615_361514


namespace calculate_expression_l3615_361560

theorem calculate_expression : (Real.pi - Real.sqrt 3) ^ 0 - 2 * Real.sin (π / 4) + |-Real.sqrt 2| + Real.sqrt 8 = 1 + 2 * Real.sqrt 2 := by
  sorry

end calculate_expression_l3615_361560


namespace find_number_l3615_361558

theorem find_number : ∃! x : ℝ, 22 * (x - 36) = 748 ∧ x = 70 := by
  sorry

end find_number_l3615_361558


namespace leftover_value_is_650_l3615_361527

/-- Represents the number of coins in a roll --/
structure RollSize where
  quarters : Nat
  dimes : Nat

/-- Represents a person's coin collection --/
structure CoinCollection where
  quarters : Nat
  dimes : Nat

/-- Calculates the value of coins in dollars --/
def coinValue (quarters dimes : Nat) : Rat :=
  (quarters * 25 + dimes * 10) / 100

/-- Calculates the number and value of leftover coins --/
def leftoverCoins (emily jack : CoinCollection) (roll : RollSize) : Rat :=
  let totalQuarters := emily.quarters + jack.quarters
  let totalDimes := emily.dimes + jack.dimes
  let leftoverQuarters := totalQuarters % roll.quarters
  let leftoverDimes := totalDimes % roll.dimes
  coinValue leftoverQuarters leftoverDimes

/-- The main theorem --/
theorem leftover_value_is_650 :
  let roll : RollSize := { quarters := 45, dimes := 60 }
  let emily : CoinCollection := { quarters := 105, dimes := 215 }
  let jack : CoinCollection := { quarters := 140, dimes := 340 }
  leftoverCoins emily jack roll = 13/2 := by sorry

end leftover_value_is_650_l3615_361527


namespace max_F_value_l3615_361591

/-- Represents a four-digit number -/
structure FourDigitNumber where
  thousands : Nat
  hundreds : Nat
  tens : Nat
  units : Nat
  is_four_digit : thousands ≥ 1 ∧ thousands ≤ 9 ∧ hundreds ≤ 9 ∧ tens ≤ 9 ∧ units ≤ 9

/-- Defines an eternal number -/
def is_eternal (m : FourDigitNumber) : Prop :=
  m.hundreds + m.tens + m.units = 12

/-- Swaps digits to create N -/
def swap_digits (m : FourDigitNumber) : FourDigitNumber :=
  { thousands := m.hundreds,
    hundreds := m.thousands,
    tens := m.units,
    units := m.tens,
    is_four_digit := by sorry }

/-- Defines the function F(M) -/
def F (m : FourDigitNumber) : Int :=
  let n := swap_digits m
  let m_value := 1000 * m.thousands + 100 * m.hundreds + 10 * m.tens + m.units
  let n_value := 1000 * n.thousands + 100 * n.hundreds + 10 * n.tens + n.units
  (m_value - n_value) / 9

/-- Main theorem -/
theorem max_F_value (m : FourDigitNumber) 
  (h_eternal : is_eternal m)
  (h_diff : m.hundreds - m.units = m.thousands)
  (h_div : (F m) % 9 = 0) :
  F m ≤ 9 ∧ ∃ (m' : FourDigitNumber), is_eternal m' ∧ m'.hundreds - m'.units = m'.thousands ∧ (F m') % 9 = 0 ∧ F m' = 9 := by
  sorry

end max_F_value_l3615_361591


namespace inscribed_cube_surface_area_l3615_361562

/-- Given a cube with a sphere inscribed within it, and another cube inscribed within that sphere,
    this theorem relates the surface area of the outer cube to the surface area of the inner cube. -/
theorem inscribed_cube_surface_area
  (outer_cube_surface_area : ℝ)
  (h_outer_surface : outer_cube_surface_area = 54)
  : ∃ (inner_cube_surface_area : ℝ),
    inner_cube_surface_area = 18 :=
by sorry

end inscribed_cube_surface_area_l3615_361562


namespace box_2_neg1_3_neg2_l3615_361577

/-- Definition of the box operation for integers a, b, c, d -/
def box (a b c d : ℤ) : ℚ := a^b - b^c + c^a + d^a

/-- Theorem stating that box(2,-1,3,-2) = 12.5 -/
theorem box_2_neg1_3_neg2 : box 2 (-1) 3 (-2) = 25/2 := by
  sorry

end box_2_neg1_3_neg2_l3615_361577


namespace train_speed_equation_l3615_361509

theorem train_speed_equation (x : ℝ) (h1 : x > 80) : 
  (353 / (x - 80) - 353 / x = 5 / 3) ↔ 
  (353 / (x - 80) - 353 / x = 100 / 60) := by sorry

end train_speed_equation_l3615_361509


namespace quadratic_sum_l3615_361551

/-- 
Given a quadratic function f(x) = -3x^2 + 18x + 108, 
there exist constants a, b, and c such that 
f(x) = a(x+b)^2 + c for all x, 
and a + b + c = 129
-/
theorem quadratic_sum (x : ℝ) : ∃ (a b c : ℝ),
  (-3 * x^2 + 18 * x + 108 = a * (x + b)^2 + c) ∧ 
  (a + b + c = 129) := by
  sorry

end quadratic_sum_l3615_361551


namespace triangle_inequality_squared_l3615_361532

theorem triangle_inequality_squared (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 + b^2 + c^2 < 2 * (a*b + b*c + c*a) := by
  sorry

end triangle_inequality_squared_l3615_361532


namespace man_speed_calculation_man_speed_proof_l3615_361593

/-- Calculates the speed of a man given the parameters of a train passing him. -/
theorem man_speed_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let relative_speed := train_length / crossing_time
  train_speed_ms - relative_speed

/-- Given the specific parameters, proves that the man's speed is approximately 0.832 m/s. -/
theorem man_speed_proof :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ 
  |man_speed_calculation 700 63 41.9966402687785 - 0.832| < ε :=
sorry

end man_speed_calculation_man_speed_proof_l3615_361593


namespace better_hay_cost_is_18_l3615_361502

/-- The cost of better quality hay per bale -/
def better_hay_cost (initial_bales : ℕ) (price_increase : ℕ) (previous_cost : ℕ) : ℕ :=
  (initial_bales * previous_cost + price_increase) / (2 * initial_bales)

/-- Proof that the cost of better quality hay is $18 per bale -/
theorem better_hay_cost_is_18 :
  better_hay_cost 10 210 15 = 18 := by
  sorry

end better_hay_cost_is_18_l3615_361502


namespace H_triple_2_l3615_361537

/-- The function H defined as H(x) = 2x - 1 for all real x -/
def H (x : ℝ) : ℝ := 2 * x - 1

/-- Theorem stating that H(H(H(2))) = 9 -/
theorem H_triple_2 : H (H (H 2)) = 9 := by
  sorry

end H_triple_2_l3615_361537


namespace inequality_solution_range_of_a_l3615_361564

-- Define the functions f and g
def f (x : ℝ) := |x - 4|
def g (x : ℝ) := |2*x + 1|

-- Theorem for the first part of the problem
theorem inequality_solution :
  ∀ x : ℝ, f x < g x ↔ x < -5 ∨ x > 1 := by sorry

-- Theorem for the second part of the problem
theorem range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, 2 * f x + g x > a * x) ↔ -4 ≤ a ∧ a < 9/4 := by sorry

end inequality_solution_range_of_a_l3615_361564


namespace quad_sum_is_six_l3615_361584

/-- A quadrilateral with given properties --/
structure Quadrilateral where
  a : ℤ
  c : ℤ
  a_pos : 0 < a
  c_pos : 0 < c
  a_gt_c : c < a
  symmetric : True  -- Represents symmetry about origin
  equal_diagonals : True  -- Represents equal diagonal lengths
  area : (2 * (a - c).natAbs * (a + c).natAbs : ℤ) = 24

/-- The sum of a and c in a quadrilateral with given properties is 6 --/
theorem quad_sum_is_six (q : Quadrilateral) : q.a + q.c = 6 := by
  sorry

end quad_sum_is_six_l3615_361584


namespace circle_diameter_endpoint_l3615_361596

/-- Given a circle with center (4, 6) and one endpoint of a diameter at (1, 2),
    the other endpoint of the diameter is at (7, 10). -/
theorem circle_diameter_endpoint (P : Set (ℝ × ℝ)) : 
  (∃ (r : ℝ), P = {p | ∃ (x y : ℝ), p = (x, y) ∧ (x - 4)^2 + (y - 6)^2 = r^2}) →
  ((1, 2) ∈ P) →
  ((7, 10) ∈ P) ∧ 
  (∀ (x y : ℝ), (x, y) ∈ P → (x - 4)^2 + (y - 6)^2 = (7 - 4)^2 + (10 - 6)^2) :=
by sorry

end circle_diameter_endpoint_l3615_361596


namespace parallel_lines_slope_l3615_361524

/-- If the lines x + 2y = 3 and nx + my = 4 are parallel, then m = 2n -/
theorem parallel_lines_slope (n m : ℝ) : 
  (∀ x y : ℝ, x + 2*y = 3 → nx + m*y = 4) →  -- Lines exist
  (∃ k : ℝ, ∀ x : ℝ, 
    (3 - x) / 2 = (4 - n*x) / m) →           -- Lines are parallel
  m = 2*n :=                                 -- Conclusion
by sorry

end parallel_lines_slope_l3615_361524


namespace investment_rate_problem_l3615_361526

theorem investment_rate_problem (total_interest amount_invested_low rate_high : ℚ) 
  (h1 : total_interest = 520)
  (h2 : amount_invested_low = 2000)
  (h3 : rate_high = 5 / 100) : 
  ∃ (rate_low : ℚ), 
    amount_invested_low * rate_low + 4 * amount_invested_low * rate_high = total_interest ∧ 
    rate_low = 6 / 100 := by
sorry

end investment_rate_problem_l3615_361526


namespace second_half_speed_l3615_361589

/-- Proves that given a journey of 224 km completed in 10 hours, where the first half is traveled at 21 km/hr, the speed for the second half of the journey is 24 km/hr. -/
theorem second_half_speed (total_distance : ℝ) (total_time : ℝ) (first_half_speed : ℝ) :
  total_distance = 224 →
  total_time = 10 →
  first_half_speed = 21 →
  let first_half_distance := total_distance / 2
  let first_half_time := first_half_distance / first_half_speed
  let second_half_time := total_time - first_half_time
  let second_half_speed := first_half_distance / second_half_time
  second_half_speed = 24 := by
  sorry

end second_half_speed_l3615_361589


namespace initial_bacteria_count_l3615_361561

/-- The number of seconds in the experiment -/
def experiment_duration : ℕ := 240

/-- The number of seconds it takes for the bacteria population to double -/
def doubling_time : ℕ := 30

/-- The number of bacteria after the experiment duration -/
def final_population : ℕ := 524288

/-- The number of times the population doubles during the experiment -/
def doubling_count : ℕ := experiment_duration / doubling_time

theorem initial_bacteria_count :
  ∃ (initial_count : ℕ), initial_count * (2 ^ doubling_count) = final_population ∧ initial_count = 2048 :=
sorry

end initial_bacteria_count_l3615_361561


namespace correct_transformation_l3615_361580

theorem correct_transformation (a b m : ℝ) : a * (m^2 + 1) = b * (m^2 + 1) → a = b := by
  sorry

end correct_transformation_l3615_361580


namespace f_properties_l3615_361536

noncomputable def f (x : ℝ) := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + Real.cos (2 * x) + 3

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ 
    ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∃ (M : ℝ), M = 5 ∧ ∀ (x : ℝ), 0 ≤ x ∧ x ≤ π/4 → f x ≤ M) ∧
  f (π/6) = 5 := by
  sorry

end f_properties_l3615_361536


namespace minimum_race_distance_l3615_361531

/-- The minimum distance a runner must travel in a race with given constraints -/
theorem minimum_race_distance (wall_length : ℝ) (distance_A : ℝ) (distance_B : ℝ) :
  wall_length = 1500 →
  distance_A = 400 →
  distance_B = 600 →
  let min_distance := Real.sqrt (wall_length ^ 2 + (distance_A + distance_B) ^ 2)
  ⌊min_distance + 0.5⌋ = 1803 := by
  sorry

end minimum_race_distance_l3615_361531


namespace regular_polygon_assembly_l3615_361521

theorem regular_polygon_assembly (interior_angle : ℝ) (h1 : interior_angle = 150) :
  ∃ (n : ℕ) (m : ℕ), n * interior_angle + m * 60 = 360 :=
sorry

end regular_polygon_assembly_l3615_361521


namespace hotdogs_sold_l3615_361500

theorem hotdogs_sold (initial : ℕ) (remaining : ℕ) (sold : ℕ) : 
  initial = 99 → remaining = 97 → sold = initial - remaining → sold = 2 := by
  sorry

end hotdogs_sold_l3615_361500


namespace expression_value_l3615_361529

theorem expression_value : 
  let x : ℝ := 2
  let y : ℝ := -1
  let z : ℝ := 3
  x^2 + y^2 - z^2 + 3*x*y = -10 := by
  sorry

end expression_value_l3615_361529


namespace money_exchange_solution_l3615_361550

/-- Represents the money exchange scenario between A, B, and C -/
def MoneyExchange (a b c : ℕ) : Prop :=
  let a₁ := a - 3*b - 3*c
  let b₁ := 4*b
  let c₁ := 4*c
  let a₂ := 4*a₁
  let b₂ := b₁ - 3*a₁ - 3*c₁
  let c₂ := 4*c₁
  let a₃ := 4*a₂
  let b₃ := 4*b₂
  let c₃ := c₂ - 3*a₂ - 3*b₂
  a₃ = 27 ∧ b₃ = 27 ∧ c₃ = 27 ∧ a + b + c = 81

theorem money_exchange_solution :
  ∃ (b c : ℕ), MoneyExchange 52 b c :=
sorry

end money_exchange_solution_l3615_361550


namespace sphere_surface_area_l3615_361569

theorem sphere_surface_area (cube_surface_area : ℝ) (sphere_radius : ℝ) : 
  cube_surface_area = 24 →
  (2 * sphere_radius) ^ 2 = 3 * (cube_surface_area / 6) →
  4 * Real.pi * sphere_radius ^ 2 = 12 * Real.pi := by
sorry

end sphere_surface_area_l3615_361569


namespace planted_area_fraction_l3615_361506

theorem planted_area_fraction (a b c x : ℝ) (h1 : a = 5) (h2 : b = 12) (h3 : c^2 = a^2 + b^2) 
  (h4 : x^2 - 7*x + 9 = 0) (h5 : x > 0) (h6 : x < a) (h7 : x < b) :
  (a*b/2 - x^2) / (a*b/2) = 30/30 - ((7 - Real.sqrt 13)/2)^2 / 30 := by
  sorry

end planted_area_fraction_l3615_361506


namespace total_earnings_theorem_l3615_361559

/-- Represents the investment and return ratios for three investors -/
structure InvestmentData where
  a_invest : ℝ
  b_invest : ℝ
  c_invest : ℝ
  a_return : ℝ
  b_return : ℝ
  c_return : ℝ

/-- Calculates the total earnings given investment data -/
def totalEarnings (data : InvestmentData) : ℝ :=
  data.a_invest * data.a_return +
  data.b_invest * data.b_return +
  data.c_invest * data.c_return

/-- Theorem stating the total earnings under given conditions -/
theorem total_earnings_theorem (data : InvestmentData) :
  data.a_invest = 3 ∧
  data.b_invest = 4 ∧
  data.c_invest = 5 ∧
  data.a_return = 6 ∧
  data.b_return = 5 ∧
  data.c_return = 4 ∧
  data.b_invest * data.b_return = data.a_invest * data.a_return + 200 →
  totalEarnings data = 58000 := by
  sorry

end total_earnings_theorem_l3615_361559


namespace area_is_two_side_a_value_l3615_361568

/-- Triangle ABC with given properties -/
structure TriangleABC where
  -- Sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- Condition: cos A = 3/5
  cos_A : a^2 = b^2 + c^2 - 2*b*c*(3/5)
  -- Condition: AB · AC = 3
  dot_product : b*c*(3/5) = 3
  -- Condition: b - c = 3
  side_diff : b - c = 3

/-- The area of triangle ABC is 2 -/
theorem area_is_two (t : TriangleABC) : (1/2) * t.b * t.c * (4/5) = 2 := by sorry

/-- The value of side a is √13 -/
theorem side_a_value (t : TriangleABC) : t.a = Real.sqrt 13 := by sorry

end area_is_two_side_a_value_l3615_361568


namespace corresponding_angles_not_always_equal_l3615_361504

/-- A structure representing angles in a geometric context -/
structure Angle where
  measure : ℝ

/-- A predicate to determine if two angles are corresponding -/
def are_corresponding (a b : Angle) : Prop := sorry

/-- The theorem stating that the general claim "Corresponding angles are equal" is false -/
theorem corresponding_angles_not_always_equal :
  ¬ (∀ (a b : Angle), are_corresponding a b → a = b) := by sorry

end corresponding_angles_not_always_equal_l3615_361504


namespace produce_worth_is_630_l3615_361547

/-- The total worth of produce Gary stocked -/
def total_worth (asparagus_bundles asparagus_price grape_boxes grape_price apple_count apple_price : ℝ) : ℝ :=
  asparagus_bundles * asparagus_price + grape_boxes * grape_price + apple_count * apple_price

/-- Proof that the total worth of produce Gary stocked is $630 -/
theorem produce_worth_is_630 :
  total_worth 60 3 40 2.5 700 0.5 = 630 := by
  sorry

#eval total_worth 60 3 40 2.5 700 0.5

end produce_worth_is_630_l3615_361547


namespace donny_savings_l3615_361592

theorem donny_savings (monday : ℕ) (wednesday : ℕ) (thursday_spent : ℕ) :
  monday = 15 →
  wednesday = 13 →
  thursday_spent = 28 →
  ∃ tuesday : ℕ, 
    tuesday = 28 ∧ 
    monday + tuesday + wednesday = 2 * thursday_spent :=
by sorry

end donny_savings_l3615_361592


namespace correct_algorithm_l3615_361544

theorem correct_algorithm : 
  ((-8) / (-4) = 8 / 4) ∧ 
  ((-5) + 9 ≠ -(9 - 5)) ∧ 
  (7 - (-10) ≠ 7 - 10) ∧ 
  ((-5) * 0 ≠ -5) := by
  sorry

end correct_algorithm_l3615_361544


namespace relay_race_probability_l3615_361549

-- Define the set of students
inductive Student : Type
  | A | B | C | D

-- Define the events
def event_A (s : Student) : Prop := s = Student.A
def event_B (s : Student) : Prop := s = Student.B

-- Define the conditional probability
def conditional_probability (A B : Student → Prop) : ℚ :=
  1 / 3

-- Theorem statement
theorem relay_race_probability :
  conditional_probability event_A event_B = 1 / 3 := by
  sorry

end relay_race_probability_l3615_361549


namespace polynomial_equality_l3615_361501

theorem polynomial_equality (x : ℝ) : let p : ℝ → ℝ := λ x => -7*x^4 - 5*x^3 - 8*x^2 + 8*x - 9
  4*x^4 + 7*x^3 - 2*x + 5 + p x = -3*x^4 + 2*x^3 - 8*x^2 + 6*x - 4 := by
  sorry

end polynomial_equality_l3615_361501


namespace intersection_condition_l3615_361583

/-- Given functions f, g, f₁, g₁ and their coefficients, prove that if their graphs intersect
    at a single point with a negative x-coordinate and ac ≠ 0, then bc = ad. -/
theorem intersection_condition (a b c d : ℝ) (x₀ : ℝ) :
  let f := fun x : ℝ => a * x^2 + b * x
  let g := fun x : ℝ => c * x^2 + d * x
  let f₁ := fun x : ℝ => a * x + b
  let g₁ := fun x : ℝ => c * x + d
  (∀ x ≠ x₀, f x ≠ g x ∧ f x ≠ f₁ x ∧ f x ≠ g₁ x ∧
             g x ≠ f₁ x ∧ g x ≠ g₁ x ∧ f₁ x ≠ g₁ x) →
  (f x₀ = g x₀ ∧ f x₀ = f₁ x₀ ∧ f x₀ = g₁ x₀) →
  x₀ < 0 →
  a * c ≠ 0 →
  b * c = a * d :=
by sorry

end intersection_condition_l3615_361583


namespace min_value_quadratic_l3615_361545

theorem min_value_quadratic (x : ℝ) : x^2 + 10*x ≥ -25 ∧ ∃ y : ℝ, y^2 + 10*y = -25 := by
  sorry

end min_value_quadratic_l3615_361545


namespace direct_proportion_unique_k_l3615_361517

/-- A function f: ℝ → ℝ is a direct proportion if there exists a non-zero constant m such that f(x) = m * x for all x -/
def is_direct_proportion (f : ℝ → ℝ) : Prop :=
  ∃ (m : ℝ), m ≠ 0 ∧ ∀ x, f x = m * x

/-- The function defined by k -/
def f (k : ℝ) (x : ℝ) : ℝ := (k - 1) * x + k^2 - 1

/-- Theorem stating that k = -1 is the only value that makes f a direct proportion function -/
theorem direct_proportion_unique_k :
  ∃! k, is_direct_proportion (f k) ∧ k = -1 :=
sorry

end direct_proportion_unique_k_l3615_361517


namespace initial_population_l3615_361503

theorem initial_population (P : ℝ) : 
  (P * (1 - 0.1)^2 = 8100) → P = 10000 := by
  sorry

end initial_population_l3615_361503


namespace palindromic_not_end_zero_two_digit_palindromic_count_three_digit_palindromic_count_four_digit_palindromic_count_ten_digit_palindromic_count_l3615_361598

/-- A number is palindromic if it reads the same backward as forward. -/
def IsPalindromic (n : ℕ) : Prop := sorry

/-- The count of palindromic numbers with a given number of digits. -/
def PalindromicCount (digits : ℕ) : ℕ := sorry

/-- Palindromic numbers with more than two digits cannot end in 0. -/
theorem palindromic_not_end_zero (n : ℕ) (h : n > 99) (h_pal : IsPalindromic n) : n % 10 ≠ 0 := sorry

/-- There are 9 two-digit palindromic numbers. -/
theorem two_digit_palindromic_count : PalindromicCount 2 = 9 := sorry

/-- There are 90 three-digit palindromic numbers. -/
theorem three_digit_palindromic_count : PalindromicCount 3 = 90 := sorry

/-- There are 90 four-digit palindromic numbers. -/
theorem four_digit_palindromic_count : PalindromicCount 4 = 90 := sorry

/-- The main theorem: There are 90000 ten-digit palindromic numbers. -/
theorem ten_digit_palindromic_count : PalindromicCount 10 = 90000 := sorry

end palindromic_not_end_zero_two_digit_palindromic_count_three_digit_palindromic_count_four_digit_palindromic_count_ten_digit_palindromic_count_l3615_361598


namespace circuit_malfunction_probability_l3615_361507

/-- Represents an electronic component with a given failure rate -/
structure Component where
  failureRate : ℝ
  hFailureRate : 0 ≤ failureRate ∧ failureRate ≤ 1

/-- Represents a circuit with two components connected in series -/
structure Circuit where
  componentA : Component
  componentB : Component

/-- The probability of a circuit malfunctioning -/
def malfunctionProbability (c : Circuit) : ℝ :=
  1 - (1 - c.componentA.failureRate) * (1 - c.componentB.failureRate)

theorem circuit_malfunction_probability (c : Circuit) 
    (hA : c.componentA.failureRate = 0.2)
    (hB : c.componentB.failureRate = 0.5) :
    malfunctionProbability c = 0.6 := by
  sorry

end circuit_malfunction_probability_l3615_361507


namespace simplify_expression_l3615_361587

theorem simplify_expression (a b : ℝ) : (1:ℝ)*(2*b)*(3*a)*(4*a^2)*(5*b^2)*(6*a^3) = 720*a^6*b^3 := by
  sorry

end simplify_expression_l3615_361587


namespace student_count_l3615_361588

/-- If a student is ranked 17th from the right and 5th from the left in a line of students,
    then the total number of students is 21. -/
theorem student_count (n : ℕ) (rank_right rank_left : ℕ) 
  (h1 : rank_right = 17)
  (h2 : rank_left = 5)
  (h3 : n = rank_right + rank_left - 1) :
  n = 21 := by
  sorry

end student_count_l3615_361588


namespace smallest_two_digit_prime_with_composite_reverse_l3615_361519

/-- Returns true if n is a two-digit number -/
def isTwoDigit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

/-- Returns true if n starts with 2 -/
def startsWith2 (n : ℕ) : Prop :=
  20 ≤ n ∧ n ≤ 29

/-- Reverses the digits of a two-digit number -/
def reverseDigits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

/-- The smallest two-digit prime number starting with 2 such that 
    reversing its digits produces a composite number is 23 -/
theorem smallest_two_digit_prime_with_composite_reverse : 
  ∃ (n : ℕ), 
    isTwoDigit n ∧ 
    startsWith2 n ∧ 
    Nat.Prime n ∧ 
    ¬(Nat.Prime (reverseDigits n)) ∧
    (∀ m, m < n → ¬(isTwoDigit m ∧ startsWith2 m ∧ Nat.Prime m ∧ ¬(Nat.Prime (reverseDigits m)))) ∧
    n = 23 := by
  sorry

end smallest_two_digit_prime_with_composite_reverse_l3615_361519


namespace arun_weight_average_l3615_361535

-- Define Arun's weight as a real number
def arun_weight : ℝ := sorry

-- Define the conditions on Arun's weight
def condition1 : Prop := 61 < arun_weight ∧ arun_weight < 72
def condition2 : Prop := 60 < arun_weight ∧ arun_weight < 70
def condition3 : Prop := arun_weight ≤ 64
def condition4 : Prop := 62 < arun_weight ∧ arun_weight < 73
def condition5 : Prop := 59 < arun_weight ∧ arun_weight < 68

-- Theorem stating that the average of possible weights is 63.5
theorem arun_weight_average :
  condition1 ∧ condition2 ∧ condition3 ∧ condition4 ∧ condition5 →
  (63 + 64) / 2 = 63.5 :=
by sorry

end arun_weight_average_l3615_361535


namespace triangular_array_digit_sum_l3615_361574

/-- The number of coins in a triangular array with n rows -/
def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem triangular_array_digit_sum :
  ∃ (n : ℕ), triangular_sum n = 2145 ∧ sum_of_digits n = 11 := by
  sorry

end triangular_array_digit_sum_l3615_361574


namespace monotonic_decreasing_interval_implies_a_value_l3615_361572

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a - 1)*x + 2

-- State the theorem
theorem monotonic_decreasing_interval_implies_a_value (a : ℝ) :
  (∀ x y : ℝ, x < y ∧ y ≤ 4 → f a x > f a y) →
  a = -3 :=
sorry

end monotonic_decreasing_interval_implies_a_value_l3615_361572


namespace inequality_system_solution_set_l3615_361597

theorem inequality_system_solution_set :
  ∀ x : ℝ, (x + 1 ≥ 0 ∧ x - 2 < 0) ↔ (-1 ≤ x ∧ x < 2) := by
  sorry

end inequality_system_solution_set_l3615_361597


namespace max_sum_AB_l3615_361523

def is_digit (n : ℕ) : Prop := n ≤ 9

theorem max_sum_AB :
  ∃ (A B C D : ℕ),
    is_digit A ∧ is_digit B ∧ is_digit C ∧ is_digit D ∧
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    (C + D > 1) ∧
    (∃ k : ℕ, k * (C + D) = A + B) ∧
    (∀ A' B' C' D' : ℕ,
      is_digit A' ∧ is_digit B' ∧ is_digit C' ∧ is_digit D' →
      A' ≠ B' ∧ A' ≠ C' ∧ A' ≠ D' ∧ B' ≠ C' ∧ B' ≠ D' ∧ C' ≠ D' →
      (C' + D' > 1) →
      (∃ k' : ℕ, k' * (C' + D') = A' + B') →
      A' + B' ≤ A + B) →
    A + B = 15 := by
  sorry

end max_sum_AB_l3615_361523


namespace vitamin_d_scientific_notation_l3615_361508

theorem vitamin_d_scientific_notation : 0.0000046 = 4.6 * 10^(-6) := by
  sorry

end vitamin_d_scientific_notation_l3615_361508


namespace equal_squares_in_5x8_grid_l3615_361539

/-- A rectangular grid with alternating light and dark squares -/
structure AlternatingGrid where
  rows : ℕ
  cols : ℕ

/-- Count of dark squares in an AlternatingGrid -/
def dark_squares (grid : AlternatingGrid) : ℕ :=
  sorry

/-- Count of light squares in an AlternatingGrid -/
def light_squares (grid : AlternatingGrid) : ℕ :=
  sorry

/-- Theorem: In a 5 × 8 grid with alternating squares, the number of dark squares equals the number of light squares -/
theorem equal_squares_in_5x8_grid :
  let grid : AlternatingGrid := ⟨5, 8⟩
  dark_squares grid = light_squares grid :=
by sorry

end equal_squares_in_5x8_grid_l3615_361539


namespace adults_at_ball_game_l3615_361557

theorem adults_at_ball_game :
  let num_children : ℕ := 11
  let adult_ticket_price : ℕ := 8
  let child_ticket_price : ℕ := 4
  let total_bill : ℕ := 124
  let num_adults : ℕ := (total_bill - num_children * child_ticket_price) / adult_ticket_price
  num_adults = 10 := by
sorry

end adults_at_ball_game_l3615_361557


namespace arithmetic_sequence_properties_l3615_361515

/-- An arithmetic sequence {a_n} -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A monotonically increasing sequence -/
def monotonically_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

theorem arithmetic_sequence_properties
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_mono : monotonically_increasing a)
  (h_sum : a 1 + a 2 + a 3 = 21)
  (h_prod : a 1 * a 2 * a 3 = 231) :
  (a 2 = 7) ∧ (∀ n : ℕ, a n = 4 * n - 1) :=
sorry

end arithmetic_sequence_properties_l3615_361515


namespace mean_salary_calculation_l3615_361563

def total_employees : ℕ := 100
def salary_group_1 : ℕ := 6000
def salary_group_2 : ℕ := 4000
def salary_group_3 : ℕ := 2500
def employees_group_1 : ℕ := 5
def employees_group_2 : ℕ := 15
def employees_group_3 : ℕ := 80

theorem mean_salary_calculation :
  (salary_group_1 * employees_group_1 + salary_group_2 * employees_group_2 + salary_group_3 * employees_group_3) / total_employees = 2900 := by
  sorry

end mean_salary_calculation_l3615_361563


namespace smallest_N_for_Q_condition_l3615_361552

def Q (N : ℕ) : ℚ := ((2 * N + 3) / 3 : ℚ) / (N + 1 : ℚ)

theorem smallest_N_for_Q_condition : 
  ∀ N : ℕ, 
    N > 0 → 
    N % 6 = 0 → 
    (∀ k : ℕ, k > 0 → k % 6 = 0 → k < N → Q k ≥ 7/10) → 
    Q N < 7/10 → 
    N = 12 := by sorry

end smallest_N_for_Q_condition_l3615_361552


namespace john_illustration_time_l3615_361582

/-- Calculates the total time spent on John's illustration project -/
def total_illustration_time (
  num_landscapes : ℕ)
  (num_portraits : ℕ)
  (landscape_draw_time : ℝ)
  (landscape_color_time_ratio : ℝ)
  (portrait_draw_time : ℝ)
  (portrait_color_time_ratio : ℝ)
  (landscape_enhance_time : ℝ)
  (portrait_enhance_time : ℝ) : ℝ :=
  let landscape_time := 
    num_landscapes * (landscape_draw_time + landscape_color_time_ratio * landscape_draw_time + landscape_enhance_time)
  let portrait_time := 
    num_portraits * (portrait_draw_time + portrait_color_time_ratio * portrait_draw_time + portrait_enhance_time)
  landscape_time + portrait_time

/-- Theorem stating the total time John spends on his illustration project -/
theorem john_illustration_time : 
  total_illustration_time 10 15 2 0.7 3 0.75 0.75 1 = 135.25 := by
  sorry

end john_illustration_time_l3615_361582


namespace min_square_sum_l3615_361553

theorem min_square_sum (y₁ y₂ y₃ : ℝ) 
  (h_pos : y₁ > 0 ∧ y₂ > 0 ∧ y₃ > 0) 
  (h_sum : 2*y₁ + 3*y₂ + 4*y₃ = 120) : 
  y₁^2 + y₂^2 + y₃^2 ≥ 14400/29 := by
  sorry

end min_square_sum_l3615_361553


namespace binomial_coefficient_properties_l3615_361541

theorem binomial_coefficient_properties (p : ℕ) (hp : Nat.Prime p) :
  (∀ k, p ∣ (Nat.choose (p - 1) k ^ 2 - 1)) ∧
  (∀ s, Even s → p ∣ (Finset.sum (Finset.range p) (λ k => Nat.choose (p - 1) k ^ s))) ∧
  (∀ s, Odd s → (Finset.sum (Finset.range p) (λ k => Nat.choose (p - 1) k ^ s)) % p = 1) :=
by sorry

end binomial_coefficient_properties_l3615_361541


namespace total_rulers_l3615_361554

/-- Given an initial number of rulers and a number of rulers added, 
    the total number of rulers is equal to their sum. -/
theorem total_rulers (initial_rulers added_rulers : ℕ) :
  initial_rulers + added_rulers = initial_rulers + added_rulers :=
by sorry

end total_rulers_l3615_361554


namespace discounted_soda_price_l3615_361575

/-- Calculate the price of discounted soda cans -/
theorem discounted_soda_price
  (regular_price : ℝ)
  (discount_percent : ℝ)
  (num_cans : ℕ)
  (h1 : regular_price = 0.60)
  (h2 : discount_percent = 20)
  (h3 : num_cans = 72) :
  let discounted_price := regular_price * (1 - discount_percent / 100)
  num_cans * discounted_price = 34.56 :=
by sorry

end discounted_soda_price_l3615_361575


namespace call_center_efficiency_l3615_361518

-- Define the number of agents in each team
variable (A B : ℕ)

-- Define the fraction of calls processed by each team
variable (calls_A calls_B : ℚ)

-- Define the theorem
theorem call_center_efficiency
  (h1 : A = (5 : ℚ) / 8 * B)  -- Team A has 5/8 as many agents as team B
  (h2 : calls_B = 8 / 11)     -- Team B processed 8/11 of the total calls
  (h3 : calls_A + calls_B = 1) -- Total calls processed by both teams is 1
  : (calls_A / A) / (calls_B / B) = 3 / 5 :=
by sorry

end call_center_efficiency_l3615_361518


namespace hash_computation_l3615_361567

def hash (a b : ℤ) : ℤ := a * b - a - 3

theorem hash_computation : hash (hash 2 0) (hash 1 4) = 2 := by
  sorry

end hash_computation_l3615_361567


namespace sum_assigned_values_zero_l3615_361522

/-- The nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- The product of the first n prime numbers -/
def primeProduct (n : ℕ) : ℕ := sorry

/-- Assigns 1 or -1 to a number based on its prime factorization -/
def assignValue (n : ℕ) : Int := sorry

/-- The sum of assigned values for all divisors of a number -/
def sumAssignedValues (n : ℕ) : Int := sorry

/-- Theorem: The sum of assigned values for divisors of the product of first k primes is 0 -/
theorem sum_assigned_values_zero (k : ℕ) : sumAssignedValues (primeProduct k) = 0 := by sorry

end sum_assigned_values_zero_l3615_361522


namespace sphere_radius_ratio_l3615_361510

theorem sphere_radius_ratio (V_L V_S r_L r_S : ℝ) : 
  V_L = 675 * Real.pi → 
  V_S = 0.2 * V_L → 
  V_L = (4/3) * Real.pi * r_L^3 → 
  V_S = (4/3) * Real.pi * r_S^3 → 
  r_S / r_L = 1 / Real.rpow 5 (1/3) :=
by sorry

end sphere_radius_ratio_l3615_361510


namespace distance_between_points_l3615_361546

/-- The Euclidean distance between two points (7, 0) and (-2, 12) is 15 -/
theorem distance_between_points : Real.sqrt ((7 - (-2))^2 + (0 - 12)^2) = 15 := by
  sorry

end distance_between_points_l3615_361546


namespace smallest_n_square_and_cube_l3615_361578

def is_perfect_square (x : ℕ) : Prop := ∃ y : ℕ, x = y^2

def is_perfect_cube (x : ℕ) : Prop := ∃ y : ℕ, x = y^3

theorem smallest_n_square_and_cube :
  let n := 54
  (∀ m : ℕ, m > 0 ∧ m < n → ¬(is_perfect_square (3*m) ∧ is_perfect_cube (4*m))) ∧
  is_perfect_square (3*n) ∧ is_perfect_cube (4*n) :=
sorry

end smallest_n_square_and_cube_l3615_361578


namespace perpendicular_line_to_plane_and_contained_line_l3615_361571

-- Define the types for lines and planes
def Line : Type := sorry
def Plane : Type := sorry

-- Define the relationships between lines and planes
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def contained_in (l : Line) (p : Plane) : Prop := sorry
def perpendicular_lines (l1 l2 : Line) : Prop := sorry

-- Theorem statement
theorem perpendicular_line_to_plane_and_contained_line 
  (m n : Line) (α : Plane) 
  (h1 : perpendicular m α) 
  (h2 : contained_in n α) : 
  perpendicular_lines m n := by sorry

end perpendicular_line_to_plane_and_contained_line_l3615_361571


namespace remaining_distance_l3615_361548

theorem remaining_distance (total_distance driven_distance : ℕ) : 
  total_distance = 1200 → driven_distance = 642 → total_distance - driven_distance = 558 := by
  sorry

end remaining_distance_l3615_361548


namespace billion_to_scientific_notation_l3615_361540

theorem billion_to_scientific_notation :
  ∀ (x : ℝ), x = 26.62 * 1000000000 → x = 2.662 * (10 ^ 9) := by
  sorry

end billion_to_scientific_notation_l3615_361540


namespace percentage_increase_l3615_361530

theorem percentage_increase (initial final : ℝ) (h : initial > 0) :
  let increase := (final - initial) / initial * 100
  initial = 150 ∧ final = 210 → increase = 40 := by
  sorry

end percentage_increase_l3615_361530


namespace part_one_part_two_l3615_361590

/-- Given positive numbers a, b, c, d such that ad = bc and a + d > b + c, 
    then |a - d| > |b - c| -/
theorem part_one (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h_ad_bc : a * d = b * c) (h_sum : a + d > b + c) :
  |a - d| > |b - c| := by sorry

/-- Given positive numbers a, b, c, d and a real number t such that 
    t * √(a² + b²) * √(c² + d²) = √(a⁴ + c⁴) + √(b⁴ + d⁴), then t ≥ √2 -/
theorem part_two (a b c d t : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h_eq : t * Real.sqrt (a^2 + b^2) * Real.sqrt (c^2 + d^2) = 
          Real.sqrt (a^4 + c^4) + Real.sqrt (b^4 + d^4)) :
  t ≥ Real.sqrt 2 := by sorry

end part_one_part_two_l3615_361590


namespace chess_team_arrangement_l3615_361505

/-- The number of boys on the chess team -/
def num_boys : ℕ := 3

/-- The number of girls on the chess team -/
def num_girls : ℕ := 2

/-- The total number of students on the chess team -/
def total_students : ℕ := num_boys + num_girls

/-- The number of ways to arrange the chess team in a row with a girl at each end and boys in the middle -/
def num_arrangements : ℕ := num_girls.factorial * num_boys.factorial

theorem chess_team_arrangement :
  num_arrangements = 12 :=
sorry

end chess_team_arrangement_l3615_361505


namespace product_of_differences_l3615_361594

theorem product_of_differences (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2005) (h₂ : y₁^3 - 3*x₁^2*y₁ = 2004)
  (h₃ : x₂^3 - 3*x₂*y₂^2 = 2005) (h₄ : y₂^3 - 3*x₂^2*y₂ = 2004)
  (h₅ : x₃^3 - 3*x₃*y₃^2 = 2005) (h₆ : y₃^3 - 3*x₃^2*y₃ = 2004) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = 1/1002 := by
  sorry

end product_of_differences_l3615_361594


namespace f_properties_l3615_361566

def f (a x : ℝ) : ℝ := x^2 + |x - a| + 1

theorem f_properties (a : ℝ) :
  (∀ x, f a x = f a (-x) ↔ a = 0) ∧
  (∀ x, f a x ≥ 
    if a ≤ -1/2 then 3/4 - a
    else if a ≤ 1/2 then a^2 + 1
    else 3/4 + a) ∧
  (∃ x, f a x = 
    if a ≤ -1/2 then 3/4 - a
    else if a ≤ 1/2 then a^2 + 1
    else 3/4 + a) := by
  sorry

end f_properties_l3615_361566


namespace greatest_divisor_three_consecutive_integers_l3615_361556

theorem greatest_divisor_three_consecutive_integers :
  ∃ (d : ℕ), d > 0 ∧ 
  (∀ (n : ℕ), n > 0 → d ∣ (n * (n + 1) * (n + 2))) ∧
  (∀ (k : ℕ), k > d → ∃ (m : ℕ), m > 0 ∧ ¬(k ∣ (m * (m + 1) * (m + 2)))) ∧
  d = 6 :=
sorry

end greatest_divisor_three_consecutive_integers_l3615_361556


namespace unique_number_with_equal_sums_l3615_361511

def ends_with_9876 (n : ℕ) : Prop :=
  n % 10000 = 9876

def masha_sum (n : ℕ) : ℕ :=
  (n / 1000) * 10 + n % 1000

def misha_sum (n : ℕ) : ℕ :=
  (n / 10000) + n % 10000

theorem unique_number_with_equal_sums :
  ∃! n : ℕ, n > 9999 ∧ ends_with_9876 n ∧ masha_sum n = misha_sum n :=
by
  sorry

end unique_number_with_equal_sums_l3615_361511


namespace ferris_wheel_seats_l3615_361595

theorem ferris_wheel_seats (people_per_seat : ℕ) (total_people : ℕ) (h1 : people_per_seat = 9) (h2 : total_people = 18) :
  total_people / people_per_seat = 2 :=
by sorry

end ferris_wheel_seats_l3615_361595


namespace team_total_score_l3615_361516

def team_score (connor_initial : ℕ) (amy_initial : ℕ) (jason_initial : ℕ) 
  (connor_bonus : ℕ) (amy_bonus : ℕ) (jason_bonus : ℕ) (emily : ℕ) : ℕ :=
  connor_initial + connor_bonus + amy_initial + amy_bonus + jason_initial + jason_bonus + emily

theorem team_total_score :
  let connor_initial := 2
  let amy_initial := connor_initial + 4
  let jason_initial := amy_initial * 2
  let connor_bonus := 3
  let amy_bonus := 5
  let jason_bonus := 1
  let emily := 3 * (connor_initial + amy_initial + jason_initial)
  team_score connor_initial amy_initial jason_initial connor_bonus amy_bonus jason_bonus emily = 89 := by
  sorry

end team_total_score_l3615_361516


namespace x_equals_160_l3615_361534

/-- Given a relationship between x, y, and z, prove that x equals 160 when y is 16 and z is 7. -/
theorem x_equals_160 (k : ℝ) (x y z : ℝ → ℝ) :
  (∀ t, x t = k * y t / (z t)^2) →  -- Relationship between x, y, and z
  (x 0 = 10 ∧ y 0 = 4 ∧ z 0 = 14) →  -- Initial condition
  (y 1 = 16 ∧ z 1 = 7) →  -- New condition
  x 1 = 160 := by
sorry

end x_equals_160_l3615_361534


namespace total_loaves_served_l3615_361565

theorem total_loaves_served (wheat_bread : Real) (white_bread : Real) :
  wheat_bread = 0.2 → white_bread = 0.4 → wheat_bread + white_bread = 0.6 := by
  sorry

end total_loaves_served_l3615_361565


namespace juice_transfer_difference_l3615_361555

/-- Represents a barrel with a certain volume of juice -/
structure Barrel where
  volume : ℝ

/-- Represents the state of two barrels -/
structure TwoBarrels where
  barrel1 : Barrel
  barrel2 : Barrel

/-- Transfers a given volume from one barrel to another -/
def transfer (barrels : TwoBarrels) (amount : ℝ) : TwoBarrels :=
  { barrel1 := { volume := barrels.barrel1.volume + amount },
    barrel2 := { volume := barrels.barrel2.volume - amount } }

/-- Calculates the difference in volume between two barrels -/
def volumeDifference (barrels : TwoBarrels) : ℝ :=
  barrels.barrel1.volume - barrels.barrel2.volume

/-- Theorem stating that after transferring 3 L from the 8 L barrel to the 10 L barrel,
    the difference in volume between the two barrels is 8 L -/
theorem juice_transfer_difference :
  let initialBarrels : TwoBarrels := { barrel1 := { volume := 10 }, barrel2 := { volume := 8 } }
  let finalBarrels := transfer initialBarrels 3
  volumeDifference finalBarrels = 8 := by
  sorry


end juice_transfer_difference_l3615_361555


namespace exp_two_pi_third_in_second_quadrant_l3615_361599

-- Define the complex exponential function
noncomputable def complex_exp (z : ℂ) : ℂ := Real.exp z.re * (Real.cos z.im + Complex.I * Real.sin z.im)

-- Define the second quadrant of the complex plane
def second_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im > 0

-- Theorem statement
theorem exp_two_pi_third_in_second_quadrant :
  second_quadrant (complex_exp (Complex.I * (2 * Real.pi / 3))) :=
sorry

end exp_two_pi_third_in_second_quadrant_l3615_361599


namespace arithmetic_sequence_first_term_l3615_361573

/-- An arithmetic sequence with second term -5 and common difference 3 has first term -8 -/
theorem arithmetic_sequence_first_term (a : ℕ → ℤ) :
  (∀ n, a (n + 1) = a n + 3) →  -- arithmetic sequence condition
  a 2 = -5 →                    -- given second term
  a 1 = -8 :=                   -- conclusion: first term
by sorry

end arithmetic_sequence_first_term_l3615_361573


namespace multiple_value_l3615_361525

theorem multiple_value (a b m : ℤ) : 
  a * b = m * (a + b) + 1 → 
  b = 7 → 
  b - a = 4 → 
  m = 2 := by
sorry

end multiple_value_l3615_361525


namespace smallest_valid_graph_size_l3615_361581

/-- A graph representing acquaintances among n people -/
def AcquaintanceGraph (n : ℕ) := Fin n → Fin n → Prop

/-- The property that any two acquainted people have no common acquaintances -/
def NoCommonAcquaintances (n : ℕ) (g : AcquaintanceGraph n) : Prop :=
  ∀ a b c : Fin n, g a b → g a c → g b c → a = b ∨ a = c ∨ b = c

/-- The property that any two non-acquainted people have exactly two common acquaintances -/
def TwoCommonAcquaintances (n : ℕ) (g : AcquaintanceGraph n) : Prop :=
  ∀ a b : Fin n, ¬g a b → ∃! (c d : Fin n), c ≠ d ∧ g a c ∧ g a d ∧ g b c ∧ g b d

/-- The main theorem stating that 11 is the smallest number satisfying the conditions -/
theorem smallest_valid_graph_size :
  (∃ (g : AcquaintanceGraph 11), NoCommonAcquaintances 11 g ∧ TwoCommonAcquaintances 11 g) ∧
  (∀ n : ℕ, 5 ≤ n → n < 11 →
    ¬∃ (g : AcquaintanceGraph n), NoCommonAcquaintances n g ∧ TwoCommonAcquaintances n g) :=
sorry

end smallest_valid_graph_size_l3615_361581


namespace average_net_income_is_399_50_l3615_361586

/-- Represents the daily income and expense for a cab driver --/
structure DailyFinance where
  income : ℝ
  expense : ℝ

/-- Calculates the net income for a single day --/
def netIncome (df : DailyFinance) : ℝ := df.income - df.expense

/-- The cab driver's finances for 10 days --/
def tenDaysFinances : List DailyFinance := [
  ⟨600, 50⟩,
  ⟨250, 70⟩,
  ⟨450, 100⟩,
  ⟨400, 30⟩,
  ⟨800, 60⟩,
  ⟨450, 40⟩,
  ⟨350, 0⟩,
  ⟨600, 55⟩,
  ⟨270, 80⟩,
  ⟨500, 90⟩
]

/-- Theorem: The average daily net income for the cab driver over 10 days is $399.50 --/
theorem average_net_income_is_399_50 :
  (tenDaysFinances.map netIncome).sum / 10 = 399.50 := by
  sorry


end average_net_income_is_399_50_l3615_361586


namespace girls_at_picnic_l3615_361579

theorem girls_at_picnic (total_students : ℕ) (picnic_attendees : ℕ) 
  (h1 : total_students = 1200)
  (h2 : picnic_attendees = 730)
  (h3 : ∃ (girls boys : ℕ), girls + boys = total_students ∧ 
    2 * girls / 3 + boys / 2 = picnic_attendees) :
  ∃ (girls : ℕ), 2 * girls / 3 = 520 := by
sorry

end girls_at_picnic_l3615_361579


namespace share_multiple_l3615_361520

theorem share_multiple (total : ℝ) (c_share : ℝ) (k : ℝ) :
  total = 427 →
  c_share = 84 →
  (∃ (a_share b_share : ℝ), 
    total = a_share + b_share + c_share ∧
    3 * a_share = 4 * b_share ∧
    3 * a_share = k * c_share) →
  k = 7 := by
sorry

end share_multiple_l3615_361520


namespace hello_arrangements_l3615_361570

theorem hello_arrangements : ℕ := by
  -- Define the word length
  let word_length : ℕ := 5

  -- Define the number of repeated letters
  let repeated_letters : ℕ := 1

  -- Define the number of repetitions of the repeated letter
  let repetitions : ℕ := 2

  -- Calculate total permutations
  let total_permutations : ℕ := Nat.factorial word_length

  -- Calculate unique permutations
  let unique_permutations : ℕ := total_permutations / Nat.factorial repeated_letters

  -- Calculate incorrect arrangements
  let incorrect_arrangements : ℕ := unique_permutations - 1

  -- Prove that the number of incorrect arrangements is 59
  sorry

end hello_arrangements_l3615_361570
