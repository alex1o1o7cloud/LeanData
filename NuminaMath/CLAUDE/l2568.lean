import Mathlib

namespace NUMINAMATH_CALUDE_concrete_components_correct_l2568_256872

/-- Represents the ratio of cement, sand, and gravel in the concrete mixture -/
def concrete_ratio : Fin 3 → ℕ
  | 0 => 2  -- cement
  | 1 => 4  -- sand
  | 2 => 5  -- gravel

/-- The total amount of concrete needed in tons -/
def total_concrete : ℕ := 121

/-- Calculates the amount of a component needed based on its ratio and the total concrete amount -/
def component_amount (ratio : ℕ) (total_ratio : ℕ) (total_amount : ℕ) : ℕ :=
  (ratio * total_amount) / total_ratio

/-- Theorem stating the correct amounts of cement and gravel needed -/
theorem concrete_components_correct :
  let total_ratio := (concrete_ratio 0) + (concrete_ratio 1) + (concrete_ratio 2)
  component_amount (concrete_ratio 0) total_ratio total_concrete = 22 ∧
  component_amount (concrete_ratio 2) total_ratio total_concrete = 55 := by
  sorry


end NUMINAMATH_CALUDE_concrete_components_correct_l2568_256872


namespace NUMINAMATH_CALUDE_triangle_inequality_l2568_256888

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  a / (b + c) + b / (c + a) + c / (a + b) ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2568_256888


namespace NUMINAMATH_CALUDE_orange_problem_l2568_256861

theorem orange_problem (initial_oranges : ℕ) : 
  (initial_oranges : ℚ) * (3/4) * (4/7) - 4 = 32 → initial_oranges = 84 := by
  sorry

end NUMINAMATH_CALUDE_orange_problem_l2568_256861


namespace NUMINAMATH_CALUDE_machine_B_performs_better_l2568_256841

def machineA : List ℕ := [0, 1, 0, 2, 2, 0, 3, 1, 2, 4]
def machineB : List ℕ := [2, 3, 1, 1, 0, 2, 1, 1, 0, 1]

def average (l : List ℕ) : ℚ := (l.sum : ℚ) / l.length

def variance (l : List ℕ) : ℚ :=
  let avg := average l
  (l.map (fun x => ((x : ℚ) - avg) ^ 2)).sum / l.length

theorem machine_B_performs_better :
  average machineB < average machineA ∧
  variance machineB < variance machineA := by
  sorry

end NUMINAMATH_CALUDE_machine_B_performs_better_l2568_256841


namespace NUMINAMATH_CALUDE_parents_gift_cost_l2568_256867

def total_budget : ℕ := 100
def num_friends : ℕ := 8
def friend_gift_cost : ℕ := 9
def num_parents : ℕ := 2

theorem parents_gift_cost (parent_gift_cost : ℕ) : 
  parent_gift_cost * num_parents + num_friends * friend_gift_cost = total_budget →
  parent_gift_cost = 14 := by
  sorry

end NUMINAMATH_CALUDE_parents_gift_cost_l2568_256867


namespace NUMINAMATH_CALUDE_intersection_implies_C_value_l2568_256853

/-- Two lines intersect on the y-axis iff their intersection point has x-coordinate 0 -/
def intersect_on_y_axis (A C : ℝ) : Prop :=
  ∃ y : ℝ, A * 0 + 3 * y + C = 0 ∧ 2 * 0 - 3 * y + 4 = 0

/-- If the lines Ax + 3y + C = 0 and 2x - 3y + 4 = 0 intersect on the y-axis, then C = -4 -/
theorem intersection_implies_C_value (A : ℝ) :
  intersect_on_y_axis A C → C = -4 :=
sorry

end NUMINAMATH_CALUDE_intersection_implies_C_value_l2568_256853


namespace NUMINAMATH_CALUDE_exam_girls_count_l2568_256828

theorem exam_girls_count (total : ℕ) (pass_rate_boys : ℚ) (pass_rate_girls : ℚ) (fail_rate_total : ℚ) :
  total = 2000 ∧
  pass_rate_boys = 30 / 100 ∧
  pass_rate_girls = 32 / 100 ∧
  fail_rate_total = 691 / 1000 →
  ∃ (girls : ℕ), girls = 900 ∧ girls ≤ total ∧
    (girls : ℚ) * pass_rate_girls + (total - girls : ℚ) * pass_rate_boys = (1 - fail_rate_total) * total :=
by sorry

end NUMINAMATH_CALUDE_exam_girls_count_l2568_256828


namespace NUMINAMATH_CALUDE_polynomial_factor_coefficients_l2568_256829

theorem polynomial_factor_coefficients :
  ∀ (a b : ℚ),
  (∃ (c d : ℚ), ∀ (x : ℚ),
    a * x^4 + b * x^3 + 40 * x^2 - 20 * x + 9 =
    (4 * x^2 - 3 * x + 2) * (c * x^2 + d * x + 4.5)) →
  a = 11 ∧ b = -121/4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factor_coefficients_l2568_256829


namespace NUMINAMATH_CALUDE_pool_volume_is_60_gallons_l2568_256895

/-- The volume of water in Lydia's pool when full -/
def pool_volume (inflow_rate outflow_rate fill_time : ℝ) : ℝ :=
  (inflow_rate - outflow_rate) * fill_time

/-- Theorem stating that the pool volume is 60 gallons -/
theorem pool_volume_is_60_gallons :
  pool_volume 1.6 0.1 40 = 60 := by
  sorry

end NUMINAMATH_CALUDE_pool_volume_is_60_gallons_l2568_256895


namespace NUMINAMATH_CALUDE_exponential_distribution_expected_value_l2568_256826

/-- The expected value of an exponentially distributed random variable -/
theorem exponential_distribution_expected_value (α : ℝ) (hα : α > 0) :
  let X : ℝ → ℝ := λ x => if x ≥ 0 then α * Real.exp (-α * x) else 0
  ∫ x in Set.Ici 0, x * X x = 1 / α :=
sorry

end NUMINAMATH_CALUDE_exponential_distribution_expected_value_l2568_256826


namespace NUMINAMATH_CALUDE_batsman_average_theorem_l2568_256832

/-- Represents a batsman's score history -/
structure Batsman where
  innings : Nat
  totalRuns : Nat
  averageIncrease : Nat
  lastInningScore : Nat

/-- Calculates the new average after the latest inning -/
def newAverage (b : Batsman) : Rat :=
  (b.totalRuns + b.lastInningScore) / (b.innings + 1)

/-- Theorem: A batsman who scores 100 runs in his 17th inning and increases his average by 5 runs will have a new average of 20 runs -/
theorem batsman_average_theorem (b : Batsman) 
  (h1 : b.innings = 16)
  (h2 : b.lastInningScore = 100)
  (h3 : b.averageIncrease = 5)
  (h4 : newAverage b = (b.totalRuns + b.lastInningScore) / (b.innings + 1)) :
  newAverage b = 20 := by
  sorry

#check batsman_average_theorem

end NUMINAMATH_CALUDE_batsman_average_theorem_l2568_256832


namespace NUMINAMATH_CALUDE_total_spider_legs_l2568_256852

/-- The number of spiders in the room -/
def num_spiders : ℕ := 4

/-- The number of legs each spider has -/
def legs_per_spider : ℕ := 8

/-- Theorem: The total number of spider legs is 32 -/
theorem total_spider_legs : num_spiders * legs_per_spider = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_spider_legs_l2568_256852


namespace NUMINAMATH_CALUDE_triangle_count_equality_l2568_256809

/-- The number of non-congruent triangles with positive area and integer side lengths summing to n -/
def T (n : ℕ) : ℕ := sorry

/-- The statement to prove -/
theorem triangle_count_equality : T 2022 = T 2019 := by sorry

end NUMINAMATH_CALUDE_triangle_count_equality_l2568_256809


namespace NUMINAMATH_CALUDE_graduates_not_both_l2568_256883

def biotechnology_class (total_graduates : ℕ) (both_job_and_degree : ℕ) : Prop :=
  total_graduates - both_job_and_degree = 60

theorem graduates_not_both : biotechnology_class 73 13 :=
  sorry

end NUMINAMATH_CALUDE_graduates_not_both_l2568_256883


namespace NUMINAMATH_CALUDE_cube_edge_length_l2568_256871

/-- The length of one edge of a cube given the sum of all edge lengths -/
theorem cube_edge_length (sum_of_edges : ℝ) (h : sum_of_edges = 144) : 
  sum_of_edges / 12 = 12 := by
  sorry

#check cube_edge_length

end NUMINAMATH_CALUDE_cube_edge_length_l2568_256871


namespace NUMINAMATH_CALUDE_min_value_inequality_l2568_256816

theorem min_value_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 4) :
  (1 / x + 4 / y) ≥ 9/4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l2568_256816


namespace NUMINAMATH_CALUDE_no_rational_solution_l2568_256836

theorem no_rational_solution : ¬∃ (a b : ℚ), a ≠ 0 ∧ b ≠ 0 ∧ a^2 * b^2 * (a^2 * b^2 + 4) = 2 * (a^6 + b^6) := by
  sorry

end NUMINAMATH_CALUDE_no_rational_solution_l2568_256836


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l2568_256801

theorem tangent_line_to_circle (m : ℝ) : 
  m > 0 → 
  (∀ x y : ℝ, x + y = 0 → (x - m)^2 + y^2 = 2) → 
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l2568_256801


namespace NUMINAMATH_CALUDE_eighth_equation_sum_l2568_256849

theorem eighth_equation_sum (a t : ℝ) (ha : a > 0) (ht : t > 0) :
  (8 + a / t).sqrt = 8 * (a / t).sqrt → a + t = 71 := by
  sorry

end NUMINAMATH_CALUDE_eighth_equation_sum_l2568_256849


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_angle_l2568_256808

theorem hyperbola_asymptote_angle (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) → 
  (Real.arctan ((b/a) / (1 - (b/a)^2)) * 2 = π / 4) →
  a / b = Real.sqrt 2 + 1 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_angle_l2568_256808


namespace NUMINAMATH_CALUDE_concert_ticket_discount_l2568_256845

theorem concert_ticket_discount (normal_price : ℝ) (scalper_markup : ℝ) (scalper_discount : ℝ) (total_paid : ℝ) :
  normal_price = 50 →
  scalper_markup = 2.4 →
  scalper_discount = 10 →
  total_paid = 360 →
  ∃ (discounted_price : ℝ),
    2 * normal_price + 2 * (scalper_markup * normal_price - scalper_discount / 2) + discounted_price = total_paid ∧
    discounted_price / normal_price = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_concert_ticket_discount_l2568_256845


namespace NUMINAMATH_CALUDE_a_greater_than_b_l2568_256802

theorem a_greater_than_b (n : ℕ) (a b : ℝ) 
  (h_n : n > 1)
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_a_eq : a^n = a + 1)
  (h_b_eq : b^(2*n) = b + 3*a) :
  a > b :=
by sorry

end NUMINAMATH_CALUDE_a_greater_than_b_l2568_256802


namespace NUMINAMATH_CALUDE_negation_equivalence_l2568_256869

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, Real.log x₀ < x₀^2 - 1) ↔ (∀ x : ℝ, Real.log x ≥ x^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2568_256869


namespace NUMINAMATH_CALUDE_triangle_side_b_value_l2568_256818

theorem triangle_side_b_value (A B C : ℝ) (a b c : ℝ) :
  c = Real.sqrt 6 →
  Real.cos C = -(1/4 : ℝ) →
  Real.sin A = 2 * Real.sin B →
  b = 1 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_b_value_l2568_256818


namespace NUMINAMATH_CALUDE_widget_sales_sum_l2568_256858

def arithmetic_sequence (n : ℕ) : ℕ := 3 * n - 1

def sum_arithmetic_sequence (n : ℕ) : ℕ :=
  n * (arithmetic_sequence 1 + arithmetic_sequence n) / 2

theorem widget_sales_sum :
  sum_arithmetic_sequence 15 = 345 := by
  sorry

end NUMINAMATH_CALUDE_widget_sales_sum_l2568_256858


namespace NUMINAMATH_CALUDE_expression_simplification_l2568_256839

theorem expression_simplification (m n x : ℝ) :
  (3 * m^2 + 2 * m * n - 5 * m^2 + 3 * m * n = -2 * m^2 + 5 * m * n) ∧
  ((x^2 + 2 * x) - 2 * (x^2 - x) = -x^2 + 4 * x) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2568_256839


namespace NUMINAMATH_CALUDE_tiles_for_wall_l2568_256899

/-- The number of tiles needed to cover a wall -/
def tiles_needed (tile_size wall_length wall_width : ℕ) : ℕ :=
  (wall_length / tile_size) * (wall_width / tile_size)

/-- Theorem: 432 tiles of size 15 cm × 15 cm are needed to cover a wall of 360 cm × 270 cm -/
theorem tiles_for_wall : tiles_needed 15 360 270 = 432 := by
  sorry

end NUMINAMATH_CALUDE_tiles_for_wall_l2568_256899


namespace NUMINAMATH_CALUDE_fabric_cost_and_length_l2568_256897

/-- Given two identical pieces of fabric with the following properties:
    1. The total cost of the first piece is 126 rubles more than the second piece
    2. The cost of 4 meters from the first piece exceeds the cost of 3 meters from the second piece by 135 rubles
    3. 3 meters from the first piece and 4 meters from the second piece cost 382.50 rubles in total

    This theorem proves that:
    1. The length of each piece is 5.6 meters
    2. The cost per meter of the first piece is 67.5 rubles
    3. The cost per meter of the second piece is 45 rubles
-/
theorem fabric_cost_and_length 
  (cost_second : ℝ) -- Total cost of the second piece
  (length : ℝ) -- Length of each piece
  (h1 : cost_second + 126 = (cost_second / length + 126 / length) * length) -- First piece costs 126 more
  (h2 : 4 * (cost_second / length + 126 / length) - 3 * (cost_second / length) = 135) -- 4m of first vs 3m of second
  (h3 : 3 * (cost_second / length + 126 / length) + 4 * (cost_second / length) = 382.5) -- Total cost of 3m+4m
  : length = 5.6 ∧ 
    cost_second / length + 126 / length = 67.5 ∧ 
    cost_second / length = 45 := by
  sorry

end NUMINAMATH_CALUDE_fabric_cost_and_length_l2568_256897


namespace NUMINAMATH_CALUDE_f_at_pi_third_l2568_256868

noncomputable def f (θ : Real) : Real :=
  (2 * Real.cos θ ^ 2 + Real.sin (2 * Real.pi - θ) ^ 2 + Real.sin (Real.pi / 2 + θ) - 3) /
  (2 + 2 * Real.cos (Real.pi + θ) ^ 2 + Real.cos (-θ))

theorem f_at_pi_third : f (Real.pi / 3) = -5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_f_at_pi_third_l2568_256868


namespace NUMINAMATH_CALUDE_transport_cost_bounds_l2568_256896

/-- Represents the transportation problem with cities A, B, C, D, and E. -/
structure TransportProblem where
  trucksA : ℕ := 10
  trucksB : ℕ := 10
  trucksC : ℕ := 8
  trucksToD : ℕ := 18
  trucksToE : ℕ := 10
  costAD : ℕ := 200
  costAE : ℕ := 800
  costBD : ℕ := 300
  costBE : ℕ := 700
  costCD : ℕ := 400
  costCE : ℕ := 500

/-- Calculates the total transportation cost given the number of trucks from A and B to D. -/
def totalCost (p : TransportProblem) (x : ℕ) : ℕ :=
  p.costAD * x + p.costBD * x + p.costCD * (p.trucksToD - 2*x) +
  p.costAE * (p.trucksA - x) + p.costBE * (p.trucksB - x) + p.costCE * (x + x - p.trucksToE)

/-- Theorem stating the minimum and maximum transportation costs. -/
theorem transport_cost_bounds (p : TransportProblem) :
  ∃ (xMin xMax : ℕ), 
    (∀ x, 5 ≤ x ∧ x ≤ 9 → totalCost p x ≥ totalCost p xMin) ∧
    (∀ x, 5 ≤ x ∧ x ≤ 9 → totalCost p x ≤ totalCost p xMax) ∧
    totalCost p xMin = 10000 ∧
    totalCost p xMax = 13200 :=
  sorry

end NUMINAMATH_CALUDE_transport_cost_bounds_l2568_256896


namespace NUMINAMATH_CALUDE_unique_solution_power_equation_l2568_256890

theorem unique_solution_power_equation :
  ∀ x y : ℕ, x ≥ 1 → y ≥ 1 → (2^x : ℤ) - 5 = 11^y → x = 4 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_power_equation_l2568_256890


namespace NUMINAMATH_CALUDE_greatest_x_with_lcm_l2568_256819

theorem greatest_x_with_lcm (x : ℕ) : 
  (∃ (lcm : ℕ), lcm = Nat.lcm x (Nat.lcm 12 18) ∧ lcm = 108) →
  x ≤ 108 ∧ ∃ (y : ℕ), y = 108 ∧ Nat.lcm y (Nat.lcm 12 18) = 108 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_with_lcm_l2568_256819


namespace NUMINAMATH_CALUDE_fgh_supermarkets_count_l2568_256859

/-- The number of FGH supermarkets in the US -/
def us_supermarkets : ℕ := 47

/-- The number of FGH supermarkets in Canada -/
def canada_supermarkets : ℕ := us_supermarkets - 10

/-- The total number of FGH supermarkets -/
def total_supermarkets : ℕ := us_supermarkets + canada_supermarkets

/-- Theorem stating that the total number of FGH supermarkets is 84 -/
theorem fgh_supermarkets_count : total_supermarkets = 84 := by
  sorry

end NUMINAMATH_CALUDE_fgh_supermarkets_count_l2568_256859


namespace NUMINAMATH_CALUDE_otimes_inequality_range_l2568_256870

/-- Custom binary operation ⊗ -/
def otimes (a b : ℝ) : ℝ := a - 2 * b

/-- Theorem stating the range of a given the conditions -/
theorem otimes_inequality_range (a : ℝ) :
  (∀ x : ℝ, x > 6 ↔ (otimes x 3 > 0 ∧ otimes x a > a)) →
  a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_otimes_inequality_range_l2568_256870


namespace NUMINAMATH_CALUDE_modulo_nine_sum_product_l2568_256831

theorem modulo_nine_sum_product : 
  (2 * (1 + 222 + 3333 + 44444 + 555555 + 6666666 + 77777777 + 888888888)) % 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_modulo_nine_sum_product_l2568_256831


namespace NUMINAMATH_CALUDE_incorrect_division_result_l2568_256878

theorem incorrect_division_result (D : ℕ) (h : D / 36 = 58) : 
  Int.floor (D / 87 : ℚ) = 24 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_division_result_l2568_256878


namespace NUMINAMATH_CALUDE_river_road_cars_l2568_256835

theorem river_road_cars (buses cars : ℕ) : 
  (buses : ℚ) / cars = 1 / 13 →
  buses = cars - 60 →
  cars = 65 := by
sorry

end NUMINAMATH_CALUDE_river_road_cars_l2568_256835


namespace NUMINAMATH_CALUDE_height_relation_holds_for_data_height_relation_generalizes_l2568_256880

/-- Represents the height of a ball falling and rebounding -/
structure BallHeight where
  x : ℝ  -- height of ball falling
  h : ℝ  -- height of ball after landing

/-- The set of observed data points -/
def observedData : Set BallHeight := {
  ⟨10, 5⟩, ⟨30, 15⟩, ⟨50, 25⟩, ⟨70, 35⟩
}

/-- The proposed relationship between x and h -/
def heightRelation (bh : BallHeight) : Prop :=
  bh.h = (1/2) * bh.x

/-- Theorem stating that the proposed relationship holds for all observed data points -/
theorem height_relation_holds_for_data : 
  ∀ bh ∈ observedData, heightRelation bh :=
sorry

/-- Theorem stating that the relationship generalizes to any height -/
theorem height_relation_generalizes (x : ℝ) : 
  ∃ h : ℝ, heightRelation ⟨x, h⟩ :=
sorry

end NUMINAMATH_CALUDE_height_relation_holds_for_data_height_relation_generalizes_l2568_256880


namespace NUMINAMATH_CALUDE_rock_collecting_contest_l2568_256823

theorem rock_collecting_contest (sydney_initial conner_initial : ℕ)
  (sydney_day1 conner_day1_multiplier : ℕ)
  (sydney_day3_multiplier conner_day3 : ℕ) :
  sydney_initial = 837 →
  conner_initial = 723 →
  sydney_day1 = 4 →
  conner_day1_multiplier = 8 →
  sydney_day3_multiplier = 2 →
  conner_day3 = 27 →
  ∃ (conner_day2 : ℕ),
    sydney_initial + sydney_day1 + sydney_day3_multiplier * (conner_day1_multiplier * sydney_day1) ≤
    conner_initial + (conner_day1_multiplier * sydney_day1) + conner_day2 + conner_day3 ∧
    conner_day2 = 123 :=
by sorry

end NUMINAMATH_CALUDE_rock_collecting_contest_l2568_256823


namespace NUMINAMATH_CALUDE_factorial_difference_quotient_l2568_256855

theorem factorial_difference_quotient : (Nat.factorial 13 - Nat.factorial 12) / Nat.factorial 10 = 1584 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_quotient_l2568_256855


namespace NUMINAMATH_CALUDE_james_height_l2568_256846

theorem james_height (tree_height : ℝ) (tree_shadow : ℝ) (james_shadow : ℝ) :
  tree_height = 60 →
  tree_shadow = 20 →
  james_shadow = 25 →
  (tree_height / tree_shadow) * james_shadow = 75 := by
  sorry

end NUMINAMATH_CALUDE_james_height_l2568_256846


namespace NUMINAMATH_CALUDE_chord_slope_l2568_256834

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 20 + y^2 / 16 = 1

-- Define the point P
def P : ℝ × ℝ := (3, -2)

-- Define the line l
def line_l (m : ℝ) (x y : ℝ) : Prop := y - P.2 = m * (x - P.1)

-- Define the midpoint property
def is_midpoint (P A B : ℝ × ℝ) : Prop :=
  P.1 = (A.1 + B.1) / 2 ∧ P.2 = (A.2 + B.2) / 2

-- Theorem statement
theorem chord_slope :
  ∃ (A B : ℝ × ℝ) (m : ℝ),
    ellipse A.1 A.2 ∧
    ellipse B.1 B.2 ∧
    line_l m A.1 A.2 ∧
    line_l m B.1 B.2 ∧
    is_midpoint P A B ∧
    m = 6/5 := by sorry

end NUMINAMATH_CALUDE_chord_slope_l2568_256834


namespace NUMINAMATH_CALUDE_paco_cookies_l2568_256804

def cookies_eaten (initial : ℕ) (given : ℕ) (left : ℕ) : ℕ :=
  initial - given - left

theorem paco_cookies : cookies_eaten 36 14 12 = 10 := by
  sorry

end NUMINAMATH_CALUDE_paco_cookies_l2568_256804


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2568_256862

theorem complex_equation_solution :
  ∀ z : ℂ, (1 + 2*I)*z = 4 + 3*I → z = 2 - I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2568_256862


namespace NUMINAMATH_CALUDE_units_digit_of_m_squared_plus_three_to_m_l2568_256854

def m : ℕ := 2021^3 + 3^2021

theorem units_digit_of_m_squared_plus_three_to_m (m : ℕ := 2021^3 + 3^2021) :
  (m^2 + 3^m) % 10 = 7 :=
by sorry

end NUMINAMATH_CALUDE_units_digit_of_m_squared_plus_three_to_m_l2568_256854


namespace NUMINAMATH_CALUDE_exam_score_deviation_l2568_256822

/-- Given an exam with mean score 74 and standard deviation σ, 
    prove that 58 is 2 standard deviations below the mean. -/
theorem exam_score_deviation :
  ∀ σ : ℝ,
  74 + 3 * σ = 98 →
  74 - 2 * σ = 58 :=
by
  sorry

end NUMINAMATH_CALUDE_exam_score_deviation_l2568_256822


namespace NUMINAMATH_CALUDE_distinct_necklaces_count_l2568_256864

/-- Represents a necklace made of white and black beads -/
structure Necklace :=
  (white_beads : ℕ)
  (black_beads : ℕ)

/-- Determines if two necklaces are equivalent under rotation and flipping -/
def necklace_equivalent (n1 n2 : Necklace) : Prop :=
  (n1.white_beads = n2.white_beads) ∧ (n1.black_beads = n2.black_beads)

/-- Counts the number of distinct necklaces with given white and black beads -/
def count_distinct_necklaces (white black : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the number of distinct necklaces with 5 white and 2 black beads is 3 -/
theorem distinct_necklaces_count :
  count_distinct_necklaces 5 2 = 3 :=
sorry

end NUMINAMATH_CALUDE_distinct_necklaces_count_l2568_256864


namespace NUMINAMATH_CALUDE_triangle_inequality_cube_l2568_256827

theorem triangle_inequality_cube (a b c : ℝ) 
  (h : 0 < a ∧ 0 < b ∧ 0 < c) 
  (triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b) : 
  a^3 + b^3 + 3*a*b*c > c^3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_cube_l2568_256827


namespace NUMINAMATH_CALUDE_parabola_triangle_area_l2568_256885

/-- The area of a triangle formed by a point on a parabola, its focus, and the origin -/
theorem parabola_triangle_area :
  ∀ (x y : ℝ),
  y^2 = 8*x →                   -- Point (x, y) is on the parabola y² = 8x
  (x - 2)^2 + y^2 = 5^2 →       -- Distance from (x, y) to focus (2, 0) is 5
  (1/2) * 2 * y = 2 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_parabola_triangle_area_l2568_256885


namespace NUMINAMATH_CALUDE_unique_element_implies_a_equals_four_l2568_256824

-- Define the set A
def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 + a * x + 1 = 0}

-- State the theorem
theorem unique_element_implies_a_equals_four :
  ∀ a : ℝ, (∃! x : ℝ, x ∈ A a) → a = 4 := by sorry

end NUMINAMATH_CALUDE_unique_element_implies_a_equals_four_l2568_256824


namespace NUMINAMATH_CALUDE_best_meeting_days_l2568_256806

-- Define the days of the week
inductive Day
| Mon
| Tue
| Wed
| Thu
| Fri

-- Define the team members
inductive Member
| Anna
| Bill
| Carl
| Dana

-- Define the availability function
def availability (m : Member) (d : Day) : Bool :=
  match m, d with
  | Member.Anna, Day.Mon => false
  | Member.Anna, Day.Wed => false
  | Member.Bill, Day.Tue => false
  | Member.Bill, Day.Thu => false
  | Member.Bill, Day.Fri => false
  | Member.Carl, Day.Mon => false
  | Member.Carl, Day.Tue => false
  | Member.Carl, Day.Thu => false
  | Member.Carl, Day.Fri => false
  | Member.Dana, Day.Wed => false
  | Member.Dana, Day.Thu => false
  | _, _ => true

-- Count available members for a given day
def availableCount (d : Day) : Nat :=
  (List.filter (fun m => availability m d) [Member.Anna, Member.Bill, Member.Carl, Member.Dana]).length

-- Define the maximum availability
def maxAvailability : Nat :=
  List.foldl max 0 (List.map availableCount [Day.Mon, Day.Tue, Day.Wed, Day.Thu, Day.Fri])

-- Theorem statement
theorem best_meeting_days :
  (availableCount Day.Mon = maxAvailability) ∧
  (availableCount Day.Tue = maxAvailability) ∧
  (availableCount Day.Wed = maxAvailability) ∧
  (availableCount Day.Thu < maxAvailability) ∧
  (availableCount Day.Fri = maxAvailability) := by
  sorry

end NUMINAMATH_CALUDE_best_meeting_days_l2568_256806


namespace NUMINAMATH_CALUDE_certain_number_multiplied_by_p_l2568_256851

theorem certain_number_multiplied_by_p (x : ℕ+) (p : ℕ) (n : ℕ) : 
  Nat.Prime p → 
  (x : ℕ) / (n * p) = 2 → 
  x ≥ 48 → 
  (∀ y : ℕ+, y < x → (y : ℕ) / (n * p) ≠ 2) →
  n = 12 := by sorry

end NUMINAMATH_CALUDE_certain_number_multiplied_by_p_l2568_256851


namespace NUMINAMATH_CALUDE_more_wins_probability_correct_l2568_256856

/-- The probability of winning, losing, or tying a single match -/
def match_probability : ℚ := 1/3

/-- The number of matches played -/
def num_matches : ℕ := 6

/-- The probability of finishing with more wins than losses -/
def more_wins_probability : ℚ := 98/243

theorem more_wins_probability_correct :
  let outcomes := 3^num_matches
  let equal_wins_losses := (num_matches.choose (num_matches/2))
                         + (num_matches.choose ((num_matches-2)/2)) * (num_matches.choose 2)
                         + (num_matches.choose ((num_matches-4)/2)) * (num_matches.choose 4)
                         + 1
  (1 - equal_wins_losses / outcomes) / 2 = more_wins_probability :=
sorry

end NUMINAMATH_CALUDE_more_wins_probability_correct_l2568_256856


namespace NUMINAMATH_CALUDE_abs_neg_five_l2568_256884

theorem abs_neg_five : |(-5 : ℝ)| = 5 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_five_l2568_256884


namespace NUMINAMATH_CALUDE_prob_blue_or_purple_l2568_256807

/-- A bag of jelly beans with different colors -/
structure JellyBeanBag where
  red : ℕ
  green : ℕ
  yellow : ℕ
  blue : ℕ
  purple : ℕ

/-- The probability of selecting either a blue or purple jelly bean -/
def bluePurpleProbability (bag : JellyBeanBag) : ℚ :=
  (bag.blue + bag.purple : ℚ) / (bag.red + bag.green + bag.yellow + bag.blue + bag.purple : ℚ)

/-- Theorem stating the probability of selecting a blue or purple jelly bean from the given bag -/
theorem prob_blue_or_purple (bag : JellyBeanBag) 
    (h : bag = { red := 7, green := 8, yellow := 9, blue := 10, purple := 4 }) : 
    bluePurpleProbability bag = 7 / 19 := by
  sorry

#eval bluePurpleProbability { red := 7, green := 8, yellow := 9, blue := 10, purple := 4 }

end NUMINAMATH_CALUDE_prob_blue_or_purple_l2568_256807


namespace NUMINAMATH_CALUDE_zoo_visitors_l2568_256860

theorem zoo_visitors (total_people : ℕ) (adult_price child_price : ℚ) (total_bill : ℚ) :
  total_people = 201 ∧ 
  adult_price = 8 ∧ 
  child_price = 4 ∧ 
  total_bill = 964 →
  ∃ (adults children : ℕ), 
    adults + children = total_people ∧
    adult_price * adults + child_price * children = total_bill ∧
    children = 161 := by
  sorry

end NUMINAMATH_CALUDE_zoo_visitors_l2568_256860


namespace NUMINAMATH_CALUDE_house_rent_percentage_l2568_256814

def total_income : ℝ := 1000
def petrol_percentage : ℝ := 0.3
def petrol_expenditure : ℝ := 300
def house_rent : ℝ := 210

theorem house_rent_percentage : 
  (house_rent / (total_income * (1 - petrol_percentage))) * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_house_rent_percentage_l2568_256814


namespace NUMINAMATH_CALUDE_ellipse_tangent_intersection_l2568_256882

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the line that P moves along
def line_P (x y : ℝ) : Prop := x + y = 3

-- Define a point on the ellipse
def point_on_ellipse (x y : ℝ) : Prop := ellipse x y

-- Define the tangent line at a point (x₀, y₀) on the ellipse
def tangent_line (x₀ y₀ x y : ℝ) : Prop :=
  point_on_ellipse x₀ y₀ → x₀*x/4 + y₀*y/3 = 1

-- Theorem statement
theorem ellipse_tangent_intersection :
  ∀ x₀ y₀ x₁ y₁ x₂ y₂,
    line_P x₀ y₀ →
    point_on_ellipse x₁ y₁ →
    point_on_ellipse x₂ y₂ →
    tangent_line x₁ y₁ x₀ y₀ →
    tangent_line x₂ y₂ x₀ y₀ →
    ∃ t, t*x₁ + (1-t)*x₂ = 4/3 ∧ t*y₁ + (1-t)*y₂ = 1 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_tangent_intersection_l2568_256882


namespace NUMINAMATH_CALUDE_negative_modulus_of_complex_l2568_256843

theorem negative_modulus_of_complex (z : ℂ) (h : z = 6 + 8*I) : -Complex.abs z = -10 := by
  sorry

end NUMINAMATH_CALUDE_negative_modulus_of_complex_l2568_256843


namespace NUMINAMATH_CALUDE_segments_form_quadrilateral_l2568_256825

/-- A function that checks if three line segments can form a quadrilateral with a fourth segment -/
def can_form_quadrilateral (a b c d : ℝ) : Prop :=
  a + b + c > d

/-- Theorem stating that line segments of length 2, 2, 2 can form a quadrilateral with a segment of length 5 -/
theorem segments_form_quadrilateral :
  can_form_quadrilateral 2 2 2 5 := by
  sorry

end NUMINAMATH_CALUDE_segments_form_quadrilateral_l2568_256825


namespace NUMINAMATH_CALUDE_count_1973_in_I_1000000_l2568_256803

-- Define the sequence type
def Sequence := List Nat

-- Define the initial sequence
def I₀ : Sequence := [1, 1]

-- Define the rule for generating the next sequence
def nextSequence (I : Sequence) : Sequence :=
  sorry

-- Define the n-th sequence
def Iₙ (n : Nat) : Sequence :=
  sorry

-- Define the count of a number in a sequence
def count (m : Nat) (I : Sequence) : Nat :=
  sorry

-- Euler's totient function
def φ (n : Nat) : Nat :=
  sorry

-- The main theorem
theorem count_1973_in_I_1000000 :
  count 1973 (Iₙ 1000000) = φ 1973 :=
sorry

end NUMINAMATH_CALUDE_count_1973_in_I_1000000_l2568_256803


namespace NUMINAMATH_CALUDE_sum_first_six_primes_mod_seventh_prime_l2568_256812

theorem sum_first_six_primes_mod_seventh_prime : 
  (2 + 3 + 5 + 7 + 11 + 13) % 17 = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_six_primes_mod_seventh_prime_l2568_256812


namespace NUMINAMATH_CALUDE_job_completion_time_l2568_256850

/-- If a group can complete a job in 20 days, twice the group can do half the job in 5 days -/
theorem job_completion_time (people : ℕ) (work : ℝ) : 
  (people * work = 20) → (2 * people) * (work / 2) = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l2568_256850


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l2568_256866

theorem quadratic_expression_value (x : ℝ) (h : x^2 - 2*x - 2 = 0) :
  3*x^2 - 6*x + 9 = 15 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l2568_256866


namespace NUMINAMATH_CALUDE_line_intersects_circle_shortest_chord_l2568_256811

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x + 1 - 2 * k

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 2*y - 7 = 0

-- Theorem 1: Line l always intersects circle C
theorem line_intersects_circle (k : ℝ) :
  ∃ x y : ℝ, line_l k x y ∧ circle_C x y :=
sorry

-- Theorem 2: The line x + 2y - 4 = 0 produces the shortest chord
theorem shortest_chord :
  ∀ k : ℝ,
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    x₁ ≠ x₂ ∧
    line_l k x₁ y₁ ∧ circle_C x₁ y₁ ∧
    line_l k x₂ y₂ ∧ circle_C x₂ y₂) →
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    x₁ ≠ x₂ ∧
    x₁ + 2*y₁ - 4 = 0 ∧ circle_C x₁ y₁ ∧
    x₂ + 2*y₂ - 4 = 0 ∧ circle_C x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 ≤ (x₁ - x₂)^2 + (y₁ - y₂)^2) :=
sorry

end NUMINAMATH_CALUDE_line_intersects_circle_shortest_chord_l2568_256811


namespace NUMINAMATH_CALUDE_saturday_extra_calories_l2568_256898

def daily_calories : ℕ := 2500
def daily_burn : ℕ := 3000
def weekly_deficit : ℕ := 2500
def days_in_week : ℕ := 7
def regular_days : ℕ := 6

def total_weekly_burn : ℕ := daily_burn * days_in_week
def regular_weekly_intake : ℕ := daily_calories * regular_days
def total_weekly_intake : ℕ := total_weekly_burn - weekly_deficit

theorem saturday_extra_calories :
  total_weekly_intake - regular_weekly_intake - daily_calories = 1000 := by
  sorry

end NUMINAMATH_CALUDE_saturday_extra_calories_l2568_256898


namespace NUMINAMATH_CALUDE_total_ladybugs_l2568_256876

theorem total_ladybugs (num_leaves : ℕ) (ladybugs_per_leaf : ℕ) 
  (h1 : num_leaves = 84) (h2 : ladybugs_per_leaf = 139) : 
  num_leaves * ladybugs_per_leaf = 11676 := by
  sorry

end NUMINAMATH_CALUDE_total_ladybugs_l2568_256876


namespace NUMINAMATH_CALUDE_sqrt_of_square_l2568_256879

theorem sqrt_of_square (x : ℝ) : Real.sqrt (x^2) = |x| := by sorry

end NUMINAMATH_CALUDE_sqrt_of_square_l2568_256879


namespace NUMINAMATH_CALUDE_original_function_equation_l2568_256887

/-- Given a vector OA and a quadratic function transformed by OA,
    prove that the original function has the form y = x^2 + 2x - 2 -/
theorem original_function_equation
  (OA : ℝ × ℝ)
  (h_OA : OA = (4, 3))
  (f : ℝ → ℝ)
  (h_f : ∀ x, f x = x^2 + b*x + c)
  (h_tangent : ∀ x y, y = f (x - 4) + 3 → (4*x + y - 8 = 0 ↔ x = 1 ∧ y = 4)) :
  b = 2 ∧ c = -2 :=
sorry

end NUMINAMATH_CALUDE_original_function_equation_l2568_256887


namespace NUMINAMATH_CALUDE_nine_qualified_possible_l2568_256881

/-- Represents the probability of a product passing inspection -/
def pass_rate : ℝ := 0.9

/-- The number of products drawn for inspection -/
def sample_size : ℕ := 10

/-- Represents whether it's possible to have exactly 9 qualified products in a sample of 10 -/
def possible_nine_qualified : Prop :=
  ∃ (p : ℝ), 0 ≤ p ∧ p ≤ 1 ∧ p ≠ 0 ∧ p ≠ 1

theorem nine_qualified_possible (h : pass_rate = 0.9) : possible_nine_qualified := by
  sorry

#check nine_qualified_possible

end NUMINAMATH_CALUDE_nine_qualified_possible_l2568_256881


namespace NUMINAMATH_CALUDE_min_max_cubic_minus_xy_squared_l2568_256842

/-- The function to be minimized -/
def f (x y : ℝ) : ℝ := |x^3 - x*y^2|

/-- The theorem statement -/
theorem min_max_cubic_minus_xy_squared :
  (∃ (m : ℝ), ∀ (y : ℝ), m ≤ (⨆ (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 2), f x y)) ∧
  (∀ (m : ℝ), (∀ (y : ℝ), m ≤ (⨆ (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 2), f x y)) → 8 ≤ m) :=
sorry

end NUMINAMATH_CALUDE_min_max_cubic_minus_xy_squared_l2568_256842


namespace NUMINAMATH_CALUDE_projectile_max_height_l2568_256830

/-- The height function of the projectile -/
def h (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 30

/-- The maximum height of the projectile -/
def max_height : ℝ := 155

/-- Theorem: The maximum height of the projectile is 155 feet -/
theorem projectile_max_height :
  ∀ t : ℝ, h t ≤ max_height :=
by
  sorry

end NUMINAMATH_CALUDE_projectile_max_height_l2568_256830


namespace NUMINAMATH_CALUDE_amy_total_tickets_l2568_256833

/-- Amy's initial number of tickets -/
def initial_tickets : ℕ := 33

/-- Number of tickets Amy bought additionally -/
def additional_tickets : ℕ := 21

/-- Theorem stating the total number of tickets Amy has -/
theorem amy_total_tickets : initial_tickets + additional_tickets = 54 := by
  sorry

end NUMINAMATH_CALUDE_amy_total_tickets_l2568_256833


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2568_256877

theorem inequality_system_solution (a b : ℝ) :
  (∀ x : ℝ, (2 * x - a < 1 ∧ x - 2 * b > 3) ↔ (-1 < x ∧ x < 1)) →
  (a + 1) * (b - 1) = -6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2568_256877


namespace NUMINAMATH_CALUDE_difference_between_numbers_difference_is_1356_l2568_256891

theorem difference_between_numbers : ℝ → Prop :=
  fun diff : ℝ =>
    let smaller : ℝ := 268.2
    let larger : ℝ := 6 * smaller + 15
    diff = larger - smaller

theorem difference_is_1356 : difference_between_numbers 1356 := by
  sorry

end NUMINAMATH_CALUDE_difference_between_numbers_difference_is_1356_l2568_256891


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2568_256874

def A : Set ℝ := {x | x^2 - 4 = 0}
def B : Set ℝ := {1, 2}

theorem intersection_of_A_and_B : A ∩ B = {2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2568_256874


namespace NUMINAMATH_CALUDE_age_ratio_proof_l2568_256863

def rahul_future_age : ℕ := 26
def years_to_future : ℕ := 2
def deepak_age : ℕ := 18

theorem age_ratio_proof :
  let rahul_age := rahul_future_age - years_to_future
  (rahul_age : ℚ) / deepak_age = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l2568_256863


namespace NUMINAMATH_CALUDE_jumping_contest_l2568_256837

/-- The jumping contest problem -/
theorem jumping_contest (grasshopper frog mouse : ℕ) 
  (h1 : grasshopper = 14)
  (h2 : mouse = frog - 16)
  (h3 : mouse = grasshopper + 21) :
  frog - grasshopper = 37 := by
  sorry

end NUMINAMATH_CALUDE_jumping_contest_l2568_256837


namespace NUMINAMATH_CALUDE_sum_of_ages_five_children_l2568_256844

/-- Calculates the sum of ages for a group of children born at regular intervals -/
def sumOfAges (numChildren : ℕ) (ageInterval : ℕ) (youngestAge : ℕ) : ℕ :=
  let ages := List.range numChildren |>.map (fun i => youngestAge + i * ageInterval)
  ages.sum

/-- Proves that the sum of ages for 5 children born at 2-year intervals, with the youngest being 6, is 50 -/
theorem sum_of_ages_five_children :
  sumOfAges 5 2 6 = 50 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_five_children_l2568_256844


namespace NUMINAMATH_CALUDE_subtraction_result_l2568_256848

theorem subtraction_result (number : ℝ) (percentage : ℝ) (subtrahend : ℝ) : 
  number = 200 → 
  percentage = 95 → 
  subtrahend = 12 → 
  (percentage / 100) * number - subtrahend = 178 := by
sorry

end NUMINAMATH_CALUDE_subtraction_result_l2568_256848


namespace NUMINAMATH_CALUDE_floor_painting_theorem_l2568_256847

/-- The number of ordered pairs (a,b) satisfying the floor painting conditions -/
def floor_painting_solutions : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 
    let (a, b) := p
    b > a ∧ (a - 4) * (b - 4) = 2 * a * b / 3
  ) (Finset.product (Finset.range 100) (Finset.range 100))).card

/-- Theorem stating that there are exactly 3 solutions to the floor painting problem -/
theorem floor_painting_theorem : floor_painting_solutions = 3 := by
  sorry

end NUMINAMATH_CALUDE_floor_painting_theorem_l2568_256847


namespace NUMINAMATH_CALUDE_quadratic_function_largest_m_l2568_256889

def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

def symmetric_about_neg_one (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x - 4) = f (2 - x)

def greater_than_or_equal_x (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x ≥ x

def less_than_or_equal_square (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ∈ Set.Ioo 0 2 → f x ≤ ((x + 1) / 2)^2

def min_value_zero (f : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, f x = 0 ∧ ∀ y : ℝ, f y ≥ f x

theorem quadratic_function_largest_m (a b c : ℝ) (h_a : a ≠ 0) :
  let f := quadratic_function a b c
  symmetric_about_neg_one f ∧
  greater_than_or_equal_x f ∧
  less_than_or_equal_square f ∧
  min_value_zero f →
  (∃ m : ℝ, m > 1 ∧
    (∃ t : ℝ, ∀ x : ℝ, x ∈ Set.Icc 1 m → f (x + t) ≤ x) ∧
    (∀ n : ℝ, n > m →
      ¬(∃ t : ℝ, ∀ x : ℝ, x ∈ Set.Icc 1 n → f (x + t) ≤ x))) ∧
  (∀ m : ℝ, m > 1 ∧
    (∃ t : ℝ, ∀ x : ℝ, x ∈ Set.Icc 1 m → f (x + t) ≤ x) ∧
    (∀ n : ℝ, n > m →
      ¬(∃ t : ℝ, ∀ x : ℝ, x ∈ Set.Icc 1 n → f (x + t) ≤ x)) →
    m = 9) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_largest_m_l2568_256889


namespace NUMINAMATH_CALUDE_factorial_fraction_simplification_l2568_256820

theorem factorial_fraction_simplification :
  (4 * Nat.factorial 6 + 24 * Nat.factorial 5) / Nat.factorial 7 = 8 / 7 := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_simplification_l2568_256820


namespace NUMINAMATH_CALUDE_total_worksheets_is_nine_l2568_256893

/-- Represents the grading problem for a teacher -/
structure GradingProblem where
  problems_per_worksheet : ℕ
  graded_worksheets : ℕ
  remaining_problems : ℕ

/-- Calculates the total number of worksheets to grade -/
def total_worksheets (gp : GradingProblem) : ℕ :=
  gp.graded_worksheets + (gp.remaining_problems / gp.problems_per_worksheet)

/-- Theorem stating that the total number of worksheets to grade is 9 -/
theorem total_worksheets_is_nine :
  ∀ (gp : GradingProblem),
    gp.problems_per_worksheet = 4 →
    gp.graded_worksheets = 5 →
    gp.remaining_problems = 16 →
    total_worksheets gp = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_total_worksheets_is_nine_l2568_256893


namespace NUMINAMATH_CALUDE_sculpture_and_base_height_l2568_256886

-- Define the height of the sculpture in inches
def sculpture_height : ℕ := 2 * 12 + 10

-- Define the height of the base in inches
def base_height : ℕ := 2

-- Define the total height in inches
def total_height : ℕ := sculpture_height + base_height

-- Theorem to prove
theorem sculpture_and_base_height :
  total_height / 12 = 3 := by sorry

end NUMINAMATH_CALUDE_sculpture_and_base_height_l2568_256886


namespace NUMINAMATH_CALUDE_fraction_equality_l2568_256840

theorem fraction_equality (x y : ℝ) (h : x ≠ y) : -x / (x - y) = x / (-x + y) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2568_256840


namespace NUMINAMATH_CALUDE_errors_per_debug_session_l2568_256821

theorem errors_per_debug_session 
  (total_lines : ℕ) 
  (debug_interval : ℕ) 
  (total_errors : ℕ) 
  (h1 : total_lines = 4300)
  (h2 : debug_interval = 100)
  (h3 : total_errors = 129) :
  total_errors / (total_lines / debug_interval) = 3 := by
sorry

end NUMINAMATH_CALUDE_errors_per_debug_session_l2568_256821


namespace NUMINAMATH_CALUDE_baker_cakes_l2568_256805

/-- Calculates the final number of cakes a baker has after selling some and buying new ones. -/
def final_cakes (initial : ℕ) (sold : ℕ) (bought : ℕ) : ℕ :=
  initial - sold + bought

/-- Proves that for the given numbers, the baker ends up with 186 cakes. -/
theorem baker_cakes : final_cakes 121 105 170 = 186 := by
  sorry

end NUMINAMATH_CALUDE_baker_cakes_l2568_256805


namespace NUMINAMATH_CALUDE_percentage_relation_l2568_256817

theorem percentage_relation (j p t m n : ℕ+) (r : ℚ) : 
  (j : ℚ) = 0.75 * p ∧
  (j : ℚ) = 0.80 * t ∧
  (t : ℚ) = p - (r / 100) * p ∧
  (m : ℚ) = 1.10 * p ∧
  (n : ℚ) = 0.70 * m ∧
  (j : ℚ) + p + t = m * n →
  r = 6.25 := by
sorry

end NUMINAMATH_CALUDE_percentage_relation_l2568_256817


namespace NUMINAMATH_CALUDE_valid_paths_count_l2568_256800

-- Define the grid dimensions
def rows : Nat := 5
def cols : Nat := 7

-- Define the blocked paths
def blocked_path1 : (Nat × Nat) × (Nat × Nat) := ((4, 2), (5, 2))
def blocked_path2 : (Nat × Nat) × (Nat × Nat) := ((2, 7), (3, 7))

-- Define a function to calculate valid paths
def valid_paths (r : Nat) (c : Nat) (blocked1 blocked2 : (Nat × Nat) × (Nat × Nat)) : Nat :=
  sorry

-- Theorem statement
theorem valid_paths_count : 
  valid_paths rows cols blocked_path1 blocked_path2 = 546 := by sorry

end NUMINAMATH_CALUDE_valid_paths_count_l2568_256800


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_when_inequality_holds_l2568_256815

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| - |a * x - 1|

-- Part 1
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x > 1} = {x : ℝ | x > 1/2} := by sorry

-- Part 2
theorem range_of_a_when_inequality_holds :
  ∀ a : ℝ, (∀ x ∈ Set.Ioo 0 1, f a x > x) ↔ a ∈ Set.Ioc 0 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_when_inequality_holds_l2568_256815


namespace NUMINAMATH_CALUDE_equation_D_is_linear_l2568_256857

/-- Definition of a linear equation in two variables -/
def is_linear_equation (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ (x y : ℝ), f x y = a * x + b * y + c

/-- The specific equation we want to prove is linear -/
def equation_D (x y : ℝ) : ℝ := 2 * x + y - 5

/-- Theorem stating that equation_D is a linear equation in two variables -/
theorem equation_D_is_linear : is_linear_equation equation_D := by
  sorry


end NUMINAMATH_CALUDE_equation_D_is_linear_l2568_256857


namespace NUMINAMATH_CALUDE_max_value_theorem_l2568_256873

theorem max_value_theorem (x y z : ℝ) (h : 9*x^2 + 4*y^2 + 25*z^2 = 1) :
  8*x + 5*y + 15*z ≤ 28 / Real.sqrt 38 :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2568_256873


namespace NUMINAMATH_CALUDE_fraction_of_fraction_tripled_l2568_256892

theorem fraction_of_fraction_tripled (a b c d : ℚ) : 
  a = 2 ∧ b = 3 ∧ c = 3 ∧ d = 8 → 
  3 * ((c / d) / (a / b)) = 27 / 16 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_fraction_tripled_l2568_256892


namespace NUMINAMATH_CALUDE_mango_boxes_count_l2568_256838

/-- Given a number of mangoes per dozen, total mangoes, and mangoes per box,
    calculate the number of boxes. -/
def calculate_boxes (mangoes_per_dozen : ℕ) (total_mangoes : ℕ) (dozens_per_box : ℕ) : ℕ :=
  total_mangoes / (mangoes_per_dozen * dozens_per_box)

/-- Prove that there are 36 boxes of mangoes given the problem conditions. -/
theorem mango_boxes_count :
  let mangoes_per_dozen : ℕ := 12
  let total_mangoes : ℕ := 4320
  let dozens_per_box : ℕ := 10
  calculate_boxes mangoes_per_dozen total_mangoes dozens_per_box = 36 := by
  sorry

#eval calculate_boxes 12 4320 10

end NUMINAMATH_CALUDE_mango_boxes_count_l2568_256838


namespace NUMINAMATH_CALUDE_candy_cost_calculation_l2568_256810

/-- The problem of calculating the total cost of candy -/
theorem candy_cost_calculation (cost_per_piece : ℕ) (num_gumdrops : ℕ) (total_cost : ℕ) : 
  cost_per_piece = 8 → num_gumdrops = 28 → total_cost = cost_per_piece * num_gumdrops → total_cost = 224 :=
by sorry

end NUMINAMATH_CALUDE_candy_cost_calculation_l2568_256810


namespace NUMINAMATH_CALUDE_problem_solution_l2568_256865

theorem problem_solution (x : ℚ) : 
  4 * x - 8 = 13 * x + 3 → 5 * (x - 2) = -145 / 9 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2568_256865


namespace NUMINAMATH_CALUDE_different_tens_digit_probability_l2568_256813

/-- The number of integers to be chosen -/
def n : ℕ := 6

/-- The lower bound of the range (inclusive) -/
def lower_bound : ℕ := 10

/-- The upper bound of the range (inclusive) -/
def upper_bound : ℕ := 79

/-- The total number of integers in the range -/
def total_numbers : ℕ := upper_bound - lower_bound + 1

/-- The number of different tens digits in the range -/
def tens_digits : ℕ := 7

/-- The probability of choosing n different integers from the range
    such that they each have a different tens digit -/
def probability : ℚ := 1750 / 2980131

theorem different_tens_digit_probability :
  probability = (tens_digits.choose n * (10 ^ n : ℕ)) / total_numbers.choose n :=
sorry

end NUMINAMATH_CALUDE_different_tens_digit_probability_l2568_256813


namespace NUMINAMATH_CALUDE_sphere_hemisphere_volume_ratio_l2568_256875

/-- The ratio of the volume of a sphere to the volume of a hemisphere -/
theorem sphere_hemisphere_volume_ratio (r : ℝ) (h : r > 0) :
  (4 / 3 * Real.pi * r ^ 3) / (1 / 2 * 4 / 3 * Real.pi * (3 * r) ^ 3) = 1 / 13.5 := by
  sorry

#check sphere_hemisphere_volume_ratio

end NUMINAMATH_CALUDE_sphere_hemisphere_volume_ratio_l2568_256875


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2568_256894

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^6 * (x + 1)^2 = a*x^8 + a₁*x^7 + a₂*x^6 + a₃*x^5 + a₄*x^4 + a₅*x^3 + a₆*x^2 + a₇*x + a₈) →
  a + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ = 4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2568_256894
