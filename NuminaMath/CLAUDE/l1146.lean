import Mathlib

namespace NUMINAMATH_CALUDE_bread_cost_l1146_114630

theorem bread_cost (initial_amount : ℕ) (amount_left : ℕ) (num_bread : ℕ) (num_milk : ℕ) 
  (h1 : initial_amount = 47)
  (h2 : amount_left = 35)
  (h3 : num_bread = 4)
  (h4 : num_milk = 2) :
  (initial_amount - amount_left) / (num_bread + num_milk) = 2 :=
by sorry

end NUMINAMATH_CALUDE_bread_cost_l1146_114630


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_minus_product_l1146_114605

theorem quadratic_roots_sum_minus_product (x₁ x₂ : ℝ) : 
  (x₁^2 - x₁ - 2022 = 0) → 
  (x₂^2 - x₂ - 2022 = 0) → 
  x₁ + x₂ - x₁ * x₂ = 2023 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_minus_product_l1146_114605


namespace NUMINAMATH_CALUDE_fourth_month_sales_l1146_114621

def sales_problem (sales1 sales2 sales3 sales5 sales6 : ℕ) (average : ℕ) : Prop :=
  let total := average * 6
  let known_sales := sales1 + sales2 + sales3 + sales5 + sales6
  total - known_sales = 7230

theorem fourth_month_sales :
  sales_problem 6735 6927 6855 6562 4691 6500 :=
by
  sorry

end NUMINAMATH_CALUDE_fourth_month_sales_l1146_114621


namespace NUMINAMATH_CALUDE_limit_special_function_l1146_114611

/-- The limit of ((x+1)/(2x))^((ln(x+2))/(ln(2-x))) as x approaches 1 is √3 -/
theorem limit_special_function :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 1| ∧ |x - 1| < δ →
    |((x + 1) / (2 * x))^((Real.log (x + 2)) / (Real.log (2 - x))) - Real.sqrt 3| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_special_function_l1146_114611


namespace NUMINAMATH_CALUDE_all_statements_correct_l1146_114696

theorem all_statements_correct :
  (∀ a b : ℕ, Odd a → Odd b → Even (a + b)) ∧
  (∀ p : ℕ, Prime p → p > 3 → ∃ k : ℕ, p^2 = 12*k + 1) ∧
  (∀ r : ℚ, ∀ i : ℝ, Irrational i → Irrational (r + i)) ∧
  (∀ n : ℕ, 2 ∣ n → 3 ∣ n → 6 ∣ n) ∧
  (∀ n : ℕ, n > 1 → Prime n ∨ ∃ (p : List ℕ), (∀ q ∈ p, Prime q) ∧ n = p.prod) :=
by sorry

end NUMINAMATH_CALUDE_all_statements_correct_l1146_114696


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l1146_114634

theorem negation_of_existence (p : ℝ → Prop) : 
  (¬ ∃ x : ℝ, p x) ↔ (∀ x : ℝ, ¬ p x) := by sorry

theorem negation_of_quadratic_equation : 
  (¬ ∃ x : ℝ, x^2 - 3*x + 2 = 0) ↔ (∀ x : ℝ, x^2 - 3*x + 2 ≠ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l1146_114634


namespace NUMINAMATH_CALUDE_revenue_is_405_main_theorem_l1146_114604

/-- Represents the rental business scenario --/
structure RentalBusiness where
  canoe_cost : ℕ
  kayak_cost : ℕ
  canoe_count : ℕ
  kayak_count : ℕ

/-- Calculates the total revenue for the rental business --/
def total_revenue (rb : RentalBusiness) : ℕ :=
  rb.canoe_cost * rb.canoe_count + rb.kayak_cost * rb.kayak_count

/-- Theorem stating that under the given conditions, the total revenue is $405 --/
theorem revenue_is_405 (rb : RentalBusiness) 
  (h1 : rb.canoe_cost = 15)
  (h2 : rb.kayak_cost = 18)
  (h3 : rb.canoe_count = (3 * rb.kayak_count) / 2)
  (h4 : rb.canoe_count = rb.kayak_count + 5) :
  total_revenue rb = 405 := by
  sorry

/-- Main theorem combining all conditions and proving the result --/
theorem main_theorem : ∃ (rb : RentalBusiness), 
  rb.canoe_cost = 15 ∧ 
  rb.kayak_cost = 18 ∧ 
  rb.canoe_count = (3 * rb.kayak_count) / 2 ∧
  rb.canoe_count = rb.kayak_count + 5 ∧
  total_revenue rb = 405 := by
  sorry

end NUMINAMATH_CALUDE_revenue_is_405_main_theorem_l1146_114604


namespace NUMINAMATH_CALUDE_multiple_power_divisibility_l1146_114669

theorem multiple_power_divisibility (a n m : ℕ) (ha : a > 0) : 
  m % (a^n) = 0 → (a + 1)^m - 1 % (a^(n+1)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_multiple_power_divisibility_l1146_114669


namespace NUMINAMATH_CALUDE_area_ratio_abc_xyz_l1146_114622

-- Define points as pairs of real numbers
def Point := ℝ × ℝ

-- Define the given points
def A : Point := (2, 0)
def B : Point := (8, 12)
def C : Point := (14, 0)
def X : Point := (6, 0)
def Y : Point := (8, 4)
def Z : Point := (10, 0)

-- Function to calculate the area of a triangle given three points
def triangleArea (p1 p2 p3 : Point) : ℝ := sorry

-- Theorem statement
theorem area_ratio_abc_xyz :
  (triangleArea X Y Z) / (triangleArea A B C) = 1 / 9 := by sorry

end NUMINAMATH_CALUDE_area_ratio_abc_xyz_l1146_114622


namespace NUMINAMATH_CALUDE_system_solution_unique_l1146_114614

theorem system_solution_unique (x y z : ℚ) : 
  x + 2*y - z = 100 ∧
  y - z = 25 ∧
  3*x - 5*y + 4*z = 230 →
  x = 101.25 ∧ y = -26.25 ∧ z = -51.25 := by
sorry

end NUMINAMATH_CALUDE_system_solution_unique_l1146_114614


namespace NUMINAMATH_CALUDE_f_max_min_range_l1146_114606

/-- A function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*((a+2)*x+1)

/-- The derivative of f(x) with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 6*a*x + 3*(a+2)

/-- Theorem stating the range of a for which f has both a maximum and minimum -/
theorem f_max_min_range (a : ℝ) : 
  (∃ (x₁ x₂ : ℝ), ∀ (x : ℝ), f a x₁ ≤ f a x ∧ f a x ≤ f a x₂) →
  a < -1 ∨ a > 2 :=
sorry

end NUMINAMATH_CALUDE_f_max_min_range_l1146_114606


namespace NUMINAMATH_CALUDE_largest_k_for_real_roots_l1146_114624

theorem largest_k_for_real_roots (k : ℤ) : 
  (∃ x : ℝ, x * (k * x + 1) - x^2 + 3 = 0) → 
  k ≠ 1 → 
  k ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_largest_k_for_real_roots_l1146_114624


namespace NUMINAMATH_CALUDE_work_completion_theorem_l1146_114692

/-- The number of men initially doing the work -/
def initial_men : ℕ := 50

/-- The number of days it takes for the initial number of men to complete the work -/
def initial_days : ℕ := 100

/-- The number of men needed to complete the work in 20 days -/
def men_for_20_days : ℕ := 250

/-- The number of days it takes for 250 men to complete the work -/
def days_for_250_men : ℕ := 20

theorem work_completion_theorem :
  initial_men * initial_days = men_for_20_days * days_for_250_men :=
by
  sorry

#check work_completion_theorem

end NUMINAMATH_CALUDE_work_completion_theorem_l1146_114692


namespace NUMINAMATH_CALUDE_other_sales_percentage_l1146_114668

/-- The percentage of sales for notebooks -/
def notebooks_sales : ℝ := 25

/-- The percentage of sales for markers -/
def markers_sales : ℝ := 40

/-- The total percentage of all sales -/
def total_sales : ℝ := 100

/-- Theorem: The percentage of sales that were neither notebooks nor markers is 35% -/
theorem other_sales_percentage : 
  total_sales - (notebooks_sales + markers_sales) = 35 := by
  sorry

end NUMINAMATH_CALUDE_other_sales_percentage_l1146_114668


namespace NUMINAMATH_CALUDE_quadratic_function_proof_l1146_114616

/-- The quadratic function a(x) -/
def a (x : ℝ) : ℝ := 2*x^2 - 14*x + 20

/-- The shape function y = 2x² -/
def shape (x : ℝ) : ℝ := 2*x^2

theorem quadratic_function_proof :
  a 2 = 0 ∧ a 5 = 0 ∧ ∃ k, ∀ x, a x = k * shape x + (a 0 - k * shape 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_proof_l1146_114616


namespace NUMINAMATH_CALUDE_total_bike_ride_l1146_114658

def morning_ride : ℝ := 2
def evening_ride_factor : ℝ := 5

theorem total_bike_ride : morning_ride + evening_ride_factor * morning_ride = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_bike_ride_l1146_114658


namespace NUMINAMATH_CALUDE_cost_per_roof_tile_is_10_l1146_114686

/-- Represents the construction costs for a house. -/
structure ConstructionCosts where
  landCostPerSqMeter : ℕ
  brickCostPer1000 : ℕ
  requiredLandArea : ℕ
  requiredBricks : ℕ
  requiredRoofTiles : ℕ
  totalCost : ℕ

/-- Calculates the cost per roof tile given the construction costs. -/
def costPerRoofTile (costs : ConstructionCosts) : ℕ :=
  let landCost := costs.landCostPerSqMeter * costs.requiredLandArea
  let brickCost := (costs.requiredBricks / 1000) * costs.brickCostPer1000
  let roofTileCost := costs.totalCost - (landCost + brickCost)
  roofTileCost / costs.requiredRoofTiles

/-- Theorem stating that the cost per roof tile is $10 given the specified construction costs. -/
theorem cost_per_roof_tile_is_10 (costs : ConstructionCosts)
    (h1 : costs.landCostPerSqMeter = 50)
    (h2 : costs.brickCostPer1000 = 100)
    (h3 : costs.requiredLandArea = 2000)
    (h4 : costs.requiredBricks = 10000)
    (h5 : costs.requiredRoofTiles = 500)
    (h6 : costs.totalCost = 106000) :
    costPerRoofTile costs = 10 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_roof_tile_is_10_l1146_114686


namespace NUMINAMATH_CALUDE_company_average_salary_l1146_114607

theorem company_average_salary
  (num_managers : ℕ)
  (num_associates : ℕ)
  (avg_salary_managers : ℚ)
  (avg_salary_associates : ℚ)
  (h1 : num_managers = 15)
  (h2 : num_associates = 75)
  (h3 : avg_salary_managers = 90000)
  (h4 : avg_salary_associates = 30000) :
  (num_managers * avg_salary_managers + num_associates * avg_salary_associates) / (num_managers + num_associates) = 40000 := by
sorry

end NUMINAMATH_CALUDE_company_average_salary_l1146_114607


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_l1146_114618

theorem closest_integer_to_cube_root : ∃ n : ℤ, 
  n = 10 ∧ ∀ m : ℤ, |n - (7^3 + 9^3)^(1/3)| ≤ |m - (7^3 + 9^3)^(1/3)| := by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_l1146_114618


namespace NUMINAMATH_CALUDE_dance_team_initial_members_l1146_114661

theorem dance_team_initial_members (initial_members quit_members new_members current_members : ℕ) 
  (h1 : quit_members = 8)
  (h2 : new_members = 13)
  (h3 : current_members = 30)
  (h4 : current_members = initial_members - quit_members + new_members) : 
  initial_members = 25 := by
  sorry

end NUMINAMATH_CALUDE_dance_team_initial_members_l1146_114661


namespace NUMINAMATH_CALUDE_jungkook_has_smallest_number_l1146_114673

def yoongi_number : ℕ := 7
def jungkook_number : ℕ := 6
def yuna_number : ℕ := 9

theorem jungkook_has_smallest_number :
  jungkook_number ≤ yoongi_number ∧ jungkook_number ≤ yuna_number :=
by
  sorry

end NUMINAMATH_CALUDE_jungkook_has_smallest_number_l1146_114673


namespace NUMINAMATH_CALUDE_factorize_quadratic_minimum_value_quadratic_sum_abc_l1146_114660

-- Problem 1
theorem factorize_quadratic (m : ℝ) : m^2 - 6*m + 5 = (m - 1)*(m - 5) := by sorry

-- Problem 2
theorem minimum_value_quadratic (a b : ℝ) :
  a^2 + b^2 - 4*a + 10*b + 33 ≥ 4 ∧
  (a^2 + b^2 - 4*a + 10*b + 33 = 4 ↔ a = 2 ∧ b = -5) := by sorry

-- Problem 3
theorem sum_abc (a b c : ℝ) (h1 : a - b = 8) (h2 : a*b + c^2 - 4*c + 20 = 0) :
  a + b + c = 2 := by sorry

end NUMINAMATH_CALUDE_factorize_quadratic_minimum_value_quadratic_sum_abc_l1146_114660


namespace NUMINAMATH_CALUDE_angle_terminal_side_point_l1146_114695

theorem angle_terminal_side_point (α : Real) :
  let P : ℝ × ℝ := (4, -3)
  (P.1 = 4 ∧ P.2 = -3) →
  2 * Real.sin α + Real.cos α = -2/5 := by
sorry

end NUMINAMATH_CALUDE_angle_terminal_side_point_l1146_114695


namespace NUMINAMATH_CALUDE_matches_for_128_teams_l1146_114638

/-- Represents a single-elimination tournament -/
structure Tournament where
  num_teams : ℕ
  num_teams_positive : 0 < num_teams

/-- The number of matches required to determine the championship team -/
def matches_required (t : Tournament) : ℕ :=
  t.num_teams - 1

/-- Theorem: In a tournament with 128 teams, 127 matches are required -/
theorem matches_for_128_teams :
  ∀ t : Tournament, t.num_teams = 128 → matches_required t = 127 := by
  sorry

#check matches_for_128_teams

end NUMINAMATH_CALUDE_matches_for_128_teams_l1146_114638


namespace NUMINAMATH_CALUDE_invalid_deduction_from_false_premise_l1146_114677

-- Define the concept of a premise
def Premise : Type := Prop

-- Define the concept of a conclusion
def Conclusion : Type := Prop

-- Define the concept of a deduction
def Deduction := Premise → Conclusion

-- Define what it means for a premise to be false
def IsFalsePremise (p : Premise) : Prop := ¬p

-- Define what it means for a conclusion to be valid
def IsValidConclusion (c : Conclusion) : Prop := c

-- Theorem: Logical deductions based on false premises cannot lead to valid conclusions
theorem invalid_deduction_from_false_premise :
  ∀ (p : Premise) (d : Deduction),
    IsFalsePremise p → ¬(IsValidConclusion (d p)) :=
by sorry

end NUMINAMATH_CALUDE_invalid_deduction_from_false_premise_l1146_114677


namespace NUMINAMATH_CALUDE_equal_probability_sums_l1146_114651

/-- Represents a standard six-sided die -/
def Die := Fin 6

/-- The number of dice being rolled -/
def numDice : ℕ := 8

/-- The sum we're comparing to -/
def targetSum : ℕ := 12

/-- Function to calculate the complementary sum -/
def complementarySum (n : ℕ) : ℕ := 2 * (numDice * 3 + numDice) - n

/-- Theorem stating that the sum of 44 occurs with the same probability as the sum of 12 -/
theorem equal_probability_sums :
  complementarySum targetSum = 44 := by
  sorry

end NUMINAMATH_CALUDE_equal_probability_sums_l1146_114651


namespace NUMINAMATH_CALUDE_no_matrix_sin_B_l1146_114682

def B : Matrix (Fin 2) (Fin 2) ℝ := !![1, 1996; 0, 1]

-- Define sin(A) using power series
noncomputable def matrix_sin (A : Matrix (Fin 2) (Fin 2) ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  A - (A^3 / 6) + (A^5 / 120) - (A^7 / 5040) + (A^9 / 362880) - (A^11 / 39916800) + higher_order_terms
where
  higher_order_terms := sorry  -- Represents the rest of the infinite series

theorem no_matrix_sin_B : ¬ ∃ (A : Matrix (Fin 2) (Fin 2) ℝ), matrix_sin A = B := by
  sorry

end NUMINAMATH_CALUDE_no_matrix_sin_B_l1146_114682


namespace NUMINAMATH_CALUDE_remaining_subtasks_l1146_114663

def total_problems : ℝ := 72.0
def completed_problems : ℝ := 32.0
def subtasks_per_problem : ℕ := 5

theorem remaining_subtasks : 
  (total_problems - completed_problems) * subtasks_per_problem = 200 := by
  sorry

end NUMINAMATH_CALUDE_remaining_subtasks_l1146_114663


namespace NUMINAMATH_CALUDE_power_tower_mod_500_l1146_114655

theorem power_tower_mod_500 : 7^(7^(7^7)) ≡ 543 [ZMOD 500] := by
  sorry

end NUMINAMATH_CALUDE_power_tower_mod_500_l1146_114655


namespace NUMINAMATH_CALUDE_max_value_problem_l1146_114632

theorem max_value_problem (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) 
  (h4 : x^2 + y^2 + z^2 = 1) : 
  3 * x * z * Real.sqrt 2 + 9 * y * z ≤ Real.sqrt 27 := by
sorry

end NUMINAMATH_CALUDE_max_value_problem_l1146_114632


namespace NUMINAMATH_CALUDE_lcm_problem_l1146_114643

theorem lcm_problem (m : ℕ+) (h1 : Nat.lcm 36 m = 180) (h2 : Nat.lcm m 50 = 300) : m = 60 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l1146_114643


namespace NUMINAMATH_CALUDE_bolts_per_box_l1146_114672

theorem bolts_per_box (bolt_boxes : ℕ) (nut_boxes : ℕ) (nuts_per_box : ℕ) 
  (bolts_left : ℕ) (nuts_left : ℕ) (bolts_and_nuts_used : ℕ) :
  bolt_boxes = 7 →
  nut_boxes = 3 →
  nuts_per_box = 15 →
  bolts_left = 3 →
  nuts_left = 6 →
  bolts_and_nuts_used = 113 →
  ∃ (bolts_per_box : ℕ),
    bolt_boxes * bolts_per_box + nut_boxes * nuts_per_box = 
    bolts_and_nuts_used + bolts_left + nuts_left ∧
    bolts_per_box = 11 :=
by sorry

end NUMINAMATH_CALUDE_bolts_per_box_l1146_114672


namespace NUMINAMATH_CALUDE_minimum_mass_for_upward_roll_l1146_114626

/-- Given a cylinder of mass M on rails inclined at angle α = 45°, 
    the minimum mass m of a weight attached to a string wound around the cylinder 
    for it to roll upward without slipping is M(√2 + 1) -/
theorem minimum_mass_for_upward_roll (M : ℝ) (α : ℝ) 
    (h_α : α = π / 4) : 
    ∃ m : ℝ, m = M * (Real.sqrt 2 + 1) ∧ 
    m * (1 - Real.sin α) = M * Real.sin α := by
  sorry

end NUMINAMATH_CALUDE_minimum_mass_for_upward_roll_l1146_114626


namespace NUMINAMATH_CALUDE_inspection_decision_l1146_114657

/-- Represents the probability of an item being defective -/
def p : Real := 0.1

/-- Total number of items in a box -/
def totalItems : Nat := 200

/-- Number of items in the initial sample -/
def sampleSize : Nat := 20

/-- Number of defective items found in the sample -/
def defectivesInSample : Nat := 2

/-- Cost of inspecting one item -/
def inspectionCost : Real := 2

/-- Compensation fee for one defective item -/
def compensationFee : Real := 25

/-- Expected number of defective items in the remaining items -/
def expectedDefectives : Real := (totalItems - sampleSize) * p

/-- Expected cost without further inspection -/
def expectedCostWithoutInspection : Real :=
  sampleSize * inspectionCost + expectedDefectives * compensationFee

/-- Cost of inspecting all items -/
def costOfInspectingAll : Real := totalItems * inspectionCost

theorem inspection_decision :
  expectedCostWithoutInspection > costOfInspectingAll :=
sorry

end NUMINAMATH_CALUDE_inspection_decision_l1146_114657


namespace NUMINAMATH_CALUDE_min_value_expression_l1146_114688

theorem min_value_expression (a b m n : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_m : 0 < m) (h_pos_n : 0 < n)
  (h_sum : a + b = 1) (h_prod : m * n = 2) :
  2 ≤ (a * m + b * n) * (b * m + a * n) :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l1146_114688


namespace NUMINAMATH_CALUDE_a_equals_3_sufficient_not_necessary_l1146_114694

def A (a : ℕ) : Set ℕ := {1, a}
def B : Set ℕ := {1, 2, 3}

theorem a_equals_3_sufficient_not_necessary :
  (∀ a : ℕ, a = 3 → A a ⊆ B) ∧
  (∃ a : ℕ, A a ⊆ B ∧ a ≠ 3) :=
by sorry

end NUMINAMATH_CALUDE_a_equals_3_sufficient_not_necessary_l1146_114694


namespace NUMINAMATH_CALUDE_marbles_problem_l1146_114679

theorem marbles_problem (jinwoo seonghyeon cheolsu : ℕ) : 
  jinwoo = (2 * seonghyeon) / 3 →
  cheolsu = 72 →
  jinwoo + cheolsu = 2 * seonghyeon →
  jinwoo = 36 := by
  sorry

end NUMINAMATH_CALUDE_marbles_problem_l1146_114679


namespace NUMINAMATH_CALUDE_video_game_players_l1146_114646

/-- The number of players who quit the game -/
def players_quit : ℕ := 5

/-- The number of lives each remaining player has -/
def lives_per_player : ℕ := 5

/-- The total number of lives for remaining players -/
def total_lives : ℕ := 30

/-- The initial number of players in the game -/
def initial_players : ℕ := players_quit + total_lives / lives_per_player

theorem video_game_players :
  initial_players = 11 :=
by sorry

end NUMINAMATH_CALUDE_video_game_players_l1146_114646


namespace NUMINAMATH_CALUDE_equation_is_hyperbola_l1146_114601

/-- A conic section type -/
inductive ConicSection
  | Parabola
  | Circle
  | Ellipse
  | Hyperbola
  | Point
  | Line
  | TwoLines
  | Empty

/-- Determines the type of conic section for a given quadratic equation -/
def determineConicSection (a b c d e f : ℝ) : ConicSection :=
  sorry

/-- The equation x^2 - 4y^2 - 2x + 8y - 8 = 0 represents a hyperbola -/
theorem equation_is_hyperbola :
  determineConicSection 1 (-4) 0 (-2) 8 (-8) = ConicSection.Hyperbola :=
sorry

end NUMINAMATH_CALUDE_equation_is_hyperbola_l1146_114601


namespace NUMINAMATH_CALUDE_prob_five_eight_sided_dice_l1146_114627

/-- The number of sides on each die -/
def n : ℕ := 8

/-- The number of dice rolled -/
def k : ℕ := 5

/-- The probability of at least two dice showing the same number when rolling k fair n-sided dice -/
def prob_at_least_two_same (n k : ℕ) : ℚ :=
  1 - (n.factorial / (n - k).factorial : ℚ) / n^k

theorem prob_five_eight_sided_dice :
  prob_at_least_two_same n k = 3256 / 4096 :=
sorry

end NUMINAMATH_CALUDE_prob_five_eight_sided_dice_l1146_114627


namespace NUMINAMATH_CALUDE_nine_sequence_sum_to_1989_l1146_114675

theorem nine_sequence_sum_to_1989 : ∃ (a b c : ℕ), 
  a + b + c = 9999999 ∧ 
  a ≤ 999 ∧ b ≤ 999 ∧ c ≤ 999 ∧
  a + b - c = 1989 := by
sorry

end NUMINAMATH_CALUDE_nine_sequence_sum_to_1989_l1146_114675


namespace NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l1146_114653

/-- Two hyperbolas have the same asymptotes if M = 4.5 -/
theorem hyperbolas_same_asymptotes :
  let h₁ : ℝ → ℝ → Prop := λ x y => x^2 / 9 - y^2 / 16 = 1
  let h₂ : ℝ → ℝ → ℝ → Prop := λ x y M => y^2 / 8 - x^2 / M = 1
  let asymptote₁ : ℝ → ℝ → Prop := λ x y => y = (4/3) * x ∨ y = -(4/3) * x
  let asymptote₂ : ℝ → ℝ → ℝ → Prop := λ x y M => y = Real.sqrt (8/M) * x ∨ y = -Real.sqrt (8/M) * x
  ∀ (M : ℝ), (∀ x y, asymptote₁ x y ↔ asymptote₂ x y M) → M = 4.5 :=
by sorry

end NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l1146_114653


namespace NUMINAMATH_CALUDE_f_derivative_at_one_is_zero_g_derivative_formula_l1146_114629

noncomputable section

def f (x : ℝ) : ℝ := Real.exp x / x

def g (x : ℝ) : ℝ := f (2 * x)

theorem f_derivative_at_one_is_zero :
  deriv f 1 = 0 := by sorry

theorem g_derivative_formula (x : ℝ) (h : x ≠ 0) :
  deriv g x = (Real.exp (2 * x) * (2 * x - 1)) / (2 * x^2) := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_one_is_zero_g_derivative_formula_l1146_114629


namespace NUMINAMATH_CALUDE_circle_coordinates_l1146_114640

theorem circle_coordinates (π : ℝ) (h : π > 0) :
  let radii : List ℝ := [2, 4, 6, 8, 10]
  let circumference (r : ℝ) : ℝ := 2 * π * r
  let area (r : ℝ) : ℝ := π * r^2
  let coordinates := radii.map (λ r => (circumference r, area r))
  coordinates = [(4*π, 4*π), (8*π, 16*π), (12*π, 36*π), (16*π, 64*π), (20*π, 100*π)] :=
by sorry

end NUMINAMATH_CALUDE_circle_coordinates_l1146_114640


namespace NUMINAMATH_CALUDE_alex_grocery_delivery_l1146_114662

theorem alex_grocery_delivery (saved : ℝ) (car_cost : ℝ) (trip_charge : ℝ) (grocery_fee_percent : ℝ) (num_trips : ℕ) 
  (h1 : saved = 14500)
  (h2 : car_cost = 14600)
  (h3 : trip_charge = 1.5)
  (h4 : grocery_fee_percent = 0.05)
  (h5 : num_trips = 40) :
  ∃ (grocery_value : ℝ), 
    grocery_value * grocery_fee_percent = car_cost - saved - (trip_charge * num_trips) ∧ 
    grocery_value = 800 := by
sorry

end NUMINAMATH_CALUDE_alex_grocery_delivery_l1146_114662


namespace NUMINAMATH_CALUDE_quadratic_function_transformation_l1146_114602

-- Define the quadratic function f(x) = ax² + bx + c
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the function g(x) = cx² + 2bx + a
def g (a b c : ℝ) (x : ℝ) : ℝ := c * x^2 + 2 * b * x + a

theorem quadratic_function_transformation (a b c : ℝ) :
  (f a b c 0 = 1) ∧ 
  (f a b c 1 = -2) ∧ 
  (f a b c (-1) = 2) →
  (∀ x, g a b c x = x^2 - 4*x - 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_transformation_l1146_114602


namespace NUMINAMATH_CALUDE_mango_price_proof_l1146_114654

/-- The cost of a single lemon in dollars -/
def lemon_cost : ℚ := 2

/-- The cost of a single papaya in dollars -/
def papaya_cost : ℚ := 1

/-- The number of fruits required to get a discount -/
def fruits_for_discount : ℕ := 4

/-- The discount amount in dollars -/
def discount_amount : ℚ := 1

/-- The number of lemons Tom bought -/
def lemons_bought : ℕ := 6

/-- The number of papayas Tom bought -/
def papayas_bought : ℕ := 4

/-- The number of mangos Tom bought -/
def mangos_bought : ℕ := 2

/-- The total amount Tom paid in dollars -/
def total_paid : ℚ := 21

/-- The cost of a single mango in dollars -/
def mango_cost : ℚ := 4

theorem mango_price_proof :
  let total_fruits := lemons_bought + papayas_bought + mangos_bought
  let total_discounts := (total_fruits / fruits_for_discount : ℚ)
  let total_discount_amount := total_discounts * discount_amount
  let total_cost_before_discount := lemon_cost * lemons_bought + papaya_cost * papayas_bought + mango_cost * mangos_bought
  total_cost_before_discount - total_discount_amount = total_paid :=
sorry

end NUMINAMATH_CALUDE_mango_price_proof_l1146_114654


namespace NUMINAMATH_CALUDE_candles_from_beehives_l1146_114639

/-- Given that 3 beehives can make enough wax for 12 candles,
    prove that 24 beehives can make enough wax for 96 candles. -/
theorem candles_from_beehives :
  ∀ (beehives candles : ℕ),
    beehives = 3 →
    candles = 12 →
    (24 : ℕ) * candles / beehives = 96 :=
by
  sorry

end NUMINAMATH_CALUDE_candles_from_beehives_l1146_114639


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l1146_114676

theorem rectangle_diagonal (l w : ℝ) (h1 : l = 8) (h2 : 2 * l + 2 * w = 46) :
  Real.sqrt (l^2 + w^2) = 17 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l1146_114676


namespace NUMINAMATH_CALUDE_choose_four_from_ten_l1146_114612

theorem choose_four_from_ten (n : ℕ) (k : ℕ) : n = 10 → k = 4 → Nat.choose n k = 210 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_from_ten_l1146_114612


namespace NUMINAMATH_CALUDE_probability_three_tails_l1146_114610

def coin_flips : ℕ := 8
def p_tails : ℚ := 3/5
def p_heads : ℚ := 2/5
def num_tails : ℕ := 3

theorem probability_three_tails :
  (Nat.choose coin_flips num_tails : ℚ) * p_tails ^ num_tails * p_heads ^ (coin_flips - num_tails) = 48624/390625 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_tails_l1146_114610


namespace NUMINAMATH_CALUDE_tangent_line_at_one_e_l1146_114683

/-- The tangent line to y = xe^x at (1, e) -/
theorem tangent_line_at_one_e :
  let f (x : ℝ) := x * Real.exp x
  let f' (x : ℝ) := Real.exp x + x * Real.exp x
  let tangent_line (x : ℝ) := 2 * Real.exp 1 * x - Real.exp 1
  f' 1 = 2 * Real.exp 1 ∧
  tangent_line 1 = f 1 ∧
  ∀ x, tangent_line x - f x = f' 1 * (x - 1) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_e_l1146_114683


namespace NUMINAMATH_CALUDE_deepak_age_l1146_114699

theorem deepak_age (arun_age deepak_age : ℕ) : 
  (arun_age : ℚ) / deepak_age = 4 / 3 →
  arun_age + 6 = 26 →
  deepak_age = 15 := by
sorry

end NUMINAMATH_CALUDE_deepak_age_l1146_114699


namespace NUMINAMATH_CALUDE_even_composition_is_even_l1146_114687

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem even_composition_is_even (f : ℝ → ℝ) (h : IsEven f) : IsEven (f ∘ f) := by
  sorry

end NUMINAMATH_CALUDE_even_composition_is_even_l1146_114687


namespace NUMINAMATH_CALUDE_chicken_cost_l1146_114693

def initial_amount : Int := 55
def banana_packs : Int := 2
def banana_cost : Int := 4
def pear_cost : Int := 2
def asparagus_cost : Int := 6
def remaining_amount : Int := 28

theorem chicken_cost : 
  initial_amount - (banana_packs * banana_cost + pear_cost + asparagus_cost) - remaining_amount = 11 := by
  sorry

end NUMINAMATH_CALUDE_chicken_cost_l1146_114693


namespace NUMINAMATH_CALUDE_least_positive_integer_divisible_by_four_primes_l1146_114685

theorem least_positive_integer_divisible_by_four_primes : ∃ n : ℕ, 
  (∃ p₁ p₂ p₃ p₄ : ℕ, Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
    n % p₁ = 0 ∧ n % p₂ = 0 ∧ n % p₃ = 0 ∧ n % p₄ = 0) ∧
  (∀ m : ℕ, m < n → 
    ¬(∃ q₁ q₂ q₃ q₄ : ℕ, Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄ ∧ 
      q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₃ ≠ q₄ ∧
      m % q₁ = 0 ∧ m % q₂ = 0 ∧ m % q₃ = 0 ∧ m % q₄ = 0)) ∧
  n = 210 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_divisible_by_four_primes_l1146_114685


namespace NUMINAMATH_CALUDE_simplified_fourth_root_l1146_114670

theorem simplified_fourth_root (c d : ℕ+) :
  (2^5 * 5^3 : ℝ)^(1/4) = c * d^(1/4) → c + d = 252 := by
  sorry

end NUMINAMATH_CALUDE_simplified_fourth_root_l1146_114670


namespace NUMINAMATH_CALUDE_square_side_length_average_l1146_114628

theorem square_side_length_average (a b c : ℝ) 
  (ha : a = 25) (hb : b = 64) (hc : c = 225) : 
  (Real.sqrt a + Real.sqrt b + Real.sqrt c) / 3 = 28 / 3 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_average_l1146_114628


namespace NUMINAMATH_CALUDE_solve_a_and_b_l1146_114603

def A : Set ℝ := {x | -2 < x ∧ x < -1 ∨ x > 1}

def B (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b ≤ 0}

theorem solve_a_and_b :
  ∃ (a b : ℝ),
    (A ∪ B a b = {x | x > -2}) ∧
    (A ∩ B a b = {x | 1 < x ∧ x ≤ 3}) ∧
    a = -4 ∧ b = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_a_and_b_l1146_114603


namespace NUMINAMATH_CALUDE_circle_line_intersection_k_range_l1146_114644

/-- Given a circle and a line, if there exists a point on the line such that a circle 
    with this point as its center and radius 1 has a common point with the given circle, 
    then k is within a specific range. -/
theorem circle_line_intersection_k_range :
  ∀ (k : ℝ),
  (∃ (x y : ℝ), x^2 + y^2 + 4*x + 3 = 0 ∧ y = k*x - 1 ∧
   ∃ (x₀ y₀ : ℝ), y₀ = k*x₀ - 1 ∧ 
   ∃ (x₁ y₁ : ℝ), (x₁ - x₀)^2 + (y₁ - y₀)^2 = 1 ∧ x₁^2 + y₁^2 + 4*x₁ + 3 = 0) →
  -4/3 ≤ k ∧ k ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_line_intersection_k_range_l1146_114644


namespace NUMINAMATH_CALUDE_water_consumption_proof_l1146_114674

/-- Proves that drinking 500 milliliters every 2 hours for 12 hours results in 3 liters of water consumption. -/
theorem water_consumption_proof (liters_goal : ℝ) (ml_per_interval : ℝ) (hours_per_interval : ℝ) :
  liters_goal = 3 ∧ ml_per_interval = 500 ∧ hours_per_interval = 2 →
  (liters_goal * 1000) / ml_per_interval * hours_per_interval = 12 := by
  sorry

end NUMINAMATH_CALUDE_water_consumption_proof_l1146_114674


namespace NUMINAMATH_CALUDE_combined_apples_l1146_114698

/-- The number of apples Sara ate -/
def sara_apples : ℕ := 16

/-- The ratio of apples Ali ate compared to Sara -/
def ali_ratio : ℕ := 4

/-- The total number of apples eaten by Ali and Sara -/
def total_apples : ℕ := sara_apples + ali_ratio * sara_apples

theorem combined_apples : total_apples = 80 := by
  sorry

end NUMINAMATH_CALUDE_combined_apples_l1146_114698


namespace NUMINAMATH_CALUDE_supplementary_angles_ratio_l1146_114665

theorem supplementary_angles_ratio (a b : ℝ) : 
  a + b = 180 →  -- angles are supplementary
  a / b = 5 / 4 →  -- angles are in ratio 5:4
  b = 80 :=  -- smaller angle is 80°
by sorry

end NUMINAMATH_CALUDE_supplementary_angles_ratio_l1146_114665


namespace NUMINAMATH_CALUDE_max_ac_value_l1146_114691

theorem max_ac_value (a c x y z m n : ℤ) : 
  x^2 + a*x + 48 = (x + y)*(x + z) →
  x^2 - 8*x + c = (x + m)*(x + n) →
  y ≥ -50 → y ≤ 50 →
  z ≥ -50 → z ≤ 50 →
  m ≥ -50 → m ≤ 50 →
  n ≥ -50 → n ≤ 50 →
  ∃ (a' c' : ℤ), a'*c' = 98441 ∧ ∀ (a'' c'' : ℤ), a''*c'' ≤ 98441 :=
by sorry

end NUMINAMATH_CALUDE_max_ac_value_l1146_114691


namespace NUMINAMATH_CALUDE_luke_coin_count_l1146_114642

/-- Represents the number of coins in each pile of quarters --/
def quarter_piles : List Nat := [4, 4, 6, 6, 6, 8]

/-- Represents the number of coins in each pile of dimes --/
def dime_piles : List Nat := [3, 5, 2, 2]

/-- Represents the number of coins in each pile of nickels --/
def nickel_piles : List Nat := [5, 5, 5, 7, 7, 10]

/-- Represents the number of coins in each pile of pennies --/
def penny_piles : List Nat := [12, 8, 20]

/-- Represents the number of coins in each pile of half dollars --/
def half_dollar_piles : List Nat := [2, 4]

/-- The total number of coins Luke has --/
def total_coins : Nat := quarter_piles.sum + dime_piles.sum + nickel_piles.sum + 
                         penny_piles.sum + half_dollar_piles.sum

theorem luke_coin_count : total_coins = 131 := by
  sorry

end NUMINAMATH_CALUDE_luke_coin_count_l1146_114642


namespace NUMINAMATH_CALUDE_emily_trivia_score_l1146_114656

/-- Emily's trivia game score calculation -/
theorem emily_trivia_score (first_round : ℤ) (last_round : ℤ) (final_score : ℤ) 
  (h1 : first_round = 16)
  (h2 : last_round = -48)
  (h3 : final_score = 1) :
  ∃ second_round : ℤ, first_round + second_round + last_round = final_score ∧ second_round = 33 := by
  sorry

end NUMINAMATH_CALUDE_emily_trivia_score_l1146_114656


namespace NUMINAMATH_CALUDE_remainder_theorem_l1146_114637

theorem remainder_theorem : (1225^3 * 1227^4 * 1229^5) % 36 = 9 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1146_114637


namespace NUMINAMATH_CALUDE_triangle_properties_l1146_114689

-- Define the triangle ABC
variable (A B C : ℝ) -- Angles of the triangle
variable (a b c : ℝ) -- Sides of the triangle opposite to A, B, C respectively

-- Define the conditions
axiom bc_cos_a : b * Real.cos A = 2
axiom area : (1/2) * b * c * Real.sin A = 2
axiom sin_relation : Real.sin B = 2 * Real.cos A * Real.sin C

-- Define the theorem
theorem triangle_properties :
  (Real.tan A = 2) ∧ (c = 5) :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l1146_114689


namespace NUMINAMATH_CALUDE_not_fourth_ABE_l1146_114652

-- Define the set of runners
inductive Runner : Type
  | A | B | C | D | E | F

-- Define the ordering relation for runners
def beats : Runner → Runner → Prop := sorry

-- Define the race result as a function from position to runner
def raceResult : Nat → Runner := sorry

-- State the given conditions
axiom A_beats_B : beats Runner.A Runner.B
axiom A_beats_C : beats Runner.A Runner.C
axiom B_beats_D : beats Runner.B Runner.D
axiom B_beats_E : beats Runner.B Runner.E
axiom C_beats_F : beats Runner.C Runner.F
axiom E_after_B_before_C : beats Runner.B Runner.E ∧ beats Runner.E Runner.C

-- Define what it means to finish in a certain position
def finishesIn (r : Runner) (pos : Nat) : Prop :=
  raceResult pos = r

-- State the theorem
theorem not_fourth_ABE :
  ¬(finishesIn Runner.A 4) ∧ ¬(finishesIn Runner.B 4) ∧ ¬(finishesIn Runner.E 4) :=
by sorry

end NUMINAMATH_CALUDE_not_fourth_ABE_l1146_114652


namespace NUMINAMATH_CALUDE_max_savings_is_90_l1146_114608

structure Airline where
  name : String
  originalPrice : ℕ
  discountPercentage : ℕ

def calculateDiscountedPrice (airline : Airline) : ℕ :=
  airline.originalPrice - (airline.originalPrice * airline.discountPercentage / 100)

def airlines : List Airline := [
  { name := "Delta", originalPrice := 850, discountPercentage := 20 },
  { name := "United", originalPrice := 1100, discountPercentage := 30 },
  { name := "American", originalPrice := 950, discountPercentage := 25 },
  { name := "Southwest", originalPrice := 900, discountPercentage := 15 },
  { name := "JetBlue", originalPrice := 1200, discountPercentage := 40 }
]

theorem max_savings_is_90 :
  let discountedPrices := airlines.map calculateDiscountedPrice
  let cheapestPrice := discountedPrices.minimum?
  let maxSavings := discountedPrices.map (fun price => price - cheapestPrice.getD 0)
  maxSavings.maximum? = some 90 := by
  sorry

end NUMINAMATH_CALUDE_max_savings_is_90_l1146_114608


namespace NUMINAMATH_CALUDE_painted_cube_problem_l1146_114619

theorem painted_cube_problem (n : ℕ) : n > 0 →
  (4 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1 / 3 → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_painted_cube_problem_l1146_114619


namespace NUMINAMATH_CALUDE_parabola_properties_l1146_114623

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 8*y

-- Define the focus
def focus : ℝ × ℝ := (0, 2)

-- Define the directrix
def directrix (x y : ℝ) : Prop := y = -2

-- Define a point on the parabola
def on_parabola (p : ℝ × ℝ) : Prop := parabola p.1 p.2

-- Define a point on the directrix
def on_directrix (p : ℝ × ℝ) : Prop := directrix p.1 p.2

-- Define the condition PF = FE
def PF_equals_FE (P E : ℝ × ℝ) : Prop :=
  (P.1 - focus.1)^2 + (P.2 - focus.2)^2 = (E.1 - focus.1)^2 + (E.2 - focus.2)^2

-- Define the dot product of two vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem parabola_properties :
  ∀ (P E : ℝ × ℝ),
  on_directrix P →
  on_parabola E →
  E.1 > 0 →
  E.2 > 0 →
  PF_equals_FE P E →
  (∃ (k : ℝ), k * P.1 - P.2 + 2 = 0 ∧ k = 1/Real.sqrt 3) ∧
  (∀ (D : ℝ × ℝ), on_parabola D →
    dot_product (D.1 - P.1, D.2 - P.2) (E.1 - P.1, E.2 - P.2) ≤ -64) ∧
  (∃ (P' : ℝ × ℝ), on_directrix P' ∧
    (P'.1 = 4 ∨ P'.1 = -4) ∧ P'.2 = -2 ∧
    (∀ (D E : ℝ × ℝ), on_parabola D → on_parabola E →
      dot_product (D.1 - P'.1, D.2 - P'.2) (E.1 - P'.1, E.2 - P'.2) = -64)) :=
sorry

end NUMINAMATH_CALUDE_parabola_properties_l1146_114623


namespace NUMINAMATH_CALUDE_place_value_sum_l1146_114650

/-- Given place values, prove the total number -/
theorem place_value_sum (thousands hundreds tens ones : ℕ) :
  thousands = 6 →
  hundreds = 3 →
  tens = 9 →
  ones = 7 →
  thousands * 1000 + hundreds * 100 + tens * 10 + ones = 6397 := by
  sorry

end NUMINAMATH_CALUDE_place_value_sum_l1146_114650


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_R_l1146_114664

/-- The solution set of a quadratic inequality is R iff a < 0 and discriminant < 0 -/
theorem quadratic_inequality_solution_set_R 
  (a b c : ℝ) (h : a ≠ 0) : 
  (∀ x, a * x^2 + b * x + c < 0) ↔ (a < 0 ∧ b^2 - 4*a*c < 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_R_l1146_114664


namespace NUMINAMATH_CALUDE_alternating_squares_sum_l1146_114671

theorem alternating_squares_sum : 
  23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 4^2 - 2^2 = 272 := by
  sorry

end NUMINAMATH_CALUDE_alternating_squares_sum_l1146_114671


namespace NUMINAMATH_CALUDE_fred_initial_money_l1146_114680

/-- Fred's money situation --/
def fred_money_problem (initial_money current_money weekend_earnings : ℕ) : Prop :=
  initial_money + weekend_earnings = current_money

theorem fred_initial_money : 
  ∃ (initial_money : ℕ), fred_money_problem initial_money 86 63 ∧ initial_money = 23 :=
sorry

end NUMINAMATH_CALUDE_fred_initial_money_l1146_114680


namespace NUMINAMATH_CALUDE_percentage_of_sum_l1146_114645

theorem percentage_of_sum (x y : ℝ) (P : ℝ) 
  (h1 : 0.5 * (x - y) = (P / 100) * (x + y)) 
  (h2 : y = 0.25 * x) : 
  P = 30 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_sum_l1146_114645


namespace NUMINAMATH_CALUDE_third_number_in_ratio_l1146_114659

theorem third_number_in_ratio (a b c : ℝ) : 
  a / 5 = b / 6 ∧ b / 6 = c / 8 ∧  -- numbers are in ratio 5 : 6 : 8
  a + c = b + 49 →                -- sum of longest and smallest equals sum of third and 49
  b = 42 :=                       -- prove that the third number (b) is 42
by sorry

end NUMINAMATH_CALUDE_third_number_in_ratio_l1146_114659


namespace NUMINAMATH_CALUDE_sum_of_digits_power_product_l1146_114649

def power_product : ℕ := 2^2009 * 5^2010 * 7

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_power_product : sum_of_digits power_product = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_power_product_l1146_114649


namespace NUMINAMATH_CALUDE_largest_share_is_12000_l1146_114625

/-- Represents the profit split ratio for four partners -/
structure ProfitSplit :=
  (a b c d : ℕ)

/-- Calculates the largest share given a total profit and a profit split ratio -/
def largest_share (total_profit : ℕ) (split : ProfitSplit) : ℕ :=
  let total_parts := split.a + split.b + split.c + split.d
  let largest_part := max split.a (max split.b (max split.c split.d))
  (total_profit / total_parts) * largest_part

/-- The theorem stating that the largest share is $12,000 -/
theorem largest_share_is_12000 :
  largest_share 30000 ⟨1, 4, 4, 6⟩ = 12000 := by
  sorry

#eval largest_share 30000 ⟨1, 4, 4, 6⟩

end NUMINAMATH_CALUDE_largest_share_is_12000_l1146_114625


namespace NUMINAMATH_CALUDE_workshop_selection_l1146_114681

/-- The number of ways to select workers for a repair job. -/
def selectWorkers (totalWorkers fitters turners masterWorkers : ℕ) : ℕ :=
  let remainingWorkers := totalWorkers - turners
  let remainingFitters := fitters + masterWorkers
  let scenario1 := Nat.choose remainingWorkers 4
  let scenario2 := Nat.choose turners 3 * Nat.choose masterWorkers 1 * Nat.choose (remainingFitters - 1) 4
  let scenario3 := Nat.choose turners 2 * Nat.choose fitters 4
  scenario1 + scenario2 + scenario3

/-- Theorem stating the number of ways to select workers for the given problem. -/
theorem workshop_selection :
  selectWorkers 11 5 4 2 = 185 := by
  sorry

end NUMINAMATH_CALUDE_workshop_selection_l1146_114681


namespace NUMINAMATH_CALUDE_henrys_cd_collection_l1146_114690

theorem henrys_cd_collection :
  ∀ (classical rock country : ℕ),
    classical = 10 →
    rock = 2 * classical →
    country = rock + 3 →
    country = 23 :=
by
  sorry

end NUMINAMATH_CALUDE_henrys_cd_collection_l1146_114690


namespace NUMINAMATH_CALUDE_jefferson_bananas_l1146_114609

theorem jefferson_bananas (jefferson_bananas : ℕ) (walter_bananas : ℕ) : 
  walter_bananas = jefferson_bananas - (1/4 : ℚ) * jefferson_bananas →
  (jefferson_bananas + walter_bananas) / 2 = 49 →
  jefferson_bananas = 56 := by
sorry

end NUMINAMATH_CALUDE_jefferson_bananas_l1146_114609


namespace NUMINAMATH_CALUDE_committee_selection_ways_l1146_114631

def club_size : ℕ := 30
def committee_size : ℕ := 5

theorem committee_selection_ways :
  Nat.choose club_size committee_size = 142506 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_ways_l1146_114631


namespace NUMINAMATH_CALUDE_regular_octagon_extended_sides_angle_l1146_114620

-- Define a regular octagon
structure RegularOctagon :=
  (vertices : Fin 8 → ℝ × ℝ)
  (is_regular : ∀ i j : Fin 8, dist (vertices i) (vertices ((i + 1) % 8)) = dist (vertices j) (vertices ((j + 1) % 8)))

-- Define the extension of sides CD and FG
def extend_sides (octagon : RegularOctagon) : ℝ × ℝ :=
  sorry

-- Define the angle at point Q
def angle_at_Q (octagon : RegularOctagon) : ℝ :=
  sorry

-- Theorem statement
theorem regular_octagon_extended_sides_angle (octagon : RegularOctagon) :
  angle_at_Q octagon = 180 :=
sorry

end NUMINAMATH_CALUDE_regular_octagon_extended_sides_angle_l1146_114620


namespace NUMINAMATH_CALUDE_solution_eq1_solution_eq2_l1146_114636

-- Define the average method for quadratic equations
def average_method (a b c : ℝ) : Set ℝ :=
  let avg := (a + b) / 2
  let diff := b - avg
  {x | (x + avg)^2 - diff^2 = c}

-- Theorem for the first equation
theorem solution_eq1 : 
  average_method 2 8 40 = {2, -12} := by sorry

-- Theorem for the second equation
theorem solution_eq2 : 
  average_method (-2) 6 4 = {-2 + 2 * Real.sqrt 5, -2 - 2 * Real.sqrt 5} := by sorry

end NUMINAMATH_CALUDE_solution_eq1_solution_eq2_l1146_114636


namespace NUMINAMATH_CALUDE_average_monthly_growth_rate_correct_l1146_114667

/-- The average monthly growth rate of a factory's production volume -/
def average_monthly_growth_rate (a : ℝ) : ℝ := a^(1/11) - 1

/-- Theorem stating that the average monthly growth rate is correct -/
theorem average_monthly_growth_rate_correct (a : ℝ) (h : a > 0) :
  (1 + average_monthly_growth_rate a)^11 = a :=
by sorry

end NUMINAMATH_CALUDE_average_monthly_growth_rate_correct_l1146_114667


namespace NUMINAMATH_CALUDE_speed_increase_proof_l1146_114647

def distance : ℝ := 210
def forward_time : ℝ := 7
def return_time : ℝ := 5

theorem speed_increase_proof :
  let forward_speed := distance / forward_time
  let return_speed := distance / return_time
  return_speed - forward_speed = 12 := by
  sorry

end NUMINAMATH_CALUDE_speed_increase_proof_l1146_114647


namespace NUMINAMATH_CALUDE_icosahedron_edges_l1146_114615

/-- A regular icosahedron is a polyhedron with 20 faces and 12 vertices, 
    where each vertex is connected to 5 edges. -/
structure RegularIcosahedron where
  faces : ℕ
  vertices : ℕ
  edges_per_vertex : ℕ
  faces_eq : faces = 20
  vertices_eq : vertices = 12
  edges_per_vertex_eq : edges_per_vertex = 5

/-- The number of edges in a regular icosahedron is 30. -/
theorem icosahedron_edges (i : RegularIcosahedron) : 
  (i.vertices * i.edges_per_vertex) / 2 = 30 := by
  sorry

#check icosahedron_edges

end NUMINAMATH_CALUDE_icosahedron_edges_l1146_114615


namespace NUMINAMATH_CALUDE_square_root_of_four_l1146_114641

theorem square_root_of_four (x : ℝ) : x^2 = 4 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_four_l1146_114641


namespace NUMINAMATH_CALUDE_cherry_pies_count_l1146_114678

/-- Given a total number of pies and a ratio for distribution among three types,
    calculate the number of pies of the third type. -/
def calculate_third_type_pies (total : ℕ) (ratio1 ratio2 ratio3 : ℕ) : ℕ :=
  let ratio_sum := ratio1 + ratio2 + ratio3
  let pies_per_part := total / ratio_sum
  ratio3 * pies_per_part

/-- Theorem stating that given 40 pies distributed in the ratio 2:5:3,
    the number of cherry pies (third type) is 12. -/
theorem cherry_pies_count :
  calculate_third_type_pies 40 2 5 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_cherry_pies_count_l1146_114678


namespace NUMINAMATH_CALUDE_opposite_of_one_sixth_l1146_114684

theorem opposite_of_one_sixth :
  -(1 / 6 : ℚ) = -1 / 6 := by sorry

end NUMINAMATH_CALUDE_opposite_of_one_sixth_l1146_114684


namespace NUMINAMATH_CALUDE_manny_marbles_after_sharing_l1146_114617

/-- Given a total number of marbles and a ratio, calculates the number of marbles for each part -/
def marbles_per_part (total : ℕ) (ratio_sum : ℕ) : ℕ := total / ratio_sum

/-- Calculates the initial number of marbles for a person given their ratio part and marbles per part -/
def initial_marbles (ratio_part : ℕ) (marbles_per_part : ℕ) : ℕ := ratio_part * marbles_per_part

/-- Calculates the final number of marbles after giving away some -/
def final_marbles (initial : ℕ) (given_away : ℕ) : ℕ := initial - given_away

theorem manny_marbles_after_sharing (total_marbles : ℕ) (mario_ratio : ℕ) (manny_ratio : ℕ) (shared_marbles : ℕ) :
  total_marbles = 36 →
  mario_ratio = 4 →
  manny_ratio = 5 →
  shared_marbles = 2 →
  final_marbles (initial_marbles manny_ratio (marbles_per_part total_marbles (mario_ratio + manny_ratio))) shared_marbles = 18 := by
  sorry

end NUMINAMATH_CALUDE_manny_marbles_after_sharing_l1146_114617


namespace NUMINAMATH_CALUDE_solution_set_when_a_zero_range_of_a_no_solution_l1146_114635

-- Define the function f(x) = |2x+2| - |x-1|
def f (x : ℝ) : ℝ := |2*x + 2| - |x - 1|

-- Part 1: Solution set when a = 0
theorem solution_set_when_a_zero :
  {x : ℝ | f x > 0} = {x : ℝ | x < -3 ∨ x > -1/3} := by sorry

-- Part 2: Range of a when no solution in [-4, 2]
theorem range_of_a_no_solution :
  ∀ a : ℝ, (∀ x ∈ Set.Icc (-4 : ℝ) 2, f x ≤ a) → a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_zero_range_of_a_no_solution_l1146_114635


namespace NUMINAMATH_CALUDE_triangle_inequality_l1146_114613

/-- Theorem: For any triangle with side lengths a, b, c and perimeter 2, 
    the inequality a^2 + b^2 + c^2 < 2(1 - abc) holds. -/
theorem triangle_inequality (a b c : ℝ) 
  (triangle_cond : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) 
  (perimeter_cond : a + b + c = 2) : 
  a^2 + b^2 + c^2 < 2*(1 - a*b*c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1146_114613


namespace NUMINAMATH_CALUDE_incorrect_inequality_l1146_114600

theorem incorrect_inequality (a b : ℝ) (h : a > b) : ¬(-2 * a > -2 * b) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_inequality_l1146_114600


namespace NUMINAMATH_CALUDE_phoenix_airport_on_time_rate_l1146_114666

/-- Calculates the on-time departure rate given the number of on-time departures and total flights -/
def onTimeRate (onTime : ℕ) (total : ℕ) : ℚ :=
  onTime / total

/-- Proves that adding one more on-time flight after 3 on-time and 1 late flight 
    results in an on-time rate higher than 60% -/
theorem phoenix_airport_on_time_rate : 
  let initialOnTime : ℕ := 3
  let initialTotal : ℕ := 4
  let additionalOnTime : ℕ := 1
  onTimeRate (initialOnTime + additionalOnTime) (initialTotal + additionalOnTime) > 60 / 100 := by
  sorry

#eval onTimeRate 4 5 > 60 / 100

end NUMINAMATH_CALUDE_phoenix_airport_on_time_rate_l1146_114666


namespace NUMINAMATH_CALUDE_prob_diamond_ace_king_l1146_114648

/-- The number of cards in the modified deck -/
def deck_size : ℕ := 56

/-- The number of cards that are either diamonds, aces, or kings -/
def target_cards : ℕ := 20

/-- The probability of drawing a card that is not a diamond, ace, or king -/
def prob_not_target : ℚ := (deck_size - target_cards) / deck_size

/-- The probability of drawing at least one diamond, ace, or king in two draws with replacement -/
def prob_at_least_one_target : ℚ := 1 - prob_not_target^2

theorem prob_diamond_ace_king : prob_at_least_one_target = 115 / 196 := by
  sorry

end NUMINAMATH_CALUDE_prob_diamond_ace_king_l1146_114648


namespace NUMINAMATH_CALUDE_tan_alpha_value_l1146_114697

theorem tan_alpha_value (α : Real) (h : Real.tan (α + π / 4) = 1 / 7) : 
  Real.tan α = -3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l1146_114697


namespace NUMINAMATH_CALUDE_line_maximizing_midpoint_distance_l1146_114633

/-- The equation of a line that intercepts a circle, maximizing the distance from the origin to the chord's midpoint -/
theorem line_maximizing_midpoint_distance 
  (x y a b c : ℝ) 
  (circle_eq : x^2 + y^2 = 16)
  (line_eq : a*x + b*y + c = 0)
  (condition : a + 2*b - c = 0)
  (is_max : ∀ (x' y' : ℝ), x'^2 + y'^2 ≤ (x^2 + y^2) / 4) :
  x + 2*y + 5 = 0 := by
sorry

end NUMINAMATH_CALUDE_line_maximizing_midpoint_distance_l1146_114633
