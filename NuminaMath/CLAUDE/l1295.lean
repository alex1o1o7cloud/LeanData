import Mathlib

namespace NUMINAMATH_CALUDE_initial_customers_l1295_129540

/-- The number of customers remaining after some customers left -/
def remaining_customers : ℕ := 5

/-- The number of customers who left -/
def departed_customers : ℕ := 3

/-- Theorem: The initial number of customers was 8 -/
theorem initial_customers : remaining_customers + departed_customers = 8 := by
  sorry

end NUMINAMATH_CALUDE_initial_customers_l1295_129540


namespace NUMINAMATH_CALUDE_function_properties_l1295_129595

def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*m*x + 10

theorem function_properties (m : ℝ) (h1 : m > 1) (h2 : f m m = 1) :
  ∃ (g : ℝ → ℝ),
    (∀ x, g x = x^2 - 6*x + 10) ∧
    (∀ x ∈ Set.Icc 3 5, g x ≤ 5) ∧
    (∀ x ∈ Set.Icc 3 5, g x ≥ 1) ∧
    (∃ x ∈ Set.Icc 3 5, g x = 5) ∧
    (∃ x ∈ Set.Icc 3 5, g x = 1) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1295_129595


namespace NUMINAMATH_CALUDE_grape_juice_mixture_proof_l1295_129524

/-- Proves that adding 10 gallons of grape juice to 40 gallons of a mixture 
    containing 10% grape juice results in a new mixture with 28.000000000000004% grape juice. -/
theorem grape_juice_mixture_proof : 
  let initial_mixture : ℝ := 40
  let initial_concentration : ℝ := 0.1
  let added_juice : ℝ := 10
  let final_concentration : ℝ := 0.28000000000000004
  (initial_mixture * initial_concentration + added_juice) / (initial_mixture + added_juice) = final_concentration := by
  sorry

end NUMINAMATH_CALUDE_grape_juice_mixture_proof_l1295_129524


namespace NUMINAMATH_CALUDE_cylinder_volume_equalization_l1295_129573

/-- The increase in radius and height that equalizes volumes of two cylinders --/
theorem cylinder_volume_equalization (r h : ℝ) (increase : ℝ) : 
  r = 5 ∧ h = 10 ∧ increase > 0 → 
  π * (r + increase)^2 * h = π * r^2 * (h + increase) → 
  increase = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_equalization_l1295_129573


namespace NUMINAMATH_CALUDE_fraction_over_65_l1295_129500

theorem fraction_over_65 (total : ℕ) (under_21 : ℕ) (over_65 : ℕ) : 
  (3 : ℚ) / 7 * total = under_21 →
  50 < total →
  total < 100 →
  under_21 = 33 →
  (over_65 : ℚ) / total = over_65 / 77 :=
by sorry

end NUMINAMATH_CALUDE_fraction_over_65_l1295_129500


namespace NUMINAMATH_CALUDE_integral_f_equals_two_l1295_129598

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ := 
  if -1 ≤ x ∧ x ≤ 1 then x^3 + Real.sin x
  else if 1 < x ∧ x ≤ 2 then 2
  else 0  -- We need to define f for all real numbers

-- State the theorem
theorem integral_f_equals_two : 
  ∫ x in (-1)..(2), f x = 2 := by sorry

end NUMINAMATH_CALUDE_integral_f_equals_two_l1295_129598


namespace NUMINAMATH_CALUDE_number_division_problem_l1295_129582

theorem number_division_problem : ∃ x : ℝ, x / 5 = 40 + x / 6 ∧ x = 7200 / 31 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l1295_129582


namespace NUMINAMATH_CALUDE_john_net_profit_l1295_129554

def gross_income : ℝ := 30000
def car_purchase_price : ℝ := 20000
def monthly_maintenance_cost : ℝ := 300
def annual_insurance_cost : ℝ := 1200
def tire_replacement_cost : ℝ := 400
def car_trade_in_value : ℝ := 6000
def tax_rate : ℝ := 0.15

def total_maintenance_cost : ℝ := monthly_maintenance_cost * 12
def car_depreciation : ℝ := car_purchase_price - car_trade_in_value
def total_expenses : ℝ := total_maintenance_cost + annual_insurance_cost + tire_replacement_cost + car_depreciation
def taxes : ℝ := tax_rate * gross_income
def net_profit : ℝ := gross_income - total_expenses - taxes

theorem john_net_profit : net_profit = 6300 := by
  sorry

end NUMINAMATH_CALUDE_john_net_profit_l1295_129554


namespace NUMINAMATH_CALUDE_twenty_one_less_than_sixty_thousand_l1295_129516

theorem twenty_one_less_than_sixty_thousand : 60000 - 21 = 59979 := by
  sorry

end NUMINAMATH_CALUDE_twenty_one_less_than_sixty_thousand_l1295_129516


namespace NUMINAMATH_CALUDE_sequence_sum_equals_square_l1295_129528

-- Define the sequence sum function
def sequenceSum (n : ℕ) : ℕ :=
  2 * (List.range n).sum + n

-- State the theorem
theorem sequence_sum_equals_square (n : ℕ) :
  n > 0 → sequenceSum n = n ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_equals_square_l1295_129528


namespace NUMINAMATH_CALUDE_dogs_left_over_l1295_129553

theorem dogs_left_over (total_dogs : ℕ) (num_houses : ℕ) (h1 : total_dogs = 50) (h2 : num_houses = 17) : 
  total_dogs - (num_houses * (total_dogs / num_houses)) = 16 := by
sorry

end NUMINAMATH_CALUDE_dogs_left_over_l1295_129553


namespace NUMINAMATH_CALUDE_journeymen_ratio_after_layoff_journeymen_fraction_is_two_thirds_l1295_129504

/-- The total number of employees in the anvil factory -/
def total_employees : ℕ := 20210

/-- The fraction of employees who are journeymen -/
def journeymen_fraction : ℚ := sorry

/-- The number of journeymen after laying off half of them -/
def remaining_journeymen : ℚ := journeymen_fraction * (total_employees : ℚ) / 2

/-- The total number of employees after laying off half of the journeymen -/
def remaining_employees : ℚ := (total_employees : ℚ) - remaining_journeymen

/-- The condition that after laying off half of the journeymen, they constitute 50% of the remaining workforce -/
theorem journeymen_ratio_after_layoff : remaining_journeymen / remaining_employees = 1 / 2 := sorry

/-- The main theorem: proving that the fraction of employees who are journeymen is 2/3 -/
theorem journeymen_fraction_is_two_thirds : journeymen_fraction = 2 / 3 := sorry

end NUMINAMATH_CALUDE_journeymen_ratio_after_layoff_journeymen_fraction_is_two_thirds_l1295_129504


namespace NUMINAMATH_CALUDE_pie_piece_price_l1295_129559

/-- Represents the price of a single piece of pie -/
def price_per_piece : ℝ := 3.83

/-- Represents the number of pieces a single pie is divided into -/
def pieces_per_pie : ℕ := 3

/-- Represents the number of pies the bakery can make in one hour -/
def pies_per_hour : ℕ := 12

/-- Represents the cost to create one pie -/
def cost_per_pie : ℝ := 0.5

/-- Represents the total revenue from selling all pie pieces -/
def total_revenue : ℝ := 138

theorem pie_piece_price :
  price_per_piece * (pieces_per_pie * pies_per_hour) = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_pie_piece_price_l1295_129559


namespace NUMINAMATH_CALUDE_min_packages_correct_l1295_129513

/-- The minimum number of packages Mary must deliver to cover the cost of her bicycle -/
def min_packages : ℕ :=
  let bicycle_cost : ℕ := 800
  let revenue_per_package : ℕ := 12
  let maintenance_cost_per_package : ℕ := 4
  let profit_per_package : ℕ := revenue_per_package - maintenance_cost_per_package
  (bicycle_cost + profit_per_package - 1) / profit_per_package

theorem min_packages_correct : min_packages = 100 := by
  sorry

end NUMINAMATH_CALUDE_min_packages_correct_l1295_129513


namespace NUMINAMATH_CALUDE_composite_has_at_least_three_factors_l1295_129546

/-- A positive integer is composite if it has a factor other than 1 and itself. -/
def IsComposite (n : ℕ) : Prop :=
  ∃ k : ℕ, 1 < k ∧ k < n ∧ k ∣ n

/-- The number of factors of a natural number. -/
def numFactors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

/-- Theorem: A composite number has at least three factors. -/
theorem composite_has_at_least_three_factors (n : ℕ) (h : IsComposite n) :
    3 ≤ numFactors n := by
  sorry

end NUMINAMATH_CALUDE_composite_has_at_least_three_factors_l1295_129546


namespace NUMINAMATH_CALUDE_painted_cube_theorem_l1295_129541

/-- Represents a 6x6x6 cube painted on three sets of opposite faces -/
structure PaintedCube where
  size : Nat
  total_cubelets : Nat
  single_color_cubelets : Nat

/-- The number of cubelets with exactly one face painted for a given color -/
def cubelets_per_color (cube : PaintedCube) : Nat :=
  cube.single_color_cubelets / 3

theorem painted_cube_theorem (cube : PaintedCube) 
  (h1 : cube.size = 6)
  (h2 : cube.total_cubelets = 216)
  (h3 : cube.single_color_cubelets = 96) :
  cubelets_per_color cube = 32 := by
  sorry

#check painted_cube_theorem

end NUMINAMATH_CALUDE_painted_cube_theorem_l1295_129541


namespace NUMINAMATH_CALUDE_remainder_problem_l1295_129507

theorem remainder_problem (N : ℤ) : N % 1927 = 131 → (3 * N) % 43 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1295_129507


namespace NUMINAMATH_CALUDE_light_ray_exits_l1295_129537

/-- Represents a line segment with a length -/
structure Segment where
  length : ℝ
  length_pos : length > 0

/-- Represents an angle formed by two segments with a common vertex -/
structure Angle where
  seg1 : Segment
  seg2 : Segment

/-- Represents a light ray traveling inside an angle -/
structure LightRay where
  angle : Angle
  position : ℝ × ℝ
  direction : ℝ × ℝ

/-- Predicate to check if a light ray has exited an angle -/
def has_exited (ray : LightRay) : Prop :=
  -- Implementation details omitted
  sorry

/-- Function to update the light ray's position and direction after a reflection -/
def reflect (ray : LightRay) : LightRay :=
  -- Implementation details omitted
  sorry

/-- Theorem stating that a light ray will eventually exit the angle -/
theorem light_ray_exits (angle : Angle) :
  ∃ (n : ℕ), ∀ (ray : LightRay), ray.angle = angle →
    has_exited (n.iterate reflect ray) :=
  sorry

end NUMINAMATH_CALUDE_light_ray_exits_l1295_129537


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l1295_129505

theorem square_plus_reciprocal_square (a : ℝ) (h : a + 1/a = 5) : a^2 + 1/a^2 = 23 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l1295_129505


namespace NUMINAMATH_CALUDE_max_sum_cubes_l1295_129551

theorem max_sum_cubes (a b c d e : ℝ) (h : a^2 + b^2 + c^2 + d^2 + e^2 = 5) :
  (∃ (x y z w v : ℝ), x^2 + y^2 + z^2 + w^2 + v^2 = 5 ∧ 
   x^3 + y^3 + z^3 + w^3 + v^3 ≥ a^3 + b^3 + c^3 + d^3 + e^3) ∧
  (∀ (x y z w v : ℝ), x^2 + y^2 + z^2 + w^2 + v^2 = 5 → 
   x^3 + y^3 + z^3 + w^3 + v^3 ≤ 5 * Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_max_sum_cubes_l1295_129551


namespace NUMINAMATH_CALUDE_sequence_relation_l1295_129584

theorem sequence_relation (a b : ℕ+ → ℚ) 
  (h1 : a 1 = 1/2)
  (h2 : ∀ n : ℕ+, a n + b n = 1)
  (h3 : ∀ n : ℕ+, b (n + 1) = b n / (1 - (a n)^2)) :
  ∀ n : ℕ+, b n = n / (n + 1) := by
sorry

end NUMINAMATH_CALUDE_sequence_relation_l1295_129584


namespace NUMINAMATH_CALUDE_function_is_linear_l1295_129511

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 - y^2) = (x - y) * (f x + f y)

/-- The main theorem stating that any function satisfying the equation is linear -/
theorem function_is_linear (f : ℝ → ℝ) (h : SatisfiesEquation f) :
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x := by
  sorry

end NUMINAMATH_CALUDE_function_is_linear_l1295_129511


namespace NUMINAMATH_CALUDE_one_common_point_condition_l1295_129520

/-- A function f(x) = mx² - 4x + 3 has only one common point with the x-axis if and only if m = 0 or m = 4/3 -/
theorem one_common_point_condition (m : ℝ) : 
  (∃! x, m * x^2 - 4 * x + 3 = 0) ↔ (m = 0 ∨ m = 4/3) := by
  sorry

end NUMINAMATH_CALUDE_one_common_point_condition_l1295_129520


namespace NUMINAMATH_CALUDE_rock_collection_difference_l1295_129529

theorem rock_collection_difference (joshua_rocks : ℕ) (jose_rocks : ℕ) (albert_rocks : ℕ)
  (joshua_80 : joshua_rocks = 80)
  (jose_fewer : jose_rocks < joshua_rocks)
  (albert_jose_diff : albert_rocks = jose_rocks + 20)
  (albert_joshua_diff : albert_rocks = joshua_rocks + 6) :
  joshua_rocks - jose_rocks = 14 := by
sorry

end NUMINAMATH_CALUDE_rock_collection_difference_l1295_129529


namespace NUMINAMATH_CALUDE_ball_returns_to_bella_l1295_129545

/-- Represents the number of girls in the circle -/
def n : ℕ := 13

/-- Represents the number of positions to move in each throw -/
def k : ℕ := 6

/-- Represents the position after a certain number of throws -/
def position (throws : ℕ) : ℕ :=
  (1 + throws * k) % n

/-- Theorem: The ball returns to Bella after exactly 13 throws -/
theorem ball_returns_to_bella :
  position 13 = 1 ∧ ∀ m : ℕ, m < 13 → position m ≠ 1 :=
sorry

end NUMINAMATH_CALUDE_ball_returns_to_bella_l1295_129545


namespace NUMINAMATH_CALUDE_parallel_vectors_m_l1295_129527

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

/-- Given two vectors a and b, where a = (m, 4) and b = (3, -2),
    if a is parallel to b, then m = -6 -/
theorem parallel_vectors_m (m : ℝ) :
  let a : ℝ × ℝ := (m, 4)
  let b : ℝ × ℝ := (3, -2)
  parallel a b → m = -6 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_l1295_129527


namespace NUMINAMATH_CALUDE_arrangement_exists_l1295_129561

theorem arrangement_exists : ∃ (p : Fin 100 → Fin 100), Function.Bijective p ∧ 
  ∀ i : Fin 99, 
    (((p (i + 1)).val = (p i).val + 2) ∨ ((p (i + 1)).val = (p i).val - 2)) ∨
    ((p (i + 1)).val = 2 * (p i).val) ∨ ((p i).val = 2 * (p (i + 1)).val) := by
  sorry

end NUMINAMATH_CALUDE_arrangement_exists_l1295_129561


namespace NUMINAMATH_CALUDE_sin_power_five_expansion_l1295_129556

theorem sin_power_five_expansion (b₁ b₂ b₃ b₄ b₅ : ℝ) : 
  (∀ θ : ℝ, Real.sin θ ^ 5 = b₁ * Real.sin θ + b₂ * Real.sin (2 * θ) + 
    b₃ * Real.sin (3 * θ) + b₄ * Real.sin (4 * θ) + b₅ * Real.sin (5 * θ)) → 
  b₁ ^ 2 + b₂ ^ 2 + b₃ ^ 2 + b₄ ^ 2 + b₅ ^ 2 = 63 / 128 := by
  sorry

end NUMINAMATH_CALUDE_sin_power_five_expansion_l1295_129556


namespace NUMINAMATH_CALUDE_max_sum_of_digits_24hour_clock_l1295_129531

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Nat
  minutes : Nat
  hour_valid : hours < 24
  minute_valid : minutes < 60

/-- Calculates the sum of digits for a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  let digits := n.repr.toList.map (fun c => c.toString.toNat!)
  digits.sum

/-- Calculates the sum of digits for a Time24 -/
def timeSumOfDigits (t : Time24) : Nat :=
  sumOfDigits t.hours + sumOfDigits t.minutes

/-- The largest possible sum of digits in a 24-hour digital clock display is 24 -/
theorem max_sum_of_digits_24hour_clock :
  (∀ t : Time24, timeSumOfDigits t ≤ 24) ∧
  (∃ t : Time24, timeSumOfDigits t = 24) :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_digits_24hour_clock_l1295_129531


namespace NUMINAMATH_CALUDE_cos_sin_10_deg_equality_l1295_129522

theorem cos_sin_10_deg_equality : 
  4 * Real.cos (10 * π / 180) - Real.cos (10 * π / 180) / Real.sin (10 * π / 180) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_10_deg_equality_l1295_129522


namespace NUMINAMATH_CALUDE_darry_smaller_ladder_climbs_l1295_129555

/-- Represents the number of steps in Darry's full ladder -/
def full_ladder_steps : ℕ := 11

/-- Represents the number of steps in Darry's smaller ladder -/
def smaller_ladder_steps : ℕ := 6

/-- Represents the number of times Darry climbed the full ladder -/
def full_ladder_climbs : ℕ := 10

/-- Represents the total number of steps Darry climbed -/
def total_steps : ℕ := 152

/-- Theorem stating that Darry climbed the smaller ladder 7 times -/
theorem darry_smaller_ladder_climbs :
  ∃ (x : ℕ), x * smaller_ladder_steps + full_ladder_climbs * full_ladder_steps = total_steps ∧ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_darry_smaller_ladder_climbs_l1295_129555


namespace NUMINAMATH_CALUDE_incorrect_vs_correct_operations_l1295_129581

theorem incorrect_vs_correct_operations (x : ℝ) :
  (x / 8 - 12 = 18) → (x * 8 * 12 = 23040) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_vs_correct_operations_l1295_129581


namespace NUMINAMATH_CALUDE_points_earned_is_thirteen_l1295_129536

/-- VideoGame represents the state of the game --/
structure VideoGame where
  totalEnemies : Nat
  redEnemies : Nat
  blueEnemies : Nat
  defeatedEnemies : Nat
  hits : Nat
  pointsPerEnemy : Nat
  bonusPoints : Nat
  pointsLostPerHit : Nat

/-- Calculate the total points earned in the game --/
def calculatePoints (game : VideoGame) : Int :=
  let basePoints := game.defeatedEnemies * game.pointsPerEnemy
  let bonusEarned := if (game.redEnemies - 1 > 0) && (game.blueEnemies - 1 > 0) then game.bonusPoints else 0
  let totalEarned := basePoints + bonusEarned
  let pointsLost := game.hits * game.pointsLostPerHit
  totalEarned - pointsLost

/-- Theorem stating that given the game conditions, the total points earned is 13 --/
theorem points_earned_is_thirteen :
  ∀ (game : VideoGame),
    game.totalEnemies = 6 →
    game.redEnemies = 3 →
    game.blueEnemies = 3 →
    game.defeatedEnemies = 4 →
    game.hits = 2 →
    game.pointsPerEnemy = 3 →
    game.bonusPoints = 5 →
    game.pointsLostPerHit = 2 →
    calculatePoints game = 13 := by
  sorry

end NUMINAMATH_CALUDE_points_earned_is_thirteen_l1295_129536


namespace NUMINAMATH_CALUDE_at_least_one_less_than_one_negation_all_not_less_than_one_l1295_129538

theorem at_least_one_less_than_one (a b c : ℝ) (ha : a < 3) (hb : b < 3) (hc : c < 3) :
  a < 1 ∨ b < 1 ∨ c < 1 := by
  sorry

theorem negation_all_not_less_than_one (a b c : ℝ) :
  (¬(a < 1 ∨ b < 1 ∨ c < 1)) ↔ (a ≥ 1 ∧ b ≥ 1 ∧ c ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_less_than_one_negation_all_not_less_than_one_l1295_129538


namespace NUMINAMATH_CALUDE_book_pages_difference_l1295_129577

theorem book_pages_difference : 
  let purple_books : ℕ := 8
  let orange_books : ℕ := 7
  let blue_books : ℕ := 5
  let purple_pages_per_book : ℕ := 320
  let orange_pages_per_book : ℕ := 640
  let blue_pages_per_book : ℕ := 450
  let total_purple_pages := purple_books * purple_pages_per_book
  let total_orange_pages := orange_books * orange_pages_per_book
  let total_blue_pages := blue_books * blue_pages_per_book
  let total_orange_blue_pages := total_orange_pages + total_blue_pages
  total_orange_blue_pages - total_purple_pages = 4170 := by
sorry

end NUMINAMATH_CALUDE_book_pages_difference_l1295_129577


namespace NUMINAMATH_CALUDE_binomial_expansion_alternating_sum_l1295_129585

theorem binomial_expansion_alternating_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2 + x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ - a₀ + a₃ - a₂ + a₅ - a₄ = -1 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_alternating_sum_l1295_129585


namespace NUMINAMATH_CALUDE_gcd_p4_minus_1_l1295_129517

theorem gcd_p4_minus_1 (p : ℕ) (hp : p.Prime) (hp_gt_5 : p > 5) : 
  ∃ k : ℕ, p^4 - 1 = 240 * k := by
sorry

end NUMINAMATH_CALUDE_gcd_p4_minus_1_l1295_129517


namespace NUMINAMATH_CALUDE_inequality_range_l1295_129526

theorem inequality_range (a : ℝ) (h : 0 ≤ a ∧ a ≤ 4) :
  ∀ x : ℝ, x^2 + a*x > 4*x + a - 3 ↔ x < -1 ∨ x > 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_range_l1295_129526


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1295_129576

theorem quadratic_equation_solution : 
  ∃ (x₁ x₂ : ℝ), x₁ = 1 ∧ x₂ = 1/3 ∧ 
  (∀ x : ℝ, 3*x^2 - 4*x + 1 = 0 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1295_129576


namespace NUMINAMATH_CALUDE_area_eq_product_segments_l1295_129594

/-- A right triangle with an inscribed circle -/
structure RightTriangleWithIncircle where
  /-- The length of one leg of the right triangle -/
  a : ℝ
  /-- The length of the other leg of the right triangle -/
  b : ℝ
  /-- The length of the hypotenuse of the right triangle -/
  c : ℝ
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The length of one segment of the hypotenuse divided by the point of tangency -/
  m : ℝ
  /-- The length of the other segment of the hypotenuse divided by the point of tangency -/
  n : ℝ
  /-- The hypotenuse is the sum of its segments -/
  hyp_sum : c = m + n
  /-- The triangle satisfies the Pythagorean theorem -/
  pythagorean : a^2 + b^2 = c^2
  /-- All lengths are positive -/
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0
  pos_r : r > 0
  pos_m : m > 0
  pos_n : n > 0

/-- The area of a right triangle with an inscribed circle is equal to the product of the 
    lengths of the segments into which the hypotenuse is divided by the point of tangency 
    with the incircle -/
theorem area_eq_product_segments (t : RightTriangleWithIncircle) : 
  (1/2) * t.a * t.b = t.m * t.n := by
  sorry

end NUMINAMATH_CALUDE_area_eq_product_segments_l1295_129594


namespace NUMINAMATH_CALUDE_paint_coverage_l1295_129542

/-- Proves that a quart of paint covers 60 square feet given the specified conditions -/
theorem paint_coverage (cube_edge : Real) (paint_cost_per_quart : Real) (total_paint_cost : Real)
  (h1 : cube_edge = 10)
  (h2 : paint_cost_per_quart = 3.2)
  (h3 : total_paint_cost = 32) :
  (6 * cube_edge^2) / (total_paint_cost / paint_cost_per_quart) = 60 := by
  sorry

end NUMINAMATH_CALUDE_paint_coverage_l1295_129542


namespace NUMINAMATH_CALUDE_calculate_interest_rate_l1295_129523

/-- Given a car price, total amount to pay, and loan amount, calculate the interest rate -/
theorem calculate_interest_rate 
  (car_price : ℝ) 
  (total_amount : ℝ) 
  (loan_amount : ℝ) 
  (h1 : car_price = 35000)
  (h2 : total_amount = 38000)
  (h3 : loan_amount = 20000) :
  (total_amount - loan_amount) / loan_amount * 100 = 90 := by
  sorry

#check calculate_interest_rate

end NUMINAMATH_CALUDE_calculate_interest_rate_l1295_129523


namespace NUMINAMATH_CALUDE_wilson_gained_money_l1295_129558

def watch_problem (selling_price : ℝ) (profit_percentage : ℝ) (loss_percentage : ℝ) : Prop :=
  let cost_price1 := selling_price / (1 + profit_percentage / 100)
  let cost_price2 := selling_price / (1 - loss_percentage / 100)
  let total_cost := cost_price1 + cost_price2
  let total_revenue := 2 * selling_price
  total_revenue > total_cost

theorem wilson_gained_money : watch_problem 150 25 15 := by
  sorry

end NUMINAMATH_CALUDE_wilson_gained_money_l1295_129558


namespace NUMINAMATH_CALUDE_sqrt_x_minus_6_meaningful_l1295_129592

theorem sqrt_x_minus_6_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 6) ↔ x ≥ 6 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_6_meaningful_l1295_129592


namespace NUMINAMATH_CALUDE_line_intersection_l1295_129549

theorem line_intersection :
  ∃! p : ℚ × ℚ, 
    (3 * p.2 = -2 * p.1 + 6) ∧ 
    (-2 * p.2 = 6 * p.1 + 4) ∧ 
    p = (-12/7, 22/7) := by
  sorry

end NUMINAMATH_CALUDE_line_intersection_l1295_129549


namespace NUMINAMATH_CALUDE_triangle_theorem_l1295_129533

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  pos_sides : 0 < a ∧ 0 < b ∧ 0 < c
  pos_angles : 0 < A ∧ 0 < B ∧ 0 < C
  angle_sum : A + B + C = π

theorem triangle_theorem (t : Triangle) 
  (h : t.b * Real.cos t.A + Real.sqrt 3 * t.b * Real.sin t.A - t.c - t.a = 0) :
  t.B = π / 3 ∧ 
  (t.b = Real.sqrt 3 → ∀ (a c : ℝ), a + c ≤ 2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_theorem_l1295_129533


namespace NUMINAMATH_CALUDE_log_2_base_10_bound_l1295_129552

theorem log_2_base_10_bound (h1 : 10^3 = 1000) (h2 : 10^5 = 100000)
  (h3 : 2^12 = 4096) (h4 : 2^15 = 32768) (h5 : 2^17 = 131072) :
  5/17 < Real.log 2 / Real.log 10 := by
  sorry

end NUMINAMATH_CALUDE_log_2_base_10_bound_l1295_129552


namespace NUMINAMATH_CALUDE_initial_milk_water_ratio_l1295_129568

theorem initial_milk_water_ratio 
  (M W : ℝ) 
  (h1 : M + W = 45) 
  (h2 : M / (W + 18) = 4/3) : 
  M / W = 4 := by
sorry

end NUMINAMATH_CALUDE_initial_milk_water_ratio_l1295_129568


namespace NUMINAMATH_CALUDE_pure_imaginary_m_l1295_129563

/-- A complex number z is pure imaginary if and only if its real part is zero and its imaginary part is non-zero. -/
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- The complex number z as a function of m. -/
def z (m : ℝ) : ℂ := Complex.mk (m^2 + m - 2) (m^2 + 4*m - 5)

theorem pure_imaginary_m : ∃! m : ℝ, is_pure_imaginary (z m) ∧ m = -2 := by sorry

end NUMINAMATH_CALUDE_pure_imaginary_m_l1295_129563


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1295_129593

theorem sufficient_but_not_necessary (x : ℝ) : 
  (∀ x, 2 < x ∧ x < 4 → Real.log x < Real.exp 1) ∧
  (∃ x, Real.log x < Real.exp 1 ∧ ¬(2 < x ∧ x < 4)) := by
sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1295_129593


namespace NUMINAMATH_CALUDE_sum_evaluation_l1295_129557

theorem sum_evaluation : 
  4/3 + 8/9 + 16/27 + 32/81 + 64/243 + 128/729 - 8 = -1/729 := by sorry

end NUMINAMATH_CALUDE_sum_evaluation_l1295_129557


namespace NUMINAMATH_CALUDE_complex_fraction_value_l1295_129548

theorem complex_fraction_value : 
  let i : ℂ := Complex.I
  (3 + i) / (1 - i) = 1 + 2*i := by sorry

end NUMINAMATH_CALUDE_complex_fraction_value_l1295_129548


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1295_129571

theorem rationalize_denominator :
  (Real.sqrt 12 + Real.sqrt 5) / (Real.sqrt 6 + Real.sqrt 5) =
  6 * Real.sqrt 2 - 2 * Real.sqrt 15 + Real.sqrt 30 - 5 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1295_129571


namespace NUMINAMATH_CALUDE_go_pieces_theorem_l1295_129587

/-- Represents the set of Go pieces -/
structure GoPieces where
  white : ℕ
  black : ℕ

/-- Calculates the probability of drawing two pieces of the same color in the first two draws -/
def prob_same_color (pieces : GoPieces) : ℚ :=
  sorry

/-- Calculates the expected value of the number of white Go pieces drawn in the first four draws -/
def expected_white_pieces (pieces : GoPieces) : ℚ :=
  sorry

theorem go_pieces_theorem (pieces : GoPieces) 
  (h1 : pieces.white = 4) 
  (h2 : pieces.black = 3) : 
  prob_same_color pieces = 3/7 ∧ expected_white_pieces pieces = 16/7 := by
  sorry

end NUMINAMATH_CALUDE_go_pieces_theorem_l1295_129587


namespace NUMINAMATH_CALUDE_prime_exists_not_dividing_power_minus_prime_l1295_129578

theorem prime_exists_not_dividing_power_minus_prime (p : ℕ) (hp : Prime p) :
  ∃ q : ℕ, Prime q ∧ ∀ n : ℕ, n > 0 → ¬(q ∣ n^p - p) := by
  sorry

end NUMINAMATH_CALUDE_prime_exists_not_dividing_power_minus_prime_l1295_129578


namespace NUMINAMATH_CALUDE_expand_product_l1295_129586

theorem expand_product (x : ℝ) : 3 * (x - 2) * (x^2 + x + 1) = 3*x^3 - 3*x^2 - 3*x - 6 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l1295_129586


namespace NUMINAMATH_CALUDE_polynomial_division_quotient_remainder_l1295_129597

theorem polynomial_division_quotient_remainder 
  (x : ℝ) (h : x ≠ 1) : 
  ∃ (q r : ℝ), 
    x^5 + 5 = (x - 1) * q + r ∧ 
    q = x^4 + x^3 + x^2 + x + 1 ∧ 
    r = 6 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_quotient_remainder_l1295_129597


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l1295_129525

theorem ratio_x_to_y (x y : ℚ) (h : (12 * x - 5 * y) / (16 * x - 3 * y) = 4 / 7) :
  x / y = 23 / 20 := by
  sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l1295_129525


namespace NUMINAMATH_CALUDE_nonzero_digits_count_l1295_129532

-- Define the fraction
def f : ℚ := 84 / (2^5 * 5^9)

-- Define a function to count non-zero digits after the decimal point
noncomputable def count_nonzero_digits_after_decimal (q : ℚ) : ℕ := sorry

-- Theorem statement
theorem nonzero_digits_count :
  count_nonzero_digits_after_decimal f = 2 := by sorry

end NUMINAMATH_CALUDE_nonzero_digits_count_l1295_129532


namespace NUMINAMATH_CALUDE_rice_bag_weight_l1295_129565

theorem rice_bag_weight 
  (rice_bags : ℕ) 
  (flour_bags : ℕ) 
  (total_weight : ℕ) 
  (h1 : rice_bags = 20)
  (h2 : flour_bags = 50)
  (h3 : total_weight = 2250)
  (h4 : ∃ (x : ℕ), x * rice_bags + (x / 2) * flour_bags = total_weight) :
  ∃ (rice_weight : ℕ), rice_weight = 50 ∧ 
    rice_weight * rice_bags + (rice_weight / 2) * flour_bags = total_weight :=
by sorry

end NUMINAMATH_CALUDE_rice_bag_weight_l1295_129565


namespace NUMINAMATH_CALUDE_existence_of_divisible_power_sum_l1295_129535

theorem existence_of_divisible_power_sum (p : Nat) (h_prime : Prime p) (h_p_gt_10 : p > 10) :
  ∃ m n : Nat, m > 0 ∧ n > 0 ∧ m + n < p ∧ (p ∣ (5^m * 7^n - 1)) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_divisible_power_sum_l1295_129535


namespace NUMINAMATH_CALUDE_parabola_equation_correct_l1295_129599

/-- Represents a parabola with equation ax^2 + bx + c --/
structure Parabola where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Checks if a parabola has a vertical axis of symmetry --/
def has_vertical_axis_of_symmetry (p : Parabola) : Prop :=
  p.a ≠ 0

/-- Computes the vertex of a parabola --/
def vertex (p : Parabola) : ℚ × ℚ :=
  (- p.b / (2 * p.a), - (p.b^2 - 4*p.a*p.c) / (4 * p.a))

/-- Checks if a point lies on the parabola --/
def point_on_parabola (p : Parabola) (x y : ℚ) : Prop :=
  y = p.a * x^2 + p.b * x + p.c

/-- The main theorem --/
theorem parabola_equation_correct :
  let p : Parabola := { a := 2/9, b := -4/3, c := 0 }
  has_vertical_axis_of_symmetry p ∧
  vertex p = (3, -2) ∧
  point_on_parabola p 6 0 := by sorry

end NUMINAMATH_CALUDE_parabola_equation_correct_l1295_129599


namespace NUMINAMATH_CALUDE_probability_of_red_ball_l1295_129539

def num_white_balls : ℕ := 4
def num_red_balls : ℕ := 2

theorem probability_of_red_ball :
  let total_balls := num_white_balls + num_red_balls
  (num_red_balls : ℚ) / total_balls = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_red_ball_l1295_129539


namespace NUMINAMATH_CALUDE_expand_expression_l1295_129580

theorem expand_expression (x : ℝ) : (5 * x^2 - 3) * 4 * x^3 = 20 * x^5 - 12 * x^3 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1295_129580


namespace NUMINAMATH_CALUDE_farm_cows_l1295_129503

theorem farm_cows (milk_per_week : ℝ) (total_milk : ℝ) (num_weeks : ℕ) :
  milk_per_week = 108 →
  total_milk = 2160 →
  num_weeks = 5 →
  (total_milk / (milk_per_week / 6 * num_weeks) : ℝ) = 24 :=
by sorry

end NUMINAMATH_CALUDE_farm_cows_l1295_129503


namespace NUMINAMATH_CALUDE_circles_intersect_l1295_129501

theorem circles_intersect (r R d : ℝ) (hr : r = 4) (hR : R = 5) (hd : d = 6) :
  let sum := r + R
  let diff := R - r
  d > diff ∧ d < sum := by sorry

end NUMINAMATH_CALUDE_circles_intersect_l1295_129501


namespace NUMINAMATH_CALUDE_rational_cubic_polynomial_existence_l1295_129502

theorem rational_cubic_polynomial_existence :
  ∃ (b c d : ℚ), 
    let P := fun (x : ℚ) => x^3 + b*x^2 + c*x + d
    let P' := fun (x : ℚ) => 3*x^2 + 2*b*x + c
    ∃ (r₁ r₂ r₃ : ℚ), r₁ ≠ r₂ ∧ r₂ ≠ r₃ ∧ r₃ ≠ r₁ ∧
    P r₁ = 0 ∧ P r₂ = 0 ∧ P r₃ = 0 ∧
    ∃ (c₁ c₂ : ℚ), P' c₁ = 0 ∧ P' c₂ = 0 := by
  sorry

end NUMINAMATH_CALUDE_rational_cubic_polynomial_existence_l1295_129502


namespace NUMINAMATH_CALUDE_smart_mart_puzzles_l1295_129506

/-- The number of science kits sold by Smart Mart last week -/
def science_kits : ℕ := 45

/-- The difference between science kits and puzzles sold -/
def difference : ℕ := 9

/-- The number of puzzles sold by Smart Mart last week -/
def puzzles : ℕ := science_kits - difference

/-- Theorem stating that the number of puzzles sold is 36 -/
theorem smart_mart_puzzles : puzzles = 36 := by
  sorry

end NUMINAMATH_CALUDE_smart_mart_puzzles_l1295_129506


namespace NUMINAMATH_CALUDE_exponent_zero_equals_one_f_equals_S_l1295_129543

-- Option C
theorem exponent_zero_equals_one (x : ℝ) (h : x ≠ 0) : x^0 = 1 := by sorry

-- Option D
def f (x : ℝ) : ℝ := x^2
def S (t : ℝ) : ℝ := t^2

theorem f_equals_S : f = S := by sorry

end NUMINAMATH_CALUDE_exponent_zero_equals_one_f_equals_S_l1295_129543


namespace NUMINAMATH_CALUDE_shaded_area_is_fifty_l1295_129560

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square with side length and partitioning points -/
structure PartitionedSquare where
  sideLength : ℝ
  pointA : Point
  pointB : Point

/-- Calculates the area of the shaded diamond region in the partitioned square -/
def shadedAreaInPartitionedSquare (square : PartitionedSquare) : ℝ :=
  sorry

/-- The theorem stating that the shaded area in the given partitioned square is 50 square cm -/
theorem shaded_area_is_fifty (square : PartitionedSquare) 
  (h1 : square.sideLength = 10)
  (h2 : square.pointA = ⟨10/3, 10⟩)
  (h3 : square.pointB = ⟨20/3, 0⟩) : 
  shadedAreaInPartitionedSquare square = 50 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_fifty_l1295_129560


namespace NUMINAMATH_CALUDE_parking_lot_perimeter_l1295_129544

theorem parking_lot_perimeter 
  (d : ℝ) (A : ℝ) (x y : ℝ) (P : ℝ) 
  (h1 : d = 20) 
  (h2 : A = 120) 
  (h3 : x = (2/3) * y) 
  (h4 : x^2 + y^2 = d^2) 
  (h5 : x * y = A) 
  (h6 : P = 2 * (x + y)) : 
  P = 20 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_perimeter_l1295_129544


namespace NUMINAMATH_CALUDE_student_selection_problem_l1295_129566

def student_selection (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

theorem student_selection_problem (total_students : ℕ) (selected_students : ℕ) 
  (h_total : total_students = 10) (h_selected : selected_students = 4) :
  (student_selection (total_students - 3) (selected_students - 2) + 
   student_selection (total_students - 3) (selected_students - 1)) = 84 := by
  sorry

end NUMINAMATH_CALUDE_student_selection_problem_l1295_129566


namespace NUMINAMATH_CALUDE_square_roots_problem_l1295_129510

theorem square_roots_problem (a : ℝ) (n : ℝ) : 
  (2*a + 3)^2 = n ∧ (a - 18)^2 = n → n = 169 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_problem_l1295_129510


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1295_129547

theorem quadratic_inequality_solution (x : ℝ) : x^2 + 7*x < 8 ↔ -8 < x ∧ x < 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1295_129547


namespace NUMINAMATH_CALUDE_keith_total_spent_l1295_129518

def speakers_cost : ℚ := 136.01
def cd_player_cost : ℚ := 139.38
def tires_cost : ℚ := 112.46

theorem keith_total_spent : 
  speakers_cost + cd_player_cost + tires_cost = 387.85 := by
  sorry

end NUMINAMATH_CALUDE_keith_total_spent_l1295_129518


namespace NUMINAMATH_CALUDE_smallest_negative_quadratic_l1295_129550

theorem smallest_negative_quadratic (n : ℤ) : 
  (∀ m : ℤ, m < n → 4 * m^2 - 28 * m + 48 ≥ 0) ∧ 
  (4 * n^2 - 28 * n + 48 < 0) → 
  n = 4 := by
sorry

end NUMINAMATH_CALUDE_smallest_negative_quadratic_l1295_129550


namespace NUMINAMATH_CALUDE_basketball_handshakes_l1295_129512

theorem basketball_handshakes :
  let team_size : ℕ := 6
  let num_teams : ℕ := 2
  let num_referees : ℕ := 3
  let player_handshakes := team_size * team_size
  let referee_handshakes := (team_size * num_teams) * num_referees
  player_handshakes + referee_handshakes = 72 := by
sorry

end NUMINAMATH_CALUDE_basketball_handshakes_l1295_129512


namespace NUMINAMATH_CALUDE_principal_amount_proof_l1295_129562

/-- Prove that for a given principal amount, interest rate, and time period,
    if the difference between compound and simple interest is 20,
    then the principal amount is 8000. -/
theorem principal_amount_proof (P : ℝ) :
  let r : ℝ := 0.05  -- 5% annual interest rate
  let t : ℝ := 2     -- 2 years time period
  let compound_interest := P * (1 + r) ^ t - P
  let simple_interest := P * r * t
  compound_interest - simple_interest = 20 →
  P = 8000 := by
sorry

end NUMINAMATH_CALUDE_principal_amount_proof_l1295_129562


namespace NUMINAMATH_CALUDE_regular_adult_ticket_price_correct_l1295_129508

/-- The regular price of an adult movie ticket given the following conditions:
  * There are 5 adults and 2 children.
  * Children's concessions cost $3 each.
  * Adults' concessions cost $5, $6, $7, $4, and $9 respectively.
  * Total cost of the trip is $139.
  * Each child's ticket costs $7.
  * Three adults have discounts of $3, $2, and $1 on their tickets.
-/
def regular_adult_ticket_price : ℚ :=
  let num_adults : ℕ := 5
  let num_children : ℕ := 2
  let child_concession_cost : ℚ := 3
  let adult_concession_costs : List ℚ := [5, 6, 7, 4, 9]
  let total_trip_cost : ℚ := 139
  let child_ticket_cost : ℚ := 7
  let adult_ticket_discounts : List ℚ := [3, 2, 1]
  18.8

theorem regular_adult_ticket_price_correct :
  let num_adults : ℕ := 5
  let num_children : ℕ := 2
  let child_concession_cost : ℚ := 3
  let adult_concession_costs : List ℚ := [5, 6, 7, 4, 9]
  let total_trip_cost : ℚ := 139
  let child_ticket_cost : ℚ := 7
  let adult_ticket_discounts : List ℚ := [3, 2, 1]
  regular_adult_ticket_price = 18.8 := by
  sorry

#eval regular_adult_ticket_price

end NUMINAMATH_CALUDE_regular_adult_ticket_price_correct_l1295_129508


namespace NUMINAMATH_CALUDE_inequality_proof_l1295_129521

theorem inequality_proof (a b c d : ℝ) 
  (h1 : 0 < a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) 
  (h5 : a + b + c + d = 1) : 
  a^2 + 3*b^2 + 5*c^2 + 7*d^2 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1295_129521


namespace NUMINAMATH_CALUDE_one_integer_solution_l1295_129588

def circle_center : ℝ × ℝ := (4, 6)
def circle_radius : ℝ := 8

def point (x : ℤ) : ℝ × ℝ := (2 * x, -x)

def inside_or_on_circle (p : ℝ × ℝ) : Prop :=
  (p.1 - circle_center.1)^2 + (p.2 - circle_center.2)^2 ≤ circle_radius^2

theorem one_integer_solution : 
  ∃! x : ℤ, inside_or_on_circle (point x) :=
sorry

end NUMINAMATH_CALUDE_one_integer_solution_l1295_129588


namespace NUMINAMATH_CALUDE_problem_statement_l1295_129583

theorem problem_statement (a b : ℝ) (h1 : a + b = 1) (h2 : a * b = -6) :
  a^3 * b - 2 * a^2 * b^2 + a * b^3 = -150 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1295_129583


namespace NUMINAMATH_CALUDE_tree_planting_theorem_l1295_129514

/-- The number of trees planted by 4th graders -/
def trees_4th : ℕ := 30

/-- The number of trees planted by 5th graders -/
def trees_5th : ℕ := 2 * trees_4th

/-- The number of trees planted by 6th graders -/
def trees_6th : ℕ := 3 * trees_5th - 30

/-- The total number of trees planted by all grades -/
def total_trees : ℕ := trees_4th + trees_5th + trees_6th

theorem tree_planting_theorem : total_trees = 240 := by
  sorry

end NUMINAMATH_CALUDE_tree_planting_theorem_l1295_129514


namespace NUMINAMATH_CALUDE_line_xz_plane_intersection_l1295_129519

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space defined by two points -/
structure Line3D where
  p1 : Point3D
  p2 : Point3D

/-- The xz-plane -/
def xzPlane : Set Point3D := {p : Point3D | p.y = 0}

/-- Function to check if a point lies on a line -/
def pointOnLine (p : Point3D) (l : Line3D) : Prop :=
  ∃ t : ℝ, p.x = l.p1.x + t * (l.p2.x - l.p1.x) ∧
            p.y = l.p1.y + t * (l.p2.y - l.p1.y) ∧
            p.z = l.p1.z + t * (l.p2.z - l.p1.z)

theorem line_xz_plane_intersection :
  let l : Line3D := {
    p1 := { x := 2, y := -1, z := 3 },
    p2 := { x := 6, y := -4, z := 7 }
  }
  let intersectionPoint : Point3D := { x := 2/3, y := 0, z := 5/3 }
  (intersectionPoint ∈ xzPlane) ∧ 
  (pointOnLine intersectionPoint l) := by sorry

end NUMINAMATH_CALUDE_line_xz_plane_intersection_l1295_129519


namespace NUMINAMATH_CALUDE_cubic_function_range_l1295_129534

/-- Given a cubic function f(x) = ax³ + bx where a and b are real constants,
    if f(2) = 2 and f'(2) = 9, then the range of f(x) for x ∈ ℝ is [-2, 18]. -/
theorem cubic_function_range (a b : ℝ) :
  (∀ x, f x = a * x^3 + b * x) →
  f 2 = 2 →
  (∀ x, deriv f x = 3 * a * x^2 + b) →
  deriv f 2 = 9 →
  ∀ y ∈ Set.range f, -2 ≤ y ∧ y ≤ 18 :=
by sorry


end NUMINAMATH_CALUDE_cubic_function_range_l1295_129534


namespace NUMINAMATH_CALUDE_currency_conversion_weight_conversion_gram_to_kg_weight_conversion_kg_to_ton_length_conversion_l1295_129590

-- Define conversion rates
def yuan_to_jiao : ℚ := 10
def yuan_to_fen : ℚ := 100
def kg_to_gram : ℚ := 1000
def ton_to_kg : ℚ := 1000
def meter_to_cm : ℚ := 100

-- Define the conversion functions
def jiao_to_yuan (j : ℚ) : ℚ := j / yuan_to_jiao
def fen_to_yuan (f : ℚ) : ℚ := f / yuan_to_fen
def gram_to_kg (g : ℚ) : ℚ := g / kg_to_gram
def kg_to_ton (k : ℚ) : ℚ := k / ton_to_kg
def cm_to_meter (c : ℚ) : ℚ := c / meter_to_cm

-- Theorem statements
theorem currency_conversion :
  5 + jiao_to_yuan 4 + fen_to_yuan 8 = 5.48 := by sorry

theorem weight_conversion_gram_to_kg :
  gram_to_kg 80 = 0.08 := by sorry

theorem weight_conversion_kg_to_ton :
  kg_to_ton 73 = 0.073 := by sorry

theorem length_conversion :
  1 + cm_to_meter 5 = 1.05 := by sorry

end NUMINAMATH_CALUDE_currency_conversion_weight_conversion_gram_to_kg_weight_conversion_kg_to_ton_length_conversion_l1295_129590


namespace NUMINAMATH_CALUDE_coin_collection_dime_difference_l1295_129569

theorem coin_collection_dime_difference :
  ∀ (nickels dimes quarters : ℕ),
  nickels + dimes + quarters = 120 →
  5 * nickels + 10 * dimes + 25 * quarters = 1265 →
  quarters ≥ 10 →
  ∃ (min_dimes max_dimes : ℕ),
    (∀ d : ℕ, 
      nickels + d + quarters = 120 ∧ 
      5 * nickels + 10 * d + 25 * quarters = 1265 →
      min_dimes ≤ d ∧ d ≤ max_dimes) ∧
    max_dimes - min_dimes = 92 :=
by sorry

end NUMINAMATH_CALUDE_coin_collection_dime_difference_l1295_129569


namespace NUMINAMATH_CALUDE_octahedron_projection_area_l1295_129530

/-- A regular octahedron -/
structure RegularOctahedron where
  -- Add necessary fields here

/-- The area of a face of a regular octahedron -/
def face_area (o : RegularOctahedron) : ℝ :=
  sorry

/-- The area of the projection of one face onto the opposite face -/
def projection_area (o : RegularOctahedron) : ℝ :=
  sorry

/-- 
  In a regular octahedron, the perpendicular projection of one face 
  onto the plane of the opposite face covers 2/3 of the area of the opposite face
-/
theorem octahedron_projection_area (o : RegularOctahedron) :
  projection_area o = (2 / 3) * face_area o :=
sorry

end NUMINAMATH_CALUDE_octahedron_projection_area_l1295_129530


namespace NUMINAMATH_CALUDE_M_is_range_of_f_l1295_129589

-- Define the set M
def M : Set ℝ := {y | ∃ x, y = x^2}

-- Define the function f
def f : ℝ → ℝ := fun x ↦ x^2

-- Theorem statement
theorem M_is_range_of_f : M = Set.range f := by sorry

end NUMINAMATH_CALUDE_M_is_range_of_f_l1295_129589


namespace NUMINAMATH_CALUDE_shelter_dogs_count_l1295_129564

theorem shelter_dogs_count :
  ∀ (dogs cats : ℕ),
  (dogs : ℚ) / cats = 15 / 7 →
  dogs / (cats + 20) = 15 / 11 →
  dogs = 75 :=
by
  sorry

end NUMINAMATH_CALUDE_shelter_dogs_count_l1295_129564


namespace NUMINAMATH_CALUDE_eunji_remaining_confetti_l1295_129515

def initial_green_confetti : ℕ := 9
def initial_red_confetti : ℕ := 1
def confetti_given_away : ℕ := 4

theorem eunji_remaining_confetti :
  initial_green_confetti + initial_red_confetti - confetti_given_away = 6 := by
  sorry

end NUMINAMATH_CALUDE_eunji_remaining_confetti_l1295_129515


namespace NUMINAMATH_CALUDE_watch_cost_price_l1295_129574

theorem watch_cost_price (loss_percent : ℚ) (gain_percent : ℚ) (price_difference : ℚ) :
  loss_percent = 16 →
  gain_percent = 4 →
  price_difference = 140 →
  ∃ (cost_price : ℚ),
    (cost_price * (1 - loss_percent / 100)) + price_difference = cost_price * (1 + gain_percent / 100) ∧
    cost_price = 700 :=
by sorry

end NUMINAMATH_CALUDE_watch_cost_price_l1295_129574


namespace NUMINAMATH_CALUDE_min_value_2x_plus_y_l1295_129579

theorem min_value_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y - 2*x*y = 0) :
  ∀ z, z = 2*x + y → z ≥ 9/2 :=
sorry

end NUMINAMATH_CALUDE_min_value_2x_plus_y_l1295_129579


namespace NUMINAMATH_CALUDE_phone_call_duration_l1295_129591

/-- Calculates the duration of a phone call given initial credit, cost per minute, and remaining credit -/
def call_duration (initial_credit : ℚ) (cost_per_minute : ℚ) (remaining_credit : ℚ) : ℚ :=
  (initial_credit - remaining_credit) / cost_per_minute

/-- Proves that given the specified conditions, the call duration is 22 minutes -/
theorem phone_call_duration :
  let initial_credit : ℚ := 30
  let cost_per_minute : ℚ := 16/100
  let remaining_credit : ℚ := 2648/100
  call_duration initial_credit cost_per_minute remaining_credit = 22 := by
sorry


end NUMINAMATH_CALUDE_phone_call_duration_l1295_129591


namespace NUMINAMATH_CALUDE_stream_speed_l1295_129572

/-- The speed of a stream given a swimmer's still water speed and relative upstream/downstream times -/
theorem stream_speed (still_speed : ℝ) (upstream_time_factor : ℝ) : 
  still_speed = 4.5 ∧ upstream_time_factor = 2 → 
  ∃ stream_speed : ℝ, stream_speed = 1.5 ∧
    upstream_time_factor * (still_speed + stream_speed) = still_speed - stream_speed :=
by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l1295_129572


namespace NUMINAMATH_CALUDE_f_properties_l1295_129570

-- Define the function f(x)
def f (x : ℝ) : ℝ := -x^2 - 4*x + 1

-- State the theorem
theorem f_properties :
  (∃ (max_value : ℝ), max_value = 5 ∧ ∀ x, f x ≤ max_value) ∧
  (∀ x y, x < y ∧ y < -2 → f x < f y) ∧
  (∀ x y, -2 < x ∧ x < y → f x > f y) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l1295_129570


namespace NUMINAMATH_CALUDE_more_silver_than_gold_fish_l1295_129567

theorem more_silver_than_gold_fish (x g s r : ℕ) : 
  x = g + s + r →
  x - g = (2 * x) / 3 - 1 →
  x - r = (2 * x) / 3 + 4 →
  s = g + 2 := by
sorry

end NUMINAMATH_CALUDE_more_silver_than_gold_fish_l1295_129567


namespace NUMINAMATH_CALUDE_vector_perpendicular_l1295_129575

/-- Given vectors a and b in ℝ², prove that a - b is perpendicular to b -/
theorem vector_perpendicular (a b : ℝ × ℝ) (h1 : a = (1, 0)) (h2 : b = (1/2, 1/2)) : 
  (a - b) • b = 0 := by
  sorry

end NUMINAMATH_CALUDE_vector_perpendicular_l1295_129575


namespace NUMINAMATH_CALUDE_square_plus_one_geq_two_abs_l1295_129596

theorem square_plus_one_geq_two_abs (x : ℝ) : x^2 + 1 ≥ 2 * |x| := by
  sorry

end NUMINAMATH_CALUDE_square_plus_one_geq_two_abs_l1295_129596


namespace NUMINAMATH_CALUDE_work_earnings_problem_l1295_129509

theorem work_earnings_problem (t : ℝ) : 
  (t + 2) * (3 * t - 2) = (3 * t - 4) * (t + 3) + 3 → t = 5 := by
  sorry

end NUMINAMATH_CALUDE_work_earnings_problem_l1295_129509
