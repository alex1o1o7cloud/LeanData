import Mathlib

namespace dog_fruit_problem_l781_78121

/-- The number of bonnies eaten by the third dog -/
def B : ℕ := sorry

/-- The number of blueberries eaten by the second dog -/
def blueberries : ℕ := sorry

/-- The number of apples eaten by the first dog -/
def apples : ℕ := sorry

/-- The total number of fruits eaten by all three dogs -/
def total_fruits : ℕ := 240

theorem dog_fruit_problem :
  (blueberries = (3 * B) / 4) →
  (apples = 3 * blueberries) →
  (B + blueberries + apples = total_fruits) →
  B = 60 := by sorry

end dog_fruit_problem_l781_78121


namespace equation_solution_l781_78177

theorem equation_solution : ∃ x : ℝ, 0.05 * x + 0.12 * (30 + x) + 0.02 * (50 + 2 * x) = 20 ∧ x = 220 / 3 := by
  sorry

end equation_solution_l781_78177


namespace max_value_of_expression_l781_78103

theorem max_value_of_expression (x : ℝ) :
  (4 * x^2 + 8 * x + 19) / (4 * x^2 + 8 * x + 9) ≤ 3 ∧
  ∀ ε > 0, ∃ y : ℝ, (4 * y^2 + 8 * y + 19) / (4 * y^2 + 8 * y + 9) > 3 - ε :=
by sorry

end max_value_of_expression_l781_78103


namespace max_distance_from_circle_to_point_l781_78141

theorem max_distance_from_circle_to_point (z : ℂ) :
  Complex.abs z = 2 → (⨆ z, Complex.abs (z - Complex.I)) = 3 := by
  sorry

end max_distance_from_circle_to_point_l781_78141


namespace optimal_profit_l781_78157

-- Define the profit function
def profit (x : ℕ) : ℝ :=
  (500 - 10 * x) * (50 + x) - (500 - 10 * x) * 40

-- Define the optimal price increase
def optimal_price_increase : ℕ := 20

-- Define the optimal selling price
def optimal_selling_price : ℕ := 50 + optimal_price_increase

-- Define the maximum profit
def max_profit : ℝ := 9000

-- Theorem statement
theorem optimal_profit :
  (∀ x : ℕ, profit x ≤ profit optimal_price_increase) ∧
  (profit optimal_price_increase = max_profit) ∧
  (optimal_selling_price = 70) :=
sorry

end optimal_profit_l781_78157


namespace inequality_solution_l781_78187

theorem inequality_solution (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x : ℝ, a^(6 - x) > a^(2 + 3*x) ↔ 
    ((0 < a ∧ a < 1 ∧ x > 1) ∨ (a > 1 ∧ x < 1))) :=
by sorry

end inequality_solution_l781_78187


namespace wall_width_l781_78100

/-- The width of a wall given its dimensions, brick dimensions, and number of bricks required. -/
theorem wall_width
  (wall_length : ℝ)
  (wall_height : ℝ)
  (brick_length : ℝ)
  (brick_width : ℝ)
  (brick_height : ℝ)
  (num_bricks : ℕ)
  (h1 : wall_length = 7)
  (h2 : wall_height = 6)
  (h3 : brick_length = 0.25)
  (h4 : brick_width = 0.1125)
  (h5 : brick_height = 0.06)
  (h6 : num_bricks = 5600) :
  ∃ (wall_width : ℝ), wall_width = 0.225 ∧
    wall_length * wall_height * wall_width = ↑num_bricks * brick_length * brick_width * brick_height :=
by sorry

end wall_width_l781_78100


namespace specific_shape_perimeter_l781_78117

/-- A shape consisting of a regular hexagon, six triangles, and six squares -/
structure Shape where
  hexagon_side : ℝ
  num_triangles : ℕ
  num_squares : ℕ

/-- The outer perimeter of the shape -/
def outer_perimeter (s : Shape) : ℝ :=
  12 * s.hexagon_side

/-- Theorem stating that the outer perimeter of the specific shape is 216 cm -/
theorem specific_shape_perimeter :
  ∃ (s : Shape), s.hexagon_side = 18 ∧ s.num_triangles = 6 ∧ s.num_squares = 6 ∧ outer_perimeter s = 216 :=
by sorry

end specific_shape_perimeter_l781_78117


namespace even_number_selection_l781_78136

theorem even_number_selection (p : ℝ) (n : ℕ) 
  (h_p : p = 0.5) 
  (h_n : n = 4) : 
  1 - p^n ≥ 0.9 := by
sorry

end even_number_selection_l781_78136


namespace probability_less_than_three_l781_78183

/-- A square in the 2D plane --/
structure Square where
  bottomLeft : ℝ × ℝ
  topRight : ℝ × ℝ

/-- The probability that a randomly chosen point in the square satisfies a given condition --/
def probability (s : Square) (condition : ℝ × ℝ → Prop) : ℝ :=
  sorry

/-- The square with vertices (0,0), (0,2), (2,2), and (2,0) --/
def unitSquare : Square :=
  { bottomLeft := (0, 0), topRight := (2, 2) }

/-- The condition x + y < 3 --/
def lessThanThree (p : ℝ × ℝ) : Prop :=
  p.1 + p.2 < 3

theorem probability_less_than_three :
  probability unitSquare lessThanThree = 7/8 := by
  sorry

end probability_less_than_three_l781_78183


namespace lines_parabolas_intersection_empty_l781_78126

-- Define the set of all lines
def Lines := {f : ℝ → ℝ | ∃ (m b : ℝ), ∀ x, f x = m * x + b}

-- Define the set of all parabolas
def Parabolas := {f : ℝ → ℝ | ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c}

-- Theorem statement
theorem lines_parabolas_intersection_empty : Lines ∩ Parabolas = ∅ := by
  sorry

end lines_parabolas_intersection_empty_l781_78126


namespace zelda_success_probability_l781_78129

theorem zelda_success_probability 
  (p_xavier : ℝ) 
  (p_yvonne : ℝ) 
  (p_xy_not_z : ℝ) 
  (h1 : p_xavier = 1/3) 
  (h2 : p_yvonne = 1/2) 
  (h3 : p_xy_not_z = 0.0625) : 
  ∃ p_zelda : ℝ, p_zelda = 0.625 ∧ p_xavier * p_yvonne * (1 - p_zelda) = p_xy_not_z :=
by sorry

end zelda_success_probability_l781_78129


namespace chord_line_equation_l781_78106

/-- The equation of a line containing a chord of an ellipse, given the ellipse equation and the midpoint of the chord. -/
theorem chord_line_equation (a b c : ℝ) (x₀ y₀ : ℝ) :
  (∀ x y, x^2 + a*y^2 = b) →  -- Ellipse equation
  (∃ x₁ y₁ x₂ y₂ : ℝ,  -- Existence of chord endpoints
    x₁^2 + a*y₁^2 = b ∧
    x₂^2 + a*y₂^2 = b ∧
    x₀ = (x₁ + x₂) / 2 ∧
    y₀ = (y₁ + y₂) / 2) →
  (∃ k m : ℝ, ∀ x y, (x - x₀) + k*(y - y₀) = 0 ↔ x + k*y = m) →
  (a = 4 ∧ b = 36 ∧ x₀ = 4 ∧ y₀ = 2 ∧ c = 8) →
  (∀ x y, x + 2*y - c = 0 ↔ (x - x₀) + 2*(y - y₀) = 0) :=
by sorry

#check chord_line_equation

end chord_line_equation_l781_78106


namespace student_comprehensive_score_l781_78133

/-- Calculates the comprehensive score of a student in a competition --/
def comprehensiveScore (theoreticalWeight : ℝ) (innovativeWeight : ℝ) (presentationWeight : ℝ)
                       (theoreticalScore : ℝ) (innovativeScore : ℝ) (presentationScore : ℝ) : ℝ :=
  theoreticalWeight * theoreticalScore + innovativeWeight * innovativeScore + presentationWeight * presentationScore

/-- Theorem stating that the student's comprehensive score is 89.5 --/
theorem student_comprehensive_score :
  let theoreticalWeight : ℝ := 0.20
  let innovativeWeight : ℝ := 0.50
  let presentationWeight : ℝ := 0.30
  let theoreticalScore : ℝ := 80
  let innovativeScore : ℝ := 90
  let presentationScore : ℝ := 95
  comprehensiveScore theoreticalWeight innovativeWeight presentationWeight
                     theoreticalScore innovativeScore presentationScore = 89.5 := by
  sorry

end student_comprehensive_score_l781_78133


namespace payback_time_l781_78163

def initial_cost : ℝ := 25000
def monthly_revenue : ℝ := 4000
def monthly_expenses : ℝ := 1500

theorem payback_time :
  let monthly_profit := monthly_revenue - monthly_expenses
  (initial_cost / monthly_profit : ℝ) = 10 := by sorry

end payback_time_l781_78163


namespace range_of_a_l781_78147

theorem range_of_a (a : ℝ) : 
  (∀ b : ℝ, ∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ |x^2 + a*x + b| ≥ 1) ↔ 
  (a ≥ 1 ∨ a ≤ -3) := by sorry

end range_of_a_l781_78147


namespace drive_time_proof_l781_78154

/-- Proves the time driven at 60 mph given the conditions of the problem -/
theorem drive_time_proof (total_distance : ℝ) (initial_speed : ℝ) (final_speed : ℝ) (total_time : ℝ)
  (h1 : total_distance = 120)
  (h2 : initial_speed = 60)
  (h3 : final_speed = 90)
  (h4 : total_time = 1.5) :
  ∃ t : ℝ, t + (total_distance - initial_speed * t) / final_speed = total_time ∧ t = 0.5 := by
  sorry

end drive_time_proof_l781_78154


namespace pet_store_parrot_count_l781_78148

theorem pet_store_parrot_count (total_birds : ℕ) (num_cages : ℕ) (parakeets_per_cage : ℕ) 
  (h1 : total_birds = 48)
  (h2 : num_cages = 6)
  (h3 : parakeets_per_cage = 2) :
  (total_birds - num_cages * parakeets_per_cage) / num_cages = 6 := by
  sorry

#check pet_store_parrot_count

end pet_store_parrot_count_l781_78148


namespace not_all_primes_in_arithmetic_progression_l781_78145

def arithmetic_progression (a d : ℤ) (n : ℕ) : ℤ := a + d * n

theorem not_all_primes_in_arithmetic_progression (a d : ℤ) (h : d ≥ 1) :
  ∃ n : ℕ, ¬ Prime (arithmetic_progression a d n) :=
sorry

end not_all_primes_in_arithmetic_progression_l781_78145


namespace a_plus_b_and_abs_a_minus_b_l781_78119

theorem a_plus_b_and_abs_a_minus_b (a b : ℝ) 
  (h1 : |a| = 2) 
  (h2 : b^2 = 25) 
  (h3 : a * b < 0) : 
  ((a + b = 3) ∨ (a + b = -3)) ∧ |a - b| = 7 := by
sorry

end a_plus_b_and_abs_a_minus_b_l781_78119


namespace glasses_in_smaller_box_l781_78115

theorem glasses_in_smaller_box :
  ∀ x : ℕ,
  (x + 16) / 2 = 15 →
  x = 14 :=
by
  sorry

end glasses_in_smaller_box_l781_78115


namespace rectangle_area_l781_78197

/-- Given a right triangle ABC composed of two right triangles and a rectangle,
    prove that the area of the rectangle is 750 square centimeters. -/
theorem rectangle_area (AE BF : ℝ) (h1 : AE = 30) (h2 : BF = 25) : AE * BF = 750 := by
  sorry

end rectangle_area_l781_78197


namespace g_of_2_eq_11_l781_78102

/-- Given a function g(x) = 3x^2 + 2x - 5, prove that g(2) = 11 -/
theorem g_of_2_eq_11 : let g : ℝ → ℝ := fun x ↦ 3 * x^2 + 2 * x - 5; g 2 = 11 := by
  sorry

end g_of_2_eq_11_l781_78102


namespace fuel_tank_capacity_l781_78180

/-- The capacity of a fuel tank given specific conditions -/
theorem fuel_tank_capacity : ∃ (C : ℝ), 
  (0.12 * 82 + 0.16 * (C - 82) = 30) ∧ 
  (C = 208) := by
  sorry

end fuel_tank_capacity_l781_78180


namespace prob_at_least_one_red_l781_78155

/-- The probability of selecting at least one red ball when randomly choosing 2 balls out of 5 balls (2 red and 3 white) is 7/10. -/
theorem prob_at_least_one_red (total : ℕ) (red : ℕ) (white : ℕ) (select : ℕ) :
  total = 5 →
  red = 2 →
  white = 3 →
  select = 2 →
  (Nat.choose total select - Nat.choose white select : ℚ) / Nat.choose total select = 7 / 10 :=
by sorry

end prob_at_least_one_red_l781_78155


namespace freds_weekend_earnings_l781_78105

/-- Fred's earnings from delivering newspapers -/
def newspaper_earnings : ℕ := 16

/-- Fred's earnings from washing cars -/
def car_washing_earnings : ℕ := 74

/-- Fred's total weekend earnings -/
def weekend_earnings : ℕ := 90

/-- Theorem stating that Fred's weekend earnings equal the sum of his newspaper delivery and car washing earnings -/
theorem freds_weekend_earnings : 
  newspaper_earnings + car_washing_earnings = weekend_earnings := by
  sorry

end freds_weekend_earnings_l781_78105


namespace jungkook_final_ball_count_l781_78143

-- Define the initial state
def jungkook_red_balls : ℕ := 3
def yoongi_blue_balls : ℕ := 2

-- Define the transfer
def transferred_balls : ℕ := 1

-- Theorem to prove
theorem jungkook_final_ball_count :
  jungkook_red_balls + transferred_balls = 4 := by
  sorry

end jungkook_final_ball_count_l781_78143


namespace no_common_solution_l781_78192

theorem no_common_solution : ¬∃ (x y : ℝ), x^2 + y^2 = 16 ∧ x^2 + 3*y + 30 = 0 := by
  sorry

end no_common_solution_l781_78192


namespace empty_can_weight_l781_78122

theorem empty_can_weight (full_can : ℝ) (half_can : ℝ) (h1 : full_can = 35) (h2 : half_can = 18) :
  full_can - 2 * (full_can - half_can) = 1 :=
by sorry

end empty_can_weight_l781_78122


namespace complex_equation_solution_l781_78156

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_equation_solution :
  ∃ (x : ℂ), (3 - 2 * i * x = 5 + 4 * i * x) ∧ (x = i / 3) :=
by
  sorry

end complex_equation_solution_l781_78156


namespace total_share_calculation_l781_78153

/-- Given three shares x, y, and z, where x is 25% more than y, y is 20% more than z,
    and z is 100, prove that the total amount shared is 370. -/
theorem total_share_calculation (x y z : ℚ) : 
  x = 1.25 * y ∧ y = 1.2 * z ∧ z = 100 → x + y + z = 370 := by
  sorry

end total_share_calculation_l781_78153


namespace plane_sphere_intersection_l781_78146

/-- Given a plane passing through (d,e,f) and intersecting the coordinate axes at D, E, F,
    with (u,v,w) as the center of the sphere through D, E, F, and the origin,
    prove that d/u + e/v + f/w = 2 -/
theorem plane_sphere_intersection (d e f u v w : ℝ) : 
  (∃ (δ ε ϕ : ℝ), 
    δ ≠ 0 ∧ ε ≠ 0 ∧ ϕ ≠ 0 ∧
    u^2 + v^2 + w^2 = (u - δ)^2 + v^2 + w^2 ∧
    u^2 + v^2 + w^2 = u^2 + (v - ε)^2 + w^2 ∧
    u^2 + v^2 + w^2 = u^2 + v^2 + (w - ϕ)^2 ∧
    d / δ + e / ε + f / ϕ = 1) →
  d / u + e / v + f / w = 2 :=
by sorry

end plane_sphere_intersection_l781_78146


namespace order_of_logarithmic_fractions_l781_78169

theorem order_of_logarithmic_fractions :
  let a : ℝ := (Real.exp 1)⁻¹
  let b : ℝ := (Real.log 2) / 2
  let c : ℝ := (Real.log 3) / 3
  b < c ∧ c < a := by sorry

end order_of_logarithmic_fractions_l781_78169


namespace orange_juice_serving_volume_l781_78144

/-- Proves that the volume of each serving of orange juice is 6 ounces given the specified conditions. -/
theorem orange_juice_serving_volume
  (concentrate_cans : ℕ)
  (concentrate_oz_per_can : ℕ)
  (water_cans_per_concentrate : ℕ)
  (total_servings : ℕ)
  (h1 : concentrate_cans = 60)
  (h2 : concentrate_oz_per_can = 5)
  (h3 : water_cans_per_concentrate = 3)
  (h4 : total_servings = 200) :
  (concentrate_cans * concentrate_oz_per_can * (water_cans_per_concentrate + 1)) / total_servings = 6 :=
by sorry

end orange_juice_serving_volume_l781_78144


namespace quadratic_equation_one_l781_78188

theorem quadratic_equation_one (x : ℝ) :
  (x + 2) * (x - 2) - 2 * (x - 3) = 3 ↔ x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2 := by
sorry

end quadratic_equation_one_l781_78188


namespace current_speed_l781_78108

/-- Proves that the speed of the current is 20 kmph, given the boat's speed in still water and upstream. -/
theorem current_speed (boat_still_speed upstream_speed : ℝ) 
  (h1 : boat_still_speed = 50)
  (h2 : upstream_speed = 30) :
  boat_still_speed - upstream_speed = 20 := by
  sorry

#check current_speed

end current_speed_l781_78108


namespace cargo_per_truck_l781_78124

/-- Represents the problem of determining the cargo per truck given certain conditions --/
theorem cargo_per_truck (x : ℝ) (n : ℕ) (h1 : 55 ≤ x ∧ x ≤ 64) 
  (h2 : x = (x / n - 0.5) * (n + 4)) : 
  x / (n + 4) = 2.5 := by
  sorry

end cargo_per_truck_l781_78124


namespace trigonometric_product_equality_l781_78165

theorem trigonometric_product_equality : 
  (1 - 1 / Real.cos (30 * π / 180)) * 
  (1 + 2 / Real.sin (60 * π / 180)) * 
  (1 - 1 / Real.sin (30 * π / 180)) * 
  (1 + 2 / Real.cos (60 * π / 180)) = 
  (25 - 10 * Real.sqrt 3) / 3 := by sorry

end trigonometric_product_equality_l781_78165


namespace min_value_theorem_l781_78170

/-- Given positive real numbers a and b, and a function f with minimum value 4 -/
def f (x a b : ℝ) : ℝ := |x + a| + |x - b|

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (hf : ∀ x, f x a b ≥ 4) (hf_min : ∃ x, f x a b = 4) :
  (a + b = 4) ∧ (∀ a b, a > 0 → b > 0 → a + b = 4 → (1/4) * a^2 + (1/4) * b^2 ≥ 3/16) ∧
  (∃ a b, a > 0 ∧ b > 0 ∧ a + b = 4 ∧ (1/4) * a^2 + (1/4) * b^2 = 3/16) :=
by sorry

end min_value_theorem_l781_78170


namespace prime_squared_plus_41_composite_l781_78184

theorem prime_squared_plus_41_composite (p : ℕ) (hp : Prime p) :
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ p^2 + 41 = a * b :=
sorry

end prime_squared_plus_41_composite_l781_78184


namespace expression_factorization_l781_78195

theorem expression_factorization (x : ℝ) :
  (12 * x^3 + 45 * x^2 - 15) - (-3 * x^3 + 6 * x^2 - 3) = 3 * (5 * x^3 + 13 * x^2 - 4) := by
  sorry

end expression_factorization_l781_78195


namespace x_value_equality_l781_78132

theorem x_value_equality : (2023^2 - 2023 - 10000) / 2023 = (2022 * 2023 - 10000) / 2023 := by
  sorry

end x_value_equality_l781_78132


namespace point_in_third_quadrant_implies_a_less_than_one_l781_78114

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point is in the third quadrant -/
def in_third_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- Theorem: If M(-1, a-1) is in the third quadrant, then a < 1 -/
theorem point_in_third_quadrant_implies_a_less_than_one (a : ℝ) :
  in_third_quadrant (Point.mk (-1) (a - 1)) → a < 1 := by
  sorry

end point_in_third_quadrant_implies_a_less_than_one_l781_78114


namespace complex_number_location_l781_78162

theorem complex_number_location (z : ℂ) (h : z = Complex.I * (1 + Complex.I)) :
  Complex.re z < 0 ∧ Complex.im z > 0 := by
  sorry

end complex_number_location_l781_78162


namespace three_distinct_solutions_l781_78160

theorem three_distinct_solutions : ∃ (x₁ x₂ x₃ : ℝ), 
  (356 * x₁ = 2492) ∧ 
  (x₂ / 39 = 235) ∧ 
  (1908 - x₃ = 529) ∧ 
  (x₁ ≠ x₂) ∧ (x₁ ≠ x₃) ∧ (x₂ ≠ x₃) :=
by sorry

end three_distinct_solutions_l781_78160


namespace sqrt_equation_solution_l781_78158

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt (5 * x + 12) = 13 - x →
  x = (31 + Real.sqrt 333) / 2 ∨ x = (31 - Real.sqrt 333) / 2 := by
  sorry

end sqrt_equation_solution_l781_78158


namespace hyperbola_asymptotes_l781_78107

/-- The equations of the asymptotes for the hyperbola x²/16 - y²/9 = 1 are y = ±(3/4)x -/
theorem hyperbola_asymptotes :
  let h : ℝ → ℝ → Prop := fun x y ↦ x^2/16 - y^2/9 = 1
  ∃ (f g : ℝ → ℝ), (∀ x, f x = (3/4) * x) ∧ (∀ x, g x = -(3/4) * x) ∧
    (∀ ε > 0, ∃ M > 0, ∀ x y, h x y → (|x| > M → |y - f x| < ε ∨ |y - g x| < ε)) :=
by sorry

end hyperbola_asymptotes_l781_78107


namespace cube_sum_zero_or_abc_function_l781_78128

theorem cube_sum_zero_or_abc_function (a b c : ℝ) 
  (nonzero_a : a ≠ 0) (nonzero_b : b ≠ 0) (nonzero_c : c ≠ 0)
  (sum_zero : a + b + c = 0)
  (fourth_sixth_power_eq : a^4 + b^4 + c^4 = a^6 + b^6 + c^6) :
  (a^3 + b^3 + c^3 = 0) ∨ (∃ f : ℝ → ℝ → ℝ → ℝ, a^3 + b^3 + c^3 = f a b c) :=
by sorry

end cube_sum_zero_or_abc_function_l781_78128


namespace smallest_missing_number_is_22_l781_78171

/-- Represents a problem in HMMT November 2023 -/
structure HMMTProblem where
  round : String
  number : Nat

/-- The set of all problems in HMMT November 2023 -/
def HMMTProblems : Set HMMTProblem := sorry

/-- A number appears in HMMT November 2023 if it's used in at least one problem -/
def appears_in_HMMT (n : Nat) : Prop :=
  ∃ (p : HMMTProblem), p ∈ HMMTProblems ∧ p.number = n

theorem smallest_missing_number_is_22 :
  (∀ n : Nat, n > 0 ∧ n ≤ 21 → appears_in_HMMT n) →
  (¬ appears_in_HMMT 22) →
  ∀ m : Nat, m > 0 ∧ ¬ appears_in_HMMT m → m ≥ 22 :=
sorry

end smallest_missing_number_is_22_l781_78171


namespace min_value_reciprocal_squares_l781_78168

theorem min_value_reciprocal_squares (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (sum_constraint : a + b + c = 3) : 
  1/a^2 + 1/b^2 + 1/c^2 ≥ 3 ∧ 
  (1/a^2 + 1/b^2 + 1/c^2 = 3 ↔ a = 1 ∧ b = 1 ∧ c = 1) := by
  sorry

#check min_value_reciprocal_squares

end min_value_reciprocal_squares_l781_78168


namespace smallest_five_digit_divisible_by_first_five_primes_l781_78112

theorem smallest_five_digit_divisible_by_first_five_primes :
  ∃ (n : ℕ), 
    (n ≥ 10000 ∧ n < 100000) ∧ 
    (∀ m : ℕ, m ≥ 10000 ∧ m < 100000 → 
      (2 ∣ m ∧ 3 ∣ m ∧ 5 ∣ m ∧ 7 ∣ m ∧ 11 ∣ m) → m ≥ n) ∧
    (2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧ 11 ∣ n) ∧
    n = 11550 :=
by
  sorry

end smallest_five_digit_divisible_by_first_five_primes_l781_78112


namespace percentage_problem_l781_78194

theorem percentage_problem (X : ℝ) (h : 0.2 * X = 300) : 
  ∃ P : ℝ, (P / 100) * X = 1800 ∧ P = 120 := by
  sorry

end percentage_problem_l781_78194


namespace range_of_m_l781_78161

theorem range_of_m (a b : ℝ) (ha : a > 0) (hb : b > 1) (hab : a + b = 2)
  (h_ineq : ∀ m : ℝ, (4/a) + 1/(b-1) > m^2 + 8*m) :
  ∀ m : ℝ, (4/a) + 1/(b-1) > m^2 + 8*m → -9 < m ∧ m < 1 :=
by sorry

end range_of_m_l781_78161


namespace match_total_weight_l781_78116

/-- The number of times Terrell lifts the original weights -/
def original_lifts : ℕ := 10

/-- The weight of each original weight in pounds -/
def original_weight : ℕ := 25

/-- The weight of each new weight in pounds -/
def new_weight : ℕ := 20

/-- The number of weights used in each setup -/
def num_weights : ℕ := 2

/-- The total weight lifted with the original setup in pounds -/
def total_weight : ℕ := num_weights * original_weight * original_lifts

/-- The number of lifts required with the new weights to match the total weight -/
def required_lifts : ℚ := total_weight / (num_weights * new_weight)

theorem match_total_weight : required_lifts = 12.5 := by sorry

end match_total_weight_l781_78116


namespace workout_solution_correct_l781_78113

/-- Laura's workout parameters -/
structure WorkoutParams where
  bike_distance : ℝ
  bike_rate : ℝ → ℝ
  transition_time : ℝ
  run_distance : ℝ
  total_time : ℝ

/-- The solution to Laura's workout problem -/
def workout_solution (p : WorkoutParams) : ℝ :=
  8

/-- Theorem stating that the workout_solution is correct -/
theorem workout_solution_correct (p : WorkoutParams) 
  (h1 : p.bike_distance = 25)
  (h2 : p.bike_rate = fun x => 3 * x + 1)
  (h3 : p.transition_time = 1/6)  -- 10 minutes in hours
  (h4 : p.run_distance = 8)
  (h5 : p.total_time = 13/6)  -- 130 minutes in hours
  : ∃ (x : ℝ), 
    x = workout_solution p ∧ 
    p.bike_distance / (p.bike_rate x) + p.transition_time + p.run_distance / x = p.total_time :=
  sorry

#check workout_solution_correct

end workout_solution_correct_l781_78113


namespace f_2023_equals_1_l781_78131

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def period_4 (f : ℝ → ℝ) : Prop := ∀ x, f (x + 4) = f x

theorem f_2023_equals_1 (f : ℝ → ℝ) 
  (h1 : is_even f)
  (h2 : ∀ x, f (x + 2) = f (2 - x))
  (h3 : ∀ x ∈ Set.Icc 0 2, f x = x^2) : 
  f 2023 = 1 := by
  sorry

end f_2023_equals_1_l781_78131


namespace odd_sum_difference_l781_78109

def sum_odd_range (a b : ℕ) : ℕ :=
  let first := if a % 2 = 1 then a else a + 1
  let last := if b % 2 = 1 then b else b - 1
  let n := (last - first) / 2 + 1
  n * (first + last) / 2

theorem odd_sum_difference : 
  sum_odd_range 101 300 - sum_odd_range 3 70 = 18776 := by
  sorry

end odd_sum_difference_l781_78109


namespace angela_is_157_cm_tall_l781_78185

def amy_height : ℕ := 150

def helen_height (amy : ℕ) : ℕ := amy + 3

def angela_height (helen : ℕ) : ℕ := helen + 4

theorem angela_is_157_cm_tall :
  angela_height (helen_height amy_height) = 157 :=
by sorry

end angela_is_157_cm_tall_l781_78185


namespace gcd_2183_1947_l781_78135

theorem gcd_2183_1947 : Nat.gcd 2183 1947 = 59 := by sorry

end gcd_2183_1947_l781_78135


namespace range_of_a_l781_78167

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (1 - 2*a)*x + 3*a else 2^(x - 1)

theorem range_of_a : 
  ∀ a : ℝ, (∀ x : ℝ, ∃ y : ℝ, f a x = y) ↔ (0 ≤ a ∧ a < 1/2) :=
sorry

end range_of_a_l781_78167


namespace shaded_area_of_squares_l781_78150

theorem shaded_area_of_squares (s₁ s₂ : ℝ) (h₁ : s₁ = 2) (h₂ : s₂ = 6) :
  (1/2 * s₁ * s₁) + (1/2 * s₂ * s₂) = 20 := by
  sorry

end shaded_area_of_squares_l781_78150


namespace chocolate_bars_distribution_l781_78198

theorem chocolate_bars_distribution (large_box_total : ℕ) (small_boxes : ℕ) (bars_per_small_box : ℕ) :
  large_box_total = 375 →
  small_boxes = 15 →
  large_box_total = small_boxes * bars_per_small_box →
  bars_per_small_box = 25 := by
  sorry

end chocolate_bars_distribution_l781_78198


namespace round_trip_ticket_holders_l781_78189

/-- The percentage of ship passengers holding round-trip tickets -/
def round_trip_percentage : ℝ := 62.5

theorem round_trip_ticket_holders (total_passengers : ℝ) (round_trip_with_car : ℝ) (round_trip_without_car : ℝ)
  (h1 : round_trip_with_car = 0.25 * total_passengers)
  (h2 : round_trip_without_car = 0.6 * (round_trip_with_car + round_trip_without_car)) :
  (round_trip_with_car + round_trip_without_car) / total_passengers * 100 = round_trip_percentage := by
  sorry

end round_trip_ticket_holders_l781_78189


namespace pet_store_cages_theorem_l781_78164

/-- Given a number of initial puppies, sold puppies, and puppies per cage,
    calculate the number of cages needed. -/
def cagesNeeded (initialPuppies soldPuppies puppiesPerCage : ℕ) : ℕ :=
  ((initialPuppies - soldPuppies) + puppiesPerCage - 1) / puppiesPerCage

theorem pet_store_cages_theorem :
  cagesNeeded 36 7 4 = 8 := by
  sorry

end pet_store_cages_theorem_l781_78164


namespace largest_base_sum_not_sixteen_l781_78199

/-- Represents a number in a given base --/
structure BaseNumber (base : ℕ) where
  digits : List ℕ
  valid : ∀ d ∈ digits, d < base

/-- Computes the sum of digits of a BaseNumber --/
def sumOfDigits {base : ℕ} (n : BaseNumber base) : ℕ :=
  n.digits.sum

/-- Represents 11^4 in different bases --/
def elevenFourth (base : ℕ) : BaseNumber base :=
  if base ≥ 7 then
    ⟨[1, 4, 6, 4, 1], sorry⟩
  else if base = 6 then
    ⟨[1, 5, 0, 4, 1], sorry⟩
  else
    ⟨[], sorry⟩  -- Undefined for bases less than 6

/-- The theorem to be proved --/
theorem largest_base_sum_not_sixteen :
  (∃ b : ℕ, b > 0 ∧ sumOfDigits (elevenFourth b) ≠ 16) ∧
  (∀ b : ℕ, b > 6 → sumOfDigits (elevenFourth b) = 16) :=
sorry

end largest_base_sum_not_sixteen_l781_78199


namespace profit_after_five_days_days_for_ten_thousand_profit_l781_78174

/-- Profit calculation function -/
def profit (x : ℝ) : ℝ :=
  (50 + 2*x) * (700 - 15*x) - 700 * 40 - 50 * x

/-- Theorem for profit after 5 days -/
theorem profit_after_five_days : profit 5 = 9250 := by sorry

/-- Theorem for days to store for 10,000 yuan profit -/
theorem days_for_ten_thousand_profit :
  ∃ x : ℝ, x ≥ 0 ∧ x ≤ 15 ∧ profit x = 10000 ∧ x = 10 := by sorry

end profit_after_five_days_days_for_ten_thousand_profit_l781_78174


namespace lcm_150_414_l781_78181

theorem lcm_150_414 : Nat.lcm 150 414 = 10350 := by
  sorry

end lcm_150_414_l781_78181


namespace discount_percentage_l781_78149

theorem discount_percentage 
  (profit_with_discount : ℝ) 
  (profit_without_discount : ℝ) 
  (h1 : profit_with_discount = 0.235) 
  (h2 : profit_without_discount = 0.30) : 
  (profit_without_discount - profit_with_discount) / (1 + profit_without_discount) = 0.05 := by
sorry

end discount_percentage_l781_78149


namespace pencil_problem_l781_78186

theorem pencil_problem (s p t : ℚ) : 
  (6 * s = 12) →
  (t = 8 * s) →
  (p = 2.5 * s + 3) →
  (t = 16 ∧ p = 8) := by
sorry

end pencil_problem_l781_78186


namespace investment_rate_proof_l781_78152

/-- Proves that given the described investment scenario, the initial interest rate is approximately 0.2 -/
theorem investment_rate_proof (initial_investment : ℝ) (years : ℕ) (final_amount : ℝ) : 
  initial_investment = 10000 →
  years = 3 →
  final_amount = 59616 →
  ∃ (r : ℝ), 
    (r ≥ 0) ∧ 
    (r ≤ 1) ∧
    (abs (r - 0.2) < 0.001) ∧
    (final_amount = 3 * initial_investment * (1 + r)^years * 1.15) :=
by sorry

end investment_rate_proof_l781_78152


namespace bicycle_spokes_theorem_l781_78110

/-- The number of spokes on each bicycle wheel given the total number of bicycles and spokes -/
def spokes_per_wheel (num_bicycles : ℕ) (total_spokes : ℕ) : ℕ :=
  total_spokes / (num_bicycles * 2)

/-- Theorem stating that 4 bicycles with a total of 80 spokes have 10 spokes per wheel -/
theorem bicycle_spokes_theorem :
  spokes_per_wheel 4 80 = 10 := by
  sorry

end bicycle_spokes_theorem_l781_78110


namespace log_power_sum_l781_78196

/-- Given a = log 25 and b = log 36, prove that 5^(a/b) + 6^(b/a) = 11 -/
theorem log_power_sum (a b : ℝ) (ha : a = Real.log 25) (hb : b = Real.log 36) :
  (5 : ℝ) ^ (a / b) + (6 : ℝ) ^ (b / a) = 11 := by sorry

end log_power_sum_l781_78196


namespace ladder_problem_l781_78166

theorem ladder_problem (ladder_length height base : ℝ) :
  ladder_length = 13 ∧ height = 12 ∧ ladder_length^2 = height^2 + base^2 → base = 5 := by
  sorry

end ladder_problem_l781_78166


namespace sum_of_squares_inequality_l781_78125

theorem sum_of_squares_inequality (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : a + b + c = 1) :
  (1 - a^2)^2 + (1 - b^2)^2 + (1 - c^2)^2 ≥ 2 := by sorry

end sum_of_squares_inequality_l781_78125


namespace max_value_product_l781_78178

theorem max_value_product (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 3) :
  (a^2 - a*b + b^2) * (a^2 - a*c + c^2) * (b^2 - b*c + c^2) ≤ 729/1296 ∧
  ∃ a b c, (a^2 - a*b + b^2) * (a^2 - a*c + c^2) * (b^2 - b*c + c^2) = 729/1296 :=
by sorry

end max_value_product_l781_78178


namespace rope_folded_six_times_l781_78190

/-- The number of segments a rope is cut into after being folded n times and cut along the middle -/
def rope_segments (n : ℕ) : ℕ := 2^n + 1

/-- Theorem: A rope folded in half 6 times and cut along the middle will result in 65 segments -/
theorem rope_folded_six_times : rope_segments 6 = 65 := by
  sorry

end rope_folded_six_times_l781_78190


namespace system_solution_l781_78176

theorem system_solution (x y a : ℝ) : 
  (3 * x + y = 1 + 3 * a) → 
  (x + 3 * y = 1 - a) → 
  (x + y = 0) → 
  (a = -1) := by
sorry

end system_solution_l781_78176


namespace ring_area_equals_three_circles_l781_78104

theorem ring_area_equals_three_circles 
  (r₁ r₂ r₃ d R r : ℝ) (h_positive : r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0 ∧ d > 0) :
  (R^2 - r^2 = r₁^2 + r₂^2 + r₃^2) ∧ (R - r = d) →
  (R = ((r₁^2 + r₂^2 + r₃^2) + d^2) / (2*d)) ∧ (r = R - d) := by
sorry

end ring_area_equals_three_circles_l781_78104


namespace inequality_holds_iff_k_in_range_l781_78179

theorem inequality_holds_iff_k_in_range (k : ℝ) : 
  (k > 0 ∧ 
   ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ + x₂ = k → 
   (1/x₁ - x₁) * (1/x₂ - x₂) ≥ (k/2 - 2/k)^2) 
  ↔ 
  (0 < k ∧ k ≤ 2 * Real.sqrt (Real.sqrt 5 - 2)) :=
by sorry

end inequality_holds_iff_k_in_range_l781_78179


namespace two_books_different_genres_l781_78191

/-- Represents the number of books in each genre -/
def books_per_genre : ℕ := 3

/-- Represents the number of genres -/
def num_genres : ℕ := 3

/-- Represents the total number of books -/
def total_books : ℕ := books_per_genre * num_genres

/-- Calculates the number of ways to choose two books of different genres -/
def choose_two_different_genres : ℕ := 
  (total_books * (total_books - books_per_genre)) / 2

theorem two_books_different_genres : 
  choose_two_different_genres = 27 := by sorry

end two_books_different_genres_l781_78191


namespace exists_valid_relation_l781_78123

-- Define the type for positive integers
def PositiveInt := {n : ℕ // n > 0}

-- Define the properties of the relation
def IsValidRelation (r : PositiveInt → PositiveInt → Prop) : Prop :=
  -- For any pair, exactly one of the three conditions holds
  (∀ a b : PositiveInt, (r a b ∨ r b a ∨ a = b) ∧ 
    (r a b → ¬r b a ∧ a ≠ b) ∧
    (r b a → ¬r a b ∧ a ≠ b) ∧
    (a = b → ¬r a b ∧ ¬r b a)) ∧
  -- Transitivity
  (∀ a b c : PositiveInt, r a b → r b c → r a c) ∧
  -- The special property
  (∀ a b c : PositiveInt, r a b → r b c → 2 * b.val ≠ a.val + c.val)

-- Theorem statement
theorem exists_valid_relation : ∃ r : PositiveInt → PositiveInt → Prop, IsValidRelation r :=
sorry

end exists_valid_relation_l781_78123


namespace remainder_after_adding_150_l781_78142

theorem remainder_after_adding_150 (n : ℤ) :
  n % 6 = 1 → (n + 150) % 6 = 1 := by
sorry

end remainder_after_adding_150_l781_78142


namespace interior_triangles_count_l781_78130

/-- The number of points on the circle -/
def n : ℕ := 9

/-- The number of triangles formed inside the circle -/
def num_triangles : ℕ := Nat.choose n 6

/-- Theorem stating the number of triangles formed inside the circle -/
theorem interior_triangles_count : num_triangles = 84 := by
  sorry

end interior_triangles_count_l781_78130


namespace exchange_cookies_to_bagels_l781_78182

/-- Represents the exchange rate between gingerbread cookies and drying rings -/
def cookie_to_ring : ℚ := 6

/-- Represents the exchange rate between drying rings and bagels -/
def ring_to_bagel : ℚ := 4/9

/-- Represents the number of gingerbread cookies we want to exchange -/
def cookies : ℚ := 3

/-- Theorem stating that 3 gingerbread cookies can be exchanged for 8 bagels -/
theorem exchange_cookies_to_bagels :
  cookies * cookie_to_ring * ring_to_bagel = 8 := by
  sorry

end exchange_cookies_to_bagels_l781_78182


namespace tom_marble_groups_l781_78111

def red_marble : ℕ := 1
def green_marble : ℕ := 1
def blue_marble : ℕ := 1
def black_marble : ℕ := 1
def yellow_marbles : ℕ := 4

def total_marbles : ℕ := red_marble + green_marble + blue_marble + black_marble + yellow_marbles

def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

theorem tom_marble_groups :
  let non_yellow_choices := choose_two (red_marble + green_marble + blue_marble + black_marble) - 1
  let yellow_combinations := choose_two yellow_marbles
  let color_with_yellow := red_marble + green_marble + blue_marble + black_marble
  non_yellow_choices + yellow_combinations + color_with_yellow = 10 :=
by sorry

end tom_marble_groups_l781_78111


namespace total_games_in_our_league_l781_78120

/-- Represents a sports league with sub-leagues and playoffs -/
structure SportsLeague where
  total_teams : Nat
  num_sub_leagues : Nat
  teams_per_sub_league : Nat
  games_against_each_team : Nat
  teams_advancing : Nat

/-- Calculates the total number of games in the entire season -/
def total_games (league : SportsLeague) : Nat :=
  let sub_league_games := league.num_sub_leagues * (league.teams_per_sub_league * (league.teams_per_sub_league - 1) / 2 * league.games_against_each_team)
  let playoff_teams := league.num_sub_leagues * league.teams_advancing
  let playoff_games := playoff_teams * (playoff_teams - 1) / 2
  sub_league_games + playoff_games

/-- The specific league configuration -/
def our_league : SportsLeague :=
  { total_teams := 100
  , num_sub_leagues := 5
  , teams_per_sub_league := 20
  , games_against_each_team := 6
  , teams_advancing := 4 }

/-- Theorem stating that the total number of games in our league is 5890 -/
theorem total_games_in_our_league : total_games our_league = 5890 := by
  sorry

end total_games_in_our_league_l781_78120


namespace sum_of_decimals_l781_78173

theorem sum_of_decimals : (1 : ℚ) + 0.101 + 0.011 + 0.001 = 1.113 := by sorry

end sum_of_decimals_l781_78173


namespace expression_equality_l781_78118

theorem expression_equality : 
  abs (-3) - Real.sqrt 8 - (1/2)⁻¹ + 2 * Real.cos (π/4) = 1 - Real.sqrt 2 := by
  sorry

end expression_equality_l781_78118


namespace range_x_when_a_zero_range_a_for_p_sufficient_q_l781_78139

-- Define the conditions p and q
def p (x : ℝ) : Prop := 4 * x^2 + 12 * x - 7 ≤ 0
def q (a x : ℝ) : Prop := a - 3 ≤ x ∧ x ≤ a + 3

-- Theorem for the first question
theorem range_x_when_a_zero :
  ∀ x : ℝ, p x ∧ ¬(q 0 x) ↔ -7/2 ≤ x ∧ x < -3 :=
sorry

-- Theorem for the second question
theorem range_a_for_p_sufficient_q :
  (∀ x : ℝ, p x → ∀ a : ℝ, q a x) ↔ ∀ a : ℝ, -5/2 ≤ a ∧ a ≤ -1/2 :=
sorry

end range_x_when_a_zero_range_a_for_p_sufficient_q_l781_78139


namespace inequality_solution_set_l781_78151

theorem inequality_solution_set (x : ℝ) : 
  (x^2 * (x + 1)) / (-x^2 - 5*x + 6) ≤ 0 ∧ (-x^2 - 5*x + 6) ≠ 0 ↔ 
  (-6 < x ∧ x ≤ -1) ∨ x = 0 ∨ x > 1 :=
sorry

end inequality_solution_set_l781_78151


namespace cube_difference_formula_l781_78127

theorem cube_difference_formula (n : ℕ) : 
  (n + 1)^3 - n^3 = 3*n^2 + 3*n + 1 := by
  sorry

end cube_difference_formula_l781_78127


namespace ball_sampling_theorem_l781_78101

/-- Represents the color of a ball -/
inductive BallColor
  | White
  | Black

/-- Represents the bag with balls -/
structure Bag :=
  (white : ℕ)
  (black : ℕ)

/-- Represents the sampling method -/
inductive SamplingMethod
  | WithReplacement
  | WithoutReplacement

/-- The probability of drawing two balls of different colors with replacement -/
def prob_diff_colors (bag : Bag) (method : SamplingMethod) : ℚ :=
  sorry

/-- The expectation of the number of white balls drawn without replacement -/
def expectation_white (bag : Bag) : ℚ :=
  sorry

/-- The variance of the number of white balls drawn without replacement -/
def variance_white (bag : Bag) : ℚ :=
  sorry

/-- The main theorem to prove -/
theorem ball_sampling_theorem (bag : Bag) :
  bag.white = 2 ∧ bag.black = 3 →
  prob_diff_colors bag SamplingMethod.WithReplacement = 12/25 ∧
  expectation_white bag = 4/5 ∧
  variance_white bag = 9/25 :=
sorry

end ball_sampling_theorem_l781_78101


namespace largest_package_size_l781_78159

theorem largest_package_size (alex bella carlos : ℕ) 
  (h_alex : alex = 36)
  (h_bella : bella = 48)
  (h_carlos : carlos = 60) :
  Nat.gcd alex (Nat.gcd bella carlos) = 12 := by
  sorry

end largest_package_size_l781_78159


namespace fixed_root_quadratic_l781_78193

theorem fixed_root_quadratic (k : ℝ) : 
  ∃ x : ℝ, x^2 + (k + 3) * x + k + 2 = 0 ∧ x = -1 := by
  sorry

end fixed_root_quadratic_l781_78193


namespace ellen_legos_l781_78140

/-- The number of legos Ellen lost -/
def lost_legos : ℕ := 57

/-- The number of legos Ellen currently has -/
def current_legos : ℕ := 323

/-- The initial number of legos Ellen had -/
def initial_legos : ℕ := lost_legos + current_legos

theorem ellen_legos : initial_legos = 380 := by
  sorry

end ellen_legos_l781_78140


namespace sin_sum_identity_l781_78172

theorem sin_sum_identity : 
  Real.sin (π/4) * Real.sin (7*π/12) + Real.sin (π/4) * Real.sin (π/12) = Real.sqrt 3 / 2 := by
  sorry

end sin_sum_identity_l781_78172


namespace smallest_p_for_integer_sqrt_l781_78134

theorem smallest_p_for_integer_sqrt : ∃ (p : ℕ), p > 0 ∧ 
  (∀ (q : ℕ), q > 0 → q < p → ¬ (∃ (n : ℕ), n ^ 2 = 2^3 * 5 * q)) ∧
  (∃ (n : ℕ), n ^ 2 = 2^3 * 5 * p) ∧
  p = 10 := by
sorry

end smallest_p_for_integer_sqrt_l781_78134


namespace odd_function_property_positive_x_property_negative_x_property_l781_78175

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x =>
  if x > 0 then x + Real.log x else x - Real.log (-x)

-- State the theorem
theorem odd_function_property (x : ℝ) : f (-x) = -f x := by sorry

-- State the positive x property
theorem positive_x_property (x : ℝ) (h : x > 0) : f x = x + Real.log x := by sorry

-- State the negative x property
theorem negative_x_property (x : ℝ) (h : x < 0) : f x = x - Real.log (-x) := by sorry

end odd_function_property_positive_x_property_negative_x_property_l781_78175


namespace factor_cubic_l781_78137

theorem factor_cubic (a b c : ℝ) : 
  (∀ x, x^3 - 12*x + 16 = (x + 4)*(a*x^2 + b*x + c)) → 
  a*x^2 + b*x + c = (x - 2)^2 := by
sorry

end factor_cubic_l781_78137


namespace degree_of_P_l781_78138

/-- The polynomial in question -/
def P (a b : ℚ) : ℚ := 2/3 * a * b^2 + 4/3 * a^3 * b + 1/3

/-- The degree of a polynomial -/
def polynomial_degree (p : ℚ → ℚ → ℚ) : ℕ :=
  sorry  -- Definition of polynomial degree

theorem degree_of_P : polynomial_degree P = 4 := by
  sorry

end degree_of_P_l781_78138
