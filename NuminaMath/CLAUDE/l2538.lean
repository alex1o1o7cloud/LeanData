import Mathlib

namespace NUMINAMATH_CALUDE_train_length_proof_l2538_253831

def train_problem (speed1 speed2 shorter_length clearing_time : ℝ) : Prop :=
  let relative_speed := (speed1 + speed2) * 1000 / 3600
  let total_distance := relative_speed * clearing_time
  let longer_length := total_distance - shorter_length
  longer_length = 164.9771230827526

theorem train_length_proof :
  train_problem 80 55 121 7.626056582140095 := by sorry

end NUMINAMATH_CALUDE_train_length_proof_l2538_253831


namespace NUMINAMATH_CALUDE_sum_lower_bound_l2538_253878

theorem sum_lower_bound (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : 1/a + 4/b = 1) :
  ∀ c : ℝ, c < 9 → a + b > c :=
by
  sorry

end NUMINAMATH_CALUDE_sum_lower_bound_l2538_253878


namespace NUMINAMATH_CALUDE_max_pairs_sum_l2538_253872

theorem max_pairs_sum (k : ℕ) 
  (a b : Fin k → ℕ) 
  (h1 : ∀ i : Fin k, a i < b i)
  (h2 : ∀ i : Fin k, a i ≤ 1500 ∧ b i ≤ 1500)
  (h3 : ∀ i j : Fin k, i ≠ j → a i ≠ a j ∧ a i ≠ b j ∧ b i ≠ a j ∧ b i ≠ b j)
  (h4 : ∀ i : Fin k, a i + b i ≤ 1500)
  (h5 : ∀ i j : Fin k, i ≠ j → a i + b i ≠ a j + b j) :
  k ≤ 599 :=
sorry

end NUMINAMATH_CALUDE_max_pairs_sum_l2538_253872


namespace NUMINAMATH_CALUDE_intersection_point_k_value_l2538_253809

/-- Given two lines -3x + y = k and 2x + y = 10 that intersect at x = -5, prove that k = 35 -/
theorem intersection_point_k_value :
  ∀ (x y k : ℝ),
  (-3 * x + y = k) →
  (2 * x + y = 10) →
  (x = -5) →
  (k = 35) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_k_value_l2538_253809


namespace NUMINAMATH_CALUDE_ruby_pizza_order_cost_l2538_253886

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

end NUMINAMATH_CALUDE_ruby_pizza_order_cost_l2538_253886


namespace NUMINAMATH_CALUDE_probability_of_27_l2538_253876

/-- Represents a die with numbered and blank faces -/
structure Die :=
  (total_faces : ℕ)
  (numbered_faces : ℕ)
  (min_number : ℕ)
  (max_number : ℕ)

/-- Calculates the number of ways to get a sum with two dice -/
def waysToGetSum (d1 d2 : Die) (sum : ℕ) : ℕ :=
  sorry

/-- Calculates the total number of possible outcomes when rolling two dice -/
def totalOutcomes (d1 d2 : Die) : ℕ :=
  d1.total_faces * d2.total_faces

/-- Theorem: Probability of rolling 27 with given dice is 3/100 -/
theorem probability_of_27 :
  let die1 : Die := ⟨20, 18, 1, 18⟩
  let die2 : Die := ⟨20, 17, 3, 20⟩
  (waysToGetSum die1 die2 27 : ℚ) / (totalOutcomes die1 die2 : ℚ) = 3 / 100 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_27_l2538_253876


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2538_253851

theorem fixed_point_of_exponential_function (a : ℝ) (h : a > 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x - 2) + 1
  f 2 = 2 := by
sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2538_253851


namespace NUMINAMATH_CALUDE_point_above_line_l2538_253816

/-- A point in the 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The line y = x -/
def line_y_eq_x : Set Point2D := {p : Point2D | p.y = p.x}

/-- The region above the line y = x -/
def region_above_line : Set Point2D := {p : Point2D | p.y > p.x}

/-- Theorem: Any point M(x, y) where y > x is located in the region above the line y = x -/
theorem point_above_line (M : Point2D) (h : M.y > M.x) : M ∈ region_above_line := by
  sorry

end NUMINAMATH_CALUDE_point_above_line_l2538_253816


namespace NUMINAMATH_CALUDE_probability_qualified_product_l2538_253818

/-- The proportion of the first batch in the total mix -/
def batch1_proportion : ℝ := 0.30

/-- The proportion of the second batch in the total mix -/
def batch2_proportion : ℝ := 0.70

/-- The defect rate of the first batch -/
def batch1_defect_rate : ℝ := 0.05

/-- The defect rate of the second batch -/
def batch2_defect_rate : ℝ := 0.04

/-- The probability of selecting a qualified product from the mixed batches -/
theorem probability_qualified_product : 
  batch1_proportion * (1 - batch1_defect_rate) + batch2_proportion * (1 - batch2_defect_rate) = 0.957 := by
  sorry

end NUMINAMATH_CALUDE_probability_qualified_product_l2538_253818


namespace NUMINAMATH_CALUDE_arithmetic_sequence_max_sum_l2538_253842

/-- The sum of the first n terms of an arithmetic sequence with a₁ = 23 and d = -2 -/
def S (n : ℕ+) : ℝ := -n.val^2 + 24 * n.val

/-- The maximum value of S(n) for positive integer n -/
def max_S : ℝ := 144

theorem arithmetic_sequence_max_sum :
  ∃ (n : ℕ+), S n = max_S ∧ ∀ (m : ℕ+), S m ≤ max_S := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_max_sum_l2538_253842


namespace NUMINAMATH_CALUDE_remainder_23_pow_2047_mod_17_l2538_253873

theorem remainder_23_pow_2047_mod_17 : 23^2047 % 17 = 11 := by
  sorry

end NUMINAMATH_CALUDE_remainder_23_pow_2047_mod_17_l2538_253873


namespace NUMINAMATH_CALUDE_fraction_problem_l2538_253850

theorem fraction_problem (x : ℝ) (f : ℝ) (h1 : x > 0) (h2 : x = 1/3) 
  (h3 : f * x = (16/216) * (1/x)) : f = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l2538_253850


namespace NUMINAMATH_CALUDE_square_of_sum_l2538_253841

theorem square_of_sum (x : ℝ) (h1 : x^2 - 49 ≥ 0) (h2 : x + 7 ≥ 0) :
  (7 - Real.sqrt (x^2 - 49) + Real.sqrt (x + 7))^2 =
  x^2 + x + 7 - 14 * Real.sqrt (x^2 - 49) - 14 * Real.sqrt (x + 7) + 2 * Real.sqrt (x^2 - 49) * Real.sqrt (x + 7) := by
  sorry

end NUMINAMATH_CALUDE_square_of_sum_l2538_253841


namespace NUMINAMATH_CALUDE_distinct_roots_range_reciprocal_roots_sum_squares_l2538_253821

-- Define the quadratic equation
def quadratic (x m : ℝ) : ℝ := x^2 - 3*x + m - 3

-- Theorem for the range of m
theorem distinct_roots_range (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic x₁ m = 0 ∧ quadratic x₂ m = 0) ↔ m < 21/4 :=
sorry

-- Theorem for the sum of squares of reciprocal roots
theorem reciprocal_roots_sum_squares (m : ℝ) (x₁ x₂ : ℝ) :
  quadratic x₁ m = 0 ∧ quadratic x₂ m = 0 ∧ x₁ * x₂ = 1 →
  x₁^2 + x₂^2 = 7 :=
sorry

end NUMINAMATH_CALUDE_distinct_roots_range_reciprocal_roots_sum_squares_l2538_253821


namespace NUMINAMATH_CALUDE_jerry_can_carry_l2538_253832

/-- Given the following conditions:
  * There are 28 cans to be recycled
  * The total time taken is 350 seconds
  * It takes 30 seconds to drain the cans
  * It takes 10 seconds to walk each way (to and from the sink/recycling bin)
  Prove that Jerry can carry 4 cans at once. -/
theorem jerry_can_carry (total_cans : ℕ) (total_time : ℕ) (drain_time : ℕ) (walk_time : ℕ) :
  total_cans = 28 →
  total_time = 350 →
  drain_time = 30 →
  walk_time = 10 →
  (total_time / (drain_time + 2 * walk_time) : ℚ) * (total_cans / (total_time / (drain_time + 2 * walk_time)) : ℚ) = 4 :=
by sorry

end NUMINAMATH_CALUDE_jerry_can_carry_l2538_253832


namespace NUMINAMATH_CALUDE_not_necessary_not_sufficient_neither_necessary_nor_sufficient_l2538_253896

/-- Two lines are parallel if they have the same slope and don't intersect. -/
def are_parallel (m : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ 
  ∀ (x y : ℝ), (m * x + 4 * y - 6 = 0) ↔ (k * (x + m * y - 3) = 0)

/-- m = 2 is not necessary for the lines to be parallel. -/
theorem not_necessary (m : ℝ) : 
  ∃ m', m' ≠ 2 ∧ are_parallel m' :=
sorry

/-- m = 2 is not sufficient for the lines to be parallel. -/
theorem not_sufficient : ¬(are_parallel 2) :=
sorry

/-- m = 2 is neither necessary nor sufficient for the lines to be parallel. -/
theorem neither_necessary_nor_sufficient : 
  (∃ m', m' ≠ 2 ∧ are_parallel m') ∧ ¬(are_parallel 2) :=
sorry

end NUMINAMATH_CALUDE_not_necessary_not_sufficient_neither_necessary_nor_sufficient_l2538_253896


namespace NUMINAMATH_CALUDE_abc_value_l2538_253866

theorem abc_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a * (b + c) = 165) (h2 : b * (c + a) = 195) (h3 : c * (a + b) = 180) :
  a * b * c = 15 * Real.sqrt 210 := by
sorry

end NUMINAMATH_CALUDE_abc_value_l2538_253866


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2538_253846

theorem polynomial_simplification (x : ℝ) :
  (2 * x^2 + 5 * x - 7) - (x^2 + 9 * x - 3) = x^2 - 4 * x - 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2538_253846


namespace NUMINAMATH_CALUDE_square_area_from_perimeter_l2538_253881

/-- The area of a square with perimeter 40 feet is 100 square feet -/
theorem square_area_from_perimeter :
  ∀ (s : ℝ), s > 0 → 4 * s = 40 → s^2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_perimeter_l2538_253881


namespace NUMINAMATH_CALUDE_bobs_weight_l2538_253813

theorem bobs_weight (jim_weight bob_weight : ℝ) 
  (h1 : jim_weight + bob_weight = 200)
  (h2 : bob_weight - jim_weight = bob_weight / 3) : 
  bob_weight = 120 := by
sorry

end NUMINAMATH_CALUDE_bobs_weight_l2538_253813


namespace NUMINAMATH_CALUDE_hockey_championship_points_l2538_253894

/-- Represents the number of points a team receives for winning a game. -/
def win_points : ℕ := 2

/-- Represents the number of games tied. -/
def games_tied : ℕ := 12

/-- Represents the number of games won. -/
def games_won : ℕ := games_tied + 12

/-- Represents the points received for a tie. -/
def tie_points : ℕ := 1

theorem hockey_championship_points :
  win_points * games_won + tie_points * games_tied = 60 :=
sorry

end NUMINAMATH_CALUDE_hockey_championship_points_l2538_253894


namespace NUMINAMATH_CALUDE_fusilli_to_penne_ratio_l2538_253830

/-- Given a survey of pasta preferences, prove the ratio of fusilli to penne preferences --/
theorem fusilli_to_penne_ratio :
  ∀ (total students_fusilli students_penne : ℕ),
  total = 800 →
  students_fusilli = 320 →
  students_penne = 160 →
  (students_fusilli : ℚ) / (students_penne : ℚ) = 2 := by
sorry

end NUMINAMATH_CALUDE_fusilli_to_penne_ratio_l2538_253830


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l2538_253807

theorem coefficient_x_squared_in_expansion :
  (Finset.range 6).sum (fun k => (Nat.choose 5 k : ℕ) * 2^k * (if k = 2 then 1 else 0)) = 40 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l2538_253807


namespace NUMINAMATH_CALUDE_arcsin_of_one_l2538_253874

theorem arcsin_of_one : Real.arcsin 1 = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_of_one_l2538_253874


namespace NUMINAMATH_CALUDE_k_of_h_10_l2538_253825

def h (x : ℝ) : ℝ := 4 * x - 5

def k (x : ℝ) : ℝ := 2 * x + 6

theorem k_of_h_10 : k (h 10) = 76 := by
  sorry

end NUMINAMATH_CALUDE_k_of_h_10_l2538_253825


namespace NUMINAMATH_CALUDE_prime_congruence_l2538_253889

theorem prime_congruence (p : ℕ) (hp : p.Prime) (hp_gt_5 : p > 5) :
  (∃ x : ℤ, (x^4 + x^3 + x^2 + x + 1) % p = 0) →
  p % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_congruence_l2538_253889


namespace NUMINAMATH_CALUDE_garden_area_increase_l2538_253833

/-- Proves that changing a 60-foot by 20-foot rectangular garden to a square garden 
    with the same perimeter results in an increase of 400 square feet in area. -/
theorem garden_area_increase : 
  let rectangle_length : ℝ := 60
  let rectangle_width : ℝ := 20
  let rectangle_area := rectangle_length * rectangle_width
  let perimeter := 2 * (rectangle_length + rectangle_width)
  let square_side := perimeter / 4
  let square_area := square_side * square_side
  square_area - rectangle_area = 400 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_increase_l2538_253833


namespace NUMINAMATH_CALUDE_overlap_length_l2538_253840

/-- Given a set of overlapping red segments, this theorem proves the length of each overlap. -/
theorem overlap_length (total_length : ℝ) (end_to_end : ℝ) (num_overlaps : ℕ) : 
  total_length = 98 →
  end_to_end = 83 →
  num_overlaps = 6 →
  (total_length - end_to_end) / num_overlaps = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_overlap_length_l2538_253840


namespace NUMINAMATH_CALUDE_assignment_time_ratio_l2538_253808

theorem assignment_time_ratio : 
  let total_time : ℕ := 120
  let first_part : ℕ := 25
  let third_part : ℕ := 45
  let second_part : ℕ := total_time - (first_part + third_part)
  (second_part : ℚ) / first_part = 2 := by
  sorry

end NUMINAMATH_CALUDE_assignment_time_ratio_l2538_253808


namespace NUMINAMATH_CALUDE_even_function_properties_l2538_253835

-- Define the function f
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x + 1

-- Define what it means for f to be even
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- State the theorem
theorem even_function_properties (b : ℝ) 
  (h : is_even_function (f b)) :
  (b = 0) ∧ 
  (Set.Ioo 1 2 = {x | f b (x - 1) < x}) :=
by sorry

end NUMINAMATH_CALUDE_even_function_properties_l2538_253835


namespace NUMINAMATH_CALUDE_triangle_side_calculation_l2538_253870

theorem triangle_side_calculation (a b : ℝ) (A B : ℝ) (h1 : a = 4) (h2 : A = π / 6) (h3 : B = π / 3) :
  b = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_calculation_l2538_253870


namespace NUMINAMATH_CALUDE_stating_prob_reach_heaven_l2538_253862

/-- A point in the 2D lattice -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- The starting point of the walk -/
def start : LatticePoint := ⟨1, 1⟩

/-- Predicate for heaven points -/
def is_heaven (p : LatticePoint) : Prop :=
  ∃ m n : ℤ, p.x = 6 * m ∧ p.y = 6 * n

/-- Predicate for hell points -/
def is_hell (p : LatticePoint) : Prop :=
  ∃ m n : ℤ, p.x = 6 * m + 3 ∧ p.y = 6 * n + 3

/-- The probability of reaching heaven -/
def prob_heaven : ℚ := 13 / 22

/-- 
Theorem stating that the probability of reaching heaven 
before hell in a random lattice walk starting from (1,1) is 13/22 
-/
theorem prob_reach_heaven : 
  prob_heaven = 13 / 22 :=
sorry

end NUMINAMATH_CALUDE_stating_prob_reach_heaven_l2538_253862


namespace NUMINAMATH_CALUDE_ratio_of_probabilities_l2538_253853

/-- The number of rational terms in the expansion -/
def rational_terms : ℕ := 5

/-- The number of irrational terms in the expansion -/
def irrational_terms : ℕ := 4

/-- The total number of terms in the expansion -/
def total_terms : ℕ := rational_terms + irrational_terms

/-- The probability of having rational terms adjacent -/
def p : ℚ := (Nat.factorial rational_terms * Nat.factorial rational_terms) / Nat.factorial total_terms

/-- The probability of having no two rational terms adjacent -/
def q : ℚ := (Nat.factorial irrational_terms * Nat.factorial rational_terms) / Nat.factorial total_terms

theorem ratio_of_probabilities : p / q = 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_probabilities_l2538_253853


namespace NUMINAMATH_CALUDE_deepak_current_age_l2538_253858

/-- Represents the ages of Rahul and Deepak -/
structure Ages where
  rahul : ℕ
  deepak : ℕ

/-- The condition that the ratio of Rahul's age to Deepak's age is 4:3 -/
def ratio_condition (ages : Ages) : Prop :=
  4 * ages.deepak = 3 * ages.rahul

/-- The condition that Rahul will be 26 years old in 6 years -/
def future_condition (ages : Ages) : Prop :=
  ages.rahul + 6 = 26

/-- The theorem stating Deepak's current age given the conditions -/
theorem deepak_current_age (ages : Ages) 
  (h1 : ratio_condition ages) 
  (h2 : future_condition ages) : 
  ages.deepak = 15 := by
  sorry

end NUMINAMATH_CALUDE_deepak_current_age_l2538_253858


namespace NUMINAMATH_CALUDE_untouched_produce_count_l2538_253893

/-- The number of untouched tomatoes and cucumbers after processing -/
def untouched_produce (tomato_plants : ℕ) (tomatoes_per_plant : ℕ) (cucumbers : ℕ) : ℕ :=
  let total_tomatoes := tomato_plants * tomatoes_per_plant
  let dried_tomatoes := (2 * total_tomatoes) / 3
  let remaining_tomatoes := total_tomatoes - dried_tomatoes
  let sauce_tomatoes := remaining_tomatoes / 2
  let untouched_tomatoes := remaining_tomatoes - sauce_tomatoes
  let pickled_cucumbers := cucumbers / 4
  let untouched_cucumbers := cucumbers - pickled_cucumbers
  untouched_tomatoes + untouched_cucumbers

/-- Theorem stating the number of untouched produce given the conditions -/
theorem untouched_produce_count :
  untouched_produce 50 15 25 = 143 := by
  sorry


end NUMINAMATH_CALUDE_untouched_produce_count_l2538_253893


namespace NUMINAMATH_CALUDE_pants_discount_percentage_l2538_253834

theorem pants_discount_percentage (cost : ℝ) (profit_percentage : ℝ) (marked_price : ℝ) :
  cost = 80 →
  profit_percentage = 0.3 →
  marked_price = 130 →
  let profit := cost * profit_percentage
  let selling_price := cost + profit
  let discount := marked_price - selling_price
  let discount_percentage := (discount / marked_price) * 100
  discount_percentage = 20 := by sorry

end NUMINAMATH_CALUDE_pants_discount_percentage_l2538_253834


namespace NUMINAMATH_CALUDE_town_street_lights_l2538_253898

/-- Calculates the total number of street lights in a town given the number of neighborhoods,
    roads per neighborhood, and street lights per side of each road. -/
def totalStreetLights (neighborhoods : ℕ) (roadsPerNeighborhood : ℕ) (lightsPerSide : ℕ) : ℕ :=
  neighborhoods * roadsPerNeighborhood * lightsPerSide * 2

/-- Theorem stating that the total number of street lights in the described town is 20000. -/
theorem town_street_lights :
  totalStreetLights 10 4 250 = 20000 := by
  sorry

end NUMINAMATH_CALUDE_town_street_lights_l2538_253898


namespace NUMINAMATH_CALUDE_savings_percentage_approx_l2538_253822

def rent : ℕ := 5000
def milk : ℕ := 1500
def groceries : ℕ := 4500
def education : ℕ := 2500
def petrol : ℕ := 2000
def miscellaneous : ℕ := 5650
def savings : ℕ := 2350

def total_expenses : ℕ := rent + milk + groceries + education + petrol + miscellaneous
def total_salary : ℕ := total_expenses + savings

def percentage_saved : ℚ := (savings : ℚ) / (total_salary : ℚ) * 100

theorem savings_percentage_approx :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ abs (percentage_saved - 8.87) < ε :=
sorry

end NUMINAMATH_CALUDE_savings_percentage_approx_l2538_253822


namespace NUMINAMATH_CALUDE_james_age_proof_l2538_253814

/-- James' age -/
def james_age : ℝ := 47.5

/-- Mara's age -/
def mara_age : ℝ := 22.5

/-- James' age is 20 years less than three times Mara's age -/
axiom age_relation : james_age = 3 * mara_age - 20

/-- The sum of their ages is 70 -/
axiom age_sum : james_age + mara_age = 70

theorem james_age_proof : james_age = 47.5 := by
  sorry

end NUMINAMATH_CALUDE_james_age_proof_l2538_253814


namespace NUMINAMATH_CALUDE_jake_read_225_pages_l2538_253892

/-- The number of pages Jake read in a week -/
def pages_read : ℕ :=
  let day1 : ℕ := 45
  let day2 : ℕ := day1 / 3
  let day3 : ℕ := 58 - 12
  let day4 : ℕ := (day1 + 1) / 2  -- Rounding up
  let day5 : ℕ := (3 * day3 + 3) / 4  -- Rounding up
  let day6 : ℕ := day2
  let day7 : ℕ := 2 * day4
  day1 + day2 + day3 + day4 + day5 + day6 + day7

/-- Theorem stating that Jake read 225 pages in total -/
theorem jake_read_225_pages : pages_read = 225 := by
  sorry

end NUMINAMATH_CALUDE_jake_read_225_pages_l2538_253892


namespace NUMINAMATH_CALUDE_square_sum_geq_neg_double_product_l2538_253810

theorem square_sum_geq_neg_double_product (a b : ℝ) : a^2 + b^2 ≥ -2*a*b := by
  sorry

end NUMINAMATH_CALUDE_square_sum_geq_neg_double_product_l2538_253810


namespace NUMINAMATH_CALUDE_only_solution_l2538_253829

/-- Represents the position of a person in a line of recruits. -/
structure Position :=
  (ahead : ℕ)

/-- Represents the three brothers in the line of recruits. -/
inductive Brother
| Peter
| Nikolay
| Denis

/-- Gets the initial number of people ahead of a given brother. -/
def initial_ahead (b : Brother) : ℕ :=
  match b with
  | Brother.Peter => 50
  | Brother.Nikolay => 100
  | Brother.Denis => 170

/-- Calculates the number of people in front after turning, given the total number of recruits. -/
def after_turn (n : ℕ) (b : Brother) : ℕ :=
  n - (initial_ahead b + 1)

/-- Checks if the condition after turning is satisfied for a given total number of recruits. -/
def satisfies_condition (n : ℕ) : Prop :=
  ∃ (b1 b2 : Brother), b1 ≠ b2 ∧ after_turn n b1 = 4 * after_turn n b2

/-- The theorem stating that 211 is the only solution. -/
theorem only_solution :
  ∀ n : ℕ, satisfies_condition n ↔ n = 211 :=
sorry

end NUMINAMATH_CALUDE_only_solution_l2538_253829


namespace NUMINAMATH_CALUDE_exam_score_calculation_l2538_253856

/-- Calculate total marks in an exam with penalties for incorrect answers -/
theorem exam_score_calculation 
  (total_questions : ℕ) 
  (correct_answers : ℕ) 
  (marks_per_correct : ℕ) 
  (penalty_per_wrong : ℕ) :
  total_questions = 60 →
  correct_answers = 36 →
  marks_per_correct = 4 →
  penalty_per_wrong = 1 →
  (correct_answers * marks_per_correct) - 
  ((total_questions - correct_answers) * penalty_per_wrong) = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_exam_score_calculation_l2538_253856


namespace NUMINAMATH_CALUDE_daisies_given_away_l2538_253843

/-- Proves the number of daisies given away based on initial count, petals per daisy, and remaining petals --/
theorem daisies_given_away 
  (initial_daisies : ℕ) 
  (petals_per_daisy : ℕ) 
  (remaining_petals : ℕ) 
  (h1 : initial_daisies = 5)
  (h2 : petals_per_daisy = 8)
  (h3 : remaining_petals = 24) :
  initial_daisies - (remaining_petals / petals_per_daisy) = 2 :=
by
  sorry

#check daisies_given_away

end NUMINAMATH_CALUDE_daisies_given_away_l2538_253843


namespace NUMINAMATH_CALUDE_mod_equivalence_l2538_253817

theorem mod_equivalence (m : ℕ) : 
  152 * 936 ≡ m [ZMOD 50] → 0 ≤ m → m < 50 → m = 22 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_l2538_253817


namespace NUMINAMATH_CALUDE_swim_team_girls_count_l2538_253871

theorem swim_team_girls_count :
  ∀ (boys girls coaches managers : ℕ),
  girls = 5 * boys →
  coaches = 4 →
  managers = 4 →
  boys + girls + coaches + managers = 104 →
  girls = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_swim_team_girls_count_l2538_253871


namespace NUMINAMATH_CALUDE_first_bakery_sacks_proof_l2538_253802

/-- The number of sacks the second bakery needs per week -/
def second_bakery_sacks : ℕ := 4

/-- The number of sacks the third bakery needs per week -/
def third_bakery_sacks : ℕ := 12

/-- The total number of weeks -/
def total_weeks : ℕ := 4

/-- The total number of sacks needed for all bakeries in 4 weeks -/
def total_sacks : ℕ := 72

/-- The number of sacks the first bakery needs per week -/
def first_bakery_sacks : ℕ := 2

theorem first_bakery_sacks_proof :
  first_bakery_sacks * total_weeks + 
  second_bakery_sacks * total_weeks + 
  third_bakery_sacks * total_weeks = total_sacks :=
by sorry

end NUMINAMATH_CALUDE_first_bakery_sacks_proof_l2538_253802


namespace NUMINAMATH_CALUDE_B_pow_100_l2538_253869

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  !![0, 1, 0;
     0, 0, 1;
     1, 0, 0]

theorem B_pow_100 : B^100 = B := by
  sorry

end NUMINAMATH_CALUDE_B_pow_100_l2538_253869


namespace NUMINAMATH_CALUDE_greatest_solution_sin_cos_equation_l2538_253823

theorem greatest_solution_sin_cos_equation :
  ∃ (x : ℝ),
    x ∈ Set.Icc 0 (10 * Real.pi) ∧
    |2 * Real.sin x - 1| + |2 * Real.cos (2 * x) - 1| = 0 ∧
    (∀ (y : ℝ), y ∈ Set.Icc 0 (10 * Real.pi) →
      |2 * Real.sin y - 1| + |2 * Real.cos (2 * y) - 1| = 0 → y ≤ x) ∧
    x = 61 * Real.pi / 6 :=
by sorry

end NUMINAMATH_CALUDE_greatest_solution_sin_cos_equation_l2538_253823


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2538_253885

theorem necessary_but_not_sufficient :
  (∀ a b : ℝ, a > b ∧ b > 0 → a / b > 1) ∧
  (∃ a b : ℝ, a / b > 1 ∧ ¬(a > b ∧ b > 0)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2538_253885


namespace NUMINAMATH_CALUDE_expression_simplification_l2538_253868

theorem expression_simplification (x y z : ℝ) :
  (x - (2 * y + z)) - ((x + 2 * y) - 3 * z) = -4 * y + 2 * z := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2538_253868


namespace NUMINAMATH_CALUDE_coefficient_x5_in_expansion_l2538_253852

theorem coefficient_x5_in_expansion : 
  (Finset.range 61).sum (fun k => Nat.choose 60 k * (1 : ℕ)^(60 - k) * (1 : ℕ)^k) = 2^60 ∧ 
  Nat.choose 60 5 = 446040 :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x5_in_expansion_l2538_253852


namespace NUMINAMATH_CALUDE_total_sweets_l2538_253812

theorem total_sweets (num_crates : ℕ) (sweets_per_crate : ℕ) 
  (h1 : num_crates = 4) 
  (h2 : sweets_per_crate = 16) : 
  num_crates * sweets_per_crate = 64 := by
  sorry

end NUMINAMATH_CALUDE_total_sweets_l2538_253812


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l2538_253863

theorem sqrt_meaningful_range (a : ℝ) : (∃ (x : ℝ), x^2 = 2 + a) ↔ a ≥ -2 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l2538_253863


namespace NUMINAMATH_CALUDE_polygon_sides_l2538_253890

theorem polygon_sides (sum_interior_angles : ℝ) : sum_interior_angles = 540 → ∃ n : ℕ, n = 5 ∧ (n - 2) * 180 = sum_interior_angles :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l2538_253890


namespace NUMINAMATH_CALUDE_cubic_fraction_factorization_l2538_253838

theorem cubic_fraction_factorization (a b c : ℝ) :
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3) 
  = (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) :=
by sorry

end NUMINAMATH_CALUDE_cubic_fraction_factorization_l2538_253838


namespace NUMINAMATH_CALUDE_least_three_digit_with_digit_product_18_l2538_253899

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_product (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * tens * ones

theorem least_three_digit_with_digit_product_18 :
  ∃ (n : ℕ), is_three_digit n ∧ digit_product n = 18 ∧
  ∀ (m : ℕ), is_three_digit m → digit_product m = 18 → n ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_least_three_digit_with_digit_product_18_l2538_253899


namespace NUMINAMATH_CALUDE_distance_between_points_l2538_253888

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (2, 6)
  let p2 : ℝ × ℝ := (5, 2)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 5 := by
sorry

end NUMINAMATH_CALUDE_distance_between_points_l2538_253888


namespace NUMINAMATH_CALUDE_x_range_l2538_253804

def p (x : ℝ) : Prop := x^2 + 2*x - 3 > 0

def q (x : ℝ) : Prop := 1 / (3 - x) > 1

theorem x_range (x : ℝ) (h1 : p x) (h2 : ¬(q x)) : 
  x ≥ 3 ∨ (1 < x ∧ x ≤ 2) ∨ x < -3 := by
  sorry

end NUMINAMATH_CALUDE_x_range_l2538_253804


namespace NUMINAMATH_CALUDE_polynomial_intersection_l2538_253861

-- Define the polynomials f and g
def f (a b x : ℝ) : ℝ := x^2 + a*x + b
def g (c d x : ℝ) : ℝ := x^2 + c*x + d

-- Define the theorem
theorem polynomial_intersection (a b c d : ℝ) : 
  -- f and g are distinct
  (∃ x, f a b x ≠ g c d x) →
  -- The x-coordinate of the vertex of f is a root of g
  g c d (-a/2) = 0 →
  -- The x-coordinate of the vertex of g is a root of f
  f a b (-c/2) = 0 →
  -- Both f and g have the same minimum value
  (b - a^2/4 = d - c^2/4) →
  -- The graphs of f and g intersect at the point (2012, -2012)
  f a b 2012 = -2012 ∧ g c d 2012 = -2012 →
  -- Conclusion
  a + c = -8048 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_intersection_l2538_253861


namespace NUMINAMATH_CALUDE_eighth_grade_gpa_l2538_253880

/-- Proves that the average GPA for 8th graders is 91 given the specified conditions -/
theorem eighth_grade_gpa (sixth_grade_gpa seventh_grade_gpa eighth_grade_gpa school_avg_gpa : ℝ) :
  sixth_grade_gpa = 93 →
  seventh_grade_gpa = sixth_grade_gpa + 2 →
  school_avg_gpa = 93 →
  school_avg_gpa = (sixth_grade_gpa + seventh_grade_gpa + eighth_grade_gpa) / 3 →
  eighth_grade_gpa = 91 := by
  sorry

end NUMINAMATH_CALUDE_eighth_grade_gpa_l2538_253880


namespace NUMINAMATH_CALUDE_binomial_probability_example_l2538_253855

/-- The probability mass function for a binomial distribution -/
def binomial_pmf (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

/-- Theorem: For X ~ B(5, 1/3), P(X=2) = C_5^2 (1/3)^2 × (2/3)^3 -/
theorem binomial_probability_example :
  binomial_pmf 5 (1/3) 2 = (Nat.choose 5 2 : ℝ) * (1/3)^2 * (2/3)^3 := by
  sorry

end NUMINAMATH_CALUDE_binomial_probability_example_l2538_253855


namespace NUMINAMATH_CALUDE_inscribed_square_area_largest_inscribed_square_area_l2538_253864

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

end NUMINAMATH_CALUDE_inscribed_square_area_largest_inscribed_square_area_l2538_253864


namespace NUMINAMATH_CALUDE_pizza_theorem_l2538_253806

/-- Calculates the total number of pizza slices brought by friends -/
def totalPizzaSlices (numFriends : ℕ) (slicesPerFriend : ℕ) : ℕ :=
  numFriends * slicesPerFriend

/-- Theorem: Given 4 friends, each bringing 4 slices of pizza, the total number of pizza slices is 16 -/
theorem pizza_theorem : totalPizzaSlices 4 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_pizza_theorem_l2538_253806


namespace NUMINAMATH_CALUDE_bananas_needed_l2538_253820

def yogurt_count : ℕ := 5
def slices_per_yogurt : ℕ := 8
def slices_per_banana : ℕ := 10

theorem bananas_needed : 
  (yogurt_count * slices_per_yogurt + slices_per_banana - 1) / slices_per_banana = 4 := by
  sorry

end NUMINAMATH_CALUDE_bananas_needed_l2538_253820


namespace NUMINAMATH_CALUDE_job_completion_solution_l2538_253891

/-- Represents the time taken by machines to complete a job -/
def job_completion_time (x : ℝ) : Prop :=
  let p_alone := x + 5
  let q_alone := x + 2
  let r_alone := 2 * x
  let pq_together := x + 3
  (1 / p_alone + 1 / q_alone + 1 / r_alone = 1 / x) ∧
  (1 / p_alone + 1 / q_alone = 1 / pq_together)

/-- Theorem stating that x = 2 satisfies the job completion time conditions -/
theorem job_completion_solution : job_completion_time 2 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_solution_l2538_253891


namespace NUMINAMATH_CALUDE_floor_length_l2538_253887

/-- Represents the dimensions of a rectangular floor -/
structure FloorDimensions where
  breadth : ℝ
  length : ℝ

/-- The properties of the floor as given in the problem -/
def FloorProperties (d : FloorDimensions) : Prop :=
  d.length = 3 * d.breadth ∧ d.length * d.breadth = 156

/-- Theorem stating the length of the floor -/
theorem floor_length (d : FloorDimensions) (h : FloorProperties d) : 
  d.length = 6 * Real.sqrt 13 := by
  sorry


end NUMINAMATH_CALUDE_floor_length_l2538_253887


namespace NUMINAMATH_CALUDE_tuesday_attendance_theorem_l2538_253845

/-- Represents the attendance status of students at Dunkley S.S. over two days -/
structure AttendanceData where
  total_students : ℕ
  monday_absent_rate : ℚ
  tuesday_return_rate : ℚ
  tuesday_absent_rate : ℚ

/-- Calculates the percentage of students present on Tuesday -/
def tuesday_present_percentage (data : AttendanceData) : ℚ :=
  let monday_present := 1 - data.monday_absent_rate
  let tuesday_present_from_monday := monday_present * (1 - data.tuesday_absent_rate)
  let tuesday_present_from_absent := data.monday_absent_rate * data.tuesday_return_rate
  (tuesday_present_from_monday + tuesday_present_from_absent) * 100

/-- Theorem stating that given the conditions, the percentage of students present on Tuesday is 82% -/
theorem tuesday_attendance_theorem (data : AttendanceData) 
  (h1 : data.total_students > 0)
  (h2 : data.monday_absent_rate = 1/10)
  (h3 : data.tuesday_return_rate = 1/10)
  (h4 : data.tuesday_absent_rate = 1/10) :
  tuesday_present_percentage data = 82 := by
  sorry


end NUMINAMATH_CALUDE_tuesday_attendance_theorem_l2538_253845


namespace NUMINAMATH_CALUDE_seth_oranges_l2538_253897

theorem seth_oranges (initial_boxes : ℕ) : 
  (initial_boxes - 1) / 2 = 4 → initial_boxes = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_seth_oranges_l2538_253897


namespace NUMINAMATH_CALUDE_choose_four_from_ten_l2538_253849

theorem choose_four_from_ten : Nat.choose 10 4 = 210 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_from_ten_l2538_253849


namespace NUMINAMATH_CALUDE_intersection_properties_l2538_253882

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

end NUMINAMATH_CALUDE_intersection_properties_l2538_253882


namespace NUMINAMATH_CALUDE_competition_results_l2538_253811

/-- Represents the final scores of competitors -/
structure Scores where
  A : ℚ
  B : ℚ
  C : ℚ

/-- Represents the points awarded for each position -/
structure PointSystem where
  first : ℚ
  second : ℚ
  third : ℚ

/-- Represents the number of times each competitor finished in each position -/
structure CompetitorResults where
  first : ℕ
  second : ℕ
  third : ℕ

/-- The main theorem statement -/
theorem competition_results 
  (scores : Scores)
  (points : PointSystem)
  (A_results B_results C_results : CompetitorResults)
  (h_scores : scores.A = 22 ∧ scores.B = 9 ∧ scores.C = 9)
  (h_B_won_100m : B_results.first ≥ 1)
  (h_no_ties : ∀ event : ℕ, 
    A_results.first + B_results.first + C_results.first = event ∧
    A_results.second + B_results.second + C_results.second = event ∧
    A_results.third + B_results.third + C_results.third = event)
  (h_score_calculation : 
    scores.A = A_results.first * points.first + A_results.second * points.second + A_results.third * points.third ∧
    scores.B = B_results.first * points.first + B_results.second * points.second + B_results.third * points.third ∧
    scores.C = C_results.first * points.first + C_results.second * points.second + C_results.third * points.third)
  : (A_results.first + A_results.second + A_results.third = 4) ∧ 
    (B_results.first + B_results.second + B_results.third = 4) ∧ 
    (C_results.first + C_results.second + C_results.third = 4) ∧
    A_results.second ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_competition_results_l2538_253811


namespace NUMINAMATH_CALUDE_total_sandcastles_and_towers_l2538_253848

/-- The number of sandcastles on Mark's beach -/
def marks_castles : ℕ := 20

/-- The number of towers per sandcastle on Mark's beach -/
def marks_towers_per_castle : ℕ := 10

/-- The ratio of Jeff's sandcastles to Mark's sandcastles -/
def jeff_castle_ratio : ℕ := 3

/-- The number of towers per sandcastle on Jeff's beach -/
def jeffs_towers_per_castle : ℕ := 5

/-- Theorem stating the combined total number of sandcastles and towers on both beaches -/
theorem total_sandcastles_and_towers :
  marks_castles * marks_towers_per_castle +
  marks_castles +
  (jeff_castle_ratio * marks_castles) * jeffs_towers_per_castle +
  (jeff_castle_ratio * marks_castles) = 580 := by
  sorry

end NUMINAMATH_CALUDE_total_sandcastles_and_towers_l2538_253848


namespace NUMINAMATH_CALUDE_negation_of_existential_proposition_l2538_253865

theorem negation_of_existential_proposition :
  (¬ ∃ x : ℝ, abs x > 0) ↔ (∀ x : ℝ, ¬(abs x > 0)) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existential_proposition_l2538_253865


namespace NUMINAMATH_CALUDE_arithmetic_sequence_m_value_l2538_253859

/-- Given an arithmetic sequence {a_n} with sum of first n terms S_n -/
def arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

/-- The theorem stating that if S_{m-1} = -3, S_m = 0, and S_{m+1} = 5, then m = 4 -/
theorem arithmetic_sequence_m_value
  (a : ℕ → ℝ) (S : ℕ → ℝ) (m : ℕ)
  (h_arithmetic : arithmetic_sequence a S)
  (h_m_minus_1 : S (m - 1) = -3)
  (h_m : S m = 0)
  (h_m_plus_1 : S (m + 1) = 5) :
  m = 4 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_m_value_l2538_253859


namespace NUMINAMATH_CALUDE_rectangle_length_l2538_253826

/-- Represents the properties of a rectangle --/
structure Rectangle where
  width : ℝ
  length : ℝ
  perimeter : ℝ
  length_width_relation : length = width + 15
  perimeter_formula : perimeter = 2 * length + 2 * width

/-- Theorem stating that a rectangle with the given properties has a length of 45 cm --/
theorem rectangle_length (rect : Rectangle) (h : rect.perimeter = 150) : rect.length = 45 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_l2538_253826


namespace NUMINAMATH_CALUDE_computer_room_arrangements_l2538_253839

/-- The number of different computer rooms -/
def n : ℕ := 6

/-- The minimum number of rooms that must be open -/
def k : ℕ := 2

/-- The number of arrangements for opening at least k out of n rooms -/
def num_arrangements (n k : ℕ) : ℕ := sorry

/-- Sum of combinations for opening 3 to 6 rooms, with 4 rooms counted twice -/
def sum_combinations (n : ℕ) : ℕ := 
  Nat.choose n 3 + 2 * Nat.choose n 4 + Nat.choose n 5 + Nat.choose n 6

/-- Total arrangements minus arrangements for 0 and 1 room -/
def power_minus_seven (n : ℕ) : ℕ := 2^n - 7

theorem computer_room_arrangements :
  num_arrangements n k = sum_combinations n ∧ 
  num_arrangements n k = power_minus_seven n := by sorry

end NUMINAMATH_CALUDE_computer_room_arrangements_l2538_253839


namespace NUMINAMATH_CALUDE_orchids_cut_count_l2538_253828

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

end NUMINAMATH_CALUDE_orchids_cut_count_l2538_253828


namespace NUMINAMATH_CALUDE_triangular_projections_imply_triangular_pyramid_l2538_253844

/-- Represents the shape of a projection in an orthographic view -/
inductive Projection
  | Triangular
  | Circular
  | Rectangular
  | Trapezoidal

/-- Represents a geometric solid -/
inductive GeometricSolid
  | Cone
  | TriangularPyramid
  | TriangularPrism
  | FrustumOfPyramid

/-- Represents the orthographic views of a solid -/
structure OrthographicViews where
  front : Projection
  top : Projection
  side : Projection

/-- Determines if a set of orthographic views corresponds to a triangular pyramid -/
def isTriangularPyramid (views : OrthographicViews) : Prop :=
  views.front = Projection.Triangular ∧
  views.top = Projection.Triangular ∧
  views.side = Projection.Triangular

theorem triangular_projections_imply_triangular_pyramid (views : OrthographicViews) :
  isTriangularPyramid views → GeometricSolid.TriangularPyramid = 
    match views with
    | ⟨Projection.Triangular, Projection.Triangular, Projection.Triangular⟩ => GeometricSolid.TriangularPyramid
    | _ => GeometricSolid.Cone  -- This is just a placeholder for other cases
    :=
  sorry

end NUMINAMATH_CALUDE_triangular_projections_imply_triangular_pyramid_l2538_253844


namespace NUMINAMATH_CALUDE_smallest_exceeding_day_l2538_253854

def tea_intake (n : ℕ) : ℚ := (n * (n + 1) * (n + 2)) / 3

theorem smallest_exceeding_day : 
  (∀ k < 13, tea_intake k ≤ 900) ∧ tea_intake 13 > 900 := by sorry

end NUMINAMATH_CALUDE_smallest_exceeding_day_l2538_253854


namespace NUMINAMATH_CALUDE_roots_of_f_minus_x_and_f_of_f_minus_x_l2538_253819

def f (x : ℝ) : ℝ := x^2 - 2*x + 2

theorem roots_of_f_minus_x_and_f_of_f_minus_x :
  (∀ x : ℝ, f x - x = 0 ↔ x = 1 ∨ x = 2) ∧
  (∀ x : ℝ, f (f x) - x = 0 ↔ x = 1 ∨ x = 2) := by
  sorry

end NUMINAMATH_CALUDE_roots_of_f_minus_x_and_f_of_f_minus_x_l2538_253819


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l2538_253884

theorem quadratic_roots_property (m : ℝ) (r s : ℝ) : 
  (∀ x, x^2 - (m+1)*x + m = 0 ↔ x = r ∨ x = s) →
  |r + s - 2*r*s| = |1 - m| := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l2538_253884


namespace NUMINAMATH_CALUDE_share_difference_l2538_253877

/-- Represents the share of money for each person -/
structure Share :=
  (amount : ℕ)

/-- Represents the distribution of money among three people -/
structure Distribution :=
  (faruk : Share)
  (vasim : Share)
  (ranjith : Share)

/-- Defines the ratio of distribution -/
def distribution_ratio : Distribution → (ℕ × ℕ × ℕ)
  | ⟨f, v, r⟩ => (3, 5, 8)

theorem share_difference (d : Distribution) :
  distribution_ratio d = (3, 5, 8) →
  d.vasim.amount = 1500 →
  d.ranjith.amount - d.faruk.amount = 1500 :=
by sorry

end NUMINAMATH_CALUDE_share_difference_l2538_253877


namespace NUMINAMATH_CALUDE_vector_decomposition_l2538_253883

/-- Given vectors in ℝ³ -/
def x : Fin 3 → ℝ := ![13, 2, 7]
def p : Fin 3 → ℝ := ![5, 1, 0]
def q : Fin 3 → ℝ := ![2, -1, 3]
def r : Fin 3 → ℝ := ![1, 0, -1]

/-- Theorem stating the decomposition of x in terms of p, q, and r -/
theorem vector_decomposition :
  x = fun i => 3 * p i + q i - 4 * r i := by
  sorry

end NUMINAMATH_CALUDE_vector_decomposition_l2538_253883


namespace NUMINAMATH_CALUDE_inequality_proof_l2538_253860

theorem inequality_proof (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_sum : a*b + b*c + c*d + d*a = 1) : 
  a^3 / (b+c+d) + b^3 / (a+c+d) + c^3 / (a+b+d) + d^3 / (a+b+c) ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2538_253860


namespace NUMINAMATH_CALUDE_shaniqua_styles_l2538_253847

def haircut_price : ℕ := 12
def style_price : ℕ := 25
def total_earned : ℕ := 221
def haircuts_given : ℕ := 8

theorem shaniqua_styles (styles : ℕ) : styles = 5 := by
  sorry

end NUMINAMATH_CALUDE_shaniqua_styles_l2538_253847


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2538_253815

/-- The repeating decimal 0.565656... expressed as a rational number -/
def repeating_decimal : ℚ := 0.56565656

/-- The fraction 56/99 -/
def fraction : ℚ := 56 / 99

/-- Theorem stating that the repeating decimal 0.565656... is equal to the fraction 56/99 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = fraction := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2538_253815


namespace NUMINAMATH_CALUDE_y_bounds_for_n_4_l2538_253875

/-- The function y(t) = (n-1)t² - 10t + 10 -/
def y (n : ℕ) (t : ℝ) : ℝ := (n - 1) * t^2 - 10 * t + 10

/-- The theorem stating that for n = 4, y(t) is always between 0 and 30 for t in (0,4] -/
theorem y_bounds_for_n_4 :
  ∀ t : ℝ, t > 0 → t ≤ 4 → 0 < y 4 t ∧ y 4 t ≤ 30 := by sorry

end NUMINAMATH_CALUDE_y_bounds_for_n_4_l2538_253875


namespace NUMINAMATH_CALUDE_find_number_l2538_253867

theorem find_number : ∃ x : ℝ, 0.45 * x - 85 = 10 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l2538_253867


namespace NUMINAMATH_CALUDE_product_divisible_by_seven_l2538_253857

theorem product_divisible_by_seven :
  (7 * 17 * 27 * 37 * 47 * 57 * 67) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_divisible_by_seven_l2538_253857


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2538_253879

/-- Triangle ABC with vertices A(0,2), B(2,0), and C(-2,-1) -/
structure Triangle where
  A : Prod ℝ ℝ := (0, 2)
  B : Prod ℝ ℝ := (2, 0)
  C : Prod ℝ ℝ := (-2, -1)

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem about the properties of triangle ABC -/
theorem triangle_abc_properties (t : Triangle) :
  ∃ (l : LineEquation) (area : ℝ),
    -- The line equation of height AH
    (l.a = 4 ∧ l.b = 1 ∧ l.c = -2) ∧
    -- The area of triangle ABC
    area = 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l2538_253879


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l2538_253801

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, real axis length 2, and focal distance 4,
    prove that its asymptotes are y = ±√3 x -/
theorem hyperbola_asymptotes (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (2 * a = 2) →  -- real axis length is 2
  (4 = 2 * Real.sqrt (a^2 + b^2)) →  -- focal distance is 4
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ (y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l2538_253801


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l2538_253803

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def second_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- The point (-1,2) -/
def point : Point :=
  { x := -1, y := 2 }

theorem point_in_second_quadrant : second_quadrant point := by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l2538_253803


namespace NUMINAMATH_CALUDE_sector_central_angle_l2538_253800

/-- Given a circular sector with perimeter 8 and area 4, prove that its central angle is 2 radians -/
theorem sector_central_angle (r : ℝ) (α : ℝ) : 
  r + r + r * α = 8 → -- perimeter condition
  (1/2) * r^2 * α = 4 → -- area condition
  α = 2 := by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l2538_253800


namespace NUMINAMATH_CALUDE_largest_cosine_in_geometric_triangle_l2538_253827

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

end NUMINAMATH_CALUDE_largest_cosine_in_geometric_triangle_l2538_253827


namespace NUMINAMATH_CALUDE_lesser_solution_quadratic_l2538_253895

theorem lesser_solution_quadratic (x : ℝ) : 
  x^2 + 10*x - 24 = 0 ∧ (∀ y : ℝ, y^2 + 10*y - 24 = 0 → x ≤ y) → x = -12 := by
  sorry

end NUMINAMATH_CALUDE_lesser_solution_quadratic_l2538_253895


namespace NUMINAMATH_CALUDE_tens_digit_of_19_pow_2023_l2538_253836

theorem tens_digit_of_19_pow_2023 : ∃ n : ℕ, 19^2023 ≡ 10 * n + 5 [ZMOD 100] :=
by sorry

end NUMINAMATH_CALUDE_tens_digit_of_19_pow_2023_l2538_253836


namespace NUMINAMATH_CALUDE_circle_M_equation_l2538_253837

/-- The equation of a line passing through the center of circle M -/
def center_line (x y : ℝ) : Prop := x - y - 4 = 0

/-- The equation of the first given circle -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 4 = 0

/-- The equation of the second given circle -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 6*y - 28 = 0

/-- The equation of circle M -/
def circle_M (x y : ℝ) : Prop := (x - 1/2)^2 + (y + 7/2)^2 = 89/2

/-- Theorem stating that the given conditions imply the equation of circle M -/
theorem circle_M_equation (x y : ℝ) :
  (∃ (xc yc : ℝ), center_line xc yc ∧ 
    (∀ (xi yi : ℝ), (circle1 xi yi ∧ circle2 xi yi) → 
      (x - xc)^2 + (y - yc)^2 = (xi - xc)^2 + (yi - yc)^2)) →
  circle_M x y :=
sorry

end NUMINAMATH_CALUDE_circle_M_equation_l2538_253837


namespace NUMINAMATH_CALUDE_simplify_expression_l2538_253805

theorem simplify_expression : 
  2 - (1 / (2 + Real.sqrt 5)) + (1 / (2 - Real.sqrt 5)) = 2 - 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2538_253805


namespace NUMINAMATH_CALUDE_smallest_b_value_l2538_253824

theorem smallest_b_value (a b : ℕ+) (h1 : a - b = 8) 
  (h2 : Nat.gcd ((a^3 + b^3) / (a + b)) (a * b) = 16) : 
  ∀ c : ℕ+, c < b → ¬(∃ d : ℕ+, d - c = 8 ∧ 
    Nat.gcd ((d^3 + c^3) / (d + c)) (d * c) = 16) :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_value_l2538_253824
