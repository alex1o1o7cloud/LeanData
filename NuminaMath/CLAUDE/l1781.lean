import Mathlib

namespace NUMINAMATH_CALUDE_appropriate_sampling_methods_l1781_178170

/-- Represents a city with its number of sales outlets -/
structure City where
  name : String
  outlets : ℕ

/-- Represents a sampling method -/
inductive SamplingMethod
  | StratifiedSampling
  | SystematicSampling
  | SimpleRandomSampling

/-- Represents an investigation with its requirements -/
structure Investigation where
  id : ℕ
  totalOutlets : ℕ
  sampleSize : ℕ
  cities : List City

/-- Determines the most appropriate sampling method for an investigation -/
def mostAppropriateMethod (inv : Investigation) : SamplingMethod := sorry

/-- The main theorem stating the appropriate sampling methods for the given investigations -/
theorem appropriate_sampling_methods 
  (cityA : City)
  (cityB : City)
  (cityC : City)
  (cityD : City)
  (inv1 : Investigation)
  (inv2 : Investigation)
  (h1 : cityA.outlets = 150)
  (h2 : cityB.outlets = 120)
  (h3 : cityC.outlets = 190)
  (h4 : cityD.outlets = 140)
  (h5 : inv1.totalOutlets = 600)
  (h6 : inv1.sampleSize = 100)
  (h7 : inv1.cities = [cityA, cityB, cityC, cityD])
  (h8 : inv2.totalOutlets = 20)
  (h9 : inv2.sampleSize = 8)
  (h10 : inv2.cities = [cityC]) :
  (mostAppropriateMethod inv1 = SamplingMethod.StratifiedSampling) ∧ 
  (mostAppropriateMethod inv2 = SamplingMethod.SimpleRandomSampling) := by
  sorry

end NUMINAMATH_CALUDE_appropriate_sampling_methods_l1781_178170


namespace NUMINAMATH_CALUDE_sum_g_79_l1781_178118

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x^2 + 3 * x - 1
def g (y : ℝ) : ℝ := y^2 - 2 * y + 2

-- Define the equation f(x) = 79
def f_eq_79 (x : ℝ) : Prop := f x = 79

-- Theorem statement
theorem sum_g_79 (x₁ x₂ : ℝ) (h₁ : f_eq_79 x₁) (h₂ : f_eq_79 x₂) (h₃ : x₁ ≠ x₂) :
  ∃ (s : ℝ), s = g (f x₁) + g (f x₂) ∧ 
  (∀ (y : ℝ), g y = s ↔ y = 79) :=
sorry

end NUMINAMATH_CALUDE_sum_g_79_l1781_178118


namespace NUMINAMATH_CALUDE_local_min_implies_a_eq_2_l1781_178175

/-- The function f(x) = x(x-a)^2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x * (x - a)^2

/-- f has a local minimum at x = 2 -/
def has_local_min_at_2 (a : ℝ) : Prop :=
  ∃ δ > 0, ∀ x, |x - 2| < δ → f a x ≥ f a 2

theorem local_min_implies_a_eq_2 :
  ∀ a : ℝ, has_local_min_at_2 a → a = 2 := by sorry

end NUMINAMATH_CALUDE_local_min_implies_a_eq_2_l1781_178175


namespace NUMINAMATH_CALUDE_train_b_completion_time_l1781_178185

/-- Proves that Train B takes 2 hours to complete the route given the conditions -/
theorem train_b_completion_time 
  (route_length : ℝ) 
  (train_a_speed : ℝ) 
  (meeting_distance : ℝ) 
  (h1 : route_length = 75) 
  (h2 : train_a_speed = 25) 
  (h3 : meeting_distance = 30) : 
  (route_length / ((route_length - meeting_distance) / (meeting_distance / train_a_speed))) = 2 := by
  sorry

end NUMINAMATH_CALUDE_train_b_completion_time_l1781_178185


namespace NUMINAMATH_CALUDE_simplify_expression_l1781_178146

theorem simplify_expression : 18 * (7/8) * (1/12)^2 = 7/768 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1781_178146


namespace NUMINAMATH_CALUDE_sine_cosine_inequality_l1781_178172

theorem sine_cosine_inequality (x : ℝ) (n m : ℕ) 
  (h1 : 0 < x) (h2 : x < π / 2) (h3 : n > m) : 
  2 * |Real.sin x ^ n - Real.cos x ^ n| ≤ 3 * |Real.sin x ^ m - Real.cos x ^ m| := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_inequality_l1781_178172


namespace NUMINAMATH_CALUDE_cookie_jar_spending_l1781_178183

theorem cookie_jar_spending (initial_amount : ℝ) (amount_left : ℝ) (doris_spent : ℝ) : 
  initial_amount = 21 →
  amount_left = 12 →
  initial_amount - (doris_spent + doris_spent / 2) = amount_left →
  doris_spent = 6 := by
sorry

end NUMINAMATH_CALUDE_cookie_jar_spending_l1781_178183


namespace NUMINAMATH_CALUDE_no_integer_satisfies_inequality_l1781_178133

theorem no_integer_satisfies_inequality : 
  ¬ ∃ (n : ℤ), n > 1 ∧ (⌊Real.sqrt (n - 2) + 2 * Real.sqrt (n + 2)⌋ : ℤ) < ⌊Real.sqrt (9 * n + 6)⌋ := by
  sorry

end NUMINAMATH_CALUDE_no_integer_satisfies_inequality_l1781_178133


namespace NUMINAMATH_CALUDE_similar_triangle_shortest_side_l1781_178142

theorem similar_triangle_shortest_side 
  (a b c : ℝ) 
  (h₁ : a^2 + b^2 = c^2) 
  (h₂ : a = 21) 
  (h₃ : c = 29) 
  (h₄ : a ≤ b) 
  (k : ℝ) 
  (h₅ : k * c = 87) : 
  k * a = 60 := by
sorry

end NUMINAMATH_CALUDE_similar_triangle_shortest_side_l1781_178142


namespace NUMINAMATH_CALUDE_smallest_overlap_percentage_l1781_178167

theorem smallest_overlap_percentage (total : ℝ) (books_percent : ℝ) (movies_percent : ℝ)
  (h_total : total > 0)
  (h_books : books_percent = 95 / 100)
  (h_movies : movies_percent = 85 / 100) :
  (books_percent + movies_percent - 1 : ℝ) = 80 / 100 := by
  sorry

end NUMINAMATH_CALUDE_smallest_overlap_percentage_l1781_178167


namespace NUMINAMATH_CALUDE_plane_binary_trees_eq_triangulations_l1781_178134

/-- A plane binary tree -/
structure PlaneBinaryTree where
  vertices : Set Nat
  edges : Set (Nat × Nat)
  root : Nat
  leaves : Set Nat

/-- A triangulation of a polygon -/
structure Triangulation where
  vertices : Set Nat
  diagonals : Set (Nat × Nat)

/-- The number of different plane binary trees with one root and n leaves -/
def num_plane_binary_trees (n : Nat) : Nat :=
  sorry

/-- The number of triangulations of an (n+1)-gon -/
def num_triangulations (n : Nat) : Nat :=
  sorry

/-- Theorem stating the equality between the number of plane binary trees and triangulations -/
theorem plane_binary_trees_eq_triangulations (n : Nat) :
  num_plane_binary_trees n = num_triangulations n :=
  sorry

end NUMINAMATH_CALUDE_plane_binary_trees_eq_triangulations_l1781_178134


namespace NUMINAMATH_CALUDE_jacks_savings_after_eight_weeks_l1781_178106

/-- Calculates the amount in Jack's savings account after a given number of weeks -/
def savings_after_weeks (initial_amount : ℝ) (weekly_allowance : ℝ) (weekly_expense : ℝ) (weeks : ℕ) : ℝ :=
  initial_amount + (weekly_allowance - weekly_expense) * weeks

/-- Proves that Jack's savings after 8 weeks equals $99 -/
theorem jacks_savings_after_eight_weeks :
  savings_after_weeks 43 10 3 8 = 99 := by
  sorry

#eval savings_after_weeks 43 10 3 8

end NUMINAMATH_CALUDE_jacks_savings_after_eight_weeks_l1781_178106


namespace NUMINAMATH_CALUDE_last_two_digits_of_sum_factorials_l1781_178174

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem last_two_digits_of_sum_factorials :
  sum_factorials 9 % 100 = 13 := by sorry

end NUMINAMATH_CALUDE_last_two_digits_of_sum_factorials_l1781_178174


namespace NUMINAMATH_CALUDE_total_pupils_l1781_178116

theorem total_pupils (girls : ℕ) (boys : ℕ) 
  (h1 : girls = 542) (h2 : boys = 387) : 
  girls + boys = 929 := by
  sorry

end NUMINAMATH_CALUDE_total_pupils_l1781_178116


namespace NUMINAMATH_CALUDE_dubblefud_yellow_chips_l1781_178166

theorem dubblefud_yellow_chips :
  ∀ (yellow blue green red : ℕ),
  -- Yellow chips are worth 2 points
  -- Blue chips are worth 4 points
  -- Green chips are worth 5 points
  -- Red chips are worth 7 points
  -- The product of the point values of the chips is 560000
  2^yellow * 4^blue * 5^green * 7^red = 560000 →
  -- The number of blue chips equals twice the number of green chips
  blue = 2 * green →
  -- The number of red chips is half the number of blue chips
  red = blue / 2 →
  -- The number of yellow chips is 2
  yellow = 2 := by
sorry

end NUMINAMATH_CALUDE_dubblefud_yellow_chips_l1781_178166


namespace NUMINAMATH_CALUDE_peaches_in_basket_l1781_178102

/-- Represents the number of peaches in a basket -/
structure Basket :=
  (red : ℕ)
  (green : ℕ)

/-- The total number of peaches in a basket is the sum of red and green peaches -/
def total_peaches (b : Basket) : ℕ := b.red + b.green

/-- Given a basket with 7 red peaches and 3 green peaches, prove that the total number of peaches is 10 -/
theorem peaches_in_basket :
  ∀ b : Basket, b.red = 7 ∧ b.green = 3 → total_peaches b = 10 :=
by
  sorry

#check peaches_in_basket

end NUMINAMATH_CALUDE_peaches_in_basket_l1781_178102


namespace NUMINAMATH_CALUDE_power_function_through_point_l1781_178173

/-- A power function that passes through the point (2, √2) has exponent 1/2 -/
theorem power_function_through_point (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = x^a) →  -- f is a power function with exponent a
  f 2 = Real.sqrt 2 →  -- f passes through the point (2, √2)
  a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_point_l1781_178173


namespace NUMINAMATH_CALUDE_total_potatoes_to_cook_l1781_178197

/-- Given a cooking scenario where:
  * 6 potatoes are already cooked
  * Each potato takes 8 minutes to cook
  * It takes 72 minutes to cook the remaining potatoes
  Prove that the total number of potatoes to be cooked is 15. -/
theorem total_potatoes_to_cook (already_cooked : ℕ) (cooking_time_per_potato : ℕ) (remaining_cooking_time : ℕ) :
  already_cooked = 6 →
  cooking_time_per_potato = 8 →
  remaining_cooking_time = 72 →
  already_cooked + (remaining_cooking_time / cooking_time_per_potato) = 15 :=
by sorry

end NUMINAMATH_CALUDE_total_potatoes_to_cook_l1781_178197


namespace NUMINAMATH_CALUDE_polygon_chain_sides_l1781_178127

/-- Represents a regular polygon with a given number of sides. -/
structure RegularPolygon where
  sides : ℕ
  sides_positive : sides > 0

/-- Represents a chain of connected regular polygons. -/
structure PolygonChain where
  polygons : List RegularPolygon
  connected : polygons.length > 1

/-- Calculates the number of exposed sides in a chain of connected polygons. -/
def exposedSides (chain : PolygonChain) : ℕ :=
  let n := chain.polygons.length
  let total_sides := (chain.polygons.map RegularPolygon.sides).sum
  let shared_sides := 2 * (n - 1) - 2
  total_sides - shared_sides

/-- The theorem to be proved. -/
theorem polygon_chain_sides :
  ∀ (chain : PolygonChain),
    chain.polygons.map RegularPolygon.sides = [3, 4, 5, 6, 7, 8, 9] →
    exposedSides chain = 30 := by
  sorry

end NUMINAMATH_CALUDE_polygon_chain_sides_l1781_178127


namespace NUMINAMATH_CALUDE_equality_condition_l1781_178110

theorem equality_condition (a b c : ℝ) : a^2 + b*c = (a - b)*(a - c) ↔ a = 0 ∨ b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_equality_condition_l1781_178110


namespace NUMINAMATH_CALUDE_gcd_98_63_l1781_178195

theorem gcd_98_63 : Nat.gcd 98 63 = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_98_63_l1781_178195


namespace NUMINAMATH_CALUDE_third_home_donation_l1781_178160

/-- Represents the donation amounts in cents to avoid floating-point issues -/
def total_donation : ℕ := 70000
def first_home_donation : ℕ := 24500
def second_home_donation : ℕ := 22500

/-- The donation to the third home is the difference between the total donation
    and the sum of donations to the first two homes -/
theorem third_home_donation :
  total_donation - first_home_donation - second_home_donation = 23000 := by
  sorry

end NUMINAMATH_CALUDE_third_home_donation_l1781_178160


namespace NUMINAMATH_CALUDE_more_solutions_without_plus_one_l1781_178163

/-- The upper bound for x, y, z, and t -/
def upperBound : ℕ := 10^6

/-- The number of integral solutions for x^2 - y^2 = z^3 - t^3 -/
def N : ℕ := sorry

/-- The number of integral solutions for x^2 - y^2 = z^3 - t^3 + 1 -/
def M : ℕ := sorry

/-- Theorem stating that N > M -/
theorem more_solutions_without_plus_one : N > M := by
  sorry

end NUMINAMATH_CALUDE_more_solutions_without_plus_one_l1781_178163


namespace NUMINAMATH_CALUDE_road_repair_fractions_l1781_178169

theorem road_repair_fractions (road_length : ℝ) (first_week_fraction second_week_fraction : ℚ) :
  road_length = 1500 →
  first_week_fraction = 5 / 17 →
  second_week_fraction = 4 / 17 →
  (first_week_fraction + second_week_fraction = 9 / 17) ∧
  (1 - (first_week_fraction + second_week_fraction) = 8 / 17) := by
  sorry

end NUMINAMATH_CALUDE_road_repair_fractions_l1781_178169


namespace NUMINAMATH_CALUDE_trig_expression_equals_32_l1781_178126

theorem trig_expression_equals_32 : 
  3 / (Real.sin (20 * π / 180))^2 - 1 / (Real.cos (20 * π / 180))^2 + 64 * (Real.sin (20 * π / 180))^2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_32_l1781_178126


namespace NUMINAMATH_CALUDE_pond_width_calculation_l1781_178171

/-- Represents a rectangular pond -/
structure RectangularPond where
  length : ℝ
  width : ℝ
  depth : ℝ
  volume : ℝ

/-- Theorem: Given a rectangular pond with length 20 meters, depth 5 meters, 
    and volume 1200 cubic meters, its width is 12 meters -/
theorem pond_width_calculation (pond : RectangularPond) 
  (h1 : pond.length = 20)
  (h2 : pond.depth = 5)
  (h3 : pond.volume = 1200)
  (h4 : pond.volume = pond.length * pond.width * pond.depth) :
  pond.width = 12 := by
  sorry

end NUMINAMATH_CALUDE_pond_width_calculation_l1781_178171


namespace NUMINAMATH_CALUDE_singer_tip_percentage_l1781_178109

/-- Proves that the tip percentage is 20% given the conditions of the problem -/
theorem singer_tip_percentage (hours : ℕ) (hourly_rate : ℚ) (total_paid : ℚ) :
  hours = 3 →
  hourly_rate = 15 →
  total_paid = 54 →
  (total_paid - hours * hourly_rate) / (hours * hourly_rate) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_singer_tip_percentage_l1781_178109


namespace NUMINAMATH_CALUDE_grape_juice_percentage_in_mixture_l1781_178176

/-- Represents a mixture with a certain volume and grape juice percentage -/
structure Mixture where
  volume : ℝ
  percentage : ℝ

/-- Calculates the total volume of grape juice in a mixture -/
def grapeJuiceVolume (m : Mixture) : ℝ := m.volume * m.percentage

/-- The problem statement -/
theorem grape_juice_percentage_in_mixture : 
  let mixtureA : Mixture := { volume := 15, percentage := 0.3 }
  let mixtureB : Mixture := { volume := 40, percentage := 0.2 }
  let mixtureC : Mixture := { volume := 25, percentage := 0.1 }
  let pureGrapeJuice : ℝ := 10

  let totalGrapeJuice := grapeJuiceVolume mixtureA + grapeJuiceVolume mixtureB + 
                         grapeJuiceVolume mixtureC + pureGrapeJuice
  let totalVolume := mixtureA.volume + mixtureB.volume + mixtureC.volume + pureGrapeJuice

  let resultPercentage := totalGrapeJuice / totalVolume

  abs (resultPercentage - 0.2778) < 0.0001 := by
  sorry


end NUMINAMATH_CALUDE_grape_juice_percentage_in_mixture_l1781_178176


namespace NUMINAMATH_CALUDE_bus_driver_compensation_l1781_178199

-- Define the constants
def regular_rate : ℝ := 16
def regular_hours : ℕ := 40
def overtime_rate_increase : ℝ := 0.75
def total_hours_worked : ℕ := 44

-- Define the functions
def overtime_rate : ℝ := regular_rate * (1 + overtime_rate_increase)

def calculate_compensation (hours : ℕ) : ℝ :=
  if hours ≤ regular_hours then
    hours * regular_rate
  else
    regular_hours * regular_rate + (hours - regular_hours) * overtime_rate

-- Theorem to prove
theorem bus_driver_compensation :
  calculate_compensation total_hours_worked = 752 :=
by sorry

end NUMINAMATH_CALUDE_bus_driver_compensation_l1781_178199


namespace NUMINAMATH_CALUDE_simplify_fraction_l1781_178194

theorem simplify_fraction (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  1 / a + 1 / b - (2 * a + b) / (2 * a * b) = 1 / (2 * a) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1781_178194


namespace NUMINAMATH_CALUDE_dress_designs_count_l1781_178186

/-- The number of fabric colors available -/
def num_colors : ℕ := 3

/-- The number of different patterns available -/
def num_patterns : ℕ := 4

/-- The number of sleeve length options available -/
def num_sleeve_lengths : ℕ := 2

/-- The total number of possible dress designs -/
def total_designs : ℕ := num_colors * num_patterns * num_sleeve_lengths

theorem dress_designs_count : total_designs = 24 := by
  sorry

end NUMINAMATH_CALUDE_dress_designs_count_l1781_178186


namespace NUMINAMATH_CALUDE_minimal_withdrawals_l1781_178112

/-- Represents a withdrawal strategy -/
structure WithdrawalStrategy where
  red : ℕ
  blue : ℕ
  green : ℕ
  count : ℕ

/-- Represents the package of marbles -/
structure MarblePackage where
  red : ℕ
  blue : ℕ
  green : ℕ

/-- Checks if a withdrawal strategy is valid according to the constraints -/
def is_valid_strategy (s : WithdrawalStrategy) : Prop :=
  s.red ≤ 1 ∧ s.blue ≤ 2 ∧ s.red + s.blue + s.green ≤ 5

/-- Checks if a list of withdrawal strategies empties the package -/
def empties_package (p : MarblePackage) (strategies : List WithdrawalStrategy) : Prop :=
  strategies.foldl (fun acc s => 
    { red := acc.red - s.red * s.count
    , blue := acc.blue - s.blue * s.count
    , green := acc.green - s.green * s.count
    }) p = ⟨0, 0, 0⟩

/-- The main theorem stating the minimal number of withdrawals -/
theorem minimal_withdrawals (p : MarblePackage) 
  (h_red : p.red = 200) (h_blue : p.blue = 300) (h_green : p.green = 400) :
  ∃ (strategies : List WithdrawalStrategy),
    (∀ s ∈ strategies, is_valid_strategy s) ∧
    empties_package p strategies ∧
    (strategies.foldl (fun acc s => acc + s.count) 0 = 200) ∧
    (∀ (other_strategies : List WithdrawalStrategy),
      (∀ s ∈ other_strategies, is_valid_strategy s) →
      empties_package p other_strategies →
      strategies.foldl (fun acc s => acc + s.count) 0 ≤ 
      other_strategies.foldl (fun acc s => acc + s.count) 0) :=
sorry

end NUMINAMATH_CALUDE_minimal_withdrawals_l1781_178112


namespace NUMINAMATH_CALUDE_sum_of_three_consecutive_integers_l1781_178143

theorem sum_of_three_consecutive_integers :
  ∃ n : ℤ, n - 1 + n + (n + 1) = 21 ∧
  (n - 1 + n + (n + 1) = 17 ∨
   n - 1 + n + (n + 1) = 11 ∨
   n - 1 + n + (n + 1) = 25 ∨
   n - 1 + n + (n + 1) = 21 ∨
   n - 1 + n + (n + 1) = 8) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_three_consecutive_integers_l1781_178143


namespace NUMINAMATH_CALUDE_polygon_quadrilateral_iff_exterior_eq_interior_l1781_178178

/-- A polygon is a quadrilateral if and only if the sum of its exterior angles
    equals the sum of its interior angles. -/
theorem polygon_quadrilateral_iff_exterior_eq_interior :
  ∀ n : ℕ, n ≥ 3 →
  (n = 4 ↔ (n - 2) * 180 = 360) :=
by sorry

end NUMINAMATH_CALUDE_polygon_quadrilateral_iff_exterior_eq_interior_l1781_178178


namespace NUMINAMATH_CALUDE_training_trip_duration_l1781_178188

/-- The number of supervisors --/
def n : ℕ := 15

/-- The number of supervisors overseeing the pool each day --/
def k : ℕ := 3

/-- The number of ways to choose 2 supervisors from n supervisors --/
def total_pairs : ℕ := n.choose 2

/-- The number of pairs formed each day --/
def pairs_per_day : ℕ := k.choose 2

/-- The number of days required for the training trip --/
def days : ℕ := total_pairs / pairs_per_day

theorem training_trip_duration :
  n = 15 → k = 3 → days = 35 := by sorry

end NUMINAMATH_CALUDE_training_trip_duration_l1781_178188


namespace NUMINAMATH_CALUDE_inverse_proportion_y_relationship_l1781_178153

/-- Given two points on an inverse proportion function, prove that y₁ < y₂ -/
theorem inverse_proportion_y_relationship (k : ℝ) (y₁ y₂ : ℝ) :
  (2 : ℝ) > 0 ∧ (3 : ℝ) > 0 ∧
  y₁ = (-k^2 - 1) / 2 ∧
  y₂ = (-k^2 - 1) / 3 →
  y₁ < y₂ :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_y_relationship_l1781_178153


namespace NUMINAMATH_CALUDE_min_x_minus_y_l1781_178129

theorem min_x_minus_y (x y : ℝ) (h1 : x > 0) (h2 : 0 > y) 
  (h3 : 1 / (x + 2) + 1 / (1 - y) = 1 / 6) : x - y ≥ 21 := by
  sorry

end NUMINAMATH_CALUDE_min_x_minus_y_l1781_178129


namespace NUMINAMATH_CALUDE_expression_evaluation_l1781_178187

theorem expression_evaluation :
  (5^1003 + 6^1002)^2 - (5^1003 - 6^1002)^2 = 600 * 30^1002 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1781_178187


namespace NUMINAMATH_CALUDE_triangle_pentagon_side_ratio_l1781_178165

theorem triangle_pentagon_side_ratio :
  ∀ (t p : ℝ),
  t > 0 ∧ p > 0 →
  3 * t = 30 →
  5 * p = 30 →
  t / p = 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_pentagon_side_ratio_l1781_178165


namespace NUMINAMATH_CALUDE_delaware_cell_phones_l1781_178130

/-- The number of cell phones in Delaware -/
def cell_phones_in_delaware (population : ℕ) (phones_per_thousand : ℕ) : ℕ :=
  (population / 1000) * phones_per_thousand

/-- Theorem stating the number of cell phones in Delaware -/
theorem delaware_cell_phones :
  cell_phones_in_delaware 974000 673 = 655502 := by
  sorry

end NUMINAMATH_CALUDE_delaware_cell_phones_l1781_178130


namespace NUMINAMATH_CALUDE_investment_problem_l1781_178190

/-- The investment problem -/
theorem investment_problem (a b total_profit a_profit : ℕ)
  (h1 : a = 6300)
  (h2 : b = 4200)
  (h3 : total_profit = 12600)
  (h4 : a_profit = 3780)
  (h5 : ∀ x : ℕ, a / (a + b + x) = a_profit / total_profit) :
  ∃ c : ℕ, c = 10500 ∧ a / (a + b + c) = a_profit / total_profit :=
sorry

end NUMINAMATH_CALUDE_investment_problem_l1781_178190


namespace NUMINAMATH_CALUDE_compare_negative_mixed_numbers_l1781_178144

theorem compare_negative_mixed_numbers :
  -6.5 > -(6 + 3/5) := by sorry

end NUMINAMATH_CALUDE_compare_negative_mixed_numbers_l1781_178144


namespace NUMINAMATH_CALUDE_cubic_equation_only_trivial_solution_l1781_178137

theorem cubic_equation_only_trivial_solution (x y z : ℤ) :
  x^3 - 2*y^3 - 4*z^3 = 0 → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_only_trivial_solution_l1781_178137


namespace NUMINAMATH_CALUDE_quadratic_equation_real_roots_l1781_178103

theorem quadratic_equation_real_roots (m : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + (3*m - 1)*x₁ + (2*m^2 - m) = 0 ∧
                x₂^2 + (3*m - 1)*x₂ + (2*m^2 - m) = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_real_roots_l1781_178103


namespace NUMINAMATH_CALUDE_investment_growth_approx_l1781_178117

/-- Approximates the future value of an investment with compound interest -/
def future_value (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Theorem: An investment of $1500 at 8% annual interest grows to approximately $13500 in 28 years -/
theorem investment_growth_approx :
  ∃ ε > 0, abs (future_value 1500 0.08 28 - 13500) < ε :=
by
  sorry

end NUMINAMATH_CALUDE_investment_growth_approx_l1781_178117


namespace NUMINAMATH_CALUDE_cat_whiskers_correct_l1781_178147

structure Cat where
  name : String
  whiskers : ℕ

def princess_puff : Cat := { name := "Princess Puff", whiskers := 14 }

def catman_do : Cat := { 
  name := "Catman Do", 
  whiskers := 2 * princess_puff.whiskers - 6 
}

def sir_whiskerson : Cat := { 
  name := "Sir Whiskerson", 
  whiskers := princess_puff.whiskers + catman_do.whiskers + 8 
}

def lady_flufflepuff : Cat := { 
  name := "Lady Flufflepuff", 
  whiskers := sir_whiskerson.whiskers / 2 + 4 
}

def mr_mittens : Cat := { 
  name := "Mr. Mittens", 
  whiskers := Int.natAbs (catman_do.whiskers - lady_flufflepuff.whiskers)
}

theorem cat_whiskers_correct : 
  princess_puff.whiskers = 14 ∧ 
  catman_do.whiskers = 22 ∧ 
  sir_whiskerson.whiskers = 44 ∧ 
  lady_flufflepuff.whiskers = 26 ∧ 
  mr_mittens.whiskers = 4 := by
  sorry

end NUMINAMATH_CALUDE_cat_whiskers_correct_l1781_178147


namespace NUMINAMATH_CALUDE_theorem_1_theorem_2_l1781_178140

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the functional equation
def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x + f y = 2 * f ((x + y) / 2) * f ((x - y) / 2)

-- Theorem 1: If f(1) = 1/2, then f(2) = -1/2
theorem theorem_1 (h : functional_equation f) (h1 : f 1 = 1/2) : f 2 = -1/2 := by
  sorry

-- Theorem 2: If f(1) = 0, then f(11/2) + f(15/2) + f(19/2) + ... + f(2019/2) + f(2023/2) = 0
theorem theorem_2 (h : functional_equation f) (h1 : f 1 = 0) :
  f (11/2) + f (15/2) + f (19/2) + f (2019/2) + f (2023/2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_theorem_1_theorem_2_l1781_178140


namespace NUMINAMATH_CALUDE_grandmas_salad_ratio_l1781_178111

/-- Given the conditions of Grandma's salad, prove the ratio of pickles to cherry tomatoes -/
theorem grandmas_salad_ratio : 
  ∀ (mushrooms cherry_tomatoes pickles bacon_bits red_bacon_bits : ℕ),
    mushrooms = 3 →
    cherry_tomatoes = 2 * mushrooms →
    bacon_bits = 4 * pickles →
    red_bacon_bits * 3 = bacon_bits →
    red_bacon_bits = 32 →
    (pickles : ℚ) / cherry_tomatoes = 4 / 1 :=
by
  sorry

end NUMINAMATH_CALUDE_grandmas_salad_ratio_l1781_178111


namespace NUMINAMATH_CALUDE_alexis_bought_21_pants_l1781_178145

/-- Given information about Isabella and Alexis's shopping -/
structure ShoppingInfo where
  isabella_total : ℕ
  alexis_dresses : ℕ
  alexis_multiplier : ℕ

/-- Calculates the number of pants Alexis bought -/
def alexis_pants (info : ShoppingInfo) : ℕ :=
  info.alexis_multiplier * (info.isabella_total - (info.alexis_dresses / info.alexis_multiplier))

/-- Theorem stating that Alexis bought 21 pants given the shopping information -/
theorem alexis_bought_21_pants (info : ShoppingInfo) 
  (h1 : info.isabella_total = 13)
  (h2 : info.alexis_dresses = 18)
  (h3 : info.alexis_multiplier = 3) : 
  alexis_pants info = 21 := by
  sorry

#eval alexis_pants ⟨13, 18, 3⟩

end NUMINAMATH_CALUDE_alexis_bought_21_pants_l1781_178145


namespace NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_8_l1781_178132

theorem factorization_of_2x_squared_minus_8 (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_8_l1781_178132


namespace NUMINAMATH_CALUDE_intersection_and_non_membership_l1781_178114

-- Define the lines
def line1 (x y : ℚ) : Prop := y = -3 * x
def line2 (x y : ℚ) : Prop := y + 3 = 9 * x
def line3 (x y : ℚ) : Prop := y = 2 * x - 1

-- Define the intersection point
def intersection_point : ℚ × ℚ := (1/4, -3/4)

-- Theorem statement
theorem intersection_and_non_membership :
  let (x, y) := intersection_point
  (line1 x y ∧ line2 x y) ∧ ¬(line3 x y) := by sorry

end NUMINAMATH_CALUDE_intersection_and_non_membership_l1781_178114


namespace NUMINAMATH_CALUDE_prosecutor_conclusion_l1781_178154

-- Define the types for guilt
inductive Guilt
| Guilty
| NotGuilty

-- Define the prosecutor's statements
def statement1 (X Y : Guilt) : Prop :=
  X = Guilt.NotGuilty ∨ Y = Guilt.Guilty

def statement2 (X : Guilt) : Prop :=
  X = Guilt.Guilty

-- Theorem to prove
theorem prosecutor_conclusion (X Y : Guilt) :
  statement1 X Y ∧ statement2 X →
  X = Guilt.Guilty ∧ Y = Guilt.Guilty :=
by
  sorry


end NUMINAMATH_CALUDE_prosecutor_conclusion_l1781_178154


namespace NUMINAMATH_CALUDE_fixed_point_of_function_l1781_178135

/-- The function f(x) = kx - k - a^(x-1) always passes through the point (1, -1) -/
theorem fixed_point_of_function (k : ℝ) (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f := fun x => k * x - k - a^(x - 1)
  f 1 = -1 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_function_l1781_178135


namespace NUMINAMATH_CALUDE_wine_problem_equations_l1781_178115

/-- Represents the number of guests intoxicated by one bottle of good wine -/
def good_wine_intoxication : ℚ := 3

/-- Represents the number of bottles of weak wine needed to intoxicate one guest -/
def weak_wine_intoxication : ℚ := 3

/-- Represents the total number of intoxicated guests -/
def total_intoxicated_guests : ℚ := 33

/-- Represents the total number of bottles of wine consumed -/
def total_bottles : ℚ := 19

/-- Represents the number of bottles of good wine -/
def x : ℚ := sorry

/-- Represents the number of bottles of weak wine -/
def y : ℚ := sorry

theorem wine_problem_equations :
  (x + y = total_bottles) ∧
  (good_wine_intoxication * x + (1 / weak_wine_intoxication) * y = total_intoxicated_guests) :=
by sorry

end NUMINAMATH_CALUDE_wine_problem_equations_l1781_178115


namespace NUMINAMATH_CALUDE_polynomial_roots_l1781_178149

theorem polynomial_roots : ∃ (x₁ x₂ x₃ : ℝ), 
  (x₁ = -2 ∧ x₂ = 2 + Real.sqrt 2 ∧ x₃ = 2 - Real.sqrt 2) ∧
  (∀ x : ℝ, x^4 - 4*x^3 + 5*x^2 - 2*x - 8 = 0 ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃)) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_roots_l1781_178149


namespace NUMINAMATH_CALUDE_parabolas_intersect_l1781_178198

/-- Parabola function -/
def parabola (p q : ℝ) (x : ℝ) : ℝ := x^2 + p*x + q

/-- Theorem: All parabolas with p + q = 2019 intersect at (1, 2020) -/
theorem parabolas_intersect (p q : ℝ) (h : p + q = 2019) : parabola p q 1 = 2020 := by
  sorry

end NUMINAMATH_CALUDE_parabolas_intersect_l1781_178198


namespace NUMINAMATH_CALUDE_eggs_at_town_hall_l1781_178101

/-- Given the number of eggs found at different locations during an Easter egg hunt, 
    this theorem proves how many eggs were found at the town hall. -/
theorem eggs_at_town_hall 
  (total_eggs : ℕ)
  (club_house_eggs : ℕ)
  (park_eggs : ℕ)
  (h1 : total_eggs = 80)
  (h2 : club_house_eggs = 40)
  (h3 : park_eggs = 25) :
  total_eggs - (club_house_eggs + park_eggs) = 15 := by
  sorry

#check eggs_at_town_hall

end NUMINAMATH_CALUDE_eggs_at_town_hall_l1781_178101


namespace NUMINAMATH_CALUDE_sarahs_deleted_folder_size_l1781_178189

theorem sarahs_deleted_folder_size 
  (initial_free : ℝ) 
  (initial_used : ℝ) 
  (new_files_size : ℝ) 
  (new_drive_size : ℝ) 
  (new_drive_free : ℝ)
  (h1 : initial_free = 2.4)
  (h2 : initial_used = 12.6)
  (h3 : new_files_size = 2)
  (h4 : new_drive_size = 20)
  (h5 : new_drive_free = 10) : 
  ∃ (deleted_folder_size : ℝ), 
    deleted_folder_size = 4.6 ∧ 
    initial_used - deleted_folder_size + new_files_size = new_drive_size - new_drive_free :=
by sorry

end NUMINAMATH_CALUDE_sarahs_deleted_folder_size_l1781_178189


namespace NUMINAMATH_CALUDE_cube_order_preserving_l1781_178105

theorem cube_order_preserving (a b : ℝ) : a^3 > b^3 → a > b := by
  sorry

end NUMINAMATH_CALUDE_cube_order_preserving_l1781_178105


namespace NUMINAMATH_CALUDE_inequality_proof_l1781_178158

theorem inequality_proof (a b t : ℝ) (h1 : 0 < t) (h2 : t < 1) (h3 : a * b > 0) :
  (a^2 / t^3) + (b^2 / (1 - t^3)) ≥ (a + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1781_178158


namespace NUMINAMATH_CALUDE_kyle_gas_and_maintenance_l1781_178184

/-- Calculates the amount left for gas and maintenance given monthly income and expenses --/
def amount_left_for_gas_and_maintenance (monthly_income : ℕ) (rent utilities retirement_savings groceries insurance misc car_payment : ℕ) : ℕ :=
  monthly_income - (rent + utilities + retirement_savings + groceries + insurance + misc + car_payment)

/-- Theorem: Kyle's amount left for gas and maintenance is $350 --/
theorem kyle_gas_and_maintenance :
  amount_left_for_gas_and_maintenance 3200 1250 150 400 300 200 200 350 = 350 := by
  sorry

end NUMINAMATH_CALUDE_kyle_gas_and_maintenance_l1781_178184


namespace NUMINAMATH_CALUDE_bob_probability_after_two_turns_l1781_178157

/-- Represents the player who has the ball -/
inductive Player : Type
| Alice : Player
| Bob : Player

/-- The probability of keeping the ball for each player -/
def keep_prob (p : Player) : ℚ :=
  match p with
  | Player.Alice => 2/3
  | Player.Bob => 3/4

/-- The probability of tossing the ball for each player -/
def toss_prob (p : Player) : ℚ :=
  1 - keep_prob p

/-- The probability that Bob has the ball after two turns, given he starts with it -/
def bob_has_ball_after_two_turns : ℚ :=
  keep_prob Player.Bob * keep_prob Player.Bob +
  keep_prob Player.Bob * toss_prob Player.Bob * keep_prob Player.Alice +
  toss_prob Player.Bob * toss_prob Player.Alice

theorem bob_probability_after_two_turns :
  bob_has_ball_after_two_turns = 37/48 := by
  sorry

end NUMINAMATH_CALUDE_bob_probability_after_two_turns_l1781_178157


namespace NUMINAMATH_CALUDE_ace_of_hearts_probability_l1781_178108

def standard_deck := 52
def ace_of_hearts_per_deck := 1

theorem ace_of_hearts_probability (combined_deck : ℕ) (ace_of_hearts : ℕ) :
  combined_deck = 2 * standard_deck →
  ace_of_hearts = 2 * ace_of_hearts_per_deck →
  (ace_of_hearts : ℚ) / combined_deck = 1 / 52 :=
by sorry

end NUMINAMATH_CALUDE_ace_of_hearts_probability_l1781_178108


namespace NUMINAMATH_CALUDE_max_projection_area_is_one_l1781_178121

/-- A tetrahedron with two adjacent isosceles right triangle faces -/
structure Tetrahedron where
  /-- The length of the hypotenuse of the isosceles right triangle faces -/
  hypotenuse : ℝ
  /-- The dihedral angle between the two adjacent isosceles right triangle faces -/
  dihedral_angle : ℝ

/-- The maximum area of the projection of a rotating tetrahedron -/
def max_projection_area (t : Tetrahedron) : ℝ := 1

/-- Theorem stating that the maximum area of the projection is 1 -/
theorem max_projection_area_is_one (t : Tetrahedron) 
  (h1 : t.hypotenuse = 2)
  (h2 : t.dihedral_angle = π / 3) : 
  max_projection_area t = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_projection_area_is_one_l1781_178121


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1781_178100

/-- Given an arithmetic sequence, prove that if the sum of the first four terms is 2l,
    the sum of the last four terms is 67, and the sum of the first n terms is 286,
    then the number of terms n is 26. -/
theorem arithmetic_sequence_problem (l : ℝ) (a d : ℝ) (n : ℕ) :
  (4 * a + 6 * d = 2 * l) →
  (4 * (a + (n - 1) * d) - 6 * d = 67) →
  (n * (2 * a + (n - 1) * d) / 2 = 286) →
  n = 26 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1781_178100


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1781_178128

theorem inequality_solution_set (a : ℕ) : 
  (∀ x, (a - 2) * x > a - 2 ↔ x < 1) → (a = 0 ∨ a = 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1781_178128


namespace NUMINAMATH_CALUDE_football_progress_l1781_178125

def yard_changes : List Int := [-5, 9, -12, 17, -15, 24, -7]

theorem football_progress : yard_changes.sum = 11 := by
  sorry

end NUMINAMATH_CALUDE_football_progress_l1781_178125


namespace NUMINAMATH_CALUDE_union_M_complement_N_equals_U_l1781_178161

def U : Set ℝ := Set.univ

def M : Set ℝ := {x | x < 1}

def N : Set ℝ := {x | 0 < x ∧ x < 1}

theorem union_M_complement_N_equals_U : M ∪ (U \ N) = U := by sorry

end NUMINAMATH_CALUDE_union_M_complement_N_equals_U_l1781_178161


namespace NUMINAMATH_CALUDE_complement_of_union_is_two_four_l1781_178141

-- Define the universal set U
def U : Set ℕ := {x | x > 0 ∧ x < 6}

-- Define sets A and B
def A : Set ℕ := {1, 3}
def B : Set ℕ := {3, 5}

-- State the theorem
theorem complement_of_union_is_two_four :
  (U \ (A ∪ B)) = {2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_is_two_four_l1781_178141


namespace NUMINAMATH_CALUDE_rectangle_iff_equal_diagonals_l1781_178107

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define the concept of a rectangle
def isRectangle (q : Quadrilateral) : Prop := sorry

-- Define the concept of diagonal length
def diagonalLength (q : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem rectangle_iff_equal_diagonals (q : Quadrilateral) :
  isRectangle q ↔ diagonalLength q = diagonalLength q := by sorry

end NUMINAMATH_CALUDE_rectangle_iff_equal_diagonals_l1781_178107


namespace NUMINAMATH_CALUDE_soccer_field_area_l1781_178159

theorem soccer_field_area (w l : ℝ) (h1 : l = 3 * w - 30) (h2 : 2 * (w + l) = 880) :
  w * l = 37906.25 := by
  sorry

end NUMINAMATH_CALUDE_soccer_field_area_l1781_178159


namespace NUMINAMATH_CALUDE_frisbee_tournament_committees_l1781_178192

theorem frisbee_tournament_committees :
  let total_teams : ℕ := 4
  let members_per_team : ℕ := 8
  let host_committee_members : ℕ := 4
  let non_host_committee_members : ℕ := 2
  let total_committee_members : ℕ := 10

  (total_teams * (Nat.choose members_per_team host_committee_members) *
   (Nat.choose members_per_team non_host_committee_members) ^ (total_teams - 1)) = 6593280 :=
by sorry

end NUMINAMATH_CALUDE_frisbee_tournament_committees_l1781_178192


namespace NUMINAMATH_CALUDE_equality_sum_l1781_178191

theorem equality_sum (M N : ℚ) : 
  (3 / 5 : ℚ) = M / 75 ∧ (3 / 5 : ℚ) = 90 / N → M + N = 195 := by
  sorry

end NUMINAMATH_CALUDE_equality_sum_l1781_178191


namespace NUMINAMATH_CALUDE_ball_throw_circle_l1781_178148

/-- Given a circular arrangement of 15 elements, prove that starting from
    element 1 and moving with a step of 5 (modulo 15), it takes exactly 3
    steps to return to element 1. -/
theorem ball_throw_circle (n : ℕ) (h : n = 15) :
  let f : ℕ → ℕ := λ x => (x + 5) % n
  ∃ k : ℕ, k > 0 ∧ (f^[k] 1 = 1) ∧ ∀ m : ℕ, 0 < m → m < k → f^[m] 1 ≠ 1 ∧ k = 3 :=
by sorry

end NUMINAMATH_CALUDE_ball_throw_circle_l1781_178148


namespace NUMINAMATH_CALUDE_parallel_vectors_y_value_l1781_178113

/-- Two planar vectors are parallel if their components are proportional -/
def are_parallel (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.2 = v1.2 * v2.1

theorem parallel_vectors_y_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (2, y)
  are_parallel a b → y = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_y_value_l1781_178113


namespace NUMINAMATH_CALUDE_expected_yolks_in_carton_l1781_178124

/-- Represents a carton of eggs with various yolk counts -/
structure EggCarton where
  total_eggs : ℕ
  double_yolk_eggs : ℕ
  triple_yolk_eggs : ℕ
  extra_yolk_probability : ℝ

/-- Calculates the expected number of yolks in a carton of eggs -/
def expected_yolks (carton : EggCarton) : ℝ :=
  let single_yolk_eggs := carton.total_eggs - carton.double_yolk_eggs - carton.triple_yolk_eggs
  let base_yolks := single_yolk_eggs + 2 * carton.double_yolk_eggs + 3 * carton.triple_yolk_eggs
  let extra_yolks := carton.extra_yolk_probability * (carton.double_yolk_eggs + carton.triple_yolk_eggs)
  base_yolks + extra_yolks

/-- Theorem stating the expected number of yolks in the given carton -/
theorem expected_yolks_in_carton :
  let carton : EggCarton := {
    total_eggs := 15,
    double_yolk_eggs := 5,
    triple_yolk_eggs := 3,
    extra_yolk_probability := 0.1
  }
  expected_yolks carton = 26.8 := by sorry

end NUMINAMATH_CALUDE_expected_yolks_in_carton_l1781_178124


namespace NUMINAMATH_CALUDE_probability_walk_300_or_less_l1781_178179

/-- Represents an airport with gates in a straight line. -/
structure Airport where
  num_gates : ℕ
  gate_distance : ℝ

/-- Calculates the number of gate pairs within a given distance. -/
def count_gate_pairs_within_distance (a : Airport) (max_distance : ℝ) : ℕ :=
  sorry

/-- Calculates the total number of possible gate pair assignments. -/
def total_gate_pairs (a : Airport) : ℕ :=
  sorry

/-- The main theorem stating the probability of walking 300 feet or less. -/
theorem probability_walk_300_or_less (a : Airport) :
  a.num_gates = 16 ∧ a.gate_distance = 75 →
  (count_gate_pairs_within_distance a 300 : ℚ) / (total_gate_pairs a : ℚ) = 9 / 20 :=
by sorry

end NUMINAMATH_CALUDE_probability_walk_300_or_less_l1781_178179


namespace NUMINAMATH_CALUDE_rectangular_field_length_l1781_178180

theorem rectangular_field_length (length width : ℝ) 
  (h1 : length * width = 144)
  (h2 : (length + 6) * width = 198) :
  length = 16 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_length_l1781_178180


namespace NUMINAMATH_CALUDE_unit_digit_product_l1781_178120

theorem unit_digit_product : ∃ n : ℕ, (5 + 1) * (5^3 + 1) * (5^6 + 1) * (5^12 + 1) ≡ 6 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_product_l1781_178120


namespace NUMINAMATH_CALUDE_binary_addition_and_predecessor_l1781_178177

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_binary (n : Nat) : List Bool :=
  sorry

theorem binary_addition_and_predecessor :
  let M : List Bool := [false, true, true, true, false, true]
  let M_plus_5 : List Bool := [true, true, false, false, true, true]
  let M_plus_5_pred : List Bool := [false, true, false, false, true, true]
  (binary_to_decimal M) + 5 = binary_to_decimal M_plus_5 ∧
  binary_to_decimal M_plus_5 - 1 = binary_to_decimal M_plus_5_pred :=
by
  sorry

#check binary_addition_and_predecessor

end NUMINAMATH_CALUDE_binary_addition_and_predecessor_l1781_178177


namespace NUMINAMATH_CALUDE_inequality_solution_l1781_178152

theorem inequality_solution (x : ℝ) : 
  (x / (x - 1) ≥ 2 * x) ↔ (1 < x ∧ x ≤ 3/2) ∨ (x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1781_178152


namespace NUMINAMATH_CALUDE_investment_value_proof_l1781_178104

theorem investment_value_proof (x : ℝ) : 
  x > 0 ∧ 
  0.07 * x + 0.23 * 1500 = 0.19 * (x + 1500) →
  x = 500 := by
sorry

end NUMINAMATH_CALUDE_investment_value_proof_l1781_178104


namespace NUMINAMATH_CALUDE_max_reciprocal_sum_l1781_178139

theorem max_reciprocal_sum (m n : ℝ) (h1 : m * n > 0) (h2 : m + n = -1) :
  1 / m + 1 / n ≤ 4 := by
sorry

end NUMINAMATH_CALUDE_max_reciprocal_sum_l1781_178139


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l1781_178150

/-- A geometric sequence with sum S_n = k^n + r^m -/
structure GeometricSequence where
  k : ℝ
  r : ℝ
  m : ℤ
  a : ℕ → ℝ
  sum : ℕ → ℝ
  is_geometric : ∀ n, a (n + 1) = a n * (a 2 / a 1)
  sum_formula : ∀ n, sum n = k^n + r^m

/-- The properties of r and m in the geometric sequence -/
theorem geometric_sequence_properties (seq : GeometricSequence) : 
  seq.r = -1 ∧ Odd seq.m :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l1781_178150


namespace NUMINAMATH_CALUDE_product_of_sums_inequality_l1781_178122

theorem product_of_sums_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b) * (b + c) * (c + a) ≥ 8 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_inequality_l1781_178122


namespace NUMINAMATH_CALUDE_smallest_square_division_l1781_178193

/-- A structure representing a division of a square into smaller squares. -/
structure SquareDivision (n : ℕ) :=
  (num_40 : ℕ)  -- number of 40x40 squares
  (num_49 : ℕ)  -- number of 49x49 squares
  (valid : 40 * num_40 + 49 * num_49 = n)
  (non_empty : num_40 > 0 ∧ num_49 > 0)

/-- The theorem stating that 2000 is the smallest n that satisfies the conditions. -/
theorem smallest_square_division :
  (∃ (d : SquareDivision 2000), True) ∧
  (∀ m : ℕ, m < 2000 → ¬∃ (d : SquareDivision m), True) :=
sorry

end NUMINAMATH_CALUDE_smallest_square_division_l1781_178193


namespace NUMINAMATH_CALUDE_complex_sum_l1781_178155

theorem complex_sum (z : ℂ) (h : z^2 + z + 1 = 0) : 
  z^101 + z^102 + z^103 + z^104 + z^105 = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_l1781_178155


namespace NUMINAMATH_CALUDE_binomial_coefficient_19_12_l1781_178138

theorem binomial_coefficient_19_12 : 
  (Nat.choose 20 12 = 125970) → 
  (Nat.choose 19 11 = 75582) → 
  (Nat.choose 18 11 = 31824) → 
  (Nat.choose 19 12 = 50388) := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_19_12_l1781_178138


namespace NUMINAMATH_CALUDE_intersection_P_Q_l1781_178123

-- Define the sets P and Q
def P : Set ℝ := {x | x > 1}
def Q : Set ℝ := {x | x < 2}

-- State the theorem
theorem intersection_P_Q : P ∩ Q = Set.Ioo 1 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l1781_178123


namespace NUMINAMATH_CALUDE_gold_bars_weighing_l1781_178162

theorem gold_bars_weighing (C₁ C₂ C₃ C₄ C₅ C₆ C₇ C₈ C₉ C₁₀ C₁₁ C₁₂ C₁₃ : ℝ) 
  (h₁ : C₁ ≥ 0) (h₂ : C₂ ≥ 0) (h₃ : C₃ ≥ 0) (h₄ : C₄ ≥ 0) (h₅ : C₅ ≥ 0)
  (h₆ : C₆ ≥ 0) (h₇ : C₇ ≥ 0) (h₈ : C₈ ≥ 0) (h₉ : C₉ ≥ 0) (h₁₀ : C₁₀ ≥ 0)
  (h₁₁ : C₁₁ ≥ 0) (h₁₂ : C₁₂ ≥ 0) (h₁₃ : C₁₃ ≥ 0)
  (W₁ : ℝ) (hW₁ : W₁ = C₁ + C₂)
  (W₂ : ℝ) (hW₂ : W₂ = C₁ + C₃)
  (W₃ : ℝ) (hW₃ : W₃ = C₂ + C₃)
  (W₄ : ℝ) (hW₄ : W₄ = C₄ + C₅)
  (W₅ : ℝ) (hW₅ : W₅ = C₆ + C₇)
  (W₆ : ℝ) (hW₆ : W₆ = C₈ + C₉)
  (W₇ : ℝ) (hW₇ : W₇ = C₁₀ + C₁₁)
  (W₈ : ℝ) (hW₈ : W₈ = C₁₂ + C₁₃) :
  C₁ + C₂ + C₃ + C₄ + C₅ + C₆ + C₇ + C₈ + C₉ + C₁₀ + C₁₁ + C₁₂ + C₁₃ = 
  (W₁ + W₂ + W₃) / 2 + (W₄ + W₅ + W₆ + W₇ + W₈) :=
by sorry

end NUMINAMATH_CALUDE_gold_bars_weighing_l1781_178162


namespace NUMINAMATH_CALUDE_right_triangle_sets_l1781_178164

theorem right_triangle_sets : ∃! (a b c : ℝ), (a = 4 ∧ b = 6 ∧ c = 8) ∧
  ¬(a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) ∧
  ((3^2 + 4^2 = 5^2) ∧ (5^2 + 12^2 = 13^2) ∧ (2^2 + 3^2 = (Real.sqrt 13)^2)) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sets_l1781_178164


namespace NUMINAMATH_CALUDE_apple_buying_difference_l1781_178131

theorem apple_buying_difference :
  ∀ (w : ℕ),
  (2 * 30 + 3 * w = 210) →
  (30 < w) →
  (w - 30 = 20) :=
by
  sorry

end NUMINAMATH_CALUDE_apple_buying_difference_l1781_178131


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_two_l1781_178182

theorem fraction_zero_implies_x_equals_two (x : ℝ) :
  (x^2 - x - 2) / (x + 1) = 0 → x = 2 :=
by sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_two_l1781_178182


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l1781_178168

theorem simplify_and_rationalize (x : ℝ) :
  1 / (2 + 1 / (Real.sqrt 5 + 2)) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l1781_178168


namespace NUMINAMATH_CALUDE_total_subjects_l1781_178181

theorem total_subjects (average_all : ℝ) (average_five : ℝ) (last_subject : ℝ) 
  (h1 : average_all = 76)
  (h2 : average_five = 74)
  (h3 : last_subject = 86) :
  ∃ n : ℕ, n = 6 ∧ 
    n * average_all = (n - 1) * average_five + last_subject :=
by
  sorry

end NUMINAMATH_CALUDE_total_subjects_l1781_178181


namespace NUMINAMATH_CALUDE_part_one_part_two_l1781_178151

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 2| - |2*x - a|

-- Part I
theorem part_one :
  {x : ℝ | f 3 x > 0} = {x : ℝ | 1/3 < x ∧ x < 5} :=
sorry

-- Part II
theorem part_two :
  ∀ a : ℝ, (∀ x : ℝ, x ≥ 0 → f a x < 3) → a < 2 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1781_178151


namespace NUMINAMATH_CALUDE_juice_left_in_cup_l1781_178119

theorem juice_left_in_cup (consumed : Rat) (h : consumed = 4/6) :
  1 - consumed = 2/6 ∨ 1 - consumed = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_juice_left_in_cup_l1781_178119


namespace NUMINAMATH_CALUDE_max_digit_sum_two_digit_primes_l1781_178156

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

theorem max_digit_sum_two_digit_primes :
  ∃ (p : ℕ), is_two_digit p ∧ is_prime p ∧
    digit_sum p = 17 ∧
    ∀ (q : ℕ), is_two_digit q → is_prime q → digit_sum q ≤ 17 :=
sorry

end NUMINAMATH_CALUDE_max_digit_sum_two_digit_primes_l1781_178156


namespace NUMINAMATH_CALUDE_reading_time_difference_l1781_178196

/-- Proves that given Xanthia's and Molly's reading speeds and a book's page count,
    the difference in reading time is 180 minutes. -/
theorem reading_time_difference
  (xanthia_speed : ℕ) -- Xanthia's reading speed in pages per hour
  (molly_speed : ℕ) -- Molly's reading speed in pages per hour
  (book_pages : ℕ) -- Number of pages in the book
  (h1 : xanthia_speed = 120)
  (h2 : molly_speed = 60)
  (h3 : book_pages = 360) :
  (book_pages / molly_speed - book_pages / xanthia_speed) * 60 = 180 :=
by
  sorry

#check reading_time_difference

end NUMINAMATH_CALUDE_reading_time_difference_l1781_178196


namespace NUMINAMATH_CALUDE_initial_working_hours_l1781_178136

/-- Given the following conditions:
  - 75 men initially working
  - Initial depth dug: 50 meters
  - New depth to dig: 70 meters
  - New working hours: 6 hours/day
  - 65 extra men added
Prove that the initial working hours H satisfy the equation:
  75 * H * 50 = (75 + 65) * 6 * 70
-/
theorem initial_working_hours (H : ℝ) : 75 * H * 50 = (75 + 65) * 6 * 70 := by
  sorry

end NUMINAMATH_CALUDE_initial_working_hours_l1781_178136
