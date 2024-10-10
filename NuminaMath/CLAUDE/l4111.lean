import Mathlib

namespace inequality_solution_set_l4111_411107

theorem inequality_solution_set (x : ℝ) : 
  (x * (1 - 3 * x) > 0) ↔ (x > 0 ∧ x < 1/3) := by sorry

end inequality_solution_set_l4111_411107


namespace polynomial_degree_product_l4111_411167

-- Define the polynomials
def p (x : ℝ) := 5*x^3 - 4*x + 7
def q (x : ℝ) := 2*x^2 + 9

-- State the theorem
theorem polynomial_degree_product : 
  Polynomial.degree ((Polynomial.monomial 0 1 + Polynomial.X)^10 * (Polynomial.monomial 0 1 + Polynomial.X)^5) = 40 := by
  sorry


end polynomial_degree_product_l4111_411167


namespace bob_oyster_shucking_l4111_411161

/-- Given that Bob can shuck 10 oysters in 5 minutes, this theorem proves
    that he can shuck 240 oysters in 2 hours. -/
theorem bob_oyster_shucking (bob_rate : ℕ) (bob_time : ℕ) (total_time : ℕ) :
  bob_rate = 10 →
  bob_time = 5 →
  total_time = 120 →
  (total_time / bob_time) * bob_rate = 240 :=
by
  sorry

#check bob_oyster_shucking

end bob_oyster_shucking_l4111_411161


namespace glove_probability_l4111_411179

/-- The probability of picking one left-handed glove and one right-handed glove -/
theorem glove_probability (left_gloves right_gloves : ℕ) 
  (h1 : left_gloves = 12) 
  (h2 : right_gloves = 10) : 
  (left_gloves * right_gloves : ℚ) / (Nat.choose (left_gloves + right_gloves) 2) = 120 / 231 := by
  sorry

end glove_probability_l4111_411179


namespace characterization_of_finite_sets_l4111_411114

def ClosedUnderAbsoluteSum (X : Set ℝ) : Prop :=
  ∀ x ∈ X, x + |x| ∈ X

theorem characterization_of_finite_sets (X : Set ℝ) 
  (h_nonempty : X.Nonempty) (h_finite : X.Finite) (h_closed : ClosedUnderAbsoluteSum X) :
  ∃ F : Set ℝ, F.Finite ∧ (∀ x ∈ F, x < 0) ∧ X = F ∪ {0} :=
sorry

end characterization_of_finite_sets_l4111_411114


namespace min_value_sum_squares_l4111_411170

theorem min_value_sum_squares (x y : ℝ) (h : x^2 + 2*x*y - 3*y^2 = 1) :
  ∃ (min : ℝ), (∀ (a b : ℝ), a^2 + 2*a*b - 3*b^2 = 1 → a^2 + b^2 ≥ min) ∧
  min = (Real.sqrt 5 + 1) / 4 := by
sorry

end min_value_sum_squares_l4111_411170


namespace exponent_rules_l4111_411116

theorem exponent_rules (a : ℝ) : (a^3 * a^2 = a^5) ∧ (a^6 / a^2 = a^4) := by
  sorry

end exponent_rules_l4111_411116


namespace square_diff_divided_by_24_l4111_411175

theorem square_diff_divided_by_24 : (145^2 - 121^2) / 24 = 266 := by sorry

end square_diff_divided_by_24_l4111_411175


namespace complex_expression_simplification_l4111_411132

theorem complex_expression_simplification :
  (-5 + 3 * Complex.I) - (2 - 7 * Complex.I) * 3 = -11 + 24 * Complex.I := by
  sorry

end complex_expression_simplification_l4111_411132


namespace notebook_cost_l4111_411109

theorem notebook_cost (total_students : Nat) (buyers : Nat) (total_cost : ℚ) 
  (h1 : total_students = 36)
  (h2 : buyers > total_students / 2)
  (h3 : ∃ (notebooks_per_student : Nat) (cost_per_notebook : ℚ),
    notebooks_per_student > 0 ∧
    cost_per_notebook > notebooks_per_student ∧
    buyers * notebooks_per_student * cost_per_notebook = total_cost)
  (h4 : total_cost = 2664 / 100) :
  ∃ (notebooks_per_student : Nat) (cost_per_notebook : ℚ),
    notebooks_per_student > 0 ∧
    cost_per_notebook > notebooks_per_student ∧
    buyers * notebooks_per_student * cost_per_notebook = total_cost ∧
    cost_per_notebook = 37 / 100 :=
by sorry

end notebook_cost_l4111_411109


namespace triangle_area_l4111_411115

/-- A triangle with sides in ratio 3:4:5 and perimeter 60 has area 150 -/
theorem triangle_area (a b c : ℝ) (h_ratio : (a, b, c) = (3, 4, 5)) 
  (h_perimeter : a + b + c = 60) : 
  (1/2) * a * b = 150 := by
sorry

end triangle_area_l4111_411115


namespace supplementary_angles_ratio_l4111_411112

theorem supplementary_angles_ratio (a b : ℝ) : 
  a + b = 180 →  -- The angles are supplementary (sum to 180°)
  a / b = 4 / 5 →  -- The ratio of the angles is 4:5
  b = 100 :=  -- The larger angle is 100°
by
  sorry

end supplementary_angles_ratio_l4111_411112


namespace park_conditions_l4111_411134

-- Define the conditions for the park
structure ParkConditions where
  temperature : ℝ
  windy : Prop

-- Define when the park is ideal for picnicking
def isIdealForPicnicking (conditions : ParkConditions) : Prop :=
  conditions.temperature ≥ 70 ∧ ¬conditions.windy

-- Theorem statement
theorem park_conditions (conditions : ParkConditions) :
  ¬(isIdealForPicnicking conditions) →
  (conditions.temperature < 70 ∨ conditions.windy) := by
  sorry

end park_conditions_l4111_411134


namespace car_fuel_usage_l4111_411182

/-- Proves that a car traveling at 40 miles per hour for 5 hours, with a fuel efficiency
    of 1 gallon per 40 miles and starting with a full 12-gallon tank, uses 5/12 of its fuel. -/
theorem car_fuel_usage (speed : ℝ) (time : ℝ) (fuel_efficiency : ℝ) (tank_capacity : ℝ) :
  speed = 40 →
  time = 5 →
  fuel_efficiency = 40 →
  tank_capacity = 12 →
  (speed * time / fuel_efficiency) / tank_capacity = 5 / 12 := by
  sorry

end car_fuel_usage_l4111_411182


namespace palmer_photos_l4111_411174

theorem palmer_photos (initial_photos : ℕ) (first_week : ℕ) (final_total : ℕ) :
  initial_photos = 100 →
  first_week = 50 →
  final_total = 380 →
  final_total - initial_photos - first_week - 2 * first_week = 130 := by
sorry

end palmer_photos_l4111_411174


namespace price_two_birdhouses_is_32_l4111_411138

/-- The price Denver charges for two birdhouses -/
def price_two_birdhouses : ℚ :=
  let pieces_per_birdhouse : ℕ := 7
  let price_per_piece : ℚ := 3/2  -- $1.50 as a rational number
  let profit_per_birdhouse : ℚ := 11/2  -- $5.50 as a rational number
  let cost_per_birdhouse : ℚ := pieces_per_birdhouse * price_per_piece
  let price_per_birdhouse : ℚ := cost_per_birdhouse + profit_per_birdhouse
  2 * price_per_birdhouse

/-- Theorem stating that the price for two birdhouses is $32.00 -/
theorem price_two_birdhouses_is_32 : price_two_birdhouses = 32 := by
  sorry

end price_two_birdhouses_is_32_l4111_411138


namespace max_n_is_14_l4111_411188

/-- A function that divides a list of integers into two groups -/
def divide_into_groups (n : ℕ) : (List ℕ) × (List ℕ) := sorry

/-- Predicate to check if a list contains no pair of numbers that sum to a perfect square -/
def no_square_sum (l : List ℕ) : Prop := sorry

/-- Predicate to check if two lists have no common elements -/
def no_common_elements (l1 l2 : List ℕ) : Prop := sorry

/-- The main theorem stating that 14 is the maximum value of n satisfying the conditions -/
theorem max_n_is_14 : 
  ∀ n : ℕ, n > 14 → 
  ¬∃ (g1 g2 : List ℕ), 
    (divide_into_groups n = (g1, g2)) ∧ 
    (no_square_sum g1) ∧ 
    (no_square_sum g2) ∧ 
    (no_common_elements g1 g2) ∧ 
    (g1.length + g2.length = n) ∧
    (∀ i : ℕ, i ∈ g1 ∨ i ∈ g2 ↔ 1 ≤ i ∧ i ≤ n) :=
sorry

end max_n_is_14_l4111_411188


namespace max_tetrahedron_volume_cube_sphere_l4111_411133

/-- The maximum volume of a tetrahedron formed by a point on the circumscribed sphere
    of a cube and one face of the cube, given the cube's edge length. -/
theorem max_tetrahedron_volume_cube_sphere (edge_length : ℝ) (h : edge_length = 2) :
  let sphere_radius : ℝ := Real.sqrt 3 * edge_length / 2
  let max_height : ℝ := sphere_radius + edge_length / 2
  let base_area : ℝ := edge_length ^ 2
  ∃ (volume : ℝ), volume = base_area * max_height / 3 ∧ 
                  volume = (4 * (1 + Real.sqrt 3)) / 3 :=
by sorry

end max_tetrahedron_volume_cube_sphere_l4111_411133


namespace pirate_treasure_l4111_411183

theorem pirate_treasure (x : ℕ) : x > 0 → (
  let paul_coins := x
  let pete_coins := x^2
  paul_coins + pete_coins = 12
) ↔ (
  -- Pete's coins follow the pattern 1, 3, 5, ..., (2x-1)
  pete_coins = x^2 ∧
  -- Paul receives x coins in total
  paul_coins = x ∧
  -- Pete has exactly three times as many coins as Paul
  pete_coins = 3 * paul_coins ∧
  -- All coins are distributed (implied by the other conditions)
  True
) := by sorry

end pirate_treasure_l4111_411183


namespace f_properties_l4111_411119

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 
  Real.sqrt 3 * Real.sin (ω * x + φ) - Real.cos (ω * x + φ)

def isEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def adjacentSymmetryDistance (f : ℝ → ℝ) (d : ℝ) : Prop :=
  ∀ x, f (x + d) = f x

theorem f_properties (ω φ : ℝ) 
  (h_φ : -π/2 < φ ∧ φ < 0) 
  (h_ω : ω > 0) 
  (h_even : isEven (f ω φ))
  (h_symmetry : adjacentSymmetryDistance (f ω φ) (π/2)) :
  f ω φ (π/24) = -(Real.sqrt 6 + Real.sqrt 2)/2 ∧
  ∃ g : ℝ → ℝ, g = fun x ↦ -2 * Real.cos (x/2 - π/3) := by
sorry

end f_properties_l4111_411119


namespace draw_balls_theorem_l4111_411160

def total_balls : ℕ := 10
def red_balls : ℕ := 4
def white_balls : ℕ := 6
def red_score : ℕ := 2
def white_score : ℕ := 1
def balls_to_draw : ℕ := 4
def min_score : ℕ := 5

/-- The number of ways to draw 4 balls from a bag containing 4 red balls and 6 white balls,
    where red balls score 2 points and white balls score 1 point,
    such that the total score is not less than 5 points. -/
def ways_to_draw : ℕ := 195

theorem draw_balls_theorem :
  ways_to_draw = 195 :=
sorry

end draw_balls_theorem_l4111_411160


namespace union_and_intersection_when_a_eq_4_subset_condition_l4111_411159

-- Define the universal set U as the real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | 1 < x ∧ x ≤ 4}

-- Define set B with parameter a
def B (a : ℝ) : Set ℝ := {x | 6 - a < x ∧ x < 2*a - 1}

-- Theorem for question 1
theorem union_and_intersection_when_a_eq_4 :
  (A ∪ B 4) = {x | 1 < x ∧ x < 7} ∧
  (B 4 ∩ (U \ A)) = {x | 4 < x ∧ x < 7} :=
sorry

-- Theorem for question 2
theorem subset_condition :
  ∀ a : ℝ, A ⊆ B a ↔ a ≥ 5 :=
sorry

end union_and_intersection_when_a_eq_4_subset_condition_l4111_411159


namespace circle_xy_bounds_l4111_411158

/-- The circle defined by x² + y² - 4x - 4y + 6 = 0 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 4*p.1 - 4*p.2 + 6 = 0}

/-- The product function xy for points on the circle -/
def xy_product (p : ℝ × ℝ) : ℝ := p.1 * p.2

theorem circle_xy_bounds :
  (∃ p ∈ Circle, ∀ q ∈ Circle, xy_product q ≤ xy_product p) ∧
  (∃ p ∈ Circle, ∀ q ∈ Circle, xy_product p ≤ xy_product q) ∧
  (∃ p ∈ Circle, xy_product p = 9) ∧
  (∃ p ∈ Circle, xy_product p = 1) :=
by sorry

end circle_xy_bounds_l4111_411158


namespace congruence_mod_nine_l4111_411111

theorem congruence_mod_nine : ∃! n : ℤ, 0 ≤ n ∧ n < 9 ∧ -1234 ≡ n [ZMOD 9] ∧ n = 1 := by
  sorry

end congruence_mod_nine_l4111_411111


namespace perfect_square_trinomial_l4111_411137

theorem perfect_square_trinomial (k : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 - k*x + 9 = (x - a)^2) → (k = 6 ∨ k = -6) := by
  sorry

end perfect_square_trinomial_l4111_411137


namespace gcd_polynomial_and_x_l4111_411110

theorem gcd_polynomial_and_x (x : ℤ) (h : ∃ k : ℤ, x = 23478 * k) :
  Int.gcd ((2*x+3)*(7*x+2)*(13*x+7)*(x+13)) x = 546 := by
  sorry

end gcd_polynomial_and_x_l4111_411110


namespace sarah_apples_to_teachers_l4111_411195

def apples_given_to_teachers (initial_apples : ℕ) (locker_apples : ℕ) (friend_apples : ℕ) 
  (classmate_apples : ℕ) (traded_apples : ℕ) (close_friends : ℕ) (eaten_apples : ℕ) 
  (final_apples : ℕ) : ℕ :=
  initial_apples - locker_apples - friend_apples - classmate_apples - traded_apples - 
  close_friends - eaten_apples - final_apples

theorem sarah_apples_to_teachers :
  apples_given_to_teachers 50 10 3 8 4 5 1 4 = 15 := by
  sorry

end sarah_apples_to_teachers_l4111_411195


namespace equation_solution_l4111_411157

theorem equation_solution (x : ℝ) (hx : x ≠ 0) : 
  (9 * x)^18 = (27 * x)^9 + 81 * x ↔ x = 1/3 := by sorry

end equation_solution_l4111_411157


namespace midpoint_trajectory_equation_l4111_411153

/-- The equation of the trajectory of the midpoint M of line segment PQ, where P is on the parabola y = x^2 + 1 and Q is (0, 1) -/
theorem midpoint_trajectory_equation :
  ∀ (x y a b : ℝ),
  y = x^2 + 1 →  -- P (x, y) is on the parabola y = x^2 + 1
  a = x / 2 →    -- M (a, b) is the midpoint of PQ, so a = x/2
  b = (y + 1) / 2 →  -- and b = (y + 1)/2
  b = 2 * a^2 + 1 :=  -- The equation of the trajectory of M is y = 2x^2 + 1
by
  sorry

end midpoint_trajectory_equation_l4111_411153


namespace molly_age_when_stopped_l4111_411151

/-- Calculates the age when a person stops riding their bike daily, given their starting age,
    daily riding distance, total distance covered, and days in a year. -/
def age_when_stopped (starting_age : ℕ) (daily_distance : ℕ) (total_distance : ℕ) (days_per_year : ℕ) : ℕ :=
  starting_age + (total_distance / daily_distance) / days_per_year

/-- Theorem stating that given the specified conditions, Molly's age when she stopped riding
    her bike daily is 16 years old. -/
theorem molly_age_when_stopped :
  let starting_age : ℕ := 13
  let daily_distance : ℕ := 3
  let total_distance : ℕ := 3285
  let days_per_year : ℕ := 365
  age_when_stopped starting_age daily_distance total_distance days_per_year = 16 := by
  sorry


end molly_age_when_stopped_l4111_411151


namespace triangle_area_proof_l4111_411173

theorem triangle_area_proof (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  a * Real.sin B = Real.sqrt 3 * b * Real.cos A →
  A = π / 3 →
  Real.cos C = 1 / 3 →
  c = 4 * Real.sqrt 2 →
  (1 / 2) * a * c * Real.sin B = 4 * Real.sqrt 3 + 3 * Real.sqrt 2 := by
  sorry


end triangle_area_proof_l4111_411173


namespace partial_fraction_sum_zero_l4111_411123

theorem partial_fraction_sum_zero (x : ℝ) (A B C D E F : ℝ) :
  (1 : ℝ) / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) =
  A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5) →
  A + B + C + D + E + F = 0 := by
sorry

end partial_fraction_sum_zero_l4111_411123


namespace largest_prime_divisor_to_test_l4111_411117

theorem largest_prime_divisor_to_test (n : ℕ) (h : 1900 ≤ n ∧ n ≤ 1950) :
  (∀ p : ℕ, p.Prime → p ≤ 43 → ¬(p ∣ n)) → n.Prime :=
sorry

end largest_prime_divisor_to_test_l4111_411117


namespace ellipse_equation_l4111_411124

/-- An ellipse with parametric equations x = 5cos(α) and y = 3sin(α) has the general equation x²/25 + y²/9 = 1 -/
theorem ellipse_equation (α : ℝ) (x y : ℝ) (h1 : x = 5 * Real.cos α) (h2 : y = 3 * Real.sin α) : 
  x^2 / 25 + y^2 / 9 = 1 := by
sorry

end ellipse_equation_l4111_411124


namespace multiplication_subtraction_difference_l4111_411169

/-- Given that x = 5, prove that the number n satisfying 3x = (16 - x) + n is equal to 4 -/
theorem multiplication_subtraction_difference (x : ℝ) (n : ℝ) : 
  x = 5 → 3 * x = (16 - x) + n → n = 4 := by
  sorry

end multiplication_subtraction_difference_l4111_411169


namespace vegetable_bins_l4111_411108

theorem vegetable_bins (soup_bins pasta_bins total_bins : Real) 
  (h1 : soup_bins = 0.125)
  (h2 : pasta_bins = 0.5)
  (h3 : total_bins = 0.75) :
  total_bins - soup_bins - pasta_bins = 0.625 := by
  sorry

end vegetable_bins_l4111_411108


namespace shelf_capacity_l4111_411129

/-- The total capacity of jars on a shelf. -/
def total_capacity (total_jars small_jars : ℕ) (small_capacity large_capacity : ℕ) : ℕ :=
  small_jars * small_capacity + (total_jars - small_jars) * large_capacity

/-- Theorem stating the total capacity of jars on the shelf. -/
theorem shelf_capacity : total_capacity 100 62 3 5 = 376 := by
  sorry

end shelf_capacity_l4111_411129


namespace mint_problem_solvable_l4111_411140

/-- Represents a set of coin denominations. -/
def CoinSet := Finset ℕ

/-- Checks if a given amount can be represented using at most 8 coins from the set. -/
def canRepresent (coins : CoinSet) (amount : ℕ) : Prop :=
  ∃ (representation : Finset ℕ), 
    representation.card ≤ 8 ∧ 
    (representation.sum (λ x => x * (coins.filter (λ c => c = x)).card)) = amount

/-- The main theorem stating that there exists a set of 12 coin denominations
    that can represent all amounts from 1 to 6543 using at most 8 coins. -/
theorem mint_problem_solvable : 
  ∃ (coins : CoinSet), 
    coins.card = 12 ∧ 
    ∀ (amount : ℕ), 1 ≤ amount ∧ amount ≤ 6543 → canRepresent coins amount :=
by
  sorry


end mint_problem_solvable_l4111_411140


namespace floor_plus_self_equal_five_l4111_411141

theorem floor_plus_self_equal_five (y : ℝ) : ⌊y⌋ + y = 5 → y = 7/3 := by
  sorry

end floor_plus_self_equal_five_l4111_411141


namespace race_first_part_length_l4111_411176

theorem race_first_part_length 
  (total_length : ℝ)
  (second_part : ℝ)
  (third_part : ℝ)
  (last_part : ℝ)
  (h1 : total_length = 74.5)
  (h2 : second_part = 21.5)
  (h3 : third_part = 21.5)
  (h4 : last_part = 16) :
  total_length - (second_part + third_part + last_part) = 15.5 := by
sorry

end race_first_part_length_l4111_411176


namespace odell_kershaw_passing_l4111_411136

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℝ  -- speed in m/min
  radius : ℝ  -- radius of the lane in meters
  direction : ℤ  -- 1 for clockwise, -1 for counterclockwise

/-- Calculates the number of times two runners pass each other on a circular track -/
def passingCount (runner1 runner2 : Runner) (totalTime : ℝ) : ℕ :=
  sorry

theorem odell_kershaw_passing :
  let odell : Runner := { speed := 240, radius := 40, direction := 1 }
  let kershaw : Runner := { speed := 320, radius := 55, direction := -1 }
  let totalTime : ℝ := 40
  passingCount odell kershaw totalTime = 75 := by
  sorry

end odell_kershaw_passing_l4111_411136


namespace circle_on_parabola_through_focus_l4111_411198

/-- A circle with center on the parabola y² = 8x and tangent to x + 2 = 0 passes through (2, 0) -/
theorem circle_on_parabola_through_focus (x y : ℝ) :
  y^2 = 8*x →  -- center (x, y) is on the parabola
  (x + 2)^2 + y^2 = (x + 4)^2 →  -- circle is tangent to x + 2 = 0
  (2 - x)^2 + y^2 = (x + 4)^2 :=  -- circle passes through (2, 0)
by sorry

end circle_on_parabola_through_focus_l4111_411198


namespace function_composition_l4111_411121

theorem function_composition (f : ℝ → ℝ) (x : ℝ) :
  (∀ x, f x = x^2 + 2*x) →
  f (2*x + 1) = 4*x^2 + 8*x + 3 := by
  sorry

end function_composition_l4111_411121


namespace batsman_average_increase_l4111_411154

/-- Represents a batsman's score statistics -/
structure BatsmanStats where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an inning -/
def newAverage (stats : BatsmanStats) (runsScored : ℕ) : ℚ :=
  (stats.totalRuns + runsScored : ℚ) / (stats.innings + 1)

/-- Theorem: If a batsman's average increases by 3 after scoring 84 in the 17th inning, the new average is 36 -/
theorem batsman_average_increase (stats : BatsmanStats) 
    (h1 : stats.innings = 16)
    (h2 : newAverage stats 84 = stats.average + 3) :
    newAverage stats 84 = 36 := by
  sorry


end batsman_average_increase_l4111_411154


namespace short_bushes_count_l4111_411184

/-- The number of short bushes initially in the park -/
def initial_short_bushes : ℕ := 37

/-- The number of short bushes planted by workers -/
def planted_short_bushes : ℕ := 20

/-- The total number of short bushes after planting -/
def total_short_bushes : ℕ := 57

/-- Theorem stating that the initial number of short bushes plus the planted ones equals the total -/
theorem short_bushes_count : 
  initial_short_bushes + planted_short_bushes = total_short_bushes := by
  sorry

end short_bushes_count_l4111_411184


namespace triangle_sine_identity_l4111_411163

theorem triangle_sine_identity (A B C : Real) (h : A + B + C = Real.pi) :
  Real.sin (2 * A) + Real.sin (2 * B) + Real.sin (2 * C) = 4 * Real.sin A * Real.sin B * Real.sin C :=
by sorry

end triangle_sine_identity_l4111_411163


namespace cube_volume_problem_l4111_411103

theorem cube_volume_problem (a : ℝ) : 
  (a > 0) →
  ((a + 2) * (a - 2) * a = a^3 - 12) →
  (a^3 = 27) := by
sorry

end cube_volume_problem_l4111_411103


namespace sum_of_coefficients_l4111_411193

/-- A quadratic function with roots at 2 and -4, and a minimum value of 32 -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  root1 : a * 2^2 + b * 2 + c = 0
  root2 : a * (-4)^2 + b * (-4) + c = 0
  min_value : ∀ x, a * x^2 + b * x + c ≥ 32

theorem sum_of_coefficients (f : QuadraticFunction) : f.a + f.b + f.c = 160 / 9 := by
  sorry

end sum_of_coefficients_l4111_411193


namespace brown_eyed_brunettes_l4111_411128

theorem brown_eyed_brunettes (total : ℕ) (blue_eyed_blondes : ℕ) (brunettes : ℕ) (brown_eyed : ℕ) :
  total = 60 →
  blue_eyed_blondes = 16 →
  brunettes = 36 →
  brown_eyed = 25 →
  (total - brunettes) - blue_eyed_blondes + brown_eyed = total →
  brown_eyed - ((total - brunettes) - blue_eyed_blondes) = 17 := by
  sorry

#check brown_eyed_brunettes

end brown_eyed_brunettes_l4111_411128


namespace largest_n_for_arithmetic_sequences_l4111_411125

theorem largest_n_for_arithmetic_sequences (a b : ℕ → ℕ) : 
  (∀ n, ∃ x y : ℤ, a n = 1 + (n - 1) * x ∧ b n = 1 + (n - 1) * y) →  -- arithmetic sequences
  (a 1 = 1 ∧ b 1 = 1) →  -- first terms are 1
  (a 2 ≤ b 2) →  -- a_2 ≤ b_2
  (∃ n, a n * b n = 1540) →  -- product condition
  (∀ n, a n * b n = 1540 → n ≤ 512) ∧  -- 512 is an upper bound
  (∃ n, a n * b n = 1540 ∧ n = 512) -- 512 is achievable
  := by sorry

end largest_n_for_arithmetic_sequences_l4111_411125


namespace student_earnings_theorem_l4111_411186

/-- Calculates the monthly earnings of a student working as a courier after tax deduction -/
def monthly_earnings_after_tax (daily_rate : ℝ) (days_per_week : ℕ) (weeks_per_month : ℕ) (tax_rate : ℝ) : ℝ :=
  let gross_monthly_earnings := daily_rate * (days_per_week : ℝ) * (weeks_per_month : ℝ)
  let tax_amount := gross_monthly_earnings * tax_rate
  gross_monthly_earnings - tax_amount

/-- Theorem stating that the monthly earnings of the student after tax is 17400 rubles -/
theorem student_earnings_theorem :
  monthly_earnings_after_tax 1250 4 4 0.13 = 17400 := by
  sorry

end student_earnings_theorem_l4111_411186


namespace tempo_premium_calculation_l4111_411105

/-- Calculate the premium amount for an insured tempo --/
theorem tempo_premium_calculation (original_value : ℝ) (insurance_ratio : ℝ) (premium_rate : ℝ) :
  original_value = 87500 →
  insurance_ratio = 4/5 →
  premium_rate = 0.013 →
  (original_value * insurance_ratio * premium_rate : ℝ) = 910 := by
  sorry

end tempo_premium_calculation_l4111_411105


namespace triangle_angle_and_area_l4111_411120

/-- Given a triangle ABC with angle A and vectors m and n, prove the measure of A and the area of the triangle -/
theorem triangle_angle_and_area 
  (A B C : ℝ) 
  (m n : ℝ × ℝ) 
  (h1 : m = (Real.sin (A/2), Real.cos (A/2)))
  (h2 : n = (Real.cos (A/2), -Real.cos (A/2)))
  (h3 : 2 * (m.1 * n.1 + m.2 * n.2) + Real.sqrt (m.1^2 + m.2^2) = Real.sqrt 2 / 2)
  (h4 : Real.cos A = 1 / (Real.sin A)) :
  A = 5 * Real.pi / 12 ∧ 
  (Real.sin A) / 2 = (2 + Real.sqrt 3) / 2 := by
  sorry

end triangle_angle_and_area_l4111_411120


namespace sprinkles_problem_l4111_411100

theorem sprinkles_problem (initial_cans remaining_cans subtracted_number : ℕ) : 
  initial_cans = 12 →
  remaining_cans = 3 →
  remaining_cans = initial_cans / 2 - subtracted_number →
  subtracted_number = 3 := by
  sorry

end sprinkles_problem_l4111_411100


namespace snail_reaches_tree_in_26_days_l4111_411192

/-- The number of days it takes for a snail to reach a tree given its daily movement pattern -/
def snail_journey_days (s l₁ l₂ : ℕ) : ℕ :=
  let daily_progress := l₁ - l₂
  let days_to_reach_near := (s - l₁) / daily_progress
  days_to_reach_near + 1

/-- Theorem stating that the snail reaches the tree in 26 days under the given conditions -/
theorem snail_reaches_tree_in_26_days :
  snail_journey_days 30 5 4 = 26 := by
  sorry

end snail_reaches_tree_in_26_days_l4111_411192


namespace max_matches_theorem_l4111_411139

/-- The maximum number of matches in a table tennis tournament -/
def max_matches : ℕ := 120

/-- Represents the number of players in each team -/
structure TeamSizes where
  x : ℕ
  y : ℕ
  z : ℕ

/-- Calculates the total number of matches given team sizes -/
def calculate_matches (teams : TeamSizes) : ℕ :=
  teams.x * teams.y + teams.y * teams.z + teams.x * teams.z

/-- Theorem stating the maximum number of matches -/
theorem max_matches_theorem :
  ∀ (teams : TeamSizes),
  teams.x + teams.y + teams.z = 19 →
  calculate_matches teams ≤ max_matches :=
by sorry

end max_matches_theorem_l4111_411139


namespace susan_missed_pay_l4111_411185

/-- Calculates the pay Susan will miss during her three-week vacation --/
def missed_pay (regular_rate : ℚ) (overtime_rate : ℚ) (sunday_rate : ℚ) 
               (regular_hours : ℕ) (overtime_hours : ℕ) (sunday_hours : ℕ)
               (sunday_count : List ℕ) (vacation_days : ℕ) (workweek_days : ℕ) : ℚ :=
  let weekly_pay := regular_rate * regular_hours + overtime_rate * overtime_hours
  let sunday_pay := sunday_rate * sunday_hours * (sunday_count.sum)
  let total_pay := weekly_pay * 3 + sunday_pay
  let paid_vacation_pay := regular_rate * regular_hours * (vacation_days / workweek_days)
  total_pay - paid_vacation_pay

/-- The main theorem stating that Susan will miss $2160 during her vacation --/
theorem susan_missed_pay : 
  missed_pay 15 20 25 40 8 8 [1, 2, 0] 6 5 = 2160 := by
  sorry

end susan_missed_pay_l4111_411185


namespace no_21_length2_segments_in_10x10_grid_l4111_411126

/-- Represents a grid skeleton -/
structure GridSkeleton :=
  (size : ℕ)

/-- Represents the division of a grid skeleton into angle pieces and segments of length 2 -/
structure GridDivision :=
  (grid : GridSkeleton)
  (length2_segments : ℕ)

/-- Theorem stating that a 10x10 grid skeleton cannot have exactly 21 segments of length 2 -/
theorem no_21_length2_segments_in_10x10_grid :
  ∀ (d : GridDivision), d.grid.size = 10 → d.length2_segments ≠ 21 := by
  sorry

end no_21_length2_segments_in_10x10_grid_l4111_411126


namespace rationalize_denominator_l4111_411143

theorem rationalize_denominator : 
  (30 : ℝ) / Real.sqrt 15 = 2 * Real.sqrt 15 := by sorry

end rationalize_denominator_l4111_411143


namespace min_points_for_isosceles_l4111_411149

/-- Represents a point in the lattice of the triangle --/
structure LatticePoint where
  x : ℝ
  y : ℝ

/-- Represents the regular triangle with its lattice points --/
structure RegularTriangleLattice where
  points : List LatticePoint
  -- Ensure there are exactly 15 lattice points
  point_count : points.length = 15

/-- Checks if three points form an isosceles triangle --/
def isIsosceles (p1 p2 p3 : LatticePoint) : Prop := sorry

/-- The main theorem to be proved --/
theorem min_points_for_isosceles (t : RegularTriangleLattice) :
  ∀ (chosen : List LatticePoint),
    chosen.length ≥ 6 →
    (∀ p, p ∈ chosen → p ∈ t.points) →
    ∃ p1 p2 p3, p1 ∈ chosen ∧ p2 ∈ chosen ∧ p3 ∈ chosen ∧ isIsosceles p1 p2 p3 :=
by sorry

end min_points_for_isosceles_l4111_411149


namespace equation_solution_l4111_411152

theorem equation_solution :
  ∃ x : ℝ, (3 / x - 2 / (x - 2) = 0) ∧ (x = 6) :=
by
  sorry

end equation_solution_l4111_411152


namespace marie_erasers_l4111_411127

theorem marie_erasers (initial final lost : ℕ) 
  (h1 : lost = 42)
  (h2 : final = 53)
  (h3 : initial = final + lost) : initial = 95 := by
  sorry

end marie_erasers_l4111_411127


namespace geometric_series_ratio_l4111_411150

/-- 
Given two infinite geometric series:
1. The first series with first term a₁ = 12 and second term a₂ = 3
2. The second series with first term b₁ = 12 and second term b₂ = 3 + n
If the sum of the second series is three times the sum of the first series,
then n = 6.
-/
theorem geometric_series_ratio (n : ℝ) : 
  let a₁ : ℝ := 12
  let a₂ : ℝ := 3
  let b₁ : ℝ := 12
  let b₂ : ℝ := 3 + n
  let r₁ : ℝ := a₂ / a₁
  let r₂ : ℝ := b₂ / b₁
  let S₁ : ℝ := a₁ / (1 - r₁)
  let S₂ : ℝ := b₁ / (1 - r₂)
  S₂ = 3 * S₁ → n = 6 := by
sorry

end geometric_series_ratio_l4111_411150


namespace bike_average_speed_l4111_411106

theorem bike_average_speed (initial_reading final_reading : ℕ) (total_time : ℝ) :
  initial_reading = 2332 →
  final_reading = 2552 →
  total_time = 9 →
  (final_reading - initial_reading : ℝ) / total_time = 220 / 9 := by
  sorry

end bike_average_speed_l4111_411106


namespace coefficient_a2_l4111_411130

theorem coefficient_a2 (x a a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (x - 1)^5 + (x - 1)^3 + (x - 1) = a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a →
  a₂ = -13 := by
  sorry

end coefficient_a2_l4111_411130


namespace isosceles_right_triangle_area_l4111_411142

/-- The area of an isosceles right triangle with hypotenuse 6√2 is 18 square units -/
theorem isosceles_right_triangle_area (h : ℝ) (a : ℝ) 
  (hyp_length : h = 6 * Real.sqrt 2)
  (isosceles_right : a = h / Real.sqrt 2) : a * a / 2 = 18 := by
  sorry

end isosceles_right_triangle_area_l4111_411142


namespace fraction_to_decimal_l4111_411104

theorem fraction_to_decimal : (58 : ℚ) / 160 = (3625 : ℚ) / 10000 := by sorry

end fraction_to_decimal_l4111_411104


namespace complex_product_modulus_l4111_411101

theorem complex_product_modulus : Complex.abs (4 - 5 * Complex.I) * Complex.abs (4 + 5 * Complex.I) = 41 := by
  sorry

end complex_product_modulus_l4111_411101


namespace prob_ace_ten_king_standard_deck_l4111_411156

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (aces : ℕ)
  (tens : ℕ)
  (kings : ℕ)

/-- The probability of drawing an Ace, then a Ten, then a King without replacement -/
def prob_ace_ten_king (d : Deck) : ℚ :=
  (d.aces : ℚ) / d.total_cards *
  (d.tens : ℚ) / (d.total_cards - 1) *
  (d.kings : ℚ) / (d.total_cards - 2)

/-- Theorem stating the probability of drawing an Ace, then a Ten, then a King from a standard deck -/
theorem prob_ace_ten_king_standard_deck : 
  prob_ace_ten_king {total_cards := 52, aces := 4, tens := 4, kings := 4} = 2 / 16575 := by
  sorry


end prob_ace_ten_king_standard_deck_l4111_411156


namespace positive_numbers_inequality_l4111_411181

theorem positive_numbers_inequality (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum_squares : a^2 + b^2 + 4*c^2 = 3) :
  (a + b + 2*c ≤ 3) ∧ 
  (b = 2*c → 1/a + 1/c ≥ 3) := by
  sorry

end positive_numbers_inequality_l4111_411181


namespace expression_equality_l4111_411190

theorem expression_equality : 1 - (1 / (1 + Real.sqrt 2)) + (1 / (1 - Real.sqrt 2)) = 1 - 2 * Real.sqrt 2 := by
  sorry

end expression_equality_l4111_411190


namespace roses_in_vase_l4111_411180

theorem roses_in_vase (initial_roses : ℕ) : initial_roses + 8 = 18 → initial_roses = 10 := by
  sorry

end roses_in_vase_l4111_411180


namespace profit_percentage_l4111_411177

/-- If selling an article at 2/3 of price P results in a 10% loss,
    then selling it at price P results in a 35% profit. -/
theorem profit_percentage (P : ℝ) (P_pos : P > 0) : 
  (∃ C : ℝ, C > 0 ∧ (2/3 * P) = (0.9 * C)) →
  ((P - ((2/3 * P) / 0.9)) / ((2/3 * P) / 0.9)) * 100 = 35 := by
sorry

end profit_percentage_l4111_411177


namespace middle_part_of_proportional_division_l4111_411168

theorem middle_part_of_proportional_division (total : ℝ) (p1 p2 p3 : ℝ) :
  total = 120 →
  p1 > 0 →
  p2 > 0 →
  p3 > 0 →
  p1 / 2 = p2 / (2/3) →
  p1 / 2 = p3 / (2/9) →
  p1 + p2 + p3 = total →
  p2 = 27.6 := by
  sorry

end middle_part_of_proportional_division_l4111_411168


namespace only_prime_with_alternating_base14_rep_l4111_411131

/-- Represents a number in base-14 with alternating 1s and 0s -/
def alternatingBaseRepresentation (n : ℕ) : ℕ :=
  (14^(2*n+2) - 1) / (14^2 - 1)

/-- Checks if a number is prime -/
def isPrime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ m : ℕ, m > 1 → m < p → ¬(p % m = 0)

theorem only_prime_with_alternating_base14_rep :
  ∃! p : ℕ, isPrime p ∧ ∃ n : ℕ, alternatingBaseRepresentation n = p :=
by
  -- The unique prime is 197
  use 197
  sorry -- Proof omitted

#eval alternatingBaseRepresentation 1  -- Should evaluate to 197

end only_prime_with_alternating_base14_rep_l4111_411131


namespace magnitude_equality_not_implies_vector_equality_l4111_411187

-- Define vectors a and b in a real vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
variable (a b : V)

-- State the theorem
theorem magnitude_equality_not_implies_vector_equality :
  ∃ a b : V, (‖a‖ = 3 * ‖b‖) ∧ (a ≠ 3 • b) ∧ (a ≠ -3 • b) := by
  sorry

end magnitude_equality_not_implies_vector_equality_l4111_411187


namespace xy_range_l4111_411165

theorem xy_range (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x + 2/x + 3*y + 4/y = 10) : 
  1 ≤ x*y ∧ x*y ≤ 8/3 := by
sorry

end xy_range_l4111_411165


namespace linear_function_quadrants_l4111_411147

/-- A linear function y = (m-2)x + m + 1 passes through the first, second, and fourth quadrants
    if and only if -1 < m < m < 2 -/
theorem linear_function_quadrants (m : ℝ) :
  (∀ x y : ℝ, y = (m - 2) * x + m + 1 →
    (y > 0 ∧ x > 0) ∨ (y < 0 ∧ x < 0) ∨ (y < 0 ∧ x > 0)) ↔
  (-1 < m ∧ m < 2) :=
sorry

end linear_function_quadrants_l4111_411147


namespace banana_bread_flour_calculation_hannahs_banana_bread_flour_l4111_411196

/-- Given the ratio of flour to banana mush, bananas per cup of mush, and total bananas used,
    calculate the number of cups of flour needed. -/
theorem banana_bread_flour_calculation 
  (flour_to_mush_ratio : ℚ) 
  (bananas_per_mush : ℕ) 
  (total_bananas : ℕ) : ℚ :=
  by
  sorry

/-- Prove that for Hannah's banana bread recipe, she needs 15 cups of flour. -/
theorem hannahs_banana_bread_flour : 
  banana_bread_flour_calculation 3 4 20 = 15 :=
  by
  sorry

end banana_bread_flour_calculation_hannahs_banana_bread_flour_l4111_411196


namespace unique_integer_solution_l4111_411162

theorem unique_integer_solution :
  ∃! (x y z : ℤ), x^2 + y^2 + z^2 + 3 < x*y + 3*y + 2*z ∧ x = 1 ∧ y = 2 ∧ z = 1 :=
by sorry

end unique_integer_solution_l4111_411162


namespace equal_roots_condition_l4111_411118

theorem equal_roots_condition (m : ℝ) : 
  (∃ (x : ℝ), (x * (x - 1) - (m + 3)) / ((x - 1) * (m - 1)) = x / m ∧ 
   (∀ (y : ℝ), (y * (y - 1) - (m + 3)) / ((y - 1) * (m - 1)) = y / m → y = x)) ↔ 
  (m = -1.5 + Real.sqrt 2 ∨ m = -1.5 - Real.sqrt 2) :=
by sorry

end equal_roots_condition_l4111_411118


namespace girls_from_valley_l4111_411172

theorem girls_from_valley (total_students : ℕ) (total_boys : ℕ) (total_girls : ℕ)
  (highland_students : ℕ) (valley_students : ℕ) (highland_boys : ℕ)
  (h1 : total_students = 120)
  (h2 : total_boys = 70)
  (h3 : total_girls = 50)
  (h4 : highland_students = 45)
  (h5 : valley_students = 75)
  (h6 : highland_boys = 30)
  (h7 : total_students = total_boys + total_girls)
  (h8 : total_students = highland_students + valley_students)
  (h9 : total_boys ≥ highland_boys) :
  valley_students - (total_boys - highland_boys) = 35 := by
sorry

end girls_from_valley_l4111_411172


namespace square_sum_power_of_two_l4111_411155

theorem square_sum_power_of_two (x y z : ℕ) (h : x^2 + y^2 = 2^z) :
  ∃ n : ℕ, x = 2^n ∧ y = 2^n ∧ z = 2*n + 1 := by
sorry

end square_sum_power_of_two_l4111_411155


namespace not_regressive_a_regressive_increasing_is_arithmetic_l4111_411178

-- Definition of a regressive sequence
def IsRegressive (x : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, ∃ m : ℕ, x n + x (n + 2) - x (n + 1) = x m

-- Part 1
def a : ℕ → ℝ
  | 0 => 1
  | n + 1 => 3 * a n

theorem not_regressive_a : ¬ IsRegressive a := by sorry

-- Part 2
theorem regressive_increasing_is_arithmetic (b : ℕ → ℝ) 
  (h_regressive : IsRegressive b) (h_increasing : ∀ n : ℕ, b n < b (n + 1)) :
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) - b n = d := by sorry

end not_regressive_a_regressive_increasing_is_arithmetic_l4111_411178


namespace line_equation_proof_l4111_411164

/-- A line in the 2D plane represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem line_equation_proof (given_line : Line) (p : Point) (result_line : Line) : 
  given_line.a = 2 ∧ given_line.b = -3 ∧ given_line.c = 5 ∧
  p.x = -2 ∧ p.y = 1 ∧
  result_line.a = 2 ∧ result_line.b = -3 ∧ result_line.c = 7 →
  pointOnLine p result_line ∧ parallel given_line result_line :=
by sorry

end line_equation_proof_l4111_411164


namespace lines_parallel_perpendicular_l4111_411148

/-- Two lines in the plane -/
structure TwoLines where
  a : ℝ
  l1 : ℝ → ℝ → Prop := λ x y => x + a * y - 2 * a - 2 = 0
  l2 : ℝ → ℝ → Prop := λ x y => a * x + y - 1 - a = 0

/-- Definition of parallel lines -/
def parallel (tl : TwoLines) : Prop :=
  ∃ k : ℝ, ∀ x y, tl.l1 x y ↔ tl.l2 (x + k) y

/-- Definition of perpendicular lines -/
def perpendicular (tl : TwoLines) : Prop :=
  ∃ x1 y1 x2 y2 : ℝ,
    tl.l1 x1 y1 ∧ tl.l2 x2 y2 ∧
    (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) ≠ 0 ∧
    (x2 - x1) * (y2 - y1) = 0

/-- Main theorem -/
theorem lines_parallel_perpendicular (tl : TwoLines) :
  (parallel tl ↔ tl.a = 1) ∧ (perpendicular tl ↔ tl.a = 0) := by
  sorry

end lines_parallel_perpendicular_l4111_411148


namespace imaginary_part_of_z_l4111_411199

theorem imaginary_part_of_z (z : ℂ) (h : (3 - 4*I)*z = 5) : z.im = 4/5 := by
  sorry

end imaginary_part_of_z_l4111_411199


namespace negative_integer_product_l4111_411144

theorem negative_integer_product (a b : ℤ) : ∃ n : ℤ,
  n < 0 ∧
  n * a < 0 ∧
  -8 * b < 0 ∧
  n * a * (-8 * b) + a * b = 89 ∧
  n = -11 := by sorry

end negative_integer_product_l4111_411144


namespace max_ab_value_l4111_411189

theorem max_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b + a + b = 1) :
  a * b ≤ 3 - 2 * Real.sqrt 2 := by
sorry

end max_ab_value_l4111_411189


namespace grid_and_circles_area_sum_l4111_411113

/-- The side length of each small square in the grid -/
def smallSquareSide : ℝ := 3

/-- The number of rows in the grid -/
def gridRows : ℕ := 4

/-- The number of columns in the grid -/
def gridColumns : ℕ := 4

/-- The radius of the large circle -/
def largeCircleRadius : ℝ := 1.5 * smallSquareSide

/-- The radius of each small circle -/
def smallCircleRadius : ℝ := 0.5 * smallSquareSide

/-- The number of small circles -/
def numSmallCircles : ℕ := 3

/-- Theorem: The sum of the total grid area and the total area of the circles is 171 square cm -/
theorem grid_and_circles_area_sum : 
  (gridRows * gridColumns * smallSquareSide^2) + 
  (π * largeCircleRadius^2 + π * numSmallCircles * smallCircleRadius^2) = 171 := by
  sorry

end grid_and_circles_area_sum_l4111_411113


namespace count_fourth_powers_between_10_and_10000_l4111_411171

theorem count_fourth_powers_between_10_and_10000 : 
  (Finset.filter (fun n : ℕ => 10 ≤ n^4 ∧ n^4 ≤ 10000) (Finset.range (10000 + 1))).card = 19 :=
by sorry

end count_fourth_powers_between_10_and_10000_l4111_411171


namespace travel_distances_l4111_411135

-- Define the given constants
def train_speed : ℚ := 100
def car_speed_ratio : ℚ := 2/3
def bicycle_speed_ratio : ℚ := 1/5
def travel_time : ℚ := 1/2  -- 30 minutes in hours

-- Define the theorem
theorem travel_distances :
  let car_distance := train_speed * car_speed_ratio * travel_time
  let bicycle_distance := train_speed * bicycle_speed_ratio * travel_time
  car_distance = 100/3 ∧ bicycle_distance = 10 := by sorry

end travel_distances_l4111_411135


namespace students_walking_home_l4111_411122

theorem students_walking_home (bus car bike scooter : ℚ) : 
  bus = 1/2 → car = 1/4 → bike = 1/10 → scooter = 1/8 → 
  1 - (bus + car + bike + scooter) = 1/40 := by
sorry

end students_walking_home_l4111_411122


namespace parallelogram_area_l4111_411146

def vector_a : Fin 2 → ℝ := ![6, -8]
def vector_b : Fin 2 → ℝ := ![15, 4]

theorem parallelogram_area : 
  |vector_a 0 * vector_b 1 - vector_a 1 * vector_b 0| = 144 := by
  sorry

end parallelogram_area_l4111_411146


namespace makeup_problem_solution_l4111_411191

/-- Represents the makeup problem with given parameters -/
structure MakeupProblem where
  people_per_tube : ℕ
  total_people : ℕ
  num_tubs : ℕ

/-- Calculates the number of tubes per tub for a given makeup problem -/
def tubes_per_tub (p : MakeupProblem) : ℕ :=
  (p.total_people / p.people_per_tube) / p.num_tubs

/-- Theorem stating that for the given problem, the number of tubes per tub is 2 -/
theorem makeup_problem_solution :
  let p : MakeupProblem := ⟨3, 36, 6⟩
  tubes_per_tub p = 2 := by
  sorry

end makeup_problem_solution_l4111_411191


namespace perfect_square_trinomial_l4111_411145

theorem perfect_square_trinomial (x : ℝ) : 
  (x + 9)^2 = x^2 + 18*x + 81 ∧ 
  ∃ (a b : ℝ), (x + 9)^2 = a^2 + 2*a*b + b^2 := by
sorry

end perfect_square_trinomial_l4111_411145


namespace largest_x_sqrt_3x_eq_5x_l4111_411166

theorem largest_x_sqrt_3x_eq_5x : 
  (∃ (x : ℝ), x > 0 ∧ Real.sqrt (3 * x) = 5 * x) →
  (∀ (y : ℝ), y > 0 ∧ Real.sqrt (3 * y) = 5 * y → y ≤ 3/25) :=
by sorry

end largest_x_sqrt_3x_eq_5x_l4111_411166


namespace total_players_l4111_411197

theorem total_players (cricket : ℕ) (hockey : ℕ) (football : ℕ) (softball : ℕ)
  (h1 : cricket = 16)
  (h2 : hockey = 12)
  (h3 : football = 18)
  (h4 : softball = 13) :
  cricket + hockey + football + softball = 59 := by
  sorry

end total_players_l4111_411197


namespace calculation_proof_l4111_411102

theorem calculation_proof : Real.sqrt 2 * Real.sqrt 2 - 4 * Real.sin (π / 6) + (1 / 2)⁻¹ = 2 := by
  sorry

end calculation_proof_l4111_411102


namespace remaining_ripe_mangoes_l4111_411194

theorem remaining_ripe_mangoes (total_mangoes : ℕ) (ripe_fraction : ℚ) (eaten_fraction : ℚ) : 
  total_mangoes = 400 →
  ripe_fraction = 3/5 →
  eaten_fraction = 3/5 →
  (total_mangoes : ℚ) * ripe_fraction * (1 - eaten_fraction) = 96 := by
  sorry

end remaining_ripe_mangoes_l4111_411194
