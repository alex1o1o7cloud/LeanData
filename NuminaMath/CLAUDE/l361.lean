import Mathlib

namespace range_of_a_l361_36190

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (1 - 2*a)*x + 3*a else Real.log x

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) →
  (-1 ≤ a ∧ a < 1/2) :=
by sorry

end range_of_a_l361_36190


namespace workshop_workers_l361_36163

theorem workshop_workers (total_average : ℝ) (tech_count : ℕ) (tech_average : ℝ) (nontech_average : ℝ) :
  total_average = 6750 →
  tech_count = 7 →
  tech_average = 12000 →
  nontech_average = 6000 →
  ∃ (total_workers : ℕ), total_workers = 56 ∧ 
    total_average * total_workers = tech_average * tech_count + nontech_average * (total_workers - tech_count) :=
by
  sorry

end workshop_workers_l361_36163


namespace intersection_of_A_and_complement_of_B_l361_36168

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.sqrt x}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = -x^2}

-- Theorem statement
theorem intersection_of_A_and_complement_of_B :
  A ∩ (U \ B) = {x : ℝ | x > 0} := by sorry

end intersection_of_A_and_complement_of_B_l361_36168


namespace complex_fraction_equality_l361_36112

theorem complex_fraction_equality (a b : ℝ) :
  (1 + Complex.I) / (1 - Complex.I) = Complex.mk a b → b = 1 := by
  sorry

end complex_fraction_equality_l361_36112


namespace solve_for_x_l361_36126

-- Define the € operation
def euro (x y : ℝ) : ℝ := 3 * x * y

-- State the theorem
theorem solve_for_x (y : ℝ) (h1 : y = 3) (h2 : euro y (euro 4 x) = 540) : x = 5 := by
  sorry

end solve_for_x_l361_36126


namespace dress_discount_problem_l361_36105

theorem dress_discount_problem (P : ℝ) (D : ℝ) : 
  P - 61.2 = 4.5 → P * (1 - D) * 1.25 = 61.2 → D = 0.255 := by
sorry

end dress_discount_problem_l361_36105


namespace x_plus_y_value_l361_36185

theorem x_plus_y_value (x y : ℝ) 
  (eq1 : x + Real.cos y = 2023)
  (eq2 : x + 2023 * Real.sin y = 2022)
  (y_range : π / 2 ≤ y ∧ y ≤ π) :
  x + y = 2023 + π / 2 := by
sorry

end x_plus_y_value_l361_36185


namespace probability_heart_then_spade_or_club_l361_36127

-- Define the total number of cards in a standard deck
def total_cards : ℕ := 52

-- Define the number of hearts in a standard deck
def num_hearts : ℕ := 13

-- Define the number of spades and clubs combined in a standard deck
def num_spades_clubs : ℕ := 26

-- Theorem statement
theorem probability_heart_then_spade_or_club :
  (num_hearts / total_cards) * (num_spades_clubs / (total_cards - 1)) = 13 / 102 := by
  sorry

end probability_heart_then_spade_or_club_l361_36127


namespace circle_tangent_origin_l361_36186

-- Define the circle equation
def circle_equation (x y D E F : ℝ) : Prop :=
  x^2 + y^2 + D*x + E*y + F = 0

-- Define the tangency condition
def tangent_at_origin (D E F : ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ 
  ∀ (x y : ℝ), circle_equation x y D E F → x^2 + y^2 ≥ r^2 ∧
  circle_equation 0 0 D E F

-- Theorem statement
theorem circle_tangent_origin (D E F : ℝ) :
  tangent_at_origin D E F → D = 0 ∧ F = 0 ∧ E ≠ 0 :=
sorry

end circle_tangent_origin_l361_36186


namespace solution_set_l361_36157

def system_solution (x y : ℝ) : Prop :=
  x + y = 20 ∧ Real.log x / Real.log 4 + Real.log y / Real.log 4 = 1 + Real.log 9 / Real.log 4

theorem solution_set : 
  {(x, y) : ℝ × ℝ | system_solution x y} = {(18, 2), (2, 18)} := by sorry

end solution_set_l361_36157


namespace polynomial_evaluation_l361_36167

theorem polynomial_evaluation (x : ℕ) (h : x = 4) :
  x^4 + x^3 + x^2 + x + 1 = 341 := by
  sorry

end polynomial_evaluation_l361_36167


namespace f_value_at_2_l361_36132

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x + 8

-- State the theorem
theorem f_value_at_2 (a b : ℝ) : f a b (-2) = 10 → f a b 2 = 6 := by sorry

end f_value_at_2_l361_36132


namespace power_product_simplification_l361_36106

theorem power_product_simplification :
  3000 * (3000 ^ 2999) = 3000 ^ 3000 := by sorry

end power_product_simplification_l361_36106


namespace range_of_m_l361_36136

theorem range_of_m (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 4) 
  (h : ∀ m : ℝ, (4/x) + (16/y) > m^2 - 3*m + 5) :
  ∀ m : ℝ, -1 < m ∧ m < 4 := by
sorry

end range_of_m_l361_36136


namespace puzzle_piece_increase_l361_36176

/-- Represents the number of puzzles John buys -/
def num_puzzles : ℕ := 3

/-- Represents the number of pieces in the first puzzle -/
def first_puzzle_pieces : ℕ := 1000

/-- Represents the total number of pieces in all puzzles -/
def total_pieces : ℕ := 4000

/-- Represents the percentage increase in pieces for the second and third puzzles -/
def percentage_increase : ℚ := 50

theorem puzzle_piece_increase :
  ∃ (second_puzzle_pieces third_puzzle_pieces : ℕ),
    second_puzzle_pieces = third_puzzle_pieces ∧
    second_puzzle_pieces = first_puzzle_pieces + (percentage_increase / 100) * first_puzzle_pieces ∧
    first_puzzle_pieces + second_puzzle_pieces + third_puzzle_pieces = total_pieces :=
by sorry

#check puzzle_piece_increase

end puzzle_piece_increase_l361_36176


namespace polynomial_divisibility_l361_36198

theorem polynomial_divisibility (c : ℤ) : 
  (∃ q : Polynomial ℤ, (X^2 + X + c) * q = X^13 - X + 106) ↔ c = 2 := by
sorry

end polynomial_divisibility_l361_36198


namespace quadratic_reciprocal_roots_l361_36121

theorem quadratic_reciprocal_roots (a b c : ℝ) (ha : a ≠ 0) :
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ x * y = 1) ↔ c = a :=
sorry

end quadratic_reciprocal_roots_l361_36121


namespace no_nonzero_perfect_square_in_sequence_l361_36187

theorem no_nonzero_perfect_square_in_sequence
  (a b : ℕ → ℤ)
  (h1 : ∀ k, b k = a k + 9)
  (h2 : ∀ k, a (k + 1) = 8 * b k + 8)
  (h3 : ∃ k, a k = 1988 ∨ b k = 1988) :
  ∀ k n, n ≠ 0 → a k ≠ n^2 :=
by sorry

end no_nonzero_perfect_square_in_sequence_l361_36187


namespace red_candies_count_l361_36199

theorem red_candies_count (green blue : ℕ) (prob_blue : ℚ) (red : ℕ) : 
  green = 5 → blue = 3 → prob_blue = 1/4 → 
  (blue : ℚ) / ((green : ℚ) + (blue : ℚ) + (red : ℚ)) = prob_blue →
  red = 4 := by
  sorry

end red_candies_count_l361_36199


namespace tangent_lines_through_A_area_of_triangle_AOC_l361_36160

-- Define the circle C
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 6*y + 12 = 0

-- Define point A
def point_A : ℝ × ℝ := (3, 5)

-- Define the origin O
def origin : ℝ × ℝ := (0, 0)

-- Define the center of the circle C
def center : ℝ × ℝ := (2, 3)

-- Theorem for the tangent lines
theorem tangent_lines_through_A :
  ∃ (k : ℝ), 
    (∀ x y : ℝ, x = 3 → circle_equation x y) ∧
    (∀ x y : ℝ, y = k*x + (11/4) → circle_equation x y) ∧
    k = 3/4 :=
sorry

-- Theorem for the area of triangle AOC
theorem area_of_triangle_AOC :
  let A := point_A
  let O := origin
  let C := center
  (1/2 : ℝ) * ‖C - O‖ * ‖A - O‖ * (|C.1 * A.2 - C.2 * A.1| / (‖C - O‖ * ‖A - O‖)) = 1/2 :=
sorry

end tangent_lines_through_A_area_of_triangle_AOC_l361_36160


namespace petya_catches_up_l361_36142

/-- Represents the race scenario between Petya and Vasya -/
structure RaceScenario where
  total_distance : ℝ
  vasya_speed : ℝ
  petya_first_half_speed : ℝ

/-- Calculates Petya's required speed for the second half of the race -/
def petya_second_half_speed (race : RaceScenario) : ℝ :=
  2 * race.vasya_speed - race.petya_first_half_speed

/-- Theorem stating that Petya's speed for the second half must be 18 km/h -/
theorem petya_catches_up (race : RaceScenario) 
  (h1 : race.total_distance > 0)
  (h2 : race.vasya_speed = 12)
  (h3 : race.petya_first_half_speed = 9) :
  petya_second_half_speed race = 18 := by
  sorry

#eval petya_second_half_speed { total_distance := 100, vasya_speed := 12, petya_first_half_speed := 9 }

end petya_catches_up_l361_36142


namespace capacity_variation_l361_36192

/-- Given positive constants e, R, and r, prove that the function C(n) = en / (R + nr^2) 
    first increases and then decreases as n increases. -/
theorem capacity_variation (e R r : ℝ) (he : e > 0) (hR : R > 0) (hr : r > 0) :
  ∃ n₀ : ℝ, n₀ > 0 ∧
    (∀ n₁ n₂ : ℝ, 0 < n₁ ∧ n₁ < n₂ ∧ n₂ < n₀ → 
      (e * n₁) / (R + n₁ * r^2) < (e * n₂) / (R + n₂ * r^2)) ∧
    (∀ n₁ n₂ : ℝ, n₀ < n₁ ∧ n₁ < n₂ → 
      (e * n₁) / (R + n₁ * r^2) > (e * n₂) / (R + n₂ * r^2)) :=
sorry

end capacity_variation_l361_36192


namespace orange_balls_count_l361_36145

def ball_problem (total red blue pink orange : ℕ) : Prop :=
  total = 50 ∧
  red = 20 ∧
  blue = 10 ∧
  total = red + blue + pink + orange ∧
  pink = 3 * orange

theorem orange_balls_count :
  ∀ total red blue pink orange : ℕ,
  ball_problem total red blue pink orange →
  orange = 5 :=
by
  sorry

end orange_balls_count_l361_36145


namespace line_passes_through_fixed_point_l361_36184

/-- The line equation is of the form (a-1)x - y + 2a + 1 = 0 where a is a real number -/
def line_equation (a x y : ℝ) : Prop :=
  (a - 1) * x - y + 2 * a + 1 = 0

/-- Theorem: The line always passes through the point (-2, 3) for all real values of a -/
theorem line_passes_through_fixed_point :
  ∀ a : ℝ, line_equation a (-2) 3 :=
by sorry

end line_passes_through_fixed_point_l361_36184


namespace smallest_n_for_P_less_than_threshold_l361_36179

def P (n : ℕ) : ℚ := 3 / ((n + 3) * (n + 4))

theorem smallest_n_for_P_less_than_threshold : 
  (∃ n : ℕ, P n < 1 / 2010) ∧ 
  (∀ m : ℕ, m < 23 → P m ≥ 1 / 2010) ∧ 
  (P 23 < 1 / 2010) := by sorry

end smallest_n_for_P_less_than_threshold_l361_36179


namespace recreation_spending_ratio_l361_36159

/-- Proves that if wages decrease by 25% and recreation spending decreases from 30% to 20%,
    then the new recreation spending is 50% of the original. -/
theorem recreation_spending_ratio (original_wages : ℝ) (original_wages_positive : original_wages > 0) :
  let new_wages := 0.75 * original_wages
  let original_recreation := 0.3 * original_wages
  let new_recreation := 0.2 * new_wages
  new_recreation / original_recreation = 0.5 := by
  sorry

end recreation_spending_ratio_l361_36159


namespace sum_in_base_5_l361_36161

/-- Given a base b, returns the value of a number in base 10 -/
def toBase10 (x : ℕ) (b : ℕ) : ℕ := sorry

/-- Given a base b, returns the square of a number in base b -/
def squareInBase (x : ℕ) (b : ℕ) : ℕ := sorry

/-- Given a base b, returns the sum of three numbers in base b -/
def sumInBase (x y z : ℕ) (b : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to another base -/
def fromBase10 (x : ℕ) (b : ℕ) : ℕ := sorry

theorem sum_in_base_5 (b : ℕ) : 
  (squareInBase 14 b + squareInBase 18 b + squareInBase 20 b = toBase10 2850 b) →
  (fromBase10 (sumInBase 14 18 20 b) 5 = 62) := by
  sorry

end sum_in_base_5_l361_36161


namespace abs_neg_three_halves_l361_36162

theorem abs_neg_three_halves : |(-3/2 : ℚ)| = 3/2 := by
  sorry

end abs_neg_three_halves_l361_36162


namespace park_planting_problem_l361_36183

/-- The number of short bushes to be planted in a park -/
def short_bushes_to_plant (current_short_bushes total_short_bushes_after : ℕ) : ℕ :=
  total_short_bushes_after - current_short_bushes

/-- Theorem stating that 20 short bushes will be planted -/
theorem park_planting_problem :
  short_bushes_to_plant 37 57 = 20 := by
  sorry

end park_planting_problem_l361_36183


namespace positive_integer_equation_l361_36141

theorem positive_integer_equation (m n p : ℕ+) : 
  3 * m.val + 3 / (n.val + 1 / p.val) = 17 → p = 2 := by
  sorry

end positive_integer_equation_l361_36141


namespace cosine_properties_l361_36174

theorem cosine_properties (x : ℝ) : 
  (fun (x : ℝ) => Real.cos x) (Real.pi + x) = -(fun (x : ℝ) => Real.cos x) x ∧ 
  (fun (x : ℝ) => Real.cos x) (-x) = (fun (x : ℝ) => Real.cos x) x :=
by sorry

end cosine_properties_l361_36174


namespace optimal_plan_l361_36110

/-- Represents a sewage treatment equipment purchasing plan -/
structure Plan where
  typeA : ℕ
  typeB : ℕ

/-- Checks if a plan is valid according to the given constraints -/
def isValidPlan (p : Plan) : Prop :=
  p.typeA + p.typeB = 20 ∧
  120000 * p.typeA + 100000 * p.typeB ≤ 2300000 ∧
  240 * p.typeA + 200 * p.typeB ≥ 4500

/-- Calculates the total cost of a plan -/
def planCost (p : Plan) : ℕ :=
  120000 * p.typeA + 100000 * p.typeB

/-- Theorem stating that the optimal plan is 13 units of A and 7 units of B -/
theorem optimal_plan :
  ∃ (optimalPlan : Plan),
    isValidPlan optimalPlan ∧
    optimalPlan.typeA = 13 ∧
    optimalPlan.typeB = 7 ∧
    planCost optimalPlan = 2260000 ∧
    ∀ (p : Plan), isValidPlan p → planCost p ≥ planCost optimalPlan :=
  sorry


end optimal_plan_l361_36110


namespace slope_of_line_l361_36102

theorem slope_of_line (x y : ℝ) :
  4 * x + 6 * y = 24 → (y - 4) / x = -2 / 3 := by
  sorry

end slope_of_line_l361_36102


namespace sum_of_solutions_l361_36169

theorem sum_of_solutions (x y : ℝ) : 
  x * y = 1 ∧ x + y = 3 → ∃ (x₁ x₂ : ℝ), x₁ + x₂ = 3 ∧ 
  (x₁ * (3 - x₁) = 1 ∧ x₁ + (3 - x₁) = 3) ∧
  (x₂ * (3 - x₂) = 1 ∧ x₂ + (3 - x₂) = 3) :=
by
  sorry

end sum_of_solutions_l361_36169


namespace quadratic_coefficient_l361_36180

/-- A quadratic function with vertex (2, 0) passing through (0, -50) has a = -12.5 -/
theorem quadratic_coefficient (a b c : ℝ) : 
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →
  (2, 0) = (2, a * 2^2 + b * 2 + c) →
  (0, -50) = (0, a * 0^2 + b * 0 + c) →
  a = -12.5 := by
sorry

end quadratic_coefficient_l361_36180


namespace max_height_is_100_l361_36118

-- Define the height function
def h (t : ℝ) : ℝ := -20 * t^2 + 80 * t + 20

-- Theorem stating that the maximum height is 100
theorem max_height_is_100 : 
  ∃ (t_max : ℝ), ∀ (t : ℝ), h t ≤ h t_max ∧ h t_max = 100 := by
  sorry

end max_height_is_100_l361_36118


namespace b_age_is_twelve_l361_36148

/-- Given three people a, b, and c, where a is two years older than b, b is twice as old as c, 
    and the sum of their ages is 32, prove that b is 12 years old. -/
theorem b_age_is_twelve (a b c : ℕ) 
    (h1 : a = b + 2) 
    (h2 : b = 2 * c) 
    (h3 : a + b + c = 32) : 
  b = 12 := by sorry

end b_age_is_twelve_l361_36148


namespace intersection_M_N_l361_36171

/-- Set M is defined as {x | 0 ≤ x ≤ 1} -/
def M : Set ℝ := {x | 0 ≤ x ∧ x ≤ 1}

/-- Set N is defined as {x | |x| ≥ 1} -/
def N : Set ℝ := {x | abs x ≥ 1}

/-- The intersection of sets M and N is equal to the set containing only 1 -/
theorem intersection_M_N : M ∩ N = {1} := by sorry

end intersection_M_N_l361_36171


namespace diophantine_equation_solvable_l361_36178

theorem diophantine_equation_solvable (n : ℤ) :
  ∃ (x y z : ℤ), x^3 + 2*y^2 + 4*z = n := by sorry

end diophantine_equation_solvable_l361_36178


namespace initial_time_is_six_hours_l361_36196

/-- Proves that the initial time to cover 288 km is 6 hours -/
theorem initial_time_is_six_hours 
  (distance : ℝ) 
  (new_speed : ℝ) 
  (time_ratio : ℝ) :
  distance = 288 →
  new_speed = 32 →
  time_ratio = 3 / 2 →
  ∃ (initial_time : ℝ), 
    initial_time = 6 ∧ 
    distance = new_speed * (time_ratio * initial_time) :=
by sorry

end initial_time_is_six_hours_l361_36196


namespace probability_three_defective_shipment_l361_36181

/-- The probability of selecting three defective smartphones from a shipment -/
def probability_three_defective (total : ℕ) (defective : ℕ) : ℚ :=
  (defective : ℚ) / total *
  ((defective - 1) : ℚ) / (total - 1) *
  ((defective - 2) : ℚ) / (total - 2)

/-- Theorem stating the probability of selecting three defective smartphones
    from a shipment of 400 smartphones, of which 150 are defective -/
theorem probability_three_defective_shipment :
  probability_three_defective 400 150 = 150 / 400 * 149 / 399 * 148 / 398 :=
by sorry

end probability_three_defective_shipment_l361_36181


namespace goldfish_feeding_l361_36146

/-- Given that one scoop of fish food can feed 8 goldfish, 
    prove that 4 scoops can feed 32 goldfish -/
theorem goldfish_feeding (scoop_capacity : ℕ) (num_scoops : ℕ) : 
  scoop_capacity = 8 → num_scoops = 4 → num_scoops * scoop_capacity = 32 := by
  sorry

end goldfish_feeding_l361_36146


namespace no_integer_solutions_l361_36130

theorem no_integer_solutions : ¬ ∃ (a b : ℤ), 3 * a^2 = b^2 + 1 := by
  sorry

end no_integer_solutions_l361_36130


namespace courier_packages_tomorrow_l361_36128

/-- The number of packages to be delivered tomorrow -/
def packages_to_deliver (yesterday : ℕ) (today : ℕ) : ℕ :=
  yesterday + today

/-- Theorem: The courier should deliver 240 packages tomorrow -/
theorem courier_packages_tomorrow :
  let yesterday := 80
  let today := 2 * yesterday
  packages_to_deliver yesterday today = 240 := by
  sorry

end courier_packages_tomorrow_l361_36128


namespace extreme_values_when_a_neg_one_max_value_when_a_positive_l361_36103

noncomputable section

-- Define the function f(x) = (ax^2 + x + a)e^x
def f (a : ℝ) (x : ℝ) : ℝ := (a * x^2 + x + a) * Real.exp x

-- Theorem for part (1)
theorem extreme_values_when_a_neg_one :
  let f := f (-1)
  (∃ x, ∀ y, f y ≥ f x) ∧ (f 0 = -1) ∧
  (∃ x, ∀ y, f y ≤ f x) ∧ (f (-1) = -3 * Real.exp (-1)) := by sorry

-- Theorem for part (2)
theorem max_value_when_a_positive (a : ℝ) (h : a > 0) :
  let f := f a
  let max_value := if a > 1 then (2*a + 1) * Real.exp (-1 - 1/a)
                   else (5*a - 2) * Real.exp (-2)
  ∀ x ∈ Set.Icc (-2) (-1), f x ≤ max_value := by sorry

end

end extreme_values_when_a_neg_one_max_value_when_a_positive_l361_36103


namespace max_ab_value_l361_36193

theorem max_ab_value (a b : ℝ) : 
  (∃ x y : ℝ, (x - a)^2 + (y - b)^2 = 1 ∧ x + 2*y - 1 = 0) →
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    (x₁ - a)^2 + (y₁ - b)^2 = 1 ∧ 
    (x₂ - a)^2 + (y₂ - b)^2 = 1 ∧ 
    x₁ + 2*y₁ - 1 = 0 ∧ 
    x₂ + 2*y₂ - 1 = 0 ∧ 
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = (4/5 * Real.sqrt 5)^2) →
  a * b ≤ 1/2 :=
sorry

end max_ab_value_l361_36193


namespace green_marble_probability_l361_36194

/-- The probability of drawing a green marble from a box with 100 marbles -/
theorem green_marble_probability :
  ∀ (p_white p_red_or_blue p_green : ℝ),
  p_white = 1/4 →
  p_red_or_blue = 0.55 →
  p_white + p_red_or_blue + p_green = 1 →
  p_green = 0.2 := by
  sorry

end green_marble_probability_l361_36194


namespace integer_triple_divisibility_l361_36175

theorem integer_triple_divisibility (a b c : ℕ+) : 
  (∃ k₁ k₂ k₃ : ℕ+, (a + 1 : ℕ) = k₁ * b ∧ (b + 1 : ℕ) = k₂ * c ∧ (c + 1 : ℕ) = k₃ * a) →
  ((a, b, c) = (1, 1, 1) ∨ (a, b, c) = (1, 2, 1) ∨ (a, b, c) = (1, 1, 2) ∨ (a, b, c) = (2, 1, 1)) :=
by sorry


end integer_triple_divisibility_l361_36175


namespace car_distance_theorem_l361_36165

/-- Calculates the total distance traveled by a car given its initial speed, acceleration, 
    acceleration time, constant speed, and constant speed time. -/
def total_distance (initial_speed : ℝ) (acceleration : ℝ) (accel_time : ℝ) 
                   (constant_speed : ℝ) (const_time : ℝ) : ℝ :=
  -- Distance covered during acceleration
  (initial_speed * accel_time + 0.5 * acceleration * accel_time^2) +
  -- Distance covered at constant speed
  (constant_speed * const_time)

/-- Theorem stating that a car with given parameters travels 250 miles in total -/
theorem car_distance_theorem : 
  total_distance 30 5 2 60 3 = 250 := by
  sorry

end car_distance_theorem_l361_36165


namespace cross_number_puzzle_solution_l361_36177

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def is_power_of (base : ℕ) (n : ℕ) : Prop := ∃ m : ℕ, n = base ^ m

theorem cross_number_puzzle_solution :
  ∃! d : ℕ, d < 10 ∧
    (∃ n₃ n₇ : ℕ,
      is_three_digit n₃ ∧
      is_three_digit n₇ ∧
      is_power_of 3 n₃ ∧
      is_power_of 7 n₇ ∧
      (∃ k₃ k₇ : ℕ, n₃ % 10^k₃ / 10^(k₃-1) = d ∧ n₇ % 10^k₇ / 10^(k₇-1) = d)) :=
sorry

end cross_number_puzzle_solution_l361_36177


namespace paving_cost_calculation_l361_36139

/-- The cost of paving a rectangular floor -/
def paving_cost (length width rate : ℝ) : ℝ :=
  length * width * rate

/-- Theorem: The cost of paving a rectangular floor with given dimensions and rate -/
theorem paving_cost_calculation (length width rate : ℝ) 
  (h1 : length = 5.5)
  (h2 : width = 3.75)
  (h3 : rate = 1000) :
  paving_cost length width rate = 20625 := by
  sorry

end paving_cost_calculation_l361_36139


namespace school_C_sample_size_l361_36124

/-- Represents the number of teachers in each school -/
structure SchoolPopulation where
  schoolA : ℕ
  schoolB : ℕ
  schoolC : ℕ

/-- Calculates the sample size for a given school in stratified sampling -/
def stratifiedSampleSize (totalSample : ℕ) (schoolPop : SchoolPopulation) (schoolSize : ℕ) : ℕ :=
  (schoolSize * totalSample) / (schoolPop.schoolA + schoolPop.schoolB + schoolPop.schoolC)

/-- Theorem: The stratified sample size for school C is 10 -/
theorem school_C_sample_size :
  let totalSample : ℕ := 60
  let schoolPop : SchoolPopulation := { schoolA := 180, schoolB := 270, schoolC := 90 }
  stratifiedSampleSize totalSample schoolPop schoolPop.schoolC = 10 := by
  sorry


end school_C_sample_size_l361_36124


namespace polynomial_coefficient_sum_l361_36116

theorem polynomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x, (1 - 2*x)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  a₀ + a₄ = 17 := by
sorry

end polynomial_coefficient_sum_l361_36116


namespace row_swap_matrix_l361_36173

theorem row_swap_matrix : ∃ (N : Matrix (Fin 2) (Fin 2) ℝ),
  ∀ (A : Matrix (Fin 2) (Fin 2) ℝ), N * A = ![![A 1 0, A 1 1], ![A 0 0, A 0 1]] :=
by
  sorry

end row_swap_matrix_l361_36173


namespace fruit_preference_ratio_l361_36151

theorem fruit_preference_ratio (total_students : ℕ) 
  (cherries_preference : ℕ) (apple_date_ratio : ℕ) (banana_cherry_ratio : ℕ) 
  (h1 : total_students = 780)
  (h2 : cherries_preference = 60)
  (h3 : apple_date_ratio = 2)
  (h4 : banana_cherry_ratio = 3) : 
  (banana_cherry_ratio * cherries_preference) / 
  ((total_students - banana_cherry_ratio * cherries_preference - cherries_preference) / 
   (apple_date_ratio + 1)) = 1 := by
sorry

end fruit_preference_ratio_l361_36151


namespace height_inequality_l361_36149

theorem height_inequality (a b : ℕ) (m : ℝ) (h_positive : a > 0 ∧ b > 0) 
  (h_right_triangle : m = (a * b : ℝ) / Real.sqrt ((a^2 + b^2 : ℕ) : ℝ)) :
  m ≤ Real.sqrt (((a^a * b^b : ℕ) : ℝ)^(1 / (a + b : ℝ))) / Real.sqrt 2 := by
  sorry

end height_inequality_l361_36149


namespace max_table_height_l361_36182

/-- Given a triangle DEF with sides 25, 28, and 31, prove that the maximum possible height h'
    of a table constructed from this triangle is equal to 4√77 / 53. -/
theorem max_table_height (DE EF FD : ℝ) (h_DE : DE = 25) (h_EF : EF = 28) (h_FD : FD = 31) :
  let s := (DE + EF + FD) / 2
  let area := Real.sqrt (s * (s - DE) * (s - EF) * (s - FD))
  let h_DE := 2 * area / DE
  let h_EF := 2 * area / EF
  ∃ h' : ℝ, h' = (h_DE * h_EF) / (h_DE + h_EF) ∧ h' = 4 * Real.sqrt 77 / 53 :=
by sorry


end max_table_height_l361_36182


namespace pencil_eraser_cost_theorem_l361_36134

theorem pencil_eraser_cost_theorem :
  ∃ (p e : ℕ), p > e ∧ p > 0 ∧ e > 0 ∧ 15 * p + 5 * e = 200 ∧ p + e = 18 := by
  sorry

end pencil_eraser_cost_theorem_l361_36134


namespace parabola_equation_l361_36123

/-- Given a point M(5,3) and a parabola y=ax^2 where the distance from M to the axis of symmetry is 6,
    prove that the equation of the parabola is either y = 1/12 x^2 or y = -1/36 x^2 -/
theorem parabola_equation (a : ℝ) (h : |5 + 1/(4*a)| = 6) :
  a = 1/12 ∨ a = -1/36 := by
  sorry

end parabola_equation_l361_36123


namespace jasons_red_marbles_indeterminate_l361_36137

theorem jasons_red_marbles_indeterminate (jason_blue : ℕ) (tom_blue : ℕ) (total_blue : ℕ) 
  (h1 : jason_blue = 44)
  (h2 : tom_blue = 24)
  (h3 : total_blue = jason_blue + tom_blue)
  (h4 : total_blue = 68) :
  ∃ (x y : ℕ), x ≠ y ∧ (jason_blue + x = jason_blue + y) :=
sorry

end jasons_red_marbles_indeterminate_l361_36137


namespace age_difference_l361_36101

theorem age_difference (A B C : ℕ) (h : A + B = B + C + 12) : A - C = 12 := by
  sorry

end age_difference_l361_36101


namespace second_year_interest_rate_l361_36120

/-- Calculates the interest rate for the second year given the initial principal,
    first year interest rate, and final amount after two years. -/
theorem second_year_interest_rate
  (initial_principal : ℝ)
  (first_year_rate : ℝ)
  (final_amount : ℝ)
  (h1 : initial_principal = 8000)
  (h2 : first_year_rate = 0.04)
  (h3 : final_amount = 8736) :
  let first_year_amount := initial_principal * (1 + first_year_rate)
  let second_year_rate := (final_amount / first_year_amount) - 1
  second_year_rate = 0.05 := by
sorry

end second_year_interest_rate_l361_36120


namespace highest_x_value_l361_36119

theorem highest_x_value (x : ℝ) :
  (((15 * x^2 - 40 * x + 18) / (4 * x - 3) + 7 * x = 9 * x - 2) →
   x ≤ 4) ∧
  (∃ y : ℝ, ((15 * y^2 - 40 * y + 18) / (4 * y - 3) + 7 * y = 9 * y - 2) ∧ y = 4) :=
by sorry

end highest_x_value_l361_36119


namespace fraction_power_equality_l361_36100

theorem fraction_power_equality (a b : ℝ) (m : ℤ) (ha : a > 0) (hb : b ≠ 0) :
  (b / a) ^ m = a ^ (-m) * b ^ m := by
  sorry

end fraction_power_equality_l361_36100


namespace abs_eq_sqrt_square_l361_36158

theorem abs_eq_sqrt_square (x : ℝ) : |x - 1| = Real.sqrt ((x - 1)^2) := by sorry

end abs_eq_sqrt_square_l361_36158


namespace parallelogram_vertex_sum_l361_36122

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram -/
structure Parallelogram where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Calculate the distance from a point to a line segment -/
def distanceToLineSegment (p : Point) (a : Point) (b : Point) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem parallelogram_vertex_sum (ABCD : Parallelogram) : 
  ABCD.A = Point.mk (-1) 2 →
  ABCD.B = Point.mk 3 (-6) →
  ABCD.D = Point.mk 7 0 →
  distanceToLineSegment ABCD.C ABCD.A ABCD.D = 3 * distanceToLineSegment ABCD.B ABCD.A ABCD.D →
  ABCD.C.x + ABCD.C.y = 11 :=
by
  sorry

end parallelogram_vertex_sum_l361_36122


namespace roy_julia_age_difference_l361_36189

theorem roy_julia_age_difference :
  ∀ (R J K : ℕ) (x : ℕ),
    R = J + x →  -- Roy is x years older than Julia
    R = K + x / 2 →  -- Roy is half of x years older than Kelly
    R + 4 = 2 * (J + 4) →  -- In 4 years, Roy will be twice as old as Julia
    (R + 4) * (K + 4) = 108 →  -- In 4 years, Roy's age multiplied by Kelly's age would be 108
    x = 6 :=  -- The difference between Roy's and Julia's ages is 6 years
by sorry

end roy_julia_age_difference_l361_36189


namespace polynomial_value_at_five_l361_36144

theorem polynomial_value_at_five (p : ℝ → ℝ) :
  (∃ a b c d : ℝ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + d) →
  p 1 = 1 →
  p 2 = 2 →
  p 3 = 3 →
  p 4 = 4 →
  p 5 = 29 := by
sorry

end polynomial_value_at_five_l361_36144


namespace profit_sharing_problem_l361_36133

/-- Given a profit shared between two partners in the ratio 2:5, where the second partner
    receives $2500, prove that the first partner will have $800 left after spending $200. -/
theorem profit_sharing_problem (total_parts : ℕ) (first_partner_parts second_partner_parts : ℕ) 
    (second_partner_share : ℕ) (shirt_cost : ℕ) :
  total_parts = first_partner_parts + second_partner_parts →
  first_partner_parts = 2 →
  second_partner_parts = 5 →
  second_partner_share = 2500 →
  shirt_cost = 200 →
  (first_partner_parts * second_partner_share / second_partner_parts) - shirt_cost = 800 :=
by sorry


end profit_sharing_problem_l361_36133


namespace min_value_product_min_value_achieved_l361_36143

theorem min_value_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x/y + y/z + z/x + y/x + z/y + x/z = 10) :
  (x/y + y/z + z/x) * (y/x + z/y + x/z) ≥ 25 := by
  sorry

theorem min_value_achieved (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x/y + y/z + z/x + y/x + z/y + x/z = 10) :
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧
    a/b + b/c + c/a + b/a + c/b + a/c = 10 ∧
    (a/b + b/c + c/a) * (b/a + c/b + a/c) = 25 := by
  sorry

end min_value_product_min_value_achieved_l361_36143


namespace total_interest_after_ten_years_l361_36114

/-- Calculate the total interest after 10 years given the conditions -/
theorem total_interest_after_ten_years
  (P : ℝ) -- Principal amount
  (R : ℝ) -- Interest rate (in percentage per annum)
  (h1 : P * R * 10 / 100 = 600) -- Simple interest on P for 10 years is Rs. 600
  : P * R * 5 / 100 + 3 * P * R * 5 / 100 = 1200 := by
  sorry

#check total_interest_after_ten_years

end total_interest_after_ten_years_l361_36114


namespace equation_holds_for_all_n_l361_36115

-- Define α as the positive root of x^2 - 1989x - 1 = 0
noncomputable def α : ℝ := (1989 + Real.sqrt (1989^2 + 4)) / 2

-- Define the equation to be proven
def equation (n : ℕ) : Prop :=
  ⌊α * n + 1989 * α * ⌊α * n⌋⌋ = 1989 * n + (1989^2 + 1) * ⌊α * n⌋

-- Theorem statement
theorem equation_holds_for_all_n : ∀ n : ℕ, equation n := by sorry

end equation_holds_for_all_n_l361_36115


namespace darwin_food_expense_l361_36125

theorem darwin_food_expense (initial_amount : ℚ) (gas_fraction : ℚ) (remaining : ℚ) : 
  initial_amount = 600 →
  gas_fraction = 1/3 →
  remaining = 300 →
  (initial_amount - gas_fraction * initial_amount - remaining) / (initial_amount - gas_fraction * initial_amount) = 1/4 := by
  sorry

end darwin_food_expense_l361_36125


namespace figure_2010_squares_l361_36140

/-- The number of squares in a figure of the sequence -/
def num_squares (n : ℕ) : ℕ := 1 + 4 * (n - 1)

/-- The theorem stating that Figure 2010 contains 8037 squares -/
theorem figure_2010_squares : num_squares 2010 = 8037 := by
  sorry

end figure_2010_squares_l361_36140


namespace negative_three_a_cubed_div_a_fourth_l361_36107

theorem negative_three_a_cubed_div_a_fourth (a : ℝ) (h : a ≠ 0) :
  -3 * a^3 / a^4 = -3 / a := by sorry

end negative_three_a_cubed_div_a_fourth_l361_36107


namespace partial_fraction_decomposition_A_l361_36138

theorem partial_fraction_decomposition_A (A B C : ℝ) :
  (∀ x : ℝ, x ≠ -2 ∧ x ≠ 1 →
    1 / (x^3 - 2*x^2 - 13*x + 10) = A / (x + 2) + B / (x - 1) + C / ((x - 1)^2)) →
  A = 1/9 := by
  sorry

end partial_fraction_decomposition_A_l361_36138


namespace linear_function_increasing_l361_36135

/-- A linear function y = (2k-6)x + (2k+1) is increasing if and only if k > 3 -/
theorem linear_function_increasing (k : ℝ) :
  (∀ x y : ℝ, y = (2*k - 6)*x + (2*k + 1)) →
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → ((2*k - 6)*x₁ + (2*k + 1) < (2*k - 6)*x₂ + (2*k + 1))) ↔
  k > 3 :=
by sorry

end linear_function_increasing_l361_36135


namespace sum_of_three_numbers_l361_36197

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 52) 
  (h2 : a*b + b*c + c*a = 72) : 
  a + b + c = 14 := by sorry

end sum_of_three_numbers_l361_36197


namespace power_of_product_l361_36166

theorem power_of_product (a : ℝ) : (2 * a) ^ 3 = 8 * a ^ 3 := by
  sorry

end power_of_product_l361_36166


namespace log_identity_l361_36153

theorem log_identity (a b P : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b ≠ 1) (ha1 : a ≠ 1) :
  (Real.log P / Real.log a) / (Real.log P / Real.log (a * b)) = 1 + Real.log b / Real.log a :=
by sorry

end log_identity_l361_36153


namespace expected_shots_value_l361_36172

/-- The probability of hitting the target -/
def p : ℝ := 0.8

/-- The maximum number of bullets -/
def max_shots : ℕ := 3

/-- The expected number of shots -/
def expected_shots : ℝ := p + 2 * (1 - p) * p + 3 * (1 - p) * (1 - p)

/-- Theorem stating that the expected number of shots is 1.24 -/
theorem expected_shots_value : expected_shots = 1.24 := by
  sorry

end expected_shots_value_l361_36172


namespace allowance_theorem_l361_36150

def initial_allowance : ℚ := 12

def first_week_spending (allowance : ℚ) : ℚ := allowance / 3

def second_week_spending (remaining : ℚ) : ℚ := remaining / 4

def final_amount (allowance : ℚ) : ℚ :=
  let after_first_week := allowance - first_week_spending allowance
  after_first_week - second_week_spending after_first_week

theorem allowance_theorem : final_amount initial_allowance = 6 := by
  sorry

end allowance_theorem_l361_36150


namespace expression_simplification_l361_36170

theorem expression_simplification :
  -2 * Real.sqrt 2 + 2^(-(1/2 : ℝ)) + 1 / (Real.sqrt 2 + 1) + (Real.sqrt 2 - 1)^(0 : ℝ) = -(Real.sqrt 2) / 2 := by
  sorry

end expression_simplification_l361_36170


namespace vector_coefficient_theorem_l361_36191

-- Define the space
variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Define points
variable (A B C P O : V)

-- Define the condition that P lies in the plane of triangle ABC
def in_plane (A B C P : V) : Prop := 
  ∃ (α β γ : ℝ), α + β + γ = 1 ∧ P = α • A + β • B + γ • C

-- Define the vector equation
def vector_equation (A B C P O : V) (x : ℝ) : Prop :=
  P - O = (1/2) • (A - O) + (1/3) • (B - O) + x • (C - O)

-- State the theorem
theorem vector_coefficient_theorem 
  (h_plane : in_plane V A B C P)
  (h_eq : vector_equation V A B C P O x) :
  x = 1/6 := by sorry

end vector_coefficient_theorem_l361_36191


namespace sum_of_three_numbers_l361_36155

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 222) 
  (h2 : a*b + b*c + c*a = 131) : 
  a + b + c = 22 := by
sorry

end sum_of_three_numbers_l361_36155


namespace sphere_surface_area_change_l361_36111

theorem sphere_surface_area_change (r₁ r₂ : ℝ) (h : r₁ > 0) (h' : r₂ > 0) : 
  (π * r₂^2 = 4 * π * r₁^2) → (4 * π * r₂^2 = 4 * (4 * π * r₁^2)) := by
  sorry

end sphere_surface_area_change_l361_36111


namespace triangle_segment_length_l361_36108

/-- Given a triangle ADE with point C on AD and point B on AC, prove that FC = 14.6 -/
theorem triangle_segment_length 
  (DC CB : ℝ) 
  (h1 : DC = 9) 
  (h2 : CB = 10) 
  (AD : ℝ) 
  (h3 : (1 : ℝ) / 3 * AD = AD - DC - CB) 
  (ED : ℝ) 
  (h4 : ED = 3 / 4 * AD) 
  (FC : ℝ) 
  (h5 : FC * AD = ED * (DC + CB + (1 / 3 * AD))) : 
  FC = 14.6 := by
sorry

end triangle_segment_length_l361_36108


namespace simplify_fraction_l361_36156

theorem simplify_fraction (k : ℝ) : 
  let expression := (6 * k + 12) / 6
  ∃ (c d : ℤ), expression = c * k + d ∧ (c : ℚ) / d = 1 / 2 := by
sorry

end simplify_fraction_l361_36156


namespace combined_age_in_five_years_l361_36117

def amyAge : ℕ := 15
def markAgeDiff : ℕ := 7
def emilyAgeFactor : ℕ := 2
def yearsPassed : ℕ := 5

theorem combined_age_in_five_years :
  (amyAge + yearsPassed) + 
  (amyAge + markAgeDiff + yearsPassed) + 
  (amyAge * emilyAgeFactor + yearsPassed) = 82 := by
  sorry

end combined_age_in_five_years_l361_36117


namespace beeswax_number_l361_36154

/-- Represents a mapping from characters to digits -/
def CodeMapping : Char → Nat
| 'A' => 1
| 'T' => 2
| 'Q' => 3
| 'B' => 4
| 'K' => 5
| 'X' => 6
| 'S' => 7
| 'W' => 8
| 'E' => 9
| 'P' => 0
| _ => 0

/-- Converts a string to a number using the code mapping -/
def stringToNumber (s : String) : Nat :=
  s.foldl (fun acc c => acc * 10 + CodeMapping c) 0

/-- The subtraction equation given in the problem -/
axiom subtraction_equation :
  stringToNumber "EASEBSBSX" - stringToNumber "BPWWKSETQ" = stringToNumber "KPEPWEKKQ"

/-- The main theorem to prove -/
theorem beeswax_number : stringToNumber "BEESWAX" = 4997816 := by
  sorry


end beeswax_number_l361_36154


namespace absolute_value_equation_solutions_l361_36131

theorem absolute_value_equation_solutions : 
  {x : ℝ | x + 1 = |x + 3| - |x - 1|} = {3, -1, -5} := by sorry

end absolute_value_equation_solutions_l361_36131


namespace josh_marbles_calculation_l361_36152

theorem josh_marbles_calculation (initial_marbles : ℕ) : 
  initial_marbles = 16 → 
  (initial_marbles * 3 * 3 / 4 : ℕ) = 36 := by
  sorry

end josh_marbles_calculation_l361_36152


namespace equal_fractions_sum_l361_36104

theorem equal_fractions_sum (n : ℕ) (sum : ℚ) (fraction : ℚ) :
  n = 450 →
  sum = 1 / 12 →
  n * fraction = sum →
  fraction = 1 / 5400 := by
  sorry

end equal_fractions_sum_l361_36104


namespace f_decreasing_on_interval_l361_36188

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 4

def f' (x : ℝ) : ℝ := 3*x^2 - 6*x

theorem f_decreasing_on_interval :
  ∀ x ∈ Set.Icc 0 2, 
    ∀ y ∈ Set.Icc 0 2, 
      x < y → f x > f y :=
by sorry

end f_decreasing_on_interval_l361_36188


namespace lollipops_for_class_l361_36147

/-- Calculates the number of lollipops given away based on the number of people and the lollipop distribution rate. -/
def lollipops_given (total_people : ℕ) (people_per_lollipop : ℕ) : ℕ :=
  total_people / people_per_lollipop

/-- Proves that given 60 people and 1 lollipop per 5 people, the teacher gives away 12 lollipops. -/
theorem lollipops_for_class (total_people : ℕ) (people_per_lollipop : ℕ) 
    (h1 : total_people = 60) 
    (h2 : people_per_lollipop = 5) : 
  lollipops_given total_people people_per_lollipop = 12 := by
  sorry

#eval lollipops_given 60 5  -- Expected output: 12

end lollipops_for_class_l361_36147


namespace hillary_stops_short_of_summit_l361_36129

/-- Represents the climbing scenario of Hillary and Eddy on Mt. Everest -/
structure ClimbingScenario where
  summit_distance : ℝ  -- Distance from base camp to summit in feet
  hillary_ascent_rate : ℝ  -- Hillary's ascent rate in ft/hr
  eddy_ascent_rate : ℝ  -- Eddy's ascent rate in ft/hr
  hillary_descent_rate : ℝ  -- Hillary's descent rate in ft/hr
  trip_duration : ℝ  -- Duration of the trip in hours

/-- Calculates the distance Hillary stops short of the summit -/
def distance_short_of_summit (scenario : ClimbingScenario) : ℝ :=
  scenario.hillary_ascent_rate * scenario.trip_duration - 
  (scenario.hillary_ascent_rate * scenario.trip_duration + 
   scenario.eddy_ascent_rate * scenario.trip_duration - scenario.summit_distance)

/-- Theorem stating that Hillary stops 3000 ft short of the summit -/
theorem hillary_stops_short_of_summit (scenario : ClimbingScenario) 
  (h1 : scenario.summit_distance = 5000)
  (h2 : scenario.hillary_ascent_rate = 800)
  (h3 : scenario.eddy_ascent_rate = 500)
  (h4 : scenario.hillary_descent_rate = 1000)
  (h5 : scenario.trip_duration = 6) :
  distance_short_of_summit scenario = 3000 := by
  sorry

#eval distance_short_of_summit {
  summit_distance := 5000,
  hillary_ascent_rate := 800,
  eddy_ascent_rate := 500,
  hillary_descent_rate := 1000,
  trip_duration := 6
}

end hillary_stops_short_of_summit_l361_36129


namespace twirly_tea_cups_l361_36195

theorem twirly_tea_cups (people_per_cup : ℕ) (total_people : ℕ) (num_cups : ℕ) :
  people_per_cup = 9 →
  total_people = 63 →
  num_cups * people_per_cup = total_people →
  num_cups = 7 := by
  sorry

end twirly_tea_cups_l361_36195


namespace inequality_problem_l361_36109

/-- Given real numbers a, b, c satisfying c < b < a and ac < 0,
    prove that cb² < ca² is not necessarily true,
    while ab > ac, c(b-a) > 0, and ac(a-c) < 0 are always true. -/
theorem inequality_problem (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) :
  (∀ x y z : ℝ, x < y ∧ y < z ∧ x * z < 0 → x * y^2 < x * z^2 → False) ∧
  (a * b > a * c) ∧
  (c * (b - a) > 0) ∧
  (a * c * (a - c) < 0) :=
sorry

end inequality_problem_l361_36109


namespace max_pieces_with_three_cuts_l361_36113

/-- The maximum number of identical pieces a cake can be divided into with a given number of cuts. -/
def max_cake_pieces (cuts : ℕ) : ℕ := 2^cuts

/-- Theorem: The maximum number of identical pieces a cake can be divided into with 3 cuts is 8. -/
theorem max_pieces_with_three_cuts :
  max_cake_pieces 3 = 8 := by
  sorry

end max_pieces_with_three_cuts_l361_36113


namespace highest_powers_sum_12_factorial_l361_36164

theorem highest_powers_sum_12_factorial : 
  let n := 12
  let factorial_n := n.factorial
  let highest_power_of_10 := (factorial_n.factorization 2).min (factorial_n.factorization 5)
  let highest_power_of_6 := (factorial_n.factorization 2).min (factorial_n.factorization 3)
  highest_power_of_10 + highest_power_of_6 = 7 := by
  sorry

end highest_powers_sum_12_factorial_l361_36164
