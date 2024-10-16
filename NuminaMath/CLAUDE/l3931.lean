import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_ten_l3931_393126

theorem arithmetic_square_root_of_ten : Real.sqrt 10 = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_ten_l3931_393126


namespace NUMINAMATH_CALUDE_tan_value_on_sqrt_graph_l3931_393180

/-- If the point (4, a) is on the graph of y = x^(1/2), then tan(aπ/6) = √3 -/
theorem tan_value_on_sqrt_graph (a : ℝ) : 
  a = 4^(1/2) → Real.tan (a * π / 6) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_value_on_sqrt_graph_l3931_393180


namespace NUMINAMATH_CALUDE_parabola_focus_coordinates_l3931_393121

/-- Given a parabola with equation x = (1/4m)y^2, its focus has coordinates (m, 0) --/
theorem parabola_focus_coordinates (m : ℝ) (h : m ≠ 0) :
  let parabola := {(x, y) : ℝ × ℝ | x = (1 / (4 * m)) * y^2}
  ∃ (focus : ℝ × ℝ), focus ∈ parabola ∧ focus = (m, 0) := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_coordinates_l3931_393121


namespace NUMINAMATH_CALUDE_length_of_AB_l3931_393189

/-- Given a line segment AB with points P and Q, prove that AB has length 48 -/
theorem length_of_AB (A B P Q : ℝ) : 
  (P - A) / (B - P) = 1 / 4 →  -- P divides AB in ratio 1:4
  (Q - A) / (B - Q) = 2 / 5 →  -- Q divides AB in ratio 2:5
  Q - P = 3 →                  -- Length of PQ is 3
  B - A = 48 := by             -- Length of AB is 48
sorry


end NUMINAMATH_CALUDE_length_of_AB_l3931_393189


namespace NUMINAMATH_CALUDE_fraction_problem_l3931_393143

theorem fraction_problem (n : ℤ) : 
  (n : ℚ) / (4 * n - 5 : ℚ) = 3 / 7 → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l3931_393143


namespace NUMINAMATH_CALUDE_cylinder_surface_area_ratio_l3931_393192

theorem cylinder_surface_area_ratio (a : ℝ) (h : a > 0) :
  let r := a / (2 * Real.pi)
  let side_area := a^2
  let base_area := Real.pi * r^2
  let total_area := 2 * base_area + side_area
  (total_area / side_area) = (1 + 2 * Real.pi) / (2 * Real.pi) := by
sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_ratio_l3931_393192


namespace NUMINAMATH_CALUDE_seven_balls_four_boxes_l3931_393123

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 92 ways to distribute 7 indistinguishable balls into 4 distinguishable boxes -/
theorem seven_balls_four_boxes : distribute_balls 7 4 = 92 := by
  sorry

end NUMINAMATH_CALUDE_seven_balls_four_boxes_l3931_393123


namespace NUMINAMATH_CALUDE_page_number_added_twice_l3931_393106

theorem page_number_added_twice (n : ℕ) (x : ℕ) : 
  n > 0 → 
  x > 0 → 
  x ≤ n → 
  (n * (n + 1)) / 2 + x = 1986 → 
  x = 33 := by
  sorry

end NUMINAMATH_CALUDE_page_number_added_twice_l3931_393106


namespace NUMINAMATH_CALUDE_positive_fourth_root_of_6561_l3931_393166

theorem positive_fourth_root_of_6561 (x : ℝ) (h1 : x > 0) (h2 : x^4 = 6561) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_positive_fourth_root_of_6561_l3931_393166


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l3931_393193

theorem sum_of_squares_of_roots (p q r : ℂ) : 
  (3 * p^3 - 4 * p^2 + 3 * p + 7 = 0) →
  (3 * q^3 - 4 * q^2 + 3 * q + 7 = 0) →
  (3 * r^3 - 4 * r^2 + 3 * r + 7 = 0) →
  p^2 + q^2 + r^2 = -2/9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l3931_393193


namespace NUMINAMATH_CALUDE_vector_inequality_l3931_393130

variable (V : Type*) [NormedAddCommGroup V]

theorem vector_inequality (v w : V) : 
  ‖v‖ + ‖w‖ ≤ ‖v + w‖ + ‖v - w‖ := by
  sorry

end NUMINAMATH_CALUDE_vector_inequality_l3931_393130


namespace NUMINAMATH_CALUDE_product_of_decimals_l3931_393195

theorem product_of_decimals : 3.6 * 0.04 = 0.144 := by
  sorry

end NUMINAMATH_CALUDE_product_of_decimals_l3931_393195


namespace NUMINAMATH_CALUDE_first_term_of_special_arithmetic_sequence_l3931_393153

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  /-- The first term of the sequence -/
  a : ℚ
  /-- The common difference of the sequence -/
  d : ℚ
  /-- The sum of the first 60 terms is 660 -/
  sum_first_60 : (60 : ℚ) / 2 * (2 * a + 59 * d) = 660
  /-- The sum of the next 60 terms (terms 61 to 120) is 3660 -/
  sum_next_60 : (60 : ℚ) / 2 * (2 * (a + 60 * d) + 59 * d) = 3660

/-- The first term of the arithmetic sequence with the given properties is -163/12 -/
theorem first_term_of_special_arithmetic_sequence (seq : ArithmeticSequence) : seq.a = -163/12 := by
  sorry

end NUMINAMATH_CALUDE_first_term_of_special_arithmetic_sequence_l3931_393153


namespace NUMINAMATH_CALUDE_eighth_power_sum_exists_l3931_393178

theorem eighth_power_sum_exists (ζ₁ ζ₂ ζ₃ : ℂ) 
  (sum_condition : ζ₁ + ζ₂ + ζ₃ = 2)
  (square_sum_condition : ζ₁^2 + ζ₂^2 + ζ₃^2 = 6)
  (cube_sum_condition : ζ₁^3 + ζ₂^3 + ζ₃^3 = 18) :
  ∃ s₈ : ℂ, ζ₁^8 + ζ₂^8 + ζ₃^8 = s₈ := by
  sorry

end NUMINAMATH_CALUDE_eighth_power_sum_exists_l3931_393178


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3931_393173

theorem sufficient_not_necessary (x y : ℝ) :
  ((x > 1 ∧ y > 1) → (x + y > 2)) ∧
  (∃ x y : ℝ, (x + y > 2) ∧ ¬(x > 1 ∧ y > 1)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3931_393173


namespace NUMINAMATH_CALUDE_probability_of_color_change_is_three_seventeenths_l3931_393156

/-- Represents the duration of each color in the traffic light cycle -/
structure TrafficLightCycle where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the probability of observing a color change in a traffic light -/
def probabilityOfColorChange (cycle : TrafficLightCycle) (observationDuration : ℕ) : ℚ :=
  let totalCycleDuration := cycle.green + cycle.yellow + cycle.red
  let changeWindows := 3 * observationDuration
  ↑changeWindows / ↑totalCycleDuration

/-- The main theorem stating the probability of observing a color change -/
theorem probability_of_color_change_is_three_seventeenths :
  let cycle := TrafficLightCycle.mk 45 5 35
  let observationDuration := 5
  probabilityOfColorChange cycle observationDuration = 3 / 17 := by
  sorry

#eval probabilityOfColorChange (TrafficLightCycle.mk 45 5 35) 5

end NUMINAMATH_CALUDE_probability_of_color_change_is_three_seventeenths_l3931_393156


namespace NUMINAMATH_CALUDE_race_time_l3931_393149

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  time : ℝ

/-- The race scenario -/
def Race (a b : Runner) : Prop :=
  -- The race is 1000 meters long
  1000 = a.speed * a.time ∧
  -- A beats B by 50 meters
  950 = b.speed * a.time ∧
  -- A beats B by 10 seconds
  b.time = a.time + 10

theorem race_time (a b : Runner) (h : Race a b) : a.time = 200 := by
  sorry

end NUMINAMATH_CALUDE_race_time_l3931_393149


namespace NUMINAMATH_CALUDE_total_new_games_is_92_l3931_393104

/-- The number of new games Katie has -/
def katie_new_games : ℕ := 84

/-- The number of new games Katie's friends have -/
def friends_new_games : ℕ := 8

/-- The total number of new games Katie and her friends have together -/
def total_new_games : ℕ := katie_new_games + friends_new_games

/-- Theorem stating that the total number of new games is 92 -/
theorem total_new_games_is_92 : total_new_games = 92 := by
  sorry

end NUMINAMATH_CALUDE_total_new_games_is_92_l3931_393104


namespace NUMINAMATH_CALUDE_tempo_original_value_l3931_393177

/-- The original value of a tempo given insurance and premium information -/
def original_value (insured_fraction : ℚ) (premium_rate : ℚ) (premium_amount : ℚ) : ℚ :=
  premium_amount / (premium_rate * insured_fraction)

/-- Theorem stating the original value of the tempo given the problem conditions -/
theorem tempo_original_value :
  let insured_fraction : ℚ := 4 / 5
  let premium_rate : ℚ := 13 / 1000
  let premium_amount : ℚ := 910
  original_value insured_fraction premium_rate premium_amount = 87500 := by
sorry

end NUMINAMATH_CALUDE_tempo_original_value_l3931_393177


namespace NUMINAMATH_CALUDE_circle_condition_l3931_393191

theorem circle_condition (k : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 - k*x + 2*y + k^2 - 2 = 0 ↔ ∃ r > 0, ∃ a b : ℝ, (x - a)^2 + (y - b)^2 = r^2) ↔
  -2 < k ∧ k < 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_condition_l3931_393191


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3931_393190

theorem inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | 12 * x^2 - a * x > a^2}
  (a > 0 → S = {x : ℝ | x < -a/4 ∨ x > a/3}) ∧
  (a = 0 → S = {x : ℝ | x ≠ 0}) ∧
  (a < 0 → S = {x : ℝ | x < a/3 ∨ x > -a/4}) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3931_393190


namespace NUMINAMATH_CALUDE_even_sum_problem_l3931_393120

theorem even_sum_problem (n : ℕ) (h1 : Odd n) 
  (h2 : (n^2 - 1) / 4 = 95 * 96) : n = 191 := by
  sorry

end NUMINAMATH_CALUDE_even_sum_problem_l3931_393120


namespace NUMINAMATH_CALUDE_floor_equation_solutions_l3931_393147

-- Define the floor function
def floor (x : ℚ) : ℤ := Int.floor x

-- Define the theorem
theorem floor_equation_solutions (a : ℚ) (ha : 0 < a) :
  ∀ x : ℕ+, (floor ((3 * x.val + a) / 4) = 2) ↔ (x.val = 1 ∨ x.val = 2 ∨ x.val = 3) :=
by sorry

end NUMINAMATH_CALUDE_floor_equation_solutions_l3931_393147


namespace NUMINAMATH_CALUDE_circle_distance_extrema_l3931_393139

-- Define the circle C
def Circle (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 1

-- Define points A and B
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)

-- Define the distance function d
def d (x y : ℝ) : ℝ := 
  (x + 1)^2 + y^2 + (x - 1)^2 + y^2

-- Theorem statement
theorem circle_distance_extrema :
  (∃ x y : ℝ, Circle x y ∧ ∀ x' y' : ℝ, Circle x' y' → d x y ≥ d x' y') ∧
  (∃ x y : ℝ, Circle x y ∧ ∀ x' y' : ℝ, Circle x' y' → d x y ≤ d x' y') ∧
  (∀ x y : ℝ, Circle x y → d x y ≤ 14) ∧
  (∀ x y : ℝ, Circle x y → d x y ≥ 10) :=
by sorry

end NUMINAMATH_CALUDE_circle_distance_extrema_l3931_393139


namespace NUMINAMATH_CALUDE_image_of_two_zero_l3931_393186

-- Define the mapping f
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 - p.2)

-- Theorem statement
theorem image_of_two_zero :
  f (2, 0) = (2, 2) := by
  sorry

end NUMINAMATH_CALUDE_image_of_two_zero_l3931_393186


namespace NUMINAMATH_CALUDE_no_twelve_parallelepipeds_l3931_393119

/-- A rectangular parallelepiped with edges parallel to coordinate axes -/
structure RectParallelepiped where
  xRange : Set ℝ
  yRange : Set ℝ
  zRange : Set ℝ

/-- Two parallelepipeds intersect if their projections on all axes intersect -/
def intersect (p q : RectParallelepiped) : Prop :=
  (p.xRange ∩ q.xRange).Nonempty ∧
  (p.yRange ∩ q.yRange).Nonempty ∧
  (p.zRange ∩ q.zRange).Nonempty

/-- The condition for intersection based on indices -/
def shouldIntersect (i j : Fin 12) : Prop :=
  i ≠ j + 1 ∧ i ≠ j - 1

/-- The main theorem stating that 12 such parallelepipeds cannot exist -/
theorem no_twelve_parallelepipeds :
  ¬ ∃ (ps : Fin 12 → RectParallelepiped),
    ∀ (i j : Fin 12), intersect (ps i) (ps j) ↔ shouldIntersect i j :=
sorry

end NUMINAMATH_CALUDE_no_twelve_parallelepipeds_l3931_393119


namespace NUMINAMATH_CALUDE_least_multiple_15_with_digit_product_multiple_15_l3931_393162

/-- Given a natural number, returns the product of its digits. -/
def digitProduct (n : ℕ) : ℕ := sorry

/-- Checks if a natural number is a multiple of 15. -/
def isMultipleOf15 (n : ℕ) : Prop := ∃ k, n = 15 * k

theorem least_multiple_15_with_digit_product_multiple_15 :
  ∀ n : ℕ, n > 0 → isMultipleOf15 n → isMultipleOf15 (digitProduct n) →
  n ≥ 315 ∧ (n = 315 → isMultipleOf15 (digitProduct 315)) := by sorry

end NUMINAMATH_CALUDE_least_multiple_15_with_digit_product_multiple_15_l3931_393162


namespace NUMINAMATH_CALUDE_expression_equality_l3931_393183

theorem expression_equality (y θ Q : ℝ) (h : 5 * (3 * y + 7 * Real.sin θ) = Q) :
  15 * (9 * y + 21 * Real.sin θ) = 9 * Q := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3931_393183


namespace NUMINAMATH_CALUDE_infinitely_many_solutions_l3931_393122

theorem infinitely_many_solutions (b : ℝ) :
  (∀ x : ℝ, 5 * (3 * x - b) = 3 * (5 * x + 15)) ↔ b = -9 := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_solutions_l3931_393122


namespace NUMINAMATH_CALUDE_sticker_trade_result_l3931_393109

/-- Calculates the final number of stickers after a given number of trades -/
def final_sticker_count (initial_count : ℕ) (num_trades : ℕ) : ℕ :=
  initial_count + num_trades * 4

/-- Theorem stating that after 50 trades, starting with 1 sticker, 
    the final count is 201 stickers -/
theorem sticker_trade_result : final_sticker_count 1 50 = 201 := by
  sorry

end NUMINAMATH_CALUDE_sticker_trade_result_l3931_393109


namespace NUMINAMATH_CALUDE_certain_fraction_proof_l3931_393185

theorem certain_fraction_proof (n : ℚ) (x : ℚ) 
  (h1 : n = 0.5833333333333333)
  (h2 : n = x + 1/4) : 
  x = 0.3333333333333333 := by
sorry

end NUMINAMATH_CALUDE_certain_fraction_proof_l3931_393185


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l3931_393159

theorem sum_of_a_and_b (a b : ℚ) 
  (eq1 : 2 * a + 5 * b = 47) 
  (eq2 : 4 * a + 3 * b = 39) : 
  a + b = 82 / 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l3931_393159


namespace NUMINAMATH_CALUDE_sum_even_coefficients_l3931_393198

theorem sum_even_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ a₁₂ : ℝ) :
  (∀ x : ℝ, (1 + x + x^2)^6 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + 
    a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10 + a₁₁*x^11 + a₁₂*x^12) →
  a₂ + a₄ + a₆ + a₈ + a₁₀ + a₁₂ = 364 := by
sorry

end NUMINAMATH_CALUDE_sum_even_coefficients_l3931_393198


namespace NUMINAMATH_CALUDE_pentagon_from_equal_segments_l3931_393111

theorem pentagon_from_equal_segments (segment_length : Real) 
  (h1 : segment_length = 2 / 5)
  (h2 : 5 * segment_length = 2) : 
  4 * segment_length > segment_length := by
  sorry

end NUMINAMATH_CALUDE_pentagon_from_equal_segments_l3931_393111


namespace NUMINAMATH_CALUDE_geometric_progression_first_term_l3931_393113

theorem geometric_progression_first_term
  (S : ℝ)
  (sum_first_two : ℝ)
  (h1 : S = 9)
  (h2 : sum_first_two = 7) :
  ∃ (a : ℝ), (a = 3 * (3 - Real.sqrt 2) ∨ a = 3 * (3 + Real.sqrt 2)) ∧
    (∃ (r : ℝ), S = a / (1 - r) ∧ sum_first_two = a + a * r) :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_first_term_l3931_393113


namespace NUMINAMATH_CALUDE_melanie_dimes_problem_l3931_393172

/-- Calculates the number of dimes Melanie's mother gave her -/
def dimes_from_mother (initial : ℕ) (given_to_dad : ℕ) (final : ℕ) : ℕ :=
  final - (initial - given_to_dad)

theorem melanie_dimes_problem :
  dimes_from_mother 8 7 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_melanie_dimes_problem_l3931_393172


namespace NUMINAMATH_CALUDE_parallel_vectors_proportional_components_l3931_393170

/-- Given two 2D vectors a and b, if they are parallel, then their components are proportional. -/
theorem parallel_vectors_proportional_components (a b : ℝ × ℝ) :
  (∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2) ↔ ∃ m : ℝ, a = (2, -1) ∧ b = (-1, m) ∧ m = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_parallel_vectors_proportional_components_l3931_393170


namespace NUMINAMATH_CALUDE_A_intersect_B_l3931_393187

-- Define the universe U
def U : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- Define set A
def A : Set Nat := {2, 4, 6, 8, 10}

-- Define complement of A with respect to U
def C_UA : Set Nat := {1, 3, 5, 7, 9}

-- Define complement of B with respect to U
def C_UB : Set Nat := {1, 4, 6, 8, 9}

-- Define set B (derived from its complement)
def B : Set Nat := U \ C_UB

theorem A_intersect_B : A ∩ B = {2} := by
  sorry

end NUMINAMATH_CALUDE_A_intersect_B_l3931_393187


namespace NUMINAMATH_CALUDE_circle_circumference_from_area_l3931_393167

theorem circle_circumference_from_area :
  ∀ (r : ℝ), r > 0 → π * r^2 = 225 * π → 2 * π * r = 30 * π := by
  sorry

end NUMINAMATH_CALUDE_circle_circumference_from_area_l3931_393167


namespace NUMINAMATH_CALUDE_cos_seventeen_pi_sixths_l3931_393148

theorem cos_seventeen_pi_sixths : Real.cos (17 * π / 6) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_seventeen_pi_sixths_l3931_393148


namespace NUMINAMATH_CALUDE_money_distribution_l3931_393110

theorem money_distribution (a b c : ℤ) : 
  a + b + c = 900 → a + c = 400 → b + c = 750 → c = 250 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l3931_393110


namespace NUMINAMATH_CALUDE_other_number_proof_l3931_393140

theorem other_number_proof (a b : ℕ+) (h1 : Nat.gcd a b = 12) (h2 : Nat.lcm a b = 396) (h3 : a = 132) : b = 36 := by
  sorry

end NUMINAMATH_CALUDE_other_number_proof_l3931_393140


namespace NUMINAMATH_CALUDE_cube_surface_area_from_volume_l3931_393134

theorem cube_surface_area_from_volume : 
  ∀ (v : ℝ) (s : ℝ), 
  v = 729 →  -- Given volume
  v = s^3 →  -- Volume formula
  6 * s^2 = 486 -- Surface area formula and result
  := by sorry

end NUMINAMATH_CALUDE_cube_surface_area_from_volume_l3931_393134


namespace NUMINAMATH_CALUDE_number_division_l3931_393102

theorem number_division (x : ℝ) : x + 8 = 88 → x / 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_number_division_l3931_393102


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l3931_393135

theorem arithmetic_mean_of_fractions : 
  let a := (7 : ℚ) / 10
  let b := (4 : ℚ) / 5
  let c := (3 : ℚ) / 4
  c = (a + b) / 2 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l3931_393135


namespace NUMINAMATH_CALUDE_square_area_ratio_l3931_393145

theorem square_area_ratio (a b : ℝ) (h : 4 * a = 2 * b) : b^2 = 4 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l3931_393145


namespace NUMINAMATH_CALUDE_binomial_expansion_and_specific_case_l3931_393101

theorem binomial_expansion_and_specific_case :
  ∀ (a b : ℝ),
    (a + b)^4 = a^4 + 4*a^3*b + 6*a^2*b^2 + 4*a*b^3 + b^4 ∧
    (2 - 1/3)^4 = 625/81 :=
by sorry

end NUMINAMATH_CALUDE_binomial_expansion_and_specific_case_l3931_393101


namespace NUMINAMATH_CALUDE_compound_interest_rate_l3931_393137

theorem compound_interest_rate (P : ℝ) (h1 : P * (1 + r)^6 = 6000) (h2 : P * (1 + r)^7 = 7500) : r = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_rate_l3931_393137


namespace NUMINAMATH_CALUDE_red_cube_latin_square_bijection_l3931_393154

/-- A Latin square of order n is an n × n array filled with n different symbols, 
    each occurring exactly once in each row and exactly once in each column. -/
def is_latin_square (s : Fin 4 → Fin 4 → Fin 4) : Prop :=
  ∀ i j k : Fin 4, 
    (∀ j' : Fin 4, j ≠ j' → s i j ≠ s i j') ∧ 
    (∀ i' : Fin 4, i ≠ i' → s i j ≠ s i' j)

/-- The number of 4 × 4 Latin squares -/
def num_latin_squares : ℕ := sorry

/-- A configuration of red cubes in a 4 × 4 × 4 cube -/
def red_cube_config : Type := Fin 4 → Fin 4 → Fin 4

/-- A valid configuration of red cubes satisfies the constraint that
    in every 1 × 1 × 4 rectangular prism, exactly 1 unit cube is red -/
def is_valid_config (c : red_cube_config) : Prop :=
  ∀ i j : Fin 4, ∃! k : Fin 4, c i j = k

/-- The number of valid red cube configurations -/
def num_valid_configs : ℕ := sorry

theorem red_cube_latin_square_bijection :
  num_valid_configs = num_latin_squares :=
sorry

end NUMINAMATH_CALUDE_red_cube_latin_square_bijection_l3931_393154


namespace NUMINAMATH_CALUDE_exterior_angle_sum_l3931_393103

theorem exterior_angle_sum (angle1 angle2 angle3 angle4 : ℝ) :
  angle1 = 100 →
  angle2 = 60 →
  angle3 = 90 →
  angle1 + angle2 + angle3 + angle4 = 360 →
  angle4 = 110 := by
  sorry

end NUMINAMATH_CALUDE_exterior_angle_sum_l3931_393103


namespace NUMINAMATH_CALUDE_angle_of_inclination_sqrt3x_minus2_l3931_393163

/-- The angle of inclination of the line y = √3x - 2 is 60°. -/
theorem angle_of_inclination_sqrt3x_minus2 :
  let line : ℝ → ℝ := λ x ↦ Real.sqrt 3 * x - 2
  let slope : ℝ := Real.sqrt 3
  let angle_of_inclination : ℝ := Real.arctan slope
  angle_of_inclination = π / 3 := by
sorry

end NUMINAMATH_CALUDE_angle_of_inclination_sqrt3x_minus2_l3931_393163


namespace NUMINAMATH_CALUDE_ball_radius_from_hole_dimensions_l3931_393181

/-- The radius of a spherical ball that leaves a circular hole in a frozen lake surface. -/
def ball_radius (hole_width : ℝ) (hole_depth : ℝ) : ℝ :=
  hole_depth

/-- Theorem stating that if a spherical ball leaves a hole 32 cm wide and 16 cm deep
    in a frozen lake surface, its radius is 16 cm. -/
theorem ball_radius_from_hole_dimensions :
  ball_radius 32 16 = 16 := by
  sorry

end NUMINAMATH_CALUDE_ball_radius_from_hole_dimensions_l3931_393181


namespace NUMINAMATH_CALUDE_equation_proof_l3931_393152

theorem equation_proof : (12 : ℕ)^3 * 6^4 / 432 = 5184 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l3931_393152


namespace NUMINAMATH_CALUDE_equation_describes_cone_l3931_393182

-- Define cylindrical coordinates
structure CylindricalCoord where
  r : ℝ
  θ : ℝ
  z : ℝ

-- Define the equation z = kr
def coneEquation (k : ℝ) (p : CylindricalCoord) : Prop :=
  p.z = k * p.r

-- Define a cone in cylindrical coordinates
def isCone (S : Set CylindricalCoord) : Prop :=
  ∃ k : ℝ, ∀ p ∈ S, coneEquation k p

-- Theorem statement
theorem equation_describes_cone (k : ℝ) :
  isCone { p : CylindricalCoord | coneEquation k p } :=
sorry

end NUMINAMATH_CALUDE_equation_describes_cone_l3931_393182


namespace NUMINAMATH_CALUDE_john_daily_calories_is_3275_l3931_393115

/-- Calculates John's total daily calorie intake based on given meal and shake information. -/
def johnDailyCalories : ℕ :=
  let breakfastCalories : ℕ := 500
  let lunchCalories : ℕ := breakfastCalories + (breakfastCalories / 4)
  let dinnerCalories : ℕ := 2 * lunchCalories
  let shakeCalories : ℕ := 3 * 300
  breakfastCalories + lunchCalories + dinnerCalories + shakeCalories

/-- Theorem stating that John's total daily calorie intake is 3275 calories. -/
theorem john_daily_calories_is_3275 : johnDailyCalories = 3275 := by
  sorry

end NUMINAMATH_CALUDE_john_daily_calories_is_3275_l3931_393115


namespace NUMINAMATH_CALUDE_power_sum_division_l3931_393158

theorem power_sum_division (x y : ℕ) (hx : x = 3) (hy : y = 4) : (x^5 + 3*y^3) / 9 = 48 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_division_l3931_393158


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l3931_393142

theorem quadratic_root_relation : 
  ∀ x₁ x₂ : ℝ, 
  x₁^2 - 2*x₁ - 8 = 0 → 
  x₂^2 - 2*x₂ - 8 = 0 → 
  (x₁ + x₂) / (x₁ * x₂) = -1/4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l3931_393142


namespace NUMINAMATH_CALUDE_max_area_inscribed_ngon_l3931_393184

/-- An n-gon with given side lengths -/
structure Ngon (n : ℕ) where
  sides : Fin n → ℝ
  area : ℝ

/-- An n-gon inscribed in a circle -/
structure InscribedNgon (n : ℕ) extends Ngon n where
  isInscribed : Bool

/-- Theorem: The area of any n-gon is less than or equal to 
    the area of the inscribed n-gon with the same side lengths -/
theorem max_area_inscribed_ngon (n : ℕ) (l : Fin n → ℝ) :
  ∀ (P : Ngon n), P.sides = l →
  ∃ (Q : InscribedNgon n), Q.sides = l ∧ P.area ≤ Q.area :=
sorry

end NUMINAMATH_CALUDE_max_area_inscribed_ngon_l3931_393184


namespace NUMINAMATH_CALUDE_units_digit_of_n_l3931_393160

/-- Given two natural numbers a and b, returns true if a has units digit 9 -/
def hasUnitsDigit9 (a : ℕ) : Prop :=
  a % 10 = 9

/-- Given a natural number n, returns its units digit -/
def unitsDigit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_of_n (a b n : ℕ) 
  (h1 : a * b = 34^8) 
  (h2 : hasUnitsDigit9 a) 
  (h3 : n = b) : 
  unitsDigit n = 4 := by
sorry

end NUMINAMATH_CALUDE_units_digit_of_n_l3931_393160


namespace NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l3931_393157

theorem arithmetic_expression_evaluation :
  37 + (87 / 29) + (15 * 19) - 100 - (450 / 15) + 13 = 208 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l3931_393157


namespace NUMINAMATH_CALUDE_minutes_after_midnight_l3931_393199

/-- Represents a date and time -/
structure DateTime where
  year : ℕ
  month : ℕ
  day : ℕ
  hour : ℕ
  minute : ℕ

/-- Adds minutes to a DateTime -/
def addMinutes (dt : DateTime) (minutes : ℕ) : DateTime :=
  sorry

/-- The starting DateTime (midnight on January 1, 2013) -/
def startDateTime : DateTime :=
  { year := 2013, month := 1, day := 1, hour := 0, minute := 0 }

/-- The resulting DateTime after adding 2537 minutes -/
def resultDateTime : DateTime :=
  { year := 2013, month := 1, day := 2, hour := 18, minute := 17 }

/-- Theorem stating that adding 2537 minutes to the start time results in the correct end time -/
theorem minutes_after_midnight (startTime : DateTime) (elapsedMinutes : ℕ) :
  startTime = startDateTime → elapsedMinutes = 2537 →
  addMinutes startTime elapsedMinutes = resultDateTime :=
by
  sorry

end NUMINAMATH_CALUDE_minutes_after_midnight_l3931_393199


namespace NUMINAMATH_CALUDE_plane_speed_theorem_l3931_393164

theorem plane_speed_theorem (v : ℝ) (h1 : v > 0) :
  5 * v + 5 * (3 * v) = 4800 →
  v = 240 ∧ 3 * v = 720 := by
  sorry

end NUMINAMATH_CALUDE_plane_speed_theorem_l3931_393164


namespace NUMINAMATH_CALUDE_interval_condition_l3931_393196

theorem interval_condition (x : ℝ) : 
  (2 < 4*x ∧ 4*x < 3 ∧ 2 < 5*x ∧ 5*x < 3) ↔ (1/2 < x ∧ x < 3/5) :=
by sorry

end NUMINAMATH_CALUDE_interval_condition_l3931_393196


namespace NUMINAMATH_CALUDE_ellipse_perpendicular_points_sum_l3931_393133

/-- Ellipse defined by parametric equations x = 2cos(α) and y = √3sin(α) -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ α : ℝ, p.1 = 2 * Real.cos α ∧ p.2 = Real.sqrt 3 * Real.sin α}

/-- Distance squared from origin to a point -/
def distanceSquared (p : ℝ × ℝ) : ℝ := p.1^2 + p.2^2

/-- Two points are perpendicular if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 = 0

theorem ellipse_perpendicular_points_sum (A B : ℝ × ℝ) 
  (hA : A ∈ Ellipse) (hB : B ∈ Ellipse) (hPerp : perpendicular A B) :
  1 / distanceSquared A + 1 / distanceSquared B = 7 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_perpendicular_points_sum_l3931_393133


namespace NUMINAMATH_CALUDE_james_out_of_pocket_cost_l3931_393114

/-- Calculates the out-of-pocket cost for a given service -/
def outOfPocketCost (cost : ℝ) (coveragePercent : ℝ) : ℝ :=
  cost - (cost * coveragePercent)

/-- Theorem: James's total out-of-pocket cost is $262.70 -/
theorem james_out_of_pocket_cost : 
  let consultation_cost : ℝ := 300
  let consultation_coverage : ℝ := 0.83
  let xray_cost : ℝ := 150
  let xray_coverage : ℝ := 0.74
  let medication_cost : ℝ := 75
  let medication_coverage : ℝ := 0.55
  let therapy_cost : ℝ := 120
  let therapy_coverage : ℝ := 0.62
  let equipment_cost : ℝ := 85
  let equipment_coverage : ℝ := 0.49
  let followup_cost : ℝ := 200
  let followup_coverage : ℝ := 0.75
  
  (outOfPocketCost consultation_cost consultation_coverage +
   outOfPocketCost xray_cost xray_coverage +
   outOfPocketCost medication_cost medication_coverage +
   outOfPocketCost therapy_cost therapy_coverage +
   outOfPocketCost equipment_cost equipment_coverage +
   outOfPocketCost followup_cost followup_coverage) = 262.70 := by
  sorry


end NUMINAMATH_CALUDE_james_out_of_pocket_cost_l3931_393114


namespace NUMINAMATH_CALUDE_miraflores_can_win_l3931_393117

/-- Represents a voting system with multiple tiers --/
structure VotingSystem :=
  (total_voters : ℕ)
  (supporter_percentage : ℚ)
  (min_group_size : ℕ)
  (max_group_size : ℕ)

/-- Checks if a candidate can win in the given voting system --/
def can_win (vs : VotingSystem) : Prop :=
  ∃ (grouping : ℕ → ℕ),
    (∀ n, vs.min_group_size ≤ grouping n ∧ grouping n ≤ vs.max_group_size) ∧
    (∃ (final_group : ℕ), 
      final_group > 1 ∧
      final_group ≤ vs.total_voters ∧
      (vs.total_voters * vs.supporter_percentage).num * 2 > 
        (vs.total_voters * vs.supporter_percentage).den * final_group)

/-- The main theorem --/
theorem miraflores_can_win :
  ∃ (vs : VotingSystem), 
    vs.total_voters = 20000000 ∧
    vs.supporter_percentage = 1/100 ∧
    vs.min_group_size = 2 ∧
    vs.max_group_size = 5 ∧
    can_win vs :=
  sorry

end NUMINAMATH_CALUDE_miraflores_can_win_l3931_393117


namespace NUMINAMATH_CALUDE_no_real_roots_for_nonzero_k_l3931_393150

theorem no_real_roots_for_nonzero_k (k : ℝ) (hk : k ≠ 0) :
  ∀ x : ℝ, x^2 + 2*k*x + 3*k^2 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_for_nonzero_k_l3931_393150


namespace NUMINAMATH_CALUDE_erased_number_problem_l3931_393129

theorem erased_number_problem (n : ℕ) (x : ℕ) : 
  x ≤ n →
  (n * (n + 1) / 2 - x : ℚ) / (n - 1 : ℚ) = 951 / 19 →
  x = 99 :=
by sorry

end NUMINAMATH_CALUDE_erased_number_problem_l3931_393129


namespace NUMINAMATH_CALUDE_not_right_triangle_l3931_393155

theorem not_right_triangle (a b c : ℕ) (h : a = 3 ∧ b = 4 ∧ c = 6) : 
  ¬(a^2 + b^2 = c^2) := by
  sorry

#check not_right_triangle

end NUMINAMATH_CALUDE_not_right_triangle_l3931_393155


namespace NUMINAMATH_CALUDE_range_of_a_l3931_393188

/-- Given that the inequality x^2 + ax - 2 > 0 has a solution in the interval [1,2],
    the range of a is (-1, +∞) -/
theorem range_of_a (a : ℝ) : 
  (∃ x ∈ Set.Icc 1 2, x^2 + a*x - 2 > 0) ↔ a ∈ Set.Ioi (-1) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3931_393188


namespace NUMINAMATH_CALUDE_consecutive_composites_exist_l3931_393107

/-- A function that checks if a number is composite -/
def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ m, 1 < m ∧ m < n ∧ n % m = 0

/-- The existence of a sequence of n consecutive composite numbers within the first 500 natural numbers -/
def exists_consecutive_composites (n : ℕ) : Prop :=
  ∃ k, k > 0 ∧ k + n - 1 ≤ 500 ∧ ∀ i, k ≤ i ∧ i < k + n → is_composite i

theorem consecutive_composites_exist :
  exists_consecutive_composites 9 ∧ exists_consecutive_composites 11 := by sorry

end NUMINAMATH_CALUDE_consecutive_composites_exist_l3931_393107


namespace NUMINAMATH_CALUDE_circles_intersect_l3931_393128

-- Define the circles C1 and C2
def C1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 8*y - 8 = 0
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 4*y - 8 = 0

-- Define the centers and radii of the circles
def center1 : ℝ × ℝ := (-1, -4)
def center2 : ℝ × ℝ := (2, -2)
def radius1 : ℝ := 5
def radius2 : ℝ := 4

-- Theorem stating that the circles are intersecting
theorem circles_intersect :
  let d := Real.sqrt ((center1.1 - center2.1)^2 + (center1.2 - center2.2)^2)
  d > abs (radius1 - radius2) ∧ d < radius1 + radius2 := by
  sorry

end NUMINAMATH_CALUDE_circles_intersect_l3931_393128


namespace NUMINAMATH_CALUDE_sum_of_squares_l3931_393175

theorem sum_of_squares (a b c : ℝ) (h1 : a * b + b * c + c * a = 6) (h2 : a + b + c = 15) :
  a^2 + b^2 + c^2 = 213 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3931_393175


namespace NUMINAMATH_CALUDE_dot_product_result_l3931_393151

def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (3, -2)

theorem dot_product_result : (3 • a - b) • (a - 2 • b) = -15 := by sorry

end NUMINAMATH_CALUDE_dot_product_result_l3931_393151


namespace NUMINAMATH_CALUDE_partnership_share_calculation_l3931_393108

/-- Given a partnership where three partners invest different amounts and one partner's share is known, 
    calculate the share of another partner. -/
theorem partnership_share_calculation 
  (investment_a investment_b investment_c : ℕ)
  (duration : ℕ)
  (share_b : ℕ) 
  (h1 : investment_a = 11000)
  (h2 : investment_b = 15000)
  (h3 : investment_c = 23000)
  (h4 : duration = 8)
  (h5 : share_b = 3315) :
  (investment_a : ℚ) / (investment_a + investment_b + investment_c) * 
  (share_b : ℚ) * ((investment_a + investment_b + investment_c) : ℚ) / investment_b = 2421 :=
by sorry

end NUMINAMATH_CALUDE_partnership_share_calculation_l3931_393108


namespace NUMINAMATH_CALUDE_right_angled_figure_l3931_393146

def top_side (X : ℝ) : ℝ := 2 + 1 + 3 + X
def bottom_side : ℝ := 3 + 4 + 5

theorem right_angled_figure (X : ℝ) : 
  top_side X = bottom_side → X = 6 := by
  sorry

end NUMINAMATH_CALUDE_right_angled_figure_l3931_393146


namespace NUMINAMATH_CALUDE_point_in_first_quadrant_l3931_393194

theorem point_in_first_quadrant (a : ℝ) (h : a < 0) :
  let P : ℝ × ℝ := (-a, a^2)
  P.1 > 0 ∧ P.2 > 0 :=
sorry

end NUMINAMATH_CALUDE_point_in_first_quadrant_l3931_393194


namespace NUMINAMATH_CALUDE_triangle_problem_l3931_393124

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_problem (t : Triangle) 
  (h1 : 2 * t.a * Real.cos t.C = 2 * t.b - t.c)
  (h2 : t.a = Real.sqrt 21)
  (h3 : t.b = 4) :
  t.A = π / 3 ∧ t.c = 5 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l3931_393124


namespace NUMINAMATH_CALUDE_petya_can_prevent_natural_sum_l3931_393179

/-- Represents a player's turn in the game -/
structure Turn where
  player : Bool  -- true for Petya, false for Vasya
  fractions : List Nat  -- List of denominators of fractions written

/-- The state of the game board -/
structure GameState where
  turns : List Turn
  sum : Rat

/-- Vasya's strategy to choose fractions -/
def vasyaStrategy (state : GameState) : List Nat := sorry

/-- Petya's strategy to choose a fraction -/
def petyaStrategy (state : GameState) : Nat := sorry

/-- Checks if the sum of fractions is a natural number -/
def isNaturalSum (sum : Rat) : Bool := sorry

/-- Simulates the game for a given number of rounds -/
def playGame (rounds : Nat) : GameState := sorry

/-- Theorem stating that Petya can prevent Vasya from achieving a natural number sum -/
theorem petya_can_prevent_natural_sum :
  ∀ (rounds : Nat), ¬(isNaturalSum (playGame rounds).sum) := by sorry

end NUMINAMATH_CALUDE_petya_can_prevent_natural_sum_l3931_393179


namespace NUMINAMATH_CALUDE_rabbit_count_l3931_393174

/-- Given a cage with chickens and rabbits, prove that the number of rabbits is 31 -/
theorem rabbit_count (total_heads : ℕ) (r c : ℕ) : 
  total_heads = 51 →
  r + c = total_heads →
  4 * r = 3 * (2 * c) + 4 →
  r = 31 := by
  sorry

end NUMINAMATH_CALUDE_rabbit_count_l3931_393174


namespace NUMINAMATH_CALUDE_solve_equation_l3931_393131

theorem solve_equation (x : ℚ) : (3 * x + 5) / 7 = 13 → x = 86 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3931_393131


namespace NUMINAMATH_CALUDE_cylinder_volume_l3931_393197

/-- The volume of a cylinder with radius 5 cm and height 8 cm is 628 cm³, given that π ≈ 3.14 -/
theorem cylinder_volume : 
  let r : ℝ := 5
  let h : ℝ := 8
  let π : ℝ := 3.14
  π * r^2 * h = 628 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_l3931_393197


namespace NUMINAMATH_CALUDE_constant_term_binomial_expansion_l3931_393132

theorem constant_term_binomial_expansion :
  let n : ℕ := 8
  let a : ℝ := 1
  let b : ℝ := 1/3
  (Finset.sum (Finset.range (n + 1)) (λ k => Nat.choose n k * a^k * b^(n - k) * (if k = n/2 then 1 else 0))) = 28 :=
by sorry

end NUMINAMATH_CALUDE_constant_term_binomial_expansion_l3931_393132


namespace NUMINAMATH_CALUDE_max_balls_in_cube_l3931_393112

theorem max_balls_in_cube (cube_side : ℝ) (ball_radius : ℝ) : 
  cube_side = 10 → 
  ball_radius = 3 → 
  ⌊(cube_side ^ 3) / ((4 / 3) * Real.pi * ball_radius ^ 3)⌋ = 8 := by
sorry

end NUMINAMATH_CALUDE_max_balls_in_cube_l3931_393112


namespace NUMINAMATH_CALUDE_alice_has_ball_after_two_turns_l3931_393168

/-- Represents the probability of Alice tossing the ball to Bob -/
def alice_toss_prob : ℚ := 5/8

/-- Represents the probability of Alice keeping the ball -/
def alice_keep_prob : ℚ := 3/8

/-- Represents the probability of Bob tossing the ball to Alice -/
def bob_toss_prob : ℚ := 1/4

/-- Represents the probability of Bob keeping the ball -/
def bob_keep_prob : ℚ := 3/4

/-- The theorem stating the probability of Alice having the ball after two turns -/
theorem alice_has_ball_after_two_turns : 
  alice_toss_prob * bob_toss_prob + alice_keep_prob * alice_keep_prob = 19/64 := by
  sorry

#check alice_has_ball_after_two_turns

end NUMINAMATH_CALUDE_alice_has_ball_after_two_turns_l3931_393168


namespace NUMINAMATH_CALUDE_fixed_point_exponential_function_l3931_393100

theorem fixed_point_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  ∃ (x y : ℝ), x = 2 ∧ y = 2 ∧ y = a^(x - 2) + 1 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_function_l3931_393100


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l3931_393105

theorem simplify_sqrt_expression : 2 * Real.sqrt 3 - Real.sqrt 12 + Real.sqrt 75 = 5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l3931_393105


namespace NUMINAMATH_CALUDE_g_five_equals_one_l3931_393118

theorem g_five_equals_one (g : ℝ → ℝ)
  (h1 : ∀ x y : ℝ, g (x - y) = g x * g y)
  (h2 : ∀ x : ℝ, g x ≠ 0) :
  g 5 = 1 := by
sorry

end NUMINAMATH_CALUDE_g_five_equals_one_l3931_393118


namespace NUMINAMATH_CALUDE_solve_candy_store_problem_l3931_393127

/-- Represents the candy store problem --/
def candy_store_problem (caramel_price toffee_price chocolate_price : ℕ)
  (initial_quantity : ℕ) (initial_money : ℕ) : Prop :=
  let chocolate_promo := initial_quantity / 3
  let toffee_to_buy := initial_quantity - chocolate_promo
  let caramel_promo := toffee_to_buy / 3
  let caramel_to_buy := initial_quantity - caramel_promo
  let total_cost := chocolate_price * initial_quantity +
                    toffee_price * toffee_to_buy +
                    caramel_price * caramel_to_buy
  initial_money - total_cost = 72

/-- Theorem stating the solution to the candy store problem --/
theorem solve_candy_store_problem :
  candy_store_problem 3 5 10 8 200 :=
sorry


end NUMINAMATH_CALUDE_solve_candy_store_problem_l3931_393127


namespace NUMINAMATH_CALUDE_estimate_sqrt_expression_l3931_393116

theorem estimate_sqrt_expression :
  7 < Real.sqrt 32 * Real.sqrt (1/2) + Real.sqrt 12 ∧
  Real.sqrt 32 * Real.sqrt (1/2) + Real.sqrt 12 < 8 :=
by
  sorry

end NUMINAMATH_CALUDE_estimate_sqrt_expression_l3931_393116


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3931_393125

theorem fraction_to_decimal : (45 : ℚ) / 72 = 0.625 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3931_393125


namespace NUMINAMATH_CALUDE_cone_volume_from_circle_sector_l3931_393176

/-- The volume of a right circular cone formed by rolling a two-thirds sector of a circle -/
theorem cone_volume_from_circle_sector (r : ℝ) (h : r = 6) :
  let sector_fraction : ℝ := 2 / 3
  let base_radius : ℝ := r * sector_fraction
  let cone_height : ℝ := (r^2 - base_radius^2).sqrt
  let cone_volume : ℝ := (1/3) * π * base_radius^2 * cone_height
  cone_volume = (32/3) * π * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_from_circle_sector_l3931_393176


namespace NUMINAMATH_CALUDE_larger_square_area_l3931_393169

theorem larger_square_area (small_side : ℝ) (small_triangles : ℕ) (large_triangles : ℕ) :
  small_side = 12 →
  small_triangles = 16 →
  large_triangles = 18 →
  (large_triangles : ℝ) / (small_triangles : ℝ) * (small_side ^ 2) = 162 := by
  sorry

end NUMINAMATH_CALUDE_larger_square_area_l3931_393169


namespace NUMINAMATH_CALUDE_sphere_intersection_area_ratio_l3931_393165

theorem sphere_intersection_area_ratio (R : ℝ) (h : R > 0) :
  let r := Real.sqrt ((3 / 4) * R^2)
  let circle_area := π * r^2
  let sphere_surface_area := 4 * π * R^2
  circle_area / sphere_surface_area = 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_sphere_intersection_area_ratio_l3931_393165


namespace NUMINAMATH_CALUDE_largest_fraction_l3931_393171

theorem largest_fraction :
  let a := (2 : ℚ) / 5
  let b := (4 : ℚ) / 9
  let c := (7 : ℚ) / 15
  let d := (11 : ℚ) / 18
  let e := (16 : ℚ) / 35
  d > a ∧ d > b ∧ d > c ∧ d > e := by
  sorry

end NUMINAMATH_CALUDE_largest_fraction_l3931_393171


namespace NUMINAMATH_CALUDE_min_n_for_divisibility_by_20_l3931_393144

theorem min_n_for_divisibility_by_20 :
  ∃ (n : ℕ), n > 0 ∧
  (∀ (S : Finset ℕ), S.card = n →
    ∃ (a b c d : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    20 ∣ (a + b - c - d)) ∧
  (∀ (m : ℕ), m < n →
    ∃ (T : Finset ℕ), T.card = m ∧
    ¬∃ (a b c d : ℕ), a ∈ T ∧ b ∈ T ∧ c ∈ T ∧ d ∈ T ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    20 ∣ (a + b - c - d)) ∧
  n = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_n_for_divisibility_by_20_l3931_393144


namespace NUMINAMATH_CALUDE_marbles_combination_l3931_393136

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem marbles_combination :
  choose 10 4 = 210 := by sorry

end NUMINAMATH_CALUDE_marbles_combination_l3931_393136


namespace NUMINAMATH_CALUDE_inverse_proportional_symmetry_axis_l3931_393141

theorem inverse_proportional_symmetry_axis (k : ℝ) (h1 : k ≠ 0) (h2 : k ≠ 1) :
  ∃ (f : ℝ → ℝ), (∀ x ≠ 0, f x = k / x) ∧
  (∀ x ≠ 0, ∃ y, f y = f x ∧ (y + x) * (-k / |k|) = y - x) :=
sorry

end NUMINAMATH_CALUDE_inverse_proportional_symmetry_axis_l3931_393141


namespace NUMINAMATH_CALUDE_leadership_structure_count_correct_l3931_393138

def colony_size : Nat := 35
def num_deputy_governors : Nat := 3
def lieutenants_per_deputy : Nat := 3
def subordinates_per_lieutenant : Nat := 2

def leadership_structure_count : Nat :=
  colony_size * 
  Nat.choose (colony_size - 1) num_deputy_governors *
  Nat.choose (colony_size - 1 - num_deputy_governors) lieutenants_per_deputy *
  Nat.choose (colony_size - 1 - num_deputy_governors - lieutenants_per_deputy) lieutenants_per_deputy *
  Nat.choose (colony_size - 1 - num_deputy_governors - 2 * lieutenants_per_deputy) lieutenants_per_deputy *
  Nat.choose (colony_size - 1 - num_deputy_governors - 3 * lieutenants_per_deputy) 2 *
  Nat.choose (colony_size - 1 - num_deputy_governors - 3 * lieutenants_per_deputy - 2) 2 *
  Nat.choose (colony_size - 1 - num_deputy_governors - 3 * lieutenants_per_deputy - 4) 2 *
  Nat.choose (colony_size - 1 - num_deputy_governors - 3 * lieutenants_per_deputy - 6) 2 *
  Nat.choose (colony_size - 1 - num_deputy_governors - 3 * lieutenants_per_deputy - 8) 2 *
  Nat.choose (colony_size - 1 - num_deputy_governors - 3 * lieutenants_per_deputy - 10) 2 *
  Nat.choose (colony_size - 1 - num_deputy_governors - 3 * lieutenants_per_deputy - 12) 2 *
  Nat.choose (colony_size - 1 - num_deputy_governors - 3 * lieutenants_per_deputy - 14) 2

theorem leadership_structure_count_correct : 
  leadership_structure_count = 35 * 5984 * 4495 * 3276 * 2300 * 120 * 91 * 66 * 45 * 28 * 15 * 6 * 1 :=
by sorry

end NUMINAMATH_CALUDE_leadership_structure_count_correct_l3931_393138


namespace NUMINAMATH_CALUDE_laundry_earnings_for_three_days_l3931_393161

def laundry_earnings (charge_per_kilo : ℝ) (day1_kilos : ℝ) : ℝ :=
  let day2_kilos := day1_kilos + 5
  let day3_kilos := 2 * day2_kilos
  charge_per_kilo * (day1_kilos + day2_kilos + day3_kilos)

theorem laundry_earnings_for_three_days :
  laundry_earnings 2 5 = 70 := by
  sorry

end NUMINAMATH_CALUDE_laundry_earnings_for_three_days_l3931_393161
