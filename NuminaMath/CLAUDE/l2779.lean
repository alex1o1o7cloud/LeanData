import Mathlib

namespace NUMINAMATH_CALUDE_fish_given_by_ben_l2779_277976

theorem fish_given_by_ben (initial_fish : ℕ) (current_fish : ℕ) 
  (h1 : initial_fish = 31) (h2 : current_fish = 49) : 
  current_fish - initial_fish = 18 := by
  sorry

end NUMINAMATH_CALUDE_fish_given_by_ben_l2779_277976


namespace NUMINAMATH_CALUDE_floor_subtraction_inequality_l2779_277910

theorem floor_subtraction_inequality (x y : ℝ) : 
  ⌊x - y⌋ ≤ ⌊x⌋ - ⌊y⌋ := by sorry

end NUMINAMATH_CALUDE_floor_subtraction_inequality_l2779_277910


namespace NUMINAMATH_CALUDE_fourth_root_over_seventh_root_of_seven_l2779_277944

theorem fourth_root_over_seventh_root_of_seven (x : ℝ) (hx : x > 0) :
  (x^(1/4)) / (x^(1/7)) = x^(3/28) :=
by sorry

end NUMINAMATH_CALUDE_fourth_root_over_seventh_root_of_seven_l2779_277944


namespace NUMINAMATH_CALUDE_lottery_probability_l2779_277926

/-- The probability of exactly one person winning a prize in a lottery. -/
theorem lottery_probability : 
  let total_tickets : ℕ := 3
  let winning_tickets : ℕ := 2
  let people_drawing : ℕ := 2
  -- Probability of exactly one person winning
  (1 : ℚ) - (winning_tickets : ℚ) / (total_tickets : ℚ) * ((winning_tickets - 1) : ℚ) / ((total_tickets - 1) : ℚ) = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_lottery_probability_l2779_277926


namespace NUMINAMATH_CALUDE_constant_term_expansion_l2779_277941

theorem constant_term_expansion (x : ℝ) : 
  let expression := (Real.sqrt x + 2) * (1 / Real.sqrt x - 1)^5
  ∃ (p : ℝ → ℝ), expression = p x ∧ p 0 = 3 :=
by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l2779_277941


namespace NUMINAMATH_CALUDE_reciprocal_not_one_others_are_l2779_277906

theorem reciprocal_not_one_others_are (x : ℝ) (hx : x = -1) : 
  (x⁻¹ ≠ 1) ∧ (-x = 1) ∧ (|x| = 1) ∧ (x^2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_not_one_others_are_l2779_277906


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l2779_277932

def vector_a : ℝ × ℝ := (1, 2)
def vector_b (m : ℝ) : ℝ × ℝ := (-2, m)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ v.1 = k * w.1 ∧ v.2 = k * w.2

theorem parallel_vectors_m_value :
  parallel vector_a (vector_b m) → m = -4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l2779_277932


namespace NUMINAMATH_CALUDE_insects_distribution_l2779_277981

/-- The number of insects collected by boys -/
def boys_insects : ℕ := 200

/-- The number of insects collected by girls -/
def girls_insects : ℕ := 300

/-- The number of groups the class is divided into -/
def num_groups : ℕ := 4

/-- The total number of insects collected -/
def total_insects : ℕ := boys_insects + girls_insects

/-- The number of insects per group -/
def insects_per_group : ℕ := total_insects / num_groups

theorem insects_distribution :
  insects_per_group = 125 := by sorry

end NUMINAMATH_CALUDE_insects_distribution_l2779_277981


namespace NUMINAMATH_CALUDE_lizette_stamps_l2779_277997

/-- Given that Lizette has 125 more stamps than Minerva and Minerva has 688 stamps,
    prove that Lizette has 813 stamps. -/
theorem lizette_stamps (minerva_stamps : ℕ) (lizette_extra : ℕ) 
  (h1 : minerva_stamps = 688)
  (h2 : lizette_extra = 125) : 
  minerva_stamps + lizette_extra = 813 := by
  sorry

end NUMINAMATH_CALUDE_lizette_stamps_l2779_277997


namespace NUMINAMATH_CALUDE_candy_box_distribution_l2779_277922

theorem candy_box_distribution :
  ∃ (x y z : ℕ), 
    x * 16 + y * 17 + z * 21 = 185 ∧ 
    x = 5 ∧ 
    y = 0 ∧ 
    z = 5 :=
by sorry

end NUMINAMATH_CALUDE_candy_box_distribution_l2779_277922


namespace NUMINAMATH_CALUDE_baseball_average_calculation_l2779_277913

/-- Proves the required average for the remaining games to achieve a target season average -/
theorem baseball_average_calculation
  (total_games : ℕ)
  (completed_games : ℕ)
  (remaining_games : ℕ)
  (current_average : ℚ)
  (target_average : ℚ)
  (h_total : total_games = completed_games + remaining_games)
  (h_completed : completed_games = 20)
  (h_remaining : remaining_games = 10)
  (h_current : current_average = 2)
  (h_target : target_average = 3) :
  (target_average * total_games - current_average * completed_games) / remaining_games = 5 := by
  sorry

#check baseball_average_calculation

end NUMINAMATH_CALUDE_baseball_average_calculation_l2779_277913


namespace NUMINAMATH_CALUDE_simplify_expression_l2779_277971

theorem simplify_expression (y : ℝ) :
  3 * y + 7 * y^2 + 10 - (5 - 3 * y - 7 * y^2) = 14 * y^2 + 6 * y + 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2779_277971


namespace NUMINAMATH_CALUDE_paper_cutting_theorem_smallest_over_2000_exactly_2005_exists_l2779_277939

/-- Represents the number of pieces cut in each step -/
def CutSequence := List Nat

/-- Calculates the total number of pieces after a sequence of cuts -/
def totalPieces (cuts : CutSequence) : Nat :=
  1 + 4 * (1 + cuts.sum)

theorem paper_cutting_theorem (cuts : CutSequence) :
  ∃ (k : Nat), totalPieces cuts = 4 * k + 1 :=
sorry

theorem smallest_over_2000 :
  ∀ (cuts : CutSequence),
    totalPieces cuts > 2000 →
    totalPieces cuts ≥ 2005 :=
sorry

theorem exactly_2005_exists :
  ∃ (cuts : CutSequence), totalPieces cuts = 2005 :=
sorry

end NUMINAMATH_CALUDE_paper_cutting_theorem_smallest_over_2000_exactly_2005_exists_l2779_277939


namespace NUMINAMATH_CALUDE_triangle_problem_l2779_277908

theorem triangle_problem (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi →
  -- Sides a, b, c are opposite to angles A, B, C respectively
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Given condition
  (Real.cos C) / (Real.cos B) = (3 * a - c) / b →
  -- Part 1: Value of sin B
  Real.sin B = (2 * Real.sqrt 2) / 3 ∧
  -- Part 2: Area of triangle ABC when b = 4√2 and a = c
  (b = 4 * Real.sqrt 2 ∧ a = c →
    (1/2) * a * c * Real.sin B = 8 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l2779_277908


namespace NUMINAMATH_CALUDE_angle_through_point_l2779_277980

theorem angle_through_point (α : Real) : 
  0 ≤ α ∧ α ≤ 2 * Real.pi → 
  (∃ r : Real, r > 0 ∧ r * Real.cos α = Real.cos (2 * Real.pi / 3) ∧ 
                      r * Real.sin α = Real.sin (2 * Real.pi / 3)) → 
  α = 5 * Real.pi / 3 := by
sorry

end NUMINAMATH_CALUDE_angle_through_point_l2779_277980


namespace NUMINAMATH_CALUDE_integer_ratio_condition_l2779_277915

theorem integer_ratio_condition (m n : ℕ) (hm : m ≥ 3) (hn : n ≥ 3) :
  (∃ (S : Set ℕ), Set.Infinite S ∧ ∀ a ∈ S, ∃ k : ℤ, (a^m + a - 1 : ℤ) = k * (a^n + a^2 - 1)) →
  (m = n + 2 ∧ m = 5 ∧ n = 3) :=
by sorry

end NUMINAMATH_CALUDE_integer_ratio_condition_l2779_277915


namespace NUMINAMATH_CALUDE_identify_radioactive_balls_l2779_277995

/-- A device that tests two balls for radioactivity -/
structure RadioactivityTester :=
  (test : Fin 100 → Fin 100 → Bool)
  (test_correct : ∀ a b, test a b = true ↔ (a.val < 51 ∧ b.val < 51))

/-- A strategy to identify radioactive balls -/
def IdentificationStrategy := RadioactivityTester → Fin 100 → Bool

/-- The number of tests performed by a strategy -/
def num_tests (strategy : IdentificationStrategy) (tester : RadioactivityTester) : ℕ :=
  sorry

theorem identify_radioactive_balls :
  ∃ (strategy : IdentificationStrategy),
    ∀ (tester : RadioactivityTester),
      (∀ i, strategy tester i = true ↔ i.val < 51) ∧
      num_tests strategy tester ≤ 145 :=
sorry

end NUMINAMATH_CALUDE_identify_radioactive_balls_l2779_277995


namespace NUMINAMATH_CALUDE_goldfish_count_l2779_277936

/-- Represents the number of fish in each tank -/
structure FishTanks where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Represents the composition of the first tank -/
structure FirstTank where
  goldfish : ℕ
  beta : ℕ

/-- The problem statement -/
theorem goldfish_count (tanks : FishTanks) (first : FirstTank) : 
  tanks.first = first.goldfish + first.beta ∧
  tanks.second = 2 * tanks.first ∧
  tanks.third = tanks.second / 3 ∧
  tanks.third = 10 ∧
  first.beta = 8 →
  first.goldfish = 7 := by
  sorry

end NUMINAMATH_CALUDE_goldfish_count_l2779_277936


namespace NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l2779_277933

theorem quadratic_equation_unique_solution (a c : ℝ) : 
  (∃! x : ℝ, a * x^2 + 10 * x + c = 0) →
  a + c = 17 →
  a > c →
  a = 15.375 ∧ c = 1.625 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l2779_277933


namespace NUMINAMATH_CALUDE_mary_max_earnings_l2779_277982

/-- Calculates Mary's weekly earnings based on her work hours and pay rates. -/
def maryEarnings (maxHours : Nat) (regularRate : ℚ) (overtimeRate : ℚ) (additionalRate : ℚ) : ℚ :=
  let regularHours := min maxHours 40
  let overtimeHours := min (maxHours - regularHours) 20
  let additionalHours := maxHours - regularHours - overtimeHours
  regularHours * regularRate + overtimeHours * overtimeRate + additionalHours * additionalRate

/-- Theorem stating Mary's earnings for working the maximum hours in a week. -/
theorem mary_max_earnings :
  let maxHours : Nat := 70
  let regularRate : ℚ := 10
  let overtimeRate : ℚ := regularRate * (1 + 30/100)
  let additionalRate : ℚ := regularRate * (1 + 60/100)
  maryEarnings maxHours regularRate overtimeRate additionalRate = 820 := by
  sorry


end NUMINAMATH_CALUDE_mary_max_earnings_l2779_277982


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2779_277999

theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_orthogonal : (x - 1) * 1 + 3 * y = 0) :
  (1 / x + 1 / (3 * y)) ≥ 4 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ (x - 1) * 1 + 3 * y = 0 ∧ 1 / x + 1 / (3 * y) = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2779_277999


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcd_six_exists_192_with_gcd_six_no_greater_than_192_solution_is_192_l2779_277938

theorem greatest_integer_with_gcd_six (n : ℕ) : n < 200 ∧ Nat.gcd n 18 = 6 → n ≤ 192 :=
by sorry

theorem exists_192_with_gcd_six : Nat.gcd 192 18 = 6 :=
by sorry

theorem no_greater_than_192 :
  ∀ m : ℕ, 192 < m → m < 200 → Nat.gcd m 18 ≠ 6 :=
by sorry

theorem solution_is_192 : 
  ∃! n : ℕ, n < 200 ∧ Nat.gcd n 18 = 6 ∧ ∀ m : ℕ, m < 200 ∧ Nat.gcd m 18 = 6 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcd_six_exists_192_with_gcd_six_no_greater_than_192_solution_is_192_l2779_277938


namespace NUMINAMATH_CALUDE_bisection_method_solution_l2779_277967

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the theorem
theorem bisection_method_solution (h1 : f 2 < 0) (h2 : f 3 > 0) (h3 : f 2.5 < 0)
  (h4 : f 2.75 > 0) (h5 : f 2.625 > 0) (h6 : f 2.5625 > 0) :
  ∃ x : ℝ, x ∈ Set.Ioo 2.5 2.5625 ∧ |f x| < 0.1 :=
by
  sorry


end NUMINAMATH_CALUDE_bisection_method_solution_l2779_277967


namespace NUMINAMATH_CALUDE_average_of_polynomials_l2779_277993

theorem average_of_polynomials (x : ℚ) : 
  (1 / 3 : ℚ) * ((x^2 - 3*x + 2) + (3*x^2 + x - 1) + (2*x^2 - 5*x + 7)) = 2*x^2 + 4 →
  x = -4/7 := by
sorry

end NUMINAMATH_CALUDE_average_of_polynomials_l2779_277993


namespace NUMINAMATH_CALUDE_cos_2a_over_1_plus_sin_2a_l2779_277948

theorem cos_2a_over_1_plus_sin_2a (a : ℝ) (h : 4 * Real.sin a = 3 * Real.cos a) :
  (Real.cos (2 * a)) / (1 + Real.sin (2 * a)) = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_cos_2a_over_1_plus_sin_2a_l2779_277948


namespace NUMINAMATH_CALUDE_no_solution_equation_l2779_277994

theorem no_solution_equation : ¬∃ (x : ℝ), (8 / (x^2 - 4) + 1 = x / (x - 2)) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_equation_l2779_277994


namespace NUMINAMATH_CALUDE_call_service_comparison_l2779_277918

/-- Represents the cost of a phone call service -/
structure CallService where
  monthly_fee : ℝ
  per_minute_rate : ℝ

/-- Calculates the total cost for a given call duration -/
def total_cost (service : CallService) (duration : ℝ) : ℝ :=
  service.monthly_fee + service.per_minute_rate * duration

/-- Global Call service -/
def global_call : CallService :=
  { monthly_fee := 50, per_minute_rate := 0.4 }

/-- China Mobile service -/
def china_mobile : CallService :=
  { monthly_fee := 0, per_minute_rate := 0.6 }

theorem call_service_comparison :
  ∃ (x : ℝ),
    (∀ (duration : ℝ), total_cost global_call duration = 50 + 0.4 * duration) ∧
    (∀ (duration : ℝ), total_cost china_mobile duration = 0.6 * duration) ∧
    (total_cost global_call x = total_cost china_mobile x ∧ x = 125) ∧
    (∀ (duration : ℝ), duration > 125 → total_cost global_call duration < total_cost china_mobile duration) :=
by sorry

end NUMINAMATH_CALUDE_call_service_comparison_l2779_277918


namespace NUMINAMATH_CALUDE_teacher_estimate_difference_l2779_277962

/-- The difference between the teacher's estimated increase and the actual increase in exam scores -/
theorem teacher_estimate_difference (expected_increase actual_increase : ℕ) 
  (h1 : expected_increase = 2152)
  (h2 : actual_increase = 1264) : 
  expected_increase - actual_increase = 888 := by
  sorry

end NUMINAMATH_CALUDE_teacher_estimate_difference_l2779_277962


namespace NUMINAMATH_CALUDE_garden_perimeter_l2779_277974

/-- The perimeter of a rectangle given its length and width -/
def rectanglePerimeter (length width : ℝ) : ℝ := 2 * (length + width)

/-- Theorem: The perimeter of a rectangular garden with length 25 meters and width 15 meters is 80 meters -/
theorem garden_perimeter :
  rectanglePerimeter 25 15 = 80 := by
  sorry

end NUMINAMATH_CALUDE_garden_perimeter_l2779_277974


namespace NUMINAMATH_CALUDE_darryl_honeydews_l2779_277900

/-- Represents the problem of determining the initial number of honeydews --/
def honeydew_problem (initial_cantaloupes : ℕ) (final_cantaloupes : ℕ) (final_honeydews : ℕ)
  (dropped_cantaloupes : ℕ) (rotten_honeydews : ℕ) (cantaloupe_price : ℕ) (honeydew_price : ℕ)
  (total_revenue : ℕ) : Prop :=
  ∃ (initial_honeydews : ℕ),
    -- Revenue calculation
    (initial_cantaloupes - dropped_cantaloupes - final_cantaloupes) * cantaloupe_price +
    (initial_honeydews - rotten_honeydews - final_honeydews) * honeydew_price = total_revenue

theorem darryl_honeydews :
  honeydew_problem 30 8 9 2 3 2 3 85 →
  ∃ (initial_honeydews : ℕ), initial_honeydews = 27 :=
sorry

end NUMINAMATH_CALUDE_darryl_honeydews_l2779_277900


namespace NUMINAMATH_CALUDE_expression_evaluation_l2779_277966

theorem expression_evaluation :
  let d : ℕ := 4
  (d^d - d*(d - 2)^d + d^2)^d = 1874164224 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2779_277966


namespace NUMINAMATH_CALUDE_binomial_divisibility_l2779_277988

theorem binomial_divisibility (x n : ℕ) : 
  x = 5 → n = 4 → ∃ k : ℤ, (1 + x)^n - 1 = 7 * k := by
  sorry

end NUMINAMATH_CALUDE_binomial_divisibility_l2779_277988


namespace NUMINAMATH_CALUDE_fraction_equality_l2779_277985

theorem fraction_equality (q r s t : ℚ) 
  (h1 : q / r = 12)
  (h2 : s / r = 8)
  (h3 : s / t = 1 / 3) :
  t / q = 2 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l2779_277985


namespace NUMINAMATH_CALUDE_parallelogram_area_l2779_277973

-- Define the parallelogram and its properties
structure Parallelogram :=
  (area : ℝ)
  (inscribed_circles : ℕ)
  (circle_radius : ℝ)
  (touching_sides : ℕ)
  (vertex_to_tangency : ℝ)

-- Define the conditions of the problem
def problem_conditions (p : Parallelogram) : Prop :=
  p.inscribed_circles = 2 ∧
  p.circle_radius = 1 ∧
  p.touching_sides = 3 ∧
  p.vertex_to_tangency = Real.sqrt 3

-- Theorem statement
theorem parallelogram_area 
  (p : Parallelogram) 
  (h : problem_conditions p) : 
  p.area = 4 * (1 + Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l2779_277973


namespace NUMINAMATH_CALUDE_gain_percent_problem_l2779_277945

def gain_percent (gain : ℚ) (cost_price : ℚ) : ℚ :=
  (gain / cost_price) * 100

theorem gain_percent_problem (gain : ℚ) (cost_price : ℚ) 
  (h1 : gain = 70 / 100)  -- 70 paise = 0.70 rupees
  (h2 : cost_price = 70) : 
  gain_percent gain cost_price = 1 := by
  sorry

end NUMINAMATH_CALUDE_gain_percent_problem_l2779_277945


namespace NUMINAMATH_CALUDE_multiplication_value_proof_l2779_277972

theorem multiplication_value_proof : 
  let initial_number : ℝ := 2.25
  let division_factor : ℝ := 3
  let multiplication_value : ℝ := 12
  let result : ℝ := 9
  (initial_number / division_factor) * multiplication_value = result := by
sorry

end NUMINAMATH_CALUDE_multiplication_value_proof_l2779_277972


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2779_277914

theorem quadratic_inequality_solution (y : ℝ) :
  -y^2 + 9*y - 20 < 0 ↔ y < 4 ∨ y > 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2779_277914


namespace NUMINAMATH_CALUDE_purple_top_implies_violet_bottom_l2779_277951

/-- Represents the colors of the cube faces -/
inductive Color
  | R | P | O | Y | G | V

/-- Represents a cube with colored faces -/
structure Cube where
  top : Color
  bottom : Color
  front : Color
  back : Color
  left : Color
  right : Color

/-- Represents the configuration of the six squares before folding -/
structure SquareConfiguration where
  square1 : Color
  square2 : Color
  square3 : Color
  square4 : Color
  square5 : Color
  square6 : Color

/-- Function to fold the squares into a cube -/
def foldIntoCube (config : SquareConfiguration) : Cube :=
  sorry

/-- Theorem stating that if P is on top, V is on the bottom -/
theorem purple_top_implies_violet_bottom (config : SquareConfiguration) :
  let cube := foldIntoCube config
  cube.top = Color.P → cube.bottom = Color.V :=
sorry

end NUMINAMATH_CALUDE_purple_top_implies_violet_bottom_l2779_277951


namespace NUMINAMATH_CALUDE_x_squared_coefficient_l2779_277996

/-- The coefficient of x² in the expansion of (3x² + 4x + 5)(6x² + 7x + 8) is 82 -/
theorem x_squared_coefficient (x : ℝ) : 
  (3*x^2 + 4*x + 5) * (6*x^2 + 7*x + 8) = 18*x^4 + 39*x^3 + 82*x^2 + 67*x + 40 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_coefficient_l2779_277996


namespace NUMINAMATH_CALUDE_prob_sum_le_10_prob_sum_le_10_is_11_12_l2779_277952

/-- The number of sides on each die -/
def numSides : ℕ := 6

/-- The total number of possible outcomes when rolling two dice -/
def totalOutcomes : ℕ := numSides * numSides

/-- The number of outcomes where the sum is greater than 10 -/
def outcomesGreaterThan10 : ℕ := 3

/-- The probability that the sum of two fair six-sided dice is less than or equal to 10 -/
theorem prob_sum_le_10 : ℚ :=
  1 - (outcomesGreaterThan10 : ℚ) / totalOutcomes

/-- Proof that the probability of the sum of two fair six-sided dice being less than or equal to 10 is 11/12 -/
theorem prob_sum_le_10_is_11_12 : prob_sum_le_10 = 11 / 12 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_le_10_prob_sum_le_10_is_11_12_l2779_277952


namespace NUMINAMATH_CALUDE_smallest_constant_for_ratio_difference_l2779_277947

theorem smallest_constant_for_ratio_difference (a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∃ (i j k l : Fin 5), i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧
    |a₁ / a₂ - a₃ / a₄| ≤ (1/2 : ℝ)) ∧
  (∀ C < (1/2 : ℝ), ∃ (b₁ b₂ b₃ b₄ b₅ : ℝ),
    ∀ (i j k l : Fin 5), i ≠ j → i ≠ k → i ≠ l → j ≠ k → j ≠ l → k ≠ l →
      |b₁ / b₂ - b₃ / b₄| > C) :=
by sorry

end NUMINAMATH_CALUDE_smallest_constant_for_ratio_difference_l2779_277947


namespace NUMINAMATH_CALUDE_arrange_five_classes_four_factories_l2779_277956

/-- The number of ways to arrange classes into factories -/
def arrange_classes (num_classes : ℕ) (num_factories : ℕ) : ℕ :=
  (num_classes.choose 2) * (num_factories.factorial)

/-- Theorem: The number of ways to arrange 5 classes into 4 factories is 240 -/
theorem arrange_five_classes_four_factories :
  arrange_classes 5 4 = 240 := by
  sorry

end NUMINAMATH_CALUDE_arrange_five_classes_four_factories_l2779_277956


namespace NUMINAMATH_CALUDE_quadratic_sum_reciprocal_l2779_277929

theorem quadratic_sum_reciprocal (t : ℝ) (h1 : t^2 - 3*t + 1 = 0) (h2 : t ≠ 0) :
  t + 1/t = 3 := by sorry

end NUMINAMATH_CALUDE_quadratic_sum_reciprocal_l2779_277929


namespace NUMINAMATH_CALUDE_problem_solution_l2779_277984

theorem problem_solution (a b : ℝ) (h : |a + 1| + (b - 2)^2 = 0) :
  (a + b)^9 + a^6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2779_277984


namespace NUMINAMATH_CALUDE_expand_product_l2779_277928

theorem expand_product (x : ℝ) : -2 * (x - 3) * (x + 4) * (2*x - 1) = -4*x^3 - 2*x^2 + 50*x - 24 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2779_277928


namespace NUMINAMATH_CALUDE_red_ball_probability_l2779_277917

/-- The probability of drawing a red ball from a pocket containing white, black, and red balls -/
theorem red_ball_probability (white black red : ℕ) (h : red = 1) :
  (red : ℚ) / (white + black + red : ℚ) = 1 / 9 :=
by
  sorry

#check red_ball_probability 3 5 1 rfl

end NUMINAMATH_CALUDE_red_ball_probability_l2779_277917


namespace NUMINAMATH_CALUDE_complex_fraction_equals_2i_l2779_277931

theorem complex_fraction_equals_2i :
  let z : ℂ := 1 + I
  (z^2 - 2*z) / (z - 1) = 2*I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_2i_l2779_277931


namespace NUMINAMATH_CALUDE_first_satellite_launched_by_soviet_union_l2779_277940

-- Define a type for countries
inductive Country
| UnitedStates
| SovietUnion
| EuropeanUnion
| Germany

-- Define a structure for a satellite launch event
structure SatelliteLaunch where
  date : Nat × Nat × Nat  -- (day, month, year)
  country : Country

-- Define the first artificial Earth satellite launch
def firstArtificialSatelliteLaunch : SatelliteLaunch :=
  { date := (4, 10, 1957),
    country := Country.SovietUnion }

-- Theorem statement
theorem first_satellite_launched_by_soviet_union :
  firstArtificialSatelliteLaunch.country = Country.SovietUnion :=
by sorry


end NUMINAMATH_CALUDE_first_satellite_launched_by_soviet_union_l2779_277940


namespace NUMINAMATH_CALUDE_sqrt_3_power_calculation_l2779_277977

theorem sqrt_3_power_calculation : 
  (Real.sqrt ((Real.sqrt 3) ^ 5)) ^ 6 = 2187 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_power_calculation_l2779_277977


namespace NUMINAMATH_CALUDE_physics_class_size_l2779_277989

/-- Proves that the number of students in the physics class is 42 --/
theorem physics_class_size :
  ∀ (total_students : ℕ) 
    (math_only : ℕ) 
    (physics_only : ℕ) 
    (both : ℕ),
  total_students = 53 →
  math_only + physics_only + both = total_students →
  physics_only + both = 2 * (math_only + both) →
  both = 10 →
  physics_only + both = 42 :=
by
  sorry

#check physics_class_size

end NUMINAMATH_CALUDE_physics_class_size_l2779_277989


namespace NUMINAMATH_CALUDE_deck_size_proof_l2779_277953

theorem deck_size_proof (r b : ℕ) : 
  r / (r + b : ℚ) = 1/4 →
  r / (r + (b + 6) : ℚ) = 1/5 →
  r + b = 24 :=
by sorry

end NUMINAMATH_CALUDE_deck_size_proof_l2779_277953


namespace NUMINAMATH_CALUDE_stamp_collection_gcd_l2779_277964

theorem stamp_collection_gcd : Nat.gcd (Nat.gcd 945 1260) 630 = 105 := by
  sorry

end NUMINAMATH_CALUDE_stamp_collection_gcd_l2779_277964


namespace NUMINAMATH_CALUDE_henry_total_score_l2779_277924

def geography_score : ℝ := 50
def math_score : ℝ := 70
def english_score : ℝ := 66
def science_score : ℝ := 84
def french_score : ℝ := 75

def geography_weight : ℝ := 0.25
def math_weight : ℝ := 0.20
def english_weight : ℝ := 0.20
def science_weight : ℝ := 0.15
def french_weight : ℝ := 0.10

def history_score : ℝ :=
  geography_score * geography_weight +
  math_score * math_weight +
  english_score * english_weight +
  science_score * science_weight +
  french_score * french_weight

def total_score : ℝ :=
  geography_score + math_score + english_score + science_score + french_score + history_score

theorem henry_total_score :
  total_score = 404.8 := by
  sorry

end NUMINAMATH_CALUDE_henry_total_score_l2779_277924


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l2779_277935

theorem infinite_geometric_series_first_term
  (r : ℝ) (S : ℝ) (a : ℝ)
  (h_r : r = 1/4)
  (h_S : S = 80)
  (h_sum : S = a / (1 - r))
  : a = 60 := by
  sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l2779_277935


namespace NUMINAMATH_CALUDE_correct_match_probability_l2779_277923

/-- The number of celebrities and baby pictures -/
def n : ℕ := 4

/-- The total number of possible arrangements -/
def total_arrangements : ℕ := n.factorial

/-- The number of correct arrangements -/
def correct_arrangements : ℕ := 1

/-- The probability of correctly matching all celebrities to their baby pictures -/
def probability : ℚ := correct_arrangements / total_arrangements

theorem correct_match_probability :
  probability = 1 / 24 := by sorry

end NUMINAMATH_CALUDE_correct_match_probability_l2779_277923


namespace NUMINAMATH_CALUDE_mouse_jump_distance_l2779_277942

/-- The jump distances of animals in a contest -/
def JumpContest (grasshopper frog mouse : ℕ) : Prop :=
  (grasshopper = 39) ∧ 
  (grasshopper = frog + 19) ∧
  (frog = mouse + 12)

/-- Theorem: Given the conditions of the jump contest, the mouse jumped 8 inches -/
theorem mouse_jump_distance (grasshopper frog mouse : ℕ) 
  (h : JumpContest grasshopper frog mouse) : mouse = 8 := by
  sorry

end NUMINAMATH_CALUDE_mouse_jump_distance_l2779_277942


namespace NUMINAMATH_CALUDE_autumn_pencils_left_l2779_277987

/-- Calculates the number of pencils Autumn has left after various changes --/
def pencils_left (initial : ℕ) (misplaced : ℕ) (broken : ℕ) (found : ℕ) (bought : ℕ) : ℕ :=
  initial - (misplaced + broken) + (found + bought)

/-- Theorem stating that Autumn has 16 pencils left --/
theorem autumn_pencils_left : pencils_left 20 7 3 4 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_autumn_pencils_left_l2779_277987


namespace NUMINAMATH_CALUDE_circle_tangent_to_axes_on_line_l2779_277992

theorem circle_tangent_to_axes_on_line (x y : ℝ) :
  ∃ (a b r : ℝ),
    (∀ t : ℝ, (2 * t - (2 * t + 6) + 6 = 0)) →  -- Center on the line 2x - y + 6 = 0
    (a = -2 ∨ a = -6) →                         -- Possible x-coordinates of the center
    (b = 2 * a + 6) →                           -- y-coordinate of the center
    (r = |a|) →                                 -- Radius equals the absolute value of x-coordinate
    (r = |b|) →                                 -- Radius equals the absolute value of y-coordinate
    ((x + a)^2 + (y - b)^2 = r^2) →             -- Standard form of circle equation
    (((x + 2)^2 + (y - 2)^2 = 4) ∨ ((x + 6)^2 + (y + 6)^2 = 36)) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_to_axes_on_line_l2779_277992


namespace NUMINAMATH_CALUDE_sqrt_19_bounds_l2779_277901

theorem sqrt_19_bounds : 4 < Real.sqrt 19 ∧ Real.sqrt 19 < 5 := by
  have h1 : 16 < 19 := by sorry
  have h2 : 19 < 25 := by sorry
  sorry

end NUMINAMATH_CALUDE_sqrt_19_bounds_l2779_277901


namespace NUMINAMATH_CALUDE_class_point_system_l2779_277949

/-- Calculates the number of tasks required for a given number of points -/
def tasksRequired (points : ℕ) : ℕ :=
  let fullSets := (points - 1) / 3
  let taskMultiplier := min fullSets 2 + 1
  taskMultiplier * ((points + 2) / 3)

/-- The point-earning system for the class -/
theorem class_point_system (points : ℕ) :
  points = 18 → tasksRequired points = 10 :=
by
  sorry

#eval tasksRequired 18  -- Should output 10

end NUMINAMATH_CALUDE_class_point_system_l2779_277949


namespace NUMINAMATH_CALUDE_min_distance_curve_line_l2779_277991

/-- The minimum distance between a point on y = 2ln(x) and a point on y = 2x + 3 is √5 -/
theorem min_distance_curve_line : 
  let curve := (fun x : ℝ => 2 * Real.log x)
  let line := (fun x : ℝ => 2 * x + 3)
  ∃ (M N : ℝ × ℝ), 
    (M.2 = curve M.1) ∧ 
    (N.2 = line N.1) ∧
    (∀ (P Q : ℝ × ℝ), P.2 = curve P.1 → Q.2 = line Q.1 → 
      Real.sqrt 5 ≤ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)) ∧
    Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_min_distance_curve_line_l2779_277991


namespace NUMINAMATH_CALUDE_grid_coloring_l2779_277990

theorem grid_coloring (n : ℕ) (k : ℕ) (h_n_pos : 0 < n) (h_k_bound : k < n^2) :
  (4 * n * k - 2 * n^3 = 50) → (k = 15 ∨ k = 313) := by
  sorry

end NUMINAMATH_CALUDE_grid_coloring_l2779_277990


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l2779_277958

/-- Given a geometric sequence {a_n} where the first three terms are a-2, a+2, and a+8,
    prove that the general term a_n is equal to 8 · (3/2)^(n-1) -/
theorem geometric_sequence_general_term (a : ℝ) (a_n : ℕ → ℝ) :
  (a_n 1 = a - 2) →
  (a_n 2 = a + 2) →
  (a_n 3 = a + 8) →
  (∀ n : ℕ, n ≥ 1 → a_n (n + 1) / a_n n = a_n 2 / a_n 1) →
  (∀ n : ℕ, n ≥ 1 → a_n n = 8 * (3/2)^(n - 1)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l2779_277958


namespace NUMINAMATH_CALUDE_arithmetic_sequence_proof_l2779_277943

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℚ := 2 * n - 1

-- Define the sequence b_n
def b (n : ℕ) : ℚ := 2 / (n * (a n + 3))

-- Define the sum S_n
def S (n : ℕ) : ℚ := n / (n + 1)

theorem arithmetic_sequence_proof :
  (a 3 = 5) ∧ (a 17 = 3 * a 6) ∧
  (∀ n : ℕ, n > 0 → b n = 1 / (n * (n + 1))) ∧
  (∀ n : ℕ, n > 0 → S n = n / (n + 1)) := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_proof_l2779_277943


namespace NUMINAMATH_CALUDE_spinner_probability_l2779_277907

theorem spinner_probability (p_A p_B p_C p_D : ℚ) : 
  p_A = 1/4 → p_B = 1/3 → p_D = 1/6 → p_A + p_B + p_C + p_D = 1 → p_C = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_spinner_probability_l2779_277907


namespace NUMINAMATH_CALUDE_count_four_digit_integers_l2779_277904

/-- The number of distinct four-digit positive integers formed with digits 3, 3, 8, and 8 -/
def fourDigitIntegersCount : ℕ := 6

/-- The set of digits used to form the integers -/
def digits : Finset ℕ := {3, 8}

/-- The number of times each digit is used -/
def digitRepetitions : ℕ := 2

/-- The total number of digits used -/
def totalDigits : ℕ := 4

theorem count_four_digit_integers :
  fourDigitIntegersCount = (totalDigits.factorial) / (digitRepetitions.factorial ^ digits.card) :=
sorry

end NUMINAMATH_CALUDE_count_four_digit_integers_l2779_277904


namespace NUMINAMATH_CALUDE_solution_set_and_minimum_t_l2779_277912

/-- The set of all numerical values of the real number a -/
def M : Set ℝ := {a | ∀ x : ℝ, a * x^2 + a * x + 2 > 0}

theorem solution_set_and_minimum_t :
  (M = {a : ℝ | 0 ≤ a ∧ a < 4}) ∧
  (∃ t₀ : ℝ, t₀ > 0 ∧ ∀ t : ℝ, t > 0 → (∀ a ∈ M, (a^2 - 2*a) * t ≤ t^2 + 3*t - 46) → t ≥ t₀) ∧
  (∀ t : ℝ, t > 0 → (∀ a ∈ M, (a^2 - 2*a) * t ≤ t^2 + 3*t - 46) → t ≥ 46) :=
by sorry

#check solution_set_and_minimum_t

end NUMINAMATH_CALUDE_solution_set_and_minimum_t_l2779_277912


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2779_277916

theorem quadratic_inequality_solution (x : ℝ) : x^2 + 7*x < 12 ↔ -4 < x ∧ x < -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2779_277916


namespace NUMINAMATH_CALUDE_angle_greater_than_120_degrees_l2779_277927

open Real Set

/-- A type representing a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculate the angle between three points -/
def angle (p1 p2 p3 : Point) : ℝ := sorry

/-- The theorem statement -/
theorem angle_greater_than_120_degrees (n : ℕ) (points : Finset Point) :
  points.card = n →
  ∃ (ordered_points : Fin n → Point),
    (∀ i : Fin n, ordered_points i ∈ points) ∧
    (∀ (i j k : Fin n), i < j → j < k →
      angle (ordered_points i) (ordered_points j) (ordered_points k) > 120 * π / 180) :=
sorry

end NUMINAMATH_CALUDE_angle_greater_than_120_degrees_l2779_277927


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2779_277911

theorem simplify_and_evaluate (a b : ℚ) (h1 : a = -2) (h2 : b = 1/3) :
  4 * (a^2 - 2*a*b) - (3*a^2 - 5*a*b + 1) = 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2779_277911


namespace NUMINAMATH_CALUDE_tree_planting_event_l2779_277920

/-- Calculates 60% of the total number of participants in a tree planting event -/
theorem tree_planting_event (boys : ℕ) (girls : ℕ) : 
  boys = 600 →
  girls - boys = 400 →
  girls > boys →
  (boys + girls) * 60 / 100 = 960 := by
sorry

end NUMINAMATH_CALUDE_tree_planting_event_l2779_277920


namespace NUMINAMATH_CALUDE_remaining_distance_l2779_277946

theorem remaining_distance (total_distance driven_distance : ℕ) 
  (h1 : total_distance = 1200)
  (h2 : driven_distance = 768) :
  total_distance - driven_distance = 432 := by
  sorry

end NUMINAMATH_CALUDE_remaining_distance_l2779_277946


namespace NUMINAMATH_CALUDE_determine_key_lock_pairs_l2779_277959

/-- Represents a lock -/
structure Lock :=
  (id : Nat)

/-- Represents a key -/
structure Key :=
  (id : Nat)

/-- Represents a pair of locks that a key can open -/
structure LockPair :=
  (lock1 : Lock)
  (lock2 : Lock)

/-- Represents the result of testing a key on a lock -/
inductive TestResult
  | Opens
  | DoesNotOpen

/-- Represents the state of knowledge about which keys open which locks -/
structure KeyLockState :=
  (locks : Finset Lock)
  (keys : Finset Key)
  (openPairs : Finset (Key × LockPair))

/-- Represents a single test of a key on a lock -/
def test (k : Key) (l : Lock) : TestResult := sorry

/-- The main theorem to prove -/
theorem determine_key_lock_pairs 
  (locks : Finset Lock) 
  (keys : Finset Key) 
  (h1 : locks.card = 4) 
  (h2 : keys.card = 6) 
  (h3 : ∀ k : Key, k ∈ keys → (∃! p : LockPair, p.lock1 ∈ locks ∧ p.lock2 ∈ locks ∧ 
    test k p.lock1 = TestResult.Opens ∧ test k p.lock2 = TestResult.Opens))
  (h4 : ∀ k1 k2 : Key, k1 ∈ keys → k2 ∈ keys → k1 ≠ k2 → 
    ¬∃ p : LockPair, p.lock1 ∈ locks ∧ p.lock2 ∈ locks ∧ 
    test k1 p.lock1 = TestResult.Opens ∧ test k1 p.lock2 = TestResult.Opens ∧
    test k2 p.lock1 = TestResult.Opens ∧ test k2 p.lock2 = TestResult.Opens) :
  ∃ (final_state : KeyLockState) (test_count : Nat),
    test_count ≤ 13 ∧
    final_state.locks = locks ∧
    final_state.keys = keys ∧
    (∀ k : Key, k ∈ keys → 
      ∃! p : LockPair, (k, p) ∈ final_state.openPairs ∧ 
        p.lock1 ∈ locks ∧ p.lock2 ∈ locks ∧
        test k p.lock1 = TestResult.Opens ∧ 
        test k p.lock2 = TestResult.Opens) :=
by
  sorry

end NUMINAMATH_CALUDE_determine_key_lock_pairs_l2779_277959


namespace NUMINAMATH_CALUDE_pride_and_prejudice_watch_time_l2779_277919

/-- The number of hours spent watching a TV series -/
def watch_time (num_episodes : ℕ) (episode_length : ℕ) : ℚ :=
  (num_episodes * episode_length : ℚ) / 60

/-- Theorem: Watching 6 episodes of 50 minutes each takes 5 hours -/
theorem pride_and_prejudice_watch_time :
  watch_time 6 50 = 5 := by sorry

end NUMINAMATH_CALUDE_pride_and_prejudice_watch_time_l2779_277919


namespace NUMINAMATH_CALUDE_integral_problem_l2779_277968

theorem integral_problem (f : ℝ → ℝ) 
  (h1 : ∫ (x : ℝ) in Set.Iic 1, f x = 1)
  (h2 : ∫ (x : ℝ) in Set.Iic 2, f x = -1) :
  ∫ (x : ℝ) in Set.Ioc 1 2, f x = -2 := by
  sorry

end NUMINAMATH_CALUDE_integral_problem_l2779_277968


namespace NUMINAMATH_CALUDE_triangle_max_perimeter_l2779_277961

theorem triangle_max_perimeter (A B C : ℝ) (a b c : ℝ) :
  A = 2 * π / 3 →
  a = 3 →
  A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  a = 2 * Real.sin (A / 2) * Real.sin (B / 2) / Real.sin ((A + B) / 2) →
  b = 2 * Real.sin (B / 2) * Real.sin (C / 2) / Real.sin ((B + C) / 2) →
  c = 2 * Real.sin (C / 2) * Real.sin (A / 2) / Real.sin ((C + A) / 2) →
  (∀ B' C' a' b' c',
    A + B' + C' = π →
    a' > 0 ∧ b' > 0 ∧ c' > 0 →
    a' = 2 * Real.sin (A / 2) * Real.sin (B' / 2) / Real.sin ((A + B') / 2) →
    b' = 2 * Real.sin (B' / 2) * Real.sin (C' / 2) / Real.sin ((B' + C') / 2) →
    c' = 2 * Real.sin (C' / 2) * Real.sin (A / 2) / Real.sin ((C' + A) / 2) →
    a' + b' + c' ≤ a + b + c) →
  a + b + c = 3 + 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_max_perimeter_l2779_277961


namespace NUMINAMATH_CALUDE_f_of_f_of_2_equals_394_l2779_277986

-- Define the function f
def f (x : ℝ) : ℝ := 4 * x^2 - 6

-- State the theorem
theorem f_of_f_of_2_equals_394 : f (f 2) = 394 := by
  sorry

end NUMINAMATH_CALUDE_f_of_f_of_2_equals_394_l2779_277986


namespace NUMINAMATH_CALUDE_convergence_bound_minimal_k_smallest_k_is_five_l2779_277921

def v : ℕ → ℚ
  | 0 => 1/8
  | n + 1 => 3 * v n - 3 * (v n)^2

def M : ℚ := 1/2

theorem convergence_bound (k : ℕ) : k ≥ 5 → |v k - M| ≤ 1/2^500 := by sorry

theorem minimal_k : ∀ j : ℕ, j < 5 → |v j - M| > 1/2^500 := by sorry

theorem smallest_k_is_five : 
  (∃ k : ℕ, |v k - M| ≤ 1/2^500) ∧ 
  (∀ j : ℕ, |v j - M| ≤ 1/2^500 → j ≥ 5) := by sorry

end NUMINAMATH_CALUDE_convergence_bound_minimal_k_smallest_k_is_five_l2779_277921


namespace NUMINAMATH_CALUDE_rock_collecting_contest_l2779_277950

theorem rock_collecting_contest (sydney_initial : ℕ) (conner_initial : ℕ) 
  (conner_day2 : ℕ) (conner_day3 : ℕ) :
  sydney_initial = 837 →
  conner_initial = 723 →
  conner_day2 = 123 →
  conner_day3 = 27 →
  ∃ (sydney_day1 : ℕ),
    sydney_day1 ≤ 4 ∧
    sydney_day1 > 0 ∧
    conner_initial + 8 * sydney_day1 + conner_day2 + conner_day3 ≥ 
    sydney_initial + sydney_day1 + 16 * sydney_day1 ∧
    ∀ (x : ℕ), x > sydney_day1 →
      conner_initial + 8 * x + conner_day2 + conner_day3 < 
      sydney_initial + x + 16 * x :=
by sorry

end NUMINAMATH_CALUDE_rock_collecting_contest_l2779_277950


namespace NUMINAMATH_CALUDE_cube_volume_from_space_diagonal_l2779_277903

theorem cube_volume_from_space_diagonal (d : ℝ) (h : d = 9) : 
  let s := d / Real.sqrt 3
  s^3 = 81 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_space_diagonal_l2779_277903


namespace NUMINAMATH_CALUDE_solve_for_A_l2779_277998

/-- Given that 3ab · A = 6a²b - 9ab², prove that A = 2a - 3b -/
theorem solve_for_A (a b A : ℝ) (h : 3 * a * b * A = 6 * a^2 * b - 9 * a * b^2) :
  A = 2 * a - 3 * b := by
sorry

end NUMINAMATH_CALUDE_solve_for_A_l2779_277998


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2779_277930

theorem simplify_and_evaluate (a : ℝ) (h1 : a ≠ 0) (h2 : a ≠ 1) (h3 : a ≠ -2) :
  ((a^2 + 1) / a - 2) / ((a + 2) * (a - 1) / (a^2 + 2*a)) = a - 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2779_277930


namespace NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l2779_277954

theorem count_integers_satisfying_inequality :
  ∃! (S : Finset ℤ), 
    (∀ n : ℤ, n ∈ S ↔ (Real.sqrt n ≤ Real.sqrt (3 * n - 9) ∧ Real.sqrt (3 * n - 9) < Real.sqrt (n + 8))) ∧
    S.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l2779_277954


namespace NUMINAMATH_CALUDE_students_neither_music_nor_art_l2779_277960

theorem students_neither_music_nor_art 
  (total : ℕ) (music : ℕ) (art : ℕ) (both : ℕ) :
  total = 500 →
  music = 40 →
  art = 20 →
  both = 10 →
  total - (music + art - both) = 450 :=
by
  sorry

end NUMINAMATH_CALUDE_students_neither_music_nor_art_l2779_277960


namespace NUMINAMATH_CALUDE_correct_systematic_sample_l2779_277979

/-- Represents a systematic sample of students -/
structure SystematicSample where
  total : Nat
  sample_size : Nat
  start : Nat
  interval : Nat

/-- Generates the sequence of selected students -/
def generate_sequence (s : SystematicSample) : List Nat :=
  List.range s.sample_size |>.map (fun i => s.start + i * s.interval)

/-- Checks if a sequence is valid for the given systematic sample -/
def is_valid_sequence (s : SystematicSample) (seq : List Nat) : Prop :=
  seq.length = s.sample_size ∧
  seq.all (· ≤ s.total) ∧
  seq = generate_sequence s

theorem correct_systematic_sample :
  let s : SystematicSample := ⟨50, 5, 3, 10⟩
  is_valid_sequence s [3, 13, 23, 33, 43] := by
  sorry

#eval generate_sequence ⟨50, 5, 3, 10⟩

end NUMINAMATH_CALUDE_correct_systematic_sample_l2779_277979


namespace NUMINAMATH_CALUDE_share_price_increase_l2779_277978

theorem share_price_increase (initial_price : ℝ) : 
  let first_quarter_price := initial_price * (1 + 0.2)
  let second_quarter_price := first_quarter_price * (1 + 1/3)
  second_quarter_price = initial_price * (1 + 0.6) := by
sorry

end NUMINAMATH_CALUDE_share_price_increase_l2779_277978


namespace NUMINAMATH_CALUDE_exradii_product_bound_l2779_277969

/-- For any triangle with side lengths a, b, c and exradii r_a, r_b, r_c,
    the product of the exradii does not exceed (3√3/8) times the product of the side lengths. -/
theorem exradii_product_bound (a b c r_a r_b r_c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hr_a : r_a > 0) (hr_b : r_b > 0) (hr_c : r_c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_exradii : r_a * (b + c - a) = r_b * (c + a - b) ∧ 
               r_b * (c + a - b) = r_c * (a + b - c)) : 
  r_a * r_b * r_c ≤ (3 * Real.sqrt 3 / 8) * a * b * c := by
  sorry

#check exradii_product_bound

end NUMINAMATH_CALUDE_exradii_product_bound_l2779_277969


namespace NUMINAMATH_CALUDE_stratified_sample_size_l2779_277937

/-- Represents the population groups in the organization -/
structure Population where
  elderly : Nat
  middleAged : Nat
  young : Nat

/-- Represents a stratified sample -/
structure StratifiedSample where
  total : Nat
  young : Nat

/-- Theorem stating the relationship between the population, sample, and sample size -/
theorem stratified_sample_size 
  (pop : Population)
  (sample : StratifiedSample)
  (h1 : pop.elderly = 20)
  (h2 : pop.middleAged = 120)
  (h3 : pop.young = 100)
  (h4 : sample.young = 10) :
  sample.total = 24 := by
  sorry

#check stratified_sample_size

end NUMINAMATH_CALUDE_stratified_sample_size_l2779_277937


namespace NUMINAMATH_CALUDE_solution_set_theorem_l2779_277957

open Set

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : f (-2) = 2013)
variable (h2 : ∀ x : ℝ, deriv f x < 2 * x)

-- Define the solution set
def solution_set (f : ℝ → ℝ) : Set ℝ :=
  {x : ℝ | f x > x^2 + 2009}

-- State the theorem
theorem solution_set_theorem (f : ℝ → ℝ) (h1 : f (-2) = 2013) (h2 : ∀ x : ℝ, deriv f x < 2 * x) :
  solution_set f = Iio (-2) :=
sorry

end NUMINAMATH_CALUDE_solution_set_theorem_l2779_277957


namespace NUMINAMATH_CALUDE_multiplicative_inverse_137_mod_391_l2779_277925

theorem multiplicative_inverse_137_mod_391 :
  ∃ x : ℕ, x < 391 ∧ (137 * x) % 391 = 1 ∧ x = 294 := by
  sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_137_mod_391_l2779_277925


namespace NUMINAMATH_CALUDE_DR_length_zero_l2779_277963

/-- Rectangle ABCD with inscribed circle ω -/
structure RectangleWithCircle where
  /-- Length of the rectangle -/
  length : ℝ
  /-- Height of the rectangle -/
  height : ℝ
  /-- Center of the inscribed circle -/
  center : ℝ × ℝ
  /-- Radius of the inscribed circle -/
  radius : ℝ
  /-- Point Q where the circle intersects AB -/
  Q : ℝ × ℝ
  /-- Point D at the bottom left corner -/
  D : ℝ × ℝ
  /-- Point R where DQ intersects the circle again -/
  R : ℝ × ℝ
  /-- The rectangle has length 2 and height 1 -/
  h_dimensions : length = 2 ∧ height = 1
  /-- The circle is inscribed in the rectangle -/
  h_inscribed : center = (0, 0) ∧ radius = height / 2
  /-- Q is on the top edge of the rectangle -/
  h_Q_on_top : Q.2 = height / 2
  /-- D is at the bottom left corner -/
  h_D_position : D = (0, -height / 2)
  /-- R is on the circle -/
  h_R_on_circle : (R.1 - center.1)^2 + (R.2 - center.2)^2 = radius^2
  /-- R is on line DQ -/
  h_R_on_DQ : R.1 = D.1 ∧ R.1 = Q.1

/-- The main theorem: DR has length 0 -/
theorem DR_length_zero (rect : RectangleWithCircle) : dist rect.D rect.R = 0 :=
  sorry


end NUMINAMATH_CALUDE_DR_length_zero_l2779_277963


namespace NUMINAMATH_CALUDE_triangle_solution_l2779_277965

theorem triangle_solution (a b c A B C : ℝ) : 
  a = 2 * Real.sqrt 2 →
  A = π / 4 →
  B = π / 6 →
  C = π - A - B →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  b = 2 ∧ c = Real.sqrt 6 + Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_solution_l2779_277965


namespace NUMINAMATH_CALUDE_odd_score_probability_is_four_ninths_l2779_277955

/-- Represents the possible points on the dart board -/
inductive DartPoints
  | Three
  | Four

/-- Represents the regions on the dart board -/
structure DartRegion where
  isInner : Bool
  points : DartPoints

/-- The dart board configuration -/
def dartBoard : List DartRegion :=
  [
    { isInner := true,  points := DartPoints.Three },
    { isInner := true,  points := DartPoints.Four },
    { isInner := true,  points := DartPoints.Four },
    { isInner := false, points := DartPoints.Four },
    { isInner := false, points := DartPoints.Three },
    { isInner := false, points := DartPoints.Three }
  ]

/-- The probability of hitting each region -/
def regionProbability (region : DartRegion) : ℚ :=
  if region.isInner then 1 / 21 else 2 / 21

/-- The probability of getting an odd score with two dart throws -/
def oddScoreProbability : ℚ := sorry

/-- Theorem stating that the probability of getting an odd score is 4/9 -/
theorem odd_score_probability_is_four_ninths :
  oddScoreProbability = 4 / 9 := by sorry

end NUMINAMATH_CALUDE_odd_score_probability_is_four_ninths_l2779_277955


namespace NUMINAMATH_CALUDE_three_digit_square_insertion_l2779_277905

theorem three_digit_square_insertion (n : ℕ) : ∃ (a b c : ℕ) (a' b' c' : ℕ),
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a' ≠ b' ∧ b' ≠ c' ∧ a' ≠ c' ∧
  0 < a ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧
  0 < a' ∧ a' < 10 ∧ b' < 10 ∧ c' < 10 ∧
  (a ≠ a' ∨ b ≠ b' ∨ c ≠ c') ∧
  ∃ (k : ℕ), a * 10^(2*n+2) + b * 10^(n+1) + c = k^2 ∧
  ∃ (k' : ℕ), a' * 10^(2*n+2) + b' * 10^(n+1) + c' = k'^2 :=
by sorry

end NUMINAMATH_CALUDE_three_digit_square_insertion_l2779_277905


namespace NUMINAMATH_CALUDE_cherry_weekly_earnings_l2779_277970

/-- Represents the charge for a cargo based on its weight range -/
def charge (weight : ℕ) : ℚ :=
  if 3 ≤ weight ∧ weight ≤ 5 then 5/2
  else if 6 ≤ weight ∧ weight ≤ 8 then 4
  else if 9 ≤ weight ∧ weight ≤ 12 then 6
  else if 13 ≤ weight ∧ weight ≤ 15 then 8
  else 0

/-- Calculates the daily earnings based on the number of deliveries for each weight -/
def dailyEarnings (deliveries : List (ℕ × ℕ)) : ℚ :=
  deliveries.foldl (fun acc (weight, count) => acc + charge weight * count) 0

/-- Cherry's daily delivery schedule -/
def cherryDeliveries : List (ℕ × ℕ) := [(5, 4), (8, 2), (10, 3), (14, 1)]

/-- Number of days in a week -/
def daysInWeek : ℕ := 7

/-- Theorem stating that Cherry's weekly earnings equal $308 -/
theorem cherry_weekly_earnings : 
  dailyEarnings cherryDeliveries * daysInWeek = 308 := by
  sorry

end NUMINAMATH_CALUDE_cherry_weekly_earnings_l2779_277970


namespace NUMINAMATH_CALUDE_joe_egg_count_l2779_277983

/-- The number of eggs Joe found around the club house -/
def club_house_eggs : ℕ := 12

/-- The number of eggs Joe found around the park -/
def park_eggs : ℕ := 5

/-- The number of eggs Joe found in the town hall garden -/
def town_hall_eggs : ℕ := 3

/-- The total number of eggs Joe found -/
def total_eggs : ℕ := club_house_eggs + park_eggs + town_hall_eggs

theorem joe_egg_count : total_eggs = 20 := by
  sorry

end NUMINAMATH_CALUDE_joe_egg_count_l2779_277983


namespace NUMINAMATH_CALUDE_new_profit_percentage_l2779_277975

theorem new_profit_percentage
  (original_profit_rate : ℝ)
  (original_selling_price : ℝ)
  (price_reduction_rate : ℝ)
  (additional_profit : ℝ)
  (h1 : original_profit_rate = 0.1)
  (h2 : original_selling_price = 439.99999999999966)
  (h3 : price_reduction_rate = 0.1)
  (h4 : additional_profit = 28) :
  let original_cost_price := original_selling_price / (1 + original_profit_rate)
  let new_cost_price := original_cost_price * (1 - price_reduction_rate)
  let new_selling_price := original_selling_price + additional_profit
  let new_profit := new_selling_price - new_cost_price
  let new_profit_percentage := (new_profit / new_cost_price) * 100
  new_profit_percentage = 30 := by
sorry

end NUMINAMATH_CALUDE_new_profit_percentage_l2779_277975


namespace NUMINAMATH_CALUDE_quadrilateral_area_relation_l2779_277902

-- Define the quadrilateral ABCD
variable (A B C D : ℝ × ℝ)

-- Define the intersection point of diagonals
def O : ℝ × ℝ := sorry

-- Define a point P inside triangle AOB
variable (P : ℝ × ℝ)

-- Assume P is inside triangle AOB
axiom P_inside_AOB : sorry

-- Define the area function for triangles
def area (X Y Z : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem quadrilateral_area_relation :
  area P C D - area P A B = area P A C + area P B D := by sorry

end NUMINAMATH_CALUDE_quadrilateral_area_relation_l2779_277902


namespace NUMINAMATH_CALUDE_xyz_inequality_l2779_277934

theorem xyz_inequality (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1) :
  0 ≤ y*z + z*x + x*y - 2*x*y*z ∧ y*z + z*x + x*y - 2*x*y*z ≤ 7/27 := by
  sorry

end NUMINAMATH_CALUDE_xyz_inequality_l2779_277934


namespace NUMINAMATH_CALUDE_unicorn_tower_theorem_l2779_277909

/-- Represents the configuration of a unicorn tethered to a cylindrical tower -/
structure UnicornTower where
  rope_length : ℝ
  tower_radius : ℝ
  unicorn_height : ℝ
  rope_end_distance : ℝ

/-- Calculates the length of rope touching the tower -/
def rope_touching_tower (ut : UnicornTower) : ℝ :=
  ut.rope_length - (ut.rope_end_distance + ut.tower_radius)

/-- Theorem stating the properties of the unicorn-tower configuration -/
theorem unicorn_tower_theorem (ut : UnicornTower) 
  (h_rope : ut.rope_length = 20)
  (h_radius : ut.tower_radius = 8)
  (h_height : ut.unicorn_height = 4)
  (h_distance : ut.rope_end_distance = 4) :
  ∃ (a b c : ℕ), 
    c.Prime ∧ 
    rope_touching_tower ut = (a : ℝ) - Real.sqrt b / c ∧
    a = 60 ∧ b = 750 ∧ c = 3 ∧
    a + b + c = 813 := by
  sorry

end NUMINAMATH_CALUDE_unicorn_tower_theorem_l2779_277909
