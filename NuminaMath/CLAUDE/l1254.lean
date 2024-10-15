import Mathlib

namespace NUMINAMATH_CALUDE_equal_division_of_sweets_and_candies_l1254_125456

theorem equal_division_of_sweets_and_candies :
  let num_sweets : ℕ := 72
  let num_candies : ℕ := 56
  let num_people : ℕ := 4
  let sweets_per_person : ℕ := num_sweets / num_people
  let candies_per_person : ℕ := num_candies / num_people
  let total_per_person : ℕ := sweets_per_person + candies_per_person
  total_per_person = 32 := by
  sorry

end NUMINAMATH_CALUDE_equal_division_of_sweets_and_candies_l1254_125456


namespace NUMINAMATH_CALUDE_function_equivalence_l1254_125455

theorem function_equivalence (x : ℝ) : (x^3 + x) / (x^2 + 1) = x := by
  sorry

end NUMINAMATH_CALUDE_function_equivalence_l1254_125455


namespace NUMINAMATH_CALUDE_smallest_mn_for_almost_shaded_square_l1254_125452

theorem smallest_mn_for_almost_shaded_square (m n : ℕ+) 
  (h_bound : 2 * n < m ∧ m < 3 * n) 
  (h_exists : ∃ (p q : ℕ) (k : ℤ), 
    p < m ∧ q < n ∧ 
    0 < (m * q - n * p) * (m * q - n * p) ∧ 
    (m * q - n * p) * (m * q - n * p) < 2 * m * n / 1000) :
  506 ≤ m * n ∧ m * n ≤ 510 := by
sorry

end NUMINAMATH_CALUDE_smallest_mn_for_almost_shaded_square_l1254_125452


namespace NUMINAMATH_CALUDE_exists_m_iff_n_power_of_two_l1254_125462

theorem exists_m_iff_n_power_of_two (n : ℕ+) :
  (∃ m : ℤ, (2^n.val - 1) ∣ (m^2 + 9)) ↔ ∃ k : ℕ, n.val = 2^k :=
sorry

end NUMINAMATH_CALUDE_exists_m_iff_n_power_of_two_l1254_125462


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l1254_125490

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| - 2 * |x + a|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x > 1} = Set.Ioo (-2) (-2/3) := by sorry

-- Part 2
theorem range_of_a_part2 :
  ∀ x ∈ Set.Icc 2 3, (∀ a : ℝ, f a x > 0) → a ∈ Set.Ioo (-5/2) (-2) := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l1254_125490


namespace NUMINAMATH_CALUDE_circle_area_increase_l1254_125414

theorem circle_area_increase (r : ℝ) (h : r > 0) :
  let new_radius := 1.5 * r
  let original_area := π * r^2
  let new_area := π * new_radius^2
  (new_area - original_area) / original_area = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_increase_l1254_125414


namespace NUMINAMATH_CALUDE_when_you_rescind_price_is_85_l1254_125407

/-- The price of a CD of "The Life Journey" -/
def life_journey_price : ℕ := 100

/-- The price of a CD of "A Day a Life" -/
def day_life_price : ℕ := 50

/-- The number of each CD type bought -/
def quantity : ℕ := 3

/-- The total amount spent -/
def total_spent : ℕ := 705

/-- The price of a CD of "When You Rescind" -/
def when_you_rescind_price : ℕ := 85

/-- Theorem stating that the price of "When You Rescind" CD is 85 -/
theorem when_you_rescind_price_is_85 :
  quantity * life_journey_price + quantity * day_life_price + quantity * when_you_rescind_price = total_spent :=
by sorry

end NUMINAMATH_CALUDE_when_you_rescind_price_is_85_l1254_125407


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l1254_125425

theorem quadratic_equation_root (m : ℝ) : 
  (∃ x : ℝ, x^2 + m*x + 3 = 0 ∧ x = 1) → m = -4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l1254_125425


namespace NUMINAMATH_CALUDE_fraction_equality_sum_l1254_125474

theorem fraction_equality_sum (P Q : ℚ) : 
  (5 : ℚ) / 7 = P / 63 ∧ (5 : ℚ) / 7 = 140 / Q → P + Q = 241 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_sum_l1254_125474


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l1254_125487

theorem complex_fraction_sum : (2 + 2 * Complex.I) / Complex.I + (1 + Complex.I) / (1 - Complex.I) = 2 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l1254_125487


namespace NUMINAMATH_CALUDE_three_dice_probability_l1254_125437

/-- A fair 6-sided die -/
def Die : Type := Fin 6

/-- A roll of three dice -/
def ThreeDiceRoll : Type := Die × Die × Die

/-- The total number of possible outcomes when rolling three 6-sided dice -/
def totalOutcomes : ℕ := 216

/-- The number of permutations of three distinct numbers -/
def permutations : ℕ := 6

/-- The probability of rolling a 2, 3, and 4 in any order with three fair 6-sided dice -/
def winProbability : ℚ := 1 / 36

/-- Theorem: The probability of rolling a 2, 3, and 4 in any order with three fair 6-sided dice is 1/36 -/
theorem three_dice_probability :
  (permutations : ℚ) / totalOutcomes = winProbability :=
sorry

end NUMINAMATH_CALUDE_three_dice_probability_l1254_125437


namespace NUMINAMATH_CALUDE_cyclists_speed_problem_l1254_125426

/-- Two cyclists problem -/
theorem cyclists_speed_problem (north_speed : ℝ) (time : ℝ) (distance : ℝ) : 
  north_speed = 10 →
  time = 1.4285714285714286 →
  distance = 50 →
  ∃ (south_speed : ℝ), south_speed = 25 ∧ (north_speed + south_speed) * time = distance :=
by sorry

end NUMINAMATH_CALUDE_cyclists_speed_problem_l1254_125426


namespace NUMINAMATH_CALUDE_upload_time_calculation_l1254_125417

def file_size : ℝ := 160
def upload_speed : ℝ := 8

theorem upload_time_calculation : 
  file_size / upload_speed = 20 := by sorry

end NUMINAMATH_CALUDE_upload_time_calculation_l1254_125417


namespace NUMINAMATH_CALUDE_right_triangle_from_sine_condition_l1254_125480

theorem right_triangle_from_sine_condition (A B C : ℝ) :
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  A + B + C = π →
  (Real.sin A + Real.sin B) * (Real.sin A - Real.sin B) = (Real.sin C)^2 →
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_from_sine_condition_l1254_125480


namespace NUMINAMATH_CALUDE_jade_transactions_jade_transactions_proof_l1254_125491

/-- Proves that Jade handled 80 transactions given the specified conditions. -/
theorem jade_transactions : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun mabel_transactions anthony_transactions cal_transactions jade_transactions =>
    mabel_transactions = 90 →
    anthony_transactions = mabel_transactions + mabel_transactions / 10 →
    cal_transactions = anthony_transactions * 2 / 3 →
    jade_transactions = cal_transactions + 14 →
    jade_transactions = 80

/-- Proof of the theorem -/
theorem jade_transactions_proof : jade_transactions 90 99 66 80 := by
  sorry

end NUMINAMATH_CALUDE_jade_transactions_jade_transactions_proof_l1254_125491


namespace NUMINAMATH_CALUDE_complex_sum_theorem_l1254_125419

theorem complex_sum_theorem (a b c d e f g h : ℝ) : 
  b = 2 → 
  g = -(a + c + e) → 
  (a + b * Complex.I) + (c + d * Complex.I) + (e + f * Complex.I) + (g + h * Complex.I) = 3 * Complex.I → 
  d + f + h = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_theorem_l1254_125419


namespace NUMINAMATH_CALUDE_seeds_in_second_plot_is_200_l1254_125465

/-- The number of seeds planted in the second plot -/
def seeds_in_second_plot : ℕ := 200

/-- The number of seeds planted in the first plot -/
def seeds_in_first_plot : ℕ := 500

/-- The germination rate of seeds in the first plot -/
def germination_rate_first : ℚ := 30 / 100

/-- The germination rate of seeds in the second plot -/
def germination_rate_second : ℚ := 50 / 100

/-- The total germination rate of all seeds -/
def total_germination_rate : ℚ := 35714285714285715 / 100000000000000000

theorem seeds_in_second_plot_is_200 : 
  (germination_rate_first * seeds_in_first_plot + 
   germination_rate_second * seeds_in_second_plot) / 
  (seeds_in_first_plot + seeds_in_second_plot) = total_germination_rate :=
sorry

end NUMINAMATH_CALUDE_seeds_in_second_plot_is_200_l1254_125465


namespace NUMINAMATH_CALUDE_equation_two_distinct_roots_l1254_125445

theorem equation_two_distinct_roots (k : ℂ) : 
  (∃ (x y : ℂ), x ≠ y ∧ 
    (∀ z : ℂ, z / (z + 3) + z / (z - 1) = k * z ↔ z = x ∨ z = y)) ↔ 
  k ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_equation_two_distinct_roots_l1254_125445


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_circle_through_origin_l1254_125450

-- Define the circle C
def circle_C (x y m : ℝ) : Prop := x^2 + y^2 + x - 6*y + m = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y - 3 = 0

-- Theorem for part I
theorem circle_tangent_to_line :
  ∃ m : ℝ, ∀ x y : ℝ, circle_C x y m ∧ line_l x y →
    (x + 1/2)^2 + (y - 3)^2 = 1/8 :=
sorry

-- Theorem for part II
theorem circle_through_origin :
  ∃ m : ℝ, m = -3/2 ∧
    (∀ x1 y1 x2 y2 : ℝ,
      (circle_C x1 y1 m ∧ line_l x1 y1) ∧
      (circle_C x2 y2 m ∧ line_l x2 y2) ∧
      x1 ≠ x2 →
      x1 * x2 + y1 * y2 = 0) :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_line_circle_through_origin_l1254_125450


namespace NUMINAMATH_CALUDE_frequency_problem_l1254_125469

theorem frequency_problem (sample_size : ℕ) (num_groups : ℕ) 
  (common_diff : ℚ) (last_seven_sum : ℚ) : 
  sample_size = 1000 →
  num_groups = 10 →
  common_diff = 0.05 →
  last_seven_sum = 0.79 →
  ∃ (x : ℚ), 
    x > 0 ∧ 
    x + common_diff > 0 ∧ 
    x + 2 * common_diff > 0 ∧
    x + (x + common_diff) + (x + 2 * common_diff) + last_seven_sum = 1 →
    (x * sample_size : ℚ) = 20 :=
by sorry

end NUMINAMATH_CALUDE_frequency_problem_l1254_125469


namespace NUMINAMATH_CALUDE_normal_distribution_probability_l1254_125479

/-- A random variable following a normal distribution with mean 2 and standard deviation 1. -/
def ξ : Real → Real := sorry

/-- The probability density function of the standard normal distribution. -/
noncomputable def φ : Real → Real := sorry

/-- The cumulative distribution function of the standard normal distribution. -/
noncomputable def Φ : Real → Real := sorry

/-- The probability that ξ is greater than 3. -/
def P_gt_3 : Real := 0.023

theorem normal_distribution_probability (h : P_gt_3 = 1 - Φ 1) : 
  Φ 1 - Φ (-1) = 0.954 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_probability_l1254_125479


namespace NUMINAMATH_CALUDE_liar_paradox_l1254_125442

/-- Represents the types of people in the land -/
inductive Person
  | Knight
  | Liar
  | Outsider

/-- Represents the statement "I am a liar" -/
def liarStatement : Prop := True

/-- A function that determines if a person tells the truth -/
def tellsTruth (p : Person) : Prop :=
  match p with
  | Person.Knight => True
  | Person.Liar => False
  | Person.Outsider => True

/-- A function that determines if a person's statement matches their nature -/
def statementMatches (p : Person) : Prop :=
  (p = Person.Liar) = tellsTruth p

theorem liar_paradox :
  ∀ p : Person, (p = Person.Knight ∨ p = Person.Liar) →
    (tellsTruth p = (p = Person.Liar)) → p = Person.Outsider := by
  sorry

end NUMINAMATH_CALUDE_liar_paradox_l1254_125442


namespace NUMINAMATH_CALUDE_pizza_payment_difference_l1254_125483

/-- Represents the pizza sharing scenario between Doug and Dave -/
structure PizzaSharing where
  total_slices : ℕ
  plain_cost : ℚ
  topping_cost : ℚ
  topped_slices : ℕ
  dave_plain_slices : ℕ

/-- Calculates the cost per slice given the total cost and number of slices -/
def cost_per_slice (total_cost : ℚ) (total_slices : ℕ) : ℚ :=
  total_cost / total_slices

/-- Calculates the payment difference between Dave and Doug -/
def payment_difference (ps : PizzaSharing) : ℚ :=
  let total_cost := ps.plain_cost + ps.topping_cost
  let per_slice_cost := cost_per_slice total_cost ps.total_slices
  let dave_slices := ps.topped_slices + ps.dave_plain_slices
  let doug_slices := ps.total_slices - dave_slices
  dave_slices * per_slice_cost - doug_slices * per_slice_cost

/-- Theorem stating that the payment difference is 2.8 under the given conditions -/
theorem pizza_payment_difference :
  let ps : PizzaSharing := {
    total_slices := 10,
    plain_cost := 10,
    topping_cost := 4,
    topped_slices := 4,
    dave_plain_slices := 2
  }
  payment_difference ps = 2.8 := by
  sorry


end NUMINAMATH_CALUDE_pizza_payment_difference_l1254_125483


namespace NUMINAMATH_CALUDE_expression_evaluation_l1254_125472

theorem expression_evaluation :
  let x : ℤ := -2
  (x^2 + 7*x - 8) = -18 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1254_125472


namespace NUMINAMATH_CALUDE_ellipse_circle_centers_distance_l1254_125489

/-- Given an ellipse with center O and semi-axes a and b, and a circle with radius r 
    and center C on the major semi-axis of the ellipse that touches the ellipse at two points, 
    prove that the square of the distance between the centers of the ellipse and the circle 
    is equal to ((a^2 - b^2) * (b^2 - r^2)) / b^2. -/
theorem ellipse_circle_centers_distance 
  (O : ℝ × ℝ) (C : ℝ × ℝ) (a b r : ℝ) : 
  (a > 0) → (b > 0) → (r > 0) → (a ≥ b) →
  (∃ (P Q : ℝ × ℝ), 
    (P.1 - O.1)^2 / a^2 + (P.2 - O.2)^2 / b^2 = 1 ∧
    (Q.1 - O.1)^2 / a^2 + (Q.2 - O.2)^2 / b^2 = 1 ∧
    (P.1 - C.1)^2 + (P.2 - C.2)^2 = r^2 ∧
    (Q.1 - C.1)^2 + (Q.2 - C.2)^2 = r^2 ∧
    C.2 = O.2) →
  (C.1 - O.1)^2 = ((a^2 - b^2) * (b^2 - r^2)) / b^2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_circle_centers_distance_l1254_125489


namespace NUMINAMATH_CALUDE_constant_function_theorem_l1254_125476

theorem constant_function_theorem (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (∀ x, f x * (deriv f x) = 0) → ∃ C, ∀ x, f x = C := by
  sorry

end NUMINAMATH_CALUDE_constant_function_theorem_l1254_125476


namespace NUMINAMATH_CALUDE_simplify_and_ratio_l1254_125427

theorem simplify_and_ratio (m n : ℤ) : 
  let simplified := (5*m + 15*n + 20) / 5
  ∃ (a b c : ℤ), 
    simplified = a*m + b*n + c ∧ 
    (a + b) / c = 1 :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_ratio_l1254_125427


namespace NUMINAMATH_CALUDE_physical_fitness_test_participation_l1254_125432

/-- The number of students who met the standards in the physical fitness test -/
def students_met_standards : ℕ := 900

/-- The percentage of students who took the test but did not meet the standards -/
def percentage_not_met_standards : ℚ := 25 / 100

/-- The percentage of students who did not participate in the test -/
def percentage_not_participated : ℚ := 4 / 100

/-- The total number of students in the sixth grade -/
def total_students : ℕ := 1200

/-- The number of students who did not participate in the physical fitness test -/
def students_not_participated : ℕ := 48

theorem physical_fitness_test_participation :
  (students_not_participated : ℚ) = percentage_not_participated * (total_students : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_physical_fitness_test_participation_l1254_125432


namespace NUMINAMATH_CALUDE_shorter_base_length_l1254_125463

/-- A trapezoid with given properties -/
structure Trapezoid where
  long_base : ℝ
  short_base : ℝ
  midpoint_segment : ℝ

/-- The trapezoid satisfies the given conditions -/
def satisfies_conditions (t : Trapezoid) : Prop :=
  t.long_base = 85 ∧ t.midpoint_segment = 5

/-- Theorem: In a trapezoid satisfying the given conditions, the shorter base is 75 -/
theorem shorter_base_length (t : Trapezoid) (h : satisfies_conditions t) : 
  t.short_base = 75 := by
  sorry

end NUMINAMATH_CALUDE_shorter_base_length_l1254_125463


namespace NUMINAMATH_CALUDE_unique_solution_to_equation_l1254_125494

theorem unique_solution_to_equation (x : ℝ) :
  x + 2 ≠ 0 →
  ((16 * x - x^2) / (x + 2) * (x + (16 - x) / (x + 2)) = 60) ↔ x = 4 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_to_equation_l1254_125494


namespace NUMINAMATH_CALUDE_puppy_weight_l1254_125461

theorem puppy_weight (puppy smaller_cat larger_cat bird : ℝ) 
  (total_weight : puppy + smaller_cat + larger_cat + bird = 34)
  (larger_cat_weight : puppy + larger_cat = 3 * bird)
  (smaller_cat_weight : puppy + smaller_cat = 2 * bird) :
  puppy = 17 := by
  sorry

end NUMINAMATH_CALUDE_puppy_weight_l1254_125461


namespace NUMINAMATH_CALUDE_ice_cream_shop_problem_l1254_125431

/-- Ice cream shop problem -/
theorem ice_cream_shop_problem (total_revenue : ℕ) (cone_price : ℕ) (free_cones : ℕ) (n : ℕ) :
  total_revenue = 100 ∧ 
  cone_price = 2 ∧ 
  free_cones = 10 ∧
  (total_revenue / cone_price + free_cones) % n = 0 →
  n = 6 := by
sorry

end NUMINAMATH_CALUDE_ice_cream_shop_problem_l1254_125431


namespace NUMINAMATH_CALUDE_equation_solution_l1254_125422

theorem equation_solution (x : ℝ) (hx : x ≠ 0) :
  3 - 5 / x + 2 / (x^2) = 0 → (3 / x = 9 / 2 ∨ 3 / x = 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1254_125422


namespace NUMINAMATH_CALUDE_function_properties_l1254_125443

/-- Given functions f and g with parameter a, proves properties about their extrema and monotonicity -/
theorem function_properties (a : ℝ) (h : a ≤ 0) :
  let f := fun x : ℝ ↦ Real.exp x + a * x
  let g := fun x : ℝ ↦ a * x - Real.log x
  -- The minimum of f occurs at ln(-a)
  (∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ x_min = Real.log (-a)) ∧
  -- The minimum value of f is -a + a * ln(-a)
  (∃ (y_min : ℝ), ∀ (x : ℝ), f x ≥ y_min ∧ y_min = -a + a * Real.log (-a)) ∧
  -- f has no maximum value
  (¬∃ (y_max : ℝ), ∀ (x : ℝ), f x ≤ y_max) ∧
  -- f and g have the same monotonicity on some interval iff a ∈ (-∞, -1)
  (∃ (M : Set ℝ), (∀ (x y : ℝ), x ∈ M → y ∈ M → x < y → (f x < f y ↔ g x < g y)) ↔ a < -1) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l1254_125443


namespace NUMINAMATH_CALUDE_combinations_permutations_relation_l1254_125470

/-- The number of combinations of n elements taken k at a time -/
def C (n k : ℕ) : ℕ := sorry

/-- The number of permutations of k elements from an n-element set -/
def A (n k : ℕ) : ℕ := sorry

/-- Theorem: The number of combinations is equal to the number of permutations divided by k factorial -/
theorem combinations_permutations_relation (n k : ℕ) : C n k = A n k / k! := by
  sorry

end NUMINAMATH_CALUDE_combinations_permutations_relation_l1254_125470


namespace NUMINAMATH_CALUDE_least_sum_of_exponents_for_520_l1254_125446

theorem least_sum_of_exponents_for_520 (n : ℕ) (h1 : n = 520) :
  ∃ (a b : ℕ), 
    n = 2^a + 2^b ∧ 
    a ≠ b ∧ 
    (a = 3 ∨ b = 3) ∧ 
    ∀ (c d : ℕ), (n = 2^c + 2^d ∧ c ≠ d ∧ (c = 3 ∨ d = 3)) → a + b ≤ c + d :=
by sorry

end NUMINAMATH_CALUDE_least_sum_of_exponents_for_520_l1254_125446


namespace NUMINAMATH_CALUDE_ratio_of_sum_and_difference_l1254_125486

theorem ratio_of_sum_and_difference (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) (h : a + b = 7 * (a - b)) : a / b = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_sum_and_difference_l1254_125486


namespace NUMINAMATH_CALUDE_reduced_price_is_34_2_l1254_125440

/-- Represents the price reduction percentage -/
def price_reduction : ℚ := 20 / 100

/-- Represents the additional amount of oil obtained after price reduction (in kg) -/
def additional_oil : ℚ := 4

/-- Represents the total cost in Rupees -/
def total_cost : ℚ := 684

/-- Calculates the reduced price per kg of oil -/
def reduced_price_per_kg (price_reduction : ℚ) (additional_oil : ℚ) (total_cost : ℚ) : ℚ :=
  total_cost / (total_cost / (total_cost / ((1 - price_reduction) * total_cost / total_cost)) + additional_oil)

/-- Theorem stating that the reduced price per kg of oil is 34.2 Rupees -/
theorem reduced_price_is_34_2 :
  reduced_price_per_kg price_reduction additional_oil total_cost = 34.2 := by
  sorry

end NUMINAMATH_CALUDE_reduced_price_is_34_2_l1254_125440


namespace NUMINAMATH_CALUDE_sum_of_digits_of_greatest_prime_divisor_l1254_125411

def n : ℕ := 4095

-- Define the greatest prime divisor of n
def greatest_prime_divisor (n : ℕ) : ℕ :=
  (Nat.factors n).foldl max 0

-- Define a function to sum the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.repr.foldl (λ sum c => sum + c.toNat - 48) 0

theorem sum_of_digits_of_greatest_prime_divisor :
  sum_of_digits (greatest_prime_divisor n) = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_greatest_prime_divisor_l1254_125411


namespace NUMINAMATH_CALUDE_det_A_equals_one_l1254_125429

open Matrix

theorem det_A_equals_one (a d : ℝ) :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![a, 1; -2, d]
  (A + A⁻¹ = 0) → det A = 1 := by
  sorry

end NUMINAMATH_CALUDE_det_A_equals_one_l1254_125429


namespace NUMINAMATH_CALUDE_special_function_at_65_l1254_125433

/-- A function satisfying f(xy) = xf(y) for all real x and y, with f(1) = 40 -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x * y) = x * f y) ∧ (f 1 = 40)

/-- Theorem: If f is a special function, then f(65) = 2600 -/
theorem special_function_at_65 (f : ℝ → ℝ) (h : special_function f) : f 65 = 2600 := by
  sorry

end NUMINAMATH_CALUDE_special_function_at_65_l1254_125433


namespace NUMINAMATH_CALUDE_min_sum_with_gcd_conditions_l1254_125475

theorem min_sum_with_gcd_conditions :
  ∃ (a b c : ℕ+),
    (Nat.gcd a b > 1) ∧
    (Nat.gcd b c > 1) ∧
    (Nat.gcd c a > 1) ∧
    (Nat.gcd a (Nat.gcd b c) = 1) ∧
    (a + b + c = 31) ∧
    (∀ (x y z : ℕ+),
      (Nat.gcd x y > 1) →
      (Nat.gcd y z > 1) →
      (Nat.gcd z x > 1) →
      (Nat.gcd x (Nat.gcd y z) = 1) →
      (x + y + z ≥ 31)) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_with_gcd_conditions_l1254_125475


namespace NUMINAMATH_CALUDE_investment_calculation_l1254_125413

/-- Given two investors P and Q, where the profit is divided in the ratio 2:3
    and P invested Rs 40000, prove that Q invested Rs 60000 -/
theorem investment_calculation (P Q : ℕ) (profit_ratio : ℚ) :
  P = 40000 →
  profit_ratio = 2 / 3 →
  Q = 60000 :=
by sorry

end NUMINAMATH_CALUDE_investment_calculation_l1254_125413


namespace NUMINAMATH_CALUDE_fish_tank_ratio_l1254_125459

/-- The number of fish in the first tank -/
def first_tank : ℕ := 7 + 8

/-- The number of fish in the second tank -/
def second_tank : ℕ := 2 * first_tank

/-- The number of fish in the third tank -/
def third_tank : ℕ := 10

theorem fish_tank_ratio : 
  (third_tank : ℚ) / second_tank = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_fish_tank_ratio_l1254_125459


namespace NUMINAMATH_CALUDE_apple_problem_l1254_125410

theorem apple_problem (initial_apples : ℕ) (sold_to_jill : ℚ) (sold_to_june : ℚ) (sold_to_jeff : ℚ) (donated_to_school : ℚ) :
  initial_apples = 150 →
  sold_to_jill = 20 / 100 →
  sold_to_june = 30 / 100 →
  sold_to_jeff = 10 / 100 →
  donated_to_school = 5 / 100 →
  let remaining_after_jill := initial_apples - ⌊initial_apples * sold_to_jill⌋
  let remaining_after_june := remaining_after_jill - ⌊remaining_after_jill * sold_to_june⌋
  let remaining_after_jeff := remaining_after_june - ⌊remaining_after_june * sold_to_jeff⌋
  let final_remaining := remaining_after_jeff - ⌈remaining_after_jeff * donated_to_school⌉
  final_remaining = 72 := by
    sorry

end NUMINAMATH_CALUDE_apple_problem_l1254_125410


namespace NUMINAMATH_CALUDE_pitcher_juice_distribution_l1254_125481

theorem pitcher_juice_distribution :
  ∀ (C : ℝ),
  C > 0 →
  let pineapple_juice := (1/2 : ℝ) * C
  let orange_juice := (1/4 : ℝ) * C
  let total_juice := pineapple_juice + orange_juice
  let cups := 4
  let juice_per_cup := total_juice / cups
  (juice_per_cup / C) * 100 = 18.75 :=
by
  sorry

end NUMINAMATH_CALUDE_pitcher_juice_distribution_l1254_125481


namespace NUMINAMATH_CALUDE_registration_theorem_l1254_125421

/-- The number of possible ways for students to register for events. -/
def registration_combinations (num_students : ℕ) (num_events : ℕ) : ℕ :=
  num_events ^ num_students

/-- Theorem stating that with 4 students and 3 events, there are 81 possible registration combinations. -/
theorem registration_theorem :
  registration_combinations 4 3 = 81 := by
  sorry

end NUMINAMATH_CALUDE_registration_theorem_l1254_125421


namespace NUMINAMATH_CALUDE_xyz_product_absolute_value_l1254_125488

theorem xyz_product_absolute_value (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (hdistinct : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (heq1 : x + 1 / y = y + 1 / z)
  (heq2 : y + 1 / z = z + 1 / x + 1) :
  |x * y * z| = 1 := by
sorry

end NUMINAMATH_CALUDE_xyz_product_absolute_value_l1254_125488


namespace NUMINAMATH_CALUDE_colonization_combinations_l1254_125460

def total_planets : ℕ := 15
def earth_like_planets : ℕ := 6
def mars_like_planets : ℕ := 9
def total_units : ℕ := 16

def valid_combination (a b : ℕ) : Prop :=
  a ≤ earth_like_planets ∧ 
  b ≤ mars_like_planets ∧ 
  2 * a + b = total_units

def combinations_count : ℕ := 
  (Nat.choose earth_like_planets 6 * Nat.choose mars_like_planets 4) +
  (Nat.choose earth_like_planets 5 * Nat.choose mars_like_planets 6) +
  (Nat.choose earth_like_planets 4 * Nat.choose mars_like_planets 8)

theorem colonization_combinations : 
  combinations_count = 765 :=
sorry

end NUMINAMATH_CALUDE_colonization_combinations_l1254_125460


namespace NUMINAMATH_CALUDE_solution_is_83_l1254_125416

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- Define the equation
def equation (x : ℝ) : Prop :=
  log 3 (x^2 - 3) + log 9 (x - 2) + log (1/3) (x^2 - 3) = 2

-- Theorem statement
theorem solution_is_83 :
  ∃ (x : ℝ), x > 0 ∧ equation x ∧ x = 83 :=
by sorry

end NUMINAMATH_CALUDE_solution_is_83_l1254_125416


namespace NUMINAMATH_CALUDE_decimal_multiplication_l1254_125430

theorem decimal_multiplication :
  (10 * 0.1 = 1) ∧ (10 * 0.01 = 0.1) ∧ (10 * 0.001 = 0.01) := by
  sorry

end NUMINAMATH_CALUDE_decimal_multiplication_l1254_125430


namespace NUMINAMATH_CALUDE_average_home_runs_l1254_125404

theorem average_home_runs : 
  let players_5 := 7
  let players_6 := 5
  let players_7 := 4
  let players_9 := 3
  let players_11 := 1
  let total_players := players_5 + players_6 + players_7 + players_9 + players_11
  let total_home_runs := 5 * players_5 + 6 * players_6 + 7 * players_7 + 9 * players_9 + 11 * players_11
  (total_home_runs : ℚ) / total_players = 131 / 20 := by
  sorry

end NUMINAMATH_CALUDE_average_home_runs_l1254_125404


namespace NUMINAMATH_CALUDE_simplify_expression_l1254_125453

theorem simplify_expression (z : ℝ) : z - 2*z + 4*z - 6 + 3 + 7 - 2 = 3*z + 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1254_125453


namespace NUMINAMATH_CALUDE_percentage_increase_decrease_l1254_125447

theorem percentage_increase_decrease (p q M : ℝ) 
  (hp : p > 0) (hq : q > 0) (hq_bound : q < 200) (hM : M > 0) :
  M * (1 + p / 100) * (1 - q / 100) > M ↔ p > 100 * q / (100 - q) := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_decrease_l1254_125447


namespace NUMINAMATH_CALUDE_composite_shape_area_l1254_125497

-- Define the dimensions of the rectangles
def rect1_width : ℕ := 6
def rect1_height : ℕ := 7
def rect2_width : ℕ := 3
def rect2_height : ℕ := 5
def rect3_width : ℕ := 5
def rect3_height : ℕ := 6

-- Define the function to calculate the area of a rectangle
def rectangle_area (width height : ℕ) : ℕ := width * height

-- Theorem statement
theorem composite_shape_area :
  rectangle_area rect1_width rect1_height +
  rectangle_area rect2_width rect2_height +
  rectangle_area rect3_width rect3_height = 87 := by
  sorry


end NUMINAMATH_CALUDE_composite_shape_area_l1254_125497


namespace NUMINAMATH_CALUDE_minimum_additional_weeks_to_win_l1254_125420

def puppy_cost : ℕ := 1000
def weekly_prize : ℕ := 100
def additional_wins_needed : ℕ := 8

theorem minimum_additional_weeks_to_win (current_savings : ℕ) : 
  (current_savings + additional_wins_needed * weekly_prize = puppy_cost) → 
  additional_wins_needed = 8 := by
  sorry

end NUMINAMATH_CALUDE_minimum_additional_weeks_to_win_l1254_125420


namespace NUMINAMATH_CALUDE_monotone_decreasing_implies_a_leq_neg_four_l1254_125495

/-- A quadratic function f(x) = x^2 + 2ax + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 2

/-- The property of f being monotonically decreasing on (-∞, 4] -/
def is_monotone_decreasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, x < y → y ≤ 4 → f a x > f a y

/-- The main theorem: if f is monotonically decreasing on (-∞, 4], then a ≤ -4 -/
theorem monotone_decreasing_implies_a_leq_neg_four (a : ℝ) :
  is_monotone_decreasing_on_interval a → a ≤ -4 :=
by
  sorry

end NUMINAMATH_CALUDE_monotone_decreasing_implies_a_leq_neg_four_l1254_125495


namespace NUMINAMATH_CALUDE_smallest_value_w_cube_plus_z_cube_l1254_125471

theorem smallest_value_w_cube_plus_z_cube (w z : ℂ) 
  (h1 : Complex.abs (w + z) = 2)
  (h2 : Complex.abs (w^2 + z^2) = 18) :
  Complex.abs (w^3 + z^3) = 50 := by
sorry

end NUMINAMATH_CALUDE_smallest_value_w_cube_plus_z_cube_l1254_125471


namespace NUMINAMATH_CALUDE_point_on_line_l1254_125441

/-- Given three points A, B, and C on a straight line, prove that the y-coordinate of C is 7. -/
theorem point_on_line (A B C : ℝ × ℝ) : 
  A = (1, -1) → B = (3, 3) → C.1 = 5 → 
  (C.2 - A.2) / (C.1 - A.1) = (B.2 - A.2) / (B.1 - A.1) → 
  C.2 = 7 := by sorry

end NUMINAMATH_CALUDE_point_on_line_l1254_125441


namespace NUMINAMATH_CALUDE_sphere_volume_radius_3_l1254_125484

/-- The volume of a sphere with radius 3 is 36π. -/
theorem sphere_volume_radius_3 :
  let r : ℝ := 3
  let volume := (4 / 3) * Real.pi * r ^ 3
  volume = 36 * Real.pi := by sorry

end NUMINAMATH_CALUDE_sphere_volume_radius_3_l1254_125484


namespace NUMINAMATH_CALUDE_rectangular_prism_sum_of_dimensions_l1254_125444

theorem rectangular_prism_sum_of_dimensions 
  (α β γ : ℝ) 
  (h1 : α * β = 18) 
  (h2 : α * γ = 36) 
  (h3 : β * γ = 72) : 
  α + β + γ = 21 := by
sorry

end NUMINAMATH_CALUDE_rectangular_prism_sum_of_dimensions_l1254_125444


namespace NUMINAMATH_CALUDE_total_crayons_l1254_125406

def packs : ℕ := 4
def crayons_per_pack : ℕ := 10
def extra_crayons : ℕ := 6

theorem total_crayons : packs * crayons_per_pack + extra_crayons = 46 := by
  sorry

end NUMINAMATH_CALUDE_total_crayons_l1254_125406


namespace NUMINAMATH_CALUDE_fraction_equals_d_minus_one_l1254_125458

theorem fraction_equals_d_minus_one (n d : ℕ) (h : d ∣ n) :
  ∃ i : ℕ, i < n ∧ (i : ℚ) / (n - i : ℚ) = d - 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_d_minus_one_l1254_125458


namespace NUMINAMATH_CALUDE_unique_residue_mod_16_l1254_125439

theorem unique_residue_mod_16 : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 15 ∧ n ≡ -3125 [ZMOD 16] := by
  sorry

end NUMINAMATH_CALUDE_unique_residue_mod_16_l1254_125439


namespace NUMINAMATH_CALUDE_f_minimum_value_l1254_125423

noncomputable def f (x : ℝ) : ℝ := x + 1/x + 1/(x + 1/x) + 1/x^2

theorem f_minimum_value (x : ℝ) (hx : x > 0) : f x ≥ 3.5 ∧ f 1 = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_f_minimum_value_l1254_125423


namespace NUMINAMATH_CALUDE_function_inequality_l1254_125498

open Set

theorem function_inequality (f g : ℝ → ℝ) (a b x : ℝ) :
  DifferentiableOn ℝ f (Icc a b) →
  DifferentiableOn ℝ g (Icc a b) →
  (∀ y ∈ Icc a b, deriv f y > deriv g y) →
  a < x →
  x < b →
  f x + g a > g x + f a :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_l1254_125498


namespace NUMINAMATH_CALUDE_expression1_equals_4_expression2_equals_neg6_l1254_125467

-- Define the expressions
def expression1 : ℚ := (-36) * (1/3 - 1/2) + 16 / ((-2)^3)
def expression2 : ℚ := (-5 + 2) * (1/3) + 5^2 / (-5)

-- Theorem statements
theorem expression1_equals_4 : expression1 = 4 := by sorry

theorem expression2_equals_neg6 : expression2 = -6 := by sorry

end NUMINAMATH_CALUDE_expression1_equals_4_expression2_equals_neg6_l1254_125467


namespace NUMINAMATH_CALUDE_red_peaches_per_basket_l1254_125499

theorem red_peaches_per_basket 
  (num_baskets : ℕ) 
  (green_per_basket : ℕ) 
  (total_peaches : ℕ) 
  (h1 : num_baskets = 15)
  (h2 : green_per_basket = 4)
  (h3 : total_peaches = 345) :
  (total_peaches - num_baskets * green_per_basket) / num_baskets = 19 := by
  sorry

end NUMINAMATH_CALUDE_red_peaches_per_basket_l1254_125499


namespace NUMINAMATH_CALUDE_game_win_fraction_l1254_125438

theorem game_win_fraction (total_matches : ℕ) (points_per_win : ℕ) (player1_points : ℕ) :
  total_matches = 8 →
  points_per_win = 10 →
  player1_points = 20 →
  (total_matches - player1_points / points_per_win) / total_matches = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_game_win_fraction_l1254_125438


namespace NUMINAMATH_CALUDE_fraction_sum_theorem_l1254_125402

theorem fraction_sum_theorem (a b c d : ℝ) 
  (h1 : a/b + b/c + c/d + d/a = 6) 
  (h2 : a/c + b/d + c/a + d/b = 8) : 
  a/b + c/d = 2 ∨ a/b + c/d = 4 := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_theorem_l1254_125402


namespace NUMINAMATH_CALUDE_beyonce_album_songs_l1254_125418

/-- The number of songs in Beyonce's first two albums -/
def songs_in_first_two_albums (total_songs num_singles num_albums songs_in_third_album : ℕ) : ℕ :=
  total_songs - num_singles - songs_in_third_album

theorem beyonce_album_songs :
  songs_in_first_two_albums 55 5 3 20 = 30 := by
  sorry

end NUMINAMATH_CALUDE_beyonce_album_songs_l1254_125418


namespace NUMINAMATH_CALUDE_principal_amount_l1254_125424

/-- Calculates the total interest paid over 11 years given the principal amount -/
def totalInterest (principal : ℝ) : ℝ :=
  principal * 0.06 * 3 + principal * 0.09 * 5 + principal * 0.13 * 3

/-- Theorem stating that the principal amount borrowed is 8000 given the total interest paid -/
theorem principal_amount (totalInterestPaid : ℝ) 
  (h : totalInterestPaid = 8160) : 
  ∃ (principal : ℝ), totalInterest principal = totalInterestPaid ∧ principal = 8000 := by
  sorry

#check principal_amount

end NUMINAMATH_CALUDE_principal_amount_l1254_125424


namespace NUMINAMATH_CALUDE_original_price_calculation_l1254_125403

/-- 
Theorem: If an item's price is increased by 15%, then decreased by 20%, 
resulting in a final price of 46 yuan, the original price was 50 yuan.
-/
theorem original_price_calculation (original_price : ℝ) : 
  (original_price * 1.15 * 0.8 = 46) → original_price = 50 := by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l1254_125403


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_product_l1254_125434

theorem absolute_value_equation_solution_product : 
  ∃ (x₁ x₂ : ℝ), 
    (|18 / x₁ + 4| = 3) ∧ 
    (|18 / x₂ + 4| = 3) ∧ 
    (x₁ ≠ x₂) ∧ 
    (x₁ * x₂ = 324 / 7) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_product_l1254_125434


namespace NUMINAMATH_CALUDE_min_students_four_correct_is_eight_l1254_125448

-- Define the total number of students
def total_students : ℕ := 100

-- Define the number of students who performed each spell correctly
def spell1_correct : ℕ := 95
def spell2_correct : ℕ := 75
def spell3_correct : ℕ := 97
def spell4_correct : ℕ := 95
def spell5_correct : ℕ := 96

-- Define the function to calculate the minimum number of students who performed exactly 4 out of 5 spells correctly
def min_students_four_correct : ℕ :=
  total_students - spell2_correct - (total_students - spell1_correct) - (total_students - spell3_correct) - (total_students - spell4_correct) - (total_students - spell5_correct)

-- Theorem statement
theorem min_students_four_correct_is_eight :
  min_students_four_correct = 8 :=
sorry

end NUMINAMATH_CALUDE_min_students_four_correct_is_eight_l1254_125448


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1254_125415

-- Define the propositions
def p (a : ℝ) : Prop := a > 2
def q (a : ℝ) : Prop := ¬(∀ x : ℝ, x^2 + a*x + 1 ≥ 0)

-- Theorem statement
theorem p_sufficient_not_necessary_for_q :
  (∃ a : ℝ, p a → q a) ∧ (∃ a : ℝ, q a ∧ ¬(p a)) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1254_125415


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_condition_l1254_125464

theorem quadratic_inequality_solution_condition (c : ℝ) :
  (c > 0) →
  (∃ x : ℝ, x^2 - 8*x + c < 0) ↔ (c > 0 ∧ c < 16) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_condition_l1254_125464


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1254_125482

/-- A quadratic equation x^2 + x + c = 0 has two real roots of opposite signs -/
def has_two_real_roots_opposite_signs (c : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ x * y < 0 ∧ x^2 + x + c = 0 ∧ y^2 + y + c = 0

theorem necessary_not_sufficient_condition :
  (∀ c : ℝ, has_two_real_roots_opposite_signs c → c < 0) ∧
  (∃ c : ℝ, c < 0 ∧ ¬has_two_real_roots_opposite_signs c) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1254_125482


namespace NUMINAMATH_CALUDE_sqrt_three_between_l1254_125436

theorem sqrt_three_between (n : ℕ+) : 
  (1 + 3 / (n + 1 : ℝ) < Real.sqrt 3 ∧ Real.sqrt 3 < 1 + 3 / (n : ℝ)) → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_between_l1254_125436


namespace NUMINAMATH_CALUDE_cubic_inequality_solution_l1254_125478

theorem cubic_inequality_solution (x : ℝ) :
  x^3 + x^2 - 4*x - 4 < 0 ↔ x < -2 ∨ (-1 < x ∧ x < 2) :=
sorry

end NUMINAMATH_CALUDE_cubic_inequality_solution_l1254_125478


namespace NUMINAMATH_CALUDE_projection_vector_l1254_125409

def a : Fin 3 → ℝ := ![0, 1, 1]
def b : Fin 3 → ℝ := ![1, 1, 0]

theorem projection_vector :
  let proj := (a • b) / (a • a) • a
  proj = ![0, 1/2, 1/2] := by sorry

end NUMINAMATH_CALUDE_projection_vector_l1254_125409


namespace NUMINAMATH_CALUDE_cos_180_degrees_l1254_125468

theorem cos_180_degrees : Real.cos (π) = -1 := by
  sorry

end NUMINAMATH_CALUDE_cos_180_degrees_l1254_125468


namespace NUMINAMATH_CALUDE_bijective_function_exists_l1254_125449

/-- A function that maps elements of ℤm × ℤn to itself -/
def bijective_function (m n : ℕ+) : (Fin m × Fin n) → (Fin m × Fin n) := sorry

/-- Predicate to check if all f(v) + v are pairwise distinct -/
def all_distinct (m n : ℕ+) (f : (Fin m × Fin n) → (Fin m × Fin n)) : Prop := sorry

/-- Main theorem statement -/
theorem bijective_function_exists (m n : ℕ+) :
  (∃ f : (Fin m × Fin n) → (Fin m × Fin n), Function.Bijective f ∧ all_distinct m n f) ↔
  (m.val % 2 = n.val % 2) := by sorry

end NUMINAMATH_CALUDE_bijective_function_exists_l1254_125449


namespace NUMINAMATH_CALUDE_function_properties_l1254_125412

open Real

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * sin (2 * x) - cos x ^ 2 - 1/2

theorem function_properties (α : ℝ) (h1 : tan α = 2 * Real.sqrt 3) (h2 : f (Real.sqrt 3 / 2) α = -3/26) :
  ∃ (m : ℝ),
    m = Real.sqrt 3 / 2 ∧
    (∀ x, f m (x + π) = f m x) ∧
    (∀ x ∈ Set.Icc 0 π, 
      (x ∈ Set.Icc 0 (π/3) ∨ x ∈ Set.Icc (5*π/6) π) → 
      ∀ y ∈ Set.Icc 0 π, x < y → f m x < f m y) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1254_125412


namespace NUMINAMATH_CALUDE_min_value_a_plus_2b_l1254_125405

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + b = a * b) :
  a + 2 * b ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_a_plus_2b_l1254_125405


namespace NUMINAMATH_CALUDE_number_count_in_average_calculation_l1254_125401

theorem number_count_in_average_calculation 
  (initial_average : ℚ)
  (correct_average : ℚ)
  (incorrect_number : ℚ)
  (correct_number : ℚ)
  (h1 : initial_average = 46)
  (h2 : correct_average = 50)
  (h3 : incorrect_number = 25)
  (h4 : correct_number = 65) :
  ∃ (n : ℕ), n > 0 ∧ 
    (n : ℚ) * correct_average = (n : ℚ) * initial_average + (correct_number - incorrect_number) ∧
    n = 10 := by
  sorry

end NUMINAMATH_CALUDE_number_count_in_average_calculation_l1254_125401


namespace NUMINAMATH_CALUDE_cricket_average_score_l1254_125473

theorem cricket_average_score (matches1 matches2 : ℕ) (avg1 avg2 : ℚ) :
  matches1 = 10 →
  matches2 = 15 →
  avg1 = 60 →
  avg2 = 70 →
  (matches1 * avg1 + matches2 * avg2) / (matches1 + matches2) = 66 := by
  sorry

end NUMINAMATH_CALUDE_cricket_average_score_l1254_125473


namespace NUMINAMATH_CALUDE_correct_transformation_l1254_125428

theorem correct_transformation (x : ℝ) : 3 * x + 5 = 4 * x → 3 * x - 4 * x = -5 := by
  sorry

end NUMINAMATH_CALUDE_correct_transformation_l1254_125428


namespace NUMINAMATH_CALUDE_arithmetic_sequence_count_l1254_125451

theorem arithmetic_sequence_count :
  ∀ (a₁ last d : ℕ) (n : ℕ),
    a₁ = 1 →
    last = 2025 →
    d = 4 →
    last = a₁ + d * (n - 1) →
    n = 507 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_count_l1254_125451


namespace NUMINAMATH_CALUDE_paint_theorem_l1254_125485

def paint_problem (initial_paint : ℚ) (first_day_fraction : ℚ) (second_day_fraction : ℚ) : Prop :=
  let remaining_after_first_day := initial_paint - (first_day_fraction * initial_paint)
  let used_second_day := second_day_fraction * remaining_after_first_day
  let remaining_after_second_day := remaining_after_first_day - used_second_day
  remaining_after_second_day = (4 : ℚ) / 9 * initial_paint

theorem paint_theorem : 
  paint_problem 1 (1/3) (1/3) := by sorry

end NUMINAMATH_CALUDE_paint_theorem_l1254_125485


namespace NUMINAMATH_CALUDE_problem_solution_l1254_125457

theorem problem_solution : (12 : ℝ) ^ 1 * 6 ^ 4 / 432 = 36 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1254_125457


namespace NUMINAMATH_CALUDE_jenny_travel_distance_l1254_125466

/-- The distance from Jenny's home to her friend's place in miles -/
def total_distance : ℝ := 155

/-- Jenny's initial speed in miles per hour -/
def initial_speed : ℝ := 45

/-- Jenny's increased speed in miles per hour -/
def increased_speed : ℝ := 65

/-- The time Jenny stops at the store in hours -/
def stop_time : ℝ := 0.25

/-- The total travel time in hours -/
def total_time : ℝ := 3.4375

theorem jenny_travel_distance :
  (initial_speed * (total_time + 1) = total_distance) ∧
  (total_distance - initial_speed = increased_speed * (total_time - stop_time - 1)) ∧
  (total_distance = initial_speed * (total_time + 1)) :=
sorry

end NUMINAMATH_CALUDE_jenny_travel_distance_l1254_125466


namespace NUMINAMATH_CALUDE_monotonic_range_k_negative_range_k_l1254_125477

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 1

-- Define the function g
def g (a b k : ℝ) (x : ℝ) : ℝ := f a b x - k * x

-- Theorem for part (1)
theorem monotonic_range_k (a b : ℝ) (h1 : a > 0) (h2 : f a b (-1) = 0) 
  (h3 : ∀ x : ℝ, f a b x ≥ 0) :
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, Monotone (g a b)) ↔ (k ≤ -2 ∨ k ≥ 6) :=
sorry

-- Theorem for part (2)
theorem negative_range_k (a b : ℝ) (h1 : a > 0) (h2 : f a b (-1) = 0) 
  (h3 : ∀ x : ℝ, f a b x ≥ 0) :
  (∀ x ∈ Set.Icc 1 2, g a b k x < 0) ↔ k > 9/2 :=
sorry

end NUMINAMATH_CALUDE_monotonic_range_k_negative_range_k_l1254_125477


namespace NUMINAMATH_CALUDE_marble_223_is_white_l1254_125400

def marble_color (n : ℕ) : String :=
  let cycle := n % 15
  if cycle < 6 then "gray"
  else if cycle < 11 then "white"
  else "black"

theorem marble_223_is_white :
  marble_color 223 = "white" := by
  sorry

end NUMINAMATH_CALUDE_marble_223_is_white_l1254_125400


namespace NUMINAMATH_CALUDE_exists_distribution_prob_white_gt_two_thirds_l1254_125408

/-- Represents a distribution of balls in two boxes -/
structure BallDistribution :=
  (white_box1 : ℕ)
  (black_box1 : ℕ)
  (white_box2 : ℕ)
  (black_box2 : ℕ)

/-- The total number of white balls -/
def total_white : ℕ := 8

/-- The total number of black balls -/
def total_black : ℕ := 8

/-- Calculates the probability of drawing a white ball given a distribution -/
def prob_white (d : BallDistribution) : ℚ :=
  let p_box1 := (d.white_box1 : ℚ) / (d.white_box1 + d.black_box1 : ℚ)
  let p_box2 := (d.white_box2 : ℚ) / (d.white_box2 + d.black_box2 : ℚ)
  (1/2 : ℚ) * p_box1 + (1/2 : ℚ) * p_box2

/-- Theorem stating that there exists a distribution where the probability of drawing a white ball is greater than 2/3 -/
theorem exists_distribution_prob_white_gt_two_thirds :
  ∃ (d : BallDistribution),
    d.white_box1 + d.white_box2 = total_white ∧
    d.black_box1 + d.black_box2 = total_black ∧
    prob_white d > 2/3 := by
  sorry

end NUMINAMATH_CALUDE_exists_distribution_prob_white_gt_two_thirds_l1254_125408


namespace NUMINAMATH_CALUDE_airplane_speed_l1254_125435

/-- Given a distance and flight times with and against wind, calculate the average speed without wind -/
theorem airplane_speed (distance : ℝ) (time_with_wind : ℝ) (time_against_wind : ℝ)
  (h1 : distance = 9360)
  (h2 : time_with_wind = 12)
  (h3 : time_against_wind = 13) :
  ∃ (speed_no_wind : ℝ) (wind_speed : ℝ),
    speed_no_wind = 750 ∧
    time_with_wind * (speed_no_wind + wind_speed) = distance ∧
    time_against_wind * (speed_no_wind - wind_speed) = distance :=
by sorry

end NUMINAMATH_CALUDE_airplane_speed_l1254_125435


namespace NUMINAMATH_CALUDE_max_third_side_length_l1254_125492

theorem max_third_side_length (a b : ℝ) (ha : a = 5) (hb : b = 10) :
  ∃ (x : ℕ), x ≤ 14 ∧
  ∀ (y : ℕ), (y : ℝ) < a + b ∧ (y : ℝ) > |a - b| → y ≤ x :=
by sorry

end NUMINAMATH_CALUDE_max_third_side_length_l1254_125492


namespace NUMINAMATH_CALUDE_weight_selection_theorem_l1254_125493

theorem weight_selection_theorem (N : ℕ) :
  (∃ (S : Finset ℕ) (k : ℕ), 
    1 < k ∧ 
    k ≤ N ∧
    (∀ i ∈ S, 1 ≤ i ∧ i ≤ N) ∧
    S.card = k ∧
    (S.sum id) * (N - k + 1) = (N * (N + 1)) / 2) ↔ 
  (∃ m : ℕ, N + 1 = m^2) :=
by sorry

end NUMINAMATH_CALUDE_weight_selection_theorem_l1254_125493


namespace NUMINAMATH_CALUDE_rafael_hourly_rate_l1254_125454

theorem rafael_hourly_rate (monday_hours : ℕ) (tuesday_hours : ℕ) (remaining_hours : ℕ) (total_earnings : ℕ) :
  monday_hours = 10 →
  tuesday_hours = 8 →
  remaining_hours = 20 →
  total_earnings = 760 →
  (total_earnings : ℚ) / (monday_hours + tuesday_hours + remaining_hours : ℚ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_rafael_hourly_rate_l1254_125454


namespace NUMINAMATH_CALUDE_complex_square_simplification_l1254_125496

theorem complex_square_simplification : 
  let i : ℂ := Complex.I
  (4 - 3*i)^2 = 7 - 24*i :=
by sorry

end NUMINAMATH_CALUDE_complex_square_simplification_l1254_125496
