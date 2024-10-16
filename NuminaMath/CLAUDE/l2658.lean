import Mathlib

namespace NUMINAMATH_CALUDE_triangle_similarity_condition_l2658_265832

theorem triangle_similarity_condition 
  (a b c a₁ b₁ c₁ : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (ha₁ : a₁ > 0) (hb₁ : b₁ > 0) (hc₁ : c₁ > 0) :
  (∃ (k : ℝ), a₁ = k * a ∧ b₁ = k * b ∧ c₁ = k * c) ↔ 
  (Real.sqrt (a * a₁) + Real.sqrt (b * b₁) + Real.sqrt (c * c₁) = 
   Real.sqrt ((a + b + c) * (a₁ + b₁ + c₁))) := by
sorry

end NUMINAMATH_CALUDE_triangle_similarity_condition_l2658_265832


namespace NUMINAMATH_CALUDE_zero_rational_others_irrational_l2658_265869

-- Define rational numbers
def IsRational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

-- State the theorem
theorem zero_rational_others_irrational :
  IsRational 0 ∧ ¬IsRational (-Real.pi) ∧ ¬IsRational (Real.sqrt 3) ∧ ¬IsRational (Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_zero_rational_others_irrational_l2658_265869


namespace NUMINAMATH_CALUDE_triangle_side_length_l2658_265879

/-- Represents a triangle with given side lengths -/
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ

/-- Calculates the perimeter of a triangle -/
def Triangle.perimeter (t : Triangle) : ℝ :=
  t.side1 + t.side2 + t.side3

theorem triangle_side_length 
  (t : Triangle) 
  (h_perimeter : t.perimeter = 160) 
  (h_side1 : t.side1 = 40) 
  (h_side3 : t.side3 = 70) : 
  t.side2 = 50 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2658_265879


namespace NUMINAMATH_CALUDE_x1_x2_ratio_lt_ae_l2658_265811

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x / a - Real.exp x

theorem x1_x2_ratio_lt_ae (a : ℝ) (x₁ x₂ : ℝ) 
  (ha : a > 0) 
  (hx : x₁ < x₂) 
  (hf₁ : f a x₁ = 0) 
  (hf₂ : f a x₂ = 0) : 
  x₁ / x₂ < a * Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_x1_x2_ratio_lt_ae_l2658_265811


namespace NUMINAMATH_CALUDE_cubic_function_value_l2658_265894

/-- Given a cubic function f(x) = ax^3 + bx - 4 where f(-2) = 2, prove that f(2) = -10 -/
theorem cubic_function_value (a b : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^3 + b * x - 4)
  (h2 : f (-2) = 2) : 
  f 2 = -10 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_value_l2658_265894


namespace NUMINAMATH_CALUDE_train_passing_platform_time_l2658_265814

/-- Calculates the time taken for a train to pass a platform -/
theorem train_passing_platform_time 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (platform_length : ℝ) 
  (h1 : train_length = 720)
  (h2 : train_speed_kmh = 72)
  (h3 : platform_length = 280) : 
  (train_length + platform_length) / (train_speed_kmh * 1000 / 3600) = 50 :=
sorry

end NUMINAMATH_CALUDE_train_passing_platform_time_l2658_265814


namespace NUMINAMATH_CALUDE_prob_at_least_three_matching_l2658_265835

/-- The number of sides on each die -/
def numSides : ℕ := 10

/-- The number of dice rolled -/
def numDice : ℕ := 5

/-- The probability of getting at least three matching dice out of five fair ten-sided dice -/
def probAtLeastThreeMatching : ℚ := 173 / 20000

/-- Theorem stating that the probability of at least three out of five fair ten-sided dice 
    showing the same value is 173/20000 -/
theorem prob_at_least_three_matching : 
  probAtLeastThreeMatching = 173 / 20000 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_three_matching_l2658_265835


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_is_empty_l2658_265831

-- Define the sets A and B
def A : Set ℝ := {x | Real.sqrt (x - 2) ≤ 0}
def B : Set ℝ := {x | 10^2 * 2 = 10^2}

-- State the theorem
theorem intersection_A_complement_B_is_empty :
  A ∩ Bᶜ = ∅ := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_is_empty_l2658_265831


namespace NUMINAMATH_CALUDE_expression_value_l2658_265808

theorem expression_value (x y : ℝ) (h : x - 2*y = -4) :
  (2*y - x)^2 - 2*x + 4*y - 1 = 23 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2658_265808


namespace NUMINAMATH_CALUDE_compute_expression_l2658_265812

theorem compute_expression : 8 * (2/3)^4 + 2 = 290/81 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l2658_265812


namespace NUMINAMATH_CALUDE_multiplication_addition_equality_l2658_265883

theorem multiplication_addition_equality : 15 * 30 + 45 * 15 + 15 * 15 = 1350 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_addition_equality_l2658_265883


namespace NUMINAMATH_CALUDE_yard_width_calculation_l2658_265885

/-- The width of a rectangular yard with a row of trees --/
def yard_width (num_trees : ℕ) (edge_distance : ℝ) (center_distance : ℝ) (end_space : ℝ) : ℝ :=
  let tree_diameter := center_distance - edge_distance
  let total_center_distance := (num_trees - 1) * center_distance
  let total_tree_width := tree_diameter * num_trees
  let total_end_space := 2 * end_space
  total_center_distance + total_tree_width + total_end_space

/-- Theorem stating the width of the yard given the specific conditions --/
theorem yard_width_calculation :
  yard_width 6 12 15 2 = 82 := by
  sorry

end NUMINAMATH_CALUDE_yard_width_calculation_l2658_265885


namespace NUMINAMATH_CALUDE_subtraction_value_l2658_265864

theorem subtraction_value (N : ℝ) (h1 : (N - 24) / 10 = 3) : 
  ∃ x : ℝ, (N - x) / 7 = 7 ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_value_l2658_265864


namespace NUMINAMATH_CALUDE_circle_line_bisection_implies_mn_range_l2658_265868

/-- The circle equation: x^2 + y^2 - 4x - 2y - 4 = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 2*y - 4 = 0

/-- The line equation: mx + 2ny - 4 = 0 -/
def line_equation (m n x y : ℝ) : Prop :=
  m*x + 2*n*y - 4 = 0

/-- The line bisects the perimeter of the circle -/
def line_bisects_circle (m n : ℝ) : Prop :=
  ∀ x y, circle_equation x y → line_equation m n x y

/-- The range of mn is (-∞, 1] -/
def mn_range (m n : ℝ) : Prop :=
  m * n ≤ 1

theorem circle_line_bisection_implies_mn_range :
  ∀ m n, line_bisects_circle m n → mn_range m n :=
sorry

end NUMINAMATH_CALUDE_circle_line_bisection_implies_mn_range_l2658_265868


namespace NUMINAMATH_CALUDE_initial_depth_is_40_l2658_265880

/-- Represents the work done by a group of workers digging to a certain depth -/
structure DiggingWork where
  workers : ℕ  -- number of workers
  hours   : ℕ  -- hours worked per day
  depth   : ℝ  -- depth dug in meters

/-- The theorem stating that given the initial and final conditions, the initial depth is 40 meters -/
theorem initial_depth_is_40 (initial final : DiggingWork) 
  (h1 : initial.workers = 45)
  (h2 : initial.hours = 8)
  (h3 : final.workers = initial.workers + 30)
  (h4 : final.hours = 6)
  (h5 : final.depth = 50)
  (h6 : initial.workers * initial.hours * initial.depth = final.workers * final.hours * final.depth) :
  initial.depth = 40 := by
  sorry

#check initial_depth_is_40

end NUMINAMATH_CALUDE_initial_depth_is_40_l2658_265880


namespace NUMINAMATH_CALUDE_phillips_cucumbers_l2658_265809

/-- Proves that Phillip has 8 cucumbers given the pickle-making conditions --/
theorem phillips_cucumbers :
  ∀ (jars : ℕ) (initial_vinegar : ℕ) (pickles_per_cucumber : ℕ) (pickles_per_jar : ℕ)
    (vinegar_per_jar : ℕ) (remaining_vinegar : ℕ),
  jars = 4 →
  initial_vinegar = 100 →
  pickles_per_cucumber = 6 →
  pickles_per_jar = 12 →
  vinegar_per_jar = 10 →
  remaining_vinegar = 60 →
  ∃ (cucumbers : ℕ),
    cucumbers = 8 ∧
    cucumbers * pickles_per_cucumber = jars * pickles_per_jar ∧
    initial_vinegar - remaining_vinegar = jars * vinegar_per_jar :=
by
  sorry


end NUMINAMATH_CALUDE_phillips_cucumbers_l2658_265809


namespace NUMINAMATH_CALUDE_largest_common_divisor_of_sticker_albums_l2658_265856

theorem largest_common_divisor_of_sticker_albums : ∃ (n : ℕ), n > 0 ∧ 
  n ∣ 1050 ∧ n ∣ 1260 ∧ n ∣ 945 ∧ 
  ∀ (m : ℕ), m > 0 → m ∣ 1050 → m ∣ 1260 → m ∣ 945 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_common_divisor_of_sticker_albums_l2658_265856


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l2658_265893

theorem sum_of_a_and_b (a b : ℝ) 
  (h1 : Real.sqrt 44 = 2 * Real.sqrt a) 
  (h2 : Real.sqrt 54 = 3 * Real.sqrt b) : 
  a + b = 17 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l2658_265893


namespace NUMINAMATH_CALUDE_car_stopping_distance_l2658_265839

/-- The distance function representing the car's motion during emergency braking -/
def s (t : ℝ) : ℝ := 30 * t - 5 * t^2

/-- The maximum distance traveled by the car before stopping -/
def max_distance : ℝ := 45

/-- Theorem stating that the maximum value of s(t) is 45 -/
theorem car_stopping_distance :
  ∃ t₀ : ℝ, ∀ t : ℝ, s t ≤ s t₀ ∧ s t₀ = max_distance :=
sorry

end NUMINAMATH_CALUDE_car_stopping_distance_l2658_265839


namespace NUMINAMATH_CALUDE_pure_imaginary_product_l2658_265891

theorem pure_imaginary_product (a : ℝ) : 
  (∃ b : ℝ, (1 - Complex.I) * (a + Complex.I) = Complex.I * b) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_product_l2658_265891


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2658_265804

theorem complex_fraction_equality : (2 : ℂ) / (1 - Complex.I) = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2658_265804


namespace NUMINAMATH_CALUDE_units_digit_of_m_cubed_plus_two_to_m_l2658_265807

theorem units_digit_of_m_cubed_plus_two_to_m (m : ℕ) : 
  m = 2021^2 + 2^2021 → (m^3 + 2^m) % 10 = 5 := by
sorry

end NUMINAMATH_CALUDE_units_digit_of_m_cubed_plus_two_to_m_l2658_265807


namespace NUMINAMATH_CALUDE_a_representation_theorem_l2658_265863

theorem a_representation_theorem (m n : ℕ) (hm : m ≥ 2) (hn : n ≥ 2) :
  ∃ k : ℕ, ((n + Real.sqrt (n^2 - 4)) / 2) ^ m = (k + Real.sqrt (k^2 - 4)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_a_representation_theorem_l2658_265863


namespace NUMINAMATH_CALUDE_decimal_between_four_and_five_l2658_265802

theorem decimal_between_four_and_five : ∃ x : ℝ, (x = 4.5) ∧ (4 < x) ∧ (x < 5) := by
  sorry

end NUMINAMATH_CALUDE_decimal_between_four_and_five_l2658_265802


namespace NUMINAMATH_CALUDE_basketball_team_allocation_schemes_l2658_265866

theorem basketball_team_allocation_schemes (n : ℕ) (k : ℕ) (m : ℕ) 
  (h1 : n = 8)  -- number of classes
  (h2 : k = 10) -- total number of players
  (h3 : m = k - n) -- remaining spots after each class contributes one player
  : (n.choose 2) + (n.choose 1) = 36 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_allocation_schemes_l2658_265866


namespace NUMINAMATH_CALUDE_sum_abs_bound_l2658_265852

theorem sum_abs_bound (x y z : ℝ) 
  (eq1 : x^2 + y^2 + z = 15)
  (eq2 : x + y + z^2 = 27)
  (eq3 : x*y + y*z + z*x = 7) :
  7 ≤ |x + y + z| ∧ |x + y + z| ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_abs_bound_l2658_265852


namespace NUMINAMATH_CALUDE_simplified_expression_l2658_265886

theorem simplified_expression (a : ℝ) (h : a ≠ 1/2) :
  1 - (2 / (1 + (2*a / (1 - 2*a)))) = 4*a - 1 := by
  sorry

end NUMINAMATH_CALUDE_simplified_expression_l2658_265886


namespace NUMINAMATH_CALUDE_quadratic_roots_to_coefficients_l2658_265862

theorem quadratic_roots_to_coefficients :
  ∀ (b c : ℝ),
  (∀ x : ℝ, x^2 - b*x + c = 0 ↔ x = 1 ∨ x = -2) →
  b = -1 ∧ c = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_to_coefficients_l2658_265862


namespace NUMINAMATH_CALUDE_sin_cos_sum_implies_tan_value_l2658_265859

theorem sin_cos_sum_implies_tan_value (x : ℝ) (h1 : x ∈ Set.Ioo 0 π) 
  (h2 : Real.sin x + Real.cos x = 3 * Real.sqrt 2 / 5) : 
  (1 - Real.cos (2 * x)) / Real.sin (2 * x) = -7 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_implies_tan_value_l2658_265859


namespace NUMINAMATH_CALUDE_women_count_l2658_265899

/-- Represents a company with workers and their retirement plan status -/
structure Company where
  total_workers : ℕ
  workers_without_plan : ℕ
  women_without_plan_ratio : ℚ
  women_with_plan_ratio : ℚ
  total_men : ℕ

/-- Calculates the number of women in the company -/
def number_of_women (c : Company) : ℚ :=
  c.women_without_plan_ratio * c.workers_without_plan +
  c.women_with_plan_ratio * (c.total_workers - c.workers_without_plan)

/-- Theorem stating the number of women in the company -/
theorem women_count (c : Company) 
  (h1 : c.total_workers = 200)
  (h2 : c.workers_without_plan = c.total_workers / 3)
  (h3 : c.women_without_plan_ratio = 2/5)
  (h4 : c.women_with_plan_ratio = 3/5)
  (h5 : c.total_men = 120) :
  ∃ (n : ℕ), n ≤ number_of_women c ∧ number_of_women c < n + 1 ∧ n = 107 := by
  sorry

end NUMINAMATH_CALUDE_women_count_l2658_265899


namespace NUMINAMATH_CALUDE_polygon_sum_l2658_265850

/-- Given a polygon JKLMNO with specific properties, prove that MN + NO = 14.5 -/
theorem polygon_sum (area_JKLMNO : ℝ) (JK KL NO : ℝ) :
  area_JKLMNO = 68 ∧ JK = 10 ∧ KL = 11 ∧ NO = 7 →
  ∃ (MN : ℝ), MN + NO = 14.5 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sum_l2658_265850


namespace NUMINAMATH_CALUDE_root_equation_problem_l2658_265836

theorem root_equation_problem (c d : ℝ) : 
  (∃! (r1 r2 r3 : ℝ), r1 ≠ r2 ∧ r1 ≠ r3 ∧ r2 ≠ r3 ∧ 
    (∀ x : ℝ, (x + c) * (x + d) * (x + 10) / (x + 2)^2 = 0 ↔ x = r1 ∨ x = r2 ∨ x = r3)) ∧
  (∃! (r : ℝ), ∀ x : ℝ, (x + 2*c) * (x + 4) * (x + 8) / ((x + d) * (x + 10)) = 0 ↔ x = r) →
  200 * c + d = 392 :=
by sorry

end NUMINAMATH_CALUDE_root_equation_problem_l2658_265836


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l2658_265853

theorem sin_2alpha_value (α : ℝ) (h : Real.sin α + Real.cos α = 1/5) : 
  Real.sin (2 * α) = -24/25 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l2658_265853


namespace NUMINAMATH_CALUDE_sum_of_roots_is_eight_l2658_265897

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the symmetry property
def symmetric_about_two (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (2 + x) = f (2 - x)

-- Define the property of having exactly four distinct real roots
def has_four_distinct_roots (f : ℝ → ℝ) : Prop :=
  ∃ (r₁ r₂ r₃ r₄ : ℝ), 
    (f r₁ = 0 ∧ f r₂ = 0 ∧ f r₃ = 0 ∧ f r₄ = 0) ∧
    (r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₁ ≠ r₄ ∧ r₂ ≠ r₃ ∧ r₂ ≠ r₄ ∧ r₃ ≠ r₄) ∧
    (∀ x : ℝ, f x = 0 → (x = r₁ ∨ x = r₂ ∨ x = r₃ ∨ x = r₄))

-- State the theorem
theorem sum_of_roots_is_eight (f : ℝ → ℝ) 
  (h_sym : symmetric_about_two f) 
  (h_roots : has_four_distinct_roots f) : 
  ∃ (r₁ r₂ r₃ r₄ : ℝ), 
    (f r₁ = 0 ∧ f r₂ = 0 ∧ f r₃ = 0 ∧ f r₄ = 0) ∧
    (r₁ + r₂ + r₃ + r₄ = 8) :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_is_eight_l2658_265897


namespace NUMINAMATH_CALUDE_fruit_condition_percentage_l2658_265876

/-- Calculates the percentage of fruits in good condition given the number of oranges and bananas and their rotten percentages -/
def percentageGoodFruits (totalOranges totalBananas : ℕ) (rottenOrangesPercent rottenBananasPercent : ℚ) : ℚ :=
  let goodOranges : ℚ := totalOranges * (1 - rottenOrangesPercent)
  let goodBananas : ℚ := totalBananas * (1 - rottenBananasPercent)
  let totalFruits : ℚ := totalOranges + totalBananas
  (goodOranges + goodBananas) / totalFruits * 100

/-- Theorem stating that given 600 oranges with 15% rotten and 400 bananas with 4% rotten, 
    the percentage of fruits in good condition is 89.4% -/
theorem fruit_condition_percentage : 
  percentageGoodFruits 600 400 (15/100) (4/100) = 89.4 := by
  sorry

end NUMINAMATH_CALUDE_fruit_condition_percentage_l2658_265876


namespace NUMINAMATH_CALUDE_shaded_area_square_with_circles_l2658_265828

/-- The area of the shaded region inside a square with circles at its vertices -/
theorem shaded_area_square_with_circles (side_length : ℝ) (circle_radius : ℝ) 
  (h_side : side_length = 8)
  (h_radius : circle_radius = 3 * Real.sqrt 2) : 
  let square_area := side_length ^ 2
  let triangle_area := (side_length / 2) ^ 2 / 2
  let circle_sector_area := π * circle_radius ^ 2 / 4
  let total_excluded_area := 4 * (triangle_area + circle_sector_area)
  square_area - total_excluded_area = 46 - 18 * π :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_square_with_circles_l2658_265828


namespace NUMINAMATH_CALUDE_equation_solution_l2658_265887

theorem equation_solution : ∃! x : ℚ, (3 - x) / (x + 2) + (3*x - 9) / (3 - x) = 2 ∧ x = -7/6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2658_265887


namespace NUMINAMATH_CALUDE_red_light_estimation_l2658_265857

theorem red_light_estimation (total_students : ℕ) (total_yes : ℕ) (known_yes_rate : ℚ) :
  total_students = 600 →
  total_yes = 180 →
  known_yes_rate = 1/2 →
  ∃ (estimated_red_light : ℕ), estimated_red_light = 60 :=
by sorry

end NUMINAMATH_CALUDE_red_light_estimation_l2658_265857


namespace NUMINAMATH_CALUDE_puzzle_assembly_time_l2658_265846

-- Define the number of pieces in the puzzle
def puzzle_pieces : ℕ := 121

-- Define the time it takes to assemble the puzzle with the original method
def original_time : ℕ := 120

-- Define the function for the original assembly method (2 pieces per minute)
def original_assembly (t : ℕ) : ℕ := puzzle_pieces - t

-- Define the function for the new assembly method (3 pieces per minute)
def new_assembly (t : ℕ) : ℕ := puzzle_pieces - 2 * t

-- State the theorem
theorem puzzle_assembly_time :
  ∃ (new_time : ℕ), 
    (original_assembly original_time = 1) ∧ 
    (new_assembly new_time = 1) ∧ 
    (new_time = original_time / 2) := by
  sorry

end NUMINAMATH_CALUDE_puzzle_assembly_time_l2658_265846


namespace NUMINAMATH_CALUDE_gcd_lcm_product_360_l2658_265871

theorem gcd_lcm_product_360 : 
  ∃! (s : Finset ℕ), 
    (∀ d ∈ s, d > 0 ∧ ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ d = Nat.gcd a b ∧ d * Nat.lcm a b = 360) ∧ 
    s.card = 17 :=
by sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_360_l2658_265871


namespace NUMINAMATH_CALUDE_shadow_problem_l2658_265870

/-- Given a cube with edge length 2 cm and a light source y cm above an upper vertex
    casting a shadow with area 98 sq cm (excluding the area beneath the cube),
    prove that the greatest integer not exceeding 1000y is 8100. -/
theorem shadow_problem (y : ℝ) : 
  (2 : ℝ) > 0 ∧ 
  y > 0 ∧ 
  (y / (Real.sqrt 102 - 2) = 1) ∧ 
  (98 : ℝ) = (Real.sqrt 102)^2 - 2^2 →
  Int.floor (1000 * y) = 8100 := by
sorry

end NUMINAMATH_CALUDE_shadow_problem_l2658_265870


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l2658_265844

/-- Represents a quadratic equation ax² + bx + c = 0 -/
structure QuadraticEquation where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Represents a solution in the form (m ± √n)/p -/
structure QuadraticSolution where
  m : ℚ
  n : ℕ
  p : ℕ

/-- Check if three numbers are coprime -/
def are_coprime (m : ℚ) (n p : ℕ) : Prop :=
  Nat.gcd (Nat.gcd (Int.natAbs (Rat.num m)) n) p = 1

/-- The main theorem -/
theorem quadratic_solution_sum (eq : QuadraticEquation) (sol : QuadraticSolution) :
  eq.a = 3 ∧ eq.b = -7 ∧ eq.c = 3 ∧
  are_coprime sol.m sol.n sol.p ∧
  (∃ x : ℚ, x * (3 * x - 7) = -3 ∧ 
    (x = (sol.m + Real.sqrt sol.n) / sol.p ∨ 
     x = (sol.m - Real.sqrt sol.n) / sol.p)) →
  sol.m + sol.n + sol.p = 26 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l2658_265844


namespace NUMINAMATH_CALUDE_number_problem_l2658_265823

theorem number_problem (x : ℚ) : (54/2 : ℚ) + 3 * x = 75 → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2658_265823


namespace NUMINAMATH_CALUDE_tan_half_sum_of_angles_l2658_265848

theorem tan_half_sum_of_angles (x y : Real) 
  (h1 : Real.cos x + Real.cos y = 3/5)
  (h2 : Real.sin x + Real.sin y = 1/5) :
  Real.tan ((x + y) / 2) = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_tan_half_sum_of_angles_l2658_265848


namespace NUMINAMATH_CALUDE_systematic_sampling_probability_l2658_265851

/-- Represents a systematic sampling scenario -/
structure SystematicSampling where
  population_size : ℕ
  sample_size : ℕ
  h_pop_size : population_size > 0
  h_sample_size : sample_size > 0
  h_sample_le_pop : sample_size ≤ population_size

/-- The probability of an individual being selected in systematic sampling -/
def selection_probability (s : SystematicSampling) : ℚ :=
  s.sample_size / s.population_size

/-- Theorem stating the probability of selection in the given scenario -/
theorem systematic_sampling_probability 
  (s : SystematicSampling) 
  (h_pop : s.population_size = 42) 
  (h_sample : s.sample_size = 10) : 
  selection_probability s = 5 / 21 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_probability_l2658_265851


namespace NUMINAMATH_CALUDE_two_sqrt_five_less_than_five_l2658_265821

theorem two_sqrt_five_less_than_five : 2 * Real.sqrt 5 < 5 := by
  sorry

end NUMINAMATH_CALUDE_two_sqrt_five_less_than_five_l2658_265821


namespace NUMINAMATH_CALUDE_complex_modulus_l2658_265875

theorem complex_modulus (z : ℂ) : z - Complex.I = 1 + Complex.I → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l2658_265875


namespace NUMINAMATH_CALUDE_smallest_sum_is_26_l2658_265824

def is_valid_pair (m n : ℕ) : Prop :=
  m > 0 ∧ n > 0 ∧ m < n ∧ (1978^m) % 1000 = (1978^n) % 1000

theorem smallest_sum_is_26 :
  ∃ (m n : ℕ), is_valid_pair m n ∧ m + n = 26 ∧
  ∀ (m' n' : ℕ), is_valid_pair m' n' → m' + n' ≥ 26 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_is_26_l2658_265824


namespace NUMINAMATH_CALUDE_vector_projection_l2658_265827

/-- The projection of vector a onto vector b is -√5/5 -/
theorem vector_projection (a b : ℝ × ℝ) : 
  a = (3, 1) → b = (-2, 4) → 
  (a.1 * b.1 + a.2 * b.2) / Real.sqrt (b.1^2 + b.2^2) = -Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_projection_l2658_265827


namespace NUMINAMATH_CALUDE_bus_motion_time_is_24_minutes_l2658_265803

/-- Represents the bus journey on a highway -/
structure BusJourney where
  distance : ℝ  -- Total distance in km
  num_stops : ℕ -- Number of intermediate stops
  stop_duration : ℝ -- Duration of each stop in minutes
  speed_difference : ℝ -- Difference in km/h between non-stop speed and average speed with stops

/-- Calculates the time the bus is in motion -/
def motion_time (journey : BusJourney) : ℝ :=
  sorry

/-- The main theorem stating that the motion time is 24 minutes for the given conditions -/
theorem bus_motion_time_is_24_minutes (journey : BusJourney) 
  (h1 : journey.distance = 10)
  (h2 : journey.num_stops = 6)
  (h3 : journey.stop_duration = 1)
  (h4 : journey.speed_difference = 5) :
  motion_time journey = 24 :=
sorry

end NUMINAMATH_CALUDE_bus_motion_time_is_24_minutes_l2658_265803


namespace NUMINAMATH_CALUDE_evaluate_expression_l2658_265878

theorem evaluate_expression : 3 * 307 + 4 * 307 + 2 * 307 + 307^2 = 97012 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2658_265878


namespace NUMINAMATH_CALUDE_cupcake_problem_l2658_265882

theorem cupcake_problem (total : ℕ) (gf v nf : ℕ) (gf_and_v gf_and_nf nf_and_v gf_and_nf_not_v : ℕ) : 
  total = 200 →
  gf = (40 * total) / 100 →
  v = (25 * total) / 100 →
  nf = (30 * total) / 100 →
  gf_and_v = (20 * gf) / 100 →
  gf_and_nf = (15 * gf) / 100 →
  nf_and_v = (25 * nf) / 100 →
  gf_and_nf_not_v = (10 * total) / 100 →
  total - (gf + v + nf - gf_and_v - gf_and_nf - nf_and_v + gf_and_nf_not_v) = 33 := by
sorry

end NUMINAMATH_CALUDE_cupcake_problem_l2658_265882


namespace NUMINAMATH_CALUDE_min_value_fraction_sum_min_value_fraction_sum_achievable_l2658_265889

theorem min_value_fraction_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (8 / x + 2 / y) ≥ 18 :=
by sorry

theorem min_value_fraction_sum_achievable :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 1 ∧ (8 / x + 2 / y) = 18 :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_sum_min_value_fraction_sum_achievable_l2658_265889


namespace NUMINAMATH_CALUDE_smallest_n_property_ratio_is_sqrt_three_l2658_265896

/-- The smallest positive integer n for which there exist positive real numbers a and b
    such that (a + bi)^n = -(a - bi)^n -/
def smallest_n : ℕ := 4

theorem smallest_n_property (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (Complex.I : ℂ)^smallest_n * (a + b * Complex.I)^smallest_n = -(a - b * Complex.I)^smallest_n :=
sorry

theorem ratio_is_sqrt_three (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : (Complex.I : ℂ)^smallest_n * (a + b * Complex.I)^smallest_n = -(a - b * Complex.I)^smallest_n) :
  a / b = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_property_ratio_is_sqrt_three_l2658_265896


namespace NUMINAMATH_CALUDE_floor_ceil_sum_seven_l2658_265845

theorem floor_ceil_sum_seven (x : ℝ) : 
  (⌊x⌋ + ⌈x⌉ = 7) ↔ (3 < x ∧ x < 4) :=
sorry

end NUMINAMATH_CALUDE_floor_ceil_sum_seven_l2658_265845


namespace NUMINAMATH_CALUDE_gcd_of_factorials_l2658_265825

-- Define factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Theorem statement
theorem gcd_of_factorials : Nat.gcd (factorial 8) (Nat.gcd (factorial 10) (factorial 11)) = factorial 8 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_factorials_l2658_265825


namespace NUMINAMATH_CALUDE_initially_tagged_fish_l2658_265820

-- Define the total number of fish in the pond
def total_fish : ℕ := 750

-- Define the number of fish in the second catch
def second_catch : ℕ := 50

-- Define the number of tagged fish in the second catch
def tagged_in_second_catch : ℕ := 2

-- Define the ratio of tagged fish in the second catch
def tagged_ratio : ℚ := tagged_in_second_catch / second_catch

-- Theorem: The number of fish initially caught and tagged is 30
theorem initially_tagged_fish : 
  ∃ (T : ℕ), T = 30 ∧ (T : ℚ) / total_fish = tagged_ratio :=
sorry

end NUMINAMATH_CALUDE_initially_tagged_fish_l2658_265820


namespace NUMINAMATH_CALUDE_complement_union_and_intersection_range_of_a_l2658_265881

-- Define the sets A, B, and C
def A : Set ℝ := {x : ℝ | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x : ℝ | x < a}

-- Theorem for part (1)
theorem complement_union_and_intersection :
  (Set.univ \ (A ∪ B) = {x : ℝ | x ≤ 2 ∨ x ≥ 10}) ∧
  ((Set.univ \ A) ∩ B = {x : ℝ | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)}) := by sorry

-- Theorem for part (2)
theorem range_of_a (h : A ∩ C a ≠ ∅) : a > 3 := by sorry

end NUMINAMATH_CALUDE_complement_union_and_intersection_range_of_a_l2658_265881


namespace NUMINAMATH_CALUDE_probability_of_U_in_SHUXUE_l2658_265873

def pinyin : String := "SHUXUE"

theorem probability_of_U_in_SHUXUE : 
  (pinyin.toList.filter (· = 'U')).length / pinyin.length = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_U_in_SHUXUE_l2658_265873


namespace NUMINAMATH_CALUDE_semicircle_path_equality_l2658_265855

theorem semicircle_path_equality :
  let large_diameter : ℝ := 20
  let small_diameter : ℝ := 10
  let large_arc_length := π * large_diameter / 2
  let small_arc_length := π * small_diameter / 2
  large_arc_length = 2 * small_arc_length :=
by sorry

end NUMINAMATH_CALUDE_semicircle_path_equality_l2658_265855


namespace NUMINAMATH_CALUDE_abc_inequality_l2658_265819

theorem abc_inequality (a b c : ℝ) (ha : |a| < 1) (hb : |b| < 1) (hc : |c| < 1) :
  a * b + b * c + c * a > -1 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l2658_265819


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l2658_265837

theorem triangle_angle_measure (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  a^2 - b^2 = Real.sqrt 3 * b * c →
  Real.sin C = 2 * Real.sqrt 3 * Real.sin B →
  a = 2 * Real.sin A →
  b = 2 * Real.sin B →
  c = 2 * Real.sin C →
  A + B + C = π →
  A = π / 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l2658_265837


namespace NUMINAMATH_CALUDE_even_function_characterization_l2658_265860

def M (f : ℝ → ℝ) (a : ℝ) : Set ℝ :=
  {t | ∃ x ≥ a, t = f x - f a}

def L (f : ℝ → ℝ) (a : ℝ) : Set ℝ :=
  {t | ∃ x ≤ a, t = f x - f a}

def has_minimum (f : ℝ → ℝ) : Prop :=
  ∃ m : ℝ, ∀ x : ℝ, f m ≤ f x

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

theorem even_function_characterization (f : ℝ → ℝ) (h : has_minimum f) :
  is_even_function f ↔ ∀ c > 0, M f (-c) = L f c := by
  sorry

end NUMINAMATH_CALUDE_even_function_characterization_l2658_265860


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l2658_265806

theorem quadratic_roots_range (a : ℝ) : 
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ x^2 + a*x + a^2 - 1 = 0 ∧ y^2 + a*y + a^2 - 1 = 0) → 
  -1 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l2658_265806


namespace NUMINAMATH_CALUDE_remainder_of_111222333_div_37_l2658_265816

theorem remainder_of_111222333_div_37 : 111222333 % 37 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_111222333_div_37_l2658_265816


namespace NUMINAMATH_CALUDE_rope_cutting_l2658_265840

theorem rope_cutting (total_length : ℝ) (piece_length : ℝ) 
  (h1 : total_length = 20) 
  (h2 : piece_length = 3.8) : 
  (∃ (num_pieces : ℕ) (remaining : ℝ), 
    num_pieces = 5 ∧ 
    remaining = 1 ∧ 
    (↑num_pieces : ℝ) * piece_length + remaining = total_length ∧ 
    remaining < piece_length) := by
  sorry

end NUMINAMATH_CALUDE_rope_cutting_l2658_265840


namespace NUMINAMATH_CALUDE_four_propositions_true_l2658_265895

theorem four_propositions_true (x y : ℝ) : 
  (((x = 0 ∧ y = 0) → (x^2 + y^2 ≠ 0)) ∧                   -- Original
   ((x^2 + y^2 ≠ 0) → (x = 0 ∧ y = 0)) ∧                   -- Converse
   (¬(x = 0 ∧ y = 0) → ¬(x^2 + y^2 ≠ 0)) ∧                 -- Inverse
   (¬(x^2 + y^2 ≠ 0) → ¬(x = 0 ∧ y = 0)))                  -- Contrapositive
  := by sorry

end NUMINAMATH_CALUDE_four_propositions_true_l2658_265895


namespace NUMINAMATH_CALUDE_hyperbola_equation_for_given_parameters_l2658_265843

/-- Represents a hyperbola with foci on the x-axis -/
structure Hyperbola where
  a : ℝ  -- Half of the real axis length
  e : ℝ  -- Eccentricity

/-- The equation of a hyperbola with given parameters -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / (h.a^2 * (h.e^2 - 1)) = 1

theorem hyperbola_equation_for_given_parameters :
  let h : Hyperbola := { a := 3, e := 5/3 }
  ∀ x y : ℝ, hyperbola_equation h x y ↔ x^2 / 9 - y^2 / 16 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_for_given_parameters_l2658_265843


namespace NUMINAMATH_CALUDE_mineral_worth_l2658_265800

/-- The worth of a mineral given its price per gram and weight -/
theorem mineral_worth (price_per_gram : ℝ) (weight_1 weight_2 : ℝ) :
  price_per_gram = 17.25 →
  weight_1 = 1000 →
  weight_2 = 10 →
  price_per_gram * (weight_1 + weight_2) = 17422.5 := by
  sorry

end NUMINAMATH_CALUDE_mineral_worth_l2658_265800


namespace NUMINAMATH_CALUDE_amelia_win_probability_l2658_265838

/-- Probability of Amelia's coin landing on heads -/
def p_amelia : ℚ := 3/7

/-- Probability of Blaine's coin landing on heads -/
def p_blaine : ℚ := 1/3

/-- Probability of at least one head in a simultaneous toss -/
def p_start : ℚ := 1 - (1 - p_amelia) * (1 - p_blaine)

/-- Probability of Amelia winning on her turn -/
def p_amelia_win : ℚ := p_amelia * p_amelia

/-- Probability of Blaine winning on his turn -/
def p_blaine_win : ℚ := p_blaine * p_blaine

/-- Probability of delay (neither wins) -/
def p_delay : ℚ := 1 - p_amelia_win - p_blaine_win

/-- The probability that Amelia wins the game -/
theorem amelia_win_probability : 
  (p_amelia_win / (1 - p_delay^2 : ℚ)) = 21609/64328 := by
  sorry

end NUMINAMATH_CALUDE_amelia_win_probability_l2658_265838


namespace NUMINAMATH_CALUDE_least_reducible_fraction_l2658_265830

theorem least_reducible_fraction (n : ℕ) : 
  (∀ m : ℕ, m > 0 ∧ m < 34 → ¬(Nat.gcd (m - 29) (3 * m + 8) > 1)) ∧ 
  (34 > 0) ∧ 
  (Nat.gcd (34 - 29) (3 * 34 + 8) > 1) :=
sorry

end NUMINAMATH_CALUDE_least_reducible_fraction_l2658_265830


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2658_265867

theorem complex_fraction_simplification :
  1 + 3 / (2 + 5/6) = 35/17 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2658_265867


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2658_265834

theorem quadratic_factorization (C D : ℤ) :
  (∀ y : ℝ, 15 * y^2 - 76 * y + 48 = (C * y - 16) * (D * y - 3)) →
  C * D + C = 20 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2658_265834


namespace NUMINAMATH_CALUDE_extended_segment_coordinates_l2658_265817

/-- Given two points A and B on a plane, and a point C such that BC = 1/2 * AB,
    this theorem proves that C has specific coordinates. -/
theorem extended_segment_coordinates (A B C : ℝ × ℝ) : 
  A = (2, -2) → 
  B = (14, 4) → 
  C.1 - B.1 = (B.1 - A.1) / 2 → 
  C.2 - B.2 = (B.2 - A.2) / 2 → 
  C = (20, 7) := by
sorry


end NUMINAMATH_CALUDE_extended_segment_coordinates_l2658_265817


namespace NUMINAMATH_CALUDE_unique_solution_iff_l2658_265898

/-- The function f(x) = x^2 + 3bx + 4b -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + 3*b*x + 4*b

/-- The property that |f(x)| ≤ 3 has exactly one solution -/
def has_unique_solution (b : ℝ) : Prop :=
  ∃! x : ℝ, |f b x| ≤ 3

/-- Theorem stating that the inequality has a unique solution iff b = 4/3 or b = 1 -/
theorem unique_solution_iff (b : ℝ) :
  has_unique_solution b ↔ b = 4/3 ∨ b = 1 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_iff_l2658_265898


namespace NUMINAMATH_CALUDE_intersection_point_equivalence_l2658_265810

theorem intersection_point_equivalence 
  (m n a b : ℝ) 
  (h1 : m * a + 2 * m * b = 5) 
  (h2 : n * a - 2 * n * b = 7) :
  (5 / (2 * m) - a / 2 = b) ∧ (a / 2 - 7 / (2 * n) = b) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_equivalence_l2658_265810


namespace NUMINAMATH_CALUDE_black_hole_convergence_l2658_265854

/-- Counts the number of even digits in a natural number -/
def countEvenDigits (n : ℕ) : ℕ := sorry

/-- Counts the number of odd digits in a natural number -/
def countOddDigits (n : ℕ) : ℕ := sorry

/-- Counts the total number of digits in a natural number -/
def countTotalDigits (n : ℕ) : ℕ := sorry

/-- Applies the transformation rule to a natural number -/
def transform (n : ℕ) : ℕ :=
  100 * (countEvenDigits n) + 10 * (countOddDigits n) + (countTotalDigits n)

/-- The black hole number -/
def blackHoleNumber : ℕ := 123

/-- Theorem stating that repeated application of the transformation 
    will always result in the black hole number -/
theorem black_hole_convergence (n : ℕ) : 
  ∃ k : ℕ, ∀ m : ℕ, m ≥ k → (transform^[m] n = blackHoleNumber) :=
sorry

end NUMINAMATH_CALUDE_black_hole_convergence_l2658_265854


namespace NUMINAMATH_CALUDE_absolute_value_two_l2658_265890

theorem absolute_value_two (m : ℝ) : |m| = 2 → m = 2 ∨ m = -2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_two_l2658_265890


namespace NUMINAMATH_CALUDE_fraction_equality_l2658_265813

theorem fraction_equality (a b : ℝ) : (0.3 * a + b) / (0.2 * a + 0.5 * b) = (3 * a + 10 * b) / (2 * a + 5 * b) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2658_265813


namespace NUMINAMATH_CALUDE_gcd_count_for_product_360_l2658_265884

theorem gcd_count_for_product_360 (a b : ℕ+) : 
  (Nat.gcd a b * Nat.lcm a b = 360) → 
  (∃ (S : Finset ℕ), S.card = 8 ∧ ∀ d, d ∈ S ↔ ∃ (x y : ℕ+), Nat.gcd x y = d ∧ Nat.gcd x y * Nat.lcm x y = 360) :=
sorry

end NUMINAMATH_CALUDE_gcd_count_for_product_360_l2658_265884


namespace NUMINAMATH_CALUDE_solution_set_implies_sum_l2658_265801

-- Define the quadratic function
def f (a b : ℝ) (x : ℝ) := a * x^2 + b * x + 2

-- State the theorem
theorem solution_set_implies_sum (a b : ℝ) :
  (∀ x, f a b x > 0 ↔ x ∈ Set.Ioo (-1/2 : ℝ) (1/3 : ℝ)) →
  a + b = -14 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_sum_l2658_265801


namespace NUMINAMATH_CALUDE_power_product_cube_l2658_265872

theorem power_product_cube (a b : ℝ) : (a * b) ^ 3 = a ^ 3 * b ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_power_product_cube_l2658_265872


namespace NUMINAMATH_CALUDE_triangle_most_stable_triangular_structures_sturdy_l2658_265818

-- Define a structure
structure Shape :=
  (stability : ℝ)

-- Define a triangle
def Triangle : Shape :=
  { stability := 1 }

-- Define other shapes (for comparison)
def Square : Shape :=
  { stability := 0.8 }

def Pentagon : Shape :=
  { stability := 0.9 }

-- Theorem: Triangles have the highest stability
theorem triangle_most_stable :
  ∀ s : Shape, Triangle.stability ≥ s.stability :=
sorry

-- Theorem: Structures using triangles are sturdy
theorem triangular_structures_sturdy (structure_stability : Shape → ℝ) :
  structure_stability Triangle = 1 →
  ∀ s : Shape, structure_stability Triangle ≥ structure_stability s :=
sorry

end NUMINAMATH_CALUDE_triangle_most_stable_triangular_structures_sturdy_l2658_265818


namespace NUMINAMATH_CALUDE_track_laying_equation_l2658_265815

theorem track_laying_equation (x : ℝ) (h : x > 0) :
  (6000 / x - 6000 / (x + 20) = 15) ↔
  (∃ (original_days revised_days : ℝ),
    original_days > 0 ∧
    revised_days > 0 ∧
    original_days = 6000 / x ∧
    revised_days = 6000 / (x + 20) ∧
    original_days - revised_days = 15) :=
by sorry

end NUMINAMATH_CALUDE_track_laying_equation_l2658_265815


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l2658_265826

/-- A quadratic equation x^2 + 2x - c = 0 has two equal real roots if and only if c = -1 -/
theorem equal_roots_quadratic (c : ℝ) : 
  (∃ x : ℝ, x^2 + 2*x - c = 0 ∧ (∀ y : ℝ, y^2 + 2*y - c = 0 → y = x)) ↔ c = -1 :=
by sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l2658_265826


namespace NUMINAMATH_CALUDE_find_k_l2658_265849

theorem find_k : ∃ (k : ℤ) (m : ℝ), ∀ (n : ℝ), 
  n * (n + 1) * (n + 2) * (n + 3) + m = (n^2 + k * n + 1)^2 → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_find_k_l2658_265849


namespace NUMINAMATH_CALUDE_tan_pi_plus_alpha_problem_l2658_265877

theorem tan_pi_plus_alpha_problem (α : Real) (h : Real.tan (Real.pi + α) = -1/2) :
  (2 * Real.cos (Real.pi - α) - 3 * Real.sin (Real.pi + α)) /
  (4 * Real.cos (α - 2 * Real.pi) + Real.cos (3/2 * Real.pi - α)) = -7/9 ∧
  Real.sin α ^ 2 - 2 * Real.sin α * Real.cos α + 4 * Real.cos α ^ 2 = 21/5 := by
  sorry

end NUMINAMATH_CALUDE_tan_pi_plus_alpha_problem_l2658_265877


namespace NUMINAMATH_CALUDE_min_perimeter_isosceles_triangles_l2658_265874

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  base : ℕ
  leg : ℕ

/-- Calculates the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ := t.base + 2 * t.leg

/-- Calculates the area of an isosceles triangle -/
noncomputable def area (t : IsoscelesTriangle) : ℝ :=
  (t.base : ℝ) * Real.sqrt ((t.leg : ℝ) ^ 2 - ((t.base : ℝ) / 2) ^ 2) / 2

/-- Theorem: The minimum possible value of the common perimeter of two noncongruent
    integer-sided isosceles triangles with the same perimeter, same area, and base
    lengths in the ratio 8:7 is 586 -/
theorem min_perimeter_isosceles_triangles :
  ∃ (t1 t2 : IsoscelesTriangle),
    t1 ≠ t2 ∧
    perimeter t1 = perimeter t2 ∧
    area t1 = area t2 ∧
    8 * t2.base = 7 * t1.base ∧
    perimeter t1 = 586 ∧
    (∀ (s1 s2 : IsoscelesTriangle),
      s1 ≠ s2 →
      perimeter s1 = perimeter s2 →
      area s1 = area s2 →
      8 * s2.base = 7 * s1.base →
      perimeter s1 ≥ 586) :=
by
  sorry

end NUMINAMATH_CALUDE_min_perimeter_isosceles_triangles_l2658_265874


namespace NUMINAMATH_CALUDE_factorization_proof_l2658_265805

variable (x y a b : ℝ)

theorem factorization_proof :
  (9*(3*x - 2*y)^2 - (x - y)^2 = (10*x - 7*y)*(8*x - 5*y)) ∧
  (a^2*b^2 + 4*a*b + 4 - b^2 = (a*b + 2 + b)*(a*b + 2 - b)) := by
  sorry


end NUMINAMATH_CALUDE_factorization_proof_l2658_265805


namespace NUMINAMATH_CALUDE_P_inter_Q_eq_interval_l2658_265865

/-- The set P defined by the inequality 3x - x^2 ≤ 0 -/
def P : Set ℝ := {x : ℝ | 3 * x - x^2 ≤ 0}

/-- The set Q defined by the inequality |x| ≤ 2 -/
def Q : Set ℝ := {x : ℝ | |x| ≤ 2}

/-- The theorem stating that the intersection of P and Q is equal to the set {x | -2 ≤ x ≤ 0} -/
theorem P_inter_Q_eq_interval : P ∩ Q = {x : ℝ | -2 ≤ x ∧ x ≤ 0} := by sorry

end NUMINAMATH_CALUDE_P_inter_Q_eq_interval_l2658_265865


namespace NUMINAMATH_CALUDE_volleyball_team_math_players_l2658_265847

theorem volleyball_team_math_players 
  (total_players : ℕ) 
  (physics_players : ℕ) 
  (both_subjects : ℕ) 
  (h1 : total_players = 15)
  (h2 : physics_players = 10)
  (h3 : both_subjects = 4)
  (h4 : physics_players ≤ total_players)
  (h5 : both_subjects ≤ physics_players)
  (h6 : ∀ p, p ∈ (Finset.range total_players) → 
    (p ∈ (Finset.range physics_players) ∨ 
     p ∈ (Finset.range (total_players - physics_players + both_subjects)))) :
  total_players - physics_players + both_subjects = 9 := by
sorry

end NUMINAMATH_CALUDE_volleyball_team_math_players_l2658_265847


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l2658_265858

/-- The perimeter of a rhombus with diagonals of 18 inches and 32 inches is 4√337 inches. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 18) (h2 : d2 = 32) :
  4 * (((d1 / 2) ^ 2 + (d2 / 2) ^ 2).sqrt) = 4 * Real.sqrt 337 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l2658_265858


namespace NUMINAMATH_CALUDE_corner_spheres_sum_diameter_l2658_265833

-- Define a sphere in a corner
structure CornerSphere where
  radius : ℝ
  center : ℝ × ℝ × ℝ

-- Define the condition for a point on the sphere
def satisfiesCondition (s : CornerSphere) : Prop :=
  ∃ (x y z : ℝ), 
    (x - s.radius)^2 + (y - s.radius)^2 + (z - s.radius)^2 = s.radius^2 ∧
    x = 5 ∧ y = 5 ∧ z = 10

theorem corner_spheres_sum_diameter :
  ∀ (s1 s2 : CornerSphere),
    satisfiesCondition s1 → satisfiesCondition s2 →
    s1.center = (s1.radius, s1.radius, s1.radius) →
    s2.center = (s2.radius, s2.radius, s2.radius) →
    2 * (s1.radius + s2.radius) = 40 := by
  sorry

end NUMINAMATH_CALUDE_corner_spheres_sum_diameter_l2658_265833


namespace NUMINAMATH_CALUDE_consecutive_integers_with_square_factors_l2658_265842

theorem consecutive_integers_with_square_factors (n : ℕ) :
  ∃ x : ℤ, ∀ k : ℕ, k ≥ 1 → k ≤ n →
    ∃ m : ℕ, m > 1 ∧ ∃ y : ℤ, x + k = m^2 * y := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_with_square_factors_l2658_265842


namespace NUMINAMATH_CALUDE_inequality_proof_l2658_265892

theorem inequality_proof (x y : ℝ) (h : x^4 + y^4 ≥ 2) :
  |x^16 - y^16| + 4 * x^8 * y^8 ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2658_265892


namespace NUMINAMATH_CALUDE_cylinder_volume_equality_l2658_265841

/-- Proves that for two cylinders with initial radius 5 inches and height 4 inches, 
    if the radius of one and the height of the other are increased by y inches, 
    and their volumes become equal, then y = 5/4 inches. -/
theorem cylinder_volume_equality (y : ℚ) : 
  y ≠ 0 → 
  π * (5 + y)^2 * 4 = π * 5^2 * (4 + y) → 
  y = 5/4 := by sorry

end NUMINAMATH_CALUDE_cylinder_volume_equality_l2658_265841


namespace NUMINAMATH_CALUDE_inequality_proof_l2658_265822

theorem inequality_proof (x m : ℝ) (hx : x ≥ 1) (hm : m ≥ 1/2) :
  x * Real.log x ≤ m * (x^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2658_265822


namespace NUMINAMATH_CALUDE_difference_of_squares_special_case_l2658_265888

theorem difference_of_squares_special_case : (831 : ℤ) * 831 - 830 * 832 = 1 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_special_case_l2658_265888


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2658_265861

theorem complex_equation_solution (b : ℝ) : 
  (2 - Complex.I) * (4 * Complex.I) = 4 - b * Complex.I → b = -8 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2658_265861


namespace NUMINAMATH_CALUDE_max_product_sum_2006_l2658_265829

theorem max_product_sum_2006 :
  ∃ (a b : ℤ), a + b = 2006 ∧
    ∀ (x y : ℤ), x + y = 2006 → x * y ≤ a * b ∧
    a * b = 1006009 := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_2006_l2658_265829
