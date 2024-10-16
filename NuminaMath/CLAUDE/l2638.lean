import Mathlib

namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2638_263845

theorem complex_fraction_equality : Complex.I * 2 / (1 - Complex.I) = -1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2638_263845


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l2638_263847

open Set

-- Define the universal set U
def U : Set Int := {-1, 0, 1, 2, 3}

-- Define set A
def A : Set Int := {-1, 0}

-- Define set B
def B : Set Int := {0, 1, 2}

-- Theorem statement
theorem complement_intersection_theorem :
  (U \ A) ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l2638_263847


namespace NUMINAMATH_CALUDE_young_inequality_l2638_263851

theorem young_inequality (x y α β : ℝ) 
  (hx : x > 0) (hy : y > 0) (hα : α > 0) (hβ : β > 0) (hsum : α + β = 1) :
  x^α * y^β ≤ α*x + β*y :=
sorry

end NUMINAMATH_CALUDE_young_inequality_l2638_263851


namespace NUMINAMATH_CALUDE_min_value_theorem_l2638_263879

/-- Given real numbers a, b, c, d satisfying the equation,
    the minimum value of the expression is 8 -/
theorem min_value_theorem (a b c d : ℝ) 
  (h : (b - 2*a^2 + 3*Real.log a)^2 + (c - d - 3)^2 = 0) :
  ∃ (m : ℝ), m = 8 ∧ ∀ (x y z w : ℝ), 
    (y - 2*x^2 + 3*Real.log x)^2 + (z - w - 3)^2 = 0 → 
    (x - z)^2 + (y - w)^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2638_263879


namespace NUMINAMATH_CALUDE_perimeter_is_14_l2638_263823

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  side_length : ℝ

/-- Represents a pentagon formed by cutting a smaller equilateral triangle from a larger one -/
structure CutTrianglePentagon where
  original_triangle : EquilateralTriangle
  cut_triangle : EquilateralTriangle

/-- Calculate the perimeter of the pentagon formed by cutting a smaller equilateral triangle from a larger one -/
def perimeter_of_cut_triangle_pentagon (p : CutTrianglePentagon) : ℝ :=
  p.original_triangle.side_length * 2 + 
  (p.original_triangle.side_length - p.cut_triangle.side_length) + 
  p.cut_triangle.side_length * 3

/-- Theorem stating that the perimeter of the pentagon is 14 units -/
theorem perimeter_is_14 (p : CutTrianglePentagon) 
  (h1 : p.original_triangle.side_length = 5) 
  (h2 : p.cut_triangle.side_length = 2) : 
  perimeter_of_cut_triangle_pentagon p = 14 := by
  sorry

#check perimeter_is_14

end NUMINAMATH_CALUDE_perimeter_is_14_l2638_263823


namespace NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l2638_263892

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  max ((b + c) / a) (max ((a + c) / b) ((a + b) / c)) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l2638_263892


namespace NUMINAMATH_CALUDE_cemc_employee_change_l2638_263806

/-- The net change in employees for Canadian Excellent Mathematics Corporation in 2018 -/
theorem cemc_employee_change (t : ℕ) (h : t = 120) : 
  (((t : ℚ) * (1 + 0.25) + (40 : ℚ) * (1 - 0.35)) - (t + 40 : ℚ)).floor = 16 := by
  sorry

end NUMINAMATH_CALUDE_cemc_employee_change_l2638_263806


namespace NUMINAMATH_CALUDE_exponent_product_square_l2638_263885

theorem exponent_product_square (x y : ℝ) : (3 * x * y)^2 = 9 * x^2 * y^2 := by
  sorry

end NUMINAMATH_CALUDE_exponent_product_square_l2638_263885


namespace NUMINAMATH_CALUDE_complex_magnitude_l2638_263839

theorem complex_magnitude (z : ℂ) (h : z * (1 + 2*Complex.I) + Complex.I = 0) : 
  Complex.abs z = Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2638_263839


namespace NUMINAMATH_CALUDE_candy_difference_l2638_263864

/-- Theorem about the difference in candy counts in a box of rainbow nerds -/
theorem candy_difference (purple : ℕ) (yellow : ℕ) (green : ℕ) (total : ℕ) : 
  purple = 10 →
  yellow = purple + 4 →
  total = 36 →
  purple + yellow + green = total →
  yellow - green = 2 := by
sorry

end NUMINAMATH_CALUDE_candy_difference_l2638_263864


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l2638_263834

theorem polynomial_coefficient_sum (a₀ a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, (5 * x + 4) ^ 3 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3) →
  (a₀ + a₂) - (a₁ + a₃) = -1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l2638_263834


namespace NUMINAMATH_CALUDE_sharons_journey_l2638_263835

theorem sharons_journey (normal_time : ℝ) (traffic_time : ℝ) (speed_reduction : ℝ) :
  normal_time = 150 →
  traffic_time = 250 →
  speed_reduction = 15 →
  ∃ (distance : ℝ),
    distance = 80 ∧
    (distance / 4) / (distance / normal_time) +
    ((3 * distance) / 4) / ((distance / normal_time) - (speed_reduction / 60)) = traffic_time :=
by sorry

end NUMINAMATH_CALUDE_sharons_journey_l2638_263835


namespace NUMINAMATH_CALUDE_functional_equation_properties_l2638_263894

/-- A function satisfying the given functional equation -/
noncomputable def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x * f y

theorem functional_equation_properties (f : ℝ → ℝ) 
  (h_eq : FunctionalEquation f) (h_nonzero : f 0 ≠ 0) : 
  (f 0 = 1) ∧ 
  (∀ x : ℝ, f (-x) = f x) ∧ 
  (∃ c : ℝ, (∀ x : ℝ, f (x + 2*c) = f x) ∧ f c = -1) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_properties_l2638_263894


namespace NUMINAMATH_CALUDE_workshop_equation_system_l2638_263868

/-- Represents the production capabilities of workers for desks and chairs -/
structure ProductionRate where
  desk : ℕ
  chair : ℕ

/-- Represents the composition of a set of furniture -/
structure FurnitureSet where
  desk : ℕ
  chair : ℕ

/-- The problem setup for the furniture workshop -/
structure WorkshopSetup where
  totalWorkers : ℕ
  productionRate : ProductionRate
  furnitureSet : FurnitureSet

/-- Theorem stating the correct system of equations for the workshop problem -/
theorem workshop_equation_system 
  (setup : WorkshopSetup)
  (h_setup : setup.totalWorkers = 32 ∧ 
             setup.productionRate = { desk := 5, chair := 6 } ∧
             setup.furnitureSet = { desk := 1, chair := 2 }) :
  ∃ (x y : ℕ), 
    x + y = setup.totalWorkers ∧ 
    2 * (setup.productionRate.desk * x) = setup.productionRate.chair * y :=
sorry

end NUMINAMATH_CALUDE_workshop_equation_system_l2638_263868


namespace NUMINAMATH_CALUDE_existence_of_differences_l2638_263837

theorem existence_of_differences (n : ℕ) (x : Fin n → Fin n → ℚ)
  (h : ∀ (i j k : Fin n), x i j + x j k + x k i = 0) :
  ∃ (a : Fin n → ℚ), ∀ (i j : Fin n), x i j = a i - a j := by
  sorry

end NUMINAMATH_CALUDE_existence_of_differences_l2638_263837


namespace NUMINAMATH_CALUDE_price_equation_system_l2638_263801

/-- Represents the price of a basketball in yuan -/
def basketball_price : ℝ := sorry

/-- Represents the price of a soccer ball in yuan -/
def soccer_ball_price : ℝ := sorry

/-- The total cost of 3 basketballs and 4 soccer balls is 330 yuan -/
axiom total_cost : 3 * basketball_price + 4 * soccer_ball_price = 330

/-- The price of a basketball is 5 yuan less than the price of a soccer ball -/
axiom price_difference : basketball_price = soccer_ball_price - 5

/-- The system of equations accurately represents the given conditions -/
theorem price_equation_system : 
  (3 * basketball_price + 4 * soccer_ball_price = 330) ∧ 
  (basketball_price = soccer_ball_price - 5) :=
by sorry

end NUMINAMATH_CALUDE_price_equation_system_l2638_263801


namespace NUMINAMATH_CALUDE_train_speed_problem_l2638_263822

theorem train_speed_problem (train_length : ℝ) (crossing_time : ℝ) :
  train_length = 120 →
  crossing_time = 4 →
  ∃ (speed : ℝ),
    speed * crossing_time = 2 * train_length ∧
    speed * 3.6 = 108 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_problem_l2638_263822


namespace NUMINAMATH_CALUDE_calculator_sales_loss_l2638_263856

theorem calculator_sales_loss (price : ℝ) (profit_percent : ℝ) (loss_percent : ℝ) :
  price = 135 ∧ profit_percent = 25 ∧ loss_percent = 25 →
  ∃ (cost1 cost2 : ℝ),
    cost1 + (profit_percent / 100) * cost1 = price ∧
    cost2 - (loss_percent / 100) * cost2 = price ∧
    2 * price - (cost1 + cost2) = -18 :=
by sorry

end NUMINAMATH_CALUDE_calculator_sales_loss_l2638_263856


namespace NUMINAMATH_CALUDE_not_always_complete_gear_possible_l2638_263862

-- Define the number of teeth on each gear
def num_teeth : ℕ := 13

-- Define the number of pairs of teeth removed
def num_removed : ℕ := 4

-- Define a type for the positions of removed teeth
def RemovedTeeth := Fin num_teeth

-- Define a function to check if two positions overlap after rotation
def overlaps (x y : RemovedTeeth) (rotation : ℕ) : Prop :=
  (x.val + rotation) % num_teeth = y.val

-- State the theorem
theorem not_always_complete_gear_possible : ∃ (removed : Fin num_removed → RemovedTeeth),
  ∀ (rotation : ℕ), ∃ (i j : Fin num_removed), i ≠ j ∧ overlaps (removed i) (removed j) rotation :=
sorry

end NUMINAMATH_CALUDE_not_always_complete_gear_possible_l2638_263862


namespace NUMINAMATH_CALUDE_soybean_experiment_results_l2638_263809

/-- Represents the weight distribution of soybean samples -/
structure WeightDistribution :=
  (low : ℕ) -- count in [100, 150) range
  (mid : ℕ) -- count in [150, 200) range
  (high : ℕ) -- count in [200, 250] range

/-- Represents the experimental setup for soybean fields -/
structure SoybeanExperiment :=
  (field_A : WeightDistribution)
  (field_B : WeightDistribution)
  (sample_size : ℕ)
  (critical_value : ℝ)

/-- Calculates the chi-square statistic for the experiment -/
def calculate_chi_square (exp : SoybeanExperiment) : ℝ :=
  sorry

/-- Calculates the probability of selecting at least one full grain from both fields -/
def probability_full_grain (exp : SoybeanExperiment) : ℚ :=
  sorry

/-- Calculates the expected number of full grains in 100 samples from field A -/
def expected_full_grains (exp : SoybeanExperiment) : ℕ :=
  sorry

/-- Calculates the variance of full grains in 100 samples from field A -/
def variance_full_grains (exp : SoybeanExperiment) : ℚ :=
  sorry

/-- Main theorem about the soybean experiment -/
theorem soybean_experiment_results (exp : SoybeanExperiment) 
  (h1 : exp.field_A = ⟨3, 6, 11⟩)
  (h2 : exp.field_B = ⟨6, 10, 4⟩)
  (h3 : exp.sample_size = 20)
  (h4 : exp.critical_value = 5.024) :
  calculate_chi_square exp > exp.critical_value ∧
  probability_full_grain exp = 89 / 100 ∧
  expected_full_grains exp = 55 ∧
  variance_full_grains exp = 99 / 4 :=
sorry

end NUMINAMATH_CALUDE_soybean_experiment_results_l2638_263809


namespace NUMINAMATH_CALUDE_G_function_iff_strictly_increasing_l2638_263857

open Real

/-- Definition of a "G" function on an open interval (a, b) --/
def is_G_function (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, a < x₁ ∧ x₁ < b ∧ a < x₂ ∧ x₂ < b ∧ x₁ ≠ x₂ →
    x₁ * f x₁ + x₂ * f x₂ > x₁ * f x₂ + x₂ * f x₁

/-- A function is strictly increasing on an open interval (a, b) --/
def strictly_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y : ℝ, a < x ∧ x < y ∧ y < b → f x < f y

/-- Theorem: A function is a "G" function on (a, b) if and only if it is strictly increasing on (a, b) --/
theorem G_function_iff_strictly_increasing (f : ℝ → ℝ) (a b : ℝ) :
  is_G_function f a b ↔ strictly_increasing_on f a b :=
sorry

end NUMINAMATH_CALUDE_G_function_iff_strictly_increasing_l2638_263857


namespace NUMINAMATH_CALUDE_garden_area_with_fountain_garden_area_calculation_l2638_263843

/-- Calculates the new available area for planting in a rectangular garden with a circular fountain -/
theorem garden_area_with_fountain (perimeter : ℝ) (side : ℝ) (fountain_radius : ℝ) : ℝ :=
  let length := (perimeter - 2 * side) / 2
  let garden_area := length * side
  let fountain_area := Real.pi * fountain_radius^2
  garden_area - fountain_area

/-- Proves that the new available area for planting is approximately 37185.84 square meters -/
theorem garden_area_calculation : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |garden_area_with_fountain 950 100 10 - 37185.84| < ε :=
sorry

end NUMINAMATH_CALUDE_garden_area_with_fountain_garden_area_calculation_l2638_263843


namespace NUMINAMATH_CALUDE_animal_shelter_problem_l2638_263884

theorem animal_shelter_problem (initial_cats initial_lizards : ℕ)
  (dog_adoption_rate cat_adoption_rate lizard_adoption_rate : ℚ)
  (new_pets total_pets_after_month : ℕ) :
  initial_cats = 28 →
  initial_lizards = 20 →
  dog_adoption_rate = 1/2 →
  cat_adoption_rate = 1/4 →
  lizard_adoption_rate = 1/5 →
  new_pets = 13 →
  total_pets_after_month = 65 →
  ∃ (initial_dogs : ℕ),
    initial_dogs = 30 ∧
    (1 - dog_adoption_rate) * initial_dogs +
    (1 - cat_adoption_rate) * initial_cats +
    (1 - lizard_adoption_rate) * initial_lizards +
    new_pets = total_pets_after_month :=
by sorry

end NUMINAMATH_CALUDE_animal_shelter_problem_l2638_263884


namespace NUMINAMATH_CALUDE_expression_value_l2638_263890

theorem expression_value (m n : ℝ) (h : m * n = m + 3) : 3 * m - 3 * (m * n) + 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2638_263890


namespace NUMINAMATH_CALUDE_ancient_chinese_journey_l2638_263850

/-- Represents the distance walked on each day of a 6-day journey -/
structure JourneyDistances where
  day1 : ℝ
  day2 : ℝ
  day3 : ℝ
  day4 : ℝ
  day5 : ℝ
  day6 : ℝ

/-- The theorem statement for the ancient Chinese mathematical problem -/
theorem ancient_chinese_journey 
  (j : JourneyDistances) 
  (total_distance : j.day1 + j.day2 + j.day3 + j.day4 + j.day5 + j.day6 = 378)
  (day2_half : j.day2 = j.day1 / 2)
  (day3_half : j.day3 = j.day2 / 2)
  (day4_half : j.day4 = j.day3 / 2)
  (day5_half : j.day5 = j.day4 / 2)
  (day6_half : j.day6 = j.day5 / 2) :
  j.day3 = 48 := by
  sorry


end NUMINAMATH_CALUDE_ancient_chinese_journey_l2638_263850


namespace NUMINAMATH_CALUDE_positive_sum_square_inequality_l2638_263816

theorem positive_sum_square_inequality (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) (h₄ : x₄ > 0) (h₅ : x₅ > 0) :
  (x₁ + x₂ + x₃ + x₄ + x₅)^2 ≥ 4 * (x₁*x₂ + x₂*x₃ + x₃*x₄ + x₄*x₅ + x₅*x₁) :=
by sorry

end NUMINAMATH_CALUDE_positive_sum_square_inequality_l2638_263816


namespace NUMINAMATH_CALUDE_union_M_N_equals_reals_l2638_263807

-- Define the sets M and N
def M : Set ℝ := {x | x > 0}
def N : Set ℝ := {x | x^2 ≥ x}

-- State the theorem
theorem union_M_N_equals_reals : M ∪ N = Set.univ := by sorry

end NUMINAMATH_CALUDE_union_M_N_equals_reals_l2638_263807


namespace NUMINAMATH_CALUDE_solution_set_equality_l2638_263865

-- Define the set of real numbers x that satisfy the inequality
def solution_set : Set ℝ := {x : ℝ | (x + 3) / (4 - x) ≥ 0 ∧ x ≠ 4}

-- Theorem stating that the solution set is equal to the interval [-3, 4)
theorem solution_set_equality : solution_set = Set.Icc (-3) 4 \ {4} :=
sorry

end NUMINAMATH_CALUDE_solution_set_equality_l2638_263865


namespace NUMINAMATH_CALUDE_star_symmetric_zero_l2638_263883

/-- Define the binary operation ⋆ for real numbers -/
def star (a b : ℝ) : ℝ := (a^2 - b^2)^2

/-- Theorem: For any real numbers x and y, (x-y)² ⋆ (y-x)² = 0 -/
theorem star_symmetric_zero (x y : ℝ) : star ((x - y)^2) ((y - x)^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_star_symmetric_zero_l2638_263883


namespace NUMINAMATH_CALUDE_smallest_unsubmitted_integer_l2638_263829

/-- Represents the HMMT tournament --/
structure HMMTTournament where
  num_questions : ℕ
  max_expected_answer : ℕ

/-- Defines the property of an integer being submitted as an answer --/
def is_submitted (t : HMMTTournament) (n : ℕ) : Prop := sorry

/-- Theorem stating the smallest unsubmitted positive integer in the HMMT tournament --/
theorem smallest_unsubmitted_integer (t : HMMTTournament) 
  (h1 : t.num_questions = 66)
  (h2 : t.max_expected_answer = 150) :
  ∃ (N : ℕ), N = 139 ∧ 
  (∀ m : ℕ, m < N → is_submitted t m) ∧
  ¬(is_submitted t N) := by
  sorry

end NUMINAMATH_CALUDE_smallest_unsubmitted_integer_l2638_263829


namespace NUMINAMATH_CALUDE_perimeter_T_shape_specific_l2638_263886

/-- Calculates the perimeter of a T shape formed by two rectangles with given dimensions and overlap. -/
def perimeter_T_shape (rect1_length rect1_width rect2_length rect2_width overlap : ℝ) : ℝ :=
  2 * (rect1_length + rect1_width) + 2 * (rect2_length + rect2_width) - 2 * overlap

/-- The perimeter of a T shape formed by two rectangles (3 inch × 5 inch and 2 inch × 6 inch) with a 1-inch overlap is 30 inches. -/
theorem perimeter_T_shape_specific : perimeter_T_shape 3 5 2 6 1 = 30 := by
  sorry

#eval perimeter_T_shape 3 5 2 6 1

end NUMINAMATH_CALUDE_perimeter_T_shape_specific_l2638_263886


namespace NUMINAMATH_CALUDE_tthh_probability_l2638_263899

def coin_flip_probability (n : Nat) (p : Real) : Real :=
  p ^ n * (1 - p) ^ (4 - n)

theorem tthh_probability :
  let p_tails : Real := 1 / 2
  coin_flip_probability 2 p_tails = 0.0625 := by
  sorry

end NUMINAMATH_CALUDE_tthh_probability_l2638_263899


namespace NUMINAMATH_CALUDE_factorization_of_x_squared_plus_2x_l2638_263836

theorem factorization_of_x_squared_plus_2x (x : ℝ) : x^2 + 2*x = x*(x + 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_x_squared_plus_2x_l2638_263836


namespace NUMINAMATH_CALUDE_fraction_addition_simplification_l2638_263861

theorem fraction_addition_simplification : 7/8 + 3/5 = 59/40 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_simplification_l2638_263861


namespace NUMINAMATH_CALUDE_sequence_product_l2638_263888

/-- Given an arithmetic sequence and a geometric sequence with specific properties,
    prove that the product of the second term of the geometric sequence and
    the difference of the second and first terms of the arithmetic sequence is -8. -/
theorem sequence_product (a₁ a₂ b₁ b₂ b₃ : ℝ) : 
  (∀ d : ℝ, -9 + d = a₁ ∧ a₁ + d = a₂ ∧ a₂ + d = -1) →  -- arithmetic sequence condition
  (∃ r : ℝ, -9 * r = b₁ ∧ b₁ * r = b₂ ∧ b₂ * r = b₃ ∧ b₃ * r = -1) →  -- geometric sequence condition
  b₂ * (a₂ - a₁) = -8 := by
sorry

end NUMINAMATH_CALUDE_sequence_product_l2638_263888


namespace NUMINAMATH_CALUDE_contrapositive_falsehood_l2638_263814

theorem contrapositive_falsehood (p q : Prop) :
  (¬(p → q)) → (¬(¬q → ¬p)) := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_falsehood_l2638_263814


namespace NUMINAMATH_CALUDE_card_distribution_l2638_263830

theorem card_distribution (total : ℕ) (ratio_m : ℕ) (ratio_h : ℕ) (ratio_g : ℕ) 
  (h_total : total = 363)
  (h_ratio : ratio_m = 35 ∧ ratio_h = 30 ∧ ratio_g = 56) :
  ∃ (m h g : ℕ), 
    m + h + g = total ∧ 
    m * (ratio_h + ratio_g) = ratio_m * (h + g) ∧
    h * (ratio_m + ratio_g) = ratio_h * (m + g) ∧
    g * (ratio_m + ratio_h) = ratio_g * (m + h) ∧
    m = 105 ∧ h = 90 ∧ g = 168 := by
  sorry

end NUMINAMATH_CALUDE_card_distribution_l2638_263830


namespace NUMINAMATH_CALUDE_probability_all_white_balls_l2638_263810

def total_balls : ℕ := 15
def white_balls : ℕ := 8
def black_balls : ℕ := 7
def drawn_balls : ℕ := 7

theorem probability_all_white_balls :
  (Nat.choose white_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ) = 8 / 6435 :=
sorry

end NUMINAMATH_CALUDE_probability_all_white_balls_l2638_263810


namespace NUMINAMATH_CALUDE_house_selling_price_l2638_263863

/-- Represents the total number of houses in the village -/
def total_houses : ℕ := 15

/-- Represents the total cost of construction for the entire village in millions of units -/
def total_cost : ℕ := 150 + 105 + 225 + 45

/-- Represents the markup percentage as a rational number -/
def markup : ℚ := 1 / 5

/-- Theorem: The selling price of each house in the village is 42 million units -/
theorem house_selling_price : 
  ∃ (cost_per_house : ℕ) (selling_price : ℕ),
    cost_per_house * total_houses = total_cost ∧
    selling_price = cost_per_house + cost_per_house * markup ∧
    selling_price = 42 :=
by sorry

end NUMINAMATH_CALUDE_house_selling_price_l2638_263863


namespace NUMINAMATH_CALUDE_least_common_multiple_plus_one_l2638_263874

def divisors : List Nat := [2, 3, 5, 7, 8, 9, 10]

theorem least_common_multiple_plus_one : 
  ∃ (n : Nat), n > 1 ∧ 
  (∀ d ∈ divisors, n % d = 1) ∧
  (∀ m : Nat, m > 1 → (∀ d ∈ divisors, m % d = 1) → m ≥ n) ∧
  n = 2521 := by
  sorry

end NUMINAMATH_CALUDE_least_common_multiple_plus_one_l2638_263874


namespace NUMINAMATH_CALUDE_part_one_part_two_l2638_263852

-- Define the function f
def f (a x : ℝ) : ℝ := |x - 2*a| - |x - a|

-- Part I
theorem part_one (a : ℝ) : f a 1 > 1 ↔ a ∈ Set.Iic (-1) ∪ Set.Ioi 1 := by sorry

-- Part II
theorem part_two (a : ℝ) (h : a < 0) :
  (∀ x y : ℝ, x ≤ a → y ≤ a → f a x ≤ |y + 2020| + |y - a|) ↔
  a ∈ Set.Icc (-1010) 0 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2638_263852


namespace NUMINAMATH_CALUDE_overtime_rate_multiple_l2638_263880

/-- Calculates the multiple of the regular rate for excess hours worked --/
theorem overtime_rate_multiple
  (regular_hours : ℝ)
  (regular_rate : ℝ)
  (total_hours : ℝ)
  (total_earnings : ℝ)
  (h1 : regular_hours = 7.5)
  (h2 : regular_rate = 4.5)
  (h3 : total_hours = 10.5)
  (h4 : total_earnings = 60.75)
  : (total_earnings - regular_hours * regular_rate) / ((total_hours - regular_hours) * regular_rate) = 2 := by
  sorry


end NUMINAMATH_CALUDE_overtime_rate_multiple_l2638_263880


namespace NUMINAMATH_CALUDE_sum_of_digits_of_greatest_prime_divisor_of_n_l2638_263889

-- Define the number we're working with
def n : ℕ := 9999

-- Define a function to check if a number is prime
def is_prime (p : ℕ) : Prop := Nat.Prime p

-- Define a function to check if a number is a divisor of n
def is_divisor_of_n (d : ℕ) : Prop := n % d = 0

-- Define a function to sum the digits of a number
def sum_of_digits (m : ℕ) : ℕ := sorry

-- Theorem stating the sum of digits of the greatest prime divisor of n is 2
theorem sum_of_digits_of_greatest_prime_divisor_of_n : 
  ∃ p : ℕ, is_prime p ∧ is_divisor_of_n p ∧ 
    (∀ q : ℕ, is_prime q → is_divisor_of_n q → q ≤ p) ∧
    sum_of_digits p = 2 :=
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_greatest_prime_divisor_of_n_l2638_263889


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2638_263866

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_pos : a 1 > 0)
  (h_sum : a 4 + a 7 = 2)
  (h_prod : a 5 * a 6 = -8) :
  a 1 + a 4 + a 7 + a 10 = -5 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2638_263866


namespace NUMINAMATH_CALUDE_quadratic_two_roots_range_l2638_263813

theorem quadratic_two_roots_range (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1/4 = 0 ∧ y^2 + m*y + 1/4 = 0) ↔ 
  (m < -1 ∨ m > 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_roots_range_l2638_263813


namespace NUMINAMATH_CALUDE_sector_arc_length_l2638_263819

/-- Given a sector with radius 2 and area 4, the length of the arc
    corresponding to the central angle is 4. -/
theorem sector_arc_length (r : ℝ) (S : ℝ) (l : ℝ) : 
  r = 2 → S = 4 → S = (1/2) * r * l → l = 4 := by sorry

end NUMINAMATH_CALUDE_sector_arc_length_l2638_263819


namespace NUMINAMATH_CALUDE_f_odd_and_increasing_l2638_263840

def f (x : ℝ) := x * |x|

theorem f_odd_and_increasing :
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x y, x < y → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_f_odd_and_increasing_l2638_263840


namespace NUMINAMATH_CALUDE_no_linear_term_implies_p_value_l2638_263873

theorem no_linear_term_implies_p_value (p : ℝ) : 
  (∀ x : ℝ, ∃ a b c : ℝ, (x - 3) * (x^2 + p*x - 1) = a*x^3 + b*x^2 + c) → 
  p = -1/3 :=
by sorry

end NUMINAMATH_CALUDE_no_linear_term_implies_p_value_l2638_263873


namespace NUMINAMATH_CALUDE_distance_between_circle_centers_l2638_263882

/-- Given a triangle DEF with side lengths, prove the distance between incircle and excircle centers --/
theorem distance_between_circle_centers (DE DF EF : ℝ) (h_DE : DE = 16) (h_DF : DF = 17) (h_EF : EF = 15) :
  let s := (DE + DF + EF) / 2
  let K := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  let r := K / s
  let DI := Real.sqrt (((s - DE) ^ 2) + (r ^ 2))
  let DE' := 3 * DI
  DE' - DI = 10 * Real.sqrt 30 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_circle_centers_l2638_263882


namespace NUMINAMATH_CALUDE_apples_given_to_teachers_l2638_263859

theorem apples_given_to_teachers 
  (total_apples : ℕ) 
  (apples_to_friends : ℕ) 
  (apples_eaten : ℕ) 
  (apples_left : ℕ) 
  (h1 : total_apples = 25)
  (h2 : apples_to_friends = 5)
  (h3 : apples_eaten = 1)
  (h4 : apples_left = 3) :
  total_apples - apples_to_friends - apples_eaten - apples_left = 16 := by
sorry

end NUMINAMATH_CALUDE_apples_given_to_teachers_l2638_263859


namespace NUMINAMATH_CALUDE_x_value_l2638_263811

theorem x_value : ∃ x : ℝ, (3 * x = (16 - x) + 4) ∧ (x = 5) := by
  sorry

end NUMINAMATH_CALUDE_x_value_l2638_263811


namespace NUMINAMATH_CALUDE_chess_tournament_games_l2638_263841

theorem chess_tournament_games (n : ℕ) (h : n = 10) : 
  (n * (n - 1)) / 2 = 45 := by
  sorry

#check chess_tournament_games

end NUMINAMATH_CALUDE_chess_tournament_games_l2638_263841


namespace NUMINAMATH_CALUDE_num_connected_subsets_2x1_l2638_263842

/-- A rectangle in the Cartesian plane -/
structure Rectangle :=
  (bottomLeft : ℝ × ℝ)
  (topRight : ℝ × ℝ)

/-- An edge of a rectangle -/
inductive Edge
  | BottomLeft
  | BottomRight
  | TopLeft
  | TopRight
  | Left
  | Middle
  | Right

/-- A subset of edges -/
def EdgeSubset := Set Edge

/-- Predicate to determine if a subset of edges is connected -/
def is_connected (s : EdgeSubset) : Prop := sorry

/-- The number of connected subsets of edges in a 2x1 rectangle divided into two unit squares -/
def num_connected_subsets (r : Rectangle) : ℕ := sorry

/-- Theorem stating that the number of connected subsets is 81 -/
theorem num_connected_subsets_2x1 :
  ∀ r : Rectangle,
  r.bottomLeft = (0, 0) ∧ r.topRight = (2, 1) →
  num_connected_subsets r = 81 :=
sorry

end NUMINAMATH_CALUDE_num_connected_subsets_2x1_l2638_263842


namespace NUMINAMATH_CALUDE_train_crossing_time_l2638_263869

/-- Proves that a train 360 m long traveling at 43.2 km/h takes 30 seconds to cross an electric pole. -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 360 →
  train_speed_kmh = 43.2 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 30 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l2638_263869


namespace NUMINAMATH_CALUDE_consecutive_numbers_sum_l2638_263821

theorem consecutive_numbers_sum (n : ℕ) : 
  n + (n + 1) + (n + 2) = 60 → (n + 2) + (n + 3) + (n + 4) = 66 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_numbers_sum_l2638_263821


namespace NUMINAMATH_CALUDE_detergent_needed_l2638_263848

/-- The amount of detergent used per pound of clothes -/
def detergent_per_pound : ℝ := 2

/-- The amount of clothes to be washed, in pounds -/
def clothes_amount : ℝ := 9

/-- Theorem stating the amount of detergent needed for a given amount of clothes -/
theorem detergent_needed (detergent_per_pound : ℝ) (clothes_amount : ℝ) :
  detergent_per_pound * clothes_amount = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_detergent_needed_l2638_263848


namespace NUMINAMATH_CALUDE_min_set_size_l2638_263891

theorem min_set_size (n : ℕ) : 
  let set_size := 2 * n + 1
  let median := 10
  let arithmetic_mean := 6
  let sum := arithmetic_mean * set_size
  let lower_bound := n * 1 + (n + 1) * 10
  sum ≥ lower_bound → n ≥ 4 :=
by
  sorry

end NUMINAMATH_CALUDE_min_set_size_l2638_263891


namespace NUMINAMATH_CALUDE_vector_equality_l2638_263877

def a : ℝ × ℝ := (4, 2)
def b (k : ℝ) : ℝ × ℝ := (2 - k, k - 1)

theorem vector_equality (k : ℝ) :
  ‖a + b k‖ = ‖a - b k‖ → k = 3 := by sorry

end NUMINAMATH_CALUDE_vector_equality_l2638_263877


namespace NUMINAMATH_CALUDE_greatest_integer_less_than_negative_seventeen_thirds_l2638_263871

theorem greatest_integer_less_than_negative_seventeen_thirds :
  ⌊-17/3⌋ = -6 :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_less_than_negative_seventeen_thirds_l2638_263871


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2638_263854

theorem arithmetic_calculation : 8 / 2 + (-3) * 4 - (-10) + 6 * (-2) = -10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2638_263854


namespace NUMINAMATH_CALUDE_gcd_2475_7350_l2638_263800

theorem gcd_2475_7350 : Nat.gcd 2475 7350 = 225 := by sorry

end NUMINAMATH_CALUDE_gcd_2475_7350_l2638_263800


namespace NUMINAMATH_CALUDE_cuboid_height_l2638_263812

theorem cuboid_height (volume : ℝ) (base_area : ℝ) (height : ℝ) 
  (h1 : volume = 144)
  (h2 : base_area = 18)
  (h3 : volume = base_area * height) :
  height = 8 := by
sorry

end NUMINAMATH_CALUDE_cuboid_height_l2638_263812


namespace NUMINAMATH_CALUDE_cyclical_fraction_bounds_l2638_263828

theorem cyclical_fraction_bounds (a b c d e : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) : 
  1 < (a / (a + b) + b / (b + c) + c / (c + d) + d / (d + e) + e / (e + a)) ∧ 
  (a / (a + b) + b / (b + c) + c / (c + d) + d / (d + e) + e / (e + a)) < 4 := by
  sorry

end NUMINAMATH_CALUDE_cyclical_fraction_bounds_l2638_263828


namespace NUMINAMATH_CALUDE_calculation_proof_l2638_263818

theorem calculation_proof : (10^8 / (2 * 10^5)) - 50 = 450 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2638_263818


namespace NUMINAMATH_CALUDE_children_in_group_l2638_263824

/-- Calculates the number of children in a restaurant group given the total bill,
    number of adults, and cost per meal. -/
def number_of_children (total_bill : ℕ) (num_adults : ℕ) (cost_per_meal : ℕ) : ℕ :=
  (total_bill - num_adults * cost_per_meal) / cost_per_meal

/-- Proves that the number of children in the group is 5. -/
theorem children_in_group : number_of_children 21 2 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_children_in_group_l2638_263824


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2638_263872

theorem geometric_sequence_property (a : ℕ → ℝ) (h_positive : ∀ n, a n > 0) 
  (h_geometric : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h_arithmetic : a 1 - (1/2 * a 3) = (1/2 * a 3) - (2 * a 2)) :
  (a 8 + a 9) / (a 6 + a 7) = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2638_263872


namespace NUMINAMATH_CALUDE_exponential_equation_solution_l2638_263831

theorem exponential_equation_solution :
  ∃ y : ℝ, (3 : ℝ) ^ (y - 4) = 9 ^ (y + 2) → y = -8 := by
  sorry

end NUMINAMATH_CALUDE_exponential_equation_solution_l2638_263831


namespace NUMINAMATH_CALUDE_rhombus_diagonal_length_l2638_263817

/-- Given a rhombus with an area equal to that of a square with side length 8,
    and one diagonal of length 8, prove that the other diagonal has length 16. -/
theorem rhombus_diagonal_length :
  ∀ (d1 : ℝ),
  (d1 * 8 / 2 = 8 * 8) →
  d1 = 16 := by
sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_length_l2638_263817


namespace NUMINAMATH_CALUDE_maggies_age_l2638_263881

theorem maggies_age (kate_age sue_age maggie_age : ℕ) 
  (total_age : kate_age + sue_age + maggie_age = 48)
  (kate : kate_age = 19)
  (sue : sue_age = 12) :
  maggie_age = 17 := by
  sorry

end NUMINAMATH_CALUDE_maggies_age_l2638_263881


namespace NUMINAMATH_CALUDE_least_difference_consecutive_primes_l2638_263896

theorem least_difference_consecutive_primes (x y z : ℕ) : 
  Prime x ∧ Prime y ∧ Prime z ∧  -- x, y, and z are prime numbers
  x < y ∧ y < z ∧                -- x < y < z
  y - x > 3 ∧                    -- y - x > 3
  Even x ∧                       -- x is an even integer
  Odd y ∧ Odd z →                -- y and z are odd integers
  ∀ w, (Prime w ∧ Prime (w + 1) ∧ Prime (w + 2) ∧ 
        w < w + 1 ∧ w + 1 < w + 2 ∧
        (w + 1) - w > 3 ∧
        Even w ∧ Odd (w + 1) ∧ Odd (w + 2)) →
    (w + 2) - w ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_least_difference_consecutive_primes_l2638_263896


namespace NUMINAMATH_CALUDE_probability_of_red_ball_l2638_263846

-- Define the total number of balls
def total_balls : ℕ := 7

-- Define the number of red balls
def red_balls : ℕ := 2

-- Define the number of black balls
def black_balls : ℕ := 4

-- Define the number of white balls
def white_balls : ℕ := 1

-- Define the probability of drawing a red ball
def prob_red_ball : ℚ := red_balls / total_balls

-- Theorem statement
theorem probability_of_red_ball :
  prob_red_ball = 2 / 7 := by sorry

end NUMINAMATH_CALUDE_probability_of_red_ball_l2638_263846


namespace NUMINAMATH_CALUDE_ramanujan_number_l2638_263876

theorem ramanujan_number (h r : ℂ) : 
  h * r = 40 - 24 * I ∧ h = 4 + 4 * I → r = 2 - 8 * I :=
by sorry

end NUMINAMATH_CALUDE_ramanujan_number_l2638_263876


namespace NUMINAMATH_CALUDE_cat_care_cost_is_40_l2638_263838

/-- The cost to care for a cat at Mr. Sean's veterinary clinic -/
def cat_care_cost : ℕ → Prop
| cost => ∃ (dog_cost : ℕ),
  dog_cost = 60 ∧
  20 * dog_cost + 60 * cost = 3600

/-- Theorem: The cost to care for a cat at Mr. Sean's clinic is $40 -/
theorem cat_care_cost_is_40 : cat_care_cost 40 := by
  sorry

end NUMINAMATH_CALUDE_cat_care_cost_is_40_l2638_263838


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l2638_263844

/-- The number of distinct diagonals in a convex nonagon -/
def diagonals_in_nonagon : ℕ := 27

/-- A convex nonagon has 27 distinct diagonals -/
theorem nonagon_diagonals :
  diagonals_in_nonagon = 27 := by sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l2638_263844


namespace NUMINAMATH_CALUDE_gcf_factorial_seven_eight_l2638_263898

theorem gcf_factorial_seven_eight : Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = Nat.factorial 7 := by
  sorry

end NUMINAMATH_CALUDE_gcf_factorial_seven_eight_l2638_263898


namespace NUMINAMATH_CALUDE_hindi_books_count_l2638_263815

def number_of_arrangements (n m : ℕ) : ℕ := Nat.choose (n + 1) m

theorem hindi_books_count : ∃ h : ℕ, 
  number_of_arrangements 22 h = 1771 ∧ h = 3 :=
by sorry

end NUMINAMATH_CALUDE_hindi_books_count_l2638_263815


namespace NUMINAMATH_CALUDE_probability_smaller_divides_larger_l2638_263802

def S : Finset ℕ := {1, 2, 3, 6, 9}

def divides_pairs (S : Finset ℕ) : Finset (ℕ × ℕ) :=
  S.product S |>.filter (fun p => p.1 < p.2 ∧ p.2 % p.1 = 0)

theorem probability_smaller_divides_larger :
  (divides_pairs S).card / (S.product S |>.filter (fun p => p.1 ≠ p.2)).card = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_smaller_divides_larger_l2638_263802


namespace NUMINAMATH_CALUDE_f_value_theorem_l2638_263853

def is_prime (p : ℕ) : Prop := ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

def f_property (f : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 1 → ∃ p : ℕ, is_prime p ∧ p ∣ n ∧ f n = f (n / p) - f p

theorem f_value_theorem (f : ℕ → ℝ) (h1 : f_property f) 
  (h2 : f (2^2007) + f (3^2008) + f (5^2009) = 2006) :
  f (2007^2) + f (2008^3) + f (2009^5) = 9 := by
  sorry

end NUMINAMATH_CALUDE_f_value_theorem_l2638_263853


namespace NUMINAMATH_CALUDE_cristine_lemons_left_l2638_263849

def dozen : ℕ := 12

def lemons_given_to_neighbor (total : ℕ) : ℕ := total / 4

def lemons_exchanged_for_oranges : ℕ := 2

theorem cristine_lemons_left (initial_lemons : ℕ) 
  (h1 : initial_lemons = dozen) 
  (h2 : lemons_given_to_neighbor initial_lemons = initial_lemons / 4) 
  (h3 : lemons_exchanged_for_oranges = 2) : 
  initial_lemons - lemons_given_to_neighbor initial_lemons - lemons_exchanged_for_oranges = 7 := by
  sorry

end NUMINAMATH_CALUDE_cristine_lemons_left_l2638_263849


namespace NUMINAMATH_CALUDE_tommys_estimate_l2638_263804

theorem tommys_estimate (x y ε : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : ε > 0) :
  (x + ε) - (y - ε) > x - y := by
  sorry

end NUMINAMATH_CALUDE_tommys_estimate_l2638_263804


namespace NUMINAMATH_CALUDE_sqrt_360000_l2638_263833

theorem sqrt_360000 : Real.sqrt 360000 = 600 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_360000_l2638_263833


namespace NUMINAMATH_CALUDE_church_rows_count_l2638_263878

/-- Represents the seating arrangement in a church --/
structure ChurchSeating where
  chairs_per_row : ℕ
  people_per_chair : ℕ
  total_people : ℕ

/-- Calculates the number of rows in the church --/
def number_of_rows (s : ChurchSeating) : ℕ :=
  s.total_people / (s.chairs_per_row * s.people_per_chair)

/-- Theorem stating the number of rows in the church --/
theorem church_rows_count (s : ChurchSeating) 
  (h1 : s.chairs_per_row = 6)
  (h2 : s.people_per_chair = 5)
  (h3 : s.total_people = 600) :
  number_of_rows s = 20 := by
  sorry

#eval number_of_rows ⟨6, 5, 600⟩

end NUMINAMATH_CALUDE_church_rows_count_l2638_263878


namespace NUMINAMATH_CALUDE_exists_permutation_distinct_columns_l2638_263897

/-- A table is represented as a function from pairs of indices to integers -/
def Table (n : ℕ) := Fin n → Fin n → ℤ

/-- A predicate stating that no two cells within a row share the same number -/
def DistinctInRows (t : Table n) : Prop :=
  ∀ i j₁ j₂, j₁ ≠ j₂ → t i j₁ ≠ t i j₂

/-- A permutation of a row is a bijection on Fin n -/
def RowPermutation (n : ℕ) := Fin n ≃ Fin n

/-- Apply a row permutation to a table -/
def ApplyRowPermutation (t : Table n) (p : Fin n → RowPermutation n) : Table n :=
  λ i j ↦ t i ((p i).toFun j)

/-- A predicate stating that all columns contain distinct numbers -/
def DistinctInColumns (t : Table n) : Prop :=
  ∀ j i₁ i₂, i₁ ≠ i₂ → t i₁ j ≠ t i₂ j

/-- The main theorem -/
theorem exists_permutation_distinct_columns (n : ℕ) (t : Table n) 
    (h : DistinctInRows t) : 
    ∃ p : Fin n → RowPermutation n, DistinctInColumns (ApplyRowPermutation t p) := by
  sorry

end NUMINAMATH_CALUDE_exists_permutation_distinct_columns_l2638_263897


namespace NUMINAMATH_CALUDE_christmas_tree_perimeter_l2638_263895

/-- A Christmas tree is a geometric shape with the following properties:
  1. It is symmetric about the y-axis
  2. It has a height of 1
  3. Its branches form a 45° angle with the vertical
  4. It consists of isosceles right triangles
-/
structure ChristmasTree where
  height : ℝ
  branchAngle : ℝ
  isSymmetric : Bool

/-- The perimeter of a Christmas tree is the sum of all its branch lengths -/
def perimeter (tree : ChristmasTree) : ℝ :=
  sorry

/-- The main theorem stating that the perimeter of a Christmas tree
    with the given properties is 2(1 + √2) -/
theorem christmas_tree_perimeter :
  ∀ (tree : ChristmasTree),
  tree.height = 1 ∧ tree.branchAngle = π/4 ∧ tree.isSymmetric = true →
  perimeter tree = 2 * (1 + Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_christmas_tree_perimeter_l2638_263895


namespace NUMINAMATH_CALUDE_max_product_sum_l2638_263855

def values : Finset ℕ := {1, 3, 5, 7}

theorem max_product_sum (a b c d : ℕ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_in_values : a ∈ values ∧ b ∈ values ∧ c ∈ values ∧ d ∈ values) :
  (a * b + b * c + c * d + d * a) ≤ 64 :=
sorry

end NUMINAMATH_CALUDE_max_product_sum_l2638_263855


namespace NUMINAMATH_CALUDE_sum_of_roots_is_zero_l2638_263870

theorem sum_of_roots_is_zero (x : ℝ) :
  (x^2 - 7*|x| + 6 = 0) →
  (∃ x₁ x₂ x₃ x₄ : ℝ,
    ((x₁ > 0 ∧ x₂ > 0 ∧ x₁^2 - 7*x₁ + 6 = 0 ∧ x₂^2 - 7*x₂ + 6 = 0) ∨
     (x₃ < 0 ∧ x₄ < 0 ∧ x₃^2 + 7*x₃ + 6 = 0 ∧ x₄^2 + 7*x₄ + 6 = 0)) ∧
    x₁ + x₂ + x₃ + x₄ = 0) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_is_zero_l2638_263870


namespace NUMINAMATH_CALUDE_cubic_equation_roots_inequality_l2638_263887

/-- Given a cubic equation x³ + ax² + bx + c = 0 with three real roots p ≤ q ≤ r,
    prove that a² - 3b ≥ 0 and √(a² - 3b) ≤ r - p -/
theorem cubic_equation_roots_inequality (a b c p q r : ℝ) :
  p ≤ q ∧ q ≤ r ∧
  p^3 + a*p^2 + b*p + c = 0 ∧
  q^3 + a*q^2 + b*q + c = 0 ∧
  r^3 + a*r^2 + b*r + c = 0 →
  a^2 - 3*b ≥ 0 ∧ Real.sqrt (a^2 - 3*b) ≤ r - p :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_inequality_l2638_263887


namespace NUMINAMATH_CALUDE_fraction_simplification_l2638_263893

theorem fraction_simplification : (8 : ℚ) / (5 * 42) = 4 / 105 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2638_263893


namespace NUMINAMATH_CALUDE_distance_between_complex_points_l2638_263858

theorem distance_between_complex_points :
  let z₁ : ℂ := 3 + 4*I
  let z₂ : ℂ := -2 + 3*I
  Complex.abs (z₁ - z₂) = Real.sqrt 26 := by sorry

end NUMINAMATH_CALUDE_distance_between_complex_points_l2638_263858


namespace NUMINAMATH_CALUDE_five_hour_study_score_l2638_263805

/-- Represents a student's test score based on study time -/
structure TestScore where
  studyTime : ℝ
  score : ℝ

/-- The maximum possible score on a test -/
def maxScore : ℝ := 100

/-- Calculates the potential score based on study time and effectiveness -/
def potentialScore (effectiveness : ℝ) (studyTime : ℝ) : ℝ :=
  effectiveness * studyTime

/-- Theorem: Given the conditions, the score for 5 hours of study is 100 -/
theorem five_hour_study_score :
  ∀ (effectiveness : ℝ),
  effectiveness > 0 →
  potentialScore effectiveness 2 = 80 →
  min (potentialScore effectiveness 5) maxScore = 100 := by
sorry

end NUMINAMATH_CALUDE_five_hour_study_score_l2638_263805


namespace NUMINAMATH_CALUDE_employees_difference_l2638_263860

/-- Given a company with total employees and employees in Korea, 
    prove the difference between employees in Korea and abroad. -/
theorem employees_difference (total : ℕ) (in_korea : ℕ) 
    (h1 : total = 928) (h2 : in_korea = 713) : 
    in_korea - (total - in_korea) = 498 := by
  sorry

end NUMINAMATH_CALUDE_employees_difference_l2638_263860


namespace NUMINAMATH_CALUDE_trig_special_angles_sum_l2638_263832

theorem trig_special_angles_sum : 
  Real.sin (π / 2) + 2 * Real.cos 0 - 3 * Real.sin (3 * π / 2) + 10 * Real.cos π = -4 := by
  sorry

end NUMINAMATH_CALUDE_trig_special_angles_sum_l2638_263832


namespace NUMINAMATH_CALUDE_discriminant_sufficient_not_necessary_l2638_263827

-- Define the quadratic equation
def quadratic (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- Define the discriminant
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

-- Define what it means for the equation to have real roots
def has_real_roots (a b c : ℝ) : Prop :=
  ∃ x : ℝ, quadratic a b c x

-- Theorem statement
theorem discriminant_sufficient_not_necessary
  (a b c : ℝ) (ha : a ≠ 0) :
  (discriminant a b c > 0 → has_real_roots a b c) ∧
  ∃ a' b' c' : ℝ, a' ≠ 0 ∧ has_real_roots a' b' c' ∧ discriminant a' b' c' ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_discriminant_sufficient_not_necessary_l2638_263827


namespace NUMINAMATH_CALUDE_candy_necklace_solution_l2638_263826

def candy_necklace_problem (pieces_per_necklace : ℕ) (pieces_per_block : ℕ) (num_friends : ℕ) : ℕ :=
  let total_pieces := pieces_per_necklace * (num_friends + 1)
  (total_pieces + pieces_per_block - 1) / pieces_per_block

theorem candy_necklace_solution :
  candy_necklace_problem 10 30 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_candy_necklace_solution_l2638_263826


namespace NUMINAMATH_CALUDE_smallest_cubic_divisible_by_810_l2638_263875

theorem smallest_cubic_divisible_by_810 : ∃ (a : ℕ), 
  (∀ (n : ℕ), n < a → ¬(∃ (k : ℕ), n = k^3 ∧ 810 ∣ n)) ∧
  (∃ (k : ℕ), a = k^3) ∧ 
  (810 ∣ a) ∧
  a = 729000 := by
  sorry

end NUMINAMATH_CALUDE_smallest_cubic_divisible_by_810_l2638_263875


namespace NUMINAMATH_CALUDE_sqrt_equation_difference_l2638_263808

theorem sqrt_equation_difference (a b : ℕ+) 
  (h1 : Real.sqrt 18 = (a : ℝ) * Real.sqrt 2) 
  (h2 : Real.sqrt 8 = 2 * Real.sqrt (b : ℝ)) : 
  (a : ℤ) - (b : ℤ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_difference_l2638_263808


namespace NUMINAMATH_CALUDE_initial_lot_cost_l2638_263803

/-- Represents the cost and composition of a lot of tickets -/
structure TicketLot where
  firstClass : ℕ
  secondClass : ℕ
  firstClassCost : ℕ
  secondClassCost : ℕ

/-- Calculates the total cost of a ticket lot -/
def totalCost (lot : TicketLot) : ℕ :=
  lot.firstClass * lot.firstClassCost + lot.secondClass * lot.secondClassCost

/-- Theorem: The cost of the initial lot of tickets is 110 Rs -/
theorem initial_lot_cost (initialLot interchangedLot : TicketLot) : 
  initialLot.firstClass + initialLot.secondClass = 18 →
  initialLot.firstClassCost = 10 →
  initialLot.secondClassCost = 3 →
  interchangedLot.firstClass = initialLot.secondClass →
  interchangedLot.secondClass = initialLot.firstClass →
  interchangedLot.firstClassCost = initialLot.firstClassCost →
  interchangedLot.secondClassCost = initialLot.secondClassCost →
  totalCost interchangedLot = 124 →
  totalCost initialLot = 110 := by
  sorry

end NUMINAMATH_CALUDE_initial_lot_cost_l2638_263803


namespace NUMINAMATH_CALUDE_joan_lost_balloons_l2638_263867

theorem joan_lost_balloons (initial_balloons current_balloons : ℕ) 
  (h1 : initial_balloons = 9)
  (h2 : current_balloons = 7) : 
  initial_balloons - current_balloons = 2 := by
  sorry

end NUMINAMATH_CALUDE_joan_lost_balloons_l2638_263867


namespace NUMINAMATH_CALUDE_puppies_brought_in_solution_l2638_263825

/-- The number of puppies brought to a pet shelter -/
def puppies_brought_in (initial_puppies : ℕ) (adoption_rate : ℕ) (days_to_adopt : ℕ) : ℕ :=
  adoption_rate * days_to_adopt - initial_puppies

/-- Theorem stating that 12 puppies were brought in given the problem conditions -/
theorem puppies_brought_in_solution :
  puppies_brought_in 9 3 7 = 12 := by
  sorry

end NUMINAMATH_CALUDE_puppies_brought_in_solution_l2638_263825


namespace NUMINAMATH_CALUDE_sqrt_32_div_sqrt_2_equals_4_l2638_263820

theorem sqrt_32_div_sqrt_2_equals_4 : Real.sqrt 32 / Real.sqrt 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_32_div_sqrt_2_equals_4_l2638_263820
