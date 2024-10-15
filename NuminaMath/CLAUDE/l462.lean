import Mathlib

namespace NUMINAMATH_CALUDE_problem_solution_l462_46274

/-- The function f(x) = x^2 - 1 --/
def f (x : ℝ) : ℝ := x^2 - 1

/-- The function g(x) = a|x-1| --/
def g (a x : ℝ) : ℝ := a * |x - 1|

/-- The function h(x) = |f(x)| + g(x) --/
def h (a x : ℝ) : ℝ := |f x| + g a x

theorem problem_solution (a : ℝ) :
  (∀ x : ℝ, f x ≥ g a x) →
  (a ≤ -2 ∧
   (∀ x ∈ Set.Icc 0 1,
      (a ≥ -3 → h a x ≤ a + 3) ∧
      (a < -3 → h a x ≤ 0))) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l462_46274


namespace NUMINAMATH_CALUDE_a_less_than_two_thirds_l462_46227

-- Define a decreasing function
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- State the theorem
theorem a_less_than_two_thirds
  (f : ℝ → ℝ) (a : ℝ)
  (h_decreasing : DecreasingFunction f)
  (h_inequality : f (1 - a) < f (2 * a - 1)) :
  a < 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_a_less_than_two_thirds_l462_46227


namespace NUMINAMATH_CALUDE_javier_first_throw_distance_l462_46222

/-- Represents the distances of Javier's three javelin throws -/
structure JavelinThrows where
  first : ℝ
  second : ℝ
  third : ℝ

/-- Theorem stating the distance of Javier's first throw given the conditions -/
theorem javier_first_throw_distance (throws : JavelinThrows) :
  throws.first = 2 * throws.second ∧
  throws.first = throws.third / 2 ∧
  throws.first + throws.second + throws.third = 1050 →
  throws.first = 300 := by
  sorry

end NUMINAMATH_CALUDE_javier_first_throw_distance_l462_46222


namespace NUMINAMATH_CALUDE_polynomial_coefficient_product_l462_46258

/-- Given a polynomial x^4 - (a-2)x^3 + 5x^2 + (b+3)x - 1 where the coefficients of x^3 and x are zero, prove that ab = -6 -/
theorem polynomial_coefficient_product (a b : ℝ) : 
  (a - 2 = 0) → (b + 3 = 0) → a * b = -6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_product_l462_46258


namespace NUMINAMATH_CALUDE_jennifer_sweets_sharing_l462_46298

theorem jennifer_sweets_sharing (total_sweets : ℕ) (sweets_per_person : ℕ) (h1 : total_sweets = 1024) (h2 : sweets_per_person = 256) :
  (total_sweets / sweets_per_person) - 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_jennifer_sweets_sharing_l462_46298


namespace NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l462_46271

theorem smallest_part_of_proportional_division (total : ℝ) (a b c d : ℝ) 
  (h_total : total = 80)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)
  (h_prop : b = 3 * a ∧ c = 5 * a ∧ d = 7 * a)
  (h_sum : a + b + c + d = total) :
  a = 5 := by
sorry

end NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l462_46271


namespace NUMINAMATH_CALUDE_simplify_expression_l462_46247

theorem simplify_expression (x : ℝ) : x^2 * x^4 + x * x^2 * x^3 = 2 * x^6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l462_46247


namespace NUMINAMATH_CALUDE_function_proof_l462_46262

theorem function_proof (f : ℤ → ℤ) (h1 : f 0 = 1) (h2 : f 2012 = 2013) :
  ∀ n : ℤ, f n = n + 1 := by
  sorry

end NUMINAMATH_CALUDE_function_proof_l462_46262


namespace NUMINAMATH_CALUDE_other_x_intercept_of_quadratic_l462_46215

/-- Given a quadratic function with vertex (5, 12) and one x-intercept at (1, 0),
    the x-coordinate of the other x-intercept is 9. -/
theorem other_x_intercept_of_quadratic (a b c : ℝ) :
  (∀ x, a * x^2 + b * x + c = 12 - a * (x - 5)^2) →  -- vertex form
  a * 1^2 + b * 1 + c = 0 →                          -- x-intercept at (1, 0)
  ∃ x, x ≠ 1 ∧ a * x^2 + b * x + c = 0 ∧ x = 9 :=    -- other x-intercept at 9
by sorry

end NUMINAMATH_CALUDE_other_x_intercept_of_quadratic_l462_46215


namespace NUMINAMATH_CALUDE_geometric_sequence_product_constant_geometric_sequence_product_l462_46273

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The product of terms equidistant from the beginning and end is constant -/
theorem geometric_sequence_product_constant {a : ℕ → ℝ} (h : geometric_sequence a) :
  ∀ n k : ℕ, a n * a (k + 1 - n) = a 1 * a k :=
sorry

/-- Main theorem: If a₃a₄ = 2 in a geometric sequence, then a₁a₂a₃a₄a₅a₆ = 8 -/
theorem geometric_sequence_product (a : ℕ → ℝ) (h : geometric_sequence a) 
  (h2 : a 3 * a 4 = 2) : a 1 * a 2 * a 3 * a 4 * a 5 * a 6 = 8 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_constant_geometric_sequence_product_l462_46273


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l462_46239

theorem diophantine_equation_solution (k ℓ : ℤ) :
  5 * k + 3 * ℓ = 32 ↔ ∃ x : ℤ, k = -32 + 3 * x ∧ ℓ = 64 - 5 * x :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l462_46239


namespace NUMINAMATH_CALUDE_polynomial_coefficient_difference_l462_46263

theorem polynomial_coefficient_difference (m n : ℝ) : 
  (∀ x : ℝ, 3 * x * (x - 1) = m * x^2 + n * x) → m - n = 6 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_difference_l462_46263


namespace NUMINAMATH_CALUDE_exists_n_sigma_gt_3n_forall_k_exists_n_sigma_gt_kn_l462_46241

-- Define the sum of divisors function
def sigma (n : ℕ) : ℕ := sorry

-- Theorem 1: There exists a positive integer n such that σ(n) > 3n
theorem exists_n_sigma_gt_3n : ∃ n : ℕ, n > 0 ∧ sigma n > 3 * n := by sorry

-- Theorem 2: For any real number k > 1, there exists a positive integer n such that σ(n) > kn
theorem forall_k_exists_n_sigma_gt_kn : ∀ k : ℝ, k > 1 → ∃ n : ℕ, n > 0 ∧ (sigma n : ℝ) > k * n := by sorry

end NUMINAMATH_CALUDE_exists_n_sigma_gt_3n_forall_k_exists_n_sigma_gt_kn_l462_46241


namespace NUMINAMATH_CALUDE_problem_solution_l462_46254

def f (n : ℕ) : ℚ := (n^2 - 5*n + 4) / (n - 4)

theorem problem_solution :
  (f 1 = 0) ∧
  (∀ n : ℕ, n ≠ 4 → (f n = 5 ↔ n = 6)) ∧
  (∀ n : ℕ, n ≠ 4 → f n ≠ 3) := by sorry

end NUMINAMATH_CALUDE_problem_solution_l462_46254


namespace NUMINAMATH_CALUDE_gravel_path_cost_l462_46292

/-- Calculates the cost of gravelling a path inside a rectangular plot. -/
theorem gravel_path_cost
  (plot_length : ℝ)
  (plot_width : ℝ)
  (path_width : ℝ)
  (cost_per_sqm : ℝ)
  (h1 : plot_length = 110)
  (h2 : plot_width = 65)
  (h3 : path_width = 2.5)
  (h4 : cost_per_sqm = 0.70) :
  let total_area := plot_length * plot_width
  let inner_length := plot_length - 2 * path_width
  let inner_width := plot_width - 2 * path_width
  let inner_area := inner_length * inner_width
  let path_area := total_area - inner_area
  path_area * cost_per_sqm = 595 :=
by sorry

end NUMINAMATH_CALUDE_gravel_path_cost_l462_46292


namespace NUMINAMATH_CALUDE_certain_number_proof_l462_46270

theorem certain_number_proof : ∃ x : ℝ, 0.45 * 60 = 0.35 * x + 13 ∧ x = 40 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l462_46270


namespace NUMINAMATH_CALUDE_cow_characteristic_difference_l462_46231

def total_cows : ℕ := 600
def male_ratio : ℕ := 5
def female_ratio : ℕ := 3
def transgender_ratio : ℕ := 2

def male_horned_percentage : ℚ := 50 / 100
def male_spotted_percentage : ℚ := 40 / 100
def male_brown_percentage : ℚ := 20 / 100

def female_spotted_percentage : ℚ := 35 / 100
def female_horned_percentage : ℚ := 25 / 100
def female_white_percentage : ℚ := 60 / 100

def transgender_unique_pattern_percentage : ℚ := 45 / 100
def transgender_spotted_horned_percentage : ℚ := 30 / 100
def transgender_black_percentage : ℚ := 50 / 100

theorem cow_characteristic_difference :
  let total_ratio := male_ratio + female_ratio + transgender_ratio
  let male_count := (male_ratio : ℚ) / total_ratio * total_cows
  let female_count := (female_ratio : ℚ) / total_ratio * total_cows
  let transgender_count := (transgender_ratio : ℚ) / total_ratio * total_cows
  let spotted_females := female_spotted_percentage * female_count
  let horned_males := male_horned_percentage * male_count
  let brown_males := male_brown_percentage * male_count
  let unique_pattern_transgender := transgender_unique_pattern_percentage * transgender_count
  let white_horned_females := female_horned_percentage * female_white_percentage * female_count
  let characteristic_sum := horned_males + brown_males + unique_pattern_transgender + white_horned_females
  spotted_females - characteristic_sum = -291 := by sorry

end NUMINAMATH_CALUDE_cow_characteristic_difference_l462_46231


namespace NUMINAMATH_CALUDE_part_one_part_two_l462_46200

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a| + |x + 5 - a|

-- Part 1
theorem part_one (a : ℝ) : 
  (∀ x, f x a - |x - a| ≤ 2 ↔ x ∈ Set.Icc (-5) (-1)) → a = 2 := by sorry

-- Part 2
theorem part_two (m : ℝ) :
  (∃ x₀ : ℝ, f x₀ 2 < 4*m + m^2) → m < -5 ∨ m > 1 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l462_46200


namespace NUMINAMATH_CALUDE_alkaline_probability_is_two_fifths_l462_46228

/-- Represents the total number of solutions -/
def total_solutions : ℕ := 5

/-- Represents the number of alkaline solutions -/
def alkaline_solutions : ℕ := 2

/-- Represents the probability of selecting an alkaline solution -/
def alkaline_probability : ℚ := alkaline_solutions / total_solutions

/-- Theorem stating that the probability of selecting an alkaline solution is 2/5 -/
theorem alkaline_probability_is_two_fifths : 
  alkaline_probability = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_alkaline_probability_is_two_fifths_l462_46228


namespace NUMINAMATH_CALUDE_maximum_garden_area_l462_46257

/-- Represents the dimensions of a rectangular garden -/
structure GardenDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular garden -/
def gardenArea (d : GardenDimensions) : ℝ := d.length * d.width

/-- Calculates the fence length required for three sides of a rectangular garden -/
def fenceLength (d : GardenDimensions) : ℝ := d.length + 2 * d.width

/-- The total available fencing -/
def totalFence : ℝ := 400

theorem maximum_garden_area :
  ∃ (d : GardenDimensions),
    fenceLength d = totalFence ∧
    ∀ (d' : GardenDimensions), fenceLength d' = totalFence → gardenArea d' ≤ gardenArea d ∧
    gardenArea d = 20000 := by
  sorry

end NUMINAMATH_CALUDE_maximum_garden_area_l462_46257


namespace NUMINAMATH_CALUDE_bounded_area_calculation_l462_46216

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Calculates the area of the region bounded by two circles and two vertical lines -/
def boundedArea (c1 c2 : Circle) (x1 x2 : ℝ) : ℝ :=
  sorry

theorem bounded_area_calculation :
  let c1 : Circle := { center := (4, 4), radius := 4 }
  let c2 : Circle := { center := (12, 12), radius := 4 }
  let x1 : ℝ := 4
  let x2 : ℝ := 12
  boundedArea c1 c2 x1 x2 = 64 - 8 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_bounded_area_calculation_l462_46216


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l462_46261

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_of_M_and_N : M ∩ N = {2, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l462_46261


namespace NUMINAMATH_CALUDE_complex_combination_equality_l462_46237

/-- Given complex numbers Q, E, D, and F, prove that their combination equals 1 + 117i -/
theorem complex_combination_equality (Q E D F : ℂ) 
  (hQ : Q = 7 + 3*I) 
  (hE : E = 2*I) 
  (hD : D = 7 - 3*I) 
  (hF : F = 1 + I) : 
  (Q * E * D) + F = 1 + 117*I := by
  sorry

end NUMINAMATH_CALUDE_complex_combination_equality_l462_46237


namespace NUMINAMATH_CALUDE_new_consumption_per_soldier_l462_46255

/-- Calculates the new daily consumption per soldier after additional soldiers join a fort, given the initial conditions and the number of new soldiers. -/
theorem new_consumption_per_soldier
  (initial_soldiers : ℕ)
  (initial_consumption : ℚ)
  (initial_duration : ℕ)
  (new_duration : ℕ)
  (new_soldiers : ℕ)
  (h_initial_soldiers : initial_soldiers = 1200)
  (h_initial_consumption : initial_consumption = 3)
  (h_initial_duration : initial_duration = 30)
  (h_new_duration : new_duration = 25)
  (h_new_soldiers : new_soldiers = 528) :
  let total_provisions := initial_soldiers * initial_consumption * initial_duration
  let total_soldiers := initial_soldiers + new_soldiers
  total_provisions / (total_soldiers * new_duration) = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_new_consumption_per_soldier_l462_46255


namespace NUMINAMATH_CALUDE_not_divisible_by_seven_l462_46285

theorem not_divisible_by_seven (k : ℕ) : ¬(7 ∣ (2^(2*k - 1) + 2^k + 1)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_seven_l462_46285


namespace NUMINAMATH_CALUDE_inequality_solution_range_l462_46244

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, |x + 1| - |x - 2| < a^2 - 4*a) → (a > 3 ∨ a < 1) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l462_46244


namespace NUMINAMATH_CALUDE_ellipse_m_values_l462_46279

def ellipse_equation (x y m : ℝ) : Prop :=
  x^2 / 12 + y^2 / m = 1

def eccentricity (e : ℝ) : Prop :=
  e = 1/2

theorem ellipse_m_values (m : ℝ) :
  (∃ x y, ellipse_equation x y m) ∧ (∃ e, eccentricity e) →
  m = 9 ∨ m = 16 := by sorry

end NUMINAMATH_CALUDE_ellipse_m_values_l462_46279


namespace NUMINAMATH_CALUDE_five_balls_four_boxes_l462_46269

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 56 ways to distribute 5 indistinguishable balls into 4 distinguishable boxes -/
theorem five_balls_four_boxes : distribute_balls 5 4 = 56 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_four_boxes_l462_46269


namespace NUMINAMATH_CALUDE_daves_tiling_area_l462_46236

theorem daves_tiling_area (total_area : ℝ) (clara_ratio : ℕ) (dave_ratio : ℕ) 
  (h1 : total_area = 330)
  (h2 : clara_ratio = 4)
  (h3 : dave_ratio = 7) : 
  (dave_ratio : ℝ) / ((clara_ratio : ℝ) + (dave_ratio : ℝ)) * total_area = 210 :=
by sorry

end NUMINAMATH_CALUDE_daves_tiling_area_l462_46236


namespace NUMINAMATH_CALUDE_sole_mart_meals_l462_46299

theorem sole_mart_meals (initial_meals : ℕ) (given_away : ℕ) (left : ℕ) 
  (h1 : initial_meals = 113)
  (h2 : given_away = 85)
  (h3 : left = 78) :
  initial_meals + (given_away + left) - initial_meals = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_sole_mart_meals_l462_46299


namespace NUMINAMATH_CALUDE_monitor_horizontal_length_l462_46211

/-- Given a rectangle with a 16:9 aspect ratio and a diagonal of 32 inches,
    prove that the horizontal length is (16 * 32) / sqrt(337) --/
theorem monitor_horizontal_length (h w d : ℝ) : 
  h / w = 9 / 16 → 
  h^2 + w^2 = d^2 → 
  d = 32 → 
  w = (16 * 32) / Real.sqrt 337 := by
  sorry

end NUMINAMATH_CALUDE_monitor_horizontal_length_l462_46211


namespace NUMINAMATH_CALUDE_paul_initial_stock_l462_46212

/-- The number of pencils Paul makes in a day -/
def daily_production : ℕ := 100

/-- The number of days Paul works in a week -/
def working_days : ℕ := 5

/-- The number of pencils Paul sold during the week -/
def pencils_sold : ℕ := 350

/-- The number of pencils in stock at the end of the week -/
def end_stock : ℕ := 230

/-- The number of pencils Paul had at the beginning of the week -/
def initial_stock : ℕ := daily_production * working_days + end_stock - pencils_sold

theorem paul_initial_stock :
  initial_stock = 380 :=
sorry

end NUMINAMATH_CALUDE_paul_initial_stock_l462_46212


namespace NUMINAMATH_CALUDE_unique_solution_lcm_gcd_equation_l462_46206

theorem unique_solution_lcm_gcd_equation : 
  ∃! n : ℕ+, Nat.lcm n 120 = Nat.gcd n 120 + 300 ∧ n = 180 := by sorry

end NUMINAMATH_CALUDE_unique_solution_lcm_gcd_equation_l462_46206


namespace NUMINAMATH_CALUDE_N2O3_molecular_weight_l462_46264

/-- The atomic weight of nitrogen in g/mol -/
def nitrogen_weight : ℝ := 14.01

/-- The atomic weight of oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The number of nitrogen atoms in N2O3 -/
def nitrogen_count : ℕ := 2

/-- The number of oxygen atoms in N2O3 -/
def oxygen_count : ℕ := 3

/-- The molecular weight of N2O3 in g/mol -/
def N2O3_weight : ℝ := nitrogen_count * nitrogen_weight + oxygen_count * oxygen_weight

theorem N2O3_molecular_weight : N2O3_weight = 76.02 := by
  sorry

end NUMINAMATH_CALUDE_N2O3_molecular_weight_l462_46264


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l462_46238

theorem expression_simplification_and_evaluation :
  let x : ℤ := -3
  let y : ℤ := -2
  3 * x^2 * y - (2 * x^2 * y - (2 * x * y - x^2 * y) - 4 * x^2 * y) - x * y = -66 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l462_46238


namespace NUMINAMATH_CALUDE_min_value_expression_l462_46248

theorem min_value_expression (x y : ℝ) : 
  ∃ (a b : ℝ), (x * y + 1)^2 + (x + y + 1)^2 ≥ 0 ∧ (a * b + 1)^2 + (a + b + 1)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l462_46248


namespace NUMINAMATH_CALUDE_neg_one_is_square_sum_of_three_squares_zero_not_sum_of_three_nonzero_squares_l462_46288

/-- A field K of characteristic p where p ≡ 1 (mod 4) -/
class CharacteristicP (K : Type) [Field K] where
  char_p : Nat
  char_p_prime : Prime char_p
  char_p_mod_4 : char_p % 4 = 1

variable {K : Type} [Field K] [CharacteristicP K]

/-- -1 is a square in K -/
theorem neg_one_is_square : ∃ x : K, x^2 = -1 := by sorry

/-- Any nonzero element in K can be written as the sum of three nonzero squares -/
theorem sum_of_three_squares (a : K) (ha : a ≠ 0) : 
  ∃ x y z : K, x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x^2 + y^2 + z^2 = a := by sorry

/-- 0 cannot be written as the sum of three nonzero squares -/
theorem zero_not_sum_of_three_nonzero_squares :
  ¬∃ x y z : K, x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x^2 + y^2 + z^2 = 0 := by sorry

end NUMINAMATH_CALUDE_neg_one_is_square_sum_of_three_squares_zero_not_sum_of_three_nonzero_squares_l462_46288


namespace NUMINAMATH_CALUDE_number_of_boys_in_class_l462_46290

/-- Given the conditions of a class weight calculation, prove the number of boys in the class -/
theorem number_of_boys_in_class 
  (incorrect_avg : ℝ) 
  (misread_weight : ℝ) 
  (correct_weight : ℝ) 
  (correct_avg : ℝ) 
  (h1 : incorrect_avg = 58.4)
  (h2 : misread_weight = 56)
  (h3 : correct_weight = 60)
  (h4 : correct_avg = 58.6) :
  ∃ n : ℕ, n * incorrect_avg + (correct_weight - misread_weight) = n * correct_avg ∧ n = 20 :=
by sorry

end NUMINAMATH_CALUDE_number_of_boys_in_class_l462_46290


namespace NUMINAMATH_CALUDE_root_twice_other_iff_a_equals_four_l462_46230

theorem root_twice_other_iff_a_equals_four (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
    x^2 - (2*a + 1)*x + a^2 + 2 = 0 ∧ 
    y^2 - (2*a + 1)*y + a^2 + 2 = 0 ∧ 
    y = 2*x) ↔ 
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_root_twice_other_iff_a_equals_four_l462_46230


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l462_46287

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_simplification :
  (3 - 2 * i) / (1 + 4 * i) = -5 / 17 - 14 / 17 * i :=
by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l462_46287


namespace NUMINAMATH_CALUDE_pages_read_today_l462_46289

theorem pages_read_today (pages_yesterday pages_total : ℕ) 
  (h1 : pages_yesterday = 21)
  (h2 : pages_total = 38) :
  pages_total - pages_yesterday = 17 := by
sorry

end NUMINAMATH_CALUDE_pages_read_today_l462_46289


namespace NUMINAMATH_CALUDE_last_digits_divisible_by_4_l462_46225

-- Define a function to check if a number is divisible by 4
def divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

-- Define a function to get the last digit of a number
def last_digit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem last_digits_divisible_by_4 :
  ∃! (s : Finset ℕ), (∀ n ∈ s, ∃ m : ℕ, divisible_by_4 m ∧ last_digit m = n) ∧ s.card = 3 :=
sorry

end NUMINAMATH_CALUDE_last_digits_divisible_by_4_l462_46225


namespace NUMINAMATH_CALUDE_pure_imaginary_m_equals_four_l462_46268

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := (2*m - 8) + (m - 2)*Complex.I

-- Define what it means for a complex number to be pure imaginary
def isPureImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- Theorem statement
theorem pure_imaginary_m_equals_four :
  ∃ m : ℝ, isPureImaginary (z m) → m = 4 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_m_equals_four_l462_46268


namespace NUMINAMATH_CALUDE_estimate_fish_population_l462_46256

/-- Estimates the total number of fish in a pond using the mark-recapture method. -/
theorem estimate_fish_population (tagged_fish : ℕ) (second_sample : ℕ) (tagged_in_sample : ℕ) :
  tagged_fish = 100 →
  second_sample = 200 →
  tagged_in_sample = 10 →
  (tagged_fish * second_sample) / tagged_in_sample = 2000 :=
by
  sorry

#check estimate_fish_population

end NUMINAMATH_CALUDE_estimate_fish_population_l462_46256


namespace NUMINAMATH_CALUDE_gcd_39_91_l462_46204

theorem gcd_39_91 : Nat.gcd 39 91 = 13 := by
  sorry

end NUMINAMATH_CALUDE_gcd_39_91_l462_46204


namespace NUMINAMATH_CALUDE_convex_curve_properties_l462_46275

/-- Represents a convex curve in a 2D plane -/
structure ConvexCurve where
  -- Add necessary fields and properties for a convex curve
  -- This is a simplified representation

/-- Defines the reflection of a curve about a point -/
def reflect (K : ConvexCurve) (O : Point) : ConvexCurve :=
  sorry

/-- Defines the arithmetic mean of two curves -/
def arithmeticMean (K1 K2 : ConvexCurve) : ConvexCurve :=
  sorry

/-- Checks if a curve has a center of symmetry -/
def hasCenterOfSymmetry (K : ConvexCurve) : Prop :=
  sorry

/-- Calculates the diameter of a curve -/
def diameter (K : ConvexCurve) : ℝ :=
  sorry

/-- Calculates the width of a curve -/
def width (K : ConvexCurve) : ℝ :=
  sorry

/-- Calculates the length of a curve -/
def length (K : ConvexCurve) : ℝ :=
  sorry

/-- Calculates the area enclosed by a curve -/
def area (K : ConvexCurve) : ℝ :=
  sorry

theorem convex_curve_properties (K : ConvexCurve) (O : Point) :
  let K' := reflect K O
  let K_star := arithmeticMean K K'
  (hasCenterOfSymmetry K_star) ∧
  (diameter K_star = diameter K) ∧
  (width K_star = width K) ∧
  (length K_star = length K) ∧
  (area K_star ≥ area K) :=
by
  sorry

end NUMINAMATH_CALUDE_convex_curve_properties_l462_46275


namespace NUMINAMATH_CALUDE_rectangle_construction_l462_46201

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  P : Point
  Q : Point
  R : Point
  S : Point

/-- Checks if four points form a rectangle -/
def isRectangle (r : Rectangle) : Prop := sorry

/-- Calculates the aspect ratio of a rectangle -/
def aspectRatio (r : Rectangle) : ℝ := sorry

/-- Checks if a point lies on a line segment between two other points -/
def onSegment (p q r : Point) : Prop := sorry

/-- Theorem: A rectangle with a given aspect ratio can be constructed
    given one point on each of its sides -/
theorem rectangle_construction
  (a : ℝ)
  (A B C D : Point)
  (h_a : a > 0) :
  ∃ (r : Rectangle),
    isRectangle r ∧
    aspectRatio r = a ∧
    onSegment r.P A r.Q ∧
    onSegment r.Q B r.R ∧
    onSegment r.R C r.S ∧
    onSegment r.S D r.P :=
by sorry

end NUMINAMATH_CALUDE_rectangle_construction_l462_46201


namespace NUMINAMATH_CALUDE_slipper_price_calculation_l462_46246

/-- Given a pair of slippers with original price P, prove that with a 10% discount,
    $5.50 embroidery cost per shoe, $10.00 shipping, and $66.00 total cost,
    the original price P must be $50.00. -/
theorem slipper_price_calculation (P : ℝ) : 
  (0.90 * P + 2 * 5.50 + 10.00 = 66.00) → P = 50.00 := by
  sorry

end NUMINAMATH_CALUDE_slipper_price_calculation_l462_46246


namespace NUMINAMATH_CALUDE_part1_part2_l462_46249

-- Define the function y
def y (a x : ℝ) : ℝ := a * x^2 + (1 - a) * x + a - 2

-- Part 1
theorem part1 : ∀ a : ℝ, (∀ x : ℝ, y a x ≥ -2) ↔ a ∈ Set.Ici (1/3) :=
sorry

-- Part 2
def solution_set (a : ℝ) : Set ℝ :=
  if a > 0 then { x | -1/a < x ∧ x < 1 }
  else if a = 0 then { x | x < 1 }
  else if -1 < a ∧ a < 0 then { x | x < 1 ∨ x > -1/a }
  else if a = -1 then { x | x ≠ 1 }
  else { x | x < -1/a ∨ x > 1 }

theorem part2 : ∀ a : ℝ, ∀ x : ℝ, x ∈ solution_set a ↔ a * x^2 + (1 - a) * x - 1 < 0 :=
sorry

end NUMINAMATH_CALUDE_part1_part2_l462_46249


namespace NUMINAMATH_CALUDE_log_one_half_of_one_eighth_l462_46245

theorem log_one_half_of_one_eighth (a : ℝ) : a = Real.log 0.125 / Real.log (1/2) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_log_one_half_of_one_eighth_l462_46245


namespace NUMINAMATH_CALUDE_sum_of_lg2_and_lg5_power_of_8_two_thirds_l462_46208

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Theorem 1: lg2 + lg5 = 1
theorem sum_of_lg2_and_lg5 : lg 2 + lg 5 = 1 := by sorry

-- Theorem 2: 8^(2/3) = 4
theorem power_of_8_two_thirds : (8 : ℝ) ^ (2/3) = 4 := by sorry

end NUMINAMATH_CALUDE_sum_of_lg2_and_lg5_power_of_8_two_thirds_l462_46208


namespace NUMINAMATH_CALUDE_derivative_of_f_l462_46229

-- Define the function f
def f (x : ℝ) : ℝ := (x + 1)^2

-- State the theorem
theorem derivative_of_f :
  deriv f = fun x ↦ 2 * x + 2 := by sorry

end NUMINAMATH_CALUDE_derivative_of_f_l462_46229


namespace NUMINAMATH_CALUDE_constant_x_coordinate_l462_46280

/-- Definition of the ellipse C -/
def C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- Right focus F -/
def F : ℝ × ℝ := (1, 0)

/-- Left vertex A -/
def A : ℝ × ℝ := (-2, 0)

/-- Right vertex B -/
def B : ℝ × ℝ := (2, 0)

/-- Line l passing through F, not coincident with x-axis -/
def l (k : ℝ) (x y : ℝ) : Prop := y = k*(x - F.1) ∧ k ≠ 0

/-- Intersection points M and N of line l with ellipse C -/
def intersectionPoints (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | C p.1 p.2 ∧ l k p.1 p.2}

/-- Line AM -/
def lineAM (M : ℝ × ℝ) (x y : ℝ) : Prop :=
  (y - A.2) * (M.1 - A.1) = (x - A.1) * (M.2 - A.2)

/-- Line BN -/
def lineBN (N : ℝ × ℝ) (x y : ℝ) : Prop :=
  (y - B.2) * (N.1 - B.1) = (x - B.1) * (N.2 - B.2)

/-- Theorem: x-coordinate of intersection point T is constant -/
theorem constant_x_coordinate (k : ℝ) (M N : ℝ × ℝ) (h1 : M ∈ intersectionPoints k) (h2 : N ∈ intersectionPoints k) (h3 : M ≠ N) :
  ∃ (T : ℝ × ℝ), lineAM M T.1 T.2 ∧ lineBN N T.1 T.2 ∧ T.1 = 4 := by sorry

end NUMINAMATH_CALUDE_constant_x_coordinate_l462_46280


namespace NUMINAMATH_CALUDE_absolute_value_equality_l462_46293

theorem absolute_value_equality (y : ℝ) : |y + 2| = |y - 3| → y = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l462_46293


namespace NUMINAMATH_CALUDE_b_10_value_l462_46259

theorem b_10_value (a b : ℕ → ℝ) 
  (h1 : ∀ n, (a n) * (a (n + 1)) = 2^n)
  (h2 : ∀ n, (a n) + (a (n + 1)) = b n)
  (h3 : a 1 = 1) :
  b 10 = 64 := by
sorry

end NUMINAMATH_CALUDE_b_10_value_l462_46259


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l462_46286

theorem sqrt_product_equality (x y : ℝ) (hx : x ≥ 0) :
  Real.sqrt (3 * x) * Real.sqrt ((1 / 3) * x * y) = x * Real.sqrt y := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l462_46286


namespace NUMINAMATH_CALUDE_tom_remaining_candy_l462_46223

/-- The number of candy pieces Tom still has after giving some away to his brother -/
def remaining_candy_pieces : ℕ :=
  let initial_chocolate_boxes : ℕ := 14
  let initial_fruit_boxes : ℕ := 10
  let initial_caramel_boxes : ℕ := 8
  let given_chocolate_boxes : ℕ := 8
  let given_fruit_boxes : ℕ := 5
  let pieces_per_chocolate_box : ℕ := 3
  let pieces_per_fruit_box : ℕ := 4
  let pieces_per_caramel_box : ℕ := 5

  let initial_total_pieces : ℕ := 
    initial_chocolate_boxes * pieces_per_chocolate_box +
    initial_fruit_boxes * pieces_per_fruit_box +
    initial_caramel_boxes * pieces_per_caramel_box

  let given_away_pieces : ℕ := 
    given_chocolate_boxes * pieces_per_chocolate_box +
    given_fruit_boxes * pieces_per_fruit_box

  initial_total_pieces - given_away_pieces

theorem tom_remaining_candy : remaining_candy_pieces = 78 := by
  sorry

end NUMINAMATH_CALUDE_tom_remaining_candy_l462_46223


namespace NUMINAMATH_CALUDE_fraction_evaluation_l462_46267

theorem fraction_evaluation : (3 : ℚ) / (2 - 5 / 4) = 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l462_46267


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l462_46278

/-- Two vectors in ℝ² are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (-2, 4)
  let b : ℝ × ℝ := (x, -2)
  are_parallel a b → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l462_46278


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_reciprocals_of_first_four_primes_l462_46260

-- Define the first four prime numbers
def first_four_primes : List ℕ := [2, 3, 5, 7]

-- Define a function to calculate the reciprocal of a natural number
def reciprocal (n : ℕ) : ℚ := 1 / n

-- Define the arithmetic mean of a list of rational numbers
def arithmetic_mean (list : List ℚ) : ℚ := (list.sum) / list.length

-- Theorem statement
theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  arithmetic_mean (first_four_primes.map reciprocal) = 247 / 840 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_reciprocals_of_first_four_primes_l462_46260


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_relation_l462_46297

theorem rectangle_area_diagonal_relation (l w d : ℝ) (h1 : l / w = 4 / 3) (h2 : l^2 + w^2 = d^2) :
  ∃ k : ℝ, l * w = k * d^2 ∧ k = 12 / 25 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_relation_l462_46297


namespace NUMINAMATH_CALUDE_geometric_series_sum_l462_46266

theorem geometric_series_sum (c d : ℝ) (h : ∑' n, c / d^n = 3) :
  ∑' n, c / (c + 2*d)^n = (3*d - 3) / (5*d - 4) := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l462_46266


namespace NUMINAMATH_CALUDE_population_growth_rate_l462_46233

/-- The time it takes for one person to be added to the population, given the rate of population increase. -/
def time_per_person (persons_per_hour : ℕ) : ℚ :=
  (60 * 60) / persons_per_hour

/-- Theorem stating that the time it takes for one person to be added to the population is 15 seconds, 
    given that the population increases by 240 persons in 60 minutes. -/
theorem population_growth_rate : time_per_person 240 = 15 := by
  sorry

end NUMINAMATH_CALUDE_population_growth_rate_l462_46233


namespace NUMINAMATH_CALUDE_ryegrass_percentage_in_mixture_l462_46277

-- Define the compositions of mixtures X and Y
def x_ryegrass : ℝ := 0.4
def x_bluegrass : ℝ := 0.6
def y_ryegrass : ℝ := 0.25
def y_fescue : ℝ := 0.75

-- Define the proportion of X in the final mixture
def x_proportion : ℝ := 0.3333333333333333

-- Define the proportion of Y in the final mixture
def y_proportion : ℝ := 1 - x_proportion

-- Theorem statement
theorem ryegrass_percentage_in_mixture :
  x_ryegrass * x_proportion + y_ryegrass * y_proportion = 0.3 := by sorry

end NUMINAMATH_CALUDE_ryegrass_percentage_in_mixture_l462_46277


namespace NUMINAMATH_CALUDE_coefficient_x_squared_expansion_l462_46226

theorem coefficient_x_squared_expansion : 
  let p : Polynomial ℤ := (X + 1)^5 * (X - 2)
  p.coeff 2 = -15 := by sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_expansion_l462_46226


namespace NUMINAMATH_CALUDE_hat_price_after_discounts_l462_46252

def initial_price : ℝ := 15
def first_discount : ℝ := 0.25
def second_discount : ℝ := 0.50

theorem hat_price_after_discounts :
  let price_after_first_discount := initial_price * (1 - first_discount)
  let final_price := price_after_first_discount * (1 - second_discount)
  final_price = 5.625 := by sorry

end NUMINAMATH_CALUDE_hat_price_after_discounts_l462_46252


namespace NUMINAMATH_CALUDE_line_intersection_y_axis_l462_46207

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line by two points
structure Line where
  p1 : Point2D
  p2 : Point2D

-- Define the y-axis
def yAxis : Line := { p1 := ⟨0, 0⟩, p2 := ⟨0, 1⟩ }

-- Function to check if a point is on a line
def isPointOnLine (p : Point2D) (l : Line) : Prop :=
  (p.y - l.p1.y) * (l.p2.x - l.p1.x) = (p.x - l.p1.x) * (l.p2.y - l.p1.y)

-- Function to check if a point is on the y-axis
def isPointOnYAxis (p : Point2D) : Prop :=
  p.x = 0

-- Theorem statement
theorem line_intersection_y_axis :
  let l : Line := { p1 := ⟨2, 9⟩, p2 := ⟨4, 13⟩ }
  let intersection : Point2D := ⟨0, 5⟩
  isPointOnLine intersection l ∧ isPointOnYAxis intersection := by
  sorry

end NUMINAMATH_CALUDE_line_intersection_y_axis_l462_46207


namespace NUMINAMATH_CALUDE_cosine_sum_l462_46253

theorem cosine_sum (α β : Real) : 
  α ∈ Set.Ioo 0 (π/3) →
  β ∈ Set.Ioo (π/6) (π/2) →
  5 * Real.sqrt 3 * Real.sin α + 5 * Real.cos α = 8 →
  Real.sqrt 2 * Real.sin β + Real.sqrt 6 * Real.cos β = 2 →
  Real.cos (α + β) = -(Real.sqrt 2) / 10 := by
sorry

end NUMINAMATH_CALUDE_cosine_sum_l462_46253


namespace NUMINAMATH_CALUDE_particle_movement_l462_46284

/-- Represents a particle in a 2D grid -/
structure Particle where
  x : ℚ
  y : ℚ

/-- Represents the probabilities of movement for Particle A -/
structure ProbA where
  left : ℚ
  right : ℚ
  up : ℚ
  down : ℚ

/-- Represents the probability of movement for Particle B -/
def ProbB : ℚ → Prop := λ y ↦ ∀ (direction : Fin 4), y = 1/4

/-- The theorem statement -/
theorem particle_movement 
  (A : Particle) 
  (B : Particle) 
  (probA : ProbA) 
  (probB : ℚ → Prop) :
  A.x = 0 ∧ A.y = 0 ∧
  B.x = 1 ∧ B.y = 1 ∧
  probA.left = 1/4 ∧ probA.right = 1/4 ∧ probA.up = 1/3 ∧
  ProbB probA.down ∧
  (∃ (x : ℚ), probA.down = x ∧ x + 1/4 + 1/4 + 1/3 = 1) →
  probA.down = 1/6 ∧
  ProbB (1/4) ∧
  (∃ (t : ℕ), t = 3 ∧ 
    (∀ (t' : ℕ), (∃ (A' B' : Particle), A'.x = 2 ∧ A'.y = 1 ∧ B'.x = 2 ∧ B'.y = 1) → t' ≥ t)) ∧
  (9 : ℚ)/1024 = 
    (3 * (1/4)^2 * 1/3) * -- Probability for A
    (1/4 * 3 * (1/4)^2)   -- Probability for B
  := by sorry

end NUMINAMATH_CALUDE_particle_movement_l462_46284


namespace NUMINAMATH_CALUDE_square_root_divided_by_six_l462_46283

theorem square_root_divided_by_six : Real.sqrt 144 / 6 = 2 := by sorry

end NUMINAMATH_CALUDE_square_root_divided_by_six_l462_46283


namespace NUMINAMATH_CALUDE_shaded_area_in_square_l462_46224

/-- Given a square with side length a, the area bounded by a semicircle on one side
    and two quarter-circle arcs on the adjacent sides is equal to a²/2 -/
theorem shaded_area_in_square (a : ℝ) (h : a > 0) :
  (π * a^2 / 8) + (π * a^2 / 8) = a^2 / 2 := by
  sorry

#check shaded_area_in_square

end NUMINAMATH_CALUDE_shaded_area_in_square_l462_46224


namespace NUMINAMATH_CALUDE_fraction_addition_l462_46296

theorem fraction_addition (c : ℝ) : (5 + 5 * c) / 7 + 3 = (26 + 5 * c) / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l462_46296


namespace NUMINAMATH_CALUDE_equation_solution_l462_46217

theorem equation_solution : ∃! (x y : ℝ), 3*x^2 + 14*y^2 - 12*x*y + 6*x - 20*y + 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l462_46217


namespace NUMINAMATH_CALUDE_flowers_per_vase_is_nine_l462_46221

/-- The number of carnations -/
def carnations : ℕ := 4

/-- The number of roses -/
def roses : ℕ := 23

/-- The total number of vases needed -/
def vases : ℕ := 3

/-- The total number of flowers -/
def total_flowers : ℕ := carnations + roses

/-- The number of flowers one vase can hold -/
def flowers_per_vase : ℕ := total_flowers / vases

theorem flowers_per_vase_is_nine : flowers_per_vase = 9 := by
  sorry

end NUMINAMATH_CALUDE_flowers_per_vase_is_nine_l462_46221


namespace NUMINAMATH_CALUDE_tan_135_degrees_l462_46210

theorem tan_135_degrees : Real.tan (135 * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_135_degrees_l462_46210


namespace NUMINAMATH_CALUDE_tetrahedron_height_formula_l462_46219

/-- Configuration of four mutually tangent spheres -/
structure SpheresConfiguration where
  small_radius : ℝ
  large_radius : ℝ
  small_spheres_count : ℕ
  on_flat_floor : Prop

/-- Tetrahedron circumscribing the spheres configuration -/
def circumscribing_tetrahedron (config : SpheresConfiguration) : Prop :=
  sorry

/-- Height of the tetrahedron from the floor to the opposite vertex -/
noncomputable def tetrahedron_height (config : SpheresConfiguration) : ℝ :=
  sorry

/-- Theorem stating the height of the tetrahedron -/
theorem tetrahedron_height_formula (config : SpheresConfiguration) 
  (h1 : config.small_radius = 2)
  (h2 : config.large_radius = 3)
  (h3 : config.small_spheres_count = 3)
  (h4 : config.on_flat_floor)
  (h5 : circumscribing_tetrahedron config) :
  tetrahedron_height config = (Real.sqrt 177 + 9 * Real.sqrt 3) / 3 :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_height_formula_l462_46219


namespace NUMINAMATH_CALUDE_f_greater_g_when_x_greater_two_sum_greater_four_when_f_equal_l462_46272

noncomputable def f (x : ℝ) : ℝ := (x - 1) / Real.exp (x - 1)

noncomputable def g (x : ℝ) : ℝ := f (4 - x)

theorem f_greater_g_when_x_greater_two :
  ∀ x : ℝ, x > 2 → f x > g x :=
sorry

theorem sum_greater_four_when_f_equal :
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → f x₁ = f x₂ → x₁ + x₂ > 4 :=
sorry

end NUMINAMATH_CALUDE_f_greater_g_when_x_greater_two_sum_greater_four_when_f_equal_l462_46272


namespace NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l462_46214

/-- Two hyperbolas have the same asymptotes if and only if M = 225/16 -/
theorem hyperbolas_same_asymptotes (M : ℝ) :
  (∀ x y : ℝ, x^2 / 9 - y^2 / 16 = 1 ↔ y^2 / 25 - x^2 / M = 1) ↔ M = 225 / 16 :=
by sorry

end NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l462_46214


namespace NUMINAMATH_CALUDE_masha_ate_ten_pies_l462_46291

/-- Represents the eating rates of Masha and the bear -/
structure EatingRates where
  masha : ℝ
  bear : ℝ
  bear_faster : bear = 3 * masha

/-- Represents the distribution of food between Masha and the bear -/
structure FoodDistribution where
  total_pies : ℕ
  total_pies_positive : total_pies > 0
  masha_pies : ℕ
  bear_pies : ℕ
  pies_sum : masha_pies + bear_pies = total_pies
  equal_raspberries : ℝ  -- Represents the fact that they ate equal raspberries

/-- Theorem stating that Masha ate 10 pies given the problem conditions -/
theorem masha_ate_ten_pies (rates : EatingRates) (food : FoodDistribution) 
  (h_total_pies : food.total_pies = 40) :
  food.masha_pies = 10 := by
  sorry


end NUMINAMATH_CALUDE_masha_ate_ten_pies_l462_46291


namespace NUMINAMATH_CALUDE_solve_triangle_problem_l462_46282

/-- Represents a right-angled isosceles triangle --/
structure RightIsoscelesTriangle where
  side : ℝ
  area : ℝ
  area_eq : area = side^2 / 2

/-- The problem setup --/
def triangle_problem (k : ℝ) : Prop :=
  let t1 := RightIsoscelesTriangle.mk k (k^2 / 2) (by rfl)
  let t2 := RightIsoscelesTriangle.mk (k * Real.sqrt 2) (k^2) (by sorry)
  let t3 := RightIsoscelesTriangle.mk (2 * k) (2 * k^2) (by sorry)
  t1.area + t2.area + t3.area = 56

/-- The theorem to prove --/
theorem solve_triangle_problem : 
  ∃ k : ℝ, triangle_problem k ∧ k = 4 := by sorry

end NUMINAMATH_CALUDE_solve_triangle_problem_l462_46282


namespace NUMINAMATH_CALUDE_solution_range_l462_46242

theorem solution_range (a : ℝ) : 
  (∃ x : ℝ, x - 1 ≥ a^2 ∧ x - 4 < 2*a) → 
  a ∈ Set.Ioo (-1 : ℝ) 3 :=
by sorry

end NUMINAMATH_CALUDE_solution_range_l462_46242


namespace NUMINAMATH_CALUDE_rectangular_hyperbola_equation_l462_46250

/-- A rectangular hyperbola with coordinate axes as its axes of symmetry
    passing through the point (2, √2) has the equation x² - y² = 2 -/
theorem rectangular_hyperbola_equation :
  ∀ (f : ℝ → ℝ → Prop),
    (∀ x y, f x y ↔ x^2 - y^2 = 2) →  -- Definition of the hyperbola equation
    (∀ x, f x 0 ↔ f 0 x) →            -- Symmetry about y = x
    (∀ x, f x 0 ↔ f (-x) 0) →         -- Symmetry about y-axis
    (∀ y, f 0 y ↔ f 0 (-y)) →         -- Symmetry about x-axis
    f 2 (Real.sqrt 2) →               -- Point (2, √2) lies on the hyperbola
    ∀ x y, f x y ↔ x^2 - y^2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_hyperbola_equation_l462_46250


namespace NUMINAMATH_CALUDE_even_function_property_l462_46243

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem even_function_property (f : ℝ → ℝ) 
  (h1 : is_even f) 
  (h2 : is_even (λ x ↦ f (x + 2))) 
  (h3 : f 1 = π / 3) : 
  f 3 + f (-3) = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_even_function_property_l462_46243


namespace NUMINAMATH_CALUDE_simplify_fraction_l462_46265

theorem simplify_fraction : (216 : ℚ) / 4536 = 1 / 21 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l462_46265


namespace NUMINAMATH_CALUDE_solve_quadratic_l462_46251

-- Define the universal set U
def U : Set ℕ := {2, 3, 5}

-- Define the set A
def A (b c : ℤ) : Set ℕ := {x ∈ U | x^2 + b*x + c = 0}

-- Define the complement of A with respect to U
def complement_A (b c : ℤ) : Set ℕ := U \ A b c

-- Theorem statement
theorem solve_quadratic (b c : ℤ) : complement_A b c = {2} → b = -8 ∧ c = 15 := by
  sorry


end NUMINAMATH_CALUDE_solve_quadratic_l462_46251


namespace NUMINAMATH_CALUDE_emerson_rowing_trip_l462_46294

theorem emerson_rowing_trip (total_distance initial_distance second_part_distance : ℕ) 
  (h1 : total_distance = 39)
  (h2 : initial_distance = 6)
  (h3 : second_part_distance = 15) :
  total_distance - (initial_distance + second_part_distance) = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_emerson_rowing_trip_l462_46294


namespace NUMINAMATH_CALUDE_opposite_numbers_and_unit_absolute_value_l462_46240

theorem opposite_numbers_and_unit_absolute_value 
  (a b c : ℝ) 
  (h1 : a + b = 0) 
  (h2 : abs c = 1) : 
  a + b - c = 1 ∨ a + b - c = -1 := by
sorry

end NUMINAMATH_CALUDE_opposite_numbers_and_unit_absolute_value_l462_46240


namespace NUMINAMATH_CALUDE_parking_arrangement_count_l462_46295

/-- The number of parking spaces -/
def n : ℕ := 50

/-- The number of cars to be arranged -/
def k : ℕ := 2

/-- The number of ways to arrange k distinct cars in n parking spaces -/
def total_arrangements (n k : ℕ) : ℕ := n * (n - 1)

/-- The number of ways to arrange k distinct cars adjacently in n parking spaces -/
def adjacent_arrangements (n : ℕ) : ℕ := 2 * (n - 1)

/-- The number of ways to arrange k distinct cars in n parking spaces with at least one empty space between them -/
def valid_arrangements (n k : ℕ) : ℕ := total_arrangements n k - adjacent_arrangements n

theorem parking_arrangement_count :
  valid_arrangements n k = 2352 :=
by sorry

end NUMINAMATH_CALUDE_parking_arrangement_count_l462_46295


namespace NUMINAMATH_CALUDE_ellipse_symmetry_l462_46281

-- Define the original ellipse
def original_ellipse (x y : ℝ) : Prop :=
  (x - 3)^2 / 9 + (y - 2)^2 / 4 = 1

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop :=
  x + y = 0

-- Define the reflection transformation
def reflect (x y : ℝ) : ℝ × ℝ :=
  (-y, -x)

-- Define the resulting ellipse C
def ellipse_c (x y : ℝ) : Prop :=
  (x + 2)^2 / 9 + (y + 3)^2 / 4 = 1

-- Theorem statement
theorem ellipse_symmetry :
  ∀ (x y : ℝ),
    original_ellipse x y →
    let (x', y') := reflect x y
    ellipse_c x' y' :=
by sorry

end NUMINAMATH_CALUDE_ellipse_symmetry_l462_46281


namespace NUMINAMATH_CALUDE_multiple_of_number_l462_46202

theorem multiple_of_number : ∃ m : ℕ, m < 4 ∧ 7 * 5 - 15 > m * 5 := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_number_l462_46202


namespace NUMINAMATH_CALUDE_roots_between_values_l462_46276

theorem roots_between_values (a b c : ℝ) (h : a < b ∧ b < c) :
  ∃ x₁ x₂ : ℝ,
    (1 / (x₁ - a) + 1 / (x₁ - b) + 1 / (x₁ - c) = 0) ∧
    (1 / (x₂ - a) + 1 / (x₂ - b) + 1 / (x₂ - c) = 0) ∧
    (a < x₁ ∧ x₁ < b ∧ b < x₂ ∧ x₂ < c) := by
  sorry

end NUMINAMATH_CALUDE_roots_between_values_l462_46276


namespace NUMINAMATH_CALUDE_tangent_length_correct_l462_46232

/-- Two circles S₁ and S₂ touching at point A with radii R and r respectively (R > r).
    B is a point on S₁ such that AB = a. -/
structure TangentCircles where
  R : ℝ
  r : ℝ
  a : ℝ
  h₁ : R > r
  h₂ : R > 0
  h₃ : r > 0
  h₄ : a > 0

/-- The length of the tangent from B to S₂ -/
noncomputable def tangentLength (c : TangentCircles) (external : Bool) : ℝ :=
  if external then
    c.a * Real.sqrt ((c.R + c.r) / c.R)
  else
    c.a * Real.sqrt ((c.R - c.r) / c.R)

theorem tangent_length_correct (c : TangentCircles) :
  (∀ external, tangentLength c external = 
    if external then c.a * Real.sqrt ((c.R + c.r) / c.R)
    else c.a * Real.sqrt ((c.R - c.r) / c.R)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_length_correct_l462_46232


namespace NUMINAMATH_CALUDE_work_completion_time_l462_46218

theorem work_completion_time (a_time b_time initial_days : ℝ) 
  (ha : a_time = 12)
  (hb : b_time = 6)
  (hi : initial_days = 3) :
  let a_rate := 1 / a_time
  let b_rate := 1 / b_time
  let initial_work := a_rate * initial_days
  let remaining_work := 1 - initial_work
  let combined_rate := a_rate + b_rate
  (remaining_work / combined_rate) = 3 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l462_46218


namespace NUMINAMATH_CALUDE_min_value_theorem_l462_46213

theorem min_value_theorem (p q r s t u v w : ℝ) 
  (h1 : p * q * r * s = 8) 
  (h2 : t * u * v * w = 27) : 
  (p * t)^2 + (q * u)^2 + (r * v)^2 + (s * w)^2 ≥ 96 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l462_46213


namespace NUMINAMATH_CALUDE_cement_mixture_weight_l462_46209

/-- Proves that a cement mixture with given proportions weighs 48 pounds -/
theorem cement_mixture_weight (sand_fraction : ℚ) (water_fraction : ℚ) (gravel_weight : ℚ) :
  sand_fraction = 1/3 →
  water_fraction = 1/2 →
  gravel_weight = 8 →
  sand_fraction + water_fraction + gravel_weight / (sand_fraction + water_fraction + gravel_weight) = 1 →
  sand_fraction + water_fraction + gravel_weight = 48 := by
  sorry

end NUMINAMATH_CALUDE_cement_mixture_weight_l462_46209


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l462_46235

def hyperbola (m n : ℝ) (x y : ℝ) : Prop :=
  x^2 / m - y^2 / n = 1

def tangent_line (m n : ℝ) (x y : ℝ) : Prop :=
  2 * m * x - n * y + 2 = 0

def asymptote (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x ∨ y = -k * x

theorem hyperbola_asymptotes (m n : ℝ) :
  (∀ x y, hyperbola m n x y) →
  (∀ x y, tangent_line m n x y) →
  (∀ x y, asymptote (Real.sqrt 2) x y) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l462_46235


namespace NUMINAMATH_CALUDE_contrapositive_correct_l462_46234

-- Define a triangle
structure Triangle :=
  (A B C : Point)

-- Define the property of being an isosceles triangle
def isIsosceles (t : Triangle) : Prop := sorry

-- Define the property of having two equal interior angles
def hasTwoEqualAngles (t : Triangle) : Prop := sorry

-- The original statement
def originalStatement (t : Triangle) : Prop :=
  ¬(isIsosceles t) → ¬(hasTwoEqualAngles t)

-- The contrapositive of the original statement
def contrapositive (t : Triangle) : Prop :=
  hasTwoEqualAngles t → isIsosceles t

-- Theorem stating that the contrapositive is correct
theorem contrapositive_correct :
  ∀ t : Triangle, originalStatement t ↔ contrapositive t :=
sorry

end NUMINAMATH_CALUDE_contrapositive_correct_l462_46234


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l462_46205

theorem sum_of_squares_of_roots (a : ℝ) : 
  (∃ x₁ x₂ x₃ x₄ : ℝ, (x₁^4 + a*x₁^2 - 2017 = 0) ∧ 
                      (x₂^4 + a*x₂^2 - 2017 = 0) ∧ 
                      (x₃^4 + a*x₃^2 - 2017 = 0) ∧ 
                      (x₄^4 + a*x₄^2 - 2017 = 0) ∧ 
                      (x₁^2 + x₂^2 + x₃^2 + x₄^2 = 4)) → 
  a = 1006.5 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l462_46205


namespace NUMINAMATH_CALUDE_puzzle_pieces_count_l462_46203

theorem puzzle_pieces_count (pieces_per_hour : ℕ) (hours_per_day : ℕ) (days : ℕ) 
  (num_500_piece_puzzles : ℕ) (num_unknown_piece_puzzles : ℕ) :
  pieces_per_hour = 100 →
  hours_per_day = 7 →
  days = 7 →
  num_500_piece_puzzles = 5 →
  num_unknown_piece_puzzles = 8 →
  (pieces_per_hour * hours_per_day * days - num_500_piece_puzzles * 500) / num_unknown_piece_puzzles = 300 := by
  sorry

end NUMINAMATH_CALUDE_puzzle_pieces_count_l462_46203


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l462_46220

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (5 * x + 9) = 11 → x = 112 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l462_46220
