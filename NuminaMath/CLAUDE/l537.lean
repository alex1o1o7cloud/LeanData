import Mathlib

namespace NUMINAMATH_CALUDE_ab_equals_six_l537_53774

theorem ab_equals_six (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_ab_equals_six_l537_53774


namespace NUMINAMATH_CALUDE_arrange_plates_eq_365240_l537_53793

/-- Number of ways to arrange plates around a circular table with constraints -/
def arrange_plates : ℕ :=
  let total_plates : ℕ := 13
  let blue_plates : ℕ := 6
  let red_plates : ℕ := 3
  let green_plates : ℕ := 3
  let orange_plates : ℕ := 1
  let total_arrangements : ℕ := (Nat.factorial (total_plates - 1)) / (Nat.factorial blue_plates * Nat.factorial red_plates * Nat.factorial green_plates)
  let green_adjacent : ℕ := (Nat.factorial (total_plates - green_plates)) / (Nat.factorial blue_plates * Nat.factorial red_plates)
  let red_adjacent : ℕ := (Nat.factorial (total_plates - red_plates)) / (Nat.factorial blue_plates * Nat.factorial green_plates)
  total_arrangements - green_adjacent - red_adjacent

theorem arrange_plates_eq_365240 : arrange_plates = 365240 := by
  sorry

end NUMINAMATH_CALUDE_arrange_plates_eq_365240_l537_53793


namespace NUMINAMATH_CALUDE_problem_solution_l537_53732

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x * Real.cos x - Real.cos x ^ 2 - 1/2

theorem problem_solution :
  ∃ (A B C a b c : ℝ),
    0 < A ∧ A < π ∧
    0 < B ∧ B < π ∧
    0 < C ∧ C < π ∧
    Real.sin B - 2 * Real.sin A = 0 ∧
    c = 3 ∧
    f C = 0 ∧
    (∀ x, f x ≥ -2) ∧
    (∀ ε > 0, ∃ x, f x < -2 + ε) ∧
    (∀ x, f (x + π) = f x) ∧
    (∀ p > 0, (∀ x, f (x + p) = f x) → p ≥ π) ∧
    a = Real.sqrt 3 ∧
    b = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l537_53732


namespace NUMINAMATH_CALUDE_no_rebus_solution_l537_53700

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_to_nat (d : Fin 10) : ℕ := d.val

def rebus_equation (K U S Y : Fin 10) : Prop :=
  let KUSY := 1000 * (digit_to_nat K) + 100 * (digit_to_nat U) + 10 * (digit_to_nat S) + (digit_to_nat Y)
  let UKSY := 1000 * (digit_to_nat U) + 100 * (digit_to_nat K) + 10 * (digit_to_nat S) + (digit_to_nat Y)
  let UKSUS := 10000 * (digit_to_nat U) + 1000 * (digit_to_nat K) + 100 * (digit_to_nat S) + 10 * (digit_to_nat U) + (digit_to_nat S)
  is_four_digit KUSY ∧ is_four_digit UKSY ∧ is_four_digit UKSUS ∧ KUSY + UKSY = UKSUS

theorem no_rebus_solution :
  ∀ (K U S Y : Fin 10), K ≠ U ∧ K ≠ S ∧ K ≠ Y ∧ U ≠ S ∧ U ≠ Y ∧ S ≠ Y → ¬(rebus_equation K U S Y) :=
sorry

end NUMINAMATH_CALUDE_no_rebus_solution_l537_53700


namespace NUMINAMATH_CALUDE_sqrt_x_minus_3_real_range_l537_53737

theorem sqrt_x_minus_3_real_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 3) ↔ x ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_3_real_range_l537_53737


namespace NUMINAMATH_CALUDE_journey_speed_proof_l537_53727

/-- Proves that the speed of the first half of a journey is 21 km/hr, given the total journey conditions -/
theorem journey_speed_proof (total_distance : ℝ) (total_time : ℝ) (second_half_speed : ℝ) :
  total_distance = 224 →
  total_time = 10 →
  second_half_speed = 24 →
  let first_half_distance : ℝ := total_distance / 2
  let second_half_time : ℝ := first_half_distance / second_half_speed
  let first_half_time : ℝ := total_time - second_half_time
  let first_half_speed : ℝ := first_half_distance / first_half_time
  first_half_speed = 21 := by
  sorry

end NUMINAMATH_CALUDE_journey_speed_proof_l537_53727


namespace NUMINAMATH_CALUDE_multiplicative_inverse_5_mod_31_l537_53740

theorem multiplicative_inverse_5_mod_31 : ∃ x : ℤ, (5 * x) % 31 = 1 ∧ x % 31 = 25 := by
  sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_5_mod_31_l537_53740


namespace NUMINAMATH_CALUDE_box_balls_problem_l537_53771

theorem box_balls_problem (B X : ℕ) (h1 : B = 57) (h2 : B - 44 = X - B) : X = 70 := by
  sorry

end NUMINAMATH_CALUDE_box_balls_problem_l537_53771


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_l537_53779

theorem absolute_value_inequality_solution (a : ℝ) : 
  (∀ x, |x - a| ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_l537_53779


namespace NUMINAMATH_CALUDE_inequality_proof_l537_53770

theorem inequality_proof (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  (x^3 / (x^3 + 2*y^2*z)) + (y^3 / (y^3 + 2*z^2*x)) + (z^3 / (z^3 + 2*x^2*y)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l537_53770


namespace NUMINAMATH_CALUDE_hyperbola_equation_l537_53709

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, focal length 10, and point (2,1) on its asymptote, 
    prove that its equation is x²/20 - y²/5 = 1 -/
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (a^2 + b^2 = 25) → (2 * b / a = 1) → 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 / 20 - y^2 / 5 = 1) := by
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l537_53709


namespace NUMINAMATH_CALUDE_max_k_value_l537_53769

theorem max_k_value (x y k : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_k : k > 0)
  (h_eq : 4 = k^2 * (x^2 / y^2 + y^2 / x^2) + k * (x / y + y / x)) :
  k ≤ (-1 + Real.sqrt 17) / 4 := by
  sorry

end NUMINAMATH_CALUDE_max_k_value_l537_53769


namespace NUMINAMATH_CALUDE_dream_car_mileage_difference_l537_53788

/-- Proves that the difference in miles driven between tomorrow and today is 200 miles -/
theorem dream_car_mileage_difference (consumption_rate : ℝ) (today_miles : ℝ) (total_consumption : ℝ)
  (h1 : consumption_rate = 4)
  (h2 : today_miles = 400)
  (h3 : total_consumption = 4000) :
  (total_consumption / consumption_rate) - today_miles = 200 := by
  sorry

end NUMINAMATH_CALUDE_dream_car_mileage_difference_l537_53788


namespace NUMINAMATH_CALUDE_similar_quadrilaterals_rectangle_areas_l537_53725

/-- Given two similar quadrilaterals with sides (a, b, c, d) and (a', b', c', d') respectively,
    and similarity ratio k, prove that the areas of rectangles formed by opposite sides
    are proportional to k^2 -/
theorem similar_quadrilaterals_rectangle_areas
  (a b c d a' b' c' d' k : ℝ)
  (h_similar : a / a' = b / b' ∧ b / b' = c / c' ∧ c / c' = d / d' ∧ d / d' = k) :
  a * c / (a' * c') = k^2 ∧ b * d / (b' * d') = k^2 := by
  sorry

end NUMINAMATH_CALUDE_similar_quadrilaterals_rectangle_areas_l537_53725


namespace NUMINAMATH_CALUDE_expression_evaluation_l537_53748

theorem expression_evaluation (x : ℝ) : 
  x * (x * (x * (3 - x) - 5) + 12) + 2 = -x^4 + 3*x^3 - 5*x^2 + 12*x + 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l537_53748


namespace NUMINAMATH_CALUDE_smallest_c_for_unique_solution_l537_53786

/-- The system of equations -/
def system (x y c : ℝ) : Prop :=
  2 * (x + 7)^2 + (y - 4)^2 = c ∧ (x + 4)^2 + 2 * (y - 7)^2 = c

/-- The system has a unique solution -/
def has_unique_solution (c : ℝ) : Prop :=
  ∃! p : ℝ × ℝ, system p.1 p.2 c

/-- The smallest value of c for which the system has a unique solution is 6.0 -/
theorem smallest_c_for_unique_solution :
  (∀ c < 6, ¬ has_unique_solution c) ∧ has_unique_solution 6 :=
sorry

end NUMINAMATH_CALUDE_smallest_c_for_unique_solution_l537_53786


namespace NUMINAMATH_CALUDE_primitive_cube_root_expression_l537_53723

/-- ω is a primitive third root of unity -/
def ω : ℂ :=
  sorry

/-- ω is a primitive third root of unity -/
axiom ω_is_primitive_cube_root : ω^3 = 1 ∧ ω ≠ 1

/-- The value of the expression (1-ω)(1-ω^2)(1-ω^4)(1-ω^8) -/
theorem primitive_cube_root_expression : (1 - ω) * (1 - ω^2) * (1 - ω^4) * (1 - ω^8) = 9 :=
  sorry

end NUMINAMATH_CALUDE_primitive_cube_root_expression_l537_53723


namespace NUMINAMATH_CALUDE_fly_distance_from_ceiling_l537_53792

theorem fly_distance_from_ceiling (x y z : ℝ) : 
  x = 3 ∧ y = 7 ∧ x^2 + y^2 + z^2 = 10^2 → z = Real.sqrt 42 := by
  sorry

end NUMINAMATH_CALUDE_fly_distance_from_ceiling_l537_53792


namespace NUMINAMATH_CALUDE_least_four_digit_divisible_by_digits_l537_53720

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def all_digits_different (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 4 ∧ digits.toFinset.card = 4

def divisible_by_nonzero_digits (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d ≠ 0 → n % d = 0

theorem least_four_digit_divisible_by_digits :
  ∃ (n : ℕ), is_four_digit n ∧
             all_digits_different n ∧
             divisible_by_nonzero_digits n ∧
             (∀ m : ℕ, is_four_digit m ∧
                       all_digits_different m ∧
                       divisible_by_nonzero_digits m →
                       n ≤ m) ∧
             n = 1240 :=
  sorry

end NUMINAMATH_CALUDE_least_four_digit_divisible_by_digits_l537_53720


namespace NUMINAMATH_CALUDE_isosceles_triangle_n_count_l537_53778

/-- The number of valid positive integer values for n in the isosceles triangle problem -/
def valid_n_count : ℕ := 7

/-- Checks if a given n satisfies the triangle inequality and angle conditions -/
def is_valid_n (n : ℕ) : Prop :=
  let ab := n + 10
  let bc := 4 * n + 2
  (ab + ab > bc) ∧ 
  (ab + bc > ab) ∧ 
  (bc + ab > ab) ∧
  (bc < ab)  -- This ensures ∠A > ∠B > ∠C in the isosceles triangle

theorem isosceles_triangle_n_count :
  (∃ (S : Finset ℕ), S.card = valid_n_count ∧ 
    (∀ n, n ∈ S ↔ (n > 0 ∧ is_valid_n n)) ∧
    (∀ n, n ∉ S → (n = 0 ∨ ¬is_valid_n n))) := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_n_count_l537_53778


namespace NUMINAMATH_CALUDE_x_intercept_of_line_A_l537_53744

/-- A line in the coordinate plane -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- The intersection point of two lines -/
structure IntersectionPoint where
  x : ℝ
  y : ℝ

/-- Theorem: The x-intercept of line A is 2 -/
theorem x_intercept_of_line_A (lineA lineB : Line) (intersection : IntersectionPoint) :
  lineA.slope = -1 →
  lineB.slope = 5 →
  lineB.yIntercept = -10 →
  intersection.x + intersection.y = 2 →
  lineA.yIntercept - lineA.slope * intersection.x = lineB.slope * intersection.x + lineB.yIntercept →
  lineA.yIntercept = 2 →
  -lineA.slope * 2 + lineA.yIntercept = 0 := by
  sorry

#check x_intercept_of_line_A

end NUMINAMATH_CALUDE_x_intercept_of_line_A_l537_53744


namespace NUMINAMATH_CALUDE_observed_pattern_solutions_for_20_l537_53738

/-- The number of integer solutions for |x| + |y| = n -/
def num_solutions (n : ℕ) : ℕ := 4 * n

/-- The observed pattern for the first three cases -/
theorem observed_pattern : 
  num_solutions 1 = 4 ∧ 
  num_solutions 2 = 8 ∧ 
  num_solutions 3 = 12 :=
sorry

/-- The main theorem: number of solutions for |x| + |y| = 20 -/
theorem solutions_for_20 : num_solutions 20 = 80 :=
sorry

end NUMINAMATH_CALUDE_observed_pattern_solutions_for_20_l537_53738


namespace NUMINAMATH_CALUDE_rectangular_plot_shorter_side_l537_53716

theorem rectangular_plot_shorter_side
  (width : ℝ)
  (num_poles : ℕ)
  (pole_distance : ℝ)
  (h1 : width = 50)
  (h2 : num_poles = 32)
  (h3 : pole_distance = 5)
  : ∃ (length : ℝ), length = 27.5 ∧ 2 * (length + width) = (num_poles - 1 : ℝ) * pole_distance :=
by sorry

end NUMINAMATH_CALUDE_rectangular_plot_shorter_side_l537_53716


namespace NUMINAMATH_CALUDE_solve_equation_l537_53781

theorem solve_equation : ∃ x : ℚ, (3 * x - 4) / 6 = 15 ∧ x = 94 / 3 := by sorry

end NUMINAMATH_CALUDE_solve_equation_l537_53781


namespace NUMINAMATH_CALUDE_equal_power_implies_equal_l537_53713

theorem equal_power_implies_equal (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 0 < a) (h4 : a < 1) (h5 : a^b = b^a) : a = b := by
  sorry

end NUMINAMATH_CALUDE_equal_power_implies_equal_l537_53713


namespace NUMINAMATH_CALUDE_system_solution_l537_53794

theorem system_solution : ∃ (x y z : ℝ), 
  (x^2 - 3*y + z = -4) ∧ 
  (x - 3*y + z^2 = -10) ∧ 
  (3*x + y^2 - 3*z = 0) ∧ 
  (x = -2) ∧ (y = 3) ∧ (z = 1) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l537_53794


namespace NUMINAMATH_CALUDE_triple_of_negative_two_l537_53758

theorem triple_of_negative_two : (3 : ℤ) * (-2 : ℤ) = -6 := by sorry

end NUMINAMATH_CALUDE_triple_of_negative_two_l537_53758


namespace NUMINAMATH_CALUDE_strawberry_plants_l537_53759

theorem strawberry_plants (initial : ℕ) : 
  (initial * 2 * 2 * 2 - 4 = 20) → initial = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_strawberry_plants_l537_53759


namespace NUMINAMATH_CALUDE_strawberry_weight_calculation_l537_53755

/-- Calculates the weight of Marco's dad's strawberries after losing some. -/
def dads_strawberry_weight (total_initial : ℕ) (lost : ℕ) (marcos : ℕ) : ℕ :=
  total_initial - lost - marcos

/-- Theorem: Given the initial total weight of strawberries, the weight of strawberries lost,
    and Marco's current weight of strawberries, Marco's dad's current weight of strawberries
    is equal to the difference between the remaining total weight and Marco's current weight. -/
theorem strawberry_weight_calculation
  (total_initial : ℕ)
  (lost : ℕ)
  (marcos : ℕ)
  (h1 : total_initial = 36)
  (h2 : lost = 8)
  (h3 : marcos = 12) :
  dads_strawberry_weight total_initial lost marcos = 16 :=
by sorry

end NUMINAMATH_CALUDE_strawberry_weight_calculation_l537_53755


namespace NUMINAMATH_CALUDE_sin_cos_identity_l537_53773

theorem sin_cos_identity : 
  Real.sin (18 * π / 180) * Real.sin (78 * π / 180) - 
  Real.cos (162 * π / 180) * Real.cos (78 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l537_53773


namespace NUMINAMATH_CALUDE_axels_alphabets_l537_53719

theorem axels_alphabets (total_alphabets : ℕ) (repetitions : ℕ) (different_alphabets : ℕ) : 
  total_alphabets = 10 ∧ repetitions = 2 → different_alphabets = 5 :=
by sorry

end NUMINAMATH_CALUDE_axels_alphabets_l537_53719


namespace NUMINAMATH_CALUDE_complex_equation_solution_l537_53751

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_equation_solution :
  ∃ (x : ℂ), (2 : ℂ) + 3 * i * x = (4 : ℂ) - 5 * i * x ∧ x = i / 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l537_53751


namespace NUMINAMATH_CALUDE_solve_inequality_find_range_of_a_l537_53772

-- Define the functions f and g
def f (x a : ℝ) : ℝ := |x - 1| + |x + a|
def g (a : ℝ) : ℝ := a^2 - a - 2

-- Theorem for part (1)
theorem solve_inequality (x : ℝ) :
  let a : ℝ := 3
  f x a > g a + 2 ↔ x < -4 ∨ x > 2 := by sorry

-- Theorem for part (2)
theorem find_range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-a) 1, f x a ≤ g a) ↔ a ≥ 3 := by sorry

end NUMINAMATH_CALUDE_solve_inequality_find_range_of_a_l537_53772


namespace NUMINAMATH_CALUDE_true_compound_propositions_l537_53777

-- Define propositions
variable (p₁ p₂ p₃ p₄ : Prop)

-- Define the truth values of the propositions
axiom p₁_true : p₁
axiom p₂_false : ¬p₂
axiom p₃_false : ¬p₃
axiom p₄_true : p₄

-- Theorem to prove
theorem true_compound_propositions :
  (p₁ ∧ p₄) ∧ (¬p₂ ∨ p₃) ∧ (¬p₃ ∨ ¬p₄) ∧ ¬(p₁ ∧ p₂) :=
by sorry

end NUMINAMATH_CALUDE_true_compound_propositions_l537_53777


namespace NUMINAMATH_CALUDE_product_of_sum_and_difference_l537_53707

theorem product_of_sum_and_difference (x y : ℝ) : 
  x + y = 26 → x - y = 8 → x * y = 153 := by
sorry

end NUMINAMATH_CALUDE_product_of_sum_and_difference_l537_53707


namespace NUMINAMATH_CALUDE_metallic_sheet_length_l537_53702

/-- Represents the dimensions and properties of a metallic sheet and the box formed from it. -/
structure MetallicSheet where
  width : ℝ
  length : ℝ
  cutSize : ℝ
  boxVolume : ℝ

/-- Theorem stating that given the conditions of the problem, the length of the sheet is 48 m. -/
theorem metallic_sheet_length
  (sheet : MetallicSheet)
  (h1 : sheet.width = 36)
  (h2 : sheet.cutSize = 8)
  (h3 : sheet.boxVolume = 5120)
  (h4 : sheet.boxVolume = (sheet.length - 2 * sheet.cutSize) * (sheet.width - 2 * sheet.cutSize) * sheet.cutSize) :
  sheet.length = 48 := by
  sorry

#check metallic_sheet_length

end NUMINAMATH_CALUDE_metallic_sheet_length_l537_53702


namespace NUMINAMATH_CALUDE_total_valid_words_count_l537_53739

def alphabet_size : ℕ := 25
def max_word_length : ℕ := 5

def valid_words (n : ℕ) : ℕ :=
  if n < 2 then 0
  else if n = 2 then 2
  else Nat.choose n 2 * alphabet_size ^ (n - 2)

def total_valid_words : ℕ :=
  (List.range (max_word_length - 1)).map (fun i => valid_words (i + 2))
    |> List.sum

theorem total_valid_words_count :
  total_valid_words = 160075 := by sorry

end NUMINAMATH_CALUDE_total_valid_words_count_l537_53739


namespace NUMINAMATH_CALUDE_count_integers_with_repeated_digits_is_140_l537_53757

/-- A function that counts the number of positive three-digit integers 
    between 500 and 999 with at least two identical digits -/
def count_integers_with_repeated_digits : ℕ :=
  let range_start := 500
  let range_end := 999
  let digits := 3
  -- Count of integers where last two digits are the same
  let case1 := 10 * (range_end.div 100 - range_start.div 100 + 1)
  -- Count of integers where first two digits are the same (and different from third)
  let case2 := (range_end.div 100 - range_start.div 100 + 1) * (digits - 1)
  -- Count of integers where first and third digits are the same (and different from second)
  let case3 := (range_end.div 100 - range_start.div 100 + 1) * (digits - 1)
  case1 + case2 + case3

/-- Theorem stating that the count of integers with repeated digits is 140 -/
theorem count_integers_with_repeated_digits_is_140 :
  count_integers_with_repeated_digits = 140 := by
  sorry

end NUMINAMATH_CALUDE_count_integers_with_repeated_digits_is_140_l537_53757


namespace NUMINAMATH_CALUDE_walnut_trees_after_planting_l537_53750

/-- The number of walnut trees in the park after planting -/
def total_walnut_trees (initial : ℕ) (planted : ℕ) : ℕ :=
  initial + planted

/-- Theorem stating the total number of walnut trees after planting -/
theorem walnut_trees_after_planting :
  total_walnut_trees 22 33 = 55 := by
  sorry

end NUMINAMATH_CALUDE_walnut_trees_after_planting_l537_53750


namespace NUMINAMATH_CALUDE_square_areas_and_perimeters_l537_53741

theorem square_areas_and_perimeters (x : ℝ) : 
  (x^2 + 4*x + 4) > 0 ∧ 
  (4*x^2 - 12*x + 9) > 0 ∧ 
  4 * (x + 2) + 4 * (2*x - 3) = 32 → 
  x = 3 := by sorry

end NUMINAMATH_CALUDE_square_areas_and_perimeters_l537_53741


namespace NUMINAMATH_CALUDE_min_value_product_l537_53785

theorem min_value_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 1) :
  (x + 3 * y) * (y + 3 * z) * (2 * x * z + 1) ≥ 24 * Real.sqrt 2 ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 1 ∧
    (x₀ + 3 * y₀) * (y₀ + 3 * z₀) * (2 * x₀ * z₀ + 1) = 24 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_product_l537_53785


namespace NUMINAMATH_CALUDE_marbles_distribution_l537_53708

/-- Given a total number of marbles and a number of boxes, calculate the number of marbles per box -/
def marblesPerBox (totalMarbles : ℕ) (numBoxes : ℕ) : ℕ :=
  totalMarbles / numBoxes

theorem marbles_distribution (totalMarbles : ℕ) (numBoxes : ℕ) 
  (h1 : totalMarbles = 18) (h2 : numBoxes = 3) :
  marblesPerBox totalMarbles numBoxes = 6 := by
  sorry

end NUMINAMATH_CALUDE_marbles_distribution_l537_53708


namespace NUMINAMATH_CALUDE_cos_theta_value_l537_53780

theorem cos_theta_value (θ : Real) 
  (h1 : 0 ≤ θ ∧ θ ≤ π/2) 
  (h2 : Real.sin (θ - π/6) = 1/3) : 
  Real.cos θ = (2 * Real.sqrt 6 - 1) / 6 := by
  sorry

end NUMINAMATH_CALUDE_cos_theta_value_l537_53780


namespace NUMINAMATH_CALUDE_parabola_coordinate_transform_l537_53710

/-- Given a parabola y = 2x² in the original coordinate system,
    prove that its equation in a new coordinate system where
    the x-axis is moved up by 2 units and the y-axis is moved
    right by 2 units is y = 2(x+2)² - 2 -/
theorem parabola_coordinate_transform :
  ∀ (x y : ℝ),
  (y = 2 * x^2) →
  (∃ (x' y' : ℝ),
    x' = x + 2 ∧
    y' = y - 2 ∧
    y' = 2 * (x' - 2)^2 - 2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_coordinate_transform_l537_53710


namespace NUMINAMATH_CALUDE_expected_heads_after_flips_l537_53766

def num_coins : ℕ := 64
def max_flips : ℕ := 4

def prob_heads_single_flip : ℚ := 1 / 2

def prob_heads_multiple_flips (n : ℕ) : ℚ :=
  1 - (1 - prob_heads_single_flip) ^ n

theorem expected_heads_after_flips :
  (num_coins : ℚ) * prob_heads_multiple_flips max_flips = 60 := by
  sorry

end NUMINAMATH_CALUDE_expected_heads_after_flips_l537_53766


namespace NUMINAMATH_CALUDE_megan_file_distribution_l537_53791

theorem megan_file_distribution (initial_files : ℕ) (deleted_files : ℕ) (num_folders : ℕ) 
  (h1 : initial_files = 93)
  (h2 : deleted_files = 21)
  (h3 : num_folders = 9)
  : (initial_files - deleted_files) / num_folders = 8 := by
  sorry

end NUMINAMATH_CALUDE_megan_file_distribution_l537_53791


namespace NUMINAMATH_CALUDE_min_throws_for_repeated_sum_l537_53701

/-- The minimum number of distinct sums possible when rolling three six-sided dice -/
def distinct_sums : ℕ := 16

/-- The minimum number of throws needed to guarantee a repeated sum -/
def min_throws : ℕ := distinct_sums + 1

/-- Theorem stating that the minimum number of throws to ensure a repeated sum is 17 -/
theorem min_throws_for_repeated_sum :
  min_throws = 17 := by sorry

end NUMINAMATH_CALUDE_min_throws_for_repeated_sum_l537_53701


namespace NUMINAMATH_CALUDE_system_solution_unique_l537_53749

theorem system_solution_unique :
  ∃! (x y : ℝ), x + 2*y = 5 ∧ 3*x + y = 5 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l537_53749


namespace NUMINAMATH_CALUDE_square_gt_of_abs_lt_l537_53782

theorem square_gt_of_abs_lt (a b : ℝ) : a > |b| → a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_gt_of_abs_lt_l537_53782


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l537_53711

/-- Represents a repeating decimal with a two-digit repetend -/
def RepeatingDecimal (a b : ℕ) : ℚ :=
  (10 * a + b : ℚ) / 99

/-- The repeating decimal 0.474747... -/
def x : ℚ := RepeatingDecimal 4 7

theorem repeating_decimal_sum : x = 47 / 99 ∧ 47 + 99 = 146 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l537_53711


namespace NUMINAMATH_CALUDE_equation_value_l537_53706

theorem equation_value (x y : ℝ) (eq1 : 2*x + y = 8) (eq2 : x + 2*y = 10) :
  8*x^2 + 10*x*y + 8*y^2 = 164 := by
  sorry

end NUMINAMATH_CALUDE_equation_value_l537_53706


namespace NUMINAMATH_CALUDE_total_apples_l537_53790

/-- Given 37 baskets with 17 apples each, prove that the total number of apples is 629. -/
theorem total_apples (baskets : ℕ) (apples_per_basket : ℕ) 
  (h1 : baskets = 37) (h2 : apples_per_basket = 17) : 
  baskets * apples_per_basket = 629 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_l537_53790


namespace NUMINAMATH_CALUDE_sqrt_sum_to_fraction_l537_53721

theorem sqrt_sum_to_fraction : 
  Real.sqrt ((25 : ℝ) / 36 + 16 / 9) = Real.sqrt 89 / 6 := by sorry

end NUMINAMATH_CALUDE_sqrt_sum_to_fraction_l537_53721


namespace NUMINAMATH_CALUDE_discount_percentage_proof_l537_53762

/-- Prove that the discount percentage is 10% given the conditions of the sale --/
theorem discount_percentage_proof (actual_sp cost_price : ℝ) (profit_rate : ℝ) : 
  actual_sp = 21000 ∧ 
  cost_price = 17500 ∧ 
  profit_rate = 0.08 → 
  (actual_sp - (cost_price * (1 + profit_rate))) / actual_sp = 0.1 := by
sorry

end NUMINAMATH_CALUDE_discount_percentage_proof_l537_53762


namespace NUMINAMATH_CALUDE_book_sale_gain_percentage_l537_53789

/-- Calculates the gain percentage when selling a book -/
def gain_percentage (cost_price selling_price : ℚ) : ℚ :=
  (selling_price - cost_price) / cost_price * 100

/-- Theorem about the gain percentage of a book sale -/
theorem book_sale_gain_percentage 
  (loss_price : ℚ) 
  (gain_price : ℚ) 
  (loss_percentage : ℚ) :
  loss_price = 450 →
  gain_price = 550 →
  loss_percentage = 10 →
  ∃ (cost_price : ℚ), 
    cost_price * (1 - loss_percentage / 100) = loss_price ∧
    gain_percentage cost_price gain_price = 10 := by
  sorry

#eval gain_percentage 500 550

end NUMINAMATH_CALUDE_book_sale_gain_percentage_l537_53789


namespace NUMINAMATH_CALUDE_strawberries_problem_l537_53724

/-- Converts kilograms to grams -/
def kg_to_g (kg : ℕ) : ℕ := kg * 1000

/-- Calculates the remaining strawberries in grams -/
def remaining_strawberries (initial_kg initial_g given_kg given_g : ℕ) : ℕ :=
  (kg_to_g initial_kg + initial_g) - (kg_to_g given_kg + given_g)

theorem strawberries_problem :
  remaining_strawberries 3 300 1 900 = 1400 := by
  sorry

end NUMINAMATH_CALUDE_strawberries_problem_l537_53724


namespace NUMINAMATH_CALUDE_six_distinct_areas_l537_53733

/-- Represents a point in a one-dimensional space -/
structure Point1D where
  x : ℝ

/-- Represents a line in a two-dimensional space -/
structure Line2D where
  points : List Point1D
  y : ℝ

/-- The configuration of points as described in the problem -/
structure PointConfiguration where
  line1 : Line2D
  line2 : Line2D
  w : Point1D
  x : Point1D
  y : Point1D
  z : Point1D
  p : Point1D
  q : Point1D

/-- Checks if the configuration satisfies the given conditions -/
def validConfiguration (config : PointConfiguration) : Prop :=
  config.w.x < config.x.x ∧ config.x.x < config.y.x ∧ config.y.x < config.z.x ∧
  config.x.x - config.w.x = 1 ∧
  config.y.x - config.x.x = 2 ∧
  config.z.x - config.y.x = 3 ∧
  config.q.x - config.p.x = 4 ∧
  config.line1.y ≠ config.line2.y ∧
  config.line1.points = [config.w, config.x, config.y, config.z] ∧
  config.line2.points = [config.p, config.q]

/-- Calculates the number of possible distinct triangle areas -/
def distinctTriangleAreas (config : PointConfiguration) : ℕ :=
  sorry

/-- The main theorem stating that there are exactly 6 possible distinct triangle areas -/
theorem six_distinct_areas (config : PointConfiguration) 
  (h : validConfiguration config) : distinctTriangleAreas config = 6 := by
  sorry

end NUMINAMATH_CALUDE_six_distinct_areas_l537_53733


namespace NUMINAMATH_CALUDE_arithmetic_sequence_20th_term_l537_53703

/-- An arithmetic sequence {a_n} satisfying given conditions -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_20th_term (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a)
  (h_sum1 : a 1 + a 3 + a 5 = 18)
  (h_sum2 : a 2 + a 4 + a 6 = 24) :
  a 20 = 40 := by
    sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_20th_term_l537_53703


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l537_53745

-- Define the lines l₁ and l₂
def l₁ (x y m : ℝ) : Prop := x + (1 + m) * y + (m - 2) = 0
def l₂ (x y m : ℝ) : Prop := m * x + 2 * y + 8 = 0

-- Define the parallel condition
def parallel (m : ℝ) : Prop := ∀ x y, l₁ x y m → l₂ x y m

-- Theorem statement
theorem parallel_lines_m_value (m : ℝ) : parallel m → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_m_value_l537_53745


namespace NUMINAMATH_CALUDE_parallel_planes_iff_parallel_lines_l537_53742

/-- Two lines are parallel -/
def parallel_lines (m n : Line) : Prop := sorry

/-- Two planes are parallel -/
def parallel_planes (α β : Plane) : Prop := sorry

/-- A line is perpendicular to a plane -/
def perpendicular_line_plane (l : Line) (π : Plane) : Prop := sorry

/-- Two objects are different -/
def different {α : Type*} (a b : α) : Prop := a ≠ b

theorem parallel_planes_iff_parallel_lines 
  (m n : Line) (α β : Plane) 
  (h1 : different m n)
  (h2 : different α β)
  (h3 : perpendicular_line_plane m β)
  (h4 : perpendicular_line_plane n β) :
  parallel_planes α β ↔ parallel_lines m n := by sorry

end NUMINAMATH_CALUDE_parallel_planes_iff_parallel_lines_l537_53742


namespace NUMINAMATH_CALUDE_value_of_x_l537_53747

theorem value_of_x : ∃ x : ℕ, x = 320 * 2 * 3 ∧ x = 1920 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l537_53747


namespace NUMINAMATH_CALUDE_max_triangles_is_eleven_l537_53730

/-- Represents an equilateral triangle with a line segment connecting midpoints of two sides -/
structure EquilateralTriangleWithMidline where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Represents the configuration of two overlapping equilateral triangles -/
structure OverlappingTriangles where
  triangle1 : EquilateralTriangleWithMidline
  triangle2 : EquilateralTriangleWithMidline
  overlap : ℝ -- Represents the degree of overlap between the triangles

/-- Counts the number of triangles formed in a given configuration -/
def countTriangles (config : OverlappingTriangles) : ℕ :=
  sorry

/-- Theorem stating that the maximum number of triangles is 11 -/
theorem max_triangles_is_eleven :
  ∃ (config : OverlappingTriangles),
    (∀ (other : OverlappingTriangles), countTriangles other ≤ countTriangles config) ∧
    countTriangles config = 11 :=
  sorry

end NUMINAMATH_CALUDE_max_triangles_is_eleven_l537_53730


namespace NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l537_53765

theorem smallest_sum_of_reciprocals (a b : ℕ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b)
  (h : (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 15) :
  ∀ (x y : ℕ), x > 0 ∧ y > 0 ∧ x ≠ y ∧ (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 15 → a + b ≤ x + y ∧ a + b = 64 := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l537_53765


namespace NUMINAMATH_CALUDE_repeating_decimal_subtraction_l537_53768

-- Define the repeating decimals
def repeating_246 : ℚ := 246 / 999
def repeating_135 : ℚ := 135 / 999
def repeating_579 : ℚ := 579 / 999

-- State the theorem
theorem repeating_decimal_subtraction :
  repeating_246 - repeating_135 - repeating_579 = -156 / 333 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_subtraction_l537_53768


namespace NUMINAMATH_CALUDE_hyperbola_equation_l537_53714

/-- Given an ellipse and a hyperbola with common foci and specified eccentricity,
    prove the equation of the hyperbola. -/
theorem hyperbola_equation (x y : ℝ) :
  (∃ a b : ℝ, a^2 = 12 ∧ b^2 = 3 ∧ x^2 / a^2 + y^2 / b^2 = 1) →  -- Ellipse equation
  (∃ c : ℝ, c^2 = 12 - 3) →                                    -- Foci distance
  (∃ e : ℝ, e = 3/2) →                                         -- Hyperbola eccentricity
  x^2 / 4 - y^2 / 5 = 1                                        -- Hyperbola equation
:= by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l537_53714


namespace NUMINAMATH_CALUDE_tuesday_rainfall_amount_l537_53783

/-- The amount of rainfall on Monday in inches -/
def monday_rain : ℝ := 0.9

/-- The difference in rainfall between Monday and Tuesday in inches -/
def rain_difference : ℝ := 0.7

/-- The amount of rainfall on Tuesday in inches -/
def tuesday_rain : ℝ := monday_rain - rain_difference

/-- Theorem stating that the amount of rain on Tuesday is 0.2 inches -/
theorem tuesday_rainfall_amount : tuesday_rain = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_tuesday_rainfall_amount_l537_53783


namespace NUMINAMATH_CALUDE_pear_juice_blend_percentage_l537_53743

/-- Represents the amount of juice extracted from a fruit -/
structure JuiceYield where
  fruit : String
  amount : ℚ
  count : ℕ

/-- Calculates the percentage of pear juice in a blend -/
def pear_juice_percentage (pear_yield orange_yield : JuiceYield) : ℚ :=
  let pear_juice := pear_yield.amount / pear_yield.count
  let orange_juice := orange_yield.amount / orange_yield.count
  let total_juice := pear_juice + orange_juice
  (pear_juice / total_juice) * 100

/-- Theorem: The percentage of pear juice in the blend is 40% -/
theorem pear_juice_blend_percentage :
  let pear_yield := JuiceYield.mk "pear" 8 3
  let orange_yield := JuiceYield.mk "orange" 8 2
  pear_juice_percentage pear_yield orange_yield = 40 := by
  sorry

end NUMINAMATH_CALUDE_pear_juice_blend_percentage_l537_53743


namespace NUMINAMATH_CALUDE_catalog_arrangements_l537_53726

theorem catalog_arrangements : 
  let n : ℕ := 7  -- number of letters in "catalog"
  Nat.factorial n = 5040 := by sorry

end NUMINAMATH_CALUDE_catalog_arrangements_l537_53726


namespace NUMINAMATH_CALUDE_total_new_games_l537_53734

/-- Given Katie's and her friends' game collections, prove the total number of new games they have together. -/
theorem total_new_games 
  (katie_new : ℕ) 
  (katie_percent : ℚ) 
  (friends_new : ℕ) 
  (friends_percent : ℚ) 
  (h1 : katie_new = 84) 
  (h2 : katie_percent = 75 / 100) 
  (h3 : friends_new = 8) 
  (h4 : friends_percent = 10 / 100) : 
  katie_new + friends_new = 92 := by
  sorry

end NUMINAMATH_CALUDE_total_new_games_l537_53734


namespace NUMINAMATH_CALUDE_businesspeople_neither_coffee_nor_tea_l537_53729

theorem businesspeople_neither_coffee_nor_tea 
  (total : ℕ) 
  (coffee : ℕ) 
  (tea : ℕ) 
  (both : ℕ) 
  (h1 : total = 35)
  (h2 : coffee = 18)
  (h3 : tea = 15)
  (h4 : both = 6) :
  total - (coffee + tea - both) = 8 := by
sorry

end NUMINAMATH_CALUDE_businesspeople_neither_coffee_nor_tea_l537_53729


namespace NUMINAMATH_CALUDE_case1_exists_case2_not_exists_l537_53761

-- Define a tetrahedron as a collection of 6 edge lengths
def Tetrahedron := Fin 6 → ℝ

-- Define the property of a valid tetrahedron
def is_valid_tetrahedron (t : Tetrahedron) : Prop := sorry

-- Define the conditions for case 1
def satisfies_case1 (t : Tetrahedron) : Prop :=
  (∃ i j, i ≠ j ∧ t i < 0.01 ∧ t j < 0.01) ∧
  (∃ a b c d, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧
    t a > 1000 ∧ t b > 1000 ∧ t c > 1000 ∧ t d > 1000)

-- Define the conditions for case 2
def satisfies_case2 (t : Tetrahedron) : Prop :=
  (∃ i j k l, i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ l ≠ i ∧
    t i < 0.01 ∧ t j < 0.01 ∧ t k < 0.01 ∧ t l < 0.01) ∧
  (∃ a b, a ≠ b ∧ t a > 1000 ∧ t b > 1000)

-- Theorem for case 1
theorem case1_exists :
  ∃ t : Tetrahedron, is_valid_tetrahedron t ∧ satisfies_case1 t := by sorry

-- Theorem for case 2
theorem case2_not_exists :
  ¬ ∃ t : Tetrahedron, is_valid_tetrahedron t ∧ satisfies_case2 t := by sorry

end NUMINAMATH_CALUDE_case1_exists_case2_not_exists_l537_53761


namespace NUMINAMATH_CALUDE_ratio_expression_equality_l537_53799

theorem ratio_expression_equality (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 1 / 3) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_expression_equality_l537_53799


namespace NUMINAMATH_CALUDE_andrea_pony_pasture_cost_l537_53776

/-- Calculates the monthly pasture cost for Andrea's pony --/
theorem andrea_pony_pasture_cost :
  let daily_food_cost : ℕ := 10
  let lesson_cost : ℕ := 60
  let lessons_per_week : ℕ := 2
  let total_annual_expense : ℕ := 15890
  let days_per_year : ℕ := 365
  let weeks_per_year : ℕ := 52

  let annual_food_cost := daily_food_cost * days_per_year
  let annual_lesson_cost := lesson_cost * lessons_per_week * weeks_per_year
  let annual_pasture_cost := total_annual_expense - (annual_food_cost + annual_lesson_cost)
  let monthly_pasture_cost := annual_pasture_cost / 12

  monthly_pasture_cost = 500 := by sorry

end NUMINAMATH_CALUDE_andrea_pony_pasture_cost_l537_53776


namespace NUMINAMATH_CALUDE_car_rental_cost_l537_53787

/-- The maximum daily rental cost for a car, given budget and mileage constraints -/
theorem car_rental_cost (budget : ℝ) (max_miles : ℝ) (cost_per_mile : ℝ) :
  budget = 88 ∧ max_miles = 190 ∧ cost_per_mile = 0.2 →
  ∃ (daily_rental : ℝ), daily_rental ≤ 50 ∧ daily_rental + max_miles * cost_per_mile ≤ budget :=
by sorry

end NUMINAMATH_CALUDE_car_rental_cost_l537_53787


namespace NUMINAMATH_CALUDE_minsu_age_is_15_l537_53717

/-- Minsu's age this year -/
def minsu_age : ℕ := 15

/-- Minsu's mother's age this year -/
def mother_age : ℕ := minsu_age + 28

/-- The age difference between Minsu and his mother is 28 years this year -/
axiom age_difference : mother_age = minsu_age + 28

/-- After 13 years, the mother's age will be twice Minsu's age -/
axiom future_age_relation : mother_age + 13 = 2 * (minsu_age + 13)

/-- Theorem: Minsu's age this year is 15 -/
theorem minsu_age_is_15 : minsu_age = 15 := by
  sorry

end NUMINAMATH_CALUDE_minsu_age_is_15_l537_53717


namespace NUMINAMATH_CALUDE_units_digit_of_expression_l537_53715

theorem units_digit_of_expression : 
  (2 * 21 * 2019 + 2^5 - 4^3) % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_expression_l537_53715


namespace NUMINAMATH_CALUDE_complex_equation_solution_l537_53722

theorem complex_equation_solution (a : ℝ) (i : ℂ) (h1 : i^2 = -1) (h2 : (1 + a*i)*i = 3 + i) : a = -3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l537_53722


namespace NUMINAMATH_CALUDE_perfect_square_natural_number_l537_53796

theorem perfect_square_natural_number (n : ℕ) :
  (∃ k : ℕ, n^2 + 5*n + 13 = k^2) → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_natural_number_l537_53796


namespace NUMINAMATH_CALUDE_murtha_pebble_collection_l537_53712

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Murtha's pebble collection problem -/
theorem murtha_pebble_collection : arithmetic_sum 1 2 12 = 144 := by
  sorry

end NUMINAMATH_CALUDE_murtha_pebble_collection_l537_53712


namespace NUMINAMATH_CALUDE_segment_length_segment_length_is_eight_l537_53798

theorem segment_length : ℝ → Prop :=
  fun length =>
    ∃ x₁ x₂ : ℝ,
      x₁ < x₂ ∧
      |x₁ - (27 : ℝ)^(1/3)| = 4 ∧
      |x₂ - (27 : ℝ)^(1/3)| = 4 ∧
      length = x₂ - x₁ ∧
      length = 8

theorem segment_length_is_eight : segment_length 8 := by
  sorry

end NUMINAMATH_CALUDE_segment_length_segment_length_is_eight_l537_53798


namespace NUMINAMATH_CALUDE_P_iff_Q_l537_53795

-- Define a triangle ABC with sides a, b, and c
structure Triangle :=
  (a b c : ℝ)
  (positive_sides : 0 < a ∧ 0 < b ∧ 0 < c)
  (triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b)

-- Define the condition P
def condition_P (t : Triangle) : Prop :=
  t.a^2 = t.b^2 + t.c^2

-- Define the condition Q
def condition_Q (t : Triangle) : Prop :=
  ∃ x : ℝ, (x^2 + 2*t.a*x + t.b^2 = 0) ∧ (x^2 + 2*t.c*x - t.b^2 = 0)

-- State the theorem
theorem P_iff_Q (t : Triangle) : condition_P t ↔ condition_Q t := by
  sorry

end NUMINAMATH_CALUDE_P_iff_Q_l537_53795


namespace NUMINAMATH_CALUDE_sqrt_2_times_2sqrt_2_plus_sqrt_5_bounds_l537_53754

theorem sqrt_2_times_2sqrt_2_plus_sqrt_5_bounds :
  7 < Real.sqrt 2 * (2 * Real.sqrt 2 + Real.sqrt 5) ∧
  Real.sqrt 2 * (2 * Real.sqrt 2 + Real.sqrt 5) < 8 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_2_times_2sqrt_2_plus_sqrt_5_bounds_l537_53754


namespace NUMINAMATH_CALUDE_abc_divisibility_problem_l537_53718

theorem abc_divisibility_problem (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 * b ∣ a^3 + b^3 + c^3) ∧
  (b^2 * c ∣ a^3 + b^3 + c^3) ∧
  (c^2 * a ∣ a^3 + b^3 + c^3) →
  ∃ k : ℕ, a = k ∧ b = k ∧ c = k := by
sorry

end NUMINAMATH_CALUDE_abc_divisibility_problem_l537_53718


namespace NUMINAMATH_CALUDE_count_dominoes_l537_53763

/-- The number of different (noncongruent) dominoes in an m × n array -/
def num_dominoes (m n : ℕ) : ℚ :=
  m * n - m^2 / 2 + m / 2 - 1

/-- Theorem: The number of different (noncongruent) dominoes in an m × n array -/
theorem count_dominoes (m n : ℕ) (h : 0 < m ∧ m ≤ n) :
  num_dominoes m n = m * n - m^2 / 2 + m / 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_count_dominoes_l537_53763


namespace NUMINAMATH_CALUDE_inverse_of_f_is_neg_g_neg_l537_53731

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the symmetry condition
def symmetric_wrt_x_plus_y_eq_0 (f g : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ g (-y) = -x

-- Define the inverse function
def has_inverse (f : ℝ → ℝ) : Prop :=
  ∃ g : ℝ → ℝ, ∀ x, g (f x) = x ∧ f (g x) = x

-- Theorem statement
theorem inverse_of_f_is_neg_g_neg (hf : has_inverse f) (h_sym : symmetric_wrt_x_plus_y_eq_0 f g) :
  ∃ f_inv : ℝ → ℝ, (∀ x, f_inv (f x) = x ∧ f (f_inv x) = x) ∧ (∀ x, f_inv x = -g (-x)) :=
sorry

end NUMINAMATH_CALUDE_inverse_of_f_is_neg_g_neg_l537_53731


namespace NUMINAMATH_CALUDE_min_cost_container_l537_53746

/-- Represents the cost function for a rectangular container -/
def cost_function (a b : ℝ) : ℝ := 20 * (a * b) + 10 * 2 * (a + b)

/-- Theorem stating the minimum cost for the container -/
theorem min_cost_container :
  ∀ a b : ℝ,
  a > 0 → b > 0 →
  a * b = 4 →
  cost_function a b ≥ 160 :=
by sorry

end NUMINAMATH_CALUDE_min_cost_container_l537_53746


namespace NUMINAMATH_CALUDE_inequality_solution_set_l537_53728

/-- The function representing the given inequality -/
def f (k x : ℝ) : ℝ := ((k^2 + 6*k + 14)*x - 9) * ((k^2 + 28)*x - 2*k^2 - 12*k)

/-- The solution set M for the inequality f(k, x) < 0 -/
def M (k : ℝ) : Set ℝ := {x | f k x < 0}

/-- The statement of the problem -/
theorem inequality_solution_set (k : ℝ) : 
  (M k ∩ Set.range (Int.cast : ℤ → ℝ) = {1}) ↔ (k < -14 ∨ (2 < k ∧ k ≤ 14/3)) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l537_53728


namespace NUMINAMATH_CALUDE_molecular_weight_correct_l537_53735

/-- The molecular weight of C6H8O7 in g/mol -/
def molecular_weight : ℝ := 192

/-- The number of moles used in the given condition -/
def given_moles : ℝ := 7

/-- The total weight of the given moles in g -/
def given_total_weight : ℝ := 1344

/-- Theorem: The molecular weight of C6H8O7 is correct given the condition -/
theorem molecular_weight_correct : 
  molecular_weight * given_moles = given_total_weight := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_correct_l537_53735


namespace NUMINAMATH_CALUDE_wedge_product_formula_l537_53760

/-- The wedge product of two 2D vectors -/
def wedge_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.2 - a.2 * b.1

/-- Theorem: The wedge product of two 2D vectors (a₁, a₂) and (b₁, b₂) is a₁b₂ - a₂b₁ -/
theorem wedge_product_formula (a b : ℝ × ℝ) : 
  wedge_product a b = a.1 * b.2 - a.2 * b.1 := by
  sorry

end NUMINAMATH_CALUDE_wedge_product_formula_l537_53760


namespace NUMINAMATH_CALUDE_coin_division_theorem_l537_53736

/-- Represents a collection of coins with their values -/
structure CoinCollection where
  num_coins : Nat
  coin_values : List Nat
  total_value : Nat

/-- Predicate to check if a coin collection can be divided into three equal groups -/
def can_divide_equally (cc : CoinCollection) : Prop :=
  ∃ (g1 g2 g3 : List Nat),
    g1 ++ g2 ++ g3 = cc.coin_values ∧
    g1.sum = g2.sum ∧ g2.sum = g3.sum

/-- Theorem stating that a specific coin collection can always be divided equally -/
theorem coin_division_theorem (cc : CoinCollection) 
    (h1 : cc.num_coins = 241)
    (h2 : cc.total_value = 360)
    (h3 : ∀ c ∈ cc.coin_values, c > 0)
    (h4 : cc.coin_values.length = cc.num_coins)
    (h5 : cc.coin_values.sum = cc.total_value) :
  can_divide_equally cc :=
sorry

end NUMINAMATH_CALUDE_coin_division_theorem_l537_53736


namespace NUMINAMATH_CALUDE_random_number_table_sampling_sequence_l537_53756

-- Define the steps as an enumeration
inductive SamplingStep
  | NumberIndividuals
  | ObtainSampleNumbers
  | SelectStartingNumber

-- Define the correct sequence
def correctSequence : List SamplingStep :=
  [SamplingStep.NumberIndividuals, SamplingStep.SelectStartingNumber, SamplingStep.ObtainSampleNumbers]

-- Theorem statement
theorem random_number_table_sampling_sequence :
  correctSequence = [SamplingStep.NumberIndividuals, SamplingStep.SelectStartingNumber, SamplingStep.ObtainSampleNumbers] :=
by sorry

end NUMINAMATH_CALUDE_random_number_table_sampling_sequence_l537_53756


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l537_53775

theorem polynomial_coefficient_sum (A B C D : ℝ) : 
  (∀ x : ℝ, (x - 3) * (2 * x^2 + 3 * x - 4) = A * x^3 + B * x^2 + C * x + D) → 
  A + B + C + D = -2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l537_53775


namespace NUMINAMATH_CALUDE_light_reflection_l537_53705

/-- Given a light ray emitted from point P (6, 4) intersecting the x-axis at point Q (2, 0)
    and reflecting off the x-axis, prove that the equations of the lines on which the
    incident and reflected rays lie are x - y - 2 = 0 and x + y - 2 = 0, respectively. -/
theorem light_reflection (P Q : ℝ × ℝ) : 
  P = (6, 4) → Q = (2, 0) → 
  ∃ (incident_ray reflected_ray : ℝ → ℝ → Prop),
    (∀ x y, incident_ray x y ↔ x - y - 2 = 0) ∧
    (∀ x y, reflected_ray x y ↔ x + y - 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_light_reflection_l537_53705


namespace NUMINAMATH_CALUDE_parakeets_fed_sixty_cups_l537_53753

/-- The number of parakeets fed with a given amount of bird seed -/
def parakeets_fed (cups : ℕ) : ℕ :=
  sorry

theorem parakeets_fed_sixty_cups :
  parakeets_fed 60 = 20 :=
by
  sorry

/-- Assumption: 30 cups of bird seed feed 10 parakeets for 5 days -/
axiom feed_ratio : parakeets_fed 30 = 10

/-- The number of parakeets fed is directly proportional to the amount of bird seed -/
axiom linear_feed : ∀ (c₁ c₂ : ℕ), c₁ ≠ 0 → c₂ ≠ 0 →
  (parakeets_fed c₁ : ℚ) / c₁ = (parakeets_fed c₂ : ℚ) / c₂

end NUMINAMATH_CALUDE_parakeets_fed_sixty_cups_l537_53753


namespace NUMINAMATH_CALUDE_routes_on_3x2_grid_l537_53704

/-- The number of routes on a grid from top-left to bottom-right -/
def num_routes (width : ℕ) (height : ℕ) : ℕ :=
  Nat.choose (width + height) height

/-- The theorem stating that the number of routes on a 3x2 grid is 10 -/
theorem routes_on_3x2_grid : num_routes 3 2 = 10 := by sorry

end NUMINAMATH_CALUDE_routes_on_3x2_grid_l537_53704


namespace NUMINAMATH_CALUDE_complex_magnitude_l537_53752

theorem complex_magnitude (a b : ℝ) :
  (Complex.I + a) * (1 - b * Complex.I) = 2 * Complex.I →
  Complex.abs (a + b * Complex.I) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l537_53752


namespace NUMINAMATH_CALUDE_set_representation_implies_sum_of_powers_l537_53764

theorem set_representation_implies_sum_of_powers (a b : ℝ) : 
  ({a, 1, b/a} : Set ℝ) = {a+b, 0, a^2} → a^2010 + b^2010 = 1 := by
  sorry

end NUMINAMATH_CALUDE_set_representation_implies_sum_of_powers_l537_53764


namespace NUMINAMATH_CALUDE_no_cube_root_sum_prime_l537_53767

theorem no_cube_root_sum_prime (x y p : ℕ+) (hp : Nat.Prime p.val) :
  (x.val : ℝ)^(1/3) + (y.val : ℝ)^(1/3) ≠ (p.val : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_no_cube_root_sum_prime_l537_53767


namespace NUMINAMATH_CALUDE_smallest_prime_after_six_nonprimes_l537_53784

/-- A function that returns true if a natural number is prime, false otherwise -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- A function that checks if there are six consecutive nonprime numbers before a given number -/
def hasSixConsecutiveNonprimes (n : ℕ) : Prop :=
  ∀ k : ℕ, n - 6 ≤ k → k < n → ¬(isPrime k)

/-- Theorem stating that 97 is the smallest prime number after six consecutive nonprimes -/
theorem smallest_prime_after_six_nonprimes :
  isPrime 97 ∧ hasSixConsecutiveNonprimes 97 ∧
  ∀ m : ℕ, m < 97 → ¬(isPrime m ∧ hasSixConsecutiveNonprimes m) :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_after_six_nonprimes_l537_53784


namespace NUMINAMATH_CALUDE_previous_average_production_l537_53797

theorem previous_average_production (n : ℕ) (today_production : ℕ) (new_average : ℚ) :
  n = 9 →
  today_production = 100 →
  new_average = 55 →
  let previous_total := n * (((n + 1) : ℚ) * new_average - today_production) / n
  previous_total / n = 50 := by sorry

end NUMINAMATH_CALUDE_previous_average_production_l537_53797
