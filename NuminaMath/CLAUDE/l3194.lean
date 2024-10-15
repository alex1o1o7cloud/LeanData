import Mathlib

namespace NUMINAMATH_CALUDE_fence_cost_calculation_l3194_319491

def parallel_side1_length : ℕ := 25
def parallel_side2_length : ℕ := 37
def non_parallel_side1_length : ℕ := 20
def non_parallel_side2_length : ℕ := 24
def parallel_side_price : ℕ := 48
def non_parallel_side_price : ℕ := 60

theorem fence_cost_calculation :
  (parallel_side1_length * parallel_side_price) +
  (parallel_side2_length * parallel_side_price) +
  (non_parallel_side1_length * non_parallel_side_price) +
  (non_parallel_side2_length * non_parallel_side_price) = 5616 := by
  sorry

end NUMINAMATH_CALUDE_fence_cost_calculation_l3194_319491


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3194_319459

-- Problem 1
theorem problem_1 (x y z : ℝ) :
  2 * x^3 * y^2 * (-2 * x * y^2 * z)^2 = 8 * x^5 * y^6 * z^2 := by
  sorry

-- Problem 2
theorem problem_2 (x : ℝ) :
  (-2 * x^2)^3 + x^2 * x^4 - (-3 * x^3)^2 = -16 * x^6 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3194_319459


namespace NUMINAMATH_CALUDE_standard_deviation_best_dispersion_measure_l3194_319443

-- Define the possible measures of central tendency and dispersion
inductive DataMeasure
  | Mode
  | Mean
  | StandardDeviation
  | Range

-- Define a function to determine if a measure reflects dispersion
def reflectsDispersion (measure : DataMeasure) : Prop :=
  match measure with
  | DataMeasure.StandardDeviation => true
  | _ => false

-- Theorem stating that standard deviation is the best measure of dispersion
theorem standard_deviation_best_dispersion_measure :
  ∀ (measure : DataMeasure),
    reflectsDispersion measure ↔ measure = DataMeasure.StandardDeviation :=
by sorry

end NUMINAMATH_CALUDE_standard_deviation_best_dispersion_measure_l3194_319443


namespace NUMINAMATH_CALUDE_mysterious_number_properties_l3194_319418

/-- A positive integer that can be expressed as the difference of the squares of two consecutive even numbers. -/
def MysteriousNumber (n : ℕ) : Prop :=
  ∃ k : ℕ, n = (2*k + 2)^2 - (2*k)^2 ∧ k ≥ 0

theorem mysterious_number_properties :
  (MysteriousNumber 28 ∧ MysteriousNumber 2020) ∧
  (∀ k : ℕ, (2*k + 2)^2 - (2*k)^2 % 4 = 0) ∧
  (∀ k : ℕ, ¬MysteriousNumber ((2*k + 1)^2 - (2*k - 1)^2)) :=
by sorry

end NUMINAMATH_CALUDE_mysterious_number_properties_l3194_319418


namespace NUMINAMATH_CALUDE_binary_1010_equals_decimal_10_l3194_319479

def binary_to_decimal (b : List Bool) : ℕ :=
  (List.zip b (List.reverse (List.range b.length))).foldl
    (fun acc (digit, power) => acc + if digit then 2^power else 0) 0

theorem binary_1010_equals_decimal_10 :
  binary_to_decimal [true, false, true, false] = 10 := by
  sorry

end NUMINAMATH_CALUDE_binary_1010_equals_decimal_10_l3194_319479


namespace NUMINAMATH_CALUDE_principal_calculation_l3194_319482

/-- Calculates the principal given simple interest, rate, and time -/
def calculate_principal (simple_interest : ℚ) (rate : ℚ) (time : ℕ) : ℚ :=
  simple_interest * 100 / (rate * time)

/-- Proves that the given conditions result in the correct principal -/
theorem principal_calculation :
  let simple_interest : ℚ := 4016.25
  let rate : ℚ := 9
  let time : ℕ := 5
  calculate_principal simple_interest rate time = 8925 := by
  sorry

end NUMINAMATH_CALUDE_principal_calculation_l3194_319482


namespace NUMINAMATH_CALUDE_power_of_product_l3194_319434

theorem power_of_product (a : ℝ) : (2 * a) ^ 3 = 8 * a ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l3194_319434


namespace NUMINAMATH_CALUDE_opera_house_seats_l3194_319480

theorem opera_house_seats (rows : ℕ) (revenue : ℕ) (ticket_price : ℕ) (occupancy_rate : ℚ) :
  rows = 150 →
  revenue = 12000 →
  ticket_price = 10 →
  occupancy_rate = 4/5 →
  ∃ (seats_per_row : ℕ), seats_per_row = 10 ∧ 
    (revenue / ticket_price : ℚ) = (occupancy_rate * (rows * seats_per_row : ℚ)) :=
by sorry

end NUMINAMATH_CALUDE_opera_house_seats_l3194_319480


namespace NUMINAMATH_CALUDE_power_of_power_at_three_l3194_319437

theorem power_of_power_at_three : (3^3)^(3^3) = 27^27 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_at_three_l3194_319437


namespace NUMINAMATH_CALUDE_max_value_trig_function_l3194_319465

theorem max_value_trig_function :
  ∃ M : ℝ, M = -1/2 ∧
  (∀ x : ℝ, 2 * Real.sin x ^ 2 + 2 * Real.cos x - 3 ≤ M) ∧
  ∀ ε > 0, ∃ x : ℝ, 2 * Real.sin x ^ 2 + 2 * Real.cos x - 3 > M - ε :=
by sorry

end NUMINAMATH_CALUDE_max_value_trig_function_l3194_319465


namespace NUMINAMATH_CALUDE_malcolm_followers_l3194_319466

def total_followers (instagram : ℕ) (facebook : ℕ) : ℕ :=
  let twitter := (instagram + facebook) / 2
  let tiktok := 3 * twitter
  let youtube := tiktok + 510
  instagram + facebook + twitter + tiktok + youtube

theorem malcolm_followers : total_followers 240 500 = 3840 := by
  sorry

end NUMINAMATH_CALUDE_malcolm_followers_l3194_319466


namespace NUMINAMATH_CALUDE_same_side_theorem_l3194_319412

/-- The set of values for parameter a where points A and B lie on the same side of the line 2x - y = 5 -/
def same_side_values : Set ℝ :=
  {a : ℝ | a ∈ Set.Ioo (-5/2) (-1/2) ∪ Set.Ioo 0 3}

/-- The equation of point A in the plane -/
def point_A_equation (a x y : ℝ) : Prop :=
  5 * a^2 - 4 * a * y + 8 * x^2 - 4 * x * y + y^2 + 12 * a * x = 0

/-- The equation of the parabola with vertex B -/
def parabola_B_equation (a x y : ℝ) : Prop :=
  a * x^2 - 2 * a^2 * x - a * y + a^3 + 3 = 0

/-- The line equation 2x - y = 5 -/
def line_equation (x y : ℝ) : Prop :=
  2 * x - y = 5

theorem same_side_theorem (a : ℝ) :
  (∃ x y : ℝ, point_A_equation a x y) ∧
  (∃ x y : ℝ, parabola_B_equation a x y) ∧
  (∀ x y : ℝ, point_A_equation a x y → ¬line_equation x y) ∧
  (∀ x y : ℝ, parabola_B_equation a x y → ¬line_equation x y) →
  (a ∈ same_side_values ↔
    (∃ x₁ y₁ x₂ y₂ : ℝ,
      point_A_equation a x₁ y₁ ∧
      parabola_B_equation a x₂ y₂ ∧
      (2 * x₁ - y₁ - 5) * (2 * x₂ - y₂ - 5) > 0)) :=
sorry

end NUMINAMATH_CALUDE_same_side_theorem_l3194_319412


namespace NUMINAMATH_CALUDE_rebecca_earnings_l3194_319451

/-- Rebecca's hair salon earnings calculation --/
theorem rebecca_earnings : 
  let haircut_price : ℕ := 30
  let perm_price : ℕ := 40
  let dye_job_price : ℕ := 60
  let dye_cost : ℕ := 10
  let haircut_count : ℕ := 4
  let perm_count : ℕ := 1
  let dye_job_count : ℕ := 2
  let tips : ℕ := 50
  
  haircut_price * haircut_count + 
  perm_price * perm_count + 
  (dye_job_price - dye_cost) * dye_job_count + 
  tips = 310 :=
by
  sorry


end NUMINAMATH_CALUDE_rebecca_earnings_l3194_319451


namespace NUMINAMATH_CALUDE_road_length_difference_l3194_319429

/-- The length of Telegraph Road in kilometers -/
def telegraph_road_length : ℝ := 162

/-- The length of Pardee Road in meters -/
def pardee_road_length : ℝ := 12000

/-- Conversion factor from kilometers to meters -/
def km_to_m : ℝ := 1000

theorem road_length_difference :
  (telegraph_road_length * km_to_m - pardee_road_length) / km_to_m = 150 := by
  sorry

end NUMINAMATH_CALUDE_road_length_difference_l3194_319429


namespace NUMINAMATH_CALUDE_square_difference_of_sums_l3194_319446

theorem square_difference_of_sums (a b : ℝ) :
  a = Real.sqrt 3 + Real.sqrt 2 →
  b = Real.sqrt 3 - Real.sqrt 2 →
  a^2 - b^2 = 4 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_of_sums_l3194_319446


namespace NUMINAMATH_CALUDE_simplify_K_simplify_L_l3194_319452

-- Part (a)
theorem simplify_K (x y : ℝ) (h : x ≥ y^2) :
  Real.sqrt (x + 2*y*Real.sqrt (x - y^2)) + Real.sqrt (x - 2*y*Real.sqrt (x - y^2)) = 
  max (2*abs y) (2*Real.sqrt (x - y^2)) := by sorry

-- Part (b)
theorem simplify_L (x y z : ℝ) (h : x*y + y*z + z*x = 1) :
  (2*x*y*z) / Real.sqrt ((1 + x^2)*(1 + y^2)*(1 + z^2)) = 
  (2*x*y*z) / abs (x + y + z - x*y*z) := by sorry

end NUMINAMATH_CALUDE_simplify_K_simplify_L_l3194_319452


namespace NUMINAMATH_CALUDE_area_between_concentric_circles_l3194_319414

/-- Given two concentric circles where a chord of length 120 units is tangent to the smaller circle
    with radius 40 units, the area between the circles is 3600π square units. -/
theorem area_between_concentric_circles :
  ∀ (r R : ℝ) (chord_length : ℝ),
  r = 40 →
  chord_length = 120 →
  chord_length^2 = 4 * R * r →
  (R^2 - r^2) * π = 3600 * π :=
by sorry

end NUMINAMATH_CALUDE_area_between_concentric_circles_l3194_319414


namespace NUMINAMATH_CALUDE_smaller_number_l3194_319464

theorem smaller_number (u v : ℝ) (hu : u > 0) (hv : v > 0) 
  (h_ratio : u / v = 3 / 5) (h_sum : u + v = 16) : 
  min u v = 6 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_l3194_319464


namespace NUMINAMATH_CALUDE_inequality_proof_l3194_319420

theorem inequality_proof : (1/2: ℝ)^(2/3) < (1/2: ℝ)^(1/3) ∧ (1/2: ℝ)^(1/3) < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3194_319420


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3194_319415

/-- Two vectors in ℝ² are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ, parallel (4, x) (-4, 4) → x = -4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3194_319415


namespace NUMINAMATH_CALUDE_investment_problem_l3194_319442

theorem investment_problem (P : ℝ) : 
  (P * 0.15 * 2 - P * 0.12 * 2 = 840) → P = 14000 := by
  sorry

end NUMINAMATH_CALUDE_investment_problem_l3194_319442


namespace NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l3194_319486

theorem sum_of_solutions_is_zero : 
  ∃ (x₁ x₂ : ℝ), (6 * x₁) / 30 = 8 / x₁ ∧ 
                 (6 * x₂) / 30 = 8 / x₂ ∧ 
                 x₁ + x₂ = 0 ∧
                 ∀ (y : ℝ), (6 * y) / 30 = 8 / y → y = x₁ ∨ y = x₂ := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l3194_319486


namespace NUMINAMATH_CALUDE_negation_equivalence_l3194_319439

-- Define the universe of discourse
variable (U : Type)

-- Define predicates for "man" and "tall"
variable (man : U → Prop)
variable (tall : U → Prop)

-- Define the original statement "all men are tall"
def all_men_are_tall : Prop := ∀ x : U, man x → tall x

-- Define the negation of the original statement
def negation_of_all_men_are_tall : Prop := ¬(∀ x : U, man x → tall x)

-- Define "some men are short"
def some_men_are_short : Prop := ∃ x : U, man x ∧ ¬(tall x)

-- Theorem stating that the negation is equivalent to "some men are short"
theorem negation_equivalence : 
  negation_of_all_men_are_tall U man tall ↔ some_men_are_short U man tall :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3194_319439


namespace NUMINAMATH_CALUDE_min_perimeter_52_l3194_319444

/-- Represents the side lengths of the squares in the rectangle --/
structure SquareSides where
  a : ℕ
  b : ℕ

/-- Calculates the perimeter of a rectangle given its length and width --/
def rectangle_perimeter (length width : ℕ) : ℕ :=
  2 * (length + width)

/-- Represents the configuration of squares in the rectangle --/
def square_configuration (sides : SquareSides) : Prop :=
  ∃ (left_column middle_column right_column bottom_row : ℕ),
    left_column = 2 * sides.a + sides.b ∧
    middle_column = 3 * sides.a + sides.b ∧
    right_column = 12 * sides.a - 2 * sides.b ∧
    bottom_row = 8 * sides.a - sides.b ∧
    left_column > 0 ∧ middle_column > 0 ∧ right_column > 0 ∧ bottom_row > 0

theorem min_perimeter_52 :
  ∀ (sides : SquareSides),
    square_configuration sides →
    ∀ (length width : ℕ),
      length = 2 * sides.a + sides.b + 3 * sides.a + sides.b + 12 * sides.a - 2 * sides.b →
      width = 2 * sides.a + sides.b + 8 * sides.a - sides.b →
      rectangle_perimeter length width ≥ 52 :=
sorry

end NUMINAMATH_CALUDE_min_perimeter_52_l3194_319444


namespace NUMINAMATH_CALUDE_power_of_two_problem_l3194_319408

theorem power_of_two_problem (a b : ℕ+) 
  (h1 : (2 ^ a.val) ^ b.val = 2 ^ 2) 
  (h2 : 2 ^ a.val * 2 ^ b.val = 8) : 
  2 ^ a.val = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_problem_l3194_319408


namespace NUMINAMATH_CALUDE_crosswalk_lines_total_l3194_319456

theorem crosswalk_lines_total (num_intersections : ℕ) (crosswalks_per_intersection : ℕ) (lines_per_crosswalk : ℕ) : 
  num_intersections = 5 → 
  crosswalks_per_intersection = 4 → 
  lines_per_crosswalk = 20 → 
  num_intersections * crosswalks_per_intersection * lines_per_crosswalk = 400 :=
by sorry

end NUMINAMATH_CALUDE_crosswalk_lines_total_l3194_319456


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_positive_l3194_319427

theorem sum_of_reciprocals_positive (a b c d : ℝ) 
  (ha : |a| > 1) (hb : |b| > 1) (hc : |c| > 1) (hd : |d| > 1)
  (h_eq : a * b * c + a * b * d + a * c * d + b * c * d + a + b + c + d = 0) :
  1 / (a - 1) + 1 / (b - 1) + 1 / (c - 1) + 1 / (d - 1) > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_positive_l3194_319427


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l3194_319405

-- Define a point in 2D space
def point : ℝ × ℝ := (-8, 2)

-- Define the second quadrant
def second_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

-- Theorem statement
theorem point_in_second_quadrant :
  second_quadrant point := by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l3194_319405


namespace NUMINAMATH_CALUDE_solution_set_inequality_l3194_319410

theorem solution_set_inequality (x : ℝ) : (x - 1) * (3 - x) ≥ 0 ↔ 1 ≤ x ∧ x ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l3194_319410


namespace NUMINAMATH_CALUDE_perpendicular_vector_k_value_l3194_319400

theorem perpendicular_vector_k_value :
  let a : Fin 2 → ℝ := ![1, 1]
  let b : Fin 2 → ℝ := ![2, -3]
  ∀ k : ℝ, (k • a - 2 • b) • a = 0 → k = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vector_k_value_l3194_319400


namespace NUMINAMATH_CALUDE_june_earnings_l3194_319413

/-- Represents the number of clovers June picks -/
def total_clovers : ℕ := 200

/-- Represents the percentage of clovers with 3 petals -/
def three_petal_percentage : ℚ := 75 / 100

/-- Represents the percentage of clovers with 2 petals -/
def two_petal_percentage : ℚ := 24 / 100

/-- Represents the percentage of clovers with 4 petals -/
def four_petal_percentage : ℚ := 1 / 100

/-- Represents the payment in cents for each clover -/
def payment_per_clover : ℕ := 1

/-- Theorem stating that June earns 200 cents -/
theorem june_earnings : 
  (total_clovers * payment_per_clover : ℕ) = 200 := by
  sorry

end NUMINAMATH_CALUDE_june_earnings_l3194_319413


namespace NUMINAMATH_CALUDE_smallest_special_number_l3194_319462

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem smallest_special_number :
  ∀ n : ℕ,
    is_two_digit n →
    n % 6 = 0 →
    n % 3 = 0 →
    is_perfect_square (digit_product n) →
    30 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_special_number_l3194_319462


namespace NUMINAMATH_CALUDE_root_of_cubic_l3194_319421

theorem root_of_cubic (x₁ x₂ x₃ : ℝ) (p q r : ℝ) :
  (∀ x, x^3 + p*x^2 + q*x + r = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) →
  (Real.sqrt 2)^3 - 3*(Real.sqrt 2)^2*(Real.sqrt 2) + 7*(Real.sqrt 2) - 3*(Real.sqrt 2) = 0 :=
by sorry

end NUMINAMATH_CALUDE_root_of_cubic_l3194_319421


namespace NUMINAMATH_CALUDE_sum_product_equality_l3194_319401

theorem sum_product_equality : 1235 + 2346 + 3412 * 2 + 4124 = 15529 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_equality_l3194_319401


namespace NUMINAMATH_CALUDE_complex_sum_powers_of_i_l3194_319424

theorem complex_sum_powers_of_i : ∃ (i : ℂ), i^2 = -1 ∧ i + i^2 + i^3 + i^4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_powers_of_i_l3194_319424


namespace NUMINAMATH_CALUDE_square_equals_multiplication_l3194_319422

theorem square_equals_multiplication (a : ℝ) : a * a = a ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_square_equals_multiplication_l3194_319422


namespace NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l3194_319431

theorem square_sum_zero_implies_both_zero (a b : ℝ) : a^2 + b^2 = 0 → a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l3194_319431


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l3194_319406

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℚ)
  (h_arith : ArithmeticSequence a)
  (h_first : a 1 = 3/5)
  (h_ninth : a 9 = 2/3) :
  a 5 = 19/30 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l3194_319406


namespace NUMINAMATH_CALUDE_average_weight_solution_l3194_319499

def average_weight_problem (a b c : ℝ) : Prop :=
  (a + b) / 2 = 40 ∧
  (b + c) / 2 = 43 ∧
  b = 31

theorem average_weight_solution :
  ∀ a b c : ℝ, average_weight_problem a b c → (a + b + c) / 3 = 45 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_solution_l3194_319499


namespace NUMINAMATH_CALUDE_sarah_test_performance_l3194_319425

theorem sarah_test_performance :
  let test1_questions : ℕ := 30
  let test2_questions : ℕ := 20
  let test3_questions : ℕ := 50
  let test1_correct_rate : ℚ := 85 / 100
  let test2_correct_rate : ℚ := 75 / 100
  let test3_correct_rate : ℚ := 90 / 100
  let calculation_mistakes : ℕ := 3
  let total_questions := test1_questions + test2_questions + test3_questions
  let correct_before_mistakes := 
    (test1_correct_rate * test1_questions).ceil +
    (test2_correct_rate * test2_questions).floor +
    (test3_correct_rate * test3_questions).floor
  let correct_after_mistakes := correct_before_mistakes - calculation_mistakes
  (correct_after_mistakes : ℚ) / total_questions = 83 / 100 :=
by sorry

end NUMINAMATH_CALUDE_sarah_test_performance_l3194_319425


namespace NUMINAMATH_CALUDE_ellipse_properties_l3194_319470

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  f1 : ℝ × ℝ  -- Focus 1
  f2 : ℝ × ℝ  -- Focus 2
  p : ℝ × ℝ   -- Point on ellipse
  h1 : a > b
  h2 : b > 0
  h3 : (p.1^2 / a^2) + (p.2^2 / b^2) = 1  -- P is on the ellipse
  h4 : (p.1 - f1.1) * (p.1 - f2.1) + (p.2 - f1.2) * (p.2 - f2.2) = 0  -- PF₁ ⟂ PF₂
  h5 : (f1.1 - f2.1)^2 + (f1.2 - f2.2)^2 = 12  -- |F₁F₂| = 2√3
  h6 : abs ((p.1 - f1.1) * (p.2 - f2.2) - (p.2 - f1.2) * (p.1 - f2.1)) = 2  -- Area of triangle PF₁F₂ is 1

/-- The theorem to be proved -/
theorem ellipse_properties (e : Ellipse) :
  (e.a = 2 ∧ e.b = 1) ∧
  (∀ m : ℝ, ∃ A B : ℝ × ℝ,
    (A.1^2 / 4 + A.2^2 = 1) ∧
    (B.1^2 / 4 + B.2^2 = 1) ∧
    (A.2 + B.2 = A.1 + B.1 + 2*m) ↔
    -3 * Real.sqrt 5 / 5 < m ∧ m < 3 * Real.sqrt 5 / 5) :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l3194_319470


namespace NUMINAMATH_CALUDE_division_multiplication_equality_l3194_319432

theorem division_multiplication_equality : (0.45 / 0.005) * 0.1 = 9 := by
  sorry

end NUMINAMATH_CALUDE_division_multiplication_equality_l3194_319432


namespace NUMINAMATH_CALUDE_complete_square_sum_l3194_319485

theorem complete_square_sum (x : ℝ) :
  ∃ (a b c : ℤ), 
    a > 0 ∧
    (25 * x^2 + 30 * x - 55 = 0 ↔ (a * x + b)^2 = c) ∧
    a + b + c = -38 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_sum_l3194_319485


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3194_319441

/-- Arithmetic sequence properties -/
structure ArithmeticSequence where
  first_term : ℝ
  last_term : ℝ
  sum : ℝ
  num_terms : ℕ

/-- Theorem: Common difference of a specific arithmetic sequence -/
theorem arithmetic_sequence_common_difference
  (seq : ArithmeticSequence)
  (h1 : seq.first_term = 5)
  (h2 : seq.last_term = 45)
  (h3 : seq.sum = 250) :
  let d := (seq.last_term - seq.first_term) / (seq.num_terms - 1)
  d = 40 / 9 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3194_319441


namespace NUMINAMATH_CALUDE_mutually_exclusive_pairs_count_l3194_319449

-- Define the type for balls
inductive Ball : Type
| Red : Ball
| White : Ball

-- Define the type for events
inductive Event : Type
| AtLeastOneWhite : Event
| BothWhite : Event
| AtLeastOneRed : Event
| ExactlyOneWhite : Event
| ExactlyTwoWhite : Event
| BothRed : Event

-- Define a function to check if two events are mutually exclusive
def mutually_exclusive (e1 e2 : Event) : Prop := sorry

-- Define the bag of balls
def bag : Multiset Ball := sorry

-- Define the function to count mutually exclusive pairs
def count_mutually_exclusive_pairs (events : List (Event × Event)) : Nat := sorry

-- Main theorem
theorem mutually_exclusive_pairs_count :
  let events : List (Event × Event) := [
    (Event.AtLeastOneWhite, Event.BothWhite),
    (Event.AtLeastOneWhite, Event.AtLeastOneRed),
    (Event.ExactlyOneWhite, Event.ExactlyTwoWhite),
    (Event.AtLeastOneWhite, Event.BothRed)
  ]
  count_mutually_exclusive_pairs events = 2 := by sorry

end NUMINAMATH_CALUDE_mutually_exclusive_pairs_count_l3194_319449


namespace NUMINAMATH_CALUDE_water_polo_team_selection_l3194_319455

/-- The number of members in the water polo club -/
def total_members : ℕ := 18

/-- The number of players in the starting team -/
def team_size : ℕ := 8

/-- The number of field players (excluding captain and goalie) -/
def field_players : ℕ := 6

/-- Calculates the number of ways to choose the starting team -/
def choose_team : ℕ := total_members * (total_members - 1) * (Nat.choose (total_members - 2) field_players)

theorem water_polo_team_selection :
  choose_team = 2459528 :=
sorry

end NUMINAMATH_CALUDE_water_polo_team_selection_l3194_319455


namespace NUMINAMATH_CALUDE_altitude_inradius_inequality_l3194_319417

-- Define a triangle with altitudes and inradius
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a : ℝ
  h_b : ℝ
  h_c : ℝ
  r : ℝ
  h_a_positive : h_a > 0
  h_b_positive : h_b > 0
  h_c_positive : h_c > 0
  r_positive : r > 0
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

-- State the theorem
theorem altitude_inradius_inequality (t : Triangle) : t.h_a + 4 * t.h_b + 9 * t.h_c > 36 * t.r := by
  sorry

end NUMINAMATH_CALUDE_altitude_inradius_inequality_l3194_319417


namespace NUMINAMATH_CALUDE_impossibility_of_simultaneous_inequalities_l3194_319463

theorem impossibility_of_simultaneous_inequalities (a b c : ℝ) 
  (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hc : 0 < c ∧ c < 1) :
  ¬(a * (1 - b) > 1/4 ∧ b * (1 - c) > 1/4 ∧ c * (1 - a) > 1/4) := by
sorry

end NUMINAMATH_CALUDE_impossibility_of_simultaneous_inequalities_l3194_319463


namespace NUMINAMATH_CALUDE_leaps_per_meter_calculation_l3194_319487

/-- Represents the number of leaps in one meter given the relationships between strides, leaps, bounds, and meters. -/
def leaps_per_meter (x y z w u v : ℚ) : ℚ :=
  (u * w) / (v * z)

/-- Theorem stating that given the relationships between units, one meter equals (uw/vz) leaps. -/
theorem leaps_per_meter_calculation
  (x y z w u v : ℚ)
  (h1 : x * 1 = y)  -- x strides = y leaps
  (h2 : z * 1 = w)  -- z bounds = w leaps
  (h3 : u * 1 = v)  -- u bounds = v meters
  : leaps_per_meter x y z w u v = (u * w) / (v * z) := by
  sorry

#check leaps_per_meter_calculation

end NUMINAMATH_CALUDE_leaps_per_meter_calculation_l3194_319487


namespace NUMINAMATH_CALUDE_negation_of_existence_is_universal_l3194_319494

variable (a : ℝ)

theorem negation_of_existence_is_universal :
  (¬ ∃ x : ℝ, x^2 + a*x + 1 < 0) ↔ (∀ x : ℝ, x^2 + a*x + 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_is_universal_l3194_319494


namespace NUMINAMATH_CALUDE_max_curved_sides_l3194_319457

/-- A figure formed by the intersection of circles -/
structure IntersectionFigure where
  n : ℕ
  n_ge_two : n ≥ 2

/-- The number of curved sides in an intersection figure -/
def curved_sides (F : IntersectionFigure) : ℕ := 2 * F.n - 2

/-- The theorem stating the maximum number of curved sides -/
theorem max_curved_sides (F : IntersectionFigure) :
  curved_sides F ≤ 2 * F.n - 2 :=
sorry

end NUMINAMATH_CALUDE_max_curved_sides_l3194_319457


namespace NUMINAMATH_CALUDE_systematic_sample_theorem_l3194_319490

/-- Represents a systematic sample from a population -/
structure SystematicSample where
  population_size : ℕ
  sample_size : ℕ
  interval : ℕ
  first_element : ℕ

/-- Generates the seat numbers in a systematic sample -/
def generate_sample (s : SystematicSample) : List ℕ :=
  List.range s.sample_size |>.map (λ i => s.first_element + i * s.interval)

theorem systematic_sample_theorem (sample : SystematicSample)
  (h1 : sample.population_size = 48)
  (h2 : sample.sample_size = 4)
  (h3 : sample.interval = sample.population_size / sample.sample_size)
  (h4 : sample.first_element = 6)
  (h5 : 30 ∈ generate_sample sample)
  (h6 : 42 ∈ generate_sample sample) :
  18 ∈ generate_sample sample :=
sorry

end NUMINAMATH_CALUDE_systematic_sample_theorem_l3194_319490


namespace NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l3194_319489

/-- Represents the repeating decimal 0.53246246246... -/
def repeating_decimal : ℚ := 0.53 + (0.246 / 999)

/-- The denominator of the target fraction -/
def target_denominator : ℕ := 999900

theorem repeating_decimal_as_fraction :
  ∃ x : ℕ, (x : ℚ) / target_denominator = repeating_decimal ∧ x = 531714 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l3194_319489


namespace NUMINAMATH_CALUDE_diane_has_27_cents_l3194_319436

/-- The amount of money Diane has, given the cost of cookies and the additional amount needed. -/
def dianes_money (cookie_cost : ℕ) (additional_needed : ℕ) : ℕ :=
  cookie_cost - additional_needed

/-- Theorem stating that Diane has 27 cents given the problem conditions. -/
theorem diane_has_27_cents :
  dianes_money 65 38 = 27 := by
  sorry

end NUMINAMATH_CALUDE_diane_has_27_cents_l3194_319436


namespace NUMINAMATH_CALUDE_triangle_most_stable_l3194_319450

-- Define the shapes
inductive Shape
  | Rectangle
  | Trapezoid
  | Parallelogram
  | Triangle

-- Define stability as a property of shapes
def is_stable (s : Shape) : Prop :=
  match s with
  | Shape.Triangle => true
  | _ => false

-- Define the stability comparison
def more_stable (s1 s2 : Shape) : Prop :=
  is_stable s1 ∧ ¬is_stable s2

-- Theorem statement
theorem triangle_most_stable :
  ∀ s : Shape, s ≠ Shape.Triangle → more_stable Shape.Triangle s :=
sorry

end NUMINAMATH_CALUDE_triangle_most_stable_l3194_319450


namespace NUMINAMATH_CALUDE_jack_quarantine_days_l3194_319492

/-- Calculates the number of days spent in quarantine given the total wait time and customs time. -/
def quarantine_days (total_hours : ℕ) (customs_hours : ℕ) : ℕ :=
  (total_hours - customs_hours) / 24

/-- Theorem stating that given a total wait time of 356 hours, including 20 hours for customs,
    the number of days spent in quarantine is 14. -/
theorem jack_quarantine_days :
  quarantine_days 356 20 = 14 := by
  sorry

end NUMINAMATH_CALUDE_jack_quarantine_days_l3194_319492


namespace NUMINAMATH_CALUDE_equivalence_theorem_l3194_319435

theorem equivalence_theorem (x y z : ℝ) : 
  (x > 0 ∧ y > 0 ∧ z > 0 ∧ 1/x + 1/y + 1/z ≤ 1) ↔ 
  (∀ (a b c d : ℝ), (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a + b + c > d) → a^2*x + b^2*y + c^2*z > d^2) :=
by sorry

end NUMINAMATH_CALUDE_equivalence_theorem_l3194_319435


namespace NUMINAMATH_CALUDE_additional_wax_is_22_l3194_319458

/-- The amount of additional wax needed for painting feathers -/
def additional_wax_needed (total_wax : ℕ) (available_wax : ℕ) : ℕ :=
  total_wax - available_wax

/-- Theorem stating that the additional wax needed is 22 grams -/
theorem additional_wax_is_22 :
  additional_wax_needed 353 331 = 22 := by
  sorry

end NUMINAMATH_CALUDE_additional_wax_is_22_l3194_319458


namespace NUMINAMATH_CALUDE_hill_climb_time_l3194_319471

/-- Proves that the time taken to reach the top of the hill is 4 hours -/
theorem hill_climb_time (descent_time : ℝ) (avg_speed_total : ℝ) (avg_speed_climb : ℝ) :
  descent_time = 2 →
  avg_speed_total = 2 →
  avg_speed_climb = 1.5 →
  let ascent_time := 4
  let total_time := ascent_time + descent_time
  let total_distance := avg_speed_total * total_time
  let climb_distance := avg_speed_climb * ascent_time
  climb_distance * 2 = total_distance →
  ascent_time = 4 := by
  sorry

end NUMINAMATH_CALUDE_hill_climb_time_l3194_319471


namespace NUMINAMATH_CALUDE_seed_germination_percentage_l3194_319469

theorem seed_germination_percentage (seeds_plot1 seeds_plot2 : ℕ) 
  (germination_rate1 germination_rate2 : ℚ) :
  seeds_plot1 = 300 →
  seeds_plot2 = 200 →
  germination_rate1 = 25 / 100 →
  germination_rate2 = 30 / 100 →
  let total_seeds := seeds_plot1 + seeds_plot2
  let germinated_seeds1 := (seeds_plot1 : ℚ) * germination_rate1
  let germinated_seeds2 := (seeds_plot2 : ℚ) * germination_rate2
  let total_germinated := germinated_seeds1 + germinated_seeds2
  (total_germinated / total_seeds) * 100 = 27 := by
sorry

end NUMINAMATH_CALUDE_seed_germination_percentage_l3194_319469


namespace NUMINAMATH_CALUDE_pencil_pen_cost_l3194_319404

theorem pencil_pen_cost (pencil_cost pen_cost : ℝ) 
  (h1 : 3 * pencil_cost + 4 * pen_cost = 5.20)
  (h2 : 4 * pencil_cost + 3 * pen_cost = 4.90) : 
  pencil_cost + 3 * pen_cost = 3.1857 := by
  sorry

end NUMINAMATH_CALUDE_pencil_pen_cost_l3194_319404


namespace NUMINAMATH_CALUDE_irregular_polygon_rotation_implies_composite_l3194_319481

/-- An n-gon inscribed in a circle -/
structure InscribedPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  center : ℝ × ℝ
  radius : ℝ

/-- Rotation of a point about a center by an angle -/
def rotate (center : ℝ × ℝ) (angle : ℝ) (point : ℝ × ℝ) : ℝ × ℝ := sorry

/-- A polygon is irregular if not all its sides are equal -/
def isIrregular (p : InscribedPolygon n) : Prop := sorry

/-- A polygon coincides with itself under rotation -/
def coincidesSelfUnderRotation (p : InscribedPolygon n) (angle : ℝ) : Prop := sorry

/-- A number is composite if it's not prime and greater than 1 -/
def isComposite (n : ℕ) : Prop := ¬(Nat.Prime n) ∧ n > 1

theorem irregular_polygon_rotation_implies_composite 
  (n : ℕ) (p : InscribedPolygon n) (α : ℝ) :
  isIrregular p →
  α ≠ 2 * Real.pi →
  coincidesSelfUnderRotation p α →
  isComposite n := by
  sorry

end NUMINAMATH_CALUDE_irregular_polygon_rotation_implies_composite_l3194_319481


namespace NUMINAMATH_CALUDE_no_linear_term_implies_equal_coefficients_l3194_319440

theorem no_linear_term_implies_equal_coefficients (x m n : ℝ) : 
  (x + m) * (x - n) = x^2 + (-m * n) → m = n :=
by sorry

end NUMINAMATH_CALUDE_no_linear_term_implies_equal_coefficients_l3194_319440


namespace NUMINAMATH_CALUDE_treat_cost_theorem_l3194_319454

/-- Represents the cost of treats -/
structure TreatCost where
  chocolate : ℚ
  popsicle : ℚ
  lollipop : ℚ

/-- The cost relationships between treats -/
def cost_relationship (c : TreatCost) : Prop :=
  3 * c.chocolate = 2 * c.popsicle ∧ 2 * c.lollipop = 5 * c.chocolate

/-- The number of popsicles that can be bought with the money for 3 lollipops -/
def popsicles_for_lollipops (c : TreatCost) : ℚ :=
  (3 * c.lollipop) / c.popsicle

/-- The number of chocolates that can be bought with the money for 3 chocolates, 2 popsicles, and 2 lollipops -/
def chocolates_for_combination (c : TreatCost) : ℚ :=
  (3 * c.chocolate + 2 * c.popsicle + 2 * c.lollipop) / c.chocolate

theorem treat_cost_theorem (c : TreatCost) :
  cost_relationship c →
  popsicles_for_lollipops c = 5 ∧
  chocolates_for_combination c = 11 := by
  sorry

end NUMINAMATH_CALUDE_treat_cost_theorem_l3194_319454


namespace NUMINAMATH_CALUDE_smallest_total_students_l3194_319474

/-- The number of successful configuration days --/
def successful_days : ℕ := 14

/-- The maximum number of students per leader --/
def max_students_per_leader : ℕ := 12

/-- The number of students per leader on the first day --/
def first_day_students_per_leader : ℕ := 12

/-- The number of students per leader on the last successful day --/
def last_day_students_per_leader : ℕ := 5

/-- A function to check if a number satisfies all conditions --/
def satisfies_conditions (n : ℕ) : Prop :=
  (n % first_day_students_per_leader = 0) ∧
  (n % last_day_students_per_leader = 0) ∧
  (∃ (configs : Finset (Finset ℕ)), configs.card = successful_days ∧
    ∀ c ∈ configs, c.card > 0 ∧ c.card ≤ n ∧
    (∀ g ∈ c, g ≤ max_students_per_leader) ∧
    (n % c.card = 0))

theorem smallest_total_students :
  satisfies_conditions 360 ∧
  ∀ m < 360, ¬ satisfies_conditions m :=
sorry

end NUMINAMATH_CALUDE_smallest_total_students_l3194_319474


namespace NUMINAMATH_CALUDE_max_value_inequality_equality_at_six_strict_inequality_for_greater_than_six_l3194_319473

theorem max_value_inequality (a : ℝ) : 
  (∀ x > 1, (x^2 + 3) / (x - 1) ≥ a) → a ≤ 6 := by
  sorry

theorem equality_at_six : 
  ∃ x > 1, (x^2 + 3) / (x - 1) = 6 := by
  sorry

theorem strict_inequality_for_greater_than_six : 
  ∀ b > 6, ∃ x > 1, (x^2 + 3) / (x - 1) < b := by
  sorry

end NUMINAMATH_CALUDE_max_value_inequality_equality_at_six_strict_inequality_for_greater_than_six_l3194_319473


namespace NUMINAMATH_CALUDE_count_integers_in_range_l3194_319411

theorem count_integers_in_range : ∃ (S : Finset Int), 
  (∀ n : Int, n ∈ S ↔ 15 < n^2 ∧ n^2 < 120) ∧ Finset.card S = 14 := by
  sorry

end NUMINAMATH_CALUDE_count_integers_in_range_l3194_319411


namespace NUMINAMATH_CALUDE_cylinder_water_transfer_l3194_319419

theorem cylinder_water_transfer (h : ℝ) (r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let original_volume := π * r^2 * h
  let new_volume := π * (1.25 * r)^2 * (0.72 * h)
  (3/5) * new_volume = 0.675 * original_volume :=
by sorry

end NUMINAMATH_CALUDE_cylinder_water_transfer_l3194_319419


namespace NUMINAMATH_CALUDE_A_investment_l3194_319402

-- Define the investments and profit shares
def investment_B : ℝ := 10000
def investment_C : ℝ := 12000
def profit_share_B : ℝ := 2500
def profit_difference_AC : ℝ := 999.9999999999998

-- Define the theorem
theorem A_investment (investment_A : ℝ) : 
  (investment_A / investment_B * profit_share_B) - 
  (investment_C / investment_B * profit_share_B) = profit_difference_AC → 
  investment_A = 16000 := by
sorry

end NUMINAMATH_CALUDE_A_investment_l3194_319402


namespace NUMINAMATH_CALUDE_A_div_B_between_zero_and_one_l3194_319428

def A : ℕ := 1 * 2 * 7 + 2 * 4 * 14 + 3 * 6 * 21 + 4 * 8 * 28
def B : ℕ := 1 * 3 * 5 + 2 * 6 * 10 + 3 * 9 * 15 + 4 * 12 * 20

theorem A_div_B_between_zero_and_one : 0 < (A : ℚ) / B ∧ (A : ℚ) / B < 1 := by
  sorry

end NUMINAMATH_CALUDE_A_div_B_between_zero_and_one_l3194_319428


namespace NUMINAMATH_CALUDE_mary_fruit_difference_l3194_319472

/-- Proves that Mary has 33 fewer peaches than apples given the conditions about Jake, Steven, and Mary's fruits. -/
theorem mary_fruit_difference :
  ∀ (steven_apples steven_peaches jake_apples jake_peaches mary_apples mary_peaches : ℕ),
  steven_apples = 11 →
  steven_peaches = 18 →
  jake_peaches + 8 = steven_peaches →
  jake_apples = steven_apples + 10 →
  mary_apples = 2 * jake_apples →
  mary_peaches * 2 = steven_peaches →
  (mary_peaches : ℤ) - (mary_apples : ℤ) = -33 := by
sorry

end NUMINAMATH_CALUDE_mary_fruit_difference_l3194_319472


namespace NUMINAMATH_CALUDE_range_of_a_l3194_319426

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x, x^2 - (a-1)*x + 1 > 0

def q (a : ℝ) : Prop := ∀ x y, x < y → (a+1)^x < (a+1)^y

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (¬(p a ∧ q a) ∧ (p a ∨ q a)) →
  ((-1 < a ∧ a ≤ 0) ∨ (a ≥ 3)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3194_319426


namespace NUMINAMATH_CALUDE_num_routes_eq_binomial_num_routes_is_six_l3194_319448

/-- The number of different routes from the bottom-left corner to the top-right corner of a 2x2 grid,
    moving only upwards or to the right one square at a time. -/
def num_routes : ℕ := 6

/-- The size of the grid (2x2 in this case) -/
def grid_size : ℕ := 2

/-- The total number of moves required to reach the top-right corner from the bottom-left corner -/
def total_moves : ℕ := grid_size * 2

/-- Theorem stating that the number of routes is equal to the binomial coefficient (total_moves choose grid_size) -/
theorem num_routes_eq_binomial :
  num_routes = Nat.choose total_moves grid_size :=
by sorry

/-- Theorem proving that the number of routes is 6 -/
theorem num_routes_is_six :
  num_routes = 6 :=
by sorry

end NUMINAMATH_CALUDE_num_routes_eq_binomial_num_routes_is_six_l3194_319448


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l3194_319493

def B : Matrix (Fin 3) (Fin 3) ℚ := !![1, 2, 3; 0, 1, 2; 1, 0, 1]

theorem matrix_equation_solution :
  B^3 + (-5 : ℚ) • B^2 + 3 • B + (-6 : ℚ) • (1 : Matrix (Fin 3) (Fin 3) ℚ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l3194_319493


namespace NUMINAMATH_CALUDE_number_remainder_l3194_319423

theorem number_remainder (A : ℤ) (h : 9 * A + 1 = 10 * A - 100) : A % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_remainder_l3194_319423


namespace NUMINAMATH_CALUDE_hari_contribution_is_9000_l3194_319453

/-- Represents the business partnership between Praveen and Hari -/
structure Partnership where
  praveen_investment : ℕ
  praveen_months : ℕ
  hari_months : ℕ
  profit_ratio_praveen : ℕ
  profit_ratio_hari : ℕ

/-- Calculates Hari's contribution to the capital -/
def hari_contribution (p : Partnership) : ℕ :=
  (p.praveen_investment * p.praveen_months * p.profit_ratio_hari) / (p.hari_months * p.profit_ratio_praveen)

/-- Theorem stating that Hari's contribution is 9000 given the specified conditions -/
theorem hari_contribution_is_9000 :
  let p : Partnership := {
    praveen_investment := 3500,
    praveen_months := 12,
    hari_months := 7,
    profit_ratio_praveen := 2,
    profit_ratio_hari := 3
  }
  hari_contribution p = 9000 := by sorry

end NUMINAMATH_CALUDE_hari_contribution_is_9000_l3194_319453


namespace NUMINAMATH_CALUDE_systematic_sample_theorem_l3194_319484

/-- Represents a systematic sample from a class -/
structure SystematicSample where
  totalStudents : ℕ
  sampleSize : ℕ
  firstSample : ℕ
  knownSamples : Finset ℕ

/-- Checks if a number is part of the systematic sample -/
def isInSample (s : SystematicSample) (n : ℕ) : Prop :=
  ∃ k : ℕ, n = s.firstSample + k * (s.totalStudents / s.sampleSize)

theorem systematic_sample_theorem (s : SystematicSample)
  (h1 : s.totalStudents = 48)
  (h2 : s.sampleSize = 6)
  (h3 : s.firstSample = 5)
  (h4 : s.knownSamples = {5, 21, 29, 37, 45})
  (h5 : ∀ n ∈ s.knownSamples, isInSample s n) :
  isInSample s 13 ∧ (∀ n, isInSample s n → n = 13 ∨ n ∈ s.knownSamples) :=
sorry

end NUMINAMATH_CALUDE_systematic_sample_theorem_l3194_319484


namespace NUMINAMATH_CALUDE_triangle_side_length_l3194_319407

theorem triangle_side_length (A B C M : ℝ × ℝ) : 
  -- Triangle ABC is right-angled at C
  (B.1 - A.1) * (C.2 - A.2) = (C.1 - A.1) * (B.2 - A.2) →
  -- AC = BC
  ((C.1 - A.1)^2 + (C.2 - A.2)^2) = ((C.1 - B.1)^2 + (C.2 - B.2)^2) →
  -- M is an interior point (implied by the distances)
  -- MC = 1
  ((M.1 - C.1)^2 + (M.2 - C.2)^2) = 1 →
  -- MA = 2
  ((M.1 - A.1)^2 + (M.2 - A.2)^2) = 4 →
  -- MB = √2
  ((M.1 - B.1)^2 + (M.2 - B.2)^2) = 2 →
  -- AB = √10
  ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3194_319407


namespace NUMINAMATH_CALUDE_tan_sum_of_roots_l3194_319476

theorem tan_sum_of_roots (α β : Real) : 
  (∃ (x : Real), x^2 - 3 * Real.sqrt 3 * x + 4 = 0 ∧ x = Real.tan α) ∧
  (∃ (y : Real), y^2 - 3 * Real.sqrt 3 * y + 4 = 0 ∧ y = Real.tan β) ∧
  α ∈ Set.Ioo (-π/2) (π/2) ∧
  β ∈ Set.Ioo (-π/2) (π/2) →
  Real.tan (α + β) = -Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_tan_sum_of_roots_l3194_319476


namespace NUMINAMATH_CALUDE_complex_purely_imaginary_l3194_319460

/-- If m^2(1+i) + (m+i)i^2 is purely imaginary and m is a real number, then m = 0 -/
theorem complex_purely_imaginary (m : ℝ) : 
  (Complex.I * (m^2 - 1) = m^2*(1 + Complex.I) + (m + Complex.I)*Complex.I^2) → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_purely_imaginary_l3194_319460


namespace NUMINAMATH_CALUDE_optimal_pan_dimensions_l3194_319488

def is_valid_pan (m n : ℕ) : Prop :=
  (m - 2) * (n - 2) = 2 * m + 2 * n - 4

def perimeter (m n : ℕ) : ℕ := 2 * m + 2 * n

def area (m n : ℕ) : ℕ := m * n

theorem optimal_pan_dimensions :
  ∀ m n : ℕ, m > 2 ∧ n > 2 → is_valid_pan m n →
    (perimeter m n ≥ perimeter 6 8) ∧
    (perimeter m n = perimeter 6 8 → area m n ≤ area 6 8) ∧
    is_valid_pan 6 8 :=
by sorry

end NUMINAMATH_CALUDE_optimal_pan_dimensions_l3194_319488


namespace NUMINAMATH_CALUDE_complex_modulus_l3194_319475

theorem complex_modulus (z : ℂ) (h : (1 + Complex.I) * z = Complex.I - 1) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l3194_319475


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3194_319498

-- Define the geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 1 ∧ ∀ n : ℕ, a (n + 1) = r * a n

-- Define the theorem
theorem geometric_sequence_ratio
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_prod : a 3 * a 7 = 6)
  (h_sum : a 2 + a 8 = 5) :
  a 10 / a 4 = 3/2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3194_319498


namespace NUMINAMATH_CALUDE_line_equations_l3194_319478

-- Define the lines
def l₁ (x y : ℝ) : Prop := 2 * x - 3 * y + 4 = 0
def l₂ (x y : ℝ) : Prop := x + y - 3 = 0
def l₃ (x y : ℝ) : Prop := x - 2 * y + 5 = 0

-- Define the intersection point M
def M : ℝ × ℝ := (1, 2)

-- Define the perpendicular line l
def l (x y : ℝ) : Prop := 3 * x + 2 * y - 7 = 0

-- Define the parallel line l'
def l' (x y : ℝ) : Prop := x - 2 * y + 3 = 0

theorem line_equations :
  (∀ x y : ℝ, l₁ x y ∧ l₂ x y → (x, y) = M) →
  (∀ x y : ℝ, l x y ↔ (3 * x + 2 * y - 7 = 0 ∧ (x, y) = M ∨ l₁ x y)) →
  (∀ x y : ℝ, l' x y ↔ (x - 2 * y + 3 = 0 ∧ (x, y) = M ∨ l₃ x y)) :=
by sorry

end NUMINAMATH_CALUDE_line_equations_l3194_319478


namespace NUMINAMATH_CALUDE_factors_of_81_l3194_319497

theorem factors_of_81 : Finset.card (Nat.divisors 81) = 5 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_81_l3194_319497


namespace NUMINAMATH_CALUDE_triangle_triple_sine_sum_l3194_319447

theorem triangle_triple_sine_sum (A B C : ℝ) : 
  A + B + C = π ∧ (A = π/3 ∨ B = π/3 ∨ C = π/3) → 
  Real.sin (3*A) + Real.sin (3*B) + Real.sin (3*C) = 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_triple_sine_sum_l3194_319447


namespace NUMINAMATH_CALUDE_sets_intersection_and_complement_l3194_319467

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | 2 * x^2 + a * x + 2 = 0}
def B (b : ℝ) : Set ℝ := {x | x^2 + 3 * x - b = 0}

-- State the theorem
theorem sets_intersection_and_complement (a b : ℝ) :
  (A a ∩ B b = {2}) →
  ∃ (U : Set ℝ),
    a = -5 ∧
    b = 10 ∧
    U = A a ∪ B b ∧
    (Uᶜ ∩ A a) ∪ (Uᶜ ∩ B b) = {-5, 1/2} := by
  sorry

end NUMINAMATH_CALUDE_sets_intersection_and_complement_l3194_319467


namespace NUMINAMATH_CALUDE_min_value_of_f_on_interval_l3194_319403

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_minus_x_squared_odd : ∀ x : ℝ, f (-x) - (-x)^2 = -(f x - x^2)
axiom f_plus_2_pow_x_even : ∀ x : ℝ, f (-x) + 2^(-x) = f x + 2^x

-- Define the interval
def interval : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ -1}

-- State the theorem
theorem min_value_of_f_on_interval :
  ∃ x₀ ∈ interval, ∀ x ∈ interval, f x₀ ≤ f x ∧ f x₀ = 7/4 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_on_interval_l3194_319403


namespace NUMINAMATH_CALUDE_chicken_price_chicken_price_is_8_l3194_319468

/-- Calculates the price of a chicken given the conditions of the farmer's sales. -/
theorem chicken_price : ℝ → Prop :=
  fun price =>
    let duck_price := 10
    let num_ducks := 2
    let num_chickens := 5
    let total_earnings := duck_price * num_ducks + price * num_chickens
    let wheelbarrow_cost := total_earnings / 2
    let wheelbarrow_sale := wheelbarrow_cost * 2
    let additional_earnings := 60
    wheelbarrow_sale - wheelbarrow_cost = additional_earnings →
    price = 8

/-- The price of a chicken is $8. -/
theorem chicken_price_is_8 : chicken_price 8 := by
  sorry

end NUMINAMATH_CALUDE_chicken_price_chicken_price_is_8_l3194_319468


namespace NUMINAMATH_CALUDE_even_function_graph_l3194_319438

/-- An even function is a function that satisfies f(-x) = f(x) for all x in its domain -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The statement that (-a, f(a)) lies on the graph of f for any even function f and any real a -/
theorem even_function_graph (f : ℝ → ℝ) (h : EvenFunction f) (a : ℝ) :
  f (-a) = f a := by sorry

end NUMINAMATH_CALUDE_even_function_graph_l3194_319438


namespace NUMINAMATH_CALUDE_inscribed_circle_segment_lengths_l3194_319461

/-- Given a triangle with sides a, b, c and an inscribed circle, 
    the lengths of the segments into which the points of tangency divide the sides 
    are (a + b - c)/2, (a + c - b)/2, and (b + c - a)/2. -/
theorem inscribed_circle_segment_lengths 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) : 
  ∃ (x y z : ℝ),
    x = (a + b - c) / 2 ∧
    y = (a + c - b) / 2 ∧
    z = (b + c - a) / 2 ∧
    x + y = a ∧
    x + z = b ∧
    y + z = c :=
by sorry


end NUMINAMATH_CALUDE_inscribed_circle_segment_lengths_l3194_319461


namespace NUMINAMATH_CALUDE_cow_field_theorem_l3194_319445

theorem cow_field_theorem (total_cows : ℕ) (female_cows : ℕ) (male_cows : ℕ) 
  (spotted_females : ℕ) (horned_males : ℕ) : 
  total_cows = 300 →
  female_cows = 2 * male_cows →
  female_cows + male_cows = total_cows →
  spotted_females = female_cows / 2 →
  horned_males = male_cows / 2 →
  spotted_females - horned_males = 50 := by
sorry

end NUMINAMATH_CALUDE_cow_field_theorem_l3194_319445


namespace NUMINAMATH_CALUDE_simplify_expression_l3194_319495

theorem simplify_expression (y : ℝ) : 
  5 * y - 6 * y^2 + 9 - (4 - 5 * y + 2 * y^2) = -8 * y^2 + 10 * y + 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3194_319495


namespace NUMINAMATH_CALUDE_power_division_equality_l3194_319433

theorem power_division_equality : (4 ^ (3^2)) / ((4^3)^2) = 64 := by sorry

end NUMINAMATH_CALUDE_power_division_equality_l3194_319433


namespace NUMINAMATH_CALUDE_solution_set_implies_a_value_l3194_319483

theorem solution_set_implies_a_value (a b : ℝ) : 
  (∀ x, |x - a| < b ↔ 2 < x ∧ x < 4) → a = 3 := by
sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_value_l3194_319483


namespace NUMINAMATH_CALUDE_unique_consecutive_set_sum_20_l3194_319477

/-- A set of consecutive positive integers -/
structure ConsecutiveSet where
  start : ℕ
  length : ℕ
  sum : ℕ
  h1 : start ≥ 2
  h2 : length ≥ 2
  h3 : sum = (length * (2 * start + length - 1)) / 2

/-- The theorem stating that there is exactly one set of consecutive positive integers
    starting from 2 or higher, with at least two numbers, whose sum is 20 -/
theorem unique_consecutive_set_sum_20 :
  ∃! (s : ConsecutiveSet), s.sum = 20 :=
sorry

end NUMINAMATH_CALUDE_unique_consecutive_set_sum_20_l3194_319477


namespace NUMINAMATH_CALUDE_tensor_equation_solution_l3194_319416

/-- Custom binary operation ⊗ for positive real numbers -/
def tensor (a b : ℝ) : ℝ := a * b + a + b^2

/-- Theorem stating that if 1⊗m = 3, then m = 1 -/
theorem tensor_equation_solution (m : ℝ) (h1 : m > 0) (h2 : tensor 1 m = 3) : m = 1 := by
  sorry

end NUMINAMATH_CALUDE_tensor_equation_solution_l3194_319416


namespace NUMINAMATH_CALUDE_chime_2500_date_l3194_319409

/-- Represents a date --/
structure Date where
  year : ℕ
  month : ℕ
  day : ℕ

/-- Represents a time --/
structure Time where
  hour : ℕ
  minute : ℕ

/-- Represents a clock with a specific chiming pattern --/
structure ChimingClock where
  /-- Chimes once at 30 minutes past each hour --/
  chimeAtHalfHour : Bool
  /-- Chimes on the hour according to the hour number --/
  chimeOnHour : ℕ → ℕ

/-- Calculates the number of chimes between two dates and times --/
def countChimes (clock : ChimingClock) (startDate : Date) (startTime : Time) (endDate : Date) (endTime : Time) : ℕ := sorry

/-- The theorem to be proved --/
theorem chime_2500_date (clock : ChimingClock) : 
  let startDate := Date.mk 2003 2 26
  let startTime := Time.mk 10 45
  let endDate := Date.mk 2003 3 21
  countChimes clock startDate startTime endDate (Time.mk 23 59) = 2500 := by sorry

end NUMINAMATH_CALUDE_chime_2500_date_l3194_319409


namespace NUMINAMATH_CALUDE_linear_expression_bounds_l3194_319496

/-- Given a system of equations and constraints, prove the bounds of a linear expression. -/
theorem linear_expression_bounds (x y z : ℝ) : 
  x ≥ 0 → y ≥ 0 → z ≥ 0 → 
  x - 2*y - 3*z = -10 → 
  x + 2*y + z = 6 → 
  ∃ (A_min A_max : ℝ), 
    (∀ A, A = 1.5*x + y - z → A ≥ A_min ∧ A ≤ A_max) ∧
    A_min = -1 ∧ A_max = 0 := by
  sorry

end NUMINAMATH_CALUDE_linear_expression_bounds_l3194_319496


namespace NUMINAMATH_CALUDE_backpack_profit_theorem_l3194_319430

/-- Represents the profit equation for a backpack sale -/
def profit_equation (x : ℝ) : Prop :=
  (1 + 0.5) * x * 0.8 - x = 8

/-- Theorem stating the profit equation holds for a backpack sale with given conditions -/
theorem backpack_profit_theorem (x : ℝ) 
  (h_markup : ℝ → ℝ := λ price => (1 + 0.5) * price)
  (h_discount : ℝ → ℝ := λ price => 0.8 * price)
  (h_profit : ℝ := 8) :
  profit_equation x := by
  sorry

end NUMINAMATH_CALUDE_backpack_profit_theorem_l3194_319430
