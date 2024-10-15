import Mathlib

namespace NUMINAMATH_CALUDE_total_holes_dug_l2899_289934

-- Define Pearl's digging rate
def pearl_rate : ℚ := 4 / 7

-- Define Miguel's digging rate
def miguel_rate : ℚ := 2 / 3

-- Define the duration of work
def work_duration : ℕ := 21

-- Theorem to prove
theorem total_holes_dug : 
  ⌊(pearl_rate * work_duration) + (miguel_rate * work_duration)⌋ = 26 := by
  sorry


end NUMINAMATH_CALUDE_total_holes_dug_l2899_289934


namespace NUMINAMATH_CALUDE_expression_evaluation_l2899_289947

/-- Proves that the given expression evaluates to -3/2 when x = -1/2 and y = 3 -/
theorem expression_evaluation :
  let x : ℚ := -1/2
  let y : ℚ := 3
  3 * (2 * x^2 * y - x * y^2) - 2 * (-2 * y^2 * x + x^2 * y) = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2899_289947


namespace NUMINAMATH_CALUDE_cherry_cost_weight_relationship_l2899_289968

/-- The relationship between the cost of cherries and their weight -/
theorem cherry_cost_weight_relationship (x y : ℝ) :
  (∀ w, w * 16 = w * (y / x)) → y = 16 * x :=
by sorry

end NUMINAMATH_CALUDE_cherry_cost_weight_relationship_l2899_289968


namespace NUMINAMATH_CALUDE_student_multiplication_problem_l2899_289924

theorem student_multiplication_problem (x y : ℝ) : 
  x = 127 → x * y - 152 = 102 → y = 2 := by
sorry

end NUMINAMATH_CALUDE_student_multiplication_problem_l2899_289924


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2899_289979

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = 1 - z) :
  Complex.im z = -1/5 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2899_289979


namespace NUMINAMATH_CALUDE_multiplication_subtraction_equality_l2899_289970

theorem multiplication_subtraction_equality : 210 * 6 - 52 * 5 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_subtraction_equality_l2899_289970


namespace NUMINAMATH_CALUDE_decreasing_f_implies_a_leq_neg_five_l2899_289959

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a+1)*x + 2

-- State the theorem
theorem decreasing_f_implies_a_leq_neg_five (a : ℝ) :
  (∀ x ≤ 4, ∀ y ≤ 4, x < y → f a x > f a y) →
  a ≤ -5 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_f_implies_a_leq_neg_five_l2899_289959


namespace NUMINAMATH_CALUDE_yonderland_license_plates_l2899_289978

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of digits (0-9) -/
def digit_count : ℕ := 10

/-- The number of non-zero digits (1-9) -/
def non_zero_digit_count : ℕ := 9

/-- The number of letters in a license plate -/
def letter_count : ℕ := 3

/-- The number of digits in a license plate -/
def digit_position_count : ℕ := 4

/-- The total number of valid license plates in Yonderland -/
def valid_license_plate_count : ℕ :=
  alphabet_size * (alphabet_size - 1) * (alphabet_size - 2) *
  non_zero_digit_count * digit_count^(digit_position_count - 1)

theorem yonderland_license_plates :
  valid_license_plate_count = 702000000 := by
  sorry

end NUMINAMATH_CALUDE_yonderland_license_plates_l2899_289978


namespace NUMINAMATH_CALUDE_smallest_integer_negative_quadratic_l2899_289925

theorem smallest_integer_negative_quadratic :
  ∃ (n : ℤ), (∀ (m : ℤ), m^2 - 11*m + 28 < 0 → n ≤ m) ∧ (n^2 - 11*n + 28 < 0) ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_negative_quadratic_l2899_289925


namespace NUMINAMATH_CALUDE_bee_colony_fraction_l2899_289955

theorem bee_colony_fraction (initial_bees : ℕ) (daily_loss : ℕ) (days : ℕ) :
  initial_bees = 80000 →
  daily_loss = 1200 →
  days = 50 →
  (initial_bees - daily_loss * days) / initial_bees = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_bee_colony_fraction_l2899_289955


namespace NUMINAMATH_CALUDE_triangle_area_extension_l2899_289982

/-- Given a triangle ABC with area 36 and base BC of length 7, and an extended triangle BCD
    with CD of length 30, prove that the area of BCD is 1080/7. -/
theorem triangle_area_extension (h : ℝ) : 
  36 = (1/2) * 7 * h →  -- Area of ABC
  (1/2) * 30 * h = 1080/7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_extension_l2899_289982


namespace NUMINAMATH_CALUDE_minimum_fourth_quarter_score_l2899_289932

def required_average : ℚ := 85
def num_quarters : ℕ := 4
def first_quarter : ℚ := 82
def second_quarter : ℚ := 77
def third_quarter : ℚ := 78

theorem minimum_fourth_quarter_score :
  let total_required := required_average * num_quarters
  let sum_first_three := first_quarter + second_quarter + third_quarter
  let minimum_fourth := total_required - sum_first_three
  minimum_fourth = 103 ∧
  (first_quarter + second_quarter + third_quarter + minimum_fourth) / num_quarters ≥ required_average :=
by sorry

end NUMINAMATH_CALUDE_minimum_fourth_quarter_score_l2899_289932


namespace NUMINAMATH_CALUDE_distance_to_y_axis_l2899_289927

/-- Given a point P with coordinates (x, -4), if the distance from the x-axis to P
    is half the distance from the y-axis to P, then the distance from the y-axis to P is 8. -/
theorem distance_to_y_axis (x : ℝ) :
  let P : ℝ × ℝ := (x, -4)
  let dist_to_x_axis : ℝ := |P.2|
  let dist_to_y_axis : ℝ := |P.1|
  dist_to_x_axis = (1/2 : ℝ) * dist_to_y_axis →
  dist_to_y_axis = 8 := by
sorry

end NUMINAMATH_CALUDE_distance_to_y_axis_l2899_289927


namespace NUMINAMATH_CALUDE_tan_beta_value_l2899_289975

theorem tan_beta_value (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.tan α = 4/3) (h4 : Real.cos (α + β) = Real.sqrt 5 / 5) :
  Real.tan β = 2/11 := by
  sorry

end NUMINAMATH_CALUDE_tan_beta_value_l2899_289975


namespace NUMINAMATH_CALUDE_peter_five_theorem_l2899_289938

theorem peter_five_theorem (N : ℕ+) :
  ∃ K : ℕ, ∀ k : ℕ, k ≥ K → (∃ d m n : ℕ, N * 5^k = 10^n * (10 * m + 5) + d ∧ d < 10^n) :=
sorry

end NUMINAMATH_CALUDE_peter_five_theorem_l2899_289938


namespace NUMINAMATH_CALUDE_range_of_p_l2899_289953

-- Define the function p(x)
def p (x : ℝ) : ℝ := x^6 + 6*x^3 + 9

-- State the theorem
theorem range_of_p :
  {y : ℝ | ∃ x ≥ 0, p x = y} = {y : ℝ | y ≥ 9} :=
sorry

end NUMINAMATH_CALUDE_range_of_p_l2899_289953


namespace NUMINAMATH_CALUDE_B_power_101_l2899_289998

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![0, 1, 0],
    ![0, 0, 1],
    ![1, 0, 0]]

theorem B_power_101 : B^101 = B^2 := by sorry

end NUMINAMATH_CALUDE_B_power_101_l2899_289998


namespace NUMINAMATH_CALUDE_min_bricks_for_cube_l2899_289931

/-- The width of a brick in centimeters -/
def brick_width : ℕ := 18

/-- The depth of a brick in centimeters -/
def brick_depth : ℕ := 12

/-- The height of a brick in centimeters -/
def brick_height : ℕ := 9

/-- The volume of a single brick in cubic centimeters -/
def brick_volume : ℕ := brick_width * brick_depth * brick_height

/-- The side length of the smallest cube that can be formed using the bricks -/
def cube_side_length : ℕ := Nat.lcm (Nat.lcm brick_width brick_depth) brick_height

/-- The volume of the smallest cube that can be formed using the bricks -/
def cube_volume : ℕ := cube_side_length ^ 3

/-- The theorem stating the minimum number of bricks required to make a cube -/
theorem min_bricks_for_cube : cube_volume / brick_volume = 24 := by
  sorry

end NUMINAMATH_CALUDE_min_bricks_for_cube_l2899_289931


namespace NUMINAMATH_CALUDE_triangle_side_range_l2899_289933

theorem triangle_side_range (a b c : ℝ) (A B C : ℝ) :
  c = Real.sqrt 2 →
  a * Real.cos C = c * Real.sin A →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  Real.sqrt 2 < b ∧ b < 2 :=
sorry

end NUMINAMATH_CALUDE_triangle_side_range_l2899_289933


namespace NUMINAMATH_CALUDE_tan_expression_value_l2899_289923

theorem tan_expression_value (x : ℝ) (h : Real.tan (3 * Real.pi - x) = 2) :
  (2 * (Real.cos (x / 2))^2 - Real.sin x - 1) / (Real.sin x + Real.cos x) = -3 := by
  sorry

end NUMINAMATH_CALUDE_tan_expression_value_l2899_289923


namespace NUMINAMATH_CALUDE_geometric_identity_l2899_289964

theorem geometric_identity 
  (a b c p x : ℝ) 
  (h1 : a + b + c = 2 * p) 
  (h2 : x = (b^2 + c^2 - a^2) / (2 * c)) 
  (h3 : c ≠ 0) : 
  b^2 - x^2 = (4 / c^2) * (p * (p - a) * (p - b) * (p - c)) := by
  sorry

end NUMINAMATH_CALUDE_geometric_identity_l2899_289964


namespace NUMINAMATH_CALUDE_correct_stratified_sample_l2899_289985

/-- Represents the number of students in each grade --/
structure GradePopulation where
  freshmen : ℕ
  sophomores : ℕ
  seniors : ℕ

/-- Represents the number of students sampled from each grade --/
structure SampleSize where
  freshmen : ℕ
  sophomores : ℕ
  seniors : ℕ

/-- Calculates the stratified sample size for each grade --/
def stratifiedSample (pop : GradePopulation) (totalSample : ℕ) : SampleSize :=
  let totalPop := pop.freshmen + pop.sophomores + pop.seniors
  { freshmen := (totalSample * pop.freshmen) / totalPop,
    sophomores := (totalSample * pop.sophomores) / totalPop,
    seniors := (totalSample * pop.seniors) / totalPop }

/-- Theorem stating the correct stratified sample sizes for the given population --/
theorem correct_stratified_sample :
  let pop : GradePopulation := { freshmen := 900, sophomores := 1200, seniors := 600 }
  let sample := stratifiedSample pop 135
  sample.freshmen = 45 ∧ sample.sophomores = 60 ∧ sample.seniors = 30 := by
  sorry

end NUMINAMATH_CALUDE_correct_stratified_sample_l2899_289985


namespace NUMINAMATH_CALUDE_geometric_sum_five_terms_l2899_289965

/-- Given a geometric sequence with first term a and common ratio r,
    find n such that the sum of the first n terms is equal to s. -/
def find_n_for_geometric_sum (a r s : ℚ) : ℕ :=
  sorry

theorem geometric_sum_five_terms
  (a r : ℚ)
  (h_a : a = 1/3)
  (h_r : r = 1/3)
  (h_sum : (a * (1 - r^5)) / (1 - r) = 80/243) :
  find_n_for_geometric_sum a r (80/243) = 5 :=
sorry

end NUMINAMATH_CALUDE_geometric_sum_five_terms_l2899_289965


namespace NUMINAMATH_CALUDE_tan_ratio_problem_l2899_289915

theorem tan_ratio_problem (x : ℝ) (h : Real.tan (x + π/4) = 2) : 
  Real.tan x / Real.tan (2*x) = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_tan_ratio_problem_l2899_289915


namespace NUMINAMATH_CALUDE_extremum_implies_slope_l2899_289958

-- Define the function f(x)
def f (c : ℝ) (x : ℝ) : ℝ := (x - 2) * (x^2 + c)

-- State the theorem
theorem extremum_implies_slope (c : ℝ) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (2 - ε) (2 + ε), f c x ≤ f c 2 ∨ f c x ≥ f c 2) →
  (deriv (f c)) 1 = -5 :=
sorry

end NUMINAMATH_CALUDE_extremum_implies_slope_l2899_289958


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2899_289910

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, S n = n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1)) →  -- Definition of sum for arithmetic sequence
  (∀ n, a (n + 1) - a n = a 2 - a 1) →                      -- Definition of arithmetic sequence
  S 3 = 6 →                                                 -- Given condition
  5 * a 1 + a 7 = 12 :=                                     -- Conclusion to prove
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2899_289910


namespace NUMINAMATH_CALUDE_closest_multiple_of_12_to_1987_is_correct_l2899_289914

def is_multiple_of_12 (n : ℤ) : Prop := n % 12 = 0

def closest_multiple_of_12_to_1987 : ℤ := 1984

theorem closest_multiple_of_12_to_1987_is_correct :
  is_multiple_of_12 closest_multiple_of_12_to_1987 ∧
  ∀ m : ℤ, is_multiple_of_12 m →
    |m - 1987| ≥ |closest_multiple_of_12_to_1987 - 1987| :=
by sorry

end NUMINAMATH_CALUDE_closest_multiple_of_12_to_1987_is_correct_l2899_289914


namespace NUMINAMATH_CALUDE_average_age_increase_l2899_289993

theorem average_age_increase (initial_count : ℕ) (replaced_count : ℕ) (age1 age2 : ℕ) (women_avg_age : ℚ) : 
  initial_count = 7 →
  replaced_count = 2 →
  age1 = 18 →
  age2 = 22 →
  women_avg_age = 30.5 →
  (2 * women_avg_age - (age1 + age2 : ℚ)) / initial_count = 3 := by
  sorry

end NUMINAMATH_CALUDE_average_age_increase_l2899_289993


namespace NUMINAMATH_CALUDE_complete_solution_set_l2899_289963

def S : Set (ℕ × ℕ × ℕ) :=
  {(4, 33, 30), (32, 9, 30), (40, 9, 18), (12, 31, 30), (24, 23, 30), (4, 15, 22), (36, 15, 42)}

def is_solution (t : ℕ × ℕ × ℕ) : Prop :=
  let (a, b, c) := t
  a^2 + b^2 + c^2 = 2005 ∧ 0 < a ∧ a ≤ b ∧ b ≤ c

theorem complete_solution_set :
  ∀ (a b c : ℕ), is_solution (a, b, c) ↔ (a, b, c) ∈ S := by
  sorry

end NUMINAMATH_CALUDE_complete_solution_set_l2899_289963


namespace NUMINAMATH_CALUDE_largest_four_digit_number_with_conditions_l2899_289952

/-- A function that checks if all digits in a number are different -/
def allDigitsDifferent (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = digits.toFinset.card

/-- The theorem statement -/
theorem largest_four_digit_number_with_conditions :
  ∃ (n : ℕ),
    n = 8910 ∧
    1000 ≤ n ∧ n < 10000 ∧
    allDigitsDifferent n ∧
    n % 2 = 0 ∧ n % 5 = 0 ∧ n % 9 = 0 ∧ n % 11 = 0 ∧
    ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ allDigitsDifferent m ∧
      m % 2 = 0 ∧ m % 5 = 0 ∧ m % 9 = 0 ∧ m % 11 = 0 → m ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_largest_four_digit_number_with_conditions_l2899_289952


namespace NUMINAMATH_CALUDE_distance_to_circle_center_l2899_289928

/-- The distance from a point in polar coordinates to the center of a circle defined by a polar equation --/
theorem distance_to_circle_center (ρ₀ : ℝ) (θ₀ : ℝ) :
  let circle := fun θ => 2 * Real.cos θ
  let center_x := 1
  let center_y := 0
  let point_x := ρ₀ * Real.cos θ₀
  let point_y := ρ₀ * Real.sin θ₀
  (ρ₀ = 2 ∧ θ₀ = Real.pi / 3) →
  Real.sqrt ((point_x - center_x)^2 + (point_y - center_y)^2) = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_distance_to_circle_center_l2899_289928


namespace NUMINAMATH_CALUDE_expression_value_l2899_289940

theorem expression_value (a b : ℤ) (ha : a = 3) (hb : b = -2) :
  -a^2 - b^3 + a*b = -7 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2899_289940


namespace NUMINAMATH_CALUDE_max_distance_with_tire_swap_l2899_289919

/-- Represents the maximum distance a tire can travel on the rear wheel before wearing out. -/
def rear_tire_limit : ℝ := 15000

/-- Represents the maximum distance a tire can travel on the front wheel before wearing out. -/
def front_tire_limit : ℝ := 25000

/-- Represents the maximum distance a truck can travel before all four tires are worn out,
    given that tires can be swapped between front and rear positions. -/
def max_truck_distance : ℝ := 18750

/-- Theorem stating that the maximum distance a truck can travel before all four tires
    are worn out is 18750 km, given the conditions on tire wear and the ability to swap tires. -/
theorem max_distance_with_tire_swap :
  max_truck_distance = 18750 :=
by sorry

end NUMINAMATH_CALUDE_max_distance_with_tire_swap_l2899_289919


namespace NUMINAMATH_CALUDE_power_equation_solution_l2899_289976

theorem power_equation_solution (m : ℕ) : 2^m = 2 * 16^2 * 4^3 → m = 15 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l2899_289976


namespace NUMINAMATH_CALUDE_solve_equation_l2899_289972

theorem solve_equation (n : ℕ) : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^18 → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2899_289972


namespace NUMINAMATH_CALUDE_correct_average_l2899_289949

theorem correct_average (n : ℕ) (initial_avg wrong_num correct_num : ℚ) :
  n = 10 →
  initial_avg = 15 →
  wrong_num = 26 →
  correct_num = 36 →
  (n : ℚ) * initial_avg + (correct_num - wrong_num) = n * 16 :=
by sorry

end NUMINAMATH_CALUDE_correct_average_l2899_289949


namespace NUMINAMATH_CALUDE_cone_height_l2899_289945

/-- A cone with volume 8192π cubic inches and a vertical cross-section vertex angle of 90 degrees has a height equal to the cube root of 24576 inches. -/
theorem cone_height (V : ℝ) (θ : ℝ) (h : ℝ) :
  V = 8192 * Real.pi ∧ θ = 90 → h = (24576 : ℝ) ^ (1/3) :=
by sorry

end NUMINAMATH_CALUDE_cone_height_l2899_289945


namespace NUMINAMATH_CALUDE_triangle_sides_theorem_l2899_289992

def triangle_exists (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_sides_theorem (x : ℕ+) :
  triangle_exists 8 11 (x.val ^ 2) ↔ x.val = 2 ∨ x.val = 3 ∨ x.val = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sides_theorem_l2899_289992


namespace NUMINAMATH_CALUDE_trig_expression_equals_half_l2899_289900

/-- Proves that the given trigonometric expression equals 1/2 --/
theorem trig_expression_equals_half : 
  (Real.sin (70 * π / 180) * Real.sin (20 * π / 180)) / 
  (Real.cos (155 * π / 180)^2 - Real.sin (155 * π / 180)^2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_half_l2899_289900


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l2899_289997

theorem polynomial_division_theorem (x : ℝ) : 
  12 * x^3 + 18 * x^2 + 27 * x + 17 = 
  (4 * x + 3) * (3 * x^2 + 2.25 * x + 5/16) + 29/16 := by
sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l2899_289997


namespace NUMINAMATH_CALUDE_largest_number_is_482_l2899_289974

/-- Represents a systematic sample from a range of products -/
structure SystematicSample where
  total_products : Nat
  first_number : Nat
  second_number : Nat

/-- Calculates the largest number in a systematic sample -/
def largest_number (s : SystematicSample) : Nat :=
  let interval := s.second_number - s.first_number
  let sample_size := s.total_products / interval
  s.first_number + interval * (sample_size - 1)

/-- Theorem stating that for the given systematic sample, the largest number is 482 -/
theorem largest_number_is_482 :
  let s : SystematicSample := ⟨500, 7, 32⟩
  largest_number s = 482 := by sorry

end NUMINAMATH_CALUDE_largest_number_is_482_l2899_289974


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_existence_l2899_289943

theorem quadratic_equation_solution_existence 
  (a b c : ℝ) 
  (h_a : a ≠ 0)
  (h_1 : a * (3.24 : ℝ)^2 + b * (3.24 : ℝ) + c = -0.02)
  (h_2 : a * (3.25 : ℝ)^2 + b * (3.25 : ℝ) + c = 0.03) :
  ∃ x : ℝ, x > 3.24 ∧ x < 3.25 ∧ a * x^2 + b * x + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_existence_l2899_289943


namespace NUMINAMATH_CALUDE_opposite_side_of_five_times_five_l2899_289903

/-- A standard 6-sided die with opposite sides summing to 7 -/
structure StandardDie where
  sides : Fin 6 → Nat
  valid_range : ∀ i, sides i ∈ Finset.range 7 \ {0}
  opposite_sum : ∀ i, sides i + sides (5 - i) = 7

/-- The number of eyes on the opposite side of 5 multiplied by 5 is 10 -/
theorem opposite_side_of_five_times_five (d : StandardDie) :
  5 * d.sides (5 - 5) = 10 := by
  sorry

end NUMINAMATH_CALUDE_opposite_side_of_five_times_five_l2899_289903


namespace NUMINAMATH_CALUDE_dividend_proof_l2899_289957

theorem dividend_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) : 
  dividend = 10918788 ∧ divisor = 12 ∧ quotient = 909899 → 
  dividend / divisor = quotient := by
  sorry

end NUMINAMATH_CALUDE_dividend_proof_l2899_289957


namespace NUMINAMATH_CALUDE_min_value_expression_l2899_289918

theorem min_value_expression (x y : ℝ) : 5*x^2 + 4*y^2 - 8*x*y + 2*x + 4 ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2899_289918


namespace NUMINAMATH_CALUDE_geometric_sequence_terms_l2899_289921

/-- 
Given a geometric sequence where:
- The first term is 9/8
- The last term is 1/3
- The common ratio is 2/3
This theorem proves that the number of terms in the sequence is 4.
-/
theorem geometric_sequence_terms : 
  ∀ (a : ℚ) (r : ℚ) (last : ℚ) (n : ℕ),
  a = 9/8 → r = 2/3 → last = 1/3 →
  last = a * r^(n-1) →
  n = 4 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_terms_l2899_289921


namespace NUMINAMATH_CALUDE_annie_extracurricular_hours_l2899_289960

/-- Calculates the total extracurricular hours before midterms -/
def extracurricular_hours_before_midterms (
  chess_hours_per_week : ℕ)
  (drama_hours_per_week : ℕ)
  (glee_hours_per_week : ℕ)
  (weeks_in_semester : ℕ)
  (weeks_off_sick : ℕ) : ℕ :=
  let total_hours_per_week := chess_hours_per_week + drama_hours_per_week + glee_hours_per_week
  let weeks_before_midterms := weeks_in_semester / 2
  let active_weeks := weeks_before_midterms - weeks_off_sick
  total_hours_per_week * active_weeks

theorem annie_extracurricular_hours :
  extracurricular_hours_before_midterms 2 8 3 12 2 = 52 := by
  sorry

end NUMINAMATH_CALUDE_annie_extracurricular_hours_l2899_289960


namespace NUMINAMATH_CALUDE_remainder_sum_l2899_289929

theorem remainder_sum (a b : ℤ) : 
  a % 45 = 37 → b % 30 = 9 → (a + b) % 15 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l2899_289929


namespace NUMINAMATH_CALUDE_same_solution_implies_c_value_l2899_289966

theorem same_solution_implies_c_value (x c : ℝ) : 
  (3 * x + 8 = 5) ∧ (c * x - 15 = -3) → c = -12 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_implies_c_value_l2899_289966


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l2899_289902

def repeating_decimal_to_fraction (n : ℕ) : ℚ := n / 9

theorem sum_of_repeating_decimals :
  let a := repeating_decimal_to_fraction 6
  let b := repeating_decimal_to_fraction 2
  let c := repeating_decimal_to_fraction 4
  let d := repeating_decimal_to_fraction 7
  a + b - c - d = -1/3 := by sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l2899_289902


namespace NUMINAMATH_CALUDE_trig_expression_equals_four_l2899_289954

theorem trig_expression_equals_four :
  1 / Real.cos (10 * π / 180) - Real.sqrt 3 / Real.sin (10 * π / 180) = 4 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_four_l2899_289954


namespace NUMINAMATH_CALUDE_rectangular_box_volume_l2899_289946

theorem rectangular_box_volume (l w h : ℝ) 
  (area1 : l * w = 40)
  (area2 : w * h = 15)
  (area3 : l * h = 12) :
  l * w * h = 60 := by
sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l2899_289946


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2899_289916

theorem inequality_equivalence (x : ℝ) : 
  (-2 < (x^2 - 16*x + 15) / (x^2 - 4*x + 5) ∧ (x^2 - 16*x + 15) / (x^2 - 4*x + 5) < 2) ↔ 
  (4 - Real.sqrt 21 < x ∧ x < 4 + Real.sqrt 21) := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2899_289916


namespace NUMINAMATH_CALUDE_right_angled_triangle_isosceles_triangle_isosceles_perimeter_l2899_289917

/-- Definition of the triangle ABC with side lengths based on the quadratic equation -/
def Triangle (k : ℝ) : Prop :=
  ∃ (a b : ℝ),
    a^2 - (2*k + 3)*a + k^2 + 3*k + 2 = 0 ∧
    b^2 - (2*k + 3)*b + k^2 + 3*k + 2 = 0 ∧
    a ≠ b

/-- The length of side BC is 5 -/
def BC_length (k : ℝ) : ℝ := 5

/-- Theorem: If ABC is a right-angled triangle with BC as the hypotenuse, then k = 2 -/
theorem right_angled_triangle (k : ℝ) :
  Triangle k → (∃ (a b : ℝ), a^2 + b^2 = (BC_length k)^2) → k = 2 :=
sorry

/-- Theorem: If ABC is an isosceles triangle, then k = 3 or k = 4 -/
theorem isosceles_triangle (k : ℝ) :
  Triangle k → (∃ (a b : ℝ), (a = b ∧ a ≠ BC_length k) ∨ (a = BC_length k ∧ b ≠ BC_length k) ∨ (b = BC_length k ∧ a ≠ BC_length k)) →
  k = 3 ∨ k = 4 :=
sorry

/-- Theorem: If ABC is an isosceles triangle, then its perimeter is 14 or 16 -/
theorem isosceles_perimeter (k : ℝ) :
  Triangle k → (∃ (a b : ℝ), (a = b ∧ a ≠ BC_length k) ∨ (a = BC_length k ∧ b ≠ BC_length k) ∨ (b = BC_length k ∧ a ≠ BC_length k)) →
  (∃ (p : ℝ), p = a + b + BC_length k ∧ (p = 14 ∨ p = 16)) :=
sorry

end NUMINAMATH_CALUDE_right_angled_triangle_isosceles_triangle_isosceles_perimeter_l2899_289917


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l2899_289935

theorem fixed_point_on_line (m : ℝ) : (m - 1) * (7/2) - (m + 3) * (5/2) - (m - 11) = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l2899_289935


namespace NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l2899_289926

-- Define the repeating decimal 0.4555...
def repeating_decimal : ℚ := 0.4555555555555555

-- Theorem statement
theorem repeating_decimal_as_fraction : repeating_decimal = 41 / 90 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l2899_289926


namespace NUMINAMATH_CALUDE_college_running_survey_l2899_289901

/-- Represents the sample data for running mileage --/
structure SampleData where
  male_0_30 : ℕ
  male_30_60 : ℕ
  male_60_90 : ℕ
  male_90_plus : ℕ
  female_0_30 : ℕ
  female_30_60 : ℕ
  female_60_90 : ℕ
  female_90_plus : ℕ

/-- Theorem representing the problem and its solution --/
theorem college_running_survey (total_students : ℕ) (male_students : ℕ) (female_students : ℕ)
    (sample : SampleData) :
    total_students = 1000 →
    male_students = 640 →
    female_students = 360 →
    sample.male_30_60 = 12 →
    sample.male_60_90 = 10 →
    sample.male_90_plus = 5 →
    sample.female_0_30 = 6 →
    sample.female_30_60 = 6 →
    sample.female_60_90 = 4 →
    sample.female_90_plus = 2 →
    (∃ (a : ℕ),
      sample.male_0_30 = a ∧
      a = 5 ∧
      ((a + 12 + 10 + 5 : ℚ) / (6 + 6 + 4 + 2) = 640 / 360) ∧
      (a * 1000 / (a + 12 + 10 + 5 + 6 + 6 + 4 + 2) = 100)) ∧
    (∃ (X : Fin 4 → ℚ),
      X 1 = 1/7 ∧ X 2 = 4/7 ∧ X 3 = 2/7 ∧
      (X 1 + X 2 + X 3 = 1) ∧
      (1 * X 1 + 2 * X 2 + 3 * X 3 = 15/7)) := by
  sorry


end NUMINAMATH_CALUDE_college_running_survey_l2899_289901


namespace NUMINAMATH_CALUDE_max_cities_is_107_l2899_289973

/-- The maximum number of cities that can be visited in a specific sequence -/
def max_cities : ℕ := 107

/-- The total number of cities in the country -/
def total_cities : ℕ := 110

/-- A function representing the number of roads for each city in the sequence -/
def roads_for_city (k : ℕ) : ℕ := k

/-- Theorem stating that the maximum number of cities that can be visited in the specific sequence is 107 -/
theorem max_cities_is_107 :
  ∀ N : ℕ, 
  (∀ k : ℕ, 2 ≤ k → k ≤ N → roads_for_city k = k) →
  N ≤ total_cities →
  N ≤ max_cities :=
sorry

end NUMINAMATH_CALUDE_max_cities_is_107_l2899_289973


namespace NUMINAMATH_CALUDE_smallest_congruent_number_l2899_289936

theorem smallest_congruent_number : ∃ n : ℕ, 
  (n > 1) ∧ 
  (n % 5 = 1) ∧ 
  (n % 7 = 1) ∧ 
  (n % 8 = 1) ∧ 
  (∀ m : ℕ, m > 1 → m % 5 = 1 → m % 7 = 1 → m % 8 = 1 → n ≤ m) ∧
  (n = 281) := by
sorry

end NUMINAMATH_CALUDE_smallest_congruent_number_l2899_289936


namespace NUMINAMATH_CALUDE_gcd_of_specific_numbers_l2899_289994

theorem gcd_of_specific_numbers : Nat.gcd 33333 666666 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_specific_numbers_l2899_289994


namespace NUMINAMATH_CALUDE_problem_solution_l2899_289942

theorem problem_solution : 45 / (7 - 3/4) = 36/5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2899_289942


namespace NUMINAMATH_CALUDE_tangent_parallel_points_tangent_equations_l2899_289986

/-- The function f(x) = x^3 + x - 2 -/
def f (x : ℝ) : ℝ := x^3 + x - 2

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

/-- The slope of the line y = 4x - 1 -/
def m : ℝ := 4

/-- The set of points where the tangent line is parallel to y = 4x - 1 -/
def tangent_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | f' p.1 = m ∧ p.2 = f p.1}

/-- The equation of the tangent line at a point (a, f(a)) -/
def tangent_line (a : ℝ) (x y : ℝ) : Prop :=
  y - f a = f' a * (x - a)

theorem tangent_parallel_points :
  tangent_points = {(1, 0), (-1, -4)} :=
sorry

theorem tangent_equations (a : ℝ) (h : (a, f a) ∈ tangent_points) :
  (∀ x y, tangent_line a x y ↔ (4 * x - y - 4 = 0 ∨ 4 * x - y = 0)) :=
sorry

end NUMINAMATH_CALUDE_tangent_parallel_points_tangent_equations_l2899_289986


namespace NUMINAMATH_CALUDE_unique_solution_l2899_289989

-- Define the equation
def equation (x : ℝ) : Prop :=
  2021 * x = 2022 * (x^2021)^(1/2021) - 1

-- Theorem statement
theorem unique_solution :
  ∃! x : ℝ, x ≥ 0 ∧ equation x :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l2899_289989


namespace NUMINAMATH_CALUDE_missing_number_proof_l2899_289951

theorem missing_number_proof (some_number : ℤ) : 
  (|4 - some_number * (3 - 12)| - |5 - 11| = 70) → some_number = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l2899_289951


namespace NUMINAMATH_CALUDE_sqrt_abs_sum_zero_implies_sum_power_l2899_289937

theorem sqrt_abs_sum_zero_implies_sum_power (a b : ℝ) :
  Real.sqrt (a - 2) + |b + 1| = 0 → (a + b)^2023 = 1 := by
sorry

end NUMINAMATH_CALUDE_sqrt_abs_sum_zero_implies_sum_power_l2899_289937


namespace NUMINAMATH_CALUDE_expression_evaluation_l2899_289944

theorem expression_evaluation :
  let m : ℚ := 2
  let expr := (m^2 - 9) / (m^2 - 6*m + 9) / (1 - 2/(m - 3))
  expr = -5/3 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2899_289944


namespace NUMINAMATH_CALUDE_smallest_solution_equation_l2899_289907

theorem smallest_solution_equation (x : ℝ) :
  (3 * x) / (x - 2) + (2 * x^2 - 28) / x = 11 →
  x ≥ ((-1 : ℝ) - Real.sqrt 17) / 2 ∧
  (3 * (((-1 : ℝ) - Real.sqrt 17) / 2)) / (((-1 : ℝ) - Real.sqrt 17) / 2 - 2) +
  (2 * (((-1 : ℝ) - Real.sqrt 17) / 2)^2 - 28) / (((-1 : ℝ) - Real.sqrt 17) / 2) = 11 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_equation_l2899_289907


namespace NUMINAMATH_CALUDE_expected_other_marbles_l2899_289950

/-- Represents the distribution of marble colors in Percius's collection -/
structure MarbleCollection where
  clear_percent : ℝ
  black_percent : ℝ
  other_percent : ℝ
  sum_to_one : clear_percent + black_percent + other_percent = 1

/-- Percius's marble collection -/
def percius_marbles : MarbleCollection where
  clear_percent := 0.4
  black_percent := 0.2
  other_percent := 0.4
  sum_to_one := by norm_num

/-- The number of marbles selected by the friend -/
def selected_marbles : ℕ := 5

/-- Theorem: The expected number of marbles of other colors when selecting 5 marbles is 2 -/
theorem expected_other_marbles :
  (selected_marbles : ℝ) * percius_marbles.other_percent = 2 := by sorry

end NUMINAMATH_CALUDE_expected_other_marbles_l2899_289950


namespace NUMINAMATH_CALUDE_even_number_2018_in_group_27_l2899_289920

/-- The sum of the number of elements in the first n groups --/
def S (n : ℕ) : ℕ := (3 * n^2 - n) / 2

/-- The proposition that 2018 is in the 27th group --/
theorem even_number_2018_in_group_27 :
  S 26 < 1009 ∧ 1009 ≤ S 27 :=
sorry

end NUMINAMATH_CALUDE_even_number_2018_in_group_27_l2899_289920


namespace NUMINAMATH_CALUDE_expression_value_l2899_289987

theorem expression_value (b c a : ℤ) (h1 : b = 10) (h2 : c = 3) (h3 : a = 2 * b) :
  (a - (b - c)) - ((a - b) - c) = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2899_289987


namespace NUMINAMATH_CALUDE_cloth_cost_price_l2899_289991

theorem cloth_cost_price 
  (selling_price : ℕ) 
  (cloth_length : ℕ) 
  (loss_per_meter : ℕ) 
  (h1 : selling_price = 18000) 
  (h2 : cloth_length = 600) 
  (h3 : loss_per_meter = 5) : 
  (selling_price + cloth_length * loss_per_meter) / cloth_length = 35 := by
sorry

end NUMINAMATH_CALUDE_cloth_cost_price_l2899_289991


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l2899_289971

/-- An ellipse with foci at (3, 5) and (23, 40) that is tangent to the y-axis has a major axis of length 43.835 -/
theorem ellipse_major_axis_length : 
  ∀ (E : Set (ℝ × ℝ)) (F₁ F₂ Y : ℝ × ℝ),
  F₁ = (3, 5) →
  F₂ = (23, 40) →
  (∀ P ∈ E, ∃ k, dist P F₁ + dist P F₂ = k) →
  (∃ t, Y = (0, t) ∧ Y ∈ E) →
  (∀ P : ℝ × ℝ, P.1 = 0 → dist P F₁ + dist P F₂ ≥ dist Y F₁ + dist Y F₂) →
  dist F₁ F₂ = 43.835 := by
sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l2899_289971


namespace NUMINAMATH_CALUDE_jessica_scores_mean_l2899_289908

def jessica_scores : List ℝ := [87, 94, 85, 92, 90, 88]

theorem jessica_scores_mean :
  (jessica_scores.sum / jessica_scores.length : ℝ) = 89.3333333333333 := by
  sorry

end NUMINAMATH_CALUDE_jessica_scores_mean_l2899_289908


namespace NUMINAMATH_CALUDE_smallest_b_probability_l2899_289988

/-- The number of cards in the deck -/
def deckSize : ℕ := 40

/-- The probability that Carly and Fiona are on the same team when Carly picks card number b and Fiona picks card number b+7 -/
def q (b : ℕ) : ℚ :=
  let totalCombinations := (deckSize - 2).choose 2
  let lowerTeamCombinations := (deckSize - b - 7).choose 2
  let higherTeamCombinations := (b - 1).choose 2
  (lowerTeamCombinations + higherTeamCombinations : ℚ) / totalCombinations

/-- The smallest value of b for which q(b) ≥ 1/2 -/
def smallestB : ℕ := 18

theorem smallest_b_probability (b : ℕ) :
  b < smallestB → q b < 1/2 ∧
  q smallestB = 318/703 :=
sorry

end NUMINAMATH_CALUDE_smallest_b_probability_l2899_289988


namespace NUMINAMATH_CALUDE_log_equation_solution_l2899_289948

theorem log_equation_solution (x : ℝ) (h1 : x > 0) (h2 : x ≠ 1) :
  (Real.log x / Real.log 4) * (Real.log 8 / Real.log x) = Real.log 8 / Real.log 4 :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2899_289948


namespace NUMINAMATH_CALUDE_gilbert_herb_plants_l2899_289939

/-- The number of herb plants Gilbert had at the end of spring -/
def herb_plants_at_end_of_spring : ℕ :=
  let initial_basil : ℕ := 3
  let initial_parsley : ℕ := 1
  let initial_mint : ℕ := 2
  let new_basil : ℕ := 1
  let eaten_mint : ℕ := 2
  (initial_basil + initial_parsley + initial_mint + new_basil) - eaten_mint

theorem gilbert_herb_plants : herb_plants_at_end_of_spring = 5 := by
  sorry

end NUMINAMATH_CALUDE_gilbert_herb_plants_l2899_289939


namespace NUMINAMATH_CALUDE_kathryn_remaining_money_l2899_289961

/-- Calculates the remaining money for Kathryn after expenses --/
def remaining_money (rent : ℕ) (salary : ℕ) : ℕ :=
  let food_travel : ℕ := 2 * rent
  let rent_share : ℕ := rent / 2
  let total_expenses : ℕ := rent_share + food_travel
  salary - total_expenses

/-- Proves that Kathryn's remaining money after expenses is $2000 --/
theorem kathryn_remaining_money :
  remaining_money 1200 5000 = 2000 := by
  sorry

#eval remaining_money 1200 5000

end NUMINAMATH_CALUDE_kathryn_remaining_money_l2899_289961


namespace NUMINAMATH_CALUDE_wire_cutting_l2899_289999

theorem wire_cutting (total_length : ℝ) (ratio : ℝ) (shorter_length : ℝ) : 
  total_length = 70 →
  ratio = 3 / 7 →
  shorter_length + (shorter_length / ratio) = total_length →
  shorter_length = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_wire_cutting_l2899_289999


namespace NUMINAMATH_CALUDE_sin_eq_cos_necessary_not_sufficient_l2899_289913

open Real

theorem sin_eq_cos_necessary_not_sufficient :
  (∃ α, sin α = cos α ∧ ¬(∃ k : ℤ, α = π / 4 + 2 * k * π)) ∧
  (∀ α, (∃ k : ℤ, α = π / 4 + 2 * k * π) → sin α = cos α) :=
by sorry

end NUMINAMATH_CALUDE_sin_eq_cos_necessary_not_sufficient_l2899_289913


namespace NUMINAMATH_CALUDE_circle_symmetric_l2899_289996

-- Define a circle
def Circle : Type := Unit

-- Define axisymmetric property
def isAxisymmetric (shape : Type) : Prop := sorry

-- Define centrally symmetric property
def isCentrallySymmetric (shape : Type) : Prop := sorry

-- Theorem stating that a circle is both axisymmetric and centrally symmetric
theorem circle_symmetric : isAxisymmetric Circle ∧ isCentrallySymmetric Circle := by
  sorry

end NUMINAMATH_CALUDE_circle_symmetric_l2899_289996


namespace NUMINAMATH_CALUDE_highest_water_level_in_narrow_neck_vase_l2899_289995

/-- Represents a vase with a specific shape --/
inductive VaseShape
  | NarrowNeck
  | Symmetrical
  | WideTop

/-- Represents a vase with its properties --/
structure Vase where
  shape : VaseShape
  height : ℝ
  volume : ℝ

/-- Calculates the water level in a vase given the amount of water --/
noncomputable def waterLevel (v : Vase) (waterAmount : ℝ) : ℝ :=
  sorry

theorem highest_water_level_in_narrow_neck_vase 
  (vases : Fin 5 → Vase)
  (h_same_height : ∀ i j, (vases i).height = (vases j).height)
  (h_same_volume : ∀ i, (vases i).volume = 1)
  (h_water_amount : ∀ i, waterLevel (vases i) 0.5 > 0)
  (h_vase_a_narrow : (vases 0).shape = VaseShape.NarrowNeck)
  (h_other_shapes : ∀ i, i ≠ 0 → (vases i).shape ≠ VaseShape.NarrowNeck) :
  ∀ i, i ≠ 0 → waterLevel (vases 0) 0.5 > waterLevel (vases i) 0.5 :=
sorry

end NUMINAMATH_CALUDE_highest_water_level_in_narrow_neck_vase_l2899_289995


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2899_289909

/-- Given a regular polygon inscribed in a circle, if the central angle corresponding to one side is 72°, then the polygon has 5 sides. -/
theorem regular_polygon_sides (n : ℕ) (central_angle : ℝ) : 
  n ≥ 3 → 
  central_angle = 72 → 
  (360 : ℝ) / n = central_angle → 
  n = 5 := by sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2899_289909


namespace NUMINAMATH_CALUDE_cone_volume_from_semicircle_l2899_289984

/-- The volume of a cone whose development diagram is a semicircle with radius 2 -/
theorem cone_volume_from_semicircle (r : Real) (l : Real) (h : Real) : 
  l = 2 → 
  2 * π * r = π * 2 → 
  h^2 + r^2 = l^2 → 
  (1/3 : Real) * π * r^2 * h = (Real.sqrt 3 * π) / 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_from_semicircle_l2899_289984


namespace NUMINAMATH_CALUDE_max_value_abc_l2899_289930

theorem max_value_abc (a b c : ℝ) (h : 9 * a^2 + 4 * b^2 + 25 * c^2 = 1) :
  10 * a + 3 * b + 15 * c ≤ Real.sqrt (337 / 36) :=
by sorry

end NUMINAMATH_CALUDE_max_value_abc_l2899_289930


namespace NUMINAMATH_CALUDE_shooting_probability_l2899_289977

/-- The probability of hitting a shot -/
def shooting_accuracy : ℚ := 9/10

/-- The probability of hitting two consecutive shots -/
def two_consecutive_hits : ℚ := 1/2

/-- The probability of hitting the next shot given that the first shot was hit -/
def next_shot_probability : ℚ := 5/9

theorem shooting_probability :
  shooting_accuracy = 9/10 →
  two_consecutive_hits = 1/2 →
  next_shot_probability = two_consecutive_hits / shooting_accuracy :=
by sorry

end NUMINAMATH_CALUDE_shooting_probability_l2899_289977


namespace NUMINAMATH_CALUDE_sum_x_coordinates_on_parabola_l2899_289905

/-- The parabola equation y = x² - 2x + 1 -/
def parabola (x : ℝ) : ℝ := x^2 - 2*x + 1

/-- Theorem: For any two points P(x₁, 1) and Q(x₂, 1) on the parabola y = x² - 2x + 1,
    the sum of their x-coordinates (x₁ + x₂) is equal to 2. -/
theorem sum_x_coordinates_on_parabola (x₁ x₂ : ℝ) 
    (h₁ : parabola x₁ = 1) 
    (h₂ : parabola x₂ = 1) : 
  x₁ + x₂ = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_x_coordinates_on_parabola_l2899_289905


namespace NUMINAMATH_CALUDE_number_of_elements_l2899_289990

theorem number_of_elements (incorrect_avg : ℝ) (correct_avg : ℝ) (difference : ℝ) : 
  incorrect_avg = 16 → correct_avg = 17 → difference = 10 →
  ∃ n : ℕ, n * correct_avg = n * incorrect_avg + difference ∧ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_number_of_elements_l2899_289990


namespace NUMINAMATH_CALUDE_problem_statement_l2899_289980

theorem problem_statement (m n : ℝ) (h : |m - 3| + (n + 2)^2 = 0) : m + 2*n = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2899_289980


namespace NUMINAMATH_CALUDE_kindergarten_attendance_l2899_289967

/-- Calculates the total number of students present in two kindergarten sessions -/
def total_students (morning_registered : Nat) (morning_absent : Nat) 
                   (afternoon_registered : Nat) (afternoon_absent : Nat) : Nat :=
  (morning_registered - morning_absent) + (afternoon_registered - afternoon_absent)

/-- Theorem: The total number of students present over two kindergarten sessions is 42 -/
theorem kindergarten_attendance : 
  total_students 25 3 24 4 = 42 := by
  sorry

end NUMINAMATH_CALUDE_kindergarten_attendance_l2899_289967


namespace NUMINAMATH_CALUDE_congruence_problem_l2899_289906

theorem congruence_problem : 
  ∀ n : ℤ, 10 ≤ n ∧ n ≤ 20 ∧ n % 7 = 12345 % 7 → n = 11 ∨ n = 18 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l2899_289906


namespace NUMINAMATH_CALUDE_fraction_bounds_l2899_289962

theorem fraction_bounds (x y : ℝ) (h : x^2*y^2 + x*y + 1 = 3*y^2) :
  let F := (y - x) / (x + 4*y)
  0 ≤ F ∧ F ≤ 4 := by sorry

end NUMINAMATH_CALUDE_fraction_bounds_l2899_289962


namespace NUMINAMATH_CALUDE_welders_problem_l2899_289983

/-- The number of days needed to complete the order with all welders -/
def total_days : ℝ := 3

/-- The number of welders that leave after the first day -/
def leaving_welders : ℕ := 12

/-- The number of additional days needed by remaining welders to complete the order -/
def remaining_days : ℝ := 3.0000000000000004

/-- The initial number of welders -/
def initial_welders : ℕ := 36

theorem welders_problem :
  (initial_welders - leaving_welders : ℝ) / initial_welders * remaining_days = 2 / 3 :=
sorry

end NUMINAMATH_CALUDE_welders_problem_l2899_289983


namespace NUMINAMATH_CALUDE_approx_625_to_four_fifths_l2899_289922

-- Define the problem
theorem approx_625_to_four_fifths : ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ 
  |Real.rpow 625 (4/5) - 238| < ε :=
sorry

end NUMINAMATH_CALUDE_approx_625_to_four_fifths_l2899_289922


namespace NUMINAMATH_CALUDE_gcd_960_1632_l2899_289969

theorem gcd_960_1632 : Nat.gcd 960 1632 = 96 := by
  sorry

end NUMINAMATH_CALUDE_gcd_960_1632_l2899_289969


namespace NUMINAMATH_CALUDE_determinant_of_2x2_matrix_l2899_289956

theorem determinant_of_2x2_matrix :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![7, -2; 3, 5]
  Matrix.det A = 41 := by
  sorry

end NUMINAMATH_CALUDE_determinant_of_2x2_matrix_l2899_289956


namespace NUMINAMATH_CALUDE_distribute_five_prizes_to_three_students_l2899_289912

/-- The number of ways to distribute n different prizes to k students,
    with each student receiving at least one prize -/
def distribute_prizes (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 different prizes to 3 students,
    with each student receiving at least one prize, is 150 -/
theorem distribute_five_prizes_to_three_students :
  distribute_prizes 5 3 = 150 := by sorry

end NUMINAMATH_CALUDE_distribute_five_prizes_to_three_students_l2899_289912


namespace NUMINAMATH_CALUDE_sequence1_correct_sequence2_correct_sequence3_correct_l2899_289911

-- Sequence 1
def sequence1 (n : ℕ) : ℤ := (-1)^n * (6*n - 5)

-- Sequence 2
def sequence2 (n : ℕ) : ℚ := 8/9 * (1 - 1/10^n)

-- Sequence 3
def sequence3 (n : ℕ) : ℚ := (-1)^n * (2^n - 3) / 2^n

theorem sequence1_correct (n : ℕ) : 
  sequence1 1 = -1 ∧ sequence1 2 = 7 ∧ sequence1 3 = -13 ∧ sequence1 4 = 19 := by sorry

theorem sequence2_correct (n : ℕ) : 
  sequence2 1 = 0.8 ∧ sequence2 2 = 0.88 ∧ sequence2 3 = 0.888 := by sorry

theorem sequence3_correct (n : ℕ) : 
  sequence3 1 = -1/2 ∧ sequence3 2 = 1/4 ∧ sequence3 3 = -5/8 ∧ 
  sequence3 4 = 13/16 ∧ sequence3 5 = -29/32 ∧ sequence3 6 = 61/64 := by sorry

end NUMINAMATH_CALUDE_sequence1_correct_sequence2_correct_sequence3_correct_l2899_289911


namespace NUMINAMATH_CALUDE_tangent_line_at_one_symmetry_condition_extreme_values_condition_l2899_289981

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := (1/x + a) * Real.log (1 + x)

theorem tangent_line_at_one (a : ℝ) :
  a = -1 →
  ∃ m b : ℝ, ∀ x : ℝ, (m * (x - 1) + b = f a x) ∧ (m = -Real.log 2) :=
sorry

theorem symmetry_condition (a b : ℝ) :
  (∀ x : ℝ, f a (1/x) = f a (1/(2*b - x))) ↔ (a = 1/2 ∧ b = -1/2) :=
sorry

theorem extreme_values_condition (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ ∀ y : ℝ, y > 0 → f a y ≤ f a x) ↔ (0 < a ∧ a < 1/2) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_symmetry_condition_extreme_values_condition_l2899_289981


namespace NUMINAMATH_CALUDE_distance_walked_l2899_289941

-- Define the walking time in hours
def walking_time : ℝ := 1.25

-- Define the walking rate in miles per hour
def walking_rate : ℝ := 4.8

-- Theorem statement
theorem distance_walked : walking_time * walking_rate = 6 := by
  sorry

end NUMINAMATH_CALUDE_distance_walked_l2899_289941


namespace NUMINAMATH_CALUDE_right_triangle_matchsticks_l2899_289904

theorem right_triangle_matchsticks (a b c : ℕ) : 
  a = 6 ∧ b = 8 ∧ c^2 = a^2 + b^2 → a + b + c = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_matchsticks_l2899_289904
