import Mathlib

namespace NUMINAMATH_CALUDE_max_value_sum_of_fractions_l3081_308154

theorem max_value_sum_of_fractions (a b c : ℝ) 
  (nonneg_a : 0 ≤ a) (nonneg_b : 0 ≤ b) (nonneg_c : 0 ≤ c)
  (sum_eq_three : a + b + c = 3) :
  (a * b) / (a + b) + (a * c) / (a + c) + (b * c) / (b + c) ≤ 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_max_value_sum_of_fractions_l3081_308154


namespace NUMINAMATH_CALUDE_f_g_f_2_equals_120_l3081_308183

def f (x : ℝ) : ℝ := 3 * x + 3

def g (x : ℝ) : ℝ := 4 * x + 3

theorem f_g_f_2_equals_120 : f (g (f 2)) = 120 := by
  sorry

end NUMINAMATH_CALUDE_f_g_f_2_equals_120_l3081_308183


namespace NUMINAMATH_CALUDE_basketball_only_count_l3081_308190

theorem basketball_only_count (total students_basketball students_table_tennis students_neither : ℕ) :
  total = 30 ∧
  students_basketball = 15 ∧
  students_table_tennis = 10 ∧
  students_neither = 8 →
  ∃ (students_both : ℕ),
    students_basketball - students_both = 12 ∧
    students_both + (students_basketball - students_both) + (students_table_tennis - students_both) + students_neither = total :=
by sorry

end NUMINAMATH_CALUDE_basketball_only_count_l3081_308190


namespace NUMINAMATH_CALUDE_c_payment_l3081_308173

def work_rate (days : ℕ) : ℚ := 1 / days

def total_payment : ℕ := 3200

def completion_days : ℕ := 3

theorem c_payment (a_days b_days : ℕ) (ha : a_days = 6) (hb : b_days = 8) : 
  let a_rate := work_rate a_days
  let b_rate := work_rate b_days
  let ab_rate := a_rate + b_rate
  let ab_work := ab_rate * completion_days
  let c_work := 1 - ab_work
  c_work * total_payment = 400 := by sorry

end NUMINAMATH_CALUDE_c_payment_l3081_308173


namespace NUMINAMATH_CALUDE_perimeter_of_quarter_circle_bounded_region_l3081_308181

/-- The perimeter of a region bounded by four quarter-circle arcs constructed at each corner of a square with sides measuring 4/π is equal to 4. -/
theorem perimeter_of_quarter_circle_bounded_region : 
  let square_side : ℝ := 4 / Real.pi
  let quarter_circle_radius : ℝ := square_side / 2
  let quarter_circle_perimeter : ℝ := Real.pi * quarter_circle_radius / 2
  let total_perimeter : ℝ := 4 * quarter_circle_perimeter
  total_perimeter = 4 := by sorry

end NUMINAMATH_CALUDE_perimeter_of_quarter_circle_bounded_region_l3081_308181


namespace NUMINAMATH_CALUDE_wheat_distribution_theorem_l3081_308102

def wheat_distribution (x y z : ℕ) : Prop :=
  x + y + z = 100 ∧ 3 * x + 2 * y + (1/2) * z = 100

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(20,0,80), (17,5,78), (14,10,76), (11,15,74), (8,20,72), (5,25,70), (2,30,68)}

theorem wheat_distribution_theorem :
  {p : ℕ × ℕ × ℕ | wheat_distribution p.1 p.2.1 p.2.2} = solution_set :=
sorry

end NUMINAMATH_CALUDE_wheat_distribution_theorem_l3081_308102


namespace NUMINAMATH_CALUDE_james_lego_collection_l3081_308151

/-- Represents the number of Legos in James' collection -/
def initial_collection : ℕ := sorry

/-- Represents the number of Legos James uses for his castle -/
def used_legos : ℕ := sorry

/-- Represents the number of Legos put back in the box -/
def legos_in_box : ℕ := 245

/-- Represents the number of missing Legos -/
def missing_legos : ℕ := 5

theorem james_lego_collection :
  (initial_collection = 500) ∧
  (used_legos = initial_collection / 2) ∧
  (legos_in_box + missing_legos = initial_collection - used_legos) :=
sorry

end NUMINAMATH_CALUDE_james_lego_collection_l3081_308151


namespace NUMINAMATH_CALUDE_second_race_lead_l3081_308138

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ

/-- Represents the race setup -/
structure Race where
  distance : ℝ
  sunny : Runner
  windy : Runner

theorem second_race_lead (h d : ℝ) (first_race second_race : Race) 
  (h_positive : h > 0)
  (d_positive : d > 0)
  (first_race_distance : first_race.distance = 2 * h)
  (second_race_distance : second_race.distance = 2 * h)
  (same_speeds : first_race.sunny.speed = second_race.sunny.speed ∧ 
                 first_race.windy.speed = second_race.windy.speed)
  (first_race_lead : first_race.sunny.speed * first_race.distance = 
                     first_race.windy.speed * (first_race.distance - 2 * d))
  (second_race_start : second_race.sunny.speed * (second_race.distance + 2 * d) = 
                       second_race.windy.speed * second_race.distance) :
  second_race.sunny.speed * second_race.distance - 
  second_race.windy.speed * second_race.distance = 2 * d^2 / h := by
  sorry

end NUMINAMATH_CALUDE_second_race_lead_l3081_308138


namespace NUMINAMATH_CALUDE_absolute_value_half_l3081_308115

theorem absolute_value_half (a : ℝ) : 
  |a| = 1/2 → (a = 1/2 ∨ a = -1/2) := by sorry

end NUMINAMATH_CALUDE_absolute_value_half_l3081_308115


namespace NUMINAMATH_CALUDE_solutions_to_z_fourth_equals_16_l3081_308178

theorem solutions_to_z_fourth_equals_16 : 
  {z : ℂ | z^4 = 16} = {2, -2, 2*I, -2*I} := by sorry

end NUMINAMATH_CALUDE_solutions_to_z_fourth_equals_16_l3081_308178


namespace NUMINAMATH_CALUDE_function_zeros_imply_k_range_l3081_308108

open Real

theorem function_zeros_imply_k_range (f : ℝ → ℝ) (k : ℝ) :
  (∀ x, f x = (log x) / x - k * x) →
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ 1/ℯ ≤ x₁ ∧ x₁ ≤ ℯ^2 ∧ 1/ℯ ≤ x₂ ∧ x₂ ≤ ℯ^2 ∧ f x₁ = 0 ∧ f x₂ = 0) →
  2/ℯ^4 ≤ k ∧ k < 1/(2*ℯ) :=
by sorry

end NUMINAMATH_CALUDE_function_zeros_imply_k_range_l3081_308108


namespace NUMINAMATH_CALUDE_triangle_area_l3081_308114

/-- Given a triangle ABC with the following properties:
  * sinB = √2 * sinA
  * ∠C = 105°
  * c = √3 + 1
  Prove that the area of triangle ABC is (√3 + 1) / 2 -/
theorem triangle_area (A B C : ℝ) (h1 : Real.sin B = Real.sqrt 2 * Real.sin A)
  (h2 : C = 105 * π / 180) (h3 : Real.sqrt 3 + 1 = 2 * Real.sin (C / 2) * Real.sin ((A + B) / 2)) :
  (Real.sqrt 3 + 1) / 2 = (Real.sin C) * (Real.sin A) * (Real.sin B) / (Real.sin (A + B)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3081_308114


namespace NUMINAMATH_CALUDE_angle_B_value_side_b_value_l3081_308187

/-- Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2

/-- The sine law holds for the triangle -/
axiom sine_law (t : AcuteTriangle) : t.a / Real.sin t.A = t.b / Real.sin t.B

/-- The cosine law holds for the triangle -/
axiom cosine_law (t : AcuteTriangle) : t.b^2 = t.a^2 + t.c^2 - 2*t.a*t.c*Real.cos t.B

/-- The given condition a = 2b sin A -/
def condition (t : AcuteTriangle) : Prop := t.a = 2*t.b*Real.sin t.A

theorem angle_B_value (t : AcuteTriangle) (h : condition t) : t.B = π/6 := by sorry

theorem side_b_value (t : AcuteTriangle) (h1 : t.a = 3*Real.sqrt 3) (h2 : t.c = 5) (h3 : t.B = π/6) : 
  t.b = Real.sqrt 7 := by sorry

end NUMINAMATH_CALUDE_angle_B_value_side_b_value_l3081_308187


namespace NUMINAMATH_CALUDE_complement_of_P_l3081_308112

def U := Set ℝ
def P : Set ℝ := {x | x^2 ≤ 1}

theorem complement_of_P : (Set.univ \ P) = {x | x < -1 ∨ x > 1} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_P_l3081_308112


namespace NUMINAMATH_CALUDE_decreasing_linear_function_k_range_l3081_308132

-- Define the linear function
def f (k : ℝ) (x : ℝ) : ℝ := (2*k - 4)*x - 1

-- State the theorem
theorem decreasing_linear_function_k_range (k : ℝ) :
  (∀ x y : ℝ, x < y → f k x > f k y) → k < 2 := by sorry

end NUMINAMATH_CALUDE_decreasing_linear_function_k_range_l3081_308132


namespace NUMINAMATH_CALUDE_square_side_length_l3081_308125

theorem square_side_length (diagonal : ℝ) (h : diagonal = 2) :
  ∃ (side : ℝ), side * side * 2 = diagonal * diagonal ∧ side = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l3081_308125


namespace NUMINAMATH_CALUDE_distance_Y_to_GH_l3081_308159

-- Define the square
def Square (t : ℝ) := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ t ∧ 0 ≤ p.2 ∧ p.2 ≤ t}

-- Define the half-circle centered at E
def ArcE (t : ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = (t/2)^2 ∧ 0 ≤ p.1 ∧ 0 ≤ p.2}

-- Define the half-circle centered at F
def ArcF (t : ℝ) := {p : ℝ × ℝ | (p.1 - t)^2 + p.2^2 = (3*t/2)^2 ∧ p.1 ≤ t ∧ 0 ≤ p.2}

-- Define the intersection point Y
def Y (t : ℝ) := {p : ℝ × ℝ | p ∈ ArcE t ∧ p ∈ ArcF t ∧ p ∈ Square t}

-- Theorem statement
theorem distance_Y_to_GH (t : ℝ) (h : t > 0) :
  ∀ y ∈ Y t, t - y.2 = t :=
sorry

end NUMINAMATH_CALUDE_distance_Y_to_GH_l3081_308159


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_one_l3081_308155

theorem fraction_zero_implies_x_one :
  ∀ x : ℝ, (x - 1) / (2 * x - 4) = 0 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_one_l3081_308155


namespace NUMINAMATH_CALUDE_lower_selling_price_l3081_308120

theorem lower_selling_price 
  (cost : ℕ) 
  (higher_price lower_price : ℕ) 
  (h1 : cost = 400)
  (h2 : higher_price = 600)
  (h3 : (higher_price - cost) = (lower_price - cost) + (cost * 5 / 100)) :
  lower_price = 580 := by
sorry

end NUMINAMATH_CALUDE_lower_selling_price_l3081_308120


namespace NUMINAMATH_CALUDE_at_most_two_solutions_l3081_308194

theorem at_most_two_solutions (a b c : ℝ) (ha : a > 2000) :
  ¬∃ (x₁ x₂ x₃ : ℤ), (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) ∧
    (|a * x₁^2 + b * x₁ + c| ≤ 1000) ∧
    (|a * x₂^2 + b * x₂ + c| ≤ 1000) ∧
    (|a * x₃^2 + b * x₃ + c| ≤ 1000) :=
by sorry

end NUMINAMATH_CALUDE_at_most_two_solutions_l3081_308194


namespace NUMINAMATH_CALUDE_john_donation_increases_average_l3081_308184

/-- Represents the donation amounts of Alice, Bob, and Carol -/
structure Donations where
  alice : ℝ
  bob : ℝ
  carol : ℝ

/-- The conditions of the problem -/
def donation_conditions (d : Donations) : Prop :=
  d.alice > 0 ∧ d.bob > 0 ∧ d.carol > 0 ∧  -- Each student donated a positive amount
  d.alice ≠ d.bob ∧ d.alice ≠ d.carol ∧ d.bob ≠ d.carol ∧  -- Each student donated a different amount
  d.alice / d.bob = 3 / 2 ∧  -- Ratio of Alice's to Bob's donation is 3:2
  d.carol / d.bob = 5 / 2 ∧  -- Ratio of Carol's to Bob's donation is 5:2
  d.alice + d.bob = 120  -- Sum of Alice's and Bob's donations is $120

/-- John's donation -/
def john_donation (d : Donations) : ℝ :=
  240

/-- The theorem to be proved -/
theorem john_donation_increases_average (d : Donations) 
  (h : donation_conditions d) : 
  (d.alice + d.bob + d.carol + john_donation d) / 4 = 
  1.5 * (d.alice + d.bob + d.carol) / 3 := by
  sorry

end NUMINAMATH_CALUDE_john_donation_increases_average_l3081_308184


namespace NUMINAMATH_CALUDE_select_five_from_ten_l3081_308126

theorem select_five_from_ten (n : ℕ) (k : ℕ) : n = 10 ∧ k = 5 → Nat.choose n k = 252 := by
  sorry

end NUMINAMATH_CALUDE_select_five_from_ten_l3081_308126


namespace NUMINAMATH_CALUDE_integer_equation_solution_l3081_308143

theorem integer_equation_solution (x y : ℤ) : x^4 - 2*y^2 = 1 → (x = 1 ∨ x = -1) ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_integer_equation_solution_l3081_308143


namespace NUMINAMATH_CALUDE_circle_symmetry_orthogonality_l3081_308142

-- Define the curve
def curve (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 6*y + 1 = 0

-- Define the symmetry line
def symmetry_line (m : ℝ) (x y : ℝ) : Prop := x + m*y + 4 = 0

-- Define the orthogonality condition
def orthogonal (x1 y1 x2 y2 : ℝ) : Prop := x1*x2 + y1*y2 = 0

theorem circle_symmetry_orthogonality :
  ∃ (m : ℝ) (x1 y1 x2 y2 : ℝ),
    curve x1 y1 ∧ curve x2 y2 ∧
    (∃ (x0 y0 : ℝ), symmetry_line m x0 y0 ∧ 
      (x1 - x0)^2 + (y1 - y0)^2 = (x2 - x0)^2 + (y2 - y0)^2) ∧
    orthogonal x1 y1 x2 y2 →
    m = -1 ∧ y2 - y1 = -(x2 - x1) := by sorry

end NUMINAMATH_CALUDE_circle_symmetry_orthogonality_l3081_308142


namespace NUMINAMATH_CALUDE_smallest_three_digit_divisible_by_4_and_5_l3081_308128

theorem smallest_three_digit_divisible_by_4_and_5 : ∃ n : ℕ, 
  (n ≥ 100 ∧ n < 1000) ∧ 
  (n % 4 = 0 ∧ n % 5 = 0) ∧
  (∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ m % 4 = 0 ∧ m % 5 = 0 → m ≥ n) ∧
  n = 100 := by
  sorry

end NUMINAMATH_CALUDE_smallest_three_digit_divisible_by_4_and_5_l3081_308128


namespace NUMINAMATH_CALUDE_range_when_p_range_when_p_xor_q_l3081_308176

-- Define the quadratic equation
def quadratic_eq (a x : ℝ) : Prop := x^2 + a*x + 1/16 = 0

-- Define the condition p
def condition_p (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ < 0 ∧ x₂ < 0 ∧ quadratic_eq a x₁ ∧ quadratic_eq a x₂

-- Define the condition q
def condition_q (a : ℝ) : Prop := 1/a > 1

-- Theorem 1
theorem range_when_p (a : ℝ) : condition_p a → a > 1/2 := by sorry

-- Theorem 2
theorem range_when_p_xor_q (a : ℝ) : 
  (condition_p a ∨ condition_q a) ∧ ¬(condition_p a ∧ condition_q a) → 
  a ≥ 1 ∨ (0 < a ∧ a ≤ 1/2) := by sorry

end NUMINAMATH_CALUDE_range_when_p_range_when_p_xor_q_l3081_308176


namespace NUMINAMATH_CALUDE_sum_first_150_remainder_l3081_308139

theorem sum_first_150_remainder (n : Nat) (sum : Nat) : 
  n = 150 → 
  sum = n * (n + 1) / 2 → 
  sum % 11300 = 25 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_150_remainder_l3081_308139


namespace NUMINAMATH_CALUDE_seed_calculation_total_seed_gallons_l3081_308131

/-- Calculates the total gallons of seed used for a football field given the specified conditions -/
theorem seed_calculation (field_area : ℝ) (seed_ratio : ℝ) (combined_gallons : ℝ) (combined_area : ℝ) : ℝ :=
  let total_parts := seed_ratio + 1
  let seed_fraction := seed_ratio / total_parts
  let seed_per_combined_area := seed_fraction * combined_gallons
  let field_coverage_factor := field_area / combined_area
  field_coverage_factor * seed_per_combined_area

/-- Proves that the total gallons of seed used for the entire football field is 768 gallons -/
theorem total_seed_gallons :
  seed_calculation 8000 4 240 2000 = 768 := by
  sorry

end NUMINAMATH_CALUDE_seed_calculation_total_seed_gallons_l3081_308131


namespace NUMINAMATH_CALUDE_max_value_expression_l3081_308167

theorem max_value_expression : 
  (∀ x : ℝ, (3 * x^2 + 9 * x + 28) / (3 * x^2 + 9 * x + 7) ≤ 85) ∧ 
  (∀ ε > 0, ∃ x : ℝ, (3 * x^2 + 9 * x + 28) / (3 * x^2 + 9 * x + 7) > 85 - ε) :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l3081_308167


namespace NUMINAMATH_CALUDE_library_wall_leftover_space_l3081_308113

theorem library_wall_leftover_space
  (wall_length : ℝ)
  (desk_length : ℝ)
  (bookcase_length : ℝ)
  (min_spacing : ℝ)
  (h_wall : wall_length = 15)
  (h_desk : desk_length = 2)
  (h_bookcase : bookcase_length = 1.5)
  (h_spacing : min_spacing = 0.5)
  : ∃ (n : ℕ), 
    n * (desk_length + bookcase_length + min_spacing) ≤ wall_length ∧
    (n + 1) * (desk_length + bookcase_length + min_spacing) > wall_length ∧
    wall_length - n * (desk_length + bookcase_length + min_spacing) = 3 :=
by sorry

end NUMINAMATH_CALUDE_library_wall_leftover_space_l3081_308113


namespace NUMINAMATH_CALUDE_construct_m_is_perfect_square_l3081_308186

/-- The number of 1's in the sequence -/
def num_ones : ℕ := 1997

/-- The number of 2's in the sequence -/
def num_twos : ℕ := 1998

/-- Constructs the number m as described in the problem -/
def construct_m : ℕ :=
  let n := (10^num_ones * (10^num_ones - 1) + 2 * (10^num_ones - 1)) / 9
  10 * n + 25

/-- Theorem stating that the constructed number m is a perfect square -/
theorem construct_m_is_perfect_square : ∃ k : ℕ, construct_m = k^2 := by
  sorry

end NUMINAMATH_CALUDE_construct_m_is_perfect_square_l3081_308186


namespace NUMINAMATH_CALUDE_complex_fourth_quadrant_m_range_l3081_308164

theorem complex_fourth_quadrant_m_range (m : ℝ) :
  let z : ℂ := (m + 3) + (m - 1) * I
  (z.re > 0 ∧ z.im < 0) ↔ -3 < m ∧ m < 1 := by sorry

end NUMINAMATH_CALUDE_complex_fourth_quadrant_m_range_l3081_308164


namespace NUMINAMATH_CALUDE_watermelon_melon_weight_comparison_l3081_308157

theorem watermelon_melon_weight_comparison (W M : ℝ) 
  (h1 : W > 0) (h2 : M > 0)
  (h3 : (2*W > 3*M) ∨ (3*W > 4*M))
  (h4 : ¬((2*W > 3*M) ∧ (3*W > 4*M))) :
  ¬(12*W > 18*M) := by
sorry

end NUMINAMATH_CALUDE_watermelon_melon_weight_comparison_l3081_308157


namespace NUMINAMATH_CALUDE_banana_permutations_l3081_308137

/-- The number of permutations of a multiset -/
def multiset_permutations (n : ℕ) (frequencies : List ℕ) : ℕ :=
  Nat.factorial n / (List.prod (List.map Nat.factorial frequencies))

/-- Theorem: The number of distinct permutations of BANANA is 60 -/
theorem banana_permutations :
  multiset_permutations 6 [3, 2, 1] = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_permutations_l3081_308137


namespace NUMINAMATH_CALUDE_red_purple_probability_l3081_308195

def total_balls : ℕ := 120
def red_balls : ℕ := 20
def purple_balls : ℕ := 5

theorem red_purple_probability : 
  (red_balls * purple_balls * 2 : ℚ) / (total_balls * (total_balls - 1)) = 5 / 357 := by
  sorry

end NUMINAMATH_CALUDE_red_purple_probability_l3081_308195


namespace NUMINAMATH_CALUDE_tony_squat_weight_l3081_308146

def curl_weight : ℕ := 90

def military_press_weight (curl : ℕ) : ℕ := 2 * curl

def squat_weight (military_press : ℕ) : ℕ := 5 * military_press

theorem tony_squat_weight : 
  squat_weight (military_press_weight curl_weight) = 900 := by
  sorry

end NUMINAMATH_CALUDE_tony_squat_weight_l3081_308146


namespace NUMINAMATH_CALUDE_b_17_value_l3081_308162

/-- A sequence where consecutive terms are roots of a quadratic equation -/
def special_sequence (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n, (a n)^2 - n * (a n) + (b n) = 0 ∧
       (a (n + 1))^2 - n * (a (n + 1)) + (b n) = 0

theorem b_17_value (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (h : special_sequence a b) (h10 : a 10 = 7) : b 17 = 66 := by
  sorry

end NUMINAMATH_CALUDE_b_17_value_l3081_308162


namespace NUMINAMATH_CALUDE_mandy_bike_time_l3081_308174

/-- Represents Mandy's exercise routine --/
structure ExerciseRoutine where
  yoga_time : ℝ
  gym_time : ℝ
  bike_time : ℝ

/-- Theorem: Given Mandy's exercise routine conditions, she spends 18 minutes riding her bike --/
theorem mandy_bike_time (routine : ExerciseRoutine) : 
  routine.yoga_time = 20 →
  routine.gym_time + routine.bike_time = 3/2 * routine.yoga_time →
  routine.gym_time = 2/3 * routine.bike_time →
  routine.bike_time = 18 := by
  sorry


end NUMINAMATH_CALUDE_mandy_bike_time_l3081_308174


namespace NUMINAMATH_CALUDE_sequence_problem_l3081_308158

theorem sequence_problem (x y : ℝ) : 
  (∃ r : ℝ, y - 1 - 1 = 1 - 2*x ∧ 1 - 2*x = r) →  -- arithmetic sequence condition
  (∃ q : ℝ, |x+1| / (y+3) = q ∧ |x-1| / |x+1| = q) →  -- geometric sequence condition
  (x+1)*(y+1) = 4 ∨ (x+1)*(y+1) = 2*(Real.sqrt 17 - 3) := by
sorry

end NUMINAMATH_CALUDE_sequence_problem_l3081_308158


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3081_308153

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h1 : ∀ n, a (n + 1) = q * a n) 
  (S : ℕ → ℝ) 
  (h2 : ∀ n, S n = (a 1) * (1 - q^n) / (1 - q)) 
  (h3 : 2 * S 4 = S 5 + S 6) :
  q = -2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3081_308153


namespace NUMINAMATH_CALUDE_smallest_dual_base_representation_l3081_308180

theorem smallest_dual_base_representation :
  ∃ (a b : ℕ), a > 3 ∧ b > 3 ∧
  (1 * a + 3 = 13) ∧
  (3 * b + 1 = 13) ∧
  (∀ (x y : ℕ), x > 3 → y > 3 →
    (1 * x + 3 = 3 * y + 1) →
    (1 * x + 3 ≥ 13)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_dual_base_representation_l3081_308180


namespace NUMINAMATH_CALUDE_point_inside_circle_l3081_308152

/-- Definition of a circle with center (a, b) and radius r -/
def Circle (a b r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - a)^2 + (p.2 - b)^2 = r^2}

/-- Definition of a point being inside a circle -/
def InsideCircle (p : ℝ × ℝ) (c : Set (ℝ × ℝ)) : Prop :=
  (p.1 - 2)^2 + (p.2 - 3)^2 < 4

/-- The main theorem -/
theorem point_inside_circle :
  let c : Set (ℝ × ℝ) := Circle 2 3 2
  let p : ℝ × ℝ := (1, 2)
  InsideCircle p c := by sorry

end NUMINAMATH_CALUDE_point_inside_circle_l3081_308152


namespace NUMINAMATH_CALUDE_angle_measure_l3081_308104

theorem angle_measure (α : Real) : 
  (90 - α) + (90 - (180 - α)) = 90 → α = 45 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_l3081_308104


namespace NUMINAMATH_CALUDE_number_classification_l3081_308188

-- Define the set of given numbers
def givenNumbers : Set ℝ := {-3, -1, 0, 20, 1/4, -6.5, 17/100, -8.5, 7, Real.pi, 16, -3.14}

-- Define the classification sets
def positiveNumbers : Set ℝ := {x | x > 0}
def integers : Set ℝ := {x | ∃ n : ℤ, x = n}
def fractions : Set ℝ := {x | ∃ a b : ℤ, b ≠ 0 ∧ x = a / b}
def positiveIntegers : Set ℝ := {x | ∃ n : ℕ, x = n ∧ n > 0}
def nonNegativeRationals : Set ℝ := {x | x ≥ 0 ∧ ∃ a b : ℤ, b ≠ 0 ∧ x = a / b}

-- Theorem statement
theorem number_classification :
  (givenNumbers ∩ positiveNumbers = {20, 1/4, 17/100, 7, 16, Real.pi}) ∧
  (givenNumbers ∩ integers = {-3, -1, 0, 20, 7, 16}) ∧
  (givenNumbers ∩ fractions = {1/4, -6.5, 17/100, -8.5, -3.14}) ∧
  (givenNumbers ∩ positiveIntegers = {20, 7, 16}) ∧
  (givenNumbers ∩ nonNegativeRationals = {0, 20, 1/4, 17/100, 7, 16}) := by
  sorry

end NUMINAMATH_CALUDE_number_classification_l3081_308188


namespace NUMINAMATH_CALUDE_square_sum_given_sum_and_product_l3081_308197

theorem square_sum_given_sum_and_product (x y : ℝ) 
  (h1 : (x + y)^2 = 36) (h2 : x * y = 8) : x^2 + y^2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_sum_and_product_l3081_308197


namespace NUMINAMATH_CALUDE_figure_can_form_square_l3081_308177

/-- Represents a point on a 2D plane --/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle on a 2D plane --/
structure Triangle :=
  (a : Point)
  (b : Point)
  (c : Point)

/-- Represents the original figure --/
def OriginalFigure : Type := List Point

/-- Represents a square --/
structure Square :=
  (topLeft : Point)
  (sideLength : ℝ)

/-- Function to cut the original figure into 5 triangles --/
def cutIntoTriangles (figure : OriginalFigure) : List Triangle := sorry

/-- Function to check if a list of triangles can form a square --/
def canFormSquare (triangles : List Triangle) : Prop := sorry

/-- Theorem stating that the original figure can be cut into 5 triangles and rearranged to form a square --/
theorem figure_can_form_square (figure : OriginalFigure) : 
  ∃ (triangles : List Triangle), 
    triangles = cutIntoTriangles figure ∧ 
    triangles.length = 5 ∧ 
    canFormSquare triangles := sorry

end NUMINAMATH_CALUDE_figure_can_form_square_l3081_308177


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3081_308144

theorem complex_fraction_simplification :
  (7 : ℂ) + 9*I / (3 : ℂ) - 4*I = 57/25 + 55/25*I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3081_308144


namespace NUMINAMATH_CALUDE_simplify_product_l3081_308168

theorem simplify_product (b : R) [CommRing R] :
  (2 : R) * b * (3 : R) * b^2 * (4 : R) * b^3 * (5 : R) * b^4 * (6 : R) * b^5 = (720 : R) * b^15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_product_l3081_308168


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3081_308171

theorem complex_equation_solution (z : ℂ) : (1 + 3*I)*z = I - 3 → z = I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3081_308171


namespace NUMINAMATH_CALUDE_cos_thirty_degrees_l3081_308191

theorem cos_thirty_degrees : Real.cos (π / 6) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_thirty_degrees_l3081_308191


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l3081_308192

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_minimum_value
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_geom : GeometricSequence a)
  (h_cond : a 7 = a 6 + 2 * a 5)
  (h_exists : ∃ m n : ℕ, m > 0 ∧ n > 0 ∧ (a m * a n).sqrt = 4 * a 1) :
  (∃ m n : ℕ, m > 0 ∧ n > 0 ∧ (a m * a n).sqrt = 4 * a 1 ∧
    ∀ k l : ℕ, k > 0 → l > 0 → (a k * a l).sqrt = 4 * a 1 →
      1 / m + 4 / n ≤ 1 / k + 4 / l) ∧
  (∃ m n : ℕ, m > 0 ∧ n > 0 ∧ (a m * a n).sqrt = 4 * a 1 ∧
    1 / m + 4 / n = 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l3081_308192


namespace NUMINAMATH_CALUDE_fraction_inequality_solution_l3081_308169

theorem fraction_inequality_solution (x : ℝ) : 
  2 / (x + 2) + 4 / (x + 4) ≥ 1 ↔ 
  x < -4 ∨ (-2 < x ∧ x < -Real.sqrt 8) ∨ x > Real.sqrt 8 :=
by sorry

end NUMINAMATH_CALUDE_fraction_inequality_solution_l3081_308169


namespace NUMINAMATH_CALUDE_largest_gold_coins_distribution_l3081_308165

theorem largest_gold_coins_distribution (n : ℕ) : 
  (n % 13 = 3) →  -- Condition: 3 people receive an extra coin after equal distribution
  (n < 150) →     -- Condition: Total coins less than 150
  (∀ m : ℕ, (m % 13 = 3) ∧ (m < 150) → m ≤ n) →  -- n is the largest number satisfying conditions
  n = 146 :=      -- Conclusion: The largest number of coins is 146
by sorry

end NUMINAMATH_CALUDE_largest_gold_coins_distribution_l3081_308165


namespace NUMINAMATH_CALUDE_probability_of_same_number_l3081_308198

/-- The upper bound for the selected numbers -/
def upper_bound : ℕ := 500

/-- Billy's number is a multiple of this value -/
def billy_multiple : ℕ := 20

/-- Bobbi's number is a multiple of this value -/
def bobbi_multiple : ℕ := 30

/-- The probability of Billy and Bobbi selecting the same number -/
def same_number_probability : ℚ := 1 / 50

/-- Theorem stating the probability of Billy and Bobbi selecting the same number -/
theorem probability_of_same_number :
  (∃ (b₁ b₂ : ℕ), b₁ > 0 ∧ b₂ > 0 ∧ b₁ < upper_bound ∧ b₂ < upper_bound ∧
   b₁ % billy_multiple = 0 ∧ b₂ % bobbi_multiple = 0) →
  same_number_probability = 1 / 50 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_same_number_l3081_308198


namespace NUMINAMATH_CALUDE_max_expected_value_l3081_308103

/-- The probability of winning when there are n red balls and 5 white balls -/
def probability (n : ℕ) : ℚ :=
  (10 * n) / ((n + 5) * (n + 4))

/-- The expected value of the game when there are n red balls -/
def expected_value (n : ℕ) : ℚ :=
  2 * probability n - 1

/-- Theorem stating that the expected value is maximized when n is 4 or 5 -/
theorem max_expected_value :
  ∀ n : ℕ, n > 0 → (expected_value n ≤ expected_value 4 ∧ expected_value n ≤ expected_value 5) :=
by sorry

end NUMINAMATH_CALUDE_max_expected_value_l3081_308103


namespace NUMINAMATH_CALUDE_rectangle_area_l3081_308135

theorem rectangle_area (square_area : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ) : 
  square_area = 36 →
  rectangle_width ^ 2 = square_area →
  rectangle_length = 3 * rectangle_width →
  rectangle_width * rectangle_length = 108 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l3081_308135


namespace NUMINAMATH_CALUDE_wickets_before_match_l3081_308166

/-- Represents a cricketer's bowling statistics -/
structure BowlingStats where
  wickets : ℕ
  runs : ℕ
  average : ℚ

/-- Calculates the new average after a match -/
def newAverage (stats : BowlingStats) (newWickets : ℕ) (newRuns : ℕ) : ℚ :=
  (stats.runs + newRuns) / (stats.wickets + newWickets)

/-- Theorem: The cricketer had taken 85 wickets before the match -/
theorem wickets_before_match (stats : BowlingStats) : 
  stats.average = 12.4 →
  newAverage stats 5 26 = 12 →
  stats.wickets = 85 := by
  sorry

#check wickets_before_match

end NUMINAMATH_CALUDE_wickets_before_match_l3081_308166


namespace NUMINAMATH_CALUDE_probability_of_red_ball_l3081_308111

/-- The probability of drawing a red ball from a bag containing 2 yellow balls and 3 red balls -/
theorem probability_of_red_ball (yellow_balls red_balls : ℕ) 
  (h1 : yellow_balls = 2)
  (h2 : red_balls = 3) :
  (red_balls : ℚ) / ((yellow_balls + red_balls) : ℚ) = 3 / 5 := by
  sorry

#check probability_of_red_ball

end NUMINAMATH_CALUDE_probability_of_red_ball_l3081_308111


namespace NUMINAMATH_CALUDE_statements_equivalent_l3081_308127

-- Define a triangle
structure Triangle where
  angles : Fin 3 → ℝ
  sum_angles : angles 0 + angles 1 + angles 2 = π

-- Define an isosceles triangle
def isIsosceles (t : Triangle) : Prop :=
  t.angles 0 = t.angles 1 ∨ t.angles 1 = t.angles 2 ∨ t.angles 0 = t.angles 2

-- Define the three statements
def statement1 (t : Triangle) : Prop :=
  (∃ i j : Fin 3, i ≠ j ∧ t.angles i = t.angles j) → isIsosceles t

def statement2 (t : Triangle) : Prop :=
  ¬isIsosceles t → (∀ i j : Fin 3, i ≠ j → t.angles i ≠ t.angles j)

def statement3 (t : Triangle) : Prop :=
  (∃ i j : Fin 3, i ≠ j ∧ t.angles i = t.angles j) → isIsosceles t

-- Theorem: The three statements are logically equivalent
theorem statements_equivalent : ∀ t : Triangle,
  (statement1 t ↔ statement2 t) ∧ (statement2 t ↔ statement3 t) :=
sorry

end NUMINAMATH_CALUDE_statements_equivalent_l3081_308127


namespace NUMINAMATH_CALUDE_linear_equation_solution_l3081_308141

theorem linear_equation_solution (x₁ y₁ x₂ y₂ : ℤ) :
  x₁ = 1 ∧ y₁ = -2 ∧ x₂ = -1 ∧ y₂ = -4 →
  x₁ - y₁ = 3 ∧ x₂ - y₂ = 3 :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l3081_308141


namespace NUMINAMATH_CALUDE_butterflies_in_garden_l3081_308116

/-- The number of butterflies left in the garden after some fly away -/
def butterflies_left (initial : ℕ) : ℕ :=
  initial - initial / 3

/-- Theorem stating that for 9 initial butterflies, 6 are left after one-third fly away -/
theorem butterflies_in_garden : butterflies_left 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_butterflies_in_garden_l3081_308116


namespace NUMINAMATH_CALUDE_smallest_m_for_distinct_roots_l3081_308163

theorem smallest_m_for_distinct_roots (m : ℤ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 4*x - m = 0 ∧ y^2 + 4*y - m = 0) → 
  (∀ k : ℤ, k < m → ¬∃ x y : ℝ, x ≠ y ∧ x^2 + 4*x - k = 0 ∧ y^2 + 4*y - k = 0) →
  m = -3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_for_distinct_roots_l3081_308163


namespace NUMINAMATH_CALUDE_roots_not_analytically_determinable_l3081_308189

/-- The polynomial equation whose roots we want to determine -/
def f (x : ℝ) : ℝ := (x - 2) * (x + 5)^3 * (5 - x) - 8

/-- Theorem stating that the roots of the polynomial equation cannot be determined analytically -/
theorem roots_not_analytically_determinable :
  ¬ ∃ (roots : Set ℝ), ∀ (x : ℝ), x ∈ roots ↔ f x = 0 ∧ 
  ∃ (formula : ℝ → ℝ), ∀ (x : ℝ), x ∈ roots → ∃ (n : ℕ), formula x = x ∧ 
  (∀ (y : ℝ), formula y = y → y ∈ roots) :=
sorry

end NUMINAMATH_CALUDE_roots_not_analytically_determinable_l3081_308189


namespace NUMINAMATH_CALUDE_simplify_fraction_l3081_308182

theorem simplify_fraction : (111 : ℚ) / 9999 * 33 = 11 / 3 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3081_308182


namespace NUMINAMATH_CALUDE_gcd_factorial_eight_and_factorial_six_squared_l3081_308100

theorem gcd_factorial_eight_and_factorial_six_squared : Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 5760 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_eight_and_factorial_six_squared_l3081_308100


namespace NUMINAMATH_CALUDE_area_after_shortening_l3081_308105

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle --/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- The original rectangle --/
def original : Rectangle := ⟨5, 7⟩

/-- Shortens either the length or the width of a rectangle by 2 --/
def shorten (r : Rectangle) (shortenLength : Bool) : Rectangle :=
  if shortenLength then ⟨r.length - 2, r.width⟩ else ⟨r.length, r.width - 2⟩

theorem area_after_shortening :
  (area (shorten original true) = 21 ∧ area (shorten original false) = 25) ∨
  (area (shorten original true) = 25 ∧ area (shorten original false) = 21) :=
by sorry

end NUMINAMATH_CALUDE_area_after_shortening_l3081_308105


namespace NUMINAMATH_CALUDE_max_value_F_l3081_308136

theorem max_value_F (a b c : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → |a * x^2 + b * x + c| ≤ 1) →
  (∃ M : ℝ, M = 2 ∧ ∀ x : ℝ, x ∈ Set.Icc (-1) 1 → 
    |(a * x^2 + b * x + c) * (c * x^2 + b * x + a)| ≤ M ∧
    ∃ y : ℝ, y ∈ Set.Icc (-1) 1 ∧ 
      |(a * y^2 + b * y + c) * (c * y^2 + b * y + a)| = M) :=
by sorry

end NUMINAMATH_CALUDE_max_value_F_l3081_308136


namespace NUMINAMATH_CALUDE_correct_equation_representation_l3081_308118

/-- Represents a rectangular field with width and length in steps -/
structure RectangularField where
  width : ℝ
  length : ℝ

/-- The area of a rectangular field in square steps -/
def area (field : RectangularField) : ℝ :=
  field.width * field.length

/-- Theorem stating that the equation x(x+12) = 864 correctly represents the problem -/
theorem correct_equation_representation (x : ℝ) :
  let field := RectangularField.mk x (x + 12)
  area field = 864 → x * (x + 12) = 864 := by
  sorry

end NUMINAMATH_CALUDE_correct_equation_representation_l3081_308118


namespace NUMINAMATH_CALUDE_parallelograms_in_hexagon_l3081_308185

/-- A regular hexagon -/
structure RegularHexagon where
  /-- The number of sides in a regular hexagon -/
  sides : Nat
  /-- The property that a regular hexagon has 6 sides -/
  has_six_sides : sides = 6

/-- A parallelogram formed by two adjacent equilateral triangles in a regular hexagon -/
structure Parallelogram (h : RegularHexagon) where

/-- The number of parallelograms in a regular hexagon -/
def num_parallelograms (h : RegularHexagon) : Nat :=
  h.sides

/-- Theorem: The number of parallelograms in a regular hexagon is 6 -/
theorem parallelograms_in_hexagon (h : RegularHexagon) : 
  num_parallelograms h = 6 := by
  sorry

end NUMINAMATH_CALUDE_parallelograms_in_hexagon_l3081_308185


namespace NUMINAMATH_CALUDE_greatest_p_value_l3081_308172

theorem greatest_p_value (x : ℝ) (p : ℝ) : 
  (∃ x, 2 * Real.cos (2 * Real.pi - Real.pi * x^2 / 6) * Real.cos (Real.pi / 3 * Real.sqrt (9 - x^2)) - 3 = 
        p - 2 * Real.sin (-Real.pi * x^2 / 6) * Real.cos (Real.pi / 3 * Real.sqrt (9 - x^2))) →
  p ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_greatest_p_value_l3081_308172


namespace NUMINAMATH_CALUDE_new_crew_member_weight_l3081_308124

/-- Given a crew of 10 oarsmen, prove that replacing a 53 kg member with a new member
    that increases the average weight by 1.8 kg results in the new member weighing 71 kg. -/
theorem new_crew_member_weight (crew_size : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  crew_size = 10 →
  weight_increase = 1.8 →
  replaced_weight = 53 →
  (crew_size : ℝ) * weight_increase + replaced_weight = 71 := by
  sorry

end NUMINAMATH_CALUDE_new_crew_member_weight_l3081_308124


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3081_308196

theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x-2) - 3
  f 2 = -2 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3081_308196


namespace NUMINAMATH_CALUDE_K_3_15_10_l3081_308106

noncomputable def K (x y z : ℝ) : ℝ := x / y + y / z + z / x

theorem K_3_15_10 : K 3 15 10 = 151 / 30 := by sorry

end NUMINAMATH_CALUDE_K_3_15_10_l3081_308106


namespace NUMINAMATH_CALUDE_ed_lost_seven_marbles_l3081_308160

/-- Represents the number of marbles each person has -/
structure MarbleCount where
  doug : ℕ
  ed : ℕ
  tim : ℕ

/-- The initial state of marble distribution -/
def initial_state (d : ℕ) : MarbleCount :=
  { doug := d
  , ed := d + 19
  , tim := d - 10 }

/-- The final state of marble distribution after transactions -/
def final_state (d : ℕ) (l : ℕ) : MarbleCount :=
  { doug := d
  , ed := d + 8
  , tim := d }

/-- Theorem stating that Ed lost 7 marbles -/
theorem ed_lost_seven_marbles (d : ℕ) :
  ∃ l : ℕ, 
    (initial_state d).ed - l - 4 = (final_state d l).ed ∧
    (initial_state d).tim + 4 + 3 = (final_state d l).tim ∧
    l = 7 := by
  sorry

#check ed_lost_seven_marbles

end NUMINAMATH_CALUDE_ed_lost_seven_marbles_l3081_308160


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3081_308175

/-- Represents the sum of the first n terms of an arithmetic sequence -/
def S (n : ℕ) (a₁ d : ℚ) : ℚ := n / 2 * (2 * a₁ + (n - 1) * d)

/-- Theorem: For an arithmetic sequence with first term a₁ and common difference d,
    if S₈ - S₃ = 20, then S₁₁ = 44 -/
theorem arithmetic_sequence_sum (a₁ d : ℚ) :
  S 8 a₁ d - S 3 a₁ d = 20 → S 11 a₁ d = 44 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3081_308175


namespace NUMINAMATH_CALUDE_monotonic_increase_interval_l3081_308121

theorem monotonic_increase_interval
  (f : ℝ → ℝ)
  (ω φ : ℝ)
  (h_ω : ω > 0)
  (h_φ : 0 < φ ∧ φ < π / 2)
  (h_f : ∀ x, f x = Real.sin (ω * x + φ))
  (x₁ x₂ : ℝ)
  (h_x₁ : f x₁ = 1)
  (h_x₂ : f x₂ = 0)
  (h_x_diff : |x₁ - x₂| = 1 / 2)
  (h_f_half : f (1 / 2) = 1 / 2) :
  ∃ k : ℤ, StrictMonoOn f (Set.Icc (- 5 / 6 + 2 * k) (1 / 6 + 2 * k)) :=
by sorry

end NUMINAMATH_CALUDE_monotonic_increase_interval_l3081_308121


namespace NUMINAMATH_CALUDE_min_value_and_sum_l3081_308140

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + 2 * |x + 1|

-- State the theorem
theorem min_value_and_sum (a b : ℝ) : 
  (∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ (∃ x₀ : ℝ, f x₀ = m) ∧ m = 2) ∧
  (a^2 + b^2 = 2 → 1 / (a^2 + 1) + 4 / (b^2 + 1) ≥ 9/4) :=
sorry

end NUMINAMATH_CALUDE_min_value_and_sum_l3081_308140


namespace NUMINAMATH_CALUDE_inscribed_circle_rectangle_area_l3081_308161

theorem inscribed_circle_rectangle_area :
  ∀ (r : ℝ) (l w : ℝ),
    r = 7 →
    l / w = 3 →
    w = 2 * r →
    l * w = 588 :=
by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_rectangle_area_l3081_308161


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3081_308130

theorem inequality_solution_set : ∀ x : ℝ, 
  (x / 4 ≤ 3 + 2 * x ∧ 3 + 2 * x < -3 * (2 + x)) ↔ 
  x ∈ Set.Icc (-12/7) (-9/5) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3081_308130


namespace NUMINAMATH_CALUDE_no_integer_solutions_l3081_308107

theorem no_integer_solutions :
  ¬ ∃ (m n : ℤ), m^2 - 11*m*n - 8*n^2 = 88 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l3081_308107


namespace NUMINAMATH_CALUDE_interior_lattice_points_collinear_l3081_308156

/-- A lattice point in the plane -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A triangle in the plane -/
structure Triangle where
  v1 : LatticePoint
  v2 : LatticePoint
  v3 : LatticePoint

/-- Check if a point is inside a triangle -/
def isInside (p : LatticePoint) (t : Triangle) : Prop := sorry

/-- Check if a point is on the boundary of a triangle -/
def isOnBoundary (p : LatticePoint) (t : Triangle) : Prop := sorry

/-- Check if points are collinear -/
def areCollinear (p1 p2 p3 p4 : LatticePoint) : Prop := sorry

/-- The main theorem -/
theorem interior_lattice_points_collinear (t : Triangle) 
  (h1 : ∀ p, isOnBoundary p t → (p = t.v1 ∨ p = t.v2 ∨ p = t.v3))
  (h2 : ∃ p1 p2 p3 p4, isInside p1 t ∧ isInside p2 t ∧ isInside p3 t ∧ isInside p4 t ∧
    ∀ p, isInside p t → (p = p1 ∨ p = p2 ∨ p = p3 ∨ p = p4)) :
  ∃ p1 p2 p3 p4, isInside p1 t ∧ isInside p2 t ∧ isInside p3 t ∧ isInside p4 t ∧
    areCollinear p1 p2 p3 p4 := by
  sorry

end NUMINAMATH_CALUDE_interior_lattice_points_collinear_l3081_308156


namespace NUMINAMATH_CALUDE_inscribed_circle_area_l3081_308134

/-- Given an equilateral triangle with a point inside at distances 1, 2, and 4 inches from its sides,
    the area of the inscribed circle is 49π/9 square inches. -/
theorem inscribed_circle_area (s : ℝ) (h : s > 0) : 
  let triangle_area := (7 * s) / 2
  let inscribed_circle_radius := triangle_area / ((3 * s) / 2)
  (π * inscribed_circle_radius ^ 2) = 49 * π / 9 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_area_l3081_308134


namespace NUMINAMATH_CALUDE_sequence_difference_l3081_308129

theorem sequence_difference (p q : ℕ+) (h : p - q = 5) :
  let S : ℕ+ → ℤ := λ n => 2 * n.val ^ 2 - 3 * n.val
  let a : ℕ+ → ℤ := λ n => S n - if n = 1 then 0 else S (n - 1)
  a p - a q = 20 := by
  sorry

end NUMINAMATH_CALUDE_sequence_difference_l3081_308129


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_division_l3081_308147

theorem imaginary_part_of_complex_division (z : ℂ) : 
  z = (3 + 4 * I) / I → Complex.im z = -3 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_division_l3081_308147


namespace NUMINAMATH_CALUDE_right_triangle_sin_c_l3081_308199

theorem right_triangle_sin_c (A B C : ℝ) (h_right : A = 90) (h_sin_b : Real.sin B = 3/5) :
  Real.sin C = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sin_c_l3081_308199


namespace NUMINAMATH_CALUDE_max_change_percentage_l3081_308117

theorem max_change_percentage (initial_yes initial_no final_yes final_no fixed_mindset_ratio : ℚ)
  (h1 : initial_yes + initial_no = 1)
  (h2 : final_yes + final_no = 1)
  (h3 : initial_yes = 2/5)
  (h4 : initial_no = 3/5)
  (h5 : final_yes = 4/5)
  (h6 : final_no = 1/5)
  (h7 : fixed_mindset_ratio = 1/5) :
  let fixed_mindset := fixed_mindset_ratio * initial_no
  let max_change := final_yes - initial_yes
  max_change ≤ initial_no - fixed_mindset ∧ max_change = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_max_change_percentage_l3081_308117


namespace NUMINAMATH_CALUDE_ratio_a_c_l3081_308122

-- Define the ratios
def ratio_a_b : ℚ := 5 / 3
def ratio_b_c : ℚ := 1 / 5

-- Theorem statement
theorem ratio_a_c (a b c : ℚ) (h1 : a / b = ratio_a_b) (h2 : b / c = ratio_b_c) : 
  a / c = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_a_c_l3081_308122


namespace NUMINAMATH_CALUDE_product_cube_square_l3081_308109

theorem product_cube_square : ((-1 : ℤ)^3) * ((-2 : ℤ)^2) = -4 := by sorry

end NUMINAMATH_CALUDE_product_cube_square_l3081_308109


namespace NUMINAMATH_CALUDE_consecutive_numbers_square_l3081_308145

theorem consecutive_numbers_square (a : ℕ) : 
  let b := a + 1
  let c := a * b
  let x := a^2 + b^2 + c^2
  ∃ (k : ℕ), x = (2*k + 1)^2 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_numbers_square_l3081_308145


namespace NUMINAMATH_CALUDE_expected_unpoked_babies_l3081_308110

/-- The number of babies in the circle -/
def num_babies : ℕ := 2006

/-- The probability of a baby poking either of its adjacent neighbors -/
def poke_prob : ℚ := 1/2

/-- The probability of a baby being unpoked -/
def unpoked_prob : ℚ := (1 - poke_prob) * (1 - poke_prob)

/-- The expected number of unpoked babies -/
def expected_unpoked : ℚ := num_babies * unpoked_prob

theorem expected_unpoked_babies :
  expected_unpoked = 1003/2 := by sorry

end NUMINAMATH_CALUDE_expected_unpoked_babies_l3081_308110


namespace NUMINAMATH_CALUDE_tech_personnel_stats_l3081_308148

def intermediate_count : ℕ := 40
def senior_count : ℕ := 10
def total_count : ℕ := intermediate_count + senior_count

def intermediate_avg : ℝ := 35
def senior_avg : ℝ := 45

def intermediate_var : ℝ := 18
def senior_var : ℝ := 73

def total_avg : ℝ := 37
def total_var : ℝ := 45

theorem tech_personnel_stats :
  (intermediate_count * intermediate_avg + senior_count * senior_avg) / total_count = total_avg ∧
  ((intermediate_count * (intermediate_var + intermediate_avg^2) + 
    senior_count * (senior_var + senior_avg^2)) / total_count - total_avg^2) = total_var :=
by sorry

end NUMINAMATH_CALUDE_tech_personnel_stats_l3081_308148


namespace NUMINAMATH_CALUDE_millet_majority_on_friday_l3081_308123

/-- Represents the amount of millet in the feeder on a given day -/
def millet_amount (day : ℕ) : ℚ :=
  0.5 * (1 - (0.7 ^ day))

/-- Represents the total amount of seeds in the feeder on a given day -/
def total_seeds (day : ℕ) : ℚ :=
  0.5 * day

/-- Theorem stating that on the 5th day, more than two-thirds of the seeds are millet -/
theorem millet_majority_on_friday :
  (millet_amount 5) / (total_seeds 5) > 2/3 ∧
  ∀ d : ℕ, d < 5 → (millet_amount d) / (total_seeds d) ≤ 2/3 :=
sorry

end NUMINAMATH_CALUDE_millet_majority_on_friday_l3081_308123


namespace NUMINAMATH_CALUDE_five_sundays_april_implies_five_mondays_may_l3081_308170

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a specific month -/
structure Month where
  numDays : Nat
  firstDay : DayOfWeek

/-- Given a day, returns the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Counts the number of occurrences of a specific day in a month -/
def countDaysInMonth (m : Month) (d : DayOfWeek) : Nat :=
  sorry

/-- Theorem: If April has five Sundays, then May has five Mondays -/
theorem five_sundays_april_implies_five_mondays_may :
  ∀ (april : Month) (may : Month),
    april.numDays = 30 →
    may.numDays = 31 →
    may.firstDay = nextDay april.firstDay →
    countDaysInMonth april DayOfWeek.Sunday = 5 →
    countDaysInMonth may DayOfWeek.Monday = 5 :=
  sorry

end NUMINAMATH_CALUDE_five_sundays_april_implies_five_mondays_may_l3081_308170


namespace NUMINAMATH_CALUDE_parallel_vectors_difference_l3081_308150

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_difference (x : ℝ) :
  let a : ℝ × ℝ := (2, -1)
  let b : ℝ × ℝ := (x - 2, -2)
  are_parallel a b → a - b = (-2, 1) := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_difference_l3081_308150


namespace NUMINAMATH_CALUDE_lollipops_remaining_l3081_308133

def raspberry_lollipops : ℕ := 57
def mint_lollipops : ℕ := 98
def blueberry_lollipops : ℕ := 13
def cola_lollipops : ℕ := 167
def num_friends : ℕ := 13

theorem lollipops_remaining :
  (raspberry_lollipops + mint_lollipops + blueberry_lollipops + cola_lollipops) % num_friends = 10 :=
by sorry

end NUMINAMATH_CALUDE_lollipops_remaining_l3081_308133


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l3081_308193

def P : Set ℝ := {1, 2}
def Q : Set ℝ := {x | |x| < 2}

theorem intersection_of_P_and_Q : P ∩ Q = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l3081_308193


namespace NUMINAMATH_CALUDE_product_scaling_l3081_308149

theorem product_scaling (a b c : ℝ) (h : 14.97 * 46 = 688.62) : 
  1.497 * 4.6 = 6.8862 := by sorry

end NUMINAMATH_CALUDE_product_scaling_l3081_308149


namespace NUMINAMATH_CALUDE_carson_seed_amount_l3081_308119

-- Define variables
variable (seed : ℝ)
variable (fertilizer : ℝ)

-- Define the conditions
def seed_fertilizer_ratio : Prop := seed = 3 * fertilizer
def total_amount : Prop := seed + fertilizer = 60

-- Theorem statement
theorem carson_seed_amount 
  (h1 : seed_fertilizer_ratio seed fertilizer)
  (h2 : total_amount seed fertilizer) :
  seed = 45 := by
  sorry

end NUMINAMATH_CALUDE_carson_seed_amount_l3081_308119


namespace NUMINAMATH_CALUDE_problem_statement_l3081_308179

theorem problem_statement (a b : ℝ) (h : (a + 2)^2 + |b - 1| = 0) :
  (a + b)^2023 = -1 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l3081_308179


namespace NUMINAMATH_CALUDE_stating_min_handshakes_in_gathering_l3081_308101

/-- Represents a gathering of people and their handshakes. -/
structure Gathering where
  people : ℕ
  min_handshakes_per_person : ℕ
  non_handshaking_group : ℕ
  total_handshakes : ℕ

/-- The specific gathering described in the problem. -/
def problem_gathering : Gathering where
  people := 25
  min_handshakes_per_person := 2
  non_handshaking_group := 3
  total_handshakes := 28

/-- 
Theorem stating that the minimum number of handshakes in the given gathering is 28.
-/
theorem min_handshakes_in_gathering (g : Gathering) 
  (h1 : g.people = 25)
  (h2 : g.min_handshakes_per_person = 2)
  (h3 : g.non_handshaking_group = 3) :
  g.total_handshakes = 28 := by
  sorry

#check min_handshakes_in_gathering

end NUMINAMATH_CALUDE_stating_min_handshakes_in_gathering_l3081_308101
