import Mathlib

namespace NUMINAMATH_CALUDE_consecutive_prints_probability_l2104_210407

/-- The number of pieces of art -/
def total_pieces : ℕ := 12

/-- The number of Escher prints -/
def escher_prints : ℕ := 3

/-- The number of Dali prints -/
def dali_prints : ℕ := 2

/-- The probability of Escher and Dali prints being consecutive -/
def consecutive_probability : ℚ := 336 / (Nat.factorial total_pieces)

/-- Theorem stating the probability of Escher and Dali prints being consecutive -/
theorem consecutive_prints_probability :
  consecutive_probability = 336 / (Nat.factorial total_pieces) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_prints_probability_l2104_210407


namespace NUMINAMATH_CALUDE_triangle_area_zero_l2104_210414

def point_a : ℝ × ℝ × ℝ := (2, 3, 1)
def point_b : ℝ × ℝ × ℝ := (8, 6, 4)
def point_c : ℝ × ℝ × ℝ := (14, 9, 7)

def triangle_area (a b c : ℝ × ℝ × ℝ) : ℝ := sorry

theorem triangle_area_zero :
  triangle_area point_a point_b point_c = 0 := by sorry

end NUMINAMATH_CALUDE_triangle_area_zero_l2104_210414


namespace NUMINAMATH_CALUDE_stall_problem_l2104_210450

theorem stall_problem (area_diff : ℝ) (cost_A cost_B : ℝ) (total_area_A total_area_B : ℝ) (total_stalls : ℕ) :
  area_diff = 2 →
  cost_A = 20 →
  cost_B = 40 →
  total_area_A = 150 →
  total_area_B = 120 →
  total_stalls = 100 →
  ∃ (area_A area_B : ℝ) (num_A num_B : ℕ),
    area_A = area_B + area_diff ∧
    (total_area_A / area_A : ℝ) = (3/4) * (total_area_B / area_B) ∧
    num_A + num_B = total_stalls ∧
    num_B ≥ 3 * num_A ∧
    area_A = 5 ∧
    area_B = 3 ∧
    cost_A * area_A * num_A + cost_B * area_B * num_B = 11500 ∧
    ∀ (other_num_A other_num_B : ℕ),
      other_num_A + other_num_B = total_stalls →
      other_num_B ≥ 3 * other_num_A →
      cost_A * area_A * other_num_A + cost_B * area_B * other_num_B ≥ 11500 :=
by sorry

end NUMINAMATH_CALUDE_stall_problem_l2104_210450


namespace NUMINAMATH_CALUDE_derivative_exp_sin_l2104_210477

theorem derivative_exp_sin (x : ℝ) : 
  deriv (fun x => Real.exp (Real.sin x)) x = Real.exp (Real.sin x) * Real.cos x := by
  sorry

end NUMINAMATH_CALUDE_derivative_exp_sin_l2104_210477


namespace NUMINAMATH_CALUDE_stratified_sampling_proportion_l2104_210432

theorem stratified_sampling_proportion (total_male : ℕ) (total_female : ℕ) (selected_male : ℕ) :
  total_male = 56 →
  total_female = 42 →
  selected_male = 8 →
  ∃ (selected_female : ℕ),
    selected_female = 6 ∧
    (selected_male : ℚ) / total_male = (selected_female : ℚ) / total_female :=
by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_proportion_l2104_210432


namespace NUMINAMATH_CALUDE_intersection_empty_set_l2104_210496

theorem intersection_empty_set (A : Set α) : ¬(¬(A ∩ ∅ = ∅)) := by
  sorry

end NUMINAMATH_CALUDE_intersection_empty_set_l2104_210496


namespace NUMINAMATH_CALUDE_intersection_and_union_when_a_is_neg_one_intersection_equals_B_iff_l2104_210479

-- Define sets A and B
def A : Set ℝ := {x | x ≤ -1 ∨ x ≥ 5}
def B (a : ℝ) : Set ℝ := {x | 2*a ≤ x ∧ x ≤ a + 2}

-- Theorem for part (1)
theorem intersection_and_union_when_a_is_neg_one :
  (A ∩ B (-1) = {x | -2 ≤ x ∧ x ≤ -1}) ∧
  (A ∪ B (-1) = {x | x ≤ 1 ∨ x ≥ 5}) := by sorry

-- Theorem for part (2)
theorem intersection_equals_B_iff :
  ∀ a : ℝ, A ∩ B a = B a ↔ a ≤ -3 ∨ a > 2 := by sorry

end NUMINAMATH_CALUDE_intersection_and_union_when_a_is_neg_one_intersection_equals_B_iff_l2104_210479


namespace NUMINAMATH_CALUDE_bikini_fraction_correct_l2104_210405

/-- The fraction of garments that are bikinis at Lindsey's Vacation Wear -/
def bikini_fraction : ℝ := 0.38

/-- The fraction of garments that are trunks at Lindsey's Vacation Wear -/
def trunk_fraction : ℝ := 0.25

/-- The fraction of garments that are either bikinis or trunks at Lindsey's Vacation Wear -/
def bikini_or_trunk_fraction : ℝ := 0.63

/-- Theorem stating that the fraction of garments that are bikinis is correct -/
theorem bikini_fraction_correct :
  bikini_fraction + trunk_fraction = bikini_or_trunk_fraction :=
by sorry

end NUMINAMATH_CALUDE_bikini_fraction_correct_l2104_210405


namespace NUMINAMATH_CALUDE_pyramid_slice_height_l2104_210480

-- Define the pyramid P
structure Pyramid :=
  (base_length : ℝ)
  (base_width : ℝ)
  (height : ℝ)

-- Define the main theorem
theorem pyramid_slice_height (P : Pyramid) (volume_ratio : ℝ) :
  P.base_length = 15 →
  P.base_width = 20 →
  P.height = 30 →
  volume_ratio = 9 →
  (P.height - (P.height / (volume_ratio ^ (1/3 : ℝ)))) = 20 := by
  sorry


end NUMINAMATH_CALUDE_pyramid_slice_height_l2104_210480


namespace NUMINAMATH_CALUDE_f_odd_when_c_zero_f_unique_root_when_b_zero_c_pos_f_symmetric_about_0_c_f_more_than_two_roots_l2104_210413

-- Define the function f
def f (b c x : ℝ) : ℝ := x * |x| + b * x + c

-- Theorem 1: When c = 0, f(-x) = -f(x) for all x
theorem f_odd_when_c_zero (b : ℝ) :
  ∀ x, f b 0 (-x) = -f b 0 x := by sorry

-- Theorem 2: When b = 0 and c > 0, f(x) = 0 has exactly one real root
theorem f_unique_root_when_b_zero_c_pos (c : ℝ) (hc : c > 0) :
  ∃! x, f 0 c x = 0 := by sorry

-- Theorem 3: The graph of y = f(x) is symmetric about (0, c)
theorem f_symmetric_about_0_c (b c : ℝ) :
  ∀ x, f b c (-x) + f b c x = 2 * c := by sorry

-- Theorem 4: There exists a case where f(x) = 0 has more than two real roots
theorem f_more_than_two_roots :
  ∃ b c, ∃ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    f b c x = 0 ∧ f b c y = 0 ∧ f b c z = 0 := by sorry

end NUMINAMATH_CALUDE_f_odd_when_c_zero_f_unique_root_when_b_zero_c_pos_f_symmetric_about_0_c_f_more_than_two_roots_l2104_210413


namespace NUMINAMATH_CALUDE_area_of_sliced_quadrilateral_l2104_210429

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a rectangular prism -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a quadrilateral formed by slicing a rectangular prism -/
structure SlicedQuadrilateral where
  prism : RectangularPrism
  A : Point3D -- vertex
  B : Point3D -- midpoint on length edge
  C : Point3D -- midpoint on width edge
  D : Point3D -- midpoint on height edge

/-- Calculate the area of the sliced quadrilateral -/
def areaOfSlicedQuadrilateral (quad : SlicedQuadrilateral) : ℝ :=
  sorry -- Placeholder for the actual calculation

/-- Theorem: The area of the sliced quadrilateral is 1.5 square units -/
theorem area_of_sliced_quadrilateral :
  let prism := RectangularPrism.mk 2 3 4
  let A := Point3D.mk 0 0 0
  let B := Point3D.mk 1 0 0
  let C := Point3D.mk 0 1.5 0
  let D := Point3D.mk 0 0 2
  let quad := SlicedQuadrilateral.mk prism A B C D
  areaOfSlicedQuadrilateral quad = 1.5 := by
  sorry


end NUMINAMATH_CALUDE_area_of_sliced_quadrilateral_l2104_210429


namespace NUMINAMATH_CALUDE_function_inequality_range_l2104_210457

theorem function_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x ≥ -1 → x^2 - 2*a*x + 2 ≥ a) ↔ -3 ≤ a ∧ a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_range_l2104_210457


namespace NUMINAMATH_CALUDE_second_term_of_sequence_l2104_210491

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) : Prop :=
  ∃ n : ℕ, n > 1 ∧ aₙ = a₁ + (n - 1) * d

theorem second_term_of_sequence (a₁ a₂ aₙ d : ℕ) :
  a₁ = 34 → d = 11 → aₙ = 89 → arithmetic_sequence a₁ d aₙ → a₂ = 45 := by
  sorry

end NUMINAMATH_CALUDE_second_term_of_sequence_l2104_210491


namespace NUMINAMATH_CALUDE_one_absent_two_present_probability_l2104_210428

def absent_probability : ℚ := 1 / 20

def present_probability : ℚ := 1 - absent_probability

def probability_one_absent_two_present (p q : ℚ) : ℚ := 3 * p * q * q

theorem one_absent_two_present_probability : 
  probability_one_absent_two_present absent_probability present_probability = 1083 / 8000 := by
  sorry

end NUMINAMATH_CALUDE_one_absent_two_present_probability_l2104_210428


namespace NUMINAMATH_CALUDE_inscribed_square_area_l2104_210474

/-- The parabola function -/
def parabola (x : ℝ) : ℝ := x^2 - 8*x + 12

/-- The square inscribed in the region bound by the parabola and the x-axis -/
structure InscribedSquare where
  center : ℝ  -- x-coordinate of the square's center
  sideLength : ℝ  -- side length of the square
  top_on_parabola : parabola (center + sideLength/2) = sideLength
  bottom_on_xaxis : center - sideLength/2 ≥ 0

/-- The theorem stating the area of the inscribed square -/
theorem inscribed_square_area :
  ∀ (s : InscribedSquare), s.sideLength^2 = 24 - 8*Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l2104_210474


namespace NUMINAMATH_CALUDE_tan_138_less_than_tan_143_l2104_210487

theorem tan_138_less_than_tan_143 :
  let angle1 : Real := 138 * π / 180
  let angle2 : Real := 143 * π / 180
  (π / 2 < angle1 ∧ angle1 < π) →
  (π / 2 < angle2 ∧ angle2 < π) →
  (∀ x y, π / 2 < x ∧ x < y ∧ y < π → Real.tan x > Real.tan y) →
  Real.tan angle1 < Real.tan angle2 :=
by
  sorry

end NUMINAMATH_CALUDE_tan_138_less_than_tan_143_l2104_210487


namespace NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l2104_210444

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n, a (n + 1) = a n * r

theorem fifth_term_of_geometric_sequence (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (∀ n, a n > 0) →
  a 3 = 16 →
  a 7 = 2 →
  a 5 = 8 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l2104_210444


namespace NUMINAMATH_CALUDE_novels_per_month_l2104_210499

/-- Given that each novel has 200 pages and 9600 pages of novels are read in a year,
    prove that 4 novels are read in a month. -/
theorem novels_per_month :
  ∀ (pages_per_novel : ℕ) (pages_per_year : ℕ) (months_per_year : ℕ),
    pages_per_novel = 200 →
    pages_per_year = 9600 →
    months_per_year = 12 →
    (pages_per_year / pages_per_novel) / months_per_year = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_novels_per_month_l2104_210499


namespace NUMINAMATH_CALUDE_max_display_sum_l2104_210403

def DigitalWatch := ℕ × ℕ

def valid_time (t : DigitalWatch) : Prop :=
  1 ≤ t.1 ∧ t.1 ≤ 12 ∧ t.2 < 60

def digit_sum (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def display_sum (t : DigitalWatch) : ℕ :=
  digit_sum t.1 + digit_sum t.2

theorem max_display_sum :
  ∃ (t : DigitalWatch), valid_time t ∧
    ∀ (t' : DigitalWatch), valid_time t' →
      display_sum t' ≤ display_sum t ∧
      display_sum t = 23 := by
  sorry

end NUMINAMATH_CALUDE_max_display_sum_l2104_210403


namespace NUMINAMATH_CALUDE_maximize_x_3_minus_3x_l2104_210404

theorem maximize_x_3_minus_3x :
  ∀ x : ℝ, 0 < x → x < 1 → x * (3 - 3 * x) ≤ 3 / 4 ∧
  (x * (3 - 3 * x) = 3 / 4 ↔ x = 1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_maximize_x_3_minus_3x_l2104_210404


namespace NUMINAMATH_CALUDE_specific_case_general_case_l2104_210440

-- Define the theorem for the specific case n = 4
theorem specific_case :
  Real.sqrt (4 + 4/15) = 8 * Real.sqrt 15 / 15 := by sorry

-- Define the theorem for the general case
theorem general_case (n : ℕ) (h : n ≥ 2) :
  Real.sqrt (n + n/(n^2 - 1)) = n * Real.sqrt (n/(n^2 - 1)) := by sorry

end NUMINAMATH_CALUDE_specific_case_general_case_l2104_210440


namespace NUMINAMATH_CALUDE_yellow_balls_count_l2104_210494

theorem yellow_balls_count (total : ℕ) (white green red purple : ℕ) (prob : ℚ) :
  total = 60 ∧
  white = 22 ∧
  green = 18 ∧
  red = 3 ∧
  purple = 1 ∧
  prob = 95 / 100 ∧
  (white + green + (total - white - green - red - purple)) / total = prob →
  total - white - green - red - purple = 17 := by
    sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l2104_210494


namespace NUMINAMATH_CALUDE_power_division_equals_729_l2104_210483

theorem power_division_equals_729 : (3 ^ 12) / (27 ^ 2) = 729 := by sorry

end NUMINAMATH_CALUDE_power_division_equals_729_l2104_210483


namespace NUMINAMATH_CALUDE_restricted_arrangements_eq_78_l2104_210473

/-- The number of ways to arrange n elements. -/
def arrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange 5 contestants with restrictions. -/
def restrictedArrangements : ℕ :=
  arrangements 4 + 3 * 3 * arrangements 3

/-- Theorem stating that the number of restricted arrangements is 78. -/
theorem restricted_arrangements_eq_78 :
  restrictedArrangements = 78 := by sorry

end NUMINAMATH_CALUDE_restricted_arrangements_eq_78_l2104_210473


namespace NUMINAMATH_CALUDE_trig_simplification_l2104_210445

/-- Proves that 1/cos(80°) - √3/sin(80°) = 4 --/
theorem trig_simplification : 
  1 / Real.cos (80 * π / 180) - Real.sqrt 3 / Real.sin (80 * π / 180) = 4 := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_l2104_210445


namespace NUMINAMATH_CALUDE_continuous_g_l2104_210410

noncomputable def g (x : ℝ) (b : ℝ) : ℝ :=
  if x ≤ 2 then 2 * x^2 + 5 * x + 3 else b * x + 7

theorem continuous_g (b : ℝ) : 
  Continuous g ↔ b = 7 := by
  sorry

end NUMINAMATH_CALUDE_continuous_g_l2104_210410


namespace NUMINAMATH_CALUDE_equation_solution_l2104_210406

theorem equation_solution : ∃ x : ℚ, (5 * x + 9 * x = 420 - 10 * (x - 4)) ∧ x = 115 / 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2104_210406


namespace NUMINAMATH_CALUDE_students_shorter_than_yoongi_l2104_210456

theorem students_shorter_than_yoongi (total_students : ℕ) (taller_than_yoongi : ℕ) :
  total_students = 20 →
  taller_than_yoongi = 11 →
  total_students - (taller_than_yoongi + 1) = 8 := by
sorry

end NUMINAMATH_CALUDE_students_shorter_than_yoongi_l2104_210456


namespace NUMINAMATH_CALUDE_walk_time_calculation_l2104_210492

/-- Represents the walking times between different locations in minutes -/
structure WalkingTimes where
  parkOfficeToHiddenLake : ℝ
  hiddenLakeToParkOffice : ℝ
  parkOfficeToLakeParkRestaurant : ℝ

/-- Represents the wind effect on walking times -/
structure WindEffect where
  favorableReduction : ℝ
  adverseIncrease : ℝ

theorem walk_time_calculation (w : WindEffect) (t : WalkingTimes) : 
  w.favorableReduction = 0.2 →
  w.adverseIncrease = 0.25 →
  t.parkOfficeToHiddenLake * (1 - w.favorableReduction) = 15 →
  t.hiddenLakeToParkOffice * (1 + w.adverseIncrease) = 7 →
  t.parkOfficeToHiddenLake * (1 - w.favorableReduction) + 
    t.hiddenLakeToParkOffice * (1 + w.adverseIncrease) + 
    t.parkOfficeToLakeParkRestaurant * (1 - w.favorableReduction) = 32 →
  t.parkOfficeToLakeParkRestaurant = 12.5 := by
  sorry

#check walk_time_calculation

end NUMINAMATH_CALUDE_walk_time_calculation_l2104_210492


namespace NUMINAMATH_CALUDE_right_triangle_trig_identity_l2104_210408

theorem right_triangle_trig_identity 
  (A B C : Real) 
  (right_angle : C = Real.pi / 2)
  (condition1 : Real.cos A ^ 2 + Real.cos B ^ 2 + 2 * Real.sin A * Real.sin B * Real.cos C = 3/2)
  (condition2 : Real.cos B ^ 2 + 2 * Real.sin B * Real.cos A = 5/3) :
  Real.cos A ^ 2 + 2 * Real.sin A * Real.cos B = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_trig_identity_l2104_210408


namespace NUMINAMATH_CALUDE_expression_simplification_l2104_210424

theorem expression_simplification 
  (a b c k x : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_k_nonzero : k ≠ 0) :
  k * ((x + a)^2 / ((a - b)*(a - c)) + 
       (x + b)^2 / ((b - a)*(b - c)) + 
       (x + c)^2 / ((c - a)*(c - b))) = k :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2104_210424


namespace NUMINAMATH_CALUDE_maximum_value_of_x_plus_reciprocal_x_l2104_210465

theorem maximum_value_of_x_plus_reciprocal_x (x : ℝ) :
  x < 0 → ∃ (max : ℝ), (∀ y, y < 0 → y + 1/y ≤ max) ∧ max = -2 :=
sorry


end NUMINAMATH_CALUDE_maximum_value_of_x_plus_reciprocal_x_l2104_210465


namespace NUMINAMATH_CALUDE_ron_eats_24_slices_l2104_210422

/-- The number of pickle slices Sammy can eat -/
def sammy_slices : ℕ := 15

/-- Tammy can eat twice as many pickle slices as Sammy -/
def tammy_slices : ℕ := 2 * sammy_slices

/-- Ron eats 20% fewer pickle slices than Tammy -/
def ron_slices : ℕ := tammy_slices - (tammy_slices * 20 / 100)

/-- Theorem stating that Ron eats 24 pickle slices -/
theorem ron_eats_24_slices : ron_slices = 24 := by sorry

end NUMINAMATH_CALUDE_ron_eats_24_slices_l2104_210422


namespace NUMINAMATH_CALUDE_fourth_test_score_l2104_210469

theorem fourth_test_score (first_three_average : ℝ) (desired_increase : ℝ) : 
  first_three_average = 85 → 
  desired_increase = 2 → 
  (3 * first_three_average + 93) / 4 = first_three_average + desired_increase := by
sorry

end NUMINAMATH_CALUDE_fourth_test_score_l2104_210469


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2104_210463

-- Define sets A and B
def A : Set ℝ := {x | x^2 + x - 12 < 0}
def B : Set ℝ := {x | 2 - x < 0}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x | 2 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2104_210463


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l2104_210431

theorem quadratic_inequality_condition (m : ℝ) :
  (m > 1 → ∃ x : ℝ, x^2 + 2*m*x + 1 < 0) ∧
  (∃ m : ℝ, m ≤ 1 ∧ ∃ x : ℝ, x^2 + 2*m*x + 1 < 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l2104_210431


namespace NUMINAMATH_CALUDE_marble_return_condition_l2104_210452

/-- Represents the motion of a marble on a horizontal table with elastic collision -/
structure MarbleMotion where
  v₀ : ℝ  -- Initial speed
  h : ℝ   -- Initial height
  D : ℝ   -- Distance to vertical wall
  g : ℝ   -- Acceleration due to gravity

/-- The condition for the marble to return to the edge of the table -/
def returns_to_edge (m : MarbleMotion) : Prop :=
  m.v₀ = 2 * m.D * Real.sqrt (m.g / (2 * m.h))

/-- Theorem stating the condition for the marble to return to the edge of the table -/
theorem marble_return_condition (m : MarbleMotion) :
  returns_to_edge m ↔ m.v₀ = 2 * m.D * Real.sqrt (m.g / (2 * m.h)) :=
by sorry

end NUMINAMATH_CALUDE_marble_return_condition_l2104_210452


namespace NUMINAMATH_CALUDE_multiplier_problem_l2104_210462

/-- Given a = 5, b = 30, and 40 ab = 1800, prove that the multiplier m such that m * a = 30 is equal to 6. -/
theorem multiplier_problem (a b : ℝ) (h1 : a = 5) (h2 : b = 30) (h3 : 40 * a * b = 1800) :
  ∃ m : ℝ, m * a = 30 ∧ m = 6 := by
sorry

end NUMINAMATH_CALUDE_multiplier_problem_l2104_210462


namespace NUMINAMATH_CALUDE_rain_probability_theorem_l2104_210448

/-- The probability of rain on each day -/
def rain_prob : ℚ := 2/3

/-- The number of consecutive days -/
def num_days : ℕ := 5

/-- The probability of no rain on a single day -/
def no_rain_prob : ℚ := 1 - rain_prob

/-- The probability of two consecutive dry days -/
def two_dry_days_prob : ℚ := no_rain_prob ^ 2

/-- The number of pairs of consecutive days in the given period -/
def num_pairs : ℕ := num_days - 1

theorem rain_probability_theorem :
  (no_rain_prob ^ num_days = 1/243) ∧
  (two_dry_days_prob * num_pairs = 4/9) := by
  sorry


end NUMINAMATH_CALUDE_rain_probability_theorem_l2104_210448


namespace NUMINAMATH_CALUDE_symmetric_point_of_P_l2104_210409

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define the symmetry operation with respect to y = x line
def symmetricPoint (p : Point) : Point :=
  { x := p.y, y := p.x }

-- Theorem statement
theorem symmetric_point_of_P :
  let P : Point := { x := 1, y := 3 }
  symmetricPoint P = { x := 3, y := 1 } := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_of_P_l2104_210409


namespace NUMINAMATH_CALUDE_triangle_side_length_l2104_210466

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  S : ℝ

-- State the theorem
theorem triangle_side_length (t : Triangle) 
  (h1 : t.a = 4)
  (h2 : t.b = 5)
  (h3 : t.S = 5 * Real.sqrt 3) :
  t.c = Real.sqrt 21 ∨ t.c = Real.sqrt 61 := by
  sorry


end NUMINAMATH_CALUDE_triangle_side_length_l2104_210466


namespace NUMINAMATH_CALUDE_sum_of_rectangle_areas_l2104_210475

/-- The number of available squares -/
def n : ℕ := 9

/-- The side length of each square in cm -/
def side_length : ℝ := 1

/-- The set of possible widths for rectangles -/
def possible_widths : Finset ℕ := Finset.range n

/-- The set of possible heights for rectangles -/
def possible_heights : Finset ℕ := Finset.range n

/-- The area of a rectangle with given width and height -/
def rectangle_area (w h : ℕ) : ℝ := (w : ℝ) * (h : ℝ) * side_length ^ 2

/-- The set of all valid rectangles (width, height) that can be formed -/
def valid_rectangles : Finset (ℕ × ℕ) :=
  Finset.filter (fun p => p.1 * p.2 ≤ n) (possible_widths.product possible_heights)

/-- The sum of areas of all distinct rectangles -/
def sum_of_areas : ℝ := Finset.sum valid_rectangles (fun p => rectangle_area p.1 p.2)

theorem sum_of_rectangle_areas :
  sum_of_areas = 72 := by sorry

end NUMINAMATH_CALUDE_sum_of_rectangle_areas_l2104_210475


namespace NUMINAMATH_CALUDE_cassette_tape_cost_cassette_tape_cost_is_nine_l2104_210471

/-- The cost of a cassette tape given Josie's shopping scenario -/
theorem cassette_tape_cost : ℝ → Prop :=
  fun x =>
    let initial_amount : ℝ := 50
    let headphone_cost : ℝ := 25
    let remaining_amount : ℝ := 7
    let num_tapes : ℝ := 2
    initial_amount - (num_tapes * x + headphone_cost) = remaining_amount →
    x = 9

/-- Proof that the cost of each cassette tape is $9 -/
theorem cassette_tape_cost_is_nine : cassette_tape_cost 9 := by
  sorry

end NUMINAMATH_CALUDE_cassette_tape_cost_cassette_tape_cost_is_nine_l2104_210471


namespace NUMINAMATH_CALUDE_triangle_altitude_angle_relation_l2104_210418

theorem triangle_altitude_angle_relation (A B C : Real) (C₁ C₂ : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = 180 →
  -- A is 60° and greater than B
  A = 60 ∧ A > B →
  -- C₁ and C₂ are parts of angle C divided by the altitude
  C = C₁ + C₂ →
  -- C₁ is adjacent to side b (opposite to angle B)
  C₁ > 0 ∧ C₂ > 0 →
  -- The altitude creates right angles
  B + C₁ = 90 ∧ A + C₂ = 90 →
  -- Conclusion
  C₁ - C₂ = 0 := by
sorry

end NUMINAMATH_CALUDE_triangle_altitude_angle_relation_l2104_210418


namespace NUMINAMATH_CALUDE_orange_pricing_theorem_l2104_210467

/-- Represents a pricing scheme for oranges -/
structure PricingScheme where
  oranges : ℕ
  price : ℕ

/-- Calculates the minimum cost for buying a given number of oranges -/
def minCost (schemes : List PricingScheme) (totalOranges : ℕ) : ℕ :=
  sorry

/-- Calculates the average cost per orange -/
def avgCost (totalCost : ℕ) (totalOranges : ℕ) : ℚ :=
  sorry

theorem orange_pricing_theorem (schemes : List PricingScheme) (totalOranges : ℕ) :
  schemes = [⟨4, 12⟩, ⟨7, 30⟩] →
  totalOranges = 20 →
  avgCost (minCost schemes totalOranges) totalOranges = 3 := by
  sorry

end NUMINAMATH_CALUDE_orange_pricing_theorem_l2104_210467


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2104_210442

theorem complex_fraction_equality (z : ℂ) (h : z + Complex.I = 4 - Complex.I) :
  z / (4 + 2 * Complex.I) = (3 - 4 * Complex.I) / 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2104_210442


namespace NUMINAMATH_CALUDE_arithmetic_mean_fraction_l2104_210411

theorem arithmetic_mean_fraction (x b c : ℝ) (hx : x ≠ 0) (hbc : b ≠ c) :
  (1 / 2) * ((x + b) / x + (x - c) / x) = 1 + (b - c) / (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_fraction_l2104_210411


namespace NUMINAMATH_CALUDE_parallel_line_y_intercept_l2104_210489

/-- A line parallel to y = 3x + 1 passing through (3,6) has y-intercept -3 -/
theorem parallel_line_y_intercept :
  ∀ (b : ℝ → ℝ),
  (∀ x y, b x = y ↔ ∃ k, y = 3 * x + k) →  -- b is parallel to y = 3x + 1
  b 3 = 6 →                               -- b passes through (3,6)
  ∃ c, ∀ x, b x = 3 * x + c ∧ c = -3      -- b has equation y = 3x + c with c = -3
  := by sorry

end NUMINAMATH_CALUDE_parallel_line_y_intercept_l2104_210489


namespace NUMINAMATH_CALUDE_taco_cheese_amount_l2104_210490

/-- The amount of cheese (in ounces) needed for a burrito -/
def cheese_per_burrito : ℝ := 4

/-- The total amount of cheese (in ounces) needed for 7 burritos and 1 taco -/
def total_cheese : ℝ := 37

/-- The amount of cheese (in ounces) needed for a taco -/
def cheese_per_taco : ℝ := total_cheese - 7 * cheese_per_burrito

theorem taco_cheese_amount : cheese_per_taco = 9 := by
  sorry

end NUMINAMATH_CALUDE_taco_cheese_amount_l2104_210490


namespace NUMINAMATH_CALUDE_inequality_theorem_l2104_210461

theorem inequality_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / c) + (c / b) ≥ (4 * a) / (a + b) ∧
  ((a / c) + (c / b) = (4 * a) / (a + b) ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorem_l2104_210461


namespace NUMINAMATH_CALUDE_abc_zero_l2104_210458

theorem abc_zero (a b c : ℝ) 
  (h1 : (a + b) * (b + c) * (c + a) = a * b * c)
  (h2 : (a^3 + b^3) * (b^3 + c^3) * (c^3 + a^3) = a^3 * b^3 * c^3) : 
  a * b * c = 0 := by
sorry

end NUMINAMATH_CALUDE_abc_zero_l2104_210458


namespace NUMINAMATH_CALUDE_playground_girls_l2104_210439

theorem playground_girls (total_children boys : ℕ) 
  (h1 : total_children = 117) 
  (h2 : boys = 40) : 
  total_children - boys = 77 := by
sorry

end NUMINAMATH_CALUDE_playground_girls_l2104_210439


namespace NUMINAMATH_CALUDE_flour_with_weevils_l2104_210481

theorem flour_with_weevils 
  (p_good_milk : ℝ) 
  (p_good_egg : ℝ) 
  (p_all_good : ℝ) 
  (h1 : p_good_milk = 0.8) 
  (h2 : p_good_egg = 0.4) 
  (h3 : p_all_good = 0.24) : 
  ∃ (p_good_flour : ℝ), 
    p_good_milk * p_good_egg * p_good_flour = p_all_good ∧ 
    1 - p_good_flour = 0.25 :=
by sorry

end NUMINAMATH_CALUDE_flour_with_weevils_l2104_210481


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2104_210472

/-- Given a hyperbola and a parabola with specific conditions, prove that the standard equation of the hyperbola is x²/16 - y²/16 = 1 -/
theorem hyperbola_equation (a b p : ℝ) (ha : a > 0) (hb : b > 0) (hp : p > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∀ x y : ℝ, y^2 = 2*p*x) →
  (∃ x : ℝ, |x + a| = 3) →
  (b*(-1) + a*1 = 0) →
  (a = 4 ∧ b = 4) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2104_210472


namespace NUMINAMATH_CALUDE_max_leftover_grapes_l2104_210451

theorem max_leftover_grapes (n : ℕ) : ∃ (k : ℕ), n = 7 * k + (n % 7) ∧ n % 7 ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_max_leftover_grapes_l2104_210451


namespace NUMINAMATH_CALUDE_min_value_theorem_l2104_210468

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 1 / (x + 3) + 1 / (y + 3) = 1 / 4) :
  x^2 + 3*y ≥ 20 + 16 * Real.sqrt 3 ∧
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧
    1 / (x₀ + 3) + 1 / (y₀ + 3) = 1 / 4 ∧
    x₀^2 + 3*y₀ = 20 + 16 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2104_210468


namespace NUMINAMATH_CALUDE_math_team_combinations_l2104_210401

/-- The number of ways to choose k items from n items. -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of girls in the math club. -/
def num_girls : ℕ := 4

/-- The number of boys in the math club. -/
def num_boys : ℕ := 6

/-- The number of girls required in the team. -/
def girls_in_team : ℕ := 3

/-- The number of boys required in the team. -/
def boys_in_team : ℕ := 4

theorem math_team_combinations :
  (choose num_girls girls_in_team) * (choose num_boys boys_in_team) = 60 := by
  sorry

end NUMINAMATH_CALUDE_math_team_combinations_l2104_210401


namespace NUMINAMATH_CALUDE_hyperbolic_amplitude_properties_l2104_210427

/-- Hyperbolic cosine -/
noncomputable def ch (x : ℝ) : ℝ := sorry

/-- Hyperbolic sine -/
noncomputable def sh (x : ℝ) : ℝ := sorry

/-- Hyperbolic tangent -/
noncomputable def th (x : ℝ) : ℝ := sorry

/-- Tangent -/
noncomputable def tg (α : ℝ) : ℝ := sorry

theorem hyperbolic_amplitude_properties (x α : ℝ) 
  (h1 : ch x ^ 2 - sh x ^ 2 = 1)
  (h2 : tg α = sh x / ch x) : 
  (ch x = 1 / Real.cos α) ∧ (th (x / 2) = tg (α / 2)) := by
  sorry

end NUMINAMATH_CALUDE_hyperbolic_amplitude_properties_l2104_210427


namespace NUMINAMATH_CALUDE_pet_store_combinations_l2104_210488

def num_puppies : ℕ := 15
def num_kittens : ℕ := 10
def num_hamsters : ℕ := 8
def num_people : ℕ := 3

theorem pet_store_combinations : 
  num_puppies * num_kittens * num_hamsters * Nat.factorial num_people = 7200 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_combinations_l2104_210488


namespace NUMINAMATH_CALUDE_museum_art_count_l2104_210421

theorem museum_art_count (total : ℕ) (asian : ℕ) (egyptian : ℕ) (european : ℕ) 
  (h1 : total = 2500)
  (h2 : asian = 465)
  (h3 : egyptian = 527)
  (h4 : european = 320) :
  total - (asian + egyptian + european) = 1188 := by
  sorry

end NUMINAMATH_CALUDE_museum_art_count_l2104_210421


namespace NUMINAMATH_CALUDE_inequality_proof_l2104_210470

theorem inequality_proof (a b c : ℝ) 
  (h1 : 0 < a ∧ a < 1) 
  (h2 : 0 < b ∧ b < 1) 
  (h3 : 0 < c ∧ c < 1) 
  (h4 : a * b * c = Real.sqrt 3 / 9) : 
  a / (1 - a^2) + b / (1 - b^2) + c / (1 - c^2) ≥ 3 * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2104_210470


namespace NUMINAMATH_CALUDE_wool_production_l2104_210425

variables (x y z w v : ℝ)
variable (breed_A_production : ℝ → ℝ → ℝ → ℝ)
variable (breed_B_production : ℝ → ℝ → ℝ → ℝ → ℝ → ℝ)

-- Breed A production rate
axiom breed_A_rate : breed_A_production x y z = y / (x * z)

-- Breed B produces twice as much as Breed A
axiom breed_B_rate : breed_B_production x y z w v = 2 * breed_A_production x y z * w * v

-- Theorem to prove
theorem wool_production : breed_B_production x y z w v = (2 * y * w * v) / (x * z) := by
  sorry

end NUMINAMATH_CALUDE_wool_production_l2104_210425


namespace NUMINAMATH_CALUDE_cosine_equality_problem_l2104_210497

theorem cosine_equality_problem :
  ∃ n : ℤ, 0 ≤ n ∧ n ≤ 180 ∧ Real.cos (n * π / 180) = Real.cos (1018 * π / 180) ∧ n = 62 := by
  sorry

end NUMINAMATH_CALUDE_cosine_equality_problem_l2104_210497


namespace NUMINAMATH_CALUDE_class_savings_theorem_l2104_210454

/-- Calculates the total amount saved by a class for a field trip over a given period. -/
def total_savings (num_students : ℕ) (contribution_per_student : ℕ) (weeks_per_month : ℕ) (num_months : ℕ) : ℕ :=
  num_students * contribution_per_student * weeks_per_month * num_months

/-- Theorem stating that a class of 30 students contributing $2 each week will save $480 in 2 months. -/
theorem class_savings_theorem :
  total_savings 30 2 4 2 = 480 := by
  sorry

#eval total_savings 30 2 4 2

end NUMINAMATH_CALUDE_class_savings_theorem_l2104_210454


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2104_210476

/-- Proves that x³ - 2x²y + xy² = x(x-y)² for all real numbers x and y -/
theorem polynomial_factorization (x y : ℝ) : 
  x^3 - 2*x^2*y + x*y^2 = x*(x-y)^2 := by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2104_210476


namespace NUMINAMATH_CALUDE_evie_shells_left_l2104_210464

/-- The number of shells Evie collects per day -/
def shells_per_day : ℕ := 10

/-- The number of days Evie collects shells -/
def collection_days : ℕ := 6

/-- The number of shells Evie gives to her brother -/
def shells_given : ℕ := 2

/-- The number of shells Evie has left after collecting and giving some away -/
def shells_left : ℕ := shells_per_day * collection_days - shells_given

/-- Theorem stating that Evie has 58 shells left -/
theorem evie_shells_left : shells_left = 58 := by sorry

end NUMINAMATH_CALUDE_evie_shells_left_l2104_210464


namespace NUMINAMATH_CALUDE_g_is_correct_l2104_210441

-- Define the polynomial g(x)
def g (x : ℝ) : ℝ := -2*x^5 + 7*x^4 + 4*x^3 - 2*x^2 - 8*x + 4

-- Theorem statement
theorem g_is_correct :
  ∀ x : ℝ, 2*x^5 - 4*x^3 + 3*x + g x = 7*x^4 - 2*x^2 - 5*x + 4 :=
by
  sorry

end NUMINAMATH_CALUDE_g_is_correct_l2104_210441


namespace NUMINAMATH_CALUDE_rectangle_tiling_existence_l2104_210447

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a tiling of rectangles -/
def Tiling := List Rectangle

/-- Checks if a list of rectangles can tile a larger rectangle -/
def canTile (tiles : Tiling) (target : Rectangle) : Prop := sorry

/-- The specific tiles we're allowed to use -/
def allowedTiles : Tiling := [Rectangle.mk 4 6, Rectangle.mk 5 7]

/-- The theorem stating the existence of N and that 840 is a valid value -/
theorem rectangle_tiling_existence :
  ∃ (N : ℕ), ∀ (m n : ℕ), m > N → n > N →
    canTile allowedTiles (Rectangle.mk m n) ∧ canTile allowedTiles (Rectangle.mk 841 841) := by
  sorry

#check rectangle_tiling_existence

end NUMINAMATH_CALUDE_rectangle_tiling_existence_l2104_210447


namespace NUMINAMATH_CALUDE_point_coordinates_proof_l2104_210495

/-- Given two points M and N, and a point P such that MP = 1/2 * MN, 
    prove that P has specific coordinates. -/
theorem point_coordinates_proof (M N P : ℝ × ℝ) : 
  M = (3, 2) → 
  N = (-5, -5) → 
  P - M = (1 / 2 : ℝ) • (N - M) → 
  P = (-1, -3/2) := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_proof_l2104_210495


namespace NUMINAMATH_CALUDE_number_wall_top_value_l2104_210446

/-- Represents a number wall pyramid --/
structure NumberWall :=
  (bottom_left : ℕ)
  (bottom_middle : ℕ)
  (bottom_right : ℕ)

/-- Calculates the value at the top of the number wall pyramid --/
def top_value (wall : NumberWall) : ℕ :=
  let m := wall.bottom_left + wall.bottom_middle
  let n := wall.bottom_middle + wall.bottom_right
  let left_mid := wall.bottom_left + m
  let right_mid := m + n
  let left_top := left_mid + right_mid
  let right_top := right_mid + wall.bottom_right
  2 * (left_top + right_top)

/-- Theorem stating that the top value of the given number wall is 320 --/
theorem number_wall_top_value :
  ∃ (wall : NumberWall), wall.bottom_left = 20 ∧ wall.bottom_middle = 34 ∧ wall.bottom_right = 44 ∧ top_value wall = 320 :=
sorry

end NUMINAMATH_CALUDE_number_wall_top_value_l2104_210446


namespace NUMINAMATH_CALUDE_polygon_construction_possible_l2104_210400

/-- Represents a line segment with a fixed length -/
structure Segment where
  length : ℝ

/-- Represents a polygon constructed from line segments -/
structure Polygon where
  segments : List Segment

/-- Calculates the area of a polygon -/
def calculateArea (p : Polygon) : ℝ := sorry

/-- Checks if all segments in a polygon are used -/
def allSegmentsUsed (p : Polygon) (segments : List Segment) : Prop := sorry

theorem polygon_construction_possible (segments : List Segment) :
  segments.length = 12 ∧ 
  ∀ s ∈ segments, s.length = 2 →
  ∃ p : Polygon, calculateArea p = 16 ∧ allSegmentsUsed p segments :=
sorry

end NUMINAMATH_CALUDE_polygon_construction_possible_l2104_210400


namespace NUMINAMATH_CALUDE_roots_on_circle_l2104_210498

open Complex

theorem roots_on_circle : ∃ (r : ℝ), r = 2 * Real.sqrt 3 / 3 ∧
  ∀ (z : ℂ), (z - 2) ^ 6 = 64 * z ^ 6 →
    ∃ (c : ℂ), abs (z - c) = r :=
by sorry

end NUMINAMATH_CALUDE_roots_on_circle_l2104_210498


namespace NUMINAMATH_CALUDE_integer_power_sum_l2104_210493

theorem integer_power_sum (x : ℝ) (h : ∃ k : ℤ, x + 1/x = k) :
  ∀ n : ℕ, ∃ m : ℤ, x^n + 1/x^n = m :=
sorry

end NUMINAMATH_CALUDE_integer_power_sum_l2104_210493


namespace NUMINAMATH_CALUDE_emily_cell_phone_cost_l2104_210486

/-- Calculates the total cost of a cell phone plan based on given parameters. -/
def calculate_total_cost (base_cost : ℕ) (included_hours : ℕ) (extra_hour_cost : ℕ)
  (base_message_cost : ℕ) (base_message_limit : ℕ) (hours_used : ℕ) (messages_sent : ℕ) : ℕ :=
  let extra_hours := max (hours_used - included_hours) 0
  let extra_hour_charge := extra_hours * extra_hour_cost
  let base_message_charge := min messages_sent base_message_limit * base_message_cost
  let extra_messages := max (messages_sent - base_message_limit) 0
  let extra_message_charge := extra_messages * (2 * base_message_cost)
  base_cost + extra_hour_charge + base_message_charge + extra_message_charge

/-- Emily's cell phone plan cost theorem -/
theorem emily_cell_phone_cost :
  calculate_total_cost 30 50 15 10 150 52 200 = 8500 :=
by sorry

end NUMINAMATH_CALUDE_emily_cell_phone_cost_l2104_210486


namespace NUMINAMATH_CALUDE_min_value_P_l2104_210412

theorem min_value_P (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 + y^2 + 1/x + 1/y = 27/4) : 
  ∀ (P : ℝ), P = 15/x - 3/(4*y) → P ≥ 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_P_l2104_210412


namespace NUMINAMATH_CALUDE_min_sum_abcd_l2104_210436

theorem min_sum_abcd (a b c d : ℕ) (h : a * b + b * c + c * d + d * a = 707) :
  ∃ (m : ℕ), (∀ (a' b' c' d' : ℕ), a' * b' + b' * c' + c' * d' + d' * a' = 707 →
    a' + b' + c' + d' ≥ m) ∧ a + b + c + d = m :=
by sorry

end NUMINAMATH_CALUDE_min_sum_abcd_l2104_210436


namespace NUMINAMATH_CALUDE_infinite_squares_sum_cube_l2104_210417

theorem infinite_squares_sum_cube :
  ∃ f : ℕ → ℕ, ∀ n : ℕ, ∃ m a : ℕ,
    m > f n ∧ m > 1 ∧ 3 * (2 * a + m + 1)^2 = 11 * m^2 + 1 :=
sorry

end NUMINAMATH_CALUDE_infinite_squares_sum_cube_l2104_210417


namespace NUMINAMATH_CALUDE_soda_price_calculation_l2104_210460

/-- Calculates the price of soda cans with applicable discounts -/
theorem soda_price_calculation (regular_price : ℝ) (case_discount : ℝ) (bulk_discount : ℝ) (cans : ℕ) : 
  regular_price = 0.15 →
  case_discount = 0.12 →
  bulk_discount = 0.05 →
  cans = 75 →
  let discounted_price := regular_price * (1 - case_discount)
  let bulk_discounted_price := discounted_price * (1 - bulk_discount)
  let total_price := bulk_discounted_price * cans
  total_price = 9.405 := by
  sorry

#check soda_price_calculation

end NUMINAMATH_CALUDE_soda_price_calculation_l2104_210460


namespace NUMINAMATH_CALUDE_gcf_of_50_and_75_l2104_210443

theorem gcf_of_50_and_75 : Nat.gcd 50 75 = 25 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_50_and_75_l2104_210443


namespace NUMINAMATH_CALUDE_stating_min_nickels_needed_l2104_210415

/-- Represents the cost of the book in cents -/
def book_cost : ℕ := 4750

/-- Represents the value of four $10 bills in cents -/
def ten_dollar_bills : ℕ := 4000

/-- Represents the value of five half-dollars in cents -/
def half_dollars : ℕ := 250

/-- Represents the value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- 
Theorem stating that the minimum number of nickels needed to reach 
or exceed the book cost, given the other money available, is 100.
-/
theorem min_nickels_needed : 
  ∀ n : ℕ, (n * nickel_value + ten_dollar_bills + half_dollars ≥ book_cost) → n ≥ 100 := by
  sorry

end NUMINAMATH_CALUDE_stating_min_nickels_needed_l2104_210415


namespace NUMINAMATH_CALUDE_square_of_binomial_l2104_210402

theorem square_of_binomial (a b : ℝ) : (2*a - 3*b)^2 = 4*a^2 - 12*a*b + 9*b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_l2104_210402


namespace NUMINAMATH_CALUDE_ceiling_of_e_l2104_210433

theorem ceiling_of_e : ⌈Real.exp 1⌉ = 3 := by sorry

end NUMINAMATH_CALUDE_ceiling_of_e_l2104_210433


namespace NUMINAMATH_CALUDE_square_of_divisibility_l2104_210449

theorem square_of_divisibility (m n : ℤ) 
  (h1 : m ≠ 0) 
  (h2 : n ≠ 0) 
  (h3 : m % 2 = n % 2) 
  (h4 : (n^2 - 1) % (m^2 - n^2 + 1) = 0) : 
  ∃ k : ℤ, m^2 - n^2 + 1 = k^2 := by
sorry

end NUMINAMATH_CALUDE_square_of_divisibility_l2104_210449


namespace NUMINAMATH_CALUDE_cost_of_four_birdhouses_l2104_210435

/-- The cost to build a given number of birdhouses -/
def cost_of_birdhouses (num_birdhouses : ℕ) : ℚ :=
  let planks_per_house : ℕ := 7
  let nails_per_house : ℕ := 20
  let cost_per_nail : ℚ := 5 / 100
  let cost_per_plank : ℕ := 3
  let cost_per_house : ℚ := (planks_per_house * cost_per_plank) + (nails_per_house * cost_per_nail)
  num_birdhouses * cost_per_house

/-- Theorem stating the cost of building 4 birdhouses is $88.00 -/
theorem cost_of_four_birdhouses :
  cost_of_birdhouses 4 = 88 :=
by sorry

end NUMINAMATH_CALUDE_cost_of_four_birdhouses_l2104_210435


namespace NUMINAMATH_CALUDE_election_vote_count_l2104_210485

theorem election_vote_count : 
  -- Define the total number of votes
  ∀ V : ℕ,
  -- First round vote percentages
  let a1 := (27 : ℚ) / 100 * V
  let b1 := (24 : ℚ) / 100 * V
  let c1 := (20 : ℚ) / 100 * V
  let d1 := (18 : ℚ) / 100 * V
  let e1 := V - (a1 + b1 + c1 + d1)
  -- Second round vote percentages
  let a2 := (30 : ℚ) / 100 * V
  let b2 := (27 : ℚ) / 100 * V
  let c2 := (22 : ℚ) / 100 * V
  let d2 := V - (a2 + b2 + c2)
  -- Final round
  let additional_votes := (10 : ℚ) / 100 * V  -- 5% each from C and D supporters
  let a_final := a2 + (5 : ℚ) / 100 * V
  let b_final := b2 + d2 + (5 : ℚ) / 100 * V
  -- B wins by 1350 votes
  b_final - a_final = 1350 →
  V = 7500 := by
sorry

end NUMINAMATH_CALUDE_election_vote_count_l2104_210485


namespace NUMINAMATH_CALUDE_andreas_living_room_area_l2104_210437

theorem andreas_living_room_area :
  ∀ (room_area carpet_area : ℝ),
    carpet_area = 6 * 12 →
    room_area * 0.2 = carpet_area →
    room_area = 360 :=
by
  sorry

end NUMINAMATH_CALUDE_andreas_living_room_area_l2104_210437


namespace NUMINAMATH_CALUDE_largest_five_digit_congruent_to_15_mod_24_l2104_210453

theorem largest_five_digit_congruent_to_15_mod_24 :
  ∀ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 ∧ n ≡ 15 [MOD 24] → n ≤ 99999 :=
by sorry

end NUMINAMATH_CALUDE_largest_five_digit_congruent_to_15_mod_24_l2104_210453


namespace NUMINAMATH_CALUDE_vector_collinearity_angle_l2104_210430

theorem vector_collinearity_angle (θ : Real) :
  let a : Fin 2 → Real := ![2 * Real.cos θ, 2 * Real.sin θ]
  let b : Fin 2 → Real := ![3, Real.sqrt 3]
  (∃ (k : Real), a = k • b) →
  θ ∈ Set.Icc 0 (2 * Real.pi) →
  θ = Real.pi / 6 ∨ θ = 7 * Real.pi / 6 := by
sorry

end NUMINAMATH_CALUDE_vector_collinearity_angle_l2104_210430


namespace NUMINAMATH_CALUDE_duck_park_problem_l2104_210426

theorem duck_park_problem (initial_ducks : ℕ) (geese_arrive : ℕ) (geese_leave : ℕ) : 
  initial_ducks = 25 →
  geese_arrive = 4 →
  geese_leave = 10 →
  ((2 * initial_ducks) - 10) - geese_leave - (initial_ducks + geese_arrive) = 1 := by
  sorry

end NUMINAMATH_CALUDE_duck_park_problem_l2104_210426


namespace NUMINAMATH_CALUDE_max_value_expression_l2104_210438

theorem max_value_expression (w x y z t : ℝ) 
  (nonneg_w : w ≥ 0) (nonneg_x : x ≥ 0) (nonneg_y : y ≥ 0) (nonneg_z : z ≥ 0) (nonneg_t : t ≥ 0)
  (sum_eq_120 : w + x + y + z + t = 120) :
  wx + xy + yz + zt ≤ 3600 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l2104_210438


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l2104_210434

theorem smallest_n_satisfying_conditions : 
  ∃ n : ℕ, 
    n > 2021 ∧ 
    Nat.gcd 63 (n + 120) = 21 ∧ 
    Nat.gcd (n + 63) 120 = 60 ∧
    (∀ m : ℕ, m > 2021 → Nat.gcd 63 (m + 120) = 21 → Nat.gcd (m + 63) 120 = 60 → m ≥ n) ∧
    n = 2337 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l2104_210434


namespace NUMINAMATH_CALUDE_ceiling_equality_abs_diff_l2104_210416

-- Define the ceiling function
noncomputable def ceiling (x : ℝ) : ℤ :=
  Int.ceil x

-- State the theorem
theorem ceiling_equality_abs_diff (x y : ℝ) :
  (∀ x y, ceiling x = ceiling y → |x - y| < 1) ∧
  (∃ x y, |x - y| < 1 ∧ ceiling x ≠ ceiling y) :=
by sorry

end NUMINAMATH_CALUDE_ceiling_equality_abs_diff_l2104_210416


namespace NUMINAMATH_CALUDE_arithmetic_equation_l2104_210482

theorem arithmetic_equation : 64 + 5 * 12 / (180 / 3) = 65 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equation_l2104_210482


namespace NUMINAMATH_CALUDE_count_even_factors_l2104_210455

def n : ℕ := 2^4 * 3^3 * 7

/-- The number of even positive factors of n -/
def num_even_factors (n : ℕ) : ℕ := sorry

theorem count_even_factors :
  num_even_factors n = 32 := by sorry

end NUMINAMATH_CALUDE_count_even_factors_l2104_210455


namespace NUMINAMATH_CALUDE_prove_abc_equation_l2104_210420

theorem prove_abc_equation (a b c : ℝ) 
  (h1 : a^4 * b^3 * c^5 = 18) 
  (h2 : a^3 * b^5 * c^4 = 8) : 
  a^5 * b * c^6 = 81/2 := by
  sorry

end NUMINAMATH_CALUDE_prove_abc_equation_l2104_210420


namespace NUMINAMATH_CALUDE_like_terms_exponent_product_l2104_210419

theorem like_terms_exponent_product (a b : ℤ) : 
  (6 = -2 * a) → (b = 2) → a * b = -6 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_exponent_product_l2104_210419


namespace NUMINAMATH_CALUDE_water_jar_problem_l2104_210478

theorem water_jar_problem (small_jar large_jar : ℝ) (h1 : small_jar > 0) (h2 : large_jar > 0) 
  (h3 : small_jar ≠ large_jar) (water : ℝ) (h4 : water > 0)
  (h5 : water / small_jar = 1 / 7) (h6 : water / large_jar = 1 / 6) :
  (2 * water) / large_jar = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_water_jar_problem_l2104_210478


namespace NUMINAMATH_CALUDE_remainder_theorem_l2104_210423

theorem remainder_theorem (x : ℤ) : 
  (2*x + 3)^504 ≡ 16*x + 5 [ZMOD (x^2 - x + 1)] :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2104_210423


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2104_210484

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x : ℤ | x^2 ≤ 2}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2104_210484


namespace NUMINAMATH_CALUDE_f_value_at_2_l2104_210459

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^5 + b * x^3 + c * x - 8

-- State the theorem
theorem f_value_at_2 (a b c : ℝ) : f a b c (-2) = 10 → f a b c 2 = -26 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_2_l2104_210459
