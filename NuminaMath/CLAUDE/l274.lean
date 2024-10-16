import Mathlib

namespace NUMINAMATH_CALUDE_bubble_radius_l274_27444

/-- Given a hemisphere with radius 4∛2 cm that has the same volume as a spherical bubble,
    the radius of the original bubble is 4 cm. -/
theorem bubble_radius (r : ℝ) (R : ℝ) : 
  r = 4 * Real.rpow 2 (1/3) → -- radius of hemisphere
  (2/3) * Real.pi * r^3 = (4/3) * Real.pi * R^3 → -- volume equality
  R = 4 := by
sorry

end NUMINAMATH_CALUDE_bubble_radius_l274_27444


namespace NUMINAMATH_CALUDE_square_circle_union_area_l274_27452

/-- The area of the union of a square and an inscribed circle -/
theorem square_circle_union_area 
  (square_side : ℝ) 
  (circle_radius : ℝ) 
  (h1 : square_side = 20) 
  (h2 : circle_radius = 10) 
  (h3 : circle_radius = square_side / 2) : 
  square_side ^ 2 = 400 := by sorry

end NUMINAMATH_CALUDE_square_circle_union_area_l274_27452


namespace NUMINAMATH_CALUDE_red_toys_removed_l274_27455

theorem red_toys_removed (total : ℕ) (red_after : ℕ) : 
  total = 134 →
  red_after = 88 →
  red_after = 2 * (total - red_after) →
  total - red_after - (red_after - 2) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_red_toys_removed_l274_27455


namespace NUMINAMATH_CALUDE_derivative_of_f_l274_27475

noncomputable def f (x : ℝ) : ℝ := 
  (1 / (2 * Real.sqrt 2)) * (Real.sin (Real.log x) - (Real.sqrt 2 - 1) * Real.cos (Real.log x)) * x^(Real.sqrt 2 + 1)

theorem derivative_of_f (x : ℝ) (h : x > 0) : 
  deriv f x = (x^(Real.sqrt 2) / (2 * Real.sqrt 2)) * 
    (2 * Real.cos (Real.log x) - Real.sqrt 2 * Real.cos (Real.log x) + 2 * Real.sqrt 2 * Real.sin (Real.log x)) := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_f_l274_27475


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l274_27471

/-- Represents a school with stratified sampling -/
structure School where
  total_students : ℕ
  first_grade : ℕ
  second_grade : ℕ
  third_grade : ℕ
  sample_size : ℕ
  first_grade_sample : ℕ
  second_grade_sample : ℕ
  third_grade_sample : ℕ

/-- The stratified sampling theorem -/
theorem stratified_sampling_theorem (s : School) 
  (h1 : s.sample_size = 45)
  (h2 : s.first_grade_sample = 20)
  (h3 : s.third_grade_sample = 10)
  (h4 : s.second_grade = 300)
  (h5 : s.sample_size = s.first_grade_sample + s.second_grade_sample + s.third_grade_sample)
  (h6 : s.total_students = s.first_grade + s.second_grade + s.third_grade)
  (h7 : s.second_grade_sample / s.second_grade = s.sample_size / s.total_students) :
  s.total_students = 900 := by
  sorry


end NUMINAMATH_CALUDE_stratified_sampling_theorem_l274_27471


namespace NUMINAMATH_CALUDE_jackie_working_hours_l274_27421

/-- Represents the number of hours in a day -/
def hours_in_day : ℕ := 24

/-- Represents the number of hours Jackie spends exercising -/
def exercise_hours : ℕ := 3

/-- Represents the number of hours Jackie spends sleeping -/
def sleep_hours : ℕ := 8

/-- Represents the number of hours Jackie has as free time -/
def free_time_hours : ℕ := 5

/-- Calculates the number of hours Jackie spends working -/
def working_hours : ℕ := hours_in_day - (sleep_hours + exercise_hours + free_time_hours)

theorem jackie_working_hours :
  working_hours = 8 := by sorry

end NUMINAMATH_CALUDE_jackie_working_hours_l274_27421


namespace NUMINAMATH_CALUDE_product_of_trig_expressions_l274_27433

theorem product_of_trig_expressions :
  (1 - Real.sin (π / 8)) * (1 - Real.sin (3 * π / 8)) *
  (1 + Real.sin (π / 8)) * (1 + Real.sin (3 * π / 8)) = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_product_of_trig_expressions_l274_27433


namespace NUMINAMATH_CALUDE_candy_probability_l274_27489

theorem candy_probability : 
  let total_candies : ℕ := 12
  let red_candies : ℕ := 5
  let blue_candies : ℕ := 2
  let green_candies : ℕ := 5
  let pick_count : ℕ := 4
  let favorable_outcomes : ℕ := (red_candies.choose 3) * (blue_candies + green_candies)
  let total_outcomes : ℕ := total_candies.choose pick_count
  (favorable_outcomes : ℚ) / total_outcomes = 14 / 99 := by sorry

end NUMINAMATH_CALUDE_candy_probability_l274_27489


namespace NUMINAMATH_CALUDE_least_sum_m_n_l274_27457

theorem least_sum_m_n : ∃ (m n : ℕ+), 
  (Nat.gcd (m.val + n.val) 330 = 1) ∧ 
  (∃ (k : ℕ), m.val^m.val = k * n.val^n.val) ∧ 
  (∀ (j : ℕ), m.val ≠ j * n.val) ∧
  (m.val + n.val = 247) ∧
  (∀ (m' n' : ℕ+), 
    (Nat.gcd (m'.val + n'.val) 330 = 1) → 
    (∃ (k : ℕ), m'.val^m'.val = k * n'.val^n'.val) → 
    (∀ (j : ℕ), m'.val ≠ j * n'.val) → 
    (m'.val + n'.val ≥ 247)) :=
by sorry

end NUMINAMATH_CALUDE_least_sum_m_n_l274_27457


namespace NUMINAMATH_CALUDE_oliver_dish_count_l274_27427

/-- Represents the buffet and Oliver's preferences -/
structure Buffet where
  total_dishes : ℕ
  mango_salsa_dishes : ℕ
  fresh_mango_dishes : ℕ
  mango_jelly_dishes : ℕ
  strawberry_dishes : ℕ
  pineapple_dishes : ℕ
  mango_dishes_oliver_can_eat : ℕ

/-- Calculates the number of dishes Oliver can eat -/
def dishes_for_oliver (b : Buffet) : ℕ :=
  b.total_dishes -
  (b.mango_salsa_dishes + b.fresh_mango_dishes + b.mango_jelly_dishes - b.mango_dishes_oliver_can_eat) -
  min b.strawberry_dishes b.pineapple_dishes

/-- Theorem stating the number of dishes Oliver can eat -/
theorem oliver_dish_count (b : Buffet) : dishes_for_oliver b = 28 :=
  by
    have h1 : b.total_dishes = 42 := by sorry
    have h2 : b.mango_salsa_dishes = 5 := by sorry
    have h3 : b.fresh_mango_dishes = 7 := by sorry
    have h4 : b.mango_jelly_dishes = 2 := by sorry
    have h5 : b.strawberry_dishes = 3 := by sorry
    have h6 : b.pineapple_dishes = 5 := by sorry
    have h7 : b.mango_dishes_oliver_can_eat = 3 := by sorry
    sorry

#eval dishes_for_oliver {
  total_dishes := 42,
  mango_salsa_dishes := 5,
  fresh_mango_dishes := 7,
  mango_jelly_dishes := 2,
  strawberry_dishes := 3,
  pineapple_dishes := 5,
  mango_dishes_oliver_can_eat := 3
}

end NUMINAMATH_CALUDE_oliver_dish_count_l274_27427


namespace NUMINAMATH_CALUDE_moving_circle_trajectory_l274_27420

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := (x + 5)^2 + y^2 = 16
def C₂ (x y : ℝ) : Prop := (x - 5)^2 + y^2 = 16

-- Define the moving circle M
structure MovingCircle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the tangency conditions
def externally_tangent (M : MovingCircle) : Prop :=
  C₁ (M.center.1 + M.radius) M.center.2

def internally_tangent (M : MovingCircle) : Prop :=
  C₂ (M.center.1 - M.radius) M.center.2

-- State the theorem
theorem moving_circle_trajectory
  (M : MovingCircle)
  (h1 : externally_tangent M)
  (h2 : internally_tangent M) :
  ∃ x y : ℝ, x > 0 ∧ x^2 / 16 - y^2 / 9 = 1 ∧ M.center = (x, y) :=
sorry

end NUMINAMATH_CALUDE_moving_circle_trajectory_l274_27420


namespace NUMINAMATH_CALUDE_group_c_marks_is_four_l274_27426

/-- Represents the examination setup with three groups of questions -/
structure Examination where
  total_questions : ℕ
  group_a_marks : ℕ
  group_b_marks : ℕ
  group_b_questions : ℕ
  group_c_questions : ℕ

/-- Theorem stating that under the given conditions, each question in group C carries 4 marks -/
theorem group_c_marks_is_four (exam : Examination)
  (h_total : exam.total_questions = 100)
  (h_group_a : exam.group_a_marks = 1)
  (h_group_b : exam.group_b_marks = 2)
  (h_group_b_count : exam.group_b_questions = 23)
  (h_group_c_count : exam.group_c_questions = 1)
  (h_group_a_percentage : 
    exam.group_a_marks * (exam.total_questions - exam.group_b_questions - exam.group_c_questions) ≥
    (3/5) * (exam.group_a_marks * (exam.total_questions - exam.group_b_questions - exam.group_c_questions) +
             exam.group_b_marks * exam.group_b_questions +
             4 * exam.group_c_questions)) :
  ∃ (group_c_marks : ℕ), group_c_marks = 4 ∧
    group_c_marks > exam.group_b_marks ∧
    exam.group_a_marks * (exam.total_questions - exam.group_b_questions - exam.group_c_questions) ≥
    (3/5) * (exam.group_a_marks * (exam.total_questions - exam.group_b_questions - exam.group_c_questions) +
             exam.group_b_marks * exam.group_b_questions +
             group_c_marks * exam.group_c_questions) := by
  sorry

end NUMINAMATH_CALUDE_group_c_marks_is_four_l274_27426


namespace NUMINAMATH_CALUDE_circle_division_theorem_l274_27472

/-- Represents a region in the circle --/
structure Region where
  value : Nat
  deriving Repr

/-- Represents a line dividing the circle --/
structure DividingLine where
  left_regions : List Region
  right_regions : List Region
  deriving Repr

/-- The configuration of regions in the circle --/
def CircleConfiguration := List Region

/-- Checks if the sums on both sides of a line are equal --/
def is_line_balanced (line : DividingLine) : Bool :=
  (line.left_regions.map Region.value).sum = (line.right_regions.map Region.value).sum

/-- Checks if all lines in the configuration are balanced --/
def is_configuration_valid (config : CircleConfiguration) (lines : List DividingLine) : Bool :=
  lines.all is_line_balanced

/-- Theorem: There exists a valid configuration for distributing numbers 1 to 7 in a circle divided by 3 lines --/
theorem circle_division_theorem :
  ∃ (config : CircleConfiguration) (lines : List DividingLine),
    config.length = 7 ∧
    (∀ n, n ∈ config.map Region.value → n ∈ List.range 7) ∧
    lines.length = 3 ∧
    is_configuration_valid config lines :=
sorry

end NUMINAMATH_CALUDE_circle_division_theorem_l274_27472


namespace NUMINAMATH_CALUDE_distance_relation_l274_27432

/-- Given four points on a directed line satisfying a certain condition, 
    prove a relationship between their distances. -/
theorem distance_relation (A B C D : ℝ) 
    (h : (C - A) / (B - C) + (D - A) / (B - D) = 0) : 
    1 / (C - A) + 1 / (D - A) = 2 / (B - A) := by
  sorry

end NUMINAMATH_CALUDE_distance_relation_l274_27432


namespace NUMINAMATH_CALUDE_running_time_calculation_l274_27412

/-- Proves that given the conditions, the time taken to cover the same distance while running is [(a + 2b) × (c + d)] / (3a - b) hours. -/
theorem running_time_calculation 
  (a b c d : ℕ+) -- a, b, c, and d are positive integers
  (walking_speed : ℝ := a + 2*b) -- Walking speed = (a + 2b) kmph
  (walking_time : ℝ := c + d) -- Walking time = (c + d) hours
  (running_speed : ℝ := 3*a - b) -- Running speed = (3a - b) kmph
  (k : ℝ := 3) -- Conversion factor k = 3
  (h : k * walking_speed = running_speed) -- Assumption that k * walking_speed = running_speed
  : 
  (walking_speed * walking_time) / running_speed = (a + 2*b) * (c + d) / (3*a - b) := 
by
  sorry


end NUMINAMATH_CALUDE_running_time_calculation_l274_27412


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l274_27428

open Set

def U : Set ℝ := univ
def A : Set ℝ := {1, 2, 3, 4}
def B : Set ℝ := {x | x ≤ 2}

theorem intersection_complement_equality :
  A ∩ (U \ B) = {3, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l274_27428


namespace NUMINAMATH_CALUDE_service_center_location_l274_27499

/-- Represents a highway with exits and a service center -/
structure Highway where
  third_exit : ℝ
  tenth_exit : ℝ
  service_center : ℝ

/-- Theorem: Given a highway with the third exit at milepost 50 and the tenth exit at milepost 170,
    a service center located two-thirds of the way from the third exit to the tenth exit
    is at milepost 130. -/
theorem service_center_location (h : Highway)
  (h_third : h.third_exit = 50)
  (h_tenth : h.tenth_exit = 170)
  (h_service : h.service_center = h.third_exit + 2 / 3 * (h.tenth_exit - h.third_exit)) :
  h.service_center = 130 := by
  sorry

end NUMINAMATH_CALUDE_service_center_location_l274_27499


namespace NUMINAMATH_CALUDE_max_value_product_l274_27437

theorem max_value_product (x y z : ℝ) 
  (nonneg_x : 0 ≤ x) (nonneg_y : 0 ≤ y) (nonneg_z : 0 ≤ z) 
  (sum_eq_three : x + y + z = 3) : 
  (x^3 - x*y^2 + y^3) * (x^3 - x*z^2 + z^3) * (y^3 - y*z^2 + z^3) ≤ 2916/2187 := by
  sorry

end NUMINAMATH_CALUDE_max_value_product_l274_27437


namespace NUMINAMATH_CALUDE_expression_evaluation_l274_27454

theorem expression_evaluation :
  let a : ℤ := -3
  let b : ℤ := -2
  (3 * a^2 * b + 2 * a * b^2) - (2 * (a^2 * b - 1) + 3 * a * b^2 + 2) = -6 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l274_27454


namespace NUMINAMATH_CALUDE_equal_probability_red_black_l274_27436

/-- Represents a deck of cards after removing face cards and 8's --/
structure Deck :=
  (total_cards : ℕ)
  (red_divisible_by_3 : ℕ)
  (black_divisible_by_3 : ℕ)

/-- Represents the probability of picking a card of a certain color divisible by 3 --/
def probability_divisible_by_3 (deck : Deck) (color : String) : ℚ :=
  if color = "red" then
    (deck.red_divisible_by_3 : ℚ) / deck.total_cards
  else if color = "black" then
    (deck.black_divisible_by_3 : ℚ) / deck.total_cards
  else
    0

/-- The main theorem stating that the probabilities are equal for red and black cards --/
theorem equal_probability_red_black (deck : Deck) 
    (h1 : deck.total_cards = 36)
    (h2 : deck.red_divisible_by_3 = 6)
    (h3 : deck.black_divisible_by_3 = 6) :
  probability_divisible_by_3 deck "red" = probability_divisible_by_3 deck "black" :=
by
  sorry

#check equal_probability_red_black

end NUMINAMATH_CALUDE_equal_probability_red_black_l274_27436


namespace NUMINAMATH_CALUDE_hash_equals_100_l274_27483

def hash (a b : ℕ) : ℕ := 4 * a^2 + 4 * b^2 + 8 * a * b

theorem hash_equals_100 (a b : ℕ) (h : a + b + 6 = 11) : hash a b = 100 := by
  sorry

end NUMINAMATH_CALUDE_hash_equals_100_l274_27483


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l274_27481

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  focus1 : Point2D
  focus2 : Point2D
  majorAxis : ℝ

/-- Represents a line in 2D space -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Returns true if the line intersects the ellipse at exactly one point -/
def intersectsAtOnePoint (e : Ellipse) (l : Line2D) : Prop :=
  sorry

theorem ellipse_major_axis_length :
  ∀ (e : Ellipse) (l : Line2D),
    e.focus1 = Point2D.mk (-2) 0 →
    e.focus2 = Point2D.mk 2 0 →
    l.a = 1 →
    l.b = Real.sqrt 3 →
    l.c = 4 →
    intersectsAtOnePoint e l →
    e.majorAxis = 2 * Real.sqrt 7 :=
  sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l274_27481


namespace NUMINAMATH_CALUDE_lindas_family_women_without_daughters_l274_27480

/-- Represents the family structure of Linda and her descendants -/
structure Family where
  total_daughters_and_granddaughters : Nat
  lindas_daughters : Nat
  daughters_with_five_children : Nat

/-- The number of women (daughters and granddaughters) who have no daughters in Linda's family -/
def women_without_daughters (f : Family) : Nat :=
  f.total_daughters_and_granddaughters - f.daughters_with_five_children

/-- Theorem stating the number of women without daughters in Linda's specific family situation -/
theorem lindas_family_women_without_daughters :
  ∀ f : Family,
  f.total_daughters_and_granddaughters = 43 →
  f.lindas_daughters = 8 →
  f.daughters_with_five_children * 5 = f.total_daughters_and_granddaughters - f.lindas_daughters →
  women_without_daughters f = 36 := by
  sorry


end NUMINAMATH_CALUDE_lindas_family_women_without_daughters_l274_27480


namespace NUMINAMATH_CALUDE_triangle_max_perimeter_l274_27470

theorem triangle_max_perimeter :
  ∀ (x : ℕ),
  x > 0 →
  x < 18 →
  x + 4*x > 18 →
  x + 18 > 4*x →
  4*x + 18 > x →
  (∀ y : ℕ, y > x → y + 4*y ≤ 18 ∨ y + 18 ≤ 4*y ∨ 4*y + 18 ≤ y) →
  x + 4*x + 18 = 38 :=
by
  sorry

#check triangle_max_perimeter

end NUMINAMATH_CALUDE_triangle_max_perimeter_l274_27470


namespace NUMINAMATH_CALUDE_percent_calculation_l274_27406

theorem percent_calculation (x : ℝ) (h : 0.2 * x = 60) : 0.8 * x = 240 := by
  sorry

end NUMINAMATH_CALUDE_percent_calculation_l274_27406


namespace NUMINAMATH_CALUDE_highest_a_divisible_by_8_l274_27405

def is_divisible_by_8 (n : ℕ) : Prop := n % 8 = 0

def last_three_digits (n : ℕ) : ℕ := n % 1000

def construct_number (a : ℕ) : ℕ := 365000 + a * 100 + 16

theorem highest_a_divisible_by_8 :
  ∀ a : ℕ, a ≤ 9 →
    (is_divisible_by_8 (construct_number a) ↔ a ≤ 8) ∧
    (a = 8 → is_divisible_by_8 (construct_number a)) ∧
    (a = 9 → ¬is_divisible_by_8 (construct_number a)) :=
sorry

end NUMINAMATH_CALUDE_highest_a_divisible_by_8_l274_27405


namespace NUMINAMATH_CALUDE_transformation_result_l274_27430

-- Define the initial point
def initial_point : ℝ × ℝ × ℝ := (2, 2, 2)

-- Define the transformations
def rotate_z_90 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-y, x, z)

def reflect_xz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, -y, z)

def rotate_x_90 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, z, -y)

def reflect_yz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-x, y, z)

-- Define the sequence of transformations
def transform (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  rotate_z_90 (reflect_yz (rotate_x_90 (reflect_xz (rotate_z_90 p))))

-- Theorem statement
theorem transformation_result :
  transform initial_point = (2, -2, -2) := by sorry

end NUMINAMATH_CALUDE_transformation_result_l274_27430


namespace NUMINAMATH_CALUDE_gnollish_valid_sentences_l274_27460

/-- Represents the number of words in the Gnollish language -/
def num_words : ℕ := 4

/-- Represents the length of sentences in the Gnollish language -/
def sentence_length : ℕ := 4

/-- Represents the number of invalid two-word sequences -/
def num_invalid_sequences : ℕ := 1

/-- Calculates the number of valid sentences in the Gnollish language -/
def num_valid_sentences : ℕ := num_words ^ sentence_length - num_invalid_sequences * num_words ^ (sentence_length - 2)

/-- Theorem stating that the number of valid 4-word sentences in the Gnollish language is 208 -/
theorem gnollish_valid_sentences : num_valid_sentences = 208 := by
  sorry

end NUMINAMATH_CALUDE_gnollish_valid_sentences_l274_27460


namespace NUMINAMATH_CALUDE_not_monotone_condition_l274_27448

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 12*x

-- Define the property of being not monotone on an interval
def not_monotone_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x y, a < x ∧ x < y ∧ y < b ∧ (f x < f y ∧ ∃ z, x < z ∧ z < y ∧ f z < f x) ∨
                                 (f x > f y ∧ ∃ z, x < z ∧ z < y ∧ f z > f x)

-- State the theorem
theorem not_monotone_condition (k : ℝ) :
  not_monotone_on f k (k + 2) ↔ (-4 < k ∧ k < -2) ∨ (0 < k ∧ k < 2) :=
sorry

end NUMINAMATH_CALUDE_not_monotone_condition_l274_27448


namespace NUMINAMATH_CALUDE_no_solution_exists_l274_27478

theorem no_solution_exists : ¬∃ (a b c : ℕ+), 
  (a * b + b * c = a * c) ∧ (a * b * c = Nat.factorial 10) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l274_27478


namespace NUMINAMATH_CALUDE_angle_r_measure_l274_27429

/-- An isosceles triangle with specific angle relationships -/
structure IsoscelesTriangle where
  /-- The measure of angle P in degrees -/
  angle_p : ℝ
  /-- The measure of angle R is 40 degrees more than angle P -/
  angle_r : ℝ := angle_p + 40
  /-- The sum of all angles in the triangle is 180 degrees -/
  angle_sum : angle_p + angle_p + angle_r = 180

/-- The measure of angle R in an isosceles triangle with the given conditions -/
theorem angle_r_measure (t : IsoscelesTriangle) : t.angle_r = 86.67 := by
  sorry

end NUMINAMATH_CALUDE_angle_r_measure_l274_27429


namespace NUMINAMATH_CALUDE_min_value_abs_sum_l274_27425

theorem min_value_abs_sum (x : ℝ) : 
  |x - 4| + |x + 7| + |x - 5| ≥ 1 ∧ ∃ y : ℝ, |y - 4| + |y + 7| + |y - 5| = 1 :=
sorry

end NUMINAMATH_CALUDE_min_value_abs_sum_l274_27425


namespace NUMINAMATH_CALUDE_average_time_per_flower_l274_27493

/-- Proves that the average time to find a flower is 10 minutes -/
theorem average_time_per_flower 
  (total_time : ℕ) 
  (total_flowers : ℕ) 
  (h1 : total_time = 330) 
  (h2 : total_flowers = 33) 
  (h3 : total_time % total_flowers = 0) :
  total_time / total_flowers = 10 := by
  sorry

#check average_time_per_flower

end NUMINAMATH_CALUDE_average_time_per_flower_l274_27493


namespace NUMINAMATH_CALUDE_complementary_angles_difference_l274_27440

theorem complementary_angles_difference (x : ℝ) (h1 : x > 0) (h2 : 3*x + x = 90) : |3*x - x| = 45 := by
  sorry

end NUMINAMATH_CALUDE_complementary_angles_difference_l274_27440


namespace NUMINAMATH_CALUDE_max_product_sum_2000_l274_27443

theorem max_product_sum_2000 :
  ∃ (x : ℤ), ∀ (y : ℤ), x * (2000 - x) ≥ y * (2000 - y) ∧ x * (2000 - x) = 1000000 :=
by sorry

end NUMINAMATH_CALUDE_max_product_sum_2000_l274_27443


namespace NUMINAMATH_CALUDE_square_sum_equals_twenty_l274_27411

theorem square_sum_equals_twenty (x y : ℝ) 
  (h1 : (x + y)^2 = 36) 
  (h2 : x * y = 8) : 
  x^2 + y^2 = 20 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_twenty_l274_27411


namespace NUMINAMATH_CALUDE_special_line_equation_l274_27445

/-- A line passing through a point with x-axis and y-axis intercepts that are opposite numbers -/
structure SpecialLine where
  -- The point through which the line passes
  point : ℝ × ℝ
  -- The equation of the line, represented as a function ℝ² → ℝ
  equation : ℝ → ℝ → ℝ
  -- Condition: The line passes through the given point
  passes_through_point : equation point.1 point.2 = 0
  -- Condition: The line has intercepts on x-axis and y-axis that are opposite numbers
  opposite_intercepts : ∃ (a : ℝ), (equation a 0 = 0 ∧ equation 0 (-a) = 0) ∨ 
                                   (equation (-a) 0 = 0 ∧ equation 0 a = 0)

/-- Theorem: The equation of the special line is either x - y - 7 = 0 or 2x + 5y = 0 -/
theorem special_line_equation (l : SpecialLine) (h : l.point = (5, -2)) :
  (l.equation = fun x y => x - y - 7) ∨ (l.equation = fun x y => 2*x + 5*y) := by
  sorry

end NUMINAMATH_CALUDE_special_line_equation_l274_27445


namespace NUMINAMATH_CALUDE_det_A_equals_one_l274_27416

open Matrix

theorem det_A_equals_one (a d : ℝ) :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![a, 1; -2, d]
  (A + A⁻¹ = 0) → det A = 1 := by
  sorry

end NUMINAMATH_CALUDE_det_A_equals_one_l274_27416


namespace NUMINAMATH_CALUDE_cosine_transformation_symmetry_l274_27477

open Real

theorem cosine_transformation_symmetry (ω : ℝ) :
  ω > 0 →
  (∀ x, ∃ y, cos (ω * (x - π / 12)) = y) →
  (∀ x, cos (ω * ((π / 4 + (π / 4 - x)) - π / 12)) = cos (ω * (x - π / 12))) →
  ω ≥ 6 :=
by sorry

end NUMINAMATH_CALUDE_cosine_transformation_symmetry_l274_27477


namespace NUMINAMATH_CALUDE_reciprocal_equality_l274_27486

theorem reciprocal_equality (a b : ℝ) 
  (ha : a⁻¹ = -8) 
  (hb : (-b)⁻¹ = 8) : 
  a = b := by sorry

end NUMINAMATH_CALUDE_reciprocal_equality_l274_27486


namespace NUMINAMATH_CALUDE_armband_break_even_l274_27401

/-- The cost of an individual ticket in dollars -/
def individual_ticket_cost : ℚ := 3/4

/-- The cost of an armband in dollars -/
def armband_cost : ℚ := 15

/-- The number of rides at which the armband cost equals the cost of individual tickets -/
def break_even_rides : ℕ := 20

theorem armband_break_even :
  (individual_ticket_cost * break_even_rides : ℚ) = armband_cost :=
by sorry

end NUMINAMATH_CALUDE_armband_break_even_l274_27401


namespace NUMINAMATH_CALUDE_money_game_total_determinable_l274_27461

/-- Represents the money redistribution game among four friends -/
structure MoneyGame where
  -- Initial amounts
  amy_initial : ℝ
  jan_initial : ℝ
  toy_initial : ℝ
  ben_initial : ℝ
  -- Final amounts
  amy_final : ℝ
  jan_final : ℝ
  toy_final : ℝ
  ben_final : ℝ

/-- The rules of the money redistribution game -/
def redistribute (game : MoneyGame) : Prop :=
  -- Amy's turn
  let amy_after := game.amy_initial - game.jan_initial - game.toy_initial - game.ben_initial
  let jan_after1 := 2 * game.jan_initial
  let toy_after1 := 2 * game.toy_initial
  let ben_after1 := 2 * game.ben_initial
  -- Jan's turn
  let amy_after2 := 2 * amy_after
  let toy_after2 := 2 * toy_after1
  let ben_after2 := 2 * ben_after1
  let jan_after2 := jan_after1 - (amy_after + toy_after1 + ben_after1)
  -- Toy's turn
  let amy_after3 := 2 * amy_after2
  let jan_after3 := 2 * jan_after2
  let ben_after3 := 2 * ben_after2
  let toy_after3 := toy_after2 - (amy_after2 + jan_after2 + ben_after2)
  -- Ben's turn
  game.amy_final = 2 * amy_after3 ∧
  game.jan_final = 2 * jan_after3 ∧
  game.toy_final = 2 * toy_after3 ∧
  game.ben_final = ben_after3 - (amy_after3 + jan_after3 + toy_after3)

/-- The theorem statement -/
theorem money_game_total_determinable (game : MoneyGame) :
  game.toy_initial = 24 ∧ 
  game.toy_final = 96 ∧ 
  redistribute game → 
  ∃ total : ℝ, total = game.amy_final + game.jan_final + game.toy_final + game.ben_final :=
by sorry


end NUMINAMATH_CALUDE_money_game_total_determinable_l274_27461


namespace NUMINAMATH_CALUDE_inverse_equals_scaled_sum_l274_27446

/-- Given a 2x2 matrix M, prove that its inverse is equal to (1/6)*M + (1/6)*I -/
theorem inverse_equals_scaled_sum (M : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : M = !![2, 0; 1, -3]) : 
  M⁻¹ = (1/6 : ℝ) • M + (1/6 : ℝ) • (1 : Matrix (Fin 2) (Fin 2) ℝ) := by
  sorry

end NUMINAMATH_CALUDE_inverse_equals_scaled_sum_l274_27446


namespace NUMINAMATH_CALUDE_cubic_root_function_l274_27479

/-- Given a function y = kx^(1/3) where y = 3√2 when x = 64, prove that y = 3 when x = 8 -/
theorem cubic_root_function (k : ℝ) :
  (∃ y : ℝ, y = k * 64^(1/3) ∧ y = 3 * Real.sqrt 2) →
  k * 8^(1/3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_function_l274_27479


namespace NUMINAMATH_CALUDE_largest_cylinder_radius_largest_cylinder_radius_is_4_l274_27439

/-- Represents the dimensions of a rectangular crate -/
structure CrateDimensions where
  length : Real
  width : Real
  height : Real

/-- Represents a right circular cylinder -/
structure Cylinder where
  radius : Real
  height : Real

/-- Checks if a cylinder fits in a crate when the crate is on its length-width side -/
def cylinderFitsInCrate (crate : CrateDimensions) (cylinder : Cylinder) : Prop :=
  cylinder.height ≤ crate.height ∧ 2 * cylinder.radius ≤ min crate.length crate.width

/-- The largest cylinder that fits in the crate has a radius equal to half the smaller of the crate's length or width -/
theorem largest_cylinder_radius (crate : CrateDimensions) :
  ∃ (cylinder : Cylinder),
    cylinderFitsInCrate crate cylinder ∧
    ∀ (other : Cylinder), cylinderFitsInCrate crate other → other.radius ≤ cylinder.radius :=
  sorry

/-- The specific crate dimensions given in the problem -/
def problemCrate : CrateDimensions :=
  { length := 12, width := 8, height := 6 }

/-- The theorem stating that the largest cylinder that fits in the problem crate has a radius of 4 feet -/
theorem largest_cylinder_radius_is_4 :
  ∃ (cylinder : Cylinder),
    cylinderFitsInCrate problemCrate cylinder ∧
    cylinder.radius = 4 ∧
    ∀ (other : Cylinder), cylinderFitsInCrate problemCrate other → other.radius ≤ cylinder.radius :=
  sorry

end NUMINAMATH_CALUDE_largest_cylinder_radius_largest_cylinder_radius_is_4_l274_27439


namespace NUMINAMATH_CALUDE_subset_relationship_l274_27469

def M : Set (ℝ × ℝ) := {p | |p.1| + |p.2| < 1}

def N : Set (ℝ × ℝ) := {p | Real.sqrt ((p.1 - 1/2)^2 + (p.2 + 1/2)^2) + Real.sqrt ((p.1 + 1/2)^2 + (p.2 - 1/2)^2) < 2 * Real.sqrt 2}

def P : Set (ℝ × ℝ) := {p | |p.1 + p.2| < 1 ∧ |p.1| < 1 ∧ |p.2| < 1}

theorem subset_relationship : M ⊆ N ∧ N ⊆ P := by sorry

end NUMINAMATH_CALUDE_subset_relationship_l274_27469


namespace NUMINAMATH_CALUDE_base9_to_base3_conversion_l274_27404

/-- Converts a digit from base 9 to base 4 -/
def base9To4Digit (d : Nat) : Nat :=
  d / 4 * 10 + d % 4

/-- Converts a number from base 9 to base 4 -/
def base9To4 (n : Nat) : Nat :=
  let d1 := n / 81
  let d2 := (n / 9) % 9
  let d3 := n % 9
  base9To4Digit d1 * 10000 + base9To4Digit d2 * 100 + base9To4Digit d3

/-- Converts a digit from base 4 to base 3 -/
def base4To3Digit (d : Nat) : Nat :=
  d / 3 * 10 + d % 3

/-- Converts a number from base 4 to base 3 -/
def base4To3 (n : Nat) : Nat :=
  let d1 := n / 10000
  let d2 := (n / 100) % 100
  let d3 := n % 100
  base4To3Digit d1 * 100000000 + base4To3Digit d2 * 10000 + base4To3Digit d3

theorem base9_to_base3_conversion :
  base4To3 (base9To4 758) = 01101002000 := by
  sorry

end NUMINAMATH_CALUDE_base9_to_base3_conversion_l274_27404


namespace NUMINAMATH_CALUDE_prob_green_face_specific_die_l274_27410

/-- A die with colored faces -/
structure ColoredDie where
  sides : ℕ
  red_faces : ℕ
  green_faces : ℕ
  blue_faces : ℕ
  total_faces_eq : sides = red_faces + green_faces + blue_faces

/-- The probability of rolling a green face on a colored die -/
def prob_green_face (d : ColoredDie) : ℚ :=
  d.green_faces / d.sides

/-- Theorem: The probability of rolling a green face on a 10-sided die
    with 5 red faces, 3 green faces, and 2 blue faces is 3/10 -/
theorem prob_green_face_specific_die :
  let d : ColoredDie := {
    sides := 10,
    red_faces := 5,
    green_faces := 3,
    blue_faces := 2,
    total_faces_eq := by rfl
  }
  prob_green_face d = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_prob_green_face_specific_die_l274_27410


namespace NUMINAMATH_CALUDE_decimal_multiplication_l274_27417

theorem decimal_multiplication :
  (10 * 0.1 = 1) ∧ (10 * 0.01 = 0.1) ∧ (10 * 0.001 = 0.01) := by
  sorry

end NUMINAMATH_CALUDE_decimal_multiplication_l274_27417


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_ABCD_l274_27463

/-- Given points A, B, C, and D in a 2D coordinate system, prove that ABCD is an isosceles trapezoid with AB parallel to CD -/
theorem isosceles_trapezoid_ABCD :
  let A : ℝ × ℝ := (-6, -1)
  let B : ℝ × ℝ := (2, 3)
  let C : ℝ × ℝ := (-1, 4)
  let D : ℝ × ℝ := (-5, 2)
  
  -- AB is parallel to CD
  (B.2 - A.2) / (B.1 - A.1) = (D.2 - C.2) / (D.1 - C.1) ∧
  
  -- AD = BC (isosceles condition)
  (D.1 - A.1)^2 + (D.2 - A.2)^2 = (C.1 - B.1)^2 + (C.2 - B.2)^2 ∧
  
  -- AB ≠ CD (trapezoid condition)
  (B.1 - A.1)^2 + (B.2 - A.2)^2 ≠ (D.1 - C.1)^2 + (D.2 - C.2)^2 :=
by
  sorry


end NUMINAMATH_CALUDE_isosceles_trapezoid_ABCD_l274_27463


namespace NUMINAMATH_CALUDE_store_savings_l274_27422

/-- The difference between the selling price and the store's cost for a pair of pants. -/
def price_difference (selling_price store_cost : ℕ) : ℕ :=
  selling_price - store_cost

/-- Theorem stating that the price difference is 8 dollars given the specific selling price and store cost. -/
theorem store_savings : price_difference 34 26 = 8 := by
  sorry

end NUMINAMATH_CALUDE_store_savings_l274_27422


namespace NUMINAMATH_CALUDE_min_value_of_expression_l274_27408

theorem min_value_of_expression (x : ℝ) :
  (x^2 + 9) / Real.sqrt (x^2 + 5) ≥ 4 ∧
  ∀ ε > 0, ∃ y : ℝ, (y^2 + 9) / Real.sqrt (y^2 + 5) < 4 + ε :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l274_27408


namespace NUMINAMATH_CALUDE_rhombus_area_l274_27458

/-- The area of a rhombus with side length 25 and one diagonal of 30 is 600 -/
theorem rhombus_area (side : ℝ) (diagonal1 : ℝ) (diagonal2 : ℝ) :
  side = 25 →
  diagonal1 = 30 →
  diagonal2 = 2 * Real.sqrt (side^2 - (diagonal1 / 2)^2) →
  (diagonal1 * diagonal2) / 2 = 600 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l274_27458


namespace NUMINAMATH_CALUDE_nonCubeSequence_250th_term_l274_27494

/-- Function that determines if a positive integer is a perfect cube --/
def isPerfectCube (n : ℕ+) : Prop :=
  ∃ m : ℕ+, n = m^3

/-- The sequence of positive integers omitting perfect cubes --/
def nonCubeSequence : ℕ+ → ℕ+ :=
  sorry

/-- The 250th term of the sequence is 256 --/
theorem nonCubeSequence_250th_term :
  nonCubeSequence 250 = 256 := by
  sorry

end NUMINAMATH_CALUDE_nonCubeSequence_250th_term_l274_27494


namespace NUMINAMATH_CALUDE_simplify_expression_l274_27495

theorem simplify_expression (x y : ℝ) : 4 * x^2 + 3 * y^2 - 2 * x^2 - 4 * y^2 = 2 * x^2 - y^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l274_27495


namespace NUMINAMATH_CALUDE_greatest_solution_is_negative_two_l274_27447

def equation (x : ℝ) : Prop :=
  x ≠ 9 ∧ (x^2 - 5*x - 84) / (x - 9) = 4 / (x + 6)

theorem greatest_solution_is_negative_two :
  ∃ x_max : ℝ, x_max = -2 ∧ equation x_max ∧ ∀ y : ℝ, equation y → y ≤ x_max :=
sorry

end NUMINAMATH_CALUDE_greatest_solution_is_negative_two_l274_27447


namespace NUMINAMATH_CALUDE_multiple_of_one_third_equals_two_ninths_l274_27496

theorem multiple_of_one_third_equals_two_ninths :
  ∃ x : ℚ, x * (1/3 : ℚ) = 2/9 ∧ x = 2/3 := by sorry

end NUMINAMATH_CALUDE_multiple_of_one_third_equals_two_ninths_l274_27496


namespace NUMINAMATH_CALUDE_points_two_units_from_negative_three_l274_27414

theorem points_two_units_from_negative_three :
  ∀ x : ℝ, |(-3) - x| = 2 ↔ x = -5 ∨ x = -1 := by
sorry

end NUMINAMATH_CALUDE_points_two_units_from_negative_three_l274_27414


namespace NUMINAMATH_CALUDE_permutation_sum_l274_27435

theorem permutation_sum (n : ℕ) : 
  n + 3 ≤ 2 * n ∧ n + 1 ≤ 4 → 
  (Nat.factorial (2 * n)) / (Nat.factorial (2 * n - (n + 3))) + 
  (Nat.factorial 4) / (Nat.factorial (4 - (n + 1))) = 744 := by
  sorry

end NUMINAMATH_CALUDE_permutation_sum_l274_27435


namespace NUMINAMATH_CALUDE_square_value_l274_27419

theorem square_value (r : ℝ) (h1 : r + r = 75) (h2 : (r + r) + 2 * r = 143) : r = 41 := by
  sorry

end NUMINAMATH_CALUDE_square_value_l274_27419


namespace NUMINAMATH_CALUDE_complex_arithmetic_result_l274_27441

theorem complex_arithmetic_result : 
  let z₁ : ℂ := 2 - 3*I
  let z₂ : ℂ := -1 + 5*I
  let z₃ : ℂ := 1 + I
  (z₁ + z₂) * z₃ = -1 + 3*I := by sorry

end NUMINAMATH_CALUDE_complex_arithmetic_result_l274_27441


namespace NUMINAMATH_CALUDE_intersection_of_polar_curves_l274_27465

/-- The intersection point of two polar curves -/
theorem intersection_of_polar_curves (ρ θ : ℝ) :
  ρ ≥ 0 →
  0 ≤ θ →
  θ < π / 2 →
  ρ * Real.cos θ = 3 →
  ρ = 4 * Real.cos θ →
  (ρ = 2 * Real.sqrt 3 ∧ θ = π / 6) :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_polar_curves_l274_27465


namespace NUMINAMATH_CALUDE_codemaster_total_codes_l274_27423

/-- The number of different colors of pegs in Codemaster -/
def num_colors : ℕ := 8

/-- The number of slots in a Codemaster code -/
def num_slots : ℕ := 5

/-- The total number of possible secret codes in Codemaster -/
def total_codes : ℕ := num_colors ^ num_slots

/-- Theorem stating that the total number of possible secret codes in Codemaster is 32768 -/
theorem codemaster_total_codes : total_codes = 32768 := by
  sorry

end NUMINAMATH_CALUDE_codemaster_total_codes_l274_27423


namespace NUMINAMATH_CALUDE_random_variable_distribution_invariance_l274_27466

-- Define a type for random variables
variable (Ω : Type) [MeasurableSpace Ω]
def RandomVariable (α : Type) [MeasurableSpace α] := Ω → α

-- Define a type for distribution functions
def DistributionFunction (α : Type) [MeasurableSpace α] := α → ℝ

-- State the theorem
theorem random_variable_distribution_invariance
  (ξ : RandomVariable Ω ℝ)
  (h_non_degenerate : ∀ (c : ℝ), ¬(∀ (ω : Ω), ξ ω = c))
  (a : ℝ)
  (b : ℝ)
  (h_a_pos : a > 0)
  (h_distribution_equal : ∀ (F : DistributionFunction ℝ),
    (∀ (x : ℝ), F x = F ((x - b) / a))) :
  a = 1 ∧ b = 0 :=
sorry

end NUMINAMATH_CALUDE_random_variable_distribution_invariance_l274_27466


namespace NUMINAMATH_CALUDE_tangent_point_x_coordinate_l274_27459

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a / Real.exp x

theorem tangent_point_x_coordinate 
  (a : ℝ) 
  (h_even : ∀ x, f a x = f a (-x)) 
  (h_slope : ∃ x, (deriv (f a)) x = 3/2) :
  ∃ x, (deriv (f a)) x = 3/2 ∧ x = Real.log 2 :=
sorry

end NUMINAMATH_CALUDE_tangent_point_x_coordinate_l274_27459


namespace NUMINAMATH_CALUDE_officer_3_years_shoe_price_l274_27451

def full_price : ℝ := 85
def discount_1_year : ℝ := 0.2
def discount_3_years : ℝ := 0.25

def price_after_1_year_discount : ℝ := full_price * (1 - discount_1_year)
def price_after_3_years_discount : ℝ := price_after_1_year_discount * (1 - discount_3_years)

theorem officer_3_years_shoe_price :
  price_after_3_years_discount = 51 :=
sorry

end NUMINAMATH_CALUDE_officer_3_years_shoe_price_l274_27451


namespace NUMINAMATH_CALUDE_predicted_holiday_shoppers_l274_27474

theorem predicted_holiday_shoppers 
  (packages_per_box : ℕ) 
  (boxes_ordered : ℕ) 
  (shopper_ratio : ℕ) 
  (h1 : packages_per_box = 25)
  (h2 : boxes_ordered = 5)
  (h3 : shopper_ratio = 3) :
  boxes_ordered * packages_per_box * shopper_ratio = 375 :=
by sorry

end NUMINAMATH_CALUDE_predicted_holiday_shoppers_l274_27474


namespace NUMINAMATH_CALUDE_divide_multiply_result_l274_27467

theorem divide_multiply_result : (3 / 4) * 12 = 9 := by
  sorry

end NUMINAMATH_CALUDE_divide_multiply_result_l274_27467


namespace NUMINAMATH_CALUDE_boys_distribution_l274_27464

theorem boys_distribution (total_amount : ℕ) (additional_amount : ℕ) : 
  total_amount = 5040 →
  additional_amount = 80 →
  ∃ (x : ℕ), 
    x * (total_amount / 18 + additional_amount) = total_amount ∧
    x = 14 := by
  sorry

end NUMINAMATH_CALUDE_boys_distribution_l274_27464


namespace NUMINAMATH_CALUDE_seth_candy_bars_l274_27485

theorem seth_candy_bars (max_candy_bars : ℕ) (seth_candy_bars : ℕ) : 
  max_candy_bars = 24 →
  seth_candy_bars = 3 * max_candy_bars + 6 →
  seth_candy_bars = 78 :=
by sorry

end NUMINAMATH_CALUDE_seth_candy_bars_l274_27485


namespace NUMINAMATH_CALUDE_circle_properties_l274_27484

/-- Circle C in the Cartesian coordinate system -/
def circle_C (x y b : ℝ) : Prop := x^2 + y^2 - 6*x - 4*y + b = 0

/-- Point A -/
def point_A : ℝ × ℝ := (0, 3)

/-- Radius of circle C -/
def radius : ℝ := 1

theorem circle_properties :
  ∃ (b : ℝ), 
    (∀ x y, circle_C x y b → (x - 3)^2 + (y - 2)^2 = 1) ∧ 
    (b < 13) ∧
    (∃ (k : ℝ), k = -3/4 ∧ ∀ x y, 3*x + 4*y - 12 = 0 → circle_C x y b) ∧
    (∀ y, circle_C 0 3 b → y = 3) :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l274_27484


namespace NUMINAMATH_CALUDE_equation_pattern_find_a_b_l274_27498

theorem equation_pattern (n : ℕ) (hn : n ≥ 2) :
  Real.sqrt (n + n / (n^2 - 1)) = n * Real.sqrt (n / (n^2 - 1)) := by sorry

theorem find_a_b (a b : ℝ) (h : Real.sqrt (6 + a / b) = 6 * Real.sqrt (a / b)) :
  a = 6 ∧ b = 35 := by sorry

end NUMINAMATH_CALUDE_equation_pattern_find_a_b_l274_27498


namespace NUMINAMATH_CALUDE_parents_per_child_l274_27491

-- Define the number of girls and boys
def num_girls : ℕ := 6
def num_boys : ℕ := 8

-- Define the total number of parents attending
def total_parents : ℕ := 28

-- Theorem statement
theorem parents_per_child (parents_per_child : ℕ) :
  parents_per_child * num_girls + parents_per_child * num_boys = total_parents →
  parents_per_child = 2 := by
sorry

end NUMINAMATH_CALUDE_parents_per_child_l274_27491


namespace NUMINAMATH_CALUDE_obtuse_triangle_area_bound_l274_27449

theorem obtuse_triangle_area_bound (a b c : ℝ) (h_obtuse : 0 < a ∧ 0 < b ∧ 0 < c ∧ c^2 > a^2 + b^2) 
  (h_longest : c = 4) (h_shortest : a = 2) : 
  (1/2 * a * b * Real.sqrt (1 - ((a^2 + b^2 - c^2) / (2*a*b))^2)) ≤ 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_obtuse_triangle_area_bound_l274_27449


namespace NUMINAMATH_CALUDE_cube_face_diagonal_edge_angle_l274_27482

/-- Represents a cube in 3D space -/
structure Cube where
  -- Define necessary properties of a cube

/-- Represents a line segment in 3D space -/
structure LineSegment where
  -- Define necessary properties of a line segment

/-- Represents an angle between two line segments -/
def angle (l1 l2 : LineSegment) : ℝ := sorry

/-- Predicate to check if a line segment is an edge of the cube -/
def is_edge (c : Cube) (l : LineSegment) : Prop := sorry

/-- Predicate to check if a line segment is a face diagonal of the cube -/
def is_face_diagonal (c : Cube) (l : LineSegment) : Prop := sorry

/-- Predicate to check if two line segments are incident to the same vertex -/
def incident_to_same_vertex (l1 l2 : LineSegment) : Prop := sorry

/-- Theorem: In a cube, the angle between a face diagonal and an edge 
    incident to the same vertex is 60 degrees -/
theorem cube_face_diagonal_edge_angle (c : Cube) (d e : LineSegment) :
  is_face_diagonal c d → is_edge c e → incident_to_same_vertex d e →
  angle d e = 60 := by sorry

end NUMINAMATH_CALUDE_cube_face_diagonal_edge_angle_l274_27482


namespace NUMINAMATH_CALUDE_log_division_simplification_l274_27497

theorem log_division_simplification : 
  Real.log 16 / Real.log (1 / 16) = -1 := by sorry

end NUMINAMATH_CALUDE_log_division_simplification_l274_27497


namespace NUMINAMATH_CALUDE_angle_W_measure_l274_27490

-- Define the quadrilateral WXYZ
structure Quadrilateral :=
  (W X Y Z : ℝ)

-- Define the properties of the quadrilateral
def is_valid_quadrilateral (q : Quadrilateral) : Prop :=
  q.W = 3 * q.X ∧ q.W = 2 * q.Y ∧ q.W = 4 * q.Z ∧
  q.W + q.X + q.Y + q.Z = 360

-- Theorem statement
theorem angle_W_measure (q : Quadrilateral) 
  (h : is_valid_quadrilateral q) : q.W = 172.8 := by
  sorry

end NUMINAMATH_CALUDE_angle_W_measure_l274_27490


namespace NUMINAMATH_CALUDE_cost_of_pens_l274_27450

/-- Given that 150 pens cost $45, prove that 3300 pens cost $990 -/
theorem cost_of_pens (pack_size : ℕ) (pack_cost : ℚ) (desired_amount : ℕ) : 
  pack_size = 150 → pack_cost = 45 → desired_amount = 3300 →
  (desired_amount : ℚ) * (pack_cost / pack_size) = 990 :=
by sorry

end NUMINAMATH_CALUDE_cost_of_pens_l274_27450


namespace NUMINAMATH_CALUDE_total_travel_time_is_144_hours_l274_27488

/-- Represents the travel times between different locations -/
structure TravelTimes where
  ngaparaToZipra : ℝ
  ningiToZipra : ℝ
  ziproToVarnasi : ℝ

/-- Calculates the total travel time given the travel times between locations -/
def totalTravelTime (t : TravelTimes) : ℝ :=
  t.ngaparaToZipra + t.ningiToZipra + t.ziproToVarnasi

/-- Theorem stating the total travel time given the conditions in the problem -/
theorem total_travel_time_is_144_hours :
  ∀ t : TravelTimes,
  t.ngaparaToZipra = 60 →
  t.ningiToZipra = 0.8 * t.ngaparaToZipra →
  t.ziproToVarnasi = 0.75 * t.ningiToZipra →
  totalTravelTime t = 144 := by
  sorry

end NUMINAMATH_CALUDE_total_travel_time_is_144_hours_l274_27488


namespace NUMINAMATH_CALUDE_woodworker_job_days_l274_27413

/-- Represents the woodworker's job details -/
structure WoodworkerJob where
  normal_days : ℕ            -- Normal number of days to complete the job
  normal_parts : ℕ           -- Normal number of parts produced
  productivity_increase : ℕ  -- Increase in parts produced per day
  extra_parts : ℕ            -- Extra parts produced with increased productivity

/-- Calculates the number of days required to finish the job with increased productivity -/
def days_with_increased_productivity (job : WoodworkerJob) : ℕ :=
  let normal_rate := job.normal_parts / job.normal_days
  let new_rate := normal_rate + job.productivity_increase
  let total_parts := job.normal_parts + job.extra_parts
  total_parts / new_rate

/-- Theorem stating that for the given conditions, the job takes 22 days with increased productivity -/
theorem woodworker_job_days (job : WoodworkerJob)
  (h1 : job.normal_days = 24)
  (h2 : job.normal_parts = 360)
  (h3 : job.productivity_increase = 5)
  (h4 : job.extra_parts = 80) :
  days_with_increased_productivity job = 22 := by
  sorry

end NUMINAMATH_CALUDE_woodworker_job_days_l274_27413


namespace NUMINAMATH_CALUDE_car_speed_problem_l274_27438

theorem car_speed_problem (v : ℝ) : v > 0 →
  (1 / v * 3600 = 1 / 120 * 3600 + 2) ↔ v = 112.5 :=
by sorry

end NUMINAMATH_CALUDE_car_speed_problem_l274_27438


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l274_27476

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def switch_outermost_digits (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  ones * 100 + tens * 10 + hundreds

theorem unique_three_digit_number : 
  ∃! n : ℕ, is_three_digit n ∧ n + 1 = 2 * (switch_outermost_digits n) :=
by
  use 793
  sorry

#eval switch_outermost_digits 793
#eval 793 + 1 = 2 * (switch_outermost_digits 793)

end NUMINAMATH_CALUDE_unique_three_digit_number_l274_27476


namespace NUMINAMATH_CALUDE_power_of_two_contains_k_zeros_l274_27468

theorem power_of_two_contains_k_zeros (k : ℕ) (hk : k ≥ 1) :
  ∃ n : ℕ+, ∃ a b : ℕ, a ≠ 0 ∧ b ≠ 0 ∧
  (∃ m : ℕ, (2 : ℕ) ^ (n : ℕ) = m * 10^(k+1) + a * 10^k + b) :=
sorry

end NUMINAMATH_CALUDE_power_of_two_contains_k_zeros_l274_27468


namespace NUMINAMATH_CALUDE_reading_time_comparison_l274_27434

/-- Given two people A and B, where A reads 5 times faster than B,
    prove that if B takes 3 hours to read a book,
    then A will take 36 minutes to read the same book. -/
theorem reading_time_comparison (reading_speed_ratio : ℝ) (person_b_time : ℝ) :
  reading_speed_ratio = 5 →
  person_b_time = 3 →
  (person_b_time * 60) / reading_speed_ratio = 36 :=
by sorry

end NUMINAMATH_CALUDE_reading_time_comparison_l274_27434


namespace NUMINAMATH_CALUDE_project_time_ratio_l274_27424

theorem project_time_ratio (total_hours : ℕ) (kate_hours : ℕ) : 
  total_hours = 153 →
  2 * kate_hours + kate_hours + (kate_hours + 85) = total_hours →
  (2 * kate_hours) / (kate_hours + 85) = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_project_time_ratio_l274_27424


namespace NUMINAMATH_CALUDE_decimal_13_equals_binary_1101_l274_27492

def decimal_to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

def binary_to_decimal (l : List Bool) : ℕ :=
  l.enum.foldl (λ sum (i, b) => sum + if b then 2^i else 0) 0

def decimal_13 : ℕ := 13

def binary_1101 : List Bool := [true, false, true, true]

theorem decimal_13_equals_binary_1101 : 
  binary_to_decimal binary_1101 = decimal_13 :=
sorry

end NUMINAMATH_CALUDE_decimal_13_equals_binary_1101_l274_27492


namespace NUMINAMATH_CALUDE_first_day_distance_l274_27402

theorem first_day_distance (total_distance : ℝ) (days : ℕ) (ratio : ℝ) 
  (h1 : total_distance = 378)
  (h2 : days = 6)
  (h3 : ratio = 1/2) :
  let first_day := total_distance * (1 - ratio) / (1 - ratio^days)
  first_day = 192 := by sorry

end NUMINAMATH_CALUDE_first_day_distance_l274_27402


namespace NUMINAMATH_CALUDE_symmetry_implies_sum_power_l274_27415

/-- Two points are symmetric with respect to the y-axis if their y-coordinates are equal
    and their x-coordinates are opposites. -/
def symmetric_y_axis (A B : ℝ × ℝ) : Prop :=
  A.2 = B.2 ∧ A.1 = -B.1

theorem symmetry_implies_sum_power (m n : ℝ) :
  symmetric_y_axis (m, 3) (4, n) → (m + n)^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_sum_power_l274_27415


namespace NUMINAMATH_CALUDE_min_value_theorem_l274_27442

theorem min_value_theorem (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  (x^3 / (y - 2) + y^3 / (x - 2)) ≥ 54 ∧
  ∃ (x₀ y₀ : ℝ), x₀ > 2 ∧ y₀ > 2 ∧ x₀^3 / (y₀ - 2) + y₀^3 / (x₀ - 2) = 54 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l274_27442


namespace NUMINAMATH_CALUDE_triangle_movement_l274_27403

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Check if a triangle is isosceles and right-angled at C -/
def isIsoscelesRightTriangle (t : Triangle) : Prop :=
  (t.A.x - t.C.x)^2 + (t.A.y - t.C.y)^2 = (t.B.x - t.C.x)^2 + (t.B.y - t.C.y)^2 ∧
  (t.A.x - t.C.x) * (t.B.x - t.C.x) + (t.A.y - t.C.y) * (t.B.y - t.C.y) = 0

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The main theorem -/
theorem triangle_movement (t : Triangle) (a b : Line) (c : ℝ) :
  isIsoscelesRightTriangle t →
  (∀ (t' : Triangle), isIsoscelesRightTriangle t' →
    pointOnLine t'.A a → pointOnLine t'.B b →
    (t'.A.x - t'.B.x)^2 + (t'.A.y - t'.B.y)^2 = c^2) →
  ∃ (l : Line),
    (l.a = 1 ∧ l.b = 1 ∧ l.c = 0) ∨ (l.a = 1 ∧ l.b = -1 ∧ l.c = 0) ∧
    ∀ (p : Point), pointOnLine p l →
      -c/2 ≤ p.x ∧ p.x ≤ c/2 →
      ∃ (t' : Triangle), isIsoscelesRightTriangle t' ∧
        pointOnLine t'.A a ∧ pointOnLine t'.B b ∧
        t'.C = p :=
sorry

end NUMINAMATH_CALUDE_triangle_movement_l274_27403


namespace NUMINAMATH_CALUDE_shaded_region_probability_is_half_l274_27456

/-- Represents a game board as an equilateral triangle with six equal regions -/
structure GameBoard where
  regions : Nat
  shaded_regions : Nat

/-- Probability of an event occurring -/
def probability (favorable_outcomes : Nat) (total_outcomes : Nat) : ℚ :=
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ)

/-- The probability of landing on a shaded region in the game board -/
def shaded_region_probability (board : GameBoard) : ℚ :=
  probability board.shaded_regions board.regions

/-- Theorem stating that the probability of landing on a shaded region
    in the described game board is 1/2 -/
theorem shaded_region_probability_is_half :
  ∀ (board : GameBoard),
    board.regions = 6 →
    board.shaded_regions = 3 →
    shaded_region_probability board = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_shaded_region_probability_is_half_l274_27456


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l274_27473

/-- Represents a hyperbola with semi-major axis a, semi-minor axis b, and focal length c -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : a > 0 ∧ b > 0
  h_equation : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1
  h_focal_length : c = 2 * c
  h_foci_to_asymptotes : b = c / 2

/-- The eccentricity of a hyperbola is 2√3/3 given the conditions -/
theorem hyperbola_eccentricity (C : Hyperbola) : 
  Real.sqrt ((C.c^2) / (C.a^2)) = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l274_27473


namespace NUMINAMATH_CALUDE_account_balance_increase_l274_27487

theorem account_balance_increase (initial_deposit : ℝ) (first_year_balance : ℝ) (total_increase_percent : ℝ) :
  initial_deposit = 1000 →
  first_year_balance = 1100 →
  total_increase_percent = 32 →
  let final_balance := initial_deposit * (1 + total_increase_percent / 100)
  let second_year_increase := final_balance - first_year_balance
  let second_year_increase_percent := (second_year_increase / first_year_balance) * 100
  second_year_increase_percent = 20 := by
sorry

end NUMINAMATH_CALUDE_account_balance_increase_l274_27487


namespace NUMINAMATH_CALUDE_min_value_abs_plus_one_l274_27407

theorem min_value_abs_plus_one :
  (∀ x : ℝ, |x - 2| + 1 ≥ 1) ∧ (∃ x : ℝ, |x - 2| + 1 = 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_abs_plus_one_l274_27407


namespace NUMINAMATH_CALUDE_system_solution_existence_l274_27409

theorem system_solution_existence (b : ℝ) :
  (∃ (a x y : ℝ), y = -b - x^2 ∧ x^2 + y^2 + 8*a^2 = 4 + 4*a*(x + y)) ↔ b ≤ 2*Real.sqrt 2 + 1/4 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_existence_l274_27409


namespace NUMINAMATH_CALUDE_min_people_for_condition_l274_27400

/-- Represents a circular table with chairs and people seated. -/
structure CircularTable where
  total_chairs : ℕ
  seated_people : ℕ

/-- Checks if the seating arrangement satisfies the condition that any additional
    person must sit next to at least one other person. -/
def satisfies_condition (table : CircularTable) : Prop :=
  table.seated_people * 4 ≥ table.total_chairs

/-- The theorem stating the minimum number of people required for the given condition. -/
theorem min_people_for_condition (table : CircularTable) 
  (h1 : table.total_chairs = 80)
  (h2 : satisfies_condition table)
  (h3 : ∀ n < table.seated_people, ¬satisfies_condition ⟨table.total_chairs, n⟩) :
  table.seated_people = 20 := by
  sorry

#check min_people_for_condition

end NUMINAMATH_CALUDE_min_people_for_condition_l274_27400


namespace NUMINAMATH_CALUDE_probability_at_least_one_five_or_six_l274_27418

def total_outcomes : ℕ := 36

def favorable_outcomes : ℕ := 18

theorem probability_at_least_one_five_or_six :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_five_or_six_l274_27418


namespace NUMINAMATH_CALUDE_f_properties_l274_27431

/-- The function f(x) defined as (2^x - a) / (2^x + a) where a > 0 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2^x - a) / (2^x + a)

/-- Theorem stating properties of the function f -/
theorem f_properties (a : ℝ) (h : a > 0) 
  (h_odd : ∀ x, f a (-x) = -(f a x)) :
  (a = 1) ∧ 
  (∀ x y, x < y → f a x < f a y) ∧
  (∀ x, x ≤ 1 → f a x ≤ 1/3) ∧
  (f a 1 = 1/3) := by
sorry

end NUMINAMATH_CALUDE_f_properties_l274_27431


namespace NUMINAMATH_CALUDE_martha_cards_l274_27462

theorem martha_cards (x : ℝ) : x - 3.0 = 73 → x = 76 := by
  sorry

end NUMINAMATH_CALUDE_martha_cards_l274_27462


namespace NUMINAMATH_CALUDE_favorite_movies_sum_l274_27453

/-- Given the movie lengths of Joyce, Michael, Nikki, and Ryn, prove their sum is 76 hours -/
theorem favorite_movies_sum (michael nikki joyce ryn : ℝ) : 
  nikki = 30 ∧ 
  michael = nikki / 3 ∧ 
  joyce = michael + 2 ∧ 
  ryn = 4 / 5 * nikki → 
  joyce + michael + nikki + ryn = 76 := by
  sorry

end NUMINAMATH_CALUDE_favorite_movies_sum_l274_27453
