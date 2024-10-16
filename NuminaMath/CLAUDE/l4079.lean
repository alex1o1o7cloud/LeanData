import Mathlib

namespace NUMINAMATH_CALUDE_james_tennis_balls_l4079_407901

/-- Given that James buys 100 tennis balls, gives half away, and distributes the remaining balls 
    equally among 5 containers, prove that each container will have 10 tennis balls. -/
theorem james_tennis_balls (total_balls : ℕ) (containers : ℕ) : 
  total_balls = 100 → 
  containers = 5 → 
  (total_balls / 2) / containers = 10 := by
  sorry

end NUMINAMATH_CALUDE_james_tennis_balls_l4079_407901


namespace NUMINAMATH_CALUDE_fibonacci_like_sequence_l4079_407966

theorem fibonacci_like_sequence (α β : ℝ) (h1 : α^2 - α - 1 = 0) (h2 : β^2 - β - 1 = 0) (h3 : α > β) :
  let s : ℕ → ℝ := λ n => α^n + β^n
  ∀ n ≥ 3, s n = s (n-1) + s (n-2) :=
by sorry

end NUMINAMATH_CALUDE_fibonacci_like_sequence_l4079_407966


namespace NUMINAMATH_CALUDE_waste_fraction_for_park_l4079_407986

/-- A kite-shaped park with specific properties -/
structure KitePark where
  -- AB and BC lengths
  side_length : ℝ
  -- Ensure side_length is positive
  side_positive : side_length > 0

/-- The fraction of the park's area from which waste is brought to the longest diagonal -/
noncomputable def waste_fraction (park : KitePark) : ℝ :=
  7071 / 10000

/-- Theorem stating the waste fraction for a kite park with side length 100 -/
theorem waste_fraction_for_park (park : KitePark) 
  (h : park.side_length = 100) : 
  waste_fraction park = 7071 / 10000 :=
by sorry

end NUMINAMATH_CALUDE_waste_fraction_for_park_l4079_407986


namespace NUMINAMATH_CALUDE_crayons_difference_l4079_407997

def birthday_crayons : ℕ := 8597
def crayons_given : ℕ := 7255
def crayons_lost : ℕ := 3689

theorem crayons_difference : crayons_given - crayons_lost = 3566 := by
  sorry

end NUMINAMATH_CALUDE_crayons_difference_l4079_407997


namespace NUMINAMATH_CALUDE_max_value_implies_a_l4079_407900

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := -9 * x^2 - 6 * a * x + 2 * a - a^2

-- State the theorem
theorem max_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-1/3 : ℝ) (1/3 : ℝ), f a x ≤ -3) ∧
  (∃ x ∈ Set.Icc (-1/3 : ℝ) (1/3 : ℝ), f a x = -3) →
  a = Real.sqrt 6 + 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_implies_a_l4079_407900


namespace NUMINAMATH_CALUDE_binomial_odd_prob_odd_prob_increases_l4079_407930

/-- Binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- Probability of X being odd in a binomial distribution -/
def prob_odd (n : ℕ) (p : ℝ) : ℝ := sorry

/-- Sum of odd-indexed binomial probabilities -/
def sum_odd_binomial (n : ℕ) (p : ℝ) : ℝ := sorry

theorem binomial_odd_prob (n : ℕ) (p : ℝ) 
  (h1 : n > 0) (h2 : 0 < p) (h3 : p < 1) :
  prob_odd n p = sum_odd_binomial n p := by sorry

theorem odd_prob_increases (n : ℕ) (p : ℝ) 
  (h1 : n > 0) (h2 : 0 < p) (h3 : p < 1/2) :
  prob_odd (n + 1) p > prob_odd n p := by sorry

end NUMINAMATH_CALUDE_binomial_odd_prob_odd_prob_increases_l4079_407930


namespace NUMINAMATH_CALUDE_sqrt_of_square_l4079_407978

theorem sqrt_of_square (x : ℝ) (h : x = 25) : Real.sqrt (x ^ 2) = |x| := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_square_l4079_407978


namespace NUMINAMATH_CALUDE_second_number_value_l4079_407907

def average_of_three (a b c : ℚ) : ℚ := (a + b + c) / 3

theorem second_number_value (x y : ℚ) 
  (h1 : average_of_three 2 y x = 5) 
  (h2 : x = -63) : 
  y = 76 := by
  sorry

end NUMINAMATH_CALUDE_second_number_value_l4079_407907


namespace NUMINAMATH_CALUDE_cube_side_length_l4079_407922

theorem cube_side_length (s : ℝ) : s > 0 → (6 * s^2 = s^3) → s = 6 := by
  sorry

end NUMINAMATH_CALUDE_cube_side_length_l4079_407922


namespace NUMINAMATH_CALUDE_magical_red_knights_fraction_l4079_407993

theorem magical_red_knights_fraction (total : ℚ) (red : ℚ) (blue : ℚ) (magical : ℚ) 
  (h1 : red = 3 / 8 * total)
  (h2 : blue = total - red)
  (h3 : magical = 1 / 4 * total)
  (h4 : ∃ (x y : ℚ), x / y > 0 ∧ red * (x / y) = 3 * (blue * (x / (3 * y))) ∧ red * (x / y) + blue * (x / (3 * y)) = magical) :
  ∃ (x y : ℚ), x / y = 3 / 7 ∧ red * (x / y) = magical := by
  sorry

end NUMINAMATH_CALUDE_magical_red_knights_fraction_l4079_407993


namespace NUMINAMATH_CALUDE_largest_common_divisor_17_30_l4079_407915

theorem largest_common_divisor_17_30 : 
  ∃ (n : ℕ), n > 0 ∧ n = 13 ∧ 
  (∀ (m : ℕ), m > 0 → 17 % m = 30 % m → m ≤ n) :=
sorry

end NUMINAMATH_CALUDE_largest_common_divisor_17_30_l4079_407915


namespace NUMINAMATH_CALUDE_class_size_roses_class_size_l4079_407933

theorem class_size (girls_present : ℕ) (boys_absent : ℕ) : ℕ :=
  let boys_present := girls_present / 2
  let total_boys := boys_present + boys_absent
  let total_students := girls_present + total_boys
  total_students

theorem roses_class_size : class_size 140 40 = 250 := by
  sorry

end NUMINAMATH_CALUDE_class_size_roses_class_size_l4079_407933


namespace NUMINAMATH_CALUDE_gcf_of_360_and_150_l4079_407971

theorem gcf_of_360_and_150 : Nat.gcd 360 150 = 30 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_360_and_150_l4079_407971


namespace NUMINAMATH_CALUDE_f_g_minus_g_f_l4079_407959

def f (x : ℝ) : ℝ := 2 * x - 1

def g (x : ℝ) : ℝ := x^2 + 2 * x + 1

theorem f_g_minus_g_f : f (g 3) - g (f 3) = -5 := by
  sorry

end NUMINAMATH_CALUDE_f_g_minus_g_f_l4079_407959


namespace NUMINAMATH_CALUDE_cubic_inequality_solution_l4079_407947

theorem cubic_inequality_solution (x : ℝ) : 
  x^3 - x^2 + 11*x - 30 < 12 ↔ (x > -2 ∧ x < 3) ∨ (x > 3 ∧ x < 7) :=
sorry

end NUMINAMATH_CALUDE_cubic_inequality_solution_l4079_407947


namespace NUMINAMATH_CALUDE_pen_count_l4079_407996

/-- The number of pens in Maria's desk drawer -/
theorem pen_count (red : ℕ) (black : ℕ) (blue : ℕ) (green : ℕ) : 
  red = 8 →
  black = 2 * red →
  blue = black + 5 →
  green = blue / 2 →
  red + black + blue + green = 55 := by
sorry

end NUMINAMATH_CALUDE_pen_count_l4079_407996


namespace NUMINAMATH_CALUDE_solution_value_l4079_407991

theorem solution_value (a : ℝ) : (2 * a = 4) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l4079_407991


namespace NUMINAMATH_CALUDE_smallest_n_for_radio_profit_l4079_407932

theorem smallest_n_for_radio_profit (n d : ℕ) (h1 : d > 0) : 
  (∃ (m : ℕ), m ≥ n ∧ 
    d - (3 * d) / (2 * m) + 10 * m - 30 = d + 100 ∧
    (∀ k : ℕ, k < m → d - (3 * d) / (2 * k) + 10 * k - 30 ≠ d + 100)) →
  n = 13 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_for_radio_profit_l4079_407932


namespace NUMINAMATH_CALUDE_tree_initial_height_l4079_407990

/-- Represents the height of a tree over time -/
def TreeHeight (H : ℝ) (t : ℕ) : ℝ := H + t

/-- The problem statement about the tree's growth -/
theorem tree_initial_height :
  ∀ H : ℝ,
  (TreeHeight H 6 = TreeHeight H 4 + (1/4) * TreeHeight H 4) →
  H = 4 := by
  sorry

end NUMINAMATH_CALUDE_tree_initial_height_l4079_407990


namespace NUMINAMATH_CALUDE_right_triangle_area_l4079_407944

/-- Given a right triangle ABC with ∠C = 90°, a + b = 14 cm, and c = 10 cm, 
    the area of the triangle is 24 cm². -/
theorem right_triangle_area (a b c : ℝ) : 
  a + b = 14 → c = 10 → a^2 + b^2 = c^2 → (1/2) * a * b = 24 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l4079_407944


namespace NUMINAMATH_CALUDE_minimum_handshakes_l4079_407940

theorem minimum_handshakes (n : ℕ) (k : ℕ) (h1 : n = 30) (h2 : k = 3) :
  (n * k) / 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_minimum_handshakes_l4079_407940


namespace NUMINAMATH_CALUDE_tan_150_degrees_l4079_407908

theorem tan_150_degrees : Real.tan (150 * π / 180) = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_150_degrees_l4079_407908


namespace NUMINAMATH_CALUDE_desired_circle_properties_l4079_407927

/-- The equation of the first given circle -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - x + y - 2 = 0

/-- The equation of the second given circle -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 5

/-- The equation of the line on which the center of the desired circle lies -/
def centerLine (x y : ℝ) : Prop := 3*x + 4*y - 1 = 0

/-- The equation of the desired circle -/
def desiredCircle (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 2*y - 11 = 0

/-- Theorem stating that the desired circle passes through the intersection points of circle1 and circle2,
    and its center lies on the centerLine -/
theorem desired_circle_properties :
  ∀ (x y : ℝ),
    (circle1 x y ∧ circle2 x y → desiredCircle x y) ∧
    (∃ (h k : ℝ), desiredCircle h k ∧ centerLine h k) :=
by sorry

end NUMINAMATH_CALUDE_desired_circle_properties_l4079_407927


namespace NUMINAMATH_CALUDE_michelle_crayons_l4079_407974

/-- The number of crayons Michelle has -/
def total_crayons (crayons_per_box : ℝ) (num_boxes : ℝ) : ℝ :=
  crayons_per_box * num_boxes

/-- Proof that Michelle has 7.0 crayons -/
theorem michelle_crayons :
  total_crayons 5.0 1.4 = 7.0 := by
  sorry

end NUMINAMATH_CALUDE_michelle_crayons_l4079_407974


namespace NUMINAMATH_CALUDE_min_triangle_area_l4079_407916

/-- A point in the 2D plane with integer coordinates -/
structure IntPoint where
  x : ℤ
  y : ℤ

/-- The rectangle OABC -/
structure Rectangle where
  O : IntPoint
  B : IntPoint

/-- Checks if a point is inside the rectangle -/
def isInside (r : Rectangle) (p : IntPoint) : Prop :=
  0 ≤ p.x ∧ p.x ≤ r.B.x ∧ 0 ≤ p.y ∧ p.y ≤ r.B.y

/-- Calculates the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : IntPoint) : ℚ :=
  (1 / 2 : ℚ) * |p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y)|

/-- The main theorem -/
theorem min_triangle_area (r : Rectangle) :
  r.O = ⟨0, 0⟩ → r.B = ⟨11, 8⟩ →
  ∃ (X : IntPoint), isInside r X ∧
    ∀ (Y : IntPoint), isInside r Y →
      triangleArea r.O r.B X ≤ triangleArea r.O r.B Y ∧
      triangleArea r.O r.B X = (1 / 2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_min_triangle_area_l4079_407916


namespace NUMINAMATH_CALUDE_second_chapter_pages_l4079_407950

/-- A book with two chapters -/
structure Book where
  total_pages : ℕ
  first_chapter_pages : ℕ
  second_chapter_pages : ℕ

/-- Properties of the book -/
def book_properties (b : Book) : Prop :=
  b.total_pages = 93 ∧ b.first_chapter_pages = 60 ∧ b.total_pages = b.first_chapter_pages + b.second_chapter_pages

/-- Theorem stating that the second chapter has 33 pages -/
theorem second_chapter_pages (b : Book) (h : book_properties b) : b.second_chapter_pages = 33 := by
  sorry

end NUMINAMATH_CALUDE_second_chapter_pages_l4079_407950


namespace NUMINAMATH_CALUDE_rectangle_width_l4079_407923

/-- Given a rectangle and a triangle, if the ratio of their areas is 2:5,
    the rectangle has length 6 cm, and the triangle has area 60 cm²,
    then the width of the rectangle is 4 cm. -/
theorem rectangle_width (length width : ℝ) (triangle_area : ℝ) : 
  length = 6 →
  triangle_area = 60 →
  (length * width) / triangle_area = 2 / 5 →
  width = 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_l4079_407923


namespace NUMINAMATH_CALUDE_outfit_choices_l4079_407937

/-- Represents the number of available items of each type -/
def num_items : ℕ := 7

/-- Represents the number of available colors -/
def num_colors : ℕ := 7

/-- Calculates the total number of possible outfits -/
def total_outfits : ℕ := num_items * num_items * num_items

/-- Calculates the number of outfits where all items are the same color -/
def same_color_outfits : ℕ := num_colors

/-- Calculates the number of valid outfits (not all items the same color) -/
def valid_outfits : ℕ := total_outfits - same_color_outfits

theorem outfit_choices :
  valid_outfits = 336 :=
sorry

end NUMINAMATH_CALUDE_outfit_choices_l4079_407937


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l4079_407938

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (1 - 3*i) / (2 - i) = 1 - i := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l4079_407938


namespace NUMINAMATH_CALUDE_unique_real_solution_l4079_407934

theorem unique_real_solution (b : ℝ) :
  ∀ a : ℝ, (∃! x : ℝ, x^3 - a*x^2 - (2*a + b)*x + a^2 + b = 0) ↔ a < 3 + 4*b :=
by sorry

end NUMINAMATH_CALUDE_unique_real_solution_l4079_407934


namespace NUMINAMATH_CALUDE_system_equation_range_l4079_407910

theorem system_equation_range (x y m : ℝ) : 
  x + 2*y = 1 + m → 
  2*x + y = -3 → 
  x + y > 0 → 
  m > 2 := by
sorry

end NUMINAMATH_CALUDE_system_equation_range_l4079_407910


namespace NUMINAMATH_CALUDE_for_loop_properties_l4079_407998

/-- Represents a for loop in a programming language --/
structure ForLoop where
  init : ℕ
  final : ℕ
  step : ℤ
  body : List (Unit → Unit)

/-- The loop expression of a for loop --/
def loopExpression (loop : ForLoop) : List (Unit → Unit) := loop.body

/-- The loop body of a for loop --/
def loopBody (loop : ForLoop) : List (Unit → Unit) := loop.body

/-- Function to check if step can be omitted --/
def canOmitStep (loop : ForLoop) : Bool := loop.step = 1

/-- Function to check if loop can proceed --/
def canProceed (loop : ForLoop) : Bool := loop.final ≠ 0

/-- Function representing the control of loop termination and new loop start --/
def loopControl (loop : ForLoop) : Unit → Unit :=
  fun _ => ()  -- Placeholder function

theorem for_loop_properties (loop : ForLoop) : 
  (loopExpression loop = loopBody loop) ∧ 
  (canOmitStep loop ↔ loop.step = 1) ∧
  (canProceed loop ↔ loop.final ≠ 0) ∧
  (loopControl loop ≠ fun _ => ()) := by sorry

end NUMINAMATH_CALUDE_for_loop_properties_l4079_407998


namespace NUMINAMATH_CALUDE_positive_A_value_l4079_407962

-- Define the # relation
def hash (A B : ℝ) : ℝ := A^2 + B^2

-- Theorem statement
theorem positive_A_value (A : ℝ) : 
  (hash A 5 = 169) → (A > 0) → (A = 12) := by
  sorry

end NUMINAMATH_CALUDE_positive_A_value_l4079_407962


namespace NUMINAMATH_CALUDE_tangent_parallel_points_tangent_points_on_curve_unique_tangent_points_l4079_407981

-- Define the function f(x) = x^3 + x - 2
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

theorem tangent_parallel_points :
  ∀ x : ℝ, (f' x = 4) ↔ (x = 1 ∨ x = -1) :=
sorry

theorem tangent_points_on_curve :
  f 1 = 0 ∧ f (-1) = -4 :=
sorry

theorem unique_tangent_points :
  ∀ x : ℝ, f' x = 4 → (x = 1 ∨ x = -1) :=
sorry

end NUMINAMATH_CALUDE_tangent_parallel_points_tangent_points_on_curve_unique_tangent_points_l4079_407981


namespace NUMINAMATH_CALUDE_triangle_inequality_l4079_407917

theorem triangle_inequality (a b c x y z : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0)
  (h1 : a * y + b * x = c)
  (h2 : c * x + a * z = b)
  (h3 : b * z + c * y = a) :
  (x / (1 - y * z)) + (y / (1 - z * x)) + (z / (1 - x * y)) ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l4079_407917


namespace NUMINAMATH_CALUDE_prob_excellent_given_pass_is_correct_l4079_407989

/-- The number of total questions in the exam -/
def total_questions : ℕ := 20

/-- The number of questions selected for the exam -/
def selected_questions : ℕ := 6

/-- The number of questions the student can correctly answer -/
def correct_answers : ℕ := 10

/-- The minimum number of correct answers required to pass the exam -/
def pass_threshold : ℕ := 4

/-- The minimum number of correct answers required for an excellent grade -/
def excellent_threshold : ℕ := 5

/-- The probability of achieving an excellent grade given that the student has passed the exam -/
def prob_excellent_given_pass : ℚ := 13/58

/-- Theorem stating that the probability of achieving an excellent grade, 
    given that the student has passed the exam, is 13/58 -/
theorem prob_excellent_given_pass_is_correct :
  prob_excellent_given_pass = 13/58 := by
  sorry

end NUMINAMATH_CALUDE_prob_excellent_given_pass_is_correct_l4079_407989


namespace NUMINAMATH_CALUDE_b_work_days_l4079_407972

/-- Represents the number of days it takes for a person to complete the work alone -/
structure WorkDays where
  days : ℕ

/-- Represents the rate at which a person completes the work per day -/
def workRate (w : WorkDays) : ℚ :=
  1 / w.days

theorem b_work_days (total_payment : ℕ) (a_work : WorkDays) (abc_work : WorkDays) (c_share : ℕ) :
  total_payment = 1200 →
  a_work.days = 6 →
  abc_work.days = 3 →
  c_share = 150 →
  ∃ b_work : WorkDays,
    b_work.days = 24 ∧
    workRate a_work + workRate b_work + (c_share : ℚ) / total_payment = workRate abc_work :=
by sorry

end NUMINAMATH_CALUDE_b_work_days_l4079_407972


namespace NUMINAMATH_CALUDE_integer_solutions_cubic_equation_l4079_407973

theorem integer_solutions_cubic_equation :
  ∀ x y : ℤ, x^3 - y^3 = 2*x*y + 8 ↔ (x = 0 ∧ y = -2) ∨ (x = 2 ∧ y = 0) :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_cubic_equation_l4079_407973


namespace NUMINAMATH_CALUDE_f_is_odd_iff_l4079_407941

-- Define the function f(x) = x|x + a| + b
def f (a b x : ℝ) : ℝ := x * abs (x + a) + b

-- State the theorem
theorem f_is_odd_iff (a b : ℝ) :
  (∀ x, f a b (-x) = -f a b x) ↔ 
  (∀ x, x * abs (-x + a) + b = -(x * abs (x + a) + b)) :=
by sorry

end NUMINAMATH_CALUDE_f_is_odd_iff_l4079_407941


namespace NUMINAMATH_CALUDE_no_square_cut_with_250_remaining_l4079_407975

theorem no_square_cut_with_250_remaining : ¬∃ (n m : ℕ), n > m ∧ n^2 - m^2 = 250 := by
  sorry

end NUMINAMATH_CALUDE_no_square_cut_with_250_remaining_l4079_407975


namespace NUMINAMATH_CALUDE_opposite_is_three_l4079_407931

theorem opposite_is_three (x : ℝ) : -x = 3 → x = -3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_is_three_l4079_407931


namespace NUMINAMATH_CALUDE_team_a_champion_probability_l4079_407967

/-- The probability of a team winning a single game -/
def win_prob : ℚ := 1/2

/-- The probability of Team A becoming the champion -/
def champion_prob : ℚ := win_prob + win_prob * win_prob

theorem team_a_champion_probability :
  champion_prob = 3/4 :=
by sorry

end NUMINAMATH_CALUDE_team_a_champion_probability_l4079_407967


namespace NUMINAMATH_CALUDE_regular_octagon_interior_angle_l4079_407921

/-- The measure of one interior angle of a regular octagon is 135 degrees. -/
theorem regular_octagon_interior_angle : ℝ :=
  let n : ℕ := 8  -- number of sides in an octagon
  let sum_interior_angles : ℝ := 180 * (n - 2)  -- sum of interior angles formula
  let one_interior_angle : ℝ := sum_interior_angles / n  -- measure of one interior angle
  135

/-- Proof of the theorem -/
lemma prove_regular_octagon_interior_angle : regular_octagon_interior_angle = 135 := by
  sorry

end NUMINAMATH_CALUDE_regular_octagon_interior_angle_l4079_407921


namespace NUMINAMATH_CALUDE_no_rain_probability_l4079_407929

/-- The probability of an event occurring on Monday -/
def prob_monday : ℝ := 0.7

/-- The probability of an event occurring on Tuesday -/
def prob_tuesday : ℝ := 0.5

/-- The probability of an event occurring on both Monday and Tuesday -/
def prob_both_days : ℝ := 0.4

/-- The probability of an event not occurring on either Monday or Tuesday -/
def prob_no_rain : ℝ := 1 - (prob_monday + prob_tuesday - prob_both_days)

theorem no_rain_probability :
  prob_no_rain = 0.2 := by sorry

end NUMINAMATH_CALUDE_no_rain_probability_l4079_407929


namespace NUMINAMATH_CALUDE_cloth_cost_l4079_407948

/-- Given a piece of cloth with the following properties:
  1. The original length is 10 meters.
  2. Increasing the length by 4 meters and decreasing the cost per meter by 1 rupee
     leaves the total cost unchanged.
  This theorem proves that the total cost of the original piece is 35 rupees. -/
theorem cloth_cost (original_length : ℝ) (cost_per_meter : ℝ) 
  (h1 : original_length = 10)
  (h2 : original_length * cost_per_meter = (original_length + 4) * (cost_per_meter - 1)) :
  original_length * cost_per_meter = 35 := by
sorry

end NUMINAMATH_CALUDE_cloth_cost_l4079_407948


namespace NUMINAMATH_CALUDE_scorpion_daily_segments_l4079_407992

/-- The number of body segments a cave scorpion needs to eat daily -/
def daily_segments : ℕ :=
  let segments_first_millipede := 60
  let segments_long_millipede := 2 * segments_first_millipede
  let segments_eaten := segments_first_millipede + 2 * segments_long_millipede
  let segments_to_eat := 10 * 50
  segments_eaten + segments_to_eat

theorem scorpion_daily_segments : daily_segments = 800 := by
  sorry

end NUMINAMATH_CALUDE_scorpion_daily_segments_l4079_407992


namespace NUMINAMATH_CALUDE_coral_reef_below_5_percent_l4079_407924

def coral_reef_count (initial_count : ℝ) (year : ℕ) : ℝ :=
  initial_count * (0.7 ^ year)

def year_2010 : ℕ := 2010

theorem coral_reef_below_5_percent (initial_count : ℝ) :
  (∀ y < 10, coral_reef_count initial_count y > 0.05 * initial_count) ∧
  coral_reef_count initial_count 10 < 0.05 * initial_count :=
sorry

end NUMINAMATH_CALUDE_coral_reef_below_5_percent_l4079_407924


namespace NUMINAMATH_CALUDE_min_lines_is_seven_l4079_407956

/-- A line in a 2D Cartesian coordinate system -/
structure Line where
  k : ℝ
  b : ℝ
  k_nonzero : k ≠ 0

/-- The set of quadrants a line passes through -/
def quadrants (l : Line) : Set (Fin 4) :=
  sorry

/-- The minimum number of lines needed to guarantee two lines pass through the same quadrants -/
def min_lines_same_quadrants : ℕ :=
  sorry

/-- Theorem stating that the minimum number of lines is 7 -/
theorem min_lines_is_seven : min_lines_same_quadrants = 7 := by
  sorry

end NUMINAMATH_CALUDE_min_lines_is_seven_l4079_407956


namespace NUMINAMATH_CALUDE_absolute_value_sum_l4079_407909

theorem absolute_value_sum (a : ℝ) (h : -2 < a ∧ a < 0) : |a| + |a+2| = 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sum_l4079_407909


namespace NUMINAMATH_CALUDE_profit_calculation_l4079_407988

/-- Calculates the actual percent profit given the markup percentage and discount percentage -/
def actualPercentProfit (markup : ℝ) (discount : ℝ) : ℝ :=
  let labeledPrice := 1 + markup
  let sellingPrice := labeledPrice * (1 - discount)
  (sellingPrice - 1) * 100

/-- Theorem stating that a 40% markup with a 5% discount results in a 33% profit -/
theorem profit_calculation (markup discount : ℝ) 
  (h1 : markup = 0.4) 
  (h2 : discount = 0.05) : 
  actualPercentProfit markup discount = 33 := by
  sorry

end NUMINAMATH_CALUDE_profit_calculation_l4079_407988


namespace NUMINAMATH_CALUDE_sum_of_possible_A_values_l4079_407951

/-- The sum of digits of a number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Check if a number is divisible by 9 -/
def is_divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

/-- The given number with A as a parameter -/
def given_number (A : ℕ) : ℕ := 7456291 * 10 + A * 10 + 2

theorem sum_of_possible_A_values : 
  (∀ A : ℕ, A < 10 → is_divisible_by_9 (given_number A) → 
    sum_of_digits (given_number A) = sum_of_digits 7456291 + A + 2) →
  (∃ A₁ A₂ : ℕ, A₁ < 10 ∧ A₂ < 10 ∧ 
    is_divisible_by_9 (given_number A₁) ∧ 
    is_divisible_by_9 (given_number A₂) ∧
    A₁ + A₂ = 9) :=
sorry

end NUMINAMATH_CALUDE_sum_of_possible_A_values_l4079_407951


namespace NUMINAMATH_CALUDE_problem_statement_l4079_407906

theorem problem_statement (x₁ x₂ x₃ a : ℝ) 
  (h_distinct : x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₃ ≠ x₁)
  (h_nonzero : x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ x₃ ≠ 0)
  (h_eq1 : x₁ * x₂ - a * x₁ + a^2 = 0)
  (h_eq2 : x₂ * x₃ - a * x₂ + a^2 = 0) :
  (x₃ * x₁ - a * x₃ + a^2 = 0) ∧
  (x₁ * x₂ * x₃ + a^3 = 0) ∧
  (1 / (x₁ - x₂) + 1 / (x₂ - x₃) + 1 / (x₃ - x₁) = 1 / a) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l4079_407906


namespace NUMINAMATH_CALUDE_common_volume_formula_l4079_407920

/-- Represents a cube with edge length a -/
structure Cube where
  a : ℝ
  a_pos : a > 0

/-- Represents a regular tetrahedron -/
structure RegularTetrahedron where
  edge_length : ℝ
  edge_length_pos : edge_length > 0

/-- Represents the configuration of a cube and a tetrahedron with aligned edges and coinciding midpoints -/
structure CubeTetrahedronConfig where
  cube : Cube
  tetrahedron : RegularTetrahedron
  aligned_edges : Bool
  coinciding_midpoints : Bool

/-- Calculates the volume of the common part of a cube and a tetrahedron in the given configuration -/
def common_volume (config : CubeTetrahedronConfig) : ℝ := sorry

/-- Theorem stating the volume of the common part of the cube and tetrahedron -/
theorem common_volume_formula (config : CubeTetrahedronConfig) 
  (h_aligned : config.aligned_edges = true) 
  (h_coincide : config.coinciding_midpoints = true) :
  common_volume config = (config.cube.a^3 * Real.sqrt 2 / 12) * (16 * Real.sqrt 2 - 17) := by
  sorry

end NUMINAMATH_CALUDE_common_volume_formula_l4079_407920


namespace NUMINAMATH_CALUDE_roots_sum_of_squares_l4079_407963

theorem roots_sum_of_squares (α β : ℝ) : 
  (α^2 - 2*α - 1 = 0) → (β^2 - 2*β - 1 = 0) → α^2 + β^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_squares_l4079_407963


namespace NUMINAMATH_CALUDE_pet_store_transactions_l4079_407925

/-- Represents the number of pets of each type -/
structure PetCounts where
  puppies : ℕ
  kittens : ℕ
  rabbits : ℕ
  guinea_pigs : ℕ
  chameleons : ℕ

/-- Calculates the total number of pets -/
def total_pets (counts : PetCounts) : ℕ :=
  counts.puppies + counts.kittens + counts.rabbits + counts.guinea_pigs + counts.chameleons

/-- Represents the sales and returns for each type of pet -/
structure Transactions where
  puppies_sold : ℕ
  kittens_sold : ℕ
  rabbits_sold : ℕ
  guinea_pigs_sold : ℕ
  chameleons_sold : ℕ
  kittens_returned : ℕ
  guinea_pigs_returned : ℕ
  chameleons_returned : ℕ

/-- Applies transactions to the pet counts -/
def apply_transactions (initial : PetCounts) (trans : Transactions) : PetCounts :=
  { puppies := initial.puppies - trans.puppies_sold,
    kittens := initial.kittens - trans.kittens_sold + trans.kittens_returned,
    rabbits := initial.rabbits - trans.rabbits_sold,
    guinea_pigs := initial.guinea_pigs - trans.guinea_pigs_sold + trans.guinea_pigs_returned,
    chameleons := initial.chameleons - trans.chameleons_sold + trans.chameleons_returned }

theorem pet_store_transactions :
  let initial_counts : PetCounts := { puppies := 7, kittens := 6, rabbits := 4, guinea_pigs := 5, chameleons := 3 }
  let transactions : Transactions := { puppies_sold := 2, kittens_sold := 3, rabbits_sold := 3,
                                       guinea_pigs_sold := 3, chameleons_sold := 0,
                                       kittens_returned := 1, guinea_pigs_returned := 1, chameleons_returned := 1 }
  let final_counts := apply_transactions initial_counts transactions
  total_pets final_counts = 15 := by
  sorry


end NUMINAMATH_CALUDE_pet_store_transactions_l4079_407925


namespace NUMINAMATH_CALUDE_prime_factor_sum_l4079_407960

theorem prime_factor_sum (w x y z t : ℕ) : 
  2^w * 3^x * 5^y * 7^z * 17^t = 107100 →
  2*w + 3*x + 5*y + 7*z + 11*t = 38 := by
sorry

end NUMINAMATH_CALUDE_prime_factor_sum_l4079_407960


namespace NUMINAMATH_CALUDE_new_refrigerator_cost_l4079_407965

/-- The daily cost of electricity for Kurt's old refrigerator in dollars -/
def old_cost : ℚ := 85/100

/-- The number of days in a month -/
def days_in_month : ℕ := 30

/-- The amount Kurt saves in a month with his new refrigerator in dollars -/
def monthly_savings : ℚ := 12

/-- The daily cost of electricity for Kurt's new refrigerator in dollars -/
def new_cost : ℚ := 45/100

theorem new_refrigerator_cost :
  (days_in_month : ℚ) * old_cost - (days_in_month : ℚ) * new_cost = monthly_savings :=
by sorry

end NUMINAMATH_CALUDE_new_refrigerator_cost_l4079_407965


namespace NUMINAMATH_CALUDE_valid_pairs_l4079_407970

def is_valid_pair (x y : ℕ+) : Prop :=
  (1 : ℚ) / x + (1 : ℚ) / y + (1 : ℚ) / Nat.lcm x y + (1 : ℚ) / Nat.gcd x y = (1 : ℚ) / 2

theorem valid_pairs : 
  ∀ x y : ℕ+, is_valid_pair x y ↔ 
    ((x = 5 ∧ y = 20) ∨ 
     (x = 6 ∧ y = 12) ∨ 
     (x = 8 ∧ y = 8) ∨ 
     (x = 8 ∧ y = 12) ∨ 
     (x = 9 ∧ y = 24) ∨ 
     (x = 12 ∧ y = 15) ∨
     (y = 5 ∧ x = 20) ∨ 
     (y = 6 ∧ x = 12) ∨ 
     (y = 8 ∧ x = 12) ∨ 
     (y = 9 ∧ x = 24) ∨ 
     (y = 12 ∧ x = 15)) :=
by sorry

end NUMINAMATH_CALUDE_valid_pairs_l4079_407970


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l4079_407946

/-- Given that x varies inversely with y³ and x = 8 when y = 1, prove that x = 1 when y = 2 -/
theorem inverse_variation_problem (x y : ℝ) (k : ℝ) : 
  (∀ y, x * y^3 = k) →  -- x varies inversely with y³
  (8 * 1^3 = k) →       -- x = 8 when y = 1
  (1 * 2^3 = k) →       -- x = 1 when y = 2
  True := by sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l4079_407946


namespace NUMINAMATH_CALUDE_iphone_discount_l4079_407926

theorem iphone_discount (iphone_price iwatch_price iwatch_discount cashback_rate final_price : ℝ) :
  iphone_price = 800 →
  iwatch_price = 300 →
  iwatch_discount = 0.1 →
  cashback_rate = 0.02 →
  final_price = 931 →
  ∃ (iphone_discount : ℝ),
    iphone_discount = 0.15 ∧
    final_price = (1 - cashback_rate) * (iphone_price * (1 - iphone_discount) + iwatch_price * (1 - iwatch_discount)) :=
by sorry

end NUMINAMATH_CALUDE_iphone_discount_l4079_407926


namespace NUMINAMATH_CALUDE_direct_variation_problem_l4079_407985

/-- z varies directly as w -/
def direct_variation (z w : ℝ) := ∃ k : ℝ, z = k * w

theorem direct_variation_problem (z w : ℝ → ℝ) :
  (∀ x, direct_variation (z x) (w x)) →  -- z varies directly as w
  z 5 = 10 →                             -- z = 10 when w = 5
  w 5 = 5 →                              -- w = 5 when z = 10
  w (-15) = -15 →                        -- w = -15
  z (-15) = -30                          -- z = -30 when w = -15
  := by sorry

end NUMINAMATH_CALUDE_direct_variation_problem_l4079_407985


namespace NUMINAMATH_CALUDE_total_students_l4079_407957

theorem total_students (students_per_group : ℕ) (groups_per_class : ℕ) (classes : ℕ)
  (h1 : students_per_group = 7)
  (h2 : groups_per_class = 9)
  (h3 : classes = 13) :
  students_per_group * groups_per_class * classes = 819 := by
  sorry

end NUMINAMATH_CALUDE_total_students_l4079_407957


namespace NUMINAMATH_CALUDE_sprocket_production_rate_l4079_407955

theorem sprocket_production_rate : ∀ (a g : ℝ),
  -- Machine G produces 10% more sprockets per hour than Machine A
  g = 1.1 * a →
  -- Machine A takes 10 hours longer to produce 660 sprockets
  660 / a = 660 / g + 10 →
  -- The production rate of Machine A
  a = 6 := by
sorry

end NUMINAMATH_CALUDE_sprocket_production_rate_l4079_407955


namespace NUMINAMATH_CALUDE_next_free_haircut_in_ten_l4079_407942

-- Define the constants from the problem
def haircuts_per_free : ℕ := 14
def free_haircuts_received : ℕ := 5
def total_haircuts : ℕ := 79

-- Define a function to calculate the number of haircuts until the next free one
def haircuts_until_next_free (total : ℕ) (free : ℕ) (per_free : ℕ) : ℕ :=
  per_free - (total - free) % per_free

-- Theorem statement
theorem next_free_haircut_in_ten :
  haircuts_until_next_free total_haircuts free_haircuts_received haircuts_per_free = 10 := by
  sorry


end NUMINAMATH_CALUDE_next_free_haircut_in_ten_l4079_407942


namespace NUMINAMATH_CALUDE_draw_condition_butterfly_wins_condition_l4079_407969

/-- Represents the outcome of the spider web game -/
inductive GameOutcome
  | Draw
  | ButterflyWins

/-- Defines the spider web game structure and rules -/
structure SpiderWebGame where
  K : Nat  -- Number of rings
  R : Nat  -- Number of radii
  butterfly_moves_first : Bool
  K_ge_2 : K ≥ 2
  R_ge_3 : R ≥ 3

/-- Determines the outcome of the spider web game -/
def game_outcome (game : SpiderWebGame) : GameOutcome :=
  if game.K ≥ Nat.ceil (game.R / 2) then
    GameOutcome.Draw
  else
    GameOutcome.ButterflyWins

/-- Theorem stating the conditions for a draw in the spider web game -/
theorem draw_condition (game : SpiderWebGame) :
  game_outcome game = GameOutcome.Draw ↔ game.K ≥ Nat.ceil (game.R / 2) :=
sorry

/-- Theorem stating the conditions for butterfly winning in the spider web game -/
theorem butterfly_wins_condition (game : SpiderWebGame) :
  game_outcome game = GameOutcome.ButterflyWins ↔ game.K < Nat.ceil (game.R / 2) :=
sorry

end NUMINAMATH_CALUDE_draw_condition_butterfly_wins_condition_l4079_407969


namespace NUMINAMATH_CALUDE_same_color_probability_l4079_407964

/-- The number of green balls in the bag -/
def green_balls : ℕ := 8

/-- The number of red balls in the bag -/
def red_balls : ℕ := 6

/-- The number of blue balls in the bag -/
def blue_balls : ℕ := 1

/-- The total number of balls in the bag -/
def total_balls : ℕ := green_balls + red_balls + blue_balls

/-- The probability of drawing two balls of the same color with replacement -/
theorem same_color_probability : 
  (green_balls^2 + red_balls^2 + blue_balls^2) / total_balls^2 = 101 / 225 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l4079_407964


namespace NUMINAMATH_CALUDE_M_intersect_complement_N_l4079_407979

def R := ℝ

def M : Set ℝ := {x : ℝ | x^2 - 2*x < 0}

def N : Set ℝ := {x : ℝ | x ≥ 1}

theorem M_intersect_complement_N :
  M ∩ (Set.univ \ N) = Set.Ioo 0 1 := by
  sorry

end NUMINAMATH_CALUDE_M_intersect_complement_N_l4079_407979


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l4079_407980

theorem least_addition_for_divisibility : ∃! x : ℕ, 
  (x ≤ 22) ∧ (∃ k : ℤ, 1077 + x = 23 * k) ∧ 
  (∀ y : ℕ, y < x → ¬∃ k : ℤ, 1077 + y = 23 * k) :=
by sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l4079_407980


namespace NUMINAMATH_CALUDE_remainder_problem_l4079_407987

theorem remainder_problem (x : ℤ) : x % 63 = 11 → x % 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l4079_407987


namespace NUMINAMATH_CALUDE_average_cost_before_gratuity_l4079_407984

/-- Given a group of 10 people with a total bill including 25% gratuity,
    calculate the average cost per person before gratuity. -/
theorem average_cost_before_gratuity
  (X : ℝ) -- Total bill including gratuity
  (h : X > 0) -- Assume the bill is positive
  : (X / 12.5 : ℝ) = (X / (1.25 * 10) : ℝ) := by
  sorry

#check average_cost_before_gratuity

end NUMINAMATH_CALUDE_average_cost_before_gratuity_l4079_407984


namespace NUMINAMATH_CALUDE_margaret_score_l4079_407949

theorem margaret_score (average_score : ℝ) (marco_percentage : ℝ) (margaret_difference : ℝ) : 
  average_score = 90 →
  marco_percentage = 0.1 →
  margaret_difference = 5 →
  let marco_score := average_score * (1 - marco_percentage)
  let margaret_score := marco_score + margaret_difference
  margaret_score = 86 := by sorry

end NUMINAMATH_CALUDE_margaret_score_l4079_407949


namespace NUMINAMATH_CALUDE_tan_half_sum_l4079_407995

theorem tan_half_sum (a b : Real) 
  (h1 : Real.cos a + Real.cos b = 3/5)
  (h2 : Real.sin a + Real.sin b = 1/5) :
  Real.tan ((a + b) / 2) = 1/3 := by
sorry

end NUMINAMATH_CALUDE_tan_half_sum_l4079_407995


namespace NUMINAMATH_CALUDE_dataset_groups_l4079_407999

/-- Calculate the number of groups for a dataset given its maximum value, minimum value, and class interval. -/
def number_of_groups (max_value min_value class_interval : ℕ) : ℕ :=
  (max_value - min_value) / class_interval + 1

/-- Theorem: For a dataset with maximum value 140, minimum value 50, and class interval 10, 
    the number of groups is 10. -/
theorem dataset_groups :
  number_of_groups 140 50 10 = 10 := by
  sorry

end NUMINAMATH_CALUDE_dataset_groups_l4079_407999


namespace NUMINAMATH_CALUDE_smallest_angle_is_22_5_degrees_l4079_407954

def smallest_positive_angle (y : ℝ) : Prop :=
  6 * Real.sin y * (Real.cos y)^3 - 6 * (Real.sin y)^3 * Real.cos y = 3/2 ∧
  y > 0 ∧
  ∀ z, z > 0 → 6 * Real.sin z * (Real.cos z)^3 - 6 * (Real.sin z)^3 * Real.cos z = 3/2 → y ≤ z

theorem smallest_angle_is_22_5_degrees :
  ∃ y, smallest_positive_angle y ∧ y = 22.5 * π / 180 :=
sorry

end NUMINAMATH_CALUDE_smallest_angle_is_22_5_degrees_l4079_407954


namespace NUMINAMATH_CALUDE_train_length_l4079_407905

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 6 → time = 2 → speed * time * (1000 / 3600) = 10 / 3 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l4079_407905


namespace NUMINAMATH_CALUDE_class_size_l4079_407983

theorem class_size (initial_avg : ℝ) (wrong_score : ℝ) (correct_score : ℝ) (new_avg : ℝ) :
  initial_avg = 87.26 →
  wrong_score = 89 →
  correct_score = 98 →
  new_avg = 87.44 →
  ∃ n : ℕ, n = 50 ∧ n * new_avg = n * initial_avg + (correct_score - wrong_score) :=
by sorry

end NUMINAMATH_CALUDE_class_size_l4079_407983


namespace NUMINAMATH_CALUDE_breakable_iff_composite_l4079_407976

def is_breakable (n : ℕ) : Prop :=
  ∃ (a b x y : ℕ), a > 0 ∧ b > 0 ∧ x > 0 ∧ y > 0 ∧ a + b = n ∧ (x : ℚ) / a + (y : ℚ) / b = 1

theorem breakable_iff_composite (n : ℕ) : is_breakable n ↔ ¬ Nat.Prime n ∧ n > 1 :=
sorry

end NUMINAMATH_CALUDE_breakable_iff_composite_l4079_407976


namespace NUMINAMATH_CALUDE_english_test_percentage_l4079_407994

theorem english_test_percentage (math_questions : ℕ) (english_questions : ℕ) 
  (math_percentage : ℚ) (total_correct : ℕ) : 
  math_questions = 40 →
  english_questions = 50 →
  math_percentage = 3/4 →
  total_correct = 79 →
  (total_correct - (math_percentage * math_questions).num) / english_questions = 49/50 := by
sorry

end NUMINAMATH_CALUDE_english_test_percentage_l4079_407994


namespace NUMINAMATH_CALUDE_fifth_term_of_sequence_l4079_407943

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem fifth_term_of_sequence (x y : ℝ) (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_first : a 0 = x + 2*y)
  (h_second : a 1 = x - 2*y)
  (h_third : a 2 = 2*x*y)
  (h_fourth : a 3 = 2*x/y)
  (h_y_neq_half : y ≠ 1/2) :
  a 4 = (-12 - 8*y^2 + 4*y) / (2*y - 1) :=
sorry

end NUMINAMATH_CALUDE_fifth_term_of_sequence_l4079_407943


namespace NUMINAMATH_CALUDE_player_a_wins_l4079_407958

/-- Represents a game state --/
structure GameState :=
  (current : ℕ)

/-- Defines a valid move in the game --/
def validMove (s : GameState) (next : ℕ) : Prop :=
  next > s.current ∧ next ≤ 2 * s.current - 1

/-- Defines the winning condition --/
def isWinningState (s : GameState) : Prop :=
  s.current = 2004

/-- Defines a winning strategy for Player A --/
def hasWinningStrategy (player : ℕ → GameState → Prop) : Prop :=
  ∀ s : GameState, s.current = 2 → 
    ∃ (strategy : GameState → ℕ),
      (∀ s, validMove s (strategy s)) ∧
      (∀ s, player 0 s → isWinningState (GameState.mk (strategy s)) ∨
        (∀ next, validMove (GameState.mk (strategy s)) next → 
          player 1 (GameState.mk next) → 
          player 0 (GameState.mk (strategy (GameState.mk next)))))

/-- The main theorem stating that Player A has a winning strategy --/
theorem player_a_wins : 
  ∃ (player : ℕ → GameState → Prop), hasWinningStrategy player :=
sorry

end NUMINAMATH_CALUDE_player_a_wins_l4079_407958


namespace NUMINAMATH_CALUDE_cyclist_average_speed_l4079_407936

theorem cyclist_average_speed (d1 d2 v1 v2 : ℝ) (h1 : d1 = 9) (h2 : d2 = 11) (h3 : v1 = 11) (h4 : v2 = 9) :
  let t1 := d1 / v1
  let t2 := d2 / v2
  let total_distance := d1 + d2
  let total_time := t1 + t2
  let avg_speed := total_distance / total_time
  ∃ ε > 0, |avg_speed - 9.8| < ε :=
by sorry

end NUMINAMATH_CALUDE_cyclist_average_speed_l4079_407936


namespace NUMINAMATH_CALUDE_jerry_needs_72_dollars_l4079_407919

/-- The amount of money Jerry needs to finish his action figure collection -/
def jerryNeedsMoney (current : ℕ) (total : ℕ) (cost : ℕ) : ℕ :=
  (total - current) * cost

/-- Proof that Jerry needs $72 to finish his collection -/
theorem jerry_needs_72_dollars :
  jerryNeedsMoney 7 16 8 = 72 := by
  sorry

end NUMINAMATH_CALUDE_jerry_needs_72_dollars_l4079_407919


namespace NUMINAMATH_CALUDE_single_elimination_tournament_games_l4079_407968

/-- 
Calculates the number of games required in a single-elimination tournament
to declare a winner, given the number of teams participating.
-/
def gamesRequired (numTeams : ℕ) : ℕ := numTeams - 1

/-- 
Theorem: In a single-elimination tournament with 25 teams and no possibility of ties,
the number of games required to declare a winner is 24.
-/
theorem single_elimination_tournament_games :
  gamesRequired 25 = 24 := by
  sorry

end NUMINAMATH_CALUDE_single_elimination_tournament_games_l4079_407968


namespace NUMINAMATH_CALUDE_fixed_point_exponential_function_l4079_407935

theorem fixed_point_exponential_function (a : ℝ) (ha : a > 0) :
  let f : ℝ → ℝ := λ x ↦ a^(x-1) + 4
  f 1 = 5 := by sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_function_l4079_407935


namespace NUMINAMATH_CALUDE_geometric_series_sum_l4079_407928

/-- Given a geometric series with first term a and common ratio r,
    this function calculates the sum of the first n terms. -/
def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- Theorem: For the geometric series with first term 7/8 and common ratio -1/2,
    the sum of the first four terms is 35/64. -/
theorem geometric_series_sum :
  let a : ℚ := 7/8
  let r : ℚ := -1/2
  let n : ℕ := 4
  geometricSum a r n = 35/64 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l4079_407928


namespace NUMINAMATH_CALUDE_perpendicular_lines_slope_l4079_407911

/-- Given two lines l and l₁ in a 2D coordinate system:
    - l has a slope of -1
    - l₁ passes through points (3,2) and (a,-1)
    - l and l₁ are perpendicular
    Then a = 6 -/
theorem perpendicular_lines_slope (a : ℝ) : 
  let slope_l : ℝ := -1
  let point_A : ℝ × ℝ := (3, 2)
  let point_B : ℝ × ℝ := (a, -1)
  let slope_l₁ : ℝ := (point_B.2 - point_A.2) / (point_B.1 - point_A.1)
  slope_l * slope_l₁ = -1 → a = 6 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_slope_l4079_407911


namespace NUMINAMATH_CALUDE_set_equiv_interval_l4079_407961

-- Define the set S as {x | x ≤ -1}
def S : Set ℝ := {x | x ≤ -1}

-- Define the interval (-∞, -1]
def I : Set ℝ := Set.Iic (-1)

-- Theorem: S is equivalent to I
theorem set_equiv_interval : S = I := by sorry

end NUMINAMATH_CALUDE_set_equiv_interval_l4079_407961


namespace NUMINAMATH_CALUDE_periodic_sequence_properties_l4079_407904

/-- A periodic sequence with period T -/
def is_periodic (a : ℕ → ℕ) (T : ℕ) : Prop :=
  ∀ n, a (n + T) = a n

/-- The smallest period of a sequence -/
def smallest_period (a : ℕ → ℕ) (t : ℕ) : Prop :=
  is_periodic a t ∧ ∀ s, is_periodic a s → t ≤ s

theorem periodic_sequence_properties {a : ℕ → ℕ} {T : ℕ} (h : is_periodic a T) :
  (∃ t, smallest_period a t) ∧ (∀ t, smallest_period a t → T % t = 0) := by
  sorry

end NUMINAMATH_CALUDE_periodic_sequence_properties_l4079_407904


namespace NUMINAMATH_CALUDE_expression_evaluation_l4079_407914

theorem expression_evaluation (a b : ℚ) (h1 : a = 2) (h2 : b = 1/2) :
  (a^3 + b^2)^2 - (a^3 - b^2)^2 = 8 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4079_407914


namespace NUMINAMATH_CALUDE_intersection_line_slope_l4079_407918

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y - 20 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 10*x - 2*y + 40 = 0

-- Define the intersection points
def intersection (C D : ℝ × ℝ) : Prop :=
  circle1 C.1 C.2 ∧ circle1 D.1 D.2 ∧
  circle2 C.1 C.2 ∧ circle2 D.1 D.2 ∧
  C ≠ D

-- Theorem statement
theorem intersection_line_slope (C D : ℝ × ℝ) (h : intersection C D) :
  (D.2 - C.2) / (D.1 - C.1) = -2/3 :=
sorry

end NUMINAMATH_CALUDE_intersection_line_slope_l4079_407918


namespace NUMINAMATH_CALUDE_company_gender_distribution_l4079_407953

theorem company_gender_distribution (total : ℕ) 
  (h1 : total / 3 = total - 2 * total / 3)  -- One-third of workers don't have a retirement plan
  (h2 : (3 * total / 5) / 3 = total / 3 - 2 * total / 5 / 3)  -- 60% of workers without a retirement plan are women
  (h3 : (2 * total / 5) / 3 = total / 3 - 3 * total / 5 / 3)  -- 40% of workers without a retirement plan are men
  (h4 : 4 * (2 * total / 3) / 10 = 2 * total / 3 - 6 * (2 * total / 3) / 10)  -- 40% of workers with a retirement plan are men
  (h5 : 6 * (2 * total / 3) / 10 = 2 * total / 3 - 4 * (2 * total / 3) / 10)  -- 60% of workers with a retirement plan are women
  (h6 : (2 * total / 5) / 3 + 4 * (2 * total / 3) / 10 = 120)  -- There are 120 men in total
  : total - 120 = 180 := by
  sorry

end NUMINAMATH_CALUDE_company_gender_distribution_l4079_407953


namespace NUMINAMATH_CALUDE_remaining_work_days_l4079_407982

/-- Given workers x and y, where x can finish a job in 24 days and y in 16 days,
    prove that x needs 9 days to finish the remaining work after y works for 10 days. -/
theorem remaining_work_days (x_days y_days y_worked_days : ℕ) 
  (hx : x_days = 24)
  (hy : y_days = 16)
  (hw : y_worked_days = 10) :
  (x_days : ℚ) * (1 - y_worked_days / y_days) = 9 := by
  sorry

end NUMINAMATH_CALUDE_remaining_work_days_l4079_407982


namespace NUMINAMATH_CALUDE_min_value_theorem_l4079_407903

theorem min_value_theorem (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a^2 + a*b + a*c + b*c = 4) : 
  ∀ x y z, x > 0 → y > 0 → z > 0 → x^2 + x*y + x*z + y*z = 4 → 2*x + y + z ≥ 2*a + b + c :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l4079_407903


namespace NUMINAMATH_CALUDE_sqrt_of_sqrt_sixteen_l4079_407952

theorem sqrt_of_sqrt_sixteen : Real.sqrt (Real.sqrt 16) = 2 ∨ Real.sqrt (Real.sqrt 16) = -2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_sqrt_sixteen_l4079_407952


namespace NUMINAMATH_CALUDE_intersection_distance_implies_k_range_l4079_407913

/-- Given a line y = kx and a circle (x-2)^2 + (y+1)^2 = 4,
    if the distance between their intersection points is at least 2√3,
    then -4/3 ≤ k ≤ 0 -/
theorem intersection_distance_implies_k_range (k : ℝ) :
  (∃ A B : ℝ × ℝ,
    (A.2 = k * A.1 ∧ B.2 = k * B.1) ∧
    ((A.1 - 2)^2 + (A.2 + 1)^2 = 4 ∧ (B.1 - 2)^2 + (B.2 + 1)^2 = 4) ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 ≥ 12)) →
  -4/3 ≤ k ∧ k ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_implies_k_range_l4079_407913


namespace NUMINAMATH_CALUDE_gcd_13n_plus_4_8n_plus_3_max_9_l4079_407945

theorem gcd_13n_plus_4_8n_plus_3_max_9 :
  (∃ (n : ℕ+), Nat.gcd (13 * n + 4) (8 * n + 3) = 9) ∧
  (∀ (n : ℕ+), Nat.gcd (13 * n + 4) (8 * n + 3) ≤ 9) := by
  sorry

end NUMINAMATH_CALUDE_gcd_13n_plus_4_8n_plus_3_max_9_l4079_407945


namespace NUMINAMATH_CALUDE_solution_in_interval_l4079_407977

open Real

theorem solution_in_interval : ∃ (x₀ : ℝ), ∃ (k : ℤ),
  (log x₀ = 5 - 2 * x₀) ∧ 
  (x₀ > k) ∧ (x₀ < k + 1) ∧
  (k = 2) := by
  sorry

end NUMINAMATH_CALUDE_solution_in_interval_l4079_407977


namespace NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l4079_407912

def geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, ∀ n : ℕ, a (n + 1) = a n * r

theorem sixth_term_of_geometric_sequence
  (a : ℕ → ℚ)
  (h_geom : geometric_sequence a)
  (h_first : a 1 = 27)
  (h_fourth : a 4 = a 3 * a 5) :
  a 6 = 1 / 9 := by
sorry

end NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l4079_407912


namespace NUMINAMATH_CALUDE_circle_integers_l4079_407939

/-- Given n integers equally spaced around a circle, if the diameter through 7 also goes through 23, then n = 32 -/
theorem circle_integers (n : ℕ) : n ≥ 23 → (∃ (k : ℕ), n = 2 * k ∧ (23 - 7) * 2 + 2 = n) → n = 32 := by
  sorry

end NUMINAMATH_CALUDE_circle_integers_l4079_407939


namespace NUMINAMATH_CALUDE_complex_product_quadrant_l4079_407902

theorem complex_product_quadrant : 
  let z : ℂ := (-1 + 2*I) * (3 - I)
  (z.re < 0 ∧ z.im > 0) := by sorry

end NUMINAMATH_CALUDE_complex_product_quadrant_l4079_407902
