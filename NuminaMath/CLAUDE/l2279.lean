import Mathlib

namespace NUMINAMATH_CALUDE_tan_function_property_l2279_227916

theorem tan_function_property (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∀ x, a * Real.tan (b * x) = a * Real.tan (b * (x + π / 2))) → 
  a * Real.tan (b * π / 8) = 4 → 
  a * b = 8 := by sorry

end NUMINAMATH_CALUDE_tan_function_property_l2279_227916


namespace NUMINAMATH_CALUDE_problem_solution_l2279_227926

theorem problem_solution : ∃ n : ℕ, 2^13 - 2^11 = 3 * n ∧ n = 2048 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2279_227926


namespace NUMINAMATH_CALUDE_centroid_distance_theorem_l2279_227998

/-- A line in a plane --/
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

/-- A point in a plane --/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A triangle defined by three points --/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Distance from a point to a line --/
def distanceToLine (p : Point) (l : Line) : ℝ :=
  sorry

/-- Centroid of a triangle --/
def centroid (t : Triangle) : Point :=
  sorry

/-- Theorem: The distance from the centroid to a line equals the average of distances from vertices to the line --/
theorem centroid_distance_theorem (t : Triangle) (l : Line) :
  distanceToLine (centroid t) l = (distanceToLine t.A l + distanceToLine t.B l + distanceToLine t.C l) / 3 :=
by sorry

end NUMINAMATH_CALUDE_centroid_distance_theorem_l2279_227998


namespace NUMINAMATH_CALUDE_all_triplets_sum_to_two_l2279_227990

theorem all_triplets_sum_to_two :
  (3/4 + 1/4 + 1 = 2) ∧
  (1.2 + 0.8 + 0 = 2) ∧
  (0.5 + 1.0 + 0.5 = 2) ∧
  (3/5 + 4/5 + 3/5 = 2) ∧
  (2 + (-3) + 3 = 2) := by
  sorry

end NUMINAMATH_CALUDE_all_triplets_sum_to_two_l2279_227990


namespace NUMINAMATH_CALUDE_quadratic_polynomial_value_l2279_227989

/-- A quadratic polynomial -/
def QuadraticPolynomial (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- Divisibility condition for the polynomial -/
def DivisibilityCondition (q : ℝ → ℝ) : Prop :=
  ∃ p : ℝ → ℝ, ∀ x : ℝ, q x^3 + x = p x * (x - 2) * (x + 2) * (x - 5)

theorem quadratic_polynomial_value (a b c : ℝ) :
  let q := QuadraticPolynomial a b c
  DivisibilityCondition q → q 10 = -139 * Real.rpow 2 (1/3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_value_l2279_227989


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_one_fourth_l2279_227992

theorem arithmetic_square_root_of_one_fourth : Real.sqrt (1 / 4) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_one_fourth_l2279_227992


namespace NUMINAMATH_CALUDE_trajectory_is_parabola_l2279_227961

-- Define the fixed point F
def F : ℝ × ℝ := (1, 0)

-- Define the directrix line
def directrix : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = -1}

-- Define the property of the moving circle
def circle_property (center : ℝ × ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ 
  (center.1 - F.1)^2 + (center.2 - F.2)^2 = r^2 ∧
  ∃ (p : ℝ × ℝ), p ∈ directrix ∧ (center.1 - p.1)^2 + (center.2 - p.2)^2 = r^2

-- Theorem statement
theorem trajectory_is_parabola :
  ∀ (center : ℝ × ℝ), circle_property center → center.2^2 = 4 * center.1 :=
sorry

end NUMINAMATH_CALUDE_trajectory_is_parabola_l2279_227961


namespace NUMINAMATH_CALUDE_triangle_equilateral_from_sequences_l2279_227920

/-- A triangle with sides a, b, c opposite to angles A, B, C respectively is equilateral
if its angles form an arithmetic sequence and its sides form a geometric sequence. -/
theorem triangle_equilateral_from_sequences (a b c A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- positive side lengths
  0 < A ∧ 0 < B ∧ 0 < C →  -- positive angles
  A + B + C = π →  -- sum of angles in a triangle
  2 * B = A + C →  -- angles form arithmetic sequence
  b^2 = a * c →  -- sides form geometric sequence
  A = B ∧ B = C ∧ a = b ∧ b = c :=
by sorry

end NUMINAMATH_CALUDE_triangle_equilateral_from_sequences_l2279_227920


namespace NUMINAMATH_CALUDE_expected_occurrences_100_rolls_l2279_227948

/-- The expected number of times a specific face appears when rolling a fair die multiple times -/
def expected_occurrences (num_rolls : ℕ) : ℚ :=
  num_rolls * (1 : ℚ) / 6

/-- Theorem: The expected number of times a specific face appears when rolling a fair die 100 times is 50/3 -/
theorem expected_occurrences_100_rolls :
  expected_occurrences 100 = 50 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expected_occurrences_100_rolls_l2279_227948


namespace NUMINAMATH_CALUDE_greatest_x_value_l2279_227942

theorem greatest_x_value (x : ℝ) : 
  (2 * x^2 + 7 * x + 3 = 5) → x ≤ (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_greatest_x_value_l2279_227942


namespace NUMINAMATH_CALUDE_modulus_of_fraction_l2279_227903

def z : ℂ := -1 + Complex.I

theorem modulus_of_fraction : Complex.abs ((z + 3) / (z + 2)) = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_fraction_l2279_227903


namespace NUMINAMATH_CALUDE_negation_equivalence_l2279_227951

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x > 0 ∧ x^2 - x ≤ 0) ↔ (∀ x : ℝ, x ≤ 0 → x^2 - x > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2279_227951


namespace NUMINAMATH_CALUDE_train_crossing_time_l2279_227996

/-- The time taken for a train to cross a stationary point -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) : 
  train_length = 150 →
  train_speed_kmh = 180 →
  (train_length / (train_speed_kmh * 1000 / 3600)) = 3 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l2279_227996


namespace NUMINAMATH_CALUDE_river_width_l2279_227945

/-- Given a river with the following properties:
  * The river is 4 meters deep
  * The river flows at a rate of 6 kilometers per hour
  * The volume of water flowing into the sea is 26000 cubic meters per minute
  Prove that the width of the river is 65 meters. -/
theorem river_width (depth : ℝ) (flow_rate : ℝ) (volume : ℝ) :
  depth = 4 →
  flow_rate = 6 →
  volume = 26000 →
  (volume / (depth * (flow_rate * 1000 / 60))) = 65 := by
  sorry

end NUMINAMATH_CALUDE_river_width_l2279_227945


namespace NUMINAMATH_CALUDE_parabola_intersection_condition_l2279_227979

theorem parabola_intersection_condition (k : ℝ) : 
  (∃! x : ℝ, -2 = x^2 + k*x - 1) → (k = 2 ∨ k = -2) := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_condition_l2279_227979


namespace NUMINAMATH_CALUDE_average_marks_math_chem_l2279_227941

theorem average_marks_math_chem (math physics chem : ℕ) : 
  math + physics = 60 →
  chem = physics + 20 →
  (math + chem) / 2 = 40 :=
by sorry

end NUMINAMATH_CALUDE_average_marks_math_chem_l2279_227941


namespace NUMINAMATH_CALUDE_max_probability_divisible_by_10_min_nonzero_probability_divisible_by_10_l2279_227928

/-- A segment of natural numbers -/
structure Segment where
  start : ℕ
  length : ℕ

/-- The probability of a number in a segment being divisible by 10 -/
def probability_divisible_by_10 (s : Segment) : ℚ :=
  (s.length.div 10) / s.length

theorem max_probability_divisible_by_10 :
  ∃ s : Segment, probability_divisible_by_10 s = 1 ∧
  ∀ t : Segment, probability_divisible_by_10 t ≤ 1 :=
sorry

theorem min_nonzero_probability_divisible_by_10 :
  ∃ s : Segment, probability_divisible_by_10 s = 1/19 ∧
  ∀ t : Segment, probability_divisible_by_10 t = 0 ∨ probability_divisible_by_10 t ≥ 1/19 :=
sorry

end NUMINAMATH_CALUDE_max_probability_divisible_by_10_min_nonzero_probability_divisible_by_10_l2279_227928


namespace NUMINAMATH_CALUDE_line_segment_length_l2279_227969

/-- The length of a line segment with endpoints (1,4) and (8,16) is √193. -/
theorem line_segment_length : Real.sqrt 193 = Real.sqrt ((8 - 1)^2 + (16 - 4)^2) := by
  sorry

end NUMINAMATH_CALUDE_line_segment_length_l2279_227969


namespace NUMINAMATH_CALUDE_cylinder_volume_l2279_227905

/-- Given a cylinder with lateral surface area 100π cm² and an inscribed rectangular solid
    with diagonal 10√2 cm, prove that the volume of the cylinder is 250π cm³. -/
theorem cylinder_volume (r h : ℝ) : 
  r > 0 → h > 0 →
  2 * Real.pi * r * h = 100 * Real.pi →
  4 * r^2 + h^2 = 200 →
  Real.pi * r^2 * h = 250 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_l2279_227905


namespace NUMINAMATH_CALUDE_binomial_coefficient_sum_l2279_227959

theorem binomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℤ) : 
  (∀ x : ℤ, (2 - x)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| = 665 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_sum_l2279_227959


namespace NUMINAMATH_CALUDE_min_sum_of_product_144_l2279_227950

theorem min_sum_of_product_144 (a b : ℤ) (h : a * b = 144) :
  ∀ (x y : ℤ), x * y = 144 → a + b ≤ x + y ∧ ∃ (a₀ b₀ : ℤ), a₀ * b₀ = 144 ∧ a₀ + b₀ = -145 :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_product_144_l2279_227950


namespace NUMINAMATH_CALUDE_product_difference_bound_l2279_227985

theorem product_difference_bound (n : ℕ+) (a b : ℕ+) 
  (h : (a : ℝ) * b = (n : ℝ)^2 + n + 1) : 
  |((a : ℝ) - b)| ≥ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_product_difference_bound_l2279_227985


namespace NUMINAMATH_CALUDE_marys_green_beans_weight_l2279_227944

/-- Proves that the weight of green beans is 4 pounds given the conditions of Mary's grocery shopping. -/
theorem marys_green_beans_weight (bag_capacity : ℝ) (milk_weight : ℝ) (remaining_space : ℝ) :
  bag_capacity = 20 →
  milk_weight = 6 →
  remaining_space = 2 →
  ∃ (green_beans_weight : ℝ),
    green_beans_weight + milk_weight + 2 * green_beans_weight = bag_capacity - remaining_space ∧
    green_beans_weight = 4 := by
  sorry

end NUMINAMATH_CALUDE_marys_green_beans_weight_l2279_227944


namespace NUMINAMATH_CALUDE_minor_premise_incorrect_l2279_227931

theorem minor_premise_incorrect : ¬ ∀ x : ℝ, x + 1/x ≥ 2 * Real.sqrt (x * (1/x)) := by
  sorry

end NUMINAMATH_CALUDE_minor_premise_incorrect_l2279_227931


namespace NUMINAMATH_CALUDE_stamp_sale_difference_l2279_227901

def red_stamps : ℕ := 30
def white_stamps : ℕ := 80
def red_stamp_price : ℚ := 50 / 100
def white_stamp_price : ℚ := 20 / 100

theorem stamp_sale_difference :
  white_stamps * white_stamp_price - red_stamps * red_stamp_price = 1 := by sorry

end NUMINAMATH_CALUDE_stamp_sale_difference_l2279_227901


namespace NUMINAMATH_CALUDE_sqrt_four_equals_plus_minus_two_l2279_227932

theorem sqrt_four_equals_plus_minus_two : ∀ (x : ℝ), x^2 = 4 → x = 2 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_four_equals_plus_minus_two_l2279_227932


namespace NUMINAMATH_CALUDE_f_monotone_decreasing_range_l2279_227921

/-- A piecewise function f dependent on parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3*a - 2)*x + 6*a - 1 else a^x

/-- Theorem stating the range of a for which f is monotonically decreasing -/
theorem f_monotone_decreasing_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a y < f a x) → 3/8 ≤ a ∧ a < 2/3 :=
by sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_range_l2279_227921


namespace NUMINAMATH_CALUDE_axis_of_symmetry_l2279_227983

-- Define the function f with the given property
def f : ℝ → ℝ := sorry

-- Define the property that f(x) = f(6-x) for all x
axiom f_symmetry (x : ℝ) : f x = f (6 - x)

-- State the theorem: x = 3 is the axis of symmetry
theorem axis_of_symmetry :
  ∀ (x y : ℝ), f x = y ↔ f (6 - x) = y :=
by sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_l2279_227983


namespace NUMINAMATH_CALUDE_factory_weekly_production_l2279_227984

/-- Represents the production of toys in a factory --/
structure ToyProduction where
  days_per_week : ℕ
  toys_per_day : ℕ
  constant_daily_production : Bool

/-- Calculates the weekly toy production --/
def weekly_production (tp : ToyProduction) : ℕ :=
  tp.days_per_week * tp.toys_per_day

/-- Theorem stating the weekly toy production for the given factory --/
theorem factory_weekly_production :
  ∀ (tp : ToyProduction),
    tp.days_per_week = 4 →
    tp.toys_per_day = 1500 →
    tp.constant_daily_production →
    weekly_production tp = 6000 := by
  sorry

end NUMINAMATH_CALUDE_factory_weekly_production_l2279_227984


namespace NUMINAMATH_CALUDE_power_zero_eq_one_l2279_227957

theorem power_zero_eq_one (a b : ℝ) (h : a - b ≠ 0) : (a - b)^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_zero_eq_one_l2279_227957


namespace NUMINAMATH_CALUDE_smallest_with_16_divisors_l2279_227930

/-- The number of positive divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- Checks if a number has exactly 16 positive divisors -/
def has_16_divisors (n : ℕ+) : Prop := num_divisors n = 16

theorem smallest_with_16_divisors :
  ∀ n : ℕ+, has_16_divisors n → n ≥ 120 ∧ has_16_divisors 120 := by sorry

end NUMINAMATH_CALUDE_smallest_with_16_divisors_l2279_227930


namespace NUMINAMATH_CALUDE_negation_unique_solution_equivalence_l2279_227958

theorem negation_unique_solution_equivalence :
  ¬(∀ a : ℝ, ∃! x : ℝ, a * x + 1 = 0) ↔
  (∃ a : ℝ, (∃ x y : ℝ, x ≠ y ∧ a * x + 1 = 0 ∧ a * y + 1 = 0) ∨ (∀ x : ℝ, a * x + 1 ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_negation_unique_solution_equivalence_l2279_227958


namespace NUMINAMATH_CALUDE_intersection_limit_l2279_227923

-- Define the parabola function
def f (x : ℝ) : ℝ := x^2 - 4

-- Define the horizontal line function
def g (m : ℝ) (x : ℝ) : ℝ := m

-- Define L(m) as the x-coordinate of the left endpoint of intersection
noncomputable def L (m : ℝ) : ℝ := -Real.sqrt (m + 4)

-- Define r as a function of m
noncomputable def r (m : ℝ) : ℝ := (L (-m) - L m) / m

-- Theorem statement
theorem intersection_limit :
  ∀ ε > 0, ∃ δ > 0, ∀ m : ℝ, 0 < |m| ∧ |m| < δ ∧ -4 < m ∧ m < 4 →
    |r m - (1/2)| < ε :=
sorry

end NUMINAMATH_CALUDE_intersection_limit_l2279_227923


namespace NUMINAMATH_CALUDE_cos_two_alpha_plus_beta_l2279_227943

theorem cos_two_alpha_plus_beta
  (α β : ℝ)
  (h1 : 3 * (Real.sin α)^2 + 2 * (Real.sin β)^2 = 1)
  (h2 : 3 * (Real.sin α + Real.cos α)^2 - 2 * (Real.sin β + Real.cos β)^2 = 1) :
  Real.cos (2 * (α + β)) = -1/3 :=
by sorry

end NUMINAMATH_CALUDE_cos_two_alpha_plus_beta_l2279_227943


namespace NUMINAMATH_CALUDE_arrange_seven_books_three_identical_l2279_227918

/-- The number of ways to arrange books with some identical copies -/
def arrange_books (total : ℕ) (identical : ℕ) : ℕ :=
  Nat.factorial total / Nat.factorial identical

/-- Theorem: Arranging 7 books with 3 identical copies yields 840 possibilities -/
theorem arrange_seven_books_three_identical :
  arrange_books 7 3 = 840 := by
  sorry

end NUMINAMATH_CALUDE_arrange_seven_books_three_identical_l2279_227918


namespace NUMINAMATH_CALUDE_fifteenth_digit_sum_one_ninth_one_eleventh_l2279_227906

/-- The decimal representation of a rational number -/
def decimalRepresentation (q : ℚ) : ℕ → ℕ := sorry

/-- The sum of decimal representations of two rational numbers -/
def sumDecimalRepresentations (q₁ q₂ : ℚ) : ℕ → ℕ := sorry

/-- The nth digit after the decimal point in a decimal representation -/
def nthDigitAfterDecimal (rep : ℕ → ℕ) (n : ℕ) : ℕ := sorry

theorem fifteenth_digit_sum_one_ninth_one_eleventh :
  nthDigitAfterDecimal (sumDecimalRepresentations (1/9) (1/11)) 15 = 1 := by sorry

end NUMINAMATH_CALUDE_fifteenth_digit_sum_one_ninth_one_eleventh_l2279_227906


namespace NUMINAMATH_CALUDE_arctan_sum_three_four_l2279_227929

theorem arctan_sum_three_four : Real.arctan (3/4) + Real.arctan (4/3) = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_three_four_l2279_227929


namespace NUMINAMATH_CALUDE_pet_shop_kittens_l2279_227995

theorem pet_shop_kittens (total : ℕ) (hamsters : ℕ) (birds : ℕ) (kittens : ℕ) : 
  total = 77 → hamsters = 15 → birds = 30 → kittens = total - hamsters - birds → kittens = 32 := by
  sorry

end NUMINAMATH_CALUDE_pet_shop_kittens_l2279_227995


namespace NUMINAMATH_CALUDE_smallest_angle_in_345_ratio_triangle_l2279_227971

theorem smallest_angle_in_345_ratio_triangle (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a + b + c = 180 →
  b = (4/3) * a →
  c = (5/3) * a →
  a = 45 := by
sorry

end NUMINAMATH_CALUDE_smallest_angle_in_345_ratio_triangle_l2279_227971


namespace NUMINAMATH_CALUDE_tangent_line_length_range_l2279_227913

-- Define the circles
def circle_C1 (x y α : ℝ) : Prop := (x + Real.cos α)^2 + (y + Real.sin α)^2 = 4
def circle_C2 (x y β : ℝ) : Prop := (x - 5 * Real.sin β)^2 + (y - 5 * Real.cos β)^2 = 1

-- Define the range of α and β
def angle_range (θ : ℝ) : Prop := 0 ≤ θ ∧ θ < 2 * Real.pi

-- Define the tangent line MN
def tangent_line (M N : ℝ × ℝ) (α β : ℝ) : Prop :=
  circle_C1 M.1 M.2 α ∧ circle_C2 N.1 N.2 β ∧
  ∃ (t : ℝ), N = (M.1 + t * (N.1 - M.1), M.2 + t * (N.2 - M.2))

-- State the theorem
theorem tangent_line_length_range :
  ∀ (M N : ℝ × ℝ) (α β : ℝ),
  angle_range α → angle_range β → tangent_line M N α β →
  2 * Real.sqrt 2 ≤ Real.sqrt ((N.1 - M.1)^2 + (N.2 - M.2)^2) ∧
  Real.sqrt ((N.1 - M.1)^2 + (N.2 - M.2)^2) ≤ 3 * Real.sqrt 7 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_length_range_l2279_227913


namespace NUMINAMATH_CALUDE_student_correct_answers_l2279_227954

theorem student_correct_answers 
  (total_questions : ℕ) 
  (score : ℤ) 
  (correct_answers : ℕ) 
  (incorrect_answers : ℕ) :
  total_questions = 100 →
  score = correct_answers - 2 * incorrect_answers →
  correct_answers + incorrect_answers = total_questions →
  score = 70 →
  correct_answers = 90 := by
sorry

end NUMINAMATH_CALUDE_student_correct_answers_l2279_227954


namespace NUMINAMATH_CALUDE_bcm_hens_count_l2279_227986

theorem bcm_hens_count (total_chickens : ℕ) (bcm_percentage : ℚ) (bcm_hen_percentage : ℚ) 
  (h1 : total_chickens = 100)
  (h2 : bcm_percentage = 1/5)
  (h3 : bcm_hen_percentage = 4/5) :
  ↑(total_chickens : ℚ) * bcm_percentage * bcm_hen_percentage = 16 := by
  sorry

end NUMINAMATH_CALUDE_bcm_hens_count_l2279_227986


namespace NUMINAMATH_CALUDE_g_neg_two_l2279_227970

def g (x : ℝ) : ℝ := x^3 - 2*x + 1

theorem g_neg_two : g (-2) = -3 := by
  sorry

end NUMINAMATH_CALUDE_g_neg_two_l2279_227970


namespace NUMINAMATH_CALUDE_solution_implies_m_equals_three_l2279_227936

theorem solution_implies_m_equals_three (x y m : ℝ) : 
  x = -2 → y = 1 → m * x + 5 * y = -1 → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_m_equals_three_l2279_227936


namespace NUMINAMATH_CALUDE_f_properties_l2279_227937

noncomputable def f (x : ℝ) : ℝ := 2^(Real.sin x) + 2^(-Real.sin x)

theorem f_properties :
  -- f is an even function
  (∀ x, f (-x) = f x) ∧
  -- π is a period of f
  (∀ x, f (x + Real.pi) = f x) ∧
  -- π is a local minimum of f
  (∃ ε > 0, ∀ x, x ∈ Set.Ioo (Real.pi - ε) (Real.pi + ε) → f Real.pi ≤ f x) ∧
  -- f is strictly increasing on (0, π/2)
  (∀ x y, x ∈ Set.Ioo 0 (Real.pi / 2) → y ∈ Set.Ioo 0 (Real.pi / 2) → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2279_227937


namespace NUMINAMATH_CALUDE_distance_from_negative_one_l2279_227953

theorem distance_from_negative_one : 
  {x : ℝ | |x - (-1)| = 5} = {4, -6} := by
sorry

end NUMINAMATH_CALUDE_distance_from_negative_one_l2279_227953


namespace NUMINAMATH_CALUDE_apple_difference_l2279_227976

/-- The number of apples Jackie has -/
def jackie_apples : ℕ := 125

/-- The number of apples Adam has -/
def adam_apples : ℕ := 98

/-- The number of apples Laura has -/
def laura_apples : ℕ := 173

/-- The difference between Laura's apples and the sum of Jackie's and Adam's apples -/
theorem apple_difference : Int.ofNat laura_apples - Int.ofNat (jackie_apples + adam_apples) = -50 := by
  sorry

end NUMINAMATH_CALUDE_apple_difference_l2279_227976


namespace NUMINAMATH_CALUDE_toys_remaining_l2279_227904

theorem toys_remaining (initial_stock : ℕ) (sold_week1 : ℕ) (sold_week2 : ℕ) 
  (h1 : initial_stock = 83) 
  (h2 : sold_week1 = 38) 
  (h3 : sold_week2 = 26) :
  initial_stock - (sold_week1 + sold_week2) = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_toys_remaining_l2279_227904


namespace NUMINAMATH_CALUDE_quadratic_rewrite_sum_l2279_227956

theorem quadratic_rewrite_sum (a b c : ℝ) :
  (∀ x, 6 * x^2 + 36 * x + 216 = a * (x + b)^2 + c) →
  a + b + c = 171 := by
sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_sum_l2279_227956


namespace NUMINAMATH_CALUDE_peach_apple_pear_pricing_l2279_227949

theorem peach_apple_pear_pricing (x y z : ℝ) 
  (h1 : 7 * x = y + 2 * z)
  (h2 : 7 * y = 10 * z + x) :
  12 * y = 18 * z := by sorry

end NUMINAMATH_CALUDE_peach_apple_pear_pricing_l2279_227949


namespace NUMINAMATH_CALUDE_no_division_for_all_n_l2279_227999

theorem no_division_for_all_n : ∀ n : ℕ, Nat.gcd (n + 2) (n^3 - 2*n^2 - 5*n + 7) = 1 := by
  sorry

end NUMINAMATH_CALUDE_no_division_for_all_n_l2279_227999


namespace NUMINAMATH_CALUDE_students_supporting_one_issue_l2279_227910

theorem students_supporting_one_issue 
  (total_students : ℕ) 
  (first_issue : ℕ) 
  (second_issue : ℕ) 
  (opposing_all : ℕ) 
  (supporting_both : ℕ) 
  (h1 : total_students = 256)
  (h2 : first_issue = 185)
  (h3 : second_issue = 142)
  (h4 : opposing_all = 37)
  (h5 : supporting_both = 105) : 
  first_issue + second_issue - 2 * supporting_both = 117 :=
by sorry

end NUMINAMATH_CALUDE_students_supporting_one_issue_l2279_227910


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2279_227917

theorem arithmetic_calculation : 15 * (1/3) + 45 * (2/3) = 35 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2279_227917


namespace NUMINAMATH_CALUDE_right_triangle_inequality_l2279_227972

theorem right_triangle_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a ≤ b) (hbc : b < c) (hright : a^2 + b^2 = c^2) :
  a^2 * (b + c) + b^2 * (c + a) + c^2 * (a + b) ≥ (2 + 3 * Real.sqrt 2) * a * b * c :=
sorry

end NUMINAMATH_CALUDE_right_triangle_inequality_l2279_227972


namespace NUMINAMATH_CALUDE_correct_assignment_properties_l2279_227939

-- Define the properties of assignment statements
inductive AssignmentProperty : Type
  | InitialValue : AssignmentProperty
  | AssignExpression : AssignmentProperty
  | MultipleAssignments : AssignmentProperty
  | NoMultipleAssignments : AssignmentProperty

-- Define a function to check if a property is correct
def isCorrectProperty (prop : AssignmentProperty) : Prop :=
  match prop with
  | AssignmentProperty.InitialValue => True
  | AssignmentProperty.AssignExpression => True
  | AssignmentProperty.MultipleAssignments => True
  | AssignmentProperty.NoMultipleAssignments => False

-- Theorem stating the correct properties of assignment statements
theorem correct_assignment_properties :
  ∀ (prop : AssignmentProperty),
    isCorrectProperty prop ↔
      (prop = AssignmentProperty.InitialValue ∨
       prop = AssignmentProperty.AssignExpression ∨
       prop = AssignmentProperty.MultipleAssignments) :=
by sorry

end NUMINAMATH_CALUDE_correct_assignment_properties_l2279_227939


namespace NUMINAMATH_CALUDE_difference_of_fractions_l2279_227987

/-- Proves that the difference between 1/10 of 8000 and 1/20% of 8000 is equal to 796 -/
theorem difference_of_fractions : 
  (8000 / 10) - (8000 * (1 / 20) / 100) = 796 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_fractions_l2279_227987


namespace NUMINAMATH_CALUDE_monotonicity_intervals_k_range_l2279_227935

/-- The function f(x) = xe^(kx) -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := x * Real.exp (k * x)

/-- Monotonicity intervals for f(x) when k > 0 -/
theorem monotonicity_intervals (k : ℝ) (h : k > 0) :
  (∀ x₁ x₂, x₁ < x₂ ∧ - 1 / k < x₁ → f k x₁ < f k x₂) ∧
  (∀ x₁ x₂, x₁ < x₂ ∧ x₂ < - 1 / k → f k x₁ > f k x₂) :=
sorry

/-- Range of k when f(x) is monotonically increasing in (-1, 1) -/
theorem k_range (k : ℝ) (h : k ≠ 0) :
  (∀ x₁ x₂, -1 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f k x₁ < f k x₂) →
  (k ∈ Set.Icc (-1) 0 ∪ Set.Ioc 0 1) :=
sorry

end NUMINAMATH_CALUDE_monotonicity_intervals_k_range_l2279_227935


namespace NUMINAMATH_CALUDE_conference_arrangements_l2279_227900

/-- The number of lecturers at the conference -/
def total_lecturers : ℕ := 8

/-- The number of lecturers with specific ordering requirements -/
def ordered_lecturers : ℕ := 3

/-- Calculate the number of permutations for the remaining lecturers -/
def remaining_permutations : ℕ := (total_lecturers - ordered_lecturers).factorial

/-- Calculate the number of ways to arrange the ordered lecturers -/
def ordered_arrangements : ℕ := (total_lecturers - 2) * (total_lecturers - 1) * total_lecturers

/-- The total number of possible arrangements -/
def total_arrangements : ℕ := ordered_arrangements * remaining_permutations

theorem conference_arrangements :
  total_arrangements = 40320 := by sorry

end NUMINAMATH_CALUDE_conference_arrangements_l2279_227900


namespace NUMINAMATH_CALUDE_remainder_problem_l2279_227963

theorem remainder_problem (n : ℕ) (h1 : n % 13 = 11) (h2 : n = 349) : n % 17 = 9 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2279_227963


namespace NUMINAMATH_CALUDE_triangle_count_is_55_l2279_227965

/-- The number of distinct triangles formed from 12 points on a circle's circumference,
    where one specific point is always a vertex. -/
def num_triangles (total_points : ℕ) (fixed_points : ℕ) : ℕ :=
  Nat.choose (total_points - fixed_points) (2 : ℕ)

/-- Theorem stating that the number of triangles formed under the given conditions is 55. -/
theorem triangle_count_is_55 : num_triangles 12 1 = 55 := by
  sorry

end NUMINAMATH_CALUDE_triangle_count_is_55_l2279_227965


namespace NUMINAMATH_CALUDE_pieces_remaining_bound_l2279_227966

/-- Represents a 2n × 2n board with black and white pieces -/
structure Board (n : ℕ) where
  black_pieces : Finset (ℕ × ℕ)
  white_pieces : Finset (ℕ × ℕ)
  valid_board : ∀ (x y : ℕ), (x, y) ∈ black_pieces ∪ white_pieces → x < 2*n ∧ y < 2*n

/-- Removes black pieces on the same vertical line as white pieces -/
def remove_black (board : Board n) : Board n := sorry

/-- Removes white pieces on the same horizontal line as remaining black pieces -/
def remove_white (board : Board n) : Board n := sorry

/-- The final state of the board after removals -/
def final_board (board : Board n) : Board n := remove_white (remove_black board)

theorem pieces_remaining_bound (n : ℕ) (board : Board n) :
  (final_board board).black_pieces.card ≤ n^2 ∨ (final_board board).white_pieces.card ≤ n^2 := by
  sorry

end NUMINAMATH_CALUDE_pieces_remaining_bound_l2279_227966


namespace NUMINAMATH_CALUDE_train_speed_in_kmh_l2279_227902

-- Define the length of the train in meters
def train_length : ℝ := 280

-- Define the time taken to cross the pole in seconds
def crossing_time : ℝ := 20

-- Define the conversion factor from m/s to km/h
def ms_to_kmh : ℝ := 3.6

-- Theorem to prove
theorem train_speed_in_kmh :
  (train_length / crossing_time) * ms_to_kmh = 50.4 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_in_kmh_l2279_227902


namespace NUMINAMATH_CALUDE_jaya_rank_from_bottom_l2279_227968

theorem jaya_rank_from_bottom (total_students : ℕ) (rank_from_top : ℕ) (rank_from_bottom : ℕ) : 
  total_students = 53 → 
  rank_from_top = 5 → 
  rank_from_bottom = total_students - rank_from_top + 1 →
  rank_from_bottom = 50 := by
sorry

end NUMINAMATH_CALUDE_jaya_rank_from_bottom_l2279_227968


namespace NUMINAMATH_CALUDE_jean_friday_calls_l2279_227988

/-- The number of calls Jean answered on each day of the week --/
structure WeekCalls where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- The average number of calls per day --/
def average_calls : ℕ := 40

/-- The number of working days in a week --/
def working_days : ℕ := 5

/-- Jean's call data for the week --/
def jean_calls : WeekCalls := {
  monday := 35,
  tuesday := 46,
  wednesday := 27,
  thursday := 61,
  friday := 31  -- This is what we want to prove
}

/-- Theorem stating that Jean answered 31 calls on Friday --/
theorem jean_friday_calls : 
  jean_calls.friday = 31 :=
by sorry

end NUMINAMATH_CALUDE_jean_friday_calls_l2279_227988


namespace NUMINAMATH_CALUDE_find_C_l2279_227973

theorem find_C (A B C : ℕ) : A = 680 → A = B + 157 → B = C + 185 → C = 338 := by
  sorry

end NUMINAMATH_CALUDE_find_C_l2279_227973


namespace NUMINAMATH_CALUDE_max_ab_is_nine_l2279_227982

/-- The function f(x) defined in the problem -/
def f (a b : ℝ) (x : ℝ) : ℝ := 4 * x^3 - a * x^2 - 2 * b * x + 2

/-- The derivative of f(x) -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 12 * x^2 - 2 * a * x - 2 * b

/-- The second derivative of f(x) -/
def f'' (a : ℝ) (x : ℝ) : ℝ := 24 * x - 2 * a

theorem max_ab_is_nine (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_extremum : f' a b 1 = 0) : 
  (∃ (max_ab : ℝ), max_ab = 9 ∧ ∀ (a' b' : ℝ), a' > 0 → b' > 0 → f' a' b' 1 = 0 → a' * b' ≤ max_ab) :=
sorry

end NUMINAMATH_CALUDE_max_ab_is_nine_l2279_227982


namespace NUMINAMATH_CALUDE_expression_evaluation_l2279_227912

theorem expression_evaluation (x : ℝ) : 
  x * (x * (x * (3 - x) - 5) + 12) + 2 = -x^4 + 3*x^3 - 5*x^2 + 12*x + 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2279_227912


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_for_ellipse_l2279_227993

/-- Predicate to determine if an equation represents an ellipse -/
def is_ellipse (a b : ℝ) : Prop := sorry

/-- Theorem stating that a > 0 and b > 0 is a necessary but not sufficient condition for ax^2 + by^2 = 1 to represent an ellipse -/
theorem necessary_not_sufficient_for_ellipse :
  (∀ a b : ℝ, is_ellipse a b → a > 0 ∧ b > 0) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ ¬is_ellipse a b) :=
sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_for_ellipse_l2279_227993


namespace NUMINAMATH_CALUDE_inequality_proof_l2279_227947

theorem inequality_proof (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hne : ¬(x = y ∧ y = z)) : 
  (x + y) * (y + z) * (z + x) > 8 * x * y * z := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2279_227947


namespace NUMINAMATH_CALUDE_cuboid_height_l2279_227977

/-- Represents a cuboid with given dimensions -/
structure Cuboid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the sum of all edges of a cuboid -/
def Cuboid.sumOfEdges (c : Cuboid) : ℝ :=
  4 * (c.length + c.width + c.height)

theorem cuboid_height (c : Cuboid) 
  (h_sum : c.sumOfEdges = 224)
  (h_width : c.width = 30)
  (h_length : c.length = 22) :
  c.height = 4 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_height_l2279_227977


namespace NUMINAMATH_CALUDE_age_difference_l2279_227934

theorem age_difference (a b : ℕ) (h1 : b = 38) (h2 : a + 10 = 2 * (b - 10)) : a - b = 8 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l2279_227934


namespace NUMINAMATH_CALUDE_unique_cube_difference_l2279_227952

theorem unique_cube_difference (m n : ℕ+) : 
  (∃ k : ℕ+, 2^n.val - 13^m.val = k^3) ↔ m = 2 ∧ n = 9 := by
sorry

end NUMINAMATH_CALUDE_unique_cube_difference_l2279_227952


namespace NUMINAMATH_CALUDE_impossibility_of_transformation_l2279_227980

/-- Represents a four-digit number --/
structure FourDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  a_bound : a < 10
  b_bound : b < 10
  c_bound : c < 10
  d_bound : d < 10

/-- The invariant quantity M for a four-digit number --/
def invariant_M (n : FourDigitNumber) : Int :=
  (n.d + n.b) - (n.a + n.c)

/-- The allowed operations on four-digit numbers --/
inductive Operation
  | AddAdjacent (i : Fin 3)
  | SubtractAdjacent (i : Fin 3)

/-- Applying an operation to a four-digit number --/
def apply_operation (n : FourDigitNumber) (op : Operation) : Option FourDigitNumber :=
  sorry

/-- The main theorem: it's impossible to transform 1234 into 2002 --/
theorem impossibility_of_transformation :
  ∀ (ops : List Operation),
    let start := FourDigitNumber.mk 1 2 3 4 (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    let target := FourDigitNumber.mk 2 0 0 2 (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    ∀ (result : FourDigitNumber),
      (ops.foldl (fun n op => (apply_operation n op).getD n) start = result) →
      result ≠ target :=
by
  sorry

end NUMINAMATH_CALUDE_impossibility_of_transformation_l2279_227980


namespace NUMINAMATH_CALUDE_pat_earned_stickers_l2279_227919

/-- The number of stickers Pat had at the beginning of the week -/
def initial_stickers : ℕ := 39

/-- The number of stickers Pat had at the end of the week -/
def final_stickers : ℕ := 61

/-- The number of stickers Pat earned during the week -/
def earned_stickers : ℕ := final_stickers - initial_stickers

theorem pat_earned_stickers : earned_stickers = 22 := by sorry

end NUMINAMATH_CALUDE_pat_earned_stickers_l2279_227919


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2279_227915

/-- Given a quadratic function f(x) = ax² + bx + c with a > 0, and roots α and β of f(x) = x 
    where 0 < α < β, prove that x < f(x) for all x such that 0 < x < α -/
theorem quadratic_inequality (a b c α β : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^2 + b * x + c)
  (h2 : a > 0)
  (h3 : f α = α)
  (h4 : f β = β)
  (h5 : 0 < α)
  (h6 : α < β) :
  ∀ x, 0 < x → x < α → x < f x :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2279_227915


namespace NUMINAMATH_CALUDE_total_interest_calculation_l2279_227967

theorem total_interest_calculation (stock1_rate stock2_rate stock3_rate : ℝ) 
  (face_value : ℝ) (h1 : stock1_rate = 0.16) (h2 : stock2_rate = 0.12) 
  (h3 : stock3_rate = 0.20) (h4 : face_value = 100) : 
  stock1_rate * face_value + stock2_rate * face_value + stock3_rate * face_value = 48 := by
  sorry

end NUMINAMATH_CALUDE_total_interest_calculation_l2279_227967


namespace NUMINAMATH_CALUDE_volume_of_given_prism_l2279_227994

/-- Represents the dimensions of a rectangular prism in centimeters -/
structure PrismDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular prism given its dimensions -/
def prismVolume (d : PrismDimensions) : ℝ :=
  d.length * d.width * d.height

/-- The dimensions of the specific rectangular prism in the problem -/
def givenPrism : PrismDimensions :=
  { length := 4
    width := 2
    height := 8 }

/-- Theorem stating that the volume of the given rectangular prism is 64 cubic centimeters -/
theorem volume_of_given_prism :
  prismVolume givenPrism = 64 := by
  sorry

#check volume_of_given_prism

end NUMINAMATH_CALUDE_volume_of_given_prism_l2279_227994


namespace NUMINAMATH_CALUDE_train_distance_trains_ab_distance_l2279_227974

/-- The distance between two trains' starting points given their speed and meeting point -/
theorem train_distance (speed : ℝ) (distance_a : ℝ) : speed > 0 → distance_a > 0 →
  2 * distance_a = (distance_a * speed + distance_a * speed) / speed := by
  sorry

/-- The specific problem of trains A and B -/
theorem trains_ab_distance : 
  let speed : ℝ := 50
  let distance_a : ℝ := 225
  2 * distance_a = 450 := by
  sorry

end NUMINAMATH_CALUDE_train_distance_trains_ab_distance_l2279_227974


namespace NUMINAMATH_CALUDE_teacher_distribution_count_l2279_227960

/-- The number of ways to distribute 4 teachers among 3 middle schools -/
def distribute_teachers : ℕ :=
  Nat.choose 4 2 * Nat.factorial 3

/-- Theorem: The number of ways to distribute 4 teachers among 3 middle schools,
    with each school having at least one teacher, is equal to 36 -/
theorem teacher_distribution_count : distribute_teachers = 36 := by
  sorry

end NUMINAMATH_CALUDE_teacher_distribution_count_l2279_227960


namespace NUMINAMATH_CALUDE_fraction_change_l2279_227946

/-- Given a fraction that changes from 1/12 to 2/15 when its numerator is increased by 20% and
    its denominator is decreased by x%, prove that x = 25. -/
theorem fraction_change (x : ℚ) : 
  (1 : ℚ) / 12 * (120 / 100) / ((100 - x) / 100) = 2 / 15 → x = 25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_change_l2279_227946


namespace NUMINAMATH_CALUDE_system_solution_l2279_227964

theorem system_solution :
  let solutions : List (ℝ × ℝ × ℝ) := [
    (1, -1, 1), (1, 3/2, -2/3), (-2, 1/2, 1), (-2, 3/2, 1/3), (3, -1, 1/3), (3, 1/2, -2/3)
  ]
  ∀ (x y z : ℝ),
    (x + 2*y + 3*z = 2 ∧
     1/x + 1/(2*y) + 1/(3*z) = 5/6 ∧
     x*y*z = -1) ↔
    (x, y, z) ∈ solutions := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2279_227964


namespace NUMINAMATH_CALUDE_allocation_schemes_l2279_227991

/-- The number of volunteers --/
def num_volunteers : ℕ := 5

/-- The number of tasks --/
def num_tasks : ℕ := 4

/-- The number of volunteers who cannot perform a specific task --/
def num_restricted : ℕ := 2

/-- Calculates the number of ways to allocate tasks to volunteers with restrictions --/
def num_allocations (n v t r : ℕ) : ℕ :=
  -- n: total number of volunteers
  -- v: number of volunteers to be selected
  -- t: number of tasks
  -- r: number of volunteers who cannot perform a specific task
  sorry

/-- Theorem stating the number of allocation schemes --/
theorem allocation_schemes :
  num_allocations num_volunteers num_tasks num_tasks num_restricted = 72 :=
by sorry

end NUMINAMATH_CALUDE_allocation_schemes_l2279_227991


namespace NUMINAMATH_CALUDE_product_of_numbers_l2279_227981

theorem product_of_numbers (x y : ℝ) : x + y = 25 ∧ x - y = 7 → x * y = 144 := by sorry

end NUMINAMATH_CALUDE_product_of_numbers_l2279_227981


namespace NUMINAMATH_CALUDE_power_of_negative_cube_l2279_227938

theorem power_of_negative_cube (a : ℝ) : (-a^3)^4 = a^12 := by
  sorry

end NUMINAMATH_CALUDE_power_of_negative_cube_l2279_227938


namespace NUMINAMATH_CALUDE_smallest_number_of_cubes_is_56_l2279_227924

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  depth : ℕ

/-- Calculates the smallest number of identical cubes that can fill a box completely -/
def smallestNumberOfCubes (box : BoxDimensions) : ℕ :=
  let cubeSideLength := Nat.gcd (Nat.gcd box.length box.width) box.depth
  (box.length / cubeSideLength) * (box.width / cubeSideLength) * (box.depth / cubeSideLength)

/-- Theorem stating that the smallest number of cubes to fill the given box is 56 -/
theorem smallest_number_of_cubes_is_56 :
  smallestNumberOfCubes ⟨35, 20, 10⟩ = 56 := by
  sorry

#eval smallestNumberOfCubes ⟨35, 20, 10⟩

end NUMINAMATH_CALUDE_smallest_number_of_cubes_is_56_l2279_227924


namespace NUMINAMATH_CALUDE_total_coins_l2279_227940

/-- Represents a 3x3 grid of cells --/
def Grid := Fin 3 → Fin 3 → ℕ

/-- The sum of coins in the corner cells --/
def corner_sum (g : Grid) : ℕ :=
  g 0 0 + g 0 2 + g 2 0 + g 2 2

/-- The number of coins in the center cell --/
def center_value (g : Grid) : ℕ :=
  g 1 1

/-- Theorem stating the total number of coins in the grid --/
theorem total_coins (g : Grid) 
  (h_corner : corner_sum g = 8) 
  (h_center : center_value g = 3) : 
  ∃ (total : ℕ), total = 8 :=
sorry

end NUMINAMATH_CALUDE_total_coins_l2279_227940


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_minimum_l2279_227955

theorem geometric_sequence_sum_minimum (q : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) :
  q > 1 →
  (∀ n, a n = a 1 * q^(n-1)) →
  (∀ n, S n = (a 1 * (1 - q^n)) / (1 - q)) →
  S 4 = 2 * S 2 + 1 →
  ∃ S_6_min : ℝ, S_6_min = 2 * Real.sqrt 3 + 3 ∧ S 6 ≥ S_6_min :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_minimum_l2279_227955


namespace NUMINAMATH_CALUDE_functional_equation_implies_g_five_l2279_227907

/-- A function g: ℝ → ℝ satisfying g(xy) = g(x)g(y) for all real x and y, and g(1) = 2 -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, g (x * y) = g x * g y) ∧ (g 1 = 2)

/-- If g satisfies the functional equation, then g(5) = 32 -/
theorem functional_equation_implies_g_five (g : ℝ → ℝ) :
  FunctionalEquation g → g 5 = 32 := by
  sorry


end NUMINAMATH_CALUDE_functional_equation_implies_g_five_l2279_227907


namespace NUMINAMATH_CALUDE_percentage_of_b_grades_l2279_227925

def scores : List ℕ := [91, 82, 68, 99, 79, 86, 88, 76, 71, 58, 80, 89, 65, 85, 93]

def is_b_grade (score : ℕ) : Bool :=
  87 ≤ score && score ≤ 94

def count_b_grades (scores : List ℕ) : ℕ :=
  scores.filter is_b_grade |>.length

theorem percentage_of_b_grades :
  (count_b_grades scores : ℚ) / (scores.length : ℚ) * 100 = 80 / 3 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_b_grades_l2279_227925


namespace NUMINAMATH_CALUDE_algebraic_expression_symmetry_l2279_227933

theorem algebraic_expression_symmetry (a b : ℝ) : 
  (a * 3^3 + b * 3 - 5 = 20) → (a * (-3)^3 + b * (-3) - 5 = -30) := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_symmetry_l2279_227933


namespace NUMINAMATH_CALUDE_rectangle_side_ratio_l2279_227914

theorem rectangle_side_ratio (a b c d : ℝ) (h1 : a * b / (c * d) = 0.16) (h2 : b / d = 2 / 5) :
  a / c = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_side_ratio_l2279_227914


namespace NUMINAMATH_CALUDE_tooth_extraction_cost_l2279_227911

def cleaning_cost : ℕ := 70
def filling_cost : ℕ := 120
def root_canal_cost : ℕ := 400
def dental_crown_cost : ℕ := 600

def total_known_costs : ℕ := cleaning_cost + 3 * filling_cost + root_canal_cost + dental_crown_cost

def total_bill : ℕ := 9 * root_canal_cost

theorem tooth_extraction_cost : 
  total_bill - total_known_costs = 2170 := by sorry

end NUMINAMATH_CALUDE_tooth_extraction_cost_l2279_227911


namespace NUMINAMATH_CALUDE_car_average_speed_l2279_227909

theorem car_average_speed (s1 s2 s3 s4 s5 : ℝ) 
  (h1 : s1 = 120) (h2 : s2 = 70) (h3 : s3 = 90) (h4 : s4 = 110) (h5 : s5 = 80) :
  (s1 + s2 + s3 + s4 + s5) / 5 = 94 := by
  sorry

end NUMINAMATH_CALUDE_car_average_speed_l2279_227909


namespace NUMINAMATH_CALUDE_profit_percentage_example_l2279_227927

/-- Calculate the profit percentage given selling price and cost price -/
def profit_percentage (selling_price : ℚ) (cost_price : ℚ) : ℚ :=
  ((selling_price - cost_price) / cost_price) * 100

/-- Theorem: The profit percentage is 25% when the selling price is 400 and the cost price is 320 -/
theorem profit_percentage_example : profit_percentage 400 320 = 25 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_example_l2279_227927


namespace NUMINAMATH_CALUDE_eighth_term_is_negative_22_l2279_227922

/-- An arithmetic sequence with a2 = -4 and common difference -3 -/
def arithmetic_sequence (n : ℕ) : ℤ :=
  -4 + (n - 2) * (-3)

/-- Theorem: The 8th term of the arithmetic sequence is -22 -/
theorem eighth_term_is_negative_22 : arithmetic_sequence 8 = -22 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_is_negative_22_l2279_227922


namespace NUMINAMATH_CALUDE_union_A_B_l2279_227962

/-- Set A is defined as the set of real numbers between -2 and 3 inclusive -/
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}

/-- Set B is defined as the set of positive real numbers -/
def B : Set ℝ := {x | x > 0}

/-- The union of sets A and B is equal to the set of real numbers greater than or equal to -2 -/
theorem union_A_B : A ∪ B = {x : ℝ | x ≥ -2} := by sorry

end NUMINAMATH_CALUDE_union_A_B_l2279_227962


namespace NUMINAMATH_CALUDE_height_of_specific_block_l2279_227978

/-- Represents a rectangular block --/
structure RectangularBlock where
  length : ℕ
  width : ℕ
  height : ℕ

/-- The volume of the block in cubic centimeters --/
def volume (block : RectangularBlock) : ℕ :=
  block.length * block.width * block.height

/-- The perimeter of the base of the block in centimeters --/
def basePerimeter (block : RectangularBlock) : ℕ :=
  2 * (block.length + block.width)

theorem height_of_specific_block :
  ∃ (block : RectangularBlock),
    volume block = 42 ∧
    basePerimeter block = 18 ∧
    block.height = 3 :=
by
  sorry

#check height_of_specific_block

end NUMINAMATH_CALUDE_height_of_specific_block_l2279_227978


namespace NUMINAMATH_CALUDE_vector_magnitude_difference_l2279_227908

theorem vector_magnitude_difference (a b : ℝ × ℝ) 
  (h1 : a ≠ (0, 0)) 
  (h2 : b ≠ (0, 0)) 
  (h3 : a + b = (-3, 6)) 
  (h4 : a - b = (-3, 2)) : 
  ‖a‖^2 - ‖b‖^2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_difference_l2279_227908


namespace NUMINAMATH_CALUDE_parabola_equation_l2279_227975

/-- A parabola with vertex at the origin and directrix x = 2 -/
structure Parabola where
  /-- The equation of the parabola in the form y² = kx -/
  equation : ℝ → ℝ → Prop
  /-- The vertex of the parabola is at the origin -/
  vertex_at_origin : equation 0 0
  /-- The directrix of the parabola has equation x = 2 -/
  directrix_at_two : ∀ y, ¬ equation 2 y

/-- The equation of the parabola is y² = -16x -/
theorem parabola_equation (C : Parabola) : 
  C.equation = fun x y ↦ y^2 = -16*x := by sorry

end NUMINAMATH_CALUDE_parabola_equation_l2279_227975


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2279_227997

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  b = 4 →
  (1/2) * a * b * Real.sin C = 6 * Real.sqrt 3 →
  c * Real.sin C - a * Real.sin A = (b - a) * Real.sin B →
  C = π / 3 ∧ c = 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l2279_227997
