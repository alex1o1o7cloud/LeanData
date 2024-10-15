import Mathlib

namespace NUMINAMATH_CALUDE_inequality_proof_l3650_365011

theorem inequality_proof (x : ℝ) (hx : x > 0) :
  (1 + x + x^2) * (1 + x + x^2 + x^3 + x^4) ≤ (1 + x + x^2 + x^3)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3650_365011


namespace NUMINAMATH_CALUDE_range_of_f_l3650_365048

def f (x : ℝ) : ℝ := x^2 - 4*x + 4

theorem range_of_f :
  ∀ x ∈ Set.Icc (-3 : ℝ) 3, ∃ y ∈ Set.Icc 0 25, f x = y ∧
  ∀ z, f x = z → z ∈ Set.Icc 0 25 :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_l3650_365048


namespace NUMINAMATH_CALUDE_min_students_with_both_traits_l3650_365006

theorem min_students_with_both_traits (total : ℕ) (blue_eyes : ℕ) (lunch_box : ℕ)
  (h1 : total = 35)
  (h2 : blue_eyes = 15)
  (h3 : lunch_box = 23)
  (h4 : blue_eyes ≤ total)
  (h5 : lunch_box ≤ total) :
  total - (total - blue_eyes) - (total - lunch_box) ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_min_students_with_both_traits_l3650_365006


namespace NUMINAMATH_CALUDE_painting_time_is_18_17_l3650_365010

/-- The time required for three painters to complete a room, given their individual rates and break times -/
def total_painting_time (linda_rate tom_rate jerry_rate : ℚ) (tom_break jerry_break : ℚ) : ℚ :=
  let combined_rate := linda_rate + tom_rate + jerry_rate
  18 / 17

/-- Theorem stating that the total painting time for Linda, Tom, and Jerry is 18/17 hours -/
theorem painting_time_is_18_17 :
  let linda_rate : ℚ := 1 / 3
  let tom_rate : ℚ := 1 / 4
  let jerry_rate : ℚ := 1 / 6
  let tom_break : ℚ := 2
  let jerry_break : ℚ := 1
  total_painting_time linda_rate tom_rate jerry_rate tom_break jerry_break = 18 / 17 := by
  sorry

#eval total_painting_time (1/3) (1/4) (1/6) 2 1

end NUMINAMATH_CALUDE_painting_time_is_18_17_l3650_365010


namespace NUMINAMATH_CALUDE_find_D_l3650_365064

theorem find_D (A B C D : ℤ) 
  (h1 : A + C = 15)
  (h2 : A - B = 1)
  (h3 : C + C = A)
  (h4 : B - D = 2)
  (h5 : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) :
  D = 7 := by
sorry

end NUMINAMATH_CALUDE_find_D_l3650_365064


namespace NUMINAMATH_CALUDE_max_consecutive_sum_36_l3650_365087

/-- The sum of consecutive integers from a to (a + n - 1) -/
def sum_consecutive (a : ℤ) (n : ℕ) : ℤ := n * a + (n * (n - 1)) / 2

/-- The proposition that 72 is the maximum number of consecutive integers summing to 36 -/
theorem max_consecutive_sum_36 :
  (∃ a : ℤ, sum_consecutive a 72 = 36) ∧
  (∀ n : ℕ, n > 72 → ∀ a : ℤ, sum_consecutive a n ≠ 36) :=
sorry

end NUMINAMATH_CALUDE_max_consecutive_sum_36_l3650_365087


namespace NUMINAMATH_CALUDE_least_multiple_17_greater_500_l3650_365094

theorem least_multiple_17_greater_500 : ∃ (n : ℕ), n * 17 = 510 ∧ 
  510 > 500 ∧ (∀ m : ℕ, m * 17 > 500 → m * 17 ≥ 510) := by
  sorry

end NUMINAMATH_CALUDE_least_multiple_17_greater_500_l3650_365094


namespace NUMINAMATH_CALUDE_cuboidal_box_volume_l3650_365041

/-- A cuboidal box with given adjacent face areas has a specific volume -/
theorem cuboidal_box_volume (l w h : ℝ) (h1 : l * w = 120) (h2 : w * h = 72) (h3 : h * l = 60) :
  l * w * h = 4320 := by
  sorry

end NUMINAMATH_CALUDE_cuboidal_box_volume_l3650_365041


namespace NUMINAMATH_CALUDE_brandon_card_count_l3650_365074

theorem brandon_card_count (malcom_cards : ℕ) (brandon_cards : ℕ) : 
  (malcom_cards = brandon_cards + 8) →
  (malcom_cards / 2 = 14) →
  brandon_cards = 20 := by
sorry

end NUMINAMATH_CALUDE_brandon_card_count_l3650_365074


namespace NUMINAMATH_CALUDE_original_group_size_l3650_365017

/-- Represents the work capacity of a group of men --/
def work_capacity (men : ℕ) (days : ℕ) : ℕ := men * days

theorem original_group_size
  (initial_days : ℕ)
  (absent_men : ℕ)
  (final_days : ℕ)
  (h1 : initial_days = 20)
  (h2 : absent_men = 10)
  (h3 : final_days = 40)
  : ∃ (original_size : ℕ),
    work_capacity original_size initial_days =
    work_capacity (original_size - absent_men) final_days ∧
    original_size = 20 :=
by sorry

end NUMINAMATH_CALUDE_original_group_size_l3650_365017


namespace NUMINAMATH_CALUDE_tan_inequality_l3650_365097

theorem tan_inequality (h1 : 130 * π / 180 > π / 2) (h2 : 130 * π / 180 < π)
                       (h3 : 140 * π / 180 > π / 2) (h4 : 140 * π / 180 < π) :
  Real.tan (130 * π / 180) < Real.tan (140 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_tan_inequality_l3650_365097


namespace NUMINAMATH_CALUDE_angle_bisector_ratio_l3650_365030

-- Define the triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the angle bisector
def angleBisector (t : Triangle) : ℝ × ℝ :=
  sorry

-- Define the intersection point
def intersectionPoint (p q r s : ℝ × ℝ) : ℝ × ℝ :=
  sorry

-- Define the distance between two points
def distance (p q : ℝ × ℝ) : ℝ :=
  sorry

theorem angle_bisector_ratio (t : Triangle) :
  let D : ℝ × ℝ := ((t.A.1 + t.B.1) / 2, (t.A.2 + t.B.2) / 2)
  let E : ℝ × ℝ := ((t.A.1 + t.C.1) / 2, (t.A.2 + t.C.2) / 2)
  let T : ℝ × ℝ := angleBisector t
  let F : ℝ × ℝ := intersectionPoint t.A T D E
  distance t.A D = distance D t.B ∧ 
  distance t.A E = distance E t.C ∧
  distance t.A D = 2 ∧
  distance t.A E = 3 →
  distance t.A F / distance t.A T = 1 / 3 :=
sorry

end NUMINAMATH_CALUDE_angle_bisector_ratio_l3650_365030


namespace NUMINAMATH_CALUDE_fireflies_joining_l3650_365026

theorem fireflies_joining (initial : ℕ) (joined : ℕ) (left : ℕ) (remaining : ℕ) : 
  initial = 3 → left = 2 → remaining = 9 → initial + joined - left = remaining → joined = 8 := by
  sorry

end NUMINAMATH_CALUDE_fireflies_joining_l3650_365026


namespace NUMINAMATH_CALUDE_evaluate_expression_l3650_365061

theorem evaluate_expression : 7^3 - 4 * 7^2 + 4 * 7 - 1 = 174 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3650_365061


namespace NUMINAMATH_CALUDE_vertex_angle_of_special_triangle_l3650_365089

/-- A triangle with angles a, b, and c is isosceles and a "double angle triangle" -/
structure SpecialTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  sum_180 : a + b + c = 180
  isosceles : b = c
  double_angle : a = 2 * b ∨ b = 2 * a

/-- The vertex angle of an isosceles "double angle triangle" is either 36° or 90° -/
theorem vertex_angle_of_special_triangle (t : SpecialTriangle) :
  t.a = 36 ∨ t.a = 90 := by
  sorry

end NUMINAMATH_CALUDE_vertex_angle_of_special_triangle_l3650_365089


namespace NUMINAMATH_CALUDE_henri_total_time_l3650_365075

/-- Represents the total time Henri has for watching movies and reading -/
def total_time : ℝ := 8

/-- Duration of the first movie Henri watches -/
def movie1_duration : ℝ := 3.5

/-- Duration of the second movie Henri watches -/
def movie2_duration : ℝ := 1.5

/-- Henri's reading speed in words per minute -/
def reading_speed : ℝ := 10

/-- Number of words Henri reads -/
def words_read : ℝ := 1800

/-- Theorem stating that Henri's total time for movies and reading is 8 hours -/
theorem henri_total_time : 
  movie1_duration + movie2_duration + (words_read / reading_speed) / 60 = total_time := by
  sorry

end NUMINAMATH_CALUDE_henri_total_time_l3650_365075


namespace NUMINAMATH_CALUDE_final_mixture_volume_l3650_365038

/-- Represents an alcohol mixture -/
structure AlcoholMixture where
  volume : ℝ
  concentration : ℝ

/-- The problem setup -/
def mixture_problem (mixture30 mixture50 mixtureFinal : AlcoholMixture) : Prop :=
  mixture30.concentration = 0.30 ∧
  mixture50.concentration = 0.50 ∧
  mixtureFinal.concentration = 0.45 ∧
  mixture30.volume = 2.5 ∧
  mixtureFinal.volume = mixture30.volume + mixture50.volume ∧
  mixture30.volume * mixture30.concentration + mixture50.volume * mixture50.concentration =
    mixtureFinal.volume * mixtureFinal.concentration

/-- The theorem statement -/
theorem final_mixture_volume
  (mixture30 mixture50 mixtureFinal : AlcoholMixture)
  (h : mixture_problem mixture30 mixture50 mixtureFinal) :
  mixtureFinal.volume = 10 :=
sorry

end NUMINAMATH_CALUDE_final_mixture_volume_l3650_365038


namespace NUMINAMATH_CALUDE_arithmetic_sequence_10th_term_l3650_365085

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Given an arithmetic sequence a with a₂ = 2 and a₃ = 4,
    prove that the 10th term a₁₀ = 18. -/
theorem arithmetic_sequence_10th_term
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_2 : a 2 = 2)
  (h_3 : a 3 = 4) :
  a 10 = 18 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_10th_term_l3650_365085


namespace NUMINAMATH_CALUDE_point_transformation_l3650_365022

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the transformation function
def transform (p : Point2D) : Point2D :=
  { x := p.x + 3, y := p.y + 5 }

theorem point_transformation :
  ∀ (x y : ℝ),
  let A : Point2D := { x := x, y := -2 }
  let B : Point2D := transform A
  B.x = 1 → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_l3650_365022


namespace NUMINAMATH_CALUDE_least_integer_satisfying_condition_l3650_365056

/-- Given a positive integer n, returns the integer formed by removing its leftmost digit. -/
def removeLeftmostDigit (n : ℕ+) : ℕ :=
  sorry

/-- Checks if a positive integer satisfies the condition that removing its leftmost digit
    results in 1/29 of the original number. -/
def satisfiesCondition (n : ℕ+) : Prop :=
  removeLeftmostDigit n = n.val / 29

/-- Proves that 725 is the least positive integer that satisfies the given condition. -/
theorem least_integer_satisfying_condition :
  satisfiesCondition 725 ∧ ∀ m : ℕ+, m < 725 → ¬satisfiesCondition m :=
sorry

end NUMINAMATH_CALUDE_least_integer_satisfying_condition_l3650_365056


namespace NUMINAMATH_CALUDE_age_difference_l3650_365005

theorem age_difference (P M Mo : ℕ) 
  (h1 : P * 5 = M * 3) 
  (h2 : M * 5 = Mo * 3) 
  (h3 : P + M + Mo = 196) : 
  Mo - P = 64 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3650_365005


namespace NUMINAMATH_CALUDE_f_monotone_decreasing_l3650_365047

noncomputable def f (x : ℝ) := (x + 1) * Real.exp x

theorem f_monotone_decreasing :
  ∀ x y, x < y → x < -2 → y < -2 → f y < f x := by sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_l3650_365047


namespace NUMINAMATH_CALUDE_function_range_is_all_reals_l3650_365088

theorem function_range_is_all_reals :
  ∀ y : ℝ, ∃ x : ℝ, y = (x^2 + 3*x + 2) / (x^2 + x + 1) := by
  sorry

end NUMINAMATH_CALUDE_function_range_is_all_reals_l3650_365088


namespace NUMINAMATH_CALUDE_line_bisects_circle_implies_b_eq_neg_two_l3650_365042

/-- The line l is defined by parametric equations x = 2t and y = 1 + bt -/
def line_l (b t : ℝ) : ℝ × ℝ := (2 * t, 1 + b * t)

/-- The circle C is defined by the equation (x - 1)^2 + y^2 = 1 -/
def circle_C (p : ℝ × ℝ) : Prop :=
  (p.1 - 1)^2 + p.2^2 = 1

/-- A line bisects the area of a circle if it passes through the center of the circle -/
def bisects_circle_area (l : ℝ → ℝ × ℝ) (c : ℝ × ℝ → Prop) : Prop :=
  ∃ t, l t = (1, 0)

/-- Main theorem: If line l bisects the area of circle C, then b = -2 -/
theorem line_bisects_circle_implies_b_eq_neg_two (b : ℝ) :
  bisects_circle_area (line_l b) circle_C → b = -2 :=
sorry

end NUMINAMATH_CALUDE_line_bisects_circle_implies_b_eq_neg_two_l3650_365042


namespace NUMINAMATH_CALUDE_min_max_abs_quadratic_l3650_365001

-- Define the function f(x) = x^2 + ax + b
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

-- State the theorem
theorem min_max_abs_quadratic (a b : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, |f a b x| ≤ 1/2) ↔ a = 0 ∧ b = -1/2 :=
sorry

end NUMINAMATH_CALUDE_min_max_abs_quadratic_l3650_365001


namespace NUMINAMATH_CALUDE_dessert_probability_l3650_365080

theorem dessert_probability (p_dessert_and_coffee : ℝ) (p_no_coffee_given_dessert : ℝ) :
  p_dessert_and_coffee = 0.6 →
  p_no_coffee_given_dessert = 0.2 →
  1 - (p_dessert_and_coffee / (1 - p_no_coffee_given_dessert)) = 0.4 :=
by sorry

end NUMINAMATH_CALUDE_dessert_probability_l3650_365080


namespace NUMINAMATH_CALUDE_correlation_coefficient_is_one_l3650_365039

/-- A structure representing a set of sample data points -/
structure SampleData where
  n : ℕ
  x : Fin n → ℝ
  y : Fin n → ℝ
  h_n : n ≥ 2
  h_distinct : ∀ i j, i ≠ j → x i ≠ x j
  h_line : ∀ i, y i = (1/3) * x i - 5

/-- The sample correlation coefficient of a set of data points -/
def sampleCorrelationCoefficient (data : SampleData) : ℝ :=
  sorry

/-- Theorem stating that the sample correlation coefficient is 1 
    for data points satisfying the given conditions -/
theorem correlation_coefficient_is_one (data : SampleData) :
  sampleCorrelationCoefficient data = 1 :=
sorry

end NUMINAMATH_CALUDE_correlation_coefficient_is_one_l3650_365039


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l3650_365060

theorem fractional_equation_solution (k : ℝ) : 
  (∃ x : ℝ, x ≠ 0 ∧ x ≠ 1 ∧ (3 / x + 6 / (x - 1) - (x + k) / (x * (x - 1)) = 0)) 
  ↔ k ≠ -3 ∧ k ≠ 5 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l3650_365060


namespace NUMINAMATH_CALUDE_factorization_equality_l3650_365096

theorem factorization_equality (a b : ℝ) : 3 * a^2 * b - 3 * a * b + 6 * b = 3 * b * (a^2 - a + 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3650_365096


namespace NUMINAMATH_CALUDE_gcd_90_252_l3650_365003

theorem gcd_90_252 : Nat.gcd 90 252 = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcd_90_252_l3650_365003


namespace NUMINAMATH_CALUDE_nell_card_difference_l3650_365079

/-- Represents the number of cards Nell has -/
structure CardCount where
  baseball : ℕ
  ace : ℕ

/-- The difference between baseball and ace cards -/
def cardDifference (cards : CardCount) : ℤ :=
  cards.baseball - cards.ace

theorem nell_card_difference (initial final : CardCount) 
  (h1 : initial.baseball = 438)
  (h2 : initial.ace = 18)
  (h3 : final.baseball = 178)
  (h4 : final.ace = 55) :
  cardDifference final = 123 := by
  sorry

end NUMINAMATH_CALUDE_nell_card_difference_l3650_365079


namespace NUMINAMATH_CALUDE_fixed_point_parabola_l3650_365018

theorem fixed_point_parabola (k : ℝ) : 9 = 9 * (-1)^2 + k * (-1) - 5 * k := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_parabola_l3650_365018


namespace NUMINAMATH_CALUDE_system_of_equations_l3650_365055

theorem system_of_equations (x y c d : ℝ) 
  (eq1 : 4 * x + 8 * y = c)
  (eq2 : 5 * x - 10 * y = d)
  (h_d_nonzero : d ≠ 0)
  (h_x_nonzero : x ≠ 0)
  (h_y_nonzero : y ≠ 0) :
  c / d = -4 / 5 := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_l3650_365055


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3650_365046

-- Define the sets A and B
def A : Set ℝ := {x | -2 < x ∧ x ≤ 1}
def B : Set ℝ := {x | -1 ≤ x ∧ x < 2}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x | -2 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3650_365046


namespace NUMINAMATH_CALUDE_farmers_field_planted_fraction_l3650_365029

theorem farmers_field_planted_fraction :
  ∀ (a b c : ℝ) (s : ℝ),
    a = 5 →
    b = 12 →
    c^2 = a^2 + b^2 →
    (s / a) = (4 / c) →
    (a * b / 2 - s^2) / (a * b / 2) = 470 / 507 :=
by sorry

end NUMINAMATH_CALUDE_farmers_field_planted_fraction_l3650_365029


namespace NUMINAMATH_CALUDE_rectangle_side_length_l3650_365031

theorem rectangle_side_length (square_side : ℝ) (rectangle_width : ℝ) :
  square_side = 5 →
  rectangle_width = 4 →
  square_side * square_side = rectangle_width * (25 / rectangle_width) :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_side_length_l3650_365031


namespace NUMINAMATH_CALUDE_factors_of_36_l3650_365052

/-- The number of distinct positive factors of 36 is 9. -/
theorem factors_of_36 : Nat.card {d : ℕ | d > 0 ∧ 36 % d = 0} = 9 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_36_l3650_365052


namespace NUMINAMATH_CALUDE_minor_axis_length_of_ellipse_l3650_365092

/-- The length of the minor axis of the ellipse x^2/4 + y^2/36 = 1 is 4 -/
theorem minor_axis_length_of_ellipse : 
  let ellipse := (fun (x y : ℝ) => x^2/4 + y^2/36 = 1)
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 
    (∀ x y, ellipse x y ↔ x^2/a^2 + y^2/b^2 = 1) ∧
    2 * min a b = 4 :=
by sorry

end NUMINAMATH_CALUDE_minor_axis_length_of_ellipse_l3650_365092


namespace NUMINAMATH_CALUDE_kates_bill_l3650_365066

theorem kates_bill (bob_bill : ℝ) (bob_discount : ℝ) (kate_discount : ℝ) (total_after_discount : ℝ) :
  bob_bill = 30 →
  bob_discount = 0.05 →
  kate_discount = 0.02 →
  total_after_discount = 53 →
  ∃ kate_bill : ℝ,
    kate_bill = 25 ∧
    bob_bill * (1 - bob_discount) + kate_bill * (1 - kate_discount) = total_after_discount :=
by sorry

end NUMINAMATH_CALUDE_kates_bill_l3650_365066


namespace NUMINAMATH_CALUDE_twelve_pharmacies_not_enough_l3650_365063

/-- Represents a grid of streets -/
structure Grid :=
  (rows : Nat)
  (cols : Nat)

/-- Represents a pharmacy on the grid -/
structure Pharmacy :=
  (row : Nat)
  (col : Nat)

/-- The maximum walking distance to a pharmacy -/
def max_walking_distance : Nat := 3

/-- Calculate the number of street segments covered by a pharmacy -/
def covered_segments (g : Grid) (p : Pharmacy) : Nat :=
  let coverage_side := 2 * max_walking_distance + 1
  min coverage_side g.rows * min coverage_side g.cols

/-- Calculate the total number of street segments in the grid -/
def total_segments (g : Grid) : Nat :=
  2 * g.rows * (g.cols - 1) + 2 * g.cols * (g.rows - 1)

/-- The main theorem to be proved -/
theorem twelve_pharmacies_not_enough :
  ∀ (pharmacies : List Pharmacy),
    pharmacies.length = 12 →
    ∃ (g : Grid),
      g.rows = 9 ∧ g.cols = 9 ∧
      (pharmacies.map (covered_segments g)).sum < total_segments g := by
  sorry


end NUMINAMATH_CALUDE_twelve_pharmacies_not_enough_l3650_365063


namespace NUMINAMATH_CALUDE_total_books_on_shelves_l3650_365076

/-- Given 150 book shelves with 15 books each, the total number of books is 2250. -/
theorem total_books_on_shelves (num_shelves : ℕ) (books_per_shelf : ℕ) : 
  num_shelves = 150 → books_per_shelf = 15 → num_shelves * books_per_shelf = 2250 := by
  sorry

end NUMINAMATH_CALUDE_total_books_on_shelves_l3650_365076


namespace NUMINAMATH_CALUDE_line_segment_lattice_points_l3650_365028

/-- The number of lattice points on a line segment with given integer coordinates --/
def latticePointCount (x1 y1 x2 y2 : ℤ) : ℕ :=
  sorry

/-- Theorem: The number of lattice points on the line segment from (5, 23) to (53, 311) is 49 --/
theorem line_segment_lattice_points :
  latticePointCount 5 23 53 311 = 49 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_lattice_points_l3650_365028


namespace NUMINAMATH_CALUDE_least_integer_satisfying_inequality_l3650_365077

theorem least_integer_satisfying_inequality :
  ∀ y : ℤ, (3 * |y| + 6 < 24) → y ≥ -5 ∧ 
  ∃ x : ℤ, x = -5 ∧ (3 * |x| + 6 < 24) := by
  sorry

end NUMINAMATH_CALUDE_least_integer_satisfying_inequality_l3650_365077


namespace NUMINAMATH_CALUDE_cos_2alpha_minus_3pi_over_7_l3650_365036

theorem cos_2alpha_minus_3pi_over_7 (α : ℝ) 
  (h : Real.sin (α + 2 * Real.pi / 7) = Real.sqrt 6 / 3) : 
  Real.cos (2 * α - 3 * Real.pi / 7) = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_cos_2alpha_minus_3pi_over_7_l3650_365036


namespace NUMINAMATH_CALUDE_smallest_power_complex_equality_l3650_365082

theorem smallest_power_complex_equality (n : ℕ) (c d : ℝ) :
  (n > 0) →
  (c > 0) →
  (d > 0) →
  (∀ k < n, ∃ a b : ℝ, (a > 0 ∧ b > 0 ∧ (a + b * I) ^ (2 * k) ≠ (a - b * I) ^ (2 * k))) →
  ((c + d * I) ^ (2 * n) = (c - d * I) ^ (2 * n)) →
  (d / c = 1) := by
sorry

end NUMINAMATH_CALUDE_smallest_power_complex_equality_l3650_365082


namespace NUMINAMATH_CALUDE_lisa_additional_marbles_l3650_365009

def minimum_additional_marbles (num_friends : ℕ) (initial_marbles : ℕ) : ℕ :=
  let required_marbles := (num_friends * (num_friends + 1)) / 2
  if required_marbles > initial_marbles then
    required_marbles - initial_marbles
  else
    0

theorem lisa_additional_marbles :
  minimum_additional_marbles 11 45 = 21 := by
  sorry

end NUMINAMATH_CALUDE_lisa_additional_marbles_l3650_365009


namespace NUMINAMATH_CALUDE_k_at_neg_one_eq_64_l3650_365070

/-- The polynomial h(x) -/
def h (p : ℝ) (x : ℝ) : ℝ := x^3 - p*x^2 + 3*x + 20

/-- The polynomial k(x) -/
def k (q r : ℝ) (x : ℝ) : ℝ := x^4 + x^3 - q*x^2 + 50*x + r

/-- Theorem stating that k(-1) = 64 given the conditions -/
theorem k_at_neg_one_eq_64 (p q r : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ h p x = 0 ∧ h p y = 0 ∧ h p z = 0) →
  (∀ x : ℝ, h p x = 0 → k q r x = 0) →
  k q r (-1) = 64 :=
by sorry

end NUMINAMATH_CALUDE_k_at_neg_one_eq_64_l3650_365070


namespace NUMINAMATH_CALUDE_min_value_theorem_equality_condition_l3650_365090

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^2 + b^2 + 1/a^2 + 1/b^2 + b/a + a/b ≥ Real.sqrt 15 := by
  sorry

theorem equality_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^2 + b^2 + 1/a^2 + 1/b^2 + b/a + a/b = Real.sqrt 15 ↔ 
  a = (3/20)^(1/4) ∧ b = 1/(2*a) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_equality_condition_l3650_365090


namespace NUMINAMATH_CALUDE_geometric_progression_iff_equal_first_two_l3650_365065

/-- A sequence of positive real numbers -/
def Sequence := ℕ → ℝ

/-- Predicate to check if a sequence is positive -/
def IsPositive (a : Sequence) : Prop :=
  ∀ n, a n > 0

/-- Predicate to check if a sequence satisfies the given recurrence relation -/
def SatisfiesRecurrence (a : Sequence) (b : ℝ) : Prop :=
  ∀ n, a (n + 2) = (b + 1) * a n * a (n + 1)

/-- Predicate to check if a sequence is a geometric progression -/
def IsGeometricProgression (a : Sequence) : Prop :=
  ∃ r, ∀ n, a (n + 1) = r * a n

/-- Main theorem -/
theorem geometric_progression_iff_equal_first_two (a : Sequence) (b : ℝ) :
  b > 0 ∧ IsPositive a ∧ SatisfiesRecurrence a b →
  IsGeometricProgression a ↔ a 1 = a 0 :=
sorry

end NUMINAMATH_CALUDE_geometric_progression_iff_equal_first_two_l3650_365065


namespace NUMINAMATH_CALUDE_solution_set_abs_inequality_l3650_365035

theorem solution_set_abs_inequality :
  {x : ℝ | |2*x - 1| < 1} = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_abs_inequality_l3650_365035


namespace NUMINAMATH_CALUDE_sin_sum_of_roots_l3650_365002

theorem sin_sum_of_roots (a b c : ℝ) (α β : ℝ) :
  (0 < α) → (α < π) →
  (0 < β) → (β < π) →
  (α ≠ β) →
  (a * Real.cos α + b * Real.sin α + c = 0) →
  (a * Real.cos β + b * Real.sin β + c = 0) →
  Real.sin (α + β) = (2 * a * b) / (a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_of_roots_l3650_365002


namespace NUMINAMATH_CALUDE_parabola_circle_intersection_l3650_365057

/-- Parabola type representing y = ax² --/
structure Parabola where
  a : ℝ

/-- Point type representing (x, y) --/
structure Point where
  x : ℝ
  y : ℝ

/-- Line type representing y = mx + b --/
structure Line where
  m : ℝ
  b : ℝ

/-- Circle type representing (x - h)² + (y - k)² = r² --/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- Given conditions of the problem --/
def given (C : Parabola) (M A B : Point) : Prop :=
  M.y = C.a * M.x^2 ∧
  M.x = 2 ∧ M.y = 1 ∧
  A.y = C.a * A.x^2 ∧
  B.y = C.a * B.x^2 ∧
  A ≠ M ∧ B ≠ M ∧
  ∃ (circ : Circle), (A.x - circ.h)^2 + (A.y - circ.k)^2 = circ.r^2 ∧
                     (B.x - circ.h)^2 + (B.y - circ.k)^2 = circ.r^2 ∧
                     (M.x - circ.h)^2 + (M.y - circ.k)^2 = circ.r^2 ∧
                     circ.r = (A.x - B.x)^2 + (A.y - B.y)^2

/-- The main theorem to be proved --/
theorem parabola_circle_intersection 
  (C : Parabola) (M A B : Point) (h : given C M A B) :
  (∃ (l : Line), l.m * (-2) + l.b = 5 ∧ l.m * A.x + l.b = A.y ∧ l.m * B.x + l.b = B.y) ∧
  (∃ (N : Point), N.x^2 + (N.y - 3)^2 = 8 ∧ N.y ≠ 1 ∧
    (N.x - M.x) * (B.x - A.x) + (N.y - M.y) * (B.y - A.y) = 0) :=
sorry

end NUMINAMATH_CALUDE_parabola_circle_intersection_l3650_365057


namespace NUMINAMATH_CALUDE_fraction_value_l3650_365069

theorem fraction_value : (900 ^ 2 : ℚ) / (153 ^ 2 - 147 ^ 2) = 450 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l3650_365069


namespace NUMINAMATH_CALUDE_min_value_complex_l3650_365013

theorem min_value_complex (z : ℂ) (h : Complex.abs z = 2) :
  ∃ (min_val : ℝ), min_val = Real.sqrt (13 + 6 * Real.sqrt 7) ∧
    ∀ w : ℂ, Complex.abs w = 2 → Complex.abs (w + 3 - 4 * Complex.I) ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_complex_l3650_365013


namespace NUMINAMATH_CALUDE_total_cost_pants_and_belt_l3650_365099

theorem total_cost_pants_and_belt (pants_price belt_price total_cost : ℝ) :
  pants_price = 34 →
  pants_price = belt_price - 2.93 →
  total_cost = pants_price + belt_price →
  total_cost = 70.93 := by
sorry

end NUMINAMATH_CALUDE_total_cost_pants_and_belt_l3650_365099


namespace NUMINAMATH_CALUDE_opposite_of_2023_l3650_365059

theorem opposite_of_2023 : 
  ∀ x : ℤ, x + 2023 = 0 → x = -2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l3650_365059


namespace NUMINAMATH_CALUDE_jills_lavender_candles_l3650_365068

/-- Represents the number of candles of each scent Jill made -/
structure CandleCounts where
  lavender : ℕ
  coconut : ℕ
  almond : ℕ
  jasmine : ℕ

/-- Represents the amount of scent (in ml) required for each type of candle -/
def scentAmounts : CandleCounts where
  lavender := 10
  coconut := 8
  almond := 12
  jasmine := 9

/-- The total number of almond candles Jill made -/
def totalAlmondCandles : ℕ := 12

/-- The ratio of coconut scent to almond scent Jill had -/
def coconutToAlmondRatio : ℚ := 5/2

theorem jills_lavender_candles (counts : CandleCounts) : counts.lavender = 135 :=
  by
  have h1 : counts.lavender = 3 * counts.coconut := by sorry
  have h2 : counts.almond = 2 * counts.jasmine := by sorry
  have h3 : counts.almond = totalAlmondCandles := by sorry
  have h4 : counts.coconut * scentAmounts.coconut = 
            coconutToAlmondRatio * (counts.almond * scentAmounts.almond) := by sorry
  have h5 : counts.jasmine * scentAmounts.jasmine = 
            counts.jasmine * scentAmounts.jasmine := by sorry
  sorry

end NUMINAMATH_CALUDE_jills_lavender_candles_l3650_365068


namespace NUMINAMATH_CALUDE_part_one_part_two_l3650_365020

-- Define propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := (x - 3) / (2 - x) ≥ 0

-- Part 1
theorem part_one (x : ℝ) (h : p x 1 ∧ q x) : 2 < x ∧ x < 3 := by sorry

-- Part 2
theorem part_two (a : ℝ) (h : ∀ x, ¬(p x a) → ¬(q x)) 
  (h_not_necessary : ∃ x, q x ∧ p x a) : 1 < a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3650_365020


namespace NUMINAMATH_CALUDE_pure_imaginary_iff_real_zero_imag_nonzero_l3650_365083

/-- A complex number is pure imaginary if and only if its real part is zero and its imaginary part is non-zero -/
theorem pure_imaginary_iff_real_zero_imag_nonzero (z : ℂ) :
  (∃ b : ℝ, b ≠ 0 ∧ z = Complex.I * b) ↔ (z.re = 0 ∧ z.im ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_iff_real_zero_imag_nonzero_l3650_365083


namespace NUMINAMATH_CALUDE_no_m_exists_for_subset_l3650_365040

theorem no_m_exists_for_subset : ¬ ∃ m : ℝ, m > 1 ∧ ∀ x : ℝ, -3 ≤ x ∧ x ≤ 4 → 1 - m ≤ x ∧ x ≤ 3 * m - 2 := by
  sorry

end NUMINAMATH_CALUDE_no_m_exists_for_subset_l3650_365040


namespace NUMINAMATH_CALUDE_rational_function_inequality_l3650_365027

theorem rational_function_inequality (f : ℚ → ℤ) :
  ∃ a b : ℚ, (f a + f b : ℚ) / 2 ≤ f ((a + b) / 2) := by
  sorry

end NUMINAMATH_CALUDE_rational_function_inequality_l3650_365027


namespace NUMINAMATH_CALUDE_inequality_solution_interval_l3650_365084

theorem inequality_solution_interval (x : ℝ) : 
  (1 / (x^2 + 1) > 3 / x + 13 / 10) ↔ -2 < x ∧ x < 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_interval_l3650_365084


namespace NUMINAMATH_CALUDE_smallest_k_for_polynomial_division_l3650_365050

theorem smallest_k_for_polynomial_division : 
  ∃ (k : ℕ), k > 0 ∧ 
  (∀ (z : ℂ), (z^10 + z^9 + z^6 + z^5 + z^4 + z + 1) ∣ (z^k - 1)) ∧
  (∀ (m : ℕ), m > 0 → m < k → 
    ¬(∀ (z : ℂ), (z^10 + z^9 + z^6 + z^5 + z^4 + z + 1) ∣ (z^m - 1))) ∧
  k = 84 :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_for_polynomial_division_l3650_365050


namespace NUMINAMATH_CALUDE_local_minimum_implies_c_equals_2_l3650_365024

/-- The function f(x) = x(x-c)^2 has a local minimum at x=2 -/
def has_local_minimum_at_2 (c : ℝ) : Prop :=
  ∃ δ > 0, ∀ x, |x - 2| < δ → x * (x - c)^2 ≥ 2 * (2 - c)^2

theorem local_minimum_implies_c_equals_2 :
  ∀ c : ℝ, has_local_minimum_at_2 c → c = 2 :=
sorry

end NUMINAMATH_CALUDE_local_minimum_implies_c_equals_2_l3650_365024


namespace NUMINAMATH_CALUDE_bottle_count_theorem_l3650_365078

/-- Represents the number of bottles for each team and the total filled -/
structure BottleCount where
  total : Nat
  football : Nat
  soccer : Nat
  lacrosse : Nat
  rugby : Nat
  unaccounted : Nat

/-- The given conditions and the statement to prove -/
theorem bottle_count_theorem (bc : BottleCount) : 
  bc.total = 254 ∧ 
  bc.football = 11 * 6 ∧ 
  bc.soccer = 53 ∧ 
  bc.lacrosse = bc.football + 12 ∧ 
  bc.rugby = 49 → 
  bc.total = bc.football + bc.soccer + bc.lacrosse + bc.rugby + bc.unaccounted :=
by sorry

end NUMINAMATH_CALUDE_bottle_count_theorem_l3650_365078


namespace NUMINAMATH_CALUDE_expression_evaluation_l3650_365054

theorem expression_evaluation (x y : ℕ) (h1 : x = 3) (h2 : y = 2) :
  5 * x^(y + 1) + 6 * y^(x + 1) = 231 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3650_365054


namespace NUMINAMATH_CALUDE_transaction_gain_per_year_l3650_365015

/-- Calculate simple interest -/
def simpleInterest (principal rate time : ℚ) : ℚ :=
  (principal * rate * time) / 100

theorem transaction_gain_per_year 
  (principal : ℚ) 
  (borrowRate lendRate : ℚ) 
  (time : ℚ) 
  (h1 : principal = 5000)
  (h2 : borrowRate = 4)
  (h3 : lendRate = 6)
  (h4 : time = 2) :
  (simpleInterest principal lendRate time - simpleInterest principal borrowRate time) / time = 200 := by
  sorry

end NUMINAMATH_CALUDE_transaction_gain_per_year_l3650_365015


namespace NUMINAMATH_CALUDE_clock_strikes_count_l3650_365034

/-- Calculates the number of clock strikes in a 24-hour period -/
def clock_strikes : ℕ :=
  -- Strikes at whole hours: sum of 1 to 12, twice (for AM and PM)
  2 * (List.range 12).sum
  -- Strikes at half hours: 24 (once every half hour)
  + 24

/-- Theorem stating that the clock strikes 180 times in a 24-hour period -/
theorem clock_strikes_count : clock_strikes = 180 := by
  sorry

end NUMINAMATH_CALUDE_clock_strikes_count_l3650_365034


namespace NUMINAMATH_CALUDE_four_thirds_of_product_l3650_365023

theorem four_thirds_of_product (a b : ℚ) (ha : a = 15/4) (hb : b = 5/2) : 
  (4/3 : ℚ) * (a * b) = 25/2 := by
  sorry

end NUMINAMATH_CALUDE_four_thirds_of_product_l3650_365023


namespace NUMINAMATH_CALUDE_jump_rope_time_difference_l3650_365058

/-- Given jump rope times for Cindy, Betsy, and Tina, prove that Tina can jump 6 minutes longer than Cindy -/
theorem jump_rope_time_difference (cindy betsy tina : ℕ) 
  (h1 : cindy = 12)
  (h2 : betsy = cindy / 2)
  (h3 : tina = 3 * betsy) :
  tina - cindy = 6 := by
  sorry

end NUMINAMATH_CALUDE_jump_rope_time_difference_l3650_365058


namespace NUMINAMATH_CALUDE_not_proportional_D_l3650_365067

/-- Represents a relation between x and y --/
inductive Relation
  | DirectlyProportional
  | InverselyProportional
  | Neither

/-- Determines the type of relation between x and y given an equation --/
def determineRelation (equation : ℝ → ℝ → Prop) : Relation :=
  sorry

/-- The equation x + y = 0 --/
def equationA (x y : ℝ) : Prop := x + y = 0

/-- The equation 3xy = 10 --/
def equationB (x y : ℝ) : Prop := 3 * x * y = 10

/-- The equation x = 5y --/
def equationC (x y : ℝ) : Prop := x = 5 * y

/-- The equation 3x + y = 10 --/
def equationD (x y : ℝ) : Prop := 3 * x + y = 10

/-- The equation x/y = √3 --/
def equationE (x y : ℝ) : Prop := x / y = Real.sqrt 3

theorem not_proportional_D :
  determineRelation equationD = Relation.Neither ∧
  determineRelation equationA ≠ Relation.Neither ∧
  determineRelation equationB ≠ Relation.Neither ∧
  determineRelation equationC ≠ Relation.Neither ∧
  determineRelation equationE ≠ Relation.Neither :=
  sorry

end NUMINAMATH_CALUDE_not_proportional_D_l3650_365067


namespace NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l3650_365072

theorem arithmetic_expression_evaluation : 4 * 12 + 5 * 11 + 6^2 + 7 * 9 = 202 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l3650_365072


namespace NUMINAMATH_CALUDE_sphere_diameter_triple_volume_l3650_365045

theorem sphere_diameter_triple_volume (π : ℝ) (h_π : π > 0) : 
  let r₁ : ℝ := 6
  let V₁ : ℝ := (4/3) * π * r₁^3
  let V₂ : ℝ := 3 * V₁
  let r₂ : ℝ := (V₂ / ((4/3) * π))^(1/3)
  2 * r₂ = 12 * (2 : ℝ)^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_sphere_diameter_triple_volume_l3650_365045


namespace NUMINAMATH_CALUDE_cyclic_difference_fourth_power_sum_l3650_365086

theorem cyclic_difference_fourth_power_sum (a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℕ) 
  (h_distinct : a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₁ ≠ a₄ ∧ a₁ ≠ a₅ ∧ a₁ ≠ a₆ ∧ a₁ ≠ a₇ ∧
                a₂ ≠ a₃ ∧ a₂ ≠ a₄ ∧ a₂ ≠ a₅ ∧ a₂ ≠ a₆ ∧ a₂ ≠ a₇ ∧
                a₃ ≠ a₄ ∧ a₃ ≠ a₅ ∧ a₃ ≠ a₆ ∧ a₃ ≠ a₇ ∧
                a₄ ≠ a₅ ∧ a₄ ≠ a₆ ∧ a₄ ≠ a₇ ∧
                a₅ ≠ a₆ ∧ a₅ ≠ a₇ ∧
                a₆ ≠ a₇) :
  (a₁ - a₂)^4 + (a₂ - a₃)^4 + (a₃ - a₄)^4 + (a₄ - a₅)^4 + 
  (a₅ - a₆)^4 + (a₆ - a₇)^4 + (a₇ - a₁)^4 ≥ 82 :=
by sorry

end NUMINAMATH_CALUDE_cyclic_difference_fourth_power_sum_l3650_365086


namespace NUMINAMATH_CALUDE_convex_quadrilateral_from_circles_in_square_l3650_365037

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a square -/
structure Square where
  sideLength : ℝ

/-- Predicate to check if a point is inside a circle -/
def isInsideCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 < c.radius^2

/-- Theorem statement -/
theorem convex_quadrilateral_from_circles_in_square 
  (s : Square) 
  (c1 c2 c3 c4 : Circle) 
  (p1 p2 p3 p4 : Point) : 
  -- The circles are centered at the vertices of the square
  (c1.center.x = 0 ∧ c1.center.y = 0) →
  (c2.center.x = s.sideLength ∧ c2.center.y = 0) →
  (c3.center.x = s.sideLength ∧ c3.center.y = s.sideLength) →
  (c4.center.x = 0 ∧ c4.center.y = s.sideLength) →
  -- The sum of the areas of the circles equals the area of the square
  (π * (c1.radius^2 + c2.radius^2 + c3.radius^2 + c4.radius^2) = s.sideLength^2) →
  -- The points are inside their respective circles
  isInsideCircle p1 c1 →
  isInsideCircle p2 c2 →
  isInsideCircle p3 c3 →
  isInsideCircle p4 c4 →
  -- The four points form a convex quadrilateral
  ∃ (a b c : ℝ), a * p1.x + b * p1.y + c < 0 ∧
                 a * p2.x + b * p2.y + c < 0 ∧
                 a * p3.x + b * p3.y + c > 0 ∧
                 a * p4.x + b * p4.y + c > 0 :=
by sorry


end NUMINAMATH_CALUDE_convex_quadrilateral_from_circles_in_square_l3650_365037


namespace NUMINAMATH_CALUDE_geometric_progression_first_term_l3650_365019

theorem geometric_progression_first_term 
  (S : ℝ) 
  (sum_first_two : ℝ) 
  (h1 : S = 6) 
  (h2 : sum_first_two = 8/3) :
  ∃ (a : ℝ), (a = 6 + 2 * Real.sqrt 5 ∨ a = 6 - 2 * Real.sqrt 5) ∧ 
  (∃ (r : ℝ), S = a / (1 - r) ∧ sum_first_two = a * (1 + r)) := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_first_term_l3650_365019


namespace NUMINAMATH_CALUDE_root_of_unity_sum_iff_cube_root_l3650_365053

theorem root_of_unity_sum_iff_cube_root (x y : ℂ) : 
  (Complex.abs x = 1 ∧ Complex.abs y = 1 ∧ x ≠ y) → 
  (Complex.abs (x + y) = 1 ↔ (y / x) ^ 3 = 1) := by
  sorry

end NUMINAMATH_CALUDE_root_of_unity_sum_iff_cube_root_l3650_365053


namespace NUMINAMATH_CALUDE_tan_30_degrees_l3650_365071

theorem tan_30_degrees : Real.tan (30 * π / 180) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_30_degrees_l3650_365071


namespace NUMINAMATH_CALUDE_line_slope_point_sum_l3650_365044

/-- Theorem: For a line with slope 8 passing through (-2, 4), m + b = 28 -/
theorem line_slope_point_sum (m b : ℝ) : 
  m = 8 → -- The slope is 8
  4 = 8 * (-2) + b → -- The line passes through (-2, 4)
  m + b = 28 := by sorry

end NUMINAMATH_CALUDE_line_slope_point_sum_l3650_365044


namespace NUMINAMATH_CALUDE_smallest_value_for_x_5_l3650_365033

theorem smallest_value_for_x_5 (x : ℝ) (h : x = 5) :
  let a := 8 / x
  let b := 8 / (x + 2)
  let c := 8 / (x - 2)
  let d := x / 8
  let e := (x + 2) / 8
  d ≤ a ∧ d ≤ b ∧ d ≤ c ∧ d ≤ e := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_for_x_5_l3650_365033


namespace NUMINAMATH_CALUDE_exists_quadrilateral_no_triangle_l3650_365081

/-- A convex quadrilateral with angles α, β, γ, and δ (in degrees) -/
structure ConvexQuadrilateral where
  α : ℝ
  β : ℝ
  γ : ℝ
  δ : ℝ
  sum_360 : α + β + γ + δ = 360
  all_positive : 0 < α ∧ 0 < β ∧ 0 < γ ∧ 0 < δ
  all_less_180 : α < 180 ∧ β < 180 ∧ γ < 180 ∧ δ < 180

/-- Check if three real numbers can form the sides of a triangle -/
def canFormTriangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem stating that there exists a convex quadrilateral where no three angles can form a triangle -/
theorem exists_quadrilateral_no_triangle : ∃ q : ConvexQuadrilateral, 
  ¬(canFormTriangle q.α q.β q.γ ∨ 
    canFormTriangle q.α q.β q.δ ∨ 
    canFormTriangle q.α q.γ q.δ ∨ 
    canFormTriangle q.β q.γ q.δ) := by
  sorry

end NUMINAMATH_CALUDE_exists_quadrilateral_no_triangle_l3650_365081


namespace NUMINAMATH_CALUDE_vector_sum_scalar_multiple_l3650_365008

/-- Given vectors a and b in ℝ², prove that a + 2b equals the expected result. -/
theorem vector_sum_scalar_multiple (a b : ℝ × ℝ) (h1 : a = (1, 2)) (h2 : b = (-2, 1)) :
  a + 2 • b = (-3, 4) := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_scalar_multiple_l3650_365008


namespace NUMINAMATH_CALUDE_chicken_salad_cost_l3650_365098

/-- Given the following conditions:
  - Lee and his friend had a total of $18
  - Chicken wings cost $6
  - Two sodas cost $2 in total
  - The tax was $3
  - They received $3 in change
  Prove that the chicken salad cost $4 -/
theorem chicken_salad_cost (total_money : ℕ) (wings_cost : ℕ) (sodas_cost : ℕ) (tax : ℕ) (change : ℕ) :
  total_money = 18 →
  wings_cost = 6 →
  sodas_cost = 2 →
  tax = 3 →
  change = 3 →
  total_money - change - (wings_cost + sodas_cost + tax) = 4 :=
by sorry

end NUMINAMATH_CALUDE_chicken_salad_cost_l3650_365098


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3650_365021

def A : Set Nat := {1,2,3,4,5}
def B : Set Nat := {2,4,6,8,10}

theorem union_of_A_and_B : A ∪ B = {1,2,3,4,5,6,8,10} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3650_365021


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3650_365016

theorem complex_equation_solution (z : ℂ) : 
  (Complex.I * z = Complex.I + z) → z = (1 - Complex.I) / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3650_365016


namespace NUMINAMATH_CALUDE_supremum_of_function_l3650_365025

theorem supremum_of_function (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  ∃ M : ℝ, (∀ x : ℝ, -1/(2*x) - 2/((1-x)) ≤ M) ∧ 
  (∀ ε > 0, ∃ y : ℝ, -1/(2*y) - 2/((1-y)) > M - ε) ∧
  M = -9/2 := by
  sorry

end NUMINAMATH_CALUDE_supremum_of_function_l3650_365025


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3650_365093

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x > 0, x^2 - a*x + 1 > 0) → a ∈ Set.Ioo (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3650_365093


namespace NUMINAMATH_CALUDE_right_triangle_height_radius_ratio_l3650_365051

theorem right_triangle_height_radius_ratio (a b c h r : ℝ) :
  a > 0 → b > 0 → c > 0 → h > 0 → r > 0 →
  a^2 + b^2 = c^2 →  -- Right triangle condition
  (a + b + c) * r = c * h →  -- Area equality condition
  2 < h / r ∧ h / r ≤ 1 + Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_height_radius_ratio_l3650_365051


namespace NUMINAMATH_CALUDE_norbs_age_l3650_365007

def guesses : List Nat := [25, 29, 31, 33, 37, 39, 42, 45, 48, 50]

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m > 1 → m < n → ¬(n % m = 0)

def count_low_guesses (age : Nat) : Nat :=
  (guesses.filter (· < age)).length

def count_off_by_one (age : Nat) : Nat :=
  (guesses.filter (λ g => g = age - 1 ∨ g = age + 1)).length

theorem norbs_age :
  ∃ (age : Nat),
    age ∈ guesses ∧
    is_prime age ∧
    count_low_guesses age < (2 * guesses.length) / 3 ∧
    count_off_by_one age = 2 ∧
    age = 29 :=
  sorry

end NUMINAMATH_CALUDE_norbs_age_l3650_365007


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3650_365004

theorem inequality_solution_set (m : ℝ) : 
  (∃ (a b c : ℤ), (∀ x : ℝ, (x^2 - 2*x + m ≤ 0) ↔ (x = a ∨ x = b ∨ x = c)) ∧ 
   (∀ y : ℤ, (y^2 - 2*y + m ≤ 0) → (y = a ∨ y = b ∨ y = c))) ↔ 
  (m = -2 ∨ m = 0) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3650_365004


namespace NUMINAMATH_CALUDE_largest_of_five_consecutive_sum_180_l3650_365043

theorem largest_of_five_consecutive_sum_180 (a : ℕ) :
  (∃ (x : ℕ), x = a ∧ 
    x + (x + 1) + (x + 2) + (x + 3) + (x + 4) = 180) →
  a + 4 = 38 :=
by sorry

end NUMINAMATH_CALUDE_largest_of_five_consecutive_sum_180_l3650_365043


namespace NUMINAMATH_CALUDE_area_of_parallelogram_EFGH_l3650_365014

/-- The area of parallelogram EFGH --/
def area_EFGH : ℝ := 15

/-- The base of parallelogram EFGH --/
def base_FG : ℝ := 3

/-- The height from point E to line FG --/
def height_E_to_FG : ℝ := 5

/-- The theorem stating that the area of parallelogram EFGH is 15 square units --/
theorem area_of_parallelogram_EFGH :
  area_EFGH = base_FG * height_E_to_FG :=
by sorry

end NUMINAMATH_CALUDE_area_of_parallelogram_EFGH_l3650_365014


namespace NUMINAMATH_CALUDE_obtuse_triangle_partition_l3650_365062

/-- A triple of positive integers forming an obtuse triangle -/
structure ObtuseTriple where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  h1 : a < b
  h2 : b < c
  h3 : a + b > c
  h4 : a * a + b * b < c * c

/-- The set of integers from 2 to 3n+1 -/
def triangleSet (n : ℕ+) : Set ℕ+ :=
  {k | 2 ≤ k ∧ k ≤ 3*n+1}

/-- A partition of the triangle set into n obtuse triples -/
def ObtusePartition (n : ℕ+) : Type :=
  { partition : Finset (Finset ℕ+) //
    partition.card = n ∧
    (∀ s ∈ partition, ∃ t : ObtuseTriple, (↑s : Set ℕ+) = {t.a, t.b, t.c}) ∧
    (⋃ (s ∈ partition), (↑s : Set ℕ+)) = triangleSet n }

/-- The main theorem -/
theorem obtuse_triangle_partition (n : ℕ+) :
  ∃ p : ObtusePartition n, True := by sorry

end NUMINAMATH_CALUDE_obtuse_triangle_partition_l3650_365062


namespace NUMINAMATH_CALUDE_jenga_remaining_blocks_l3650_365095

/-- Represents a Jenga game state -/
structure JengaGame where
  initialBlocks : ℕ
  players : ℕ
  completeRounds : ℕ
  extraBlocksRemoved : ℕ

/-- Calculates the number of blocks remaining before the last player's turn -/
def remainingBlocks (game : JengaGame) : ℕ :=
  game.initialBlocks - (game.players * game.completeRounds + game.extraBlocksRemoved)

/-- Theorem stating the number of blocks remaining in the specific Jenga game scenario -/
theorem jenga_remaining_blocks :
  let game : JengaGame := {
    initialBlocks := 54,
    players := 5,
    completeRounds := 5,
    extraBlocksRemoved := 1
  }
  remainingBlocks game = 28 := by sorry

end NUMINAMATH_CALUDE_jenga_remaining_blocks_l3650_365095


namespace NUMINAMATH_CALUDE_number_problem_l3650_365012

theorem number_problem (x : ℝ) : (x - 14) / 10 = 4 → (x - 5) / 7 = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3650_365012


namespace NUMINAMATH_CALUDE_marshmallow_challenge_l3650_365032

/-- The marshmallow challenge problem -/
theorem marshmallow_challenge 
  (haley : ℕ) 
  (michael : ℕ) 
  (brandon : ℕ) 
  (h1 : haley = 8)
  (h2 : michael = 3 * haley)
  (h3 : haley + michael + brandon = 44) :
  brandon / michael = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_marshmallow_challenge_l3650_365032


namespace NUMINAMATH_CALUDE_hair_extension_length_l3650_365073

def original_length : ℕ := 18

def extension_factor : ℕ := 2

theorem hair_extension_length : 
  original_length * extension_factor = 36 := by sorry

end NUMINAMATH_CALUDE_hair_extension_length_l3650_365073


namespace NUMINAMATH_CALUDE_triangle_property_l3650_365091

open Real

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given condition
def satisfiesCondition (t : Triangle) : Prop :=
  2 * t.a * sin t.A = (2 * t.b + t.c) * sin t.B + (2 * t.c + t.b) * sin t.C

-- Define the perimeter of the triangle
def perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

-- Theorem statement
theorem triangle_property (t : Triangle) 
  (h : satisfiesCondition t) : 
  t.A = 2 * π / 3 ∧ 
  (t.a = 2 → 4 < perimeter t ∧ perimeter t ≤ 2 + 4 * Real.sqrt 3 / 3) :=
sorry

end NUMINAMATH_CALUDE_triangle_property_l3650_365091


namespace NUMINAMATH_CALUDE_line_m_equation_l3650_365000

/-- Two distinct lines in the xy-plane that intersect at the origin -/
structure IntersectingLines where
  ℓ : Set (ℝ × ℝ)
  m : Set (ℝ × ℝ)
  distinct : ℓ ≠ m
  intersect_origin : (0, 0) ∈ ℓ ∩ m

/-- Reflection of a point about a line -/
def reflect (p : ℝ × ℝ) (line : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

/-- The equation of line ℓ is 2x - y = 0 -/
def line_ℓ_eq (p : ℝ × ℝ) : Prop := 2 * p.1 - p.2 = 0

theorem line_m_equation (lines : IntersectingLines) 
  (h_ℓ_eq : ∀ p ∈ lines.ℓ, line_ℓ_eq p)
  (h_Q : reflect (reflect (-2, 3) lines.ℓ) lines.m = (3, -1)) :
  ∀ p ∈ lines.m, 3 * p.1 + p.2 = 0 := by sorry

end NUMINAMATH_CALUDE_line_m_equation_l3650_365000


namespace NUMINAMATH_CALUDE_expression_value_l3650_365049

theorem expression_value (x y z : ℤ) (hx : x = -5) (hy : y = 8) (hz : z = 3) :
  2 * (x - y)^2 - x^3 * y + z^4 * y^2 - x^2 * z^3 = 5847 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3650_365049
