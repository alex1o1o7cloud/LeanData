import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l419_41986

/-- The smallest positive integer a such that 450 * a is a perfect square -/
def a : ℕ := 2

/-- The smallest positive integer b such that 450 * b is a perfect cube -/
def b : ℕ := 60

/-- 450 * a is a perfect square -/
axiom h1 : ∃ n : ℕ, 450 * a = n^2

/-- 450 * b is a perfect cube -/
axiom h2 : ∃ n : ℕ, 450 * b = n^3

/-- a is the smallest positive integer satisfying the square condition -/
axiom h3 : ∀ x : ℕ, 0 < x → x < a → ¬∃ n : ℕ, 450 * x = n^2

/-- b is the smallest positive integer satisfying the cube condition -/
axiom h4 : ∀ x : ℕ, 0 < x → x < b → ¬∃ n : ℕ, 450 * x = n^3

theorem sum_of_a_and_b : a + b = 62 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l419_41986


namespace NUMINAMATH_CALUDE_mike_total_score_l419_41973

/-- Given that Mike played six games of basketball and scored four points in each game,
    prove that his total score is 24 points. -/
theorem mike_total_score :
  let games_played : ℕ := 6
  let points_per_game : ℕ := 4
  let total_score := games_played * points_per_game
  total_score = 24 := by sorry

end NUMINAMATH_CALUDE_mike_total_score_l419_41973


namespace NUMINAMATH_CALUDE_distance_origin_to_line_l419_41929

/-- The distance from the origin to the line 4x + 3y - 12 = 0 is 12/5 -/
theorem distance_origin_to_line : 
  let line := {(x, y) : ℝ × ℝ | 4 * x + 3 * y - 12 = 0}
  ∃ d : ℝ, d = 12/5 ∧ ∀ (p : ℝ × ℝ), p ∈ line → Real.sqrt ((p.1)^2 + (p.2)^2) ≥ d := by
  sorry

end NUMINAMATH_CALUDE_distance_origin_to_line_l419_41929


namespace NUMINAMATH_CALUDE_fraction_not_simplifiable_l419_41972

theorem fraction_not_simplifiable (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_not_simplifiable_l419_41972


namespace NUMINAMATH_CALUDE_distance_from_y_axis_is_18_l419_41967

def point_P (x : ℝ) : ℝ × ℝ := (x, -9)

def distance_to_x_axis (p : ℝ × ℝ) : ℝ := |p.2|

def distance_to_y_axis (p : ℝ × ℝ) : ℝ := |p.1|

theorem distance_from_y_axis_is_18 (x : ℝ) :
  let p := point_P x
  distance_to_x_axis p = (1/2) * distance_to_y_axis p →
  distance_to_y_axis p = 18 := by sorry

end NUMINAMATH_CALUDE_distance_from_y_axis_is_18_l419_41967


namespace NUMINAMATH_CALUDE_race_distance_l419_41958

/-- Race conditions and proof of distance -/
theorem race_distance (d x y z : ℝ) 
  (h1 : d / x = (d - 25) / y)  -- X beats Y by 25 meters
  (h2 : d / y = (d - 15) / z)  -- Y beats Z by 15 meters
  (h3 : d / x = (d - 37) / z)  -- X beats Z by 37 meters
  (h4 : d > 0) : d = 125 := by
  sorry

end NUMINAMATH_CALUDE_race_distance_l419_41958


namespace NUMINAMATH_CALUDE_product_of_sums_zero_l419_41951

theorem product_of_sums_zero (x y z w : ℝ) 
  (sum_zero : x + y + z + w = 0)
  (sum_seventh_power_zero : x^7 + y^7 + z^7 + w^7 = 0) : 
  w * (w + x) * (w + y) * (w + z) = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_zero_l419_41951


namespace NUMINAMATH_CALUDE_square_root_reverses_squaring_l419_41920

theorem square_root_reverses_squaring (x : ℝ) (hx : x = 25) : 
  Real.sqrt (x ^ 2) = x := by sorry

end NUMINAMATH_CALUDE_square_root_reverses_squaring_l419_41920


namespace NUMINAMATH_CALUDE_integer_solutions_of_inequalities_l419_41905

theorem integer_solutions_of_inequalities :
  let S : Set ℤ := {x | (2 + x : ℝ) > (7 - 4*x) ∧ (x : ℝ) < ((4 + x) / 2)}
  S = {2, 3} := by sorry

end NUMINAMATH_CALUDE_integer_solutions_of_inequalities_l419_41905


namespace NUMINAMATH_CALUDE_remaining_amount_is_10_95_l419_41922

def initial_amount : ℝ := 60

def frame_price : ℝ := 15
def frame_discount : ℝ := 0.1

def wheel_price : ℝ := 25
def wheel_discount : ℝ := 0.05

def seat_price : ℝ := 8
def seat_discount : ℝ := 0.15

def tape_price : ℝ := 5
def tape_discount : ℝ := 0

def discounted_price (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

def total_cost : ℝ :=
  discounted_price frame_price frame_discount +
  discounted_price wheel_price wheel_discount +
  discounted_price seat_price seat_discount +
  discounted_price tape_price tape_discount

theorem remaining_amount_is_10_95 :
  initial_amount - total_cost = 10.95 :=
by sorry

end NUMINAMATH_CALUDE_remaining_amount_is_10_95_l419_41922


namespace NUMINAMATH_CALUDE_min_distance_circle_point_l419_41944

noncomputable section

-- Define the circle C
def circle_C : Set (ℝ × ℝ) :=
  {p | (p.1 - 1)^2 + (p.2 + 1)^2 = 4}

-- Define point Q
def point_Q : ℝ × ℝ := (Real.sqrt 2, -Real.sqrt 2)

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem min_distance_circle_point :
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 2 ∧
  ∀ (P : ℝ × ℝ), P ∈ circle_C →
  distance P point_Q ≥ min_dist :=
sorry

end

end NUMINAMATH_CALUDE_min_distance_circle_point_l419_41944


namespace NUMINAMATH_CALUDE_vector_magnitude_l419_41961

/-- Given two vectors a and b in ℝ², prove that the magnitude of b is 5 -/
theorem vector_magnitude (a b : ℝ × ℝ) : 
  a = (3, -2) → a + b = (0, 2) → ‖b‖ = 5 := by sorry

end NUMINAMATH_CALUDE_vector_magnitude_l419_41961


namespace NUMINAMATH_CALUDE_square_difference_equality_l419_41943

theorem square_difference_equality : 1010^2 - 990^2 - 1005^2 + 995^2 = 20000 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l419_41943


namespace NUMINAMATH_CALUDE_sara_marbles_l419_41988

/-- Given that Sara has 10 marbles initially and loses 7 marbles, prove that she will have 3 marbles left. -/
theorem sara_marbles (initial_marbles : ℕ) (lost_marbles : ℕ) (h1 : initial_marbles = 10) (h2 : lost_marbles = 7) :
  initial_marbles - lost_marbles = 3 := by
  sorry

end NUMINAMATH_CALUDE_sara_marbles_l419_41988


namespace NUMINAMATH_CALUDE_solve_equation_l419_41962

theorem solve_equation : ∃ x : ℝ, (5*x + 9*x = 350 - 10*(x - 4)) ∧ x = 16.25 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l419_41962


namespace NUMINAMATH_CALUDE_exist_distinct_indices_with_difference_not_t_l419_41906

theorem exist_distinct_indices_with_difference_not_t 
  (n : ℕ+) (t : ℝ) (ht : t ≠ 0) (a : Fin (2*n - 1) → ℝ) :
  ∃ (s : Finset (Fin (2*n - 1))), 
    s.card = n ∧ 
    ∀ (i j : Fin n), i ≠ j → 
      ∃ (x y : Fin (2*n - 1)), x ∈ s ∧ y ∈ s ∧ a x - a y ≠ t :=
by sorry

end NUMINAMATH_CALUDE_exist_distinct_indices_with_difference_not_t_l419_41906


namespace NUMINAMATH_CALUDE_sqrt_21_is_11th_term_l419_41945

theorem sqrt_21_is_11th_term (a : ℕ → ℝ) :
  (∀ n, a n = Real.sqrt (2 * n - 1)) →
  a 11 = Real.sqrt 21 :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_21_is_11th_term_l419_41945


namespace NUMINAMATH_CALUDE_max_sum_of_squares_of_roots_l419_41940

theorem max_sum_of_squares_of_roots (k : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x : ℝ, x^2 - (k-2)*x + (k^2+3*k+5) = 0 ↔ x = x₁ ∨ x = x₂) →
  (∃ k : ℝ, x₁^2 + x₂^2 = 18) ∧
  (∀ k : ℝ, x₁^2 + x₂^2 ≤ 18) :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_of_roots_l419_41940


namespace NUMINAMATH_CALUDE_simplify_expression_l419_41916

theorem simplify_expression (k : ℝ) (h : k ≠ 0) :
  ∃ (a b c : ℤ), (8 * k + 3 + 6 * k^2) + (5 * k^2 + 4 * k + 7) = a * k^2 + b * k + c ∧ a + b + c = 33 :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l419_41916


namespace NUMINAMATH_CALUDE_total_triangles_is_seventeen_l419_41957

/-- Represents a 2x2 square grid where each square is divided diagonally into two right-angled triangles -/
structure DiagonallyDividedGrid :=
  (size : Nat)
  (is_two_by_two : size = 2)
  (diagonally_divided : Bool)

/-- Counts the total number of triangles in the grid, including all possible combinations -/
def count_triangles (grid : DiagonallyDividedGrid) : Nat :=
  sorry

/-- Theorem stating that the total number of triangles in the described grid is 17 -/
theorem total_triangles_is_seventeen (grid : DiagonallyDividedGrid) 
  (h1 : grid.diagonally_divided = true) : 
  count_triangles grid = 17 := by
  sorry

end NUMINAMATH_CALUDE_total_triangles_is_seventeen_l419_41957


namespace NUMINAMATH_CALUDE_a_share_is_3630_l419_41910

/-- Calculates the share of profit for an investor in a partnership business. -/
def calculate_share_of_profit (investment_a investment_b investment_c total_profit : ℚ) : ℚ :=
  let total_investment := investment_a + investment_b + investment_c
  let ratio_a := investment_a / total_investment
  ratio_a * total_profit

/-- Theorem stating that A's share of the profit is 3630 given the investments and total profit. -/
theorem a_share_is_3630 :
  calculate_share_of_profit 6300 4200 10500 12100 = 3630 := by
  sorry

end NUMINAMATH_CALUDE_a_share_is_3630_l419_41910


namespace NUMINAMATH_CALUDE_cos_2alpha_eq_neg_four_fifths_l419_41923

theorem cos_2alpha_eq_neg_four_fifths (α : Real) 
  (h : (Real.tan α + 1) / (Real.tan α - 1) = 2) : 
  Real.cos (2 * α) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_eq_neg_four_fifths_l419_41923


namespace NUMINAMATH_CALUDE_max_area_is_100_l419_41963

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Checks if the rectangle satisfies the given conditions -/
def isValidRectangle (r : Rectangle) : Prop :=
  r.length + r.width = 20 ∧ Even r.width

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℕ :=
  r.length * r.width

/-- Theorem: The maximum area of a valid rectangle is 100 -/
theorem max_area_is_100 :
  ∃ (r : Rectangle), isValidRectangle r ∧
    area r = 100 ∧
    ∀ (s : Rectangle), isValidRectangle s → area s ≤ 100 :=
by sorry

end NUMINAMATH_CALUDE_max_area_is_100_l419_41963


namespace NUMINAMATH_CALUDE_line_inclination_theorem_l419_41952

theorem line_inclination_theorem (a b c : ℝ) (α : ℝ) : 
  (∃ x y, a * x + b * y + c = 0) →  -- Line exists
  (Real.tan α = -a / b) →           -- Relationship between inclination angle and coefficients
  (Real.sin α + Real.cos α = 0) →   -- Given condition
  a - b = 0 := by
sorry

end NUMINAMATH_CALUDE_line_inclination_theorem_l419_41952


namespace NUMINAMATH_CALUDE_result_circle_properties_l419_41974

/-- The equation of the first given circle -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y = 0

/-- The equation of the second given circle -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - x = 0

/-- The equation of the resulting circle -/
def resultCircle (x y : ℝ) : Prop := 9*x^2 + 9*y^2 - 14*x + 4*y = 0

/-- Theorem stating that the resulting circle passes through the intersection points of the given circles and the point (1, -1) -/
theorem result_circle_properties :
  (∀ x y : ℝ, circle1 x y ∧ circle2 x y → resultCircle x y) ∧
  resultCircle 1 (-1) := by
  sorry

end NUMINAMATH_CALUDE_result_circle_properties_l419_41974


namespace NUMINAMATH_CALUDE_circle_center_correct_l419_41954

/-- The equation of a circle in the form ax² + bx + cy² + dy + e = 0 -/
structure CircleEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- The center of a circle -/
structure CircleCenter where
  x : ℝ
  y : ℝ

/-- Given a circle equation, find its center -/
def findCircleCenter (eq : CircleEquation) : CircleCenter :=
  sorry

theorem circle_center_correct :
  let eq := CircleEquation.mk 1 (-6) 1 10 (-7)
  findCircleCenter eq = CircleCenter.mk 3 (-5) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_correct_l419_41954


namespace NUMINAMATH_CALUDE_office_salary_problem_l419_41925

/-- Represents the average salary of non-officers in Rs/month -/
def average_salary_non_officers : ℝ := 110

theorem office_salary_problem (total_employees : ℕ) (officers : ℕ) (non_officers : ℕ)
  (avg_salary_all : ℝ) (avg_salary_officers : ℝ) :
  total_employees = officers + non_officers →
  total_employees = 495 →
  officers = 15 →
  non_officers = 480 →
  avg_salary_all = 120 →
  avg_salary_officers = 440 →
  average_salary_non_officers = 
    (total_employees * avg_salary_all - officers * avg_salary_officers) / non_officers :=
by sorry

end NUMINAMATH_CALUDE_office_salary_problem_l419_41925


namespace NUMINAMATH_CALUDE_four_digit_integer_problem_l419_41926

theorem four_digit_integer_problem (n : ℕ) (a b c d : ℕ) :
  n = a * 1000 + b * 100 + c * 10 + d →
  a ≥ 1 →
  a ≤ 9 →
  b ≤ 9 →
  c ≤ 9 →
  d ≤ 9 →
  a + b + c + d = 17 →
  b + c = 10 →
  a - d = 3 →
  n % 13 = 0 →
  n = 5732 := by sorry

end NUMINAMATH_CALUDE_four_digit_integer_problem_l419_41926


namespace NUMINAMATH_CALUDE_triangle_properties_l419_41933

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC exists
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π →
  -- Given conditions
  b = a * Real.cos C + (Real.sqrt 3 / 3) * c * Real.sin A →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 →
  ((1/4) * a^2 + (1/4) * b^2 - (1/4) * c^2) * 2 / a = 2 →
  -- Conclusions
  A = π/3 ∧ b = Real.sqrt 2 ∧ c = 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_properties_l419_41933


namespace NUMINAMATH_CALUDE_sunzi_suanjing_congruence_l419_41987

theorem sunzi_suanjing_congruence : ∃ (m : ℕ+), (3 ^ 20 : ℤ) ≡ 2013 [ZMOD m] := by
  sorry

end NUMINAMATH_CALUDE_sunzi_suanjing_congruence_l419_41987


namespace NUMINAMATH_CALUDE_unique_factorial_sum_l419_41985

/-- Factorial function -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Sum of factorials of digits -/
def sum_factorial_digits (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  factorial hundreds + factorial tens + factorial ones

/-- Theorem stating that 145 is the only three-digit number equal to the sum of factorials of its digits -/
theorem unique_factorial_sum :
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 → (n = sum_factorial_digits n ↔ n = 145) := by
  sorry

#eval sum_factorial_digits 145  -- Should output 145

end NUMINAMATH_CALUDE_unique_factorial_sum_l419_41985


namespace NUMINAMATH_CALUDE_max_students_distribution_l419_41927

theorem max_students_distribution (pens pencils : ℕ) (h1 : pens = 640) (h2 : pencils = 520) :
  (∃ (students : ℕ), students > 0 ∧ pens % students = 0 ∧ pencils % students = 0) ∧
  (∀ (n : ℕ), n > 0 ∧ pens % n = 0 ∧ pencils % n = 0 → n ≤ 40) ∧
  (pens % 40 = 0 ∧ pencils % 40 = 0) :=
by sorry

end NUMINAMATH_CALUDE_max_students_distribution_l419_41927


namespace NUMINAMATH_CALUDE_march_text_messages_l419_41928

def T (n : ℕ) : ℕ := ((n^2) + 1) * n.factorial

theorem march_text_messages : T 5 = 3120 := by
  sorry

end NUMINAMATH_CALUDE_march_text_messages_l419_41928


namespace NUMINAMATH_CALUDE_orange_face_probability_l419_41902

/-- Represents a die with a specific number of sides and orange faces. -/
structure Die where
  totalSides : ℕ
  orangeFaces : ℕ
  orangeFaces_le_totalSides : orangeFaces ≤ totalSides

/-- Calculates the probability of rolling an orange face on a given die. -/
def probabilityOrangeFace (d : Die) : ℚ :=
  d.orangeFaces / d.totalSides

/-- The specific 10-sided die with 4 orange faces. -/
def tenSidedDie : Die where
  totalSides := 10
  orangeFaces := 4
  orangeFaces_le_totalSides := by norm_num

/-- Theorem stating that the probability of rolling an orange face on the 10-sided die is 2/5. -/
theorem orange_face_probability :
  probabilityOrangeFace tenSidedDie = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_orange_face_probability_l419_41902


namespace NUMINAMATH_CALUDE_calculation_1_calculation_2_calculation_3_calculation_4_calculation_5_calculation_6_l419_41976

theorem calculation_1 : 320 + 16 * 27 = 752 := by sorry

theorem calculation_2 : 1500 - 125 * 8 = 500 := by sorry

theorem calculation_3 : 22 * 22 - 84 = 400 := by sorry

theorem calculation_4 : 25 * 8 * 9 = 1800 := by sorry

theorem calculation_5 : (25 + 38) * 15 = 945 := by sorry

theorem calculation_6 : (62 + 12) * 38 = 2812 := by sorry

end NUMINAMATH_CALUDE_calculation_1_calculation_2_calculation_3_calculation_4_calculation_5_calculation_6_l419_41976


namespace NUMINAMATH_CALUDE_perfect_square_condition_l419_41953

theorem perfect_square_condition (n : ℤ) :
  (∃ k : ℤ, 7 * n + 2 = k ^ 2) ↔ 
  (∃ m : ℤ, (n = 7 * m ^ 2 + 6 * m + 1) ∨ (n = 7 * m ^ 2 - 6 * m + 1)) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l419_41953


namespace NUMINAMATH_CALUDE_count_valid_pairs_l419_41994

/-- The number of distinct ordered pairs of positive integers (x,y) satisfying 1/x + 1/y = 1/5 -/
def count_pairs : ℕ := 3

/-- Predicate defining valid pairs -/
def is_valid_pair (x y : ℕ+) : Prop :=
  (1 : ℚ) / x.val + (1 : ℚ) / y.val = (1 : ℚ) / 5

theorem count_valid_pairs :
  ∃ (S : Finset (ℕ+ × ℕ+)), 
    (∀ (p : ℕ+ × ℕ+), p ∈ S ↔ is_valid_pair p.1 p.2) ∧ 
    S.card = count_pairs :=
  sorry


end NUMINAMATH_CALUDE_count_valid_pairs_l419_41994


namespace NUMINAMATH_CALUDE_quadratic_set_intersection_l419_41964

theorem quadratic_set_intersection (p q : ℝ) : 
  let A := {x : ℝ | x^2 + p*x + q = 0}
  let B := {x : ℝ | x^2 - 3*x + 2 = 0}
  (A ∩ B = A) ↔ 
  ((p^2 < 4*q) ∨ (p = -2 ∧ q = 1) ∨ (p = -4 ∧ q = 4) ∨ (p = -3 ∧ q = 2)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_set_intersection_l419_41964


namespace NUMINAMATH_CALUDE_age_ratio_l419_41993

def sachin_age : ℕ := 63
def age_difference : ℕ := 18

def rahul_age : ℕ := sachin_age + age_difference

theorem age_ratio : 
  (sachin_age : ℚ) / (rahul_age : ℚ) = 7 / 9 := by sorry

end NUMINAMATH_CALUDE_age_ratio_l419_41993


namespace NUMINAMATH_CALUDE_perimeter_of_specific_rectangle_l419_41918

/-- A figure that can be completed to form a rectangle -/
structure CompletableRectangle where
  length : ℝ
  width : ℝ

/-- The perimeter of a CompletableRectangle -/
def perimeter (r : CompletableRectangle) : ℝ :=
  2 * (r.length + r.width)

theorem perimeter_of_specific_rectangle :
  ∃ (r : CompletableRectangle), r.length = 6 ∧ r.width = 5 ∧ perimeter r = 22 :=
by sorry

end NUMINAMATH_CALUDE_perimeter_of_specific_rectangle_l419_41918


namespace NUMINAMATH_CALUDE_three_circles_arrangement_l419_41971

/-- A circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The intersection points of two circles --/
def intersectionPoints (c1 c2 : Circle) : Set (ℝ × ℝ) := sorry

/-- Three circles have only two common points --/
def haveOnlyTwoCommonPoints (c1 c2 c3 : Circle) : Prop :=
  ∃ (p q : ℝ × ℝ), p ≠ q ∧
    intersectionPoints c1 c2 ∩ intersectionPoints c2 c3 ∩ intersectionPoints c1 c3 = {p, q}

/-- All three circles intersect at the same two points --/
def allIntersectAtSamePoints (c1 c2 c3 : Circle) : Prop :=
  ∃ (p q : ℝ × ℝ), p ≠ q ∧
    intersectionPoints c1 c2 = {p, q} ∧
    intersectionPoints c2 c3 = {p, q} ∧
    intersectionPoints c1 c3 = {p, q}

/-- One circle intersects each of the other two circles at two distinct points --/
def oneIntersectsOthersAtDistinctPoints (c1 c2 c3 : Circle) : Prop :=
  ∃ (p q : ℝ × ℝ), p ≠ q ∧
    ((intersectionPoints c1 c2 = {p, q} ∧ intersectionPoints c1 c3 = {p, q}) ∨
     (intersectionPoints c2 c1 = {p, q} ∧ intersectionPoints c2 c3 = {p, q}) ∨
     (intersectionPoints c3 c1 = {p, q} ∧ intersectionPoints c3 c2 = {p, q}))

/-- The main theorem --/
theorem three_circles_arrangement (c1 c2 c3 : Circle) :
  haveOnlyTwoCommonPoints c1 c2 c3 →
  allIntersectAtSamePoints c1 c2 c3 ∨ oneIntersectsOthersAtDistinctPoints c1 c2 c3 := by
  sorry


end NUMINAMATH_CALUDE_three_circles_arrangement_l419_41971


namespace NUMINAMATH_CALUDE_ninth_minus_eighth_square_tiles_l419_41969

/-- The side length of the nth square in the sequence -/
def L (n : ℕ) : ℕ := 2 * n + 1

/-- The number of tiles in the nth square -/
def tiles (n : ℕ) : ℕ := (L n) ^ 2

theorem ninth_minus_eighth_square_tiles : tiles 9 - tiles 8 = 72 := by
  sorry

end NUMINAMATH_CALUDE_ninth_minus_eighth_square_tiles_l419_41969


namespace NUMINAMATH_CALUDE_initial_mixture_volume_l419_41931

/-- Given a mixture of milk and water with an initial ratio of 3:2, 
    prove that if 66 liters of water are added to change the ratio to 3:4, 
    the initial volume of the mixture was 165 liters. -/
theorem initial_mixture_volume 
  (initial_milk : ℝ) 
  (initial_water : ℝ) 
  (h1 : initial_milk / initial_water = 3 / 2) 
  (h2 : initial_milk / (initial_water + 66) = 3 / 4) : 
  initial_milk + initial_water = 165 := by
  sorry

#check initial_mixture_volume

end NUMINAMATH_CALUDE_initial_mixture_volume_l419_41931


namespace NUMINAMATH_CALUDE_marbles_sharing_l419_41999

theorem marbles_sharing (sienna_initial jordan_initial : ℕ)
  (h1 : sienna_initial = 150)
  (h2 : jordan_initial = 90)
  (shared : ℕ)
  (h3 : sienna_initial - shared = 3 * (jordan_initial + shared)) :
  shared = 30 := by
  sorry

end NUMINAMATH_CALUDE_marbles_sharing_l419_41999


namespace NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l419_41978

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = -3) : x^3 + 1/x^3 = -18 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l419_41978


namespace NUMINAMATH_CALUDE_student_scores_theorem_l419_41955

/-- A score is a triple of integers, each between 0 and 7 inclusive -/
def Score := { s : Fin 3 → Fin 8 // True }

/-- Given two scores, returns true if the first score is at least as high as the second for each problem -/
def ScoreGreaterEq (s1 s2 : Score) : Prop :=
  ∀ i : Fin 3, s1.val i ≥ s2.val i

theorem student_scores_theorem (scores : Fin 49 → Score) :
  ∃ i j : Fin 49, i ≠ j ∧ ScoreGreaterEq (scores i) (scores j) := by
  sorry

#check student_scores_theorem

end NUMINAMATH_CALUDE_student_scores_theorem_l419_41955


namespace NUMINAMATH_CALUDE_optimal_plan_is_most_cost_effective_l419_41914

/-- Represents a sewage treatment equipment purchase plan -/
structure PurchasePlan where
  modelA : ℕ
  modelB : ℕ

/-- Checks if a purchase plan is valid according to the given conditions -/
def isValidPlan (p : PurchasePlan) : Prop :=
  p.modelA + p.modelB = 10 ∧
  12 * p.modelA + 10 * p.modelB ≤ 105 ∧
  240 * p.modelA + 200 * p.modelB ≥ 2040

/-- Calculates the total cost of a purchase plan -/
def totalCost (p : PurchasePlan) : ℕ :=
  12 * p.modelA + 10 * p.modelB

/-- The optimal purchase plan -/
def optimalPlan : PurchasePlan :=
  { modelA := 1, modelB := 9 }

/-- Theorem stating that the optimal plan is the most cost-effective valid plan -/
theorem optimal_plan_is_most_cost_effective :
  isValidPlan optimalPlan ∧
  ∀ p : PurchasePlan, isValidPlan p → totalCost optimalPlan ≤ totalCost p :=
sorry

end NUMINAMATH_CALUDE_optimal_plan_is_most_cost_effective_l419_41914


namespace NUMINAMATH_CALUDE_cubes_with_four_neighbors_eq_108_l419_41942

/-- Represents a parallelepiped with dimensions a, b, and c. -/
structure Parallelepiped where
  a : ℕ
  b : ℕ
  c : ℕ
  h1 : a > 3
  h2 : b > 3
  h3 : c > 3
  h4 : (a - 2) * (b - 2) * (c - 2) = 429

/-- The number of unit cubes with exactly 4 neighbors in a parallelepiped. -/
def cubes_with_four_neighbors (p : Parallelepiped) : ℕ :=
  4 * ((p.a - 2) + (p.b - 2) + (p.c - 2))

theorem cubes_with_four_neighbors_eq_108 (p : Parallelepiped) :
  cubes_with_four_neighbors p = 108 := by
  sorry

end NUMINAMATH_CALUDE_cubes_with_four_neighbors_eq_108_l419_41942


namespace NUMINAMATH_CALUDE_max_soap_boxes_in_carton_l419_41900

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  width : ℕ
  length : ℕ
  height : ℕ

/-- Represents a carton with base and top dimensions -/
structure Carton where
  base : Dimensions
  top : Dimensions
  height : ℕ

/-- Represents a soap box with its dimensions and weight -/
structure SoapBox where
  dimensions : Dimensions
  weight : ℕ

def carton : Carton := {
  base := { width := 25, length := 42, height := 0 },
  top := { width := 20, length := 35, height := 0 },
  height := 60
}

def soapBox : SoapBox := {
  dimensions := { width := 7, length := 6, height := 10 },
  weight := 3
}

def maxWeight : ℕ := 150

theorem max_soap_boxes_in_carton :
  let spaceConstraint := (carton.top.width / soapBox.dimensions.width) *
                         (carton.top.length / soapBox.dimensions.length) *
                         (carton.height / soapBox.dimensions.height)
  let weightConstraint := maxWeight / soapBox.weight
  min spaceConstraint weightConstraint = 50 := by
  sorry

end NUMINAMATH_CALUDE_max_soap_boxes_in_carton_l419_41900


namespace NUMINAMATH_CALUDE_pie_remainder_l419_41950

theorem pie_remainder (carlos_share maria_share remainder : ℝ) : 
  carlos_share = 65 ∧ 
  maria_share = (100 - carlos_share) / 2 ∧ 
  remainder = 100 - carlos_share - maria_share →
  remainder = 17.5 := by
  sorry

end NUMINAMATH_CALUDE_pie_remainder_l419_41950


namespace NUMINAMATH_CALUDE_meaningful_range_l419_41903

def is_meaningful (x : ℝ) : Prop :=
  1 - x ≥ 0 ∧ 2 + x ≠ 0

theorem meaningful_range :
  ∀ x : ℝ, is_meaningful x ↔ x ≤ 1 ∧ x ≠ -2 := by
sorry

end NUMINAMATH_CALUDE_meaningful_range_l419_41903


namespace NUMINAMATH_CALUDE_problem_1_l419_41992

theorem problem_1 : 96 * 15 / (45 * 16) = 2 := by sorry

end NUMINAMATH_CALUDE_problem_1_l419_41992


namespace NUMINAMATH_CALUDE_greatest_common_remainder_l419_41948

theorem greatest_common_remainder (a b c d : ℕ) (h1 : a % 2 = 0 ∧ a % 3 = 0 ∧ a % 5 = 0 ∧ a % 7 = 0 ∧ a % 11 = 0)
                                               (h2 : b % 2 = 0 ∧ b % 3 = 0 ∧ b % 5 = 0 ∧ b % 7 = 0 ∧ b % 11 = 0)
                                               (h3 : c % 2 = 0 ∧ c % 3 = 0 ∧ c % 5 = 0 ∧ c % 7 = 0 ∧ c % 11 = 0)
                                               (h4 : d % 2 = 0 ∧ d % 3 = 0 ∧ d % 5 = 0 ∧ d % 7 = 0 ∧ d % 11 = 0)
                                               (ha : a = 1260) (hb : b = 2310) (hc : c = 30030) (hd : d = 72930) :
  ∃! k : ℕ, k > 0 ∧ k ≤ 30 ∧ 
  ∃ r : ℕ, a % k = r ∧ b % k = r ∧ c % k = r ∧ d % k = r ∧
  ∀ m : ℕ, m > k → ¬(∃ s : ℕ, a % m = s ∧ b % m = s ∧ c % m = s ∧ d % m = s) :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_remainder_l419_41948


namespace NUMINAMATH_CALUDE_arithmetic_geometric_progression_existence_l419_41975

theorem arithmetic_geometric_progression_existence :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (∃ (d : ℝ), b = a + d ∧ c = b + d) ∧
  (∃ (r : ℝ), (a = b ∧ b = c * r) ∨ (a = b * r ∧ b = c) ∨ (a = c ∧ c = b * r)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_progression_existence_l419_41975


namespace NUMINAMATH_CALUDE_no_modular_inverse_of_3_mod_33_l419_41938

theorem no_modular_inverse_of_3_mod_33 : ¬ ∃ x : ℕ, x ≤ 32 ∧ (3 * x) % 33 = 1 := by
  sorry

end NUMINAMATH_CALUDE_no_modular_inverse_of_3_mod_33_l419_41938


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l419_41935

/-- Given two quadratic equations with a specific relationship between their roots, 
    prove that the ratio of certain coefficients is 3. -/
theorem quadratic_root_relation (m n p : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hp : p ≠ 0) :
  (∃ (r₁ r₂ : ℝ), (r₁ + r₂ = -p ∧ r₁ * r₂ = m) ∧
                  (3 * r₁ + 3 * r₂ = -m ∧ 9 * r₁ * r₂ = n)) →
  n / p = 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l419_41935


namespace NUMINAMATH_CALUDE_simplify_expression_l419_41956

theorem simplify_expression (x : ℝ) (h : x^2 ≥ 49) :
  (7 - Real.sqrt (x^2 - 49))^2 = x^2 - 14 * Real.sqrt (x^2 - 49) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l419_41956


namespace NUMINAMATH_CALUDE_geometric_series_sum_l419_41983

theorem geometric_series_sum (a : ℝ) (h : |a| < 1) :
  (∑' n, a^n) = 1 / (1 - a) :=
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l419_41983


namespace NUMINAMATH_CALUDE_cos_minus_sin_value_l419_41913

theorem cos_minus_sin_value (α : Real) (h1 : π/4 < α) (h2 : α < π/2) (h3 : Real.sin (2*α) = 24/25) :
  Real.cos α - Real.sin α = -1/5 := by
sorry

end NUMINAMATH_CALUDE_cos_minus_sin_value_l419_41913


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l419_41947

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {2, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {2, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l419_41947


namespace NUMINAMATH_CALUDE_simple_interest_rate_calculation_l419_41990

theorem simple_interest_rate_calculation (P A T : ℝ) (h1 : P = 750) (h2 : A = 900) (h3 : T = 10) :
  let SI := A - P
  let R := (SI * 100) / (P * T)
  R = 2 := by sorry

end NUMINAMATH_CALUDE_simple_interest_rate_calculation_l419_41990


namespace NUMINAMATH_CALUDE_pencil_buyers_difference_l419_41981

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- The total amount spent by eighth graders in cents -/
def eighth_grade_total : ℕ := 162

/-- The total amount spent by fifth graders in cents -/
def fifth_grade_total : ℕ := 216

/-- The cost of each pencil in cents -/
def pencil_cost : ℕ := 18

theorem pencil_buyers_difference : 
  ∃ (eighth_buyers fifth_buyers : ℕ),
    eighth_grade_total = eighth_buyers * pencil_cost ∧
    fifth_grade_total = fifth_buyers * pencil_cost ∧
    fifth_buyers - eighth_buyers = 3 :=
by sorry

end NUMINAMATH_CALUDE_pencil_buyers_difference_l419_41981


namespace NUMINAMATH_CALUDE_champagne_bottle_volume_l419_41939

theorem champagne_bottle_volume
  (hot_tub_volume : ℚ)
  (quarts_per_gallon : ℚ)
  (bottle_cost : ℚ)
  (discount_rate : ℚ)
  (total_spent : ℚ)
  (h1 : hot_tub_volume = 40)
  (h2 : quarts_per_gallon = 4)
  (h3 : bottle_cost = 50)
  (h4 : discount_rate = 0.2)
  (h5 : total_spent = 6400) :
  (hot_tub_volume * quarts_per_gallon) / ((total_spent / (1 - discount_rate)) / bottle_cost) = 1 :=
by sorry

end NUMINAMATH_CALUDE_champagne_bottle_volume_l419_41939


namespace NUMINAMATH_CALUDE_bingo_prize_distribution_l419_41934

theorem bingo_prize_distribution (total_prize : ℚ) (first_winner_fraction : ℚ) 
  (num_subsequent_winners : ℕ) (each_subsequent_winner_prize : ℚ) :
  total_prize = 2400 →
  first_winner_fraction = 1/3 →
  num_subsequent_winners = 10 →
  each_subsequent_winner_prize = 160 →
  let remaining_prize := total_prize - first_winner_fraction * total_prize
  (each_subsequent_winner_prize / remaining_prize) = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_bingo_prize_distribution_l419_41934


namespace NUMINAMATH_CALUDE_percentage_increase_calculation_l419_41932

def original_earnings : ℝ := 60
def new_earnings : ℝ := 68

theorem percentage_increase_calculation :
  (new_earnings - original_earnings) / original_earnings * 100 = 13.33333333333333 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_calculation_l419_41932


namespace NUMINAMATH_CALUDE_initial_amount_at_racetrack_l419_41946

/-- Represents the sequence of bets and their outcomes at the racetrack --/
def racetrack_bets (initial_amount : ℝ) : ℝ :=
  let after_first := initial_amount * 2
  let after_second := after_first - 60
  let after_third := after_second * 2
  let after_fourth := after_third - 60
  let after_fifth := after_fourth * 2
  after_fifth - 60

/-- Theorem stating that the initial amount at the racetrack was 52.5 francs --/
theorem initial_amount_at_racetrack : 
  ∃ (x : ℝ), x > 0 ∧ racetrack_bets x = 0 ∧ x = 52.5 :=
sorry

end NUMINAMATH_CALUDE_initial_amount_at_racetrack_l419_41946


namespace NUMINAMATH_CALUDE_greene_nursery_roses_l419_41982

/-- The Greene Nursery flower counting problem -/
theorem greene_nursery_roses (total_flowers yellow_carnations white_roses : ℕ) 
  (h1 : total_flowers = 6284)
  (h2 : yellow_carnations = 3025)
  (h3 : white_roses = 1768) :
  total_flowers - yellow_carnations - white_roses = 1491 := by
  sorry

end NUMINAMATH_CALUDE_greene_nursery_roses_l419_41982


namespace NUMINAMATH_CALUDE_croissant_baking_time_l419_41980

/-- Calculates the baking time for croissants given the process parameters -/
theorem croissant_baking_time 
  (num_folds : ℕ) 
  (fold_time : ℕ) 
  (rest_time : ℕ) 
  (mixing_time : ℕ) 
  (total_time : ℕ) 
  (h1 : num_folds = 4)
  (h2 : fold_time = 5)
  (h3 : rest_time = 75)
  (h4 : mixing_time = 10)
  (h5 : total_time = 360) :
  total_time - (mixing_time + num_folds * fold_time + num_folds * rest_time) = 30 := by
  sorry

#check croissant_baking_time

end NUMINAMATH_CALUDE_croissant_baking_time_l419_41980


namespace NUMINAMATH_CALUDE_unique_solution_system_l419_41991

theorem unique_solution_system (x y : ℝ) :
  (x - 2*y = 1 ∧ 2*x - y = 11) ↔ (x = 7 ∧ y = 3) := by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l419_41991


namespace NUMINAMATH_CALUDE_bobs_favorite_number_l419_41930

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem bobs_favorite_number :
  ∃! n : ℕ,
    50 < n ∧ n < 100 ∧
    n % 11 = 0 ∧
    n % 2 ≠ 0 ∧
    sum_of_digits n % 3 = 0 ∧
    n = 99 := by
  sorry

end NUMINAMATH_CALUDE_bobs_favorite_number_l419_41930


namespace NUMINAMATH_CALUDE_productivity_wage_relation_l419_41904

/-- Represents the initial workday length in hours -/
def initial_workday : ℝ := 8

/-- Represents the reduced workday length in hours -/
def reduced_workday : ℝ := 7

/-- Represents the wage increase percentage -/
def wage_increase : ℝ := 5

/-- Represents the required productivity increase percentage -/
def productivity_increase : ℝ := 20

/-- Proves that a 20% productivity increase results in a 5% wage increase
    when the workday is reduced from 8 to 7 hours -/
theorem productivity_wage_relation :
  (reduced_workday / initial_workday) * (1 + productivity_increase / 100) = 1 + wage_increase / 100 :=
by sorry

end NUMINAMATH_CALUDE_productivity_wage_relation_l419_41904


namespace NUMINAMATH_CALUDE_rectangle_ratio_theorem_l419_41912

theorem rectangle_ratio_theorem (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (∃ (k l m n : ℕ), k * a + l * b = a * Real.sqrt 30 ∧
                     m * a + n * b = b * Real.sqrt 30 ∧
                     k * n = l * m ∧ l * m = 30) →
  (a / b = Real.sqrt 30 ∨
   a / b = Real.sqrt 30 / 2 ∨
   a / b = Real.sqrt 30 / 3 ∨
   a / b = Real.sqrt 30 / 5 ∨
   a / b = Real.sqrt 30 / 6 ∨
   a / b = Real.sqrt 30 / 10 ∨
   a / b = Real.sqrt 30 / 15 ∨
   a / b = Real.sqrt 30 / 30) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_ratio_theorem_l419_41912


namespace NUMINAMATH_CALUDE_total_shot_cost_l419_41959

/-- Represents the types of dogs Chuck breeds -/
inductive DogBreed
  | GoldenRetriever
  | GermanShepherd
  | Bulldog

/-- Represents the information for each dog breed -/
structure BreedInfo where
  pregnantDogs : Nat
  puppiesPerDog : Nat
  shotsPerPuppy : Nat
  costPerShot : Nat

/-- Calculates the total cost of shots for a specific breed -/
def breedShotCost (info : BreedInfo) : Nat :=
  info.pregnantDogs * info.puppiesPerDog * info.shotsPerPuppy * info.costPerShot

/-- Represents Chuck's dog breeding operation -/
def ChucksDogs : DogBreed → BreedInfo
  | DogBreed.GoldenRetriever => ⟨3, 4, 2, 5⟩
  | DogBreed.GermanShepherd => ⟨2, 5, 3, 8⟩
  | DogBreed.Bulldog => ⟨4, 3, 4, 10⟩

/-- Theorem stating the total cost of shots for all puppies -/
theorem total_shot_cost :
  (breedShotCost (ChucksDogs DogBreed.GoldenRetriever) +
   breedShotCost (ChucksDogs DogBreed.GermanShepherd) +
   breedShotCost (ChucksDogs DogBreed.Bulldog)) = 840 := by
  sorry

end NUMINAMATH_CALUDE_total_shot_cost_l419_41959


namespace NUMINAMATH_CALUDE_min_stamps_for_47_cents_l419_41997

def stamps (x y : ℕ) : ℕ := 5 * x + 7 * y

theorem min_stamps_for_47_cents :
  ∃ (x y : ℕ), stamps x y = 47 ∧
  (∀ (a b : ℕ), stamps a b = 47 → x + y ≤ a + b) ∧
  x + y = 7 := by
  sorry

end NUMINAMATH_CALUDE_min_stamps_for_47_cents_l419_41997


namespace NUMINAMATH_CALUDE_discriminant_irrational_l419_41941

/-- A quadratic polynomial without roots -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℚ
  c : ℝ
  no_roots : ∀ x : ℝ, a * x^2 + b * x + c ≠ 0

/-- The function f(x) for a QuadraticPolynomial -/
def f (p : QuadraticPolynomial) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- The discriminant of a QuadraticPolynomial -/
def discriminant (p : QuadraticPolynomial) : ℝ :=
  p.b^2 - 4 * p.a * p.c

/-- Exactly one of c or f(c) is irrational -/
axiom one_irrational (p : QuadraticPolynomial) :
  (¬ Irrational p.c ∧ Irrational (f p p.c)) ∨
  (Irrational p.c ∧ ¬ Irrational (f p p.c))

theorem discriminant_irrational (p : QuadraticPolynomial) :
  Irrational (discriminant p) :=
sorry

end NUMINAMATH_CALUDE_discriminant_irrational_l419_41941


namespace NUMINAMATH_CALUDE_solution_set_f_less_than_2_range_of_a_for_solution_exists_l419_41937

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 1|

-- Theorem for part I
theorem solution_set_f_less_than_2 :
  {x : ℝ | f x < 2} = Set.Ioo (-4 : ℝ) (2/3) :=
sorry

-- Theorem for part II
theorem range_of_a_for_solution_exists :
  {a : ℝ | ∃ x, f x ≤ a - a^2/2} = Set.Icc (-1 : ℝ) 3 :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_less_than_2_range_of_a_for_solution_exists_l419_41937


namespace NUMINAMATH_CALUDE_odd_integer_m_exists_l419_41977

def g (n : ℤ) : ℤ :=
  if n % 2 = 1 then n + 5 else n / 3

theorem odd_integer_m_exists (m : ℤ) (h_odd : m % 2 = 1) (h_g : g (g (g m)) = 14) : m = 121 := by
  sorry

end NUMINAMATH_CALUDE_odd_integer_m_exists_l419_41977


namespace NUMINAMATH_CALUDE_divisible_pair_count_l419_41970

/-- Given a set of 2117 cards numbered from 1 to 2117, this function calculates
    the number of ways to choose two cards such that their sum is divisible by 100. -/
def count_divisible_pairs : ℕ := 
  let card_count := 2117
  sorry

/-- Theorem stating that the number of ways to choose two cards with a sum
    divisible by 100 from a set of 2117 cards numbered 1 to 2117 is 23058. -/
theorem divisible_pair_count : count_divisible_pairs = 23058 := by
  sorry

end NUMINAMATH_CALUDE_divisible_pair_count_l419_41970


namespace NUMINAMATH_CALUDE_muffin_banana_price_ratio_l419_41917

/-- The price ratio of a muffin to a banana -/
def price_ratio (muffin_price banana_price : ℚ) : ℚ :=
  muffin_price / banana_price

/-- Susie's total cost for 4 muffins and 5 bananas -/
def susie_cost (muffin_price banana_price : ℚ) : ℚ :=
  4 * muffin_price + 5 * banana_price

/-- Calvin's total cost for 2 muffins and 12 bananas -/
def calvin_cost (muffin_price banana_price : ℚ) : ℚ :=
  2 * muffin_price + 12 * banana_price

theorem muffin_banana_price_ratio :
  ∀ (muffin_price banana_price : ℚ),
    muffin_price > 0 →
    banana_price > 0 →
    calvin_cost muffin_price banana_price = 3 * susie_cost muffin_price banana_price →
    price_ratio muffin_price banana_price = 3 / 10 := by
  sorry


end NUMINAMATH_CALUDE_muffin_banana_price_ratio_l419_41917


namespace NUMINAMATH_CALUDE_perfume_price_problem_l419_41968

/-- Proves that given the conditions of the perfume price changes, the original price must be $1200 -/
theorem perfume_price_problem (P : ℝ) : 
  (P * 1.10 * 0.85 = P - 78) → P = 1200 := by
  sorry

end NUMINAMATH_CALUDE_perfume_price_problem_l419_41968


namespace NUMINAMATH_CALUDE_youngest_brother_age_l419_41998

theorem youngest_brother_age (a b c : ℕ) : 
  b = a + 1 → c = b + 1 → a + b + c = 96 → a = 31 := by
sorry

end NUMINAMATH_CALUDE_youngest_brother_age_l419_41998


namespace NUMINAMATH_CALUDE_cody_dumplings_l419_41924

theorem cody_dumplings (A B : ℕ) (P1 Q1 Q2 P2 : ℚ) : 
  A = 14 → 
  B = 20 → 
  P1 = 1/2 → 
  Q1 = 1/4 → 
  Q2 = 2/5 → 
  P2 = 3/20 → 
  ∃ (remaining : ℕ), remaining = 16 ∧ 
    remaining = A - Int.floor (P1 * A) - Int.floor (Q1 * (A - Int.floor (P1 * A))) + 
                B - Int.floor (Q2 * B) - 
                Int.floor (P2 * (A - Int.floor (P1 * A) - Int.floor (Q1 * (A - Int.floor (P1 * A))) + 
                                 B - Int.floor (Q2 * B))) :=
by sorry

end NUMINAMATH_CALUDE_cody_dumplings_l419_41924


namespace NUMINAMATH_CALUDE_max_distinct_roots_special_polynomial_l419_41979

/-- A polynomial with the property that the product of any two distinct roots is also a root -/
def SpecialPolynomial (P : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≠ y → P x = 0 → P y = 0 → P (x * y) = 0

/-- The maximum number of distinct real roots for a special polynomial is 4 -/
theorem max_distinct_roots_special_polynomial :
  ∃ (P : ℝ → ℝ), SpecialPolynomial P ∧
    (∃ (roots : Finset ℝ), (∀ x ∈ roots, P x = 0) ∧ roots.card = 4) ∧
    (∀ (Q : ℝ → ℝ), SpecialPolynomial Q →
      ∀ (roots : Finset ℝ), (∀ x ∈ roots, Q x = 0) → roots.card ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_max_distinct_roots_special_polynomial_l419_41979


namespace NUMINAMATH_CALUDE_magic_deck_cost_l419_41919

/-- Calculates the cost per deck given the initial number of decks, remaining decks, and total earnings -/
def cost_per_deck (initial_decks : ℕ) (remaining_decks : ℕ) (total_earnings : ℕ) : ℚ :=
  total_earnings / (initial_decks - remaining_decks)

/-- Proves that the cost per deck is 7 dollars given the problem conditions -/
theorem magic_deck_cost :
  let initial_decks : ℕ := 16
  let remaining_decks : ℕ := 8
  let total_earnings : ℕ := 56
  cost_per_deck initial_decks remaining_decks total_earnings = 7 := by
  sorry


end NUMINAMATH_CALUDE_magic_deck_cost_l419_41919


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l419_41907

theorem quadratic_equation_roots (a : ℝ) : 
  (∃ x : ℝ, x^2 + (1/2)*x + a - 2 = 0 ∧ x = 1) → 
  (∃ y : ℝ, y^2 + (1/2)*y + a - 2 = 0 ∧ y = -3/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l419_41907


namespace NUMINAMATH_CALUDE_belinda_age_difference_l419_41915

/-- Given the ages of Tony and Belinda, prove that Belinda's age is 8 years more than twice Tony's age. -/
theorem belinda_age_difference (tony_age belinda_age : ℕ) : 
  tony_age = 16 →
  belinda_age = 40 →
  tony_age + belinda_age = 56 →
  belinda_age > 2 * tony_age →
  belinda_age - 2 * tony_age = 8 := by
sorry

end NUMINAMATH_CALUDE_belinda_age_difference_l419_41915


namespace NUMINAMATH_CALUDE_x_value_l419_41995

theorem x_value (x y : ℝ) (h1 : x - y = 6) (h2 : x + y = 12) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l419_41995


namespace NUMINAMATH_CALUDE_infinite_series_sum_l419_41911

/-- The sum of the infinite series ∑(k=1 to ∞) k³/3ᵏ is equal to 12 -/
theorem infinite_series_sum : ∑' k, (k : ℝ)^3 / 3^k = 12 := by sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l419_41911


namespace NUMINAMATH_CALUDE_fib_gcd_consecutive_fib_gcd_identity_l419_41989

def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

theorem fib_gcd_consecutive (n : ℕ) : Nat.gcd (fib n) (fib (n + 1)) = 1 := by
  sorry

theorem fib_gcd_identity (m n : ℕ) 
  (h : ∀ a b : ℕ, fib (a + b) = fib b * fib (a + 1) + fib (b - 1) * fib a) : 
  fib (Nat.gcd m n) = Nat.gcd (fib m) (fib n) := by
  sorry

end NUMINAMATH_CALUDE_fib_gcd_consecutive_fib_gcd_identity_l419_41989


namespace NUMINAMATH_CALUDE_granola_bar_distribution_l419_41996

theorem granola_bar_distribution (total : ℕ) (eaten_by_parents : ℕ) (num_children : ℕ) :
  total = 200 →
  eaten_by_parents = 80 →
  num_children = 6 →
  (total - eaten_by_parents) / num_children = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_granola_bar_distribution_l419_41996


namespace NUMINAMATH_CALUDE_embankment_project_additional_days_l419_41901

/-- Represents the embankment construction project -/
structure EmbankmentProject where
  initial_workers : ℕ
  initial_days : ℕ
  reassigned_workers : ℕ
  reassignment_day : ℕ
  productivity_factor : ℚ

/-- Calculates the additional days needed to complete the project -/
def additional_days_needed (project : EmbankmentProject) : ℚ :=
  let initial_rate : ℚ := 1 / (project.initial_workers * project.initial_days)
  let work_done_before_reassignment : ℚ := project.initial_workers * initial_rate * project.reassignment_day
  let remaining_work : ℚ := 1 - work_done_before_reassignment
  let remaining_workers : ℕ := project.initial_workers - project.reassigned_workers
  let new_rate : ℚ := initial_rate * project.productivity_factor
  let total_days : ℚ := project.reassignment_day + (remaining_work / (remaining_workers * new_rate))
  total_days - project.reassignment_day

/-- Theorem stating the additional days needed for the specific project -/
theorem embankment_project_additional_days :
  let project : EmbankmentProject := {
    initial_workers := 100,
    initial_days := 5,
    reassigned_workers := 40,
    reassignment_day := 2,
    productivity_factor := 3/4
  }
  additional_days_needed project = 53333 / 1000 := by sorry

end NUMINAMATH_CALUDE_embankment_project_additional_days_l419_41901


namespace NUMINAMATH_CALUDE_number_division_problem_l419_41921

theorem number_division_problem : ∃ x : ℝ, x / 5 = 80 + x / 6 ∧ x = 2400 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l419_41921


namespace NUMINAMATH_CALUDE_max_cube_sum_on_sphere_l419_41908

theorem max_cube_sum_on_sphere (x y z : ℝ) (h : x^2 + y^2 + z^2 = 9) :
  x^3 + y^3 + z^3 ≤ 27 ∧ ∃ x y z : ℝ, x^2 + y^2 + z^2 = 9 ∧ x^3 + y^3 + z^3 = 27 := by
  sorry

end NUMINAMATH_CALUDE_max_cube_sum_on_sphere_l419_41908


namespace NUMINAMATH_CALUDE_no_three_digit_divisible_by_15_ending_in_7_l419_41949

theorem no_three_digit_divisible_by_15_ending_in_7 : 
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 10 = 7 → ¬(n % 15 = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_three_digit_divisible_by_15_ending_in_7_l419_41949


namespace NUMINAMATH_CALUDE_geometric_sequence_equality_l419_41936

theorem geometric_sequence_equality (a b c d : ℝ) :
  (a / b = c / d) ↔ (a * d = b * c) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_equality_l419_41936


namespace NUMINAMATH_CALUDE_least_multiple_24_greater_500_l419_41960

theorem least_multiple_24_greater_500 : ∃ n : ℕ, 
  (24 * n > 500) ∧ (∀ m : ℕ, 24 * m > 500 → 24 * n ≤ 24 * m) ∧ (24 * n = 504) := by
  sorry

end NUMINAMATH_CALUDE_least_multiple_24_greater_500_l419_41960


namespace NUMINAMATH_CALUDE_factorization_equality_l419_41965

theorem factorization_equality (a b : ℝ) : 3 * a^2 + 6 * a * b + 3 * b^2 = 3 * (a + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l419_41965


namespace NUMINAMATH_CALUDE_probability_of_red_ball_in_bag_A_l419_41984

theorem probability_of_red_ball_in_bag_A 
  (m n : ℕ) 
  (h1 : m > 0) 
  (h2 : n > 0) :
  let total_A := m + n
  let total_B := 7
  let prob_red_A := m / total_A
  let prob_white_A := n / total_A
  let prob_red_B_after_red := (3 + 1) / (total_B + 1)
  let prob_red_B_after_white := 3 / (total_B + 1)
  let total_prob_red := prob_red_A * prob_red_B_after_red + prob_white_A * prob_red_B_after_white
  total_prob_red = 15/32 → prob_red_A = 3/4 := by
sorry

end NUMINAMATH_CALUDE_probability_of_red_ball_in_bag_A_l419_41984


namespace NUMINAMATH_CALUDE_finite_values_l419_41966

def recurrence (A B : ℕ) (x : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → x (n + 1) = A * Nat.gcd (x n) (x (n - 1)) + B

theorem finite_values (A B : ℕ) (x : ℕ → ℕ) (h : recurrence A B x) :
  ∃ (S : Finset ℕ), ∀ n : ℕ, x n ∈ S :=
sorry

end NUMINAMATH_CALUDE_finite_values_l419_41966


namespace NUMINAMATH_CALUDE_gregory_age_l419_41909

/-- Represents the ages of Dmitry and Gregory at different points in time -/
structure Ages where
  gregory_past : ℕ
  dmitry_past : ℕ
  gregory_current : ℕ
  dmitry_current : ℕ
  gregory_future : ℕ
  dmitry_future : ℕ

/-- The conditions of the problem -/
def age_conditions (a : Ages) : Prop :=
  a.dmitry_current = 3 * a.gregory_past ∧
  a.gregory_current = a.dmitry_past ∧
  a.gregory_future = a.dmitry_current ∧
  a.gregory_future + a.dmitry_future = 49 ∧
  a.dmitry_future - a.gregory_future = a.dmitry_current - a.gregory_current

theorem gregory_age (a : Ages) : age_conditions a → a.gregory_current = 14 := by
  sorry

#check gregory_age

end NUMINAMATH_CALUDE_gregory_age_l419_41909
