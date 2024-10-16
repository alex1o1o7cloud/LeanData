import Mathlib

namespace NUMINAMATH_CALUDE_ab_positive_sufficient_not_necessary_l4018_401865

theorem ab_positive_sufficient_not_necessary :
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a * b > 0) ∧
  (∃ a b : ℝ, a * b > 0 ∧ ¬(a > 0 ∧ b > 0)) :=
by sorry

end NUMINAMATH_CALUDE_ab_positive_sufficient_not_necessary_l4018_401865


namespace NUMINAMATH_CALUDE_max_ski_trips_l4018_401802

/-- Represents the time in minutes for a single trip up and down the mountain -/
def trip_time : ℕ := 15 + 5

/-- Represents the total available time in minutes -/
def total_time : ℕ := 2 * 60

/-- Theorem stating the maximum number of times a person can ski down the mountain in 2 hours -/
theorem max_ski_trips : (total_time / trip_time : ℕ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_max_ski_trips_l4018_401802


namespace NUMINAMATH_CALUDE_q_gt_one_neither_sufficient_nor_necessary_l4018_401875

/-- A geometric sequence with common ratio q -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

/-- An increasing sequence -/
def IncreasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

theorem q_gt_one_neither_sufficient_nor_necessary :
  ∃ (a₁ b₁ : ℕ → ℝ) (q₁ q₂ : ℝ),
    GeometricSequence a₁ q₁ ∧ q₁ > 1 ∧ ¬IncreasingSequence a₁ ∧
    GeometricSequence b₁ q₂ ∧ q₂ ≤ 1 ∧ IncreasingSequence b₁ :=
  sorry

end NUMINAMATH_CALUDE_q_gt_one_neither_sufficient_nor_necessary_l4018_401875


namespace NUMINAMATH_CALUDE_family_tickets_count_l4018_401813

theorem family_tickets_count :
  let adult_ticket_cost : ℕ := 19
  let child_ticket_cost : ℕ := 13
  let adult_count : ℕ := 2
  let child_count : ℕ := 3
  let total_cost : ℕ := 77
  adult_ticket_cost = child_ticket_cost + 6 ∧
  total_cost = adult_count * adult_ticket_cost + child_count * child_ticket_cost →
  adult_count + child_count = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_family_tickets_count_l4018_401813


namespace NUMINAMATH_CALUDE_f_properties_l4018_401855

noncomputable def f (x : ℝ) : ℝ := (1 / (2^x - 1) + 1/2) * x^3

theorem f_properties :
  (∀ x : ℝ, x ≠ 0 → f x ≠ 0) ∧
  (∀ x : ℝ, x ≠ 0 → f (-x) = f x) ∧
  (∀ x : ℝ, x ≠ 0 → f x > 0) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l4018_401855


namespace NUMINAMATH_CALUDE_x_squared_mod_26_l4018_401822

theorem x_squared_mod_26 (x : ℤ) (h1 : 6 * x ≡ 14 [ZMOD 26]) (h2 : 4 * x ≡ 20 [ZMOD 26]) :
  x^2 ≡ 12 [ZMOD 26] := by
  sorry

end NUMINAMATH_CALUDE_x_squared_mod_26_l4018_401822


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l4018_401846

/-- Properties of a rectangle and an ellipse -/
structure RectangleEllipseSystem where
  /-- Length of the rectangle -/
  x : ℝ
  /-- Width of the rectangle -/
  y : ℝ
  /-- Semi-major axis of the ellipse -/
  a : ℝ
  /-- Semi-minor axis of the ellipse -/
  b : ℝ
  /-- The area of the rectangle is 3260 -/
  area_rectangle : x * y = 3260
  /-- The area of the ellipse is 3260π -/
  area_ellipse : π * a * b = 3260 * π
  /-- The sum of length and width equals twice the semi-major axis -/
  major_axis : x + y = 2 * a
  /-- The rectangle diagonal equals twice the focal distance -/
  focal_distance : x^2 + y^2 = 4 * (a^2 - b^2)

/-- The perimeter of the rectangle is 8√1630 -/
theorem rectangle_perimeter (s : RectangleEllipseSystem) : 
  2 * (s.x + s.y) = 8 * Real.sqrt 1630 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l4018_401846


namespace NUMINAMATH_CALUDE_inverse_proportion_ratio_l4018_401899

theorem inverse_proportion_ratio (x₁ x₂ y₁ y₂ : ℝ) (hx₁ : x₁ ≠ 0) (hx₂ : x₂ ≠ 0) 
  (hy₁ : y₁ ≠ 0) (hy₂ : y₂ ≠ 0) (h_inv : ∃ k, k ≠ 0 ∧ ∀ x y, x * y = k) 
  (h_ratio : x₁ / x₂ = 3 / 5) : 
  y₁ / y₂ = 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_ratio_l4018_401899


namespace NUMINAMATH_CALUDE_tv_show_cost_l4018_401850

/-- Calculates the total cost of producing a TV show with the given parameters -/
def total_cost_tv_show (
  num_seasons : ℕ
  ) (first_season_cost_per_episode : ℕ
  ) (first_season_episodes : ℕ
  ) (last_season_episodes : ℕ
  ) : ℕ :=
  let other_season_cost_per_episode := 2 * first_season_cost_per_episode
  let first_season_cost := first_season_cost_per_episode * first_season_episodes
  let other_seasons_cost := 
    (other_season_cost_per_episode * first_season_episodes * 3 / 2) +
    (other_season_cost_per_episode * first_season_episodes * 9 / 4) +
    (other_season_cost_per_episode * first_season_episodes * 27 / 8) +
    (other_season_cost_per_episode * last_season_episodes)
  first_season_cost + other_seasons_cost

/-- The total cost of producing the TV show is $23,000,000 -/
theorem tv_show_cost :
  total_cost_tv_show 5 100000 12 24 = 23000000 := by
  sorry

end NUMINAMATH_CALUDE_tv_show_cost_l4018_401850


namespace NUMINAMATH_CALUDE_simplify_expression_l4018_401826

theorem simplify_expression (z : ℝ) : (5 - 4 * z^2) - (7 - 6 * z + 3 * z^2) = -2 - 7 * z^2 + 6 * z := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l4018_401826


namespace NUMINAMATH_CALUDE_distinct_cubes_modulo_prime_l4018_401890

theorem distinct_cubes_modulo_prime (a b c p : ℤ) : 
  Prime p → 
  p = a * b + b * c + a * c → 
  a ≠ b → b ≠ c → a ≠ c → 
  (a^3 % p ≠ b^3 % p) ∧ (b^3 % p ≠ c^3 % p) ∧ (a^3 % p ≠ c^3 % p) :=
by sorry

end NUMINAMATH_CALUDE_distinct_cubes_modulo_prime_l4018_401890


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l4018_401888

def polynomial (b₂ b₁ : ℤ) (x : ℤ) : ℤ := x^3 + b₂*x^2 + b₁*x + 18

theorem integer_roots_of_polynomial (b₂ b₁ : ℤ) :
  ∀ x : ℤ, polynomial b₂ b₁ x = 0 →
    x ∈ ({-18, -9, -6, -3, -2, -1, 1, 2, 3, 6, 9, 18} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l4018_401888


namespace NUMINAMATH_CALUDE_cubic_root_sum_squares_l4018_401809

theorem cubic_root_sum_squares (a b c : ℝ) : 
  (a^3 - 4*a^2 + 7*a - 2 = 0) → 
  (b^3 - 4*b^2 + 7*b - 2 = 0) → 
  (c^3 - 4*c^2 + 7*c - 2 = 0) → 
  a^2 + b^2 + c^2 = 2 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_squares_l4018_401809


namespace NUMINAMATH_CALUDE_soccer_teams_count_l4018_401814

theorem soccer_teams_count (n : ℕ) (k : ℕ) (h : n = 12 ∧ k = 6) :
  (Nat.choose n k : ℕ) = (Nat.choose n (k - 1) : ℕ) / k :=
by sorry

#check soccer_teams_count

end NUMINAMATH_CALUDE_soccer_teams_count_l4018_401814


namespace NUMINAMATH_CALUDE_age_ratio_problem_l4018_401819

/-- Proves that the ratio of b's age to c's age is 2:1 given the problem conditions -/
theorem age_ratio_problem (a b c : ℕ) : 
  a = b + 2 →  -- a is two years older than b
  a + b + c = 52 →  -- total of ages is 52
  b = 20 →  -- b is 20 years old
  b = 2 * c  -- ratio of b's age to c's age is 2:1
:= by sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l4018_401819


namespace NUMINAMATH_CALUDE_count_inequalities_l4018_401833

def is_inequality (e : String) : Bool :=
  match e with
  | "x - y" => false
  | "x ≤ y" => true
  | "x + y" => false
  | "x^2 - 3y" => false
  | "x ≥ 0" => true
  | "1/2x ≠ 3" => true
  | _ => false

def expressions : List String := [
  "x - y",
  "x ≤ y",
  "x + y",
  "x^2 - 3y",
  "x ≥ 0",
  "1/2x ≠ 3"
]

theorem count_inequalities :
  (expressions.filter is_inequality).length = 3 := by
  sorry

end NUMINAMATH_CALUDE_count_inequalities_l4018_401833


namespace NUMINAMATH_CALUDE_joans_clothing_expenditure_l4018_401838

/-- The total amount Joan spent on clothing --/
def total_spent (shorts jacket shirt shoes hat belt : ℝ)
  (jacket_discount shirt_discount : ℝ) (shoes_coupon : ℝ) : ℝ :=
  shorts + (jacket * (1 - jacket_discount)) + (shirt * shirt_discount) +
  (shoes - shoes_coupon) + hat + belt

/-- Theorem stating the total amount Joan spent on clothing --/
theorem joans_clothing_expenditure :
  let shorts : ℝ := 15
  let jacket : ℝ := 14.82
  let shirt : ℝ := 12.51
  let shoes : ℝ := 21.67
  let hat : ℝ := 8.75
  let belt : ℝ := 6.34
  let jacket_discount : ℝ := 0.1  -- 10% discount on jacket
  let shirt_discount : ℝ := 0.5   -- half price for shirt
  let shoes_coupon : ℝ := 3       -- $3 off coupon for shoes
  total_spent shorts jacket shirt shoes hat belt jacket_discount shirt_discount shoes_coupon = 68.353 := by
  sorry


end NUMINAMATH_CALUDE_joans_clothing_expenditure_l4018_401838


namespace NUMINAMATH_CALUDE_rotation_of_P_l4018_401897

/-- Rotate a point 180 degrees counterclockwise about the origin -/
def rotate180 (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

theorem rotation_of_P :
  let P : ℝ × ℝ := (-3, 2)
  rotate180 P = (3, -2) := by
sorry

end NUMINAMATH_CALUDE_rotation_of_P_l4018_401897


namespace NUMINAMATH_CALUDE_volunteer_distribution_l4018_401880

theorem volunteer_distribution (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 4) :
  (Nat.choose (n - 1) (k - 1) : ℕ) = 84 :=
by sorry

end NUMINAMATH_CALUDE_volunteer_distribution_l4018_401880


namespace NUMINAMATH_CALUDE_largest_common_term_l4018_401878

def isInFirstSequence (n : ℕ) : Prop := ∃ k : ℕ, n = 3 + 8 * k

def isInSecondSequence (n : ℕ) : Prop := ∃ m : ℕ, n = 5 + 9 * m

theorem largest_common_term : 
  (∀ n : ℕ, n > 59 ∧ n ≤ 90 → ¬(isInFirstSequence n ∧ isInSecondSequence n)) ∧ 
  isInFirstSequence 59 ∧ 
  isInSecondSequence 59 :=
sorry

end NUMINAMATH_CALUDE_largest_common_term_l4018_401878


namespace NUMINAMATH_CALUDE_factorization_equality_l4018_401885

theorem factorization_equality (x : ℝ) : -3*x^3 + 12*x^2 - 12*x = -3*x*(x-2)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l4018_401885


namespace NUMINAMATH_CALUDE_abc_sum_mod_five_l4018_401811

theorem abc_sum_mod_five (a b c : ℕ) : 
  a < 5 → b < 5 → c < 5 → a > 0 → b > 0 → c > 0 →
  (a * b * c) % 5 = 1 →
  (4 * c) % 5 = 3 →
  (3 * b) % 5 = (2 + b) % 5 →
  (a + b + c) % 5 = 1 := by
sorry

end NUMINAMATH_CALUDE_abc_sum_mod_five_l4018_401811


namespace NUMINAMATH_CALUDE_coefficient_x_cube_in_expansion_l4018_401812

theorem coefficient_x_cube_in_expansion : ∃ (c : ℤ), c = -10 ∧ 
  ∀ (x : ℝ), x * (x - 1)^5 = x^6 - 5*x^5 + 10*x^4 + c*x^3 + 5*x^2 - x := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_cube_in_expansion_l4018_401812


namespace NUMINAMATH_CALUDE_tablet_diagonal_comparison_l4018_401818

theorem tablet_diagonal_comparison (d : ℝ) : 
  d > 0 →  -- d is positive (diagonal length)
  (6 / Real.sqrt 2)^2 = (d / Real.sqrt 2)^2 + 5.5 →  -- area comparison
  d = 5 := by
sorry

end NUMINAMATH_CALUDE_tablet_diagonal_comparison_l4018_401818


namespace NUMINAMATH_CALUDE_distinct_pairs_solution_l4018_401821

theorem distinct_pairs_solution (x y : ℝ) : 
  x ≠ y ∧ 
  x^100 - y^100 = 2^99 * (x - y) ∧ 
  x^200 - y^200 = 2^199 * (x - y) → 
  (x = 2 ∧ y = 0) ∨ (x = 0 ∧ y = 2) := by
sorry

end NUMINAMATH_CALUDE_distinct_pairs_solution_l4018_401821


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l4018_401803

/-- Two 2D vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (-3, 2)
  let b : ℝ × ℝ := (x, 4)
  are_parallel a b → x = -6 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l4018_401803


namespace NUMINAMATH_CALUDE_minimum_transport_cost_l4018_401894

theorem minimum_transport_cost
  (total_trees : ℕ)
  (chinese_scholar_trees : ℕ)
  (white_pines : ℕ)
  (type_a_capacity_chinese : ℕ)
  (type_a_capacity_pine : ℕ)
  (type_b_capacity : ℕ)
  (type_a_cost : ℕ)
  (type_b_cost : ℕ)
  (total_trucks : ℕ)
  (h1 : total_trees = 320)
  (h2 : chinese_scholar_trees = white_pines + 80)
  (h3 : chinese_scholar_trees + white_pines = total_trees)
  (h4 : type_a_capacity_chinese = 40)
  (h5 : type_a_capacity_pine = 10)
  (h6 : type_b_capacity = 20)
  (h7 : type_a_cost = 400)
  (h8 : type_b_cost = 360)
  (h9 : total_trucks = 8) :
  ∃ (type_a_trucks : ℕ) (type_b_trucks : ℕ),
    type_a_trucks + type_b_trucks = total_trucks ∧
    type_a_trucks * type_a_capacity_chinese + type_b_trucks * type_b_capacity ≥ chinese_scholar_trees ∧
    type_a_trucks * type_a_capacity_pine + type_b_trucks * type_b_capacity ≥ white_pines ∧
    type_a_trucks * type_a_cost + type_b_trucks * type_b_cost = 2960 ∧
    ∀ (other_a : ℕ) (other_b : ℕ),
      other_a + other_b = total_trucks →
      other_a * type_a_capacity_chinese + other_b * type_b_capacity ≥ chinese_scholar_trees →
      other_a * type_a_capacity_pine + other_b * type_b_capacity ≥ white_pines →
      other_a * type_a_cost + other_b * type_b_cost ≥ 2960 :=
by sorry

end NUMINAMATH_CALUDE_minimum_transport_cost_l4018_401894


namespace NUMINAMATH_CALUDE_tax_free_items_cost_l4018_401856

def total_cost : ℝ := 120

def first_bracket_percentage : ℝ := 0.4
def second_bracket_percentage : ℝ := 0.3
def tax_free_percentage : ℝ := 1 - first_bracket_percentage - second_bracket_percentage

def first_bracket_tax_rate : ℝ := 0.06
def second_bracket_tax_rate : ℝ := 0.08
def second_bracket_discount : ℝ := 0.05

def first_bracket_cost : ℝ := total_cost * first_bracket_percentage
def second_bracket_cost : ℝ := total_cost * second_bracket_percentage
def tax_free_cost : ℝ := total_cost * tax_free_percentage

theorem tax_free_items_cost :
  tax_free_cost = 36 := by sorry

end NUMINAMATH_CALUDE_tax_free_items_cost_l4018_401856


namespace NUMINAMATH_CALUDE_y_value_when_x_is_zero_l4018_401893

theorem y_value_when_x_is_zero (t : ℚ) (x y : ℚ) 
  (h1 : x = 3 - 2*t) 
  (h2 : y = 3*t + 6) 
  (h3 : x = 0) : 
  y = 21/2 := by
  sorry

end NUMINAMATH_CALUDE_y_value_when_x_is_zero_l4018_401893


namespace NUMINAMATH_CALUDE_exist_numbers_not_triangle_l4018_401844

/-- Theorem: There exist natural numbers a and b, both greater than 1000,
    such that for any perfect square c, the triple (a, b, c) does not
    satisfy the triangle inequality. -/
theorem exist_numbers_not_triangle : ∃ a b : ℕ,
  a > 1000 ∧ b > 1000 ∧
  ∀ c : ℕ, (∃ d : ℕ, c = d * d) →
    ¬(a + b > c ∧ b + c > a ∧ a + c > b) := by
  sorry

end NUMINAMATH_CALUDE_exist_numbers_not_triangle_l4018_401844


namespace NUMINAMATH_CALUDE_sum_of_roots_l4018_401882

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x + 14

-- State the theorem
theorem sum_of_roots (a b : ℝ) (ha : f a = 1) (hb : f b = 19) : a + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l4018_401882


namespace NUMINAMATH_CALUDE_student_hall_ratio_l4018_401891

theorem student_hall_ratio : 
  let general_hall : ℕ := 30
  let total_students : ℕ := 144
  let math_hall (biology_hall : ℕ) : ℚ := (3/5) * (general_hall + biology_hall)
  ∃ biology_hall : ℕ, 
    (general_hall : ℚ) + biology_hall + math_hall biology_hall = total_students ∧
    biology_hall / general_hall = 2 := by
  sorry

end NUMINAMATH_CALUDE_student_hall_ratio_l4018_401891


namespace NUMINAMATH_CALUDE_divisibility_of_sum_of_cubes_l4018_401852

theorem divisibility_of_sum_of_cubes (n m : ℕ+) 
  (h : n^3 + (n+1)^3 + (n+2)^3 = m^3) : 
  4 ∣ (n+1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_sum_of_cubes_l4018_401852


namespace NUMINAMATH_CALUDE_tan_theta_value_l4018_401867

theorem tan_theta_value (θ : Real) 
  (h1 : 0 < θ) (h2 : θ < π/2)
  (h3 : (Real.sin θ + Real.cos θ)^2 + Real.sqrt 3 * Real.cos (2*θ) = 3) :
  Real.tan θ = 2 - Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_tan_theta_value_l4018_401867


namespace NUMINAMATH_CALUDE_intersection_point_coords_l4018_401896

/-- A line in a 2D plane represented by its slope and a point it passes through. -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The y-axis, represented as a vertical line passing through (0, 0). -/
def yAxis : Line := { slope := 0, point := (0, 0) }

/-- Two lines are parallel if they have the same slope. -/
def parallel (l1 l2 : Line) : Prop := l1.slope = l2.slope

/-- The point where a line intersects the y-axis. -/
def yAxisIntersection (l : Line) : ℝ × ℝ :=
  (0, l.point.2 + l.slope * (0 - l.point.1))

theorem intersection_point_coords (l1 l2 : Line) (P : ℝ × ℝ) :
  l1.slope = 2 →
  parallel l1 l2 →
  l2.point = (-1, 1) →
  P = yAxisIntersection l2 →
  P = (0, 3) := by sorry

end NUMINAMATH_CALUDE_intersection_point_coords_l4018_401896


namespace NUMINAMATH_CALUDE_yellow_curlers_count_l4018_401853

/-- Given the total number of curlers and the proportions of different types,
    prove that the number of extra-large yellow curlers is 18. -/
theorem yellow_curlers_count (total : ℕ) (pink : ℕ) (blue : ℕ) (green : ℕ) (yellow : ℕ) : 
  total = 120 →
  pink = total / 5 →
  blue = 2 * pink →
  green = total / 4 →
  yellow = total - pink - blue - green →
  yellow = 18 := by
sorry

end NUMINAMATH_CALUDE_yellow_curlers_count_l4018_401853


namespace NUMINAMATH_CALUDE_same_solution_implies_a_value_l4018_401877

theorem same_solution_implies_a_value : ∀ a : ℝ,
  (∀ x : ℝ, (2*x - a)/3 - (x - a)/2 = x - 1 ↔ 3*(x - 2) - 4*(x - 5/4) = 0) →
  a = -11 := by sorry

end NUMINAMATH_CALUDE_same_solution_implies_a_value_l4018_401877


namespace NUMINAMATH_CALUDE_inverse_f_243_l4018_401820

def f (x : ℝ) : ℝ := sorry

theorem inverse_f_243 (h1 : f 5 = 3) (h2 : ∀ x, f (3 * x) = 3 * f x) : 
  f 405 = 243 := by sorry

end NUMINAMATH_CALUDE_inverse_f_243_l4018_401820


namespace NUMINAMATH_CALUDE_population_growth_l4018_401884

theorem population_growth (initial_population : ℝ) (final_population : ℝ) (second_year_increase : ℝ) :
  initial_population = 1000 →
  final_population = 1320 →
  second_year_increase = 0.20 →
  ∃ first_year_increase : ℝ,
    first_year_increase = 0.10 ∧
    final_population = initial_population * (1 + first_year_increase) * (1 + second_year_increase) :=
by sorry

end NUMINAMATH_CALUDE_population_growth_l4018_401884


namespace NUMINAMATH_CALUDE_gcd_282_470_l4018_401859

theorem gcd_282_470 : Nat.gcd 282 470 = 94 := by
  sorry

end NUMINAMATH_CALUDE_gcd_282_470_l4018_401859


namespace NUMINAMATH_CALUDE_a_must_be_negative_l4018_401840

theorem a_must_be_negative (a b c d e : ℝ) 
  (h1 : a / b < -(c / d))
  (h2 : b > 0)
  (h3 : d > 0)
  (h4 : e > 0)
  (h5 : a + e > 0) :
  a < 0 := by
  sorry

end NUMINAMATH_CALUDE_a_must_be_negative_l4018_401840


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l4018_401831

def f (x : ℤ) : ℤ := x^3 - 4*x^2 - 7*x + 10

theorem integer_roots_of_polynomial :
  {x : ℤ | f x = 0} = {1, -2, 5} := by sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l4018_401831


namespace NUMINAMATH_CALUDE_probability_of_event_A_is_half_events_A_and_C_mutually_exclusive_l4018_401837

/-- Represents the labels on the balls -/
inductive Label : Type
  | one : Label
  | two : Label
  | three : Label

/-- Represents a pair of drawn balls -/
structure DrawnBalls :=
  (fromA : Label)
  (fromB : Label)

/-- The sample space of all possible outcomes -/
def sampleSpace : List DrawnBalls := sorry

/-- Event A: sum of labels < 4 -/
def eventA (db : DrawnBalls) : Prop := sorry

/-- Event C: product of labels > 3 -/
def eventC (db : DrawnBalls) : Prop := sorry

/-- The probability of an event -/
def probability (event : DrawnBalls → Prop) : ℚ := sorry

theorem probability_of_event_A_is_half :
  probability eventA = 1 / 2 := sorry

theorem events_A_and_C_mutually_exclusive :
  ∀ db : DrawnBalls, ¬(eventA db ∧ eventC db) := sorry

end NUMINAMATH_CALUDE_probability_of_event_A_is_half_events_A_and_C_mutually_exclusive_l4018_401837


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l4018_401870

theorem sum_of_squares_of_roots (a b : ℝ) : 
  (a^2 - 8*a + 8 = 0) → (b^2 - 8*b + 8 = 0) → a^2 + b^2 = 48 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l4018_401870


namespace NUMINAMATH_CALUDE_books_in_box_l4018_401836

theorem books_in_box (total : ℕ) (difference : ℕ) (books_a : ℕ) (books_b : ℕ) : 
  total = 20 → 
  difference = 4 → 
  books_a + books_b = total → 
  books_a = books_b + difference → 
  books_a = 12 := by
sorry

end NUMINAMATH_CALUDE_books_in_box_l4018_401836


namespace NUMINAMATH_CALUDE_smallest_3digit_prime_factor_of_binom_300_150_l4018_401857

theorem smallest_3digit_prime_factor_of_binom_300_150 :
  let n := Nat.choose 300 150
  ∃ (p : Nat), Prime p ∧ 100 ≤ p ∧ p < 1000 ∧ p ∣ n ∧
    ∀ (q : Nat), Prime q → 100 ≤ q → q < p → ¬(q ∣ n) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_3digit_prime_factor_of_binom_300_150_l4018_401857


namespace NUMINAMATH_CALUDE_calculate_expression_l4018_401869

theorem calculate_expression : 18 * 35 + 45 * 18 - 18 * 10 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l4018_401869


namespace NUMINAMATH_CALUDE_cheapest_solution_for_1096_days_l4018_401808

/-- Represents the cost and coverage of a ticket type -/
structure Ticket where
  days : ℕ
  cost : ℚ

/-- Finds the minimum cost to cover at least a given number of days using two types of tickets -/
def minCost (ticket1 ticket2 : Ticket) (totalDays : ℕ) : ℚ :=
  sorry

theorem cheapest_solution_for_1096_days :
  let sevenDayTicket : Ticket := ⟨7, 703/100⟩
  let thirtyDayTicket : Ticket := ⟨30, 30⟩
  minCost sevenDayTicket thirtyDayTicket 1096 = 140134/100 := by sorry

end NUMINAMATH_CALUDE_cheapest_solution_for_1096_days_l4018_401808


namespace NUMINAMATH_CALUDE_student_difference_l4018_401868

theorem student_difference (lower_grades : ℕ) (middle_upper_grades : ℕ) : 
  lower_grades = 325 →
  middle_upper_grades = 4 * lower_grades →
  middle_upper_grades - lower_grades = 975 := by
  sorry

end NUMINAMATH_CALUDE_student_difference_l4018_401868


namespace NUMINAMATH_CALUDE_surface_area_increase_percentage_l4018_401866

/-- The percentage increase in surface area when placing a hemispherical cap on a sphere -/
theorem surface_area_increase_percentage (R : ℝ) (R_pos : R > 0) : 
  let sphere_area := 4 * Real.pi * R^2
  let cap_radius := R * Real.sqrt 3 / 2
  let cap_area := 2 * Real.pi * cap_radius^2
  let covered_cap_height := R / 2
  let covered_cap_area := 2 * Real.pi * R * covered_cap_height
  let area_increase := cap_area - covered_cap_area
  area_increase / sphere_area * 100 = 12.5 :=
sorry

end NUMINAMATH_CALUDE_surface_area_increase_percentage_l4018_401866


namespace NUMINAMATH_CALUDE_salt_water_ratio_l4018_401889

theorem salt_water_ratio (salt : ℕ) (water : ℕ) :
  salt = 1 ∧ water = 10 →
  (salt : ℚ) / (salt + water : ℚ) = 1 / 11 :=
by sorry

end NUMINAMATH_CALUDE_salt_water_ratio_l4018_401889


namespace NUMINAMATH_CALUDE_percent_equivalence_l4018_401892

theorem percent_equivalence : ∃ x : ℚ, (60 / 100 * 500 : ℚ) = x / 100 * 600 ∧ x = 50 := by
  sorry

end NUMINAMATH_CALUDE_percent_equivalence_l4018_401892


namespace NUMINAMATH_CALUDE_solve_linear_equation_l4018_401839

theorem solve_linear_equation :
  ∃! x : ℚ, 3 * x - 5 = 8 ∧ x = 13 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l4018_401839


namespace NUMINAMATH_CALUDE_remainder_98765432_mod_25_l4018_401806

theorem remainder_98765432_mod_25 : 98765432 % 25 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_98765432_mod_25_l4018_401806


namespace NUMINAMATH_CALUDE_base_seven_to_ten_63524_l4018_401898

/-- Converts a digit in base 7 to its value in base 10 -/
def baseSevenDigitToBaseTen (d : Nat) : Nat :=
  if d < 7 then d else 0

/-- Converts a list of digits in base 7 to its value in base 10 -/
def baseSevenToBaseTen (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => acc * 7 + baseSevenDigitToBaseTen d) 0

/-- The base 7 number 63524 converted to base 10 equals 15698 -/
theorem base_seven_to_ten_63524 :
  baseSevenToBaseTen [6, 3, 5, 2, 4] = 15698 := by
  sorry

end NUMINAMATH_CALUDE_base_seven_to_ten_63524_l4018_401898


namespace NUMINAMATH_CALUDE_prime_triple_divisibility_l4018_401834

theorem prime_triple_divisibility (p q r : ℕ) : 
  Prime p ∧ Prime q ∧ Prime r ∧
  (p ∣ 1 + q^r) ∧ (q ∣ 1 + r^p) ∧ (r ∣ 1 + p^q) →
  ((p = 2 ∧ q = 5 ∧ r = 3) ∨ 
   (p = 5 ∧ q = 3 ∧ r = 2) ∨ 
   (p = 3 ∧ q = 2 ∧ r = 5)) :=
by sorry

end NUMINAMATH_CALUDE_prime_triple_divisibility_l4018_401834


namespace NUMINAMATH_CALUDE_average_problem_l4018_401815

theorem average_problem (y : ℝ) : (15 + 30 + 45 + y) / 4 = 35 → y = 50 := by
  sorry

end NUMINAMATH_CALUDE_average_problem_l4018_401815


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l4018_401886

theorem algebraic_expression_value (x y : ℝ) 
  (sum_eq : x + y = 2) 
  (diff_eq : x - y = 4) : 
  1 + x^2 - y^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l4018_401886


namespace NUMINAMATH_CALUDE_probability_even_or_greater_than_4_l4018_401879

/-- A fair six-sided die. -/
structure Die :=
  (faces : Finset ℕ)
  (fair : faces.card = 6)
  (labeled : faces = {1, 2, 3, 4, 5, 6})

/-- The event "the number facing up is even or greater than 4". -/
def EventEvenOrGreaterThan4 (d : Die) : Finset ℕ :=
  d.faces.filter (λ x => x % 2 = 0 ∨ x > 4)

/-- The probability of an event for a fair die. -/
def Probability (d : Die) (event : Finset ℕ) : ℚ :=
  event.card / d.faces.card

theorem probability_even_or_greater_than_4 (d : Die) :
  Probability d (EventEvenOrGreaterThan4 d) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_probability_even_or_greater_than_4_l4018_401879


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l4018_401871

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b, a < b ∧ b < 0 → 1/a > 1/b) ∧
  (∃ a b, 1/a > 1/b ∧ ¬(a < b ∧ b < 0)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l4018_401871


namespace NUMINAMATH_CALUDE_living_room_size_l4018_401849

/-- Given an apartment with the following properties:
  * Total area is 160 square feet
  * There are 6 rooms in total
  * The living room is as big as 3 other rooms
  * All rooms except the living room are the same size
Prove that the living room's area is 96 square feet. -/
theorem living_room_size (total_area : ℝ) (num_rooms : ℕ) (living_room_ratio : ℕ) :
  total_area = 160 →
  num_rooms = 6 →
  living_room_ratio = 3 →
  ∃ (room_unit : ℝ),
    room_unit * (num_rooms - 1 + living_room_ratio) = total_area ∧
    living_room_ratio * room_unit = 96 := by
  sorry

end NUMINAMATH_CALUDE_living_room_size_l4018_401849


namespace NUMINAMATH_CALUDE_sum_of_solutions_l4018_401845

theorem sum_of_solutions (x₁ x₂ : ℝ) : 
  (2 * x₁^2 - 7 * x₁ - 9 = 0) → 
  (2 * x₂^2 - 7 * x₂ - 9 = 0) → 
  (x₁ ≠ x₂) →
  (x₁ + x₂ = 7/2) := by
sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l4018_401845


namespace NUMINAMATH_CALUDE_basketball_games_lost_l4018_401874

theorem basketball_games_lost (total_games : ℕ) (games_won : ℕ) (win_difference : ℕ) 
  (h1 : total_games = 62)
  (h2 : games_won = 45)
  (h3 : games_won = win_difference + (total_games - games_won)) :
  total_games - games_won = 17 := by
  sorry

end NUMINAMATH_CALUDE_basketball_games_lost_l4018_401874


namespace NUMINAMATH_CALUDE_sum_2011_is_29_l4018_401854

/-- Given a sequence of 2011 consecutive five-digit numbers, this function
    returns the sum of digits for the nth number in the sequence. -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- The sequence starts with a five-digit number. -/
axiom start_five_digit : ∃ k : ℕ, 10000 ≤ k ∧ k < 100000 ∧ ∀ i, 1 ≤ i ∧ i ≤ 2011 → k + i - 1 < 100000

/-- The sum of digits of the 21st number is 37. -/
axiom sum_21 : sumOfDigits 21 = 37

/-- The sum of digits of the 54th number is 7. -/
axiom sum_54 : sumOfDigits 54 = 7

/-- The main theorem: the sum of digits of the 2011th number is 29. -/
theorem sum_2011_is_29 : sumOfDigits 2011 = 29 := sorry

end NUMINAMATH_CALUDE_sum_2011_is_29_l4018_401854


namespace NUMINAMATH_CALUDE_even_power_plus_one_all_digits_equal_l4018_401858

def is_all_digits_equal (n : ℕ) : Prop :=
  ∃ d : ℕ, ∀ k : ℕ, k < (Nat.log 10 n + 1) → (n / 10^k) % 10 = d

def solution_set : Set (ℕ × ℕ) :=
  {(2, 2), (2, 3), (2, 5), (6, 5)}

theorem even_power_plus_one_all_digits_equal :
  ∀ a b : ℕ,
    a ≥ 2 →
    b ≥ 2 →
    Even a →
    is_all_digits_equal (a^b + 1) →
    (a, b) ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_even_power_plus_one_all_digits_equal_l4018_401858


namespace NUMINAMATH_CALUDE_point_on_circle_l4018_401873

/-- 
Given a line ax + by - 1 = 0 that is tangent to the circle x² + y² = 1,
prove that the point P(a, b) lies on the circle.
-/
theorem point_on_circle (a b : ℝ) 
  (h_tangent : ∃ (x y : ℝ), x^2 + y^2 = 1 ∧ a*x + b*y = 1) : 
  a^2 + b^2 = 1 := by
  sorry


end NUMINAMATH_CALUDE_point_on_circle_l4018_401873


namespace NUMINAMATH_CALUDE_diagonals_150_sided_polygon_l4018_401817

/-- Number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The number of diagonals in a 150-sided polygon is 11025 -/
theorem diagonals_150_sided_polygon :
  num_diagonals 150 = 11025 := by
  sorry

end NUMINAMATH_CALUDE_diagonals_150_sided_polygon_l4018_401817


namespace NUMINAMATH_CALUDE_cos_2alpha_minus_pi_3_l4018_401823

theorem cos_2alpha_minus_pi_3 (α : ℝ) 
  (h : Real.sin (α + π/6) - Real.cos α = 1/3) : 
  Real.cos (2*α - π/3) = 7/9 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_minus_pi_3_l4018_401823


namespace NUMINAMATH_CALUDE_min_ones_in_sum_l4018_401810

/-- Count the number of '1's in the binary representation of an integer -/
def countOnes (n : ℕ) : ℕ := sorry

/-- The theorem statement -/
theorem min_ones_in_sum (a b : ℕ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (ca : countOnes a = 20041) 
  (cb : countOnes b = 20051) : 
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ countOnes x = 20041 ∧ countOnes y = 20051 ∧ countOnes (x + y) = 1 := by
  sorry

end NUMINAMATH_CALUDE_min_ones_in_sum_l4018_401810


namespace NUMINAMATH_CALUDE_complex_equation_solution_l4018_401862

def i : ℂ := Complex.I

theorem complex_equation_solution :
  ∃ z : ℂ, z * (1 - i) = 2 * i ∧ z = -1 + i := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l4018_401862


namespace NUMINAMATH_CALUDE_total_chairs_moved_l4018_401860

/-- The total number of chairs agreed to be moved is equal to the sum of
    chairs moved by Carey, chairs moved by Pat, and chairs left to move. -/
theorem total_chairs_moved (carey_chairs pat_chairs left_chairs : ℕ)
  (h1 : carey_chairs = 28)
  (h2 : pat_chairs = 29)
  (h3 : left_chairs = 17) :
  carey_chairs + pat_chairs + left_chairs = 74 := by
  sorry

end NUMINAMATH_CALUDE_total_chairs_moved_l4018_401860


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l4018_401827

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  a 1 = 1 →  -- given condition
  4 * a 2 - 2 * a 3 = 2 * a 3 - a 4 →  -- arithmetic sequence condition
  a 2 + a 3 + a 4 = 14 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l4018_401827


namespace NUMINAMATH_CALUDE_factorial_division_l4018_401824

theorem factorial_division : 
  (10 : ℕ).factorial / (4 : ℕ).factorial = 151200 :=
by
  have h1 : (10 : ℕ).factorial = 3628800 := by sorry
  sorry

end NUMINAMATH_CALUDE_factorial_division_l4018_401824


namespace NUMINAMATH_CALUDE_rainfall_difference_l4018_401832

/-- Rainfall data for Tropical Storm Sally -/
structure RainfallData where
  day1 : ℕ
  day2 : ℕ
  day3 : ℕ

/-- Conditions for Tropical Storm Sally's rainfall -/
def sallysRainfall : RainfallData where
  day1 := 4
  day2 := 5 * 4
  day3 := 18

/-- Theorem: The difference between the sum of the first two days' rainfall and the third day's rainfall is 6 inches -/
theorem rainfall_difference (data : RainfallData := sallysRainfall) :
  (data.day1 + data.day2) - data.day3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_rainfall_difference_l4018_401832


namespace NUMINAMATH_CALUDE_stream_speed_l4018_401842

/-- Given a boat traveling downstream, prove the speed of the stream. -/
theorem stream_speed (boat_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) :
  boat_speed = 30 →
  downstream_distance = 70 →
  downstream_time = 2 →
  ∃ stream_speed : ℝ,
    stream_speed = 5 ∧
    downstream_distance = (boat_speed + stream_speed) * downstream_time :=
by sorry

end NUMINAMATH_CALUDE_stream_speed_l4018_401842


namespace NUMINAMATH_CALUDE_square_units_tens_digits_l4018_401851

theorem square_units_tens_digits (x : ℤ) (h : x^2 % 100 = 9) : 
  x^2 % 200 = 0 ∨ x^2 % 200 = 100 := by
  sorry

end NUMINAMATH_CALUDE_square_units_tens_digits_l4018_401851


namespace NUMINAMATH_CALUDE_no_solutions_exist_l4018_401847

theorem no_solutions_exist : ¬∃ (x y : ℕ), 
  x^4 * y^4 - 14 * x^2 * y^2 + 49 = 0 ∧ x + y = 10 := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_exist_l4018_401847


namespace NUMINAMATH_CALUDE_triangle_properties_l4018_401801

theorem triangle_properties (a b c : ℝ) (A : ℝ) (area : ℝ) :
  (a + b + c) * (b + c - a) = 3 * b * c →
  a = 2 →
  area = Real.sqrt 3 →
  A = π / 3 ∧ b = 2 ∧ c = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l4018_401801


namespace NUMINAMATH_CALUDE_brad_age_is_13_l4018_401876

def shara_age : ℕ := 10

def jaymee_age : ℕ := 2 * shara_age + 2

def average_age : ℕ := (shara_age + jaymee_age) / 2

def brad_age : ℕ := average_age - 3

theorem brad_age_is_13 : brad_age = 13 := by
  sorry

end NUMINAMATH_CALUDE_brad_age_is_13_l4018_401876


namespace NUMINAMATH_CALUDE_min_value_of_function_min_value_achieved_l4018_401816

theorem min_value_of_function (x : ℝ) (h : x > 1) :
  (x^2 + x + 1) / (x - 1) ≥ 3 + 2 * Real.sqrt 3 :=
sorry

theorem min_value_achieved (x : ℝ) (h : x > 1) :
  ∃ x₀ > 1, (x₀^2 + x₀ + 1) / (x₀ - 1) = 3 + 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_function_min_value_achieved_l4018_401816


namespace NUMINAMATH_CALUDE_two_digit_subtraction_l4018_401895

/-- Given two different natural numbers A and B that satisfy the two-digit subtraction equation 6A - B2 = 36, prove that A - B = 5 -/
theorem two_digit_subtraction (A B : ℕ) (h1 : A ≠ B) (h2 : 10 ≤ A) (h3 : A < 100) (h4 : 10 ≤ B) (h5 : B < 100) (h6 : 60 + A - (10 * B + 2) = 36) : A - B = 5 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_subtraction_l4018_401895


namespace NUMINAMATH_CALUDE_binomial_sum_first_six_l4018_401883

theorem binomial_sum_first_six (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ) :
  (∀ x : ℝ, (x + 1)^11 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + 
    a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10 + a₁₁*x^11) →
  a + a₁ + a₂ + a₃ + a₄ + a₅ = 1024 := by
sorry

end NUMINAMATH_CALUDE_binomial_sum_first_six_l4018_401883


namespace NUMINAMATH_CALUDE_combined_blanket_thickness_l4018_401841

/-- The combined thickness of 5 blankets, each with an initial thickness of 3 inches
    and folded according to their color code (1 to 5), is equal to 186 inches. -/
theorem combined_blanket_thickness :
  let initial_thickness : ℝ := 3
  let color_codes : List ℕ := [1, 2, 3, 4, 5]
  let folded_thickness (c : ℕ) : ℝ := initial_thickness * (2 ^ c)
  List.sum (List.map folded_thickness color_codes) = 186 := by
  sorry


end NUMINAMATH_CALUDE_combined_blanket_thickness_l4018_401841


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l4018_401830

theorem polynomial_divisibility (F : ℤ → ℤ) 
  (h1 : ∀ x y : ℤ, F (x + y) - F x - F y = (x * y) * (F 1 - 1))
  (h2 : (F 2) % 5 = 0)
  (h3 : (F 5) % 2 = 0) :
  (F 7) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l4018_401830


namespace NUMINAMATH_CALUDE_max_band_members_l4018_401872

/-- Represents a band formation --/
structure BandFormation where
  rows : ℕ
  membersPerRow : ℕ
  totalMembers : ℕ

/-- Checks if a band formation is valid according to the problem conditions --/
def isValidFormation (bf : BandFormation) : Prop :=
  bf.totalMembers < 120 ∧
  bf.totalMembers = bf.rows * bf.membersPerRow + 3 ∧
  bf.totalMembers = (bf.rows + 1) * (bf.membersPerRow + 2)

/-- Theorem stating the maximum number of band members --/
theorem max_band_members :
  ∀ bf : BandFormation, isValidFormation bf → bf.totalMembers ≤ 119 :=
by sorry

end NUMINAMATH_CALUDE_max_band_members_l4018_401872


namespace NUMINAMATH_CALUDE_second_investment_value_l4018_401861

theorem second_investment_value (x : ℝ) : 
  (0.07 * 500 + 0.09 * x = 0.085 * (500 + x)) → x = 1500 := by
  sorry

end NUMINAMATH_CALUDE_second_investment_value_l4018_401861


namespace NUMINAMATH_CALUDE_rotated_rectangle_area_fraction_l4018_401805

/-- Represents a point on a 2D grid -/
structure Point where
  x : ℤ
  y : ℤ

/-- Represents a rectangle on a 2D grid -/
structure Rectangle where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- Calculates the area of a rectangle given its vertices -/
def rectangleArea (r : Rectangle) : ℝ :=
  sorry

/-- Calculates the area of a square grid -/
def gridArea (size : ℤ) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem rotated_rectangle_area_fraction :
  let grid_size : ℤ := 6
  let r : Rectangle := {
    v1 := { x := 2, y := 2 },
    v2 := { x := 4, y := 4 },
    v3 := { x := 2, y := 4 },
    v4 := { x := 4, y := 6 }
  }
  rectangleArea r / gridArea grid_size = Real.sqrt 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_rotated_rectangle_area_fraction_l4018_401805


namespace NUMINAMATH_CALUDE_parabola_directrix_l4018_401887

/-- The parabola equation -/
def parabola_eq (x y : ℝ) : Prop := x^2 = 4*y

/-- The directrix equation -/
def directrix_eq (y : ℝ) : Prop := y = -1

/-- Theorem: The directrix of the parabola x^2 = 4y is y = -1 -/
theorem parabola_directrix : 
  ∀ x y : ℝ, parabola_eq x y → ∃ d : ℝ, directrix_eq d ∧ 
  (∀ p : ℝ × ℝ, p.1^2 = 4*p.2 → (p.1^2 + (p.2 - d)^2) = (p.2 - d)^2) :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l4018_401887


namespace NUMINAMATH_CALUDE_tangent_line_at_point_one_four_l4018_401843

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 + x + 2

/-- The derivative of the parabola function -/
def f' (x : ℝ) : ℝ := 2*x + 1

theorem tangent_line_at_point_one_four :
  let x₀ : ℝ := 1
  let y₀ : ℝ := 4
  -- The point (1,4) lies on the parabola
  (f x₀ = y₀) →
  -- The slope of the tangent line at (1,4) is 3
  (f' x₀ = 3) ∧
  -- The equation of the tangent line is 3x - y + 1 = 0
  (∀ x y, y - y₀ = f' x₀ * (x - x₀) ↔ 3*x - y + 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_point_one_four_l4018_401843


namespace NUMINAMATH_CALUDE_unique_divisible_digit_l4018_401863

def is_single_digit (n : ℕ) : Prop := n < 10

def number_with_A (A : ℕ) : ℕ := 26372 * 100 + A * 10 + 21

theorem unique_divisible_digit :
  ∃! A : ℕ, is_single_digit A ∧ 
    (∃ k₁ k₂ k₃ : ℕ, 
      number_with_A A = 2 * k₁ ∧
      number_with_A A = 3 * k₂ ∧
      number_with_A A = 4 * k₃) :=
sorry

end NUMINAMATH_CALUDE_unique_divisible_digit_l4018_401863


namespace NUMINAMATH_CALUDE_spoons_multiple_of_groups_l4018_401829

/-- Represents the number of commemorative plates Daniel has -/
def num_plates : ℕ := 44

/-- Represents the number of groups Daniel can form -/
def num_groups : ℕ := 11

/-- Represents the number of commemorative spoons Daniel has -/
def num_spoons : ℕ := sorry

/-- Theorem stating that the number of spoons is a multiple of the number of groups -/
theorem spoons_multiple_of_groups :
  ∃ k : ℕ, num_spoons = k * num_groups :=
sorry

end NUMINAMATH_CALUDE_spoons_multiple_of_groups_l4018_401829


namespace NUMINAMATH_CALUDE_laura_debt_l4018_401807

/-- Calculates the total amount owed after applying simple interest -/
def total_amount_owed (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal + principal * rate * time

/-- Proves that Laura owes $36.40 after one year -/
theorem laura_debt : 
  let principal : ℝ := 35
  let rate : ℝ := 0.04
  let time : ℝ := 1
  total_amount_owed principal rate time = 36.40 := by
  sorry

end NUMINAMATH_CALUDE_laura_debt_l4018_401807


namespace NUMINAMATH_CALUDE_second_class_revenue_l4018_401825

/-- The amount collected from II class passengers given the passenger and fare ratios --/
theorem second_class_revenue (total_revenue : ℚ) 
  (h1 : total_revenue = 1325)
  (h2 : ∃ (x y : ℚ), x * y * 53 = total_revenue ∧ x > 0 ∧ y > 0) :
  ∃ (x y : ℚ), 50 * x * y = 1250 :=
sorry

end NUMINAMATH_CALUDE_second_class_revenue_l4018_401825


namespace NUMINAMATH_CALUDE_perfect_square_k_l4018_401835

theorem perfect_square_k (K : ℕ) (h1 : K > 1) (h2 : 1000 < K^4) (h3 : K^4 < 5000) :
  ∃ (n : ℕ), K^4 = n^2 ↔ K = 6 ∨ K = 7 ∨ K = 8 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_k_l4018_401835


namespace NUMINAMATH_CALUDE_unique_divisor_property_l4018_401800

theorem unique_divisor_property (p : ℕ) (h_prime : Nat.Prime p) (h_p_gt_5 : p > 5) :
  ∃! x : ℕ, x ≠ 0 ∧ ∀ n : ℕ, n > 0 → (5 * p + x) ∣ (5 * p^n + x^n) ∧ x = p := by
  sorry

end NUMINAMATH_CALUDE_unique_divisor_property_l4018_401800


namespace NUMINAMATH_CALUDE_two_digit_number_50th_power_l4018_401828

theorem two_digit_number_50th_power (log2 log3 log11 : ℝ) 
  (h_log2 : log2 = 0.3010)
  (h_log3 : log3 = 0.4771)
  (h_log11 : log11 = 1.0414) :
  ∃! P : ℕ, 
    10 ≤ P ∧ P < 100 ∧ 
    (10^68 : ℝ) ≤ (P^50 : ℝ) ∧ (P^50 : ℝ) < (10^69 : ℝ) ∧
    P = 23 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_50th_power_l4018_401828


namespace NUMINAMATH_CALUDE_quadratic_coefficient_bounds_l4018_401804

theorem quadratic_coefficient_bounds (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hroots : b^2 - 4*a*c ≥ 0) : 
  (max a (max b c) ≥ 4/9 * (a + b + c)) ∧ 
  (min a (min b c) ≤ 1/4 * (a + b + c)) := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_bounds_l4018_401804


namespace NUMINAMATH_CALUDE_ln_plus_const_increasing_l4018_401864

theorem ln_plus_const_increasing (x : ℝ) (h : x > 0) :
  Monotone (fun x => Real.log x + 2) :=
sorry

end NUMINAMATH_CALUDE_ln_plus_const_increasing_l4018_401864


namespace NUMINAMATH_CALUDE_binary_exponentiation_not_always_optimal_l4018_401848

/-- The minimum number of multiplications needed to compute x^n -/
noncomputable def l (n : ℕ) : ℕ := sorry

/-- The number of multiplications needed to compute x^n using the binary exponentiation method -/
def b (n : ℕ) : ℕ := sorry

/-- Theorem stating that the binary exponentiation method is not always optimal -/
theorem binary_exponentiation_not_always_optimal :
  ∃ n : ℕ, l n < b n := by sorry

end NUMINAMATH_CALUDE_binary_exponentiation_not_always_optimal_l4018_401848


namespace NUMINAMATH_CALUDE_consecutive_sum_inequality_l4018_401881

theorem consecutive_sum_inequality (nums : Fin 100 → ℝ) 
  (h_distinct : ∀ i j : Fin 100, i ≠ j → nums i ≠ nums j) :
  ∃ i : Fin 100, nums i + nums ((i + 3) % 100) > nums ((i + 1) % 100) + nums ((i + 2) % 100) :=
sorry

end NUMINAMATH_CALUDE_consecutive_sum_inequality_l4018_401881
