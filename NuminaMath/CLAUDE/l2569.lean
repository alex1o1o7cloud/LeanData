import Mathlib

namespace NUMINAMATH_CALUDE_baker_cakes_problem_l2569_256906

theorem baker_cakes_problem (initial_cakes : ℕ) (sold_cakes : ℕ) (extra_sold : ℕ) :
  initial_cakes = 8 →
  sold_cakes = 145 →
  extra_sold = 6 →
  ∃ (new_cakes : ℕ), 
    new_cakes + initial_cakes = sold_cakes + extra_sold ∧
    new_cakes = 131 :=
by sorry

end NUMINAMATH_CALUDE_baker_cakes_problem_l2569_256906


namespace NUMINAMATH_CALUDE_complex_power_four_l2569_256983

theorem complex_power_four (i : ℂ) (h : i^2 = -1) : (2 + i)^4 = -7 + 24*i := by
  sorry

end NUMINAMATH_CALUDE_complex_power_four_l2569_256983


namespace NUMINAMATH_CALUDE_number_of_workers_l2569_256970

theorem number_of_workers (
  avg_salary_with_first_supervisor : ℝ)
  (first_supervisor_salary : ℝ)
  (avg_salary_with_new_supervisor : ℝ)
  (new_supervisor_salary : ℝ)
  (h1 : avg_salary_with_first_supervisor = 430)
  (h2 : first_supervisor_salary = 870)
  (h3 : avg_salary_with_new_supervisor = 440)
  (h4 : new_supervisor_salary = 960) :
  ∃ (w : ℕ), w = 8 ∧
  (w + 1) * avg_salary_with_first_supervisor - first_supervisor_salary =
  9 * avg_salary_with_new_supervisor - new_supervisor_salary :=
by sorry

end NUMINAMATH_CALUDE_number_of_workers_l2569_256970


namespace NUMINAMATH_CALUDE_complex_purely_imaginary_l2569_256934

theorem complex_purely_imaginary (a : ℝ) : 
  (Complex.I : ℂ).re = 0 ∧ (Complex.I : ℂ).im = 1 →
  ((a : ℂ) + Complex.I) / ((1 : ℂ) + 2 * Complex.I) = Complex.I * ((1 - 2 * a : ℝ) / 5) →
  a = -2 := by
sorry

end NUMINAMATH_CALUDE_complex_purely_imaginary_l2569_256934


namespace NUMINAMATH_CALUDE_min_xyz_value_l2569_256974

theorem min_xyz_value (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x * y + z = (x + z) * (y + z)) :
  x * y * z ≥ (1 : ℝ) / 27 := by sorry

end NUMINAMATH_CALUDE_min_xyz_value_l2569_256974


namespace NUMINAMATH_CALUDE_international_call_rate_l2569_256994

/-- Represents the cost and duration of phone calls -/
structure PhoneCall where
  localRate : ℚ
  localDuration : ℚ
  internationalDuration : ℚ
  totalCost : ℚ

/-- Calculates the cost per minute of an international call -/
def internationalRate (call : PhoneCall) : ℚ :=
  (call.totalCost - call.localRate * call.localDuration) / call.internationalDuration

/-- Theorem: Given the specified conditions, the international call rate is 25 cents per minute -/
theorem international_call_rate (call : PhoneCall) 
  (h1 : call.localRate = 5/100)
  (h2 : call.localDuration = 45)
  (h3 : call.internationalDuration = 31)
  (h4 : call.totalCost = 10) :
  internationalRate call = 25/100 := by
  sorry


end NUMINAMATH_CALUDE_international_call_rate_l2569_256994


namespace NUMINAMATH_CALUDE_tshirt_cost_l2569_256982

theorem tshirt_cost (initial_amount : ℝ) (sweater_cost : ℝ) (shoes_cost : ℝ) 
                    (refund_percentage : ℝ) (final_amount : ℝ) :
  initial_amount = 74 →
  sweater_cost = 9 →
  shoes_cost = 30 →
  refund_percentage = 0.9 →
  final_amount = 51 →
  ∃ (tshirt_cost : ℝ),
    tshirt_cost = 14 ∧
    final_amount = initial_amount - sweater_cost - tshirt_cost - shoes_cost + refund_percentage * shoes_cost :=
by sorry

end NUMINAMATH_CALUDE_tshirt_cost_l2569_256982


namespace NUMINAMATH_CALUDE_yarn_length_multiple_l2569_256973

theorem yarn_length_multiple (green_length red_length total_length x : ℝ) : 
  green_length = 156 →
  red_length = green_length * x + 8 →
  total_length = green_length + red_length →
  total_length = 632 →
  x = 3 := by
sorry

end NUMINAMATH_CALUDE_yarn_length_multiple_l2569_256973


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l2569_256971

-- Define the quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 - 2*x - 3 = 0

-- Theorem statement
theorem point_in_second_quadrant (m n : ℝ) 
  (hm : quadratic_eq m) (hn : quadratic_eq n) (hlt : m < n) : 
  m < 0 ∧ n > 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l2569_256971


namespace NUMINAMATH_CALUDE_championship_outcomes_8_3_l2569_256951

/-- The number of possible outcomes for championships -/
def championship_outcomes (num_students : ℕ) (num_championships : ℕ) : ℕ :=
  num_students ^ num_championships

/-- Theorem: The number of possible outcomes for 3 championships among 8 students is 512 -/
theorem championship_outcomes_8_3 :
  championship_outcomes 8 3 = 512 := by
  sorry

end NUMINAMATH_CALUDE_championship_outcomes_8_3_l2569_256951


namespace NUMINAMATH_CALUDE_function_value_at_specific_point_l2569_256913

/-- Given a function f(x) = ax^3 + b*sin(x) + 4 where a and b are real numbers,
    and f(lg(log_2(10))) = 5, prove that f(lg(lg(2))) = 3 -/
theorem function_value_at_specific_point
  (a b : ℝ)
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = a * x^3 + b * Real.sin x + 4)
  (h2 : f (Real.log 10 / Real.log 2) = 5) :
  f (Real.log (Real.log 2) / Real.log 10) = 3 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_specific_point_l2569_256913


namespace NUMINAMATH_CALUDE_triangle_sine_sum_inequality_l2569_256963

theorem triangle_sine_sum_inequality (A B C : Real) 
  (h : A + B + C = Real.pi) : 
  Real.sin A + Real.sin B + Real.sin C ≤ (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_sum_inequality_l2569_256963


namespace NUMINAMATH_CALUDE_max_surface_area_inscribed_cylinder_l2569_256966

/-- Given a cone with height h and base radius r, where h > 2r, 
    the maximum total surface area of an inscribed cylinder is πh²r / (2(h - r)). -/
theorem max_surface_area_inscribed_cylinder (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) (h_gt_2r : h > 2 * r) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧
    x ≤ r ∧ y ≤ h ∧
    (∀ (x' y' : ℝ), x' > 0 → y' > 0 → x' ≤ r → y' ≤ h →
      2 * π * x' * (x' + y') ≤ 2 * π * x * (x + y)) ∧
    2 * π * x * (x + y) = π * h^2 * r / (2 * (h - r)) :=
sorry

end NUMINAMATH_CALUDE_max_surface_area_inscribed_cylinder_l2569_256966


namespace NUMINAMATH_CALUDE_gcd_a4_3a2_1_a3_2a_eq_one_l2569_256926

theorem gcd_a4_3a2_1_a3_2a_eq_one (a : ℕ) : 
  Nat.gcd (a^4 + 3*a^2 + 1) (a^3 + 2*a) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_a4_3a2_1_a3_2a_eq_one_l2569_256926


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_l2569_256962

/-- A circle with center (1, 0) and radius √m is tangent to the line x + y = 1 if and only if m = 1/2 -/
theorem circle_tangent_to_line (m : ℝ) : 
  (∃ (x y : ℝ), (x - 1)^2 + y^2 = m ∧ x + y = 1 ∧ 
    ∀ (x' y' : ℝ), (x' - 1)^2 + y'^2 = m → x' + y' ≥ 1) ↔ 
  m = 1/2 :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_line_l2569_256962


namespace NUMINAMATH_CALUDE_floor_cube_difference_l2569_256997

theorem floor_cube_difference : 
  ⌊(2007^3 : ℝ) / (2005 * 2006) - (2008^3 : ℝ) / (2006 * 2007)⌋ = -4 := by
  sorry

end NUMINAMATH_CALUDE_floor_cube_difference_l2569_256997


namespace NUMINAMATH_CALUDE_dormitory_expenditure_l2569_256957

theorem dormitory_expenditure 
  (initial_students : ℕ) 
  (new_students : ℕ) 
  (cost_decrease : ℕ) 
  (expenditure_increase : ℕ) 
  (h1 : initial_students = 250)
  (h2 : new_students = 75)
  (h3 : cost_decrease = 20)
  (h4 : expenditure_increase = 10000) :
  (initial_students + new_students) * 
  ((initial_students + new_students) * expenditure_increase / initial_students - cost_decrease) = 65000 := by
  sorry

end NUMINAMATH_CALUDE_dormitory_expenditure_l2569_256957


namespace NUMINAMATH_CALUDE_max_term_binomial_expansion_max_term_value_max_term_specific_case_l2569_256914

theorem max_term_binomial_expansion (n : ℕ) (x : ℝ) (h : x > 0) :
  ∃ k : ℕ, k ≤ n ∧
  ∀ j : ℕ, j ≤ n →
    (n.choose k : ℝ) * x^k ≥ (n.choose j : ℝ) * x^j :=
by sorry

theorem max_term_value (n : ℕ) (x : ℝ) (h : x > 0) :
  let k := ⌊n * x / (1 + x)⌋ + 1
  ∃ m : ℕ, m ≤ n ∧
    ∀ j : ℕ, j ≤ n →
      (n.choose m : ℝ) * x^m ≥ (n.choose j : ℝ) * x^j ∧
    (m = k ∨ m = k - 1) :=
by sorry

theorem max_term_specific_case :
  let n : ℕ := 210
  let x : ℝ := Real.sqrt 13
  let k : ℕ := 165
  ∃ m : ℕ, m ≤ n ∧
    ∀ j : ℕ, j ≤ n →
      (n.choose m : ℝ) * x^m ≥ (n.choose j : ℝ) * x^j ∧
    m = k :=
by sorry

end NUMINAMATH_CALUDE_max_term_binomial_expansion_max_term_value_max_term_specific_case_l2569_256914


namespace NUMINAMATH_CALUDE_least_common_multiple_first_ten_l2569_256930

theorem least_common_multiple_first_ten : ∃ n : ℕ, 
  (∀ k : ℕ, k ≤ 10 → k > 0 → n % k = 0) ∧ 
  (∀ m : ℕ, m < n → ∃ k : ℕ, k ≤ 10 ∧ k > 0 ∧ m % k ≠ 0) :=
by
  use 2520
  sorry

end NUMINAMATH_CALUDE_least_common_multiple_first_ten_l2569_256930


namespace NUMINAMATH_CALUDE_xy_product_cardinality_l2569_256928

def X : Finset ℕ := {1, 2, 3, 4}
def Y : Finset ℕ := {5, 6, 7, 8}

theorem xy_product_cardinality :
  Finset.card ((X.product Y).image (λ (p : ℕ × ℕ) => p.1 * p.2)) = 15 := by
  sorry

end NUMINAMATH_CALUDE_xy_product_cardinality_l2569_256928


namespace NUMINAMATH_CALUDE_max_unique_sums_l2569_256935

def coin_values : List ℕ := [1, 1, 1, 5, 10, 25]

def unique_sums (values : List ℕ) : Finset ℕ :=
  (values.map (λ x => values.map (λ y => x + y))).join.toFinset

theorem max_unique_sums :
  Finset.card (unique_sums coin_values) = 7 := by sorry

end NUMINAMATH_CALUDE_max_unique_sums_l2569_256935


namespace NUMINAMATH_CALUDE_profit_maximization_l2569_256996

/-- Represents the daily sales volume as a function of the selling price. -/
def sales_volume (x : ℝ) : ℝ := -20 * x + 1600

/-- Represents the daily profit as a function of the selling price. -/
def profit (x : ℝ) : ℝ := (x - 40) * (sales_volume x)

theorem profit_maximization (x : ℝ) (h1 : x ≥ 45) (h2 : x < 80) :
  profit x ≤ 8000 ∧ profit 60 = 8000 :=
by sorry

#check profit_maximization

end NUMINAMATH_CALUDE_profit_maximization_l2569_256996


namespace NUMINAMATH_CALUDE_prob_green_is_one_eighth_l2569_256958

-- Define the number of cubes for each color
def pink_cubes : ℕ := 36
def blue_cubes : ℕ := 18
def green_cubes : ℕ := 9
def red_cubes : ℕ := 6
def purple_cubes : ℕ := 3

-- Define the total number of cubes
def total_cubes : ℕ := pink_cubes + blue_cubes + green_cubes + red_cubes + purple_cubes

-- Define the probability of selecting a green cube
def prob_green : ℚ := green_cubes / total_cubes

-- Theorem statement
theorem prob_green_is_one_eighth : prob_green = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_prob_green_is_one_eighth_l2569_256958


namespace NUMINAMATH_CALUDE_obtuse_triangle_condition_l2569_256902

-- Define a triangle structure
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  -- Add triangle inequality constraints
  hpos_a : 0 < a
  hpos_b : 0 < b
  hpos_c : 0 < c
  hab : a + b > c
  hbc : b + c > a
  hca : c + a > b

-- Define what it means for a triangle to be obtuse
def is_obtuse (t : Triangle) : Prop :=
  t.a^2 + t.b^2 < t.c^2 ∨ t.b^2 + t.c^2 < t.a^2 ∨ t.c^2 + t.a^2 < t.b^2

-- State the theorem
theorem obtuse_triangle_condition (t : Triangle) :
  (t.a^2 + t.b^2 < t.c^2 → is_obtuse t) ∧
  ∃ (t' : Triangle), is_obtuse t' ∧ t'.a^2 + t'.b^2 ≥ t'.c^2 :=
sorry

end NUMINAMATH_CALUDE_obtuse_triangle_condition_l2569_256902


namespace NUMINAMATH_CALUDE_smallest_number_with_ten_even_five_or_seven_l2569_256981

def containsDigit (n : ℕ) (d : ℕ) : Prop :=
  ∃ k m : ℕ, n = 10 * k + d + 10 * m

def isEvenWithFiveOrSeven (n : ℕ) : Prop :=
  n % 2 = 0 ∧ (containsDigit n 5 ∨ containsDigit n 7)

theorem smallest_number_with_ten_even_five_or_seven : 
  (∃! m : ℕ, m > 0 ∧ (∃ S : Finset ℕ, Finset.card S = 10 ∧ 
    (∀ n ∈ S, n < m ∧ isEvenWithFiveOrSeven n) ∧
    (∀ n : ℕ, n < m → isEvenWithFiveOrSeven n → n ∈ S))) ∧
  (∀ m : ℕ, m > 0 → (∃ S : Finset ℕ, Finset.card S = 10 ∧ 
    (∀ n ∈ S, n < m ∧ isEvenWithFiveOrSeven n) ∧
    (∀ n : ℕ, n < m → isEvenWithFiveOrSeven n → n ∈ S)) → m ≥ 160) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_ten_even_five_or_seven_l2569_256981


namespace NUMINAMATH_CALUDE_apex_distance_theorem_l2569_256984

/-- Represents a right octagonal pyramid with two parallel cross sections -/
structure RightOctagonalPyramid where
  small_area : ℝ
  large_area : ℝ
  plane_distance : ℝ

/-- The distance from the apex to the plane of the larger cross section -/
def apex_to_large_section (p : RightOctagonalPyramid) : ℝ :=
  36 -- We define this as 36 based on the problem statement

/-- Theorem stating the relationship between the pyramid's properties and the apex distance -/
theorem apex_distance_theorem (p : RightOctagonalPyramid) 
  (h1 : p.small_area = 256 * Real.sqrt 2)
  (h2 : p.large_area = 576 * Real.sqrt 2)
  (h3 : p.plane_distance = 12) :
  apex_to_large_section p = 36 := by
  sorry

#check apex_distance_theorem

end NUMINAMATH_CALUDE_apex_distance_theorem_l2569_256984


namespace NUMINAMATH_CALUDE_total_weekly_time_l2569_256905

def parking_time : ℕ := 5
def walking_time : ℕ := 3
def long_wait_days : ℕ := 2
def short_wait_days : ℕ := 3
def long_wait_time : ℕ := 30
def short_wait_time : ℕ := 10
def work_days : ℕ := 5

theorem total_weekly_time :
  (parking_time + walking_time) * work_days +
  long_wait_days * long_wait_time +
  short_wait_days * short_wait_time = 130 := by
sorry

end NUMINAMATH_CALUDE_total_weekly_time_l2569_256905


namespace NUMINAMATH_CALUDE_max_gcd_consecutive_b_terms_l2569_256989

def b (n : ℕ) : ℕ := n.factorial + 2^n + n

theorem max_gcd_consecutive_b_terms :
  ∃ (k : ℕ), ∀ (n : ℕ), Nat.gcd (b n) (b (n + 1)) ≤ k ∧
  ∃ (m : ℕ), Nat.gcd (b m) (b (m + 1)) = k ∧
  k = 2 :=
sorry

end NUMINAMATH_CALUDE_max_gcd_consecutive_b_terms_l2569_256989


namespace NUMINAMATH_CALUDE_real_sum_greater_than_two_l2569_256901

theorem real_sum_greater_than_two (x y : ℝ) : x + y > 2 → x > 1 ∨ y > 1 := by
  sorry

end NUMINAMATH_CALUDE_real_sum_greater_than_two_l2569_256901


namespace NUMINAMATH_CALUDE_train_distance_difference_l2569_256959

/-- Represents the distance traveled by a train given its speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Represents the total distance between two points -/
def total_distance : ℝ := 900

/-- Represents the speed of the first train -/
def speed1 : ℝ := 50

/-- Represents the speed of the second train -/
def speed2 : ℝ := 40

/-- Theorem stating the difference in distance traveled by two trains -/
theorem train_distance_difference :
  ∃ (time : ℝ), 
    time > 0 ∧
    distance speed1 time + distance speed2 time = total_distance ∧
    distance speed1 time - distance speed2 time = 100 :=
sorry

end NUMINAMATH_CALUDE_train_distance_difference_l2569_256959


namespace NUMINAMATH_CALUDE_vector_collinearity_l2569_256949

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 * w.2 = k * v.2 * w.1

theorem vector_collinearity (x : ℝ) : 
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (x, 1)
  collinear a (a.1 - b.1, a.2 - b.2) → x = 1/2 := by
sorry

end NUMINAMATH_CALUDE_vector_collinearity_l2569_256949


namespace NUMINAMATH_CALUDE_comic_book_stacking_theorem_l2569_256918

def num_spiderman : ℕ := 7
def num_archie : ℕ := 6
def num_garfield : ℕ := 4

def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

def permutations_within_groups : ℕ := 
  factorial num_spiderman * factorial num_archie * factorial num_garfield

def group_arrangements : ℕ := 2 * 2

theorem comic_book_stacking_theorem :
  permutations_within_groups * group_arrangements = 19353600 := by
  sorry

end NUMINAMATH_CALUDE_comic_book_stacking_theorem_l2569_256918


namespace NUMINAMATH_CALUDE_percentage_comparison_l2569_256936

theorem percentage_comparison (w x y z t : ℝ) 
  (hw : w = 0.6 * x)
  (hx : x = 0.6 * y)
  (hz : z = 0.54 * y)
  (ht : t = 0.48 * x) :
  (z - w) / w * 100 = 50 ∧ (w - t) / w * 100 = 20 :=
by sorry

end NUMINAMATH_CALUDE_percentage_comparison_l2569_256936


namespace NUMINAMATH_CALUDE_cylinder_unique_non_identical_views_l2569_256972

-- Define the types of solid objects
inductive SolidObject
  | Sphere
  | TriangularPyramid
  | Cube
  | Cylinder

-- Define a function that checks if all views are identical
def hasIdenticalViews (obj : SolidObject) : Prop :=
  match obj with
  | SolidObject.Sphere => True
  | SolidObject.TriangularPyramid => False
  | SolidObject.Cube => True
  | SolidObject.Cylinder => False

-- Theorem statement
theorem cylinder_unique_non_identical_views :
  ∀ (obj : SolidObject), ¬(hasIdenticalViews obj) ↔ obj = SolidObject.Cylinder :=
by sorry

end NUMINAMATH_CALUDE_cylinder_unique_non_identical_views_l2569_256972


namespace NUMINAMATH_CALUDE_cookout_2006_attendance_l2569_256929

def cookout_2004 : ℕ := 60

def cookout_2005 : ℕ := cookout_2004 / 2

def cookout_2006 : ℕ := (cookout_2005 * 2) / 3

theorem cookout_2006_attendance : cookout_2006 = 20 := by
  sorry

end NUMINAMATH_CALUDE_cookout_2006_attendance_l2569_256929


namespace NUMINAMATH_CALUDE_fermat_for_small_exponents_l2569_256991

theorem fermat_for_small_exponents (x y z n : ℕ) (h : n ≥ z) :
  x^n + y^n ≠ z^n := by
  sorry

end NUMINAMATH_CALUDE_fermat_for_small_exponents_l2569_256991


namespace NUMINAMATH_CALUDE_question_1_question_2_l2569_256955

-- Define propositions p and q
def p (x : ℝ) : Prop := (x + 1) * (2 - x) ≥ 0
def q (m : ℝ) (x : ℝ) : Prop := x^2 + 2*m*x - m + 6 > 0

-- Theorem for question 1
theorem question_1 (m : ℝ) : (∀ x, q m x) → m ∈ Set.Ioo (-3 : ℝ) 2 :=
sorry

-- Theorem for question 2
theorem question_2 (m : ℝ) : 
  ((∀ x, p x → q m x) ∧ (∃ x, q m x ∧ ¬p x)) → m ∈ Set.Ioc (-3 : ℝ) 2 :=
sorry

end NUMINAMATH_CALUDE_question_1_question_2_l2569_256955


namespace NUMINAMATH_CALUDE_like_terms_exponent_l2569_256956

theorem like_terms_exponent (a b : ℕ) : 
  (∀ x y : ℝ, ∃ k : ℝ, 2 * x^a * y^3 = k * (-x^2 * y^b)) → a^b = 8 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_exponent_l2569_256956


namespace NUMINAMATH_CALUDE_probability_ratio_l2569_256952

/-- The number of slips in the hat -/
def total_slips : ℕ := 50

/-- The number of distinct numbers on the slips -/
def distinct_numbers : ℕ := 10

/-- The number of slips drawn -/
def drawn_slips : ℕ := 5

/-- The number of slips for each number -/
def slips_per_number : ℕ := 5

/-- The probability of drawing all five slips with the same number -/
def p : ℚ := (distinct_numbers : ℚ) / (Nat.choose total_slips drawn_slips : ℚ)

/-- The probability of drawing three slips with one number and two slips with another number -/
def q : ℚ := (Nat.choose distinct_numbers 2 * Nat.choose slips_per_number 3 * Nat.choose slips_per_number 2 : ℚ) / (Nat.choose total_slips drawn_slips : ℚ)

theorem probability_ratio :
  q / p = 450 := by sorry

end NUMINAMATH_CALUDE_probability_ratio_l2569_256952


namespace NUMINAMATH_CALUDE_puppies_given_away_l2569_256986

def initial_puppies : ℕ := 12
def current_puppies : ℕ := 5

theorem puppies_given_away : initial_puppies - current_puppies = 7 := by
  sorry

end NUMINAMATH_CALUDE_puppies_given_away_l2569_256986


namespace NUMINAMATH_CALUDE_square_side_length_l2569_256917

theorem square_side_length (perimeter : ℝ) (h : perimeter = 28) : 
  perimeter / 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l2569_256917


namespace NUMINAMATH_CALUDE_product_mod_seventeen_l2569_256922

theorem product_mod_seventeen : (2021 * 2023 * 2025 * 2027 * 2029) % 17 = 13 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seventeen_l2569_256922


namespace NUMINAMATH_CALUDE_sunzi_wood_measurement_problem_l2569_256948

/-- Represents the wood measurement problem from "The Mathematical Classic of Sunzi" -/
theorem sunzi_wood_measurement_problem 
  (x y : ℝ) -- x: length of rope, y: length of wood
  (h1 : x - y = 4.5) -- condition: 4.5 feet of rope left when measuring
  (h2 : (1/2) * x + 1 = y) -- condition: 1 foot left when rope is folded in half
  : (x - y = 4.5) ∧ ((1/2) * x + 1 = y) := by
  sorry

end NUMINAMATH_CALUDE_sunzi_wood_measurement_problem_l2569_256948


namespace NUMINAMATH_CALUDE_ratio_problem_l2569_256916

theorem ratio_problem (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) 
  (h_eq : x / y + y / x = 4) : 
  (x + 2 * y) / (x - 2 * y) = Real.sqrt (11 / 3) := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l2569_256916


namespace NUMINAMATH_CALUDE_equation_solution_l2569_256976

theorem equation_solution : ∃ x : ℚ, 3 * x + 6 = |(-19 + 5)| ∧ x = 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2569_256976


namespace NUMINAMATH_CALUDE_collinear_relation_vector_relation_l2569_256919

-- Define points A, B, and C
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (3, -1)
def C : ℝ → ℝ → ℝ × ℝ := λ a b => (a, b)

-- Define vector from A to B
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Define vector from A to C
def AC (a b : ℝ) : ℝ × ℝ := (a - A.1, b - A.2)

-- Define collinearity condition
def collinear (a b : ℝ) : Prop :=
  ∃ (t : ℝ), AC a b = (t * AB.1, t * AB.2)

-- Theorem 1: If A, B, and C are collinear, then a = 2-b
theorem collinear_relation (a b : ℝ) :
  collinear a b → a = 2 - b := by sorry

-- Theorem 2: If AC = 2AB, then C = (5, -3)
theorem vector_relation :
  ∃ (a b : ℝ), AC a b = (2 * AB.1, 2 * AB.2) ∧ C a b = (5, -3) := by sorry

end NUMINAMATH_CALUDE_collinear_relation_vector_relation_l2569_256919


namespace NUMINAMATH_CALUDE_maximum_garden_area_l2569_256925

/-- Represents the dimensions of a rectangular garden -/
structure GardenDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular garden -/
def gardenArea (d : GardenDimensions) : ℝ := d.length * d.width

/-- Calculates the fence length required for three sides of a rectangular garden -/
def fenceLength (d : GardenDimensions) : ℝ := d.length + 2 * d.width

/-- The total available fencing -/
def totalFence : ℝ := 400

theorem maximum_garden_area :
  ∃ (d : GardenDimensions),
    fenceLength d = totalFence ∧
    ∀ (d' : GardenDimensions), fenceLength d' = totalFence → gardenArea d' ≤ gardenArea d ∧
    gardenArea d = 20000 := by
  sorry

end NUMINAMATH_CALUDE_maximum_garden_area_l2569_256925


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l2569_256946

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  2 * x^2 - 8 * y - 6 = -2 * y^2 - 8 * x

-- Define the center and radius
def is_center_radius (c d s : ℝ) : Prop :=
  ∀ (x y : ℝ), circle_equation x y ↔ (x - c)^2 + (y - d)^2 = s^2

-- Theorem statement
theorem circle_center_radius_sum :
  ∃ (c d s : ℝ), is_center_radius c d s ∧ c + d + s = Real.sqrt 7 :=
sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l2569_256946


namespace NUMINAMATH_CALUDE_a_completion_time_l2569_256911

def job_completion_time (a b c : ℝ) : Prop :=
  (1 / b = 8) ∧ 
  (1 / c = 12) ∧ 
  (2340 / (1 / a + 1 / b + 1 / c) = 780 / (1 / b))

theorem a_completion_time (a b c : ℝ) : 
  job_completion_time a b c → 1 / a = 6 := by
  sorry

end NUMINAMATH_CALUDE_a_completion_time_l2569_256911


namespace NUMINAMATH_CALUDE_unique_three_digit_number_divisible_by_11_l2569_256998

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def units_digit (n : ℕ) : ℕ := n % 10

def hundreds_digit (n : ℕ) : ℕ := (n / 100) % 10

def divisible_by (a b : ℕ) : Prop := b ∣ a

theorem unique_three_digit_number_divisible_by_11 :
  ∃! n : ℕ, is_three_digit n ∧ 
            units_digit n = 3 ∧ 
            hundreds_digit n = 6 ∧ 
            divisible_by n 11 ∧
            n = 693 :=
sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_divisible_by_11_l2569_256998


namespace NUMINAMATH_CALUDE_sqrt_50_between_consecutive_integers_l2569_256921

theorem sqrt_50_between_consecutive_integers :
  ∃ (n : ℕ), (n : ℝ) < Real.sqrt 50 ∧ Real.sqrt 50 < (n + 1 : ℝ) ∧ n * (n + 1) = 56 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_50_between_consecutive_integers_l2569_256921


namespace NUMINAMATH_CALUDE_heather_counts_209_l2569_256941

-- Define the range of numbers
def range : Set ℕ := {n | 1 ≤ n ∧ n ≤ 500}

-- Define Alice's skipping pattern
def aliceSkips (n : ℕ) : Prop := ∃ k, n = 5 * k - 2 ∧ 1 ≤ k ∧ k ≤ 100

-- Define the general skipping pattern for Barbara and the next 5 students
def otherSkips (n : ℕ) : Prop := ∃ m, n = 3 * m - 1 ∧ ¬(aliceSkips n)

-- Define Heather's number
def heatherNumber : ℕ := 209

-- Theorem statement
theorem heather_counts_209 :
  heatherNumber ∈ range ∧
  ¬(aliceSkips heatherNumber) ∧
  ¬(otherSkips heatherNumber) ∧
  ∀ n ∈ range, n ≠ heatherNumber → aliceSkips n ∨ otherSkips n :=
sorry

end NUMINAMATH_CALUDE_heather_counts_209_l2569_256941


namespace NUMINAMATH_CALUDE_largest_circle_area_l2569_256939

/-- The area of the largest circle formed from a string that fits exactly around a rectangle -/
theorem largest_circle_area (string_length : ℝ) (rectangle_area : ℝ) : 
  string_length = 60 →
  rectangle_area = 200 →
  (∃ (x y : ℝ), x * y = rectangle_area ∧ 2 * (x + y) = string_length) →
  (π * (string_length / (2 * π))^2 : ℝ) = 900 / π :=
by sorry

end NUMINAMATH_CALUDE_largest_circle_area_l2569_256939


namespace NUMINAMATH_CALUDE_sum_of_digits_divisible_by_11_l2569_256908

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: Among 39 sequential natural numbers, there's always one with sum of digits divisible by 11 -/
theorem sum_of_digits_divisible_by_11 (N : ℕ) : 
  ∃ k : ℕ, k ∈ Finset.range 39 ∧ (sum_of_digits (N + k) % 11 = 0) := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_divisible_by_11_l2569_256908


namespace NUMINAMATH_CALUDE_fourth_number_in_sequence_l2569_256938

theorem fourth_number_in_sequence (seq : Fin 6 → ℕ) 
  (h1 : seq 0 = 29)
  (h2 : seq 1 = 35)
  (h3 : seq 2 = 41)
  (h5 : seq 4 = 53)
  (h6 : seq 5 = 59)
  (h_arithmetic : ∀ i : Fin 4, seq (i + 1) - seq i = seq 1 - seq 0) :
  seq 3 = 47 := by
  sorry

end NUMINAMATH_CALUDE_fourth_number_in_sequence_l2569_256938


namespace NUMINAMATH_CALUDE_emily_small_gardens_l2569_256975

theorem emily_small_gardens (total_seeds : ℕ) (big_garden_seeds : ℕ) (seeds_per_small_garden : ℕ) 
  (h1 : total_seeds = 41)
  (h2 : big_garden_seeds = 29)
  (h3 : seeds_per_small_garden = 4) :
  (total_seeds - big_garden_seeds) / seeds_per_small_garden = 3 := by
  sorry

end NUMINAMATH_CALUDE_emily_small_gardens_l2569_256975


namespace NUMINAMATH_CALUDE_work_completion_time_l2569_256980

/-- Given a work that can be completed by A in 14 days and by A and B together in 10 days,
    prove that B can complete the work alone in 35 days. -/
theorem work_completion_time (work : ℝ) (A B : ℝ → ℝ) : 
  (A work = work / 14) →
  (A work + B work = work / 10) →
  B work = work / 35 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2569_256980


namespace NUMINAMATH_CALUDE_tickets_found_is_zero_l2569_256900

/-- The number of carnival games --/
def num_games : ℕ := 5

/-- The value of each ticket in dollars --/
def ticket_value : ℕ := 3

/-- The total value of all tickets in dollars --/
def total_value : ℕ := 30

/-- The number of tickets won from each game --/
def tickets_per_game : ℕ := total_value / (num_games * ticket_value)

/-- The number of tickets found on the floor --/
def tickets_found : ℕ := total_value - (num_games * tickets_per_game * ticket_value)

theorem tickets_found_is_zero : tickets_found = 0 := by
  sorry

end NUMINAMATH_CALUDE_tickets_found_is_zero_l2569_256900


namespace NUMINAMATH_CALUDE_unique_function_satisfying_equation_l2569_256933

-- Define the property that a function must satisfy
def SatisfiesEquation (f : ℝ → ℝ) (k : ℝ) : Prop :=
  ∀ x y : ℝ, f (x + f (k * y)) = x + y

-- State the theorem
theorem unique_function_satisfying_equation (k : ℝ) (hk : k ≠ 0) :
  ∃! f : ℝ → ℝ, SatisfiesEquation f k ∧ f = id := by sorry

end NUMINAMATH_CALUDE_unique_function_satisfying_equation_l2569_256933


namespace NUMINAMATH_CALUDE_pear_problem_solution_l2569_256987

/-- Represents the pear selling problem --/
def PearProblem (initial_pears : ℝ) : Prop :=
  let sold_day1 := 0.20 * initial_pears
  let remaining_after_sale := initial_pears - sold_day1
  let thrown_day1 := 0.50 * remaining_after_sale
  let remaining_day2 := remaining_after_sale - thrown_day1
  let total_thrown := 0.72 * initial_pears
  let thrown_day2 := total_thrown - thrown_day1
  let sold_day2 := remaining_day2 - thrown_day2
  (sold_day2 / remaining_day2) = 0.20

/-- Theorem stating that the percentage of remaining pears sold on day 2 is 20% --/
theorem pear_problem_solution : 
  ∀ initial_pears : ℝ, initial_pears > 0 → PearProblem initial_pears :=
by
  sorry


end NUMINAMATH_CALUDE_pear_problem_solution_l2569_256987


namespace NUMINAMATH_CALUDE_green_bows_count_l2569_256947

theorem green_bows_count (total : ℕ) (white : ℕ) :
  white = 40 →
  (1 : ℚ) / 4 + (1 : ℚ) / 3 + (1 : ℚ) / 6 + (white : ℚ) / total = 1 →
  (1 : ℚ) / 6 * total = 27 :=
by sorry

end NUMINAMATH_CALUDE_green_bows_count_l2569_256947


namespace NUMINAMATH_CALUDE_prob_second_red_given_first_red_l2569_256942

/-- The probability of drawing a red ball on the second draw, given that a red ball was drawn on the first -/
theorem prob_second_red_given_first_red 
  (total_balls : ℕ) 
  (red_balls : ℕ) 
  (white_balls : ℕ) 
  (h1 : total_balls = 6)
  (h2 : red_balls = 4)
  (h3 : white_balls = 2)
  (h4 : total_balls = red_balls + white_balls) :
  (red_balls - 1 : ℚ) / (total_balls - 1) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_prob_second_red_given_first_red_l2569_256942


namespace NUMINAMATH_CALUDE_quadratic_always_has_real_roots_min_a_for_positive_integer_roots_min_a_is_zero_l2569_256992

/-- The quadratic equation x^2 - (a+2)x + (a+1) = 0 -/
def quadratic (a : ℝ) (x : ℝ) : ℝ := x^2 - (a+2)*x + (a+1)

/-- The discriminant of the quadratic equation -/
def discriminant (a : ℝ) : ℝ := (a+2)^2 - 4*(a+1)

theorem quadratic_always_has_real_roots (a : ℝ) :
  discriminant a ≥ 0 := by sorry

theorem min_a_for_positive_integer_roots :
  ∀ a : ℕ, (∃ x y : ℕ, x ≠ y ∧ quadratic a x = 0 ∧ quadratic a y = 0) →
  a ≥ 0 := by sorry

theorem min_a_is_zero :
  ∃ a : ℕ, a = 0 ∧
  (∃ x y : ℕ, x ≠ y ∧ quadratic a x = 0 ∧ quadratic a y = 0) ∧
  ∀ b : ℕ, b < a →
  ¬(∃ x y : ℕ, x ≠ y ∧ quadratic b x = 0 ∧ quadratic b y = 0) := by sorry

end NUMINAMATH_CALUDE_quadratic_always_has_real_roots_min_a_for_positive_integer_roots_min_a_is_zero_l2569_256992


namespace NUMINAMATH_CALUDE_beef_order_proof_l2569_256953

/-- Calculates the amount of beef ordered given the costs and total amount --/
def beef_ordered (beef_cost chicken_cost total_cost : ℚ) : ℚ :=
  total_cost / (beef_cost + 2 * chicken_cost)

/-- Proves that the amount of beef ordered is 1000 pounds given the problem conditions --/
theorem beef_order_proof :
  beef_ordered 8 3 14000 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_beef_order_proof_l2569_256953


namespace NUMINAMATH_CALUDE_parallelogram_area_l2569_256937

/-- The area of a parallelogram with base 36 cm and height 24 cm is 864 square centimeters. -/
theorem parallelogram_area : 
  let base : ℝ := 36
  let height : ℝ := 24
  let area := base * height
  area = 864 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l2569_256937


namespace NUMINAMATH_CALUDE_common_root_is_one_l2569_256954

/-- Given two quadratic equations with coefficients a and b that have exactly one common root, prove that this root is 1 -/
theorem common_root_is_one (a b : ℝ) 
  (h : ∃! x : ℝ, (x^2 + a*x + b = 0) ∧ (x^2 + b*x + a = 0)) : 
  ∃ x : ℝ, (x^2 + a*x + b = 0) ∧ (x^2 + b*x + a = 0) ∧ x = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_common_root_is_one_l2569_256954


namespace NUMINAMATH_CALUDE_strawberry_weight_difference_l2569_256915

/-- The weight difference between Marco's and his dad's strawberries -/
def weight_difference (marco_weight dad_weight : ℝ) : ℝ :=
  marco_weight - dad_weight

/-- The total weight of Marco's and his dad's strawberries -/
def total_weight (marco_weight dad_weight : ℝ) : ℝ :=
  marco_weight + dad_weight

theorem strawberry_weight_difference :
  ∀ (marco_weight dad_weight : ℝ),
  marco_weight = 30 →
  total_weight marco_weight dad_weight = 47 →
  weight_difference marco_weight dad_weight = 13 := by
sorry

end NUMINAMATH_CALUDE_strawberry_weight_difference_l2569_256915


namespace NUMINAMATH_CALUDE_car_ordering_theorem_l2569_256961

/-- Represents a car with its speeds in different road segments -/
structure Car where
  citySpeed : ℝ
  nonCitySpeed : ℝ

/-- Represents a point on the road -/
structure RoadPoint where
  cityDistance : ℝ
  nonCityDistance : ℝ

/-- The theorem statement -/
theorem car_ordering_theorem 
  (cars : Fin 10 → Car) 
  (points : Fin 2011 → RoadPoint) :
  ∃ i j, i ≠ j ∧ 
    (∀ (c₁ c₂ : Fin 10), 
      (cars c₁).citySpeed / (cars c₂).citySpeed < (cars c₁).nonCitySpeed / (cars c₂).nonCitySpeed →
      ((points i).cityDistance / (cars c₁).citySpeed + (points i).nonCityDistance / (cars c₁).nonCitySpeed <
       (points i).cityDistance / (cars c₂).citySpeed + (points i).nonCityDistance / (cars c₂).nonCitySpeed) ↔
      ((points j).cityDistance / (cars c₁).citySpeed + (points j).nonCityDistance / (cars c₁).nonCitySpeed <
       (points j).cityDistance / (cars c₂).citySpeed + (points j).nonCityDistance / (cars c₂).nonCitySpeed)) :=
by sorry

end NUMINAMATH_CALUDE_car_ordering_theorem_l2569_256961


namespace NUMINAMATH_CALUDE_digit_puzzle_proof_l2569_256904

theorem digit_puzzle_proof (P Q R S : ℕ) : 
  (P < 10 ∧ Q < 10 ∧ R < 10 ∧ S < 10) →
  (10 * P + Q) + (10 * R + P) = 10 * S + P →
  (10 * P + Q) - (10 * R + P) = P →
  S = 0 := by
  sorry

end NUMINAMATH_CALUDE_digit_puzzle_proof_l2569_256904


namespace NUMINAMATH_CALUDE_line_through_points_l2569_256907

/-- Given a line y = ax + b passing through points (3, -2) and (7, 14), prove that a + b = -10 -/
theorem line_through_points (a b : ℝ) : 
  (∀ x y : ℝ, y = a * x + b) → 
  (-2 : ℝ) = a * 3 + b → 
  (14 : ℝ) = a * 7 + b → 
  a + b = -10 := by sorry

end NUMINAMATH_CALUDE_line_through_points_l2569_256907


namespace NUMINAMATH_CALUDE_paper_stack_thickness_sheets_in_six_cm_stack_l2569_256912

/-- Calculates the number of sheets in a stack of paper given the thickness of the stack and the number of sheets per unit thickness. -/
def sheets_in_stack (stack_thickness : ℝ) (sheets_per_unit : ℝ) : ℝ :=
  stack_thickness * sheets_per_unit

theorem paper_stack_thickness (bundle_sheets : ℝ) (bundle_thickness : ℝ) (stack_thickness : ℝ) :
  bundle_sheets > 0 → bundle_thickness > 0 → stack_thickness > 0 →
  sheets_in_stack stack_thickness (bundle_sheets / bundle_thickness) = 
    (stack_thickness / bundle_thickness) * bundle_sheets := by
  sorry

/-- The main theorem that proves the number of sheets in a 6 cm stack given a 400-sheet bundle is 4 cm thick. -/
theorem sheets_in_six_cm_stack : 
  sheets_in_stack 6 (400 / 4) = 600 := by
  sorry

end NUMINAMATH_CALUDE_paper_stack_thickness_sheets_in_six_cm_stack_l2569_256912


namespace NUMINAMATH_CALUDE_triangle_side_length_l2569_256965

noncomputable section

/-- Given a triangle ABC with BC = 1, if sin(A/2) * cos(B/2) = sin(B/2) * cos(A/2), then AC = sin(A) / sin(C) -/
theorem triangle_side_length (A B C : Real) (BC : Real) (h1 : BC = 1) 
  (h2 : Real.sin (A / 2) * Real.cos (B / 2) = Real.sin (B / 2) * Real.cos (A / 2)) :
  ∃ (AC : Real), AC = Real.sin A / Real.sin C :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2569_256965


namespace NUMINAMATH_CALUDE_sum_and_round_l2569_256990

def round_to_nearest_hundred (x : ℤ) : ℤ :=
  100 * ((x + 50) / 100)

theorem sum_and_round : round_to_nearest_hundred (128 + 264) = 400 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_round_l2569_256990


namespace NUMINAMATH_CALUDE_complex_number_parts_opposite_l2569_256968

theorem complex_number_parts_opposite (b : ℝ) : 
  let z : ℂ := (2 - b * I) / (1 + 2 * I)
  (z.re = -z.im) → b = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_parts_opposite_l2569_256968


namespace NUMINAMATH_CALUDE_sum_of_digits_B_is_seven_l2569_256932

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- A is the sum of digits of 4444^4444 -/
def A : ℕ := sumOfDigits (4444^4444)

/-- B is the sum of digits of A -/
def B : ℕ := sumOfDigits A

/-- Theorem: The sum of digits of B is 7 -/
theorem sum_of_digits_B_is_seven : sumOfDigits B = 7 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_B_is_seven_l2569_256932


namespace NUMINAMATH_CALUDE_point_on_unit_circle_l2569_256969

theorem point_on_unit_circle (s : ℝ) :
  let x := (s^2 - 1) / (s^2 + 1)
  let y := 2*s / (s^2 + 1)
  x^2 + y^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_point_on_unit_circle_l2569_256969


namespace NUMINAMATH_CALUDE_weight_loss_percentage_l2569_256967

def weight_before : ℝ := 840
def weight_after : ℝ := 546

theorem weight_loss_percentage : 
  (weight_before - weight_after) / weight_before * 100 = 35 := by
  sorry

end NUMINAMATH_CALUDE_weight_loss_percentage_l2569_256967


namespace NUMINAMATH_CALUDE_two_fifths_in_four_fifths_minus_one_tenth_l2569_256927

theorem two_fifths_in_four_fifths_minus_one_tenth : 
  (4/5 - 1/10) / (2/5) = 7/4 := by
  sorry

end NUMINAMATH_CALUDE_two_fifths_in_four_fifths_minus_one_tenth_l2569_256927


namespace NUMINAMATH_CALUDE_campsite_tents_l2569_256964

theorem campsite_tents (total : ℕ) (south : ℕ) (north : ℕ) : 
  total = 900 →
  south = 200 →
  north + 2 * north + 4 * north + south = total →
  north = 100 := by
sorry

end NUMINAMATH_CALUDE_campsite_tents_l2569_256964


namespace NUMINAMATH_CALUDE_triangle_theorem_l2569_256960

noncomputable section

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition for the triangle -/
def triangle_condition (t : Triangle) : Prop :=
  2 * t.a * Real.sin t.A = (2 * t.b + t.c) * Real.sin t.B + (2 * t.c + t.b) * Real.sin t.C

/-- The theorem to be proved -/
theorem triangle_theorem (t : Triangle) 
  (h1 : triangle_condition t) 
  (h2 : t.a = 2 * Real.sqrt 3) : 
  t.A = 2 * Real.pi / 3 ∧ 
  (∀ (s : ℝ), s = (1/2) * t.b * t.c * Real.sin t.A → s ≤ Real.sqrt 3) ∧
  (∃ (s : ℝ), s = (1/2) * t.b * t.c * Real.sin t.A ∧ s = Real.sqrt 3) :=
by sorry

end

end NUMINAMATH_CALUDE_triangle_theorem_l2569_256960


namespace NUMINAMATH_CALUDE_quadratic_equation_result_l2569_256978

theorem quadratic_equation_result (m : ℝ) (h : m^2 + 2*m = 3) : 4*m^2 + 8*m - 1 = 11 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_result_l2569_256978


namespace NUMINAMATH_CALUDE_average_speed_two_part_trip_l2569_256910

/-- Calculates the average speed of a two-part trip -/
theorem average_speed_two_part_trip
  (total_distance : ℝ)
  (distance1 : ℝ)
  (speed1 : ℝ)
  (distance2 : ℝ)
  (speed2 : ℝ)
  (h1 : total_distance = distance1 + distance2)
  (h2 : distance1 = 35)
  (h3 : distance2 = 35)
  (h4 : speed1 = 48)
  (h5 : speed2 = 24)
  (h6 : total_distance = 70) :
  ∃ (avg_speed : ℝ), abs (avg_speed - 32) < 0.1 ∧
  avg_speed = total_distance / (distance1 / speed1 + distance2 / speed2) := by
  sorry


end NUMINAMATH_CALUDE_average_speed_two_part_trip_l2569_256910


namespace NUMINAMATH_CALUDE_initial_pencils_count_l2569_256979

/-- The number of pencils Sara added to the drawer -/
def pencils_added : ℕ := 100

/-- The total number of pencils in the drawer after Sara's addition -/
def total_pencils : ℕ := 215

/-- The initial number of pencils in the drawer -/
def initial_pencils : ℕ := total_pencils - pencils_added

theorem initial_pencils_count : initial_pencils = 115 := by
  sorry

end NUMINAMATH_CALUDE_initial_pencils_count_l2569_256979


namespace NUMINAMATH_CALUDE_flower_production_percentage_l2569_256944

theorem flower_production_percentage
  (daisy_seeds sunflower_seeds : ℕ)
  (daisy_germination_rate sunflower_germination_rate : ℚ)
  (flowering_plants : ℕ)
  (h1 : daisy_seeds = 25)
  (h2 : sunflower_seeds = 25)
  (h3 : daisy_germination_rate = 0.6)
  (h4 : sunflower_germination_rate = 0.8)
  (h5 : flowering_plants = 28)
  : (flowering_plants : ℚ) / ((daisy_germination_rate * daisy_seeds + sunflower_germination_rate * sunflower_seeds) : ℚ) = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_flower_production_percentage_l2569_256944


namespace NUMINAMATH_CALUDE_point_coordinates_in_third_quadrant_l2569_256920

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The third quadrant of the 2D plane -/
def ThirdQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- Distance from a point to the x-axis -/
def DistToXAxis (p : Point) : ℝ :=
  |p.y|

/-- Distance from a point to the y-axis -/
def DistToYAxis (p : Point) : ℝ :=
  |p.x|

theorem point_coordinates_in_third_quadrant :
  ∃ (p : Point), ThirdQuadrant p ∧ DistToXAxis p = 2 ∧ DistToYAxis p = 3 → p = Point.mk (-3) (-2) := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_in_third_quadrant_l2569_256920


namespace NUMINAMATH_CALUDE_clubsuit_equation_solution_l2569_256924

/-- Definition of the clubsuit operation -/
def clubsuit (A B : ℝ) : ℝ := 4 * A + 3 * B + 7

/-- Theorem stating that A clubsuit 6 = 85 when A = 15 -/
theorem clubsuit_equation_solution :
  clubsuit 15 6 = 85 := by sorry

end NUMINAMATH_CALUDE_clubsuit_equation_solution_l2569_256924


namespace NUMINAMATH_CALUDE_inverse_sum_theorem_l2569_256988

-- Define the function f(x) = x|x|
def f (x : ℝ) : ℝ := x * |x|

-- State the theorem
theorem inverse_sum_theorem : 
  ∃ (a b : ℝ), f a = 9 ∧ f b = -64 ∧ a + b = -5 := by sorry

end NUMINAMATH_CALUDE_inverse_sum_theorem_l2569_256988


namespace NUMINAMATH_CALUDE_circle_area_increase_l2569_256945

theorem circle_area_increase (r : ℝ) (h : r > 0) :
  let new_radius := 1.5 * r
  let original_area := π * r^2
  let new_area := π * new_radius^2
  (new_area - original_area) / original_area = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_increase_l2569_256945


namespace NUMINAMATH_CALUDE_weekly_rate_is_190_l2569_256931

/-- Represents the car rental problem --/
structure CarRental where
  dailyRate : ℕ
  totalDays : ℕ
  totalCost : ℕ
  weeklyRate : ℕ

/-- The car rental agency's pricing policy --/
def rentalPolicy (r : CarRental) : Prop :=
  r.dailyRate = 30 ∧
  r.totalDays = 11 ∧
  r.totalCost = 310 ∧
  r.weeklyRate = r.totalCost - (r.totalDays - 7) * r.dailyRate

/-- Theorem stating that the weekly rate is $190 --/
theorem weekly_rate_is_190 (r : CarRental) :
  rentalPolicy r → r.weeklyRate = 190 := by
  sorry

#check weekly_rate_is_190

end NUMINAMATH_CALUDE_weekly_rate_is_190_l2569_256931


namespace NUMINAMATH_CALUDE_reciprocal_square_sum_l2569_256999

theorem reciprocal_square_sum : (((1 : ℚ) / 4 + 1 / 6) ^ 2)⁻¹ = 144 / 25 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_square_sum_l2569_256999


namespace NUMINAMATH_CALUDE_square_perimeter_l2569_256923

theorem square_perimeter (area : ℝ) (side : ℝ) (h1 : area = 625) (h2 : side * side = area) : 
  4 * side = 100 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l2569_256923


namespace NUMINAMATH_CALUDE_matrix_commute_equality_l2569_256977

theorem matrix_commute_equality (A B : Matrix (Fin 2) (Fin 2) ℝ) :
  A + B = A * B →
  A * B = ![![1, 2], ![3, 4]] →
  (A * B = B * A) →
  B * A = ![![1, 2], ![3, 4]] := by
  sorry

end NUMINAMATH_CALUDE_matrix_commute_equality_l2569_256977


namespace NUMINAMATH_CALUDE_special_square_side_length_l2569_256909

/-- Square with special points -/
structure SpecialSquare where
  /-- Side length of the square -/
  side : ℝ
  /-- Point M on side CD -/
  m : ℝ × ℝ
  /-- Point E where AM intersects the circumscribed circle -/
  e : ℝ × ℝ

/-- The theorem statement -/
theorem special_square_side_length (s : SpecialSquare) :
  /- Point M is on side CD -/
  s.m.1 = s.side ∧ 0 ≤ s.m.2 ∧ s.m.2 ≤ s.side ∧
  /- CM:MD = 1:3 -/
  s.m.2 = s.side / 4 ∧
  /- E is on the circumscribed circle -/
  (s.e.1 - s.side / 2)^2 + (s.e.2 - s.side / 2)^2 = 2 * (s.side / 2)^2 ∧
  /- Area of triangle ACE is 14 -/
  1/2 * s.e.1 * s.e.2 = 14 →
  /- The side length of the square is 10 -/
  s.side = 10 := by
  sorry

end NUMINAMATH_CALUDE_special_square_side_length_l2569_256909


namespace NUMINAMATH_CALUDE_james_budget_theorem_l2569_256950

/-- James's budget and expenses --/
def budget : ℝ := 1000
def food_percentage : ℝ := 0.22
def accommodation_percentage : ℝ := 0.15
def entertainment_percentage : ℝ := 0.18
def transportation_percentage : ℝ := 0.12
def clothes_percentage : ℝ := 0.08
def miscellaneous_percentage : ℝ := 0.05

/-- Theorem: James's savings percentage and combined expenses --/
theorem james_budget_theorem :
  let food := budget * food_percentage
  let accommodation := budget * accommodation_percentage
  let entertainment := budget * entertainment_percentage
  let transportation := budget * transportation_percentage
  let clothes := budget * clothes_percentage
  let miscellaneous := budget * miscellaneous_percentage
  let total_spent := food + accommodation + entertainment + transportation + clothes + miscellaneous
  let savings := budget - total_spent
  let savings_percentage := (savings / budget) * 100
  let combined_expenses := entertainment + transportation + miscellaneous
  savings_percentage = 20 ∧ combined_expenses = 350 := by
  sorry

end NUMINAMATH_CALUDE_james_budget_theorem_l2569_256950


namespace NUMINAMATH_CALUDE_family_event_handshakes_l2569_256993

/-- The number of sets of twins at the family event -/
def twin_sets : ℕ := 12

/-- The number of sets of triplets at the family event -/
def triplet_sets : ℕ := 4

/-- The total number of twins at the family event -/
def total_twins : ℕ := twin_sets * 2

/-- The total number of triplets at the family event -/
def total_triplets : ℕ := triplet_sets * 3

/-- The fraction of triplets each twin shakes hands with -/
def twin_triplet_fraction : ℚ := 1 / 3

/-- The fraction of twins each triplet shakes hands with -/
def triplet_twin_fraction : ℚ := 2 / 3

/-- The total number of unique handshakes at the family event -/
def total_handshakes : ℕ := 462

theorem family_event_handshakes :
  (total_twins * (total_twins - 2) / 2) +
  (total_triplets * (total_triplets - 3) / 2) +
  ((total_twins * (total_triplets * twin_triplet_fraction).floor +
    total_triplets * (total_twins * triplet_twin_fraction).floor) / 2) =
  total_handshakes := by sorry

end NUMINAMATH_CALUDE_family_event_handshakes_l2569_256993


namespace NUMINAMATH_CALUDE_special_triangle_angle_difference_l2569_256940

/-- A triangle with special angle properties -/
structure SpecialTriangle where
  /-- The smallest angle of the triangle -/
  a : ℕ
  /-- The middle angle of the triangle -/
  b : ℕ
  /-- The largest angle of the triangle -/
  c : ℕ
  /-- One of the angles is a prime number -/
  h1 : Prime a ∨ Prime b ∨ Prime c
  /-- Two of the angles are squares of prime numbers -/
  h2 : ∃ p q : ℕ, Prime p ∧ Prime q ∧ 
       ((b = p^2 ∧ c = q^2) ∨ (a = p^2 ∧ c = q^2) ∨ (a = p^2 ∧ b = q^2))
  /-- The sum of the angles is 180 degrees -/
  h3 : a + b + c = 180
  /-- The angles are in ascending order -/
  h4 : a ≤ b ∧ b ≤ c

/-- The theorem stating the difference between the largest and smallest angles -/
theorem special_triangle_angle_difference (t : SpecialTriangle) : t.c - t.a = 167 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_angle_difference_l2569_256940


namespace NUMINAMATH_CALUDE_truck_fuel_relationship_l2569_256985

/-- Represents the fuel consumption model of a truck -/
structure TruckFuelModel where
  tankCapacity : ℝ
  fuelConsumptionRate : ℝ

/-- Calculates the remaining fuel in the tank after a given time -/
def remainingFuel (model : TruckFuelModel) (time : ℝ) : ℝ :=
  model.tankCapacity - model.fuelConsumptionRate * time

/-- Theorem: The relationship between remaining fuel and traveling time for the given truck -/
theorem truck_fuel_relationship (model : TruckFuelModel) 
  (h1 : model.tankCapacity = 60)
  (h2 : model.fuelConsumptionRate = 8) :
  ∀ t : ℝ, remainingFuel model t = 60 - 8 * t :=
by sorry

end NUMINAMATH_CALUDE_truck_fuel_relationship_l2569_256985


namespace NUMINAMATH_CALUDE_unique_integer_pair_existence_l2569_256943

theorem unique_integer_pair_existence (a b : ℤ) :
  ∃! (x y : ℤ), (x + 2*y - a)^2 + (2*x - y - b)^2 ≤ 1 := by sorry

end NUMINAMATH_CALUDE_unique_integer_pair_existence_l2569_256943


namespace NUMINAMATH_CALUDE_rectangle_to_equilateral_triangle_l2569_256903

/-- Given a rectangle with length L and width W, and an equilateral triangle with side s,
    if both shapes have the same area A, then s = √(4LW/√3) -/
theorem rectangle_to_equilateral_triangle (L W s A : ℝ) (h1 : A = L * W) 
    (h2 : A = (s^2 * Real.sqrt 3) / 4) : s = Real.sqrt ((4 * L * W) / Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_to_equilateral_triangle_l2569_256903


namespace NUMINAMATH_CALUDE_f_derivative_l2569_256995

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x) * Real.cos x

theorem f_derivative :
  deriv f = fun x => Real.exp (2 * x) * (2 * Real.cos x - Real.sin x) := by
  sorry

end NUMINAMATH_CALUDE_f_derivative_l2569_256995
