import Mathlib

namespace NUMINAMATH_CALUDE_shifted_roots_polynomial_l719_71964

theorem shifted_roots_polynomial (a b c : ℂ) : 
  (a^3 - 4*a^2 + 6*a - 3 = 0) →
  (b^3 - 4*b^2 + 6*b - 3 = 0) →
  (c^3 - 4*c^2 + 6*c - 3 = 0) →
  ∀ x, (x - (a + 3)) * (x - (b + 3)) * (x - (c + 3)) = x^3 - 13*x^2 + 57*x - 84 :=
by sorry

end NUMINAMATH_CALUDE_shifted_roots_polynomial_l719_71964


namespace NUMINAMATH_CALUDE_adults_fed_is_22_l719_71956

/-- Represents the resources and feeding capabilities of a community center -/
structure CommunityCenter where
  soup_cans : ℕ
  bread_loaves : ℕ
  adults_per_can : ℕ
  children_per_can : ℕ
  adults_per_loaf : ℕ
  children_per_loaf : ℕ

/-- Calculates the number of adults that can be fed with remaining resources -/
def adults_fed_after_children (cc : CommunityCenter) (children_to_feed : ℕ) : ℕ :=
  let cans_for_children := (children_to_feed + cc.children_per_can - 1) / cc.children_per_can
  let remaining_cans := cc.soup_cans - cans_for_children
  let adults_fed_by_cans := remaining_cans * cc.adults_per_can
  let adults_fed_by_bread := cc.bread_loaves * cc.adults_per_loaf
  adults_fed_by_cans + adults_fed_by_bread

/-- Theorem stating that 22 adults can be fed with remaining resources -/
theorem adults_fed_is_22 (cc : CommunityCenter) (h1 : cc.soup_cans = 8) (h2 : cc.bread_loaves = 2)
    (h3 : cc.adults_per_can = 4) (h4 : cc.children_per_can = 7) (h5 : cc.adults_per_loaf = 3)
    (h6 : cc.children_per_loaf = 4) :
    adults_fed_after_children cc 24 = 22 := by
  sorry

end NUMINAMATH_CALUDE_adults_fed_is_22_l719_71956


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l719_71914

theorem isosceles_triangle_perimeter (m : ℝ) : 
  (2 : ℝ) ^ 2 - (5 + m) * 2 + 5 * m = 0 →
  ∃ (a b : ℝ), a ^ 2 - (5 + m) * a + 5 * m = 0 ∧
                b ^ 2 - (5 + m) * b + 5 * m = 0 ∧
                a ≠ b ∧
                (a = 2 ∨ b = 2) ∧
                (a + a + b = 12 ∨ a + b + b = 12) := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l719_71914


namespace NUMINAMATH_CALUDE_triangle_inequality_l719_71960

theorem triangle_inequality (a b c : ℝ) 
  (triangle_cond : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) 
  (abc_cond : a * b * c = 1) : 
  (Real.sqrt (b + c - a)) / a + (Real.sqrt (c + a - b)) / b + (Real.sqrt (a + b - c)) / c ≥ a + b + c :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l719_71960


namespace NUMINAMATH_CALUDE_factorization_equality_l719_71950

theorem factorization_equality (x y : ℝ) : 2 * x^2 * y - 8 * y = 2 * y * (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l719_71950


namespace NUMINAMATH_CALUDE_max_value_complex_expression_l719_71904

theorem max_value_complex_expression (z : ℂ) (h : Complex.abs (z + Complex.I) = 2) :
  ∃ (max_val : ℝ), max_val = 4 * Real.sqrt 3 ∧
  ∀ (w : ℂ), Complex.abs (w + Complex.I) = 2 →
    Complex.abs ((w - (2 - Complex.I))^2 * (w - Complex.I)) ≤ max_val :=
sorry

end NUMINAMATH_CALUDE_max_value_complex_expression_l719_71904


namespace NUMINAMATH_CALUDE_rectangle_no_integer_points_l719_71988

-- Define the rectangle type
structure Rectangle where
  a : ℝ
  b : ℝ
  h : a < b

-- Define the property of having no integer points
def hasNoIntegerPoints (r : Rectangle) : Prop :=
  ∀ x y : ℤ, ¬(0 ≤ x ∧ x ≤ r.b ∧ 0 ≤ y ∧ y ≤ r.a)

-- Theorem statement
theorem rectangle_no_integer_points (r : Rectangle) :
  hasNoIntegerPoints r ↔ min r.a r.b < 1 := by sorry

end NUMINAMATH_CALUDE_rectangle_no_integer_points_l719_71988


namespace NUMINAMATH_CALUDE_log_expression_equals_four_l719_71900

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_expression_equals_four :
  (log10 2)^2 + log10 2 * log10 50 + log10 25 = 4 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equals_four_l719_71900


namespace NUMINAMATH_CALUDE_train_length_l719_71916

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 90 → time_s = 9 → speed_kmh * (5/18) * time_s = 225 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l719_71916


namespace NUMINAMATH_CALUDE_circle_equation_l719_71970

-- Define the circle C
def circle_C : Set (ℝ × ℝ) := {p : ℝ × ℝ | ∃ (center : ℝ × ℝ), (center.1 - p.1)^2 + (center.2 - p.2)^2 = 4 ∧ 3 * center.1 - center.2 - 3 = 0}

-- Define points A and B
def point_A : ℝ × ℝ := (2, 5)
def point_B : ℝ × ℝ := (4, 3)

-- Theorem statement
theorem circle_equation : 
  (∀ p : ℝ × ℝ, p ∈ circle_C ↔ (p.1 - 2)^2 + (p.2 - 3)^2 = 4) ∧
  point_A ∈ circle_C ∧
  point_B ∈ circle_C :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l719_71970


namespace NUMINAMATH_CALUDE_s₂_is_zero_l719_71985

-- Define the polynomial division operation
def poly_div (p q : ℝ → ℝ) : (ℝ → ℝ) × ℝ := sorry

-- Define p₁(x) and s₁
def p₁_and_s₁ : (ℝ → ℝ) × ℝ := poly_div (λ x => x^6) (λ x => x - 1/2)

def p₁ : ℝ → ℝ := (p₁_and_s₁.1)
def s₁ : ℝ := (p₁_and_s₁.2)

-- Define p₂(x) and s₂
def p₂_and_s₂ : (ℝ → ℝ) × ℝ := poly_div p₁ (λ x => x - 1/2)

def p₂ : ℝ → ℝ := (p₂_and_s₂.1)
def s₂ : ℝ := (p₂_and_s₂.2)

-- The theorem to prove
theorem s₂_is_zero : s₂ = 0 := by sorry

end NUMINAMATH_CALUDE_s₂_is_zero_l719_71985


namespace NUMINAMATH_CALUDE_horse_race_theorem_l719_71979

def horse_primes : List Nat := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def is_valid_subset (s : List Nat) : Prop :=
  s.length = 5 ∧ s.toFinset ⊆ horse_primes.toFinset

def least_common_time (s : List Nat) : Nat :=
  s.foldl Nat.lcm 1

def sum_of_digits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem horse_race_theorem :
  ∃ (s : List Nat), is_valid_subset s ∧
    (∀ (t : List Nat), is_valid_subset t → least_common_time s ≤ least_common_time t) ∧
    least_common_time s = 2310 ∧
    sum_of_digits (least_common_time s) = 6 :=
  sorry

end NUMINAMATH_CALUDE_horse_race_theorem_l719_71979


namespace NUMINAMATH_CALUDE_intersection_E_F_l719_71989

open Set Real

def E : Set ℝ := {θ | cos θ < sin θ ∧ 0 ≤ θ ∧ θ ≤ 2 * π}
def F : Set ℝ := {θ | tan θ < sin θ}

theorem intersection_E_F : E ∩ F = Ioo (π / 2) π := by
  sorry

end NUMINAMATH_CALUDE_intersection_E_F_l719_71989


namespace NUMINAMATH_CALUDE_opposite_direction_speed_l719_71991

/-- Given two people moving in opposite directions, this theorem proves
    the speed of the second person given the conditions of the problem. -/
theorem opposite_direction_speed
  (time : ℝ)
  (distance : ℝ)
  (speed1 : ℝ)
  (h1 : time = 4)
  (h2 : distance = 28)
  (h3 : speed1 = 4)
  (h4 : distance = time * (speed1 + speed2)) :
  speed2 = 3 := by
  sorry


end NUMINAMATH_CALUDE_opposite_direction_speed_l719_71991


namespace NUMINAMATH_CALUDE_min_value_of_f_l719_71936

/-- The quadratic function f(x) = 3x^2 + 8x + 15 -/
def f (x : ℝ) : ℝ := 3 * x^2 + 8 * x + 15

/-- The minimum value of f(x) is 29/3 -/
theorem min_value_of_f : 
  ∀ x : ℝ, f x ≥ 29/3 ∧ ∃ x₀ : ℝ, f x₀ = 29/3 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l719_71936


namespace NUMINAMATH_CALUDE_line_circle_intersection_l719_71951

/-- The line y = kx + 1 intersects the circle (x-2)² + (y-1)² = 4 at points P and Q.
    If the distance between P and Q is greater than or equal to 2√2,
    then k is in the interval [-1, 1]. -/
theorem line_circle_intersection (k : ℝ) : 
  (∃ P Q : ℝ × ℝ, 
    (P.2 = k * P.1 + 1) ∧ 
    (Q.2 = k * Q.1 + 1) ∧
    ((P.1 - 2)^2 + (P.2 - 1)^2 = 4) ∧
    ((Q.1 - 2)^2 + (Q.2 - 1)^2 = 4) ∧
    ((P.1 - Q.1)^2 + (P.2 - Q.2)^2 ≥ 8)) →
  -1 ≤ k ∧ k ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l719_71951


namespace NUMINAMATH_CALUDE_greatest_whole_number_inequality_l719_71931

theorem greatest_whole_number_inequality (x : ℤ) : 
  (∀ y : ℤ, y > x → ¬(6*y - 5 < 7 - 3*y)) → 
  (6*x - 5 < 7 - 3*x) → 
  x = 1 := by
sorry

end NUMINAMATH_CALUDE_greatest_whole_number_inequality_l719_71931


namespace NUMINAMATH_CALUDE_perpendicular_bisector_equation_l719_71901

def is_perpendicular_bisector (a b c : ℝ) (p1 p2 : ℝ × ℝ) : Prop :=
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  a * midpoint.1 + b * midpoint.2 + c = 0

theorem perpendicular_bisector_equation (b : ℝ) :
  is_perpendicular_bisector 1 (-1) (-b) (2, 4) (10, -6) → b = 7 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_equation_l719_71901


namespace NUMINAMATH_CALUDE_find_y_value_l719_71911

theorem find_y_value (x y z : ℝ) 
  (h1 : x^2 * y = z) 
  (h2 : x / y = 36)
  (h3 : Real.sqrt (x * y) = z)
  (hx : x > 0)
  (hy : y > 0)
  (hz : z > 0) :
  y = 1 / 14.7 := by
sorry

end NUMINAMATH_CALUDE_find_y_value_l719_71911


namespace NUMINAMATH_CALUDE_bodies_of_water_is_six_l719_71963

/-- Represents the aquatic reserve scenario -/
structure AquaticReserve where
  total_fish : ℕ
  fish_per_body : ℕ
  h_total : total_fish = 1050
  h_per_body : fish_per_body = 175

/-- The number of bodies of water in the aquatic reserve -/
def bodies_of_water (reserve : AquaticReserve) : ℕ :=
  reserve.total_fish / reserve.fish_per_body

/-- Theorem stating that the number of bodies of water is 6 -/
theorem bodies_of_water_is_six (reserve : AquaticReserve) : 
  bodies_of_water reserve = 6 := by
  sorry


end NUMINAMATH_CALUDE_bodies_of_water_is_six_l719_71963


namespace NUMINAMATH_CALUDE_pizza_class_size_l719_71948

/-- Proves that the number of students in a class is 68, given the pizza ordering scenario. -/
theorem pizza_class_size :
  let pizza_slices : ℕ := 18  -- Number of slices in a large pizza
  let total_pizzas : ℕ := 6   -- Total number of pizzas ordered
  let cheese_leftover : ℕ := 8  -- Number of cheese slices leftover
  let onion_leftover : ℕ := 4   -- Number of onion slices leftover
  let cheese_per_student : ℕ := 2  -- Number of cheese slices per student
  let onion_per_student : ℕ := 1   -- Number of onion slices per student

  let total_slices : ℕ := pizza_slices * total_pizzas
  let used_cheese : ℕ := total_slices - cheese_leftover
  let used_onion : ℕ := total_slices - onion_leftover

  (∃ (num_students : ℕ),
    num_students * cheese_per_student = used_cheese ∧
    num_students * onion_per_student = used_onion) →
  (∃! (num_students : ℕ), num_students = 68) :=
by sorry

end NUMINAMATH_CALUDE_pizza_class_size_l719_71948


namespace NUMINAMATH_CALUDE_math_test_questions_l719_71986

/-- Proves that the total number of questions in a math test is 60 -/
theorem math_test_questions : ∃ N : ℕ,
  (N : ℚ) * (80 : ℚ) / 100 + 35 - N / 2 = N - 7 ∧
  N = 60 := by
  sorry

end NUMINAMATH_CALUDE_math_test_questions_l719_71986


namespace NUMINAMATH_CALUDE_square_perimeter_l719_71980

theorem square_perimeter (side_length : ℝ) (h : side_length = 40) :
  4 * side_length = 160 :=
by sorry

end NUMINAMATH_CALUDE_square_perimeter_l719_71980


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l719_71932

/-- Given a hyperbola with the following properties:
    - Equation: x²/a² - y²/b² = 1
    - a > 0, b > 0
    - Focal distance is 8
    - Left vertex A is at (-a, 0)
    - Point B is at (0, b)
    - Right focus F is at (4, 0)
    - Dot product of BA and BF equals 2a
    The eccentricity of the hyperbola is 2 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let A : ℝ × ℝ := (-a, 0)
  let B : ℝ × ℝ := (0, b)
  let F : ℝ × ℝ := (4, 0)
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → 
    (x - (-a))^2 + y^2 = (x - 4)^2 + y^2) →
  (B.1 - A.1) * (F.1 - B.1) + (B.2 - A.2) * (F.2 - B.2) = 2 * a →
  4 / a = 2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l719_71932


namespace NUMINAMATH_CALUDE_cereal_serving_size_l719_71915

/-- Represents the number of cups of cereal in a box -/
def total_cups : ℕ := 18

/-- Represents the number of servings in a box -/
def total_servings : ℕ := 9

/-- Represents the number of cups per serving -/
def cups_per_serving : ℚ := total_cups / total_servings

theorem cereal_serving_size : cups_per_serving = 2 := by
  sorry

end NUMINAMATH_CALUDE_cereal_serving_size_l719_71915


namespace NUMINAMATH_CALUDE_race_problem_l719_71977

/-- The race problem -/
theorem race_problem (jack_first_half jack_second_half jill_total : ℕ) 
  (h1 : jack_first_half = 19)
  (h2 : jack_second_half = 6)
  (h3 : jill_total = 32) :
  jill_total - (jack_first_half + jack_second_half) = 7 := by
  sorry

end NUMINAMATH_CALUDE_race_problem_l719_71977


namespace NUMINAMATH_CALUDE_smallest_disguisable_triangle_two_sides_perfect_squares_l719_71930

/-- A triangle with integer side lengths a, b, and c is disguisable if there exists a similar triangle
    with side lengths d, a, b where d ≥ a ≥ b > c -/
def IsDisguisableTriangle (a b c : ℕ) : Prop :=
  ∃ d : ℚ, d ≥ a ∧ a ≥ b ∧ b > c ∧ (d : ℚ) / a = (a : ℚ) / b ∧ (a : ℚ) / b = (b : ℚ) / c

/-- The perimeter of a triangle with side lengths a, b, and c -/
def Perimeter (a b c : ℕ) : ℕ := a + b + c

/-- A number is a perfect square if it's equal to some integer squared -/
def IsPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem smallest_disguisable_triangle :
  ∀ a b c : ℕ, IsDisguisableTriangle a b c →
    Perimeter a b c ≥ 19 ∧
    (Perimeter a b c = 19 → (a, b, c) = (9, 6, 4)) :=
sorry

theorem two_sides_perfect_squares :
  ∀ a b c : ℕ, IsDisguisableTriangle a b c →
    (∀ k : ℕ, k < a → ¬IsDisguisableTriangle k (k * b / a) (k * c / a)) →
    (IsPerfectSquare a ∧ IsPerfectSquare c) ∨
    (IsPerfectSquare a ∧ IsPerfectSquare b) ∨
    (IsPerfectSquare b ∧ IsPerfectSquare c) :=
sorry

end NUMINAMATH_CALUDE_smallest_disguisable_triangle_two_sides_perfect_squares_l719_71930


namespace NUMINAMATH_CALUDE_smallest_n_for_integer_sqrt_12n_l719_71929

theorem smallest_n_for_integer_sqrt_12n :
  ∀ n : ℕ+, (∃ k : ℕ, k^2 = 12*n) → (∀ m : ℕ+, m < n → ¬∃ j : ℕ, j^2 = 12*m) → n = 3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_integer_sqrt_12n_l719_71929


namespace NUMINAMATH_CALUDE_donated_books_count_l719_71966

/-- Represents the number of books in the library over time --/
structure LibraryBooks where
  initial_old : ℕ
  bought_two_years_ago : ℕ
  bought_last_year : ℕ
  current_total : ℕ

/-- Calculates the number of old books donated --/
def books_donated (lib : LibraryBooks) : ℕ :=
  lib.initial_old + lib.bought_two_years_ago + lib.bought_last_year - lib.current_total

/-- Theorem stating the number of old books donated --/
theorem donated_books_count (lib : LibraryBooks) 
  (h1 : lib.initial_old = 500)
  (h2 : lib.bought_two_years_ago = 300)
  (h3 : lib.bought_last_year = lib.bought_two_years_ago + 100)
  (h4 : lib.current_total = 1000) :
  books_donated lib = 200 := by
  sorry

#eval books_donated ⟨500, 300, 400, 1000⟩

end NUMINAMATH_CALUDE_donated_books_count_l719_71966


namespace NUMINAMATH_CALUDE_cube_root_unity_sum_l719_71998

/-- Given a nonreal cube root of unity ω, prove that (1 - ω + ω^2)^6 + (1 + ω - ω^2)^6 = 128 -/
theorem cube_root_unity_sum (ω : ℂ) (h1 : ω^3 = 1) (h2 : ω ≠ 1) :
  (1 - ω + ω^2)^6 + (1 + ω - ω^2)^6 = 128 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_unity_sum_l719_71998


namespace NUMINAMATH_CALUDE_nine_twin_functions_l719_71944

-- Define the function f(x) = 2x^2 + 1
def f (x : ℝ) : ℝ := 2 * x^2 + 1

-- Define the range set
def range_set : Set ℝ := {5, 19}

-- Define the property for a valid domain
def is_valid_domain (D : Set ℝ) : Prop :=
  (∀ x ∈ D, f x ∈ range_set) ∧ 
  (∀ y ∈ range_set, ∃ x ∈ D, f x = y)

-- State the theorem
theorem nine_twin_functions :
  ∃! (domains : Finset (Set ℝ)), 
    Finset.card domains = 9 ∧ 
    (∀ D ∈ domains, is_valid_domain D) ∧
    (∀ D : Set ℝ, is_valid_domain D → D ∈ domains) :=
sorry

end NUMINAMATH_CALUDE_nine_twin_functions_l719_71944


namespace NUMINAMATH_CALUDE_square_sum_given_sum_and_product_l719_71973

theorem square_sum_given_sum_and_product (a b : ℝ) (h1 : a + b = 5) (h2 : a * b = 3) :
  2 * a^2 + 2 * b^2 = 38 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_sum_and_product_l719_71973


namespace NUMINAMATH_CALUDE_trigonometric_identity_l719_71954

theorem trigonometric_identity (h1 : Real.tan (10 * π / 180) * Real.tan (20 * π / 180) + 
                                     Real.tan (20 * π / 180) * Real.tan (60 * π / 180) + 
                                     Real.tan (60 * π / 180) * Real.tan (10 * π / 180) = 1)
                               (h2 : Real.tan (5 * π / 180) * Real.tan (10 * π / 180) + 
                                     Real.tan (10 * π / 180) * Real.tan (75 * π / 180) + 
                                     Real.tan (75 * π / 180) * Real.tan (5 * π / 180) = 1) :
  Real.tan (8 * π / 180) * Real.tan (12 * π / 180) + 
  Real.tan (12 * π / 180) * Real.tan (70 * π / 180) + 
  Real.tan (70 * π / 180) * Real.tan (8 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l719_71954


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l719_71934

theorem quadratic_inequality_condition (a : ℝ) :
  (∀ x : ℝ, a * x^2 - a * x + 1 > 0) ↔ (0 < a ∧ a < 4) := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l719_71934


namespace NUMINAMATH_CALUDE_complex_division_result_l719_71962

theorem complex_division_result : (1 + 2*I) / (1 - 2*I) = -3/5 + 4/5*I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_result_l719_71962


namespace NUMINAMATH_CALUDE_vertex_is_correct_l719_71946

/-- The quadratic function f(x) = 3(x+5)^2 - 2 -/
def f (x : ℝ) : ℝ := 3 * (x + 5)^2 - 2

/-- The vertex of the quadratic function f -/
def vertex : ℝ × ℝ := (-5, -2)

theorem vertex_is_correct : 
  (∀ x : ℝ, f x ≥ f (vertex.1)) ∧ f (vertex.1) = vertex.2 := by
  sorry

end NUMINAMATH_CALUDE_vertex_is_correct_l719_71946


namespace NUMINAMATH_CALUDE_sum_of_special_sequence_l719_71922

/-- Given positive real numbers a and b that form an arithmetic sequence with -2,
    and can also form a geometric sequence after rearrangement, prove their sum is 5 -/
theorem sum_of_special_sequence (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∃ d : ℝ, (a = b + d ∧ b = -2 + d) ∨ (b = a + d ∧ a = -2 + d) ∨ (a = -2 + d ∧ -2 = b + d)) →
  (∃ r : ℝ, r ≠ 0 ∧ ((a = b * r ∧ b = -2 * r) ∨ (b = a * r ∧ a = -2 * r) ∨ (a = -2 * r ∧ -2 = b * r))) →
  a + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_special_sequence_l719_71922


namespace NUMINAMATH_CALUDE_inscribed_circles_radii_l719_71906

/-- Three circles inscribed in a corner -/
structure InscribedCircles where
  r : ℝ  -- radius of small circle
  a : ℝ  -- distance from center of small circle to corner vertex
  x : ℝ  -- radius of medium circle
  y : ℝ  -- radius of large circle

/-- Conditions for the inscribed circles -/
def valid_inscribed_circles (c : InscribedCircles) : Prop :=
  c.r > 0 ∧ c.a > c.r ∧ c.x > c.r ∧ c.y > c.x

/-- Theorem stating the radii of medium and large circles -/
theorem inscribed_circles_radii (c : InscribedCircles) 
  (h : valid_inscribed_circles c) : 
  c.x = c.a * c.r / (c.a - c.r) ∧ 
  c.y = c.a^2 * c.r / (c.a - c.r)^2 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circles_radii_l719_71906


namespace NUMINAMATH_CALUDE_victors_total_money_l719_71987

/-- Victor's initial money in dollars -/
def initial_money : ℕ := 10

/-- Victor's allowance in dollars -/
def allowance : ℕ := 8

/-- Theorem: Victor's total money is $18 -/
theorem victors_total_money : initial_money + allowance = 18 := by
  sorry

end NUMINAMATH_CALUDE_victors_total_money_l719_71987


namespace NUMINAMATH_CALUDE_whitewashing_cost_l719_71923

def room_length : ℝ := 25
def room_width : ℝ := 15
def room_height : ℝ := 12
def door_height : ℝ := 6
def door_width : ℝ := 3
def window_height : ℝ := 4
def window_width : ℝ := 3
def num_windows : ℕ := 3
def cost_per_sqft : ℝ := 3

theorem whitewashing_cost :
  let wall_area := 2 * (room_length * room_height + room_width * room_height)
  let door_area := door_height * door_width
  let window_area := num_windows * (window_height * window_width)
  let whitewash_area := wall_area - door_area - window_area
  whitewash_area * cost_per_sqft = 2718 :=
by sorry

end NUMINAMATH_CALUDE_whitewashing_cost_l719_71923


namespace NUMINAMATH_CALUDE_initial_honey_amount_l719_71994

/-- The amount of honey remaining after each replacement, as a fraction of the initial amount -/
def honey_fraction : ℕ → ℚ
  | 0 => 1
  | n + 1 => honey_fraction n * (4/5)

/-- The proposition that the initial amount of honey is 1250 grams given the problem conditions -/
theorem initial_honey_amount (initial_honey : ℚ) :
  (honey_fraction 4 * initial_honey = 512) →
  initial_honey = 1250 := by
  sorry

end NUMINAMATH_CALUDE_initial_honey_amount_l719_71994


namespace NUMINAMATH_CALUDE_two_integers_sum_l719_71997

theorem two_integers_sum (x y : ℕ+) : 
  (x : ℤ) - (y : ℤ) = 5 ∧ 
  (x : ℕ) * y = 180 → 
  (x : ℕ) + y = 25 := by
sorry

end NUMINAMATH_CALUDE_two_integers_sum_l719_71997


namespace NUMINAMATH_CALUDE_complex_power_magnitude_l719_71939

theorem complex_power_magnitude : Complex.abs ((2 - 2 * Complex.I) ^ 6) = 512 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_magnitude_l719_71939


namespace NUMINAMATH_CALUDE_sqrt_3_irrational_l719_71957

theorem sqrt_3_irrational : Irrational (Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_irrational_l719_71957


namespace NUMINAMATH_CALUDE_total_wheat_weight_l719_71965

def wheat_weights : List ℝ := [91, 91, 91.5, 89, 91.2, 91.3, 88.7, 88.8, 91.8, 91.1]
def standard_weight : ℝ := 90

theorem total_wheat_weight :
  (wheat_weights.sum) = 905.4 := by
  sorry

end NUMINAMATH_CALUDE_total_wheat_weight_l719_71965


namespace NUMINAMATH_CALUDE_triangle_problem_l719_71925

theorem triangle_problem (a b c A B C : ℝ) : 
  -- Conditions
  (2 * Real.sin (7 * π / 6) * Real.sin (π / 6 + C) + Real.cos C = -1 / 2) →
  (c = Real.sqrt 13) →
  (1 / 2 * a * b * Real.sin C = 3 * Real.sqrt 3) →
  -- Conclusions
  (C = π / 3) ∧ 
  (Real.sin A + Real.sin B = 7 * Real.sqrt 39 / 26) := by
sorry


end NUMINAMATH_CALUDE_triangle_problem_l719_71925


namespace NUMINAMATH_CALUDE_sequence_inequality_l719_71909

theorem sequence_inequality (a : ℕ → ℕ) (n N : ℕ) 
  (h1 : ∀ m k, a (m + k) ≤ a m + a k) 
  (h2 : N ≥ n) :
  a n + a N ≤ n * a 1 + (N / n) * a n :=
sorry

end NUMINAMATH_CALUDE_sequence_inequality_l719_71909


namespace NUMINAMATH_CALUDE_coin_bag_total_l719_71918

theorem coin_bag_total (p : ℕ) : ∃ (p : ℕ), 
  (0.01 * p + 0.05 * (3 * p) + 0.50 * (12 * p) : ℚ) = 616 := by
  sorry

end NUMINAMATH_CALUDE_coin_bag_total_l719_71918


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l719_71969

/-- An arithmetic sequence with 5 terms -/
structure ArithmeticSequence5 where
  a : ℝ  -- first term
  b : ℝ  -- second term
  c : ℝ  -- third term (middle term)
  d : ℝ  -- fourth term
  e : ℝ  -- fifth term
  is_arithmetic : ∃ r : ℝ, b = a + r ∧ c = b + r ∧ d = c + r ∧ e = d + r

/-- The theorem stating that in an arithmetic sequence with first term 20, 
    last term 50, and middle term y, the value of y is 35 -/
theorem arithmetic_sequence_middle_term 
  (seq : ArithmeticSequence5) 
  (h1 : seq.a = 20) 
  (h2 : seq.e = 50) : 
  seq.c = 35 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l719_71969


namespace NUMINAMATH_CALUDE_quilt_shaded_fraction_l719_71975

/-- Represents a square quilt block -/
structure QuiltBlock where
  totalSquares : ℕ
  dividedSquares : ℕ
  shadePerDividedSquare : ℚ

/-- The fraction of the quilt block that is shaded -/
def shadedFraction (q : QuiltBlock) : ℚ :=
  (q.dividedSquares : ℚ) * q.shadePerDividedSquare / q.totalSquares

/-- Theorem stating that for a quilt block with 16 total squares, 
    4 divided squares, and half of each divided square shaded,
    the shaded fraction is 1/8 -/
theorem quilt_shaded_fraction :
  ∀ (q : QuiltBlock), 
    q.totalSquares = 16 → 
    q.dividedSquares = 4 → 
    q.shadePerDividedSquare = 1/2 →
    shadedFraction q = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_quilt_shaded_fraction_l719_71975


namespace NUMINAMATH_CALUDE_repeating_decimal_problem_l719_71945

theorem repeating_decimal_problem (a b : ℕ) (h1 : a < 10) (h2 : b < 10) : 
  66 * (1 + (10 * a + b) / 99) - 66 * (1 + (10 * a + b) / 100) = 1/2 → 
  10 * a + b = 75 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_problem_l719_71945


namespace NUMINAMATH_CALUDE_angle_terminal_side_problem_tan_equation_problem_l719_71903

-- Problem 1
theorem angle_terminal_side_problem (α : Real) 
  (h : Real.tan α = -3/4 ∧ Real.sin α = 3/5) :
  (Real.cos (π/2 + α) * Real.sin (-π - α)) / 
  (Real.cos (2019*π/2 - α) * Real.tan (9*π/2 + α)) = 9/20 := by sorry

-- Problem 2
theorem tan_equation_problem (x : Real) (h : Real.tan (π/4 + x) = 2018) :
  1 / Real.cos (2*x) + Real.tan (2*x) = 2018 := by sorry

end NUMINAMATH_CALUDE_angle_terminal_side_problem_tan_equation_problem_l719_71903


namespace NUMINAMATH_CALUDE_sunflower_seeds_weight_l719_71955

/-- The weight of a bag of sunflower seeds in grams -/
def bag_weight : ℝ := 250

/-- The number of bags -/
def num_bags : ℕ := 8

/-- Conversion factor from grams to kilograms -/
def grams_to_kg : ℝ := 1000

theorem sunflower_seeds_weight :
  (bag_weight * num_bags) / grams_to_kg = 2 := by
  sorry

end NUMINAMATH_CALUDE_sunflower_seeds_weight_l719_71955


namespace NUMINAMATH_CALUDE_salary_increase_l719_71907

/-- Given an original salary and a salary increase, 
    proves that the new salary is $90,000 if the percent increase is 38.46153846153846% --/
theorem salary_increase (S : ℝ) (increase : ℝ) : 
  increase = 25000 →
  (increase / S) * 100 = 38.46153846153846 →
  S + increase = 90000 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_l719_71907


namespace NUMINAMATH_CALUDE_fermat_5_divisible_by_641_fermat_numbers_coprime_l719_71993

-- Define Fermat numbers
def F (n : ℕ) : ℕ := 2^(2^n) + 1

-- Theorem 1: F_5 is divisible by 641
theorem fermat_5_divisible_by_641 : 
  641 ∣ F 5 := by sorry

-- Theorem 2: F_k and F_n are relatively prime for k ≠ n
theorem fermat_numbers_coprime {k n : ℕ} (h : k ≠ n) : 
  Nat.gcd (F k) (F n) = 1 := by sorry

end NUMINAMATH_CALUDE_fermat_5_divisible_by_641_fermat_numbers_coprime_l719_71993


namespace NUMINAMATH_CALUDE_job_applicant_age_range_l719_71927

/-- The maximum number of different integer ages within a range defined by
    an average age and a number of standard deviations. -/
def max_different_ages (average_age : ℕ) (std_dev : ℕ) (num_std_devs : ℕ) : ℕ :=
  2 * num_std_devs * std_dev + 1

/-- Theorem stating that for the given problem parameters, 
    the maximum number of different ages is 41. -/
theorem job_applicant_age_range : 
  max_different_ages 40 10 2 = 41 := by
  sorry

end NUMINAMATH_CALUDE_job_applicant_age_range_l719_71927


namespace NUMINAMATH_CALUDE_cubic_inequality_l719_71981

theorem cubic_inequality (a b : ℝ) (h : a < b) :
  a^3 - 3*a ≤ b^3 - 3*b + 4 ∧
  (a^3 - 3*a = b^3 - 3*b + 4 ↔ a = -1 ∧ b = 1) :=
by sorry

end NUMINAMATH_CALUDE_cubic_inequality_l719_71981


namespace NUMINAMATH_CALUDE_certain_number_equation_l719_71913

theorem certain_number_equation (x : ℚ) : 4 / (1 + 3 / x) = 1 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_equation_l719_71913


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l719_71999

def M : Set Int := {-1, 0, 1}
def N : Set Int := {0, 1, 2}

theorem union_of_M_and_N : M ∪ N = {-1, 0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l719_71999


namespace NUMINAMATH_CALUDE_total_votes_l719_71959

theorem total_votes (veggies_votes : ℕ) (meat_votes : ℕ) 
  (h1 : veggies_votes = 337) (h2 : meat_votes = 335) : 
  veggies_votes + meat_votes = 672 := by
  sorry

end NUMINAMATH_CALUDE_total_votes_l719_71959


namespace NUMINAMATH_CALUDE_b_share_is_seven_fifteenths_l719_71967

/-- A partnership with four partners A, B, C, and D -/
structure Partnership where
  total_capital : ℝ
  a_share : ℝ
  b_share : ℝ
  c_share : ℝ
  d_share : ℝ
  total_profit : ℝ
  a_profit : ℝ

/-- The conditions of the partnership -/
def partnership_conditions (p : Partnership) : Prop :=
  p.a_share = (1/3) * p.total_capital ∧
  p.c_share = (1/5) * p.total_capital ∧
  p.d_share = p.total_capital - (p.a_share + p.b_share + p.c_share) ∧
  p.total_profit = 2430 ∧
  p.a_profit = 810

/-- Theorem stating B's share of the capital -/
theorem b_share_is_seven_fifteenths (p : Partnership) 
  (h : partnership_conditions p) : 
  p.b_share = (7/15) * p.total_capital := by
  sorry

end NUMINAMATH_CALUDE_b_share_is_seven_fifteenths_l719_71967


namespace NUMINAMATH_CALUDE_shaded_to_white_ratio_is_five_thirds_l719_71974

/-- A nested square figure where vertices of inner squares are at the midpoints of the sides of the outer squares. -/
structure NestedSquareFigure where
  /-- The number of nested squares in the figure -/
  num_squares : ℕ
  /-- The side length of the outermost square -/
  outer_side_length : ℝ
  /-- Assumption that the figure is constructed with vertices at midpoints -/
  vertices_at_midpoints : Bool

/-- The ratio of the shaded area to the white area in the nested square figure -/
def shaded_to_white_ratio (figure : NestedSquareFigure) : ℚ :=
  5 / 3

/-- Theorem stating that the ratio of shaded to white area is 5/3 -/
theorem shaded_to_white_ratio_is_five_thirds (figure : NestedSquareFigure) 
  (h : figure.vertices_at_midpoints = true) : 
  shaded_to_white_ratio figure = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_shaded_to_white_ratio_is_five_thirds_l719_71974


namespace NUMINAMATH_CALUDE_sixth_root_unity_product_l719_71952

theorem sixth_root_unity_product (r : ℂ) (h1 : r^6 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) = 6 := by
  sorry

end NUMINAMATH_CALUDE_sixth_root_unity_product_l719_71952


namespace NUMINAMATH_CALUDE_max_angle_A_l719_71976

/-- Represents the side lengths of a triangle sequence -/
structure TriangleSequence where
  a : ℕ → ℝ
  b : ℕ → ℝ
  c : ℕ → ℝ

/-- Conditions for the triangle sequence -/
def ValidTriangleSequence (t : TriangleSequence) : Prop :=
  (t.b 1 > t.c 1) ∧
  (t.b 1 + t.c 1 = 2 * t.a 1) ∧
  (∀ n, t.a (n + 1) = t.a n) ∧
  (∀ n, t.b (n + 1) = (t.c n + t.a n) / 2) ∧
  (∀ n, t.c (n + 1) = (t.b n + t.a n) / 2)

/-- The angle A_n in the triangle sequence -/
noncomputable def angleA (t : TriangleSequence) (n : ℕ) : ℝ :=
  Real.arccos ((t.b n ^ 2 + t.c n ^ 2 - t.a n ^ 2) / (2 * t.b n * t.c n))

/-- The theorem stating the maximum value of angle A_n -/
theorem max_angle_A (t : TriangleSequence) (h : ValidTriangleSequence t) :
    (∀ n, angleA t n ≤ π / 3) ∧ (∃ n, angleA t n = π / 3) := by
  sorry


end NUMINAMATH_CALUDE_max_angle_A_l719_71976


namespace NUMINAMATH_CALUDE_increasing_function_condition_l719_71953

/-- A function f(x) = x - 5/x - a*ln(x) is increasing on [1, +∞) if and only if a ≤ 2√5 -/
theorem increasing_function_condition (a : ℝ) :
  (∀ x : ℝ, x ≥ 1 → Monotone (fun x => x - 5 / x - a * Real.log x)) ↔ a ≤ 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_increasing_function_condition_l719_71953


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l719_71928

theorem polynomial_division_remainder : ∃ (q r : Polynomial ℝ),
  (X^5 + 3*X^3 + 1 : Polynomial ℝ) = (X + 1)^2 * q + r ∧
  r.degree < 2 ∧
  r = 5*X + 9 := by
sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l719_71928


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l719_71920

/-- A regular polygon with perimeter 150 cm and side length 15 cm has 10 sides -/
theorem regular_polygon_sides (P : ℝ) (s : ℝ) (n : ℕ) : 
  P = 150 → s = 15 → P = n * s → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l719_71920


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l719_71949

def repeating_decimal_to_fraction (d : ℚ) : ℚ := d

theorem repeating_decimal_sum (a b : ℕ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : Nat.gcd a b = 1) (h4 : repeating_decimal_to_fraction (35/99 : ℚ) = a / b) : 
  a + b = 134 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l719_71949


namespace NUMINAMATH_CALUDE_min_value_n_minus_m_l719_71938

open Real

noncomputable def f (x : ℝ) : ℝ := log (x / 2) + 1 / 2

noncomputable def g (x : ℝ) : ℝ := exp (x - 2)

theorem min_value_n_minus_m :
  (∀ m : ℝ, ∃ n : ℝ, n > 0 ∧ g m = f n) →
  (∃ m n : ℝ, n > 0 ∧ g m = f n ∧ n - m = log 2) ∧
  (∀ m n : ℝ, n > 0 → g m = f n → n - m ≥ log 2) :=
sorry

end NUMINAMATH_CALUDE_min_value_n_minus_m_l719_71938


namespace NUMINAMATH_CALUDE_min_period_sin_2x_plus_pi_third_l719_71971

/-- The minimum positive period of y = sin(2x + π/3) is π -/
theorem min_period_sin_2x_plus_pi_third (x : ℝ) :
  let f : ℝ → ℝ := λ x => Real.sin (2 * x + π / 3)
  ∃ T : ℝ, T > 0 ∧ (∀ t : ℝ, f (x + T) = f x) ∧
    (∀ S : ℝ, S > 0 ∧ (∀ t : ℝ, f (x + S) = f x) → T ≤ S) ∧
    T = π :=
by sorry

end NUMINAMATH_CALUDE_min_period_sin_2x_plus_pi_third_l719_71971


namespace NUMINAMATH_CALUDE_a_minus_b_value_l719_71984

theorem a_minus_b_value (a b : ℝ) (h1 : |a| = 8) (h2 : |b| = 5) (h3 : a + b > 0) :
  a - b = 3 ∨ a - b = 13 := by
sorry

end NUMINAMATH_CALUDE_a_minus_b_value_l719_71984


namespace NUMINAMATH_CALUDE_third_root_of_cubic_l719_71990

theorem third_root_of_cubic (a b : ℚ) (h : a ≠ 0) :
  (∃ x : ℚ, a * x^3 - (3*a - b) * x^2 + 2*(a + b) * x - (6 - 2*a) = 0) ∧
  (a * 1^3 - (3*a - b) * 1^2 + 2*(a + b) * 1 - (6 - 2*a) = 0) ∧
  (a * (-3)^3 - (3*a - b) * (-3)^2 + 2*(a + b) * (-3) - (6 - 2*a) = 0) →
  ∃ x : ℚ, x ≠ 1 ∧ x ≠ -3 ∧ a * x^3 - (3*a - b) * x^2 + 2*(a + b) * x - (6 - 2*a) = 0 ∧ x = 322/21 :=
by sorry

end NUMINAMATH_CALUDE_third_root_of_cubic_l719_71990


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l719_71926

/-- Given a cubic equation x√x - 9x + 8√x - 2 = 0 with all roots real and positive,
    the sum of the squares of its roots is 65. -/
theorem sum_of_squares_of_roots : ∃ (r s t : ℝ), 
  (∀ x : ℝ, x > 0 → (x * Real.sqrt x - 9*x + 8*Real.sqrt x - 2 = 0) ↔ (x = r ∨ x = s ∨ x = t)) →
  r > 0 ∧ s > 0 ∧ t > 0 →
  r^2 + s^2 + t^2 = 65 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l719_71926


namespace NUMINAMATH_CALUDE_ellipse_equation_l719_71919

/-- An ellipse with center at origin, eccentricity √3/2, and one focus coinciding with
    the focus of the parabola x² = -4√3y has the equation x² + y²/4 = 1 -/
theorem ellipse_equation (x y : ℝ) : 
  let e : ℝ := Real.sqrt 3 / 2
  let c : ℝ := Real.sqrt 3  -- Distance from center to focus
  let a : ℝ := c / e        -- Semi-major axis
  let b : ℝ := Real.sqrt (a^2 - c^2)  -- Semi-minor axis
  (e = Real.sqrt 3 / 2) → 
  (c = Real.sqrt 3) →      -- Focus coincides with parabola focus
  (x^2 + y^2 / 4 = 1) ↔ (x^2 / a^2 + y^2 / b^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l719_71919


namespace NUMINAMATH_CALUDE_reciprocal_difference_sequence_l719_71935

theorem reciprocal_difference_sequence (a : ℕ → ℚ) :
  a 1 = 1/3 ∧
  (∀ n : ℕ, n > 1 → a n = 1 / (1 - a (n-1))) →
  a 2023 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_difference_sequence_l719_71935


namespace NUMINAMATH_CALUDE_light_reflection_theorem_l719_71995

/-- The reflection of a point with respect to a line --/
def reflect_point (p : ℝ × ℝ) (l : ℝ × ℝ × ℝ) : ℝ × ℝ := sorry

/-- Check if a point lies on a line --/
def point_on_line (p : ℝ × ℝ) (l : ℝ × ℝ × ℝ) : Prop := sorry

/-- The intersection point of two lines --/
def line_intersection (l1 l2 : ℝ × ℝ × ℝ) : ℝ × ℝ := sorry

theorem light_reflection_theorem :
  let A : ℝ × ℝ := (2, 3)
  let B : ℝ × ℝ := (1, 1)
  let mirror_line : ℝ × ℝ × ℝ := (1, 1, 1)  -- Represents x + y + 1 = 0
  let A' := reflect_point A mirror_line
  let C := line_intersection mirror_line ((A'.1 - B.1, A'.2 - B.2, A'.1 * B.2 - A'.2 * B.1))
  let incident_ray : ℝ × ℝ × ℝ := (5, -4, 2)  -- Represents 5x - 4y + 2 = 0
  let reflected_ray : ℝ × ℝ × ℝ := (4, -5, 1)  -- Represents 4x - 5y + 1 = 0
  point_on_line A incident_ray ∧
  point_on_line C incident_ray ∧
  point_on_line C reflected_ray ∧
  point_on_line B reflected_ray ∧
  point_on_line C mirror_line :=
by sorry

end NUMINAMATH_CALUDE_light_reflection_theorem_l719_71995


namespace NUMINAMATH_CALUDE_time_difference_is_56_minutes_l719_71961

def minnie_uphill_distance : ℝ := 12
def minnie_flat_distance : ℝ := 18
def minnie_downhill_distance : ℝ := 22
def minnie_uphill_speed : ℝ := 4
def minnie_flat_speed : ℝ := 25
def minnie_downhill_speed : ℝ := 32

def penny_downhill_distance : ℝ := 22
def penny_flat_distance : ℝ := 18
def penny_uphill_distance : ℝ := 12
def penny_downhill_speed : ℝ := 15
def penny_flat_speed : ℝ := 35
def penny_uphill_speed : ℝ := 8

theorem time_difference_is_56_minutes :
  let minnie_time := minnie_uphill_distance / minnie_uphill_speed +
                     minnie_flat_distance / minnie_flat_speed +
                     minnie_downhill_distance / minnie_downhill_speed
  let penny_time := penny_downhill_distance / penny_downhill_speed +
                    penny_flat_distance / penny_flat_speed +
                    penny_uphill_distance / penny_uphill_speed
  (minnie_time - penny_time) * 60 = 56 := by
  sorry

end NUMINAMATH_CALUDE_time_difference_is_56_minutes_l719_71961


namespace NUMINAMATH_CALUDE_ratio_of_squares_to_difference_l719_71958

theorem ratio_of_squares_to_difference (a b : ℝ) : 
  0 < b → 0 < a → a > b → (a^2 + b^2 = 7 * (a - b)) → (a / b = Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_squares_to_difference_l719_71958


namespace NUMINAMATH_CALUDE_rectangle_width_l719_71982

/-- Given a rectangle with length 13 cm and perimeter 50 cm, prove its width is 12 cm. -/
theorem rectangle_width (length : ℝ) (perimeter : ℝ) (width : ℝ) : 
  length = 13 → perimeter = 50 → perimeter = 2 * (length + width) → width = 12 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_l719_71982


namespace NUMINAMATH_CALUDE_negation_of_forall_positive_l719_71968

theorem negation_of_forall_positive (f : ℝ → ℝ) :
  (¬ ∀ x : ℝ, f x > 0) ↔ (∃ x : ℝ, f x ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_positive_l719_71968


namespace NUMINAMATH_CALUDE_arrangement_count_is_288_l719_71972

/-- The number of ways to arrange 4 mathematics books and 4 history books with constraints -/
def arrangement_count : ℕ :=
  let math_books : ℕ := 4
  let history_books : ℕ := 4
  let block_arrangements : ℕ := 2  -- Math block and history block
  let math_internal_arrangements : ℕ := Nat.factorial (math_books - 1)  -- Excluding M1
  let history_internal_arrangements : ℕ := Nat.factorial history_books
  block_arrangements * math_internal_arrangements * history_internal_arrangements

/-- Theorem stating that the number of valid arrangements is 288 -/
theorem arrangement_count_is_288 : arrangement_count = 288 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_is_288_l719_71972


namespace NUMINAMATH_CALUDE_profit_per_meter_is_55_l719_71940

/-- Profit per meter of cloth -/
def profit_per_meter (total_meters : ℕ) (total_profit : ℕ) : ℚ :=
  total_profit / total_meters

/-- Theorem: The profit per meter of cloth is 55 rupees -/
theorem profit_per_meter_is_55
  (total_meters : ℕ)
  (selling_price : ℕ)
  (total_profit : ℕ)
  (h1 : total_meters = 40)
  (h2 : selling_price = 8200)
  (h3 : total_profit = 2200) :
  profit_per_meter total_meters total_profit = 55 := by
  sorry

end NUMINAMATH_CALUDE_profit_per_meter_is_55_l719_71940


namespace NUMINAMATH_CALUDE_earnings_before_car_purchase_l719_71983

/-- Calculates the total earnings before saving enough to buy a car. -/
def totalEarningsBeforePurchase (monthlyEarnings : ℕ) (monthlySavings : ℕ) (carCost : ℕ) : ℕ :=
  (carCost / monthlySavings) * monthlyEarnings

/-- Theorem stating the total earnings before saving enough to buy the car. -/
theorem earnings_before_car_purchase :
  totalEarningsBeforePurchase 4000 500 45000 = 360000 := by
  sorry

end NUMINAMATH_CALUDE_earnings_before_car_purchase_l719_71983


namespace NUMINAMATH_CALUDE_smallest_integer_with_given_remainders_l719_71910

theorem smallest_integer_with_given_remainders : ∃ x : ℕ+, 
  (x : ℕ) % 6 = 5 ∧ 
  (x : ℕ) % 7 = 6 ∧ 
  (x : ℕ) % 8 = 7 ∧ 
  ∀ y : ℕ+, 
    (y : ℕ) % 6 = 5 → 
    (y : ℕ) % 7 = 6 → 
    (y : ℕ) % 8 = 7 → 
    x ≤ y :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_given_remainders_l719_71910


namespace NUMINAMATH_CALUDE_negative_a_power_five_l719_71996

theorem negative_a_power_five (a : ℝ) : (-a)^3 * (-a)^2 = -a^5 := by
  sorry

end NUMINAMATH_CALUDE_negative_a_power_five_l719_71996


namespace NUMINAMATH_CALUDE_seashell_difference_l719_71912

theorem seashell_difference (fred_shells tom_shells : ℕ) 
  (h1 : fred_shells = 43)
  (h2 : tom_shells = 15) :
  fred_shells - tom_shells = 28 := by
  sorry

end NUMINAMATH_CALUDE_seashell_difference_l719_71912


namespace NUMINAMATH_CALUDE_remainder_problem_l719_71992

theorem remainder_problem (y : ℤ) (h : y % 276 = 42) : y % 23 = 19 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l719_71992


namespace NUMINAMATH_CALUDE_box_paperclips_relation_small_box_medium_box_large_box_l719_71947

/-- Represents the number of paperclips a box can hold based on its volume -/
noncomputable def paperclips (volume : ℝ) : ℝ :=
  50 * (volume / 16)

theorem box_paperclips_relation (v : ℝ) :
  paperclips v = 50 * (v / 16) :=
by sorry

theorem small_box : paperclips 16 = 50 :=
by sorry

theorem medium_box : paperclips 32 = 100 :=
by sorry

theorem large_box : paperclips 64 = 200 :=
by sorry

end NUMINAMATH_CALUDE_box_paperclips_relation_small_box_medium_box_large_box_l719_71947


namespace NUMINAMATH_CALUDE_range_of_a_l719_71978

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - a| < 1 ↔ (1/2 : ℝ) < x ∧ x < (3/2 : ℝ)) ↔ 
  ((1/2 : ℝ) ≤ a ∧ a ≤ (3/2 : ℝ)) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l719_71978


namespace NUMINAMATH_CALUDE_segment_length_in_triangle_l719_71943

/-- Given a triangle with sides a, b, c, and three lines parallel to the sides
    intersecting at one point, with segments of length x cut off by the sides,
    prove that x = abc / (ab + bc + ac) -/
theorem segment_length_in_triangle (a b c x : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  x = (a * b * c) / (a * b + b * c + a * c) := by
  sorry

end NUMINAMATH_CALUDE_segment_length_in_triangle_l719_71943


namespace NUMINAMATH_CALUDE_min_value_xy_plus_four_over_xy_l719_71924

theorem min_value_xy_plus_four_over_xy (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) : 
  ∀ z w : ℝ, z > 0 → w > 0 → z + w = 2 → x * y + 4 / (x * y) ≤ z * w + 4 / (z * w) ∧ 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = 2 ∧ a * b + 4 / (a * b) = 5 :=
sorry

end NUMINAMATH_CALUDE_min_value_xy_plus_four_over_xy_l719_71924


namespace NUMINAMATH_CALUDE_stratified_sampling_size_l719_71917

theorem stratified_sampling_size (total_population : ℕ) (stratum_size : ℕ) (stratum_sample : ℕ) (h1 : total_population = 3600) (h2 : stratum_size = 1000) (h3 : stratum_sample = 25) : 
  (stratum_size : ℚ) / total_population * (total_sample : ℚ) = stratum_sample → total_sample = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_size_l719_71917


namespace NUMINAMATH_CALUDE_power_sum_geq_product_l719_71933

theorem power_sum_geq_product (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
  (sum_one : a + b + c = 1) : 
  a^4 + b^4 + c^4 ≥ a*b*c :=
by sorry

end NUMINAMATH_CALUDE_power_sum_geq_product_l719_71933


namespace NUMINAMATH_CALUDE_concert_songs_count_l719_71942

/-- Calculates the number of songs in a concert given the total duration,
    intermission time, regular song duration, and special song duration. -/
def number_of_songs (total_duration intermission_time regular_song_duration special_song_duration : ℕ) : ℕ :=
  let singing_time := total_duration - intermission_time
  let regular_songs_time := singing_time - special_song_duration
  (regular_songs_time / regular_song_duration) + 1

/-- Theorem stating that the number of songs in the given concert is 13. -/
theorem concert_songs_count :
  number_of_songs 80 10 5 10 = 13 := by
  sorry

end NUMINAMATH_CALUDE_concert_songs_count_l719_71942


namespace NUMINAMATH_CALUDE_class_size_l719_71937

/-- Represents the number of students in a class with English and German courses -/
structure ClassEnrollment where
  total : ℕ
  bothSubjects : ℕ
  onlyEnglish : ℕ
  onlyGerman : ℕ
  germanTotal : ℕ

/-- Theorem stating the total number of students in the class -/
theorem class_size (c : ClassEnrollment) 
  (h1 : c.bothSubjects = 12)
  (h2 : c.germanTotal = 22)
  (h3 : c.onlyEnglish = 18)
  (h4 : c.germanTotal = c.bothSubjects + c.onlyGerman)
  (h5 : c.total = c.onlyEnglish + c.onlyGerman + c.bothSubjects) :
  c.total = 40 := by
  sorry


end NUMINAMATH_CALUDE_class_size_l719_71937


namespace NUMINAMATH_CALUDE_conjugate_complex_abs_l719_71908

theorem conjugate_complex_abs (α β : ℂ) : 
  (∃ (x y : ℝ), α = x + y * I ∧ β = x - y * I) →  -- α and β are conjugates
  (∃ (r : ℝ), α / β^2 = r) →                     -- α/β² is real
  Complex.abs (α - β) = 4 * Real.sqrt 3 →        -- |α - β| = 4√3
  Complex.abs α = 4 :=                           -- |α| = 4
by sorry

end NUMINAMATH_CALUDE_conjugate_complex_abs_l719_71908


namespace NUMINAMATH_CALUDE_smallest_n_square_and_cube_l719_71902

def is_perfect_square (x : ℕ) : Prop := ∃ y : ℕ, x = y^2

def is_perfect_cube (x : ℕ) : Prop := ∃ y : ℕ, x = y^3

theorem smallest_n_square_and_cube : 
  (∀ n : ℕ, n > 0 ∧ is_perfect_square (5*n) ∧ is_perfect_cube (4*n) → n ≥ 80) ∧
  (is_perfect_square (5*80) ∧ is_perfect_cube (4*80)) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_square_and_cube_l719_71902


namespace NUMINAMATH_CALUDE_triangular_fence_perimeter_l719_71905

/-- Calculates the perimeter of a triangular fence with evenly spaced posts -/
theorem triangular_fence_perimeter
  (num_posts : ℕ)
  (post_width : ℝ)
  (post_spacing : ℝ)
  (h_num_posts : num_posts = 18)
  (h_post_width : post_width = 0.5)
  (h_post_spacing : post_spacing = 4)
  (h_divisible : num_posts % 3 = 0) :
  let posts_per_side := num_posts / 3
  let side_length := (posts_per_side - 1) * post_spacing + posts_per_side * post_width
  3 * side_length = 69 := by sorry

end NUMINAMATH_CALUDE_triangular_fence_perimeter_l719_71905


namespace NUMINAMATH_CALUDE_sqrt_four_cubed_sum_l719_71941

theorem sqrt_four_cubed_sum : Real.sqrt (4^3 + 4^3 + 4^3 + 4^3) = 16 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_four_cubed_sum_l719_71941


namespace NUMINAMATH_CALUDE_savings_to_earnings_ratio_l719_71921

/-- Proves that the ratio of monthly savings to monthly earnings is 1/2 -/
theorem savings_to_earnings_ratio
  (monthly_earnings : ℝ)
  (vehicle_cost : ℝ)
  (saving_months : ℝ)
  (h1 : monthly_earnings = 4000)
  (h2 : vehicle_cost = 16000)
  (h3 : saving_months = 8) :
  vehicle_cost / saving_months / monthly_earnings = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_savings_to_earnings_ratio_l719_71921
