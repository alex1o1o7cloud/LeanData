import Mathlib

namespace NUMINAMATH_CALUDE_min_sum_squares_roots_l1386_138681

/-- For a quadratic equation x^2 + (2-p)x - p - 3 = 0, 
    the sum of the squares of its roots is minimized when p = 1 -/
theorem min_sum_squares_roots (p : ℝ) : 
  let f : ℝ → ℝ := λ p => p^2 - 2*p + 10
  ∀ q : ℝ, f p ≥ f 1 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_roots_l1386_138681


namespace NUMINAMATH_CALUDE_smallest_two_digit_prime_with_odd_composite_reverse_l1386_138675

/-- A function that reverses the digits of a two-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/-- A predicate that checks if a number is a two-digit number -/
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

/-- The main theorem statement -/
theorem smallest_two_digit_prime_with_odd_composite_reverse :
  ∃ (n : ℕ), is_two_digit n ∧ Nat.Prime n ∧
  Odd (reverse_digits n) ∧ ¬(Nat.Prime (reverse_digits n)) ∧
  (∀ m : ℕ, is_two_digit m → Nat.Prime m →
    Odd (reverse_digits m) → ¬(Nat.Prime (reverse_digits m)) → n ≤ m) ∧
  n = 19 :=
sorry

end NUMINAMATH_CALUDE_smallest_two_digit_prime_with_odd_composite_reverse_l1386_138675


namespace NUMINAMATH_CALUDE_dress_design_combinations_l1386_138655

theorem dress_design_combinations (num_colors num_patterns : ℕ) 
  (h_colors : num_colors = 5)
  (h_patterns : num_patterns = 6) :
  num_colors * num_patterns = 30 := by
sorry

end NUMINAMATH_CALUDE_dress_design_combinations_l1386_138655


namespace NUMINAMATH_CALUDE_cosine_function_properties_l1386_138660

/-- Given function f(x) = cos(ωx + φ) -/
noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x + φ)

theorem cosine_function_properties (ω φ : ℝ) 
  (h_ω_pos : ω > 0)
  (h_φ_range : 0 < φ ∧ φ < π / 2)
  (h_period : ∀ x, f ω φ (x + π) = f ω φ x)
  (h_value : f ω φ (π / 3) = -Real.sqrt 3 / 2) :
  (ω = 2 ∧ φ = π / 6) ∧
  (∀ x, f ω φ x > 1 / 2 ↔ ∃ k : ℤ, k * π - π / 4 < x ∧ x < k * π + π / 12) :=
by sorry

end NUMINAMATH_CALUDE_cosine_function_properties_l1386_138660


namespace NUMINAMATH_CALUDE_min_cubes_for_majority_interior_min_total_cubes_l1386_138628

/-- A function that calculates the number of interior cubes in a cube of side length n -/
def interior_cubes (n : ℕ) : ℕ := (n - 2)^3

/-- A function that calculates the total number of unit cubes in a cube of side length n -/
def total_cubes (n : ℕ) : ℕ := n^3

/-- The minimum side length of a cube where more than half of the cubes are interior -/
def min_side_length : ℕ := 10

theorem min_cubes_for_majority_interior :
  (∀ k < min_side_length, 2 * interior_cubes k ≤ total_cubes k) ∧
  2 * interior_cubes min_side_length > total_cubes min_side_length :=
by sorry

theorem min_total_cubes : total_cubes min_side_length = 1000 :=
by sorry

end NUMINAMATH_CALUDE_min_cubes_for_majority_interior_min_total_cubes_l1386_138628


namespace NUMINAMATH_CALUDE_m_minus_n_equals_six_l1386_138662

theorem m_minus_n_equals_six (m n : ℤ) 
  (h1 : |m| = 2)
  (h2 : |n| = 4)
  (h3 : m > 0)
  (h4 : n < 0) :
  m - n = 6 := by
  sorry

end NUMINAMATH_CALUDE_m_minus_n_equals_six_l1386_138662


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1386_138647

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, ax^2 + bx + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) → a + b = -14 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1386_138647


namespace NUMINAMATH_CALUDE_complex_number_magnitude_l1386_138621

theorem complex_number_magnitude (z : ℂ) (h : z * Complex.I = 1) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_magnitude_l1386_138621


namespace NUMINAMATH_CALUDE_hyperbola_min_focal_distance_l1386_138677

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, prove that the minimum semi-focal distance c is 4 -/
theorem hyperbola_min_focal_distance (a b c : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y, x^2 / a^2 - y^2 / b^2 = 1) →
  (a^2 + b^2 = c^2) →
  (a * b / c = c / 4 + 1) →
  c ≥ 4 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_min_focal_distance_l1386_138677


namespace NUMINAMATH_CALUDE_fraction_invariance_l1386_138614

theorem fraction_invariance (x y : ℝ) (h : x ≠ y) : x / (x - y) = (2 * x) / (2 * x - 2 * y) := by
  sorry

end NUMINAMATH_CALUDE_fraction_invariance_l1386_138614


namespace NUMINAMATH_CALUDE_sqrt_equality_implies_square_l1386_138654

theorem sqrt_equality_implies_square (x : ℝ) : 
  Real.sqrt (3 * x + 5) = 5 → (3 * x + 5)^2 = 625 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equality_implies_square_l1386_138654


namespace NUMINAMATH_CALUDE_inequality_proof_l1386_138698

theorem inequality_proof (a b c : ℝ) 
  (h1 : 4 * a * c - b^2 ≥ 0) 
  (h2 : a > 0) : 
  a + c - Real.sqrt ((a - c)^2 + b^2) ≤ (4 * a * c - b^2) / (2 * a) ∧ 
  (a + c - Real.sqrt ((a - c)^2 + b^2) = (4 * a * c - b^2) / (2 * a) ↔ 
    (b = 0 ∧ a ≥ c) ∨ 4 * a * c = b^2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1386_138698


namespace NUMINAMATH_CALUDE_ellipse_focal_length_l1386_138629

/-- An ellipse with equation y^2/2 + x^2 = 1 -/
def Ellipse := {p : ℝ × ℝ | p.2^2 / 2 + p.1^2 = 1}

/-- The focal length of an ellipse -/
def focalLength (E : Set (ℝ × ℝ)) : ℝ := sorry

/-- Theorem: The focal length of the ellipse y^2/2 + x^2 = 1 is 2 -/
theorem ellipse_focal_length : focalLength Ellipse = 2 := by sorry

end NUMINAMATH_CALUDE_ellipse_focal_length_l1386_138629


namespace NUMINAMATH_CALUDE_gear_r_rpm_calculation_l1386_138676

/-- The number of revolutions per minute for Gear L -/
def gear_l_rpm : ℚ := 20

/-- The time elapsed in seconds -/
def elapsed_time : ℚ := 6

/-- The additional revolutions made by Gear R compared to Gear L -/
def additional_revolutions : ℚ := 6

/-- Calculate the number of revolutions per minute for Gear R -/
def gear_r_rpm : ℚ :=
  (gear_l_rpm * elapsed_time / 60 + additional_revolutions) * 60 / elapsed_time

theorem gear_r_rpm_calculation :
  gear_r_rpm = 80 := by sorry

end NUMINAMATH_CALUDE_gear_r_rpm_calculation_l1386_138676


namespace NUMINAMATH_CALUDE_two_satisfying_functions_l1386_138623

/-- A function satisfying the given property -/
def SatisfiesProperty (f : ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ, f (x^2 - y * f z) = x * f x - z * f y

/-- The set of functions satisfying the property -/
def SatisfyingFunctions : Set (ℝ → ℝ) :=
  {f | SatisfiesProperty f}

/-- The constant zero function -/
def ZeroFunction : ℝ → ℝ := λ _ ↦ 0

/-- The identity function -/
def IdentityFunction : ℝ → ℝ := λ x ↦ x

theorem two_satisfying_functions :
  SatisfyingFunctions = {ZeroFunction, IdentityFunction} := by
  sorry

#check two_satisfying_functions

end NUMINAMATH_CALUDE_two_satisfying_functions_l1386_138623


namespace NUMINAMATH_CALUDE_students_in_diligence_before_transfer_l1386_138688

theorem students_in_diligence_before_transfer 
  (total_students : ℕ) 
  (transferred_students : ℕ) 
  (h1 : total_students = 50)
  (h2 : transferred_students = 2)
  (h3 : ∃ (x : ℕ), x + transferred_students = total_students - x) :
  ∃ (initial_diligence : ℕ), initial_diligence = (total_students / 2) - transferred_students :=
sorry

end NUMINAMATH_CALUDE_students_in_diligence_before_transfer_l1386_138688


namespace NUMINAMATH_CALUDE_bob_position_2023_l1386_138613

-- Define the movement pattern
def spiral_move (n : ℕ) : ℤ × ℤ := sorry

-- Define Bob's position after n steps
def bob_position (n : ℕ) : ℤ × ℤ := sorry

-- Theorem statement
theorem bob_position_2023 :
  bob_position 2023 = (0, 43) := sorry

end NUMINAMATH_CALUDE_bob_position_2023_l1386_138613


namespace NUMINAMATH_CALUDE_stratified_sampling_participation_l1386_138650

/-- Given a school with 1000 students, 300 of which are in the third year,
    prove that when 20 students are selected using stratified sampling,
    14 first and second-year students participate in the activity. -/
theorem stratified_sampling_participation
  (total_students : ℕ) (third_year_students : ℕ) (selected_students : ℕ)
  (h_total : total_students = 1000)
  (h_third_year : third_year_students = 300)
  (h_selected : selected_students = 20) :
  (selected_students : ℚ) * (total_students - third_year_students : ℚ) / total_students = 14 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_participation_l1386_138650


namespace NUMINAMATH_CALUDE_triangle_properties_l1386_138649

/-- Proves the properties of an acute triangle ABC with given conditions -/
theorem triangle_properties (A B C : Real) (a b c : Real) :
  0 < A ∧ A < π / 2 →  -- A is acute
  0 < B ∧ B < π / 2 →  -- B is acute
  0 < C ∧ C < π / 2 →  -- C is acute
  a * Real.sin B = Real.sqrt 3 * b * Real.cos A →
  a = Real.sqrt 21 →
  b = 5 →
  A + B + C = π →  -- Sum of angles in a triangle
  a * Real.sin B = b * Real.sin A →  -- Sine law
  a^2 = b^2 + c^2 - 2 * b * c * Real.cos A →  -- Cosine law
  A = π / 3 ∧ c = 4 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l1386_138649


namespace NUMINAMATH_CALUDE_bounded_expression_l1386_138657

theorem bounded_expression (x y : ℝ) :
  -1/2 ≤ ((x + y) * (1 - x * y)) / ((1 + x^2) * (1 + y^2)) ∧
  ((x + y) * (1 - x * y)) / ((1 + x^2) * (1 + y^2)) ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_bounded_expression_l1386_138657


namespace NUMINAMATH_CALUDE_nn_plus_one_prime_l1386_138684

theorem nn_plus_one_prime (n : ℕ) : n ∈ Finset.range 16 \ {0} →
  Nat.Prime (n^n + 1) ↔ n = 1 ∨ n = 2 ∨ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_nn_plus_one_prime_l1386_138684


namespace NUMINAMATH_CALUDE_remove_number_for_target_average_l1386_138636

def original_list : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

def removed_number : ℕ := 5

def target_average : ℚ := 61/10

theorem remove_number_for_target_average :
  let remaining_list := original_list.filter (· ≠ removed_number)
  (remaining_list.sum : ℚ) / remaining_list.length = target_average := by
  sorry

end NUMINAMATH_CALUDE_remove_number_for_target_average_l1386_138636


namespace NUMINAMATH_CALUDE_sin_15_cos_15_l1386_138694

theorem sin_15_cos_15 : Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_cos_15_l1386_138694


namespace NUMINAMATH_CALUDE_concave_number_probability_l1386_138699

/-- A five-digit natural number formed by digits 0, 1, 2, 3, and 4 -/
def FiveDigitNumber := Fin 5 → Fin 5

/-- Predicate for a "concave number" -/
def IsConcave (n : FiveDigitNumber) : Prop :=
  n 0 > n 1 ∧ n 1 > n 2 ∧ n 2 < n 3 ∧ n 3 < n 4

/-- The set of all possible five-digit numbers -/
def AllNumbers : Finset FiveDigitNumber := sorry

/-- The set of all concave numbers -/
def ConcaveNumbers : Finset FiveDigitNumber := sorry

theorem concave_number_probability :
  (Finset.card ConcaveNumbers : ℚ) / (Finset.card AllNumbers : ℚ) = 23 / 1250 := by sorry

end NUMINAMATH_CALUDE_concave_number_probability_l1386_138699


namespace NUMINAMATH_CALUDE_logarithm_inequality_l1386_138658

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem logarithm_inequality (x : ℝ) (h : 1 < x ∧ x < 10) :
  lg (x^2) > (lg x)^2 ∧ (lg x)^2 > lg (lg x) := by
  sorry

end NUMINAMATH_CALUDE_logarithm_inequality_l1386_138658


namespace NUMINAMATH_CALUDE_expand_expression_l1386_138632

theorem expand_expression (x y : ℝ) : (3 * x + 15) * (4 * y + 12) = 12 * x * y + 36 * x + 60 * y + 180 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1386_138632


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l1386_138625

theorem complex_number_quadrant (i : ℂ) (z : ℂ) :
  i * i = -1 →
  z = 1 - i →
  (0 < z.re ∧ z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l1386_138625


namespace NUMINAMATH_CALUDE_max_value_on_curve_l1386_138674

-- Define the curve C in polar coordinates
def C (ρ θ : ℝ) : Prop := ρ^2 = 36 / (4 * (Real.cos θ)^2 + 9 * (Real.sin θ)^2)

-- Define a point P on the curve C
def P (x y : ℝ) : Prop := ∃ (ρ θ : ℝ), C ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ

-- State the theorem
theorem max_value_on_curve :
  ∀ (x y : ℝ), P x y → (∀ (x' y' : ℝ), P x' y' → 3 * x + 4 * y ≤ 3 * x' + 4 * y') →
  3 * x + 4 * y = Real.sqrt 145 :=
sorry

end NUMINAMATH_CALUDE_max_value_on_curve_l1386_138674


namespace NUMINAMATH_CALUDE_whale_sixth_hour_consumption_l1386_138652

/-- Represents the whale's feeding pattern over 9 hours -/
def WhaleFeedingPattern (x : ℕ) : List ℕ :=
  List.range 9 |>.map (fun i => x + 3 * i)

/-- The total amount of plankton consumed by the whale -/
def TotalConsumption (x : ℕ) : ℕ :=
  (WhaleFeedingPattern x).sum

theorem whale_sixth_hour_consumption :
  ∃ x : ℕ, 
    TotalConsumption x = 450 ∧ 
    (WhaleFeedingPattern x).get! 5 = 53 := by
  sorry

end NUMINAMATH_CALUDE_whale_sixth_hour_consumption_l1386_138652


namespace NUMINAMATH_CALUDE_manuscript_revision_cost_l1386_138604

/-- Given a manuscript typing service with the following conditions:
    - 100 total pages
    - 30 pages revised once
    - 20 pages revised twice
    - $5 per page for initial typing
    - $780 total cost
    Prove that the cost per page for each revision is $4. -/
theorem manuscript_revision_cost :
  let total_pages : ℕ := 100
  let pages_revised_once : ℕ := 30
  let pages_revised_twice : ℕ := 20
  let initial_cost_per_page : ℚ := 5
  let total_cost : ℚ := 780
  let revision_cost_per_page : ℚ := 4
  (total_pages * initial_cost_per_page + 
   (pages_revised_once * revision_cost_per_page + 
    pages_revised_twice * 2 * revision_cost_per_page) = total_cost) :=
by sorry

end NUMINAMATH_CALUDE_manuscript_revision_cost_l1386_138604


namespace NUMINAMATH_CALUDE_mary_shirts_problem_l1386_138679

theorem mary_shirts_problem (blue_shirts : ℕ) (brown_shirts : ℕ) (remaining_shirts : ℕ) :
  blue_shirts = 26 →
  brown_shirts = 36 →
  remaining_shirts = 37 →
  ∃ (f : ℚ), f = 1/2 ∧
    blue_shirts * (1 - f) + brown_shirts * (2/3) = remaining_shirts :=
by sorry

end NUMINAMATH_CALUDE_mary_shirts_problem_l1386_138679


namespace NUMINAMATH_CALUDE_speaking_orders_eq_720_l1386_138606

/-- The number of ways to select 4 students from 7 students (including A and B) to speak,
    where at least one of A and B must participate. -/
def speaking_orders : ℕ :=
  let n : ℕ := 7  -- Total number of students
  let k : ℕ := 4  -- Number of students to be selected
  let special : ℕ := 2  -- Number of special students (A and B)
  let others : ℕ := n - special  -- Number of other students

  -- Case 1: Exactly one of A and B participates
  let case1 : ℕ := special * (Nat.choose others (k - 1)) * (Nat.factorial k)

  -- Case 2: Both A and B participate
  let case2 : ℕ := (Nat.choose others (k - special)) * (Nat.factorial k)

  -- Total number of ways
  case1 + case2

/-- Theorem stating that the number of speaking orders is 720 -/
theorem speaking_orders_eq_720 : speaking_orders = 720 := by
  sorry

end NUMINAMATH_CALUDE_speaking_orders_eq_720_l1386_138606


namespace NUMINAMATH_CALUDE_books_on_shelf_initial_books_count_l1386_138686

/-- The number of books on the shelf before Marta added more -/
def initial_books : ℕ := sorry

/-- The number of books Marta added to the shelf -/
def books_added : ℕ := 10

/-- The total number of books on the shelf after Marta added more -/
def total_books : ℕ := 48

/-- Theorem stating that the initial number of books plus the added books equals the total books -/
theorem books_on_shelf : initial_books + books_added = total_books := by sorry

/-- Theorem proving that the initial number of books is 38 -/
theorem initial_books_count : initial_books = 38 := by sorry

end NUMINAMATH_CALUDE_books_on_shelf_initial_books_count_l1386_138686


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l1386_138609

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n m : ℕ, a (n + m) - a n = m * (a 2 - a 1)

theorem arithmetic_sequence_formula (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 = 2 →
  (∀ n : ℕ, a n < a (n + 1)) →
  (a 2 / a 1 = a 4 / a 2) →
  ∀ n : ℕ, a n = 2 * n :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l1386_138609


namespace NUMINAMATH_CALUDE_circle_centers_distance_l1386_138641

theorem circle_centers_distance (r R : ℝ) (h : r > 0 ∧ R > 0) :
  let d := Real.sqrt (R^2 + r^2 + (10/3) * R * r)
  ∃ (ext_tangent int_tangent : ℝ),
    ext_tangent > 0 ∧ int_tangent > 0 ∧
    ext_tangent = 2 * int_tangent ∧
    d^2 = (R + r)^2 - int_tangent^2 ∧
    d^2 = (R - r)^2 + ext_tangent^2 / 4 :=
by sorry

end NUMINAMATH_CALUDE_circle_centers_distance_l1386_138641


namespace NUMINAMATH_CALUDE_playground_count_l1386_138602

theorem playground_count (a b c d e x : ℕ) (h1 : a = 6) (h2 : b = 12) (h3 : c = 1) (h4 : d = 12) (h5 : e = 7)
  (h_mean : (a + b + c + d + e + x) / 6 = 7) : x = 4 := by
  sorry

end NUMINAMATH_CALUDE_playground_count_l1386_138602


namespace NUMINAMATH_CALUDE_ones_digit_of_largest_power_of_two_dividing_32_factorial_l1386_138624

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def largest_power_of_two_dividing (n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc x => acc + (x + 1).log 2) 0

def ones_digit (n : ℕ) : ℕ := n % 10

theorem ones_digit_of_largest_power_of_two_dividing_32_factorial :
  ones_digit (2^(largest_power_of_two_dividing (factorial 32))) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_of_largest_power_of_two_dividing_32_factorial_l1386_138624


namespace NUMINAMATH_CALUDE_angle_is_90_degrees_l1386_138668

/-- Represents a point on or above the Earth's surface -/
structure EarthPoint where
  latitude : Real
  longitude : Real
  elevation : Real

/-- Calculate the angle between three points on or above the Earth's surface -/
def angleBAC (earthRadius : Real) (a b c : EarthPoint) : Real :=
  sorry

theorem angle_is_90_degrees (earthRadius : Real) :
  let a : EarthPoint := { latitude := 0, longitude := 100, elevation := 0 }
  let b : EarthPoint := { latitude := 30, longitude := -90, elevation := 0 }
  let c : EarthPoint := { latitude := 90, longitude := 0, elevation := 2 }
  angleBAC earthRadius a b c = 90 := by
  sorry

end NUMINAMATH_CALUDE_angle_is_90_degrees_l1386_138668


namespace NUMINAMATH_CALUDE_enclosed_area_is_four_l1386_138638

-- Define the functions for the curve and the line
def f (x : ℝ) := 3 * x^2
def g (x : ℝ) := 3

-- Define the intersection points
def x₁ : ℝ := -1
def x₂ : ℝ := 1

-- State the theorem
theorem enclosed_area_is_four :
  (∫ (x : ℝ) in x₁..x₂, g x - f x) = 4 := by
  sorry

end NUMINAMATH_CALUDE_enclosed_area_is_four_l1386_138638


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l1386_138603

theorem polynomial_evaluation : 
  ∃ x : ℝ, x > 0 ∧ x^2 - 3*x - 10 = 0 ∧ x^4 - 3*x^3 - 4*x^2 + 12*x + 9 = 219 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l1386_138603


namespace NUMINAMATH_CALUDE_abc_sum_l1386_138678

theorem abc_sum (a b c : ℝ) (ha : |a| = 1) (hb : |b| = 2) (hc : |c| = 4) (horder : a > b ∧ b > c) :
  a - b + c = -1 ∨ a - b + c = -3 := by
sorry

end NUMINAMATH_CALUDE_abc_sum_l1386_138678


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l1386_138696

-- Problem 1
theorem problem_1 : (1 * (-12)) - 5 + (-14) - (-39) = 8 := by sorry

-- Problem 2
theorem problem_2 : (1 : ℚ) / 3 + (-3 / 4) + (-1 / 3) + (-1 / 4) + 18 / 19 = -1 / 19 := by sorry

-- Problem 3
theorem problem_3 : (10 + 1 / 3) + (-11.5) + (-(10 + 1 / 3)) - 4.5 = -16 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l1386_138696


namespace NUMINAMATH_CALUDE_product_selection_l1386_138693

/-- Given 12 products with 10 genuine and 2 defective, prove the following:
    (1) The number of ways to select 3 products is 220.
    (2) The number of ways to select exactly 1 defective product out of 3 is 90.
    (3) The number of ways to select at least 1 defective product out of 3 is 100. -/
theorem product_selection (total : Nat) (genuine : Nat) (defective : Nat) (select : Nat)
    (h1 : total = 12)
    (h2 : genuine = 10)
    (h3 : defective = 2)
    (h4 : select = 3)
    (h5 : total = genuine + defective) :
    (Nat.choose total select = 220) ∧
    (Nat.choose defective 1 * Nat.choose genuine 2 = 90) ∧
    (Nat.choose total select - Nat.choose genuine select = 100) :=
  sorry

end NUMINAMATH_CALUDE_product_selection_l1386_138693


namespace NUMINAMATH_CALUDE_product_sum_inequality_l1386_138661

theorem product_sum_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (h_prod : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a*b + a*c + a*d + b*c + b*d + c*d ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_inequality_l1386_138661


namespace NUMINAMATH_CALUDE_marco_coins_l1386_138697

/-- Represents the number of coins of each denomination -/
structure CoinCounts where
  five_cent : ℕ
  ten_cent : ℕ
  twenty_cent : ℕ

/-- Calculates the number of different values obtainable from a given set of coins -/
def different_values (coins : CoinCounts) : ℕ :=
  59 - 3 * coins.five_cent - 2 * coins.ten_cent

theorem marco_coins :
  ∀ (coins : CoinCounts),
    coins.five_cent + coins.ten_cent + coins.twenty_cent = 15 →
    different_values coins = 28 →
    coins.twenty_cent = 4 := by
  sorry

end NUMINAMATH_CALUDE_marco_coins_l1386_138697


namespace NUMINAMATH_CALUDE_hyperbola_real_axis_x_implies_mn_negative_mn_negative_not_sufficient_for_real_axis_x_mn_negative_necessary_not_sufficient_l1386_138616

/-- Represents a hyperbola equation of the form x²/m + y²/n = 1 -/
structure Hyperbola (m n : ℝ) where
  equation : ∀ (x y : ℝ), x^2 / m + y^2 / n = 1

/-- Predicate to check if a hyperbola has its real axis on the x-axis -/
def has_real_axis_on_x (h : Hyperbola m n) : Prop :=
  m > 0 ∧ n < 0

theorem hyperbola_real_axis_x_implies_mn_negative 
  (m n : ℝ) (h : Hyperbola m n) :
  has_real_axis_on_x h → m * n < 0 := by
  sorry

theorem mn_negative_not_sufficient_for_real_axis_x :
  ∃ (m n : ℝ), m * n < 0 ∧ 
  ∃ (h : Hyperbola m n), ¬(has_real_axis_on_x h) := by
  sorry

/-- The main theorem stating that m * n < 0 is a necessary but not sufficient condition -/
theorem mn_negative_necessary_not_sufficient (m n : ℝ) (h : Hyperbola m n) :
  (has_real_axis_on_x h → m * n < 0) ∧
  ¬(m * n < 0 → has_real_axis_on_x h) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_real_axis_x_implies_mn_negative_mn_negative_not_sufficient_for_real_axis_x_mn_negative_necessary_not_sufficient_l1386_138616


namespace NUMINAMATH_CALUDE_players_who_quit_correct_players_who_quit_l1386_138620

theorem players_who_quit (initial_players : ℕ) (lives_per_player : ℕ) (total_lives : ℕ) : ℕ :=
  initial_players - (total_lives / lives_per_player)

theorem correct_players_who_quit :
  players_who_quit 8 3 15 = 3 := by
  sorry

end NUMINAMATH_CALUDE_players_who_quit_correct_players_who_quit_l1386_138620


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l1386_138631

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 + x + 1 ≥ 0) ↔ (∃ x : ℝ, x^2 + x + 1 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l1386_138631


namespace NUMINAMATH_CALUDE_subset_implies_b_equals_two_l1386_138671

theorem subset_implies_b_equals_two :
  (∀ x y : ℝ, x + y - 2 = 0 ∧ x - 2*y + 4 = 0 → y = 3*x + b) →
  b = 2 :=
by sorry

end NUMINAMATH_CALUDE_subset_implies_b_equals_two_l1386_138671


namespace NUMINAMATH_CALUDE_profit_percent_for_2_3_ratio_l1386_138639

/-- Given a cost price to selling price ratio of 2:3, the profit percent is 50%. -/
theorem profit_percent_for_2_3_ratio :
  ∀ (cp sp : ℝ), cp > 0 → sp > 0 →
  cp / sp = 2 / 3 →
  ((sp - cp) / cp) * 100 = 50 := by
sorry

end NUMINAMATH_CALUDE_profit_percent_for_2_3_ratio_l1386_138639


namespace NUMINAMATH_CALUDE_first_digit_of_1122001_base_3_in_base_9_l1386_138610

def base_3_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

def first_digit_base_9 (n : Nat) : Nat :=
  if n = 0 then 0
  else
    let log9 := Nat.log 9 n
    n / (9 ^ log9)

theorem first_digit_of_1122001_base_3_in_base_9 :
  let x := base_3_to_10 [1, 0, 0, 2, 2, 1, 1]
  first_digit_base_9 x = 1 := by
  sorry

end NUMINAMATH_CALUDE_first_digit_of_1122001_base_3_in_base_9_l1386_138610


namespace NUMINAMATH_CALUDE_test_problem_value_l1386_138673

theorem test_problem_value (total_points total_problems four_point_problems : ℕ)
  (h1 : total_points = 100)
  (h2 : total_problems = 30)
  (h3 : four_point_problems = 10)
  (h4 : four_point_problems < total_problems) :
  (total_points - 4 * four_point_problems) / (total_problems - four_point_problems) = 3 :=
by sorry

end NUMINAMATH_CALUDE_test_problem_value_l1386_138673


namespace NUMINAMATH_CALUDE_max_garden_area_l1386_138682

/-- Represents a rectangular garden with one side bounded by a house. -/
structure Garden where
  width : ℝ
  length : ℝ

/-- The total fencing available -/
def total_fencing : ℝ := 500

/-- Calculates the area of the garden -/
def garden_area (g : Garden) : ℝ := g.width * g.length

/-- Calculates the amount of fencing used for three sides of the garden -/
def fencing_used (g : Garden) : ℝ := g.length + 2 * g.width

/-- Theorem stating the maximum area of the garden -/
theorem max_garden_area :
  ∃ (g : Garden), fencing_used g = total_fencing ∧
    ∀ (h : Garden), fencing_used h = total_fencing → garden_area h ≤ garden_area g ∧
    garden_area g = 31250 := by
  sorry

end NUMINAMATH_CALUDE_max_garden_area_l1386_138682


namespace NUMINAMATH_CALUDE_circular_binary_arrangement_l1386_138646

/-- A type representing a binary number using only 1 and 2 -/
def BinaryNumber (n : ℕ) := Fin n → Fin 2

/-- A function to check if two binary numbers differ by exactly one digit -/
def differByOneDigit (n : ℕ) (a b : BinaryNumber n) : Prop :=
  ∃! i : Fin n, a i ≠ b i

/-- A type representing an arrangement of binary numbers in a circle -/
def CircularArrangement (n : ℕ) := Fin (2^n) → BinaryNumber n

/-- The main theorem statement -/
theorem circular_binary_arrangement (n : ℕ) :
  ∃ (arrangement : CircularArrangement n),
    (∀ i j : Fin (2^n), i ≠ j → arrangement i ≠ arrangement j) ∧
    (∀ i : Fin (2^n), differByOneDigit n (arrangement i) (arrangement (i + 1))) :=
sorry

end NUMINAMATH_CALUDE_circular_binary_arrangement_l1386_138646


namespace NUMINAMATH_CALUDE_xiaoli_estimation_l1386_138645

theorem xiaoli_estimation (x y : ℝ) (h : x > y ∧ y > 0) :
  (1.1 * x - (y - 2) = x - y + 0.1 * x + 2) ∧
  (1.1 * x * (y - 2) = 1.1 * x * y - 2.2 * x) := by
  sorry

end NUMINAMATH_CALUDE_xiaoli_estimation_l1386_138645


namespace NUMINAMATH_CALUDE_circle_C_equation_line_l_equation_l1386_138642

-- Define the circle C
def circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 1)^2 ≤ 2}

-- Define the tangent line
def tangent_line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 = 4}

-- Define the point that line l passes through
def point_on_l : ℝ × ℝ := (2, 3)

-- Define the length of the chord intercepted by circle C on line l
def chord_length : ℝ := 2

-- Theorem stating the standard equation of circle C
theorem circle_C_equation :
  ∀ p : ℝ × ℝ, p ∈ circle_C ↔ (p.1 - 1)^2 + (p.2 - 1)^2 = 2 :=
sorry

-- Theorem stating the equation of line l
theorem line_l_equation :
  ∀ p : ℝ × ℝ, (p ∈ circle_C ∧ (∃ q : ℝ × ℝ, q ∈ circle_C ∧ q ≠ p ∧
    ((p.1 - q.1)^2 + (p.2 - q.2)^2 = chord_length^2) ∧
    (p.1 - point_on_l.1) * (q.2 - point_on_l.2) = (q.1 - point_on_l.1) * (p.2 - point_on_l.2)))
  → (3 * p.1 - 4 * p.2 + 6 = 0 ∨ p.1 = 2) :=
sorry

end NUMINAMATH_CALUDE_circle_C_equation_line_l_equation_l1386_138642


namespace NUMINAMATH_CALUDE_intersection_angle_implies_ratio_l1386_138605

-- Define the ellipse and hyperbola
def is_on_ellipse (x y a₁ b₁ : ℝ) : Prop := x^2 / a₁^2 + y^2 / b₁^2 = 1
def is_on_hyperbola (x y a₂ b₂ : ℝ) : Prop := x^2 / a₂^2 - y^2 / b₂^2 = 1

-- Define the common foci
def are_common_foci (F₁ F₂ : ℝ × ℝ) (a₁ b₁ a₂ b₂ : ℝ) : Prop := 
  ∃ c : ℝ, c^2 = a₁^2 - b₁^2 ∧ c^2 = a₂^2 + b₂^2 ∧
  F₁ = (c, 0) ∧ F₂ = (-c, 0)

-- Define the angle between foci
def angle_F₁PF₂ (P F₁ F₂ : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem intersection_angle_implies_ratio 
  (P F₁ F₂ : ℝ × ℝ) (a₁ b₁ a₂ b₂ : ℝ) :
  a₁ > b₁ ∧ b₁ > 0 ∧ a₂ > 0 ∧ b₂ > 0 →
  is_on_ellipse P.1 P.2 a₁ b₁ →
  is_on_hyperbola P.1 P.2 a₂ b₂ →
  are_common_foci F₁ F₂ a₁ b₁ a₂ b₂ →
  angle_F₁PF₂ P F₁ F₂ = π / 3 →
  b₁ / b₂ = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_angle_implies_ratio_l1386_138605


namespace NUMINAMATH_CALUDE_correct_regression_sequence_l1386_138611

-- Define the steps of linear regression analysis
inductive RegressionStep
  | InterpretEquation
  | CollectData
  | CalculateEquation
  | DrawScatterPlot

-- Define a sequence of steps
def StepSequence := List RegressionStep

-- Define the correct sequence
def correctSequence : StepSequence :=
  [RegressionStep.CollectData,
   RegressionStep.DrawScatterPlot,
   RegressionStep.CalculateEquation,
   RegressionStep.InterpretEquation]

-- Define a proposition that x and y are linearly related
def linearlyRelated (x y : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ t : ℝ, y t = a * x t + b

-- Theorem stating that given linear relationship, the correct sequence is as defined
theorem correct_regression_sequence (x y : ℝ → ℝ) :
  linearlyRelated x y →
  (∀ seq : StepSequence,
    seq = correctSequence ↔
    seq = [RegressionStep.CollectData,
           RegressionStep.DrawScatterPlot,
           RegressionStep.CalculateEquation,
           RegressionStep.InterpretEquation]) :=
by sorry


end NUMINAMATH_CALUDE_correct_regression_sequence_l1386_138611


namespace NUMINAMATH_CALUDE_eggs_per_basket_l1386_138637

theorem eggs_per_basket (red_eggs blue_eggs min_eggs : ℕ) 
  (h1 : red_eggs = 30)
  (h2 : blue_eggs = 42)
  (h3 : min_eggs = 5) :
  ∃ (n : ℕ), n ≥ min_eggs ∧ 
             n ∣ red_eggs ∧ 
             n ∣ blue_eggs ∧
             ∀ (m : ℕ), m ≥ min_eggs ∧ m ∣ red_eggs ∧ m ∣ blue_eggs → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_eggs_per_basket_l1386_138637


namespace NUMINAMATH_CALUDE_rectangle_area_l1386_138683

theorem rectangle_area (w : ℝ) (h1 : w > 0) : 
  let l := 2 * w
  let perimeter := 2 * l + 2 * w
  perimeter = 4 → w * l = 8 / 9 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l1386_138683


namespace NUMINAMATH_CALUDE_record_storage_cost_l1386_138607

def box_length : ℝ := 15
def box_width : ℝ := 12
def box_height : ℝ := 10
def total_volume : ℝ := 1080000
def cost_per_box : ℝ := 0.8

theorem record_storage_cost : 
  let box_volume := box_length * box_width * box_height
  let num_boxes := total_volume / box_volume
  num_boxes * cost_per_box = 480 := by sorry

end NUMINAMATH_CALUDE_record_storage_cost_l1386_138607


namespace NUMINAMATH_CALUDE_largest_intersection_point_l1386_138630

/-- Polynomial P(x) -/
def P (a : ℝ) (x : ℝ) : ℝ := x^6 - 13*x^5 + 42*x^4 - 30*x^3 + a*x^2

/-- Line L(x) -/
def L (c : ℝ) (x : ℝ) : ℝ := 3*x + c

/-- The set of intersection points between P and L -/
def intersectionPoints (a c : ℝ) : Set ℝ := {x : ℝ | P a x = L c x}

theorem largest_intersection_point (a c : ℝ) :
  (∃ p q r : ℝ, intersectionPoints a c = {p, q, r} ∧ p < q ∧ q < r) →
  (∀ x : ℝ, x ∉ intersectionPoints a c → P a x < L c x) →
  (∃ x ∈ intersectionPoints a c, ∀ y ∈ intersectionPoints a c, y ≤ x) →
  (∃ x ∈ intersectionPoints a c, x = 4) :=
sorry

end NUMINAMATH_CALUDE_largest_intersection_point_l1386_138630


namespace NUMINAMATH_CALUDE_father_son_age_problem_l1386_138667

theorem father_son_age_problem (x : ℕ) : x = 4 :=
by
  -- Son's current age
  let son_age : ℕ := 8
  -- Father's current age
  let father_age : ℕ := 4 * son_age
  -- In x years, father's age will be 3 times son's age
  have h : father_age + x = 3 * (son_age + x) := by sorry
  sorry

end NUMINAMATH_CALUDE_father_son_age_problem_l1386_138667


namespace NUMINAMATH_CALUDE_least_five_digit_congruent_to_8_mod_17_l1386_138651

theorem least_five_digit_congruent_to_8_mod_17 : 
  ∀ n : ℕ, n ≥ 10000 ∧ n ≡ 8 [MOD 17] → n ≥ 10004 :=
by sorry

end NUMINAMATH_CALUDE_least_five_digit_congruent_to_8_mod_17_l1386_138651


namespace NUMINAMATH_CALUDE_eliminate_uvw_l1386_138670

theorem eliminate_uvw (a b c d u v w : ℝ) 
  (eq1 : a = Real.cos u + Real.cos v + Real.cos w)
  (eq2 : b = Real.sin u + Real.sin v + Real.sin w)
  (eq3 : c = Real.cos (2*u) + Real.cos (2*v) + Real.cos (2*w))
  (eq4 : d = Real.sin (2*u) + Real.sin (2*v) + Real.sin (2*w)) :
  (a^2 - b^2 - c)^2 + (2*a*b - d)^2 = 4*(a^2 + b^2) := by
sorry

end NUMINAMATH_CALUDE_eliminate_uvw_l1386_138670


namespace NUMINAMATH_CALUDE_bianca_cupcakes_l1386_138627

/-- The number of cupcakes Bianca initially made -/
def initial_cupcakes : ℕ := 14

/-- The number of cupcakes Bianca sold -/
def sold_cupcakes : ℕ := 6

/-- The number of additional cupcakes Bianca made -/
def additional_cupcakes : ℕ := 17

/-- The final number of cupcakes Bianca had -/
def final_cupcakes : ℕ := 25

theorem bianca_cupcakes : 
  initial_cupcakes - sold_cupcakes + additional_cupcakes = final_cupcakes :=
by sorry

end NUMINAMATH_CALUDE_bianca_cupcakes_l1386_138627


namespace NUMINAMATH_CALUDE_factorial_four_div_one_l1386_138656

/-- Definition of factorial for natural numbers -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Theorem stating that 4! / (4 - 3)! = 24 -/
theorem factorial_four_div_one : factorial 4 / factorial (4 - 3) = 24 := by
  sorry

end NUMINAMATH_CALUDE_factorial_four_div_one_l1386_138656


namespace NUMINAMATH_CALUDE_nathans_score_l1386_138672

theorem nathans_score (total_students : ℕ) (students_without_nathan : ℕ) 
  (avg_without_nathan : ℚ) (avg_with_nathan : ℚ) :
  total_students = 18 →
  students_without_nathan = 17 →
  avg_without_nathan = 84 →
  avg_with_nathan = 87 →
  (total_students * avg_with_nathan - students_without_nathan * avg_without_nathan : ℚ) = 138 :=
by sorry

end NUMINAMATH_CALUDE_nathans_score_l1386_138672


namespace NUMINAMATH_CALUDE_kelly_snacks_weight_l1386_138615

/-- The weight of peanuts Kelly bought in pounds -/
def peanuts_weight : ℝ := 0.1

/-- The weight of raisins Kelly bought in pounds -/
def raisins_weight : ℝ := 0.4

/-- The weight of almonds Kelly bought in pounds -/
def almonds_weight : ℝ := 0.3

/-- The total weight of snacks Kelly bought -/
def total_weight : ℝ := peanuts_weight + raisins_weight + almonds_weight

theorem kelly_snacks_weight : total_weight = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_kelly_snacks_weight_l1386_138615


namespace NUMINAMATH_CALUDE_inequality_proof_l1386_138608

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt (a^2 - a*b + b^2) + Real.sqrt (b^2 - b*c + c^2) ≥ Real.sqrt (a^2 + a*c + c^2) ∧
  (Real.sqrt (a^2 - a*b + b^2) + Real.sqrt (b^2 - b*c + c^2) = Real.sqrt (a^2 + a*c + c^2) ↔ 1/b = 1/a + 1/c) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1386_138608


namespace NUMINAMATH_CALUDE_hidden_face_sum_l1386_138695

/-- Represents a standard six-sided die -/
def Die := Fin 6

/-- The sum of all numbers on a standard die -/
def dieTotalSum : ℕ := 21

/-- The visible numbers on the stacked dice -/
def visibleNumbers : List ℕ := [2, 2, 3, 3, 4, 5, 6]

/-- The number of dice stacked -/
def numberOfDice : ℕ := 3

theorem hidden_face_sum :
  (numberOfDice * dieTotalSum) - visibleNumbers.sum = 38 := by
  sorry

end NUMINAMATH_CALUDE_hidden_face_sum_l1386_138695


namespace NUMINAMATH_CALUDE_gasohol_mixture_proof_l1386_138618

/-- Proves that the initial percentage of gasoline in the gasohol mixture is 95% --/
theorem gasohol_mixture_proof (initial_volume : ℝ) (initial_ethanol_percent : ℝ) 
  (desired_ethanol_percent : ℝ) (added_ethanol : ℝ) :
  initial_volume = 45 →
  initial_ethanol_percent = 5 →
  desired_ethanol_percent = 10 →
  added_ethanol = 2.5 →
  (100 - initial_ethanol_percent) = 95 :=
by
  sorry

#check gasohol_mixture_proof

end NUMINAMATH_CALUDE_gasohol_mixture_proof_l1386_138618


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l1386_138691

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sixth_term
  (a : ℕ → ℝ)
  (h_geo : is_geometric_sequence a)
  (h_2 : a 2 = 5)
  (h_4 : a 4 = 20) :
  a 6 = 80 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l1386_138691


namespace NUMINAMATH_CALUDE_gcd_count_for_360_l1386_138626

theorem gcd_count_for_360 (a b : ℕ+) : 
  (Nat.gcd a b * Nat.lcm a b = 360) → 
  (∃ (S : Finset ℕ), (∀ d ∈ S, ∃ (x y : ℕ+), Nat.gcd x y = d ∧ Nat.gcd x y * Nat.lcm x y = 360) ∧ 
                      (∀ d : ℕ, (∃ (x y : ℕ+), Nat.gcd x y = d ∧ Nat.gcd x y * Nat.lcm x y = 360) → d ∈ S) ∧
                      S.card = 10) :=
sorry

end NUMINAMATH_CALUDE_gcd_count_for_360_l1386_138626


namespace NUMINAMATH_CALUDE_f_value_at_inverse_f_3_l1386_138612

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 1 - 2 * x^2 else x^2 + 3 * x - 2

theorem f_value_at_inverse_f_3 : f (1 / f 3) = 127 / 128 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_inverse_f_3_l1386_138612


namespace NUMINAMATH_CALUDE_total_nylon_needed_l1386_138666

/-- The amount of nylon needed for a dog collar in inches -/
def dog_collar_nylon : ℕ := 18

/-- The amount of nylon needed for a cat collar in inches -/
def cat_collar_nylon : ℕ := 10

/-- The number of dog collars to be made -/
def num_dog_collars : ℕ := 9

/-- The number of cat collars to be made -/
def num_cat_collars : ℕ := 3

/-- Theorem stating the total amount of nylon needed -/
theorem total_nylon_needed : 
  dog_collar_nylon * num_dog_collars + cat_collar_nylon * num_cat_collars = 192 := by
  sorry

end NUMINAMATH_CALUDE_total_nylon_needed_l1386_138666


namespace NUMINAMATH_CALUDE_platform_length_l1386_138601

/-- Calculates the length of a platform given train specifications -/
theorem platform_length
  (train_length : ℝ)
  (time_cross_platform : ℝ)
  (time_cross_pole : ℝ)
  (h1 : train_length = 700)
  (h2 : time_cross_platform = 45)
  (h3 : time_cross_pole = 15) :
  let train_speed := train_length / time_cross_pole
  let platform_length := train_speed * time_cross_platform - train_length
  platform_length = 1400 := by
sorry

end NUMINAMATH_CALUDE_platform_length_l1386_138601


namespace NUMINAMATH_CALUDE_inequality_system_integer_solutions_l1386_138685

theorem inequality_system_integer_solutions :
  ∀ x : ℤ, (x - 3 * (x - 2) ≤ 4 ∧ (1 + 2 * x) / 3 > x - 1) ↔ (x = 1 ∨ x = 2 ∨ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_integer_solutions_l1386_138685


namespace NUMINAMATH_CALUDE_original_earnings_l1386_138680

theorem original_earnings (new_earnings : ℝ) (percentage_increase : ℝ) 
  (h1 : new_earnings = 84)
  (h2 : percentage_increase = 40) :
  let original_earnings := new_earnings / (1 + percentage_increase / 100)
  original_earnings = 60 := by
sorry

end NUMINAMATH_CALUDE_original_earnings_l1386_138680


namespace NUMINAMATH_CALUDE_equation_solution_l1386_138690

theorem equation_solution : 
  ∃ x : ℝ, (((1 - x) / (x - 4)) + (1 / (4 - x)) = 1) ∧ (x = 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1386_138690


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1386_138648

-- Define the quadratic function
def f (x : ℝ) : ℝ := -3 * x^2 + 9 * x + 12

-- Define the solution set
def solution_set : Set ℝ := {x : ℝ | -1 < x ∧ x < 4}

-- Theorem statement
theorem quadratic_inequality_solution :
  {x : ℝ | f x > 0} = solution_set :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1386_138648


namespace NUMINAMATH_CALUDE_system_solution_l1386_138644

theorem system_solution (x y z : ℝ) : 
  x + y + z = 2 ∧ 
  x^2 + y^2 + z^2 = 6 ∧ 
  x^3 + y^3 + z^3 = 8 ↔ 
  ((x = 1 ∧ y = 2 ∧ z = -1) ∨
   (x = 1 ∧ y = -1 ∧ z = 2) ∨
   (x = 2 ∧ y = 1 ∧ z = -1) ∨
   (x = 2 ∧ y = -1 ∧ z = 1) ∨
   (x = -1 ∧ y = 1 ∧ z = 2) ∨
   (x = -1 ∧ y = 2 ∧ z = 1)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1386_138644


namespace NUMINAMATH_CALUDE_bridge_bricks_l1386_138653

theorem bridge_bricks (type_a : ℕ) (type_b : ℕ) (other_types : ℕ) : 
  type_a ≥ 40 →
  type_b = type_a / 2 →
  type_a + type_b + other_types = 150 →
  other_types = 90 := by
sorry

end NUMINAMATH_CALUDE_bridge_bricks_l1386_138653


namespace NUMINAMATH_CALUDE_amit_work_days_l1386_138633

theorem amit_work_days (ananthu_days : ℝ) (amit_worked : ℝ) (total_days : ℝ) :
  ananthu_days = 90 ∧ amit_worked = 3 ∧ total_days = 75 →
  ∃ x : ℝ, 
    x > 0 ∧
    (3 / x) + ((total_days - amit_worked) / ananthu_days) = 1 ∧
    x = 15 := by
  sorry

end NUMINAMATH_CALUDE_amit_work_days_l1386_138633


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_mod_l1386_138622

theorem arithmetic_sequence_sum_mod (a d : ℕ) (n : ℕ) (h : n > 0) :
  let last_term := a + (n - 1) * d
  let sum := n * (a + last_term) / 2
  sum % 17 = 12 :=
by
  sorry

#check arithmetic_sequence_sum_mod 3 5 21

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_mod_l1386_138622


namespace NUMINAMATH_CALUDE_shaded_area_of_concentric_circles_l1386_138617

theorem shaded_area_of_concentric_circles (R r : ℝ) : 
  R > 0 → r > 0 → r = R / 2 → R^2 * π = 81 * π → 
  (π * R^2) / 2 + (π * r^2) / 2 = 50.625 * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_of_concentric_circles_l1386_138617


namespace NUMINAMATH_CALUDE_brendan_tax_payment_l1386_138689

/-- Calculates the weekly tax payment for a waiter named Brendan --/
def brendan_weekly_tax (hourly_wage : ℝ) (shift_hours : List ℝ) (tip_per_hour : ℝ) (tax_rate : ℝ) (tip_report_ratio : ℝ) : ℝ :=
  let total_hours := shift_hours.sum
  let wage_income := hourly_wage * total_hours
  let total_tips := tip_per_hour * total_hours
  let reported_tips := total_tips * tip_report_ratio
  let reported_income := wage_income + reported_tips
  reported_income * tax_rate

/-- Theorem stating that Brendan's weekly tax payment is $56 --/
theorem brendan_tax_payment :
  brendan_weekly_tax 6 [8, 8, 12] 12 0.2 (1/3) = 56 := by
  sorry

end NUMINAMATH_CALUDE_brendan_tax_payment_l1386_138689


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1386_138669

theorem sufficient_not_necessary_condition (a : ℝ) : 
  (∀ x : ℝ, x > a → x^2 - 5*x + 6 ≥ 0) ∧ 
  (∃ x : ℝ, x^2 - 5*x + 6 ≥ 0 ∧ x ≤ a) ↔ 
  a ≥ 3 := by
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1386_138669


namespace NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l1386_138663

/-- Definition of an ellipse with semi-major axis 4 and semi-minor axis 3 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 16) + (p.2^2 / 9) = 1}

/-- The right focus of the ellipse -/
def F₂ : ℝ × ℝ := sorry

/-- The left focus of the ellipse -/
def F₁ : ℝ × ℝ := sorry

/-- Theorem: The perimeter of triangle AF₁B is 16 for any A and B on the ellipse -/
theorem ellipse_triangle_perimeter 
  (A B : ℝ × ℝ) 
  (hA : A ∈ Ellipse) 
  (hB : B ∈ Ellipse) : 
  dist A F₁ + dist B F₁ + dist A B = 16 := 
sorry

end NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l1386_138663


namespace NUMINAMATH_CALUDE_max_unglazed_windows_l1386_138664

/-- Represents a window or a pane of glass with a specific size. -/
structure Pane :=
  (size : ℕ)

/-- Represents the state of glazing process. -/
structure GlazingState :=
  (windows : List Pane)
  (glasses : List Pane)

/-- Simulates the glazier's process of matching glasses to windows. -/
def glazierProcess (state : GlazingState) : ℕ :=
  sorry

/-- Theorem stating the maximum number of unglazed windows. -/
theorem max_unglazed_windows :
  ∀ (initial_state : GlazingState),
    initial_state.windows.length = 15 ∧
    initial_state.glasses.length = 15 ∧
    (∀ w ∈ initial_state.windows, ∃ g ∈ initial_state.glasses, w.size = g.size) →
    glazierProcess initial_state ≤ 7 :=
  sorry

end NUMINAMATH_CALUDE_max_unglazed_windows_l1386_138664


namespace NUMINAMATH_CALUDE_captain_selection_criterion_l1386_138665

-- Define the universe of players
variable (Player : Type)

-- Define predicates
variable (attends_all_sessions : Player → Prop)
variable (always_on_time : Player → Prop)
variable (considered_for_captain : Player → Prop)

-- Theorem statement
theorem captain_selection_criterion
  (h : ∀ p : Player, (attends_all_sessions p ∧ always_on_time p) → considered_for_captain p) :
  ∀ p : Player, ¬(considered_for_captain p) → (¬(attends_all_sessions p) ∨ ¬(always_on_time p)) :=
by sorry

end NUMINAMATH_CALUDE_captain_selection_criterion_l1386_138665


namespace NUMINAMATH_CALUDE_f_is_quadratic_l1386_138643

-- Define what a quadratic equation is
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- Define the specific function we're checking
def f (x : ℝ) : ℝ := x^2 + 6*x

-- Theorem statement
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l1386_138643


namespace NUMINAMATH_CALUDE_compute_expression_l1386_138659

theorem compute_expression : 6^3 - 4*5 + 2^4 = 212 := by sorry

end NUMINAMATH_CALUDE_compute_expression_l1386_138659


namespace NUMINAMATH_CALUDE_water_current_speed_l1386_138640

/-- Proves that the speed of a water current is 2 km/h given specific swimming conditions -/
theorem water_current_speed 
  (swimmer_speed : ℝ) 
  (distance : ℝ) 
  (time : ℝ) 
  (h1 : swimmer_speed = 4) 
  (h2 : distance = 7) 
  (h3 : time = 3.5) : 
  ∃ (current_speed : ℝ), 
    current_speed = 2 ∧ 
    (swimmer_speed - current_speed) * time = distance :=
by sorry

end NUMINAMATH_CALUDE_water_current_speed_l1386_138640


namespace NUMINAMATH_CALUDE_glorys_favorite_number_l1386_138687

theorem glorys_favorite_number (glory misty : ℕ) : 
  misty = glory / 3 →
  misty + glory = 600 →
  glory = 450 := by sorry

end NUMINAMATH_CALUDE_glorys_favorite_number_l1386_138687


namespace NUMINAMATH_CALUDE_addition_puzzle_l1386_138634

theorem addition_puzzle (E S X : Nat) : 
  E ≠ 0 → S ≠ 0 → X ≠ 0 →
  E ≠ S → E ≠ X → S ≠ X →
  E * 100 + E * 10 + E + E * 100 + E * 10 + E = S * 100 + X * 10 + S →
  X = 7 := by
sorry

end NUMINAMATH_CALUDE_addition_puzzle_l1386_138634


namespace NUMINAMATH_CALUDE_unique_solution_l1386_138619

-- Define the properties of p, q, and r
def is_valid_solution (p q r : ℕ) : Prop :=
  Nat.Prime p ∧ 
  Nat.Prime q ∧ 
  Nat.Prime r ∧
  q - p = r ∧
  5 < p ∧ p < 15 ∧
  q < 15

-- Theorem statement
theorem unique_solution : 
  ∃! q : ℕ, ∃ (p r : ℕ), is_valid_solution p q r ∧ q = 13 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l1386_138619


namespace NUMINAMATH_CALUDE_rectangle_dimension_increase_l1386_138635

theorem rectangle_dimension_increase (L B : ℝ) (L' B' : ℝ) (h1 : B' = 1.3 * B) (h2 : L' * B' = 1.43 * L * B) : L' = 1.1 * L := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimension_increase_l1386_138635


namespace NUMINAMATH_CALUDE_sum_of_cubes_is_twelve_l1386_138600

/-- Given real numbers a, b, and c satisfying certain conditions, 
    prove that the sum of their cubes is 12. -/
theorem sum_of_cubes_is_twelve (a b c : ℝ) 
    (sum_eq_three : a + b + c = 3)
    (sum_of_products_eq_three : a * b + a * c + b * c = 3)
    (product_eq_neg_one : a * b * c = -1) : 
  a^3 + b^3 + c^3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_is_twelve_l1386_138600


namespace NUMINAMATH_CALUDE_f_composition_value_l1386_138692

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 3
  else 2^x

-- State the theorem
theorem f_composition_value : f (f (1/3)) = 1/2 := by sorry

end NUMINAMATH_CALUDE_f_composition_value_l1386_138692
