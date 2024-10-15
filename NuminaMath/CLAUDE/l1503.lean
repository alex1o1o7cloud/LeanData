import Mathlib

namespace NUMINAMATH_CALUDE_symmetric_points_x_axis_l1503_150378

/-- Given two points A and B that are symmetric with respect to the x-axis,
    prove that the y-coordinate of A determines m to be 1. -/
theorem symmetric_points_x_axis (m : ℝ) : 
  let A : ℝ × ℝ := (-3, 2*m - 1)
  let B : ℝ × ℝ := (-3, -1)
  (A.1 = B.1 ∧ A.2 = -B.2) → m = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_x_axis_l1503_150378


namespace NUMINAMATH_CALUDE_fraction_equality_l1503_150370

theorem fraction_equality : (1 : ℚ) / 2 = 4 / 8 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l1503_150370


namespace NUMINAMATH_CALUDE_square_with_quarter_circles_area_l1503_150331

theorem square_with_quarter_circles_area (π : Real) : 
  let square_side : Real := 4
  let quarter_circle_radius : Real := square_side / 2
  let square_area : Real := square_side ^ 2
  let quarter_circle_area : Real := π * quarter_circle_radius ^ 2 / 4
  let total_quarter_circles_area : Real := 4 * quarter_circle_area
  square_area - total_quarter_circles_area = 16 - 4 * π := by sorry

end NUMINAMATH_CALUDE_square_with_quarter_circles_area_l1503_150331


namespace NUMINAMATH_CALUDE_frustum_volume_ratio_l1503_150348

/-- Given a frustum with base area ratio 1:9, prove the volume ratio of parts divided by midsection is 7:19 -/
theorem frustum_volume_ratio (A₁ A₂ V₁ V₂ : ℝ) (h_area_ratio : A₁ / A₂ = 1 / 9) :
  V₁ / V₂ = 7 / 19 := by
  sorry

end NUMINAMATH_CALUDE_frustum_volume_ratio_l1503_150348


namespace NUMINAMATH_CALUDE_one_quarter_of_6_75_l1503_150341

theorem one_quarter_of_6_75 : (6.75 : ℚ) / 4 = 27 / 16 := by sorry

end NUMINAMATH_CALUDE_one_quarter_of_6_75_l1503_150341


namespace NUMINAMATH_CALUDE_phoebe_age_proof_l1503_150332

/-- Phoebe's current age -/
def phoebe_age : ℕ := 10

/-- Raven's current age -/
def raven_age : ℕ := 55

theorem phoebe_age_proof :
  (raven_age + 5 = 4 * (phoebe_age + 5)) → phoebe_age = 10 := by
  sorry

end NUMINAMATH_CALUDE_phoebe_age_proof_l1503_150332


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1503_150367

def A : Set ℝ := {0, 1, 2}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1503_150367


namespace NUMINAMATH_CALUDE_library_book_purchase_l1503_150379

theorem library_book_purchase (initial_books : ℕ) (current_books : ℕ) (last_year_purchase : ℕ) : 
  initial_books = 100 →
  current_books = 300 →
  current_books = initial_books + last_year_purchase + 3 * last_year_purchase →
  last_year_purchase = 50 := by
  sorry

end NUMINAMATH_CALUDE_library_book_purchase_l1503_150379


namespace NUMINAMATH_CALUDE_eriks_mother_money_l1503_150344

/-- The amount of money Erik's mother gave him. -/
def money_from_mother : ℕ := sorry

/-- The number of loaves of bread Erik bought. -/
def bread_loaves : ℕ := 3

/-- The number of cartons of orange juice Erik bought. -/
def juice_cartons : ℕ := 3

/-- The cost of one loaf of bread in dollars. -/
def bread_cost : ℕ := 3

/-- The cost of one carton of orange juice in dollars. -/
def juice_cost : ℕ := 6

/-- The amount of money Erik has left in dollars. -/
def money_left : ℕ := 59

/-- Theorem stating that the amount of money Erik's mother gave him is $86. -/
theorem eriks_mother_money : money_from_mother = 86 := by sorry

end NUMINAMATH_CALUDE_eriks_mother_money_l1503_150344


namespace NUMINAMATH_CALUDE_two_fifths_divided_by_one_fifth_l1503_150382

theorem two_fifths_divided_by_one_fifth : (2 : ℚ) / 5 / ((1 : ℚ) / 5) = 2 := by sorry

end NUMINAMATH_CALUDE_two_fifths_divided_by_one_fifth_l1503_150382


namespace NUMINAMATH_CALUDE_sin_cos_difference_equals_half_l1503_150335

theorem sin_cos_difference_equals_half : 
  Real.sin (43 * π / 180) * Real.cos (13 * π / 180) - 
  Real.sin (13 * π / 180) * Real.cos (43 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_difference_equals_half_l1503_150335


namespace NUMINAMATH_CALUDE_max_quarters_l1503_150329

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- The value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- The total amount Sasha has in dollars -/
def total_amount : ℚ := 4.50

/-- Proves that the maximum number of quarters (and dimes) Sasha can have is 12 -/
theorem max_quarters : 
  ∀ q : ℕ, 
    (q : ℚ) * (quarter_value + dime_value) ≤ total_amount → 
    q ≤ 12 := by
  sorry

#check max_quarters

end NUMINAMATH_CALUDE_max_quarters_l1503_150329


namespace NUMINAMATH_CALUDE_angle_sum_in_circle_l1503_150345

theorem angle_sum_in_circle (x : ℝ) : 
  (6 * x + 3 * x + 2 * x + x = 360) → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_in_circle_l1503_150345


namespace NUMINAMATH_CALUDE_hiker_route_length_l1503_150396

theorem hiker_route_length (rate_up : ℝ) (days_up : ℝ) (rate_down_factor : ℝ) : 
  rate_up = 7 →
  days_up = 2 →
  rate_down_factor = 1.5 →
  (rate_up * days_up) * rate_down_factor = 21 := by
  sorry

end NUMINAMATH_CALUDE_hiker_route_length_l1503_150396


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1503_150303

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {1, 2, 3, 4}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1503_150303


namespace NUMINAMATH_CALUDE_min_trapezium_perimeter_l1503_150361

/-- A right-angled isosceles triangle with hypotenuse √2 cm -/
structure RightIsoscelesTriangle where
  hypotenuse : ℝ
  hypotenuse_eq : hypotenuse = Real.sqrt 2

/-- A trapezium formed by assembling right-angled isosceles triangles -/
structure Trapezium where
  triangles : List RightIsoscelesTriangle
  is_trapezium : Bool  -- This should be a predicate ensuring the shape is a trapezium

/-- The perimeter of a trapezium -/
def trapezium_perimeter (t : Trapezium) : ℝ := sorry

/-- Theorem stating the minimum perimeter of a trapezium formed by right-angled isosceles triangles -/
theorem min_trapezium_perimeter :
  ∀ t : Trapezium, trapezium_perimeter t ≥ 4 + 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_min_trapezium_perimeter_l1503_150361


namespace NUMINAMATH_CALUDE_junk_mail_per_block_l1503_150339

/-- Given that a mailman distributes junk mail to blocks with the following conditions:
  1. The mailman gives 8 mails to each house in a block.
  2. There are 4 houses in a block.
Prove that the number of pieces of junk mail given to each block is 32. -/
theorem junk_mail_per_block (mails_per_house : ℕ) (houses_per_block : ℕ) 
  (h1 : mails_per_house = 8) (h2 : houses_per_block = 4) : 
  mails_per_house * houses_per_block = 32 := by
  sorry

#check junk_mail_per_block

end NUMINAMATH_CALUDE_junk_mail_per_block_l1503_150339


namespace NUMINAMATH_CALUDE_reciprocal_sum_pairs_l1503_150381

theorem reciprocal_sum_pairs : 
  ∃! (count : ℕ), ∃ (pairs : Finset (ℕ × ℕ)),
    pairs.card = count ∧
    (∀ (m n : ℕ), (m, n) ∈ pairs ↔ m > 0 ∧ n > 0 ∧ (1 : ℚ) / m + (1 : ℚ) / n = (1 : ℚ) / 3) ∧
    count = 3 :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_sum_pairs_l1503_150381


namespace NUMINAMATH_CALUDE_probability_triangle_or_circle_l1503_150384

theorem probability_triangle_or_circle :
  let total_figures : ℕ := 10
  let triangle_count : ℕ := 4
  let circle_count : ℕ := 3
  let target_count : ℕ := triangle_count + circle_count
  (target_count : ℚ) / total_figures = 7 / 10 :=
by sorry

end NUMINAMATH_CALUDE_probability_triangle_or_circle_l1503_150384


namespace NUMINAMATH_CALUDE_power_six_sum_l1503_150375

theorem power_six_sum (x : ℝ) (h : x + 1/x = 4) : x^6 + 1/x^6 = 2702 := by
  sorry

end NUMINAMATH_CALUDE_power_six_sum_l1503_150375


namespace NUMINAMATH_CALUDE_equivalent_form_proof_l1503_150386

theorem equivalent_form_proof (x y : ℝ) 
  (hx1 : x ≠ 0) (hx2 : x ≠ 3) (hy1 : y ≠ 0) (hy2 : y ≠ 7) 
  (h : (5 / x) + (4 / y) = (1 / 3)) : 
  x = (15 * y) / (y - 12) := by
sorry

end NUMINAMATH_CALUDE_equivalent_form_proof_l1503_150386


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1503_150324

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -2 < x ∧ x ≤ 1}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x ≤ 3}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -1 < x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1503_150324


namespace NUMINAMATH_CALUDE_parabola_normal_min_area_l1503_150374

noncomputable def min_y_coordinate : ℝ := (-3 + Real.sqrt 33) / 24

theorem parabola_normal_min_area (x₀ : ℝ) :
  let y₀ := x₀^2
  let normal_slope := -1 / (2 * x₀)
  let x₁ := -1 / (2 * x₀) - x₀
  let y₁ := x₁^2
  let triangle_area := (1/2) * (x₀ - x₁) * (y₀ + 1/2)
  (∀ x : ℝ, triangle_area ≤ ((1/2) * (x - (-1 / (2 * x) - x)) * (x^2 + 1/2))) →
  y₀ = min_y_coordinate := by
sorry

end NUMINAMATH_CALUDE_parabola_normal_min_area_l1503_150374


namespace NUMINAMATH_CALUDE_min_value_of_sum_l1503_150373

theorem min_value_of_sum (x y : ℝ) (h : x^2 - 2*x*y + y^2 - Real.sqrt 2*x - Real.sqrt 2*y + 6 = 0) :
  ∃ (u : ℝ), u = x + y ∧ u ≥ 3 * Real.sqrt 2 ∧ ∀ (v : ℝ), v = x + y → v ≥ u := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l1503_150373


namespace NUMINAMATH_CALUDE_quadrilateral_inequality_l1503_150363

-- Define the points
variable (A B C D O M N : ℝ × ℝ)

-- Define the triangle area function
def triangle_area (P Q R : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem quadrilateral_inequality 
  (h_convex : sorry) -- ABCD is a convex quadrilateral
  (h_intersect : sorry) -- AC and BD intersect at O
  (h_line : sorry) -- Line through O intersects AB at M and CD at N
  (h_ineq1 : triangle_area O M B > triangle_area O N D)
  (h_ineq2 : triangle_area O C N > triangle_area O A M) :
  triangle_area O A M + triangle_area O B C + triangle_area O N D >
  triangle_area O D A + triangle_area O M B + triangle_area O C N :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_inequality_l1503_150363


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l1503_150304

theorem root_sum_reciprocal (p q r A B C : ℝ) : 
  p ≠ q ∧ q ≠ r ∧ p ≠ r →
  (∀ x : ℝ, x^3 - 14*x^2 + 49*x - 24 = 0 ↔ x = p ∨ x = q ∨ x = r) →
  (∀ s : ℝ, s ≠ p ∧ s ≠ q ∧ s ≠ r → 
    1 / (s^3 - 14*s^2 + 49*s - 24) = A / (s - p) + B / (s - q) + C / (s - r)) →
  1 / A + 1 / B + 1 / C = 123 := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l1503_150304


namespace NUMINAMATH_CALUDE_range_of_m_l1503_150307

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {x | x + m ≥ 0}
def B : Set ℝ := {x | -2 < x ∧ x < 4}

-- Define the universal set U
def U : Set ℝ := Set.univ

-- State the theorem
theorem range_of_m (m : ℝ) : 
  (Set.compl (A m) ∩ B = ∅) → m ∈ Set.Ici 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1503_150307


namespace NUMINAMATH_CALUDE_right_triangle_area_l1503_150337

theorem right_triangle_area (a b : ℝ) (ha : a = 5) (hb : b = 12) : 
  (1/2 : ℝ) * a * b = 30 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1503_150337


namespace NUMINAMATH_CALUDE_marathon_completion_time_l1503_150383

/-- The time to complete a marathon given the distance and average pace -/
theorem marathon_completion_time 
  (distance : ℕ) 
  (avg_pace : ℕ) 
  (h1 : distance = 24)  -- marathon distance in miles
  (h2 : avg_pace = 9)   -- average pace in minutes per mile
  : distance * avg_pace = 216 := by
  sorry

end NUMINAMATH_CALUDE_marathon_completion_time_l1503_150383


namespace NUMINAMATH_CALUDE_marble_fraction_after_tripling_l1503_150318

theorem marble_fraction_after_tripling (total : ℝ) (h : total > 0) :
  let initial_green := (3/4 : ℝ) * total
  let initial_yellow := (1/4 : ℝ) * total
  let new_yellow := 3 * initial_yellow
  let new_total := initial_green + new_yellow
  new_yellow / new_total = (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_marble_fraction_after_tripling_l1503_150318


namespace NUMINAMATH_CALUDE_rectangle_division_l1503_150390

theorem rectangle_division (w₁ h₁ w₂ h₂ : ℝ) :
  w₁ > 0 ∧ h₁ > 0 ∧ w₂ > 0 ∧ h₂ > 0 →
  w₁ * h₁ = 6 →
  w₂ * h₁ = 15 →
  w₂ * h₂ = 25 →
  w₁ * h₂ = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_division_l1503_150390


namespace NUMINAMATH_CALUDE_polynomial_equality_l1503_150302

theorem polynomial_equality (x : ℝ) (g : ℝ → ℝ) : 
  (4 * x^5 + 3 * x^3 - 2 * x + 5 + g x = 7 * x^3 - 4 * x^2 + x + 2) → 
  (g x = -4 * x^5 + 4 * x^3 - 4 * x^2 + 3 * x - 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l1503_150302


namespace NUMINAMATH_CALUDE_square_area_error_l1503_150399

theorem square_area_error (actual_side : ℝ) (h : actual_side > 0) :
  let measured_side := actual_side * 1.1
  let actual_area := actual_side ^ 2
  let calculated_area := measured_side ^ 2
  let area_error := (calculated_area - actual_area) / actual_area
  area_error = 0.21 := by
sorry

end NUMINAMATH_CALUDE_square_area_error_l1503_150399


namespace NUMINAMATH_CALUDE_triangle_ratio_l1503_150336

theorem triangle_ratio (A B C : ℝ) (a b c : ℝ) :
  A = π / 3 →  -- 60° in radians
  a = Real.sqrt 13 →
  (a / Real.sin A = b / Real.sin B) →
  (b / Real.sin B = c / Real.sin C) →
  (a + b + c) / (Real.sin A + Real.sin B + Real.sin C) = 2 * Real.sqrt 39 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ratio_l1503_150336


namespace NUMINAMATH_CALUDE_horner_rule_v3_l1503_150353

def f (x : ℝ) : ℝ := 2*x^5 - 3*x^3 + 2*x^2 + x - 3

def horner_v3 (x : ℝ) : ℝ := 
  let v0 := 2*x
  let v1 := v0*x - 3
  let v2 := v1*x + 2
  v2*x + 1

theorem horner_rule_v3 : horner_v3 2 = 12 := by sorry

end NUMINAMATH_CALUDE_horner_rule_v3_l1503_150353


namespace NUMINAMATH_CALUDE_p_arithmetic_fibonacci_subsequence_l1503_150322

/-- Definition of a p-arithmetic Fibonacci sequence -/
def pArithmeticFibonacci (p : ℕ) (v : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, v (n + 2) = v (n + 1) + v n

/-- Theorem: The terms of a p-arithmetic Fibonacci sequence whose indices are divisible by p
    form another arithmetic Fibonacci sequence -/
theorem p_arithmetic_fibonacci_subsequence (p : ℕ) (v : ℕ → ℕ) 
    (h : pArithmeticFibonacci p v) :
  ∀ n : ℕ, n ≥ 1 → v ((n - 1) * p) + v (n * p) = v ((n + 1) * p) :=
by sorry

end NUMINAMATH_CALUDE_p_arithmetic_fibonacci_subsequence_l1503_150322


namespace NUMINAMATH_CALUDE_certain_number_bound_l1503_150320

theorem certain_number_bound (x y z : ℤ) (N : ℝ) 
  (h1 : x < y ∧ y < z)
  (h2 : (y - x : ℝ) > N)
  (h3 : Even x)
  (h4 : Odd y ∧ Odd z)
  (h5 : ∀ (a b : ℤ), (Even a ∧ Odd b ∧ a < b) → (b - a ≥ 7) → (z - x ≤ b - a)) :
  N < 3 := by
sorry

end NUMINAMATH_CALUDE_certain_number_bound_l1503_150320


namespace NUMINAMATH_CALUDE_divisibility_by_six_l1503_150366

theorem divisibility_by_six (y : ℕ) : y < 10 → (62000 + y * 100 + 16) % 6 = 0 ↔ y = 3 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_six_l1503_150366


namespace NUMINAMATH_CALUDE_seashell_count_l1503_150301

theorem seashell_count (sally_shells tom_shells jessica_shells : ℕ) 
  (h1 : sally_shells = 9)
  (h2 : tom_shells = 7)
  (h3 : jessica_shells = 5) :
  sally_shells + tom_shells + jessica_shells = 21 := by
  sorry

end NUMINAMATH_CALUDE_seashell_count_l1503_150301


namespace NUMINAMATH_CALUDE_club_choices_l1503_150325

/-- Represents a club with boys and girls -/
structure Club where
  boys : ℕ
  girls : ℕ

/-- The number of ways to choose a president and vice-president of the same gender -/
def sameGenderChoices (c : Club) : ℕ :=
  c.boys * (c.boys - 1) + c.girls * (c.girls - 1)

/-- Theorem stating that for a club with 10 boys and 10 girls, 
    there are 180 ways to choose a president and vice-president of the same gender -/
theorem club_choices (c : Club) (h1 : c.boys = 10) (h2 : c.girls = 10) :
  sameGenderChoices c = 180 := by
  sorry

#check club_choices

end NUMINAMATH_CALUDE_club_choices_l1503_150325


namespace NUMINAMATH_CALUDE_no_good_integers_l1503_150346

theorem no_good_integers : 
  ¬∃ (n : ℕ), n ≥ 1 ∧ 
  (∀ (k : ℕ), k > 0 → 
    ((∀ i ∈ Finset.range 9, k % (n + i + 1) = 0) → k % (n + 10) = 0)) :=
by sorry

end NUMINAMATH_CALUDE_no_good_integers_l1503_150346


namespace NUMINAMATH_CALUDE_simplify_fraction_with_sqrt_two_l1503_150355

theorem simplify_fraction_with_sqrt_two : 
  (1 / (1 + Real.sqrt 2)) * (1 / (1 - Real.sqrt 2)) = -1 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_with_sqrt_two_l1503_150355


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1503_150316

-- Problem 1
theorem problem_1 : (4 - Real.pi) ^ 0 + (1/3)⁻¹ - 2 * Real.cos (45 * π / 180) = 4 - Real.sqrt 2 := by
  sorry

-- Problem 2
theorem problem_2 (x : ℝ) (h : x ≠ 1 ∧ x ≠ -1) : 
  (1 + 1 / (x - 1)) / (x / (x^2 - 1)) = x + 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1503_150316


namespace NUMINAMATH_CALUDE_linear_equation_solution_l1503_150334

theorem linear_equation_solution :
  let x : ℝ := -4
  let y : ℝ := 2
  x + 3 * y = 2 := by sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l1503_150334


namespace NUMINAMATH_CALUDE_isabellas_hair_growth_l1503_150342

/-- Isabella's hair growth problem -/
theorem isabellas_hair_growth (initial_length : ℝ) : 
  initial_length + 6 = 24 → initial_length = 18 := by
  sorry

end NUMINAMATH_CALUDE_isabellas_hair_growth_l1503_150342


namespace NUMINAMATH_CALUDE_max_axes_of_symmetry_is_six_l1503_150349

/-- A line segment in a plane -/
structure LineSegment where
  -- Define properties of a line segment here
  -- For simplicity, we'll just use a placeholder
  id : Nat

/-- A configuration of three line segments in a plane -/
structure ThreeSegmentConfiguration where
  segments : Fin 3 → LineSegment

/-- An axis of symmetry for a configuration of line segments -/
structure AxisOfSymmetry where
  -- Define properties of an axis of symmetry here
  -- For simplicity, we'll just use a placeholder
  id : Nat

/-- The set of axes of symmetry for a given configuration -/
def axesOfSymmetry (config : ThreeSegmentConfiguration) : Set AxisOfSymmetry :=
  sorry

/-- The maximum number of axes of symmetry for any configuration of three line segments -/
def maxAxesOfSymmetry : Nat :=
  sorry

theorem max_axes_of_symmetry_is_six :
  maxAxesOfSymmetry = 6 :=
sorry

end NUMINAMATH_CALUDE_max_axes_of_symmetry_is_six_l1503_150349


namespace NUMINAMATH_CALUDE_min_balls_to_draw_for_given_counts_l1503_150327

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat

/-- The minimum number of balls to draw to guarantee the desired outcome -/
def minBallsToDraw (counts : BallCounts) : Nat :=
  sorry

/-- The theorem stating the minimum number of balls to draw for the given problem -/
theorem min_balls_to_draw_for_given_counts :
  let counts := BallCounts.mk 30 25 20 15 10
  minBallsToDraw counts = 81 := by
  sorry

end NUMINAMATH_CALUDE_min_balls_to_draw_for_given_counts_l1503_150327


namespace NUMINAMATH_CALUDE_smallest_a_correct_l1503_150380

/-- The smallest natural number a such that there are exactly 50 perfect squares in the interval (a, 3a) -/
def smallest_a : ℕ := 4486

/-- The number of perfect squares in the interval (a, 3a) -/
def count_squares (a : ℕ) : ℕ :=
  (Nat.sqrt (3 * a) - Nat.sqrt a).pred

theorem smallest_a_correct :
  (∀ b < smallest_a, count_squares b ≠ 50) ∧
  count_squares smallest_a = 50 :=
sorry

#eval smallest_a
#eval count_squares smallest_a

end NUMINAMATH_CALUDE_smallest_a_correct_l1503_150380


namespace NUMINAMATH_CALUDE_square_perimeter_l1503_150354

theorem square_perimeter (s : ℝ) (h : s > 0) : 
  (2 * s = 32) → (4 * s = 64) := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l1503_150354


namespace NUMINAMATH_CALUDE_hostel_provisions_l1503_150306

/-- The number of men initially in the hostel -/
def initial_men : ℕ := 250

/-- The number of days the provisions last initially -/
def initial_days : ℕ := 40

/-- The number of men who leave the hostel -/
def men_who_leave : ℕ := 50

/-- The number of days the provisions last after some men leave -/
def days_after_leaving : ℕ := 50

theorem hostel_provisions :
  initial_men * initial_days = (initial_men - men_who_leave) * days_after_leaving :=
by sorry

#check hostel_provisions

end NUMINAMATH_CALUDE_hostel_provisions_l1503_150306


namespace NUMINAMATH_CALUDE_divisibility_property_l1503_150351

theorem divisibility_property (A B n : ℕ) (hn : n = 7 ∨ n = 11 ∨ n = 13) 
  (h : n ∣ (B - A)) : n ∣ (1000 * A + B) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l1503_150351


namespace NUMINAMATH_CALUDE_prime_sums_count_l1503_150397

/-- Sequence of prime numbers -/
def primes : List Nat := sorry

/-- Function to generate sums by adding primes and skipping every third -/
def generateSums (n : Nat) : List Nat :=
  sorry

/-- Check if a number is prime -/
def isPrime (n : Nat) : Bool :=
  sorry

/-- Count prime sums in the first n generated sums -/
def countPrimeSums (n : Nat) : Nat :=
  sorry

/-- Main theorem: The number of prime sums among the first 12 generated sums is 5 -/
theorem prime_sums_count : countPrimeSums 12 = 5 := by
  sorry

end NUMINAMATH_CALUDE_prime_sums_count_l1503_150397


namespace NUMINAMATH_CALUDE_hall_volume_l1503_150388

/-- Proves that a rectangular hall with given dimensions and area equality has a volume of 972 cubic meters -/
theorem hall_volume (length width height : ℝ) : 
  length = 18 ∧ 
  width = 9 ∧ 
  2 * (length * width) = 2 * (length * height) + 2 * (width * height) → 
  length * width * height = 972 := by
  sorry

end NUMINAMATH_CALUDE_hall_volume_l1503_150388


namespace NUMINAMATH_CALUDE_square_equal_area_rectangle_l1503_150321

theorem square_equal_area_rectangle (rectangle_length rectangle_width square_side : ℝ) :
  rectangle_length = 25 ∧ 
  rectangle_width = 9 ∧ 
  square_side = 15 →
  rectangle_length * rectangle_width = square_side * square_side :=
by sorry

end NUMINAMATH_CALUDE_square_equal_area_rectangle_l1503_150321


namespace NUMINAMATH_CALUDE_monotonic_increase_interval_l1503_150391

-- Define a decreasing function on ℝ
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- Define the composition function g(x) = f(|x+2|)
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f (|x + 2|)

-- State the theorem
theorem monotonic_increase_interval
  (f : ℝ → ℝ) (h : DecreasingFunction f) :
  StrictMonoOn (g f) (Set.Iio (-2)) :=
sorry

end NUMINAMATH_CALUDE_monotonic_increase_interval_l1503_150391


namespace NUMINAMATH_CALUDE_sine_equality_proof_l1503_150359

theorem sine_equality_proof (n : ℤ) : 
  -90 ≤ n ∧ n ≤ 90 ∧ Real.sin (n * π / 180) = Real.sin (720 * π / 180) → n = 0 :=
by sorry

end NUMINAMATH_CALUDE_sine_equality_proof_l1503_150359


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1503_150371

theorem inequality_equivalence (x : ℝ) : 
  |((8 - x) / 4)|^2 < 4 ↔ 0 < x ∧ x < 16 :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1503_150371


namespace NUMINAMATH_CALUDE_expression_change_l1503_150311

/-- The change in the expression x^3 - 3x + 1 when x changes by a -/
def expressionChange (x a : ℝ) : ℝ := 
  (x + a)^3 - 3*(x + a) + 1 - (x^3 - 3*x + 1)

theorem expression_change (x a : ℝ) (h : a > 0) : 
  expressionChange x a = 3*a*x^2 + 3*a^2*x + a^3 - 3*a ∧
  expressionChange x (-a) = -3*a*x^2 + 3*a^2*x - a^3 + 3*a := by
  sorry

end NUMINAMATH_CALUDE_expression_change_l1503_150311


namespace NUMINAMATH_CALUDE_pairings_of_six_items_l1503_150317

/-- The number of possible pairings between two sets of 6 distinct items -/
def num_pairings (n : ℕ) : ℕ := n * n

/-- Theorem: The number of possible pairings between two sets of 6 distinct items is 36 -/
theorem pairings_of_six_items :
  num_pairings 6 = 36 := by
  sorry

end NUMINAMATH_CALUDE_pairings_of_six_items_l1503_150317


namespace NUMINAMATH_CALUDE_integer_root_of_polynomial_l1503_150300

-- Define the polynomial
def polynomial (d e f g x : ℚ) : ℚ := x^4 + d*x^3 + e*x^2 + f*x + g

-- State the theorem
theorem integer_root_of_polynomial (d e f g : ℚ) :
  (∃ (x : ℚ), x = 3 + Real.sqrt 5 ∧ polynomial d e f g x = 0) →
  (∃ (n : ℤ), polynomial d e f g (↑n) = 0 ∧ 
    (∀ (m : ℤ), m ≠ n → polynomial d e f g (↑m) ≠ 0)) →
  polynomial d e f g (-3) = 0 :=
sorry

end NUMINAMATH_CALUDE_integer_root_of_polynomial_l1503_150300


namespace NUMINAMATH_CALUDE_middle_digit_zero_l1503_150369

theorem middle_digit_zero (a b c : Nat) (M : Nat) :
  (0 ≤ a ∧ a < 6) →
  (0 ≤ b ∧ b < 6) →
  (0 ≤ c ∧ c < 6) →
  M = 36 * a + 6 * b + c →
  M = 64 * a + 8 * b + c →
  b = 0 := by
  sorry

end NUMINAMATH_CALUDE_middle_digit_zero_l1503_150369


namespace NUMINAMATH_CALUDE_triangle_area_l1503_150328

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    and area S, prove that if 4S = √3(a² + b² - c²) and 
    f(x) = 4sin(x)cos(x + π/6) + 1 attains its maximum value b when x = A,
    then the area S of the triangle is √3/2. -/
theorem triangle_area (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  4 * S = Real.sqrt 3 * (a^2 + b^2 - c^2) →
  (∀ x, 4 * Real.sin x * Real.cos (x + π/6) + 1 ≤ b) →
  (4 * Real.sin A * Real.cos (A + π/6) + 1 = b) →
  S = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1503_150328


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_range_l1503_150392

/-- Given real numbers a, b, c forming a geometric sequence with sum 1,
    prove that a + c is non-negative and unbounded above. -/
theorem geometric_sequence_sum_range (a b c : ℝ) : 
  (∃ r : ℝ, a = r ∧ b = r^2 ∧ c = r^3) →  -- geometric sequence condition
  a + b + c = 1 →                        -- sum condition
  (a + c ≥ 0 ∧ ∀ M : ℝ, ∃ x y z : ℝ, 
    (∃ r : ℝ, x = r ∧ y = r^2 ∧ z = r^3) ∧ 
    x + y + z = 1 ∧ 
    x + z > M) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_range_l1503_150392


namespace NUMINAMATH_CALUDE_min_sum_fraction_l1503_150305

theorem min_sum_fraction (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (3 * b)) + (b / (6 * c)) + (c / (9 * a)) ≥ 3 / Real.rpow 162 (1/3) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_fraction_l1503_150305


namespace NUMINAMATH_CALUDE_channel_transmission_theorem_l1503_150315

/-- Channel transmission probabilities -/
structure ChannelProb where
  α : ℝ
  β : ℝ
  h_α_pos : 0 < α
  h_α_lt_one : α < 1
  h_β_pos : 0 < β
  h_β_lt_one : β < 1

/-- Single transmission probability for sequence 1, 0, 1 -/
def single_trans_prob (cp : ChannelProb) : ℝ := (1 - cp.α) * (1 - cp.β)^2

/-- Triple transmission probability for decoding 0 as 0 -/
def triple_trans_prob_0 (cp : ChannelProb) : ℝ :=
  (1 - cp.α)^3 + 3 * cp.α * (1 - cp.α)^2

/-- Single transmission probability for decoding 0 as 0 -/
def single_trans_prob_0 (cp : ChannelProb) : ℝ := 1 - cp.α

theorem channel_transmission_theorem (cp : ChannelProb) :
  single_trans_prob cp = (1 - cp.α) * (1 - cp.β)^2 ∧
  (cp.α < 1/2 → triple_trans_prob_0 cp > single_trans_prob_0 cp) := by sorry

end NUMINAMATH_CALUDE_channel_transmission_theorem_l1503_150315


namespace NUMINAMATH_CALUDE_min_value_expression_l1503_150385

theorem min_value_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  let M := Real.sqrt (1 + 2 * a^2) + 2 * Real.sqrt ((5/12)^2 + b^2)
  ∀ x y, x > 0 ∧ y > 0 ∧ x + y = 1 →
    Real.sqrt (1 + 2 * x^2) + 2 * Real.sqrt ((5/12)^2 + y^2) ≥ 5 * Real.sqrt 34 / 12 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1503_150385


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1503_150357

theorem polynomial_factorization (a b : ℝ) : 
  a^2 - b^2 + 2*a + 1 = (a - b + 1) * (a + b + 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1503_150357


namespace NUMINAMATH_CALUDE_problem_statement_l1503_150394

theorem problem_statement (a b : ℝ) 
  (h1 : a + 1 / (a + 1) = b + 1 / (b - 1) - 2)
  (h2 : a - b + 2 ≠ 0) : 
  a * b - a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1503_150394


namespace NUMINAMATH_CALUDE_platyfish_count_l1503_150377

/-- The number of goldfish in the tank -/
def num_goldfish : ℕ := 3

/-- The number of red balls each goldfish plays with -/
def red_balls_per_goldfish : ℕ := 10

/-- The number of white balls each platyfish plays with -/
def white_balls_per_platyfish : ℕ := 5

/-- The total number of balls in the fish tank -/
def total_balls : ℕ := 80

/-- The number of platyfish in the tank -/
def num_platyfish : ℕ := (total_balls - num_goldfish * red_balls_per_goldfish) / white_balls_per_platyfish

theorem platyfish_count : num_platyfish = 10 := by
  sorry

end NUMINAMATH_CALUDE_platyfish_count_l1503_150377


namespace NUMINAMATH_CALUDE_negation_of_p_l1503_150338

/-- Proposition p: a and b are both even numbers -/
def p (a b : ℤ) : Prop := Even a ∧ Even b

/-- The negation of proposition p -/
theorem negation_of_p (a b : ℤ) : ¬(p a b) ↔ ¬(Even a ∧ Even b) := by sorry

end NUMINAMATH_CALUDE_negation_of_p_l1503_150338


namespace NUMINAMATH_CALUDE_range_of_f_l1503_150393

def f (x : ℕ) : ℤ := 2 * x - 3

def domain : Set ℕ := {x | 1 ≤ x ∧ x ≤ 5}

theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {-1, 1, 3, 5, 7} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l1503_150393


namespace NUMINAMATH_CALUDE_problem_statement_l1503_150364

open Real

theorem problem_statement :
  (∀ x > 0, exp x - 2 > x - 1 ∧ x - 1 ≥ log x) ∧
  (∀ m : ℤ, m < 1 → ¬∃ x y, 0 < x ∧ 0 < y ∧ x ≠ y ∧ exp x - log x - m - 2 = 0 ∧ exp y - log y - m - 2 = 0) ∧
  (∃ x y, 0 < x ∧ 0 < y ∧ x ≠ y ∧ exp x - log x - 1 - 2 = 0 ∧ exp y - log y - 1 - 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1503_150364


namespace NUMINAMATH_CALUDE_integral_x_squared_minus_x_l1503_150389

theorem integral_x_squared_minus_x : ∫ x in (0)..(2), (x^2 - x) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_integral_x_squared_minus_x_l1503_150389


namespace NUMINAMATH_CALUDE_bank_profit_maximization_l1503_150376

/-- The bank's profit maximization problem -/
theorem bank_profit_maximization
  (k : ℝ) -- Proportionality constant
  (h_k_pos : k > 0) -- k is positive
  (loan_rate : ℝ := 0.048) -- Loan interest rate
  (deposit_rate : ℝ) -- Deposit interest rate
  (h_deposit_rate : deposit_rate > 0 ∧ deposit_rate < loan_rate) -- Deposit rate is between 0 and loan rate
  (deposit_amount : ℝ := k * deposit_rate^2) -- Deposit amount formula
  (profit : ℝ → ℝ := λ x => loan_rate * k * x^2 - k * x^3) -- Profit function
  : (∀ x, x > 0 ∧ x < loan_rate → profit x ≤ profit 0.032) :=
by sorry

end NUMINAMATH_CALUDE_bank_profit_maximization_l1503_150376


namespace NUMINAMATH_CALUDE_class_test_problem_l1503_150343

theorem class_test_problem (first_correct : Real) (second_correct : Real) (both_correct : Real)
  (h1 : first_correct = 0.75)
  (h2 : second_correct = 0.65)
  (h3 : both_correct = 0.60) :
  1 - (first_correct + second_correct - both_correct) = 0.20 := by
  sorry

end NUMINAMATH_CALUDE_class_test_problem_l1503_150343


namespace NUMINAMATH_CALUDE_sphere_volume_l1503_150398

theorem sphere_volume (prism_length prism_width prism_height : ℝ) 
  (sphere_volume : ℝ → ℝ) (L : ℝ) :
  prism_length = 4 →
  prism_width = 2 →
  prism_height = 1 →
  (∀ r : ℝ, sphere_volume r = (4 / 3) * π * r^3) →
  (∃ r : ℝ, 4 * π * r^2 = 2 * (prism_length * prism_width + 
    prism_length * prism_height + prism_width * prism_height)) →
  (∃ r : ℝ, sphere_volume r = L * Real.sqrt 2 / Real.sqrt π) →
  L = 14 * Real.sqrt 14 / 3 :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_l1503_150398


namespace NUMINAMATH_CALUDE_product_and_reciprocal_relation_l1503_150330

theorem product_and_reciprocal_relation (x y : ℝ) : 
  x > 0 → y > 0 → x * y = 16 → 1 / x = 3 * (1 / y) → |x - y| = (8 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_product_and_reciprocal_relation_l1503_150330


namespace NUMINAMATH_CALUDE_diane_gingerbreads_l1503_150310

/-- Proves that given Diane's baking conditions, each of the four trays contains 25 gingerbreads -/
theorem diane_gingerbreads :
  ∀ (x : ℕ),
  (4 * x + 3 * 20 = 160) →
  x = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_diane_gingerbreads_l1503_150310


namespace NUMINAMATH_CALUDE_inequality_solution_inequality_proof_l1503_150323

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1|

-- Theorem for part (I)
theorem inequality_solution (x : ℝ) :
  f (x - 1) + f (x + 3) ≥ 6 ↔ x ≤ -3 ∨ x ≥ 3 :=
sorry

-- Theorem for part (II)
theorem inequality_proof (a b : ℝ) (h1 : |a| < 1) (h2 : |b| < 1) (h3 : a ≠ 0) :
  f (a * b) > |a| * f (b / a) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_inequality_proof_l1503_150323


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1503_150360

theorem inequality_solution_set (x : ℝ) : 
  (5 / (x + 2) ≥ 1 ∧ x + 2 ≠ 0) ↔ -2 < x ∧ x ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1503_150360


namespace NUMINAMATH_CALUDE_intersection_not_in_third_quadrant_l1503_150319

/-- The intersection point of y = 2x + m and y = -x + 3 cannot be in the third quadrant -/
theorem intersection_not_in_third_quadrant (m : ℝ) : 
  ∀ x y : ℝ, y = 2*x + m ∧ y = -x + 3 → ¬(x < 0 ∧ y < 0) :=
by sorry

end NUMINAMATH_CALUDE_intersection_not_in_third_quadrant_l1503_150319


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1503_150356

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + b = 1) :
  ∃ (min : ℝ), min = 3 + 2 * Real.sqrt 2 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → 2 * x + y = 1 → 1 / x + 1 / y ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1503_150356


namespace NUMINAMATH_CALUDE_water_consumption_proof_l1503_150314

/-- Calculates the total water consumption for horses over a given number of days -/
def total_water_consumption (initial_horses : ℕ) (added_horses : ℕ) (drinking_water : ℕ) (bathing_water : ℕ) (days : ℕ) : ℕ :=
  (initial_horses + added_horses) * (drinking_water + bathing_water) * days

/-- Proves that under given conditions, the total water consumption for 28 days is 1568 liters -/
theorem water_consumption_proof :
  total_water_consumption 3 5 5 2 28 = 1568 := by
  sorry

#eval total_water_consumption 3 5 5 2 28

end NUMINAMATH_CALUDE_water_consumption_proof_l1503_150314


namespace NUMINAMATH_CALUDE_heart_ratio_eq_half_l1503_150358

def heart (n m : ℕ) : ℕ := n^3 * m^2

theorem heart_ratio_eq_half :
  (heart 2 4) / (heart 4 2) = 1/2 := by sorry

end NUMINAMATH_CALUDE_heart_ratio_eq_half_l1503_150358


namespace NUMINAMATH_CALUDE_modulus_of_2_plus_i_times_i_l1503_150368

theorem modulus_of_2_plus_i_times_i : Complex.abs ((2 + Complex.I) * Complex.I) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_2_plus_i_times_i_l1503_150368


namespace NUMINAMATH_CALUDE_three_digit_number_problem_l1503_150387

theorem three_digit_number_problem (A B : ℝ) : 
  (100 ≤ A ∧ A < 1000) →  -- A is a three-digit number
  (B = A / 10 ∨ B = A / 100 ∨ B = A / 1000) →  -- B is obtained by placing a decimal point in front of one of A's digits
  (A - B = 478.8) →  -- Given condition
  A = 532 := by
sorry

end NUMINAMATH_CALUDE_three_digit_number_problem_l1503_150387


namespace NUMINAMATH_CALUDE_proportion_solution_l1503_150350

theorem proportion_solution (x : ℚ) : (3 : ℚ) / 12 = x / 16 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l1503_150350


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l1503_150362

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y - x - 2 * y = 0) :
  x + y ≥ 3 + 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l1503_150362


namespace NUMINAMATH_CALUDE_dave_earnings_l1503_150365

/-- Calculates the total money earned from selling video games -/
def total_money_earned (action_games adventure_games roleplaying_games : ℕ) 
  (action_price adventure_price roleplaying_price : ℕ) : ℕ :=
  action_games * action_price + 
  adventure_games * adventure_price + 
  roleplaying_games * roleplaying_price

/-- Proves that Dave earns $49 by selling all working games -/
theorem dave_earnings : 
  total_money_earned 3 2 3 6 5 7 = 49 := by
  sorry

end NUMINAMATH_CALUDE_dave_earnings_l1503_150365


namespace NUMINAMATH_CALUDE_pretzels_eaten_difference_l1503_150372

/-- The number of pretzels Marcus ate compared to John -/
def pretzels_difference (total : ℕ) (john : ℕ) (alan : ℕ) (marcus : ℕ) : ℕ :=
  marcus - john

/-- Theorem stating the difference in pretzels eaten between Marcus and John -/
theorem pretzels_eaten_difference 
  (total : ℕ) 
  (john : ℕ) 
  (alan : ℕ) 
  (marcus : ℕ) 
  (h1 : total = 95)
  (h2 : john = 28)
  (h3 : alan = john - 9)
  (h4 : marcus > john)
  (h5 : marcus = 40) :
  pretzels_difference total john alan marcus = 12 := by
  sorry

end NUMINAMATH_CALUDE_pretzels_eaten_difference_l1503_150372


namespace NUMINAMATH_CALUDE_candy_problem_l1503_150352

theorem candy_problem (x : ℚ) : 
  (2/9 * x - 2/3 - 4 = 8) → x = 57 := by
  sorry

end NUMINAMATH_CALUDE_candy_problem_l1503_150352


namespace NUMINAMATH_CALUDE_soccer_penalty_kicks_l1503_150395

theorem soccer_penalty_kicks (total_players : ℕ) (goalkeepers : ℕ) : 
  total_players = 16 → goalkeepers = 2 → (total_players - goalkeepers) * goalkeepers = 30 := by
  sorry

end NUMINAMATH_CALUDE_soccer_penalty_kicks_l1503_150395


namespace NUMINAMATH_CALUDE_increase_when_multiplied_l1503_150326

theorem increase_when_multiplied (n : ℕ) (m : ℕ) (increase : ℕ) : n = 14 → m = 15 → increase = m * n - n → increase = 196 := by
  sorry

end NUMINAMATH_CALUDE_increase_when_multiplied_l1503_150326


namespace NUMINAMATH_CALUDE_sin_52pi_over_3_l1503_150308

theorem sin_52pi_over_3 : Real.sin (52 * π / 3) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_52pi_over_3_l1503_150308


namespace NUMINAMATH_CALUDE_plan_comparison_l1503_150333

def suit_price : ℝ := 500
def tie_price : ℝ := 80
def num_suits : ℕ := 20

def plan1_cost (x : ℝ) : ℝ := 8400 + 80 * x
def plan2_cost (x : ℝ) : ℝ := 9000 + 72 * x

theorem plan_comparison (x : ℝ) (h : x > 20) :
  plan1_cost x ≤ plan2_cost x ↔ x ≤ 75 := by sorry

end NUMINAMATH_CALUDE_plan_comparison_l1503_150333


namespace NUMINAMATH_CALUDE_minervas_stamps_l1503_150340

/-- Given that Lizette has 813 stamps and 125 more stamps than Minerva,
    prove that Minerva has 688 stamps. -/
theorem minervas_stamps (lizette_stamps : ℕ) (difference : ℕ) 
  (h1 : lizette_stamps = 813)
  (h2 : difference = 125)
  (h3 : lizette_stamps = difference + minerva_stamps) :
  minerva_stamps = 688 := by
  sorry

end NUMINAMATH_CALUDE_minervas_stamps_l1503_150340


namespace NUMINAMATH_CALUDE_equation_equivalence_l1503_150309

-- Define the original equation
def original_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 4)^2 + y^2) + Real.sqrt ((x + 4)^2 + y^2) = 10

-- Define the simplified equation
def simplified_equation (x y : ℝ) : Prop :=
  x^2 / 25 + y^2 / 9 = 1

-- Theorem statement
theorem equation_equivalence :
  ∀ x y : ℝ, original_equation x y ↔ simplified_equation x y :=
by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l1503_150309


namespace NUMINAMATH_CALUDE_towel_area_decrease_l1503_150312

/-- Represents the properties of a fabric material -/
structure Material where
  cotton_percent : Real
  polyester_percent : Real
  cotton_length_shrinkage : Real
  cotton_breadth_shrinkage : Real
  polyester_length_shrinkage : Real
  polyester_breadth_shrinkage : Real

/-- Calculates the area decrease percentage of a fabric after shrinkage -/
def calculate_area_decrease (m : Material) : Real :=
  let effective_length_shrinkage := 
    m.cotton_length_shrinkage * m.cotton_percent + m.polyester_length_shrinkage * m.polyester_percent
  let effective_breadth_shrinkage := 
    m.cotton_breadth_shrinkage * m.cotton_percent + m.polyester_breadth_shrinkage * m.polyester_percent
  1 - (1 - effective_length_shrinkage) * (1 - effective_breadth_shrinkage)

/-- The towel material properties -/
def towel : Material := {
  cotton_percent := 0.60
  polyester_percent := 0.40
  cotton_length_shrinkage := 0.35
  cotton_breadth_shrinkage := 0.45
  polyester_length_shrinkage := 0.25
  polyester_breadth_shrinkage := 0.30
}

/-- Theorem: The area decrease of the towel after bleaching is approximately 57.91% -/
theorem towel_area_decrease : 
  ∃ ε > 0, |calculate_area_decrease towel - 0.5791| < ε :=
by sorry

end NUMINAMATH_CALUDE_towel_area_decrease_l1503_150312


namespace NUMINAMATH_CALUDE_quadratic_inequality_implies_a_greater_than_two_l1503_150347

theorem quadratic_inequality_implies_a_greater_than_two (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ x^2 - a*x + 1 < 0) → a > 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implies_a_greater_than_two_l1503_150347


namespace NUMINAMATH_CALUDE_special_ellipse_eccentricity_special_ellipse_equation_l1503_150313

/-- An ellipse with the given properties -/
structure SpecialEllipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  h_foci : F₁.1 < F₂.1 -- F₁ is left focus, F₂ is right focus
  h_ellipse : ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 ↔ (x, y) ∈ ({A, B} : Set (ℝ × ℝ))
  h_line : A.2 - F₁.2 = A.1 - F₁.1 ∧ B.2 - F₁.2 = B.1 - F₁.1 -- Line through F₁ with slope 1
  h_arithmetic : ∃ (d : ℝ), dist A F₂ + d = dist A B ∧ dist A B + d = dist B F₂
  h_circle : ∃ (r : ℝ), dist A (-2, 0) = r ∧ dist B (-2, 0) = r

/-- The eccentricity of the special ellipse is √2/2 -/
theorem special_ellipse_eccentricity (E : SpecialEllipse) : 
  (E.a^2 - E.b^2) / E.a^2 = 1/2 := by sorry

/-- The equation of the special ellipse is x²/72 + y²/36 = 1 -/
theorem special_ellipse_equation (E : SpecialEllipse) : 
  E.a^2 = 72 ∧ E.b^2 = 36 := by sorry

end NUMINAMATH_CALUDE_special_ellipse_eccentricity_special_ellipse_equation_l1503_150313
