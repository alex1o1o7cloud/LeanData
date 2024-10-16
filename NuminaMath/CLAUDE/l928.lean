import Mathlib

namespace NUMINAMATH_CALUDE_solution_set_min_sum_reciprocals_equality_condition_l928_92812

-- Define the inequality
def inequality (x : ℝ) : Prop := |x + 1| + |2*x - 1| ≤ 3

-- Theorem 1: Solution set of the inequality
theorem solution_set : 
  {x : ℝ | inequality x} = {x : ℝ | -1 ≤ x ∧ x ≤ 1} := by sorry

-- Theorem 2: Minimum value of the sum of reciprocals
theorem min_sum_reciprocals (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_constraint : a + b + c = 2) :
  1/a + 1/b + 1/c ≥ 9/2 := by sorry

-- Theorem 3: Condition for equality
theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_constraint : a + b + c = 2) :
  1/a + 1/b + 1/c = 9/2 ↔ a = 2/3 ∧ b = 2/3 ∧ c = 2/3 := by sorry

end NUMINAMATH_CALUDE_solution_set_min_sum_reciprocals_equality_condition_l928_92812


namespace NUMINAMATH_CALUDE_A_union_B_eq_A_l928_92873

def A : Set ℝ := {x | -1 < x ∧ x < 4}
def B : Set ℝ := {x | 0 < x ∧ x < Real.exp 1}

theorem A_union_B_eq_A : A ∪ B = A := by sorry

end NUMINAMATH_CALUDE_A_union_B_eq_A_l928_92873


namespace NUMINAMATH_CALUDE_irreducibility_of_polynomial_l928_92819

theorem irreducibility_of_polynomial :
  ¬∃ (p q : Polynomial ℤ), (Polynomial.degree p ≥ 1) ∧ (Polynomial.degree q ≥ 1) ∧ (p * q = X^5 + 2*X + 1) :=
by sorry

end NUMINAMATH_CALUDE_irreducibility_of_polynomial_l928_92819


namespace NUMINAMATH_CALUDE_quiz_competition_l928_92887

theorem quiz_competition (total_questions : ℕ) (correct_score : ℤ) (incorrect_score : ℤ) (total_score : ℤ) 
  (h1 : total_questions = 100)
  (h2 : correct_score = 10)
  (h3 : incorrect_score = -5)
  (h4 : total_score = 850) :
  ∃ (incorrect : ℕ), 
    incorrect = 10 ∧ 
    (total_questions - incorrect : ℤ) * correct_score + incorrect * incorrect_score = total_score :=
by sorry

end NUMINAMATH_CALUDE_quiz_competition_l928_92887


namespace NUMINAMATH_CALUDE_apple_juice_cost_l928_92842

def orange_juice_cost : ℚ := 70 / 100
def total_bottles : ℕ := 70
def total_cost : ℚ := 4620 / 100
def orange_juice_bottles : ℕ := 42

theorem apple_juice_cost :
  let apple_juice_bottles : ℕ := total_bottles - orange_juice_bottles
  let apple_juice_total_cost : ℚ := total_cost - (orange_juice_cost * orange_juice_bottles)
  apple_juice_total_cost / apple_juice_bottles = 60 / 100 :=
by sorry

end NUMINAMATH_CALUDE_apple_juice_cost_l928_92842


namespace NUMINAMATH_CALUDE_horner_method_v4_l928_92803

def horner_polynomial (x : ℝ) : ℝ := 1 + 8*x + 7*x^2 + 5*x^4 + 4*x^5 + 3*x^6

def horner_v4 (x : ℝ) : ℝ :=
  let v0 := 3
  let v1 := v0 * x + 4
  let v2 := v1 * x + 5
  let v3 := v2 * x + 0
  v3 * x + 7

theorem horner_method_v4 :
  horner_v4 5 = 2507 :=
by sorry

end NUMINAMATH_CALUDE_horner_method_v4_l928_92803


namespace NUMINAMATH_CALUDE_subset_removal_distinctness_l928_92828

theorem subset_removal_distinctness (n : ℕ) :
  ∀ (S : Finset ℕ) (A : Fin n → Finset ℕ),
    S = Finset.range n →
    (∀ i j, i ≠ j → A i ≠ A j) →
    (∀ i, A i ⊆ S) →
    ∃ x ∈ S, ∀ i j, i ≠ j → A i \ {x} ≠ A j \ {x} :=
by sorry

end NUMINAMATH_CALUDE_subset_removal_distinctness_l928_92828


namespace NUMINAMATH_CALUDE_car_distance_covered_l928_92893

/-- Proves that a car traveling at 97.5 km/h for 4 hours covers a distance of 390 km -/
theorem car_distance_covered (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 97.5 → time = 4 → distance = speed * time → distance = 390 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_covered_l928_92893


namespace NUMINAMATH_CALUDE_factors_of_243_l928_92817

theorem factors_of_243 : Finset.card (Nat.divisors 243) = 6 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_243_l928_92817


namespace NUMINAMATH_CALUDE_square_side_length_l928_92895

/-- Given a rectangle with width 2 and a square placed next to it,
    if the total length of the bottom side is 7,
    then the side length of the square is 5. -/
theorem square_side_length (rectangle_width square_side total_length : ℝ) : 
  rectangle_width = 2 →
  total_length = 7 →
  total_length = rectangle_width + square_side →
  square_side = 5 := by
sorry


end NUMINAMATH_CALUDE_square_side_length_l928_92895


namespace NUMINAMATH_CALUDE_complex_purely_imaginary_solution_l928_92899

/-- A complex number is purely imaginary if its real part is zero -/
def isPurelyImaginary (z : ℂ) : Prop := z.re = 0

/-- Given that z and (z+2)^2 - 8i are both purely imaginary, prove that z = -2i -/
theorem complex_purely_imaginary_solution (z : ℂ) 
  (h1 : isPurelyImaginary z) 
  (h2 : isPurelyImaginary ((z + 2)^2 - 8*I)) : 
  z = -2*I := by
  sorry

end NUMINAMATH_CALUDE_complex_purely_imaginary_solution_l928_92899


namespace NUMINAMATH_CALUDE_circle_radius_is_sqrt_two_l928_92861

/-- A circle inside a right angle with specific properties -/
structure CircleInRightAngle where
  /-- The radius of the circle -/
  R : ℝ
  /-- The length of chord AB -/
  AB : ℝ
  /-- The length of chord CD -/
  CD : ℝ
  /-- The circle is inside a right angle -/
  inside_right_angle : True
  /-- The circle is tangent to one side of the angle -/
  tangent_to_side : True
  /-- The circle intersects the other side at points A and B -/
  intersects_side : True
  /-- The circle intersects the angle bisector at points C and D -/
  intersects_bisector : True
  /-- AB = √6 -/
  h_AB : AB = Real.sqrt 6
  /-- CD = √7 -/
  h_CD : CD = Real.sqrt 7

/-- The theorem stating that the radius of the circle is √2 -/
theorem circle_radius_is_sqrt_two (c : CircleInRightAngle) : c.R = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_is_sqrt_two_l928_92861


namespace NUMINAMATH_CALUDE_smallest_number_of_sweets_l928_92875

theorem smallest_number_of_sweets (x : ℕ) : 
  x > 0 ∧ 
  x % 6 = 5 ∧ 
  x % 8 = 7 ∧ 
  x % 9 = 8 ∧ 
  (∀ y : ℕ, y > 0 → y % 6 = 5 → y % 8 = 7 → y % 9 = 8 → x ≤ y) → 
  x = 71 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_of_sweets_l928_92875


namespace NUMINAMATH_CALUDE_decrease_by_percentage_decrease_80_by_150_percent_l928_92838

theorem decrease_by_percentage (n : ℝ) (p : ℝ) : 
  n - (p / 100) * n = n * (1 - p / 100) := by sorry

theorem decrease_80_by_150_percent : 
  80 - (150 / 100) * 80 = -40 := by sorry

end NUMINAMATH_CALUDE_decrease_by_percentage_decrease_80_by_150_percent_l928_92838


namespace NUMINAMATH_CALUDE_tissues_used_l928_92898

theorem tissues_used (tissues_per_box : ℕ) (boxes_bought : ℕ) (tissues_left : ℕ)
  (h1 : tissues_per_box = 160)
  (h2 : boxes_bought = 3)
  (h3 : tissues_left = 270) :
  tissues_per_box * boxes_bought - tissues_left = 210 :=
by sorry

end NUMINAMATH_CALUDE_tissues_used_l928_92898


namespace NUMINAMATH_CALUDE_shells_added_correct_l928_92864

/-- Given an initial amount of shells and a final amount of shells,
    calculate the amount of shells added. -/
def shells_added (initial final : ℕ) : ℕ :=
  final - initial

/-- Theorem stating that given the initial amount of 5 pounds and
    final amount of 28 pounds, the amount of shells added is 23 pounds. -/
theorem shells_added_correct :
  shells_added 5 28 = 23 := by
  sorry

end NUMINAMATH_CALUDE_shells_added_correct_l928_92864


namespace NUMINAMATH_CALUDE_first_neighbor_height_l928_92891

/-- The height of Lucille's house in feet -/
def lucille_height : ℝ := 80

/-- The height of the second neighbor's house in feet -/
def neighbor2_height : ℝ := 99

/-- The height difference between Lucille's house and the average height in feet -/
def height_difference : ℝ := 3

/-- The height of the first neighbor's house in feet -/
def neighbor1_height : ℝ := 70

theorem first_neighbor_height :
  (lucille_height + neighbor1_height + neighbor2_height) / 3 - height_difference = lucille_height :=
by sorry

end NUMINAMATH_CALUDE_first_neighbor_height_l928_92891


namespace NUMINAMATH_CALUDE_product_loss_percentage_l928_92869

/-- Proves the percentage loss of a product given specific selling prices and gain percentages --/
theorem product_loss_percentage 
  (cp : ℝ) -- Cost price
  (sp_gain : ℝ) -- Selling price with gain
  (sp_loss : ℝ) -- Selling price with loss
  (gain_percent : ℝ) -- Gain percentage
  (h1 : sp_gain = cp * (1 + gain_percent / 100)) -- Condition for selling price with gain
  (h2 : sp_gain = 168) -- Given selling price with gain
  (h3 : gain_percent = 20) -- Given gain percentage
  (h4 : sp_loss = 119) -- Given selling price with loss
  : (cp - sp_loss) / cp * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_product_loss_percentage_l928_92869


namespace NUMINAMATH_CALUDE_reflection_of_circle_center_l928_92865

/-- Reflects a point (x, y) about the line y = -x --/
def reflect_about_negative_diagonal (p : ℝ × ℝ) : ℝ × ℝ :=
  (-(p.2), -(p.1))

/-- The original center of the circle --/
def original_center : ℝ × ℝ := (4, -3)

/-- The expected reflected center of the circle --/
def expected_reflected_center : ℝ × ℝ := (3, -4)

theorem reflection_of_circle_center :
  reflect_about_negative_diagonal original_center = expected_reflected_center := by
  sorry

end NUMINAMATH_CALUDE_reflection_of_circle_center_l928_92865


namespace NUMINAMATH_CALUDE_largest_number_with_distinct_digits_summing_to_19_l928_92859

/-- Checks if all digits in a number are different -/
def hasDistinctDigits (n : ℕ) : Bool := sorry

/-- Calculates the sum of digits of a number -/
def digitSum (n : ℕ) : ℕ := sorry

/-- The target sum of digits -/
def targetSum : ℕ := 19

/-- The proposed largest number -/
def largestNumber : ℕ := 943210

theorem largest_number_with_distinct_digits_summing_to_19 :
  (∀ m : ℕ, m > largestNumber → 
    ¬(hasDistinctDigits m ∧ digitSum m = targetSum)) ∧
  hasDistinctDigits largestNumber ∧
  digitSum largestNumber = targetSum :=
sorry

end NUMINAMATH_CALUDE_largest_number_with_distinct_digits_summing_to_19_l928_92859


namespace NUMINAMATH_CALUDE_regular_polygon_vertices_l928_92837

/-- A regular polygon with an angle of 135° between a vertex and the vertex two positions away has 12 vertices. -/
theorem regular_polygon_vertices (n : ℕ) (h_regular : n ≥ 3) : 
  (2 * (360 : ℝ) / n = 135) → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_vertices_l928_92837


namespace NUMINAMATH_CALUDE_tan_45_degrees_equals_one_l928_92811

theorem tan_45_degrees_equals_one :
  Real.tan (π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_45_degrees_equals_one_l928_92811


namespace NUMINAMATH_CALUDE_tangent_line_determines_b_l928_92872

/-- A curve defined by y = x³ + ax + b -/
def curve (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x + b

/-- The derivative of the curve -/
def curve_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + a

/-- A line defined by y = mx + c -/
def line (m c : ℝ) (x : ℝ) : ℝ := m*x + c

theorem tangent_line_determines_b (a b : ℝ) :
  curve a b 1 = 3 ∧
  curve_derivative a 1 = 2 →
  b = 3 := by sorry

end NUMINAMATH_CALUDE_tangent_line_determines_b_l928_92872


namespace NUMINAMATH_CALUDE_inequality_proof_l928_92894

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a + 1) * Real.log x + a * x^2 + 1

theorem inequality_proof (a : ℝ) (h : a ≤ -2) :
  ∀ (x₁ x₂ : ℝ), x₁ > 0 → x₂ > 0 → (f a x₁ - f a x₂) / (x₂ - x₁) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l928_92894


namespace NUMINAMATH_CALUDE_max_m_value_l928_92862

def f (x : ℝ) := x^3 - 3*x^2

theorem max_m_value (m : ℝ) :
  (∀ x ∈ Set.Icc (-1) m, f x ∈ Set.Icc (-4) 0) →
  (∃ x ∈ Set.Icc (-1) m, f x = -4) →
  (∃ x ∈ Set.Icc (-1) m, f x = 0) →
  m ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_max_m_value_l928_92862


namespace NUMINAMATH_CALUDE_tangent_slope_angle_at_one_l928_92816

noncomputable def f (x : ℝ) : ℝ := -Real.sqrt 3 / 3 * x^3 + 2

theorem tangent_slope_angle_at_one :
  let f' : ℝ → ℝ := λ x ↦ deriv f x
  let slope : ℝ := f' 1
  let slope_angle : ℝ := Real.pi - Real.arctan slope
  slope_angle = 2 * Real.pi / 3 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_angle_at_one_l928_92816


namespace NUMINAMATH_CALUDE_nelly_painting_payment_l928_92870

/-- The amount Nelly paid for a painting at an auction, given Joe's bid and the condition of her payment. -/
theorem nelly_painting_payment (joe_bid : ℕ) (h : joe_bid = 160000) : 
  3 * joe_bid + 2000 = 482000 := by
  sorry

end NUMINAMATH_CALUDE_nelly_painting_payment_l928_92870


namespace NUMINAMATH_CALUDE_quadratic_polynomial_satisfies_conditions_l928_92892

theorem quadratic_polynomial_satisfies_conditions :
  ∃ (q : ℝ → ℝ),
    (∀ x, q x = 1.1 * x^2 - 2.1 * x + 5) ∧
    q (-1) = 4 ∧
    q 2 = 1 ∧
    q 4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_satisfies_conditions_l928_92892


namespace NUMINAMATH_CALUDE_brianna_reading_time_l928_92896

-- Define the reading speeds
def anna_speed : ℝ := 1
def carole_speed : ℝ := anna_speed
def brianna_speed : ℝ := 2 * carole_speed

-- Define the book length
def book_pages : ℕ := 100

-- Theorem to prove
theorem brianna_reading_time :
  (book_pages : ℝ) / brianna_speed = 50 := by
  sorry

end NUMINAMATH_CALUDE_brianna_reading_time_l928_92896


namespace NUMINAMATH_CALUDE_worker_earnings_l928_92847

/-- Calculate the total earnings of a worker for a week --/
theorem worker_earnings (ordinary_rate : ℚ) (overtime_rate : ℚ) 
  (total_hours : ℕ) (overtime_hours : ℕ) : 
  ordinary_rate = 60/100 →
  overtime_rate = 90/100 →
  total_hours = 50 →
  overtime_hours = 8 →
  (total_hours - overtime_hours) * ordinary_rate + 
    overtime_hours * overtime_rate = 3240/100 := by
  sorry

end NUMINAMATH_CALUDE_worker_earnings_l928_92847


namespace NUMINAMATH_CALUDE_lena_always_greater_probability_lena_greater_l928_92874

def lena_set : Finset ℕ := {7, 8, 9}
def jonah_set : Finset ℕ := {2, 4, 6}

def lena_result (a b : ℕ) : ℕ := a * b

def jonah_result (a b c : ℕ) : ℕ := (a + b) * c

theorem lena_always_greater :
  ∀ (a b : ℕ) (c d e : ℕ),
    a ∈ lena_set → b ∈ lena_set → a ≠ b →
    c ∈ jonah_set → d ∈ jonah_set → e ∈ jonah_set →
    c ≠ d → c ≠ e → d ≠ e →
    lena_result a b > jonah_result c d e :=
by
  sorry

theorem probability_lena_greater : ℚ :=
  1

#check lena_always_greater
#check probability_lena_greater

end NUMINAMATH_CALUDE_lena_always_greater_probability_lena_greater_l928_92874


namespace NUMINAMATH_CALUDE_melissa_bananas_l928_92813

/-- Calculates the number of bananas Melissa has left after sharing some. -/
def bananas_left (initial : ℕ) (shared : ℕ) : ℕ :=
  initial - shared

/-- Proves that Melissa has 84 bananas left after sharing 4 out of her initial 88 bananas. -/
theorem melissa_bananas : bananas_left 88 4 = 84 := by
  sorry

end NUMINAMATH_CALUDE_melissa_bananas_l928_92813


namespace NUMINAMATH_CALUDE_imaginary_part_of_one_plus_i_cubed_l928_92866

theorem imaginary_part_of_one_plus_i_cubed (i : ℂ) : Complex.im ((1 + i)^3) = 2 :=
by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_one_plus_i_cubed_l928_92866


namespace NUMINAMATH_CALUDE_multiplicative_inverse_154_mod_257_l928_92888

theorem multiplicative_inverse_154_mod_257 : ∃ x : ℕ, x < 257 ∧ (154 * x) % 257 = 1 :=
  by
    use 20
    sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_154_mod_257_l928_92888


namespace NUMINAMATH_CALUDE_power_multiplication_l928_92833

theorem power_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l928_92833


namespace NUMINAMATH_CALUDE_half_difference_donations_l928_92814

theorem half_difference_donations (julie_donation margo_donation : ℕ) 
  (h1 : julie_donation = 4700)
  (h2 : margo_donation = 4300) :
  (julie_donation - margo_donation) / 2 = 200 := by
  sorry

end NUMINAMATH_CALUDE_half_difference_donations_l928_92814


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l928_92809

/-- Prove that the equation (ax^2 - 2xy + y^2) - (-x^2 + bxy + 2y^2) = 5x^2 - 9xy + cy^2 
    holds true if and only if a = 4, b = 7, and c = -1 -/
theorem quadratic_equation_solution (a b c : ℝ) (x y : ℝ) :
  (a * x^2 - 2 * x * y + y^2) - (-x^2 + b * x * y + 2 * y^2) = 5 * x^2 - 9 * x * y + c * y^2 ↔ 
  a = 4 ∧ b = 7 ∧ c = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l928_92809


namespace NUMINAMATH_CALUDE_triangle_side_length_simplification_l928_92897

theorem triangle_side_length_simplification 
  (a b c : ℝ) 
  (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b) : 
  |a + b - c| - |b - a - c| = 2*b - 2*c := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_simplification_l928_92897


namespace NUMINAMATH_CALUDE_proportional_segments_l928_92844

theorem proportional_segments (a b c d : ℝ) : 
  a / b = c / d → a = 2 → b = 4 → c = 3 → d = 6 := by
  sorry

end NUMINAMATH_CALUDE_proportional_segments_l928_92844


namespace NUMINAMATH_CALUDE_transform_sin_function_l928_92832

open Real

theorem transform_sin_function (φ : ℝ) (h1 : 0 < φ) (h2 : φ < π/2) :
  let f : ℝ → ℝ := λ x ↦ 2 * sin (3*x + φ)
  let g : ℝ → ℝ := λ x ↦ 2 * sin (3*x) + 1
  (∀ x, f x = f (2*φ - x)) →  -- (φ, 0) is center of symmetry
  (∃ h : ℝ → ℝ, ∀ x, g x = h (f (x - π/12)) + 1) :=
by sorry

end NUMINAMATH_CALUDE_transform_sin_function_l928_92832


namespace NUMINAMATH_CALUDE_first_us_space_shuttle_is_columbia_l928_92857

/-- Represents a space shuttle -/
structure SpaceShuttle where
  name : String
  country : String
  year : Nat
  manned_flight_completed : Bool

/-- The world's first space shuttle developed by the United States in 1981 -/
def first_us_space_shuttle : SpaceShuttle :=
  { name := "Columbia"
  , country := "United States"
  , year := 1981
  , manned_flight_completed := true }

/-- Theorem stating that the first US space shuttle's name is Columbia -/
theorem first_us_space_shuttle_is_columbia :
  first_us_space_shuttle.name = "Columbia" :=
by sorry

end NUMINAMATH_CALUDE_first_us_space_shuttle_is_columbia_l928_92857


namespace NUMINAMATH_CALUDE_saramago_readers_ratio_l928_92849

/-- Represents the bookstore scenario with workers and their reading habits. -/
structure Bookstore where
  total_workers : ℕ
  saramago_readers : ℕ
  kureishi_readers : ℕ
  both_readers : ℕ
  neither_readers : ℕ

/-- Conditions for the Palabras bookstore scenario. -/
def palabras_conditions (b : Bookstore) : Prop :=
  b.total_workers = 42 ∧
  b.kureishi_readers = b.total_workers / 6 ∧
  b.both_readers = 3 ∧
  b.neither_readers = (b.saramago_readers - b.both_readers) - 1 ∧
  b.total_workers = (b.saramago_readers - b.both_readers) + (b.kureishi_readers - b.both_readers) + b.both_readers + b.neither_readers

/-- Theorem stating that under the given conditions, the ratio of Saramago readers to total workers is 1:2. -/
theorem saramago_readers_ratio (b : Bookstore) (h : palabras_conditions b) :
  (b.saramago_readers : ℚ) / b.total_workers = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_saramago_readers_ratio_l928_92849


namespace NUMINAMATH_CALUDE_probability_at_least_two_green_l928_92836

theorem probability_at_least_two_green (total : ℕ) (red : ℕ) (green : ℕ) (yellow : ℕ) :
  total = 10 ∧ red = 5 ∧ green = 3 ∧ yellow = 2 →
  (Nat.choose total 3 : ℚ) ≠ 0 →
  (Nat.choose green 2 * Nat.choose (total - green) 1 + Nat.choose green 3 : ℚ) / Nat.choose total 3 = 11 / 60 :=
by sorry

end NUMINAMATH_CALUDE_probability_at_least_two_green_l928_92836


namespace NUMINAMATH_CALUDE_boat_speed_theorem_l928_92853

/-- Given a boat with still water speed and downstream speed, calculate its upstream speed -/
def boat_upstream_speed (still_water_speed downstream_speed : ℝ) : ℝ :=
  2 * still_water_speed - downstream_speed

/-- Theorem: A boat with 7 km/hr still water speed and 10 km/hr downstream speed has 4 km/hr upstream speed -/
theorem boat_speed_theorem :
  boat_upstream_speed 7 10 = 4 := by
  sorry

#eval boat_upstream_speed 7 10

end NUMINAMATH_CALUDE_boat_speed_theorem_l928_92853


namespace NUMINAMATH_CALUDE_symmetric_point_of_P_l928_92884

/-- The symmetric point of P(1, 3) with respect to the line y=x is (3, 1) -/
theorem symmetric_point_of_P : ∃ (P' : ℝ × ℝ), 
  (P' = (3, 1) ∧ 
   (∀ (Q : ℝ × ℝ), Q.1 = Q.2 → (1 - Q.1)^2 + (3 - Q.2)^2 = (P'.1 - Q.1)^2 + (P'.2 - Q.2)^2)) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_point_of_P_l928_92884


namespace NUMINAMATH_CALUDE_fraction_order_l928_92802

theorem fraction_order : 
  let f1 := 18 / 14
  let f2 := 16 / 12
  let f3 := 20 / 16
  5 / 4 < f1 ∧ f1 < f2 ∧ f3 < f1 := by sorry

end NUMINAMATH_CALUDE_fraction_order_l928_92802


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l928_92856

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  (∀ n, S n = (n : ℝ) / 2 * (a 1 + a n)) →  -- sum formula
  (a 4 + a 8 = 4) →  -- given condition
  (S 11 + a 6 = 24) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l928_92856


namespace NUMINAMATH_CALUDE_friends_assignment_l928_92863

/-- The number of ways to assign friends to rooms -/
def assignFriends (n : ℕ) (m : ℕ) (maxPerRoom : ℕ) : ℕ :=
  sorry

/-- Theorem stating the correct number of assignments for the given problem -/
theorem friends_assignment :
  assignFriends 7 7 3 = 15120 := by
  sorry

end NUMINAMATH_CALUDE_friends_assignment_l928_92863


namespace NUMINAMATH_CALUDE_sector_area_l928_92867

/-- Given a circular sector where the arc length is 4 cm and the central angle is 2 radians,
    prove that the area of the sector is 4 cm². -/
theorem sector_area (s : ℝ) (θ : ℝ) (A : ℝ) : 
  s = 4 → θ = 2 → s = 2 * θ → A = (1/2) * (s/θ)^2 * θ → A = 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l928_92867


namespace NUMINAMATH_CALUDE_initial_alcohol_percentage_l928_92881

/-- Proves that the initial alcohol percentage is 5% given the problem conditions -/
theorem initial_alcohol_percentage
  (initial_volume : ℝ)
  (added_alcohol : ℝ)
  (added_water : ℝ)
  (final_percentage : ℝ)
  (h1 : initial_volume = 40)
  (h2 : added_alcohol = 5.5)
  (h3 : added_water = 4.5)
  (h4 : final_percentage = 15)
  (h5 : final_percentage / 100 * (initial_volume + added_alcohol + added_water) =
        initial_percentage / 100 * initial_volume + added_alcohol) :
  initial_percentage = 5 :=
by sorry


end NUMINAMATH_CALUDE_initial_alcohol_percentage_l928_92881


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l928_92878

theorem quadratic_equation_solution :
  ∃! y : ℝ, y^2 + 6*y + 8 = -(y + 2)*(y + 6) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l928_92878


namespace NUMINAMATH_CALUDE_max_period_linear_recurrence_l928_92826

/-- The maximum period of a second-order linear recurrence sequence modulo a prime -/
theorem max_period_linear_recurrence (p : Nat) (hp : Prime p) 
  (a b c d : Int) : ∃ (x : Nat → Int), 
  (x 0 = c) ∧ 
  (x 1 = d) ∧ 
  (∀ n, x (n + 2) = a * x (n + 1) + b * x n) ∧ 
  (∃ t, t ≤ p^2 - 1 ∧ 
    ∀ n ≥ p^2, (x (n + t) : ZMod p) = (x n : ZMod p)) ∧
  (∀ t' < p^2 - 1, ∃ n ≥ p^2, (x (n + t') : ZMod p) ≠ (x n : ZMod p)) :=
sorry

end NUMINAMATH_CALUDE_max_period_linear_recurrence_l928_92826


namespace NUMINAMATH_CALUDE_profit_percent_from_cost_price_ratio_l928_92846

/-- Calculates the profit percent given the cost price as a percentage of the selling price -/
theorem profit_percent_from_cost_price_ratio (cost_price_ratio : ℝ) :
  cost_price_ratio = 0.25 → (1 / cost_price_ratio - 1) * 100 = 300 := by
  sorry

end NUMINAMATH_CALUDE_profit_percent_from_cost_price_ratio_l928_92846


namespace NUMINAMATH_CALUDE_quadratic_two_roots_l928_92871

theorem quadratic_two_roots (a b c : ℝ) (ha : a ≠ 0) (hac : a * c < 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x = x₁ ∨ x = x₂) :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_roots_l928_92871


namespace NUMINAMATH_CALUDE_sin_double_alpha_l928_92807

theorem sin_double_alpha (α : Real) : 
  Real.sin (45 * π / 180 + α) = Real.sqrt 5 / 5 → Real.sin (2 * α) = -3 / 5 := by
sorry

end NUMINAMATH_CALUDE_sin_double_alpha_l928_92807


namespace NUMINAMATH_CALUDE_kyle_monthly_income_l928_92843

def rent : ℕ := 1250
def utilities : ℕ := 150
def retirement_savings : ℕ := 400
def groceries_eating_out : ℕ := 300
def insurance : ℕ := 200
def miscellaneous : ℕ := 200
def car_payment : ℕ := 350
def gas_maintenance : ℕ := 350

def total_expenses : ℕ := rent + utilities + retirement_savings + groceries_eating_out + insurance + miscellaneous + car_payment + gas_maintenance

theorem kyle_monthly_income : total_expenses = 3200 := by
  sorry

end NUMINAMATH_CALUDE_kyle_monthly_income_l928_92843


namespace NUMINAMATH_CALUDE_investment_dividend_income_l928_92831

/-- Calculates the annual dividend income based on investment parameters -/
def annual_dividend_income (investment : ℚ) (face_value : ℚ) (quoted_price : ℚ) (dividend_rate : ℚ) : ℚ :=
  let num_shares := investment / quoted_price
  let dividend_per_share := (dividend_rate / 100) * face_value
  num_shares * dividend_per_share

/-- Theorem stating that the annual dividend income for the given parameters is 728 -/
theorem investment_dividend_income :
  annual_dividend_income 4940 10 9.50 14 = 728 := by
  sorry

end NUMINAMATH_CALUDE_investment_dividend_income_l928_92831


namespace NUMINAMATH_CALUDE_cyclist_round_time_l928_92829

/-- Proves that a cyclist completes one round of a rectangular park in 8 minutes
    given the specified conditions. -/
theorem cyclist_round_time (length width : ℝ) (area perimeter : ℝ) (speed : ℝ) : 
  length / width = 4 →
  area = length * width →
  area = 102400 →
  perimeter = 2 * (length + width) →
  speed = 12 * 1000 / 3600 →
  (perimeter / speed) / 60 = 8 :=
by sorry

end NUMINAMATH_CALUDE_cyclist_round_time_l928_92829


namespace NUMINAMATH_CALUDE_conic_pair_eccentricity_relation_l928_92885

/-- An ellipse and a hyperbola with common foci -/
structure ConicPair where
  F₁ : ℝ × ℝ  -- Left focus
  F₂ : ℝ × ℝ  -- Right focus
  P : ℝ × ℝ   -- Intersection point
  e₁ : ℝ      -- Eccentricity of the ellipse
  e₂ : ℝ      -- Eccentricity of the hyperbola
  h₁ : e₁ > 0 -- e₁ is positive
  h₂ : e₂ > 1 -- e₂ is greater than 1

/-- The theorem to be proved -/
theorem conic_pair_eccentricity_relation (cp : ConicPair) 
  (h : (cp.P.1 - cp.F₁.1) * (cp.P.1 - cp.F₂.1) + (cp.P.2 - cp.F₁.2) * (cp.P.2 - cp.F₂.2) = 0) : 
  1 / cp.e₁^2 + 1 / cp.e₂^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_conic_pair_eccentricity_relation_l928_92885


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l928_92815

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 90 → b = 120 → c^2 = a^2 + b^2 → c = 150 := by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l928_92815


namespace NUMINAMATH_CALUDE_light_ray_reflection_l928_92806

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space using the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Function to check if a point lies on a line -/
def Point.on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The starting point A -/
def A : Point := ⟨-3, 4⟩

/-- The final point B -/
def B : Point := ⟨-2, 6⟩

/-- The equation of the light ray after reflecting off the y-axis -/
def final_ray : Line := ⟨2, 1, -2⟩

/-- Theorem stating that the final ray passes through point B and has the correct equation -/
theorem light_ray_reflection :
  (B.on_line final_ray) ∧
  (final_ray.a = 2 ∧ final_ray.b = 1 ∧ final_ray.c = -2) := by sorry

end NUMINAMATH_CALUDE_light_ray_reflection_l928_92806


namespace NUMINAMATH_CALUDE_NaCl_moles_formed_l928_92841

-- Define the chemical species as types
structure ChemicalSpecies where
  name : String

-- Define the reaction equation
structure ReactionEquation where
  reactants : List (ChemicalSpecies × ℕ)
  products : List (ChemicalSpecies × ℕ)

-- Define the available reactants
def available_reactants : List (ChemicalSpecies × ℚ) :=
  [(⟨"NaOH"⟩, 2), (⟨"Cl2"⟩, 1)]

-- Define the balanced equation
def balanced_equation : ReactionEquation :=
  { reactants := [(⟨"NaOH"⟩, 2), (⟨"Cl2"⟩, 1)]
  , products := [(⟨"NaCl"⟩, 2), (⟨"H2O"⟩, 1)] }

-- Define the function to calculate the moles of product formed
def moles_of_product_formed (product : ChemicalSpecies) (eq : ReactionEquation) (reactants : List (ChemicalSpecies × ℚ)) : ℚ :=
  sorry

-- Theorem statement
theorem NaCl_moles_formed :
  moles_of_product_formed ⟨"NaCl"⟩ balanced_equation available_reactants = 2 :=
sorry

end NUMINAMATH_CALUDE_NaCl_moles_formed_l928_92841


namespace NUMINAMATH_CALUDE_peach_tart_fraction_l928_92880

theorem peach_tart_fraction (total : ℝ) (cherry : ℝ) (blueberry : ℝ) 
  (h1 : total = 0.91)
  (h2 : cherry = 0.08)
  (h3 : blueberry = 0.75) :
  total - (cherry + blueberry) = 0.08 := by
  sorry

end NUMINAMATH_CALUDE_peach_tart_fraction_l928_92880


namespace NUMINAMATH_CALUDE_congruence_and_infinite_primes_l928_92801

theorem congruence_and_infinite_primes (p : ℕ) (hp : Prime p) (hp3 : p > 3) :
  (∃ x : ℕ, (x^2 + x + 1) % p = 0) →
  (p % 6 = 1 ∧ ∀ n : ℕ, ∃ q > n, Prime q ∧ q % 6 = 1) := by
  sorry

end NUMINAMATH_CALUDE_congruence_and_infinite_primes_l928_92801


namespace NUMINAMATH_CALUDE_bun_sets_problem_l928_92830

theorem bun_sets_problem (N : ℕ) : 
  (∃ x y u v : ℕ, 
    3 * x + 5 * y = 25 ∧ 
    3 * u + 5 * v = 35 ∧ 
    x + y = N ∧ 
    u + v = N) → 
  N = 7 := by
sorry

end NUMINAMATH_CALUDE_bun_sets_problem_l928_92830


namespace NUMINAMATH_CALUDE_cos_sixty_degrees_l928_92804

theorem cos_sixty_degrees : Real.cos (60 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sixty_degrees_l928_92804


namespace NUMINAMATH_CALUDE_solve_system_l928_92889

theorem solve_system (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (eq1 : x = 2 + 1 / y) (eq2 : y = 3 + 1 / x) :
  y = 3 / 2 + Real.sqrt 15 / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l928_92889


namespace NUMINAMATH_CALUDE_existence_of_large_solutions_l928_92834

theorem existence_of_large_solutions :
  ∃ (x y z u v : ℕ), 
    x > 2000 ∧ y > 2000 ∧ z > 2000 ∧ u > 2000 ∧ v > 2000 ∧
    x^2 + y^2 + z^2 + u^2 + v^2 = x*y*z*u*v - 65 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_large_solutions_l928_92834


namespace NUMINAMATH_CALUDE_circle_equation_characterization_l928_92882

/-- A circle with center on the x-axis, radius √2, passing through (-2, 1) -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  passes_through : ℝ × ℝ
  center_on_x_axis : center.2 = 0
  radius_is_sqrt_2 : radius = Real.sqrt 2
  passes_through_point : passes_through = (-2, 1)

/-- The equation of the circle -/
def circle_equation (c : Circle) (x y : ℝ) : Prop :=
  ((x - c.center.1) ^ 2 + y ^ 2) = c.radius ^ 2

theorem circle_equation_characterization (c : Circle) :
  ∃ a : ℝ, (a = -1 ∨ a = -3) ∧
    ∀ x y : ℝ, circle_equation c x y ↔ ((x + a) ^ 2 + y ^ 2 = 2) :=
  sorry

end NUMINAMATH_CALUDE_circle_equation_characterization_l928_92882


namespace NUMINAMATH_CALUDE_composition_equality_l928_92890

theorem composition_equality (δ φ : ℝ → ℝ) (h1 : ∀ x, δ x = 5 * x + 6) (h2 : ∀ x, φ x = 7 * x + 4) :
  (∀ x, δ (φ x) = 1) ↔ (∀ x, x = -5/7) :=
by sorry

end NUMINAMATH_CALUDE_composition_equality_l928_92890


namespace NUMINAMATH_CALUDE_expression_simplification_l928_92805

theorem expression_simplification 
  (a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (sum_zero : a + b + c = 0) :
  (a^4 * b^4 + a^4 * c^4 + b^4 * c^4) / 
  ((a^2 - b*c)^2 * (b^2 - a*c)^2 * (c^2 - a*b)^2) = 
  1 / (a^2 - b*c)^2 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l928_92805


namespace NUMINAMATH_CALUDE_complex_sum_product_nonzero_l928_92877

theorem complex_sum_product_nonzero (z₁ z₂ z₃ z₄ : ℂ) 
  (h₁ : Complex.abs z₁ = 1) (h₂ : Complex.abs z₂ = 1) 
  (h₃ : Complex.abs z₃ = 1) (h₄ : Complex.abs z₄ = 1)
  (n₁ : z₁ ≠ 1) (n₂ : z₂ ≠ 1) (n₃ : z₃ ≠ 1) (n₄ : z₄ ≠ 1) :
  3 - z₁ - z₂ - z₃ - z₄ + z₁ * z₂ * z₃ * z₄ ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_product_nonzero_l928_92877


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l928_92868

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

-- Main theorem
theorem geometric_sequence_properties (a : ℕ → ℝ) (h : is_geometric_sequence a) :
  (is_geometric_sequence (fun n => |a n|)) ∧
  (is_geometric_sequence (fun n => a n * a (n + 1))) ∧
  (is_geometric_sequence (fun n => 1 / a n)) ∧
  ¬(∀ (a : ℕ → ℝ), is_geometric_sequence a → is_geometric_sequence (fun n => Real.log (a n ^ 2))) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l928_92868


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l928_92800

theorem quadratic_roots_problem (k : ℝ) (x₁ x₂ : ℝ) :
  (∀ x, x^2 - 2*(k+1)*x + k^2 + 3 = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ ≠ x₂ →
  1/x₁ + 1/x₂ = 6/7 →
  k = 2 ∧ x₁^2 + x₂^2 > 8 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l928_92800


namespace NUMINAMATH_CALUDE_mushroom_soup_total_l928_92822

theorem mushroom_soup_total (team1 team2 team3 : ℕ) 
  (h1 : team1 = 90) 
  (h2 : team2 = 120) 
  (h3 : team3 = 70) : 
  team1 + team2 + team3 = 280 := by
  sorry

end NUMINAMATH_CALUDE_mushroom_soup_total_l928_92822


namespace NUMINAMATH_CALUDE_original_price_calculation_l928_92852

theorem original_price_calculation (initial_price : ℚ) : 
  (initial_price * (1 + 20 / 100) * (1 - 10 / 100) = 2) → 
  (initial_price = 100 / 54) := by
sorry

end NUMINAMATH_CALUDE_original_price_calculation_l928_92852


namespace NUMINAMATH_CALUDE_inequality_proof_l928_92883

theorem inequality_proof (n : ℕ) (x y z : ℝ) 
  (h_n : n ≥ 3) 
  (h_x : x > 0) (h_y : y > 0) (h_z : z > 0)
  (h_sum : x + y + z = 1) :
  (1 / x^(n-1) - x) * (1 / y^(n-1) - y) * (1 / z^(n-1) - z) ≥ ((3^n - 1) / 3)^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l928_92883


namespace NUMINAMATH_CALUDE_kylies_coins_l928_92854

/-- Kylie's coin collection problem -/
theorem kylies_coins (coins_from_piggy_bank coins_from_father coins_to_laura coins_left : ℕ) 
  (h1 : coins_from_piggy_bank = 15)
  (h2 : coins_from_father = 8)
  (h3 : coins_to_laura = 21)
  (h4 : coins_left = 15) :
  coins_from_piggy_bank + coins_from_father + coins_to_laura - coins_left = 13 := by
  sorry

#check kylies_coins

end NUMINAMATH_CALUDE_kylies_coins_l928_92854


namespace NUMINAMATH_CALUDE_chess_tournament_games_l928_92879

/-- The number of games played in a chess tournament with n participants,
    where each participant plays exactly one game with each of the others. -/
def tournament_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a chess tournament with 19 participants, where each participant
    plays exactly one game with each of the remaining participants,
    the total number of games played is 171. -/
theorem chess_tournament_games :
  tournament_games 19 = 171 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l928_92879


namespace NUMINAMATH_CALUDE_equation_solution_l928_92886

theorem equation_solution : 
  ∃ x : ℝ, (225 - 4209520 / ((1000795 + (250 + x) * 50) / 27)) = 113 ∧ x = 40 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l928_92886


namespace NUMINAMATH_CALUDE_expression_value_l928_92858

theorem expression_value : 
  let x : ℕ := 3
  x + x * (x ^ (x + 1)) = 246 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l928_92858


namespace NUMINAMATH_CALUDE_john_sells_20_woodburnings_l928_92855

/-- The number of woodburnings John sells -/
def num_woodburnings : ℕ := 20

/-- The selling price of each woodburning in dollars -/
def selling_price : ℕ := 15

/-- The cost of wood in dollars -/
def wood_cost : ℕ := 100

/-- John's profit in dollars -/
def profit : ℕ := 200

/-- Theorem stating that the number of woodburnings John sells is 20 -/
theorem john_sells_20_woodburnings :
  num_woodburnings = 20 ∧
  selling_price * num_woodburnings = wood_cost + profit := by
  sorry

end NUMINAMATH_CALUDE_john_sells_20_woodburnings_l928_92855


namespace NUMINAMATH_CALUDE_equation_solution_l928_92823

theorem equation_solution (y : ℝ) (h : y ≠ 0) : 
  (2 / y + (3 / y) / (6 / y) = 1.5) → y = 2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l928_92823


namespace NUMINAMATH_CALUDE_tournament_games_count_l928_92825

/-- Calculates the number of games in a single-elimination tournament. -/
def gamesInSingleElimination (n : ℕ) : ℕ := n - 1

/-- Represents the tournament structure and calculates the total number of games. -/
def totalGames (interestedTeams preliminaryTeams mainTournamentTeams : ℕ) : ℕ :=
  let preliminaryGames := gamesInSingleElimination preliminaryTeams
  let mainTournamentGames := gamesInSingleElimination mainTournamentTeams
  preliminaryGames + mainTournamentGames

/-- Theorem stating that the total number of games in the described tournament is 23. -/
theorem tournament_games_count :
  totalGames 25 9 16 = 23 := by
  sorry

#eval totalGames 25 9 16

end NUMINAMATH_CALUDE_tournament_games_count_l928_92825


namespace NUMINAMATH_CALUDE_unique_intersection_l928_92850

/-- The value of k for which the line x = k intersects the parabola x = -y^2 - 4y + 2 at exactly one point -/
def intersection_k : ℝ := 6

/-- The parabola equation -/
def parabola (y : ℝ) : ℝ := -y^2 - 4*y + 2

theorem unique_intersection :
  ∀ k : ℝ, (∃! y : ℝ, k = parabola y) ↔ k = intersection_k :=
by sorry

end NUMINAMATH_CALUDE_unique_intersection_l928_92850


namespace NUMINAMATH_CALUDE_f_9_eq_two_thirds_l928_92851

/-- A function satisfying the given conditions -/
def f (x : ℝ) : ℝ := sorry

/-- f is odd -/
axiom f_odd : ∀ x, f (-x) = -f x

/-- f(x-2) = f(x+2) for all x -/
axiom f_period : ∀ x, f (x - 2) = f (x + 2)

/-- f(x) = 3^x - 1 for x in [-2,0] -/
axiom f_def : ∀ x, -2 ≤ x ∧ x ≤ 0 → f x = 3^x - 1

/-- The main theorem: f(9) = 2/3 -/
theorem f_9_eq_two_thirds : f 9 = 2/3 := by sorry

end NUMINAMATH_CALUDE_f_9_eq_two_thirds_l928_92851


namespace NUMINAMATH_CALUDE_path_count_through_checkpoint_l928_92821

def grid_paths (right1 right2 up1 up2 : ℕ) : ℕ :=
  (Nat.choose (right1 + up1) up1) * (Nat.choose (right2 + up2) up2)

theorem path_count_through_checkpoint :
  grid_paths 3 2 2 2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_path_count_through_checkpoint_l928_92821


namespace NUMINAMATH_CALUDE_ballpoint_pen_price_l928_92827

-- Define the problem parameters
def total_pens : Nat := 30
def total_pencils : Nat := 75
def total_cost : ℝ := 690

def gel_pens : Nat := 20
def ballpoint_pens : Nat := 10
def standard_pencils : Nat := 50
def mechanical_pencils : Nat := 25

def avg_price_gel : ℝ := 1.5
def avg_price_mechanical : ℝ := 3
def avg_price_standard : ℝ := 2

-- Theorem to prove
theorem ballpoint_pen_price :
  ∃ (avg_price_ballpoint : ℝ),
    avg_price_ballpoint = 48.5 ∧
    total_cost = 
      gel_pens * avg_price_gel +
      mechanical_pencils * avg_price_mechanical +
      standard_pencils * avg_price_standard +
      ballpoint_pens * avg_price_ballpoint :=
by sorry

end NUMINAMATH_CALUDE_ballpoint_pen_price_l928_92827


namespace NUMINAMATH_CALUDE_polynomial_roots_sum_property_l928_92840

theorem polynomial_roots_sum_property (x₁ x₂ : ℝ) (h₁ : x₁^2 - 6*x₁ + 1 = 0) (h₂ : x₂^2 - 6*x₂ + 1 = 0) :
  ∀ n : ℕ, ∃ k : ℤ, (x₁^n + x₂^n = k) ∧ ¬(5 ∣ k) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_roots_sum_property_l928_92840


namespace NUMINAMATH_CALUDE_intersection_distance_difference_l928_92808

/-- The line y - 2x - 1 = 0 intersects the parabola y^2 = 4x + 1 at points C and D.
    Q is the point (2, 0). -/
theorem intersection_distance_difference (C D Q : ℝ × ℝ) : 
  (C.2 - 2 * C.1 - 1 = 0) →
  (D.2 - 2 * D.1 - 1 = 0) →
  (C.2^2 = 4 * C.1 + 1) →
  (D.2^2 = 4 * D.1 + 1) →
  (Q = (2, 0)) →
  |Real.sqrt ((C.1 - Q.1)^2 + (C.2 - Q.2)^2) - Real.sqrt ((D.1 - Q.1)^2 + (D.2 - Q.2)^2)| = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_distance_difference_l928_92808


namespace NUMINAMATH_CALUDE_complex_square_equality_l928_92818

theorem complex_square_equality (a b : ℝ) : 
  (a + Complex.I = 2 - b * Complex.I) → (a + b * Complex.I)^2 = 3 - 4 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_square_equality_l928_92818


namespace NUMINAMATH_CALUDE_independence_test_most_appropriate_l928_92848

/-- Represents the methods available for analysis -/
inductive AnalysisMethod
  | Mean
  | Regression
  | IndependenceTest
  | Probability

/-- Represents the data from the survey -/
structure SurveyData where
  carOwners : Nat
  carOwnersOpposed : Nat
  nonCarOwners : Nat
  nonCarOwnersOpposed : Nat

/-- Determines the most appropriate method for analyzing the relationship
    between car ownership and policy opposition given survey data -/
def mostAppropriateMethod (data : SurveyData) : AnalysisMethod :=
  AnalysisMethod.IndependenceTest

/-- Theorem stating that the Independence test is the most appropriate method
    for analyzing the relationship between car ownership and policy opposition -/
theorem independence_test_most_appropriate (data : SurveyData) :
  mostAppropriateMethod data = AnalysisMethod.IndependenceTest :=
by sorry

end NUMINAMATH_CALUDE_independence_test_most_appropriate_l928_92848


namespace NUMINAMATH_CALUDE_complex_subtraction_l928_92810

theorem complex_subtraction (c d : ℂ) (hc : c = 5 - 3*I) (hd : d = 2 + 4*I) :
  c - 3*d = -1 - 15*I := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_l928_92810


namespace NUMINAMATH_CALUDE_negation_equivalence_l928_92835

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - x + 2 > 0) ↔ (∀ x : ℝ, x^2 - x + 2 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l928_92835


namespace NUMINAMATH_CALUDE_regression_line_intercept_l928_92824

/-- Represents a linear regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- The regression line passes through a given point -/
def passes_through (line : RegressionLine) (x y : ℝ) : Prop :=
  y = line.slope * x + line.intercept

theorem regression_line_intercept (b : ℝ) (x₀ y₀ : ℝ) :
  let line := RegressionLine.mk b ((y₀ : ℝ) - b * x₀)
  passes_through line x₀ y₀ ∧ line.slope = 1.23 ∧ x₀ = 4 ∧ y₀ = 5 →
  line.intercept = 0.08 := by
  sorry

end NUMINAMATH_CALUDE_regression_line_intercept_l928_92824


namespace NUMINAMATH_CALUDE_zach_stadium_goal_years_l928_92860

/-- The number of years required to save enough money to visit all major league baseball stadiums. -/
def years_to_visit_stadiums (num_stadiums : ℕ) (cost_per_stadium : ℕ) (annual_savings : ℕ) : ℕ :=
  (num_stadiums * cost_per_stadium) / annual_savings

/-- Theorem stating that it takes 18 years to save enough money to visit all 30 major league baseball stadiums
    given an average cost of $900 per stadium and annual savings of $1,500. -/
theorem zach_stadium_goal_years :
  years_to_visit_stadiums 30 900 1500 = 18 := by
  sorry

end NUMINAMATH_CALUDE_zach_stadium_goal_years_l928_92860


namespace NUMINAMATH_CALUDE_hockey_league_games_l928_92845

theorem hockey_league_games (n : ℕ) (m : ℕ) (total_games : ℕ) 
  (h1 : n = 25)  -- number of teams
  (h2 : m = 15)  -- number of times each pair of teams face each other
  : total_games = 4500 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_hockey_league_games_l928_92845


namespace NUMINAMATH_CALUDE_negation_divisible_by_two_even_l928_92820

theorem negation_divisible_by_two_even :
  (¬ ∀ n : ℤ, 2 ∣ n → Even n) ↔ (∃ n : ℤ, 2 ∣ n ∧ ¬ Even n) :=
by sorry

end NUMINAMATH_CALUDE_negation_divisible_by_two_even_l928_92820


namespace NUMINAMATH_CALUDE_valid_basis_vectors_l928_92839

def vector_a : Fin 2 → ℝ := ![3, 4]

def vector_e1 : Fin 2 → ℝ := ![-1, 2]
def vector_e2 : Fin 2 → ℝ := ![3, -1]

theorem valid_basis_vectors :
  ∃ (x y : ℝ), vector_a = x • vector_e1 + y • vector_e2 ∧
  ¬(∃ (k : ℝ), vector_e1 = k • vector_e2) :=
by sorry

end NUMINAMATH_CALUDE_valid_basis_vectors_l928_92839


namespace NUMINAMATH_CALUDE_margaret_swimming_time_l928_92876

/-- Billy's swimming times for different parts of the race in seconds -/
def billy_times : List ℕ := [120, 240, 60, 150]

/-- The time difference between Billy and Margaret in seconds -/
def time_difference : ℕ := 30

/-- Calculate the total time Billy spent swimming -/
def billy_total_time : ℕ := billy_times.sum

/-- Calculate Margaret's total swimming time in seconds -/
def margaret_time_seconds : ℕ := billy_total_time + time_difference

/-- Convert seconds to minutes -/
def seconds_to_minutes (seconds : ℕ) : ℕ := seconds / 60

theorem margaret_swimming_time :
  seconds_to_minutes margaret_time_seconds = 10 := by
  sorry

end NUMINAMATH_CALUDE_margaret_swimming_time_l928_92876
