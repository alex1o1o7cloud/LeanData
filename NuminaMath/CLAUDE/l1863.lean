import Mathlib

namespace NUMINAMATH_CALUDE_shoe_repair_time_calculation_l1863_186324

/-- Given the total time spent on repairing shoes and the time required to replace buckles,
    calculate the time needed to even out the heel for each shoe. -/
theorem shoe_repair_time_calculation 
  (total_time : ℝ)
  (buckle_time : ℝ)
  (h_total : total_time = 30)
  (h_buckle : buckle_time = 5)
  : (total_time - buckle_time) / 2 = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_shoe_repair_time_calculation_l1863_186324


namespace NUMINAMATH_CALUDE_zoo_field_trip_l1863_186336

theorem zoo_field_trip (students : ℕ) (adults : ℕ) (vans : ℕ) : 
  students = 12 → adults = 3 → vans = 3 → (students + adults) / vans = 5 := by
  sorry

end NUMINAMATH_CALUDE_zoo_field_trip_l1863_186336


namespace NUMINAMATH_CALUDE_odd_function_properties_l1863_186339

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 2 / (2^x + 1)

theorem odd_function_properties (a : ℝ) :
  (∀ x, f a x = -f a (-x)) →
  (a = 1 ∧
   ∀ x, f a x = 1 - 2 / (2^x + 1) ∧
   StrictMono (f a)) :=
by sorry

end NUMINAMATH_CALUDE_odd_function_properties_l1863_186339


namespace NUMINAMATH_CALUDE_notebook_purchase_savings_l1863_186386

theorem notebook_purchase_savings (s : ℚ) (n : ℚ) (p : ℚ) 
  (h1 : s > 0) (h2 : n > 0) (h3 : p > 0) 
  (h4 : (1/4) * s = (1/2) * n * p) : 
  s - n * p = (1/2) * s := by
sorry

end NUMINAMATH_CALUDE_notebook_purchase_savings_l1863_186386


namespace NUMINAMATH_CALUDE_series_sum_equals_first_term_l1863_186304

def decreasing_to_zero (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∀ n, a n ≥ a (n + 1)) ∧ (∀ ε > 0, ∃ N, ∀ n ≥ N, a n < ε)

def b (a : ℕ → ℝ) (n : ℕ) : ℝ := a n - 2 * a (n + 1) + a (n + 2)

theorem series_sum_equals_first_term (a : ℕ → ℝ) :
  decreasing_to_zero a →
  (∀ n, b a n ≥ 0) →
  (∑' n, n * b a n) = a 1 :=
sorry

end NUMINAMATH_CALUDE_series_sum_equals_first_term_l1863_186304


namespace NUMINAMATH_CALUDE_freshman_class_size_l1863_186323

theorem freshman_class_size : ∃! n : ℕ, n < 500 ∧ n % 23 = 22 ∧ n % 21 = 14 ∧ n = 413 := by
  sorry

end NUMINAMATH_CALUDE_freshman_class_size_l1863_186323


namespace NUMINAMATH_CALUDE_count_valid_n_l1863_186387

def is_perfect_square (x : ℕ) : Prop := ∃ y : ℕ, y * y = x

def valid_n (n : ℕ) : Prop :=
  n > 0 ∧
  is_perfect_square (1 * 4 + 2112) ∧
  is_perfect_square (1 * n + 2112) ∧
  is_perfect_square (4 * n + 2112)

theorem count_valid_n :
  ∃ (S : Finset ℕ), (∀ n ∈ S, valid_n n) ∧ S.card = 7 ∧ (∀ n, valid_n n → n ∈ S) :=
sorry

end NUMINAMATH_CALUDE_count_valid_n_l1863_186387


namespace NUMINAMATH_CALUDE_sector_area_l1863_186314

/-- Theorem: Area of a circular sector with central angle 2π/3 and arc length 2 --/
theorem sector_area (r : ℝ) (h1 : (2 * π / 3) * r = 2) : 
  (1 / 2) * r^2 * (2 * π / 3) = 3 / π := by
  sorry


end NUMINAMATH_CALUDE_sector_area_l1863_186314


namespace NUMINAMATH_CALUDE_lara_has_largest_result_l1863_186370

def starting_number : ℕ := 12

def john_result : ℕ := ((starting_number + 3) * 2) - 4
def lara_result : ℕ := (starting_number * 3 + 5) - 6
def miguel_result : ℕ := (starting_number * 2 - 2) + 2

theorem lara_has_largest_result :
  lara_result > john_result ∧ lara_result > miguel_result := by
  sorry

end NUMINAMATH_CALUDE_lara_has_largest_result_l1863_186370


namespace NUMINAMATH_CALUDE_complex_addition_simplification_l1863_186365

theorem complex_addition_simplification :
  (-5 : ℂ) + 3*I + (2 : ℂ) - 7*I = -3 - 4*I :=
by sorry

end NUMINAMATH_CALUDE_complex_addition_simplification_l1863_186365


namespace NUMINAMATH_CALUDE_no_rational_solution_to_5x2_plus_3y2_eq_1_l1863_186352

theorem no_rational_solution_to_5x2_plus_3y2_eq_1 :
  ¬ ∃ (x y : ℚ), 5 * x^2 + 3 * y^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_no_rational_solution_to_5x2_plus_3y2_eq_1_l1863_186352


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l1863_186340

theorem fractional_equation_solution :
  ∃ x : ℝ, (x * (x - 2) ≠ 0) ∧ (4 / (x - 2) = 2 / x) ∧ (x = -2) := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l1863_186340


namespace NUMINAMATH_CALUDE_framed_painting_ratio_l1863_186385

theorem framed_painting_ratio : 
  let painting_width : ℝ := 28
  let painting_height : ℝ := 32
  let frame_side_width : ℝ := 10/3
  let frame_top_bottom_width : ℝ := 3 * frame_side_width
  let framed_width : ℝ := painting_width + 2 * frame_side_width
  let framed_height : ℝ := painting_height + 2 * frame_top_bottom_width
  let frame_area : ℝ := framed_width * framed_height - painting_width * painting_height
  frame_area = painting_width * painting_height →
  framed_width / framed_height = 26 / 35 :=
by
  sorry

end NUMINAMATH_CALUDE_framed_painting_ratio_l1863_186385


namespace NUMINAMATH_CALUDE_hyperbola_parabola_relation_l1863_186341

theorem hyperbola_parabola_relation (a b p : ℝ) (ha : a > 0) (hb : b > 0) (hp : p > 0) :
  (∃ (c : ℝ), c = 2 * a ∧ c^2 = a^2 + b^2) →
  (2 = (p / 2 / b) / Real.sqrt ((1 / a^2) + (1 / b^2))) →
  p = 8 := by sorry

end NUMINAMATH_CALUDE_hyperbola_parabola_relation_l1863_186341


namespace NUMINAMATH_CALUDE_purely_imaginary_iff_a_eq_two_l1863_186349

/-- For a complex number z = (a^2 - 4) + (a + 2)i where a is real,
    z is purely imaginary if and only if a = 2 -/
theorem purely_imaginary_iff_a_eq_two (a : ℝ) :
  let z : ℂ := (a^2 - 4) + (a + 2)*I
  (z.re = 0 ∧ z.im ≠ 0) ↔ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_iff_a_eq_two_l1863_186349


namespace NUMINAMATH_CALUDE_rotation_of_P_l1863_186383

/-- Rotate a point 180 degrees counterclockwise about the origin -/
def rotate180 (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

theorem rotation_of_P :
  let P : ℝ × ℝ := (-3, 2)
  rotate180 P = (3, -2) := by
sorry

end NUMINAMATH_CALUDE_rotation_of_P_l1863_186383


namespace NUMINAMATH_CALUDE_roots_sum_reciprocal_l1863_186309

theorem roots_sum_reciprocal (x₁ x₂ : ℝ) : 
  x₁^2 + 8*x₁ + 4 = 0 → x₂^2 + 8*x₂ + 4 = 0 → x₁ ≠ x₂ → 
  (1 / x₁) + (1 / x₂) = -2 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_reciprocal_l1863_186309


namespace NUMINAMATH_CALUDE_triangle_properties_l1863_186326

theorem triangle_properties (A B C : Real) (a : Real × Real) :
  -- A, B, C are angles of a triangle
  A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π →
  -- Definition of vector a
  a = (Real.sqrt 2 * Real.cos ((A + B) / 2), Real.sin ((A - B) / 2)) →
  -- Magnitude of a
  Real.sqrt (a.1^2 + a.2^2) = Real.sqrt 6 / 2 →
  -- Conclusions
  (Real.tan A * Real.tan B = 1 / 3) ∧
  (∀ C', C' = π - A - B → Real.tan C' ≤ -Real.sqrt 3) ∧
  (∃ C', C' = π - A - B ∧ Real.tan C' = -Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1863_186326


namespace NUMINAMATH_CALUDE_square_difference_of_integers_l1863_186338

theorem square_difference_of_integers (a b : ℤ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 60) (h4 : a - b = 16) : 
  a^2 - b^2 = 960 := by
sorry

end NUMINAMATH_CALUDE_square_difference_of_integers_l1863_186338


namespace NUMINAMATH_CALUDE_two_digit_subtraction_l1863_186381

/-- Given two different natural numbers A and B that satisfy the two-digit subtraction equation 6A - B2 = 36, prove that A - B = 5 -/
theorem two_digit_subtraction (A B : ℕ) (h1 : A ≠ B) (h2 : 10 ≤ A) (h3 : A < 100) (h4 : 10 ≤ B) (h5 : B < 100) (h6 : 60 + A - (10 * B + 2) = 36) : A - B = 5 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_subtraction_l1863_186381


namespace NUMINAMATH_CALUDE_max_cars_in_parking_lot_l1863_186391

/-- Represents a parking lot configuration --/
structure ParkingLot :=
  (grid : Fin 7 → Fin 7 → Bool)
  (gate : Fin 7 × Fin 7)

/-- Checks if a car can exit from its position --/
def canExit (lot : ParkingLot) (pos : Fin 7 × Fin 7) : Prop :=
  sorry

/-- Counts the number of cars in the parking lot --/
def carCount (lot : ParkingLot) : Nat :=
  sorry

/-- Checks if the parking lot configuration is valid --/
def isValidConfig (lot : ParkingLot) : Prop :=
  ∀ pos, lot.grid pos.1 pos.2 → canExit lot pos

/-- The main theorem --/
theorem max_cars_in_parking_lot :
  ∃ (lot : ParkingLot),
    isValidConfig lot ∧
    carCount lot = 28 ∧
    ∀ (other : ParkingLot), isValidConfig other → carCount other ≤ 28 :=
  sorry

end NUMINAMATH_CALUDE_max_cars_in_parking_lot_l1863_186391


namespace NUMINAMATH_CALUDE_officer_selection_theorem_l1863_186355

/-- Represents a club with members of two genders -/
structure Club where
  total_members : ℕ
  boys : ℕ
  girls : ℕ

/-- Calculates the number of ways to choose officers from a single gender -/
def waysToChooseOfficers (n : ℕ) : ℕ := n * (n - 1) * (n - 2)

/-- Calculates the total number of ways to choose officers in the club -/
def totalWaysToChooseOfficers (club : Club) : ℕ :=
  waysToChooseOfficers club.boys + waysToChooseOfficers club.girls

/-- The main theorem stating the number of ways to choose officers -/
theorem officer_selection_theorem (club : Club) 
    (h1 : club.total_members = 30)
    (h2 : club.boys = 18)
    (h3 : club.girls = 12) :
    totalWaysToChooseOfficers club = 6216 := by
  sorry


end NUMINAMATH_CALUDE_officer_selection_theorem_l1863_186355


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1863_186361

theorem sum_of_roots_quadratic (x : ℝ) : 
  (∃ r1 r2 : ℝ, r1 + r2 = 5 ∧ x^2 - 5*x + 6 = (x - r1) * (x - r2)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1863_186361


namespace NUMINAMATH_CALUDE_largest_integer_not_exceeding_a_n_l1863_186374

/-- Sequence a_n defined recursively -/
def a (a₀ : ℕ) : ℕ → ℚ
  | 0 => a₀
  | n + 1 => (a a₀ n)^2 / ((a a₀ n) + 1)

/-- Theorem stating the largest integer not exceeding a_n is a - n -/
theorem largest_integer_not_exceeding_a_n (a₀ : ℕ) (n : ℕ) 
  (h : n ≤ a₀/2 + 1) : 
  ⌊a a₀ n⌋ = a₀ - n := by sorry

end NUMINAMATH_CALUDE_largest_integer_not_exceeding_a_n_l1863_186374


namespace NUMINAMATH_CALUDE_second_division_percentage_l1863_186363

/-- Proves that the percentage of students who got second division is 54% -/
theorem second_division_percentage
  (total_students : ℕ)
  (first_division_percentage : ℚ)
  (just_passed : ℕ)
  (h_total : total_students = 300)
  (h_first : first_division_percentage = 26 / 100)
  (h_passed : just_passed = 60)
  (h_all_passed : total_students = 
    (first_division_percentage * total_students).floor + 
    (total_students - (first_division_percentage * total_students).floor - just_passed) + 
    just_passed) :
  (total_students - (first_division_percentage * total_students).floor - just_passed : ℚ) / 
  total_students * 100 = 54 := by
  sorry

end NUMINAMATH_CALUDE_second_division_percentage_l1863_186363


namespace NUMINAMATH_CALUDE_exam_scores_l1863_186376

theorem exam_scores (total_items : Nat) (lowella_percentage : Nat) (pamela_increase : Nat) :
  total_items = 100 →
  lowella_percentage = 35 →
  pamela_increase = 20 →
  let lowella_score := total_items * lowella_percentage / 100
  let pamela_score := lowella_score + lowella_score * pamela_increase / 100
  let mandy_score := 2 * pamela_score
  mandy_score = 84 := by sorry

end NUMINAMATH_CALUDE_exam_scores_l1863_186376


namespace NUMINAMATH_CALUDE_vertex_on_x_axis_l1863_186322

/-- The parabola equation -/
def parabola (x d : ℝ) : ℝ := x^2 - 6*x + d

/-- The x-coordinate of the vertex -/
def vertex_x : ℝ := 3

/-- The y-coordinate of the vertex -/
def vertex_y (d : ℝ) : ℝ := parabola vertex_x d

theorem vertex_on_x_axis (d : ℝ) : vertex_y d = 0 ↔ d = 9 := by sorry

end NUMINAMATH_CALUDE_vertex_on_x_axis_l1863_186322


namespace NUMINAMATH_CALUDE_constant_term_expansion_l1863_186366

def binomial_coeff (n k : ℕ) : ℕ := sorry

def general_term (r : ℕ) : ℤ :=
  (binomial_coeff 5 r : ℤ) * (-1)^r

theorem constant_term_expansion :
  (general_term 1) + (general_term 3) + (general_term 5) = -51 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l1863_186366


namespace NUMINAMATH_CALUDE_notebook_cost_l1863_186396

/-- Given a notebook and its cover with a total pre-tax cost of $3.00,
    where the notebook costs $2 more than its cover,
    prove that the pre-tax cost of the notebook is $2.50. -/
theorem notebook_cost (notebook_cost cover_cost : ℝ) : 
  notebook_cost + cover_cost = 3 →
  notebook_cost = cover_cost + 2 →
  notebook_cost = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_notebook_cost_l1863_186396


namespace NUMINAMATH_CALUDE_expand_and_simplify_polynomial_l1863_186335

theorem expand_and_simplify_polynomial (a : ℝ) : 
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) = 
  a^5 + 19*a^4 + 137*a^3 + 461*a^2 + 702*a + 360 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_polynomial_l1863_186335


namespace NUMINAMATH_CALUDE_inscribed_circles_radius_l1863_186399

/-- Given a circle segment with radius R and central angle α, 
    this theorem proves the radius of two equal inscribed circles 
    that touch each other, the arc, and the chord. -/
theorem inscribed_circles_radius 
  (R : ℝ) 
  (α : ℝ) 
  (h_α_pos : 0 < α) 
  (h_α_lt_pi : α < π) : 
  ∃ x : ℝ, 
    x = R * Real.sin (α / 4) ^ 2 ∧ 
    x > 0 ∧
    (∀ y : ℝ, y = R * Real.sin (α / 4) ^ 2 → y = x) :=
sorry

end NUMINAMATH_CALUDE_inscribed_circles_radius_l1863_186399


namespace NUMINAMATH_CALUDE_class_average_score_l1863_186305

theorem class_average_score (total_students : ℕ) 
  (assigned_day_percentage : ℚ) (makeup_day_percentage : ℚ)
  (assigned_day_average : ℚ) (makeup_day_average : ℚ) :
  total_students = 100 →
  assigned_day_percentage = 70 / 100 →
  makeup_day_percentage = 30 / 100 →
  assigned_day_average = 65 / 100 →
  makeup_day_average = 95 / 100 →
  (assigned_day_percentage * assigned_day_average + 
   makeup_day_percentage * makeup_day_average) = 74 / 100 := by
sorry

end NUMINAMATH_CALUDE_class_average_score_l1863_186305


namespace NUMINAMATH_CALUDE_factorization_equality_l1863_186388

theorem factorization_equality (x : ℝ) : 
  (x - 3) * (x - 1) * (x - 2) * (x + 4) + 24 = (x - 2) * (x + 3) * (x^2 + x - 8) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1863_186388


namespace NUMINAMATH_CALUDE_sunzi_wood_measurement_l1863_186377

theorem sunzi_wood_measurement 
  (x y : ℝ) 
  (h1 : y - x = 4.5) 
  (h2 : (1/2) * y < x) 
  (h3 : x < (1/2) * y + 1) : 
  y - x = 4.5 ∧ (1/2) * y = x - 1 := by
  sorry

end NUMINAMATH_CALUDE_sunzi_wood_measurement_l1863_186377


namespace NUMINAMATH_CALUDE_max_profit_l1863_186332

/-- Profit function for price greater than 120 yuan -/
def profit_above (x : ℝ) : ℝ := -10 * x^2 + 2500 * x - 150000

/-- Profit function for price between 100 and 120 yuan -/
def profit_below (x : ℝ) : ℝ := -30 * x^2 + 6900 * x - 390000

/-- The maximum profit occurs at 115 yuan and equals 6750 yuan -/
theorem max_profit :
  ∃ (x : ℝ), x = 115 ∧ 
  profit_below x = 6750 ∧
  ∀ (y : ℝ), y > 100 → profit_above y ≤ profit_below x ∧ profit_below y ≤ profit_below x :=
sorry

end NUMINAMATH_CALUDE_max_profit_l1863_186332


namespace NUMINAMATH_CALUDE_nine_b_value_l1863_186354

theorem nine_b_value (a b : ℚ) (h1 : 8 * a + 3 * b = 0) (h2 : b - 3 = a) :
  9 * b = 216 / 11 := by
  sorry

end NUMINAMATH_CALUDE_nine_b_value_l1863_186354


namespace NUMINAMATH_CALUDE_gcd_lcm_product_l1863_186368

theorem gcd_lcm_product (a b : ℕ) (h : a = 140 ∧ b = 175) : 
  (Nat.gcd a b) * (Nat.lcm a b) = 24500 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_l1863_186368


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1863_186333

/-- An arithmetic sequence with its partial sums -/
structure ArithmeticSequence where
  S : ℕ → ℝ  -- S_n is the sum of the first n terms

/-- Given an arithmetic sequence with S_10 = 10 and S_20 = 40, prove S_30 = 90 -/
theorem arithmetic_sequence_sum (a : ArithmeticSequence) 
  (h1 : a.S 10 = 10) (h2 : a.S 20 = 40) : a.S 30 = 90 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1863_186333


namespace NUMINAMATH_CALUDE_albert_betty_age_ratio_l1863_186316

/-- Given the ages of Albert, Mary, and Betty, prove that the ratio of Albert's age to Betty's age is 4:1 -/
theorem albert_betty_age_ratio :
  ∀ (albert mary betty : ℕ),
  albert = 2 * mary →
  mary = albert - 8 →
  betty = 4 →
  (albert : ℚ) / betty = 4 / 1 := by
sorry

end NUMINAMATH_CALUDE_albert_betty_age_ratio_l1863_186316


namespace NUMINAMATH_CALUDE_alcohol_in_mixture_l1863_186359

/-- Proves that the amount of alcohol in a mixture is 7.5 liters given specific conditions -/
theorem alcohol_in_mixture :
  ∀ (A W : ℝ), 
    (A / W = 4 / 3) →  -- Initial ratio of alcohol to water
    (A / (W + 5) = 4 / 5) →  -- Ratio after adding 5 liters of water
    A = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_alcohol_in_mixture_l1863_186359


namespace NUMINAMATH_CALUDE_base7_351_to_base6_l1863_186350

/-- Converts a number from base 7 to base 10 -/
def base7ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 6 -/
def base10ToBase6 (n : ℕ) : ℕ := sorry

theorem base7_351_to_base6 :
  base10ToBase6 (base7ToBase10 351) = 503 := by sorry

end NUMINAMATH_CALUDE_base7_351_to_base6_l1863_186350


namespace NUMINAMATH_CALUDE_square_sum_identity_l1863_186325

theorem square_sum_identity (x : ℝ) : (x + 2)^2 + 2*(x + 2)*(5 - x) + (5 - x)^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_identity_l1863_186325


namespace NUMINAMATH_CALUDE_chocolate_box_problem_l1863_186360

theorem chocolate_box_problem (day1 day2 day3 day4 remaining : ℕ) :
  day1 = 4 →
  day2 = 2 * day1 - 3 →
  day3 = day1 - 2 →
  day4 = day3 - 1 →
  remaining = 12 →
  day1 + day2 + day3 + day4 + remaining = 24 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_box_problem_l1863_186360


namespace NUMINAMATH_CALUDE_group_size_proof_l1863_186358

theorem group_size_proof (n : ℕ) (k : ℕ) : 
  k = 7 → 
  (n : ℚ) - k ≠ 0 → 
  ((n - k) / n - k / n : ℚ) = 0.30000000000000004 → 
  n = 20 := by
sorry

end NUMINAMATH_CALUDE_group_size_proof_l1863_186358


namespace NUMINAMATH_CALUDE_volume_cube_inscribed_sphere_l1863_186310

/-- The volume of a cube inscribed in a sphere -/
theorem volume_cube_inscribed_sphere (R : ℝ) (h : R > 0) :
  ∃ (V : ℝ), V = (8 / 9) * Real.sqrt 3 * R^3 ∧ V > 0 := by sorry

end NUMINAMATH_CALUDE_volume_cube_inscribed_sphere_l1863_186310


namespace NUMINAMATH_CALUDE_sum_product_uniqueness_l1863_186348

theorem sum_product_uniqueness (S P : ℝ) (x y : ℝ) 
  (h_sum : x + y = S) (h_product : x * y = P) :
  (x = (S + Real.sqrt (S^2 - 4*P)) / 2 ∧ y = (S - Real.sqrt (S^2 - 4*P)) / 2) ∨
  (x = (S - Real.sqrt (S^2 - 4*P)) / 2 ∧ y = (S + Real.sqrt (S^2 - 4*P)) / 2) :=
by sorry

end NUMINAMATH_CALUDE_sum_product_uniqueness_l1863_186348


namespace NUMINAMATH_CALUDE_expected_balls_original_positions_l1863_186302

/-- The number of balls arranged in a circle -/
def num_balls : ℕ := 7

/-- The probability that a ball is in its original position after Chris and Silva's actions -/
def prob_original_position : ℚ := 25 / 49

/-- The expected number of balls in their original positions -/
def expected_original_positions : ℚ := num_balls * prob_original_position

/-- Theorem stating the expected number of balls in their original positions -/
theorem expected_balls_original_positions :
  expected_original_positions = 175 / 49 := by sorry

end NUMINAMATH_CALUDE_expected_balls_original_positions_l1863_186302


namespace NUMINAMATH_CALUDE_unique_prime_p_l1863_186327

theorem unique_prime_p : ∃! p : ℕ, Prime p ∧ Prime (5 * p + 1) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_prime_p_l1863_186327


namespace NUMINAMATH_CALUDE_problem_solution_l1863_186367

theorem problem_solution (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_eq : a^2*(a-1) + b^2*(b-1) + c^2*(c-1) = a*(a-1) + b*(b-1) + c*(c-1)) :
  1956*a^2 + 1986*b^2 + 2016*c^2 = 5958 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1863_186367


namespace NUMINAMATH_CALUDE_intersection_range_l1863_186313

theorem intersection_range (a : ℝ) : 
  (∃ x y : ℝ, x - y + 1 = 0 ∧ (x - a)^2 + y^2 = 2) ↔ 
  -3 ≤ a ∧ a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_intersection_range_l1863_186313


namespace NUMINAMATH_CALUDE_total_cost_for_six_people_l1863_186371

/-- The total cost of buying soda and pizza for a group -/
def total_cost (num_people : ℕ) (soda_price pizza_price : ℚ) : ℚ :=
  num_people * (soda_price + pizza_price)

/-- Theorem: The total cost for 6 people is $9.00 -/
theorem total_cost_for_six_people :
  total_cost 6 (1/2) 1 = 9 :=
by sorry

end NUMINAMATH_CALUDE_total_cost_for_six_people_l1863_186371


namespace NUMINAMATH_CALUDE_even_power_plus_one_all_digits_equal_l1863_186311

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

end NUMINAMATH_CALUDE_even_power_plus_one_all_digits_equal_l1863_186311


namespace NUMINAMATH_CALUDE_dilution_proof_l1863_186389

def initial_volume : ℝ := 12
def initial_concentration : ℝ := 0.60
def final_concentration : ℝ := 0.40
def water_added : ℝ := 6

theorem dilution_proof :
  let initial_alcohol := initial_volume * initial_concentration
  let final_volume := initial_volume + water_added
  initial_alcohol / final_volume = final_concentration :=
by sorry

end NUMINAMATH_CALUDE_dilution_proof_l1863_186389


namespace NUMINAMATH_CALUDE_min_value_theorem_l1863_186303

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

/-- The first circle equation -/
def circle1 (x y : ℝ) : Prop := (x+2)^2 + y^2 = 4

/-- The second circle equation -/
def circle2 (x y : ℝ) : Prop := (x-2)^2 + y^2 = 1

/-- The expression |PM|^2 - |PN|^2 -/
def expr (x : ℝ) : ℝ := 8*x - 3

/-- The theorem stating the minimum value of |PM|^2 - |PN|^2 -/
theorem min_value_theorem (x y : ℝ) (h1 : hyperbola x y) (h2 : x ≥ 1) :
  ∃ (m : ℝ), m = 5 ∧ ∀ (x' y' : ℝ), hyperbola x' y' → x' ≥ 1 → expr x' ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1863_186303


namespace NUMINAMATH_CALUDE_parabola_point_coordinates_l1863_186395

/-- Given a parabola y^2 = 4x, point A(3,0), and a point P on the parabola,
    if a line through P intersects perpendicularly with x = -1 at B,
    and |PB| = |PA|, then the x-coordinate of P is 2. -/
theorem parabola_point_coordinates (P : ℝ × ℝ) :
  P.2^2 = 4 * P.1 →  -- P is on the parabola y^2 = 4x
  ∃ B : ℝ × ℝ, 
    B.1 = -1 ∧  -- B is on the line x = -1
    (P.2 - B.2) * (P.1 - B.1) = -1 ∧  -- PB is perpendicular to x = -1
    (P.1 - B.1)^2 + (P.2 - B.2)^2 = (P.1 - 3)^2 + P.2^2 →  -- |PB| = |PA|
  P.1 = 2 :=
sorry

end NUMINAMATH_CALUDE_parabola_point_coordinates_l1863_186395


namespace NUMINAMATH_CALUDE_smallest_five_digit_mod_13_l1863_186329

theorem smallest_five_digit_mod_13 : 
  ∀ n : ℕ, n ≥ 10000 ∧ n ≡ 11 [MOD 13] → n ≥ 10009 :=
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_mod_13_l1863_186329


namespace NUMINAMATH_CALUDE_rectangle_no_integer_points_l1863_186301

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

end NUMINAMATH_CALUDE_rectangle_no_integer_points_l1863_186301


namespace NUMINAMATH_CALUDE_a_equals_two_l1863_186342

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x + 4

-- State the theorem
theorem a_equals_two (a : ℝ) : f a a = f a 1 + 2 * (1 - 1) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_a_equals_two_l1863_186342


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l1863_186362

theorem quadratic_equation_coefficients :
  ∀ (a b c : ℝ),
  (∀ x, a * x^2 + b * x + c = 0 ↔ 3 * x^2 - 4 * x + 1 = 0) →
  a = 3 ∧ b = -4 ∧ c = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l1863_186362


namespace NUMINAMATH_CALUDE_great_eight_teams_l1863_186394

/-- The number of teams in the GREAT EIGHT conference -/
def num_teams : ℕ := 9

/-- The total number of games played in the conference -/
def total_games : ℕ := 36

/-- The number of games played by one team -/
def games_per_team : ℕ := 8

/-- Calculates the number of games in a round-robin tournament -/
def round_robin_games (n : ℕ) : ℕ := n * (n - 1) / 2

theorem great_eight_teams :
  (round_robin_games num_teams = total_games) ∧
  (num_teams - 1 = games_per_team) := by
  sorry

end NUMINAMATH_CALUDE_great_eight_teams_l1863_186394


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l1863_186357

theorem max_sum_of_squares (m n : ℕ) : 
  m ∈ Finset.range 1982 →
  n ∈ Finset.range 1982 →
  (n^2 - m*n - m^2)^2 = 1 →
  m^2 + n^2 ≤ 3524578 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l1863_186357


namespace NUMINAMATH_CALUDE_shaded_triangle_area_and_percentage_l1863_186337

/-- Given an equilateral triangle with side length 4 cm, prove the area and percentage of a shaded region -/
theorem shaded_triangle_area_and_percentage :
  let side_length : ℝ := 4
  let original_height : ℝ := side_length * (Real.sqrt 3) / 2
  let original_area : ℝ := side_length^2 * (Real.sqrt 3) / 4
  let shaded_base : ℝ := side_length * 3 / 4
  let shaded_height : ℝ := original_height / 2
  let shaded_area : ℝ := shaded_base * shaded_height / 2
  let percentage : ℝ := shaded_area / original_area * 100
  shaded_area = 3 * (Real.sqrt 3) / 2 ∧ percentage = 37.5 := by
  sorry


end NUMINAMATH_CALUDE_shaded_triangle_area_and_percentage_l1863_186337


namespace NUMINAMATH_CALUDE_sane_person_identified_l1863_186343

/-- Represents the types of individuals in Transylvania -/
inductive PersonType
| Sane
| Transylvanian

/-- Represents possible answers to a question -/
inductive Answer
| Yes
| No

/-- A function that determines how a person of a given type would answer the question -/
def wouldAnswer (t : PersonType) : Answer :=
  match t with
  | PersonType.Sane => Answer.No
  | PersonType.Transylvanian => Answer.Yes

/-- Theorem stating that if an answer allows immediate identification, the person must be sane -/
theorem sane_person_identified
  (answer : Answer)
  (h_immediate : ∃ (t : PersonType), wouldAnswer t = answer) :
  answer = Answer.No ∧ wouldAnswer PersonType.Sane = answer :=
sorry

end NUMINAMATH_CALUDE_sane_person_identified_l1863_186343


namespace NUMINAMATH_CALUDE_geometric_sum_first_six_terms_l1863_186375

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_first_six_terms :
  geometric_sum (1/4) (1/4) 6 = 4095/12288 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_first_six_terms_l1863_186375


namespace NUMINAMATH_CALUDE_intersection_point_coords_l1863_186382

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

end NUMINAMATH_CALUDE_intersection_point_coords_l1863_186382


namespace NUMINAMATH_CALUDE_y_in_terms_of_x_l1863_186331

theorem y_in_terms_of_x (x y : ℝ) (h : x - 2 = y + 3*x) : y = -2*x - 2 := by
  sorry

end NUMINAMATH_CALUDE_y_in_terms_of_x_l1863_186331


namespace NUMINAMATH_CALUDE_max_sin_a_value_l1863_186344

theorem max_sin_a_value (a b c : Real) 
  (h1 : Real.cos a = Real.tan b)
  (h2 : Real.cos b = Real.tan c)
  (h3 : Real.cos c = Real.tan a) :
  ∃ (max_sin_a : Real), 
    (∀ a' b' c' : Real, 
      Real.cos a' = Real.tan b' → 
      Real.cos b' = Real.tan c' → 
      Real.cos c' = Real.tan a' → 
      Real.sin a' ≤ max_sin_a) ∧
    max_sin_a = Real.sqrt ((3 - Real.sqrt 5) / 2) :=
by sorry

end NUMINAMATH_CALUDE_max_sin_a_value_l1863_186344


namespace NUMINAMATH_CALUDE_geometry_propositions_l1863_186320

theorem geometry_propositions (p₁ p₂ p₃ p₄ : Prop) 
  (h₁ : p₁) (h₂ : ¬p₂) (h₃ : ¬p₃) (h₄ : p₄) :
  (p₁ ∧ p₄) ∧ (¬p₂ ∨ p₃) ∧ (¬p₃ ∨ ¬p₄) ∧ ¬(p₁ ∧ p₂) :=
sorry

end NUMINAMATH_CALUDE_geometry_propositions_l1863_186320


namespace NUMINAMATH_CALUDE_min_value_sum_l1863_186373

theorem min_value_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_prod : a * b * c = 27) :
  ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → x * y * z = 27 → a + 3 * b + 9 * c ≤ x + 3 * y + 9 * z :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_l1863_186373


namespace NUMINAMATH_CALUDE_simplify_expression_solve_equation_solve_system_l1863_186353

-- Part 1
theorem simplify_expression (a b : ℝ) :
  3 * (a - b)^2 - 6 * (a - b)^2 + 2 * (a - b)^2 = -(a - b)^2 := by sorry

-- Part 2
theorem solve_equation (x y : ℝ) (h : x^2 - 2*y = 4) :
  23 - 3*x^2 + 6*y = 11 := by sorry

-- Part 3
theorem solve_system (a b c d : ℝ) 
  (h1 : a - 2*b = 3) (h2 : 2*b - c = -5) (h3 : c - d = -9) :
  (a - c) + (2*b - d) - (2*b - c) = -11 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_solve_equation_solve_system_l1863_186353


namespace NUMINAMATH_CALUDE_positive_number_equality_l1863_186330

theorem positive_number_equality (x : ℝ) (h1 : x > 0) :
  (2 / 3) * x = (25 / 216) * (1 / x) → x = 5 / 12 := by sorry

end NUMINAMATH_CALUDE_positive_number_equality_l1863_186330


namespace NUMINAMATH_CALUDE_gcd_282_470_l1863_186312

theorem gcd_282_470 : Nat.gcd 282 470 = 94 := by
  sorry

end NUMINAMATH_CALUDE_gcd_282_470_l1863_186312


namespace NUMINAMATH_CALUDE_simplify_trig_expression_simplify_trig_expression_second_quadrant_l1863_186392

-- Problem 1
theorem simplify_trig_expression : 
  (Real.sqrt (1 - 2 * Real.sin (130 * π / 180) * Real.cos (130 * π / 180))) / 
  (Real.sin (130 * π / 180) + Real.sqrt (1 - Real.sin (130 * π / 180) ^ 2)) = 1 := by sorry

-- Problem 2
theorem simplify_trig_expression_second_quadrant (α : Real) 
  (h : π / 2 < α ∧ α < π) : 
  Real.cos α * Real.sqrt ((1 - Real.sin α) / (1 + Real.sin α)) + 
  Real.sin α * Real.sqrt ((1 - Real.cos α) / (1 + Real.cos α)) = 
  Real.sin α - Real.cos α := by sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_simplify_trig_expression_second_quadrant_l1863_186392


namespace NUMINAMATH_CALUDE_hot_dogs_remainder_l1863_186318

theorem hot_dogs_remainder : 16789537 % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_hot_dogs_remainder_l1863_186318


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1863_186397

theorem polynomial_factorization (x : ℝ) : 
  x^9 + x^6 + x^3 + 1 = (x^3 + 1) * (x^6 - x^3 + 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1863_186397


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l1863_186398

def angle_between_vectors (a b : ℝ × ℝ) : ℝ := sorry

theorem vector_sum_magnitude 
  (a b : ℝ × ℝ) 
  (h1 : angle_between_vectors a b = π / 3)
  (h2 : a = (2, 0))
  (h3 : Real.sqrt ((Prod.fst b)^2 + (Prod.snd b)^2) = 1) :
  Real.sqrt ((Prod.fst (a + 2 • b))^2 + (Prod.snd (a + 2 • b))^2) = 2 * Real.sqrt 3 := by
    sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l1863_186398


namespace NUMINAMATH_CALUDE_triangle_third_side_l1863_186317

theorem triangle_third_side (a b : ℝ) (h1 : a = 3.14) (h2 : b = 0.67) : 
  ∃ c : ℕ, c = 3 ∧ 
    a + b > c ∧
    a + c > b ∧
    b + c > a := by
  sorry

end NUMINAMATH_CALUDE_triangle_third_side_l1863_186317


namespace NUMINAMATH_CALUDE_sum_x_y_equals_nine_fifths_l1863_186321

theorem sum_x_y_equals_nine_fifths (x y : ℝ) 
  (eq1 : x + |x| + y = 5)
  (eq2 : x + |y| - y = 6) : 
  x + y = 9/5 := by
  sorry

end NUMINAMATH_CALUDE_sum_x_y_equals_nine_fifths_l1863_186321


namespace NUMINAMATH_CALUDE_smallest_share_is_five_thirds_l1863_186384

/-- Represents the shares of bread in an arithmetic sequence -/
structure BreadShares where
  a : ℚ  -- The middle term of the arithmetic sequence
  d : ℚ  -- The common difference of the arithmetic sequence
  sum_equals_100 : 5 * a = 100
  larger_three_seventh_smaller_two : 3 * a + 3 * d = 7 * (2 * a - 3 * d)
  d_positive : d > 0

/-- The smallest share of bread -/
def smallest_share (shares : BreadShares) : ℚ :=
  shares.a - 2 * shares.d

/-- Theorem stating that the smallest share is 5/3 -/
theorem smallest_share_is_five_thirds (shares : BreadShares) :
  smallest_share shares = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_share_is_five_thirds_l1863_186384


namespace NUMINAMATH_CALUDE_rectangle_side_ratio_l1863_186369

/-- Represents the configuration of rectangles around a square -/
structure RectangleSquareConfig where
  inner_square_side : ℝ
  rectangle_short_side : ℝ
  rectangle_long_side : ℝ

/-- The theorem stating the ratio of rectangle sides given the square configuration -/
theorem rectangle_side_ratio
  (config : RectangleSquareConfig)
  (h1 : config.inner_square_side + 2 * config.rectangle_short_side = 2 * config.inner_square_side)
  (h2 : config.rectangle_long_side + config.inner_square_side = 2 * config.inner_square_side)
  (h3 : (2 * config.inner_square_side) ^ 2 = 4 * config.inner_square_side ^ 2) :
  config.rectangle_long_side / config.rectangle_short_side = 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_side_ratio_l1863_186369


namespace NUMINAMATH_CALUDE_sum_range_l1863_186319

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then -2*x - x^2 else |Real.log x|

theorem sum_range (a b c d : ℝ) :
  a < b ∧ b < c ∧ c < d ∧ f a = f b ∧ f b = f c ∧ f c = f d →
  1 < a + b + c + 2*d ∧ a + b + c + 2*d < 181/10 :=
sorry

end NUMINAMATH_CALUDE_sum_range_l1863_186319


namespace NUMINAMATH_CALUDE_days_for_C_alone_is_8_l1863_186308

/-- The number of days it takes for C to finish the work alone, given that:
    - A, B, and C together can finish the work in 4 days
    - A alone can finish the work in 12 days
    - B alone can finish the work in 24 days
-/
def days_for_C_alone (days_together days_A_alone days_B_alone : ℚ) : ℚ :=
  let work_rate_together := 1 / days_together
  let work_rate_A := 1 / days_A_alone
  let work_rate_B := 1 / days_B_alone
  let work_rate_C := work_rate_together - work_rate_A - work_rate_B
  1 / work_rate_C

theorem days_for_C_alone_is_8 :
  days_for_C_alone 4 12 24 = 8 := by
  sorry

end NUMINAMATH_CALUDE_days_for_C_alone_is_8_l1863_186308


namespace NUMINAMATH_CALUDE_a_power_value_l1863_186347

theorem a_power_value (a n : ℝ) (h : a^(2*n) = 3) : 2*a^(6*n) - 1 = 53 := by
  sorry

end NUMINAMATH_CALUDE_a_power_value_l1863_186347


namespace NUMINAMATH_CALUDE_new_person_weight_l1863_186334

theorem new_person_weight (n : Nat) (original_weight replaced_weight increase : ℝ) :
  n = 8 ∧ 
  replaced_weight = 50 ∧ 
  increase = 2.5 →
  (n : ℝ) * increase + replaced_weight = 70 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l1863_186334


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1863_186306

theorem polynomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x - 3)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ = 10 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1863_186306


namespace NUMINAMATH_CALUDE_triangle_inequality_with_interior_point_l1863_186351

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the perimeter of the triangle
def perimeter (t : Triangle) : ℝ := sorry

-- Define a point inside the triangle
def insidePoint (t : Triangle) : ℝ × ℝ := sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_inequality_with_interior_point (t : Triangle) :
  let P := perimeter t
  let O := insidePoint t
  P / 2 < distance O t.A + distance O t.B + distance O t.C ∧
  distance O t.A + distance O t.B + distance O t.C < P :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_with_interior_point_l1863_186351


namespace NUMINAMATH_CALUDE_malvina_money_l1863_186315

theorem malvina_money (m n : ℕ) : 
  m + n < 40 →
  n < 8 * m →
  n ≥ 4 * m + 15 →
  n = 31 :=
by sorry

end NUMINAMATH_CALUDE_malvina_money_l1863_186315


namespace NUMINAMATH_CALUDE_students_without_favorite_l1863_186393

theorem students_without_favorite (total : ℕ) (math_frac english_frac history_frac science_frac : ℚ) : 
  total = 120 →
  math_frac = 3 / 10 →
  english_frac = 5 / 12 →
  history_frac = 1 / 8 →
  science_frac = 3 / 20 →
  total - (↑total * math_frac).floor - (↑total * english_frac).floor - 
  (↑total * history_frac).floor - (↑total * science_frac).floor = 1 := by
  sorry

end NUMINAMATH_CALUDE_students_without_favorite_l1863_186393


namespace NUMINAMATH_CALUDE_bills_age_l1863_186378

theorem bills_age (caroline_age : ℝ) 
  (h1 : caroline_age + (2 * caroline_age - 1) + (caroline_age - 4) = 45) : 
  2 * caroline_age - 1 = 24 := by
  sorry

#check bills_age

end NUMINAMATH_CALUDE_bills_age_l1863_186378


namespace NUMINAMATH_CALUDE_rowing_current_rate_l1863_186356

/-- Proves that the rate of the current is 1.4 km/hr given the conditions of the rowing problem -/
theorem rowing_current_rate (rowing_speed : ℝ) (upstream_time downstream_time : ℝ) : 
  rowing_speed = 4.2 →
  upstream_time = 2 * downstream_time →
  let current_rate := (rowing_speed / 3 : ℝ)
  current_rate = 1.4 := by sorry

end NUMINAMATH_CALUDE_rowing_current_rate_l1863_186356


namespace NUMINAMATH_CALUDE_certain_fraction_proof_l1863_186307

theorem certain_fraction_proof : 
  ∃ (x y : ℚ), (3 / 7) / (x / y) = (2 / 5) / (1 / 7) ∧ x / y = 15 / 98 :=
by sorry

end NUMINAMATH_CALUDE_certain_fraction_proof_l1863_186307


namespace NUMINAMATH_CALUDE_unit_circle_dot_product_l1863_186380

theorem unit_circle_dot_product 
  (x₁ y₁ x₂ y₂ θ : ℝ) 
  (h₁ : x₁^2 + y₁^2 = 1) 
  (h₂ : x₂^2 + y₂^2 = 1)
  (h₃ : π/2 < θ ∧ θ < π)
  (h₄ : Real.sin (θ + π/4) = 3/5) : 
  x₁ * x₂ + y₁ * y₂ = -Real.sqrt 2 / 10 := by
sorry

end NUMINAMATH_CALUDE_unit_circle_dot_product_l1863_186380


namespace NUMINAMATH_CALUDE_calculation_correction_l1863_186346

theorem calculation_correction (x : ℝ) (h : 63 / x = 9) : 36 - x = 29 := by
  sorry

end NUMINAMATH_CALUDE_calculation_correction_l1863_186346


namespace NUMINAMATH_CALUDE_quadratic_tangent_to_x_axis_l1863_186345

/-- A quadratic function f(x) = ax^2 + bx + c is tangent to the x-axis
    if and only if c = b^2 / (4a) -/
theorem quadratic_tangent_to_x_axis (a b c : ℝ) (h : c = b^2 / (4 * a)) :
  let f := λ x : ℝ => a * x^2 + b * x + c
  ∃ x₀ : ℝ, f x₀ = 0 ∧ ∀ x : ℝ, x ≠ x₀ → f x > 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_tangent_to_x_axis_l1863_186345


namespace NUMINAMATH_CALUDE_carnival_ticket_cost_l1863_186372

/-- The cost of carnival tickets -/
theorem carnival_ticket_cost :
  ∀ (cost_12 : ℚ) (cost_4 : ℚ),
  cost_12 = 3 →
  12 * cost_4 = cost_12 →
  cost_4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_carnival_ticket_cost_l1863_186372


namespace NUMINAMATH_CALUDE_exist_distant_points_on_polyhedron_l1863_186364

/-- A sphere with a given radius -/
structure Sphere where
  radius : ℝ

/-- A polyhedron with a given number of faces -/
structure Polyhedron where
  faces : ℕ

/-- A polyhedron is circumscribed around a sphere -/
def is_circumscribed (p : Polyhedron) (s : Sphere) : Prop :=
  sorry

/-- The distance between two points on the surface of a polyhedron -/
def surface_distance (p : Polyhedron) (point1 point2 : ℝ × ℝ × ℝ) : ℝ :=
  sorry

/-- The main theorem -/
theorem exist_distant_points_on_polyhedron (s : Sphere) (p : Polyhedron) 
  (h_radius : s.radius = 10)
  (h_faces : p.faces = 19)
  (h_circumscribed : is_circumscribed p s) :
  ∃ (point1 point2 : ℝ × ℝ × ℝ), surface_distance p point1 point2 > 21 :=
sorry

end NUMINAMATH_CALUDE_exist_distant_points_on_polyhedron_l1863_186364


namespace NUMINAMATH_CALUDE_triangle_construction_l1863_186300

-- Define the necessary structures
structure Line where
  -- Add necessary fields for a line

structure Point where
  -- Add necessary fields for a point

structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the necessary functions
def is_on_line (p : Point) (l : Line) : Prop :=
  sorry

def is_foot_of_altitude (m : Point) (v : Point) (s : Point) (t : Point) : Prop :=
  sorry

-- Main theorem
theorem triangle_construction (L : Line) (M₁ M₂ : Point) :
  ∃ (ABC A'B'C' : Triangle),
    (is_on_line ABC.C L ∧ is_on_line ABC.B L) ∧
    (is_on_line A'B'C'.C L ∧ is_on_line A'B'C'.B L) ∧
    (is_foot_of_altitude M₁ ABC.A ABC.B ABC.C) ∧
    (is_foot_of_altitude M₂ ABC.B ABC.A ABC.C) ∧
    (is_foot_of_altitude M₁ A'B'C'.A A'B'C'.B A'B'C'.C) ∧
    (is_foot_of_altitude M₂ A'B'C'.B A'B'C'.A A'B'C'.C) :=
  sorry


end NUMINAMATH_CALUDE_triangle_construction_l1863_186300


namespace NUMINAMATH_CALUDE_sqrt_product_equation_l1863_186390

theorem sqrt_product_equation (x : ℝ) (hx : x > 0) :
  Real.sqrt (16 * x) * Real.sqrt (25 * x) * Real.sqrt (5 * x) * Real.sqrt (20 * x) = 40 →
  x = 1 / Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_sqrt_product_equation_l1863_186390


namespace NUMINAMATH_CALUDE_factorial_ratio_45_43_l1863_186328

-- Define factorial function
def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- State the theorem
theorem factorial_ratio_45_43 : factorial 45 / factorial 43 = 1980 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_45_43_l1863_186328


namespace NUMINAMATH_CALUDE_tan_alpha_value_l1863_186379

theorem tan_alpha_value (α β : ℝ) 
  (h1 : Real.tan (α + β) = 3/5) 
  (h2 : Real.tan β = 1/3) : 
  Real.tan α = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l1863_186379
