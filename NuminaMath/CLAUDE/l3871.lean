import Mathlib

namespace NUMINAMATH_CALUDE_oil_price_reduction_l3871_387104

/-- Calculates the additional amount of oil a housewife can obtain after a price reduction --/
theorem oil_price_reduction (original_price reduced_price budget : ℝ) : 
  original_price > 0 →
  reduced_price > 0 →
  budget > 0 →
  reduced_price = original_price * (1 - 0.35) →
  reduced_price = 56 →
  budget = 800 →
  let additional_amount := budget / reduced_price - budget / original_price
  ∃ ε > 0, |additional_amount - 5.01| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_oil_price_reduction_l3871_387104


namespace NUMINAMATH_CALUDE_scallop_cost_per_pound_l3871_387101

/-- The cost per pound of jumbo scallops -/
def cost_per_pound (scallops_per_pound : ℕ) (num_people : ℕ) (scallops_per_person : ℕ) (total_cost : ℚ) : ℚ :=
  let total_scallops := num_people * scallops_per_person
  let pounds_needed := total_scallops / scallops_per_pound
  total_cost / pounds_needed

/-- Theorem stating the cost per pound of jumbo scallops is $24 -/
theorem scallop_cost_per_pound :
  cost_per_pound 8 8 2 48 = 24 := by
  sorry

end NUMINAMATH_CALUDE_scallop_cost_per_pound_l3871_387101


namespace NUMINAMATH_CALUDE_max_min_x2_minus_y2_l3871_387127

theorem max_min_x2_minus_y2 (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 - x*y + y^2 = 1) :
  ∃ (min max : ℝ), (∀ z, z = x^2 - y^2 → min ≤ z ∧ z ≤ max) ∧
                   min = -2*Real.sqrt 3/3 ∧
                   max = 2*Real.sqrt 3/3 := by
  sorry

end NUMINAMATH_CALUDE_max_min_x2_minus_y2_l3871_387127


namespace NUMINAMATH_CALUDE_polynomial_identity_l3871_387182

theorem polynomial_identity (a b c : ℝ) :
  a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2 = 2 * (a - b) * (b - c) * (c - a) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l3871_387182


namespace NUMINAMATH_CALUDE_point_distance_equality_l3871_387116

/-- Given points A(4, 2) and B(0, b) in the Cartesian coordinate system,
    if |BO| = |BA|, then b = 5. -/
theorem point_distance_equality (b : ℝ) : 
  let A : ℝ × ℝ := (4, 2)
  let B : ℝ × ℝ := (0, b)
  let O : ℝ × ℝ := (0, 0)
  (‖B - O‖ = ‖B - A‖) → b = 5 :=
by sorry

end NUMINAMATH_CALUDE_point_distance_equality_l3871_387116


namespace NUMINAMATH_CALUDE_max_value_theorem_l3871_387174

theorem max_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : a * Real.sqrt (1 - b^2) - b * Real.sqrt (1 - a^2) = a * b) :
  (a / b + b / a) ≤ Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3871_387174


namespace NUMINAMATH_CALUDE_probability_of_seven_in_three_eighths_l3871_387135

theorem probability_of_seven_in_three_eighths :
  let decimal_rep := (3 : ℚ) / 8
  let digits := [3, 7, 5]
  (digits.count 7 : ℚ) / digits.length = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_probability_of_seven_in_three_eighths_l3871_387135


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l3871_387170

theorem arithmetic_sequence_length
  (a : ℕ) (d : ℕ) (last : ℕ) (n : ℕ)
  (h1 : a = 3)
  (h2 : d = 4)
  (h3 : last = 47)
  (h4 : last = a + (n - 1) * d) :
  n = 12 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l3871_387170


namespace NUMINAMATH_CALUDE_linear_function_intersection_l3871_387130

/-- A linear function passing through (-1, -2) intersects the y-axis at (0, -1) -/
theorem linear_function_intersection (k : ℝ) : 
  (∀ x y, y = k * (x - 1) → (x = -1 ∧ y = -2) → (0 = k * (-1 - 1) + 2)) → 
  (∃ y, y = k * (0 - 1) ∧ y = -1) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_intersection_l3871_387130


namespace NUMINAMATH_CALUDE_odd_digits_sum_152_345_l3871_387193

/-- Converts a base 10 number to base 4 -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Counts odd digits in a list of natural numbers -/
def countOddDigits (digits : List ℕ) : ℕ :=
  sorry

theorem odd_digits_sum_152_345 : 
  let base4_152 := toBase4 152
  let base4_345 := toBase4 345
  countOddDigits base4_152 + countOddDigits base4_345 = 6 := by
  sorry

end NUMINAMATH_CALUDE_odd_digits_sum_152_345_l3871_387193


namespace NUMINAMATH_CALUDE_parabola_point_distance_l3871_387131

theorem parabola_point_distance (x₀ y₀ : ℝ) : 
  x₀^2 = 28 * y₀ →                           -- Point is on the parabola
  (y₀ + 7/2)^2 + x₀^2 = 9 * y₀^2 →           -- Distance to focus is 3 times distance to x-axis
  y₀ = 7/2 := by sorry

end NUMINAMATH_CALUDE_parabola_point_distance_l3871_387131


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l3871_387147

theorem arithmetic_calculations :
  ((1 : ℤ) * (-5) - (-6) + (-7) = -6) ∧
  ((-1 : ℤ)^2021 + (-18) * |(-2 : ℚ) / 9| - 4 / (-2 : ℤ) = -3) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l3871_387147


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l3871_387161

theorem sqrt_meaningful_range (x : ℝ) : (∃ y : ℝ, y^2 = x - 1) → x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l3871_387161


namespace NUMINAMATH_CALUDE_sample_size_theorem_l3871_387173

theorem sample_size_theorem (N : ℕ) (sample_size : ℕ) (probability : ℚ) 
  (h1 : sample_size = 30)
  (h2 : probability = 1/4)
  (h3 : (sample_size : ℚ) / N = probability) : 
  N = 120 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_theorem_l3871_387173


namespace NUMINAMATH_CALUDE_irrational_sqrt_three_rational_others_l3871_387114

theorem irrational_sqrt_three_rational_others : 
  (Irrational (Real.sqrt 3)) ∧ 
  (¬ Irrational (22 / 7 : ℝ)) ∧ 
  (¬ Irrational (0 : ℝ)) ∧ 
  (¬ Irrational (3.14 : ℝ)) := by sorry

end NUMINAMATH_CALUDE_irrational_sqrt_three_rational_others_l3871_387114


namespace NUMINAMATH_CALUDE_real_roots_of_polynomial_l3871_387156

theorem real_roots_of_polynomial (x : ℝ) : x^2 - 4 = 0 ↔ x = 2 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_real_roots_of_polynomial_l3871_387156


namespace NUMINAMATH_CALUDE_smallest_positive_angle_same_terminal_side_l3871_387168

/-- Given an angle α = 2012°, the smallest positive angle θ with the same terminal side is 212°. -/
theorem smallest_positive_angle_same_terminal_side :
  let α : Real := 2012
  ∃ θ : Real,
    0 < θ ∧ 
    θ ≤ 360 ∧
    ∃ k : Int, α = k * 360 + θ ∧
    ∀ φ : Real, (0 < φ ∧ φ ≤ 360 ∧ ∃ m : Int, α = m * 360 + φ) → θ ≤ φ ∧
    θ = 212 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_angle_same_terminal_side_l3871_387168


namespace NUMINAMATH_CALUDE_r_nonzero_l3871_387166

/-- A polynomial of degree 5 with specific properties -/
def Q (p q r s t : ℝ) (x : ℝ) : ℝ :=
  x^5 + p*x^4 + q*x^3 + r*x^2 + s*x + t

/-- The property that Q has five distinct x-intercepts including (0,0) -/
def has_five_distinct_intercepts (p q r s t : ℝ) : Prop :=
  ∃ (α β : ℝ), α ≠ 0 ∧ β ≠ 0 ∧ α ≠ β ∧
    ∀ x, Q p q r s t x = 0 ↔ x = 0 ∨ x = α ∨ x = -α ∨ x = β ∨ x = -β

/-- The theorem stating that r must be non-zero given the conditions -/
theorem r_nonzero (p q r s t : ℝ) 
  (h : has_five_distinct_intercepts p q r s t) : r ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_r_nonzero_l3871_387166


namespace NUMINAMATH_CALUDE_emails_evening_l3871_387110

def emails_problem (afternoon evening morning total : ℕ) : Prop :=
  afternoon = 3 ∧ morning = 6 ∧ total = 10 ∧ afternoon + evening + morning = total

theorem emails_evening : ∃ evening : ℕ, emails_problem 3 evening 6 10 ∧ evening = 1 :=
  sorry

end NUMINAMATH_CALUDE_emails_evening_l3871_387110


namespace NUMINAMATH_CALUDE_heartsuit_three_five_l3871_387155

-- Define the heartsuit operation
def heartsuit (x y : ℤ) : ℤ := 4*x + 6*y

-- Theorem statement
theorem heartsuit_three_five : heartsuit 3 5 = 42 := by
  sorry

end NUMINAMATH_CALUDE_heartsuit_three_five_l3871_387155


namespace NUMINAMATH_CALUDE_basketball_free_throw_probability_l3871_387132

theorem basketball_free_throw_probability :
  ∀ p : ℝ,
  0 ≤ p ∧ p ≤ 1 →
  (1 - p^2 = 16/25) →
  p = 3/5 :=
by sorry

end NUMINAMATH_CALUDE_basketball_free_throw_probability_l3871_387132


namespace NUMINAMATH_CALUDE_youtube_video_length_l3871_387171

theorem youtube_video_length (total_time : ℕ) (video1_length : ℕ) (video2_length : ℕ) :
  total_time = 510 ∧
  video1_length = 120 ∧
  video2_length = 270 →
  ∃ (last_video_length : ℕ),
    last_video_length * 2 = total_time - (video1_length + video2_length) ∧
    last_video_length = 60 := by
  sorry

end NUMINAMATH_CALUDE_youtube_video_length_l3871_387171


namespace NUMINAMATH_CALUDE_intersection_point_correct_l3871_387167

/-- The intersection point of two lines y = -2x and y = x -/
def intersection_point : ℝ × ℝ := (0, 0)

/-- Function representing y = -2x -/
def f (x : ℝ) : ℝ := -2 * x

/-- Function representing y = x -/
def g (x : ℝ) : ℝ := x

/-- Theorem stating that (0, 0) is the unique intersection point of y = -2x and y = x -/
theorem intersection_point_correct :
  (∃! p : ℝ × ℝ, f p.1 = p.2 ∧ g p.1 = p.2) ∧
  (∀ p : ℝ × ℝ, f p.1 = p.2 ∧ g p.1 = p.2 → p = intersection_point) :=
sorry

end NUMINAMATH_CALUDE_intersection_point_correct_l3871_387167


namespace NUMINAMATH_CALUDE_odd_function_domain_symmetry_l3871_387157

/-- A function f is odd if its domain is symmetric about the origin -/
def is_odd_function (f : ℝ → ℝ) (domain : Set ℝ) : Prop :=
  ∀ x ∈ domain, -x ∈ domain

/-- The domain of the function -/
def function_domain (t : ℝ) : Set ℝ := Set.Ioo t (2*t + 3)

/-- Theorem: If f is an odd function with domain (t, 2t+3), then t = -1 -/
theorem odd_function_domain_symmetry (f : ℝ → ℝ) (t : ℝ) 
  (h : is_odd_function f (function_domain t)) : 
  t = -1 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_domain_symmetry_l3871_387157


namespace NUMINAMATH_CALUDE_factorial_30_trailing_zeros_l3871_387197

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- 30! has 7 trailing zeros -/
theorem factorial_30_trailing_zeros : trailingZeros 30 = 7 := by sorry

end NUMINAMATH_CALUDE_factorial_30_trailing_zeros_l3871_387197


namespace NUMINAMATH_CALUDE_triangle_property_l3871_387172

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The condition that a*sin(B) - √3*b*cos(A) = 0 -/
def satisfiesCondition (t : Triangle) : Prop :=
  t.a * Real.sin t.B - Real.sqrt 3 * t.b * Real.cos t.A = 0

theorem triangle_property (t : Triangle) 
  (h : satisfiesCondition t) : 
  t.A = π / 3 ∧ 
  (t.a = Real.sqrt 7 ∧ t.b = 2 → 
    (1/2 : ℝ) * t.b * t.c * Real.sin t.A = (3 * Real.sqrt 3) / 2) :=
sorry


end NUMINAMATH_CALUDE_triangle_property_l3871_387172


namespace NUMINAMATH_CALUDE_expansion_properties_l3871_387103

def sum_of_coefficients (n : ℕ) : ℝ := (3 + 1)^n

def sum_of_binomial_coefficients (n : ℕ) : ℝ := 2^n

theorem expansion_properties (n : ℕ) 
  (h : sum_of_coefficients n - sum_of_binomial_coefficients n = 240) :
  n = 4 ∧ 
  ∃ (a b c : ℝ), a = 81 ∧ b = 54 ∧ c = 1 ∧
  (∀ (x : ℝ), (3*x + x^(1/2))^n = a*x^4 + b*x^3 + c*x^2 + x^(7/2) + 6*x^(5/2) + 4*x^(3/2) + x^(1/2)) :=
by sorry

end NUMINAMATH_CALUDE_expansion_properties_l3871_387103


namespace NUMINAMATH_CALUDE_cans_per_person_day2_is_2_5_l3871_387181

/-- Represents the food bank scenario --/
structure FoodBank where
  initial_stock : ℕ
  day1_people : ℕ
  day1_cans_per_person : ℕ
  day1_restock : ℕ
  day2_people : ℕ
  day2_restock : ℕ
  total_cans_given : ℕ

/-- Calculates the number of cans each person took on the second day --/
def cans_per_person_day2 (fb : FoodBank) : ℚ :=
  let day1_remaining := fb.initial_stock - fb.day1_people * fb.day1_cans_per_person
  let after_day1_restock := day1_remaining + fb.day1_restock
  let day2_given := fb.total_cans_given - fb.day1_people * fb.day1_cans_per_person
  day2_given / fb.day2_people

/-- Theorem stating that given the conditions, each person took 2.5 cans on the second day --/
theorem cans_per_person_day2_is_2_5 (fb : FoodBank)
  (h1 : fb.initial_stock = 2000)
  (h2 : fb.day1_people = 500)
  (h3 : fb.day1_cans_per_person = 1)
  (h4 : fb.day1_restock = 1500)
  (h5 : fb.day2_people = 1000)
  (h6 : fb.day2_restock = 3000)
  (h7 : fb.total_cans_given = 2500) :
  cans_per_person_day2 fb = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_cans_per_person_day2_is_2_5_l3871_387181


namespace NUMINAMATH_CALUDE_quadratic_rational_solutions_l3871_387151

/-- A function that checks if a number is a perfect square -/
def is_perfect_square (n : ℤ) : Prop := ∃ m : ℤ, n = m * m

/-- The discriminant of the quadratic equation kx^2 - 24x + 4k = 0 -/
def discriminant (k : ℤ) : ℤ := 576 - 16 * k * k

/-- The property that k is a valid solution -/
def is_valid_k (k : ℤ) : Prop :=
  k > 0 ∧ is_perfect_square (discriminant k)

theorem quadratic_rational_solutions :
  ∀ k : ℤ, is_valid_k k ↔ k = 3 ∨ k = 6 := by sorry

end NUMINAMATH_CALUDE_quadratic_rational_solutions_l3871_387151


namespace NUMINAMATH_CALUDE_min_occupied_seats_for_150_l3871_387164

/-- Given a row of seats, calculates the minimum number of occupied seats
    required to ensure the next person must sit next to someone. -/
def min_occupied_seats (total_seats : ℕ) : ℕ :=
  (total_seats + 2) / 4

theorem min_occupied_seats_for_150 :
  min_occupied_seats 150 = 37 := by
  sorry

#eval min_occupied_seats 150

end NUMINAMATH_CALUDE_min_occupied_seats_for_150_l3871_387164


namespace NUMINAMATH_CALUDE_pie_arrangement_l3871_387148

/-- Given the number of pecan and apple pies, calculates the number of complete rows when arranged with a fixed number of pies per row. -/
def calculate_rows (pecan_pies apple_pies pies_per_row : ℕ) : ℕ :=
  (pecan_pies + apple_pies) / pies_per_row

/-- Theorem stating that with 16 pecan pies and 14 apple pies, arranged in rows of 5 pies each, there will be 6 complete rows. -/
theorem pie_arrangement : calculate_rows 16 14 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_pie_arrangement_l3871_387148


namespace NUMINAMATH_CALUDE_complement_union_eq_specific_set_l3871_387111

open Set

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {2, 3, 5}
def N : Set ℕ := {4, 5}

theorem complement_union_eq_specific_set :
  (U \ (M ∪ N)) = {1, 6} := by sorry

end NUMINAMATH_CALUDE_complement_union_eq_specific_set_l3871_387111


namespace NUMINAMATH_CALUDE_airplane_seats_multiple_l3871_387159

theorem airplane_seats_multiple (total_seats first_class_seats : ℕ) 
  (h1 : total_seats = 387)
  (h2 : first_class_seats = 77)
  (h3 : ∃ m : ℕ, total_seats = first_class_seats + (m * first_class_seats + 2)) :
  ∃ m : ℕ, m = 4 ∧ total_seats = first_class_seats + (m * first_class_seats + 2) :=
by sorry

end NUMINAMATH_CALUDE_airplane_seats_multiple_l3871_387159


namespace NUMINAMATH_CALUDE_line_parallel_perpendicular_implies_planes_perpendicular_l3871_387190

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (planesPerpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_parallel_perpendicular_implies_planes_perpendicular
  (l : Line) (α β : Plane) (h_distinct : α ≠ β)
  (h_parallel : parallel l α) (h_perpendicular : perpendicular l β) :
  planesPerpendicular α β :=
sorry

end NUMINAMATH_CALUDE_line_parallel_perpendicular_implies_planes_perpendicular_l3871_387190


namespace NUMINAMATH_CALUDE_number_manipulation_l3871_387115

theorem number_manipulation (x : ℝ) : 
  (x - 34) / 10 = 2 → (x - 5) / 7 = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_manipulation_l3871_387115


namespace NUMINAMATH_CALUDE_total_tickets_sold_l3871_387154

def total_revenue : ℕ := 1933
def student_ticket_price : ℕ := 2
def nonstudent_ticket_price : ℕ := 3
def student_tickets_sold : ℕ := 530

theorem total_tickets_sold :
  ∃ (nonstudent_tickets : ℕ),
    student_tickets_sold * student_ticket_price +
    nonstudent_tickets * nonstudent_ticket_price = total_revenue ∧
    student_tickets_sold + nonstudent_tickets = 821 :=
by sorry

end NUMINAMATH_CALUDE_total_tickets_sold_l3871_387154


namespace NUMINAMATH_CALUDE_xunzi_wangzhi_interpretation_l3871_387100

/-- Represents the four seasonal agricultural activities -/
inductive SeasonalActivity
| SpringPlowing
| SummerWeeding
| AutumnHarvesting
| WinterStoring

/-- Represents the result of following the seasonal activities -/
def SurplusFood : Prop := True

/-- Represents the concept of objective laws in nature -/
def ObjectiveLaw : Prop := True

/-- Represents the concept of subjective initiative -/
def SubjectiveInitiative : Prop := True

/-- Represents the concept of expected results -/
def ExpectedResults : Prop := True

/-- The main theorem based on the given problem -/
theorem xunzi_wangzhi_interpretation 
  (seasonal_activities : List SeasonalActivity)
  (follow_activities_lead_to_surplus : seasonal_activities.length = 4 → SurplusFood) :
  (ObjectiveLaw → ExpectedResults) ∧ 
  (SubjectiveInitiative → ObjectiveLaw) := by
  sorry


end NUMINAMATH_CALUDE_xunzi_wangzhi_interpretation_l3871_387100


namespace NUMINAMATH_CALUDE_product_trailing_zeros_l3871_387136

/-- The number of trailing zeros in a natural number -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- The product of 15, 360, and 125 -/
def product : ℕ := 15 * 360 * 125

theorem product_trailing_zeros : trailingZeros product = 3 := by sorry

end NUMINAMATH_CALUDE_product_trailing_zeros_l3871_387136


namespace NUMINAMATH_CALUDE_arithmetic_progression_problem_l3871_387188

def arithmetic_progression (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem arithmetic_progression_problem (a₁ d : ℝ) :
  (arithmetic_progression a₁ d 13 = 3 * arithmetic_progression a₁ d 3) ∧
  (arithmetic_progression a₁ d 18 = 2 * arithmetic_progression a₁ d 7 + 8) →
  d = 4 ∧ a₁ = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_problem_l3871_387188


namespace NUMINAMATH_CALUDE_deepak_age_l3871_387141

theorem deepak_age (arun_age deepak_age : ℕ) : 
  (arun_age : ℚ) / deepak_age = 4 / 3 →
  arun_age + 6 = 26 →
  deepak_age = 15 := by
  sorry

end NUMINAMATH_CALUDE_deepak_age_l3871_387141


namespace NUMINAMATH_CALUDE_min_sum_a_b_l3871_387106

theorem min_sum_a_b (a b : ℕ+) (l : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + l = 0 ∧ a * x₂^2 + b * x₂ + l = 0) →
  (∀ x : ℝ, a * x^2 + b * x + l = 0 → abs x < 1) →
  (∀ c d : ℕ+, c + d < a + b → ¬(∃ y : ℝ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    c * x₁^2 + d * x₁ + y = 0 ∧ c * x₂^2 + d * x₂ + y = 0) ∧
    (∀ x : ℝ, c * x^2 + d * x + y = 0 → abs x < 1))) →
  a + b = 10 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_a_b_l3871_387106


namespace NUMINAMATH_CALUDE_triangle_isosceles_l3871_387176

theorem triangle_isosceles (A B C : Real) (h1 : A + B + C = Real.pi) 
  (h2 : Real.sin C = 2 * Real.cos A * Real.sin B) : A = B := by
  sorry

end NUMINAMATH_CALUDE_triangle_isosceles_l3871_387176


namespace NUMINAMATH_CALUDE_two_digit_number_property_l3871_387118

theorem two_digit_number_property (a b : ℕ) : 
  b = 2 * a →
  10 * a + b - (10 * b + a) = 36 →
  (a + b) - (b - a) = 8 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_property_l3871_387118


namespace NUMINAMATH_CALUDE_highest_probability_C_l3871_387184

-- Define the events and their probabilities
def prob_A : ℝ := 0.5
def prob_B : ℝ := 0.1
def prob_C : ℝ := 0.9

-- Theorem: C has the highest probability
theorem highest_probability_C : 
  prob_C > prob_A ∧ prob_C > prob_B :=
by sorry

end NUMINAMATH_CALUDE_highest_probability_C_l3871_387184


namespace NUMINAMATH_CALUDE_luke_fillets_l3871_387142

/-- Calculates the total number of fish fillets Luke has after fishing for a given number of days -/
def total_fillets (fish_per_day : ℕ) (days : ℕ) (fillets_per_fish : ℕ) : ℕ :=
  fish_per_day * days * fillets_per_fish

/-- Proves that Luke has 120 fillets after fishing for 30 days, catching 2 fish per day, with 2 fillets per fish -/
theorem luke_fillets :
  total_fillets 2 30 2 = 120 := by
  sorry

end NUMINAMATH_CALUDE_luke_fillets_l3871_387142


namespace NUMINAMATH_CALUDE_network_connections_l3871_387123

/-- The number of unique connections in a network of switches -/
def num_connections (n : ℕ) (k : ℕ) : ℕ :=
  (n * k) / 2

/-- Theorem: In a network of 30 switches, where each switch is directly connected 
    to exactly 4 other switches, the total number of unique connections is 60 -/
theorem network_connections : num_connections 30 4 = 60 := by
  sorry

end NUMINAMATH_CALUDE_network_connections_l3871_387123


namespace NUMINAMATH_CALUDE_students_without_A_l3871_387169

theorem students_without_A (total : ℕ) (science_A : ℕ) (english_A : ℕ) (both_A : ℕ) : 
  total - (science_A + english_A - both_A) = 18 :=
by
  sorry

#check students_without_A 40 10 18 6

end NUMINAMATH_CALUDE_students_without_A_l3871_387169


namespace NUMINAMATH_CALUDE_batsman_average_l3871_387105

/-- Represents a batsman's performance -/
structure Batsman where
  innings : ℕ
  runsLastInning : ℕ
  averageIncrease : ℕ

/-- Calculates the new average of a batsman after their last inning -/
def newAverage (b : Batsman) : ℕ :=
  b.averageIncrease + (b.innings - 1) * b.averageIncrease + b.runsLastInning

/-- Theorem stating that given the conditions, the batsman's new average is 140 -/
theorem batsman_average (b : Batsman) 
  (h1 : b.innings = 17) 
  (h2 : b.runsLastInning = 300) 
  (h3 : b.averageIncrease = 10) : 
  newAverage b = 140 := by
  sorry


end NUMINAMATH_CALUDE_batsman_average_l3871_387105


namespace NUMINAMATH_CALUDE_rectangle_width_equals_six_l3871_387119

theorem rectangle_width_equals_six (square_side : ℝ) (rect_length : ℝ) (rect_width : ℝ) : 
  square_side = 12 →
  rect_length = 24 →
  square_side * square_side = rect_length * rect_width →
  rect_width = 6 := by
sorry

end NUMINAMATH_CALUDE_rectangle_width_equals_six_l3871_387119


namespace NUMINAMATH_CALUDE_angle_B_is_pi_over_four_max_area_when_b_is_two_max_area_equality_condition_l3871_387177

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangleCondition (t : Triangle) : Prop :=
  t.a = t.b * Real.cos t.C + t.c * Real.sin t.B

-- Theorem for part 1
theorem angle_B_is_pi_over_four (t : Triangle) (h : triangleCondition t) :
  t.B = π / 4 := by sorry

-- Theorem for part 2
theorem max_area_when_b_is_two (t : Triangle) (h1 : triangleCondition t) (h2 : t.b = 2) :
  (1 / 2) * t.a * t.c * Real.sin t.B ≤ Real.sqrt 2 + 1 := by sorry

-- Theorem for equality condition in part 2
theorem max_area_equality_condition (t : Triangle) (h1 : triangleCondition t) (h2 : t.b = 2) :
  (1 / 2) * t.a * t.c * Real.sin t.B = Real.sqrt 2 + 1 ↔ t.a = t.c := by sorry

end NUMINAMATH_CALUDE_angle_B_is_pi_over_four_max_area_when_b_is_two_max_area_equality_condition_l3871_387177


namespace NUMINAMATH_CALUDE_worker_savings_l3871_387129

theorem worker_savings (P : ℝ) (h : P > 0) :
  let monthly_savings := (1/3) * P
  let monthly_not_saved := (2/3) * P
  let yearly_savings := 12 * monthly_savings
  yearly_savings = 6 * monthly_not_saved :=
by sorry

end NUMINAMATH_CALUDE_worker_savings_l3871_387129


namespace NUMINAMATH_CALUDE_solve_for_a_l3871_387113

-- Define the function f
def f (x a : ℝ) : ℝ := |x + 1| + |x - a|

-- State the theorem
theorem solve_for_a (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ x, f x a ≥ 5 ↔ x ≤ -2 ∨ x > 3) : a = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l3871_387113


namespace NUMINAMATH_CALUDE_greatest_NPMPP_l3871_387163

/-- A function that checks if a number's square ends with the number itself -/
def endsWithSelf (n : Nat) : Prop :=
  n % 10 = (n * n) % 10

/-- A function that generates a four-digit number with all identical digits -/
def fourIdenticalDigits (d : Nat) : Nat :=
  d * 1000 + d * 100 + d * 10 + d

/-- The theorem stating the greatest possible value of NPMPP -/
theorem greatest_NPMPP : 
  ∃ (M : Nat), 
    M ≤ 9 ∧ 
    endsWithSelf M ∧ 
    ∀ (N : Nat), N ≤ 9 → endsWithSelf N → M ≥ N ∧
    fourIdenticalDigits M * M = 89991 :=
sorry

end NUMINAMATH_CALUDE_greatest_NPMPP_l3871_387163


namespace NUMINAMATH_CALUDE_emily_score_calculation_l3871_387122

/-- Emily's trivia game score calculation -/
theorem emily_score_calculation 
  (first_round : ℕ) 
  (second_round : ℕ) 
  (final_score : ℕ) 
  (h1 : first_round = 16) 
  (h2 : second_round = 33) 
  (h3 : final_score = 1) : 
  (first_round + second_round) - final_score = 48 := by
  sorry

end NUMINAMATH_CALUDE_emily_score_calculation_l3871_387122


namespace NUMINAMATH_CALUDE_complex_equation_implies_sum_l3871_387196

theorem complex_equation_implies_sum (x y : ℝ) :
  (x + y : ℂ) + (y - 1) * I = (2 * x + 3 * y : ℂ) + (2 * y + 1) * I →
  x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_implies_sum_l3871_387196


namespace NUMINAMATH_CALUDE_intersection_P_Q_l3871_387125

def P : Set ℤ := {x | |x - 1| < 2}
def Q : Set ℤ := {x | -1 ≤ x ∧ x ≤ 2}

theorem intersection_P_Q : P ∩ Q = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l3871_387125


namespace NUMINAMATH_CALUDE_sweeties_remainder_l3871_387198

theorem sweeties_remainder (m : ℕ) (h : m % 6 = 4) : (2 * m) % 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sweeties_remainder_l3871_387198


namespace NUMINAMATH_CALUDE_product_of_primes_l3871_387189

theorem product_of_primes : 
  ∃ (p q : Nat), 
    Prime p ∧ 
    Prime q ∧ 
    p = 1021031 ∧ 
    q = 237019 ∧ 
    p * q = 241940557349 := by
  sorry

end NUMINAMATH_CALUDE_product_of_primes_l3871_387189


namespace NUMINAMATH_CALUDE_triangle_packing_l3871_387191

/-- Represents an equilateral triangle with side length L -/
structure EquilateralTriangle (L : ℝ) where
  sideLength : L > 0

/-- Represents a configuration of unit equilateral triangles inside a larger triangle -/
structure TriangleConfiguration (L : ℝ) where
  largeTriangle : EquilateralTriangle L
  numUnitTriangles : ℕ
  nonOverlapping : Bool
  parallelSides : Bool
  oppositeOrientation : Bool

/-- The theorem statement -/
theorem triangle_packing (L : ℝ) (config : TriangleConfiguration L) :
  config.nonOverlapping ∧ config.parallelSides ∧ config.oppositeOrientation →
  (config.numUnitTriangles : ℝ) ≤ (2 / 3) * L^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_packing_l3871_387191


namespace NUMINAMATH_CALUDE_baseball_league_games_l3871_387128

theorem baseball_league_games (N M : ℕ) : 
  N > M →
  M > 5 →
  4 * N + 5 * M = 90 →
  4 * N = 60 := by
sorry

end NUMINAMATH_CALUDE_baseball_league_games_l3871_387128


namespace NUMINAMATH_CALUDE_ratio_of_fraction_equation_l3871_387192

theorem ratio_of_fraction_equation (P Q : ℤ) : 
  (∀ x : ℝ, x ≠ 3 ∧ x ≠ -3 ∧ x ≠ 2 ∧ x ≠ 0 → 
    (P / (x - 3) + Q / (x^2 + x - 6) = (x^2 + 3*x + 1) / (x^3 - x^2 - 12*x))) →
  Q / P = -6 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_fraction_equation_l3871_387192


namespace NUMINAMATH_CALUDE_inequality_proof_l3871_387102

theorem inequality_proof (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_one : a + b + c = 1) : 
  (1 / (b*c + a + 1/a)) + (1 / (a*c + b + 1/b)) + (1 / (a*b + c + 1/c)) ≤ 27/31 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3871_387102


namespace NUMINAMATH_CALUDE_line_perp_parallel_implies_planes_perp_l3871_387140

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)

-- Define the perpendicular relationship between planes
variable (perpendicular_planes : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_parallel_implies_planes_perp
  (l : Line) (α β : Plane) :
  perpendicular l α → parallel l β → perpendicular_planes α β :=
sorry

end NUMINAMATH_CALUDE_line_perp_parallel_implies_planes_perp_l3871_387140


namespace NUMINAMATH_CALUDE_choral_group_max_size_l3871_387199

theorem choral_group_max_size :
  ∀ (n s : ℕ),
  (∃ (m : ℕ),
    m < 150 ∧
    n * s + 4 = m ∧
    (s - 3) * (n + 2) = m) →
  (∀ (m : ℕ),
    m < 150 ∧
    (∃ (x y : ℕ),
      x * y + 4 = m ∧
      (y - 3) * (x + 2) = m) →
    m ≤ 144) :=
by sorry

end NUMINAMATH_CALUDE_choral_group_max_size_l3871_387199


namespace NUMINAMATH_CALUDE_spring_festival_gala_arrangements_l3871_387126

theorem spring_festival_gala_arrangements :
  let original_programs : ℕ := 10
  let new_programs : ℕ := 3
  let available_spaces : ℕ := original_programs + 1 - 2  -- excluding first and last positions
  
  (available_spaces.choose new_programs) * (new_programs.factorial) = 990 :=
by
  sorry

end NUMINAMATH_CALUDE_spring_festival_gala_arrangements_l3871_387126


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3871_387117

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |2*x - 3| + |x + 3| < 7} = Set.Icc (-1) (7/3) := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3871_387117


namespace NUMINAMATH_CALUDE_absolute_sum_zero_implies_values_l3871_387185

theorem absolute_sum_zero_implies_values (m n : ℝ) :
  |1 + m| + |n - 2| = 0 → m = -1 ∧ n = 2 ∧ m^n = 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_sum_zero_implies_values_l3871_387185


namespace NUMINAMATH_CALUDE_survival_probabilities_correct_l3871_387109

/-- Mortality table data -/
structure MortalityData :=
  (reach28 : ℕ)
  (reach35 : ℕ)
  (reach48 : ℕ)
  (reach55 : ℕ)
  (total : ℕ)

/-- Survival probabilities after 20 years -/
structure SurvivalProbabilities :=
  (bothAlive : ℚ)
  (husbandDead : ℚ)
  (wifeDead : ℚ)
  (bothDead : ℚ)
  (husbandDeadWifeAlive : ℚ)
  (husbandAliveWifeDead : ℚ)

/-- Calculate survival probabilities based on mortality data -/
def calculateSurvivalProbabilities (data : MortalityData) : SurvivalProbabilities :=
  sorry

/-- Theorem stating the correct survival probabilities -/
theorem survival_probabilities_correct (data : MortalityData) 
  (h1 : data.reach28 = 675)
  (h2 : data.reach35 = 630)
  (h3 : data.reach48 = 540)
  (h4 : data.reach55 = 486)
  (h5 : data.total = 1000) :
  let probs := calculateSurvivalProbabilities data
  probs.bothAlive = 108 / 175 ∧
  probs.husbandDead = 8 / 35 ∧
  probs.wifeDead = 1 / 5 ∧
  probs.bothDead = 8 / 175 ∧
  probs.husbandDeadWifeAlive = 32 / 175 ∧
  probs.husbandAliveWifeDead = 27 / 175 :=
by
  sorry

#check survival_probabilities_correct

end NUMINAMATH_CALUDE_survival_probabilities_correct_l3871_387109


namespace NUMINAMATH_CALUDE_arithmetic_sequence_constant_l3871_387194

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_constant
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_const : ∃ k : ℝ, a 2 + a 4 + a 15 = k) :
  ∃ c : ℝ, a 7 = c :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_constant_l3871_387194


namespace NUMINAMATH_CALUDE_train_passenger_count_l3871_387137

/-- Calculates the total number of passengers transported by a train -/
theorem train_passenger_count (one_way : ℕ) (return_way : ℕ) (additional_trips : ℕ) : 
  one_way = 100 → return_way = 60 → additional_trips = 3 → 
  (one_way + return_way) * (additional_trips + 1) = 640 := by
  sorry

end NUMINAMATH_CALUDE_train_passenger_count_l3871_387137


namespace NUMINAMATH_CALUDE_student_number_problem_l3871_387162

theorem student_number_problem (x : ℝ) : 4 * x - 138 = 102 → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l3871_387162


namespace NUMINAMATH_CALUDE_net_sales_for_10000_yuan_l3871_387153

/-- Represents the relationship between advertising expenses and sales revenue -/
def sales_model (x : ℝ) : ℝ := 8.5 * x + 17.5

/-- Calculates the net sales revenue given advertising expenses -/
def net_sales_revenue (x : ℝ) : ℝ := sales_model x - x

/-- Theorem: When advertising expenses are 1 (10,000 yuan), 
    the net sales revenue is 9.25 (92,500 yuan) -/
theorem net_sales_for_10000_yuan : net_sales_revenue 1 = 9.25 := by
  sorry

end NUMINAMATH_CALUDE_net_sales_for_10000_yuan_l3871_387153


namespace NUMINAMATH_CALUDE_rectangle_to_cylinders_volume_ratio_l3871_387145

/-- Given a rectangle with dimensions 6 × 10, prove that when rolled into two cylinders,
    the ratio of the larger cylinder volume to the smaller cylinder volume is 5/3. -/
theorem rectangle_to_cylinders_volume_ratio :
  let rectangle_width : ℝ := 6
  let rectangle_height : ℝ := 10
  let cylinder1_height : ℝ := rectangle_height
  let cylinder1_circumference : ℝ := rectangle_width
  let cylinder2_height : ℝ := rectangle_width
  let cylinder2_circumference : ℝ := rectangle_height
  let cylinder1_volume := π * (cylinder1_circumference / (2 * π))^2 * cylinder1_height
  let cylinder2_volume := π * (cylinder2_circumference / (2 * π))^2 * cylinder2_height
  let larger_volume := max cylinder1_volume cylinder2_volume
  let smaller_volume := min cylinder1_volume cylinder2_volume
  larger_volume / smaller_volume = 5 / 3 := by
sorry


end NUMINAMATH_CALUDE_rectangle_to_cylinders_volume_ratio_l3871_387145


namespace NUMINAMATH_CALUDE_negation_of_for_all_positive_negation_of_specific_quadratic_l3871_387160

theorem negation_of_for_all_positive (f : ℝ → ℝ) :
  (¬ ∀ x : ℝ, f x > 0) ↔ (∃ x : ℝ, f x ≤ 0) := by sorry

theorem negation_of_specific_quadratic :
  (¬ ∀ x : ℝ, 2 * x^2 - 3 * x + 4 > 0) ↔ (∃ x : ℝ, 2 * x^2 - 3 * x + 4 ≤ 0) := by
  apply negation_of_for_all_positive (fun x ↦ 2 * x^2 - 3 * x + 4)

end NUMINAMATH_CALUDE_negation_of_for_all_positive_negation_of_specific_quadratic_l3871_387160


namespace NUMINAMATH_CALUDE_janinas_pancakes_l3871_387178

-- Define the variables
def daily_rent : ℕ := 30
def daily_supplies : ℕ := 12
def price_per_pancake : ℕ := 2

-- Define the function to calculate the number of pancakes needed
def pancakes_needed (rent : ℕ) (supplies : ℕ) (price : ℕ) : ℕ :=
  (rent + supplies) / price

-- Theorem statement
theorem janinas_pancakes :
  pancakes_needed daily_rent daily_supplies price_per_pancake = 21 := by
sorry

end NUMINAMATH_CALUDE_janinas_pancakes_l3871_387178


namespace NUMINAMATH_CALUDE_solve_for_A_l3871_387180

theorem solve_for_A : ∃ A : ℤ, (2 * A - 6 + 4 = 26) ∧ A = 14 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_A_l3871_387180


namespace NUMINAMATH_CALUDE_books_sold_is_24_l3871_387183

/-- Calculates the number of books sold to buy a clarinet -/
def books_sold_for_clarinet (initial_savings : ℕ) (clarinet_cost : ℕ) (book_price : ℕ) : ℕ :=
  let additional_needed := clarinet_cost - initial_savings
  let halfway_savings := additional_needed / 2
  let total_to_save := halfway_savings + additional_needed
  total_to_save / book_price

theorem books_sold_is_24 :
  books_sold_for_clarinet 10 90 5 = 24 := by
  sorry

#eval books_sold_for_clarinet 10 90 5

end NUMINAMATH_CALUDE_books_sold_is_24_l3871_387183


namespace NUMINAMATH_CALUDE_parallelogram_acute_angle_iff_diagonal_equation_l3871_387143

/-- A parallelogram with side lengths a and b, and diagonal lengths m and n -/
structure Parallelogram where
  a : ℝ
  b : ℝ
  m : ℝ
  n : ℝ
  ha : a > 0
  hb : b > 0
  hm : m > 0
  hn : n > 0

/-- The acute angle of a parallelogram -/
def acute_angle (p : Parallelogram) : ℝ := sorry

theorem parallelogram_acute_angle_iff_diagonal_equation (p : Parallelogram) :
  p.a^4 + p.b^4 = p.m^2 * p.n^2 ↔ acute_angle p = π/4 := by sorry

end NUMINAMATH_CALUDE_parallelogram_acute_angle_iff_diagonal_equation_l3871_387143


namespace NUMINAMATH_CALUDE_complex_sum_theorem_l3871_387133

theorem complex_sum_theorem : 
  let i : ℂ := Complex.I
  let z₁ : ℂ := 2 + i
  let z₂ : ℂ := 1 - 2*i
  z₁ + z₂ = 3 - i := by sorry

end NUMINAMATH_CALUDE_complex_sum_theorem_l3871_387133


namespace NUMINAMATH_CALUDE_quadratic_inequality_product_l3871_387150

/-- Given a quadratic inequality ax^2 + bx + 1 > 0 with solution set (-1, 1/3), prove that ab = 6 -/
theorem quadratic_inequality_product (a b : ℝ) : 
  (∀ x, -1 < x ∧ x < 1/3 ↔ a * x^2 + b * x + 1 > 0) →
  a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_product_l3871_387150


namespace NUMINAMATH_CALUDE_square_root_equality_l3871_387138

theorem square_root_equality (k : ℕ) (h : k > 0) :
  (∀ (i : ℕ), i > 0 → Real.sqrt (i + i / (i^2 - 1)) = i * Real.sqrt (i / (i^2 - 1))) →
  (let n : ℝ := 6
   let m : ℝ := 35
   Real.sqrt (6 + n / m) = 6 * Real.sqrt (n / m) ∧ m + n = 41) := by
sorry

end NUMINAMATH_CALUDE_square_root_equality_l3871_387138


namespace NUMINAMATH_CALUDE_integer_between_sqrt_7_and_sqrt_15_l3871_387112

theorem integer_between_sqrt_7_and_sqrt_15 (a : ℤ) :
  (Real.sqrt 7 < a) ∧ (a < Real.sqrt 15) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_integer_between_sqrt_7_and_sqrt_15_l3871_387112


namespace NUMINAMATH_CALUDE_part_to_whole_ratio_l3871_387120

theorem part_to_whole_ratio (N P : ℚ) (h1 : N = 160) (h2 : (1/5) * N + 4 = P - 4) :
  (P - 4) / N = 9 / 40 := by
  sorry

end NUMINAMATH_CALUDE_part_to_whole_ratio_l3871_387120


namespace NUMINAMATH_CALUDE_p_money_calculation_l3871_387146

theorem p_money_calculation (p q r : ℝ) 
  (h1 : p = (1/7 * p + 1/7 * p) + 35)
  (h2 : q = 1/7 * p) 
  (h3 : r = 1/7 * p) : 
  p = 49 := by
  sorry

end NUMINAMATH_CALUDE_p_money_calculation_l3871_387146


namespace NUMINAMATH_CALUDE_factorial_equation_solution_l3871_387121

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem factorial_equation_solution (N : ℕ) (h : N > 0) :
  factorial 5 * factorial 9 = 12 * factorial N → N = 10 := by
  sorry

end NUMINAMATH_CALUDE_factorial_equation_solution_l3871_387121


namespace NUMINAMATH_CALUDE_inverse_square_relation_l3871_387175

/-- Given that x varies inversely as the square of y and y = 3 when x = 1,
    prove that x = 1/9 when y = 9. -/
theorem inverse_square_relation (x y : ℝ) (k : ℝ) : 
  (∀ (x y : ℝ), x = k / (y^2)) →  -- x varies inversely as square of y
  (1 = k / (3^2)) →               -- y = 3 when x = 1
  (x = k / (9^2)) →               -- condition for y = 9
  x = 1/9 := by
sorry

end NUMINAMATH_CALUDE_inverse_square_relation_l3871_387175


namespace NUMINAMATH_CALUDE_prism_volume_l3871_387195

/-- 
A right rectangular prism with one side length of 4 inches, 
and two faces with areas of 24 and 16 square inches respectively, 
has a volume of 64 cubic inches.
-/
theorem prism_volume : 
  ∀ (x y z : ℝ), 
  x = 4 → 
  x * y = 24 → 
  y * z = 16 → 
  x * y * z = 64 := by
sorry

end NUMINAMATH_CALUDE_prism_volume_l3871_387195


namespace NUMINAMATH_CALUDE_boat_distance_theorem_l3871_387144

/-- Calculates the distance traveled downstream by a boat -/
def distance_downstream (boat_speed : ℝ) (stream_speed : ℝ) (time : ℝ) : ℝ :=
  (boat_speed + stream_speed) * time

/-- Proves that a boat traveling downstream for 5 hours covers 125 km -/
theorem boat_distance_theorem (boat_speed : ℝ) (stream_speed : ℝ) (time : ℝ) 
  (h1 : boat_speed = 20)
  (h2 : stream_speed = 5)
  (h3 : time = 5) :
  distance_downstream boat_speed stream_speed time = 125 := by
  sorry

#check boat_distance_theorem

end NUMINAMATH_CALUDE_boat_distance_theorem_l3871_387144


namespace NUMINAMATH_CALUDE_correct_assignment_l3871_387134

-- Define the colors and labels
inductive Color : Type
| White : Color
| Red : Color
| Yellow : Color
| Green : Color

def Label := Color

-- Define a package as a pair of label and actual color
structure Package where
  label : Label
  actual : Color

-- Define the condition that no label matches its actual content
def labelMismatch (p : Package) : Prop := p.label ≠ p.actual

-- Define the set of all packages
def allPackages : Finset Package := sorry

-- Define the property that all labels are different
def allLabelsDifferent (packages : Finset Package) : Prop := sorry

-- Define the property that all actual colors are different
def allActualColorsDifferent (packages : Finset Package) : Prop := sorry

-- Main theorem
theorem correct_assignment :
  ∀ (packages : Finset Package),
    packages = allPackages →
    (∀ p ∈ packages, labelMismatch p) →
    allLabelsDifferent packages →
    allActualColorsDifferent packages →
    ∃! (w r y g : Package),
      w ∈ packages ∧ r ∈ packages ∧ y ∈ packages ∧ g ∈ packages ∧
      w.label = Color.Red ∧ w.actual = Color.White ∧
      r.label = Color.White ∧ r.actual = Color.Red ∧
      y.label = Color.Green ∧ y.actual = Color.Yellow ∧
      g.label = Color.Yellow ∧ g.actual = Color.Green :=
by sorry

end NUMINAMATH_CALUDE_correct_assignment_l3871_387134


namespace NUMINAMATH_CALUDE_root_sum_product_l3871_387107

/-- Given two polynomials with specified roots, prove that u = 32 -/
theorem root_sum_product (α β γ : ℂ) (q s u : ℂ) : 
  (α^3 + 4*α^2 + 6*α - 8 = 0) → 
  (β^3 + 4*β^2 + 6*β - 8 = 0) → 
  (γ^3 + 4*γ^2 + 6*γ - 8 = 0) →
  ((α+β)^3 + q*(α+β)^2 + s*(α+β) + u = 0) →
  ((β+γ)^3 + q*(β+γ)^2 + s*(β+γ) + u = 0) →
  ((γ+α)^3 + q*(γ+α)^2 + s*(γ+α) + u = 0) →
  u = 32 := by
sorry

end NUMINAMATH_CALUDE_root_sum_product_l3871_387107


namespace NUMINAMATH_CALUDE_difference_product_sum_equals_difference_of_squares_l3871_387149

theorem difference_product_sum_equals_difference_of_squares (a b : ℝ) :
  (a - b) * (b + a) = a^2 - b^2 := by
  sorry

end NUMINAMATH_CALUDE_difference_product_sum_equals_difference_of_squares_l3871_387149


namespace NUMINAMATH_CALUDE_sum_of_cubes_l3871_387158

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 7) (h2 : x * y = 12) :
  x^3 + y^3 = 91 := by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l3871_387158


namespace NUMINAMATH_CALUDE_seven_digit_divisible_by_11_l3871_387108

/-- A seven-digit number in the form 945k317 is divisible by 11 if and only if k = 8 -/
theorem seven_digit_divisible_by_11 (k : ℕ) : k < 10 → (945000 + k * 1000 + 317) % 11 = 0 ↔ k = 8 := by
  sorry

end NUMINAMATH_CALUDE_seven_digit_divisible_by_11_l3871_387108


namespace NUMINAMATH_CALUDE_tea_maker_capacity_l3871_387165

/-- A cylindrical tea maker with capacity x cups -/
structure TeaMaker where
  capacity : ℝ
  cylindrical : Bool

/-- Theorem: A cylindrical tea maker that contains 54 cups when 45% full has a total capacity of 120 cups -/
theorem tea_maker_capacity (tm : TeaMaker) (h1 : tm.cylindrical = true) 
    (h2 : 0.45 * tm.capacity = 54) : tm.capacity = 120 := by
  sorry

end NUMINAMATH_CALUDE_tea_maker_capacity_l3871_387165


namespace NUMINAMATH_CALUDE_test_questions_l3871_387139

theorem test_questions (points_correct : ℕ) (points_incorrect : ℕ) 
  (total_score : ℕ) (correct_answers : ℕ) :
  points_correct = 20 →
  points_incorrect = 5 →
  total_score = 325 →
  correct_answers = 19 →
  ∃ (total_questions : ℕ),
    total_questions = correct_answers + (total_questions - correct_answers) ∧
    total_score = points_correct * correct_answers - 
      points_incorrect * (total_questions - correct_answers) ∧
    total_questions = 30 := by
  sorry

end NUMINAMATH_CALUDE_test_questions_l3871_387139


namespace NUMINAMATH_CALUDE_rectangular_prism_parallel_edges_l3871_387124

/-- A rectangular prism with distinctly different dimensions -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ
  length_ne_width : length ≠ width
  length_ne_height : length ≠ height
  width_ne_height : width ≠ height

/-- The number of pairs of parallel edges in a rectangular prism -/
def parallel_edge_pairs (prism : RectangularPrism) : ℕ := 12

/-- Theorem: A rectangular prism with distinctly different dimensions has 12 pairs of parallel edges -/
theorem rectangular_prism_parallel_edges (prism : RectangularPrism) :
  parallel_edge_pairs prism = 12 := by sorry

end NUMINAMATH_CALUDE_rectangular_prism_parallel_edges_l3871_387124


namespace NUMINAMATH_CALUDE_three_digit_square_sum_numbers_l3871_387152

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a b c : ℕ),
    a ≠ 0 ∧
    a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧
    n = 100 * a + 10 * b + c ∧
    n = 11 * (a^2 + b^2 + c^2)

theorem three_digit_square_sum_numbers :
  {n : ℕ | is_valid_number n} = {550, 803} :=
by sorry

end NUMINAMATH_CALUDE_three_digit_square_sum_numbers_l3871_387152


namespace NUMINAMATH_CALUDE_road_project_solution_l3871_387186

/-- Road construction project parameters -/
structure RoadProject where
  total_length : ℝ
  small_eq_rate : ℝ
  large_eq_rate : ℝ
  large_eq_time_ratio : ℝ
  length_increase : ℝ
  small_eq_time_increase : ℝ
  large_eq_rate_decrease : ℝ
  large_eq_time_increase : ℝ → ℝ

/-- Theorem stating the correct small equipment usage time and the value of m -/
theorem road_project_solution (project : RoadProject)
  (h1 : project.total_length = 39000)
  (h2 : project.small_eq_rate = 30)
  (h3 : project.large_eq_rate = 60)
  (h4 : project.large_eq_time_ratio = 5/3)
  (h5 : project.length_increase = 9000)
  (h6 : project.small_eq_time_increase = 18)
  (h7 : project.large_eq_time_increase = λ m => 150 + 2*m) :
  ∃ (small_eq_time m : ℝ),
    small_eq_time = 300 ∧
    m = 5 ∧
    project.small_eq_rate * small_eq_time +
    project.large_eq_rate * (project.large_eq_time_ratio * small_eq_time) = project.total_length ∧
    project.small_eq_rate * (small_eq_time + project.small_eq_time_increase) +
    (project.large_eq_rate - m) * (project.large_eq_time_ratio * small_eq_time + project.large_eq_time_increase m) =
    project.total_length + project.length_increase :=
sorry

end NUMINAMATH_CALUDE_road_project_solution_l3871_387186


namespace NUMINAMATH_CALUDE_parabola_vertex_x_coordinate_l3871_387179

/-- The x-coordinate of the vertex of a parabola given three points it passes through -/
theorem parabola_vertex_x_coordinate 
  (a b c : ℝ) 
  (h1 : a * (-2)^2 + b * (-2) + c = 8)
  (h2 : a * 4^2 + b * 4 + c = 8)
  (h3 : a * 7^2 + b * 7 + c = 15) :
  let f := fun x => a * x^2 + b * x + c
  ∃ x₀, ∀ x, f x ≥ f x₀ ∧ x₀ = 1 :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_x_coordinate_l3871_387179


namespace NUMINAMATH_CALUDE_union_A_B_when_a_zero_complement_A_intersect_B_nonempty_iff_l3871_387187

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 2 < x ∧ x < a + 2}
def B (a : ℝ) : Set ℝ := {x | x^2 - (a + 2)*x + 2*a = 0}

-- Theorem 1
theorem union_A_B_when_a_zero :
  A 0 ∪ B 0 = {x | -2 < x ∧ x ≤ 2} := by sorry

-- Theorem 2
theorem complement_A_intersect_B_nonempty_iff (a : ℝ) :
  ((Set.univ \ A a) ∩ B a).Nonempty ↔ a ≤ 0 ∨ a ≥ 4 := by sorry

end NUMINAMATH_CALUDE_union_A_B_when_a_zero_complement_A_intersect_B_nonempty_iff_l3871_387187
