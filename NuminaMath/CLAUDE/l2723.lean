import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_roots_product_l2723_272307

theorem quadratic_roots_product (a b : ℝ) : 
  (∀ x : ℝ, x^2 - a*x + b = 0 → (x = a ∨ x = b)) → 
  a + b = 5 → 
  a * b = 6 → 
  a * b = 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_product_l2723_272307


namespace NUMINAMATH_CALUDE_sin_90_degrees_l2723_272335

theorem sin_90_degrees : 
  Real.sin (90 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_90_degrees_l2723_272335


namespace NUMINAMATH_CALUDE_problem_solution_l2723_272336

theorem problem_solution (a b c : ℝ) 
  (h1 : a + 2*b + 3*c = 12) 
  (h2 : a^2 + b^2 + c^2 = a*b + a*c + b*c) : 
  a + b^2 + c^3 = 14 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2723_272336


namespace NUMINAMATH_CALUDE_real_part_of_z_z_equals_1_plus_2i_purely_imaginary_l2723_272311

-- Define a complex number z as x + yi
def z (x y : ℝ) : ℂ := Complex.mk x y

-- Statement 1: The real part of z is x
theorem real_part_of_z (x y : ℝ) : (z x y).re = x := by sorry

-- Statement 2: If z = 1 + 2i, then x = 1 and y = 2
theorem z_equals_1_plus_2i (x y : ℝ) : 
  z x y = Complex.mk 1 2 → x = 1 ∧ y = 2 := by sorry

-- Statement 3: When x = 0 and y ≠ 0, z is a purely imaginary number
theorem purely_imaginary (y : ℝ) : 
  y ≠ 0 → (z 0 y).re = 0 ∧ (z 0 y).im ≠ 0 := by sorry

end NUMINAMATH_CALUDE_real_part_of_z_z_equals_1_plus_2i_purely_imaginary_l2723_272311


namespace NUMINAMATH_CALUDE_yellow_surface_fraction_l2723_272391

/-- Represents a large cube constructed from smaller cubes -/
structure LargeCube where
  edge_length : ℕ
  total_small_cubes : ℕ
  yellow_cubes : ℕ
  blue_cubes : ℕ

/-- Calculates the minimum possible yellow surface area for a given large cube configuration -/
def min_yellow_surface_area (cube : LargeCube) : ℚ :=
  sorry

/-- Calculates the total surface area of the large cube -/
def total_surface_area (cube : LargeCube) : ℚ :=
  sorry

/-- The main theorem to prove -/
theorem yellow_surface_fraction (cube : LargeCube) 
  (h1 : cube.edge_length = 4)
  (h2 : cube.total_small_cubes = 64)
  (h3 : cube.yellow_cubes = 14)
  (h4 : cube.blue_cubes = 50)
  (h5 : cube.yellow_cubes + cube.blue_cubes = cube.total_small_cubes) :
  (min_yellow_surface_area cube) / (total_surface_area cube) = 7 / 48 :=
sorry

end NUMINAMATH_CALUDE_yellow_surface_fraction_l2723_272391


namespace NUMINAMATH_CALUDE_sum_of_combined_sequence_l2723_272312

/-- Given geometric sequence {aₙ} and arithmetic sequence {bₙ} -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

def arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

/-- Theorem stating the sum of the first n terms of the sequence cₙ = aₙ + bₙ -/
theorem sum_of_combined_sequence
  (a b : ℕ → ℝ)
  (h_a : geometric_sequence a)
  (h_b : arithmetic_sequence b)
  (h_a1 : a 1 = 1)
  (h_a4 : a 4 = 8)
  (h_b1 : b 1 = 3)
  (h_b4 : b 4 = 12) :
  ∃ S : ℕ → ℝ, ∀ n : ℕ, S n = 2^n - 1 + (3/2 * n^2) + (3/2 * n) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_combined_sequence_l2723_272312


namespace NUMINAMATH_CALUDE_wills_initial_amount_l2723_272399

/-- The amount of money Will's mom gave him initially -/
def initial_amount : ℕ := 74

/-- The cost of the sweater Will bought -/
def sweater_cost : ℕ := 9

/-- The cost of the T-shirt Will bought -/
def tshirt_cost : ℕ := 11

/-- The cost of the shoes Will bought -/
def shoes_cost : ℕ := 30

/-- The refund percentage for the returned shoes -/
def refund_percentage : ℚ := 90 / 100

/-- The amount of money Will has left after all transactions -/
def money_left : ℕ := 51

theorem wills_initial_amount :
  initial_amount = 
    money_left + 
    sweater_cost + 
    tshirt_cost + 
    shoes_cost - 
    (↑shoes_cost * refund_percentage).floor :=
by sorry

end NUMINAMATH_CALUDE_wills_initial_amount_l2723_272399


namespace NUMINAMATH_CALUDE_investment_growth_l2723_272332

/-- Represents the investment growth over a two-year period -/
theorem investment_growth 
  (initial_investment : ℝ) 
  (final_investment : ℝ) 
  (growth_rate : ℝ) 
  (h1 : initial_investment = 800) 
  (h2 : final_investment = 960) 
  (h3 : initial_investment * (1 + growth_rate)^2 = final_investment) : 
  800 * (1 + growth_rate)^2 = 960 :=
by sorry

end NUMINAMATH_CALUDE_investment_growth_l2723_272332


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2723_272329

theorem polynomial_factorization (x y : ℝ) : 
  4 * x^2 - 4 * x - y^2 + 4 * y - 3 = (2 * x + y - 3) * (2 * x - y + 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2723_272329


namespace NUMINAMATH_CALUDE_trigonometric_identities_l2723_272370

theorem trigonometric_identities (x : Real) 
  (h1 : -π < x ∧ x < 0) 
  (h2 : Real.sin x + Real.cos x = 1/5) : 
  (Real.sin x - Real.cos x = -7/5) ∧ 
  ((3 * (Real.sin (x/2))^2 - 2 * Real.sin (x/2) * Real.cos (x/2) + (Real.cos (x/2))^2) / 
   (Real.tan x + 1 / Real.tan x) = -132/125) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l2723_272370


namespace NUMINAMATH_CALUDE_symmetry_coincidence_l2723_272318

-- Define the type for points in the plane
def Point : Type := ℝ × ℝ

-- Define the symmetry operation
def symmetric (A B O : Point) : Prop := 
  ∃ (x y : ℝ), A = (x, y) ∧ B = (2 * O.1 - x, 2 * O.2 - y)

-- Define the given points
variable (A A₁ A₂ A₃ A₄ A₅ A₆ O₁ O₂ O₃ : Point)

-- State the theorem
theorem symmetry_coincidence 
  (h1 : symmetric A A₁ O₁)
  (h2 : symmetric A₁ A₂ O₂)
  (h3 : symmetric A₂ A₃ O₃)
  (h4 : symmetric A₃ A₄ O₁)
  (h5 : symmetric A₄ A₅ O₂)
  (h6 : symmetric A₅ A₆ O₃) :
  A = A₆ := by sorry

end NUMINAMATH_CALUDE_symmetry_coincidence_l2723_272318


namespace NUMINAMATH_CALUDE_afternoon_and_evening_emails_l2723_272377

def morning_emails : ℕ := 4
def afternoon_emails : ℕ := 5
def evening_emails : ℕ := 8

theorem afternoon_and_evening_emails :
  afternoon_emails + evening_emails = 13 :=
by sorry

end NUMINAMATH_CALUDE_afternoon_and_evening_emails_l2723_272377


namespace NUMINAMATH_CALUDE_usual_walking_time_l2723_272364

theorem usual_walking_time (usual_speed : ℝ) (usual_time : ℝ) : 
  usual_speed > 0 → usual_time > 0 →
  (4 / 5 * usual_speed) * (usual_time + 10) = usual_speed * usual_time →
  usual_time = 40 := by
sorry

end NUMINAMATH_CALUDE_usual_walking_time_l2723_272364


namespace NUMINAMATH_CALUDE_tan_eleven_pi_fourths_l2723_272388

theorem tan_eleven_pi_fourths : Real.tan (11 * π / 4) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_eleven_pi_fourths_l2723_272388


namespace NUMINAMATH_CALUDE_combined_age_proof_l2723_272375

/-- Given that Hezekiah is 4 years old and Ryanne is 7 years older than Hezekiah,
    prove that their combined age is 15 years. -/
theorem combined_age_proof (hezekiah_age : ℕ) (ryanne_age : ℕ) : 
  hezekiah_age = 4 → 
  ryanne_age = hezekiah_age + 7 → 
  hezekiah_age + ryanne_age = 15 := by
sorry

end NUMINAMATH_CALUDE_combined_age_proof_l2723_272375


namespace NUMINAMATH_CALUDE_whale_third_hour_consumption_l2723_272348

/-- Represents the whale's plankton consumption during a feeding frenzy -/
def WhaleFeedingFrenzy (x : ℕ) : Prop :=
  let first_hour := x
  let second_hour := x + 3
  let third_hour := x + 6
  let fourth_hour := x + 9
  let fifth_hour := x + 12
  (first_hour + second_hour + third_hour + fourth_hour + fifth_hour = 450) ∧
  (third_hour = 90)

/-- Theorem stating that the whale consumes 90 kilos in the third hour -/
theorem whale_third_hour_consumption : ∃ x : ℕ, WhaleFeedingFrenzy x := by
  sorry

end NUMINAMATH_CALUDE_whale_third_hour_consumption_l2723_272348


namespace NUMINAMATH_CALUDE_hyperbola_condition_l2723_272339

theorem hyperbola_condition (m : ℝ) :
  m > 0 →
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  ∀ (x y : ℝ), x^2 / (2 + m) - y^2 / (1 + m) = 1 ↔ x^2 / a^2 - y^2 / b^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l2723_272339


namespace NUMINAMATH_CALUDE_garden_comparison_l2723_272324

-- Define the dimensions of the gardens
def chris_length : ℝ := 30
def chris_width : ℝ := 60
def jordan_length : ℝ := 35
def jordan_width : ℝ := 55

-- Define the area difference
def area_difference : ℝ := jordan_length * jordan_width - chris_length * chris_width

-- Define the perimeters
def chris_perimeter : ℝ := 2 * (chris_length + chris_width)
def jordan_perimeter : ℝ := 2 * (jordan_length + jordan_width)

-- Theorem statement
theorem garden_comparison :
  area_difference = 125 ∧ chris_perimeter = jordan_perimeter := by
  sorry

end NUMINAMATH_CALUDE_garden_comparison_l2723_272324


namespace NUMINAMATH_CALUDE_parabola_point_relationship_l2723_272376

/-- A parabola with equation y = x² - 4x - m -/
structure Parabola where
  m : ℝ

/-- A point on the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point lies on a parabola -/
def lies_on (p : Point) (para : Parabola) : Prop :=
  p.y = p.x^2 - 4*p.x - para.m

theorem parabola_point_relationship (para : Parabola) (A B C : Point)
    (hA : A.x = 2) (hB : B.x = -3) (hC : C.x = -1)
    (onA : lies_on A para) (onB : lies_on B para) (onC : lies_on C para) :
    A.y < C.y ∧ C.y < B.y := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_relationship_l2723_272376


namespace NUMINAMATH_CALUDE_money_distribution_l2723_272396

theorem money_distribution (a b c : ℤ) 
  (total : a + b + c = 500)
  (ac_sum : a + c = 200)
  (bc_sum : b + c = 360) :
  c = 60 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l2723_272396


namespace NUMINAMATH_CALUDE_science_fair_sophomores_fraction_l2723_272380

theorem science_fair_sophomores_fraction (s j n : ℕ) : 
  s > 0 → -- Ensure s is positive to avoid division by zero
  s = j → -- Equal number of sophomores and juniors
  j = n → -- Number of juniors equals number of seniors
  (4 * s / 5 : ℚ) / ((4 * s / 5 : ℚ) + (3 * j / 4 : ℚ) + (n / 3 : ℚ)) = 240 / 565 := by
  sorry

#check science_fair_sophomores_fraction

end NUMINAMATH_CALUDE_science_fair_sophomores_fraction_l2723_272380


namespace NUMINAMATH_CALUDE_smallest_perimeter_600_smallest_perimeter_144_l2723_272365

/-- A triangle with integer side lengths -/
structure IntTriangle where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

/-- The perimeter of a triangle -/
def perimeter (t : IntTriangle) : ℕ := t.a + t.b + t.c

/-- The product of side lengths of a triangle -/
def product (t : IntTriangle) : ℕ := t.a * t.b * t.c

theorem smallest_perimeter_600 :
  ∀ t : IntTriangle, product t = 600 →
  perimeter t ≥ perimeter ⟨10, 10, 6, sorry⟩ := by sorry

theorem smallest_perimeter_144 :
  ∀ t : IntTriangle, product t = 144 →
  perimeter t ≥ perimeter ⟨4, 6, 6, sorry⟩ := by sorry

end NUMINAMATH_CALUDE_smallest_perimeter_600_smallest_perimeter_144_l2723_272365


namespace NUMINAMATH_CALUDE_circle_area_ratio_l2723_272319

theorem circle_area_ratio : 
  ∀ (r₁ r₂ : ℝ), 
  r₁ > 0 → r₂ > r₁ →
  r₂ = 3 * r₁ →
  (π * r₂^2 - π * r₁^2) / (π * r₁^2) = 8 :=
by sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l2723_272319


namespace NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l2723_272309

def C : Finset Nat := {51, 53, 54, 56, 57}

def has_smallest_prime_factor (n : Nat) (s : Finset Nat) : Prop :=
  n ∈ s ∧ ∀ m ∈ s, (Nat.minFac n ≤ Nat.minFac m)

theorem smallest_prime_factor_in_C :
  has_smallest_prime_factor 54 C := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l2723_272309


namespace NUMINAMATH_CALUDE_evelyn_marbles_count_l2723_272321

def initial_marbles : ℕ := 95
def marbles_from_henry : ℕ := 9
def cards_bought : ℕ := 6

theorem evelyn_marbles_count :
  initial_marbles + marbles_from_henry = 104 :=
by sorry

end NUMINAMATH_CALUDE_evelyn_marbles_count_l2723_272321


namespace NUMINAMATH_CALUDE_spanish_books_count_l2723_272323

theorem spanish_books_count (total : ℕ) (english : ℕ) (french : ℕ) (italian : ℕ) (spanish : ℕ) :
  total = 280 ∧
  english = total / 5 ∧
  french = total / 7 ∧
  italian = total / 4 ∧
  spanish = total - (english + french + italian) →
  spanish = 114 := by
sorry

end NUMINAMATH_CALUDE_spanish_books_count_l2723_272323


namespace NUMINAMATH_CALUDE_correct_quadratic_equation_l2723_272342

theorem correct_quadratic_equation 
  (a b c : ℝ) 
  (h1 : ∃ c', (a * 7^2 + b * 7 + c' = 0) ∧ (a * 3^2 + b * 3 + c' = 0))
  (h2 : ∃ b', (a * (-12)^2 + b' * (-12) + c = 0) ∧ (a * 3^2 + b' * 3 + c = 0)) :
  a = 1 ∧ b = -10 ∧ c = -36 := by
sorry

end NUMINAMATH_CALUDE_correct_quadratic_equation_l2723_272342


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l2723_272386

theorem complex_expression_simplification (x : ℝ) : 
  x * (x * (x * (3 - x) - 5) + 12) + 2 = -x^4 + 3*x^3 - 5*x^2 + 12*x + 2 := by
sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l2723_272386


namespace NUMINAMATH_CALUDE_merchant_salt_problem_l2723_272374

theorem merchant_salt_problem (x : ℝ) : 
  (x > 0) →
  (x + 100 > x) →
  (x + 220 > x + 100) →
  (x / (x + 100) = (x + 100) / (x + 220)) →
  (x = 500) :=
by
  sorry

end NUMINAMATH_CALUDE_merchant_salt_problem_l2723_272374


namespace NUMINAMATH_CALUDE_arithmetic_equation_proof_l2723_272337

theorem arithmetic_equation_proof : 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 * 9 = 100 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equation_proof_l2723_272337


namespace NUMINAMATH_CALUDE_min_value_expression_l2723_272352

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 3) :
  a^2 + 8*a*b + 24*b^2 + 16*b*c + 6*c^2 ≥ 54 ∧
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ * b₀ * c₀ = 3 ∧
    a₀^2 + 8*a₀*b₀ + 24*b₀^2 + 16*b₀*c₀ + 6*c₀^2 = 54 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2723_272352


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2723_272302

open Set

-- Define sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 4}
def B : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = Icc (-1) 4 := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2723_272302


namespace NUMINAMATH_CALUDE_min_value_fraction_sum_l2723_272328

theorem min_value_fraction_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  1/x + 4/y ≥ 9 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + y₀ = 1 ∧ 1/x₀ + 4/y₀ = 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_sum_l2723_272328


namespace NUMINAMATH_CALUDE_rainfall_depth_calculation_l2723_272354

/-- Calculates the approximate rainfall depth given container dimensions and collected water depth -/
theorem rainfall_depth_calculation (container_side : ℝ) (container_height : ℝ) (water_depth : ℝ) 
  (h1 : container_side = 20)
  (h2 : container_height = 40)
  (h3 : water_depth = 10) : 
  ∃ (rainfall_depth : ℝ), abs (rainfall_depth - 12.7) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_rainfall_depth_calculation_l2723_272354


namespace NUMINAMATH_CALUDE_triangle_existence_implies_m_greater_than_six_l2723_272397

/-- The function f(x) = x^3 - 3x + m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + m

/-- Theorem: If there exists a triangle with side lengths f(a), f(b), and f(c) for a, b, c in [0,2], then m > 6 -/
theorem triangle_existence_implies_m_greater_than_six (m : ℝ) :
  (∃ (a b c : ℝ), 0 ≤ a ∧ a ≤ 2 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 2 ∧ 
    f m a + f m b > f m c ∧ 
    f m b + f m c > f m a ∧ 
    f m c + f m a > f m b) →
  m > 6 := by
sorry


end NUMINAMATH_CALUDE_triangle_existence_implies_m_greater_than_six_l2723_272397


namespace NUMINAMATH_CALUDE_price_of_pants_l2723_272359

theorem price_of_pants (total_cost shirt_price pants_price shoes_price : ℝ) : 
  total_cost = 340 →
  shirt_price = (3/4) * pants_price →
  shoes_price = pants_price + 10 →
  total_cost = shirt_price + pants_price + shoes_price →
  pants_price = 120 := by
sorry

end NUMINAMATH_CALUDE_price_of_pants_l2723_272359


namespace NUMINAMATH_CALUDE_all_semifinalists_advanced_no_semifinalists_eliminated_l2723_272334

/-- The number of semifinalists -/
def total_semifinalists : ℕ := 8

/-- The number of medal winners in the final round -/
def medal_winners : ℕ := 3

/-- The number of possible groups of medal winners -/
def possible_groups : ℕ := 56

/-- The number of semifinalists who advanced to the final round -/
def advanced_semifinalists : ℕ := total_semifinalists

theorem all_semifinalists_advanced :
  advanced_semifinalists = total_semifinalists ∧
  Nat.choose advanced_semifinalists medal_winners = possible_groups :=
by sorry

theorem no_semifinalists_eliminated :
  total_semifinalists - advanced_semifinalists = 0 :=
by sorry

end NUMINAMATH_CALUDE_all_semifinalists_advanced_no_semifinalists_eliminated_l2723_272334


namespace NUMINAMATH_CALUDE_no_solution_absolute_value_equation_l2723_272344

theorem no_solution_absolute_value_equation :
  ¬ ∃ x : ℝ, 3 * |x + 2| + 2 = 0 := by
sorry

end NUMINAMATH_CALUDE_no_solution_absolute_value_equation_l2723_272344


namespace NUMINAMATH_CALUDE_total_students_l2723_272300

theorem total_students (passed_first : ℕ) (passed_second : ℕ) (passed_both : ℕ) (failed_both : ℕ) 
  (h1 : passed_first = 60)
  (h2 : passed_second = 40)
  (h3 : passed_both = 20)
  (h4 : failed_both = 20) :
  passed_first + passed_second - passed_both + failed_both = 100 := by
  sorry

#check total_students

end NUMINAMATH_CALUDE_total_students_l2723_272300


namespace NUMINAMATH_CALUDE_interest_rate_difference_l2723_272345

/-- Given a principal amount, time period, and difference in interest earned between two simple interest rates, 
    this theorem proves that the difference between these rates is 5%. -/
theorem interest_rate_difference
  (principal : ℝ)
  (time : ℝ)
  (interest_diff : ℝ)
  (h1 : principal = 600)
  (h2 : time = 10)
  (h3 : interest_diff = 300) :
  let rate_diff := interest_diff / (principal * time / 100)
  rate_diff = 5 := by sorry

end NUMINAMATH_CALUDE_interest_rate_difference_l2723_272345


namespace NUMINAMATH_CALUDE_concentric_circles_area_ratio_l2723_272384

theorem concentric_circles_area_ratio :
  ∀ (r₁ r₂ : ℝ), r₁ > 0 → r₂ > 0 → r₂ = 3 * r₁ →
  (π * r₂^2 - π * r₁^2) / (π * r₁^2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_area_ratio_l2723_272384


namespace NUMINAMATH_CALUDE_two_valid_selections_l2723_272398

def numbers : List ℕ := [1, 2, 3, 4, 5]

def average (lst : List ℕ) : ℚ :=
  (lst.sum : ℚ) / lst.length

def validSelection (a b : ℕ) : Prop :=
  a ∈ numbers ∧ b ∈ numbers ∧ a ≠ b ∧
  average (numbers.filter (λ x => x ≠ a ∧ x ≠ b)) = average numbers

theorem two_valid_selections :
  (∃! (pair : ℕ × ℕ), validSelection pair.1 pair.2) ∨
  (∃! (pair1 pair2 : ℕ × ℕ), 
    validSelection pair1.1 pair1.2 ∧ 
    validSelection pair2.1 pair2.2 ∧ 
    pair1 ≠ pair2) :=
  sorry

end NUMINAMATH_CALUDE_two_valid_selections_l2723_272398


namespace NUMINAMATH_CALUDE_trig_ratio_simplification_l2723_272385

theorem trig_ratio_simplification :
  (Real.sin (40 * π / 180) + Real.sin (80 * π / 180)) /
  (Real.cos (40 * π / 180) + Real.cos (80 * π / 180)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_ratio_simplification_l2723_272385


namespace NUMINAMATH_CALUDE_rectangle_area_l2723_272353

theorem rectangle_area (L B : ℝ) (h1 : L - B = 23) (h2 : 2 * L + 2 * B = 166) : L * B = 1590 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2723_272353


namespace NUMINAMATH_CALUDE_smallest_positive_integer_ending_in_6_divisible_by_13_l2723_272369

/-- A number ends in 6 if it's of the form 10n + 6 for some integer n -/
def ends_in_6 (x : ℕ) : Prop := ∃ n : ℕ, x = 10 * n + 6

/-- A number is divisible by 13 if there exists an integer k such that x = 13k -/
def divisible_by_13 (x : ℕ) : Prop := ∃ k : ℕ, x = 13 * k

theorem smallest_positive_integer_ending_in_6_divisible_by_13 :
  (ends_in_6 26 ∧ divisible_by_13 26) ∧
  ∀ x : ℕ, 0 < x ∧ x < 26 → ¬(ends_in_6 x ∧ divisible_by_13 x) :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_ending_in_6_divisible_by_13_l2723_272369


namespace NUMINAMATH_CALUDE_not_rhombus_from_equal_adjacent_sides_l2723_272306

/-- A quadrilateral is a polygon with four sides -/
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

/-- A rhombus is a quadrilateral with all sides equal -/
def is_rhombus (q : Quadrilateral) : Prop :=
  ∀ i j : Fin 4, dist (q.vertices i) (q.vertices ((i + 1) % 4)) = 
                 dist (q.vertices j) (q.vertices ((j + 1) % 4))

/-- Two sides are adjacent if they share a vertex -/
def are_adjacent_sides (q : Quadrilateral) (i j : Fin 4) : Prop :=
  (j = (i + 1) % 4) ∨ (i = (j + 1) % 4)

/-- A pair of adjacent sides are equal -/
def has_equal_adjacent_sides (q : Quadrilateral) : Prop :=
  ∃ i j : Fin 4, are_adjacent_sides q i j ∧ 
    dist (q.vertices i) (q.vertices ((i + 1) % 4)) = 
    dist (q.vertices j) (q.vertices ((j + 1) % 4))

/-- The statement to be proved false -/
theorem not_rhombus_from_equal_adjacent_sides :
  ¬(∀ q : Quadrilateral, has_equal_adjacent_sides q → is_rhombus q) :=
sorry

end NUMINAMATH_CALUDE_not_rhombus_from_equal_adjacent_sides_l2723_272306


namespace NUMINAMATH_CALUDE_price_reduction_equation_l2723_272367

/-- Theorem: For an item with an original price of 289 yuan and a final price of 256 yuan
    after two consecutive price reductions, where x represents the average percentage
    reduction each time, the equation 289(1-x)^2 = 256 holds true. -/
theorem price_reduction_equation (x : ℝ) 
  (h1 : 0 ≤ x) (h2 : x < 1) : 289 * (1 - x)^2 = 256 := by
  sorry

#check price_reduction_equation

end NUMINAMATH_CALUDE_price_reduction_equation_l2723_272367


namespace NUMINAMATH_CALUDE_rabbit_population_growth_l2723_272358

theorem rabbit_population_growth (initial_rabbits new_rabbits : ℕ) 
  (h1 : initial_rabbits = 8) 
  (h2 : new_rabbits = 5) : 
  initial_rabbits + new_rabbits = 13 := by
  sorry

end NUMINAMATH_CALUDE_rabbit_population_growth_l2723_272358


namespace NUMINAMATH_CALUDE_range_of_m_l2723_272393

-- Define propositions p and q
def p (m : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*x + m ≤ 0

def q (m : ℝ) : Prop := ∀ x y : ℝ, x < y → (3-m)^x < (3-m)^y

-- Define the compound statements
def p_or_q (m : ℝ) : Prop := p m ∨ q m

def p_and_q (m : ℝ) : Prop := p m ∧ q m

-- State the theorem
theorem range_of_m : 
  ∀ m : ℝ, (p_or_q m ∧ ¬(p_and_q m)) → (1 < m ∧ m < 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2723_272393


namespace NUMINAMATH_CALUDE_bank_deposit_exceeds_500_first_day_exceeding_500_l2723_272357

def bank_deposit (n : ℕ) : ℚ :=
  3 * (3^n - 1) / 2

theorem bank_deposit_exceeds_500 :
  ∃ n : ℕ, bank_deposit n > 500 ∧ ∀ m : ℕ, m < n → bank_deposit m ≤ 500 :=
by
  sorry

theorem first_day_exceeding_500 :
  (∃ n : ℕ, bank_deposit n > 500 ∧ ∀ m : ℕ, m < n → bank_deposit m ≤ 500) →
  (∃ n : ℕ, n = 6 ∧ bank_deposit n > 500 ∧ ∀ m : ℕ, m < n → bank_deposit m ≤ 500) :=
by
  sorry

end NUMINAMATH_CALUDE_bank_deposit_exceeds_500_first_day_exceeding_500_l2723_272357


namespace NUMINAMATH_CALUDE_truncated_pyramid_sphere_area_relation_l2723_272343

/-- Given a regular n-gonal truncated pyramid circumscribed around a sphere:
    S1 is the area of the base surface,
    S2 is the area of the lateral surface,
    S is the total surface area,
    σ is the area of the polygon whose vertices are the tangential points of the sphere and the lateral faces of the pyramid.
    This theorem states that σS = 4S1S2 cos²(π/n). -/
theorem truncated_pyramid_sphere_area_relation (n : ℕ) (S1 S2 S σ : ℝ) :
  n ≥ 3 →
  S1 > 0 →
  S2 > 0 →
  S = S1 + S2 →
  σ > 0 →
  σ * S = 4 * S1 * S2 * (Real.cos (π / n : ℝ))^2 := by
  sorry

end NUMINAMATH_CALUDE_truncated_pyramid_sphere_area_relation_l2723_272343


namespace NUMINAMATH_CALUDE_area_of_trapezoid_TUVW_l2723_272310

/-- Represents a triangle in the problem -/
structure Triangle where
  isIsosceles : Bool
  area : ℝ

/-- Represents the large triangle XYZ -/
def XYZ : Triangle where
  isIsosceles := true
  area := 135

/-- Represents a small triangle -/
def SmallTriangle : Triangle where
  isIsosceles := true
  area := 3

/-- The number of small triangles in XYZ -/
def numSmallTriangles : ℕ := 9

/-- The number of small triangles in trapezoid TUVW -/
def numSmallTrianglesInTUVW : ℕ := 4

/-- The area of trapezoid TUVW -/
def areaTUVW : ℝ := numSmallTrianglesInTUVW * SmallTriangle.area

theorem area_of_trapezoid_TUVW : areaTUVW = 123 := by
  sorry

end NUMINAMATH_CALUDE_area_of_trapezoid_TUVW_l2723_272310


namespace NUMINAMATH_CALUDE_movies_to_watch_l2723_272333

/-- Given a series with 17 movies, if 7 movies have been watched,
    then the number of movies still to watch is 10. -/
theorem movies_to_watch (total_movies : ℕ) (watched_movies : ℕ) : 
  total_movies = 17 → watched_movies = 7 → total_movies - watched_movies = 10 := by
  sorry

end NUMINAMATH_CALUDE_movies_to_watch_l2723_272333


namespace NUMINAMATH_CALUDE_domain_of_sqrt_2cos_plus_1_l2723_272327

open Real

theorem domain_of_sqrt_2cos_plus_1 (x : ℝ) (k : ℤ) :
  (∃ y : ℝ, y = Real.sqrt (2 * Real.cos x + 1)) ↔ 
  (x ∈ Set.Icc (2 * Real.pi * k - 2 * Real.pi / 3) (2 * Real.pi * k + 2 * Real.pi / 3)) :=
sorry

end NUMINAMATH_CALUDE_domain_of_sqrt_2cos_plus_1_l2723_272327


namespace NUMINAMATH_CALUDE_ABABCDCD_square_theorem_l2723_272363

/-- Represents an 8-digit number in the form ABABCDCD -/
def ABABCDCD (A B C D : Nat) : Nat :=
  A * 10000000 + B * 1000000 + A * 100000 + B * 10000 + C * 1000 + D * 100 + C * 10 + D

/-- Checks if four numbers are distinct digits -/
def areDistinctDigits (A B C D : Nat) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10

/-- The main theorem stating that only two sets of digits satisfy the conditions -/
theorem ABABCDCD_square_theorem :
  ∀ A B C D : Nat,
    (ABABCDCD A B C D)^2 = ABABCDCD A B C D ∧ areDistinctDigits A B C D →
    ((A = 9 ∧ B = 7 ∧ C = 0 ∧ D = 4) ∨ (A = 8 ∧ B = 0 ∧ C = 2 ∧ D = 1)) :=
by sorry

end NUMINAMATH_CALUDE_ABABCDCD_square_theorem_l2723_272363


namespace NUMINAMATH_CALUDE_two_sevenths_as_unit_fractions_l2723_272373

theorem two_sevenths_as_unit_fractions : 
  ∃ (a b c : ℕ+), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (2 : ℚ) / 7 = 1 / a + 1 / b + 1 / c :=
sorry

end NUMINAMATH_CALUDE_two_sevenths_as_unit_fractions_l2723_272373


namespace NUMINAMATH_CALUDE_prime_exponent_assignment_l2723_272315

theorem prime_exponent_assignment (k : ℕ) (p : Fin k → ℕ) 
  (h_prime : ∀ i, Prime (p i)) 
  (h_distinct : ∀ i j, i ≠ j → p i ≠ p j) :
  (Finset.univ : Finset (Fin k → Fin k)).card = k ^ k :=
sorry

end NUMINAMATH_CALUDE_prime_exponent_assignment_l2723_272315


namespace NUMINAMATH_CALUDE_rope_knot_reduction_l2723_272390

theorem rope_knot_reduction 
  (total_length : ℝ) 
  (num_pieces : ℕ) 
  (tied_pieces : ℕ) 
  (final_length : ℝ) 
  (h1 : total_length = 72) 
  (h2 : num_pieces = 12) 
  (h3 : tied_pieces = 3) 
  (h4 : final_length = 15) : 
  (total_length / num_pieces * tied_pieces - final_length) / (tied_pieces - 1) = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_rope_knot_reduction_l2723_272390


namespace NUMINAMATH_CALUDE_average_minutes_run_per_day_l2723_272322

/-- The average number of minutes run per day by sixth graders -/
def sixth_grade_avg : ℚ := 20

/-- The average number of minutes run per day by seventh graders -/
def seventh_grade_avg : ℚ := 12

/-- The average number of minutes run per day by eighth graders -/
def eighth_grade_avg : ℚ := 18

/-- The ratio of sixth graders to eighth graders -/
def sixth_to_eighth_ratio : ℚ := 3

/-- The ratio of seventh graders to eighth graders -/
def seventh_to_eighth_ratio : ℚ := 3

/-- The theorem stating the average number of minutes run per day by all students -/
theorem average_minutes_run_per_day :
  let total_students := sixth_to_eighth_ratio + seventh_to_eighth_ratio + 1
  let total_minutes := sixth_grade_avg * sixth_to_eighth_ratio + 
                       seventh_grade_avg * seventh_to_eighth_ratio + 
                       eighth_grade_avg
  total_minutes / total_students = 114 / 7 := by
  sorry

end NUMINAMATH_CALUDE_average_minutes_run_per_day_l2723_272322


namespace NUMINAMATH_CALUDE_min_value_xyz_l2723_272331

theorem min_value_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 27) :
  x + 3 * y + 9 * z ≥ 27 ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 27 ∧ x₀ + 3 * y₀ + 9 * z₀ = 27 :=
by sorry

end NUMINAMATH_CALUDE_min_value_xyz_l2723_272331


namespace NUMINAMATH_CALUDE_shift_sin_left_specific_sin_shift_l2723_272395

/-- Shifting a sinusoidal function to the left --/
theorem shift_sin_left (A ω φ δ : ℝ) :
  let f (x : ℝ) := A * Real.sin (ω * x + φ)
  let g (x : ℝ) := A * Real.sin (ω * (x + δ) + φ)
  ∀ x, f (x - δ) = g x := by sorry

/-- The specific shift problem --/
theorem specific_sin_shift :
  let f (x : ℝ) := 3 * Real.sin (2 * x - π / 6)
  let g (x : ℝ) := 3 * Real.sin (2 * x + π / 3)
  ∀ x, f (x - π / 4) = g x := by sorry

end NUMINAMATH_CALUDE_shift_sin_left_specific_sin_shift_l2723_272395


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_of_cubes_l2723_272362

/-- Given an arithmetic sequence with first term x and common difference 2,
    this function returns the sum of cubes of the first n+1 terms. -/
def sumOfCubes (x : ℤ) (n : ℕ) : ℤ :=
  (Finset.range (n+1)).sum (fun i => (x + 2 * i)^3)

/-- Theorem stating that for an arithmetic sequence with integer first term,
    if the sum of cubes of its terms is -6859 and the number of terms is greater than 6,
    then the number of terms is exactly 7 (i.e., n = 6). -/
theorem arithmetic_sequence_sum_of_cubes (x : ℤ) (n : ℕ) 
    (h1 : sumOfCubes x n = -6859)
    (h2 : n > 5) : n = 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_of_cubes_l2723_272362


namespace NUMINAMATH_CALUDE_exists_valid_division_l2723_272317

/-- A grid-based figure --/
structure GridFigure where
  cells : ℕ

/-- Represents a division of a grid figure --/
structure Division where
  removed : ℕ
  part1 : ℕ
  part2 : ℕ

/-- Checks if a division is valid for a given grid figure --/
def is_valid_division (g : GridFigure) (d : Division) : Prop :=
  d.removed = 1 ∧ 
  d.part1 = d.part2 ∧
  d.part1 + d.part2 + d.removed = g.cells

/-- Theorem stating that a valid division exists for any grid figure --/
theorem exists_valid_division (g : GridFigure) : 
  ∃ (d : Division), is_valid_division g d :=
sorry

end NUMINAMATH_CALUDE_exists_valid_division_l2723_272317


namespace NUMINAMATH_CALUDE_cube_root_of_product_l2723_272371

theorem cube_root_of_product (n : ℕ) : (2^9 * 5^3 * 7^6 : ℝ)^(1/3) = 1960 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_product_l2723_272371


namespace NUMINAMATH_CALUDE_solution_set_l2723_272389

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem solution_set (h_increasing : ∀ x y, x < y → f x < f y)
                     (h_f_0 : f 0 = -1)
                     (h_f_3 : f 3 = 1) :
  {x : ℝ | |f (x + 1)| < 1} = Set.Ioo (-1) 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_l2723_272389


namespace NUMINAMATH_CALUDE_angle_sum_ninety_degrees_l2723_272308

theorem angle_sum_ninety_degrees (α β : Real) 
  (acute_α : 0 < α ∧ α < Real.pi / 2)
  (acute_β : 0 < β ∧ β < Real.pi / 2)
  (eq1 : 3 * Real.sin α ^ 2 + 2 * Real.sin β ^ 2 = 1)
  (eq2 : 3 * Real.sin (2 * α) - 2 * Real.sin (2 * β) = 0) :
  α + 2 * β = Real.pi / 2 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_ninety_degrees_l2723_272308


namespace NUMINAMATH_CALUDE_probability_two_girls_l2723_272394

theorem probability_two_girls (p : ℝ) (h1 : p = 1 / 2) : p * p = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_girls_l2723_272394


namespace NUMINAMATH_CALUDE_factorial_sum_mod_30_l2723_272381

theorem factorial_sum_mod_30 : (1 + 2 + 6 + 24 + 120) % 30 = 3 := by sorry

end NUMINAMATH_CALUDE_factorial_sum_mod_30_l2723_272381


namespace NUMINAMATH_CALUDE_inequality_implication_l2723_272355

theorem inequality_implication (x y : ℝ) : x > y → -2*x < -2*y := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l2723_272355


namespace NUMINAMATH_CALUDE_unique_modular_inverse_in_range_l2723_272301

theorem unique_modular_inverse_in_range (p : Nat) (a : Nat) 
  (h_prime : Nat.Prime p) 
  (h_odd : Odd p)
  (h_a_range : 2 ≤ a ∧ a ≤ p - 2) :
  ∃! i : Nat, 2 ≤ i ∧ i ≤ p - 2 ∧ 
    (i * a) % p = 1 ∧ 
    Nat.gcd i a = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_inverse_in_range_l2723_272301


namespace NUMINAMATH_CALUDE_find_number_l2723_272378

theorem find_number (k : ℝ) (x : ℝ) (h1 : x / k = 4) (h2 : k = 16) : x = 64 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l2723_272378


namespace NUMINAMATH_CALUDE_second_caterer_cheaper_l2723_272372

/-- Represents a caterer's pricing structure -/
structure Caterer where
  basic_fee : ℕ
  per_person : ℕ

/-- Calculates the total cost for a given number of people -/
def total_cost (c : Caterer) (people : ℕ) : ℕ :=
  c.basic_fee + c.per_person * people

/-- The first caterer's pricing structure -/
def caterer1 : Caterer :=
  { basic_fee := 150, per_person := 18 }

/-- The second caterer's pricing structure -/
def caterer2 : Caterer :=
  { basic_fee := 250, per_person := 15 }

/-- Theorem stating the minimum number of people for which the second caterer is cheaper -/
theorem second_caterer_cheaper : 
  (∀ n : ℕ, n ≥ 34 → total_cost caterer2 n < total_cost caterer1 n) ∧
  (∀ n : ℕ, n < 34 → total_cost caterer2 n ≥ total_cost caterer1 n) :=
sorry

end NUMINAMATH_CALUDE_second_caterer_cheaper_l2723_272372


namespace NUMINAMATH_CALUDE_grocer_sales_l2723_272330

theorem grocer_sales (sales : List ℕ) (average : ℕ) : 
  sales = [800, 900, 1000, 800, 900] →
  average = 850 →
  (sales.sum + 700) / 6 = average →
  700 = 6 * average - sales.sum :=
by sorry

end NUMINAMATH_CALUDE_grocer_sales_l2723_272330


namespace NUMINAMATH_CALUDE_rachel_second_level_treasures_l2723_272338

/-- Represents the video game scoring system and Rachel's performance --/
structure GameScore where
  points_per_treasure : ℕ
  treasures_first_level : ℕ
  total_score : ℕ

/-- Calculates the number of treasures found on the second level --/
def treasures_second_level (game : GameScore) : ℕ :=
  (game.total_score - game.points_per_treasure * game.treasures_first_level) / game.points_per_treasure

/-- Theorem stating that Rachel found 2 treasures on the second level --/
theorem rachel_second_level_treasures :
  let game : GameScore := {
    points_per_treasure := 9,
    treasures_first_level := 5,
    total_score := 63
  }
  treasures_second_level game = 2 := by
  sorry

end NUMINAMATH_CALUDE_rachel_second_level_treasures_l2723_272338


namespace NUMINAMATH_CALUDE_percent_relation_l2723_272379

theorem percent_relation (a b c : ℝ) (h1 : c = 0.25 * a) (h2 : b = 0.5 * a) :
  c = 0.5 * b := by
  sorry

end NUMINAMATH_CALUDE_percent_relation_l2723_272379


namespace NUMINAMATH_CALUDE_bd_length_l2723_272383

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the length function
def length (p q : ℝ × ℝ) : ℝ := sorry

-- Define our specific quadrilateral
def quadABCD : Quadrilateral :=
  { A := sorry,
    B := sorry,
    C := sorry,
    D := sorry }

-- State the theorem
theorem bd_length :
  let ABCD := quadABCD
  (length ABCD.A ABCD.B = 5) →
  (length ABCD.B ABCD.C = 17) →
  (length ABCD.C ABCD.D = 5) →
  (length ABCD.D ABCD.A = 9) →
  ∃ n : ℕ, (length ABCD.B ABCD.D = n) ∧ (n = 13) :=
sorry

end NUMINAMATH_CALUDE_bd_length_l2723_272383


namespace NUMINAMATH_CALUDE_sum_58_46_rounded_to_hundred_l2723_272325

def round_to_nearest_hundred (n : ℤ) : ℤ :=
  (n + 50) / 100 * 100

theorem sum_58_46_rounded_to_hundred : 
  round_to_nearest_hundred (58 + 46) = 100 := by
  sorry

end NUMINAMATH_CALUDE_sum_58_46_rounded_to_hundred_l2723_272325


namespace NUMINAMATH_CALUDE_derivative_of_odd_function_is_even_l2723_272346

theorem derivative_of_odd_function_is_even 
  (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h_odd : ∀ x, f (-x) = -f x) : 
  ∀ x, (deriv f) (-x) = (deriv f) x := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_odd_function_is_even_l2723_272346


namespace NUMINAMATH_CALUDE_prob_ace_then_king_standard_deck_l2723_272349

/-- A standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (ace_count : ℕ)
  (king_count : ℕ)

/-- The probability of drawing an Ace then a King from a standard deck -/
def prob_ace_then_king (d : Deck) : ℚ :=
  (d.ace_count : ℚ) / d.total_cards * (d.king_count : ℚ) / (d.total_cards - 1)

/-- A standard 52-card deck -/
def standard_deck : Deck :=
  { total_cards := 52
  , ace_count := 4
  , king_count := 4 }

/-- Theorem: The probability of drawing an Ace then a King from a standard 52-card deck is 4/663 -/
theorem prob_ace_then_king_standard_deck :
  prob_ace_then_king standard_deck = 4 / 663 := by
  sorry

end NUMINAMATH_CALUDE_prob_ace_then_king_standard_deck_l2723_272349


namespace NUMINAMATH_CALUDE_fractional_equation_solution_range_l2723_272340

theorem fractional_equation_solution_range (m : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ x ≠ 2 ∧ (x + m) / (x - 2) + (2 * m) / (2 - x) = 3) → 
  m < 6 ∧ m ≠ 2 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_range_l2723_272340


namespace NUMINAMATH_CALUDE_rectangle_side_multiple_of_6_l2723_272368

/-- A rectangle constructed from 1 x 6 rectangles -/
structure Rectangle :=
  (length : ℕ)
  (width : ℕ)
  (area : ℕ)
  (area_eq : area = length * width)
  (divisible_by_6 : 6 ∣ area)

/-- Theorem: One side of a rectangle constructed from 1 x 6 rectangles is a multiple of 6 -/
theorem rectangle_side_multiple_of_6 (r : Rectangle) : 
  6 ∣ r.length ∨ 6 ∣ r.width :=
sorry

end NUMINAMATH_CALUDE_rectangle_side_multiple_of_6_l2723_272368


namespace NUMINAMATH_CALUDE_negation_existence_real_l2723_272350

theorem negation_existence_real : 
  (¬ ∃ x : ℝ, x > 1) ↔ (∀ x : ℝ, x ≤ 1) := by sorry

end NUMINAMATH_CALUDE_negation_existence_real_l2723_272350


namespace NUMINAMATH_CALUDE_f_zero_values_l2723_272360

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x * f y

/-- Theorem stating that f(0) is either 0 or 1 for functions satisfying the functional equation -/
theorem f_zero_values (f : ℝ → ℝ) (h : FunctionalEquation f) :
  f 0 = 0 ∨ f 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_zero_values_l2723_272360


namespace NUMINAMATH_CALUDE_equation_solution_l2723_272351

theorem equation_solution : 
  ∃ x : ℝ, x > 0 ∧ (3 + x)^5 = (1 + 3*x)^4 ∧ x = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2723_272351


namespace NUMINAMATH_CALUDE_range_of_expression_l2723_272320

theorem range_of_expression (a b c : ℝ) 
  (h1 : -3 < b) (h2 : b < a) (h3 : a < -1) 
  (h4 : -2 < c) (h5 : c < -1) : 
  ∃ (x : ℝ), 0 < x ∧ x < 8 ∧ x = (a - b) * c^2 :=
sorry

end NUMINAMATH_CALUDE_range_of_expression_l2723_272320


namespace NUMINAMATH_CALUDE_cost_price_calculation_l2723_272387

/-- Proves that given a selling price of 400 and a profit percentage of 60%, 
    the cost price of the article is 250. -/
theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) :
  selling_price = 400 ∧ profit_percentage = 60 →
  (selling_price / (1 + profit_percentage / 100) : ℝ) = 250 := by
sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l2723_272387


namespace NUMINAMATH_CALUDE_base_representation_comparison_l2723_272356

theorem base_representation_comparison (n : ℕ) (h : n = 1357) :
  (Nat.log 3 n + 1) = (Nat.log 5 n + 1) + (Nat.log 8 n + 1) - 2 :=
by sorry

end NUMINAMATH_CALUDE_base_representation_comparison_l2723_272356


namespace NUMINAMATH_CALUDE_sum_of_squares_equation_l2723_272361

theorem sum_of_squares_equation (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (eq1 : a^2 + a*b + b^2 = 1)
  (eq2 : b^2 + b*c + c^2 = 3)
  (eq3 : c^2 + c*a + a^2 = 4) :
  a + b + c = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_equation_l2723_272361


namespace NUMINAMATH_CALUDE_side_b_value_l2723_272326

-- Define the triangle ABC
def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  -- Add conditions to ensure it's a valid triangle
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  A > 0 ∧ B > 0 ∧ C > 0 ∧
  A + B + C = Real.pi

-- State the theorem
theorem side_b_value (a b c : ℝ) (A B C : ℝ) :
  triangle_ABC a b c A B C →
  c = Real.sqrt 3 →
  B = Real.pi / 4 →
  C = Real.pi / 3 →
  b = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_side_b_value_l2723_272326


namespace NUMINAMATH_CALUDE_six_people_round_table_one_reserved_l2723_272316

/-- The number of ways to arrange people around a round table --/
def roundTableArrangements (n : ℕ) (reserved : ℕ) : ℕ :=
  Nat.factorial (n - reserved)

/-- Theorem: 6 people around a round table with 1 reserved seat --/
theorem six_people_round_table_one_reserved :
  roundTableArrangements 6 1 = 120 := by
  sorry

end NUMINAMATH_CALUDE_six_people_round_table_one_reserved_l2723_272316


namespace NUMINAMATH_CALUDE_line_moved_down_by_two_l2723_272382

/-- Represents a line in the form y = mx + b -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Moves a line vertically by a given amount -/
def moveVertically (l : Line) (amount : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept - amount }

theorem line_moved_down_by_two :
  let original := Line.mk 3 0
  let moved := moveVertically original 2
  moved = Line.mk 3 (-2) := by sorry

end NUMINAMATH_CALUDE_line_moved_down_by_two_l2723_272382


namespace NUMINAMATH_CALUDE_work_completion_time_l2723_272347

theorem work_completion_time (b c total_time : ℝ) (total_payment c_payment : ℕ) 
  (hb : b = 8)
  (hc : c = 3)
  (htotal_payment : total_payment = 3680)
  (hc_payment : c_payment = 460) :
  ∃ a : ℝ, a = 24 / 5 ∧ 1 / a + 1 / b = 1 / c := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l2723_272347


namespace NUMINAMATH_CALUDE_factorization_difference_of_squares_factorization_quadratic_l2723_272304

-- Problem 1
theorem factorization_difference_of_squares (x y : ℝ) :
  x^2 - 4*y^2 = (x + 2*y) * (x - 2*y) := by sorry

-- Problem 2
theorem factorization_quadratic (a x : ℝ) :
  3*a*x^2 - 6*a*x + 3*a = 3*a*(x - 1)^2 := by sorry

end NUMINAMATH_CALUDE_factorization_difference_of_squares_factorization_quadratic_l2723_272304


namespace NUMINAMATH_CALUDE_participation_related_to_city_probability_one_from_each_city_l2723_272314

-- Define the contingency table
def contingency_table : Matrix (Fin 2) (Fin 2) ℕ :=
  ![![60, 40],
    ![30, 70]]

-- Define the K^2 formula
def K_squared (a b c d : ℕ) : ℚ :=
  let n := a + b + c + d
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the critical value for 99.9% certainty
def critical_value : ℚ := 10828 / 1000

-- Theorem for part 1
theorem participation_related_to_city :
  let a := contingency_table 0 0
  let b := contingency_table 0 1
  let c := contingency_table 1 0
  let d := contingency_table 1 1
  K_squared a b c d > critical_value :=
sorry

-- Define the number of people from each city
def city_A_count : ℕ := 4
def city_B_count : ℕ := 2
def total_count : ℕ := city_A_count + city_B_count

-- Theorem for part 2
theorem probability_one_from_each_city :
  (Nat.choose city_A_count 1 * Nat.choose city_B_count 1 : ℚ) / Nat.choose total_count 2 = 8 / 15 :=
sorry

end NUMINAMATH_CALUDE_participation_related_to_city_probability_one_from_each_city_l2723_272314


namespace NUMINAMATH_CALUDE_ellipse_and_line_properties_l2723_272303

-- Define the ellipse C
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

-- Define the line l
structure Line where
  k : ℝ
  m : ℝ

-- Define the properties of the ellipse
def is_valid_ellipse (C : Ellipse) : Prop :=
  C.a = 2 ∧ C.b^2 = 3 ∧ C.a > C.b ∧ C.b > 0

-- Define the intersection of line and ellipse
def line_intersects_ellipse (l : Line) (C : Ellipse) : Prop :=
  ∃ x y, (x^2 / C.a^2) + (y^2 / C.b^2) = 1 ∧ y = l.k * x + l.m

-- Define the condition for the circle passing through origin
def circle_passes_through_origin (l : Line) (C : Ellipse) : Prop :=
  ∃ x₁ y₁ x₂ y₂, 
    line_intersects_ellipse l C ∧
    x₁ * x₂ + y₁ * y₂ = 0 ∧
    y₁ = l.k * x₁ + l.m ∧
    y₂ = l.k * x₂ + l.m

-- Main theorem
theorem ellipse_and_line_properties (C : Ellipse) (l : Line) :
  is_valid_ellipse C →
  circle_passes_through_origin l C →
  (C.a^2 = 4 ∧ C.b^2 = 3) ∧
  (l.m < -2 * Real.sqrt 21 / 7 ∨ l.m > 2 * Real.sqrt 21 / 7) :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_line_properties_l2723_272303


namespace NUMINAMATH_CALUDE_largest_band_members_l2723_272366

theorem largest_band_members :
  ∀ (m r x : ℕ),
    m < 100 →
    m = r * x + 4 →
    m = (r - 3) * (x + 2) →
    (∀ m' r' x' : ℕ,
      m' < 100 →
      m' = r' * x' + 4 →
      m' = (r' - 3) * (x' + 2) →
      m' ≤ m) →
    m = 88 :=
by sorry

end NUMINAMATH_CALUDE_largest_band_members_l2723_272366


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2723_272313

/-- A regular polygon with an exterior angle of 12 degrees has 30 sides. -/
theorem regular_polygon_sides (n : ℕ) (exterior_angle : ℝ) : 
  exterior_angle = 12 → (n : ℝ) * exterior_angle = 360 → n = 30 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2723_272313


namespace NUMINAMATH_CALUDE_mean_transformation_l2723_272392

theorem mean_transformation (x₁ x₂ x₃ x₄ : ℝ) 
  (h : (x₁ + x₂ + x₃ + x₄) / 4 = 5) : 
  ((x₁ + 1) + (x₂ + 2) + (x₃ + x₄ + 4) + (5 + 5)) / 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_mean_transformation_l2723_272392


namespace NUMINAMATH_CALUDE_unique_line_through_point_l2723_272341

/-- A line in the xy-plane --/
structure Line where
  x_intercept : ℕ+
  y_intercept : ℕ+

/-- Checks if a natural number is prime --/
def isPrime (n : ℕ+) : Prop :=
  n > 1 ∧ ∀ m : ℕ+, m < n → m ∣ n → m = 1

/-- Checks if a line passes through the point (5,4) --/
def passesThrough (l : Line) : Prop :=
  5 / l.x_intercept.val + 4 / l.y_intercept.val = 1

/-- The main theorem --/
theorem unique_line_through_point :
  ∃! l : Line, passesThrough l ∧ isPrime l.y_intercept :=
sorry

end NUMINAMATH_CALUDE_unique_line_through_point_l2723_272341


namespace NUMINAMATH_CALUDE_function_value_at_2017_l2723_272305

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ a b : ℝ, 3 * f ((a + 2 * b) / 3) = f a + 2 * f b) ∧
  f 1 = 1 ∧
  f 4 = 7

/-- The main theorem -/
theorem function_value_at_2017 (f : ℝ → ℝ) (h : special_function f) : f 2017 = 4033 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_2017_l2723_272305
