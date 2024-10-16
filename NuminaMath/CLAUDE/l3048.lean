import Mathlib

namespace NUMINAMATH_CALUDE_contrapositive_square_sum_l3048_304835

theorem contrapositive_square_sum (a b : ℝ) :
  (¬(a = 0 ∧ b = 0) → a^2 + b^2 ≠ 0) ↔ (a^2 + b^2 = 0 → a = 0 ∧ b = 0) :=
sorry

end NUMINAMATH_CALUDE_contrapositive_square_sum_l3048_304835


namespace NUMINAMATH_CALUDE_orange_cost_l3048_304889

theorem orange_cost (num_bananas : ℕ) (num_oranges : ℕ) (banana_cost : ℚ) (total_cost : ℚ) :
  num_bananas = 5 →
  num_oranges = 10 →
  banana_cost = 2 →
  total_cost = 25 →
  (total_cost - num_bananas * banana_cost) / num_oranges = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_orange_cost_l3048_304889


namespace NUMINAMATH_CALUDE_abs_neg_two_thirds_l3048_304840

theorem abs_neg_two_thirds : |(-2 : ℚ) / 3| = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_two_thirds_l3048_304840


namespace NUMINAMATH_CALUDE_system_solution_unique_l3048_304836

theorem system_solution_unique (x y : ℝ) : 
  (x + 2*y = 8 ∧ 3*x + y = 9) ↔ (x = 2 ∧ y = 3) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l3048_304836


namespace NUMINAMATH_CALUDE_inequality_implication_l3048_304857

theorem inequality_implication (a b : ℝ) : -2 * a + 1 < -2 * b + 1 → a > b := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l3048_304857


namespace NUMINAMATH_CALUDE_one_fourth_of_hundred_equals_ten_percent_of_250_l3048_304812

theorem one_fourth_of_hundred_equals_ten_percent_of_250 : 
  (1 / 4 : ℚ) * 100 = (10 / 100 : ℚ) * 250 := by
  sorry

end NUMINAMATH_CALUDE_one_fourth_of_hundred_equals_ten_percent_of_250_l3048_304812


namespace NUMINAMATH_CALUDE_tangent_and_perpendicular_l3048_304873

-- Define the curve f(x)
def f (x : ℝ) : ℝ := x^3 + 3*x^2 - 1

-- Define the line perpendicular to the given line
def perp_line (x y : ℝ) : Prop := 2*x - 6*y + 1 = 0

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := 3*x + y + 2 = 0

-- Define the theorem
theorem tangent_and_perpendicular :
  ∃ (x₀ y₀ : ℝ),
    -- The point (x₀, y₀) is on the curve
    f x₀ = y₀ ∧
    -- The tangent line passes through (x₀, y₀)
    tangent_line x₀ y₀ ∧
    -- The tangent line is perpendicular to the given line
    (∀ (x y : ℝ), perp_line x y → 
      (y - y₀) = -(1/3) * (x - x₀)) ∧
    -- The slope of the tangent line at (x₀, y₀) is the derivative of f at x₀
    (3*x₀^2 + 6*x₀ = -3) :=
sorry

end NUMINAMATH_CALUDE_tangent_and_perpendicular_l3048_304873


namespace NUMINAMATH_CALUDE_win_sector_area_l3048_304891

theorem win_sector_area (r : ℝ) (p : ℝ) (h1 : r = 8) (h2 : p = 3/8) :
  p * π * r^2 = 24 * π := by
  sorry

end NUMINAMATH_CALUDE_win_sector_area_l3048_304891


namespace NUMINAMATH_CALUDE_computer_price_increase_l3048_304831

/-- Given a computer with original price x dollars, where 2x = 540,
    prove that after a 30% increase, the new price is 351 dollars. -/
theorem computer_price_increase (x : ℝ) (h1 : 2 * x = 540) :
  x * 1.3 = 351 := by
  sorry

end NUMINAMATH_CALUDE_computer_price_increase_l3048_304831


namespace NUMINAMATH_CALUDE_circle_m_equation_l3048_304876

/-- A circle M with center on the negative x-axis and radius 4, tangent to the line 3x + 4y + 4 = 0 -/
structure CircleM where
  /-- The x-coordinate of the center of the circle -/
  a : ℝ
  /-- The center is on the negative x-axis -/
  h_negative : a < 0
  /-- The radius of the circle is 4 -/
  radius : ℝ := 4
  /-- The line 3x + 4y + 4 = 0 is tangent to the circle -/
  h_tangent : |3 * a + 4| / Real.sqrt (3^2 + 4^2) = radius

/-- The equation of circle M is (x+8)² + y² = 16 -/
theorem circle_m_equation (m : CircleM) : 
  ∀ x y : ℝ, (x - m.a)^2 + y^2 = m.radius^2 ↔ (x + 8)^2 + y^2 = 16 :=
sorry

end NUMINAMATH_CALUDE_circle_m_equation_l3048_304876


namespace NUMINAMATH_CALUDE_percentage_b_of_d_l3048_304842

theorem percentage_b_of_d (A B C D : ℝ) 
  (hB : B = 1.71 * A) 
  (hC : C = 1.80 * A) 
  (hD : D = 1.90 * B) : 
  ∃ ε > 0, |100 * B / D - 52.63| < ε :=
sorry

end NUMINAMATH_CALUDE_percentage_b_of_d_l3048_304842


namespace NUMINAMATH_CALUDE_equation_equivalence_l3048_304851

theorem equation_equivalence : ∀ x : ℝ, 2 * (x + 1) = x + 7 ↔ x = 5 := by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l3048_304851


namespace NUMINAMATH_CALUDE_polygon_with_five_triangles_l3048_304811

/-- A polygon with n sides -/
structure Polygon (n : ℕ) where
  -- We don't need to define the structure, just declare it

/-- The number of triangles formed when drawing diagonals from a single vertex -/
def triangles_from_vertex (n : ℕ) : ℕ := n - 2

/-- Theorem: If the diagonals from the same vertex of an n-sided polygon
    exactly divide the polygon into 5 triangles, then n = 7 -/
theorem polygon_with_five_triangles (n : ℕ) :
  triangles_from_vertex n = 5 → n = 7 := by
  sorry


end NUMINAMATH_CALUDE_polygon_with_five_triangles_l3048_304811


namespace NUMINAMATH_CALUDE_B_2_2_l3048_304808

def B : ℕ → ℕ → ℕ
| 0, n => n + 2
| m + 1, 0 => B m 2
| m + 1, n + 1 => B m (B (m + 1) n)

theorem B_2_2 : B 2 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_B_2_2_l3048_304808


namespace NUMINAMATH_CALUDE_rose_painting_time_l3048_304875

/-- Time to paint a lily in minutes -/
def lily_time : ℕ := 5

/-- Time to paint an orchid in minutes -/
def orchid_time : ℕ := 3

/-- Time to paint a vine in minutes -/
def vine_time : ℕ := 2

/-- Total time spent painting in minutes -/
def total_time : ℕ := 213

/-- Number of lilies painted -/
def lily_count : ℕ := 17

/-- Number of roses painted -/
def rose_count : ℕ := 10

/-- Number of orchids painted -/
def orchid_count : ℕ := 6

/-- Number of vines painted -/
def vine_count : ℕ := 20

/-- Time to paint a rose in minutes -/
def rose_time : ℕ := 7

theorem rose_painting_time : 
  lily_count * lily_time + rose_count * rose_time + orchid_count * orchid_time + vine_count * vine_time = total_time := by
  sorry

end NUMINAMATH_CALUDE_rose_painting_time_l3048_304875


namespace NUMINAMATH_CALUDE_sqrt_six_and_quarter_equals_five_halves_l3048_304870

theorem sqrt_six_and_quarter_equals_five_halves :
  Real.sqrt (6 + 1/4) = 5/2 := by sorry

end NUMINAMATH_CALUDE_sqrt_six_and_quarter_equals_five_halves_l3048_304870


namespace NUMINAMATH_CALUDE_irrational_angle_cosine_one_third_l3048_304837

theorem irrational_angle_cosine_one_third (α : ℝ) : 
  Real.cos (α * π / 180) = 1/3 → ¬ ∃ (q : ℚ), α = ↑q := by
  sorry

end NUMINAMATH_CALUDE_irrational_angle_cosine_one_third_l3048_304837


namespace NUMINAMATH_CALUDE_chairs_remaining_l3048_304883

def classroom_chairs (total red yellow blue green orange : ℕ) : Prop :=
  total = 62 ∧
  red = 4 ∧
  yellow = 2 * red ∧
  blue = 3 * yellow ∧
  green = blue / 2 ∧
  orange = green + 2 ∧
  total = red + yellow + blue + green + orange

def lisa_borrows (total borrowed : ℕ) : Prop :=
  borrowed = total / 10

def carla_borrows (remaining borrowed : ℕ) : Prop :=
  borrowed = remaining / 5

theorem chairs_remaining 
  (total red yellow blue green orange : ℕ)
  (lisa_borrowed carla_borrowed : ℕ)
  (h1 : classroom_chairs total red yellow blue green orange)
  (h2 : lisa_borrows total lisa_borrowed)
  (h3 : carla_borrows (total - lisa_borrowed) carla_borrowed) :
  total - lisa_borrowed - carla_borrowed = 45 :=
sorry

end NUMINAMATH_CALUDE_chairs_remaining_l3048_304883


namespace NUMINAMATH_CALUDE_cylinder_heights_sum_l3048_304874

theorem cylinder_heights_sum (p₁ p₂ p₃ : ℝ) 
  (h₁ : p₁ = 6) 
  (h₂ : p₂ = 9) 
  (h₃ : p₃ = 11) : 
  p₁ + p₂ + p₃ = 26 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_heights_sum_l3048_304874


namespace NUMINAMATH_CALUDE_roberts_initial_balls_prove_roberts_initial_balls_l3048_304880

theorem roberts_initial_balls (tim_balls : ℕ) (robert_final : ℕ) : ℕ :=
  let tim_gave := tim_balls / 2
  let robert_initial := robert_final - tim_gave
  robert_initial

theorem prove_roberts_initial_balls :
  roberts_initial_balls 40 45 = 25 := by
  sorry

end NUMINAMATH_CALUDE_roberts_initial_balls_prove_roberts_initial_balls_l3048_304880


namespace NUMINAMATH_CALUDE_janet_pay_calculation_l3048_304813

/-- Represents Janet's work parameters and calculates her pay per post. -/
def janet_pay_per_post (check_time : ℕ) (hourly_rate : ℚ) : ℚ :=
  let seconds_per_hour : ℕ := 3600
  let posts_per_hour : ℕ := seconds_per_hour / check_time
  hourly_rate / posts_per_hour

/-- Proves that Janet's pay per post is $0.25 given the specified conditions. -/
theorem janet_pay_calculation :
  janet_pay_per_post 10 90 = 1/4 := by
  sorry

#eval janet_pay_per_post 10 90

end NUMINAMATH_CALUDE_janet_pay_calculation_l3048_304813


namespace NUMINAMATH_CALUDE_polygon_sides_l3048_304832

theorem polygon_sides (n : ℕ) (angle_sum : ℝ) (excluded_angle : ℝ) :
  angle_sum = 2970 ∧
  angle_sum = (n - 2) * 180 - 2 * excluded_angle ∧
  excluded_angle > 0 ∧
  excluded_angle < 180 →
  n = 19 :=
by sorry

end NUMINAMATH_CALUDE_polygon_sides_l3048_304832


namespace NUMINAMATH_CALUDE_mutually_expressible_implies_symmetric_zero_l3048_304869

/-- A function f is symmetric if f(x, y) = f(y, x) for all x and y. -/
def IsSymmetric (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ x y, f x y = f y x

/-- Two variables x and y are mutually expressible if there exists a symmetric function f
    such that f(x, y) = 0 implies both y = g(x) and x = g(y) for some function g. -/
def MutuallyExpressible (x y : ℝ) : Prop :=
  ∃ (f : ℝ → ℝ → ℝ) (g : ℝ → ℝ), IsSymmetric f ∧ f x y = 0 ∧ y = g x ∧ x = g y

/-- Theorem: If two variables are mutually expressible, then there exists a symmetric function
    that equals zero for those variables. -/
theorem mutually_expressible_implies_symmetric_zero (x y : ℝ) :
  MutuallyExpressible x y → ∃ (f : ℝ → ℝ → ℝ), IsSymmetric f ∧ f x y = 0 := by
  sorry

end NUMINAMATH_CALUDE_mutually_expressible_implies_symmetric_zero_l3048_304869


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3048_304847

theorem purely_imaginary_complex_number (a : ℝ) : 
  (∃ (z : ℂ), z = Complex.mk (a * (a - 1)) (a) ∧ z.re = 0) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3048_304847


namespace NUMINAMATH_CALUDE_bedroom_paint_area_l3048_304801

/-- Calculates the total paintable area in multiple identical bedrooms -/
def total_paintable_area (
  num_bedrooms : ℕ
  ) (length width height : ℝ
  ) (unpaintable_area : ℝ
  ) : ℝ :=
  num_bedrooms * (2 * (length * height + width * height) - unpaintable_area)

/-- Proves that the total paintable area in the given conditions is 1288 square feet -/
theorem bedroom_paint_area :
  total_paintable_area 4 10 12 9 74 = 1288 := by
  sorry

end NUMINAMATH_CALUDE_bedroom_paint_area_l3048_304801


namespace NUMINAMATH_CALUDE_fraction_equality_l3048_304859

theorem fraction_equality : (10^9 + 10^6) / (3 * 10^4) = 100100 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3048_304859


namespace NUMINAMATH_CALUDE_matrix_power_2023_l3048_304890

def A : Matrix (Fin 2) (Fin 2) ℕ := !![1, 0; 2, 1]

theorem matrix_power_2023 : 
  A ^ 2023 = !![1, 0; 4046, 1] := by sorry

end NUMINAMATH_CALUDE_matrix_power_2023_l3048_304890


namespace NUMINAMATH_CALUDE_triangle_properties_l3048_304819

/-- Theorem about a triangle ABC with specific angle and side properties -/
theorem triangle_properties (A B C : Real) (a b c : Real) (D : Real) :
  -- Triangle conditions
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Given equation
  Real.sin A ^ 2 + Real.sin B ^ 2 - Real.sin C ^ 2 = Real.sqrt 2 * Real.sin A * Real.sin B →
  -- Additional conditions
  Real.cos B = 3 / 5 →
  0 < D ∧ D < 1 →  -- Representing CD = 4BD as D = 4/(1+4) = 4/5
  -- Area condition (using scaled version to avoid square root)
  a * c * D * Real.sin A = 14 / 5 →
  -- Conclusions
  C = π / 4 ∧ a = 2 := by
  sorry

#check triangle_properties

end NUMINAMATH_CALUDE_triangle_properties_l3048_304819


namespace NUMINAMATH_CALUDE_parabola_m_minus_one_opens_downward_l3048_304817

/-- A parabola y = ax^2 opens downward if and only if a < 0 -/
axiom parabola_opens_downward (a : ℝ) : (∀ x y : ℝ, y = a * x^2) → (∀ x : ℝ, a * x^2 ≤ 0) ↔ a < 0

/-- For the parabola y = (m-1)x^2 to open downward, m must be less than 1 -/
theorem parabola_m_minus_one_opens_downward (m : ℝ) :
  (∀ x y : ℝ, y = (m - 1) * x^2) → (∀ x : ℝ, (m - 1) * x^2 ≤ 0) → m < 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_m_minus_one_opens_downward_l3048_304817


namespace NUMINAMATH_CALUDE_certain_number_problem_l3048_304844

theorem certain_number_problem : ∃ x : ℝ, (0.60 * 50 = 0.40 * x + 18) ∧ (x = 30) := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l3048_304844


namespace NUMINAMATH_CALUDE_james_age_l3048_304862

theorem james_age (dan_age james_age : ℕ) : 
  (dan_age : ℚ) / james_age = 6 / 5 →
  dan_age + 4 = 28 →
  james_age = 20 := by
sorry

end NUMINAMATH_CALUDE_james_age_l3048_304862


namespace NUMINAMATH_CALUDE_expression_value_l3048_304855

theorem expression_value (x : ℝ) (h : x^2 - x - 3 = 0) :
  (x + 2) * (x - 2) - x * (2 - x) = 2 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l3048_304855


namespace NUMINAMATH_CALUDE_triangle_area_l3048_304810

/-- The area of the triangle formed by the intersection of two lines and the y-axis --/
theorem triangle_area (line1 line2 : ℝ → ℝ) : 
  line1 = (λ x => 3 * x - 6) →
  line2 = (λ x => -4 * x + 24) →
  let x_intersect := (30 : ℝ) / 7
  let y_intersect := (48 : ℝ) / 7
  let base := 30
  let height := x_intersect
  (1 / 2 : ℝ) * base * height = 450 / 7 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l3048_304810


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l3048_304838

theorem fractional_equation_solution :
  ∃ x : ℚ, (1 / (2 * x) = 2 / (x - 3)) ∧ x = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l3048_304838


namespace NUMINAMATH_CALUDE_max_shoe_pairs_l3048_304872

theorem max_shoe_pairs (initial_pairs : ℕ) (lost_shoes : ℕ) (max_remaining_pairs : ℕ) : 
  initial_pairs = 23 → lost_shoes = 9 → max_remaining_pairs = 14 →
  max_remaining_pairs = initial_pairs - lost_shoes / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_shoe_pairs_l3048_304872


namespace NUMINAMATH_CALUDE_fraction_division_specific_fraction_division_l3048_304863

theorem fraction_division (a b c d : ℚ) (hb : b ≠ 0) (hd : d ≠ 0) :
  (a / b) / (c / d) = (a * d) / (b * c) := by sorry

theorem specific_fraction_division :
  (3 : ℚ) / 7 / ((2 : ℚ) / 5) = 15 / 14 := by sorry

end NUMINAMATH_CALUDE_fraction_division_specific_fraction_division_l3048_304863


namespace NUMINAMATH_CALUDE_inequality_proof_l3048_304899

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a / Real.sqrt b + b / Real.sqrt a ≥ Real.sqrt a + Real.sqrt b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3048_304899


namespace NUMINAMATH_CALUDE_third_term_value_l3048_304846

/-- An arithmetic sequence with five terms -/
structure ArithmeticSequence :=
  (a : ℝ)  -- First term
  (d : ℝ)  -- Common difference

/-- The third term of an arithmetic sequence -/
def third_term (seq : ArithmeticSequence) : ℝ := seq.a + 2 * seq.d

theorem third_term_value :
  ∀ seq : ArithmeticSequence,
  seq.a = 12 ∧ seq.a + 4 * seq.d = 32 →
  third_term seq = 22 :=
by sorry

end NUMINAMATH_CALUDE_third_term_value_l3048_304846


namespace NUMINAMATH_CALUDE_arithmetic_sequence_1001st_term_l3048_304809

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  p : ℚ
  q : ℚ
  first_term : ℚ
  second_term : ℚ
  third_term : ℚ
  fourth_term : ℚ
  is_arithmetic : ∃ (d : ℚ), second_term = first_term + d ∧
                              third_term = second_term + d ∧
                              fourth_term = third_term + d
  first_is_p : first_term = p
  second_is_12 : second_term = 12
  third_is_3p_minus_q : third_term = 3 * p - q
  fourth_is_3p_plus_2q : fourth_term = 3 * p + 2 * q

/-- The 1001st term of the sequence is 5545 -/
theorem arithmetic_sequence_1001st_term (seq : ArithmeticSequence) : 
  seq.first_term + 1000 * (seq.second_term - seq.first_term) = 5545 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_1001st_term_l3048_304809


namespace NUMINAMATH_CALUDE_train_length_l3048_304818

/-- Given a train that crosses a platform in 39 seconds, crosses a signal pole in 8 seconds,
    and the platform length is 1162.5 meters, prove that the length of the train is 300 meters. -/
theorem train_length (crossing_platform_time : ℝ) (crossing_pole_time : ℝ) (platform_length : ℝ)
  (h1 : crossing_platform_time = 39)
  (h2 : crossing_pole_time = 8)
  (h3 : platform_length = 1162.5) :
  ∃ (train_length : ℝ), train_length = 300 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3048_304818


namespace NUMINAMATH_CALUDE_bob_apples_correct_l3048_304852

/-- The number of apples Bob and Carla share -/
def total_apples : ℕ := 30

/-- Represents the number of apples Bob eats -/
def bob_apples : ℕ := 10

/-- Carla eats twice as many apples as Bob -/
def carla_apples (b : ℕ) : ℕ := 2 * b

theorem bob_apples_correct :
  bob_apples + carla_apples bob_apples = total_apples := by sorry

end NUMINAMATH_CALUDE_bob_apples_correct_l3048_304852


namespace NUMINAMATH_CALUDE_total_cost_with_tax_l3048_304827

def earbuds_cost : ℝ := 200
def smartwatch_cost : ℝ := 300
def earbuds_tax_rate : ℝ := 0.15
def smartwatch_tax_rate : ℝ := 0.12

theorem total_cost_with_tax : 
  earbuds_cost * (1 + earbuds_tax_rate) + smartwatch_cost * (1 + smartwatch_tax_rate) = 566 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_with_tax_l3048_304827


namespace NUMINAMATH_CALUDE_expression_evaluation_l3048_304867

theorem expression_evaluation :
  let a : ℚ := -1/2
  (a + 3)^2 + (a + 3)*(a - 3) - 2*a*(3 - a) = 1 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3048_304867


namespace NUMINAMATH_CALUDE_nell_initial_cards_l3048_304802

/-- The number of cards Nell gave to John -/
def cards_to_john : ℕ := 195

/-- The number of cards Nell gave to Jeff -/
def cards_to_jeff : ℕ := 168

/-- The number of cards Nell has left -/
def cards_left : ℕ := 210

/-- The initial number of cards Nell had -/
def initial_cards : ℕ := cards_to_john + cards_to_jeff + cards_left

theorem nell_initial_cards : initial_cards = 573 := by
  sorry

end NUMINAMATH_CALUDE_nell_initial_cards_l3048_304802


namespace NUMINAMATH_CALUDE_unique_m_value_l3048_304850

def A (m : ℝ) : Set ℝ := {0, m, m^2 - 3*m + 2}

theorem unique_m_value : ∃! m : ℝ, 2 ∈ A m ∧ m ≠ 0 ∧ m ≠ 2 :=
  sorry

end NUMINAMATH_CALUDE_unique_m_value_l3048_304850


namespace NUMINAMATH_CALUDE_no_solution_in_fourth_quadrant_l3048_304896

-- Define the complex number z
variable (z : ℂ)

-- Define m as a real number
variable (m : ℝ)

-- Define the equation (1-i)z = m + i
def equation (z : ℂ) (m : ℝ) : Prop :=
  (1 - Complex.I) * z = m + Complex.I

-- Theorem statement
theorem no_solution_in_fourth_quadrant :
  ¬ ∃ (z : ℂ) (m : ℝ), equation z m ∧ z.re > 0 ∧ z.im < 0 :=
sorry

end NUMINAMATH_CALUDE_no_solution_in_fourth_quadrant_l3048_304896


namespace NUMINAMATH_CALUDE_new_man_weight_l3048_304895

/-- Given a group of 8 men, if replacing a 60 kg man with a new man increases the average weight by 1 kg, then the new man weighs 68 kg. -/
theorem new_man_weight (initial_average : ℝ) : 
  (8 * initial_average + 68 = 8 * (initial_average + 1) + 60) → 68 = 68 := by
  sorry

end NUMINAMATH_CALUDE_new_man_weight_l3048_304895


namespace NUMINAMATH_CALUDE_shooting_probability_l3048_304834

/-- The probability of person A hitting the target in a single shot -/
def prob_A : ℚ := 3/4

/-- The probability of person B hitting the target in a single shot -/
def prob_B : ℚ := 4/5

/-- The probability that A has taken two shots when they stop shooting -/
def prob_A_two_shots : ℚ := 19/400

theorem shooting_probability :
  let p1 := (1 - prob_A) * (1 - prob_B) * prob_A
  let p2 := (1 - prob_A) * (1 - prob_B) * (1 - prob_A) * prob_B
  p1 + p2 = prob_A_two_shots := by sorry

end NUMINAMATH_CALUDE_shooting_probability_l3048_304834


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l3048_304800

/-- Sum of first n terms of an arithmetic sequence -/
def T (b : ℚ) (n : ℕ) : ℚ := n * (2 * b + (n - 1) * 5) / 2

/-- The problem statement -/
theorem arithmetic_sequence_first_term (b : ℚ) :
  (∃ k : ℚ, ∀ n : ℕ, n > 0 → T b (4 * n) / T b n = k) →
  b = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l3048_304800


namespace NUMINAMATH_CALUDE_simple_interest_rate_calculation_l3048_304894

theorem simple_interest_rate_calculation (P A T : ℝ) (h1 : P = 750) (h2 : A = 950) (h3 : T = 5) :
  let SI := A - P
  let R := (SI * 100) / (P * T)
  R = (200 * 100) / (750 * 5) := by sorry

end NUMINAMATH_CALUDE_simple_interest_rate_calculation_l3048_304894


namespace NUMINAMATH_CALUDE_shaded_area_circle_and_tangents_l3048_304861

theorem shaded_area_circle_and_tangents (r : ℝ) (θ : ℝ) :
  r = 3 →
  θ = Real.pi / 3 →
  let circle_area := π * r^2
  let sector_angle := 2 * θ
  let sector_area := (sector_angle / (2 * Real.pi)) * circle_area
  let triangle_area := r^2 * Real.tan θ
  sector_area + 2 * triangle_area = 6 * π + 9 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_circle_and_tangents_l3048_304861


namespace NUMINAMATH_CALUDE_uncool_parents_count_l3048_304898

theorem uncool_parents_count (total : ℕ) (cool_dads : ℕ) (cool_moms : ℕ) (both_cool : ℕ) :
  total = 35 →
  cool_dads = 18 →
  cool_moms = 22 →
  both_cool = 11 →
  total - (cool_dads + cool_moms - both_cool) = 6 :=
by sorry

end NUMINAMATH_CALUDE_uncool_parents_count_l3048_304898


namespace NUMINAMATH_CALUDE_find_m_l3048_304807

-- Define the function f
def f (x : ℝ) : ℝ := 4 * x + 7

-- State the theorem
theorem find_m (m : ℝ) : (∀ x, f (1/2 * x - 1) = 2 * x + 3) → f m = 6 → m = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_find_m_l3048_304807


namespace NUMINAMATH_CALUDE_platform_length_l3048_304877

/-- Given a train of length 300 meters that takes 36 seconds to cross a platform
    and 18 seconds to cross a signal pole, the length of the platform is 300 meters. -/
theorem platform_length (train_length : ℝ) (platform_crossing_time : ℝ) (pole_crossing_time : ℝ)
  (h1 : train_length = 300)
  (h2 : platform_crossing_time = 36)
  (h3 : pole_crossing_time = 18) :
  let train_speed := train_length / pole_crossing_time
  let platform_length := train_speed * platform_crossing_time - train_length
  platform_length = 300 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_l3048_304877


namespace NUMINAMATH_CALUDE_common_ratio_is_three_l3048_304828

/-- Geometric sequence with sum of first n terms -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum of first n terms
  is_geometric : ∀ n, a (n + 1) = (a 1) * (a 2 / a 1) ^ n

/-- The common ratio of a geometric sequence is 3 given specific conditions -/
theorem common_ratio_is_three (seq : GeometricSequence)
  (h1 : seq.a 5 = 2 * seq.S 4 + 3)
  (h2 : seq.a 6 = 2 * seq.S 5 + 3) :
  seq.a 2 / seq.a 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_common_ratio_is_three_l3048_304828


namespace NUMINAMATH_CALUDE_sports_competition_results_l3048_304882

-- Define the probabilities of School A winning each event
def p1 : ℝ := 0.5
def p2 : ℝ := 0.4
def p3 : ℝ := 0.8

-- Define the score for winning an event
def win_score : ℕ := 10

-- Define the probability of School A winning the championship
def prob_A_wins : ℝ := p1 * p2 * p3 + p1 * p2 * (1 - p3) + p1 * (1 - p2) * p3 + (1 - p1) * p2 * p3

-- Define the distribution of School B's total score
def dist_B : List (ℝ × ℝ) := [
  (0, (1 - p1) * (1 - p2) * (1 - p3)),
  (10, p1 * (1 - p2) * (1 - p3) + (1 - p1) * p2 * (1 - p3) + (1 - p1) * (1 - p2) * p3),
  (20, p1 * p2 * (1 - p3) + p1 * (1 - p2) * p3 + (1 - p1) * p2 * p3),
  (30, p1 * p2 * p3)
]

-- Define the expectation of School B's total score
def exp_B : ℝ := (dist_B.map (λ p => p.1 * p.2)).sum

-- Theorem statement
theorem sports_competition_results :
  prob_A_wins = 0.6 ∧ exp_B = 13 := by sorry

end NUMINAMATH_CALUDE_sports_competition_results_l3048_304882


namespace NUMINAMATH_CALUDE_john_computer_cost_l3048_304881

/-- The total cost of a computer after upgrades -/
def total_cost (initial_cost old_video_card old_memory old_processor new_video_card new_memory new_processor : ℕ) : ℕ :=
  initial_cost + new_video_card + new_memory + new_processor - old_video_card - old_memory - old_processor

/-- Theorem: The total cost of John's computer after upgrades is $2500 -/
theorem john_computer_cost : 
  total_cost 2000 300 100 150 500 200 350 = 2500 := by
  sorry

end NUMINAMATH_CALUDE_john_computer_cost_l3048_304881


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3048_304839

theorem inequality_solution_set (x : ℝ) :
  (x^2 - 2*x - 3) * (x^2 + 1) < 0 ↔ -1 < x ∧ x < 3 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3048_304839


namespace NUMINAMATH_CALUDE_parabola_point_to_directrix_distance_l3048_304848

/-- The distance from a point on a parabola to its directrix -/
theorem parabola_point_to_directrix_distance 
  (p : ℝ) -- Parameter of the parabola
  (A : ℝ × ℝ) -- Point A
  (h1 : A.1 = 1) -- x-coordinate of A is 1
  (h2 : A.2 = Real.sqrt 5) -- y-coordinate of A is √5
  (h3 : A.2^2 = 2 * p * A.1) -- A lies on the parabola y² = 2px
  : |A.1 - (-p/2)| = 9/4 := by
sorry


end NUMINAMATH_CALUDE_parabola_point_to_directrix_distance_l3048_304848


namespace NUMINAMATH_CALUDE_tangent_line_parallel_l3048_304820

/-- Given a function f(x) = x^3 - ax^2 + x, prove that if its tangent line at x=1 
    is parallel to y=2x, then a = 1. -/
theorem tangent_line_parallel (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^3 - a*x^2 + x
  let f' : ℝ → ℝ := λ x ↦ 3*x^2 - 2*a*x + 1
  (f' 1 = 2) → a = 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_parallel_l3048_304820


namespace NUMINAMATH_CALUDE_eight_valid_numbers_l3048_304822

/-- A function that reverses the digits of a two-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/-- A predicate that checks if a number is a two-digit number -/
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

/-- A predicate that checks if a number is a positive perfect square -/
def is_positive_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m > 0 ∧ m * m = n

/-- The main theorem stating that there are exactly 8 two-digit numbers satisfying the condition -/
theorem eight_valid_numbers :
  ∃! (s : Finset ℕ),
    Finset.card s = 8 ∧
    ∀ n ∈ s, is_two_digit n ∧
      is_positive_perfect_square (n - reverse_digits n) :=
sorry

end NUMINAMATH_CALUDE_eight_valid_numbers_l3048_304822


namespace NUMINAMATH_CALUDE_group_trip_cost_l3048_304864

/-- The total cost for a group trip, given the number of people and cost per person. -/
def total_cost (num_people : ℕ) (cost_per_person : ℕ) : ℕ :=
  num_people * cost_per_person

/-- Proof that the total cost for 15 people at $900 each is $13,500. -/
theorem group_trip_cost :
  total_cost 15 900 = 13500 := by
  sorry

end NUMINAMATH_CALUDE_group_trip_cost_l3048_304864


namespace NUMINAMATH_CALUDE_max_red_socks_l3048_304829

theorem max_red_socks (r g : ℕ) : 
  let t := r + g
  (t ≤ 3000) → 
  (r * (r - 1) + g * (g - 1)) / (t * (t - 1)) = 3/5 →
  r ≤ 1199 :=
sorry

end NUMINAMATH_CALUDE_max_red_socks_l3048_304829


namespace NUMINAMATH_CALUDE_at_least_one_correct_l3048_304814

theorem at_least_one_correct (pA pB : ℝ) (hA : pA = 0.8) (hB : pB = 0.9) :
  1 - (1 - pA) * (1 - pB) = 0.98 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_correct_l3048_304814


namespace NUMINAMATH_CALUDE_binomial_coefficient_two_l3048_304830

theorem binomial_coefficient_two (n : ℕ) (h : n > 1) : 
  Nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_two_l3048_304830


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l3048_304803

/-- Represents the stratified sampling problem --/
structure StratifiedSample where
  total_population : ℕ
  sample_size : ℕ
  elderly_count : ℕ
  middle_aged_count : ℕ
  young_count : ℕ

/-- Calculates the sample size for a specific group --/
def group_sample_size (s : StratifiedSample) (group_count : ℕ) : ℕ :=
  (group_count * s.sample_size) / s.total_population

/-- Theorem statement for the stratified sampling problem --/
theorem stratified_sampling_theorem (s : StratifiedSample)
  (h1 : s.total_population = s.elderly_count + s.middle_aged_count + s.young_count)
  (h2 : s.total_population = 162)
  (h3 : s.sample_size = 36)
  (h4 : s.elderly_count = 27)
  (h5 : s.middle_aged_count = 54)
  (h6 : s.young_count = 81) :
  group_sample_size s s.elderly_count = 6 ∧
  group_sample_size s s.middle_aged_count = 12 ∧
  group_sample_size s s.young_count = 18 := by
  sorry


end NUMINAMATH_CALUDE_stratified_sampling_theorem_l3048_304803


namespace NUMINAMATH_CALUDE_tobias_driveways_shoveled_tobias_driveways_shoveled_proof_l3048_304856

theorem tobias_driveways_shoveled : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → Prop :=
  fun shoe_cost saving_months allowance lawn_charge shovel_charge wage change hours_worked lawns_mowed driveways_shoveled =>
    shoe_cost = 95 ∧
    saving_months = 3 ∧
    allowance = 5 ∧
    lawn_charge = 15 ∧
    shovel_charge = 7 ∧
    wage = 8 ∧
    change = 15 ∧
    hours_worked = 10 ∧
    lawns_mowed = 4 →
    driveways_shoveled = 6

theorem tobias_driveways_shoveled_proof : tobias_driveways_shoveled 95 3 5 15 7 8 15 10 4 6 := by
  sorry

end NUMINAMATH_CALUDE_tobias_driveways_shoveled_tobias_driveways_shoveled_proof_l3048_304856


namespace NUMINAMATH_CALUDE_group_average_difference_l3048_304826

/-- Represents the first element of the n-th group -/
def a (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n % 2 = 1 then a (n - 1) + (n - 1)
  else a (n - 1) + (n - 2)

/-- Sum of elements in the n-th group -/
def S (n : ℕ) : ℕ :=
  n * (2 * a n + (n - 1) * 2) / 2

/-- Average of elements in the n-th group -/
def avg (n : ℕ) : ℚ :=
  (S n : ℚ) / n

theorem group_average_difference (n : ℕ) :
  avg (2 * n + 1) - avg (2 * n) = 2 * n := by
  sorry

end NUMINAMATH_CALUDE_group_average_difference_l3048_304826


namespace NUMINAMATH_CALUDE_star_theorems_l3048_304841

variable {S : Type*} [Inhabited S] [Nontrivial S]
variable (star : S → S → S)

axiom star_property : ∀ a b : S, star a (star b a) = b

theorem star_theorems :
  (∀ a b : S, star (star a (star b a)) (star a b) = a) ∧
  (∀ b : S, star b (star b b) = b) ∧
  (∀ a b : S, star (star a b) (star b (star a b)) = b) :=
by sorry

end NUMINAMATH_CALUDE_star_theorems_l3048_304841


namespace NUMINAMATH_CALUDE_seven_heads_or_tails_probability_l3048_304854

-- Define the number of coins
def num_coins : ℕ := 8

-- Define the probability of getting heads or tails on a single coin
def coin_prob : ℚ := 1/2

-- Define the function to calculate binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the probability of the event (7 heads or 7 tails)
def event_prob : ℚ := sorry

-- Theorem statement
theorem seven_heads_or_tails_probability :
  event_prob = 1/16 := by sorry

end NUMINAMATH_CALUDE_seven_heads_or_tails_probability_l3048_304854


namespace NUMINAMATH_CALUDE_hyperbola_foci_coordinates_l3048_304843

/-- Given a hyperbola with equation 5x^2 - 4y^2 + 60 = 0, its foci have coordinates (0, ±3√3) -/
theorem hyperbola_foci_coordinates :
  let hyperbola := fun (x y : ℝ) => 5 * x^2 - 4 * y^2 + 60
  ∃ (c : ℝ), c = 3 * Real.sqrt 3 ∧
    (∀ (x y : ℝ), hyperbola x y = 0 →
      (hyperbola 0 c = 0 ∧ hyperbola 0 (-c) = 0)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_coordinates_l3048_304843


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3048_304804

/-- A line that always passes through a fixed point regardless of the parameter m -/
def line (m x y : ℝ) : Prop :=
  (m - 1) * x - y + 2 * m + 1 = 0

/-- The fixed point that the line always passes through -/
def fixed_point : ℝ × ℝ := (-2, 3)

/-- Theorem stating that the line always passes through the fixed point -/
theorem line_passes_through_fixed_point :
  ∀ m : ℝ, line m (fixed_point.1) (fixed_point.2) :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3048_304804


namespace NUMINAMATH_CALUDE_work_completion_time_l3048_304866

-- Define the work completion time for Person A
def person_a_time : ℝ := 24

-- Define the combined work completion time for Person A and Person B
def combined_time : ℝ := 15

-- Define the work completion time for Person B
def person_b_time : ℝ := 40

-- Theorem statement
theorem work_completion_time :
  (1 / person_a_time + 1 / person_b_time = 1 / combined_time) := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3048_304866


namespace NUMINAMATH_CALUDE_sum_of_fractions_zero_l3048_304860

theorem sum_of_fractions_zero (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
    (h : a + b + 2*c = 0) : 
  1 / (b^2 + 4*c^2 - a^2) + 1 / (a^2 + 4*c^2 - b^2) + 1 / (a^2 + b^2 - 4*c^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_zero_l3048_304860


namespace NUMINAMATH_CALUDE_construction_company_gravel_purchase_l3048_304885

theorem construction_company_gravel_purchase
  (total_material : ℝ)
  (sand : ℝ)
  (gravel : ℝ)
  (h1 : total_material = 14.02)
  (h2 : sand = 8.11)
  (h3 : total_material = sand + gravel) :
  gravel = 5.91 :=
by
  sorry

end NUMINAMATH_CALUDE_construction_company_gravel_purchase_l3048_304885


namespace NUMINAMATH_CALUDE_tan_alpha_value_l3048_304816

theorem tan_alpha_value (α : ℝ) :
  (3 * Real.sin (Real.pi + α) + Real.cos (-α)) / (4 * Real.sin (-α) - Real.cos (9 * Real.pi + α)) = 2 →
  Real.tan α = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l3048_304816


namespace NUMINAMATH_CALUDE_birds_wings_count_l3048_304897

/-- Calculates the total number of wings on birds purchased with money from grandparents --/
theorem birds_wings_count (money_per_grandparent : ℕ) (num_grandparents : ℕ) (cost_per_bird : ℕ) : 
  money_per_grandparent = 50 →
  num_grandparents = 4 →
  cost_per_bird = 20 →
  ((money_per_grandparent * num_grandparents) / cost_per_bird) * 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_birds_wings_count_l3048_304897


namespace NUMINAMATH_CALUDE_parallelepiped_properties_l3048_304893

/-- A rectangular parallelepiped with an inscribed sphere -/
structure Parallelepiped :=
  (k : ℝ)  -- Ratio of parallelepiped volume to sphere volume
  (h : k > 0)  -- k is positive

/-- Theorem about the angles and permissible values of k for a parallelepiped with an inscribed sphere -/
theorem parallelepiped_properties (p : Parallelepiped) :
  let α := Real.arcsin (6 / (Real.pi * p.k))
  ∃ (angle1 angle2 : ℝ),
    (angle1 = α ∧ angle2 = Real.pi - α) ∧  -- Angles at the base
    p.k ≥ 6 / Real.pi :=  -- Permissible values of k
by sorry

end NUMINAMATH_CALUDE_parallelepiped_properties_l3048_304893


namespace NUMINAMATH_CALUDE_two_digit_number_difference_l3048_304888

/-- Given a two-digit number, prove that if the difference between the original number
    and the number with interchanged digits is 54, then the difference between its two digits is 6. -/
theorem two_digit_number_difference (x y : ℕ) : 
  x < 10 → y < 10 → (10 * x + y) - (10 * y + x) = 54 → x - y = 6 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_difference_l3048_304888


namespace NUMINAMATH_CALUDE_solids_of_revolution_l3048_304824

-- Define the type for geometric solids
inductive GeometricSolid
  | Cylinder
  | HexagonalPyramid
  | Cube
  | Sphere
  | Tetrahedron

-- Define the property of being a solid of revolution
def isSolidOfRevolution : GeometricSolid → Prop :=
  fun solid => match solid with
    | GeometricSolid.Cylinder => True
    | GeometricSolid.Sphere => True
    | _ => False

-- Theorem statement
theorem solids_of_revolution :
  ∀ s : GeometricSolid,
    isSolidOfRevolution s ↔ (s = GeometricSolid.Cylinder ∨ s = GeometricSolid.Sphere) :=
by
  sorry

#check solids_of_revolution

end NUMINAMATH_CALUDE_solids_of_revolution_l3048_304824


namespace NUMINAMATH_CALUDE_area_perimeter_ratio_l3048_304858

/-- The side length of the square -/
def square_side : ℝ := 5

/-- The side length of the equilateral triangle -/
def triangle_side : ℝ := 6

/-- The area of a square given its side length -/
def square_area (side : ℝ) : ℝ := side * side

/-- The perimeter of an equilateral triangle given its side length -/
def equilateral_triangle_perimeter (side : ℝ) : ℝ := 3 * side

/-- Theorem stating the ratio of the square's area to the triangle's perimeter -/
theorem area_perimeter_ratio :
  (square_area square_side) / (equilateral_triangle_perimeter triangle_side) = 25 / 18 := by
  sorry


end NUMINAMATH_CALUDE_area_perimeter_ratio_l3048_304858


namespace NUMINAMATH_CALUDE_range_of_e_l3048_304815

theorem range_of_e (a b c d e : ℝ) 
  (sum_condition : a + b + c + d + e = 8)
  (sum_squares_condition : a^2 + b^2 + c^2 + d^2 + e^2 = 16) :
  0 ≤ e ∧ e ≤ 16/5 := by
sorry

end NUMINAMATH_CALUDE_range_of_e_l3048_304815


namespace NUMINAMATH_CALUDE_quadratic_function_property_l3048_304845

def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_function_property (f : ℝ → ℝ) (h_quad : is_quadratic f)
  (h_cond : ∀ a b : ℝ, a ≠ b → f a = f b → f (a^2 - 6*b - 1) = f (b^2 + 8)) :
  f 2 = f 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l3048_304845


namespace NUMINAMATH_CALUDE_parallelogram_opposite_sides_l3048_304865

/-- A parallelogram in a 2D Cartesian plane -/
structure Parallelogram where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

def diagonalIntersection (p : Parallelogram) : ℝ × ℝ := (0, 1)

def lineAB : LineEquation := { a := 1, b := -2, c := -2 }

theorem parallelogram_opposite_sides (p : Parallelogram) 
  (h1 : diagonalIntersection p = (0, 1))
  (h2 : lineAB = { a := 1, b := -2, c := -2 }) :
  ∃ (lineCD : LineEquation), lineCD = { a := 1, b := -2, c := 6 } := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_opposite_sides_l3048_304865


namespace NUMINAMATH_CALUDE_tom_car_lease_cost_l3048_304892

/-- Calculates the total yearly cost for Tom's car lease -/
theorem tom_car_lease_cost :
  let miles_per_week : ℕ := 4 * 50 + 3 * 100
  let cost_per_mile : ℚ := 1 / 10
  let weekly_fee : ℕ := 100
  let weeks_per_year : ℕ := 52
  (miles_per_week : ℚ) * cost_per_mile * weeks_per_year + (weekly_fee : ℚ) * weeks_per_year = 7800 := by
  sorry

end NUMINAMATH_CALUDE_tom_car_lease_cost_l3048_304892


namespace NUMINAMATH_CALUDE_polynomial_evaluation_and_coefficient_sum_l3048_304878

theorem polynomial_evaluation_and_coefficient_sum (d : ℝ) (h : d ≠ 0) :
  (10*d + 16 + 17*d^2 + 3*d^3) + (5*d + 4 + 2*d^2 + 2*d^3) = 5*d^3 + 19*d^2 + 15*d + 20 ∧
  5 + 19 + 15 + 20 = 59 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_and_coefficient_sum_l3048_304878


namespace NUMINAMATH_CALUDE_max_value_of_sum_products_l3048_304821

theorem max_value_of_sum_products (w x y z : ℝ) : 
  w ≥ 0 → x ≥ 0 → y ≥ 0 → z ≥ 0 → 
  w + x + y + z = 200 →
  w * x + x * y + y * z + w * z ≤ 10000 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_of_sum_products_l3048_304821


namespace NUMINAMATH_CALUDE_no_triples_satisfying_lcm_conditions_l3048_304879

theorem no_triples_satisfying_lcm_conditions :
  ¬∃ (x y z : ℕ+), 
    (Nat.lcm x.val y.val = 48) ∧ 
    (Nat.lcm x.val z.val = 900) ∧ 
    (Nat.lcm y.val z.val = 180) :=
by sorry

end NUMINAMATH_CALUDE_no_triples_satisfying_lcm_conditions_l3048_304879


namespace NUMINAMATH_CALUDE_elizabeth_stickers_l3048_304805

/-- Represents the number of stickers Elizabeth placed on each water bottle. -/
def stickers_per_bottle (initial_bottles : ℕ) (lost_bottles : ℕ) (stolen_bottles : ℕ) (total_stickers : ℕ) : ℕ :=
  total_stickers / (initial_bottles - lost_bottles - stolen_bottles)

/-- Theorem: Elizabeth placed 3 stickers on each remaining water bottle. -/
theorem elizabeth_stickers :
  stickers_per_bottle 10 2 1 21 = 3 := by
  sorry

end NUMINAMATH_CALUDE_elizabeth_stickers_l3048_304805


namespace NUMINAMATH_CALUDE_no_real_solutions_for_abs_equation_l3048_304833

theorem no_real_solutions_for_abs_equation :
  ∀ x : ℝ, |2*x - 6| ≠ x^2 - x + 2 := by
sorry

end NUMINAMATH_CALUDE_no_real_solutions_for_abs_equation_l3048_304833


namespace NUMINAMATH_CALUDE_shortest_distance_parabola_to_line_l3048_304884

/-- The shortest distance from a point on the parabola y = x^2 to the line 2x - y = 4 -/
theorem shortest_distance_parabola_to_line :
  let parabola := {p : ℝ × ℝ | p.2 = p.1^2}
  let line := {p : ℝ × ℝ | 2 * p.1 - p.2 = 4}
  let distance (p : ℝ × ℝ) := |2 * p.1 - p.2 - 4| / Real.sqrt 5
  ∃ (p : ℝ × ℝ), p ∈ parabola ∧
    (∀ (q : ℝ × ℝ), q ∈ parabola → distance p ≤ distance q) ∧
    distance p = 3 * Real.sqrt 5 / 5 :=
sorry

end NUMINAMATH_CALUDE_shortest_distance_parabola_to_line_l3048_304884


namespace NUMINAMATH_CALUDE_seating_theorem_l3048_304823

/-- Number of seats in a row -/
def total_seats : ℕ := 7

/-- Number of people to be seated -/
def people_to_seat : ℕ := 4

/-- Number of adjacent empty seats required -/
def adjacent_empty_seats : ℕ := 2

/-- Function to calculate the number of seating arrangements -/
def seating_arrangements (total_seats : ℕ) (people_to_seat : ℕ) (adjacent_empty_seats : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of seating arrangements -/
theorem seating_theorem :
  seating_arrangements total_seats people_to_seat adjacent_empty_seats = 336 :=
sorry

end NUMINAMATH_CALUDE_seating_theorem_l3048_304823


namespace NUMINAMATH_CALUDE_original_fraction_l3048_304853

theorem original_fraction (x y : ℚ) : 
  (1.15 * x) / (0.92 * y) = 15 / 16 → x / y = 69 / 92 := by
sorry

end NUMINAMATH_CALUDE_original_fraction_l3048_304853


namespace NUMINAMATH_CALUDE_orthocenter_centroid_perpendicular_l3048_304886

/-- Represents a triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The orthocenter of a triangle -/
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

/-- The centroid of a triangle -/
def centroid (t : Triangle) : ℝ × ℝ := sorry

/-- The area of a triangle given its vertices -/
def triangleArea (p q r : ℝ × ℝ) : ℝ := sorry

/-- Checks if a triangle is acute-angled -/
def isAcuteAngled (t : Triangle) : Prop := sorry

/-- Checks if two points are not equal -/
def notEqual (p q : ℝ × ℝ) : Prop := p ≠ q

/-- Calculates the angle between three points -/
def angle (p q r : ℝ × ℝ) : ℝ := sorry

theorem orthocenter_centroid_perpendicular (t : Triangle) :
  isAcuteAngled t →
  notEqual t.A t.B →
  notEqual t.A t.C →
  let H := orthocenter t
  let G := centroid t
  1 / triangleArea H t.A t.B + 1 / triangleArea H t.A t.C = 1 / triangleArea H t.B t.C →
  angle t.A G H = 90 := by sorry

end NUMINAMATH_CALUDE_orthocenter_centroid_perpendicular_l3048_304886


namespace NUMINAMATH_CALUDE_cubic_sum_over_product_square_l3048_304868

theorem cubic_sum_over_product_square (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (sum_zero : x + y + z = 0) (sum_prod_nonzero : x*y + x*z + y*z ≠ 0) :
  (x^3 + y^3 + z^3) / (x*y*z * (x*y + x*z + y*z)^2) = 3 / (x^2 + x*y + y^2)^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_over_product_square_l3048_304868


namespace NUMINAMATH_CALUDE_power_of_power_l3048_304849

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l3048_304849


namespace NUMINAMATH_CALUDE_bug_meeting_time_l3048_304806

theorem bug_meeting_time (r₁ r₂ v₁ v₂ : ℝ) (h₁ : r₁ = 4) (h₂ : r₂ = 3) 
  (h₃ : v₁ = 4 * Real.pi) (h₄ : v₂ = 3 * Real.pi) : ∃ t : ℝ, t = 2.5 ∧ 
  (∃ n₁ n₂ : ℕ, t * v₁ = 2 * Real.pi * r₁ * n₁ ∧ 
   t * v₂ = 2 * Real.pi * r₂ * (n₂ + 1/4)) := by
  sorry

end NUMINAMATH_CALUDE_bug_meeting_time_l3048_304806


namespace NUMINAMATH_CALUDE_bus_speed_problem_l3048_304871

/-- Proves that given the conditions of the bus problem, the average speed for the 220 km distance is 40 kmph -/
theorem bus_speed_problem (total_distance : ℝ) (total_time : ℝ) (distance_at_x : ℝ) (speed_known : ℝ) :
  total_distance = 250 →
  total_time = 6 →
  distance_at_x = 220 →
  speed_known = 60 →
  ∃ x : ℝ,
    x > 0 ∧
    (distance_at_x / x) + ((total_distance - distance_at_x) / speed_known) = total_time ∧
    x = 40 :=
by sorry

end NUMINAMATH_CALUDE_bus_speed_problem_l3048_304871


namespace NUMINAMATH_CALUDE_tank_capacity_proof_l3048_304887

/-- The capacity of a gasoline tank in gallons -/
def tank_capacity : ℚ := 100 / 3

/-- The amount of gasoline added to the tank in gallons -/
def added_gasoline : ℚ := 5

/-- The initial fill level of the tank as a fraction of its capacity -/
def initial_fill : ℚ := 3 / 4

/-- The final fill level of the tank as a fraction of its capacity -/
def final_fill : ℚ := 9 / 10

theorem tank_capacity_proof :
  (final_fill * tank_capacity - initial_fill * tank_capacity = added_gasoline) ∧
  (tank_capacity > 0) := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_proof_l3048_304887


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3048_304825

theorem simplify_and_evaluate (a : ℝ) (h : a = 2023) :
  (a + 1) / a / (a - 1 / a) = 1 / 2022 := by
sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3048_304825
