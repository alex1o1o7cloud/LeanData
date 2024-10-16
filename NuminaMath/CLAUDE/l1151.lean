import Mathlib

namespace NUMINAMATH_CALUDE_min_angles_in_circle_l1151_115173

theorem min_angles_in_circle (n : ℕ) (h : n ≥ 3) : ℕ :=
  let S : ℕ → ℕ := fun n =>
    if n % 2 = 1 then
      (n - 1)^2 / 4
    else
      n^2 / 4 - n / 2
  S n

#check min_angles_in_circle

end NUMINAMATH_CALUDE_min_angles_in_circle_l1151_115173


namespace NUMINAMATH_CALUDE_number_composition_proof_l1151_115152

theorem number_composition_proof : 
  let ones : ℕ := 5
  let tenths : ℕ := 7
  let hundredths : ℕ := 21
  let thousandths : ℕ := 53
  let composed_number := 
    (ones : ℝ) + 
    (tenths : ℝ) * 0.1 + 
    (hundredths : ℝ) * 0.01 + 
    (thousandths : ℝ) * 0.001
  10 * composed_number = 59.63 := by
sorry

end NUMINAMATH_CALUDE_number_composition_proof_l1151_115152


namespace NUMINAMATH_CALUDE_fallen_piece_theorem_l1151_115172

/-- A function that checks if two numbers have the same digits --/
def same_digits (a b : ℕ) : Prop := sorry

/-- The number of pages in a fallen piece of a book --/
def fallen_piece_pages (first_page last_page : ℕ) : ℕ :=
  last_page - first_page + 1

theorem fallen_piece_theorem :
  ∃ (last_page : ℕ),
    last_page > 328 ∧
    same_digits last_page 328 ∧
    fallen_piece_pages 328 last_page = 496 := by
  sorry

end NUMINAMATH_CALUDE_fallen_piece_theorem_l1151_115172


namespace NUMINAMATH_CALUDE_range_of_a_for_monotonic_f_l1151_115189

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then a * x^2 + 1 else (a + 3) * Real.exp (a * x)

theorem range_of_a_for_monotonic_f :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ∨ (∀ x y : ℝ, x < y → f a x > f a y) →
  a ∈ Set.Icc (-2) 0 ∧ a ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_monotonic_f_l1151_115189


namespace NUMINAMATH_CALUDE_lower_bound_of_expression_l1151_115133

theorem lower_bound_of_expression (L : ℤ) : 
  (∃ (S : Finset ℤ), 
    (∀ n ∈ S, L < 4*n + 7 ∧ 4*n + 7 < 120) ∧ 
    S.card = 30) →
  L = 5 :=
sorry

end NUMINAMATH_CALUDE_lower_bound_of_expression_l1151_115133


namespace NUMINAMATH_CALUDE_parabola_equation_after_coordinate_shift_l1151_115104

/-- Represents a parabola in a 2D coordinate system -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ
  eq : ℝ → ℝ := λ x => a * (x - h)^2 + k

/-- Represents a 2D coordinate system -/
structure CoordinateSystem where
  origin : ℝ × ℝ

/-- Translates a point from one coordinate system to another -/
def translate (p : ℝ × ℝ) (old_sys new_sys : CoordinateSystem) : ℝ × ℝ :=
  (p.1 - (new_sys.origin.1 - old_sys.origin.1), 
   p.2 - (new_sys.origin.2 - old_sys.origin.2))

theorem parabola_equation_after_coordinate_shift 
  (p : Parabola) 
  (old_sys new_sys : CoordinateSystem) :
  p.a = 3 ∧ 
  p.h = 0 ∧ 
  p.k = 0 ∧
  new_sys.origin = (-1, -1) →
  ∀ x y : ℝ, 
    (translate (x, y) new_sys old_sys).2 = p.eq (translate (x, y) new_sys old_sys).1 ↔
    y = 3 * (x + 1)^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_after_coordinate_shift_l1151_115104


namespace NUMINAMATH_CALUDE_five_balls_four_boxes_l1151_115113

/-- The number of ways to distribute n indistinguishable balls into k indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 6 ways to distribute 5 indistinguishable balls into 4 indistinguishable boxes -/
theorem five_balls_four_boxes : distribute_balls 5 4 = 6 := by sorry

end NUMINAMATH_CALUDE_five_balls_four_boxes_l1151_115113


namespace NUMINAMATH_CALUDE_sqrt_square_not_always_equal_to_a_l1151_115194

theorem sqrt_square_not_always_equal_to_a : ¬ ∀ a : ℝ, Real.sqrt (a^2) = a := by
  sorry

end NUMINAMATH_CALUDE_sqrt_square_not_always_equal_to_a_l1151_115194


namespace NUMINAMATH_CALUDE_tax_discount_commute_l1151_115160

theorem tax_discount_commute (P T D : ℝ) (h1 : 0 ≤ P) (h2 : 0 ≤ T) (h3 : 0 ≤ D) (h4 : D < 1) :
  P * (1 + T) * (1 - D) = P * (1 - D) * (1 + T) :=
by sorry

end NUMINAMATH_CALUDE_tax_discount_commute_l1151_115160


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_by_two_l1151_115187

theorem least_subtraction_for_divisibility_by_two : 
  ∃ (n : ℕ), n = 1 ∧ 
  (∀ m : ℕ, (9671 - m) % 2 = 0 → m ≥ n) ∧
  (9671 - n) % 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_by_two_l1151_115187


namespace NUMINAMATH_CALUDE_percentage_of_democratic_voters_l1151_115135

theorem percentage_of_democratic_voters :
  ∀ (d r : ℝ),
    d + r = 100 →
    0.8 * d + 0.3 * r = 65 →
    d = 70 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_democratic_voters_l1151_115135


namespace NUMINAMATH_CALUDE_g_negative_101_l1151_115164

/-- A function g satisfying the given functional equation -/
def g_function (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g (x * y) + x = x * g y + g x

theorem g_negative_101 (g : ℝ → ℝ) (h1 : g_function g) (h2 : g 1 = 7) : 
  g (-101) = -95 :=
sorry

end NUMINAMATH_CALUDE_g_negative_101_l1151_115164


namespace NUMINAMATH_CALUDE_distance_swam_against_current_l1151_115123

/-- Calculates the distance swam against a river current -/
theorem distance_swam_against_current
  (speed_still_water : ℝ)
  (current_speed : ℝ)
  (time : ℝ)
  (h1 : speed_still_water = 5)
  (h2 : current_speed = 1.2)
  (h3 : time = 3.1578947368421053)
  : (speed_still_water - current_speed) * time = 12 := by
  sorry

end NUMINAMATH_CALUDE_distance_swam_against_current_l1151_115123


namespace NUMINAMATH_CALUDE_mary_berry_spending_l1151_115168

theorem mary_berry_spending (total apples peaches : ℝ) (h1 : total = 34.72) (h2 : apples = 14.33) (h3 : peaches = 9.31) :
  total - (apples + peaches) = 11.08 := by
  sorry

end NUMINAMATH_CALUDE_mary_berry_spending_l1151_115168


namespace NUMINAMATH_CALUDE_complex_fraction_equals_one_tenth_l1151_115101

-- Define the expression
def complex_fraction : ℚ :=
  (⌈(23 / 9 : ℚ) - ⌈(35 / 23 : ℚ)⌉⌉ : ℚ) /
  (⌈(35 / 9 : ℚ) + ⌈(9 * 23 / 35 : ℚ)⌉⌉ : ℚ)

-- State the theorem
theorem complex_fraction_equals_one_tenth : complex_fraction = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_one_tenth_l1151_115101


namespace NUMINAMATH_CALUDE_not_fifteen_percent_less_l1151_115103

theorem not_fifteen_percent_less (A B : ℝ) (h : A = B * (1 + 0.15)) : 
  B ≠ A * (1 - 0.15) := by
sorry

end NUMINAMATH_CALUDE_not_fifteen_percent_less_l1151_115103


namespace NUMINAMATH_CALUDE_women_average_age_l1151_115178

/-- The average age of two women given the following conditions:
    1. There are initially 10 men.
    2. When two women replace two men (aged 10 and 12), the average age increases by 2 years.
    3. The number of people remains 10 after the replacement. -/
theorem women_average_age (T : ℕ) : 
  (T : ℝ) / 10 + 2 = (T - 10 - 12 + 42) / 10 → 21 = 42 / 2 := by
  sorry

end NUMINAMATH_CALUDE_women_average_age_l1151_115178


namespace NUMINAMATH_CALUDE_problem_statement_l1151_115125

theorem problem_statement (m n : ℝ) (h : 5 * m + 3 * n = 2) : 
  10 * m + 6 * n - 5 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1151_115125


namespace NUMINAMATH_CALUDE_beths_underwater_time_l1151_115153

/-- Calculates the total underwater time for a scuba diver -/
def total_underwater_time (primary_tank_time : ℕ) (supplemental_tanks : ℕ) (time_per_supplemental_tank : ℕ) : ℕ :=
  primary_tank_time + supplemental_tanks * time_per_supplemental_tank

/-- Proves that Beth's total underwater time is 8 hours -/
theorem beths_underwater_time :
  let primary_tank_time : ℕ := 2
  let supplemental_tanks : ℕ := 6
  let time_per_supplemental_tank : ℕ := 1
  total_underwater_time primary_tank_time supplemental_tanks time_per_supplemental_tank = 8 := by
  sorry

#eval total_underwater_time 2 6 1

end NUMINAMATH_CALUDE_beths_underwater_time_l1151_115153


namespace NUMINAMATH_CALUDE_runners_speed_ratio_l1151_115122

/-- Two runners with different speeds start d miles apart. When running towards each other,
    they meet in s hours. When running in the same direction, the faster runner catches up
    to the slower one in u hours. This theorem proves that the ratio of their speeds is 2. -/
theorem runners_speed_ratio
  (d : ℝ) -- distance between starting points
  (s : ℝ) -- time to meet when running towards each other
  (u : ℝ) -- time for faster runner to catch up when running in same direction
  (h_d : d > 0)
  (h_s : s > 0)
  (h_u : u > 0) :
  ∃ (v_f v_s : ℝ), v_f > v_s ∧ v_f / v_s = 2 ∧
    v_f + v_s = d / s ∧
    (v_f - v_s) * u = v_s * u :=
by sorry

end NUMINAMATH_CALUDE_runners_speed_ratio_l1151_115122


namespace NUMINAMATH_CALUDE_perfect_squares_from_products_l1151_115107

theorem perfect_squares_from_products (a b c d : ℕ) 
  (h1 : ∃ x : ℕ, a * b * c = x ^ 2)
  (h2 : ∃ x : ℕ, a * c * d = x ^ 2)
  (h3 : ∃ x : ℕ, b * c * d = x ^ 2)
  (h4 : ∃ x : ℕ, a * b * d = x ^ 2) :
  (∃ w : ℕ, a = w ^ 2) ∧ 
  (∃ x : ℕ, b = x ^ 2) ∧ 
  (∃ y : ℕ, c = y ^ 2) ∧ 
  (∃ z : ℕ, d = z ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_perfect_squares_from_products_l1151_115107


namespace NUMINAMATH_CALUDE_two_bedroom_units_count_l1151_115100

theorem two_bedroom_units_count 
  (total_units : ℕ) 
  (one_bedroom_cost two_bedroom_cost : ℕ) 
  (total_cost : ℕ) 
  (h1 : total_units = 12)
  (h2 : one_bedroom_cost = 360)
  (h3 : two_bedroom_cost = 450)
  (h4 : total_cost = 4950) :
  ∃ (one_bedroom_count two_bedroom_count : ℕ),
    one_bedroom_count + two_bedroom_count = total_units ∧
    one_bedroom_count * one_bedroom_cost + two_bedroom_count * two_bedroom_cost = total_cost ∧
    two_bedroom_count = 7 := by
  sorry

end NUMINAMATH_CALUDE_two_bedroom_units_count_l1151_115100


namespace NUMINAMATH_CALUDE_linear_equation_root_conditions_l1151_115199

/-- Conditions for roots of a linear equation -/
theorem linear_equation_root_conditions (a b : ℝ) :
  let x := -b / a
  (x > 0 ↔ a * b < 0) ∧
  (x < 0 ↔ a * b > 0) ∧
  (x = 0 ↔ b = 0 ∧ a ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_root_conditions_l1151_115199


namespace NUMINAMATH_CALUDE_fraction_sum_squared_l1151_115184

theorem fraction_sum_squared (x y z m n p : ℝ) 
  (h1 : x/m + y/n + z/p = 1)
  (h2 : m/x + n/y + p/z = 0) :
  x^2/m^2 + y^2/n^2 + z^2/p^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_squared_l1151_115184


namespace NUMINAMATH_CALUDE_aquarium_visitors_l1151_115169

theorem aquarium_visitors (total : ℕ) (ill_percent : ℚ) (not_ill : ℕ) 
  (h1 : ill_percent = 40 / 100)
  (h2 : not_ill = 300)
  (h3 : (1 - ill_percent) * total = not_ill) : 
  total = 500 := by
  sorry

end NUMINAMATH_CALUDE_aquarium_visitors_l1151_115169


namespace NUMINAMATH_CALUDE_min_triangle_forming_number_l1151_115134

def CanFormTriangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def MinTriangleForming : ℕ → Prop
| n => ∀ (S : Finset ℕ), S.card = n → (∀ x ∈ S, x ≥ 1 ∧ x ≤ 1000) →
       ∃ a b c, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ CanFormTriangle a b c

theorem min_triangle_forming_number : MinTriangleForming 16 ∧ ∀ k < 16, ¬MinTriangleForming k :=
  sorry

end NUMINAMATH_CALUDE_min_triangle_forming_number_l1151_115134


namespace NUMINAMATH_CALUDE_isosceles_triangle_ef_length_l1151_115102

/-- In an isosceles triangle DEF, G is the point where the altitude from D meets EF. -/
structure IsoscelesTriangle where
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ
  G : ℝ × ℝ
  is_isosceles : dist D E = dist D F
  altitude : (G.1 - D.1) * (E.1 - F.1) + (G.2 - D.2) * (E.2 - F.2) = 0
  on_base : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ G = (1 - t) • E + t • F

/-- The length of EF in the isosceles triangle DEF. -/
def EF_length (triangle : IsoscelesTriangle) : ℝ :=
  dist triangle.E triangle.F

/-- The theorem stating the length of EF in the specific isosceles triangle. -/
theorem isosceles_triangle_ef_length 
  (triangle : IsoscelesTriangle)
  (de_length : dist triangle.D triangle.E = 5)
  (eg_gf_ratio : dist triangle.E triangle.G = 4 * dist triangle.G triangle.F) :
  EF_length triangle = (5 * Real.sqrt 10) / 4 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_ef_length_l1151_115102


namespace NUMINAMATH_CALUDE_penny_species_count_l1151_115151

/-- The number of distinct species Penny identified at the aquarium -/
def distinctSpecies (sharks eels whales dolphins rays octopuses uniqueSpecies doubleCounted : ℕ) : ℕ :=
  sharks + eels + whales + dolphins + rays + octopuses - doubleCounted

/-- Theorem stating the number of distinct species Penny identified -/
theorem penny_species_count :
  distinctSpecies 35 15 5 12 8 25 6 3 = 97 := by
  sorry

end NUMINAMATH_CALUDE_penny_species_count_l1151_115151


namespace NUMINAMATH_CALUDE_notebooks_per_student_in_second_half_l1151_115190

/-- Given a classroom with students and notebooks, prove that each student
    in the second half has 3 notebooks. -/
theorem notebooks_per_student_in_second_half
  (total_students : ℕ)
  (total_notebooks : ℕ)
  (notebooks_per_first_half_student : ℕ)
  (h1 : total_students = 28)
  (h2 : total_notebooks = 112)
  (h3 : notebooks_per_first_half_student = 5)
  (h4 : 2 ∣ total_students) :
  (total_notebooks - (total_students / 2 * notebooks_per_first_half_student)) / (total_students / 2) = 3 :=
by sorry

end NUMINAMATH_CALUDE_notebooks_per_student_in_second_half_l1151_115190


namespace NUMINAMATH_CALUDE_max_total_marks_is_1127_l1151_115162

/-- Represents the pass requirements and scores for a student's exam -/
structure ExamResults where
  math_pass_percent : ℚ
  physics_pass_percent : ℚ
  chem_pass_percent : ℚ
  math_score : ℕ
  math_fail_margin : ℕ
  physics_score : ℕ
  physics_fail_margin : ℕ
  chem_score : ℕ
  chem_fail_margin : ℕ

/-- Calculates the maximum total marks obtainable across all subjects -/
def maxTotalMarks (results : ExamResults) : ℕ :=
  sorry

/-- Theorem stating that given the exam results, the maximum total marks is 1127 -/
theorem max_total_marks_is_1127 (results : ExamResults) 
  (h1 : results.math_pass_percent = 36/100)
  (h2 : results.physics_pass_percent = 40/100)
  (h3 : results.chem_pass_percent = 45/100)
  (h4 : results.math_score = 130)
  (h5 : results.math_fail_margin = 14)
  (h6 : results.physics_score = 120)
  (h7 : results.physics_fail_margin = 20)
  (h8 : results.chem_score = 160)
  (h9 : results.chem_fail_margin = 10) :
  maxTotalMarks results = 1127 :=
  sorry

end NUMINAMATH_CALUDE_max_total_marks_is_1127_l1151_115162


namespace NUMINAMATH_CALUDE_tens_digit_of_sum_is_zero_l1151_115110

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  (n / 100) = (n % 10) - 1 ∧
  ((n / 10) % 10) = (n % 10) + 3

def reverse_number (n : ℕ) : ℕ :=
  (n % 10) * 100 + ((n / 10) % 10) * 10 + (n / 100)

theorem tens_digit_of_sum_is_zero (n : ℕ) (h : is_valid_number n) :
  ((n + reverse_number n) / 10) % 10 = 0 :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_sum_is_zero_l1151_115110


namespace NUMINAMATH_CALUDE_inheritance_calculation_l1151_115137

theorem inheritance_calculation (x : ℝ) : 
  (0.2 * x + 0.1 * (0.8 * x) = 10500) → x = 37500 := by
  sorry

end NUMINAMATH_CALUDE_inheritance_calculation_l1151_115137


namespace NUMINAMATH_CALUDE_range_of_sine_function_l1151_115159

open Set
open Real

theorem range_of_sine_function (x : ℝ) (h : 0 < x ∧ x < 2*π/3) :
  ∃ y, y ∈ Ioo 0 1 ∧ y = 2 * sin (x + π/6) - 1 ∧
  ∀ z, z = 2 * sin (x + π/6) - 1 → z ∈ Ioc 0 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_sine_function_l1151_115159


namespace NUMINAMATH_CALUDE_cube_preserves_order_l1151_115163

theorem cube_preserves_order (a b : ℝ) : a > b → a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_preserves_order_l1151_115163


namespace NUMINAMATH_CALUDE_complex_modulus_l1151_115185

theorem complex_modulus (z : ℂ) (h : (1 + Complex.I) * z = -3 + Complex.I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l1151_115185


namespace NUMINAMATH_CALUDE_cardinal_transitivity_l1151_115195

-- Define the theorem
theorem cardinal_transitivity (α β γ : Cardinal) 
  (h1 : α < β) (h2 : β < γ) : α < γ := by
  sorry

end NUMINAMATH_CALUDE_cardinal_transitivity_l1151_115195


namespace NUMINAMATH_CALUDE_exterior_angle_of_regular_polygon_l1151_115144

theorem exterior_angle_of_regular_polygon (n : ℕ) (h : (n - 2) * 180 = 720) :
  360 / n = 60 := by
  sorry

end NUMINAMATH_CALUDE_exterior_angle_of_regular_polygon_l1151_115144


namespace NUMINAMATH_CALUDE_age_of_fifteenth_student_l1151_115196

theorem age_of_fifteenth_student
  (total_students : ℕ)
  (average_age : ℝ)
  (group1_count : ℕ)
  (group1_average : ℝ)
  (group2_count : ℕ)
  (group2_average : ℝ)
  (h1 : total_students = 15)
  (h2 : average_age = 15)
  (h3 : group1_count = 4)
  (h4 : group1_average = 14)
  (h5 : group2_count = 10)
  (h6 : group2_average = 16)
  (h7 : group1_count + group2_count + 1 = total_students) :
  (total_students : ℝ) * average_age - 
  ((group1_count : ℝ) * group1_average + (group2_count : ℝ) * group2_average) = 9 :=
by sorry

end NUMINAMATH_CALUDE_age_of_fifteenth_student_l1151_115196


namespace NUMINAMATH_CALUDE_cost_of_green_pill_l1151_115188

/-- Prove that the cost of one green pill is $20 -/
theorem cost_of_green_pill (treatment_duration : ℕ) (daily_green_pills : ℕ) (daily_pink_pills : ℕ) 
  (total_cost : ℕ) : ℕ :=
by
  sorry

#check cost_of_green_pill 3 1 1 819

end NUMINAMATH_CALUDE_cost_of_green_pill_l1151_115188


namespace NUMINAMATH_CALUDE_p_18_equals_negative_one_l1151_115155

/-- A quadratic function with specific properties -/
def QuadraticFunction (d e f : ℝ) : ℝ → ℝ := fun x ↦ d * x^2 + e * x + f

/-- Theorem: For a quadratic function with given properties, p(18) = -1 -/
theorem p_18_equals_negative_one
  (d e f : ℝ)
  (p : ℝ → ℝ)
  (h_quad : p = QuadraticFunction d e f)
  (h_sym : p 6 = p 12)
  (h_max : IsLocalMax p 10)
  (h_p0 : p 0 = -1) :
  p 18 = -1 := by
  sorry

end NUMINAMATH_CALUDE_p_18_equals_negative_one_l1151_115155


namespace NUMINAMATH_CALUDE_f_increasing_l1151_115191

-- Define the function f(x) = x³ + x + 1
def f (x : ℝ) : ℝ := x^3 + x + 1

-- Theorem statement
theorem f_increasing : ∀ x y : ℝ, x < y → f x < f y := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_l1151_115191


namespace NUMINAMATH_CALUDE_unique_solution_l1151_115139

theorem unique_solution : ∃! (x : ℕ+), (1 : ℕ)^(x.val + 2) + 2^(x.val + 1) + 3^(x.val - 1) + 4^x.val = 1170 ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1151_115139


namespace NUMINAMATH_CALUDE_sum_ages_in_three_years_l1151_115147

/-- The sum of Josiah and Hans' ages in three years -/
def sum_ages (hans_age : ℕ) (josiah_multiplier : ℕ) (years_later : ℕ) : ℕ :=
  (hans_age * josiah_multiplier + hans_age) + 2 * years_later

/-- Theorem stating the sum of Josiah and Hans' ages in three years -/
theorem sum_ages_in_three_years :
  sum_ages 15 3 3 = 66 := by
  sorry

#eval sum_ages 15 3 3

end NUMINAMATH_CALUDE_sum_ages_in_three_years_l1151_115147


namespace NUMINAMATH_CALUDE_height_equals_median_implies_angle_leq_60_height_equals_median_and_bisector_implies_equilateral_l1151_115118

/-- Represents a triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Checks if a triangle is acute -/
def Triangle.isAcute (t : Triangle) : Prop := sorry

/-- Returns the length of the height from vertex A to side BC -/
def Triangle.heightAH (t : Triangle) : ℝ := sorry

/-- Returns the length of the median from vertex B to side AC -/
def Triangle.medianBM (t : Triangle) : ℝ := sorry

/-- Returns the length of the angle bisector from vertex C -/
def Triangle.angleBisectorCD (t : Triangle) : ℝ := sorry

/-- Returns the measure of angle ABC in degrees -/
def Triangle.angleABC (t : Triangle) : ℝ := sorry

/-- Checks if a triangle is equilateral -/
def Triangle.isEquilateral (t : Triangle) : Prop := sorry

/-- Theorem: If the largest height AH is equal to the median BM in an acute triangle,
    then angle ABC is not greater than 60 degrees -/
theorem height_equals_median_implies_angle_leq_60 (t : Triangle) 
  (h1 : t.isAcute) 
  (h2 : t.heightAH = t.medianBM) : 
  t.angleABC ≤ 60 := by sorry

/-- Theorem: If the height AH is equal to both the median BM and the angle bisector CD 
    in an acute triangle, then the triangle is equilateral -/
theorem height_equals_median_and_bisector_implies_equilateral (t : Triangle) 
  (h1 : t.isAcute) 
  (h2 : t.heightAH = t.medianBM) 
  (h3 : t.heightAH = t.angleBisectorCD) : 
  t.isEquilateral := by sorry

end NUMINAMATH_CALUDE_height_equals_median_implies_angle_leq_60_height_equals_median_and_bisector_implies_equilateral_l1151_115118


namespace NUMINAMATH_CALUDE_reservoir_overflow_time_l1151_115111

/-- Represents the state of a reservoir with four pipes -/
structure ReservoirSystem where
  fill_rate_a : ℚ  -- Rate at which Pipe A fills the reservoir (in reservoir/hour)
  fill_rate_c : ℚ  -- Rate at which Pipe C fills the reservoir (in reservoir/hour)
  drain_rate_b : ℚ  -- Rate at which Pipe B drains the reservoir (in reservoir/hour)
  drain_rate_d : ℚ  -- Rate at which Pipe D drains the reservoir (in reservoir/hour)
  initial_level : ℚ  -- Initial water level in the reservoir (as a fraction of full)

/-- Calculates the time until the reservoir overflows -/
def time_to_overflow (sys : ReservoirSystem) : ℚ :=
  sorry

/-- Theorem stating the time to overflow for the given reservoir system -/
theorem reservoir_overflow_time : 
  let sys : ReservoirSystem := {
    fill_rate_a := 1/3,
    fill_rate_c := 1/5,
    drain_rate_b := -1/4,
    drain_rate_d := -1/6,
    initial_level := 1/6
  }
  time_to_overflow sys = 83/4 := by
  sorry


end NUMINAMATH_CALUDE_reservoir_overflow_time_l1151_115111


namespace NUMINAMATH_CALUDE_calculation_difference_is_zero_l1151_115183

def salesTaxRate : ℝ := 0.08
def originalPrice : ℝ := 120.00
def mainDiscount : ℝ := 0.25
def additionalDiscount : ℝ := 0.10
def numberOfSweaters : ℕ := 4

def amyCalculation : ℝ :=
  numberOfSweaters * (originalPrice * (1 + salesTaxRate) * (1 - mainDiscount) * (1 - additionalDiscount))

def bobCalculation : ℝ :=
  numberOfSweaters * (originalPrice * (1 - mainDiscount) * (1 - additionalDiscount) * (1 + salesTaxRate))

theorem calculation_difference_is_zero :
  amyCalculation = bobCalculation :=
by sorry

end NUMINAMATH_CALUDE_calculation_difference_is_zero_l1151_115183


namespace NUMINAMATH_CALUDE_correct_proposition_l1151_115129

-- Define proposition p
def p : Prop := ∃ x : ℝ, x^2 + x + 1 < 0

-- Define proposition q
def q : Prop := ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - 1 ≥ 0

-- Theorem to prove
theorem correct_proposition : ¬p ∧ q := by
  sorry

end NUMINAMATH_CALUDE_correct_proposition_l1151_115129


namespace NUMINAMATH_CALUDE_circle_properties_l1151_115166

-- Define the circle family
def circle_family (t : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*(t+3)*x - 2*t*y + t^2 + 4*t + 8 = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y - 3 = 0

-- Define the tangent line m
def line_m (x y : ℝ) : Prop := y = -1 ∧ x = 2

theorem circle_properties :
  -- Part 1: Centers lie on y = x - 3
  (∀ t : ℝ, t ≠ -1 → ∃ x y : ℝ, circle_family t x y ∧ y = x - 3) ∧
  -- Part 2: Maximum chord length is 2√2
  (∃ max_length : ℝ, max_length = 2 * Real.sqrt 2 ∧
    ∀ t : ℝ, t ≠ -1 →
      ∀ x₁ y₁ x₂ y₂ : ℝ,
        circle_family t x₁ y₁ ∧ line_l x₁ y₁ ∧
        circle_family t x₂ y₂ ∧ line_l x₂ y₂ →
        Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) ≤ max_length) ∧
  -- Part 3: Line m is tangent to all circles
  (∀ t : ℝ, t ≠ -1 →
    ∃ x y : ℝ, circle_family t x y ∧ line_m x y ∧
    ∀ x' y' : ℝ, circle_family t x' y' →
      (x' - x)^2 + (y' - y)^2 ≥ 0 ∧
      ((x' - x)^2 + (y' - y)^2 = 0 → x' = x ∧ y' = y)) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_properties_l1151_115166


namespace NUMINAMATH_CALUDE_twenty_fifth_number_l1151_115161

def twisted_sequence (n : ℕ) : ℕ :=
  match n with
  | 0 => 0  -- We define 0th term as 0 for convenience
  | 1 => 1  -- First term is 1
  | n + 1 => 
    if n % 5 = 0 then twisted_sequence n + 1  -- Every 6th number (5th index) is previous + 1
    else 2 * twisted_sequence n  -- Otherwise, double the previous number

theorem twenty_fifth_number : twisted_sequence 25 = 69956 := by
  sorry

end NUMINAMATH_CALUDE_twenty_fifth_number_l1151_115161


namespace NUMINAMATH_CALUDE_last_remaining_number_l1151_115198

def josephus_sequence (n : ℕ) : ℕ → ℕ
| 0 => 1
| m + 1 => 2 * josephus_sequence n m

theorem last_remaining_number :
  ∃ k : ℕ, josephus_sequence 200 k = 128 ∧ josephus_sequence 200 (k + 1) > 200 :=
by sorry

end NUMINAMATH_CALUDE_last_remaining_number_l1151_115198


namespace NUMINAMATH_CALUDE_subtraction_absolute_value_l1151_115105

theorem subtraction_absolute_value (x y : ℝ) : 
  |8 - 3| - |x - y| = 3 → |x - y| = 2 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_absolute_value_l1151_115105


namespace NUMINAMATH_CALUDE_sum_with_radical_conjugate_l1151_115117

theorem sum_with_radical_conjugate :
  ∃ (x : ℝ), x^2 = 2023 ∧ (15 - x) + (15 + x) = 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_with_radical_conjugate_l1151_115117


namespace NUMINAMATH_CALUDE_complex_fraction_power_l1151_115112

theorem complex_fraction_power (i : ℂ) : i^2 = -1 → ((1 + i) / (1 - i))^2013 = i := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_power_l1151_115112


namespace NUMINAMATH_CALUDE_factor_expression_l1151_115124

theorem factor_expression (x : ℝ) : 75 * x^19 + 225 * x^38 = 75 * x^19 * (1 + 3 * x^19) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1151_115124


namespace NUMINAMATH_CALUDE_coloring_books_per_shelf_l1151_115192

theorem coloring_books_per_shelf 
  (initial_stock : ℕ) 
  (sold : ℕ) 
  (shelves : ℕ) 
  (h1 : initial_stock = 87) 
  (h2 : sold = 33) 
  (h3 : shelves = 9) 
  (h4 : shelves > 0) : 
  (initial_stock - sold) / shelves = 6 := by
sorry

end NUMINAMATH_CALUDE_coloring_books_per_shelf_l1151_115192


namespace NUMINAMATH_CALUDE_ten_factorial_div_four_factorial_l1151_115174

theorem ten_factorial_div_four_factorial :
  (∃ (ten_factorial : ℕ), ten_factorial = 3628800 ∧ ten_factorial / 24 = 151200) :=
by sorry

end NUMINAMATH_CALUDE_ten_factorial_div_four_factorial_l1151_115174


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1151_115154

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, (a - 2) * x^2 - 2 * (a - 2) * x - 4 < 0) ↔ a ∈ Set.Ioc (-2) 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1151_115154


namespace NUMINAMATH_CALUDE_pot_count_l1151_115120

/-- The number of pots given the number of flowers and sticks per pot and the total number of flowers and sticks -/
def number_of_pots (flowers_per_pot : ℕ) (sticks_per_pot : ℕ) (total_items : ℕ) : ℕ :=
  total_items / (flowers_per_pot + sticks_per_pot)

/-- Theorem stating that there are 466 pots given the conditions -/
theorem pot_count : number_of_pots 53 181 109044 = 466 := by
  sorry

#eval number_of_pots 53 181 109044

end NUMINAMATH_CALUDE_pot_count_l1151_115120


namespace NUMINAMATH_CALUDE_min_value_expression_l1151_115167

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (6 * z) / (3 * x + y) + (6 * x) / (y + 3 * z) + (2 * y) / (x + 2 * z) ≥ 3 ∧
  ((6 * z) / (3 * x + y) + (6 * x) / (y + 3 * z) + (2 * y) / (x + 2 * z) = 3 ↔ 3 * x = y ∧ y = 3 * z) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1151_115167


namespace NUMINAMATH_CALUDE_lunchroom_students_l1151_115170

theorem lunchroom_students (tables : ℕ) (avg_students : ℚ) : 
  tables = 34 →
  avg_students = 5666666667 / 1000000000 →
  ∃ (total_students : ℕ), total_students = 204 ∧ total_students % tables = 0 := by
  sorry

end NUMINAMATH_CALUDE_lunchroom_students_l1151_115170


namespace NUMINAMATH_CALUDE_road_length_difference_l1151_115150

/-- The length of Telegraph Road in kilometers -/
def telegraph_road_length : ℝ := 162

/-- The length of Pardee Road in meters -/
def pardee_road_length : ℝ := 12000

/-- Conversion factor from meters to kilometers -/
def meters_to_km : ℝ := 1000

theorem road_length_difference :
  telegraph_road_length - (pardee_road_length / meters_to_km) = 150 := by
  sorry

end NUMINAMATH_CALUDE_road_length_difference_l1151_115150


namespace NUMINAMATH_CALUDE_line_direction_vector_l1151_115176

def point1 : ℝ × ℝ := (-4, 3)
def point2 : ℝ × ℝ := (2, -2)
def direction_vector (a : ℝ) : ℝ × ℝ := (a, -1)

theorem line_direction_vector (a : ℝ) :
  (∃ k : ℝ, k ≠ 0 ∧ (point2.1 - point1.1, point2.2 - point1.2) = k • direction_vector a) →
  a = 6/5 := by
sorry

end NUMINAMATH_CALUDE_line_direction_vector_l1151_115176


namespace NUMINAMATH_CALUDE_seating_arrangements_l1151_115121

/-- The number of seats on the bench -/
def total_seats : ℕ := 7

/-- The number of people to be seated -/
def people_to_seat : ℕ := 4

/-- The number of empty seats -/
def empty_seats : ℕ := total_seats - people_to_seat

/-- The total number of unrestricted seating arrangements -/
def total_arrangements : ℕ := 840

theorem seating_arrangements :
  (∃ (arrangements_with_adjacent : ℕ),
    arrangements_with_adjacent = total_arrangements - 24 ∧
    arrangements_with_adjacent = 816) ∧
  (∃ (arrangements_without_all_empty_adjacent : ℕ),
    arrangements_without_all_empty_adjacent = total_arrangements - 120 ∧
    arrangements_without_all_empty_adjacent = 720) := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_l1151_115121


namespace NUMINAMATH_CALUDE_sum_of_squares_and_square_of_sum_l1151_115157

theorem sum_of_squares_and_square_of_sum : (4 + 8)^2 + (4^2 + 8^2) = 224 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_and_square_of_sum_l1151_115157


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1151_115119

theorem line_passes_through_fixed_point :
  ∀ (m : ℝ), (2*m - 1) * 2 - (m + 3) * 3 - (m - 11) = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1151_115119


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slopes_l1151_115138

/-- The slopes of the asymptotes for the hyperbola (y^2/16) - (x^2/9) = 1 are ±4/3 -/
theorem hyperbola_asymptote_slopes :
  let f (x y : ℝ) := y^2 / 16 - x^2 / 9
  ∃ (m : ℝ), m = 4/3 ∧ 
    (∀ ε > 0, ∃ M > 0, ∀ x y, |x| > M → |y| > M → f x y = 1 → 
      (|y - m*x| < ε*|x| ∨ |y + m*x| < ε*|x|)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slopes_l1151_115138


namespace NUMINAMATH_CALUDE_chord_length_polar_curves_l1151_115197

/-- The length of the chord formed by the intersection of two curves in polar coordinates -/
theorem chord_length_polar_curves : 
  ∃ (ρ₁ ρ₂ : ℝ → ℝ) (θ₁ θ₂ : ℝ),
    (∀ θ, ρ₁ θ * Real.sin θ = 1) →
    (∀ θ, ρ₂ θ = 4 * Real.sin θ) →
    ∃ (x₁ y₁ x₂ y₂ : ℝ),
      x₁^2 + y₁^2 = (ρ₁ θ₁)^2 ∧
      x₂^2 + y₂^2 = (ρ₁ θ₂)^2 ∧
      x₁ = ρ₁ θ₁ * Real.cos θ₁ ∧
      y₁ = ρ₁ θ₁ * Real.sin θ₁ ∧
      x₂ = ρ₁ θ₂ * Real.cos θ₂ ∧
      y₂ = ρ₁ θ₂ * Real.sin θ₂ ∧
      Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_polar_curves_l1151_115197


namespace NUMINAMATH_CALUDE_bacon_only_count_l1151_115165

theorem bacon_only_count (total_bacon : ℕ) (both : ℕ) (h1 : total_bacon = 569) (h2 : both = 218) :
  total_bacon - both = 351 := by
  sorry

end NUMINAMATH_CALUDE_bacon_only_count_l1151_115165


namespace NUMINAMATH_CALUDE_expression_simplification_l1151_115128

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 5 + 1) :
  (x + 1) / (x + 2) / (x - 2 + 3 / (x + 2)) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1151_115128


namespace NUMINAMATH_CALUDE_integer_fraction_characterization_l1151_115136

def solution_set : Set (Nat × Nat) :=
  {(2, 1), (3, 1), (2, 2), (5, 2), (5, 3), (1, 2), (1, 3)}

theorem integer_fraction_characterization (m n : Nat) :
  m > 0 ∧ n > 0 →
  (∃ k : Int, (n^3 + 1 : Int) = k * (m^2 - 1)) ↔ (m, n) ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_integer_fraction_characterization_l1151_115136


namespace NUMINAMATH_CALUDE_product_of_successive_numbers_l1151_115171

theorem product_of_successive_numbers : 
  let n : ℝ := 64.4980619863884
  let product := n * (n + 1)
  ∀ ε > 0, |product - 4225| < ε
:= by sorry

end NUMINAMATH_CALUDE_product_of_successive_numbers_l1151_115171


namespace NUMINAMATH_CALUDE_sqrt_ratio_implies_sum_ratio_l1151_115132

theorem sqrt_ratio_implies_sum_ratio (x y : ℝ) (h : x ≥ 0) (k : y > 0) :
  (Real.sqrt x / Real.sqrt y = 5) → ((x + y) / (2 * y) = 13) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_ratio_implies_sum_ratio_l1151_115132


namespace NUMINAMATH_CALUDE_age_difference_proof_l1151_115177

theorem age_difference_proof (ramesh_age mahesh_age : ℝ) : 
  ramesh_age / mahesh_age = 2 / 5 →
  (ramesh_age + 10) / (mahesh_age + 10) = 10 / 15 →
  mahesh_age - ramesh_age = 7.5 := by
sorry

end NUMINAMATH_CALUDE_age_difference_proof_l1151_115177


namespace NUMINAMATH_CALUDE_sum_is_integer_l1151_115108

theorem sum_is_integer (x y z : ℝ) 
  (h1 : x^2 = y + 2) 
  (h2 : y^2 = z + 2) 
  (h3 : z^2 = x + 2) : 
  ∃ n : ℤ, (x + y + z : ℝ) = n := by
sorry

end NUMINAMATH_CALUDE_sum_is_integer_l1151_115108


namespace NUMINAMATH_CALUDE_lin_peeled_fifteen_l1151_115126

/-- Represents the potato peeling scenario -/
structure PotatoPeeling where
  total_potatoes : ℕ
  homer_rate : ℕ
  christen_rate : ℕ
  lin_rate : ℕ
  christen_join_time : ℕ
  lin_join_time : ℕ

/-- Calculates the number of potatoes Lin peeled -/
def lin_potatoes_peeled (scenario : PotatoPeeling) : ℕ :=
  sorry

/-- Theorem stating that Lin peeled 15 potatoes -/
theorem lin_peeled_fifteen (scenario : PotatoPeeling) 
  (h1 : scenario.total_potatoes = 60)
  (h2 : scenario.homer_rate = 2)
  (h3 : scenario.christen_rate = 3)
  (h4 : scenario.lin_rate = 4)
  (h5 : scenario.christen_join_time = 6)
  (h6 : scenario.lin_join_time = 9) :
  lin_potatoes_peeled scenario = 15 := by
  sorry

end NUMINAMATH_CALUDE_lin_peeled_fifteen_l1151_115126


namespace NUMINAMATH_CALUDE_number_of_boys_l1151_115186

/-- The number of boys in a class, given the average weights and number of students. -/
theorem number_of_boys (avg_weight_boys : ℝ) (avg_weight_class : ℝ) (total_students : ℕ)
  (num_girls : ℕ) (avg_weight_girls : ℝ)
  (h1 : avg_weight_boys = 48)
  (h2 : avg_weight_class = 45)
  (h3 : total_students = 25)
  (h4 : num_girls = 15)
  (h5 : avg_weight_girls = 40.5) :
  total_students - num_girls = 10 := by
  sorry

#check number_of_boys

end NUMINAMATH_CALUDE_number_of_boys_l1151_115186


namespace NUMINAMATH_CALUDE_arithmetic_matrix_properties_l1151_115181

/-- Represents a matrix with the given properties -/
def ArithmeticMatrix (n : ℕ) (d : ℕ → ℝ) : Prop :=
  n ≥ 3 ∧
  ∀ m k, m ≤ n → k ≤ n → 
    (∃ a : ℕ → ℕ → ℝ, 
      a m k = 1 + (k - 1) * d m ∧
      (∀ i, i ≤ n → a i 1 = 1) ∧
      (∀ i j, i ≤ n → j < n → a i (j + 1) - a i j = d i) ∧
      (∀ i j, i < n → j ≤ n → a (i + 1) j - a i j = a (i + 1) 1 - a i 1))

/-- The main theorem -/
theorem arithmetic_matrix_properties {n : ℕ} {d : ℕ → ℝ} 
  (h : ArithmeticMatrix n d) :
  (∃ c : ℝ, d 2 - d 1 = d 3 - d 2) ∧
  (∀ m, 3 ≤ m → m ≤ n → d m = (2 - m) * d 1 + (m - 1) * d 2) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_matrix_properties_l1151_115181


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1151_115145

theorem right_triangle_hypotenuse : ∀ (a b c : ℝ),
  a = 80 ∧ b = 150 ∧ c^2 = a^2 + b^2 → c = 170 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1151_115145


namespace NUMINAMATH_CALUDE_period_of_cos_3x_l1151_115180

/-- The period of cos(3x) is 2π/3 -/
theorem period_of_cos_3x :
  let f : ℝ → ℝ := λ x ↦ Real.cos (3 * x)
  ∃ T : ℝ, T > 0 ∧ ∀ x, f (x + T) = f x ∧ ∀ S, 0 < S ∧ S < T → ∃ x, f (x + S) ≠ f x :=
by sorry

end NUMINAMATH_CALUDE_period_of_cos_3x_l1151_115180


namespace NUMINAMATH_CALUDE_problem_statement_l1151_115106

theorem problem_statement (x y : ℝ) (h : |2*x - y| + Real.sqrt (x + 3*y - 7) = 0) :
  (Real.sqrt ((x - y)^2)) / (y - x) = 1 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1151_115106


namespace NUMINAMATH_CALUDE_probability_red_or_blue_specific_l1151_115141

/-- Represents the probability of drawing a red or blue marble after a previous draw -/
def probability_red_or_blue (red blue yellow : ℕ) : ℚ :=
  let total := red + blue + yellow
  let p_yellow := yellow / total
  let p_not_yellow := 1 - p_yellow
  let p_red_or_blue_after_yellow := (red + blue) / (total - 1)
  let p_red_or_blue_after_not_yellow := (red + blue) / total
  p_yellow * p_red_or_blue_after_yellow + p_not_yellow * p_red_or_blue_after_not_yellow

/-- Theorem stating the probability of drawing a red or blue marble
    after a previous draw from a bag with 4 red, 3 blue, and 6 yellow marbles -/
theorem probability_red_or_blue_specific :
  probability_red_or_blue 4 3 6 = 91 / 169 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_or_blue_specific_l1151_115141


namespace NUMINAMATH_CALUDE_color_change_probability_is_0_15_l1151_115143

/-- Represents the duration of each color in the traffic light cycle -/
structure TrafficLightCycle where
  green : ℕ
  yellow : ℕ
  blue : ℕ
  red : ℕ

/-- Calculates the probability of observing a color change in a traffic light cycle -/
def probability_of_color_change (cycle : TrafficLightCycle) (observation_window : ℕ) : ℚ :=
  let total_cycle_time := cycle.green + cycle.yellow + cycle.blue + cycle.red
  let favorable_time := 3 * observation_window  -- 3 color transitions
  (favorable_time : ℚ) / total_cycle_time

/-- Theorem stating the probability of observing a color change is 0.15 for the given cycle -/
theorem color_change_probability_is_0_15 :
  let cycle := TrafficLightCycle.mk 45 5 10 40
  let observation_window := 5
  probability_of_color_change cycle observation_window = 15 / 100 := by
  sorry


end NUMINAMATH_CALUDE_color_change_probability_is_0_15_l1151_115143


namespace NUMINAMATH_CALUDE_runners_visibility_probability_l1151_115182

/-- Represents a runner on a circular track -/
structure Runner where
  name : String
  lapTime : ℕ
  direction : Bool  -- True for counterclockwise, False for clockwise

/-- Represents the circular track -/
structure Track where
  circumference : ℝ
  photoCoverage : ℝ
  shadowInterval : ℕ
  shadowDuration : ℕ

/-- Calculates the probability of both runners being visible in the photo -/
def calculateVisibilityProbability (sarah : Runner) (sam : Runner) (track : Track) : ℚ :=
  sorry

/-- The main theorem to prove -/
theorem runners_visibility_probability :
  let sarah : Runner := ⟨"Sarah", 120, true⟩
  let sam : Runner := ⟨"Sam", 100, false⟩
  let track : Track := ⟨1, 1/3, 45, 15⟩
  calculateVisibilityProbability sarah sam track = 1333/6000 := by
  sorry

end NUMINAMATH_CALUDE_runners_visibility_probability_l1151_115182


namespace NUMINAMATH_CALUDE_parabola_vertex_l1151_115175

/-- The parabola defined by y = x^2 - 2 has its vertex at (0, -2) -/
theorem parabola_vertex (x y : ℝ) : 
  y = x^2 - 2 → (0, -2) = (x, y) ↔ x = 0 ∧ y = -2 := by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1151_115175


namespace NUMINAMATH_CALUDE_exactly_four_false_l1151_115109

/-- Represents a statement about the number of false statements -/
inductive Statement
  | one
  | two
  | three
  | four
  | five

/-- Returns true if the statement is consistent with the given number of false statements -/
def isConsistent (s : Statement) (numFalse : Nat) : Bool :=
  match s with
  | .one => numFalse = 1
  | .two => numFalse = 2
  | .three => numFalse = 3
  | .four => numFalse = 4
  | .five => numFalse = 5

/-- The list of all statements on the card -/
def allStatements : List Statement := [.one, .two, .three, .four, .five]

/-- Counts the number of false statements given a predicate -/
def countFalse (pred : Statement → Bool) : Nat :=
  allStatements.filter (fun s => !pred s) |>.length

theorem exactly_four_false :
  ∃ (pred : Statement → Bool),
    (∀ s, pred s ↔ isConsistent s (countFalse pred)) ∧
    countFalse pred = 4 := by
  sorry

end NUMINAMATH_CALUDE_exactly_four_false_l1151_115109


namespace NUMINAMATH_CALUDE_a_value_l1151_115115

theorem a_value (a : ℝ) : 3 ∈ ({1, -a^2, a-1} : Set ℝ) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_a_value_l1151_115115


namespace NUMINAMATH_CALUDE_triangle_abc_area_l1151_115127

/-- Reflection of a point over the y-axis -/
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

/-- Reflection of a point over the line y = -x -/
def reflect_neg_x (p : ℝ × ℝ) : ℝ × ℝ := (-p.2, -p.1)

/-- Calculate the area of a triangle given three points -/
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  0.5 * abs ((x1 - x3) * (y2 - y1) - (x1 - x2) * (y3 - y1))

theorem triangle_abc_area :
  let A : ℝ × ℝ := (2, 3)
  let B : ℝ × ℝ := reflect_y_axis A
  let C : ℝ × ℝ := reflect_neg_x B
  triangle_area A B C = 10 := by sorry

end NUMINAMATH_CALUDE_triangle_abc_area_l1151_115127


namespace NUMINAMATH_CALUDE_smallest_perfect_square_multiplier_l1151_115193

def y : ℕ := 3^(4^(5^(6^(7^(8^(9^10))))))

theorem smallest_perfect_square_multiplier :
  ∃ (k : ℕ), k > 0 ∧ 
  (∃ (n : ℕ), k * y = n^2) ∧
  (∀ (m : ℕ), m > 0 → m < k → ¬∃ (n : ℕ), m * y = n^2) ∧
  k = 75 := by
sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_multiplier_l1151_115193


namespace NUMINAMATH_CALUDE_highest_score_is_242_l1151_115142

/-- Represents a batsman's statistics -/
structure BatsmanStats where
  total_innings : ℕ
  average : ℚ
  score_difference : ℕ
  average_drop : ℚ

/-- Calculates the highest score of a batsman given their statistics -/
def highest_score (stats : BatsmanStats) : ℕ :=
  sorry

/-- Theorem stating that for the given conditions, the highest score is 242 -/
theorem highest_score_is_242 (stats : BatsmanStats) 
  (h1 : stats.total_innings = 60)
  (h2 : stats.average = 55)
  (h3 : stats.score_difference = 200)
  (h4 : stats.average_drop = 3) :
  highest_score stats = 242 :=
by sorry

end NUMINAMATH_CALUDE_highest_score_is_242_l1151_115142


namespace NUMINAMATH_CALUDE_ceiling_sqrt_225_l1151_115114

theorem ceiling_sqrt_225 : ⌈Real.sqrt 225⌉ = 15 := by sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_225_l1151_115114


namespace NUMINAMATH_CALUDE_intersection_of_sets_l1151_115156

theorem intersection_of_sets :
  let A : Set ℝ := {x | -2 < x ∧ x < 3}
  let B : Set ℝ := {x | ∃ n : ℤ, x = 2 * n}
  A ∩ B = {0, 2} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l1151_115156


namespace NUMINAMATH_CALUDE_roots_arithmetic_progression_implies_sum_zero_l1151_115146

theorem roots_arithmetic_progression_implies_sum_zero 
  (a b c : ℝ) 
  (p₁ p₂ q₁ q₂ : ℝ) 
  (h₁ : a * p₁^2 + b * p₁ + c = 0)
  (h₂ : a * p₂^2 + b * p₂ + c = 0)
  (h₃ : c * q₁^2 + b * q₁ + a = 0)
  (h₄ : c * q₂^2 + b * q₂ + a = 0)
  (h₅ : ∃ (d : ℝ), d ≠ 0 ∧ q₁ - p₁ = d ∧ p₂ - q₁ = d ∧ q₂ - p₂ = d)
  (h₆ : p₁ ≠ q₁ ∧ q₁ ≠ p₂ ∧ p₂ ≠ q₂) :
  a + c = 0 := by
sorry

end NUMINAMATH_CALUDE_roots_arithmetic_progression_implies_sum_zero_l1151_115146


namespace NUMINAMATH_CALUDE_gcd_612_840_468_l1151_115116

theorem gcd_612_840_468 : Nat.gcd 612 (Nat.gcd 840 468) = 12 := by
  sorry

end NUMINAMATH_CALUDE_gcd_612_840_468_l1151_115116


namespace NUMINAMATH_CALUDE_new_light_wattage_l1151_115131

theorem new_light_wattage (original_wattage : ℝ) (increase_percentage : ℝ) :
  original_wattage = 110 →
  increase_percentage = 30 →
  original_wattage * (1 + increase_percentage / 100) = 143 :=
by sorry

end NUMINAMATH_CALUDE_new_light_wattage_l1151_115131


namespace NUMINAMATH_CALUDE_division_result_l1151_115179

theorem division_result : (180 : ℚ) / (12 + 13 * 3 - 5) = 90 / 23 := by
  sorry

end NUMINAMATH_CALUDE_division_result_l1151_115179


namespace NUMINAMATH_CALUDE_enemy_plane_hit_probability_l1151_115148

theorem enemy_plane_hit_probability 
  (prob_A : ℝ) 
  (prob_B : ℝ) 
  (h_prob_A : prob_A = 0.7) 
  (h_prob_B : prob_B = 0.5) : 
  1 - (1 - prob_A) * (1 - prob_B) = 0.85 := by
sorry

end NUMINAMATH_CALUDE_enemy_plane_hit_probability_l1151_115148


namespace NUMINAMATH_CALUDE_albert_investment_final_amount_l1151_115149

/-- Calculates the final amount after compound interest --/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Theorem stating the final amount after compound interest for Albert's investment --/
theorem albert_investment_final_amount :
  let principal : ℝ := 6500
  let rate : ℝ := 0.065
  let time : ℕ := 2
  let final_amount := compound_interest principal rate time
  ∃ ε > 0, |final_amount - 7359.46| < ε :=
by sorry

end NUMINAMATH_CALUDE_albert_investment_final_amount_l1151_115149


namespace NUMINAMATH_CALUDE_ruth_school_days_l1151_115158

/-- Ruth's school schedule -/
def school_schedule (days_per_week : ℝ) : Prop :=
  let hours_per_day : ℝ := 8
  let math_class_fraction : ℝ := 0.25
  let math_hours_per_week : ℝ := 10
  (hours_per_day * days_per_week * math_class_fraction = math_hours_per_week)

theorem ruth_school_days : ∃ (d : ℝ), school_schedule d ∧ d = 5 := by
  sorry

end NUMINAMATH_CALUDE_ruth_school_days_l1151_115158


namespace NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l1151_115130

theorem quadratic_inequality_equivalence :
  ∀ x : ℝ, x * (2 * x + 3) < -15 ↔ x ∈ Set.Ioo (-5/2) 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l1151_115130


namespace NUMINAMATH_CALUDE_bad_carrots_count_l1151_115140

theorem bad_carrots_count (carol_carrots mom_carrots brother_carrots good_carrots : ℕ) 
  (h1 : carol_carrots = 29)
  (h2 : mom_carrots = 16)
  (h3 : brother_carrots = 23)
  (h4 : good_carrots = 52) :
  carol_carrots + mom_carrots + brother_carrots - good_carrots = 16 := by
sorry

end NUMINAMATH_CALUDE_bad_carrots_count_l1151_115140
