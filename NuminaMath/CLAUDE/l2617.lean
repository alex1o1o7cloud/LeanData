import Mathlib

namespace NUMINAMATH_CALUDE_first_year_exceeding_target_l2617_261759

def initial_investment : ℝ := 1.3
def annual_increase_rate : ℝ := 0.12
def target_investment : ℝ := 2.0
def start_year : ℕ := 2015

def investment (year : ℕ) : ℝ :=
  initial_investment * (1 + annual_increase_rate) ^ (year - start_year)

theorem first_year_exceeding_target :
  (∀ y < 2019, investment y ≤ target_investment) ∧
  investment 2019 > target_investment :=
sorry

end NUMINAMATH_CALUDE_first_year_exceeding_target_l2617_261759


namespace NUMINAMATH_CALUDE_second_company_visit_charge_l2617_261739

/-- Paul's Plumbing visit charge -/
def pauls_visit_charge : ℕ := 55

/-- Paul's Plumbing hourly labor charge -/
def pauls_hourly_charge : ℕ := 35

/-- Second company's hourly labor charge -/
def second_hourly_charge : ℕ := 30

/-- Number of labor hours -/
def labor_hours : ℕ := 4

/-- Second company's visit charge -/
def second_visit_charge : ℕ := 75

theorem second_company_visit_charge :
  pauls_visit_charge + labor_hours * pauls_hourly_charge =
  second_visit_charge + labor_hours * second_hourly_charge :=
by sorry

end NUMINAMATH_CALUDE_second_company_visit_charge_l2617_261739


namespace NUMINAMATH_CALUDE_cookie_problem_l2617_261725

/-- The number of cookies in each box -/
def cookies_per_box : ℕ := 12

/-- The number of boxes -/
def num_boxes : ℕ := 8

/-- The number of bags -/
def num_bags : ℕ := 9

/-- The difference in cookies between boxes and bags -/
def cookie_difference : ℕ := 33

/-- The number of cookies in each bag -/
def cookies_per_bag : ℕ := 7

theorem cookie_problem :
  cookies_per_box * num_boxes = cookies_per_bag * num_bags + cookie_difference :=
by sorry

end NUMINAMATH_CALUDE_cookie_problem_l2617_261725


namespace NUMINAMATH_CALUDE_pythagorean_squares_area_l2617_261713

theorem pythagorean_squares_area (a b c : ℝ) (h_right_angle : a^2 + b^2 = c^2)
  (h_sum_areas : a^2 + b^2 + 2*c^2 = 500) : c^2 = 500/3 :=
by sorry

end NUMINAMATH_CALUDE_pythagorean_squares_area_l2617_261713


namespace NUMINAMATH_CALUDE_thirteenth_row_sum_l2617_261714

def row_sum (n : ℕ) : ℕ :=
  3 * 2^(n-1)

theorem thirteenth_row_sum :
  row_sum 13 = 12288 :=
by sorry

end NUMINAMATH_CALUDE_thirteenth_row_sum_l2617_261714


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l2617_261757

-- Define the polynomials
def f (x : ℝ) : ℝ := 3*x^5 + 7*x^4 - 15*x^3 - 35*x^2 + 22*x + 24
def g (x : ℝ) : ℝ := x^3 + 5*x^2 - 4*x + 2
def r (x : ℝ) : ℝ := -258*x^2 + 186*x - 50

-- State the theorem
theorem polynomial_division_theorem :
  ∃ q : ℝ → ℝ, ∀ x, f x = g x * q x + r x ∧ (∀ x, r x = -258*x^2 + 186*x - 50) :=
sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l2617_261757


namespace NUMINAMATH_CALUDE_min_group_size_repunit_sum_l2617_261767

def is_repunit (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 0 ∧ n = (10^k - 1) / 9

theorem min_group_size_repunit_sum :
  ∃ m : ℕ, m > 1 ∧
    (∀ m' : ℕ, m' > 1 → m' < m →
      ¬∃ n k : ℕ, n > k ∧ k > 1 ∧
        is_repunit n ∧ is_repunit k ∧ n = k * m') ∧
    (∃ n k : ℕ, n > k ∧ k > 1 ∧
      is_repunit n ∧ is_repunit k ∧ n = k * m) ∧
  m = 101 :=
sorry

end NUMINAMATH_CALUDE_min_group_size_repunit_sum_l2617_261767


namespace NUMINAMATH_CALUDE_average_book_width_l2617_261784

def book_widths : List ℚ := [7, 3/4, 1.25, 3, 8, 2.5, 12]

theorem average_book_width :
  (book_widths.sum / book_widths.length : ℚ) = 241/49 := by
  sorry

end NUMINAMATH_CALUDE_average_book_width_l2617_261784


namespace NUMINAMATH_CALUDE_complement_M_intersect_N_l2617_261708

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.log (1 - x)}
def N : Set ℝ := {x | (2 : ℝ) ^ (x * (x - 2)) < 1}

-- State the theorem
theorem complement_M_intersect_N : 
  (Mᶜ ∩ N : Set ℝ) = {x | 1 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_complement_M_intersect_N_l2617_261708


namespace NUMINAMATH_CALUDE_complex_square_equality_l2617_261785

theorem complex_square_equality (c d : ℕ+) :
  (↑c - Complex.I * ↑d) ^ 2 = 18 - 8 * Complex.I →
  ↑c - Complex.I * ↑d = 5 - Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_square_equality_l2617_261785


namespace NUMINAMATH_CALUDE_haunted_house_entry_exit_l2617_261750

theorem haunted_house_entry_exit (total_windows : ℕ) (magical_barrier : ℕ) : 
  total_windows = 8 →
  magical_barrier = 1 →
  (total_windows - magical_barrier - 1) * (total_windows - 2) + 
  magical_barrier * (total_windows - 1) = 49 :=
by
  sorry

end NUMINAMATH_CALUDE_haunted_house_entry_exit_l2617_261750


namespace NUMINAMATH_CALUDE_train_speed_calculation_l2617_261790

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed_calculation (train_length bridge_length : ℝ) (time : ℝ) :
  train_length = 130 →
  bridge_length = 150 →
  time = 27.997760179185665 →
  ∃ (speed : ℝ), (abs (speed - 36.0036) < 0.0001 ∧ 
    speed = (train_length + bridge_length) / time * 3.6) := by
  sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l2617_261790


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2617_261772

theorem sqrt_equation_solution (n : ℝ) : Real.sqrt (8 + n) = 9 → n = 73 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2617_261772


namespace NUMINAMATH_CALUDE_expected_black_pairs_60_30_l2617_261791

/-- The expected number of adjacent black card pairs in a circular arrangement -/
def expected_black_pairs (total_cards : ℕ) (black_cards : ℕ) : ℚ :=
  (black_cards : ℚ) * ((black_cards - 1 : ℚ) / (total_cards - 1 : ℚ))

/-- Theorem: Expected number of adjacent black pairs in a 60-card deck with 30 black cards -/
theorem expected_black_pairs_60_30 :
  expected_black_pairs 60 30 = 870 / 59 := by
  sorry

end NUMINAMATH_CALUDE_expected_black_pairs_60_30_l2617_261791


namespace NUMINAMATH_CALUDE_sum_of_fractions_l2617_261738

theorem sum_of_fractions : (1 : ℚ) / 3 + 5 / 9 = 8 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l2617_261738


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l2617_261703

-- Define propositions p and q
def p (x y : ℝ) : Prop := x + y ≠ -2
def q (x y : ℝ) : Prop := ¬(x = -1 ∧ y = -1)

-- Theorem statement
theorem p_sufficient_not_necessary_for_q :
  (∀ x y : ℝ, p x y → q x y) ∧ 
  (∃ x y : ℝ, q x y ∧ ¬(p x y)) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l2617_261703


namespace NUMINAMATH_CALUDE_sum_of_angles_l2617_261706

/-- The number of 90-degree angles in a rectangle -/
def rectangle_angles : ℕ := 4

/-- The number of 90-degree angles in a square -/
def square_angles : ℕ := 4

/-- The sum of 90-degree angles in a rectangle and a square -/
def total_angles : ℕ := rectangle_angles + square_angles

theorem sum_of_angles : total_angles = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_angles_l2617_261706


namespace NUMINAMATH_CALUDE_expression_simplification_l2617_261751

theorem expression_simplification (x : ℤ) 
  (h1 : x - 3 * (x - 2) ≥ 2) 
  (h2 : 4 * x - 2 < 5 * x - 1) : 
  (3 / (x - 1) - x - 1) / ((x - 2) / (x - 1)) = -2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2617_261751


namespace NUMINAMATH_CALUDE_complex_point_location_l2617_261756

theorem complex_point_location (z : ℂ) (h : z = 1 + I) :
  let w := 2 / z + z^2
  0 < w.re ∧ 0 < w.im :=
by sorry

end NUMINAMATH_CALUDE_complex_point_location_l2617_261756


namespace NUMINAMATH_CALUDE_max_vertex_sum_l2617_261728

/-- Represents a parabola passing through specific points -/
structure Parabola where
  a : ℤ
  T : ℤ
  h_T_pos : T > 0
  h_passes_through : ∀ x y, y = a * x * (x - T) → 
    ((x = 0 ∧ y = 0) ∨ (x = T ∧ y = 0) ∨ (x = T + 1 ∧ y = 50))

/-- The sum of coordinates of the vertex of the parabola -/
def vertexSum (p : Parabola) : ℚ :=
  p.T / 2 - (p.a * p.T^2) / 4

/-- The theorem stating the maximum value of the vertex sum -/
theorem max_vertex_sum (p : Parabola) : 
  vertexSum p ≤ -23/2 :=
sorry

end NUMINAMATH_CALUDE_max_vertex_sum_l2617_261728


namespace NUMINAMATH_CALUDE_jungkook_has_bigger_number_l2617_261721

theorem jungkook_has_bigger_number :
  let yoongi_collected : ℕ := 4
  let jungkook_collected : ℕ := 6 + 3
  jungkook_collected > yoongi_collected :=
by
  sorry

end NUMINAMATH_CALUDE_jungkook_has_bigger_number_l2617_261721


namespace NUMINAMATH_CALUDE_ceiling_fraction_equality_l2617_261752

theorem ceiling_fraction_equality : 
  (⌈(23 : ℚ) / 9 - ⌈(35 : ℚ) / 23⌉⌉) / (⌈(35 : ℚ) / 9 + ⌈(9 : ℚ) * 23 / 35⌉⌉) = (1 : ℚ) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_fraction_equality_l2617_261752


namespace NUMINAMATH_CALUDE_total_shells_is_61_l2617_261702

def bucket_a_initial : ℕ := 5
def bucket_a_additional : ℕ := 12

def bucket_b_initial : ℕ := 8
def bucket_b_additional : ℕ := 15

def bucket_c_initial : ℕ := 3
def bucket_c_additional : ℕ := 18

def total_shells : ℕ := 
  (bucket_a_initial + bucket_a_additional) + 
  (bucket_b_initial + bucket_b_additional) + 
  (bucket_c_initial + bucket_c_additional)

theorem total_shells_is_61 : total_shells = 61 := by
  sorry

end NUMINAMATH_CALUDE_total_shells_is_61_l2617_261702


namespace NUMINAMATH_CALUDE_visitor_growth_and_optimal_price_l2617_261775

def visitors_2022 : ℕ := 200000
def visitors_2024 : ℕ := 288000
def cost_price : ℚ := 6
def initial_price : ℚ := 25
def initial_sales : ℕ := 300
def sales_increase : ℕ := 30
def target_profit : ℚ := 6300

theorem visitor_growth_and_optimal_price :
  -- Part 1: Average annual growth rate
  ∃ (growth_rate : ℚ),
    (1 + growth_rate) ^ 2 * visitors_2022 = visitors_2024 ∧
    growth_rate = 1/5 ∧
  -- Part 2: Optimal selling price
  ∃ (optimal_price : ℚ),
    optimal_price ≤ initial_price ∧
    (optimal_price - cost_price) *
      (initial_sales + sales_increase * (initial_price - optimal_price)) =
      target_profit ∧
    optimal_price = 20 :=
  sorry

end NUMINAMATH_CALUDE_visitor_growth_and_optimal_price_l2617_261775


namespace NUMINAMATH_CALUDE_power_of_three_equality_l2617_261792

theorem power_of_three_equality (m : ℕ) : 3^m = 27 * 81^4 * 243^3 → m = 34 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_equality_l2617_261792


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l2617_261701

open Set

def U : Set ℝ := univ

def M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

def N : Set ℝ := {x | x ≥ 0}

theorem intersection_complement_theorem :
  M ∩ (U \ N) = {x : ℝ | -1 ≤ x ∧ x < 0} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l2617_261701


namespace NUMINAMATH_CALUDE_sum_c_plus_d_l2617_261705

theorem sum_c_plus_d (a b c d : ℤ) 
  (h1 : a + b = 14) 
  (h2 : b + c = 9) 
  (h3 : a + d = 8) : 
  c + d = 3 := by
sorry

end NUMINAMATH_CALUDE_sum_c_plus_d_l2617_261705


namespace NUMINAMATH_CALUDE_least_positive_angle_theorem_l2617_261761

/-- The least positive angle θ (in degrees) satisfying cos 10° = sin 15° + sin θ is 32.5° -/
theorem least_positive_angle_theorem : 
  ∃ θ : ℝ, θ > 0 ∧ θ = 32.5 ∧ 
  (∀ φ : ℝ, φ > 0 ∧ Real.cos (10 * π / 180) = Real.sin (15 * π / 180) + Real.sin (φ * π / 180) → θ ≤ φ) ∧
  Real.cos (10 * π / 180) = Real.sin (15 * π / 180) + Real.sin (θ * π / 180) := by
  sorry


end NUMINAMATH_CALUDE_least_positive_angle_theorem_l2617_261761


namespace NUMINAMATH_CALUDE_square_filling_theorem_l2617_261798

def is_valid_permutation (p : Fin 5 → Fin 5) : Prop :=
  Function.Injective p ∧ Function.Surjective p

theorem square_filling_theorem :
  ∃ (p : Fin 5 → Fin 5), is_valid_permutation p ∧
    (p 0).val + 1 + (p 1).val + 1 = ((p 2).val + 1) * ((p 3).val + 1 - ((p 4).val + 1)) :=
by sorry

end NUMINAMATH_CALUDE_square_filling_theorem_l2617_261798


namespace NUMINAMATH_CALUDE_f_inequality_l2617_261734

/-- A function that is continuous and differentiable on ℝ -/
def ContinuousDifferentiableFunction (f : ℝ → ℝ) : Prop :=
  Continuous f ∧ Differentiable ℝ f

theorem f_inequality (f : ℝ → ℝ) 
  (h_f : ContinuousDifferentiableFunction f)
  (h_ineq : ∀ x, 2 * f x - deriv f x > 0) :
  f 1 > f 2 / Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l2617_261734


namespace NUMINAMATH_CALUDE_seven_valid_configurations_l2617_261782

/-- A polygon shape made of congruent squares -/
structure SquarePolygon where
  squares : ℕ
  shape : String

/-- Possible positions to attach an additional square -/
def AttachmentPositions : ℕ := 11

/-- A cube with one face missing requires this many squares -/
def CubeSquares : ℕ := 5

/-- The base cross-shaped polygon -/
def baseCross : SquarePolygon :=
  { squares := 6, shape := "cross" }

/-- Predicate for whether a polygon can form a cube with one face missing -/
def canFormCube (p : SquarePolygon) : Prop := sorry

/-- The number of valid configurations that can form a cube with one face missing -/
def validConfigurations : ℕ := 7

/-- Main theorem: There are exactly 7 valid configurations -/
theorem seven_valid_configurations :
  (∃ (configs : Finset SquarePolygon),
    configs.card = validConfigurations ∧
    (∀ p ∈ configs, p.squares = baseCross.squares + 1 ∧ canFormCube p) ∧
    (∀ p : SquarePolygon, p.squares = baseCross.squares + 1 →
      canFormCube p → p ∈ configs)) := by sorry

end NUMINAMATH_CALUDE_seven_valid_configurations_l2617_261782


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2617_261786

/-- Given a geometric sequence of positive terms {a_n}, prove that if the sum of logarithms of certain terms equals 6, then the product of the first and fifteenth terms is 10000. -/
theorem geometric_sequence_property (a : ℕ → ℝ) (r : ℝ) :
  (∀ n, a n > 0) →  -- Sequence of positive terms
  (∀ n, a (n + 1) = a n * r) →  -- Geometric sequence property
  Real.log (a 3) + Real.log (a 8) + Real.log (a 13) = 6 →
  a 1 * a 15 = 10000 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2617_261786


namespace NUMINAMATH_CALUDE_teacher_student_ratio_l2617_261733

theorem teacher_student_ratio 
  (initial_student_teacher_ratio : ℚ) 
  (current_teachers : ℕ) 
  (student_increase : ℕ) 
  (teacher_increase : ℕ) 
  (new_student_teacher_ratio : ℚ) 
  (h1 : initial_student_teacher_ratio = 50 / current_teachers)
  (h2 : current_teachers = 3)
  (h3 : student_increase = 50)
  (h4 : teacher_increase = 5)
  (h5 : new_student_teacher_ratio = 25)
  (h6 : (initial_student_teacher_ratio * current_teachers + student_increase) / 
        (current_teachers + teacher_increase) = new_student_teacher_ratio) :
  (1 : ℚ) / initial_student_teacher_ratio = 1 / 50 :=
sorry

end NUMINAMATH_CALUDE_teacher_student_ratio_l2617_261733


namespace NUMINAMATH_CALUDE_container_capacity_l2617_261745

theorem container_capacity : 
  ∀ (C : ℝ), 
    C > 0 → 
    (0.40 * C + 28 = 0.75 * C) → 
    C = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_container_capacity_l2617_261745


namespace NUMINAMATH_CALUDE_max_complex_norm_squared_l2617_261712

theorem max_complex_norm_squared (θ : ℝ) : 
  let z : ℂ := 2 * Complex.cos θ + Complex.I * Complex.sin θ
  ∃ (M : ℝ), M = 4 ∧ ∀ θ' : ℝ, Complex.normSq z ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_complex_norm_squared_l2617_261712


namespace NUMINAMATH_CALUDE_fraction_comparison_l2617_261731

def first_numerator : ℕ := 100^99
def first_denominator : ℕ := 9777777  -- 97...7 with 7 digits

def second_numerator : ℕ := 55555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555  -- 100 digits of 5
def second_denominator : ℕ := 55555  -- 5 digits of 5

theorem fraction_comparison :
  (first_numerator : ℚ) / first_denominator < (second_numerator : ℚ) / second_denominator :=
sorry

end NUMINAMATH_CALUDE_fraction_comparison_l2617_261731


namespace NUMINAMATH_CALUDE_multiply_subtract_equal_computation_result_l2617_261716

theorem multiply_subtract_equal (a b c : ℕ) : a * c - b * c = (a - b) * c := by sorry

theorem computation_result : 65 * 1515 - 25 * 1515 = 60600 := by sorry

end NUMINAMATH_CALUDE_multiply_subtract_equal_computation_result_l2617_261716


namespace NUMINAMATH_CALUDE_min_value_expression_l2617_261768

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 3 * b = 4) :
  1 / (a + 1) + 3 / (b + 1) ≥ 2 ∧
  (1 / (a + 1) + 3 / (b + 1) = 2 ↔ a = 1 ∧ b = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2617_261768


namespace NUMINAMATH_CALUDE_product_of_base6_digits_7891_l2617_261799

/-- The base 6 representation of a natural number -/
def toBase6 (n : ℕ) : List ℕ :=
  sorry

/-- The product of a list of natural numbers -/
def listProduct (l : List ℕ) : ℕ :=
  sorry

/-- Theorem: The product of the digits in the base 6 representation of 7891 is 0 -/
theorem product_of_base6_digits_7891 :
  listProduct (toBase6 7891) = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_of_base6_digits_7891_l2617_261799


namespace NUMINAMATH_CALUDE_acid_solution_volume_l2617_261776

/-- Given a volume of pure acid in a solution with a known concentration,
    calculate the total volume of the solution. -/
theorem acid_solution_volume (pure_acid : ℝ) (concentration : ℝ) 
    (h1 : pure_acid = 4.8)
    (h2 : concentration = 0.4) : 
    pure_acid / concentration = 12 := by
  sorry

end NUMINAMATH_CALUDE_acid_solution_volume_l2617_261776


namespace NUMINAMATH_CALUDE_nate_cooking_for_eight_l2617_261742

/-- The number of scallops per pound -/
def scallops_per_pound : ℕ := 8

/-- The cost of scallops per pound in cents -/
def cost_per_pound : ℕ := 2400

/-- The number of scallops per person -/
def scallops_per_person : ℕ := 2

/-- The total cost of scallops Nate is spending in cents -/
def total_cost : ℕ := 4800

/-- The number of people Nate is cooking for -/
def number_of_people : ℕ := total_cost / cost_per_pound * scallops_per_pound / scallops_per_person

theorem nate_cooking_for_eight : number_of_people = 8 := by
  sorry

end NUMINAMATH_CALUDE_nate_cooking_for_eight_l2617_261742


namespace NUMINAMATH_CALUDE_binomial_arithmetic_sequence_l2617_261707

theorem binomial_arithmetic_sequence (n k : ℕ) :
  (∃ (u : ℕ), u > 2 ∧ n = u^2 - 2 ∧ (k = u.choose 2 - 1 ∨ k = (u + 1).choose 2 - 1)) ↔
  (Nat.choose n (k - 1) - 2 * Nat.choose n k + Nat.choose n (k + 1) = 0) := by
  sorry

end NUMINAMATH_CALUDE_binomial_arithmetic_sequence_l2617_261707


namespace NUMINAMATH_CALUDE_min_value_expression_l2617_261743

theorem min_value_expression (r s t : ℝ) 
  (h1 : 1 ≤ r) (h2 : r ≤ s) (h3 : s ≤ t) (h4 : t ≤ 4) :
  (r - 1)^2 + (s/r - 1)^2 + (t/s - 1)^2 + (4/t - 1)^2 ≥ 12 - 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2617_261743


namespace NUMINAMATH_CALUDE_triangle_side_length_l2617_261723

theorem triangle_side_length (a b c : ℝ) (A B C : Real) :
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Positive side lengths
  2 * b = a + c →  -- Arithmetic sequence condition
  B = Real.pi / 3 →  -- 60 degrees in radians
  (1 / 2) * a * c * Real.sin B = 3 * Real.sqrt 3 →  -- Area condition
  b = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2617_261723


namespace NUMINAMATH_CALUDE_nell_baseball_cards_count_l2617_261722

/-- Represents the number of cards Nell has --/
structure NellCards where
  initialBaseball : Nat
  initialAce : Nat
  currentAce : Nat
  baseballDifference : Nat

/-- Calculates the current number of baseball cards Nell has --/
def currentBaseballCards (cards : NellCards) : Nat :=
  cards.currentAce + cards.baseballDifference

/-- Theorem stating that Nell's current baseball cards equal 178 --/
theorem nell_baseball_cards_count (cards : NellCards) 
  (h1 : cards.initialBaseball = 438)
  (h2 : cards.initialAce = 18)
  (h3 : cards.currentAce = 55)
  (h4 : cards.baseballDifference = 123) :
  currentBaseballCards cards = 178 := by
  sorry


end NUMINAMATH_CALUDE_nell_baseball_cards_count_l2617_261722


namespace NUMINAMATH_CALUDE_discount_percentage_proof_l2617_261769

/-- Represents the discount percentage on bulk photocopy orders -/
def discount_percentage : ℝ := 25

/-- Represents the regular cost per photocopy in dollars -/
def regular_cost_per_copy : ℝ := 0.02

/-- Represents the number of copies in a bulk order -/
def bulk_order_size : ℕ := 160

/-- Represents the individual savings when placing a bulk order -/
def individual_savings : ℝ := 0.40

/-- Represents the total savings when two people place a bulk order together -/
def total_savings : ℝ := 2 * individual_savings

/-- Proves that the discount percentage is correct given the problem conditions -/
theorem discount_percentage_proof :
  discount_percentage = (total_savings / (regular_cost_per_copy * bulk_order_size)) * 100 :=
by sorry

end NUMINAMATH_CALUDE_discount_percentage_proof_l2617_261769


namespace NUMINAMATH_CALUDE_divisibility_property_l2617_261726

theorem divisibility_property (n : ℕ) : (n - 1) ∣ (n^2 + n - 2) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l2617_261726


namespace NUMINAMATH_CALUDE_percentage_increase_proof_l2617_261794

def initial_earnings : ℝ := 65
def new_earnings : ℝ := 72

theorem percentage_increase_proof :
  let difference := new_earnings - initial_earnings
  let percentage_increase := (difference / initial_earnings) * 100
  ∀ ε > 0, |percentage_increase - 10.77| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_proof_l2617_261794


namespace NUMINAMATH_CALUDE_min_area_at_one_eighth_l2617_261741

-- Define the lines l₁ and l₂
def l₁ (k x y : ℝ) : Prop := k * x - 2 * y - 2 * k + 8 = 0
def l₂ (k x y : ℝ) : Prop := 2 * x + k^2 * y - 4 * k^2 - 4 = 0

-- Define the area of the quadrilateral as a function of k
noncomputable def quadrilateral_area (k : ℝ) : ℝ := 
  let x₁ := (2 * k - 8) / k
  let y₁ := 4 - k
  let x₂ := 2 * k^2 + 2
  let y₂ := 4 + 4 / k^2
  (x₁ * y₁) / 2 + (x₂ * y₂) / 2

-- State the theorem
theorem min_area_at_one_eighth (k : ℝ) (h : 0 < k ∧ k < 4) :
  ∃ (min_k : ℝ), min_k = 1/8 ∧ 
  ∀ k', 0 < k' ∧ k' < 4 → quadrilateral_area min_k ≤ quadrilateral_area k' :=
sorry

end NUMINAMATH_CALUDE_min_area_at_one_eighth_l2617_261741


namespace NUMINAMATH_CALUDE_number_ratio_l2617_261770

theorem number_ratio : 
  ∀ (s l : ℕ), 
  s > 0 → 
  l > s → 
  l - s = 16 → 
  s = 28 → 
  (l : ℚ) / s = 11 / 7 := by
sorry

end NUMINAMATH_CALUDE_number_ratio_l2617_261770


namespace NUMINAMATH_CALUDE_max_value_of_a_l2617_261732

noncomputable section

variable (x : ℝ) (a : ℝ)

def f (x : ℝ) (a : ℝ) : ℝ := (x^2 + a*x + 1) / x

def g (x : ℝ) : ℝ := Real.exp x - Real.log x + 2*x^2 + 1

theorem max_value_of_a (h : ∀ x > 0, x * f x a ≤ g x) :
  a ≤ Real.exp 1 + 1 :=
sorry

end

end NUMINAMATH_CALUDE_max_value_of_a_l2617_261732


namespace NUMINAMATH_CALUDE_recipe_soap_amount_l2617_261718

/-- Given a container capacity, ounces per cup, and total soap amount, 
    calculate the amount of soap per cup of water. -/
def soapPerCup (containerCapacity : ℚ) (ouncesPerCup : ℚ) (totalSoap : ℚ) : ℚ :=
  totalSoap / (containerCapacity / ouncesPerCup)

/-- Prove that the recipe calls for 3 tablespoons of soap per cup of water. -/
theorem recipe_soap_amount :
  soapPerCup 40 8 15 = 3 := by
  sorry

end NUMINAMATH_CALUDE_recipe_soap_amount_l2617_261718


namespace NUMINAMATH_CALUDE_triangle_problem_l2617_261709

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) (S : ℝ) (CM : ℝ) :
  b * (3 * b - c) * Real.cos A = b * a * Real.cos C →
  S = 2 * Real.sqrt 2 →
  CM = Real.sqrt 17 / 2 →
  (Real.cos A = 1 / 3) ∧
  ((b = 2 ∧ c = 3) ∨ (b = 3 / 2 ∧ c = 4)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l2617_261709


namespace NUMINAMATH_CALUDE_trig_equation_solution_l2617_261747

theorem trig_equation_solution (t : ℝ) :
  4 * (Real.sin t * Real.cos t ^ 5 + Real.cos t * Real.sin t ^ 5) + Real.sin (2 * t) ^ 3 = 1 ↔
  ∃ k : ℤ, t = (-1) ^ k * (Real.pi / 12) + k * (Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_trig_equation_solution_l2617_261747


namespace NUMINAMATH_CALUDE_bananas_and_cantaloupe_cost_l2617_261760

/-- Represents the cost of various fruits -/
structure FruitCosts where
  apples : ℝ
  bananas : ℝ
  cantaloupe : ℝ
  dates : ℝ
  figs : ℝ

/-- The conditions of the fruit purchase problem -/
def fruitProblemConditions (costs : FruitCosts) : Prop :=
  costs.apples + costs.bananas + costs.cantaloupe + costs.dates + costs.figs = 30 ∧
  costs.dates = 3 * costs.apples ∧
  costs.cantaloupe = costs.apples - costs.bananas ∧
  costs.figs = costs.bananas

/-- The theorem stating that the cost of bananas and cantaloupe is 6 -/
theorem bananas_and_cantaloupe_cost (costs : FruitCosts) 
  (h : fruitProblemConditions costs) : 
  costs.bananas + costs.cantaloupe = 6 := by
  sorry


end NUMINAMATH_CALUDE_bananas_and_cantaloupe_cost_l2617_261760


namespace NUMINAMATH_CALUDE_vins_bike_trips_l2617_261749

theorem vins_bike_trips (distance_to_school : ℕ) (distance_from_school : ℕ) (total_distance : ℕ) :
  distance_to_school = 6 →
  distance_from_school = 7 →
  total_distance = 65 →
  total_distance / (distance_to_school + distance_from_school) = 5 := by
sorry

end NUMINAMATH_CALUDE_vins_bike_trips_l2617_261749


namespace NUMINAMATH_CALUDE_polyhedron_inequality_l2617_261774

/-- A convex polyhedron is represented by its number of vertices, edges, and maximum number of triangular faces sharing a common vertex. -/
structure ConvexPolyhedron where
  V : ℕ  -- number of vertices
  E : ℕ  -- number of edges
  T : ℕ  -- maximum number of triangular faces sharing a common vertex

/-- The inequality V ≤ √E + T holds for any convex polyhedron. -/
theorem polyhedron_inequality (P : ConvexPolyhedron) : P.V ≤ Real.sqrt (P.E : ℝ) + P.T := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_inequality_l2617_261774


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l2617_261797

theorem complex_number_in_first_quadrant (m : ℝ) (h : m > 1) :
  let z : ℂ := m * (3 + Complex.I) - (2 + Complex.I)
  z.re > 0 ∧ z.im > 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l2617_261797


namespace NUMINAMATH_CALUDE_fraction_calculation_l2617_261748

theorem fraction_calculation : 
  (((4 : ℚ) / 9 + (1 : ℚ) / 9) / ((5 : ℚ) / 8 - (1 : ℚ) / 8)) = (10 : ℚ) / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l2617_261748


namespace NUMINAMATH_CALUDE_relay_race_ratio_l2617_261764

/-- Relay race problem -/
theorem relay_race_ratio (mary susan jen tiffany : ℕ) : 
  susan = jen + 10 →
  jen = 30 →
  tiffany = mary - 7 →
  mary + susan + jen + tiffany = 223 →
  mary / susan = 2 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_ratio_l2617_261764


namespace NUMINAMATH_CALUDE_circle_center_sum_l2617_261777

/-- Given a circle with equation x^2 + y^2 = 4x - 6y + 9, prove that its center (h, k) satisfies h + k = -1 -/
theorem circle_center_sum (h k : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 4*x - 6*y + 9 ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 4*h + 6*k - 9)) → 
  h + k = -1 := by
sorry

end NUMINAMATH_CALUDE_circle_center_sum_l2617_261777


namespace NUMINAMATH_CALUDE_greatest_negative_root_of_equation_l2617_261765

open Real

theorem greatest_negative_root_of_equation :
  ∃ (x : ℝ), x = -7/6 ∧ 
  (sin (π * x) - cos (2 * π * x)) / ((sin (π * x) + 1)^2 + cos (π * x)^2) = 0 ∧
  (∀ y < 0, y > x → 
    (sin (π * y) - cos (2 * π * y)) / ((sin (π * y) + 1)^2 + cos (π * y)^2) ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_greatest_negative_root_of_equation_l2617_261765


namespace NUMINAMATH_CALUDE_cross_section_area_theorem_l2617_261758

/-- A rectangular parallelepiped inscribed in a sphere -/
structure InscribedParallelepiped where
  R : ℝ  -- radius of the sphere
  diagonal_inclination : ℝ  -- angle between diagonals and base plane
  diagonal_inclination_is_45 : diagonal_inclination = Real.pi / 4

/-- The cross-section plane of the parallelepiped -/
structure CrossSectionPlane (p : InscribedParallelepiped) where
  angle_with_diagonal : ℝ  -- angle between the plane and diagonal BD₁
  angle_is_arcsin_sqrt2_4 : angle_with_diagonal = Real.arcsin (Real.sqrt 2 / 4)

/-- The area of the cross-section -/
noncomputable def cross_section_area (p : InscribedParallelepiped) (plane : CrossSectionPlane p) : ℝ :=
  2 * p.R^2 * Real.sqrt 3 / 3

/-- Theorem stating that the area of the cross-section is (2R²√3)/3 -/
theorem cross_section_area_theorem (p : InscribedParallelepiped) (plane : CrossSectionPlane p) :
    cross_section_area p plane = 2 * p.R^2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cross_section_area_theorem_l2617_261758


namespace NUMINAMATH_CALUDE_expected_value_biased_coin_l2617_261781

/-- Expected value of winnings for a biased coin flip -/
theorem expected_value_biased_coin : 
  let prob_heads : ℚ := 2/5
  let prob_tails : ℚ := 3/5
  let win_heads : ℚ := 5
  let loss_tails : ℚ := 1
  prob_heads * win_heads - prob_tails * loss_tails = 7/5 := by
sorry

end NUMINAMATH_CALUDE_expected_value_biased_coin_l2617_261781


namespace NUMINAMATH_CALUDE_unique_parallel_line_l2617_261779

/-- Two planes are parallel -/
def parallel_planes (α β : Set (Fin 3 → ℝ)) : Prop := sorry

/-- A line is contained in a plane -/
def line_in_plane (l : Set (Fin 3 → ℝ)) (p : Set (Fin 3 → ℝ)) : Prop := sorry

/-- A point is in a plane -/
def point_in_plane (x : Fin 3 → ℝ) (p : Set (Fin 3 → ℝ)) : Prop := sorry

/-- Two lines are parallel -/
def parallel_lines (l₁ l₂ : Set (Fin 3 → ℝ)) : Prop := sorry

/-- The set of all lines in a plane passing through a point -/
def lines_through_point (p : Set (Fin 3 → ℝ)) (x : Fin 3 → ℝ) : Set (Set (Fin 3 → ℝ)) := sorry

theorem unique_parallel_line 
  (α β : Set (Fin 3 → ℝ)) 
  (a : Set (Fin 3 → ℝ)) 
  (B : Fin 3 → ℝ) 
  (h₁ : parallel_planes α β) 
  (h₂ : line_in_plane a α) 
  (h₃ : point_in_plane B β) : 
  ∃! l, l ∈ lines_through_point β B ∧ parallel_lines l a := by
  sorry

end NUMINAMATH_CALUDE_unique_parallel_line_l2617_261779


namespace NUMINAMATH_CALUDE_employee_payment_l2617_261717

theorem employee_payment (x y : ℝ) : 
  x + y = 770 →
  x = 1.2 * y →
  y = 350 := by
sorry

end NUMINAMATH_CALUDE_employee_payment_l2617_261717


namespace NUMINAMATH_CALUDE_inequality_proof_l2617_261700

theorem inequality_proof (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) :
  a^4 + b^4 + c^4 - 2*(a^2*b^2 + a^2*c^2 + b^2*c^2) + a^2*b*c + b^2*a*c + c^2*a*b ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2617_261700


namespace NUMINAMATH_CALUDE_division_value_problem_l2617_261766

theorem division_value_problem (x : ℝ) : 
  (1376 / x) - 160 = 12 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_division_value_problem_l2617_261766


namespace NUMINAMATH_CALUDE_line_equation_through_point_with_slope_l2617_261787

/-- A line passing through (1, 0) with slope 3 has the equation 3x - y - 3 = 0 -/
theorem line_equation_through_point_with_slope (x y : ℝ) :
  (3 : ℝ) * x - y - 3 = 0 ↔ (y - 0 = 3 * (x - 1) ∧ (1, 0) ∈ {p : ℝ × ℝ | (3 : ℝ) * p.1 - p.2 - 3 = 0}) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_through_point_with_slope_l2617_261787


namespace NUMINAMATH_CALUDE_largest_solution_of_quartic_l2617_261740

theorem largest_solution_of_quartic (x : ℝ) : 
  x^4 - 50*x^2 + 625 = 0 → x ≤ 5 ∧ ∃ y, y^4 - 50*y^2 + 625 = 0 ∧ y = 5 :=
by sorry

end NUMINAMATH_CALUDE_largest_solution_of_quartic_l2617_261740


namespace NUMINAMATH_CALUDE_function_and_tangent_line_properties_l2617_261727

noncomputable section

-- Define the constant e
def e : ℝ := Real.exp 1

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x^2

-- Define the tangent line function
def tangentLine (b : ℝ) (x : ℝ) : ℝ := (e - 2) * x + b

theorem function_and_tangent_line_properties :
  ∃ (a b : ℝ),
    (∀ x : ℝ, tangentLine b x = (Real.exp 1 - f a 1) + (Real.exp 1 - 2 * a) * (x - 1)) ∧
    a = 1 ∧
    b = 1 ∧
    (∀ x : ℝ, x ≥ 0 → f a x > x^2 + 4*x - 14) :=
sorry

end NUMINAMATH_CALUDE_function_and_tangent_line_properties_l2617_261727


namespace NUMINAMATH_CALUDE_certain_number_proof_l2617_261719

theorem certain_number_proof (x : ℝ) : 
  (3 - (1/5) * x) - (4 - (1/7) * 210) = 114 → x = -425 :=
by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2617_261719


namespace NUMINAMATH_CALUDE_brick_fence_height_l2617_261753

/-- Calculates the height of a brick fence given its specifications -/
theorem brick_fence_height 
  (wall_length : ℕ) 
  (wall_depth : ℕ) 
  (num_walls : ℕ) 
  (total_bricks : ℕ) 
  (h1 : wall_length = 20)
  (h2 : wall_depth = 2)
  (h3 : num_walls = 4)
  (h4 : total_bricks = 800) : 
  total_bricks / (wall_length * num_walls * wall_depth) = 5 := by
sorry

end NUMINAMATH_CALUDE_brick_fence_height_l2617_261753


namespace NUMINAMATH_CALUDE_strip_covering_theorem_l2617_261710

/-- A strip of width w -/
def Strip (w : ℝ) := Set (ℝ × ℝ)

/-- A set of points can be covered by a strip -/
def Coverable (S : Set (ℝ × ℝ)) (w : ℝ) :=
  ∃ (strip : Strip w), S ⊆ strip

/-- Main theorem -/
theorem strip_covering_theorem (S : Set (ℝ × ℝ)) (n : ℕ) 
  (h1 : Fintype S)
  (h2 : Fintype.card S = n)
  (h3 : n ≥ 3)
  (h4 : ∀ (A B C : ℝ × ℝ), A ∈ S → B ∈ S → C ∈ S → 
    Coverable {A, B, C} 1) :
  Coverable S 2 := by
  sorry

end NUMINAMATH_CALUDE_strip_covering_theorem_l2617_261710


namespace NUMINAMATH_CALUDE_corn_price_is_ten_cents_l2617_261788

/-- Represents the farmer's corn production and sales --/
structure CornFarmer where
  seeds_per_ear : ℕ
  seeds_per_bag : ℕ
  cost_per_bag : ℚ
  profit : ℚ
  ears_sold : ℕ

/-- Calculates the price per ear of corn --/
def price_per_ear (farmer : CornFarmer) : ℚ :=
  let total_seeds := farmer.ears_sold * farmer.seeds_per_ear
  let bags_needed := (total_seeds + farmer.seeds_per_bag - 1) / farmer.seeds_per_bag
  let seed_cost := bags_needed * farmer.cost_per_bag
  let total_revenue := farmer.profit + seed_cost
  total_revenue / farmer.ears_sold

/-- Theorem stating the price per ear of corn is $0.10 --/
theorem corn_price_is_ten_cents (farmer : CornFarmer) 
    (h1 : farmer.seeds_per_ear = 4)
    (h2 : farmer.seeds_per_bag = 100)
    (h3 : farmer.cost_per_bag = 1/2)
    (h4 : farmer.profit = 40)
    (h5 : farmer.ears_sold = 500) : 
  price_per_ear farmer = 1/10 := by
  sorry


end NUMINAMATH_CALUDE_corn_price_is_ten_cents_l2617_261788


namespace NUMINAMATH_CALUDE_proposition_condition_l2617_261711

theorem proposition_condition (p q : Prop) 
  (h : (¬p → q) ∧ ¬(q → ¬p)) : 
  (¬q → p) ∧ ¬(p → ¬q) :=
sorry

end NUMINAMATH_CALUDE_proposition_condition_l2617_261711


namespace NUMINAMATH_CALUDE_johns_number_is_55_l2617_261773

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_reversal (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  ones * 100 + tens * 10 + hundreds

theorem johns_number_is_55 :
  ∃! n : ℕ, is_three_digit n ∧
    321 ≤ digit_reversal (2 * n + 13) ∧
    digit_reversal (2 * n + 13) ≤ 325 ∧
    n = 55 :=
sorry

end NUMINAMATH_CALUDE_johns_number_is_55_l2617_261773


namespace NUMINAMATH_CALUDE_optimal_planting_solution_l2617_261744

/-- Represents the planting problem with two types of flowers -/
structure PlantingProblem where
  costA3B4 : ℕ  -- Cost of 3 pots of A and 4 pots of B
  costA4B3 : ℕ  -- Cost of 4 pots of A and 3 pots of B
  totalPots : ℕ  -- Total number of pots to be planted
  survivalRateA : ℚ  -- Survival rate of type A flowers
  survivalRateB : ℚ  -- Survival rate of type B flowers
  maxReplacement : ℕ  -- Maximum number of pots to be replaced next year

/-- Represents the solution to the planting problem -/
structure PlantingSolution where
  costA : ℕ  -- Cost of each pot of type A flowers
  costB : ℕ  -- Cost of each pot of type B flowers
  potsA : ℕ  -- Number of pots of type A flowers to plant
  potsB : ℕ  -- Number of pots of type B flowers to plant
  totalCost : ℕ  -- Total cost of planting

/-- Theorem stating the optimal solution for the planting problem -/
theorem optimal_planting_solution (problem : PlantingProblem) 
  (h1 : problem.costA3B4 = 360)
  (h2 : problem.costA4B3 = 340)
  (h3 : problem.totalPots = 600)
  (h4 : problem.survivalRateA = 7/10)
  (h5 : problem.survivalRateB = 9/10)
  (h6 : problem.maxReplacement = 100) :
  ∃ (solution : PlantingSolution),
    solution.costA = 40 ∧
    solution.costB = 60 ∧
    solution.potsA = 200 ∧
    solution.potsB = 400 ∧
    solution.totalCost = 32000 ∧
    solution.potsA + solution.potsB = problem.totalPots ∧
    (1 - problem.survivalRateA) * solution.potsA + (1 - problem.survivalRateB) * solution.potsB ≤ problem.maxReplacement ∧
    ∀ (altSolution : PlantingSolution),
      altSolution.potsA + altSolution.potsB = problem.totalPots →
      (1 - problem.survivalRateA) * altSolution.potsA + (1 - problem.survivalRateB) * altSolution.potsB ≤ problem.maxReplacement →
      solution.totalCost ≤ altSolution.totalCost :=
by
  sorry


end NUMINAMATH_CALUDE_optimal_planting_solution_l2617_261744


namespace NUMINAMATH_CALUDE_nina_total_homework_l2617_261780

/-- Represents the number of homework assignments for a student -/
structure Homework where
  math : ℕ
  reading : ℕ

/-- Calculates the total number of homework assignments -/
def totalHomework (hw : Homework) : ℕ := hw.math + hw.reading

theorem nina_total_homework :
  let ruby : Homework := { math := 6, reading := 2 }
  let nina : Homework := { math := 4 * ruby.math, reading := 8 * ruby.reading }
  totalHomework nina = 40 := by
  sorry

end NUMINAMATH_CALUDE_nina_total_homework_l2617_261780


namespace NUMINAMATH_CALUDE_chess_draw_probability_l2617_261762

theorem chess_draw_probability (p_win p_not_lose : ℝ) 
  (h1 : p_win = 0.3) 
  (h2 : p_not_lose = 0.8) : 
  p_not_lose - p_win = 0.5 := by
sorry

end NUMINAMATH_CALUDE_chess_draw_probability_l2617_261762


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2617_261735

theorem inequality_solution_set (a : ℝ) : 
  (∃ x : ℝ, |x - 4| + |x - 3| < a) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2617_261735


namespace NUMINAMATH_CALUDE_min_students_in_class_l2617_261783

theorem min_students_in_class (boys girls : ℕ) : 
  (boys / 2 = girls * 2 / 3) →  -- Equal number of boys and girls passed
  (boys > 0) →                  -- There is at least one boy
  (girls > 0) →                 -- There is at least one girl
  (boys + girls ≥ 7) →          -- Total number of students is at least 7
  ∃ (min_students : ℕ), 
    min_students = boys + girls ∧ 
    min_students = 7 :=
by sorry

end NUMINAMATH_CALUDE_min_students_in_class_l2617_261783


namespace NUMINAMATH_CALUDE_absolute_value_sum_l2617_261763

theorem absolute_value_sum (a b c d : ℝ) 
  (h1 : a < b) (h2 : b < c) (h3 : c < d)
  (h4 : |a - b| = 3) (h5 : |b - c| = 4) (h6 : |c - d| = 5) :
  |a - d| = 12 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_sum_l2617_261763


namespace NUMINAMATH_CALUDE_common_external_tangent_y_intercept_l2617_261720

def circle1_center : ℝ × ℝ := (3, 3)
def circle2_center : ℝ × ℝ := (15, 10)
def circle1_radius : ℝ := 5
def circle2_radius : ℝ := 10

theorem common_external_tangent_y_intercept :
  ∃ (m b : ℝ), m > 0 ∧ 
  (∀ (x y : ℝ), y = m * x + b → 
    ((x - circle1_center.1)^2 + (y - circle1_center.2)^2 = circle1_radius^2 ∨
     (x - circle2_center.1)^2 + (y - circle2_center.2)^2 = circle2_radius^2) →
    ((x - circle1_center.1)^2 + (y - circle1_center.2)^2 > circle1_radius^2 ∧
     (x - circle2_center.1)^2 + (y - circle2_center.2)^2 > circle2_radius^2) ∨
    ((x - circle1_center.1)^2 + (y - circle1_center.2)^2 < circle1_radius^2 ∧
     (x - circle2_center.1)^2 + (y - circle2_center.2)^2 < circle2_radius^2)) ∧
  b = 446 / 95 := by
sorry

end NUMINAMATH_CALUDE_common_external_tangent_y_intercept_l2617_261720


namespace NUMINAMATH_CALUDE_evaluate_expression_l2617_261771

theorem evaluate_expression : 2 - (-3) - 4 - (-5) - 6 - (-7) - 8 - (-9) = 8 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2617_261771


namespace NUMINAMATH_CALUDE_only_subtraction_negative_positive_l2617_261746

theorem only_subtraction_negative_positive : 
  (1 + (-2) ≤ 0) ∧ 
  (1 - (-2) > 0) ∧ 
  (1 * (-2) ≤ 0) ∧ 
  (1 / (-2) < 0) :=
by sorry

end NUMINAMATH_CALUDE_only_subtraction_negative_positive_l2617_261746


namespace NUMINAMATH_CALUDE_caravan_feet_head_difference_l2617_261729

theorem caravan_feet_head_difference : 
  let num_hens : ℕ := 50
  let num_goats : ℕ := 45
  let num_camels : ℕ := 8
  let num_keepers : ℕ := 15
  let feet_per_hen : ℕ := 2
  let feet_per_goat : ℕ := 4
  let feet_per_camel : ℕ := 4
  let feet_per_keeper : ℕ := 2
  let total_heads : ℕ := num_hens + num_goats + num_camels + num_keepers
  let total_feet : ℕ := num_hens * feet_per_hen + num_goats * feet_per_goat + 
                        num_camels * feet_per_camel + num_keepers * feet_per_keeper
  total_feet - total_heads = 224 := by
sorry

end NUMINAMATH_CALUDE_caravan_feet_head_difference_l2617_261729


namespace NUMINAMATH_CALUDE_cake_division_possible_l2617_261795

/-- Represents the different ways a cake can be divided -/
inductive CakePortion
  | Whole
  | Half
  | Third

/-- Represents the distribution of cakes to children -/
structure CakeDistribution where
  whole : Nat
  half : Nat
  third : Nat

/-- Calculates the total portion of cake for a given distribution -/
def totalPortion (d : CakeDistribution) : Rat :=
  d.whole + d.half / 2 + d.third / 3

theorem cake_division_possible : ∃ (d : CakeDistribution),
  -- Each child gets the same amount
  totalPortion d = 13 / 6 ∧
  -- The distribution uses exactly 13 cakes
  d.whole + d.half + d.third = 13 ∧
  -- The number of half cakes is even (so they can be paired)
  d.half % 2 = 0 ∧
  -- The number of third cakes is divisible by 3 (so they can be grouped)
  d.third % 3 = 0 :=
sorry

end NUMINAMATH_CALUDE_cake_division_possible_l2617_261795


namespace NUMINAMATH_CALUDE_pyramid_height_formula_l2617_261715

/-- A right pyramid with a square base -/
structure RightPyramid where
  /-- The perimeter of the square base -/
  base_perimeter : ℝ
  /-- The distance from the apex to each vertex of the square base -/
  apex_to_vertex : ℝ

/-- The height of the pyramid from its peak to the center of the square base -/
def pyramid_height (p : RightPyramid) : ℝ :=
  sorry

theorem pyramid_height_formula (p : RightPyramid) 
  (h1 : p.base_perimeter = 40)
  (h2 : p.apex_to_vertex = 15) : 
  pyramid_height p = 5 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_height_formula_l2617_261715


namespace NUMINAMATH_CALUDE_smallest_equivalent_angle_proof_l2617_261793

/-- The smallest positive angle in [0°, 360°) with the same terminal side as 2011° -/
def smallest_equivalent_angle : ℝ := 211

/-- Two angles have the same terminal side if they differ by a multiple of 360° -/
def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, α = β + k * 360

theorem smallest_equivalent_angle_proof :
  same_terminal_side smallest_equivalent_angle 2011 ∧
  smallest_equivalent_angle ≥ 0 ∧
  smallest_equivalent_angle < 360 ∧
  ∀ θ, 0 ≤ θ ∧ θ < 360 ∧ same_terminal_side θ 2011 → θ ≥ smallest_equivalent_angle := by
  sorry


end NUMINAMATH_CALUDE_smallest_equivalent_angle_proof_l2617_261793


namespace NUMINAMATH_CALUDE_trajectory_and_intersection_l2617_261737

-- Define the circles M and N
def circle_M (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1
def circle_N (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 9

-- Define the trajectory curve C
def curve_C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1 ∧ x ≠ -2

-- Define the line l passing through (-4,0) and tangent to circle M
def line_l (x y : ℝ) : Prop := ∃ k : ℝ, y = k * (x + 4) ∧ k^2 / (1 + k^2) = 1/9

-- Theorem statement
theorem trajectory_and_intersection :
  -- The trajectory of the center of circle P forms curve C
  (∀ x y : ℝ, (∃ r : ℝ, 0 < r ∧ r < 3 ∧
    (∀ x' y' : ℝ, (x' - x)^2 + (y' - y)^2 = r^2 →
      (circle_M x' y' → (x' - x)^2 + (y' - y)^2 = (1 + r)^2) ∧
      (circle_N x' y' → (x' - x)^2 + (y' - y)^2 = (3 - r)^2))
  ) → curve_C x y) ∧
  -- The line l intersects curve C at two points with distance 18/7
  (∀ x₁ y₁ x₂ y₂ : ℝ, curve_C x₁ y₁ ∧ curve_C x₂ y₂ ∧
    line_l x₁ y₁ ∧ line_l x₂ y₂ ∧ (x₁, y₁) ≠ (x₂, y₂) →
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = (18/7)^2) :=
by sorry

end NUMINAMATH_CALUDE_trajectory_and_intersection_l2617_261737


namespace NUMINAMATH_CALUDE_daisy_crown_problem_l2617_261754

theorem daisy_crown_problem (white pink red : ℕ) : 
  white = 6 →
  pink = 9 * white →
  white + pink + red = 273 →
  4 * pink - red = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_daisy_crown_problem_l2617_261754


namespace NUMINAMATH_CALUDE_units_digit_of_quotient_l2617_261704

theorem units_digit_of_quotient (n : ℕ) : (4^1993 + 5^1993) % 9 = 0 :=
by sorry

end NUMINAMATH_CALUDE_units_digit_of_quotient_l2617_261704


namespace NUMINAMATH_CALUDE_camping_match_ratio_l2617_261755

def match_ratio (initial matches_dropped final : ℕ) : ℚ :=
  let matches_lost := initial - final
  let matches_eaten := matches_lost - matches_dropped
  matches_eaten / matches_dropped

theorem camping_match_ratio :
  match_ratio 70 10 40 = 2 := by sorry

end NUMINAMATH_CALUDE_camping_match_ratio_l2617_261755


namespace NUMINAMATH_CALUDE_rental_car_distance_l2617_261730

theorem rental_car_distance (fixed_fee : ℝ) (per_km_charge : ℝ) (total_bill : ℝ) (km_travelled : ℝ) : 
  fixed_fee = 45 →
  per_km_charge = 0.12 →
  total_bill = 74.16 →
  total_bill = fixed_fee + per_km_charge * km_travelled →
  km_travelled = 243 := by
sorry

end NUMINAMATH_CALUDE_rental_car_distance_l2617_261730


namespace NUMINAMATH_CALUDE_am_length_l2617_261789

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a line
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

-- Define the problem setup
def problem_setup (c : Circle) (l1 l2 : Line) (M A B C : ℝ × ℝ) : Prop :=
  ∃ (BC BM : ℝ),
    -- l1 is tangent to c at A
    (∃ (t : ℝ), l1.point1 = c.center + t • (A - c.center) ∧ 
                l1.point2 = c.center + (t + 1) • (A - c.center) ∧
                ‖A - c.center‖ = c.radius) ∧
    -- l2 intersects c at B and C
    (∃ (t1 t2 : ℝ), l2.point1 + t1 • (l2.point2 - l2.point1) = B ∧
                    l2.point1 + t2 • (l2.point2 - l2.point1) = C ∧
                    ‖B - c.center‖ = c.radius ∧
                    ‖C - c.center‖ = c.radius) ∧
    -- BC = 7
    BC = 7 ∧
    -- BM = 9
    BM = 9

-- Theorem statement
theorem am_length (c : Circle) (l1 l2 : Line) (M A B C : ℝ × ℝ) 
  (h : problem_setup c l1 l2 M A B C) :
  ‖A - M‖ = 12 ∨ ‖A - M‖ = 3 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_am_length_l2617_261789


namespace NUMINAMATH_CALUDE_order_of_even_monotone_increasing_l2617_261796

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def monotone_increasing_on (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ S → y ∈ S → x ≤ y → f x ≤ f y

-- State the theorem
theorem order_of_even_monotone_increasing (heven : is_even f)
  (hmono : monotone_increasing_on f (Set.Ici 0)) :
  f (-Real.pi) > f 3 ∧ f 3 > f (-2) := by
  sorry

end NUMINAMATH_CALUDE_order_of_even_monotone_increasing_l2617_261796


namespace NUMINAMATH_CALUDE_equality_implies_product_equality_l2617_261778

theorem equality_implies_product_equality (a b c : ℝ) : a = b → a * c = b * c := by sorry

end NUMINAMATH_CALUDE_equality_implies_product_equality_l2617_261778


namespace NUMINAMATH_CALUDE_yoongi_age_l2617_261736

theorem yoongi_age (yoongi_age hoseok_age : ℕ) 
  (sum_of_ages : yoongi_age + hoseok_age = 16)
  (age_difference : yoongi_age = hoseok_age + 2) : 
  yoongi_age = 9 := by
sorry

end NUMINAMATH_CALUDE_yoongi_age_l2617_261736


namespace NUMINAMATH_CALUDE_perpendicular_lines_l2617_261724

/-- The slope of the line 2x - 3y + 5 = 0 -/
def m₁ : ℚ := 2 / 3

/-- The slope of the line bx - 3y + 1 = 0 -/
def m₂ (b : ℚ) : ℚ := b / 3

/-- The condition for perpendicular lines -/
def perpendicular (m₁ m₂ : ℚ) : Prop := m₁ * m₂ = -1

theorem perpendicular_lines (b : ℚ) : 
  perpendicular m₁ (m₂ b) → b = -9/2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l2617_261724
