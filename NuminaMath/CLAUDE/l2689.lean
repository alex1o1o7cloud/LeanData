import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2689_268971

/-- Given an arithmetic sequence {a_n} where a_3 and a_15 are the roots of x^2 - 6x + 8 = 0,
    the sum a_7 + a_8 + a_9 + a_10 + a_11 is equal to 15. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) : 
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  (a 3)^2 - 6 * (a 3) + 8 = 0 →  -- a_3 is a root
  (a 15)^2 - 6 * (a 15) + 8 = 0 →  -- a_15 is a root
  a 7 + a 8 + a 9 + a 10 + a 11 = 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2689_268971


namespace NUMINAMATH_CALUDE_propositions_correctness_l2689_268960

theorem propositions_correctness : 
  (∃ x : ℝ, x^2 ≥ x) ∧ 
  (4 ≥ 3) ∧ 
  ¬(∀ x : ℝ, x^2 ≥ x) ∧
  ¬(∀ x : ℝ, x^2 ≠ 1 ↔ (x ≠ 1 ∨ x ≠ -1)) := by
  sorry

end NUMINAMATH_CALUDE_propositions_correctness_l2689_268960


namespace NUMINAMATH_CALUDE_min_value_implies_a_l2689_268964

theorem min_value_implies_a (f : ℝ → ℝ) (a : ℝ) :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x = Real.sin x ^ 2 - 2 * a * Real.sin x + 1) →
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≥ 1/2) →
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = 1/2) →
  a = Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_implies_a_l2689_268964


namespace NUMINAMATH_CALUDE_m_eq_one_sufficient_not_necessary_l2689_268968

/-- A function f(x) = ax^2 is a power function if a = 1 -/
def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, a = 1 ∧ ∀ x, f x = a * x^2

/-- The function f(x) = (m^2 - 4m + 4)x^2 -/
def f (m : ℝ) : ℝ → ℝ := λ x ↦ (m^2 - 4*m + 4) * x^2

/-- Theorem: m = 1 is sufficient but not necessary for f to be a power function -/
theorem m_eq_one_sufficient_not_necessary :
  (∃ m : ℝ, m = 1 → is_power_function (f m)) ∧
  ¬(∀ m : ℝ, is_power_function (f m) → m = 1) :=
by sorry

end NUMINAMATH_CALUDE_m_eq_one_sufficient_not_necessary_l2689_268968


namespace NUMINAMATH_CALUDE_expression_value_l2689_268917

theorem expression_value (x y : ℤ) (hx : x = 3) (hy : y = 4) : 3 * x - 2 * y = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2689_268917


namespace NUMINAMATH_CALUDE_one_positive_integer_satisfies_condition_l2689_268952

theorem one_positive_integer_satisfies_condition : 
  ∃! (n : ℕ), n > 0 ∧ 21 - 3 * n > 15 :=
sorry

end NUMINAMATH_CALUDE_one_positive_integer_satisfies_condition_l2689_268952


namespace NUMINAMATH_CALUDE_percentage_difference_l2689_268913

theorem percentage_difference (a b : ℝ) 
  (ha : 3 = 0.15 * a) 
  (hb : 3 = 0.25 * b) : 
  a - b = 8 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l2689_268913


namespace NUMINAMATH_CALUDE_divisors_of_572_divisors_of_572a3bc_case1_divisors_of_572a3bc_case2_l2689_268988

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem divisors_of_572 :
  count_divisors 572 = 12 :=
sorry

theorem divisors_of_572a3bc_case1 (a b c : ℕ) 
  (ha : is_prime a) (hb : is_prime b) (hc : is_prime c)
  (ha_gt : a > 20) (hb_gt : b > 20) (hc_gt : c > 20)
  (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
  count_divisors (572 * a^3 * b * c) = 192 :=
sorry

theorem divisors_of_572a3bc_case2 :
  count_divisors (572 * 31^3 * 32 * 33) = 384 :=
sorry

end NUMINAMATH_CALUDE_divisors_of_572_divisors_of_572a3bc_case1_divisors_of_572a3bc_case2_l2689_268988


namespace NUMINAMATH_CALUDE_fourth_number_nth_row_l2689_268938

/-- The kth number in the triangular array -/
def triangular_array (k : ℕ) : ℕ := 2^(k - 1)

/-- The position of the 4th number from left to right in the nth row -/
def fourth_number_position (n : ℕ) : ℕ := n * (n - 1) / 2 + 4

theorem fourth_number_nth_row (n : ℕ) (h : n ≥ 4) :
  triangular_array (fourth_number_position n) = 2^((n^2 - n + 6) / 2) :=
sorry

end NUMINAMATH_CALUDE_fourth_number_nth_row_l2689_268938


namespace NUMINAMATH_CALUDE_box_product_digits_l2689_268998

def box_product (n : ℕ) : ℕ := n * 100 + 28 * 4

theorem box_product_digits :
  (∀ n : ℕ, n ≤ 2 → box_product n < 1000) ∧
  (∀ n : ℕ, n ≥ 3 → box_product n ≥ 1000) :=
by sorry

end NUMINAMATH_CALUDE_box_product_digits_l2689_268998


namespace NUMINAMATH_CALUDE_cookie_count_l2689_268926

theorem cookie_count (bags : ℕ) (cookies_per_bag : ℕ) (h1 : bags = 37) (h2 : cookies_per_bag = 19) :
  bags * cookies_per_bag = 703 := by
  sorry

end NUMINAMATH_CALUDE_cookie_count_l2689_268926


namespace NUMINAMATH_CALUDE_min_area_triangle_abc_l2689_268995

/-- Triangle ABC with A at origin, B at (48,18), and C with integer coordinates has minimum area 3 -/
theorem min_area_triangle_abc : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (48, 18)
  ∃ (min_area : ℝ), min_area = 3 ∧ 
    ∀ (C : ℤ × ℤ), 
      let area := (1/2) * |A.1 * (B.2 - C.2) + B.1 * C.2 + C.1 * A.2 - (B.2 * C.1 + A.1 * B.2 + C.2 * A.1)|
      area ≥ min_area :=
by
  sorry


end NUMINAMATH_CALUDE_min_area_triangle_abc_l2689_268995


namespace NUMINAMATH_CALUDE_equation_solution_l2689_268937

theorem equation_solution (x : ℝ) (h : x ≥ -1) :
  Real.sqrt (x + 1) - 1 = x / (Real.sqrt (x + 1) + 1) := by
  sorry

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l2689_268937


namespace NUMINAMATH_CALUDE_data_analytics_course_hours_l2689_268940

/-- Represents the total hours spent on a data analytics course -/
def course_total_hours (weeks : ℕ) (weekly_class_hours : ℕ) (weekly_homework_hours : ℕ) 
  (lab_sessions : ℕ) (lab_session_hours : ℕ) (project_hours : List ℕ) : ℕ :=
  weeks * (weekly_class_hours + weekly_homework_hours) + 
  lab_sessions * lab_session_hours + 
  project_hours.sum

/-- Theorem stating the total hours spent on the specific data analytics course -/
theorem data_analytics_course_hours : 
  course_total_hours 24 10 4 8 6 [10, 14, 18] = 426 := by
  sorry

end NUMINAMATH_CALUDE_data_analytics_course_hours_l2689_268940


namespace NUMINAMATH_CALUDE_traditionalist_fraction_l2689_268939

theorem traditionalist_fraction (num_provinces : ℕ) (num_traditionalists_per_province : ℚ) 
  (total_progressives : ℚ) :
  num_provinces = 4 →
  num_traditionalists_per_province = total_progressives / 12 →
  (num_provinces : ℚ) * num_traditionalists_per_province / 
    (total_progressives + (num_provinces : ℚ) * num_traditionalists_per_province) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_traditionalist_fraction_l2689_268939


namespace NUMINAMATH_CALUDE_henry_walk_distance_l2689_268943

/-- Represents a 2D point --/
structure Point where
  x : Float
  y : Float

/-- Calculates the distance between two points --/
def distance (p1 p2 : Point) : Float :=
  Float.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Converts meters to feet --/
def metersToFeet (meters : Float) : Float :=
  meters * 3.281

theorem henry_walk_distance : 
  let start := Point.mk 0 0
  let end_point := Point.mk 40 (-(metersToFeet 15 + 48))
  Float.abs (distance start end_point - 105.1) < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_henry_walk_distance_l2689_268943


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l2689_268972

-- Define a geometric sequence
def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the problem statement
theorem geometric_sequence_problem (a : ℕ → ℝ) :
  is_geometric a →
  (a 3)^2 + 7*(a 3) + 9 = 0 →
  (a 7)^2 + 7*(a 7) + 9 = 0 →
  (a 5 = 3 ∨ a 5 = -3) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l2689_268972


namespace NUMINAMATH_CALUDE_probability_d_divides_z_l2689_268908

def D : Finset Nat := Finset.filter (λ x => 100 % x = 0) (Finset.range 101)
def Z : Finset Nat := Finset.range 101

theorem probability_d_divides_z : 
  (Finset.sum D (λ d => (Finset.filter (λ z => z % d = 0) Z).card)) / (D.card * Z.card) = 217 / 900 := by
  sorry

end NUMINAMATH_CALUDE_probability_d_divides_z_l2689_268908


namespace NUMINAMATH_CALUDE_sports_club_overlap_l2689_268945

theorem sports_club_overlap (N B T BT Neither : ℕ) : 
  N = 35 →
  B = 15 →
  T = 18 →
  Neither = 5 →
  B + T - BT = N - Neither →
  BT = 3 :=
by sorry

end NUMINAMATH_CALUDE_sports_club_overlap_l2689_268945


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l2689_268989

/-- Calculates the molecular weight of a compound given its composition and atomic weights -/
def molecularWeight (fe_count o_count ca_count c_count : ℕ) 
                    (fe_weight o_weight ca_weight c_weight : ℝ) : ℝ :=
  fe_count * fe_weight + o_count * o_weight + ca_count * ca_weight + c_count * c_weight

/-- Theorem stating that the molecular weight of the given compound is 223.787 amu -/
theorem compound_molecular_weight :
  let fe_count : ℕ := 2
  let o_count : ℕ := 3
  let ca_count : ℕ := 1
  let c_count : ℕ := 2
  let fe_weight : ℝ := 55.845
  let o_weight : ℝ := 15.999
  let ca_weight : ℝ := 40.078
  let c_weight : ℝ := 12.011
  molecularWeight fe_count o_count ca_count c_count fe_weight o_weight ca_weight c_weight = 223.787 := by
  sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l2689_268989


namespace NUMINAMATH_CALUDE_trailingZeroes_15_factorial_base12_l2689_268999

/-- The number of trailing zeroes in the base 12 representation of 15! -/
def trailingZeroesBase12Factorial15 : ℕ :=
  min (Nat.factorial 15 / 12^5) 1

theorem trailingZeroes_15_factorial_base12 :
  trailingZeroesBase12Factorial15 = 5 := by
  sorry

end NUMINAMATH_CALUDE_trailingZeroes_15_factorial_base12_l2689_268999


namespace NUMINAMATH_CALUDE_inequality_condition_l2689_268947

theorem inequality_condition (b : ℝ) (h : b > 0) :
  (∃ x : ℝ, |x - 5| + |x - 2| < b) ↔ b > 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_condition_l2689_268947


namespace NUMINAMATH_CALUDE_complex_number_properties_l2689_268965

def complex_number (m : ℝ) : ℂ := (m^2 - 5*m + 6 : ℝ) + (m^2 - 3*m : ℝ) * Complex.I

theorem complex_number_properties :
  (∀ m : ℝ, (complex_number m).im = 0 ↔ m = 0 ∨ m = 3) ∧
  (∀ m : ℝ, (complex_number m).re = 0 ↔ m = 2) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_properties_l2689_268965


namespace NUMINAMATH_CALUDE_total_pens_equals_sum_l2689_268915

/-- The number of pens given to friends -/
def pens_given : ℕ := 22

/-- The number of pens kept for herself -/
def pens_kept : ℕ := 34

/-- The total number of pens bought by her parents -/
def total_pens : ℕ := pens_given + pens_kept

/-- Theorem stating that the total number of pens is the sum of pens given and pens kept -/
theorem total_pens_equals_sum : total_pens = pens_given + pens_kept := by sorry

end NUMINAMATH_CALUDE_total_pens_equals_sum_l2689_268915


namespace NUMINAMATH_CALUDE_locus_of_centers_l2689_268933

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 4
def C₂ (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 9

-- Define external tangency to C₁
def externally_tangent_C₁ (a b r : ℝ) : Prop := a^2 + b^2 = (r + 2)^2

-- Define internal tangency to C₂
def internally_tangent_C₂ (a b r : ℝ) : Prop := (a - 1)^2 + b^2 = (3 - r)^2

theorem locus_of_centers (a b : ℝ) : 
  (∃ r : ℝ, externally_tangent_C₁ a b r ∧ internally_tangent_C₂ a b r) → 
  4 * a^2 + 4 * b^2 - 25 = 0 := by
sorry

end NUMINAMATH_CALUDE_locus_of_centers_l2689_268933


namespace NUMINAMATH_CALUDE_screen_paper_difference_l2689_268984

/-- The perimeter of a square-shaped piece of paper is shorter than the height of a computer screen. 
    The height of the screen is 100 cm, and the side of the square paper is 20 cm. 
    This theorem proves that the difference between the screen height and the paper perimeter is 20 cm. -/
theorem screen_paper_difference (screen_height paper_side : ℝ) 
  (h1 : screen_height = 100)
  (h2 : paper_side = 20)
  (h3 : 4 * paper_side < screen_height) : 
  screen_height - 4 * paper_side = 20 := by
  sorry

end NUMINAMATH_CALUDE_screen_paper_difference_l2689_268984


namespace NUMINAMATH_CALUDE_soda_consumption_l2689_268914

theorem soda_consumption (carol_soda bob_soda : ℝ) 
  (h1 : carol_soda = 20)
  (h2 : bob_soda = carol_soda * 1.25)
  (h3 : carol_soda ≥ 0)
  (h4 : bob_soda ≥ 0) :
  ∃ (transfer : ℝ),
    0 ≤ transfer ∧
    transfer ≤ bob_soda * 0.2 ∧
    carol_soda * 0.8 + transfer = bob_soda * 0.8 - transfer ∧
    carol_soda * 0.8 + transfer + (bob_soda * 0.8 - transfer) = 36 :=
by sorry

end NUMINAMATH_CALUDE_soda_consumption_l2689_268914


namespace NUMINAMATH_CALUDE_hongfu_supermarket_salt_purchase_l2689_268950

/-- The number of bags of salt initially purchased by Hongfu Supermarket -/
def initial_salt : ℕ := 1200

/-- The fraction of salt sold in the first month -/
def first_month_sold : ℚ := 2/5

/-- The number of bags of salt sold in the second month -/
def second_month_sold : ℕ := 420

/-- The ratio of sold salt to remaining salt after the second month -/
def sold_to_remaining_ratio : ℚ := 3

theorem hongfu_supermarket_salt_purchase :
  initial_salt = 1200 ∧
  (initial_salt : ℚ) * first_month_sold + second_month_sold =
    sold_to_remaining_ratio * (initial_salt - (initial_salt : ℚ) * first_month_sold - second_month_sold) :=
by sorry

end NUMINAMATH_CALUDE_hongfu_supermarket_salt_purchase_l2689_268950


namespace NUMINAMATH_CALUDE_karen_ham_sandwich_days_l2689_268996

/-- The number of school days in a week -/
def school_days : ℕ := 5

/-- The number of days Karen packs peanut butter sandwiches -/
def peanut_butter_days : ℕ := 2

/-- The number of days Karen packs cake -/
def cake_days : ℕ := 1

/-- The probability of packing a ham sandwich and cake on the same day -/
def prob_ham_and_cake : ℚ := 12 / 100

/-- The number of days Karen packs ham sandwiches -/
def ham_days : ℕ := school_days - peanut_butter_days

theorem karen_ham_sandwich_days :
  ham_days = 3 ∧
  (ham_days : ℚ) / school_days * (cake_days : ℚ) / school_days = prob_ham_and_cake :=
sorry

end NUMINAMATH_CALUDE_karen_ham_sandwich_days_l2689_268996


namespace NUMINAMATH_CALUDE_function_shift_l2689_268904

noncomputable def f (x : ℝ) (φ : ℝ) := Real.sin (1/2 * x + φ)

theorem function_shift (φ : ℝ) (h1 : |φ| < Real.pi/2) 
  (h2 : ∀ x, f x φ = f (Real.pi/3 - x) φ) :
  ∀ x, f (x + Real.pi/3) φ = Real.cos (1/2 * x) := by
sorry

end NUMINAMATH_CALUDE_function_shift_l2689_268904


namespace NUMINAMATH_CALUDE_geometric_progression_equality_l2689_268967

/-- Given a geometric progression a, b, c, d, prove that 
    (a^2 + b^2 + c^2)(b^2 + c^2 + d^2) = (ab + bc + cd)^2 -/
theorem geometric_progression_equality 
  (a b c d : ℝ) (h : ∃ (q : ℝ), b = a * q ∧ c = b * q ∧ d = c * q) : 
  (a^2 + b^2 + c^2) * (b^2 + c^2 + d^2) = (a*b + b*c + c*d)^2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_equality_l2689_268967


namespace NUMINAMATH_CALUDE_ceiling_sqrt_200_l2689_268932

theorem ceiling_sqrt_200 : ⌈Real.sqrt 200⌉ = 15 := by sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_200_l2689_268932


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l2689_268929

/-- A line y = 3x + c is tangent to the parabola y^2 = 12x if and only if c = 1 -/
theorem line_tangent_to_parabola (c : ℝ) : 
  (∃ x y : ℝ, y = 3*x + c ∧ y^2 = 12*x ∧ 
   ∀ x' y' : ℝ, y' = 3*x' + c → y'^2 = 12*x' → (x', y') = (x, y)) ↔ 
  c = 1 := by sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l2689_268929


namespace NUMINAMATH_CALUDE_wire_length_for_cube_l2689_268946

-- Define the length of one edge of the cube
def edge_length : ℝ := 13

-- Define the number of edges in a cube
def cube_edges : ℕ := 12

-- Theorem stating the total wire length needed for the cube
theorem wire_length_for_cube : edge_length * cube_edges = 156 := by
  sorry

end NUMINAMATH_CALUDE_wire_length_for_cube_l2689_268946


namespace NUMINAMATH_CALUDE_evaluate_expression_l2689_268969

theorem evaluate_expression (a : ℝ) (h : a = 2) : (5 * a^2 - 13 * a + 4) * (2 * a - 3) = -2 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2689_268969


namespace NUMINAMATH_CALUDE_sunflower_plants_count_l2689_268986

/-- The number of corn plants -/
def corn_plants : ℕ := 81

/-- The number of tomato plants -/
def tomato_plants : ℕ := 63

/-- The maximum number of plants in one row -/
def max_plants_per_row : ℕ := 9

/-- The number of rows for corn plants -/
def corn_rows : ℕ := corn_plants / max_plants_per_row

/-- The number of rows for tomato plants -/
def tomato_rows : ℕ := tomato_plants / max_plants_per_row

/-- The number of rows for sunflower plants -/
def sunflower_rows : ℕ := max corn_rows tomato_rows

/-- The theorem stating the number of sunflower plants -/
theorem sunflower_plants_count : 
  ∃ (sunflower_plants : ℕ), 
    sunflower_plants = sunflower_rows * max_plants_per_row ∧ 
    sunflower_plants = 81 :=
by sorry

end NUMINAMATH_CALUDE_sunflower_plants_count_l2689_268986


namespace NUMINAMATH_CALUDE_angle_properties_l2689_268948

/-- Given an angle α whose terminal side passes through the point (sin(5π/6), cos(5π/6)),
    prove that α is in the fourth quadrant and the smallest positive angle with the same
    terminal side as α is 5π/3 -/
theorem angle_properties (α : Real) :
  (∃ r : Real, r > 0 ∧ r * Real.cos α = Real.cos (5 * Real.pi / 6) ∧
                    r * Real.sin α = Real.sin (5 * Real.pi / 6)) →
  (Real.cos α > 0 ∧ Real.sin α < 0) ∧
  (∀ β : Real, β > 0 ∧ Real.cos β = Real.cos α ∧ Real.sin β = Real.sin α → β ≥ 5 * Real.pi / 3) ∧
  (Real.cos (5 * Real.pi / 3) = Real.cos α ∧ Real.sin (5 * Real.pi / 3) = Real.sin α) :=
by sorry

end NUMINAMATH_CALUDE_angle_properties_l2689_268948


namespace NUMINAMATH_CALUDE_rectangle_properties_l2689_268944

/-- A rectangle with one side of length 8 and another of length x -/
structure Rectangle where
  x : ℝ
  h_positive : x > 0

/-- The perimeter of the rectangle -/
def perimeter (rect : Rectangle) : ℝ := 2 * (8 + rect.x)

/-- The area of the rectangle -/
def area (rect : Rectangle) : ℝ := 8 * rect.x

theorem rectangle_properties (rect : Rectangle) :
  (perimeter rect = 16 + 2 * rect.x) ∧
  (area rect = 8 * rect.x) ∧
  (area rect = 80 → perimeter rect = 36) := by
  sorry


end NUMINAMATH_CALUDE_rectangle_properties_l2689_268944


namespace NUMINAMATH_CALUDE_cubic_minus_linear_factorization_l2689_268918

theorem cubic_minus_linear_factorization (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) := by
  sorry

end NUMINAMATH_CALUDE_cubic_minus_linear_factorization_l2689_268918


namespace NUMINAMATH_CALUDE_exams_left_to_grade_l2689_268956

theorem exams_left_to_grade (total_exams : ℕ) (monday_percent : ℚ) (tuesday_percent : ℚ)
  (h1 : total_exams = 120)
  (h2 : monday_percent = 60 / 100)
  (h3 : tuesday_percent = 75 / 100) :
  total_exams - (monday_percent * total_exams).floor - (tuesday_percent * (total_exams - (monday_percent * total_exams).floor)).floor = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_exams_left_to_grade_l2689_268956


namespace NUMINAMATH_CALUDE_C_always_answers_yes_l2689_268966

-- Define the type of islander
inductive IslanderType
  | Knight
  | Liar

-- Define the islanders
def A : IslanderType := sorry
def B : IslanderType := sorry
def C : IslanderType := sorry

-- Define A's statement
def A_statement : Prop := (B = C)

-- Define the question asked to C
def question_to_C : Prop := (A = B)

-- Define C's answer
def C_answer : Prop := 
  match C with
  | IslanderType.Knight => question_to_C
  | IslanderType.Liar => ¬question_to_C

-- Theorem: C will always answer "Yes"
theorem C_always_answers_yes :
  ∀ (A B C : IslanderType),
  (A_statement ↔ (B = C)) →
  C_answer = true :=
sorry

end NUMINAMATH_CALUDE_C_always_answers_yes_l2689_268966


namespace NUMINAMATH_CALUDE_survey_results_l2689_268954

theorem survey_results (total : ℕ) (believe_percent : ℚ) (not_believe_percent : ℚ) 
  (h_total : total = 1240)
  (h_believe : believe_percent = 46/100)
  (h_not_believe : not_believe_percent = 31/100)
  (h_rounding : ∀ x : ℚ, 0 ≤ x → x < 1 → ⌊x * total⌋ + 1 = ⌈x * total⌉) :
  let min_believers := ⌈(believe_percent - 1/200) * total⌉
  let min_non_believers := ⌈(not_believe_percent - 1/200) * total⌉
  let max_refusals := total - min_believers - min_non_believers
  min_believers = 565 ∧ max_refusals = 296 := by
  sorry

#check survey_results

end NUMINAMATH_CALUDE_survey_results_l2689_268954


namespace NUMINAMATH_CALUDE_alice_acorn_price_l2689_268958

/-- The price Alice paid for each acorn -/
def alice_price_per_acorn (alice_acorns : ℕ) (alice_bob_price_ratio : ℕ) (bob_total_price : ℕ) : ℚ :=
  (alice_bob_price_ratio * bob_total_price : ℚ) / alice_acorns

/-- Proof that Alice paid $15 for each acorn -/
theorem alice_acorn_price :
  alice_price_per_acorn 3600 9 6000 = 15 := by
  sorry

#eval alice_price_per_acorn 3600 9 6000

end NUMINAMATH_CALUDE_alice_acorn_price_l2689_268958


namespace NUMINAMATH_CALUDE_infinite_perfect_square_phi_and_d_l2689_268975

/-- Euler's totient function -/
def phi (n : ℕ+) : ℕ := sorry

/-- Number of positive divisors function -/
def d (n : ℕ+) : ℕ := sorry

/-- A natural number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

/-- The set of positive integers n for which both φ(n) and d(n) are perfect squares -/
def S : Set ℕ+ := {n : ℕ+ | is_perfect_square (phi n) ∧ is_perfect_square (d n)}

theorem infinite_perfect_square_phi_and_d : Set.Infinite S := by sorry

end NUMINAMATH_CALUDE_infinite_perfect_square_phi_and_d_l2689_268975


namespace NUMINAMATH_CALUDE_book_ratio_l2689_268916

def book_tournament (candice amanda kara patricia taylor : ℕ) : Prop :=
  candice = 3 * amanda ∧
  kara = amanda / 2 ∧
  patricia = 7 * kara ∧
  taylor = (candice + amanda + kara + patricia) / 4 ∧
  candice = 18

theorem book_ratio (candice amanda kara patricia taylor : ℕ) :
  book_tournament candice amanda kara patricia taylor →
  taylor * 5 = candice + amanda + kara + patricia + taylor :=
by sorry

end NUMINAMATH_CALUDE_book_ratio_l2689_268916


namespace NUMINAMATH_CALUDE_number_with_special_average_l2689_268900

theorem number_with_special_average (x : ℝ) (h1 : x ≠ 0) :
  (x + x^2) / 2 = 5 * x → x = 9 := by
sorry

end NUMINAMATH_CALUDE_number_with_special_average_l2689_268900


namespace NUMINAMATH_CALUDE_number_reduced_then_increased_l2689_268977

theorem number_reduced_then_increased : ∃ x : ℝ, (20 * (x / 5) = 40) ∧ (x = 10) := by
  sorry

end NUMINAMATH_CALUDE_number_reduced_then_increased_l2689_268977


namespace NUMINAMATH_CALUDE_field_B_most_stable_l2689_268927

-- Define the variances for each field
def variance_A : ℝ := 3.6
def variance_B : ℝ := 2.89
def variance_C : ℝ := 13.4
def variance_D : ℝ := 20.14

-- Define a function to compare two variances
def is_more_stable (v1 v2 : ℝ) : Prop := v1 < v2

-- Theorem stating that Field B has the lowest variance
theorem field_B_most_stable :
  is_more_stable variance_B variance_A ∧
  is_more_stable variance_B variance_C ∧
  is_more_stable variance_B variance_D :=
by sorry

end NUMINAMATH_CALUDE_field_B_most_stable_l2689_268927


namespace NUMINAMATH_CALUDE_zongzi_problem_l2689_268934

-- Define the types of zongzi gift boxes
inductive ZongziType
| RedDate
| EggYolk

-- Define the price and quantity of a zongzi gift box
structure ZongziBox where
  type : ZongziType
  price : ℕ
  quantity : ℕ

-- Define the problem parameters
def total_boxes : ℕ := 8
def max_cost : ℕ := 300
def total_recipients : ℕ := 65

-- Define the conditions of the problem
axiom price_relation : ∀ (rd : ZongziBox) (ey : ZongziBox),
  rd.type = ZongziType.RedDate → ey.type = ZongziType.EggYolk →
  3 * rd.price = 4 * ey.price

axiom combined_cost : ∀ (rd : ZongziBox) (ey : ZongziBox),
  rd.type = ZongziType.RedDate → ey.type = ZongziType.EggYolk →
  rd.price + 2 * ey.price = 100

axiom red_date_quantity : ∀ (rd : ZongziBox),
  rd.type = ZongziType.RedDate → rd.quantity = 10

axiom egg_yolk_quantity : ∀ (ey : ZongziBox),
  ey.type = ZongziType.EggYolk → ey.quantity = 6

-- Define the theorem to be proved
theorem zongzi_problem :
  ∃ (rd : ZongziBox) (ey : ZongziBox) (rd_count ey_count : ℕ),
    rd.type = ZongziType.RedDate ∧
    ey.type = ZongziType.EggYolk ∧
    rd.price = 40 ∧
    ey.price = 30 ∧
    rd_count = 5 ∧
    ey_count = 3 ∧
    rd_count + ey_count = total_boxes ∧
    rd_count * rd.price + ey_count * ey.price < max_cost ∧
    rd_count * rd.quantity + ey_count * ey.quantity ≥ total_recipients :=
  sorry


end NUMINAMATH_CALUDE_zongzi_problem_l2689_268934


namespace NUMINAMATH_CALUDE_vanaspati_percentage_l2689_268942

/-- Proves that the percentage of vanaspati in the original ghee mixture is 40% -/
theorem vanaspati_percentage
  (original_quantity : ℝ)
  (pure_ghee_percentage : ℝ)
  (added_pure_ghee : ℝ)
  (new_vanaspati_percentage : ℝ)
  (h1 : original_quantity = 10)
  (h2 : pure_ghee_percentage = 0.6)
  (h3 : added_pure_ghee = 10)
  (h4 : new_vanaspati_percentage = 0.2)
  (h5 : (1 - pure_ghee_percentage) * original_quantity = 
        new_vanaspati_percentage * (original_quantity + added_pure_ghee)) :
  (1 - pure_ghee_percentage) * 100 = 40 := by
sorry

end NUMINAMATH_CALUDE_vanaspati_percentage_l2689_268942


namespace NUMINAMATH_CALUDE_combined_train_length_l2689_268959

/-- Calculates the combined length of two trains given their speeds and passing times. -/
theorem combined_train_length
  (speed_A speed_B speed_bike : ℝ)
  (time_A time_B : ℝ)
  (h1 : speed_A = 120)
  (h2 : speed_B = 100)
  (h3 : speed_bike = 64)
  (h4 : time_A = 75)
  (h5 : time_B = 90)
  (h6 : speed_A > speed_bike)
  (h7 : speed_B > speed_bike)
  (h8 : (speed_A - speed_bike) * time_A / 3600 + (speed_B - speed_bike) * time_B / 3600 = 2.067) :
  (speed_A - speed_bike) * time_A * 1000 / 3600 + (speed_B - speed_bike) * time_B * 1000 / 3600 = 2067 := by
  sorry

#check combined_train_length

end NUMINAMATH_CALUDE_combined_train_length_l2689_268959


namespace NUMINAMATH_CALUDE_complex_product_pure_imaginary_l2689_268978

-- Define the imaginary unit
noncomputable def i : ℂ := Complex.I

-- Define the property of being a pure imaginary number
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- Theorem statement
theorem complex_product_pure_imaginary (a : ℝ) :
  is_pure_imaginary ((1 + i) * (1 + a * i)) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_pure_imaginary_l2689_268978


namespace NUMINAMATH_CALUDE_k_range_given_one_integer_solution_l2689_268957

/-- The inequality system has only one integer solution -/
def has_one_integer_solution (k : ℝ) : Prop :=
  ∃! (x : ℤ), (x^2 - 2*x - 8 > 0) ∧ (2*x^2 + (2*k+7)*x + 7*k < 0)

/-- The range of k -/
def k_range (k : ℝ) : Prop :=
  (k ≥ -5 ∧ k < 3) ∨ (k > 4 ∧ k ≤ 5)

/-- Theorem stating the range of k given the conditions -/
theorem k_range_given_one_integer_solution :
  ∀ k : ℝ, has_one_integer_solution k ↔ k_range k :=
sorry

end NUMINAMATH_CALUDE_k_range_given_one_integer_solution_l2689_268957


namespace NUMINAMATH_CALUDE_appended_digit_problem_l2689_268919

theorem appended_digit_problem (x y : ℕ) : 
  x > 0 → y < 10 → (10 * x + y) - x^2 = 8 * x → 
  ((x = 2 ∧ y = 0) ∨ (x = 3 ∧ y = 3) ∨ (x = 4 ∧ y = 8)) := by sorry

end NUMINAMATH_CALUDE_appended_digit_problem_l2689_268919


namespace NUMINAMATH_CALUDE_max_value_product_l2689_268928

theorem max_value_product (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 8) :
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = 8 → (1 + a) * (1 + b) ≤ (1 + x) * (1 + y)) ∧
  (1 + x) * (1 + y) = 25 :=
sorry

end NUMINAMATH_CALUDE_max_value_product_l2689_268928


namespace NUMINAMATH_CALUDE_set_cardinality_lower_bound_l2689_268911

variable (m : ℕ) (A : Finset ℤ) (B : Fin m → Finset ℤ)

theorem set_cardinality_lower_bound
  (h_m : m ≥ 2)
  (h_subset : ∀ i : Fin m, B i ⊆ A)
  (h_sum : ∀ i : Fin m, (B i).sum id = m ^ (i : ℕ).succ) :
  A.card ≥ m / 2 :=
sorry

end NUMINAMATH_CALUDE_set_cardinality_lower_bound_l2689_268911


namespace NUMINAMATH_CALUDE_max_receptivity_and_duration_receptivity_comparison_insufficient_duration_l2689_268974

-- Define the piecewise function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 10 then -0.1 * x^2 + 2.6 * x + 44
  else if 10 < x ∧ x ≤ 15 then 60
  else if 15 < x ∧ x ≤ 25 then -3 * x + 105
  else if 25 < x ∧ x ≤ 40 then 30
  else 0

-- Theorem 1: Maximum receptivity and duration
theorem max_receptivity_and_duration :
  (∀ x, 0 < x → x ≤ 40 → f x ≤ 60) ∧
  (∀ x, 10 ≤ x → x ≤ 15 → f x = 60) :=
sorry

-- Theorem 2: Receptivity comparison
theorem receptivity_comparison :
  f 5 > f 20 ∧ f 20 > f 35 :=
sorry

-- Theorem 3: Insufficient duration for required receptivity
theorem insufficient_duration :
  ¬ ∃ a : ℝ, 0 < a ∧ a + 12 ≤ 40 ∧ ∀ x, a ≤ x → x ≤ a + 12 → f x ≥ 56 :=
sorry

end NUMINAMATH_CALUDE_max_receptivity_and_duration_receptivity_comparison_insufficient_duration_l2689_268974


namespace NUMINAMATH_CALUDE_sum_divides_8n_count_l2689_268923

theorem sum_divides_8n_count : 
  (∃ (S : Finset ℕ), S.card = 4 ∧ 
    (∀ n : ℕ, n > 0 → (n ∈ S ↔ (8 * n) % ((n * (n + 1)) / 2) = 0))) := by
  sorry

end NUMINAMATH_CALUDE_sum_divides_8n_count_l2689_268923


namespace NUMINAMATH_CALUDE_item_value_proof_l2689_268992

def import_tax_rate : ℝ := 0.07
def tax_threshold : ℝ := 1000
def tax_paid : ℝ := 87.50

theorem item_value_proof (total_value : ℝ) : 
  total_value = 2250 := by
  sorry

end NUMINAMATH_CALUDE_item_value_proof_l2689_268992


namespace NUMINAMATH_CALUDE_arcsin_arccos_equation_solution_l2689_268903

theorem arcsin_arccos_equation_solution :
  ∃ x : ℝ, x = -1 / Real.sqrt 7 ∧ Real.arcsin (3 * x) - Real.arccos (2 * x) = π / 6 :=
by sorry

end NUMINAMATH_CALUDE_arcsin_arccos_equation_solution_l2689_268903


namespace NUMINAMATH_CALUDE_ln_cube_relation_l2689_268902

theorem ln_cube_relation (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (Real.log x > Real.log y → x^3 > y^3) ∧
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a^3 > b^3 ∧ ¬(Real.log a > Real.log b) :=
sorry

end NUMINAMATH_CALUDE_ln_cube_relation_l2689_268902


namespace NUMINAMATH_CALUDE_inequality_proof_l2689_268981

theorem inequality_proof :
  (∀ x y : ℝ, x^2 + y^2 + 1 > x * (y + 1)) ∧
  (∀ k : ℝ, (∀ x y : ℝ, x^2 + y^2 + 1 ≥ k * x * (y + 1)) → k ≤ Real.sqrt 2) ∧
  (∀ k : ℝ, (∀ m n : ℤ, (m : ℝ)^2 + (n : ℝ)^2 + 1 ≥ k * (m : ℝ) * ((n : ℝ) + 1)) → k ≤ 3/2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2689_268981


namespace NUMINAMATH_CALUDE_cubic_factorization_l2689_268941

theorem cubic_factorization (a : ℝ) : a^3 - 4*a^2 + 4*a = a*(a-2)^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l2689_268941


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2689_268962

/-- Given two parallel vectors a = (2, 5) and b = (x, -2), prove that x = -4/5 -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : Fin 2 → ℝ := ![2, 5]
  let b : Fin 2 → ℝ := ![x, -2]
  (∃ (k : ℝ), k ≠ 0 ∧ (∀ i, b i = k * a i)) →
  x = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2689_268962


namespace NUMINAMATH_CALUDE_unique_c_value_l2689_268991

theorem unique_c_value (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_order : a < b ∧ b < c) (h_sum : a + b + c = 11)
  (h_frac : (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c = 1) : c = 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_c_value_l2689_268991


namespace NUMINAMATH_CALUDE_vector_c_coordinates_l2689_268970

def a : Fin 3 → ℝ := ![0, 1, -1]
def b : Fin 3 → ℝ := ![1, 2, 3]
def c : Fin 3 → ℝ := λ i => 3 * a i - b i

theorem vector_c_coordinates :
  c = ![-1, 1, -6] := by sorry

end NUMINAMATH_CALUDE_vector_c_coordinates_l2689_268970


namespace NUMINAMATH_CALUDE_min_value_a_plus_9b_l2689_268980

theorem min_value_a_plus_9b (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_arith_seq : (1/a + 1/b) / 2 = 1/2) : 
  ∀ x y : ℝ, x > 0 → y > 0 → (1/x + 1/y) / 2 = 1/2 → x + 9*y ≥ 16 ∧ 
  ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ (1/x₀ + 1/y₀) / 2 = 1/2 ∧ x₀ + 9*y₀ = 16 :=
by sorry

#check min_value_a_plus_9b

end NUMINAMATH_CALUDE_min_value_a_plus_9b_l2689_268980


namespace NUMINAMATH_CALUDE_downstream_speed_l2689_268955

/-- Represents the speed of a rower in different conditions -/
structure RowerSpeed where
  upstream : ℝ
  still_water : ℝ
  downstream : ℝ

/-- 
Given a rower's speed upstream and in still water, 
calculates and proves the rower's speed downstream
-/
theorem downstream_speed (r : RowerSpeed) 
  (h_upstream : r.upstream = 35)
  (h_still : r.still_water = 40) :
  r.downstream = 45 := by
  sorry

#check downstream_speed

end NUMINAMATH_CALUDE_downstream_speed_l2689_268955


namespace NUMINAMATH_CALUDE_beatrice_gilbert_ratio_l2689_268920

/-- The number of crayons in each person's box -/
structure CrayonBoxes where
  karen : ℕ
  beatrice : ℕ
  gilbert : ℕ
  judah : ℕ

/-- The conditions of the problem -/
def problem_conditions (boxes : CrayonBoxes) : Prop :=
  boxes.karen = 2 * boxes.beatrice ∧
  boxes.beatrice = boxes.gilbert ∧
  boxes.gilbert = 4 * boxes.judah ∧
  boxes.karen = 128 ∧
  boxes.judah = 8

/-- The theorem stating that Beatrice and Gilbert have the same number of crayons -/
theorem beatrice_gilbert_ratio (boxes : CrayonBoxes) 
  (h : problem_conditions boxes) : boxes.beatrice = boxes.gilbert := by
  sorry

#check beatrice_gilbert_ratio

end NUMINAMATH_CALUDE_beatrice_gilbert_ratio_l2689_268920


namespace NUMINAMATH_CALUDE_circle_condition_l2689_268961

theorem circle_condition (m : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 - x + y + m = 0 ∧ 
   ∃ (h k r : ℝ), r > 0 ∧ ∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = r^2 ↔ x^2 + y^2 - x + y + m = 0) 
  → m < (1/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_circle_condition_l2689_268961


namespace NUMINAMATH_CALUDE_largest_x_sqrt_3x_eq_5x_l2689_268993

theorem largest_x_sqrt_3x_eq_5x : 
  (∃ (x : ℝ), x > 0 ∧ Real.sqrt (3 * x) = 5 * x) → 
  (∀ (y : ℝ), y > 0 ∧ Real.sqrt (3 * y) = 5 * y → y ≤ 3/25) ∧
  (Real.sqrt (3 * (3/25)) = 5 * (3/25)) := by
sorry

end NUMINAMATH_CALUDE_largest_x_sqrt_3x_eq_5x_l2689_268993


namespace NUMINAMATH_CALUDE_expression_evaluation_l2689_268982

theorem expression_evaluation : (3^2 - 3 + 1) - (4^2 - 4 + 1) + (5^2 - 5 + 1) - (6^2 - 6 + 1) = -16 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2689_268982


namespace NUMINAMATH_CALUDE_black_burger_cost_l2689_268994

theorem black_burger_cost (salmon_cost chicken_cost total_bill : ℝ) 
  (h1 : salmon_cost = 40)
  (h2 : chicken_cost = 25)
  (h3 : total_bill = 92) : 
  ∃ (burger_cost : ℝ), 
    burger_cost = 15 ∧ 
    total_bill = (salmon_cost + burger_cost + chicken_cost) * 1.15 := by
  sorry

end NUMINAMATH_CALUDE_black_burger_cost_l2689_268994


namespace NUMINAMATH_CALUDE_f_strictly_decreasing_on_interval_l2689_268906

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 1

-- Theorem statement
theorem f_strictly_decreasing_on_interval :
  ∀ x y, -1 < x ∧ x < y ∧ y < 3 → f x > f y := by
  sorry

end NUMINAMATH_CALUDE_f_strictly_decreasing_on_interval_l2689_268906


namespace NUMINAMATH_CALUDE_pascal_triangle_51_row_5th_number_l2689_268953

theorem pascal_triangle_51_row_5th_number : 
  let n : ℕ := 51  -- number of elements in the row
  let k : ℕ := 4   -- index of the number we're looking for (0-based)
  Nat.choose (n - 1) k = 220500 := by
sorry

end NUMINAMATH_CALUDE_pascal_triangle_51_row_5th_number_l2689_268953


namespace NUMINAMATH_CALUDE_circular_path_diameter_increase_l2689_268901

theorem circular_path_diameter_increase 
  (original_rounds : ℕ) 
  (original_time : ℝ) 
  (new_time : ℝ) 
  (original_rounds_pos : original_rounds > 0)
  (original_time_pos : original_time > 0)
  (new_time_pos : new_time > 0)
  (h_original : original_rounds = 8)
  (h_original_time : original_time = 40)
  (h_new_time : new_time = 50) :
  let original_single_round_time := original_time / original_rounds
  let diameter_increase_factor := new_time / original_single_round_time
  diameter_increase_factor = 10 := by
  sorry

end NUMINAMATH_CALUDE_circular_path_diameter_increase_l2689_268901


namespace NUMINAMATH_CALUDE_complex_number_equation_l2689_268935

theorem complex_number_equation (z : ℂ) : (z * Complex.I = Complex.I + z) → z = (1/2 : ℂ) - (1/2 : ℂ) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equation_l2689_268935


namespace NUMINAMATH_CALUDE_sqrt_product_sqrt_main_theorem_l2689_268909

theorem sqrt_product_sqrt (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  Real.sqrt (a * Real.sqrt b) = Real.sqrt a * Real.sqrt (Real.sqrt b) :=
by sorry

theorem main_theorem : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_product_sqrt_main_theorem_l2689_268909


namespace NUMINAMATH_CALUDE_late_time_calculation_l2689_268922

/-- Calculates the total late time for five students given the lateness of one student and the additional lateness of the other four. -/
def totalLateTime (firstStudentLateness : ℕ) (additionalLateness : ℕ) : ℕ :=
  firstStudentLateness + 4 * (firstStudentLateness + additionalLateness)

/-- Theorem stating that for the given scenario, the total late time is 140 minutes. -/
theorem late_time_calculation :
  totalLateTime 20 10 = 140 := by
  sorry

end NUMINAMATH_CALUDE_late_time_calculation_l2689_268922


namespace NUMINAMATH_CALUDE_three_disjoint_edges_exist_l2689_268924

/-- A graph with 6 vertices where each vertex has degree at least 3 -/
structure SixVertexGraph where
  vertices : Finset (Fin 6)
  edges : Finset (Fin 6 × Fin 6)
  vertex_count : vertices.card = 6
  edge_symmetry : ∀ (u v : Fin 6), (u, v) ∈ edges → (v, u) ∈ edges
  min_degree : ∀ v ∈ vertices, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card ≥ 3

/-- A set of 3 disjoint edges that cover all vertices -/
def ThreeDisjointEdges (G : SixVertexGraph) : Prop :=
  ∃ (e₁ e₂ e₃ : Fin 6 × Fin 6),
    e₁ ∈ G.edges ∧ e₂ ∈ G.edges ∧ e₃ ∈ G.edges ∧
    e₁.1 ≠ e₁.2 ∧ e₂.1 ≠ e₂.2 ∧ e₃.1 ≠ e₃.2 ∧
    e₁.1 ≠ e₂.1 ∧ e₁.1 ≠ e₂.2 ∧ e₁.1 ≠ e₃.1 ∧ e₁.1 ≠ e₃.2 ∧
    e₁.2 ≠ e₂.1 ∧ e₁.2 ≠ e₂.2 ∧ e₁.2 ≠ e₃.1 ∧ e₁.2 ≠ e₃.2 ∧
    e₂.1 ≠ e₃.1 ∧ e₂.1 ≠ e₃.2 ∧ e₂.2 ≠ e₃.1 ∧ e₂.2 ≠ e₃.2

/-- Theorem: In a graph with 6 vertices where each vertex has degree at least 3,
    there exists a set of 3 disjoint edges that cover all vertices -/
theorem three_disjoint_edges_exist (G : SixVertexGraph) : ThreeDisjointEdges G :=
sorry

end NUMINAMATH_CALUDE_three_disjoint_edges_exist_l2689_268924


namespace NUMINAMATH_CALUDE_eliot_account_balance_l2689_268979

theorem eliot_account_balance (al eliot : ℝ) 
  (h1 : al > eliot)
  (h2 : al - eliot = (1 / 12) * (al + eliot))
  (h3 : 1.1 * al = 1.2 * eliot + 21) :
  eliot = 210 := by
sorry

end NUMINAMATH_CALUDE_eliot_account_balance_l2689_268979


namespace NUMINAMATH_CALUDE_probability_diamond_or_ace_in_two_draws_l2689_268931

/-- The probability of at least one of two cards being a diamond or an ace
    when drawn with replacement from a modified deck. -/
theorem probability_diamond_or_ace_in_two_draws :
  let total_cards : ℕ := 54
  let diamond_cards : ℕ := 13
  let ace_cards : ℕ := 4
  let diamond_or_ace_cards : ℕ := diamond_cards + ace_cards
  let prob_not_diamond_or_ace : ℚ := (total_cards - diamond_or_ace_cards) / total_cards
  let prob_at_least_one_diamond_or_ace : ℚ := 1 - prob_not_diamond_or_ace ^ 2
  prob_at_least_one_diamond_or_ace = 368 / 729 :=
by sorry

end NUMINAMATH_CALUDE_probability_diamond_or_ace_in_two_draws_l2689_268931


namespace NUMINAMATH_CALUDE_cos_right_angle_l2689_268976

theorem cos_right_angle (D E F : ℝ) (h1 : D = 90) (h2 : E = 9) (h3 : F = 40) : Real.cos D = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_right_angle_l2689_268976


namespace NUMINAMATH_CALUDE_curve_has_axis_of_symmetry_l2689_268951

/-- The equation of the curve -/
def curve_equation (x y : ℝ) : Prop :=
  x^2 - x*y + y^2 + x - y - 1 = 0

/-- The proposed axis of symmetry -/
def axis_of_symmetry (x y : ℝ) : Prop :=
  x + y = 0

/-- Theorem stating that the curve has the given axis of symmetry -/
theorem curve_has_axis_of_symmetry :
  ∀ (x y : ℝ), curve_equation x y ↔ curve_equation (-y) (-x) :=
sorry

end NUMINAMATH_CALUDE_curve_has_axis_of_symmetry_l2689_268951


namespace NUMINAMATH_CALUDE_stream_speed_l2689_268936

/-- Prove that the speed of a stream is 3.75 km/h given the boat's travel times and distances -/
theorem stream_speed (downstream_distance : ℝ) (downstream_time : ℝ) 
  (upstream_distance : ℝ) (upstream_time : ℝ) 
  (h1 : downstream_distance = 100)
  (h2 : downstream_time = 8)
  (h3 : upstream_distance = 75)
  (h4 : upstream_time = 15) :
  ∃ (boat_speed stream_speed : ℝ),
    downstream_distance = (boat_speed + stream_speed) * downstream_time ∧
    upstream_distance = (boat_speed - stream_speed) * upstream_time ∧
    stream_speed = 3.75 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l2689_268936


namespace NUMINAMATH_CALUDE_square_diagonal_length_l2689_268983

/-- The length of the diagonal of a square with side length 50√2 cm is 100 cm. -/
theorem square_diagonal_length :
  let side_length : ℝ := 50 * Real.sqrt 2
  let diagonal_length : ℝ := 100
  diagonal_length = Real.sqrt (2 * side_length ^ 2) :=
by sorry

end NUMINAMATH_CALUDE_square_diagonal_length_l2689_268983


namespace NUMINAMATH_CALUDE_chord_max_surface_area_l2689_268905

/-- 
Given a circle with radius R, the chord of length R√2 maximizes the surface area 
of the cylindrical shell formed when rotating the chord around the diameter parallel to it.
-/
theorem chord_max_surface_area (R : ℝ) (R_pos : R > 0) : 
  let chord_length (x : ℝ) := 2 * x
  let surface_area (x : ℝ) := 4 * Real.pi * x * Real.sqrt (R^2 - x^2)
  ∃ (x : ℝ), x > 0 ∧ x < R ∧ 
    chord_length x = R * Real.sqrt 2 ∧
    ∀ (y : ℝ), y > 0 → y < R → surface_area x ≥ surface_area y :=
by sorry

end NUMINAMATH_CALUDE_chord_max_surface_area_l2689_268905


namespace NUMINAMATH_CALUDE_potato_bag_weight_l2689_268930

theorem potato_bag_weight (weight : ℝ) (fraction : ℝ) : 
  weight = 36 → weight / fraction = 36 → fraction = 1 := by sorry

end NUMINAMATH_CALUDE_potato_bag_weight_l2689_268930


namespace NUMINAMATH_CALUDE_max_label_in_sample_l2689_268963

/-- Systematic sampling function that returns the maximum label in the sample -/
def systematic_sample_max (total : ℕ) (sample_size : ℕ) (first_item : ℕ) : ℕ :=
  let interval := total / sample_size
  let position := (first_item % interval) + 1
  (sample_size - (sample_size - position)) * interval + first_item

/-- Theorem stating the maximum label in the systematic sample -/
theorem max_label_in_sample :
  systematic_sample_max 80 5 10 = 74 := by
  sorry

#eval systematic_sample_max 80 5 10

end NUMINAMATH_CALUDE_max_label_in_sample_l2689_268963


namespace NUMINAMATH_CALUDE_sqrt_sum_squares_equals_sum_l2689_268997

theorem sqrt_sum_squares_equals_sum (a b c : ℝ) :
  Real.sqrt (a^2 + b^2 + c^2) = a + b + c ↔ a * b + a * c + b * c = 0 ∧ a + b + c ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_squares_equals_sum_l2689_268997


namespace NUMINAMATH_CALUDE_jolene_raised_180_l2689_268910

/-- Represents Jolene's fundraising activities --/
structure JoleneFundraising where
  num_babysitting_families : ℕ
  babysitting_rate : ℕ
  num_cars_washed : ℕ
  car_wash_rate : ℕ

/-- Calculates the total amount Jolene raised --/
def total_raised (j : JoleneFundraising) : ℕ :=
  j.num_babysitting_families * j.babysitting_rate + j.num_cars_washed * j.car_wash_rate

/-- Theorem stating that Jolene raised $180 --/
theorem jolene_raised_180 :
  ∃ j : JoleneFundraising,
    j.num_babysitting_families = 4 ∧
    j.babysitting_rate = 30 ∧
    j.num_cars_washed = 5 ∧
    j.car_wash_rate = 12 ∧
    total_raised j = 180 :=
  sorry

end NUMINAMATH_CALUDE_jolene_raised_180_l2689_268910


namespace NUMINAMATH_CALUDE_smallest_number_quotient_remainder_difference_l2689_268912

theorem smallest_number_quotient_remainder_difference : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (n % 5 = 0) ∧
  (n / 5 > n % 34) ∧
  (∀ m : ℕ, m > 0 → m % 5 = 0 → m / 5 > m % 34 → m ≥ n) ∧
  (n / 5 - n % 34 = 8) := by
sorry

end NUMINAMATH_CALUDE_smallest_number_quotient_remainder_difference_l2689_268912


namespace NUMINAMATH_CALUDE_angle_ADE_measure_l2689_268921

/-- Triangle ABC -/
structure Triangle :=
  (A B C : ℝ)
  (sum_angles : A + B + C = 180)

/-- Pentagon ABCDE -/
structure Pentagon :=
  (A B C D E : ℝ)
  (sum_angles : A + B + C + D + E = 540)

/-- Circle circumscribed around a pentagon -/
structure CircumscribedCircle (p : Pentagon) := 
  (is_circumscribed : Bool)

/-- Pentagon with sides tangent to a circle -/
structure TangentPentagon (p : Pentagon) (c : CircumscribedCircle p) :=
  (is_tangent : Bool)

/-- Theorem: In a pentagon ABCDE constructed as described, the measure of angle ADE is 108° -/
theorem angle_ADE_measure 
  (t : Triangle)
  (p : Pentagon)
  (c : CircumscribedCircle p)
  (tp : TangentPentagon p c)
  (h1 : t.A = 60)
  (h2 : t.B = 50)
  (h3 : t.C = 70)
  (h4 : p.D ∈ Set.Ioo 0 (t.A + t.B))  -- D is on side AB
  (h5 : p.E ∈ Set.Ioo (t.A + t.B) (t.A + t.B + t.C))  -- E is on side BC
  : p.D = 108 :=
sorry

end NUMINAMATH_CALUDE_angle_ADE_measure_l2689_268921


namespace NUMINAMATH_CALUDE_exponent_calculation_l2689_268907

theorem exponent_calculation : (8^5 / 8^2) * 4^4 = 2^17 := by
  sorry

end NUMINAMATH_CALUDE_exponent_calculation_l2689_268907


namespace NUMINAMATH_CALUDE_ryan_marble_distribution_l2689_268985

theorem ryan_marble_distribution (total_marbles : ℕ) (marbles_per_friend : ℕ) (num_friends : ℕ) :
  total_marbles = 72 →
  marbles_per_friend = 8 →
  total_marbles = marbles_per_friend * num_friends →
  num_friends = 9 := by
sorry

end NUMINAMATH_CALUDE_ryan_marble_distribution_l2689_268985


namespace NUMINAMATH_CALUDE_parabola_directrix_l2689_268990

-- Define a parabola with equation y^2 = 8x
def parabola (x y : ℝ) : Prop := y^2 = 8 * x

-- Define the directrix of a parabola
def directrix (x : ℝ) : Prop := x = -2

-- Theorem statement
theorem parabola_directrix :
  ∀ (x y : ℝ), parabola x y → directrix x :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l2689_268990


namespace NUMINAMATH_CALUDE_marathon_day3_miles_l2689_268973

/-- Represents the marathon runner's training schedule over 3 days -/
structure MarathonTraining where
  total_miles : ℝ
  day1_percent : ℝ
  day2_percent : ℝ

/-- Calculates the miles run on day 3 given the training schedule -/
def miles_on_day3 (mt : MarathonTraining) : ℝ :=
  mt.total_miles - (mt.total_miles * mt.day1_percent) - ((mt.total_miles - (mt.total_miles * mt.day1_percent)) * mt.day2_percent)

/-- Theorem stating that given the specific training schedule, the miles run on day 3 is 28 -/
theorem marathon_day3_miles :
  let mt : MarathonTraining := ⟨70, 0.2, 0.5⟩
  miles_on_day3 mt = 28 := by
  sorry

end NUMINAMATH_CALUDE_marathon_day3_miles_l2689_268973


namespace NUMINAMATH_CALUDE_babysitting_earnings_proof_l2689_268949

/-- Calculates earnings from babysitting given net profit, lemonade stand revenue, and operating cost -/
def earnings_from_babysitting (net_profit : ℕ) (lemonade_revenue : ℕ) (operating_cost : ℕ) : ℕ :=
  (net_profit + operating_cost) - lemonade_revenue

/-- Proves that earnings from babysitting equal $31 given the specific values -/
theorem babysitting_earnings_proof :
  earnings_from_babysitting 44 47 34 = 31 := by
sorry

end NUMINAMATH_CALUDE_babysitting_earnings_proof_l2689_268949


namespace NUMINAMATH_CALUDE_rectangle_area_problem_l2689_268987

theorem rectangle_area_problem (p q : ℝ) : 
  q = (2/5) * p →  -- point (p, q) is on the line y = 2/5 x
  p * q = 90 →     -- area of the rectangle is 90
  p = 15 :=        -- prove that p = 15
by sorry

end NUMINAMATH_CALUDE_rectangle_area_problem_l2689_268987


namespace NUMINAMATH_CALUDE_square_perimeter_ratio_l2689_268925

theorem square_perimeter_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) (h_area_ratio : a^2 / b^2 = 16 / 25) :
  (4 * a) / (4 * b) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_ratio_l2689_268925
