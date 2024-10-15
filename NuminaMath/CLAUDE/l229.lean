import Mathlib

namespace NUMINAMATH_CALUDE_parabola_max_value_implies_a_greater_than_two_l229_22950

/-- Given a parabola y = (2-a)x^2 + 3x - 2, if it has a maximum value, then a > 2 -/
theorem parabola_max_value_implies_a_greater_than_two (a : ℝ) :
  (∃ (y_max : ℝ), ∀ (x : ℝ), (2 - a) * x^2 + 3 * x - 2 ≤ y_max) →
  a > 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_max_value_implies_a_greater_than_two_l229_22950


namespace NUMINAMATH_CALUDE_min_sum_squares_l229_22924

theorem min_sum_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 8) :
  ∃ (m : ℝ), m = 3 ∧ (∀ a b c : ℝ, a^3 + b^3 + c^3 - 3*a*b*c = 8 → a^2 + b^2 + c^2 ≥ m) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l229_22924


namespace NUMINAMATH_CALUDE_subject_selection_l229_22998

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem subject_selection (n : ℕ) (h : n = 7) :
  -- Total number of ways to choose any three subjects
  choose n 3 = choose 7 3 ∧
  -- If at least one of two specific subjects is chosen
  choose 2 1 * choose 5 2 + choose 2 2 * choose 5 1 = choose 2 1 * choose 5 2 + choose 2 2 * choose 5 1 ∧
  -- If two specific subjects cannot be chosen at the same time
  choose n 3 - choose 2 2 * choose 5 1 = choose 7 3 - choose 2 2 * choose 5 1 ∧
  -- If at least one of two specific subjects is chosen, and two other specific subjects are not chosen at the same time
  (choose 1 1 * choose 4 2 + choose 1 1 * choose 5 2 + choose 2 2 * choose 4 1 = 20) := by
  sorry

end NUMINAMATH_CALUDE_subject_selection_l229_22998


namespace NUMINAMATH_CALUDE_gcd_g_x_l229_22988

def g (x : ℤ) : ℤ := (3*x+5)*(9*x+4)*(11*x+8)*(x+11)

theorem gcd_g_x (x : ℤ) (h : ∃ k : ℤ, x = 34914 * k) : 
  Nat.gcd (Int.natAbs (g x)) (Int.natAbs x) = 1760 := by
  sorry

end NUMINAMATH_CALUDE_gcd_g_x_l229_22988


namespace NUMINAMATH_CALUDE_floor_paving_cost_l229_22953

/-- The cost of paving a rectangular floor -/
theorem floor_paving_cost 
  (length : ℝ) 
  (width : ℝ) 
  (rate : ℝ) 
  (h1 : length = 5.5) 
  (h2 : width = 3.75) 
  (h3 : rate = 1200) : 
  length * width * rate = 24750 := by
  sorry

end NUMINAMATH_CALUDE_floor_paving_cost_l229_22953


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l229_22984

theorem quadratic_equal_roots (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - m * x + 2 * x + 10 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 - m * y + 2 * y + 10 = 0 → y = x) ↔ 
  (m = 2 + 2 * Real.sqrt 30 ∨ m = 2 - 2 * Real.sqrt 30) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l229_22984


namespace NUMINAMATH_CALUDE_quadratic_inequality_quadratic_inequality_negative_m_l229_22912

theorem quadratic_inequality (m : ℝ) :
  (∀ x : ℝ, m * x^2 + (1 - m) * x + m - 2 ≥ -2) ↔ m ≥ 1/3 := by sorry

theorem quadratic_inequality_negative_m (m : ℝ) (hm : m < 0) :
  (∀ x : ℝ, m * x^2 + (1 - m) * x + m - 2 < m - 1) ↔
  ((m ≤ -1 ∧ (x < -1/m ∨ x > 1)) ∨
   (-1 < m ∧ m < 0 ∧ (x < 1 ∨ x > -1/m))) := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_quadratic_inequality_negative_m_l229_22912


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l229_22900

/-- The equation (2kx^2 + 4kx + 2) = 0 has equal roots when k = 1 -/
theorem equal_roots_quadratic (k : ℝ) : 
  (∀ x : ℝ, 2 * k * x^2 + 4 * k * x + 2 = 0) → 
  (∃! r : ℝ, 2 * x^2 + 4 * x + 2 = 0) := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l229_22900


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l229_22927

theorem arithmetic_calculation : 4 * 6 * 8 + 18 / 3 - 2^3 = 190 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l229_22927


namespace NUMINAMATH_CALUDE_coloring_books_distribution_l229_22981

theorem coloring_books_distribution (initial_stock : ℕ) (books_sold : ℕ) (num_shelves : ℕ) 
  (h1 : initial_stock = 27)
  (h2 : books_sold = 6)
  (h3 : num_shelves = 3) :
  (initial_stock - books_sold) / num_shelves = 7 := by
  sorry

end NUMINAMATH_CALUDE_coloring_books_distribution_l229_22981


namespace NUMINAMATH_CALUDE_cube_of_negative_a_b_squared_l229_22979

theorem cube_of_negative_a_b_squared (a b : ℝ) : (-a * b^2)^3 = -a^3 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_negative_a_b_squared_l229_22979


namespace NUMINAMATH_CALUDE_max_value_m_plus_n_l229_22907

theorem max_value_m_plus_n (a b m n : ℝ) : 
  (a < 0 ∧ b < 0) →  -- a and b have the same sign (negative)
  (∀ x, ax^2 + 2*x + b < 0 ↔ x ≠ -1/a) →  -- solution set condition
  m = b + 1/a →  -- definition of m
  n = a + 1/b →  -- definition of n
  (∀ k, m + n ≤ k) → k = -4 :=
by sorry

end NUMINAMATH_CALUDE_max_value_m_plus_n_l229_22907


namespace NUMINAMATH_CALUDE_centipede_human_ratio_theorem_l229_22964

/-- Represents the population of an island with centipedes, humans, and sheep. -/
structure IslandPopulation where
  centipedes : ℕ
  humans : ℕ
  sheep : ℕ

/-- The ratio of centipedes to humans on the island. -/
def centipede_human_ratio (pop : IslandPopulation) : ℚ :=
  pop.centipedes / pop.humans

/-- Theorem stating the ratio of centipedes to humans given the conditions. -/
theorem centipede_human_ratio_theorem (pop : IslandPopulation) 
  (h1 : pop.centipedes = 100)
  (h2 : pop.sheep = pop.humans / 2) :
  centipede_human_ratio pop = 100 / pop.humans := by
  sorry

end NUMINAMATH_CALUDE_centipede_human_ratio_theorem_l229_22964


namespace NUMINAMATH_CALUDE_unique_solution_floor_equation_l229_22937

theorem unique_solution_floor_equation :
  ∀ n : ℤ, (⌊(n^2 : ℚ) / 4⌋ - ⌊(n : ℚ) / 2⌋^2 = 3) ↔ n = 7 := by sorry

end NUMINAMATH_CALUDE_unique_solution_floor_equation_l229_22937


namespace NUMINAMATH_CALUDE_tan_sin_ratio_thirty_degrees_l229_22968

theorem tan_sin_ratio_thirty_degrees :
  let tan_30_sq := sin_30_sq / cos_30_sq
  let sin_30_sq := (1 : ℝ) / 4
  let cos_30_sq := (3 : ℝ) / 4
  (tan_30_sq - sin_30_sq) / (tan_30_sq * sin_30_sq) = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_tan_sin_ratio_thirty_degrees_l229_22968


namespace NUMINAMATH_CALUDE_time_to_empty_tank_l229_22991

/-- Represents the volume of the tank in cubic feet -/
def tank_volume : ℝ := 30

/-- Represents the rate of the inlet pipe in cubic inches per minute -/
def inlet_rate : ℝ := 3

/-- Represents the rate of the first outlet pipe in cubic inches per minute -/
def outlet_rate_1 : ℝ := 12

/-- Represents the rate of the second outlet pipe in cubic inches per minute -/
def outlet_rate_2 : ℝ := 6

/-- Conversion factor from feet to inches -/
def feet_to_inches : ℝ := 12

/-- Theorem stating the time to empty the tank -/
theorem time_to_empty_tank :
  let tank_volume_inches := tank_volume * feet_to_inches * feet_to_inches * feet_to_inches
  let net_emptying_rate := outlet_rate_1 + outlet_rate_2 - inlet_rate
  tank_volume_inches / net_emptying_rate = 3456 := by
  sorry


end NUMINAMATH_CALUDE_time_to_empty_tank_l229_22991


namespace NUMINAMATH_CALUDE_bank_deposit_is_50_l229_22949

def total_income : ℚ := 200

def provident_fund_ratio : ℚ := 1 / 16
def insurance_premium_ratio : ℚ := 1 / 15
def domestic_needs_ratio : ℚ := 5 / 7

def provident_fund : ℚ := provident_fund_ratio * total_income
def remaining_after_provident_fund : ℚ := total_income - provident_fund

def insurance_premium : ℚ := insurance_premium_ratio * remaining_after_provident_fund
def remaining_after_insurance : ℚ := remaining_after_provident_fund - insurance_premium

def domestic_needs : ℚ := domestic_needs_ratio * remaining_after_insurance
def bank_deposit : ℚ := remaining_after_insurance - domestic_needs

theorem bank_deposit_is_50 : bank_deposit = 50 := by
  sorry

end NUMINAMATH_CALUDE_bank_deposit_is_50_l229_22949


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_15_l229_22931

theorem smallest_four_digit_multiple_of_15 : 
  ∀ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 15 = 0 → n ≥ 1005 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_15_l229_22931


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_three_l229_22945

theorem fraction_zero_implies_x_three (x : ℝ) :
  (x - 3) / (2 * x + 5) = 0 ∧ 2 * x + 5 ≠ 0 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_three_l229_22945


namespace NUMINAMATH_CALUDE_y_derivative_l229_22916

noncomputable def y (x : ℝ) : ℝ := 3 * (Real.sin x / Real.cos x ^ 2) + 2 * (Real.sin x / Real.cos x ^ 4)

theorem y_derivative (x : ℝ) :
  deriv y x = (3 + 3 * Real.sin x ^ 2) / Real.cos x ^ 3 + (2 - 6 * Real.sin x ^ 2) / Real.cos x ^ 5 :=
by sorry

end NUMINAMATH_CALUDE_y_derivative_l229_22916


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l229_22954

theorem arithmetic_calculations :
  (4 + (-7) - (-5) = 2) ∧
  (-1^2023 + 27 * (-1/3)^2 - |(-5)| = -3) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l229_22954


namespace NUMINAMATH_CALUDE_equidistant_point_y_coordinate_l229_22922

/-- The y-coordinate of the point on the y-axis equidistant from C(-3,0) and D(4,5) is 16/5 -/
theorem equidistant_point_y_coordinate :
  let C : ℝ × ℝ := (-3, 0)
  let D : ℝ × ℝ := (4, 5)
  let P : ℝ → ℝ × ℝ := λ y => (0, y)
  ∃ y : ℝ, (dist (P y) C = dist (P y) D) ∧ (y = 16/5)
  := by sorry

where
  dist : ℝ × ℝ → ℝ × ℝ → ℝ
  | (x₁, y₁), (x₂, y₂) => Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

end NUMINAMATH_CALUDE_equidistant_point_y_coordinate_l229_22922


namespace NUMINAMATH_CALUDE_radish_distribution_l229_22955

theorem radish_distribution (total : ℕ) (difference : ℕ) : 
  total = 88 → difference = 14 → ∃ (first second : ℕ), 
    first + second = total ∧ 
    second = first + difference ∧ 
    first = 37 := by
  sorry

end NUMINAMATH_CALUDE_radish_distribution_l229_22955


namespace NUMINAMATH_CALUDE_remainder_theorem_l229_22962

theorem remainder_theorem (x y u v : ℕ) (hx : x > 0) (hy : y > 0) 
  (h_div : x = u * y + v) (h_rem : v < y) : 
  (x^2 + 3*u*y + v^2) % y = (2*v^2) % y := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l229_22962


namespace NUMINAMATH_CALUDE_exact_five_blue_probability_l229_22901

def total_marbles : ℕ := 12
def blue_marbles : ℕ := 8
def red_marbles : ℕ := 4
def total_draws : ℕ := 8
def blue_draws : ℕ := 5

def probability_blue : ℚ := blue_marbles / total_marbles
def probability_red : ℚ := red_marbles / total_marbles

theorem exact_five_blue_probability :
  (Nat.choose total_draws blue_draws : ℚ) *
  (probability_blue ^ blue_draws) *
  (probability_red ^ (total_draws - blue_draws)) =
  (56 : ℚ) * 32 / 6561 :=
sorry

end NUMINAMATH_CALUDE_exact_five_blue_probability_l229_22901


namespace NUMINAMATH_CALUDE_parabola_directrix_distance_l229_22921

/-- Proves that for a parabola y = ax² (a > 0) with a point M(3, 2),
    if the distance from M to the directrix is 4, then a = 1/8 -/
theorem parabola_directrix_distance (a : ℝ) : 
  a > 0 → 
  (let M : ℝ × ℝ := (3, 2)
   let directrix_y : ℝ := -1 / (4 * a)
   let distance_to_directrix : ℝ := |M.2 - directrix_y|
   distance_to_directrix = 4) →
  a = 1/8 := by
sorry

end NUMINAMATH_CALUDE_parabola_directrix_distance_l229_22921


namespace NUMINAMATH_CALUDE_union_equals_reals_l229_22913

def S : Set ℝ := {x | x < -1 ∨ x > 5}
def T (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 8}

theorem union_equals_reals (a : ℝ) : 
  S ∪ T a = Set.univ ↔ -3 < a ∧ a < -1 := by sorry

end NUMINAMATH_CALUDE_union_equals_reals_l229_22913


namespace NUMINAMATH_CALUDE_product_properties_l229_22917

theorem product_properties (x y z : ℕ) : 
  x = 15 ∧ y = 5 ∧ z = 8 →
  (x * y * z = 600) ∧
  ((x - 10) * y * z = 200) ∧
  ((x + 5) * y * z = 1200) := by
sorry

end NUMINAMATH_CALUDE_product_properties_l229_22917


namespace NUMINAMATH_CALUDE_division_remainder_l229_22948

theorem division_remainder (x y : ℕ+) (h1 : x = 7 * y + 3) (h2 : 11 * y - x = 1) : 
  2 * x ≡ 2 [ZMOD 6] := by
sorry

end NUMINAMATH_CALUDE_division_remainder_l229_22948


namespace NUMINAMATH_CALUDE_cone_volume_l229_22977

/-- The volume of a cone with given slant height and lateral surface area -/
theorem cone_volume (l : ℝ) (lateral_area : ℝ) (h : l = 2) (h' : lateral_area = 2 * Real.pi) :
  ∃ (r : ℝ) (h : ℝ),
    r > 0 ∧ h > 0 ∧
    lateral_area = Real.pi * r * l ∧
    h^2 + r^2 = l^2 ∧
    (1/3 : ℝ) * Real.pi * r^2 * h = (Real.sqrt 3 * Real.pi) / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_cone_volume_l229_22977


namespace NUMINAMATH_CALUDE_problem_statement_l229_22956

theorem problem_statement : (-1)^53 + 2^(3^4 + 4^3 - 6 * 7) = 2^103 - 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l229_22956


namespace NUMINAMATH_CALUDE_line_equation_with_given_area_l229_22978

/-- Given a line passing through points (a, 0) and (b, 0) where b > a, 
    cutting a triangular region from the first quadrant with area S,
    prove that the equation of the line is 0 = -2Sx + (b-a)^2y + 2Sa - 2Sb -/
theorem line_equation_with_given_area (a b S : ℝ) (h1 : b > a) (h2 : S > 0) :
  ∃ (f : ℝ → ℝ), 
    (∀ x, f x = 0 ↔ -2 * S * x + (b - a)^2 * x + 2 * S * a - 2 * S * b = 0) ∧
    f a = 0 ∧ 
    f b = 0 ∧
    (∃ k, k > 0 ∧ f k > 0 ∧ (k - 0) * (b - a) / 2 = S) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_with_given_area_l229_22978


namespace NUMINAMATH_CALUDE_sum_inequality_l229_22965

theorem sum_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 2/y = 2) :
  8*x + y ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l229_22965


namespace NUMINAMATH_CALUDE_lg_calculation_l229_22973

-- Define lg as the base 10 logarithm
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem lg_calculation : lg 5 * lg 20 + (lg 2)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_lg_calculation_l229_22973


namespace NUMINAMATH_CALUDE_nicky_run_time_l229_22969

/-- The time Nicky runs before Cristina catches up to him in a 200-meter race --/
theorem nicky_run_time (race_distance : ℝ) (head_start : ℝ) (cristina_speed : ℝ) (nicky_speed : ℝ)
  (h1 : race_distance = 200)
  (h2 : head_start = 12)
  (h3 : cristina_speed = 5)
  (h4 : nicky_speed = 3) :
  let catch_up_time := (nicky_speed * head_start) / (cristina_speed - nicky_speed)
  head_start + catch_up_time = 30 := by
  sorry

#check nicky_run_time

end NUMINAMATH_CALUDE_nicky_run_time_l229_22969


namespace NUMINAMATH_CALUDE_unique_solution_l229_22958

/-- Represents a three-digit number with distinct digits -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  distinct : hundreds ≠ tens ∧ tens ≠ ones ∧ hundreds ≠ ones
  valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≥ 0 ∧ tens ≤ 9 ∧ ones ≥ 0 ∧ ones ≤ 9

/-- The statements made by the students -/
def statements (n : ThreeDigitNumber) : Prop :=
  (n.tens > n.hundreds ∧ n.tens > n.ones) ∧  -- Petya's statement
  (n.ones = 8) ∧                             -- Vasya's statement
  (n.ones > n.hundreds ∧ n.ones > n.tens) ∧  -- Tolya's statement
  (n.ones = (n.hundreds + n.tens) / 2)       -- Dima's statement

/-- The theorem to prove -/
theorem unique_solution :
  ∃! n : ThreeDigitNumber, (∃ (i : Fin 4), ¬statements n) ∧
    (∀ (j : Fin 4), j ≠ i → statements n) ∧
    n.hundreds = 7 ∧ n.tens = 9 ∧ n.ones = 8 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l229_22958


namespace NUMINAMATH_CALUDE_square_difference_equals_one_l229_22942

theorem square_difference_equals_one : (825 : ℤ) * 825 - 824 * 826 = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equals_one_l229_22942


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l229_22944

-- Define the triangle ABC
def A : ℝ × ℝ := (-2, 4)
def B : ℝ × ℝ := (-1, 1)
def C : ℝ × ℝ := (3, 3)

-- Define the perpendicular bisector of BC
def perpendicular_bisector_BC (x y : ℝ) : Prop :=
  2 * x + y - 4 = 0

-- Define the area of triangle ABC
def area_ABC : ℝ := 7

-- Theorem statement
theorem triangle_ABC_properties :
  (perpendicular_bisector_BC (A.1 + B.1 + C.1) (A.2 + B.2 + C.2)) ∧
  (area_ABC = 7) := by
  sorry

end NUMINAMATH_CALUDE_triangle_ABC_properties_l229_22944


namespace NUMINAMATH_CALUDE_quadratic_roots_positive_implies_a_zero_l229_22919

theorem quadratic_roots_positive_implies_a_zero 
  (a b c : ℝ) 
  (h : ∀ (p : ℝ), p > 0 → ∀ (x : ℝ), a * x^2 + b * x + c + p = 0 → x > 0) :
  a = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_positive_implies_a_zero_l229_22919


namespace NUMINAMATH_CALUDE_copper_button_percentage_is_28_percent_l229_22911

/-- Represents the composition of items in a basket --/
structure BasketComposition where
  pin_percentage : ℝ
  brass_button_percentage : ℝ
  copper_button_percentage : ℝ

/-- The percentage of copper buttons in the basket --/
def copper_button_percentage (b : BasketComposition) : ℝ :=
  b.copper_button_percentage

/-- Theorem stating the percentage of copper buttons in the basket --/
theorem copper_button_percentage_is_28_percent 
  (b : BasketComposition)
  (h1 : b.pin_percentage = 0.3)
  (h2 : b.brass_button_percentage = 0.42)
  (h3 : b.pin_percentage + b.brass_button_percentage + b.copper_button_percentage = 1) :
  copper_button_percentage b = 0.28 := by
  sorry

#check copper_button_percentage_is_28_percent

end NUMINAMATH_CALUDE_copper_button_percentage_is_28_percent_l229_22911


namespace NUMINAMATH_CALUDE_rectangle_area_l229_22982

theorem rectangle_area (L B : ℝ) (h1 : L - B = 23) (h2 : 2 * L + 2 * B = 246) : L * B = 3650 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l229_22982


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l229_22994

theorem complex_fraction_evaluation :
  2 + (3 / (4 + (5 / (6 + (7/8))))) = 137/52 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l229_22994


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_difference_l229_22996

theorem consecutive_odd_integers_difference (x y z : ℤ) : 
  (y = x + 2 ∧ z = y + 2) →  -- consecutive odd integers
  z = 15 →                   -- third integer is 15
  3 * x > 2 * z →            -- 3 times first is more than twice third
  3 * x - 2 * z = 3 :=       -- difference is 3
by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_difference_l229_22996


namespace NUMINAMATH_CALUDE_arithmetic_sequence_n_equals_10_l229_22985

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  first_term : a 1 = 1
  sum_3_5 : a 3 + a 5 = 14
  sum_n : ∃ n : ℕ, n > 0 ∧ (n : ℝ) * (a 1 + a n) / 2 = 100

/-- The theorem stating that n = 10 for the given arithmetic sequence -/
theorem arithmetic_sequence_n_equals_10 (seq : ArithmeticSequence) :
  ∃ n : ℕ, n > 0 ∧ (n : ℝ) * (seq.a 1 + seq.a n) / 2 = 100 ∧ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_n_equals_10_l229_22985


namespace NUMINAMATH_CALUDE_quadratic_solution_absolute_value_l229_22987

theorem quadratic_solution_absolute_value : ∃ (x : ℝ), x^2 + 18*x + 81 = 0 ∧ |x| = 9 := by sorry

end NUMINAMATH_CALUDE_quadratic_solution_absolute_value_l229_22987


namespace NUMINAMATH_CALUDE_fourth_root_equivalence_l229_22990

theorem fourth_root_equivalence (x : ℝ) (hx : x > 0) : 
  (x * x^(1/3))^(1/4) = x^(1/3) := by
sorry

end NUMINAMATH_CALUDE_fourth_root_equivalence_l229_22990


namespace NUMINAMATH_CALUDE_dice_probability_l229_22976

/-- The number of dice being rolled -/
def n : ℕ := 7

/-- The number of faces on each die -/
def faces : ℕ := 6

/-- The number of faces showing a number greater than 4 -/
def favorable_faces : ℕ := 2

/-- The number of dice that should show a number greater than 4 -/
def k : ℕ := 3

/-- The probability of rolling a number greater than 4 on a single die -/
def p : ℚ := favorable_faces / faces

/-- The probability of not rolling a number greater than 4 on a single die -/
def q : ℚ := 1 - p

theorem dice_probability :
  (n.choose k * p^k * q^(n-k) : ℚ) = 560/2187 := by sorry

end NUMINAMATH_CALUDE_dice_probability_l229_22976


namespace NUMINAMATH_CALUDE_katherine_age_when_mel_21_l229_22910

/-- Katherine's age when Mel is 21 years old -/
def katherines_age (mels_age : ℕ) (age_difference : ℕ) : ℕ :=
  mels_age + age_difference

/-- Theorem stating Katherine's age when Mel is 21 -/
theorem katherine_age_when_mel_21 :
  katherines_age 21 3 = 24 := by
  sorry

end NUMINAMATH_CALUDE_katherine_age_when_mel_21_l229_22910


namespace NUMINAMATH_CALUDE_calculate_interest_rate_l229_22959

/-- Calculate the interest rate given principal, simple interest, and time -/
theorem calculate_interest_rate (P SI T : ℝ) (h_positive : P > 0 ∧ SI > 0 ∧ T > 0) :
  ∃ R : ℝ, SI = P * R * T / 100 := by
  sorry

end NUMINAMATH_CALUDE_calculate_interest_rate_l229_22959


namespace NUMINAMATH_CALUDE_min_value_theorem_l229_22906

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (1 / x) + (4 / (y + 1)) ≥ 9 / 2 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + y₀ = 1 ∧ (1 / x₀) + (4 / (y₀ + 1)) = 9 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l229_22906


namespace NUMINAMATH_CALUDE_percentage_relation_l229_22957

theorem percentage_relation (p t j : ℝ) (e : ℝ) : 
  j = 0.75 * p → 
  j = 0.8 * t → 
  t = p * (1 - e / 100) → 
  e = 6.25 :=
by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l229_22957


namespace NUMINAMATH_CALUDE_polynomial_coefficient_b_l229_22974

theorem polynomial_coefficient_b (a c d : ℝ) : 
  ∃ (p q r s : ℂ),
    (∀ x : ℂ, x^4 + a*x^3 + 49*x^2 + c*x + d = 0 ↔ x = p ∨ x = q ∨ x = r ∨ x = s) ∧
    p + q = 5 + 2*I ∧
    r * s = 10 - I ∧
    p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧
    p.im ≠ 0 ∧ q.im ≠ 0 ∧ r.im ≠ 0 ∧ s.im ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_b_l229_22974


namespace NUMINAMATH_CALUDE_median_sum_ge_four_circumradius_l229_22941

/-- A triangle is represented by its three vertices in the real plane -/
structure Triangle where
  A : Real × Real
  B : Real × Real
  C : Real × Real

/-- The circumradius of a triangle -/
noncomputable def circumradius (t : Triangle) : Real := sorry

/-- The length of a median in a triangle -/
noncomputable def median_length (t : Triangle) (vertex : Fin 3) : Real := sorry

/-- Predicate to check if a triangle is not obtuse -/
def is_not_obtuse (t : Triangle) : Prop := sorry

/-- Theorem: For any non-obtuse triangle, the sum of its median lengths
    is greater than or equal to four times its circumradius -/
theorem median_sum_ge_four_circumradius (t : Triangle) 
  (h : is_not_obtuse t) : 
  (median_length t 0) + (median_length t 1) + (median_length t 2) ≥ 4 * (circumradius t) := by
  sorry

end NUMINAMATH_CALUDE_median_sum_ge_four_circumradius_l229_22941


namespace NUMINAMATH_CALUDE_sticker_count_l229_22971

/-- The number of stickers Karl has -/
def karl_stickers : ℕ := 25

/-- The number of stickers Ryan has -/
def ryan_stickers : ℕ := karl_stickers + 20

/-- The number of stickers Ben has -/
def ben_stickers : ℕ := ryan_stickers - 10

/-- The total number of stickers placed in the book -/
def total_stickers : ℕ := karl_stickers + ryan_stickers + ben_stickers

theorem sticker_count : total_stickers = 105 := by
  sorry

end NUMINAMATH_CALUDE_sticker_count_l229_22971


namespace NUMINAMATH_CALUDE_rotate_A_equals_B_l229_22997

-- Define a 2x2 grid
structure Grid2x2 :=
  (cells : Fin 2 → Fin 2 → Bool)

-- Define rotations
def rotate90CounterClockwise (g : Grid2x2) : Grid2x2 :=
  { cells := λ i j => g.cells (1 - j) i }

-- Define the initial position of 'A'
def initialA : Grid2x2 :=
  { cells := λ i j => (i = 1 ∧ j = 0) ∨ (i = 1 ∧ j = 1) ∨ (i = 0 ∧ j = 1) }

-- Define the final position of 'A' (option B)
def finalA : Grid2x2 :=
  { cells := λ i j => (i = 0 ∧ j = 1) ∨ (i = 1 ∧ j = 1) ∨ (i = 2 ∧ j = 1) ∨ (i = 2 ∧ j = 0) }

-- Theorem statement
theorem rotate_A_equals_B : rotate90CounterClockwise initialA = finalA := by
  sorry

end NUMINAMATH_CALUDE_rotate_A_equals_B_l229_22997


namespace NUMINAMATH_CALUDE_equation_solutions_l229_22935

theorem equation_solutions : 
  {x : ℝ | (x - 2)^2 + (x - 2) = 0} = {2, 1} := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l229_22935


namespace NUMINAMATH_CALUDE_trajectory_is_two_circles_l229_22908

-- Define the set of complex numbers satisfying the equation
def S : Set ℂ := {z : ℂ | Complex.abs z ^ 2 - 3 * Complex.abs z + 2 = 0}

-- Define the trajectory of z
def trajectory (z : ℂ) : Set ℂ := {w : ℂ | Complex.abs w = Complex.abs z}

-- Theorem statement
theorem trajectory_is_two_circles :
  ∃ (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ r₁ > 0 ∧ r₂ > 0 ∧
  (∀ z ∈ S, (trajectory z = {w : ℂ | Complex.abs w = r₁} ∨
             trajectory z = {w : ℂ | Complex.abs w = r₂})) :=
sorry

end NUMINAMATH_CALUDE_trajectory_is_two_circles_l229_22908


namespace NUMINAMATH_CALUDE_max_notebooks_buyable_l229_22947

def john_money : ℚ := 35.45
def notebook_cost : ℚ := 3.75

theorem max_notebooks_buyable :
  ⌊john_money / notebook_cost⌋ = 9 :=
sorry

end NUMINAMATH_CALUDE_max_notebooks_buyable_l229_22947


namespace NUMINAMATH_CALUDE_hexagon_side_length_l229_22963

/-- A regular hexagon with a line segment connecting opposite vertices. -/
structure RegularHexagon :=
  (side_length : ℝ)
  (center_to_midpoint : ℝ)

/-- Theorem: If the distance from the center to the midpoint of a line segment
    connecting opposite vertices in a regular hexagon is 9, then the side length
    is 6√3. -/
theorem hexagon_side_length (h : RegularHexagon) 
    (h_center_to_midpoint : h.center_to_midpoint = 9) : 
    h.side_length = 6 * Real.sqrt 3 := by
  sorry

#check hexagon_side_length

end NUMINAMATH_CALUDE_hexagon_side_length_l229_22963


namespace NUMINAMATH_CALUDE_range_of_m_l229_22915

theorem range_of_m (x y m : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, ∀ y ∈ Set.Icc 2 3, y^2 - x*y - m*x^2 ≤ 0) →
  m ∈ Set.Ioi 6 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l229_22915


namespace NUMINAMATH_CALUDE_old_machine_rate_proof_l229_22904

/-- The rate at which the new machine makes bolts (in bolts per hour) -/
def new_machine_rate : ℝ := 150

/-- The time both machines work together (in hours) -/
def work_time : ℝ := 2

/-- The total number of bolts produced by both machines -/
def total_bolts : ℝ := 500

/-- The rate at which the old machine makes bolts (in bolts per hour) -/
def old_machine_rate : ℝ := 100

theorem old_machine_rate_proof :
  old_machine_rate * work_time + new_machine_rate * work_time = total_bolts :=
by sorry

end NUMINAMATH_CALUDE_old_machine_rate_proof_l229_22904


namespace NUMINAMATH_CALUDE_part_one_part_two_l229_22966

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 1|

-- Part 1
theorem part_one (m : ℝ) (h1 : m > 0) 
  (h2 : ∀ x, x ∈ Set.Icc (-2 : ℝ) 2 ↔ f (x + 1/2) ≤ 2*m + 1) : 
  m = 3/2 := by sorry

-- Part 2
theorem part_two : 
  (∃ a : ℝ, ∀ x y : ℝ, f x ≤ 2^y + a/(2^y) + |2*x + 3|) ∧ 
  (∀ a : ℝ, (∀ x y : ℝ, f x ≤ 2^y + a/(2^y) + |2*x + 3|) → a ≥ 4) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l229_22966


namespace NUMINAMATH_CALUDE_quadratic_square_completion_l229_22940

theorem quadratic_square_completion (a b c : ℤ) : 
  (∀ x : ℝ, 64 * x^2 - 96 * x - 48 = 0 ↔ (a * x + b)^2 = c) →
  a > 0 →
  a + b + c = 86 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_square_completion_l229_22940


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_roots_l229_22933

theorem isosceles_right_triangle_roots (a b : ℂ) : 
  a ^ 2 = 2 * b ∧ b ≠ 0 ↔ 
  ∃ (x₁ x₂ : ℂ), x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ 
    (∀ x, x ^ 2 + a * x + b = 0 ↔ x = x₁ ∨ x = x₂) ∧
    (x₂ / x₁ = Complex.I ∨ x₂ / x₁ = -Complex.I) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_roots_l229_22933


namespace NUMINAMATH_CALUDE_cistern_length_l229_22999

/-- Represents a rectangular cistern with water --/
structure Cistern where
  length : ℝ
  width : ℝ
  depth : ℝ

/-- Calculates the total wet surface area of a cistern --/
def wetSurfaceArea (c : Cistern) : ℝ :=
  c.length * c.width + 2 * c.length * c.depth + 2 * c.width * c.depth

/-- Theorem: A cistern with given dimensions has a length of 4 meters --/
theorem cistern_length : 
  ∃ (c : Cistern), c.width = 8 ∧ c.depth = 1.25 ∧ wetSurfaceArea c = 62 → c.length = 4 := by
  sorry

end NUMINAMATH_CALUDE_cistern_length_l229_22999


namespace NUMINAMATH_CALUDE_max_fraction_two_digit_nums_l229_22932

theorem max_fraction_two_digit_nums (x y z : ℕ) : 
  (10 ≤ x ∧ x ≤ 99) → 
  (10 ≤ y ∧ y ≤ 99) → 
  (10 ≤ z ∧ z ≤ 99) → 
  (x + y + z) / 3 = 60 → 
  (x + y) / z ≤ 17 := by
sorry

end NUMINAMATH_CALUDE_max_fraction_two_digit_nums_l229_22932


namespace NUMINAMATH_CALUDE_probability_red_or_white_l229_22970

-- Define the total number of marbles
def total_marbles : ℕ := 60

-- Define the number of blue marbles
def blue_marbles : ℕ := 5

-- Define the number of red marbles
def red_marbles : ℕ := 9

-- Define the number of white marbles
def white_marbles : ℕ := total_marbles - (blue_marbles + red_marbles)

-- Theorem statement
theorem probability_red_or_white :
  (red_marbles + white_marbles : ℚ) / total_marbles = 11 / 12 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_or_white_l229_22970


namespace NUMINAMATH_CALUDE_wall_building_time_l229_22993

theorem wall_building_time (avery_time tom_time : ℝ) : 
  avery_time = 4 →
  1 / avery_time + 1 / tom_time + 0.5 / tom_time = 1 →
  tom_time = 2 :=
by sorry

end NUMINAMATH_CALUDE_wall_building_time_l229_22993


namespace NUMINAMATH_CALUDE_plant_arrangement_theorem_l229_22938

/-- Represents the number of ways to arrange plants under lamps -/
def plant_arrangement_count : ℕ :=
  let basil_count : ℕ := 3
  let aloe_count : ℕ := 2
  let white_lamp_count : ℕ := 3
  let red_lamp_count : ℕ := 3
  sorry

/-- Theorem stating that the number of plant arrangements is 128 -/
theorem plant_arrangement_theorem : plant_arrangement_count = 128 := by
  sorry

end NUMINAMATH_CALUDE_plant_arrangement_theorem_l229_22938


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l229_22902

theorem quadratic_inequality_range (a : ℝ) :
  (∃ x : ℝ, x^2 - a*x + 5 < 0) ↔ (a < -2 * Real.sqrt 5 ∨ a > 2 * Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l229_22902


namespace NUMINAMATH_CALUDE_min_value_implies_m_l229_22905

/-- The function f(x) = -x^3 + 6x^2 + m -/
def f (x m : ℝ) : ℝ := -x^3 + 6*x^2 + m

/-- Theorem: If f(x) has a minimum value of 23, then m = 23 -/
theorem min_value_implies_m (m : ℝ) : 
  (∃ (y : ℝ), ∀ (x : ℝ), f x m ≥ y ∧ ∃ (x₀ : ℝ), f x₀ m = y) ∧ 
  (∃ (x₀ : ℝ), f x₀ m = 23) → 
  m = 23 := by
  sorry


end NUMINAMATH_CALUDE_min_value_implies_m_l229_22905


namespace NUMINAMATH_CALUDE_number_of_girls_l229_22930

/-- The number of boys -/
def num_boys : ℕ := 4

/-- The number of possible arrangements -/
def total_arrangements : ℕ := 2880

/-- A function that calculates the number of possible arrangements given the number of boys and girls -/
def calculate_arrangements (boys girls : ℕ) : ℕ :=
  Nat.factorial boys * Nat.factorial girls

/-- Theorem stating that there are 5 girls -/
theorem number_of_girls : ∃ (girls : ℕ), girls = 5 ∧ 
  calculate_arrangements num_boys girls = total_arrangements :=
sorry

end NUMINAMATH_CALUDE_number_of_girls_l229_22930


namespace NUMINAMATH_CALUDE_square_binomial_identity_l229_22992

theorem square_binomial_identity : (1/2)^2 + 2*(1/2)*5 + 5^2 = 121/4 := by
  sorry

end NUMINAMATH_CALUDE_square_binomial_identity_l229_22992


namespace NUMINAMATH_CALUDE_intersection_implies_m_in_range_l229_22960

/-- A line intersects a circle if the distance from the circle's center to the line is less than the circle's radius -/
def line_intersects_circle (a b c : ℝ) (x₀ y₀ r : ℝ) : Prop :=
  (|a * x₀ + b * y₀ + c| / Real.sqrt (a^2 + b^2)) < r

/-- The problem statement -/
theorem intersection_implies_m_in_range :
  ∃ m : ℤ, (3 ≤ m ∧ m ≤ 6) ∧
  line_intersects_circle 4 3 (2 * ↑m) (-3) 1 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_m_in_range_l229_22960


namespace NUMINAMATH_CALUDE_number_of_pairs_l229_22936

theorem number_of_pairs (n : ℕ) (h : n = 12) : Nat.choose n 2 = 66 := by
  sorry

end NUMINAMATH_CALUDE_number_of_pairs_l229_22936


namespace NUMINAMATH_CALUDE_olaf_collection_l229_22943

def total_cars (initial : ℕ) (uncle : ℕ) : ℕ :=
  let grandpa := 2 * uncle
  let dad := 10
  let mum := dad + 5
  let auntie := 6
  let cousin_liam := dad / 2
  let cousin_emma := uncle / 3
  let grandmother := 3 * auntie
  initial + grandpa + dad + mum + auntie + uncle + cousin_liam + cousin_emma + grandmother

theorem olaf_collection (initial : ℕ) (uncle : ℕ) 
  (h1 : initial = 150)
  (h2 : uncle = 5)
  (h3 : auntie = uncle + 1) :
  total_cars initial uncle = 220 := by
  sorry

#eval total_cars 150 5

end NUMINAMATH_CALUDE_olaf_collection_l229_22943


namespace NUMINAMATH_CALUDE_system_solution_equivalence_l229_22909

-- Define the system of equations
def system (x y z : ℝ) : Prop :=
  (x * y + 2 * x * z + 3 * y * z = -6) ∧
  (x^2 * y^2 + 4 * x^2 * z^2 - 9 * y^2 * z^2 = 36) ∧
  (x^3 * y^3 + 8 * x^3 * z^3 + 27 * y^3 * z^3 = -216)

-- Define the solution set
def solution_set (x y z : ℝ) : Prop :=
  (y = 0 ∧ x * z = -3) ∨
  (z = 0 ∧ x * y = -6) ∨
  (x = 3 ∧ y = -2) ∨
  (x = -3 ∧ y = 2) ∨
  (x = 3 ∧ z = -1) ∨
  (x = -3 ∧ z = 1)

-- State the theorem
theorem system_solution_equivalence :
  ∀ x y z : ℝ, system x y z ↔ solution_set x y z :=
sorry

end NUMINAMATH_CALUDE_system_solution_equivalence_l229_22909


namespace NUMINAMATH_CALUDE_gcd_problem_l229_22903

theorem gcd_problem (p : Nat) (h : Nat.Prime p) (hp : p = 107) :
  Nat.gcd (p^7 + 1) (p^7 + p^3 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l229_22903


namespace NUMINAMATH_CALUDE_whitewashing_cost_example_l229_22967

/-- Calculate the cost of white washing a room's walls given its dimensions and openings. -/
def whitewashingCost (length width height doorWidth doorHeight windowWidth windowHeight : ℝ)
  (numWindows : ℕ) (costPerSquareFoot : ℝ) : ℝ :=
  let wallArea := 2 * (length * height + width * height)
  let doorArea := doorWidth * doorHeight
  let windowArea := numWindows * (windowWidth * windowHeight)
  let areaToPaint := wallArea - doorArea - windowArea
  areaToPaint * costPerSquareFoot

/-- The cost of white washing the room is Rs. 2718. -/
theorem whitewashing_cost_example :
  whitewashingCost 25 15 12 6 3 4 3 3 3 = 2718 := by
  sorry

end NUMINAMATH_CALUDE_whitewashing_cost_example_l229_22967


namespace NUMINAMATH_CALUDE_tara_book_sales_tara_clarinet_purchase_l229_22952

theorem tara_book_sales (initial_savings : ℕ) (clarinet_cost : ℕ) (book_price : ℕ) : ℕ :=
  let halfway_goal := clarinet_cost / 2
  let books_to_halfway := (halfway_goal - initial_savings) / book_price
  let books_to_full_goal := clarinet_cost / book_price
  books_to_halfway + books_to_full_goal

theorem tara_clarinet_purchase : tara_book_sales 10 90 5 = 25 := by
  sorry

end NUMINAMATH_CALUDE_tara_book_sales_tara_clarinet_purchase_l229_22952


namespace NUMINAMATH_CALUDE_parallel_lines_distance_l229_22928

/-- A circle intersected by three equally spaced parallel lines -/
structure ParallelLinesCircle where
  /-- The radius of the circle -/
  r : ℝ
  /-- The distance between adjacent parallel lines -/
  d : ℝ
  /-- The length of the first chord -/
  chord1 : ℝ
  /-- The length of the second chord -/
  chord2 : ℝ
  /-- The length of the third chord -/
  chord3 : ℝ
  /-- The first chord has length 40 -/
  chord1_eq : chord1 = 40
  /-- The second chord has length 40 -/
  chord2_eq : chord2 = 40
  /-- The third chord has length 30 -/
  chord3_eq : chord3 = 30

/-- The theorem stating that the distance between adjacent parallel lines is 20√6 -/
theorem parallel_lines_distance (c : ParallelLinesCircle) : c.d = 20 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_distance_l229_22928


namespace NUMINAMATH_CALUDE_angle_sum_in_hexagon_with_triangles_l229_22980

/-- Represents a hexagon with two connected triangles -/
structure HexagonWithTriangles where
  /-- Angle A of the hexagon -/
  angle_A : ℝ
  /-- Angle B of the hexagon -/
  angle_B : ℝ
  /-- Angle C of one of the connected triangles -/
  angle_C : ℝ
  /-- An angle x in the figure -/
  x : ℝ
  /-- An angle y in the figure -/
  y : ℝ
  /-- The sum of angles in a hexagon is 720° -/
  hexagon_sum : angle_A + angle_B + (360 - x) + 90 + (114 - y) = 720

/-- Theorem stating that x + y = 50° in the given hexagon with triangles -/
theorem angle_sum_in_hexagon_with_triangles (h : HexagonWithTriangles)
    (h_A : h.angle_A = 30)
    (h_B : h.angle_B = 76)
    (h_C : h.angle_C = 24) :
    h.x + h.y = 50 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_in_hexagon_with_triangles_l229_22980


namespace NUMINAMATH_CALUDE_no_preimage_set_l229_22989

def f (x : ℝ) : ℝ := -x^2 + 2*x

theorem no_preimage_set (p : ℝ) : 
  (∀ x : ℝ, f x ≠ p) ↔ p ∈ Set.Ioi 1 :=
sorry

end NUMINAMATH_CALUDE_no_preimage_set_l229_22989


namespace NUMINAMATH_CALUDE_red_shirt_pairs_l229_22972

theorem red_shirt_pairs (green_students : ℕ) (red_students : ℕ) (total_students : ℕ) (total_pairs : ℕ) (green_green_pairs : ℕ) : 
  green_students = 65 →
  red_students = 85 →
  total_students = 150 →
  total_pairs = 75 →
  green_green_pairs = 30 →
  (green_students + red_students = total_students) →
  (2 * total_pairs = total_students) →
  (∃ (red_red_pairs : ℕ), red_red_pairs = 40 ∧ 
    green_green_pairs + red_red_pairs + (total_pairs - green_green_pairs - red_red_pairs) = total_pairs) :=
by sorry

end NUMINAMATH_CALUDE_red_shirt_pairs_l229_22972


namespace NUMINAMATH_CALUDE_prob_one_male_is_three_fifths_l229_22923

/-- Represents the class composition and sampling results -/
structure ClassSampling where
  total_students : ℕ
  male_students : ℕ
  selected_students : ℕ
  chosen_students : ℕ

/-- Calculates the number of male students selected in stratified sampling -/
def male_selected (c : ClassSampling) : ℕ :=
  (c.selected_students * c.male_students) / c.total_students

/-- Calculates the number of female students selected in stratified sampling -/
def female_selected (c : ClassSampling) : ℕ :=
  c.selected_students - male_selected c

/-- Calculates the probability of selecting exactly one male student from the chosen students -/
def prob_one_male (c : ClassSampling) : ℚ :=
  (male_selected c * female_selected c : ℚ) / (Nat.choose c.selected_students c.chosen_students : ℚ)

/-- Theorem stating the probability of selecting exactly one male student is 3/5 -/
theorem prob_one_male_is_three_fifths (c : ClassSampling) 
  (h1 : c.total_students = 50)
  (h2 : c.male_students = 30)
  (h3 : c.selected_students = 5)
  (h4 : c.chosen_students = 2) :
  prob_one_male c = 3/5 := by
  sorry

#eval prob_one_male ⟨50, 30, 5, 2⟩

end NUMINAMATH_CALUDE_prob_one_male_is_three_fifths_l229_22923


namespace NUMINAMATH_CALUDE_simple_interest_rate_l229_22920

/-- Given a principal amount and a simple interest rate,
    if the amount after 5 years is 7/6 of the principal,
    then the rate is 1/30 -/
theorem simple_interest_rate (P R : ℚ) (P_pos : 0 < P) :
  P + P * R * 5 = (7 / 6) * P →
  R = 1 / 30 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_l229_22920


namespace NUMINAMATH_CALUDE_largest_and_smallest_decimal_l229_22926

def Digits : Set ℕ := {0, 1, 2, 3}

def IsValidDecimal (d : ℚ) : Prop :=
  ∃ (a b c : ℕ) (n : ℕ), 
    a ∈ Digits ∧ b ∈ Digits ∧ c ∈ Digits ∧
    d = (100 * a + 10 * b + c : ℚ) / (10^n : ℚ) ∧
    (n = 0 ∨ n = 1 ∨ n = 2 ∨ n = 3)

theorem largest_and_smallest_decimal :
  (∀ d : ℚ, IsValidDecimal d → d ≤ 321) ∧
  (∀ d : ℚ, IsValidDecimal d → 0.123 ≤ d) :=
sorry

end NUMINAMATH_CALUDE_largest_and_smallest_decimal_l229_22926


namespace NUMINAMATH_CALUDE_matrix_sum_theorem_l229_22925

theorem matrix_sum_theorem (x y z k : ℝ) 
  (h1 : x * (x^2 - y*z) - y * (z^2 - y*x) + z * (z*x - y^2) = 0)
  (h2 : x + y + z = k)
  (h3 : y + z ≠ k)
  (h4 : z + x ≠ k)
  (h5 : x + y ≠ k) :
  x / (y + z - k) + y / (z + x - k) + z / (x + y - k) = -3 := by
sorry

end NUMINAMATH_CALUDE_matrix_sum_theorem_l229_22925


namespace NUMINAMATH_CALUDE_cube_volume_ratio_l229_22934

theorem cube_volume_ratio (a b : ℝ) (h : a^2 / b^2 = 9 / 25) :
  (b^3) / (a^3) = 125 / 27 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_ratio_l229_22934


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l229_22946

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l229_22946


namespace NUMINAMATH_CALUDE_wilson_pays_twelve_l229_22975

/-- The total amount Wilson pays at the fast-food restaurant -/
def wilsonTotalPaid (hamburgerPrice : ℕ) (hamburgerCount : ℕ) (colaPrice : ℕ) (colaCount : ℕ) (discountAmount : ℕ) : ℕ :=
  hamburgerPrice * hamburgerCount + colaPrice * colaCount - discountAmount

/-- Theorem: Wilson pays $12 in total -/
theorem wilson_pays_twelve :
  wilsonTotalPaid 5 2 2 3 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_wilson_pays_twelve_l229_22975


namespace NUMINAMATH_CALUDE_constant_function_sqrt_l229_22995

/-- Given a function f that is constant 3 for all real inputs, 
    prove that f(√x) + 1 = 4 for all non-negative real x -/
theorem constant_function_sqrt (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = 3) :
  ∀ x : ℝ, x ≥ 0 → f (Real.sqrt x) + 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_constant_function_sqrt_l229_22995


namespace NUMINAMATH_CALUDE_milk_cartons_problem_l229_22961

theorem milk_cartons_problem (total : ℕ) (ratio : ℚ) : total = 24 → ratio = 7/1 → ∃ regular : ℕ, regular = 3 ∧ regular * (1 + ratio) = total := by
  sorry

end NUMINAMATH_CALUDE_milk_cartons_problem_l229_22961


namespace NUMINAMATH_CALUDE_ratio_and_average_theorem_l229_22951

theorem ratio_and_average_theorem (a b c d : ℕ+) : 
  (a : ℚ) / b = 2 / 3 ∧ 
  (b : ℚ) / c = 3 / 4 ∧ 
  (c : ℚ) / d = 4 / 5 ∧ 
  (a + b + c + d : ℚ) / 4 = 42 →
  a = 24 := by sorry

end NUMINAMATH_CALUDE_ratio_and_average_theorem_l229_22951


namespace NUMINAMATH_CALUDE_triangle_similarity_properties_l229_22918

/-- Triangle properties for medians and altitudes similarity --/
theorem triangle_similarity_properties (a b c : ℝ) (h_order : a ≤ b ∧ b ≤ c) :
  (∀ (ma mb mc : ℝ), 
    (ma = (1/2) * Real.sqrt (2*b^2 + 2*c^2 - a^2)) →
    (mb = (1/2) * Real.sqrt (2*a^2 + 2*c^2 - b^2)) →
    (mc = (1/2) * Real.sqrt (2*a^2 + 2*b^2 - c^2)) →
    (a / mc = b / mb ∧ b / mb = c / ma) →
    (2 * b^2 = a^2 + c^2)) ∧
  (∀ (ha hb hc : ℝ),
    (ha * a = hb * b ∧ hb * b = hc * c) →
    (ha / hb = b / a ∧ ha / hc = c / a ∧ hb / hc = c / b)) := by
  sorry


end NUMINAMATH_CALUDE_triangle_similarity_properties_l229_22918


namespace NUMINAMATH_CALUDE_common_root_quadratics_l229_22914

theorem common_root_quadratics (a : ℝ) : 
  (∃ x : ℝ, x^2 + a*x + 1 = 0 ∧ x^2 + x + a = 0) ↔ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_common_root_quadratics_l229_22914


namespace NUMINAMATH_CALUDE_work_earnings_equation_l229_22929

theorem work_earnings_equation (t : ℝ) : 
  (t + 2) * (4 * t - 4) = (2 * t - 3) * (t + 3) + 3 → 
  t = (-1 + Real.sqrt 5) / 2 := by
sorry

end NUMINAMATH_CALUDE_work_earnings_equation_l229_22929


namespace NUMINAMATH_CALUDE_cost_per_deck_is_8_l229_22939

/-- The cost of a single trick deck -/
def cost_per_deck : ℝ := sorry

/-- The number of decks Victor bought -/
def victor_decks : ℕ := 6

/-- The number of decks Victor's friend bought -/
def friend_decks : ℕ := 2

/-- The total amount spent -/
def total_spent : ℝ := 64

/-- Theorem stating that the cost per deck is 8 dollars -/
theorem cost_per_deck_is_8 : cost_per_deck = 8 :=
  by sorry

end NUMINAMATH_CALUDE_cost_per_deck_is_8_l229_22939


namespace NUMINAMATH_CALUDE_sculpture_cost_equivalence_l229_22983

/-- Represents the exchange rate between US dollars and Namibian dollars -/
def usd_to_nad : ℚ := 8

/-- Represents the exchange rate between US dollars and Chinese yuan -/
def usd_to_cny : ℚ := 5

/-- Represents the cost of the sculpture in Namibian dollars -/
def sculpture_cost_nad : ℚ := 160

/-- Theorem stating the equivalence of the sculpture's cost in Chinese yuan -/
theorem sculpture_cost_equivalence :
  (sculpture_cost_nad / usd_to_nad) * usd_to_cny = 100 := by
  sorry

end NUMINAMATH_CALUDE_sculpture_cost_equivalence_l229_22983


namespace NUMINAMATH_CALUDE_evaluate_expression_l229_22986

theorem evaluate_expression (x y : ℝ) (hx : x = 2) (hy : y = 3) :
  y * (y - 3 * x + 2) = -3 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l229_22986
