import Mathlib

namespace NUMINAMATH_CALUDE_fraction_sum_theorem_l3309_330991

theorem fraction_sum_theorem (a b c d : ℝ) 
  (h1 : a/b + b/c + c/d + d/a = 6) 
  (h2 : a/c + b/d + c/a + d/b = 8) : 
  a/b + c/d = 2 ∨ a/b + c/d = 4 := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_theorem_l3309_330991


namespace NUMINAMATH_CALUDE_investment_calculation_l3309_330903

/-- Given two investors P and Q, where the profit is divided in the ratio 2:3
    and P invested Rs 40000, prove that Q invested Rs 60000 -/
theorem investment_calculation (P Q : ℕ) (profit_ratio : ℚ) :
  P = 40000 →
  profit_ratio = 2 / 3 →
  Q = 60000 :=
by sorry

end NUMINAMATH_CALUDE_investment_calculation_l3309_330903


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l3309_330936

/-- Given two lines in the form ax + by + c = 0, this function returns true if they are parallel -/
def are_parallel (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  a1 * b2 = a2 * b1

/-- Given a line ax + by + c = 0 and a point (x0, y0), this function returns true if the point lies on the line -/
def point_on_line (a b c x0 y0 : ℝ) : Prop :=
  a * x0 + b * y0 + c = 0

theorem line_through_point_parallel_to_line :
  are_parallel 2 (-3) 12 2 (-3) 4 ∧
  point_on_line 2 (-3) 12 (-3) 2 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l3309_330936


namespace NUMINAMATH_CALUDE_thirty_five_million_scientific_notation_l3309_330997

/-- Expresses a number in scientific notation -/
def scientific_notation (n : ℕ) : ℝ × ℤ :=
  sorry

theorem thirty_five_million_scientific_notation :
  scientific_notation 35000000 = (3.5, 7) :=
sorry

end NUMINAMATH_CALUDE_thirty_five_million_scientific_notation_l3309_330997


namespace NUMINAMATH_CALUDE_toy_car_cost_l3309_330929

-- Define the given values
def initial_amount : ℚ := 17.80
def num_cars : ℕ := 4
def race_track_cost : ℚ := 6.00
def remaining_amount : ℚ := 8.00

-- Define the theorem
theorem toy_car_cost :
  (initial_amount - remaining_amount - race_track_cost) / num_cars = 0.95 := by
  sorry

end NUMINAMATH_CALUDE_toy_car_cost_l3309_330929


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_and_remainder_l3309_330933

def arithmetic_sequence_sum (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) : ℕ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem arithmetic_sequence_sum_and_remainder :
  let sum := arithmetic_sequence_sum 3 5 103
  sum = 1113 ∧ sum % 10 = 3 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_and_remainder_l3309_330933


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l3309_330996

theorem consecutive_odd_integers_sum (x : ℤ) : 
  x > 0 ∧ 
  Odd x ∧ 
  Odd (x + 2) ∧ 
  x * (x + 2) = 945 → 
  x + (x + 2) = 60 := by
sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l3309_330996


namespace NUMINAMATH_CALUDE_sqrt_seven_to_sixth_power_l3309_330957

theorem sqrt_seven_to_sixth_power : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_seven_to_sixth_power_l3309_330957


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3309_330958

theorem pure_imaginary_complex_number (a : ℝ) : 
  let z : ℂ := Complex.mk (a^2 + 2*a - 3) (a^2 - 4*a + 3)
  (z.re = 0 ∧ z.im ≠ 0) → a = -3 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3309_330958


namespace NUMINAMATH_CALUDE_point_on_number_line_l3309_330993

/-- Given points P, Q, and R on a number line, where Q is halfway between P and R,
    P is at -6, and Q is at -1, prove that R is at 4. -/
theorem point_on_number_line (P Q R : ℝ) : 
  Q = (P + R) / 2 → P = -6 → Q = -1 → R = 4 := by
  sorry

end NUMINAMATH_CALUDE_point_on_number_line_l3309_330993


namespace NUMINAMATH_CALUDE_double_inequality_solution_l3309_330902

theorem double_inequality_solution (x : ℝ) : 
  (3 ≤ |x - 3| ∧ |x - 3| ≤ 6 ∧ x ≤ 8) ↔ ((-3 ≤ x ∧ x ≤ 3) ∨ (6 ≤ x ∧ x ≤ 8)) :=
by sorry

end NUMINAMATH_CALUDE_double_inequality_solution_l3309_330902


namespace NUMINAMATH_CALUDE_problem_solution_l3309_330935

-- Define the type for a triple of real numbers
def Triple := ℝ × ℝ × ℝ

-- Define the conditions for the problem
def is_geometric_progression (t : Triple) : Prop :=
  let (x, y, z) := t
  y^2 = x * z

def is_arithmetic_progression_after_subtraction (t : Triple) : Prop :=
  let (x, y, z) := t
  2 * y = x + (z - 16)

def is_geometric_progression_after_subtractions (t : Triple) : Prop :=
  let (x, y, z) := t
  (y - 2)^2 = x * (z - 16)

-- Define the main theorem
theorem problem_solution :
  ∀ t : Triple,
    is_geometric_progression t ∧
    is_arithmetic_progression_after_subtraction t ∧
    is_geometric_progression_after_subtractions t →
    t = (1, 5, 25) ∨ t = (1/9, 13/9, 169/9) :=
by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3309_330935


namespace NUMINAMATH_CALUDE_min_value_implies_a_l3309_330944

theorem min_value_implies_a (a : ℝ) (h_a : a > 0) :
  (∃ x y : ℝ, x ≥ 1 ∧ x + y ≤ 3 ∧ y ≥ a * (x - 3)) →
  (∀ x y : ℝ, x ≥ 1 → x + y ≤ 3 → y ≥ a * (x - 3) → 2 * x + y ≥ 1) →
  (∃ x y : ℝ, x ≥ 1 ∧ x + y ≤ 3 ∧ y ≥ a * (x - 3) ∧ 2 * x + y = 1) →
  a = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_implies_a_l3309_330944


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_9883_l3309_330959

theorem largest_prime_factor_of_9883 :
  ∃ (p : ℕ), p.Prime ∧ p ∣ 9883 ∧ ∀ (q : ℕ), q.Prime → q ∣ 9883 → q ≤ p :=
by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_9883_l3309_330959


namespace NUMINAMATH_CALUDE_vovochka_candy_theorem_l3309_330982

/-- Represents the candy distribution problem --/
structure CandyDistribution where
  total_candies : ℕ
  num_classmates : ℕ
  min_group_size : ℕ
  min_group_candies : ℕ

/-- Calculates the maximum number of candies that can be kept --/
def max_kept_candies (cd : CandyDistribution) : ℕ := 
  cd.total_candies - (cd.num_classmates * (cd.min_group_candies / cd.min_group_size))

/-- The theorem stating the maximum number of candies that can be kept --/
theorem vovochka_candy_theorem (cd : CandyDistribution) 
  (h1 : cd.total_candies = 200)
  (h2 : cd.num_classmates = 25)
  (h3 : cd.min_group_size = 16)
  (h4 : cd.min_group_candies = 100) :
  max_kept_candies cd = 37 := by
  sorry

#eval max_kept_candies { total_candies := 200, num_classmates := 25, min_group_size := 16, min_group_candies := 100 }

end NUMINAMATH_CALUDE_vovochka_candy_theorem_l3309_330982


namespace NUMINAMATH_CALUDE_g_composition_equals_514_l3309_330960

def g (x : ℝ) : ℝ := 7 * x + 3

theorem g_composition_equals_514 : g (g (g 1)) = 514 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_equals_514_l3309_330960


namespace NUMINAMATH_CALUDE_gain_percentage_calculation_l3309_330948

def cost_price : ℝ := 180
def selling_price : ℝ := 216

theorem gain_percentage_calculation : 
  let gain_percentage := (selling_price / cost_price - 1) * 100
  gain_percentage = 20 := by sorry

end NUMINAMATH_CALUDE_gain_percentage_calculation_l3309_330948


namespace NUMINAMATH_CALUDE_skew_lines_sufficient_not_necessary_l3309_330923

/-- Represents a line in 3D space -/
structure Line3D where
  -- Add necessary fields to define a line in 3D space
  -- This is a placeholder structure

/-- Two lines are skew if they are not coplanar -/
def are_skew (l1 l2 : Line3D) : Prop :=
  sorry

/-- Two lines have no common point if they don't intersect -/
def have_no_common_point (l1 l2 : Line3D) : Prop :=
  sorry

/-- Theorem stating that "are_skew" is a sufficient but not necessary condition for "have_no_common_point" -/
theorem skew_lines_sufficient_not_necessary :
  ∃ (l1 l2 l3 l4 : Line3D),
    (are_skew l1 l2 → have_no_common_point l1 l2) ∧
    (have_no_common_point l3 l4 ∧ ¬are_skew l3 l4) :=
  sorry

end NUMINAMATH_CALUDE_skew_lines_sufficient_not_necessary_l3309_330923


namespace NUMINAMATH_CALUDE_final_result_proof_l3309_330976

theorem final_result_proof (chosen_number : ℕ) (h : chosen_number = 2976) :
  (chosen_number / 12) - 240 = 8 := by
  sorry

end NUMINAMATH_CALUDE_final_result_proof_l3309_330976


namespace NUMINAMATH_CALUDE_probability_different_specialties_l3309_330946

def total_students : ℕ := 50
def art_students : ℕ := 15
def dance_students : ℕ := 35

theorem probability_different_specialties :
  let total_combinations := total_students.choose 2
  let different_specialty_combinations := art_students * dance_students
  (different_specialty_combinations : ℚ) / total_combinations = 3 / 7 :=
sorry

end NUMINAMATH_CALUDE_probability_different_specialties_l3309_330946


namespace NUMINAMATH_CALUDE_sum_of_odd_numbers_l3309_330989

theorem sum_of_odd_numbers (n : ℕ) (sum_of_first_n_odds : ℕ → ℕ) 
  (h1 : ∀ k, sum_of_first_n_odds k = k^2)
  (h2 : sum_of_first_n_odds 100 = 10000)
  (h3 : sum_of_first_n_odds 50 = 2500) :
  sum_of_first_n_odds 100 - sum_of_first_n_odds 50 = 7500 := by
  sorry

#check sum_of_odd_numbers

end NUMINAMATH_CALUDE_sum_of_odd_numbers_l3309_330989


namespace NUMINAMATH_CALUDE_line_properties_l3309_330910

-- Define the line
def line_equation (x : ℝ) : ℝ := -4 * x - 12

-- Theorem statement
theorem line_properties :
  (∀ x, line_equation x = -4 * x - 12) →
  (line_equation (-3) = 0) →
  (line_equation 0 = -12) ∧
  (line_equation 2 = -20) := by
  sorry

end NUMINAMATH_CALUDE_line_properties_l3309_330910


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3309_330913

theorem complex_equation_solution (z : ℂ) : z = Complex.I * (2 - z) → z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3309_330913


namespace NUMINAMATH_CALUDE_burger_length_l3309_330987

theorem burger_length (share : ℝ) (h1 : share = 6) : 2 * share = 12 := by
  sorry

#check burger_length

end NUMINAMATH_CALUDE_burger_length_l3309_330987


namespace NUMINAMATH_CALUDE_z_purely_imaginary_z_squared_over_z_plus_5_plus_2i_l3309_330953

def z (m : ℝ) : ℂ := (2 * m^2 - 3 * m - 2 : ℝ) + (m^2 - 3 * m + 2 : ℝ) * Complex.I

theorem z_purely_imaginary : z (-1/2) = Complex.I * ((-1/4)^2 - 3 * (-1/4) + 2) := by sorry

theorem z_squared_over_z_plus_5_plus_2i :
  z 0 ^ 2 / (z 0 + 5 + 2 * Complex.I) = -32/25 - 24/25 * Complex.I := by sorry

end NUMINAMATH_CALUDE_z_purely_imaginary_z_squared_over_z_plus_5_plus_2i_l3309_330953


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3309_330979

theorem quadratic_equation_solution (a b : ℕ+) :
  (∃ x : ℝ, x^2 + 14*x = 24 ∧ x > 0 ∧ x = Real.sqrt a - b) →
  a + b = 80 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3309_330979


namespace NUMINAMATH_CALUDE_clara_stickers_l3309_330973

theorem clara_stickers (initial_stickers : ℕ) (stickers_to_boy : ℕ) (final_stickers : ℕ) : 
  initial_stickers = 100 →
  final_stickers = 45 →
  final_stickers = (initial_stickers - stickers_to_boy) / 2 →
  stickers_to_boy = 10 := by
  sorry

end NUMINAMATH_CALUDE_clara_stickers_l3309_330973


namespace NUMINAMATH_CALUDE_percentage_not_sold_is_62_14_l3309_330969

/-- Represents the book inventory and sales data for a bookshop --/
structure BookshopData where
  initial_fiction : ℕ
  initial_nonfiction : ℕ
  fiction_sold : ℕ
  nonfiction_sold : ℕ
  fiction_returned : ℕ
  nonfiction_returned : ℕ

/-- Calculates the percentage of books not sold --/
def percentage_not_sold (data : BookshopData) : ℚ :=
  let total_initial := data.initial_fiction + data.initial_nonfiction
  let net_fiction_sold := data.fiction_sold - data.fiction_returned
  let net_nonfiction_sold := data.nonfiction_sold - data.nonfiction_returned
  let total_sold := net_fiction_sold + net_nonfiction_sold
  let not_sold := total_initial - total_sold
  (not_sold : ℚ) / (total_initial : ℚ) * 100

/-- The main theorem stating the percentage of books not sold --/
theorem percentage_not_sold_is_62_14 (data : BookshopData)
  (h1 : data.initial_fiction = 400)
  (h2 : data.initial_nonfiction = 300)
  (h3 : data.fiction_sold = 150)
  (h4 : data.nonfiction_sold = 160)
  (h5 : data.fiction_returned = 30)
  (h6 : data.nonfiction_returned = 15) :
  percentage_not_sold data = 62.14 := by
  sorry

#eval percentage_not_sold {
  initial_fiction := 400,
  initial_nonfiction := 300,
  fiction_sold := 150,
  nonfiction_sold := 160,
  fiction_returned := 30,
  nonfiction_returned := 15
}

end NUMINAMATH_CALUDE_percentage_not_sold_is_62_14_l3309_330969


namespace NUMINAMATH_CALUDE_binary_110101_is_53_l3309_330919

def binary_to_decimal (b : List Bool) : ℕ :=
  (List.enum b).foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_110101_is_53 :
  binary_to_decimal [true, false, true, false, true, true] = 53 := by
  sorry

end NUMINAMATH_CALUDE_binary_110101_is_53_l3309_330919


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l3309_330927

-- Define the propositions
def p (a : ℝ) : Prop := a > 2
def q (a : ℝ) : Prop := ¬(∀ x : ℝ, x^2 + a*x + 1 ≥ 0)

-- Theorem statement
theorem p_sufficient_not_necessary_for_q :
  (∃ a : ℝ, p a → q a) ∧ (∃ a : ℝ, q a ∧ ¬(p a)) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l3309_330927


namespace NUMINAMATH_CALUDE_f_zero_points_range_l3309_330994

/-- The function f(x) = ax^2 + x - 1 + 3a -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x - 1 + 3 * a

/-- The set of a values for which f has zero points in [-1, 1] -/
def A : Set ℝ := {a : ℝ | ∃ x, x ∈ Set.Icc (-1 : ℝ) 1 ∧ f a x = 0}

theorem f_zero_points_range :
  A = Set.Icc (0 : ℝ) (1/2) :=
sorry

end NUMINAMATH_CALUDE_f_zero_points_range_l3309_330994


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3309_330932

theorem imaginary_part_of_z : 
  let z : ℂ := (1 - I) / (1 + 3*I)
  Complex.im z = -2/5 :=
by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3309_330932


namespace NUMINAMATH_CALUDE_line_equation_proof_l3309_330920

/-- 
Given a line mx + (n/2)y - 1 = 0 with a y-intercept of -1 and an angle of inclination 
twice that of the line √3x - y - 3√3 = 0, prove that m = -√3 and n = -2.
-/
theorem line_equation_proof (m n : ℝ) : 
  (∀ x y, m * x + (n / 2) * y - 1 = 0) →  -- Line equation
  (0 + (n / 2) * (-1) - 1 = 0) →  -- y-intercept is -1
  (Real.arctan m = 2 * Real.arctan (Real.sqrt 3)) →  -- Angle of inclination relation
  (m = -Real.sqrt 3 ∧ n = -2) := by
sorry

end NUMINAMATH_CALUDE_line_equation_proof_l3309_330920


namespace NUMINAMATH_CALUDE_rope_length_problem_l3309_330977

theorem rope_length_problem (total_ropes : ℕ) (avg_length_all : ℝ) (avg_length_third : ℝ) :
  total_ropes = 6 →
  avg_length_all = 80 →
  avg_length_third = 70 →
  let third_ropes := total_ropes / 3
  let remaining_ropes := total_ropes - third_ropes
  let total_length := total_ropes * avg_length_all
  let third_length := third_ropes * avg_length_third
  let remaining_length := total_length - third_length
  remaining_length / remaining_ropes = 85 := by
sorry

end NUMINAMATH_CALUDE_rope_length_problem_l3309_330977


namespace NUMINAMATH_CALUDE_two_numbers_difference_l3309_330906

theorem two_numbers_difference (a b : ℕ) 
  (sum_eq : a + b = 17402)
  (b_div_10 : 10 ∣ b)
  (a_eq_b_div_10 : a = b / 10) :
  b - a = 14238 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l3309_330906


namespace NUMINAMATH_CALUDE_jenny_activities_lcm_l3309_330986

theorem jenny_activities_lcm : Nat.lcm (Nat.lcm 6 12) 18 = 36 := by
  sorry

end NUMINAMATH_CALUDE_jenny_activities_lcm_l3309_330986


namespace NUMINAMATH_CALUDE_game_win_fraction_l3309_330904

theorem game_win_fraction (total_matches : ℕ) (points_per_win : ℕ) (player1_points : ℕ) :
  total_matches = 8 →
  points_per_win = 10 →
  player1_points = 20 →
  (total_matches - player1_points / points_per_win) / total_matches = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_game_win_fraction_l3309_330904


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l3309_330940

theorem quadratic_roots_condition (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > -2 ∧ x₂ > -2 ∧
    x₁^2 + (2*m + 6)*x₁ + 4*m + 12 = 0 ∧
    x₂^2 + (2*m + 6)*x₂ + 4*m + 12 = 0) ↔
  m ≤ -3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l3309_330940


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l3309_330914

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (((x^2 + y^2) * (4*x^2 + 2*y^2)).sqrt) / (x*y) ≥ 2 + Real.sqrt 2 :=
sorry

theorem min_value_achievable :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧
  (((x^2 + y^2) * (4*x^2 + 2*y^2)).sqrt) / (x*y) = 2 + Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l3309_330914


namespace NUMINAMATH_CALUDE_least_product_xy_l3309_330999

theorem least_product_xy (x y : ℕ+) (h : (x : ℚ)⁻¹ + (3 * y : ℚ)⁻¹ = (6 : ℚ)⁻¹) :
  (∀ a b : ℕ+, (a : ℚ)⁻¹ + (3 * b : ℚ)⁻¹ = (6 : ℚ)⁻¹ → x * y ≤ a * b) ∧ x * y = 48 :=
sorry

end NUMINAMATH_CALUDE_least_product_xy_l3309_330999


namespace NUMINAMATH_CALUDE_range_of_m_l3309_330985

theorem range_of_m (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_eq : 1/x + 4/y = 1) (h_ineq : ∃ m : ℝ, x + y/4 < m^2 - 3*m) :
  ∃ m : ℝ, m < -1 ∨ m > 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3309_330985


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l3309_330918

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x > 0, P x) ↔ ∀ x > 0, ¬ P x := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x > 0, x^2 - 2*x + 1 > 0) ↔ (∀ x > 0, x^2 - 2*x + 1 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l3309_330918


namespace NUMINAMATH_CALUDE_cube_root_monotone_l3309_330970

theorem cube_root_monotone (a b : ℝ) : a ≤ b → (a ^ (1/3 : ℝ)) ≤ (b ^ (1/3 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_monotone_l3309_330970


namespace NUMINAMATH_CALUDE_number_count_in_average_calculation_l3309_330990

theorem number_count_in_average_calculation 
  (initial_average : ℚ)
  (correct_average : ℚ)
  (incorrect_number : ℚ)
  (correct_number : ℚ)
  (h1 : initial_average = 46)
  (h2 : correct_average = 50)
  (h3 : incorrect_number = 25)
  (h4 : correct_number = 65) :
  ∃ (n : ℕ), n > 0 ∧ 
    (n : ℚ) * correct_average = (n : ℚ) * initial_average + (correct_number - incorrect_number) ∧
    n = 10 := by
  sorry

end NUMINAMATH_CALUDE_number_count_in_average_calculation_l3309_330990


namespace NUMINAMATH_CALUDE_quadratic_polynomial_with_complex_root_l3309_330954

theorem quadratic_polynomial_with_complex_root :
  ∃ (a b c : ℝ), 
    (a = 3) ∧ 
    (∀ (z : ℂ), z = (2 : ℝ) + (2 : ℝ) * I → a * z^2 + b * z + c = 0) ∧
    (a * X^2 + b * X + c = 3 * X^2 - 12 * X + 24) :=
sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_with_complex_root_l3309_330954


namespace NUMINAMATH_CALUDE_all_fruits_fallen_by_day_10_l3309_330925

def fruits_on_tree (n : ℕ) : ℕ := 46

def fruits_falling (day : ℕ) : ℕ :=
  if day ≤ 9 then day
  else 1

def total_fallen_fruits (day : ℕ) : ℕ :=
  if day ≤ 9 then day * (day + 1) / 2
  else 45 + (day - 9)

theorem all_fruits_fallen_by_day_10 :
  ∀ day : ℕ, day ≥ 10 → total_fallen_fruits day ≥ fruits_on_tree 0 :=
by sorry

end NUMINAMATH_CALUDE_all_fruits_fallen_by_day_10_l3309_330925


namespace NUMINAMATH_CALUDE_container_volume_ratio_l3309_330966

theorem container_volume_ratio : 
  ∀ (V1 V2 : ℝ), V1 > 0 → V2 > 0 → 
  (3/5 : ℝ) * V1 = (2/3 : ℝ) * V2 → 
  V1 / V2 = 10/9 := by
sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l3309_330966


namespace NUMINAMATH_CALUDE_derivative_of_y_l3309_330984

noncomputable def y (x : ℝ) : ℝ :=
  (1 / Real.sqrt 8) * Real.log ((4 + Real.sqrt 8 * Real.tanh (x / 2)) / (4 - Real.sqrt 8 * Real.tanh (x / 2)))

theorem derivative_of_y (x : ℝ) :
  deriv y x = 1 / (2 * (Real.cosh (x / 2) ^ 2 + 1)) :=
sorry

end NUMINAMATH_CALUDE_derivative_of_y_l3309_330984


namespace NUMINAMATH_CALUDE_rational_function_representation_unique_l3309_330924

-- Define the structure of a rational function representation
structure RationalFunctionRep where
  terms : List (ℚ × ℚ × ℕ)  -- List of (coefficient, root, multiplicity)

-- Define equality for RationalFunctionRep
def rep_equal (r1 r2 : RationalFunctionRep) : Prop :=
  r1.terms = r2.terms

-- Define the evaluation of a RationalFunctionRep at a point
noncomputable def evaluate (r : RationalFunctionRep) (x : ℚ) : ℚ :=
  sorry

-- State the theorem
theorem rational_function_representation_unique 
  (R : ℚ → ℚ) (rep1 rep2 : RationalFunctionRep) :
  (∀ x, evaluate rep1 x = R x) → 
  (∀ x, evaluate rep2 x = R x) → 
  rep_equal rep1 rep2 :=
sorry

end NUMINAMATH_CALUDE_rational_function_representation_unique_l3309_330924


namespace NUMINAMATH_CALUDE_power_function_quadrant_propositions_l3309_330951

-- Define a power function
def is_power_function (f : ℝ → ℝ) : Prop := 
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x ^ b

-- Define the property of not passing through the fourth quadrant
def not_in_fourth_quadrant (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x = y → ¬(x > 0 ∧ y < 0)

-- The main theorem
theorem power_function_quadrant_propositions :
  let P : (ℝ → ℝ) → Prop := λ f => is_power_function f → not_in_fourth_quadrant f
  let contrapositive : (ℝ → ℝ) → Prop := λ f => ¬(not_in_fourth_quadrant f) → ¬(is_power_function f)
  let converse : (ℝ → ℝ) → Prop := λ f => not_in_fourth_quadrant f → is_power_function f
  let inverse : (ℝ → ℝ) → Prop := λ f => ¬(is_power_function f) → ¬(not_in_fourth_quadrant f)
  (∀ f : ℝ → ℝ, P f) ∧
  (∀ f : ℝ → ℝ, contrapositive f) ∧
  ¬(∀ f : ℝ → ℝ, converse f) ∧
  ¬(∀ f : ℝ → ℝ, inverse f) :=
by sorry

end NUMINAMATH_CALUDE_power_function_quadrant_propositions_l3309_330951


namespace NUMINAMATH_CALUDE_choir_arrangement_l3309_330939

theorem choir_arrangement (n : ℕ) : 
  (n % 9 = 0 ∧ n % 10 = 0 ∧ n % 11 = 0 ∧ n % 14 = 0) ↔ n ≥ 6930 ∧ ∀ m : ℕ, m < n → (m % 9 ≠ 0 ∨ m % 10 ≠ 0 ∨ m % 11 ≠ 0 ∨ m % 14 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_choir_arrangement_l3309_330939


namespace NUMINAMATH_CALUDE_unknown_number_problem_l3309_330978

theorem unknown_number_problem (x : ℝ) : 
  (50 : ℝ) / 100 * 100 = (20 : ℝ) / 100 * x + 47 → x = 15 := by
sorry

end NUMINAMATH_CALUDE_unknown_number_problem_l3309_330978


namespace NUMINAMATH_CALUDE_quinn_free_donuts_l3309_330926

/-- The number of books required to earn one donut coupon -/
def books_per_coupon : ℕ := 5

/-- The number of books Quinn reads per week -/
def books_per_week : ℕ := 2

/-- The number of weeks Quinn reads -/
def weeks_read : ℕ := 10

/-- The total number of books Quinn reads -/
def total_books : ℕ := books_per_week * weeks_read

/-- The number of free donuts Quinn is eligible for -/
def free_donuts : ℕ := total_books / books_per_coupon

theorem quinn_free_donuts : free_donuts = 4 := by
  sorry

end NUMINAMATH_CALUDE_quinn_free_donuts_l3309_330926


namespace NUMINAMATH_CALUDE_sum_of_specific_arithmetic_progression_l3309_330901

/-- Sum of an arithmetic progression -/
def sum_arithmetic_progression (a : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Theorem: The sum of the first 20 terms of an arithmetic progression
    with first term 30 and common difference -3 is equal to 30 -/
theorem sum_of_specific_arithmetic_progression :
  sum_arithmetic_progression 30 (-3) 20 = 30 := by
sorry

end NUMINAMATH_CALUDE_sum_of_specific_arithmetic_progression_l3309_330901


namespace NUMINAMATH_CALUDE_wire_ratio_proof_l3309_330931

theorem wire_ratio_proof (total_length longer_length shorter_length : ℝ) 
  (h1 : total_length = 14)
  (h2 : shorter_length = 4)
  (h3 : longer_length = total_length - shorter_length) :
  shorter_length / longer_length = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_wire_ratio_proof_l3309_330931


namespace NUMINAMATH_CALUDE_angle_120_degrees_is_200_vens_l3309_330930

/-- Represents the number of vens in a full circle -/
def vens_in_full_circle : ℕ := 600

/-- Represents the number of degrees in a full circle -/
def degrees_in_full_circle : ℕ := 360

/-- Represents the angle in degrees we want to convert to vens -/
def angle_in_degrees : ℕ := 120

/-- Theorem stating that 120 degrees is equivalent to 200 vens -/
theorem angle_120_degrees_is_200_vens :
  (angle_in_degrees : ℚ) * vens_in_full_circle / degrees_in_full_circle = 200 := by
  sorry


end NUMINAMATH_CALUDE_angle_120_degrees_is_200_vens_l3309_330930


namespace NUMINAMATH_CALUDE_digit_move_correction_l3309_330917

theorem digit_move_correction : ∃ (a b c : ℕ), 
  (a = 101 ∧ b = 102 ∧ c = 1) ∧ 
  (a - b ≠ c) ∧
  (a - 10^2 = c) := by
  sorry

end NUMINAMATH_CALUDE_digit_move_correction_l3309_330917


namespace NUMINAMATH_CALUDE_smallest_integer_square_75_more_than_double_l3309_330921

theorem smallest_integer_square_75_more_than_double :
  ∃ x : ℤ, x^2 = 2*x + 75 ∧ ∀ y : ℤ, y^2 = 2*y + 75 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_square_75_more_than_double_l3309_330921


namespace NUMINAMATH_CALUDE_book_arrangement_count_l3309_330950

theorem book_arrangement_count : ℕ := by
  -- Define the number of math books and English books
  let math_books : ℕ := 4
  let english_books : ℕ := 4

  -- Define the number of ways to arrange math books
  let math_arrangements : ℕ := Nat.factorial math_books

  -- Define the number of ways to arrange English books
  let english_arrangements : ℕ := Nat.factorial english_books

  -- Define the number of ways to arrange the two blocks (always 1 in this case)
  let block_arrangements : ℕ := 1

  -- Calculate the total number of arrangements
  let total_arrangements : ℕ := block_arrangements * math_arrangements * english_arrangements

  -- Prove that the total number of arrangements is 576
  sorry

-- The final statement to be proven
#check book_arrangement_count

end NUMINAMATH_CALUDE_book_arrangement_count_l3309_330950


namespace NUMINAMATH_CALUDE_coastal_village_population_l3309_330995

theorem coastal_village_population (total_population : ℕ) 
  (h1 : total_population = 540) 
  (h2 : ∃ (part_size : ℕ), 4 * part_size = total_population) 
  (h3 : ∃ (male_population : ℕ), male_population = 2 * (total_population / 4)) :
  ∃ (male_population : ℕ), male_population = 270 := by
sorry

end NUMINAMATH_CALUDE_coastal_village_population_l3309_330995


namespace NUMINAMATH_CALUDE_original_price_calculation_l3309_330992

/-- 
Theorem: If an item's price is increased by 15%, then decreased by 20%, 
resulting in a final price of 46 yuan, the original price was 50 yuan.
-/
theorem original_price_calculation (original_price : ℝ) : 
  (original_price * 1.15 * 0.8 = 46) → original_price = 50 := by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l3309_330992


namespace NUMINAMATH_CALUDE_circle_equation_solution_l3309_330934

theorem circle_equation_solution (a b : ℝ) (h : a^2 + b^2 = 12*a - 4*b + 20) : a - b = 8 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_solution_l3309_330934


namespace NUMINAMATH_CALUDE_pastry_chef_eggs_l3309_330905

theorem pastry_chef_eggs :
  ∃ n : ℕ,
    n > 0 ∧
    n % 43 = 0 ∧
    n % 2 = 1 ∧
    n % 3 = 1 ∧
    n % 4 = 1 ∧
    n % 5 = 1 ∧
    n % 6 = 1 ∧
    n / 43 < 9 ∧
    (∀ m : ℕ, m > 0 ∧ m < n →
      ¬(m % 43 = 0 ∧
        m % 2 = 1 ∧
        m % 3 = 1 ∧
        m % 4 = 1 ∧
        m % 5 = 1 ∧
        m % 6 = 1 ∧
        m / 43 < 9)) ∧
    n = 301 := by
  sorry

end NUMINAMATH_CALUDE_pastry_chef_eggs_l3309_330905


namespace NUMINAMATH_CALUDE_bottle_recycling_result_l3309_330916

/-- Calculates the number of new bottles created through recycling -/
def recycleBottles (initialBottles : ℕ) : ℕ :=
  let firstRound := initialBottles / 5
  let secondRound := firstRound / 5
  let thirdRound := secondRound / 5
  firstRound + secondRound + thirdRound

/-- Represents the recycling process with initial conditions -/
def bottleRecyclingProcess (initialBottles : ℕ) : Prop :=
  recycleBottles initialBottles = 179

/-- Theorem stating the result of the bottle recycling process -/
theorem bottle_recycling_result :
  bottleRecyclingProcess 729 := by sorry

end NUMINAMATH_CALUDE_bottle_recycling_result_l3309_330916


namespace NUMINAMATH_CALUDE_platform_length_l3309_330949

/-- Given a train of length 200 meters that crosses a platform in 50 seconds
    and a signal pole in 42 seconds, the length of the platform is 38 meters. -/
theorem platform_length (train_length : ℝ) (time_platform : ℝ) (time_pole : ℝ) 
  (h1 : train_length = 200)
  (h2 : time_platform = 50)
  (h3 : time_pole = 42) :
  let train_speed := train_length / time_pole
  let platform_length := train_speed * time_platform - train_length
  platform_length = 38 := by
  sorry


end NUMINAMATH_CALUDE_platform_length_l3309_330949


namespace NUMINAMATH_CALUDE_expression_simplification_l3309_330945

theorem expression_simplification (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a^(2/3) * b^(1/2)) * (-3 * a^(1/2) * b^(1/3)) / ((1/3) * a^(1/6) * b^(5/6)) = -9 * a :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l3309_330945


namespace NUMINAMATH_CALUDE_square_division_theorem_l3309_330907

theorem square_division_theorem :
  ∃ (s : ℝ) (a b : ℝ) (n m : ℕ),
    s > 0 ∧ a > 0 ∧ b > 0 ∧
    b / a ≤ 1.25 ∧
    n + m = 40 ∧
    s * s = n * a * a + m * b * b :=
by sorry

end NUMINAMATH_CALUDE_square_division_theorem_l3309_330907


namespace NUMINAMATH_CALUDE_function_properties_l3309_330983

open Real

theorem function_properties (f : ℝ → ℝ) 
  (h : ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) < -1) :
  (f (-2) > f 2 + 4) ∧ 
  (∀ x : ℝ, f x > f (x + 1) + 1) ∧ 
  (∃ x : ℝ, x ≥ 0 ∧ f (sqrt x) + sqrt x < f 0) ∧
  (∀ a : ℝ, a ≠ 0 → f (|a| + 1 / |a|) + |a| + 1 / |a| < f 2 + 3) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3309_330983


namespace NUMINAMATH_CALUDE_natasha_quarters_l3309_330968

theorem natasha_quarters (q : ℕ) : 
  (10 < (q : ℚ) * (1/4) ∧ (q : ℚ) * (1/4) < 200) ∧ 
  q % 4 = 2 ∧ q % 5 = 2 ∧ q % 6 = 2 ↔ 
  ∃ k : ℕ, k ≥ 1 ∧ k ≤ 13 ∧ q = 60 * k + 2 :=
by sorry

end NUMINAMATH_CALUDE_natasha_quarters_l3309_330968


namespace NUMINAMATH_CALUDE_secret_spread_day_secret_spread_saturday_unique_day_for_3280_l3309_330911

def secret_spread (n : ℕ) : ℕ := (3^(n+1) - 1) / 2 + 1

theorem secret_spread_day : ∃ (d : ℕ), secret_spread d = 3280 :=
  sorry

theorem secret_spread_saturday : secret_spread 7 = 3280 :=
  sorry

theorem unique_day_for_3280 : ∀ (d : ℕ), secret_spread d = 3280 → d = 7 :=
  sorry

end NUMINAMATH_CALUDE_secret_spread_day_secret_spread_saturday_unique_day_for_3280_l3309_330911


namespace NUMINAMATH_CALUDE_provider_choice_count_l3309_330908

/-- The total number of service providers --/
def total_providers : ℕ := 25

/-- The number of providers available to the youngest child --/
def restricted_providers : ℕ := 15

/-- The number of children --/
def num_children : ℕ := 4

/-- The number of ways to choose service providers for the children --/
def choose_providers : ℕ := total_providers * (total_providers - 1) * (total_providers - 2) * restricted_providers

theorem provider_choice_count :
  choose_providers = 207000 :=
sorry

end NUMINAMATH_CALUDE_provider_choice_count_l3309_330908


namespace NUMINAMATH_CALUDE_fermat_little_theorem_extension_l3309_330909

theorem fermat_little_theorem_extension (p : ℕ) (a b : ℤ) 
  (hp : Nat.Prime p) (hab : a ≡ b [ZMOD p]) : 
  a^p ≡ b^p [ZMOD p^2] := by
  sorry

end NUMINAMATH_CALUDE_fermat_little_theorem_extension_l3309_330909


namespace NUMINAMATH_CALUDE_paint_cans_used_l3309_330971

/-- Given:
  - Paul originally had enough paint for 50 rooms.
  - He lost 5 cans of paint.
  - After losing the paint, he had enough for 40 rooms.
Prove that the number of cans of paint used for 40 rooms is 20. -/
theorem paint_cans_used (original_rooms : ℕ) (lost_cans : ℕ) (remaining_rooms : ℕ)
  (h1 : original_rooms = 50)
  (h2 : lost_cans = 5)
  (h3 : remaining_rooms = 40) :
  (remaining_rooms : ℚ) / ((original_rooms - remaining_rooms : ℕ) / lost_cans : ℚ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_paint_cans_used_l3309_330971


namespace NUMINAMATH_CALUDE_cake_mix_tray_difference_l3309_330942

theorem cake_mix_tray_difference :
  ∀ (tray1_capacity tray2_capacity : ℕ),
    tray1_capacity + tray2_capacity = 500 →
    tray2_capacity = 240 →
    tray1_capacity > tray2_capacity →
    tray1_capacity - tray2_capacity = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_cake_mix_tray_difference_l3309_330942


namespace NUMINAMATH_CALUDE_gcd_problem_l3309_330938

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 1116 * k) :
  Int.gcd (b^2 + 11*b + 36) (b + 6) = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l3309_330938


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3309_330967

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x + 1| + |x - 4| ≥ 7} = Set.Ici 5 ∪ Set.Iic (-2) := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3309_330967


namespace NUMINAMATH_CALUDE_total_height_climbed_l3309_330964

/-- The number of staircases John climbs -/
def num_staircases : ℕ := 3

/-- The number of steps in the first staircase -/
def first_staircase : ℕ := 20

/-- The number of steps in the second staircase -/
def second_staircase : ℕ := 2 * first_staircase

/-- The number of steps in the third staircase -/
def third_staircase : ℕ := second_staircase - 10

/-- The height of each step in feet -/
def step_height : ℚ := 1/2

/-- The total number of steps climbed -/
def total_steps : ℕ := first_staircase + second_staircase + third_staircase

/-- The total height climbed in feet -/
def total_feet : ℚ := (total_steps : ℚ) * step_height

theorem total_height_climbed : total_feet = 45 := by
  sorry

end NUMINAMATH_CALUDE_total_height_climbed_l3309_330964


namespace NUMINAMATH_CALUDE_linear_function_unique_l3309_330952

/-- A function f: ℝ → ℝ is increasing if for all x, y ∈ ℝ, x < y implies f(x) < f(y) -/
def Increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem linear_function_unique
  (f : ℝ → ℝ)
  (h1 : ∀ x, f (f x) = 4 * x + 6)
  (h2 : Increasing f) :
  ∀ x, f x = 2 * x + 2 :=
sorry

end NUMINAMATH_CALUDE_linear_function_unique_l3309_330952


namespace NUMINAMATH_CALUDE_congruence_mod_nine_l3309_330961

theorem congruence_mod_nine : ∃! n : ℤ, 0 ≤ n ∧ n < 9 ∧ -2222 ≡ n [ZMOD 9] := by sorry

end NUMINAMATH_CALUDE_congruence_mod_nine_l3309_330961


namespace NUMINAMATH_CALUDE_art_class_price_l3309_330941

/-- Represents the price of Claudia's one-hour art class -/
def class_price : ℝ := 10

/-- Number of kids attending Saturday's class -/
def saturday_attendance : ℕ := 20

/-- Number of kids attending Sunday's class -/
def sunday_attendance : ℕ := saturday_attendance / 2

/-- Total earnings for both days -/
def total_earnings : ℝ := 300

theorem art_class_price :
  class_price * (saturday_attendance + sunday_attendance) = total_earnings :=
sorry

end NUMINAMATH_CALUDE_art_class_price_l3309_330941


namespace NUMINAMATH_CALUDE_sphere_intersection_l3309_330988

/-- Sphere intersection problem -/
theorem sphere_intersection (center : ℝ × ℝ × ℝ) (R : ℝ) :
  let (x₀, y₀, z₀) := center
  -- Conditions
  (x₀ = 3 ∧ y₀ = -2 ∧ z₀ = 5) →  -- Sphere center
  (R^2 = 29) →  -- Sphere radius
  -- xy-plane intersection
  ((3 - x₀)^2 + (-2 - y₀)^2 = 2^2) →
  -- yz-plane intersection
  ((0 - x₀)^2 + (5 - z₀)^2 = 3^2) →
  -- xz-plane intersection
  (∃ (x z : ℝ), (x - x₀)^2 + (z - z₀)^2 = 8 ∧ z = -x + 3) →
  -- Conclusion
  (3^2 = 3^2 ∧ 8 = (2 * Real.sqrt 2)^2) :=
by sorry

end NUMINAMATH_CALUDE_sphere_intersection_l3309_330988


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_l3309_330972

theorem gcd_lcm_sum : Nat.gcd 54 72 + Nat.lcm 50 15 = 168 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_l3309_330972


namespace NUMINAMATH_CALUDE_well_depth_is_896_l3309_330922

/-- The depth of the well in feet -/
def well_depth : ℝ := 896

/-- The initial velocity of the stone in feet per second (downward) -/
def initial_velocity : ℝ := 32

/-- The total time until the sound is heard in seconds -/
def total_time : ℝ := 8.5

/-- The velocity of sound in feet per second -/
def sound_velocity : ℝ := 1120

/-- The stone's displacement function in feet, given time t in seconds -/
def stone_displacement (t : ℝ) : ℝ := 32 * t + 16 * t^2

/-- Theorem stating that the well depth is 896 feet given the conditions -/
theorem well_depth_is_896 :
  ∃ (t₁ : ℝ), 
    stone_displacement t₁ = well_depth ∧
    t₁ + well_depth / sound_velocity = total_time :=
by sorry

end NUMINAMATH_CALUDE_well_depth_is_896_l3309_330922


namespace NUMINAMATH_CALUDE_candy_store_revenue_calculation_l3309_330928

/-- Calculates the total revenue of a candy store based on their sales of fudge, chocolate truffles, and chocolate-covered pretzels. -/
def candy_store_revenue (fudge_pounds : ℝ) (fudge_price : ℝ) 
                        (truffle_dozens : ℝ) (truffle_price : ℝ)
                        (pretzel_dozens : ℝ) (pretzel_price : ℝ) : ℝ :=
  fudge_pounds * fudge_price +
  truffle_dozens * 12 * truffle_price +
  pretzel_dozens * 12 * pretzel_price

/-- The candy store's revenue from selling fudge, chocolate truffles, and chocolate-covered pretzels is $212.00. -/
theorem candy_store_revenue_calculation :
  candy_store_revenue 20 2.50 5 1.50 3 2.00 = 212.00 := by
  sorry

end NUMINAMATH_CALUDE_candy_store_revenue_calculation_l3309_330928


namespace NUMINAMATH_CALUDE_jimmy_has_more_sheets_l3309_330965

/-- Given the initial number of sheets and additional sheets received,
    calculate the difference between Jimmy's and Tommy's final sheet counts. -/
def sheet_difference (jimmy_initial : ℕ) (tommy_more_initial : ℕ) 
  (jimmy_additional1 : ℕ) (jimmy_additional2 : ℕ)
  (tommy_additional1 : ℕ) (tommy_additional2 : ℕ) : ℕ :=
  let tommy_initial := jimmy_initial + tommy_more_initial
  let jimmy_final := jimmy_initial + jimmy_additional1 + jimmy_additional2
  let tommy_final := tommy_initial + tommy_additional1 + tommy_additional2
  jimmy_final - tommy_final

/-- Theorem stating that Jimmy will have 58 more sheets than Tommy
    after receiving additional sheets. -/
theorem jimmy_has_more_sheets :
  sheet_difference 58 25 85 47 30 19 = 58 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_has_more_sheets_l3309_330965


namespace NUMINAMATH_CALUDE_shift_function_unit_shift_l3309_330947

/-- A function satisfying specific inequalities for shifts of 24 and 77 -/
def ShiftFunction (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (x + 24) ≤ f x + 24) ∧ (∀ x : ℝ, f (x + 77) ≥ f x + 77)

/-- Theorem stating that a ShiftFunction satisfies f(x+1) = f(x)+1 for all real x -/
theorem shift_function_unit_shift (f : ℝ → ℝ) (hf : ShiftFunction f) :
  ∀ x : ℝ, f (x + 1) = f x + 1 := by
  sorry

end NUMINAMATH_CALUDE_shift_function_unit_shift_l3309_330947


namespace NUMINAMATH_CALUDE_apple_problem_l3309_330900

theorem apple_problem (initial_apples : ℕ) (sold_to_jill : ℚ) (sold_to_june : ℚ) (sold_to_jeff : ℚ) (donated_to_school : ℚ) :
  initial_apples = 150 →
  sold_to_jill = 20 / 100 →
  sold_to_june = 30 / 100 →
  sold_to_jeff = 10 / 100 →
  donated_to_school = 5 / 100 →
  let remaining_after_jill := initial_apples - ⌊initial_apples * sold_to_jill⌋
  let remaining_after_june := remaining_after_jill - ⌊remaining_after_jill * sold_to_june⌋
  let remaining_after_jeff := remaining_after_june - ⌊remaining_after_june * sold_to_jeff⌋
  let final_remaining := remaining_after_jeff - ⌈remaining_after_jeff * donated_to_school⌉
  final_remaining = 72 := by
    sorry

end NUMINAMATH_CALUDE_apple_problem_l3309_330900


namespace NUMINAMATH_CALUDE_system_solution_range_l3309_330956

theorem system_solution_range (x y z : ℝ) (a : ℝ) :
  (3 * x^2 + 2 * y^2 + 2 * z^2 = a ∧ 
   4 * x^2 + 4 * y^2 + 5 * z^2 = 1 - a) →
  (2/7 : ℝ) ≤ a ∧ a ≤ (3/7 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_system_solution_range_l3309_330956


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_subtraction_for_509_divisible_by_9_least_subtraction_509_divisible_by_9_l3309_330998

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (k : ℕ), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : ℕ), m < k → (n - m) % d ≠ 0 :=
by
  sorry

theorem subtraction_for_509_divisible_by_9 :
  ∃ (k : ℕ), k < 9 ∧ (509 - k) % 9 = 0 ∧ ∀ (m : ℕ), m < k → (509 - m) % 9 ≠ 0 :=
by
  sorry

#eval (509 - 5) % 9  -- Should output 0

theorem least_subtraction_509_divisible_by_9 :
  5 < 9 ∧ (509 - 5) % 9 = 0 ∧ ∀ (m : ℕ), m < 5 → (509 - m) % 9 ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_subtraction_for_509_divisible_by_9_least_subtraction_509_divisible_by_9_l3309_330998


namespace NUMINAMATH_CALUDE_number_multiplication_l3309_330980

theorem number_multiplication (x : ℤ) : 50 = x + 26 → 9 * x = 216 := by
  sorry

end NUMINAMATH_CALUDE_number_multiplication_l3309_330980


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3309_330975

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > -1) (hab : a + b = 1) :
  1 / a + 1 / (b + 1) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3309_330975


namespace NUMINAMATH_CALUDE_fraction_equality_l3309_330915

theorem fraction_equality : (1-2+4-8+16-32+64)/(2-4+8-16+32-64+128) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3309_330915


namespace NUMINAMATH_CALUDE_bathroom_width_l3309_330974

/-- Proves that the width of Mrs. Garvey's bathroom is 6 feet -/
theorem bathroom_width : 
  ∀ (length width : ℝ) (tile_side : ℝ) (num_tiles : ℕ),
  length = 10 →
  tile_side = 0.5 →
  num_tiles = 240 →
  width * length = (tile_side^2 * num_tiles) →
  width = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_bathroom_width_l3309_330974


namespace NUMINAMATH_CALUDE_cubic_roots_inequality_l3309_330912

theorem cubic_roots_inequality (a b c r s t : ℝ) :
  (∀ x, x^3 + a*x^2 + b*x + c = 0 ↔ x = r ∨ x = s ∨ x = t) →
  r ≥ s →
  s ≥ t →
  (a^2 - 3*b ≥ 0) ∧ (Real.sqrt (a^2 - 3*b) ≤ r - t) := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_inequality_l3309_330912


namespace NUMINAMATH_CALUDE_sam_spent_93_pennies_l3309_330962

/-- The number of pennies Sam spent -/
def pennies_spent (initial : ℕ) (remaining : ℕ) : ℕ := initial - remaining

/-- Proof that Sam spent 93 pennies -/
theorem sam_spent_93_pennies (initial : ℕ) (remaining : ℕ) 
  (h1 : initial = 98) (h2 : remaining = 5) : 
  pennies_spent initial remaining = 93 := by
  sorry

end NUMINAMATH_CALUDE_sam_spent_93_pennies_l3309_330962


namespace NUMINAMATH_CALUDE_inequality_proof_l3309_330943

theorem inequality_proof (a b : ℝ) (m : ℤ) (ha : a > 0) (hb : b > 0) :
  (1 + a / b) ^ m + (1 + b / a) ^ m ≥ 2^(m + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3309_330943


namespace NUMINAMATH_CALUDE_max_savings_63_l3309_330955

/-- Represents the price of a pastry package -/
structure PastryPrice where
  quantity : Nat
  price : Nat

/-- Represents the discount options for a type of pastry -/
structure PastryDiscount where
  regular_price : Nat
  discounts : List PastryPrice

/-- Calculates the minimum cost for a given quantity using available discounts -/
def min_cost (discount : PastryDiscount) (quantity : Nat) : Nat :=
  sorry

/-- Calculates the cost without any discounts -/
def regular_cost (discount : PastryDiscount) (quantity : Nat) : Nat :=
  sorry

/-- Doughnut discount options -/
def doughnut_discount : PastryDiscount :=
  { regular_price := 8,
    discounts := [
      { quantity := 12, price := 8 },
      { quantity := 24, price := 14 },
      { quantity := 48, price := 26 }
    ] }

/-- Croissant discount options -/
def croissant_discount : PastryDiscount :=
  { regular_price := 10,
    discounts := [
      { quantity := 12, price := 10 },
      { quantity := 36, price := 28 },
      { quantity := 60, price := 45 }
    ] }

/-- Muffin discount options -/
def muffin_discount : PastryDiscount :=
  { regular_price := 6,
    discounts := [
      { quantity := 12, price := 6 },
      { quantity := 24, price := 11 },
      { quantity := 72, price := 30 }
    ] }

theorem max_savings_63 :
  let doughnut_qty := 20 * 12
  let croissant_qty := 15 * 12
  let muffin_qty := 18 * 12
  let total_discounted := min_cost doughnut_discount doughnut_qty +
                          min_cost croissant_discount croissant_qty +
                          min_cost muffin_discount muffin_qty
  let total_regular := regular_cost doughnut_discount doughnut_qty +
                       regular_cost croissant_discount croissant_qty +
                       regular_cost muffin_discount muffin_qty
  total_regular - total_discounted = 63 :=
by sorry

end NUMINAMATH_CALUDE_max_savings_63_l3309_330955


namespace NUMINAMATH_CALUDE_total_players_l3309_330981

theorem total_players (outdoor : ℕ) (indoor : ℕ) (both : ℕ)
  (h1 : outdoor = 350)
  (h2 : indoor = 110)
  (h3 : both = 60) :
  outdoor + indoor - both = 400 := by
  sorry

end NUMINAMATH_CALUDE_total_players_l3309_330981


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3309_330937

theorem complex_equation_solution (z : ℂ) :
  (2 - Complex.I) * z = Complex.I ^ 2022 →
  z = -2/5 - (1/5) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3309_330937


namespace NUMINAMATH_CALUDE_min_value_theorem_l3309_330963

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_sum : 2*a + b = 1) :
  (2/a + 1/b) ≥ 9 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3309_330963
