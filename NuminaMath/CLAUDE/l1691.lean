import Mathlib

namespace NUMINAMATH_CALUDE_irregular_hexagon_perimeter_l1691_169131

/-- An irregular hexagon with specific angle measurements and equal side lengths -/
structure IrregularHexagon where
  -- Side length of the hexagon
  side_length : ℝ
  -- Assumption that all sides are equal
  all_sides_equal : True
  -- Three nonadjacent angles measure 120°
  three_angles_120 : True
  -- The other three angles measure 60°
  three_angles_60 : True
  -- The enclosed area of the hexagon
  area : ℝ
  -- The area is 24
  area_is_24 : area = 24

/-- The perimeter of an irregular hexagon with the given conditions -/
def perimeter (h : IrregularHexagon) : ℝ := 6 * h.side_length

/-- Theorem stating that the perimeter of the irregular hexagon is 24 / (3^(1/4)) -/
theorem irregular_hexagon_perimeter (h : IrregularHexagon) : 
  perimeter h = 24 / Real.rpow 3 (1/4) := by
  sorry

end NUMINAMATH_CALUDE_irregular_hexagon_perimeter_l1691_169131


namespace NUMINAMATH_CALUDE_businessmen_neither_coffee_nor_tea_l1691_169112

theorem businessmen_neither_coffee_nor_tea 
  (total : ℕ) 
  (coffee : ℕ) 
  (tea : ℕ) 
  (both : ℕ) 
  (h1 : total = 25) 
  (h2 : coffee = 12) 
  (h3 : tea = 10) 
  (h4 : both = 5) : 
  total - (coffee + tea - both) = 8 := by
  sorry

end NUMINAMATH_CALUDE_businessmen_neither_coffee_nor_tea_l1691_169112


namespace NUMINAMATH_CALUDE_parabola_focus_theorem_l1691_169140

-- Define the line on which the focus lies
def focus_line (x y : ℝ) : Prop := x + 2 * y + 3 = 0

-- Define the two possible standard equations for the parabola
def parabola_eq1 (x y : ℝ) : Prop := y^2 = -12 * x
def parabola_eq2 (x y : ℝ) : Prop := x^2 = -6 * y

-- Theorem statement
theorem parabola_focus_theorem :
  ∀ (x y : ℝ), focus_line x y →
  (parabola_eq1 x y ∨ parabola_eq2 x y) :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_theorem_l1691_169140


namespace NUMINAMATH_CALUDE_sum_of_P_roots_l1691_169196

variable (a b c d : ℂ)

def P (X : ℂ) : ℂ := X^6 - X^5 - X^4 - X^3 - X

theorem sum_of_P_roots :
  (a^4 - a^3 - a^2 - 1 = 0) →
  (b^4 - b^3 - b^2 - 1 = 0) →
  (c^4 - c^3 - c^2 - 1 = 0) →
  (d^4 - d^3 - d^2 - 1 = 0) →
  P a + P b + P c + P d = -2 := by sorry

end NUMINAMATH_CALUDE_sum_of_P_roots_l1691_169196


namespace NUMINAMATH_CALUDE_dinner_arrangement_count_l1691_169195

def number_of_friends : ℕ := 5
def number_of_cooks : ℕ := 2

theorem dinner_arrangement_count :
  Nat.choose number_of_friends number_of_cooks = 10 := by
  sorry

end NUMINAMATH_CALUDE_dinner_arrangement_count_l1691_169195


namespace NUMINAMATH_CALUDE_square_of_fourth_power_zero_l1691_169107

theorem square_of_fourth_power_zero (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : A ^ 4 = 0) : A ^ 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_of_fourth_power_zero_l1691_169107


namespace NUMINAMATH_CALUDE_fraction_evaluation_l1691_169193

theorem fraction_evaluation : 
  (1 / 5 + 1 / 3) / (3 / 7 - 1 / 4) = 224 / 75 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l1691_169193


namespace NUMINAMATH_CALUDE_definite_integral_equals_ln3_minus_ln2_plus_1_l1691_169164

theorem definite_integral_equals_ln3_minus_ln2_plus_1 :
  let a : ℝ := 2 * Real.arctan (1 / 3)
  let b : ℝ := 2 * Real.arctan (1 / 2)
  let f (x : ℝ) := 1 / (Real.sin x * (1 - Real.sin x))
  ∫ x in a..b, f x = Real.log 3 - Real.log 2 + 1 := by sorry

end NUMINAMATH_CALUDE_definite_integral_equals_ln3_minus_ln2_plus_1_l1691_169164


namespace NUMINAMATH_CALUDE_expression_simplification_l1691_169121

theorem expression_simplification :
  (((1 + 2 + 3) * 2)^2 / 3) + ((3 * 4 + 6 + 2) / 5) = 52 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1691_169121


namespace NUMINAMATH_CALUDE_cousins_ages_sum_l1691_169136

def is_single_digit (n : ℕ) : Prop := 0 < n ∧ n < 10

theorem cousins_ages_sum :
  ∀ (a b c d : ℕ),
    is_single_digit a ∧ is_single_digit b ∧ is_single_digit c ∧ is_single_digit d →
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
    ((a * b = 24 ∧ c * d = 30) ∨ (a * c = 24 ∧ b * d = 30) ∨ (a * d = 24 ∧ b * c = 30)) →
    a + b + c + d = 22 :=
by sorry

end NUMINAMATH_CALUDE_cousins_ages_sum_l1691_169136


namespace NUMINAMATH_CALUDE_b_work_time_l1691_169187

/-- Represents the time taken by A, B, and C to complete the work individually --/
structure WorkTime where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The conditions of the problem --/
def work_conditions (t : WorkTime) : Prop :=
  t.a = 2 * t.b ∧ 
  t.a = 3 * t.c ∧ 
  1 / t.a + 1 / t.b + 1 / t.c = 1 / 6

/-- The theorem stating that B takes 18 days to complete the work alone --/
theorem b_work_time (t : WorkTime) : work_conditions t → t.b = 18 := by
  sorry

end NUMINAMATH_CALUDE_b_work_time_l1691_169187


namespace NUMINAMATH_CALUDE_no_solution_for_inequality_l1691_169194

theorem no_solution_for_inequality (a b : ℝ) (h : |a - b| > 2) :
  ¬∃ x : ℝ, |x - a| + |x - b| ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_for_inequality_l1691_169194


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_squares_l1691_169110

/-- Given that x₁ and x₂ are real roots of a quadratic equation, this theorem proves
    properties about y = x₁² + x₂² as a function of m. -/
theorem quadratic_roots_sum_squares (m : ℝ) (x₁ x₂ : ℝ) :
  x₁^2 - 2*(m-1)*x₁ + m + 1 = 0 →
  x₂^2 - 2*(m-1)*x₂ + m + 1 = 0 →
  let y := x₁^2 + x₂^2
  -- 1. y as a function of m
  y = 4*m^2 - 10*m + 2 ∧
  -- 2. Minimum value of y
  (∃ (m₀ : ℝ), y = 6 ∧ ∀ (m' : ℝ), y ≥ 6) ∧
  -- 3. y ≥ 6 for all valid m
  y ≥ 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_squares_l1691_169110


namespace NUMINAMATH_CALUDE_max_value_product_sum_l1691_169152

theorem max_value_product_sum (A M C : ℕ) (h : A + M + C = 15) :
  (∀ a m c : ℕ, a + m + c = 15 →
    A * M * C + A * M + M * C + C * A + A + M + C ≥
    a * m * c + a * m + m * c + c * a + a + m + c) →
  A * M * C + A * M + M * C + C * A + A + M + C = 215 :=
sorry

end NUMINAMATH_CALUDE_max_value_product_sum_l1691_169152


namespace NUMINAMATH_CALUDE_region_upper_left_l1691_169176

def line (x y : ℝ) : ℝ := 3 * x - 2 * y - 6

theorem region_upper_left :
  ∀ (x y : ℝ), line x y < 0 →
  ∃ (x' y' : ℝ), x' > x ∧ y' < y ∧ line x' y' = 0 :=
by sorry

end NUMINAMATH_CALUDE_region_upper_left_l1691_169176


namespace NUMINAMATH_CALUDE_max_added_value_l1691_169147

/-- The added value function for the car manufacturer's production line renovation --/
def f (a : ℝ) (x : ℝ) : ℝ := 8 * (a - x) * x^2

/-- The theorem stating the maximum value of the added value function --/
theorem max_added_value (a : ℝ) (h_a : a > 0) :
  ∃ (x : ℝ), x ∈ Set.Ioo 0 (4*a/5) ∧ 
    (∀ (y : ℝ), y ∈ Set.Ioo 0 (4*a/5) → f a y ≤ f a x) ∧
    f a x = 32 * a^3 / 27 ∧
    x = 2*a/3 := by
  sorry

end NUMINAMATH_CALUDE_max_added_value_l1691_169147


namespace NUMINAMATH_CALUDE_complicated_expression_equality_l1691_169142

theorem complicated_expression_equality : 
  Real.sqrt (11 * 13) * (1/3) + 2 * (Real.sqrt 17 / 3) - 4 * (Real.sqrt 7 / 5) = 
  (5 * Real.sqrt 143 + 10 * Real.sqrt 17 - 12 * Real.sqrt 7) / 15 := by
sorry

end NUMINAMATH_CALUDE_complicated_expression_equality_l1691_169142


namespace NUMINAMATH_CALUDE_tempo_original_value_l1691_169114

/-- The original value of a tempo given its insured value and insurance extent --/
theorem tempo_original_value 
  (insured_value : ℝ) 
  (insurance_extent : ℝ) 
  (h1 : insured_value = 70000) 
  (h2 : insurance_extent = 4/5) : 
  ∃ (original_value : ℝ), 
    original_value = 87500 ∧ 
    insured_value = insurance_extent * original_value :=
by
  sorry

#check tempo_original_value

end NUMINAMATH_CALUDE_tempo_original_value_l1691_169114


namespace NUMINAMATH_CALUDE_sin_sum_of_complex_exponentials_l1691_169162

theorem sin_sum_of_complex_exponentials (γ δ : ℝ) :
  Complex.exp (Complex.I * γ) = (4/5 : ℂ) + (3/5 : ℂ) * Complex.I ∧
  Complex.exp (Complex.I * δ) = (-5/13 : ℂ) + (12/13 : ℂ) * Complex.I →
  Real.sin (γ + δ) = 21/65 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_of_complex_exponentials_l1691_169162


namespace NUMINAMATH_CALUDE_circular_seating_arrangement_l1691_169124

/-- Given a circular arrangement of students where the 5th position
    is opposite the 20th position, prove that there are 32 students in total. -/
theorem circular_seating_arrangement (n : ℕ) 
  (h : n > 0)  -- Ensure positive number of students
  (opposite : ∀ (a b : ℕ), a ≤ n → b ≤ n → (a + n / 2) % n = b % n → a = 5 ∧ b = 20) :
  n = 32 := by
  sorry

end NUMINAMATH_CALUDE_circular_seating_arrangement_l1691_169124


namespace NUMINAMATH_CALUDE_largest_m_for_factorization_l1691_169122

theorem largest_m_for_factorization : 
  ∀ m : ℤ, (∃ a b c d : ℤ, 5 * x^2 + m * x + 120 = (a * x + b) * (c * x + d)) → m ≤ 601 :=
by sorry

end NUMINAMATH_CALUDE_largest_m_for_factorization_l1691_169122


namespace NUMINAMATH_CALUDE_cone_prism_volume_ratio_l1691_169172

/-- The ratio of the volume of a right circular cone inscribed in a right rectangular prism -/
theorem cone_prism_volume_ratio (r h : ℝ) (h1 : r > 0) (h2 : h > 0) : 
  (1 / 3 * π * r^2 * h) / (6 * r^2 * h) = π / 18 := by
  sorry

#check cone_prism_volume_ratio

end NUMINAMATH_CALUDE_cone_prism_volume_ratio_l1691_169172


namespace NUMINAMATH_CALUDE_symmetric_points_product_l1691_169192

/-- 
If point A (2008, y) and point B (x, -1) are symmetric about the origin,
then xy = -2008.
-/
theorem symmetric_points_product (x y : ℝ) : 
  (2008 = -x ∧ y = 1) → x * y = -2008 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_product_l1691_169192


namespace NUMINAMATH_CALUDE_quadratic_equation_with_given_roots_l1691_169166

theorem quadratic_equation_with_given_roots (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c = 0 ↔ x = 7 ∨ x = -1) →
  b = -6 ∧ c = -7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_with_given_roots_l1691_169166


namespace NUMINAMATH_CALUDE_school_students_l1691_169134

/-- Prove that the total number of students in a school is 1000, given the conditions described. -/
theorem school_students (S : ℕ) : 
  (S / 2) / 2 = 250 → S = 1000 := by
  sorry

end NUMINAMATH_CALUDE_school_students_l1691_169134


namespace NUMINAMATH_CALUDE_part_to_whole_ratio_l1691_169197

theorem part_to_whole_ratio (N : ℝ) (x : ℝ) (h1 : N = 280) (h2 : x + 7 = (N / 4) - 7) : 
  x / N = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_part_to_whole_ratio_l1691_169197


namespace NUMINAMATH_CALUDE_triangle_angle_theorem_l1691_169104

-- Define a triangle structure
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h₃ : ℝ
  angleA : ℝ
  angleC : ℝ

-- Define the theorem
theorem triangle_angle_theorem (t : Triangle) 
  (h : 1 / t.h₃^2 = 1 / t.a^2 + 1 / t.b^2) : 
  t.angleC = 90 ∨ |t.angleA - t.angleC| = 90 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_theorem_l1691_169104


namespace NUMINAMATH_CALUDE_coin_denomination_problem_l1691_169120

/-- Given a total of 334 coins, with 250 coins of 20 paise each, and a total sum of 7100 paise,
    the denomination of the remaining coins is 25 paise. -/
theorem coin_denomination_problem (total_coins : ℕ) (twenty_paise_coins : ℕ) (total_sum : ℕ) :
  total_coins = 334 →
  twenty_paise_coins = 250 →
  total_sum = 7100 →
  (total_coins - twenty_paise_coins) * (total_sum - twenty_paise_coins * 20) / (total_coins - twenty_paise_coins) = 25 := by
  sorry

#eval (334 - 250) * (7100 - 250 * 20) / (334 - 250)  -- Should output 25

end NUMINAMATH_CALUDE_coin_denomination_problem_l1691_169120


namespace NUMINAMATH_CALUDE_inequality_solution_l1691_169128

theorem inequality_solution (x : ℝ) : 
  (2 * x) / (x - 2) + (x - 3) / (3 * x) ≥ 2 ↔ 
  (0 < x ∧ x ≤ 5/6) ∨ (2 < x) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1691_169128


namespace NUMINAMATH_CALUDE_percentage_gain_calculation_l1691_169125

/-- Calculates the percentage gain when selling an article --/
theorem percentage_gain_calculation (cost_price selling_price : ℚ) : 
  cost_price = 160 → 
  selling_price = 192 → 
  (selling_price - cost_price) / cost_price * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_gain_calculation_l1691_169125


namespace NUMINAMATH_CALUDE_complete_factorization_l1691_169151

theorem complete_factorization (x : ℝ) :
  x^12 - 729 = (x^2 + 3) * (x^4 - 3*x^2 + 9) * (x^3 - 3) * (x^3 + 3) := by
  sorry

end NUMINAMATH_CALUDE_complete_factorization_l1691_169151


namespace NUMINAMATH_CALUDE_f_negative_two_equals_negative_eight_l1691_169130

noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then 3^x - 1 else -3^(-x) + 1

theorem f_negative_two_equals_negative_eight :
  f (-2) = -8 :=
by sorry

end NUMINAMATH_CALUDE_f_negative_two_equals_negative_eight_l1691_169130


namespace NUMINAMATH_CALUDE_fixed_amount_more_economical_l1691_169184

theorem fixed_amount_more_economical (p₁ p₂ : ℝ) (h₁ : p₁ > 0) (h₂ : p₂ > 0) :
  2 / (1 / p₁ + 1 / p₂) ≤ (p₁ + p₂) / 2 := by
  sorry

#check fixed_amount_more_economical

end NUMINAMATH_CALUDE_fixed_amount_more_economical_l1691_169184


namespace NUMINAMATH_CALUDE_base_n_representation_of_b_l1691_169189

theorem base_n_representation_of_b (n a b : ℕ) : 
  n > 9 → 
  n^2 - a*n + b = 0 → 
  a = 2*n + 1 → 
  b = n^2 + n := by
sorry

end NUMINAMATH_CALUDE_base_n_representation_of_b_l1691_169189


namespace NUMINAMATH_CALUDE_career_preference_graph_degrees_l1691_169126

theorem career_preference_graph_degrees 
  (total_students : ℕ) 
  (male_ratio female_ratio : ℚ) 
  (male_preference female_preference : ℚ) :
  male_ratio / (male_ratio + female_ratio) = 2 / 5 →
  female_ratio / (male_ratio + female_ratio) = 3 / 5 →
  male_preference = 1 / 4 →
  female_preference = 1 / 2 →
  (male_ratio * male_preference + female_ratio * female_preference) / (male_ratio + female_ratio) * 360 = 144 := by
  sorry

#check career_preference_graph_degrees

end NUMINAMATH_CALUDE_career_preference_graph_degrees_l1691_169126


namespace NUMINAMATH_CALUDE_fixed_distance_vector_l1691_169123

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem fixed_distance_vector (a b : E) :
  ∃ t u : ℝ, ∀ p : E,
    (‖p - b‖ = 3 * ‖p - a‖) →
    (∃ c : ℝ, ∀ q : E, (‖p - b‖ = 3 * ‖p - a‖) → ‖q - (t • a + u • b)‖ = c) →
    t = 9/8 ∧ u = -1/8 :=
by sorry

end NUMINAMATH_CALUDE_fixed_distance_vector_l1691_169123


namespace NUMINAMATH_CALUDE_tangent_line_parallel_points_l1691_169173

def f (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_line_parallel_points :
  ∀ x y : ℝ, f x = y → (3 * x^2 + 1 = 4) ↔ ((x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = -4)) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_parallel_points_l1691_169173


namespace NUMINAMATH_CALUDE_array_sum_proof_l1691_169127

def grid := [[1, 0, 0, 0], [0, 9, 0, 5], [0, 0, 14, 0]]
def available_numbers := [2, 3, 4, 7, 10, 11, 12, 13, 15]

theorem array_sum_proof :
  ∃ (arrangement : List (List Nat)),
    (∀ row ∈ arrangement, row.sum = 32) ∧
    (∀ col ∈ arrangement.transpose, col.sum = 32) ∧
    (arrangement.join.toFinset = (available_numbers.toFinset \ {10}) ∪ grid.join.toFinset) :=
  by sorry

end NUMINAMATH_CALUDE_array_sum_proof_l1691_169127


namespace NUMINAMATH_CALUDE_angle_in_fourth_quadrant_l1691_169106

theorem angle_in_fourth_quadrant (α : Real) :
  (0 < α) ∧ (α < π / 2) → (3 * π / 2 < (2 * π - α)) ∧ ((2 * π - α) < 2 * π) := by
  sorry

end NUMINAMATH_CALUDE_angle_in_fourth_quadrant_l1691_169106


namespace NUMINAMATH_CALUDE_corveus_weekly_lack_of_sleep_l1691_169153

/-- Calculates the weekly lack of sleep given actual and recommended daily sleep hours -/
def weeklyLackOfSleep (actualSleep recommendedSleep : ℕ) : ℕ :=
  (recommendedSleep - actualSleep) * 7

/-- Proves that Corveus lacks 14 hours of sleep in a week -/
theorem corveus_weekly_lack_of_sleep :
  weeklyLackOfSleep 4 6 = 14 := by
  sorry

end NUMINAMATH_CALUDE_corveus_weekly_lack_of_sleep_l1691_169153


namespace NUMINAMATH_CALUDE_circle_distance_inequality_l1691_169133

theorem circle_distance_inequality (x y : ℝ) : 
  x^2 + y^2 + 2*x - 6*y = 6 → (x - 1)^2 + (y - 2)^2 ≠ 2 := by
sorry

end NUMINAMATH_CALUDE_circle_distance_inequality_l1691_169133


namespace NUMINAMATH_CALUDE_expression_value_l1691_169190

/-- Given x, y, and z as defined, prove that the expression equals 20 -/
theorem expression_value (x y z : ℝ) 
  (hx : x = -Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5)
  (hy : y = Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 5)
  (hz : z = Real.sqrt 2 + Real.sqrt 3 - Real.sqrt 5) :
  (x^4 / ((x-y)*(x-z))) + (y^4 / ((y-z)*(y-x))) + (z^4 / ((z-x)*(z-y))) = 20 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1691_169190


namespace NUMINAMATH_CALUDE_boat_license_count_l1691_169132

/-- The number of possible letters for a boat license -/
def num_letters : Nat := 3

/-- The number of possible digits for each position in a boat license -/
def num_digits : Nat := 10

/-- The number of digit positions in a boat license -/
def num_positions : Nat := 5

/-- The total number of possible boat licenses -/
def total_licenses : Nat := num_letters * (num_digits ^ num_positions)

theorem boat_license_count : total_licenses = 300000 := by
  sorry

end NUMINAMATH_CALUDE_boat_license_count_l1691_169132


namespace NUMINAMATH_CALUDE_flight_savings_l1691_169161

/-- Calculates the savings by choosing the cheaper flight between two airlines with given prices and discounts -/
theorem flight_savings (delta_price united_price : ℝ) (delta_discount united_discount : ℝ) :
  delta_price = 850 →
  united_price = 1100 →
  delta_discount = 0.20 →
  united_discount = 0.30 →
  let delta_final := delta_price * (1 - delta_discount)
  let united_final := united_price * (1 - united_discount)
  min delta_final united_final = delta_final →
  united_final - delta_final = 90 := by
sorry


end NUMINAMATH_CALUDE_flight_savings_l1691_169161


namespace NUMINAMATH_CALUDE_parabola_tangent_line_l1691_169169

/-- A parabola is tangent to a line if they intersect at exactly one point. -/
def is_tangent (a : ℝ) : Prop :=
  ∃! x : ℝ, a * x^2 + 10 = 2 * x

/-- The value of a for which the parabola y = ax^2 + 10 is tangent to the line y = 2x -/
theorem parabola_tangent_line : 
  ∃ a : ℝ, is_tangent a ∧ a = (1 : ℝ) / 10 :=
sorry

end NUMINAMATH_CALUDE_parabola_tangent_line_l1691_169169


namespace NUMINAMATH_CALUDE_school_children_count_l1691_169170

theorem school_children_count (total_bananas : ℕ) : 
  (∃ (children : ℕ), 
    total_bananas = 2 * children ∧ 
    total_bananas = 4 * (children - 360)) →
  ∃ (children : ℕ), children = 720 := by
sorry

end NUMINAMATH_CALUDE_school_children_count_l1691_169170


namespace NUMINAMATH_CALUDE_sue_votes_count_l1691_169183

def total_votes : ℕ := 1000
def candidate1_percentage : ℚ := 20 / 100
def candidate2_percentage : ℚ := 45 / 100

theorem sue_votes_count :
  let sue_percentage : ℚ := 1 - (candidate1_percentage + candidate2_percentage)
  (sue_percentage * total_votes : ℚ) = 350 := by sorry

end NUMINAMATH_CALUDE_sue_votes_count_l1691_169183


namespace NUMINAMATH_CALUDE_conic_is_ellipse_l1691_169182

-- Define the equation
def conic_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y-2)^2) + Real.sqrt ((x-6)^2 + (y+4)^2) = 12

-- Define the two fixed points
def focus1 : ℝ × ℝ := (0, 2)
def focus2 : ℝ × ℝ := (6, -4)

-- Theorem stating that the equation describes an ellipse
theorem conic_is_ellipse :
  ∃ (a b : ℝ) (center : ℝ × ℝ),
    a > 0 ∧ b > 0 ∧ a > b ∧
    ∀ (x y : ℝ),
      conic_equation x y ↔
        (x - center.1)^2 / a^2 + (y - center.2)^2 / b^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_conic_is_ellipse_l1691_169182


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1691_169178

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 (a > 0, b > 0) and right focus F(c, 0),
    if point P on the hyperbola satisfies |FM| = 2|FP| where M is the intersection of the circle
    centered at F with radius 2c and the positive y-axis, then the eccentricity of the hyperbola
    is √3 + 1. -/
theorem hyperbola_eccentricity (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  let F : ℝ × ℝ := (c, 0)
  let M : ℝ × ℝ := (0, Real.sqrt 3 * c)
  let P : ℝ × ℝ := (c / 2, Real.sqrt 3 / 2 * c)
  (P.1 ^ 2 / a ^ 2 - P.2 ^ 2 / b ^ 2 = 1) →  -- P is on the hyperbola
  (Real.sqrt ((M.1 - F.1) ^ 2 + (M.2 - F.2) ^ 2) = 2 * Real.sqrt ((P.1 - F.1) ^ 2 + (P.2 - F.2) ^ 2)) →  -- |FM| = 2|FP|
  (c ^ 2 / a ^ 2 - b ^ 2 / a ^ 2 = 1) →  -- Relation between a, b, and c for a hyperbola
  Real.sqrt (c ^ 2 / a ^ 2) = Real.sqrt 3 + 1  -- Eccentricity is √3 + 1
:= by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1691_169178


namespace NUMINAMATH_CALUDE_regular_polygon_area_condition_l1691_169158

/-- A regular polygon with n sides inscribed in a circle of radius 2R has an area of 6R² if and only if n = 12 -/
theorem regular_polygon_area_condition (n : ℕ) (R : ℝ) (h : R > 0) :
  (2 * n * R^2 * Real.sin (2 * Real.pi / n) = 6 * R^2) ↔ n = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_area_condition_l1691_169158


namespace NUMINAMATH_CALUDE_max_good_permutations_l1691_169146

/-- A sequence of points in the plane is "good" if no three points are collinear,
    the polyline is non-self-intersecting, and each triangle formed by three
    consecutive points is oriented counterclockwise. -/
def is_good_sequence (points : List (ℝ × ℝ)) : Prop :=
  sorry

/-- The number of distinct permutations of n points that form a good sequence -/
def num_good_permutations (n : ℕ) : ℕ :=
  sorry

/-- Theorem: For any integer n ≥ 3, the maximum number of distinct permutations
    of n points in the plane that form a "good" sequence is n^2 - 4n + 6. -/
theorem max_good_permutations (n : ℕ) (h : n ≥ 3) :
  num_good_permutations n = n^2 - 4*n + 6 :=
sorry

end NUMINAMATH_CALUDE_max_good_permutations_l1691_169146


namespace NUMINAMATH_CALUDE_race_track_inner_circumference_l1691_169144

/-- Given a circular race track with an outer radius of 140.0563499208679 m and a width of 18 m, 
    the inner circumference is approximately 767.145882893066 m. -/
theorem race_track_inner_circumference :
  let outer_radius : ℝ := 140.0563499208679
  let track_width : ℝ := 18
  let inner_radius : ℝ := outer_radius - track_width
  let inner_circumference : ℝ := 2 * Real.pi * inner_radius
  ∃ ε > 0, abs (inner_circumference - 767.145882893066) < ε :=
by sorry

end NUMINAMATH_CALUDE_race_track_inner_circumference_l1691_169144


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1691_169186

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 and asymptotes y = ±√2x,
    prove that its eccentricity is √3 -/
theorem hyperbola_eccentricity (a b : ℝ) (h : b / a = Real.sqrt 2) :
  let c := Real.sqrt (a^2 + b^2)
  c / a = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1691_169186


namespace NUMINAMATH_CALUDE_triangle_is_equilateral_l1691_169141

theorem triangle_is_equilateral (a b c : ℝ) (A B C : ℝ) 
  (h1 : b^2 + c^2 - a^2 = b*c)
  (h2 : 2 * Real.cos B * Real.sin C = Real.sin A)
  (h3 : A + B + C = π)
  (h4 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h5 : 0 < A ∧ 0 < B ∧ 0 < C)
  (h6 : A < π ∧ B < π ∧ C < π) :
  a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_triangle_is_equilateral_l1691_169141


namespace NUMINAMATH_CALUDE_negation_of_universal_quantifier_negation_of_inequality_negation_of_proposition_l1691_169137

theorem negation_of_universal_quantifier (P : ℝ → Prop) :
  (¬ ∀ x ∈ Set.Ioo 0 1, P x) ↔ (∃ x ∈ Set.Ioo 0 1, ¬ P x) := by sorry

theorem negation_of_inequality (x : ℝ) : ¬(x^2 - x < 0) ↔ x^2 - x ≥ 0 := by sorry

theorem negation_of_proposition :
  (¬ ∀ x ∈ Set.Ioo 0 1, x^2 - x < 0) ↔ (∃ x ∈ Set.Ioo 0 1, x^2 - x ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_quantifier_negation_of_inequality_negation_of_proposition_l1691_169137


namespace NUMINAMATH_CALUDE_lemonade_problem_l1691_169159

/-- Given a total number of lemons and the number of lemons needed per glass,
    calculate the number of glasses of lemonade that can be made. -/
def lemonade_glasses (total_lemons : ℕ) (lemons_per_glass : ℕ) : ℕ :=
  total_lemons / lemons_per_glass

/-- Theorem: With 18 lemons and 2 lemons needed per glass,
    9 glasses of lemonade can be made. -/
theorem lemonade_problem : lemonade_glasses 18 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_problem_l1691_169159


namespace NUMINAMATH_CALUDE_table_tennis_probabilities_l1691_169191

-- Define the probability of scoring on a serve
def p_score : ℝ := 0.6

-- Define events
def A_i (i : Nat) : ℝ := 
  if i = 0 then (1 - p_score)^2
  else if i = 1 then 2 * p_score * (1 - p_score)
  else p_score^2

def B_i (i : Nat) : ℝ := 
  if i = 0 then p_score^2
  else if i = 1 then 2 * (1 - p_score) * p_score
  else (1 - p_score)^2

def A : ℝ := 1 - p_score

-- Define the probabilities we want to prove
def p_B : ℝ := A_i 0 * A + A_i 1 * (1 - A)
def p_C : ℝ := A_i 1 * B_i 2 + A_i 2 * B_i 1 + A_i 2 * B_i 2

theorem table_tennis_probabilities : 
  p_B = 0.352 ∧ p_C = 0.3072 := by sorry

end NUMINAMATH_CALUDE_table_tennis_probabilities_l1691_169191


namespace NUMINAMATH_CALUDE_negation_equivalence_l1691_169135

theorem negation_equivalence :
  (¬ (∃ x : ℝ, x > 0 ∧ x^2 - 2*x - 3 > 0)) ↔ 
  (∀ x : ℝ, x > 0 → x^2 - 2*x - 3 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1691_169135


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l1691_169180

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person

-- Define the set of cards
inductive Card : Type
| Red : Card
| Blue : Card
| White : Card

-- Define a distribution of cards to people
def Distribution := Person → Card

-- Define the event "Person A receives the white card"
def event_A_white (d : Distribution) : Prop := d Person.A = Card.White

-- Define the event "Person B receives the white card"
def event_B_white (d : Distribution) : Prop := d Person.B = Card.White

-- State the theorem
theorem events_mutually_exclusive_not_complementary :
  (∀ d : Distribution, ¬(event_A_white d ∧ event_B_white d)) ∧
  (∃ d : Distribution, ¬event_A_white d ∧ ¬event_B_white d) :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l1691_169180


namespace NUMINAMATH_CALUDE_alex_bill_correct_l1691_169111

/-- Calculates the cell phone bill based on the given parameters. -/
def calculate_bill (base_cost : ℚ) (text_cost : ℚ) (extra_minute_cost : ℚ) 
                   (discount : ℚ) (texts_sent : ℕ) (hours_talked : ℕ) : ℚ :=
  let text_charge := text_cost * texts_sent
  let extra_minutes := max (hours_talked * 60 - 25 * 60) 0
  let extra_minute_charge := extra_minute_cost * extra_minutes
  let subtotal := base_cost + text_charge + extra_minute_charge
  let final_bill := if hours_talked > 35 then subtotal - discount else subtotal
  final_bill

theorem alex_bill_correct :
  calculate_bill 30 0.1 0.12 5 150 36 = 119.2 := by
  sorry

end NUMINAMATH_CALUDE_alex_bill_correct_l1691_169111


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1691_169108

/-- The line kx + y + k = 0 passes through the point (-1, 0) for all real k. -/
theorem line_passes_through_fixed_point :
  ∀ (k : ℝ), (k * (-1) + 0 + k = 0) := by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1691_169108


namespace NUMINAMATH_CALUDE_cookies_per_box_l1691_169155

/-- The number of cookies Basil consumes per day -/
def cookies_per_day : ℚ := 1/2 + 1/2 + 2

/-- The number of days Basil's cookies should last -/
def days : ℕ := 30

/-- The number of boxes needed for the given number of days -/
def boxes : ℕ := 2

/-- Theorem stating the number of cookies in each box -/
theorem cookies_per_box : 
  (cookies_per_day * days) / boxes = 45 := by sorry

end NUMINAMATH_CALUDE_cookies_per_box_l1691_169155


namespace NUMINAMATH_CALUDE_power_of_power_l1691_169154

theorem power_of_power : (3^4)^2 = 6561 := by sorry

end NUMINAMATH_CALUDE_power_of_power_l1691_169154


namespace NUMINAMATH_CALUDE_smartphone_sales_l1691_169117

theorem smartphone_sales (units_at_400 price_400 price_800 : ℝ) 
  (h1 : units_at_400 = 20)
  (h2 : price_400 = 400)
  (h3 : price_800 = 800)
  (h4 : ∀ (p c : ℝ), p * c = units_at_400 * price_400) :
  (units_at_400 * price_400) / price_800 = 10 := by
  sorry

end NUMINAMATH_CALUDE_smartphone_sales_l1691_169117


namespace NUMINAMATH_CALUDE_min_distance_on_parabola_l1691_169102

/-- The minimum distance between two points on y = 2x² where the line
    connecting them is perpendicular to the tangent at one point -/
theorem min_distance_on_parabola :
  let f (x : ℝ) := 2 * x^2
  let tangent_slope (a : ℝ) := 4 * a
  let perpendicular_slope (a : ℝ) := -1 / (tangent_slope a)
  let distance (a : ℝ) := 
    let t := 4 * a^2
    Real.sqrt ((1 / (64 * t^2)) + (1 / (2 * t)) + t + 9/4)
  ∃ (min_dist : ℝ), min_dist = 3 * Real.sqrt 3 / 4 ∧
    ∀ (a : ℝ), distance a ≥ min_dist :=
by sorry

end NUMINAMATH_CALUDE_min_distance_on_parabola_l1691_169102


namespace NUMINAMATH_CALUDE_first_triangular_covering_all_remainders_triangular_22_is_253_l1691_169177

/-- Triangular number function -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Function to check if a number covers all remainders modulo 10 -/
def covers_all_remainders (n : ℕ) : Prop :=
  ∀ r : Fin 10, ∃ k : ℕ, k ≤ n ∧ triangular_number k % 10 = r

/-- Main theorem: 22 is the smallest n for which triangular_number n covers all remainders modulo 10 -/
theorem first_triangular_covering_all_remainders :
  (covers_all_remainders 22 ∧ ∀ m < 22, ¬ covers_all_remainders m) :=
sorry

/-- Corollary: The 22nd triangular number is 253 -/
theorem triangular_22_is_253 : triangular_number 22 = 253 :=
sorry

end NUMINAMATH_CALUDE_first_triangular_covering_all_remainders_triangular_22_is_253_l1691_169177


namespace NUMINAMATH_CALUDE_marble_distribution_theorem_l1691_169149

/-- The number of ways to distribute marbles to students under specific conditions -/
def marbleDistributionWays : ℕ := 3150

/-- The total number of marbles -/
def totalMarbles : ℕ := 12

/-- The number of red marbles -/
def redMarbles : ℕ := 3

/-- The number of blue marbles -/
def blueMarbles : ℕ := 4

/-- The number of green marbles -/
def greenMarbles : ℕ := 5

/-- The total number of students -/
def totalStudents : ℕ := 12

theorem marble_distribution_theorem :
  marbleDistributionWays = 3150 ∧
  totalMarbles = redMarbles + blueMarbles + greenMarbles ∧
  totalStudents = totalMarbles ∧
  ∃ (distribution : Fin totalStudents → Fin 3),
    (∃ (i j : Fin totalStudents), i ≠ j ∧ distribution i = distribution j) ∧
    (∃ (k : Fin totalStudents), distribution k = 2) :=
by sorry

end NUMINAMATH_CALUDE_marble_distribution_theorem_l1691_169149


namespace NUMINAMATH_CALUDE_parabola_focus_distance_range_l1691_169139

theorem parabola_focus_distance_range :
  ∀ (A : ℝ × ℝ) (θ : ℝ),
    let F : ℝ × ℝ := (1/4, 0)
    let y : ℝ → ℝ := λ x => Real.sqrt x
    let l : ℝ → ℝ := λ x => Real.tan θ * (x - F.1) + F.2
    A.2 = y A.1 ∧  -- A is on the parabola
    A.2 > 0 ∧  -- A is above x-axis
    l A.1 = A.2 ∧  -- A is on line l
    θ ≥ π/4 →
    ∃ (FA : ℝ), FA > 1/4 ∧ FA ≤ 1 + Real.sqrt 2 / 2 ∧
      FA = Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_distance_range_l1691_169139


namespace NUMINAMATH_CALUDE_correct_number_increase_l1691_169174

theorem correct_number_increase : 
  ∀ (a b c d : ℕ), 
    (a = 3 ∧ b = 5 ∧ c = 7 ∧ d = 9) →
    (a + (b + 1) * c - d = 36) ∧
    (¬(a + 1 + b * c - d = 36)) ∧
    (¬(a + b * (c + 1) - d = 36)) ∧
    (¬(a + b * c - (d + 1) = 36)) :=
by sorry

end NUMINAMATH_CALUDE_correct_number_increase_l1691_169174


namespace NUMINAMATH_CALUDE_fish_weight_l1691_169167

/-- Represents the weight of a fish with specific relationships between its parts. -/
structure Fish where
  /-- Weight of the tail in kg -/
  tail : ℝ
  /-- Weight of the head in kg -/
  head : ℝ
  /-- Weight of the body in kg -/
  body : ℝ
  /-- The tail weighs 1 kg -/
  tail_weight : tail = 1
  /-- The head weighs as much as the tail and half the body -/
  head_weight : head = tail + body / 2
  /-- The body weighs as much as the head and the tail together -/
  body_weight : body = head + tail

/-- The total weight of the fish is 8 kg -/
theorem fish_weight (f : Fish) : f.tail + f.head + f.body = 8 := by
  sorry


end NUMINAMATH_CALUDE_fish_weight_l1691_169167


namespace NUMINAMATH_CALUDE_haley_recycling_cans_l1691_169157

theorem haley_recycling_cans : ∃ (c : ℕ), c = 9 ∧ c - 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_haley_recycling_cans_l1691_169157


namespace NUMINAMATH_CALUDE_no_solution_x5_y2_plus4_l1691_169129

theorem no_solution_x5_y2_plus4 : ¬ ∃ (x y : ℕ), x ≥ 1 ∧ y ≥ 1 ∧ x^5 = y^2 + 4 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_x5_y2_plus4_l1691_169129


namespace NUMINAMATH_CALUDE_basketball_game_scores_l1691_169113

/-- Represents the quarterly scores of a team -/
structure QuarterlyScores :=
  (q1 q2 q3 q4 : ℕ)

/-- Checks if the scores form an increasing geometric sequence -/
def is_increasing_geometric (s : QuarterlyScores) : Prop :=
  ∃ (r : ℚ), r > 1 ∧ s.q2 = s.q1 * r ∧ s.q3 = s.q2 * r ∧ s.q4 = s.q3 * r

/-- Checks if the scores form an increasing arithmetic sequence -/
def is_increasing_arithmetic (s : QuarterlyScores) : Prop :=
  ∃ (d : ℕ), d > 0 ∧ s.q2 = s.q1 + d ∧ s.q3 = s.q2 + d ∧ s.q4 = s.q3 + d

/-- Calculates the total score for a team -/
def total_score (s : QuarterlyScores) : ℕ :=
  s.q1 + s.q2 + s.q3 + s.q4

/-- Calculates the first half score for a team -/
def first_half_score (s : QuarterlyScores) : ℕ :=
  s.q1 + s.q2

theorem basketball_game_scores :
  ∀ (raiders wildcats : QuarterlyScores),
    is_increasing_geometric raiders →
    is_increasing_arithmetic wildcats →
    raiders.q1 = wildcats.q1 + 1 →
    total_score raiders = total_score wildcats + 2 →
    total_score raiders ≤ 100 →
    total_score wildcats ≤ 100 →
    first_half_score raiders + first_half_score wildcats = 25 := by
  sorry

end NUMINAMATH_CALUDE_basketball_game_scores_l1691_169113


namespace NUMINAMATH_CALUDE_inequality_proof_l1691_169138

theorem inequality_proof (x y : ℝ) (n : ℕ+) (hx : x > 0) (hy : y > 0) :
  (x^n.val / (1 + x^2)) + (y^n.val / (1 + y^2)) ≤ (x^n.val + y^n.val) / (1 + x*y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1691_169138


namespace NUMINAMATH_CALUDE_burger_cost_l1691_169145

theorem burger_cost (alice_burgers alice_sodas alice_total bill_burgers bill_sodas bill_total : ℕ)
  (h_alice : alice_burgers = 4 ∧ alice_sodas = 3 ∧ alice_total = 420)
  (h_bill : bill_burgers = 3 ∧ bill_sodas = 2 ∧ bill_total = 310) :
  ∃ (burger_cost soda_cost : ℕ),
    alice_burgers * burger_cost + alice_sodas * soda_cost = alice_total ∧
    bill_burgers * burger_cost + bill_sodas * soda_cost = bill_total ∧
    burger_cost = 90 := by
  sorry

end NUMINAMATH_CALUDE_burger_cost_l1691_169145


namespace NUMINAMATH_CALUDE_tetrahedron_edge_length_is_sqrt_2_l1691_169179

/-- Represents a cube with unit side length -/
structure UnitCube where
  center : ℝ × ℝ × ℝ

/-- Represents a tetrahedron circumscribed around four unit cubes -/
structure Tetrahedron where
  cubes : Fin 4 → UnitCube

/-- The edge length of the tetrahedron -/
def tetrahedron_edge_length (t : Tetrahedron) : ℝ := sorry

/-- The configuration of four unit cubes as described in the problem -/
def cube_configuration : Tetrahedron := sorry

theorem tetrahedron_edge_length_is_sqrt_2 :
  tetrahedron_edge_length cube_configuration = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_tetrahedron_edge_length_is_sqrt_2_l1691_169179


namespace NUMINAMATH_CALUDE_tetrahedron_planes_intersection_l1691_169171

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane3D where
  normal : Point3D
  point : Point3D

/-- Represents a tetrahedron -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- The circumcenter of a triangle -/
def circumcenter (a b c : Point3D) : Point3D := sorry

/-- The center of the circumsphere of a tetrahedron -/
def circumsphere_center (t : Tetrahedron) : Point3D := sorry

/-- A plane passing through a point and perpendicular to a line -/
def perpendicular_plane (point line_start line_end : Point3D) : Plane3D := sorry

/-- Check if a point lies on a plane -/
def point_on_plane (point : Point3D) (plane : Plane3D) : Prop := sorry

/-- Check if a tetrahedron is regular -/
def is_regular (t : Tetrahedron) : Prop := sorry

/-- The main theorem -/
theorem tetrahedron_planes_intersection
  (t : Tetrahedron)
  (A' : Point3D) (B' : Point3D) (C' : Point3D) (D' : Point3D)
  (h_A' : A' = circumcenter t.B t.C t.D)
  (h_B' : B' = circumcenter t.C t.D t.A)
  (h_C' : C' = circumcenter t.D t.A t.B)
  (h_D' : D' = circumcenter t.A t.B t.C)
  (P_A : Plane3D) (P_B : Plane3D) (P_C : Plane3D) (P_D : Plane3D)
  (h_P_A : P_A = perpendicular_plane t.A C' D')
  (h_P_B : P_B = perpendicular_plane t.B D' A')
  (h_P_C : P_C = perpendicular_plane t.C A' B')
  (h_P_D : P_D = perpendicular_plane t.D B' C')
  (P : Point3D)
  (h_P : P = circumsphere_center t) :
  ∃ (I : Point3D),
    point_on_plane I P_A ∧
    point_on_plane I P_B ∧
    point_on_plane I P_C ∧
    point_on_plane I P_D ∧
    (I = P ↔ is_regular t) := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_planes_intersection_l1691_169171


namespace NUMINAMATH_CALUDE_x_sixth_minus_six_x_squared_l1691_169105

theorem x_sixth_minus_six_x_squared (x : ℝ) (h : x = 3) : x^6 - 6*x^2 = 675 := by
  sorry

end NUMINAMATH_CALUDE_x_sixth_minus_six_x_squared_l1691_169105


namespace NUMINAMATH_CALUDE_equation_rewrite_product_l1691_169143

theorem equation_rewrite_product (a b x y : ℝ) (m' n' p' : ℤ) :
  (a^8*x*y - a^7*y - a^6*x = a^5*(b^5 - 2)) →
  ((a^m'*x - a^n') * (a^p'*y - a^3) = a^5*b^5) →
  m' * n' * p' = 48 := by
  sorry

end NUMINAMATH_CALUDE_equation_rewrite_product_l1691_169143


namespace NUMINAMATH_CALUDE_regular_pentagon_most_symmetric_l1691_169156

/-- Represents a geometric figure -/
inductive Figure
  | EquilateralTriangle
  | NonSquareRhombus
  | NonSquareRectangle
  | IsoscelesTrapezoid
  | RegularPentagon

/-- Returns the number of lines of symmetry for a given figure -/
def linesOfSymmetry (f : Figure) : ℕ :=
  match f with
  | Figure.EquilateralTriangle => 3
  | Figure.NonSquareRhombus => 2
  | Figure.NonSquareRectangle => 2
  | Figure.IsoscelesTrapezoid => 1
  | Figure.RegularPentagon => 5

/-- Theorem stating that the regular pentagon has the greatest number of lines of symmetry -/
theorem regular_pentagon_most_symmetric :
  ∀ f : Figure, f ≠ Figure.RegularPentagon → linesOfSymmetry Figure.RegularPentagon > linesOfSymmetry f :=
by sorry

end NUMINAMATH_CALUDE_regular_pentagon_most_symmetric_l1691_169156


namespace NUMINAMATH_CALUDE_necessary_condition_when_m_is_one_necessary_condition_range_l1691_169119

/-- Proposition P -/
def P : Set ℝ := {x | -2 ≤ x ∧ x ≤ 10}

/-- Proposition q -/
def q (m : ℝ) : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ 1 + m}

/-- P is a necessary but not sufficient condition for q -/
def necessary_not_sufficient (m : ℝ) : Prop :=
  (q m ⊆ P) ∧ (q m ≠ P) ∧ (m > 0)

theorem necessary_condition_when_m_is_one :
  necessary_not_sufficient 1 := by sorry

theorem necessary_condition_range :
  ∀ m : ℝ, necessary_not_sufficient m ↔ m ≥ 9 := by sorry

end NUMINAMATH_CALUDE_necessary_condition_when_m_is_one_necessary_condition_range_l1691_169119


namespace NUMINAMATH_CALUDE_max_ab_value_l1691_169163

theorem max_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : Real.sqrt 2 = Real.sqrt (2^a * 2^b)) : 
  (∀ x y : ℝ, x > 0 → y > 0 → Real.sqrt 2 = Real.sqrt (2^x * 2^y) → x * y ≤ a * b) → 
  a * b = 1/4 := by
sorry

end NUMINAMATH_CALUDE_max_ab_value_l1691_169163


namespace NUMINAMATH_CALUDE_cube_sum_theorem_l1691_169109

/-- Represents a cube with integers on its faces -/
structure Cube where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  d : ℕ+
  e : ℕ+
  f : ℕ+

/-- The sum of vertex products for a cube -/
def vertexSum (cube : Cube) : ℕ :=
  2 * (cube.a * cube.b * cube.c +
       cube.a * cube.b * cube.f +
       cube.d * cube.b * cube.c +
       cube.d * cube.b * cube.f)

/-- The sum of face numbers for a cube -/
def faceSum (cube : Cube) : ℕ :=
  cube.a + cube.b + cube.c + cube.d + cube.e + cube.f

/-- Theorem stating the relationship between vertex sum and face sum -/
theorem cube_sum_theorem (cube : Cube) 
  (h1 : vertexSum cube = 1332)
  (h2 : cube.b = cube.e) : 
  faceSum cube = 47 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_theorem_l1691_169109


namespace NUMINAMATH_CALUDE_min_sum_with_constraints_min_sum_achieved_l1691_169115

theorem min_sum_with_constraints (x y z : ℝ) 
  (hx : x ≥ 4) (hy : y ≥ 5) (hz : z ≥ 6) (h_sum_sq : x^2 + y^2 + z^2 ≥ 90) : 
  x + y + z ≥ 16 := by
  sorry

theorem min_sum_achieved (x y z : ℝ) 
  (hx : x ≥ 4) (hy : y ≥ 5) (hz : z ≥ 6) (h_sum_sq : x^2 + y^2 + z^2 ≥ 90) : 
  ∃ (a b c : ℝ), a ≥ 4 ∧ b ≥ 5 ∧ c ≥ 6 ∧ a^2 + b^2 + c^2 ≥ 90 ∧ a + b + c = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_with_constraints_min_sum_achieved_l1691_169115


namespace NUMINAMATH_CALUDE_calculation_difference_l1691_169168

def correct_calculation : ℤ := 12 - (3 * 4 + 2)

def incorrect_calculation : ℤ := 12 - 3 * 4 + 2

theorem calculation_difference :
  correct_calculation - incorrect_calculation = -4 := by
  sorry

end NUMINAMATH_CALUDE_calculation_difference_l1691_169168


namespace NUMINAMATH_CALUDE_help_sign_white_area_l1691_169175

/-- Represents the dimensions of a rectangular sign -/
structure SignDimensions where
  width : ℕ
  height : ℕ

/-- Calculates the area of a letter painted with 1-unit wide strokes -/
def letterArea (letter : Char) : ℕ :=
  match letter with
  | 'H' => 13
  | 'E' => 9
  | 'L' => 8
  | 'P' => 10
  | _ => 0

/-- Calculates the total area of a word painted with 1-unit wide strokes -/
def wordArea (word : String) : ℕ :=
  word.toList.map letterArea |> List.sum

/-- Theorem: The white area of the sign with "HELP" painted is 35 square units -/
theorem help_sign_white_area (sign : SignDimensions) 
  (h1 : sign.width = 15) 
  (h2 : sign.height = 5) : 
  sign.width * sign.height - wordArea "HELP" = 35 := by
  sorry

end NUMINAMATH_CALUDE_help_sign_white_area_l1691_169175


namespace NUMINAMATH_CALUDE_complementary_angle_of_35_30_l1691_169100

-- Define the angle in degrees and minutes
def angle_alpha : ℚ := 35 + 30 / 60

-- Define the complementary angle function
def complementary_angle (α : ℚ) : ℚ := 90 - α

-- Theorem statement
theorem complementary_angle_of_35_30 :
  let result := complementary_angle angle_alpha
  ⌊result⌋ = 54 ∧ (result - ⌊result⌋) * 60 = 30 := by
  sorry

#eval complementary_angle angle_alpha

end NUMINAMATH_CALUDE_complementary_angle_of_35_30_l1691_169100


namespace NUMINAMATH_CALUDE_computer_price_increase_l1691_169116

theorem computer_price_increase (y : ℝ) (h1 : 2 * y = 540) : 
  y * (1 + 0.3) = 351 := by sorry

end NUMINAMATH_CALUDE_computer_price_increase_l1691_169116


namespace NUMINAMATH_CALUDE_custom_mul_four_three_l1691_169101

/-- Custom multiplication operation -/
def custom_mul (a b : ℝ) : ℝ := a^2 - a*b + b^2

/-- Theorem stating that 4*3 = 13 under the custom multiplication -/
theorem custom_mul_four_three : custom_mul 4 3 = 13 := by
  sorry

end NUMINAMATH_CALUDE_custom_mul_four_three_l1691_169101


namespace NUMINAMATH_CALUDE_scientific_notation_of_120_million_l1691_169103

theorem scientific_notation_of_120_million :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 120000000 = a * (10 : ℝ) ^ n ∧ a = 1.2 ∧ n = 7 :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_of_120_million_l1691_169103


namespace NUMINAMATH_CALUDE_trajectory_of_midpoint_l1691_169150

-- Define the ellipse
def on_ellipse (x y : ℝ) : Prop := x^2 + 4*y^2 = 4

-- Define the midpoint relationship
def is_midpoint (mx my px py : ℝ) : Prop :=
  mx = (px + 4) / 2 ∧ my = py / 2

-- Theorem statement
theorem trajectory_of_midpoint :
  ∀ (x y : ℝ), 
    (∃ (x1 y1 : ℝ), on_ellipse x1 y1 ∧ is_midpoint x y x1 y1) →
    (x - 2)^2 + 4*y^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_trajectory_of_midpoint_l1691_169150


namespace NUMINAMATH_CALUDE_circle_radius_condition_l1691_169165

theorem circle_radius_condition (c : ℝ) : 
  (∀ x y : ℝ, x^2 + 8*x + y^2 - 6*y + c = 0 ↔ (x+4)^2 + (y-3)^2 = 25) → c = 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_condition_l1691_169165


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1691_169185

theorem necessary_but_not_sufficient : 
  (∀ x : ℝ, x^2 < x → |x - 1| < 2) ∧ 
  (∃ x : ℝ, |x - 1| < 2 ∧ x^2 ≥ x) := by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1691_169185


namespace NUMINAMATH_CALUDE_min_sum_reciprocals_l1691_169188

theorem min_sum_reciprocals (x y : ℕ+) (hxy : x ≠ y) (h : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 15) :
  ∃ (a b : ℕ+), a ≠ b ∧ (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 15 ∧ (↑a + ↑b : ℕ) = 64 ∧
    ∀ (c d : ℕ+), c ≠ d → (1 : ℚ) / c + (1 : ℚ) / d = (1 : ℚ) / 15 → (↑a + ↑b : ℕ) ≤ (↑c + ↑d : ℕ) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_reciprocals_l1691_169188


namespace NUMINAMATH_CALUDE_cricket_run_rate_theorem_l1691_169199

/-- Represents a cricket game scenario -/
structure CricketGame where
  total_overs : ℕ
  first_overs : ℕ
  first_run_rate : ℚ
  target : ℕ

/-- Calculates the required run rate for the remaining overs -/
def required_run_rate (game : CricketGame) : ℚ :=
  let remaining_overs := game.total_overs - game.first_overs
  let runs_scored := game.first_run_rate * game.first_overs
  let runs_needed := game.target - runs_scored
  runs_needed / remaining_overs

/-- Theorem stating the required run rate for the given scenario -/
theorem cricket_run_rate_theorem (game : CricketGame) 
  (h1 : game.total_overs = 50)
  (h2 : game.first_overs = 10)
  (h3 : game.first_run_rate = 4.8)
  (h4 : game.target = 282) :
  required_run_rate game = 5.85 := by
  sorry

#eval required_run_rate { total_overs := 50, first_overs := 10, first_run_rate := 4.8, target := 282 }

end NUMINAMATH_CALUDE_cricket_run_rate_theorem_l1691_169199


namespace NUMINAMATH_CALUDE_difference_not_arithmetic_for_k_ge_4_l1691_169160

/-- Two geometric sequences -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The difference sequence -/
def difference_sequence (a b : ℕ → ℝ) (n : ℕ) : ℝ :=
  a n - b n

/-- Arithmetic sequence with non-zero common difference -/
def is_arithmetic_with_nonzero_diff (c : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, d ≠ 0 ∧ ∀ n : ℕ, c (n + 1) - c n = d

theorem difference_not_arithmetic_for_k_ge_4 (a b : ℕ → ℝ) (k : ℕ) 
  (h1 : geometric_sequence a)
  (h2 : geometric_sequence b)
  (h3 : k ≥ 4) :
  ¬ is_arithmetic_with_nonzero_diff (difference_sequence a b) :=
sorry

end NUMINAMATH_CALUDE_difference_not_arithmetic_for_k_ge_4_l1691_169160


namespace NUMINAMATH_CALUDE_more_boys_than_girls_l1691_169118

/-- Given a school with 34 girls and 841 boys, prove that there are 807 more boys than girls. -/
theorem more_boys_than_girls (girls : ℕ) (boys : ℕ) 
  (h1 : girls = 34) (h2 : boys = 841) : boys - girls = 807 := by
  sorry

end NUMINAMATH_CALUDE_more_boys_than_girls_l1691_169118


namespace NUMINAMATH_CALUDE_ratio_is_two_to_one_l1691_169181

/-- An isosceles right-angled triangle with an inscribed square -/
structure IsoscelesRightTriangleWithSquare where
  /-- The side length of the isosceles right triangle -/
  x : ℝ
  /-- The distance from O to P on OB -/
  a : ℝ
  /-- The distance from O to Q on OA -/
  b : ℝ
  /-- The side length of the inscribed square PQRS -/
  s : ℝ
  /-- The side length of the triangle is positive -/
  x_pos : 0 < x
  /-- a and b are positive and their sum equals x -/
  ab_sum : 0 < a ∧ 0 < b ∧ a + b = x
  /-- The side length of the square is the sum of a and b -/
  square_side : s = a + b
  /-- The area of the square is 2/5 of the area of the triangle -/
  area_ratio : s^2 = (2/5) * (x^2/2)

/-- 
If an isosceles right-angled triangle AOB has a square PQRS inscribed as described, 
and the area of PQRS is 2/5 of the area of AOB, then the ratio of OP to OQ is 2:1.
-/
theorem ratio_is_two_to_one (t : IsoscelesRightTriangleWithSquare) : 
  t.a / t.b = 2 := by sorry

end NUMINAMATH_CALUDE_ratio_is_two_to_one_l1691_169181


namespace NUMINAMATH_CALUDE_least_five_digit_square_cube_l1691_169148

theorem least_five_digit_square_cube : ∃ n : ℕ,
  (n ≥ 10000 ∧ n < 100000) ∧  -- five-digit number
  (∃ a : ℕ, n = a^2) ∧        -- perfect square
  (∃ b : ℕ, n = b^3) ∧        -- perfect cube
  (∀ m : ℕ, m < n →
    (m < 10000 ∨ m ≥ 100000) ∨
    (∀ a : ℕ, m ≠ a^2) ∨
    (∀ b : ℕ, m ≠ b^3)) ∧
  n = 15625 :=
by sorry

end NUMINAMATH_CALUDE_least_five_digit_square_cube_l1691_169148


namespace NUMINAMATH_CALUDE_interior_angles_sum_increase_l1691_169198

/-- The sum of interior angles of a convex polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- Theorem: If the sum of interior angles of a convex polygon with n sides is 2340°,
    then the sum of interior angles of a convex polygon with (n + 4) sides is 3060°. -/
theorem interior_angles_sum_increase (n : ℕ) :
  sum_interior_angles n = 2340 → sum_interior_angles (n + 4) = 3060 := by
  sorry

end NUMINAMATH_CALUDE_interior_angles_sum_increase_l1691_169198
