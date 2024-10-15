import Mathlib

namespace NUMINAMATH_CALUDE_third_number_in_systematic_sampling_l78_7874

/-- Systematic sampling function that returns the nth number drawn -/
def systematicSample (totalStudents : Nat) (sampleSize : Nat) (firstDrawn : Nat) (n : Nat) : Nat :=
  firstDrawn + (n - 1) * (totalStudents / sampleSize)

theorem third_number_in_systematic_sampling
  (totalStudents : Nat)
  (sampleSize : Nat)
  (firstPartEnd : Nat)
  (firstDrawn : Nat)
  (h1 : totalStudents = 1000)
  (h2 : sampleSize = 50)
  (h3 : firstPartEnd = 20)
  (h4 : firstDrawn = 15)
  (h5 : firstDrawn ≤ firstPartEnd) :
  systematicSample totalStudents sampleSize firstDrawn 3 = 55 := by
sorry

#eval systematicSample 1000 50 15 3

end NUMINAMATH_CALUDE_third_number_in_systematic_sampling_l78_7874


namespace NUMINAMATH_CALUDE_cross_section_distance_l78_7826

/-- Represents a right hexagonal pyramid -/
structure RightHexagonalPyramid where
  /-- Height of the pyramid -/
  height : ℝ
  /-- Side length of the base hexagon -/
  base_side : ℝ

/-- Represents a cross section of the pyramid -/
structure CrossSection where
  /-- Distance from the apex of the pyramid -/
  distance : ℝ
  /-- Area of the cross section -/
  area : ℝ

/-- 
Theorem: In a right hexagonal pyramid, if two cross sections parallel to the base 
have areas of 300√3 sq ft and 675√3 sq ft, and these planes are 12 feet apart, 
then the distance from the apex to the larger cross section is 36 feet.
-/
theorem cross_section_distance 
  (pyramid : RightHexagonalPyramid) 
  (cs1 cs2 : CrossSection) 
  (h_area1 : cs1.area = 300 * Real.sqrt 3)
  (h_area2 : cs2.area = 675 * Real.sqrt 3)
  (h_distance : cs2.distance - cs1.distance = 12)
  (h_order : cs1.distance < cs2.distance) :
  cs2.distance = 36 := by
  sorry

end NUMINAMATH_CALUDE_cross_section_distance_l78_7826


namespace NUMINAMATH_CALUDE_fraction_comparison_l78_7872

theorem fraction_comparison : 
  (2.00000000004 / ((1.00000000004)^2 + 2.00000000004)) < 
  (2.00000000002 / ((1.00000000002)^2 + 2.00000000002)) := by
sorry

end NUMINAMATH_CALUDE_fraction_comparison_l78_7872


namespace NUMINAMATH_CALUDE_square_area_is_121_l78_7859

/-- A square in a 2D coordinate system --/
structure Square where
  x : ℝ
  y : ℝ

/-- The area of a square --/
def square_area (s : Square) : ℝ :=
  (20 - 9) ^ 2

/-- Theorem: The area of the given square is 121 square units --/
theorem square_area_is_121 (s : Square) : square_area s = 121 := by
  sorry

end NUMINAMATH_CALUDE_square_area_is_121_l78_7859


namespace NUMINAMATH_CALUDE_stratified_sampling_medium_supermarkets_l78_7815

theorem stratified_sampling_medium_supermarkets 
  (total_large : ℕ) (total_medium : ℕ) (total_small : ℕ) (sample_size : ℕ) :
  total_large = 200 →
  total_medium = 400 →
  total_small = 1400 →
  sample_size = 100 →
  (total_large + total_medium + total_small) * (sample_size / (total_large + total_medium + total_small)) = sample_size →
  total_medium * (sample_size / (total_large + total_medium + total_small)) = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_medium_supermarkets_l78_7815


namespace NUMINAMATH_CALUDE_anns_age_l78_7839

theorem anns_age (A B : ℕ) : 
  A + B = 52 → 
  B = (2 * B - A / 3) → 
  A = 39 := by sorry

end NUMINAMATH_CALUDE_anns_age_l78_7839


namespace NUMINAMATH_CALUDE_unit_digit_of_12_pow_100_l78_7899

-- Define the function to get the unit digit of a natural number
def unitDigit (n : ℕ) : ℕ := n % 10

-- Define the theorem
theorem unit_digit_of_12_pow_100 : unitDigit (12^100) = 6 := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_of_12_pow_100_l78_7899


namespace NUMINAMATH_CALUDE_homework_problem_solution_l78_7856

theorem homework_problem_solution :
  ∃ (a b c d : ℤ),
    a ≤ -1 ∧ b ≤ -1 ∧ c ≤ -1 ∧ d ≤ -1 ∧
    -a - b = -a * b ∧
    c * d = -182 * (1 / (-c - d)) :=
by sorry

end NUMINAMATH_CALUDE_homework_problem_solution_l78_7856


namespace NUMINAMATH_CALUDE_initial_balloons_l78_7865

theorem initial_balloons (lost_balloons current_balloons : ℕ) 
  (h1 : lost_balloons = 2)
  (h2 : current_balloons = 7) : 
  current_balloons + lost_balloons = 9 := by
  sorry

end NUMINAMATH_CALUDE_initial_balloons_l78_7865


namespace NUMINAMATH_CALUDE_certain_number_problem_l78_7879

theorem certain_number_problem (x : ℝ) : x * 11 = 99 → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l78_7879


namespace NUMINAMATH_CALUDE_sqrt_seven_sixth_power_l78_7860

theorem sqrt_seven_sixth_power : (Real.sqrt 7) ^ 6 = 343 := by sorry

end NUMINAMATH_CALUDE_sqrt_seven_sixth_power_l78_7860


namespace NUMINAMATH_CALUDE_sum_of_altitudes_triangle_l78_7896

/-- The sum of altitudes of a triangle formed by the line 8x + 3y = 48 and the coordinate axes -/
theorem sum_of_altitudes_triangle (x y : ℝ) (h : 8 * x + 3 * y = 48) :
  let x_intercept : ℝ := 48 / 8
  let y_intercept : ℝ := 48 / 3
  let hypotenuse : ℝ := Real.sqrt (x_intercept^2 + y_intercept^2)
  let altitude_to_hypotenuse : ℝ := 96 / hypotenuse
  x_intercept + y_intercept + altitude_to_hypotenuse = (22 * Real.sqrt 292 + 96) / Real.sqrt 292 := by
sorry

end NUMINAMATH_CALUDE_sum_of_altitudes_triangle_l78_7896


namespace NUMINAMATH_CALUDE_N_equals_one_l78_7870

theorem N_equals_one :
  let N := (Real.sqrt (Real.sqrt 5 + 2) + Real.sqrt (Real.sqrt 5 - 2)) / Real.sqrt (Real.sqrt 5 + 1) - Real.sqrt (3 - 2 * Real.sqrt 2)
  N = 1 := by sorry

end NUMINAMATH_CALUDE_N_equals_one_l78_7870


namespace NUMINAMATH_CALUDE_lucy_integers_l78_7810

theorem lucy_integers (x y : ℤ) (h1 : 3 * x + 2 * y = 85) (h2 : x = 19 ∨ y = 19) : 
  (x = 19 ∧ y = 14) ∨ (y = 19 ∧ x = 14) :=
sorry

end NUMINAMATH_CALUDE_lucy_integers_l78_7810


namespace NUMINAMATH_CALUDE_systematic_sampling_questionnaire_C_l78_7827

/-- Systematic sampling problem -/
theorem systematic_sampling_questionnaire_C (total_population : ℕ) 
  (sample_size : ℕ) (first_number : ℕ) : 
  total_population = 960 →
  sample_size = 32 →
  first_number = 9 →
  (960 - 750) / (960 / 32) = 7 :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_questionnaire_C_l78_7827


namespace NUMINAMATH_CALUDE_cube_three_times_cube_six_l78_7806

theorem cube_three_times_cube_six : 3^3 * 6^3 = 5832 := by
  sorry

end NUMINAMATH_CALUDE_cube_three_times_cube_six_l78_7806


namespace NUMINAMATH_CALUDE_z_sixth_power_l78_7884

theorem z_sixth_power (z : ℂ) : z = (-Real.sqrt 3 + Complex.I) / 2 → z^6 = -1 := by
  sorry

end NUMINAMATH_CALUDE_z_sixth_power_l78_7884


namespace NUMINAMATH_CALUDE_coin_problem_l78_7825

/-- Given a total of 12 coins consisting of quarters and nickels with a total value of 220 cents, 
    prove that the number of nickels is 4. -/
theorem coin_problem (q n : ℕ) : 
  q + n = 12 → 
  25 * q + 5 * n = 220 → 
  n = 4 := by
  sorry

end NUMINAMATH_CALUDE_coin_problem_l78_7825


namespace NUMINAMATH_CALUDE_fraction_product_theorem_l78_7883

theorem fraction_product_theorem :
  (7 / 5 : ℚ) * (8 / 12 : ℚ) * (21 / 15 : ℚ) * (16 / 24 : ℚ) * 
  (35 / 25 : ℚ) * (20 / 30 : ℚ) * (49 / 35 : ℚ) * (32 / 48 : ℚ) = 38416 / 50625 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_theorem_l78_7883


namespace NUMINAMATH_CALUDE_brother_age_l78_7849

theorem brother_age (man_age brother_age : ℕ) : 
  man_age = brother_age + 12 →
  man_age + 2 = 2 * (brother_age + 2) →
  brother_age = 10 := by
sorry

end NUMINAMATH_CALUDE_brother_age_l78_7849


namespace NUMINAMATH_CALUDE_ceiling_squared_negative_fraction_l78_7893

theorem ceiling_squared_negative_fraction : ⌈(-7/4)^2⌉ = 4 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_squared_negative_fraction_l78_7893


namespace NUMINAMATH_CALUDE_cash_refund_per_bottle_l78_7888

/-- The number of bottles of kombucha Henry drinks per month -/
def bottles_per_month : ℕ := 15

/-- The cost of each bottle of kombucha in dollars -/
def bottle_cost : ℚ := 3

/-- The number of bottles Henry can buy with his cash refund after 1 year -/
def bottles_from_refund : ℕ := 6

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- Theorem: The cash refund per bottle is $0.10 -/
theorem cash_refund_per_bottle :
  (bottles_from_refund * bottle_cost) / (bottles_per_month * months_in_year) = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_cash_refund_per_bottle_l78_7888


namespace NUMINAMATH_CALUDE_geometric_progression_ratio_l78_7869

theorem geometric_progression_ratio (x y z r : ℝ) : 
  x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z →
  ∃ (a : ℝ), a ≠ 0 ∧ 
    x * (y - z) = a ∧
    y * (z - x) = a * r ∧
    z * (y - x) = a * r^2 →
  r^2 - r + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_ratio_l78_7869


namespace NUMINAMATH_CALUDE_complex_multiplication_l78_7881

/-- Given two complex numbers z₁ and z₂, prove that their product is equal to the specified result. -/
theorem complex_multiplication (z₁ z₂ : ℂ) : 
  z₁ = 1 - 3*I → z₂ = 6 - 8*I → z₁ * z₂ = -18 - 26*I := by
  sorry


end NUMINAMATH_CALUDE_complex_multiplication_l78_7881


namespace NUMINAMATH_CALUDE_quadratic_function_inequality_l78_7890

theorem quadratic_function_inequality (a b : ℝ) : 
  let f : ℝ → ℝ := fun x ↦ x^2 + a*x + b
  |f 1| + 2 * |f 2| + |f 3| ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_inequality_l78_7890


namespace NUMINAMATH_CALUDE_waiter_shift_earnings_l78_7851

/-- Calculates the waiter's earnings during a shift --/
def waiter_earnings (total_customers : ℕ) 
                    (three_dollar_tippers : ℕ) 
                    (four_fifty_tippers : ℕ) 
                    (non_tippers : ℕ) 
                    (tip_pool_contribution : ℚ) 
                    (meal_cost : ℚ) : ℚ :=
  (3 * three_dollar_tippers + 4.5 * four_fifty_tippers) - tip_pool_contribution - meal_cost

theorem waiter_shift_earnings :
  waiter_earnings 15 6 4 5 10 5 = 21 := by
  sorry

end NUMINAMATH_CALUDE_waiter_shift_earnings_l78_7851


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l78_7812

/-- Given a parabola with equation x^2 = 12y, the distance from its focus to its directrix is 6 -/
theorem parabola_focus_directrix_distance :
  ∀ (x y : ℝ), x^2 = 12*y →
  ∃ (focus_x focus_y directrix_y : ℝ),
    (focus_x = 0 ∧ focus_y = 3 ∧ directrix_y = -3) ∧
    (focus_y - directrix_y = 6) :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l78_7812


namespace NUMINAMATH_CALUDE_median_in_65_interval_l78_7895

/-- Represents a score interval with its lower bound and frequency -/
structure ScoreInterval :=
  (lower_bound : ℕ)
  (frequency : ℕ)

/-- Finds the interval containing the median score -/
def find_median_interval (intervals : List ScoreInterval) : Option ℕ :=
  let total_students := intervals.foldl (fun acc i => acc + i.frequency) 0
  let median_position := (total_students + 1) / 2
  let rec find_interval (acc : ℕ) (remaining : List ScoreInterval) : Option ℕ :=
    match remaining with
    | [] => none
    | i :: is =>
        if acc + i.frequency ≥ median_position then
          some i.lower_bound
        else
          find_interval (acc + i.frequency) is
  find_interval 0 intervals

theorem median_in_65_interval (score_data : List ScoreInterval) :
  score_data = [
    ⟨80, 20⟩, ⟨75, 15⟩, ⟨70, 10⟩, ⟨65, 25⟩, ⟨60, 15⟩, ⟨55, 15⟩
  ] →
  find_median_interval score_data = some 65 :=
by sorry

end NUMINAMATH_CALUDE_median_in_65_interval_l78_7895


namespace NUMINAMATH_CALUDE_family_income_problem_l78_7850

theorem family_income_problem (x : ℝ) : 
  (4 * x - 1178) / 3 = 650 → x = 782 := by
  sorry

end NUMINAMATH_CALUDE_family_income_problem_l78_7850


namespace NUMINAMATH_CALUDE_ribbon_length_reduction_l78_7820

theorem ribbon_length_reduction (original_length : ℝ) (ratio_original : ℝ) (ratio_new : ℝ) (new_length : ℝ) : 
  original_length = 55 →
  ratio_original = 11 →
  ratio_new = 7 →
  new_length = (original_length * ratio_new) / ratio_original →
  new_length = 35 := by
sorry

end NUMINAMATH_CALUDE_ribbon_length_reduction_l78_7820


namespace NUMINAMATH_CALUDE_anna_phone_chargers_l78_7814

/-- The number of phone chargers Anna has -/
def phone_chargers : ℕ := sorry

/-- The number of laptop chargers Anna has -/
def laptop_chargers : ℕ := sorry

/-- The total number of chargers Anna has -/
def total_chargers : ℕ := 24

theorem anna_phone_chargers :
  (laptop_chargers = 5 * phone_chargers) →
  (phone_chargers + laptop_chargers = total_chargers) →
  phone_chargers = 4 := by
sorry

end NUMINAMATH_CALUDE_anna_phone_chargers_l78_7814


namespace NUMINAMATH_CALUDE_tan_theta_half_l78_7889

theorem tan_theta_half (θ : Real) (h : (1 + Real.cos (2 * θ)) / Real.sin (2 * θ) = 2) : 
  Real.tan θ = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_half_l78_7889


namespace NUMINAMATH_CALUDE_first_four_eq_last_four_l78_7866

/-- A finite sequence of 0s and 1s with special properties -/
def SpecialSequence : Type :=
  {s : List Bool // 
    (∀ i j, i ≠ j → i + 5 ≤ s.length → j + 5 ≤ s.length → 
      (List.take 5 (List.drop i s) ≠ List.take 5 (List.drop j s))) ∧
    (¬∀ i j, i ≠ j → i + 5 ≤ (s ++ [true]).length → j + 5 ≤ (s ++ [true]).length → 
      (List.take 5 (List.drop i (s ++ [true])) ≠ List.take 5 (List.drop j (s ++ [true])))) ∧
    (¬∀ i j, i ≠ j → i + 5 ≤ (s ++ [false]).length → j + 5 ≤ (s ++ [false]).length → 
      (List.take 5 (List.drop i (s ++ [false])) ≠ List.take 5 (List.drop j (s ++ [false]))))}

/-- The theorem stating that the first 4 digits are the same as the last 4 digits -/
theorem first_four_eq_last_four (s : SpecialSequence) : 
  List.take 4 s.val = List.take 4 (List.reverse s.val) := by
  sorry

end NUMINAMATH_CALUDE_first_four_eq_last_four_l78_7866


namespace NUMINAMATH_CALUDE_conic_eccentricity_l78_7863

/-- Given that 1, m, and 9 form a geometric sequence, 
    the eccentricity of the conic section x²/m + y² = 1 is either √6/3 or 2 -/
theorem conic_eccentricity (m : ℝ) : 
  (1 * 9 = m^2) →  -- geometric sequence condition
  (∃ e : ℝ, (e = Real.sqrt 6 / 3 ∨ e = 2) ∧
   ∀ x y : ℝ, x^2 / m + y^2 = 1 → 
   e = if m > 0 
       then Real.sqrt (1 - 1 / m) 
       else Real.sqrt (1 - m) / Real.sqrt (-m)) :=
by sorry

end NUMINAMATH_CALUDE_conic_eccentricity_l78_7863


namespace NUMINAMATH_CALUDE_shirt_selection_theorem_l78_7832

/-- The number of shirts of each color in the drawer -/
def shirts : Finset (Nat × Nat) := {(4, 1), (7, 2), (9, 3)}

/-- The total number of shirts in the drawer -/
def total_shirts : Nat := 20

/-- The minimum number of shirts to select to ensure n shirts of the same color -/
def min_select (n : Nat) : Nat :=
  if n ≤ 4 then 3 * (n - 1) + 1
  else min (3 * (n - 1) + 1) total_shirts

/-- Theorem stating the minimum number of shirts to select for each case -/
theorem shirt_selection_theorem :
  (min_select 4 = 10) ∧
  (min_select 5 = 13) ∧
  (min_select 6 = 16) ∧
  (min_select 7 = 17) ∧
  (min_select 8 = 19) ∧
  (min_select 9 = 20) := by
  sorry

end NUMINAMATH_CALUDE_shirt_selection_theorem_l78_7832


namespace NUMINAMATH_CALUDE_exists_quadrilateral_equal_angle_tangents_l78_7836

/-- Represents a quadrilateral with four interior angles -/
structure Quadrilateral where
  α : Real
  β : Real
  γ : Real
  δ : Real
  sum_360 : α + β + γ + δ = 2 * Real.pi

/-- The theorem stating the existence of a quadrilateral with equal angle tangents -/
theorem exists_quadrilateral_equal_angle_tangents :
  ∃ q : Quadrilateral, Real.tan q.α = Real.tan q.β ∧ Real.tan q.α = Real.tan q.γ ∧ Real.tan q.α = Real.tan q.δ :=
by sorry

end NUMINAMATH_CALUDE_exists_quadrilateral_equal_angle_tangents_l78_7836


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_negative_two_m_l78_7857

theorem sqrt_difference_equals_negative_two_m (m n : ℝ) (h1 : n < m) (h2 : m < 0) :
  Real.sqrt (m^2 + 2*m*n + n^2) - Real.sqrt (m^2 - 2*m*n + n^2) = -2*m := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_negative_two_m_l78_7857


namespace NUMINAMATH_CALUDE_square_root_fraction_simplification_l78_7802

theorem square_root_fraction_simplification :
  Real.sqrt (8^2 + 6^2) / Real.sqrt (25 + 16) = 10 / Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_square_root_fraction_simplification_l78_7802


namespace NUMINAMATH_CALUDE_min_value_theorem_l78_7841

def is_arithmetic_geometric (a : ℕ → ℝ) : Prop :=
  ∃ (d q : ℝ), q > 0 ∧ ∀ n, a (n + 1) = a n * q + d

theorem min_value_theorem (a : ℕ → ℝ) (m n : ℕ) :
  is_arithmetic_geometric a →
  (∀ n, a n > 0) →
  a 7 = a 6 + 2 * a 5 →
  a m * a n = 4 * (a 1) ^ 2 →
  (1 : ℝ) / m + 4 / n ≥ 9 / 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l78_7841


namespace NUMINAMATH_CALUDE_third_derivative_at_negative_one_l78_7886

/-- Given a function f where f(x) = e^(-x) + 2f''(0)x, prove that f'''(-1) = 2 - e -/
theorem third_derivative_at_negative_one (f : ℝ → ℝ) (h : ∀ x, f x = Real.exp (-x) + 2 * (deriv^[2] f 0) * x) :
  (deriv^[3] f) (-1) = 2 - Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_third_derivative_at_negative_one_l78_7886


namespace NUMINAMATH_CALUDE_M_equals_set_l78_7894

def M (x y z : ℝ) : Set ℝ :=
  { w | ∃ (x y z : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧
    w = (x / abs x) + (y / abs y) + (z / abs z) + (abs (x * y * z) / (x * y * z)) }

theorem M_equals_set (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  M x y z = {4, -4, 0} := by
  sorry

end NUMINAMATH_CALUDE_M_equals_set_l78_7894


namespace NUMINAMATH_CALUDE_shaded_circle_fraction_l78_7804

/-- Given a circle divided into equal regions, this theorem proves that
    if there are 4 regions and 1 is shaded, then the shaded fraction is 1/4. -/
theorem shaded_circle_fraction (total_regions shaded_regions : ℕ) :
  total_regions = 4 →
  shaded_regions = 1 →
  (shaded_regions : ℚ) / total_regions = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_shaded_circle_fraction_l78_7804


namespace NUMINAMATH_CALUDE_log_equation_solution_l78_7821

theorem log_equation_solution (p q : ℝ) (h1 : p > q) (h2 : q > 0) :
  Real.log p + Real.log q = Real.log (p - q) ↔ p = q / (1 - q) ∧ q < 1 :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_l78_7821


namespace NUMINAMATH_CALUDE_triangle_problem_l78_7871

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- b(cos A - 2cos C) = (2c - a)cos B
  b * (Real.cos A - 2 * Real.cos C) = (2 * c - a) * Real.cos B →
  -- Part I: Prove c/a = 2
  c / a = 2 ∧
  -- Part II: If cos B = 1/4 and perimeter = 5, prove b = 2
  (Real.cos B = 1/4 ∧ a + b + c = 5 → b = 2) := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l78_7871


namespace NUMINAMATH_CALUDE_anands_income_is_2000_l78_7811

/-- Represents the financial data of a person --/
structure FinancialData where
  income : ℕ
  expenditure : ℕ
  savings : ℕ

/-- Proves that Anand's income is 2000 given the conditions --/
theorem anands_income_is_2000 
  (anand balu : FinancialData)
  (income_ratio : anand.income * 4 = balu.income * 5)
  (expenditure_ratio : anand.expenditure * 2 = balu.expenditure * 3)
  (anand_savings : anand.income - anand.expenditure = 800)
  (balu_savings : balu.income - balu.expenditure = 800) :
  anand.income = 2000 := by
  sorry

#check anands_income_is_2000

end NUMINAMATH_CALUDE_anands_income_is_2000_l78_7811


namespace NUMINAMATH_CALUDE_total_money_l78_7824

theorem total_money (mark carolyn dave : ℚ) 
  (h1 : mark = 4/5)
  (h2 : carolyn = 2/5)
  (h3 : dave = 1/2) :
  mark + carolyn + dave = 17/10 := by
  sorry

end NUMINAMATH_CALUDE_total_money_l78_7824


namespace NUMINAMATH_CALUDE_tan_function_value_l78_7813

theorem tan_function_value (f : ℝ → ℝ) :
  (∀ x, f x = Real.tan (2 * x + π / 3)) →
  f (25 * π / 6) = -Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_tan_function_value_l78_7813


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l78_7808

theorem complex_number_in_fourth_quadrant :
  let z : ℂ := (1 - 2*I) * (2 + I)
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l78_7808


namespace NUMINAMATH_CALUDE_jakes_weight_l78_7867

/-- Given the weights of Mildred, Carol, and Jake, prove Jake's weight -/
theorem jakes_weight (mildred_weight : ℕ) (carol_weight : ℕ) (jake_weight : ℕ) 
  (h1 : mildred_weight = 59)
  (h2 : carol_weight = mildred_weight + 9)
  (h3 : jake_weight = 2 * carol_weight) : 
  jake_weight = 136 := by
  sorry

end NUMINAMATH_CALUDE_jakes_weight_l78_7867


namespace NUMINAMATH_CALUDE_grade_improvement_l78_7835

/-- Represents the distribution of grades --/
structure GradeDistribution where
  a : ℕ  -- number of 1's
  b : ℕ  -- number of 2's
  c : ℕ  -- number of 3's
  d : ℕ  -- number of 4's
  e : ℕ  -- number of 5's

/-- Calculates the average grade --/
def averageGrade (g : GradeDistribution) : ℚ :=
  (g.a + 2 * g.b + 3 * g.c + 4 * g.d + 5 * g.e) / (g.a + g.b + g.c + g.d + g.e)

/-- Represents the change in grade distribution after changing 1's to 3's --/
def changeGrades (g : GradeDistribution) : GradeDistribution :=
  { a := 0, b := g.b, c := g.c + g.a, d := g.d, e := g.e }

theorem grade_improvement (g : GradeDistribution) :
  averageGrade g < 3 → averageGrade (changeGrades g) ≤ 4 := by
  sorry


end NUMINAMATH_CALUDE_grade_improvement_l78_7835


namespace NUMINAMATH_CALUDE_abs_value_sum_and_diff_l78_7834

theorem abs_value_sum_and_diff (a b : ℝ) :
  (abs a = 5 ∧ abs b = 3) →
  ((a > 0 ∧ b < 0) → a + b = 2) ∧
  (abs (a + b) = a + b → (a - b = 2 ∨ a - b = 8)) :=
by sorry

end NUMINAMATH_CALUDE_abs_value_sum_and_diff_l78_7834


namespace NUMINAMATH_CALUDE_victors_lives_l78_7816

theorem victors_lives (lost : ℕ) (diff : ℕ) (current : ℕ) : 
  lost = 14 → diff = 12 → lost - current = diff → current = 2 := by
  sorry

end NUMINAMATH_CALUDE_victors_lives_l78_7816


namespace NUMINAMATH_CALUDE_same_solution_implies_c_equals_nine_l78_7828

theorem same_solution_implies_c_equals_nine (x c : ℝ) :
  (3 * x + 5 = 4) ∧ (c * x + 6 = 3) → c = 9 :=
by sorry

end NUMINAMATH_CALUDE_same_solution_implies_c_equals_nine_l78_7828


namespace NUMINAMATH_CALUDE_quadratic_root_m_value_l78_7885

theorem quadratic_root_m_value : ∀ m : ℝ, 
  (1 : ℝ)^2 + m * 1 - 6 = 0 → m = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_m_value_l78_7885


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l78_7861

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a + b + c = 40 →
  (1/2) * a * b = 30 →
  a^2 + b^2 = c^2 →
  c = 18.5 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l78_7861


namespace NUMINAMATH_CALUDE_simplify_expression_l78_7803

theorem simplify_expression (b : ℝ) : 3*b*(3*b^2 + 2*b) - 2*b^2 = 9*b^3 + 4*b^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l78_7803


namespace NUMINAMATH_CALUDE_cube_plus_135002_l78_7873

theorem cube_plus_135002 (n : ℤ) : 
  (n = 149 ∨ n = -151) → n^3 + 135002 = (n + 1)^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_plus_135002_l78_7873


namespace NUMINAMATH_CALUDE_even_q_l78_7805

theorem even_q (p q : ℕ) 
  (h1 : ∃ (n : ℕ), n^2 = 2*p - q) 
  (h2 : ∃ (m : ℕ), m^2 = 2*p + q) : 
  Even q := by
sorry

end NUMINAMATH_CALUDE_even_q_l78_7805


namespace NUMINAMATH_CALUDE_smallest_possible_d_l78_7833

theorem smallest_possible_d : ∃ d : ℝ, d > 0 ∧ 
  (5 * Real.sqrt 2)^2 + (d + 4)^2 = (4 * d)^2 ∧ 
  ∀ d' : ℝ, d' > 0 → (5 * Real.sqrt 2)^2 + (d' + 4)^2 = (4 * d')^2 → d ≤ d' :=
by sorry

end NUMINAMATH_CALUDE_smallest_possible_d_l78_7833


namespace NUMINAMATH_CALUDE_function_decreasing_interval_l78_7807

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := x^2 * (a*x + b)

-- Define the derivative of f(x)
def f_derivative (a b x : ℝ) : ℝ := 3*a*x^2 + 2*b*x

theorem function_decreasing_interval (a b : ℝ) :
  (∀ x, f_derivative a b x = 0 → x = 2) →  -- Extremum at x = 2
  (f_derivative a b 1 = -3) →              -- Tangent line parallel to 3x + y = 0
  (∀ x ∈ (Set.Ioo 0 2), f_derivative a b x < 0) ∧ 
  (∀ x ∉ (Set.Icc 0 2), f_derivative a b x ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_function_decreasing_interval_l78_7807


namespace NUMINAMATH_CALUDE_hall_volume_proof_l78_7862

/-- Represents a rectangular wall with a width and height -/
structure RectWall where
  width : ℝ
  height : ℝ

/-- Represents a rectangular hall with length, width, and height -/
structure RectHall where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculate the area of a rectangular wall -/
def wallArea (w : RectWall) : ℝ := w.width * w.height

/-- Calculate the volume of a rectangular hall -/
def hallVolume (h : RectHall) : ℝ := h.length * h.width * h.height

theorem hall_volume_proof (h : RectHall) 
  (a1 a2 : RectWall) 
  (b1 b2 : RectWall) 
  (c1 c2 : RectWall) :
  h.length = 30 ∧ 
  h.width = 20 ∧ 
  h.height = 10 ∧
  a1.width = a2.width ∧
  b1.height = b2.height ∧
  c1.height = c2.height ∧
  b1.height = h.height ∧
  c1.height = h.height ∧
  wallArea a1 + wallArea a2 = wallArea b1 + wallArea b2 ∧
  wallArea c1 + wallArea c2 = 2 * h.length * h.width ∧
  a1.width + a2.width = h.width ∧
  b1.width + b2.width = h.length ∧
  c1.width + c2.width = h.width →
  hallVolume h = 6000 := by
sorry

end NUMINAMATH_CALUDE_hall_volume_proof_l78_7862


namespace NUMINAMATH_CALUDE_second_number_value_l78_7878

theorem second_number_value (x y z : ℚ) 
  (sum_eq : x + y + z = 120)
  (ratio_xy : x / y = 3 / 4)
  (ratio_yz : y / z = 7 / 9) :
  y = 672 / 17 := by
sorry

end NUMINAMATH_CALUDE_second_number_value_l78_7878


namespace NUMINAMATH_CALUDE_distance_difference_l78_7854

-- Define the distances
def john_distance : ℝ := 0.7
def nina_distance : ℝ := 0.4

-- Theorem statement
theorem distance_difference : john_distance - nina_distance = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_distance_difference_l78_7854


namespace NUMINAMATH_CALUDE_fruit_stand_problem_l78_7853

/-- Proves that the price of each apple is $0.90 given the conditions of the fruit stand problem -/
theorem fruit_stand_problem (total_cost : ℝ) (total_fruits : ℕ) (banana_price : ℝ)
  (h_total_cost : total_cost = 6.50)
  (h_total_fruits : total_fruits = 9)
  (h_banana_price : banana_price = 0.70) :
  ∃ (apple_price : ℝ) (num_apples : ℕ),
    apple_price = 0.90 ∧
    num_apples + (total_fruits - num_apples) = total_fruits ∧
    apple_price * num_apples + banana_price * (total_fruits - num_apples) = total_cost :=
by
  sorry

#check fruit_stand_problem

end NUMINAMATH_CALUDE_fruit_stand_problem_l78_7853


namespace NUMINAMATH_CALUDE_max_cups_in_kitchen_l78_7868

theorem max_cups_in_kitchen (a b : ℕ) : 
  (a.choose 2) * (b.choose 3) = 1200 → a + b ≤ 29 :=
by sorry

end NUMINAMATH_CALUDE_max_cups_in_kitchen_l78_7868


namespace NUMINAMATH_CALUDE_smallest_multiple_of_all_up_to_ten_l78_7830

def is_multiple_of_all (n : ℕ) : Prop :=
  ∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 → x ∣ n

theorem smallest_multiple_of_all_up_to_ten :
  ∃ n : ℕ, is_multiple_of_all n ∧ ∀ m : ℕ, is_multiple_of_all m → n ≤ m :=
sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_all_up_to_ten_l78_7830


namespace NUMINAMATH_CALUDE_prob_three_even_out_of_six_l78_7864

/-- The probability of rolling an even number on a fair 12-sided die -/
def prob_even : ℚ := 1 / 2

/-- The number of dice rolled -/
def num_dice : ℕ := 6

/-- The number of dice we want to show even numbers -/
def target_even : ℕ := 3

/-- The probability of exactly three out of six fair 12-sided dice showing an even number -/
theorem prob_three_even_out_of_six :
  (Nat.choose num_dice target_even : ℚ) * prob_even ^ num_dice = 5 / 16 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_even_out_of_six_l78_7864


namespace NUMINAMATH_CALUDE_fraction_of_juices_consumed_l78_7844

/-- Represents the fraction of juices consumed at a summer picnic -/
theorem fraction_of_juices_consumed (total_people : ℕ) (soda_cans : ℕ) (water_bottles : ℕ) (juice_bottles : ℕ)
  (soda_drinkers : ℚ) (water_drinkers : ℚ) (total_recyclables : ℕ) :
  total_people = 90 →
  soda_cans = 50 →
  water_bottles = 50 →
  juice_bottles = 50 →
  soda_drinkers = 1/2 →
  water_drinkers = 1/3 →
  total_recyclables = 115 →
  (juice_bottles - (total_recyclables - (soda_drinkers * total_people + water_drinkers * total_people))) / juice_bottles = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_juices_consumed_l78_7844


namespace NUMINAMATH_CALUDE_business_join_time_l78_7882

/-- Represents the number of months in a year -/
def months_in_year : ℕ := 12

/-- Represents A's investment in Rupees -/
def investment_A : ℕ := 36000

/-- Represents B's investment in Rupees -/
def investment_B : ℕ := 54000

/-- Represents the ratio of A's profit share to B's profit share -/
def profit_ratio : ℚ := 2 / 1

theorem business_join_time (x : ℕ) : 
  (investment_A * months_in_year : ℚ) / (investment_B * (months_in_year - x)) = profit_ratio →
  x = 8 :=
by sorry

end NUMINAMATH_CALUDE_business_join_time_l78_7882


namespace NUMINAMATH_CALUDE_grace_mowing_hours_l78_7817

/-- Represents the rates and hours worked by Grace in her landscaping business -/
structure LandscapingWork where
  mowing_rate : ℕ
  weeding_rate : ℕ
  mulching_rate : ℕ
  weeding_hours : ℕ
  mulching_hours : ℕ
  total_earnings : ℕ

/-- Calculates the number of hours spent mowing lawns given the landscaping work details -/
def mowing_hours (work : LandscapingWork) : ℕ :=
  (work.total_earnings - (work.weeding_rate * work.weeding_hours + work.mulching_rate * work.mulching_hours)) / work.mowing_rate

/-- Theorem stating that Grace spent 63 hours mowing lawns in September -/
theorem grace_mowing_hours :
  let work : LandscapingWork := {
    mowing_rate := 6,
    weeding_rate := 11,
    mulching_rate := 9,
    weeding_hours := 9,
    mulching_hours := 10,
    total_earnings := 567
  }
  mowing_hours work = 63 := by sorry

end NUMINAMATH_CALUDE_grace_mowing_hours_l78_7817


namespace NUMINAMATH_CALUDE_expression_value_l78_7848

theorem expression_value : (3^2 - 3) + (4^2 - 4) - (5^2 - 5) = -2 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l78_7848


namespace NUMINAMATH_CALUDE_line_segment_proportion_l78_7822

-- Define the line segments as real numbers (representing their lengths in cm)
def a : ℝ := 1
def b : ℝ := 4
def c : ℝ := 2

-- Define the proportion relationship
def are_proportional (a b c d : ℝ) : Prop := a * d = b * c

-- State the theorem
theorem line_segment_proportion :
  ∀ d : ℝ, are_proportional a b c d → d = 8 :=
by sorry

end NUMINAMATH_CALUDE_line_segment_proportion_l78_7822


namespace NUMINAMATH_CALUDE_trapezoid_base_ratio_l78_7837

/-- A trapezoid with bases a and b, where a > b, and its midsegment is divided into three equal parts by the diagonals. -/
structure Trapezoid where
  a : ℝ
  b : ℝ
  h : a > b

/-- The ratio of the bases of a trapezoid with the given properties is 2:1 -/
theorem trapezoid_base_ratio (t : Trapezoid) : t.a / t.b = 2 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_base_ratio_l78_7837


namespace NUMINAMATH_CALUDE_exists_1992_gon_l78_7876

/-- A convex polygon with n sides that is circumscribable about a circle -/
structure CircumscribablePolygon (n : ℕ) where
  sides : Fin n → ℝ
  convex : sorry
  circumscribable : sorry

/-- The condition that the side lengths are 1, 2, 3, ..., n in some order -/
def valid_side_lengths (n : ℕ) (p : CircumscribablePolygon n) : Prop :=
  ∃ (σ : Equiv (Fin n) (Fin n)), ∀ i, p.sides i = (σ i).val + 1

/-- The main theorem stating the existence of a 1992-sided circumscribable polygon
    with side lengths 1, 2, 3, ..., 1992 in some order -/
theorem exists_1992_gon :
  ∃ (p : CircumscribablePolygon 1992), valid_side_lengths 1992 p :=
sorry

end NUMINAMATH_CALUDE_exists_1992_gon_l78_7876


namespace NUMINAMATH_CALUDE_max_peak_consumption_theorem_l78_7843

/-- Represents the electricity pricing and consumption parameters for a household. -/
structure ElectricityParams where
  originalPrice : ℝ
  peakPrice : ℝ
  offPeakPrice : ℝ
  totalConsumption : ℝ
  savingsPercentage : ℝ

/-- Calculates the maximum peak hour consumption given electricity parameters. -/
def maxPeakConsumption (params : ElectricityParams) : ℝ := by
  sorry

/-- Theorem stating the maximum peak hour consumption for the given scenario. -/
theorem max_peak_consumption_theorem (params : ElectricityParams) 
  (h1 : params.originalPrice = 0.52)
  (h2 : params.peakPrice = 0.55)
  (h3 : params.offPeakPrice = 0.35)
  (h4 : params.totalConsumption = 200)
  (h5 : params.savingsPercentage = 0.1) :
  maxPeakConsumption params = 118 := by
  sorry

end NUMINAMATH_CALUDE_max_peak_consumption_theorem_l78_7843


namespace NUMINAMATH_CALUDE_cycle_alignment_l78_7801

def letter_cycle_length : ℕ := 5
def digit_cycle_length : ℕ := 4

theorem cycle_alignment :
  Nat.lcm letter_cycle_length digit_cycle_length = 20 := by
  sorry

end NUMINAMATH_CALUDE_cycle_alignment_l78_7801


namespace NUMINAMATH_CALUDE_max_reciprocal_sum_l78_7898

theorem max_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 + y^2 = 1) :
  (1 / x + 1 / y) ≤ 2 * Real.sqrt 2 ∧ 
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀^2 + y₀^2 = 1 ∧ 1 / x₀ + 1 / y₀ = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_reciprocal_sum_l78_7898


namespace NUMINAMATH_CALUDE_distance_to_line_l78_7897

/-- Given a line l with slope k passing through point A(0,2), and a normal vector n to l,
    prove that for any point B satisfying |n⋅AB| = |n|, the distance from B to l is 1. -/
theorem distance_to_line (k : ℝ) (n : ℝ × ℝ) (B : ℝ × ℝ) :
  let A : ℝ × ℝ := (0, 2)
  let l := {(x, y) : ℝ × ℝ | y - 2 = k * x}
  n.1 = -k ∧ n.2 = 1 →  -- n is a normal vector to l
  |n.1 * (B.1 - A.1) + n.2 * (B.2 - A.2)| = Real.sqrt (n.1^2 + n.2^2) →
  Real.sqrt ((B.1 - 0)^2 + (B.2 - (k * B.1 + 2))^2) / Real.sqrt (1 + k^2) = 1 :=
by sorry

end NUMINAMATH_CALUDE_distance_to_line_l78_7897


namespace NUMINAMATH_CALUDE_coeff_x_cubed_expansion_l78_7875

/-- The coefficient of x^3 in the expansion of (x^2 - x + 1)^10 -/
def coeff_x_cubed : ℤ := -210

/-- The binomial coefficient function -/
def binomial (n k : ℕ) : ℕ := sorry

theorem coeff_x_cubed_expansion :
  coeff_x_cubed = binomial 10 8 * binomial 2 1 * (-1) + binomial 10 7 * binomial 3 3 * (-1) :=
sorry

end NUMINAMATH_CALUDE_coeff_x_cubed_expansion_l78_7875


namespace NUMINAMATH_CALUDE_max_chesslike_subsquares_l78_7809

/-- Represents the color of a square on the board -/
inductive Color
| Red
| Green

/-- Represents a 6x6 board -/
def Board := Fin 6 → Fin 6 → Color

/-- Checks if four adjacent squares in a given direction are of the same color -/
def fourAdjacentSameColor (board : Board) : Bool := sorry

/-- Checks if a 2x2 subsquare is chesslike -/
def isChesslike (board : Board) (row col : Fin 5) : Bool := sorry

/-- Counts the number of chesslike 2x2 subsquares on the board -/
def countChesslike (board : Board) : Nat := sorry

/-- Theorem: The maximal number of chesslike 2x2 subsquares on a 6x6 board 
    with the given constraints is 25 -/
theorem max_chesslike_subsquares (board : Board) 
  (h : ¬fourAdjacentSameColor board) : 
  (∃ (b : Board), ¬fourAdjacentSameColor b ∧ countChesslike b = 25) ∧ 
  (∀ (b : Board), ¬fourAdjacentSameColor b → countChesslike b ≤ 25) := by
  sorry

end NUMINAMATH_CALUDE_max_chesslike_subsquares_l78_7809


namespace NUMINAMATH_CALUDE_people_with_banners_l78_7858

/-- Given a stadium with a certain number of seats, prove that the number of people
    holding banners is equal to the number of attendees minus the number of empty seats. -/
theorem people_with_banners (total_seats attendees empty_seats : ℕ) :
  total_seats = 92 →
  attendees = 47 →
  empty_seats = 45 →
  attendees - empty_seats = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_people_with_banners_l78_7858


namespace NUMINAMATH_CALUDE_add_and_round_to_nearest_ten_l78_7847

def round_to_nearest_ten (n : ℤ) : ℤ :=
  10 * ((n + 5) / 10)

theorem add_and_round_to_nearest_ten : round_to_nearest_ten (58 + 29) = 90 := by
  sorry

end NUMINAMATH_CALUDE_add_and_round_to_nearest_ten_l78_7847


namespace NUMINAMATH_CALUDE_rectangle_perimeter_after_increase_l78_7852

/-- Given a rectangle with width 10 meters and original area 150 square meters,
    if its length is increased such that the new area is 4/3 times the original area,
    then the new perimeter is 60 meters. -/
theorem rectangle_perimeter_after_increase (original_length : ℝ) (new_length : ℝ) : 
  original_length * 10 = 150 →
  new_length * 10 = 150 * (4/3) →
  2 * (new_length + 10) = 60 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_after_increase_l78_7852


namespace NUMINAMATH_CALUDE_unique_integer_solution_l78_7829

theorem unique_integer_solution : ∃! (x : ℕ), x > 0 ∧ (3 * x)^2 - x = 2016 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l78_7829


namespace NUMINAMATH_CALUDE_all_pairs_divisible_by_seven_l78_7842

-- Define the type for pairs on the board
def BoardPair := ℤ × ℤ

-- Define the property that 2a - b is divisible by 7
def DivisibleBySeven (p : BoardPair) : Prop :=
  ∃ k : ℤ, 2 * p.1 - p.2 = 7 * k

-- Define the set of all pairs that can appear on the board
inductive ValidPair : BoardPair → Prop where
  | initial : ValidPair (1, 2)
  | negate (a b : ℤ) : ValidPair (a, b) → ValidPair (-a, -b)
  | rotate (a b : ℤ) : ValidPair (a, b) → ValidPair (-b, a + b)
  | add (a b c d : ℤ) : ValidPair (a, b) → ValidPair (c, d) → ValidPair (a + c, b + d)

-- Theorem statement
theorem all_pairs_divisible_by_seven :
  ∀ p : BoardPair, ValidPair p → DivisibleBySeven p :=
  sorry

end NUMINAMATH_CALUDE_all_pairs_divisible_by_seven_l78_7842


namespace NUMINAMATH_CALUDE_valid_field_area_is_189_l78_7877

/-- Represents a rectangular sports field with posts -/
structure SportsField where
  total_posts : ℕ
  post_distance : ℕ
  long_side_posts : ℕ
  short_side_posts : ℕ

/-- Checks if the field configuration is valid according to the problem conditions -/
def is_valid_field (field : SportsField) : Prop :=
  field.total_posts = 24 ∧
  field.post_distance = 3 ∧
  field.long_side_posts = 2 * field.short_side_posts ∧
  2 * (field.long_side_posts + field.short_side_posts - 2) = field.total_posts

/-- Calculates the area of the field given its configuration -/
def field_area (field : SportsField) : ℕ :=
  (field.short_side_posts - 1) * field.post_distance * 
  (field.long_side_posts - 1) * field.post_distance

/-- Theorem stating that a valid field configuration results in an area of 189 square yards -/
theorem valid_field_area_is_189 (field : SportsField) :
  is_valid_field field → field_area field = 189 := by
  sorry

#check valid_field_area_is_189

end NUMINAMATH_CALUDE_valid_field_area_is_189_l78_7877


namespace NUMINAMATH_CALUDE_platform_length_l78_7800

/-- Given a train of length 300 meters, which takes 39 seconds to cross a platform
    and 16 seconds to cross a signal pole, prove that the length of the platform
    is 431.25 meters. -/
theorem platform_length
  (train_length : ℝ)
  (time_cross_platform : ℝ)
  (time_cross_pole : ℝ)
  (h1 : train_length = 300)
  (h2 : time_cross_platform = 39)
  (h3 : time_cross_pole = 16) :
  let speed := train_length / time_cross_pole
  let platform_length := speed * time_cross_platform - train_length
  platform_length = 431.25 := by
sorry

end NUMINAMATH_CALUDE_platform_length_l78_7800


namespace NUMINAMATH_CALUDE_sum_of_m_and_n_l78_7823

theorem sum_of_m_and_n (m n : ℕ) (hm : m > 1) (hn : n > 1) 
  (h : 2005^2 + m^2 = 2004^2 + n^2) : m + n = 211 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_m_and_n_l78_7823


namespace NUMINAMATH_CALUDE_cost_per_foot_metal_roofing_l78_7887

/-- Calculates the cost per foot of metal roofing --/
theorem cost_per_foot_metal_roofing (total_required : ℕ) (free_provided : ℕ) (cost_remaining : ℕ) :
  total_required = 300 →
  free_provided = 250 →
  cost_remaining = 400 →
  (cost_remaining : ℚ) / (total_required - free_provided : ℚ) = 8 := by
  sorry

#check cost_per_foot_metal_roofing

end NUMINAMATH_CALUDE_cost_per_foot_metal_roofing_l78_7887


namespace NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l78_7840

theorem rectangular_to_polar_conversion :
  ∀ (x y r θ : ℝ),
    x = -Real.sqrt 3 →
    y = Real.sqrt 3 →
    r > 0 →
    0 ≤ θ ∧ θ < 2 * Real.pi →
    r = 3 ∧ θ = 3 * Real.pi / 4 →
    x = -r * Real.cos θ ∧
    y = r * Real.sin θ :=
by sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l78_7840


namespace NUMINAMATH_CALUDE_cylinder_section_area_l78_7831

/-- Represents a cylinder -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Represents a plane passing through two points on the top rim of a cylinder and its axis -/
structure CuttingPlane where
  cylinder : Cylinder
  arcAngle : ℝ  -- Angle of the arc PQ in radians

/-- Area of the new section formed when a plane cuts the cylinder -/
def newSectionArea (plane : CuttingPlane) : ℝ := sorry

theorem cylinder_section_area
  (c : Cylinder)
  (p : CuttingPlane)
  (h1 : c.radius = 5)
  (h2 : c.height = 10)
  (h3 : p.cylinder = c)
  (h4 : p.arcAngle = 5 * π / 6)  -- 150° in radians
  : newSectionArea p = 48 * π :=
sorry

end NUMINAMATH_CALUDE_cylinder_section_area_l78_7831


namespace NUMINAMATH_CALUDE_rectangular_box_side_area_l78_7846

theorem rectangular_box_side_area 
  (l w h : ℝ) 
  (front_top : w * h = 0.5 * (l * w))
  (top_side : l * w = 1.5 * (l * h))
  (volume : l * w * h = 3000) :
  l * h = 200 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_box_side_area_l78_7846


namespace NUMINAMATH_CALUDE_max_daily_profit_daily_profit_correct_l78_7855

/-- Represents the daily profit function for a store selling an item --/
def daily_profit (x : ℕ) : ℝ :=
  if x ≤ 30 then -x^2 + 54*x + 640
  else -40*x + 2560

/-- Theorem stating the maximum daily profit and the day it occurs --/
theorem max_daily_profit :
  ∃ (max_profit : ℝ) (max_day : ℕ),
    max_profit = 1369 ∧ 
    max_day = 27 ∧
    (∀ x : ℕ, 1 ≤ x ∧ x ≤ 60 → daily_profit x ≤ max_profit) ∧
    daily_profit max_day = max_profit :=
  sorry

/-- Cost price of the item --/
def cost_price : ℝ := 30

/-- Selling price function --/
def selling_price (x : ℕ) : ℝ :=
  if x ≤ 30 then 0.5 * x + 35
  else 50

/-- Quantity sold function --/
def quantity_sold (x : ℕ) : ℝ := 128 - 2 * x

/-- Verifies that the daily_profit function is correct --/
theorem daily_profit_correct (x : ℕ) (h : 1 ≤ x ∧ x ≤ 60) :
  daily_profit x = (selling_price x - cost_price) * quantity_sold x :=
  sorry

end NUMINAMATH_CALUDE_max_daily_profit_daily_profit_correct_l78_7855


namespace NUMINAMATH_CALUDE_calculation_proof_l78_7891

theorem calculation_proof : (4 - Real.sqrt 3) ^ 0 - 3 * Real.tan (π / 3) - (-1/2)⁻¹ + Real.sqrt 12 = 3 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l78_7891


namespace NUMINAMATH_CALUDE_rock_band_fuel_cost_l78_7818

theorem rock_band_fuel_cost (x : ℝ) :
  (2 * (0.5 * x + 100) + 2 * (0.75 * x + 100) = 550) →
  x = 60 := by
  sorry

end NUMINAMATH_CALUDE_rock_band_fuel_cost_l78_7818


namespace NUMINAMATH_CALUDE_trig_expression_equals_half_l78_7838

theorem trig_expression_equals_half : 
  2 * Real.cos (30 * π / 180) - Real.tan (60 * π / 180) + Real.sin (45 * π / 180) * Real.cos (45 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_half_l78_7838


namespace NUMINAMATH_CALUDE_brahmagupta_theorem_l78_7880

/-- An inscribed quadrilateral with side lengths a, b, c, d and diagonals p, q -/
structure InscribedQuadrilateral (a b c d p q : ℝ) : Prop where
  positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < p ∧ 0 < q
  inscribed : ∃ (r : ℝ), 0 < r ∧ a + c = b + d -- Condition for inscribability

/-- Brahmagupta's theorem for inscribed quadrilaterals -/
theorem brahmagupta_theorem {a b c d p q : ℝ} (quad : InscribedQuadrilateral a b c d p q) :
  p^2 + q^2 = a^2 + b^2 + c^2 + d^2 ∧ 2*p*q = a^2 + c^2 - b^2 - d^2 := by
  sorry

#check brahmagupta_theorem

end NUMINAMATH_CALUDE_brahmagupta_theorem_l78_7880


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l78_7845

/-- Given two planar vectors a and b, where a is perpendicular to b,
    prove that the value of m in a = (m, m-1) and b = (1, 2) is 2/3. -/
theorem perpendicular_vectors_m_value :
  ∀ (m : ℝ),
  let a : ℝ × ℝ := (m, m - 1)
  let b : ℝ × ℝ := (1, 2)
  (a.1 * b.1 + a.2 * b.2 = 0) →  -- dot product = 0 for perpendicular vectors
  m = 2/3 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l78_7845


namespace NUMINAMATH_CALUDE_dance_studios_total_l78_7819

/-- The total number of students in three dance studios -/
def total_students (studio1 studio2 studio3 : ℕ) : ℕ :=
  studio1 + studio2 + studio3

/-- Theorem: The total number of students in three specific dance studios is 376 -/
theorem dance_studios_total : total_students 110 135 131 = 376 := by
  sorry

end NUMINAMATH_CALUDE_dance_studios_total_l78_7819


namespace NUMINAMATH_CALUDE_billys_songs_l78_7892

theorem billys_songs (total_songs : ℕ) (can_play : ℕ) (to_learn : ℕ) :
  total_songs = 52 →
  can_play = 24 →
  to_learn = 28 →
  can_play = total_songs - to_learn :=
by sorry

end NUMINAMATH_CALUDE_billys_songs_l78_7892
