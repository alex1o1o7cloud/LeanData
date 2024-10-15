import Mathlib

namespace NUMINAMATH_CALUDE_min_value_theorem_equality_condition_l3112_311219

theorem min_value_theorem (a : ℝ) (h : a > 0) : 
  2 * a + 1 / a ≥ 2 * Real.sqrt 2 :=
by sorry

theorem equality_condition (a : ℝ) (h : a > 0) : 
  (2 * a + 1 / a = 2 * Real.sqrt 2) ↔ (a = Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_equality_condition_l3112_311219


namespace NUMINAMATH_CALUDE_relationship_one_l3112_311294

theorem relationship_one (a b : ℝ) : (a - b)^2 + (a * b + 1)^2 = (a^2 + 1) * (b^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_relationship_one_l3112_311294


namespace NUMINAMATH_CALUDE_boat_speed_problem_l3112_311235

/-- Proves that given a boat traveling upstream at 3 km/h and having an average
    round-trip speed of 4.2 km/h, its downstream speed is 7 km/h. -/
theorem boat_speed_problem (upstream_speed downstream_speed average_speed : ℝ) 
    (h1 : upstream_speed = 3)
    (h2 : average_speed = 4.2)
    (h3 : average_speed = (2 * upstream_speed * downstream_speed) / (upstream_speed + downstream_speed)) :
  downstream_speed = 7 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_problem_l3112_311235


namespace NUMINAMATH_CALUDE_essay_section_ratio_l3112_311299

theorem essay_section_ratio (total_words introduction_words body_section_words : ℕ)
  (h1 : total_words = 5000)
  (h2 : introduction_words = 450)
  (h3 : body_section_words = 800)
  (h4 : ∃ (k : ℕ), total_words = introduction_words + 4 * body_section_words + k * introduction_words) :
  ∃ (conclusion_words : ℕ), conclusion_words = 3 * introduction_words :=
by sorry

end NUMINAMATH_CALUDE_essay_section_ratio_l3112_311299


namespace NUMINAMATH_CALUDE_system_solution_l3112_311201

theorem system_solution : ∃! (x y : ℝ), 
  (x + Real.sqrt (x + 2*y) - 2*y = 7/2) ∧ 
  (x^2 + x + 2*y - 4*y^2 = 27/2) ∧
  (x = 19/4) ∧ (y = 17/8) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3112_311201


namespace NUMINAMATH_CALUDE_simplify_fraction_l3112_311214

theorem simplify_fraction (a b : ℝ) (h1 : a ≠ -b) (h2 : a ≠ 2*b) :
  (a + 2*b) / (a + b) - (a - b) / (a - 2*b) / ((a^2 - b^2) / (a^2 - 4*a*b + 4*b^2)) = 4*b / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3112_311214


namespace NUMINAMATH_CALUDE_thread_needed_proof_l3112_311209

def thread_per_keychain : ℕ := 12
def friends_from_classes : ℕ := 6
def friends_from_clubs : ℕ := friends_from_classes / 2

def total_friends : ℕ := friends_from_classes + friends_from_clubs

theorem thread_needed_proof : 
  thread_per_keychain * total_friends = 108 := by
  sorry

end NUMINAMATH_CALUDE_thread_needed_proof_l3112_311209


namespace NUMINAMATH_CALUDE_inverse_proportion_comparison_l3112_311286

/-- An inverse proportion function passing through (-2, 4) with points (1, y₁) and (3, y₂) on its graph -/
def InverseProportion (k : ℝ) (y₁ y₂ : ℝ) : Prop :=
  k ≠ 0 ∧ 
  4 = k / (-2) ∧ 
  y₁ = k / 1 ∧ 
  y₂ = k / 3

theorem inverse_proportion_comparison (k : ℝ) (y₁ y₂ : ℝ) 
  (h : InverseProportion k y₁ y₂) : y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_comparison_l3112_311286


namespace NUMINAMATH_CALUDE_pencils_per_row_l3112_311258

theorem pencils_per_row (total_pencils : ℕ) (num_rows : ℕ) (pencils_per_row : ℕ) : 
  total_pencils = 25 → num_rows = 5 → total_pencils = num_rows * pencils_per_row → pencils_per_row = 5 := by
  sorry

end NUMINAMATH_CALUDE_pencils_per_row_l3112_311258


namespace NUMINAMATH_CALUDE_quadratic_no_fixed_points_l3112_311292

/-- A quadratic function f(x) = x^2 + ax + 1 has no fixed points if and only if -1 < a < 3 -/
theorem quadratic_no_fixed_points (a : ℝ) :
  (∀ x : ℝ, x^2 + a*x + 1 ≠ x) ↔ -1 < a ∧ a < 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_fixed_points_l3112_311292


namespace NUMINAMATH_CALUDE_hyperbola_focus_smaller_x_l3112_311262

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  center : Point

/-- Returns true if the given point is a focus of the hyperbola -/
def isFocus (h : Hyperbola) (p : Point) : Prop :=
  let c := Real.sqrt (h.a^2 + h.b^2)
  (p.x = h.center.x - c ∧ p.y = h.center.y) ∨
  (p.x = h.center.x + c ∧ p.y = h.center.y)

/-- Returns true if p1 has a smaller x-coordinate than p2 -/
def hasSmaller_x (p1 p2 : Point) : Prop :=
  p1.x < p2.x

theorem hyperbola_focus_smaller_x (h : Hyperbola) :
  h.a = 7 ∧ h.b = 3 ∧ h.center = { x := 1, y := -8 } →
  ∃ (f : Point), isFocus h f ∧ ∀ (f' : Point), isFocus h f' → hasSmaller_x f f' ∨ f = f' →
  f = { x := 1 - Real.sqrt 58, y := -8 } := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_focus_smaller_x_l3112_311262


namespace NUMINAMATH_CALUDE_marks_spending_l3112_311290

-- Define constants for item quantities
def notebooks : ℕ := 4
def pens : ℕ := 3
def books : ℕ := 1
def magazines : ℕ := 2

-- Define prices
def notebook_price : ℚ := 2
def pen_price : ℚ := 1.5
def book_price : ℚ := 12
def magazine_original_price : ℚ := 3

-- Define discount and coupon
def magazine_discount : ℚ := 0.25
def coupon_value : ℚ := 3
def coupon_threshold : ℚ := 20

-- Calculate discounted magazine price
def discounted_magazine_price : ℚ := magazine_original_price * (1 - magazine_discount)

-- Calculate total cost before coupon
def total_before_coupon : ℚ :=
  notebooks * notebook_price +
  pens * pen_price +
  books * book_price +
  magazines * discounted_magazine_price

-- Apply coupon if total is over the threshold
def final_total : ℚ :=
  if total_before_coupon ≥ coupon_threshold
  then total_before_coupon - coupon_value
  else total_before_coupon

-- Theorem to prove
theorem marks_spending :
  final_total = 26 := by sorry

end NUMINAMATH_CALUDE_marks_spending_l3112_311290


namespace NUMINAMATH_CALUDE_correct_calculation_l3112_311207

theorem correct_calculation (x y : ℝ) : 3 * x - (-2 * y + 4) = 3 * x + 2 * y - 4 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3112_311207


namespace NUMINAMATH_CALUDE_remainder_3_800_mod_17_l3112_311250

theorem remainder_3_800_mod_17 : 3^800 % 17 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_800_mod_17_l3112_311250


namespace NUMINAMATH_CALUDE_lemonade_percentage_in_solution1_l3112_311208

/-- Represents a solution mixture of lemonade and carbonated water -/
structure Solution :=
  (lemonade : ℝ)
  (carbonated_water : ℝ)
  (h_sum : lemonade + carbonated_water = 100)

/-- Represents a mixture of two solutions -/
structure Mixture :=
  (solution1 : Solution)
  (solution2 : Solution)
  (proportion1 : ℝ)
  (proportion2 : ℝ)
  (h_prop_sum : proportion1 + proportion2 = 100)

theorem lemonade_percentage_in_solution1
  (s1 : Solution)
  (s2 : Solution)
  (mix : Mixture)
  (h1 : s2.lemonade = 45)
  (h2 : s2.carbonated_water = 55)
  (h3 : mix.solution1 = s1)
  (h4 : mix.solution2 = s2)
  (h5 : mix.proportion1 = 40)
  (h6 : mix.proportion2 = 60)
  (h7 : mix.proportion1 / 100 * s1.carbonated_water + mix.proportion2 / 100 * s2.carbonated_water = 65) :
  s1.lemonade = 20 := by
sorry

end NUMINAMATH_CALUDE_lemonade_percentage_in_solution1_l3112_311208


namespace NUMINAMATH_CALUDE_basketball_team_combinations_l3112_311267

/-- The number of players in the basketball team -/
def total_players : ℕ := 12

/-- The number of players in the starting lineup (excluding the captain) -/
def starting_lineup : ℕ := 5

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Theorem: The number of ways to select 1 captain from 12 players and then 5 players 
    from the remaining 11 for the starting lineup is equal to 5544 -/
theorem basketball_team_combinations : 
  total_players * choose (total_players - 1) starting_lineup = 5544 := by
  sorry


end NUMINAMATH_CALUDE_basketball_team_combinations_l3112_311267


namespace NUMINAMATH_CALUDE_sparse_characterization_l3112_311270

/-- A number s grows to r if there exists some integer n > 0 such that s^n = r -/
def GrowsTo (s r : ℝ) : Prop :=
  ∃ n : ℕ, n > 0 ∧ s^n = r

/-- A real number r is sparse if there are only finitely many real numbers s that grow to r -/
def Sparse (r : ℝ) : Prop :=
  Set.Finite {s : ℝ | GrowsTo s r}

/-- The characterization of sparse real numbers -/
theorem sparse_characterization (r : ℝ) : Sparse r ↔ r = -1 ∨ r = 0 ∨ r = 1 := by
  sorry

end NUMINAMATH_CALUDE_sparse_characterization_l3112_311270


namespace NUMINAMATH_CALUDE_parallelogram_angle_l3112_311200

/-- 
Given a parallelogram with the following properties:
- One angle exceeds the other by 40 degrees
- An inscribed circle touches the extended line of the smaller angle
- This touch point forms a triangle exterior to the parallelogram
- The angle at this point is 60 degrees less than double the smaller angle

Prove that the smaller angle of the parallelogram is 70 degrees.
-/
theorem parallelogram_angle (x : ℝ) : 
  x > 0 ∧ 
  x + 40 > x ∧
  x + (x + 40) = 180 ∧
  2 * x - 60 > 0 → 
  x = 70 := by sorry

end NUMINAMATH_CALUDE_parallelogram_angle_l3112_311200


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3112_311223

/-- Given a linear function y = cx + 2c, prove that the quadratic function
    y = 0.5c(x + 2)^2 passes through the points (0, 2c) and (-2, 0) -/
theorem quadratic_function_properties (c : ℝ) :
  let f (x : ℝ) := 0.5 * c * (x + 2)^2
  (f 0 = 2 * c) ∧ (f (-2) = 0) := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3112_311223


namespace NUMINAMATH_CALUDE_abs_neg_two_equals_two_l3112_311217

theorem abs_neg_two_equals_two : |-2| = 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_two_equals_two_l3112_311217


namespace NUMINAMATH_CALUDE_project_time_ratio_l3112_311230

/-- Proves that the ratio of time charged by Pat to Kate is 2:1 given the problem conditions -/
theorem project_time_ratio : 
  ∀ (p k m : ℕ) (r : ℚ),
  p + k + m = 153 →
  p = r * k →
  p = m / 3 →
  m = k + 85 →
  r = 2 := by
sorry

end NUMINAMATH_CALUDE_project_time_ratio_l3112_311230


namespace NUMINAMATH_CALUDE_mini_bank_withdrawal_l3112_311279

theorem mini_bank_withdrawal (d c : ℕ) : 
  (0 < c) → (c < 100) →
  (100 * c + d - 350 = 2 * (100 * d + c)) →
  (d = 14 ∧ c = 32) := by
sorry

end NUMINAMATH_CALUDE_mini_bank_withdrawal_l3112_311279


namespace NUMINAMATH_CALUDE_positive_operation_on_negative_two_l3112_311287

theorem positive_operation_on_negative_two (op : ℝ → ℝ → ℝ) : 
  (op 1 (-2) > 0) → (1 - (-2) > 0) :=
by sorry

end NUMINAMATH_CALUDE_positive_operation_on_negative_two_l3112_311287


namespace NUMINAMATH_CALUDE_absent_fraction_proof_l3112_311274

/-- Proves that if work increases by 1/6 when a fraction of members are absent,
    then the fraction of absent members is 1/7 -/
theorem absent_fraction_proof (p : ℕ) (p_pos : p > 0) :
  let increase_factor : ℚ := 1 / 6
  let absent_fraction : ℚ := 1 / 7
  (1 : ℚ) + increase_factor = 1 / (1 - absent_fraction) :=
by sorry

end NUMINAMATH_CALUDE_absent_fraction_proof_l3112_311274


namespace NUMINAMATH_CALUDE_exchange_result_l3112_311255

/-- Represents the number of exchanges between Xiao Zhang and Xiao Li -/
def exchanges : ℕ := 4

/-- Xiao Zhang's initial number of pencils -/
def zhang_initial_pencils : ℕ := 200

/-- Xiao Li's initial number of pens -/
def li_initial_pens : ℕ := 20

/-- Number of pencils Xiao Zhang gives in each exchange -/
def pencils_per_exchange : ℕ := 6

/-- Number of pens Xiao Li gives in each exchange -/
def pens_per_exchange : ℕ := 1

/-- Xiao Zhang's pencils after exchanges -/
def zhang_final_pencils : ℕ := zhang_initial_pencils - exchanges * pencils_per_exchange

/-- Xiao Li's pens after exchanges -/
def li_final_pens : ℕ := li_initial_pens - exchanges * pens_per_exchange

theorem exchange_result : zhang_final_pencils = 11 * li_final_pens := by
  sorry

end NUMINAMATH_CALUDE_exchange_result_l3112_311255


namespace NUMINAMATH_CALUDE_equal_angle_slope_value_l3112_311236

/-- The slope of a line that forms equal angles with y = x and y = 2x --/
def equal_angle_slope : ℝ → Prop := λ k =>
  let l₁ : ℝ → ℝ := λ x => x
  let l₂ : ℝ → ℝ := λ x => 2 * x
  let angle (m₁ m₂ : ℝ) : ℝ := |((m₂ - m₁) / (1 + m₁ * m₂))|
  (angle k 1 = angle 2 k) ∧ (3 * k^2 - 2 * k - 3 = 0)

/-- The slope of a line that forms equal angles with y = x and y = 2x
    is (1 ± √10) / 3 --/
theorem equal_angle_slope_value :
  ∃ k : ℝ, equal_angle_slope k ∧ (k = (1 + Real.sqrt 10) / 3 ∨ k = (1 - Real.sqrt 10) / 3) :=
sorry

end NUMINAMATH_CALUDE_equal_angle_slope_value_l3112_311236


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3112_311280

theorem complex_number_in_first_quadrant : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (1 / (1 + Complex.I) + Complex.I) = Complex.mk a b := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3112_311280


namespace NUMINAMATH_CALUDE_division_expression_equality_l3112_311238

theorem division_expression_equality : 
  (1 : ℚ) / 12 / ((1 : ℚ) / 3 - (1 : ℚ) / 4 - (5 : ℚ) / 12) = -(1 : ℚ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_division_expression_equality_l3112_311238


namespace NUMINAMATH_CALUDE_sum_of_N_and_K_is_8_l3112_311283

/-- The complex conjugate of a complex number -/
noncomputable def conj (z : ℂ) : ℂ := sorry

/-- The transformation function g -/
noncomputable def g (z : ℂ) : ℂ := 2 * Complex.I * conj z

/-- The polynomial P -/
def P (z : ℂ) : ℂ := z^4 + 6*z^3 + 2*z^2 + 4*z + 1

/-- The roots of P -/
noncomputable def z1 : ℂ := sorry
noncomputable def z2 : ℂ := sorry
noncomputable def z3 : ℂ := sorry
noncomputable def z4 : ℂ := sorry

/-- The polynomial R -/
noncomputable def R (z : ℂ) : ℂ := z^4 + M*z^3 + N*z^2 + L*z + K
  where
  M : ℂ := sorry
  N : ℂ := sorry
  L : ℂ := sorry
  K : ℂ := sorry

theorem sum_of_N_and_K_is_8 : N + K = 8 := by sorry

end NUMINAMATH_CALUDE_sum_of_N_and_K_is_8_l3112_311283


namespace NUMINAMATH_CALUDE_min_value_theorem_l3112_311229

theorem min_value_theorem (x y N : ℝ) : 
  (x + 4) * (y - 4) = N → 
  (∀ a b : ℝ, a^2 + b^2 ≥ x^2 + y^2) → 
  x^2 + y^2 = 16 → 
  N = 0 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3112_311229


namespace NUMINAMATH_CALUDE_blueberry_carton_size_l3112_311251

/-- The number of ounces in a carton of blueberries -/
def blueberry_carton_ounces : ℝ := 6

/-- The cost of a carton of blueberries in dollars -/
def blueberry_carton_cost : ℝ := 5

/-- The cost of a carton of raspberries in dollars -/
def raspberry_carton_cost : ℝ := 3

/-- The number of ounces in a carton of raspberries -/
def raspberry_carton_ounces : ℝ := 8

/-- The number of batches of muffins being made -/
def num_batches : ℝ := 4

/-- The number of ounces of fruit required per batch -/
def ounces_per_batch : ℝ := 12

/-- The amount saved by using raspberries instead of blueberries -/
def amount_saved : ℝ := 22

theorem blueberry_carton_size :
  blueberry_carton_ounces = 6 :=
sorry

end NUMINAMATH_CALUDE_blueberry_carton_size_l3112_311251


namespace NUMINAMATH_CALUDE_tutors_next_meeting_l3112_311297

/-- Anthony's work schedule in days -/
def anthony : ℕ := 5

/-- Beth's work schedule in days -/
def beth : ℕ := 6

/-- Carlos's work schedule in days -/
def carlos : ℕ := 8

/-- Diana's work schedule in days -/
def diana : ℕ := 10

/-- The number of days until all tutors work together again -/
def next_meeting : ℕ := 120

theorem tutors_next_meeting :
  Nat.lcm anthony (Nat.lcm beth (Nat.lcm carlos diana)) = next_meeting := by
  sorry

end NUMINAMATH_CALUDE_tutors_next_meeting_l3112_311297


namespace NUMINAMATH_CALUDE_complex_distance_l3112_311273

theorem complex_distance (z₁ z₂ : ℂ) 
  (h₁ : Complex.abs (z₁ + z₂) = 2 * Real.sqrt 2)
  (h₂ : Complex.abs z₁ = Real.sqrt 3)
  (h₃ : Complex.abs z₂ = Real.sqrt 2) :
  Complex.abs (z₁ - z₂) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_distance_l3112_311273


namespace NUMINAMATH_CALUDE_exponential_graph_not_in_second_quadrant_l3112_311216

/-- Given a > 1 and b < -1, the graph of y = a^x + b does not intersect the second quadrant -/
theorem exponential_graph_not_in_second_quadrant 
  (a b : ℝ) (ha : a > 1) (hb : b < -1) :
  ∀ x y : ℝ, y = a^x + b → ¬(x < 0 ∧ y > 0) :=
by sorry

end NUMINAMATH_CALUDE_exponential_graph_not_in_second_quadrant_l3112_311216


namespace NUMINAMATH_CALUDE_largest_divisor_of_five_consecutive_integers_l3112_311260

theorem largest_divisor_of_five_consecutive_integers : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (k : ℤ), (k * (k+1) * (k+2) * (k+3) * (k+4)) % n = 0) ∧
  (∀ (m : ℕ), m > n → ∃ (l : ℤ), (l * (l+1) * (l+2) * (l+3) * (l+4)) % m ≠ 0) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_divisor_of_five_consecutive_integers_l3112_311260


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3112_311206

/-- Simplification of polynomial expression -/
theorem polynomial_simplification (x : ℝ) :
  (3 * x^10 + 5 * x^9 + 2 * x^8) + (7 * x^12 - x^10 + 4 * x^9 + x^7 + 6 * x^4 + 9) =
  7 * x^12 + 2 * x^10 + 9 * x^9 + 2 * x^8 + x^7 + 6 * x^4 + 9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3112_311206


namespace NUMINAMATH_CALUDE_equation_solution_l3112_311233

theorem equation_solution : ∃! y : ℝ, 5 * y - 100 = 125 ∧ y = 45 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3112_311233


namespace NUMINAMATH_CALUDE_simple_interest_rate_calculation_l3112_311291

theorem simple_interest_rate_calculation (principal : ℝ) (h : principal > 0) :
  let final_amount := (7 / 6 : ℝ) * principal
  let time := 4
  let interest := final_amount - principal
  let rate := (interest / (principal * time)) * 100
  rate = 100 / 24 := by sorry

end NUMINAMATH_CALUDE_simple_interest_rate_calculation_l3112_311291


namespace NUMINAMATH_CALUDE_matrix_multiplication_example_l3112_311252

theorem matrix_multiplication_example :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 0; 0, 3]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![2, 1; -1, 0]
  A * B = !![4, 2; -3, 0] := by sorry

end NUMINAMATH_CALUDE_matrix_multiplication_example_l3112_311252


namespace NUMINAMATH_CALUDE_student_sampling_interval_l3112_311203

/-- Calculates the interval for systematic sampling -/
def systematicSamplingInterval (populationSize sampleSize : ℕ) : ℕ :=
  populationSize / sampleSize

/-- Theorem: The systematic sampling interval for 1000 students with a sample size of 40 is 25 -/
theorem student_sampling_interval :
  systematicSamplingInterval 1000 40 = 25 := by
  sorry

end NUMINAMATH_CALUDE_student_sampling_interval_l3112_311203


namespace NUMINAMATH_CALUDE_small_sphere_radius_l3112_311263

/-- Represents a sphere in 3D space -/
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

/-- Checks if two spheres are externally tangent -/
def are_externally_tangent (s1 s2 : Sphere) : Prop :=
  let (x1, y1, z1) := s1.center
  let (x2, y2, z2) := s2.center
  (x1 - x2)^2 + (y1 - y2)^2 + (z1 - z2)^2 = (s1.radius + s2.radius)^2

/-- The configuration of 5 spheres as described in the problem -/
structure SpheresConfiguration where
  s1 : Sphere
  s2 : Sphere
  s3 : Sphere
  s4 : Sphere
  small : Sphere
  h1 : s1.radius = 2
  h2 : s2.radius = 2
  h3 : s3.radius = 3
  h4 : s4.radius = 3
  h5 : are_externally_tangent s1 s2
  h6 : are_externally_tangent s1 s3
  h7 : are_externally_tangent s1 s4
  h8 : are_externally_tangent s2 s3
  h9 : are_externally_tangent s2 s4
  h10 : are_externally_tangent s3 s4
  h11 : are_externally_tangent s1 small
  h12 : are_externally_tangent s2 small
  h13 : are_externally_tangent s3 small
  h14 : are_externally_tangent s4 small

/-- The main theorem stating that the radius of the small sphere is 6/11 -/
theorem small_sphere_radius (config : SpheresConfiguration) : config.small.radius = 6/11 := by
  sorry

end NUMINAMATH_CALUDE_small_sphere_radius_l3112_311263


namespace NUMINAMATH_CALUDE_rectangle_exists_in_octagon_decomposition_l3112_311215

/-- A regular octagon -/
structure RegularOctagon where
  -- Add necessary fields

/-- A parallelogram -/
structure Parallelogram where
  -- Add necessary fields

/-- A decomposition of a regular octagon into parallelograms -/
structure OctagonDecomposition where
  octagon : RegularOctagon
  parallelograms : Finset Parallelogram
  is_valid : Bool  -- Predicate to check if the decomposition is valid

/-- Predicate to check if a parallelogram is a rectangle -/
def is_rectangle (p : Parallelogram) : Prop :=
  sorry

/-- Main theorem: In any valid decomposition of a regular octagon into parallelograms,
    there exists at least one rectangle among the parallelograms -/
theorem rectangle_exists_in_octagon_decomposition (d : OctagonDecomposition) 
    (h : d.is_valid) : ∃ p ∈ d.parallelograms, is_rectangle p :=
  sorry

end NUMINAMATH_CALUDE_rectangle_exists_in_octagon_decomposition_l3112_311215


namespace NUMINAMATH_CALUDE_unique_point_exists_l3112_311288

-- Define the circle
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 ≤ 4}

-- Define the diameter endpoints
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)

-- Define the conditions for point P
def IsValidP (p : ℝ × ℝ) : Prop :=
  p ∈ Circle ∧
  (p.1 - A.1)^2 + (p.2 - A.2)^2 + (p.1 - B.1)^2 + (p.2 - B.2)^2 = 10 ∧
  Real.cos (Real.arccos ((p.1 - A.1) * (p.1 - B.1) + (p.2 - A.2) * (p.2 - B.2)) /
    (Real.sqrt ((p.1 - A.1)^2 + (p.2 - A.2)^2) * Real.sqrt ((p.1 - B.1)^2 + (p.2 - B.2)^2))) = 1/2

theorem unique_point_exists : ∃! p, IsValidP p :=
  sorry

end NUMINAMATH_CALUDE_unique_point_exists_l3112_311288


namespace NUMINAMATH_CALUDE_polynomial_coefficients_l3112_311277

/-- The polynomial f(x) = ax^4 - 7x^3 + bx^2 - 12x - 8 -/
def f (a b x : ℝ) : ℝ := a * x^4 - 7 * x^3 + b * x^2 - 12 * x - 8

/-- Theorem stating that if f(2) = -7 and f(-3) = -80, then a = -9/4 and b = 29.25 -/
theorem polynomial_coefficients (a b : ℝ) :
  f a b 2 = -7 ∧ f a b (-3) = -80 → a = -9/4 ∧ b = 29.25 := by
  sorry

#check polynomial_coefficients

end NUMINAMATH_CALUDE_polynomial_coefficients_l3112_311277


namespace NUMINAMATH_CALUDE_tan_four_theta_l3112_311289

theorem tan_four_theta (θ : Real) (h : Real.tan θ = 3) : Real.tan (4 * θ) = -24 / 7 := by
  sorry

end NUMINAMATH_CALUDE_tan_four_theta_l3112_311289


namespace NUMINAMATH_CALUDE_system_solution_l3112_311226

theorem system_solution :
  ∃ (x y z : ℝ),
    (1 / x + 2 / y - 3 / z = 3) ∧
    (4 / x - 1 / y - 2 / z = 5) ∧
    (3 / x + 4 / y + 1 / z = 23) ∧
    (x = 1 / 3) ∧ (y = 1 / 3) ∧ (z = 1 / 2) :=
by
  use 1/3, 1/3, 1/2
  sorry

#check system_solution

end NUMINAMATH_CALUDE_system_solution_l3112_311226


namespace NUMINAMATH_CALUDE_felix_betty_length_difference_l3112_311218

-- Define the given constants
def betty_steps_per_gap : ℕ := 36
def felix_jumps_per_gap : ℕ := 9
def total_posts : ℕ := 51
def total_distance : ℝ := 7920

-- Define the theorem
theorem felix_betty_length_difference :
  let total_gaps := total_posts - 1
  let betty_total_steps := betty_steps_per_gap * total_gaps
  let felix_total_jumps := felix_jumps_per_gap * total_gaps
  let betty_step_length := total_distance / betty_total_steps
  let felix_jump_length := total_distance / felix_total_jumps
  felix_jump_length - betty_step_length = 13.2 := by
sorry

end NUMINAMATH_CALUDE_felix_betty_length_difference_l3112_311218


namespace NUMINAMATH_CALUDE_monochromatic_square_exists_l3112_311244

/-- A color type with two possible values -/
inductive Color
  | Red
  | Blue

/-- A point in the 2D grid -/
structure Point where
  x : Nat
  y : Nat
  h_x : x ≥ 1 ∧ x ≤ 5
  h_y : y ≥ 1 ∧ y ≤ 5

/-- A coloring of the 5x5 grid -/
def Coloring := Point → Color

/-- Check if four points form a square with sides parallel to the axes -/
def isSquare (p1 p2 p3 p4 : Point) : Prop :=
  ∃ k : Nat, k > 0 ∧
    ((p1.x + k = p2.x ∧ p1.y = p2.y ∧
      p2.x = p3.x ∧ p2.y + k = p3.y ∧
      p3.x - k = p4.x ∧ p3.y = p4.y ∧
      p4.x = p1.x ∧ p4.y + k = p1.y) ∨
     (p1.y + k = p2.y ∧ p1.x = p2.x ∧
      p2.y = p3.y ∧ p2.x + k = p3.x ∧
      p3.y - k = p4.y ∧ p3.x = p4.x ∧
      p4.y = p1.y ∧ p4.x + k = p1.x))

/-- The main theorem -/
theorem monochromatic_square_exists (c : Coloring) :
  ∃ p1 p2 p3 p4 : Point,
    isSquare p1 p2 p3 p4 ∧
    (c p1 = c p2 ∧ c p2 = c p3 ∨
     c p1 = c p2 ∧ c p2 = c p4 ∨
     c p1 = c p3 ∧ c p3 = c p4 ∨
     c p2 = c p3 ∧ c p3 = c p4) := by
  sorry

end NUMINAMATH_CALUDE_monochromatic_square_exists_l3112_311244


namespace NUMINAMATH_CALUDE_luncheon_invitees_l3112_311254

/-- The number of people who didn't show up to the luncheon -/
def no_shows : ℕ := 50

/-- The number of people each table can hold -/
def people_per_table : ℕ := 3

/-- The number of tables needed for the people who showed up -/
def tables_used : ℕ := 6

/-- The total number of people originally invited to the luncheon -/
def total_invited : ℕ := no_shows + people_per_table * tables_used + 1

/-- Theorem stating that the number of people originally invited to the luncheon is 101 -/
theorem luncheon_invitees : total_invited = 101 := by
  sorry

end NUMINAMATH_CALUDE_luncheon_invitees_l3112_311254


namespace NUMINAMATH_CALUDE_hexagon_reachability_l3112_311211

def Hexagon := Fin 6 → ℤ

def initial_hexagon : Hexagon := ![12, 1, 10, 6, 8, 3]

def is_valid_move (h1 h2 : Hexagon) : Prop :=
  ∃ i : Fin 6, 
    (h2 i = h1 i + 1 ∧ h2 ((i + 1) % 6) = h1 ((i + 1) % 6) + 1) ∨
    (h2 i = h1 i - 1 ∧ h2 ((i + 1) % 6) = h1 ((i + 1) % 6) - 1) ∧
    ∀ j : Fin 6, j ≠ i ∧ j ≠ (i + 1) % 6 → h2 j = h1 j

def is_reachable (start goal : Hexagon) : Prop :=
  ∃ (n : ℕ) (sequence : Fin (n + 1) → Hexagon),
    sequence 0 = start ∧
    sequence n = goal ∧
    ∀ i : Fin n, is_valid_move (sequence i) (sequence (i + 1))

theorem hexagon_reachability :
  (is_reachable initial_hexagon ![14, 6, 13, 4, 5, 2]) ∧
  ¬(is_reachable initial_hexagon ![6, 17, 14, 3, 15, 2]) := by
  sorry

end NUMINAMATH_CALUDE_hexagon_reachability_l3112_311211


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_l3112_311221

theorem gcd_lcm_sum : Nat.gcd 45 75 + Nat.lcm 24 18 = 87 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_l3112_311221


namespace NUMINAMATH_CALUDE_previous_salary_calculation_l3112_311210

/-- Represents the salary and commission structure of Tom's new job -/
structure NewJob where
  base_salary : ℝ
  commission_rate : ℝ
  sale_value : ℝ

/-- Calculates the total earnings from the new job given a number of sales -/
def earnings_new_job (job : NewJob) (num_sales : ℝ) : ℝ :=
  job.base_salary + job.commission_rate * job.sale_value * num_sales

/-- Theorem stating that if Tom needs to make at least 266.67 sales to not lose money,
    then his previous job salary was $75,000 -/
theorem previous_salary_calculation (job : NewJob) 
    (h1 : job.base_salary = 45000)
    (h2 : job.commission_rate = 0.15)
    (h3 : job.sale_value = 750)
    (h4 : earnings_new_job job 266.67 ≥ earnings_new_job job 266.66) :
    earnings_new_job job 266.67 = 75000 := by
  sorry

#check previous_salary_calculation

end NUMINAMATH_CALUDE_previous_salary_calculation_l3112_311210


namespace NUMINAMATH_CALUDE_handshake_count_l3112_311225

theorem handshake_count (n : ℕ) (h : n = 9) : (n * (n - 1)) / 2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_handshake_count_l3112_311225


namespace NUMINAMATH_CALUDE_max_intersection_points_for_circles_l3112_311266

/-- The maximum number of intersection points for n circles in a plane -/
def max_intersection_points (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: Given n circles in a plane, where n ≥ 2, that intersect each other pairwise,
    the maximum number of intersection points is n(n-1). -/
theorem max_intersection_points_for_circles (n : ℕ) (h : n ≥ 2) :
  max_intersection_points n = n * (n - 1) :=
by sorry

end NUMINAMATH_CALUDE_max_intersection_points_for_circles_l3112_311266


namespace NUMINAMATH_CALUDE_exist_four_lines_eight_regions_l3112_311242

/-- A line in the coordinate plane defined by y = kx + b --/
structure Line where
  k : ℕ
  b : ℕ
  k_in_range : k ∈ Finset.range 9 \ {0}
  b_in_range : b ∈ Finset.range 9 \ {0}

/-- The set of four lines --/
def FourLines : Type := Fin 4 → Line

/-- All coefficients and constants are distinct --/
def all_distinct (lines : FourLines) : Prop :=
  ∀ i j, i ≠ j → lines i ≠ lines j

/-- The number of regions formed by the lines --/
def num_regions (lines : FourLines) : ℕ := sorry

/-- Theorem: There exist 4 lines that divide the plane into 8 regions --/
theorem exist_four_lines_eight_regions :
  ∃ (lines : FourLines), all_distinct lines ∧ num_regions lines = 8 := by sorry

end NUMINAMATH_CALUDE_exist_four_lines_eight_regions_l3112_311242


namespace NUMINAMATH_CALUDE_max_blue_chips_l3112_311275

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem max_blue_chips 
  (total : ℕ) 
  (red : ℕ) 
  (blue : ℕ) 
  (h_total : total = 72)
  (h_sum : red + blue = total)
  (h_prime : ∃ p : ℕ, is_prime p ∧ red = blue + p) :
  blue ≤ 35 ∧ ∃ blue_max : ℕ, blue_max = 35 ∧ 
    ∃ red_max : ℕ, ∃ p_min : ℕ, 
      is_prime p_min ∧ 
      red_max + blue_max = total ∧ 
      red_max = blue_max + p_min :=
sorry

end NUMINAMATH_CALUDE_max_blue_chips_l3112_311275


namespace NUMINAMATH_CALUDE_table_size_lower_bound_l3112_311234

/-- Represents a table with 10 columns and n rows, where each cell contains a digit. -/
structure DigitTable (n : ℕ) :=
  (rows : Fin n → Fin 10 → Fin 10)

/-- 
Given a table with 10 columns and n rows, where each cell contains a digit, 
and for any row A and any two columns, there exists a row that differs from A 
in exactly these two columns, prove that n ≥ 512.
-/
theorem table_size_lower_bound {n : ℕ} (t : DigitTable n) 
  (h : ∀ (A : Fin n) (i j : Fin 10), i ≠ j → 
    ∃ (B : Fin n), (∀ k : Fin 10, k ≠ i ∧ k ≠ j → t.rows A k = t.rows B k) ∧
                   t.rows A i ≠ t.rows B i ∧ 
                   t.rows A j ≠ t.rows B j) : 
  n ≥ 512 := by
  sorry

end NUMINAMATH_CALUDE_table_size_lower_bound_l3112_311234


namespace NUMINAMATH_CALUDE_circular_tank_properties_l3112_311202

theorem circular_tank_properties (AB DC : ℝ) (h1 : AB = 20) (h2 : DC = 12) :
  let r := (AB / 2)^2 + DC^2
  (π * r = 244 * π) ∧ (2 * π * Real.sqrt r = 2 * π * Real.sqrt 244) := by
  sorry

end NUMINAMATH_CALUDE_circular_tank_properties_l3112_311202


namespace NUMINAMATH_CALUDE_sequence_properties_l3112_311239

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ) : ℤ := 33 * n - n^2

/-- The n-th term of the sequence -/
def a (n : ℕ) : ℤ := 34 - 2 * n

theorem sequence_properties :
  (∀ n : ℕ, n ≥ 1 → a n = S n - S (n-1)) ∧
  (a 1 = 32) ∧
  (∀ n : ℕ, a (n+1) - a n = -2) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l3112_311239


namespace NUMINAMATH_CALUDE_project_B_highest_score_l3112_311243

structure Project where
  name : String
  innovation : ℝ
  practicality : ℝ

def totalScore (p : Project) : ℝ :=
  0.6 * p.innovation + 0.4 * p.practicality

def projectA : Project := ⟨"A", 90, 90⟩
def projectB : Project := ⟨"B", 95, 90⟩
def projectC : Project := ⟨"C", 90, 95⟩
def projectD : Project := ⟨"D", 90, 85⟩

def projects : List Project := [projectA, projectB, projectC, projectD]

theorem project_B_highest_score :
  ∀ p ∈ projects, p ≠ projectB → totalScore p ≤ totalScore projectB :=
sorry

end NUMINAMATH_CALUDE_project_B_highest_score_l3112_311243


namespace NUMINAMATH_CALUDE_roller_coaster_cars_l3112_311249

theorem roller_coaster_cars (people_in_line : ℕ) (people_per_car : ℕ) (num_runs : ℕ) 
  (h1 : people_in_line = 84)
  (h2 : people_per_car = 2)
  (h3 : num_runs = 6)
  (h4 : people_in_line = num_runs * (num_cars * people_per_car)) :
  num_cars = 7 :=
by sorry

end NUMINAMATH_CALUDE_roller_coaster_cars_l3112_311249


namespace NUMINAMATH_CALUDE_farmer_land_calculation_l3112_311220

/-- Represents the total land owned by the farmer in acres -/
def total_land : ℝ := 7000

/-- Represents the proportion of land that was cleared for planting -/
def cleared_proportion : ℝ := 0.90

/-- Represents the proportion of cleared land planted with potato -/
def potato_proportion : ℝ := 0.20

/-- Represents the proportion of cleared land planted with tomato -/
def tomato_proportion : ℝ := 0.70

/-- Represents the amount of cleared land planted with corn in acres -/
def corn_land : ℝ := 630

theorem farmer_land_calculation :
  total_land * cleared_proportion * (potato_proportion + tomato_proportion) + corn_land = 
  total_land * cleared_proportion := by sorry

end NUMINAMATH_CALUDE_farmer_land_calculation_l3112_311220


namespace NUMINAMATH_CALUDE_greatest_of_three_consecutive_integers_l3112_311272

theorem greatest_of_three_consecutive_integers (x : ℤ) :
  (x + (x + 1) + (x + 2) = 33) → (max x (max (x + 1) (x + 2)) = 12) :=
by sorry

end NUMINAMATH_CALUDE_greatest_of_three_consecutive_integers_l3112_311272


namespace NUMINAMATH_CALUDE_darryl_earnings_l3112_311213

/-- Calculates the total earnings from selling melons --/
def melon_earnings (
  cantaloupe_price : ℕ)
  (honeydew_price : ℕ)
  (initial_cantaloupes : ℕ)
  (initial_honeydews : ℕ)
  (dropped_cantaloupes : ℕ)
  (rotten_honeydews : ℕ)
  (remaining_cantaloupes : ℕ)
  (remaining_honeydews : ℕ) : ℕ :=
  let sold_cantaloupes := initial_cantaloupes - dropped_cantaloupes - remaining_cantaloupes
  let sold_honeydews := initial_honeydews - rotten_honeydews - remaining_honeydews
  cantaloupe_price * sold_cantaloupes + honeydew_price * sold_honeydews

/-- Theorem stating that Darryl's earnings are $85 --/
theorem darryl_earnings : 
  melon_earnings 2 3 30 27 2 3 8 9 = 85 := by
  sorry

end NUMINAMATH_CALUDE_darryl_earnings_l3112_311213


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l3112_311227

theorem max_sum_of_squares (a b c : ℤ) : 
  a + b + c = 3 → a^3 + b^3 + c^3 = 3 → a^2 + b^2 + c^2 ≤ 57 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l3112_311227


namespace NUMINAMATH_CALUDE_square_tiles_count_l3112_311261

theorem square_tiles_count (total_tiles : ℕ) (total_edges : ℕ) (square_tiles : ℕ) (pentagonal_tiles : ℕ) :
  total_tiles = 30 →
  total_edges = 110 →
  total_tiles = square_tiles + pentagonal_tiles →
  4 * square_tiles + 5 * pentagonal_tiles = total_edges →
  square_tiles = 20 := by
  sorry

end NUMINAMATH_CALUDE_square_tiles_count_l3112_311261


namespace NUMINAMATH_CALUDE_problem_solution_l3112_311298

theorem problem_solution (x y z : ℝ) 
  (h1 : x*z/(x+y) + y*z/(y+z) + x*y/(z+x) = -18)
  (h2 : z*y/(x+y) + z*x/(y+z) + y*x/(z+x) = 20) :
  y/(x+y) + z/(y+z) + x/(z+x) = 20.5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3112_311298


namespace NUMINAMATH_CALUDE_unique_modular_congruence_l3112_311268

theorem unique_modular_congruence : 
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -2023 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_congruence_l3112_311268


namespace NUMINAMATH_CALUDE_min_value_theorem_inequality_theorem_l3112_311204

variable (a b c : ℝ)

-- Define the conditions
def sum_condition (a b c : ℝ) : Prop := a + 2 * b + 3 * c = 6

-- Define the non-zero condition
def non_zero (x : ℝ) : Prop := x ≠ 0

-- Theorem for the first part
theorem min_value_theorem (ha : non_zero a) (hb : non_zero b) (hc : non_zero c) 
  (h_sum : sum_condition a b c) : 
  a^2 + 2 * b^2 + 3 * c^2 ≥ 6 := by sorry

-- Theorem for the second part
theorem inequality_theorem (ha : non_zero a) (hb : non_zero b) (hc : non_zero c) 
  (h_sum : sum_condition a b c) : 
  a^2 / (1 + a) + 2 * b^2 / (3 + b) + 3 * c^2 / (5 + c) ≥ 9/7 := by sorry

end NUMINAMATH_CALUDE_min_value_theorem_inequality_theorem_l3112_311204


namespace NUMINAMATH_CALUDE_cubic_factorization_l3112_311257

theorem cubic_factorization (x y z : ℝ) :
  x^3 + y^3 + z^3 - 3*x*y*z = (x + y + z) * (x^2 + y^2 + z^2 - x*y - y*z - z*x) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l3112_311257


namespace NUMINAMATH_CALUDE_simple_interest_principle_l3112_311247

theorem simple_interest_principle (r t A : ℚ) (h1 : r = 5 / 100) (h2 : t = 12 / 5) (h3 : A = 896) :
  ∃ P : ℚ, P * (1 + r * t) = A ∧ P = 800 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_principle_l3112_311247


namespace NUMINAMATH_CALUDE_smallest_divisors_sum_of_powers_l3112_311232

theorem smallest_divisors_sum_of_powers (n a b : ℕ) : 
  (a > 1) →
  (∀ k, 1 < k → k < a → ¬(k ∣ n)) →
  (a ∣ n) →
  (b > a) →
  (b ∣ n) →
  (∀ k, a < k → k < b → ¬(k ∣ n)) →
  (n = a^a + b^b) →
  (n = 260) :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisors_sum_of_powers_l3112_311232


namespace NUMINAMATH_CALUDE_download_speed_calculation_l3112_311246

theorem download_speed_calculation (file_size : ℝ) (speed_ratio : ℝ) (time_diff : ℝ) :
  file_size = 600 ∧ speed_ratio = 15 ∧ time_diff = 140 →
  ∃ (speed_4g : ℝ) (speed_5g : ℝ),
    speed_5g = speed_ratio * speed_4g ∧
    file_size / speed_4g - file_size / speed_5g = time_diff ∧
    speed_4g = 4 ∧ speed_5g = 60 := by
  sorry

end NUMINAMATH_CALUDE_download_speed_calculation_l3112_311246


namespace NUMINAMATH_CALUDE_inverse_101_mod_102_l3112_311222

theorem inverse_101_mod_102 : (101⁻¹ : ZMod 102) = 101 := by sorry

end NUMINAMATH_CALUDE_inverse_101_mod_102_l3112_311222


namespace NUMINAMATH_CALUDE_edwards_remaining_money_l3112_311248

/-- Calculates the remaining money after a purchase with sales tax -/
def remaining_money (initial_amount purchase_amount tax_rate : ℚ) : ℚ :=
  let sales_tax := purchase_amount * tax_rate
  let total_cost := purchase_amount + sales_tax
  initial_amount - total_cost

/-- Theorem stating that Edward's remaining money is $0.42 -/
theorem edwards_remaining_money :
  remaining_money 18 16.35 (75 / 1000) = 42 / 100 := by
  sorry

end NUMINAMATH_CALUDE_edwards_remaining_money_l3112_311248


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_l3112_311256

def is_simplest_quadratic_radical (x : ℝ → ℝ) (others : List (ℝ → ℝ)) : Prop :=
  ∀ y ∈ others, ∃ k : ℝ, k ≠ 0 ∧ ∀ a : ℝ, (x a) = k * (y a) → k = 1

theorem simplest_quadratic_radical :
  let x : ℝ → ℝ := λ a => Real.sqrt (a^2 + 1)
  let y₁ : ℝ → ℝ := λ _ => Real.sqrt 8
  let y₂ : ℝ → ℝ := λ _ => 1 / Real.sqrt 3
  let y₃ : ℝ → ℝ := λ _ => Real.sqrt 0.5
  is_simplest_quadratic_radical x [y₁, y₂, y₃] :=
sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_l3112_311256


namespace NUMINAMATH_CALUDE_fish_tagging_problem_l3112_311228

/-- The number of fish initially tagged in a pond -/
def initially_tagged (total_fish : ℕ) (later_catch : ℕ) (tagged_in_catch : ℕ) : ℕ :=
  (tagged_in_catch * total_fish) / later_catch

theorem fish_tagging_problem (total_fish : ℕ) (later_catch : ℕ) (tagged_in_catch : ℕ)
  (h1 : total_fish = 1800)
  (h2 : later_catch = 60)
  (h3 : tagged_in_catch = 2)
  (h4 : initially_tagged total_fish later_catch tagged_in_catch = (tagged_in_catch * total_fish) / later_catch) :
  initially_tagged total_fish later_catch tagged_in_catch = 60 :=
by sorry

end NUMINAMATH_CALUDE_fish_tagging_problem_l3112_311228


namespace NUMINAMATH_CALUDE_rhombus_diagonals_l3112_311284

/-- Given a rhombus with area 117 cm² and the perimeter of the rectangle formed by
    the midpoints of its sides is 31 cm, prove that its diagonals are 18 cm and 13 cm. -/
theorem rhombus_diagonals (area : ℝ) (perimeter : ℝ) (d₁ d₂ : ℝ) :
  area = 117 →
  perimeter = 31 →
  d₁ * d₂ / 2 = area →
  d₁ + d₂ = perimeter →
  (d₁ = 18 ∧ d₂ = 13) ∨ (d₁ = 13 ∧ d₂ = 18) := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonals_l3112_311284


namespace NUMINAMATH_CALUDE_roots_imply_p_zero_q_negative_l3112_311245

theorem roots_imply_p_zero_q_negative (α β p q : ℝ) : 
  α ≠ β →  -- α and β are distinct
  α^2 + p*α + q = 0 →  -- α is a root of the equation
  β^2 + p*β + q = 0 →  -- β is a root of the equation
  α^3 - α^2*β - α*β^2 + β^3 = 0 →  -- given condition
  p = 0 ∧ q < 0 := by
  sorry

end NUMINAMATH_CALUDE_roots_imply_p_zero_q_negative_l3112_311245


namespace NUMINAMATH_CALUDE_parallel_vectors_y_value_l3112_311264

/-- Given two parallel vectors a and b, prove that y = 4 -/
theorem parallel_vectors_y_value (a b : ℝ × ℝ) (y : ℝ) :
  a = (2, 6) →
  b = (1, -1 + y) →
  ∃ (k : ℝ), a = k • b →
  y = 4 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_y_value_l3112_311264


namespace NUMINAMATH_CALUDE_bisecting_line_min_value_l3112_311295

/-- A line that bisects the circumference of a circle -/
structure BisetingLine where
  a : ℝ
  b : ℝ
  h1 : a ≥ b
  h2 : b > 0
  h3 : ∀ (x y : ℝ), a * x + 2 * b * y - 2 = 0 → x^2 + y^2 - 4*x - 2*y - 8 = 0

/-- The minimum value of 1/a + 2/b for a bisecting line is 6 -/
theorem bisecting_line_min_value (l : BisetingLine) : 
  (∀ (a' b' : ℝ), a' ≥ b' ∧ b' > 0 → 1 / a' + 2 / b' ≥ 1 / l.a + 2 / l.b) ∧
  1 / l.a + 2 / l.b = 6 :=
sorry

end NUMINAMATH_CALUDE_bisecting_line_min_value_l3112_311295


namespace NUMINAMATH_CALUDE_sold_to_production_ratio_l3112_311212

def last_year_production : ℕ := 5000
def this_year_production : ℕ := 2 * last_year_production
def phones_left : ℕ := 7500

def sold_phones : ℕ := this_year_production - phones_left

theorem sold_to_production_ratio : 
  (sold_phones : ℚ) / this_year_production = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_sold_to_production_ratio_l3112_311212


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3112_311269

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x : ℝ | x < 2}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -1 ≤ x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3112_311269


namespace NUMINAMATH_CALUDE_barrel_capacity_l3112_311278

/-- Prove that given the conditions, each barrel stores 2 gallons less than twice the capacity of a cask. -/
theorem barrel_capacity (num_barrels : ℕ) (cask_capacity : ℕ) (total_capacity : ℕ) :
  num_barrels = 4 →
  cask_capacity = 20 →
  total_capacity = 172 →
  (total_capacity - cask_capacity) / num_barrels = 2 * cask_capacity - 2 := by
  sorry


end NUMINAMATH_CALUDE_barrel_capacity_l3112_311278


namespace NUMINAMATH_CALUDE_range_of_m_l3112_311237

-- Define the propositions P and Q
def P (m : ℝ) : Prop := ∃ (x y : ℝ), x^2 + y^2 - 2*m*x + 2*m^2 - 2*m = 0

def Q (m : ℝ) : Prop := 
  let e := Real.sqrt (1 + m/5)
  1 < e ∧ e < 2

-- State the theorem
theorem range_of_m : 
  ∀ m : ℝ, (¬(P m) ∧ Q m) → 2 ≤ m ∧ m < 15 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3112_311237


namespace NUMINAMATH_CALUDE_three_fourths_to_fifth_power_l3112_311240

theorem three_fourths_to_fifth_power : (3 / 4 : ℚ) ^ 5 = 243 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_three_fourths_to_fifth_power_l3112_311240


namespace NUMINAMATH_CALUDE_least_square_value_l3112_311241

theorem least_square_value (a x y : ℕ+) 
  (h1 : 15 * a + 165 = x^2)
  (h2 : 16 * a - 155 = y^2) :
  min (x^2) (y^2) ≥ 231361 := by
  sorry

end NUMINAMATH_CALUDE_least_square_value_l3112_311241


namespace NUMINAMATH_CALUDE_easter_egg_distribution_l3112_311296

theorem easter_egg_distribution (baskets : ℕ) (eggs_per_basket : ℕ) (people : ℕ) :
  baskets = 15 →
  eggs_per_basket = 12 →
  people = 20 →
  (baskets * eggs_per_basket) / people = 9 := by
sorry

end NUMINAMATH_CALUDE_easter_egg_distribution_l3112_311296


namespace NUMINAMATH_CALUDE_cow_increase_is_24_l3112_311282

/-- Represents the number of cows at different stages --/
structure CowCount where
  initial : Nat
  after_deaths : Nat
  after_sales : Nat
  current : Nat

/-- Calculates the increase in cow count given the initial conditions and final count --/
def calculate_increase (c : CowCount) (bought : Nat) (gifted : Nat) : Nat :=
  c.current - (c.after_sales + bought + gifted)

/-- Theorem stating that the increase in cows is 24 given the problem conditions --/
theorem cow_increase_is_24 :
  let c := CowCount.mk 39 (39 - 25) ((39 - 25) - 6) 83
  let bought := 43
  let gifted := 8
  calculate_increase c bought gifted = 24 := by
  sorry

end NUMINAMATH_CALUDE_cow_increase_is_24_l3112_311282


namespace NUMINAMATH_CALUDE_range_of_a_l3112_311259

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x + 3| + |x - 1| ≥ a^2 - 3*a) ↔ -1 ≤ a ∧ a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3112_311259


namespace NUMINAMATH_CALUDE_oplus_properties_l3112_311281

def oplus (a b : ℚ) : ℚ := a * b + 2 * a

theorem oplus_properties :
  (oplus 2 (-1) = 2) ∧
  (oplus (-3) (oplus (-4) (1/2)) = 24) := by
  sorry

end NUMINAMATH_CALUDE_oplus_properties_l3112_311281


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_l3112_311224

theorem sum_of_fourth_powers (a b c : ℝ) 
  (sum_eq : a + b + c = 8)
  (sum_prod_eq : a * b + a * c + b * c = 13)
  (prod_eq : a * b * c = -22) :
  a^4 + b^4 + c^4 = 1378 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_l3112_311224


namespace NUMINAMATH_CALUDE_min_value_a_squared_plus_b_squared_l3112_311276

theorem min_value_a_squared_plus_b_squared (a b : ℝ) : 
  (∃ x : ℝ, x^4 + a*x^3 + b*x^2 + a*x + 1 = 0) → 
  (∀ a' b' : ℝ, (∃ x : ℝ, x^4 + a'*x^3 + b'*x^2 + a'*x + 1 = 0) → a'^2 + b'^2 ≥ 4) ∧
  (∃ a' b' : ℝ, (∃ x : ℝ, x^4 + a'*x^3 + b'*x^2 + a'*x + 1 = 0) ∧ a'^2 + b'^2 = 4) :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_squared_plus_b_squared_l3112_311276


namespace NUMINAMATH_CALUDE_markup_percentage_l3112_311231

theorem markup_percentage (cost selling_price markup : ℝ) : 
  markup = selling_price - cost →
  markup = 0.0909090909090909 * selling_price →
  markup = 0.1 * cost := by
  sorry

end NUMINAMATH_CALUDE_markup_percentage_l3112_311231


namespace NUMINAMATH_CALUDE_function_properties_l3112_311271

def f (a : ℝ) (x : ℝ) : ℝ := a * x
def g (a : ℝ) (x : ℝ) : ℝ := x^2 + a

theorem function_properties (a : ℝ) :
  (∀ x, f a (-x) = -(f a x)) ∧
  (∀ x, g a (-x) = g a x) ∧
  (∀ x, f a x + g a x = x^2 + a*x + a) ∧
  ((∀ x ∈ Set.Icc 1 2, f a x ≥ 1) ∨ (∃ x ∈ Set.Icc (-1) 2, g a x ≤ -1)) →
  (a ≥ 1 ∨ a ≤ -1) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3112_311271


namespace NUMINAMATH_CALUDE_alien_martian_limb_difference_l3112_311253

/-- Number of arms an Alien has -/
def alien_arms : ℕ := 3

/-- Number of legs an Alien has -/
def alien_legs : ℕ := 8

/-- Number of arms a Martian has -/
def martian_arms : ℕ := 2 * alien_arms

/-- Number of legs a Martian has -/
def martian_legs : ℕ := alien_legs / 2

/-- Total number of limbs for an Alien -/
def alien_limbs : ℕ := alien_arms + alien_legs

/-- Total number of limbs for a Martian -/
def martian_limbs : ℕ := martian_arms + martian_legs

/-- Number of Aliens and Martians in the comparison -/
def group_size : ℕ := 5

theorem alien_martian_limb_difference :
  group_size * alien_limbs - group_size * martian_limbs = 5 := by
  sorry

end NUMINAMATH_CALUDE_alien_martian_limb_difference_l3112_311253


namespace NUMINAMATH_CALUDE_polar_curve_is_line_and_circle_l3112_311205

/-- The curve represented by the polar equation ρsin(θ) = sin(2θ) -/
def polar_curve (ρ θ : ℝ) : Prop :=
  ρ * Real.sin θ = Real.sin (2 * θ)

/-- The line part of the curve -/
def line_part (x y : ℝ) : Prop :=
  y = 0

/-- The circle part of the curve -/
def circle_part (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 1

/-- Theorem stating that the polar curve consists of a line and a circle -/
theorem polar_curve_is_line_and_circle :
  ∀ ρ θ x y : ℝ, polar_curve ρ θ → 
  (∃ ρ' θ', x = ρ' * Real.cos θ' ∧ y = ρ' * Real.sin θ') →
  (line_part x y ∨ circle_part x y) :=
sorry

end NUMINAMATH_CALUDE_polar_curve_is_line_and_circle_l3112_311205


namespace NUMINAMATH_CALUDE_cupcake_packages_l3112_311285

theorem cupcake_packages (x y z : ℕ) (hx : x = 50) (hy : y = 5) (hz : z = 5) :
  (x - y) / z = 9 := by
  sorry

end NUMINAMATH_CALUDE_cupcake_packages_l3112_311285


namespace NUMINAMATH_CALUDE_pizza_order_l3112_311293

theorem pizza_order (cost_per_box : ℚ) (tip_ratio : ℚ) (total_paid : ℚ) : 
  cost_per_box = 7 →
  tip_ratio = 1 / 7 →
  total_paid = 40 →
  ∃ (num_boxes : ℕ), 
    (↑num_boxes * cost_per_box) * (1 + tip_ratio) = total_paid ∧
    num_boxes = 5 := by
  sorry

end NUMINAMATH_CALUDE_pizza_order_l3112_311293


namespace NUMINAMATH_CALUDE_not_both_perfect_squares_l3112_311265

theorem not_both_perfect_squares (p q : ℕ) (hp : p > 0) (hq : q > 0) :
  ¬(∃ (a b : ℕ), p^2 + q = a^2 ∧ p + q^2 = b^2) := by
  sorry

end NUMINAMATH_CALUDE_not_both_perfect_squares_l3112_311265
