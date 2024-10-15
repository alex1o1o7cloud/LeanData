import Mathlib

namespace NUMINAMATH_GPT_find_y_l732_73269

theorem find_y (a b : ℝ) (y : ℝ) (h0 : b ≠ 0) (h1 : (3 * a)^(2 * b) = a^b * y^b) : y = 9 * a := by
  sorry

end NUMINAMATH_GPT_find_y_l732_73269


namespace NUMINAMATH_GPT_number_of_square_free_odds_l732_73265

def is_square_free (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 1 → m * m ∣ n → false

theorem number_of_square_free_odds (n : ℕ) (h1 : 1 < n) (h2 : n < 200) (h3 : n % 2 = 1) :
  (is_square_free n) ↔ (n = 79) := by
  sorry

end NUMINAMATH_GPT_number_of_square_free_odds_l732_73265


namespace NUMINAMATH_GPT_test_total_points_l732_73224

def total_points (total_problems comp_problems : ℕ) (points_comp points_word : ℕ) : ℕ :=
  let word_problems := total_problems - comp_problems
  (comp_problems * points_comp) + (word_problems * points_word)

theorem test_total_points :
  total_points 30 20 3 5 = 110 := by
  sorry

end NUMINAMATH_GPT_test_total_points_l732_73224


namespace NUMINAMATH_GPT_greatest_power_of_2_factor_l732_73215

theorem greatest_power_of_2_factor
    : ∃ k : ℕ, (2^k) ∣ (10^1503 - 4^752) ∧ ∀ m : ℕ, (2^(m+1)) ∣ (10^1503 - 4^752) → m < k :=
by
    sorry

end NUMINAMATH_GPT_greatest_power_of_2_factor_l732_73215


namespace NUMINAMATH_GPT_ellipse_solution_length_AB_l732_73234

noncomputable def ellipse_equation (a b : ℝ) (e : ℝ) (minor_axis : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ e = 3 / 4 ∧ 2 * b = minor_axis ∧ minor_axis = 2 * Real.sqrt 7

theorem ellipse_solution (a b : ℝ) (e : ℝ) (minor_axis : ℝ) :
  ellipse_equation a b e minor_axis →
  (a^2 = 16 ∧ b^2 = 7 ∧ (1 / a^2) = 1 / 16 ∧ (1 / b^2) = 1 / 7) :=
by 
  intros h
  sorry

noncomputable def area_ratio (S1 S2 : ℝ) : Prop :=
  S1 / S2 = 9 / 13

theorem length_AB (S1 S2 : ℝ) :
  area_ratio S1 S2 →
  |S1 / S2| = |(9 * Real.sqrt 105) / 26| :=
by
  intros h
  sorry

end NUMINAMATH_GPT_ellipse_solution_length_AB_l732_73234


namespace NUMINAMATH_GPT_sum_every_second_term_is_1010_l732_73261

def sequence_sum (n : ℕ) (a d : ℤ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

theorem sum_every_second_term_is_1010 :
  ∃ (x1 : ℤ) (d : ℤ) (S : ℤ), 
  (sequence_sum 2020 x1 d = 6060) ∧
  (d = 2) ∧
  (S = (1010 : ℤ)) ∧ 
  (2 * S + 4040 = 6060) :=
  sorry

end NUMINAMATH_GPT_sum_every_second_term_is_1010_l732_73261


namespace NUMINAMATH_GPT_solve_inequality_l732_73203

theorem solve_inequality :
  {x : ℝ | (3 * x + 1) * (2 * x - 1) < 0} = {x : ℝ | -1 / 3 < x ∧ x < 1 / 2} :=
  sorry

end NUMINAMATH_GPT_solve_inequality_l732_73203


namespace NUMINAMATH_GPT_volumes_of_rotated_solids_l732_73296

theorem volumes_of_rotated_solids
  (π : ℝ)
  (b c a : ℝ)
  (h₁ : a^2 = b^2 + c^2)
  (v v₁ v₂ : ℝ)
  (hv : v = (1/3) * π * (b^2 * c^2) / a)
  (hv₁ : v₁ = (1/3) * π * c^2 * b)
  (hv₂ : v₂ = (1/3) * π * b^2 * c) :
  (1 / v^2) = (1 / v₁^2) + (1 / v₂^2) := 
by sorry

end NUMINAMATH_GPT_volumes_of_rotated_solids_l732_73296


namespace NUMINAMATH_GPT_solve_for_x_l732_73243

theorem solve_for_x (x : ℚ) (h : 10 * x = x + 20) : x = 20 / 9 :=
  sorry

end NUMINAMATH_GPT_solve_for_x_l732_73243


namespace NUMINAMATH_GPT_ab_over_a_minus_b_l732_73228

theorem ab_over_a_minus_b (a b : ℝ) (h : (1 / a) - (1 / b) = 1 / 3) : (a * b) / (a - b) = -3 := by
  sorry

end NUMINAMATH_GPT_ab_over_a_minus_b_l732_73228


namespace NUMINAMATH_GPT_unique_intersection_value_l732_73235

theorem unique_intersection_value :
  (∀ (x y : ℝ), y = x^2 → y = 4 * x + k) → (k = -4) := 
by
  sorry

end NUMINAMATH_GPT_unique_intersection_value_l732_73235


namespace NUMINAMATH_GPT_find_x_l732_73275

theorem find_x (x : ℚ) (h : x ≠ 2 ∧ x ≠ 4/5) :
  (x^2 - 11*x + 24)/(x - 2) + (5*x^2 + 22*x - 48)/(5*x - 4) = -7 → x = -4/3 :=
by
  intro h1
  sorry

end NUMINAMATH_GPT_find_x_l732_73275


namespace NUMINAMATH_GPT_nautical_mile_to_land_mile_l732_73209

theorem nautical_mile_to_land_mile 
    (speed_one_sail : ℕ := 25) 
    (speed_two_sails : ℕ := 50) 
    (travel_time_one_sail : ℕ := 4) 
    (travel_time_two_sails : ℕ := 4)
    (total_distance : ℕ := 345) : 
    ∃ (x : ℚ), x = 1.15 ∧ 
    total_distance = travel_time_one_sail * speed_one_sail * x +
                    travel_time_two_sails * speed_two_sails * x := 
by
  sorry

end NUMINAMATH_GPT_nautical_mile_to_land_mile_l732_73209


namespace NUMINAMATH_GPT_discount_given_l732_73212

variables (initial_money : ℕ) (extra_fraction : ℕ) (additional_money_needed : ℕ)
variables (total_with_discount : ℕ) (discount_amount : ℕ)

def total_without_discount (initial_money : ℕ) (extra_fraction : ℕ) : ℕ :=
  initial_money + extra_fraction

def discount (initial_money : ℕ) (total_without_discount : ℕ) (total_with_discount : ℕ) : ℕ :=
  total_without_discount - total_with_discount

def discount_percentage (discount_amount : ℕ) (total_without_discount : ℕ) : ℚ :=
  (discount_amount : ℚ) / (total_without_discount : ℚ) * 100

theorem discount_given 
  (initial_money : ℕ := 500)
  (extra_fraction : ℕ := 200)
  (additional_money_needed : ℕ := 95)
  (total_without_discount₀ : ℕ := total_without_discount initial_money extra_fraction)
  (total_with_discount₀ : ℕ := initial_money + additional_money_needed)
  (discount_amount₀ : ℕ := discount initial_money total_without_discount₀ total_with_discount₀)
  : discount_percentage discount_amount₀ total_without_discount₀ = 15 :=
by sorry

end NUMINAMATH_GPT_discount_given_l732_73212


namespace NUMINAMATH_GPT_triangle_ABC_proof_l732_73285

noncomputable def sin2C_eq_sqrt3sinC (C : ℝ) : Prop := Real.sin (2 * C) = Real.sqrt 3 * Real.sin C

theorem triangle_ABC_proof (C a b c : ℝ) 
  (H1 : sin2C_eq_sqrt3sinC C) 
  (H2 : 0 < Real.sin C)
  (H3 : b = 6) 
  (H4 : a + b + c = 6*Real.sqrt 3 + 6) :
  (C = π/6) ∧ (1/2 * a * b * Real.sin C = 6*Real.sqrt 3) :=
sorry

end NUMINAMATH_GPT_triangle_ABC_proof_l732_73285


namespace NUMINAMATH_GPT_volume_of_prism_l732_73229

theorem volume_of_prism
  (a b c : ℝ)
  (h1 : a * b = 30)
  (h2 : a * c = 40)
  (h3 : b * c = 60) :
  a * b * c = 120 * Real.sqrt 5 := 
by
  sorry

end NUMINAMATH_GPT_volume_of_prism_l732_73229


namespace NUMINAMATH_GPT_find_number_l732_73225

theorem find_number (x : ℝ) (h : 0.2 * x = 0.3 * 120 + 80) : x = 580 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l732_73225


namespace NUMINAMATH_GPT_jana_distance_l732_73290

theorem jana_distance (time_to_walk_one_mile : ℝ) (time_to_walk : ℝ) :
  (time_to_walk_one_mile = 18) → (time_to_walk = 15) →
  ((time_to_walk / time_to_walk_one_mile) * 1 = 0.8) :=
  by
    intros h1 h2
    rw [h1, h2]
    -- Here goes the proof, but it is skipped as per requirements
    sorry

end NUMINAMATH_GPT_jana_distance_l732_73290


namespace NUMINAMATH_GPT_sum_is_945_l732_73239

def sum_of_integers_from_90_to_99 : ℕ :=
  90 + 91 + 92 + 93 + 94 + 95 + 96 + 97 + 98 + 99

theorem sum_is_945 : sum_of_integers_from_90_to_99 = 945 := 
by
  sorry

end NUMINAMATH_GPT_sum_is_945_l732_73239


namespace NUMINAMATH_GPT_coordinates_of_B_l732_73218

noncomputable def B_coordinates := 
  let A : ℝ × ℝ := (-1, -5)
  let a : ℝ × ℝ := (2, 3)
  let AB := (3 * a.1, 3 * a.2)
  let B := (A.1 + AB.1, A.2 + AB.2)
  B

theorem coordinates_of_B : B_coordinates = (5, 4) := 
by 
  sorry

end NUMINAMATH_GPT_coordinates_of_B_l732_73218


namespace NUMINAMATH_GPT_greatest_int_less_than_200_gcd_30_is_5_l732_73288

theorem greatest_int_less_than_200_gcd_30_is_5 : ∃ n, n < 200 ∧ gcd n 30 = 5 ∧ ∀ m, m < 200 ∧ gcd m 30 = 5 → m ≤ n :=
by
  sorry

end NUMINAMATH_GPT_greatest_int_less_than_200_gcd_30_is_5_l732_73288


namespace NUMINAMATH_GPT_vector_parallel_find_k_l732_73282

theorem vector_parallel_find_k (k : ℝ) (a : ℝ × ℝ) (b : ℝ × ℝ) 
  (h₁ : a = (3 * k + 1, 2)) 
  (h₂ : b = (k, 1)) 
  (h₃ : ∃ c : ℝ, a = c • b) : k = -1 := 
by 
  sorry

end NUMINAMATH_GPT_vector_parallel_find_k_l732_73282


namespace NUMINAMATH_GPT_investment_difference_l732_73278

noncomputable def future_value_semi_annual (principal : ℝ) (annual_rate : ℝ) (years : ℝ) : ℝ :=
  principal * (1 + annual_rate / 2)^((years * 2))

noncomputable def future_value_monthly (principal : ℝ) (annual_rate : ℝ) (years : ℝ) : ℝ :=
  principal * (1 + annual_rate / 12)^((years * 12))

theorem investment_difference :
  let jose_investment := future_value_semi_annual 30000 0.03 3
  let patricia_investment := future_value_monthly 30000 0.025 3
  round (jose_investment) - round (patricia_investment) = 317 :=
by
  sorry

end NUMINAMATH_GPT_investment_difference_l732_73278


namespace NUMINAMATH_GPT_determine_original_volume_of_tank_l732_73205

noncomputable def salt_volume (x : ℝ) := 0.20 * x
noncomputable def new_volume_after_evaporation (x : ℝ) := (3 / 4) * x
noncomputable def new_volume_after_additions (x : ℝ) := (3 / 4) * x + 6 + 12
noncomputable def new_salt_after_addition (x : ℝ) := 0.20 * x + 12
noncomputable def resulting_salt_concentration (x : ℝ) := (0.20 * x + 12) / ((3 / 4) * x + 18)

theorem determine_original_volume_of_tank (x : ℝ) :
  resulting_salt_concentration x = 1 / 3 → x = 120 := 
by 
  sorry

end NUMINAMATH_GPT_determine_original_volume_of_tank_l732_73205


namespace NUMINAMATH_GPT_cistern_length_l732_73273

def cistern_conditions (L : ℝ) : Prop := 
  let width := 4
  let depth := 1.25
  let wet_surface_area := 42.5
  (L * width) + (2 * (L * depth)) + (2 * (width * depth)) = wet_surface_area

theorem cistern_length : 
  ∃ L : ℝ, cistern_conditions L ∧ L = 5 := sorry

end NUMINAMATH_GPT_cistern_length_l732_73273


namespace NUMINAMATH_GPT_cos_value_l732_73241

theorem cos_value (α : ℝ) (h : Real.sin (π / 5 - α) = 1 / 4) : 
  Real.cos (2 * α + 3 * π / 5) = -7 / 8 := 
by
  sorry

end NUMINAMATH_GPT_cos_value_l732_73241


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l732_73208

theorem hyperbola_eccentricity (a : ℝ) (h : 0 < a) (h_asymptote : Real.tan (Real.pi / 6) = 1 / a) : 
  (a = Real.sqrt 3) → 
  (∃ e : ℝ, e = (2 * Real.sqrt 3) / 3) :=
by
  intros
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l732_73208


namespace NUMINAMATH_GPT_equation_of_line_through_point_with_given_slope_l732_73272

-- Define the condition that line L passes through point P(-2, 5) and has slope -3/4
def line_through_point_with_slope (x1 y1 m : ℚ) (x y : ℚ) : Prop :=
  y - y1 = m * (x - x1)

-- Define the specific point (-2, 5) and slope -3/4
def P : ℚ × ℚ := (-2, 5)
def m : ℚ := -3 / 4

-- The standard form equation of the line as the target
def standard_form (x y : ℚ) : Prop :=
  3 * x + 4 * y - 14 = 0

-- The theorem to prove
theorem equation_of_line_through_point_with_given_slope :
  ∀ x y : ℚ, line_through_point_with_slope (-2) 5 (-3 / 4) x y → standard_form x y :=
  by
    intros x y h
    sorry

end NUMINAMATH_GPT_equation_of_line_through_point_with_given_slope_l732_73272


namespace NUMINAMATH_GPT_part1_monotonicity_part2_range_a_l732_73262

noncomputable def f (a x : ℝ) : ℝ := Real.log x - a * x + 1

theorem part1_monotonicity (a : ℝ) :
  (∀ x > 0, (0 : ℝ) < x → 0 < 1 / x - a) ∨
  (a > 0 → ∀ x > 0, (0 : ℝ) < x ∧ x < 1 / a → 0 < 1 / x - a ∧ 1 / a < x → 1 / x - a < 0) := sorry

theorem part2_range_a (a : ℝ) :
  (∀ x > 0, Real.log x - a * x + 1 ≤ 0) → 1 ≤ a := sorry

end NUMINAMATH_GPT_part1_monotonicity_part2_range_a_l732_73262


namespace NUMINAMATH_GPT_cube_face_sum_l732_73219

theorem cube_face_sum (a d b e c f g : ℕ)
    (h1 : g = 2)
    (h2 : 2310 = 2 * 3 * 5 * 7 * 11)
    (h3 : (a + d) * (b + e) * (c + f) = 3 * 5 * 7 * 11):
    (a + d) + (b + e) + (c + f) = 47 :=
by
    sorry

end NUMINAMATH_GPT_cube_face_sum_l732_73219


namespace NUMINAMATH_GPT_prove_range_of_m_prove_m_value_l732_73246

def quadratic_roots (m : ℝ) (x1 x2 : ℝ) : Prop := 
  x1 * x1 - (2 * m - 3) * x1 + m * m = 0 ∧ 
  x2 * x2 - (2 * m - 3) * x2 + m * m = 0

def range_of_m (m : ℝ) : Prop := 
  m <= 3/4

def condition_on_m (m : ℝ) (x1 x2 : ℝ) : Prop :=
  x1 + x2 = -(x1 * x2)

theorem prove_range_of_m (m : ℝ) :
  (∃ x1 x2 : ℝ, quadratic_roots m x1 x2) → range_of_m m :=
sorry

theorem prove_m_value (m : ℝ) (x1 x2 : ℝ) :
  quadratic_roots m x1 x2 → condition_on_m m x1 x2 → m = -3 :=
sorry

end NUMINAMATH_GPT_prove_range_of_m_prove_m_value_l732_73246


namespace NUMINAMATH_GPT_mary_age_l732_73260

theorem mary_age :
  ∃ M R : ℕ, (R = M + 30) ∧ (R + 20 = 2 * (M + 20)) ∧ (M = 10) :=
by
  sorry

end NUMINAMATH_GPT_mary_age_l732_73260


namespace NUMINAMATH_GPT_dog_running_direction_undeterminable_l732_73216

/-- Given the conditions:
 1. A dog is tied to a tree with a nylon cord of length 10 feet.
 2. The dog runs from one side of the tree to the opposite side with the cord fully extended.
 3. The dog runs approximately 30 feet.
 Prove that it is not possible to determine the specific starting direction of the dog.
-/
theorem dog_running_direction_undeterminable (r : ℝ) (full_length : r = 10) (distance_ran : ℝ) (approx_distance : distance_ran = 30) : (
  ∀ (d : ℝ), d < 2 * π * r → ¬∃ θ : ℝ, 0 ≤ θ ∧ θ < 2 * π ∧ (distance_ran = r * θ)
  ) :=
by
  sorry

end NUMINAMATH_GPT_dog_running_direction_undeterminable_l732_73216


namespace NUMINAMATH_GPT_min_value_expression_l732_73255

theorem min_value_expression (x : ℝ) (h : x > 2) : 
  ∃ y, y = x + 1 / (x - 2) ∧ y = 4 := 
sorry

end NUMINAMATH_GPT_min_value_expression_l732_73255


namespace NUMINAMATH_GPT_expression_equiv_l732_73299

theorem expression_equiv (x y : ℝ) (h : x ≠ 0 ∧ y ≠ 0) :
  ((x^4 + 1) / x^2) * ((y^4 + 1) / y^2) + ((x^4 - 1) / y^2) * ((y^4 - 1) / x^2) =
  2*x^2*y^2 + 2/(x^2*y^2) :=
by 
  sorry

end NUMINAMATH_GPT_expression_equiv_l732_73299


namespace NUMINAMATH_GPT_meal_cost_l732_73213

variable (s c p : ℝ)

axiom cond1 : 5 * s + 8 * c + p = 5.00
axiom cond2 : 7 * s + 12 * c + p = 7.20
axiom cond3 : 4 * s + 6 * c + 2 * p = 6.00

theorem meal_cost : s + c + p = 1.90 :=
by
  sorry

end NUMINAMATH_GPT_meal_cost_l732_73213


namespace NUMINAMATH_GPT_correct_statements_l732_73259

-- Definitions for each statement
def statement_1 := ∀ p q : ℤ, q ≠ 0 → (∃ n : ℤ, ∃ d : ℤ, p = n ∧ q = d ∧ (n, d) = (p, q))
def statement_2 := ∀ r : ℚ, (r > 0 ∨ r < 0) ∨ (∃ d : ℚ, d ≥ 0)
def statement_3 := ∀ x y : ℚ, abs x = abs y → x = y
def statement_4 := ∀ x : ℚ, (-x = x ∧ abs x = x) → x = 0
def statement_5 := ∀ x y : ℚ, abs x > abs y → x > y
def statement_6 := (∃ n : ℕ, n > 0) ∧ (∀ r : ℚ, r > 0 → ∃ q : ℚ, q > 0 ∧ q < r)

-- Main theorem: Prove that exactly 3 statements are correct
theorem correct_statements : 
  (statement_1 ∧ statement_4 ∧ statement_6) ∧ 
  (¬ statement_2 ∧ ¬ statement_3 ∧ ¬ statement_5) :=
by
  sorry

end NUMINAMATH_GPT_correct_statements_l732_73259


namespace NUMINAMATH_GPT_p_q_2r_value_l732_73286

variable (p q r : ℝ) (f : ℝ → ℝ)

-- The conditions as definitions
def f_def : f = fun x => p * x^2 + q * x + r := by sorry
def f_at_0 : f 0 = 9 := by sorry
def f_at_1 : f 1 = 6 := by sorry

-- The theorem statement
theorem p_q_2r_value : p + q + 2 * r = 15 :=
by
  -- utilizing the given definitions 
  have h₁ : r = 9 := by sorry
  have h₂ : p + q + r = 6 := by sorry
  -- substitute into p + q + 2r
  sorry

end NUMINAMATH_GPT_p_q_2r_value_l732_73286


namespace NUMINAMATH_GPT_rectangle_area_l732_73263

theorem rectangle_area (w l : ℕ) (h1 : l = 15) (h2 : (2 * l + 2 * w) / w = 5) : l * w = 150 :=
by
  -- We provide the conditions in the theorem's signature:
  -- l is the length which is 15 cm, given by h1
  -- The ratio of the perimeter to the width is 5:1, given by h2
  sorry

end NUMINAMATH_GPT_rectangle_area_l732_73263


namespace NUMINAMATH_GPT_compare_f_values_l732_73217

noncomputable def f (x : Real) : Real := 
  Real.cos x + 2 * x * (1 / 2)  -- given f''(pi/6) = 1/2

theorem compare_f_values :
  f (-Real.pi / 3) < f (Real.pi / 3) :=
by
  sorry

end NUMINAMATH_GPT_compare_f_values_l732_73217


namespace NUMINAMATH_GPT_clipping_per_friend_l732_73244

def GluePerClipping : Nat := 6
def TotalGlue : Nat := 126
def TotalFriends : Nat := 7

theorem clipping_per_friend :
  (TotalGlue / GluePerClipping) / TotalFriends = 3 := by
  sorry

end NUMINAMATH_GPT_clipping_per_friend_l732_73244


namespace NUMINAMATH_GPT_find_integer_closest_expression_l732_73222

theorem find_integer_closest_expression :
  let a := (7 + Real.sqrt 48) ^ 2023
  let b := (7 - Real.sqrt 48) ^ 2023
  ((a + b) ^ 2 - (a - b) ^ 2) = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_integer_closest_expression_l732_73222


namespace NUMINAMATH_GPT_largest_4_digit_number_divisible_by_12_l732_73210

theorem largest_4_digit_number_divisible_by_12 : ∃ (n : ℕ), 1000 ≤ n ∧ n ≤ 9999 ∧ n % 12 = 0 ∧ ∀ m, 1000 ≤ m ∧ m ≤ 9999 ∧ m % 12 = 0 → m ≤ n := 
sorry

end NUMINAMATH_GPT_largest_4_digit_number_divisible_by_12_l732_73210


namespace NUMINAMATH_GPT_pieces_brought_to_school_on_friday_l732_73202

def pieces_of_fruit_mark_had := 10
def pieces_eaten_first_four_days := 5
def pieces_kept_for_next_week := 2

theorem pieces_brought_to_school_on_friday :
  pieces_of_fruit_mark_had - pieces_eaten_first_four_days - pieces_kept_for_next_week = 3 :=
by
  sorry

end NUMINAMATH_GPT_pieces_brought_to_school_on_friday_l732_73202


namespace NUMINAMATH_GPT_f_is_even_if_g_is_odd_l732_73258

variable (g : ℝ → ℝ)

def is_odd (h : ℝ → ℝ) :=
  ∀ x : ℝ, h (-x) = -h x

def is_even (f : ℝ → ℝ) :=
  ∀ x : ℝ, f (-x) = f x

theorem f_is_even_if_g_is_odd (hg : is_odd g) :
  is_even (fun x => |g (x^4)|) :=
by
  sorry

end NUMINAMATH_GPT_f_is_even_if_g_is_odd_l732_73258


namespace NUMINAMATH_GPT_geometric_sequence_problem_l732_73242

noncomputable def geom_sequence (a r : ℝ) (n : ℕ) : ℝ := a * r ^ n

theorem geometric_sequence_problem (a r : ℝ) (a4 a8 a6 a10 : ℝ) :
  a4 = geom_sequence a r 4 →
  a8 = geom_sequence a r 8 →
  a6 = geom_sequence a r 6 →
  a10 = geom_sequence a r 10 →
  a4 + a8 = -2 →
  a4^2 + 2 * a6^2 + a6 * a10 = 4 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_geometric_sequence_problem_l732_73242


namespace NUMINAMATH_GPT_inequality_relation_l732_73256

theorem inequality_relation (a b : ℝ) :
  (∃ a b : ℝ, a > b ∧ ¬(1/a < 1/b)) ∧ (∃ a b : ℝ, (1/a < 1/b) ∧ ¬(a > b)) :=
by {
  sorry
}

end NUMINAMATH_GPT_inequality_relation_l732_73256


namespace NUMINAMATH_GPT_range_of_real_number_a_l732_73266

theorem range_of_real_number_a (a : ℝ) : (∀ (x : ℝ), 0 < x → a < x + 1/x) → a < 2 := 
by
  sorry

end NUMINAMATH_GPT_range_of_real_number_a_l732_73266


namespace NUMINAMATH_GPT_solve_xyz_sum_l732_73245

theorem solve_xyz_sum :
  ∃ (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ (x+y+z)^3 - x^3 - y^3 - z^3 = 378 ∧ x+y+z = 9 :=
by
  sorry

end NUMINAMATH_GPT_solve_xyz_sum_l732_73245


namespace NUMINAMATH_GPT_SallyCarrots_l732_73211

-- Definitions of the conditions
def FredGrew (F : ℕ) := F = 4
def TotalGrew (T : ℕ) := T = 10
def SallyGrew (S : ℕ) (F T : ℕ) := S + F = T

-- The theorem to be proved
theorem SallyCarrots : ∃ S : ℕ, FredGrew 4 ∧ TotalGrew 10 ∧ SallyGrew S 4 10 ∧ S = 6 :=
  sorry

end NUMINAMATH_GPT_SallyCarrots_l732_73211


namespace NUMINAMATH_GPT_second_largest_subtract_smallest_correct_l732_73253

-- Definition of the elements
def numbers : List ℕ := [10, 11, 12, 13, 14]

-- Conditions derived from the problem
def smallest_number : ℕ := 10
def second_largest_number : ℕ := 13

-- Lean theorem statement representing the problem
theorem second_largest_subtract_smallest_correct :
  (second_largest_number - smallest_number) = 3 := 
by
  sorry

end NUMINAMATH_GPT_second_largest_subtract_smallest_correct_l732_73253


namespace NUMINAMATH_GPT_new_cost_after_decrease_l732_73207

def actual_cost : ℝ := 2400
def decrease_percentage : ℝ := 0.50
def decreased_amount (cost percentage : ℝ) : ℝ := percentage * cost
def new_cost (cost decreased : ℝ) : ℝ := cost - decreased

theorem new_cost_after_decrease :
  new_cost actual_cost (decreased_amount actual_cost decrease_percentage) = 1200 :=
by sorry

end NUMINAMATH_GPT_new_cost_after_decrease_l732_73207


namespace NUMINAMATH_GPT_same_terminal_side_angle_l732_73281

theorem same_terminal_side_angle (θ : ℤ) : θ = -390 → ∃ k : ℤ, 0 ≤ θ + k * 360 ∧ θ + k * 360 < 360 ∧ θ + k * 360 = 330 :=
  by
    sorry

end NUMINAMATH_GPT_same_terminal_side_angle_l732_73281


namespace NUMINAMATH_GPT_percent_not_red_balls_l732_73237

theorem percent_not_red_balls (percent_cubes percent_red_balls : ℝ) 
  (h1 : percent_cubes = 0.3) (h2 : percent_red_balls = 0.25) : 
  (1 - percent_red_balls) * (1 - percent_cubes) = 0.525 :=
by
  sorry

end NUMINAMATH_GPT_percent_not_red_balls_l732_73237


namespace NUMINAMATH_GPT_martha_black_butterflies_l732_73294

theorem martha_black_butterflies (total_butterflies blue_butterflies yellow_butterflies black_butterflies : ℕ) 
    (h1 : total_butterflies = 19)
    (h2 : blue_butterflies = 2 * yellow_butterflies)
    (h3 : blue_butterflies = 6) :
    black_butterflies = 10 :=
by
  -- Prove the theorem assuming the conditions are met
  sorry

end NUMINAMATH_GPT_martha_black_butterflies_l732_73294


namespace NUMINAMATH_GPT_grandma_red_bacon_bits_l732_73247

def mushrooms := 3
def cherry_tomatoes := 2 * mushrooms
def pickles := 4 * cherry_tomatoes
def bacon_bits := 4 * pickles
def red_bacon_bits := bacon_bits / 3

theorem grandma_red_bacon_bits : red_bacon_bits = 32 := by
  sorry

end NUMINAMATH_GPT_grandma_red_bacon_bits_l732_73247


namespace NUMINAMATH_GPT_intensity_on_Thursday_l732_73223

-- Step a) - Definitions from Conditions
def inversely_proportional (i b k : ℕ) : Prop := i * b = k

-- Translation of the proof problem
theorem intensity_on_Thursday (k b : ℕ) (h₁ : k = 24) (h₂ : b = 3) : ∃ i, inversely_proportional i b k ∧ i = 8 := 
by
  sorry

end NUMINAMATH_GPT_intensity_on_Thursday_l732_73223


namespace NUMINAMATH_GPT_arcsin_neg_sqrt_two_over_two_l732_73283

theorem arcsin_neg_sqrt_two_over_two : Real.arcsin (-Real.sqrt 2 / 2) = -Real.pi / 4 :=
  sorry

end NUMINAMATH_GPT_arcsin_neg_sqrt_two_over_two_l732_73283


namespace NUMINAMATH_GPT_percentage_of_women_picnic_l732_73297

theorem percentage_of_women_picnic (E : ℝ) (h1 : 0.20 * 0.55 * E + W * 0.45 * E = 0.29 * E) : 
  W = 0.4 := 
  sorry

end NUMINAMATH_GPT_percentage_of_women_picnic_l732_73297


namespace NUMINAMATH_GPT_polynomial_no_negative_roots_l732_73230

theorem polynomial_no_negative_roots (x : ℝ) (h : x < 0) : x^4 - 4 * x^3 - 6 * x^2 - 3 * x + 9 ≠ 0 := 
by 
  sorry

end NUMINAMATH_GPT_polynomial_no_negative_roots_l732_73230


namespace NUMINAMATH_GPT_points_per_round_l732_73284

-- Definitions based on conditions
def final_points (jane_points : ℕ) : Prop := jane_points = 60
def lost_points (jane_lost : ℕ) : Prop := jane_lost = 20
def rounds_played (jane_rounds : ℕ) : Prop := jane_rounds = 8

-- The theorem we want to prove
theorem points_per_round (jane_points jane_lost jane_rounds points_per_round : ℕ) 
  (h1 : final_points jane_points) 
  (h2 : lost_points jane_lost) 
  (h3 : rounds_played jane_rounds) : 
  points_per_round = ((jane_points + jane_lost) / jane_rounds) := 
sorry

end NUMINAMATH_GPT_points_per_round_l732_73284


namespace NUMINAMATH_GPT_total_weekly_cost_correct_l732_73252

def daily_consumption (cups_per_day : ℕ) (ounces_per_cup : ℝ) : ℝ :=
  cups_per_day * ounces_per_cup

def weekly_consumption (cups_per_day : ℕ) (ounces_per_cup : ℝ) (days_per_week : ℕ) : ℝ :=
  daily_consumption cups_per_day ounces_per_cup * days_per_week

def weekly_cost (weekly_ounces : ℝ) (cost_per_ounce : ℝ) : ℝ :=
  weekly_ounces * cost_per_ounce

def person_A_weekly_cost : ℝ :=
  weekly_cost (weekly_consumption 3 0.4 7) 1.40

def person_B_weekly_cost : ℝ :=
  weekly_cost (weekly_consumption 1 0.6 7) 1.20

def person_C_weekly_cost : ℝ :=
  weekly_cost (weekly_consumption 2 0.5 5) 1.35

def james_weekly_cost : ℝ :=
  weekly_cost (weekly_consumption 2 0.5 7) 1.25

def total_weekly_cost : ℝ :=
  person_A_weekly_cost + person_B_weekly_cost + person_C_weekly_cost + james_weekly_cost

theorem total_weekly_cost_correct : total_weekly_cost = 32.30 := by
  unfold total_weekly_cost person_A_weekly_cost person_B_weekly_cost person_C_weekly_cost james_weekly_cost
  unfold weekly_cost weekly_consumption daily_consumption
  sorry

end NUMINAMATH_GPT_total_weekly_cost_correct_l732_73252


namespace NUMINAMATH_GPT_average_of_solutions_l732_73233

theorem average_of_solutions (a b : ℝ) (h : ∃ x1 x2 : ℝ, a * x1 ^ 2 + 3 * a * x1 + b = 0 ∧ a * x2 ^ 2 + 3 * a * x2 + b = 0) :
  ((-3 : ℝ) / 2) = - 3 / 2 :=
by sorry

end NUMINAMATH_GPT_average_of_solutions_l732_73233


namespace NUMINAMATH_GPT_marissa_lunch_calories_l732_73267

theorem marissa_lunch_calories :
  (1 * 400) + (5 * 20) + (5 * 50) = 750 :=
by
  sorry

end NUMINAMATH_GPT_marissa_lunch_calories_l732_73267


namespace NUMINAMATH_GPT_find_A_plus_B_l732_73221

def isFourDigitNumber (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def isOdd (n : ℕ) : Prop :=
  n % 2 = 1

def isMultipleOf5 (n : ℕ) : Prop :=
  n % 5 = 0

def countFourDigitOddNumbers : ℕ :=
  ((9 : ℕ) * 10 * 10 * 5)

def countFourDigitMultiplesOf5 : ℕ :=
  ((9 : ℕ) * 10 * 10 * 2)

theorem find_A_plus_B : countFourDigitOddNumbers + countFourDigitMultiplesOf5 = 6300 := by
  sorry

end NUMINAMATH_GPT_find_A_plus_B_l732_73221


namespace NUMINAMATH_GPT_gcd_polynomial_multiple_528_l732_73201

-- Definition of the problem
theorem gcd_polynomial_multiple_528 (k : ℕ) : 
  gcd (3 * (528 * k) ^ 3 + (528 * k) ^ 2 + 4 * (528 * k) + 66) (528 * k) = 66 :=
by
  sorry

end NUMINAMATH_GPT_gcd_polynomial_multiple_528_l732_73201


namespace NUMINAMATH_GPT_find_x_l732_73249

-- Define conditions
def simple_interest (x y : ℝ) : Prop :=
  x * y * 2 / 100 = 800

def compound_interest (x y : ℝ) : Prop :=
  x * ((1 + y / 100)^2 - 1) = 820

-- Prove x = 8000 given the conditions
theorem find_x (x y : ℝ) (h1 : simple_interest x y) (h2 : compound_interest x y) : x = 8000 :=
  sorry

end NUMINAMATH_GPT_find_x_l732_73249


namespace NUMINAMATH_GPT_BaSO4_molecular_weight_l732_73254

noncomputable def Ba : ℝ := 137.327
noncomputable def S : ℝ := 32.065
noncomputable def O : ℝ := 15.999
noncomputable def BaSO4 : ℝ := Ba + S + 4 * O

theorem BaSO4_molecular_weight : BaSO4 = 233.388 := by
  sorry

end NUMINAMATH_GPT_BaSO4_molecular_weight_l732_73254


namespace NUMINAMATH_GPT_number_of_students_playing_soccer_l732_73298

variables (T B girls_total soccer_total G no_girls_soccer perc_boys_soccer : ℕ)

-- Conditions:
def total_students := T = 420
def boys_students := B = 312
def girls_students := G = 420 - 312
def girls_not_playing_soccer := no_girls_soccer = 63
def perc_boys_play_soccer := perc_boys_soccer = 82
def girls_playing_soccer := G - no_girls_soccer = 45

-- Proof Problem:
theorem number_of_students_playing_soccer (h1 : total_students T) (h2 : boys_students B) (h3 : girls_students G) (h4 : girls_not_playing_soccer no_girls_soccer) (h5 : girls_playing_soccer G no_girls_soccer) (h6 : perc_boys_play_soccer perc_boys_soccer) : soccer_total = 250 :=
by {
  -- The proof would be inserted here.
  sorry
}

end NUMINAMATH_GPT_number_of_students_playing_soccer_l732_73298


namespace NUMINAMATH_GPT_min_value_fracs_l732_73295

-- Define the problem and its conditions in Lean.
theorem min_value_fracs (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 1) : 
  (2 / a + 3 / b) ≥ 8 + 4 * Real.sqrt 3 :=
  sorry

end NUMINAMATH_GPT_min_value_fracs_l732_73295


namespace NUMINAMATH_GPT_factor_expression_l732_73226

theorem factor_expression :
  (8 * x^6 + 36 * x^4 - 5) - (2 * x^6 - 6 * x^4 + 5) = 2 * (3 * x^6 + 21 * x^4 - 5) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l732_73226


namespace NUMINAMATH_GPT_functions_with_inverses_l732_73206

-- Definitions for the conditions
def passes_Horizontal_Line_Test_A : Prop := false
def passes_Horizontal_Line_Test_B : Prop := true
def passes_Horizontal_Line_Test_C : Prop := true
def passes_Horizontal_Line_Test_D : Prop := false
def passes_Horizontal_Line_Test_E : Prop := false

-- Proof statement
theorem functions_with_inverses :
  (passes_Horizontal_Line_Test_A = false) ∧
  (passes_Horizontal_Line_Test_B = true) ∧
  (passes_Horizontal_Line_Test_C = true) ∧
  (passes_Horizontal_Line_Test_D = false) ∧
  (passes_Horizontal_Line_Test_E = false) →
  ([B, C] = which_functions_have_inverses) :=
sorry

end NUMINAMATH_GPT_functions_with_inverses_l732_73206


namespace NUMINAMATH_GPT_interview_passing_probability_l732_73232

def probability_of_passing_interview (p : ℝ) : ℝ :=
  p + (1 - p) * p + (1 - p) * (1 - p) * p

theorem interview_passing_probability : probability_of_passing_interview 0.7 = 0.973 :=
by
  -- proof steps to be filled
  sorry

end NUMINAMATH_GPT_interview_passing_probability_l732_73232


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l732_73251

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = - f x
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def is_even_function (h : ℝ → ℝ) : Prop := ∀ x, h (-x) = h x

-- This definition states that both f and g are either odd or even functions
def is_odd_or_even (f g : ℝ → ℝ) : Prop := 
  (is_odd f ∧ is_odd g) ∨ (is_even f ∧ is_even g)

theorem sufficient_but_not_necessary_condition (f g : ℝ → ℝ)
  (h : is_odd_or_even f g) : 
  ¬(is_odd f ∧ is_odd g) → is_even_function (f * g) :=
sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l732_73251


namespace NUMINAMATH_GPT_problem_l732_73277

theorem problem
  (a b : ℝ)
  (h : a^2 + b^2 - 2 * a + 6 * b + 10 = 0) :
  2 * a^100 - 3 * b⁻¹ = 3 := 
by {
  -- Proof steps go here
  sorry
}

end NUMINAMATH_GPT_problem_l732_73277


namespace NUMINAMATH_GPT_find_n_l732_73279

theorem find_n (n : ℝ) (h : (2 : ℂ) / (1 - I) = (1 : ℂ) + n * I) : n = 1 := by
  sorry

end NUMINAMATH_GPT_find_n_l732_73279


namespace NUMINAMATH_GPT_parabola_y_intercepts_l732_73220

theorem parabola_y_intercepts : ∃ y1 y2 : ℝ, (3 * y1^2 - 6 * y1 + 2 = 0) ∧ (3 * y2^2 - 6 * y2 + 2 = 0) ∧ (y1 ≠ y2) :=
by 
  sorry

end NUMINAMATH_GPT_parabola_y_intercepts_l732_73220


namespace NUMINAMATH_GPT_remainder_of_3_pow_100_mod_7_is_4_l732_73227

theorem remainder_of_3_pow_100_mod_7_is_4
  (h1 : 3^1 ≡ 3 [MOD 7])
  (h2 : 3^2 ≡ 2 [MOD 7])
  (h3 : 3^3 ≡ 6 [MOD 7])
  (h4 : 3^4 ≡ 4 [MOD 7])
  (h5 : 3^5 ≡ 5 [MOD 7])
  (h6 : 3^6 ≡ 1 [MOD 7]) :
  3^100 ≡ 4 [MOD 7] :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_3_pow_100_mod_7_is_4_l732_73227


namespace NUMINAMATH_GPT_A_equals_k_with_conditions_l732_73248

theorem A_equals_k_with_conditions (n m : ℕ) (h_n : 1 < n) (h_m : 1 < m) :
  ∃ k : ℤ, (1 : ℝ) < k ∧ (( (n + Real.sqrt (n^2 - 4)) / 2 ) ^ m = (k + Real.sqrt (k^2 - 4)) / 2) :=
sorry

end NUMINAMATH_GPT_A_equals_k_with_conditions_l732_73248


namespace NUMINAMATH_GPT_triangle_sine_equality_l732_73274

theorem triangle_sine_equality {a b c : ℝ} {α β γ : ℝ} 
  (cos_rule : c^2 = a^2 + b^2 - 2 * a * b * Real.cos γ)
  (area : ∃ T : ℝ, T = (1 / 2) * a * b * Real.sin γ)
  (sin_addition_γ : Real.sin (γ + Real.pi / 6) = Real.sin γ * (Real.sqrt 3 / 2) + Real.cos γ * (1 / 2))
  (sin_addition_β : Real.sin (β + Real.pi / 6) = Real.sin β * (Real.sqrt 3 / 2) + Real.cos β * (1 / 2))
  (sin_addition_α : Real.sin (α + Real.pi / 6) = Real.sin α * (Real.sqrt 3 / 2) + Real.cos α * (1 / 2)) :
  c^2 + 2 * a * b * Real.sin (γ + Real.pi / 6) = b^2 + 2 * a * c * Real.sin (β + Real.pi / 6) ∧
  b^2 + 2 * a * c * Real.sin (β + Real.pi / 6) = a^2 + 2 * b * c * Real.sin (α + Real.pi / 6) :=
sorry

end NUMINAMATH_GPT_triangle_sine_equality_l732_73274


namespace NUMINAMATH_GPT_number_of_full_boxes_l732_73292

theorem number_of_full_boxes (peaches_in_basket baskets_eaten_peaches box_capacity : ℕ) (h1 : peaches_in_basket = 23) (h2 : baskets = 7) (h3 : eaten_peaches = 7) (h4 : box_capacity = 13) :
  (peaches_in_basket * baskets - eaten_peaches) / box_capacity = 11 :=
by
  sorry

end NUMINAMATH_GPT_number_of_full_boxes_l732_73292


namespace NUMINAMATH_GPT_number_of_girls_on_playground_l732_73204

theorem number_of_girls_on_playground (boys girls total : ℕ) 
  (h1 : boys = 44) (h2 : total = 97) (h3 : total = boys + girls) : 
  girls = 53 :=
by sorry

end NUMINAMATH_GPT_number_of_girls_on_playground_l732_73204


namespace NUMINAMATH_GPT_intersect_circle_line_l732_73214

theorem intersect_circle_line (k m : ℝ) : 
  (∃ (x y : ℝ), y = k * x + 2 * k ∧ x^2 + y^2 + m * x + 4 = 0) → m > 4 :=
by
  -- This statement follows from the conditions given in the problem
  -- You can use implicit for pure documentation
  -- We include a sorry here to skip the proof
  sorry

end NUMINAMATH_GPT_intersect_circle_line_l732_73214


namespace NUMINAMATH_GPT_trigonometric_relationship_l732_73291

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < (π / 2))
variable (hβ : 0 < β ∧ β < π)
variable (h : Real.tan α = Real.cos β / (1 - Real.sin β))

theorem trigonometric_relationship : 
    2 * α - β = π / 2 :=
sorry

end NUMINAMATH_GPT_trigonometric_relationship_l732_73291


namespace NUMINAMATH_GPT_ratio_of_volumes_of_cones_l732_73250

theorem ratio_of_volumes_of_cones (r θ h1 h2 : ℝ) (hθ : 3 * θ + 4 * θ = 2 * π)
    (hr1 : r₁ = 3 * r / 7) (hr2 : r₂ = 4 * r / 7) :
    let V₁ := (1 / 3) * π * r₁^2 * h1
    let V₂ := (1 / 3) * π * r₂^2 * h2
    V₁ / V₂ = (9 : ℝ) / 16 := by
  sorry

end NUMINAMATH_GPT_ratio_of_volumes_of_cones_l732_73250


namespace NUMINAMATH_GPT_third_number_eq_l732_73231

theorem third_number_eq :
  ∃ x : ℝ, (0.625 * 0.0729 * x) / (0.0017 * 0.025 * 8.1) = 382.5 ∧ x = 2.33075 := 
by
  sorry

end NUMINAMATH_GPT_third_number_eq_l732_73231


namespace NUMINAMATH_GPT_expression_equiv_l732_73236

theorem expression_equiv :
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) = 5^128 - 4^128 :=
by
  sorry

end NUMINAMATH_GPT_expression_equiv_l732_73236


namespace NUMINAMATH_GPT_potato_bag_weight_l732_73268

-- Defining the weight of the bag of potatoes as a variable W
variable (W : ℝ)

-- Given condition: The weight of the bag is described by the equation
def weight_condition (W : ℝ) := W = 12 / (W / 2)

-- Proving the weight of the bag of potatoes is 12 lbs:
theorem potato_bag_weight : weight_condition W → W = 12 :=
by
  sorry

end NUMINAMATH_GPT_potato_bag_weight_l732_73268


namespace NUMINAMATH_GPT_first_comparison_second_comparison_l732_73257

theorem first_comparison (x y : ℕ) (h1 : x = 2^40) (h2 : y = 3^28) : x < y := 
by sorry

theorem second_comparison (a b : ℕ) (h3 : a = 31^11) (h4 : b = 17^14) : a < b := 
by sorry

end NUMINAMATH_GPT_first_comparison_second_comparison_l732_73257


namespace NUMINAMATH_GPT_original_number_l732_73238

theorem original_number (x : ℝ) (h1 : 1.5 * x = 135) : x = 90 :=
by
  sorry

end NUMINAMATH_GPT_original_number_l732_73238


namespace NUMINAMATH_GPT_non_empty_solution_set_inequality_l732_73289

theorem non_empty_solution_set_inequality (a : ℝ) :
  (∃ x : ℝ, |x + 1| - |x - 3| < a) ↔ a > -4 := 
sorry

end NUMINAMATH_GPT_non_empty_solution_set_inequality_l732_73289


namespace NUMINAMATH_GPT_f_properties_l732_73280

noncomputable def f : ℕ → ℕ := sorry

theorem f_properties (f : ℕ → ℕ) :
  (∀ x y : ℕ, x > 0 → y > 0 → f (x * y) = f x + f y) →
  (f 10 = 16) →
  (f 40 = 24) →
  (f 3 = 5) →
  (f 800 = 44) :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_f_properties_l732_73280


namespace NUMINAMATH_GPT_arrangement_with_one_ball_per_box_arrangement_with_one_empty_box_l732_73240

-- Definition of the first proof problem
theorem arrangement_with_one_ball_per_box:
  ∃ n : ℕ, n = 24 ∧ 
    -- Number of ways to arrange 4 different balls in 4 boxes such that each box has exactly one ball
    n = Nat.factorial 4 :=
by sorry

-- Definition of the second proof problem
theorem arrangement_with_one_empty_box:
  ∃ n : ℕ, n = 144 ∧ 
    -- Number of ways to arrange 4 different balls in 4 boxes such that exactly one box is empty
    n = Nat.choose 4 2 * Nat.factorial 3 :=
by sorry

end NUMINAMATH_GPT_arrangement_with_one_ball_per_box_arrangement_with_one_empty_box_l732_73240


namespace NUMINAMATH_GPT_sin_210_l732_73200

theorem sin_210 : Real.sin (210 * Real.pi / 180) = -1/2 := by
  sorry

end NUMINAMATH_GPT_sin_210_l732_73200


namespace NUMINAMATH_GPT_cube_problem_l732_73270

theorem cube_problem (n : ℕ) (h1 : n > 3) :
  (12 * (n - 4) = (n - 2)^3) → n = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_cube_problem_l732_73270


namespace NUMINAMATH_GPT_charlie_coins_worth_44_cents_l732_73293

-- Definitions based on the given conditions
def total_coins := 17
def p_eq_n_plus_2 (p n : ℕ) := p = n + 2

-- The main theorem stating the problem and the expected answer
theorem charlie_coins_worth_44_cents (p n : ℕ) (h1 : p + n = total_coins) (h2 : p_eq_n_plus_2 p n) :
  (7 * 5 + p * 1 = 44) :=
sorry

end NUMINAMATH_GPT_charlie_coins_worth_44_cents_l732_73293


namespace NUMINAMATH_GPT_hyperbola_asymptotes_correct_l732_73276

noncomputable def asymptotes_for_hyperbola : Prop :=
  ∀ (x y : ℂ),
    9 * (x : ℂ) ^ 2 - 4 * (y : ℂ) ^ 2 = -36 → 
    (y = (3 / 2) * (-Complex.I) * x) ∨ (y = -(3 / 2) * (-Complex.I) * x)

theorem hyperbola_asymptotes_correct :
  asymptotes_for_hyperbola := 
sorry

end NUMINAMATH_GPT_hyperbola_asymptotes_correct_l732_73276


namespace NUMINAMATH_GPT_workers_complete_time_l732_73264

theorem workers_complete_time 
  (time_A time_B time_C : ℕ) 
  (hA : time_A = 10)
  (hB : time_B = 12) 
  (hC : time_C = 15) : 
  let rate_A := (1: ℚ) / time_A
  let rate_B := (1: ℚ) / time_B
  let rate_C := (1: ℚ) / time_C
  let total_rate := rate_A + rate_B + rate_C
  1 / total_rate = 4 := 
by
  sorry

end NUMINAMATH_GPT_workers_complete_time_l732_73264


namespace NUMINAMATH_GPT_max_band_members_l732_73271

theorem max_band_members (n : ℤ) (h1 : 30 * n % 21 = 9) (h2 : 30 * n < 1500) : 30 * n ≤ 1470 :=
by
  -- Proof to be filled in later
  sorry

end NUMINAMATH_GPT_max_band_members_l732_73271


namespace NUMINAMATH_GPT_find_line_equation_l732_73287

noncomputable def perpendicular_origin_foot := 
  ∃ l : ℝ → ℝ → Prop,
    (∀ (x y : ℝ), l x y ↔ y = 2 * x + 5) ∧
    l (-2) 1

theorem find_line_equation : 
  ∃ l : ℝ → ℝ → Prop,
    (∀ (x y : ℝ), l x y ↔ 2 * x - y + 5 = 0) ∧
    l (-2) 1 ∧
    ∀ p q : ℝ, p = 0 → q = 0 → ¬ (l p q)
:= sorry

end NUMINAMATH_GPT_find_line_equation_l732_73287
