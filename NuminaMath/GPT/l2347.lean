import Mathlib

namespace NUMINAMATH_GPT_polynomial_divisible_by_7_l2347_234722

theorem polynomial_divisible_by_7 (n : ℤ) : 7 ∣ ((n + 7)^2 - n^2) :=
sorry

end NUMINAMATH_GPT_polynomial_divisible_by_7_l2347_234722


namespace NUMINAMATH_GPT_ribbons_left_l2347_234736

theorem ribbons_left {initial_ribbons morning_giveaway afternoon_giveaway ribbons_left : ℕ} 
    (h1 : initial_ribbons = 38) 
    (h2 : morning_giveaway = 14) 
    (h3 : afternoon_giveaway = 16) 
    (h4 : ribbons_left = initial_ribbons - (morning_giveaway + afternoon_giveaway)) : 
  ribbons_left = 8 := 
by 
  sorry

end NUMINAMATH_GPT_ribbons_left_l2347_234736


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l2347_234727

def M : Set ℤ := {m | -3 < m ∧ m < 2}
def N : Set ℤ := {n | -1 ≤ n ∧ n ≤ 3}

theorem intersection_of_M_and_N : M ∩ N = {-1, 0, 1} := 
by sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_l2347_234727


namespace NUMINAMATH_GPT_triangle_side_length_l2347_234715

theorem triangle_side_length (A B C : ℝ) (h1 : AC = Real.sqrt 2) (h2: AB = 2)
  (h3 : (Real.sqrt 3 * Real.sin A + Real.cos A) / (Real.sqrt 3 * Real.cos A - Real.sin A) = Real.tan (5 * Real.pi / 12)) :
  BC = Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_triangle_side_length_l2347_234715


namespace NUMINAMATH_GPT_system1_solution_l2347_234753

variable (x y : ℝ)

theorem system1_solution :
  (3 * x - y = -1) ∧ (x + 2 * y = 9) ↔ (x = 1) ∧ (y = 4) := by
  sorry

end NUMINAMATH_GPT_system1_solution_l2347_234753


namespace NUMINAMATH_GPT_min_x_y_l2347_234704

open Real

theorem min_x_y {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : 1/x + 9/y = 1) : x + y ≥ 16 := 
sorry

end NUMINAMATH_GPT_min_x_y_l2347_234704


namespace NUMINAMATH_GPT_alpha_div_beta_is_rational_l2347_234725

noncomputable def alpha_is_multiple (α : ℝ) (k : ℕ) : Prop :=
  ∃ k : ℕ, α = k * (2 * Real.pi / 1996)

noncomputable def beta_is_multiple (β : ℝ) (m : ℕ) : Prop :=
  β ≠ 0 ∧ ∃ m : ℕ, β = m * (2 * Real.pi / 1996)

theorem alpha_div_beta_is_rational (α β : ℝ) (k m : ℕ)
  (hα : alpha_is_multiple α k) (hβ : beta_is_multiple β m) :
  ∃ r : ℚ, α / β = r := by
    sorry

end NUMINAMATH_GPT_alpha_div_beta_is_rational_l2347_234725


namespace NUMINAMATH_GPT_ratio_of_dimensions_128_l2347_234773

noncomputable def volume128 (w l h : ℕ) : Prop := w * l * h = 128

theorem ratio_of_dimensions_128 (w l h : ℕ) (h_volume : volume128 w l h) : 
  ∃ wratio lratio, (w / l = wratio) ∧ (w / h = lratio) :=
sorry

end NUMINAMATH_GPT_ratio_of_dimensions_128_l2347_234773


namespace NUMINAMATH_GPT_colorings_equivalence_l2347_234757

-- Define the problem setup
structure ProblemSetup where
  n : ℕ  -- Number of disks (8)
  blue : ℕ  -- Number of blue disks (3)
  red : ℕ  -- Number of red disks (3)
  green : ℕ  -- Number of green disks (2)
  rotations : ℕ  -- Number of rotations (4: 90°, 180°, 270°, 360°)
  reflections : ℕ  -- Number of reflections (8: 4 through vertices and 4 through midpoints)

def number_of_colorings (setup : ProblemSetup) : ℕ :=
  sorry -- This represents the complex implementation details

def correct_answer : ℕ := 43

theorem colorings_equivalence : ∀ (setup : ProblemSetup),
  setup.n = 8 → setup.blue = 3 → setup.red = 3 → setup.green = 2 → setup.rotations = 4 → setup.reflections = 8 →
  number_of_colorings setup = correct_answer :=
by
  intros setup h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_colorings_equivalence_l2347_234757


namespace NUMINAMATH_GPT_middle_number_is_eight_l2347_234781

theorem middle_number_is_eight
    (x y z : ℕ)
    (h1 : x + y = 14)
    (h2 : x + z = 20)
    (h3 : y + z = 22) :
    y = 8 := by
  sorry

end NUMINAMATH_GPT_middle_number_is_eight_l2347_234781


namespace NUMINAMATH_GPT_standard_equation_line_standard_equation_circle_intersection_range_a_l2347_234726

theorem standard_equation_line (a t x y : ℝ) (h1 : x = a - 2 * t * y) (h2 : y = -4 * t) : 
    2 * x - y - 2 * a = 0 :=
sorry

theorem standard_equation_circle (θ x y : ℝ) (h1 : x = 4 * Real.cos θ) (h2 : y = 4 * Real.sin θ) : 
    x ^ 2 + y ^ 2 = 16 :=
sorry

theorem intersection_range_a (a : ℝ) (h : ∃ (t θ : ℝ), (a - 2 * t * (-4 * t)) = 4 * (Real.cos θ) ∧ (-4 * t) = 4 * (Real.sin θ)) :
    -4 * Real.sqrt 5 <= a ∧ a <= 4 * Real.sqrt 5 :=
sorry

end NUMINAMATH_GPT_standard_equation_line_standard_equation_circle_intersection_range_a_l2347_234726


namespace NUMINAMATH_GPT_four_digit_numbers_condition_l2347_234719

theorem four_digit_numbers_condition :
  ∃ (N : Nat), (1000 ≤ N ∧ N < 10000) ∧
               (∃ x a : Nat, N = 1000 * a + x ∧ x = 200 * a ∧ 1 ≤ a ∧ a ≤ 4) :=
by
  sorry

end NUMINAMATH_GPT_four_digit_numbers_condition_l2347_234719


namespace NUMINAMATH_GPT_slices_left_for_tomorrow_is_four_l2347_234784

def initial_slices : ℕ := 12
def lunch_slices : ℕ := initial_slices / 2
def remaining_slices_after_lunch : ℕ := initial_slices - lunch_slices
def dinner_slices : ℕ := remaining_slices_after_lunch / 3
def slices_left_for_tomorrow : ℕ := remaining_slices_after_lunch - dinner_slices

theorem slices_left_for_tomorrow_is_four : slices_left_for_tomorrow = 4 := by
  sorry

end NUMINAMATH_GPT_slices_left_for_tomorrow_is_four_l2347_234784


namespace NUMINAMATH_GPT_running_time_of_BeastOfWar_is_100_l2347_234754

noncomputable def Millennium := 120  -- minutes
noncomputable def AlphaEpsilon := Millennium - 30  -- minutes
noncomputable def BeastOfWar := AlphaEpsilon + 10  -- minutes
noncomputable def DeltaSquadron := 2 * BeastOfWar  -- minutes

theorem running_time_of_BeastOfWar_is_100 :
  BeastOfWar = 100 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_running_time_of_BeastOfWar_is_100_l2347_234754


namespace NUMINAMATH_GPT_samara_tire_spending_l2347_234783

theorem samara_tire_spending :
  ∀ (T : ℕ), 
    (2457 = 25 + 79 + T + 1886) → 
    T = 467 :=
by intros T h
   sorry

end NUMINAMATH_GPT_samara_tire_spending_l2347_234783


namespace NUMINAMATH_GPT_domain_of_f_l2347_234760

noncomputable def f (x : ℝ) : ℝ := 1 / x + Real.log (x + 2)

theorem domain_of_f :
  {x : ℝ | (x ≠ 0) ∧ (x > -2)} = {x : ℝ | (-2 < x ∧ x < 0) ∨ (0 < x)} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l2347_234760


namespace NUMINAMATH_GPT_exposed_surface_area_l2347_234793

theorem exposed_surface_area (r h : ℝ) (π : ℝ) (sphere_surface_area : ℝ) (cylinder_lateral_surface_area : ℝ) 
  (cond1 : r = 10) (cond2 : h = 5) (cond3 : sphere_surface_area = 4 * π * r^2) 
  (cond4 : cylinder_lateral_surface_area = 2 * π * r * h) :
  let hemisphere_curved_surface_area := sphere_surface_area / 2
  let hemisphere_base_area := π * r^2
  let total_surface_area := hemisphere_curved_surface_area + hemisphere_base_area + cylinder_lateral_surface_area
  total_surface_area = 400 * π :=
by
  sorry

end NUMINAMATH_GPT_exposed_surface_area_l2347_234793


namespace NUMINAMATH_GPT_octahedron_non_blue_probability_l2347_234724

theorem octahedron_non_blue_probability :
  let total_faces := 8
  let blue_faces := 3
  let red_faces := 3
  let green_faces := 2
  let non_blue_faces := total_faces - blue_faces
  (non_blue_faces / total_faces : ℚ) = (5 / 8 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_octahedron_non_blue_probability_l2347_234724


namespace NUMINAMATH_GPT_value_of_coins_is_77_percent_l2347_234769

theorem value_of_coins_is_77_percent :
  let pennies := 2 * 1  -- value of two pennies in cents
  let nickel := 5       -- value of one nickel in cents
  let dimes := 2 * 10   -- value of two dimes in cents
  let half_dollar := 50 -- value of one half-dollar in cents
  let total_cents := pennies + nickel + dimes + half_dollar
  let dollar_in_cents := 100
  (total_cents / dollar_in_cents) * 100 = 77 :=
by
  sorry

end NUMINAMATH_GPT_value_of_coins_is_77_percent_l2347_234769


namespace NUMINAMATH_GPT_space_shuttle_speed_conversion_l2347_234717

-- Define the given conditions
def speed_km_per_sec : ℕ := 6  -- Speed in km/s
def seconds_per_hour : ℕ := 3600  -- Seconds in an hour

-- Define the computed speed in km/hr
def expected_speed_km_per_hr : ℕ := 21600  -- Expected speed in km/hr

-- The main theorem statement to be proven
theorem space_shuttle_speed_conversion : speed_km_per_sec * seconds_per_hour = expected_speed_km_per_hr := by
  sorry

end NUMINAMATH_GPT_space_shuttle_speed_conversion_l2347_234717


namespace NUMINAMATH_GPT_probability_sum_even_for_three_cubes_l2347_234712

-- Define the probability function
def probability_even_sum (n: ℕ) : ℚ :=
  if n > 0 then 1 / 2 else 0

theorem probability_sum_even_for_three_cubes : probability_even_sum 3 = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_probability_sum_even_for_three_cubes_l2347_234712


namespace NUMINAMATH_GPT_sum_of_fractions_l2347_234746

theorem sum_of_fractions : 
  (1 / 1.01) + (1 / 1.1) + (1 / 1) + (1 / 11) + (1 / 101) = 3 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_fractions_l2347_234746


namespace NUMINAMATH_GPT_meaning_of_implication_l2347_234774

theorem meaning_of_implication (p q : Prop) : (p → q) = ((p → q) = True) :=
sorry

end NUMINAMATH_GPT_meaning_of_implication_l2347_234774


namespace NUMINAMATH_GPT_magic_8_ball_probability_l2347_234792

def probability_positive (p_pos : ℚ) (questions : ℕ) (positive_responses : ℕ) : ℚ :=
  (Nat.choose questions positive_responses : ℚ) * (p_pos ^ positive_responses) * ((1 - p_pos) ^ (questions - positive_responses))

theorem magic_8_ball_probability :
  probability_positive (1/3) 7 3 = 560 / 2187 :=
by
  sorry

end NUMINAMATH_GPT_magic_8_ball_probability_l2347_234792


namespace NUMINAMATH_GPT_find_c_l2347_234702

def P (x : ℝ) (c : ℝ) : ℝ :=
  x^3 + 3*x^2 + c*x + 15

theorem find_c (c : ℝ) : (x - 3 = P x c → c = -23) := by
  sorry

end NUMINAMATH_GPT_find_c_l2347_234702


namespace NUMINAMATH_GPT_find_common_ratio_l2347_234700

-- Define the geometric sequence
def geometric_sequence (a₁ q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n-1)

-- Given conditions
lemma a2_eq_8 (a₁ q : ℝ) : geometric_sequence a₁ q 2 = 8 :=
by sorry

lemma a5_eq_64 (a₁ q : ℝ) : geometric_sequence a₁ q 5 = 64 :=
by sorry

-- The common ratio q
theorem find_common_ratio (a₁ q : ℝ) (hq : 0 < q) :
  (geometric_sequence a₁ q 2 = 8) → (geometric_sequence a₁ q 5 = 64) → q = 2 :=
by sorry

end NUMINAMATH_GPT_find_common_ratio_l2347_234700


namespace NUMINAMATH_GPT_original_price_of_dish_l2347_234714

theorem original_price_of_dish (P : ℝ) (h1 : ∃ P, John's_payment = (0.9 * P) + (0.15 * P))
                               (h2 : ∃ P, Jane's_payment = (0.9 * P) + (0.135 * P))
                               (h3 : John's_payment = Jane's_payment + 0.51) : P = 34 := by
  -- John's Payment
  let John's_payment := (0.9 * P) + (0.15 * P)
  -- Jane's Payment
  let Jane's_payment := (0.9 * P) + (0.135 * P)
  -- Condition that John paid $0.51 more than Jane
  have h3 : John's_payment = Jane's_payment + 0.51 := sorry
  -- From the given conditions, we need to prove P = 34
  sorry

end NUMINAMATH_GPT_original_price_of_dish_l2347_234714


namespace NUMINAMATH_GPT_total_preparation_and_cooking_time_l2347_234740

def time_to_chop_pepper := 3
def time_to_chop_onion := 4
def time_to_slice_mushroom := 2
def time_to_dice_tomato := 3
def time_to_grate_cheese := 1
def time_to_assemble_and_cook_omelet := 6

def num_peppers := 8
def num_onions := 4
def num_mushrooms := 6
def num_tomatoes := 6
def num_omelets := 10

theorem total_preparation_and_cooking_time :
  (num_peppers * time_to_chop_pepper) +
  (num_onions * time_to_chop_onion) +
  (num_mushrooms * time_to_slice_mushroom) +
  (num_tomatoes * time_to_dice_tomato) +
  (num_omelets * time_to_grate_cheese) +
  (num_omelets * time_to_assemble_and_cook_omelet) = 140 :=
by
  sorry

end NUMINAMATH_GPT_total_preparation_and_cooking_time_l2347_234740


namespace NUMINAMATH_GPT_value_of_expression_l2347_234771

theorem value_of_expression (a b : ℤ) (h1 : a = 4) (h2 : b = -1) : -a^2 - b^2 + a * b = -21 := by
  sorry

end NUMINAMATH_GPT_value_of_expression_l2347_234771


namespace NUMINAMATH_GPT_apples_total_l2347_234767

theorem apples_total (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end NUMINAMATH_GPT_apples_total_l2347_234767


namespace NUMINAMATH_GPT_a_divides_b_l2347_234798

theorem a_divides_b (a b : ℕ) (h_pos : 0 < a ∧ 0 < b)
    (h : ∀ n : ℕ, a^n ∣ b^(n+1)) : a ∣ b :=
by
  sorry

end NUMINAMATH_GPT_a_divides_b_l2347_234798


namespace NUMINAMATH_GPT_parallel_lines_m_eq_neg4_l2347_234729

theorem parallel_lines_m_eq_neg4 (m : ℝ) (h1 : (m-2) ≠ -m) 
  (h2 : (m-2) / 3 = -m / (m + 2)) : m = -4 :=
sorry

end NUMINAMATH_GPT_parallel_lines_m_eq_neg4_l2347_234729


namespace NUMINAMATH_GPT_miles_on_first_day_l2347_234733

variable (x : ℝ)

/-- The distance traveled on the first day is x miles. -/
noncomputable def second_day_distance := (3/4) * x

/-- The distance traveled on the second day is (3/4)x miles. -/
noncomputable def third_day_distance := (1/2) * (x + second_day_distance x)

theorem miles_on_first_day
    (total_distance : x + second_day_distance x + third_day_distance x = 525)
    : x = 200 :=
sorry

end NUMINAMATH_GPT_miles_on_first_day_l2347_234733


namespace NUMINAMATH_GPT_solution_set_of_quadratic_inequality_l2347_234701

theorem solution_set_of_quadratic_inequality 
  (f : ℝ → ℝ) 
  (h₁ : ∀ x, f x < 0 ↔ x < -1 ∨ x > 1 / 3)
  (h₂ : ∀ x, f (Real.exp x) > 0 ↔ x < -Real.log 3) : 
  ∀ x, f (Real.exp x) > 0 ↔ x < -Real.log 3 := 
by
  intro x
  exact h₂ x

end NUMINAMATH_GPT_solution_set_of_quadratic_inequality_l2347_234701


namespace NUMINAMATH_GPT_profit_from_ad_l2347_234721

def advertising_cost : ℝ := 1000
def customers : ℕ := 100
def purchase_rate : ℝ := 0.8
def purchase_price : ℝ := 25

theorem profit_from_ad (advertising_cost customers purchase_rate purchase_price : ℝ) : 
  (customers * purchase_rate * purchase_price - advertising_cost) = 1000 :=
by
  -- assumptions as conditions
  let bought_customers := (customers : ℝ) * purchase_rate
  let revenue := bought_customers * purchase_price
  let profit := revenue - advertising_cost
  -- state the proof goal
  have goal : profit = 1000 :=
    sorry
  exact goal

end NUMINAMATH_GPT_profit_from_ad_l2347_234721


namespace NUMINAMATH_GPT_find_missing_number_l2347_234787

theorem find_missing_number (x : ℤ) : x + 64 = 16 → x = -48 := by
  intro h
  linarith

end NUMINAMATH_GPT_find_missing_number_l2347_234787


namespace NUMINAMATH_GPT_median_inequality_l2347_234750

variables {α : ℝ} (A B C M : Point) (a b c : ℝ)

-- Definitions and conditions
def isTriangle (A B C : Point) : Prop := -- definition of triangle
sorry

def isMedian (A B C M : Point) : Prop := -- definition of median
sorry

-- Statement we want to prove
theorem median_inequality (h1 : isTriangle A B C) (h2 : isMedian A B C M) :
  2 * AM ≥ (b + c) * Real.cos (α / 2) :=
sorry

end NUMINAMATH_GPT_median_inequality_l2347_234750


namespace NUMINAMATH_GPT_time_b_used_l2347_234775

noncomputable def time_b_used_for_proof : ℚ :=
  let C : ℚ := 1
  let C_a : ℚ := 1 / 4 * C
  let t_a : ℚ := 15
  let p_a : ℚ := 1 / 3
  let p_b : ℚ := 2 / 3
  let ratio : ℚ := (C_a * t_a) / ((C - C_a) * (t_a * p_a / p_b))
  t_a * p_a / p_b

theorem time_b_used : time_b_used_for_proof = 10 / 3 := by
  sorry

end NUMINAMATH_GPT_time_b_used_l2347_234775


namespace NUMINAMATH_GPT_smallest_number_among_neg2_neg1_0_pi_l2347_234764

/-- The smallest number among -2, -1, 0, and π is -2. -/
theorem smallest_number_among_neg2_neg1_0_pi : min (min (min (-2 : ℝ) (-1)) 0) π = -2 := 
sorry

end NUMINAMATH_GPT_smallest_number_among_neg2_neg1_0_pi_l2347_234764


namespace NUMINAMATH_GPT_min_value_inequality_l2347_234738

theorem min_value_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h_sum : x + y + z = 3) :
  (1 / x) + (4 / y) + (9 / z) ≥ 12 := 
sorry

end NUMINAMATH_GPT_min_value_inequality_l2347_234738


namespace NUMINAMATH_GPT_problem1_problem2_l2347_234728

-- Problem 1
theorem problem1 (x : ℚ) (h : x = -1/3) : 6 * x^2 + 5 * x^2 - 2 * (3 * x - 2 * x^2) = 11 / 3 :=
by sorry

-- Problem 2
theorem problem2 (a b : ℚ) (ha : a = -2) (hb : b = -1) : 5 * a^2 - a * b - 2 * (3 * a * b - (a * b - 2 * a^2)) = -6 :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_l2347_234728


namespace NUMINAMATH_GPT_sum_of_coordinates_B_l2347_234788

theorem sum_of_coordinates_B 
  (M : ℝ × ℝ)
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (hM_def : M = (-3, 2))
  (hA_def : A = (-8, 5))
  (hM_midpoint : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) :
  B.1 + B.2 = 1 := 
sorry

end NUMINAMATH_GPT_sum_of_coordinates_B_l2347_234788


namespace NUMINAMATH_GPT_find_b_l2347_234772

theorem find_b (a b : ℤ) (h1 : 3 * a + 1 = 4) (h2 : b - a = 1) : b = 2 :=
sorry

end NUMINAMATH_GPT_find_b_l2347_234772


namespace NUMINAMATH_GPT_chromium_percentage_new_alloy_l2347_234759

-- Define the weights and chromium percentages of the alloys
def weight_alloy1 : ℝ := 15
def weight_alloy2 : ℝ := 35
def chromium_percent_alloy1 : ℝ := 0.15
def chromium_percent_alloy2 : ℝ := 0.08

-- Define the theorem to calculate the chromium percentage of the new alloy
theorem chromium_percentage_new_alloy :
  ((weight_alloy1 * chromium_percent_alloy1 + weight_alloy2 * chromium_percent_alloy2)
  / (weight_alloy1 + weight_alloy2) * 100) = 10.1 :=
by
  sorry

end NUMINAMATH_GPT_chromium_percentage_new_alloy_l2347_234759


namespace NUMINAMATH_GPT_p_or_q_then_p_and_q_is_false_l2347_234710

theorem p_or_q_then_p_and_q_is_false (p q : Prop) (hpq : p ∨ q) : ¬(p ∧ q) :=
sorry

end NUMINAMATH_GPT_p_or_q_then_p_and_q_is_false_l2347_234710


namespace NUMINAMATH_GPT_flare_initial_velocity_and_duration_l2347_234730

noncomputable def h (v : ℝ) (t : ℝ) : ℝ := v * t - 4.9 * t^2

theorem flare_initial_velocity_and_duration (v t : ℝ) :
  (h v 5 = 245) ↔ (v = 73.5) ∧ (5 < t ∧ t < 10) :=
by {
  sorry
}

end NUMINAMATH_GPT_flare_initial_velocity_and_duration_l2347_234730


namespace NUMINAMATH_GPT_intersect_x_axis_once_l2347_234751

theorem intersect_x_axis_once (k : ℝ) : 
  (∀ x : ℝ, (k - 3) * x^2 + 2 * x + 1 = 0 → x = 0) → (k = 3 ∨ k = 4) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_intersect_x_axis_once_l2347_234751


namespace NUMINAMATH_GPT_ball_travel_approximately_80_l2347_234743

noncomputable def ball_travel_distance : ℝ :=
  let h₀ := 20
  let ratio := 2 / 3
  h₀ + -- first descent
  h₀ * ratio + -- first ascent
  h₀ * ratio + -- second descent
  h₀ * ratio^2 + -- second ascent
  h₀ * ratio^2 + -- third descent
  h₀ * ratio^3 + -- third ascent
  h₀ * ratio^3 + -- fourth descent
  h₀ * ratio^4 -- fourth ascent

theorem ball_travel_approximately_80 :
  abs (ball_travel_distance - 80) < 1 :=
sorry

end NUMINAMATH_GPT_ball_travel_approximately_80_l2347_234743


namespace NUMINAMATH_GPT_find_tangent_circles_tangent_circle_at_given_point_l2347_234755

noncomputable def circle_C (x y : ℝ) : Prop :=
  (x - 2)^2 + (y + 1)^2 = 4

def is_tangent (x y : ℝ) (a b : ℝ) : Prop :=
  ∃ (u v : ℝ), (u - a)^2 + (v - b)^2 = 1 ∧
  (x - u)^2 + (y - v)^2 = 4 ∧
  (x = u ∧ y = v)

theorem find_tangent_circles (x y a b : ℝ) (hx : circle_C x y)
  (ha_b : is_tangent x y a b) :
  (a = 5 ∧ b = -1) ∨ (a = 3 ∧ b = -1) :=
sorry

theorem tangent_circle_at_given_point (x y : ℝ) (hx : circle_C x y) (y_pos : y = -1)
  : ((x - 5)^2 + (y + 1)^2 = 1) ∨ ((x - 3)^2 + (y + 1)^2 = 1) :=
sorry

end NUMINAMATH_GPT_find_tangent_circles_tangent_circle_at_given_point_l2347_234755


namespace NUMINAMATH_GPT_root_expression_value_l2347_234770

theorem root_expression_value (a : ℝ) (h : a^2 + a - 1 = 0) : 2021 - 2 * a^2 - 2 * a = 2019 := 
by sorry

end NUMINAMATH_GPT_root_expression_value_l2347_234770


namespace NUMINAMATH_GPT_charlie_delta_four_products_l2347_234720

noncomputable def charlie_delta_purchase_ways : ℕ := 1363

theorem charlie_delta_four_products :
  let cakes := 6
  let cookies := 4
  let total := cakes + cookies
  ∃ ways : ℕ, ways = charlie_delta_purchase_ways :=
by
  sorry

end NUMINAMATH_GPT_charlie_delta_four_products_l2347_234720


namespace NUMINAMATH_GPT_joseph_cards_percentage_left_l2347_234735

theorem joseph_cards_percentage_left (h1 : ℕ := 16) (h2 : ℚ := 3/8) (h3 : ℕ := 2) :
  ((h1 - (h2 * h1 + h3)) / h1 * 100) = 50 :=
by
  sorry

end NUMINAMATH_GPT_joseph_cards_percentage_left_l2347_234735


namespace NUMINAMATH_GPT_symmetric_points_power_l2347_234789

theorem symmetric_points_power 
  (a b : ℝ) 
  (h1 : 2 * a = 8) 
  (h2 : 2 = a + b) :
  a^b = 1/16 := 
by sorry

end NUMINAMATH_GPT_symmetric_points_power_l2347_234789


namespace NUMINAMATH_GPT_solution_y_values_l2347_234732
-- Import the necessary libraries

-- Define the system of equations and the necessary conditions
def equation1 (x : ℝ) := x^2 - 6*x + 8 = 0
def equation2 (x y : ℝ) := 2*x - y = 6

-- The main theorem to be proven
theorem solution_y_values : ∃ x1 x2 y1 y2 : ℝ, 
  (equation1 x1 ∧ equation1 x2 ∧ equation2 x1 y1 ∧ equation2 x2 y2 ∧ 
  y1 = 2 ∧ y2 = -2) :=
by
  -- Use the provided solutions in the problem statement
  use 4, 2, 2, -2
  sorry  -- The details of the proof are omitted.

end NUMINAMATH_GPT_solution_y_values_l2347_234732


namespace NUMINAMATH_GPT_part1_part2_case1_part2_case2_part2_case3_1_part2_case3_2_part2_case3_3_l2347_234716

def f (a x : ℝ) : ℝ := a * x ^ 2 + (1 - a) * x + a - 2

theorem part1 (a : ℝ) : (∀ x : ℝ, f a x ≥ -2) ↔ a ≥ 1/3 :=
sorry

theorem part2_case1 (a : ℝ) (ha : a = 0) : ∀ x : ℝ, f a x < a - 1 ↔ x < 1 :=
sorry

theorem part2_case2 (a : ℝ) (ha : a > 0) : ∀ x : ℝ, (f a x < a - 1) ↔ (-1 / a < x ∧ x < 1) :=
sorry

theorem part2_case3_1 (a : ℝ) (ha : a = -1) : ∀ x : ℝ, (f a x < a - 1) ↔ x ≠ 1 :=
sorry

theorem part2_case3_2 (a : ℝ) (ha : -1 < a ∧ a < 0) : ∀ x : ℝ, (f a x < a - 1) ↔ (x > -1 / a ∨ x < 1) :=
sorry

theorem part2_case3_3 (a : ℝ) (ha : a < -1) : ∀ x : ℝ, (f a x < a - 1) ↔ (x > 1 ∨ x < -1 / a) :=
sorry

end NUMINAMATH_GPT_part1_part2_case1_part2_case2_part2_case3_1_part2_case3_2_part2_case3_3_l2347_234716


namespace NUMINAMATH_GPT_range_of_m_l2347_234766

theorem range_of_m (m : ℝ) : (∀ (x : ℝ), |3 - x| + |5 + x| > m) → m < 8 :=
sorry

end NUMINAMATH_GPT_range_of_m_l2347_234766


namespace NUMINAMATH_GPT_circle_eq_l2347_234756

theorem circle_eq (D E : ℝ) :
  (∀ {x y : ℝ}, (x = 0 ∧ y = 0) ∨
               (x = 4 ∧ y = 0) ∨
               (x = -1 ∧ y = 1) → 
               x^2 + y^2 + D * x + E * y = 0) →
  (D = -4 ∧ E = -6) :=
by
  intros h
  have h1 : 0^2 + 0^2 + D * 0 + E * 0 = 0 := by exact h (Or.inl ⟨rfl, rfl⟩)
  have h2 : 4^2 + 0^2 + D * 4 + E * 0 = 0 := by exact h (Or.inr (Or.inl ⟨rfl, rfl⟩))
  have h3 : (-1)^2 + 1^2 + D * (-1) + E * 1 = 0 := by exact h (Or.inr (Or.inr ⟨rfl, rfl⟩))
  sorry -- proof steps would go here to eventually show D = -4 and E = -6

end NUMINAMATH_GPT_circle_eq_l2347_234756


namespace NUMINAMATH_GPT_man_total_earnings_l2347_234762

-- Define the conditions
def total_days := 30
def wage_per_day := 10
def fine_per_absence := 2
def days_absent := 7
def days_worked := total_days - days_absent
def earned := days_worked * wage_per_day
def fine := days_absent * fine_per_absence
def total_earnings := earned - fine

-- State the theorem
theorem man_total_earnings : total_earnings = 216 := by
  -- Using the definitions provided, the proof should show that the calculations result in 216
  sorry

end NUMINAMATH_GPT_man_total_earnings_l2347_234762


namespace NUMINAMATH_GPT_range_of_u_l2347_234779

def satisfies_condition (x y : ℝ) : Prop := (x^2 / 3) + y^2 = 1

def u (x y : ℝ) : ℝ := |2 * x + y - 4| + |3 - x - 2 * y|

theorem range_of_u {x y : ℝ} (h : satisfies_condition x y) : ∀ u, 1 ≤ u ∧ u ≤ 13 :=
sorry

end NUMINAMATH_GPT_range_of_u_l2347_234779


namespace NUMINAMATH_GPT_alpha_plus_beta_l2347_234723

theorem alpha_plus_beta (α β : ℝ) (hα_range : -Real.pi / 2 < α ∧ α < Real.pi / 2)
    (hβ_range : -Real.pi / 2 < β ∧ β < Real.pi / 2)
    (h_roots : ∃ (x1 x2 : ℝ), x1 = Real.tan α ∧ x2 = Real.tan β ∧ (x1^2 + 3 * Real.sqrt 3 * x1 + 4 = 0) ∧ (x2^2 + 3 * Real.sqrt 3 * x2 + 4 = 0)) :
    α + β = -2 * Real.pi / 3 :=
sorry

end NUMINAMATH_GPT_alpha_plus_beta_l2347_234723


namespace NUMINAMATH_GPT_jade_initial_pieces_l2347_234748

theorem jade_initial_pieces (n w l p : ℕ) (hn : n = 11) (hw : w = 7) (hl : l = 23) (hp : p = n * w + l) : p = 100 :=
by
  sorry

end NUMINAMATH_GPT_jade_initial_pieces_l2347_234748


namespace NUMINAMATH_GPT_frank_total_points_l2347_234763

def points_defeating_enemies (enemies : ℕ) (points_per_enemy : ℕ) : ℕ :=
  enemies * points_per_enemy

def total_points (points_from_enemies : ℕ) (completion_points : ℕ) : ℕ :=
  points_from_enemies + completion_points

theorem frank_total_points :
  let enemies := 6
  let points_per_enemy := 9
  let completion_points := 8
  let points_from_enemies := points_defeating_enemies enemies points_per_enemy
  total_points points_from_enemies completion_points = 62 :=
by
  let enemies := 6
  let points_per_enemy := 9
  let completion_points := 8
  let points_from_enemies := points_defeating_enemies enemies points_per_enemy
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_frank_total_points_l2347_234763


namespace NUMINAMATH_GPT_distance_scientific_notation_l2347_234796

theorem distance_scientific_notation :
  55000000 = 5.5 * 10^7 :=
sorry

end NUMINAMATH_GPT_distance_scientific_notation_l2347_234796


namespace NUMINAMATH_GPT_no_solution_for_equation_l2347_234713

/-- The given equation expressed using letters as unique digits:
    ∑ (letters as digits) from БАРАНКА + БАРАБАН + КАРАБАС = ПАРАЗИТ
    We aim to prove that there are no valid digit assignments satisfying the equation. -/
theorem no_solution_for_equation :
  ∀ (b a r n k s p i t: ℕ),
  b ≠ a ∧ b ≠ r ∧ b ≠ n ∧ b ≠ k ∧ b ≠ s ∧ b ≠ p ∧ b ≠ i ∧ b ≠ t ∧
  a ≠ r ∧ a ≠ n ∧ a ≠ k ∧ a ≠ s ∧ a ≠ p ∧ a ≠ i ∧ a ≠ t ∧
  r ≠ n ∧ r ≠ k ∧ r ≠ s ∧ r ≠ p ∧ r ≠ i ∧ r ≠ t ∧
  n ≠ k ∧ n ≠ s ∧ n ≠ p ∧ n ≠ i ∧ n ≠ t ∧
  k ≠ s ∧ k ≠ p ∧ k ≠ i ∧ k ≠ t ∧
  s ≠ p ∧ s ≠ i ∧ s ≠ t ∧
  p ≠ i ∧ p ≠ t ∧
  i ≠ t →
  100000 * b + 10000 * a + 1000 * r + 100 * a + 10 * n + k +
  100000 * b + 10000 * a + 1000 * r + 100 * a + 10 * b + a + n +
  100000 * k + 10000 * a + 1000 * r + 100 * a + 10 * b + a + s ≠ 
  100000 * p + 10000 * a + 1000 * r + 100 * a + 10 * z + i + t :=
sorry

end NUMINAMATH_GPT_no_solution_for_equation_l2347_234713


namespace NUMINAMATH_GPT_product_of_binomials_l2347_234768

-- Definition of the binomials
def binomial1 (x : ℝ) : ℝ := 4 * x - 3
def binomial2 (x : ℝ) : ℝ := x + 7

-- The theorem to be proved
theorem product_of_binomials (x : ℝ) : 
  binomial1 x * binomial2 x = 4 * x^2 + 25 * x - 21 :=
by
  sorry

end NUMINAMATH_GPT_product_of_binomials_l2347_234768


namespace NUMINAMATH_GPT_CA_eq_A_intersection_CB_eq_l2347_234741

-- Definitions as per conditions
def U : Set ℝ := Set.univ
def A : Set ℝ := { x | x < 2 }
def B : Set ℝ := { x | x > 1 }

-- Proof problems as per questions and answers
theorem CA_eq : (U \ A) = { x : ℝ | x ≥ 2 } :=
by
  sorry

theorem A_intersection_CB_eq : (A ∩ (U \ B)) = { x : ℝ | x ≤ 1 } :=
by
  sorry

end NUMINAMATH_GPT_CA_eq_A_intersection_CB_eq_l2347_234741


namespace NUMINAMATH_GPT_negation_of_p_l2347_234752

theorem negation_of_p :
  (¬ (∀ x > 0, (x+1)*Real.exp x > 1)) ↔ 
  (∃ x ≤ 0, (x+1)*Real.exp x ≤ 1) :=
sorry

end NUMINAMATH_GPT_negation_of_p_l2347_234752


namespace NUMINAMATH_GPT_base_number_of_exponentiation_l2347_234742

theorem base_number_of_exponentiation (n : ℕ) (some_number : ℕ) (h1 : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = some_number^22) (h2 : n = 21) : some_number = 4 :=
  sorry

end NUMINAMATH_GPT_base_number_of_exponentiation_l2347_234742


namespace NUMINAMATH_GPT_roger_total_distance_l2347_234782

theorem roger_total_distance :
  let morning_ride_miles := 2
  let evening_ride_miles := 5 * morning_ride_miles
  let next_day_morning_ride_km := morning_ride_miles * 1.6
  let next_day_ride_km := 2 * next_day_morning_ride_km
  let next_day_ride_miles := next_day_ride_km / 1.6
  morning_ride_miles + evening_ride_miles + next_day_ride_miles = 16 :=
by
  sorry

end NUMINAMATH_GPT_roger_total_distance_l2347_234782


namespace NUMINAMATH_GPT_lateral_area_cone_l2347_234780

-- Define the cone problem with given conditions
def radius : ℝ := 5
def slant_height : ℝ := 10

-- Given these conditions, prove the lateral area is 50π
theorem lateral_area_cone (r : ℝ) (l : ℝ) (h_r : r = 5) (h_l : l = 10) : (1/2) * 2 * Real.pi * r * l = 50 * Real.pi :=
by 
  -- import useful mathematical tools
  sorry

end NUMINAMATH_GPT_lateral_area_cone_l2347_234780


namespace NUMINAMATH_GPT_remainder_of_P_div_by_D_is_333_l2347_234778

-- Define the polynomial P(x)
def P (x : ℝ) : ℝ := 8 * x^4 - 18 * x^3 + 27 * x^2 - 14 * x - 30

-- Define the divisor D(x) and simplify it, but this is not necessary for the theorem statement.
-- def D (x : ℝ) : ℝ := 4 * x - 12  

-- Prove the remainder is 333 when x = 3
theorem remainder_of_P_div_by_D_is_333 : P 3 = 333 := by
  sorry

end NUMINAMATH_GPT_remainder_of_P_div_by_D_is_333_l2347_234778


namespace NUMINAMATH_GPT_obtuse_vertex_angle_is_135_l2347_234785

-- Define the obtuse scalene triangle with the given properties
variables {a b c : ℝ} (triangle : Triangle ℝ)
variables (φ : ℝ) (h_obtuse : φ > 90 ∧ φ < 180) (h_scalene : a ≠ b ∧ b ≠ c ∧ c ≠ a)
variables (h_side_relation : a^2 + b^2 = 2 * c^2) (h_sine_obtuse : Real.sin φ = Real.sqrt 2 / 2)

-- The measure of the obtuse vertex angle is 135 degrees
theorem obtuse_vertex_angle_is_135 :
  φ = 135 := by
  sorry

end NUMINAMATH_GPT_obtuse_vertex_angle_is_135_l2347_234785


namespace NUMINAMATH_GPT_shirt_cost_l2347_234794

variables (J S : ℝ)

theorem shirt_cost :
  (3 * J + 2 * S = 69) ∧
  (2 * J + 3 * S = 86) →
  S = 24 :=
by
  sorry

end NUMINAMATH_GPT_shirt_cost_l2347_234794


namespace NUMINAMATH_GPT_find_other_number_l2347_234703

theorem find_other_number (A B : ℕ) (HCF LCM : ℕ)
  (hA : A = 24)
  (hHCF: (HCF : ℚ) = 16)
  (hLCM: (LCM : ℚ) = 312)
  (hHCF_LCM: HCF * LCM = A * B) : 
  B = 208 :=
by
  sorry

end NUMINAMATH_GPT_find_other_number_l2347_234703


namespace NUMINAMATH_GPT_correct_average_l2347_234711

theorem correct_average (initial_avg : ℝ) (n : ℕ) (error1 : ℝ) (wrong_num : ℝ) (correct_num : ℝ) :
  initial_avg = 40.2 → n = 10 → error1 = 19 → wrong_num = 13 → correct_num = 31 →
  (initial_avg * n - error1 - wrong_num + correct_num) / n = 40.1 :=
by
  intros
  sorry

end NUMINAMATH_GPT_correct_average_l2347_234711


namespace NUMINAMATH_GPT_max_min_difference_l2347_234761

open Real

theorem max_min_difference (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + 2 * y = 4) :
  ∃(max min : ℝ), (∀z, z = (|2 * x - y| / (|x| + |y|)) → z ≤ max) ∧ 
                  (∀z, z = (|2 * x - y| / (|x| + |y|)) → min ≤ z) ∧ 
                  (max - min = 5) :=
by
  sorry

end NUMINAMATH_GPT_max_min_difference_l2347_234761


namespace NUMINAMATH_GPT_solution_set_ineq_l2347_234708

theorem solution_set_ineq (x : ℝ) :
  (x - 1) / (1 - 2 * x) ≥ 0 ↔ (1 / 2 < x ∧ x ≤ 1) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_ineq_l2347_234708


namespace NUMINAMATH_GPT_son_work_time_l2347_234718

theorem son_work_time (M S : ℝ) 
  (hM : M = 1 / 4)
  (hCombined : M + S = 1 / 3) : 
  S = 1 / 12 :=
by
  sorry

end NUMINAMATH_GPT_son_work_time_l2347_234718


namespace NUMINAMATH_GPT_solve_equation_l2347_234745

theorem solve_equation {x y z : ℝ} (h₁ : x + 95 / 12 * y + 4 * z = 0)
  (h₂ : 4 * x + 95 / 12 * y - 3 * z = 0)
  (h₃ : 3 * x + 5 * y - 4 * z = 0)
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  x^2 * z / y^3 = -60 :=
sorry

end NUMINAMATH_GPT_solve_equation_l2347_234745


namespace NUMINAMATH_GPT_cos_half_pi_plus_alpha_correct_l2347_234786

noncomputable def cos_half_pi_plus_alpha
  (α : ℝ)
  (h1 : Real.tan α = -3/4)
  (h2 : α ∈ Set.Ioo (3 * Real.pi / 2) (2 * Real.pi)) : Real :=
  Real.cos (Real.pi / 2 + α)

theorem cos_half_pi_plus_alpha_correct
  (α : ℝ)
  (h1 : Real.tan α = -3/4)
  (h2 : α ∈ Set.Ioo (3 * Real.pi / 2) (2 * Real.pi)) :
  cos_half_pi_plus_alpha α h1 h2 = 3/5 := by
  sorry

end NUMINAMATH_GPT_cos_half_pi_plus_alpha_correct_l2347_234786


namespace NUMINAMATH_GPT_ratio_expression_l2347_234749

theorem ratio_expression (a b c : ℝ) (ha : a / b = 20) (hb : b / c = 10) : (a + b) / (b + c) = 210 / 11 := by
  sorry

end NUMINAMATH_GPT_ratio_expression_l2347_234749


namespace NUMINAMATH_GPT_graph_passes_through_point_l2347_234795

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x-2) + 1

theorem graph_passes_through_point (a : ℝ) (h_pos : a > 0) (h_neq : a ≠ 1) : f a 2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_graph_passes_through_point_l2347_234795


namespace NUMINAMATH_GPT_equal_parts_count_l2347_234799

def scale_length_in_inches : ℕ := (7 * 12) + 6
def part_length_in_inches : ℕ := 18
def number_of_parts (total_length part_length : ℕ) : ℕ := total_length / part_length

theorem equal_parts_count :
  number_of_parts scale_length_in_inches part_length_in_inches = 5 :=
by
  sorry

end NUMINAMATH_GPT_equal_parts_count_l2347_234799


namespace NUMINAMATH_GPT_smallest_n_lil_wayne_rain_l2347_234706

noncomputable def probability_rain (n : ℕ) : ℝ := 
  1 / 2 - 1 / 2^(n + 1)

theorem smallest_n_lil_wayne_rain :
  ∃ n : ℕ, probability_rain n > 0.499 ∧ (∀ m : ℕ, m < n → probability_rain m ≤ 0.499) ∧ n = 9 := 
by
  sorry

end NUMINAMATH_GPT_smallest_n_lil_wayne_rain_l2347_234706


namespace NUMINAMATH_GPT_pounds_per_ton_l2347_234747

theorem pounds_per_ton (weight_pounds : ℕ) (weight_tons : ℕ) (h_weight : weight_pounds = 6000) (h_tons : weight_tons = 3) : 
  weight_pounds / weight_tons = 2000 :=
by
  sorry

end NUMINAMATH_GPT_pounds_per_ton_l2347_234747


namespace NUMINAMATH_GPT_fraction_simplification_l2347_234709

theorem fraction_simplification (a b c x y : ℝ) (m : ℝ) :
  (∀ (x y : ℝ), (y ≠ 0 → (y^2 / x^2) ≠ (y / x))) ∧
  (∀ (a b c : ℝ), (a + c^2) / (b + c^2) ≠ a / b) ∧
  (∀ (a b m : ℝ), ¬(m ≠ -1 → (a + b) / (m * a + m * b) = 1 / 2)) ∧
  (∃ a b : ℝ, (a - b) / (b - a) = -1) :=
  by
  sorry

end NUMINAMATH_GPT_fraction_simplification_l2347_234709


namespace NUMINAMATH_GPT_bob_total_profit_l2347_234776

-- Define the given inputs
def n_dogs : ℕ := 2
def c_dog : ℝ := 250.00
def n_puppies : ℕ := 6
def c_food_vac : ℝ := 500.00
def c_ad : ℝ := 150.00
def p_puppy : ℝ := 350.00

-- The statement to prove
theorem bob_total_profit : 
  (n_puppies * p_puppy - (n_dogs * c_dog + c_food_vac + c_ad)) = 950.00 :=
by
  sorry

end NUMINAMATH_GPT_bob_total_profit_l2347_234776


namespace NUMINAMATH_GPT_cost_of_items_l2347_234737

namespace GardenCost

variables (B T C : ℝ)

/-- Given conditions defining the cost relationships and combined cost,
prove the specific costs of bench, table, and chair. -/
theorem cost_of_items
  (h1 : T + B + C = 650)
  (h2 : T = 2 * B - 50)
  (h3 : C = 1.5 * B - 25) :
  B = 161.11 ∧ T = 272.22 ∧ C = 216.67 :=
sorry

end GardenCost

end NUMINAMATH_GPT_cost_of_items_l2347_234737


namespace NUMINAMATH_GPT_total_number_of_trees_l2347_234777

-- Definitions of the conditions
def side_length : ℝ := 100
def trees_per_sq_meter : ℝ := 4

-- Calculations based on the conditions
def area_of_street : ℝ := side_length * side_length
def area_of_forest : ℝ := 3 * area_of_street

-- The statement to prove
theorem total_number_of_trees : 
  trees_per_sq_meter * area_of_forest = 120000 := 
sorry

end NUMINAMATH_GPT_total_number_of_trees_l2347_234777


namespace NUMINAMATH_GPT_remainder_of_division_l2347_234797

theorem remainder_of_division (x y R : ℕ) 
  (h1 : y = 1782)
  (h2 : y - x = 1500)
  (h3 : y = 6 * x + R) :
  R = 90 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_division_l2347_234797


namespace NUMINAMATH_GPT_jenna_less_than_bob_l2347_234765

theorem jenna_less_than_bob :
  ∀ (bob jenna phil : ℕ),
  (bob = 60) →
  (phil = bob / 3) →
  (jenna = 2 * phil) →
  (bob - jenna = 20) :=
by
  intros bob jenna phil h1 h2 h3
  sorry

end NUMINAMATH_GPT_jenna_less_than_bob_l2347_234765


namespace NUMINAMATH_GPT_sqrt_meaningful_range_l2347_234739

theorem sqrt_meaningful_range (x : ℝ) : 2 * x - 6 ≥ 0 ↔ x ≥ 3 := by
  sorry

end NUMINAMATH_GPT_sqrt_meaningful_range_l2347_234739


namespace NUMINAMATH_GPT_base5_addition_correct_l2347_234731

-- Definitions to interpret base-5 numbers
def base5_to_base10 (n : List ℕ) : ℕ :=
  n.reverse.foldl (λ acc d => acc * 5 + d) 0

-- Conditions given in the problem
def num1 : ℕ := base5_to_base10 [2, 0, 1, 4]  -- (2014)_5 in base-10
def num2 : ℕ := base5_to_base10 [2, 2, 3]    -- (223)_5 in base-10

-- Statement to prove
theorem base5_addition_correct :
  base5_to_base10 ([2, 0, 1, 4]) + base5_to_base10 ([2, 2, 3]) = base5_to_base10 ([2, 2, 4, 2]) :=
by
  -- Proof goes here
  sorry

#print axioms base5_addition_correct

end NUMINAMATH_GPT_base5_addition_correct_l2347_234731


namespace NUMINAMATH_GPT_sum_a_b_eq_negative_one_l2347_234734

theorem sum_a_b_eq_negative_one 
  (a b : ℝ) 
  (h1 : ∀ x : ℝ, 2 < x ∧ x < 3 → x^2 - a * x - b < 0)
  (h2 : ∀ x : ℝ, x^2 - a * x - b = 0 → x = 2 ∨ x = 3) :
  a + b = -1 := 
sorry

end NUMINAMATH_GPT_sum_a_b_eq_negative_one_l2347_234734


namespace NUMINAMATH_GPT_book_E_chapters_l2347_234758

def total_chapters: ℕ := 97
def chapters_A: ℕ := 17
def chapters_B: ℕ := chapters_A + 5
def chapters_C: ℕ := chapters_B - 7
def chapters_D: ℕ := chapters_C * 2
def chapters_sum : ℕ := chapters_A + chapters_B + chapters_C + chapters_D

theorem book_E_chapters :
  total_chapters - chapters_sum = 13 :=
by
  sorry

end NUMINAMATH_GPT_book_E_chapters_l2347_234758


namespace NUMINAMATH_GPT_planting_trees_system_of_equations_l2347_234705

/-- This formalizes the problem where we have 20 young pioneers in total, 
each boy planted 3 trees, each girl planted 2 trees,
and together they planted a total of 52 tree seedlings.
We need to formalize proving that the system of linear equations is as follows:
x + y = 20
3x + 2y = 52
-/
theorem planting_trees_system_of_equations (x y : ℕ) (h1 : x + y = 20)
  (h2 : 3 * x + 2 * y = 52) : 
  (x + y = 20 ∧ 3 * x + 2 * y = 52) :=
by
  exact ⟨h1, h2⟩

end NUMINAMATH_GPT_planting_trees_system_of_equations_l2347_234705


namespace NUMINAMATH_GPT_find_number_l2347_234707

theorem find_number (quotient : ℕ) (remainder : ℕ) (divisor : ℕ) (number : ℕ) :
  quotient = 9 ∧ remainder = 1 ∧ divisor = 30 → number = 271 := by
  intro h
  sorry

end NUMINAMATH_GPT_find_number_l2347_234707


namespace NUMINAMATH_GPT_different_ways_to_eat_spaghetti_l2347_234791

-- Define the conditions
def red_spaghetti := 5
def blue_spaghetti := 5
def total_spaghetti := 6

-- This is the proof statement
theorem different_ways_to_eat_spaghetti : 
  ∃ (ways : ℕ), ways = 62 ∧ 
  (∃ r b : ℕ, r ≤ red_spaghetti ∧ b ≤ blue_spaghetti ∧ r + b = total_spaghetti) := 
sorry

end NUMINAMATH_GPT_different_ways_to_eat_spaghetti_l2347_234791


namespace NUMINAMATH_GPT_lcm_of_two_numbers_l2347_234790

theorem lcm_of_two_numbers (a b : ℕ) (h_ratio : a * 3 = b * 2) (h_sum : a + b = 30) : Nat.lcm a b = 18 :=
  sorry

end NUMINAMATH_GPT_lcm_of_two_numbers_l2347_234790


namespace NUMINAMATH_GPT_factorize_expression_l2347_234744

theorem factorize_expression (a b : ℝ) : a * b^2 - 9 * a = a * (b + 3) * (b - 3) :=
by 
  sorry

end NUMINAMATH_GPT_factorize_expression_l2347_234744
