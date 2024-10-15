import Mathlib

namespace NUMINAMATH_GPT_compare_solutions_l1135_113505

theorem compare_solutions 
  (c d p q : ℝ) 
  (hc : c ≠ 0) 
  (hp : p ≠ 0) :
  (-d / c) < (-q / p) ↔ (q / p) < (d / c) :=
by
  sorry

end NUMINAMATH_GPT_compare_solutions_l1135_113505


namespace NUMINAMATH_GPT_complex_div_eq_i_l1135_113573

open Complex

theorem complex_div_eq_i : (1 + I) / (1 - I) = I := by
  sorry

end NUMINAMATH_GPT_complex_div_eq_i_l1135_113573


namespace NUMINAMATH_GPT_two_digit_sum_of_original_and_reverse_l1135_113518

theorem two_digit_sum_of_original_and_reverse
  (a b : ℕ)
  (h1 : 1 ≤ a ∧ a ≤ 9) -- a is a digit
  (h2 : 0 ≤ b ∧ b ≤ 9) -- b is a digit
  (h3 : (10 * a + b) - (10 * b + a) = 7 * (a + b)) :
  (10 * a + b) + (10 * b + a) = 99 :=
by
  sorry

end NUMINAMATH_GPT_two_digit_sum_of_original_and_reverse_l1135_113518


namespace NUMINAMATH_GPT_triangle_larger_segment_cutoff_l1135_113506

open Real

theorem triangle_larger_segment_cutoff (a b c h s₁ s₂ : ℝ) (habc : a = 35) (hbc : b = 85) (hca : c = 90)
  (hh : h = 90)
  (eq₁ : a^2 = s₁^2 + h^2)
  (eq₂ : b^2 = s₂^2 + h^2)
  (h_sum : s₁ + s₂ = c) :
  max s₁ s₂ = 78.33 :=
by
  sorry

end NUMINAMATH_GPT_triangle_larger_segment_cutoff_l1135_113506


namespace NUMINAMATH_GPT_min_value_k_l1135_113569

variables (x : ℕ → ℚ) (k n c : ℚ)

theorem min_value_k
  (k_gt_one : k > 1) -- condition that k > 1
  (n_gt_2018 : n > 2018) -- condition that n > 2018
  (n_odd : n % 2 = 1) -- condition that n is odd
  (non_zero_rational : ∀ i : ℕ, x i ≠ 0) -- non-zero rational numbers x₁, x₂, ..., xₙ
  (not_all_equal : ∃ i j : ℕ, x i ≠ x j) -- they are not all equal
  (relations : ∀ i : ℕ, x i + k / x (i + 1) = c) -- given relations
  : k = 4 :=
sorry

end NUMINAMATH_GPT_min_value_k_l1135_113569


namespace NUMINAMATH_GPT_second_number_value_l1135_113577

def first_number := ℚ
def second_number := ℚ

variables (x y : ℚ)

/-- Given conditions: 
      (1) \( \frac{1}{5}x = \frac{5}{8}y \)
      (2) \( x + 35 = 4y \)
    Prove that \( y = 40 \) 
-/
theorem second_number_value (h1 : (1/5 : ℚ) * x = (5/8 : ℚ) * y) (h2 : x + 35 = 4 * y) : 
  y = 40 :=
sorry

end NUMINAMATH_GPT_second_number_value_l1135_113577


namespace NUMINAMATH_GPT_circle_radius_center_l1135_113585

theorem circle_radius_center (x y : ℝ) (h : x^2 + y^2 - 2*x - 2*y - 2 = 0) :
  (∃ a b r, (x - a)^2 + (y - b)^2 = r^2 ∧ a = 1 ∧ b = 1 ∧ r = 2) := 
sorry

end NUMINAMATH_GPT_circle_radius_center_l1135_113585


namespace NUMINAMATH_GPT_largest_n_divisible_by_n_plus_10_l1135_113528

theorem largest_n_divisible_by_n_plus_10 :
  ∃ n : ℕ, (n^3 + 100) % (n + 10) = 0 ∧ ∀ m : ℕ, ((m^3 + 100) % (m + 10) = 0 → m ≤ n) ∧ n = 890 := 
sorry

end NUMINAMATH_GPT_largest_n_divisible_by_n_plus_10_l1135_113528


namespace NUMINAMATH_GPT_translate_quadratic_l1135_113513

-- Define the original quadratic function
def original_quadratic (x : ℝ) : ℝ := (x - 2)^2 - 4

-- Define the translation of the graph one unit to the left and two units up
def translated_quadratic (x : ℝ) : ℝ := (x - 1)^2 - 2

-- Statement to be proved
theorem translate_quadratic :
  ∀ x : ℝ, translated_quadratic x = original_quadratic (x-1) + 2 :=
by
  intro x
  unfold translated_quadratic original_quadratic
  sorry

end NUMINAMATH_GPT_translate_quadratic_l1135_113513


namespace NUMINAMATH_GPT_chang_apple_problem_l1135_113500

theorem chang_apple_problem 
  (A : ℝ)
  (h1 : 0.50 * A * 0.50 + 0.25 * A * 0.10 + 0.15 * A * 0.30 + 0.10 * A * 0.20 = 80)
  : A = 235 := 
sorry

end NUMINAMATH_GPT_chang_apple_problem_l1135_113500


namespace NUMINAMATH_GPT_solve_for_y_l1135_113535

theorem solve_for_y (y : ℝ) (h : (1 / 4) - (1 / 6) = 2 / y) : y = 24 :=
sorry

end NUMINAMATH_GPT_solve_for_y_l1135_113535


namespace NUMINAMATH_GPT_angle_triple_supplement_l1135_113589

theorem angle_triple_supplement {x : ℝ} (h1 : ∀ y : ℝ, y + (180 - y) = 180) (h2 : x = 3 * (180 - x)) :
  x = 135 :=
by
  sorry

end NUMINAMATH_GPT_angle_triple_supplement_l1135_113589


namespace NUMINAMATH_GPT_polygon_interior_angle_l1135_113526

theorem polygon_interior_angle (n : ℕ) (hn : 3 * (180 - 180 * (n - 2) / n) + 180 = 180 * (n - 2) / n + 180) : n = 9 :=
by {
  sorry
}

end NUMINAMATH_GPT_polygon_interior_angle_l1135_113526


namespace NUMINAMATH_GPT_find_the_number_l1135_113599

theorem find_the_number (x : ℕ) (h : 18396 * x = 183868020) : x = 9990 :=
by
  sorry

end NUMINAMATH_GPT_find_the_number_l1135_113599


namespace NUMINAMATH_GPT_initial_carrots_l1135_113530

theorem initial_carrots (n : ℕ) 
    (h1: 3640 = 180 * (n - 4) + 760) 
    (h2: 180 * (n - 4) < 3640) 
    (h3: 4 * 190 = 760) : 
    n = 20 :=
by
  sorry

end NUMINAMATH_GPT_initial_carrots_l1135_113530


namespace NUMINAMATH_GPT_divide_into_two_groups_l1135_113510

theorem divide_into_two_groups (n : ℕ) (A : Fin n → Type) 
  (acquaintances : (Fin n) → (Finset (Fin n)))
  (c : (Fin n) → ℕ) (d : (Fin n) → ℕ) :
  (∀ i : Fin n, c i = (acquaintances i).card) →
  ∃ G1 G2 : Finset (Fin n), G1 ∩ G2 = ∅ ∧ G1 ∪ G2 = Finset.univ ∧
  (∀ i : Fin n, d i = (acquaintances i ∩ (if i ∈ G1 then G2 else G1)).card ∧ d i ≥ (c i) / 2) :=
by 
  sorry

end NUMINAMATH_GPT_divide_into_two_groups_l1135_113510


namespace NUMINAMATH_GPT_dogwood_trees_proof_l1135_113540

def dogwood_trees_left (a b c : Float) : Float :=
  a + b - c

theorem dogwood_trees_proof : dogwood_trees_left 5.0 4.0 7.0 = 2.0 :=
by
  -- The proof itself is left out intentionally as per the instructions
  sorry

end NUMINAMATH_GPT_dogwood_trees_proof_l1135_113540


namespace NUMINAMATH_GPT_squared_expression_is_matching_string_l1135_113532

theorem squared_expression_is_matching_string (n : ℕ) (h : n > 0) :
  let a := (10^n - 1) / 9
  let term1 := 4 * a * (9 * a + 2)
  let term2 := 10 * a + 1
  let term3 := 6 * a
  let exp := term1 + term2 - term3
  Nat.sqrt exp = 6 * a + 1 := by
  sorry

end NUMINAMATH_GPT_squared_expression_is_matching_string_l1135_113532


namespace NUMINAMATH_GPT_four_consecutive_integers_divisible_by_12_l1135_113565

theorem four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
sorry

end NUMINAMATH_GPT_four_consecutive_integers_divisible_by_12_l1135_113565


namespace NUMINAMATH_GPT_area_of_region_l1135_113543

noncomputable def area : ℝ :=
  ∫ x in Set.Icc (-2 : ℝ) 0, (2 - (x + 1)^2 / 4) +
  ∫ x in Set.Icc (0 : ℝ) 2, (2 - x - (x + 1)^2 / 4)

theorem area_of_region : area = 5 / 3 := 
sorry

end NUMINAMATH_GPT_area_of_region_l1135_113543


namespace NUMINAMATH_GPT_range_of_m_l1135_113534

noncomputable def G (x : ℝ) (m : ℝ) : ℝ := (8 * x ^ 2 + 24 * x + 5 * m) / 8

theorem range_of_m (x : ℝ) (m : ℝ) : 
  (∃ c : ℝ, G x m = (x + c) ^ 2 ∧ c ^ 2 = 3) → 4 ≤ m ∧ m ≤ 5 := 
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1135_113534


namespace NUMINAMATH_GPT_boxes_count_l1135_113512

theorem boxes_count (notebooks_per_box : ℕ) (total_notebooks : ℕ) (h1 : notebooks_per_box = 9) (h2 : total_notebooks = 27) : (total_notebooks / notebooks_per_box) = 3 :=
by
  sorry

end NUMINAMATH_GPT_boxes_count_l1135_113512


namespace NUMINAMATH_GPT_meals_per_day_l1135_113560

-- Definitions based on given conditions
def number_of_people : Nat := 6
def total_plates_used : Nat := 144
def number_of_days : Nat := 4
def plates_per_meal : Nat := 2

-- Theorem to prove
theorem meals_per_day : (total_plates_used / number_of_days) / plates_per_meal / number_of_people = 3 :=
by
  sorry

end NUMINAMATH_GPT_meals_per_day_l1135_113560


namespace NUMINAMATH_GPT_inscribed_circle_area_ratio_l1135_113521

theorem inscribed_circle_area_ratio
  (R : ℝ) -- Radius of the original circle
  (r : ℝ) -- Radius of the inscribed circle
  (h : R = 3 * r) -- Relationship between the radii based on geometry problem
  :
  (π * R^2) / (π * r^2) = 9 :=
by sorry

end NUMINAMATH_GPT_inscribed_circle_area_ratio_l1135_113521


namespace NUMINAMATH_GPT_quadratic_root_inequality_l1135_113567

theorem quadratic_root_inequality (a : ℝ) :
  2015 < a ∧ a < 2017 ↔ 
  ∃ x₁ x₂ : ℝ, (2 * x₁^2 - 2016 * (x₁ - 2016 + a) - 1 = a^2) ∧ 
               (2 * x₂^2 - 2016 * (x₂ - 2016 + a) - 1 = a^2) ∧
               x₁ < a ∧ a < x₂ :=
sorry

end NUMINAMATH_GPT_quadratic_root_inequality_l1135_113567


namespace NUMINAMATH_GPT_new_volume_is_80_gallons_l1135_113587

-- Define the original volume
def V_original : ℝ := 5

-- Define the factors by which length, width, and height are increased
def length_factor : ℝ := 2
def width_factor : ℝ := 2
def height_factor : ℝ := 4

-- Define the new volume
def V_new : ℝ := V_original * (length_factor * width_factor * height_factor)

-- Theorem to prove the new volume is 80 gallons
theorem new_volume_is_80_gallons : V_new = 80 := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_new_volume_is_80_gallons_l1135_113587


namespace NUMINAMATH_GPT_waiting_time_boarding_l1135_113597

noncomputable def time_taken_uber_to_house : ℕ := 10
noncomputable def time_taken_uber_to_airport : ℕ := 5 * time_taken_uber_to_house
noncomputable def time_taken_bag_check : ℕ := 15
noncomputable def time_taken_security : ℕ := 3 * time_taken_bag_check
noncomputable def total_process_time : ℕ := 180
noncomputable def remaining_time : ℕ := total_process_time - (time_taken_uber_to_house + time_taken_uber_to_airport + time_taken_bag_check + time_taken_security)
noncomputable def time_before_takeoff (B : ℕ) := 2 * B

theorem waiting_time_boarding : ∃ B : ℕ, B + time_before_takeoff B = remaining_time ∧ B = 20 := 
by 
  sorry

end NUMINAMATH_GPT_waiting_time_boarding_l1135_113597


namespace NUMINAMATH_GPT_minimum_daily_expense_l1135_113555

-- Defining the context
variables (x y : ℕ)
def total_capacity (x y : ℕ) : ℕ := 24 * x + 30 * y
def cost (x y : ℕ) : ℕ := 320 * x + 504 * y

theorem minimum_daily_expense :
  (total_capacity x y ≥ 180) →
  (x ≤ 8) →
  (y ≤ 4) →
  cost x y = 2560 := sorry

end NUMINAMATH_GPT_minimum_daily_expense_l1135_113555


namespace NUMINAMATH_GPT_find_x_if_perpendicular_l1135_113572

-- Define the vectors a and b
def vector_a (x : ℝ) : ℝ × ℝ := (x, 3)
def vector_b (x : ℝ) : ℝ × ℝ := (2, x - 5)

-- Define the condition that vectors a and b are perpendicular
def perpendicular (x : ℝ) : Prop :=
  let a := vector_a x
  let b := vector_b x
  a.1 * b.1 + a.2 * b.2 = 0

-- Prove that x = 3 if a and b are perpendicular
theorem find_x_if_perpendicular :
  ∃ x : ℝ, perpendicular x ∧ x = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_x_if_perpendicular_l1135_113572


namespace NUMINAMATH_GPT_g_seven_l1135_113551

def g (x : ℚ) : ℚ := (2 * x + 3) / (4 * x - 5)

theorem g_seven : g 7 = 17 / 23 := by
  sorry

end NUMINAMATH_GPT_g_seven_l1135_113551


namespace NUMINAMATH_GPT_seating_arrangements_l1135_113584

theorem seating_arrangements (n : ℕ) (max_capacity : ℕ) 
  (h_n : n = 6) (h_max : max_capacity = 4) :
  ∃ k : ℕ, k = 50 :=
by
  sorry

end NUMINAMATH_GPT_seating_arrangements_l1135_113584


namespace NUMINAMATH_GPT_find_number_l1135_113516

theorem find_number (k r n : ℤ) (hk : k = 38) (hr : r = 7) (h : n = 23 * k + r) : n = 881 := 
  by
  sorry

end NUMINAMATH_GPT_find_number_l1135_113516


namespace NUMINAMATH_GPT_quadratic_rewrite_l1135_113508

theorem quadratic_rewrite  (a b c x : ℤ) (h : 25 * x^2 + 30 * x - 35 = 0) (hp : 25 * x^2 + 30 * x + 9 = (5 * x + 3) ^ 2)
(hc : c = 44) : a = 5 → b = 3 → a + b + c = 52 := 
by
  intro ha hb
  sorry

end NUMINAMATH_GPT_quadratic_rewrite_l1135_113508


namespace NUMINAMATH_GPT_smallest_number_divisible_by_conditions_l1135_113544

theorem smallest_number_divisible_by_conditions (N : ℕ) (X : ℕ) (H1 : (N - 12) % 8 = 0) (H2 : (N - 12) % 12 = 0)
(H3 : (N - 12) % X = 0) (H4 : (N - 12) % 24 = 0) (H5 : (N - 12) / 24 = 276) : N = 6636 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_divisible_by_conditions_l1135_113544


namespace NUMINAMATH_GPT_area_of_square_l1135_113557

theorem area_of_square (ABCD MN : ℝ) (h1 : 4 * (ABCD / 4) = ABCD) (h2 : MN = 3) : ABCD = 64 :=
by
  sorry

end NUMINAMATH_GPT_area_of_square_l1135_113557


namespace NUMINAMATH_GPT_total_cost_in_dollars_l1135_113502

theorem total_cost_in_dollars :
  (500 * 3 + 300 * 2) / 100 = 21 := 
by
  sorry

end NUMINAMATH_GPT_total_cost_in_dollars_l1135_113502


namespace NUMINAMATH_GPT_find_x_l1135_113515

-- Define the angles as real numbers representing degrees.
variable (angle_SWR angle_WRU angle_x : ℝ)

-- Conditions given in the problem
def conditions (angle_SWR angle_WRU angle_x : ℝ) : Prop :=
  angle_SWR = 50 ∧ angle_WRU = 30 ∧ angle_SWR = angle_WRU + angle_x

-- Main theorem to prove that x = 20 given the conditions
theorem find_x (angle_SWR angle_WRU angle_x : ℝ) :
  conditions angle_SWR angle_WRU angle_x → angle_x = 20 := by
  sorry

end NUMINAMATH_GPT_find_x_l1135_113515


namespace NUMINAMATH_GPT_area_of_trapezoid_l1135_113520

theorem area_of_trapezoid
  (r : ℝ)
  (AD BC : ℝ)
  (center_on_base : Bool)
  (height : ℝ)
  (area : ℝ)
  (inscribed_circle : r = 6)
  (base_AD : AD = 8)
  (base_BC : BC = 4)
  (K_height : height = 4 * Real.sqrt 2)
  (calc_area : area = (1 / 2) * (AD + BC) * height)
  : area = 32 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_area_of_trapezoid_l1135_113520


namespace NUMINAMATH_GPT_net_rate_of_pay_l1135_113537

theorem net_rate_of_pay :
  ∀ (duration_travel : ℝ) (speed : ℝ) (fuel_efficiency : ℝ) (earnings_rate : ℝ) (gas_cost : ℝ),
  duration_travel = 3 → speed = 50 → fuel_efficiency = 30 → earnings_rate = 0.75 → gas_cost = 2.50 →
  (earnings_rate * speed * duration_travel - (speed * duration_travel / fuel_efficiency) * gas_cost) / duration_travel = 33.33 :=
by
  intros duration_travel speed fuel_efficiency earnings_rate gas_cost
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_net_rate_of_pay_l1135_113537


namespace NUMINAMATH_GPT_exists_disjoint_A_B_l1135_113564

def S (C : Finset ℕ) := C.sum id

theorem exists_disjoint_A_B : 
  ∃ (A B : Finset ℕ), 
  A ≠ ∅ ∧ B ≠ ∅ ∧ 
  A ∩ B = ∅ ∧ 
  A ∪ B = (Finset.range (2021 + 1)).erase 0 ∧ 
  ∃ k : ℕ, S A * S B = k^2 :=
by 
  sorry

end NUMINAMATH_GPT_exists_disjoint_A_B_l1135_113564


namespace NUMINAMATH_GPT_ArletteAge_l1135_113542

/-- Define the ages of Omi, Kimiko, and Arlette -/
def OmiAge (K : ℕ) : ℕ := 2 * K
def KimikoAge : ℕ := 28   /- K = 28 -/
def averageAge (O K A : ℕ) : Prop := (O + K + A) / 3 = 35

/-- Prove Arlette's age given the conditions -/
theorem ArletteAge (A : ℕ) (h1 : A + OmiAge KimikoAge + KimikoAge = 3 * 35) : A = 21 := by
  /- Hypothesis h1 unpacks the third condition into equality involving O, K, and A -/
  sorry

end NUMINAMATH_GPT_ArletteAge_l1135_113542


namespace NUMINAMATH_GPT_smallest_n_Sn_gt_2023_l1135_113525

open Nat

theorem smallest_n_Sn_gt_2023 (a : ℕ → ℕ) (S : ℕ → ℕ) :
  (a 1 = 4) →
  (∀ n : ℕ, n > 0 → a n + a (n + 1) = 4 * n + 2) →
  (∀ m : ℕ, S m = if m % 2 = 0 then m ^ 2 + m else m ^ 2 + m + 2) →
  ∃ n : ℕ, S n > 2023 ∧ ∀ k : ℕ, k < n → S k ≤ 2023 :=
sorry

end NUMINAMATH_GPT_smallest_n_Sn_gt_2023_l1135_113525


namespace NUMINAMATH_GPT_no_sum_of_three_squares_l1135_113514

theorem no_sum_of_three_squares (a k : ℕ) : 
  ¬ ∃ x y z : ℤ, 4^a * (8*k + 7) = x^2 + y^2 + z^2 :=
by
  sorry

end NUMINAMATH_GPT_no_sum_of_three_squares_l1135_113514


namespace NUMINAMATH_GPT_find_polynomial_R_l1135_113503

-- Define the polynomials S(x), Q(x), and the remainder R(x)

noncomputable def S (x : ℝ) := 7 * x ^ 31 + 3 * x ^ 13 + 10 * x ^ 11 - 5 * x ^ 9 - 10 * x ^ 7 + 5 * x ^ 5 - 2
noncomputable def Q (x : ℝ) := x ^ 4 + x ^ 3 + x ^ 2 + x + 1
noncomputable def R (x : ℝ) := 13 * x ^ 3 + 5 * x ^ 2 + 12 * x + 3

-- Statement of the proof
theorem find_polynomial_R :
  ∃ (P : ℝ → ℝ), ∀ x : ℝ, S x = P x * Q x + R x := sorry

end NUMINAMATH_GPT_find_polynomial_R_l1135_113503


namespace NUMINAMATH_GPT_divisor_between_40_and_50_l1135_113547

theorem divisor_between_40_and_50 (n : ℕ) (h1 : 40 ≤ n) (h2 : n ≤ 50) (h3 : n ∣ (2^36 - 1)) : n = 49 :=
sorry

end NUMINAMATH_GPT_divisor_between_40_and_50_l1135_113547


namespace NUMINAMATH_GPT_fraction_b_plus_c_over_a_l1135_113559

variable (a b c d : ℝ)

theorem fraction_b_plus_c_over_a :
  (a ≠ 0) →
  (a * 4^3 + b * 4^2 + c * 4 + d = 0) →
  (a * (-3)^3 + b * (-3)^2 + c * (-3) + d = 0) →
  (b + c) / a = -13 :=
by
  intros h₁ h₂ h₃ 
  sorry

end NUMINAMATH_GPT_fraction_b_plus_c_over_a_l1135_113559


namespace NUMINAMATH_GPT_smallest_portion_is_2_l1135_113579

theorem smallest_portion_is_2 (a d : ℝ) (h1 : 5 * a = 120) (h2 : 3 * a + 3 * d = 7 * (2 * a - 3 * d)) : a - 2 * d = 2 :=
by sorry

end NUMINAMATH_GPT_smallest_portion_is_2_l1135_113579


namespace NUMINAMATH_GPT_four_digit_numbers_no_5s_8s_l1135_113570

def count_valid_four_digit_numbers : Nat :=
  let thousand_place := 7  -- choices: 1, 2, 3, 4, 6, 7, 9
  let other_places := 8  -- choices: 0, 1, 2, 3, 4, 6, 7, 9
  thousand_place * other_places * other_places * other_places

theorem four_digit_numbers_no_5s_8s : count_valid_four_digit_numbers = 3584 :=
by
  rfl

end NUMINAMATH_GPT_four_digit_numbers_no_5s_8s_l1135_113570


namespace NUMINAMATH_GPT_nancy_kept_chips_correct_l1135_113536

/-- Define the initial conditions -/
def total_chips : ℕ := 22
def chips_to_brother : ℕ := 7
def chips_to_sister : ℕ := 5

/-- Define the number of chips Nancy kept -/
def chips_kept : ℕ := total_chips - (chips_to_brother + chips_to_sister)

theorem nancy_kept_chips_correct : chips_kept = 10 := by
  /- This is a placeholder. The proof would go here. -/
  sorry

end NUMINAMATH_GPT_nancy_kept_chips_correct_l1135_113536


namespace NUMINAMATH_GPT_average_selections_correct_l1135_113596

noncomputable def cars := 18
noncomputable def selections_per_client := 3
noncomputable def clients := 18
noncomputable def total_selections := clients * selections_per_client
noncomputable def average_selections_per_car := total_selections / cars

theorem average_selections_correct :
  average_selections_per_car = 3 :=
by
  sorry

end NUMINAMATH_GPT_average_selections_correct_l1135_113596


namespace NUMINAMATH_GPT_Jeremy_payment_total_l1135_113519

theorem Jeremy_payment_total :
  let room_rate := (13 : ℚ) / 3
  let rooms_cleaned := (8 : ℚ) / 5
  let window_rate := (5 : ℚ) / 2
  let windows_cleaned := (11 : ℚ) / 4
  let payment_rooms := room_rate * rooms_cleaned
  let payment_windows := window_rate * windows_cleaned
  let total_payment := payment_rooms + payment_windows
  total_payment = (553 : ℚ) / 40 :=
by {
  -- Definitions
  let room_rate := (13 : ℚ) / 3
  let rooms_cleaned := (8 : ℚ) / 5
  let window_rate := (5 : ℚ) / 2
  let windows_cleaned := (11 : ℚ) / 4
  let payment_rooms := room_rate * rooms_cleaned
  let payment_windows := window_rate * windows_cleaned
  let total_payment := payment_rooms + payment_windows
  
  -- Main goal
  sorry
}

end NUMINAMATH_GPT_Jeremy_payment_total_l1135_113519


namespace NUMINAMATH_GPT_solution_set_range_ineq_l1135_113554

theorem solution_set_range_ineq (m : ℝ) :
  ∀ x : ℝ, (m^2 - 2*m - 3)*x^2 - (m - 3)*x - 1 < 0 ↔ (-5: ℝ)⁻¹ < m ∧ m ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_range_ineq_l1135_113554


namespace NUMINAMATH_GPT_speed_of_man_in_still_water_l1135_113598

theorem speed_of_man_in_still_water (v_m v_s : ℝ) (h1 : v_m + v_s = 6.2) (h2 : v_m - v_s = 6) : v_m = 6.1 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_man_in_still_water_l1135_113598


namespace NUMINAMATH_GPT_factors_of_12_factors_of_18_l1135_113511

def is_factor (n k : ℕ) : Prop := k > 0 ∧ n % k = 0

theorem factors_of_12 : 
  {k : ℕ | is_factor 12 k} = {1, 12, 2, 6, 3, 4} :=
by
  sorry

theorem factors_of_18 : 
  {k : ℕ | is_factor 18 k} = {1, 18, 2, 9, 3, 6} :=
by
  sorry

end NUMINAMATH_GPT_factors_of_12_factors_of_18_l1135_113511


namespace NUMINAMATH_GPT_golden_ratio_problem_l1135_113538

theorem golden_ratio_problem
  (m n : ℝ) (sin cos : ℝ → ℝ)
  (h1 : m = 2 * sin (Real.pi / 10))
  (h2 : m ^ 2 + n = 4)
  (sin63 : sin (7 * Real.pi / 18) ≠ 0) :
  (m + Real.sqrt n) / (sin (7 * Real.pi / 18)) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_golden_ratio_problem_l1135_113538


namespace NUMINAMATH_GPT_james_vacuuming_hours_l1135_113583

/-- James spends some hours vacuuming and 3 times as long on the rest of his chores. 
    He spends 12 hours on his chores in total. -/
theorem james_vacuuming_hours (V : ℝ) (h : V + 3 * V = 12) : V = 3 := 
sorry

end NUMINAMATH_GPT_james_vacuuming_hours_l1135_113583


namespace NUMINAMATH_GPT_derivative_sqrt_l1135_113590

/-- The derivative of the function y = sqrt x is 1 / (2 * sqrt x) -/
theorem derivative_sqrt (x : ℝ) (h : 0 < x) : (deriv (fun x => Real.sqrt x) x) = 1 / (2 * Real.sqrt x) :=
sorry

end NUMINAMATH_GPT_derivative_sqrt_l1135_113590


namespace NUMINAMATH_GPT_angle_between_v1_v2_l1135_113531

-- Define vectors
def v1 : ℝ × ℝ := (3, -4)
def v2 : ℝ × ℝ := (4, 6)

-- Define the dot product function
def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

-- Define the magnitude function
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Define the cosine of the angle between two vectors
noncomputable def cos_theta (a b : ℝ × ℝ) : ℝ := (dot_product a b) / (magnitude a * magnitude b)

-- Define the angle in degrees between two vectors
noncomputable def angle_between_vectors (a b : ℝ × ℝ) : ℝ := Real.arccos (cos_theta a b) * (180 / Real.pi)

-- The statement to prove
theorem angle_between_v1_v2 : angle_between_vectors v1 v2 = Real.arccos (-6 * Real.sqrt 13 / 65) * (180 / Real.pi) :=
sorry

end NUMINAMATH_GPT_angle_between_v1_v2_l1135_113531


namespace NUMINAMATH_GPT_solve_fractional_eq_l1135_113593

theorem solve_fractional_eq (x : ℝ) (hx1 : x ≠ 0) (hx2 : x ≠ -3) : (1 / x = 6 / (x + 3)) → (x = 0.6) :=
by
  sorry

end NUMINAMATH_GPT_solve_fractional_eq_l1135_113593


namespace NUMINAMATH_GPT_n_power_of_two_if_2_pow_n_plus_one_odd_prime_l1135_113549

-- Definition: a positive integer n is a power of 2
def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

-- Theorem: if 2^n +1 is an odd prime, then n must be a power of 2
theorem n_power_of_two_if_2_pow_n_plus_one_odd_prime (n : ℕ) (hp : Prime (2^n + 1)) (hn : Odd (2^n + 1)) : is_power_of_two n :=
by
  sorry

end NUMINAMATH_GPT_n_power_of_two_if_2_pow_n_plus_one_odd_prime_l1135_113549


namespace NUMINAMATH_GPT_range_of_a_l1135_113595

theorem range_of_a (a : ℝ) : 
  (∃! x : ℤ, 4 - 2 * x ≥ 0 ∧ (1 / 2 : ℝ) * x - a > 0) ↔ -1 ≤ a ∧ a < -0.5 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1135_113595


namespace NUMINAMATH_GPT_find_m_value_l1135_113550

theorem find_m_value
  (y_squared_4x : ∀ x y : ℝ, y^2 = 4 * x)
  (Focus_F : ℝ × ℝ)
  (M N : ℝ × ℝ)
  (E : ℝ)
  (P Q : ℝ × ℝ)
  (k1 k2 : ℝ)
  (MN_slope : k1 = (N.snd - M.snd) / (N.fst - M.fst))
  (PQ_slope : k2 = (Q.snd - P.snd) / (Q.fst - P.fst))
  (slope_condition : k1 = 3 * k2) :
  E = 3 := 
sorry

end NUMINAMATH_GPT_find_m_value_l1135_113550


namespace NUMINAMATH_GPT_range_of_m_l1135_113558

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sqrt (1 + x) + Real.sqrt (1 - x)) * (2 * Real.sqrt (1 - x^2) - 1)

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ f x = m) ↔ -Real.sqrt 2 ≤ m ∧ m ≤ Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_range_of_m_l1135_113558


namespace NUMINAMATH_GPT_find_m_l1135_113523

-- Definition of the function as a direct proportion function with respect to x
def isDirectProportion (m : ℝ) : Prop :=
  m^2 - 8 = 1

-- Definition of the graph passing through the second and fourth quadrants
def passesThroughQuadrants (m : ℝ) : Prop :=
  m - 2 < 0

-- The theorem combining the conditions and proving the correct value of m
theorem find_m (m : ℝ) 
  (h1 : isDirectProportion m)
  (h2 : passesThroughQuadrants m) : 
  m = -3 :=
  sorry

end NUMINAMATH_GPT_find_m_l1135_113523


namespace NUMINAMATH_GPT_nancy_rose_bracelets_l1135_113591

-- Definitions based on conditions
def metal_beads_nancy : ℕ := 40
def pearl_beads_nancy : ℕ := metal_beads_nancy + 20
def total_beads_nancy : ℕ := metal_beads_nancy + pearl_beads_nancy

def crystal_beads_rose : ℕ := 20
def stone_beads_rose : ℕ := 2 * crystal_beads_rose
def total_beads_rose : ℕ := crystal_beads_rose + stone_beads_rose

def number_of_bracelets (total_beads : ℕ) (beads_per_bracelet : ℕ) : ℕ :=
  total_beads / beads_per_bracelet

-- Theorem to be proved
theorem nancy_rose_bracelets : number_of_bracelets (total_beads_nancy + total_beads_rose) 8 = 20 := 
by
  -- Definitions will be expanded here
  sorry

end NUMINAMATH_GPT_nancy_rose_bracelets_l1135_113591


namespace NUMINAMATH_GPT_min_m_l1135_113552

theorem min_m (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 1) : 
    27 * (a^3 + b^3 + c^3) ≥ 6 * (a^2 + b^2 + c^2) + 1 :=
by
  sorry

end NUMINAMATH_GPT_min_m_l1135_113552


namespace NUMINAMATH_GPT_monkey_climbing_distance_l1135_113574

theorem monkey_climbing_distance
  (x : ℝ)
  (h1 : ∀ t : ℕ, t % 2 = 0 → t ≠ 0 → x - 3 > 0) -- condition (2,4)
  (h2 : ∀ t : ℕ, t % 2 = 1 → x > 0) -- condition (5)
  (h3 : 18 * (x - 3) + x = 60) -- condition (6)
  : x = 6 :=
sorry

end NUMINAMATH_GPT_monkey_climbing_distance_l1135_113574


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l1135_113575

theorem simplify_and_evaluate_expression (x y : ℝ) (h1 : x = 1 / 2) (h2 : y = 2023) :
  (x + y)^2 + (x + y) * (x - y) - 2 * x^2 = 2023 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l1135_113575


namespace NUMINAMATH_GPT_smaller_angle_between_east_and_northwest_l1135_113563

theorem smaller_angle_between_east_and_northwest
  (rays : ℕ)
  (each_angle : ℕ)
  (direction : ℕ → ℝ)
  (h1 : rays = 10)
  (h2 : each_angle = 36)
  (h3 : direction 0 = 0) -- ray at due North
  (h4 : direction 3 = 90) -- ray at due East
  (h5 : direction 5 = 135) -- ray at due Northwest
  : direction 5 - direction 3 = each_angle :=
by
  -- to be proved
  sorry

end NUMINAMATH_GPT_smaller_angle_between_east_and_northwest_l1135_113563


namespace NUMINAMATH_GPT_combined_tax_rate_correct_l1135_113509

noncomputable def combined_tax_rate (income_john income_ingrid tax_rate_john tax_rate_ingrid : ℝ) : ℝ :=
  let tax_john := tax_rate_john * income_john
  let tax_ingrid := tax_rate_ingrid * income_ingrid
  let total_tax := tax_john + tax_ingrid
  let combined_income := income_john + income_ingrid
  total_tax / combined_income * 100

theorem combined_tax_rate_correct :
  combined_tax_rate 56000 74000 0.30 0.40 = 35.69 := by
  sorry

end NUMINAMATH_GPT_combined_tax_rate_correct_l1135_113509


namespace NUMINAMATH_GPT_local_minimum_of_reflected_function_l1135_113562

noncomputable def f : ℝ → ℝ := sorry

theorem local_minimum_of_reflected_function (f : ℝ → ℝ) (x_0 : ℝ) (h1 : x_0 ≠ 0) (h2 : ∃ ε > 0, ∀ x, abs (x - x_0) < ε → f x ≤ f x_0) :
  ∃ δ > 0, ∀ x, abs (x - (-x_0)) < δ → -f (-x) ≥ -f (-x_0) :=
sorry

end NUMINAMATH_GPT_local_minimum_of_reflected_function_l1135_113562


namespace NUMINAMATH_GPT_relationship_of_new_stationary_points_l1135_113576

noncomputable def g (x : ℝ) : ℝ := Real.sin x
noncomputable def h (x : ℝ) : ℝ := Real.log x
noncomputable def phi (x : ℝ) : ℝ := x^3

noncomputable def g' (x : ℝ) : ℝ := Real.cos x
noncomputable def h' (x : ℝ) : ℝ := 1 / x
noncomputable def phi' (x : ℝ) : ℝ := 3 * x^2

-- Definitions of the new stationary points
noncomputable def new_stationary_point_g (x : ℝ) : Prop := g x = g' x
noncomputable def new_stationary_point_h (x : ℝ) : Prop := h x = h' x
noncomputable def new_stationary_point_phi (x : ℝ) : Prop := phi x = phi' x

theorem relationship_of_new_stationary_points :
  ∃ (a b c : ℝ), (0 < a ∧ a < π) ∧ (1 < b ∧ b < Real.exp 1) ∧ (c ≠ 0) ∧
  new_stationary_point_g a ∧ new_stationary_point_h b ∧ new_stationary_point_phi c ∧
  c > b ∧ b > a :=
by
  sorry

end NUMINAMATH_GPT_relationship_of_new_stationary_points_l1135_113576


namespace NUMINAMATH_GPT_mr_bird_exact_speed_l1135_113504

-- Define the properties and calculating the exact speed
theorem mr_bird_exact_speed (d t : ℝ) (h1 : d = 50 * (t + 1 / 12)) (h2 : d = 70 * (t - 1 / 12)) :
  d / t = 58 :=
by 
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_mr_bird_exact_speed_l1135_113504


namespace NUMINAMATH_GPT_dave_ice_cubes_total_l1135_113501

theorem dave_ice_cubes_total : 
  let trayA_initial := 2
  let trayA_final := trayA_initial + 7
  let trayB := (1 / 3) * trayA_final
  let trayC := 2 * trayA_final
  trayA_final + trayB + trayC = 30 := by
  sorry

end NUMINAMATH_GPT_dave_ice_cubes_total_l1135_113501


namespace NUMINAMATH_GPT_intersection_of_lines_l1135_113566

theorem intersection_of_lines :
  ∃ (x y : ℚ), (6 * x - 5 * y = 15) ∧ (8 * x + 3 * y = 1) ∧ x = 25 / 29 ∧ y = -57 / 29 :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_lines_l1135_113566


namespace NUMINAMATH_GPT_ceil_square_range_count_l1135_113580

theorem ceil_square_range_count (x : ℝ) (h : ⌈x⌉ = 12) : 
  ∃ n : ℕ, n = 23 ∧ (∀ y : ℝ, 11 < y ∧ y ≤ 12 → ⌈y^2⌉ = n) := 
sorry

end NUMINAMATH_GPT_ceil_square_range_count_l1135_113580


namespace NUMINAMATH_GPT_counterexample_to_proposition_l1135_113571

theorem counterexample_to_proposition (x y : ℤ) (h1 : x = -1) (h2 : y = -2) : x > y ∧ ¬ (x^2 > y^2) := by
  sorry

end NUMINAMATH_GPT_counterexample_to_proposition_l1135_113571


namespace NUMINAMATH_GPT_minimize_f_l1135_113561

noncomputable def f (x : ℝ) : ℝ := 3 * x ^ 2 - 18 * x + 7

theorem minimize_f : ∀ x : ℝ, f x ≥ f 3 :=
by
  sorry

end NUMINAMATH_GPT_minimize_f_l1135_113561


namespace NUMINAMATH_GPT_probability_of_same_color_l1135_113533

noncomputable def prob_same_color (P_A P_B : ℚ) : ℚ :=
  P_A + P_B

theorem probability_of_same_color :
  let P_A := (1 : ℚ) / 7
  let P_B := (12 : ℚ) / 35
  prob_same_color P_A P_B = 17 / 35 := 
by 
  -- Definition of P_A and P_B
  let P_A := (1 : ℚ) / 7
  let P_B := (12 : ℚ) / 35
  -- Use the definition of prob_same_color
  let result := prob_same_color P_A P_B
  -- Now we are supposed to prove that result = 17 / 35
  have : result = (5 : ℚ) / 35 + (12 : ℚ) / 35 := by
    -- Simplifying the fractions individually can be done at this intermediate step
    sorry
  sorry

end NUMINAMATH_GPT_probability_of_same_color_l1135_113533


namespace NUMINAMATH_GPT_sum_of_ages_of_cousins_l1135_113545

noncomputable def is_valid_age_group (a b c d : ℕ) : Prop :=
  (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (b ≠ c) ∧ (b ≠ d) ∧ (c ≠ d) ∧
  (1 ≤ a) ∧ (a ≤ 9) ∧ (1 ≤ b) ∧ (b ≤ 9) ∧ (1 ≤ c) ∧ (c ≤ 9) ∧ (1 ≤ d) ∧ (d ≤ 9)

theorem sum_of_ages_of_cousins :
  ∃ (a b c d : ℕ), is_valid_age_group a b c d ∧ (a * b = 40) ∧ (c * d = 36) ∧ (a + b + c + d = 26) := 
sorry

end NUMINAMATH_GPT_sum_of_ages_of_cousins_l1135_113545


namespace NUMINAMATH_GPT_valid_outfit_combinations_l1135_113553

theorem valid_outfit_combinations :
  let shirts := 8
  let pants := 5
  let hats := 6
  let shared_colors := 5
  let extra_colors := 2
  let total_combinations := shirts * pants * hats
  let invalid_shared_combinations := shared_colors * pants
  let invalid_extra_combinations := extra_colors * pants
  let invalid_combinations := invalid_shared_combinations + invalid_extra_combinations
  total_combinations - invalid_combinations = 205 :=
by
  let shirts := 8
  let pants := 5
  let hats := 6
  let shared_colors := 5
  let extra_colors := 2
  let total_combinations := shirts * pants * hats
  let invalid_shared_combinations := shared_colors * pants
  let invalid_extra_combinations := extra_colors * pants
  let invalid_combinations := invalid_shared_combinations + invalid_extra_combinations
  have h : total_combinations - invalid_combinations = 205 := sorry
  exact h

end NUMINAMATH_GPT_valid_outfit_combinations_l1135_113553


namespace NUMINAMATH_GPT_maximum_value_of_f_l1135_113546

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 0 then x^3 - 3 * x else -2 * x + 1

theorem maximum_value_of_f : ∃ (m : ℝ), (∀ x : ℝ, f x ≤ m) ∧ (m = 2) := by
  sorry

end NUMINAMATH_GPT_maximum_value_of_f_l1135_113546


namespace NUMINAMATH_GPT_smallest_solution_of_quadratic_eq_l1135_113507

theorem smallest_solution_of_quadratic_eq : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁ < x₂) ∧ (x₁^2 + 10 * x₁ - 40 = 0) ∧ (x₂^2 + 10 * x₂ - 40 = 0) ∧ x₁ = -8 :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_solution_of_quadratic_eq_l1135_113507


namespace NUMINAMATH_GPT_sequence_general_formula_l1135_113548

theorem sequence_general_formula (a : ℕ → ℕ) (h1 : a 1 = 1) (rec : ∀ n : ℕ, n > 0 → a n = n * (a (n + 1) - a n)) : 
  ∀ n, a n = n := 
by 
  sorry

end NUMINAMATH_GPT_sequence_general_formula_l1135_113548


namespace NUMINAMATH_GPT_triangle_cos_C_correct_l1135_113524

noncomputable def triangle_cos_C (A B C : ℝ) (hABC : A + B + C = π)
  (hSinA : Real.sin A = 3 / 5) (hCosB : Real.cos B = 5 / 13) : ℝ :=
  Real.cos C -- This will be defined correctly in the proof phase.

theorem triangle_cos_C_correct (A B C : ℝ) (hABC : A + B + C = π)
  (hSinA : Real.sin A = 3 / 5) (hCosB : Real.cos B = 5 / 13) : 
  triangle_cos_C A B C hABC hSinA hCosB = 16 / 65 :=
sorry

end NUMINAMATH_GPT_triangle_cos_C_correct_l1135_113524


namespace NUMINAMATH_GPT_cone_base_radius_half_l1135_113594

theorem cone_base_radius_half :
  let R : ℝ := sorry
  let semicircle_radius : ℝ := 1
  let unfolded_circumference : ℝ := π
  let base_circumference : ℝ := 2 * π * R
  base_circumference = unfolded_circumference -> R = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cone_base_radius_half_l1135_113594


namespace NUMINAMATH_GPT_quadratic_roots_l1135_113522

theorem quadratic_roots (p q r : ℝ) (h : p ≠ q) (k : ℝ) :
  (p * (q - r) * (-1)^2 + q * (r - p) * (-1) + r * (p - q) = 0) →
  ((p * (q - r)) * k^2 + (q * (r - p)) * k + r * (p - q) = 0) →
  k = - (r * (p - q)) / (p * (q - r)) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_l1135_113522


namespace NUMINAMATH_GPT_ramesh_share_correct_l1135_113517

-- Define basic conditions
def suresh_investment := 24000
def ramesh_investment := 40000
def total_profit := 19000

-- Define Ramesh's share calculation
def ramesh_share : ℤ :=
  let ratio_ramesh := ramesh_investment / (suresh_investment + ramesh_investment)
  ratio_ramesh * total_profit

-- Proof statement
theorem ramesh_share_correct : ramesh_share = 11875 := by
  sorry

end NUMINAMATH_GPT_ramesh_share_correct_l1135_113517


namespace NUMINAMATH_GPT_gasoline_added_l1135_113588

noncomputable def initial_amount (capacity: ℕ) : ℝ :=
  (3 / 4) * capacity

noncomputable def final_amount (capacity: ℕ) : ℝ :=
  (9 / 10) * capacity

theorem gasoline_added (capacity: ℕ) (initial_fraction final_fraction: ℝ) (initial_amount final_amount: ℝ) : 
  capacity = 54 ∧ initial_fraction = 3/4 ∧ final_fraction = 9/10 ∧ 
  initial_amount = initial_fraction * capacity ∧ 
  final_amount = final_fraction * capacity →
  final_amount - initial_amount = 8.1 :=
sorry

end NUMINAMATH_GPT_gasoline_added_l1135_113588


namespace NUMINAMATH_GPT_john_books_per_day_l1135_113578

theorem john_books_per_day (books_per_week := 2) (weeks := 6) (total_books := 48) :
  (total_books / (books_per_week * weeks) = 4) :=
by
  sorry

end NUMINAMATH_GPT_john_books_per_day_l1135_113578


namespace NUMINAMATH_GPT_inequality_proof_l1135_113527

theorem inequality_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + b + c = 3) :
  (a^2 + 9) / (2*a^2 + (b+c)^2) + (b^2 + 9) / (2*b^2 + (c+a)^2) + (c^2 + 9) / (2*c^2 + (a+b)^2) ≤ 5 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1135_113527


namespace NUMINAMATH_GPT_minimum_value_inequality_l1135_113586

theorem minimum_value_inequality {x y z : ℝ} (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (hxyz : x + y + z = 9) :
  (x^3 + y^3) / (x + y) + (x^3 + z^3) / (x + z) + (y^3 + z^3) / (y + z) ≥ 27 :=
sorry

end NUMINAMATH_GPT_minimum_value_inequality_l1135_113586


namespace NUMINAMATH_GPT_arithmetic_sequence_a3_l1135_113556

theorem arithmetic_sequence_a3 (a1 d : ℤ) (h : a1 + (a1 + d) + (a1 + 2 * d) + (a1 + 3 * d) + (a1 + 4 * d) = 20) : 
  a1 + 2 * d = 4 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a3_l1135_113556


namespace NUMINAMATH_GPT_valid_digit_for_multiple_of_5_l1135_113568

theorem valid_digit_for_multiple_of_5 (d : ℕ) (h : d < 10) : (45670 + d) % 5 = 0 ↔ d = 0 ∨ d = 5 :=
by
  sorry

end NUMINAMATH_GPT_valid_digit_for_multiple_of_5_l1135_113568


namespace NUMINAMATH_GPT_square_area_is_4802_l1135_113581

-- Condition: the length of the diagonal of the square is 98 meters.
def diagonal (d : ℝ) := d = 98

-- Goal: Prove that the area of the square field is 4802 square meters.
theorem square_area_is_4802 (d : ℝ) (h : diagonal d) : ∃ (A : ℝ), A = 4802 := 
by sorry

end NUMINAMATH_GPT_square_area_is_4802_l1135_113581


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_perpendicular_lines_l1135_113539

def are_perpendicular (a : ℝ) : Prop :=
  ∀ x y : ℝ, (x + y = 0) → (x - a * y = 0) → x = 0 ∧ y = 0

theorem necessary_and_sufficient_condition_perpendicular_lines :
  ∀ (a : ℝ), are_perpendicular a → a = 1 :=
sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_perpendicular_lines_l1135_113539


namespace NUMINAMATH_GPT_number_of_valid_pairs_l1135_113541

theorem number_of_valid_pairs :
  (∀ (m n : ℕ), 1 ≤ m ∧ m ≤ 2044 ∧ 5^n < 2^m ∧ 2^m < 2^(m + 1) ∧ 2^(m + 1) < 5^(n + 1)) ↔
  ((∃ (x y : ℕ), 2^2100 < 5^900 ∧ 5^900 < 2^2101)) → 
  (∃ (count : ℕ), count = 900) :=
by sorry

end NUMINAMATH_GPT_number_of_valid_pairs_l1135_113541


namespace NUMINAMATH_GPT_exists_arithmetic_progression_with_11_numbers_exists_arithmetic_progression_with_10000_numbers_not_exists_infinite_arithmetic_progression_l1135_113529

open Nat

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem exists_arithmetic_progression_with_11_numbers :
  ∃ a d : ℕ, ∀ i j : ℕ, i < 11 → j < 11 → i < j → a + i * d < a + j * d ∧ 
  sum_of_digits (a + i * d) < sum_of_digits (a + j * d) := by
  sorry

theorem exists_arithmetic_progression_with_10000_numbers :
  ∃ a d : ℕ, ∀ i j : ℕ, i < 10000 → j < 10000 → i < j → a + i * d < a + j * d ∧
  sum_of_digits (a + i * d) < sum_of_digits (a + j * d) := by
  sorry

theorem not_exists_infinite_arithmetic_progression :
  ¬ (∃ a d : ℕ, ∀ i j : ℕ, i < j → a + i * d < a + j * d ∧
  sum_of_digits (a + i * d) < sum_of_digits (a + j * d)) := by
  sorry

end NUMINAMATH_GPT_exists_arithmetic_progression_with_11_numbers_exists_arithmetic_progression_with_10000_numbers_not_exists_infinite_arithmetic_progression_l1135_113529


namespace NUMINAMATH_GPT_vasya_purchase_l1135_113582

theorem vasya_purchase : ∃ x y z w : ℕ, x + y + z + w = 15 ∧ 9 * x + 4 * z = 30 ∧ 2 * y + z = 9 ∧ w = 7 :=
by
  sorry

end NUMINAMATH_GPT_vasya_purchase_l1135_113582


namespace NUMINAMATH_GPT_bike_sharing_problem_l1135_113592

def combinations (n k : ℕ) : ℕ := (Nat.choose n k)

theorem bike_sharing_problem:
  let total_bikes := 10
  let blue_bikes := 4
  let yellow_bikes := 6
  let inspected_bikes := 4
  let way_two_blue := combinations blue_bikes 2 * combinations yellow_bikes 2
  let way_three_blue := combinations blue_bikes 3 * combinations yellow_bikes 1
  let way_four_blue := combinations blue_bikes 4
  way_two_blue + way_three_blue + way_four_blue = 115 :=
by
  sorry

end NUMINAMATH_GPT_bike_sharing_problem_l1135_113592
