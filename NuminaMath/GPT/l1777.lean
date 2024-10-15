import Mathlib

namespace NUMINAMATH_GPT_combined_resistance_parallel_l1777_177776

theorem combined_resistance_parallel (R1 R2 R3 : ℝ) (r : ℝ) (h1 : R1 = 2) (h2 : R2 = 5) (h3 : R3 = 6) :
  (1 / r) = (1 / R1) + (1 / R2) + (1 / R3) → r = 15 / 13 :=
by
  sorry

end NUMINAMATH_GPT_combined_resistance_parallel_l1777_177776


namespace NUMINAMATH_GPT_cricket_team_members_eq_11_l1777_177750

-- Definitions based on conditions:
def captain_age : ℕ := 26
def wicket_keeper_age : ℕ := 31
def avg_age_whole_team : ℕ := 24
def avg_age_remaining_players : ℕ := 23

-- Definition of n based on the problem conditions
def number_of_members (n : ℕ) : Prop :=
  n * avg_age_whole_team = (n - 2) * avg_age_remaining_players + (captain_age + wicket_keeper_age)

-- The proof statement:
theorem cricket_team_members_eq_11 : ∃ n, number_of_members n ∧ n = 11 := 
by
  use 11
  unfold number_of_members
  sorry

end NUMINAMATH_GPT_cricket_team_members_eq_11_l1777_177750


namespace NUMINAMATH_GPT_rope_length_after_100_cuts_l1777_177742

noncomputable def rope_cut (initial_length : ℝ) (num_cuts : ℕ) (cut_fraction : ℝ) : ℝ :=
  initial_length * (1 - cut_fraction) ^ num_cuts

theorem rope_length_after_100_cuts :
  rope_cut 1 100 (3 / 4) = (1 / 4) ^ 100 :=
by
  sorry

end NUMINAMATH_GPT_rope_length_after_100_cuts_l1777_177742


namespace NUMINAMATH_GPT__l1777_177731

noncomputable def urn_marble_theorem (r w b g y : Nat) : Prop :=
  let n := r + w + b + g + y
  ∃ k : Nat, 
  (k * r * (r-1) * (r-2) * (r-3) * (r-4) / 120 = w * r * (r-1) * (r-2) * (r-3) / 24)
  ∧ (w * r * (r-1) * (r-2) * (r-3) / 24 = w * b * r * (r-1) * (r-2) / 6)
  ∧ (w * b * r * (r-1) * (r-2) / 6 = w * b * g * r * (r-1) / 2)
  ∧ (w * b * g * r * (r-1) / 2 = w * b * g * r * y)
  ∧ n = 55

example : ∃ (r w b g y : Nat), urn_marble_theorem r w b g y := sorry

end NUMINAMATH_GPT__l1777_177731


namespace NUMINAMATH_GPT_complex_addition_l1777_177746

theorem complex_addition :
  (⟨6, -5⟩ : ℂ) + (⟨3, 2⟩ : ℂ) = ⟨9, -3⟩ := 
sorry

end NUMINAMATH_GPT_complex_addition_l1777_177746


namespace NUMINAMATH_GPT_num_solutions_even_pairs_l1777_177758

theorem num_solutions_even_pairs : ∃ n : ℕ, n = 25 ∧ ∀ (x y : ℕ),
  x % 2 = 0 ∧ y % 2 = 0 ∧ 4 * x + 6 * y = 600 → n = 25 :=
by
  sorry

end NUMINAMATH_GPT_num_solutions_even_pairs_l1777_177758


namespace NUMINAMATH_GPT_David_fewer_crunches_l1777_177786

-- Definitions as per conditions.
def Zachary_crunches := 62
def David_crunches := 45

-- Proof statement for how many fewer crunches David did compared to Zachary.
theorem David_fewer_crunches : Zachary_crunches - David_crunches = 17 := by
  -- Proof details would go here, but we skip them with 'sorry'.
  sorry

end NUMINAMATH_GPT_David_fewer_crunches_l1777_177786


namespace NUMINAMATH_GPT_rahul_deepak_present_ages_l1777_177783

theorem rahul_deepak_present_ages (R D : ℕ) 
  (h1 : R / D = 4 / 3)
  (h2 : R + 6 = 26)
  (h3 : D + 6 = 1/2 * (R + (R + 6)))
  (h4 : (R + 11) + (D + 11) = 59) 
  : R = 20 ∧ D = 17 :=
sorry

end NUMINAMATH_GPT_rahul_deepak_present_ages_l1777_177783


namespace NUMINAMATH_GPT_domain_f_l1777_177768

noncomputable def f (x : ℝ) : ℝ := (x - 3) / (x^2 - 5 * x + 6)

theorem domain_f :
  {x : ℝ | f x ≠ f x} = {x : ℝ | (x < 2) ∨ (2 < x ∧ x < 3) ∨ (3 < x)} :=
by sorry

end NUMINAMATH_GPT_domain_f_l1777_177768


namespace NUMINAMATH_GPT_simplify_expression_l1777_177757

theorem simplify_expression (x : ℝ) : 3 * x + 5 * x ^ 2 + 2 - (9 - 4 * x - 5 * x ^ 2) = 10 * x ^ 2 + 7 * x - 7 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1777_177757


namespace NUMINAMATH_GPT_six_digit_numbers_with_zero_l1777_177797

-- Define the total number of 6-digit numbers
def total_six_digit_numbers : ℕ := 900000

-- Define the number of 6-digit numbers with no zero
def six_digit_numbers_no_zero : ℕ := 531441

-- Prove the number of 6-digit numbers with at least one zero
theorem six_digit_numbers_with_zero : 
  (total_six_digit_numbers - six_digit_numbers_no_zero) = 368559 :=
by
  sorry

end NUMINAMATH_GPT_six_digit_numbers_with_zero_l1777_177797


namespace NUMINAMATH_GPT_max_value_expression_l1777_177707

theorem max_value_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 * a + 3 * b = 5) : 
  (∀ x y : ℝ, x = 2 * a + 2 → y = 3 * b + 1 → x * y ≤ 16) := by
  sorry

end NUMINAMATH_GPT_max_value_expression_l1777_177707


namespace NUMINAMATH_GPT_cone_base_radius_l1777_177790

/-- Given a semicircular piece of paper with a diameter of 2 cm is used to construct the 
  lateral surface of a cone, prove that the radius of the base of the cone is 0.5 cm. --/
theorem cone_base_radius (d : ℝ) (arc_length : ℝ) (circumference : ℝ) (r : ℝ)
  (h₀ : d = 2)
  (h₁ : arc_length = (1 / 2) * d * Real.pi)
  (h₂ : circumference = arc_length)
  (h₃ : r = circumference / (2 * Real.pi)) :
  r = 0.5 :=
by
  sorry

end NUMINAMATH_GPT_cone_base_radius_l1777_177790


namespace NUMINAMATH_GPT_number_of_dogs_l1777_177723

-- Conditions
def ratio_cats_dogs : ℚ := 3 / 4
def number_cats : ℕ := 18

-- Define the theorem to prove
theorem number_of_dogs : ∃ (dogs : ℕ), dogs = 24 :=
by
  -- Proof steps will go here, but we can use sorry for now to skip actual proving.
  sorry

end NUMINAMATH_GPT_number_of_dogs_l1777_177723


namespace NUMINAMATH_GPT_raine_steps_l1777_177720

theorem raine_steps (steps_per_trip : ℕ) (num_days : ℕ) (total_steps : ℕ) : 
  steps_per_trip = 150 → 
  num_days = 5 → 
  total_steps = steps_per_trip * 2 * num_days → 
  total_steps = 1500 := 
by 
  intros h1 h2 h3
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end NUMINAMATH_GPT_raine_steps_l1777_177720


namespace NUMINAMATH_GPT_minimum_P_ge_37_l1777_177705

noncomputable def minimum_P (x y z : ℝ) : ℝ := 
  (x / y + y / z + z / x) * (y / x + z / y + x / z)

theorem minimum_P_ge_37 (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h : (x / y + y / z + z / x) + (y / x + z / y + x / z) = 10) : 
  minimum_P x y z ≥ 37 :=
sorry

end NUMINAMATH_GPT_minimum_P_ge_37_l1777_177705


namespace NUMINAMATH_GPT_train_speed_kmph_l1777_177713

noncomputable def train_length : ℝ := 200
noncomputable def crossing_time : ℝ := 3.3330666879982935

theorem train_speed_kmph : (train_length / crossing_time) * 3.6 = 216.00072 := by
  sorry

end NUMINAMATH_GPT_train_speed_kmph_l1777_177713


namespace NUMINAMATH_GPT_solution_in_quadrant_II_l1777_177785

theorem solution_in_quadrant_II (k x y : ℝ) (h1 : 2 * x + y = 6) (h2 : k * x - y = 4) : x < 0 ∧ y > 0 ↔ k < -2 :=
by
  sorry

end NUMINAMATH_GPT_solution_in_quadrant_II_l1777_177785


namespace NUMINAMATH_GPT_total_amount_spent_l1777_177729

variable (B D : ℝ)

-- Conditions
def condition1 : Prop := D = (1/2) * B
def condition2 : Prop := B = D + 15

-- Proof statement
theorem total_amount_spent (h1 : condition1 B D) (h2 : condition2 B D) : B + D = 45 := by
  sorry

end NUMINAMATH_GPT_total_amount_spent_l1777_177729


namespace NUMINAMATH_GPT_difference_in_price_l1777_177700

noncomputable def total_cost : ℝ := 70.93
noncomputable def pants_price : ℝ := 34.00

theorem difference_in_price (total_cost pants_price : ℝ) (h_total : total_cost = 70.93) (h_pants : pants_price = 34.00) :
  (total_cost - pants_price) - pants_price = 2.93 :=
by
  sorry

end NUMINAMATH_GPT_difference_in_price_l1777_177700


namespace NUMINAMATH_GPT_sum_of_numbers_l1777_177702

theorem sum_of_numbers (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) 
  (h1 : x^2 + y^2 + z^2 = 52) (h2 : x * y + y * z + z * x = 24) : 
  x + y + z = 10 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_numbers_l1777_177702


namespace NUMINAMATH_GPT_ducks_in_marsh_l1777_177703

theorem ducks_in_marsh (total_birds geese : ℕ) (h1 : total_birds = 95) (h2 : geese = 58) : total_birds - geese = 37 := by
  sorry

end NUMINAMATH_GPT_ducks_in_marsh_l1777_177703


namespace NUMINAMATH_GPT_apple_picking_ratio_l1777_177794

theorem apple_picking_ratio (a b c : ℕ) 
  (h1 : a = 66) 
  (h2 : b = 2 * 66) 
  (h3 : a + b + c = 220) :
  c = 22 → a = 66 → c / a = 1 / 3 := by
    intros
    sorry

end NUMINAMATH_GPT_apple_picking_ratio_l1777_177794


namespace NUMINAMATH_GPT_max_value_2019m_2020n_l1777_177745

theorem max_value_2019m_2020n (m n : ℤ) (h1 : 0 ≤ m - n) (h2 : m - n ≤ 1) (h3 : 2 ≤ m + n) (h4 : m + n ≤ 4) :
  (∀ (m' n' : ℤ), (0 ≤ m' - n') → (m' - n' ≤ 1) → (2 ≤ m' + n') → (m' + n' ≤ 4) → (m - 2 * n ≥ m' - 2 * n')) →
  2019 * m + 2020 * n = 2019 :=
by
  sorry

end NUMINAMATH_GPT_max_value_2019m_2020n_l1777_177745


namespace NUMINAMATH_GPT_concert_tickets_l1777_177765

theorem concert_tickets : ∃ (A B : ℕ), 8 * A + 425 * B = 3000000 ∧ A + B = 4500 ∧ A = 2900 := by
  sorry

end NUMINAMATH_GPT_concert_tickets_l1777_177765


namespace NUMINAMATH_GPT_Lance_workdays_per_week_l1777_177712

theorem Lance_workdays_per_week (weekly_hours hourly_wage daily_earnings : ℕ) 
  (h1 : weekly_hours = 35)
  (h2 : hourly_wage = 9)
  (h3 : daily_earnings = 63) :
  weekly_hours / (daily_earnings / hourly_wage) = 5 := by
  sorry

end NUMINAMATH_GPT_Lance_workdays_per_week_l1777_177712


namespace NUMINAMATH_GPT_find_F_l1777_177737

theorem find_F (F C : ℝ) (h1 : C = 4/7 * (F - 40)) (h2 : C = 28) : F = 89 := 
by
  sorry

end NUMINAMATH_GPT_find_F_l1777_177737


namespace NUMINAMATH_GPT_solve_inequality_l1777_177781

theorem solve_inequality (a x : ℝ) :
  (a = 0 → x < 1) ∧
  (a ≠ 0 → ((a > 0 → (a-1)/a < x ∧ x < 1) ∧
            (a < 0 → (x < 1 ∨ x > (a-1)/a)))) :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l1777_177781


namespace NUMINAMATH_GPT_Z_evaluation_l1777_177793

def Z (x y : ℕ) : ℕ := x^2 - x * y + y^2

theorem Z_evaluation : Z 5 3 = 19 := by
  sorry

end NUMINAMATH_GPT_Z_evaluation_l1777_177793


namespace NUMINAMATH_GPT_fred_change_received_l1777_177755

theorem fred_change_received :
  let ticket_price := 5.92
  let ticket_count := 2
  let borrowed_movie_price := 6.79
  let amount_paid := 20.00
  let total_cost := (ticket_price * ticket_count) + borrowed_movie_price
  let change := amount_paid - total_cost
  change = 1.37 :=
by
  sorry

end NUMINAMATH_GPT_fred_change_received_l1777_177755


namespace NUMINAMATH_GPT_main_theorem_l1777_177764

noncomputable def exists_coprime_integers (a b p : ℤ) : Prop :=
  ∃ (m n : ℤ), Int.gcd m n = 1 ∧ p ∣ (a * m + b * n)

theorem main_theorem (a b p : ℤ) : exists_coprime_integers a b p := 
  sorry

end NUMINAMATH_GPT_main_theorem_l1777_177764


namespace NUMINAMATH_GPT_no_partition_of_integers_l1777_177736

theorem no_partition_of_integers (A B C : Set ℕ) :
  (A ≠ ∅ ∧ B ≠ ∅ ∧ C ≠ ∅) ∧
  (∀ a b, a ∈ A ∧ b ∈ B → (a^2 - a * b + b^2) ∈ C) ∧
  (∀ a b, a ∈ B ∧ b ∈ C → (a^2 - a * b + b^2) ∈ A) ∧
  (∀ a b, a ∈ C ∧ b ∈ A → (a^2 - a * b + b^2) ∈ B) →
  False := 
sorry

end NUMINAMATH_GPT_no_partition_of_integers_l1777_177736


namespace NUMINAMATH_GPT_rectangular_prism_volume_l1777_177769

theorem rectangular_prism_volume
  (l w h : ℝ)
  (face1 : l * w = 6)
  (face2 : w * h = 8)
  (face3 : l * h = 12) : l * w * h = 24 := sorry

end NUMINAMATH_GPT_rectangular_prism_volume_l1777_177769


namespace NUMINAMATH_GPT_neither_drinkers_eq_nine_l1777_177709

-- Define the number of businessmen at the conference
def total_businessmen : Nat := 30

-- Define the number of businessmen who drank coffee
def coffee_drinkers : Nat := 15

-- Define the number of businessmen who drank tea
def tea_drinkers : Nat := 13

-- Define the number of businessmen who drank both coffee and tea
def both_drinkers : Nat := 7

-- Prove the number of businessmen who drank neither coffee nor tea
theorem neither_drinkers_eq_nine : 
  total_businessmen - ((coffee_drinkers + tea_drinkers) - both_drinkers) = 9 := 
by
  sorry

end NUMINAMATH_GPT_neither_drinkers_eq_nine_l1777_177709


namespace NUMINAMATH_GPT_inequality_proof_l1777_177780

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 / b) + (b^2 / c) + (c^2 / a) ≥ a + b + c + 4 * (a - b)^2 / (a + b + c) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1777_177780


namespace NUMINAMATH_GPT_inverse_function_ratio_l1777_177715

noncomputable def g (x : ℚ) : ℚ := (3 * x + 2) / (2 * x - 5)

noncomputable def g_inv (x : ℚ) : ℚ := (-5 * x + 2) / (-2 * x + 3)

theorem inverse_function_ratio :
  ∀ x : ℚ, g (g_inv x) = x ∧ (∃ a b c d : ℚ, a = -5 ∧ b = 2 ∧ c = -2 ∧ d = 3 ∧ a / c = 2.5) :=
by
  sorry

end NUMINAMATH_GPT_inverse_function_ratio_l1777_177715


namespace NUMINAMATH_GPT_area_PQR_l1777_177791

-- Define the point P
def P : ℝ × ℝ := (1, 6)

-- Define the functions for lines passing through P with slopes 1 and 3
def line1 (x : ℝ) : ℝ := x + 5
def line2 (x : ℝ) : ℝ := 3 * x + 3

-- Define the x-intercepts of the lines
def Q : ℝ × ℝ := (-5, 0)
def R : ℝ × ℝ := (-1, 0)

-- Calculate the distance QR
def distance_QR : ℝ := abs (-1 - (-5))

-- Calculate the height from P to the x-axis
def height_P : ℝ := 6

-- State and prove the area of the triangle PQR
theorem area_PQR : 1 / 2 * distance_QR * height_P = 12 := by
  sorry -- The actual proof would be provided here

end NUMINAMATH_GPT_area_PQR_l1777_177791


namespace NUMINAMATH_GPT_solution_inequality_l1777_177740

theorem solution_inequality {x : ℝ} : x - 1 > 0 ↔ x > 1 := 
by
  sorry

end NUMINAMATH_GPT_solution_inequality_l1777_177740


namespace NUMINAMATH_GPT_jar_filling_fraction_l1777_177796

theorem jar_filling_fraction (C1 C2 C3 W : ℝ)
  (h1 : W = (1/7) * C1)
  (h2 : W = (2/9) * C2)
  (h3 : W = (3/11) * C3)
  (h4 : C3 > C1 ∧ C3 > C2) :
  (3 * W) = (9 / 11) * C3 :=
by sorry

end NUMINAMATH_GPT_jar_filling_fraction_l1777_177796


namespace NUMINAMATH_GPT_quadratic_roots_r6_s6_l1777_177717

theorem quadratic_roots_r6_s6 (r s : ℝ) (h1 : r + s = 3 * Real.sqrt 2) (h2 : r * s = 4) : r^6 + s^6 = 648 := by
  sorry

end NUMINAMATH_GPT_quadratic_roots_r6_s6_l1777_177717


namespace NUMINAMATH_GPT_right_angled_trapezoid_base_height_l1777_177704

theorem right_angled_trapezoid_base_height {a b : ℝ} (h : a = b) :
  ∃ (base height : ℝ), base = a ∧ height = b := 
by
  sorry

end NUMINAMATH_GPT_right_angled_trapezoid_base_height_l1777_177704


namespace NUMINAMATH_GPT_no_real_y_for_common_solution_l1777_177739

theorem no_real_y_for_common_solution :
  ∀ (x y : ℝ), x^2 + y^2 = 25 → x^2 + 3 * y = 45 → false :=
by 
sorry

end NUMINAMATH_GPT_no_real_y_for_common_solution_l1777_177739


namespace NUMINAMATH_GPT_j_h_five_l1777_177734

-- Define the functions h and j
def h (x : ℤ) : ℤ := 4 * x + 5
def j (x : ℤ) : ℤ := 6 * x - 11

-- State the theorem to prove j(h(5)) = 139
theorem j_h_five : j (h 5) = 139 := by
  sorry

end NUMINAMATH_GPT_j_h_five_l1777_177734


namespace NUMINAMATH_GPT_sum_of_roots_of_polynomials_l1777_177719

theorem sum_of_roots_of_polynomials :
  ∃ (a b : ℝ), (a^4 - 16 * a^3 + 40 * a^2 - 50 * a + 25 = 0) ∧ (b^4 - 24 * b^3 + 216 * b^2 - 720 * b + 625 = 0) ∧ (a + b = 7 ∨ a + b = 3) :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_roots_of_polynomials_l1777_177719


namespace NUMINAMATH_GPT_find_positive_integer_cube_root_divisible_by_21_l1777_177727

theorem find_positive_integer_cube_root_divisible_by_21 (m : ℕ) (h1: m = 735) :
  m % 21 = 0 ∧ 9 < (m : ℝ)^(1/3) ∧ (m : ℝ)^(1/3) < 9.1 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_positive_integer_cube_root_divisible_by_21_l1777_177727


namespace NUMINAMATH_GPT_calculate_expression_l1777_177767

theorem calculate_expression :
  4 + ((-2)^2) * 2 + (-36) / 4 = 3 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1777_177767


namespace NUMINAMATH_GPT_total_cost_of_color_drawing_l1777_177726

def cost_bwch_drawing : ℕ := 160
def bwch_to_color_cost_multiplier : ℝ := 1.5

theorem total_cost_of_color_drawing 
  (cost_bwch : ℕ)
  (bwch_to_color_mult : ℝ)
  (h₁ : cost_bwch = 160)
  (h₂ : bwch_to_color_mult = 1.5) :
  cost_bwch * bwch_to_color_mult = 240 := 
  by
    sorry

end NUMINAMATH_GPT_total_cost_of_color_drawing_l1777_177726


namespace NUMINAMATH_GPT_smallest_int_square_eq_3x_plus_72_l1777_177784

theorem smallest_int_square_eq_3x_plus_72 :
  ∃ x : ℤ, x^2 = 3 * x + 72 ∧ (∀ y : ℤ, y^2 = 3 * y + 72 → x ≤ y) :=
sorry

end NUMINAMATH_GPT_smallest_int_square_eq_3x_plus_72_l1777_177784


namespace NUMINAMATH_GPT_tangent_line_to_parabola_l1777_177752

noncomputable def parabola (x : ℝ) : ℝ := 4 * x^2

def derivative_parabola (x : ℝ) : ℝ := 8 * x

def tangent_line_eq (x y : ℝ) : Prop := 8 * x - y - 4 = 0

theorem tangent_line_to_parabola (x : ℝ) (hx : x = 1) (hy : parabola x = 4) :
    tangent_line_eq 1 4 :=
by 
  -- Sorry to skip the detailed proof, but it should follow the steps outlined in the solution.
  sorry

end NUMINAMATH_GPT_tangent_line_to_parabola_l1777_177752


namespace NUMINAMATH_GPT_largest_n_cube_condition_l1777_177751

theorem largest_n_cube_condition :
  ∃ n : ℕ, (n^3 + 4 * n^2 - 15 * n - 18 = k^3) ∧ ∀ m : ℕ, (m^3 + 4 * m^2 - 15 * m - 18 = k^3 → m ≤ n) → n = 19 :=
by
  sorry

end NUMINAMATH_GPT_largest_n_cube_condition_l1777_177751


namespace NUMINAMATH_GPT_ratio_Andrea_Jude_l1777_177763

-- Definitions
def number_of_tickets := 100
def tickets_left := 40
def tickets_sold := number_of_tickets - tickets_left

def Jude_tickets := 16
def Sandra_tickets := 4 + 1/2 * Jude_tickets
def Andrea_tickets := tickets_sold - (Jude_tickets + Sandra_tickets)

-- Assertion that needs proof
theorem ratio_Andrea_Jude : 
  (Andrea_tickets / Jude_tickets) = 2 := by
  sorry

end NUMINAMATH_GPT_ratio_Andrea_Jude_l1777_177763


namespace NUMINAMATH_GPT_sin_B_over_sin_A_eq_two_max_value_sin_A_sin_B_l1777_177710

-- Given conditions for the triangle ABC
variables {A B C a b c : ℝ}
axiom angle_C_eq_two_pi_over_three : C = 2 * Real.pi / 3
axiom c_squared_eq_five_a_squared_plus_ab : c^2 = 5 * a^2 + a * b

-- Proof statements
theorem sin_B_over_sin_A_eq_two (hAC: C = 2 * Real.pi / 3) (hCond: c^2 = 5 * a^2 + a * b) :
  Real.sin B / Real.sin A = 2 :=
sorry

theorem max_value_sin_A_sin_B (hAC: C = 2 * Real.pi / 3) :
  ∃ A B : ℝ, 0 < A ∧ A < Real.pi / 3 ∧ B = (Real.pi / 3 - A) ∧ Real.sin A * Real.sin B ≤ 1 / 4 :=
sorry

end NUMINAMATH_GPT_sin_B_over_sin_A_eq_two_max_value_sin_A_sin_B_l1777_177710


namespace NUMINAMATH_GPT_like_apple_orange_mango_l1777_177753

theorem like_apple_orange_mango (A B C: ℕ) 
  (h1: A = 40) 
  (h2: B = 7) 
  (h3: C = 10) 
  (total: ℕ) 
  (h_total: total = 47) 
: ∃ x: ℕ, 40 + (10 - x) + x = 47 ∧ x = 3 := 
by 
  sorry

end NUMINAMATH_GPT_like_apple_orange_mango_l1777_177753


namespace NUMINAMATH_GPT_extra_interest_amount_l1777_177771

def principal : ℝ := 15000
def rate1 : ℝ := 0.15
def rate2 : ℝ := 0.12
def time : ℕ := 2

theorem extra_interest_amount :
  principal * (rate1 - rate2) * time = 900 := by
  sorry

end NUMINAMATH_GPT_extra_interest_amount_l1777_177771


namespace NUMINAMATH_GPT_angle_in_quadrant_l1777_177721

-- Define the problem statement as a theorem to prove
theorem angle_in_quadrant (α : ℝ) (k : ℤ) 
  (hα : 2 * (k:ℝ) * Real.pi + Real.pi < α ∧ α < 2 * (k:ℝ) * Real.pi + 3 * Real.pi / 2) :
  (k:ℝ) * Real.pi + Real.pi / 2 < α / 2 ∧ α / 2 < (k:ℝ) * Real.pi + 3 * Real.pi / 4 := 
sorry

end NUMINAMATH_GPT_angle_in_quadrant_l1777_177721


namespace NUMINAMATH_GPT_noodles_initial_count_l1777_177766

theorem noodles_initial_count (noodles_given : ℕ) (noodles_now : ℕ) (initial_noodles : ℕ) 
  (h_given : noodles_given = 12) (h_now : noodles_now = 54) (h_initial_noodles : initial_noodles = noodles_now + noodles_given) : 
  initial_noodles = 66 :=
by 
  rw [h_now, h_given] at h_initial_noodles
  exact h_initial_noodles

-- Adding 'sorry' since the solution steps are not required

end NUMINAMATH_GPT_noodles_initial_count_l1777_177766


namespace NUMINAMATH_GPT_sin_2alpha_plus_pi_over_3_cos_beta_minus_pi_over_6_l1777_177788

variable (α β : ℝ)
variable (hα : α < π/2) (hβ : β < π/2) -- acute angles
variable (h1 : Real.cos (α + π/6) = 3/5)
variable (h2 : Real.cos (α + β) = -Real.sqrt 5 / 5)

theorem sin_2alpha_plus_pi_over_3 :
  Real.sin (2 * α + π/3) = 24 / 25 :=
by
  sorry

theorem cos_beta_minus_pi_over_6 :
  Real.cos (β - π/6) = Real.sqrt 5 / 5 :=
by
  sorry

end NUMINAMATH_GPT_sin_2alpha_plus_pi_over_3_cos_beta_minus_pi_over_6_l1777_177788


namespace NUMINAMATH_GPT_line_through_circle_center_l1777_177706

theorem line_through_circle_center
  (C : ℝ × ℝ)
  (hC : C = (-1, 0))
  (hCircle : ∀ (x y : ℝ), x^2 + 2 * x + y^2 = 0 → (x, y) = (-1, 0))
  (hPerpendicular : ∀ (m₁ m₂ : ℝ), (m₁ * m₂ = -1) → m₁ = -1 → m₂ = 1)
  (line_eq : ∀ (x y : ℝ), y = x + 1)
  : ∀ (x y : ℝ), x - y + 1 = 0 :=
sorry

end NUMINAMATH_GPT_line_through_circle_center_l1777_177706


namespace NUMINAMATH_GPT_symmetric_point_coordinates_l1777_177754

noncomputable def symmetric_with_respect_to_y_axis (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  match p with
  | (x, y, z) => (-x, y, -z)

theorem symmetric_point_coordinates : symmetric_with_respect_to_y_axis (-2, 1, 4) = (2, 1, -4) :=
by sorry

end NUMINAMATH_GPT_symmetric_point_coordinates_l1777_177754


namespace NUMINAMATH_GPT_b_is_dk_squared_l1777_177760

theorem b_is_dk_squared (a b : ℤ) (h : ∃ r1 r2 r3 : ℤ, (r1 * r2 * r3 = b) ∧ (r1 + r2 + r3 = a) ∧ (r1 * r2 + r1 * r3 + r2 * r3 = 0))
  : ∃ d k : ℤ, (b = d * k^2) ∧ (d ∣ a) := 
sorry

end NUMINAMATH_GPT_b_is_dk_squared_l1777_177760


namespace NUMINAMATH_GPT_parking_garage_capacity_l1777_177722

open Nat

-- Definitions from the conditions
def first_level_spaces : Nat := 90
def second_level_spaces : Nat := first_level_spaces + 8
def third_level_spaces : Nat := second_level_spaces + 12
def fourth_level_spaces : Nat := third_level_spaces - 9
def initial_parked_cars : Nat := 100

-- The proof statement
theorem parking_garage_capacity : 
  (first_level_spaces + second_level_spaces + third_level_spaces + fourth_level_spaces - initial_parked_cars) = 299 := 
  by 
    sorry

end NUMINAMATH_GPT_parking_garage_capacity_l1777_177722


namespace NUMINAMATH_GPT_square_side_length_l1777_177748

variable (s : ℕ)
variable (P A : ℕ)

theorem square_side_length (h1 : P = 52) (h2 : A = 169) (h3 : P = 4 * s) (h4 : A = s * s) : s = 13 :=
sorry

end NUMINAMATH_GPT_square_side_length_l1777_177748


namespace NUMINAMATH_GPT_permutations_with_k_in_first_position_l1777_177799

noncomputable def numberOfPermutationsWithKInFirstPosition (N k : ℕ) (h : k < N) : ℕ :=
  (2 : ℕ)^(N-1)

theorem permutations_with_k_in_first_position (N k : ℕ) (h : k < N) :
  numberOfPermutationsWithKInFirstPosition N k h = (2 : ℕ)^(N-1) :=
sorry

end NUMINAMATH_GPT_permutations_with_k_in_first_position_l1777_177799


namespace NUMINAMATH_GPT_find_solution_set_l1777_177732

-- Define the problem
def absolute_value_equation_solution_set (x : ℝ) : Prop :=
  |x - 2| + |2 * x - 3| = |3 * x - 5|

-- Define the expected solution set
def solution_set (x : ℝ) : Prop :=
  x ≤ 3 / 2 ∨ 2 ≤ x

-- The proof problem statement
theorem find_solution_set :
  ∀ x : ℝ, absolute_value_equation_solution_set x ↔ solution_set x :=
sorry -- No proof required, so we use 'sorry' to skip the proof

end NUMINAMATH_GPT_find_solution_set_l1777_177732


namespace NUMINAMATH_GPT_mary_donated_books_l1777_177749

theorem mary_donated_books 
  (s : ℕ) (b_c : ℕ) (b_b : ℕ) (b_y : ℕ) (g_d : ℕ) (g_m : ℕ) (e : ℕ) (s_s : ℕ) 
  (total : ℕ) (out_books : ℕ) (d : ℕ)
  (h1 : s = 72)
  (h2 : b_c = 12)
  (h3 : b_b = 5)
  (h4 : b_y = 2)
  (h5 : g_d = 1)
  (h6 : g_m = 4)
  (h7 : e = 81)
  (h8 : s_s = 3)
  (ht : total = s + b_c + b_b + b_y + g_d + g_m)
  (ho : out_books = total - e)
  (hd : d = out_books - s_s) :
  d = 12 :=
by { sorry }

end NUMINAMATH_GPT_mary_donated_books_l1777_177749


namespace NUMINAMATH_GPT_cube_partition_exists_l1777_177718

theorem cube_partition_exists : ∃ (n_0 : ℕ), (0 < n_0) ∧ (∀ (n : ℕ), n ≥ n_0 → ∃ k : ℕ, n = k) := sorry

end NUMINAMATH_GPT_cube_partition_exists_l1777_177718


namespace NUMINAMATH_GPT_total_worth_all_crayons_l1777_177770

def cost_of_crayons (packs: ℕ) (cost_per_pack: ℝ) : ℝ := packs * cost_per_pack

def discounted_cost (cost: ℝ) (discount_rate: ℝ) : ℝ := cost * (1 - discount_rate)

def tax_amount (cost: ℝ) (tax_rate: ℝ) : ℝ := cost * tax_rate

theorem total_worth_all_crayons : 
  let cost_per_pack := 2.5
  let discount_rate := 0.15
  let tax_rate := 0.07
  let packs_already_have := 4
  let packs_to_buy := 2
  let cost_two_packs := cost_of_crayons packs_to_buy cost_per_pack
  let discounted_two_packs := discounted_cost cost_two_packs discount_rate
  let tax_two_packs := tax_amount cost_two_packs tax_rate
  let total_cost_two_packs := discounted_two_packs + tax_two_packs
  let cost_four_packs := cost_of_crayons packs_already_have cost_per_pack
  cost_four_packs + total_cost_two_packs = 14.60 := 
by 
  sorry

end NUMINAMATH_GPT_total_worth_all_crayons_l1777_177770


namespace NUMINAMATH_GPT_general_term_of_sequence_l1777_177743

theorem general_term_of_sequence
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (h_pos_a : ∀ n, 0 < a n)
  (h_pos_b : ∀ n, 0 < b n)
  (h_arith : ∀ n, 2 * b n = a n + a (n + 1))
  (h_geom : ∀ n, (a (n + 1))^2 = b n * b (n + 1))
  (h_a1 : a 1 = 1)
  (h_a2 : a 2 = 3)
  : ∀ n, a n = (n^2 + n) / 2 :=
by
  sorry

end NUMINAMATH_GPT_general_term_of_sequence_l1777_177743


namespace NUMINAMATH_GPT_hexagon_perimeter_l1777_177756

def side_length : ℕ := 10
def num_sides : ℕ := 6

theorem hexagon_perimeter : num_sides * side_length = 60 := by
  sorry

end NUMINAMATH_GPT_hexagon_perimeter_l1777_177756


namespace NUMINAMATH_GPT_ellipse_standard_equation_l1777_177787

theorem ellipse_standard_equation
  (a b : ℝ) (P : ℝ × ℝ) (h_center : P = (3, 0))
  (h_a_eq_3b : a = 3 * b) 
  (h1 : a = 3) 
  (h2 : b = 1) : 
  (∀ (x y : ℝ), (x = 3 → y = 0) → (x = 0 → y = 3)) → 
  ((x^2 / a^2) + y^2 = 1 ∨ (x^2 / b^2) + (y^2 / a^2) = 1) := 
by sorry

end NUMINAMATH_GPT_ellipse_standard_equation_l1777_177787


namespace NUMINAMATH_GPT_inequality_product_lt_zero_l1777_177772

theorem inequality_product_lt_zero (a b c : ℝ) (h1 : a > b) (h2 : c < 1) : (a - b) * (c - 1) < 0 :=
  sorry

end NUMINAMATH_GPT_inequality_product_lt_zero_l1777_177772


namespace NUMINAMATH_GPT_math_problem_l1777_177724
-- Import necessary modules

-- Define the condition as a hypothesis and state the theorem
theorem math_problem (x : ℝ) (h : 8 * x - 6 = 10) : 50 * (1 / x) + 150 = 175 :=
sorry

end NUMINAMATH_GPT_math_problem_l1777_177724


namespace NUMINAMATH_GPT_largest_angle_of_triangle_l1777_177777

theorem largest_angle_of_triangle 
  (α β γ : ℝ) 
  (h1 : α = 60) 
  (h2 : β = 70) 
  (h3 : α + β + γ = 180) : 
  max α (max β γ) = 70 := 
by 
  sorry

end NUMINAMATH_GPT_largest_angle_of_triangle_l1777_177777


namespace NUMINAMATH_GPT_compare_logs_l1777_177774

noncomputable def a := Real.log 6 / Real.log 3
noncomputable def b := Real.log 10 / Real.log 5
noncomputable def c := Real.log 14 / Real.log 7

theorem compare_logs : a > b ∧ b > c := by
  -- Proof will be written here, currently placeholder
  sorry

end NUMINAMATH_GPT_compare_logs_l1777_177774


namespace NUMINAMATH_GPT_total_thread_needed_l1777_177759

def keychain_length : Nat := 12
def friends_in_classes : Nat := 10
def multiplier_for_club_friends : Nat := 2
def thread_per_class_friend : Nat := 16
def thread_per_club_friend : Nat := 20

theorem total_thread_needed :
  10 * thread_per_class_friend + (10 * multiplier_for_club_friends) * thread_per_club_friend = 560 := by
  sorry

end NUMINAMATH_GPT_total_thread_needed_l1777_177759


namespace NUMINAMATH_GPT_find_a_l1777_177741

open Real

noncomputable def valid_solutions (a b : ℝ) : Prop :=
  a + 2 / b = 17 ∧ b + 2 / a = 1 / 3

theorem find_a (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : valid_solutions a b) :
  a = 6 ∨ a = 17 :=
by sorry

end NUMINAMATH_GPT_find_a_l1777_177741


namespace NUMINAMATH_GPT_min_questions_any_three_cards_min_questions_consecutive_three_cards_l1777_177782

-- Definitions for numbers on cards and necessary questions
variables (n : ℕ) (h_n : n > 3)
  (cards : Fin n → ℤ)
  (h_cards_range : ∀ i, cards i = 1 ∨ cards i = -1)

-- Case (a): Product of any three cards
theorem min_questions_any_three_cards :
  (∃ (k : ℕ), n = 3 * k ∧ p = k) ∨
  (∃ (k : ℕ), n = 3 * k + 1 ∧ p = k + 1) ∨
  (∃ (k : ℕ), n = 3 * k + 2 ∧ p = k + 2) :=
sorry
  
-- Case (b): Product of any three consecutive cards
theorem min_questions_consecutive_three_cards :
  (∃ (k : ℕ), n = 3 * k ∧ p = k) ∨
  (¬(∃ (k : ℕ), n = 3 * k) ∧ p = n) :=
sorry

end NUMINAMATH_GPT_min_questions_any_three_cards_min_questions_consecutive_three_cards_l1777_177782


namespace NUMINAMATH_GPT_probability_ratio_l1777_177735

theorem probability_ratio :
  let draws := 4
  let total_slips := 40
  let numbers := 10
  let slips_per_number := 4
  let p := 10 / (Nat.choose total_slips draws)
  let q := (Nat.choose numbers 2) * (Nat.choose slips_per_number 2) * (Nat.choose slips_per_number 2) / (Nat.choose total_slips draws)
  p ≠ 0 →
  (q / p) = 162 :=
by
  sorry

end NUMINAMATH_GPT_probability_ratio_l1777_177735


namespace NUMINAMATH_GPT_altitude_line_equation_equal_distance_lines_l1777_177761

-- Define the points A, B, and C
def A : ℝ × ℝ := (4, 0)
def B : ℝ × ℝ := (8, 10)
def C : ℝ × ℝ := (0, 6)

-- The equation of the line for the altitude from A to BC
theorem altitude_line_equation :
  ∃ (a b c : ℝ), 2 * a - 3 * b + 14 = 0 :=
sorry

-- The equations of the line passing through B such that the distances from A and C are equal
theorem equal_distance_lines :
  ∃ (a b c : ℝ), (7 * a - 6 * b + 4 = 0) ∧ (3 * a + 2 * b - 44 = 0) :=
sorry

end NUMINAMATH_GPT_altitude_line_equation_equal_distance_lines_l1777_177761


namespace NUMINAMATH_GPT_largest_multiple_of_12_neg_gt_neg_150_l1777_177789

theorem largest_multiple_of_12_neg_gt_neg_150 : ∃ m : ℤ, (m % 12 = 0) ∧ (-m > -150) ∧ ∀ n : ℤ, (n % 12 = 0) ∧ (-n > -150) → n ≤ m := sorry

end NUMINAMATH_GPT_largest_multiple_of_12_neg_gt_neg_150_l1777_177789


namespace NUMINAMATH_GPT_problem_solution_l1777_177795

theorem problem_solution
  (x1 y1 x2 y2 x3 y3 : ℝ)
  (h1 : x1^3 - 3 * x1 * y1^2 = 2007)
  (h2 : y1^3 - 3 * x1^2 * y1 = 2006)
  (h3 : x2^3 - 3 * x2 * y2^2 = 2007)
  (h4 : y2^3 - 3 * x2^2 * y2 = 2006)
  (h5 : x3^3 - 3 * x3 * y3^2 = 2007)
  (h6 : y3^3 - 3 * x3^2 * y3 = 2006) :
  (1 - x1 / y1) * (1 - x2 / y2) * (1 - x3 / y3) = 1 / 1003 := 
sorry

end NUMINAMATH_GPT_problem_solution_l1777_177795


namespace NUMINAMATH_GPT_total_spent_is_195_l1777_177716

def hoodie_cost : ℝ := 80
def flashlight_cost : ℝ := 0.2 * hoodie_cost
def boots_original_cost : ℝ := 110
def boots_discount : ℝ := 0.1
def boots_discounted_cost : ℝ := boots_original_cost * (1 - boots_discount)
def total_cost : ℝ := hoodie_cost + flashlight_cost + boots_discounted_cost

theorem total_spent_is_195 : total_cost = 195 := by
  sorry

end NUMINAMATH_GPT_total_spent_is_195_l1777_177716


namespace NUMINAMATH_GPT_determine_h_l1777_177728

theorem determine_h (h : ℝ) : (∃ x : ℝ, x = 3 ∧ x^3 - 2 * h * x + 15 = 0) → h = 7 :=
by
  intro hx
  sorry

end NUMINAMATH_GPT_determine_h_l1777_177728


namespace NUMINAMATH_GPT_kathleen_allowance_l1777_177778

theorem kathleen_allowance (x : ℝ) :
  let middle_school_allowance := 10
  let senior_year_allowance := 10 * x + 5
  let percentage_increase := ((senior_year_allowance - middle_school_allowance) / middle_school_allowance) * 100
  percentage_increase = 150 → x = 2 :=
by
  -- Definitions and conditions setup
  let middle_school_allowance := 10
  let senior_year_allowance := 10 * x + 5
  let percentage_increase := ((senior_year_allowance - middle_school_allowance) / middle_school_allowance) * 100
  intros h
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_kathleen_allowance_l1777_177778


namespace NUMINAMATH_GPT_sum_first_8_even_numbers_is_72_l1777_177779

theorem sum_first_8_even_numbers_is_72 : (2 + 4 + 6 + 8 + 10 + 12 + 14 + 16) = 72 :=
by
  sorry

end NUMINAMATH_GPT_sum_first_8_even_numbers_is_72_l1777_177779


namespace NUMINAMATH_GPT_probability_correct_l1777_177747

-- Define the set and the probability calculation
def set : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Function to check if the difference condition holds
def valid_triplet (a b c: ℕ) : Prop := a < b ∧ b < c ∧ c - a = 4

-- Total number of ways to pick 3 numbers and ways that fit the condition
noncomputable def total_ways : ℕ := Nat.choose 9 3
noncomputable def valid_ways : ℕ := 5 * 2

-- Calculate the probability
noncomputable def probability : ℚ := valid_ways / total_ways

-- The theorem statement
theorem probability_correct : probability = 5 / 42 := by sorry

end NUMINAMATH_GPT_probability_correct_l1777_177747


namespace NUMINAMATH_GPT_roots_expression_l1777_177701

theorem roots_expression (p q : ℝ) (hpq : (∀ x, 3*x^2 + 9*x - 21 = 0 → x = p ∨ x = q)) 
  (sum_roots : p + q = -3) 
  (prod_roots : p * q = -7) : (3*p - 4) * (6*q - 8) = 122 :=
by
  sorry

end NUMINAMATH_GPT_roots_expression_l1777_177701


namespace NUMINAMATH_GPT_university_theater_ticket_sales_l1777_177744

theorem university_theater_ticket_sales (total_tickets : ℕ) (adult_price : ℕ) (senior_price : ℕ) (senior_tickets : ℕ) 
  (h1 : total_tickets = 510) (h2 : adult_price = 21) (h3 : senior_price = 15) (h4 : senior_tickets = 327) : 
  (total_tickets - senior_tickets) * adult_price + senior_tickets * senior_price = 8748 :=
by 
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_university_theater_ticket_sales_l1777_177744


namespace NUMINAMATH_GPT_toys_left_after_two_weeks_l1777_177798

theorem toys_left_after_two_weeks
  (initial_stock : ℕ)
  (sold_first_week : ℕ)
  (sold_second_week : ℕ)
  (total_stock : initial_stock = 83)
  (first_week_sales : sold_first_week = 38)
  (second_week_sales : sold_second_week = 26) :
  initial_stock - (sold_first_week + sold_second_week) = 19 :=
by
  sorry

end NUMINAMATH_GPT_toys_left_after_two_weeks_l1777_177798


namespace NUMINAMATH_GPT_smallest_portion_proof_l1777_177708

theorem smallest_portion_proof :
  ∃ (a d : ℚ), 5 * a = 100 ∧ 3 * (a + d) = 2 * d + 7 * (a - 2 * d) ∧ a - 2 * d = 5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_smallest_portion_proof_l1777_177708


namespace NUMINAMATH_GPT_x_lt_2_necessary_not_sufficient_x_sq_lt_4_l1777_177773

theorem x_lt_2_necessary_not_sufficient_x_sq_lt_4 (x : ℝ) :
  (x < 2) → (x^2 < 4) ∧ ¬((x^2 < 4) → (x < 2)) :=
by
  sorry

end NUMINAMATH_GPT_x_lt_2_necessary_not_sufficient_x_sq_lt_4_l1777_177773


namespace NUMINAMATH_GPT_arithmetic_sequences_integer_ratio_count_l1777_177711

theorem arithmetic_sequences_integer_ratio_count 
  (a_n b_n : ℕ → ℕ)
  (A_n B_n : ℕ → ℕ)
  (h₁ : ∀ n, A_n n = n * (a_n 1 + a_n (2 * n - 1)) / 2)
  (h₂ : ∀ n, B_n n = n * (b_n 1 + b_n (2 * n - 1)) / 2)
  (h₃ : ∀ n, A_n n / B_n n = (7 * n + 41) / (n + 3)) :
  ∃ (cnt : ℕ), cnt = 3 ∧ ∀ n, (∃ k, n = 1 + 3 * k) → (a_n n) / (b_n n) = 7 + (10 / (n + 1)) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequences_integer_ratio_count_l1777_177711


namespace NUMINAMATH_GPT_smallest_positive_four_digit_integer_equivalent_to_3_mod_4_l1777_177725

theorem smallest_positive_four_digit_integer_equivalent_to_3_mod_4 : 
  ∃ n : ℤ, n ≥ 1000 ∧ n % 4 = 3 ∧ n = 1003 := 
by {
  sorry
}

end NUMINAMATH_GPT_smallest_positive_four_digit_integer_equivalent_to_3_mod_4_l1777_177725


namespace NUMINAMATH_GPT_percentage_spent_on_household_items_eq_50_l1777_177733

-- Definitions for the conditions in the problem
def MonthlyIncome : ℝ := 90000
def ClothesPercentage : ℝ := 0.25
def MedicinesPercentage : ℝ := 0.15
def Savings : ℝ := 9000

-- Definition of the statement where we need to calculate the percentage spent on household items
theorem percentage_spent_on_household_items_eq_50 :
  let ClothesExpense := ClothesPercentage * MonthlyIncome
  let MedicinesExpense := MedicinesPercentage * MonthlyIncome
  let TotalExpense := ClothesExpense + MedicinesExpense + Savings
  let HouseholdItemsExpense := MonthlyIncome - TotalExpense
  let TotalIncome := MonthlyIncome
  (HouseholdItemsExpense / TotalIncome) * 100 = 50 :=
by
  sorry

end NUMINAMATH_GPT_percentage_spent_on_household_items_eq_50_l1777_177733


namespace NUMINAMATH_GPT_fraction_is_square_l1777_177730

theorem fraction_is_square (a b : ℕ) (hpos_a : a > 0) (hpos_b : b > 0) 
  (hdiv : (ab + 1) ∣ (a^2 + b^2)) :
  ∃ k : ℕ, k^2 = (a^2 + b^2) / (ab + 1) :=
sorry

end NUMINAMATH_GPT_fraction_is_square_l1777_177730


namespace NUMINAMATH_GPT_min_phi_l1777_177762

theorem min_phi
  (ϕ : ℝ) (hϕ : ϕ > 0)
  (h_symm : ∃ k : ℤ, 2 * (π / 6) - 2 * ϕ = k * π + π / 2) :
  ϕ = 5 * π / 12 :=
sorry

end NUMINAMATH_GPT_min_phi_l1777_177762


namespace NUMINAMATH_GPT_engineer_walk_duration_l1777_177714

variables (D : ℕ) (S : ℕ) (v : ℕ) (t : ℕ) (t1 : ℕ)

-- Stating the conditions
-- The time car normally takes to travel distance D
-- Speed (S) times the time (t) equals distance (D)
axiom speed_distance_relation : S * t = D

-- Engineer arrives at station at 7:00 AM and walks towards the car
-- They meet at t1 minutes past 7:00 AM, and the car covers part of the distance
-- Engineer reaches factory 20 minutes earlier than usual
-- Therefore, the car now meets the engineer covering less distance and time
axiom car_meets_engineer : S * t1 + v * t1 = D

-- The total travel time to the factory is reduced by 20 minutes
axiom travel_time_reduction : t - t1 = (t - 20 / 60)

-- Mathematically equivalent proof problem
theorem engineer_walk_duration : t1 = 50 := by
  sorry

end NUMINAMATH_GPT_engineer_walk_duration_l1777_177714


namespace NUMINAMATH_GPT_circle_radius_tangent_to_ellipse_l1777_177792

theorem circle_radius_tangent_to_ellipse (r : ℝ) :
  (∀ x y : ℝ, (x - r)^2 + y^2 = r^2 → x^2 + 4*y^2 = 8) ↔ r = (Real.sqrt 6) / 2 :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_tangent_to_ellipse_l1777_177792


namespace NUMINAMATH_GPT_find_blue_balls_l1777_177775

/-- 
Given the conditions that a bag contains:
- 5 red balls
- B blue balls
- 2 green balls
And the probability of picking 2 red balls at random is 0.1282051282051282,
prove that the number of blue balls (B) is 6.
--/

theorem find_blue_balls (B : ℕ) (h : 0.1282051282051282 = (10 : ℚ) / (↑((7 + B) * (6 + B)) / 2)) : B = 6 := 
by sorry

end NUMINAMATH_GPT_find_blue_balls_l1777_177775


namespace NUMINAMATH_GPT_stratified_sampling_male_students_l1777_177738

theorem stratified_sampling_male_students (total_students : ℕ) (female_students : ℕ) (sample_size : ℕ)
  (h1 : total_students = 900) (h2 : female_students = 0) (h3 : sample_size = 45) : 
  ((total_students - female_students) * sample_size / total_students) = 25 := 
by {
  sorry
}

end NUMINAMATH_GPT_stratified_sampling_male_students_l1777_177738
