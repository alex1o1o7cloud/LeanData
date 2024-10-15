import Mathlib

namespace NUMINAMATH_GPT_total_earnings_l487_48747

-- Definitions from the conditions.
def LaurynEarnings : ℝ := 2000
def AureliaEarnings : ℝ := 0.7 * LaurynEarnings

-- The theorem to prove.
theorem total_earnings (hL : LaurynEarnings = 2000) (hA : AureliaEarnings = 0.7 * 2000) :
    LaurynEarnings + AureliaEarnings = 3400 :=
by
    sorry  -- Placeholder for the proof.

end NUMINAMATH_GPT_total_earnings_l487_48747


namespace NUMINAMATH_GPT_loss_recorded_as_negative_l487_48714

-- Define the condition that a profit of 100 yuan is recorded as +100 yuan
def recorded_profit (p : ℤ) : Prop :=
  p = 100

-- Define the condition about how a profit is recorded
axiom profit_condition : recorded_profit 100

-- Define the function for recording profit or loss
def record (x : ℤ) : ℤ :=
  if x > 0 then x
  else -x

-- Theorem: If a profit of 100 yuan is recorded as +100 yuan, then a loss of 50 yuan is recorded as -50 yuan.
theorem loss_recorded_as_negative : ∀ x: ℤ, (x < 0) → record x = -x :=
by
  intros x h
  unfold record
  simp [h]
  -- sorry indicates the proof is not provided
  sorry

end NUMINAMATH_GPT_loss_recorded_as_negative_l487_48714


namespace NUMINAMATH_GPT_amy_local_calls_l487_48778

theorem amy_local_calls (L I : ℕ) 
  (h1 : 2 * L = 5 * I)
  (h2 : 3 * L = 5 * (I + 3)) : 
  L = 15 :=
by
  sorry

end NUMINAMATH_GPT_amy_local_calls_l487_48778


namespace NUMINAMATH_GPT_winning_post_distance_l487_48781

theorem winning_post_distance (v x : ℝ) (h₁ : x ≠ 0) (h₂ : v ≠ 0)
  (h₃ : 1.75 * v = v) 
  (h₄ : x = 1.75 * (x - 84)) : 
  x = 196 :=
by 
  sorry

end NUMINAMATH_GPT_winning_post_distance_l487_48781


namespace NUMINAMATH_GPT_base_length_of_isosceles_triangle_l487_48798

theorem base_length_of_isosceles_triangle (a b : ℕ) 
    (h₁ : a = 8) 
    (h₂ : 2 * a + b = 25) : 
    b = 9 :=
by
  -- This is the proof stub. Proof will be provided here.
  sorry

end NUMINAMATH_GPT_base_length_of_isosceles_triangle_l487_48798


namespace NUMINAMATH_GPT_loss_per_metre_eq_12_l487_48734

-- Definitions based on the conditions
def totalMetres : ℕ := 200
def totalSellingPrice : ℕ := 12000
def costPricePerMetre : ℕ := 72

-- Theorem statement to prove the loss per metre of cloth
theorem loss_per_metre_eq_12 : (costPricePerMetre * totalMetres - totalSellingPrice) / totalMetres = 12 := 
by sorry

end NUMINAMATH_GPT_loss_per_metre_eq_12_l487_48734


namespace NUMINAMATH_GPT_bc_eq_one_area_of_triangle_l487_48704

variable (a b c A B : ℝ)

-- Conditions
def condition_1 : Prop := (b^2 + c^2 - a^2) / (Real.cos A) = 2
def condition_2 : Prop := (a * (Real.cos B) - b * (Real.cos A)) / (a * (Real.cos B) + b * (Real.cos A)) - b / c = 1

-- Equivalent proof problems
theorem bc_eq_one (h1 : condition_1 a b c A) : b * c = 1 := 
by 
  sorry

theorem area_of_triangle (h2 : condition_2 a b c A B) : (1/2) * b * c * Real.sin A = (Real.sqrt 3) / 4 := 
by 
  sorry

end NUMINAMATH_GPT_bc_eq_one_area_of_triangle_l487_48704


namespace NUMINAMATH_GPT_solve_cos_2x_eq_cos_x_plus_sin_x_l487_48753

open Real

theorem solve_cos_2x_eq_cos_x_plus_sin_x :
  ∀ x : ℝ,
    (cos (2 * x) = cos x + sin x) ↔
    (∃ k : ℤ, x = k * π - π / 4) ∨ 
    (∃ k : ℤ, x = 2 * k * π) ∨
    (∃ k : ℤ, x = 2 * k * π - π / 2) := 
sorry

end NUMINAMATH_GPT_solve_cos_2x_eq_cos_x_plus_sin_x_l487_48753


namespace NUMINAMATH_GPT_count_divisible_by_45_l487_48700

theorem count_divisible_by_45 : ∃ n : ℕ, n = 10 ∧ (∀ x : ℕ, 1000 ≤ x ∧ x < 10000 ∧ x % 100 = 45 → x % 45 = 0 → n = 10) :=
by {
  sorry
}

end NUMINAMATH_GPT_count_divisible_by_45_l487_48700


namespace NUMINAMATH_GPT_min_value_expression_l487_48717

theorem min_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  (x + y) * (1 / x + 1 / y) ≥ 6 := 
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l487_48717


namespace NUMINAMATH_GPT_geese_more_than_ducks_l487_48772

-- Define initial conditions
def initial_ducks : ℕ := 25
def initial_geese : ℕ := 2 * initial_ducks - 10
def additional_ducks : ℕ := 4
def geese_leaving : ℕ := 15 - 5

-- Calculate the number of remaining ducks
def remaining_ducks : ℕ := initial_ducks + additional_ducks

-- Calculate the number of remaining geese
def remaining_geese : ℕ := initial_geese - geese_leaving

-- Prove the final statement
theorem geese_more_than_ducks : (remaining_geese - remaining_ducks) = 1 := by
  sorry

end NUMINAMATH_GPT_geese_more_than_ducks_l487_48772


namespace NUMINAMATH_GPT_find_breadth_of_cuboid_l487_48764

variable (l : ℝ) (h : ℝ) (surface_area : ℝ) (b : ℝ)

theorem find_breadth_of_cuboid (hL : l = 10) (hH : h = 6) (hSA : surface_area = 480) 
  (hFormula : surface_area = 2 * (l * b + b * h + h * l)) : b = 11.25 := by
  sorry

end NUMINAMATH_GPT_find_breadth_of_cuboid_l487_48764


namespace NUMINAMATH_GPT_speed_maintained_l487_48724

-- Given conditions:
def distance : ℕ := 324
def original_time : ℕ := 6
def new_time : ℕ := (3 * original_time) / 2

-- Correct answer:
def required_speed : ℕ := 36

-- Lean 4 statement to prove the equivalence:
theorem speed_maintained :
  (distance / new_time) = required_speed :=
sorry

end NUMINAMATH_GPT_speed_maintained_l487_48724


namespace NUMINAMATH_GPT_range_a_implies_not_purely_imaginary_l487_48758

def is_not_purely_imaginary (z : ℂ) : Prop :=
  z.re ≠ 0

theorem range_a_implies_not_purely_imaginary (a : ℝ) :
  ¬ is_not_purely_imaginary ⟨a^2 - a - 2, abs (a - 1) - 1⟩ ↔ a ≠ -1 :=
by
  sorry

end NUMINAMATH_GPT_range_a_implies_not_purely_imaginary_l487_48758


namespace NUMINAMATH_GPT_area_of_WIN_sector_l487_48706

theorem area_of_WIN_sector (r : ℝ) (p : ℝ) (A_circ : ℝ) (A_WIN : ℝ) 
    (h_r : r = 15) 
    (h_p : p = 1 / 3) 
    (h_A_circ : A_circ = π * r^2) 
    (h_A_WIN : A_WIN = p * A_circ) :
    A_WIN = 75 * π := 
sorry

end NUMINAMATH_GPT_area_of_WIN_sector_l487_48706


namespace NUMINAMATH_GPT_triangle_ABCD_lengths_l487_48745

theorem triangle_ABCD_lengths (AB BC CA : ℝ) (h_AB : AB = 20) (h_BC : BC = 40) (h_CA : CA = 49) :
  ∃ DA DC : ℝ, DA = 27.88 ∧ DC = 47.88 ∧
  (AB + DC = BC + DA) ∧ 
  (((AB^2 + BC^2 - CA^2) / (2 * AB * BC)) + ((DC^2 + DA^2 - CA^2) / (2 * DC * DA)) = 0) :=
sorry

end NUMINAMATH_GPT_triangle_ABCD_lengths_l487_48745


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l487_48722

theorem problem1 : 6 + (-8) - (-5) = 3 := by
  sorry

theorem problem2 : (5 + 3/5) + (-(5 + 2/3)) + (4 + 2/5) + (-1/3) = 4 := by
  sorry

theorem problem3 : ((-1/2) + 1/6 - 1/4) * 12 = -7 := by
  sorry

theorem problem4 : -1^2022 + 27 * (-1/3)^2 - |(-5)| = -3 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l487_48722


namespace NUMINAMATH_GPT_max_difference_of_mean_505_l487_48770

theorem max_difference_of_mean_505 (x y : ℕ) (h1 : 100 ≤ x ∧ x ≤ 999) (h2 : 100 ≤ y ∧ y ≤ 999) (h3 : (x + y) / 2 = 505) : 
  x - y ≤ 810 :=
sorry

end NUMINAMATH_GPT_max_difference_of_mean_505_l487_48770


namespace NUMINAMATH_GPT_garden_length_l487_48784

theorem garden_length (P b l : ℕ) (h1 : P = 500) (h2 : b = 100) : l = 150 :=
by
  sorry

end NUMINAMATH_GPT_garden_length_l487_48784


namespace NUMINAMATH_GPT_total_cars_l487_48777

-- Definitions of the conditions
def cathy_cars : Nat := 5

def carol_cars : Nat := 2 * cathy_cars

def susan_cars : Nat := carol_cars - 2

def lindsey_cars : Nat := cathy_cars + 4

-- The theorem statement (problem)
theorem total_cars : cathy_cars + carol_cars + susan_cars + lindsey_cars = 32 :=
by
  -- sorry is added to skip the proof
  sorry

end NUMINAMATH_GPT_total_cars_l487_48777


namespace NUMINAMATH_GPT_train_speed_is_18_kmh_l487_48744

noncomputable def speed_of_train (length_of_bridge length_of_train time : ℝ) : ℝ :=
  (length_of_bridge + length_of_train) / time * 3.6

theorem train_speed_is_18_kmh
  (length_of_bridge : ℝ)
  (length_of_train : ℝ)
  (time : ℝ)
  (h1 : length_of_bridge = 200)
  (h2 : length_of_train = 100)
  (h3 : time = 60) :
  speed_of_train length_of_bridge length_of_train time = 18 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_is_18_kmh_l487_48744


namespace NUMINAMATH_GPT_root_of_quadratic_l487_48780

theorem root_of_quadratic (a : ℝ) (ha : a ≠ 1) (hroot : (a-1) * 1^2 - a * 1 + a^2 = 0) : a = -1 := by
  sorry

end NUMINAMATH_GPT_root_of_quadratic_l487_48780


namespace NUMINAMATH_GPT_total_profit_is_35000_l487_48727

open Real

-- Define the subscriptions of A, B, and C
def subscriptions (A B C : ℝ) : Prop :=
  A + B + C = 50000 ∧
  A = B + 4000 ∧
  B = C + 5000

-- Define the profit distribution and the condition for C's received profit
def profit (total_profit : ℝ) (A B C : ℝ) (C_profit : ℝ) : Prop :=
  C_profit / total_profit = C / (A + B + C) ∧
  C_profit = 8400

-- Lean 4 statement to prove total profit
theorem total_profit_is_35000 :
  ∃ A B C total_profit, subscriptions A B C ∧ profit total_profit A B C 8400 ∧ total_profit = 35000 :=
by
  sorry

end NUMINAMATH_GPT_total_profit_is_35000_l487_48727


namespace NUMINAMATH_GPT_smallest_palindrome_div_3_5_l487_48782

theorem smallest_palindrome_div_3_5 : ∃ n : ℕ, n = 50205 ∧ 
  (∃ a b c : ℕ, n = 5 * 10^4 + a * 10^3 + b * 10^2 + a * 10 + 5) ∧ 
  n % 5 = 0 ∧ 
  n % 3 = 0 ∧ 
  n ≥ 10000 ∧ 
  n < 100000 :=
by
  sorry

end NUMINAMATH_GPT_smallest_palindrome_div_3_5_l487_48782


namespace NUMINAMATH_GPT_evaluate_expression_l487_48715

theorem evaluate_expression : 10 * 0.2 * 5 * 0.1 + 5 = 6 :=
by
  -- transformed step-by-step mathematical proof goes here
  sorry

end NUMINAMATH_GPT_evaluate_expression_l487_48715


namespace NUMINAMATH_GPT_triangle_area_PQR_l487_48762

section TriangleArea

variables {a b c d : ℝ}
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
variables (hOppositeSides : (0 - c) * b - (a - 0) * d < 0)

theorem triangle_area_PQR :
  let P := (0, a)
  let Q := (b, 0)
  let R := (c, d)
  let area := (1 / 2) * (a * c + b * d - a * b)
  area = (1 / 2) * (a * c + b * d - a * b) := 
by
  sorry

end TriangleArea

end NUMINAMATH_GPT_triangle_area_PQR_l487_48762


namespace NUMINAMATH_GPT_screws_per_pile_l487_48766

-- Definitions based on the given conditions
def initial_screws : ℕ := 8
def multiplier : ℕ := 2
def sections : ℕ := 4

-- Derived values based on the conditions
def additional_screws : ℕ := initial_screws * multiplier
def total_screws : ℕ := initial_screws + additional_screws

-- Proposition statement
theorem screws_per_pile : total_screws / sections = 6 := by
  sorry

end NUMINAMATH_GPT_screws_per_pile_l487_48766


namespace NUMINAMATH_GPT_mail_handling_in_six_months_l487_48792

theorem mail_handling_in_six_months (daily_letters daily_packages days_per_month months : ℕ) :
  daily_letters = 60 →
  daily_packages = 20 →
  days_per_month = 30 →
  months = 6 →
  (daily_letters + daily_packages) * days_per_month * months = 14400 :=
by
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_mail_handling_in_six_months_l487_48792


namespace NUMINAMATH_GPT_reflect_and_shift_l487_48726

def f : ℝ → ℝ := sorry  -- Assume f is some function from ℝ to ℝ

def h (f : ℝ → ℝ) (x : ℝ) : ℝ := f (6 - x)

theorem reflect_and_shift (f : ℝ → ℝ) (x : ℝ) : h f x = f (6 - x) :=
by
  -- provide the proof here
  sorry

end NUMINAMATH_GPT_reflect_and_shift_l487_48726


namespace NUMINAMATH_GPT_minimum_value_fraction_l487_48797

theorem minimum_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 1) : 
  (2 / x + 1 / y) >= 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_minimum_value_fraction_l487_48797


namespace NUMINAMATH_GPT_add_one_five_times_l487_48790

theorem add_one_five_times (m n : ℕ) (h : n = m + 5) : n - (m + 1) = 4 :=
by
  sorry

end NUMINAMATH_GPT_add_one_five_times_l487_48790


namespace NUMINAMATH_GPT_x_squared_plus_y_squared_value_l487_48711

theorem x_squared_plus_y_squared_value (x y : ℝ) (h : (x^2 + y^2 + 1) * (x^2 + y^2 + 2) = 6) : x^2 + y^2 = 1 :=
by
  sorry

end NUMINAMATH_GPT_x_squared_plus_y_squared_value_l487_48711


namespace NUMINAMATH_GPT_find_common_difference_max_sum_first_n_terms_max_n_Sn_positive_l487_48754

-- Define the arithmetic sequence
def a (n : ℕ) (d : ℤ) := 23 + n * d

-- Define the sum of the first n terms of the sequence
def S (n : ℕ) (d : ℤ) := n * 23 + (n * (n - 1) / 2) * d

-- Prove the common difference is -4
theorem find_common_difference (d : ℤ) :
  a 5 d > 0 ∧ a 6 d < 0 → d = -4 := sorry

-- Prove the maximum value of the sum S_n of the first n terms
theorem max_sum_first_n_terms (S_n : ℕ) :
  S 6 -4 = 78 := sorry

-- Prove the maximum value of n such that S_n > 0
theorem max_n_Sn_positive (n : ℕ) :
  S n -4 > 0 → n ≤ 12 := sorry

end NUMINAMATH_GPT_find_common_difference_max_sum_first_n_terms_max_n_Sn_positive_l487_48754


namespace NUMINAMATH_GPT_michael_truck_meet_once_l487_48708

-- Michael's walking speed.
def michael_speed := 4 -- feet per second

-- Distance between trash pails.
def pail_distance := 100 -- feet

-- Truck's speed.
def truck_speed := 8 -- feet per second

-- Time truck stops at each pail.
def truck_stop_time := 20 -- seconds

-- Prove how many times Michael and the truck will meet given the initial condition.
theorem michael_truck_meet_once :
  ∃ n : ℕ, michael_truck_meet_count == 1 :=
sorry

end NUMINAMATH_GPT_michael_truck_meet_once_l487_48708


namespace NUMINAMATH_GPT_fraction_numerator_l487_48712

theorem fraction_numerator (x : ℤ) (h₁ : 2 * x + 11 ≠ 0) (h₂ : (x : ℚ) / (2 * x + 11) = 3 / 4) : x = -33 / 2 :=
by
  sorry

end NUMINAMATH_GPT_fraction_numerator_l487_48712


namespace NUMINAMATH_GPT_cyclic_sum_inequality_l487_48756

theorem cyclic_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^3 + 3 * b^3) / (5 * a + b) + (b^3 + 3 * c^3) / (5 * b + c) + (c^3 + 3 * a^3) / (5 * c + a) ≥ (2 / 3) * (a^2 + b^2 + c^2) :=
  sorry

end NUMINAMATH_GPT_cyclic_sum_inequality_l487_48756


namespace NUMINAMATH_GPT_probability_individual_selected_l487_48799

theorem probability_individual_selected :
  ∀ (N M : ℕ) (m : ℕ), N = 100 → M = 5 → (m < N) →
  (probability_of_selecting_m : ℝ) =
  (1 / N * M) :=
by
  intros N M m hN hM hm
  sorry

end NUMINAMATH_GPT_probability_individual_selected_l487_48799


namespace NUMINAMATH_GPT_pencils_calculation_l487_48755

def num_pencil_boxes : ℝ := 4.0
def pencils_per_box : ℝ := 648.0
def total_pencils : ℝ := 2592.0

theorem pencils_calculation : (num_pencil_boxes * pencils_per_box) = total_pencils := 
by
  sorry

end NUMINAMATH_GPT_pencils_calculation_l487_48755


namespace NUMINAMATH_GPT_find_initial_lion_population_l487_48787

-- Define the conditions as integers
def lion_cubs_per_month : ℕ := 5
def lions_die_per_month : ℕ := 1
def total_lions_after_one_year : ℕ := 148

-- Define a formula for calculating the initial number of lions
def initial_number_of_lions (net_increase : ℕ) (final_count : ℕ) (months : ℕ) : ℕ :=
  final_count - (net_increase * months)

-- Main theorem statement
theorem find_initial_lion_population : initial_number_of_lions (lion_cubs_per_month - lions_die_per_month) total_lions_after_one_year 12 = 100 :=
  sorry

end NUMINAMATH_GPT_find_initial_lion_population_l487_48787


namespace NUMINAMATH_GPT_solve_for_z_l487_48785

theorem solve_for_z (z : ℂ) (h : z * (1 - I) = 2 + I) : z = (1 / 2) + (3 / 2) * I :=
  sorry

end NUMINAMATH_GPT_solve_for_z_l487_48785


namespace NUMINAMATH_GPT_num_disks_to_sell_l487_48775

-- Define the buying and selling price conditions.
def cost_per_disk := 6 / 5
def sell_per_disk := 7 / 4

-- Define the desired profit
def desired_profit := 120

-- Calculate the profit per disk.
def profit_per_disk := sell_per_disk - cost_per_disk

-- Statement of the problem: Determine number of disks to sell.
theorem num_disks_to_sell
  (h₁ : cost_per_disk = 6 / 5)
  (h₂ : sell_per_disk = 7 / 4)
  (h₃ : desired_profit = 120)
  (h₄ : profit_per_disk = 7 / 4 - 6 / 5) :
  ∃ disks_to_sell : ℕ, disks_to_sell = 219 ∧ 
  disks_to_sell * profit_per_disk ≥ 120 ∧
  (disks_to_sell - 1) * profit_per_disk < 120 :=
sorry

end NUMINAMATH_GPT_num_disks_to_sell_l487_48775


namespace NUMINAMATH_GPT_c_share_l487_48719

theorem c_share (A B C D : ℝ) 
    (h1 : A = 1/2 * B) 
    (h2 : B = 1/2 * C) 
    (h3 : D = 1/4 * 392) 
    (h4 : A + B + C + D = 392) : 
    C = 168 := 
by 
    sorry

end NUMINAMATH_GPT_c_share_l487_48719


namespace NUMINAMATH_GPT_winning_votes_l487_48751

theorem winning_votes (V : ℝ) (h1 : 0.62 * V - 0.38 * V = 312) : 0.62 * V = 806 :=
by
  -- The proof should be written here, but we'll skip it as per the instructions.
  sorry

end NUMINAMATH_GPT_winning_votes_l487_48751


namespace NUMINAMATH_GPT_find_a_b_l487_48752

def curve (x : ℝ) (a b : ℝ) : ℝ := x^2 + a * x + b
def tangent_line (x y : ℝ) : Prop := x - y + 1 = 0

theorem find_a_b :
  ∀ (a b : ℝ),
  (∀ x, (curve x a b) = x^2 + a * x + b) →
  (tangent_line 0 (curve 0 a b)) →
  (tangent_line x y → y = x + 1) →
  (tangent_line x y → ∃ m c, y = m * x + c ∧ m = 1 ∧ c = 1) →
  (∃ a b : ℝ, a = 1 ∧ b = 1) :=
by
  intros a b h_curve h_tangent_line h_tangent_line_form h_tangent_line_eq
  sorry

end NUMINAMATH_GPT_find_a_b_l487_48752


namespace NUMINAMATH_GPT_candy_sampling_percentage_l487_48796

theorem candy_sampling_percentage (total_percentage caught_percentage not_caught_percentage : ℝ) 
  (h1 : caught_percentage = 22 / 100) 
  (h2 : total_percentage = 24.444444444444443 / 100) 
  (h3 : not_caught_percentage = 2.444444444444443 / 100) :
  total_percentage = caught_percentage + not_caught_percentage :=
by
  sorry

end NUMINAMATH_GPT_candy_sampling_percentage_l487_48796


namespace NUMINAMATH_GPT_average_sale_six_months_l487_48750

theorem average_sale_six_months :
  let sale1 := 2500
  let sale2 := 6500
  let sale3 := 9855
  let sale4 := 7230
  let sale5 := 7000
  let sale6 := 11915
  let total_sales := sale1 + sale2 + sale3 + sale4 + sale5 + sale6
  let num_months := 6
  (total_sales / num_months) = 7500 :=
by
  sorry

end NUMINAMATH_GPT_average_sale_six_months_l487_48750


namespace NUMINAMATH_GPT_parallelogram_area_l487_48721

theorem parallelogram_area (a b : ℝ) (theta : ℝ)
  (h1 : a = 10) (h2 : b = 20) (h3 : theta = 150) : a * b * Real.sin (theta * Real.pi / 180) = 100 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_parallelogram_area_l487_48721


namespace NUMINAMATH_GPT_area_square_B_l487_48794

theorem area_square_B (a b : ℝ) (h1 : a^2 = 25) (h2 : abs (a - b) = 4) : b^2 = 81 :=
by
  sorry

end NUMINAMATH_GPT_area_square_B_l487_48794


namespace NUMINAMATH_GPT_quadratic_perfect_square_form_l487_48703

def quadratic_is_perfect_square (a b c : ℤ) : Prop :=
  ∀ x : ℤ, ∃ k : ℤ, a * x^2 + b * x + c = k^2

theorem quadratic_perfect_square_form (a b c : ℤ) (h : quadratic_is_perfect_square a b c) :
  ∃ d e : ℤ, ∀ x : ℤ, a * x^2 + b * x + c = (d * x + e)^2 :=
  sorry

end NUMINAMATH_GPT_quadratic_perfect_square_form_l487_48703


namespace NUMINAMATH_GPT_product_of_three_numbers_l487_48743

theorem product_of_three_numbers : 
  ∃ x y z : ℚ, x + y + z = 30 ∧ x = 3 * (y + z) ∧ y = 6 * z ∧ x * y * z = 23625 / 686 :=
by
  sorry

end NUMINAMATH_GPT_product_of_three_numbers_l487_48743


namespace NUMINAMATH_GPT_range_of_a_l487_48742

open Real

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, a ≤ abs (x - 5) + abs (x - 3)) → a ≤ 2 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l487_48742


namespace NUMINAMATH_GPT_margie_driving_distance_l487_48763

-- Define the constants given in the conditions
def mileage_per_gallon : ℝ := 40
def cost_per_gallon : ℝ := 5
def total_money : ℝ := 25

-- Define the expected result/answer
def expected_miles : ℝ := 200

-- The theorem that needs to be proved
theorem margie_driving_distance :
  (total_money / cost_per_gallon) * mileage_per_gallon = expected_miles :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_margie_driving_distance_l487_48763


namespace NUMINAMATH_GPT_bubble_bath_per_guest_l487_48746

def rooms_couple : ℕ := 13
def rooms_single : ℕ := 14
def total_bubble_bath : ℕ := 400

theorem bubble_bath_per_guest :
  (total_bubble_bath / (rooms_couple * 2 + rooms_single)) = 10 :=
by
  sorry

end NUMINAMATH_GPT_bubble_bath_per_guest_l487_48746


namespace NUMINAMATH_GPT_unknown_number_is_five_l487_48701

theorem unknown_number_is_five (x : ℕ) (h : 64 + x * 12 / (180 / 3) = 65) : x = 5 := 
by 
  sorry

end NUMINAMATH_GPT_unknown_number_is_five_l487_48701


namespace NUMINAMATH_GPT_inequality_proof_l487_48749

variable (f : ℕ → ℕ → ℕ)

theorem inequality_proof :
  f 1 6 * f 2 5 * f 3 4 + f 1 5 * f 2 4 * f 3 6 + f 1 4 * f 2 6 * f 3 5 ≥
  f 1 6 * f 2 4 * f 3 5 + f 1 5 * f 2 6 * f 3 4 + f 1 4 * f 2 5 * f 3 6 :=
by sorry

end NUMINAMATH_GPT_inequality_proof_l487_48749


namespace NUMINAMATH_GPT_lines_intersection_example_l487_48769

theorem lines_intersection_example (m b : ℝ) 
  (h1 : 8 = m * 4 + 2) 
  (h2 : 8 = 4 * 4 + b) : 
  b + m = -13 / 2 := 
by
  sorry

end NUMINAMATH_GPT_lines_intersection_example_l487_48769


namespace NUMINAMATH_GPT_sum_of_digits_divisible_by_7_l487_48710

theorem sum_of_digits_divisible_by_7
  (a b : ℕ)
  (h_three_digit : 100 * a + 11 * b ≥ 100 ∧ 100 * a + 11 * b < 1000)
  (h_last_two_digits_equal : true)
  (h_divisible_by_7 : (100 * a + 11 * b) % 7 = 0) :
  (a + 2 * b) % 7 = 0 :=
sorry

end NUMINAMATH_GPT_sum_of_digits_divisible_by_7_l487_48710


namespace NUMINAMATH_GPT_nat_numbers_in_segment_l487_48736

theorem nat_numbers_in_segment (a : ℕ → ℕ) (blue_index red_index : Set ℕ)
  (cond1 : ∀ i ∈ blue_index, i ≤ 200 → a (i - 1) = i)
  (cond2 : ∀ i ∈ red_index, i ≤ 200 → a (i - 1) = 201 - i) :
    ∀ i, 1 ≤ i ∧ i ≤ 100 → ∃ j, j < 100 ∧ a j = i := 
by
  sorry

end NUMINAMATH_GPT_nat_numbers_in_segment_l487_48736


namespace NUMINAMATH_GPT_problem_statement_l487_48720

noncomputable def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n : ℕ, ∃ d : ℤ, a (n + 1) = a n + d

noncomputable def given_conditions (a : ℕ → ℤ) : Prop :=
a 2 = 2 ∧ a 3 = 4

theorem problem_statement (a : ℕ → ℤ) (h1 : given_conditions a) (h2 : arithmetic_sequence a) :
  a 10 = 18 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l487_48720


namespace NUMINAMATH_GPT_largest_integral_x_l487_48789

theorem largest_integral_x (x y : ℤ) (h1 : (1 : ℚ)/4 < x/7) (h2 : x/7 < (2 : ℚ)/3) (h3 : x + y = 10) : x = 4 :=
by
  sorry

end NUMINAMATH_GPT_largest_integral_x_l487_48789


namespace NUMINAMATH_GPT_a_equals_bc_l487_48739

theorem a_equals_bc (f g : ℝ → ℝ) (a b c : ℝ) :
  (∀ x y : ℝ, f x * g y = a * x * y + b * x + c * y + 1) → a = b * c :=
sorry

end NUMINAMATH_GPT_a_equals_bc_l487_48739


namespace NUMINAMATH_GPT_namjoonKoreanScore_l487_48788

variables (mathScore englishScore : ℝ) (averageScore : ℝ := 95) (koreanScore : ℝ)

def namjoonMathScore : Prop := mathScore = 100
def namjoonEnglishScore : Prop := englishScore = 95
def namjoonAverage : Prop := (koreanScore + mathScore + englishScore) / 3 = averageScore

theorem namjoonKoreanScore
  (H1 : namjoonMathScore 100)
  (H2 : namjoonEnglishScore 95)
  (H3 : namjoonAverage koreanScore 100 95 95) :
  koreanScore = 90 :=
by
  sorry

end NUMINAMATH_GPT_namjoonKoreanScore_l487_48788


namespace NUMINAMATH_GPT_graph_of_equation_pair_of_lines_l487_48765

theorem graph_of_equation_pair_of_lines (x y : ℝ) : x^2 - 9 * y^2 = 0 ↔ (x = 3 * y ∨ x = -3 * y) :=
by
  sorry

end NUMINAMATH_GPT_graph_of_equation_pair_of_lines_l487_48765


namespace NUMINAMATH_GPT_f_value_at_5_l487_48768

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 3 / 2 then 2 * x^2 else sorry

theorem f_value_at_5 (f : ℝ → ℝ)
  (h_even : even_function f)
  (h_periodic : periodic_function f 3)
  (h_definition : ∀ x, 0 ≤ x ∧ x ≤ 3 / 2 → f x = 2 * x^2) :
  f 5 = 2 :=
by
  sorry

end NUMINAMATH_GPT_f_value_at_5_l487_48768


namespace NUMINAMATH_GPT_total_distance_thrown_l487_48779

theorem total_distance_thrown (D : ℝ) (total_distance : ℝ) 
  (h1 : total_distance = 20 * D + 60 * D) : 
  total_distance = 1600 := 
by
  sorry

end NUMINAMATH_GPT_total_distance_thrown_l487_48779


namespace NUMINAMATH_GPT_problem1_problem2_l487_48760

variable {a b c : ℝ}
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

-- Statement for the first proof
theorem problem1 (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  bc / a + ca / b + ab / c ≥ a + b + c :=
sorry

-- Statement for the second proof
theorem problem2 (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 1) :
  (1 - a) / a + (1 - b) / b + (1 - c) / c ≥ 6 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l487_48760


namespace NUMINAMATH_GPT_base_conversion_and_operations_l487_48713

-- Definitions to convert numbers from bases 7, 5, and 6 to base 10
def base7_to_nat (n : ℕ) : ℕ := 
  8 * 7^0 + 6 * 7^1 + 4 * 7^2 + 2 * 7^3

def base5_to_nat (n : ℕ) : ℕ := 
  1 * 5^0 + 2 * 5^1 + 1 * 5^2

def base6_to_nat (n : ℕ) : ℕ := 
  1 * 6^0 + 5 * 6^1 + 4 * 6^2 + 3 * 6^3

def base7_to_nat2 (n : ℕ) : ℕ := 
  1 * 7^0 + 9 * 7^1 + 8 * 7^2 + 7 * 7^3

-- Problem statement: Perform the arithmetical operations
theorem base_conversion_and_operations : 
  (base7_to_nat 2468 / base5_to_nat 121) - base6_to_nat 3451 + base7_to_nat2 7891 = 2059 := 
by
  sorry

end NUMINAMATH_GPT_base_conversion_and_operations_l487_48713


namespace NUMINAMATH_GPT_night_rides_total_l487_48725

-- Definitions corresponding to the conditions in the problem
def total_ferris_wheel_rides : Nat := 13
def total_roller_coaster_rides : Nat := 9
def ferris_wheel_day_rides : Nat := 7
def roller_coaster_day_rides : Nat := 4

-- The total night rides proof problem
theorem night_rides_total :
  let ferris_wheel_night_rides := total_ferris_wheel_rides - ferris_wheel_day_rides
  let roller_coaster_night_rides := total_roller_coaster_rides - roller_coaster_day_rides
  ferris_wheel_night_rides + roller_coaster_night_rides = 11 :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_night_rides_total_l487_48725


namespace NUMINAMATH_GPT_hens_count_l487_48729

theorem hens_count (H C : ℕ) (h1 : H + C = 48) (h2 : 2 * H + 4 * C = 140) : H = 26 := by
  sorry

end NUMINAMATH_GPT_hens_count_l487_48729


namespace NUMINAMATH_GPT_rose_needs_more_money_l487_48771

def cost_of_paintbrush : ℝ := 2.4
def cost_of_paints : ℝ := 9.2
def cost_of_easel : ℝ := 6.5
def amount_rose_has : ℝ := 7.1
def total_cost : ℝ := cost_of_paintbrush + cost_of_paints + cost_of_easel

theorem rose_needs_more_money : (total_cost - amount_rose_has) = 11 := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_rose_needs_more_money_l487_48771


namespace NUMINAMATH_GPT_desired_percentage_of_alcohol_l487_48757

theorem desired_percentage_of_alcohol 
  (original_volume : ℝ)
  (original_percentage : ℝ)
  (added_volume : ℝ)
  (added_percentage : ℝ)
  (final_percentage : ℝ) :
  original_volume = 6 →
  original_percentage = 0.35 →
  added_volume = 1.8 →
  added_percentage = 1.0 →
  final_percentage = 50 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_desired_percentage_of_alcohol_l487_48757


namespace NUMINAMATH_GPT_chess_tournament_l487_48783

theorem chess_tournament (n : ℕ) (h : (n * (n - 1)) / 2 - ((n - 3) * (n - 4)) / 2 = 130) : n = 19 :=
sorry

end NUMINAMATH_GPT_chess_tournament_l487_48783


namespace NUMINAMATH_GPT_julia_song_download_l487_48759

theorem julia_song_download : 
  let internet_speed := 20 -- in MBps
  let half_hour_in_minutes := 30
  let size_per_song := 5 -- in MB
  (internet_speed * 60 * half_hour_in_minutes) / size_per_song = 7200 :=
by
  sorry

end NUMINAMATH_GPT_julia_song_download_l487_48759


namespace NUMINAMATH_GPT_stratified_sampling_l487_48731

-- We are defining the data given in the problem
def numStudents : ℕ := 50
def numFemales : ℕ := 20
def sampledFemales : ℕ := 4
def genderRatio := (numFemales : ℚ) / (numStudents : ℚ)

-- The theorem stating the given problem and its conclusion
theorem stratified_sampling : ∀ (n : ℕ), (sampledFemales : ℚ) / (n : ℚ) = genderRatio → n = 10 :=
by
  intro n
  intro h
  sorry

end NUMINAMATH_GPT_stratified_sampling_l487_48731


namespace NUMINAMATH_GPT_tangent_line_circle_l487_48709

theorem tangent_line_circle (k : ℝ) (h1 : k = Real.sqrt 3) (h2 : ∃ (x y : ℝ), y = k * x + 2 ∧ x^2 + y^2 = 1) :
  (k = Real.sqrt 3 → (∃ (x y : ℝ), y = k * x + 2 ∧ x^2 + y^2 = 1)) ∧ (¬ (∀ (k : ℝ), (∃ (x y : ℝ), y = k * x + 2 ∧ x^2 + y^2 = 1) → k = Real.sqrt 3)) :=
  sorry

end NUMINAMATH_GPT_tangent_line_circle_l487_48709


namespace NUMINAMATH_GPT_pump_X_time_l487_48773

-- Definitions for the problem conditions.
variables (W : ℝ) (T_x : ℝ) (R_x R_y : ℝ)

-- Condition 1: Rate of pump X
def pump_X_rate := R_x = (W / 2) / T_x

-- Condition 2: Rate of pump Y
def pump_Y_rate := R_y = W / 18

-- Condition 3: Combined rate when both pumps work together for 3 hours to pump the remaining water
def combined_rate := (R_x + R_y) = (W / 2) / 3

-- The statement to prove
theorem pump_X_time : 
  pump_X_rate W T_x R_x →
  pump_Y_rate W R_y →
  combined_rate W R_x R_y →
  T_x = 9 :=
sorry

end NUMINAMATH_GPT_pump_X_time_l487_48773


namespace NUMINAMATH_GPT_total_payment_is_53_l487_48733

-- Conditions
def bobBill : ℝ := 30
def kateBill : ℝ := 25
def bobDiscountRate : ℝ := 0.05
def kateDiscountRate : ℝ := 0.02

-- Calculations
def bobDiscount := bobBill * bobDiscountRate
def kateDiscount := kateBill * kateDiscountRate
def bobPayment := bobBill - bobDiscount
def katePayment := kateBill - kateDiscount

-- Goal
def totalPayment := bobPayment + katePayment

-- Theorem statement
theorem total_payment_is_53 : totalPayment = 53 := by
  sorry

end NUMINAMATH_GPT_total_payment_is_53_l487_48733


namespace NUMINAMATH_GPT_expression_value_l487_48738

theorem expression_value : (5 - 2) / (2 + 1) = 1 := by
  sorry

end NUMINAMATH_GPT_expression_value_l487_48738


namespace NUMINAMATH_GPT_triangle_area_l487_48748

/-- 
  Given:
  - A smaller rectangle OABD with OA = 4 cm, AB = 4 cm
  - A larger rectangle ABEC with AB = 12 cm, BC = 12 cm
  - Point O at (0,0)
  - Point A at (4,0)
  - Point B at (16,0)
  - Point C at (16,12)
  - Point D at (4,12)
  - Point E is on the line from A to C
  
  Prove the area of the triangle CDE is 54 cm²
-/
theorem triangle_area (OA AB OB DE DC : ℕ) : 
  OA = 4 ∧ AB = 4 ∧ OB = 16 ∧ DE = 12 - 3 ∧ DC = 12 → (1 / 2) * DE * DC = 54 := by 
  intros h
  sorry

end NUMINAMATH_GPT_triangle_area_l487_48748


namespace NUMINAMATH_GPT_slope_symmetric_line_l487_48767

  theorem slope_symmetric_line {l1 l2 : ℝ → ℝ} 
     (hl1 : ∀ x, l1 x = 2 * x + 3)
     (hl2_sym : ∀ x, l2 x = 2 * x + 3 -> l2 (-x) = -2 * x - 3) :
     ∀ x, l2 x = -2 * x + 3 :=
  sorry
  
end NUMINAMATH_GPT_slope_symmetric_line_l487_48767


namespace NUMINAMATH_GPT_matchsticks_distribution_l487_48786

open Nat

theorem matchsticks_distribution
  (length_sticks : ℕ)
  (width_sticks : ℕ)
  (length_condition : length_sticks = 60)
  (width_condition : width_sticks = 10)
  (total_sticks : ℕ)
  (total_sticks_condition : total_sticks = 60 * 11 + 10 * 61)
  (children_count : ℕ)
  (children_condition : children_count > 100)
  (division_condition : total_sticks % children_count = 0) :
  children_count = 127 := by
  sorry

end NUMINAMATH_GPT_matchsticks_distribution_l487_48786


namespace NUMINAMATH_GPT_initial_holes_count_additional_holes_needed_l487_48761

-- Defining the conditions as variables
def circumference : ℕ := 400
def initial_interval : ℕ := 50
def new_interval : ℕ := 40

-- Defining the problems

-- Problem 1: Calculate the number of holes for the initial interval
theorem initial_holes_count (circumference : ℕ) (initial_interval : ℕ) : 
  circumference % initial_interval = 0 → 
  circumference / initial_interval = 8 := 
sorry

-- Problem 2: Calculate the additional holes needed
theorem additional_holes_needed (circumference : ℕ) (initial_interval : ℕ) 
  (new_interval : ℕ) (lcm_interval : ℕ) :
  lcm new_interval initial_interval = lcm_interval →
  circumference % new_interval = 0 →
  circumference / new_interval - 
  (circumference / lcm_interval) = 8 :=
sorry

end NUMINAMATH_GPT_initial_holes_count_additional_holes_needed_l487_48761


namespace NUMINAMATH_GPT_find_x_from_percentage_l487_48795

theorem find_x_from_percentage (x : ℝ) (h : 0.2 * 30 = 0.25 * x + 2) : x = 16 :=
sorry

end NUMINAMATH_GPT_find_x_from_percentage_l487_48795


namespace NUMINAMATH_GPT_maximize_profit_l487_48735

noncomputable section

def price (x : ℕ) : ℝ :=
  if 0 < x ∧ x ≤ 100 then 60
  else if 100 < x ∧ x ≤ 600 then 62 - 0.02 * x
  else 0

def profit (x : ℕ) : ℝ :=
  (price x - 40) * x

theorem maximize_profit :
  ∃ x : ℕ, (1 ≤ x ∧ x ≤ 600) ∧ (∀ y : ℕ, (1 ≤ y ∧ y ≤ 600 → profit y ≤ profit x)) ∧ profit x = 6050 :=
by sorry

end NUMINAMATH_GPT_maximize_profit_l487_48735


namespace NUMINAMATH_GPT_no_solution_exists_l487_48702

theorem no_solution_exists : ¬ ∃ (x : ℕ), (42 + x = 3 * (8 + x) ∧ 42 + x = 2 * (10 + x)) :=
by
  sorry

end NUMINAMATH_GPT_no_solution_exists_l487_48702


namespace NUMINAMATH_GPT_range_of_a_l487_48791

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x + 9| > a) → a < 8 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l487_48791


namespace NUMINAMATH_GPT_perpendicular_condition_parallel_condition_parallel_opposite_direction_l487_48723

variables (a b : ℝ × ℝ) (k : ℝ)

-- Define the vectors
def vec_a : ℝ × ℝ := (1, 2)
def vec_b : ℝ × ℝ := (-3, 2)

-- Define the given expressions
def expression1 (k : ℝ) : ℝ × ℝ := (k * vec_a.1 + vec_b.1, k * vec_a.2 + vec_b.2)
def expression2 : ℝ × ℝ := (vec_a.1 - 3 * vec_b.1, vec_a.2 - 3 * vec_b.2)

-- Dot product function
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Perpendicular condition
theorem perpendicular_condition : (k : ℝ) → dot_product (expression1 k) expression2 = 0 → k = 19 :=
by sorry

-- Parallel and opposite condition
theorem parallel_condition : (k : ℝ) → (∃ m : ℝ, expression1 k = m • expression2) → k = -1 / 3 :=
by sorry

noncomputable def m (k : ℝ) : ℝ × ℝ := 
  let ex1 := expression1 k
  let ex2 := expression2
  (ex2.1 / ex1.1, ex2.2 / ex1.2)

theorem parallel_opposite_direction : (k : ℝ) → expression1 k = -1 / 3 • expression2 → k = -1 / 3 :=
by sorry

end NUMINAMATH_GPT_perpendicular_condition_parallel_condition_parallel_opposite_direction_l487_48723


namespace NUMINAMATH_GPT_min_notebooks_needed_l487_48740

variable (cost_pen cost_notebook num_pens discount_threshold : ℕ)

theorem min_notebooks_needed (x : ℕ)
    (h1 : cost_pen = 10)
    (h2 : cost_notebook = 4)
    (h3 : num_pens = 3)
    (h4 : discount_threshold = 100)
    (h5 : num_pens * cost_pen + x * cost_notebook ≥ discount_threshold) :
    x ≥ 18 := 
sorry

end NUMINAMATH_GPT_min_notebooks_needed_l487_48740


namespace NUMINAMATH_GPT_total_points_correct_l487_48737

variable (H Q T : ℕ)

-- Given conditions
def hw_points : ℕ := 40
def quiz_points := hw_points + 5
def test_points := 4 * quiz_points

-- Question: Prove the total points assigned are 265
theorem total_points_correct :
  H = hw_points →
  Q = quiz_points →
  T = test_points →
  H + Q + T = 265 :=
by
  intros h_hw h_quiz h_test
  rw [h_hw, h_quiz, h_test]
  exact sorry

end NUMINAMATH_GPT_total_points_correct_l487_48737


namespace NUMINAMATH_GPT_parker_total_stamps_l487_48741

-- Definitions based on conditions
def original_stamps := 430
def addie_stamps := 1890
def addie_fraction := 3 / 7
def stamps_added_by_addie := addie_fraction * addie_stamps

-- Theorem statement to prove the final number of stamps
theorem parker_total_stamps : original_stamps + stamps_added_by_addie = 1240 :=
by
  -- definitions instantiated above
  sorry  -- proof required

end NUMINAMATH_GPT_parker_total_stamps_l487_48741


namespace NUMINAMATH_GPT_minhyuk_needs_slices_l487_48730

-- Definitions of Yeongchan and Minhyuk's apple division
def yeongchan_portion : ℚ := 1 / 3
def minhyuk_slices : ℚ := 1 / 12

-- Statement to prove
theorem minhyuk_needs_slices (x : ℕ) : yeongchan_portion = x * minhyuk_slices → x = 4 :=
by
  sorry

end NUMINAMATH_GPT_minhyuk_needs_slices_l487_48730


namespace NUMINAMATH_GPT_fraction_addition_l487_48707

theorem fraction_addition (x : ℝ) (h : x + 1 ≠ 0) : (x / (x + 1) + 1 / (x + 1) = 1) :=
sorry

end NUMINAMATH_GPT_fraction_addition_l487_48707


namespace NUMINAMATH_GPT_price_reduction_equation_l487_48793

theorem price_reduction_equation (x : ℝ) : 200 * (1 - x) ^ 2 = 162 :=
by
  sorry

end NUMINAMATH_GPT_price_reduction_equation_l487_48793


namespace NUMINAMATH_GPT_yoongi_has_smallest_points_l487_48776

def points_jungkook : ℕ := 6 + 3
def points_yoongi : ℕ := 4
def points_yuna : ℕ := 5

theorem yoongi_has_smallest_points : points_yoongi < points_jungkook ∧ points_yoongi < points_yuna :=
by
  sorry

end NUMINAMATH_GPT_yoongi_has_smallest_points_l487_48776


namespace NUMINAMATH_GPT_probability_one_white_one_black_two_touches_l487_48732

def probability_white_ball : ℚ := 7 / 10
def probability_black_ball : ℚ := 3 / 10

theorem probability_one_white_one_black_two_touches :
  (probability_white_ball * probability_black_ball) + (probability_black_ball * probability_white_ball) = (7 / 10) * (3 / 10) + (3 / 10) * (7 / 10) :=
by
  -- The proof is omitted here.
  sorry

end NUMINAMATH_GPT_probability_one_white_one_black_two_touches_l487_48732


namespace NUMINAMATH_GPT_intersection_with_complement_l487_48718

-- Definitions for the universal set and set A
def U : Set ℝ := Set.univ

def A : Set ℝ := { -1, 0, 1 }

-- Definition for set B using the given condition
def B : Set ℝ := { x : ℝ | (x - 2) / (x + 1) > 0 }

-- Definition for the complement of B
def B_complement : Set ℝ := { x : ℝ | -1 <= x ∧ x <= 0 }

-- Theorem stating the intersection of A and the complement of B equals {-1, 0, 1}
theorem intersection_with_complement : 
  A ∩ B_complement = { -1, 0, 1 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_with_complement_l487_48718


namespace NUMINAMATH_GPT_reduced_price_of_oil_l487_48728

theorem reduced_price_of_oil (P R : ℝ) (h1: R = 0.75 * P) (h2: 600 / (0.75 * P) = 600 / P + 5) :
  R = 30 :=
by
  sorry

end NUMINAMATH_GPT_reduced_price_of_oil_l487_48728


namespace NUMINAMATH_GPT_michael_twenty_dollar_bills_l487_48774

/--
Michael has $280 dollars and each bill is $20 dollars.
We need to prove that the number of $20 dollar bills Michael has is 14.
-/
theorem michael_twenty_dollar_bills (total_money : ℕ) (bill_denomination : ℕ) (number_of_bills : ℕ) :
  total_money = 280 →
  bill_denomination = 20 →
  number_of_bills = total_money / bill_denomination →
  number_of_bills = 14 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_michael_twenty_dollar_bills_l487_48774


namespace NUMINAMATH_GPT_purely_imaginary_z_value_l487_48716

theorem purely_imaginary_z_value (a : ℝ) (h : (a^2 - a - 2) = 0 ∧ (a + 1) ≠ 0) : a = 2 :=
sorry

end NUMINAMATH_GPT_purely_imaginary_z_value_l487_48716


namespace NUMINAMATH_GPT_equivalence_of_sum_cubed_expression_l487_48705

theorem equivalence_of_sum_cubed_expression (a b : ℝ) 
  (h₁ : a + b = 5) (h₂ : a * b = -14) : a^3 + a^2 * b + a * b^2 + b^3 = 265 :=
sorry

end NUMINAMATH_GPT_equivalence_of_sum_cubed_expression_l487_48705
