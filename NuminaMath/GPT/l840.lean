import Mathlib

namespace NUMINAMATH_GPT_smallest_three_digit_in_pascals_triangle_l840_84070

theorem smallest_three_digit_in_pascals_triangle : ∃ k n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ ∀ m, ((m <= n) ∧ (m >= 100)) → m ≥ n :=
by
  sorry

end NUMINAMATH_GPT_smallest_three_digit_in_pascals_triangle_l840_84070


namespace NUMINAMATH_GPT_work_completion_l840_84087

theorem work_completion (Rp Rq Dp W : ℕ) 
  (h1 : Rq = W / 12) 
  (h2 : W = 4*Rp + 6*(Rp + Rq)) 
  (h3 : Rp = W / Dp) 
  : Dp = 20 :=
by
  sorry

end NUMINAMATH_GPT_work_completion_l840_84087


namespace NUMINAMATH_GPT_sin_half_alpha_l840_84002

theorem sin_half_alpha (α : ℝ) (hα : 0 < α ∧ α < π / 2) (hcos : Real.cos α = (1 + Real.sqrt 5) / 4) : 
  Real.sin (α / 2) = (Real.sqrt 5 - 1) / 4 := 
by 
  sorry

end NUMINAMATH_GPT_sin_half_alpha_l840_84002


namespace NUMINAMATH_GPT_solve_for_x_l840_84039

theorem solve_for_x (x : ℝ) (h : 5 * x + 3 = 10 * x - 22) : x = 5 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l840_84039


namespace NUMINAMATH_GPT_election_winner_votes_l840_84028

theorem election_winner_votes :
  ∃ V W : ℝ, (V = (71.42857142857143 / 100) * V + 3000 + 5000) ∧
            (W = (71.42857142857143 / 100) * V) ∧
            W = 20000 := by
  sorry

end NUMINAMATH_GPT_election_winner_votes_l840_84028


namespace NUMINAMATH_GPT_smallest_number_divisible_by_15_and_36_l840_84038

theorem smallest_number_divisible_by_15_and_36 : 
  ∃ x, (∀ y, (y % 15 = 0 ∧ y % 36 = 0) → y ≥ x) ∧ x = 180 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_divisible_by_15_and_36_l840_84038


namespace NUMINAMATH_GPT_solution_l840_84099

-- Define the conditions
def equation (x : ℝ) : Prop :=
  (x / 15) = (15 / x)

theorem solution (x : ℝ) : equation x → x = 15 ∨ x = -15 :=
by
  intros h
  -- The proof would go here.
  sorry

end NUMINAMATH_GPT_solution_l840_84099


namespace NUMINAMATH_GPT_sqrt_sixteen_is_four_l840_84050

theorem sqrt_sixteen_is_four : Real.sqrt 16 = 4 := 
by 
  sorry

end NUMINAMATH_GPT_sqrt_sixteen_is_four_l840_84050


namespace NUMINAMATH_GPT_inequality_solution_l840_84013

theorem inequality_solution : { x : ℝ | (x - 1) / (x + 3) < 0 } = { x : ℝ | -3 < x ∧ x < 1 } :=
sorry

end NUMINAMATH_GPT_inequality_solution_l840_84013


namespace NUMINAMATH_GPT_tom_and_elizabeth_climb_ratio_l840_84005

theorem tom_and_elizabeth_climb_ratio :
  let elizabeth_time := 30
  let tom_time_hours := 2
  let tom_time_minutes := tom_time_hours * 60
  (tom_time_minutes / elizabeth_time) = 4 :=
by sorry

end NUMINAMATH_GPT_tom_and_elizabeth_climb_ratio_l840_84005


namespace NUMINAMATH_GPT_cupcake_cookie_price_ratio_l840_84052

theorem cupcake_cookie_price_ratio
  (c k : ℚ)
  (h1 : 5 * c + 3 * k = 23)
  (h2 : 4 * c + 4 * k = 21) :
  k / c = 13 / 29 :=
  sorry

end NUMINAMATH_GPT_cupcake_cookie_price_ratio_l840_84052


namespace NUMINAMATH_GPT_exists_c_d_in_set_of_13_reals_l840_84021

theorem exists_c_d_in_set_of_13_reals (a : Fin 13 → ℝ) :
  ∃ (c d : ℝ), c ∈ Set.range a ∧ d ∈ Set.range a ∧ 0 < (c - d) / (1 + c * d) ∧ (c - d) / (1 + c * d) < 2 - Real.sqrt 3 := 
by
  sorry

end NUMINAMATH_GPT_exists_c_d_in_set_of_13_reals_l840_84021


namespace NUMINAMATH_GPT_find_cos_value_l840_84004

open Real

noncomputable def cos_value (α : ℝ) : ℝ :=
  cos (2 * π / 3 + 2 * α)

theorem find_cos_value (α : ℝ) (h : sin (π / 6 - α) = 1 / 4) :
  cos_value α = -7 / 8 :=
sorry

end NUMINAMATH_GPT_find_cos_value_l840_84004


namespace NUMINAMATH_GPT_part_a_part_b_l840_84051

open Nat

theorem part_a (n: ℕ) (h_pos: 0 < n) : (2^n - 1) % 7 = 0 ↔ ∃ k : ℕ, k > 0 ∧ n = 3 * k :=
sorry

theorem part_b (n: ℕ) (h_pos: 0 < n) : (2^n + 1) % 7 ≠ 0 :=
sorry

end NUMINAMATH_GPT_part_a_part_b_l840_84051


namespace NUMINAMATH_GPT_gasoline_tank_capacity_l840_84011

theorem gasoline_tank_capacity
  (initial_fill : ℝ) (final_fill : ℝ) (gallons_used : ℝ) (x : ℝ)
  (h1 : initial_fill = 3 / 4)
  (h2 : final_fill = 1 / 3)
  (h3 : gallons_used = 18)
  (h4 : initial_fill * x - final_fill * x = gallons_used) :
  x = 43 :=
by
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_gasoline_tank_capacity_l840_84011


namespace NUMINAMATH_GPT_miss_davis_sticks_left_l840_84003

theorem miss_davis_sticks_left (initial_sticks groups_per_class sticks_per_group : ℕ) 
(h1 : initial_sticks = 170) (h2 : groups_per_class = 10) (h3 : sticks_per_group = 15) : 
initial_sticks - (groups_per_class * sticks_per_group) = 20 :=
by sorry

end NUMINAMATH_GPT_miss_davis_sticks_left_l840_84003


namespace NUMINAMATH_GPT_problem_part_I_problem_part_II_l840_84094

-- Define the problem and the proof requirements in Lean 4
theorem problem_part_I (a b c : ℝ) (A B C : ℝ) (sinB_nonneg : 0 ≤ Real.sin B) 
(sinB_squared : Real.sin B ^ 2 = 2 * Real.sin A * Real.sin C) 
(h_a : a = 2) (h_b : b = 2) : 
Real.cos B = 1/4 :=
sorry

theorem problem_part_II (a b c : ℝ) (A B C : ℝ) (h_B : B = π / 2) 
(h_a : a = Real.sqrt 2) 
(sinB_squared : Real.sin B ^ 2 = 2 * Real.sin A * Real.sin C) :
1/2 * a * c = 1 :=
sorry

end NUMINAMATH_GPT_problem_part_I_problem_part_II_l840_84094


namespace NUMINAMATH_GPT_find_positive_solutions_l840_84073

theorem find_positive_solutions (x₁ x₂ x₃ x₄ x₅ : ℝ) (h_pos : 0 < x₁ ∧ 0 < x₂ ∧ 0 < x₃ ∧ 0 < x₄ ∧ 0 < x₅)
    (h1 : x₁ + x₂ = x₃^2)
    (h2 : x₂ + x₃ = x₄^2)
    (h3 : x₃ + x₄ = x₅^2)
    (h4 : x₄ + x₅ = x₁^2)
    (h5 : x₅ + x₁ = x₂^2) :
    x₁ = 2 ∧ x₂ = 2 ∧ x₃ = 2 ∧ x₄ = 2 ∧ x₅ = 2 := 
    by {
        -- Proof goes here
        sorry
    }

end NUMINAMATH_GPT_find_positive_solutions_l840_84073


namespace NUMINAMATH_GPT_function_monotonically_increasing_range_l840_84009

theorem function_monotonically_increasing_range (a : ℝ) :
  (∀ x y : ℝ, x ≤ 1 ∧ y ≤ 1 ∧ x ≤ y → ((4 - a / 2) * x + 2) ≤ ((4 - a / 2) * y + 2)) ∧
  (∀ x y : ℝ, x > 1 ∧ y > 1 ∧ x ≤ y → a^x ≤ a^y) ∧
  (∀ x : ℝ, if x = 1 then a^1 ≥ (4 - a / 2) * 1 + 2 else true) ↔
  4 ≤ a ∧ a < 8 :=
sorry

end NUMINAMATH_GPT_function_monotonically_increasing_range_l840_84009


namespace NUMINAMATH_GPT_eq_x2_inv_x2_and_x8_inv_x8_l840_84056

theorem eq_x2_inv_x2_and_x8_inv_x8 (x : ℝ) 
  (h : 47 = x^4 + 1 / x^4) : 
  (x^2 + 1 / x^2 = 7) ∧ (x^8 + 1 / x^8 = -433) :=
by
  sorry

end NUMINAMATH_GPT_eq_x2_inv_x2_and_x8_inv_x8_l840_84056


namespace NUMINAMATH_GPT_surface_area_increase_l840_84020

def cube_dimensions : ℝ × ℝ × ℝ := (10, 10, 10)

def number_of_cuts := 3

def initial_surface_area (length : ℝ) (width : ℝ) (height : ℝ) : ℝ :=
  6 * (length * width)

def increase_in_surface_area (cuts : ℕ) (length : ℝ) (width : ℝ) : ℝ :=
  cuts * 2 * (length * width)

theorem surface_area_increase : 
  initial_surface_area 10 10 10 + increase_in_surface_area 3 10 10 = 
  initial_surface_area 10 10 10 + 600 :=
by
  sorry

end NUMINAMATH_GPT_surface_area_increase_l840_84020


namespace NUMINAMATH_GPT_correct_average_is_19_l840_84093

-- Definitions
def incorrect_avg : ℕ := 16
def num_values : ℕ := 10
def incorrect_reading : ℕ := 25
def correct_reading : ℕ := 55

-- Theorem to prove
theorem correct_average_is_19 :
  ((incorrect_avg * num_values - incorrect_reading + correct_reading) / num_values) = 19 :=
by
  sorry

end NUMINAMATH_GPT_correct_average_is_19_l840_84093


namespace NUMINAMATH_GPT_find_extrema_l840_84074

noncomputable def f (x : ℝ) : ℝ := x^3 + (-3/2) * x^2 + (-3) * x + 1
noncomputable def f' (x : ℝ) : ℝ := 3 * x^2 + 2 * (-3/2) * x + (-3)
noncomputable def g (x : ℝ) : ℝ := f' x * Real.exp x

theorem find_extrema :
  (a = -3/2 ∧ b = -3 ∧ f' (1) = (3 * (1:ℝ)^2 - 3/2 * (1:ℝ) - 3) ) ∧
  (g 1 = -3 * Real.exp 1 ∧ g (-2) = 15 * Real.exp (-2)) := 
by
  -- Sorry for skipping the proof
  sorry

end NUMINAMATH_GPT_find_extrema_l840_84074


namespace NUMINAMATH_GPT_mul_exponent_property_l840_84090

variable (m : ℕ)  -- Assuming m is a natural number for simplicity

theorem mul_exponent_property : m^2 * m^3 = m^5 := 
by {
  sorry
}

end NUMINAMATH_GPT_mul_exponent_property_l840_84090


namespace NUMINAMATH_GPT_jeff_average_skips_is_14_l840_84006

-- Definitions of the given conditions in the problem
def sam_skips_per_round : ℕ := 16
def rounds : ℕ := 4

-- Number of skips by Jeff in each round based on the conditions
def jeff_first_round_skips : ℕ := sam_skips_per_round - 1
def jeff_second_round_skips : ℕ := sam_skips_per_round - 3
def jeff_third_round_skips : ℕ := sam_skips_per_round + 4
def jeff_fourth_round_skips : ℕ := sam_skips_per_round / 2

-- Total skips by Jeff in all rounds
def jeff_total_skips : ℕ := jeff_first_round_skips + 
                           jeff_second_round_skips + 
                           jeff_third_round_skips + 
                           jeff_fourth_round_skips

-- Average skips per round by Jeff
def jeff_average_skips : ℕ := jeff_total_skips / rounds

-- Theorem statement
theorem jeff_average_skips_is_14 : jeff_average_skips = 14 := 
by 
    sorry

end NUMINAMATH_GPT_jeff_average_skips_is_14_l840_84006


namespace NUMINAMATH_GPT_mary_earnings_max_hours_l840_84083

noncomputable def earnings (hours : ℕ) : ℝ :=
  if hours <= 40 then 
    hours * 10
  else if hours <= 60 then 
    (40 * 10) + ((hours - 40) * 13)
  else 
    (40 * 10) + (20 * 13) + ((hours - 60) * 16)

theorem mary_earnings_max_hours : 
  earnings 70 = 820 :=
by
  sorry

end NUMINAMATH_GPT_mary_earnings_max_hours_l840_84083


namespace NUMINAMATH_GPT_trapezoid_area_l840_84034

variable (a b : ℝ) (h1 : a > b)

theorem trapezoid_area (h2 : ∃ (angle1 angle2 : ℝ), angle1 = 30 ∧ angle2 = 45) : 
  (1/4) * ((a^2 - b^2) * (Real.sqrt 3 - 1)) = 
    ((1/2) * (a + b) * ((b - a) * (Real.sqrt 3 - 1) / 2)) := 
sorry

end NUMINAMATH_GPT_trapezoid_area_l840_84034


namespace NUMINAMATH_GPT_find_multiple_l840_84060

theorem find_multiple (m : ℤ) (h : 38 + m * 43 = 124) : m = 2 := by
    sorry

end NUMINAMATH_GPT_find_multiple_l840_84060


namespace NUMINAMATH_GPT_new_average_weight_l840_84015

theorem new_average_weight 
  (average_weight_19 : ℕ → ℝ)
  (weight_new_student : ℕ → ℝ)
  (new_student_count : ℕ)
  (old_student_count : ℕ)
  (h1 : average_weight_19 old_student_count = 15.0)
  (h2 : weight_new_student new_student_count = 11.0)
  : (average_weight_19 (old_student_count + new_student_count) = 14.8) :=
by
  sorry

end NUMINAMATH_GPT_new_average_weight_l840_84015


namespace NUMINAMATH_GPT_oranges_left_to_sell_today_l840_84091

theorem oranges_left_to_sell_today (initial_dozen : Nat)
    (reserved_fraction1 reserved_fraction2 sold_fraction eaten_fraction : ℚ)
    (rotten_oranges : Nat) 
    (h1 : initial_dozen = 7)
    (h2 : reserved_fraction1 = 1/4)
    (h3 : reserved_fraction2 = 1/6)
    (h4 : sold_fraction = 3/7)
    (h5 : eaten_fraction = 1/10)
    (h6 : rotten_oranges = 4) : 
    let total_oranges := initial_dozen * 12
    let reserved1 := total_oranges * reserved_fraction1
    let reserved2 := total_oranges * reserved_fraction2
    let remaining_after_reservation := total_oranges - reserved1 - reserved2
    let sold_yesterday := remaining_after_reservation * sold_fraction
    let remaining_after_sale := remaining_after_reservation - sold_yesterday
    let eaten_by_birds := remaining_after_sale * eaten_fraction
    let remaining_after_birds := remaining_after_sale - eaten_by_birds
    let final_remaining := remaining_after_birds - rotten_oranges
    final_remaining = 22 :=
by
    sorry

end NUMINAMATH_GPT_oranges_left_to_sell_today_l840_84091


namespace NUMINAMATH_GPT_carnations_count_l840_84053

theorem carnations_count (c : ℕ) : 
  (9 * 6 = 54) ∧ (47 ≤ c + 47) ∧ (c + 47 = 54) → c = 7 := 
by
  sorry

end NUMINAMATH_GPT_carnations_count_l840_84053


namespace NUMINAMATH_GPT_combined_total_years_l840_84079

theorem combined_total_years (A : ℕ) (V : ℕ) (D : ℕ)
(h1 : V = A + 9)
(h2 : V = D - 9)
(h3 : D = 34) : A + V + D = 75 :=
by sorry

end NUMINAMATH_GPT_combined_total_years_l840_84079


namespace NUMINAMATH_GPT_area_of_square_l840_84096

theorem area_of_square 
  (a : ℝ)
  (h : 4 * a = 28) :
  a^2 = 49 :=
sorry

end NUMINAMATH_GPT_area_of_square_l840_84096


namespace NUMINAMATH_GPT_natural_numbers_satisfying_condition_l840_84024

open Nat

theorem natural_numbers_satisfying_condition (r : ℕ) :
  ∃ k : Set ℕ, k = { k | ∃ s t : ℕ, k = 2^(r + s) * t ∧ 2 ∣ t ∧ 2 ∣ s } :=
by
  sorry

end NUMINAMATH_GPT_natural_numbers_satisfying_condition_l840_84024


namespace NUMINAMATH_GPT_chips_left_uneaten_l840_84031

theorem chips_left_uneaten 
    (chips_per_cookie : ℕ)
    (cookies_per_dozen : ℕ)
    (dozens_of_cookies : ℕ)
    (cookies_eaten_ratio : ℕ) 
    (h_chips : chips_per_cookie = 7)
    (h_cookies_dozen : cookies_per_dozen = 12)
    (h_dozens : dozens_of_cookies = 4)
    (h_eaten_ratio : cookies_eaten_ratio = 2) : 
  (cookies_per_dozen * dozens_of_cookies / cookies_eaten_ratio) * chips_per_cookie = 168 :=
by 
  sorry

end NUMINAMATH_GPT_chips_left_uneaten_l840_84031


namespace NUMINAMATH_GPT_binom_sub_floor_divisible_by_prime_l840_84068

theorem binom_sub_floor_divisible_by_prime (p n : ℕ) (hp : Nat.Prime p) (hn : n ≥ p) :
  p ∣ (Nat.choose n p - (n / p)) :=
sorry

end NUMINAMATH_GPT_binom_sub_floor_divisible_by_prime_l840_84068


namespace NUMINAMATH_GPT_equation_solution_l840_84012

def solve_equation (x : ℝ) : Prop :=
  ((3 * x + 6) / (x^2 + 5 * x - 6) = (4 - x) / (x - 2)) ↔ 
  x = -3 ∨ x = (1 + Real.sqrt 17) / 2 ∨ x = (1 - Real.sqrt 17) / 2

theorem equation_solution (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -6) (h3 : x ≠ 2) : solve_equation x :=
by
  sorry

end NUMINAMATH_GPT_equation_solution_l840_84012


namespace NUMINAMATH_GPT_stratified_sampling_second_grade_l840_84082

theorem stratified_sampling_second_grade (r1 r2 r3 : ℕ) (total_sample : ℕ) (total_ratio : ℕ):
  r1 = 3 ∧ r2 = 3 ∧ r3 = 4 ∧ total_sample = 50 ∧ total_ratio = r1 + r2 + r3 →
  (r2 * total_sample) / total_ratio = 15 :=
by
  sorry

end NUMINAMATH_GPT_stratified_sampling_second_grade_l840_84082


namespace NUMINAMATH_GPT_alpha_cubed_plus_5beta_plus_10_l840_84075

noncomputable def α: ℝ := sorry
noncomputable def β: ℝ := sorry

-- Given conditions
axiom roots_eq : ∀ x : ℝ, x^2 + 2 * x - 1 = 0 → (x = α ∨ x = β)
axiom sum_eq : α + β = -2
axiom prod_eq : α * β = -1

-- The theorem stating the desired result
theorem alpha_cubed_plus_5beta_plus_10 :
  α^3 + 5 * β + 10 = -2 :=
sorry

end NUMINAMATH_GPT_alpha_cubed_plus_5beta_plus_10_l840_84075


namespace NUMINAMATH_GPT_find_divisor_l840_84000

theorem find_divisor (D N : ℕ) (h₁ : N = 265) (h₂ : N / D + 8 = 61) : D = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_divisor_l840_84000


namespace NUMINAMATH_GPT_prove_perpendicular_planes_l840_84059

-- Defining the non-coincident lines m and n
variables {m n : Set Point} {α β : Set Point}

-- Lines and plane relationship definitions
def parallel (x y : Set Point) : Prop := sorry
def perpendicular (x y : Set Point) : Prop := sorry
def subset (x y : Set Point) : Prop := sorry

-- Given conditions
axiom h1 : parallel m n
axiom h2 : subset m α
axiom h3 : perpendicular n β

-- Prove that α is perpendicular to β
theorem prove_perpendicular_planes :
  perpendicular α β :=
  sorry

end NUMINAMATH_GPT_prove_perpendicular_planes_l840_84059


namespace NUMINAMATH_GPT_simplify_sqrt_450_l840_84064

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_simplify_sqrt_450_l840_84064


namespace NUMINAMATH_GPT_find_a_l840_84071

theorem find_a (a : ℝ) (h : (2:ℝ)^2 + 2 * a - 3 * a = 0) : a = 4 :=
sorry

end NUMINAMATH_GPT_find_a_l840_84071


namespace NUMINAMATH_GPT_chess_group_players_l840_84026

theorem chess_group_players (n : ℕ) (h : n * (n - 1) / 2 = 1225) : n = 50 :=
sorry

end NUMINAMATH_GPT_chess_group_players_l840_84026


namespace NUMINAMATH_GPT_granger_cisco_combined_spots_l840_84040

theorem granger_cisco_combined_spots :
  let R := 46
  let C := (R / 2) - 5
  let G := 5 * C
  G + C = 108 := by 
  let R := 46
  let C := (R / 2) - 5
  let G := 5 * C
  sorry

end NUMINAMATH_GPT_granger_cisco_combined_spots_l840_84040


namespace NUMINAMATH_GPT_average_price_per_book_l840_84033

theorem average_price_per_book
  (spent1 spent2 spent3 spent4 : ℝ) (books1 books2 books3 books4 : ℕ)
  (h1 : spent1 = 1080) (h2 : spent2 = 840) (h3 : spent3 = 765) (h4 : spent4 = 630)
  (hb1 : books1 = 65) (hb2 : books2 = 55) (hb3 : books3 = 45) (hb4 : books4 = 35) :
  (spent1 + spent2 + spent3 + spent4) / (books1 + books2 + books3 + books4) = 16.575 :=
by {
  sorry
}

end NUMINAMATH_GPT_average_price_per_book_l840_84033


namespace NUMINAMATH_GPT_a_1994_is_7_l840_84014

def f (m : ℕ) : ℕ := m % 10

def a (n : ℕ) : ℕ := f (2^(n + 1) - 1)

theorem a_1994_is_7 : a 1994 = 7 :=
by
  sorry

end NUMINAMATH_GPT_a_1994_is_7_l840_84014


namespace NUMINAMATH_GPT_quadratic_inequality_iff_l840_84084

noncomputable def quadratic_inequality_solution (x : ℝ) : Prop := x^2 + 4*x - 96 > abs x

theorem quadratic_inequality_iff (x : ℝ) : quadratic_inequality_solution x ↔ x < -12 ∨ x > 8 := by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_iff_l840_84084


namespace NUMINAMATH_GPT_ratio_of_football_to_hockey_l840_84041

variables (B F H s : ℕ)

-- Definitions from conditions
def condition1 : Prop := B = F - 50
def condition2 : Prop := F = s * H
def condition3 : Prop := H = 200
def condition4 : Prop := B + F + H = 1750

-- Proof statement
theorem ratio_of_football_to_hockey (B F H s : ℕ) 
  (h1 : condition1 B F)
  (h2 : condition2 F s H)
  (h3 : condition3 H)
  (h4 : condition4 B F H) : F / H = 4 :=
sorry

end NUMINAMATH_GPT_ratio_of_football_to_hockey_l840_84041


namespace NUMINAMATH_GPT_relationship_a_b_l840_84092

open Real

noncomputable def f (x : ℝ) : ℝ := x^2 + x - 2

noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ -2 ∨ x ≥ 1 then 0 else -x^2 - x + 2

theorem relationship_a_b (a b : ℝ) (h_pos : a > 0) :
  (∀ x : ℝ, a * x + b = g x) → (2 * a < b ∧ b < (a + 1)^2 / 4 + 2 ∧ 0 < a ∧ a < 3) :=
sorry

end NUMINAMATH_GPT_relationship_a_b_l840_84092


namespace NUMINAMATH_GPT_find_a_plus_b_l840_84098

theorem find_a_plus_b (a b : ℝ) (h1 : (a + Real.sqrt b) + (a - Real.sqrt b) = 0)
                      (h2 : (a + Real.sqrt b) * (a - Real.sqrt b) = 16) : a + b = -16 :=
by sorry

end NUMINAMATH_GPT_find_a_plus_b_l840_84098


namespace NUMINAMATH_GPT_contradiction_proof_l840_84022

theorem contradiction_proof (a b c d : ℝ) (h1 : a + b = 1) (h2 : c + d = 1) (h3 : ac + bd > 1) : ¬ (0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d) :=
by 
  sorry

end NUMINAMATH_GPT_contradiction_proof_l840_84022


namespace NUMINAMATH_GPT_ratio_of_running_speed_l840_84019

theorem ratio_of_running_speed (distance : ℝ) (time_jack : ℝ) (time_jill : ℝ) 
  (h_distance_eq : distance = 42) (h_time_jack_eq : time_jack = 6) 
  (h_time_jill_eq : time_jill = 4.2) :
  (distance / time_jack) / (distance / time_jill) = 7 / 10 := by 
  sorry

end NUMINAMATH_GPT_ratio_of_running_speed_l840_84019


namespace NUMINAMATH_GPT_math_problem_proof_l840_84036

-- Define the system of equations
structure equations :=
  (x y m : ℝ)
  (eq1 : x + 2*y - 6 = 0)
  (eq2 : x - 2*y + m*x + 5 = 0)

-- Define the problem conditions and prove the required solutions in Lean 4
theorem math_problem_proof :
  -- Part 1: Positive integer solutions for x + 2y - 6 = 0
  (∀ x y : ℕ, x + 2*y = 6 → (x, y) = (2, 2) ∨ (x, y) = (4, 1)) ∧
  -- Part 2: Given x + y = 0, find m
  (∀ x y : ℝ, x + y = 0 → x + 2*y - 6 = 0 → x - 2*y - (13/6)*x + 5 = 0) ∧
  -- Part 3: Fixed solution for x - 2y + mx + 5 = 0
  (∀ m : ℝ, 0 - 2*2.5 + m*0 + 5 = 0) :=
sorry

end NUMINAMATH_GPT_math_problem_proof_l840_84036


namespace NUMINAMATH_GPT_prob_neither_prime_nor_composite_l840_84008

theorem prob_neither_prime_nor_composite :
  (1 / 95 : ℚ) = 1 / 95 := by
  sorry

end NUMINAMATH_GPT_prob_neither_prime_nor_composite_l840_84008


namespace NUMINAMATH_GPT_find_overall_mean_score_l840_84007

variable (M N E : ℝ)
variable (m n e : ℝ)

theorem find_overall_mean_score :
  M = 85 → N = 75 → E = 65 →
  m / n = 4 / 5 → n / e = 3 / 2 →
  ((85 * m) + (75 * n) + (65 * e)) / (m + n + e) = 82 :=
by
  sorry

end NUMINAMATH_GPT_find_overall_mean_score_l840_84007


namespace NUMINAMATH_GPT_quadratic_has_two_equal_real_roots_l840_84097

theorem quadratic_has_two_equal_real_roots : ∃ c : ℝ, ∀ x : ℝ, (x^2 - 6*x + c = 0 ↔ (x = 3)) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_has_two_equal_real_roots_l840_84097


namespace NUMINAMATH_GPT_chord_length_intercepted_l840_84017

theorem chord_length_intercepted 
  (line_eq : ∀ x y : ℝ, 3 * x - 4 * y = 0)
  (circle_eq : ∀ x y : ℝ, (x - 1)^2 + (y - 2)^2 = 2) : 
  ∃ l : ℝ, l = 2 :=
by 
  sorry

end NUMINAMATH_GPT_chord_length_intercepted_l840_84017


namespace NUMINAMATH_GPT_cost_of_green_pill_l840_84061

-- Let the cost of a green pill be g and the cost of a pink pill be p
variables (g p : ℕ)
-- Beth takes two green pills and one pink pill each day
-- A green pill costs twice as much as a pink pill
-- The total cost for the pills over three weeks (21 days) is $945

theorem cost_of_green_pill : 
  (2 * g + p) * 21 = 945 ∧ g = 2 * p → g = 18 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_green_pill_l840_84061


namespace NUMINAMATH_GPT_assign_grades_l840_84001

-- Definitions based on the conditions:
def num_students : ℕ := 12
def num_grades : ℕ := 4

-- Statement of the theorem
theorem assign_grades : num_grades ^ num_students = 16777216 := by
  sorry

end NUMINAMATH_GPT_assign_grades_l840_84001


namespace NUMINAMATH_GPT_unique_function_l840_84048

theorem unique_function (f : ℝ → ℝ) (h1 : ∀ x : ℝ, f (x + 1) ≥ f x + 1) 
  (h2 : ∀ x y : ℝ, f (x * y) ≥ f x * f y) : 
  ∀ x : ℝ, f x = x := 
sorry

end NUMINAMATH_GPT_unique_function_l840_84048


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l840_84029

-- Proof statement for problem 1
theorem problem1 : (1 : ℤ) * (-8) + 10 + 2 + (-1) = 3 := sorry

-- Proof statement for problem 2
theorem problem2 : (-21.6 : ℝ) - (-3) - |(-7.4)| + (-2 / 5) = -26.4 := sorry

-- Proof statement for problem 3
theorem problem3 : (-12 / 5) / (-1 / 10) * (-5 / 6) * (-0.4 : ℝ) = 8 := sorry

-- Proof statement for problem 4
theorem problem4 : ((5 / 8) - (1 / 6) + (7 / 12)) * (-24 : ℝ) = -25 := sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l840_84029


namespace NUMINAMATH_GPT_radius_squared_l840_84042

theorem radius_squared (r : ℝ) (AB_len CD_len BP_len : ℝ) (angle_APD : ℝ) (r_squared : ℝ) :
  AB_len = 10 →
  CD_len = 7 →
  BP_len = 8 →
  angle_APD = 60 →
  r_squared = r^2 →
  r_squared = 73 :=
by
  intros h₁ h₂ h₃ h₄ h₅
  sorry

end NUMINAMATH_GPT_radius_squared_l840_84042


namespace NUMINAMATH_GPT_chess_tournament_time_spent_l840_84076

theorem chess_tournament_time_spent (games : ℕ) (moves_per_game : ℕ)
  (opening_moves : ℕ) (middle_moves : ℕ) (endgame_moves : ℕ)
  (polly_opening_time : ℝ) (peter_opening_time : ℝ)
  (polly_middle_time : ℝ) (peter_middle_time : ℝ)
  (polly_endgame_time : ℝ) (peter_endgame_time : ℝ)
  (total_time_hours : ℝ) :
  games = 4 →
  moves_per_game = 38 →
  opening_moves = 12 →
  middle_moves = 18 →
  endgame_moves = 8 →
  polly_opening_time = 35 →
  peter_opening_time = 45 →
  polly_middle_time = 30 →
  peter_middle_time = 45 →
  polly_endgame_time = 40 →
  peter_endgame_time = 60 →
  total_time_hours = (4 * ((12 * 35 + 18 * 30 + 8 * 40) + (12 * 45 + 18 * 45 + 8 * 60))) / 3600 :=
sorry

end NUMINAMATH_GPT_chess_tournament_time_spent_l840_84076


namespace NUMINAMATH_GPT_poplar_more_than_pine_l840_84023

theorem poplar_more_than_pine (pine poplar : ℕ) (h1 : pine = 180) (h2 : poplar = 4 * pine) : poplar - pine = 540 :=
by
  -- Proof will be filled here
  sorry

end NUMINAMATH_GPT_poplar_more_than_pine_l840_84023


namespace NUMINAMATH_GPT_total_distance_travelled_eight_boys_on_circle_l840_84081

noncomputable def distance_travelled_by_boys (radius : ℝ) : ℝ :=
  let n := 8
  let angle := 2 * Real.pi / n
  let distance_to_non_adjacent := 2 * radius * Real.sin (2 * angle / 2)
  n * (100 + 3 * distance_to_non_adjacent)

theorem total_distance_travelled_eight_boys_on_circle :
  distance_travelled_by_boys 50 = 800 + 1200 * Real.sqrt 2 :=
  by
    sorry

end NUMINAMATH_GPT_total_distance_travelled_eight_boys_on_circle_l840_84081


namespace NUMINAMATH_GPT_original_weight_of_marble_l840_84044

theorem original_weight_of_marble (W : ℝ) (h1 : W * 0.75 * 0.85 * 0.90 = 109.0125) : W = 190 :=
by
  sorry

end NUMINAMATH_GPT_original_weight_of_marble_l840_84044


namespace NUMINAMATH_GPT_product_of_two_equal_numbers_l840_84045

-- Definitions and conditions
def arithmetic_mean (xs : List ℚ) : ℚ :=
  xs.sum / xs.length

-- Theorem stating the product of the two equal numbers
theorem product_of_two_equal_numbers (a b c : ℚ) (x : ℚ) :
  arithmetic_mean [a, b, c, x, x] = 20 → a = 22 → b = 18 → c = 32 → x * x = 196 :=
by
  intros h_mean h_a h_b h_c
  sorry

end NUMINAMATH_GPT_product_of_two_equal_numbers_l840_84045


namespace NUMINAMATH_GPT_total_length_of_ribbon_l840_84058

-- Define the conditions
def length_per_piece : ℕ := 73
def number_of_pieces : ℕ := 51

-- The theorem to prove
theorem total_length_of_ribbon : length_per_piece * number_of_pieces = 3723 :=
by
  sorry

end NUMINAMATH_GPT_total_length_of_ribbon_l840_84058


namespace NUMINAMATH_GPT_amin_probability_four_attempts_before_three_hits_amin_probability_not_qualified_stops_after_two_consecutive_misses_l840_84030

/-- Prove that the probability Amin makes 4 attempts before hitting 3 times (given the probability of each hit is 1/2) is 3/16. -/
theorem amin_probability_four_attempts_before_three_hits (p_hit : ℚ := 1 / 2) : 
  (∃ (P : ℚ), P = 3/16) :=
sorry

/-- Prove that the probability Amin stops shooting after missing two consecutive shots and not qualifying as level B or A player is 25/32, given the probability of each hit is 1/2. -/
theorem amin_probability_not_qualified_stops_after_two_consecutive_misses (p_hit : ℚ := 1 / 2) : 
  (∃ (P : ℚ), P = 25/32) :=
sorry

end NUMINAMATH_GPT_amin_probability_four_attempts_before_three_hits_amin_probability_not_qualified_stops_after_two_consecutive_misses_l840_84030


namespace NUMINAMATH_GPT_ball_in_boxes_l840_84086

theorem ball_in_boxes (n : ℕ) (k : ℕ) (h1 : n = 5) (h2 : k = 3) :
  k^n = 243 :=
by
  sorry

end NUMINAMATH_GPT_ball_in_boxes_l840_84086


namespace NUMINAMATH_GPT_billboard_shorter_side_length_l840_84043

theorem billboard_shorter_side_length
  (L W : ℝ)
  (h1 : L * W = 91)
  (h2 : 2 * L + 2 * W = 40) :
  L = 7 ∨ W = 7 :=
by sorry

end NUMINAMATH_GPT_billboard_shorter_side_length_l840_84043


namespace NUMINAMATH_GPT_geo_seq_b_formula_b_n_sum_T_n_l840_84037

-- Define the sequence a_n 
def a (n : ℕ) : ℕ :=
  if n = 0 then 1 else sorry -- Definition based on provided conditions

-- Define the partial sum S_n
def S (n : ℕ) : ℕ :=
  if n = 0 then 1 else 4 * a (n-1) + 2 -- Given condition S_{n+1} = 4a_n + 2

-- Condition for b_n
def b (n : ℕ) : ℕ :=
  a (n+1) - 2 * a n

-- Definition for c_n
def c (n : ℕ) := (b n) / 3

-- Define the sequence terms for c_n based sequence
def T (n : ℕ) : ℝ :=
  sorry -- Needs explicit definition from given sequence part

-- Proof statements
theorem geo_seq_b : ∀ n : ℕ, b (n + 1) = 2 * b n :=
  sorry

theorem formula_b_n : ∀ n : ℕ, b n = 3 * 2^(n-1) :=
  sorry

theorem sum_T_n : ∀ n : ℕ, T n = n / (n + 1) :=
  sorry

end NUMINAMATH_GPT_geo_seq_b_formula_b_n_sum_T_n_l840_84037


namespace NUMINAMATH_GPT_max_additional_hours_l840_84047

/-- Define the additional hours of studying given the investments in dorms, food, and parties -/
def additional_hours (a b c : ℝ) : ℝ :=
  5 * a + 3 * b + (11 * c - c^2)

/-- Define the total investment constraint -/
def investment_constraint (a b c : ℝ) : Prop :=
  a + b + c = 5

/-- Prove the maximal additional hours of studying -/
theorem max_additional_hours : ∃ (a b c : ℝ), investment_constraint a b c ∧ additional_hours a b c = 34 :=
by
  sorry

end NUMINAMATH_GPT_max_additional_hours_l840_84047


namespace NUMINAMATH_GPT_new_time_between_maintenance_checks_l840_84032

-- Definitions based on the conditions
def original_time : ℝ := 25
def percentage_increase : ℝ := 0.20

-- Statement to be proved
theorem new_time_between_maintenance_checks : original_time * (1 + percentage_increase) = 30 := by
  sorry

end NUMINAMATH_GPT_new_time_between_maintenance_checks_l840_84032


namespace NUMINAMATH_GPT_maximum_triangle_area_l840_84077

-- Define the maximum area of a triangle given two sides.
theorem maximum_triangle_area (a b : ℝ) (h_a : a = 1984) (h_b : b = 2016) :
  ∃ (max_area : ℝ), max_area = 1998912 :=
by
  sorry

end NUMINAMATH_GPT_maximum_triangle_area_l840_84077


namespace NUMINAMATH_GPT_usb_drive_total_capacity_l840_84035

-- Define the conditions as α = total capacity, β = busy space (50%), γ = available space (50%)
variable (α : ℕ) -- Total capacity of the USB drive in gigabytes
variable (β γ : ℕ) -- Busy space and available space in gigabytes
variable (h1 : β = α / 2) -- 50% of total capacity is busy
variable (h2 : γ = 8)  -- 8 gigabytes are still available

-- Define the problem as a theorem that these conditions imply the total capacity
theorem usb_drive_total_capacity (h : γ = α / 2) : α = 16 :=
by
  -- defer the proof
  sorry

end NUMINAMATH_GPT_usb_drive_total_capacity_l840_84035


namespace NUMINAMATH_GPT_max_value_expression_l840_84046

theorem max_value_expression (a b c d : ℤ) (hb_pos : b > 0)
  (h1 : a + b = c) (h2 : b + c = d) (h3 : c + d = a) : 
  a - 2 * b + 3 * c - 4 * d = -7 := 
sorry

end NUMINAMATH_GPT_max_value_expression_l840_84046


namespace NUMINAMATH_GPT_greatest_num_fruit_in_each_basket_l840_84025

theorem greatest_num_fruit_in_each_basket : 
  let oranges := 15
  let peaches := 9
  let pears := 18
  let gcd := Nat.gcd (Nat.gcd oranges peaches) pears
  gcd = 3 :=
by
  sorry

end NUMINAMATH_GPT_greatest_num_fruit_in_each_basket_l840_84025


namespace NUMINAMATH_GPT_radii_difference_of_concentric_circles_l840_84055

theorem radii_difference_of_concentric_circles 
  (r : ℝ) 
  (h_area_ratio : (π * (2 * r)^2) / (π * r^2) = 4) : 
  (2 * r) - r = r :=
by
  sorry

end NUMINAMATH_GPT_radii_difference_of_concentric_circles_l840_84055


namespace NUMINAMATH_GPT_mistake_position_is_34_l840_84049

def arithmetic_sequence_sum (n : ℕ) (a_1 : ℕ) (d : ℕ) : ℕ :=
  n * (2 * a_1 + (n - 1) * d) / 2

def modified_sequence_sum (n : ℕ) (a_1 : ℕ) (d : ℕ) (mistake_index : ℕ) : ℕ :=
  let correct_sum := arithmetic_sequence_sum n a_1 d
  correct_sum - 2 * d

theorem mistake_position_is_34 :
  ∃ mistake_index : ℕ, mistake_index = 34 ∧ 
    modified_sequence_sum 37 1 3 mistake_index = 2011 :=
by
  sorry

end NUMINAMATH_GPT_mistake_position_is_34_l840_84049


namespace NUMINAMATH_GPT_car_distance_l840_84095

noncomputable def distance_covered (S : ℝ) (T : ℝ) (new_speed : ℝ) : ℝ :=
  S * T

theorem car_distance (S : ℝ) (T : ℝ) (new_time : ℝ) (new_speed : ℝ)
  (h1 : T = 12)
  (h2 : new_time = (3/4) * T)
  (h3 : new_speed = 60)
  (h4 : distance_covered new_speed new_time = 540) :
    distance_covered S T = 540 :=
by
  sorry

end NUMINAMATH_GPT_car_distance_l840_84095


namespace NUMINAMATH_GPT_remaining_fruits_correct_l840_84080

-- The definitions for the number of fruits in terms of the number of plums
def apples := 180
def plums := apples / 3
def pears := 2 * plums
def cherries := 4 * apples

-- Damien's portion of each type of fruit picked
def apples_picked := (3/5) * apples
def plums_picked := (2/3) * plums
def pears_picked := (3/4) * pears
def cherries_picked := (7/10) * cherries

-- The remaining number of fruits
def apples_remaining := apples - apples_picked
def plums_remaining := plums - plums_picked
def pears_remaining := pears - pears_picked
def cherries_remaining := cherries - cherries_picked

-- The total remaining number of fruits
def total_remaining_fruits := apples_remaining + plums_remaining + pears_remaining + cherries_remaining

theorem remaining_fruits_correct :
  total_remaining_fruits = 338 :=
by {
  -- The conditions ensure that the imported libraries are broad
  sorry
}

end NUMINAMATH_GPT_remaining_fruits_correct_l840_84080


namespace NUMINAMATH_GPT_top_quality_soccer_balls_l840_84072

theorem top_quality_soccer_balls (N : ℕ) (f : ℝ) (hN : N = 10000) (hf : f = 0.975) : N * f = 9750 := by
  sorry

end NUMINAMATH_GPT_top_quality_soccer_balls_l840_84072


namespace NUMINAMATH_GPT_cookie_distribution_l840_84088

theorem cookie_distribution : 
  ∀ (n c T : ℕ), n = 6 → c = 4 → T = n * c → T = 24 :=
by 
  intros n c T h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_cookie_distribution_l840_84088


namespace NUMINAMATH_GPT_positive_difference_l840_84027

theorem positive_difference (a b : ℝ) (h₁ : a + b = 10) (h₂ : a^2 - b^2 = 40) : |a - b| = 4 :=
by
  sorry

end NUMINAMATH_GPT_positive_difference_l840_84027


namespace NUMINAMATH_GPT_percentage_fewer_than_50000_l840_84016

def percentage_lt_20000 : ℝ := 35
def percentage_20000_to_49999 : ℝ := 45
def percentage_lt_50000 : ℝ := 80

theorem percentage_fewer_than_50000 :
  percentage_lt_20000 + percentage_20000_to_49999 = percentage_lt_50000 := 
by
  sorry

end NUMINAMATH_GPT_percentage_fewer_than_50000_l840_84016


namespace NUMINAMATH_GPT_pyramid_apex_angle_l840_84010

theorem pyramid_apex_angle (A B C D E O : Type) 
  (square_base : Π (P Q : Type), Prop) 
  (isosceles_triangle : Π (R S T : Type), Prop)
  (AEB_angle : Π (X Y Z : Type), Prop) 
  (angle_AOB : ℝ)
  (angle_AEB : ℝ)
  (square_base_conditions : square_base A B ∧ square_base B C ∧ square_base C D ∧ square_base D A)
  (isosceles_triangle_conditions : isosceles_triangle A E B ∧ isosceles_triangle B E C ∧ isosceles_triangle C E D ∧ isosceles_triangle D E A)
  (center : O)
  (diagonals_intersect_at_right_angle : angle_AOB = 90)
  (measured_angle_at_apex : angle_AEB = 100) :
False :=
sorry

end NUMINAMATH_GPT_pyramid_apex_angle_l840_84010


namespace NUMINAMATH_GPT_g_value_l840_84054

noncomputable def g (x : ℝ) : ℝ := sorry

theorem g_value (h : ∀ x ≠ 0, g x - 3 * g (1 / x) = 3^x) :
  g 3 = -(27 + 3 * (3:ℝ)^(1/3)) / 8 :=
sorry

end NUMINAMATH_GPT_g_value_l840_84054


namespace NUMINAMATH_GPT_trapezium_distance_l840_84018

theorem trapezium_distance (a b area : ℝ) (h : ℝ) :
  a = 20 ∧ b = 18 ∧ area = 266 ∧
  area = (1/2) * (a + b) * h -> h = 14 :=
by
  sorry

end NUMINAMATH_GPT_trapezium_distance_l840_84018


namespace NUMINAMATH_GPT_inclination_angle_of_line_l840_84085

theorem inclination_angle_of_line 
  (l : ℝ) (h : l = Real.tan (-π / 6)) : 
  ∀ θ, θ = Real.pi / 2 :=
by
  -- Placeholder proof
  sorry

end NUMINAMATH_GPT_inclination_angle_of_line_l840_84085


namespace NUMINAMATH_GPT_nonincreasing_7_digit_integers_l840_84065

theorem nonincreasing_7_digit_integers : 
  ∃ n : ℕ, n = 11439 ∧ (∀ x : ℕ, (10^6 ≤ x ∧ x < 10^7) → 
    (∀ i j : ℕ, 1 ≤ i ∧ i < j ∧ j ≤ 7 → (x / 10^(7 - i) % 10) ≥ (x / 10^(7 - j) % 10))) :=
by
  sorry

end NUMINAMATH_GPT_nonincreasing_7_digit_integers_l840_84065


namespace NUMINAMATH_GPT_circle_area_l840_84089

theorem circle_area (r : ℝ) (h : 5 * (1 / (2 * π * r)) = r / 2) : π * r^2 = 5 := 
by
  sorry -- Proof is not required, placeholder for the actual proof

end NUMINAMATH_GPT_circle_area_l840_84089


namespace NUMINAMATH_GPT_qudrilateral_diagonal_length_l840_84067

theorem qudrilateral_diagonal_length (A h1 h2 d : ℝ) 
  (h_area : A = 140) (h_offsets : h1 = 8) (h_offsets2 : h2 = 2) 
  (h_formula : A = 1 / 2 * d * (h1 + h2)) : 
  d = 28 :=
by
  sorry

end NUMINAMATH_GPT_qudrilateral_diagonal_length_l840_84067


namespace NUMINAMATH_GPT_find_value_of_m_l840_84063

theorem find_value_of_m :
  (∃ y : ℝ, y = 20 - (0.5 * -6.7)) →
  (m : ℝ) = 3 * -6.7 + (20 - (0.5 * -6.7)) :=
by {
  sorry
}

end NUMINAMATH_GPT_find_value_of_m_l840_84063


namespace NUMINAMATH_GPT_gcd_of_12012_and_18018_l840_84069

theorem gcd_of_12012_and_18018 : Int.gcd 12012 18018 = 6006 := 
by
  -- Here we are assuming the factorization given in the conditions
  have h₁ : 12012 = 12 * 1001 := sorry
  have h₂ : 18018 = 18 * 1001 := sorry
  have gcd_12_18 : Int.gcd 12 18 = 6 := sorry
  -- This sorry will be replaced by the actual proof involving the above conditions to conclude the stated theorem
  sorry

end NUMINAMATH_GPT_gcd_of_12012_and_18018_l840_84069


namespace NUMINAMATH_GPT_largest_value_l840_84057

def X := (2010 / 2009) + (2010 / 2011)
def Y := (2010 / 2011) + (2012 / 2011)
def Z := (2011 / 2010) + (2011 / 2012)

theorem largest_value : X > Y ∧ X > Z := 
by
  sorry

end NUMINAMATH_GPT_largest_value_l840_84057


namespace NUMINAMATH_GPT_range_m_l840_84078

noncomputable def even_function (f : ℝ → ℝ) : Prop := 
  ∀ x, f x = f (-x)

noncomputable def decreasing_on_non_neg (f : ℝ → ℝ) : Prop := 
  ∀ ⦃x y⦄, 0 ≤ x → x ≤ y → f y ≤ f x

theorem range_m (f : ℝ → ℝ)
  (h_even : even_function f)
  (h_dec : decreasing_on_non_neg f) :
  ∀ m, f (1 - m) < f m → m < 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_range_m_l840_84078


namespace NUMINAMATH_GPT_escalator_length_l840_84066

theorem escalator_length :
  ∃ L : ℝ, L = 150 ∧ 
    (∀ t : ℝ, t = 10 → ∀ v_p : ℝ, v_p = 3 → ∀ v_e : ℝ, v_e = 12 → L = (v_p + v_e) * t) :=
by sorry

end NUMINAMATH_GPT_escalator_length_l840_84066


namespace NUMINAMATH_GPT_cos_seven_pi_over_four_l840_84062

theorem cos_seven_pi_over_four :
  Real.cos (7 * Real.pi / 4) = Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cos_seven_pi_over_four_l840_84062
