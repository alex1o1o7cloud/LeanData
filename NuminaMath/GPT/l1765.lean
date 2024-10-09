import Mathlib

namespace area_of_triangle_ABC_eq_3_l1765_176575

variable {n : ℕ}

def arithmetic_seq (a_1 d : ℤ) : ℕ → ℤ
| 0     => 0
| (n+1) => a_1 + n * d

def sum_arithmetic_seq (a_1 d : ℤ) : ℕ → ℤ
| 0     => 0
| (n+1) => (n + 1) * a_1 + (n * (n + 1) / 2) * d

def f (n : ℕ) : ℤ := sum_arithmetic_seq 4 6 n

def point_A (n : ℕ) : ℤ × ℤ := (n, f n)
def point_B (n : ℕ) : ℤ × ℤ := (n + 1, f (n + 1))
def point_C (n : ℕ) : ℤ × ℤ := (n + 2, f (n + 2))

def area_of_triangle (A B C : ℤ × ℤ) : ℤ :=
  (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)).natAbs / 2

theorem area_of_triangle_ABC_eq_3 : 
  ∀ (n : ℕ), area_of_triangle (point_A n) (point_B n) (point_C n) = 3 := 
sorry

end area_of_triangle_ABC_eq_3_l1765_176575


namespace change_in_expression_l1765_176517

theorem change_in_expression (x a : ℝ) (h : 0 < a) :
  (x + a)^3 - 3 * (x + a) - (x^3 - 3 * x) = 3 * a * x^2 + 3 * a^2 * x + a^3 - 3 * a
  ∨ (x - a)^3 - 3 * (x - a) - (x^3 - 3 * x) = -3 * a * x^2 + 3 * a^2 * x - a^3 + 3 * a :=
sorry

end change_in_expression_l1765_176517


namespace calc_is_a_pow4_l1765_176549

theorem calc_is_a_pow4 (a : ℕ) : (a^2)^2 = a^4 := 
by 
  sorry

end calc_is_a_pow4_l1765_176549


namespace exists_large_natural_with_high_digit_sum_l1765_176552

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 10) + sum_of_digits (n / 10)

theorem exists_large_natural_with_high_digit_sum :
  ∃ b : ℕ, ∀ n : ℕ, n > b → sum_of_digits (factorial n) ≥ 10 ^ 100 :=
by sorry

end exists_large_natural_with_high_digit_sum_l1765_176552


namespace binary_mul_1101_111_eq_1001111_l1765_176567

theorem binary_mul_1101_111_eq_1001111 :
  let n1 := 0b1101 -- binary representation of 13
  let n2 := 0b111  -- binary representation of 7
  let product := 0b1001111 -- binary representation of 79
  n1 * n2 = product :=
by
  sorry

end binary_mul_1101_111_eq_1001111_l1765_176567


namespace grandfather_time_difference_l1765_176547

-- Definitions based on the conditions
def treadmill_days : ℕ := 4
def miles_per_day : ℕ := 2
def monday_speed : ℕ := 6
def tuesday_speed : ℕ := 3
def wednesday_speed : ℕ := 4
def thursday_speed : ℕ := 3
def walk_speed : ℕ := 3

-- The theorem statement
theorem grandfather_time_difference :
  let monday_time := (miles_per_day : ℚ) / monday_speed
  let tuesday_time := (miles_per_day : ℚ) / tuesday_speed
  let wednesday_time := (miles_per_day : ℚ) / wednesday_speed
  let thursday_time := (miles_per_day : ℚ) / thursday_speed
  let actual_total_time := monday_time + tuesday_time + wednesday_time + thursday_time
  let walk_total_time := (treadmill_days * miles_per_day : ℚ) / walk_speed
  (walk_total_time - actual_total_time) * 60 = 80 := sorry

end grandfather_time_difference_l1765_176547


namespace find_numbers_l1765_176592

/-- Given the sums of three pairs of numbers, we prove the individual numbers. -/
theorem find_numbers (x y z : ℕ) (h1 : x + y = 40) (h2 : y + z = 50) (h3 : z + x = 70) :
  x = 30 ∧ y = 10 ∧ z = 40 :=
by
  sorry

end find_numbers_l1765_176592


namespace solve_expression_l1765_176576

theorem solve_expression (x : ℝ) (h : 5 * x - 7 = 15 * x + 13) : 3 * (x + 4) = 6 :=
sorry

end solve_expression_l1765_176576


namespace train_length_proof_l1765_176582

/-- Given a train's speed of 45 km/hr, time to cross a bridge of 30 seconds, and the bridge length of 225 meters, prove that the length of the train is 150 meters. -/
theorem train_length_proof (speed_km_hr : ℝ) (time_sec : ℝ) (bridge_length_m : ℝ) (train_length_m : ℝ)
    (h_speed : speed_km_hr = 45) (h_time : time_sec = 30) (h_bridge_length : bridge_length_m = 225) :
  train_length_m = 150 :=
by
  sorry

end train_length_proof_l1765_176582


namespace converse_proposition_l1765_176523

theorem converse_proposition (a b c : ℝ) (h : c ≠ 0) :
  a * c^2 > b * c^2 → a > b :=
by
  sorry

end converse_proposition_l1765_176523


namespace tax_rate_l1765_176579

noncomputable def payroll_tax : Float := 300000
noncomputable def tax_paid : Float := 200
noncomputable def tax_threshold : Float := 200000

theorem tax_rate (tax_rate : Float) : 
  (payroll_tax - tax_threshold) * tax_rate = tax_paid → tax_rate = 0.002 := 
by
  sorry

end tax_rate_l1765_176579


namespace smallest_n_l1765_176596

theorem smallest_n (n : ℕ) (h : 503 * n % 48 = 1019 * n % 48) : n = 4 := by
  sorry

end smallest_n_l1765_176596


namespace four_digit_number_sum_l1765_176597

theorem four_digit_number_sum (x y z w : ℕ) (h1 : 1001 * x + 101 * y + 11 * z + 2 * w = 2003)
  (h2 : x = 1) : (x = 1 ∧ y = 9 ∧ z = 7 ∧ w = 8) ↔ (1000 * x + 100 * y + 10 * z + w = 1978) :=
by sorry

end four_digit_number_sum_l1765_176597


namespace expression_value_l1765_176557

theorem expression_value (a b : ℕ) (h₁ : a = 2023) (h₂ : b = 2020) :
  ((
     (3 / (a - b) + (3 * a) / (a^3 - b^3) * ((a^2 + a * b + b^2) / (a + b))) * ((2 * a + b) / (a^2 + 2 * a * b + b^2))
  ) * (3 / (a + b))) = 3 :=
by
  -- Use the provided conditions
  rw [h₁, h₂]
  -- Execute the following steps as per the mathematical solution steps 
  sorry

end expression_value_l1765_176557


namespace smallest_pos_int_terminating_decimal_with_9_l1765_176540

theorem smallest_pos_int_terminating_decimal_with_9 : ∃ n : ℕ, (∃ m k : ℕ, n = 2^m * 5^k ∧ (∃ d : ℕ, d ∈ n.digits 10 ∧ d = 9)) ∧ n = 4096 :=
by {
    sorry
}

end smallest_pos_int_terminating_decimal_with_9_l1765_176540


namespace total_books_in_library_l1765_176570

theorem total_books_in_library :
  ∃ (total_books : ℕ),
  (∀ (books_per_floor : ℕ), books_per_floor - 2 = 20 → 
  total_books = (28 * 6 * books_per_floor)) ∧ total_books = 3696 :=
by
  sorry

end total_books_in_library_l1765_176570


namespace simplify_expression_l1765_176508

theorem simplify_expression (a : ℝ) : 
  ( (a^(16 / 8))^(1 / 4) )^3 * ( (a^(16 / 4))^(1 / 8) )^3 = a^3 := by
  sorry

end simplify_expression_l1765_176508


namespace tram_length_proof_l1765_176551
-- Import the necessary library

-- Define the conditions
def tram_length : ℕ := 32 -- The length of the tram we want to prove

-- The main theorem to be stated
theorem tram_length_proof (L : ℕ) (v : ℕ) 
  (h1 : v = L / 4)  -- The tram passed by Misha in 4 seconds
  (h2 : v = (L + 64) / 12)  -- The tram passed through a tunnel of 64 meters in 12 seconds
  : L = tram_length :=
by
  sorry

end tram_length_proof_l1765_176551


namespace number_of_positive_integers_with_erased_digit_decreased_by_nine_times_l1765_176574

theorem number_of_positive_integers_with_erased_digit_decreased_by_nine_times : 
  ∃ n : ℕ, 
  ∀ (m a k : ℕ),
  (m + 10^k * a + 10^(k + 1) * n = 9 * (m + 10^k * n)) → 
  m < 10^k ∧ n > 0 ∧ n < m ∧  m ≠ 0 → 
  (m + 10^k * n  = 9 * (m - a) ) ∧ 
  (m % 10 = 5 ∨ m % 10 = 0) → 
  n = 28 :=
by
  sorry

end number_of_positive_integers_with_erased_digit_decreased_by_nine_times_l1765_176574


namespace min_value_a_plus_2b_minus_3c_l1765_176538

theorem min_value_a_plus_2b_minus_3c
  (a b c : ℝ)
  (h : ∀ (x y : ℝ), x + 2 * y - 3 ≤ a * x + b * y + c ∧ a * x + b * y + c ≤ x + 2 * y + 3) :
  ∃ m : ℝ, m = a + 2 * b - 3 * c ∧ m = -4 :=
by
  sorry

end min_value_a_plus_2b_minus_3c_l1765_176538


namespace identical_solutions_of_quadratic_linear_l1765_176539

theorem identical_solutions_of_quadratic_linear (k : ℝ) :
  (∃ x : ℝ, x^2 = 4 * x + k ∧ x^2 = 4 * x + k) ↔ k = -4 :=
by
  sorry

end identical_solutions_of_quadratic_linear_l1765_176539


namespace find_m_value_l1765_176554

def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem find_m_value (m : ℝ) 
  (h : dot_product (2 * m - 1, 3) (1, -1) = 2) : 
  m = 3 := by
  sorry

end find_m_value_l1765_176554


namespace gcd_72_168_gcd_98_280_f_at_3_l1765_176527

/-- 
Prove that the GCD of 72 and 168 using the method of mutual subtraction is 24.
-/
theorem gcd_72_168 : Nat.gcd 72 168 = 24 :=
sorry

/-- 
Prove that the GCD of 98 and 280 using the Euclidean algorithm is 14.
-/
theorem gcd_98_280 : Nat.gcd 98 280 = 14 :=
sorry

/-- 
Prove that the value of f(3) where f(x) = x^5 + x^3 + x^2 + x + 1 is 283 using Horner's method.
-/
def f (x : ℕ) : ℕ := x^5 + x^3 + x^2 + x + 1

theorem f_at_3 : f 3 = 283 :=
sorry

end gcd_72_168_gcd_98_280_f_at_3_l1765_176527


namespace combined_share_of_A_and_C_l1765_176532

-- Definitions based on the conditions
def total_money : Float := 15800
def charity_investment : Float := 0.10 * total_money
def savings_investment : Float := 0.08 * total_money
def remaining_money : Float := total_money - charity_investment - savings_investment

def ratio_A : Nat := 5
def ratio_B : Nat := 9
def ratio_C : Nat := 6
def ratio_D : Nat := 5
def sum_of_ratios : Nat := ratio_A + ratio_B + ratio_C + ratio_D

def share_A : Float := (ratio_A.toFloat / sum_of_ratios.toFloat) * remaining_money
def share_C : Float := (ratio_C.toFloat / sum_of_ratios.toFloat) * remaining_money
def combined_share_A_C : Float := share_A + share_C

-- Statement to be proven
theorem combined_share_of_A_and_C : combined_share_A_C = 5700.64 := by
  sorry

end combined_share_of_A_and_C_l1765_176532


namespace milk_fraction_in_cup1_is_one_third_l1765_176590

-- Define the initial state of the cups
structure CupsState where
  cup1_tea : ℚ  -- amount of tea in cup1
  cup1_milk : ℚ -- amount of milk in cup1
  cup2_tea : ℚ  -- amount of tea in cup2
  cup2_milk : ℚ -- amount of milk in cup2

def initial_cups_state : CupsState := {
  cup1_tea := 8,
  cup1_milk := 0,
  cup2_tea := 0,
  cup2_milk := 8
}

-- Function to transfer a fraction of tea from cup 1 to cup 2
def transfer_tea (s : CupsState) (frac : ℚ) : CupsState := {
  cup1_tea := s.cup1_tea * (1 - frac),
  cup1_milk := s.cup1_milk,
  cup2_tea := s.cup2_tea + s.cup1_tea * frac,
  cup2_milk := s.cup2_milk
}

-- Function to transfer a fraction of the mixture from cup 2 to cup 1
def transfer_mixture (s : CupsState) (frac : ℚ) : CupsState := {
  cup1_tea := s.cup1_tea + (frac * s.cup2_tea),
  cup1_milk := s.cup1_milk + (frac * s.cup2_milk),
  cup2_tea := s.cup2_tea * (1 - frac),
  cup2_milk := s.cup2_milk * (1 - frac)
}

-- Define the state after each transfer
def state_after_tea_transfer := transfer_tea initial_cups_state (1 / 4)
def final_state := transfer_mixture state_after_tea_transfer (1 / 3)

-- Prove the fraction of milk in the first cup is 1/3
theorem milk_fraction_in_cup1_is_one_third : 
  (final_state.cup1_milk / (final_state.cup1_tea + final_state.cup1_milk)) = 1 / 3 :=
by
  -- skipped proof
  sorry

end milk_fraction_in_cup1_is_one_third_l1765_176590


namespace cliff_shiny_igneous_l1765_176506

variables (I S : ℕ)

theorem cliff_shiny_igneous :
  I = S / 2 ∧ I + S = 270 → I / 3 = 30 := 
by
  intro h
  sorry

end cliff_shiny_igneous_l1765_176506


namespace george_painting_combinations_l1765_176558

namespace Combinations

/-- George's painting problem -/
theorem george_painting_combinations :
  let colors := 10
  let colors_to_pick := 3
  let textures := 2
  ((colors) * (colors - 1) * (colors - 2) / (colors_to_pick * (colors_to_pick - 1) * 1)) * (textures ^ colors_to_pick) = 960 :=
by
  sorry

end Combinations

end george_painting_combinations_l1765_176558


namespace probability_white_given_popped_is_7_over_12_l1765_176578

noncomputable def probability_white_given_popped : ℚ :=
  let P_W := 0.4
  let P_Y := 0.4
  let P_R := 0.2
  let P_popped_given_W := 0.7
  let P_popped_given_Y := 0.5
  let P_popped_given_R := 0
  let P_popped := P_popped_given_W * P_W + P_popped_given_Y * P_Y + P_popped_given_R * P_R
  (P_popped_given_W * P_W) / P_popped

theorem probability_white_given_popped_is_7_over_12 : probability_white_given_popped = 7 / 12 := 
  by
    sorry

end probability_white_given_popped_is_7_over_12_l1765_176578


namespace rectangles_with_trapezoid_area_l1765_176509

-- Define the necessary conditions
def small_square_area : ℝ := 1
def total_squares : ℕ := 12
def rows : ℕ := 4
def columns : ℕ := 3
def trapezoid_area : ℝ := 3

-- Statement of the proof problem
theorem rectangles_with_trapezoid_area :
  (∀ rows columns : ℕ, rows * columns = total_squares) →
  (∀ area : ℝ, area = small_square_area) →
  (∀ trapezoid_area : ℝ, trapezoid_area = 3) →
  (rows = 4) →
  (columns = 3) →
  ∃ rectangles : ℕ, rectangles = 10 :=
by
  sorry

end rectangles_with_trapezoid_area_l1765_176509


namespace highest_of_seven_consecutive_with_average_33_l1765_176520

theorem highest_of_seven_consecutive_with_average_33 (x : ℤ) 
    (h : (x - 3 + x - 2 + x - 1 + x + x + 1 + x + 2 + x + 3) / 7 = 33) : 
    x + 3 = 36 := 
sorry

end highest_of_seven_consecutive_with_average_33_l1765_176520


namespace intersection_of_lines_l1765_176534

theorem intersection_of_lines : 
  ∃ (x y : ℚ), 5 * x - 2 * y = 8 ∧ 6 * x + 3 * y = 21 ∧ x = 22 / 9 ∧ y = 19 / 9 :=
by 
  sorry

end intersection_of_lines_l1765_176534


namespace smallest_positive_n_l1765_176501

theorem smallest_positive_n (n : ℕ) (h : 77 * n ≡ 308 [MOD 385]) : n = 4 :=
sorry

end smallest_positive_n_l1765_176501


namespace find_smaller_number_l1765_176586

theorem find_smaller_number (x y : ℤ) (h1 : x + y = 60) (h2 : x - y = 8) : y = 26 :=
by
  sorry

end find_smaller_number_l1765_176586


namespace range_of_m_l1765_176515

noncomputable def f (m x : ℝ) : ℝ := m * x^2 - 2 * m * x + m + 3
noncomputable def g (x : ℝ) : ℝ := 2^(x - 2)

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, f m x < 0 ∨ g x < 0) ↔ -4 < m ∧ m < 0 :=
by sorry

end range_of_m_l1765_176515


namespace distance_between_stripes_l1765_176563

/-
Problem statement:
Given:
1. The street has parallel curbs 30 feet apart.
2. The length of the curb between the stripes is 10 feet.
3. Each stripe is 60 feet long.

Prove:
The distance between the stripes is 5 feet.
-/

-- Definitions:
def distance_between_curbs : ℝ := 30
def length_between_stripes_on_curb : ℝ := 10
def length_of_each_stripe : ℝ := 60

-- Theorem statement:
theorem distance_between_stripes :
  ∃ d : ℝ, (length_between_stripes_on_curb * distance_between_curbs = length_of_each_stripe * d) ∧ d = 5 :=
by
  sorry

end distance_between_stripes_l1765_176563


namespace mean_score_l1765_176594

theorem mean_score (M SD : ℝ) (h1 : 58 = M - 2 * SD) (h2 : 98 = M + 3 * SD) : M = 74 :=
by
  sorry

end mean_score_l1765_176594


namespace ian_saves_per_day_l1765_176561

-- Let us define the given conditions
def total_saved : ℝ := 0.40 -- Ian saved a total of $0.40
def days : ℕ := 40 -- Ian saved for 40 days

-- Now, we need to prove that Ian saved 0.01 dollars/day
theorem ian_saves_per_day (h : total_saved = 0.40 ∧ days = 40) : total_saved / days = 0.01 :=
by
  sorry

end ian_saves_per_day_l1765_176561


namespace length_of_room_l1765_176513

noncomputable def room_length (width cost rate : ℝ) : ℝ :=
  let area := cost / rate
  area / width

theorem length_of_room :
  room_length 4.75 38475 900 = 9 := by
  sorry

end length_of_room_l1765_176513


namespace sum_consecutive_numbers_last_digit_diff_l1765_176500

theorem sum_consecutive_numbers_last_digit_diff (a : ℕ) : 
    (2015 * (a + 1007) % 10) ≠ (2019 * (a + 3024) % 10) := 
by 
  sorry

end sum_consecutive_numbers_last_digit_diff_l1765_176500


namespace center_temperature_l1765_176598

-- Define the conditions as a structure
structure SquareSheet (f : ℝ × ℝ → ℝ) :=
  (temp_0: ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → f (x, 0) = 0 ∧ f (0, x) = 0 ∧ f (1, x) = 0)
  (temp_100: ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → f (x, 1) = 100)
  (no_radiation_loss: True) -- Just a placeholder since this condition is theoretical in nature

-- Define the claim as a theorem
theorem center_temperature (f : ℝ × ℝ → ℝ) (h : SquareSheet f) : f (0.5, 0.5) = 25 :=
by
  sorry -- Proof is not required and skipped

end center_temperature_l1765_176598


namespace find_fg_satisfy_l1765_176560

noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x) / 2
noncomputable def g (x : ℝ) (c : ℝ) : ℝ := (Real.sin x - Real.cos x) / 2 + c

theorem find_fg_satisfy (c : ℝ) : ∀ x y : ℝ,
  Real.sin x + Real.cos y = f x + f y + g x c - g y c := 
by 
  intros;
  rw [f, g, g, f];
  sorry

end find_fg_satisfy_l1765_176560


namespace inequality_selection_l1765_176583

theorem inequality_selection (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  1/a + 4/b ≥ 9/(a + b) :=
sorry

end inequality_selection_l1765_176583


namespace fraction_human_habitable_surface_l1765_176537

variable (fraction_water_coverage : ℚ)
variable (fraction_inhabitable_remaining_land : ℚ)
variable (fraction_reserved_for_agriculture : ℚ)

def fraction_inhabitable_land (f_water : ℚ) (f_inhabitable : ℚ) : ℚ :=
  (1 - f_water) * f_inhabitable

def fraction_habitable_land (f_inhabitable_land : ℚ) (f_reserved : ℚ) : ℚ :=
  f_inhabitable_land * (1 - f_reserved)

theorem fraction_human_habitable_surface 
  (h1 : fraction_water_coverage = 3/5)
  (h2 : fraction_inhabitable_remaining_land = 2/3)
  (h3 : fraction_reserved_for_agriculture = 1/2) :
  fraction_habitable_land 
    (fraction_inhabitable_land fraction_water_coverage fraction_inhabitable_remaining_land)
    fraction_reserved_for_agriculture = 2/15 :=
by {
  sorry
}

end fraction_human_habitable_surface_l1765_176537


namespace coordinates_of_A_l1765_176544

-- Definition of the point A with coordinates (-1, 3)
def point_A : ℝ × ℝ := (-1, 3)

-- Statement that the coordinates of point A with respect to the origin are (-1, 3)
theorem coordinates_of_A : point_A = (-1, 3) := by
  sorry

end coordinates_of_A_l1765_176544


namespace find_xy_l1765_176516

variable {x y : ℝ}

theorem find_xy (h₁ : x + y = 10) (h₂ : x^3 + y^3 = 370) : x * y = 21 :=
by
  sorry

end find_xy_l1765_176516


namespace direct_proportion_point_l1765_176533

theorem direct_proportion_point (k : ℝ) (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : y₁ = k * x₁) (hx₁ : x₁ = -1) (hy₁ : y₁ = 2) (hx₂ : x₂ = 1) (hy₂ : y₂ = -2) 
  : y₂ = k * x₂ := 
by
  -- sorry will skip the proof
  sorry

end direct_proportion_point_l1765_176533


namespace negation_of_proposition_l1765_176572

theorem negation_of_proposition (x y : ℝ) :
  (¬ (x + y = 1 → xy ≤ 1)) ↔ (x + y ≠ 1 → xy > 1) :=
by 
  sorry

end negation_of_proposition_l1765_176572


namespace compute_difference_of_squares_l1765_176571

theorem compute_difference_of_squares :
    75^2 - 25^2 = 5000 :=
by
  sorry

end compute_difference_of_squares_l1765_176571


namespace polynomial_roots_a_ge_five_l1765_176555

theorem polynomial_roots_a_ge_five (a b c : ℤ) (h_a_pos : a > 0)
    (h_distinct_roots : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 0 < x₁ ∧ x₁ < 1 ∧ 0 < x₂ ∧ x₂ < 1 ∧ 
        a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0) : a ≥ 5 := sorry

end polynomial_roots_a_ge_five_l1765_176555


namespace comb_5_1_eq_5_l1765_176531

theorem comb_5_1_eq_5 : Nat.choose 5 1 = 5 :=
by
  sorry

end comb_5_1_eq_5_l1765_176531


namespace overall_percentage_badminton_l1765_176541

theorem overall_percentage_badminton (N S : ℕ) (pN pS : ℝ) :
  N = 1500 → S = 1800 → pN = 0.30 → pS = 0.35 → 
  ( (N * pN + S * pS) / (N + S) ) * 100 = 33 := 
by
  intros hN hS hpN hpS
  sorry

end overall_percentage_badminton_l1765_176541


namespace part_I_part_II_l1765_176536

namespace ArithmeticGeometricSequences

-- Definitions of sequences and their properties
def a1 : ℕ := 1
def b1 : ℕ := 2
def b (n : ℕ) : ℕ := 2 * 3 ^ (n - 1) -- General term of the geometric sequence

-- Definitions from given conditions
def a (n : ℕ) : ℕ := 3 * n - 2 -- General term of the arithmetic sequence

-- Sum of the first n terms of the geometric sequence
def S (n : ℕ) : ℕ := if n = 0 then 0 else 3 ^ n - 1

-- Theorem statement
theorem part_I (n : ℕ) : 
  (a1 = 1) ∧ 
  (b1 = 2) ∧ 
  (∀ n > 0, b n > 0) ∧ 
  (∀ b2 : ℕ, 2 * (1 + b2 / 2) = 2 + b2) ∧ 
  (∀ b2 a2 : ℕ, (1 + b2 / 2)^2 = b2 * ((a 3) + 2)) →
  (a n = 3 * n - 2) ∧ 
  (b n = 2 * 3 ^ (n - 1)) :=
  sorry

theorem part_II (n : ℕ) (m : ℝ) :
  (a1 = 1) ∧ 
  (b1 = 2) ∧ 
  (∀ n > 0, b n > 0) ∧ 
  (∀ b2 : ℕ, 2 * (1 + b2 / 2) = 2 + b2) ∧ 
  (∀ b2 a2 : ℕ, (1 + b2 / 2)^2 = b2 * ((a 3) + 2)) → 
  (∀ n > 0, S n + a n > m) → 
  (m < 3) :=
  sorry

end ArithmeticGeometricSequences

end part_I_part_II_l1765_176536


namespace select_best_athlete_l1765_176521

theorem select_best_athlete
  (avg_A avg_B avg_C avg_D: ℝ)
  (var_A var_B var_C var_D: ℝ)
  (h_avg_A: avg_A = 185)
  (h_avg_B: avg_B = 180)
  (h_avg_C: avg_C = 185)
  (h_avg_D: avg_D = 180)
  (h_var_A: var_A = 3.6)
  (h_var_B: var_B = 3.6)
  (h_var_C: var_C = 7.4)
  (h_var_D: var_D = 8.1) :
  (avg_A > avg_B ∧ avg_A > avg_D ∧ var_A < var_C) →
  (avg_A = 185 ∧ var_A = 3.6) :=
by
  sorry

end select_best_athlete_l1765_176521


namespace total_new_bottles_l1765_176588

theorem total_new_bottles (initial_bottles : ℕ) (recycle_ratio : ℕ) (bonus_ratio : ℕ) (final_bottles : ℕ) :
  initial_bottles = 625 →
  recycle_ratio = 5 →
  bonus_ratio = 20 →
  final_bottles = 163 :=
by {
  sorry -- Proof goes here
}

end total_new_bottles_l1765_176588


namespace chord_length_eq_l1765_176502

noncomputable def length_of_chord (radius : ℝ) (distance_to_chord : ℝ) : ℝ :=
  2 * Real.sqrt (radius^2 - distance_to_chord^2)

theorem chord_length_eq {radius distance_to_chord : ℝ} (h_radius : radius = 5) (h_distance : distance_to_chord = 4) :
  length_of_chord radius distance_to_chord = 6 :=
by
  sorry

end chord_length_eq_l1765_176502


namespace totalWheelsInStorageArea_l1765_176593

def numberOfBicycles := 24
def numberOfTricycles := 14
def wheelsPerBicycle := 2
def wheelsPerTricycle := 3

theorem totalWheelsInStorageArea :
  numberOfBicycles * wheelsPerBicycle + numberOfTricycles * wheelsPerTricycle = 90 :=
by
  sorry

end totalWheelsInStorageArea_l1765_176593


namespace max_sum_of_ABC_l1765_176565

/-- Theorem: The maximum value of A + B + C for distinct positive integers A, B, and C such that A * B * C = 2023 is 297. -/
theorem max_sum_of_ABC (A B C : ℕ) (h1 : A ≠ B) (h2 : B ≠ C) (h3 : A ≠ C) (h4 : A * B * C = 2023) :
  A + B + C ≤ 297 :=
sorry

end max_sum_of_ABC_l1765_176565


namespace actual_area_of_lawn_l1765_176581

-- Definitions and conditions
variable (blueprint_area : ℝ)
variable (side_on_blueprint : ℝ)
variable (actual_side_length : ℝ)

-- Given conditions
def blueprint_conditions := 
  blueprint_area = 300 ∧ 
  side_on_blueprint = 5 ∧ 
  actual_side_length = 15

-- Prove the actual area of the lawn
theorem actual_area_of_lawn (blueprint_area : ℝ) (side_on_blueprint : ℝ) (actual_side_length : ℝ) (x : ℝ) :
  blueprint_conditions blueprint_area side_on_blueprint actual_side_length →
  (x = 27000000 ∧ x / 10000 = 2700) :=
by
  sorry

end actual_area_of_lawn_l1765_176581


namespace functions_of_same_family_count_l1765_176504

theorem functions_of_same_family_count : 
  (∃ (y : ℝ → ℝ), ∀ x, y x = x^2) ∧ 
  (∃ (range_set : Set ℝ), range_set = {1, 2}) → 
  ∃ n, n = 9 :=
by
  sorry

end functions_of_same_family_count_l1765_176504


namespace find_consecutive_numbers_l1765_176566

theorem find_consecutive_numbers (a b c : ℕ) (h1 : a + 1 = b) (h2 : b + 1 = c)
    (h_lcm : Nat.lcm a (Nat.lcm b c) = 660) : a = 10 ∧ b = 11 ∧ c = 12 := 
    sorry

end find_consecutive_numbers_l1765_176566


namespace solve_for_x_l1765_176526

theorem solve_for_x (x : ℚ) : 
  x + 5 / 6 = 11 / 18 - 2 / 9 → x = -4 / 9 := 
by
  intro h
  sorry

end solve_for_x_l1765_176526


namespace lines_parallel_iff_m_eq_1_l1765_176585

-- Define the two lines l1 and l2
def l1 (m : ℝ) (x y : ℝ) : Prop := x + (1 + m) * y = 2 - m
def l2 (m : ℝ) (x y : ℝ) : Prop := 2 * m * x + 4 * y = -16

-- Parallel lines condition
def parallel_condition (m : ℝ) : Prop := (1 * 4 - 2 * m * (1 + m) = 0) ∧ (1 * 16 - 2 * m * (m - 2) ≠ 0)

-- The theorem to prove
theorem lines_parallel_iff_m_eq_1 (m : ℝ) : l1 m = l2 m → parallel_condition m → m = 1 :=
by 
  sorry

end lines_parallel_iff_m_eq_1_l1765_176585


namespace moles_of_SO2_formed_l1765_176512

variable (n_NaHSO3 n_HCl n_SO2 : ℕ)

/--
The reaction between sodium bisulfite (NaHSO3) and hydrochloric acid (HCl) is:
NaHSO3 + HCl → NaCl + H2O + SO2
Given 2 moles of NaHSO3 and 2 moles of HCl, prove that the number of moles of SO2 formed is 2.
-/
theorem moles_of_SO2_formed :
  (n_NaHSO3 = 2) →
  (n_HCl = 2) →
  (∀ (n : ℕ), (n_NaHSO3 = n) → (n_HCl = n) → (n_SO2 = n)) →
  n_SO2 = 2 :=
by 
  intros hNaHSO3 hHCl hReaction
  exact hReaction 2 hNaHSO3 hHCl

end moles_of_SO2_formed_l1765_176512


namespace simple_interest_rate_l1765_176577

theorem simple_interest_rate (SI : ℝ) (P : ℝ) (T : ℝ) (R : ℝ)
  (h1 : SI = 130) (h2 : P = 780) (h3 : T = 4) :
  R = 4.17 :=
sorry

end simple_interest_rate_l1765_176577


namespace multiply_polynomials_l1765_176535

def polynomial_multiplication (x : ℝ) : Prop :=
  (x^4 + 24*x^2 + 576) * (x^2 - 24) = x^6 - 13824

theorem multiply_polynomials (x : ℝ) : polynomial_multiplication x :=
by
  sorry

end multiply_polynomials_l1765_176535


namespace lesson_duration_tuesday_l1765_176522

theorem lesson_duration_tuesday
  (monday_lessons : ℕ)
  (monday_duration : ℕ)
  (tuesday_lessons : ℕ)
  (wednesday_multiplier : ℕ)
  (total_time : ℕ)
  (monday_hours : ℕ)
  (tuesday_hours : ℕ)
  (wednesday_hours : ℕ)
  (H1 : monday_lessons = 6)
  (H2 : monday_duration = 30)
  (H3 : tuesday_lessons = 3)
  (H4 : wednesday_multiplier = 2)
  (H5 : total_time = 12)
  (H6 : monday_hours = monday_lessons * monday_duration / 60)
  (H7 : tuesday_hours = tuesday_lessons * T)
  (H8 : wednesday_hours = wednesday_multiplier * tuesday_hours)
  (H9 : monday_hours + tuesday_hours + wednesday_hours = total_time) :
  T = 1 := by
  sorry

end lesson_duration_tuesday_l1765_176522


namespace symmetric_circle_eq_a_l1765_176530

theorem symmetric_circle_eq_a :
  ∀ (a : ℝ), (∀ x y : ℝ, (x^2 + y^2 - a * x + 2 * y + 1 = 0) ↔ (∃ x y : ℝ, (x - y = 1) ∧ ( x^2 + y^2 = 1))) → a = 2 :=
by
  sorry

end symmetric_circle_eq_a_l1765_176530


namespace motorcyclist_average_speed_l1765_176503

theorem motorcyclist_average_speed :
  ∀ (t : ℝ), 120 / t = 60 * 3 → 
  3 * t / 4 = 45 :=
by
  sorry

end motorcyclist_average_speed_l1765_176503


namespace right_triangle_hypotenuse_unique_l1765_176568

theorem right_triangle_hypotenuse_unique :
  ∃ (a b c : ℚ) (d e : ℕ), 
    (c^2 = a^2 + b^2) ∧
    (a = 10 * e + d) ∧
    (c = 10 * d + e) ∧
    (d + e = 11) ∧
    (d ≠ e) ∧
    (a = 56) ∧
    (b = 33) ∧
    (c = 65) :=
by {
  sorry
}

end right_triangle_hypotenuse_unique_l1765_176568


namespace value_of_x_squared_plus_9y_squared_l1765_176525

theorem value_of_x_squared_plus_9y_squared (x y : ℝ) (h1 : x - 3 * y = 3) (h2 : x * y = -9) : x^2 + 9 * y^2 = -45 :=
by
  sorry

end value_of_x_squared_plus_9y_squared_l1765_176525


namespace question_correct_statements_l1765_176595

noncomputable def f : ℝ → ℝ := sorry

axiom additivity (f : ℝ → ℝ) : ∀ x y : ℝ, f (x + y) = f x + f y
axiom periodicity (f : ℝ → ℝ) : f 2 = 0

theorem question_correct_statements : 
  (∀ x : ℝ, f (x + 2) = f x) ∧ -- ensuring the function is periodic
  (∀ x : ℝ, f x = -f (-x)) ∧ -- ensuring the function is odd
  (∀ x : ℝ, f (x+2) = -f (-x)) :=  -- ensuring symmetry about point (1,0)
by
  -- We'll prove this using the conditions given and properties derived from it
  sorry 

end question_correct_statements_l1765_176595


namespace expand_expression_l1765_176562

theorem expand_expression (x y : ℝ) :
  (x + 3) * (4 * x - 5 * y) = 4 * x ^ 2 - 5 * x * y + 12 * x - 15 * y :=
by
  sorry

end expand_expression_l1765_176562


namespace range_of_m_l1765_176542

-- Definitions of Propositions p and q
def Proposition_p (m : ℝ) : Prop :=
  (m^2 - 4 > 0) ∧ (-m > 0) ∧ (1 > 0)  -- where x₁ + x₂ = -m > 0 and x₁x₂ = 1

def Proposition_q (m : ℝ) : Prop :=
  16 * (m + 2)^2 - 16 < 0  -- discriminant of 4x^2 + 4(m+2)x + 1 = 0 is less than 0

-- Given: "Proposition p or Proposition q" is true
def given (m : ℝ) : Prop :=
  Proposition_p m ∨ Proposition_q m

-- Prove: Range of values for m is (-∞, -1)
theorem range_of_m (m : ℝ) (h : given m) : m < -1 :=
sorry

end range_of_m_l1765_176542


namespace g_is_correct_l1765_176599

-- Define the given polynomial equation
def poly_lhs (x : ℝ) : ℝ := 2 * x^5 - x^3 + 4 * x^2 + 3 * x - 5
def poly_rhs (x : ℝ) : ℝ := 7 * x^3 - 4 * x + 2

-- Define the function g(x)
def g (x : ℝ) : ℝ := -2 * x^5 + 6 * x^3 - 4 * x^2 - x + 7

-- The theorem to be proven
theorem g_is_correct : ∀ x : ℝ, poly_lhs x + g x = poly_rhs x :=
by
  intro x
  unfold poly_lhs poly_rhs g
  sorry

end g_is_correct_l1765_176599


namespace hexagon_perimeter_l1765_176514

-- Definitions of the conditions
def side_length : ℕ := 5
def number_of_sides : ℕ := 6

-- The perimeter of the hexagon
def perimeter : ℕ := side_length * number_of_sides

-- Proof statement
theorem hexagon_perimeter : perimeter = 30 :=
by
  sorry

end hexagon_perimeter_l1765_176514


namespace find_m_abc_inequality_l1765_176569

-- Define properties and the theorem for the first problem
def f (x m : ℝ) := m - |x - 2|

theorem find_m (m : ℝ) : (∀ x, f (x + 2) m ≥ 0 ↔ x ∈ Set.Icc (-1 : ℝ) 1) → m = 1 := by
  intros h
  sorry

-- Define properties and the theorem for the second problem
theorem abc_inequality (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) :
  (1 / a + 1 / (2 * b) + 1 / (3 * c) = 1) → (a + 2 * b + 3 * c ≥ 9) := by
  intros h
  sorry

end find_m_abc_inequality_l1765_176569


namespace set_intersection_complement_l1765_176584

theorem set_intersection_complement (U : Set ℝ) (A B : Set ℝ) 
  (hU : U = Set.univ) 
  (hA : ∀ x : ℝ, A x ↔ x^2 - x - 6 ≤ 0) 
  (hB : ∀ x : ℝ, B x ↔ Real.log x / Real.log (1/2) ≥ -1) :
  A ∩ (U \ B) = (Set.Icc (-2 : ℝ) 0 ∪ Set.Ioc 2 3) :=
by
  ext x
  -- Proof here would follow
  sorry

end set_intersection_complement_l1765_176584


namespace polynomial_coeff_sum_l1765_176548

theorem polynomial_coeff_sum (A B C D : ℤ) 
  (h : ∀ x : ℤ, (x + 3) * (4 * x^2 - 2 * x + 7) = A * x^3 + B * x^2 + C * x + D) : 
  A + B + C + D = 36 :=
by 
  sorry

end polynomial_coeff_sum_l1765_176548


namespace problem_l1765_176580

theorem problem (m : ℝ) (h : m^2 + 3 * m = -1) : m - 1 / (m + 1) = -2 :=
by
  sorry

end problem_l1765_176580


namespace P_sufficient_for_Q_P_not_necessary_for_Q_l1765_176511

variable (x : ℝ)
def P : Prop := x >= 0
def Q : Prop := 2 * x + 1 / (2 * x + 1) >= 1

theorem P_sufficient_for_Q : P x -> Q x := 
by sorry

theorem P_not_necessary_for_Q : ¬ (Q x -> P x) := 
by sorry

end P_sufficient_for_Q_P_not_necessary_for_Q_l1765_176511


namespace mary_income_percent_of_juan_l1765_176553

variable (J : ℝ)
variable (T : ℝ)
variable (M : ℝ)

-- Conditions
def tim_income := T = 0.60 * J
def mary_income := M = 1.40 * T

-- Theorem to prove that Mary's income is 84 percent of Juan's income
theorem mary_income_percent_of_juan : tim_income J T → mary_income T M → M = 0.84 * J :=
by
  sorry

end mary_income_percent_of_juan_l1765_176553


namespace solve_equation_l1765_176510

theorem solve_equation {n k l m : ℕ} (h_l : l > 1) :
  (1 + n^k)^l = 1 + n^m ↔ (n = 2 ∧ k = 1 ∧ l = 2 ∧ m = 3) :=
sorry

end solve_equation_l1765_176510


namespace initial_population_l1765_176589

theorem initial_population (P : ℝ) : 
  (0.9 * P * 0.85 = 2907) → P = 3801 := by
  sorry

end initial_population_l1765_176589


namespace relative_error_comparison_l1765_176564

theorem relative_error_comparison :
  let e₁ := 0.05
  let l₁ := 25.0
  let e₂ := 0.4
  let l₂ := 200.0
  let relative_error (e l : ℝ) : ℝ := (e / l) * 100
  (relative_error e₁ l₁ = relative_error e₂ l₂) :=
by
  sorry

end relative_error_comparison_l1765_176564


namespace pool_capacity_is_800_l1765_176559

-- Definitions for the given problem conditions
def fill_time_all_valves : ℝ := 36
def fill_time_first_valve : ℝ := 180
def fill_time_second_valve : ℝ := 240
def third_valve_more_than_first : ℝ := 30
def third_valve_more_than_second : ℝ := 10
def leak_rate : ℝ := 20

-- Function definition for the capacity of the pool
def capacity (W : ℝ) : Prop :=
  let V1 := W / fill_time_first_valve
  let V2 := W / fill_time_second_valve
  let V3 := (W / fill_time_first_valve) + third_valve_more_than_first
  let effective_rate := V1 + V2 + V3 - leak_rate
  (W / fill_time_all_valves) = effective_rate

-- Proof statement that the capacity of the pool is 800 cubic meters
theorem pool_capacity_is_800 : capacity 800 :=
by
  -- Proof is omitted
  sorry

end pool_capacity_is_800_l1765_176559


namespace arrange_letters_l1765_176543

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem arrange_letters : factorial 7 / (factorial 3 * factorial 2 * factorial 2) = 210 := 
by
  sorry

end arrange_letters_l1765_176543


namespace length_of_platform_is_350_l1765_176518

-- Define the parameters as given in the problem
def train_length : ℕ := 300
def time_to_cross_post : ℕ := 18
def time_to_cross_platform : ℕ := 39

-- Define the speed of the train as a ratio of the length of the train and the time to cross the post
def train_speed : ℚ := train_length / time_to_cross_post

-- Formalize the problem statement: Prove that the length of the platform is 350 meters
theorem length_of_platform_is_350 : ∃ (L : ℕ), (train_speed * time_to_cross_platform) = train_length + L := by
  use 350
  sorry

end length_of_platform_is_350_l1765_176518


namespace no_digit_C_makes_2C4_multiple_of_5_l1765_176507

theorem no_digit_C_makes_2C4_multiple_of_5 : ∀ (C : ℕ), (2 * 100 + C * 10 + 4 ≠ 0 ∨ 2 * 100 + C * 10 + 4 ≠ 5) := 
by 
  intros C
  have h : 4 ≠ 0 := by norm_num
  have h2 : 4 ≠ 5 := by norm_num
  sorry

end no_digit_C_makes_2C4_multiple_of_5_l1765_176507


namespace sum_of_prime_factors_240345_l1765_176519

theorem sum_of_prime_factors_240345 : ∀ {p1 p2 p3 : ℕ}, 
  Prime p1 → Prime p2 → Prime p3 →
  p1 * p2 * p3 = 240345 →
  p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 →
  p1 + p2 + p3 = 16011 :=
by
  intros p1 p2 p3 hp1 hp2 hp3 hprod hdiff
  sorry

end sum_of_prime_factors_240345_l1765_176519


namespace domain_eq_l1765_176524

def domain_of_function :
    Set ℝ := {x | (x - 1 ≥ 0) ∧ (x + 1 > 0)}

theorem domain_eq :
    domain_of_function = {x | x ≥ 1} :=
by
  sorry

end domain_eq_l1765_176524


namespace solve_x_l1765_176529

noncomputable def diamond (a b : ℝ) : ℝ := a / b

axiom diamond_assoc (a b c : ℝ) (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) (c_nonzero : c ≠ 0) : 
  diamond a (diamond b c) = a / (b / c)

axiom diamond_id (a : ℝ) (a_nonzero : a ≠ 0) : diamond a a = 1

theorem solve_x (x : ℝ) (h₁ : 1008 ≠ 0) (h₂ : 12 ≠ 0) (h₃ : x ≠ 0) : diamond 1008 (diamond 12 x) = 50 → x = 25 / 42 :=
by
  sorry

end solve_x_l1765_176529


namespace geometric_sequence_sum_l1765_176505

theorem geometric_sequence_sum (S : ℕ → ℝ) (a₄_to_a₁₂_sum : ℝ):
  (S 3 = 2) → (S 6 = 6) → a₄_to_a₁₂_sum = (S 12 - S 3)  :=
by
  sorry

end geometric_sequence_sum_l1765_176505


namespace a_pow_10_plus_b_pow_10_l1765_176545

theorem a_pow_10_plus_b_pow_10 (a b : ℝ)
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11)
  (hn : ∀ n ≥ 3, a^(n) + b^(n) = (a^(n-1) + b^(n-1)) + (a^(n-2) + b^(n-2))) :
  a^10 + b^10 = 123 :=
by
  sorry

end a_pow_10_plus_b_pow_10_l1765_176545


namespace sufficient_but_not_necessary_for_monotonic_l1765_176546

noncomputable def is_monotonically_increasing (f : ℝ → ℝ) : Prop :=
∀ x y, x ≤ y → f x ≤ f y

noncomputable def is_sufficient_condition (P Q : Prop) : Prop :=
P → Q

noncomputable def is_not_necessary_condition (P Q : Prop) : Prop :=
¬ Q → ¬ P

noncomputable def is_sufficient_but_not_necessary (P Q : Prop) : Prop :=
is_sufficient_condition P Q ∧ is_not_necessary_condition P Q

theorem sufficient_but_not_necessary_for_monotonic (f : ℝ → ℝ) :
  (∀ x, 0 ≤ deriv f x) → is_monotonically_increasing f :=
sorry

end sufficient_but_not_necessary_for_monotonic_l1765_176546


namespace ray_climbs_l1765_176587

theorem ray_climbs (n : ℕ) (h1 : n % 3 = 1) (h2 : n % 5 = 3) (h3 : n % 7 = 1) (h4 : n > 15) : n = 73 :=
sorry

end ray_climbs_l1765_176587


namespace num_of_consec_int_sum_18_l1765_176528

theorem num_of_consec_int_sum_18 : 
  ∃! (a n : ℕ), n ≥ 3 ∧ (n * (2 * a + n - 1)) = 36 :=
sorry

end num_of_consec_int_sum_18_l1765_176528


namespace inequality_proof_l1765_176556

variables (a b c d e f : ℝ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f)
variable (hcond : |sqrt (a * d) - sqrt (b * c)| ≤ 1)

theorem inequality_proof :
  (a * e + b / e) * (c * e + d / e) ≥ (a^2 * f^2 - b^2 / f^2) * (d^2 / f^2 - c^2 * f^2) := by
  sorry

end inequality_proof_l1765_176556


namespace evaluate_function_at_neg_one_l1765_176573

def f (x : ℝ) : ℝ := -2 * x^2 + 1

theorem evaluate_function_at_neg_one : f (-1) = -1 :=
by
  sorry

end evaluate_function_at_neg_one_l1765_176573


namespace polynomial_value_at_n_plus_1_l1765_176550

theorem polynomial_value_at_n_plus_1 
  (f : ℕ → ℝ) 
  (n : ℕ)
  (hdeg : ∃ m, m = n) 
  (hvalues : ∀ k (hk : k ≤ n), f k = k / (k + 1)) : 
  f (n + 1) = (n + 1 + (-1) ^ (n + 1)) / (n + 2) := 
by
  sorry

end polynomial_value_at_n_plus_1_l1765_176550


namespace students_with_green_eyes_l1765_176591

-- Define the variables and given conditions
def total_students : ℕ := 36
def students_with_red_hair (y : ℕ) : ℕ := 3 * y
def students_with_both : ℕ := 12
def students_with_neither : ℕ := 4

-- Define the proof statement
theorem students_with_green_eyes :
  ∃ y : ℕ, 
  (students_with_red_hair y + y - students_with_both + students_with_neither = total_students) ∧
  (students_with_red_hair y ≠ y) → y = 11 :=
by
  sorry

end students_with_green_eyes_l1765_176591
