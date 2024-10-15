import Mathlib

namespace NUMINAMATH_GPT_polynomial_square_b_value_l165_16516

theorem polynomial_square_b_value
  (a b : ℚ)
  (h : ∃ (p q r : ℚ), (x^4 - x^3 + x^2 + a * x + b) = (p * x^2 + q * x + r)^2) :
  b = 9 / 64 :=
sorry

end NUMINAMATH_GPT_polynomial_square_b_value_l165_16516


namespace NUMINAMATH_GPT_original_solution_concentration_l165_16536

variable (C : ℝ) -- Concentration of the original solution as a percentage.
variable (v_orig : ℝ := 12) -- 12 ounces of the original vinegar solution.
variable (w_added : ℝ := 50) -- 50 ounces of water added.
variable (v_final_pct : ℝ := 7) -- Final concentration of 7%.

theorem original_solution_concentration :
  (C / 100 * v_orig = v_final_pct / 100 * (v_orig + w_added)) →
  C = (v_final_pct * (v_orig + w_added)) / v_orig :=
sorry

end NUMINAMATH_GPT_original_solution_concentration_l165_16536


namespace NUMINAMATH_GPT_bowling_tournament_prize_orders_l165_16522
-- Import necessary Lean library

-- Define the conditions
def match_outcome (num_games : ℕ) : ℕ := 2 ^ num_games

-- Theorem statement
theorem bowling_tournament_prize_orders : match_outcome 5 = 32 := by
  -- This is the statement, proof is not required
  sorry

end NUMINAMATH_GPT_bowling_tournament_prize_orders_l165_16522


namespace NUMINAMATH_GPT_cost_of_bananas_l165_16586

/-- We are given that the rate of bananas is $6 per 3 kilograms. -/
def rate_per_3_kg : ℝ := 6

/-- We need to find the cost for 12 kilograms of bananas. -/
def weight_in_kg : ℝ := 12

/-- We are asked to prove that the cost of 12 kilograms of bananas is $24. -/
theorem cost_of_bananas (rate_per_3_kg weight_in_kg : ℝ) :
  (weight_in_kg / 3) * rate_per_3_kg = 24 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_bananas_l165_16586


namespace NUMINAMATH_GPT_sum_of_volumes_is_correct_l165_16528

-- Define the dimensions of the base of the tank
def tank_base_length : ℝ := 44
def tank_base_width : ℝ := 35

-- Define the increase in water height when the train and the car are submerged
def train_water_height_increase : ℝ := 7
def car_water_height_increase : ℝ := 3

-- Calculate the area of the base of the tank
def base_area : ℝ := tank_base_length * tank_base_width

-- Calculate the volumes of the toy train and the toy car
def volume_train : ℝ := base_area * train_water_height_increase
def volume_car : ℝ := base_area * car_water_height_increase

-- Theorem to prove the sum of the volumes is 15400 cubic centimeters
theorem sum_of_volumes_is_correct : volume_train + volume_car = 15400 := by
  sorry

end NUMINAMATH_GPT_sum_of_volumes_is_correct_l165_16528


namespace NUMINAMATH_GPT_solve_m_l165_16595

def f (x : ℝ) := 4 * x ^ 2 - 3 * x + 5
def g (x : ℝ) (m : ℝ) := x ^ 2 - m * x - 8

theorem solve_m : ∃ (m : ℝ), f 8 - g 8 m = 20 ∧ m = -25.5 := by
  sorry

end NUMINAMATH_GPT_solve_m_l165_16595


namespace NUMINAMATH_GPT_solution_greater_iff_l165_16584

variables {c c' d d' : ℝ}
variables (hc : c ≠ 0) (hc' : c' ≠ 0)

theorem solution_greater_iff : (∃ x, x = -d / c) > (∃ x, x = -d' / c') ↔ (d' / c') < (d / c) :=
by sorry

end NUMINAMATH_GPT_solution_greater_iff_l165_16584


namespace NUMINAMATH_GPT_figure4_total_length_l165_16538

-- Define the conditions
def top_segments_sum := 3 + 1 + 1  -- Sum of top segments in Figure 3
def bottom_segment := top_segments_sum -- Bottom segment length in Figure 3
def vertical_segment1 := 10  -- First vertical segment length
def vertical_segment2 := 9  -- Second vertical segment length
def remaining_segment := 1  -- The remaining horizontal segment

-- Total length of remaining segments in Figure 4
theorem figure4_total_length : 
  bottom_segment + vertical_segment1 + vertical_segment2 + remaining_segment = 25 := by
  sorry

end NUMINAMATH_GPT_figure4_total_length_l165_16538


namespace NUMINAMATH_GPT_primes_quadratic_roots_conditions_l165_16546

theorem primes_quadratic_roots_conditions (p q : ℕ)
  (hp : Prime p) (hq : Prime q)
  (h1 : ∃ (x y : ℕ), x ≠ y ∧ x * y = 2 * q ∧ x + y = p) :
  (¬ (∀ (x y : ℕ), x ≠ y ∧ x * y = 2 * q ∧ x + y = p → (x - y) % 2 = 0)) ∧
  (∃ (x : ℕ), x * 2 = 2 * q ∨ x * q = 2 * q ∧ Prime x) ∧
  (¬ Prime (p * p + 2 * q)) ∧
  (Prime (p - q)) :=
by sorry

end NUMINAMATH_GPT_primes_quadratic_roots_conditions_l165_16546


namespace NUMINAMATH_GPT_product_mod_five_remainder_l165_16562

theorem product_mod_five_remainder :
  (114 * 232 * 454 * 454 * 678) % 5 = 4 := by
  sorry

end NUMINAMATH_GPT_product_mod_five_remainder_l165_16562


namespace NUMINAMATH_GPT_proof_problem_l165_16521

/- Define relevant concepts -/
def is_factor (a b : Nat) := ∃ k, b = a * k
def is_divisor := is_factor

/- Given conditions with their translations -/
def condition_A : Prop := is_factor 5 35
def condition_B : Prop := is_divisor 21 252 ∧ ¬ is_divisor 21 48
def condition_C : Prop := ¬ (is_divisor 15 90 ∨ is_divisor 15 74)
def condition_D : Prop := is_divisor 18 36 ∧ ¬ is_divisor 18 72
def condition_E : Prop := is_factor 9 180

/- The main proof problem statement -/
theorem proof_problem : condition_A ∧ condition_B ∧ ¬ condition_C ∧ ¬ condition_D ∧ condition_E :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l165_16521


namespace NUMINAMATH_GPT_alice_cranes_ratio_alice_cranes_l165_16561

theorem alice_cranes {A : ℕ} (h1 : A + (1/5 : ℝ) * (1000 - A) + 400 = 1000) :
  A = 500 := by
  sorry

theorem ratio_alice_cranes :
  (500 : ℝ) / 1000 = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_alice_cranes_ratio_alice_cranes_l165_16561


namespace NUMINAMATH_GPT_sum_of_roots_is_zero_l165_16533

variables {R : Type*} [Field R] {a b c p q : R}

theorem sum_of_roots_is_zero (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : a ≠ c)
  (h₄ : a^3 + p * a + q = 0) (h₅ : b^3 + p * b + q = 0) (h₆ : c^3 + p * c + q = 0) :
  a + b + c = 0 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_roots_is_zero_l165_16533


namespace NUMINAMATH_GPT_milburg_population_l165_16565

theorem milburg_population 
    (adults : ℕ := 5256) 
    (children : ℕ := 2987) 
    (teenagers : ℕ := 1709) 
    (seniors : ℕ := 2340) : 
    adults + children + teenagers + seniors = 12292 := 
by 
  sorry

end NUMINAMATH_GPT_milburg_population_l165_16565


namespace NUMINAMATH_GPT_blue_bordered_area_on_outer_sphere_l165_16583

theorem blue_bordered_area_on_outer_sphere :
  let r := 1 -- cm
  let r1 := 4 -- cm
  let r2 := 6 -- cm
  let A_inner := 27 -- cm^2
  let h := A_inner / (2 * π * r1)
  let A_outer := 2 * π * r2 * h
  A_outer = 60.75 := sorry

end NUMINAMATH_GPT_blue_bordered_area_on_outer_sphere_l165_16583


namespace NUMINAMATH_GPT_triangle_inequality_l165_16572

theorem triangle_inequality {A B C : ℝ} {n : ℕ} (h : B = n * C) (hA : A + B + C = π) :
  B ≤ n * C :=
by
  sorry

end NUMINAMATH_GPT_triangle_inequality_l165_16572


namespace NUMINAMATH_GPT_find_middle_number_l165_16582

theorem find_middle_number (a b c : ℕ) (h1 : a + b = 16) (h2 : a + c = 21) (h3 : b + c = 27) : b = 11 := by
  sorry

end NUMINAMATH_GPT_find_middle_number_l165_16582


namespace NUMINAMATH_GPT_scientific_notation_1_3_billion_l165_16537

theorem scientific_notation_1_3_billion : 1300000000 = 1.3 * 10^9 := 
sorry

end NUMINAMATH_GPT_scientific_notation_1_3_billion_l165_16537


namespace NUMINAMATH_GPT_sin_75_mul_sin_15_eq_one_fourth_l165_16597

theorem sin_75_mul_sin_15_eq_one_fourth : 
  Real.sin (75 * Real.pi / 180) * Real.sin (15 * Real.pi / 180) = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_sin_75_mul_sin_15_eq_one_fourth_l165_16597


namespace NUMINAMATH_GPT_trapezoid_area_l165_16571

theorem trapezoid_area (x y : ℝ) (hx : y^2 + x^2 = 625) (hy : y^2 + (25 - x)^2 = 900) :
  1 / 2 * (11 + 36) * 24 = 564 :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_area_l165_16571


namespace NUMINAMATH_GPT_birds_in_sky_l165_16550

theorem birds_in_sky (wings total_wings : ℕ) (h1 : total_wings = 26) (h2 : wings = 2) : total_wings / wings = 13 := 
by
  sorry

end NUMINAMATH_GPT_birds_in_sky_l165_16550


namespace NUMINAMATH_GPT_master_parts_per_hour_l165_16531

variable (x : ℝ)

theorem master_parts_per_hour (h1 : 300 / x = 100 / (40 - x)) : 300 / x = 100 / (40 - x) :=
sorry

end NUMINAMATH_GPT_master_parts_per_hour_l165_16531


namespace NUMINAMATH_GPT_possible_to_divide_into_two_groups_l165_16505

-- Define a type for People
universe u
variable {Person : Type u}

-- Define friend and enemy relations (assume they are given as functions)
variable (friend enemy : Person → Person)

-- Define the main statement
theorem possible_to_divide_into_two_groups (h_friend : ∀ p : Person, ∃ q : Person, friend p = q)
                                           (h_enemy : ∀ p : Person, ∃ q : Person, enemy p = q) :
  ∃ (company : Person → Bool),
    ∀ p : Person, company p ≠ company (friend p) ∧ company p ≠ company (enemy p) :=
by
  sorry

end NUMINAMATH_GPT_possible_to_divide_into_two_groups_l165_16505


namespace NUMINAMATH_GPT_proof_statement_l165_16567

variables {K_c A_c K_d B_d A_d B_c : ℕ}

def conditions (K_c A_c K_d B_d A_d B_c : ℕ) :=
  K_c > A_c ∧ K_d > B_d ∧ A_d > K_d ∧ B_c > A_c

noncomputable def statement (K_c A_c K_d B_d A_d B_c : ℕ) (h : conditions K_c A_c K_d B_d A_d B_c) : Prop :=
  A_d > max K_d B_d

theorem proof_statement (K_c A_c K_d B_d A_d B_c : ℕ) (h : conditions K_c A_c K_d B_d A_d B_c) : statement K_c A_c K_d B_d A_d B_c h :=
sorry

end NUMINAMATH_GPT_proof_statement_l165_16567


namespace NUMINAMATH_GPT_calculate_discount_l165_16506

def original_price := 22
def sale_price := 16

theorem calculate_discount : original_price - sale_price = 6 := 
by
  sorry

end NUMINAMATH_GPT_calculate_discount_l165_16506


namespace NUMINAMATH_GPT_conversion_1_conversion_2_conversion_3_l165_16594

theorem conversion_1 : 2 * 1000 = 2000 := sorry

theorem conversion_2 : 9000 / 1000 = 9 := sorry

theorem conversion_3 : 8 * 1000 = 8000 := sorry

end NUMINAMATH_GPT_conversion_1_conversion_2_conversion_3_l165_16594


namespace NUMINAMATH_GPT_units_digit_7_pow_1023_l165_16552

-- Define a function for the units digit of a number
def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_7_pow_1023 :
  units_digit (7 ^ 1023) = 3 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_7_pow_1023_l165_16552


namespace NUMINAMATH_GPT_compare_exponents_l165_16591

theorem compare_exponents :
  let a := (3 / 2) ^ 0.1
  let b := (3 / 2) ^ 0.2
  let c := (3 / 2) ^ 0.08
  c < a ∧ a < b := by
  sorry

end NUMINAMATH_GPT_compare_exponents_l165_16591


namespace NUMINAMATH_GPT_solve_inequality_l165_16543

theorem solve_inequality (a x : ℝ) : 
  if a > 0 then -a < x ∧ x < 2*a else if a < 0 then 2*a < x ∧ x < -a else False :=
by sorry

end NUMINAMATH_GPT_solve_inequality_l165_16543


namespace NUMINAMATH_GPT_number_of_solutions_eq_one_l165_16542

theorem number_of_solutions_eq_one :
  ∃! (n : ℕ), 0 < n ∧ 
              (∃ k : ℕ, (n + 1500) = 90 * k ∧ k = Int.floor (Real.sqrt n)) :=
sorry

end NUMINAMATH_GPT_number_of_solutions_eq_one_l165_16542


namespace NUMINAMATH_GPT_ribbon_length_per_gift_l165_16560

theorem ribbon_length_per_gift (gifts : ℕ) (initial_ribbon remaining_ribbon : ℝ) (total_used_ribbon : ℝ) (length_per_gift : ℝ):
  gifts = 8 →
  initial_ribbon = 15 →
  remaining_ribbon = 3 →
  total_used_ribbon = initial_ribbon - remaining_ribbon →
  length_per_gift = total_used_ribbon / gifts →
  length_per_gift = 1.5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_ribbon_length_per_gift_l165_16560


namespace NUMINAMATH_GPT_polygon_sides_l165_16596

theorem polygon_sides (n : ℕ) (hn : (n - 2) * 180 = 5 * 360) : n = 12 :=
by
  sorry

end NUMINAMATH_GPT_polygon_sides_l165_16596


namespace NUMINAMATH_GPT_isosceles_triangle_sides_l165_16579

theorem isosceles_triangle_sides (a b : ℝ) (h1 : 2 * a + a = 14 ∨ 2 * a + a = 18)
  (h2 : a + b = 18 ∨ a + b = 14) : 
  (a = 14/3 ∧ b = 40/3 ∨ a = 6 ∧ b = 8) :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_sides_l165_16579


namespace NUMINAMATH_GPT_inversely_proportional_x_y_l165_16514

theorem inversely_proportional_x_y (x y c : ℝ) 
  (h1 : x * y = c) (h2 : 8 * 16 = c) : y = -32 → x = -4 :=
by
  sorry

end NUMINAMATH_GPT_inversely_proportional_x_y_l165_16514


namespace NUMINAMATH_GPT_finish_together_in_4_days_l165_16555

-- Definitions for the individual days taken by A, B, and C
def days_for_A := 12
def days_for_B := 24
def days_for_C := 8 -- C's approximated days

-- The rates are the reciprocals of the days
def rate_A := 1 / days_for_A
def rate_B := 1 / days_for_B
def rate_C := 1 / days_for_C

-- The combined rate of A, B, and C
def combined_rate := rate_A + rate_B + rate_C

-- The total days required to finish the work together
def total_days := 1 / combined_rate

-- Theorem stating that the total days required is 4
theorem finish_together_in_4_days : total_days = 4 := 
by 
-- proof omitted
sorry

end NUMINAMATH_GPT_finish_together_in_4_days_l165_16555


namespace NUMINAMATH_GPT_triangle_area_l165_16512

theorem triangle_area (a b c : ℝ) (h₁ : a = 5) (h₂ : b = 12) (h₃ : c = 13) (h₄ : a * a + b * b = c * c) :
  (1/2) * a * b = 30 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_l165_16512


namespace NUMINAMATH_GPT_eq_sets_M_N_l165_16508

def setM : Set ℤ := { u | ∃ m n l : ℤ, u = 12 * m + 8 * n + 4 * l }
def setN : Set ℤ := { u | ∃ p q r : ℤ, u = 20 * p + 16 * q + 12 * r }

theorem eq_sets_M_N : setM = setN := by
  sorry

end NUMINAMATH_GPT_eq_sets_M_N_l165_16508


namespace NUMINAMATH_GPT_at_least_one_shooter_hits_target_l165_16525

-- Definition stating the probability of the first shooter hitting the target
def prob_A1 : ℝ := 0.7

-- Definition stating the probability of the second shooter hitting the target
def prob_A2 : ℝ := 0.8

-- The event that at least one shooter hits the target
def prob_at_least_one_hit : ℝ := prob_A1 + prob_A2 - (prob_A1 * prob_A2)

-- Prove that the probability that at least one shooter hits the target is 0.94
theorem at_least_one_shooter_hits_target : prob_at_least_one_hit = 0.94 :=
by
  sorry

end NUMINAMATH_GPT_at_least_one_shooter_hits_target_l165_16525


namespace NUMINAMATH_GPT_parametric_to_standard_l165_16513

theorem parametric_to_standard (θ : ℝ) (x y : ℝ)
  (h1 : x = 1 + 2 * Real.cos θ)
  (h2 : y = 2 * Real.sin θ) :
  (x - 1)^2 + y^2 = 4 := 
sorry

end NUMINAMATH_GPT_parametric_to_standard_l165_16513


namespace NUMINAMATH_GPT_fifth_graders_buy_more_l165_16599

-- Define the total payments made by eighth graders and fifth graders
def eighth_graders_payment : ℕ := 210
def fifth_graders_payment : ℕ := 240
def number_of_fifth_graders : ℕ := 25

-- The price per notebook in whole cents
def price_per_notebook (p : ℕ) : Prop :=
  ∃ k1 k2 : ℕ, k1 * p = eighth_graders_payment ∧ k2 * p = fifth_graders_payment

-- The difference in the number of notebooks bought by the fifth graders and the eighth graders
def notebook_difference (p : ℕ) : ℕ :=
  let eighth_graders_notebooks := eighth_graders_payment / p
  let fifth_graders_notebooks := fifth_graders_payment / p
  fifth_graders_notebooks - eighth_graders_notebooks

-- Theorem stating the difference in the number of notebooks equals 2
theorem fifth_graders_buy_more (p : ℕ) (h : price_per_notebook p) : notebook_difference p = 2 :=
  sorry

end NUMINAMATH_GPT_fifth_graders_buy_more_l165_16599


namespace NUMINAMATH_GPT_election_total_votes_l165_16544

theorem election_total_votes
  (total_votes : ℕ)
  (votes_A : ℕ)
  (votes_B : ℕ)
  (votes_C : ℕ)
  (h1 : votes_A = 55 * total_votes / 100)
  (h2 : votes_B = 35 * total_votes / 100)
  (h3 : votes_C = total_votes - votes_A - votes_B)
  (h4 : votes_A = votes_B + 400) :
  total_votes = 2000 := by
  sorry

end NUMINAMATH_GPT_election_total_votes_l165_16544


namespace NUMINAMATH_GPT_fraction_representation_correct_l165_16504

theorem fraction_representation_correct (h : ∀ (x y z w: ℕ), 9*x = y ∧ 47*z = w ∧ 2*47*5 = 235):
  (18: ℚ) / (9 * 47 * 5) = (2: ℚ) / 235 :=
by
  sorry

end NUMINAMATH_GPT_fraction_representation_correct_l165_16504


namespace NUMINAMATH_GPT_inequality_proof_l165_16559

variable (a b c : ℝ)

theorem inequality_proof :
  1 < (a / Real.sqrt (a^2 + b^2) + b / Real.sqrt (b^2 + c^2) + c / Real.sqrt (c^2 + a^2)) ∧
  (a / Real.sqrt (a^2 + b^2) + b / Real.sqrt (b^2 + c^2) + c / Real.sqrt (c^2 + a^2)) ≤ 3 * Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_GPT_inequality_proof_l165_16559


namespace NUMINAMATH_GPT_sleep_hours_for_desired_average_l165_16588

theorem sleep_hours_for_desired_average 
  (s_1 s_2 : ℝ) (h_1 h_2 : ℝ) (k : ℝ) 
  (h_inverse_relation : ∀ s h, s * h = k)
  (h_s1 : s_1 = 75)
  (h_h1 : h_1 = 6)
  (h_average : (s_1 + s_2) / 2 = 85) : 
  h_2 = 450 / 95 := 
by 
  sorry

end NUMINAMATH_GPT_sleep_hours_for_desired_average_l165_16588


namespace NUMINAMATH_GPT_calculate_expression_l165_16541

theorem calculate_expression :
  2^3 - (Real.tan (Real.pi / 3))^2 = 5 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l165_16541


namespace NUMINAMATH_GPT_photos_per_album_correct_l165_16535

-- Define the conditions
def total_photos : ℕ := 4500
def first_batch_photos : ℕ := 1500
def first_batch_albums : ℕ := 30
def second_batch_albums : ℕ := 60
def remaining_photos : ℕ := total_photos - first_batch_photos

-- Define the number of photos per album for the first batch (should be 50)
def photos_per_album_first_batch : ℕ := first_batch_photos / first_batch_albums

-- Define the number of photos per album for the second batch (should be 50)
def photos_per_album_second_batch : ℕ := remaining_photos / second_batch_albums

-- Statement to prove
theorem photos_per_album_correct :
  photos_per_album_first_batch = 50 ∧ photos_per_album_second_batch = 50 :=
by
  simp [photos_per_album_first_batch, photos_per_album_second_batch, remaining_photos]
  sorry

end NUMINAMATH_GPT_photos_per_album_correct_l165_16535


namespace NUMINAMATH_GPT_parallelogram_to_rhombus_l165_16540

theorem parallelogram_to_rhombus {a b m1 m2 x : ℝ} (h_area : a * m1 = x * m2) (h_proportion : b / m1 = x / m2) : x = Real.sqrt (a * b) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_parallelogram_to_rhombus_l165_16540


namespace NUMINAMATH_GPT_smallest_of_powers_l165_16587

theorem smallest_of_powers :
  min (2^55) (min (3^44) (min (5^33) (6^22))) = 2^55 :=
by
  sorry

end NUMINAMATH_GPT_smallest_of_powers_l165_16587


namespace NUMINAMATH_GPT_no_transform_to_1998_power_7_l165_16526

theorem no_transform_to_1998_power_7 :
  ∀ n : ℕ, (exists m : ℕ, n = 7^m) ->
  ∀ k : ℕ, n = 10 * k + (n % 10) ->
  ¬ (∃ t : ℕ, (t = (1998 ^ 7))) := 
by sorry

end NUMINAMATH_GPT_no_transform_to_1998_power_7_l165_16526


namespace NUMINAMATH_GPT_arithmetic_sequence_zero_term_l165_16566

theorem arithmetic_sequence_zero_term (a : ℕ → ℤ) (d : ℤ) (h : d ≠ 0) 
  (h_seq : ∀ n, a n = a 1 + (n-1) * d)
  (h_condition : a 3 + a 9 = a 10 - a 8) :
  ∃ n, a n = 0 ∧ n = 5 :=
by { sorry }

end NUMINAMATH_GPT_arithmetic_sequence_zero_term_l165_16566


namespace NUMINAMATH_GPT_shaded_area_is_20_l165_16551

theorem shaded_area_is_20 (large_square_side : ℕ) (num_small_squares : ℕ) 
  (shaded_squares : ℕ) 
  (h1 : large_square_side = 10) (h2 : num_small_squares = 25) 
  (h3 : shaded_squares = 5) : 
  (large_square_side^2 / num_small_squares) * shaded_squares = 20 :=
by
  sorry

end NUMINAMATH_GPT_shaded_area_is_20_l165_16551


namespace NUMINAMATH_GPT_lcm_18_35_l165_16500

theorem lcm_18_35 : Nat.lcm 18 35 = 630 := 
by 
  sorry

end NUMINAMATH_GPT_lcm_18_35_l165_16500


namespace NUMINAMATH_GPT_rabbit_jumps_before_dog_catches_l165_16576

/-- Prove that the number of additional jumps the rabbit can make before the dog catches up is 700,
    given the initial conditions:
      1. The rabbit has a 50-jump head start.
      2. The dog makes 5 jumps in the time the rabbit makes 6 jumps.
      3. The distance covered by 7 jumps of the dog equals the distance covered by 9 jumps of the rabbit. -/
theorem rabbit_jumps_before_dog_catches (h_head_start : ℕ) (h_time_ratio : ℚ) (h_distance_ratio : ℚ) : 
    h_head_start = 50 → h_time_ratio = 5/6 → h_distance_ratio = 7/9 → 
    ∃ (rabbit_additional_jumps : ℕ), rabbit_additional_jumps = 700 :=
by
  intro h_head_start_intro h_time_ratio_intro h_distance_ratio_intro
  have rabbit_additional_jumps := 700
  use rabbit_additional_jumps
  sorry

end NUMINAMATH_GPT_rabbit_jumps_before_dog_catches_l165_16576


namespace NUMINAMATH_GPT_history_paper_pages_l165_16501

/-
Stacy has a history paper due in 3 days.
She has to write 21 pages per day to finish on time.
Prove that the total number of pages for the history paper is 63.
-/

theorem history_paper_pages (pages_per_day : ℕ) (days : ℕ) (total_pages : ℕ) 
  (h1 : pages_per_day = 21) (h2 : days = 3) : total_pages = 63 :=
by
  -- We would include the proof here, but for now, we use sorry to skip the proof.
  sorry

end NUMINAMATH_GPT_history_paper_pages_l165_16501


namespace NUMINAMATH_GPT_completing_square_l165_16557

theorem completing_square (x : ℝ) : x^2 - 6 * x + 8 = 0 → (x - 3)^2 = 1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_completing_square_l165_16557


namespace NUMINAMATH_GPT_meaningful_expression_range_l165_16548

theorem meaningful_expression_range (x : ℝ) : (¬ (x - 1 = 0)) ↔ (x ≠ 1) := 
by
  sorry

end NUMINAMATH_GPT_meaningful_expression_range_l165_16548


namespace NUMINAMATH_GPT_remainder_of_M_mod_210_l165_16519

def M : ℤ := 1234567891011

theorem remainder_of_M_mod_210 :
  (M % 210) = 31 :=
by
  have modulus1 : M % 6 = 3 := by sorry
  have modulus2 : M % 5 = 1 := by sorry
  have modulus3 : M % 7 = 2 := by sorry
  -- Using Chinese Remainder Theorem
  sorry

end NUMINAMATH_GPT_remainder_of_M_mod_210_l165_16519


namespace NUMINAMATH_GPT_quadratic_coefficients_l165_16573

theorem quadratic_coefficients (a b c : ℝ) (h₀: 0 < a) 
  (h₁: |a + b + c| = 3) 
  (h₂: |4 * a + 2 * b + c| = 3) 
  (h₃: |9 * a + 3 * b + c| = 3) : 
  (a = 6 ∧ b = -24 ∧ c = 21) ∨ (a = 3 ∧ b = -15 ∧ c = 15) ∨ (a = 3 ∧ b = -9 ∧ c = 3) :=
sorry

end NUMINAMATH_GPT_quadratic_coefficients_l165_16573


namespace NUMINAMATH_GPT_range_of_a_l165_16563

theorem range_of_a (a : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → (a - 2)^x₁ > (a - 2)^x₂) → (2 < a ∧ a < 3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l165_16563


namespace NUMINAMATH_GPT_determine_x_value_l165_16520

theorem determine_x_value (a b c x : ℕ) (h1 : x = a + 7) (h2 : a = b + 12) (h3 : b = c + 25) (h4 : c = 95) : x = 139 := by
  sorry

end NUMINAMATH_GPT_determine_x_value_l165_16520


namespace NUMINAMATH_GPT_exists_k_composite_l165_16556

theorem exists_k_composite (h : Nat) : ∃ k : ℕ, ∀ n : ℕ, 0 < n → ∃ p : ℕ, Prime p ∧ p ∣ (k * 2 ^ n + 1) :=
by
  sorry

end NUMINAMATH_GPT_exists_k_composite_l165_16556


namespace NUMINAMATH_GPT_rice_containers_l165_16534

theorem rice_containers (pound_to_ounce : ℕ) (total_rice_lb : ℚ) (container_oz : ℕ) : 
  pound_to_ounce = 16 → 
  total_rice_lb = 33 / 4 → 
  container_oz = 33 → 
  (total_rice_lb * pound_to_ounce) / container_oz = 4 :=
by sorry

end NUMINAMATH_GPT_rice_containers_l165_16534


namespace NUMINAMATH_GPT_Jason_saturday_hours_l165_16578

theorem Jason_saturday_hours (x y : ℕ) 
  (h1 : 4 * x + 6 * y = 88)
  (h2 : x + y = 18) : 
  y = 8 :=
sorry

end NUMINAMATH_GPT_Jason_saturday_hours_l165_16578


namespace NUMINAMATH_GPT_necessary_condition_l165_16503

theorem necessary_condition :
  ∃ x : ℝ, (x < 0 ∨ x > 2) → (2 * x^2 - 5 * x - 3 ≥ 0) :=
sorry

end NUMINAMATH_GPT_necessary_condition_l165_16503


namespace NUMINAMATH_GPT_triangle_type_l165_16575

theorem triangle_type (A B C : ℝ) (a b c : ℝ)
  (h1 : B = 30) 
  (h2 : c = 15) 
  (h3 : b = 5 * Real.sqrt 3) 
  (h4 : a ≠ 0) 
  (h5 : b ≠ 0)
  (h6 : c ≠ 0) 
  (h7 : 0 < A ∧ A < 180) 
  (h8 : 0 < B ∧ B < 180) 
  (h9 : 0 < C ∧ C < 180) 
  (h10 : A + B + C = 180) : 
  (A = 90 ∨ A = C) ∧ A + B + C = 180 :=
by 
  sorry

end NUMINAMATH_GPT_triangle_type_l165_16575


namespace NUMINAMATH_GPT_distinct_positive_integer_roots_pq_l165_16564

theorem distinct_positive_integer_roots_pq :
  ∃ (p q : ℝ), (∃ (a b c : ℕ), (a ≠ b ∧ b ≠ c ∧ c ≠ a) ∧ (a + b + c = 9) ∧ (a * b + a * c + b * c = p) ∧ (a * b * c = q)) ∧ p + q = 38 :=
by sorry


end NUMINAMATH_GPT_distinct_positive_integer_roots_pq_l165_16564


namespace NUMINAMATH_GPT_stephen_speed_l165_16511

theorem stephen_speed (v : ℝ) 
  (time : ℝ := 0.25)
  (speed_second_third : ℝ := 12)
  (speed_last_third : ℝ := 20)
  (total_distance : ℝ := 12) :
  (v * time + speed_second_third * time + speed_last_third * time = total_distance) → 
  v = 16 :=
by
  intro h
  -- introducing the condition h: v * 0.25 + 3 + 5 = 12
  sorry

end NUMINAMATH_GPT_stephen_speed_l165_16511


namespace NUMINAMATH_GPT_find_positive_integer_pair_l165_16553

theorem find_positive_integer_pair (a b : ℕ) (h : ∀ n : ℕ, n > 0 → ∃ c_n : ℕ, a^n + b^n = c_n^(n + 1)) : a = 2 ∧ b = 2 := 
sorry

end NUMINAMATH_GPT_find_positive_integer_pair_l165_16553


namespace NUMINAMATH_GPT_linear_system_reduction_transformation_l165_16539

theorem linear_system_reduction_transformation :
  ∀ (use_substitution_or_elimination : Bool), 
    (use_substitution_or_elimination = true) ∨ (use_substitution_or_elimination = false) → 
    "Reduction and transformation" = "Reduction and transformation" :=
by
  intro use_substitution_or_elimination h
  sorry

end NUMINAMATH_GPT_linear_system_reduction_transformation_l165_16539


namespace NUMINAMATH_GPT_brittany_money_times_brooke_l165_16589

theorem brittany_money_times_brooke 
  (kent_money : ℕ) (brooke_money : ℕ) (brittany_money : ℕ) (alison_money : ℕ)
  (h1 : kent_money = 1000)
  (h2 : brooke_money = 2 * kent_money)
  (h3 : alison_money = 4000)
  (h4 : alison_money = brittany_money / 2) :
  brittany_money = 4 * brooke_money :=
by
  sorry

end NUMINAMATH_GPT_brittany_money_times_brooke_l165_16589


namespace NUMINAMATH_GPT_smallest_result_l165_16507

theorem smallest_result :
  let a := (-2)^3
  let b := (-2) + 3
  let c := (-2) * 3
  let d := (-2) - 3
  a < b ∧ a < c ∧ a < d :=
by
  -- Lean proof steps would go here
  sorry

end NUMINAMATH_GPT_smallest_result_l165_16507


namespace NUMINAMATH_GPT_value_of_x_is_10_l165_16581

-- Define the conditions
def condition1 (x : ℕ) : ℕ := 3 * x
def condition2 (x : ℕ) : ℕ := (26 - x) + 14

-- Define the proof problem
theorem value_of_x_is_10 (x : ℕ) (h1 : condition1 x = condition2 x) : x = 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_value_of_x_is_10_l165_16581


namespace NUMINAMATH_GPT_min_blue_eyes_with_lunchbox_l165_16517

theorem min_blue_eyes_with_lunchbox (B L : Finset Nat) (hB : B.card = 15) (hL : L.card = 25) (students : Finset Nat) (hst : students.card = 35)  : 
  ∃ (x : Finset Nat), x ⊆ B ∧ x ⊆ L ∧ x.card ≥ 5 :=
by
  sorry

end NUMINAMATH_GPT_min_blue_eyes_with_lunchbox_l165_16517


namespace NUMINAMATH_GPT_fraction_of_roll_used_l165_16523

theorem fraction_of_roll_used 
  (x : ℚ) 
  (h1 : 3 * x + 3 * x + x + 2 * x = 9 * x)
  (h2 : 9 * x = (2 / 5)) : 
  x = 2 / 45 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_roll_used_l165_16523


namespace NUMINAMATH_GPT_find_a_if_odd_f_monotonically_increasing_on_pos_l165_16527

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (x^2 - 1) / (x + a)

-- Part 1: Proving that a = 0
theorem find_a_if_odd : (∀ x : ℝ, f x a = -f (-x) a) → a = 0 := by sorry

-- Part 2: Proving that f(x) is monotonically increasing on (0, +∞) given a = 0
theorem f_monotonically_increasing_on_pos : (∀ x : ℝ, x > 0 → 
  ∃ y : ℝ, y > 0 ∧ f x 0 < f y 0) := by sorry

end NUMINAMATH_GPT_find_a_if_odd_f_monotonically_increasing_on_pos_l165_16527


namespace NUMINAMATH_GPT_s_eq_sin_c_eq_cos_l165_16568

open Real

variables (s c : ℝ → ℝ)

-- Conditions
def s_prime := ∀ x, deriv s x = c x
def c_prime := ∀ x, deriv c x = -s x
def initial_conditions := (s 0 = 0) ∧ (c 0 = 1)

-- Theorem to prove
theorem s_eq_sin_c_eq_cos
  (h1 : s_prime s c)
  (h2 : c_prime s c)
  (h3 : initial_conditions s c) :
  (∀ x, s x = sin x) ∧ (∀ x, c x = cos x) :=
sorry

end NUMINAMATH_GPT_s_eq_sin_c_eq_cos_l165_16568


namespace NUMINAMATH_GPT_book_width_l165_16530

noncomputable def golden_ratio : Real := (1 + Real.sqrt 5) / 2

theorem book_width (length : Real) (width : Real) 
(h1 : length = 20) 
(h2 : width / length = golden_ratio) : 
width = 12.36 := 
by 
  sorry

end NUMINAMATH_GPT_book_width_l165_16530


namespace NUMINAMATH_GPT_sum_of_products_l165_16585

theorem sum_of_products (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 52) 
  (h2 : a + b + c = 14) : 
  ab + bc + ac = 72 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_products_l165_16585


namespace NUMINAMATH_GPT_percentage_of_hundred_l165_16510

theorem percentage_of_hundred : (30 / 100) * 100 = 30 := 
by
  sorry

end NUMINAMATH_GPT_percentage_of_hundred_l165_16510


namespace NUMINAMATH_GPT_some_number_value_correct_l165_16574

noncomputable def value_of_some_number (a : ℕ) : ℕ :=
  (a^3) / (25 * 45 * 49)

theorem some_number_value_correct :
  value_of_some_number 105 = 21 := by
  sorry

end NUMINAMATH_GPT_some_number_value_correct_l165_16574


namespace NUMINAMATH_GPT_factorize_a3_sub_a_l165_16590

theorem factorize_a3_sub_a (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) :=
sorry

end NUMINAMATH_GPT_factorize_a3_sub_a_l165_16590


namespace NUMINAMATH_GPT_least_negative_b_l165_16592

theorem least_negative_b (x b : ℤ) (h1 : x^2 + b * x = 22) (h2 : b < 0) : b = -21 :=
sorry

end NUMINAMATH_GPT_least_negative_b_l165_16592


namespace NUMINAMATH_GPT_basketball_game_l165_16518

theorem basketball_game 
    (a b x : ℕ)
    (h1 : 3 * b = 2 * a)
    (h2 : x = 2 * b)
    (h3 : 2 * a + 3 * b + x = 72) : 
    x = 18 :=
sorry

end NUMINAMATH_GPT_basketball_game_l165_16518


namespace NUMINAMATH_GPT_angle_bisector_median_ineq_l165_16570

variables {A B C : Type} [AddCommGroup A] [AddCommGroup B] [AddCommGroup C]
variables (l_a l_b l_c m_a m_b m_c : ℝ)

theorem angle_bisector_median_ineq
  (hl_a : l_a > 0) (hl_b : l_b > 0) (hl_c : l_c > 0)
  (hm_a : m_a > 0) (hm_b : m_b > 0) (hm_c : m_c > 0) :
  l_a / m_a + l_b / m_b + l_c / m_c > 1 :=
sorry

end NUMINAMATH_GPT_angle_bisector_median_ineq_l165_16570


namespace NUMINAMATH_GPT_inequality_solution_l165_16532

def solution_set_inequality : Set ℝ := {x | x < -1/3 ∨ x > 1/2}

theorem inequality_solution (x : ℝ) : 
  (2 * x - 1) / (3 * x + 1) > 0 ↔ x ∈ solution_set_inequality :=
by 
  sorry

end NUMINAMATH_GPT_inequality_solution_l165_16532


namespace NUMINAMATH_GPT_abs_eq_5_iff_l165_16593

   theorem abs_eq_5_iff (x : ℝ) : |x| = 5 ↔ x = 5 ∨ x = -5 :=
   by
     sorry
   
end NUMINAMATH_GPT_abs_eq_5_iff_l165_16593


namespace NUMINAMATH_GPT_square_division_possible_l165_16549

theorem square_division_possible :
  ∃ (S a b c : ℕ), 
    S^2 = a^2 + 3 * b^2 + 5 * c^2 ∧ 
    a = 3 ∧ 
    b = 2 ∧ 
    c = 1 :=
  by {
    sorry
  }

end NUMINAMATH_GPT_square_division_possible_l165_16549


namespace NUMINAMATH_GPT_at_least_one_zero_l165_16524

theorem at_least_one_zero (a b : ℤ) : (¬ (a ≠ 0) ∨ ¬ (b ≠ 0)) ↔ (a = 0 ∨ b = 0) :=
by
  sorry

end NUMINAMATH_GPT_at_least_one_zero_l165_16524


namespace NUMINAMATH_GPT_distributive_property_l165_16554

theorem distributive_property (a b : ℝ) : 3 * a * (2 * a - b) = 6 * a^2 - 3 * a * b :=
by
  sorry

end NUMINAMATH_GPT_distributive_property_l165_16554


namespace NUMINAMATH_GPT_buses_required_is_12_l165_16545

-- Define the conditions given in the problem
def students : ℕ := 535
def bus_capacity : ℕ := 45

-- Define the minimum number of buses required
def buses_needed (students : ℕ) (bus_capacity : ℕ) : ℕ :=
  (students + bus_capacity - 1) / bus_capacity

-- The theorem stating the number of buses required is 12
theorem buses_required_is_12 :
  buses_needed students bus_capacity = 12 :=
sorry

end NUMINAMATH_GPT_buses_required_is_12_l165_16545


namespace NUMINAMATH_GPT_conditional_probability_l165_16580

variable (P : ℕ → ℚ)
variable (A B : ℕ)

def EventRain : Prop := P A = 4/15
def EventWind : Prop := P B = 2/15
def EventBoth : Prop := P (A * B) = 1/10

theorem conditional_probability 
  (h1 : EventRain P A) 
  (h2 : EventWind P B) 
  (h3 : EventBoth P A B) 
  : (P (A * B) / P A) = 3 / 8 := 
by
  sorry

end NUMINAMATH_GPT_conditional_probability_l165_16580


namespace NUMINAMATH_GPT_possible_values_of_p_l165_16515

theorem possible_values_of_p (a b c : ℝ) (h₁ : (-a + b + c) / a = (a - b + c) / b)
  (h₂ : (a - b + c) / b = (a + b - c) / c) :
  ∃ p ∈ ({-1, 8} : Set ℝ), p = (a + b) * (b + c) * (c + a) / (a * b * c) :=
by sorry

end NUMINAMATH_GPT_possible_values_of_p_l165_16515


namespace NUMINAMATH_GPT_total_candies_is_36_l165_16502

-- Defining the conditions
def candies_per_day (day : String) : Nat :=
  if day = "Monday" ∨ day = "Wednesday" then 2 else 1

def total_candies_per_week : Nat :=
  (candies_per_day "Monday" + candies_per_day "Tuesday"
  + candies_per_day "Wednesday" + candies_per_day "Thursday"
  + candies_per_day "Friday" + candies_per_day "Saturday"
  + candies_per_day "Sunday")

def total_candies_in_weeks (weeks : Nat) : Nat :=
  weeks * total_candies_per_week

-- Stating the theorem
theorem total_candies_is_36 : total_candies_in_weeks 4 = 36 :=
  sorry

end NUMINAMATH_GPT_total_candies_is_36_l165_16502


namespace NUMINAMATH_GPT_youngest_person_age_l165_16569

theorem youngest_person_age (n : ℕ) (average_age : ℕ) (average_age_when_youngest_born : ℕ) 
    (h1 : n = 7) (h2 : average_age = 30) (h3 : average_age_when_youngest_born = 24) :
    ∃ Y : ℚ, Y = 66 / 7 :=
by
  sorry

end NUMINAMATH_GPT_youngest_person_age_l165_16569


namespace NUMINAMATH_GPT_folder_cost_calc_l165_16509

noncomputable def pencil_cost : ℚ := 0.5
noncomputable def dozen_pencils : ℕ := 24
noncomputable def num_folders : ℕ := 20
noncomputable def total_cost : ℚ := 30
noncomputable def total_pencil_cost : ℚ := dozen_pencils * pencil_cost
noncomputable def remaining_cost := total_cost - total_pencil_cost
noncomputable def folder_cost := remaining_cost / num_folders

theorem folder_cost_calc : folder_cost = 0.9 := by
  -- Definitions
  have pencil_cost_def : pencil_cost = 0.5 := rfl
  have dozen_pencils_def : dozen_pencils = 24 := rfl
  have num_folders_def : num_folders = 20 := rfl
  have total_cost_def : total_cost = 30 := rfl
  have total_pencil_cost_def : total_pencil_cost = dozen_pencils * pencil_cost := rfl
  have remaining_cost_def : remaining_cost = total_cost - total_pencil_cost := rfl
  have folder_cost_def : folder_cost = remaining_cost / num_folders := rfl

  -- Calculation steps given conditions
  sorry

end NUMINAMATH_GPT_folder_cost_calc_l165_16509


namespace NUMINAMATH_GPT_count_integers_l165_16558

theorem count_integers (n : ℕ) (h : n = 33000) :
  ∃ k : ℕ, k = 1600 ∧
  (∀ x, 1 ≤ x ∧ x ≤ n → (x % 11 = 0 → (x % 3 ≠ 0 ∧ x % 5 ≠ 0) → x ≤ x)) :=
by 
  sorry

end NUMINAMATH_GPT_count_integers_l165_16558


namespace NUMINAMATH_GPT_maria_towels_l165_16547

-- Define the initial total towels
def initial_total : ℝ := 124.5 + 67.7

-- Define the towels given to her mother
def towels_given : ℝ := 85.35

-- Define the remaining towels (this is what we need to prove)
def towels_remaining : ℝ := 106.85

-- The theorem that states Maria ended up with the correct number of towels
theorem maria_towels :
  initial_total - towels_given = towels_remaining :=
by
  -- Here we would provide the proof, but we use sorry for now
  sorry

end NUMINAMATH_GPT_maria_towels_l165_16547


namespace NUMINAMATH_GPT_range_of_a_l165_16577

def p (a x : ℝ) : Prop := a * x^2 + a * x - 1 < 0
def q (a : ℝ) : Prop := (3 / (a - 1)) + 1 < 0

theorem range_of_a (a : ℝ) :
  ¬ (∀ x, p a x ∨ q a) → a ≤ -4 ∨ 1 ≤ a :=
by sorry

end NUMINAMATH_GPT_range_of_a_l165_16577


namespace NUMINAMATH_GPT_total_earnings_first_three_months_l165_16529

-- Definitions
def earning_first_month : ℕ := 350
def earning_second_month : ℕ := 2 * earning_first_month + 50
def earning_third_month : ℕ := 4 * (earning_first_month + earning_second_month)

-- Question restated as a theorem
theorem total_earnings_first_three_months : 
  (earning_first_month + earning_second_month + earning_third_month = 5500) :=
by 
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_total_earnings_first_three_months_l165_16529


namespace NUMINAMATH_GPT_solution_interval_l165_16598

theorem solution_interval (x : ℝ) (h1 : x / 2 ≤ 5 - x) (h2 : 5 - x < -3 * (2 + x)) :
  x < -11 / 2 := 
sorry

end NUMINAMATH_GPT_solution_interval_l165_16598
