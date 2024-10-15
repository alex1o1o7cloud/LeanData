import Mathlib

namespace NUMINAMATH_GPT_combined_bus_rides_length_l2035_203555

theorem combined_bus_rides_length :
  let v := 0.62
  let z := 0.5
  let a := 0.72
  v + z + a = 1.84 :=
by
  let v := 0.62
  let z := 0.5
  let a := 0.72
  show v + z + a = 1.84
  sorry

end NUMINAMATH_GPT_combined_bus_rides_length_l2035_203555


namespace NUMINAMATH_GPT_initial_walking_speed_l2035_203533

open Real

theorem initial_walking_speed :
  ∃ (v : ℝ), (∀ (d : ℝ), d = 9.999999999999998 →
  (∀ (lateness_time : ℝ), lateness_time = 10 / 60 →
  ((d / v) - (d / 15) = lateness_time + lateness_time)) → v = 11.25) :=
by
  sorry

end NUMINAMATH_GPT_initial_walking_speed_l2035_203533


namespace NUMINAMATH_GPT_smallest_positive_integer_congruence_l2035_203522

theorem smallest_positive_integer_congruence :
  ∃ x : ℕ, 5 * x ≡ 14 [MOD 31] ∧ 0 < x ∧ x < 31 := 
sorry

end NUMINAMATH_GPT_smallest_positive_integer_congruence_l2035_203522


namespace NUMINAMATH_GPT_num_square_free_odds_l2035_203556

noncomputable def is_square_free (m : ℕ) : Prop :=
  ∀ n : ℕ, n^2 ∣ m → n = 1

noncomputable def count_square_free_odds : ℕ :=
  (199 - 1) / 2 - (11 + 4 + 2 + 1 + 1 + 1)

theorem num_square_free_odds : count_square_free_odds = 79 := by
  sorry

end NUMINAMATH_GPT_num_square_free_odds_l2035_203556


namespace NUMINAMATH_GPT_initial_dogwood_trees_in_park_l2035_203596

def num_added_trees := 5 + 4
def final_num_trees := 16
def initial_num_trees (x : ℕ) := x

theorem initial_dogwood_trees_in_park (x : ℕ) 
  (h1 : num_added_trees = 9) 
  (h2 : final_num_trees = 16) : 
  initial_num_trees x + num_added_trees = final_num_trees → 
  x = 7 := 
by 
  intro h3
  rw [initial_num_trees, num_added_trees] at h3
  linarith

end NUMINAMATH_GPT_initial_dogwood_trees_in_park_l2035_203596


namespace NUMINAMATH_GPT_physics_marks_l2035_203510

theorem physics_marks (P C M : ℕ) 
  (h1 : P + C + M = 180) 
  (h2 : P + M = 180) 
  (h3 : P + C = 140) : 
  P = 140 := 
by 
  sorry

end NUMINAMATH_GPT_physics_marks_l2035_203510


namespace NUMINAMATH_GPT_number_of_participants_l2035_203558

theorem number_of_participants (total_gloves : ℕ) (gloves_per_participant : ℕ)
  (h : total_gloves = 126) (h' : gloves_per_participant = 2) : 
  (total_gloves / gloves_per_participant = 63) :=
by
  sorry

end NUMINAMATH_GPT_number_of_participants_l2035_203558


namespace NUMINAMATH_GPT_find_side_difference_l2035_203572

def triangle_ABC : Type := ℝ
def angle_B := 20
def angle_C := 40
def length_AD := 2

theorem find_side_difference (ABC : triangle_ABC) (B : ℝ) (C : ℝ) (AD : ℝ) (BC AB : ℝ) :
  B = angle_B → C = angle_C → AD = length_AD → BC - AB = 2 :=
by 
  sorry

end NUMINAMATH_GPT_find_side_difference_l2035_203572


namespace NUMINAMATH_GPT_prove_logical_proposition_l2035_203512

theorem prove_logical_proposition (p q : Prop) (hp : p) (hq : ¬q) : (¬p ∨ ¬q) :=
by
  sorry

end NUMINAMATH_GPT_prove_logical_proposition_l2035_203512


namespace NUMINAMATH_GPT_third_month_sale_l2035_203524

theorem third_month_sale (s3 : ℝ)
  (s1 s2 s4 s5 s6 : ℝ)
  (h1 : s1 = 2435)
  (h2 : s2 = 2920)
  (h4 : s4 = 3230)
  (h5 : s5 = 2560)
  (h6 : s6 = 1000)
  (average : (s1 + s2 + s3 + s4 + s5 + s6) / 6 = 2500) :
  s3 = 2855 := 
by sorry

end NUMINAMATH_GPT_third_month_sale_l2035_203524


namespace NUMINAMATH_GPT_robert_balls_l2035_203542

theorem robert_balls (R T : ℕ) (hR : R = 25) (hT : T = 40 / 2) : R + T = 45 :=
by
  sorry

end NUMINAMATH_GPT_robert_balls_l2035_203542


namespace NUMINAMATH_GPT_minimum_value_f_minimum_value_abc_l2035_203591

noncomputable def f (x : ℝ) : ℝ := abs (x - 4) + abs (x - 3)

theorem minimum_value_f : ∃ m : ℝ, m = 1 ∧ ∀ x : ℝ, f x ≥ m := 
by
  let m := 1
  existsi m
  sorry

theorem minimum_value_abc (a b c : ℝ) (h : a + 2 * b + 3 * c = 1) : ∃ n : ℝ, n = 1/14 ∧ a^2 + b^2 + c^2 ≥ n :=
by
  let n := 1 / 14
  existsi n
  sorry

end NUMINAMATH_GPT_minimum_value_f_minimum_value_abc_l2035_203591


namespace NUMINAMATH_GPT_minimum_value_x_2y_l2035_203578

theorem minimum_value_x_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2 * y = x * y) : x + 2 * y = 8 :=
sorry

end NUMINAMATH_GPT_minimum_value_x_2y_l2035_203578


namespace NUMINAMATH_GPT_solve_real_equation_l2035_203565

theorem solve_real_equation (x : ℝ) (h : x^4 + (3 - x)^4 = 82) : x = 2.5 ∨ x = 0.5 :=
sorry

end NUMINAMATH_GPT_solve_real_equation_l2035_203565


namespace NUMINAMATH_GPT_estimate_shaded_area_l2035_203532

theorem estimate_shaded_area 
  (side_length : ℝ)
  (points_total : ℕ)
  (points_shaded : ℕ)
  (area_shaded_estimation : ℝ) :
  side_length = 6 →
  points_total = 800 →
  points_shaded = 200 →
  area_shaded_estimation = (36 * (200 / 800)) →
  area_shaded_estimation = 9 :=
by
  intros h_side_length h_points_total h_points_shaded h_area_shaded_estimation
  rw [h_side_length, h_points_total, h_points_shaded] at *
  norm_num at h_area_shaded_estimation
  exact h_area_shaded_estimation

end NUMINAMATH_GPT_estimate_shaded_area_l2035_203532


namespace NUMINAMATH_GPT_eq_triangle_perimeter_l2035_203561

theorem eq_triangle_perimeter (s : ℝ) (h : (s^2 * Real.sqrt 3) / 4 = 2 * s) : 
  3 * s = 8 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_eq_triangle_perimeter_l2035_203561


namespace NUMINAMATH_GPT_find_v_l2035_203536

def B : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![1, 2], ![0, 1]]

def v : Matrix (Fin 2) (Fin 1) ℚ :=
  ![![3], ![1]]

def target : Matrix (Fin 2) (Fin 1) ℚ :=
  ![![15], ![5]]

theorem find_v :
  let B2 := B * B
  let B3 := B2 * B
  let B4 := B3 * B
  (B4 + B3 + B2 + B + (1 : Matrix (Fin 2) (Fin 2) ℚ)) * v = target :=
by
  sorry

end NUMINAMATH_GPT_find_v_l2035_203536


namespace NUMINAMATH_GPT_females_on_police_force_l2035_203593

theorem females_on_police_force (H : ∀ (total_female_officers total_officers_on_duty female_officers_on_duty : ℕ), 
  total_officers_on_duty = 500 ∧ female_officers_on_duty = total_officers_on_duty / 2 ∧ female_officers_on_duty = total_female_officers / 4) :
  ∃ total_female_officers : ℕ, total_female_officers = 1000 := 
by {
  sorry
}

end NUMINAMATH_GPT_females_on_police_force_l2035_203593


namespace NUMINAMATH_GPT_tangency_point_l2035_203588

def parabola1 (x : ℝ) : ℝ := x^2 + 10 * x + 18
def parabola2 (y : ℝ) : ℝ := y^2 + 60 * y + 910

theorem tangency_point (x y : ℝ) (h1 : y = parabola1 x) (h2 : x = parabola2 y) :
  x = -9 / 2 ∧ y = -59 / 2 :=
by
  sorry

end NUMINAMATH_GPT_tangency_point_l2035_203588


namespace NUMINAMATH_GPT_max_value_of_z_l2035_203528

theorem max_value_of_z (x y : ℝ) (h1 : x + 2 * y - 5 ≥ 0) (h2 : x - 2 * y + 3 ≥ 0) (h3 : x - 5 ≤ 0) :
  ∃ x y, x + y = 9 :=
by {
  sorry
}

end NUMINAMATH_GPT_max_value_of_z_l2035_203528


namespace NUMINAMATH_GPT_fraction_multiplication_l2035_203541

theorem fraction_multiplication :
  (2 / 3) * (3 / 8) = (1 / 4) :=
sorry

end NUMINAMATH_GPT_fraction_multiplication_l2035_203541


namespace NUMINAMATH_GPT_area_of_annulus_l2035_203501

-- Define the conditions
def concentric_circles (r s : ℝ) (h : r > s) (x : ℝ) := 
  r^2 = s^2 + x^2

-- State the theorem
theorem area_of_annulus (r s x : ℝ) (h : r > s) (h₁ : concentric_circles r s h x) :
  π * x^2 = π * r^2 - π * s^2 :=
by 
  rw [concentric_circles] at h₁
  sorry

end NUMINAMATH_GPT_area_of_annulus_l2035_203501


namespace NUMINAMATH_GPT_allison_uploads_480_hours_in_june_l2035_203560

noncomputable def allison_upload_total_hours : Nat :=
  let before_june_16 := 10 * 15
  let from_june_16_to_23 := 15 * 8
  let from_june_24_to_end := 30 * 7
  before_june_16 + from_june_16_to_23 + from_june_24_to_end

theorem allison_uploads_480_hours_in_june :
  allison_upload_total_hours = 480 := by
  sorry

end NUMINAMATH_GPT_allison_uploads_480_hours_in_june_l2035_203560


namespace NUMINAMATH_GPT_marys_income_percent_of_juans_income_l2035_203547

variables (M T J : ℝ)

theorem marys_income_percent_of_juans_income (h1 : M = 1.40 * T) (h2 : T = 0.60 * J) : M = 0.84 * J :=
by
  sorry

end NUMINAMATH_GPT_marys_income_percent_of_juans_income_l2035_203547


namespace NUMINAMATH_GPT_triangle_inequality_l2035_203566

variables {a b c : ℝ}

def sides_of_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_inequality (h : sides_of_triangle a b c) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 ∧ 
  (a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) = 0 ↔ a = b ∧ b = c) :=
sorry

end NUMINAMATH_GPT_triangle_inequality_l2035_203566


namespace NUMINAMATH_GPT_g_g1_eq_43_l2035_203535

def g (x : ℝ) : ℝ := 2 * x^2 + 3 * x - 1

theorem g_g1_eq_43 : g (g 1) = 43 :=
by
  sorry

end NUMINAMATH_GPT_g_g1_eq_43_l2035_203535


namespace NUMINAMATH_GPT_kem_hourly_wage_l2035_203509

theorem kem_hourly_wage (shem_total_earnings: ℝ) (shem_hours_worked: ℝ) (ratio: ℝ)
  (h1: shem_total_earnings = 80)
  (h2: shem_hours_worked = 8)
  (h3: ratio = 2.5) :
  (shem_total_earnings / shem_hours_worked) / ratio = 4 :=
by 
  sorry

end NUMINAMATH_GPT_kem_hourly_wage_l2035_203509


namespace NUMINAMATH_GPT_geometric_seq_problem_l2035_203590

-- Definitions to capture the geometric sequence and the known condition
def is_geometric_sequence (a : ℕ → ℝ) :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

variables (a : ℕ → ℝ)

-- Given the condition a_1 * a_8^3 * a_15 = 243
axiom geom_seq_condition : a 1 * (a 8)^3 * a 15 = 243

theorem geometric_seq_problem 
  (h : is_geometric_sequence a) : (a 9)^3 / (a 11) = 9 :=
sorry

end NUMINAMATH_GPT_geometric_seq_problem_l2035_203590


namespace NUMINAMATH_GPT_cos_double_angle_l2035_203564

open Real

theorem cos_double_angle (α β : ℝ) 
    (h1 : sin α = 2 * sin β) 
    (h2 : tan α = 3 * tan β) :
  cos (2 * α) = -1 / 4 ∨ cos (2 * α) = 1 := 
sorry

end NUMINAMATH_GPT_cos_double_angle_l2035_203564


namespace NUMINAMATH_GPT_coins_division_remainder_l2035_203594

theorem coins_division_remainder
  (n : ℕ)
  (h1 : n % 6 = 4)
  (h2 : n % 5 = 3)
  (h3 : n = 28) :
  n % 7 = 0 :=
by
  sorry

end NUMINAMATH_GPT_coins_division_remainder_l2035_203594


namespace NUMINAMATH_GPT_problem_statement_l2035_203557

def star (x y : Nat) : Nat :=
  match x, y with
  | 1, 1 => 4 | 1, 2 => 3 | 1, 3 => 2 | 1, 4 => 1
  | 2, 1 => 1 | 2, 2 => 4 | 2, 3 => 3 | 2, 4 => 2
  | 3, 1 => 2 | 3, 2 => 1 | 3, 3 => 4 | 3, 4 => 3
  | 4, 1 => 3 | 4, 2 => 2 | 4, 3 => 1 | 4, 4 => 4
  | _, _ => 0  -- This line handles unexpected inputs.

theorem problem_statement : star (star 3 2) (star 2 1) = 4 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l2035_203557


namespace NUMINAMATH_GPT_max_a_value_l2035_203502

theorem max_a_value (a b c d : ℕ) (h1 : a < 3 * b) (h2 : b < 4 * c) (h3 : c < 5 * d) (h4 : d < 50) :
  a ≤ 2924 :=
by sorry

end NUMINAMATH_GPT_max_a_value_l2035_203502


namespace NUMINAMATH_GPT_trig_identity_solution_l2035_203518

-- Define the necessary trigonometric functions
noncomputable def tan (x : ℝ) : ℝ := Real.sin x / Real.cos x
noncomputable def cot (x : ℝ) : ℝ := Real.cos x / Real.sin x

-- Statement of the theorem
theorem trig_identity_solution (x : ℝ) (k : ℤ) (hcos : Real.cos x ≠ 0) (hsin : Real.sin x ≠ 0) :
  (Real.sin x) ^ 2 * tan x + (Real.cos x) ^ 2 * cot x + 2 * Real.sin x * Real.cos x = (4 * Real.sqrt 3) / 3 →
  ∃ k : ℤ, x = (-1) ^ k * (Real.pi / 6) + (Real.pi / 2) :=
sorry

end NUMINAMATH_GPT_trig_identity_solution_l2035_203518


namespace NUMINAMATH_GPT_total_pears_after_giving_away_l2035_203503

def alyssa_pears : ℕ := 42
def nancy_pears : ℕ := 17
def carlos_pears : ℕ := 25
def pears_given_away_per_person : ℕ := 5

theorem total_pears_after_giving_away :
  (alyssa_pears + nancy_pears + carlos_pears) - (3 * pears_given_away_per_person) = 69 :=
by
  sorry

end NUMINAMATH_GPT_total_pears_after_giving_away_l2035_203503


namespace NUMINAMATH_GPT_range_of_x_for_acute_angle_l2035_203589

theorem range_of_x_for_acute_angle (x : ℝ) (h₁ : (x, 2*x) ≠ (0, 0)) (h₂ : (x+1, x+3) ≠ (0, 0)) (h₃ : (3*x^2 + 7*x > 0)) : 
  x < -7/3 ∨ (0 < x ∧ x < 1) ∨ x > 1 :=
by {
  -- This theorem asserts the given range of x given the dot product solution.
  sorry
}

end NUMINAMATH_GPT_range_of_x_for_acute_angle_l2035_203589


namespace NUMINAMATH_GPT_exp_values_l2035_203554

variable {a x y : ℝ}

theorem exp_values (hx : a^x = 3) (hy : a^y = 2) :
  a^(x - y) = 3 / 2 ∧ a^(2 * x + y) = 18 :=
by
  sorry

end NUMINAMATH_GPT_exp_values_l2035_203554


namespace NUMINAMATH_GPT_shaded_percentage_of_large_square_l2035_203550

theorem shaded_percentage_of_large_square
  (side_length_small_square : ℕ)
  (side_length_large_square : ℕ)
  (total_border_squares : ℕ)
  (shaded_border_squares : ℕ)
  (central_region_shaded_fraction : ℚ)
  (total_area_large_square : ℚ)
  (shaded_area_border_squares : ℚ)
  (shaded_area_central_region : ℚ) :
  side_length_small_square = 1 →
  side_length_large_square = 5 →
  total_border_squares = 16 →
  shaded_border_squares = 8 →
  central_region_shaded_fraction = 3 / 4 →
  total_area_large_square = 25 →
  shaded_area_border_squares = 8 →
  shaded_area_central_region = (3 / 4) * 9 →
  (shaded_area_border_squares + shaded_area_central_region) / total_area_large_square = 0.59 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end NUMINAMATH_GPT_shaded_percentage_of_large_square_l2035_203550


namespace NUMINAMATH_GPT_factorization_correct_l2035_203506

theorem factorization_correct (a b : ℝ) : a * b^2 - 25 * a = a * (b + 5) * (b - 5) :=
by
  -- The actual proof will be written here.
  sorry

end NUMINAMATH_GPT_factorization_correct_l2035_203506


namespace NUMINAMATH_GPT_sticker_distribution_ways_l2035_203537

theorem sticker_distribution_ways : 
  ∃ ways : ℕ, ways = Nat.choose (9) (4) ∧ ways = 126 :=
by
  sorry

end NUMINAMATH_GPT_sticker_distribution_ways_l2035_203537


namespace NUMINAMATH_GPT_triangle_area_l2035_203531

theorem triangle_area (a b c : ℝ) (A B C : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hA : A ≠ 0) (hB : B ≠ 0) (hC : C ≠ 0) 
  (h_c : c = 2) (h_C : C = π / 3)
  (h_sin : Real.sin B = 2 * Real.sin A) :
  (1/2) * a * b * Real.sin C = 2 * Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_GPT_triangle_area_l2035_203531


namespace NUMINAMATH_GPT_cone_height_l2035_203543

theorem cone_height (V : ℝ) (h r : ℝ) (π : ℝ) (h_eq_r : h = r) (volume_eq : V = 12288 * π) (V_def : V = (1/3) * π * r^3) : h = 36 := 
by
  sorry

end NUMINAMATH_GPT_cone_height_l2035_203543


namespace NUMINAMATH_GPT_original_number_of_men_l2035_203514

theorem original_number_of_men (M : ℕ) : 
  (∀ t : ℕ, (t = 8) -> (8:ℕ) * M = 8 * 10 / (M - 3) ) -> ( M = 12 ) :=
by sorry

end NUMINAMATH_GPT_original_number_of_men_l2035_203514


namespace NUMINAMATH_GPT_four_digit_integer_l2035_203595

theorem four_digit_integer (a b c d : ℕ) 
(h1: a + b + c + d = 14) (h2: b + c = 9) (h3: a - d = 1)
(h4: (a - b + c - d) % 11 = 0) : 1000 * a + 100 * b + 10 * c + d = 3542 :=
by
  sorry

end NUMINAMATH_GPT_four_digit_integer_l2035_203595


namespace NUMINAMATH_GPT_repeating_decimal_eq_fraction_l2035_203517

-- Define the repeating decimal 0.363636... as a limit of its geometric series representation
noncomputable def repeating_decimal := ∑' n : ℕ, (36 / 100^(n + 1))

-- Define the fraction
def fraction := 4 / 11

theorem repeating_decimal_eq_fraction : repeating_decimal = fraction := by sorry

end NUMINAMATH_GPT_repeating_decimal_eq_fraction_l2035_203517


namespace NUMINAMATH_GPT_product_of_numbers_l2035_203523

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 22) (h2 : x^2 + y^2 = 404) : x * y = 40 := sorry

end NUMINAMATH_GPT_product_of_numbers_l2035_203523


namespace NUMINAMATH_GPT_birds_are_crows_l2035_203505

theorem birds_are_crows (total_birds pigeons crows sparrows parrots non_pigeons: ℕ)
    (h1: pigeons = 20)
    (h2: crows = 40)
    (h3: sparrows = 15)
    (h4: parrots = total_birds - pigeons - crows - sparrows)
    (h5: total_birds = pigeons + crows + sparrows + parrots)
    (h6: non_pigeons = total_birds - pigeons) :
    (crows * 100 / non_pigeons = 50) :=
by sorry

end NUMINAMATH_GPT_birds_are_crows_l2035_203505


namespace NUMINAMATH_GPT_am_gm_inequality_l2035_203548

theorem am_gm_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a^a * b^b * c^c ≥ (a * b * c)^((a + b + c) / 3) :=
sorry

end NUMINAMATH_GPT_am_gm_inequality_l2035_203548


namespace NUMINAMATH_GPT_f_is_odd_and_increasing_l2035_203583

noncomputable def f (x : ℝ) : ℝ := 3^x - (1/3)^x

theorem f_is_odd_and_increasing : 
  (∀ x : ℝ, f (-x) = - f x) ∧ (∀ x y : ℝ, x < y → f x < f y) :=
sorry

end NUMINAMATH_GPT_f_is_odd_and_increasing_l2035_203583


namespace NUMINAMATH_GPT_quotient_of_x6_plus_8_by_x_minus_1_l2035_203544

theorem quotient_of_x6_plus_8_by_x_minus_1 :
  ∀ (x : ℝ), x ≠ 1 →
  (∃ Q : ℝ → ℝ, x^6 + 8 = (x - 1) * Q x + 9 ∧ Q x = x^5 + x^4 + x^3 + x^2 + x + 1) := 
  by
    intros x hx
    sorry

end NUMINAMATH_GPT_quotient_of_x6_plus_8_by_x_minus_1_l2035_203544


namespace NUMINAMATH_GPT_simplify_fraction_rationalize_denominator_l2035_203534

theorem simplify_fraction_rationalize_denominator :
  (5 : ℝ) / (Real.sqrt 50 + 3 * Real.sqrt 8 + Real.sqrt 18) = 5 * Real.sqrt 2 / 28 :=
by
  have sqrt_50 : Real.sqrt 50 = 5 * Real.sqrt 2 := sorry
  have sqrt_8 : 3 * Real.sqrt 8 = 6 * Real.sqrt 2 := sorry
  have sqrt_18 : Real.sqrt 18 = 3 * Real.sqrt 2 := sorry
  sorry

end NUMINAMATH_GPT_simplify_fraction_rationalize_denominator_l2035_203534


namespace NUMINAMATH_GPT_non_degenerate_ellipse_condition_l2035_203545

theorem non_degenerate_ellipse_condition (x y k a : ℝ) :
  (3 * x^2 + 9 * y^2 - 12 * x + 27 * y = k) ∧
  (∃ h : ℝ, 3 * (x - h)^2 + 9 * (y + 3/2)^2 = k + 129/4) ∧
  (k > a) ↔ (a = -129 / 4) :=
by
  sorry

end NUMINAMATH_GPT_non_degenerate_ellipse_condition_l2035_203545


namespace NUMINAMATH_GPT_whipped_cream_needed_l2035_203530

/- Problem conditions -/
def pies_per_day : ℕ := 3
def days : ℕ := 11
def pies_total : ℕ := pies_per_day * days
def pies_eaten_by_tiffany : ℕ := 4
def pies_remaining : ℕ := pies_total - pies_eaten_by_tiffany
def whipped_cream_per_pie : ℕ := 2

/- Proof statement -/
theorem whipped_cream_needed : whipped_cream_per_pie * pies_remaining = 58 := by
  sorry

end NUMINAMATH_GPT_whipped_cream_needed_l2035_203530


namespace NUMINAMATH_GPT_initial_money_proof_l2035_203580

-- Definition: Dan's initial money, the money spent, and the money left.
def initial_money : ℝ := sorry
def spent_money : ℝ := 1.0
def left_money : ℝ := 2.0

-- Theorem: Prove that Dan's initial money is the sum of the money spent and the money left.
theorem initial_money_proof : initial_money = spent_money + left_money :=
sorry

end NUMINAMATH_GPT_initial_money_proof_l2035_203580


namespace NUMINAMATH_GPT_last_four_digits_of_5_pow_9000_l2035_203567

theorem last_four_digits_of_5_pow_9000 (h : 5^300 ≡ 1 [MOD 1250]) : 
  5^9000 ≡ 1 [MOD 1250] :=
sorry

end NUMINAMATH_GPT_last_four_digits_of_5_pow_9000_l2035_203567


namespace NUMINAMATH_GPT_carpet_rate_l2035_203559

theorem carpet_rate (length breadth cost area: ℝ) (h₁ : length = 13) (h₂ : breadth = 9) (h₃ : cost = 1872) (h₄ : area = length * breadth) :
  cost / area = 16 := by
  sorry

end NUMINAMATH_GPT_carpet_rate_l2035_203559


namespace NUMINAMATH_GPT_b_remaining_work_days_l2035_203527

-- Definitions of the conditions
def together_work (a b: ℕ) := a + b = 12
def alone_work (a: ℕ) := a = 20
def c_work (c: ℕ) := c = 30
def initial_work_days := 5

-- Question to prove:
theorem b_remaining_work_days (a b c : ℕ) (h1 : together_work a b) (h2 : alone_work a) (h3 : c_work c) : 
  let b_rate := 1 / 30 
  let remaining_work := 25 / 60
  let work_to_days := remaining_work / b_rate
  work_to_days = 12.5 := 
sorry

end NUMINAMATH_GPT_b_remaining_work_days_l2035_203527


namespace NUMINAMATH_GPT_fraction_of_widgets_second_shift_l2035_203581

theorem fraction_of_widgets_second_shift (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let first_shift_widgets := x * y
  let second_shift_widgets := (2 / 3) * x * (4 / 3) * y
  let total_widgets := first_shift_widgets + second_shift_widgets
  let fraction_second_shift := second_shift_widgets / total_widgets
  fraction_second_shift = 8 / 17 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_widgets_second_shift_l2035_203581


namespace NUMINAMATH_GPT_total_surface_area_first_rectangular_parallelepiped_equals_22_l2035_203540

theorem total_surface_area_first_rectangular_parallelepiped_equals_22
  (x y z : ℝ)
  (h1 : (x + 1) * (y + 1) * (z + 1) = x * y * z + 18)
  (h2 : 2 * ((x + 1) * (y + 1) + (y + 1) * (z + 1) + (z + 1) * (x + 1)) = 2 * (x * y + x * z + y * z) + 30) :
  2 * (x * y + x * z + y * z) = 22 := sorry

end NUMINAMATH_GPT_total_surface_area_first_rectangular_parallelepiped_equals_22_l2035_203540


namespace NUMINAMATH_GPT_range_of_x_in_function_l2035_203582

theorem range_of_x_in_function (x : ℝ) : (y = 1/(x + 3) → x ≠ -3) :=
sorry

end NUMINAMATH_GPT_range_of_x_in_function_l2035_203582


namespace NUMINAMATH_GPT_first_day_exceeds_200_l2035_203563

def bacteria_count (n : ℕ) : ℕ := 4 * 3^n

def exceeds_200 (n : ℕ) : Prop := bacteria_count n > 200

theorem first_day_exceeds_200 : ∃ n, exceeds_200 n ∧ ∀ m < n, ¬ exceeds_200 m :=
by sorry

end NUMINAMATH_GPT_first_day_exceeds_200_l2035_203563


namespace NUMINAMATH_GPT_scientific_notation_equivalent_l2035_203570

theorem scientific_notation_equivalent : ∃ a n, (3120000 : ℝ) = a * 10^n ∧ a = 3.12 ∧ n = 6 :=
by
  exists 3.12
  exists 6
  sorry

end NUMINAMATH_GPT_scientific_notation_equivalent_l2035_203570


namespace NUMINAMATH_GPT_bus_driver_total_compensation_l2035_203504

-- Definitions of conditions
def regular_rate : ℝ := 16
def regular_hours : ℝ := 40
def overtime_rate : ℝ := regular_rate * 1.75
def total_hours : ℝ := 65
def total_compensation : ℝ := (regular_rate * regular_hours) + (overtime_rate * (total_hours - regular_hours))

-- Theorem stating the total compensation
theorem bus_driver_total_compensation : total_compensation = 1340 :=
by
  sorry

end NUMINAMATH_GPT_bus_driver_total_compensation_l2035_203504


namespace NUMINAMATH_GPT_cube_inscribed_circumscribed_volume_ratio_l2035_203585

theorem cube_inscribed_circumscribed_volume_ratio
  (S_1 S_2 V_1 V_2 : ℝ)
  (h : S_1 / S_2 = (1 / Real.sqrt 2) ^ 2) :
  V_1 / V_2 = (Real.sqrt 3 / 3) ^ 3 :=
sorry

end NUMINAMATH_GPT_cube_inscribed_circumscribed_volume_ratio_l2035_203585


namespace NUMINAMATH_GPT_inverse_proportional_ratios_l2035_203584

variables {x y x1 x2 y1 y2 : ℝ}
variables (h_inv_prop : ∀ (x y : ℝ), x * y = 1) (hx1_ne : x1 ≠ 0) (hx2_ne : x2 ≠ 0) (hy1_ne : y1 ≠ 0) (hy2_ne : y2 ≠ 0)

theorem inverse_proportional_ratios 
  (h1 : x1 * y1 = x2 * y2)
  (h2 : (x1 / x2) = (3 / 4)) : 
  (y1 / y2) = (4 / 3) :=
by 

sorry

end NUMINAMATH_GPT_inverse_proportional_ratios_l2035_203584


namespace NUMINAMATH_GPT_sin_minus_cos_eq_sqrt2_l2035_203507

theorem sin_minus_cos_eq_sqrt2 (x : ℝ) (hx1: 0 ≤ x) (hx2: x < 2 * Real.pi) (h: Real.sin x - Real.cos x = Real.sqrt 2) : x = (3 * Real.pi) / 4 :=
sorry

end NUMINAMATH_GPT_sin_minus_cos_eq_sqrt2_l2035_203507


namespace NUMINAMATH_GPT_shopkeeper_loss_percent_l2035_203539

theorem shopkeeper_loss_percent
  (initial_value : ℝ)
  (profit_percent : ℝ)
  (loss_percent : ℝ)
  (remaining_value_percent : ℝ)
  (profit_percent_10 : profit_percent = 0.10)
  (loss_percent_70 : loss_percent = 0.70)
  (initial_value_100 : initial_value = 100)
  (remaining_value_percent_30 : remaining_value_percent = 0.30)
  (selling_price : ℝ := initial_value * (1 + profit_percent))
  (remaining_value : ℝ := initial_value * remaining_value_percent)
  (remaining_selling_price : ℝ := remaining_value * (1 + profit_percent))
  (loss_value : ℝ := initial_value - remaining_selling_price)
  (shopkeeper_loss_percent : ℝ := loss_value / initial_value * 100) : 
  shopkeeper_loss_percent = 67 :=
sorry

end NUMINAMATH_GPT_shopkeeper_loss_percent_l2035_203539


namespace NUMINAMATH_GPT_initial_number_of_trees_l2035_203515

theorem initial_number_of_trees (trees_removed remaining_trees initial_trees : ℕ) 
  (h1 : trees_removed = 4) 
  (h2 : remaining_trees = 2) 
  (h3 : remaining_trees + trees_removed = initial_trees) : 
  initial_trees = 6 :=
by
  sorry

end NUMINAMATH_GPT_initial_number_of_trees_l2035_203515


namespace NUMINAMATH_GPT_find_f3_l2035_203551

noncomputable def f : ℝ → ℝ := sorry

theorem find_f3 (h : ∀ x : ℝ, x ≠ 0 → f x - 2 * f (1 / x) = 3 ^ x) : f 3 = -11 :=
sorry

end NUMINAMATH_GPT_find_f3_l2035_203551


namespace NUMINAMATH_GPT_george_collected_50_marbles_l2035_203579

theorem george_collected_50_marbles (w y g r total : ℕ)
  (hw : w = total / 2)
  (hy : y = 12)
  (hg : g = y / 2)
  (hr : r = 7)
  (htotal : total = w + y + g + r) :
  total = 50 := by
  sorry

end NUMINAMATH_GPT_george_collected_50_marbles_l2035_203579


namespace NUMINAMATH_GPT_find_a_b_max_min_values_l2035_203549

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  (1/3) * x^3 + a * x^2 + b * x

noncomputable def f' (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  x^2 + 2 * a * x + b

theorem find_a_b (a b : ℝ) :
  f' (-3) a b = 0 ∧ f (-3) a b = 9 → a = 1 ∧ b = -3 :=
  by sorry

theorem max_min_values (a b : ℝ) (h₁ : a = 1) (h₂ : b = -3):
  ∀ x ∈ Set.Icc (-3 : ℝ) 3, f x a b ≥ -5 / 3 ∧ f x a b ≤ 9 :=
  by sorry

end NUMINAMATH_GPT_find_a_b_max_min_values_l2035_203549


namespace NUMINAMATH_GPT_inequality_solution_l2035_203577

theorem inequality_solution (a b c : ℝ) :
  (∀ x : ℝ, -4 < x ∧ x < 7 → a * x^2 + b * x + c > 0) →
  (∀ x : ℝ, (x < -1/7 ∨ x > 1/4) ↔ c * x^2 - b * x + a > 0) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l2035_203577


namespace NUMINAMATH_GPT_solve_fraction_eq_l2035_203538

theorem solve_fraction_eq (x : ℝ) (h : (x + 6) / (x - 3) = 4) : x = 6 :=
sorry

end NUMINAMATH_GPT_solve_fraction_eq_l2035_203538


namespace NUMINAMATH_GPT_second_polygon_sides_l2035_203571

theorem second_polygon_sides (s : ℝ) (h : s ≠ 0) : 
  ∀ (P1 P2 : ℝ) (n1 n2 : ℕ), 
    n1 = 50 →
    P1 = n1 * (3 * s) →
    P2 = n2 * s →
    P1 = P2 →
    n2 = 150 :=
by
  intros P1 P2 n1 n2 hn1 hp1 hp2 hp
  sorry

end NUMINAMATH_GPT_second_polygon_sides_l2035_203571


namespace NUMINAMATH_GPT_rhinos_horn_segment_area_l2035_203574

theorem rhinos_horn_segment_area :
  let full_circle_area (r : ℝ) := π * r^2
  let quarter_circle_area (r : ℝ) := (1 / 4) * full_circle_area r
  let half_circle_area (r : ℝ) := (1 / 2) * full_circle_area r
  let larger_quarter_circle_area := quarter_circle_area 4
  let smaller_half_circle_area := half_circle_area 2
  let rhinos_horn_segment_area := larger_quarter_circle_area - smaller_half_circle_area
  rhinos_horn_segment_area = 2 * π := 
by sorry 

end NUMINAMATH_GPT_rhinos_horn_segment_area_l2035_203574


namespace NUMINAMATH_GPT_sum_of_digits_is_21_l2035_203529

theorem sum_of_digits_is_21 :
  ∃ (a b c d : ℕ), 
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧ 
  (a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10) ∧ 
  ((10 * a + b) * (10 * c + b) = 111 * d) ∧ 
  (d = 9) ∧ 
  (a + b + c + d = 21) := by
  sorry

end NUMINAMATH_GPT_sum_of_digits_is_21_l2035_203529


namespace NUMINAMATH_GPT_quadratic_roots_always_implies_l2035_203525

variable {k x1 x2 : ℝ}

theorem quadratic_roots_always_implies (h1 : k^2 > 16) 
  (h2 : x1 + x2 = -k)
  (h3 : x1 * x2 = 4) : x1^2 + x2^2 > 8 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_always_implies_l2035_203525


namespace NUMINAMATH_GPT_product_bases_l2035_203521

def base2_to_nat (s : String) : Nat :=
  s.foldl (λ acc d => acc * 2 + (d.toNat - '0'.toNat)) 0

def base3_to_nat (s : String) : Nat :=
  s.foldl (λ acc d => acc * 3 + (d.toNat - '0'.toNat)) 0

def base4_to_nat (s : String) : Nat :=
  s.foldl (λ acc d => acc * 4 + (d.toNat - '0'.toNat)) 0

theorem product_bases :
  base2_to_nat "1101" * base3_to_nat "202" * base4_to_nat "22" = 2600 :=
by
  sorry

end NUMINAMATH_GPT_product_bases_l2035_203521


namespace NUMINAMATH_GPT_problem1_problem2_l2035_203500

open Set

variable (a : Real)

-- Problem 1: Prove the intersection M ∩ (C_R N) equals the given set
theorem problem1 :
  let M := { x : ℝ | x^2 - 3*x ≤ 10 }
  let N := { x : ℝ | 3 ≤ x ∧ x ≤ 5 }
  let C_RN := { x : ℝ | x < 3 ∨ 5 < x }
  M ∩ C_RN = { x : ℝ | -2 ≤ x ∧ x < 3 } :=
by
  sorry

-- Problem 2: Prove the range of values for a such that M ∪ N = M
theorem problem2 :
  let M := { x : ℝ | x^2 - 3*x ≤ 10 }
  let N := { x : ℝ | a+1 ≤ x ∧ x ≤ 2*a+1 }
  (M ∪ N = M) → a ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l2035_203500


namespace NUMINAMATH_GPT_roots_positive_range_no_negative_roots_opposite_signs_range_l2035_203553

theorem roots_positive_range (m : ℝ) : (8 * x^2 - m * x + (m - 6) = 0) → (6 < m ∧ m ≤ 8 ∨ m ≥ 24) :=
sorry

theorem no_negative_roots (m : ℝ) : (8 * x^2 - m * x + (m - 6) = 0) → ¬ (∀ α β, (α < 0 ∧ β < 0)) :=
sorry

theorem opposite_signs_range (m : ℝ) : (8 * x^2 - m * x + (m - 6) = 0) → m < 6 :=
sorry

end NUMINAMATH_GPT_roots_positive_range_no_negative_roots_opposite_signs_range_l2035_203553


namespace NUMINAMATH_GPT_boys_from_pine_l2035_203568

/-- 
Given the following conditions:
1. There are 150 students at the camp.
2. There are 90 boys at the camp.
3. There are 60 girls at the camp.
4. There are 70 students from Maple High School.
5. There are 80 students from Pine High School.
6. There are 20 girls from Oak High School.
7. There are 30 girls from Maple High School.

Prove that the number of boys from Pine High School is 70.
--/
theorem boys_from_pine (total_students boys girls maple_high pine_high oak_girls maple_girls : ℕ)
  (H1 : total_students = 150)
  (H2 : boys = 90)
  (H3 : girls = 60)
  (H4 : maple_high = 70)
  (H5 : pine_high = 80)
  (H6 : oak_girls = 20)
  (H7 : maple_girls = 30) : 
  ∃ pine_boys : ℕ, pine_boys = 70 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_boys_from_pine_l2035_203568


namespace NUMINAMATH_GPT_trigonometric_identity_l2035_203552

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = -2) : 
  2 * Real.sin α * Real.cos α - (Real.cos α)^2 = -1 := 
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l2035_203552


namespace NUMINAMATH_GPT_liquid_flow_problem_l2035_203587

variables (x y z : ℝ)

theorem liquid_flow_problem 
    (h1 : 1/x + 1/y + 1/z = 1/6) 
    (h2 : y = 0.75 * x) 
    (h3 : z = y + 10) : 
    x = 56/3 ∧ y = 14 ∧ z = 24 :=
sorry

end NUMINAMATH_GPT_liquid_flow_problem_l2035_203587


namespace NUMINAMATH_GPT_C_converges_l2035_203516

noncomputable def behavior_of_C (e R r : ℝ) (n : ℕ) : ℝ := e * (n^2) / (R + n * (r^2))

theorem C_converges (e R r : ℝ) (h₁ : 0 < r) : ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |behavior_of_C e R r n - e / r^2| < ε := 
sorry

end NUMINAMATH_GPT_C_converges_l2035_203516


namespace NUMINAMATH_GPT_camping_trip_percentage_l2035_203598

theorem camping_trip_percentage (t : ℕ) (h1 : 22 / 100 * t > 0) (h2 : 75 / 100 * (22 / 100 * t) ≤ t) :
  (88 / 100 * t) = t :=
by
  sorry

end NUMINAMATH_GPT_camping_trip_percentage_l2035_203598


namespace NUMINAMATH_GPT_factor_problem_l2035_203592

theorem factor_problem (x y m : ℝ) (h : (1 - 2 * x + y) ∣ (4 * x * y - 4 * x^2 - y^2 - m)) :
  m = -1 :=
by
  sorry

end NUMINAMATH_GPT_factor_problem_l2035_203592


namespace NUMINAMATH_GPT_trig_expression_identity_l2035_203576

theorem trig_expression_identity (a : ℝ) (h : 2 * Real.sin a = 3 * Real.cos a) : 
  (4 * Real.sin a + Real.cos a) / (5 * Real.sin a - 2 * Real.cos a) = 14 / 11 :=
by
  sorry

end NUMINAMATH_GPT_trig_expression_identity_l2035_203576


namespace NUMINAMATH_GPT_garden_area_increase_l2035_203508

theorem garden_area_increase :
    let length := 60
    let width := 20
    let perimeter := 2 * (length + width)
    let side_of_square := perimeter / 4
    let area_rectangular := length * width
    let area_square := side_of_square * side_of_square
    area_square - area_rectangular = 400 :=
by
  sorry

end NUMINAMATH_GPT_garden_area_increase_l2035_203508


namespace NUMINAMATH_GPT_sum_of_x_and_y_l2035_203546

theorem sum_of_x_and_y (x y : ℕ) (hx : 0 < x) (hy : 0 < y)
(hx15 : x < 15) (hy15 : y < 15) (h : x + y + x * y = 119) : x + y = 21 ∨ x + y = 20 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_x_and_y_l2035_203546


namespace NUMINAMATH_GPT_trigonometric_identity_l2035_203599

variable {α : ℝ}

theorem trigonometric_identity (h : Real.tan α = 3) :
  (2 * Real.sin α - Real.cos α) / (Real.sin α + 3 * Real.cos α) = 5 / 6 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l2035_203599


namespace NUMINAMATH_GPT_expand_polynomial_l2035_203586

variable {x y z : ℝ}

theorem expand_polynomial : (x + 10 * z + 5) * (2 * y + 15) = 2 * x * y + 20 * y * z + 15 * x + 10 * y + 150 * z + 75 :=
  sorry

end NUMINAMATH_GPT_expand_polynomial_l2035_203586


namespace NUMINAMATH_GPT_swimmer_upstream_distance_l2035_203597

theorem swimmer_upstream_distance (v : ℝ) (c : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) 
                                   (downstream_speed : ℝ) (upstream_time : ℝ) : 
  c = 4.5 →
  downstream_distance = 55 →
  downstream_time = 5 →
  downstream_speed = downstream_distance / downstream_time →
  v + c = downstream_speed →
  upstream_time = 5 →
  (v - c) * upstream_time = 10 := 
by
  intro h_c
  intro h_downstream_distance
  intro h_downstream_time
  intro h_downstream_speed
  intro h_effective_downstream
  intro h_upstream_time
  sorry

end NUMINAMATH_GPT_swimmer_upstream_distance_l2035_203597


namespace NUMINAMATH_GPT_minimum_S_l2035_203526

theorem minimum_S (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) :
  S = (a + 1/a)^2 + (b + 1/b)^2 → S ≥ 8 :=
by
  sorry

end NUMINAMATH_GPT_minimum_S_l2035_203526


namespace NUMINAMATH_GPT_no_integer_n_such_that_squares_l2035_203511

theorem no_integer_n_such_that_squares :
  ¬ ∃ n : ℤ, (∃ k1 : ℤ, 10 * n - 1 = k1 ^ 2) ∧
             (∃ k2 : ℤ, 13 * n - 1 = k2 ^ 2) ∧
             (∃ k3 : ℤ, 85 * n - 1 = k3 ^ 2) := 
by sorry

end NUMINAMATH_GPT_no_integer_n_such_that_squares_l2035_203511


namespace NUMINAMATH_GPT_scientific_notation_correct_l2035_203519

-- Define the problem conditions
def original_number : ℝ := 6175700

-- Define the expected output in scientific notation
def scientific_notation_representation (x : ℝ) : Prop :=
  x = 6.1757 * 10^6

-- The theorem to prove
theorem scientific_notation_correct : scientific_notation_representation original_number :=
by sorry

end NUMINAMATH_GPT_scientific_notation_correct_l2035_203519


namespace NUMINAMATH_GPT_find_a_l2035_203562

noncomputable def f (x a : ℝ) : ℝ := x / (x^2 + a)

theorem find_a (a : ℝ) (h_positive : a > 0) (h_max : ∀ x, x ∈ Set.Ici 1 → f x a ≤ f 1 a) :
  a = Real.sqrt 3 - 1 := by
  sorry

end NUMINAMATH_GPT_find_a_l2035_203562


namespace NUMINAMATH_GPT_sin_alpha_value_l2035_203569

open Real

theorem sin_alpha_value (α : ℝ) (h₁ : 0 < α ∧ α < π / 2) (h₂ : cos (α + π / 4) = 1 / 3) :
  sin α = (4 - sqrt 2) / 6 :=
sorry

end NUMINAMATH_GPT_sin_alpha_value_l2035_203569


namespace NUMINAMATH_GPT_average_score_group2_l2035_203513

-- Total number of students
def total_students : ℕ := 50

-- Overall average score
def overall_average_score : ℝ := 92

-- Number of students from 1 to 30
def group1_students : ℕ := 30

-- Average score of students from 1 to 30
def group1_average_score : ℝ := 90

-- Total number of students - group1_students = 50 - 30 = 20
def group2_students : ℕ := total_students - group1_students

-- Lean 4 statement to prove the average score of students with student numbers 31 to 50 is 95
theorem average_score_group2 :
  (overall_average_score * total_students = group1_average_score * group1_students + x * group2_students) →
  x = 95 :=
sorry

end NUMINAMATH_GPT_average_score_group2_l2035_203513


namespace NUMINAMATH_GPT_original_cost_of_statue_l2035_203575

theorem original_cost_of_statue (sale_price : ℝ) (profit_percent : ℝ) (original_cost : ℝ) 
  (h1 : sale_price = 620) 
  (h2 : profit_percent = 0.25) 
  (h3 : sale_price = (1 + profit_percent) * original_cost) : 
  original_cost = 496 :=
by
  sorry

end NUMINAMATH_GPT_original_cost_of_statue_l2035_203575


namespace NUMINAMATH_GPT_pens_distribution_l2035_203573

theorem pens_distribution (friends : ℕ) (pens : ℕ) (at_least_one : ℕ) 
  (h1 : friends = 4) (h2 : pens = 10) (h3 : at_least_one = 1) 
  (h4 : ∀ f : ℕ, f < friends → at_least_one ≤ f) :
  ∃ ways : ℕ, ways = 142 := 
sorry

end NUMINAMATH_GPT_pens_distribution_l2035_203573


namespace NUMINAMATH_GPT_M1_M2_product_l2035_203520

theorem M1_M2_product (M_1 M_2 : ℝ) :
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ 3 →
  (42 * x - 51) / (x^2 - 5 * x + 6) = (M_1 / (x - 2)) + (M_2 / (x - 3))) →
  M_1 * M_2 = -2981.25 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_M1_M2_product_l2035_203520
