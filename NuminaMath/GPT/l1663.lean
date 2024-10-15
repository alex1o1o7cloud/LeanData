import Mathlib

namespace NUMINAMATH_GPT_largest_positive_integer_n_exists_l1663_166384

theorem largest_positive_integer_n_exists (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ n, 
    0 < n ∧ 
    (n^2 = x^2 + y^2 + z^2 + 2 * x * y + 2 * y * z + 2 * z * x + 4 * x + 4 * y + 4 * z - 9) ∧ 
    ∀ m, 0 < m → 
      (m^2 = x^2 + y^2 + z^2 + 2 * x * y + 2 * y * z + 2 * z * x + 4 * x + 4 * y + 4 * z - 9) → 
      m ≤ n :=
  sorry

end NUMINAMATH_GPT_largest_positive_integer_n_exists_l1663_166384


namespace NUMINAMATH_GPT_least_possible_sum_of_bases_l1663_166319

theorem least_possible_sum_of_bases : 
  ∃ (c d : ℕ), (2 * c + 9 = 9 * d + 2) ∧ (c + d = 13) :=
by
  sorry

end NUMINAMATH_GPT_least_possible_sum_of_bases_l1663_166319


namespace NUMINAMATH_GPT_max_lines_with_specific_angles_l1663_166389

def intersecting_lines : ℕ := 6

theorem max_lines_with_specific_angles :
  ∀ (n : ℕ), (∀ (i j : ℕ), i ≠ j → (∃ θ : ℝ, θ = 30 ∨ θ = 60 ∨ θ = 90)) → n ≤ 6 :=
  sorry

end NUMINAMATH_GPT_max_lines_with_specific_angles_l1663_166389


namespace NUMINAMATH_GPT_monotonically_increasing_interval_l1663_166307

noncomputable def f (x : ℝ) : ℝ := Real.log (-3 * x^2 + 4 * x + 4)

theorem monotonically_increasing_interval :
  ∀ x, x ∈ Set.Ioc (-2/3 : ℝ) (2/3 : ℝ) → MonotoneOn f (Set.Ioc (-2/3) (2/3)) :=
sorry

end NUMINAMATH_GPT_monotonically_increasing_interval_l1663_166307


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_for_gt_l1663_166399

variable {a b : ℝ}

theorem necessary_but_not_sufficient_for_gt : a > b → a > b - 1 :=
by sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_for_gt_l1663_166399


namespace NUMINAMATH_GPT_nursing_home_beds_l1663_166391

/-- A community plans to build a nursing home with 100 rooms, consisting of single, double, and triple rooms.
    Let t be the number of single rooms (1 nursing bed), double rooms (2 nursing beds) is twice the single rooms,
    and the rest are triple rooms (3 nursing beds).
    The equations are:
    - number of double rooms: 2 * t
    - number of single rooms: t
    - number of triple rooms: 100 - 3 * t
    - total number of nursing beds: t + 2 * (2 * t) + 3 * (100 - 3 * t) 
    Prove the following:
    1. If the total number of nursing beds is 200, then t = 25.
    2. The maximum number of nursing beds is 260.
    3. The minimum number of nursing beds is 180.
-/
theorem nursing_home_beds (t : ℕ) (h1 : 10 ≤ t ∧ t ≤ 30) (total_rooms : ℕ := 100) :
  (∀ total_beds, (total_beds = t + 2 * (2 * t) + 3 * (100 - 3 * t)) → total_beds = 200 → t = 25) ∧
  (∀ max_beds, (max_beds = t + 2 * (2 * t) + 3 * (100 - 3 * t)) → t = 10 → max_beds = 260) ∧
  (∀ min_beds, (min_beds = t + 2 * (2 * t) + 3 * (100 - 3 * t)) → t = 30 → min_beds = 180) := 
by
  sorry

end NUMINAMATH_GPT_nursing_home_beds_l1663_166391


namespace NUMINAMATH_GPT_taxi_service_charge_l1663_166300

theorem taxi_service_charge (initial_fee : ℝ) (additional_charge : ℝ) (increment : ℝ) (total_charge : ℝ) 
  (h_initial_fee : initial_fee = 2.25) 
  (h_additional_charge : additional_charge = 0.4) 
  (h_increment : increment = 2 / 5) 
  (h_total_charge : total_charge = 5.85) : 
  ∃ distance : ℝ, distance = 3.6 :=
by
  sorry

end NUMINAMATH_GPT_taxi_service_charge_l1663_166300


namespace NUMINAMATH_GPT_total_number_of_people_l1663_166363

variables (A B : ℕ)

def pencils_brought_by_assoc_profs (A : ℕ) : ℕ := 2 * A
def pencils_brought_by_asst_profs (B : ℕ) : ℕ := B
def charts_brought_by_assoc_profs (A : ℕ) : ℕ := A
def charts_brought_by_asst_profs (B : ℕ) : ℕ := 2 * B

axiom pencils_total : pencils_brought_by_assoc_profs A + pencils_brought_by_asst_profs B = 10
axiom charts_total : charts_brought_by_assoc_profs A + charts_brought_by_asst_profs B = 11

theorem total_number_of_people : A + B = 7 :=
sorry

end NUMINAMATH_GPT_total_number_of_people_l1663_166363


namespace NUMINAMATH_GPT_num_combinations_two_dresses_l1663_166304

def num_colors : ℕ := 4
def num_patterns : ℕ := 5

def combinations_first_dress : ℕ := num_colors * num_patterns
def combinations_second_dress : ℕ := (num_colors - 1) * (num_patterns - 1)

theorem num_combinations_two_dresses :
  (combinations_first_dress * combinations_second_dress) = 240 := by
  sorry

end NUMINAMATH_GPT_num_combinations_two_dresses_l1663_166304


namespace NUMINAMATH_GPT_probability_of_selection_l1663_166359

-- Problem setup
def number_of_students : ℕ := 54
def number_of_students_eliminated : ℕ := 4
def number_of_remaining_students : ℕ := number_of_students - number_of_students_eliminated
def number_of_students_selected : ℕ := 5

-- Statement to be proved
theorem probability_of_selection :
  (number_of_students_selected : ℚ) / (number_of_students : ℚ) = 5 / 54 :=
sorry

end NUMINAMATH_GPT_probability_of_selection_l1663_166359


namespace NUMINAMATH_GPT_exists_range_of_real_numbers_l1663_166379

theorem exists_range_of_real_numbers (x : ℝ) :
  (x^2 - 5 * x + 7 ≠ 1) ↔ (x ≠ 3 ∧ x ≠ 2) := 
sorry

end NUMINAMATH_GPT_exists_range_of_real_numbers_l1663_166379


namespace NUMINAMATH_GPT_ladder_base_distance_l1663_166393

variable (x : ℕ)
variable (ladder_length : ℕ := 13)
variable (height : ℕ := 12)

theorem ladder_base_distance : ∃ x, (x^2 + height^2 = ladder_length^2) ∧ x = 5 := by
  sorry

end NUMINAMATH_GPT_ladder_base_distance_l1663_166393


namespace NUMINAMATH_GPT_value_expression_at_5_l1663_166339

theorem value_expression_at_5 (x : ℕ) (hx : x = 5) : 2 * x^2 + 4 = 54 :=
by
  -- Adding sorry to skip the proof.
  sorry

end NUMINAMATH_GPT_value_expression_at_5_l1663_166339


namespace NUMINAMATH_GPT_scientific_notation_of_86000000_l1663_166353

theorem scientific_notation_of_86000000 :
  ∃ (x : ℝ) (y : ℤ), 86000000 = x * 10^y ∧ x = 8.6 ∧ y = 7 :=
by
  use 8.6
  use 7
  sorry

end NUMINAMATH_GPT_scientific_notation_of_86000000_l1663_166353


namespace NUMINAMATH_GPT_function_domain_l1663_166396

theorem function_domain (x : ℝ) : x + 3 ≥ 0 ↔ x ≥ -3 := by
  sorry

end NUMINAMATH_GPT_function_domain_l1663_166396


namespace NUMINAMATH_GPT_range_of_a_l1663_166345

variable {f : ℝ → ℝ}

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def increasing_on_non_neg (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), (0 ≤ x ∧ x ≤ y) → f x ≤ f y

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) :
  even_function f → increasing_on_non_neg f → f a ≤ f 2 → -2 ≤ a ∧ a ≤ 2 :=
by
  intro h_even h_increasing h_le
  sorry

end NUMINAMATH_GPT_range_of_a_l1663_166345


namespace NUMINAMATH_GPT_complete_half_job_in_six_days_l1663_166397

theorem complete_half_job_in_six_days (x : ℕ) (h1 : 2 * x = x + 6) : x = 6 :=
  by
    sorry

end NUMINAMATH_GPT_complete_half_job_in_six_days_l1663_166397


namespace NUMINAMATH_GPT_ab_plus_cd_l1663_166378

variable (a b c d : ℝ)

theorem ab_plus_cd (h1 : a + b + c = -4)
                  (h2 : a + b + d = 2)
                  (h3 : a + c + d = 15)
                  (h4 : b + c + d = 10) :
                  a * b + c * d = 485 / 9 :=
by
  sorry

end NUMINAMATH_GPT_ab_plus_cd_l1663_166378


namespace NUMINAMATH_GPT_product_of_16_and_21_point_3_l1663_166321

theorem product_of_16_and_21_point_3 (h1 : 213 * 16 = 3408) : 16 * 21.3 = 340.8 :=
by sorry

end NUMINAMATH_GPT_product_of_16_and_21_point_3_l1663_166321


namespace NUMINAMATH_GPT_base_b_conversion_l1663_166312

theorem base_b_conversion (b : ℝ) (h₁ : 1 * 5^2 + 3 * 5^1 + 2 * 5^0 = 42) (h₂ : 2 * b^2 + 2 * b + 1 = 42) :
  b = (-1 + Real.sqrt 83) / 2 := 
  sorry

end NUMINAMATH_GPT_base_b_conversion_l1663_166312


namespace NUMINAMATH_GPT_f_periodic_with_period_one_l1663_166394

noncomputable def is_periodic (f : ℝ → ℝ) :=
  ∃ c : ℝ, c ≠ 0 ∧ ∀ x : ℝ, f (x + c) = f x

theorem f_periodic_with_period_one
  (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, |f x| ≤ 1)
  (h2 : ∀ x : ℝ, f (x + 13/42) + f x = f (x + 1/6) + f (x + 1/7)) :
  is_periodic f := 
sorry

end NUMINAMATH_GPT_f_periodic_with_period_one_l1663_166394


namespace NUMINAMATH_GPT_geometric_seq_value_l1663_166349

variable (a : ℕ → ℝ)
variable (g : ∀ n m : ℕ, a n * a m = a ((n + m) / 2) ^ 2)

theorem geometric_seq_value (h1 : a 2 = 1 / 3) (h2 : a 8 = 27) : a 5 = 3 ∨ a 5 = -3 := by
  sorry

end NUMINAMATH_GPT_geometric_seq_value_l1663_166349


namespace NUMINAMATH_GPT_fermat_prime_sum_not_possible_l1663_166318

-- Definitions of the conditions
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, (m ∣ p) → (m = 1 ∨ m = p)

-- The Lean statement
theorem fermat_prime_sum_not_possible 
  (n : ℕ) (x y z : ℤ) (p : ℕ) 
  (h_odd : is_odd n) 
  (h_gt_one : n > 1) 
  (h_prime : is_prime p)
  (h_sum: x + y = ↑p) :
  ¬ (x ^ n + y ^ n = z ^ n) :=
by
  sorry


end NUMINAMATH_GPT_fermat_prime_sum_not_possible_l1663_166318


namespace NUMINAMATH_GPT_binom_20_4_l1663_166327

theorem binom_20_4 : Nat.choose 20 4 = 4845 := by
  sorry

end NUMINAMATH_GPT_binom_20_4_l1663_166327


namespace NUMINAMATH_GPT_financing_amount_correct_l1663_166377

-- Define the conditions
def monthly_payment : ℕ := 150
def years : ℕ := 5
def months_per_year : ℕ := 12

-- Define the total financed amount
def total_financed : ℕ := monthly_payment * years * months_per_year

-- The statement that we need to prove
theorem financing_amount_correct : total_financed = 9000 := 
by
  sorry

end NUMINAMATH_GPT_financing_amount_correct_l1663_166377


namespace NUMINAMATH_GPT_inverse_of_5_mod_34_l1663_166310

theorem inverse_of_5_mod_34 : ∃ x : ℕ, (5 * x) % 34 = 1 ∧ 0 ≤ x ∧ x < 34 :=
by
  use 7
  have h : (5 * 7) % 34 = 1 := by sorry
  exact ⟨h, by norm_num, by norm_num⟩

end NUMINAMATH_GPT_inverse_of_5_mod_34_l1663_166310


namespace NUMINAMATH_GPT_volume_of_region_l1663_166330

-- Define the conditions
def condition1 (x y z : ℝ) := abs (x + y + 2 * z) + abs (x + y - 2 * z) ≤ 12
def condition2 (x : ℝ) := x ≥ 0
def condition3 (y : ℝ) := y ≥ 0
def condition4 (z : ℝ) := z ≥ 0

-- Define the volume function
def volume (x y z : ℝ) := 18 * 3

-- Proof statement
theorem volume_of_region : ∀ (x y z : ℝ),
  condition1 x y z →
  condition2 x →
  condition3 y →
  condition4 z →
  volume x y z = 54 := by
  sorry

end NUMINAMATH_GPT_volume_of_region_l1663_166330


namespace NUMINAMATH_GPT_tan_double_angle_l1663_166390

theorem tan_double_angle (θ : ℝ) (P : ℝ × ℝ) 
  (h_vertex : θ = 0) 
  (h_initial_side : ∀ x, θ = x)
  (h_terminal_side : P = (-1, 2)) : 
  Real.tan (2 * θ) = 4 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_tan_double_angle_l1663_166390


namespace NUMINAMATH_GPT_lines_perpendicular_to_same_plane_are_parallel_l1663_166368

theorem lines_perpendicular_to_same_plane_are_parallel 
  (parallel_proj_parallel_lines : Prop)
  (planes_parallel_to_same_line : Prop)
  (planes_perpendicular_to_same_plane : Prop)
  (lines_perpendicular_to_same_plane : Prop) 
  (h1 : ¬ parallel_proj_parallel_lines)
  (h2 : ¬ planes_parallel_to_same_line)
  (h3 : ¬ planes_perpendicular_to_same_plane) :
  lines_perpendicular_to_same_plane := 
sorry

end NUMINAMATH_GPT_lines_perpendicular_to_same_plane_are_parallel_l1663_166368


namespace NUMINAMATH_GPT_domain_of_function_l1663_166326

def domain_condition_1 (x : ℝ) : Prop := 1 - x > 0
def domain_condition_2 (x : ℝ) : Prop := x + 3 ≥ 0

def in_domain (x : ℝ) : Prop := domain_condition_1 x ∧ domain_condition_2 x

theorem domain_of_function : ∀ x : ℝ, in_domain x ↔ (-3 : ℝ) ≤ x ∧ x < 1 := 
by sorry

end NUMINAMATH_GPT_domain_of_function_l1663_166326


namespace NUMINAMATH_GPT_prob_both_correct_l1663_166373

def prob_A : ℤ := 70
def prob_B : ℤ := 55
def prob_neither : ℤ := 20

theorem prob_both_correct : (prob_A + prob_B - (100 - prob_neither)) = 45 :=
by
  sorry

end NUMINAMATH_GPT_prob_both_correct_l1663_166373


namespace NUMINAMATH_GPT_average_decrease_l1663_166325

theorem average_decrease (avg_6 : ℝ) (obs_7 : ℝ) (new_avg : ℝ) (decrease : ℝ) :
  avg_6 = 11 → obs_7 = 4 → (6 * avg_6 + obs_7) / 7 = new_avg → avg_6 - new_avg = decrease → decrease = 1 :=
  by
    intros h1 h2 h3 h4
    rw [h1, h2] at *
    sorry

end NUMINAMATH_GPT_average_decrease_l1663_166325


namespace NUMINAMATH_GPT_polynomial_expansion_identity_l1663_166383

variable (a0 a1 a2 a3 a4 : ℝ)

theorem polynomial_expansion_identity
  (h : (2 - (x : ℝ))^4 = a0 + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4) :
  a0 - a1 + a2 - a3 + a4 = 81 :=
sorry

end NUMINAMATH_GPT_polynomial_expansion_identity_l1663_166383


namespace NUMINAMATH_GPT_max_diagonals_in_chessboard_l1663_166392

/-- The maximum number of non-intersecting diagonals that can be drawn in an 8x8 chessboard is 36. -/
theorem max_diagonals_in_chessboard : 
  ∃ (diagonals : Finset (ℕ × ℕ)), 
  diagonals.card = 36 ∧ 
  ∀ (d1 d2 : ℕ × ℕ), d1 ∈ diagonals → d2 ∈ diagonals → d1 ≠ d2 → d1.fst ≠ d2.fst ∧ d1.snd ≠ d2.snd := 
  sorry

end NUMINAMATH_GPT_max_diagonals_in_chessboard_l1663_166392


namespace NUMINAMATH_GPT_incorrect_reasoning_C_l1663_166358

theorem incorrect_reasoning_C
  {Point : Type} {Line Plane : Type}
  (A B : Point) (l : Line) (α β : Plane)
  (in_line : Point → Line → Prop)
  (in_plane : Point → Plane → Prop)
  (line_in_plane : Line → Plane → Prop)
  (disjoint : Line → Plane → Prop) :

  ¬(line_in_plane l α) ∧ in_line A l ∧ in_plane A α :=
sorry

end NUMINAMATH_GPT_incorrect_reasoning_C_l1663_166358


namespace NUMINAMATH_GPT_factorize_x_squared_minus_four_l1663_166366

theorem factorize_x_squared_minus_four (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) :=
by
  sorry

end NUMINAMATH_GPT_factorize_x_squared_minus_four_l1663_166366


namespace NUMINAMATH_GPT_sum_of_tangents_l1663_166380

noncomputable def function_f (x : ℝ) : ℝ :=
  max (max (4 * x + 20) (-x + 2)) (5 * x - 3)

theorem sum_of_tangents (q : ℝ → ℝ) (a b c : ℝ) (h1 : ∀ x, q x - (4 * x + 20) = q x - function_f x)
  (h2 : ∀ x, q x - (-x + 2) = q x - function_f x)
  (h3 : ∀ x, q x - (5 * x - 3) = q x - function_f x) :
  a + b + c = -83 / 10 :=
sorry

end NUMINAMATH_GPT_sum_of_tangents_l1663_166380


namespace NUMINAMATH_GPT_jenna_stamp_division_l1663_166302

theorem jenna_stamp_division (a b c : ℕ) (h₁ : a = 945) (h₂ : b = 1260) (h₃ : c = 630) :
  Nat.gcd (Nat.gcd a b) c = 105 :=
by
  rw [h₁, h₂, h₃]
  -- Now we need to prove Nat.gcd (Nat.gcd 945 1260) 630 = 105
  sorry

end NUMINAMATH_GPT_jenna_stamp_division_l1663_166302


namespace NUMINAMATH_GPT_find_correct_result_l1663_166331

noncomputable def correct_result : Prop :=
  ∃ (x : ℝ), (-1.25 * x - 0.25 = 1.25 * x) ∧ (-1.25 * x = 0.125)

theorem find_correct_result : correct_result :=
  sorry

end NUMINAMATH_GPT_find_correct_result_l1663_166331


namespace NUMINAMATH_GPT_smallest_value_a1_l1663_166328

def seq (a : ℕ → ℝ) : Prop :=
  ∀ n > 1, a n = 7 * a (n-1) - 2 * n

theorem smallest_value_a1 (a : ℕ → ℝ) (h1 : ∀ n, 0 < a n) (h2 : seq a) : 
  a 1 ≥ 13 / 18 :=
sorry

end NUMINAMATH_GPT_smallest_value_a1_l1663_166328


namespace NUMINAMATH_GPT_savings_promotion_l1663_166303

theorem savings_promotion (reg_price promo_price num_pizzas : ℕ) (h1 : reg_price = 18) (h2 : promo_price = 5) (h3 : num_pizzas = 3) :
  reg_price * num_pizzas - promo_price * num_pizzas = 39 := by
  sorry

end NUMINAMATH_GPT_savings_promotion_l1663_166303


namespace NUMINAMATH_GPT_amount_sharpened_off_l1663_166374

-- Defining the initial length of the pencil
def initial_length : ℕ := 31

-- Defining the length of the pencil after sharpening
def after_sharpening_length : ℕ := 14

-- Proving the amount sharpened off the pencil
theorem amount_sharpened_off : initial_length - after_sharpening_length = 17 := 
by 
  -- Here we would insert the proof steps, 
  -- but as instructed we leave it as sorry.
  sorry

end NUMINAMATH_GPT_amount_sharpened_off_l1663_166374


namespace NUMINAMATH_GPT_distinct_diagonals_in_convex_nonagon_l1663_166371

-- Definitions
def is_convex_nonagon (n : ℕ) : Prop := n = 9

-- Main theorem stating the number of distinct diagonals in a convex nonagon
theorem distinct_diagonals_in_convex_nonagon (n : ℕ) (h : is_convex_nonagon n) : 
  ∃ d : ℕ, d = 27 :=
by
  -- Use Lean constructs to formalize the proof
  sorry

end NUMINAMATH_GPT_distinct_diagonals_in_convex_nonagon_l1663_166371


namespace NUMINAMATH_GPT_center_of_circle_l1663_166336

noncomputable def center_is_correct (x y : ℚ) : Prop :=
  (5 * x - 2 * y = -10) ∧ (3 * x + y = 0)

theorem center_of_circle : center_is_correct (-10 / 11) (30 / 11) :=
by
  sorry

end NUMINAMATH_GPT_center_of_circle_l1663_166336


namespace NUMINAMATH_GPT_common_ratio_geometric_series_l1663_166334

theorem common_ratio_geometric_series
  (a₁ a₂ a₃ : ℚ)
  (h₁ : a₁ = 7 / 8)
  (h₂ : a₂ = -14 / 27)
  (h₃ : a₃ = 56 / 81) :
  (a₂ / a₁ = a₃ / a₂) ∧ (a₂ / a₁ = -2 / 3) :=
by
  -- The proof will follow here
  sorry

end NUMINAMATH_GPT_common_ratio_geometric_series_l1663_166334


namespace NUMINAMATH_GPT_total_attendance_l1663_166367

theorem total_attendance (first_concert : ℕ) (second_concert : ℕ) (third_concert : ℕ) :
  first_concert = 65899 →
  second_concert = first_concert + 119 →
  third_concert = 2 * second_concert →
  first_concert + second_concert + third_concert = 263953 :=
by
  intros h_first h_second h_third
  rw [h_first, h_second, h_third]
  sorry

end NUMINAMATH_GPT_total_attendance_l1663_166367


namespace NUMINAMATH_GPT_megan_popsicles_consumed_l1663_166314

noncomputable def popsicles_consumed_in_time_period (time: ℕ) (interval: ℕ) : ℕ :=
  (time / interval)

theorem megan_popsicles_consumed:
  popsicles_consumed_in_time_period 315 30 = 10 :=
by
  sorry

end NUMINAMATH_GPT_megan_popsicles_consumed_l1663_166314


namespace NUMINAMATH_GPT_find_k_for_parallel_vectors_l1663_166308

theorem find_k_for_parallel_vectors (k : ℝ) :
  let a := (1, k)
  let b := (9, k - 6)
  (1 * (k - 6) - 9 * k = 0) → k = -3 / 4 :=
by
  intros a b parallel_cond
  sorry

end NUMINAMATH_GPT_find_k_for_parallel_vectors_l1663_166308


namespace NUMINAMATH_GPT_dice_sum_not_11_l1663_166365

/-- Jeremy rolls three standard six-sided dice, with each showing a different number and the product of the numbers on the upper faces is 72.
    Prove that the sum 11 is not possible. --/
theorem dice_sum_not_11 : 
  ∃ (a b c : ℕ), 
    (1 ≤ a ∧ a ≤ 6) ∧ 
    (1 ≤ b ∧ b ≤ 6) ∧ 
    (1 ≤ c ∧ c ≤ 6) ∧ 
    (a ≠ b ∧ a ≠ c ∧ b ≠ c) ∧ 
    (a * b * c = 72) ∧ 
    (a > 4 ∨ b > 4 ∨ c > 4) → 
    a + b + c ≠ 11 := 
by
  sorry

end NUMINAMATH_GPT_dice_sum_not_11_l1663_166365


namespace NUMINAMATH_GPT_incorrect_intersections_l1663_166311

theorem incorrect_intersections :
  (∃ x, (x = x ∧ x = Real.sqrt (x + 2)) ↔ x = 1 ∨ x = 2) →
  (∃ x, (x^2 - 3 * x + 2 = 2 ∧ x = 2) ↔ x = 1 ∨ x = 2) →
  (∃ x, (Real.sin x = 3 * x - 4 ∧ x = 2) ↔ x = 1 ∨ x = 2) → False :=
by {
  sorry
}

end NUMINAMATH_GPT_incorrect_intersections_l1663_166311


namespace NUMINAMATH_GPT_two_presses_printing_time_l1663_166305

def printing_time (presses newspapers hours : ℕ) : ℕ := sorry

theorem two_presses_printing_time :
  ∀ (presses newspapers hours : ℕ),
    (presses = 4) →
    (newspapers = 8000) →
    (hours = 6) →
    printing_time 2 6000 hours = 9 := sorry

end NUMINAMATH_GPT_two_presses_printing_time_l1663_166305


namespace NUMINAMATH_GPT_last_three_digits_of_7_pow_103_l1663_166352

theorem last_three_digits_of_7_pow_103 : (7^103) % 1000 = 327 :=
by
  sorry

end NUMINAMATH_GPT_last_three_digits_of_7_pow_103_l1663_166352


namespace NUMINAMATH_GPT_smallest_four_digit_number_divisible_by_40_l1663_166350

theorem smallest_four_digit_number_divisible_by_40 : ∃ n : ℕ, n >= 1000 ∧ n < 10000 ∧ n % 40 = 0 ∧ ∀ m : ℕ, m >= 1000 ∧ m < 10000 ∧ m % 40 = 0 → n <= m :=
by
  use 1000
  sorry

end NUMINAMATH_GPT_smallest_four_digit_number_divisible_by_40_l1663_166350


namespace NUMINAMATH_GPT_polygon_coloring_l1663_166332

theorem polygon_coloring (n m : ℕ) (h1 : n ≥ 3) (h2 : m ≥ 3) :
    ∃ b_n : ℕ, b_n = (m - 1) * ((m - 1) ^ (n - 1) + (-1 : ℤ) ^ n) :=
sorry

end NUMINAMATH_GPT_polygon_coloring_l1663_166332


namespace NUMINAMATH_GPT_quadratic_root_m_eq_neg_fourteen_l1663_166344

theorem quadratic_root_m_eq_neg_fourteen : ∀ (m : ℝ), (∃ x : ℝ, x = 2 ∧ x^2 + 5 * x + m = 0) → m = -14 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_root_m_eq_neg_fourteen_l1663_166344


namespace NUMINAMATH_GPT_temperature_in_quebec_city_is_negative_8_l1663_166369

def temperature_vancouver : ℝ := 22
def temperature_calgary (temperature_vancouver : ℝ) : ℝ := temperature_vancouver - 19
def temperature_quebec_city (temperature_calgary : ℝ) : ℝ := temperature_calgary - 11

theorem temperature_in_quebec_city_is_negative_8 :
  temperature_quebec_city (temperature_calgary temperature_vancouver) = -8 := by
  sorry

end NUMINAMATH_GPT_temperature_in_quebec_city_is_negative_8_l1663_166369


namespace NUMINAMATH_GPT_carpenters_time_l1663_166362

theorem carpenters_time (t1 t2 t3 t4 : ℝ) (ht1 : t1 = 1) (ht2 : t2 = 2)
  (ht3 : t3 = 3) (ht4 : t4 = 4) : (1 / (1 / t1 + 1 / t2 + 1 / t3 + 1 / t4)) = 12 / 25 := by
  sorry

end NUMINAMATH_GPT_carpenters_time_l1663_166362


namespace NUMINAMATH_GPT_smallest_positive_angle_l1663_166347

theorem smallest_positive_angle (k : ℤ) : ∃ α, α = 400 + k * 360 ∧ α > 0 ∧ α = 40 :=
by
  use 40
  sorry

end NUMINAMATH_GPT_smallest_positive_angle_l1663_166347


namespace NUMINAMATH_GPT_four_kids_wash_three_whiteboards_in_20_minutes_l1663_166398

-- Condition: It takes one kid 160 minutes to wash six whiteboards
def time_per_whiteboard_for_one_kid : ℚ := 160 / 6

-- Calculation involving four kids
def time_per_whiteboard_for_four_kids : ℚ := time_per_whiteboard_for_one_kid / 4

-- The total time it takes for four kids to wash three whiteboards together
def total_time_for_four_kids_washing_three_whiteboards : ℚ := time_per_whiteboard_for_four_kids * 3

-- Statement to prove
theorem four_kids_wash_three_whiteboards_in_20_minutes : 
  total_time_for_four_kids_washing_three_whiteboards = 20 :=
by
  sorry

end NUMINAMATH_GPT_four_kids_wash_three_whiteboards_in_20_minutes_l1663_166398


namespace NUMINAMATH_GPT_find_int_solutions_l1663_166385

theorem find_int_solutions (x y : ℤ) (h : x^4 - 2 * y^2 = 1) : (x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = 0) :=
sorry

end NUMINAMATH_GPT_find_int_solutions_l1663_166385


namespace NUMINAMATH_GPT_sum_and_ratio_l1663_166323

theorem sum_and_ratio (x y : ℝ) (h1 : x + y = 480) (h2 : x / y = 0.8) : y - x = 53.34 :=
by
  sorry

end NUMINAMATH_GPT_sum_and_ratio_l1663_166323


namespace NUMINAMATH_GPT_square_of_sum_opposite_l1663_166343

theorem square_of_sum_opposite (a b : ℝ) : (-(a) + b)^2 = (-a + b)^2 :=
by
  sorry

end NUMINAMATH_GPT_square_of_sum_opposite_l1663_166343


namespace NUMINAMATH_GPT_sum_of_possible_values_of_x_l1663_166375

theorem sum_of_possible_values_of_x :
  ∀ x : ℝ, (x + 2) * (x - 3) = 20 → ∃ s, s = 1 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_possible_values_of_x_l1663_166375


namespace NUMINAMATH_GPT_apartment_building_floors_l1663_166381

theorem apartment_building_floors (K E P : ℕ) (h1 : 1 < K) (h2 : K < E) (h3 : E < P) (h4 : K * E * P = 715) : 
  E = 11 :=
sorry

end NUMINAMATH_GPT_apartment_building_floors_l1663_166381


namespace NUMINAMATH_GPT_equal_distances_sum_of_distances_moving_distances_equal_l1663_166395

-- Define the points A, B, origin O, and moving point P
def A : ℝ := -1
def B : ℝ := 3
def O : ℝ := 0

-- Define the moving point P
def P (x : ℝ) : ℝ := x

-- Define the velocities of each point
def vP : ℝ := -1
def vA : ℝ := -5
def vB : ℝ := -20

-- Proof statement ①: Distance from P to A and B are equal implies x = 1
theorem equal_distances (x : ℝ) (h : abs (x + 1) = abs (x - 3)) : x = 1 :=
sorry

-- Proof statement ②: Sum of distances from P to A and B is 5 implies x = -3/2 or 7/2
theorem sum_of_distances (x : ℝ) (h : abs (x + 1) + abs (x - 3) = 5) : x = -3/2 ∨ x = 7/2 :=
sorry

-- Proof statement ③: Moving distances equal at times t = 4/15 or 2/23
theorem moving_distances_equal (t : ℝ) (h : abs (4 * t + 1) = abs (19 * t - 3)) : t = 4/15 ∨ t = 2/23 :=
sorry

end NUMINAMATH_GPT_equal_distances_sum_of_distances_moving_distances_equal_l1663_166395


namespace NUMINAMATH_GPT_deaths_during_operation_l1663_166341

noncomputable def initial_count : ℕ := 1000
noncomputable def first_day_remaining (n : ℕ) := 5 * n / 6
noncomputable def second_day_remaining (n : ℕ) := (35 * n / 48) - 1
noncomputable def third_day_remaining (n : ℕ) := (105 * n / 192) - 3 / 4

theorem deaths_during_operation : ∃ n : ℕ, initial_count - n = 472 ∧ n = 528 :=
  by sorry

end NUMINAMATH_GPT_deaths_during_operation_l1663_166341


namespace NUMINAMATH_GPT_carson_seed_l1663_166357

variable (s f : ℕ) -- Define the variables s and f as nonnegative integers

-- Conditions given in the problem
axiom h1 : s = 3 * f
axiom h2 : s + f = 60

-- The theorem to prove
theorem carson_seed : s = 45 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_carson_seed_l1663_166357


namespace NUMINAMATH_GPT_twenty_percent_greater_l1663_166333

theorem twenty_percent_greater (x : ℕ) : 
  x = 80 + (20 * 80 / 100) → x = 96 :=
by
  sorry

end NUMINAMATH_GPT_twenty_percent_greater_l1663_166333


namespace NUMINAMATH_GPT_f_3_2_plus_f_5_1_l1663_166337

def f (a b : ℤ) : ℚ :=
  if a - b ≤ 2 then (a * b - a - 1) / (3 * a)
  else (a * b + b - 1) / (-3 * b)

theorem f_3_2_plus_f_5_1 :
  f 3 2 + f 5 1 = -13 / 9 :=
by
  sorry

end NUMINAMATH_GPT_f_3_2_plus_f_5_1_l1663_166337


namespace NUMINAMATH_GPT_range_of_m_if_neg_proposition_false_l1663_166329

theorem range_of_m_if_neg_proposition_false :
  (¬ ∃ x_0 : ℝ, x_0^2 + m * x_0 + 2 * m - 3 < 0) ↔ (2 ≤ m ∧ m ≤ 6) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_if_neg_proposition_false_l1663_166329


namespace NUMINAMATH_GPT_bananas_to_pears_ratio_l1663_166386

theorem bananas_to_pears_ratio (B P : ℕ) (hP : P = 50) (h1 : B + 10 = 160) (h2: ∃ k : ℕ, B = k * P) : B / P = 3 :=
by
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_bananas_to_pears_ratio_l1663_166386


namespace NUMINAMATH_GPT_book_distribution_l1663_166388

theorem book_distribution (x : ℕ) (books : ℕ) :
  (books = 3 * x + 8) ∧ (books < 5 * x - 5 + 2) → (x = 6 ∧ books = 26) :=
by
  sorry

end NUMINAMATH_GPT_book_distribution_l1663_166388


namespace NUMINAMATH_GPT_workman_B_days_l1663_166322

theorem workman_B_days (A B : ℝ) (hA : A = (1 / 2) * B) (hTogether : (A + B) * 14 = 1) :
  1 / B = 21 :=
sorry

end NUMINAMATH_GPT_workman_B_days_l1663_166322


namespace NUMINAMATH_GPT_cube_path_length_l1663_166361

noncomputable def path_length_dot_cube : ℝ :=
  let edge_length := 2
  let radius1 := Real.sqrt 5
  let radius2 := 1
  (radius1 + radius2) * Real.pi

theorem cube_path_length :
  path_length_dot_cube = (Real.sqrt 5 + 1) * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_cube_path_length_l1663_166361


namespace NUMINAMATH_GPT_parabola_intersects_x_axis_expression_l1663_166364

theorem parabola_intersects_x_axis_expression (m : ℝ) (h : m^2 - m - 1 = 0) : m^2 - m + 2017 = 2018 := 
by 
  sorry

end NUMINAMATH_GPT_parabola_intersects_x_axis_expression_l1663_166364


namespace NUMINAMATH_GPT_sonika_years_in_bank_l1663_166356

variable (P A1 A2 : ℚ)
variables (r t : ℚ)

def simple_interest (P r t : ℚ) : ℚ := P * r * t / 100
def amount_with_interest (P r t : ℚ) : ℚ := P + simple_interest P r t

theorem sonika_years_in_bank :
  P = 9000 → A1 = 10200 → A2 = 10740 →
  amount_with_interest P r t = A1 →
  amount_with_interest P (r + 2) t = A2 →
  t = 3 :=
by
  intros hP hA1 hA2 hA1_eq hA2_eq
  sorry

end NUMINAMATH_GPT_sonika_years_in_bank_l1663_166356


namespace NUMINAMATH_GPT_least_number_remainder_l1663_166348

theorem least_number_remainder (n : ℕ) :
  (n % 7 = 4) ∧ (n % 9 = 4) ∧ (n % 12 = 4) ∧ (n % 18 = 4) → n = 256 :=
by
  sorry

end NUMINAMATH_GPT_least_number_remainder_l1663_166348


namespace NUMINAMATH_GPT_unit_vector_perpendicular_l1663_166313

theorem unit_vector_perpendicular (x y : ℝ) (h : 3 * x + 4 * y = 0) (m : x^2 + y^2 = 1) : 
  (x = -4/5 ∧ y = 3/5) ∨ (x = 4/5 ∧ y = -3/5) :=
by
  sorry

end NUMINAMATH_GPT_unit_vector_perpendicular_l1663_166313


namespace NUMINAMATH_GPT_janet_roses_l1663_166370

def total_flowers (used_flowers extra_flowers : Nat) : Nat :=
  used_flowers + extra_flowers

def number_of_roses (total tulips : Nat) : Nat :=
  total - tulips

theorem janet_roses :
  ∀ (used_flowers extra_flowers tulips : Nat),
  used_flowers = 11 → extra_flowers = 4 → tulips = 4 →
  number_of_roses (total_flowers used_flowers extra_flowers) tulips = 11 :=
by
  intros used_flowers extra_flowers tulips h_used h_extra h_tulips
  rw [h_used, h_extra, h_tulips]
  -- proof steps skipped
  sorry

end NUMINAMATH_GPT_janet_roses_l1663_166370


namespace NUMINAMATH_GPT_polyhedron_edges_l1663_166320

theorem polyhedron_edges (F V E : ℕ) (h1 : F = 12) (h2 : V = 20) (h3 : F + V = E + 2) : E = 30 :=
by
  -- Additional details would go here, proof omitted as instructed.
  sorry

end NUMINAMATH_GPT_polyhedron_edges_l1663_166320


namespace NUMINAMATH_GPT_isosceles_triangle_side_length_l1663_166317

theorem isosceles_triangle_side_length (total_length : ℝ) (one_side_length : ℝ) (remaining_wire : ℝ) (equal_side : ℝ) :
  total_length = 20 → one_side_length = 6 → remaining_wire = total_length - one_side_length → remaining_wire / 2 = equal_side →
  equal_side = 7 :=
by
  intros h_total h_one_side h_remaining h_equal_side
  sorry

end NUMINAMATH_GPT_isosceles_triangle_side_length_l1663_166317


namespace NUMINAMATH_GPT_problem_statement_l1663_166346

theorem problem_statement (x1 x2 x3 : ℝ) 
  (h1 : x1 < x2)
  (h2 : x2 < x3)
  (h3 : (45*x1^3 - 4050*x1^2 - 4 = 0) ∧ 
        (45*x2^3 - 4050*x2^2 - 4 = 0) ∧ 
        (45*x3^3 - 4050*x3^2 - 4 = 0)) :
  x2 * (x1 + x3) = 0 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1663_166346


namespace NUMINAMATH_GPT_jason_and_lisa_cards_l1663_166351

-- Define the number of cards Jason originally had
def jason_original_cards (remaining: ℕ) (given_away: ℕ) : ℕ :=
  remaining + given_away

-- Define the number of cards Lisa originally had
def lisa_original_cards (remaining: ℕ) (given_away: ℕ) : ℕ :=
  remaining + given_away

-- State the main theorem to be proved
theorem jason_and_lisa_cards :
  jason_original_cards 4 9 + lisa_original_cards 7 15 = 35 :=
by
  sorry

end NUMINAMATH_GPT_jason_and_lisa_cards_l1663_166351


namespace NUMINAMATH_GPT_medal_award_ways_l1663_166309

open Nat

theorem medal_award_ways :
  let sprinters := 10
  let italians := 4
  let medals := 3
  let gold_medal_ways := choose italians 1
  let remaining_sprinters := sprinters - 1
  let non_italians := remaining_sprinters - (italians - 1)
  let silver_medal_ways := choose non_italians 1
  let new_remaining_sprinters := remaining_sprinters - 1
  let new_non_italians := new_remaining_sprinters - (italians - 1)
  let bronze_medal_ways := choose new_non_italians 1
  gold_medal_ways * silver_medal_ways * bronze_medal_ways = 120 := by
    sorry

end NUMINAMATH_GPT_medal_award_ways_l1663_166309


namespace NUMINAMATH_GPT_sin_alpha_cos_beta_value_l1663_166324

variables {α β : ℝ}

theorem sin_alpha_cos_beta_value 
  (h1 : Real.sin (α + β) = 1/2) 
  (h2 : 2 * Real.sin (α - β) = 1/2) : 
  Real.sin α * Real.cos β = 3/8 := by
sorry

end NUMINAMATH_GPT_sin_alpha_cos_beta_value_l1663_166324


namespace NUMINAMATH_GPT_longer_piece_length_l1663_166315

theorem longer_piece_length (x : ℝ) (h1 : x + (x + 2) = 30) : x + 2 = 16 :=
by sorry

end NUMINAMATH_GPT_longer_piece_length_l1663_166315


namespace NUMINAMATH_GPT_A_days_to_complete_alone_l1663_166355

theorem A_days_to_complete_alone
  (work_left : ℝ := 0.41666666666666663)
  (B_days : ℝ := 20)
  (combined_days : ℝ := 5)
  : ∃ (A_days : ℝ), A_days = 15 := 
by
  sorry

end NUMINAMATH_GPT_A_days_to_complete_alone_l1663_166355


namespace NUMINAMATH_GPT_function_conditions_satisfied_l1663_166301

noncomputable def function_satisfying_conditions : ℝ → ℝ := fun x => -2 * x^2 + 3 * x

theorem function_conditions_satisfied :
  (function_satisfying_conditions 1 = 1) ∧
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ function_satisfying_conditions x = y) ∧
  (∀ x y : ℝ, x > 1 ∧ y = function_satisfying_conditions x → ∃ ε > 0, ∀ δ > 0, (x + δ > 1 → function_satisfying_conditions (x + δ) < y)) :=
by
  sorry

end NUMINAMATH_GPT_function_conditions_satisfied_l1663_166301


namespace NUMINAMATH_GPT_books_borrowed_by_lunchtime_l1663_166372

theorem books_borrowed_by_lunchtime (x : ℕ) :
  (∀ x : ℕ, 100 - x + 40 - 30 = 60) → (x = 50) :=
by
  intro h
  have eqn := h x
  sorry

end NUMINAMATH_GPT_books_borrowed_by_lunchtime_l1663_166372


namespace NUMINAMATH_GPT_identify_ATM_mistakes_additional_security_measures_l1663_166360

-- Define the conditions as Boolean variables representing different mistakes and measures
variables (writing_PIN_on_card : Prop)
variables (using_ATM_despite_difficulty : Prop)
variables (believing_stranger : Prop)
variables (walking_away_without_card : Prop)
variables (use_trustworthy_locations : Prop)
variables (presence_during_transactions : Prop)
variables (enable_SMS_notifications : Prop)
variables (call_bank_for_suspicious_activities : Prop)
variables (be_cautious_of_fake_SMS_alerts : Prop)
variables (store_transaction_receipts : Prop)
variables (shield_PIN : Prop)
variables (use_chipped_cards : Prop)
variables (avoid_high_risk_ATMs : Prop)

-- Prove that the identified mistakes occur given the conditions
theorem identify_ATM_mistakes :
  writing_PIN_on_card ∧ using_ATM_despite_difficulty ∧ 
  believing_stranger ∧ walking_away_without_card := sorry

-- Prove that the additional security measures should be followed
theorem additional_security_measures :
  use_trustworthy_locations ∧ presence_during_transactions ∧ 
  enable_SMS_notifications ∧ call_bank_for_suspicious_activities ∧ 
  be_cautious_of_fake_SMS_alerts ∧ store_transaction_receipts ∧ 
  shield_PIN ∧ use_chipped_cards ∧ avoid_high_risk_ATMs := sorry

end NUMINAMATH_GPT_identify_ATM_mistakes_additional_security_measures_l1663_166360


namespace NUMINAMATH_GPT_inv_matrix_A_l1663_166342

def A : Matrix (Fin 2) (Fin 2) ℚ :=
  ![ ![ -2, 1 ],
     ![ (3/2 : ℚ), -1/2 ] ]

def A_inv : Matrix (Fin 2) (Fin 2) ℚ :=
  ![ ![ 1, 2 ],
     ![ 3, 4 ] ]

theorem inv_matrix_A : A⁻¹ = A_inv := by
  sorry

end NUMINAMATH_GPT_inv_matrix_A_l1663_166342


namespace NUMINAMATH_GPT_fraction_identity_l1663_166354

theorem fraction_identity (N F : ℝ) (hN : N = 8) (h : 0.5 * N = F * N + 2) : F = 1 / 4 :=
by {
  -- proof will go here
  sorry
}

end NUMINAMATH_GPT_fraction_identity_l1663_166354


namespace NUMINAMATH_GPT_total_num_of_cars_l1663_166338

-- Define conditions
def row_from_front := 14
def row_from_left := 19
def row_from_back := 11
def row_from_right := 16

-- Compute total number of rows from front to back
def rows_front_to_back : ℕ := (row_from_front - 1) + 1 + (row_from_back - 1)

-- Compute total number of rows from left to right
def rows_left_to_right : ℕ := (row_from_left - 1) + 1 + (row_from_right - 1)

theorem total_num_of_cars :
  rows_front_to_back = 24 ∧
  rows_left_to_right = 34 ∧
  24 * 34 = 816 :=
by
  sorry

end NUMINAMATH_GPT_total_num_of_cars_l1663_166338


namespace NUMINAMATH_GPT_tenth_graders_science_only_l1663_166387

theorem tenth_graders_science_only (total_students science_students art_students : ℕ) 
  (h1 : total_students = 140) 
  (h2 : science_students = 100) 
  (h3 : art_students = 75) : 
  (science_students - (science_students + art_students - total_students)) = 65 :=
by
  sorry

end NUMINAMATH_GPT_tenth_graders_science_only_l1663_166387


namespace NUMINAMATH_GPT_irreducible_fraction_l1663_166306

theorem irreducible_fraction (n : ℤ) : Int.gcd (2 * n + 1) (3 * n + 1) = 1 :=
sorry

end NUMINAMATH_GPT_irreducible_fraction_l1663_166306


namespace NUMINAMATH_GPT_num_rooms_with_2_windows_l1663_166382

theorem num_rooms_with_2_windows:
  ∃ (num_rooms_with_2_windows: ℕ),
  (∀ (num_rooms_with_4_windows num_rooms_with_3_windows: ℕ), 
    num_rooms_with_4_windows = 5 ∧ 
    num_rooms_with_3_windows = 8 ∧
    4 * num_rooms_with_4_windows + 3 * num_rooms_with_3_windows + 2 * num_rooms_with_2_windows = 122) → 
    num_rooms_with_2_windows = 39 :=
by
  sorry

end NUMINAMATH_GPT_num_rooms_with_2_windows_l1663_166382


namespace NUMINAMATH_GPT_negation_of_universal_statement_l1663_166316

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x ^ 2 ≠ x) ↔ ∃ x : ℝ, x ^ 2 = x :=
by
  sorry

end NUMINAMATH_GPT_negation_of_universal_statement_l1663_166316


namespace NUMINAMATH_GPT_smallest_int_cond_l1663_166376

theorem smallest_int_cond (b : ℕ) :
  (b % 9 = 5) ∧ (b % 11 = 7) → b = 95 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_smallest_int_cond_l1663_166376


namespace NUMINAMATH_GPT_perimeter_of_smaller_rectangle_l1663_166335

theorem perimeter_of_smaller_rectangle :
  ∀ (L W n : ℕ), 
  L = 16 → W = 20 → n = 10 →
  (∃ (x y : ℕ), L % 2 = 0 ∧ W % 5 = 0 ∧ 2 * y = L ∧ 5 * x = W ∧ (L * W) / n = x * y ∧ 2 * (x + y) = 24) :=
by
  intros L W n H1 H2 H3
  use 4, 8
  sorry

end NUMINAMATH_GPT_perimeter_of_smaller_rectangle_l1663_166335


namespace NUMINAMATH_GPT_juanita_loss_l1663_166340

theorem juanita_loss
  (entry_fee : ℝ) (hit_threshold : ℕ) (drum_payment_per_hit : ℝ) (drums_hit : ℕ) :
  entry_fee = 10 →
  hit_threshold = 200 →
  drum_payment_per_hit = 0.025 →
  drums_hit = 300 →
  - (entry_fee - ((drums_hit - hit_threshold) * drum_payment_per_hit)) = 7.50 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_juanita_loss_l1663_166340
