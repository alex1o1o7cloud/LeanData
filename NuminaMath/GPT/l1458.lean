import Mathlib

namespace NUMINAMATH_GPT_rectangular_prism_volume_dependency_l1458_145845

theorem rectangular_prism_volume_dependency (a : ℝ) (V : ℝ) (h : a > 2) :
  V = a * 2 * 1 → (∀ a₀ > 2, a ≠ a₀ → V ≠ a₀ * 2 * 1) :=
by
  sorry

end NUMINAMATH_GPT_rectangular_prism_volume_dependency_l1458_145845


namespace NUMINAMATH_GPT_second_fisherman_more_fish_l1458_145858

-- Define the given conditions
def days_in_season : ℕ := 213
def rate_first_fisherman : ℕ := 3
def rate_second_fisherman_phase_1 : ℕ := 1
def rate_second_fisherman_phase_2 : ℕ := 2
def rate_second_fisherman_phase_3 : ℕ := 4
def days_phase_1 : ℕ := 30
def days_phase_2 : ℕ := 60
def days_phase_3 : ℕ := days_in_season - (days_phase_1 + days_phase_2)

-- Define the total number of fish caught by each fisherman
def total_fish_first_fisherman : ℕ := rate_first_fisherman * days_in_season
def total_fish_second_fisherman : ℕ := 
  (rate_second_fisherman_phase_1 * days_phase_1) + 
  (rate_second_fisherman_phase_2 * days_phase_2) + 
  (rate_second_fisherman_phase_3 * days_phase_3)

-- Define the theorem statement
theorem second_fisherman_more_fish : 
  total_fish_second_fisherman = total_fish_first_fisherman + 3 := by sorry

end NUMINAMATH_GPT_second_fisherman_more_fish_l1458_145858


namespace NUMINAMATH_GPT_largest_n_for_two_digit_quotient_l1458_145885

-- Lean statement for the given problem.
theorem largest_n_for_two_digit_quotient (n : ℕ) (h₀ : 0 ≤ n) (h₃ : n ≤ 9) :
  (10 ≤ (n * 100 + 5) / 5 ∧ (n * 100 + 5) / 5 < 100) ↔ n = 4 :=
by sorry

end NUMINAMATH_GPT_largest_n_for_two_digit_quotient_l1458_145885


namespace NUMINAMATH_GPT_pet_store_earnings_l1458_145893

theorem pet_store_earnings :
  let kitten_price := 6
  let puppy_price := 5
  let kittens_sold := 2
  let puppies_sold := 1 
  let total_earnings := kittens_sold * kitten_price + puppies_sold * puppy_price
  total_earnings = 17 :=
by
  sorry

end NUMINAMATH_GPT_pet_store_earnings_l1458_145893


namespace NUMINAMATH_GPT_last_digit_base5_89_l1458_145819

theorem last_digit_base5_89 (n : ℕ) (h : n = 89) : (n % 5) = 4 :=
by 
  sorry

end NUMINAMATH_GPT_last_digit_base5_89_l1458_145819


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1458_145861

-- Definitions for the conditions
def a := 70
def d := 3
def n := 10
def l := 97

-- Sum of the arithmetic series
def S := (n / 2) * (a + l)

-- Final calculation
theorem arithmetic_sequence_sum :
  3 * (70 + 73 + 76 + 79 + 82 + 85 + 88 + 91 + 94 + 97) = 2505 :=
by
  -- Lean will calculate these interactively when proving.
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1458_145861


namespace NUMINAMATH_GPT_min_value_expression_l1458_145809

variable {a b : ℝ}

theorem min_value_expression
  (h1 : a > 0) 
  (h2 : b > 0)
  (h3 : a + b = 4) : 
  (∃ C, (∀ a b, a > 0 → b > 0 → a + b = 4 → (b / a + 4 / b) ≥ C) ∧ 
         (∀ a b, a > 0 → b > 0 → a + b = 4 → (b / a + 4 / b) = C)) ∧ 
         C = 3 :=
  by sorry

end NUMINAMATH_GPT_min_value_expression_l1458_145809


namespace NUMINAMATH_GPT_train_length_l1458_145842

theorem train_length (L : ℝ) (v1 v2 : ℝ) 
  (h1 : v1 = (L + 140) / 15)
  (h2 : v2 = (L + 250) / 20) 
  (h3 : v1 = v2) :
  L = 190 :=
by sorry

end NUMINAMATH_GPT_train_length_l1458_145842


namespace NUMINAMATH_GPT_sum_partition_36_l1458_145857

theorem sum_partition_36 : 
  ∃ (S : Finset ℕ), S.card = 36 ∧ S.sum id = ((Finset.range 72).sum id) / 2 :=
by
  sorry

end NUMINAMATH_GPT_sum_partition_36_l1458_145857


namespace NUMINAMATH_GPT_find_number_l1458_145843

theorem find_number (x : ℝ) : 0.75 * x + 2 = 8 → x = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1458_145843


namespace NUMINAMATH_GPT_original_price_color_tv_l1458_145835

theorem original_price_color_tv (x : ℝ) : 
  1.4 * x * 0.8 - x = 270 → x = 2250 :=
by
  intro h
  simp at h
  sorry

end NUMINAMATH_GPT_original_price_color_tv_l1458_145835


namespace NUMINAMATH_GPT_jori_remaining_water_l1458_145877

-- Having the necessary libraries for arithmetic and fractions.

-- Definitions directly from the conditions in a).
def initial_water_quantity : ℚ := 4
def used_water_quantity : ℚ := 9 / 4 -- Converted 2 1/4 to an improper fraction

-- The statement proving the remaining quantity of water is 1 3/4 gallons.
theorem jori_remaining_water : initial_water_quantity - used_water_quantity = 7 / 4 := by
  sorry

end NUMINAMATH_GPT_jori_remaining_water_l1458_145877


namespace NUMINAMATH_GPT_sum_of_roots_of_P_is_8029_l1458_145838

-- Define the polynomial
noncomputable def P : Polynomial ℚ :=
  (Polynomial.X - 1)^2008 + 
  3 * (Polynomial.X - 2)^2007 + 
  5 * (Polynomial.X - 3)^2006 + 
  -- Continue defining all terms up to:
  2009 * (Polynomial.X - 2008)^2 + 
  2011 * (Polynomial.X - 2009)

-- The proof problem statement
theorem sum_of_roots_of_P_is_8029 :
  (P.roots.sum = 8029) :=
sorry

end NUMINAMATH_GPT_sum_of_roots_of_P_is_8029_l1458_145838


namespace NUMINAMATH_GPT_parallelogram_angle_B_eq_130_l1458_145899

theorem parallelogram_angle_B_eq_130 (A C B D : ℝ) (parallelogram_ABCD : true) 
(angles_sum_A_C : A + C = 100) (A_eq_C : A = C): B = 130 := by
  sorry

end NUMINAMATH_GPT_parallelogram_angle_B_eq_130_l1458_145899


namespace NUMINAMATH_GPT_even_factors_count_l1458_145817

theorem even_factors_count (n : ℕ) (h : n = 2^3 * 3 * 7^2 * 5) : 
  ∃ k, k = 36 ∧ 
       (∀ a b c d : ℕ, 1 ≤ a ∧ a ≤ 3 ∧ 0 ≤ b ∧ b ≤ 1 ∧ 0 ≤ c ∧ c ≤ 2 ∧ 0 ≤ d ∧ d ≤ 1 →
       ∃ m, m = 2^a * 3^b * 7^c * 5^d ∧ 2 ∣ m ∧ m ∣ n) := sorry

end NUMINAMATH_GPT_even_factors_count_l1458_145817


namespace NUMINAMATH_GPT_cos_14_pi_over_3_l1458_145833

theorem cos_14_pi_over_3 : Real.cos (14 * Real.pi / 3) = -1 / 2 :=
by 
  -- Proof is omitted according to the instructions
  sorry

end NUMINAMATH_GPT_cos_14_pi_over_3_l1458_145833


namespace NUMINAMATH_GPT_find_x_l1458_145856

variable (x : ℝ)
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (x, 1)
def u : ℝ × ℝ := (1 + 2 * x, 4)
def v : ℝ × ℝ := (2 - 2 * x, 2)

theorem find_x (h : 2 * (1 + 2 * x) = 4 * (2 - 2 * x)) : x = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_find_x_l1458_145856


namespace NUMINAMATH_GPT_swimming_distance_l1458_145855

theorem swimming_distance
  (t : ℝ) (d_up : ℝ) (d_down : ℝ) (v_man : ℝ) (v_stream : ℝ)
  (h1 : v_man = 5) (h2 : t = 5) (h3 : d_up = 20) 
  (h4 : d_up = (v_man - v_stream) * t) :
  d_down = (v_man + v_stream) * t :=
by
  sorry

end NUMINAMATH_GPT_swimming_distance_l1458_145855


namespace NUMINAMATH_GPT_Pythagorean_triple_l1458_145882

theorem Pythagorean_triple (n : ℕ) (hn : n % 2 = 1) (hn_geq : n ≥ 3) :
  n^2 + ((n^2 - 1) / 2)^2 = ((n^2 + 1) / 2)^2 := by
  sorry

end NUMINAMATH_GPT_Pythagorean_triple_l1458_145882


namespace NUMINAMATH_GPT_negation_of_proposition_l1458_145812

theorem negation_of_proposition (x : ℝ) :
  ¬ (x > 1 → x ^ 2 > x) ↔ (x ≤ 1 → x ^ 2 ≤ x) :=
by 
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l1458_145812


namespace NUMINAMATH_GPT_book_chapters_not_determinable_l1458_145828

variable (pages_initially pages_later pages_total total_pages book_chapters : ℕ)

def problem_statement : Prop :=
  pages_initially = 37 ∧ pages_later = 25 ∧ pages_total = 62 ∧ total_pages = 95 ∧ book_chapters = 0

theorem book_chapters_not_determinable (h: problem_statement pages_initially pages_later pages_total total_pages book_chapters) :
  book_chapters = 0 :=
by
  sorry

end NUMINAMATH_GPT_book_chapters_not_determinable_l1458_145828


namespace NUMINAMATH_GPT_min_value_of_function_l1458_145863

theorem min_value_of_function (x : ℝ) (h : x > 0) : (∃ y : ℝ, y = x^2 + 3 * x + 1 ∧ ∀ z, z = x^2 + 3 * x + 1 → y ≤ z) → y = 5 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_function_l1458_145863


namespace NUMINAMATH_GPT_place_value_ratio_56439_2071_l1458_145894

theorem place_value_ratio_56439_2071 :
  let n := 56439.2071
  let digit_6_place_value := 1000
  let digit_2_place_value := 0.1
  digit_6_place_value / digit_2_place_value = 10000 :=
by
  sorry

end NUMINAMATH_GPT_place_value_ratio_56439_2071_l1458_145894


namespace NUMINAMATH_GPT_sam_initial_dimes_l1458_145878

theorem sam_initial_dimes (given_away : ℕ) (left : ℕ) (initial : ℕ) 
  (h1 : given_away = 7) (h2 : left = 2) (h3 : initial = given_away + left) : 
  initial = 9 := by
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_sam_initial_dimes_l1458_145878


namespace NUMINAMATH_GPT_base12_remainder_l1458_145805

def base12_to_base10 (a : ℕ) (b : ℕ) (c : ℕ) (d : ℕ) : ℕ :=
  a * 12^3 + b * 12^2 + c * 12^1 + d * 12^0

theorem base12_remainder (a b c d : ℕ) 
  (h1531 : base12_to_base10 a b c d = 1 * 12^3 + 5 * 12^2 + 3 * 12^1 + 1 * 12^0):
  (base12_to_base10 a b c d) % 8 = 5 :=
by
  unfold base12_to_base10 at h1531
  sorry

end NUMINAMATH_GPT_base12_remainder_l1458_145805


namespace NUMINAMATH_GPT_sector_area_l1458_145847

-- Define radius and central angle as conditions
def radius : ℝ := 1
def central_angle : ℝ := 2

-- Define the theorem to prove that the area of the sector is 1 cm² given the conditions
theorem sector_area : (1 / 2) * radius * central_angle = 1 := 
by 
  -- sorry is used to skip the actual proof
  sorry

end NUMINAMATH_GPT_sector_area_l1458_145847


namespace NUMINAMATH_GPT_theo_cookie_price_l1458_145800

theorem theo_cookie_price :
  (∃ (dough_amount total_earnings per_cookie_earnings_carla per_cookie_earnings_theo : ℕ) 
     (cookies_carla cookies_theo : ℝ), 
  dough_amount = 120 ∧ 
  cookies_carla = 20 ∧ 
  per_cookie_earnings_carla = 50 ∧ 
  cookies_theo = 15 ∧ 
  total_earnings = cookies_carla * per_cookie_earnings_carla ∧ 
  per_cookie_earnings_theo = total_earnings / cookies_theo ∧ 
  per_cookie_earnings_theo = 67) :=
sorry

end NUMINAMATH_GPT_theo_cookie_price_l1458_145800


namespace NUMINAMATH_GPT_distance_traveled_eq_2400_l1458_145867

-- Definitions of the conditions
def circumference_front : ℕ := 30
def circumference_back : ℕ := 32
def revolutions_difference : ℕ := 5

-- Define the number of revolutions made by the back wheel
def revs_back (R : ℕ) := R

-- Define the number of revolutions made by the front wheel
def revs_front (R : ℕ) := R + revolutions_difference

-- Define the distance traveled by the back and front wheels
def distance_back (R : ℕ) : ℕ := revs_back R * circumference_back
def distance_front (R : ℕ) : ℕ := revs_front R * circumference_front

-- State the theorem without a proof (using sorry)
theorem distance_traveled_eq_2400 :
  ∃ R : ℕ, distance_back R = 2400 ∧ distance_back R = distance_front R :=
by {
  sorry
}

end NUMINAMATH_GPT_distance_traveled_eq_2400_l1458_145867


namespace NUMINAMATH_GPT_six_times_more_coats_l1458_145806

/-- The number of lab coats is 6 times the number of uniforms. --/
def coats_per_uniforms (c u : ℕ) : Prop := c = 6 * u

/-- There are 12 uniforms. --/
def uniforms : ℕ := 12

/-- Each lab tech gets 14 coats and uniforms in total. --/
def total_per_tech : ℕ := 14

/-- Show that the number of lab coats is 6 times the number of uniforms. --/
theorem six_times_more_coats (c u : ℕ) (h1 : coats_per_uniforms c u) (h2 : u = 12) :
  c / u = 6 :=
by
  sorry

end NUMINAMATH_GPT_six_times_more_coats_l1458_145806


namespace NUMINAMATH_GPT_school_girls_more_than_boys_l1458_145873

def num_initial_girls := 632
def num_initial_boys := 410
def num_new_girls := 465
def num_total_girls := num_initial_girls + num_new_girls
def num_difference_girls_boys := num_total_girls - num_initial_boys

theorem school_girls_more_than_boys :
  num_difference_girls_boys = 687 :=
by
  sorry

end NUMINAMATH_GPT_school_girls_more_than_boys_l1458_145873


namespace NUMINAMATH_GPT_max_xy_l1458_145830

theorem max_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 18) : xy ≤ 81 :=
by sorry

end NUMINAMATH_GPT_max_xy_l1458_145830


namespace NUMINAMATH_GPT_find_a20_l1458_145836

variable (a : ℕ → ℝ)
variable (d : ℝ)

axiom a_arithmetic : ∀ n, a (n + 1) = a 1 + n * d
axiom a1_a3_a5_eq_105 : a 1 + a 3 + a 5 = 105
axiom a2_a4_a6_eq_99 : a 2 + a 4 + a 6 = 99

theorem find_a20 : a 20 = 1 :=
by sorry

end NUMINAMATH_GPT_find_a20_l1458_145836


namespace NUMINAMATH_GPT_least_positive_integer_solution_l1458_145846

theorem least_positive_integer_solution :
  ∃ b : ℕ, b ≡ 2 [MOD 3] ∧ b ≡ 3 [MOD 4] ∧ b ≡ 4 [MOD 5] ∧ b ≡ 8 [MOD 9] ∧ b = 179 :=
by
  sorry

end NUMINAMATH_GPT_least_positive_integer_solution_l1458_145846


namespace NUMINAMATH_GPT_compare_polynomials_l1458_145822

noncomputable def f (x : ℝ) : ℝ := 2*x^2 + 5*x + 3
noncomputable def g (x : ℝ) : ℝ := x^2 + 4*x + 2

theorem compare_polynomials (x : ℝ) : f x > g x :=
by sorry

end NUMINAMATH_GPT_compare_polynomials_l1458_145822


namespace NUMINAMATH_GPT_average_income_A_B_l1458_145853

def monthly_incomes (A B C : ℝ) : Prop :=
  (A = 4000) ∧
  ((B + C) / 2 = 6250) ∧
  ((A + C) / 2 = 5200)

theorem average_income_A_B (A B C X : ℝ) (h : monthly_incomes A B C) : X = 5050 :=
by
  have hA : A = 4000 := h.1
  have hBC : (B + C) / 2 = 6250 := h.2.1
  have hAC : (A + C) / 2 = 5200 := h.2.2
  sorry

end NUMINAMATH_GPT_average_income_A_B_l1458_145853


namespace NUMINAMATH_GPT_not_converge_to_a_l1458_145870

theorem not_converge_to_a (x : ℕ → ℝ) (a : ℝ) :
  (∀ ε > 0, ∀ k : ℕ, ∃ n : ℕ, n > k ∧ |x n - a| ≥ ε) →
  ¬ (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |x n - a| < ε) :=
by sorry

end NUMINAMATH_GPT_not_converge_to_a_l1458_145870


namespace NUMINAMATH_GPT_victor_earnings_l1458_145862

def hourly_wage := 6 -- dollars per hour
def hours_monday := 5 -- hours
def hours_tuesday := 5 -- hours

theorem victor_earnings : (hourly_wage * (hours_monday + hours_tuesday)) = 60 :=
by
  sorry

end NUMINAMATH_GPT_victor_earnings_l1458_145862


namespace NUMINAMATH_GPT_inequality_pos_reals_l1458_145881

theorem inequality_pos_reals (x y z : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 0 < z) : 
  (x^2 + 2) * (y^2 + 2) * (z^2 + 2) ≥ 9 * (x * y + y * z + z * x) :=
by
  sorry

end NUMINAMATH_GPT_inequality_pos_reals_l1458_145881


namespace NUMINAMATH_GPT_bankers_gain_correct_l1458_145821

def PW : ℝ := 600
def R : ℝ := 0.10
def n : ℕ := 2

def A : ℝ := PW * (1 + R)^n
def BG : ℝ := A - PW

theorem bankers_gain_correct : BG = 126 :=
by
  sorry

end NUMINAMATH_GPT_bankers_gain_correct_l1458_145821


namespace NUMINAMATH_GPT_value_of_a_l1458_145892

theorem value_of_a (m : ℝ) (f : ℝ → ℝ) (h : f = fun x => (1/3)^x + m - 1/3) 
  (h_m : ∀ x, f x ≥ 0 ↔ m ≥ -2/3) : m ≥ -2/3 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l1458_145892


namespace NUMINAMATH_GPT_fraction_denominator_l1458_145829

theorem fraction_denominator (x y Z : ℚ) (h : x / y = 7 / 3) (h2 : (x + y) / Z = 2.5) :
    Z = (4 * y) / 3 :=
by sorry

end NUMINAMATH_GPT_fraction_denominator_l1458_145829


namespace NUMINAMATH_GPT_prime_sum_of_primes_unique_l1458_145824

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_sum_of_primes_unique (p q : ℕ) (hp : is_prime p) (hq : is_prime q) (h_sum_prime : is_prime (p^q + q^p)) :
  p = 2 ∧ q = 3 :=
sorry

end NUMINAMATH_GPT_prime_sum_of_primes_unique_l1458_145824


namespace NUMINAMATH_GPT_trajectory_of_point_l1458_145852

theorem trajectory_of_point 
  (P : ℝ × ℝ) 
  (h1 : abs (P.1 - 4) + P.2^2 - 1 = abs (P.1 + 5)) : 
  P.2^2 = 16 * P.1 := 
sorry

end NUMINAMATH_GPT_trajectory_of_point_l1458_145852


namespace NUMINAMATH_GPT_range_of_a_l1458_145815

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, ¬ (ax^2 - ax + 1 = 0)) ↔ (0 ≤ a ∧ a < 4) :=
sorry

end NUMINAMATH_GPT_range_of_a_l1458_145815


namespace NUMINAMATH_GPT_find_initial_amount_l1458_145865

-- Definitions for conditions
def final_amount : ℝ := 5565
def rate_year1 : ℝ := 0.05
def rate_year2 : ℝ := 0.06

-- Theorem statement to prove the initial amount
theorem find_initial_amount (P : ℝ) 
  (H : final_amount = (P * (1 + rate_year1)) * (1 + rate_year2)) :
  P = 5000 := 
sorry

end NUMINAMATH_GPT_find_initial_amount_l1458_145865


namespace NUMINAMATH_GPT_truncated_pyramid_volume_ratio_l1458_145823

/-
Statement: Given a truncated triangular pyramid with a plane drawn through a side of the upper base parallel to the opposite lateral edge,
and the corresponding sides of the bases in the ratio 1:2, prove that the volume of the truncated pyramid is divided in the ratio 3:4.
-/

theorem truncated_pyramid_volume_ratio (S1 S2 h : ℝ) 
  (h_ratio : S1 = 4 * S2) :
  (h * S2) / ((7 * h * S2) / 3 - h * S2) = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_truncated_pyramid_volume_ratio_l1458_145823


namespace NUMINAMATH_GPT_initial_integers_is_three_l1458_145875

def num_initial_integers (n m : Int) : Prop :=
  3 * n + m = 17 ∧ 2 * m + n = 23

theorem initial_integers_is_three {n m : Int} (h : num_initial_integers n m) : n = 3 :=
by
  sorry

end NUMINAMATH_GPT_initial_integers_is_three_l1458_145875


namespace NUMINAMATH_GPT_intersecting_lines_l1458_145891

theorem intersecting_lines {c d : ℝ} 
  (h₁ : 12 = 2 * 4 + c) 
  (h₂ : 12 = -4 + d) : 
  c + d = 20 := 
sorry

end NUMINAMATH_GPT_intersecting_lines_l1458_145891


namespace NUMINAMATH_GPT_cricket_target_runs_l1458_145897

theorem cricket_target_runs 
  (run_rate1 : ℝ) (run_rate2 : ℝ) (overs : ℕ)
  (h1 : run_rate1 = 5.4) (h2 : run_rate2 = 10.6) (h3 : overs = 25) :
  (run_rate1 * overs + run_rate2 * overs = 400) :=
by sorry

end NUMINAMATH_GPT_cricket_target_runs_l1458_145897


namespace NUMINAMATH_GPT_percentage_paid_to_x_l1458_145813

theorem percentage_paid_to_x (X Y : ℕ) (h₁ : Y = 350) (h₂ : X + Y = 770) :
  (X / Y) * 100 = 120 :=
by
  sorry

end NUMINAMATH_GPT_percentage_paid_to_x_l1458_145813


namespace NUMINAMATH_GPT_dani_pants_after_5_years_l1458_145810

theorem dani_pants_after_5_years :
  ∀ (pairs_per_year : ℕ) (pants_per_pair : ℕ) (initial_pants : ℕ) (years : ℕ),
  pairs_per_year = 4 →
  pants_per_pair = 2 →
  initial_pants = 50 →
  years = 5 →
  initial_pants + years * (pairs_per_year * pants_per_pair) = 90 :=
by sorry

end NUMINAMATH_GPT_dani_pants_after_5_years_l1458_145810


namespace NUMINAMATH_GPT_undefined_sum_slope_y_intercept_of_vertical_line_l1458_145808

theorem undefined_sum_slope_y_intercept_of_vertical_line :
  ∀ (C D : ℝ × ℝ), C.1 = 8 → D.1 = 8 → C.2 ≠ D.2 →
  ∃ (m b : ℝ), false :=
by
  intros
  sorry

end NUMINAMATH_GPT_undefined_sum_slope_y_intercept_of_vertical_line_l1458_145808


namespace NUMINAMATH_GPT_minimum_value_of_expression_l1458_145804

theorem minimum_value_of_expression (x : ℝ) (hx : x ≠ 0) : 
  (x^2 + 1 / x^2) ≥ 2 ∧ (x^2 + 1 / x^2 = 2 ↔ x = 1 ∨ x = -1) := 
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l1458_145804


namespace NUMINAMATH_GPT_find_integer_pairs_l1458_145848

theorem find_integer_pairs (x y: ℤ) :
  x^2 - y^4 = 2009 → (x = 45 ∧ (y = 2 ∨ y = -2)) ∨ (x = -45 ∧ (y = 2 ∨ y = -2)) :=
by
  sorry

end NUMINAMATH_GPT_find_integer_pairs_l1458_145848


namespace NUMINAMATH_GPT_power_sums_l1458_145827

-- Definitions as per the given conditions
variables (m n a b : ℕ)
variables (hm : 0 < m) (hn : 0 < n)
variables (ha : 2^m = a) (hb : 2^n = b)

-- The theorem statement
theorem power_sums (hmn : 0 < m + n) : 2^(m + n) = a * b :=
by
  sorry

end NUMINAMATH_GPT_power_sums_l1458_145827


namespace NUMINAMATH_GPT_train_speed_correct_l1458_145807

noncomputable def train_speed : ℝ :=
  let distance := 120 -- meters
  let time := 5.999520038396929 -- seconds
  let speed_m_s := distance / time -- meters per second
  speed_m_s * 3.6 -- converting to km/hr

theorem train_speed_correct : train_speed = 72.004800384 := by
  simp [train_speed]
  sorry

end NUMINAMATH_GPT_train_speed_correct_l1458_145807


namespace NUMINAMATH_GPT_min_value_inequality_l1458_145866

theorem min_value_inequality (θ φ : ℝ) : 
  (3 * Real.cos θ + 4 * Real.sin φ - 10)^2 + (3 * Real.sin θ + 4 * Real.cos φ - 20)^2 ≥ 549 - 140 * Real.sqrt 5 := 
by
  sorry

end NUMINAMATH_GPT_min_value_inequality_l1458_145866


namespace NUMINAMATH_GPT_interest_rate_per_annum_l1458_145876

theorem interest_rate_per_annum :
  ∃ (r : ℝ), 338 = 312.50 * (1 + r) ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_interest_rate_per_annum_l1458_145876


namespace NUMINAMATH_GPT_hyperbola_a_value_l1458_145872

theorem hyperbola_a_value (a : ℝ) :
  (∀ x y : ℝ, (x^2 / (a + 3) - y^2 / 3 = 1)) ∧ 
  (∀ e : ℝ, e = 2) → 
  a = -2 :=
by sorry

end NUMINAMATH_GPT_hyperbola_a_value_l1458_145872


namespace NUMINAMATH_GPT_soap_box_height_l1458_145895

theorem soap_box_height
  (carton_length carton_width carton_height : ℕ)
  (soap_length soap_width h : ℕ)
  (max_soap_boxes : ℕ)
  (h_carton_dim : carton_length = 30)
  (h_carton_width : carton_width = 42)
  (h_carton_height : carton_height = 60)
  (h_soap_length : soap_length = 7)
  (h_soap_width : soap_width = 6)
  (h_max_soap_boxes : max_soap_boxes = 360) :
  h = 1 :=
by
  sorry

end NUMINAMATH_GPT_soap_box_height_l1458_145895


namespace NUMINAMATH_GPT_jane_number_of_muffins_l1458_145841

theorem jane_number_of_muffins 
    (m b c : ℕ) 
    (h1 : m + b + c = 6) 
    (h2 : b = 2) 
    (h3 : (50 * m + 75 * b + 65 * c) % 100 = 0) : 
    m = 4 := 
sorry

end NUMINAMATH_GPT_jane_number_of_muffins_l1458_145841


namespace NUMINAMATH_GPT_smallest_angle_in_icosagon_l1458_145814

-- Definitions for the conditions:
def sum_of_interior_angles (n : ℕ) : ℕ := (n - 2) * 180
def average_angle (n : ℕ) (sum_of_angles : ℕ) : ℕ := sum_of_angles / n
def is_convex (angle : ℕ) : Prop := angle < 180
def arithmetic_sequence_smallest_angle (n : ℕ) (average : ℕ) (d : ℕ) : ℕ := average - 9 * d

theorem smallest_angle_in_icosagon
  (d : ℕ)
  (d_condition : d = 1)
  (convex_condition : ∀ i, is_convex (162 + (i - 1) * 2 * d))
  : arithmetic_sequence_smallest_angle 20 162 d = 153 := by
  sorry

end NUMINAMATH_GPT_smallest_angle_in_icosagon_l1458_145814


namespace NUMINAMATH_GPT_slope_of_asymptotes_l1458_145879

noncomputable def hyperbola_asymptote_slope (x y : ℝ) : Prop :=
  (x^2 / 144 - y^2 / 81 = 1)

theorem slope_of_asymptotes (x y : ℝ) (h : hyperbola_asymptote_slope x y) :
  ∃ m : ℝ, m = 3 / 4 ∨ m = -3 / 4 :=
sorry

end NUMINAMATH_GPT_slope_of_asymptotes_l1458_145879


namespace NUMINAMATH_GPT_rectangle_area_eq_l1458_145890

theorem rectangle_area_eq (d : ℝ) (w : ℝ) (h1 : w = d / (2 * (5 : ℝ) ^ (1/2))) (h2 : 3 * w = (3 * d) / (2 * (5 : ℝ) ^ (1/2))) : 
  (3 * w^2) = (3 / 10) * d^2 := 
by sorry

end NUMINAMATH_GPT_rectangle_area_eq_l1458_145890


namespace NUMINAMATH_GPT_kids_prefer_peas_l1458_145859

variable (total_kids children_prefer_carrots children_prefer_corn : ℕ)

theorem kids_prefer_peas (H1 : children_prefer_carrots = 9)
(H2 : children_prefer_corn = 5)
(H3 : children_prefer_corn * 4 = total_kids) :
total_kids - (children_prefer_carrots + children_prefer_corn) = 6 := by
sorry

end NUMINAMATH_GPT_kids_prefer_peas_l1458_145859


namespace NUMINAMATH_GPT_fraction_sum_of_lcm_and_gcd_l1458_145831

theorem fraction_sum_of_lcm_and_gcd 
  (m n : ℕ) 
  (h_gcd : Nat.gcd m n = 6) 
  (h_lcm : Nat.lcm m n = 210) 
  (h_sum : m + n = 72) :
  1 / (m : ℚ) + 1 / (n : ℚ) = 12 / 210 := 
by
sorry

end NUMINAMATH_GPT_fraction_sum_of_lcm_and_gcd_l1458_145831


namespace NUMINAMATH_GPT_remainder_when_two_pow_thirty_three_div_nine_l1458_145818

-- Define the base and the exponent
def base : ℕ := 2
def exp : ℕ := 33
def modulus : ℕ := 9

-- The main statement to prove
theorem remainder_when_two_pow_thirty_three_div_nine :
  (base ^ exp) % modulus = 8 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_two_pow_thirty_three_div_nine_l1458_145818


namespace NUMINAMATH_GPT_ways_to_place_7_balls_in_3_boxes_l1458_145896

theorem ways_to_place_7_balls_in_3_boxes : ∃ n : ℕ, n = 8 ∧ (∀ x y z : ℕ, x + y + z = 7 → x ≥ y → y ≥ z → z ≥ 0) := 
by
  sorry

end NUMINAMATH_GPT_ways_to_place_7_balls_in_3_boxes_l1458_145896


namespace NUMINAMATH_GPT_carbon_atoms_in_compound_l1458_145889

theorem carbon_atoms_in_compound 
    (molecular_weight : ℕ := 65)
    (carbon_weight : ℕ := 12)
    (hydrogen_weight : ℕ := 1)
    (oxygen_weight : ℕ := 16)
    (hydrogen_atoms : ℕ := 1)
    (oxygen_atoms : ℕ := 1) :
    ∃ (carbon_atoms : ℕ), molecular_weight = (carbon_atoms * carbon_weight) + (hydrogen_atoms * hydrogen_weight) + (oxygen_atoms * oxygen_weight) ∧ carbon_atoms = 4 :=
by
  sorry

end NUMINAMATH_GPT_carbon_atoms_in_compound_l1458_145889


namespace NUMINAMATH_GPT_dog_catches_fox_at_120m_l1458_145834

theorem dog_catches_fox_at_120m :
  let initial_distance := 30
  let dog_leap := 2
  let fox_leap := 1
  let dog_leap_frequency := 2
  let fox_leap_frequency := 3
  let dog_distance_per_time_unit := dog_leap * dog_leap_frequency
  let fox_distance_per_time_unit := fox_leap * fox_leap_frequency
  let relative_closure_rate := dog_distance_per_time_unit - fox_distance_per_time_unit
  let time_units_to_catch := initial_distance / relative_closure_rate
  let total_dog_distance := time_units_to_catch * dog_distance_per_time_unit
  total_dog_distance = 120 := sorry

end NUMINAMATH_GPT_dog_catches_fox_at_120m_l1458_145834


namespace NUMINAMATH_GPT_find_value_of_expression_l1458_145837

theorem find_value_of_expression (x : ℝ) (h : 5 * x^2 + 4 = 3 * x + 9) : (10 * x - 3)^2 = 109 := 
sorry

end NUMINAMATH_GPT_find_value_of_expression_l1458_145837


namespace NUMINAMATH_GPT_sequence_formula_l1458_145832

theorem sequence_formula (a : ℕ → ℚ) (h₁ : a 1 = 0)
  (h : ∀ n, a (n + 1) = 1 / (2 - a n)) :
  ∀ n, a n = (n - 1) / n :=
sorry

end NUMINAMATH_GPT_sequence_formula_l1458_145832


namespace NUMINAMATH_GPT_part1_part2_l1458_145868

open Real

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := log x - a * x - 1
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := exp (a * x) * f x a + x

theorem part1 (a : ℝ) : 
  (a ≤ 0 → ∀ x, ∀ y, f x a ≤ y) ∧ (a > 0 → ∃ x, ∀ y, f x a ≤ y ∧ y = log (1 / a) - 2) :=
sorry

theorem part2 (a m : ℝ) (h_a : a > 0) (x1 x2 : ℝ) (h_x1 : 0 < x1) (h_x2 : x1 < x2) 
  (h_g1 : g x1 a = 0) (h_g2 : g x2 a = 0) : x1 * (x2 ^ 2) > exp m → m ≤ 3 :=
sorry

end NUMINAMATH_GPT_part1_part2_l1458_145868


namespace NUMINAMATH_GPT_find_a_l1458_145884

noncomputable def star (a b : ℝ) := a * (a + b) + b

theorem find_a (a : ℝ) (h : star a 2.5 = 28.5) : a = 4 ∨ a = -13/2 := 
sorry

end NUMINAMATH_GPT_find_a_l1458_145884


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l1458_145864

theorem simplify_and_evaluate_expression :
  ∀ (x y : ℝ), 
  x = -1 / 3 → y = -2 → 
  (3 * x + 2 * y) * (3 * x - 2 * y) - 5 * x * (x - y) - (2 * x - y)^2 = -14 :=
by
  intros x y hx hy
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l1458_145864


namespace NUMINAMATH_GPT_sampled_individual_l1458_145874

theorem sampled_individual {population_size sample_size : ℕ} (population_size_cond : population_size = 1000)
  (sample_size_cond : sample_size = 20) (sampled_number : ℕ) (sampled_number_cond : sampled_number = 15) :
  (∃ n : ℕ, sampled_number + n * (population_size / sample_size) = 65) :=
by 
  sorry

end NUMINAMATH_GPT_sampled_individual_l1458_145874


namespace NUMINAMATH_GPT_polygon_to_triangle_l1458_145886

theorem polygon_to_triangle {n : ℕ} (h : n > 4) :
  ∃ (a b c : ℕ), (a + b > c ∧ a + c > b ∧ b + c > a) :=
sorry

end NUMINAMATH_GPT_polygon_to_triangle_l1458_145886


namespace NUMINAMATH_GPT_part1_part2_part3_l1458_145811

def folklore {a b m n : ℤ} (h1 : a + b * Real.sqrt 3 = (m + n * Real.sqrt 3) ^ 2) : Prop :=
  a = m ^ 2 + 3 * n ^ 2 ∧ b = 2 * m * n

theorem part1 (a b m n : ℤ) (h : a + b * Real.sqrt 3 = (m + n * Real.sqrt 3) ^ 2) : a = m ^ 2 + 3 * n ^ 2 ∧ b = 2 * m * n :=
by sorry

theorem part2 : 13 + 4 * Real.sqrt 3 = (1 + 2 * Real.sqrt 3) ^ 2 :=
by sorry

theorem part3 (a m n : ℤ) (h1 : 4 = 2 * m * n) (h2 : a + 4 * Real.sqrt 3 = (m + n * Real.sqrt 3) ^ 2) : a = 7 ∨ a = 13 :=
by sorry

end NUMINAMATH_GPT_part1_part2_part3_l1458_145811


namespace NUMINAMATH_GPT_least_positive_integer_a_l1458_145851

theorem least_positive_integer_a (a : ℕ) (n : ℕ) 
  (h1 : 2001 = 3 * 23 * 29)
  (h2 : 55 % 3 = 1)
  (h3 : 32 % 3 = -1)
  (h4 : 55 % 23 = 32 % 23)
  (h5 : 55 % 29 = -32 % 29)
  (h6 : n % 2 = 1)
  : a = 436 := 
sorry

end NUMINAMATH_GPT_least_positive_integer_a_l1458_145851


namespace NUMINAMATH_GPT_option_B_not_well_defined_l1458_145826

-- Definitions based on given conditions 
def is_well_defined_set (description : String) : Prop :=
  match description with
  | "All positive numbers" => True
  | "All elderly people" => False
  | "All real numbers that are not equal to 0" => True
  | "The four great inventions of ancient China" => True
  | _ => False

-- Theorem stating option B "All elderly people" is not a well-defined set
theorem option_B_not_well_defined : ¬ is_well_defined_set "All elderly people" :=
  by sorry

end NUMINAMATH_GPT_option_B_not_well_defined_l1458_145826


namespace NUMINAMATH_GPT_units_digit_product_of_four_consecutive_integers_l1458_145844

theorem units_digit_product_of_four_consecutive_integers (n : ℕ) (h : n % 2 = 1) : (n * (n + 1) * (n + 2) * (n + 3)) % 10 = 0 := 
by 
  sorry

end NUMINAMATH_GPT_units_digit_product_of_four_consecutive_integers_l1458_145844


namespace NUMINAMATH_GPT_jaguars_total_games_l1458_145801

-- Defining constants for initial conditions
def initial_win_rate : ℚ := 0.55
def additional_wins : ℕ := 8
def additional_losses : ℕ := 2
def final_win_rate : ℚ := 0.6

-- Defining the main problem statement
theorem jaguars_total_games : 
  ∃ y x : ℕ, (x = initial_win_rate * y) ∧ (x + additional_wins = final_win_rate * (y + (additional_wins + additional_losses))) ∧ (y + (additional_wins + additional_losses) = 50) :=
sorry

end NUMINAMATH_GPT_jaguars_total_games_l1458_145801


namespace NUMINAMATH_GPT_max_value_of_function_l1458_145825

theorem max_value_of_function (x : ℝ) (h : 0 < x ∧ x < 1.5) : 
  ∃ m, ∀ y, y = 4 * x * (3 - 2 * x) → m = 9 / 2 :=
sorry

end NUMINAMATH_GPT_max_value_of_function_l1458_145825


namespace NUMINAMATH_GPT_circumcircle_eq_l1458_145898

noncomputable def A : (ℝ × ℝ) := (0, 0)
noncomputable def B : (ℝ × ℝ) := (4, 0)
noncomputable def C : (ℝ × ℝ) := (0, 6)

theorem circumcircle_eq :
  ∃ h k r, h = 2 ∧ k = 3 ∧ r = 13 ∧ (∀ x y, ((x - h)^2 + (y - k)^2 = r) ↔ (x - 2)^2 + (y - 3)^2 = 13) := sorry

end NUMINAMATH_GPT_circumcircle_eq_l1458_145898


namespace NUMINAMATH_GPT_tony_comics_average_l1458_145854

theorem tony_comics_average :
  let a1 := 10
  let d := 6
  let n := 8
  let a_n (n : ℕ) := a1 + (n - 1) * d
  let S_n (n : ℕ) := n / 2 * (a1 + a_n n)
  (S_n n) / n = 31 := by
  sorry

end NUMINAMATH_GPT_tony_comics_average_l1458_145854


namespace NUMINAMATH_GPT_find_a_and_other_root_l1458_145887

theorem find_a_and_other_root (a : ℝ) (h : (2 : ℝ) ^ 2 - 3 * (2 : ℝ) + a = 0) :
  a = 2 ∧ ∃ x : ℝ, x ^ 2 - 3 * x + a = 0 ∧ x ≠ 2 ∧ x = 1 := 
by
  sorry

end NUMINAMATH_GPT_find_a_and_other_root_l1458_145887


namespace NUMINAMATH_GPT_circle_radius_l1458_145883

-- Parameters of the problem
variables (k : ℝ) (r : ℝ)
-- Conditions
axiom cond_k_positive : k > 8
axiom tangency_y_8 : r = k - 8
axiom tangency_y_x : r = k / (Real.sqrt 2)

-- Statement to prove
theorem circle_radius (k : ℝ) (hk : k > 8) (r : ℝ) (hr1 : r = k - 8) (hr2 : r = k / (Real.sqrt 2)) : r = 8 * Real.sqrt 2 + 8 :=
sorry

end NUMINAMATH_GPT_circle_radius_l1458_145883


namespace NUMINAMATH_GPT_evaluate_expression_l1458_145840

theorem evaluate_expression :
  -5^2 + 2 * (-3)^2 - (-8) / (-1 + 1/3) = -13 :=
by 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1458_145840


namespace NUMINAMATH_GPT_prime_divisor_of_form_l1458_145802

theorem prime_divisor_of_form (a p : ℕ) (hp1 : a > 0) (hp2 : Prime p) (hp3 : p ∣ (a^3 - 3 * a + 1)) (hp4 : p ≠ 3) :
  ∃ k : ℤ, p = 9 * k + 1 ∨ p = 9 * k - 1 :=
by
  sorry

end NUMINAMATH_GPT_prime_divisor_of_form_l1458_145802


namespace NUMINAMATH_GPT_sum_geom_seq_nine_l1458_145871

noncomputable def geom_seq (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem sum_geom_seq_nine {a : ℕ → ℝ} {q : ℝ} (h_geom : geom_seq a q)
  (h1 : a 1 * (1 + q + q^2) = 30) 
  (h2 : a 4 * (1 + q + q^2) = 120) :
  a 7 + a 8 + a 9 = 480 :=
  sorry

end NUMINAMATH_GPT_sum_geom_seq_nine_l1458_145871


namespace NUMINAMATH_GPT_find_m_l1458_145860

def U : Set Nat := {1, 2, 3}
def A (m : Nat) : Set Nat := {1, m}
def complement (s t : Set Nat) : Set Nat := {x | x ∈ s ∧ x ∉ t}

theorem find_m (m : Nat) (h1 : complement U (A m) = {2}) : m = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l1458_145860


namespace NUMINAMATH_GPT_mabel_marble_ratio_l1458_145803

variable (A K M : ℕ)

-- Conditions
def condition1 : Prop := A + 12 = 2 * K
def condition2 : Prop := M = 85
def condition3 : Prop := M = A + 63

-- The main statement to prove
theorem mabel_marble_ratio (h1 : condition1 A K) (h2 : condition2 M) (h3 : condition3 A M) : M / K = 5 :=
by
  sorry

end NUMINAMATH_GPT_mabel_marble_ratio_l1458_145803


namespace NUMINAMATH_GPT_difference_of_integers_l1458_145820

theorem difference_of_integers : ∃ (x y : ℕ), x + y = 20 ∧ x * y = 96 ∧ (x - y = 4 ∨ y - x = 4) :=
by
  sorry

end NUMINAMATH_GPT_difference_of_integers_l1458_145820


namespace NUMINAMATH_GPT_solve_for_z_l1458_145849

theorem solve_for_z (i z : ℂ) (h0 : i^2 = -1) (h1 : i / z = 1 + i) : z = (1 + i) / 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_z_l1458_145849


namespace NUMINAMATH_GPT_expand_expression_l1458_145850

theorem expand_expression (x : ℝ) : 12 * (3 * x - 4) = 36 * x - 48 := by
  sorry

end NUMINAMATH_GPT_expand_expression_l1458_145850


namespace NUMINAMATH_GPT_smallest_N_triangle_ineq_l1458_145869

theorem smallest_N_triangle_ineq (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c)
  (h₄ : c < a + b) : (a^2 + b^2 + a * b) / c^2 < 1 := 
sorry

end NUMINAMATH_GPT_smallest_N_triangle_ineq_l1458_145869


namespace NUMINAMATH_GPT_points_opposite_sides_line_l1458_145888

theorem points_opposite_sides_line (m : ℝ) :
  (3 * 3 - 2 * 1 + m) * (3 * (-4) - 2 * 6 + m) < 0 ↔ -7 < m ∧ m < 24 :=
by
  sorry

end NUMINAMATH_GPT_points_opposite_sides_line_l1458_145888


namespace NUMINAMATH_GPT_number_of_solutions_l1458_145880

theorem number_of_solutions : ∃ n : ℕ, 1 < n ∧ 
  (∃ a b : ℕ, gcd a b = 1 ∧
  (∃ x y : ℕ, x^(a*n) + y^(b*n) = 2^2010)) ∧
  (∃ count : ℕ, count = 54) :=
sorry

end NUMINAMATH_GPT_number_of_solutions_l1458_145880


namespace NUMINAMATH_GPT_find_factor_l1458_145816

theorem find_factor (n f : ℤ) (h₁ : n = 124) (h₂ : n * f - 138 = 110) : f = 2 := by
  sorry

end NUMINAMATH_GPT_find_factor_l1458_145816


namespace NUMINAMATH_GPT_a5_value_l1458_145839

variable (a : ℕ → ℤ) (S : ℕ → ℤ)

-- Assume the sequence is arithmetic
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
axiom sum_S6 : S 6 = 12
axiom term_a2 : a 2 = 5
axiom sum_formula (n : ℕ) : S n = n * (a 1 + a n) / 2

-- Prove a5 is -1
theorem a5_value (h_arith : arithmetic_sequence a)
  (h_S6 : S 6 = 12) (h_a2 : a 2 = 5) (h_sum_formula : ∀ n, S n = n * (a 1 + a n) / 2) :
  a 5 = -1 :=
sorry

end NUMINAMATH_GPT_a5_value_l1458_145839
