import Mathlib

namespace NUMINAMATH_GPT_value_of_M_l2040_204065

theorem value_of_M (M : ℝ) (H : 0.25 * M = 0.55 * 1500) : M = 3300 := 
by
  sorry

end NUMINAMATH_GPT_value_of_M_l2040_204065


namespace NUMINAMATH_GPT_cone_base_diameter_l2040_204016

theorem cone_base_diameter (l r : ℝ) 
  (h1 : (1/2) * π * l^2 + π * r^2 = 3 * π) 
  (h2 : π * l = 2 * π * r) : 2 * r = 2 :=
by
  sorry

end NUMINAMATH_GPT_cone_base_diameter_l2040_204016


namespace NUMINAMATH_GPT_train_length_is_199_95_l2040_204098

noncomputable def convert_speed_to_m_s (speed_kmh : ℝ) : ℝ :=
  (speed_kmh * 1000) / 3600

noncomputable def length_of_train (bridge_length : ℝ) (time_seconds : ℝ) (speed_kmh : ℝ) : ℝ :=
  let speed_ms := convert_speed_to_m_s speed_kmh
  speed_ms * time_seconds - bridge_length

theorem train_length_is_199_95 :
  length_of_train 300 45 40 = 199.95 := by
  sorry

end NUMINAMATH_GPT_train_length_is_199_95_l2040_204098


namespace NUMINAMATH_GPT_find_e_m_l2040_204069

variable {R : Type} [Field R]

def matrix_B (e : R) : Matrix (Fin 2) (Fin 2) R :=
  !![3, 4; 6, e]

theorem find_e_m (e m : R) (hB_inv : (matrix_B e)⁻¹ = m • (matrix_B e)) :
  e = -3 ∧ m = (1 / 11) := by
  sorry

end NUMINAMATH_GPT_find_e_m_l2040_204069


namespace NUMINAMATH_GPT_expected_non_empty_urns_correct_l2040_204059

open ProbabilityTheory

noncomputable def expected_non_empty_urns (n k : ℕ) : ℝ :=
  n * (1 - (1 - 1 / n) ^ k)

theorem expected_non_empty_urns_correct (n k : ℕ) : expected_non_empty_urns n k = n * (1 - ((n - 1) / n) ^ k) :=
by 
  sorry

end NUMINAMATH_GPT_expected_non_empty_urns_correct_l2040_204059


namespace NUMINAMATH_GPT_product_of_two_numbers_l2040_204074

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 24) (h2 : x^2 + y^2 = 404) : x * y = 86 :=
sorry

end NUMINAMATH_GPT_product_of_two_numbers_l2040_204074


namespace NUMINAMATH_GPT_work_days_for_A_l2040_204067

/-- If A is thrice as fast as B and together they can do a work in 15 days, A alone can do the work in 20 days. -/
theorem work_days_for_A (Wb : ℕ) (Wa : ℕ) (H_wa : Wa = 3 * Wb) (H_total : (Wa + Wb) * 15 = Wa * 20) : A_work_days = 20 :=
by
  sorry

end NUMINAMATH_GPT_work_days_for_A_l2040_204067


namespace NUMINAMATH_GPT_fraction_tabs_closed_l2040_204060

theorem fraction_tabs_closed (x : ℝ) (h₁ : 400 * (1 - x) * (3/5) * (1/2) = 90) : 
  x = 1 / 4 :=
by
  have := h₁
  sorry

end NUMINAMATH_GPT_fraction_tabs_closed_l2040_204060


namespace NUMINAMATH_GPT_teams_match_count_l2040_204068

theorem teams_match_count
  (n : ℕ)
  (h : n = 6)
: (n * (n - 1)) / 2 = 15 := by
  sorry

end NUMINAMATH_GPT_teams_match_count_l2040_204068


namespace NUMINAMATH_GPT_infinitely_many_good_numbers_seven_does_not_divide_good_number_l2040_204062

-- Define what it means for a number to be good
def is_good_number (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a + b = n ∧ (a * b) ∣ (n^2 + n + 1)

-- Part (a): Show that there are infinitely many good numbers
theorem infinitely_many_good_numbers : ∃ (f : ℕ → ℕ), ∀ n, is_good_number (f n) :=
sorry

-- Part (b): Show that if n is a good number, then 7 does not divide n
theorem seven_does_not_divide_good_number (n : ℕ) (h : is_good_number n) : ¬ (7 ∣ n) :=
sorry

end NUMINAMATH_GPT_infinitely_many_good_numbers_seven_does_not_divide_good_number_l2040_204062


namespace NUMINAMATH_GPT_problem1_problem2_l2040_204086

noncomputable def f (x a : ℝ) := x - (x^2 + a * x) / Real.exp x

theorem problem1 (x : ℝ) : (f x 1) ≥ 0 := by
  sorry

theorem problem2 (x : ℝ) : (1 - (Real.log x) / x) * (f x (-1)) > 1 - 1/(Real.exp 2) := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l2040_204086


namespace NUMINAMATH_GPT_circle_center_radius_l2040_204015

theorem circle_center_radius {x y : ℝ} :
  (∃ r : ℝ, (x - 1)^2 + y^2 = r^2) ↔ (x^2 + y^2 - 2*x - 5 = 0) :=
by sorry

end NUMINAMATH_GPT_circle_center_radius_l2040_204015


namespace NUMINAMATH_GPT_increase_percent_exceeds_l2040_204025

theorem increase_percent_exceeds (p q M : ℝ) (M_positive : 0 < M) (p_positive : 0 < p) (q_positive : 0 < q) (q_less_p : q < p) :
  (M * (1 + p / 100) * (1 + q / 100) > M) ↔ (0 < p ∧ 0 < q) :=
by
  sorry

end NUMINAMATH_GPT_increase_percent_exceeds_l2040_204025


namespace NUMINAMATH_GPT_zero_point_interval_l2040_204018

noncomputable def f (x : ℝ) : ℝ := (4 / x) - (2^x)

theorem zero_point_interval : ∃ x : ℝ, (1 < x ∧ x < 1.5) ∧ f x = 0 :=
sorry

end NUMINAMATH_GPT_zero_point_interval_l2040_204018


namespace NUMINAMATH_GPT_problem1_problem2_l2040_204009

open Set

-- Part (1)
theorem problem1 (a : ℝ) :
  (∀ x, x ∉ Icc (0 : ℝ) (2 : ℝ) → x ∈ Icc (a : ℝ) (3 - 2 * a : ℝ)) ∨ (∀ x, x ∈ Icc (a : ℝ) (3 - 2 * a : ℝ) → x ∉ Icc (0 : ℝ) (2 : ℝ)) → a ≤ 0 := 
sorry

-- Part (2)
theorem problem2 (a : ℝ) :
  (¬ ∀ x, x ∈ Icc (a : ℝ) (3 - 2 * a : ℝ) → x ∈ Icc (0 : ℝ) (2 : ℝ)) → (a < 0.5 ∨ a > 1) :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l2040_204009


namespace NUMINAMATH_GPT_first_divisor_l2040_204078

theorem first_divisor (y : ℝ) (x : ℝ) (h1 : 320 / (y * 3) = x) (h2 : x = 53.33) : y = 2 :=
sorry

end NUMINAMATH_GPT_first_divisor_l2040_204078


namespace NUMINAMATH_GPT_inequality_solution_sets_min_value_exists_l2040_204045

-- Define the function f
def f (x : ℝ) (m : ℝ) : ℝ := m * x^2 - 2 * x - 3

-- Existence of roots at -1 and n
def roots_of_quadratic (m : ℝ) (n : ℝ) : Prop :=
  m * (-1)^2 - 2 * (-1) - 3 = 0 ∧ m * n^2 - 2 * n - 3 = 0 ∧ m > 0

-- Main problem statements
theorem inequality_solution_sets (a : ℝ) (m : ℝ) (n : ℝ)
  (h1 : roots_of_quadratic m n) (h2 : m = 1) (h3 : n = 3) (h4 : a > 0) :
  if 0 < a ∧ a ≤ 1 then 
    ∀ x : ℝ, x > 2 / a ∨ x < 2
  else if 1 < a ∧ a < 2 then
    ∀ x : ℝ, x > 2 ∨ x < 2 / a
  else 
    False :=
sorry

theorem min_value_exists (a : ℝ) (m : ℝ)
  (h1 : 0 < a ∧ a < 1) (h2 : m = 1) (h3 : f (a^2) m - 3*a^3 = -5) :
  a = (Real.sqrt 5 - 1) / 2 :=
sorry

end NUMINAMATH_GPT_inequality_solution_sets_min_value_exists_l2040_204045


namespace NUMINAMATH_GPT_people_per_apartment_l2040_204040

/-- A 25 story building has 4 apartments on each floor. 
There are 200 people in the building. 
Prove that each apartment houses 2 people. -/
theorem people_per_apartment (stories : ℕ) (apartments_per_floor : ℕ) (total_people : ℕ)
    (h_stories : stories = 25)
    (h_apartments_per_floor : apartments_per_floor = 4)
    (h_total_people : total_people = 200) :
  (total_people / (stories * apartments_per_floor)) = 2 :=
by
  sorry

end NUMINAMATH_GPT_people_per_apartment_l2040_204040


namespace NUMINAMATH_GPT_greatest_constant_right_triangle_l2040_204099

theorem greatest_constant_right_triangle (a b c : ℝ) (h : c^2 = a^2 + b^2) (K : ℝ) 
    (hK : (a^2 + b^2) / (a^2 + b^2 + c^2) > K) : 
    K ≤ 1 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_greatest_constant_right_triangle_l2040_204099


namespace NUMINAMATH_GPT_class_books_transfer_l2040_204081

theorem class_books_transfer :
  ∀ (A B n : ℕ), 
    A = 200 → B = 200 → 
    (B + n = 3/2 * (A - n)) →
    n = 40 :=
by sorry

end NUMINAMATH_GPT_class_books_transfer_l2040_204081


namespace NUMINAMATH_GPT_problem_part1_problem_part2_problem_part3_l2040_204019

noncomputable def find_ab (a b : ℝ) : Prop :=
  (5 * a + b = 40) ∧ (30 * a + b = 140)

noncomputable def production_cost (x : ℕ) : Prop :=
  (4 * x + 20 + 7 * (100 - x) = 660)

noncomputable def transport_cost (m : ℝ) : Prop :=
  ∃ n : ℝ, 10 ≤ n ∧ n ≤ 20 ∧ (m - 2) * n + 130 = 150

theorem problem_part1 : ∃ (a b : ℝ), find_ab a b ∧ a = 4 ∧ b = 20 := 
  sorry

theorem problem_part2 : ∃ (x : ℕ), production_cost x ∧ x = 20 := 
  sorry

theorem problem_part3 : ∃ (m : ℝ), transport_cost m ∧ m = 4 := 
  sorry

end NUMINAMATH_GPT_problem_part1_problem_part2_problem_part3_l2040_204019


namespace NUMINAMATH_GPT_find_p_l2040_204084

variable (m n p : ℚ)

theorem find_p (h1 : m = 8 * n + 5) (h2 : m + 2 = 8 * (n + p) + 5) : p = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_find_p_l2040_204084


namespace NUMINAMATH_GPT_deepak_age_l2040_204093

theorem deepak_age
  (A D : ℕ)
  (h1 : A / D = 5 / 7)
  (h2 : A + 6 = 36) :
  D = 42 :=
by sorry

end NUMINAMATH_GPT_deepak_age_l2040_204093


namespace NUMINAMATH_GPT_rowing_time_to_place_and_back_l2040_204075

open Real

/-- Definitions of the problem conditions -/
def rowing_speed_still_water : ℝ := 5
def current_speed : ℝ := 1
def distance_to_place : ℝ := 2.4

/-- Proof statement: the total time taken to row to the place and back is 1 hour -/
theorem rowing_time_to_place_and_back :
  (distance_to_place / (rowing_speed_still_water + current_speed)) + 
  (distance_to_place / (rowing_speed_still_water - current_speed)) =
  1 := by
  sorry

end NUMINAMATH_GPT_rowing_time_to_place_and_back_l2040_204075


namespace NUMINAMATH_GPT_find_integer_to_satisfy_eq_l2040_204095

theorem find_integer_to_satisfy_eq (n : ℤ) (h : n - 5 = 2) : n = 7 :=
sorry

end NUMINAMATH_GPT_find_integer_to_satisfy_eq_l2040_204095


namespace NUMINAMATH_GPT_percentage_of_x_l2040_204097

theorem percentage_of_x (x y : ℝ) (h1 : y = x / 4) (p : ℝ) (h2 : p / 100 * x = 20 / 100 * y) : p = 5 :=
by sorry

end NUMINAMATH_GPT_percentage_of_x_l2040_204097


namespace NUMINAMATH_GPT_g_of_zero_l2040_204071

theorem g_of_zero (f g : ℤ → ℤ) (h₁ : ∀ x, f x = 2 * x + 3) (h₂ : ∀ x, g (x + 2) = f x) : 
  g 0 = -1 :=
by
  sorry

end NUMINAMATH_GPT_g_of_zero_l2040_204071


namespace NUMINAMATH_GPT_sum_of_dimensions_l2040_204083

theorem sum_of_dimensions (A B C : ℝ) (h1 : A * B = 50) (h2 : A * C = 90) (h3 : B * C = 100) : A + B + C = 24 :=
  sorry

end NUMINAMATH_GPT_sum_of_dimensions_l2040_204083


namespace NUMINAMATH_GPT_quadratic_sum_l2040_204048

theorem quadratic_sum (b c : ℝ) : 
  (∀ x : ℝ, x^2 - 24 * x + 50 = (x + b)^2 + c) → b + c = -106 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_quadratic_sum_l2040_204048


namespace NUMINAMATH_GPT_mass_of_substance_l2040_204076

-- The conditions
def substance_density (mass_cubic_meter_kg : ℝ) (volume_cubic_meter_cm3 : ℝ) : Prop :=
  mass_cubic_meter_kg = 100 ∧ volume_cubic_meter_cm3 = 1*1000000

def specific_amount_volume_cm3 (volume_cm3 : ℝ) : Prop :=
  volume_cm3 = 10

-- The Proof Statement
theorem mass_of_substance (mass_cubic_meter_kg : ℝ) (volume_cubic_meter_cm3 : ℝ) (volume_cm3 : ℝ) (mass_grams : ℝ) :
  substance_density mass_cubic_meter_kg volume_cubic_meter_cm3 →
  specific_amount_volume_cm3 volume_cm3 →
  mass_grams = 10 :=
by
  intros hDensity hVolume
  sorry

end NUMINAMATH_GPT_mass_of_substance_l2040_204076


namespace NUMINAMATH_GPT_white_tshirts_per_pack_l2040_204089

def packs_of_white := 3
def packs_of_blue := 2
def blue_in_each_pack := 4
def total_tshirts := 26

theorem white_tshirts_per_pack :
  ∃ W : ℕ, packs_of_white * W + packs_of_blue * blue_in_each_pack = total_tshirts ∧ W = 6 :=
by
  sorry

end NUMINAMATH_GPT_white_tshirts_per_pack_l2040_204089


namespace NUMINAMATH_GPT_unique_intersection_l2040_204034

open Real

-- Defining the functions f and g as per the conditions
def f (b : ℝ) (x : ℝ) : ℝ := b * x^2 + 5 * x + 3
def g (x : ℝ) : ℝ := -2 * x - 2

-- The condition that the intersection occurs at one point translates to a specific b satisfying the discriminant condition.
theorem unique_intersection (b : ℝ) : (∃ x : ℝ, f b x = g x) ∧ (f b x = g x → ∀ y : ℝ, y ≠ x → f b y ≠ g y) ↔ b = 49 / 20 :=
by {
  sorry
}

end NUMINAMATH_GPT_unique_intersection_l2040_204034


namespace NUMINAMATH_GPT_largest_inscribed_rectangle_l2040_204094

theorem largest_inscribed_rectangle {a b m : ℝ} (h : m ≥ b) :
  ∃ (base height area : ℝ),
    base = a * (b + m) / m ∧ 
    height = (b + m) / 2 ∧ 
    area = a * (b + m)^2 / (2 * m) :=
sorry

end NUMINAMATH_GPT_largest_inscribed_rectangle_l2040_204094


namespace NUMINAMATH_GPT_disk_max_areas_l2040_204032

-- Conditions Definition
def disk_divided (n : ℕ) : ℕ :=
  let radii := 3 * n
  let secant_lines := 2
  let total_areas := 9 * n
  total_areas

theorem disk_max_areas (n : ℕ) : disk_divided n = 9 * n :=
by
  sorry

end NUMINAMATH_GPT_disk_max_areas_l2040_204032


namespace NUMINAMATH_GPT_find_a_l2040_204003

-- Define the domains of the functions f and g
def A : Set ℝ :=
  {x | x < -1 ∨ x ≥ 1}

def B (a : ℝ) : Set ℝ :=
  {x | 2 * a < x ∧ x < a + 1}

-- Restate the problem as a Lean proposition
theorem find_a (a : ℝ) (h : a < 1) (hb : B a ⊆ A) :
  a ∈ {x | x ≤ -2 ∨ (1 / 2 ≤ x ∧ x < 1)} :=
sorry

end NUMINAMATH_GPT_find_a_l2040_204003


namespace NUMINAMATH_GPT_total_snacks_l2040_204012

variable (peanuts : ℝ) (raisins : ℝ)

theorem total_snacks (h1 : peanuts = 0.1) (h2 : raisins = 0.4) : peanuts + raisins = 0.5 :=
by
  sorry

end NUMINAMATH_GPT_total_snacks_l2040_204012


namespace NUMINAMATH_GPT_find_number_l2040_204061

theorem find_number (x : ℝ) : (35 - x) * 2 + 12 = 72 → ((35 - x) * 2 + 12) / 8 = 9 → x = 5 :=
by
  -- assume the first condition
  intro h1
  -- assume the second condition
  intro h2
  -- the proof goes here
  sorry

end NUMINAMATH_GPT_find_number_l2040_204061


namespace NUMINAMATH_GPT_xiaohua_apples_l2040_204021

theorem xiaohua_apples (x : ℕ) (h1 : ∃ n, (n = 4 * x + 20)) 
                       (h2 : (4 * x + 20 - 8 * (x - 1) > 0) ∧ (4 * x + 20 - 8 * (x - 1) < 8)) : 
                       4 * x + 20 = 44 := by
  sorry

end NUMINAMATH_GPT_xiaohua_apples_l2040_204021


namespace NUMINAMATH_GPT_find_x_l2040_204063

theorem find_x (y x : ℝ) (h : x / (x - 1) = (y^2 + 2 * y - 1) / (y^2 + 2 * y - 2)) : 
  x = y^2 + 2 * y - 1 := 
sorry

end NUMINAMATH_GPT_find_x_l2040_204063


namespace NUMINAMATH_GPT_kids_go_to_camp_l2040_204046

theorem kids_go_to_camp (total_kids: Nat) (kids_stay_home: Nat) 
  (h1: total_kids = 1363293) (h2: kids_stay_home = 907611) : total_kids - kids_stay_home = 455682 :=
by
  have h_total : total_kids = 1363293 := h1
  have h_stay_home : kids_stay_home = 907611 := h2
  sorry

end NUMINAMATH_GPT_kids_go_to_camp_l2040_204046


namespace NUMINAMATH_GPT_minimum_value_l2040_204014

theorem minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 20) :
  (∃ (m : ℝ), m = (1 / x ^ 2 + 1 / y ^ 2) ∧ m ≥ 2 / 25) :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_l2040_204014


namespace NUMINAMATH_GPT_complement_intersection_eq_interval_l2040_204079

open Set

noncomputable def M : Set ℝ := {x | 3 * x - 1 >= 0}
noncomputable def N : Set ℝ := {x | 0 < x ∧ x < 1 / 2}

theorem complement_intersection_eq_interval :
  (M ∩ N)ᶜ = (Iio (1 / 3) ∪ Ici (1 / 2)) :=
by
  -- proof will go here in the actual development
  sorry

end NUMINAMATH_GPT_complement_intersection_eq_interval_l2040_204079


namespace NUMINAMATH_GPT_length_of_book_l2040_204056

theorem length_of_book (A W L : ℕ) (hA : A = 50) (hW : W = 10) (hArea : A = L * W) : L = 5 := 
sorry

end NUMINAMATH_GPT_length_of_book_l2040_204056


namespace NUMINAMATH_GPT_chord_length_count_l2040_204091

noncomputable def number_of_chords (d r : ℕ) : ℕ := sorry

theorem chord_length_count {d r : ℕ} (h1 : d = 12) (h2 : r = 13) :
  number_of_chords d r = 17 :=
sorry

end NUMINAMATH_GPT_chord_length_count_l2040_204091


namespace NUMINAMATH_GPT_greatest_valid_number_l2040_204033

-- Define the conditions
def is_valid_number (n : ℕ) : Prop :=
  n < 200 ∧ Nat.gcd n 30 = 5

-- Formulate the proof problem
theorem greatest_valid_number : ∃ n, is_valid_number n ∧ (∀ m, is_valid_number m → m ≤ n) ∧ n = 185 := 
by
  sorry

end NUMINAMATH_GPT_greatest_valid_number_l2040_204033


namespace NUMINAMATH_GPT_greatest_possible_length_l2040_204047

theorem greatest_possible_length (a b c : ℕ) (h1 : a = 28) (h2 : b = 45) (h3 : c = 63) : 
  Nat.gcd (Nat.gcd a b) c = 7 :=
by
  sorry

end NUMINAMATH_GPT_greatest_possible_length_l2040_204047


namespace NUMINAMATH_GPT_hiring_manager_acceptance_l2040_204070

theorem hiring_manager_acceptance :
  let average_age := 31
  let std_dev := 9
  let max_diff_ages := 19
  let k := max_diff_ages / (2 * std_dev)
  k = 19 / 18 :=
by
  let average_age := 31
  let std_dev := 9
  let max_diff_ages := 19
  let k := max_diff_ages / (2 * std_dev)
  show k = 19 / 18
  sorry

end NUMINAMATH_GPT_hiring_manager_acceptance_l2040_204070


namespace NUMINAMATH_GPT_range_of_a_l2040_204035

variable (a : ℝ) (x : ℝ)

theorem range_of_a
  (h1 : 2 * x < 3 * (x - 3) + 1)
  (h2 : (3 * x + 2) / 4 > x + a) :
  -11 / 4 ≤ a ∧ a < -5 / 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l2040_204035


namespace NUMINAMATH_GPT_multiplication_is_valid_l2040_204052

-- Define that the three-digit number n = 306
def three_digit_number := 306

-- The multiplication by 1995 should result in the defined product
def valid_multiplication (n : ℕ) := 1995 * n

theorem multiplication_is_valid : valid_multiplication three_digit_number = 1995 * 306 := by
  -- Since we only need the statement, we use sorry here
  sorry

end NUMINAMATH_GPT_multiplication_is_valid_l2040_204052


namespace NUMINAMATH_GPT_arithmetic_mean_common_difference_l2040_204022

theorem arithmetic_mean_common_difference (a : ℕ → ℝ) (d : ℝ) 
    (h1 : ∀ n, a (n + 1) = a n + d) 
    (h2 : a 1 + a 4 = 2 * (a 2 + 1))
    : d = 2 := 
by 
  -- Proof is omitted as it is not required.
  sorry

end NUMINAMATH_GPT_arithmetic_mean_common_difference_l2040_204022


namespace NUMINAMATH_GPT_max_value_of_x_plus_y_l2040_204037

variable (x y : ℝ)

-- Define the condition
def condition : Prop := x^2 + y + 3 * x - 3 = 0

-- Define the proof statement
theorem max_value_of_x_plus_y (hx : condition x y) : x + y ≤ 4 :=
sorry

end NUMINAMATH_GPT_max_value_of_x_plus_y_l2040_204037


namespace NUMINAMATH_GPT_hexahedron_volume_l2040_204054

open Real

noncomputable def volume_of_hexahedron (AB A1B1 AA1 : ℝ) : ℝ :=
  let S_base := (3 * sqrt 3 / 2) * AB^2
  let S_top := (3 * sqrt 3 / 2) * A1B1^2
  let h := AA1
  (1 / 3) * h * (S_base + sqrt (S_base * S_top) + S_top)

theorem hexahedron_volume : volume_of_hexahedron 2 3 (sqrt 10) = 57 * sqrt 3 / 2 := by
  sorry

end NUMINAMATH_GPT_hexahedron_volume_l2040_204054


namespace NUMINAMATH_GPT_smallest_possible_value_abs_sum_l2040_204073

theorem smallest_possible_value_abs_sum : 
  ∀ (x : ℝ), 
    (|x + 3| + |x + 6| + |x + 7| + 2) ≥ 8 :=
by
  sorry

end NUMINAMATH_GPT_smallest_possible_value_abs_sum_l2040_204073


namespace NUMINAMATH_GPT_variation_of_variables_l2040_204008

variables (k j : ℝ) (x y z : ℝ)

theorem variation_of_variables (h1 : x = k * y^2) (h2 : y = j * z^3) : ∃ m : ℝ, x = m * z^6 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_variation_of_variables_l2040_204008


namespace NUMINAMATH_GPT_intersection_with_y_axis_l2040_204066

-- Define the original linear function
def original_function (x : ℝ) : ℝ := -2 * x + 3

-- Define the function after moving it up by 2 units
def moved_up_function (x : ℝ) : ℝ := original_function x + 2

-- State the theorem to prove the intersection with the y-axis
theorem intersection_with_y_axis : moved_up_function 0 = 5 :=
by
  sorry

end NUMINAMATH_GPT_intersection_with_y_axis_l2040_204066


namespace NUMINAMATH_GPT_number_of_people_study_only_cooking_l2040_204049

def total_yoga : Nat := 25
def total_cooking : Nat := 18
def total_weaving : Nat := 10
def cooking_and_yoga : Nat := 5
def all_three : Nat := 4
def cooking_and_weaving : Nat := 5

theorem number_of_people_study_only_cooking :
  (total_cooking - (cooking_and_yoga + cooking_and_weaving - all_three)) = 12 :=
by
  sorry

end NUMINAMATH_GPT_number_of_people_study_only_cooking_l2040_204049


namespace NUMINAMATH_GPT_smallest_m_l2040_204050

theorem smallest_m (m : ℕ) (p q : ℤ) (h_eq : 12 * p * p - m * p + 432 = 0) (h_sum : p + q = m / 12) (h_prod : p * q = 36) :
  m = 144 :=
by
  sorry

end NUMINAMATH_GPT_smallest_m_l2040_204050


namespace NUMINAMATH_GPT_selling_price_range_l2040_204011

theorem selling_price_range
  (unit_purchase_price : ℝ)
  (initial_selling_price : ℝ)
  (initial_sales_volume : ℝ)
  (price_increase_effect : ℝ)
  (daily_profit_threshold : ℝ)
  (x : ℝ) :
  unit_purchase_price = 8 →
  initial_selling_price = 10 →
  initial_sales_volume = 100 →
  price_increase_effect = 10 →
  daily_profit_threshold = 320 →
  (initial_selling_price - unit_purchase_price) * initial_sales_volume > daily_profit_threshold →
  12 < x → x < 16 →
  (x - unit_purchase_price) * (initial_sales_volume - price_increase_effect * (x - initial_selling_price)) > daily_profit_threshold :=
sorry

end NUMINAMATH_GPT_selling_price_range_l2040_204011


namespace NUMINAMATH_GPT_identify_letter_R_l2040_204023

variable (x y : ℕ)

def date_A : ℕ := x + 2
def date_B : ℕ := x + 5
def date_E : ℕ := x

def y_plus_x := y + x
def combined_dates := date_A x + 2 * date_B x

theorem identify_letter_R (h1 : y_plus_x x y = combined_dates x) : 
  y = 2 * x + 12 ∧ ∃ (letter : String), letter = "R" := sorry

end NUMINAMATH_GPT_identify_letter_R_l2040_204023


namespace NUMINAMATH_GPT_residue_mod_13_l2040_204085

theorem residue_mod_13 : 
  (156 % 13 = 0) ∧ (52 % 13 = 0) ∧ (182 % 13 = 0) ∧ (26 % 13 = 0) →
  (156 + 3 * 52 + 4 * 182 + 6 * 26) % 13 = 0 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_residue_mod_13_l2040_204085


namespace NUMINAMATH_GPT_cos_double_angle_l2040_204051

theorem cos_double_angle (theta : ℝ) (h : Real.cos theta = 3 / 5) : Real.cos (2 * theta) = -7 / 25 :=
by
  sorry

end NUMINAMATH_GPT_cos_double_angle_l2040_204051


namespace NUMINAMATH_GPT_number_of_soccer_campers_l2040_204002

-- Conditions as definitions in Lean
def total_campers : ℕ := 88
def basketball_campers : ℕ := 24
def football_campers : ℕ := 32
def soccer_campers : ℕ := total_campers - (basketball_campers + football_campers)

-- Theorem statement to prove
theorem number_of_soccer_campers : soccer_campers = 32 := by
  sorry

end NUMINAMATH_GPT_number_of_soccer_campers_l2040_204002


namespace NUMINAMATH_GPT_cody_initial_marbles_l2040_204024

theorem cody_initial_marbles (x : ℕ) (h1 : x - 5 = 7) : x = 12 := by
  sorry

end NUMINAMATH_GPT_cody_initial_marbles_l2040_204024


namespace NUMINAMATH_GPT_polynomial_solution_l2040_204090

noncomputable def q (x : ℝ) : ℝ :=
  -20 / 93 * x^3 - 110 / 93 * x^2 - 372 / 93 * x - 525 / 93

theorem polynomial_solution :
  (q 1 = -11) ∧
  (q 2 = -15) ∧
  (q 3 = -25) ∧
  (q 5 = -65) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_solution_l2040_204090


namespace NUMINAMATH_GPT_minimize_area_eq_l2040_204058

theorem minimize_area_eq {l : ℝ → ℝ → Prop}
  (P : ℝ × ℝ) (A B : ℝ × ℝ)
  (condition1 : l P.1 P.2)
  (condition2 : A.1 > 0 ∧ A.2 = 0)
  (condition3 : B.1 = 0 ∧ B.2 > 0)
  (line_eq : ∀ x y : ℝ, l x y ↔ (2 * x + y = 4)) :
  ∀ (a b : ℝ), a = 2 → b = 4 → 2 * P.1 + P.2 = 4 :=
by sorry

end NUMINAMATH_GPT_minimize_area_eq_l2040_204058


namespace NUMINAMATH_GPT_number_of_flowers_alissa_picked_l2040_204039

-- Define the conditions
variable (A : ℕ) -- Number of flowers Alissa picked
variable (M : ℕ) -- Number of flowers Melissa picked
variable (flowers_gifted : ℕ := 18) -- Flowers given to mother
variable (flowers_left : ℕ := 14) -- Flowers left after gifting

-- Define that Melissa picked the same number of flowers as Alissa
axiom pick_equal : M = A

-- Define the total number of flowers they had initially
axiom total_flowers : 2 * A = flowers_gifted + flowers_left

-- Prove that Alissa picked 16 flowers
theorem number_of_flowers_alissa_picked : A = 16 := by
  -- Use placeholders for proof steps
  sorry

end NUMINAMATH_GPT_number_of_flowers_alissa_picked_l2040_204039


namespace NUMINAMATH_GPT_jane_vases_per_day_l2040_204017

theorem jane_vases_per_day : 
  ∀ (total_vases : ℝ) (days : ℝ), 
  total_vases = 248 → days = 16 → 
  (total_vases / days) = 15.5 :=
by
  intros total_vases days h_total_vases h_days
  rw [h_total_vases, h_days]
  norm_num

end NUMINAMATH_GPT_jane_vases_per_day_l2040_204017


namespace NUMINAMATH_GPT_Joan_seashells_l2040_204057

theorem Joan_seashells (J_J : ℕ) (J : ℕ) (h : J + J_J = 14) (hJJ : J_J = 8) : J = 6 :=
by
  sorry

end NUMINAMATH_GPT_Joan_seashells_l2040_204057


namespace NUMINAMATH_GPT_find_x_l2040_204077

theorem find_x (x : ℝ) (h : (1 / 2) * x + (1 / 3) * x = (1 / 4) * x + 7) : x = 12 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l2040_204077


namespace NUMINAMATH_GPT_largest_possible_b_l2040_204072

theorem largest_possible_b (a b c : ℕ) (h1 : 1 < c) (h2 : c < b) (h3 : b < a) (h4 : a * b * c = 360) : b = 10 :=
sorry

end NUMINAMATH_GPT_largest_possible_b_l2040_204072


namespace NUMINAMATH_GPT_total_bill_l2040_204001

def num_adults := 2
def num_children := 5
def cost_per_meal := 3

theorem total_bill : (num_adults + num_children) * cost_per_meal = 21 := 
by 
  sorry

end NUMINAMATH_GPT_total_bill_l2040_204001


namespace NUMINAMATH_GPT_total_spending_eq_total_is_19_l2040_204027

variable (friend_spending your_spending total_spending : ℕ)

-- Conditions
def friend_spending_eq : friend_spending = 11 := by sorry
def friend_spent_more : friend_spending = your_spending + 3 := by sorry

-- Proof that total_spending is 19
theorem total_spending_eq : total_spending = friend_spending + your_spending :=
  by sorry

theorem total_is_19 : total_spending = 19 :=
  by sorry

end NUMINAMATH_GPT_total_spending_eq_total_is_19_l2040_204027


namespace NUMINAMATH_GPT_pascal_sixth_element_row_20_l2040_204087

theorem pascal_sixth_element_row_20 : (Nat.choose 20 5) = 7752 := 
  by
  sorry

end NUMINAMATH_GPT_pascal_sixth_element_row_20_l2040_204087


namespace NUMINAMATH_GPT_charles_cleaning_time_l2040_204007

theorem charles_cleaning_time :
  let Alice_time := 20
  let Bob_time := (3/4) * Alice_time
  let Charles_time := (2/3) * Bob_time
  Charles_time = 10 :=
by
  sorry

end NUMINAMATH_GPT_charles_cleaning_time_l2040_204007


namespace NUMINAMATH_GPT_find_y_l2040_204030

theorem find_y (y : ℝ) (hy : 0 < y) 
  (h : (Real.sqrt (12 * y)) * (Real.sqrt (6 * y)) * (Real.sqrt (18 * y)) * (Real.sqrt (9 * y)) = 27) : 
  y = 1 / 2 := 
sorry

end NUMINAMATH_GPT_find_y_l2040_204030


namespace NUMINAMATH_GPT_circle_radius_zero_l2040_204092

theorem circle_radius_zero :
  ∀ (x y : ℝ),
    (4 * x^2 - 8 * x + 4 * y^2 - 16 * y + 20 = 0) →
    ((x - 1)^2 + (y - 2)^2 = 0) → 
    0 = 0 :=
by
  intros x y h_eq h_circle
  sorry

end NUMINAMATH_GPT_circle_radius_zero_l2040_204092


namespace NUMINAMATH_GPT_total_students_in_school_l2040_204044

noncomputable def total_students (girls boys : ℕ) (ratio_girls boys_ratio : ℕ) : ℕ :=
  let parts := ratio_girls + boys_ratio
  let students_per_part := girls / ratio_girls
  students_per_part * parts

theorem total_students_in_school (girls : ℕ) (ratio_girls boys_ratio : ℕ) (h1 : ratio_girls = 5) (h2 : boys_ratio = 8) (h3 : girls = 160) :
  total_students girls boys_ratio ratio_girls = 416 :=
  by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_total_students_in_school_l2040_204044


namespace NUMINAMATH_GPT_range_of_a_l2040_204029

theorem range_of_a (a : ℝ) : (4 - a < 0) → (a > 4) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_range_of_a_l2040_204029


namespace NUMINAMATH_GPT_distinct_nonzero_reals_xy_six_l2040_204036

theorem distinct_nonzero_reals_xy_six (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + 6/x = y + 6/y) (h_distinct : x ≠ y) : x * y = 6 := 
sorry

end NUMINAMATH_GPT_distinct_nonzero_reals_xy_six_l2040_204036


namespace NUMINAMATH_GPT_arithmetic_sequence_a1_geometric_sequence_sum_l2040_204013

-- Definition of the arithmetic sequence problem
theorem arithmetic_sequence_a1 (a_n s_n : ℕ) (d : ℕ) (h1 : a_n = 32) (h2 : s_n = 63) (h3 : d = 11) :
  ∃ a_1 : ℕ, a_1 = 10 :=
by
  sorry

-- Definition of the geometric sequence problem
theorem geometric_sequence_sum (a_1 q : ℕ) (h1 : a_1 = 1) (h2 : q = 2) (m : ℕ) :
  let a_m := a_1 * (q ^ (m - 1))
  let a_m_sq := a_m * a_m
  let sm'_sum := (1 - 4^m) / (1 - 4)
  sm'_sum = (4^m - 1) / 3 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a1_geometric_sequence_sum_l2040_204013


namespace NUMINAMATH_GPT_friends_count_l2040_204005

variables (F : ℕ)
def cindy_initial_marbles : ℕ := 500
def marbles_per_friend : ℕ := 80
def marbles_given : ℕ := F * marbles_per_friend
def marbles_remaining := cindy_initial_marbles - marbles_given

theorem friends_count (h : 4 * marbles_remaining = 720) : F = 4 :=
by sorry

end NUMINAMATH_GPT_friends_count_l2040_204005


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l2040_204000

variable {a : ℕ → ℤ} 
variable {a_3 a_4 a_5 : ℤ}

-- Hypothesis: arithmetic sequence and given condition
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n, a (n+1) - a n = a 2 - a 1

theorem arithmetic_sequence_sum (h : is_arithmetic_sequence a) (h_sum : a_3 + a_4 + a_5 = 12) : 
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 28 := 
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l2040_204000


namespace NUMINAMATH_GPT_investment_ratio_correct_l2040_204064

variable (P Q : ℝ)
variable (investment_ratio: ℝ := 7 / 5)
variable (profit_ratio: ℝ := 7 / 10)
variable (time_p: ℝ := 7)
variable (time_q: ℝ := 14)

theorem investment_ratio_correct :
  (P * time_p) / (Q * time_q) = profit_ratio → (P / Q) = investment_ratio := 
by
  sorry

end NUMINAMATH_GPT_investment_ratio_correct_l2040_204064


namespace NUMINAMATH_GPT_solution_set_abs_le_one_inteval_l2040_204026

theorem solution_set_abs_le_one_inteval (x : ℝ) : |x| ≤ 1 ↔ -1 ≤ x ∧ x ≤ 1 :=
by sorry

end NUMINAMATH_GPT_solution_set_abs_le_one_inteval_l2040_204026


namespace NUMINAMATH_GPT_exists_two_linear_functions_l2040_204010

-- Define the quadratic trinomials and their general forms
variables (a b c d e f : ℝ)
-- Assuming coefficients a and d are non-zero
variable (ha : a ≠ 0)
variable (hd : d ≠ 0)

-- Define the linear function
def ell (m n x : ℝ) : ℝ := m * x + n

-- Define the quadratic trinomials P(x) and Q(x) 
def P (x : ℝ) := a * x^2 + b * x + c
def Q (x : ℝ) := d * x^2 + e * x + f

-- Prove that there exist exactly two linear functions ell(x) that satisfy the condition for all x
theorem exists_two_linear_functions : 
  ∃ (m1 m2 n1 n2 : ℝ), 
  (∀ x, P a b c x = Q d e f (ell m1 n1 x)) ∧ 
  (∀ x, P a b c x = Q d e f (ell m2 n2 x)) := 
sorry

end NUMINAMATH_GPT_exists_two_linear_functions_l2040_204010


namespace NUMINAMATH_GPT_mixed_sum_in_range_l2040_204031

def mixed_to_improper (a : ℕ) (b c : ℕ) : ℚ := a + b / c

def mixed_sum (a1 a2 a3 b1 b2 b3 c1 c2 c3 : ℕ) : ℚ :=
  (mixed_to_improper a1 b1 c1) + (mixed_to_improper a2 b2 c2) + (mixed_to_improper a3 b3 c3)

theorem mixed_sum_in_range :
  11 < mixed_sum 1 4 6 3 1 2 8 3 21 ∧ mixed_sum 1 4 6 3 1 2 8 3 21 < 12 :=
by { sorry }

end NUMINAMATH_GPT_mixed_sum_in_range_l2040_204031


namespace NUMINAMATH_GPT_triangle_base_length_l2040_204042

theorem triangle_base_length (base : ℝ) (h1 : ∃ (side : ℝ), side = 6 ∧ (side^2 = (base * 12) / 2)) : base = 6 :=
sorry

end NUMINAMATH_GPT_triangle_base_length_l2040_204042


namespace NUMINAMATH_GPT_second_smallest_packs_of_hot_dogs_l2040_204020

theorem second_smallest_packs_of_hot_dogs (n m : ℕ) (k : ℕ) :
  (12 * n ≡ 5 [MOD 10]) ∧ (10 * m ≡ 3 [MOD 12]) → n = 15 :=
by
  sorry

end NUMINAMATH_GPT_second_smallest_packs_of_hot_dogs_l2040_204020


namespace NUMINAMATH_GPT_find_prices_l2040_204041

variables (C S : ℕ) -- Using natural numbers to represent rubles

theorem find_prices (h1 : C + S = 2500) (h2 : 4 * C + 3 * S = 8870) :
  C = 1370 ∧ S = 1130 :=
by
  sorry

end NUMINAMATH_GPT_find_prices_l2040_204041


namespace NUMINAMATH_GPT_problem_I_problem_II_problem_III_l2040_204055

-- The function f(x)
noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) := (1/2) * x^2 - a * Real.log x + b

-- Tangent line at x = 1
def tangent_condition (a : ℝ) (b : ℝ) :=
  1 - a = 3 ∧ f 1 a b = 0

-- Extreme point at x = 1
def extreme_condition (a : ℝ) :=
  1 - a = 0 

-- Monotonicity and minimum m
def inequality_condition (a m : ℝ) :=
  -2 ≤ a ∧ a < 0 ∧ ∀ (x1 x2 : ℝ), 0 < x1 ∧ x1 ≤ 2 ∧ 0 < x2 ∧ x2 ≤ 2 → 
  |f x1 a (0 : ℝ) - f x2 a 0| ≤ m * |1 / x1 - 1 / x2|

-- Proof problem 1
theorem problem_I : ∃ (a b : ℝ), tangent_condition a b → a = -2 ∧ b = -0.5 := sorry

-- Proof problem 2
theorem problem_II : ∃ (a : ℝ), extreme_condition a → a = 1 := sorry

-- Proof problem 3
theorem problem_III : ∃ (m : ℝ), inequality_condition (-2 : ℝ) m → m = 12 := sorry

end NUMINAMATH_GPT_problem_I_problem_II_problem_III_l2040_204055


namespace NUMINAMATH_GPT_solve_system_addition_l2040_204082

theorem solve_system_addition (a b : ℝ) (h1 : 3 * a + 7 * b = 1977) (h2 : 5 * a + b = 2007) : a + b = 498 :=
by
  sorry

end NUMINAMATH_GPT_solve_system_addition_l2040_204082


namespace NUMINAMATH_GPT_inequalities_validity_l2040_204088

theorem inequalities_validity (x y a b : ℝ) (hx : x ≤ a) (hy : y ≤ b) (hstrict : x < a ∨ y < b) :
  (x + y ≤ a + b) ∧
  ¬((x - y ≤ a - b) ∧ (x * y ≤ a * b) ∧ (x / y ≤ a / b)) :=
by
  -- Here is where the proof would go.
  sorry

end NUMINAMATH_GPT_inequalities_validity_l2040_204088


namespace NUMINAMATH_GPT_part1_part2_l2040_204043

open Set

namespace ProofProblem

variable (m : ℝ)

def A (m : ℝ) := {x : ℝ | 0 < x - m ∧ x - m < 3}
def B := {x : ℝ | x ≤ 0 ∨ x ≥ 3}

theorem part1 : (A 1 ∩ B) = {x : ℝ | 3 ≤ x ∧ x < 4} := by
  sorry

theorem part2 : (∀ m, (A m ∪ B) = B ↔ (m ≥ 3 ∨ m ≤ -3)) := by
  sorry

end ProofProblem

end NUMINAMATH_GPT_part1_part2_l2040_204043


namespace NUMINAMATH_GPT_valid_permutations_remainder_l2040_204096

def countValidPermutations : Nat :=
  let total := (Finset.range 3).sum (fun j =>
    Nat.choose 3 (j + 2) * Nat.choose 5 j * Nat.choose 7 (j + 3))
  total % 1000

theorem valid_permutations_remainder :
  countValidPermutations = 60 := 
  sorry

end NUMINAMATH_GPT_valid_permutations_remainder_l2040_204096


namespace NUMINAMATH_GPT_evaluate_expression_l2040_204006

theorem evaluate_expression : (Real.sqrt ((Real.sqrt 2)^4))^6 = 64 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2040_204006


namespace NUMINAMATH_GPT_sin_X_value_l2040_204053

theorem sin_X_value (a b X : ℝ) (h₁ : (1/2) * a * b * Real.sin X = 72) (h₂ : Real.sqrt (a * b) = 16) :
  Real.sin X = 9 / 16 := by
  sorry

end NUMINAMATH_GPT_sin_X_value_l2040_204053


namespace NUMINAMATH_GPT_calculate_total_cups_l2040_204028

variable (butter : ℕ) (flour : ℕ) (sugar : ℕ) (total_cups : ℕ)

def ratio_condition : Prop :=
  3 * butter = 2 * sugar ∧ 3 * flour = 5 * sugar

def sugar_condition : Prop :=
  sugar = 9

def total_cups_calculation : Prop :=
  total_cups = butter + flour + sugar

theorem calculate_total_cups (h1 : ratio_condition butter flour sugar) (h2 : sugar_condition sugar) :
  total_cups_calculation butter flour sugar total_cups -> total_cups = 30 := by
  sorry

end NUMINAMATH_GPT_calculate_total_cups_l2040_204028


namespace NUMINAMATH_GPT_paula_remaining_money_l2040_204080

theorem paula_remaining_money (initial_amount cost_per_shirt cost_of_pants : ℕ) 
                             (num_shirts : ℕ) (H1 : initial_amount = 109)
                             (H2 : cost_per_shirt = 11) (H3 : num_shirts = 2)
                             (H4 : cost_of_pants = 13) :
  initial_amount - (num_shirts * cost_per_shirt + cost_of_pants) = 74 := 
by
  -- Calculation of total spent and remaining would go here.
  sorry

end NUMINAMATH_GPT_paula_remaining_money_l2040_204080


namespace NUMINAMATH_GPT_intersection_sets_l2040_204038

-- Define set A as all x such that x >= -2
def setA : Set ℝ := {x | x >= -2}

-- Define set B as all x such that x < 1
def setB : Set ℝ := {x | x < 1}

-- The statement to prove in Lean 4
theorem intersection_sets : (setA ∩ setB) = {x | -2 <= x ∧ x < 1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_sets_l2040_204038


namespace NUMINAMATH_GPT_find_b_over_a_find_angle_B_l2040_204004

-- Definitions and main theorems
noncomputable def sides_in_triangle (A B C a b c : ℝ) : Prop :=
  a * (Real.sin A) * (Real.sin B) + b * (Real.cos A) ^ 2 = Real.sqrt 2 * a

noncomputable def cos_law_condition (a b c : ℝ) : Prop :=
  c^2 = b^2 + Real.sqrt 3 * a^2

theorem find_b_over_a {A B C a b c : ℝ} (h : sides_in_triangle A B C a b c) : b / a = Real.sqrt 2 :=
  sorry

theorem find_angle_B {A B C a b c : ℝ} (h1 : sides_in_triangle A B C a b c) (h2 : cos_law_condition a b c)
  (h3 : b / a = Real.sqrt 2) : B = Real.pi / 4 :=
  sorry

end NUMINAMATH_GPT_find_b_over_a_find_angle_B_l2040_204004
