import Mathlib

namespace NUMINAMATH_GPT_original_polygon_sides_l492_49220

noncomputable def sum_of_interior_angles (n : ℕ) : ℝ :=
(n - 2) * 180

theorem original_polygon_sides (x : ℕ) (h1 : sum_of_interior_angles (2 * x) = 2160) : x = 7 :=
by
  sorry

end NUMINAMATH_GPT_original_polygon_sides_l492_49220


namespace NUMINAMATH_GPT_car_journey_delay_l492_49264

theorem car_journey_delay (distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) (time1 : ℝ) (time2 : ℝ) (delay : ℝ) :
  distance = 225 ∧ speed1 = 60 ∧ speed2 = 50 ∧ time1 = distance / speed1 ∧ time2 = distance / speed2 ∧ 
  delay = (time2 - time1) * 60 → delay = 45 :=
by
  sorry

end NUMINAMATH_GPT_car_journey_delay_l492_49264


namespace NUMINAMATH_GPT_residue_neg_998_mod_28_l492_49294

theorem residue_neg_998_mod_28 : ∃ r : ℤ, r = -998 % 28 ∧ 0 ≤ r ∧ r < 28 ∧ r = 10 := 
by sorry

end NUMINAMATH_GPT_residue_neg_998_mod_28_l492_49294


namespace NUMINAMATH_GPT_coloring_ways_l492_49289

-- Define a factorial function
def factorial : Nat → Nat
| 0       => 1
| (n + 1) => (n + 1) * factorial n

-- Define a derangement function
def derangement : Nat → Nat
| 0       => 1
| 1       => 0
| (n + 1) => n * (derangement n + derangement (n - 1))

-- Prove the main theorem
theorem coloring_ways : 
  let six_factorial := factorial 6
  let derangement_6 := derangement 6
  let derangement_5 := derangement 5
  720 * (derangement_6 + derangement_5) = 222480 := by
    let six_factorial := 720
    let derangement_6 := derangement 6
    let derangement_5 := derangement 5
    show six_factorial * (derangement_6 + derangement_5) = 222480
    sorry

end NUMINAMATH_GPT_coloring_ways_l492_49289


namespace NUMINAMATH_GPT_shrimp_cost_per_pound_l492_49284

theorem shrimp_cost_per_pound 
    (shrimp_per_guest : ℕ) 
    (num_guests : ℕ) 
    (shrimp_per_pound : ℕ) 
    (total_cost : ℝ)
    (H1 : shrimp_per_guest = 5)
    (H2 : num_guests = 40)
    (H3 : shrimp_per_pound = 20)
    (H4 : total_cost = 170) : 
    (total_cost / ((num_guests * shrimp_per_guest) / shrimp_per_pound) = 17) :=
by
    sorry

end NUMINAMATH_GPT_shrimp_cost_per_pound_l492_49284


namespace NUMINAMATH_GPT_good_carrots_l492_49227

theorem good_carrots (Faye_picked : ℕ) (Mom_picked : ℕ) (bad_carrots : ℕ)
    (total_carrots : Faye_picked + Mom_picked = 28)
    (bad_carrots_count : bad_carrots = 16) : 
    28 - bad_carrots = 12 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_good_carrots_l492_49227


namespace NUMINAMATH_GPT_students_in_class_l492_49209

theorem students_in_class (b n : ℕ) :
  6 * (b + 1) = n ∧ 9 * (b - 1) = n → n = 36 :=
by
  sorry

end NUMINAMATH_GPT_students_in_class_l492_49209


namespace NUMINAMATH_GPT_interval_intersection_l492_49251

theorem interval_intersection :
  {x : ℝ | 1 < 3 * x ∧ 3 * x < 2 ∧ 1 < 5 * x ∧ 5 * x < 2} =
  {x : ℝ | (1 / 3 : ℝ) < x ∧ x < (2 / 5 : ℝ)} :=
by
  -- Need a proof here
  sorry

end NUMINAMATH_GPT_interval_intersection_l492_49251


namespace NUMINAMATH_GPT_nth_position_equation_l492_49263

theorem nth_position_equation (n : ℕ) (h : n > 0) : 9 * (n - 1) + n = 10 * n - 9 := by
  sorry

end NUMINAMATH_GPT_nth_position_equation_l492_49263


namespace NUMINAMATH_GPT_exist_m_eq_l492_49212

theorem exist_m_eq (n b : ℕ) (p : ℕ) (hp_prime : Nat.Prime p) (hp_odd : p % 2 = 1) (hn_zero : n ≠ 0) (hb_zero : b ≠ 0)
  (h_div : p ∣ (b^(2^n) + 1)) :
  ∃ m : ℕ, p = 2^(n+1) * m + 1 :=
by
  sorry

end NUMINAMATH_GPT_exist_m_eq_l492_49212


namespace NUMINAMATH_GPT_moe_pie_share_l492_49297

theorem moe_pie_share
  (leftover_pie : ℚ)
  (num_people : ℕ)
  (H_leftover : leftover_pie = 5 / 8)
  (H_people : num_people = 4) :
  (leftover_pie / num_people = 5 / 32) :=
by
  sorry

end NUMINAMATH_GPT_moe_pie_share_l492_49297


namespace NUMINAMATH_GPT_combined_molecular_weight_l492_49223

-- Define atomic masses of elements
def atomic_mass_Ca : Float := 40.08
def atomic_mass_Br : Float := 79.904
def atomic_mass_Sr : Float := 87.62
def atomic_mass_Cl : Float := 35.453

-- Define number of moles for each compound
def moles_CaBr2 : Float := 4
def moles_SrCl2 : Float := 3

-- Define molar masses of compounds
def molar_mass_CaBr2 : Float := atomic_mass_Ca + 2 * atomic_mass_Br
def molar_mass_SrCl2 : Float := atomic_mass_Sr + 2 * atomic_mass_Cl

-- Define total mass calculation for each compound
def total_mass_CaBr2 : Float := moles_CaBr2 * molar_mass_CaBr2
def total_mass_SrCl2 : Float := moles_SrCl2 * molar_mass_SrCl2

-- Prove the combined molecular weight
theorem combined_molecular_weight :
  total_mass_CaBr2 + total_mass_SrCl2 = 1275.13 :=
  by
    -- The proof will be here
    sorry

end NUMINAMATH_GPT_combined_molecular_weight_l492_49223


namespace NUMINAMATH_GPT_min_le_max_condition_l492_49290

variable (a b c : ℝ)

theorem min_le_max_condition
  (h1 : a ≠ 0)
  (h2 : ∃ t : ℝ, 2*a*t^2 + b*t + c = 0 ∧ |t| ≤ 1) :
  min c (a + c + 1) ≤ max (|b - a + 1|) (|b + a - 1|) :=
sorry

end NUMINAMATH_GPT_min_le_max_condition_l492_49290


namespace NUMINAMATH_GPT_find_ordered_pair_l492_49262

variables {A B Q : Type} -- Points A, B, Q
variables [AddCommGroup A] [AddCommGroup B] [AddCommGroup Q]
variables {a b q : A} -- Vectors at points A, B, Q
variables (r : ℝ) -- Ratio constant

-- Define the conditions from the original problem
def ratio_aq_qb (A B Q : Type) [AddCommGroup A] [AddCommGroup B] [AddCommGroup Q] (a b q : A) (r : ℝ) :=
  r = 7 / 2

-- Define the goal theorem using the conditions above
theorem find_ordered_pair (h : ratio_aq_qb A B Q a b q r) : 
  q = (7 / 9) • a + (2 / 9) • b :=
sorry

end NUMINAMATH_GPT_find_ordered_pair_l492_49262


namespace NUMINAMATH_GPT_part_a_part_b_l492_49206

-- Part (a): Prove that \( 2^n - 1 \) is divisible by 7 if and only if \( 3 \mid n \).
theorem part_a (n : ℕ) : 7 ∣ (2^n - 1) ↔ 3 ∣ n := sorry

-- Part (b): Prove that \( 2^n + 1 \) is not divisible by 7 for all natural numbers \( n \).
theorem part_b (n : ℕ) : ¬ (7 ∣ (2^n + 1)) := sorry

end NUMINAMATH_GPT_part_a_part_b_l492_49206


namespace NUMINAMATH_GPT_g_value_range_l492_49214

noncomputable def g (x y z : ℝ) : ℝ :=
  (x^2 / (x^2 + y^2)) + (y^2 / (y^2 + z^2)) + (z^2 / (z^2 + x^2))

theorem g_value_range (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) : 
  (3/2 : ℝ) ≤ g x y z ∧ g x y z ≤ (3 : ℝ) / 2 := 
sorry

end NUMINAMATH_GPT_g_value_range_l492_49214


namespace NUMINAMATH_GPT_first_company_managers_percentage_l492_49225

-- Definitions from the conditions
variable (F M : ℝ) -- total workforce of first company and merged company
variable (x : ℝ) -- percentage of managers in the first company
variable (cond1 : 0.25 * M = F) -- 25% of merged company's workforce originated from the first company
variable (cond2 : 0.25 * M / M = 0.25) -- resulting merged company's workforce consists of 25% managers

-- The statement to prove
theorem first_company_managers_percentage : x = 25 :=
by
  sorry

end NUMINAMATH_GPT_first_company_managers_percentage_l492_49225


namespace NUMINAMATH_GPT_arithmetic_progression_complete_iff_divides_l492_49202

-- Definitions from the conditions
def complete_sequence (s : ℕ → ℤ) : Prop :=
  (∀ n : ℕ, s n ≠ 0) ∧ (∀ m : ℤ, m ≠ 0 → ∃ n : ℕ, s n = m)

-- Arithmetic progression definition
def arithmetic_progression (a r : ℤ) (n : ℕ) : ℤ :=
  a + n * r

-- Lean theorem statement
theorem arithmetic_progression_complete_iff_divides (a r : ℤ) :
  (complete_sequence (arithmetic_progression a r)) ↔ (r ∣ a) := by
  sorry

end NUMINAMATH_GPT_arithmetic_progression_complete_iff_divides_l492_49202


namespace NUMINAMATH_GPT_range_a_and_inequality_l492_49281

noncomputable def f (x a : ℝ) : ℝ := x^2 - a * Real.log (x + 2)
noncomputable def f' (x a : ℝ) : ℝ := 2 * x - a / (x + 2)

theorem range_a_and_inequality (a x1 x2 : ℝ) (h_deriv: ∀ (x : ℝ), f' x a = 0 → x = x1 ∨ x = x2) (h_lt: x1 < x2) (h_extreme: f (x1) a = f (x2) a):
  (-2 < a ∧ a < 0) → 
  (f (x1) a / x2 + 1 < 0) :=
by
  sorry

end NUMINAMATH_GPT_range_a_and_inequality_l492_49281


namespace NUMINAMATH_GPT_prime_roots_eq_l492_49259

theorem prime_roots_eq (n : ℕ) (hn : 0 < n) :
  (∃ (x1 x2 : ℕ), Prime x1 ∧ Prime x2 ∧ 2*x1^2 - 8*n*x1 + 10*x1 - n^2 + 35*n - 76 = 0 ∧ 
                    2*x2^2 - 8*n*x2 + 10*x2 - n^2 + 35*n - 76 = 0 ∧ x1 ≠ x2 ∧ x1 < x2) →
  n = 3 ∧ ∃ x1 x2 : ℕ, x1 = 2 ∧ x2 = 5 ∧ Prime x1 ∧ Prime x2 ∧
    2*x1^2 - 8*n*x1 + 10*x1 - n^2 + 35*n - 76 = 0 ∧
    2*x2^2 - 8*n*x2 + 10*x2 - n^2 + 35*n - 76 = 0 := 
by
  sorry

end NUMINAMATH_GPT_prime_roots_eq_l492_49259


namespace NUMINAMATH_GPT_grandmother_current_age_l492_49230

theorem grandmother_current_age (yoojung_age_current yoojung_age_future grandmother_age_future : ℕ)
    (h1 : yoojung_age_current = 5)
    (h2 : yoojung_age_future = 10)
    (h3 : grandmother_age_future = 60) :
    grandmother_age_future - (yoojung_age_future - yoojung_age_current) = 55 :=
by 
  sorry

end NUMINAMATH_GPT_grandmother_current_age_l492_49230


namespace NUMINAMATH_GPT_arrange_abc_l492_49286

open Real

noncomputable def a := log 4 / log 5
noncomputable def b := (log 3 / log 5)^2
noncomputable def c := 1 / (log 4 / log 5)

theorem arrange_abc : b < a ∧ a < c :=
by
  -- Mathematical translations as Lean proof obligations
  have a_lt_one : a < 1 := by sorry
  have c_gt_one : c > 1 := by sorry
  have b_lt_a : b < a := by sorry
  have a_lt_c : a < c := by sorry
  exact ⟨b_lt_a, a_lt_c⟩

end NUMINAMATH_GPT_arrange_abc_l492_49286


namespace NUMINAMATH_GPT_g_six_l492_49208

theorem g_six (g : ℝ → ℝ) (H1 : ∀ x y : ℝ, g (x + y) = g x * g y) (H2 : g 2 = 4) : g 6 = 64 :=
by
  sorry

end NUMINAMATH_GPT_g_six_l492_49208


namespace NUMINAMATH_GPT_blue_shoes_in_warehouse_l492_49276

theorem blue_shoes_in_warehouse (total blue purple green : ℕ) (h1 : total = 1250) (h2 : green = purple) (h3 : purple = 355) :
    blue = total - (green + purple) := by
  sorry

end NUMINAMATH_GPT_blue_shoes_in_warehouse_l492_49276


namespace NUMINAMATH_GPT_number_of_5_letter_words_with_at_least_one_vowel_l492_49256

theorem number_of_5_letter_words_with_at_least_one_vowel :
  let letters := ['A', 'B', 'C', 'D', 'E', 'F', 'G']
  let vowels := ['A', 'E']
  ∃ n : ℕ, n = 7^5 - 5^5 ∧ n = 13682 :=
by
  sorry

end NUMINAMATH_GPT_number_of_5_letter_words_with_at_least_one_vowel_l492_49256


namespace NUMINAMATH_GPT_ratio_of_democrats_l492_49221

theorem ratio_of_democrats (F M : ℕ) (h1 : F + M = 750) (h2 : (1/2 : ℚ) * F = 125) (h3 : (1/4 : ℚ) * M = 125) :
  (125 + 125 : ℚ) / 750 = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_ratio_of_democrats_l492_49221


namespace NUMINAMATH_GPT_solve_equation_l492_49228

theorem solve_equation :
  ∀ x : ℝ, (1 / 7 + 7 / x = 15 / x + 1 / 15) → x = 105 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_solve_equation_l492_49228


namespace NUMINAMATH_GPT_largest_of_four_integers_l492_49218

theorem largest_of_four_integers (n : ℤ) (h1 : n % 2 = 0) (h2 : (n+2) % 2 = 0) (h3 : (n+4) % 2 = 0) (h4 : (n+6) % 2 = 0) (h : n * (n+2) * (n+4) * (n+6) = 6720) : max (max (max n (n+2)) (n+4)) (n+6) = 14 := 
sorry

end NUMINAMATH_GPT_largest_of_four_integers_l492_49218


namespace NUMINAMATH_GPT_quadratic_factorization_l492_49255

theorem quadratic_factorization (a b : ℕ) (h1 : x^2 - 18 * x + 72 = (x - a) * (x - b))
  (h2 : a > b) : 2 * b - a = 0 :=
sorry

end NUMINAMATH_GPT_quadratic_factorization_l492_49255


namespace NUMINAMATH_GPT_probability_not_snow_l492_49233

theorem probability_not_snow (P_snow : ℚ) (h : P_snow = 2 / 5) : (1 - P_snow = 3 / 5) :=
by 
  rw [h]
  norm_num

end NUMINAMATH_GPT_probability_not_snow_l492_49233


namespace NUMINAMATH_GPT_completing_the_square_correct_l492_49215

theorem completing_the_square_correct :
  ∀ x : ℝ, (x^2 - 4*x + 1 = 0) → ((x - 2)^2 = 3) :=
by
  intro x h
  sorry

end NUMINAMATH_GPT_completing_the_square_correct_l492_49215


namespace NUMINAMATH_GPT_bill_soaking_time_l492_49235

theorem bill_soaking_time 
  (G M : ℕ) 
  (h₁ : M = G + 7) 
  (h₂ : 3 * G + M = 19) : 
  G = 3 := 
by {
  sorry
}

end NUMINAMATH_GPT_bill_soaking_time_l492_49235


namespace NUMINAMATH_GPT_find_y_when_x_is_8_l492_49248

theorem find_y_when_x_is_8 (x y : ℕ) (k : ℕ) (h1 : x + y = 36) (h2 : x - y = 12) (h3 : x * y = k) (h4 : k = 288) : y = 36 :=
by
  -- Given the conditions
  sorry

end NUMINAMATH_GPT_find_y_when_x_is_8_l492_49248


namespace NUMINAMATH_GPT_express_set_M_l492_49216

def is_divisor (a b : ℤ) : Prop := ∃ k : ℤ, a = b * k

def M : Set ℤ := {m | is_divisor 10 (m + 1)}

theorem express_set_M :
  M = {-11, -6, -3, -2, 0, 1, 4, 9} :=
by
  sorry

end NUMINAMATH_GPT_express_set_M_l492_49216


namespace NUMINAMATH_GPT_stratified_sampling_elderly_count_l492_49274

-- Definitions of conditions
def elderly := 30
def middleAged := 90
def young := 60
def totalPeople := elderly + middleAged + young
def sampleSize := 36
def samplingFraction := sampleSize / totalPeople
def expectedElderlySample := elderly * samplingFraction

-- The theorem we want to prove
theorem stratified_sampling_elderly_count : expectedElderlySample = 6 := 
by 
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_stratified_sampling_elderly_count_l492_49274


namespace NUMINAMATH_GPT_cone_volume_l492_49295

theorem cone_volume (V_cylinder V_frustum V_cone : ℝ)
  (h₁ : V_cylinder = 9)
  (h₂ : V_frustum = 63) :
  V_cone = 64 :=
sorry

end NUMINAMATH_GPT_cone_volume_l492_49295


namespace NUMINAMATH_GPT_cylinder_surface_area_correct_l492_49252

noncomputable def cylinder_surface_area :=
  let r := 8   -- radius in cm
  let h := 10  -- height in cm
  let arc_angle := 90 -- degrees
  let x := 40
  let y := -40
  let z := 2
  x + y + z

theorem cylinder_surface_area_correct : cylinder_surface_area = 2 := by
  sorry

end NUMINAMATH_GPT_cylinder_surface_area_correct_l492_49252


namespace NUMINAMATH_GPT_nailcutter_sound_count_l492_49271

-- Definitions based on conditions
def nails_per_person : ℕ := 20
def number_of_customers : ℕ := 3
def sound_per_nail : ℕ := 1

-- The statement to prove 
theorem nailcutter_sound_count :
  (nails_per_person * number_of_customers * sound_per_nail) = 60 := by
  sorry

end NUMINAMATH_GPT_nailcutter_sound_count_l492_49271


namespace NUMINAMATH_GPT_residue_625_mod_17_l492_49207

theorem residue_625_mod_17 : 625 % 17 = 13 :=
by
  sorry

end NUMINAMATH_GPT_residue_625_mod_17_l492_49207


namespace NUMINAMATH_GPT_complement_A_in_U_l492_49268

noncomputable def U : Set ℕ := {0, 1, 2}
noncomputable def A : Set ℕ := {x | x^2 - x = 0}
noncomputable def complement_U (A : Set ℕ) : Set ℕ := U \ A

theorem complement_A_in_U : 
  complement_U {x | x^2 - x = 0} = {2} := 
sorry

end NUMINAMATH_GPT_complement_A_in_U_l492_49268


namespace NUMINAMATH_GPT_parabola_opening_downwards_l492_49211

theorem parabola_opening_downwards (a : ℝ) :
  (∀ x, 0 < x ∧ x < 3 → ax^2 - 2 * a * x + 3 > 0) → -1 < a ∧ a < 0 :=
by 
  intro h
  sorry

end NUMINAMATH_GPT_parabola_opening_downwards_l492_49211


namespace NUMINAMATH_GPT_find_x_l492_49243

theorem find_x (x : ℤ) (h1 : 5 < x) (h2 : x < 21) (h3 : 7 < x) (h4 : x < 18) (h5 : 2 < x) (h6 : x < 13) (h7 : 9 < x) (h8 : x < 12) (h9 : x < 12) :
  x = 10 :=
sorry

end NUMINAMATH_GPT_find_x_l492_49243


namespace NUMINAMATH_GPT_min_value_of_exp_l492_49217

noncomputable def minimum_value_of_expression (a b : ℝ) : ℝ :=
  (1 - a)^2 + (1 - 2 * b)^2 + (a - 2 * b)^2

theorem min_value_of_exp (a b : ℝ) (h : a^2 ≥ 8 * b) : minimum_value_of_expression a b = 9 / 8 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_exp_l492_49217


namespace NUMINAMATH_GPT_abc_value_l492_49280

theorem abc_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (h1 : a * (b + c) = 171) 
    (h2 : b * (c + a) = 180) 
    (h3 : c * (a + b) = 189) :
    a * b * c = 270 :=
by
  -- Place proofs here
  sorry

end NUMINAMATH_GPT_abc_value_l492_49280


namespace NUMINAMATH_GPT_johny_travelled_South_distance_l492_49213

theorem johny_travelled_South_distance :
  ∃ S : ℝ, S + (S + 20) + 2 * (S + 20) = 220 ∧ S = 40 :=
by
  sorry

end NUMINAMATH_GPT_johny_travelled_South_distance_l492_49213


namespace NUMINAMATH_GPT_problem_inequality_l492_49288

theorem problem_inequality (k m n : ℕ) (hk1 : 1 < k) (hkm : k ≤ m) (hmn : m < n) :
  (1 + m) ^ 2 > (1 + n) ^ m :=
  sorry

end NUMINAMATH_GPT_problem_inequality_l492_49288


namespace NUMINAMATH_GPT_arithmetic_sequence_count_l492_49296

noncomputable def count_arithmetic_triplets : ℕ := 17

theorem arithmetic_sequence_count :
  ∃ S : Finset (Finset ℕ), 
    (∀ s ∈ S, s.card = 3 ∧ (∃ d, ∀ x ∈ s, ∀ y ∈ s, ∀ z ∈ s, (x ≠ y ∧ y ≠ z ∧ x ≠ z) → ((x = y + d ∨ x = z + d ∨ y = z + d) ∧ x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x ≤ 9 ∧ y ≤ 9 ∧ z ≤ 9))) ∧ 
    S.card = count_arithmetic_triplets :=
by
  -- placeholder for proof
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_count_l492_49296


namespace NUMINAMATH_GPT_exists_five_integers_l492_49249

theorem exists_five_integers :
  ∃ (a b c d e : ℤ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
    c ≠ d ∧ c ≠ e ∧ 
    d ≠ e ∧
    ∃ (k1 k2 k3 k4 k5 : ℕ), 
      k1^2 = (a + b + c + d) ∧ 
      k2^2 = (a + b + c + e) ∧ 
      k3^2 = (a + b + d + e) ∧ 
      k4^2 = (a + c + d + e) ∧ 
      k5^2 = (b + c + d + e) := 
sorry

end NUMINAMATH_GPT_exists_five_integers_l492_49249


namespace NUMINAMATH_GPT_garden_area_l492_49210

theorem garden_area (length perimeter : ℝ) (length_50 : 50 * length = 1500) (perimeter_20 : 20 * perimeter = 1500) (rectangular : perimeter = 2 * length + 2 * (perimeter / 2 - length)) :
  length * (perimeter / 2 - length) = 225 := 
by
  sorry

end NUMINAMATH_GPT_garden_area_l492_49210


namespace NUMINAMATH_GPT_find_larger_number_l492_49244

theorem find_larger_number (a b : ℤ) (h1 : a + b = 27) (h2 : a - b = 5) : a = 16 := by
  sorry

end NUMINAMATH_GPT_find_larger_number_l492_49244


namespace NUMINAMATH_GPT_quadratic_eq_solutions_l492_49278

theorem quadratic_eq_solutions (x : ℝ) : x * (x + 1) = 3 * (x + 1) ↔ x = -1 ∨ x = 3 := by
  sorry

end NUMINAMATH_GPT_quadratic_eq_solutions_l492_49278


namespace NUMINAMATH_GPT_incorrect_statement_is_B_l492_49237

-- Define the conditions
def genotype_AaBb_meiosis_results (sperm_genotypes : List String) : Prop :=
  sperm_genotypes = ["AB", "Ab", "aB", "ab"]

def spermatogonial_cell_AaXbY (malformed_sperm_genotype : String) (other_sperm_genotypes : List String) : Prop :=
  malformed_sperm_genotype = "AAaY" ∧ other_sperm_genotypes = ["aY", "X^b", "X^b"]

def spermatogonial_secondary_spermatocyte_Y_chromosomes (contains_two_Y : Bool) : Prop :=
  ¬ contains_two_Y

def female_animal_meiosis (primary_oocyte_alleles : Nat) (max_oocyte_b_alleles : Nat) : Prop :=
  primary_oocyte_alleles = 10 ∧ max_oocyte_b_alleles ≤ 5

-- The main statement that needs to be proved
theorem incorrect_statement_is_B :
  ∃ (sperm_genotypes : List String) 
    (malformed_sperm_genotype : String) 
    (other_sperm_genotypes : List String) 
    (contains_two_Y : Bool) 
    (primary_oocyte_alleles max_oocyte_b_alleles : Nat),
    genotype_AaBb_meiosis_results sperm_genotypes ∧ 
    spermatogonial_cell_AaXbY malformed_sperm_genotype other_sperm_genotypes ∧ 
    spermatogonial_secondary_spermatocyte_Y_chromosomes contains_two_Y ∧ 
    female_animal_meiosis primary_oocyte_alleles max_oocyte_b_alleles 
    ∧ (malformed_sperm_genotype = "AAaY" → false) := 
sorry

end NUMINAMATH_GPT_incorrect_statement_is_B_l492_49237


namespace NUMINAMATH_GPT_cubic_inequality_l492_49265

theorem cubic_inequality (a b : ℝ) : a > b → a^3 > b^3 :=
sorry

end NUMINAMATH_GPT_cubic_inequality_l492_49265


namespace NUMINAMATH_GPT_r_exceeds_s_by_two_l492_49204

theorem r_exceeds_s_by_two (x y r s : ℝ) (h1 : 3 * x + 2 * y = 16) (h2 : 5 * x + 3 * y = 26)
  (hr : r = x) (hs : s = y) : r - s = 2 :=
by
  sorry

end NUMINAMATH_GPT_r_exceeds_s_by_two_l492_49204


namespace NUMINAMATH_GPT_find_number_l492_49266

theorem find_number (x : ℝ) (h : 3 * (2 * x + 5) = 129) : x = 19 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l492_49266


namespace NUMINAMATH_GPT_add_fractions_l492_49291

theorem add_fractions :
  (11 / 12) + (7 / 8) + (3 / 4) = 61 / 24 :=
by
  sorry

end NUMINAMATH_GPT_add_fractions_l492_49291


namespace NUMINAMATH_GPT_notebook_area_l492_49253

variable (w h : ℝ)

def width_to_height_ratio (w h : ℝ) : Prop := w / h = 7 / 5
def perimeter (w h : ℝ) : Prop := 2 * w + 2 * h = 48
def area (w h : ℝ) : ℝ := w * h

theorem notebook_area (w h : ℝ) (ratio : width_to_height_ratio w h) (peri : perimeter w h) :
  area w h = 140 :=
by
  sorry

end NUMINAMATH_GPT_notebook_area_l492_49253


namespace NUMINAMATH_GPT_initially_planned_days_l492_49261

theorem initially_planned_days (D : ℕ) (h1 : 6 * 3 + 10 * 3 = 6 * D) : D = 8 := by
  sorry

end NUMINAMATH_GPT_initially_planned_days_l492_49261


namespace NUMINAMATH_GPT_minimum_positive_temperatures_announced_l492_49269

theorem minimum_positive_temperatures_announced (x y : ℕ) :
  x * (x - 1) = 110 →
  y * (y - 1) + (x - y) * (x - y - 1) = 54 →
  (∀ z : ℕ, z * (z - 1) + (x - z) * (x - z - 1) = 54 → y ≤ z) →
  y = 4 :=
by
  sorry

end NUMINAMATH_GPT_minimum_positive_temperatures_announced_l492_49269


namespace NUMINAMATH_GPT_quadruple_dimensions_increase_volume_l492_49245

theorem quadruple_dimensions_increase_volume 
  (V_original : ℝ) (quad_factor : ℝ)
  (initial_volume : V_original = 5)
  (quad_factor_val : quad_factor = 4) :
  V_original * (quad_factor ^ 3) = 320 := 
by 
  -- Introduce necessary variables and conditions
  let V_modified := V_original * (quad_factor ^ 3)
  
  -- Assert the calculations based on the given conditions
  have initial : V_original = 5 := initial_volume
  have quad : quad_factor = 4 := quad_factor_val
  
  -- Skip the detailed proof with sorry
  sorry


end NUMINAMATH_GPT_quadruple_dimensions_increase_volume_l492_49245


namespace NUMINAMATH_GPT_isosceles_triangle_largest_angle_l492_49236

theorem isosceles_triangle_largest_angle (A B C : ℝ) (h_isosceles : A = B) (h_angles : A = 60 ∧ B = 60) :
  max A (max B C) = 60 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_largest_angle_l492_49236


namespace NUMINAMATH_GPT_yellow_paint_percentage_l492_49203

theorem yellow_paint_percentage 
  (total_gallons_mixture : ℝ)
  (light_green_paint_gallons : ℝ)
  (dark_green_paint_gallons : ℝ)
  (dark_green_paint_percentage : ℝ)
  (mixture_percentage : ℝ)
  (X : ℝ) 
  (h_total_gallons : total_gallons_mixture = light_green_paint_gallons + dark_green_paint_gallons)
  (h_dark_green_paint_yellow_amount : dark_green_paint_gallons * dark_green_paint_percentage = 1.66666666667 * 0.4)
  (h_mixture_yellow_amount : total_gallons_mixture * mixture_percentage = 5 * X + 1.66666666667 * 0.4) :
  X = 0.2 :=
by
  sorry

end NUMINAMATH_GPT_yellow_paint_percentage_l492_49203


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l492_49246

-- Problem 1
theorem problem1 :
  1 - 1^2022 + ((-1/2)^2) * (-2)^3 * (-2)^2 - |Real.pi - 3.14|^0 = -10 :=
by sorry

-- Problem 2
variables (a b : ℝ)

theorem problem2 :
  a^3 * (-b^3)^2 + (-2 * a * b)^3 = a^3 * b^6 - 8 * a^3 * b^3 :=
by sorry

-- Problem 3
theorem problem3 (a b : ℝ) :
  (2 * a^3 * b^2 - 3 * a^2 * b - 4 * a) * 2 * b = 4 * a^3 * b^3 - 6 * a^2 * b^2 - 8 * a * b :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l492_49246


namespace NUMINAMATH_GPT_fraction_to_decimal_l492_49222

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end NUMINAMATH_GPT_fraction_to_decimal_l492_49222


namespace NUMINAMATH_GPT_sin_theta_plus_45_l492_49283

-- Statement of the problem in Lean 4

theorem sin_theta_plus_45 (θ : ℝ) (h : 0 < θ ∧ θ < π / 2) (sin_θ_eq : Real.sin θ = 3 / 5) :
  Real.sin (θ + π / 4) = 7 * Real.sqrt 2 / 10 :=
sorry

end NUMINAMATH_GPT_sin_theta_plus_45_l492_49283


namespace NUMINAMATH_GPT_total_area_painted_correct_l492_49239

-- Defining the properties of the shed
def shed_w := 12  -- width in yards
def shed_l := 15  -- length in yards
def shed_h := 7   -- height in yards

-- Calculating area to be painted
def wall_area_1 := 2 * (shed_w * shed_h)
def wall_area_2 := 2 * (shed_l * shed_h)
def floor_ceiling_area := 2 * (shed_w * shed_l)
def total_painted_area := wall_area_1 + wall_area_2 + floor_ceiling_area

-- The theorem to be proved
theorem total_area_painted_correct :
  total_painted_area = 738 := by
  sorry

end NUMINAMATH_GPT_total_area_painted_correct_l492_49239


namespace NUMINAMATH_GPT_remainders_equal_l492_49277

theorem remainders_equal (P P' D R k s s' : ℕ) (h1 : P > P') 
  (h2 : P % D = 2 * R) (h3 : P' % D = R) (h4 : R < D) :
  (k * (P + P')) % D = s → (k * (2 * R + R)) % D = s' → s = s' :=
by
  sorry

end NUMINAMATH_GPT_remainders_equal_l492_49277


namespace NUMINAMATH_GPT_ordering_abc_l492_49270

noncomputable def a : ℝ := Real.sqrt 1.01
noncomputable def b : ℝ := Real.exp 0.01 / 1.01
noncomputable def c : ℝ := Real.log (1.01 * Real.exp 1)

theorem ordering_abc : b < a ∧ a < c := by
  -- Proof of the theorem goes here
  sorry

end NUMINAMATH_GPT_ordering_abc_l492_49270


namespace NUMINAMATH_GPT_captain_age_l492_49282

theorem captain_age
  (C W : ℕ)
  (avg_team_age : ℤ)
  (avg_remaining_players_age : ℤ)
  (total_team_age : ℤ)
  (total_remaining_players_age : ℤ)
  (remaining_players_count : ℕ)
  (total_team_count : ℕ)
  (total_team_age_eq : total_team_age = total_team_count * avg_team_age)
  (remaining_players_age_eq : total_remaining_players_age = remaining_players_count * avg_remaining_players_age)
  (total_team_eq : total_team_count = 11)
  (remaining_players_eq : remaining_players_count = 9)
  (avg_team_age_eq : avg_team_age = 23)
  (avg_remaining_players_age_eq : avg_remaining_players_age = avg_team_age - 1)
  (age_diff : W = C + 5)
  (players_age_sum : total_team_age = total_remaining_players_age + C + W) :
  C = 25 :=
by
  sorry

end NUMINAMATH_GPT_captain_age_l492_49282


namespace NUMINAMATH_GPT_roots_n_not_divisible_by_5_for_any_n_l492_49293

theorem roots_n_not_divisible_by_5_for_any_n (x1 x2 : ℝ) (n : ℕ)
  (hx : x1^2 - 6 * x1 + 1 = 0)
  (hy : x2^2 - 6 * x2 + 1 = 0)
  : ¬(∃ (k : ℕ), (x1^k + x2^k) % 5 = 0) :=
sorry

end NUMINAMATH_GPT_roots_n_not_divisible_by_5_for_any_n_l492_49293


namespace NUMINAMATH_GPT_soccer_camp_ratio_l492_49231

theorem soccer_camp_ratio :
  let total_kids := 2000
  let half_total := total_kids / 2
  let afternoon_camp := 750
  let morning_camp := half_total - afternoon_camp
  half_total ≠ 0 → 
  (morning_camp / half_total) = 1 / 4 := by
  sorry

end NUMINAMATH_GPT_soccer_camp_ratio_l492_49231


namespace NUMINAMATH_GPT_triangle_area_45_45_90_l492_49232

/--
A right triangle has one angle of 45 degrees, and its hypotenuse measures 10√2 inches.
Prove that the area of the triangle is 50 square inches.
-/
theorem triangle_area_45_45_90 {x : ℝ} (h1 : 0 < x) (h2 : x * Real.sqrt 2 = 10 * Real.sqrt 2) : 
  (1 / 2) * x * x = 50 :=
sorry

end NUMINAMATH_GPT_triangle_area_45_45_90_l492_49232


namespace NUMINAMATH_GPT_smallest_n_for_property_l492_49292

theorem smallest_n_for_property (n x : ℕ) (d : ℕ) (c : ℕ) 
  (hx : x = 10 * c + d) 
  (hx_prop : 10^(n-1) * d + c = 2 * x) :
  n = 18 := 
sorry

end NUMINAMATH_GPT_smallest_n_for_property_l492_49292


namespace NUMINAMATH_GPT_complements_intersection_l492_49240

open Set

noncomputable def U : Set ℕ := { x | x ≤ 5 }
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {1, 4}

theorem complements_intersection :
  (U \ A) ∩ (U \ B) = {0, 5} :=
by
  sorry

end NUMINAMATH_GPT_complements_intersection_l492_49240


namespace NUMINAMATH_GPT_compare_magnitude_l492_49257

theorem compare_magnitude (a b : ℝ) (h : a ≠ 1) : a^2 + b^2 > 2 * (a - b - 1) :=
by
  sorry

end NUMINAMATH_GPT_compare_magnitude_l492_49257


namespace NUMINAMATH_GPT_proposition_relation_l492_49205

theorem proposition_relation :
  (∀ (x : ℝ), x < 3 → x < 5) ↔ (∀ (x : ℝ), x ≥ 5 → x ≥ 3) :=
by
  sorry

end NUMINAMATH_GPT_proposition_relation_l492_49205


namespace NUMINAMATH_GPT_correct_location_l492_49241

variable (A B C D : Prop)

axiom student_A_statement : ¬ A ∧ B
axiom student_B_statement : ¬ B ∧ C
axiom student_C_statement : ¬ B ∧ ¬ D
axiom ms_Hu_response : 
  ( (¬ A ∧ B = true) ∨ (¬ B ∧ C = true) ∨ (¬ B ∧ ¬ D = true) ) ∧ 
  ( (¬ A ∧ B = false) ∨ (¬ B ∧ C = false) ∨ (¬ B ∧ ¬ D = false) = false ) ∧ 
  ( (¬ A ∧ B ∨ ¬ B ∧ C ∨ ¬ B ∧ ¬ D) -> false )

theorem correct_location : B ∨ A := 
sorry

end NUMINAMATH_GPT_correct_location_l492_49241


namespace NUMINAMATH_GPT_number_of_correct_conclusions_l492_49224

noncomputable def A (x : ℝ) : ℝ := 2 * x^2
noncomputable def B (x : ℝ) : ℝ := x + 1
noncomputable def C (x : ℝ) : ℝ := -2 * x
noncomputable def D (y : ℝ) : ℝ := y^2
noncomputable def E (x y : ℝ) : ℝ := 2 * x - y

def conclusion1 (y : ℤ) : Prop := 
  0 < ((B (0 : ℝ)) * (C (0 : ℝ)) + A (0 : ℝ) + D y + E (0) (y : ℝ))

def conclusion2 : Prop := 
  ∃ (x y : ℝ), A x + D y + 2 * E x y = -2

def M (A B C : ℝ → ℝ) (x m : ℝ) : ℝ :=
  3 * (A x - B x) + m * B x * C x

def linear_term_exists (m : ℝ) : Prop :=
  (0 : ℝ) ≠ -3 - 2 * m

def conclusion3 : Prop := 
 ∀ m : ℝ, (¬ linear_term_exists m ∧ M A B C (0 : ℝ) m > -3) 

def p (x y : ℝ) := 
  2 * (x + 1) ^ 2 + (y - 1) ^ 2 = 1

theorem number_of_correct_conclusions : Prop := 
  (¬ conclusion1 1) ∧ (conclusion2) ∧ (¬ conclusion3)

end NUMINAMATH_GPT_number_of_correct_conclusions_l492_49224


namespace NUMINAMATH_GPT_second_quarter_profit_l492_49226

theorem second_quarter_profit (q1 q3 q4 annual : ℕ) (h1 : q1 = 1500) (h2 : q3 = 3000) (h3 : q4 = 2000) (h4 : annual = 8000) :
  annual - (q1 + q3 + q4) = 1500 :=
by
  sorry

end NUMINAMATH_GPT_second_quarter_profit_l492_49226


namespace NUMINAMATH_GPT_fraction_sum_proof_l492_49287

theorem fraction_sum_proof :
    (19 / ((2^3 - 1) * (3^3 - 1)) + 
     37 / ((3^3 - 1) * (4^3 - 1)) + 
     61 / ((4^3 - 1) * (5^3 - 1)) + 
     91 / ((5^3 - 1) * (6^3 - 1))) = (208 / 1505) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_fraction_sum_proof_l492_49287


namespace NUMINAMATH_GPT_set_intersection_complement_equiv_l492_49201

open Set

variable {α : Type*}
variable {x : α}

def U : Set ℝ := univ
def M : Set ℝ := {x | 0 ≤ x}
def N : Set ℝ := {x | x^2 < 1}

theorem set_intersection_complement_equiv :
  M ∩ (U \ N) = {x | 1 ≤ x} :=
by
  sorry

end NUMINAMATH_GPT_set_intersection_complement_equiv_l492_49201


namespace NUMINAMATH_GPT_degrees_to_radians_conversion_l492_49273

theorem degrees_to_radians_conversion : (-300 : ℝ) * (Real.pi / 180) = - (5 / 3) * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_degrees_to_radians_conversion_l492_49273


namespace NUMINAMATH_GPT_petals_per_ounce_l492_49298

-- Definitions of the given conditions
def petals_per_rose : ℕ := 8
def roses_per_bush : ℕ := 12
def bushes_harvested : ℕ := 800
def bottles_produced : ℕ := 20
def ounces_per_bottle : ℕ := 12

-- Calculation of petals per bush
def petals_per_bush : ℕ := roses_per_bush * petals_per_rose

-- Calculation of total petals harvested
def total_petals_harvested : ℕ := bushes_harvested * petals_per_bush

-- Calculation of total ounces of perfume
def total_ounces_produced : ℕ := bottles_produced * ounces_per_bottle

-- Main theorem statement
theorem petals_per_ounce : total_petals_harvested / total_ounces_produced = 320 :=
by
  sorry

end NUMINAMATH_GPT_petals_per_ounce_l492_49298


namespace NUMINAMATH_GPT_bouncy_balls_per_package_l492_49250

variable (x : ℝ)

def maggie_bought_packs : ℝ := 8.0 * x
def maggie_gave_away_packs : ℝ := 4.0 * x
def maggie_bought_again_packs : ℝ := 4.0 * x
def total_kept_bouncy_balls : ℝ := 80

theorem bouncy_balls_per_package :
  (maggie_bought_packs x = total_kept_bouncy_balls) → 
  x = 10 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_bouncy_balls_per_package_l492_49250


namespace NUMINAMATH_GPT_ConeCannotHaveSquarePlanView_l492_49258

def PlanViewIsSquare (solid : Type) : Prop :=
  -- Placeholder to denote the property that the plan view of a solid is a square
  sorry

def IsCone (solid : Type) : Prop :=
  -- Placeholder to denote the property that the solid is a cone
  sorry

theorem ConeCannotHaveSquarePlanView (solid : Type) :
  (PlanViewIsSquare solid) → ¬ (IsCone solid) :=
sorry

end NUMINAMATH_GPT_ConeCannotHaveSquarePlanView_l492_49258


namespace NUMINAMATH_GPT_complement_of_A_inter_B_eq_l492_49285

noncomputable def A : Set ℝ := {x | abs (x - 1) ≤ 1}
noncomputable def B : Set ℝ := {y | ∃ x, y = -x^2 ∧ -Real.sqrt 2 ≤ x ∧ x < 1}
noncomputable def A_inter_B : Set ℝ := {x | x ∈ A ∧ x ∈ B}
noncomputable def complement_A_inter_B : Set ℝ := {x | x ∉ A_inter_B}

theorem complement_of_A_inter_B_eq :
  complement_A_inter_B = {x : ℝ | x ≠ 0} :=
  sorry

end NUMINAMATH_GPT_complement_of_A_inter_B_eq_l492_49285


namespace NUMINAMATH_GPT_geometric_sequence_product_l492_49279

theorem geometric_sequence_product (a1 a5 : ℚ) (a b c : ℚ) (q : ℚ) 
  (h1 : a1 = 8 / 3) 
  (h5 : a5 = 27 / 2)
  (h_common_ratio_pos : q = 3 / 2)
  (h_a : a = a1 * q)
  (h_b : b = a * q)
  (h_c : c = b * q)
  (h5_eq : a5 = a1 * q^4)
  (h_common_ratio_neg : q = -3 / 2 ∨ q = 3 / 2) :
  a * b * c = 216 := by
    sorry

end NUMINAMATH_GPT_geometric_sequence_product_l492_49279


namespace NUMINAMATH_GPT_find_a_l492_49267

theorem find_a (a : ℝ) :
  let A := {5}
  let B := { x : ℝ | a * x - 1 = 0 }
  A ∩ B = B ↔ (a = 0 ∨ a = 1 / 5) :=
by
  sorry

end NUMINAMATH_GPT_find_a_l492_49267


namespace NUMINAMATH_GPT_fabric_woven_in_30_days_l492_49272

theorem fabric_woven_in_30_days :
  let a1 := 5
  let d := 16 / 29
  (30 * a1 + (30 * (30 - 1) / 2) * d) = 390 :=
by
  let a1 := 5
  let d := 16 / 29
  sorry

end NUMINAMATH_GPT_fabric_woven_in_30_days_l492_49272


namespace NUMINAMATH_GPT_quadratic_complete_square_l492_49260

theorem quadratic_complete_square (b c : ℝ) (h : ∀ x : ℝ, x^2 - 24 * x + 50 = (x + b)^2 + c) : b + c = -106 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_complete_square_l492_49260


namespace NUMINAMATH_GPT_probability_is_one_over_145_l492_49219

-- Define the domain and properties
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def even (n : ℕ) : Prop :=
  n % 2 = 0

-- Total number of ways to pick 2 distinct numbers from 1 to 30
def total_ways_to_pick_two_distinct : ℕ :=
  (30 * 29) / 2

-- Calculate prime numbers between 1 and 30
def primes_from_1_to_30 : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Filter valid pairs where both numbers are prime and at least one of them is 2
def valid_pairs (l : List ℕ) : List (ℕ × ℕ) :=
  [(2, 3), (2, 5), (2, 7), (2, 11), (2, 13), (2, 17), (2, 19), (2, 23), (2, 29)]

def count_valid_pairs (l : List (ℕ × ℕ)) : ℕ :=
  l.length

-- Probability calculation
def probability_prime_and_even : ℚ :=
  count_valid_pairs (valid_pairs primes_from_1_to_30) / total_ways_to_pick_two_distinct

-- Prove that the probability is 1/145
theorem probability_is_one_over_145 : probability_prime_and_even = 1 / 145 :=
by
  sorry

end NUMINAMATH_GPT_probability_is_one_over_145_l492_49219


namespace NUMINAMATH_GPT_g_five_eq_248_l492_49229

-- We define g, and assume it meets the conditions described.
variable (g : ℤ → ℤ)

-- Condition 1: g(1) > 1
axiom g_one_gt_one : g 1 > 1

-- Condition 2: Functional equation for g
axiom g_funct_eq (x y : ℤ) : g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y

-- Condition 3: Recursive relationship for g
axiom g_recur_eq (x : ℤ) : 3 * g x = g (x + 1) + 2 * x - 1

-- Theorem we want to prove
theorem g_five_eq_248 : g 5 = 248 := by
  sorry

end NUMINAMATH_GPT_g_five_eq_248_l492_49229


namespace NUMINAMATH_GPT_adult_meals_sold_l492_49200

theorem adult_meals_sold (k a : ℕ) (h1 : 10 * a = 7 * k) (h2 : k = 70) : a = 49 :=
by
  sorry

end NUMINAMATH_GPT_adult_meals_sold_l492_49200


namespace NUMINAMATH_GPT_ellipse_conjugate_diameters_l492_49299

variable (A B C D E : ℝ)

theorem ellipse_conjugate_diameters :
  (A * E - B * D = 0) ∧ (2 * B ^ 2 + (A - C) * A = 0) :=
sorry

end NUMINAMATH_GPT_ellipse_conjugate_diameters_l492_49299


namespace NUMINAMATH_GPT_sixth_term_sequence_l492_49247

theorem sixth_term_sequence (a : ℕ → ℕ) (h₁ : a 0 = 3) (h₂ : ∀ n, a (n + 1) = (a n)^2) : 
  a 5 = 1853020188851841 := 
by {
  sorry
}

end NUMINAMATH_GPT_sixth_term_sequence_l492_49247


namespace NUMINAMATH_GPT_friends_share_difference_l492_49238

-- Define the initial conditions
def gift_cost : ℕ := 120
def initial_friends : ℕ := 10
def remaining_friends : ℕ := 6

-- Define the initial and new shares
def initial_share : ℕ := gift_cost / initial_friends
def new_share : ℕ := gift_cost / remaining_friends

-- Define the difference between the new share and the initial share
def share_difference : ℕ := new_share - initial_share

-- The theorem to be proved
theorem friends_share_difference : share_difference = 8 :=
by
  sorry

end NUMINAMATH_GPT_friends_share_difference_l492_49238


namespace NUMINAMATH_GPT_smaller_denom_is_five_l492_49242

-- Define the conditions
def num_smaller_bills : ℕ := 4
def num_ten_dollar_bills : ℕ := 8
def total_bills : ℕ := num_smaller_bills + num_ten_dollar_bills
def ten_dollar_bill_value : ℕ := 10
def total_value : ℕ := 100

-- Define the smaller denomination value
def value_smaller_denom (x : ℕ) : Prop :=
  num_smaller_bills * x + num_ten_dollar_bills * ten_dollar_bill_value = total_value

-- Prove that the value of the smaller denomination bill is 5
theorem smaller_denom_is_five : value_smaller_denom 5 :=
by
  sorry

end NUMINAMATH_GPT_smaller_denom_is_five_l492_49242


namespace NUMINAMATH_GPT_radius_of_ball_is_13_l492_49254

-- Define the conditions
def hole_radius : ℝ := 12
def hole_depth : ℝ := 8

-- The statement to prove
theorem radius_of_ball_is_13 : (∃ x : ℝ, x^2 + hole_radius^2 = (x + hole_depth)^2) → x + hole_depth = 13 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_ball_is_13_l492_49254


namespace NUMINAMATH_GPT_min_expression_value_l492_49275

noncomputable def minimum_value (a b c : ℝ) : ℝ :=
  a^2 + 4 * a * b + 8 * b^2 + 10 * b * c + 3 * c^2

theorem min_expression_value (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_abc : a * b * c = 3) :
  minimum_value a b c ≥ 27 :=
sorry

end NUMINAMATH_GPT_min_expression_value_l492_49275


namespace NUMINAMATH_GPT_geometric_sequence_ratio_l492_49234

theorem geometric_sequence_ratio
  (a₁ : ℝ) (q : ℝ) (hq : q ≠ 1)
  (S : ℕ → ℝ)
  (hS₃ : S 3 = a₁ * (1 - q^3) / (1 - q))
  (hS₆ : S 6 = a₁ * (1 - q^6) / (1 - q))
  (hS₃_val : S 3 = 2)
  (hS₆_val : S 6 = 18) :
  S 10 / S 5 = 1 + 2^(1/3) + 2^(2/3) :=
sorry

end NUMINAMATH_GPT_geometric_sequence_ratio_l492_49234
