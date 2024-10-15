import Mathlib

namespace NUMINAMATH_GPT_recipe_butter_per_cup_l1713_171309

theorem recipe_butter_per_cup (coconut_oil_to_butter_substitution : ℝ)
  (remaining_butter : ℝ)
  (planned_baking_mix : ℝ)
  (used_coconut_oil : ℝ)
  (butter_per_cup : ℝ)
  (h1 : coconut_oil_to_butter_substitution = 1)
  (h2 : remaining_butter = 4)
  (h3 : planned_baking_mix = 6)
  (h4 : used_coconut_oil = 8) :
  butter_per_cup = 4 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_recipe_butter_per_cup_l1713_171309


namespace NUMINAMATH_GPT_strictly_positive_integer_le_36_l1713_171312

theorem strictly_positive_integer_le_36 (n : ℕ) (h_pos : n > 0) :
  (∀ a : ℤ, (a % 2 = 1) → (a * a ≤ n) → (a ∣ n)) → n ≤ 36 := by
  sorry

end NUMINAMATH_GPT_strictly_positive_integer_le_36_l1713_171312


namespace NUMINAMATH_GPT_can_invent_1001_sad_stories_l1713_171316

-- Definitions
def is_natural (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 17

def is_sad_story (a b c : ℕ) : Prop :=
  ∀ x y : ℤ, a * x + b * y ≠ c

-- The Statement
theorem can_invent_1001_sad_stories :
  ∃ stories : Finset (ℕ × ℕ × ℕ),
    (∀ (a b c : ℕ), (a, b, c) ∈ stories → is_natural a ∧ is_natural b ∧ is_natural c ∧ is_sad_story a b c) ∧
    stories.card ≥ 1001 :=
by
  sorry

end NUMINAMATH_GPT_can_invent_1001_sad_stories_l1713_171316


namespace NUMINAMATH_GPT_find_a_l1713_171301

theorem find_a (a t : ℝ) 
    (h1 : (a + t) / 2 = 2020) 
    (h2 : t / 2 = 11) : 
    a = 4018 := 
by 
    sorry

end NUMINAMATH_GPT_find_a_l1713_171301


namespace NUMINAMATH_GPT_marty_combination_count_l1713_171362

theorem marty_combination_count (num_colors : ℕ) (num_methods : ℕ) 
  (h1 : num_colors = 5) (h2 : num_methods = 4) : 
  num_colors * num_methods = 20 := by
  sorry

end NUMINAMATH_GPT_marty_combination_count_l1713_171362


namespace NUMINAMATH_GPT_problem_solution_l1713_171386

theorem problem_solution (x : ℕ) (h : x = 3) : x + x * (x ^ (x + 1)) = 246 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l1713_171386


namespace NUMINAMATH_GPT_number_of_pairs_l1713_171370

theorem number_of_pairs :
  (∃ (pairs : Finset (ℕ × ℕ)), (∀ (pair : ℕ × ℕ), pair ∈ pairs → 1 ≤ pair.1 ∧ pair.1 ≤ 30 ∧ 3 ≤ pair.2 ∧ pair.2 ≤ 30 ∧ (pair.1 % pair.2 = 0) ∧ (pair.1 % (pair.2 - 2) = 0)) ∧ pairs.card = 22) := by
  sorry

end NUMINAMATH_GPT_number_of_pairs_l1713_171370


namespace NUMINAMATH_GPT_probability_five_dice_same_l1713_171325

-- Define a function that represents the probability problem
noncomputable def probability_all_dice_same : ℚ :=
  (1 / 6) * (1 / 6) * (1 / 6) * (1 / 6)

-- The main theorem to state the proof problem
theorem probability_five_dice_same : probability_all_dice_same = 1 / 1296 :=
by
  sorry

end NUMINAMATH_GPT_probability_five_dice_same_l1713_171325


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1713_171324

theorem arithmetic_sequence_sum :
  ∃ (a : ℕ → ℝ) (d : ℝ), 
  (∀ n, a n = a 0 + n * d) ∧ 
  (∃ b c, b^2 - 6*b + 5 = 0 ∧ c^2 - 6*c + 5 = 0 ∧ a 3 = b ∧ a 15 = c) →
  a 7 + a 8 + a 9 + a 10 + a 11 = 15 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1713_171324


namespace NUMINAMATH_GPT_calculate_non_defective_m3_percentage_l1713_171323

def percentage_non_defective_m3 : ℝ := 93

theorem calculate_non_defective_m3_percentage 
  (P : ℝ) -- Total number of products
  (P_pos : 0 < P) -- Total number of products is positive
  (percentage_m1 : ℝ := 0.40)
  (percentage_m2 : ℝ := 0.30)
  (percentage_m3 : ℝ := 0.30)
  (defective_m1 : ℝ := 0.03)
  (defective_m2 : ℝ := 0.01)
  (total_defective : ℝ := 0.036) :
  percentage_non_defective_m3 = 93 :=
by sorry -- The actual proof is omitted

end NUMINAMATH_GPT_calculate_non_defective_m3_percentage_l1713_171323


namespace NUMINAMATH_GPT_days_per_week_l1713_171375

def threeChildren := 3
def schoolYearWeeks := 25
def totalJuiceBoxes := 375

theorem days_per_week (d : ℕ) :
  (threeChildren * d * schoolYearWeeks = totalJuiceBoxes) → d = 5 :=
by
  sorry

end NUMINAMATH_GPT_days_per_week_l1713_171375


namespace NUMINAMATH_GPT_find_slope_of_line_l1713_171398

theorem find_slope_of_line
  (k : ℝ)
  (P : ℝ × ℝ)
  (hP : P = (3, 0))
  (C : ℝ → ℝ → Prop)
  (hC : ∀ x y, C x y ↔ x^2 - y^2 / 3 = 1)
  (A B : ℝ × ℝ)
  (hA : C A.1 A.2)
  (hB : C B.1 B.2)
  (line : ℝ → ℝ → Prop)
  (hline : ∀ x y, line x y ↔ y = k * (x - 3))
  (hintersectA : line A.1 A.2)
  (hintersectB : line B.1 B.2)
  (F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hfoci_sum : ∀ z : ℝ × ℝ, |z.1 - F.1| + |z.2 - F.2| = 16) :
  k = 3 ∨ k = -3 :=
by
  sorry

end NUMINAMATH_GPT_find_slope_of_line_l1713_171398


namespace NUMINAMATH_GPT_seats_selection_l1713_171300

theorem seats_selection (n k d : ℕ) (hn : n ≥ 4) (hk : k ≥ 2) (hd : d ≥ 2) (hkd : k * d ≤ n) :
  ∃ ways : ℕ, ways = (n / k) * Nat.choose (n - k * d + k - 1) (k - 1) :=
sorry

end NUMINAMATH_GPT_seats_selection_l1713_171300


namespace NUMINAMATH_GPT_age_of_beckett_l1713_171343

variables (B O S J : ℕ)

theorem age_of_beckett
  (h1 : B = O - 3)
  (h2 : S = O - 2)
  (h3 : J = 2 * S + 5)
  (h4 : B + O + S + J = 71) :
  B = 12 :=
by
  sorry

end NUMINAMATH_GPT_age_of_beckett_l1713_171343


namespace NUMINAMATH_GPT_candy_distribution_l1713_171361

theorem candy_distribution (n : ℕ) (h1 : n > 0) (h2 : 100 % n = 0) (h3 : 99 % n = 0) : n = 11 :=
sorry

end NUMINAMATH_GPT_candy_distribution_l1713_171361


namespace NUMINAMATH_GPT_problem_l1713_171377

theorem problem (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1 / 2) :
  (1 - x) / (1 + x) * (1 - y) / (1 + y) * (1 - z) / (1 + z) ≥ 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_problem_l1713_171377


namespace NUMINAMATH_GPT_part_a_l1713_171318

theorem part_a (n : ℕ) (hn : 0 < n) : 
  ∃ x y z : ℕ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x^(n-1) + y^n = z^(n+1) :=
sorry

end NUMINAMATH_GPT_part_a_l1713_171318


namespace NUMINAMATH_GPT_average_mark_of_excluded_students_l1713_171326

theorem average_mark_of_excluded_students 
  (N A E A_remaining : ℕ) 
  (hN : N = 25) 
  (hA : A = 80) 
  (hE : E = 5) 
  (hA_remaining : A_remaining = 95) : 
  ∃ A_excluded : ℕ, A_excluded = 20 :=
by
  -- Use the conditions in the proof.
  sorry

end NUMINAMATH_GPT_average_mark_of_excluded_students_l1713_171326


namespace NUMINAMATH_GPT_cos_neg_300_eq_half_l1713_171368

theorem cos_neg_300_eq_half : Real.cos (-300 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cos_neg_300_eq_half_l1713_171368


namespace NUMINAMATH_GPT_sum_of_last_two_digits_l1713_171392

-- Definitions based on given conditions
def six_power_twenty_five := 6^25
def fourteen_power_twenty_five := 14^25
def expression := six_power_twenty_five + fourteen_power_twenty_five
def modulo := 100

-- The statement we need to prove
theorem sum_of_last_two_digits : expression % modulo = 0 := by
  sorry

end NUMINAMATH_GPT_sum_of_last_two_digits_l1713_171392


namespace NUMINAMATH_GPT_relationship_m_n_l1713_171306

variable (a b : ℝ)
variable (m n : ℝ)

theorem relationship_m_n (h1 : a > b) (h2 : b > 0) (hm : m = Real.sqrt a - Real.sqrt b) (hn : n = Real.sqrt (a - b)) : m < n := sorry

end NUMINAMATH_GPT_relationship_m_n_l1713_171306


namespace NUMINAMATH_GPT_stockholm_to_uppsala_distance_l1713_171345

-- Definition of conditions
def map_distance : ℝ := 45 -- in cm
def scale1 : ℝ := 10 -- first scale 1 cm : 10 km
def scale2 : ℝ := 5 -- second scale 1 cm : 5 km
def boundary : ℝ := 15 -- first 15 cm at scale 2

-- Calculation of the two parts
def part1_distance (boundary : ℝ) (scale2 : ℝ) := boundary * scale2
def remaining_distance (map_distance boundary : ℝ) := map_distance - boundary
def part2_distance (remaining_distance : ℝ) (scale1 : ℝ) := remaining_distance * scale1

-- Total distance
def total_distance (part1 part2: ℝ) := part1 + part2

theorem stockholm_to_uppsala_distance : 
  total_distance (part1_distance boundary scale2) 
                 (part2_distance (remaining_distance map_distance boundary) scale1) 
  = 375 := 
by
  -- Proof to be provided
  sorry

end NUMINAMATH_GPT_stockholm_to_uppsala_distance_l1713_171345


namespace NUMINAMATH_GPT_sqrt_of_16_is_4_l1713_171304

theorem sqrt_of_16_is_4 : Real.sqrt 16 = 4 :=
sorry

end NUMINAMATH_GPT_sqrt_of_16_is_4_l1713_171304


namespace NUMINAMATH_GPT_geometric_sequence_const_k_l1713_171334

noncomputable def sum_of_terms (n : ℕ) (k : ℤ) : ℤ := 3 * 2^n + k
noncomputable def a1 (k : ℤ) : ℤ := sum_of_terms 1 k
noncomputable def a2 (k : ℤ) : ℤ := sum_of_terms 2 k - sum_of_terms 1 k
noncomputable def a3 (k : ℤ) : ℤ := sum_of_terms 3 k - sum_of_terms 2 k

theorem geometric_sequence_const_k :
  (∀ (k : ℤ), (a1 k * a3 k = a2 k * a2 k) → k = -3) :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_const_k_l1713_171334


namespace NUMINAMATH_GPT_factorize_expression_l1713_171311

theorem factorize_expression (m : ℝ) : 
  4 * m^2 - 64 = 4 * (m + 4) * (m - 4) :=
sorry

end NUMINAMATH_GPT_factorize_expression_l1713_171311


namespace NUMINAMATH_GPT_rental_cost_per_day_l1713_171376

theorem rental_cost_per_day (p m c : ℝ) (d : ℝ) (hc : c = 0.08) (hm : m = 214.0) (hp : p = 46.12) (h_total : p = d + m * c) : d = 29.00 := 
by
  sorry

end NUMINAMATH_GPT_rental_cost_per_day_l1713_171376


namespace NUMINAMATH_GPT_find_length_QR_l1713_171393

-- Conditions
variables {D E F Q R : Type} [MetricSpace D] [MetricSpace E] [MetricSpace F] [MetricSpace Q] [MetricSpace R]
variables {DE EF DF QR : ℝ} (tangent : Q = E ∧ R = D)
variables (t₁ : de = 5) (t₂ : ef = 12) (t₃ : df = 13)

-- Problem: Prove that QR = 5 given the conditions.
theorem find_length_QR : QR = 5 :=
sorry

end NUMINAMATH_GPT_find_length_QR_l1713_171393


namespace NUMINAMATH_GPT_solution_set_ineq1_solution_set_ineq2_l1713_171332

theorem solution_set_ineq1 (x : ℝ) : 
  (-3 * x ^ 2 + x + 1 > 0) ↔ (x ∈ Set.Ioo ((1 - Real.sqrt 13) / 6) ((1 + Real.sqrt 13) / 6)) := 
sorry

theorem solution_set_ineq2 (x : ℝ) : 
  (x ^ 2 - 2 * x + 1 ≤ 0) ↔ (x = 1) := 
sorry

end NUMINAMATH_GPT_solution_set_ineq1_solution_set_ineq2_l1713_171332


namespace NUMINAMATH_GPT_number_of_relatively_prime_to_18_l1713_171320

theorem number_of_relatively_prime_to_18 : 
  ∃ N : ℕ, N = 30 ∧ ∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 → Nat.gcd n 18 = 1 ↔ false :=
by
  sorry

end NUMINAMATH_GPT_number_of_relatively_prime_to_18_l1713_171320


namespace NUMINAMATH_GPT_height_of_pole_l1713_171360

-- Definitions for the conditions
def ascends_first_minute := 2
def slips_second_minute := 1
def net_ascent_per_two_minutes := ascends_first_minute - slips_second_minute
def total_minutes := 17
def pairs_of_minutes := (total_minutes - 1) / 2  -- because the 17th minute is separate
def net_ascent_first_16_minutes := pairs_of_minutes * net_ascent_per_two_minutes

-- The final ascent in the 17th minute
def ascent_final_minute := 2

-- Total ascent
def total_ascent := net_ascent_first_16_minutes + ascent_final_minute

-- Statement to prove the height of the pole
theorem height_of_pole : total_ascent = 10 :=
by
  sorry

end NUMINAMATH_GPT_height_of_pole_l1713_171360


namespace NUMINAMATH_GPT_soft_drink_company_bottle_count_l1713_171391

theorem soft_drink_company_bottle_count
  (B : ℕ)
  (initial_small_bottles : ℕ := 6000)
  (percent_sold_small : ℝ := 0.12)
  (percent_sold_big : ℝ := 0.14)
  (bottles_remaining_total : ℕ := 18180) :
  (initial_small_bottles * (1 - percent_sold_small) + B * (1 - percent_sold_big) = bottles_remaining_total) → B = 15000 :=
by
  sorry

end NUMINAMATH_GPT_soft_drink_company_bottle_count_l1713_171391


namespace NUMINAMATH_GPT_retail_price_machine_l1713_171354

theorem retail_price_machine (P : ℝ) :
  let wholesale_price := 99
  let discount_rate := 0.10
  let profit_rate := 0.20
  let selling_price := wholesale_price + (profit_rate * wholesale_price)
  0.90 * P = selling_price → P = 132 :=

by
  intro wholesale_price discount_rate profit_rate selling_price h
  sorry -- Proof will be handled here

end NUMINAMATH_GPT_retail_price_machine_l1713_171354


namespace NUMINAMATH_GPT_smallest_n_for_Qn_l1713_171390

theorem smallest_n_for_Qn (n : ℕ) : 
  (∃ n : ℕ, 1 / (n * (2 * n + 1)) < 1 / 2023 ∧ ∀ m < n, 1 / (m * (2 * m + 1)) ≥ 1 / 2023) ↔ n = 32 := by
sorry

end NUMINAMATH_GPT_smallest_n_for_Qn_l1713_171390


namespace NUMINAMATH_GPT_original_length_of_ribbon_l1713_171382

theorem original_length_of_ribbon (n : ℕ) (cm_per_piece : ℝ) (remaining_meters : ℝ) 
  (pieces_cm_to_m : cm_per_piece / 100 = 0.15) (remaining_ribbon : remaining_meters = 36) 
  (pieces_cut : n = 100) : n * (cm_per_piece / 100) + remaining_meters = 51 := 
by 
  sorry

end NUMINAMATH_GPT_original_length_of_ribbon_l1713_171382


namespace NUMINAMATH_GPT_min_value_frac_add_x_l1713_171335

theorem min_value_frac_add_x (x : ℝ) (h : x > 3) : (∃ m, (∀ (y : ℝ), y > 3 → (4 / y - 3 + y) ≥ m) ∧ m = 7) :=
sorry

end NUMINAMATH_GPT_min_value_frac_add_x_l1713_171335


namespace NUMINAMATH_GPT_shadow_boundary_eqn_l1713_171331

noncomputable def boundary_of_shadow (x : ℝ) : ℝ := x^2 / 10 - 1

theorem shadow_boundary_eqn (radius : ℝ) (center : ℝ × ℝ × ℝ) (light_source : ℝ × ℝ × ℝ) (x y: ℝ) :
  radius = 2 →
  center = (0, 0, 2) →
  light_source = (0, -2, 3) →
  y = boundary_of_shadow x :=
by
  intros hradius hcenter hlight
  sorry

end NUMINAMATH_GPT_shadow_boundary_eqn_l1713_171331


namespace NUMINAMATH_GPT_charges_needed_to_vacuum_house_l1713_171373

-- Conditions definitions
def battery_last_minutes : ℕ := 10
def vacuum_time_per_room : ℕ := 4
def number_of_bedrooms : ℕ := 3
def number_of_kitchens : ℕ := 1
def number_of_living_rooms : ℕ := 1

-- Question (proof problem statement)
theorem charges_needed_to_vacuum_house :
  ((number_of_bedrooms + number_of_kitchens + number_of_living_rooms) * vacuum_time_per_room) / battery_last_minutes = 2 :=
by
  sorry

end NUMINAMATH_GPT_charges_needed_to_vacuum_house_l1713_171373


namespace NUMINAMATH_GPT_smallest_m_exists_l1713_171329

theorem smallest_m_exists :
  ∃ (m : ℕ), 0 < m ∧ (∃ k : ℕ, 5 * m = k^2) ∧ (∃ l : ℕ, 3 * m = l^3) ∧ m = 243 :=
by
  sorry

end NUMINAMATH_GPT_smallest_m_exists_l1713_171329


namespace NUMINAMATH_GPT_system1_solution_system2_solution_l1713_171379

theorem system1_solution (x y : ℝ) (h₁ : y = 2 * x) (h₂ : 3 * y + 2 * x = 8) : x = 1 ∧ y = 2 := 
by sorry

theorem system2_solution (x y : ℝ) (h₁ : x - 3 * y = -2) (h₂ : 2 * x + 3 * y = 3) : x = (1 / 3) ∧ y = (7 / 9) := 
by sorry

end NUMINAMATH_GPT_system1_solution_system2_solution_l1713_171379


namespace NUMINAMATH_GPT_cos_150_eq_negative_cos_30_l1713_171347

theorem cos_150_eq_negative_cos_30 :
  Real.cos (150 * Real.pi / 180) = - (Real.cos (30 * Real.pi / 180)) :=
by sorry

end NUMINAMATH_GPT_cos_150_eq_negative_cos_30_l1713_171347


namespace NUMINAMATH_GPT_eq_solutions_count_l1713_171342

theorem eq_solutions_count : 
  ∃! (n : ℕ), n = 126 ∧ (∀ x y : ℕ, 2*x + 3*y = 768 → x > 0 ∧ y > 0 → ∃ t : ℤ, x = 384 + 3*t ∧ y = -2*t ∧ -127 ≤ t ∧ t <= -1) := sorry

end NUMINAMATH_GPT_eq_solutions_count_l1713_171342


namespace NUMINAMATH_GPT_find_a10_l1713_171341

-- Define the arithmetic sequence with its common difference and initial term
axiom a_seq : ℕ → ℝ
axiom a1 : ℝ
axiom d : ℝ

-- Conditions
axiom a3 : a_seq 3 = a1 + 2 * d
axiom a5_a8 : a_seq 5 + a_seq 8 = 15

-- Theorem statement
theorem find_a10 : a_seq 10 = 13 :=
by sorry

end NUMINAMATH_GPT_find_a10_l1713_171341


namespace NUMINAMATH_GPT_sqrt_product_simplifies_l1713_171353

theorem sqrt_product_simplifies (p : ℝ) : 
  Real.sqrt (12 * p) * Real.sqrt (20 * p) * Real.sqrt (15 * p^2) = 60 * p^2 := 
by
  sorry

end NUMINAMATH_GPT_sqrt_product_simplifies_l1713_171353


namespace NUMINAMATH_GPT_max_diff_real_roots_l1713_171351

-- Definitions of the quadratic equations
def eq1 (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0
def eq2 (a b c x : ℝ) : Prop := b * x^2 + c * x + a = 0
def eq3 (a b c x : ℝ) : Prop := c * x^2 + a * x + b = 0

-- The proof statement
theorem max_diff_real_roots (a b c : ℝ) (hnz : a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) :
  ∃ x y : ℝ, eq1 a b c x ∧ eq1 a b c y ∧ eq2 a b c x ∧ eq2 a b c y ∧ eq3 a b c x ∧ eq3 a b c y ∧ 
  abs (x - y) = 0 := sorry

end NUMINAMATH_GPT_max_diff_real_roots_l1713_171351


namespace NUMINAMATH_GPT_gcd_91_49_l1713_171364

theorem gcd_91_49 : Nat.gcd 91 49 = 7 :=
by
  -- Using the Euclidean algorithm
  -- 91 = 49 * 1 + 42
  -- 49 = 42 * 1 + 7
  -- 42 = 7 * 6 + 0
  sorry

end NUMINAMATH_GPT_gcd_91_49_l1713_171364


namespace NUMINAMATH_GPT_minimum_value_of_reciprocal_squares_l1713_171366

theorem minimum_value_of_reciprocal_squares
  (a b : ℝ)
  (h : a ≠ 0 ∧ b ≠ 0)
  (h_eq : (a^2) + 4 * (b^2) = 9)
  : (1/(a^2) + 1/(b^2)) = 1 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_reciprocal_squares_l1713_171366


namespace NUMINAMATH_GPT_inequality_proof_l1713_171328

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_sum : a + b + c = 1) :
  1 / (b * c + a + 1 / a) + 1 / (a * c + b + 1 / b) + 1 / (a * b + c + 1 / c) ≤ 27 / 31 :=
sorry

end NUMINAMATH_GPT_inequality_proof_l1713_171328


namespace NUMINAMATH_GPT_statement_A_statement_B_statement_C_l1713_171338

variable {a b : ℝ}
variable (ha : a > 0) (hb : b > 0)

theorem statement_A : (ab ≤ 1) → (1/a + 1/b ≥ 2) :=
by
  sorry

theorem statement_B : (a + b = 4) → (∀ x, (x = 1/a + 9/b) → (x ≥ 4)) :=
by
  sorry

theorem statement_C : (a^2 + b^2 = 4) → (ab ≤ 2) :=
by
  sorry

end NUMINAMATH_GPT_statement_A_statement_B_statement_C_l1713_171338


namespace NUMINAMATH_GPT_find_base_l1713_171319
-- Import the necessary library

-- Define the conditions and the result
theorem find_base (x y b : ℕ) (h1 : x - y = 9) (h2 : x = 9) (h3 : b^x * 4^y = 19683) : b = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_base_l1713_171319


namespace NUMINAMATH_GPT_exists_four_scientists_l1713_171367

theorem exists_four_scientists {n : ℕ} (h1 : n = 50)
  (knows : Fin n → Finset (Fin n))
  (h2 : ∀ x, (knows x).card ≥ 25) :
  ∃ a b c d : Fin n, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧
  a ≠ c ∧ b ≠ d ∧
  a ∈ knows b ∧ b ∈ knows c ∧ c ∈ knows d ∧ d ∈ knows a :=
by
  sorry

end NUMINAMATH_GPT_exists_four_scientists_l1713_171367


namespace NUMINAMATH_GPT_infinite_geometric_series_sum_l1713_171381

-- First term of the geometric series
def a : ℚ := 5/3

-- Common ratio of the geometric series
def r : ℚ := -1/4

-- The sum of the infinite geometric series
def S : ℚ := a / (1 - r)

-- Prove that the sum of the series is equal to 4/3
theorem infinite_geometric_series_sum : S = 4/3 := by
  sorry

end NUMINAMATH_GPT_infinite_geometric_series_sum_l1713_171381


namespace NUMINAMATH_GPT_expression_evaluation_l1713_171395

theorem expression_evaluation :
  (4⁻¹ - Real.sqrt (1 / 16) + (3 - Real.sqrt 2) ^ 0 = 1) :=
by
  -- Step by step calculations skipped
  sorry

end NUMINAMATH_GPT_expression_evaluation_l1713_171395


namespace NUMINAMATH_GPT_even_number_of_divisors_less_than_100_l1713_171348

theorem even_number_of_divisors_less_than_100 : 
  ∃ n, n = 90 ∧ ∀ x < 100, (∃ k, k * k = x → false) = (x ∣ 99 - 9) :=
sorry

end NUMINAMATH_GPT_even_number_of_divisors_less_than_100_l1713_171348


namespace NUMINAMATH_GPT_thomas_weekly_wage_l1713_171374

theorem thomas_weekly_wage (monthly_wage : ℕ) (weeks_in_month : ℕ) (weekly_wage : ℕ) 
    (h1 : monthly_wage = 19500) (h2 : weeks_in_month = 4) :
    weekly_wage = 4875 :=
by
  have h3 : weekly_wage = monthly_wage / weeks_in_month := sorry
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_thomas_weekly_wage_l1713_171374


namespace NUMINAMATH_GPT_find_p_and_q_l1713_171371

theorem find_p_and_q (p q : ℝ)
    (M : Set ℝ := {x | x^2 + p * x - 2 = 0})
    (N : Set ℝ := {x | x^2 - 2 * x + q = 0})
    (h : M ∪ N = {-1, 0, 2}) :
    p = -1 ∧ q = 0 :=
sorry

end NUMINAMATH_GPT_find_p_and_q_l1713_171371


namespace NUMINAMATH_GPT_conor_total_vegetables_weekly_l1713_171344

def conor_vegetables_daily (e c p o z : ℕ) : ℕ :=
  e + c + p + o + z

def conor_vegetables_weekly (vegetables_daily days_worked : ℕ) : ℕ :=
  vegetables_daily * days_worked

theorem conor_total_vegetables_weekly :
  conor_vegetables_weekly (conor_vegetables_daily 12 9 8 15 7) 6 = 306 := by
  sorry

end NUMINAMATH_GPT_conor_total_vegetables_weekly_l1713_171344


namespace NUMINAMATH_GPT_beach_relaxing_people_l1713_171399

def row1_original := 24
def row1_got_up := 3

def row2_original := 20
def row2_got_up := 5

def row3_original := 18

def total_left_relaxing (r1o r1u r2o r2u r3o : Nat) : Nat :=
  r1o + r2o + r3o - (r1u + r2u)

theorem beach_relaxing_people : total_left_relaxing row1_original row1_got_up row2_original row2_got_up row3_original = 54 :=
by
  sorry

end NUMINAMATH_GPT_beach_relaxing_people_l1713_171399


namespace NUMINAMATH_GPT_transform_equation_l1713_171305

theorem transform_equation (x y : ℝ) (h : y = x + x⁻¹) :
  x^4 + x^3 - 5 * x^2 + x + 1 = 0 ↔ x^2 * (y^2 + y - 7) = 0 := 
sorry

end NUMINAMATH_GPT_transform_equation_l1713_171305


namespace NUMINAMATH_GPT_non_zero_real_solution_of_equation_l1713_171389

noncomputable def equation_solution : Prop :=
  ∀ (x : ℝ), x ≠ 0 ∧ (7 * x) ^ 14 = (14 * x) ^ 7 → x = 2 / 7

theorem non_zero_real_solution_of_equation : equation_solution := sorry

end NUMINAMATH_GPT_non_zero_real_solution_of_equation_l1713_171389


namespace NUMINAMATH_GPT_arithmetic_mean_l1713_171396

variable (x b : ℝ)

theorem arithmetic_mean (hx : x ≠ 0) :
  ((x + b) / x + (x - 2 * b) / x) / 2 = 1 - b / (2 * x) := by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_l1713_171396


namespace NUMINAMATH_GPT_claudia_ratio_of_kids_l1713_171333

def claudia_art_class :=
  let saturday_kids := 20
  let sunday_kids := (300 - saturday_kids * 10) / 10
  sunday_kids / saturday_kids = 1 / 2

theorem claudia_ratio_of_kids :
  let saturday_kids := 20
  let sunday_kids := (300 - saturday_kids * 10) / 10
  (sunday_kids / saturday_kids = 1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_claudia_ratio_of_kids_l1713_171333


namespace NUMINAMATH_GPT_parallel_implies_not_contained_l1713_171308

variables {Line Plane : Type} (l : Line) (α : Plane)

-- Define the predicate for a line being parallel to a plane
def parallel (l : Line) (α : Plane) : Prop := sorry

-- Define the predicate for a line not being contained in a plane
def not_contained (l : Line) (α : Plane) : Prop := sorry

theorem parallel_implies_not_contained (l : Line) (α : Plane) (h : parallel l α) : not_contained l α :=
sorry

end NUMINAMATH_GPT_parallel_implies_not_contained_l1713_171308


namespace NUMINAMATH_GPT_inequality_x_y_z_l1713_171352

theorem inequality_x_y_z (x y z : ℝ) (hx : 0 < x ∧ x < 1) (hy : 0 < y ∧ y < 1) (hz : 0 < z ∧ z < 1) :
  x * (1 - y) + y * (1 - z) + z * (1 - x) < 1 := 
by sorry

end NUMINAMATH_GPT_inequality_x_y_z_l1713_171352


namespace NUMINAMATH_GPT_triangle_side_relationship_l1713_171355

theorem triangle_side_relationship
  (a b c : ℝ)
  (habc : a < b + c)
  (ha_pos : a > 0) :
  a^2 < a * b + a * c :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_relationship_l1713_171355


namespace NUMINAMATH_GPT_area_triangle_ABC_l1713_171322

-- Definitions of the lengths and height
def BD : ℝ := 3
def DC : ℝ := 2 * BD
def BC : ℝ := BD + DC
def h_A_BC : ℝ := 4

-- The triangle area formula
def areaOfTriangle (base height : ℝ) : ℝ := 0.5 * base * height

-- The goal to prove that the area of triangle ABC is 18 square units
theorem area_triangle_ABC : areaOfTriangle BC h_A_BC = 18 := by
  sorry

end NUMINAMATH_GPT_area_triangle_ABC_l1713_171322


namespace NUMINAMATH_GPT_max_value_function_l1713_171307

theorem max_value_function (x : ℝ) (h : x > 4) : -x + (1 / (4 - x)) ≤ -6 :=
sorry

end NUMINAMATH_GPT_max_value_function_l1713_171307


namespace NUMINAMATH_GPT_simplify_expression_l1713_171394

variable (x y : ℝ)

theorem simplify_expression : 3 * x^2 * y * (2 / (9 * x^3 * y)) = 2 / (3 * x) :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l1713_171394


namespace NUMINAMATH_GPT_find_x_such_that_custom_op_neg3_eq_one_l1713_171372

def custom_op (x y : Int) : Int := x * y - 2 * (x + y)

theorem find_x_such_that_custom_op_neg3_eq_one :
  ∃ x : Int, custom_op x (-3) = 1 ∧ x = 1 :=
by
  use 1
  sorry

end NUMINAMATH_GPT_find_x_such_that_custom_op_neg3_eq_one_l1713_171372


namespace NUMINAMATH_GPT_charles_remaining_skittles_l1713_171310

def c : ℕ := 25
def d : ℕ := 7
def remaining_skittles : ℕ := c - d

theorem charles_remaining_skittles : remaining_skittles = 18 := by
  sorry

end NUMINAMATH_GPT_charles_remaining_skittles_l1713_171310


namespace NUMINAMATH_GPT_total_length_of_scale_l1713_171380

theorem total_length_of_scale 
  (n : ℕ) (len_per_part : ℕ) 
  (h_n : n = 5) 
  (h_len_per_part : len_per_part = 25) :
  n * len_per_part = 125 :=
by
  sorry

end NUMINAMATH_GPT_total_length_of_scale_l1713_171380


namespace NUMINAMATH_GPT_no_solution_system_l1713_171357

theorem no_solution_system (a : ℝ) : 
  (∀ x : ℝ, (x - 2 * a > 0) → (3 - 2 * x > x - 6) → false) ↔ a ≥ 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_no_solution_system_l1713_171357


namespace NUMINAMATH_GPT_part1_part2_l1713_171363

variable (x α β : ℝ)

noncomputable def f (x : ℝ) : ℝ := 
  2 * Real.sqrt 3 * (Real.cos x) ^ 2 + 2 * (Real.sin x) * (Real.cos x) - Real.sqrt 3

theorem part1 (hx : 0 ≤ x ∧ x ≤ Real.pi / 2) : 
  -Real.sqrt 3 ≤ f x ∧ f x ≤ 2 := 
sorry

theorem part2 (hα : 0 < α ∧ α < Real.pi / 2) (hβ : 0 < β ∧ β < Real.pi / 2) 
(h1 : f (α / 2 - Real.pi / 6) = 8 / 5) 
(h2 : Real.cos (α + β) = -12 / 13) : 
  Real.sin β = 63 / 65 := 
sorry

end NUMINAMATH_GPT_part1_part2_l1713_171363


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l1713_171356

theorem hyperbola_eccentricity (a b : ℝ) (h_asymptote : a = 3 * b) : 
    (a^2 + b^2) / a^2 = 10 / 9 := 
by
    sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l1713_171356


namespace NUMINAMATH_GPT_equal_real_roots_possible_values_l1713_171302

theorem equal_real_roots_possible_values (a : ℝ): 
  (∀ x : ℝ, x^2 + a * x + 1 = 0) → (a = 2 ∨ a = -2) :=
by
  sorry

end NUMINAMATH_GPT_equal_real_roots_possible_values_l1713_171302


namespace NUMINAMATH_GPT_value_of_g_at_13_l1713_171350

-- Define the function g
def g (n : ℕ) : ℕ := n^2 + n + 23

-- The theorem to prove
theorem value_of_g_at_13 : g 13 = 205 := by
  -- Rewrite using the definition of g
  unfold g
  -- Perform the arithmetic
  sorry

end NUMINAMATH_GPT_value_of_g_at_13_l1713_171350


namespace NUMINAMATH_GPT_add_fractions_l1713_171349

theorem add_fractions : (7 / 12) + (3 / 8) = 23 / 24 := by
  sorry

end NUMINAMATH_GPT_add_fractions_l1713_171349


namespace NUMINAMATH_GPT_min_u_value_l1713_171327

theorem min_u_value (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2 * y = 1) : 
  (x + 1 / x) * (y + 1 / (4 * y)) ≥ 25 / 8 :=
by
  sorry

end NUMINAMATH_GPT_min_u_value_l1713_171327


namespace NUMINAMATH_GPT_triangle_is_isosceles_right_l1713_171336

theorem triangle_is_isosceles_right (a b c : ℝ) (A B C : ℝ) (h1 : b = a * Real.sin C) (h2 : c = a * Real.cos B) : 
  A = π / 2 ∧ b = c := 
sorry

end NUMINAMATH_GPT_triangle_is_isosceles_right_l1713_171336


namespace NUMINAMATH_GPT_evaluate_nested_operation_l1713_171321

def operation (a b c : ℕ) : ℕ := (a + b) / c

theorem evaluate_nested_operation : operation (operation 72 36 108) (operation 4 2 6) (operation 12 6 18) = 2 := by
  -- Here we assume all operations are valid (c ≠ 0 for each case)
  sorry

end NUMINAMATH_GPT_evaluate_nested_operation_l1713_171321


namespace NUMINAMATH_GPT_andy_max_cookies_l1713_171384

-- Definitions for the problem conditions
def total_cookies := 36
def bella_eats (andy_cookies : ℕ) := 2 * andy_cookies
def charlie_eats (andy_cookies : ℕ) := andy_cookies
def consumed_cookies (andy_cookies : ℕ) := andy_cookies + bella_eats andy_cookies + charlie_eats andy_cookies

-- The statement to prove
theorem andy_max_cookies : ∃ (a : ℕ), consumed_cookies a = total_cookies ∧ a = 9 :=
by
  sorry

end NUMINAMATH_GPT_andy_max_cookies_l1713_171384


namespace NUMINAMATH_GPT_simplify_poly_l1713_171339

-- Define the polynomial expressions
def poly1 (r : ℝ) := 2 * r^3 + 4 * r^2 + 5 * r - 3
def poly2 (r : ℝ) := r^3 + 6 * r^2 + 8 * r - 7

-- Simplification goal
theorem simplify_poly (r : ℝ) : (poly1 r) - (poly2 r) = r^3 - 2 * r^2 - 3 * r + 4 :=
by 
  -- We declare the proof is omitted using sorry
  sorry

end NUMINAMATH_GPT_simplify_poly_l1713_171339


namespace NUMINAMATH_GPT_option_d_correct_l1713_171313

theorem option_d_correct (x : ℝ) : (-3 * x + 2) * (-3 * x - 2) = 9 * x^2 - 4 := 
  sorry

end NUMINAMATH_GPT_option_d_correct_l1713_171313


namespace NUMINAMATH_GPT_largest_n_satisfies_conditions_l1713_171365

theorem largest_n_satisfies_conditions :
  ∃ (n m a : ℤ), n = 313 ∧ n^2 = (m + 1)^3 - m^3 ∧ ∃ (k : ℤ), 2 * n + 103 = k^2 :=
by
  sorry

end NUMINAMATH_GPT_largest_n_satisfies_conditions_l1713_171365


namespace NUMINAMATH_GPT_min_value_l1713_171378

theorem min_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 4 * y = 1) : 
  ∃ m, m = (1 / x + 1 / y) ∧ m = 9 :=
by
  sorry

end NUMINAMATH_GPT_min_value_l1713_171378


namespace NUMINAMATH_GPT_no_solutions_iff_a_positive_and_discriminant_non_positive_l1713_171397

theorem no_solutions_iff_a_positive_and_discriminant_non_positive (a b c : ℝ) (h1 : a ≠ 0) :
  (∀ x : ℝ, ¬ (a * x^2 + b * x + c < 0)) ↔ (a > 0 ∧ (b^2 - 4 * a * c) ≤ 0) :=
  sorry

end NUMINAMATH_GPT_no_solutions_iff_a_positive_and_discriminant_non_positive_l1713_171397


namespace NUMINAMATH_GPT_sector_max_area_l1713_171387

noncomputable def max_sector_area (R c : ℝ) : ℝ := 
  if h : R = c / 4 then c^2 / 16 else 0 -- This is just a skeleton, actual proof requires conditions
-- State the theorem that relates conditions to the maximum area.
theorem sector_max_area (R c α : ℝ) 
  (hc : c = 2 * R + R * α) : 
  (∃ R, R = c / 4) → max_sector_area R c = c^2 / 16 :=
by 
  sorry

end NUMINAMATH_GPT_sector_max_area_l1713_171387


namespace NUMINAMATH_GPT_ratio_AB_PQ_f_half_func_f_l1713_171315

-- Define given conditions
variables {m n : ℝ} -- Lengths of AB and PQ
variables {h : ℝ} -- Height of triangle and rectangle (both are 1)
variables {x : ℝ} -- Variable in the range [0, 1]

-- Same area and height conditions
axiom areas_equal : m / 2 = n
axiom height_equal : h = 1

-- Given the areas are equal and height is 1
theorem ratio_AB_PQ : m / n = 2 :=
by sorry -- Proof of the ratio 

-- Given the specific calculation for x = 1/2
theorem f_half (hx : x = 1 / 2) (f : ℝ → ℝ) (heq : ∀ x, (f x) * n = (m / 2) * (2 * x - x^2)) :
  f (1 / 2) = 3 / 4 :=
by sorry -- Proof of function value at 1/2

-- Prove the expression of the function f(x)
theorem func_f (f : ℝ → ℝ) (heq : ∀ x, (f x) * n = (m / 2) * (2 * x - x^2)) :
  ∀ x, 0 ≤ x → x ≤ 1 → f x = 2 * x - x^2 :=
by sorry -- Proof of the function expression


end NUMINAMATH_GPT_ratio_AB_PQ_f_half_func_f_l1713_171315


namespace NUMINAMATH_GPT_moles_of_C2H6_formed_l1713_171303

-- Definitions of the quantities involved
def moles_H2 : ℕ := 3
def moles_C2H4 : ℕ := 3
def moles_C2H6 : ℕ := 3

-- Stoichiometry condition stated in a way that Lean can understand.
axiom stoichiometry : moles_H2 = moles_C2H4

theorem moles_of_C2H6_formed : moles_C2H6 = 3 :=
by
  -- Assume the constraints and state the final result
  have h : moles_H2 = moles_C2H4 := stoichiometry
  show moles_C2H6 = 3
  sorry

end NUMINAMATH_GPT_moles_of_C2H6_formed_l1713_171303


namespace NUMINAMATH_GPT_minimum_trips_needed_l1713_171359

def masses : List ℕ := [150, 60, 70, 71, 72, 100, 101, 102, 103]
def capacity : ℕ := 200

theorem minimum_trips_needed (masses : List ℕ) (capacity : ℕ) : 
  masses = [150, 60, 70, 71, 72, 100, 101, 102, 103] →
  capacity = 200 →
  ∃ trips : ℕ, trips = 5 :=
by
  sorry

end NUMINAMATH_GPT_minimum_trips_needed_l1713_171359


namespace NUMINAMATH_GPT_ellipse_eccentricity_l1713_171337

theorem ellipse_eccentricity :
  (∃ (e : ℝ), (∀ (x y : ℝ), ((x^2 / 9) + y^2 = 1) → (e = 2 * Real.sqrt 2 / 3))) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_eccentricity_l1713_171337


namespace NUMINAMATH_GPT_ants_meet_at_QS_l1713_171369

theorem ants_meet_at_QS (P Q R S : Type)
  (dist_PQ : Nat)
  (dist_QR : Nat)
  (dist_PR : Nat)
  (ants_meet : 2 * (dist_PQ + (5 : Nat)) = dist_PQ + dist_QR + dist_PR)
  (perimeter : dist_PQ + dist_QR + dist_PR = 24)
  (distance_each_ant_crawls : (dist_PQ + 5) = 12) :
  5 = 5 :=
by
  sorry

end NUMINAMATH_GPT_ants_meet_at_QS_l1713_171369


namespace NUMINAMATH_GPT_christine_needs_32_tablespoons_l1713_171330

-- Define the conditions
def tablespoons_per_egg_white : ℕ := 2
def egg_whites_per_cake : ℕ := 8
def number_of_cakes : ℕ := 2

-- Define the calculation for total tablespoons of aquafaba needed
def total_tbs_aquafaba : ℕ :=
  tablespoons_per_egg_white * (egg_whites_per_cake * number_of_cakes)

-- The theorem to prove
theorem christine_needs_32_tablespoons :
  total_tbs_aquafaba = 32 :=
by 
  -- Placeholder for proof, as proof steps are not required
  sorry

end NUMINAMATH_GPT_christine_needs_32_tablespoons_l1713_171330


namespace NUMINAMATH_GPT_range_of_a_l1713_171317

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → a ≥ Real.exp x) ∧ (∃ x : ℝ, x^2 - 4 * x + a ≤ 0) →
  a ∈ Set.Icc (Real.exp 1) 4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1713_171317


namespace NUMINAMATH_GPT_frosting_cupcakes_l1713_171358

theorem frosting_cupcakes (R_Cagney R_Lacey R_Jamie : ℕ)
  (H1 : R_Cagney = 1 / 20)
  (H2 : R_Lacey = 1 / 30)
  (H3 : R_Jamie = 1 / 40)
  (TotalTime : ℕ)
  (H4 : TotalTime = 600) :
  (R_Cagney + R_Lacey + R_Jamie) * TotalTime = 65 :=
by
  sorry

end NUMINAMATH_GPT_frosting_cupcakes_l1713_171358


namespace NUMINAMATH_GPT_find_rth_term_l1713_171388

theorem find_rth_term (n r : ℕ) (S : ℕ → ℕ) (hS : ∀ n, S n = 4 * n + 5 * n^2) :
  r > 0 → (S r) - (S (r - 1)) = 10 * r - 1 :=
by
  intro h
  have hr_pos := h
  sorry

end NUMINAMATH_GPT_find_rth_term_l1713_171388


namespace NUMINAMATH_GPT_find_n_l1713_171340

-- Define the polynomial function
def polynomial (n : ℤ) : ℤ :=
  n^4 + 2 * n^3 + 6 * n^2 + 12 * n + 25

-- Define the condition that n is a positive integer
def is_positive_integer (n : ℤ) : Prop :=
  n > 0

-- Define the condition that polynomial is a perfect square
def is_perfect_square (k : ℤ) : Prop :=
  ∃ m : ℤ, m^2 = k

-- The theorem we need to prove
theorem find_n (n : ℤ) (h1 : is_positive_integer n) (h2 : is_perfect_square (polynomial n)) : n = 8 :=
sorry

end NUMINAMATH_GPT_find_n_l1713_171340


namespace NUMINAMATH_GPT_factor_expression_l1713_171346

theorem factor_expression (x : ℝ) : 25 * x^2 + 10 * x = 5 * x * (5 * x + 2) :=
sorry

end NUMINAMATH_GPT_factor_expression_l1713_171346


namespace NUMINAMATH_GPT_Ruth_sandwiches_l1713_171385

theorem Ruth_sandwiches (sandwiches_left sandwiches_ruth sandwiches_brother sandwiches_first_cousin sandwiches_two_cousins total_sandwiches : ℕ)
  (h_ruth : sandwiches_ruth = 1)
  (h_brother : sandwiches_brother = 2)
  (h_first_cousin : sandwiches_first_cousin = 2)
  (h_two_cousins : sandwiches_two_cousins = 2)
  (h_left : sandwiches_left = 3) :
  total_sandwiches = sandwiches_left + sandwiches_two_cousins + sandwiches_first_cousin + sandwiches_ruth + sandwiches_brother :=
by
  sorry

end NUMINAMATH_GPT_Ruth_sandwiches_l1713_171385


namespace NUMINAMATH_GPT_probability_after_first_new_draw_is_five_ninths_l1713_171383

-- Defining the conditions in Lean
def total_balls : ℕ := 10
def new_balls : ℕ := 6
def old_balls : ℕ := 4

def balls_remaining_after_first_draw : ℕ := total_balls - 1
def new_balls_after_first_draw : ℕ := new_balls - 1

-- Using the classic probability definition
def probability_of_drawing_second_new_ball := (new_balls_after_first_draw : ℚ) / (balls_remaining_after_first_draw : ℚ)

-- Stating the theorem to be proved
theorem probability_after_first_new_draw_is_five_ninths :
  probability_of_drawing_second_new_ball = 5/9 := sorry

end NUMINAMATH_GPT_probability_after_first_new_draw_is_five_ninths_l1713_171383


namespace NUMINAMATH_GPT_constant_term_of_second_eq_l1713_171314

theorem constant_term_of_second_eq (x y : ℝ) 
  (h1 : 7*x + y = 19) 
  (h2 : 2*x + y = 5) : 
  ∃ k : ℝ, x + 3*y = k ∧ k = 15 := 
by
  sorry

end NUMINAMATH_GPT_constant_term_of_second_eq_l1713_171314
