import Mathlib

namespace probability_P_is_1_over_3_l1954_195442

-- Definitions and conditions
def A := 0
def B := 3
def C := 1
def D := 2
def length_AB := B - A
def length_CD := D - C

-- Problem statement to prove
theorem probability_P_is_1_over_3 : (length_CD / length_AB) = 1 / 3 := by
  sorry

end probability_P_is_1_over_3_l1954_195442


namespace jace_gave_to_neighbor_l1954_195416

theorem jace_gave_to_neighbor
  (earnings : ℕ) (debt : ℕ) (remaining : ℕ) (cents_per_dollar : ℕ) :
  earnings = 1000 →
  debt = 358 →
  remaining = 642 →
  cents_per_dollar = 100 →
  earnings - debt - remaining = 0
:= by
  intros h1 h2 h3 h4
  sorry

end jace_gave_to_neighbor_l1954_195416


namespace michelle_phone_bill_l1954_195427

def base_cost : ℝ := 20
def text_cost_per_message : ℝ := 0.05
def minute_cost_over_20h : ℝ := 0.20
def messages_sent : ℝ := 150
def hours_talked : ℝ := 22
def allowed_hours : ℝ := 20

theorem michelle_phone_bill :
  base_cost + (messages_sent * text_cost_per_message) +
  ((hours_talked - allowed_hours) * 60 * minute_cost_over_20h) = 51.50 := by
  sorry

end michelle_phone_bill_l1954_195427


namespace no_four_consecutive_perf_square_l1954_195437

theorem no_four_consecutive_perf_square :
  ¬ ∃ (x : ℕ), x > 0 ∧ ∃ (k : ℕ), x * (x + 1) * (x + 2) * (x + 3) = k^2 :=
by
  sorry

end no_four_consecutive_perf_square_l1954_195437


namespace large_cross_area_is_60_cm_squared_l1954_195487

noncomputable def small_square_area (s : ℝ) := s * s
noncomputable def large_square_area (s : ℝ) := 4 * small_square_area s
noncomputable def small_cross_area (s : ℝ) := 5 * small_square_area s
noncomputable def large_cross_area (s : ℝ) := 5 * large_square_area s
noncomputable def remaining_area (s : ℝ) := large_cross_area s - small_cross_area s

theorem large_cross_area_is_60_cm_squared :
  ∃ (s : ℝ), remaining_area s = 45 → large_cross_area s = 60 :=
by
  sorry

end large_cross_area_is_60_cm_squared_l1954_195487


namespace quotient_of_a_by_b_l1954_195480

-- Definitions based on given conditions
def a : ℝ := 0.0204
def b : ℝ := 17

-- Statement to be proven
theorem quotient_of_a_by_b : a / b = 0.0012 := 
by
  sorry

end quotient_of_a_by_b_l1954_195480


namespace ratio_first_term_l1954_195497

theorem ratio_first_term (x : ℕ) (r : ℕ × ℕ) (h₀ : r = (6 - x, 7 - x)) 
        (h₁ : x ≥ 3) (h₂ : r.1 < r.2) : r.1 < 4 :=
by
  sorry

end ratio_first_term_l1954_195497


namespace total_people_on_playground_l1954_195459

open Nat

-- Conditions
def num_girls := 28
def num_boys := 35
def num_3rd_grade_girls := 15
def num_3rd_grade_boys := 18
def num_teachers := 4

-- Derived values (from conditions)
def num_4th_grade_girls := num_girls - num_3rd_grade_girls
def num_4th_grade_boys := num_boys - num_3rd_grade_boys
def num_3rd_graders := num_3rd_grade_girls + num_3rd_grade_boys
def num_4th_graders := num_4th_grade_girls + num_4th_grade_boys

-- Total number of people
def total_people := num_3rd_graders + num_4th_graders + num_teachers

-- Proof statement
theorem total_people_on_playground : total_people = 67 :=
  by
     -- This is where the proof would go
     sorry

end total_people_on_playground_l1954_195459


namespace range_of_a_l1954_195436

def tensor (x y : ℝ) : ℝ := x * (1 - y)

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → tensor (x - a) (x + a) < 2) → -1 < a ∧ a < 2 := by
  sorry

end range_of_a_l1954_195436


namespace det_2x2_matrix_l1954_195435

open Matrix

theorem det_2x2_matrix : 
  det ![![7, -2], ![-3, 5]] = 29 := by
  sorry

end det_2x2_matrix_l1954_195435


namespace sets_are_equal_l1954_195433

def setA : Set ℤ := {a | ∃ m n l : ℤ, a = 12 * m + 8 * n + 4 * l}
def setB : Set ℤ := {b | ∃ p q r : ℤ, b = 20 * p + 16 * q + 12 * r}

theorem sets_are_equal : setA = setB := sorry

end sets_are_equal_l1954_195433


namespace distance_at_1_5_l1954_195485

def total_distance : ℝ := 174
def speed : ℝ := 60
def travel_time (x : ℝ) : ℝ := total_distance - speed * x

theorem distance_at_1_5 :
  travel_time 1.5 = 84 := by
  sorry

end distance_at_1_5_l1954_195485


namespace ab_zero_l1954_195418

theorem ab_zero (a b : ℝ)
  (h1 : a + b = 5)
  (h2 : a^3 + b^3 = 125) : 
  a * b = 0 :=
by
  sorry

end ab_zero_l1954_195418


namespace subtracted_amount_l1954_195401

theorem subtracted_amount (N A : ℝ) (h1 : 0.30 * N - A = 20) (h2 : N = 300) : A = 70 :=
by
  sorry

end subtracted_amount_l1954_195401


namespace certain_event_abs_nonneg_l1954_195479

theorem certain_event_abs_nonneg (x : ℝ) : |x| ≥ 0 :=
by
  sorry

end certain_event_abs_nonneg_l1954_195479


namespace area_of_square_field_l1954_195421

theorem area_of_square_field (d : ℝ) (s : ℝ) (A : ℝ) (h_d : d = 28) (h_relation : d = s * Real.sqrt 2) (h_area : A = s^2) :
  A = 391.922 :=
by sorry

end area_of_square_field_l1954_195421


namespace Lee_charge_per_lawn_l1954_195491

theorem Lee_charge_per_lawn
  (x : ℝ)
  (mowed_lawns : ℕ)
  (total_earned : ℝ)
  (tips : ℝ)
  (tip_amount : ℝ)
  (num_customers_tipped : ℕ)
  (earnings_from_mowing : ℝ)
  (total_earning_with_tips : ℝ) :
  mowed_lawns = 16 →
  total_earned = 558 →
  num_customers_tipped = 3 →
  tip_amount = 10 →
  tips = num_customers_tipped * tip_amount →
  earnings_from_mowing = mowed_lawns * x →
  total_earning_with_tips = earnings_from_mowing + tips →
  total_earning_with_tips = total_earned →
  x = 33 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end Lee_charge_per_lawn_l1954_195491


namespace evaluate_fraction_sum_l1954_195482

-- Define the problem conditions and target equation
theorem evaluate_fraction_sum
    (p q r : ℝ)
    (h : p / (30 - p) + q / (75 - q) + r / (45 - r) = 8) :
    6 / (30 - p) + 15 / (75 - q) + 9 / (45 - r) = 11 / 5 := by
  sorry

end evaluate_fraction_sum_l1954_195482


namespace limsup_subset_l1954_195461

variable {Ω : Type*} -- assuming a universal sample space Ω for the events A_n and B_n

def limsup (A : ℕ → Set Ω) : Set Ω := 
  ⋂ k, ⋃ n ≥ k, A n

theorem limsup_subset {A B : ℕ → Set Ω} (h : ∀ n, A n ⊆ B n) : 
  limsup A ⊆ limsup B :=
by
  -- here goes the proof
  sorry

end limsup_subset_l1954_195461


namespace mixed_solution_concentration_l1954_195419

-- Defining the conditions as given in the question
def weight1 : ℕ := 200
def concentration1 : ℕ := 25
def saltInFirstSolution : ℕ := (concentration1 * weight1) / 100

def weight2 : ℕ := 300
def saltInSecondSolution : ℕ := 60

def totalSalt : ℕ := saltInFirstSolution + saltInSecondSolution
def totalWeight : ℕ := weight1 + weight2

-- Statement of the proof
theorem mixed_solution_concentration :
  ((totalSalt : ℚ) / (totalWeight : ℚ)) * 100 = 22 :=
by
  sorry

end mixed_solution_concentration_l1954_195419


namespace Roger_first_bag_candies_is_11_l1954_195458

-- Define the conditions
def Sandra_bags : ℕ := 2
def Sandra_candies_per_bag : ℕ := 6
def Roger_bags : ℕ := 2
def Roger_second_bag_candies : ℕ := 3
def Extra_candies_Roger_has_than_Sandra : ℕ := 2

-- Define the total candy for Sandra
def Sandra_total_candies : ℕ := Sandra_bags * Sandra_candies_per_bag

-- Using the conditions, we define the total candy for Roger
def Roger_total_candies : ℕ := Sandra_total_candies + Extra_candies_Roger_has_than_Sandra

-- Define the candy in Roger's first bag
def Roger_first_bag_candies : ℕ := Roger_total_candies - Roger_second_bag_candies

-- The proof statement we need to prove
theorem Roger_first_bag_candies_is_11 : Roger_first_bag_candies = 11 := by
  sorry

end Roger_first_bag_candies_is_11_l1954_195458


namespace items_left_in_store_l1954_195453

theorem items_left_in_store: (4458 - 1561) + 575 = 3472 :=
by 
  sorry

end items_left_in_store_l1954_195453


namespace bridge_length_is_correct_l1954_195460

noncomputable def speed_km_per_hour_to_m_per_s (speed_kmph : ℝ) : ℝ :=
  speed_kmph * (1000 / 3600)

noncomputable def total_distance_covered (speed_m_per_s time_s : ℝ) : ℝ :=
  speed_m_per_s * time_s

def bridge_length (total_distance train_length : ℝ) : ℝ :=
  total_distance - train_length

theorem bridge_length_is_correct : 
  let train_length := 110 
  let speed_kmph := 72
  let time_s := 12.099
  let speed_m_per_s := speed_km_per_hour_to_m_per_s speed_kmph
  let total_distance := total_distance_covered speed_m_per_s time_s
  bridge_length total_distance train_length = 131.98 := 
by
  sorry

end bridge_length_is_correct_l1954_195460


namespace loaf_bread_cost_correct_l1954_195444

-- Given conditions
def total : ℕ := 32
def candy_bar : ℕ := 2
def final_remaining : ℕ := 18

-- Intermediate calculations as definitions
def remaining_after_candy_bar : ℕ := total - candy_bar
def turkey_cost : ℕ := remaining_after_candy_bar / 3
def remaining_after_turkey : ℕ := remaining_after_candy_bar - turkey_cost
def loaf_bread_cost : ℕ := remaining_after_turkey - final_remaining

-- Theorem stating the problem question and expected answer
theorem loaf_bread_cost_correct : loaf_bread_cost = 2 :=
sorry

end loaf_bread_cost_correct_l1954_195444


namespace prime_square_mod_12_l1954_195403

theorem prime_square_mod_12 (p : ℕ) (h_prime : Nat.Prime p) (h_ne2 : p ≠ 2) (h_ne3 : p ≠ 3) :
    (∃ n : ℤ, p = 6 * n + 1 ∨ p = 6 * n + 5) → (p^2 % 12 = 1) := by
  sorry

end prime_square_mod_12_l1954_195403


namespace compute_v_l1954_195447

variable (a b c : ℝ)

theorem compute_v (H1 : (a * c) / (a + b) + (b * a) / (b + c) + (c * b) / (c + a) = -8)
                  (H2 : (b * c) / (a + b) + (c * a) / (b + c) + (a * b) / (c + a) = 12)
                  (H3 : a * b * c = 1) :
  (b / (a + b) + c / (b + c) + a / (c + a)) = -8.5 :=
sorry

end compute_v_l1954_195447


namespace apple_tree_production_l1954_195420

def first_year_production : ℕ := 40
def second_year_production (first_year_production : ℕ) : ℕ := 2 * first_year_production + 8
def third_year_production (second_year_production : ℕ) : ℕ := second_year_production - (second_year_production / 4)
def total_production (first_year_production second_year_production third_year_production : ℕ) : ℕ :=
    first_year_production + second_year_production + third_year_production

-- Proof statement
theorem apple_tree_production : total_production 40 88 66 = 194 := by
  sorry

end apple_tree_production_l1954_195420


namespace no_common_root_of_polynomials_l1954_195493

theorem no_common_root_of_polynomials (a b c d : ℝ) (h : 0 < a ∧ a < b ∧ b < c ∧ c < d) : 
  ∀ x : ℝ, ¬ (x^2 + b*x + c = 0 ∧ x^2 + a*x + d = 0) :=
by
  intro x
  sorry

end no_common_root_of_polynomials_l1954_195493


namespace min_value_function_l1954_195428

theorem min_value_function (x y : ℝ) (h1 : -2 < x ∧ x < 2) (h2 : -2 < y ∧ y < 2) (h3 : x * y = -1) :
  ∃ u : ℝ, u = (4 / (4 - x^2)) + (9 / (9 - y^2)) ∧ u = 12 / 5 :=
by
  sorry

end min_value_function_l1954_195428


namespace find_first_term_l1954_195406

variable {a : ℕ → ℕ}

-- Given conditions
axiom arithmetic_seq : ∀ n : ℕ, a (n + 1) + a n = 4 * n

-- Question to prove
theorem find_first_term : a 0 = 1 :=
sorry

end find_first_term_l1954_195406


namespace sahil_selling_price_l1954_195441

-- Define the conditions
def purchased_price := 9000
def repair_cost := 5000
def transportation_charges := 1000
def profit_percentage := 50 / 100

-- Calculate the total cost
def total_cost := purchased_price + repair_cost + transportation_charges

-- Calculate the selling price
def selling_price := total_cost + (profit_percentage * total_cost)

-- The theorem to prove the selling price
theorem sahil_selling_price : selling_price = 22500 :=
by
  -- This is where the proof would go, but we skip it with sorry.
  sorry

end sahil_selling_price_l1954_195441


namespace median_on_AB_eq_altitude_on_BC_eq_perp_bisector_on_AC_eq_l1954_195443

-- Definition of points A, B, and C
def A : ℝ × ℝ := (4, 0)
def B : ℝ × ℝ := (6, 7)
def C : ℝ × ℝ := (0, 3)

-- The problem statements as Lean theorems
theorem median_on_AB_eq : ∀ (A B : ℝ × ℝ), A = (4, 0) ∧ B = (6, 7) → ∃ (x y : ℝ), x - 10 * y + 30 = 0 := by
  intros
  sorry

theorem altitude_on_BC_eq : ∀ (B C : ℝ × ℝ), B = (6, 7) ∧ C = (0, 3) → ∃ (x y : ℝ), 3 * x + 2 * y - 12 = 0 := by
  intros
  sorry

theorem perp_bisector_on_AC_eq : ∀ (A C : ℝ × ℝ), A = (4, 0) ∧ C = (0, 3) → ∃ (x y : ℝ), 8 * x - 6 * y - 7 = 0 := by
  intros
  sorry

end median_on_AB_eq_altitude_on_BC_eq_perp_bisector_on_AC_eq_l1954_195443


namespace arccos_range_l1954_195455

theorem arccos_range (a : ℝ) (x : ℝ) (h1 : x = Real.sin a) (h2 : a ∈ Set.Icc (-Real.pi / 4) (3 * Real.pi / 4)) :
  Set.Icc 0 (3 * Real.pi / 4) = Set.image Real.arccos (Set.Icc (-Real.sqrt 2 / 2) 1) :=
by
  sorry

end arccos_range_l1954_195455


namespace equivalent_sum_of_exponents_l1954_195440

theorem equivalent_sum_of_exponents : 3^3 + 3^3 + 3^3 = 3^4 :=
by
  sorry

end equivalent_sum_of_exponents_l1954_195440


namespace molly_ate_11_suckers_l1954_195430

/-- 
Sienna gave Bailey half of her suckers.
Jen ate 11 suckers and gave the rest to Molly.
Molly ate some suckers and gave the rest to Harmony.
Harmony kept 3 suckers and passed the remainder to Taylor.
Taylor ate one and gave the last 5 suckers to Callie.
How many suckers did Molly eat?
-/
theorem molly_ate_11_suckers
  (sienna_bailey_suckers : ℕ)
  (jen_ate : ℕ)
  (jens_remainder_to_molly : ℕ)
  (molly_remainder_to_harmony : ℕ) 
  (harmony_kept : ℕ) 
  (harmony_remainder_to_taylor : ℕ)
  (taylor_ate : ℕ)
  (taylor_remainder_to_callie : ℕ)
  (jen_condition : jen_ate = 11)
  (harmony_condition : harmony_kept = 3)
  (taylor_condition : taylor_ate = 1)
  (taylor_final_suckers : taylor_remainder_to_callie = 5) :
  molly_ate = 11 :=
by sorry

end molly_ate_11_suckers_l1954_195430


namespace symmetric_line_l1954_195488

theorem symmetric_line (x y : ℝ) : 
  (∀ (x y  : ℝ), 2 * x + y - 1 = 0) ∧ (∀ (x  : ℝ), x = 1) → (2 * x - y - 3 = 0) :=
by
  sorry

end symmetric_line_l1954_195488


namespace find_a_and_b_l1954_195424

variable {x : ℝ}

/-- The problem statement: Given the function y = b + a * sin x (with a < 0), and the maximum value is -1, and the minimum value is -5,
    find the values of a and b. --/
theorem find_a_and_b (a b : ℝ) (h : a < 0) 
  (h1 : ∀ x, b + a * Real.sin x ≤ -1)
  (h2 : ∀ x, b + a * Real.sin x ≥ -5) : 
  a = -2 ∧ b = -3 := sorry

end find_a_and_b_l1954_195424


namespace sets_of_consecutive_integers_summing_to_20_l1954_195431

def sum_of_consecutive_integers (a n : ℕ) : ℕ := n * a + (n * (n - 1)) / 2

theorem sets_of_consecutive_integers_summing_to_20 : 
  (∃ (a n : ℕ), n ≥ 2 ∧ sum_of_consecutive_integers a n = 20) ∧ 
  (∀ (a1 n1 a2 n2 : ℕ), 
    (n1 ≥ 2 ∧ sum_of_consecutive_integers a1 n1 = 20 ∧ 
    n2 ≥ 2 ∧ sum_of_consecutive_integers a2 n2 = 20) → 
    (a1 = a2 ∧ n1 = n2)) :=
sorry

end sets_of_consecutive_integers_summing_to_20_l1954_195431


namespace area_excluding_holes_l1954_195410

theorem area_excluding_holes (x : ℝ) :
  let A_large : ℝ := (x + 8) * (x + 6)
  let A_hole : ℝ := (2 * x - 4) * (x - 3)
  A_large - 2 * A_hole = -3 * x^2 + 34 * x + 24 := by
  sorry

end area_excluding_holes_l1954_195410


namespace statement_a_correct_statement_b_correct_l1954_195466

open Real

theorem statement_a_correct (a b c : ℝ) (ha : a > b) (hc : c < 0) : a + c > b + c := by
  sorry

theorem statement_b_correct (a b : ℝ) (ha : a > b) (hb : b > 0) : (a + b) / 2 > Real.sqrt (a * b) := by
  sorry

end statement_a_correct_statement_b_correct_l1954_195466


namespace correct_operations_l1954_195417

theorem correct_operations : 
  (∀ x y : ℝ, x^2 + x^4 ≠ x^6) ∧
  (∀ x y : ℝ, 2*x + 4*y ≠ 6*x*y) ∧
  (∀ x : ℝ, x^6 / x^3 = x^3) ∧
  (∀ x : ℝ, (x^3)^2 = x^6) :=
by 
  sorry

end correct_operations_l1954_195417


namespace remaining_volume_correct_l1954_195456

noncomputable def diameter_sphere : ℝ := 24
noncomputable def radius_sphere : ℝ := diameter_sphere / 2
noncomputable def height_hole1 : ℝ := 10
noncomputable def diameter_hole1 : ℝ := 3
noncomputable def radius_hole1 : ℝ := diameter_hole1 / 2
noncomputable def height_hole2 : ℝ := 10
noncomputable def diameter_hole2 : ℝ := 3
noncomputable def radius_hole2 : ℝ := diameter_hole2 / 2
noncomputable def height_hole3 : ℝ := 5
noncomputable def diameter_hole3 : ℝ := 4
noncomputable def radius_hole3 : ℝ := diameter_hole3 / 2

noncomputable def volume_sphere : ℝ := (4 / 3) * Real.pi * (radius_sphere ^ 3)
noncomputable def volume_hole1 : ℝ := Real.pi * (radius_hole1 ^ 2) * height_hole1
noncomputable def volume_hole2 : ℝ := Real.pi * (radius_hole2 ^ 2) * height_hole2
noncomputable def volume_hole3 : ℝ := Real.pi * (radius_hole3 ^ 2) * height_hole3

noncomputable def remaining_volume : ℝ := 
  volume_sphere - (2 * volume_hole1 + volume_hole3)

theorem remaining_volume_correct : remaining_volume = 2239 * Real.pi := by
  sorry

end remaining_volume_correct_l1954_195456


namespace new_perimeter_of_rectangle_l1954_195464

theorem new_perimeter_of_rectangle (w : ℝ) (A : ℝ) (new_area_factor : ℝ) (L : ℝ) (L' : ℝ) (P' : ℝ) 
  (h_w : w = 10) (h_A : A = 150) (h_new_area_factor: new_area_factor = 4 / 3)
  (h_orig_length : L = A / w) (h_new_area: A' = new_area_factor * A) (h_A' : A' = 200)
  (h_new_length : L' = A' / w) (h_perimeter : P' = 2 * (L' + w)) 
  : P' = 60 :=
sorry

end new_perimeter_of_rectangle_l1954_195464


namespace sum_of_all_ks_l1954_195408

theorem sum_of_all_ks (k : ℤ) :
  (∃ r s : ℤ, r ≠ s ∧ (3 * x^2 - k * x + 12 = 0) ∧ (r + s = k / 3) ∧ (r * s = 4)) → (k = 15 ∨ k = -15) → k + k = 0 :=
by
  sorry

end sum_of_all_ks_l1954_195408


namespace last_8_digits_of_product_l1954_195400

theorem last_8_digits_of_product :
  let p := 11 * 101 * 1001 * 10001 * 1000001 * 111
  (p % 100000000) = 87654321 :=
by
  let p := 11 * 101 * 1001 * 10001 * 1000001 * 111
  have : p % 100000000 = 87654321 := sorry
  exact this

end last_8_digits_of_product_l1954_195400


namespace factor_expression_l1954_195414

theorem factor_expression (a b c d : ℝ) : 
  a * (b - c)^3 + b * (c - d)^3 + c * (d - a)^3 + d * (a - b)^3 
        = ((a - b) * (b - c) * (c - d) * (d - a)) * (a + b + c + d) := 
by
  sorry

end factor_expression_l1954_195414


namespace bottle_caps_remaining_l1954_195426

-- Define the problem using the conditions and the desired proof.
theorem bottle_caps_remaining (original_count removed_count remaining_count : ℕ) 
    (h_original : original_count = 87) 
    (h_removed : removed_count = 47)
    (h_remaining : remaining_count = original_count - removed_count) :
    remaining_count = 40 :=
by 
  rw [h_original, h_removed] at h_remaining 
  exact h_remaining

end bottle_caps_remaining_l1954_195426


namespace side_length_of_S2_l1954_195477

theorem side_length_of_S2 (r s : ℝ) 
  (h1 : 2 * r + s = 2025) 
  (h2 : 2 * r + 3 * s = 3320) :
  s = 647.5 :=
by {
  -- proof omitted
  sorry
}

end side_length_of_S2_l1954_195477


namespace combined_swim_time_l1954_195438

theorem combined_swim_time 
    (freestyle_time: ℕ)
    (backstroke_without_factors: ℕ)
    (backstroke_with_factors: ℕ)
    (butterfly_without_factors: ℕ)
    (butterfly_with_factors: ℕ)
    (breaststroke_without_factors: ℕ)
    (breaststroke_with_factors: ℕ) :
    freestyle_time = 48 ∧
    backstroke_without_factors = freestyle_time + 4 ∧
    backstroke_with_factors = backstroke_without_factors + 2 ∧
    butterfly_without_factors = backstroke_without_factors + 3 ∧
    butterfly_with_factors = butterfly_without_factors + 3 ∧
    breaststroke_without_factors = butterfly_without_factors + 2 ∧
    breaststroke_with_factors = breaststroke_without_factors - 1 →
    freestyle_time + backstroke_with_factors + butterfly_with_factors + breaststroke_with_factors = 216 :=
by
  sorry

end combined_swim_time_l1954_195438


namespace find_r_cubed_and_reciprocal_cubed_l1954_195475

variable (r : ℝ)
variable (h : (r + 1 / r) ^ 2 = 5)

theorem find_r_cubed_and_reciprocal_cubed (r : ℝ) (h : (r + 1 / r) ^ 2 = 5) : r ^ 3 + 1 / r ^ 3 = 2 * Real.sqrt 5 := by
  sorry

end find_r_cubed_and_reciprocal_cubed_l1954_195475


namespace quadratic_inequality_solution_l1954_195432

theorem quadratic_inequality_solution : 
  ∀ x : ℝ, (2 * x ^ 2 + 7 * x + 3 > 0) ↔ (x < -3 ∨ x > -0.5) :=
by
  sorry

end quadratic_inequality_solution_l1954_195432


namespace least_n_divisible_by_some_not_all_l1954_195462

theorem least_n_divisible_by_some_not_all (n : ℕ) (h : 1 ≤ n):
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ 10 ∧ k ∣ (n^2 - n)) ∧ ¬ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → k ∣ (n^2 - n)) ↔ n = 3 :=
by
  sorry

end least_n_divisible_by_some_not_all_l1954_195462


namespace circles_disjoint_l1954_195467

-- Definitions of the circles
def circleM (x y : ℝ) : Prop := x^2 + y^2 = 1
def circleN (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 1

-- Prove that the circles are disjoint
theorem circles_disjoint : 
  (¬ ∃ (x y : ℝ), circleM x y ∧ circleN x y) :=
by sorry

end circles_disjoint_l1954_195467


namespace valid_shirt_tie_combinations_l1954_195402

theorem valid_shirt_tie_combinations
  (num_shirts : ℕ)
  (num_ties : ℕ)
  (restricted_shirts : ℕ)
  (restricted_ties : ℕ)
  (h_shirts : num_shirts = 8)
  (h_ties : num_ties = 7)
  (h_restricted_shirts : restricted_shirts = 3)
  (h_restricted_ties : restricted_ties = 2) :
  num_shirts * num_ties - restricted_shirts * restricted_ties = 50 := by
  sorry

end valid_shirt_tie_combinations_l1954_195402


namespace remainder_415_pow_420_div_16_l1954_195465

theorem remainder_415_pow_420_div_16 : 415^420 % 16 = 1 := by
  sorry

end remainder_415_pow_420_div_16_l1954_195465


namespace intersection_of_sets_l1954_195494

noncomputable def setM : Set ℝ := { x | x + 1 > 0 }
noncomputable def setN : Set ℝ := { x | 2 * x - 1 < 0 }

theorem intersection_of_sets : setM ∩ setN = { x : ℝ | -1 < x ∧ x < 1 / 2 } := by
  sorry

end intersection_of_sets_l1954_195494


namespace tangent_line_is_tangent_l1954_195495

noncomputable def func1 (x : ℝ) : ℝ := x + 1 + Real.log x
noncomputable def func2 (a x : ℝ) : ℝ := a * x^2 + (a + 2) * x + 1

theorem tangent_line_is_tangent
  (a : ℝ) (h_tangent : ∃ x₀ : ℝ, func2 a x₀ = 2 * x₀ ∧ (deriv (func2 a) x₀ = 2))
  (deriv_eq : deriv func1 1 = 2)
  : a = 4 :=
by
  sorry

end tangent_line_is_tangent_l1954_195495


namespace initial_students_count_l1954_195422

theorem initial_students_count (n : ℕ) (T T' : ℚ)
    (h1 : T = n * 61.5)
    (h2 : T' = T - 24)
    (h3 : T' = (n - 1) * 64) :
  n = 16 :=
by
  sorry

end initial_students_count_l1954_195422


namespace find_n_l1954_195457

/-- Given a natural number n such that LCM(n, 12) = 48 and GCF(n, 12) = 8, prove that n = 32. -/
theorem find_n (n : ℕ) (h1 : Nat.lcm n 12 = 48) (h2 : Nat.gcd n 12 = 8) : n = 32 :=
sorry

end find_n_l1954_195457


namespace incorrect_statement_l1954_195407

def consecutive_interior_angles_are_supplementary (l1 l2 : ℝ) : Prop :=
  ∀ (θ₁ θ₂ : ℝ), θ₁ + θ₂ = 180 → l1 = l2

def alternate_interior_angles_are_equal (l1 l2 : ℝ) : Prop :=
  ∀ (θ₁ θ₂ : ℝ), θ₁ = θ₂ → l1 = l2

def corresponding_angles_are_equal (l1 l2 : ℝ) : Prop :=
  ∀ (θ₁ θ₂ : ℝ), θ₁ = θ₂ → l1 = l2

def complementary_angles (θ₁ θ₂ : ℝ) : Prop :=
  θ₁ + θ₂ = 90

def supplementary_angles (θ₁ θ₂ : ℝ) : Prop :=
  θ₁ + θ₂ = 180

theorem incorrect_statement :
  ¬ (∀ (θ₁ θ₂ : ℝ), θ₁ = θ₂ → l1 = l2) →
    consecutive_interior_angles_are_supplementary l1 l2 →
    alternate_interior_angles_are_equal l1 l2 →
    corresponding_angles_are_equal l1 l2 →
    (∀ (θ₁ θ₂ : ℝ), supplementary_angles θ₁ θ₂) →
    (∀ (θ₁ θ₂ : ℝ), complementary_angles θ₁ θ₂) :=
sorry

end incorrect_statement_l1954_195407


namespace sum_of_diagonal_elements_l1954_195451

/-- Odd numbers from 1 to 49 arranged in a 5x5 grid. -/
def table : ℕ → ℕ → ℕ
| 0, 0 => 1
| 0, 1 => 3
| 0, 2 => 5
| 0, 3 => 7
| 0, 4 => 9
| 1, 0 => 11
| 1, 1 => 13
| 1, 2 => 15
| 1, 3 => 17
| 1, 4 => 19
| 2, 0 => 21
| 2, 1 => 23
| 2, 2 => 25
| 2, 3 => 27
| 2, 4 => 29
| 3, 0 => 31
| 3, 1 => 33
| 3, 2 => 35
| 3, 3 => 37
| 3, 4 => 39
| 4, 0 => 41
| 4, 1 => 43
| 4, 2 => 45
| 4, 3 => 47
| 4, 4 => 49
| _, _ => 0

/-- Proof that the sum of five numbers chosen from the table such that no two of them are in the same row or column equals 125. -/
theorem sum_of_diagonal_elements : 
  (table 0 0 + table 1 1 + table 2 2 + table 3 3 + table 4 4) = 125 := by
  sorry

end sum_of_diagonal_elements_l1954_195451


namespace find_x_squared_plus_y_squared_l1954_195452

theorem find_x_squared_plus_y_squared (x y : ℝ) (h₁ : x * y = -8) (h₂ : x^2 * y + x * y^2 + 3 * x + 3 * y = 100) : x^2 + y^2 = 416 :=
sorry

end find_x_squared_plus_y_squared_l1954_195452


namespace football_game_cost_l1954_195489

theorem football_game_cost :
  ∀ (total_spent strategy_game_cost batman_game_cost football_game_cost : ℝ),
  total_spent = 35.52 →
  strategy_game_cost = 9.46 →
  batman_game_cost = 12.04 →
  total_spent - strategy_game_cost - batman_game_cost = football_game_cost →
  football_game_cost = 13.02 :=
by
  intros total_spent strategy_game_cost batman_game_cost football_game_cost h1 h2 h3 h4
  have : football_game_cost = 13.02 := sorry
  exact this

end football_game_cost_l1954_195489


namespace ratio_s_to_t_l1954_195492

theorem ratio_s_to_t (b : ℝ) (s t : ℝ)
  (h1 : s = -b / 10)
  (h2 : t = -b / 6) :
  s / t = 3 / 5 :=
by sorry

end ratio_s_to_t_l1954_195492


namespace pure_imaginary_a_zero_l1954_195423

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0

theorem pure_imaginary_a_zero (a : ℝ) (h : is_pure_imaginary (i / (1 + a * i))) : a = 0 :=
sorry

end pure_imaginary_a_zero_l1954_195423


namespace part_a_1_part_a_2_l1954_195411

noncomputable def P (x k : ℝ) := x^3 - k*x + 2

theorem part_a_1 (k : ℝ) (h : k = 5) : P 2 k = 0 :=
sorry

theorem part_a_2 {x : ℝ} : P x 5 = (x - 2) * (x^2 + 2*x - 1) :=
sorry

end part_a_1_part_a_2_l1954_195411


namespace bob_is_47_5_l1954_195496

def bob_age (a b : ℝ) := b = 3 * a - 20
def sum_of_ages (a b : ℝ) := b + a = 70

theorem bob_is_47_5 (a b : ℝ) (h1 : bob_age a b) (h2 : sum_of_ages a b) : b = 47.5 :=
by
  sorry

end bob_is_47_5_l1954_195496


namespace greatest_third_term_of_arithmetic_sequence_l1954_195405

def is_arithmetic_sequence (a b c d : ℤ) : Prop := (b - a = c - b) ∧ (c - b = d - c)

theorem greatest_third_term_of_arithmetic_sequence :
  ∃ a b c d : ℤ, a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  is_arithmetic_sequence a b c d ∧
  (a + b + c + d = 52) ∧
  (c = 17) :=
sorry

end greatest_third_term_of_arithmetic_sequence_l1954_195405


namespace frac_ab_eq_five_thirds_l1954_195446

theorem frac_ab_eq_five_thirds (a b : ℝ) (hb : b ≠ 0) (h : (a - b) / b = 2 / 3) : a / b = 5 / 3 :=
by
  sorry

end frac_ab_eq_five_thirds_l1954_195446


namespace temperature_at_midnight_l1954_195412

def morning_temp : ℝ := 30
def afternoon_increase : ℝ := 1
def midnight_decrease : ℝ := 7

theorem temperature_at_midnight : morning_temp + afternoon_increase - midnight_decrease = 24 := by
  sorry

end temperature_at_midnight_l1954_195412


namespace solve_for_a_l1954_195449

theorem solve_for_a (a : ℝ) 
  (h : (2 * a + 16 + (3 * a - 8)) / 2 = 89) : 
  a = 34 := 
sorry

end solve_for_a_l1954_195449


namespace abs_neg_three_l1954_195469

theorem abs_neg_three : abs (-3) = 3 :=
by
  sorry

end abs_neg_three_l1954_195469


namespace solution_of_abs_square_inequality_l1954_195484

def solution_set := {x : ℝ | (1 ≤ x ∧ x ≤ 3) ∨ x = -2}

theorem solution_of_abs_square_inequality (x : ℝ) :
  (abs (x^2 - 4) ≤ x + 2) ↔ (x ∈ solution_set) :=
by
  sorry

end solution_of_abs_square_inequality_l1954_195484


namespace chess_mixed_games_l1954_195445

theorem chess_mixed_games (W M : ℕ) (hW : W * (W - 1) / 2 = 45) (hM : M * (M - 1) / 2 = 190) : M * W = 200 :=
by
  sorry

end chess_mixed_games_l1954_195445


namespace sum_of_first_11_terms_l1954_195434

variable (a : ℕ → ℤ)
variable (d : ℤ)

-- Condition: the sequence is an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Conditions given in the problem
axiom h1 : a 1 + a 5 + a 9 = 39
axiom h2 : a 3 + a 7 + a 11 = 27
axiom h3 : is_arithmetic_sequence a d

-- Proof statement
theorem sum_of_first_11_terms : (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 + a 11) = 121 := 
sorry

end sum_of_first_11_terms_l1954_195434


namespace sufficient_but_not_necessary_l1954_195425

-- Define the equations of the lines
def line1 (a : ℝ) (x y : ℝ) : ℝ := 2 * x + a * y + 1
def line2 (a : ℝ) (x y : ℝ) : ℝ := (a - 1) * x + 3 * y - 2

-- Define the condition for parallel lines by comparing their slopes
def parallel_condition (a : ℝ) : Prop :=  (2 * 3 = a * (a - 1))

theorem sufficient_but_not_necessary (a : ℝ) : 3 ≤ a :=
  sorry

end sufficient_but_not_necessary_l1954_195425


namespace Trent_traveled_distance_l1954_195415

variable (blocks_length : ℕ := 50)
variables (walking_blocks : ℕ := 4) (bus_blocks : ℕ := 7) (bicycle_blocks : ℕ := 5)
variables (walking_round_trip : ℕ := 2 * walking_blocks * blocks_length)
variables (bus_round_trip : ℕ := 2 * bus_blocks * blocks_length)
variables (bicycle_round_trip : ℕ := 2 * bicycle_blocks * blocks_length)

def total_distance_traveleed : ℕ :=
  walking_round_trip + bus_round_trip + bicycle_round_trip

theorem Trent_traveled_distance :
  total_distance_traveleed = 1600 := by
    sorry

end Trent_traveled_distance_l1954_195415


namespace is_divisible_by_N2_l1954_195498

def are_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

noncomputable def eulers_totient (n : ℕ) : ℕ :=
  Nat.totient n

theorem is_divisible_by_N2 (N1 N2 : ℕ) (h_coprime : are_coprime N1 N2) 
  (k := eulers_totient N2) : 
  (N1 ^ k - 1) % N2 = 0 :=
by
  sorry

end is_divisible_by_N2_l1954_195498


namespace smallest_number_of_groups_l1954_195448

theorem smallest_number_of_groups
  (participants : ℕ)
  (max_group_size : ℕ)
  (h1 : participants = 36)
  (h2 : max_group_size = 12) :
  participants / max_group_size = 3 :=
by
  sorry

end smallest_number_of_groups_l1954_195448


namespace solve_m_l1954_195483

theorem solve_m (m : ℝ) :
  (∃ x > 0, (2 * m - 4) ^ 2 = x ∧ (3 * m - 1) ^ 2 = x) →
  (m = -3 ∨ m = 1) :=
by 
  sorry

end solve_m_l1954_195483


namespace sin_pi_minus_2alpha_l1954_195471

theorem sin_pi_minus_2alpha (α : ℝ) (h1 : Real.sin (π / 2 + α) = -3 / 5) (h2 : π / 2 < α ∧ α < π) : 
  Real.sin (π - 2 * α) = -24 / 25 := by
  sorry

end sin_pi_minus_2alpha_l1954_195471


namespace roots_of_polynomial_l1954_195478

def P (x : ℝ) : ℝ := x^3 - 3 * x^2 - x + 3

theorem roots_of_polynomial :
  {x : ℝ | P x = 0} = {1, -1, 3} := 
sorry

end roots_of_polynomial_l1954_195478


namespace trapezoid_area_l1954_195413

def trapezoid_diagonals_and_height (AC BD h : ℕ) :=
  (AC = 17) ∧ (BD = 113) ∧ (h = 15)

theorem trapezoid_area (AC BD h : ℕ) (area1 area2 : ℕ) 
  (H : trapezoid_diagonals_and_height AC BD h) :
  (area1 = 900 ∨ area2 = 780) :=
by
  sorry

end trapezoid_area_l1954_195413


namespace al_initial_amount_l1954_195429

theorem al_initial_amount
  (a b c : ℕ)
  (h₁ : a + b + c = 2000)
  (h₂ : 3 * a + 2 * b + 2 * c = 3500) :
  a = 500 :=
sorry

end al_initial_amount_l1954_195429


namespace minEmployees_correct_l1954_195481

noncomputable def minEmployees (seaTurtles birdMigration bothTurtlesBirds turtlesPlants allThree : ℕ) : ℕ :=
  let onlySeaTurtles := seaTurtles - (bothTurtlesBirds + turtlesPlants - allThree)
  let onlyBirdMigration := birdMigration - (bothTurtlesBirds + allThree - turtlesPlants)
  onlySeaTurtles + onlyBirdMigration + bothTurtlesBirds + turtlesPlants + allThree

theorem minEmployees_correct :
  minEmployees 120 90 30 50 15 = 245 := by
  sorry

end minEmployees_correct_l1954_195481


namespace frequency_of_scoring_l1954_195470

def shots : ℕ := 80
def goals : ℕ := 50
def frequency : ℚ := goals / shots

theorem frequency_of_scoring : frequency = 0.625 := by
  sorry

end frequency_of_scoring_l1954_195470


namespace range_of_m_l1954_195490

-- Define the conditions for p and q
def p (m : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ 
  (x₁^2 + 2 * m * x₁ + 1 = 0) ∧ (x₂^2 + 2 * m * x₂ + 1 = 0)

def q (m : ℝ) : Prop := ¬ ∃ x : ℝ, x^2 + 2 * (m-2) * x - 3 * m + 10 = 0

-- The main theorem
theorem range_of_m (m : ℝ) : (p m ∨ q m) ∧ ¬ (p m ∧ q m) ↔ 
  (m ≤ -2 ∨ (-1 ≤ m ∧ m < 3)) := 
by
  sorry

end range_of_m_l1954_195490


namespace geese_initial_formation_l1954_195473

theorem geese_initial_formation (G : ℕ) 
  (h1 : G / 2 + 4 = 12) : G = 16 := 
sorry

end geese_initial_formation_l1954_195473


namespace total_cost_correct_l1954_195468

-- Define the costs for each repair
def engine_labor_cost := 75 * 16
def engine_part_cost := 1200
def brake_labor_cost := 85 * 10
def brake_part_cost := 800
def tire_labor_cost := 50 * 4
def tire_part_cost := 600

-- Calculate the total costs
def engine_total_cost := engine_labor_cost + engine_part_cost
def brake_total_cost := brake_labor_cost + brake_part_cost
def tire_total_cost := tire_labor_cost + tire_part_cost

-- Calculate the total combined cost
def total_combined_cost := engine_total_cost + brake_total_cost + tire_total_cost

-- The theorem to prove
theorem total_cost_correct : total_combined_cost = 4850 := by
  sorry

end total_cost_correct_l1954_195468


namespace f_20_plus_f_neg20_l1954_195463

noncomputable def f (a b : ℝ) (x : ℝ) := a * x^4 + b * x^2 + 5

theorem f_20_plus_f_neg20 (a b : ℝ) (h : f a b 20 = 3) : f a b 20 + f a b (-20) = 6 := by
  sorry

end f_20_plus_f_neg20_l1954_195463


namespace triangle_area_proof_l1954_195439

def vector2 := ℝ × ℝ

def a : vector2 := (6, 3)
def b : vector2 := (-4, 5)

noncomputable def det (u v : vector2) : ℝ := u.1 * v.2 - u.2 * v.1

noncomputable def parallelogram_area (u v : vector2) : ℝ := |det u v|

noncomputable def triangle_area (u v : vector2) : ℝ := parallelogram_area u v / 2

theorem triangle_area_proof : triangle_area a b = 21 := 
by 
  sorry

end triangle_area_proof_l1954_195439


namespace distance_between_trees_l1954_195486

-- Variables representing the total length of the yard and the number of trees.
variable (length_of_yard : ℕ) (number_of_trees : ℕ)

-- The given conditions
def yard_conditions (length_of_yard number_of_trees : ℕ) :=
  length_of_yard = 700 ∧ number_of_trees = 26

-- The proof statement: If the yard is 700 meters long and there are 26 trees, 
-- then the distance between two consecutive trees is 28 meters.
theorem distance_between_trees (length_of_yard : ℕ) (number_of_trees : ℕ)
  (h : yard_conditions length_of_yard number_of_trees) : 
  (length_of_yard / (number_of_trees - 1)) = 28 := 
by
  sorry

end distance_between_trees_l1954_195486


namespace largest_number_l1954_195499

theorem largest_number (P Q R S T : ℕ) 
  (hP_digits_prime : ∃ p1 p2, P = 10 * p1 + p2 ∧ Prime P ∧ Prime (p1 + p2))
  (hQ_multiple_of_5 : Q % 5 = 0)
  (hR_odd_non_prime : Odd R ∧ ¬ Prime R)
  (hS_prime_square : ∃ p, Prime p ∧ S = p * p)
  (hT_mean_prime : T = (P + Q) / 2 ∧ Prime T)
  (hP_range : 10 ≤ P ∧ P ≤ 99)
  (hQ_range : 2 ≤ Q ∧ Q ≤ 19)
  (hR_range : 2 ≤ R ∧ R ≤ 19)
  (hS_range : 2 ≤ S ∧ S ≤ 19)
  (hT_range : 2 ≤ T ∧ T ≤ 19) :
  max P (max Q (max R (max S T))) = Q := 
by 
  sorry

end largest_number_l1954_195499


namespace product_of_possible_values_of_N_l1954_195474

theorem product_of_possible_values_of_N (N B D : ℤ) 
  (h1 : B = D - N) 
  (h2 : B + 10 - (D - 4) = 1 ∨ B + 10 - (D - 4) = -1) :
  N = 13 ∨ N = 15 → (13 * 15) = 195 :=
by sorry

end product_of_possible_values_of_N_l1954_195474


namespace geom_sequence_sum_l1954_195409

theorem geom_sequence_sum (S : ℕ → ℤ) (a : ℕ → ℤ) (r : ℤ) 
    (h1 : ∀ n : ℕ, n ≥ 1 → S n = 3^n + r) 
    (h2 : ∀ n : ℕ, n ≥ 2 → a n = S n - S (n - 1)) 
    (h3 : a 1 = S 1) :
  r = -1 := 
sorry

end geom_sequence_sum_l1954_195409


namespace down_payment_calculation_l1954_195476

noncomputable def tablet_price : ℝ := 450
noncomputable def installment_1 : ℝ := 4 * 40
noncomputable def installment_2 : ℝ := 4 * 35
noncomputable def installment_3 : ℝ := 4 * 30
noncomputable def total_savings : ℝ := 70
noncomputable def total_installments := tablet_price + total_savings
noncomputable def installment_payments := installment_1 + installment_2 + installment_3
noncomputable def down_payment := total_installments - installment_payments

theorem down_payment_calculation : down_payment = 100 := by
  unfold down_payment
  unfold total_installments
  unfold installment_payments
  unfold tablet_price
  unfold total_savings
  unfold installment_1
  unfold installment_2
  unfold installment_3
  sorry

end down_payment_calculation_l1954_195476


namespace total_pebbles_l1954_195450

theorem total_pebbles (white_pebbles : ℕ) (red_pebbles : ℕ)
  (h1 : white_pebbles = 20)
  (h2 : red_pebbles = white_pebbles / 2) :
  white_pebbles + red_pebbles = 30 := by
  sorry

end total_pebbles_l1954_195450


namespace parabola_symmetric_points_l1954_195472

-- Define the parabola and the symmetry condition
def parabola (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 1

def symmetric_points (P Q : ℝ × ℝ) : Prop :=
  P.1 + P.2 = 0 ∧ Q.1 + Q.2 = 0 ∧ P.1 = -Q.1 ∧ P.2 = -Q.2

-- Problem definition: Prove that if there exist symmetric points on the parabola, then a > 3/4
theorem parabola_symmetric_points (a : ℝ) :
  (∃ P Q : ℝ × ℝ, symmetric_points P Q ∧ parabola a P.1 = P.2 ∧ parabola a Q.1 = Q.2) → a > 3 / 4 :=
by
  sorry

end parabola_symmetric_points_l1954_195472


namespace maximum_b_value_l1954_195404

noncomputable def f (x : ℝ) := Real.exp x - x - 1
def g (x : ℝ) := -x^2 + 4 * x - 3

theorem maximum_b_value (a b : ℝ) (h1 : 0 ≤ a) (h2 : f a = g b) : b ≤ 3 := by
  sorry

end maximum_b_value_l1954_195404


namespace lambs_total_l1954_195454

/-
Each of farmer Cunningham's lambs is either black or white.
There are 193 white lambs, and 5855 black lambs.
Prove that the total number of lambs is 6048.
-/

theorem lambs_total (white_lambs : ℕ) (black_lambs : ℕ) (h1 : white_lambs = 193) (h2 : black_lambs = 5855) :
  white_lambs + black_lambs = 6048 :=
by
  -- proof goes here
  sorry

end lambs_total_l1954_195454
