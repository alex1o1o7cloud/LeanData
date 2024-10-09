import Mathlib

namespace road_signs_count_l945_94517

theorem road_signs_count (n1 n2 n3 n4 : ℕ) (h1 : n1 = 40) (h2 : n2 = n1 + n1 / 4) (h3 : n3 = 2 * n2) (h4 : n4 = n3 - 20) : 
  n1 + n2 + n3 + n4 = 270 := 
by
  sorry

end road_signs_count_l945_94517


namespace at_least_one_less_than_two_l945_94535

theorem at_least_one_less_than_two (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : 2 < a + b) :
  (1 + b) / a < 2 ∨ (1 + a) / b < 2 := 
by
  sorry

end at_least_one_less_than_two_l945_94535


namespace calculate_probability_l945_94562

theorem calculate_probability :
  let letters_in_bag : List Char := ['C', 'A', 'L', 'C', 'U', 'L', 'A', 'T', 'E']
  let target_letters : List Char := ['C', 'U', 'T']
  let total_outcomes := letters_in_bag.length
  let favorable_outcomes := (letters_in_bag.filter (λ c => c ∈ target_letters)).length
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ) = 4 / 9 := sorry

end calculate_probability_l945_94562


namespace all_numbers_non_positive_l945_94588

theorem all_numbers_non_positive 
  (a : ℕ → ℝ) 
  (n : ℕ) 
  (h0 : a 0 = 0) 
  (hn : a n = 0) 
  (h : ∀ k, 1 ≤ k → k ≤ n - 1 → (a (k - 1) - 2 * a k + a (k + 1) ≥ 0)) : 
  ∀ k, 0 ≤ k → k ≤ n → a k ≤ 0 := 
by 
  sorry

end all_numbers_non_positive_l945_94588


namespace total_tiles_l945_94550

/-- A square-shaped floor is covered with congruent square tiles. 
If the total number of tiles on the two diagonals is 88 and the floor 
forms a perfect square with an even side length, then the number of tiles 
covering the floor is 1936. -/
theorem total_tiles (n : ℕ) (hn_even : n % 2 = 0) (h_diag : 2 * n = 88) : n^2 = 1936 := 
by 
  sorry

end total_tiles_l945_94550


namespace least_number_subtracted_l945_94530

theorem least_number_subtracted (n : ℕ) (x : ℕ) (h_n : n = 4273981567) (h_x : x = 17) : 
  (n - x) % 25 = 0 := by
  sorry

end least_number_subtracted_l945_94530


namespace simplify_expression_l945_94538

variable (a : Real)

theorem simplify_expression : (-2 * a) * a - (-2 * a)^2 = -6 * a^2 := by
  sorry

end simplify_expression_l945_94538


namespace value_of_expression_l945_94502

theorem value_of_expression (x : ℝ) (h : 2 * x^2 + 3 * x + 7 = 8) : 4 * x^2 + 6 * x - 9 = -7 := by
  sorry

end value_of_expression_l945_94502


namespace two_f_one_lt_f_four_l945_94501

theorem two_f_one_lt_f_four
  (f : ℝ → ℝ)
  (h1 : ∀ x, f (x + 2) = f (-x - 2))
  (h2 : ∀ x, x > 2 → x * (deriv f x) > 2 * (deriv f x) + f x) :
  2 * f 1 < f 4 :=
sorry

end two_f_one_lt_f_four_l945_94501


namespace coprime_exists_pow_divisible_l945_94573

theorem coprime_exists_pow_divisible (a n : ℕ) (h_coprime : Nat.gcd a n = 1) : 
  ∃ m : ℕ, n ∣ a^m - 1 :=
by
  sorry

end coprime_exists_pow_divisible_l945_94573


namespace ratio_of_sugar_to_flour_l945_94591

theorem ratio_of_sugar_to_flour
  (F B : ℕ)
  (h1 : F = 10 * B)
  (h2 : F = 8 * (B + 60))
  (sugar : ℕ)
  (hs : sugar = 2000) :
  sugar / F = 5 / 6 :=
by {
  sorry -- proof omitted
}

end ratio_of_sugar_to_flour_l945_94591


namespace average_speed_of_tiger_exists_l945_94527

-- Conditions
def head_start_distance (v_t : ℝ) : ℝ := 5 * v_t
def zebra_distance : ℝ := 6 * 55
def tiger_distance (v_t : ℝ) : ℝ := 6 * v_t

-- Problem statement
theorem average_speed_of_tiger_exists (v_t : ℝ) (h : zebra_distance = head_start_distance v_t + tiger_distance v_t) : v_t = 30 :=
by
  sorry

end average_speed_of_tiger_exists_l945_94527


namespace donna_received_total_interest_l945_94541

-- Donna's investment conditions
def totalInvestment : ℝ := 33000
def investmentAt4Percent : ℝ := 13000
def investmentAt225Percent : ℝ := totalInvestment - investmentAt4Percent
def rate4Percent : ℝ := 0.04
def rate225Percent : ℝ := 0.0225

-- The interest calculation
def interestFrom4PercentInvestment : ℝ := investmentAt4Percent * rate4Percent
def interestFrom225PercentInvestment : ℝ := investmentAt225Percent * rate225Percent
def totalInterest : ℝ := interestFrom4PercentInvestment + interestFrom225PercentInvestment

-- The proof statement
theorem donna_received_total_interest :
  totalInterest = 970 := by
sorry

end donna_received_total_interest_l945_94541


namespace calc_price_per_litre_l945_94506

noncomputable def pricePerLitre (initial final totalCost : ℝ) : ℝ :=
  totalCost / (final - initial)

theorem calc_price_per_litre :
  pricePerLitre 10 50 36.60 = 91.5 :=
by
  sorry

end calc_price_per_litre_l945_94506


namespace sum_of_perimeters_l945_94528

theorem sum_of_perimeters (x y : ℝ) (h₁ : x^2 + y^2 = 125) (h₂ : x^2 - y^2 = 65) : 4 * x + 4 * y = 60 := 
by
  sorry

end sum_of_perimeters_l945_94528


namespace sum_even_numbers_l945_94593

def is_even (n : ℕ) : Prop := n % 2 = 0

def largest_even_less_than_or_equal (n m : ℕ) : ℕ :=
if h : m % 2 = 0 ∧ m ≤ n then m else
if h : m % 2 = 1 ∧ (m - 1) ≤ n then m - 1 else 0

def smallest_even_less_than_or_equal (n : ℕ) : ℕ :=
if h : 2 ≤ n then 2 else 0

theorem sum_even_numbers (n : ℕ) (h : n = 49) :
  largest_even_less_than_or_equal n 48 + smallest_even_less_than_or_equal n = 50 :=
by sorry

end sum_even_numbers_l945_94593


namespace division_correct_multiplication_correct_l945_94577

theorem division_correct : 400 / 5 = 80 := by
  sorry

theorem multiplication_correct : 230 * 3 = 690 := by
  sorry

end division_correct_multiplication_correct_l945_94577


namespace remaining_walking_time_is_30_l945_94552

-- Define all the given conditions
def total_distance_to_store : ℝ := 2.5
def distance_already_walked : ℝ := 1.0
def time_per_mile : ℝ := 20.0

-- Define the target remaining walking time
def remaining_distance : ℝ := total_distance_to_store - distance_already_walked
def remaining_time : ℝ := remaining_distance * time_per_mile

-- Prove the remaining walking time is 30 minutes
theorem remaining_walking_time_is_30 : remaining_time = 30 :=
by
  -- Formal proof would go here using corresponding Lean tactics
  sorry

end remaining_walking_time_is_30_l945_94552


namespace train_speed_is_72_kmph_l945_94511

noncomputable def train_length : ℝ := 110
noncomputable def bridge_length : ℝ := 112
noncomputable def crossing_time : ℝ := 11.099112071034318

theorem train_speed_is_72_kmph :
  let total_distance := train_length + bridge_length
  let speed_m_per_s := total_distance / crossing_time
  let speed_kmph := speed_m_per_s * 3.6
  speed_kmph = 72 :=
by
  sorry

end train_speed_is_72_kmph_l945_94511


namespace hikers_count_l945_94580

theorem hikers_count (B H K : ℕ) (h1 : H = B + 178) (h2 : K = B / 2) (h3 : H + B + K = 920) : H = 474 :=
by
  sorry

end hikers_count_l945_94580


namespace choir_minimum_members_l945_94547

theorem choir_minimum_members (n : ℕ) :
  (∃ k1, n = 8 * k1) ∧ (∃ k2, n = 9 * k2) ∧ (∃ k3, n = 10 * k3) → n = 360 :=
by
  sorry

end choir_minimum_members_l945_94547


namespace deck_of_1000_transformable_l945_94564

def shuffle (n : ℕ) (deck : List ℕ) : List ℕ :=
  -- Definition of the shuffle operation as described in the problem
  sorry

noncomputable def transformable_in_56_shuffles (n : ℕ) : Prop :=
  ∀ (initial final : List ℕ) (h₁ : initial.length = n) (h₂ : final.length = n),
  -- Prove that any initial arrangement can be transformed to any final arrangement in at most 56 shuffles
  sorry

theorem deck_of_1000_transformable : transformable_in_56_shuffles 1000 :=
  -- Implement the proof here
  sorry

end deck_of_1000_transformable_l945_94564


namespace certain_number_is_213_l945_94531

theorem certain_number_is_213 (x : ℝ) (h1 : x * 16 = 3408) (h2 : x * 1.6 = 340.8) : x = 213 :=
sorry

end certain_number_is_213_l945_94531


namespace find_other_discount_l945_94523

def other_discount (list_price final_price : ℝ) (first_discount : ℝ) (second_discount : ℝ) : Prop :=
  let price_after_first_discount := list_price - (first_discount / 100) * list_price
  final_price = price_after_first_discount - (second_discount / 100) * price_after_first_discount

theorem find_other_discount : 
  other_discount 70 59.22 10 6 :=
by
  sorry

end find_other_discount_l945_94523


namespace terminal_side_quadrant_l945_94503

theorem terminal_side_quadrant (α : ℝ) (k : ℤ) (hk : α = 45 + k * 180) :
  (∃ n : ℕ, k = 2 * n ∧ α = 45) ∨ (∃ n : ℕ, k = 2 * n + 1 ∧ α = 225) :=
sorry

end terminal_side_quadrant_l945_94503


namespace volume_of_tetrahedron_eq_20_l945_94548

noncomputable def volume_tetrahedron (a b c : ℝ) : ℝ :=
  1 / 3 * a * b * c

theorem volume_of_tetrahedron_eq_20 {x y z : ℝ} (h1 : x^2 + y^2 = 25) (h2 : y^2 + z^2 = 41) (h3 : z^2 + x^2 = 34) :
  volume_tetrahedron 3 4 5 = 20 :=
by
  sorry

end volume_of_tetrahedron_eq_20_l945_94548


namespace most_suitable_candidate_l945_94595

-- Definitions for variances
def variance_A := 3.4
def variance_B := 2.1
def variance_C := 2.5
def variance_D := 2.7

-- We start the theorem to state the most suitable candidate based on given variances and average scores.
theorem most_suitable_candidate :
  (variance_A = 3.4) ∧ (variance_B = 2.1) ∧ (variance_C = 2.5) ∧ (variance_D = 2.7) →
  true := 
by
  sorry

end most_suitable_candidate_l945_94595


namespace prime_factors_identity_l945_94599

theorem prime_factors_identity (w x y z k : ℕ) 
    (h : 2^w * 3^x * 5^y * 7^z * 11^k = 900) : 
      2 * w + 3 * x + 5 * y + 7 * z + 11 * k = 20 :=
by
  sorry

end prime_factors_identity_l945_94599


namespace initial_pigs_count_l945_94551

theorem initial_pigs_count (P : ℕ) (h1 : 2 + P + 6 + 3 + 5 + 2 = 21) : P = 3 :=
by
  sorry

end initial_pigs_count_l945_94551


namespace parallel_lines_condition_l945_94598

theorem parallel_lines_condition {a : ℝ} :
  (∀ x y : ℝ, a * x + 2 * y + 3 * a = 0) ∧ (∀ x y : ℝ, 3 * x + (a - 1) * y = a - 7) ↔ a = 3 :=
by
  sorry

end parallel_lines_condition_l945_94598


namespace probability_interval_l945_94584

theorem probability_interval (P_A P_B : ℚ) (h1 : P_A = 5/6) (h2 : P_B = 3/4) :
  ∃ p : ℚ, (5/12 ≤ p ∧ p ≤ 3/4) :=
sorry

end probability_interval_l945_94584


namespace polygon_a_largest_area_l945_94574

open Real

/-- Lean 4 statement to prove that Polygon A has the largest area among the given polygons -/
theorem polygon_a_largest_area :
  let area_polygon_a := 4 + 2 * (1 / 2 * 2 * 2)
  let area_polygon_b := 3 + 3 * (1 / 2 * 1 * 1)
  let area_polygon_c := 6
  let area_polygon_d := 5 + (1 / 2) * π * 1^2
  let area_polygon_e := 7
  area_polygon_a > area_polygon_b ∧
  area_polygon_a > area_polygon_c ∧
  area_polygon_a > area_polygon_d ∧
  area_polygon_a > area_polygon_e :=
by
  let area_polygon_a := 4 + 2 * (1 / 2 * 2 * 2)
  let area_polygon_b := 3 + 3 * (1 / 2 * 1 * 1)
  let area_polygon_c := 6
  let area_polygon_d := 5 + (1 / 2) * π * 1^2
  let area_polygon_e := 7
  sorry

end polygon_a_largest_area_l945_94574


namespace distributor_B_lower_avg_price_l945_94572

theorem distributor_B_lower_avg_price (p_1 p_2 : ℝ) (h : p_1 < p_2) :
  (p_1 + p_2) / 2 > (2 * p_1 * p_2) / (p_1 + p_2) :=
by {
  sorry
}

end distributor_B_lower_avg_price_l945_94572


namespace jack_correct_percentage_l945_94542

theorem jack_correct_percentage (y : ℝ) (h : y ≠ 0) :
  ((8 * y - (2 * y - 3)) / (8 * y)) * 100 = 75 + (75 / (2 * y)) :=
by
  sorry

end jack_correct_percentage_l945_94542


namespace total_eggs_found_l945_94597

def eggs_club_house := 12
def eggs_park := 5
def eggs_town_hall_garden := 3

theorem total_eggs_found : eggs_club_house + eggs_park + eggs_town_hall_garden = 20 :=
by
  sorry

end total_eggs_found_l945_94597


namespace JessicaPathsAvoidRiskySite_l945_94566

-- Definitions for the conditions.
def West (x y : ℕ) : Prop := (x > 0)
def East (x y : ℕ) : Prop := (x < 4)
def North (x y : ℕ) : Prop := (y < 3)
def AtOrigin (x y : ℕ) : Prop := (x = 0 ∧ y = 0)
def AtAnna (x y : ℕ) : Prop := (x = 4 ∧ y = 3)
def RiskySite (x y : ℕ) : Prop := (x = 2 ∧ y = 1)

-- Function to calculate binomial coefficient, binom(n, k)
def binom : ℕ → ℕ → ℕ
  | n, 0 => 1
  | 0, k + 1 => 0
  | n + 1, k + 1 => binom n k + binom n (k + 1)

-- Number of total valid paths avoiding the risky site.
theorem JessicaPathsAvoidRiskySite :
  let totalPaths := binom 7 4
  let pathsThroughRisky := binom 3 2 * binom 4 2
  (totalPaths - pathsThroughRisky) = 17 :=
by
  sorry

end JessicaPathsAvoidRiskySite_l945_94566


namespace option_B_is_equal_to_a_8_l945_94508

-- Statement: (a^2)^4 equals a^8
theorem option_B_is_equal_to_a_8 (a : ℝ) : (a^2)^4 = a^8 :=
by { sorry }

end option_B_is_equal_to_a_8_l945_94508


namespace find_n_from_sum_of_coeffs_l945_94529

-- The mathematical conditions and question translated to Lean

def sum_of_coefficients (n : ℕ) : ℕ := 6 ^ n
def binomial_coefficients_sum (n : ℕ) : ℕ := 2 ^ n

theorem find_n_from_sum_of_coeffs (n : ℕ) (M N : ℕ) (hM : M = sum_of_coefficients n) (hN : N = binomial_coefficients_sum n) (condition : M - N = 240) : n = 4 :=
by
  sorry

end find_n_from_sum_of_coeffs_l945_94529


namespace no_k_such_that_a_divides_2k_plus_1_and_b_divides_2k_minus_1_l945_94559

theorem no_k_such_that_a_divides_2k_plus_1_and_b_divides_2k_minus_1 :
  ∀ (a b n : ℕ), (a > 1) → (b > 1) → (a ∣ 2^n - 1) → (b ∣ 2^n + 1) → ∀ (k : ℕ), ¬ (a ∣ 2^k + 1 ∧ b ∣ 2^k - 1) :=
by
  intros a b n a_gt_1 b_gt_1 a_div_2n_minus_1 b_div_2n_plus_1 k
  sorry

end no_k_such_that_a_divides_2k_plus_1_and_b_divides_2k_minus_1_l945_94559


namespace train_speed_l945_94519

theorem train_speed (length : ℝ) (time : ℝ) (h_length : length = 3500) (h_time : time = 80) : 
  length / time = 43.75 := 
by 
  sorry

end train_speed_l945_94519


namespace jackie_walks_daily_l945_94567

theorem jackie_walks_daily (x : ℝ) :
  (∀ t : ℕ, t = 6 →
    6 * x = 6 * 1.5 + 3) →
  x = 2 :=
by
  sorry

end jackie_walks_daily_l945_94567


namespace determine_number_of_quarters_l945_94544

def number_of_coins (Q D : ℕ) : Prop := Q + D = 23

def total_value (Q D : ℕ) : Prop := 25 * Q + 10 * D = 335

theorem determine_number_of_quarters (Q D : ℕ) 
  (h1 : number_of_coins Q D) 
  (h2 : total_value Q D) : 
  Q = 7 :=
by
  -- Equating and simplifying using h2, we find 15Q = 105, hence Q = 7
  sorry

end determine_number_of_quarters_l945_94544


namespace min_of_x_squared_y_squared_z_squared_l945_94576

theorem min_of_x_squared_y_squared_z_squared (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 8) :
  x^2 + y^2 + z^2 ≥ 4 :=
by sorry

end min_of_x_squared_y_squared_z_squared_l945_94576


namespace remainder_of_product_mod_7_l945_94512

   theorem remainder_of_product_mod_7 :
     (7 * 17 * 27 * 37 * 47 * 57 * 67) % 7 = 0 := 
   by
     sorry
   
end remainder_of_product_mod_7_l945_94512


namespace reciprocal_opposite_neg_two_thirds_l945_94545

noncomputable def opposite (a : ℚ) : ℚ := -a
noncomputable def reciprocal (a : ℚ) : ℚ := 1 / a

theorem reciprocal_opposite_neg_two_thirds : reciprocal (opposite (-2 / 3)) = 3 / 2 :=
by sorry

end reciprocal_opposite_neg_two_thirds_l945_94545


namespace proof_problem_l945_94500

-- Necessary types and noncomputable definitions
noncomputable def a_seq : ℕ → ℕ := sorry
noncomputable def b_seq : ℕ → ℕ := sorry

-- The conditions in the problem are used as assumptions
axiom partition : ∀ (n : ℕ), n > 0 → a_seq n < a_seq (n + 1)
axiom b_def : ∀ (n : ℕ), n > 0 → b_seq n = a_seq n + n

-- The mathematical equivalent proof problem stated
theorem proof_problem (n : ℕ) (hn : n > 0) : a_seq n + b_seq n = a_seq (b_seq n) :=
sorry

end proof_problem_l945_94500


namespace total_seats_round_table_l945_94587

theorem total_seats_round_table (n : ℕ) (h : n = 38)
  (ka_position : ℕ) (sl_position : ℕ) 
  (h1 : ka_position = 10) 
  (h2 : sl_position = 29) 
  (h3 : (ka_position + n/2) % n = sl_position) : 
  n = 38 :=
by
  -- All steps and solution proof
  sorry

end total_seats_round_table_l945_94587


namespace sin_2012_eq_neg_sin_32_l945_94583

theorem sin_2012_eq_neg_sin_32 : Real.sin (2012 * Real.pi / 180) = - Real.sin (32 * Real.pi / 180) :=
by
  sorry

end sin_2012_eq_neg_sin_32_l945_94583


namespace percentage_of_women_in_study_group_l945_94526

theorem percentage_of_women_in_study_group
  (W : ℝ) -- percentage of women in decimal form
  (h1 : 0 < W ∧ W ≤ 1) -- percentage of women should be between 0 and 1
  (h2 : 0.4 * W = 0.32) -- 40 percent of women are lawyers, and probability is 0.32
  : W = 0.8 :=
  sorry

end percentage_of_women_in_study_group_l945_94526


namespace deny_evenness_l945_94594

-- We need to define the natural numbers and their parity.
variables {a b c : ℕ}

-- Define what it means for a number to be odd and even.
def is_odd (n : ℕ) := ∃ k, n = 2 * k + 1
def is_even (n : ℕ) := ∃ k, n = 2 * k

-- The Lean theorem statement translating the given problem.
theorem deny_evenness :
  (is_odd a ∧ is_odd b ∧ is_odd c) → ¬(is_even a ∨ is_even b ∨ is_even c) :=
by sorry

end deny_evenness_l945_94594


namespace money_per_postcard_l945_94533

def postcards_per_day : ℕ := 30
def days : ℕ := 6
def total_earning : ℕ := 900
def total_postcards := postcards_per_day * days
def price_per_postcard := total_earning / total_postcards

theorem money_per_postcard :
  price_per_postcard = 5 := 
sorry

end money_per_postcard_l945_94533


namespace bridgette_total_baths_l945_94549

def bridgette_baths (dogs baths_per_dog_per_month cats baths_per_cat_per_month birds baths_per_bird_per_month : ℕ) : ℕ :=
  (dogs * baths_per_dog_per_month * 12) + (cats * baths_per_cat_per_month * 12) + (birds * (12 / baths_per_bird_per_month))

theorem bridgette_total_baths :
  bridgette_baths 2 2 3 1 4 4 = 96 :=
by
  -- Proof omitted
  sorry

end bridgette_total_baths_l945_94549


namespace extremum_problem_l945_94513

def f (x a b : ℝ) := x^3 + a*x^2 + b*x + a^2

def f_prime (x a b : ℝ) := 3*x^2 + 2*a*x + b

theorem extremum_problem (a b : ℝ) 
  (cond1 : f_prime 1 a b = 0)
  (cond2 : f 1 a b = 10) :
  (a, b) = (4, -11) := 
sorry

end extremum_problem_l945_94513


namespace train_speed_l945_94585

theorem train_speed (length : ℝ) (time : ℝ)
  (length_pos : length = 160) (time_pos : time = 8) : 
  (length / time) * 3.6 = 72 :=
by
  sorry

end train_speed_l945_94585


namespace geometric_seq_a6_l945_94568

noncomputable def geometric_sequence (a : ℕ → ℝ) := ∃ q, ∀ n, a (n + 1) = a n * q

theorem geometric_seq_a6 {a : ℕ → ℝ} (h : geometric_sequence a) (h1 : a 1 * a 3 = 4) (h2 : a 4 = 4) : a 6 = 8 :=
sorry

end geometric_seq_a6_l945_94568


namespace height_of_wooden_box_l945_94524

theorem height_of_wooden_box 
  (height : ℝ)
  (h₁ : ∀ (length width : ℝ), length = 8 ∧ width = 10)
  (h₂ : ∀ (small_length small_width small_height : ℕ), small_length = 4 ∧ small_width = 5 ∧ small_height = 6)
  (h₃ : ∀ (num_boxes : ℕ), num_boxes = 4000000) :
  height = 6 := 
sorry

end height_of_wooden_box_l945_94524


namespace roberts_test_score_l945_94534

structure ClassState where
  num_students : ℕ
  avg_19_students : ℕ
  class_avg_20_students : ℕ

def calculate_roberts_score (s : ClassState) : ℕ :=
  let total_19_students := s.num_students * s.avg_19_students
  let total_20_students := (s.num_students + 1) * s.class_avg_20_students
  total_20_students - total_19_students

theorem roberts_test_score 
  (state : ClassState) 
  (h1 : state.num_students = 19) 
  (h2 : state.avg_19_students = 74)
  (h3 : state.class_avg_20_students = 75) : 
  calculate_roberts_score state = 94 := by
  sorry

end roberts_test_score_l945_94534


namespace range_of_independent_variable_l945_94509

theorem range_of_independent_variable (x : ℝ) : x ≠ -3 ↔ ∃ y : ℝ, y = 1 / (x + 3) :=
by 
  -- Proof is omitted
  sorry

end range_of_independent_variable_l945_94509


namespace log_squared_sum_eq_one_l945_94521

open Real

theorem log_squared_sum_eq_one :
  (log 2)^2 * log 250 + (log 5)^2 * log 40 = 1 := by
  sorry

end log_squared_sum_eq_one_l945_94521


namespace find_x_l945_94504

-- Define the conditions
def is_purely_imaginary (z : Complex) : Prop :=
  z.re = 0

-- Define the problem
theorem find_x (x : ℝ) (z : Complex) (h1 : z = Complex.ofReal (x^2 - 1) + Complex.I * (x + 1)) (h2 : is_purely_imaginary z) : x = 1 :=
sorry

end find_x_l945_94504


namespace evaluate_expression_l945_94581

theorem evaluate_expression (x y : ℕ) (h₁ : x = 3) (h₂ : y = 4) : 
  (x^5 + 3*y^2 + 7) / (x + 4) = 298 / 7 := by
  sorry

end evaluate_expression_l945_94581


namespace solution_set_inequality_l945_94560

noncomputable def solution_set := {x : ℝ | (x + 1) * (x - 2) ≤ 0 ∧ x ≠ -1}

theorem solution_set_inequality :
  solution_set = {x : ℝ | -1 < x ∧ x ≤ 2} :=
by {
-- Insert proof here
sorry
}

end solution_set_inequality_l945_94560


namespace projection_inequality_l945_94546

theorem projection_inequality
  (a b c : ℝ)
  (h : c^2 = a^2 + b^2) :
  c ≥ (a + b) / Real.sqrt 2 :=
by
  sorry

end projection_inequality_l945_94546


namespace trig_identity_l945_94553

open Real

theorem trig_identity (α β : ℝ) (h : cos α * cos β - sin α * sin β = 0) : sin α * cos β + cos α * sin β = 1 ∨ sin α * cos β + cos α * sin β = -1 :=
by
  sorry

end trig_identity_l945_94553


namespace find_starting_number_l945_94532

theorem find_starting_number : 
  ∃ x : ℕ, (∀ k : ℕ, (k < 12 → (x + 3 * k) ≤ 46) ∧ 12 = (46 - x) / 3 + 1) 
  ∧ x = 12 := 
by 
  sorry

end find_starting_number_l945_94532


namespace min_value_expression_l945_94540

open Real

theorem min_value_expression (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_condition : a * b * c = 1) :
  a^2 + 8 * a * b + 32 * b^2 + 24 * b * c + 8 * c^2 ≥ 36 :=
by
  sorry

end min_value_expression_l945_94540


namespace sqrt_six_estimation_l945_94510

theorem sqrt_six_estimation : 2 < Real.sqrt 6 ∧ Real.sqrt 6 < 3 :=
by 
  sorry

end sqrt_six_estimation_l945_94510


namespace coordinate_of_point_A_l945_94586

theorem coordinate_of_point_A (a b : ℝ) 
    (h1 : |b| = 3) 
    (h2 : |a| = 4) 
    (h3 : a > b) : 
    (a, b) = (4, 3) ∨ (a, b) = (4, -3) :=
by
    sorry

end coordinate_of_point_A_l945_94586


namespace find_value_of_x_l945_94561

theorem find_value_of_x (b : ℕ) (x : ℝ) (h_b_pos : b > 0) (h_x_pos : x > 0) 
  (h_r1 : r = 4 ^ (2 * b)) (h_r2 : r = 2 ^ b * x ^ b) : x = 8 :=
by
  -- Proof omitted for brevity
  sorry

end find_value_of_x_l945_94561


namespace find_other_endpoint_l945_94554

theorem find_other_endpoint (x_m y_m : ℤ) (x1 y1 : ℤ) 
(m_cond : x_m = (x1 + (-1)) / 2) (m_cond' : y_m = (y1 + (-4)) / 2) : 
(x_m, y_m) = (3, -1) ∧ (x1, y1) = (7, 2) → (-1, -4) = (-1, -4) :=
by
  sorry

end find_other_endpoint_l945_94554


namespace changfei_class_l945_94539

theorem changfei_class (m n : ℕ) (h : m * (m - 1) + m * n + n = 51) : m + n = 9 :=
sorry

end changfei_class_l945_94539


namespace integral_one_over_x_l945_94569

theorem integral_one_over_x:
  ∫ x in (1 : ℝ)..(Real.exp 1), 1 / x = 1 := 
by 
  sorry

end integral_one_over_x_l945_94569


namespace avg_ticket_cost_per_person_l945_94592

-- Define the conditions
def full_price : ℤ := 150
def half_price : ℤ := full_price / 2
def num_full_price_tickets : ℤ := 2
def num_half_price_tickets : ℤ := 2
def free_tickets : ℤ := 1
def total_people : ℤ := 5

-- Prove that the average cost of tickets per person is 90 yuan
theorem avg_ticket_cost_per_person : ((num_full_price_tickets * full_price + num_half_price_tickets * half_price) / total_people) = 90 := 
by 
  sorry

end avg_ticket_cost_per_person_l945_94592


namespace letter_addition_problem_l945_94578

theorem letter_addition_problem (S I X : ℕ) (E L V N : ℕ) 
  (hS : S = 8) 
  (hX_odd : X % 2 = 1)
  (h_diff_digits : ∀ (a b c d e f : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ e ∧ e ≠ f ∧ f ≠ a)
  (h_sum : 2 * S * 100 + 2 * I * 10 + 2 * X = E * 10000 + L * 1000 + E * 100 + V * 10 + E + N) :
  I = 3 :=
by
  sorry

end letter_addition_problem_l945_94578


namespace even_function_behavior_l945_94516

noncomputable def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

noncomputable def condition (f : ℝ → ℝ) : Prop :=
∀ x1 x2 : ℝ, x1 < 0 → x2 < 0 → x1 ≠ x2 → (x2 - x1) * (f x2 - f x1) > 0

theorem even_function_behavior (f : ℝ → ℝ) (h_even : is_even_function f) (h_condition : condition f) 
  (n : ℕ) (h_n : n > 0) : 
  f (n+1) < f (-n) ∧ f (-n) < f (n-1) :=
sorry

end even_function_behavior_l945_94516


namespace total_octopus_legs_l945_94589

-- Define the number of octopuses Carson saw
def num_octopuses : ℕ := 5

-- Define the number of legs per octopus
def legs_per_octopus : ℕ := 8

-- Define or state the theorem for total number of legs
theorem total_octopus_legs : num_octopuses * legs_per_octopus = 40 := by
  sorry

end total_octopus_legs_l945_94589


namespace determine_m_l945_94556

theorem determine_m {m : ℕ} : 
  (∃ (p : ℕ), p = 5 ∧ p = max (max (max 1 (1 + (m+1))) (3+1)) 4) → m = 3 := by
  sorry

end determine_m_l945_94556


namespace part1_part2_part3_l945_94505

-- Define the complex number z
def z (m : ℝ) : ℂ :=
  ⟨m^2 - 3*m + 2, m^2 - 1⟩  -- Note: This forms a complex number with real and imaginary parts

-- (1) Proof for z = 0 if and only if m = 1
theorem part1 (m : ℝ) : z m = 0 ↔ m = 1 :=
by sorry

-- (2) Proof for z being a pure imaginary number if and only if m = 2
theorem part2 (m : ℝ) : (z m).re = 0 ∧ (z m).im ≠ 0 ↔ m = 2 :=
by sorry

-- (3) Proof for the point corresponding to z being in the second quadrant if and only if 1 < m < 2
theorem part3 (m : ℝ) : (z m).re < 0 ∧ (z m).im > 0 ↔ 1 < m ∧ m < 2 :=
by sorry

end part1_part2_part3_l945_94505


namespace part1_part2_l945_94555

/- Define the function f(x) = |x-1| + |x-a| -/
def f (x a : ℝ) := abs (x - 1) + abs (x - a)

/- Part 1: Prove that if f(x) ≥ 2 implies the solution set {x | x ≤ 1/2 or x ≥ 5/2}, then a = 2 -/
theorem part1 (a : ℝ) (h : ∀ x : ℝ, f x a ≥ 2 → (x ≤ 1/2 ∨ x ≥ 5/2)) : a = 2 :=
  sorry

/- Part 2: Prove that for all x ∈ ℝ, f(x) + |x-1| ≥ 1 implies a ∈ [2, +∞) -/
theorem part2 (a : ℝ) (h : ∀ x : ℝ, f x a + abs (x - 1) ≥ 1) : 2 ≤ a :=
  sorry

end part1_part2_l945_94555


namespace spent_amount_l945_94522

def initial_amount : ℕ := 15
def final_amount : ℕ := 11

theorem spent_amount : initial_amount - final_amount = 4 :=
by
  sorry

end spent_amount_l945_94522


namespace height_of_wall_l945_94563

-- Definitions
def brick_length : ℝ := 25
def brick_width : ℝ := 11.25
def brick_height : ℝ := 6
def wall_length : ℝ := 850
def wall_width : ℝ := 22.5
def num_bricks : ℝ := 6800

-- Total volume of bricks
def total_brick_volume : ℝ := num_bricks * brick_length * brick_width * brick_height

-- Volume of the wall
def wall_volume (height : ℝ) : ℝ := wall_length * wall_width * height

-- Proof statement
theorem height_of_wall : ∃ h : ℝ, wall_volume h = total_brick_volume ∧ h = 600 := 
sorry

end height_of_wall_l945_94563


namespace union_sets_l945_94507

def A : Set ℕ := {1, 2, 4}
def B : Set ℕ := {2, 4, 6}

theorem union_sets : A ∪ B = {1, 2, 4, 6} := by
  sorry

end union_sets_l945_94507


namespace delta_epsilon_time_l945_94558

variable (D E Z h t : ℕ)

theorem delta_epsilon_time :
  (t = D - 8) →
  (t = E - 3) →
  (t = Z / 3) →
  (h = 3 * t) → 
  h = 15 / 8 :=
by
  intros h₁ h₂ h₃ h₄
  sorry

end delta_epsilon_time_l945_94558


namespace min_value_l945_94596

theorem min_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = a + b) : a + 4 * b ≥ 9 :=
sorry

end min_value_l945_94596


namespace gardner_bakes_brownies_l945_94525

theorem gardner_bakes_brownies : 
  ∀ (cookies cupcakes brownies students sweet_treats_per_student total_sweet_treats total_cookies_and_cupcakes : ℕ),
  cookies = 20 →
  cupcakes = 25 →
  students = 20 →
  sweet_treats_per_student = 4 →
  total_sweet_treats = students * sweet_treats_per_student →
  total_cookies_and_cupcakes = cookies + cupcakes →
  brownies = total_sweet_treats - total_cookies_and_cupcakes →
  brownies = 35 :=
by
  intros cookies cupcakes brownies students sweet_treats_per_student total_sweet_treats total_cookies_and_cupcakes
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end gardner_bakes_brownies_l945_94525


namespace abs_difference_l945_94570

theorem abs_difference (a b : ℝ) (h₁ : a * b = 9) (h₂ : a + b = 10) : |a - b| = 8 :=
sorry

end abs_difference_l945_94570


namespace surface_area_of_second_cube_l945_94514

theorem surface_area_of_second_cube (V1 V2: ℝ) (a2: ℝ):
  (V1 = 16 ∧ V2 = 4 * V1 ∧ a2 = (V2)^(1/3)) → 6 * a2^2 = 96 :=
by intros h; sorry

end surface_area_of_second_cube_l945_94514


namespace any_integer_amount_purchasable_amount_over_mn_minus_two_payable_l945_94590
open Int

variable (m n : ℕ) (h : Nat.gcd m n = 1)

theorem any_integer_amount_purchasable (x : ℤ) : 
  ∃ (a b : ℤ), a * n + b * m = x :=
by sorry

theorem amount_over_mn_minus_two_payable (k : ℤ) (hk : k > m * n - 2) : 
  ∃ (a b : ℤ), a * n + b * m = k :=
by sorry

end any_integer_amount_purchasable_amount_over_mn_minus_two_payable_l945_94590


namespace parallelogram_angle_A_l945_94520

theorem parallelogram_angle_A 
  (A B : ℝ) (h1 : A + B = 180) (h2 : A - B = 40) :
  A = 110 :=
by sorry

end parallelogram_angle_A_l945_94520


namespace complex_problem_l945_94557

open Complex

noncomputable def z : ℂ := (1 + I) / Real.sqrt 2

theorem complex_problem :
  1 + z^50 + z^100 = I := 
by
  -- Subproofs or transformations will be here.
  sorry

end complex_problem_l945_94557


namespace symmetric_point_x_axis_l945_94536

def symmetric_point (M : ℝ × ℝ) : ℝ × ℝ := (M.1, -M.2)

theorem symmetric_point_x_axis :
  ∀ (M : ℝ × ℝ), M = (3, -4) → symmetric_point M = (3, 4) :=
by
  intros M h
  rw [h]
  dsimp [symmetric_point]
  congr
  sorry

end symmetric_point_x_axis_l945_94536


namespace min_operator_result_l945_94515

theorem min_operator_result : 
  min ((-3) + (-6)) (min ((-3) - (-6)) (min ((-3) * (-6)) ((-3) / (-6)))) = -9 := 
by 
  sorry

end min_operator_result_l945_94515


namespace specified_time_eq_l945_94543

def distance : ℕ := 900
def ts (x : ℕ) : ℕ := x + 1
def tf (x : ℕ) : ℕ := x - 3

theorem specified_time_eq (x : ℕ) (h1 : x > 3) : 
  (distance / tf x) = 2 * (distance / ts x) :=
sorry

end specified_time_eq_l945_94543


namespace company_employees_after_reduction_l945_94579

theorem company_employees_after_reduction :
  let original_number := 224.13793103448276
  let reduction := 0.13 * original_number
  let current_number := original_number - reduction
  current_number = 195 :=
by
  let original_number := 224.13793103448276
  let reduction := 0.13 * original_number
  let current_number := original_number - reduction
  sorry

end company_employees_after_reduction_l945_94579


namespace count_six_digit_numbers_with_at_least_one_zero_l945_94518

theorem count_six_digit_numbers_with_at_least_one_zero : 
  900000 - 531441 = 368559 :=
by
  sorry

end count_six_digit_numbers_with_at_least_one_zero_l945_94518


namespace five_coins_total_cannot_be_30_cents_l945_94537

theorem five_coins_total_cannot_be_30_cents :
  ¬ ∃ (a b c d e : ℕ), 
  a + b + c + d + e = 5 ∧ 
  (a * 1 + b * 5 + c * 10 + d * 25 + e * 50) = 30 := 
sorry

end five_coins_total_cannot_be_30_cents_l945_94537


namespace ratio_amyl_alcohol_to_ethanol_l945_94575

noncomputable def mol_amyl_alcohol : ℕ := 3
noncomputable def mol_hcl : ℕ := 3
noncomputable def mol_ethanol : ℕ := 1
noncomputable def mol_h2so4 : ℕ := 1
noncomputable def mol_ch3_cl2_c5_h9 : ℕ := 3
noncomputable def mol_h2o : ℕ := 3
noncomputable def mol_ethyl_dimethylpropyl_sulfate : ℕ := 1

theorem ratio_amyl_alcohol_to_ethanol : 
  (mol_amyl_alcohol / mol_ethanol = 3) :=
by 
  have h1 : mol_amyl_alcohol = 3 := by rfl
  have h2 : mol_ethanol = 1 := by rfl
  sorry

end ratio_amyl_alcohol_to_ethanol_l945_94575


namespace rook_path_exists_l945_94582

theorem rook_path_exists :
  ∃ (path : Finset (Fin 8 × Fin 8)) (s1 s2 : Fin 8 × Fin 8),
  s1 ≠ s2 ∧
  s1.1 % 2 = s2.1 % 2 ∧ s1.2 % 2 = s2.2 % 2 ∧
  ∀ s : Fin 8 × Fin 8, s ∈ path ∧ s ≠ s2 :=
sorry

end rook_path_exists_l945_94582


namespace remainder_of_13_pow_a_mod_37_l945_94565

theorem remainder_of_13_pow_a_mod_37 (a : ℕ) (h_pos : a > 0) (h_mult : ∃ k : ℕ, a = 3 * k) : (13^a) % 37 = 1 := 
sorry

end remainder_of_13_pow_a_mod_37_l945_94565


namespace solve_problem_l945_94571

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem solve_problem :
  is_prime 2017 :=
by
  have h1 : 2017 > 1 := by linarith
  have h2 : ∀ m : ℕ, m ∣ 2017 → m = 1 ∨ m = 2017 :=
    sorry
  exact ⟨h1, h2⟩

end solve_problem_l945_94571
