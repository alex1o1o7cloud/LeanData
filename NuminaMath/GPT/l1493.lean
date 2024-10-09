import Mathlib

namespace multiplier_for_obsolete_books_l1493_149353

theorem multiplier_for_obsolete_books 
  (x : ℕ) 
  (total_books_removed number_of_damaged_books : ℕ) 
  (h1 : total_books_removed = 69) 
  (h2 : number_of_damaged_books = 11) 
  (h3 : number_of_damaged_books + (x * number_of_damaged_books - 8) = total_books_removed) 
  : x = 6 := 
by 
  sorry

end multiplier_for_obsolete_books_l1493_149353


namespace range_of_quadratic_function_l1493_149390

noncomputable def quadratic_function_range : Set ℝ :=
  { y : ℝ | ∃ x : ℝ, y = x^2 - 6 * x + 7 }

theorem range_of_quadratic_function :
  quadratic_function_range = { y : ℝ | y ≥ -2 } :=
by
  -- Insert proof here
  sorry

end range_of_quadratic_function_l1493_149390


namespace stations_equation_l1493_149338

theorem stations_equation (x : ℕ) (h : x * (x - 1) = 1482) : true :=
by
  sorry

end stations_equation_l1493_149338


namespace elizabeth_money_l1493_149352

theorem elizabeth_money :
  (∀ (P N : ℝ), P = 5 → N = 6 → 
    (P * 1.60 + N * 2.00) = 20.00) :=
by
  sorry

end elizabeth_money_l1493_149352


namespace flag_design_combinations_l1493_149380

-- Definitions
def colors : Nat := 3  -- Number of colors: purple, gold, and silver
def stripes : Nat := 3  -- Number of horizontal stripes in the flag

-- The Lean statement
theorem flag_design_combinations :
  (colors ^ stripes) = 27 :=
by
  sorry

end flag_design_combinations_l1493_149380


namespace base_conversion_problem_l1493_149339

variable (A C : ℕ)
variable (h1 : 0 ≤ A ∧ A < 8)
variable (h2 : 0 ≤ C ∧ C < 5)

theorem base_conversion_problem (h : 8 * A + C = 5 * C + A) : 8 * A + C = 39 := 
sorry

end base_conversion_problem_l1493_149339


namespace find_common_ratio_l1493_149394

variable {a : ℕ → ℝ}
variable {q : ℝ}

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * q

axiom a2 : a 2 = 9
axiom a3_plus_a4 : a 3 + a 4 = 18
axiom q_not_one : q ≠ 1

-- Proof problem
theorem find_common_ratio
  (h : is_geometric_sequence a q)
  (ha2 : a 2 = 9)
  (ha3a4 : a 3 + a 4 = 18)
  (hq : q ≠ 1) :
  q = -2 :=
sorry

end find_common_ratio_l1493_149394


namespace prob_kong_meng_is_one_sixth_l1493_149383

variable (bag : List String := ["孔", "孟", "之", "乡"])
variable (draws : List String := [])
def total_events : ℕ := 4 * 3
def favorable_events : ℕ := 2
def probability_kong_meng : ℚ := favorable_events / total_events

theorem prob_kong_meng_is_one_sixth :
  (probability_kong_meng = 1 / 6) :=
by
  sorry

end prob_kong_meng_is_one_sixth_l1493_149383


namespace percentage_increase_l1493_149386

theorem percentage_increase (S P : ℝ) (h1 : (S * (1 + P / 100)) * 0.8 = 1.04 * S) : P = 30 :=
by 
  sorry

end percentage_increase_l1493_149386


namespace price_of_table_l1493_149371

variable (C T : ℝ)

theorem price_of_table :
  2 * C + T = 0.6 * (C + 2 * T) ∧
  C + T = 96 →
  T = 84 := by
sorry

end price_of_table_l1493_149371


namespace primitive_root_exists_mod_pow_of_two_l1493_149361

theorem primitive_root_exists_mod_pow_of_two (n : ℕ) : 
  (∃ x : ℤ, ∀ k : ℕ, 1 ≤ k → x^k % (2^n) ≠ 1 % (2^n)) ↔ (n ≤ 2) := sorry

end primitive_root_exists_mod_pow_of_two_l1493_149361


namespace triangle_centroid_eq_l1493_149309

-- Define the proof problem
theorem triangle_centroid_eq
  (P Q R G : ℝ × ℝ) -- Points P, Q, R, and G (the centroid of the triangle PQR)
  (centroid_eq : G = ((P.1 + Q.1 + R.1) / 3, (P.2 + Q.2 + R.2) / 3)) -- Condition that G is the centroid
  (gp_sq_gq_sq_gr_sq_eq : dist G P ^ 2 + dist G Q ^ 2 + dist G R ^ 2 = 22) -- Given GP^2 + GQ^2 + GR^2 = 22
  : dist P Q ^ 2 + dist P R ^ 2 + dist Q R ^ 2 = 66 := -- Prove PQ^2 + PR^2 + QR^2 = 66
sorry -- Proof is omitted

end triangle_centroid_eq_l1493_149309


namespace distance_CD_l1493_149343

noncomputable def ellipse_eq (x y : ℝ) : Prop :=
  16 * (x + 2)^2 + 4 * y^2 = 64

def major_axis_distance : ℝ := 4
def minor_axis_distance : ℝ := 2

theorem distance_CD : ∃ (d : ℝ), 16 * (x + 2)^2 + 4 * y^2 = 64 → d = 2 * Real.sqrt 5 :=
by
  sorry

end distance_CD_l1493_149343


namespace number_in_central_region_l1493_149348

theorem number_in_central_region (a b c d : ℤ) :
  a + b + c + d = -4 →
  ∃ x : ℤ, x = -4 + 2 :=
by
  intros h
  use -2
  sorry

end number_in_central_region_l1493_149348


namespace max_value_y_eq_x_mul_2_minus_x_min_value_y_eq_x_plus_4_div_x_minus_3_l1493_149303

theorem max_value_y_eq_x_mul_2_minus_x (x : ℝ) (h : 0 < x ∧ x < 3 / 2) : ∃ y : ℝ, y = x * (2 - x) ∧ y ≤ 1 :=
sorry

theorem min_value_y_eq_x_plus_4_div_x_minus_3 (x : ℝ) (h : x > 3) : ∃ y : ℝ, y = x + 4 / (x - 3) ∧ y ≥ 7 :=
sorry

end max_value_y_eq_x_mul_2_minus_x_min_value_y_eq_x_plus_4_div_x_minus_3_l1493_149303


namespace washing_whiteboards_l1493_149396

/-- Define the conditions from the problem:
1. Four kids can wash three whiteboards in 20 minutes.
2. It takes one kid 160 minutes to wash a certain number of whiteboards. -/
def four_kids_wash_in_20_min : ℕ := 3
def time_per_batch : ℕ := 20
def one_kid_time : ℕ := 160
def intervals : ℕ := one_kid_time / time_per_batch

/-- Proving the answer based on the conditions:
one kid can wash six whiteboards in 160 minutes given these conditions. -/
theorem washing_whiteboards : intervals * (four_kids_wash_in_20_min / 4) = 6 :=
by
  sorry

end washing_whiteboards_l1493_149396


namespace similar_segments_areas_proportional_to_chords_squares_l1493_149373

variables {k k₁ Δ Δ₁ r r₁ a a₁ S S₁ : ℝ}

-- Conditions given in the problem
def similar_segments (r r₁ a a₁ Δ Δ₁ k k₁ : ℝ) :=
  (Δ / Δ₁ = (a^2 / a₁^2) ∧ (Δ / Δ₁ = r^2 / r₁^2)) ∧ (k / k₁ = r^2 / r₁^2)

-- Given the areas of the segments in terms of sectors and triangles
def area_of_segment (k Δ : ℝ) := k - Δ

-- Theorem statement proving the desired relationship
theorem similar_segments_areas_proportional_to_chords_squares
  (h : similar_segments r r₁ a a₁ Δ Δ₁ k k₁) :
  (S = area_of_segment k Δ) → (S₁ = area_of_segment k₁ Δ₁) → (S / S₁ = a^2 / a₁^2) :=
by
  sorry

end similar_segments_areas_proportional_to_chords_squares_l1493_149373


namespace find_packs_of_yellow_bouncy_balls_l1493_149370

noncomputable def packs_of_yellow_bouncy_balls (red_packs : ℕ) (balls_per_pack : ℕ) (extra_balls : ℕ) : ℕ :=
  (red_packs * balls_per_pack - extra_balls) / balls_per_pack

theorem find_packs_of_yellow_bouncy_balls :
  packs_of_yellow_bouncy_balls 5 18 18 = 4 := 
by
  sorry

end find_packs_of_yellow_bouncy_balls_l1493_149370


namespace white_pairs_coincide_l1493_149363

def triangles_in_each_half (red blue white: Nat) : Prop :=
  red = 5 ∧ blue = 6 ∧ white = 9

def folding_over_centerline (r_pairs b_pairs rw_pairs bw_pairs: Nat) : Prop :=
  r_pairs = 3 ∧ b_pairs = 2 ∧ rw_pairs = 3 ∧ bw_pairs = 1

theorem white_pairs_coincide
    (red_triangles blue_triangles white_triangles : Nat)
    (r_pairs b_pairs rw_pairs bw_pairs : Nat) :
    triangles_in_each_half red_triangles blue_triangles white_triangles →
    folding_over_centerline r_pairs b_pairs rw_pairs bw_pairs →
    ∃ coinciding_white_pairs, coinciding_white_pairs = 5 :=
by
  intros half_cond fold_cond
  sorry

end white_pairs_coincide_l1493_149363


namespace expression_equals_value_l1493_149355

theorem expression_equals_value : 97^3 + 3 * (97^2) + 3 * 97 + 1 = 940792 := 
by
  sorry

end expression_equals_value_l1493_149355


namespace tank_capacity_l1493_149385

theorem tank_capacity (w c : ℕ) (h1 : w = c / 3) (h2 : w + 7 = 2 * c / 5) : c = 105 :=
sorry

end tank_capacity_l1493_149385


namespace rainfall_in_may_l1493_149346

-- Define the rainfalls for the months
def march_rain : ℝ := 3.79
def april_rain : ℝ := 4.5
def june_rain : ℝ := 3.09
def july_rain : ℝ := 4.67

-- Define the average rainfall over five months
def avg_rain : ℝ := 4

-- Define total rainfall calculation
def calc_total_rain (may_rain : ℝ) : ℝ :=
  march_rain + april_rain + may_rain + june_rain + july_rain

-- Problem statement: proving the rainfall in May
theorem rainfall_in_may : ∃ (may_rain : ℝ), calc_total_rain may_rain = avg_rain * 5 ∧ may_rain = 3.95 :=
sorry

end rainfall_in_may_l1493_149346


namespace investor_share_price_l1493_149357

theorem investor_share_price (dividend_rate : ℝ) (face_value : ℝ) (roi : ℝ) (price_per_share : ℝ) : 
  dividend_rate = 0.125 →
  face_value = 40 →
  roi = 0.25 →
  ((dividend_rate * face_value) / price_per_share) = roi →
  price_per_share = 20 :=
by 
  intros h1 h2 h3 h4
  sorry

end investor_share_price_l1493_149357


namespace example_problem_l1493_149340

def operation (a b : ℕ) : ℕ := (a + b) * (a - b)

theorem example_problem : 50 - operation 8 5 = 11 := by
  sorry

end example_problem_l1493_149340


namespace leak_empties_tank_in_4_hours_l1493_149354

theorem leak_empties_tank_in_4_hours
  (A_fills_in : ℝ)
  (A_with_leak_fills_in : ℝ) : 
  (∀ (L : ℝ), A_fills_in = 2 ∧ A_with_leak_fills_in = 4 → L = (1 / 4) → 1 / L = 4) :=
by 
  sorry

end leak_empties_tank_in_4_hours_l1493_149354


namespace unique_fraction_representation_l1493_149321

theorem unique_fraction_representation (p : ℕ) (h_prime : Nat.Prime p) (h_gt_2 : p > 2) :
  ∃! (x y : ℕ), (x ≠ y) ∧ (2 * x * y = p * (x + y)) :=
by
  sorry

end unique_fraction_representation_l1493_149321


namespace sum_of_reciprocals_of_roots_l1493_149356

theorem sum_of_reciprocals_of_roots 
  (r₁ r₂ : ℝ)
  (h_roots : ∀ (x : ℝ), x^2 - 17*x + 8 = 0 → (∃ r, (r = r₁ ∨ r = r₂) ∧ x = r))
  (h_sum : r₁ + r₂ = 17)
  (h_prod : r₁ * r₂ = 8) :
  1/r₁ + 1/r₂ = 17/8 := 
by
  sorry

end sum_of_reciprocals_of_roots_l1493_149356


namespace weavers_in_first_group_l1493_149378

theorem weavers_in_first_group 
  (W : ℕ)
  (H1 : 4 / (W * 4) = 1 / W) 
  (H2 : (9 / 6) / 6 = 0.25) :
  W = 4 :=
sorry

end weavers_in_first_group_l1493_149378


namespace initial_kittens_count_l1493_149307

-- Let's define the initial conditions first.
def kittens_given_away : ℕ := 2
def kittens_remaining : ℕ := 6

-- The main theorem to prove the initial number of kittens.
theorem initial_kittens_count : (kittens_given_away + kittens_remaining) = 8 :=
by sorry

end initial_kittens_count_l1493_149307


namespace most_reasonable_sampling_method_is_stratified_l1493_149377

def population_has_significant_differences 
    (grades : List String)
    (understanding : String → ℕ)
    : Prop := sorry -- This would be defined based on the details of "significant differences"

theorem most_reasonable_sampling_method_is_stratified
    (grades : List String)
    (understanding : String → ℕ)
    (h : population_has_significant_differences grades understanding)
    : (method : String) → (method = "Stratified sampling") :=
sorry

end most_reasonable_sampling_method_is_stratified_l1493_149377


namespace Jessica_cut_roses_l1493_149389

theorem Jessica_cut_roses
  (initial_roses : ℕ) (initial_orchids : ℕ)
  (new_roses : ℕ) (new_orchids : ℕ)
  (cut_roses : ℕ) :
  initial_roses = 15 → initial_orchids = 62 →
  new_roses = 17 → new_orchids = 96 →
  new_roses = initial_roses + cut_roses →
  cut_roses = 2 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h3] at h5
  linarith

end Jessica_cut_roses_l1493_149389


namespace circle_condition_l1493_149337

-- Define the given equation
def equation (m x y : ℝ) : Prop := x^2 + y^2 + 4 * m * x - 2 * y + 5 * m = 0

-- Define the condition for the equation to represent a circle
def represents_circle (m x y : ℝ) : Prop :=
  (x + 2 * m)^2 + (y - 1)^2 = 4 * m^2 - 5 * m + 1 ∧ 4 * m^2 - 5 * m + 1 > 0

-- The main theorem to be proven
theorem circle_condition (m : ℝ) : represents_circle m x y → (m < 1/4 ∨ m > 1) := 
sorry

end circle_condition_l1493_149337


namespace power_function_monotonic_l1493_149368

theorem power_function_monotonic (m : ℝ) :
  2 * m^2 + m > 0 ∧ m > 0 → m = 1 / 2 := 
by
  intro h
  sorry

end power_function_monotonic_l1493_149368


namespace largest_two_digit_number_with_remainder_2_div_13_l1493_149359

theorem largest_two_digit_number_with_remainder_2_div_13 : 
  ∃ (N : ℕ), (10 ≤ N ∧ N ≤ 99) ∧ N % 13 = 2 ∧ ∀ (M : ℕ), (10 ≤ M ∧ M ≤ 99) ∧ M % 13 = 2 → M ≤ N :=
  sorry

end largest_two_digit_number_with_remainder_2_div_13_l1493_149359


namespace total_ttaki_count_l1493_149325

noncomputable def total_ttaki_used (n : ℕ): ℕ := n * n

theorem total_ttaki_count {n : ℕ} (h : 4 * n - 4 = 240) : total_ttaki_used n = 3721 := by
  sorry

end total_ttaki_count_l1493_149325


namespace deductible_amount_l1493_149369

-- This definition represents the conditions of the problem.
def current_annual_deductible_is_increased (D : ℝ) : Prop :=
  (2 / 3) * D = 2000

-- This is the Lean statement, expressing the problem that needs to be proven.
theorem deductible_amount (D : ℝ) (h : current_annual_deductible_is_increased D) : D = 3000 :=
by
  sorry

end deductible_amount_l1493_149369


namespace product_of_five_consecutive_numbers_not_square_l1493_149335

theorem product_of_five_consecutive_numbers_not_square (a b c d e : ℕ)
  (ha : a > 0) (hb : b = a + 1) (hc : c = b + 1) (hd : d = c + 1) (he : e = d + 1) :
  ¬ ∃ k : ℕ, a * b * c * d * e = k^2 := by
  sorry

end product_of_five_consecutive_numbers_not_square_l1493_149335


namespace alicia_tax_cents_per_hour_l1493_149375

-- Define Alicia's hourly wage in dollars.
def alicia_hourly_wage_dollars : ℝ := 25
-- Define the conversion rate from dollars to cents.
def cents_per_dollar : ℝ := 100
-- Define the local tax rate as a percentage.
def tax_rate_percent : ℝ := 2

-- Convert Alicia's hourly wage to cents.
def alicia_hourly_wage_cents : ℝ := alicia_hourly_wage_dollars * cents_per_dollar

-- Define the theorem that needs to be proved.
theorem alicia_tax_cents_per_hour : alicia_hourly_wage_cents * (tax_rate_percent / 100) = 50 := by
  sorry

end alicia_tax_cents_per_hour_l1493_149375


namespace area_of_triangle_abe_l1493_149391

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
10 -- Dummy definition, in actual scenario appropriate area calculation will be required.

def length_AD : ℝ := 2
def length_BD : ℝ := 3

def areas_equal (S_ABE S_DBFE : ℝ) : Prop :=
    S_ABE = S_DBFE

theorem area_of_triangle_abe
  (area_abc : ℝ)
  (length_ad length_bd : ℝ)
  (equal_areas : areas_equal (triangle_area 1 1 1) 1) -- Dummy values, should be substituted with correct arguments
  : triangle_area 1 1 1 = 6 :=
sorry -- proof will be filled later

end area_of_triangle_abe_l1493_149391


namespace apples_in_boxes_l1493_149358

theorem apples_in_boxes (apples_per_box : ℕ) (number_of_boxes : ℕ) (total_apples : ℕ) 
  (h1 : apples_per_box = 12) (h2 : number_of_boxes = 90) : total_apples = 1080 :=
by
  sorry

end apples_in_boxes_l1493_149358


namespace sum_first_10_terms_eq_65_l1493_149341

section ArithmeticSequence

variables (a d : ℕ) (S : ℕ → ℕ) 

-- Define the nth term of the arithmetic sequence
def nth_term (n : ℕ) : ℕ := a + (n - 1) * d

-- Define the sum of the first n terms of the arithmetic sequence
def sum_first_n_terms (n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

-- Condition 1: nth term at n = 3
axiom a3_eq_4 : nth_term 3 = 4

-- Condition 2: difference in sums between n = 9 and n = 6
axiom S9_minus_S6_eq_27 : sum_first_n_terms 9 - sum_first_n_terms 6 = 27

-- To prove: sum of the first 10 terms equals 65
theorem sum_first_10_terms_eq_65 : sum_first_n_terms 10 = 65 :=
sorry

end ArithmeticSequence

end sum_first_10_terms_eq_65_l1493_149341


namespace correct_factorization_l1493_149397

theorem correct_factorization (a b : ℝ) : 
  ((x + 6) * (x - 1) = x^2 + 5 * x - 6) →
  ((x - 2) * (x + 1) = x^2 - x - 2) →
  (a = 1 ∧ b = -6) →
  (x^2 - x - 6 = (x + 2) * (x - 3)) :=
sorry

end correct_factorization_l1493_149397


namespace problem_geometric_description_of_set_T_l1493_149333

open Complex

def set_T (a b : ℝ) : ℂ := a + b * I

theorem problem_geometric_description_of_set_T :
  {w : ℂ | ∃ a b : ℝ, w = set_T a b ∧
    (im ((5 - 3 * I) * w) = 2 * re ((5 - 3 * I) * w))} =
  {w : ℂ | ∃ a : ℝ, w = set_T a (-(13/5) * a)} :=
sorry

end problem_geometric_description_of_set_T_l1493_149333


namespace value_of_inverse_product_l1493_149342

theorem value_of_inverse_product (x y : ℝ) (h1 : x * y > 0) (h2 : 1/x + 1/y = 15) (h3 : (x + y) / 5 = 0.6) :
  1 / (x * y) = 5 :=
by 
  sorry

end value_of_inverse_product_l1493_149342


namespace find_ordered_pair_l1493_149301

noncomputable def discriminant_eq_zero (a c : ℝ) : Prop :=
  a * c = 9

def sum_eq_14 (a c : ℝ) : Prop :=
  a + c = 14

def a_greater_than_c (a c : ℝ) : Prop :=
  a > c

theorem find_ordered_pair : 
  ∃ (a c : ℝ), 
    sum_eq_14 a c ∧ 
    discriminant_eq_zero a c ∧ 
    a_greater_than_c a c ∧ 
    a = 7 + 2 * Real.sqrt 10 ∧ 
    c = 7 - 2 * Real.sqrt 10 :=
by {
  sorry
}

end find_ordered_pair_l1493_149301


namespace distinct_integers_division_l1493_149319

theorem distinct_integers_division (n : ℤ) (h : n > 1) :
  ∃ (a b c : ℤ), a = n^2 + n + 1 ∧ b = n^2 + 2 ∧ c = n^2 + 1 ∧
  n^2 < a ∧ a < (n + 1)^2 ∧ 
  n^2 < b ∧ b < (n + 1)^2 ∧ 
  n^2 < c ∧ c < (n + 1)^2 ∧ 
  a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ c ∣ (a ^ 2 + b ^ 2) := 
by
  sorry

end distinct_integers_division_l1493_149319


namespace roots_quadratic_eq_identity1_roots_quadratic_eq_identity2_l1493_149345

variables {α : Type*} [Field α] (a b c x1 x2 : α)

theorem roots_quadratic_eq_identity1 (h_eq_roots: ∀ x, a * x^2 + b * x + c = 0 → (x = x1 ∨ x = x2)) 
(h_root1: a * x1^2 + b * x1 + c = 0) (h_root2: a * x2^2 + b * x2 + c = 0) :
  x1^2 + x2^2 = (b^2 - 2 * a * c) / a^2 :=
sorry

theorem roots_quadratic_eq_identity2 (h_eq_roots: ∀ x, a * x^2 + b * x + c = 0 → (x = x1 ∨ x = x2)) 
(h_root1: a * x1^2 + b * x1 + c = 0) (h_root2: a * x2^2 + b * x2 + c = 0) :
  x1^3 + x2^3 = (3 * a * b * c - b^3) / a^3 :=
sorry

end roots_quadratic_eq_identity1_roots_quadratic_eq_identity2_l1493_149345


namespace square_of_binomial_l1493_149316

theorem square_of_binomial (a : ℝ) :
  (∃ b : ℝ, (3 * x + b)^2 = 9 * x^2 + 30 * x + a) → a = 25 :=
by
  sorry

end square_of_binomial_l1493_149316


namespace probability_of_drawing_jingyuetan_ticket_l1493_149350

-- Definitions from the problem
def num_jingyuetan_tickets : ℕ := 3
def num_changying_tickets : ℕ := 2
def total_tickets : ℕ := num_jingyuetan_tickets + num_changying_tickets
def num_envelopes : ℕ := total_tickets

-- Probability calculation
def probability_jingyuetan : ℚ := (num_jingyuetan_tickets : ℚ) / (num_envelopes : ℚ)

-- Theorem statement
theorem probability_of_drawing_jingyuetan_ticket : probability_jingyuetan = 3 / 5 :=
by
  sorry

end probability_of_drawing_jingyuetan_ticket_l1493_149350


namespace incorrect_statement_l1493_149374

noncomputable def function_y (x : ℝ) : ℝ := 4 / x

theorem incorrect_statement (x : ℝ) (hx : x ≠ 0) : ¬(∀ x1 x2 : ℝ, (hx1 : x1 ≠ 0) → (hx2 : x2 ≠ 0) → x1 < x2 → function_y x1 > function_y x2) := 
sorry

end incorrect_statement_l1493_149374


namespace quadratic_function_solution_l1493_149331

noncomputable def f (x : ℝ) : ℝ := 1/2 * x^2 + 1/2 * x

theorem quadratic_function_solution (f : ℝ → ℝ)
  (h1 : ∃ a b c : ℝ, (a ≠ 0) ∧ (∀ x, f x = a * x^2 + b * x + c))
  (h2 : f 0 = 0)
  (h3 : ∀ x, f (x+1) = f x + x + 1) :
  ∀ x, f x = 1/2 * x^2 + 1/2 * x :=
by
  sorry

end quadratic_function_solution_l1493_149331


namespace parabola_vertex_l1493_149384

theorem parabola_vertex :
  (∃ h k : ℝ, ∀ x : ℝ, (y : ℝ) = (x - 2)^2 + 5 ∧ h = 2 ∧ k = 5) :=
sorry

end parabola_vertex_l1493_149384


namespace checkerboard_problem_l1493_149323

def checkerboard_rectangles : ℕ := 2025
def checkerboard_squares : ℕ := 285

def relatively_prime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

theorem checkerboard_problem :
  ∃ m n : ℕ, relatively_prime m n ∧ m + n = 154 ∧ (285 : ℚ) / 2025 = m / n :=
by {
  sorry
}

end checkerboard_problem_l1493_149323


namespace natural_number_between_squares_l1493_149381

open Nat

theorem natural_number_between_squares (n m k l : ℕ)
  (h1 : n > m^2)
  (h2 : n < (m+1)^2)
  (h3 : n - k = m^2)
  (h4 : n + l = (m+1)^2) : ∃ x : ℕ, n - k * l = x^2 := by
  sorry

end natural_number_between_squares_l1493_149381


namespace distribute_problems_l1493_149362

theorem distribute_problems :
  (12 ^ 6) = 2985984 := by
  sorry

end distribute_problems_l1493_149362


namespace product_pos_implies_pos_or_neg_pos_pair_implies_product_pos_product_pos_necessary_for_pos_product_pos_not_sufficient_for_pos_l1493_149379

variable {x y : ℝ}

-- The formal statement in Lean
theorem product_pos_implies_pos_or_neg (h : x * y > 0) : (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0) :=
sorry

theorem pos_pair_implies_product_pos (hx : x > 0) (hy : y > 0) : x * y > 0 :=
sorry

theorem product_pos_necessary_for_pos (h : x > 0 ∧ y > 0) : x * y > 0 :=
pos_pair_implies_product_pos h.1 h.2

theorem product_pos_not_sufficient_for_pos (h : x * y > 0) : ¬ (x > 0 ∧ y > 0) :=
sorry

end product_pos_implies_pos_or_neg_pos_pair_implies_product_pos_product_pos_necessary_for_pos_product_pos_not_sufficient_for_pos_l1493_149379


namespace weight_loss_l1493_149312

def initial_weight : ℕ := 69
def current_weight : ℕ := 34

theorem weight_loss :
  initial_weight - current_weight = 35 :=
by
  sorry

end weight_loss_l1493_149312


namespace inequality_solution_l1493_149305

theorem inequality_solution (x : ℝ) (hx : x > 0) : (1 / x > 1) ↔ (0 < x ∧ x < 1) := 
sorry

end inequality_solution_l1493_149305


namespace age_of_25th_student_l1493_149304

theorem age_of_25th_student 
(A : ℤ) (B : ℤ) (C : ℤ) (D : ℤ)
(total_students : ℤ)
(total_age : ℤ)
(age_all_students : ℤ)
(avg_age_all_students : ℤ)
(avg_age_7_students : ℤ)
(avg_age_12_students : ℤ)
(avg_age_5_students : ℤ)
:
total_students = 25 →
avg_age_all_students = 18 →
avg_age_7_students = 20 →
avg_age_12_students = 16 →
avg_age_5_students = 19 →
total_age = total_students * avg_age_all_students →
age_all_students = total_age - (7 * avg_age_7_students + 12 * avg_age_12_students + 5 * avg_age_5_students) →
A = 7 * avg_age_7_students →
B = 12 * avg_age_12_students →
C = 5 * avg_age_5_students →
D = total_age - (A + B + C) →
D = 23 :=
by {
  sorry
}

end age_of_25th_student_l1493_149304


namespace range_of_a_for_monotonic_function_l1493_149329

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - a * x

def is_monotonic_on (f : ℝ → ℝ) (s : Set ℝ) :=
  ∀ ⦃x y : ℝ⦄, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

theorem range_of_a_for_monotonic_function :
  ∀ (a : ℝ), is_monotonic_on (f · a) (Set.Iic (-1)) → a ≤ 3 :=
by
  intros a h
  sorry

end range_of_a_for_monotonic_function_l1493_149329


namespace total_school_population_220_l1493_149314

theorem total_school_population_220 (x B : ℕ) 
  (h1 : 242 = (x * B) / 100) 
  (h2 : B = (50 * x) / 100) : x = 220 := by
  sorry

end total_school_population_220_l1493_149314


namespace saplings_problem_l1493_149326

theorem saplings_problem (x : ℕ) :
  (∃ n : ℕ, 5 * x + 3 = n ∧ 6 * x - 4 = n) ↔ 5 * x + 3 = 6 * x - 4 :=
by
  sorry

end saplings_problem_l1493_149326


namespace total_games_proof_l1493_149364

def num_teams : ℕ := 20
def num_games_per_team_regular_season : ℕ := 38
def total_regular_season_games : ℕ := num_teams * (num_games_per_team_regular_season / 2)
def num_games_per_team_mid_season : ℕ := 3
def total_mid_season_games : ℕ := num_teams * num_games_per_team_mid_season
def quarter_finals_teams : ℕ := 8
def quarter_finals_matchups : ℕ := quarter_finals_teams / 2
def quarter_finals_games : ℕ := quarter_finals_matchups * 2
def semi_finals_teams : ℕ := quarter_finals_matchups
def semi_finals_matchups : ℕ := semi_finals_teams / 2
def semi_finals_games : ℕ := semi_finals_matchups * 2
def final_teams : ℕ := semi_finals_matchups
def final_games : ℕ := final_teams * 2
def total_playoff_games : ℕ := quarter_finals_games + semi_finals_games + final_games

def total_season_games : ℕ := total_regular_season_games + total_mid_season_games + total_playoff_games

theorem total_games_proof : total_season_games = 454 := by
  -- The actual proof will go here
  sorry

end total_games_proof_l1493_149364


namespace perimeter_of_plot_l1493_149302

variable (length breadth : ℝ)
variable (h_ratio : length / breadth = 7 / 5)
variable (h_area : length * breadth = 5040)

theorem perimeter_of_plot (h_ratio : length / breadth = 7 / 5) (h_area : length * breadth = 5040) : 
  (2 * length + 2 * breadth = 288) :=
sorry

end perimeter_of_plot_l1493_149302


namespace positive_real_solutions_unique_l1493_149376

variable (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
variable (x y z : ℝ)

theorem positive_real_solutions_unique :
    x + y + z = a + b + c ∧
    4 * x * y * z - (a^2 * x + b^2 * y + c^2 * z) = abc →
    (x = (b + c) / 2 ∧ y = (c + a) / 2 ∧ z = (a + b) / 2) :=
by
  intros
  sorry

end positive_real_solutions_unique_l1493_149376


namespace a8_div_b8_l1493_149399

variable (a b : ℕ → ℕ)
variable (S T : ℕ → ℕ)

-- Given Conditions
axiom sum_a (n : ℕ) : S n = (n * (a 1 + (n - 1) * a 2)) / 2 -- Sum of first n terms of arithmetic sequence a_n
axiom sum_b (n : ℕ) : T n = (n * (b 1 + (n - 1) * b 2)) / 2 -- Sum of first n terms of arithmetic sequence b_n
axiom ratio (n : ℕ) : S n / T n = (7 * n + 3) / (n + 3)

-- Proof statement
theorem a8_div_b8 : a 8 / b 8 = 6 := by
  sorry

end a8_div_b8_l1493_149399


namespace blake_initial_amount_l1493_149366

theorem blake_initial_amount (X : ℝ) (h1 : X > 0) (h2 : 3 * X / 2 = 30000) : X = 20000 :=
sorry

end blake_initial_amount_l1493_149366


namespace min_vertices_in_hex_grid_l1493_149344

-- Define a hexagonal grid and the condition on the midpoint property.
def hexagonal_grid (p : ℤ × ℤ) : Prop :=
  ∃ m n : ℤ, p = (m, n)

-- Statement: Prove that among any 9 points in a hexagonal grid, there are two points whose midpoint is also a grid point.
theorem min_vertices_in_hex_grid :
  ∀ points : Finset (ℤ × ℤ), points.card = 9 →
  (∃ p1 p2 : (ℤ × ℤ), p1 ∈ points ∧ p2 ∈ points ∧ 
  (∃ midpoint : ℤ × ℤ, hexagonal_grid midpoint ∧ midpoint = ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2))) :=
by
  intros points h_points_card
  sorry

end min_vertices_in_hex_grid_l1493_149344


namespace proportion_correct_l1493_149330

theorem proportion_correct (x y : ℝ) (h : 3 * x = 2 * y) (hy : y ≠ 0) : x / 2 = y / 3 :=
by
  sorry

end proportion_correct_l1493_149330


namespace no_integer_n_squared_plus_one_div_by_seven_l1493_149387

theorem no_integer_n_squared_plus_one_div_by_seven (n : ℤ) : ¬ (n^2 + 1) % 7 = 0 := 
sorry

end no_integer_n_squared_plus_one_div_by_seven_l1493_149387


namespace abs_sum_a_to_7_l1493_149324

-- Sequence definition with domain
def a (n : ℕ) : ℤ := 2 * (n + 1) - 7  -- Lean's ℕ includes 0, so use (n + 1) instead of n here.

-- Prove absolute value sum of first seven terms
theorem abs_sum_a_to_7 : (|a 0| + |a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| = 25) :=
by
  -- Placeholder for actual proof
  sorry

end abs_sum_a_to_7_l1493_149324


namespace integer_solutions_yk_eq_x2_plus_x_l1493_149372

-- Define the problem in Lean
theorem integer_solutions_yk_eq_x2_plus_x (k : ℕ) (hk : k > 1) :
  ∀ (x y : ℤ), y^k = x^2 + x → (x = 0 ∧ y = 0) ∨ (x = -1 ∧ y = 0) :=
by
  sorry

end integer_solutions_yk_eq_x2_plus_x_l1493_149372


namespace classify_tangents_through_point_l1493_149388

-- Definitions for the Lean theorem statement
noncomputable def curve (x : ℝ) : ℝ :=
  x^3 - x

noncomputable def phi (t x₀ y₀ : ℝ) : ℝ :=
  2*t^3 - 3*x₀*t^2 + (x₀ + y₀)

theorem classify_tangents_through_point (x₀ y₀ : ℝ) :
  (if (x₀ + y₀ < 0 ∨ y₀ > x₀^3 - x₀)
   then 1
   else if (x₀ + y₀ > 0 ∧ x₀ + y₀ - x₀^3 < 0)
   then 3
   else if (x₀ + y₀ = 0 ∨ y₀ = x₀^3 - x₀)
   then 2
   else 0) = 
  (if (x₀ + y₀ < 0 ∨ y₀ > x₀^3 - x₀)
   then 1
   else if (x₀ + y₀ > 0 ∧ x₀ + y₀ - x₀^3 < 0)
   then 3
   else if (x₀ + y₀ = 0 ∨ y₀ = x₀^3 - x₀)
   then 2
   else 0) :=
  sorry

end classify_tangents_through_point_l1493_149388


namespace find_ab_l1493_149336

theorem find_ab (a b q r : ℕ) (h : a > 0) (h2 : b > 0) (h3 : (a^2 + b^2) / (a + b) = q) (h4 : (a^2 + b^2) % (a + b) = r) (h5 : q^2 + r = 2010) : a * b = 1643 :=
sorry

end find_ab_l1493_149336


namespace geologists_probability_l1493_149322

theorem geologists_probability :
  let r := 4 -- speed of each geologist in km/h
  let d := 6 -- distance in km
  let sectors := 8 -- number of sectors (roads)
  let total_outcomes := sectors * sectors
  let favorable_outcomes := sectors * 3 -- when distance > 6 km

  -- Calculating probability
  let P := (favorable_outcomes: ℝ) / (total_outcomes: ℝ)

  P = 0.375 :=
by
  sorry

end geologists_probability_l1493_149322


namespace complex_fraction_eval_l1493_149395

theorem complex_fraction_eval (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + a * b + b^2 = 0) :
  (a^15 + b^15) / (a + b)^15 = -2 := by
sorry

end complex_fraction_eval_l1493_149395


namespace find_solutions_l1493_149351

theorem find_solutions (x : ℝ) : (x = -9 ∨ x = -3 ∨ x = 3) →
  (1 / (x^2 + 12 * x - 9) + 1 / (x^2 + 3 * x - 9) + 1 / (x^2 - 14 * x - 9) = 0) :=
by {
  sorry
}

end find_solutions_l1493_149351


namespace print_time_l1493_149367

/-- Define the number of pages per minute printed by the printer -/
def pages_per_minute : ℕ := 25

/-- Define the total number of pages to be printed -/
def total_pages : ℕ := 350

/-- Prove that the time to print 350 pages at a rate of 25 pages per minute is 14 minutes -/
theorem print_time :
  (total_pages / pages_per_minute) = 14 :=
by
  sorry

end print_time_l1493_149367


namespace buffalo_weight_rounding_l1493_149306

theorem buffalo_weight_rounding
  (weight_kg : ℝ) (conversion_factor : ℝ) (expected_weight_lb : ℕ) :
  weight_kg = 850 →
  conversion_factor = 0.454 →
  expected_weight_lb = 1872 →
  Nat.floor (weight_kg / conversion_factor + 0.5) = expected_weight_lb :=
by
  intro h1 h2 h3
  sorry

end buffalo_weight_rounding_l1493_149306


namespace product_of_numbers_l1493_149310

theorem product_of_numbers (x y : ℝ) (h1 : x - y = 1 * k) (h2 : x + y = 8 * k) (h3 : x * y = 30 * k) : 
  x * y = 400 / 7 := 
sorry

end product_of_numbers_l1493_149310


namespace smallest_integer_sum_consecutive_l1493_149313

theorem smallest_integer_sum_consecutive
  (l m n a : ℤ)
  (h1 : a = 9 * l + 36)
  (h2 : a = 10 * m + 45)
  (h3 : a = 11 * n + 55)
  : a = 495 :=
sorry

end smallest_integer_sum_consecutive_l1493_149313


namespace trig_identity_l1493_149320

theorem trig_identity :
  (Real.tan (30 * Real.pi / 180) * Real.cos (60 * Real.pi / 180) + Real.tan (45 * Real.pi / 180) * Real.cos (30 * Real.pi / 180)) = (2 * Real.sqrt 3) / 3 :=
by
  -- Proof is omitted
  sorry

end trig_identity_l1493_149320


namespace correct_statement_l1493_149334

-- Defining the conditions
def freq_eq_prob : Prop :=
  ∀ (f p : ℝ), f = p

def freq_objective : Prop :=
  ∀ (f : ℝ) (n : ℕ), f = f

def freq_stabilizes : Prop :=
  ∀ (p : ℝ), ∃ (f : ℝ) (n : ℕ), f = p

def prob_random : Prop :=
  ∀ (p : ℝ), p = p

-- The statement we need to prove
theorem correct_statement :
  ¬freq_eq_prob ∧ ¬freq_objective ∧ freq_stabilizes ∧ ¬prob_random :=
by
  sorry

end correct_statement_l1493_149334


namespace company_fund_initial_amount_l1493_149347

theorem company_fund_initial_amount (n : ℕ) 
  (h : 45 * n + 95 = 50 * n - 5) : 50 * n - 5 = 995 := by
  sorry

end company_fund_initial_amount_l1493_149347


namespace watch_cost_price_l1493_149318

noncomputable def cost_price : ℝ := 1166.67

theorem watch_cost_price (CP : ℝ) (loss_percent gain_percent : ℝ) (delta : ℝ) 
  (h1 : loss_percent = 0.10) 
  (h2 : gain_percent = 0.02) 
  (h3 : delta = 140) 
  (h4 : (1 - loss_percent) * CP + delta = (1 + gain_percent) * CP) : 
  CP = cost_price := 
by 
  sorry

end watch_cost_price_l1493_149318


namespace acute_triangle_angle_A_range_of_bc_l1493_149327

-- Definitions
variables {A B C : ℝ} {a b c : ℝ}
variable (Δ : ∃ (A B C : ℝ), a = sqrt 2 ∧ ∀ (a b c A B C : ℝ), 
  (a = sqrt 2) ∧ (b = b) ∧ (c = c) ∧ 
  (sin A * cos A / cos (A + C) = a * c / (b^2 - a^2 - c^2)))

-- Problem statement
theorem acute_triangle_angle_A (h : Δ) : A = π / 4 :=
sorry

theorem range_of_bc (h : Δ) : 0 < b * c ∧ b * c ≤ 2 + sqrt 2 :=
sorry

end acute_triangle_angle_A_range_of_bc_l1493_149327


namespace other_asymptote_of_hyperbola_l1493_149315

theorem other_asymptote_of_hyperbola (a b : ℝ) :
  (∀ x : ℝ, a * x + b = 2 * x) →
  (∀ p : ℝ × ℝ, (p.1 = 3)) →
  ∀ (c : ℝ × ℝ), (c.1 = 3 ∧ c.2 = 6) ->
  ∃ (m : ℝ), m = -1/2 ∧ (∀ x, c.2 = -1/2 * x + 15/2) :=
by
  sorry

end other_asymptote_of_hyperbola_l1493_149315


namespace first_place_clay_l1493_149328

def Clay := "Clay"
def Allen := "Allen"
def Bart := "Bart"
def Dick := "Dick"

-- Statements made by the participants
def Allen_statements := ["I finished right before Bart", "I am not the first"]
def Bart_statements := ["I finished right before Clay", "I am not the second"]
def Clay_statements := ["I finished right before Dick", "I am not the third"]
def Dick_statements := ["I finished right before Allen", "I am not the last"]

-- Conditions
def only_two_true_statements : Prop := sorry -- This represents the condition that only two of these statements are true.
def first_place_told_truth : Prop := sorry -- This represents the condition that the person who got first place told at least one truth.

def person_first_place := Clay

theorem first_place_clay : person_first_place = Clay ∧ only_two_true_statements ∧ first_place_told_truth := 
sorry

end first_place_clay_l1493_149328


namespace probability_red_ball_is_correct_l1493_149360

noncomputable def probability_red_ball : ℚ :=
  let prob_A := 1 / 3
  let prob_B := 1 / 3
  let prob_C := 1 / 3
  let prob_red_A := 3 / 10
  let prob_red_B := 7 / 10
  let prob_red_C := 5 / 11
  (prob_A * prob_red_A) + (prob_B * prob_red_B) + (prob_C * prob_red_C)

theorem probability_red_ball_is_correct : probability_red_ball = 16 / 33 := 
by
  sorry

end probability_red_ball_is_correct_l1493_149360


namespace max_min_values_l1493_149392

noncomputable def y (x : ℝ) : ℝ :=
  3 - 4 * Real.sin x - 4 * (Real.cos x)^2

theorem max_min_values :
  (∀ k : ℤ, y (- (Real.pi/2) + 2 * k * Real.pi) = 7) ∧
  (∀ k : ℤ, y (Real.pi/6 + 2 * k * Real.pi) = -2) ∧
  (∀ k : ℤ, y (5 * Real.pi/6 + 2 * k * Real.pi) = -2) := by
  sorry

end max_min_values_l1493_149392


namespace total_watermelons_l1493_149382

theorem total_watermelons 
  (A B C : ℕ) 
  (h1 : A + B = C - 6) 
  (h2 : B + C = A + 16) 
  (h3 : C + A = B + 8) :
  A + B + C = 18 :=
by
  sorry

end total_watermelons_l1493_149382


namespace longest_interval_between_friday_13ths_l1493_149332

theorem longest_interval_between_friday_13ths
  (friday_the_13th : ℕ → ℕ → Prop)
  (at_least_once_per_year : ∀ year, ∃ month, friday_the_13th year month)
  (friday_occurs : ℕ) :
  ∃ (interval : ℕ), interval = 14 :=
by
  sorry

end longest_interval_between_friday_13ths_l1493_149332


namespace inequality_solution_l1493_149300

-- Condition definitions in lean
def numerator (x : ℝ) : ℝ := (x^5 - 13 * x^3 + 36 * x) * (x^4 - 17 * x^2 + 16)
def denominator (y : ℝ) : ℝ := (y^5 - 13 * y^3 + 36 * y) * (y^4 - 17 * y^2 + 16)

-- Given the critical conditions
def is_zero_or_pm1_pm2_pm3_pm4 (y : ℝ) : Prop := 
  y = 0 ∨ y = 1 ∨ y = -1 ∨ y = 2 ∨ y = -2 ∨ y = 3 ∨ y = -3 ∨ y = 4 ∨ y = -4

-- The theorem statement
theorem inequality_solution (x y : ℝ) : 
  (numerator x / denominator y) ≥ 0 ↔ ¬ (is_zero_or_pm1_pm2_pm3_pm4 y) :=
sorry -- proof to be filled in later

end inequality_solution_l1493_149300


namespace vector_dot_product_sum_l1493_149349

noncomputable def points_in_plane (A B C : Type) (dist_AB dist_BC dist_CA : ℝ) : Prop :=
  dist_AB = 3 ∧ dist_BC = 5 ∧ dist_CA = 6

theorem vector_dot_product_sum (A B C : Type) (dist_AB dist_BC dist_CA : ℝ) (HA : points_in_plane A B C dist_AB dist_BC dist_CA) :
    ∃ (AB BC CA : ℝ), AB * BC + BC * CA + CA * AB = -35 :=
by
  sorry

end vector_dot_product_sum_l1493_149349


namespace sum_of_ten_numbers_l1493_149398

theorem sum_of_ten_numbers (average count : ℝ) (h_avg : average = 5.3) (h_count : count = 10) : 
  average * count = 53 :=
by
  sorry

end sum_of_ten_numbers_l1493_149398


namespace problem_part1_problem_part2_l1493_149311

-- Definitions based on conditions
def f (x a : ℝ) : ℝ := x^2 + abs (2*x - a)

-- Proof statements
theorem problem_part1 (a : ℝ) (h_even : ∀ x : ℝ, f x a = f (-x) a) : a = 0 := sorry

theorem problem_part2 (a : ℝ) (h_a_gt_two : a > 2) : 
  ∃ x : ℝ, ∀ y : ℝ, f x a ≤ f y a ∧ f x a = a - 1 := sorry

end problem_part1_problem_part2_l1493_149311


namespace sin2x_value_l1493_149365

theorem sin2x_value (x : ℝ) (h : Real.sin (x + π / 4) = 3 / 5) : 
  Real.sin (2 * x) = 8 * Real.sqrt 2 / 25 := 
by sorry

end sin2x_value_l1493_149365


namespace ratio_of_areas_l1493_149317

theorem ratio_of_areas (s1 s2 : ℝ) (h1 : s1 = 10) (h2 : s2 = 5) :
  let area_equilateral (s : ℝ) := (Real.sqrt 3 / 4) * s^2
  let area_large_triangle := area_equilateral s1
  let area_small_triangle := area_equilateral s2
  let area_trapezoid := area_large_triangle - area_small_triangle
  area_small_triangle / area_trapezoid = 1 / 3 := 
by
  sorry

end ratio_of_areas_l1493_149317


namespace rectangle_perimeter_is_104_l1493_149393

noncomputable def perimeter_of_rectangle (b : ℝ) (h1 : b > 0) (h2 : 3 * b * b = 507) : ℝ :=
  2 * (3 * b) + 2 * b

theorem rectangle_perimeter_is_104 {b : ℝ} (h1 : b > 0) (h2 : 3 * b * b = 507) :
  perimeter_of_rectangle b h1 h2 = 104 :=
by
  sorry

end rectangle_perimeter_is_104_l1493_149393


namespace f_sum_positive_l1493_149308

noncomputable def f (x : ℝ) : ℝ := x + x^3

theorem f_sum_positive (x₁ x₂ x₃ : ℝ) (h₁₂ : x₁ + x₂ > 0) (h₂₃ : x₂ + x₃ > 0) (h₃₁ : x₃ + x₁ > 0) : 
  f x₁ + f x₂ + f x₃ > 0 := 
sorry

end f_sum_positive_l1493_149308
