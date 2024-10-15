import Mathlib

namespace NUMINAMATH_GPT_not_multiple_of_3_l651_65193

noncomputable def exists_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n*(n + 3) = m^2

theorem not_multiple_of_3 
  (n : ℕ) (h1 : 0 < n) (h2 : exists_perfect_square n) : ¬ ∃ k : ℕ, n = 3 * k := 
sorry

end NUMINAMATH_GPT_not_multiple_of_3_l651_65193


namespace NUMINAMATH_GPT_minimum_words_to_learn_for_90_percent_l651_65186

-- Define the conditions
def total_vocabulary_words : ℕ := 800
def minimum_percentage_required : ℚ := 0.90

-- Define the proof goal
theorem minimum_words_to_learn_for_90_percent (x : ℕ) (h1 : (x : ℚ) / total_vocabulary_words ≥ minimum_percentage_required) : x ≥ 720 :=
sorry

end NUMINAMATH_GPT_minimum_words_to_learn_for_90_percent_l651_65186


namespace NUMINAMATH_GPT_simplification_l651_65163

-- Define all relevant powers
def pow2_8 : ℤ := 2^8
def pow4_5 : ℤ := 4^5
def pow2_3 : ℤ := 2^3
def pow_neg2_2 : ℤ := (-2)^2

-- Define the expression inside the parentheses
def inner_expr : ℤ := pow2_3 - pow_neg2_2

-- Define the exponentiation of the inner expression
def inner_expr_pow11 : ℤ := inner_expr^11

-- Define the entire expression
def full_expr : ℤ := (pow2_8 + pow4_5) * inner_expr_pow11

-- State the proof goal
theorem simplification : full_expr = 5368709120 := by
  sorry

end NUMINAMATH_GPT_simplification_l651_65163


namespace NUMINAMATH_GPT_how_many_necklaces_given_away_l651_65125

-- Define the initial conditions
def initial_necklaces := 50
def broken_necklaces := 3
def bought_necklaces := 5
def final_necklaces := 37

-- Define the question proof statement
theorem how_many_necklaces_given_away : 
  (initial_necklaces - broken_necklaces + bought_necklaces - final_necklaces) = 15 :=
by sorry

end NUMINAMATH_GPT_how_many_necklaces_given_away_l651_65125


namespace NUMINAMATH_GPT_area_change_factor_l651_65129

theorem area_change_factor (k b : ℝ) (hk : 0 < k) (hb : 0 < b) :
  let S1 := (b * b) / (2 * k)
  let S2 := (b * b) / (16 * k)
  S1 / S2 = 8 :=
by
  sorry

end NUMINAMATH_GPT_area_change_factor_l651_65129


namespace NUMINAMATH_GPT_g_432_l651_65123

theorem g_432 (g : ℕ → ℤ)
  (h_mul : ∀ x y : ℕ, 0 < x → 0 < y → g (x * y) = g x + g y)
  (h8 : g 8 = 21)
  (h18 : g 18 = 26) :
  g 432 = 47 :=
  sorry

end NUMINAMATH_GPT_g_432_l651_65123


namespace NUMINAMATH_GPT_determine_female_athletes_count_l651_65191

theorem determine_female_athletes_count (m : ℕ) (n : ℕ) (x y : ℕ) (probability : ℚ)
  (h_team : 56 + m = 56 + m) -- redundant, but setting up context
  (h_sample_size : n = 28)
  (h_probability : probability = 1 / 28)
  (h_sample_diff : x - y = 4)
  (h_sample_sum : x + y = n)
  (h_ratio : 56 * y = m * x) : m = 42 :=
by
  sorry

end NUMINAMATH_GPT_determine_female_athletes_count_l651_65191


namespace NUMINAMATH_GPT_red_blue_tile_difference_is_15_l651_65146

def num_blue_tiles : ℕ := 17
def num_red_tiles_initial : ℕ := 8
def additional_red_tiles : ℕ := 24
def num_red_tiles_new : ℕ := num_red_tiles_initial + additional_red_tiles
def tile_difference : ℕ := num_red_tiles_new - num_blue_tiles

theorem red_blue_tile_difference_is_15 : tile_difference = 15 :=
by
  sorry

end NUMINAMATH_GPT_red_blue_tile_difference_is_15_l651_65146


namespace NUMINAMATH_GPT_complex_sum_series_l651_65107

theorem complex_sum_series (ω : ℂ) (h1 : ω ^ 7 = 1) (h2 : ω ≠ 1) :
  ω ^ 16 + ω ^ 18 + ω ^ 20 + ω ^ 22 + ω ^ 24 + ω ^ 26 + ω ^ 28 + ω ^ 30 + 
  ω ^ 32 + ω ^ 34 + ω ^ 36 + ω ^ 38 + ω ^ 40 + ω ^ 42 + ω ^ 44 + ω ^ 46 +
  ω ^ 48 + ω ^ 50 + ω ^ 52 + ω ^ 54 = -1 :=
sorry

end NUMINAMATH_GPT_complex_sum_series_l651_65107


namespace NUMINAMATH_GPT_small_z_value_l651_65168

noncomputable def w (n : ℕ) := n
noncomputable def x (n : ℕ) := n + 1
noncomputable def y (n : ℕ) := n + 2
noncomputable def z (n : ℕ) := n + 4

theorem small_z_value (n : ℕ) 
  (h : w n ^ 3 + x n ^ 3 + y n ^ 3 = z n ^ 3)
  : z n = 9 :=
sorry

end NUMINAMATH_GPT_small_z_value_l651_65168


namespace NUMINAMATH_GPT_find_prime_pairs_l651_65172

theorem find_prime_pairs (p q : ℕ) (p_prime : Nat.Prime p) (q_prime : Nat.Prime q) 
  (h1 : ∃ a : ℤ, a^2 = p - q)
  (h2 : ∃ b : ℤ, b^2 = p * q - q) : 
  (p, q) = (3, 2) :=
by {
    sorry
}

end NUMINAMATH_GPT_find_prime_pairs_l651_65172


namespace NUMINAMATH_GPT_carol_total_peanuts_l651_65184

-- Conditions as definitions
def carol_initial_peanuts : Nat := 2
def carol_father_peanuts : Nat := 5

-- Theorem stating that the total number of peanuts Carol has is 7
theorem carol_total_peanuts : carol_initial_peanuts + carol_father_peanuts = 7 := by
  -- Proof would go here, but we use sorry to skip
  sorry

end NUMINAMATH_GPT_carol_total_peanuts_l651_65184


namespace NUMINAMATH_GPT_saddle_value_l651_65189

theorem saddle_value (S : ℝ) (H : ℝ) (h1 : S + H = 100) (h2 : H = 7 * S) : S = 12.50 :=
by
  sorry

end NUMINAMATH_GPT_saddle_value_l651_65189


namespace NUMINAMATH_GPT_product_of_three_equal_numbers_l651_65136

theorem product_of_three_equal_numbers
    (a b : ℕ) (x : ℕ)
    (h1 : a = 12)
    (h2 : b = 22)
    (h_mean : (a + b + 3 * x) / 5 = 20) :
    x * x * x = 10648 := by
  sorry

end NUMINAMATH_GPT_product_of_three_equal_numbers_l651_65136


namespace NUMINAMATH_GPT_minimum_knights_in_tournament_l651_65185

def knights_tournament : Prop :=
  ∃ (N : ℕ), (∀ (x : ℕ), x = N / 4 →
    ∃ (k : ℕ), k = (3 * x - 1) / 7 → N = 20)

theorem minimum_knights_in_tournament : knights_tournament :=
  sorry

end NUMINAMATH_GPT_minimum_knights_in_tournament_l651_65185


namespace NUMINAMATH_GPT_find_n_correct_l651_65117

noncomputable def find_n : Prop :=
  ∃ n : ℕ, 0 ≤ n ∧ n ≤ 180 ∧ Real.cos (n * (Real.pi / 180)) = Real.cos (317 * (Real.pi / 180)) → n = 43

theorem find_n_correct : find_n :=
  sorry

end NUMINAMATH_GPT_find_n_correct_l651_65117


namespace NUMINAMATH_GPT_x_ge_3_is_necessary_but_not_sufficient_for_x_gt_3_l651_65127

theorem x_ge_3_is_necessary_but_not_sufficient_for_x_gt_3 :
  (∀ x : ℝ, x > 3 → x ≥ 3) ∧ (∃ x : ℝ, x ≥ 3 ∧ ¬ (x > 3)) :=
by
  sorry

end NUMINAMATH_GPT_x_ge_3_is_necessary_but_not_sufficient_for_x_gt_3_l651_65127


namespace NUMINAMATH_GPT_range_of_a_minus_b_l651_65149

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^2 + a * x + b

theorem range_of_a_minus_b (a b : ℝ) (h1 : ∃ α β : ℝ, α ≠ β ∧ f α a b = 0 ∧ f β a b = 0)
  (h2 : ∃ x1 x2 x3 x4 : ℝ, x1 < x2 ∧ x2 < x3 ∧ x3 < x4 ∧
                         (x2 - x1 = x3 - x2) ∧ (x3 - x2 = x4 - x3) ∧
                         f (x1^2 + 2 * x1 - 1) a b = 0 ∧
                         f (x2^2 + 2 * x2 - 1) a b = 0 ∧
                         f (x3^2 + 2 * x3 - 1) a b = 0 ∧
                         f (x4^2 + 2 * x4 - 1) a b = 0) :
  a - b ≤ 25 / 9 :=
sorry

end NUMINAMATH_GPT_range_of_a_minus_b_l651_65149


namespace NUMINAMATH_GPT_distinct_special_sums_l651_65175

def is_special_fraction (a b : ℕ) : Prop := a + b = 18

def is_special_sum (n : ℤ) : Prop :=
  ∃ (a1 b1 a2 b2 : ℕ), is_special_fraction a1 b1 ∧ is_special_fraction a2 b2 ∧ 
  n = (a1 : ℤ) * (b2 : ℤ) * b1 + (a2 : ℤ) * (b1 : ℤ) / a1

theorem distinct_special_sums : 
  (∃ (sums : Finset ℤ), 
    (∀ n, n ∈ sums ↔ is_special_sum n) ∧ 
    sums.card = 7) :=
sorry

end NUMINAMATH_GPT_distinct_special_sums_l651_65175


namespace NUMINAMATH_GPT_sally_weekend_reading_l651_65192

theorem sally_weekend_reading (pages_on_weekdays : ℕ) (total_pages : ℕ) (weeks : ℕ) (weekdays_per_week : ℕ) (total_days : ℕ) 
  (finishing_time : ℕ) (weekend_days : ℕ) (pages_weekdays_total : ℕ) :
  pages_on_weekdays = 10 →
  total_pages = 180 →
  weeks = 2 →
  weekdays_per_week = 5 →
  weekend_days = (total_days - weekdays_per_week * weeks) →
  total_days = 7 * weeks →
  finishing_time = weeks →
  pages_weekdays_total = pages_on_weekdays * weekdays_per_week * weeks →
  (total_pages - pages_weekdays_total) / weekend_days = 20 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end NUMINAMATH_GPT_sally_weekend_reading_l651_65192


namespace NUMINAMATH_GPT_three_digit_multiples_of_24_l651_65155

theorem three_digit_multiples_of_24 : 
  let lower_bound := 100
  let upper_bound := 999
  let div_by := 24
  let first := lower_bound + (div_by - lower_bound % div_by) % div_by
  let last := upper_bound - (upper_bound % div_by)
  ∃ n : ℕ, (n + 1) = (last - first) / div_by + 1 := 
sorry

end NUMINAMATH_GPT_three_digit_multiples_of_24_l651_65155


namespace NUMINAMATH_GPT_tea_drinking_proof_l651_65139

theorem tea_drinking_proof :
  ∃ (k : ℝ), 
    (∃ (c_sunday t_sunday c_wednesday t_wednesday : ℝ),
      c_sunday = 8.5 ∧ 
      t_sunday = 4 ∧ 
      c_wednesday = 5 ∧ 
      t_sunday * c_sunday = k ∧ 
      t_wednesday * c_wednesday = k ∧ 
      t_wednesday = 6.8) :=
sorry

end NUMINAMATH_GPT_tea_drinking_proof_l651_65139


namespace NUMINAMATH_GPT_elvis_ralph_matchsticks_l651_65124

/-- 
   Elvis and Ralph are making square shapes with matchsticks from a box containing 
   50 matchsticks. Elvis makes 4-matchstick squares and Ralph makes 8-matchstick 
   squares. If Elvis makes 5 squares and Ralph makes 3, prove the number of matchsticks 
   left in the box is 6. 
-/
def matchsticks_left_in_box
  (initial_matchsticks : ℕ)
  (elvis_squares : ℕ)
  (elvis_matchsticks : ℕ)
  (ralph_squares : ℕ)
  (ralph_matchsticks : ℕ)
  (elvis_squares_count : ℕ)
  (ralph_squares_count : ℕ) : ℕ :=
  initial_matchsticks - (elvis_squares_count * elvis_matchsticks + ralph_squares_count * ralph_matchsticks)

theorem elvis_ralph_matchsticks : matchsticks_left_in_box 50 4 5 8 3 = 6 := 
  sorry

end NUMINAMATH_GPT_elvis_ralph_matchsticks_l651_65124


namespace NUMINAMATH_GPT_pen_price_first_day_l651_65102

theorem pen_price_first_day (x y : ℕ) 
  (h1 : x * y = (x - 1) * (y + 100)) 
  (h2 : x * y = (x + 2) * (y - 100)) : x = 4 :=
by
  sorry

end NUMINAMATH_GPT_pen_price_first_day_l651_65102


namespace NUMINAMATH_GPT_john_age_proof_l651_65103

theorem john_age_proof (d j : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 :=
by
  sorry

end NUMINAMATH_GPT_john_age_proof_l651_65103


namespace NUMINAMATH_GPT_ellipse_y_axis_intersection_l651_65170

open Real

/-- Defines an ellipse with given foci and a point on the ellipse,
    and establishes the coordinate of the other y-axis intersection. -/
theorem ellipse_y_axis_intersection :
  ∃ y : ℝ, (dist (0, y) (1, -1) + dist (0, y) (-2, 2) = 3 * sqrt 2) ∧ y = sqrt ((9 * sqrt 2 - 4) / 2) :=
sorry

end NUMINAMATH_GPT_ellipse_y_axis_intersection_l651_65170


namespace NUMINAMATH_GPT_percentage_republicans_vote_X_l651_65109

theorem percentage_republicans_vote_X (R : ℝ) (P_R : ℝ) :
  (3 * R * P_R + 2 * R * 0.15) - (3 * R * (1 - P_R) + 2 * R * 0.85) = 0.019999999999999927 * (3 * R + 2 * R) →
  P_R = 4.1 / 6 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_percentage_republicans_vote_X_l651_65109


namespace NUMINAMATH_GPT_part1_part2_l651_65144

def op (a b : ℤ) := 2 * a - 3 * b

theorem part1 : op (-2) 3 = -13 := 
by
  -- Proof omitted
  sorry

theorem part2 (x : ℤ) : 
  let A := op (3 * x - 2) (x + 1)
  let B := op (-3 / 2 * x + 1) (-1 - 2 * x)
  B > A :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_part1_part2_l651_65144


namespace NUMINAMATH_GPT_min_sum_fraction_sqrt_l651_65199

open Real

theorem min_sum_fraction_sqrt (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) :
  ∃ min, min = sqrt 2 ∧ ∀ z, (z = (x / sqrt (1 - x) + y / sqrt (1 - y))) → z ≥ sqrt 2 :=
sorry

end NUMINAMATH_GPT_min_sum_fraction_sqrt_l651_65199


namespace NUMINAMATH_GPT_value_of_m_l651_65142

theorem value_of_m (m : ℕ) (h : 3 * 6^4 + m * 6^3 + 5 * 6^2 + 0 * 6^1 + 2 * 6^0 = 4934) : m = 4 :=
by
  sorry

end NUMINAMATH_GPT_value_of_m_l651_65142


namespace NUMINAMATH_GPT_only_statement_4_is_correct_l651_65130

-- Defining conditions for input/output statement correctness
def INPUT_statement_is_correct (s : String) : Prop :=
  s = "INPUT x=, 2"

def PRINT_statement_is_correct (s : String) : Prop :=
  s = "PRINT 20, 4"

-- List of statements
def statement_1 := "INPUT a; b; c"
def statement_2 := "PRINT a=1"
def statement_3 := "INPUT x=2"
def statement_4 := "PRINT 20, 4"

-- Predicate for correctness of statements
def statement_is_correct (s : String) : Prop :=
  (s = statement_4) ∧
  ¬(s = statement_1 ∨ s = statement_2 ∨ s = statement_3)

-- Theorem to prove that only statement 4 is correct
theorem only_statement_4_is_correct :
  ∀ s : String, (statement_is_correct s) ↔ (s = statement_4) :=
by
  intros s
  sorry

end NUMINAMATH_GPT_only_statement_4_is_correct_l651_65130


namespace NUMINAMATH_GPT_remainder_when_divided_by_44_l651_65195

theorem remainder_when_divided_by_44 (N : ℕ) (Q : ℕ) (R : ℕ)
  (h1 : N = 44 * 432 + R)
  (h2 : N = 31 * Q + 5) :
  R = 2 :=
sorry

end NUMINAMATH_GPT_remainder_when_divided_by_44_l651_65195


namespace NUMINAMATH_GPT_initial_ratio_l651_65156

-- Definitions of the initial state and conditions
variables (M W : ℕ)
def initial_men : ℕ := M
def initial_women : ℕ := W
def men_after_entry : ℕ := M + 2
def women_after_exit_and_doubling : ℕ := (W - 3) * 2
def current_men : ℕ := 14
def current_women : ℕ := 24

-- Theorem to prove the initial ratio
theorem initial_ratio (M W : ℕ) 
    (hm : men_after_entry M = current_men)
    (hw : women_after_exit_and_doubling W = current_women) :
  M / Nat.gcd M W = 4 ∧ W / Nat.gcd M W = 5 :=
by
  sorry

end NUMINAMATH_GPT_initial_ratio_l651_65156


namespace NUMINAMATH_GPT_virginia_taught_fewer_years_l651_65143

-- Definitions based on conditions
variable (V A D : ℕ)

-- Dennis has taught for 34 years
axiom h1 : D = 34

-- Virginia has taught for 9 more years than Adrienne
axiom h2 : V = A + 9

-- Combined total of years taught is 75
axiom h3 : V + A + D = 75

-- Proof statement: Virginia has taught for 9 fewer years than Dennis
theorem virginia_taught_fewer_years : D - V = 9 :=
  sorry

end NUMINAMATH_GPT_virginia_taught_fewer_years_l651_65143


namespace NUMINAMATH_GPT_cost_of_rice_l651_65111

-- Define the cost variables
variables (E R K : ℝ)

-- State the conditions as assumptions
def conditions (E R K : ℝ) : Prop :=
  (E = R) ∧
  (K = (2 / 3) * E) ∧
  (2 * K = 48)

-- State the theorem to be proven
theorem cost_of_rice (E R K : ℝ) (h : conditions E R K) : R = 36 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_rice_l651_65111


namespace NUMINAMATH_GPT_calculate_area_of_region_l651_65188

theorem calculate_area_of_region :
  let region := {p : ℝ × ℝ | p.1^2 + p.2^2 + 2 * p.1 - 4 * p.2 = 12}
  ∃ area, area = 17 * Real.pi
:= by
  sorry

end NUMINAMATH_GPT_calculate_area_of_region_l651_65188


namespace NUMINAMATH_GPT_toothpicks_needed_for_cube_grid_l651_65141

-- Defining the conditions: a cube-shaped grid with dimensions 5x5x5.
def grid_length : ℕ := 5
def grid_width : ℕ := 5
def grid_height : ℕ := 5

-- The theorem to prove the number of toothpicks needed is 2340.
theorem toothpicks_needed_for_cube_grid (L W H : ℕ) (h1 : L = grid_length) (h2 : W = grid_width) (h3 : H = grid_height) :
  (L + 1) * (W + 1) * H + 2 * (L + 1) * W * (H + 1) = 2340 :=
  by
    -- Proof goes here
    sorry

end NUMINAMATH_GPT_toothpicks_needed_for_cube_grid_l651_65141


namespace NUMINAMATH_GPT_length_of_other_train_l651_65115

def speed1 := 90 -- speed in km/hr
def speed2 := 90 -- speed in km/hr
def length_train1 := 1.10 -- length in km
def crossing_time := 40 -- time in seconds

theorem length_of_other_train : 
  ∀ s1 s2 l1 t l2 : ℝ,
  s1 = 90 → s2 = 90 → l1 = 1.10 → t = 40 → 
  ((s1 + s2) / 3600 * t - l1 = l2) → 
  l2 = 0.90 :=
by
  intros s1 s2 l1 t l2 hs1 hs2 hl1 ht hdist
  sorry

end NUMINAMATH_GPT_length_of_other_train_l651_65115


namespace NUMINAMATH_GPT_percent_counties_l651_65161

def p1 : ℕ := 21
def p2 : ℕ := 44
def p3 : ℕ := 18

theorem percent_counties (h1 : p1 = 21) (h2 : p2 = 44) (h3 : p3 = 18) : p1 + p2 + p3 = 83 :=
by sorry

end NUMINAMATH_GPT_percent_counties_l651_65161


namespace NUMINAMATH_GPT_find_a2_plus_b2_l651_65183

theorem find_a2_plus_b2 (a b : ℝ) (h1 : a * b = 16) (h2 : a + b = 10) : a^2 + b^2 = 68 :=
sorry

end NUMINAMATH_GPT_find_a2_plus_b2_l651_65183


namespace NUMINAMATH_GPT_binom_7_4_plus_5_l651_65162

theorem binom_7_4_plus_5 : ((Nat.choose 7 4) + 5) = 40 := by
  sorry

end NUMINAMATH_GPT_binom_7_4_plus_5_l651_65162


namespace NUMINAMATH_GPT_intersection_is_correct_l651_65100

def A : Set ℝ := { x | x * (x - 2) < 0 }
def B : Set ℝ := { x | Real.log x > 0 }

theorem intersection_is_correct : A ∩ B = { x | 1 < x ∧ x < 2 } := by
  sorry

end NUMINAMATH_GPT_intersection_is_correct_l651_65100


namespace NUMINAMATH_GPT_cory_fruit_eating_orders_l651_65177

open Nat

theorem cory_fruit_eating_orders : 
    let apples := 4
    let oranges := 3
    let bananas := 2
    let grape := 1
    let total_fruits := apples + oranges + bananas + grape
    apples + oranges + bananas + grape = 10 →
    total_fruits = 10 →
    apples ≥ 1 →
    factorial 9 / (factorial 3 * factorial 3 * factorial 2 * factorial 1) = 5040 :=
by
  intros apples oranges bananas grape total_fruits h_total h_sum h_apples
  sorry

end NUMINAMATH_GPT_cory_fruit_eating_orders_l651_65177


namespace NUMINAMATH_GPT_inequality_proof_l651_65118

theorem inequality_proof
  (a b c d : ℝ) 
  (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d)
  (h_cond : a * b + b * c + c * d + d * a = 1) :
  (a^3 / (b + c + d) + b^3 / (c + d + a) + c^3 / (a + b + d) + d^3 / (a + b + c) ≥ 1 / 3) :=
by {
  sorry
}

end NUMINAMATH_GPT_inequality_proof_l651_65118


namespace NUMINAMATH_GPT_lcm_first_ten_l651_65197

-- Define the set of first ten positive integers
def first_ten_integers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the LCM of a list of integers
noncomputable def lcm_list (l : List ℕ) : ℕ :=
List.foldr Nat.lcm 1 l

-- The theorem stating that the LCM of the first ten integers is 2520
theorem lcm_first_ten : lcm_list first_ten_integers = 2520 := by
  sorry

end NUMINAMATH_GPT_lcm_first_ten_l651_65197


namespace NUMINAMATH_GPT_find_e_l651_65152

theorem find_e (a b c d e : ℝ) (h1 : a < b) (h2 : b < c) (h3 : c < d) (h4 : d < e)
    (h_lb1 : a + b = 32) (h_lb2 : a + c = 36) (h_lb3 : b + c = 37)
    (h_ub1 : c + e = 48) (h_ub2 : d + e = 51) : e = 27.5 :=
sorry

end NUMINAMATH_GPT_find_e_l651_65152


namespace NUMINAMATH_GPT_solve_for_t_l651_65101

theorem solve_for_t (t : ℝ) (h1 : 60 * t + 80 * ((10 : ℝ)/3 - t) = 220) 
  (h2 : 0 ≤ t) : 60 * t + 80 * ((10 : ℝ)/3 - t) = 220 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_t_l651_65101


namespace NUMINAMATH_GPT_probability_product_divisible_by_four_l651_65116

open Finset

theorem probability_product_divisible_by_four :
  (∃ (favorable_pairs total_pairs : ℕ), favorable_pairs = 70 ∧ total_pairs = 190 ∧ favorable_pairs / total_pairs = 7 / 19) := 
sorry

end NUMINAMATH_GPT_probability_product_divisible_by_four_l651_65116


namespace NUMINAMATH_GPT_obtuse_triangle_existence_l651_65133

theorem obtuse_triangle_existence :
  ∃ (a b c : ℝ), (a = 2 ∧ b = 6 ∧ c = 7 ∧ 
  (a^2 + b^2 < c^2 ∨ b^2 + c^2 < a^2 ∨ c^2 + a^2 < b^2)) ∧
  ¬(6^2 + 7^2 < 8^2 ∨ 7^2 + 8^2 < 6^2 ∨ 8^2 + 6^2 < 7^2) ∧
  ¬(7^2 + 8^2 < 10^2 ∨ 8^2 + 10^2 < 7^2 ∨ 10^2 + 7^2 < 8^2) ∧
  ¬(5^2 + 12^2 < 13^2 ∨ 12^2 + 13^2 < 5^2 ∨ 13^2 + 5^2 < 12^2) :=
sorry

end NUMINAMATH_GPT_obtuse_triangle_existence_l651_65133


namespace NUMINAMATH_GPT_find_a_l651_65108

theorem find_a (f : ℝ → ℝ)
  (h : ∀ x : ℝ, x < 2 → a - 3 * x > 0) :
  a = 6 :=
by sorry

end NUMINAMATH_GPT_find_a_l651_65108


namespace NUMINAMATH_GPT_find_integers_l651_65145

def isPerfectSquare (n : ℤ) : Prop := ∃ m : ℤ, m * m = n

theorem find_integers (x : ℤ) (h : isPerfectSquare (x^2 + 19 * x + 95)) : x = -14 ∨ x = -5 := by
  sorry

end NUMINAMATH_GPT_find_integers_l651_65145


namespace NUMINAMATH_GPT_calculate_expression_l651_65176

theorem calculate_expression 
  (a1 : 84 + 4 / 19 = 1600 / 19) 
  (a2 : 105 + 5 / 19 = 2000 / 19) 
  (a3 : 1.375 = 11 / 8) 
  (a4 : 0.8 = 4 / 5) :
  84 * (4 / 19) * (11 / 8) + 105 * (5 / 19) * (4 / 5) = 200 := 
sorry

end NUMINAMATH_GPT_calculate_expression_l651_65176


namespace NUMINAMATH_GPT_sue_travel_time_correct_l651_65157

-- Define the flight and layover times as constants
def NO_to_ATL_flight_hours : ℕ := 2
def ATL_layover_hours : ℕ := 4
def ATL_to_CHI_flight_hours : ℕ := 5
def CHI_time_diff_hours : ℤ := -1
def CHI_layover_hours : ℕ := 3
def CHI_to_NY_flight_hours : ℕ := 3
def NY_time_diff_hours : ℤ := 1
def NY_layover_hours : ℕ := 16
def NY_to_DEN_flight_hours : ℕ := 6
def DEN_time_diff_hours : ℤ := -2
def DEN_layover_hours : ℕ := 5
def DEN_to_SF_flight_hours : ℕ := 4
def SF_time_diff_hours : ℤ := -1

-- Total time calculation including flights, layovers, and time zone changes
def total_travel_time_hours : ℕ :=
  NO_to_ATL_flight_hours +
  ATL_layover_hours +
  (ATL_to_CHI_flight_hours + CHI_time_diff_hours).toNat +  -- Handle time difference (ensure non-negative)
  CHI_layover_hours +
  (CHI_to_NY_flight_hours + NY_time_diff_hours).toNat +
  NY_layover_hours +
  (NY_to_DEN_flight_hours + DEN_time_diff_hours).toNat +
  DEN_layover_hours +
  (DEN_to_SF_flight_hours + SF_time_diff_hours).toNat

-- Statement to prove in Lean:
theorem sue_travel_time_correct : total_travel_time_hours = 45 :=
by {
  -- Skipping proof details since only the statement is required
  sorry
}

end NUMINAMATH_GPT_sue_travel_time_correct_l651_65157


namespace NUMINAMATH_GPT_correct_ignition_time_l651_65182

noncomputable def ignition_time_satisfying_condition (initial_length : ℝ) (l : ℝ) : ℕ :=
  let burn_rate1 := l / 240
  let burn_rate2 := l / 360
  let stub1 t := l - burn_rate1 * t
  let stub2 t := l - burn_rate2 * t
  let stub_length_condition t := stub2 t = 3 * stub1 t
  let time_difference_at_6AM := 360 -- 6 AM is 360 minutes after midnight
  360 - 180 -- time to ignite the candles

theorem correct_ignition_time : ignition_time_satisfying_condition l 6 = 180 := 
by sorry

end NUMINAMATH_GPT_correct_ignition_time_l651_65182


namespace NUMINAMATH_GPT_Andrey_knows_the_secret_l651_65196

/-- Question: Does Andrey know the secret?
    Conditions:
    - Andrey says: "I know the secret!"
    - Boris says to Andrey: "No, you don't!"
    - Victor says to Boris: "Boris, you are wrong!"
    - Gosha says to Victor: "No, you are wrong!"
    - Dima says to Gosha: "Gosha, you are lying!"
    - More than half of the kids told the truth (i.e., at least 3 out of 5). --/
theorem Andrey_knows_the_secret (Andrey Boris Victor Gosha Dima : Prop) (truth_count : ℕ)
    (h1 : Andrey)   -- Andrey says he knows the secret
    (h2 : ¬Andrey → Boris)   -- Boris says Andrey does not know the secret
    (h3 : ¬Boris → Victor)   -- Victor says Boris is wrong
    (h4 : ¬Victor → Gosha)   -- Gosha says Victor is wrong
    (h5 : ¬Gosha → Dima)   -- Dima says Gosha is lying
    (h6 : truth_count > 2)   -- More than half of the friends tell the truth (at least 3 out of 5)
    : Andrey := 
sorry

end NUMINAMATH_GPT_Andrey_knows_the_secret_l651_65196


namespace NUMINAMATH_GPT_brick_height_l651_65106

def surface_area (l w h : ℝ) : ℝ := 2 * l * w + 2 * l * h + 2 * w * h

theorem brick_height 
  (l : ℝ) (w : ℝ) (SA : ℝ) (h : ℝ) 
  (surface_area_eq : surface_area l w h = SA)
  (length_eq : l = 10)
  (width_eq : w = 4)
  (surface_area_given : SA = 164) :
  h = 3 :=
by
  sorry

end NUMINAMATH_GPT_brick_height_l651_65106


namespace NUMINAMATH_GPT_speed_of_stream_l651_65114

theorem speed_of_stream (v : ℝ) (canoe_speed : ℝ) 
  (upstream_speed_condition : canoe_speed - v = 3) 
  (downstream_speed_condition : canoe_speed + v = 12) :
  v = 4.5 := 
by 
  sorry

end NUMINAMATH_GPT_speed_of_stream_l651_65114


namespace NUMINAMATH_GPT_total_seeds_correct_l651_65120

def seeds_per_bed : ℕ := 6
def flower_beds : ℕ := 9
def total_seeds : ℕ := seeds_per_bed * flower_beds

theorem total_seeds_correct : total_seeds = 54 := by
  sorry

end NUMINAMATH_GPT_total_seeds_correct_l651_65120


namespace NUMINAMATH_GPT_water_evaporation_correct_l651_65121

noncomputable def water_evaporation_each_day (initial_water: ℝ) (percentage_evaporated: ℝ) (days: ℕ) : ℝ :=
  let total_evaporated := (percentage_evaporated / 100) * initial_water
  total_evaporated / days

theorem water_evaporation_correct :
  water_evaporation_each_day 10 6 30 = 0.02 := by
  sorry

end NUMINAMATH_GPT_water_evaporation_correct_l651_65121


namespace NUMINAMATH_GPT_find_m_l651_65167

theorem find_m 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ)
  (h_arith_seq : ∀ n, a (n - 1) + a (n + 1) = 2 * a n)
  (h_cond1 : a (m - 1) + a (m + 1) - a m ^ 2 = 0)
  (h_cond2 : S (2 * m - 1) = 38) 
  : m = 10 :=
sorry

end NUMINAMATH_GPT_find_m_l651_65167


namespace NUMINAMATH_GPT_log_equivalence_l651_65165

theorem log_equivalence :
  (Real.log 9 / Real.log 2) * (Real.log 4 / Real.log 3) = 4 :=
by
  sorry

end NUMINAMATH_GPT_log_equivalence_l651_65165


namespace NUMINAMATH_GPT_solve_x_for_equation_l651_65151

theorem solve_x_for_equation (x : ℝ) (h : 2 / (x + 3) + 3 * x / (x + 3) - 4 / (x + 3) = 4) : x = -14 :=
by 
  sorry

end NUMINAMATH_GPT_solve_x_for_equation_l651_65151


namespace NUMINAMATH_GPT_find_x_when_y_is_72_l651_65166

theorem find_x_when_y_is_72 
  (x y : ℝ) (k : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) 
  (h_const : ∀ x y, 0 < x → 0 < y → x^2 * y = k)
  (h_initial : 9 * 8 = k)
  (h_y_72 : y = 72)
  (h_x2_factor : x^2 = 4 * 9) :
  x = 1 :=
sorry

end NUMINAMATH_GPT_find_x_when_y_is_72_l651_65166


namespace NUMINAMATH_GPT_triangle_inequality_l651_65159

variable (a b c R : ℝ)

-- Assuming a, b, c as the sides of a triangle
-- and R as the circumradius.

theorem triangle_inequality:
  (1 / (a * b)) + (1 / (b * c)) + (1 / (c * a)) ≥ (1 / (R * R)) :=
by
  sorry

end NUMINAMATH_GPT_triangle_inequality_l651_65159


namespace NUMINAMATH_GPT_stickers_after_exchange_l651_65169

-- Given conditions
def Ryan_stickers : ℕ := 30
def Steven_stickers : ℕ := 3 * Ryan_stickers
def Terry_stickers : ℕ := Steven_stickers + 20
def Emily_stickers : ℕ := Steven_stickers / 2
def Jasmine_stickers : ℕ := Terry_stickers + Terry_stickers / 10

def total_stickers_before : ℕ := 
  Ryan_stickers + Steven_stickers + Terry_stickers + Emily_stickers + Jasmine_stickers

noncomputable def total_stickers_after : ℕ := 
  total_stickers_before - 2 * 5

-- The goal is to prove that the total stickers after the exchange event is 386
theorem stickers_after_exchange : total_stickers_after = 386 := 
  by sorry

end NUMINAMATH_GPT_stickers_after_exchange_l651_65169


namespace NUMINAMATH_GPT_scientific_notation_150_billion_l651_65179

theorem scientific_notation_150_billion : 150000000000 = 1.5 * 10^11 :=
sorry

end NUMINAMATH_GPT_scientific_notation_150_billion_l651_65179


namespace NUMINAMATH_GPT_travel_time_l651_65190

noncomputable def convert_kmh_to_mps (speed_kmh : ℝ) : ℝ :=
  speed_kmh * 1000 / 3600

theorem travel_time
  (speed_kmh : ℝ)
  (distance_m : ℝ) :
  speed_kmh = 63 →
  distance_m = 437.535 →
  (distance_m / convert_kmh_to_mps speed_kmh) = 25 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_travel_time_l651_65190


namespace NUMINAMATH_GPT_latest_start_time_for_liz_l651_65164

def latest_start_time (weight : ℕ) (roast_time_per_pound : ℕ) (num_turkeys : ℕ) (dinner_time : ℕ) : ℕ :=
  dinner_time - (num_turkeys * weight * roast_time_per_pound) / 60

theorem latest_start_time_for_liz : 
  latest_start_time 16 15 2 18 = 10 := by
  sorry

end NUMINAMATH_GPT_latest_start_time_for_liz_l651_65164


namespace NUMINAMATH_GPT_mod_remainder_of_sum_of_primes_l651_65153

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def sum_of_odd_primes : ℕ := List.sum odd_primes_less_than_32

theorem mod_remainder_of_sum_of_primes : sum_of_odd_primes % 32 = 30 := by
  sorry

end NUMINAMATH_GPT_mod_remainder_of_sum_of_primes_l651_65153


namespace NUMINAMATH_GPT_isabel_games_problem_l651_65128

noncomputable def prime_sum : ℕ := 83 + 89 + 97

theorem isabel_games_problem (initial_games : ℕ) (X : ℕ) (H1 : initial_games = 90) (H2 : X = prime_sum) : X > initial_games :=
by 
  sorry

end NUMINAMATH_GPT_isabel_games_problem_l651_65128


namespace NUMINAMATH_GPT_find_f_lg_lg2_l651_65154

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + x ^ 2) - x) + 4

theorem find_f_lg_lg2 :
  f (Real.logb 10 (2)) = 3 :=
sorry

end NUMINAMATH_GPT_find_f_lg_lg2_l651_65154


namespace NUMINAMATH_GPT_target_water_percentage_is_two_percent_l651_65148

variable (initial_milk_volume pure_milk_volume : ℕ)
variable (initial_water_percentage target_water_percentage : ℚ)

-- Conditions: Initial milk contains 5% water and we add 15 liters of pure milk
axiom initial_milk_condition : initial_milk_volume = 10
axiom pure_milk_condition : pure_milk_volume = 15
axiom initial_water_condition : initial_water_percentage = 5 / 100

-- Prove that target percentage of water in the milk is 2%
theorem target_water_percentage_is_two_percent :
  target_water_percentage = 2 / 100 := by
  sorry

end NUMINAMATH_GPT_target_water_percentage_is_two_percent_l651_65148


namespace NUMINAMATH_GPT_problem_1_problem_2_l651_65140

noncomputable def f (a x : ℝ) : ℝ := |x + a| + |x + 1/a|

theorem problem_1 (x : ℝ) : f 2 x > 3 ↔ x < -(11 / 4) ∨ x > 1 / 4 := sorry

theorem problem_2 (a m : ℝ) (ha : a > 0) : f a m + f a (-1 / m) ≥ 4 := sorry

end NUMINAMATH_GPT_problem_1_problem_2_l651_65140


namespace NUMINAMATH_GPT_scientific_notation_of_1653_billion_l651_65158

theorem scientific_notation_of_1653_billion :
  (1653 * (10 ^ 9) = 1.6553 * (10 ^ 12)) :=
sorry

end NUMINAMATH_GPT_scientific_notation_of_1653_billion_l651_65158


namespace NUMINAMATH_GPT_minibus_seat_count_l651_65132

theorem minibus_seat_count 
  (total_children : ℕ) 
  (seats_with_3_children : ℕ) 
  (children_per_3_child_seat : ℕ) 
  (remaining_children : ℕ) 
  (children_per_2_child_seat : ℕ) 
  (total_seats : ℕ) :
  total_children = 19 →
  seats_with_3_children = 5 →
  children_per_3_child_seat = 3 →
  remaining_children = total_children - (seats_with_3_children * children_per_3_child_seat) →
  children_per_2_child_seat = 2 →
  total_seats = seats_with_3_children + (remaining_children / children_per_2_child_seat) →
  total_seats = 7 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_minibus_seat_count_l651_65132


namespace NUMINAMATH_GPT_triangle_count_relationship_l651_65173

theorem triangle_count_relationship :
  let n_0 : ℕ := 20
  let n_1 : ℕ := 19
  let n_2 : ℕ := 18
  n_0 > n_1 ∧ n_1 > n_2 :=
by
  let n_0 := 20
  let n_1 := 19
  let n_2 := 18
  have h0 : n_0 > n_1 := by sorry
  have h1 : n_1 > n_2 := by sorry
  exact ⟨h0, h1⟩

end NUMINAMATH_GPT_triangle_count_relationship_l651_65173


namespace NUMINAMATH_GPT_product_of_g_xi_l651_65187

noncomputable def x1 : ℂ := sorry
noncomputable def x2 : ℂ := sorry
noncomputable def x3 : ℂ := sorry
noncomputable def x4 : ℂ := sorry
noncomputable def x5 : ℂ := sorry

def f (x : ℂ) : ℂ := x^5 + x^2 + 1
def g (x : ℂ) : ℂ := x^3 - 2

axiom roots_of_f (x : ℂ) : f x = 0 ↔ x = x1 ∨ x = x2 ∨ x = x3 ∨ x = x4 ∨ x = x5

theorem product_of_g_xi : (g x1) * (g x2) * (g x3) * (g x4) * (g x5) = -243 := sorry

end NUMINAMATH_GPT_product_of_g_xi_l651_65187


namespace NUMINAMATH_GPT_band_total_earnings_l651_65110

variables (earnings_per_gig_per_member : ℕ)
variables (number_of_members : ℕ)
variables (number_of_gigs : ℕ)

theorem band_total_earnings :
  earnings_per_gig_per_member = 20 →
  number_of_members = 4 →
  number_of_gigs = 5 →
  earnings_per_gig_per_member * number_of_members * number_of_gigs = 400 :=
by
  intros
  sorry

end NUMINAMATH_GPT_band_total_earnings_l651_65110


namespace NUMINAMATH_GPT_divisibility_of_product_l651_65137

theorem divisibility_of_product (a b : ℕ) (ha : a > 0) (hb : b > 0) (h : (a * b) % 5 = 0) :
  a % 5 = 0 ∨ b % 5 = 0 :=
sorry

end NUMINAMATH_GPT_divisibility_of_product_l651_65137


namespace NUMINAMATH_GPT_crates_of_oranges_l651_65178

theorem crates_of_oranges (C : ℕ) (h1 : ∀ crate, crate = 150) (h2 : ∀ box, box = 30) (num_boxes : ℕ) (total_fruits : ℕ) : 
  num_boxes = 16 → total_fruits = 2280 → 150 * C + 16 * 30 = 2280 → C = 12 :=
by
  intros num_boxes_eq total_fruits_eq fruit_eq
  sorry

end NUMINAMATH_GPT_crates_of_oranges_l651_65178


namespace NUMINAMATH_GPT_length_MN_l651_65198

variables {A B C D M N : Type}
variables {BC AD AB : ℝ} -- Lengths of sides
variables {a b : ℝ}

-- Given conditions
def is_trapezoid (a b BC AD AB : ℝ) : Prop :=
  BC = a ∧ AD = b ∧ AB = AD + BC

-- Given, side AB is divided into 5 equal parts and a line parallel to bases is drawn through the 3rd division point
def is_divided (AB : ℝ) : Prop := ∃ P_1 P_2 P_3 P_4, AB = P_4 + P_3 + P_2 + P_1

-- Prove the length of MN
theorem length_MN (a b : ℝ) (h_trapezoid : is_trapezoid a b BC AD AB) (h_divided : is_divided AB) : 
  MN = (2 * BC + 3 * AD) / 5 :=
sorry

end NUMINAMATH_GPT_length_MN_l651_65198


namespace NUMINAMATH_GPT_product_of_d_l651_65122

theorem product_of_d (d1 d2 : ℕ) (h1 : ∃ k1 : ℤ, 49 - 12 * d1 = k1^2)
  (h2 : ∃ k2 : ℤ, 49 - 12 * d2 = k2^2) (h3 : 0 < d1) (h4 : 0 < d2)
  (h5 : d1 ≠ d2) : d1 * d2 = 8 := 
sorry

end NUMINAMATH_GPT_product_of_d_l651_65122


namespace NUMINAMATH_GPT_total_bill_first_month_l651_65119

theorem total_bill_first_month (F C : ℝ) 
  (h1 : F + C = 50) 
  (h2 : F + 2 * C = 76) 
  (h3 : 2 * C = 2 * C) : 
  F + C = 50 := by
  sorry

end NUMINAMATH_GPT_total_bill_first_month_l651_65119


namespace NUMINAMATH_GPT_regular_polygon_properties_l651_65180

theorem regular_polygon_properties
  (n : ℕ)
  (h1 : (n - 2) * 180 = 3 * 360 + 180)
  (h2 : n > 2) :
  n = 9 ∧ (n - 2) * 180 / n = 140 := by
  sorry

end NUMINAMATH_GPT_regular_polygon_properties_l651_65180


namespace NUMINAMATH_GPT_least_y_solution_l651_65104

theorem least_y_solution :
  (∃ y : ℝ, 3 * y^2 + 5 * y + 2 = 4 ∧ ∀ z : ℝ, 3 * z^2 + 5 * z + 2 = 4 → y ≤ z) →
  ∃ y : ℝ, y = -2 :=
by
  sorry

end NUMINAMATH_GPT_least_y_solution_l651_65104


namespace NUMINAMATH_GPT_Ivan_returns_alive_Ivan_takes_princesses_l651_65174

theorem Ivan_returns_alive (Tsarevnas Koscheis: Finset ℕ) (five_girls: Finset ℕ) 
  (cond1: Tsarevnas.card = 3) (cond2: Koscheis.card = 2) (cond3: five_girls.card = 5)
  (cond4: Tsarevnas ∪ Koscheis = five_girls)
  (cond5: ∀ g ∈ five_girls, ∃ t ∈ Tsarevnas, t ≠ g ∧ ∃ k ∈ Koscheis, k ≠ g)
  (cond6: ∀ girl : ℕ, girl ∈ five_girls → 
          ∃ truth_count : ℕ, 
          (truth_count = (if girl ∈ Tsarevnas then 2 else 3))): 
  ∃ princesses : Finset ℕ, princesses.card = 3 ∧ princesses ⊆ Tsarevnas ∧ ∀ k ∈ Koscheis, k ∉ princesses :=
sorry

theorem Ivan_takes_princesses (Tsarevnas Koscheis: Finset ℕ) (five_girls: Finset ℕ) 
  (cond1: Tsarevnas.card = 3) (cond2: Koscheis.card = 2) (cond3: five_girls.card = 5)
  (cond4: Tsarevnas ∪ Koscheis = five_girls)
  (cond5: ∀ g ∈ five_girls, ∃ t ∈ Tsarevnas, t ≠ g ∧ ∃ k ∈ Koscheis, k ≠ g)
  (cond6 and cond7: ∀ girl1 girl2 girl3 : ℕ, girl1 ≠ girl2 → girl2 ≠ girl3 → girl1 ∈ Tsarevnas → girl2 ∈ Tsarevnas → girl3 ∈ Tsarevnas → 
          ∃ (eldest middle youngest : ℕ), 
              (eldest ∈ Tsarevnas ∧ middle ∈ Tsarevnas ∧ youngest ∈ Tsarevnas) 
          ∧
              (eldest ≠ middle ∧ eldest ≠ youngest ∧ middle ≠ youngest)
          ∧
              (∀ k ∈ Koscheis, k ≠ eldest ∧ k ≠ middle ∧ k ≠ youngest)
  ):
  ∃ princesses : Finset ℕ, 
          princesses.card = 3 ∧ princesses ⊆ Tsarevnas ∧ 
          (∃ eldest ,∃ middle,∃ youngest : ℕ, eldest ∈ princesses ∧ middle ∈ princesses ∧ youngest ∈ princesses ∧ 
                 eldest ≠ middle ∧ eldest ≠ youngest ∧ middle ≠ youngest)
:=
sorry

end NUMINAMATH_GPT_Ivan_returns_alive_Ivan_takes_princesses_l651_65174


namespace NUMINAMATH_GPT_ratio_of_areas_l651_65171

theorem ratio_of_areas 
  (A B C D E F : Type)
  (AB AC AD : ℝ)
  (h1 : AB = 130)
  (h2 : AC = 130)
  (h3 : AD = 26)
  (CF : ℝ)
  (h4 : CF = 91)
  (BD : ℝ)
  (h5 : BD = 104)
  (AF : ℝ)
  (h6 : AF = 221)
  (EF DE BE CE : ℝ)
  (h7 : EF / DE = 91 / 104)
  (h8 : CE / BE = 3.5) :
  EF * CE = 318.5 * DE * BE :=
sorry

end NUMINAMATH_GPT_ratio_of_areas_l651_65171


namespace NUMINAMATH_GPT_parallelogram_area_l651_65113

theorem parallelogram_area (d : ℝ) (h : ℝ) (α : ℝ) (h_d : d = 30) (h_h : h = 20) : 
  ∃ A : ℝ, A = d * h ∧ A = 600 :=
by
  sorry

end NUMINAMATH_GPT_parallelogram_area_l651_65113


namespace NUMINAMATH_GPT_no_real_solution_l651_65131

-- Define the given equation as a function
def equation (x y : ℝ) : ℝ := 3 * x^2 + 5 * y^2 - 9 * x - 20 * y + 30 + 4 * x * y

-- State that the equation equals zero has no real solution.
theorem no_real_solution : ∀ x y : ℝ, equation x y ≠ 0 :=
by sorry

end NUMINAMATH_GPT_no_real_solution_l651_65131


namespace NUMINAMATH_GPT_polynomial_inequality_holds_l651_65126

theorem polynomial_inequality_holds (a : ℝ) : (∀ x : ℝ, x^4 + (a-2)*x^2 + a ≥ 0) ↔ a ≥ 4 - 2 * Real.sqrt 3 := 
by
  sorry

end NUMINAMATH_GPT_polynomial_inequality_holds_l651_65126


namespace NUMINAMATH_GPT_probability_blue_or_green_is_two_thirds_l651_65147

-- Definitions for the given conditions
def blue_faces := 3
def red_faces := 2
def green_faces := 1
def total_faces := blue_faces + red_faces + green_faces
def successful_outcomes := blue_faces + green_faces

-- Probability definition
def probability_blue_or_green := (successful_outcomes : ℚ) / total_faces

-- The theorem we want to prove
theorem probability_blue_or_green_is_two_thirds :
  probability_blue_or_green = (2 / 3 : ℚ) :=
by
  -- here would be the proof steps, but we replace them with sorry as per the instructions
  sorry

end NUMINAMATH_GPT_probability_blue_or_green_is_two_thirds_l651_65147


namespace NUMINAMATH_GPT_martha_cakes_required_l651_65181

-- Conditions
def number_of_children : ℝ := 3.0
def cakes_per_child : ℝ := 18.0

-- The main statement to prove
theorem martha_cakes_required:
  (number_of_children * cakes_per_child) = 54.0 := 
by
  sorry

end NUMINAMATH_GPT_martha_cakes_required_l651_65181


namespace NUMINAMATH_GPT_prob_and_relation_proof_l651_65138

-- Defining conditions
def total_buses : ℕ := 500

def A_on_time : ℕ := 240
def A_not_on_time : ℕ := 20
def B_on_time : ℕ := 210
def B_not_on_time : ℕ := 30

def A_total : ℕ := A_on_time + A_not_on_time
def B_total : ℕ := B_on_time + B_not_on_time

def prob_A_on_time : ℚ := A_on_time / A_total
def prob_B_on_time : ℚ := B_on_time / B_total

-- Defining K^2 calculation
def n : ℕ := total_buses
def a : ℕ := A_on_time
def b : ℕ := A_not_on_time
def c : ℕ := B_on_time
def d : ℕ := B_not_on_time

def K_squared : ℚ :=
  n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

def threshold_90_percent : ℚ := 2.706

-- Lean theorem statement
theorem prob_and_relation_proof :
  prob_A_on_time = 12 / 13 ∧
  prob_B_on_time = 7 / 8 ∧
  K_squared > threshold_90_percent :=
by {
   sorry
}

end NUMINAMATH_GPT_prob_and_relation_proof_l651_65138


namespace NUMINAMATH_GPT_find_a_l651_65160

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry

axiom cond1 : a^2 / b = 5
axiom cond2 : b^2 / c = 3
axiom cond3 : c^2 / a = 7

theorem find_a : a = 15 := sorry

end NUMINAMATH_GPT_find_a_l651_65160


namespace NUMINAMATH_GPT_diameter_of_circumscribed_circle_l651_65112

noncomputable def circumscribed_circle_diameter (a : ℝ) (A : ℝ) : ℝ :=
  a / Real.sin A

theorem diameter_of_circumscribed_circle :
  circumscribed_circle_diameter 15 (Real.pi / 4) = 15 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_diameter_of_circumscribed_circle_l651_65112


namespace NUMINAMATH_GPT_chris_babysitting_hours_l651_65194

theorem chris_babysitting_hours (h : ℕ) (video_game_cost candy_cost earn_per_hour leftover total_cost : ℕ) :
  video_game_cost = 60 ∧
  candy_cost = 5 ∧
  earn_per_hour = 8 ∧
  leftover = 7 ∧
  total_cost = video_game_cost + candy_cost ∧
  earn_per_hour * h = total_cost + leftover
  → h = 9 := by
  intros
  sorry

end NUMINAMATH_GPT_chris_babysitting_hours_l651_65194


namespace NUMINAMATH_GPT_symmetry_P_over_xOz_l651_65105

-- Definition for the point P and the plane xOz
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def P : Point3D := { x := 2, y := 3, z := 4 }

def symmetry_over_xOz_plane (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := p.z }

theorem symmetry_P_over_xOz : symmetry_over_xOz_plane P = { x := 2, y := -3, z := 4 } :=
by
  -- The proof is omitted.
  sorry

end NUMINAMATH_GPT_symmetry_P_over_xOz_l651_65105


namespace NUMINAMATH_GPT_good_pair_bound_all_good_pairs_l651_65135

namespace good_pairs

-- Definition of a "good" pair
def is_good_pair (r s : ℕ) : Prop :=
  ∃ (P : ℤ → ℤ) (a : Fin r → ℤ) (b : Fin s → ℤ),
  (∀ i j : Fin r, i ≠ j → a i ≠ a j) ∧
  (∀ i j : Fin s, i ≠ j → b i ≠ b j) ∧
  (∀ i : Fin r, P (a i) = 2) ∧
  (∀ j : Fin s, P (b j) = 5)

-- (a) Show that for every good pair (r, s), r, s ≤ 3
theorem good_pair_bound (r s : ℕ) (h : is_good_pair r s) : r ≤ 3 ∧ s ≤ 3 :=
sorry

-- (b) Determine all good pairs
theorem all_good_pairs (r s : ℕ) : is_good_pair r s ↔ (r ≤ 3 ∧ s ≤ 3 ∧ (
  (r = 1 ∧ s = 1) ∨ (r = 1 ∧ s = 2) ∨ (r = 1 ∧ s = 3) ∨
  (r = 2 ∧ s = 1) ∨ (r = 2 ∧ s = 2) ∨ (r = 2 ∧ s = 3) ∨
  (r = 3 ∧ s = 1) ∨ (r = 3 ∧ s = 2))) :=
sorry

end good_pairs

end NUMINAMATH_GPT_good_pair_bound_all_good_pairs_l651_65135


namespace NUMINAMATH_GPT_dan_money_left_l651_65134

theorem dan_money_left (initial_amount spent_amount remaining_amount : ℤ) (h1 : initial_amount = 300) (h2 : spent_amount = 100) : remaining_amount = 200 :=
by 
  sorry

end NUMINAMATH_GPT_dan_money_left_l651_65134


namespace NUMINAMATH_GPT_cone_new_height_eq_sqrt_85_l651_65150

/-- A cone has a uniform circular base of radius 6 feet and a slant height of 13 feet.
    After the side breaks, the slant height reduces by 2 feet, making the new slant height 11 feet.
    We need to determine the new height from the base to the tip of the cone, and prove it is sqrt(85). -/
theorem cone_new_height_eq_sqrt_85 :
  let r : ℝ := 6
  let l : ℝ := 13
  let l' : ℝ := 11
  let h : ℝ := Real.sqrt (13^2 - 6^2)
  let H : ℝ := Real.sqrt (11^2 - 6^2)
  H = Real.sqrt 85 :=
by
  sorry


end NUMINAMATH_GPT_cone_new_height_eq_sqrt_85_l651_65150
