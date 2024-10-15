import Mathlib

namespace NUMINAMATH_GPT_cube_truncation_edges_l791_79156

-- Define the initial condition: a cube
def initial_cube_edges : ℕ := 12

-- Define the condition of each corner being cut off
def corners_cut (corners : ℕ) (edges_added : ℕ) : ℕ :=
  corners * edges_added

-- Define the proof problem
theorem cube_truncation_edges : initial_cube_edges + corners_cut 8 3 = 36 := by
  sorry

end NUMINAMATH_GPT_cube_truncation_edges_l791_79156


namespace NUMINAMATH_GPT_john_total_shirts_l791_79112

-- Define initial conditions
def initial_shirts : ℕ := 12
def additional_shirts : ℕ := 4

-- Statement of the problem
theorem john_total_shirts : initial_shirts + additional_shirts = 16 := by
  sorry

end NUMINAMATH_GPT_john_total_shirts_l791_79112


namespace NUMINAMATH_GPT_least_number_to_subtract_l791_79132

theorem least_number_to_subtract (x : ℕ) :
  1439 - x ≡ 3 [MOD 5] ∧ 
  1439 - x ≡ 3 [MOD 11] ∧ 
  1439 - x ≡ 3 [MOD 13] ↔ 
  x = 9 :=
by sorry

end NUMINAMATH_GPT_least_number_to_subtract_l791_79132


namespace NUMINAMATH_GPT_A_minus_B_l791_79188

def A : ℕ := 3^7 + Nat.choose 7 2 * 3^5 + Nat.choose 7 4 * 3^3 + Nat.choose 7 6 * 3
def B : ℕ := Nat.choose 7 1 * 3^6 + Nat.choose 7 3 * 3^4 + Nat.choose 7 5 * 3^2 + 1

theorem A_minus_B : A - B = 128 := by
  sorry

end NUMINAMATH_GPT_A_minus_B_l791_79188


namespace NUMINAMATH_GPT_intersection_A_B_l791_79152

def A : Set ℝ := { x | -1 ≤ x ∧ x ≤ 3 }
def B : Set ℝ := { x | x < 2 }

theorem intersection_A_B : A ∩ B = { x | -1 ≤ x ∧ x < 2 } := 
by 
  sorry

end NUMINAMATH_GPT_intersection_A_B_l791_79152


namespace NUMINAMATH_GPT_chickens_on_farm_are_120_l791_79111

-- Given conditions
def Number_of_hens : ℕ := 52
def Difference_hens_roosters : ℕ := 16

-- Define the number of roosters based on the conditions
def Number_of_roosters : ℕ := Number_of_hens + Difference_hens_roosters

-- The total number of chickens is the sum of hens and roosters
def Total_number_of_chickens : ℕ := Number_of_hens + Number_of_roosters

-- Prove that the total number of chickens is 120
theorem chickens_on_farm_are_120 : Total_number_of_chickens = 120 := by
  -- leave this part unimplemented for proof.
  -- The steps would involve computing the values based on definitions
  sorry

end NUMINAMATH_GPT_chickens_on_farm_are_120_l791_79111


namespace NUMINAMATH_GPT_grace_mowing_hours_l791_79157

-- Definitions for conditions
def earnings_mowing (x : ℕ) : ℕ := 6 * x
def earnings_weeds : ℕ := 11 * 9
def earnings_mulch : ℕ := 9 * 10
def total_september_earnings (x : ℕ) : ℕ := earnings_mowing x + earnings_weeds + earnings_mulch

-- Proof statement (with the total earnings of 567 specified)
theorem grace_mowing_hours (x : ℕ) (h : total_september_earnings x = 567) : x = 63 := by
  sorry

end NUMINAMATH_GPT_grace_mowing_hours_l791_79157


namespace NUMINAMATH_GPT_angle_CBE_minimal_l791_79147

theorem angle_CBE_minimal
    (ABC ABD DBE: ℝ)
    (h1: ABC = 40)
    (h2: ABD = 28)
    (h3: DBE = 10) : 
    CBE = 2 :=
by
  sorry

end NUMINAMATH_GPT_angle_CBE_minimal_l791_79147


namespace NUMINAMATH_GPT_no_six_digit_number_meets_criteria_l791_79109

def valid_digit (n : ℕ) := 2 ≤ n ∧ n ≤ 8

theorem no_six_digit_number_meets_criteria :
  ¬ ∃ (digits : Finset ℕ), digits.card = 6 ∧ (∀ x ∈ digits, valid_digit x) ∧ (digits.sum id = 42) :=
by {
  sorry
}

end NUMINAMATH_GPT_no_six_digit_number_meets_criteria_l791_79109


namespace NUMINAMATH_GPT_no_real_roots_contradiction_l791_79193

open Real

variables (a b : ℝ)

theorem no_real_roots_contradiction (h : ∀ x : ℝ, a * x^3 + a * x + b ≠ 0) : false :=
by
  sorry

end NUMINAMATH_GPT_no_real_roots_contradiction_l791_79193


namespace NUMINAMATH_GPT_hours_of_rain_l791_79165

def totalHours : ℕ := 9
def noRainHours : ℕ := 5
def rainHours : ℕ := totalHours - noRainHours

theorem hours_of_rain : rainHours = 4 := by
  sorry

end NUMINAMATH_GPT_hours_of_rain_l791_79165


namespace NUMINAMATH_GPT_factorize_poly1_l791_79169

variable (a : ℝ)

theorem factorize_poly1 : a^4 + 2 * a^3 + 1 = (a + 1) * (a^3 + a^2 - a + 1) := 
sorry

end NUMINAMATH_GPT_factorize_poly1_l791_79169


namespace NUMINAMATH_GPT_no_possible_numbering_for_equal_sidesum_l791_79144

theorem no_possible_numbering_for_equal_sidesum (O : Point) (A : Fin 10 → Point) 
  (side_numbers : (Fin 10) → ℕ) (segment_numbers : (Fin 10) → ℕ) : 
  ¬ ∃ (side_segment_sum_equal : Fin 10 → ℕ) (sum_equal : ℕ),
    (∀ i, side_segment_sum_equal i = side_numbers i + segment_numbers i) ∧ 
    (∀ i, side_segment_sum_equal i = sum_equal) := 
sorry

end NUMINAMATH_GPT_no_possible_numbering_for_equal_sidesum_l791_79144


namespace NUMINAMATH_GPT_coins_in_second_stack_l791_79180

theorem coins_in_second_stack (total_coins : ℕ) (stack1_coins : ℕ) (stack2_coins : ℕ) 
  (H1 : total_coins = 12) (H2 : stack1_coins = 4) : stack2_coins = 8 :=
by
  -- The proof is omitted.
  sorry

end NUMINAMATH_GPT_coins_in_second_stack_l791_79180


namespace NUMINAMATH_GPT_triangle_similarity_length_RY_l791_79153

theorem triangle_similarity_length_RY
  (P Q R X Y Z : Type)
  [MetricSpace P] [MetricSpace Q] [MetricSpace R]
  [MetricSpace X] [MetricSpace Y] [MetricSpace Z]
  (PQ : ℝ) (XY : ℝ) (RY_length : ℝ)
  (h1 : PQ = 10)
  (h2 : XY = 6)
  (h3 : ∀ (PR QR PX QX RZ : ℝ) (angle_PY_RZ : ℝ),
    PR + RY_length = PX ∧
    QR + RY_length = QX ∧ 
    angle_PY_RZ = 120 ∧
    PR > 0 ∧ QR > 0 ∧ RY_length > 0)
  (h4 : XY / PQ = RY_length / (PQ + RY_length)) :
  RY_length = 15 := by
  sorry

end NUMINAMATH_GPT_triangle_similarity_length_RY_l791_79153


namespace NUMINAMATH_GPT_power_mod_five_l791_79128

theorem power_mod_five (n : ℕ) (hn : n ≡ 0 [MOD 4]): (3^2000 ≡ 1 [MOD 5]) :=
by 
  sorry

end NUMINAMATH_GPT_power_mod_five_l791_79128


namespace NUMINAMATH_GPT_prob1_prob2_l791_79145

theorem prob1:
  (6 * (Real.tan (30 * Real.pi / 180))^2 - Real.sqrt 3 * Real.sin (60 * Real.pi / 180) - 2 * Real.sin (45 * Real.pi / 180)) = (1 / 2 - Real.sqrt 2) :=
sorry

theorem prob2:
  ((Real.sqrt 2 / 2) * Real.cos (45 * Real.pi / 180) - (Real.tan (40 * Real.pi / 180) + 1)^0 + Real.sqrt (1 / 4) + Real.sin (30 * Real.pi / 180)) = (1 / 2) :=
sorry

end NUMINAMATH_GPT_prob1_prob2_l791_79145


namespace NUMINAMATH_GPT_sum_of_consecutive_integers_product_384_l791_79183

theorem sum_of_consecutive_integers_product_384 :
  ∃ (a : ℤ), a * (a + 1) * (a + 2) = 384 ∧ a + (a + 1) + (a + 2) = 24 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_consecutive_integers_product_384_l791_79183


namespace NUMINAMATH_GPT_ratio_rocks_eaten_to_collected_l791_79137

def rocks_collected : ℕ := 10
def rocks_left : ℕ := 7
def rocks_spit_out : ℕ := 2

theorem ratio_rocks_eaten_to_collected : 
  (rocks_collected - rocks_left + rocks_spit_out) * 2 = rocks_collected := 
by 
  sorry

end NUMINAMATH_GPT_ratio_rocks_eaten_to_collected_l791_79137


namespace NUMINAMATH_GPT_power_of_5_in_8_factorial_l791_79117

theorem power_of_5_in_8_factorial :
  let x := Nat.factorial 8
  ∃ (i k m p : ℕ), 0 < i ∧ 0 < k ∧ 0 < m ∧ 0 < p ∧ x = 2^i * 3^k * 5^m * 7^p ∧ m = 1 :=
by
  sorry

end NUMINAMATH_GPT_power_of_5_in_8_factorial_l791_79117


namespace NUMINAMATH_GPT_vector_magnitude_positive_l791_79146

variable {V : Type} [NormedAddCommGroup V] [NormedSpace ℝ V]

variables (a b : V)

-- Given: 
-- a is any non-zero vector
-- b is a unit vector
theorem vector_magnitude_positive (ha : a ≠ 0) (hb : ‖b‖ = 1) : ‖a‖ > 0 := 
sorry

end NUMINAMATH_GPT_vector_magnitude_positive_l791_79146


namespace NUMINAMATH_GPT_c_sum_formula_l791_79107

noncomputable section

def arithmetic_sequence (a : Nat -> ℚ) : Prop :=
  a 3 = 2 ∧ (a 1 + 2 * ((a 2 - a 1) : ℚ)) = 2

def geometric_sequence (b : Nat -> ℚ) (a : Nat -> ℚ) : Prop :=
  b 1 = a 1 ∧ b 4 = a 15

def c_sequence (a : Nat -> ℚ) (b : Nat -> ℚ) (n : Nat) : ℚ :=
  a n + b n

def Tn (c : Nat -> ℚ) (n : Nat) : ℚ :=
  (Finset.range n).sum c

theorem c_sum_formula
  (a b c : Nat -> ℚ)
  (k : Nat) 
  (ha : arithmetic_sequence a)
  (hb : geometric_sequence b a)
  (hc : ∀ n, c n = c_sequence a b n) :
  Tn c k = k * (k + 3) / 4 + 2^k - 1 :=
by
  sorry

end NUMINAMATH_GPT_c_sum_formula_l791_79107


namespace NUMINAMATH_GPT_sequence_general_term_l791_79101

theorem sequence_general_term (a : ℕ → ℤ) (h1 : a 1 = 1) (h_rec : ∀ n, a (n + 1) = 2 * a n + 1) :
  ∀ n, a n = (2 ^ n) - 1 := 
sorry

end NUMINAMATH_GPT_sequence_general_term_l791_79101


namespace NUMINAMATH_GPT_distance_each_player_runs_l791_79175

-- Definitions based on conditions
def length : ℝ := 100
def width : ℝ := 50
def laps : ℝ := 6

def perimeter (l w : ℝ) : ℝ := 2 * (l + w)

def total_distance (l w laps : ℝ) : ℝ := laps * perimeter l w

-- Theorem statement
theorem distance_each_player_runs :
  total_distance length width laps = 1800 := 
by 
  sorry

end NUMINAMATH_GPT_distance_each_player_runs_l791_79175


namespace NUMINAMATH_GPT_Taehyung_mother_age_l791_79103

theorem Taehyung_mother_age (Taehyung_young_brother_age : ℕ) (Taehyung_age_diff : ℕ) (Mother_age_diff : ℕ) (H1 : Taehyung_young_brother_age = 7) (H2 : Taehyung_age_diff = 5) (H3 : Mother_age_diff = 31) :
  ∃ (Mother_age : ℕ), Mother_age = 43 := 
by
  have Taehyung_age : ℕ := Taehyung_young_brother_age + Taehyung_age_diff
  have Mother_age := Taehyung_age + Mother_age_diff
  existsi (Mother_age)
  sorry

end NUMINAMATH_GPT_Taehyung_mother_age_l791_79103


namespace NUMINAMATH_GPT_calculate_sum_of_squares_l791_79185

theorem calculate_sum_of_squares :
  23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 268 := 
  sorry

end NUMINAMATH_GPT_calculate_sum_of_squares_l791_79185


namespace NUMINAMATH_GPT_verify_n_l791_79105

noncomputable def find_n (n : ℕ) : Prop :=
  let widget_rate1 := 3                             -- Widgets per worker-hour from the first condition
  let whoosit_rate1 := 2                            -- Whoosits per worker-hour from the first condition
  let widget_rate3 := 1                             -- Widgets per worker-hour from the third condition
  let minutes_per_widget := 1                       -- Arbitrary unit time for one widget
  let minutes_per_whoosit := 2                      -- 2 times unit time for one whoosit based on problem statement
  let whoosit_rate3 := 2 / 3                        -- Whoosits per worker-hour from the third condition
  let widget_rate2 := 540 / (90 * 3 : ℕ)            -- Widgets per hour in the second condition
  let whoosit_rate2 := n / (90 * 3 : ℕ)             -- Whoosits per hour in the second condition
  widget_rate2 = 2 ∧ whoosit_rate2 = 4 / 3 ∧
  (minutes_per_widget < minutes_per_whoosit) ∧
  (whoosit_rate2 = (4 / 3 : ℚ) ↔ n = 360)

theorem verify_n : find_n 360 :=
by sorry

end NUMINAMATH_GPT_verify_n_l791_79105


namespace NUMINAMATH_GPT_proof_problem_l791_79171

-- Define the rates of P and Q
def P_rate : ℚ := 1/3
def Q_rate : ℚ := 1/18

-- Define the time they work together
def combined_time : ℚ := 2

-- Define the job completion rates
def combined_rate (P_rate Q_rate : ℚ) : ℚ := P_rate + Q_rate

-- Define the job completed together in given time
def job_completed_together (rate time : ℚ) : ℚ := rate * time

-- Define the remaining job
def remaining_job (total_job completed_job : ℚ) : ℚ := total_job - completed_job

-- Define the time required for P to complete the remaining job
def time_for_P (P_rate remaining_job : ℚ) : ℚ := remaining_job / P_rate

-- Define the total job as 1
def total_job : ℚ := 1

-- Correct answer in minutes
def correct_answer_in_minutes (time_in_hours : ℚ) : ℚ := time_in_hours * 60

-- Problem statement
theorem proof_problem : 
  correct_answer_in_minutes (time_for_P P_rate (remaining_job total_job 
    (job_completed_together (combined_rate P_rate Q_rate) combined_time))) = 40 := 
by
  sorry

end NUMINAMATH_GPT_proof_problem_l791_79171


namespace NUMINAMATH_GPT_greatest_value_of_sum_l791_79138

theorem greatest_value_of_sum (x y : ℝ) (h₁ : x^2 + y^2 = 100) (h₂ : x * y = 40) :
  x + y = 6 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_greatest_value_of_sum_l791_79138


namespace NUMINAMATH_GPT_tan_identity_proof_l791_79179

theorem tan_identity_proof :
  (1 - Real.tan (100 * Real.pi / 180)) * (1 - Real.tan (35 * Real.pi / 180)) = 2 :=
by
  have tan_135 : Real.tan (135 * Real.pi / 180) = -1 := by sorry -- This needs a separate proof.
  have tan_sum_formula : ∀ A B : ℝ, Real.tan (A + B) = (Real.tan A + Real.tan B) / (1 - Real.tan A * Real.tan B) := by sorry -- This needs a deeper exploration
  sorry -- Main proof to be filled

end NUMINAMATH_GPT_tan_identity_proof_l791_79179


namespace NUMINAMATH_GPT_no_integer_solution_for_large_n_l791_79174

theorem no_integer_solution_for_large_n (n : ℕ) (m : ℤ) (h : n ≥ 11) : ¬(m^2 + 2 * 3^n = m * (2^(n+1) - 1)) :=
sorry

end NUMINAMATH_GPT_no_integer_solution_for_large_n_l791_79174


namespace NUMINAMATH_GPT_total_votes_polled_l791_79184

theorem total_votes_polled (V: ℝ) (h: 0 < V) (h1: 0.70 * V - 0.30 * V = 320) : V = 800 :=
sorry

end NUMINAMATH_GPT_total_votes_polled_l791_79184


namespace NUMINAMATH_GPT_geom_series_ratio_l791_79166

noncomputable def geomSeries (a q : ℝ) (n : ℕ) : ℝ :=
a * ((1 - q ^ n) / (1 - q))

theorem geom_series_ratio (a1 q : ℝ) (h : 8 * a1 * q + a1 * q^4 = 0) :
  (geomSeries a1 q 5) / (geomSeries a1 q 2) = -11 :=
sorry

end NUMINAMATH_GPT_geom_series_ratio_l791_79166


namespace NUMINAMATH_GPT_find_side_AB_l791_79151

theorem find_side_AB 
  (B C : ℝ) (BC : ℝ) (hB : B = 45) (hC : C = 45) (hBC : BC = 10) : 
  ∃ AB : ℝ, AB = 5 * Real.sqrt 2 :=
by
  -- We add 'sorry' here to indicate that the proof is not provided.
  sorry

end NUMINAMATH_GPT_find_side_AB_l791_79151


namespace NUMINAMATH_GPT_coin_toss_probability_l791_79199

-- Define the sample space of the coin toss
inductive Coin
| heads : Coin
| tails : Coin

-- Define the probability function
def probability (outcome : Coin) : ℝ :=
  match outcome with
  | Coin.heads => 0.5
  | Coin.tails => 0.5

-- The theorem to be proved: In a fair coin toss, the probability of getting "heads" or "tails" is 0.5
theorem coin_toss_probability (outcome : Coin) : probability outcome = 0.5 :=
sorry

end NUMINAMATH_GPT_coin_toss_probability_l791_79199


namespace NUMINAMATH_GPT_unique_y_for_diamond_l791_79100

def diamond (x y : ℝ) : ℝ := 5 * x - 4 * y + 2 * x * y + 1

theorem unique_y_for_diamond :
  ∃! y : ℝ, diamond 4 y = 21 :=
by
  sorry

end NUMINAMATH_GPT_unique_y_for_diamond_l791_79100


namespace NUMINAMATH_GPT_max_positive_root_satisfies_range_l791_79178

noncomputable def max_positive_root_in_range (b c d : ℝ) (hb : |b| ≤ 1) (hc : |c| ≤ 1.5) (hd : |d| ≤ 1) : Prop :=
  ∃ s : ℝ, 2.5 ≤ s ∧ s < 3 ∧ ∃ x : ℝ, x > 0 ∧ x^3 + b * x^2 + c * x + d = 0

theorem max_positive_root_satisfies_range (b c d : ℝ) (hb : |b| ≤ 1) (hc : |c| ≤ 1.5) (hd : |d| ≤ 1) :
  max_positive_root_in_range b c d hb hc hd := sorry

end NUMINAMATH_GPT_max_positive_root_satisfies_range_l791_79178


namespace NUMINAMATH_GPT_parallel_lines_iff_a_eq_2_l791_79143

-- Define line equations
def l1 (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y - a + 1 = 0
def l2 (a : ℝ) (x y : ℝ) : Prop := x + (a - 1) * y - 2 = 0

-- Prove that a = 2 is necessary and sufficient for the lines to be parallel.
theorem parallel_lines_iff_a_eq_2 (a : ℝ) :
  (∀ x y : ℝ, l1 a x y → ∃ u v : ℝ, l2 a u v → x = u ∧ y = v) ↔ (a = 2) :=
by {
  sorry
}

end NUMINAMATH_GPT_parallel_lines_iff_a_eq_2_l791_79143


namespace NUMINAMATH_GPT_scientific_notation_equivalence_l791_79134

/-- The scientific notation for 20.26 thousand hectares in square meters is equal to 2.026 × 10^9. -/
theorem scientific_notation_equivalence :
  (20.26 * 10^3 * 10^4) = 2.026 * 10^9 := 
sorry

end NUMINAMATH_GPT_scientific_notation_equivalence_l791_79134


namespace NUMINAMATH_GPT_x_value_l791_79159

theorem x_value (x : ℤ) (h : x = (2009^2 - 2009) / 2009) : x = 2008 := by
  sorry

end NUMINAMATH_GPT_x_value_l791_79159


namespace NUMINAMATH_GPT_valid_factorizations_of_1870_l791_79141

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_valid_factor1 (n : ℕ) : Prop := 
  ∃ p1 p2 : ℕ, is_prime p1 ∧ is_prime p2 ∧ n = p1 * p2

def is_valid_factor2 (n : ℕ) : Prop := 
  ∃ (p k : ℕ), is_prime p ∧ (k = 4 ∨ k = 6 ∨ k = 8 ∨ k = 9) ∧ n = p * k

theorem valid_factorizations_of_1870 : 
  ∃ a b : ℕ, a * b = 1870 ∧ 10 ≤ a ∧ a ≤ 99 ∧ 10 ≤ b ∧ b ≤ 99 ∧ 
  ((is_valid_factor1 a ∧ is_valid_factor2 b) ∨ (is_valid_factor1 b ∧ is_valid_factor2 a)) ∧ 
  (a = 34 ∧ b = 55 ∨ a = 55 ∧ b = 34) ∧ 
  (¬∃ x y : ℕ, x * y = 1870 ∧ 10 ≤ x ∧ x ≤ 99 ∧ 10 ≤ y ∧ y ≤ 99 ∧ 
  ((is_valid_factor1 x ∧ is_valid_factor2 y) ∨ (is_valid_factor1 y ∧ is_valid_factor2 x)) ∧ 
  (x ≠ 34 ∨ y ≠ 55 ∨ x ≠ 55 ∨ y ≠ 34)) :=
sorry

end NUMINAMATH_GPT_valid_factorizations_of_1870_l791_79141


namespace NUMINAMATH_GPT_closest_multiple_of_12_l791_79189

def is_multiple_of (n m : ℕ) : Prop := ∃ k, n = m * k

-- Define the closest multiple of 4 to 2050 (2048 and 2052)
def closest_multiple_of_4 (n m : ℕ) : ℕ :=
if n % 4 < m % 4 then n - (n % 4)
else m + (4 - (m % 4))

-- Define the conditions for being divisible by both 3 and 4
def is_multiple_of_12 (n : ℕ) : Prop := is_multiple_of n 12

-- Theorem statement
theorem closest_multiple_of_12 (n m : ℕ) (h : n = 2050) (hm : m = 2052) :
  is_multiple_of_12 m :=
sorry

end NUMINAMATH_GPT_closest_multiple_of_12_l791_79189


namespace NUMINAMATH_GPT_gcd_of_polynomial_and_multiple_of_12600_l791_79142

theorem gcd_of_polynomial_and_multiple_of_12600 (x : ℕ) (h : 12600 ∣ x) : gcd ((5 * x + 7) * (11 * x + 3) * (17 * x + 8) * (4 * x + 5)) x = 840 := by
  sorry

end NUMINAMATH_GPT_gcd_of_polynomial_and_multiple_of_12600_l791_79142


namespace NUMINAMATH_GPT_sharon_trip_distance_l791_79125

theorem sharon_trip_distance
  (x : ℝ)
  (usual_speed : ℝ := x / 180)
  (reduced_speed : ℝ := usual_speed - 1/3)
  (time_before_storm : ℝ := (x / 3) / usual_speed)
  (time_during_storm : ℝ := (2 * x / 3) / reduced_speed)
  (total_trip_time : ℝ := 276)
  (h : time_before_storm + time_during_storm = total_trip_time) :
  x = 135 :=
sorry

end NUMINAMATH_GPT_sharon_trip_distance_l791_79125


namespace NUMINAMATH_GPT_fraction_of_repeating_decimal_l791_79139

theorem fraction_of_repeating_decimal :
  ∃ (f : ℚ), f = 0.73 ∧ f = 73 / 99 := by
  sorry

end NUMINAMATH_GPT_fraction_of_repeating_decimal_l791_79139


namespace NUMINAMATH_GPT_water_usage_l791_79172

noncomputable def litres_per_household_per_month (total_litres : ℕ) (number_of_households : ℕ) : ℕ :=
  total_litres / number_of_households

theorem water_usage : litres_per_household_per_month 2000 10 = 200 :=
by
  sorry

end NUMINAMATH_GPT_water_usage_l791_79172


namespace NUMINAMATH_GPT_cat_weights_ratio_l791_79148

variable (meg_cat_weight : ℕ) (anne_extra_weight : ℕ) (meg_cat_weight := 20) (anne_extra_weight := 8)

/-- The ratio of the weight of Meg's cat to the weight of Anne's cat -/
theorem cat_weights_ratio : (meg_cat_weight / Nat.gcd meg_cat_weight (meg_cat_weight + anne_extra_weight)) 
                            = 5 ∧ ((meg_cat_weight + anne_extra_weight) / Nat.gcd meg_cat_weight (meg_cat_weight + anne_extra_weight)) 
                            = 7 := by
  sorry

end NUMINAMATH_GPT_cat_weights_ratio_l791_79148


namespace NUMINAMATH_GPT_integer_roots_of_quadratic_l791_79195

theorem integer_roots_of_quadratic (a : ℤ) : 
  (∃ x : ℤ , x^2 + a * x + a = 0) ↔ (a = 0 ∨ a = 4) := 
sorry

end NUMINAMATH_GPT_integer_roots_of_quadratic_l791_79195


namespace NUMINAMATH_GPT_volume_of_orange_concentrate_l791_79198

theorem volume_of_orange_concentrate
  (h_jug : ℝ := 8) -- height of the jug in inches
  (d_jug : ℝ := 3) -- diameter of the jug in inches
  (fraction_full : ℝ := 3 / 4) -- jug is three-quarters full
  (ratio_concentrate_to_water : ℝ := 1 / 5) -- ratio of concentrate to water
  : abs ((fraction_full * π * ((d_jug / 2)^2) * h_jug * (1 / (1 + ratio_concentrate_to_water))) - 2.25) < 0.01 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_orange_concentrate_l791_79198


namespace NUMINAMATH_GPT_marbles_left_calculation_l791_79127

/-- A magician starts with 20 red marbles and 30 blue marbles.
    He removes 3 red marbles and 12 blue marbles. We need to 
    prove that he has 35 marbles left in total. -/
theorem marbles_left_calculation (initial_red : ℕ) (initial_blue : ℕ) (removed_red : ℕ) 
    (removed_blue : ℕ) (H1 : initial_red = 20) (H2 : initial_blue = 30) 
    (H3 : removed_red = 3) (H4 : removed_blue = 4 * removed_red) :
    (initial_red - removed_red) + (initial_blue - removed_blue) = 35 :=
by
   -- sorry to skip the proof
   sorry

end NUMINAMATH_GPT_marbles_left_calculation_l791_79127


namespace NUMINAMATH_GPT_ratio_of_discounted_bricks_l791_79122

theorem ratio_of_discounted_bricks (total_bricks discounted_price full_price total_spending: ℝ) 
  (h1 : total_bricks = 1000) 
  (h2 : discounted_price = 0.25) 
  (h3 : full_price = 0.50) 
  (h4 : total_spending = 375) : 
  ∃ D : ℝ, (D / total_bricks = 1 / 2) ∧ (0.25 * D + 0.50 * (total_bricks - D) = total_spending) := 
  sorry

end NUMINAMATH_GPT_ratio_of_discounted_bricks_l791_79122


namespace NUMINAMATH_GPT_no_intersection_points_of_polar_graphs_l791_79131

theorem no_intersection_points_of_polar_graphs :
  let c1_center := (3 / 2, 0)
  let r1 := 3 / 2
  let c2_center := (0, 3)
  let r2 := 3
  let distance_between_centers := Real.sqrt ((3 / 2 - 0) ^ 2 + (0 - 3) ^ 2)
  distance_between_centers > r1 + r2 :=
by
  sorry

end NUMINAMATH_GPT_no_intersection_points_of_polar_graphs_l791_79131


namespace NUMINAMATH_GPT_range_of_c_over_a_l791_79176

theorem range_of_c_over_a (a b c : ℝ) (h1 : a > b) (h2 : b > 1) (h3 : a + b + c = 0) : -2 < c / a ∧ c / a < -1 :=
by {
  sorry
}

end NUMINAMATH_GPT_range_of_c_over_a_l791_79176


namespace NUMINAMATH_GPT_spinner_probabilities_l791_79149

theorem spinner_probabilities (pA pB pC pD : ℚ) (h1 : pA = 1/4) (h2 : pB = 1/3) (h3 : pA + pB + pC + pD = 1) :
  pC + pD = 5/12 :=
by
  -- Here you would construct the proof (left as sorry for this example)
  sorry

end NUMINAMATH_GPT_spinner_probabilities_l791_79149


namespace NUMINAMATH_GPT_polynomial_transformation_l791_79123

theorem polynomial_transformation (g : Polynomial ℝ) (x : ℝ)
  (h : g.eval (x^2 + 2) = x^4 + 6 * x^2 + 8 * x) : 
  g.eval (x^2 - 1) = x^4 - 1 := by
  sorry

end NUMINAMATH_GPT_polynomial_transformation_l791_79123


namespace NUMINAMATH_GPT_systematic_sampling_example_l791_79177

theorem systematic_sampling_example : 
  ∃ (a : ℕ → ℕ), (∀ i : ℕ, 5 ≤ i ∧ i ≤ 5 → a i = 5 + 10 * (i - 1)) ∧ 
  ∀ i : ℕ, 1 ≤ i ∧ i < 6 → a i - a (i - 1) = a (i + 1) - a i :=
sorry

end NUMINAMATH_GPT_systematic_sampling_example_l791_79177


namespace NUMINAMATH_GPT_correct_weight_of_misread_boy_l791_79187

variable (num_boys : ℕ) (avg_weight_incorrect : ℝ) (misread_weight : ℝ) (avg_weight_correct : ℝ)

theorem correct_weight_of_misread_boy
  (h1 : num_boys = 20)
  (h2 : avg_weight_incorrect = 58.4)
  (h3 : misread_weight = 56)
  (h4 : avg_weight_correct = 58.6) : 
  misread_weight + (num_boys * avg_weight_correct - num_boys * avg_weight_incorrect) / num_boys = 60 := 
by 
  -- skipping proof
  sorry

end NUMINAMATH_GPT_correct_weight_of_misread_boy_l791_79187


namespace NUMINAMATH_GPT_calculate_selling_prices_l791_79161

noncomputable def selling_prices
  (cost1 cost2 cost3 : ℝ) (profit1 profit2 profit3 : ℝ) : ℝ × ℝ × ℝ :=
  let selling_price1 := cost1 + (profit1 / 100) * cost1
  let selling_price2 := cost2 + (profit2 / 100) * cost2
  let selling_price3 := cost3 + (profit3 / 100) * cost3
  (selling_price1, selling_price2, selling_price3)

theorem calculate_selling_prices :
  selling_prices 500 750 1000 20 25 30 = (600, 937.5, 1300) :=
by
  sorry

end NUMINAMATH_GPT_calculate_selling_prices_l791_79161


namespace NUMINAMATH_GPT_max_trading_cards_l791_79119

theorem max_trading_cards (h : 10 ≥ 1.25 * nat):
  nat ≤ 8 :=
sorry

end NUMINAMATH_GPT_max_trading_cards_l791_79119


namespace NUMINAMATH_GPT_ratio_of_b_to_a_is_4_l791_79160

theorem ratio_of_b_to_a_is_4 (b a : ℚ) (h1 : b = 4 * a) (h2 : b = 15 - 4 * a) : a = 15 / 8 := by
  sorry

end NUMINAMATH_GPT_ratio_of_b_to_a_is_4_l791_79160


namespace NUMINAMATH_GPT_calories_needed_l791_79158

def calories_per_orange : ℕ := 80
def cost_per_orange : ℝ := 1.2
def initial_amount : ℝ := 10
def remaining_amount : ℝ := 4

theorem calories_needed : calories_per_orange * (initial_amount - remaining_amount) / cost_per_orange = 400 := 
by 
  sorry

end NUMINAMATH_GPT_calories_needed_l791_79158


namespace NUMINAMATH_GPT_problem_ab_cd_l791_79167

theorem problem_ab_cd
    (a b c d : ℝ)
    (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
    (habcd : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
    (h1 : (a^2012 - c^2012) * (a^2012 - d^2012) = 2012)
    (h2 : (b^2012 - c^2012) * (b^2012 - d^2012) = 2012) :
  (ab)^2012 - (cd)^2012 = -2012 := 
sorry

end NUMINAMATH_GPT_problem_ab_cd_l791_79167


namespace NUMINAMATH_GPT_min_value_expression_l791_79164

theorem min_value_expression (x : ℝ) (h : x > 3) : x + 4 / (x - 3) ≥ 7 :=
sorry

end NUMINAMATH_GPT_min_value_expression_l791_79164


namespace NUMINAMATH_GPT_orange_slices_needed_l791_79191

theorem orange_slices_needed (total_slices containers_capacity leftover_slices: ℕ) 
(h1 : containers_capacity = 4) 
(h2 : total_slices = 329) 
(h3 : leftover_slices = 1) :
    containers_capacity - leftover_slices = 3 :=
by
  sorry

end NUMINAMATH_GPT_orange_slices_needed_l791_79191


namespace NUMINAMATH_GPT_two_digit_number_reverse_sum_eq_99_l791_79173

theorem two_digit_number_reverse_sum_eq_99 :
  ∃ a b : ℕ, (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ ((10 * a + b) - (10 * b + a) = 5 * (a + b))
  ∧ (10 * a + b) + (10 * b + a) = 99 := 
by
  sorry

end NUMINAMATH_GPT_two_digit_number_reverse_sum_eq_99_l791_79173


namespace NUMINAMATH_GPT_learn_at_least_537_words_l791_79135

theorem learn_at_least_537_words (total_words : ℕ) (guess_percentage : ℝ) (required_percentage : ℝ) :
  total_words = 600 → guess_percentage = 0.05 → required_percentage = 0.90 → 
  ∀ (words_learned : ℕ), words_learned ≥ 537 → 
  (words_learned + guess_percentage * (total_words - words_learned)) / total_words ≥ required_percentage :=
by
  intros h_total_words h_guess_percentage h_required_percentage words_learned h_words_learned
  sorry

end NUMINAMATH_GPT_learn_at_least_537_words_l791_79135


namespace NUMINAMATH_GPT_guise_hot_dogs_l791_79133

theorem guise_hot_dogs (x : ℤ) (h1 : x + (x + 2) + (x + 4) = 36) : x = 10 :=
by
  sorry

end NUMINAMATH_GPT_guise_hot_dogs_l791_79133


namespace NUMINAMATH_GPT_find_c_l791_79181

theorem find_c (c d : ℝ) (h1 : c < 0) (h2 : d > 0)
    (max_min_condition : ∀ x, c * Real.cos (d * x) ≤ 3 ∧ c * Real.cos (d * x) ≥ -3) :
    c = -3 :=
by
  -- The statement says if c < 0, d > 0, and given the cosine function hitting max 3 and min -3, then c = -3.
  sorry

end NUMINAMATH_GPT_find_c_l791_79181


namespace NUMINAMATH_GPT_find_n_l791_79118

theorem find_n (n : ℕ) (h1 : 0 < n) : 
  ∃ n, n > 0 ∧ (Real.tan (Real.pi / (2 * n)) + Real.sin (Real.pi / (2 * n)) = n / 3) := 
sorry

end NUMINAMATH_GPT_find_n_l791_79118


namespace NUMINAMATH_GPT_car_cleaning_ratio_l791_79136

theorem car_cleaning_ratio
    (outside_cleaning_time : ℕ)
    (total_cleaning_time : ℕ)
    (h1 : outside_cleaning_time = 80)
    (h2 : total_cleaning_time = 100) :
    (total_cleaning_time - outside_cleaning_time) / outside_cleaning_time = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_car_cleaning_ratio_l791_79136


namespace NUMINAMATH_GPT_sum_of_integers_with_even_product_l791_79162

theorem sum_of_integers_with_even_product (a b : ℤ) (h : ∃ k, a * b = 2 * k) : 
∃ k1 k2, a = 2 * k1 ∨ a = 2 * k1 + 1 ∧ (a + b = 2 * k2 ∨ a + b = 2 * k2 + 1) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_integers_with_even_product_l791_79162


namespace NUMINAMATH_GPT_sqrt_expression_eq_twelve_l791_79140

theorem sqrt_expression_eq_twelve : Real.sqrt 3 * (Real.sqrt 3 + Real.sqrt 27) = 12 := 
sorry

end NUMINAMATH_GPT_sqrt_expression_eq_twelve_l791_79140


namespace NUMINAMATH_GPT_non_athletic_parents_l791_79120

-- Define the conditions
variables (total_students athletic_dads athletic_moms both_athletic : ℕ)

-- Assume the given conditions
axiom h1 : total_students = 45
axiom h2 : athletic_dads = 17
axiom h3 : athletic_moms = 20
axiom h4 : both_athletic = 11

-- Statement to be proven
theorem non_athletic_parents : total_students - (athletic_dads - both_athletic + athletic_moms - both_athletic + both_athletic) = 19 :=
by {
  -- We intentionally skip the proof here
  sorry
}

end NUMINAMATH_GPT_non_athletic_parents_l791_79120


namespace NUMINAMATH_GPT_find_x_l791_79106

theorem find_x (x : ℝ) (h : x > 0) (area : 1 / 2 * (2 * x) * x = 72) : x = 6 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l791_79106


namespace NUMINAMATH_GPT_scientific_notation_conversion_l791_79110

theorem scientific_notation_conversion : 450000000 = 4.5 * 10^8 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_conversion_l791_79110


namespace NUMINAMATH_GPT_dhoni_leftover_percentage_l791_79121

variable (E : ℝ) (spent_on_rent : ℝ) (spent_on_dishwasher : ℝ)

def percent_spent_on_rent : ℝ := 0.40
def percent_spent_on_dishwasher : ℝ := 0.32

theorem dhoni_leftover_percentage (E : ℝ) :
  (1 - (percent_spent_on_rent + percent_spent_on_dishwasher)) * E / E = 0.28 :=
by
  sorry

end NUMINAMATH_GPT_dhoni_leftover_percentage_l791_79121


namespace NUMINAMATH_GPT_people_left_line_l791_79194

-- Definitions based on the conditions given in the problem
def initial_people := 7
def new_people := 8
def final_people := 11

-- Proof statement
theorem people_left_line (L : ℕ) (h : 7 - L + 8 = 11) : L = 4 :=
by
  -- Adding the proof steps directly skips to the required proof
  sorry

end NUMINAMATH_GPT_people_left_line_l791_79194


namespace NUMINAMATH_GPT_total_discount_is_15_l791_79182

structure Item :=
  (price : ℝ)      -- Regular price
  (discount_rate : ℝ) -- Discount rate in decimal form

def t_shirt : Item := {price := 25, discount_rate := 0.3}
def jeans : Item := {price := 75, discount_rate := 0.1}

def discount (item : Item) : ℝ :=
  item.discount_rate * item.price

def total_discount (items : List Item) : ℝ :=
  items.map discount |>.sum

theorem total_discount_is_15 :
  total_discount [t_shirt, jeans] = 15 := by
  sorry

end NUMINAMATH_GPT_total_discount_is_15_l791_79182


namespace NUMINAMATH_GPT_reciprocal_inverse_proportional_l791_79192

variable {x y k c : ℝ}

-- Given condition: x * y = k
axiom inverse_proportional (h : x * y = k) : ∃ c, (1/x) * (1/y) = c

theorem reciprocal_inverse_proportional (h : x * y = k) :
  ∃ c, (1/x) * (1/y) = c :=
inverse_proportional h

end NUMINAMATH_GPT_reciprocal_inverse_proportional_l791_79192


namespace NUMINAMATH_GPT_bus_driver_hours_worked_l791_79124

-- Definitions based on the problem's conditions.
def regular_rate : ℕ := 20
def regular_hours : ℕ := 40
def overtime_rate : ℕ := regular_rate + (3 * (regular_rate / 4))  -- 75% higher
def total_compensation : ℕ := 1000

-- Theorem statement: The bus driver worked a total of 45 hours last week.
theorem bus_driver_hours_worked : 40 + ((total_compensation - (regular_rate * regular_hours)) / overtime_rate) = 45 := 
by 
  sorry

end NUMINAMATH_GPT_bus_driver_hours_worked_l791_79124


namespace NUMINAMATH_GPT_systematic_sample_first_segment_number_l791_79190

theorem systematic_sample_first_segment_number :
  ∃ a_1 : ℕ, ∀ d k : ℕ, k = 5 → a_1 + (59 - 1) * k = 293 → a_1 = 3 :=
by
  sorry

end NUMINAMATH_GPT_systematic_sample_first_segment_number_l791_79190


namespace NUMINAMATH_GPT_spelling_bee_students_count_l791_79155

theorem spelling_bee_students_count (x : ℕ) (h1 : x / 2 * 1 / 4 * 2 = 30) : x = 240 :=
by
  sorry

end NUMINAMATH_GPT_spelling_bee_students_count_l791_79155


namespace NUMINAMATH_GPT_algebraic_identity_l791_79130

theorem algebraic_identity (a : ℚ) (h : a + a⁻¹ = 3) : a^2 + a⁻¹^2 = 7 := 
  sorry

end NUMINAMATH_GPT_algebraic_identity_l791_79130


namespace NUMINAMATH_GPT_a4_value_l791_79186

axiom a_n : ℕ → ℝ
axiom S_n : ℕ → ℝ
axiom q : ℝ

-- Conditions
axiom a1_eq_1 : a_n 1 = 1
axiom S6_eq_4S3 : S_n 6 = 4 * S_n 3
axiom q_ne_1 : q ≠ 1

-- Arithmetic Sequence Sum Formula
axiom sum_formula : ∀ n, S_n n = (1 - q^n) / (1 - q)

-- nth-term Formula
axiom nth_term_formula : ∀ n, a_n n = a_n 1 * q^(n - 1)

-- Prove the value of the 4th term
theorem a4_value : a_n 4 = 3 := sorry

end NUMINAMATH_GPT_a4_value_l791_79186


namespace NUMINAMATH_GPT_miguel_run_time_before_ariana_catches_up_l791_79163

theorem miguel_run_time_before_ariana_catches_up
  (head_start : ℕ := 20)
  (ariana_speed : ℕ := 6)
  (miguel_speed : ℕ := 4)
  (head_start_distance : ℕ := miguel_speed * head_start)
  (t_catchup : ℕ := (head_start_distance) / (ariana_speed - miguel_speed))
  (total_time : ℕ := t_catchup + head_start) :
  total_time = 60 := sorry

end NUMINAMATH_GPT_miguel_run_time_before_ariana_catches_up_l791_79163


namespace NUMINAMATH_GPT_donald_paul_ratio_l791_79197

-- Let P be the number of bottles Paul drinks in one day.
-- Let D be the number of bottles Donald drinks in one day.
def paul_bottles (P : ℕ) := P = 3
def donald_bottles (D : ℕ) := D = 9

theorem donald_paul_ratio (P D : ℕ) (hP : paul_bottles P) (hD : donald_bottles D) : D / P = 3 :=
by {
  -- Insert proof steps here using the conditions.
  sorry
}

end NUMINAMATH_GPT_donald_paul_ratio_l791_79197


namespace NUMINAMATH_GPT_factorize_expr1_factorize_expr2_l791_79168

-- Define the expressions
def expr1 (m x y : ℝ) : ℝ := 3 * m * x - 6 * m * y
def expr2 (x : ℝ) : ℝ := 1 - 25 * x^2

-- Define the factorized forms
def factorized_expr1 (m x y : ℝ) : ℝ := 3 * m * (x - 2 * y)
def factorized_expr2 (x : ℝ) : ℝ := (1 + 5 * x) * (1 - 5 * x)

-- Proof problems
theorem factorize_expr1 (m x y : ℝ) : expr1 m x y = factorized_expr1 m x y := sorry
theorem factorize_expr2 (x : ℝ) : expr2 x = factorized_expr2 x := sorry

end NUMINAMATH_GPT_factorize_expr1_factorize_expr2_l791_79168


namespace NUMINAMATH_GPT_common_tangent_at_point_l791_79196

theorem common_tangent_at_point (x₀ b : ℝ) 
  (h₁ : 6 * x₀^2 = 6 * x₀) 
  (h₂ : 1 + 2 * x₀^3 = 3 * x₀^2 - b) :
  b = 0 ∨ b = -1 :=
sorry

end NUMINAMATH_GPT_common_tangent_at_point_l791_79196


namespace NUMINAMATH_GPT_range_of_independent_variable_l791_79129

theorem range_of_independent_variable (x : ℝ) : (1 - x > 0) → x < 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_independent_variable_l791_79129


namespace NUMINAMATH_GPT_sum_of_octahedron_faces_l791_79126

theorem sum_of_octahedron_faces (n : ℕ) :
  n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6) + (n+7) = 8 * n + 28 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_octahedron_faces_l791_79126


namespace NUMINAMATH_GPT_problem_1_problem_2_l791_79108

def A (x : ℝ) : Prop := x^2 - 3*x - 10 ≤ 0
def B (m x : ℝ) : Prop := m + 1 ≤ x ∧ x ≤ 2*m - 1

theorem problem_1 (m : ℝ) : (∀ x, B m x → A x)  →  m ≤ 3 := 
sorry

theorem problem_2 (m : ℝ) : (¬ ∃ x, A x ∧ B m x) ↔ (m < 2 ∨ 4 < m) := 
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l791_79108


namespace NUMINAMATH_GPT_average_speed_l791_79150

theorem average_speed (d1 d2 d3 d4 d5 t: ℕ) 
  (h1: d1 = 120) 
  (h2: d2 = 70) 
  (h3: d3 = 90) 
  (h4: d4 = 110) 
  (h5: d5 = 80) 
  (total_time: t = 5): 
  (d1 + d2 + d3 + d4 + d5) / t = 94 := 
by 
  -- proof will go here
  sorry

end NUMINAMATH_GPT_average_speed_l791_79150


namespace NUMINAMATH_GPT_find_abc_sum_l791_79113

-- Definitions and statements directly taken from conditions
def Q1 (x y : ℝ) : Prop := y = x^2 + 51/50
def Q2 (x y : ℝ) : Prop := x = y^2 + 23/2
def common_tangent_rational_slope (a b c : ℤ) : Prop :=
  ∃ (x y : ℝ), (a * x + b * y = c) ∧ (Q1 x y ∨ Q2 x y)

theorem find_abc_sum :
  ∃ (a b c : ℕ), 
    gcd (a) (gcd (b) (c)) = 1 ∧
    common_tangent_rational_slope (a) (b) (c) ∧
    a + b + c = 9 :=
  by sorry

end NUMINAMATH_GPT_find_abc_sum_l791_79113


namespace NUMINAMATH_GPT_certain_number_is_18_l791_79170

theorem certain_number_is_18 (p q : ℚ) (h₁ : 3 / p = 8) (h₂ : p - q = 0.20833333333333334) : 3 / q = 18 :=
sorry

end NUMINAMATH_GPT_certain_number_is_18_l791_79170


namespace NUMINAMATH_GPT_sally_initial_cards_l791_79116

def initial_baseball_cards (t w s a : ℕ) : Prop :=
  a = w + s + t

theorem sally_initial_cards :
  ∃ (initial_cards : ℕ), initial_baseball_cards 9 24 15 initial_cards ∧ initial_cards = 48 :=
by
  use 48
  sorry

end NUMINAMATH_GPT_sally_initial_cards_l791_79116


namespace NUMINAMATH_GPT_negation_of_exists_proposition_l791_79104

theorem negation_of_exists_proposition :
  (¬ ∃ n : ℕ, n^2 > 2^n) → (∀ n : ℕ, n^2 ≤ 2^n) := 
by 
  sorry

end NUMINAMATH_GPT_negation_of_exists_proposition_l791_79104


namespace NUMINAMATH_GPT_instantaneous_velocity_at_t_2_l791_79102

theorem instantaneous_velocity_at_t_2 
  (t : ℝ) (x1 y1 x2 y2: ℝ) : 
  (t = 2) → 
  (x1 = 0) → (y1 = 4) → 
  (x2 = 12) → (y2 = -2) → 
  ((y2 - y1) / (x2 - x1) = -1 / 2) := 
by 
  intros ht hx1 hy1 hx2 hy2
  sorry

end NUMINAMATH_GPT_instantaneous_velocity_at_t_2_l791_79102


namespace NUMINAMATH_GPT_tens_digit_8_pow_2023_l791_79154

theorem tens_digit_8_pow_2023 : (8 ^ 2023 % 100) / 10 % 10 = 1 := 
sorry

end NUMINAMATH_GPT_tens_digit_8_pow_2023_l791_79154


namespace NUMINAMATH_GPT_tan_alpha_eq_one_l791_79115

theorem tan_alpha_eq_one (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h : Real.cos (α + β) = Real.sin (α - β)) : Real.tan α = 1 :=
sorry

end NUMINAMATH_GPT_tan_alpha_eq_one_l791_79115


namespace NUMINAMATH_GPT_max_cars_and_quotient_l791_79114

-- Definition of the problem parameters
def car_length : ℕ := 5
def speed_per_car_length : ℕ := 10
def hour_in_seconds : ℕ := 3600
def one_kilometer_in_meters : ℕ := 1000
def distance_in_meters_per_hour (n : ℕ) : ℕ := (10 * n) * one_kilometer_in_meters
def unit_distance (n : ℕ) : ℕ := car_length * (n + 1)

-- Hypotheses
axiom car_spacing : ∀ n : ℕ, unit_distance n = car_length * (n + 1)
axiom car_speed : ∀ n : ℕ, distance_in_meters_per_hour n = (10 * n) * one_kilometer_in_meters

-- Maximum whole number of cars M that can pass in one hour and the quotient when M is divided by 10
theorem max_cars_and_quotient : ∃ (M : ℕ), M = 3000 ∧ M / 10 = 300 := by
  sorry

end NUMINAMATH_GPT_max_cars_and_quotient_l791_79114
