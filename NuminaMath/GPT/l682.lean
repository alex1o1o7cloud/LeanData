import Mathlib

namespace NUMINAMATH_GPT_problem_solution_l682_68295

theorem problem_solution : (275^2 - 245^2) / 30 = 520 := by
  sorry

end NUMINAMATH_GPT_problem_solution_l682_68295


namespace NUMINAMATH_GPT_probability_divisor_of_60_l682_68223

theorem probability_divisor_of_60 : 
  (∃ n : ℕ, 1 ≤ n ∧ n ≤ 60 ∧ (∃ a b c : ℕ, n = 2 ^ a * 3 ^ b * 5 ^ c ∧ a ≤ 2 ∧ b ≤ 1 ∧ c ≤ 1)) → 
  ∃ p : ℚ, p = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_probability_divisor_of_60_l682_68223


namespace NUMINAMATH_GPT_find_a1_plus_a9_l682_68273

variable (a : ℕ → ℝ) (d : ℝ)

-- condition: arithmetic sequence
def is_arithmetic_seq : Prop := ∀ n, a (n + 1) = a n + d

-- condition: sum of specific terms
def sum_specific_terms : Prop := a 3 + a 4 + a 5 + a 6 + a 7 = 450

-- theorem: prove the desired sum
theorem find_a1_plus_a9 (h1 : is_arithmetic_seq a d) (h2 : sum_specific_terms a) : 
  a 1 + a 9 = 180 :=
  sorry

end NUMINAMATH_GPT_find_a1_plus_a9_l682_68273


namespace NUMINAMATH_GPT_range_of_a_l682_68252

def p (a : ℝ) := ∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → x^2 - a ≥ 0
def q (a : ℝ) := ∃ x₀ : ℝ, x₀^2 + 2 * a * x₀ + 2 - a = 0

theorem range_of_a (a : ℝ) (hp : p a) (hq : q a) : a ≤ -2 ∨ a = 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l682_68252


namespace NUMINAMATH_GPT_find_remaining_rectangle_area_l682_68257

-- Definitions of given areas
def S_DEIH : ℝ := 20
def S_HILK : ℝ := 40
def S_ABHG : ℝ := 126
def S_GHKJ : ℝ := 63
def S_DFMK : ℝ := 161

-- Definition of areas of the remaining rectangle
def S_EFML : ℝ := 101

-- Theorem statement to prove the area of the remaining rectangle
theorem find_remaining_rectangle_area :
  S_DFMK - S_DEIH - S_HILK = S_EFML :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_find_remaining_rectangle_area_l682_68257


namespace NUMINAMATH_GPT_triangular_weight_60_grams_l682_68275

-- Define the weights as variables
variables {R T : ℝ} -- round weights and triangular weights are real numbers

-- Define the conditions as hypotheses
theorem triangular_weight_60_grams
  (h1 : R + T = 3 * R)
  (h2 : 4 * R + T = T + R + 90) :
  T = 60 :=
by
  -- indicate that the actual proof is omitted
  sorry

end NUMINAMATH_GPT_triangular_weight_60_grams_l682_68275


namespace NUMINAMATH_GPT_inequality_proof_l682_68297

theorem inequality_proof (a b c : ℝ) 
    (ha : a > 1) (hb : b > 1) (hc : c > 1) :
    (a^2 / (b - 1)) + (b^2 / (c - 1)) + (c^2 / (a - 1)) ≥ 12 :=
by {
    sorry
}

end NUMINAMATH_GPT_inequality_proof_l682_68297


namespace NUMINAMATH_GPT_enclosed_area_correct_l682_68200

noncomputable def enclosed_area : ℝ :=
  ∫ x in (1/2)..2, (-x + 5/2 - 1/x)

theorem enclosed_area_correct :
  enclosed_area = (15/8) - 2 * Real.log 2 :=
by
  sorry

end NUMINAMATH_GPT_enclosed_area_correct_l682_68200


namespace NUMINAMATH_GPT_five_letter_words_start_end_same_l682_68264

def num_five_letter_words_start_end_same : ℕ :=
  26 ^ 4

theorem five_letter_words_start_end_same :
  num_five_letter_words_start_end_same = 456976 :=
by
  -- Sorry is used as a placeholder for the proof.
  sorry

end NUMINAMATH_GPT_five_letter_words_start_end_same_l682_68264


namespace NUMINAMATH_GPT_determine_pairs_l682_68263

theorem determine_pairs (p : ℕ) (hp: Nat.Prime p) :
  ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ p^x - y^3 = 1 ∧ ((x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 2)) := 
sorry

end NUMINAMATH_GPT_determine_pairs_l682_68263


namespace NUMINAMATH_GPT_theresa_sons_count_l682_68213

theorem theresa_sons_count (total_meat_left : ℕ) (meat_per_plate : ℕ) (frac_left : ℚ) (s : ℕ) :
  total_meat_left = meat_per_plate ∧ meat_per_plate * frac_left * s = 3 → s = 9 :=
by sorry

end NUMINAMATH_GPT_theresa_sons_count_l682_68213


namespace NUMINAMATH_GPT_even_function_implies_f2_eq_neg5_l682_68209

def f (x a : ℝ) : ℝ := (x - a) * (x + 3)

theorem even_function_implies_f2_eq_neg5 (a : ℝ) (h_even : ∀ x : ℝ, f x a = f (-x) a) :
  f 2 a = -5 :=
by
  sorry

end NUMINAMATH_GPT_even_function_implies_f2_eq_neg5_l682_68209


namespace NUMINAMATH_GPT_distinct_factors_1320_l682_68299

theorem distinct_factors_1320 : ∃ (n : ℕ), (n = 24) ∧ (∀ d : ℕ, d ∣ 1320 → d.divisors.card = n) ∧ (1320 = 2^2 * 3 * 5 * 11) :=
sorry

end NUMINAMATH_GPT_distinct_factors_1320_l682_68299


namespace NUMINAMATH_GPT_Rogers_age_more_than_twice_Jills_age_l682_68230

/--
Jill is 20 years old.
Finley is 40 years old.
Roger's age is more than twice Jill's age.
In 15 years, the age difference between Roger and Jill will be 30 years less than Finley's age.
Prove that Roger's age is 5 years more than twice Jill's age.
-/
theorem Rogers_age_more_than_twice_Jills_age 
  (J F : ℕ) (hJ : J = 20) (hF : F = 40) (R x : ℕ)
  (hR : R = 2 * J + x) 
  (age_diff_condition : (R + 15) - (J + 15) = (F + 15) - 30) :
  x = 5 := 
sorry

end NUMINAMATH_GPT_Rogers_age_more_than_twice_Jills_age_l682_68230


namespace NUMINAMATH_GPT_lcm_12_18_l682_68277

theorem lcm_12_18 : Nat.lcm 12 18 = 36 :=
by
  -- Definitions of the conditions
  have h12 : 12 = 2 * 2 * 3 := by norm_num
  have h18 : 18 = 2 * 3 * 3 := by norm_num
  
  -- Calculating LCM using the built-in Nat.lcm
  rw [Nat.lcm_comm]  -- Ordering doesn't matter for lcm
  rw [Nat.lcm, h12, h18]
  -- Prime factorizations checks are implicitly handled
  
  -- Calculate the LCM based on the highest powers from the factorizations
  have lcm_val : 4 * 9 = 36 := by norm_num
  
  -- So, the LCM of 12 and 18 is
  exact lcm_val

end NUMINAMATH_GPT_lcm_12_18_l682_68277


namespace NUMINAMATH_GPT_transformation_1_transformation_2_l682_68207

theorem transformation_1 (x y x' y' : ℝ) 
  (h1 : x' = x / 2) 
  (h2 : y' = y / 3) 
  (eq1 : 5 * x + 2 * y = 0) : 
  5 * x' + 3 * y' = 0 := 
sorry

theorem transformation_2 (x y x' y' : ℝ) 
  (h1 : x' = x / 2) 
  (h2 : y' = y / 3) 
  (eq2 : x^2 + y^2 = 1) : 
  4 * x' ^ 2 + 9 * y' ^ 2 = 1 := 
sorry

end NUMINAMATH_GPT_transformation_1_transformation_2_l682_68207


namespace NUMINAMATH_GPT_profit_diff_is_560_l682_68243

-- Define the initial conditions
def capital_A : ℕ := 8000
def capital_B : ℕ := 10000
def capital_C : ℕ := 12000
def profit_share_B : ℕ := 1400

-- Define the ratio parts
def ratio_A : ℕ := 4
def ratio_B : ℕ := 5
def ratio_C : ℕ := 6

-- Define the value of one part based on B's profit share and ratio part
def value_per_part : ℕ := profit_share_B / ratio_B

-- Define the profit shares of A and C
def profit_share_A : ℕ := ratio_A * value_per_part
def profit_share_C : ℕ := ratio_C * value_per_part

-- Define the difference between the profit shares of A and C
def profit_difference : ℕ := profit_share_C - profit_share_A

-- The theorem to prove
theorem profit_diff_is_560 : profit_difference = 560 := 
by sorry

end NUMINAMATH_GPT_profit_diff_is_560_l682_68243


namespace NUMINAMATH_GPT_roots_g_eq_zero_l682_68268

noncomputable def g : ℝ → ℝ := sorry

theorem roots_g_eq_zero :
  (∀ x : ℝ, g (3 + x) = g (3 - x)) →
  (∀ x : ℝ, g (8 + x) = g (8 - x)) →
  (∀ x : ℝ, g (12 + x) = g (12 - x)) →
  g 0 = 0 →
  ∃ L : ℕ, 
  (∀ k, 0 ≤ k ∧ k ≤ L → g (k * 48) = 0) ∧ 
  (∀ k : ℤ, -1000 ≤ k ∧ k ≤ 1000 → (∃ n : ℕ, k = n * 48)) ∧ 
  L + 1 = 42 := 
by sorry

end NUMINAMATH_GPT_roots_g_eq_zero_l682_68268


namespace NUMINAMATH_GPT_parabola_inequality_l682_68238

theorem parabola_inequality (k : ℝ) : 
  (∀ x : ℝ, 2 * x^2 - 2 * k * x + (k^2 + 2 * k + 2) > x^2 + 2 * k * x - 2 * k^2 - 1) ↔ (-1 < k ∧ k < 3) := 
sorry

end NUMINAMATH_GPT_parabola_inequality_l682_68238


namespace NUMINAMATH_GPT_john_initial_pens_l682_68272

theorem john_initial_pens (P S C : ℝ) (n : ℕ) 
  (h1 : 20 * S = P) 
  (h2 : C = (2 / 3) * S) 
  (h3 : n * C = P)
  (h4 : P > 0) 
  (h5 : S > 0) 
  (h6 : C > 0)
  : n = 30 :=
by
  sorry

end NUMINAMATH_GPT_john_initial_pens_l682_68272


namespace NUMINAMATH_GPT_sum_expression_l682_68260

theorem sum_expression : 3 * 501 + 2 * 501 + 4 * 501 + 500 = 5009 := by
  sorry

end NUMINAMATH_GPT_sum_expression_l682_68260


namespace NUMINAMATH_GPT_matrix_power_is_correct_l682_68217

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -1; 1, 1]
def A_cubed : Matrix (Fin 2) (Fin 2) ℤ := !![3, -6; 6, -3]

theorem matrix_power_is_correct : A ^ 3 = A_cubed := by 
  sorry

end NUMINAMATH_GPT_matrix_power_is_correct_l682_68217


namespace NUMINAMATH_GPT_find_dividend_l682_68262

theorem find_dividend (divisor quotient remainder : ℕ) (h_divisor : divisor = 38) (h_quotient : quotient = 19) (h_remainder : remainder = 7) :
  divisor * quotient + remainder = 729 := by
  sorry

end NUMINAMATH_GPT_find_dividend_l682_68262


namespace NUMINAMATH_GPT_calculation_proof_l682_68248

theorem calculation_proof
    (a : ℝ) (b : ℝ) (c : ℝ)
    (h1 : a = 3.6)
    (h2 : b = 0.25)
    (h3 : c = 0.5) :
    (a * b) / c = 1.8 := 
by
  sorry

end NUMINAMATH_GPT_calculation_proof_l682_68248


namespace NUMINAMATH_GPT_annual_fixed_costs_l682_68292

theorem annual_fixed_costs
  (profit : ℝ := 30500000)
  (selling_price : ℝ := 9035)
  (variable_cost : ℝ := 5000)
  (units_sold : ℕ := 20000) :
  ∃ (fixed_costs : ℝ), profit = (selling_price * units_sold) - (variable_cost * units_sold) - fixed_costs :=
sorry

end NUMINAMATH_GPT_annual_fixed_costs_l682_68292


namespace NUMINAMATH_GPT_find_number_l682_68227

theorem find_number (x : ℕ) (h : x * 48 = 173 * 240) : x = 865 :=
sorry

end NUMINAMATH_GPT_find_number_l682_68227


namespace NUMINAMATH_GPT_find_x_l682_68215

variable (a b x : ℝ)

def condition1 : Prop := a / b = 5 / 4
def condition2 : Prop := (4 * a + x * b) / (4 * a - x * b) = 4

theorem find_x (h1 : condition1 a b) (h2 : condition2 a b x) : x = 3 :=
  sorry

end NUMINAMATH_GPT_find_x_l682_68215


namespace NUMINAMATH_GPT_fraction_reach_impossible_l682_68276

theorem fraction_reach_impossible :
  ¬ ∃ (a b : ℕ), (2 + 2013 * a) / (3 + 2014 * b) = 3 / 5 := by
  sorry

end NUMINAMATH_GPT_fraction_reach_impossible_l682_68276


namespace NUMINAMATH_GPT_g_84_value_l682_68246

-- Define the function g with the given conditions
def g (x : ℝ) : ℝ := sorry

-- Conditions given in the problem
axiom g_property1 : ∀ x y : ℝ, g (x * y) = y * g x
axiom g_property2 : g 2 = 48

-- Statement to prove
theorem g_84_value : g 84 = 2016 :=
by
  sorry

end NUMINAMATH_GPT_g_84_value_l682_68246


namespace NUMINAMATH_GPT_find_angle_l682_68269

theorem find_angle (θ : ℝ) (h : 180 - θ = 3 * (90 - θ)) : θ = 45 :=
by
  sorry

end NUMINAMATH_GPT_find_angle_l682_68269


namespace NUMINAMATH_GPT_probability_of_winning_noughts_l682_68270

def ticTacToeProbability : ℚ :=
  let totalWays := Nat.choose 9 3
  let winningWays := 8
  winningWays / totalWays

theorem probability_of_winning_noughts :
  ticTacToeProbability = 2 / 21 := by
  sorry

end NUMINAMATH_GPT_probability_of_winning_noughts_l682_68270


namespace NUMINAMATH_GPT_student_council_profit_l682_68203

def boxes : ℕ := 48
def erasers_per_box : ℕ := 24
def price_per_eraser : ℝ := 0.75

theorem student_council_profit :
  boxes * erasers_per_box * price_per_eraser = 864 := 
by
  sorry

end NUMINAMATH_GPT_student_council_profit_l682_68203


namespace NUMINAMATH_GPT_foreign_students_next_sem_eq_740_l682_68278

def total_students : ℕ := 1800
def percentage_foreign : ℕ := 30
def new_foreign_students : ℕ := 200

def initial_foreign_students : ℕ := total_students * percentage_foreign / 100
def total_foreign_students_next_semester : ℕ :=
  initial_foreign_students + new_foreign_students

theorem foreign_students_next_sem_eq_740 :
  total_foreign_students_next_semester = 740 :=
by
  sorry

end NUMINAMATH_GPT_foreign_students_next_sem_eq_740_l682_68278


namespace NUMINAMATH_GPT_angle_measure_l682_68286

theorem angle_measure (x y : ℝ) (h1 : x + y = 90) (h2 : x = 3 * y) : x = 67.5 :=
by
  -- sorry is used here to indicate that the proof is omitted
  sorry

end NUMINAMATH_GPT_angle_measure_l682_68286


namespace NUMINAMATH_GPT_n_plus_d_is_155_l682_68266

noncomputable def n_and_d_sum : Nat :=
sorry

theorem n_plus_d_is_155 (n d : Nat) (hn : 0 < n) (hd : d < 10) 
  (h1 : 4 * n^2 + 2 * n + d = 305) 
  (h2 : 4 * n^3 + 2 * n^2 + d * n + 1 = 577 + 8 * d) : n + d = 155 := 
sorry

end NUMINAMATH_GPT_n_plus_d_is_155_l682_68266


namespace NUMINAMATH_GPT_smallest_x_for_palindrome_l682_68235

-- Define the condition for a number to be a palindrome
def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

-- Mathematically equivalent proof problem statement
theorem smallest_x_for_palindrome : ∃ (x : ℕ), x > 0 ∧ is_palindrome (x + 2345) ∧ x = 97 :=
by sorry

end NUMINAMATH_GPT_smallest_x_for_palindrome_l682_68235


namespace NUMINAMATH_GPT_find_length_of_AB_l682_68218

theorem find_length_of_AB (x y : ℝ) (AP PB AQ QB PQ AB : ℝ) 
  (h1 : AP = 3 * x) 
  (h2 : PB = 4 * x) 
  (h3 : AQ = 4 * y) 
  (h4 : QB = 5 * y)
  (h5 : PQ = 5) 
  (h6 : AP + PB = AB)
  (h7 : AQ + QB = AB)
  (h8 : PQ = AQ - AP)
  (h9 : 7 * x = 9 * y) : 
  AB = 315 := 
by
  sorry

end NUMINAMATH_GPT_find_length_of_AB_l682_68218


namespace NUMINAMATH_GPT_intersection_A_B_l682_68245

def A : Set ℤ := {-2, -1, 1, 2}

def B : Set ℤ := {x | x^2 - x - 2 ≥ 0}

theorem intersection_A_B : (A ∩ B) = {-2, -1, 2} := by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l682_68245


namespace NUMINAMATH_GPT_remainder_17_plus_x_mod_31_l682_68226

theorem remainder_17_plus_x_mod_31 {x : ℕ} (h : 13 * x ≡ 3 [MOD 31]) : (17 + x) % 31 = 22 := 
sorry

end NUMINAMATH_GPT_remainder_17_plus_x_mod_31_l682_68226


namespace NUMINAMATH_GPT_graph_is_empty_l682_68247

/-- The given equation 3x² + 4y² - 12x - 16y + 36 = 0 has no real solutions. -/
theorem graph_is_empty : ∀ (x y : ℝ), 3 * x^2 + 4 * y^2 - 12 * x - 16 * y + 36 ≠ 0 :=
by
  intro x y
  sorry

end NUMINAMATH_GPT_graph_is_empty_l682_68247


namespace NUMINAMATH_GPT_S_ploughing_time_l682_68242

theorem S_ploughing_time (R S : ℝ) (hR_rate : R = 1 / 15) (h_combined_rate : R + S = 1 / 10) : S = 1 / 30 := sorry

end NUMINAMATH_GPT_S_ploughing_time_l682_68242


namespace NUMINAMATH_GPT_domain_f_2x_minus_1_l682_68294

theorem domain_f_2x_minus_1 (f : ℝ → ℝ) :
  (∀ x, -2 ≤ x ∧ x ≤ 3 → ∃ y, f (x + 1) = y) →
  (∀ x, 0 ≤ x ∧ x ≤ 5 / 2 → ∃ y, f (2 * x - 1) = y) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_domain_f_2x_minus_1_l682_68294


namespace NUMINAMATH_GPT_square_division_l682_68280

theorem square_division (n : ℕ) (h : n ≥ 6) :
  ∃ (sq_div : ℕ → Prop), sq_div 6 ∧ (∀ n, sq_div n → sq_div (n + 3)) :=
by
  sorry

end NUMINAMATH_GPT_square_division_l682_68280


namespace NUMINAMATH_GPT_cube_painting_l682_68287

-- Let's start with importing Mathlib for natural number operations

theorem cube_painting (n : ℕ) (h : 2 < n)
  (num_one_black_face : ℕ := 3 * (n - 2)^2)
  (num_unpainted : ℕ := (n - 2)^3) :
  num_one_black_face = num_unpainted → n = 5 :=
by
  sorry

end NUMINAMATH_GPT_cube_painting_l682_68287


namespace NUMINAMATH_GPT_bike_ride_distance_l682_68231

theorem bike_ride_distance (D : ℝ) (h : D / 10 = D / 15 + 0.5) : D = 15 :=
  sorry

end NUMINAMATH_GPT_bike_ride_distance_l682_68231


namespace NUMINAMATH_GPT_Andy_is_late_l682_68288

def school_start_time : Nat := 8 * 60 -- in minutes (8:00 AM)
def normal_travel_time : Nat := 30 -- in minutes
def delay_red_lights : Nat := 4 * 3 -- in minutes (4 red lights * 3 minutes each)
def delay_construction : Nat := 10 -- in minutes
def delay_detour_accident : Nat := 7 -- in minutes
def delay_store_stop : Nat := 5 -- in minutes
def delay_searching_store : Nat := 2 -- in minutes
def delay_traffic : Nat := 15 -- in minutes
def delay_neighbor_help : Nat := 6 -- in minutes
def delay_closed_road : Nat := 8 -- in minutes
def all_delays : Nat := delay_red_lights + delay_construction + delay_detour_accident + delay_store_stop + delay_searching_store + delay_traffic + delay_neighbor_help + delay_closed_road
def departure_time : Nat := 7 * 60 + 15 -- in minutes (7:15 AM)

def arrival_time : Nat := departure_time + normal_travel_time + all_delays
def late_minutes : Nat := arrival_time - school_start_time

theorem Andy_is_late : late_minutes = 50 := by
  sorry

end NUMINAMATH_GPT_Andy_is_late_l682_68288


namespace NUMINAMATH_GPT_age_difference_is_24_l682_68204

theorem age_difference_is_24 (d f : ℕ) (h1 : d = f / 9) (h2 : f + 1 = 7 * (d + 1)) : f - d = 24 := sorry

end NUMINAMATH_GPT_age_difference_is_24_l682_68204


namespace NUMINAMATH_GPT_ratio_of_solving_linear_equations_to_algebra_problems_l682_68206

theorem ratio_of_solving_linear_equations_to_algebra_problems:
  let total_problems := 140
  let algebra_percentage := 0.40
  let solving_linear_equations := 28
  let total_algebra_problems := algebra_percentage * total_problems
  let ratio := solving_linear_equations / total_algebra_problems
  ratio = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_ratio_of_solving_linear_equations_to_algebra_problems_l682_68206


namespace NUMINAMATH_GPT_area_of_absolute_value_sum_l682_68222

theorem area_of_absolute_value_sum :
  ∃ area : ℝ, (area = 80) ∧ (∀ x y : ℝ, |2 * x| + |5 * y| = 20 → area = 80) :=
by
  sorry

end NUMINAMATH_GPT_area_of_absolute_value_sum_l682_68222


namespace NUMINAMATH_GPT_problem_l682_68253

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
if h : (-1 : ℝ) ≤ x ∧ x < 0 then a*x + 1
else if h : (0 : ℝ) ≤ x ∧ x ≤ 1 then (b*x + 2) / (x + 1)
else 0 -- This should not matter as we only care about the given ranges

theorem problem (a b : ℝ) (h₁ : f 0.5 a b = f 1.5 a b) : a + 3 * b = -10 :=
by
  -- We'll derive equations from given conditions and prove the result.
  sorry

end NUMINAMATH_GPT_problem_l682_68253


namespace NUMINAMATH_GPT_bobby_pancakes_left_l682_68225

theorem bobby_pancakes_left (initial_pancakes : ℕ) (bobby_ate : ℕ) (dog_ate : ℕ) :
  initial_pancakes = 21 → bobby_ate = 5 → dog_ate = 7 → initial_pancakes - (bobby_ate + dog_ate) = 9 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_bobby_pancakes_left_l682_68225


namespace NUMINAMATH_GPT_weight_of_B_l682_68202

theorem weight_of_B (A B C : ℝ)
  (h1 : (A + B + C) / 3 = 45)
  (h2 : (A + B) / 2 = 40)
  (h3 : (B + C) / 2 = 43) :
  B = 31 :=
by
  sorry

end NUMINAMATH_GPT_weight_of_B_l682_68202


namespace NUMINAMATH_GPT_diophantine_infinite_solutions_l682_68271

theorem diophantine_infinite_solutions :
  ∃ (a b c x y : ℤ), (a + b + c = x + y) ∧ (a^3 + b^3 + c^3 = x^3 + y^3) ∧ 
  ∃ (d : ℤ), (a = b - d) ∧ (c = b + d) :=
sorry

end NUMINAMATH_GPT_diophantine_infinite_solutions_l682_68271


namespace NUMINAMATH_GPT_distribute_positions_l682_68290

structure DistributionProblem :=
  (volunteer_positions : ℕ)
  (schools : ℕ)
  (min_positions : ℕ)
  (distinct_allocations : ∀ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c)

noncomputable def count_ways (p : DistributionProblem) : ℕ :=
  if p.volunteer_positions = 7 ∧ p.schools = 3 ∧ p.min_positions = 1 then 6 else 0

theorem distribute_positions (p : DistributionProblem) :
  count_ways p = 6 :=
by
  sorry

end NUMINAMATH_GPT_distribute_positions_l682_68290


namespace NUMINAMATH_GPT_circular_sequence_zero_if_equidistant_l682_68232

noncomputable def circular_sequence_property (x y z : ℤ): Prop :=
  (x = 0 ∧ y = 0 ∧ dist x y = dist y z) → z = 0

theorem circular_sequence_zero_if_equidistant {x y z : ℤ} :
  (x = 0 ∧ y = 0 ∧ dist x y = dist y z) → z = 0 :=
by sorry

end NUMINAMATH_GPT_circular_sequence_zero_if_equidistant_l682_68232


namespace NUMINAMATH_GPT_fernanda_total_time_eq_90_days_l682_68214

-- Define the conditions
def num_audiobooks : ℕ := 6
def hours_per_audiobook : ℕ := 30
def hours_listened_per_day : ℕ := 2

-- Define the total time calculation
def total_time_to_finish_audiobooks (a h r : ℕ) : ℕ :=
  (h / r) * a

-- The assertion we need to prove
theorem fernanda_total_time_eq_90_days :
  total_time_to_finish_audiobooks num_audiobooks hours_per_audiobook hours_listened_per_day = 90 :=
by sorry

end NUMINAMATH_GPT_fernanda_total_time_eq_90_days_l682_68214


namespace NUMINAMATH_GPT_journey_time_proof_l682_68285

noncomputable def journey_time_on_wednesday (d s x : ℝ) : ℝ :=
  d / s

theorem journey_time_proof (d s x : ℝ) (usual_speed_nonzero : s ≠ 0) :
  (journey_time_on_wednesday d s x) = 11 * x :=
by
  have thursday_speed : ℝ := 1.1 * s
  have thursday_time : ℝ := d / thursday_speed
  have time_diff : ℝ := (d / s) - thursday_time
  have reduced_time_eq_x : time_diff = x := by sorry
  have journey_time_eq : (d / s) = 11 * x := by sorry
  exact journey_time_eq

end NUMINAMATH_GPT_journey_time_proof_l682_68285


namespace NUMINAMATH_GPT_todd_initial_money_l682_68228

-- Definitions of the conditions
def cost_per_candy_bar : ℕ := 2
def number_of_candy_bars : ℕ := 4
def money_left : ℕ := 12
def total_money_spent := number_of_candy_bars * cost_per_candy_bar

-- The statement proving the initial amount of money Todd had
theorem todd_initial_money : 
  (total_money_spent + money_left) = 20 :=
by
  sorry

end NUMINAMATH_GPT_todd_initial_money_l682_68228


namespace NUMINAMATH_GPT_directrix_of_parabola_l682_68267

theorem directrix_of_parabola (a : ℝ) (h : a = -4) : ∃ k : ℝ, k = 1/16 ∧ ∀ x : ℕ, y = ax ^ 2 → y = k := 
by 
  sorry

end NUMINAMATH_GPT_directrix_of_parabola_l682_68267


namespace NUMINAMATH_GPT_prime_719_exists_l682_68265

theorem prime_719_exists (a b c : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c) :
  (a^4 + b^4 + c^4 - 3 = 719) → Nat.Prime (a^4 + b^4 + c^4 - 3) := sorry

end NUMINAMATH_GPT_prime_719_exists_l682_68265


namespace NUMINAMATH_GPT_toad_difference_l682_68293

variables (Tim_toads Jim_toads Sarah_toads : ℕ)

theorem toad_difference (h1 : Tim_toads = 30) 
                        (h2 : Jim_toads > Tim_toads) 
                        (h3 : Sarah_toads = 2 * Jim_toads) 
                        (h4 : Sarah_toads = 100) :
  Jim_toads - Tim_toads = 20 :=
by
  -- The next lines are placeholders for the logical steps which need to be proven
  sorry

end NUMINAMATH_GPT_toad_difference_l682_68293


namespace NUMINAMATH_GPT_initial_inventory_correct_l682_68261

-- Define the conditions as given in the problem
def bottles_sold_monday : ℕ := 2445
def bottles_sold_tuesday : ℕ := 900
def bottles_sold_per_day_wed_to_sun : ℕ := 50
def days_wed_to_sun : ℕ := 5
def bottles_delivered_saturday : ℕ := 650
def final_inventory : ℕ := 1555

-- Define the total number of bottles sold during the week
def total_bottles_sold : ℕ :=
  bottles_sold_monday + bottles_sold_tuesday + (bottles_sold_per_day_wed_to_sun * days_wed_to_sun)

-- Define the initial inventory calculation
def initial_inventory : ℕ :=
  final_inventory + total_bottles_sold - bottles_delivered_saturday

-- The theorem we want to prove
theorem initial_inventory_correct :
  initial_inventory = 4500 :=
by
  sorry

end NUMINAMATH_GPT_initial_inventory_correct_l682_68261


namespace NUMINAMATH_GPT_donation_to_treetown_and_forest_reserve_l682_68256

noncomputable def donation_problem (x : ℕ) :=
  x + (x + 140) = 1000

theorem donation_to_treetown_and_forest_reserve :
  ∃ x : ℕ, donation_problem x ∧ (x + 140 = 570) := 
by
  sorry

end NUMINAMATH_GPT_donation_to_treetown_and_forest_reserve_l682_68256


namespace NUMINAMATH_GPT_cost_per_dvd_l682_68236

theorem cost_per_dvd (total_cost : ℝ) (num_dvds : ℕ) 
  (h1 : total_cost = 4.8) (h2 : num_dvds = 4) : (total_cost / num_dvds) = 1.2 :=
by
  sorry

end NUMINAMATH_GPT_cost_per_dvd_l682_68236


namespace NUMINAMATH_GPT_cannot_form_triangle_l682_68249

theorem cannot_form_triangle (a b c : ℕ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) : 
  ¬ ∃ a b c : ℕ, (a, b, c) = (1, 2, 3) := 
  sorry

end NUMINAMATH_GPT_cannot_form_triangle_l682_68249


namespace NUMINAMATH_GPT_fraction_defined_range_l682_68255

theorem fraction_defined_range (x : ℝ) : 
  (∃ y, y = 3 / (x - 2)) ↔ x ≠ 2 :=
by
  sorry

end NUMINAMATH_GPT_fraction_defined_range_l682_68255


namespace NUMINAMATH_GPT_max_marks_l682_68291

theorem max_marks (M : ℝ) (h1 : 0.33 * M = 59 + 40) : M = 300 :=
by
  sorry

end NUMINAMATH_GPT_max_marks_l682_68291


namespace NUMINAMATH_GPT_hotel_flat_fee_l682_68233

theorem hotel_flat_fee (f n : ℝ) (h1 : f + n = 120) (h2 : f + 6 * n = 330) : f = 78 :=
by
  sorry

end NUMINAMATH_GPT_hotel_flat_fee_l682_68233


namespace NUMINAMATH_GPT_distance_traveled_by_car_l682_68254

theorem distance_traveled_by_car (total_distance : ℕ) (fraction_foot : ℚ) (fraction_bus : ℚ)
  (h_total : total_distance = 40) (h_fraction_foot : fraction_foot = 1/4)
  (h_fraction_bus : fraction_bus = 1/2) :
  (total_distance * (1 - fraction_foot - fraction_bus)) = 10 :=
by
  sorry

end NUMINAMATH_GPT_distance_traveled_by_car_l682_68254


namespace NUMINAMATH_GPT_selected_numbers_in_range_l682_68296

noncomputable def systematic_sampling (n_students selected_students interval_num start_num n : ℕ) : ℕ :=
  start_num + interval_num * (n - 1)

theorem selected_numbers_in_range (x : ℕ) :
  (500 = 500) ∧ (50 = 50) ∧ (10 = 500 / 50) ∧ (6 ∈ {y : ℕ | 1 ≤ y ∧ y ≤ 10}) ∧ (125 ≤ x ∧ x ≤ 140) → 
  (x = systematic_sampling 500 50 10 6 13 ∨ x = systematic_sampling 500 50 10 6 14) :=
by
  sorry

end NUMINAMATH_GPT_selected_numbers_in_range_l682_68296


namespace NUMINAMATH_GPT_eden_stuffed_bears_l682_68281

theorem eden_stuffed_bears
  (initial_bears : ℕ)
  (favorite_bears : ℕ)
  (sisters : ℕ)
  (eden_initial_bears : ℕ)
  (remaining_bears := initial_bears - favorite_bears)
  (bears_per_sister := remaining_bears / sisters)
  (eden_bears_now := eden_initial_bears + bears_per_sister)
  (h1 : initial_bears = 20)
  (h2 : favorite_bears = 8)
  (h3 : sisters = 3)
  (h4 : eden_initial_bears = 10) :
  eden_bears_now = 14 := by
{
  sorry
}

end NUMINAMATH_GPT_eden_stuffed_bears_l682_68281


namespace NUMINAMATH_GPT_price_returns_to_initial_l682_68289

theorem price_returns_to_initial (x : ℝ) (h : 0.918 * (100 + x) = 100) : x = 9 := 
by
  sorry

end NUMINAMATH_GPT_price_returns_to_initial_l682_68289


namespace NUMINAMATH_GPT_a1_a9_sum_l682_68237

noncomputable def arithmetic_sequence (a: ℕ → ℝ) : Prop :=
∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem a1_a9_sum
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a3_a7_roots : (a 3 = 3 ∧ a 7 = -1) ∨ (a 3 = -1 ∧ a 7 = 3)) :
  a 1 + a 9 = 2 :=
by
  sorry

end NUMINAMATH_GPT_a1_a9_sum_l682_68237


namespace NUMINAMATH_GPT_logarithmic_inequality_and_integral_l682_68274

theorem logarithmic_inequality_and_integral :
  let a := Real.log 3 / Real.log 2
  let b := Real.log 2 / Real.log 3
  let c := 2 / Real.pi^2
  a > b ∧ b > c :=
by
  let a := Real.log 3 / Real.log 2
  let b := Real.log 2 / Real.log 3
  let c := 2 / Real.pi^2
  sorry

end NUMINAMATH_GPT_logarithmic_inequality_and_integral_l682_68274


namespace NUMINAMATH_GPT_pet_center_final_count_l682_68283

def initial_dogs : Nat := 36
def initial_cats : Nat := 29
def adopted_dogs : Nat := 20
def collected_cats : Nat := 12
def final_pets : Nat := 57

theorem pet_center_final_count :
  (initial_dogs - adopted_dogs) + (initial_cats + collected_cats) = final_pets := 
by
  sorry

end NUMINAMATH_GPT_pet_center_final_count_l682_68283


namespace NUMINAMATH_GPT_john_paid_correct_amount_l682_68259

def cost_bw : ℝ := 160
def markup_percentage : ℝ := 0.5

def cost_color : ℝ := cost_bw * (1 + markup_percentage)

theorem john_paid_correct_amount : 
  cost_color = 240 := 
by
  -- proof required here
  sorry

end NUMINAMATH_GPT_john_paid_correct_amount_l682_68259


namespace NUMINAMATH_GPT_divisor_of_p_l682_68284

theorem divisor_of_p (p q r s : ℕ) (h₁ : Nat.gcd p q = 30) (h₂ : Nat.gcd q r = 45) (h₃ : Nat.gcd r s = 75) (h₄ : 120 < Nat.gcd s p) (h₅ : Nat.gcd s p < 180) : 5 ∣ p := 
sorry

end NUMINAMATH_GPT_divisor_of_p_l682_68284


namespace NUMINAMATH_GPT_exists_circle_with_exactly_n_integer_points_l682_68212

noncomputable def circle_with_n_integer_points (n : ℕ) : Prop :=
  ∃ r : ℤ, ∃ (xs ys : List ℤ), 
    xs.length = n ∧ ys.length = n ∧
    ∀ (x y : ℤ), x ∈ xs → y ∈ ys → x^2 + y^2 = r^2

theorem exists_circle_with_exactly_n_integer_points (n : ℕ) : 
  circle_with_n_integer_points n := 
sorry

end NUMINAMATH_GPT_exists_circle_with_exactly_n_integer_points_l682_68212


namespace NUMINAMATH_GPT_compare_magnitudes_l682_68211

theorem compare_magnitudes (a b c d e : ℝ) (h₁ : a > b) (h₂ : b > 0) (h₃ : c < d) (h₄ : d < 0) (h₅ : e < 0) :
  (e / (a - c)) > (e / (b - d)) :=
  sorry

end NUMINAMATH_GPT_compare_magnitudes_l682_68211


namespace NUMINAMATH_GPT_problem_statement_l682_68210

noncomputable def f : ℝ → ℝ := sorry

variable (α : ℝ)

axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom periodic_function : ∀ x : ℝ, f (x + 3) = -f x
axiom tan_alpha : Real.tan α = 2

theorem problem_statement : f (15 * Real.sin α * Real.cos α) = 0 := 
by {
  sorry
}

end NUMINAMATH_GPT_problem_statement_l682_68210


namespace NUMINAMATH_GPT_rectangle_distances_sum_l682_68205

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem rectangle_distances_sum :
  let A : (ℝ × ℝ) := (0, 0)
  let B : (ℝ × ℝ) := (3, 0)
  let C : (ℝ × ℝ) := (3, 4)
  let D : (ℝ × ℝ) := (0, 4)

  let M : (ℝ × ℝ) := ((B.1 + A.1) / 2, (B.2 + A.2) / 2)
  let N : (ℝ × ℝ) := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let O : (ℝ × ℝ) := ((C.1 + D.1) / 2, (C.2 + D.2) / 2)
  let P : (ℝ × ℝ) := ((D.1 + A.1) / 2, (D.2 + A.2) / 2)

  distance A.1 A.2 M.1 M.2 + distance A.1 A.2 N.1 N.2 + distance A.1 A.2 O.1 O.2 + distance A.1 A.2 P.1 P.2 = 7.77 + Real.sqrt 13 :=
sorry

end NUMINAMATH_GPT_rectangle_distances_sum_l682_68205


namespace NUMINAMATH_GPT_output_correct_l682_68208

-- Define the initial values and assignments
def initial_a : ℕ := 1
def initial_b : ℕ := 2
def initial_c : ℕ := 3

-- Perform the assignments in sequence
def after_c_assignment : ℕ := initial_b
def after_b_assignment : ℕ := initial_a
def after_a_assignment : ℕ := after_c_assignment

-- Final values after all assignments
def final_a := after_a_assignment
def final_b := after_b_assignment
def final_c := after_c_assignment

-- Theorem statement
theorem output_correct :
  final_a = 2 ∧ final_b = 1 ∧ final_c = 2 :=
by {
  -- Proof is omitted
  sorry
}

end NUMINAMATH_GPT_output_correct_l682_68208


namespace NUMINAMATH_GPT_ratio_of_money_earned_l682_68224

variable (L T J : ℕ) 

theorem ratio_of_money_earned 
  (total_earned : L + T + J = 60)
  (lisa_earning : L = 30)
  (lisa_tommy_diff : L = T + 15) : 
  T / L = 1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_money_earned_l682_68224


namespace NUMINAMATH_GPT_soccer_season_length_l682_68251

def total_games : ℕ := 27
def games_per_month : ℕ := 9
def months_in_season : ℕ := total_games / games_per_month

theorem soccer_season_length : months_in_season = 3 := by
  unfold months_in_season
  unfold total_games
  unfold games_per_month
  sorry

end NUMINAMATH_GPT_soccer_season_length_l682_68251


namespace NUMINAMATH_GPT_jill_vs_jack_arrival_time_l682_68216

def distance_to_park : ℝ := 1.2
def jill_speed : ℝ := 8
def jack_speed : ℝ := 5

theorem jill_vs_jack_arrival_time :
  let jill_time := distance_to_park / jill_speed
  let jack_time := distance_to_park / jack_speed
  let jill_time_minutes := jill_time * 60
  let jack_time_minutes := jack_time * 60
  jill_time_minutes < jack_time_minutes ∧ jack_time_minutes - jill_time_minutes = 5.4 :=
by
  sorry

end NUMINAMATH_GPT_jill_vs_jack_arrival_time_l682_68216


namespace NUMINAMATH_GPT_max_distance_between_circle_and_ellipse_l682_68250

noncomputable def max_distance_PQ : ℝ :=
  1 + (3 * Real.sqrt 6) / 2

theorem max_distance_between_circle_and_ellipse :
  ∀ (P Q : ℝ × ℝ), (P.1^2 + (P.2 - 2)^2 = 1) → 
                   (Q.1^2 / 9 + Q.2^2 = 1) →
                   dist P Q ≤ max_distance_PQ :=
by
  intros P Q hP hQ
  sorry

end NUMINAMATH_GPT_max_distance_between_circle_and_ellipse_l682_68250


namespace NUMINAMATH_GPT_intersection_point_l682_68221

theorem intersection_point (a b c d : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ d) :
  let x := (d - c) / (2 * b)
  let y := (a * (d - c)^2) / (4 * b^2) + (d + c) / 2
  (ax^2 + bx + c = y) ∧ (ax^2 - bx + d = y) :=
by
  sorry

end NUMINAMATH_GPT_intersection_point_l682_68221


namespace NUMINAMATH_GPT_seq_a_seq_b_l682_68239

theorem seq_a (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) :
  a 1 = 1 ∧ (∀ n, S (n + 1) = 3 * S n + 2) →
  (∀ n, a n = if n = 1 then 1 else 4 * 3 ^ (n - 2)) :=
by
  sorry

theorem seq_b (b : ℕ → ℕ) (a : ℕ → ℕ) (T : ℕ → ℕ) (n : ℕ) :
  (b n = 8 * n / (a (n + 1) - a n)) →
  (T n = 77 / 12 - (n / 2 + 3 / 4) * (1 / 3) ^ (n - 2)) :=
by
  sorry

end NUMINAMATH_GPT_seq_a_seq_b_l682_68239


namespace NUMINAMATH_GPT_combined_cost_price_l682_68234

theorem combined_cost_price :
  let stock1_price := 100
  let stock1_discount := 5 / 100
  let stock1_brokerage := 1.5 / 100
  let stock2_price := 200
  let stock2_discount := 7 / 100
  let stock2_brokerage := 0.75 / 100
  let stock3_price := 300
  let stock3_discount := 3 / 100
  let stock3_brokerage := 1 / 100

  -- Calculated values
  let stock1_discounted_price := stock1_price * (1 - stock1_discount)
  let stock1_total_price := stock1_discounted_price * (1 + stock1_brokerage)
  
  let stock2_discounted_price := stock2_price * (1 - stock2_discount)
  let stock2_total_price := stock2_discounted_price * (1 + stock2_brokerage)
  
  let stock3_discounted_price := stock3_price * (1 - stock3_discount)
  let stock3_total_price := stock3_discounted_price * (1 + stock3_brokerage)
  
  let combined_cost := stock1_total_price + stock2_total_price + stock3_total_price
  combined_cost = 577.73 := sorry

end NUMINAMATH_GPT_combined_cost_price_l682_68234


namespace NUMINAMATH_GPT_hyperbola_ellipse_b_value_l682_68258

theorem hyperbola_ellipse_b_value (a c b : ℝ) (h1 : c = 5 * a / 4) (h2 : c^2 - a^2 = (9 * a^2) / 16) (h3 : 4 * (b^2 - 4) = 16 * b^2 / 25) :
  b = 6 / 5 ∨ b = 10 / 3 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_ellipse_b_value_l682_68258


namespace NUMINAMATH_GPT_candy_groups_l682_68241

theorem candy_groups (total_candies group_size : Nat) (h1 : total_candies = 30) (h2 : group_size = 3) : total_candies / group_size = 10 := by
  sorry

end NUMINAMATH_GPT_candy_groups_l682_68241


namespace NUMINAMATH_GPT_expand_expression_l682_68229

theorem expand_expression (x : ℝ) : (x - 3) * (x + 6) = x^2 + 3 * x - 18 :=
by
  sorry

end NUMINAMATH_GPT_expand_expression_l682_68229


namespace NUMINAMATH_GPT_lesser_solution_of_quadratic_eq_l682_68220

theorem lesser_solution_of_quadratic_eq : ∃ x ∈ {x | x^2 + 10*x - 24 = 0}, x = -12 :=
by 
  sorry

end NUMINAMATH_GPT_lesser_solution_of_quadratic_eq_l682_68220


namespace NUMINAMATH_GPT_smallest_nat_div3_and_5_rem1_l682_68201

theorem smallest_nat_div3_and_5_rem1 : ∃ N : ℕ, N > 1 ∧ (N % 3 = 1) ∧ (N % 5 = 1) ∧ ∀ M : ℕ, M > 1 ∧ (M % 3 = 1) ∧ (M % 5 = 1) → N ≤ M := 
by
  sorry

end NUMINAMATH_GPT_smallest_nat_div3_and_5_rem1_l682_68201


namespace NUMINAMATH_GPT_simplify_expression_l682_68219

-- Define the initial expression
def expr (q : ℚ) := (5 * q^4 - 4 * q^3 + 7 * q - 8) + (3 - 5 * q^2 + q^3 - 2 * q)

-- Define the simplified expression
def simplified_expr (q : ℚ) := 5 * q^4 - 3 * q^3 - 5 * q^2 + 5 * q - 5

-- The theorem stating that the two expressions are equal
theorem simplify_expression (q : ℚ) : expr q = simplified_expr q :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l682_68219


namespace NUMINAMATH_GPT_concert_attendance_difference_l682_68279

noncomputable def first_concert : ℕ := 65899
noncomputable def second_concert : ℕ := 66018

theorem concert_attendance_difference :
  (second_concert - first_concert) = 119 :=
by
  sorry

end NUMINAMATH_GPT_concert_attendance_difference_l682_68279


namespace NUMINAMATH_GPT_ellipse_area_l682_68282

def ellipse_equation (x y : ℝ) : Prop :=
  4 * x^2 - 8 * x + 9 * y^2 - 36 * y + 36 = 0

theorem ellipse_area :
  (∀ x y : ℝ, ellipse_equation x y → true) →
  (π * 1 * (4/3) = 4 * π / 3) :=
by
  intro h
  norm_num
  sorry

end NUMINAMATH_GPT_ellipse_area_l682_68282


namespace NUMINAMATH_GPT_central_angle_of_sector_l682_68298

theorem central_angle_of_sector (r l : ℝ) (h1 : r = 1) (h2 : l = 4 - 2*r) : 
    ∃ α : ℝ, α = 2 :=
by
  use l / r
  have hr : r = 1 := h1
  have hl : l = 4 - 2*r := h2
  sorry

end NUMINAMATH_GPT_central_angle_of_sector_l682_68298


namespace NUMINAMATH_GPT_senior_citizen_ticket_cost_l682_68240

theorem senior_citizen_ticket_cost 
  (total_tickets : ℕ)
  (regular_ticket_cost : ℕ)
  (total_sales : ℕ)
  (sold_regular_tickets : ℕ)
  (x : ℕ)
  (h1 : total_tickets = 65)
  (h2 : regular_ticket_cost = 15)
  (h3 : total_sales = 855)
  (h4 : sold_regular_tickets = 41)
  (h5 : total_sales = (sold_regular_tickets * regular_ticket_cost) + ((total_tickets - sold_regular_tickets) * x)) :
  x = 10 :=
by
  sorry

end NUMINAMATH_GPT_senior_citizen_ticket_cost_l682_68240


namespace NUMINAMATH_GPT_man_l682_68244

-- Given conditions
def V_m := 15 - 3.2
def V_c := 3.2
def man's_speed_with_current : Real := 15

-- Required to prove
def man's_speed_against_current := V_m - V_c

theorem man's_speed_against_current_is_correct : man's_speed_against_current = 8.6 := by
  sorry

end NUMINAMATH_GPT_man_l682_68244
