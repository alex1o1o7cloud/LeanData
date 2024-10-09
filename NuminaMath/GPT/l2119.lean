import Mathlib

namespace solve_for_w_squared_l2119_211990

-- Define the original equation
def eqn (w : ℝ) := 2 * (w + 15)^2 = (4 * w + 9) * (3 * w + 6)

-- Define the goal to prove w^2 = 6.7585 based on the given equation
theorem solve_for_w_squared : ∃ w : ℝ, eqn w ∧ w^2 = 6.7585 :=
by
  sorry

end solve_for_w_squared_l2119_211990


namespace twelve_year_olds_count_l2119_211988

theorem twelve_year_olds_count (x y z w : ℕ) 
  (h1 : x + y + z + w = 23)
  (h2 : 10 * x + 11 * y + 12 * z + 13 * w = 253)
  (h3 : z = 3 * w / 2) : 
  z = 6 :=
by sorry

end twelve_year_olds_count_l2119_211988


namespace pipe_cut_l2119_211932

theorem pipe_cut (x : ℝ) (h1 : x + 2 * x = 177) : 2 * x = 118 :=
by
  sorry

end pipe_cut_l2119_211932


namespace intersection_complement_A_l2119_211979

def A : Set ℝ := {x | abs (x - 1) < 1}

def B : Set ℝ := {x | x < 1}

def CRB : Set ℝ := {x | x ≥ 1}

theorem intersection_complement_A :
  (CRB ∩ A) = {x | 1 ≤ x ∧ x < 2} :=
by
  sorry

end intersection_complement_A_l2119_211979


namespace value_of_m_div_x_l2119_211987

variables (a b : ℝ) (k : ℝ)
-- Condition: The ratio of a to b is 4 to 5
def ratio_a_to_b : Prop := a / b = 4 / 5

-- Condition: x equals a increased by 75 percent of a
def x := a + 0.75 * a

-- Condition: m equals b decreased by 80 percent of b
def m := b - 0.80 * b

-- Prove the given question
theorem value_of_m_div_x (h1 : ratio_a_to_b a b) (ha_pos : 0 < a) (hb_pos : 0 < b) :
  m / x = 1 / 7 := by
sorry

end value_of_m_div_x_l2119_211987


namespace base6_add_sub_l2119_211913

theorem base6_add_sub (a b c : ℕ) (ha : a = 5 * 6^2 + 5 * 6^1 + 5 * 6^0)
  (hb : b = 6 * 6^1 + 5 * 6^0) (hc : c = 1 * 6^1 + 1 * 6^0) :
  (a + b - c) = 1 * 6^3 + 0 * 6^2 + 5 * 6^1 + 3 * 6^0 :=
by
  -- We should translate the problem context into equivalence
  -- but this part of the actual proof is skipped with sorry.
  sorry

end base6_add_sub_l2119_211913


namespace first_ship_rescued_boy_l2119_211926

noncomputable def river_speed : ℝ := 3 -- River speed is 3 km/h

-- Define the speeds of the ships
def ship1_speed_upstream : ℝ := 4 
def ship2_speed_upstream : ℝ := 6 
def ship3_speed_upstream : ℝ := 10 

-- Define the distance downstream where the boy was found
def boy_distance_from_bridge : ℝ := 6

-- Define the equation for the first ship
def first_ship_equation (c : ℝ) : Prop := (10 - c) / (4 + c) = 1 + 6 / c

-- The problem to prove:
theorem first_ship_rescued_boy : first_ship_equation river_speed :=
by sorry

end first_ship_rescued_boy_l2119_211926


namespace _l2119_211994

noncomputable def tan_alpha_theorem (α : ℝ) (h1 : Real.tan (Real.pi / 4 + α) = 2) : Real.tan α = 1 / 3 :=
by
  sorry

noncomputable def evaluate_expression_theorem (α β : ℝ) 
  (h1 : Real.tan (Real.pi / 4 + α) = 2) 
  (h2 : Real.tan β = 1 / 2) 
  (h3 : Real.tan α = 1 / 3) : 
  (Real.sin (α + β) - 2 * Real.sin α * Real.cos β) / (2 * Real.sin α * Real.sin β + Real.cos (α + β)) = 1 / 7 :=
by
  sorry

end _l2119_211994


namespace exponent_equality_l2119_211956

theorem exponent_equality (x : ℕ) (hx : (1 / 8 : ℝ) * (2 : ℝ) ^ 40 = (2 : ℝ) ^ x) : x = 37 :=
sorry

end exponent_equality_l2119_211956


namespace max_constant_k_l2119_211925

theorem max_constant_k (x y : ℤ) : 4 * x^2 + y^2 + 1 ≥ 3 * x * (y + 1) :=
sorry

end max_constant_k_l2119_211925


namespace fff1_eq_17_l2119_211955

def f (n : ℕ) : ℕ :=
  if n < 3 then n^2 + 1
  else if n < 6 then 3 * n + 2
  else 2 * n - 1

theorem fff1_eq_17 : f (f (f 1)) = 17 :=
  by sorry

end fff1_eq_17_l2119_211955


namespace line_circle_no_intersection_l2119_211992

theorem line_circle_no_intersection :
  (∀ (x y : ℝ), 3 * x + 4 * y = 12 ∨ x^2 + y^2 = 4) →
  (∃ (x y : ℝ), 3 * x + 4 * y = 12 ∧ x^2 + y^2 = 4) →
  false :=
by
  sorry

end line_circle_no_intersection_l2119_211992


namespace find_g_x2_minus_2_l2119_211971

def g : ℝ → ℝ := sorry -- Define g as some real-valued polynomial function.

theorem find_g_x2_minus_2 (x : ℝ) 
(h1 : g (x^2 + 2) = x^4 + 5 * x^2 + 1) : 
  g (x^2 - 2) = x^4 - 3 * x^2 - 7 := 
by sorry

end find_g_x2_minus_2_l2119_211971


namespace largest_5_digit_congruent_l2119_211974

theorem largest_5_digit_congruent (n : ℕ) (h1 : 29 * n + 17 < 100000) : 29 * 3447 + 17 = 99982 :=
by
  -- Proof goes here
  sorry

end largest_5_digit_congruent_l2119_211974


namespace smallest_divisible_by_3_and_4_is_12_l2119_211931

theorem smallest_divisible_by_3_and_4_is_12 
  (n : ℕ) 
  (h1 : ∃ k1 : ℕ, n = 3 * k1) 
  (h2 : ∃ k2 : ℕ, n = 4 * k2) 
  : n ≥ 12 := sorry

end smallest_divisible_by_3_and_4_is_12_l2119_211931


namespace height_eight_times_initial_maximum_growth_year_l2119_211980

noncomputable def t : ℝ := 2^(-2/3 : ℝ)
noncomputable def f (n : ℕ) (A a b t : ℝ) : ℝ := 9 * A / (a + b * t^n)

theorem height_eight_times_initial (A : ℝ) : 
  ∀ n : ℕ, f n A 1 8 t = 8 * A ↔ n = 9 :=
sorry

theorem maximum_growth_year (A : ℝ) :
  ∃ n : ℕ, (∀ k : ℕ, (f n A 1 8 t - f (n-1) A 1 8 t) ≥ (f k A 1 8 t - f (k-1) A 1 8 t))
  ∧ n = 5 :=
sorry

end height_eight_times_initial_maximum_growth_year_l2119_211980


namespace geometric_series_sum_l2119_211968

noncomputable def geometric_sum : ℚ :=
  let a := (2^3 : ℚ) / (3^3)
  let r := (2 : ℚ) / 3
  let n := 12 - 3 + 1
  a * (1 - r^n) / (1 - r)

theorem geometric_series_sum :
  geometric_sum = 1440600 / 59049 :=
by
  sorry

end geometric_series_sum_l2119_211968


namespace pairwise_sums_l2119_211958

theorem pairwise_sums (
  a b c d e : ℕ
) : 
  a < b ∧ b < c ∧ c < d ∧ d < e ∧
  (a + b = 21) ∧ (a + c = 26) ∧ (a + d = 35) ∧ (a + e = 40) ∧
  (b + c = 49) ∧ (b + d = 51) ∧ (b + e = 54) ∧ (c + d = 60) ∧
  (c + e = 65) ∧ (d + e = 79)
  ↔ 
  (a = 6) ∧ (b = 15) ∧ (c = 20) ∧ (d = 34) ∧ (e = 45) := 
by 
  sorry

end pairwise_sums_l2119_211958


namespace num_ways_to_choose_officers_same_gender_l2119_211923

-- Definitions based on conditions
def num_members : Nat := 24
def num_boys : Nat := 12
def num_girls : Nat := 12
def num_officers : Nat := 3

-- Theorem statement using these definitions
theorem num_ways_to_choose_officers_same_gender :
  (num_boys * (num_boys-1) * (num_boys-2) * 2) = 2640 :=
by
  sorry

end num_ways_to_choose_officers_same_gender_l2119_211923


namespace cost_per_play_l2119_211982

-- Conditions
def initial_money : ℝ := 3
def points_per_red_bucket : ℝ := 2
def points_per_green_bucket : ℝ := 3
def rings_per_play : ℕ := 5
def games_played : ℕ := 2
def red_buckets : ℕ := 4
def green_buckets : ℕ := 5
def total_games : ℕ := 3
def total_points : ℝ := 38

-- Point calculations
def points_from_red_buckets : ℝ := red_buckets * points_per_red_bucket
def points_from_green_buckets : ℝ := green_buckets * points_per_green_bucket
def current_points : ℝ := points_from_red_buckets + points_from_green_buckets
def points_needed : ℝ := total_points - current_points

-- Define the theorem statement
theorem cost_per_play :
  (initial_money / (games_played : ℝ)) = 1.50 :=
  sorry

end cost_per_play_l2119_211982


namespace amy_total_score_l2119_211919

theorem amy_total_score :
  let points_per_treasure := 4
  let treasures_first_level := 6
  let treasures_second_level := 2
  let score_first_level := treasures_first_level * points_per_treasure
  let score_second_level := treasures_second_level * points_per_treasure
  let total_score := score_first_level + score_second_level
  total_score = 32 := by
sorry

end amy_total_score_l2119_211919


namespace cos_ninety_degrees_l2119_211970

theorem cos_ninety_degrees : Real.cos (90 * Real.pi / 180) = 0 := 
by 
  sorry

end cos_ninety_degrees_l2119_211970


namespace problem_statement_l2119_211953

theorem problem_statement (x y : ℝ) (p : x > 0 ∧ y > 0) : (∃ p, p → xy > 0) ∧ ¬(xy > 0 → x > 0 ∧ y > 0) :=
by
  sorry

end problem_statement_l2119_211953


namespace prove_fraction_identity_l2119_211908

-- Define the conditions and the entities involved
variables {x y : ℝ} (h₁ : x ≠ 0) (h₂ : y ≠ 0) (h₃ : 3 * x + y / 3 ≠ 0)

-- Formulate the theorem statement
theorem prove_fraction_identity :
  (3 * x + y / 3)⁻¹ * ((3 * x)⁻¹ + (y / 3)⁻¹) = (x * y)⁻¹ :=
sorry

end prove_fraction_identity_l2119_211908


namespace probability_of_two_green_apples_l2119_211904

theorem probability_of_two_green_apples (total_apples green_apples choose_apples : ℕ)
  (h_total : total_apples = 8)
  (h_green : green_apples = 4)
  (h_choose : choose_apples = 2) 
: (Nat.choose green_apples choose_apples : ℚ) / (Nat.choose total_apples choose_apples) = 3 / 14 := 
by
  -- This part we would provide a proof, but for now we will use sorry
  sorry

end probability_of_two_green_apples_l2119_211904


namespace vertical_asymptote_x_value_l2119_211915

theorem vertical_asymptote_x_value (x : ℝ) : 4 * x - 9 = 0 → x = 9 / 4 :=
by
  sorry

end vertical_asymptote_x_value_l2119_211915


namespace ratio_sqrt_2_l2119_211957

theorem ratio_sqrt_2 {a b : ℝ} (h1 : a > b) (h2 : b > 0) (h3 : a^2 + b^2 = 6 * a * b) :
  (a + b) / (a - b) = Real.sqrt 2 :=
by
  sorry

end ratio_sqrt_2_l2119_211957


namespace prove_intersection_l2119_211916

-- Defining the set M
def M : Set ℝ := { x | x^2 - 2 * x < 0 }

-- Defining the set N
def N : Set ℝ := { x | x ≥ 1 }

-- Defining the complement of N in ℝ
def complement_N : Set ℝ := { x | x < 1 }

-- The intersection M ∩ complement_N
def intersection : Set ℝ := { x | 0 < x ∧ x < 1 }

-- The statement to be proven
theorem prove_intersection : M ∩ complement_N = intersection :=
by
  sorry

end prove_intersection_l2119_211916


namespace gcd_47_pow6_plus_1_l2119_211906

theorem gcd_47_pow6_plus_1 (h_prime : Prime 47) : 
  Nat.gcd (47^6 + 1) (47^6 + 47^3 + 1) = 1 := 
by 
  sorry

end gcd_47_pow6_plus_1_l2119_211906


namespace find_certain_number_l2119_211911

theorem find_certain_number (h1 : 2994 / 14.5 = 173) (h2 : ∃ x, x / 1.45 = 17.3) : ∃ x, x = 25.085 :=
by
  -- Proof goes here
  sorry

end find_certain_number_l2119_211911


namespace ship_illuminated_by_lighthouse_l2119_211947

theorem ship_illuminated_by_lighthouse (d v : ℝ) (hv : v > 0) (ship_speed : ℝ) 
    (hship_speed : ship_speed ≤ v / 8) (rock_distance : ℝ) 
    (hrock_distance : rock_distance = d):
    ∀ t : ℝ, ∃ t' : ℝ, t' ≤ t ∧ t' = (d * t / v) := sorry

end ship_illuminated_by_lighthouse_l2119_211947


namespace find_x_l2119_211961

theorem find_x :
  let a := 5^3
  let b := 6^2
  a - 7 = b + 82 := 
by
  sorry

end find_x_l2119_211961


namespace arrangement_meeting_ways_l2119_211909

-- For convenience, define the number of members per school and the combination function.
def num_members_per_school : ℕ := 6
def num_schools : ℕ :=  4
def combination (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

theorem arrangement_meeting_ways : 
  let host_ways := num_schools
  let host_reps_ways := combination num_members_per_school 2
  let non_host_schools := num_schools - 1
  let non_host_reps_ways := combination num_members_per_school 2
  let total_non_host_reps_ways := non_host_reps_ways ^ non_host_schools
  let total_ways := host_ways * host_reps_ways * total_non_host_reps_ways
  total_ways = 202500 :=
by 
  -- Definitions and computation is deferred to the steps,
  -- which are to be filled during the proof.
  sorry

end arrangement_meeting_ways_l2119_211909


namespace math_proof_problem_l2119_211984

noncomputable def discriminant (a : ℝ) : ℝ := a^2 - 4 * a + 2

def is_real_roots (a : ℝ) : Prop := discriminant a ≥ 0

def solution_set_a : Set ℝ := { a | is_real_roots a ∧ (a ≤ 2 - Real.sqrt 2 ∨ a ≥ 2 + Real.sqrt 2) }

def f (a : ℝ) : ℝ := -3 * a^2 + 16 * a - 8

def inequality_m (m t : ℝ) : Prop := m^2 + t * m + 4 * Real.sqrt 2 + 6 ≥ f (2 + Real.sqrt 2)

theorem math_proof_problem :
  (∀ a ∈ solution_set_a, ∃ m : ℝ, ∀ t ∈ Set.Icc (-1 : ℝ) (1 : ℝ), inequality_m m t) ∧
  (∀ m t, inequality_m m t → m ≤ -1 ∨ m = 0 ∨ m ≥ 1) :=
by
  sorry

end math_proof_problem_l2119_211984


namespace stratified_sampling_yogurt_adult_milk_powder_sum_l2119_211928

theorem stratified_sampling_yogurt_adult_milk_powder_sum :
  let liquid_milk_brands := 40
  let yogurt_brands := 10
  let infant_formula_brands := 30
  let adult_milk_powder_brands := 20
  let total_brands := liquid_milk_brands + yogurt_brands + infant_formula_brands + adult_milk_powder_brands
  let sample_size := 20
  let yogurt_sample := sample_size * yogurt_brands / total_brands
  let adult_milk_powder_sample := sample_size * adult_milk_powder_brands / total_brands
  yogurt_sample + adult_milk_powder_sample = 6 :=
by
  sorry

end stratified_sampling_yogurt_adult_milk_powder_sum_l2119_211928


namespace smallest_pos_int_gcd_gt_one_l2119_211940

theorem smallest_pos_int_gcd_gt_one : ∃ n: ℕ, n > 0 ∧ (Nat.gcd (8 * n - 3) (5 * n + 4) > 1) ∧ n = 121 :=
by
  sorry

end smallest_pos_int_gcd_gt_one_l2119_211940


namespace solve_for_b_l2119_211942

/-- 
Given the ellipse \( x^2 + \frac{y^2}{b^2 + 1} = 1 \) where \( b > 0 \),
and the eccentricity of the ellipse is \( \frac{\sqrt{10}}{10} \),
prove that \( b = \frac{1}{3} \).
-/
theorem solve_for_b (b : ℝ) (hb : b > 0) (heccentricity : b / (Real.sqrt (b^2 + 1)) = Real.sqrt 10 / 10) : 
  b = 1 / 3 :=
sorry

end solve_for_b_l2119_211942


namespace erin_days_to_receive_30_l2119_211999

theorem erin_days_to_receive_30 (x : ℕ) (h : 3 * x = 30) : x = 10 :=
by
  sorry

end erin_days_to_receive_30_l2119_211999


namespace david_dogs_left_l2119_211937

def total_dogs_left (boxes_small: Nat) (dogs_per_small: Nat) (boxes_large: Nat) (dogs_per_large: Nat) (giveaway_small: Nat) (giveaway_large: Nat): Nat :=
  let total_small := boxes_small * dogs_per_small
  let total_large := boxes_large * dogs_per_large
  let remaining_small := total_small - giveaway_small
  let remaining_large := total_large - giveaway_large
  remaining_small + remaining_large

theorem david_dogs_left :
  total_dogs_left 7 4 5 3 2 1 = 40 := by
  sorry

end david_dogs_left_l2119_211937


namespace no_positive_ints_cube_l2119_211991

theorem no_positive_ints_cube (n : ℕ) : ¬ ∃ y : ℕ, 3 * n^2 + 3 * n + 7 = y^3 := 
sorry

end no_positive_ints_cube_l2119_211991


namespace leonards_age_l2119_211912

variable (L N J : ℕ)

theorem leonards_age (h1 : L = N - 4) (h2 : N = J / 2) (h3 : L + N + J = 36) : L = 6 := 
by 
  sorry

end leonards_age_l2119_211912


namespace smallest_x_for_gx_eq_g1458_l2119_211986

noncomputable def g : ℝ → ℝ := sorry -- You can define the function later.

theorem smallest_x_for_gx_eq_g1458 :
  (∀ x : ℝ, x > 0 → g (3 * x) = 4 * g x) ∧ (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → g x = 2 - 2 * |x - 2|)
  → ∃ x : ℝ, x ≥ 0 ∧ g x = g 1458 ∧ ∀ y : ℝ, y ≥ 0 ∧ g y = g 1458 → x ≤ y ∧ x = 162 := 
by
  sorry

end smallest_x_for_gx_eq_g1458_l2119_211986


namespace simon_age_is_10_l2119_211995

-- Define the conditions
def alvin_age := 30
def half_alvin_age := alvin_age / 2
def simon_age := half_alvin_age - 5

-- State the theorem
theorem simon_age_is_10 : simon_age = 10 :=
by
  sorry

end simon_age_is_10_l2119_211995


namespace median_in_interval_65_69_l2119_211910

-- Definitions for student counts in each interval
def count_50_54 := 5
def count_55_59 := 7
def count_60_64 := 22
def count_65_69 := 19
def count_70_74 := 15
def count_75_79 := 10
def count_80_84 := 18
def count_85_89 := 5

-- Total number of students
def total_students := 101

-- Calculation of the position of the median
def median_position := (total_students + 1) / 2

-- Cumulative counts
def cumulative_up_to_59 := count_50_54 + count_55_59
def cumulative_up_to_64 := cumulative_up_to_59 + count_60_64
def cumulative_up_to_69 := cumulative_up_to_64 + count_65_69

-- Proof statement
theorem median_in_interval_65_69 :
  34 < median_position ∧ median_position ≤ cumulative_up_to_69 :=
by
  sorry

end median_in_interval_65_69_l2119_211910


namespace real_solutions_count_l2119_211989

theorem real_solutions_count :
  ∃ S : Set ℝ, (∀ x : ℝ, x ∈ S ↔ (|x-2| + |x-3| = 1)) ∧ (S = Set.Icc 2 3) :=
sorry

end real_solutions_count_l2119_211989


namespace problem_statement_l2119_211917

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem problem_statement :
  ¬ is_pythagorean_triple 2 3 4 ∧ 
  is_pythagorean_triple 3 4 5 ∧ 
  is_pythagorean_triple 6 8 10 ∧ 
  is_pythagorean_triple 5 12 13 :=
by 
  constructor
  sorry
  constructor
  sorry
  constructor
  sorry
  sorry

end problem_statement_l2119_211917


namespace max_teams_in_chess_tournament_l2119_211924

theorem max_teams_in_chess_tournament :
  ∃ n : ℕ, n * (n - 1) ≤ 500 / 9 ∧ ∀ m : ℕ, m * (m - 1) ≤ 500 / 9 → m ≤ n :=
sorry

end max_teams_in_chess_tournament_l2119_211924


namespace cans_collected_by_first_group_l2119_211914

def class_total_students : ℕ := 30
def students_didnt_collect : ℕ := 2
def students_collected_4 : ℕ := 13
def total_cans_collected : ℕ := 232

theorem cans_collected_by_first_group :
  let remaining_students := class_total_students - (students_didnt_collect + students_collected_4)
  let cans_by_13_students := students_collected_4 * 4
  let cans_by_first_group := total_cans_collected - cans_by_13_students
  let cans_per_student := cans_by_first_group / remaining_students
  cans_per_student = 12 := by
  sorry

end cans_collected_by_first_group_l2119_211914


namespace product_of_two_numbers_less_than_the_smaller_of_the_two_factors_l2119_211900

theorem product_of_two_numbers_less_than_the_smaller_of_the_two_factors
    (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (ha1 : a < 1) (hb1 : b < 1) : 
  a * b < min a b := 
sorry

end product_of_two_numbers_less_than_the_smaller_of_the_two_factors_l2119_211900


namespace valid_numbers_count_l2119_211939

-- Define a predicate that checks if a number is a three-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Define a function that counts how many numbers between 100 and 999 are multiples of 13
def count_multiples_of_13 (start finish : ℕ) : ℕ :=
  (finish - start) / 13 + 1

-- Define a function that checks if a permutation of digits of n is a multiple of 13
-- (actual implementation would require digit manipulation, but we assume its existence here)
def is_permutation_of_digits_multiple_of_13 (n : ℕ) : Prop :=
  ∃ (perm : ℕ), is_three_digit perm ∧ perm % 13 = 0

noncomputable def count_valid_permutations (multiples_of_13 : ℕ) : ℕ :=
  multiples_of_13 * 3 -- Assuming on average

-- Problem statement: Prove that there are 207 valid numbers satisfying the condition
theorem valid_numbers_count : (count_valid_permutations (count_multiples_of_13 104 988)) = 207 := 
by {
  -- Place for proof which is omitted here
  sorry
}

end valid_numbers_count_l2119_211939


namespace original_price_l2119_211907

theorem original_price (SP : ℝ) (gain_percent : ℝ) (P : ℝ) : SP = 1080 → gain_percent = 0.08 → SP = P * (1 + gain_percent) → P = 1000 :=
by
  intro hSP hGainPercent hEquation
  sorry

end original_price_l2119_211907


namespace students_in_trumpet_or_trombone_l2119_211997

theorem students_in_trumpet_or_trombone (h₁ : 0.5 + 0.12 = 0.62) : 
  0.5 + 0.12 = 0.62 :=
by
  exact h₁

end students_in_trumpet_or_trombone_l2119_211997


namespace num_marbles_removed_l2119_211952

theorem num_marbles_removed (total_marbles red_marbles : ℕ) (prob_neither_red : ℚ) 
  (h₁ : total_marbles = 84) (h₂ : red_marbles = 12) (h₃ : prob_neither_red = 36 / 49) : 
  total_marbles - red_marbles = 2 :=
by
  sorry

end num_marbles_removed_l2119_211952


namespace range_c_of_sets_l2119_211969

noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem range_c_of_sets (c : ℝ) (h₀ : c > 0)
  (A := { x : ℝ | log2 x < 1 })
  (B := { x : ℝ | 0 < x ∧ x < c })
  (hA_union_B_eq_B : A ∪ B = B) :
  c ≥ 2 :=
by
  -- Minimum outline is provided, the proof part is replaced with "sorry" to indicate the point to be proved
  sorry

end range_c_of_sets_l2119_211969


namespace isosceles_trapezoid_inscribed_circle_ratio_l2119_211951

noncomputable def ratio_perimeter_inscribed_circle (x : ℝ) : ℝ := 
  (50 * x) / (10 * Real.pi * x)

theorem isosceles_trapezoid_inscribed_circle_ratio 
  (x : ℝ)
  (h1 : x > 0)
  (r : ℝ) 
  (OK OP : ℝ) 
  (h2 : OK = 3 * x) 
  (h3 : OP = 5 * x) : 
  ratio_perimeter_inscribed_circle x = 5 / Real.pi :=
by
  sorry

end isosceles_trapezoid_inscribed_circle_ratio_l2119_211951


namespace chocolate_chip_difference_l2119_211954

noncomputable def V_v : ℕ := 20 -- Viviana's vanilla chips
noncomputable def S_c : ℕ := 25 -- Susana's chocolate chips
noncomputable def S_v : ℕ := 3 * V_v / 4 -- Susana's vanilla chips

theorem chocolate_chip_difference (V_c : ℕ) (h1 : V_c + V_v + S_c + S_v = 90) :
  V_c - S_c = 5 := by sorry

end chocolate_chip_difference_l2119_211954


namespace no_solution_fraction_eq_l2119_211996

theorem no_solution_fraction_eq (a : ℝ) :
  (∀ x : ℝ, x ≠ 1 → (a * x / (x - 1) + 3 / (1 - x) = 2) → false) ↔ a = 2 :=
by
  sorry

end no_solution_fraction_eq_l2119_211996


namespace inequality_holds_for_positive_reals_equality_condition_l2119_211973

theorem inequality_holds_for_positive_reals (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  4 * (a^3 + b^3 + c^3 + 3) ≥ 3 * (a + 1) * (b + 1) * (c + 1) :=
sorry

theorem equality_condition (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (4 * (a^3 + b^3 + c^3 + 3) = 3 * (a + 1) * (b + 1) * (c + 1)) ↔ (a = 1 ∧ b = 1 ∧ c = 1) :=
sorry

end inequality_holds_for_positive_reals_equality_condition_l2119_211973


namespace ap_square_sequel_l2119_211941

theorem ap_square_sequel {a b c : ℝ} (h1 : a ≠ b ∧ b ≠ c ∧ a ≠ c)
                     (h2 : 2 * (b / (c + a)) = (a / (b + c)) + (c / (a + b))) :
  (a^2 + c^2 = 2 * b^2) :=
by
  sorry

end ap_square_sequel_l2119_211941


namespace find_number_l2119_211929

def number_of_faces : ℕ := 6

noncomputable def probability (n : ℕ) : ℚ :=
  (number_of_faces - n : ℕ) / number_of_faces

theorem find_number (n : ℕ) (h: n < number_of_faces) :
  probability n = 1 / 3 → n = 4 :=
by
  -- proof goes here
  sorry

end find_number_l2119_211929


namespace james_weekly_earnings_l2119_211966

def hourly_rate : ℕ := 20
def hours_per_day : ℕ := 8
def days_per_week : ℕ := 4

theorem james_weekly_earnings : hourly_rate * (hours_per_day * days_per_week) = 640 := by
  sorry

end james_weekly_earnings_l2119_211966


namespace natural_numbers_partition_l2119_211902

def isSquare (m : ℕ) : Prop := ∃ k : ℕ, k * k = m

def subsets_with_square_sum (n : ℕ) : Prop :=
  ∀ (A B : Finset ℕ), (A ∪ B = Finset.range (n + 1) ∧ A ∩ B = ∅) →
  ∃ (a b : ℕ), a ≠ b ∧ isSquare (a + b) ∧ (a ∈ A ∨ a ∈ B) ∧ (b ∈ A ∨ b ∈ B)

theorem natural_numbers_partition (n : ℕ) : n ≥ 15 → subsets_with_square_sum n := 
sorry

end natural_numbers_partition_l2119_211902


namespace probability_equal_2s_after_4040_rounds_l2119_211972

/-- 
Given three players Diana, Nathan, and Olivia each starting with $2, each player (with at least $1) 
simultaneously gives $1 to one of the other two players randomly every 20 seconds. 
Prove that the probability that after the bell has rung 4040 times, 
each player will have $2$ is $\frac{1}{4}$.
-/
theorem probability_equal_2s_after_4040_rounds 
  (n_rounds : ℕ) (start_money : ℕ) (probability_outcome : ℚ) :
  n_rounds = 4040 →
  start_money = 2 →
  probability_outcome = 1 / 4 :=
by
  sorry

end probability_equal_2s_after_4040_rounds_l2119_211972


namespace triangle_side_length_sum_l2119_211960

theorem triangle_side_length_sum :
  ∃ (a b c : ℕ), (5: ℝ) ^ 2 + (7: ℝ) ^ 2 - 2 * (5: ℝ) * (7: ℝ) * (Real.cos (Real.pi * 80 / 180)) = (a: ℝ) + Real.sqrt b + Real.sqrt c ∧
  b = 62 ∧ c = 0 :=
sorry

end triangle_side_length_sum_l2119_211960


namespace find_values_l2119_211949

theorem find_values (a b c : ℝ)
  (h1 : 0.005 * a = 0.8)
  (h2 : 0.0025 * b = 0.6)
  (h3 : c = 0.5 * a - 0.1 * b) :
  a = 160 ∧ b = 240 ∧ c = 56 :=
by sorry

end find_values_l2119_211949


namespace find_hyperbola_equation_hyperbola_equation_l2119_211945

-- Define the original hyperbola
def original_hyperbola (x y : ℝ) := (x^2 / 2) - y^2 = 1

-- Define the new hyperbola with unknown constant m
def new_hyperbola (x y m : ℝ) := (x^2 / (m * 2)) - (y^2 / m) = 1

variable (m : ℝ)

-- The point (2, 0)
def point_on_hyperbola (x y : ℝ) := x = 2 ∧ y = 0

theorem find_hyperbola_equation (h : ∀ (x y : ℝ), point_on_hyperbola x y → new_hyperbola x y m) :
  m = 2 :=
    sorry

theorem hyperbola_equation :
  ∀ (x y : ℝ), (x = 2 ∧ y = 0) → (x^2 / 4 - y^2 / 2 = 1) :=
    sorry

end find_hyperbola_equation_hyperbola_equation_l2119_211945


namespace num_divisible_by_10_in_range_correct_l2119_211935

noncomputable def num_divisible_by_10_in_range : ℕ :=
  let a1 := 100
  let d := 10
  let an := 500
  (an - a1) / d + 1

theorem num_divisible_by_10_in_range_correct :
  num_divisible_by_10_in_range = 41 := by
  sorry

end num_divisible_by_10_in_range_correct_l2119_211935


namespace hyperbola_range_l2119_211938

theorem hyperbola_range (m : ℝ) : m * (2 * m - 1) < 0 → 0 < m ∧ m < (1 / 2) :=
by
  intro h
  sorry

end hyperbola_range_l2119_211938


namespace find_subtracted_value_l2119_211921

theorem find_subtracted_value (N V : ℕ) (h1 : N = 1376) (h2 : N / 8 - V = 12) : V = 160 :=
by
  sorry

end find_subtracted_value_l2119_211921


namespace PRINT_3_3_2_l2119_211976

def PRINT (a b : Nat) : Nat × Nat := (a, b)

theorem PRINT_3_3_2 :
  PRINT 3 (3 + 2) = (3, 5) :=
by
  sorry

end PRINT_3_3_2_l2119_211976


namespace compute_expression_l2119_211930

noncomputable def quadratic_roots (a b c : ℝ) :
  {x : ℝ × ℝ // a * x.fst^2 + b * x.fst + c = 0 ∧ a * x.snd^2 + b * x.snd + c = 0} :=
  let Δ := b^2 - 4 * a * c
  let root1 := (-b + Real.sqrt Δ) / (2 * a)
  let root2 := (-b - Real.sqrt Δ) / (2 * a)
  ⟨(root1, root2), by sorry⟩

theorem compute_expression :
  let roots := quadratic_roots 5 (-3) (-4)
  let x1 := roots.val.fst
  let x2 := roots.val.snd
  2 * x1^2 + 3 * x2^2 = (178 : ℝ) / 25 := by
  sorry

end compute_expression_l2119_211930


namespace academic_academy_pass_criteria_l2119_211962

theorem academic_academy_pass_criteria :
  ∀ (total_problems : ℕ) (passing_percentage : ℕ)
  (max_missed : ℕ),
  total_problems = 35 →
  passing_percentage = 80 →
  max_missed = total_problems - (passing_percentage * total_problems) / 100 →
  max_missed = 7 :=
by 
  intros total_problems passing_percentage max_missed
  intros h_total_problems h_passing_percentage h_calculation
  rw [h_total_problems, h_passing_percentage] at h_calculation
  sorry

end academic_academy_pass_criteria_l2119_211962


namespace range_for_a_l2119_211903

variable (a : ℝ)

theorem range_for_a (h : ∀ x : ℝ, x^2 + 2 * x + a > 0) : 1 < a := 
sorry

end range_for_a_l2119_211903


namespace arithmetic_sequence_problem_l2119_211975

noncomputable def arithmetic_sequence_sum : ℕ → ℕ := sorry  -- Define S_n here

theorem arithmetic_sequence_problem (S : ℕ → ℕ) (a : ℕ → ℕ) (h1 : S 8 - S 3 = 10)
    (h2 : ∀ n, S (n + 1) = S n + a (n + 1)) (h3 : a 6 = 2) : S 11 = 22 :=
  sorry

end arithmetic_sequence_problem_l2119_211975


namespace largest_gcd_l2119_211948

theorem largest_gcd (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h_sum : a + b = 1008) : 
  ∃ d : ℕ, d = Int.gcd a b ∧ d = 504 :=
by
  sorry

end largest_gcd_l2119_211948


namespace correct_conclusions_l2119_211901

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.exp x

theorem correct_conclusions :
  (∃ (a b : ℝ), a < b ∧ f a < f b ∧ ∀ x, a < x ∧ x < b → f x < f (x+1)) ∧
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = (x₁ - 2012) ∧ f x₂ = (x₂ - 2012)) :=
by
  sorry

end correct_conclusions_l2119_211901


namespace spanish_teams_in_final_probability_l2119_211993

noncomputable def probability_of_spanish_teams_in_final : ℚ :=
  let teams := 16
  let spanish_teams := 3
  let non_spanish_teams := teams - spanish_teams
  -- Probability calculation based on given conditions and solution steps
  1 - 7 / 15 * 6 / 14

theorem spanish_teams_in_final_probability :
  probability_of_spanish_teams_in_final = 4 / 5 :=
sorry

end spanish_teams_in_final_probability_l2119_211993


namespace coupon_value_l2119_211981

theorem coupon_value
  (bill : ℝ)
  (milk_cost : ℝ)
  (bread_cost : ℝ)
  (detergent_cost : ℝ)
  (banana_cost_per_pound : ℝ)
  (banana_weight : ℝ)
  (half_off : ℝ)
  (amount_left : ℝ)
  (total_without_coupon : ℝ)
  (total_spent : ℝ)
  (coupon_value : ℝ) :
  bill = 20 →
  milk_cost = 4 →
  bread_cost = 3.5 →
  detergent_cost = 10.25 →
  banana_cost_per_pound = 0.75 →
  banana_weight = 2 →
  half_off = 0.5 →
  amount_left = 4 →
  total_without_coupon = milk_cost * half_off + bread_cost + detergent_cost + banana_cost_per_pound * banana_weight →
  total_spent = bill - amount_left →
  coupon_value = total_without_coupon - total_spent →
  coupon_value = 1.25 :=
by
  sorry

end coupon_value_l2119_211981


namespace sheep_daddy_input_l2119_211905

-- Conditions for black box transformations
def black_box (k : ℕ) : ℕ :=
  if k % 2 = 1 then 4 * k + 1 else k / 2

-- The transformation chain with three black boxes
def black_box_chain (k : ℕ) : ℕ :=
  black_box (black_box (black_box k))

-- Theorem statement capturing the problem:
-- Final output m is 2, and the largest input leading to this is 64.
theorem sheep_daddy_input : ∃ k : ℕ, ∀ (k1 k2 k3 k4 : ℕ), 
  black_box_chain k1 = 2 ∧ 
  black_box_chain k2 = 2 ∧ 
  black_box_chain k3 = 2 ∧ 
  black_box_chain k4 = 2 ∧ 
  k1 ≠ k2 ∧ k2 ≠ k3 ∧ k3 ≠ k4 ∧ k4 ≠ k1 ∧ 
  k = max k1 (max k2 (max k3 k4)) → k = 64 :=
sorry  -- Proof is not required

end sheep_daddy_input_l2119_211905


namespace distance_between_first_and_last_pots_l2119_211943

theorem distance_between_first_and_last_pots (n : ℕ) (d : ℕ) 
  (h₁ : n = 8) 
  (h₂ : d = 100) : 
  ∃ total_distance : ℕ, total_distance = 175 := 
by 
  sorry

end distance_between_first_and_last_pots_l2119_211943


namespace cake_and_milk_tea_cost_l2119_211950

noncomputable def slice_cost (milk_tea_cost : ℚ) : ℚ := (3 / 4) * milk_tea_cost

noncomputable def total_cost (milk_tea_cost : ℚ) (slice_cost : ℚ) : ℚ :=
  2 * slice_cost + milk_tea_cost

theorem cake_and_milk_tea_cost 
  (milk_tea_cost : ℚ)
  (h : milk_tea_cost = 2.40) :
  total_cost milk_tea_cost (slice_cost milk_tea_cost) = 6.00 :=
by
  sorry

end cake_and_milk_tea_cost_l2119_211950


namespace range_of_a_iff_condition_l2119_211983

theorem range_of_a_iff_condition (a : ℝ) : 
  (∀ x : ℝ, |x + 3| + |x - 7| ≥ a^2 - 3 * a) ↔ (a ≥ -2 ∧ a ≤ 5) :=
by
  sorry

end range_of_a_iff_condition_l2119_211983


namespace smallest_percentage_boys_correct_l2119_211998

noncomputable def smallest_percentage_boys (B : ℝ) : ℝ :=
  if h : 0 ≤ B ∧ B ≤ 1 then B else 0

theorem smallest_percentage_boys_correct :
  ∃ B : ℝ,
    0 ≤ B ∧ B ≤ 1 ∧
    (67.5 / 100 * B * 200 + 25 / 100 * (1 - B) * 200) ≥ 101 ∧
    B = 0.6 :=
by
  sorry

end smallest_percentage_boys_correct_l2119_211998


namespace possible_atomic_numbers_l2119_211918

/-
Given the following conditions:
1. An element X is from Group IIA and exhibits a +2 charge.
2. An element Y is from Group VIIA and exhibits a -1 charge.
Prove that the possible atomic numbers for elements X and Y that can form an ionic compound with the formula XY₂ are 12 for X and 9 for Y.
-/

structure Element :=
  (atomic_number : Nat)
  (group : Nat)
  (charge : Int)

def GroupIIACharge := 2
def GroupVIIACharge := -1

axiom X : Element
axiom Y : Element

theorem possible_atomic_numbers (X_group_IIA : X.group = 2)
                                (X_charge : X.charge = GroupIIACharge)
                                (Y_group_VIIA : Y.group = 7)
                                (Y_charge : Y.charge = GroupVIIACharge) :
  (X.atomic_number = 12) ∧ (Y.atomic_number = 9) :=
sorry

end possible_atomic_numbers_l2119_211918


namespace maximum_value_of_expression_l2119_211934

noncomputable def calc_value (x y z : ℝ) : ℝ :=
  (x^2 - x * y + y^2) * (x^2 - x * z + z^2) * (y^2 - y * z + z^2)

theorem maximum_value_of_expression :
  ∃ x y z : ℝ, 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x + y + z = 3 ∧ x ≥ y ∧ y ≥ z ∧
  calc_value x y z = 2916 / 729 :=
by
  sorry

end maximum_value_of_expression_l2119_211934


namespace line_through_point_inequality_l2119_211920

theorem line_through_point_inequality
  (a b θ : ℝ)
  (h : (b * Real.cos θ + a * Real.sin θ = a * b)) :
  1 / a^2 + 1 / b^2 ≥ 1 := 
  sorry

end line_through_point_inequality_l2119_211920


namespace all_xi_equal_l2119_211922

theorem all_xi_equal (P : Polynomial ℤ) (n : ℕ) (hn : n % 2 = 1) (x : Fin n → ℤ) 
  (hP : ∀ i : Fin n, P.eval (x i) = x ⟨i + 1, sorry⟩) : 
  ∀ i j : Fin n, x i = x j :=
by
  sorry

end all_xi_equal_l2119_211922


namespace sequence_value_at_99_l2119_211933

theorem sequence_value_at_99 :
  ∃ a : ℕ → ℚ, (a 1 = 2) ∧ (∀ n : ℕ, a (n + 1) = a n + n / 2) ∧ (a 99 = 2427.5) :=
by
  sorry

end sequence_value_at_99_l2119_211933


namespace inequality_transitive_l2119_211927

theorem inequality_transitive {a b c d : ℝ} (h1 : a > b) (h2 : c > d) (h3 : c ≠ 0) (h4 : d ≠ 0) :
  a + c > b + d :=
by {
  sorry
}

end inequality_transitive_l2119_211927


namespace technicians_count_l2119_211965

-- Define the number of workers
def total_workers : ℕ := 21

-- Define the average salaries
def avg_salary_all : ℕ := 8000
def avg_salary_technicians : ℕ := 12000
def avg_salary_rest : ℕ := 6000

-- Define the number of technicians and rest of workers
variable (T R : ℕ)

-- Define the equations based on given conditions
def equation1 := T + R = total_workers
def equation2 := (T * avg_salary_technicians) + (R * avg_salary_rest) = total_workers * avg_salary_all

-- Prove the number of technicians
theorem technicians_count : T = 7 :=
by
  sorry

end technicians_count_l2119_211965


namespace find_p_q_l2119_211936

noncomputable def roots_of_polynomial (a b c : ℝ) :=
  a^3 - 2018 * a + 2018 = 0 ∧ b^3 - 2018 * b + 2018 = 0 ∧ c^3 - 2018 * c + 2018 = 0

theorem find_p_q (a b c : ℝ) (p q : ℕ) 
  (h1 : roots_of_polynomial a b c)
  (h2 : 0 < p ∧ p ≤ q) 
  (h3 : (a^(p+q) + b^(p+q) + c^(p+q))/(p+q) = (a^p + b^p + c^p)/p * (a^q + b^q + c^q)/q) : 
  p^2 + q^2 = 20 := 
sorry

end find_p_q_l2119_211936


namespace polygon_sides_eq_six_l2119_211985

theorem polygon_sides_eq_six (n : ℕ) 
  (h1 : (n - 2) * 180 = (2 * 360)) 
  (h2 : exterior_sum = 360) :
  n = 6 := 
by
  sorry

end polygon_sides_eq_six_l2119_211985


namespace quadratic_no_real_roots_l2119_211946

theorem quadratic_no_real_roots (b c : ℝ) (h : ∀ x : ℝ, x^2 + b * x + c > 0) : 
  ¬ ∃ x : ℝ, x^2 + b * x + c = 0 :=
sorry

end quadratic_no_real_roots_l2119_211946


namespace simplify_expression_l2119_211964

variable (x y : ℝ)

theorem simplify_expression : 3 * x + 6 * x + 9 * x + 12 * x + 15 * x + 9 * y = 45 * x + 9 * y := 
by sorry

end simplify_expression_l2119_211964


namespace sum_of_arithmetic_sequence_l2119_211959

theorem sum_of_arithmetic_sequence
  (a : ℕ → ℚ)
  (S : ℕ → ℚ)
  (h1 : a 2 * a 4 * a 6 * a 8 = 120)
  (h2 : 1 / (a 4 * a 6 * a 8) + 1 / (a 2 * a 6 * a 8) + 1 / (a 2 * a 4 * a 8) + 1 / (a 2 * a 4 * a 6) = 7/60) :
  S 9 = 63/2 :=
by
  sorry

end sum_of_arithmetic_sequence_l2119_211959


namespace medium_supermarkets_in_sample_l2119_211967

-- Define the conditions
def large_supermarkets : ℕ := 200
def medium_supermarkets : ℕ := 400
def small_supermarkets : ℕ := 1400
def total_supermarkets : ℕ := large_supermarkets + medium_supermarkets + small_supermarkets
def sample_size : ℕ := 100
def proportion_medium := (medium_supermarkets : ℚ) / (total_supermarkets : ℚ)

-- The main theorem to prove
theorem medium_supermarkets_in_sample : sample_size * proportion_medium = 20 := by
  sorry

end medium_supermarkets_in_sample_l2119_211967


namespace intersection_A_B_l2119_211977

def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {-2, -1, 1, 2}

theorem intersection_A_B : A ∩ B = {1, 2} :=
by 
  sorry

end intersection_A_B_l2119_211977


namespace percent_students_own_only_cats_l2119_211944

theorem percent_students_own_only_cats (total_students : ℕ) (students_owning_cats : ℕ) (students_owning_dogs : ℕ) (students_owning_both : ℕ) (h_total : total_students = 500) (h_cats : students_owning_cats = 80) (h_dogs : students_owning_dogs = 150) (h_both : students_owning_both = 40) : 
  (students_owning_cats - students_owning_both) * 100 / total_students = 8 := 
by
  sorry

end percent_students_own_only_cats_l2119_211944


namespace boat_current_ratio_l2119_211963

noncomputable def boat_speed_ratio (b c : ℝ) (d : ℝ) : Prop :=
  let time_upstream := 6
  let time_downstream := 10
  d = time_upstream * (b - c) ∧ 
  d = time_downstream * (b + c) → 
  b / c = 4

theorem boat_current_ratio (b c d : ℝ) (h1 : d = 6 * (b - c)) (h2 : d = 10 * (b + c)) : b / c = 4 :=
by sorry

end boat_current_ratio_l2119_211963


namespace square_area_with_circles_l2119_211978

theorem square_area_with_circles 
  (radius : ℝ) 
  (circle_count : ℕ) 
  (side_length : ℝ) 
  (total_area : ℝ)
  (h1 : radius = 7)
  (h2 : circle_count = 4)
  (h3 : side_length = 2 * (2 * radius))
  (h4 : total_area = side_length * side_length)
  : total_area = 784 :=
sorry

end square_area_with_circles_l2119_211978
