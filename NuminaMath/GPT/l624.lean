import Mathlib

namespace NUMINAMATH_GPT_trivia_team_absentees_l624_62474

theorem trivia_team_absentees (total_members : ℕ) (total_points : ℕ) (points_per_member : ℕ) 
  (h1 : total_members = 5) 
  (h2 : total_points = 6) 
  (h3 : points_per_member = 2) : 
  total_members - (total_points / points_per_member) = 2 := 
by 
  sorry

end NUMINAMATH_GPT_trivia_team_absentees_l624_62474


namespace NUMINAMATH_GPT_problem_inequality_l624_62454

open Real

theorem problem_inequality (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) (h_prod : x * y * z = 1) :
    1 / (x^3 * y) + 1 / (y^3 * z) + 1 / (z^3 * x) ≥ x * y + y * z + z * x :=
sorry

end NUMINAMATH_GPT_problem_inequality_l624_62454


namespace NUMINAMATH_GPT_find_theta_even_fn_l624_62415

noncomputable def f (x θ : ℝ) := Real.sin (x + θ) + Real.cos (x + θ)

theorem find_theta_even_fn (θ : ℝ) (hθ: 0 ≤ θ ∧ θ ≤ π / 2) 
  (h: ∀ x : ℝ, f x θ = f (-x) θ) : θ = π / 4 :=
by sorry

end NUMINAMATH_GPT_find_theta_even_fn_l624_62415


namespace NUMINAMATH_GPT_min_value_of_expression_min_value_achieved_l624_62427

theorem min_value_of_expression (x : ℝ) (h : x > 0) : 
  (x + 3 / (x + 1)) ≥ 2 * Real.sqrt 3 - 1 := 
sorry

theorem min_value_achieved (x : ℝ) (h : x = Real.sqrt 3 - 1) : 
  (x + 3 / (x + 1)) = 2 * Real.sqrt 3 - 1 := 
sorry

end NUMINAMATH_GPT_min_value_of_expression_min_value_achieved_l624_62427


namespace NUMINAMATH_GPT_ninety_eight_squared_l624_62419

theorem ninety_eight_squared : 98^2 = 9604 := by
  sorry

end NUMINAMATH_GPT_ninety_eight_squared_l624_62419


namespace NUMINAMATH_GPT_min_value_fraction_l624_62495

theorem min_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2 * y = 1) : 
  ( (x + 1) * (y + 1) / (x * y) ) >= 8 + 4 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_min_value_fraction_l624_62495


namespace NUMINAMATH_GPT_luke_pages_lemma_l624_62492

def number_of_new_cards : ℕ := 3
def number_of_old_cards : ℕ := 9
def cards_per_page : ℕ := 3
def total_number_of_cards := number_of_new_cards + number_of_old_cards
def total_number_of_pages := total_number_of_cards / cards_per_page

theorem luke_pages_lemma : total_number_of_pages = 4 := by
  sorry

end NUMINAMATH_GPT_luke_pages_lemma_l624_62492


namespace NUMINAMATH_GPT_gcd_problem_l624_62445

-- Define the conditions
def a (d : ℕ) : ℕ := d - 3
def b (d : ℕ) : ℕ := d - 2
def c (d : ℕ) : ℕ := d - 1

-- Define the number formed by digits in the specific form
def abcd (d : ℕ) : ℕ := 1000 * a d + 100 * b d + 10 * c d + d
def dcba (d : ℕ) : ℕ := 1000 * d + 100 * c d + 10 * b d + a d

-- Summing the two numbers
def num_sum (d : ℕ) : ℕ := abcd d + dcba d

-- The GCD of all num_sum(d) where d ranges from 3 to 9
def gcd_of_nums : ℕ := 
  Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd (num_sum 3) (num_sum 4)) (num_sum 5)) (num_sum 6)) (Nat.gcd (num_sum 7) (Nat.gcd (num_sum 8) (num_sum 9)))

theorem gcd_problem : gcd_of_nums = 1111 := sorry

end NUMINAMATH_GPT_gcd_problem_l624_62445


namespace NUMINAMATH_GPT_find_k_for_two_identical_solutions_l624_62444

theorem find_k_for_two_identical_solutions (k : ℝ) :
  (∃ x : ℝ, x^2 = 4 * x + k) ∧ (∀ x : ℝ, x^2 = 4 * x + k → x = 2) ↔ k = -4 :=
by
  sorry

end NUMINAMATH_GPT_find_k_for_two_identical_solutions_l624_62444


namespace NUMINAMATH_GPT_fraction_sum_is_ten_l624_62408

theorem fraction_sum_is_ten :
  (1 / 10) + (2 / 10) + (3 / 10) + (4 / 10) + (5 / 10) + (6 / 10) + (7 / 10) + (8 / 10) + (9 / 10) + (55 / 10) = 10 :=
by
  sorry

end NUMINAMATH_GPT_fraction_sum_is_ten_l624_62408


namespace NUMINAMATH_GPT_sum_lent_is_correct_l624_62414

variable (P : ℝ) -- Sum lent
variable (R : ℝ) -- Interest rate
variable (T : ℝ) -- Time period
variable (I : ℝ) -- Simple interest

-- Conditions
axiom interest_rate : R = 8
axiom time_period : T = 8
axiom simple_interest_formula : I = (P * R * T) / 100
axiom interest_condition : I = P - 900

-- The proof problem
theorem sum_lent_is_correct : P = 2500 := by
  -- The proof is skipped
  sorry

end NUMINAMATH_GPT_sum_lent_is_correct_l624_62414


namespace NUMINAMATH_GPT_least_k_cubed_divisible_by_168_l624_62429

theorem least_k_cubed_divisible_by_168 : ∃ k : ℤ, (k ^ 3) % 168 = 0 ∧ ∀ n : ℤ, (n ^ 3) % 168 = 0 → k ≤ n :=
sorry

end NUMINAMATH_GPT_least_k_cubed_divisible_by_168_l624_62429


namespace NUMINAMATH_GPT_cos_105_degree_value_l624_62462

noncomputable def cos105 : ℝ := Real.cos (105 * Real.pi / 180)

theorem cos_105_degree_value :
  cos105 = (Real.sqrt 2 - Real.sqrt 6) / 4 :=
by
  sorry

end NUMINAMATH_GPT_cos_105_degree_value_l624_62462


namespace NUMINAMATH_GPT_real_roots_condition_l624_62456

theorem real_roots_condition (k m : ℝ) (h : m ≠ 0) : (∃ x : ℝ, x^2 + k * x + m = 0) ↔ (m ≤ k^2 / 4) :=
by
  sorry

end NUMINAMATH_GPT_real_roots_condition_l624_62456


namespace NUMINAMATH_GPT_find_prime_p_l624_62497

noncomputable def isPerfectSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem find_prime_p (p : ℕ) (hp : p.Prime) (hsquare : isPerfectSquare (5^p + 12^p)) : p = 2 := 
sorry

end NUMINAMATH_GPT_find_prime_p_l624_62497


namespace NUMINAMATH_GPT_kayak_total_until_May_l624_62447

noncomputable def kayak_number (n : ℕ) : ℕ :=
  if n = 0 then 5
  else 3 * kayak_number (n - 1)

theorem kayak_total_until_May : kayak_number 0 + kayak_number 1 + kayak_number 2 + kayak_number 3 = 200 := by
  sorry

end NUMINAMATH_GPT_kayak_total_until_May_l624_62447


namespace NUMINAMATH_GPT_no_positive_n_for_prime_expr_l624_62471

noncomputable def is_prime (p : ℤ) : Prop := p > 1 ∧ (∀ m : ℤ, 1 < m → m < p → ¬ (m ∣ p))

theorem no_positive_n_for_prime_expr : 
  ∀ n : ℕ, 0 < n → ¬ is_prime (n^3 - 9 * n^2 + 23 * n - 17) := by
  sorry

end NUMINAMATH_GPT_no_positive_n_for_prime_expr_l624_62471


namespace NUMINAMATH_GPT_positive_number_l624_62486

theorem positive_number (x : ℝ) (h1 : 0 < x) (h2 : (2 / 3) * x = (144 / 216) * (1 / x)) : x = 1 := sorry

end NUMINAMATH_GPT_positive_number_l624_62486


namespace NUMINAMATH_GPT_math_problem_l624_62435

theorem math_problem (a b : ℝ) (h : |a + 1| + (b - 2)^2 = 0) : (a + b)^9 + a^6 = 2 :=
sorry

end NUMINAMATH_GPT_math_problem_l624_62435


namespace NUMINAMATH_GPT_bricks_required_l624_62411

theorem bricks_required (courtyard_length_m : ℕ) (courtyard_width_m : ℕ)
  (brick_length_cm : ℕ) (brick_width_cm : ℕ)
  (h1 : courtyard_length_m = 30) (h2 : courtyard_width_m = 16)
  (h3 : brick_length_cm = 20) (h4 : brick_width_cm = 10) :
  (3000 * 1600) / (20 * 10) = 24000 :=
by sorry

end NUMINAMATH_GPT_bricks_required_l624_62411


namespace NUMINAMATH_GPT_compound_interest_l624_62421

theorem compound_interest (SI : ℝ) (R : ℝ) (T : ℝ) (P : ℝ) (CI : ℝ) :
  SI = 40 → R = 5 → T = 2 → SI = (P * R * T) / 100 → CI = P * ((1 + R / 100) ^ T - 1) → CI = 41 :=
by sorry

end NUMINAMATH_GPT_compound_interest_l624_62421


namespace NUMINAMATH_GPT_Jason_4week_visits_l624_62479

-- Definitions
def William_weekly_visits : ℕ := 2
def Jason_weekly_multiplier : ℕ := 4
def weeks_period : ℕ := 4

-- We need to prove that Jason goes to the library 32 times in 4 weeks.
theorem Jason_4week_visits : William_weekly_visits * Jason_weekly_multiplier * weeks_period = 32 := 
by sorry

end NUMINAMATH_GPT_Jason_4week_visits_l624_62479


namespace NUMINAMATH_GPT_wreaths_per_greek_l624_62418

variable (m : ℕ) (m_pos : m > 0)

theorem wreaths_per_greek : ∃ x, x = 4 * m := 
sorry

end NUMINAMATH_GPT_wreaths_per_greek_l624_62418


namespace NUMINAMATH_GPT_compute_n_binom_l624_62441

-- Definitions based on conditions
def n : ℕ := sorry  -- Assume n is a positive integer defined elsewhere
def k : ℕ := 4

-- The binomial coefficient definition
def binom (n k : ℕ) : ℕ :=
  if h₁ : k ≤ n then
    (Nat.factorial n) / ((Nat.factorial k) * Nat.factorial (n - k))
  else 0

-- The theorem to prove
theorem compute_n_binom : n * binom k 3 = 4 * n :=
by
  sorry

end NUMINAMATH_GPT_compute_n_binom_l624_62441


namespace NUMINAMATH_GPT_race_distance_l624_62402

theorem race_distance (d v_A v_B v_C : ℝ) (h1 : d / v_A = (d - 20) / v_B)
  (h2 : d / v_B = (d - 10) / v_C) (h3 : d / v_A = (d - 28) / v_C) : d = 100 :=
by
  sorry

end NUMINAMATH_GPT_race_distance_l624_62402


namespace NUMINAMATH_GPT_distance_traveled_is_correct_l624_62437

noncomputable def speed_in_mph : ℝ := 23.863636363636363
noncomputable def seconds : ℝ := 2

-- constants for conversion
def miles_to_feet : ℝ := 5280
def hours_to_seconds : ℝ := 3600

-- speed in feet per second
noncomputable def speed_in_fps : ℝ := speed_in_mph * miles_to_feet / hours_to_seconds

-- distance traveled
noncomputable def distance : ℝ := speed_in_fps * seconds

theorem distance_traveled_is_correct : distance = 69.68 := by
  sorry

end NUMINAMATH_GPT_distance_traveled_is_correct_l624_62437


namespace NUMINAMATH_GPT_my_current_age_l624_62478

-- Definitions based on the conditions
def bro_age (x : ℕ) : ℕ := 2 * x - 5

-- Main theorem to prove that my current age is 13 given the conditions
theorem my_current_age 
  (x y : ℕ)
  (h1 : y - 5 = 2 * (x - 5))
  (h2 : (x + 8) + (y + 8) = 50) :
  x = 13 :=
sorry

end NUMINAMATH_GPT_my_current_age_l624_62478


namespace NUMINAMATH_GPT_alicia_tax_correct_l624_62430

theorem alicia_tax_correct :
  let hourly_wage_dollars := 25
  let hourly_wage_cents := hourly_wage_dollars * 100
  let basic_tax_rate := 0.01
  let additional_tax_rate := 0.0075
  let basic_tax := basic_tax_rate * hourly_wage_cents
  let excess_amount_cents := (hourly_wage_dollars - 20) * 100
  let additional_tax := additional_tax_rate * excess_amount_cents
  basic_tax + additional_tax = 28.75 := 
by
  sorry

end NUMINAMATH_GPT_alicia_tax_correct_l624_62430


namespace NUMINAMATH_GPT_area_R3_l624_62407

-- Define the initial dimensions of rectangle R1
def length_R1 := 8
def width_R1 := 4

-- Define the dimensions of rectangle R2 after bisecting R1
def length_R2 := length_R1 / 2
def width_R2 := width_R1

-- Define the dimensions of rectangle R3 after bisecting R2
def length_R3 := length_R2 / 2
def width_R3 := width_R2

-- Prove that the area of R3 is 8
theorem area_R3 : (length_R3 * width_R3) = 8 := by
  -- Calculation for the theorem
  sorry

end NUMINAMATH_GPT_area_R3_l624_62407


namespace NUMINAMATH_GPT_fruit_basket_count_l624_62423

/-- We have seven identical apples and twelve identical oranges.
    A fruit basket must contain at least one piece of fruit.
    Prove that the number of different fruit baskets we can make
    is 103. -/
theorem fruit_basket_count :
  let apples := 7
  let oranges := 12
  let total_possible_baskets := (apples + 1) * (oranges + 1) - 1
  total_possible_baskets = 103 :=
by
  let apples := 7
  let oranges := 12
  let total_possible_baskets := (apples + 1) * (oranges + 1) - 1
  show total_possible_baskets = 103
  sorry

end NUMINAMATH_GPT_fruit_basket_count_l624_62423


namespace NUMINAMATH_GPT_simplify_expression_correct_l624_62488

def simplify_expression : ℚ :=
  (5^5 + 5^3) / (5^4 - 5^2)

theorem simplify_expression_correct : simplify_expression = 65 / 12 :=
  sorry

end NUMINAMATH_GPT_simplify_expression_correct_l624_62488


namespace NUMINAMATH_GPT_percentage_fescue_in_Y_l624_62466

-- Define the seed mixtures and their compositions
structure SeedMixture :=
  (ryegrass : ℝ)  -- percentage of ryegrass

-- Seed mixture X
def X : SeedMixture := { ryegrass := 0.40 }

-- Seed mixture Y
def Y : SeedMixture := { ryegrass := 0.25 }

-- Mixture of X and Y contains 32 percent ryegrass
def mixture_percentage := 0.32

-- 46.67 percent of the weight of this mixture is X
def weight_X := 0.4667

-- Question: What percent of seed mixture Y is fescue
theorem percentage_fescue_in_Y : (1 - Y.ryegrass) = 0.75 := by
  sorry

end NUMINAMATH_GPT_percentage_fescue_in_Y_l624_62466


namespace NUMINAMATH_GPT_sequences_of_lemon_recipients_l624_62468

theorem sequences_of_lemon_recipients :
  let students := 15
  let days := 5
  let total_sequences := students ^ days
  total_sequences = 759375 :=
by
  let students := 15
  let days := 5
  let total_sequences := students ^ days
  have h : total_sequences = 759375 := by sorry
  exact h

end NUMINAMATH_GPT_sequences_of_lemon_recipients_l624_62468


namespace NUMINAMATH_GPT_numbers_divisible_l624_62420

theorem numbers_divisible (n : ℕ) (d1 d2 : ℕ) (lcm_d1_d2 : ℕ) (limit : ℕ) (h_lcm: lcm d1 d2 = lcm_d1_d2) (h_limit : limit = 2011)
(h_d1 : d1 = 117) (h_d2 : d2 = 2) : 
  ∃ k : ℕ, k = 8 ∧ ∀ m : ℕ, m < limit → (m % lcm_d1_d2 = 0 ↔ ∃ i : ℕ, i < k ∧ m = lcm_d1_d2 * (i + 1)) :=
by
  sorry

end NUMINAMATH_GPT_numbers_divisible_l624_62420


namespace NUMINAMATH_GPT_basketball_weight_calc_l624_62401

-- Define the variables and conditions
variable (weight_basketball weight_watermelon : ℕ)
variable (h1 : 8 * weight_basketball = 4 * weight_watermelon)
variable (h2 : weight_watermelon = 32)

-- Statement to prove
theorem basketball_weight_calc : weight_basketball = 16 :=
by
  sorry

end NUMINAMATH_GPT_basketball_weight_calc_l624_62401


namespace NUMINAMATH_GPT_percentage_defective_meters_l624_62448

theorem percentage_defective_meters (total_meters : ℕ) (defective_meters : ℕ) (percentage : ℚ) :
  total_meters = 2500 →
  defective_meters = 2 →
  percentage = (defective_meters / total_meters) * 100 →
  percentage = 0.08 := 
sorry

end NUMINAMATH_GPT_percentage_defective_meters_l624_62448


namespace NUMINAMATH_GPT_Mina_digits_l624_62496

theorem Mina_digits (Carlos Sam Mina : ℕ) 
  (h1 : Sam = Carlos + 6) 
  (h2 : Mina = 6 * Carlos) 
  (h3 : Sam = 10) : 
  Mina = 24 := 
sorry

end NUMINAMATH_GPT_Mina_digits_l624_62496


namespace NUMINAMATH_GPT_sin_seven_pi_div_six_l624_62469

theorem sin_seven_pi_div_six : Real.sin (7 * Real.pi / 6) = -1 / 2 := 
  sorry

end NUMINAMATH_GPT_sin_seven_pi_div_six_l624_62469


namespace NUMINAMATH_GPT_students_count_geometry_history_science_l624_62493

noncomputable def number_of_students (geometry_only history_only science_only 
                                      geometry_and_history geometry_and_science : ℕ) : ℕ :=
  geometry_only + history_only + science_only

theorem students_count_geometry_history_science (geometry_total history_only science_only 
                                                 geometry_and_history geometry_and_science : ℕ) :
  geometry_total = 30 →
  geometry_and_history = 15 →
  history_only = 15 →
  geometry_and_science = 8 →
  science_only = 10 →
  number_of_students (geometry_total - geometry_and_history - geometry_and_science)
                     history_only
                     science_only = 32 :=
by
  sorry

end NUMINAMATH_GPT_students_count_geometry_history_science_l624_62493


namespace NUMINAMATH_GPT_cone_volume_l624_62443

theorem cone_volume :
  ∀ (l h : ℝ) (r : ℝ), l = 15 ∧ h = 9 ∧ h = 3 * r → 
  (1 / 3) * Real.pi * r^2 * h = 27 * Real.pi :=
by
  intros l h r
  intro h_eqns
  sorry

end NUMINAMATH_GPT_cone_volume_l624_62443


namespace NUMINAMATH_GPT_total_limes_picked_l624_62458

theorem total_limes_picked (Alyssa_limes Mike_limes : ℕ) 
        (hAlyssa : Alyssa_limes = 25) (hMike : Mike_limes = 32) : 
       Alyssa_limes + Mike_limes = 57 :=
by {
  sorry
}

end NUMINAMATH_GPT_total_limes_picked_l624_62458


namespace NUMINAMATH_GPT_sin_2theta_value_l624_62422

theorem sin_2theta_value (θ : ℝ) (h : ∑' n, (Real.sin θ)^(2 * n) = 3) : Real.sin (2 * θ) = (2 * Real.sqrt 2) / 3 :=
by
  sorry

end NUMINAMATH_GPT_sin_2theta_value_l624_62422


namespace NUMINAMATH_GPT_gino_gave_away_l624_62487

theorem gino_gave_away (initial_sticks given_away left_sticks : ℝ) 
  (h1 : initial_sticks = 63.0) (h2 : left_sticks = 13.0) 
  (h3 : left_sticks = initial_sticks - given_away) : 
  given_away = 50.0 :=
by
  sorry

end NUMINAMATH_GPT_gino_gave_away_l624_62487


namespace NUMINAMATH_GPT_right_triangle_side_length_l624_62431

theorem right_triangle_side_length
  (c : ℕ) (a : ℕ) (h_c : c = 13) (h_a : a = 12) :
  ∃ b : ℕ, b = 5 ∧ c^2 = a^2 + b^2 :=
by
  -- Definitions from conditions
  have h_c_square : c^2 = 169 := by rw [h_c]; norm_num
  have h_a_square : a^2 = 144 := by rw [h_a]; norm_num
  -- Prove the final result
  sorry

end NUMINAMATH_GPT_right_triangle_side_length_l624_62431


namespace NUMINAMATH_GPT_production_value_n_l624_62438

theorem production_value_n :
  -- Definitions based on conditions:
  (∀ a b : ℝ,
    (120 * a + 120 * b) / 60 = 6 ∧
    (100 * a + 100 * b) / 30 = 30) →
  (∃ n : ℝ, 80 * 3 * (a + b) = 480 * a + n * b) →
  n = 120 :=
by
  sorry

end NUMINAMATH_GPT_production_value_n_l624_62438


namespace NUMINAMATH_GPT_inequality_solution_range_4_l624_62457

theorem inequality_solution_range_4 (a : ℝ) : 
  (∃ x : ℝ, |x - 2| - |x + 2| ≥ a) → a ≤ 4 :=
sorry

end NUMINAMATH_GPT_inequality_solution_range_4_l624_62457


namespace NUMINAMATH_GPT_first_candidate_percentage_l624_62453

-- Conditions
def total_votes : ℕ := 600
def second_candidate_votes : ℕ := 240
def first_candidate_votes : ℕ := total_votes - second_candidate_votes

-- Question and correct answer
theorem first_candidate_percentage : (first_candidate_votes * 100) / total_votes = 60 := by
  sorry

end NUMINAMATH_GPT_first_candidate_percentage_l624_62453


namespace NUMINAMATH_GPT_range_of_a_l624_62400

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a * x > 0) → a < 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l624_62400


namespace NUMINAMATH_GPT_sheila_saving_years_l624_62499

theorem sheila_saving_years 
  (initial_amount : ℝ) 
  (monthly_saving : ℝ) 
  (secret_addition : ℝ) 
  (final_amount : ℝ) 
  (years : ℝ) : 
  initial_amount = 3000 ∧ 
  monthly_saving = 276 ∧ 
  secret_addition = 7000 ∧ 
  final_amount = 23248 → 
  years = 4 := 
sorry

end NUMINAMATH_GPT_sheila_saving_years_l624_62499


namespace NUMINAMATH_GPT_remainder_of_modified_expression_l624_62406

theorem remainder_of_modified_expression (x y u v : ℕ) (h : x = u * y + v) (hy_pos : y > 0) (hv_bound : 0 ≤ v ∧ v < y) :
  (x + 3 * u * y + 4) % y = v + 4 :=
by sorry

end NUMINAMATH_GPT_remainder_of_modified_expression_l624_62406


namespace NUMINAMATH_GPT_most_stable_performance_l624_62434

-- Define the variances for each player
def variance_A : ℝ := 0.66
def variance_B : ℝ := 0.52
def variance_C : ℝ := 0.58
def variance_D : ℝ := 0.62

-- State the theorem
theorem most_stable_performance : variance_B < variance_C ∧ variance_C < variance_D ∧ variance_D < variance_A :=
by
  -- Since we are tasked to write only the statement, the proof part is skipped.
  sorry

end NUMINAMATH_GPT_most_stable_performance_l624_62434


namespace NUMINAMATH_GPT_prob1_prob2_prob3_l624_62476

noncomputable def f (x : ℝ) : ℝ :=
  if h : x ≥ 0 then x^2 + 2
  else x

theorem prob1 :
  (∀ x, x ≥ 0 → f x = x^2 + 2) ∧
  (∀ x, x < 0 → f x = x) :=
by
  sorry

theorem prob2 : f 5 = 27 :=
by 
  sorry

theorem prob3 : ∀ (x : ℝ), f x = 0 → false :=
by
  sorry

end NUMINAMATH_GPT_prob1_prob2_prob3_l624_62476


namespace NUMINAMATH_GPT_remainder_when_divided_l624_62416

theorem remainder_when_divided (k : ℕ) (h_pos : 0 < k) (h_rem : 80 % k = 8) : 150 % (k^2) = 69 := by 
  sorry

end NUMINAMATH_GPT_remainder_when_divided_l624_62416


namespace NUMINAMATH_GPT_range_of_m_l624_62473

noncomputable def intersects_x_axis (m : ℝ) : Prop :=
  ∃ x : ℝ, m * x^2 - 4 * x + 1 = 0

theorem range_of_m (m : ℝ) (h : intersects_x_axis m) : m ≤ 4 := by
  sorry

end NUMINAMATH_GPT_range_of_m_l624_62473


namespace NUMINAMATH_GPT_selling_price_is_correct_l624_62412

-- Define the constants used in the problem
noncomputable def cost_price : ℝ := 540
noncomputable def markup_percentage : ℝ := 0.15
noncomputable def discount_percentage : ℝ := 26.570048309178745 / 100

-- Define the conditions in the problem
noncomputable def marked_price : ℝ := cost_price * (1 + markup_percentage)
noncomputable def discount_amount : ℝ := marked_price * discount_percentage
noncomputable def selling_price : ℝ := marked_price - discount_amount

-- Theorem stating the problem
theorem selling_price_is_correct : selling_price = 456 := by 
  sorry

end NUMINAMATH_GPT_selling_price_is_correct_l624_62412


namespace NUMINAMATH_GPT_value_of_n_l624_62467

theorem value_of_n 
  {a b n : ℕ} (ha : a > 0) (hb : b > 0) 
  (h : (1 + b)^n = 243) : 
  n = 5 := by 
  sorry

end NUMINAMATH_GPT_value_of_n_l624_62467


namespace NUMINAMATH_GPT_max_students_total_l624_62477

def max_students_class (a b : ℕ) (h : 3 * a + 5 * b = 115) : ℕ :=
  a + b

theorem max_students_total :
  ∃ a b : ℕ, 3 * a + 5 * b = 115 ∧ max_students_class a b (by sorry) = 37 :=
sorry

end NUMINAMATH_GPT_max_students_total_l624_62477


namespace NUMINAMATH_GPT_slope_of_line_l624_62455

theorem slope_of_line (x y : ℝ) : (4 * y = 5 * x - 20) → (y = (5/4) * x - 5) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_slope_of_line_l624_62455


namespace NUMINAMATH_GPT_leak_drain_time_l624_62417

theorem leak_drain_time (P L : ℝ) (h1 : P = 0.5) (h2 : (P - L) = (6 / 13)) :
    (1 / L) = 26 := by
  sorry

end NUMINAMATH_GPT_leak_drain_time_l624_62417


namespace NUMINAMATH_GPT_segment_length_segment_fraction_three_segments_fraction_l624_62428

noncomputable def total_length : ℝ := 4
noncomputable def number_of_segments : ℕ := 5

theorem segment_length (L : ℝ) (n : ℕ) (hL : L = total_length) (hn : n = number_of_segments) :
  L / n = (4 / 5 : ℝ) := by
sorry

theorem segment_fraction (n : ℕ) (hn : n = number_of_segments) :
  (1 / n : ℝ) = (1 / 5 : ℝ) := by
sorry

theorem three_segments_fraction (n : ℕ) (hn : n = number_of_segments) :
  (3 / n : ℝ) = (3 / 5 : ℝ) := by
sorry

end NUMINAMATH_GPT_segment_length_segment_fraction_three_segments_fraction_l624_62428


namespace NUMINAMATH_GPT_div_condition_l624_62452

theorem div_condition
  (a b : ℕ)
  (h₁ : a < 1000)
  (h₂ : b ≠ 0)
  (h₃ : b ∣ a ^ 21)
  (h₄ : b ^ 10 ∣ a ^ 21) :
  b ∣ a ^ 2 :=
sorry

end NUMINAMATH_GPT_div_condition_l624_62452


namespace NUMINAMATH_GPT_face_value_of_share_l624_62475

theorem face_value_of_share (FV : ℝ) (market_value : ℝ) (dividend_rate : ℝ) (desired_return_rate : ℝ) 
  (H1 : market_value = 15) 
  (H2 : dividend_rate = 0.09) 
  (H3 : desired_return_rate = 0.12) 
  (H4 : dividend_rate * FV = desired_return_rate * market_value) :
  FV = 20 := 
by
  sorry

end NUMINAMATH_GPT_face_value_of_share_l624_62475


namespace NUMINAMATH_GPT_trigonometric_identity_l624_62404

variable (α : Real)

theorem trigonometric_identity 
  (h : Real.sin (π / 6 + α) = Real.sqrt 3 / 3) :
  Real.cos (π / 3 - α) = Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_GPT_trigonometric_identity_l624_62404


namespace NUMINAMATH_GPT_symmetric_point_l624_62461

theorem symmetric_point (x y : ℝ) : 
  (x - 2 * y + 1 = 0) ∧ (y / x * 1 / 2 = -1) → (x = -2/5 ∧ y = 4/5) :=
by 
  sorry

end NUMINAMATH_GPT_symmetric_point_l624_62461


namespace NUMINAMATH_GPT_island_solution_l624_62464

-- Definitions based on conditions
def is_liar (n : ℕ) (m : ℕ) : Prop := n = m + 2 ∨ n = m - 2
def is_truth_teller (n : ℕ) (m : ℕ) : Prop := n = m

-- Residents' statements
def first_resident_statement (liars : ℕ) (truth_tellers : ℕ) : Prop :=
  is_truth_teller liars 1001 ∧ is_truth_teller truth_tellers 1002 ∨
  is_liar liars 1001 ∧ is_liar truth_tellers 1002

def second_resident_statement (liars : ℕ) (truth_tellers : ℕ) : Prop :=
  is_truth_teller liars 1000 ∧ is_truth_teller truth_tellers 999 ∨
  is_liar liars 1000 ∧ is_liar truth_tellers 999

-- Proving the correct number of liars and truth-tellers, and identifying the residents
theorem island_solution :
  ∃ (liars : ℕ) (truth_tellers : ℕ),
    first_resident_statement (liars + 1) (truth_tellers + 1) ∧
    second_resident_statement (liars + 1) (truth_tellers + 1) ∧
    liars = 1000 ∧ truth_tellers = 1000 ∧
    first_resident_statement liars truth_tellers ∧ second_resident_statement liars truth_tellers :=
by
  sorry

end NUMINAMATH_GPT_island_solution_l624_62464


namespace NUMINAMATH_GPT_triangle_third_side_lengths_l624_62426

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_third_side_lengths :
  ∃ (n : ℕ), n = 15 ∧ ∀ x : ℕ, (3 < x ∧ x < 19) → (∃ k, x = k) :=
by
  sorry

end NUMINAMATH_GPT_triangle_third_side_lengths_l624_62426


namespace NUMINAMATH_GPT_find_max_z_plus_x_l624_62463

theorem find_max_z_plus_x : 
  (∃ (x y z t: ℝ), x^2 + y^2 = 4 ∧ z^2 + t^2 = 9 ∧ xt + yz ≥ 6 ∧ z + x = 5) :=
sorry

end NUMINAMATH_GPT_find_max_z_plus_x_l624_62463


namespace NUMINAMATH_GPT_fixed_point_of_function_l624_62482

theorem fixed_point_of_function :
  (4, 4) ∈ { p : ℝ × ℝ | ∃ x : ℝ, p = (x, 2^(x-4) + 3) } :=
by
  sorry

end NUMINAMATH_GPT_fixed_point_of_function_l624_62482


namespace NUMINAMATH_GPT_shopkeeper_loss_percent_l624_62439

noncomputable def loss_percentage (cost_price profit_percent theft_percent: ℝ) :=
  let selling_price := cost_price * (1 + profit_percent / 100)
  let value_lost := cost_price * (theft_percent / 100)
  let remaining_cost_price := cost_price * (1 - theft_percent / 100)
  (value_lost / remaining_cost_price) * 100

theorem shopkeeper_loss_percent
  (cost_price : ℝ)
  (profit_percent : ℝ := 10)
  (theft_percent : ℝ := 20)
  (expected_loss_percent : ℝ := 25)
  (h1 : profit_percent = 10) (h2 : theft_percent = 20) : 
  loss_percentage cost_price profit_percent theft_percent = expected_loss_percent := 
by
  sorry

end NUMINAMATH_GPT_shopkeeper_loss_percent_l624_62439


namespace NUMINAMATH_GPT_find_distance_l624_62494

variable (A B : Point)
variable (distAB : ℝ) -- the distance between A and B
variable (meeting1 : ℝ) -- first meeting distance from A
variable (meeting2 : ℝ) -- second meeting distance from B

-- Conditions
axiom meeting_conditions_1 : meeting1 = 70
axiom meeting_conditions_2 : meeting2 = 90

-- Prove the distance between A and B is 120 km
def distance_from_A_to_B : ℝ := 120

theorem find_distance : distAB = distance_from_A_to_B := 
sorry

end NUMINAMATH_GPT_find_distance_l624_62494


namespace NUMINAMATH_GPT_plane_equation_exists_l624_62432

noncomputable def equation_of_plane (A B C D : ℤ) (hA : A > 0) (hGCD : Int.gcd (Int.gcd A B) (Int.gcd C D) = 1) : Prop :=
∃ (x y z : ℤ),
  x = 1 ∧ y = -2 ∧ z = 2 ∧ D = -18 ∧
  (2 * x + (-3) * y + 5 * z + D = 0) ∧  -- Point (2, -3, 5) satisfies equation
  (4 * x + (-3) * y + 6 * z + D = 0) ∧  -- Point (4, -3, 6) satisfies equation
  (6 * x + (-4) * y + 8 * z + D = 0)    -- Point (6, -4, 8) satisfies equation

theorem plane_equation_exists : equation_of_plane 1 (-2) 2 (-18) (by decide) (by decide) :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_plane_equation_exists_l624_62432


namespace NUMINAMATH_GPT_x3_y3_sum_sq_sq_leq_4a10_equality_holds_when_x_eq_y_l624_62424

-- Conditions: x, y are positive real numbers and x + y = 2a
variables {x y a : ℝ}
variable (hxy : x + y = 2 * a)
variable (hx_pos : 0 < x)
variable (hy_pos : 0 < y)

-- Math proof problem: Prove the inequality
theorem x3_y3_sum_sq_sq_leq_4a10 : 
  x^3 * y^3 * (x^2 + y^2)^2 ≤ 4 * a^10 :=
by sorry

-- Equality condition: Equality holds when x = y
theorem equality_holds_when_x_eq_y (h : x = y) :
  x^3 * y^3 * (x^2 + y^2)^2 = 4 * a^10 :=
by sorry

end NUMINAMATH_GPT_x3_y3_sum_sq_sq_leq_4a10_equality_holds_when_x_eq_y_l624_62424


namespace NUMINAMATH_GPT_find_number_l624_62472

theorem find_number (x k : ℕ) (h1 : x / k = 4) (h2 : k = 16) : x = 64 := by
  sorry

end NUMINAMATH_GPT_find_number_l624_62472


namespace NUMINAMATH_GPT_number_of_outcomes_exactly_two_evening_l624_62409

theorem number_of_outcomes_exactly_two_evening (chickens : Finset ℕ) (h_chickens : chickens.card = 4) 
    (day_places evening_places : ℕ) (h_day_places : day_places = 2) (h_evening_places : evening_places = 3) :
    ∃ n, n = (chickens.card.choose 2) ∧ n = 6 :=
by
  sorry

end NUMINAMATH_GPT_number_of_outcomes_exactly_two_evening_l624_62409


namespace NUMINAMATH_GPT_total_distance_maria_l624_62446

theorem total_distance_maria (D : ℝ)
  (half_dist : D/2 + (D/2 - D/8) + 180 = D) :
  3 * D / 8 = 180 → 
  D = 480 :=
by
  sorry

end NUMINAMATH_GPT_total_distance_maria_l624_62446


namespace NUMINAMATH_GPT_average_weight_increase_l624_62440

theorem average_weight_increase (A : ℝ) :
  let initial_weight := 8 * A
  let new_weight := initial_weight - 65 + 89
  let new_average := new_weight / 8
  let increase := new_average - A
  increase = (89 - 65) / 8 := 
by 
  sorry

end NUMINAMATH_GPT_average_weight_increase_l624_62440


namespace NUMINAMATH_GPT_jesse_bananas_l624_62451

def number_of_bananas_shared (friends : ℕ) (bananas_per_friend : ℕ) : ℕ :=
  friends * bananas_per_friend

theorem jesse_bananas :
  number_of_bananas_shared 3 7 = 21 :=
by
  sorry

end NUMINAMATH_GPT_jesse_bananas_l624_62451


namespace NUMINAMATH_GPT_sum_of_exterior_angles_of_triangle_l624_62491

theorem sum_of_exterior_angles_of_triangle
  {α β γ α' β' γ' : ℝ} 
  (h1 : α + β + γ = 180)
  (h2 : α + α' = 180)
  (h3 : β + β' = 180)
  (h4 : γ + γ' = 180) :
  α' + β' + γ' = 360 := 
by 
sorry

end NUMINAMATH_GPT_sum_of_exterior_angles_of_triangle_l624_62491


namespace NUMINAMATH_GPT_find_B_l624_62485

theorem find_B (A B : ℕ) (h₁ : 6 * A + 10 * B + 2 = 77) (h₂ : A ≤ 9) (h₃ : B ≤ 9) : B = 1 := sorry

end NUMINAMATH_GPT_find_B_l624_62485


namespace NUMINAMATH_GPT_radius_of_inscribed_circle_l624_62470

variable (A p s r : ℝ)

theorem radius_of_inscribed_circle (h1 : A = 2 * p) (h2 : A = r * s) (h3 : p = 2 * s) : r = 4 := by
  sorry

end NUMINAMATH_GPT_radius_of_inscribed_circle_l624_62470


namespace NUMINAMATH_GPT_number_divisible_by_11_l624_62459

theorem number_divisible_by_11 (N Q : ℕ) (h1 : N = 11 * Q) (h2 : Q + N + 11 = 71) : N = 55 :=
by
  sorry

end NUMINAMATH_GPT_number_divisible_by_11_l624_62459


namespace NUMINAMATH_GPT_quadratic_passing_point_calc_l624_62489

theorem quadratic_passing_point_calc :
  (∀ (x y : ℤ), y = 2 * x ^ 2 - 3 * x + 4 → ∃ (x' y' : ℤ), x' = 2 ∧ y' = 6) →
  (2 * 2 - 3 * (-3) + 4 * 4 = 29) :=
by
  intro h
  -- The corresponding proof would follow by providing the necessary steps.
  -- For now, let's just use sorry to meet the requirement.
  sorry

end NUMINAMATH_GPT_quadratic_passing_point_calc_l624_62489


namespace NUMINAMATH_GPT_bug_twelfth_move_l624_62433

theorem bug_twelfth_move (Q : ℕ → ℚ)
  (hQ0 : Q 0 = 1)
  (hQ1 : Q 1 = 0)
  (hQ2 : Q 2 = 1/2)
  (h_recursive : ∀ n, Q (n + 1) = 1/2 * (1 - Q n)) :
  let m := 683
  let n := 2048
  (Nat.gcd m n = 1) ∧ (m + n = 2731) :=
by
  sorry

end NUMINAMATH_GPT_bug_twelfth_move_l624_62433


namespace NUMINAMATH_GPT_proof_q1_a1_proof_q2_a2_proof_q3_a3_proof_q4_a4_l624_62465

variables (G : Type) [Group G] (kidney testis liver : G)
variables (SudanIII gentianViolet JanusGreenB dissociationFixative : G)

-- Conditions c1, c2, c3
def c1 : Prop := True -- Meiosis occurs in gonads, we simplify this in Lean to a true condition for brevity
def c2 : Prop := True -- Steps for slide preparation
def c3 : Prop := True -- Materials available

-- Questions
def q1 : G := testis
def q2 : G := dissociationFixative
def q3 : G := gentianViolet
def q4 : List G := [kidney, dissociationFixative, gentianViolet] -- Assume these are placeholders for correct cell types

-- Answers
def a1 : G := testis
def a2 : G := dissociationFixative
def a3 : G := gentianViolet
def a4 : List G := [testis, dissociationFixative, gentianViolet] -- Correct cells

-- Proving the equivalence of questions and answers given the conditions
theorem proof_q1_a1 : c1 ∧ c2 ∧ c3 → q1 = a1 := 
by sorry

theorem proof_q2_a2 : c1 ∧ c2 ∧ c3 → q2 = a2 := 
by sorry

theorem proof_q3_a3 : c1 ∧ c2 ∧ c3 → q3 = a3 := 
by sorry

theorem proof_q4_a4 : c1 ∧ c2 ∧ c3 → q4 = a4 := 
by sorry

end NUMINAMATH_GPT_proof_q1_a1_proof_q2_a2_proof_q3_a3_proof_q4_a4_l624_62465


namespace NUMINAMATH_GPT_luke_base_points_per_round_l624_62460

theorem luke_base_points_per_round
    (total_score : ℕ)
    (rounds : ℕ)
    (bonus : ℕ)
    (penalty : ℕ)
    (adjusted_total : ℕ) :
    total_score = 370 → rounds = 5 → bonus = 50 → penalty = 30 → adjusted_total = total_score + bonus - penalty → (adjusted_total / rounds) = 78 :=
by
  intros
  sorry

end NUMINAMATH_GPT_luke_base_points_per_round_l624_62460


namespace NUMINAMATH_GPT_weight_cut_percentage_unknown_l624_62498

-- Define the initial conditions
def original_speed : ℝ := 150
def new_speed : ℝ := 205
def increase_supercharge : ℝ := original_speed * 0.3
def speed_after_supercharge : ℝ := original_speed + increase_supercharge
def increase_weight_cut : ℝ := new_speed - speed_after_supercharge

-- Theorem statement
theorem weight_cut_percentage_unknown : 
  (original_speed = 150) →
  (new_speed = 205) →
  (increase_supercharge = 150 * 0.3) →
  (speed_after_supercharge = 150 + increase_supercharge) →
  (increase_weight_cut = 205 - speed_after_supercharge) →
  increase_weight_cut = 10 →
  sorry := 
by
  intros h_orig h_new h_inc_scharge h_speed_scharge h_inc_weight h_inc_10
  sorry

end NUMINAMATH_GPT_weight_cut_percentage_unknown_l624_62498


namespace NUMINAMATH_GPT_three_pow_sub_two_pow_prime_power_prime_l624_62436

theorem three_pow_sub_two_pow_prime_power_prime (n : ℕ) (hn : n > 0) (hp : ∃ p k : ℕ, Nat.Prime p ∧ 3^n - 2^n = p^k) : Nat.Prime n := 
sorry

end NUMINAMATH_GPT_three_pow_sub_two_pow_prime_power_prime_l624_62436


namespace NUMINAMATH_GPT_geometric_sequence_sixth_term_correct_l624_62481

noncomputable def geometric_sequence_sixth_term (a r : ℝ) (pos_a : 0 < a) (pos_r : 0 < r)
    (third_term : a * r^2 = 27)
    (ninth_term : a * r^8 = 3) : ℝ :=
  a * r^5

theorem geometric_sequence_sixth_term_correct (a r : ℝ) (pos_a : 0 < a) (pos_r : 0 < r) 
    (third_term : a * r^2 = 27)
    (ninth_term : a * r^8 = 3) : geometric_sequence_sixth_term a r pos_a pos_r third_term ninth_term = 9 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_sixth_term_correct_l624_62481


namespace NUMINAMATH_GPT_min_even_integers_zero_l624_62483

theorem min_even_integers_zero (x y a b m n : ℤ)
(h1 : x + y = 28) 
(h2 : x + y + a + b = 46) 
(h3 : x + y + a + b + m + n = 64) : 
∃ e, e = 0 :=
by {
  -- The conditions assure the sums of pairs are even including x, y, a, b, m, n.
  sorry
}

end NUMINAMATH_GPT_min_even_integers_zero_l624_62483


namespace NUMINAMATH_GPT_pears_remaining_l624_62410

theorem pears_remaining (K_picked : ℕ) (M_picked : ℕ) (S_picked : ℕ)
                        (K_gave : ℕ) (M_gave : ℕ) (S_gave : ℕ)
                        (hK_pick : K_picked = 47)
                        (hM_pick : M_picked = 12)
                        (hS_pick : S_picked = 22)
                        (hK_give : K_gave = 46)
                        (hM_give : M_gave = 5)
                        (hS_give : S_gave = 15) :
  (K_picked - K_gave) + (M_picked - M_gave) + (S_picked - S_gave) = 15 :=
by
  sorry

end NUMINAMATH_GPT_pears_remaining_l624_62410


namespace NUMINAMATH_GPT_possible_items_l624_62405

-- Mathematical definitions derived from the conditions.
def item_cost_kopecks (a : ℕ) : ℕ := 100 * a + 99
def total_cost_kopecks : ℕ := 20083

-- The theorem stating the possible number of items Kolya could have bought.
theorem possible_items (a n : ℕ) (hn : n * item_cost_kopecks a = total_cost_kopecks) :
  n = 17 ∨ n = 117 :=
sorry

end NUMINAMATH_GPT_possible_items_l624_62405


namespace NUMINAMATH_GPT_ruth_hours_per_week_l624_62425

theorem ruth_hours_per_week :
  let daily_hours := 8
  let days_per_week := 5
  let monday_wednesday_friday := 3
  let tuesday_thursday := 2
  let percentage_to_hours (percent : ℝ) (hours : ℕ) : ℝ := percent * hours
  let total_weekly_hours := daily_hours * days_per_week
  let monday_wednesday_friday_math_hours := percentage_to_hours 0.25 daily_hours
  let monday_wednesday_friday_science_hours := percentage_to_hours 0.15 daily_hours
  let tuesday_thursday_math_hours := percentage_to_hours 0.2 daily_hours
  let tuesday_thursday_science_hours := percentage_to_hours 0.35 daily_hours
  let tuesday_thursday_history_hours := percentage_to_hours 0.15 daily_hours
  let weekly_math_hours := monday_wednesday_friday_math_hours * monday_wednesday_friday + tuesday_thursday_math_hours * tuesday_thursday
  let weekly_science_hours := monday_wednesday_friday_science_hours * monday_wednesday_friday + tuesday_thursday_science_hours * tuesday_thursday
  let weekly_history_hours := tuesday_thursday_history_hours * tuesday_thursday
  let total_hours := weekly_math_hours + weekly_science_hours + weekly_history_hours
  total_hours = 20.8 := by
  sorry

end NUMINAMATH_GPT_ruth_hours_per_week_l624_62425


namespace NUMINAMATH_GPT_find_integer_pairs_l624_62480

theorem find_integer_pairs :
  ∀ x y : ℤ, x^2 = 2 + 6 * y^2 + y^4 ↔ (x = 3 ∧ y = 1) ∨ (x = -3 ∧ y = 1) ∨ (x = 3 ∧ y = -1) ∨ (x = -3 ∧ y = -1) :=
by {
  sorry
}

end NUMINAMATH_GPT_find_integer_pairs_l624_62480


namespace NUMINAMATH_GPT_stationery_box_cost_l624_62403

theorem stationery_box_cost (unit_price : ℕ) (quantity : ℕ) (total_cost : ℕ) :
  unit_price = 23 ∧ quantity = 3 ∧ total_cost = 3 * 23 → total_cost = 69 :=
by
  sorry

end NUMINAMATH_GPT_stationery_box_cost_l624_62403


namespace NUMINAMATH_GPT_repeating_decimal_product_as_fraction_l624_62449

theorem repeating_decimal_product_as_fraction :
  let x := 37 / 999
  let y := 7 / 9
  x * y = 259 / 8991 := by {
    sorry
  }

end NUMINAMATH_GPT_repeating_decimal_product_as_fraction_l624_62449


namespace NUMINAMATH_GPT_total_paintable_area_correct_l624_62484

-- Bedroom dimensions and unoccupied wall space
def bedroom1_length : ℕ := 14
def bedroom1_width : ℕ := 12
def bedroom1_height : ℕ := 9
def bedroom1_unoccupied : ℕ := 70

def bedroom2_length : ℕ := 12
def bedroom2_width : ℕ := 11
def bedroom2_height : ℕ := 9
def bedroom2_unoccupied : ℕ := 65

def bedroom3_length : ℕ := 13
def bedroom3_width : ℕ := 12
def bedroom3_height : ℕ := 9
def bedroom3_unoccupied : ℕ := 68

-- Total paintable area calculation
def calculate_paintable_area (length width height unoccupied : ℕ) : ℕ :=
  2 * (length * height + width * height) - unoccupied

-- Total paintable area of all bedrooms
def total_paintable_area : ℕ :=
  calculate_paintable_area bedroom1_length bedroom1_width bedroom1_height bedroom1_unoccupied +
  calculate_paintable_area bedroom2_length bedroom2_width bedroom2_height bedroom2_unoccupied +
  calculate_paintable_area bedroom3_length bedroom3_width bedroom3_height bedroom3_unoccupied

theorem total_paintable_area_correct : 
  total_paintable_area = 1129 :=
by
  unfold total_paintable_area
  unfold calculate_paintable_area
  norm_num
  sorry

end NUMINAMATH_GPT_total_paintable_area_correct_l624_62484


namespace NUMINAMATH_GPT_unique_real_solution_system_l624_62413

/-- There is exactly one real solution (x, y, z, w) to the given system of equations:
  x + 1 = z + w + z * w * x,
  y - 1 = w + x + w * x * y,
  z + 2 = x + y + x * y * z,
  w - 2 = y + z + y * z * w
-/
theorem unique_real_solution_system :
  let eq1 (x y z w : ℝ) := x + 1 = z + w + z * w * x
  let eq2 (x y z w : ℝ) := y - 1 = w + x + w * x * y
  let eq3 (x y z w : ℝ) := z + 2 = x + y + x * y * z
  let eq4 (x y z w : ℝ) := w - 2 = y + z + y * z * w
  ∃! (x y z w : ℝ), eq1 x y z w ∧ eq2 x y z w ∧ eq3 x y z w ∧ eq4 x y z w := by {
  sorry
}

end NUMINAMATH_GPT_unique_real_solution_system_l624_62413


namespace NUMINAMATH_GPT_cos_half_alpha_l624_62490

open Real -- open the Real namespace for convenience

theorem cos_half_alpha {α : ℝ} (h1 : cos α = 1 / 5) (h2 : 0 < α ∧ α < π) :
  cos (α / 2) = sqrt (15) / 5 :=
by
  sorry -- Proof is omitted

end NUMINAMATH_GPT_cos_half_alpha_l624_62490


namespace NUMINAMATH_GPT_part1_part2_l624_62442

-- Definition of the quadratic equation and its real roots condition
def quadratic_has_real_roots (k : ℝ) : Prop :=
  let Δ := (2 * k - 1)^2 - 4 * (k^2 - 1)
  Δ ≥ 0

-- Proving part (1): The range of real number k
theorem part1 (k : ℝ) (hk : quadratic_has_real_roots k) : k ≤ 5 / 4 := 
  sorry

-- Definition using the given condition in part (2)
def roots_condition (x₁ x₂ : ℝ) : Prop :=
  x₁^2 + x₂^2 = 16 + x₁ * x₂

-- Sum and product of roots of the quadratic equation
theorem part2 (k : ℝ) (h : quadratic_has_real_roots k) 
  (hx_sum : ∃ x₁ x₂ : ℝ, x₁ + x₂ = 1 - 2 * k ∧ x₁ * x₂ = k^2 - 1 ∧ roots_condition x₁ x₂) : k = -2 :=
  sorry

end NUMINAMATH_GPT_part1_part2_l624_62442


namespace NUMINAMATH_GPT_hank_route_distance_l624_62450

theorem hank_route_distance 
  (d : ℝ) 
  (h1 : ∃ t1 : ℝ, t1 = d / 70 ∧ t1 = d / 70 + 1 / 60) 
  (h2 : ∃ t2 : ℝ, t2 = d / 75 ∧ t2 = d / 75 - 1 / 60) 
  (time_diff : (d / 70 - d / 75) = 1 / 30) : 
  d = 35 :=
sorry

end NUMINAMATH_GPT_hank_route_distance_l624_62450
