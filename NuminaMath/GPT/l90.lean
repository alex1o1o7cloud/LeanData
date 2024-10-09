import Mathlib

namespace tourists_went_free_l90_9029

theorem tourists_went_free (x : ℕ) : 
  (13 + 4 * x = x + 100) → x = 29 :=
by
  intros h
  sorry

end tourists_went_free_l90_9029


namespace Jeanine_more_pencils_than_Clare_l90_9038

variables (Jeanine_pencils : ℕ) (Clare_pencils : ℕ)

def Jeanine_initial_pencils := 18
def Clare_initial_pencils := Jeanine_initial_pencils / 2
def Jeanine_pencils_given_to_Abby := Jeanine_initial_pencils / 3
def Jeanine_remaining_pencils := Jeanine_initial_pencils - Jeanine_pencils_given_to_Abby

theorem Jeanine_more_pencils_than_Clare :
  Jeanine_remaining_pencils - Clare_initial_pencils = 3 :=
by
  -- This is just the statement, the proof is not provided as instructed.
  sorry

end Jeanine_more_pencils_than_Clare_l90_9038


namespace factorization_of_a_cubed_minus_a_l90_9090

variable (a : ℝ)

theorem factorization_of_a_cubed_minus_a : a^3 - a = a * (a + 1) * (a - 1) := by
  sorry

end factorization_of_a_cubed_minus_a_l90_9090


namespace system_solution_l90_9057

theorem system_solution (u v w : ℚ) 
  (h1 : 3 * u - 4 * v + w = 26)
  (h2 : 6 * u + 5 * v - 2 * w = -17) :
  u + v + w = 101 / 3 :=
sorry

end system_solution_l90_9057


namespace wechat_group_member_count_l90_9093

theorem wechat_group_member_count :
  (∃ x : ℕ, x * (x - 1) / 2 = 72) → ∃ x : ℕ, x = 9 :=
by
  sorry

end wechat_group_member_count_l90_9093


namespace brooke_sidney_ratio_l90_9031

-- Definitions for the conditions
def sidney_monday : ℕ := 20
def sidney_tuesday : ℕ := 36
def sidney_wednesday : ℕ := 40
def sidney_thursday : ℕ := 50
def brooke_total : ℕ := 438

-- Total jumping jacks by Sidney
def sidney_total : ℕ := sidney_monday + sidney_tuesday + sidney_wednesday + sidney_thursday

-- The ratio of Brooke’s jumping jacks to Sidney's total jumping jacks
def ratio := brooke_total / sidney_total

-- The proof goal
theorem brooke_sidney_ratio : ratio = 3 :=
by
  sorry

end brooke_sidney_ratio_l90_9031


namespace alice_has_winning_strategy_l90_9023

def alice_has_winning_strategy_condition (nums : List ℤ) : Prop :=
  nums.length = 17 ∧ ∀ x ∈ nums, ¬ (x % 17 = 0)

theorem alice_has_winning_strategy (nums : List ℤ) (H : alice_has_winning_strategy_condition nums) : ∃ (f : List ℤ → List ℤ), ∀ k, (f^[k] nums).sum % 17 = 0 :=
sorry

end alice_has_winning_strategy_l90_9023


namespace formula1_correct_formula2_correct_formula3_correct_l90_9002

noncomputable def formula1 (n : ℕ) := (Real.sqrt 2 / 2) * (1 - (-1 : ℝ) ^ n)
noncomputable def formula2 (n : ℕ) := Real.sqrt (1 - (-1 : ℝ) ^ n)
noncomputable def formula3 (n : ℕ) := if (n % 2 = 1) then Real.sqrt 2 else 0

theorem formula1_correct (n : ℕ) : 
  (n % 2 = 1 → formula1 n = Real.sqrt 2) ∧ 
  (n % 2 = 0 → formula1 n = 0) := 
by
  sorry

theorem formula2_correct (n : ℕ) : 
  (n % 2 = 1 → formula2 n = Real.sqrt 2) ∧ 
  (n % 2 = 0 → formula2 n = 0) := 
by
  sorry
  
theorem formula3_correct (n : ℕ) : 
  (n % 2 = 1 → formula3 n = Real.sqrt 2) ∧ 
  (n % 2 = 0 → formula3 n = 0) := 
by
  sorry

end formula1_correct_formula2_correct_formula3_correct_l90_9002


namespace carrie_profit_l90_9013

def total_hours_worked (hours_per_day: ℕ) (days: ℕ): ℕ := hours_per_day * days
def total_earnings (hours_worked: ℕ) (hourly_wage: ℕ): ℕ := hours_worked * hourly_wage
def profit (total_earnings: ℕ) (cost_of_supplies: ℕ): ℕ := total_earnings - cost_of_supplies

theorem carrie_profit (hours_per_day: ℕ) (days: ℕ) (hourly_wage: ℕ) (cost_of_supplies: ℕ): 
    hours_per_day = 2 → days = 4 → hourly_wage = 22 → cost_of_supplies = 54 → 
    profit (total_earnings (total_hours_worked hours_per_day days) hourly_wage) cost_of_supplies = 122 := 
by
    intros hpd d hw cos
    sorry

end carrie_profit_l90_9013


namespace sum_of_roots_l90_9000

theorem sum_of_roots (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hroots : ∀ x : ℝ, x^2 - p*x + 2*q = 0) :
  p + q = p :=
by sorry

end sum_of_roots_l90_9000


namespace ratio_of_octagon_areas_l90_9065

-- Define the relevant terms and conditions
noncomputable def radius_of_circle (r : ℝ) := r
noncomputable def side_length_inscribed_octagon (r : ℝ) := r * Real.sqrt 2
noncomputable def side_length_circumscribed_octagon (r : ℝ) := r * Real.sqrt 2

-- Area of a regular octagon in terms of its side length
noncomputable def area_regular_octagon (s : ℝ) := 2 * (1 + Real.sqrt 2) * s^2

-- Areas of inscribed and circumscribed octagons
noncomputable def area_inscribed_octagon (r : ℝ) := area_regular_octagon (side_length_inscribed_octagon r)
noncomputable def area_circumscribed_octagon (r : ℝ) := area_regular_octagon (side_length_circumscribed_octagon r)

-- The main statement to prove
theorem ratio_of_octagon_areas (r : ℝ) : 
  area_circumscribed_octagon r / area_inscribed_octagon r = 1 :=
by
  sorry

end ratio_of_octagon_areas_l90_9065


namespace ways_to_divide_friends_l90_9001

theorem ways_to_divide_friends : (4 ^ 8 = 65536) := by
  sorry

end ways_to_divide_friends_l90_9001


namespace chairs_to_remove_is_33_l90_9085

-- Definitions for the conditions
def chairs_per_row : ℕ := 11
def total_chairs : ℕ := 110
def students : ℕ := 70

-- Required statement
theorem chairs_to_remove_is_33 
  (h_divisible_by_chairs_per_row : ∀ n, n = total_chairs - students → ∃ k, n = chairs_per_row * k) :
  ∃ rem_chairs : ℕ, rem_chairs = total_chairs - 77 ∧ rem_chairs = 33 := sorry

end chairs_to_remove_is_33_l90_9085


namespace proof_problem_l90_9058

variables {x1 y1 x2 y2 : ℝ}

-- Definitions
def unit_vector (x y : ℝ) : Prop := x^2 + y^2 = 1
def angle_with_p (x y : ℝ) : Prop := (x + y) / Real.sqrt 2 = Real.sqrt 3 / 2
def m := (x1, y1)
def n := (x2, y2)
def p := (1, 1)

-- Conditions
lemma unit_m : unit_vector x1 y1 := sorry
lemma unit_n : unit_vector x2 y2 := sorry
lemma angle_m_p : angle_with_p x1 y1 := sorry
lemma angle_n_p : angle_with_p x2 y2 := sorry

-- Theorem to prove
theorem proof_problem (h1 : unit_vector x1 y1)
                      (h2 : unit_vector x2 y2)
                      (h3 : angle_with_p x1 y1)
                      (h4 : angle_with_p x2 y2) :
                      (x1 * x2 + y1 * y2 = 1/2) ∧ (y1 * y2 / (x1 * x2) = 1) :=
sorry

end proof_problem_l90_9058


namespace ana_wins_probability_l90_9012

noncomputable def probability_ana_wins : ℚ := 
  let a := (1 / 2)^5
  let r := (1 / 2)^4
  a / (1 - r)

theorem ana_wins_probability :
  probability_ana_wins = 1 / 30 :=
by
  sorry

end ana_wins_probability_l90_9012


namespace age_difference_l90_9078

theorem age_difference (A B : ℕ) (h1 : B = 37) (h2 : A + 10 = 2 * (B - 10)) : A - B = 7 :=
by
  sorry

end age_difference_l90_9078


namespace train_speed_correct_l90_9050

noncomputable def train_speed_kmh (length : ℝ) (time : ℝ) (conversion_factor : ℝ) : ℝ :=
  (length / time) * conversion_factor

theorem train_speed_correct 
  (length : ℝ := 350) 
  (time : ℝ := 8.7493) 
  (conversion_factor : ℝ := 3.6) : 
  train_speed_kmh length time conversion_factor = 144.02 := 
sorry

end train_speed_correct_l90_9050


namespace smallest_value_x_abs_eq_32_l90_9033

theorem smallest_value_x_abs_eq_32 : ∃ x : ℚ, (x = -29 / 5) ∧ (|5 * x - 3| = 32) ∧ 
  (∀ y : ℚ, (|5 * y - 3| = 32) → (x ≤ y)) :=
by
  sorry

end smallest_value_x_abs_eq_32_l90_9033


namespace find_N_l90_9018

theorem find_N (N : ℕ) (hN : N > 1) (h1 : 2019 ≡ 1743 [MOD N]) (h2 : 3008 ≡ 2019 [MOD N]) : N = 23 :=
by
  sorry

end find_N_l90_9018


namespace twice_product_of_numbers_l90_9039

theorem twice_product_of_numbers (x y : ℝ) (h1 : x + y = 80) (h2 : x - y = 10) : 2 * (x * y) = 3150 := by
  sorry

end twice_product_of_numbers_l90_9039


namespace Jordan_income_l90_9008

theorem Jordan_income (q A : ℝ) (h : A > 30000)
  (h1 : (q / 100 * 30000 + (q + 3) / 100 * (A - 30000) - 600) = (q + 0.5) / 100 * A) :
  A = 60000 :=
by
  sorry

end Jordan_income_l90_9008


namespace parallel_lines_solution_l90_9026

theorem parallel_lines_solution (m : ℝ) :
  (∀ x y : ℝ, (x + (1 + m) * y + (m - 2) = 0) → (m * x + 2 * y + 8 = 0)) → m = 1 :=
by
  sorry

end parallel_lines_solution_l90_9026


namespace quadratic_binomial_form_l90_9037

theorem quadratic_binomial_form (y : ℝ) : ∃ (k : ℝ), y^2 + 14 * y + 40 = (y + 7)^2 + k :=
by
  use -9
  sorry

end quadratic_binomial_form_l90_9037


namespace composite_has_at_least_three_factors_l90_9080

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d : ℕ, d ∣ n ∧ d ≠ 1 ∧ d ≠ n

theorem composite_has_at_least_three_factors (n : ℕ) (h : is_composite n) : ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a ∣ n ∧ b ∣ n ∧ c ∣ n :=
sorry

end composite_has_at_least_three_factors_l90_9080


namespace monotonic_increasing_interval_l90_9092

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)

theorem monotonic_increasing_interval :
  ∃ a b : ℝ, a < b ∧
    ∀ x y : ℝ, (a < x ∧ x < b) → (a < y ∧ y < b) → x < y → f x < f y ∧ a = -Real.pi / 6 ∧ b = Real.pi / 3 :=
by
  sorry

end monotonic_increasing_interval_l90_9092


namespace ads_ratio_l90_9059

theorem ads_ratio 
  (first_ads : ℕ := 12)
  (second_ads : ℕ)
  (third_ads := second_ads + 24)
  (fourth_ads := (3 / 4) * second_ads)
  (clicked_ads := 68)
  (total_ads := (3 / 2) * clicked_ads == 102)
  (ads_eq : first_ads + second_ads + third_ads + fourth_ads = total_ads) :
  second_ads / first_ads = 2 :=
by sorry

end ads_ratio_l90_9059


namespace shortest_altitude_triangle_l90_9051

/-- Given a triangle with sides 18, 24, and 30, prove that its shortest altitude is 18. -/
theorem shortest_altitude_triangle (a b c : ℝ) (h1 : a = 18) (h2 : b = 24) (h3 : c = 30) 
  (h_right : a ^ 2 + b ^ 2 = c ^ 2) : 
  exists h : ℝ, h = 18 :=
by
  sorry

end shortest_altitude_triangle_l90_9051


namespace alyssa_games_last_year_l90_9076

theorem alyssa_games_last_year (games_this_year games_next_year games_total games_last_year : ℕ) (h1 : games_this_year = 11) (h2 : games_next_year = 15) (h3 : games_total = 39) (h4 : games_last_year + games_this_year + games_next_year = games_total) : games_last_year = 13 :=
by
  rw [h1, h2, h3] at h4
  sorry

end alyssa_games_last_year_l90_9076


namespace distributor_profit_percentage_l90_9044

theorem distributor_profit_percentage 
    (commission_rate : ℝ) (cost_price : ℝ) (final_price : ℝ) (P : ℝ) (profit : ℝ) 
    (profit_percentage: ℝ) :
  commission_rate = 0.20 →
  cost_price = 15 →
  final_price = 19.8 →
  0.80 * P = final_price →
  P = cost_price + profit →
  profit_percentage = (profit / cost_price) * 100 →
  profit_percentage = 65 :=
by
  intros h_commission_rate h_cost_price h_final_price h_equation h_profit_eq h_percent_eq
  sorry

end distributor_profit_percentage_l90_9044


namespace exp_calculation_l90_9073

theorem exp_calculation : 0.125^8 * (-8)^7 = -0.125 :=
by
  -- conditions used directly in proof
  have h1 : 0.125 = 1 / 8 := sorry
  have h2 : (-1)^7 = -1 := sorry
  -- the problem statement
  sorry

end exp_calculation_l90_9073


namespace largest_bucket_capacity_l90_9064

-- Let us define the initial conditions
def capacity_5_liter_bucket : ℕ := 5
def capacity_3_liter_bucket : ℕ := 3
def remaining_after_pour := capacity_5_liter_bucket - capacity_3_liter_bucket
def additional_capacity_without_overflow : ℕ := 4

-- Problem statement: Prove that the capacity of the largest bucket is 6 liters
theorem largest_bucket_capacity : ∀ (c : ℕ), remaining_after_pour + additional_capacity_without_overflow = c → c = 6 := 
by
  sorry

end largest_bucket_capacity_l90_9064


namespace Joey_weekend_study_hours_l90_9081

noncomputable def hours_weekday_per_week := 2 * 5 -- 2 hours/night * 5 nights/week
noncomputable def total_hours_weekdays := hours_weekday_per_week * 6 -- Multiply by 6 weeks
noncomputable def remaining_hours_weekends := 96 - total_hours_weekdays -- 96 total hours - weekday hours
noncomputable def total_weekend_days := 6 * 2 -- 6 weekends * 2 days/weekend
noncomputable def hours_per_day_weekend := remaining_hours_weekends / total_weekend_days

theorem Joey_weekend_study_hours : hours_per_day_weekend = 3 :=
by
  sorry

end Joey_weekend_study_hours_l90_9081


namespace episodes_count_l90_9094

variable (minutes_per_episode : ℕ) (total_watching_time_minutes : ℕ)
variable (episodes_watched : ℕ)

theorem episodes_count 
  (h1 : minutes_per_episode = 50) 
  (h2 : total_watching_time_minutes = 300) 
  (h3 : total_watching_time_minutes / minutes_per_episode = episodes_watched) :
  episodes_watched = 6 := sorry

end episodes_count_l90_9094


namespace total_weight_of_nuts_l90_9007

theorem total_weight_of_nuts (weight_almonds weight_pecans : ℝ) (h1 : weight_almonds = 0.14) (h2 : weight_pecans = 0.38) : weight_almonds + weight_pecans = 0.52 :=
by
  sorry

end total_weight_of_nuts_l90_9007


namespace bottles_count_l90_9083

-- Defining the conditions from the problem statement
def condition1 (x y : ℕ) : Prop := 3 * x + 4 * y = 108
def condition2 (x y : ℕ) : Prop := 2 * x + 3 * y = 76

-- The proof statement combining conditions and the solution
theorem bottles_count (x y : ℕ) (h1 : condition1 x y) (h2 : condition2 x y) : x = 20 ∧ y = 12 :=
sorry

end bottles_count_l90_9083


namespace inequality_holds_l90_9046

theorem inequality_holds (k n : ℕ) (x : ℝ) (hx1 : 0 ≤ x) (hx2 : x ≤ 1) :
  (1 - (1 - x)^n)^k ≥ 1 - (1 - x^k)^n :=
by
  sorry

end inequality_holds_l90_9046


namespace person_B_age_l90_9088

variables (a b c d e f g : ℕ)

-- Conditions
axiom cond1 : a = b + 2
axiom cond2 : b = 2 * c
axiom cond3 : c = d / 2
axiom cond4 : d = e - 3
axiom cond5 : f = a * d
axiom cond6 : g = b + e
axiom cond7 : a + b + c + d + e + f + g = 292

-- Theorem statement
theorem person_B_age : b = 14 :=
sorry

end person_B_age_l90_9088


namespace example_problem_l90_9024

theorem example_problem (a b : ℕ) : a = 1 → a * (a + b) + 1 ∣ (a + b) * (b + 1) - 1 :=
by
  sorry

end example_problem_l90_9024


namespace intersection_correct_l90_9054

def M : Set Int := {-1, 1, 3, 5}
def N : Set Int := {-3, 1, 5}

theorem intersection_correct : M ∩ N = {1, 5} := 
by 
    sorry

end intersection_correct_l90_9054


namespace factory_sample_size_l90_9022

noncomputable def sample_size (A B C : ℕ) (sample_A : ℕ) : ℕ :=
  let total_ratio := A + B + C
  let ratio_A := A / total_ratio
  sample_A / ratio_A

theorem factory_sample_size
  (A B C : ℕ) (h_ratio : A = 2 ∧ B = 3 ∧ C = 5)
  (sample_A : ℕ) (h_sample_A : sample_A = 16) :
  sample_size A B C sample_A = 80 :=
by
  simp [h_ratio, h_sample_A, sample_size]
  sorry

end factory_sample_size_l90_9022


namespace quadratic_solution_exists_l90_9072

-- Define the conditions
variables (a b : ℝ) (h₀ : a ≠ 0)
-- The condition that the first quadratic equation has at most one solution
def has_at_most_one_solution (a b : ℝ) : Prop :=
  b^2 + 4*a*(a - 3) <= 0

-- The second quadratic equation
def second_equation (a b x : ℝ) : ℝ :=
  (b - 3) * x^2 + (a - 2 * b) * x + 3 * a + 3
  
-- The proof problem invariant in Lean 4
theorem quadratic_solution_exists (h₁ : has_at_most_one_solution a b) :
  ∃ x : ℝ, second_equation a b x = 0 :=
by
  sorry

end quadratic_solution_exists_l90_9072


namespace distance_focus_parabola_to_line_l90_9014

theorem distance_focus_parabola_to_line :
  let focus : ℝ × ℝ := (1, 0)
  let distance (p : ℝ × ℝ) (A B C : ℝ) : ℝ := |A * p.1 + B * p.2 + C| / Real.sqrt (A^2 + B^2)
  distance focus 1 (-Real.sqrt 3) 0 = 1 / 2 :=
by
  sorry

end distance_focus_parabola_to_line_l90_9014


namespace probability_three_digit_divisible_by_5_with_ones_digit_9_l90_9049

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def ones_digit (n : ℕ) : ℕ := n % 10

def is_divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

theorem probability_three_digit_divisible_by_5_with_ones_digit_9 : 
  ∀ (M : ℕ), is_three_digit M → ones_digit M = 9 → ¬ is_divisible_by_5 M := by
  intros M h1 h2
  sorry

end probability_three_digit_divisible_by_5_with_ones_digit_9_l90_9049


namespace roots_of_quadratic_eq_l90_9060

theorem roots_of_quadratic_eq (x : ℝ) : (x + 1) ^ 2 = 0 → x = -1 := by
  sorry

end roots_of_quadratic_eq_l90_9060


namespace line_perpendicular_through_P_l90_9062

/-
  Given:
  1. The point P(-2, 2).
  2. The line 2x - y + 1 = 0.
  Prove:
  The equation of the line that passes through P and is perpendicular to the given line is x + 2y - 2 = 0.
-/

def P : ℝ × ℝ := (-2, 2)
def line1 (x y : ℝ) : Prop := 2 * x - y + 1 = 0

theorem line_perpendicular_through_P :
  ∃ (x y : ℝ) (m : ℝ), (x = -2) ∧ (y = 2) ∧ (m = -1/2) ∧ 
  (∀ (x₁ y₁ : ℝ), (y₁ - y) = m * (x₁ - x)) ∧ 
  (∀ (lx ly : ℝ), line1 lx ly → x + 2 * y - 2 = 0) := sorry

end line_perpendicular_through_P_l90_9062


namespace airline_num_airplanes_l90_9030

-- Definitions based on the conditions
def rows_per_airplane : ℕ := 20
def seats_per_row : ℕ := 7
def flights_per_day_per_airplane : ℕ := 2
def total_passengers_per_day : ℕ := 1400

-- The theorem to prove the number of airplanes owned by the company
theorem airline_num_airplanes : 
  (total_passengers_per_day = 
   rows_per_airplane * seats_per_row * flights_per_day_per_airplane * n) → 
  n = 5 := 
by 
  sorry

end airline_num_airplanes_l90_9030


namespace bells_toll_together_l90_9055

theorem bells_toll_together (a b c d : ℕ) (h1 : a = 5) (h2 : b = 8) (h3 : c = 11) (h4 : d = 15) : 
  Nat.lcm a (Nat.lcm b (Nat.lcm c d)) = 1320 :=
by
  rw [h1, h2, h3, h4]
  sorry

end bells_toll_together_l90_9055


namespace annual_interest_rate_l90_9086

noncomputable def compound_interest_rate (P A : ℝ) (n t : ℕ) : ℝ :=
  ((A / P) ^ (1 / t)) - 1

-- Define the given parameters
def P : ℝ := 1200
def A : ℝ := 2488.32
def n : ℕ := 1
def t : ℕ := 4

theorem annual_interest_rate : compound_interest_rate P A n t = 0.25 :=
by
  sorry

end annual_interest_rate_l90_9086


namespace ratio_SP_CP_l90_9099

variables (CP SP P : ℝ)
axiom ratio_profit_CP : P / CP = 2

theorem ratio_SP_CP : SP / CP = 3 :=
by
  -- Proof statement (not required as per the instruction)
  sorry

end ratio_SP_CP_l90_9099


namespace sum_d_e_f_l90_9053

-- Define the variables
variables (d e f : ℤ)

-- Given conditions
def condition1 : Prop := ∀ x : ℤ, x^2 + 18 * x + 77 = (x + d) * (x + e)
def condition2 : Prop := ∀ x : ℤ, x^2 - 19 * x + 88 = (x - e) * (x - f)

-- Prove the statement
theorem sum_d_e_f : condition1 d e → condition2 e f → d + e + f = 26 :=
by
  intros h1 h2
  -- Proof omitted
  sorry

end sum_d_e_f_l90_9053


namespace area_of_EFCD_l90_9056

theorem area_of_EFCD (AB CD h : ℝ) (H_AB : AB = 10) (H_CD : CD = 30) (H_h : h = 15) :
  let EF := (AB + CD) / 2
  let h_EFCD := h / 2
  let area_EFCD := (1 / 2) * (CD + EF) * h_EFCD
  area_EFCD = 187.5 :=
by
  intros EF h_EFCD area_EFCD
  sorry

end area_of_EFCD_l90_9056


namespace cylinder_ratio_l90_9027

theorem cylinder_ratio
  (V : ℝ) (r h : ℝ)
  (h_volume : π * r^2 * h = V)
  (h_surface_area : 2 * π * r * h = 2 * (V / r)) :
  h / r = 2 :=
sorry

end cylinder_ratio_l90_9027


namespace paula_remaining_money_l90_9004

-- Define the given conditions
def given_amount : ℕ := 109
def cost_shirt : ℕ := 11
def number_shirts : ℕ := 2
def cost_pants : ℕ := 13

-- Calculate total spending
def total_spent : ℕ := (cost_shirt * number_shirts) + cost_pants

-- Define the remaining amount Paula has
def remaining_amount : ℕ := given_amount - total_spent

-- State the theorem
theorem paula_remaining_money : remaining_amount = 74 := by
  -- Proof goes here
  sorry

end paula_remaining_money_l90_9004


namespace AB_squared_eq_AB_AB_minus_BA_cubed_eq_zero_l90_9095

variables {n : ℕ} (A B : Matrix (Fin n) (Fin n) ℂ)

-- Condition
def AB2A_eq_AB := A * B ^ 2 * A = A * B * A

-- Part (a): Prove that (AB)^2 = AB
theorem AB_squared_eq_AB (h : AB2A_eq_AB A B) : (A * B) ^ 2 = A * B :=
sorry

-- Part (b): Prove that (AB - BA)^3 = 0
theorem AB_minus_BA_cubed_eq_zero (h : AB2A_eq_AB A B) : (A * B - B * A) ^ 3 = 0 :=
sorry

end AB_squared_eq_AB_AB_minus_BA_cubed_eq_zero_l90_9095


namespace ratio_of_inscribed_squares_l90_9071

-- Definitions of the conditions
def right_triangle_sides (a b c : ℕ) : Prop := a = 6 ∧ b = 8 ∧ c = 10 ∧ a^2 + b^2 = c^2

def inscribed_square_1 (x : ℚ) : Prop := x = 18 / 7

def inscribed_square_2 (y : ℚ) : Prop := y = 32 / 7

-- Statement of the problem
theorem ratio_of_inscribed_squares (x y : ℚ) : right_triangle_sides 6 8 10 ∧ inscribed_square_1 x ∧ inscribed_square_2 y → (x / y) = 9 / 16 :=
by
  sorry

end ratio_of_inscribed_squares_l90_9071


namespace geometric_sequence_S4_l90_9068

-- Definitions from the conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∃ q : ℝ, (∀ n : ℕ, a (n + 1) = q * a n)

def sum_of_geometric_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
∀ n : ℕ, S n = a 1 * ((1 - (a 2 / a 1)^(n+1)) / (1 - (a 2 / a 1)))

def given_condition (S : ℕ → ℝ) : Prop :=
S 7 - 4 * S 6 + 3 * S 5 = 0

-- Problem statement to prove
theorem geometric_sequence_S4 (a : ℕ → ℝ) (S : ℕ → ℝ) (h_geom : is_geometric_sequence a)
  (h_a1 : a 1 = 1) (h_sum : sum_of_geometric_sequence a S) (h_cond : given_condition S) :
  S 4 = 40 := 
sorry

end geometric_sequence_S4_l90_9068


namespace range_of_m_l90_9009

theorem range_of_m (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1^2 - 4 * x1 + m - 1 = 0 ∧ x2^2 - 4 * x2 + m - 1 = 0 ∧ x1 ≠ x2) ∧ 
  (3 * (m - 1) - 4 > 2) →

  3 < m ∧ m ≤ 5 :=
sorry

end range_of_m_l90_9009


namespace sin_780_eq_sqrt3_div_2_l90_9048

theorem sin_780_eq_sqrt3_div_2 :
  Real.sin (780 * Real.pi / 180) = (Real.sqrt 3) / 2 :=
by
  sorry

end sin_780_eq_sqrt3_div_2_l90_9048


namespace problem_statement_l90_9036

theorem problem_statement (r p q : ℝ) (hr : r < 0) (hpq_ne_zero : p * q ≠ 0) (hp2r_gt_q2r : p^2 * r > q^2 * r) :
  ¬ (-p > -q) ∧ ¬ (-p < q) ∧ ¬ (1 < -q / p) ∧ ¬ (1 > q / p) :=
by
  sorry

end problem_statement_l90_9036


namespace area_union_of_reflected_triangles_l90_9032

def point : Type := ℝ × ℝ

def triangle_area (A B C : point) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

def reflect_y_eq_1 (P : point) : point := (P.1, 2 * 1 - P.2)

def area_of_union (A B C : point) (f : point → point) : ℝ :=
  let A' := f A
  let B' := f B
  let C' := f C
  triangle_area A B C + triangle_area A' B' C'

theorem area_union_of_reflected_triangles :
  area_of_union (3, 4) (5, -2) (6, 2) reflect_y_eq_1 = 11 :=
  sorry

end area_union_of_reflected_triangles_l90_9032


namespace ramesh_paid_price_l90_9042

variable (P : ℝ) (P_paid : ℝ)

-- conditions
def discount_price (P : ℝ) : ℝ := 0.80 * P
def additional_cost : ℝ := 125 + 250
def total_cost_with_discount (P : ℝ) : ℝ := discount_price P + additional_cost
def selling_price_without_discount (P : ℝ) : ℝ := 1.10 * P
def given_selling_price : ℝ := 18975

-- the theorem to prove
theorem ramesh_paid_price :
  (∃ P : ℝ, selling_price_without_discount P = given_selling_price ∧ total_cost_with_discount P = 14175) :=
by
  sorry

end ramesh_paid_price_l90_9042


namespace count_not_squares_or_cubes_l90_9075

theorem count_not_squares_or_cubes {A B : Finset ℕ} (h_range : ∀ x, x ∈ A ∪ B → x ≤ 200)
  (h_squares : A.card = 14) (h_cubes : B.card = 5) (h_inter : (A ∩ B).card = 1) :
  (200 - (A ∪ B).card) = 182 :=
by
  sorry

end count_not_squares_or_cubes_l90_9075


namespace remainder_of_product_l90_9040

theorem remainder_of_product (a b c : ℕ) (hc : c ≥ 3) (h1 : a % c = 1) (h2 : b % c = 2) : (a * b) % c = 2 :=
by
  sorry

end remainder_of_product_l90_9040


namespace present_age_of_son_l90_9019

theorem present_age_of_son (S F : ℕ) (h1 : F = S + 22) (h2 : F + 2 = 2 * (S + 2)) : S = 20 :=
by
  sorry

end present_age_of_son_l90_9019


namespace Samantha_purse_value_l90_9087

def cents_per_penny := 1
def cents_per_nickel := 5
def cents_per_dime := 10
def cents_per_quarter := 25

def number_of_pennies := 2
def number_of_nickels := 1
def number_of_dimes := 3
def number_of_quarters := 2

def total_cents := 
  number_of_pennies * cents_per_penny + 
  number_of_nickels * cents_per_nickel + 
  number_of_dimes * cents_per_dime + 
  number_of_quarters * cents_per_quarter

def percent_of_dollar := (total_cents * 100) / 100

theorem Samantha_purse_value : percent_of_dollar = 87 := by
  sorry

end Samantha_purse_value_l90_9087


namespace right_angled_triangle_sets_l90_9052

theorem right_angled_triangle_sets :
  (¬ (1 ^ 2 + 2 ^ 2 = 3 ^ 2)) ∧
  (¬ (2 ^ 2 + 3 ^ 2 = 4 ^ 2)) ∧
  (3 ^ 2 + 4 ^ 2 = 5 ^ 2) ∧
  (¬ (4 ^ 2 + 5 ^ 2 = 6 ^ 2)) :=
by
  sorry

end right_angled_triangle_sets_l90_9052


namespace total_cakes_served_l90_9070

-- Conditions
def cakes_lunch : Nat := 6
def cakes_dinner : Nat := 9

-- Statement of the problem
theorem total_cakes_served : cakes_lunch + cakes_dinner = 15 := 
by
  sorry

end total_cakes_served_l90_9070


namespace school_dinner_theater_tickets_l90_9045

theorem school_dinner_theater_tickets (x y : ℕ)
  (h1 : x + y = 225)
  (h2 : 6 * x + 9 * y = 1875) :
  x = 50 :=
by
  sorry

end school_dinner_theater_tickets_l90_9045


namespace canal_cross_section_area_l90_9079

/-- Definitions of the conditions -/
def top_width : Real := 6
def bottom_width : Real := 4
def depth : Real := 257.25

/-- Proof statement -/
theorem canal_cross_section_area : 
  (1 / 2) * (top_width + bottom_width) * depth = 1286.25 :=
by
  sorry

end canal_cross_section_area_l90_9079


namespace algae_difference_l90_9061

theorem algae_difference :
  let original_algae := 809
  let current_algae := 3263
  current_algae - original_algae = 2454 :=
by
  sorry

end algae_difference_l90_9061


namespace arun_working_days_l90_9091

theorem arun_working_days (A T : ℝ) 
  (h1 : A + T = 1/10) 
  (h2 : A = 1/18) : 
  (1 / A) = 18 :=
by
  -- Proof will be skipped
  sorry

end arun_working_days_l90_9091


namespace total_profit_calculation_l90_9069

variable (investment_Tom : ℝ) (investment_Jose : ℝ) (time_Jose : ℝ) (share_Jose : ℝ) (total_time : ℝ) 
variable (total_profit : ℝ)

theorem total_profit_calculation 
  (h1 : investment_Tom = 30000) 
  (h2 : investment_Jose = 45000) 
  (h3 : time_Jose = 10) -- Jose joined 2 months later, so he invested for 10 months out of 12
  (h4 : share_Jose = 30000) 
  (h5 : total_time = 12) 
  : total_profit = 54000 :=
sorry

end total_profit_calculation_l90_9069


namespace intersection_points_zero_l90_9020

noncomputable def geometric_sequence (a b c : ℝ) : Prop :=
  b^2 = a * c

noncomputable def quadratic_function (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem intersection_points_zero
  (a b c : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (h_gp : geometric_sequence a b c)
  (h_ac_pos : a * c > 0) :
  ∃ x : ℝ, quadratic_function a b c x = 0 → false :=
by
  -- Proof to be completed
  sorry

end intersection_points_zero_l90_9020


namespace time_spent_driving_l90_9096

def distance_home_to_work: ℕ := 60
def speed_mph: ℕ := 40

theorem time_spent_driving:
  (2 * distance_home_to_work) / speed_mph = 3 := by
  sorry

end time_spent_driving_l90_9096


namespace white_paint_amount_is_correct_l90_9035

noncomputable def totalAmountOfPaint (bluePaint: ℝ) (bluePercentage: ℝ): ℝ :=
  bluePaint / bluePercentage

noncomputable def whitePaintAmount (totalPaint: ℝ) (whitePercentage: ℝ): ℝ :=
  totalPaint * whitePercentage

theorem white_paint_amount_is_correct (bluePaint: ℝ) (bluePercentage: ℝ) (whitePercentage: ℝ) (totalPaint: ℝ) :
  bluePaint = 140 → bluePercentage = 0.7 → whitePercentage = 0.1 → totalPaint = totalAmountOfPaint 140 0.7 →
  whitePaintAmount totalPaint 0.1 = 20 :=
by
  intros
  sorry

end white_paint_amount_is_correct_l90_9035


namespace susie_investment_l90_9043

theorem susie_investment :
  ∃ x : ℝ, x * (1 + 0.04)^3 + (2000 - x) * (1 + 0.06)^3 = 2436.29 → x = 820 :=
by
  sorry

end susie_investment_l90_9043


namespace work_completion_rate_l90_9084

theorem work_completion_rate (A B D : ℝ) (W : ℝ) (hB : B = W / 9) (hA : A = W / 10) (hD : D = 90 / 19) : 
  (A + B) * D = W := 
by 
  sorry

end work_completion_rate_l90_9084


namespace parabola_ratio_l90_9015

noncomputable def ratio_AF_BF (p : ℝ) (h_pos : p > 0) : ℝ :=
  let y1 := (Real.sqrt (2 * p * (3 / 2 * p)))
  let y2 := (Real.sqrt (2 * p * (1 / 6 * p)))
  let dist1 := Real.sqrt ((3 / 2 * p - (p / 2))^2 + y1^2)
  let dist2 := Real.sqrt ((1 / 6 * p - p / 2)^2 + y2^2)
  dist1 / dist2

theorem parabola_ratio (p : ℝ) (h_pos : p > 0) : ratio_AF_BF p h_pos = 3 :=
  sorry

end parabola_ratio_l90_9015


namespace combined_apples_sold_l90_9097

theorem combined_apples_sold (red_apples green_apples total_apples : ℕ) 
    (h1 : red_apples = 32) 
    (h2 : green_apples = (3 * (32 / 8))) 
    (h3 : total_apples = red_apples + green_apples) : 
    total_apples = 44 :=
by
  sorry

end combined_apples_sold_l90_9097


namespace solve_for_y_l90_9082

theorem solve_for_y : ∀ y : ℚ, (8 * y^2 + 78 * y + 5) / (2 * y + 19) = 4 * y + 2 → y = -16.5 :=
by
  intro y
  intro h
  sorry

end solve_for_y_l90_9082


namespace hexagon_angle_sum_l90_9025

theorem hexagon_angle_sum (a1 a2 a3 a4 b1 b2 b3 b4 : ℝ) :
  a1 + a2 + a3 + a4 = 360 ∧ b1 + b2 + b3 + b4 = 360 → 
  a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4 = 720 :=
by
  sorry

end hexagon_angle_sum_l90_9025


namespace response_rate_increase_l90_9011

theorem response_rate_increase :
  let original_customers := 70
  let original_responses := 7
  let redesigned_customers := 63
  let redesigned_responses := 9
  let original_response_rate := (original_responses : ℝ) / original_customers
  let redesigned_response_rate := (redesigned_responses : ℝ) / redesigned_customers
  let percentage_increase := ((redesigned_response_rate - original_response_rate) / original_response_rate) * 100
  abs (percentage_increase - 42.86) < 0.01 :=
by
  sorry

end response_rate_increase_l90_9011


namespace bode_law_planet_9_l90_9006

theorem bode_law_planet_9 :
  ∃ (a b : ℝ),
    (a + b = 0.7) ∧ (a + 2 * b = 1) ∧ 
    (70 < a + b * 2^8) ∧ (a + b * 2^8 < 80) :=
by
  -- Define variables and equations based on given conditions
  let a : ℝ := 0.4
  let b : ℝ := 0.3
  
  have h1 : a + b = 0.7 := by 
    sorry  -- Proof that a + b = 0.7
  
  have h2 : a + 2 * b = 1 := by
    sorry  -- Proof that a + 2 * b = 1
  
  have hnine : 70 < a + b * 2^8 ∧ a + b * 2^8 < 80 := by
    -- Calculate a + b * 2^8 and then check the range
    sorry  -- Proof that 70 < a + b * 2^8 < 80

  exact ⟨a, b, h1, h2, hnine⟩

end bode_law_planet_9_l90_9006


namespace betty_age_l90_9016

variable (C A B : ℝ)

-- conditions
def Carol_five_times_Alice := C = 5 * A
def Alice_twelve_years_younger_than_Carol := A = C - 12
def Carol_twice_as_old_as_Betty := C = 2 * B

-- goal
theorem betty_age (hc1 : Carol_five_times_Alice C A)
                  (hc2 : Alice_twelve_years_younger_than_Carol C A)
                  (hc3 : Carol_twice_as_old_as_Betty C B) : B = 7.5 := 
  by
  sorry

end betty_age_l90_9016


namespace increased_time_between_maintenance_checks_l90_9047

theorem increased_time_between_maintenance_checks (original_time : ℕ) (percentage_increase : ℕ) : 
  original_time = 20 → percentage_increase = 25 →
  original_time + (original_time * percentage_increase / 100) = 25 :=
by
  intros
  sorry

end increased_time_between_maintenance_checks_l90_9047


namespace average_first_set_eq_3_more_than_second_set_l90_9017

theorem average_first_set_eq_3_more_than_second_set (x : ℤ) :
  let avg_first_set := (14 + 32 + 53) / 3
  let avg_second_set := (x + 47 + 22) / 3
  avg_first_set = avg_second_set + 3 → x = 21 := by
  sorry

end average_first_set_eq_3_more_than_second_set_l90_9017


namespace work_problem_l90_9003

theorem work_problem (x : ℝ) (hx : x > 0)
    (hB : B_work_rate = 1 / 18)
    (hTogether : together_work_rate = 1 / 7.2)
    (hCombined : together_work_rate = 1 / x + B_work_rate) :
    x = 2 := by
    sorry

end work_problem_l90_9003


namespace solve_inequality_l90_9066

theorem solve_inequality :
  {x : ℝ | x ∈ { y | (y^2 - 5*y + 6) / (y - 3)^2 > 0 }} = {x : ℝ | x < 2} ∪ {x : ℝ | x > 3} :=
by
  sorry

end solve_inequality_l90_9066


namespace Pyarelal_loss_share_l90_9077

-- Define the conditions
variables (P : ℝ) (A : ℝ) (total_loss : ℝ)

-- Ashok's capital is 1/9 of Pyarelal's capital
axiom Ashok_capital : A = (1 / 9) * P

-- Total loss is Rs 900
axiom total_loss_val : total_loss = 900

-- Prove Pyarelal's share of the loss is Rs 810
theorem Pyarelal_loss_share : (P / (A + P)) * total_loss = 810 :=
by
  sorry

end Pyarelal_loss_share_l90_9077


namespace common_difference_arithmetic_geometric_sequence_l90_9005

theorem common_difference_arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) 
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_geom : ∃ r, ∀ n, a (n+1) = a n * r)
  (h_a1 : a 1 = 1) :
  d = 0 :=
by
  sorry

end common_difference_arithmetic_geometric_sequence_l90_9005


namespace towels_after_a_week_l90_9010

theorem towels_after_a_week 
  (initial_green : ℕ) (initial_white : ℕ) (initial_blue : ℕ) 
  (daily_green : ℕ) (daily_white : ℕ) (daily_blue : ℕ) 
  (days : ℕ) 
  (H1 : initial_green = 35)
  (H2 : initial_white = 21)
  (H3 : initial_blue = 15)
  (H4 : daily_green = 3)
  (H5 : daily_white = 1)
  (H6 : daily_blue = 1)
  (H7 : days = 7) :
  (initial_green - daily_green * days) + (initial_white - daily_white * days) + (initial_blue - daily_blue * days) = 36 :=
by 
  sorry

end towels_after_a_week_l90_9010


namespace wood_needed_l90_9067

variable (total_needed : ℕ) (friend_pieces : ℕ) (brother_pieces : ℕ)

/-- Alvin's total needed wood is 376 pieces, he got 123 from his friend and 136 from his brother.
    Prove that Alvin needs 117 more pieces. -/
theorem wood_needed (h1 : total_needed = 376) (h2 : friend_pieces = 123) (h3 : brother_pieces = 136) :
  total_needed - (friend_pieces + brother_pieces) = 117 := by
  sorry

end wood_needed_l90_9067


namespace hypotenuse_of_right_triangle_l90_9063

theorem hypotenuse_of_right_triangle (h : height_dropped_to_hypotenuse = 1) (a : acute_angle = 15) :
∃ (hypotenuse : ℝ), hypotenuse = 4 :=
sorry

end hypotenuse_of_right_triangle_l90_9063


namespace roots_of_polynomial_l90_9074

-- Define the polynomial P(x) = x^3 - 3x^2 - x + 3
def P (x : ℝ) : ℝ := x^3 - 3 * x^2 - x + 3

-- Define the statement to prove the roots of the polynomial
theorem roots_of_polynomial :
  ∀ x : ℝ, (P x = 0 ↔ x = 1 ∨ x = -1 ∨ x = 3) :=
by
  sorry

end roots_of_polynomial_l90_9074


namespace fraction_comparison_l90_9089

theorem fraction_comparison :
  (1998:ℝ) ^ 2000 / (2000:ℝ) ^ 1998 > (1997:ℝ) ^ 1999 / (1999:ℝ) ^ 1997 :=
by sorry

end fraction_comparison_l90_9089


namespace total_number_of_balls_l90_9028

def number_of_yellow_balls : Nat := 6
def probability_yellow_ball : Rat := 1 / 9

theorem total_number_of_balls (N : Nat) (h1 : number_of_yellow_balls = 6) (h2 : probability_yellow_ball = 1 / 9) :
    6 / N = 1 / 9 → N = 54 := 
by
  sorry

end total_number_of_balls_l90_9028


namespace polynomial_term_equality_l90_9041

theorem polynomial_term_equality (p q : ℝ) (hpq_pos : 0 < p) (hq_pos : 0 < q) 
  (h_sum : p + q = 1) (h_eq : 28 * p^6 * q^2 = 56 * p^5 * q^3) : p = 2 / 3 :=
by
  sorry

end polynomial_term_equality_l90_9041


namespace area_of_circle_l90_9021

theorem area_of_circle (r : ℝ) (h : r = 3) : 
  (∀ A : ℝ, A = π * r^2) → A = 9 * π :=
by
  intro area_formula
  sorry

end area_of_circle_l90_9021


namespace prob_exactly_two_trains_on_time_is_0_398_l90_9098

-- Definitions and conditions
def eventA := true
def eventB := true
def eventC := true

def P_A : ℝ := 0.8
def P_B : ℝ := 0.7
def P_C : ℝ := 0.9

def P_not_A : ℝ := 1 - P_A
def P_not_B : ℝ := 1 - P_B
def P_not_C : ℝ := 1 - P_C

-- Question definition (to be proved)
def exact_two_on_time : ℝ :=
  P_A * P_B * P_not_C + P_A * P_not_B * P_C + P_not_A * P_B * P_C

-- Theorem statement
theorem prob_exactly_two_trains_on_time_is_0_398 :
  exact_two_on_time = 0.398 := sorry

end prob_exactly_two_trains_on_time_is_0_398_l90_9098


namespace pool_filling_times_l90_9034

theorem pool_filling_times:
  ∃ (x y z u : ℕ),
    (1/x + 1/y = 1/70) ∧
    (1/x + 1/z = 1/84) ∧
    (1/y + 1/z = 1/140) ∧
    (1/u = 1/x + 1/y + 1/z) ∧
    (x = 105) ∧
    (y = 210) ∧
    (z = 420) ∧
    (u = 60) := 
  sorry

end pool_filling_times_l90_9034
