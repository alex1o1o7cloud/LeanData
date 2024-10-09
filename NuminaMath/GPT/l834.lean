import Mathlib

namespace car_travel_distance_l834_83473

theorem car_travel_distance (v d : ℕ) 
  (h1 : d = v * 7)
  (h2 : d = (v + 12) * 5) : 
  d = 210 := by 
  sorry

end car_travel_distance_l834_83473


namespace converse_proposition_l834_83403

-- Define the predicate variables p and q
variables (p q : Prop)

-- State the theorem about the converse of the proposition
theorem converse_proposition (hpq : p → q) : q → p :=
sorry

end converse_proposition_l834_83403


namespace ice_cream_arrangements_is_correct_l834_83480

-- Let us define the problem: counting the number of unique stacks of ice cream flavors
def ice_cream_scoops_arrangements : ℕ :=
  let total_scoops := 5
  let vanilla_scoops := 2
  Nat.factorial total_scoops / Nat.factorial vanilla_scoops

-- Assertion that needs to be proved
theorem ice_cream_arrangements_is_correct : ice_cream_scoops_arrangements = 60 := by
  -- Proof to be filled in; current placeholder
  sorry

end ice_cream_arrangements_is_correct_l834_83480


namespace this_week_usage_less_next_week_usage_less_l834_83432

def last_week_usage : ℕ := 91

def usage_this_week : ℕ := (4 * 8) + (3 * 10)

def usage_next_week : ℕ := (5 * 5) + (2 * 12)

theorem this_week_usage_less : last_week_usage - usage_this_week = 29 := by
  -- proof goes here
  sorry

theorem next_week_usage_less : last_week_usage - usage_next_week = 42 := by
  -- proof goes here
  sorry

end this_week_usage_less_next_week_usage_less_l834_83432


namespace puja_runs_distance_in_meters_l834_83463

noncomputable def puja_distance (time_in_seconds : ℝ) (speed_kmph : ℝ) : ℝ :=
  let time_in_hours := time_in_seconds / 3600
  let distance_km := speed_kmph * time_in_hours
  distance_km * 1000

theorem puja_runs_distance_in_meters :
  abs (puja_distance 59.995200383969284 30 - 499.96) < 0.01 :=
by
  sorry

end puja_runs_distance_in_meters_l834_83463


namespace noah_billed_amount_l834_83475

theorem noah_billed_amount
  (minutes_per_call : ℕ)
  (cost_per_minute : ℝ)
  (weeks_per_year : ℕ)
  (total_cost : ℝ)
  (h_minutes_per_call : minutes_per_call = 30)
  (h_cost_per_minute : cost_per_minute = 0.05)
  (h_weeks_per_year : weeks_per_year = 52)
  (h_total_cost : total_cost = 78) :
  (minutes_per_call * cost_per_minute * weeks_per_year = total_cost) :=
by
  sorry

end noah_billed_amount_l834_83475


namespace f_x_minus_1_pass_through_l834_83487

variable (a : ℝ)

def f (x : ℝ) : ℝ := a * x^2 + x

theorem f_x_minus_1_pass_through (a : ℝ) : f a (1 - 1) = 0 :=
by
  -- Proof is omitted here
  sorry

end f_x_minus_1_pass_through_l834_83487


namespace future_age_ratio_l834_83479

theorem future_age_ratio (j e x : ℕ) 
  (h1 : j - 3 = 5 * (e - 3)) 
  (h2 : j - 7 = 6 * (e - 7)) 
  (h3 : x = 17) : (j + x) / (e + x) = 3 := 
by
  sorry

end future_age_ratio_l834_83479


namespace fraction_of_red_marbles_after_tripling_blue_l834_83424

theorem fraction_of_red_marbles_after_tripling_blue (x : ℕ) (h₁ : ∃ y, y = (4 * x) / 7) (h₂ : ∃ z, z = (3 * x) / 7) :
  (3 * x / 7) / (((12 * x) / 7) + ((3 * x) / 7)) = 1 / 5 :=
by
  sorry

end fraction_of_red_marbles_after_tripling_blue_l834_83424


namespace tub_emptying_time_l834_83496

variables (x C D T : ℝ) (hx : x > 0) (hC : C > 0) (hD : D > 0)

theorem tub_emptying_time (h1 : 4 * (D - x) = (5 / 7) * C) :
  T = 8 / (5 + (28 * x) / C) :=
by sorry

end tub_emptying_time_l834_83496


namespace tournament_games_count_l834_83491

-- Defining the problem conditions
def num_players : Nat := 12
def plays_twice : Bool := true

-- Theorem statement
theorem tournament_games_count (n : Nat) (plays_twice : Bool) (h : n = num_players ∧ plays_twice = true) :
  (n * (n - 1) * 2) = 264 := by
  sorry

end tournament_games_count_l834_83491


namespace calories_in_300g_lemonade_proof_l834_83420

def g_lemon := 150
def g_sugar := 200
def g_water := 450

def c_lemon_per_100g := 30
def c_sugar_per_100g := 400
def c_water := 0

def total_calories :=
  g_lemon * c_lemon_per_100g / 100 +
  g_sugar * c_sugar_per_100g / 100 +
  g_water * c_water

def total_weight := g_lemon + g_sugar + g_water

def caloric_density := total_calories / total_weight

def calories_in_300g_lemonade := 300 * caloric_density

theorem calories_in_300g_lemonade_proof : calories_in_300g_lemonade = 317 := by
  sorry

end calories_in_300g_lemonade_proof_l834_83420


namespace age_sum_is_ninety_l834_83412

theorem age_sum_is_ninety (a b c : ℕ)
  (h1 : a = 20 + b + c)
  (h2 : a^2 = 1800 + (b + c)^2) :
  a + b + c = 90 := 
sorry

end age_sum_is_ninety_l834_83412


namespace ratio_n_over_p_l834_83409

-- Definitions and conditions from the problem
variables {m n p : ℝ}

-- The quadratic equation x^2 + mx + n = 0 has roots that are thrice those of x^2 + px + m = 0.
-- None of m, n, and p is zero.

-- Prove that n / p = 27 given these conditions.
theorem ratio_n_over_p (hmn0 : m ≠ 0) (hn : n = 9 * m) (hp : p = m / 3):
  n / p = 27 :=
  by
    sorry -- Formal proof will go here.

end ratio_n_over_p_l834_83409


namespace rt_triangle_case1_rt_triangle_case2_rt_triangle_case3_l834_83418

-- Case 1
theorem rt_triangle_case1
  (a : ℝ) (b : ℝ) (c : ℝ) (A B C : ℝ)
  (h : A = 30) (h_bc : B + C = 90) (h_ac : A + C = 90)
  (ha : a = 4) (hb : b = 4 * Real.sqrt 3) (hc : c = 8)
  : b = 4 * Real.sqrt 3 ∧ c = 8 := by
  sorry

-- Case 2
theorem rt_triangle_case2
  (a : ℝ) (b : ℝ) (c : ℝ) (A B C : ℝ)
  (h : B = 60) (h_bc : B + C = 90) (h_ac : A + C = 90)
  (ha : a = Real.sqrt 3 - 1) (hb : b = 3 - Real.sqrt 3) 
  (ha_b: A = 30)
  (h_c: c = 2 * Real.sqrt 3 - 2)
  : B = 60 ∧ A = 30 ∧ c = 2 * Real.sqrt 3 - 2 := by
  sorry

-- Case 3
theorem rt_triangle_case3
  (a : ℝ) (b : ℝ) (c : ℝ) (A B C : ℝ)
  (h : A = 60) (h_bc : B + C = 90) (h_ac : A + C = 90)
  (hc : c = 2 + Real.sqrt 3)
  (ha : a = Real.sqrt 3 + 3/2) 
  (hb: b = (2 + Real.sqrt 3) / 2)
  : a = Real.sqrt 3 + 3/2 ∧ b = (2 + Real.sqrt 3) / 2 := by
  sorry

end rt_triangle_case1_rt_triangle_case2_rt_triangle_case3_l834_83418


namespace value_of_t_plus_one_over_t_l834_83453

theorem value_of_t_plus_one_over_t
  (t : ℝ)
  (h1 : t^2 - 3 * t + 1 = 0)
  (h2 : t ≠ 0) :
  t + 1 / t = 3 :=
by
  sorry

end value_of_t_plus_one_over_t_l834_83453


namespace yancheng_marathon_half_marathon_estimated_probability_l834_83481

noncomputable def estimated_probability
  (surveyed_participants_frequencies : List (ℕ × Real)) : Real :=
by
  -- Define the surveyed participants and their corresponding frequencies
  -- In this example, [(20, 0.35), (50, 0.40), (100, 0.39), (200, 0.415), (500, 0.418), (2000, 0.411)]
  sorry

theorem yancheng_marathon_half_marathon_estimated_probability :
  let surveyed_participants_frequencies := [
    (20, 0.350),
    (50, 0.400),
    (100, 0.390),
    (200, 0.415),
    (500, 0.418),
    (2000, 0.411)
  ]
  estimated_probability surveyed_participants_frequencies = 0.40 :=
by
  sorry

end yancheng_marathon_half_marathon_estimated_probability_l834_83481


namespace prove_angle_sum_l834_83492

open Real

theorem prove_angle_sum (α β : ℝ) 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 0 < β ∧ β < π / 2) 
  (h3 : cos α / sin β + cos β / sin α = 2) : 
  α + β = π / 2 := 
sorry

end prove_angle_sum_l834_83492


namespace intersection_complement_l834_83483

-- Define the sets A and B
def A : Set ℝ := { x : ℝ | -1 ≤ x ∧ x ≤ 1 }
def B : Set ℝ := { x : ℝ | x > 0 }

-- Define the complement of B
def complement_B : Set ℝ := { x : ℝ | x ≤ 0 }

-- The theorem we need to prove
theorem intersection_complement :
  A ∩ complement_B = { x : ℝ | -1 ≤ x ∧ x ≤ 0 } := 
by
  sorry

end intersection_complement_l834_83483


namespace combined_average_score_l834_83422

theorem combined_average_score (M E : ℕ) (m e : ℕ) (h1 : M = 82) (h2 : E = 68) (h3 : m = 5 * e / 7) :
  ((m * M) + (e * E)) / (m + e) = 72 :=
by
  -- Placeholder for the proof
  sorry

end combined_average_score_l834_83422


namespace triangle_inequality_third_side_l834_83427

theorem triangle_inequality_third_side (a b x : ℝ) (h₁ : a = 3) (h₂ : b = 5) (h₃ : 0 < x) (h₄ : x < a + b) (h₅ : a < b + x) (h₆ : b < a + x) :
  ¬(x = 9) := by
  sorry

end triangle_inequality_third_side_l834_83427


namespace kamal_marks_in_english_l834_83404

theorem kamal_marks_in_english :
  ∀ (E Math Physics Chemistry Biology Average : ℕ), 
    Math = 65 → 
    Physics = 82 → 
    Chemistry = 67 → 
    Biology = 85 → 
    Average = 79 → 
    (Math + Physics + Chemistry + Biology + E) / 5 = Average → 
    E = 96 :=
by
  intros E Math Physics Chemistry Biology Average
  intros hMath hPhysics hChemistry hBiology hAverage hTotal
  sorry

end kamal_marks_in_english_l834_83404


namespace quadratic_product_fact_l834_83457

def quadratic_factors_product : Prop :=
  let integer_pairs := [(-1, 24), (-2, 12), (-3, 8), (-4, 6), (-6, 4), (-8, 3), (-12, 2), (-24, 1)]
  let t_values := integer_pairs.map (fun (c, d) => c + d)
  let product_t := t_values.foldl (fun acc t => acc * t) 1
  product_t = -5290000

theorem quadratic_product_fact : quadratic_factors_product :=
by sorry

end quadratic_product_fact_l834_83457


namespace expression_value_l834_83468

/-- The value of the expression 1.5 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) is 1200. -/
theorem expression_value : 
  1.5 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 1200 :=
by
  sorry

end expression_value_l834_83468


namespace points_on_circle_l834_83430

theorem points_on_circle (t : ℝ) : 
  let x := (2 - t^2) / (2 + t^2)
  let y := (3 * t) / (2 + t^2)
  x^2 + y^2 = 1 := 
by 
  let x := (2 - t^2) / (2 + t^2)
  let y := (3 * t) / (2 + t^2)
  sorry

end points_on_circle_l834_83430


namespace savings_same_l834_83423

theorem savings_same (A_salary B_salary total_salary : ℝ)
  (A_spend_perc B_spend_perc : ℝ)
  (h_total : A_salary + B_salary = total_salary)
  (h_A_salary : A_salary = 4500)
  (h_A_spend_perc : A_spend_perc = 0.95)
  (h_B_spend_perc : B_spend_perc = 0.85)
  (h_total_salary : total_salary = 6000) :
  ((1 - A_spend_perc) * A_salary) = ((1 - B_spend_perc) * B_salary) :=
by
  sorry

end savings_same_l834_83423


namespace remainder_when_divided_by_eleven_l834_83440

-- Definitions from the conditions
def two_pow_five_mod_eleven : ℕ := 10
def two_pow_ten_mod_eleven : ℕ := 1
def ten_mod_eleven : ℕ := 10
def ten_square_mod_eleven : ℕ := 1

-- Proposition we want to prove
theorem remainder_when_divided_by_eleven :
  (7 * 10^20 + 2^20) % 11 = 8 := 
by 
  -- Proof goes here
  sorry

end remainder_when_divided_by_eleven_l834_83440


namespace square_binomial_formula_l834_83474

variable {x y : ℝ}

theorem square_binomial_formula :
  (2 * x + y) * (y - 2 * x) = y^2 - 4 * x^2 := 
  sorry

end square_binomial_formula_l834_83474


namespace average_speed_l834_83401

   theorem average_speed (x : ℝ) : 
     let s1 := 40
     let s2 := 20
     let d1 := x
     let d2 := 2 * x
     let total_distance := d1 + d2
     let time1 := d1 / s1
     let time2 := d2 / s2
     let total_time := time1 + time2
     total_distance / total_time = 24 :=
   by
     sorry
   
end average_speed_l834_83401


namespace find_x_l834_83486

theorem find_x
  (p q : ℝ)
  (h1 : 3 / p = x)
  (h2 : 3 / q = 18)
  (h3 : p - q = 0.33333333333333337) :
  x = 6 :=
sorry

end find_x_l834_83486


namespace fraction_is_three_eights_l834_83441

-- The given number
def number := 48

-- The fraction 'x' by which the number exceeds by 30
noncomputable def fraction (x : ℝ) : Prop :=
number = number * x + 30

-- Our goal is to prove that the fraction is 3/8
theorem fraction_is_three_eights : fraction (3 / 8) :=
by
  -- We reduced the goal proof to a simpler form for illustration, you can solve it rigorously
  sorry

end fraction_is_three_eights_l834_83441


namespace geometric_sequence_common_ratio_l834_83449

theorem geometric_sequence_common_ratio (a1 a2 a3 a4 : ℝ)
  (h₁ : a1 = 32) (h₂ : a2 = -48) (h₃ : a3 = 72) (h₄ : a4 = -108)
  (h_geom : ∃ r, a2 = r * a1 ∧ a3 = r * a2 ∧ a4 = r * a3) :
  ∃ r, r = -3/2 :=
by
  sorry

end geometric_sequence_common_ratio_l834_83449


namespace margo_total_distance_travelled_l834_83471

noncomputable def total_distance_walked (walking_time_in_minutes: ℝ) (stopping_time_in_minutes: ℝ) (additional_walking_time_in_minutes: ℝ) (walking_speed: ℝ) : ℝ :=
  walking_speed * ((walking_time_in_minutes + stopping_time_in_minutes + additional_walking_time_in_minutes) / 60)

noncomputable def total_distance_cycled (cycling_time_in_minutes: ℝ) (cycling_speed: ℝ) : ℝ :=
  cycling_speed * (cycling_time_in_minutes / 60)

theorem margo_total_distance_travelled :
  let walking_time := 10
  let stopping_time := 15
  let additional_walking_time := 10
  let cycling_time := 15
  let walking_speed := 4
  let cycling_speed := 10

  total_distance_walked walking_time stopping_time additional_walking_time walking_speed +
  total_distance_cycled cycling_time cycling_speed = 4.8333 := 
by 
  sorry

end margo_total_distance_travelled_l834_83471


namespace hexagon_largest_angle_l834_83488

-- Definitions for conditions
def hexagon_interior_angle_sum : ℝ := 720  -- Sum of all interior angles of hexagon

def angle_A : ℝ := 100
def angle_B : ℝ := 120

-- Define x for angles C and D
variables (x : ℝ)
def angle_C : ℝ := x
def angle_D : ℝ := x
def angle_F : ℝ := 3 * x + 10

-- The formal statement to prove
theorem hexagon_largest_angle (x : ℝ) : 
  100 + 120 + x + x + (3 * x + 10) = 720 → 
  3 * x + 10 = 304 :=
by 
  sorry

end hexagon_largest_angle_l834_83488


namespace total_difference_is_correct_l834_83437

-- Define the harvest rates
def valencia_weekday_ripe := 90
def valencia_weekday_unripe := 38
def navel_weekday_ripe := 125
def navel_weekday_unripe := 65
def blood_weekday_ripe := 60
def blood_weekday_unripe := 42

def valencia_weekend_ripe := 75
def valencia_weekend_unripe := 33
def navel_weekend_ripe := 100
def navel_weekend_unripe := 57
def blood_weekend_ripe := 45
def blood_weekend_unripe := 36

-- Define the number of weekdays and weekend days
def weekdays := 5
def weekend_days := 2

-- Calculate the total harvests
def total_valencia_ripe := valencia_weekday_ripe * weekdays + valencia_weekend_ripe * weekend_days
def total_valencia_unripe := valencia_weekday_unripe * weekdays + valencia_weekend_unripe * weekend_days
def total_navel_ripe := navel_weekday_ripe * weekdays + navel_weekend_ripe * weekend_days
def total_navel_unripe := navel_weekday_unripe * weekdays + navel_weekend_unripe * weekend_days
def total_blood_ripe := blood_weekday_ripe * weekdays + blood_weekend_ripe * weekend_days
def total_blood_unripe := blood_weekday_unripe * weekdays + blood_weekend_unripe * weekend_days

-- Calculate the total differences
def valencia_difference := total_valencia_ripe - total_valencia_unripe
def navel_difference := total_navel_ripe - total_navel_unripe
def blood_difference := total_blood_ripe - total_blood_unripe

-- Define the total difference
def total_difference := valencia_difference + navel_difference + blood_difference

-- Theorem statement
theorem total_difference_is_correct :
  total_difference = 838 := by
  sorry

end total_difference_is_correct_l834_83437


namespace tomatoes_cheaper_than_cucumbers_percentage_l834_83456

noncomputable def P_c := 5
noncomputable def two_T_three_P_c := 23
noncomputable def T := (two_T_three_P_c - 3 * P_c) / 2
noncomputable def percentage_by_which_tomatoes_cheaper_than_cucumbers := ((P_c - T) / P_c) * 100

theorem tomatoes_cheaper_than_cucumbers_percentage : 
  P_c = 5 → 
  (2 * T + 3 * P_c = 23) →
  T < P_c →
  percentage_by_which_tomatoes_cheaper_than_cucumbers = 20 :=
by
  intros
  sorry

end tomatoes_cheaper_than_cucumbers_percentage_l834_83456


namespace xiao_ying_should_pay_l834_83493

variable (x y z : ℝ)

def equation1 := 3 * x + 7 * y + z = 14
def equation2 := 4 * x + 10 * y + z = 16
def equation3 := 2 * (x + y + z) = 20

theorem xiao_ying_should_pay :
  equation1 x y z →
  equation2 x y z →
  equation3 x y z :=
by
  intros h1 h2
  sorry

end xiao_ying_should_pay_l834_83493


namespace time_lent_to_C_eq_l834_83452

variable (principal_B : ℝ := 5000)
variable (time_B : ℕ := 2)
variable (principal_C : ℝ := 3000)
variable (total_interest : ℝ := 1980)
variable (rate_of_interest_per_annum : ℝ := 0.09)

theorem time_lent_to_C_eq (n : ℝ) (H : principal_B * rate_of_interest_per_annum * time_B + principal_C * rate_of_interest_per_annum * n = total_interest) : 
  n = 2 / 3 :=
by
  sorry

end time_lent_to_C_eq_l834_83452


namespace range_of_k_l834_83462

theorem range_of_k (k : ℝ) (f : ℝ → ℝ) (h : ∀ x : ℝ, 0 ≤ k * x^2 + k * x + 3) :
  0 ≤ k ∧ k ≤ 12 :=
sorry

end range_of_k_l834_83462


namespace complement_of_alpha_l834_83417

-- Define that the angle α is given as 44 degrees 36 minutes
def alpha : ℚ := 44 + 36 / 60  -- using rational numbers to represent the degrees and minutes

-- Define the complement function
def complement (angle : ℚ) : ℚ := 90 - angle

-- State the proposition to prove
theorem complement_of_alpha : complement alpha = 45 + 24 / 60 := 
by
  sorry

end complement_of_alpha_l834_83417


namespace simplify_expression_l834_83425

theorem simplify_expression (x y : ℤ) (h1 : x = -1) (h2 : y = 2) :
  (3 * x^2 * y - 2 * x * y^2) - (x * y^2 - 2 * x^2 * y) - 2 * (-3 * x^2 * y - x * y^2) = 26 :=
by
  rw [h1, h2]
  sorry

end simplify_expression_l834_83425


namespace num_triangles_with_perimeter_9_l834_83498

-- Definitions for side lengths and their conditions
def is_triangle (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

def perimeter (a b c : ℕ) : Prop :=
  a + b + c = 9

-- The main theorem
theorem num_triangles_with_perimeter_9 : 
  ∃ (n : ℕ), n = 2 ∧ ∀ (a b c : ℕ), a + b + c = 9 → a ≤ b ∧ b ≤ c → is_triangle a b c ↔ 
  (a = 2 ∧ b = 3 ∧ c = 4) ∨ (a = 3 ∧ b = 3 ∧ c = 3) :=
by
  sorry

end num_triangles_with_perimeter_9_l834_83498


namespace determine_radius_l834_83482

variable (R r : ℝ)

theorem determine_radius (h1 : R = 10) (h2 : π * R^2 = 2 * (π * R^2 - π * r^2)) : r = 5 * Real.sqrt 2 :=
  sorry

end determine_radius_l834_83482


namespace find_A_l834_83434

theorem find_A (A B C D E F G H I J : ℕ)
  (h1 : A > B ∧ B > C)
  (h2 : D > E ∧ E > F)
  (h3 : G > H ∧ H > I ∧ I > J)
  (h4 : (D = E + 2) ∧ (E = F + 2))
  (h5 : (G = H + 2) ∧ (H = I + 2) ∧ (I = J + 2))
  (h6 : A + B + C = 10) : A = 6 :=
sorry

end find_A_l834_83434


namespace min_product_of_positive_numbers_l834_83407

theorem min_product_of_positive_numbers {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (h : a * b = a + b) : a * b = 4 :=
sorry

end min_product_of_positive_numbers_l834_83407


namespace train_passing_time_l834_83458

def train_distance_km : ℝ := 10
def train_time_min : ℝ := 15
def train_length_m : ℝ := 111.11111111111111

theorem train_passing_time : 
  let time_to_pass_signal_post := train_length_m / ((train_distance_km * 1000) / (train_time_min * 60))
  time_to_pass_signal_post = 10 :=
by
  sorry

end train_passing_time_l834_83458


namespace divisibility_condition_l834_83466

theorem divisibility_condition
  (a p q : ℕ) (hpq : p ≤ q) (hp_pos : 0 < p) (hq_pos : 0 < q) (ha_pos : 0 < a) :
  (p ∣ a^p ∨ p ∣ a^q) → (p ∣ a^p ∧ p ∣ a^q) :=
by
  sorry

end divisibility_condition_l834_83466


namespace proposition_a_sufficient_not_necessary_negation_of_proposition_b_incorrect_proposition_c_not_necessary_proposition_d_necessary_not_sufficient_final_answer_correct_l834_83450

theorem proposition_a_sufficient_not_necessary (a : ℝ) : (a > 1 → 1 / a < 1) ∧ (1 / a < 1 → a > 1 ∨ a < 1) :=
sorry

theorem negation_of_proposition_b_incorrect (x : ℝ) : ¬(∀ x < 1, x^2 < 1) ↔ ∃ x < 1, x^2 ≥ 1 :=
sorry

theorem proposition_c_not_necessary (x y : ℝ) : (x ≥ 2 ∧ y ≥ 2 → x^2 + y^2 ≥ 8) ∧ (x^2 + y^2 ≥ 4 → ¬(x ≥ 2 ∧ y ≥ 2)) :=
sorry

theorem proposition_d_necessary_not_sufficient (a b : ℝ) : (a ≠ 0 → ab ≠ 0) ∧ (ab ≠ 0 → a ≠ 0 ∨ b ≠ 0) :=
sorry

theorem final_answer_correct :
  let proposition_A := (∃ (a : ℝ), a > 1 ∧ 1 / a < 1 ∧ (1 / a < 1 → a > 1 ∨ a < 1))
  let proposition_B := (¬(∀ (x : ℝ), x < 1 → x^2 < 1) ↔ ∃ (x : ℝ), x < 1 ∧ x^2 ≥ 1)
  let proposition_C := (∃ (x y : ℝ), (x ≥ 2 ∧ y ≥ 2 → x^2 + y^2 ≥ 8) ∧ (x^2 + y^2 ≥ 4 → ¬(x ≥ 2 ∧ y ≥ 2)))
  let proposition_D := (∃ (a b : ℝ), a ≠ 0 ∧ ab ≠ 0 ∧ (ab ≠ 0 → a ≠ 0 ∨ b ≠ 0))
  proposition_A ∧ proposition_D
:= 
sorry

end proposition_a_sufficient_not_necessary_negation_of_proposition_b_incorrect_proposition_c_not_necessary_proposition_d_necessary_not_sufficient_final_answer_correct_l834_83450


namespace trajectory_is_ellipse_l834_83495

noncomputable def trajectory_of_P (P : ℝ × ℝ) : Prop :=
  ∃ (N : ℝ × ℝ), N.fst^2 + N.snd^2 = 8 ∧ 
                 ∃ (M : ℝ × ℝ), M.fst = 0 ∧ M.snd = N.snd ∧
                 P.fst = N.fst / 2 ∧ P.snd = N.snd

theorem trajectory_is_ellipse (P : ℝ × ℝ) (h : trajectory_of_P P) : 
  P.fst^2 / 2 + P.snd^2 / 8 = 1 :=
by
  sorry

end trajectory_is_ellipse_l834_83495


namespace max_a9_l834_83467

theorem max_a9 (a : Fin 18 → ℕ) (h_pos: ∀ i, 1 ≤ a i) (h_incr: ∀ i j, i < j → a i < a j) (h_sum: (Finset.univ : Finset (Fin 18)).sum a = 2001) : a 8 ≤ 192 :=
by
  -- Proof goes here
  sorry

end max_a9_l834_83467


namespace first_term_of_geometric_series_l834_83469

theorem first_term_of_geometric_series (r a S : ℝ) (h_r : r = 1 / 4) (h_S : S = 40) 
  (h_geometric_sum : S = a / (1 - r)) : a = 30 :=
by
  -- The proof would go here, but we place a sorry to skip the proof.
  sorry

end first_term_of_geometric_series_l834_83469


namespace fill_missing_digits_l834_83415

noncomputable def first_number (a : ℕ) : ℕ := a * 1000 + 2 * 100 + 5 * 10 + 7
noncomputable def second_number (b c : ℕ) : ℕ := 2 * 1000 + b * 100 + 9 * 10 + c

theorem fill_missing_digits (a b c : ℕ) : a = 1 ∧ b = 5 ∧ c = 6 → first_number a + second_number b c = 5842 :=
by
  intros
  sorry

end fill_missing_digits_l834_83415


namespace rectangle_division_impossible_l834_83446

theorem rectangle_division_impossible :
  ¬ ∃ n m : ℕ, n * 5 = 55 ∧ m * 11 = 39 :=
by
  sorry

end rectangle_division_impossible_l834_83446


namespace solve_quadratic_eq_l834_83494

theorem solve_quadratic_eq (x : ℝ) : 4 * x ^ 2 - (x - 1) ^ 2 = 0 ↔ x = -1 ∨ x = 1 / 3 :=
by
  sorry

end solve_quadratic_eq_l834_83494


namespace min_sum_of_dimensions_l834_83454

theorem min_sum_of_dimensions (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 3003) :
  a + b + c = 45 := sorry

end min_sum_of_dimensions_l834_83454


namespace part_one_part_two_l834_83499

-- Part (1)
theorem part_one (m : ℝ) : 
  (∀ x, -1 ≤ x ∧ x ≤ 2 → (2 * m < x ∧ x < 1 → -1 ≤ x ∧ x ≤ 2 ∧ - (1 / 2) ≤ m)) → 
  (m ≥ - (1 / 2)) :=
by sorry

-- Part (2)
theorem part_two (m : ℝ) : 
  (∃ x : ℤ, (2 * m < x ∧ x < 1) ∧ (x < -1 ∨ x > 2)) ∧ 
  (∀ y : ℤ, (2 * m < y ∧ y < 1) ∧ (y < -1 ∨ y > 2) → y = x) → 
  (- (3 / 2) ≤ m ∧ m < -1) :=
by sorry

end part_one_part_two_l834_83499


namespace excess_calories_l834_83447

theorem excess_calories (bags : ℕ) (ounces_per_bag : ℕ) (calories_per_ounce : ℕ)
  (run_minutes : ℕ) (calories_per_minute : ℕ)
  (h_bags : bags = 3) (h_ounces_per_bag : ounces_per_bag = 2)
  (h_calories_per_ounce : calories_per_ounce = 150)
  (h_run_minutes : run_minutes = 40)
  (h_calories_per_minute : calories_per_minute = 12) :
  (bags * ounces_per_bag * calories_per_ounce) - (run_minutes * calories_per_minute) = 420 := by
  sorry

end excess_calories_l834_83447


namespace maximize_profit_price_l834_83461

-- Definitions from the conditions
def initial_price : ℝ := 80
def initial_sales : ℝ := 200
def price_reduction_per_unit : ℝ := 1
def sales_increase_per_unit : ℝ := 20
def cost_price_per_helmet : ℝ := 50

-- Profit function
def profit (x : ℝ) : ℝ :=
  (x - cost_price_per_helmet) * (initial_sales + (initial_price - x) * sales_increase_per_unit)

-- The theorem statement
theorem maximize_profit_price : 
  ∃ x, (x = 70) ∧ (∀ y, profit y ≤ profit x) :=
sorry

end maximize_profit_price_l834_83461


namespace average_marks_for_class_l834_83478

theorem average_marks_for_class (total_students : ℕ) (marks_group1 marks_group2 marks_group3 : ℕ) (num_students_group1 num_students_group2 num_students_group3 : ℕ) 
  (h1 : total_students = 50) 
  (h2 : num_students_group1 = 10) 
  (h3 : marks_group1 = 90) 
  (h4 : num_students_group2 = 15) 
  (h5 : marks_group2 = 80) 
  (h6 : num_students_group3 = total_students - num_students_group1 - num_students_group2) 
  (h7 : marks_group3 = 60) : 
  (10 * 90 + 15 * 80 + (total_students - 10 - 15) * 60) / total_students = 72 := 
by
  sorry

end average_marks_for_class_l834_83478


namespace material_for_one_pillowcase_l834_83460

def material_in_first_bale (x : ℝ) : Prop :=
  4 * x + 1100 = 5000

def material_in_third_bale : ℝ := 0.22 * 5000

def total_material_used_for_producing_items (x y : ℝ) : Prop :=
  150 * (y + 3.25) + 240 * y = x

theorem material_for_one_pillowcase :
  ∀ (x y : ℝ), 
    material_in_first_bale x → 
    material_in_third_bale = 1100 → 
    (x = 975) → 
    total_material_used_for_producing_items x y →
    y = 1.25 :=
by
  intro x y h1 h2 h3 h4
  rw [h3] at h4
  have : 150 * (y + 3.25) + 240 * y = 975 := h4
  sorry

end material_for_one_pillowcase_l834_83460


namespace nickel_chocolates_l834_83477

theorem nickel_chocolates (N : ℕ) (h : 7 = N + 2) : N = 5 :=
by
  sorry

end nickel_chocolates_l834_83477


namespace directrix_of_parabola_l834_83470

-- Define the variables and constants
variables (x y a : ℝ) (h₁ : x^2 = 4 * a * y) (h₂ : x = -2) (h₃ : y = 1)

theorem directrix_of_parabola (h : (-2)^2 = 4 * a * 1) : y = -1 := 
by
  -- Our proof will happen here, but we omit the details
  sorry

end directrix_of_parabola_l834_83470


namespace brian_video_watching_time_l834_83416

/--
Brian watches a 4-minute video of cats.
Then he watches a video twice as long as the cat video involving dogs.
Finally, he watches a video on gorillas that's twice as long as the combined duration of the first two videos.
Prove that Brian spends a total of 36 minutes watching animal videos.
-/
theorem brian_video_watching_time (cat_video dog_video gorilla_video : ℕ) 
  (h₁ : cat_video = 4) 
  (h₂ : dog_video = 2 * cat_video) 
  (h₃ : gorilla_video = 2 * (cat_video + dog_video)) : 
  cat_video + dog_video + gorilla_video = 36 := by
  sorry

end brian_video_watching_time_l834_83416


namespace find_number_l834_83497

theorem find_number (x : ℝ) (h : 0.4 * x + 60 = x) : x = 100 :=
by
  sorry

end find_number_l834_83497


namespace tan_11pi_over_6_l834_83445

theorem tan_11pi_over_6 : Real.tan (11 * Real.pi / 6) = - (Real.sqrt 3 / 3) :=
by
  sorry

end tan_11pi_over_6_l834_83445


namespace christine_wander_time_l834_83442

noncomputable def distance : ℝ := 80
noncomputable def speed : ℝ := 20
noncomputable def time : ℝ := distance / speed

theorem christine_wander_time : time = 4 := 
by
  sorry

end christine_wander_time_l834_83442


namespace trams_to_add_l834_83455

theorem trams_to_add (initial_trams : ℕ) (initial_interval new_interval : ℤ)
  (reduce_by_fraction : ℤ) (total_distance : ℤ)
  (h1 : initial_trams = 12)
  (h2 : initial_interval = total_distance / initial_trams)
  (h3 : reduce_by_fraction = 5)
  (h4 : new_interval = initial_interval - initial_interval / reduce_by_fraction) :
  initial_trams + (total_distance / new_interval - initial_trams) = 15 :=
by
  sorry

end trams_to_add_l834_83455


namespace benny_picked_proof_l834_83402

-- Define the number of apples Dan picked
def dan_picked: ℕ := 9

-- Define the total number of apples picked
def total_apples: ℕ := 11

-- Define the number of apples Benny picked
def benny_picked (dan_picked total_apples: ℕ): ℕ :=
  total_apples - dan_picked

-- The theorem we need to prove
theorem benny_picked_proof: benny_picked dan_picked total_apples = 2 :=
by
  -- We calculate the number of apples Benny picked
  sorry

end benny_picked_proof_l834_83402


namespace jennifer_total_discount_is_28_l834_83472

-- Define the conditions in the Lean context

def initial_whole_milk_cans : ℕ := 40 
def mark_whole_milk_cans : ℕ := 30 
def mark_skim_milk_cans : ℕ := 15 
def almond_milk_per_3_whole_milk : ℕ := 2 
def whole_milk_per_5_skim_milk : ℕ := 4 
def discount_per_10_whole_milk : ℕ := 4 
def discount_per_7_almond_milk : ℕ := 3 
def discount_per_3_almond_milk : ℕ := 1

def jennifer_additional_almond_milk := (mark_whole_milk_cans / 3) * almond_milk_per_3_whole_milk
def jennifer_additional_whole_milk := (mark_skim_milk_cans / 5) * whole_milk_per_5_skim_milk

def jennifer_whole_milk_cans := initial_whole_milk_cans + jennifer_additional_whole_milk
def jennifer_almond_milk_cans := jennifer_additional_almond_milk

def jennifer_whole_milk_discount := (jennifer_whole_milk_cans / 10) * discount_per_10_whole_milk
def jennifer_almond_milk_discount := 
  (jennifer_almond_milk_cans / 7) * discount_per_7_almond_milk + 
  ((jennifer_almond_milk_cans % 7) / 3) * discount_per_3_almond_milk

def total_jennifer_discount := jennifer_whole_milk_discount + jennifer_almond_milk_discount

-- Theorem stating the total discount 
theorem jennifer_total_discount_is_28 : total_jennifer_discount = 28 := by
  sorry

end jennifer_total_discount_is_28_l834_83472


namespace flower_bed_length_l834_83490

theorem flower_bed_length (a b : ℝ) :
  ∀ width : ℝ, (6 * a^2 - 4 * a * b + 2 * a = 2 * a * width) → width = 3 * a - 2 * b + 1 :=
by
  intros width h
  sorry

end flower_bed_length_l834_83490


namespace y_squared_plus_three_y_is_perfect_square_l834_83421

theorem y_squared_plus_three_y_is_perfect_square (y : ℕ) :
  (∃ x : ℕ, y^2 + 3^y = x^2) ↔ y = 1 ∨ y = 3 := 
by
  sorry

end y_squared_plus_three_y_is_perfect_square_l834_83421


namespace points_symmetric_about_x_axis_l834_83408

def point := ℝ × ℝ

def symmetric_x_axis (A B : point) : Prop :=
  A.1 = B.1 ∧ A.2 = -B.2

theorem points_symmetric_about_x_axis : symmetric_x_axis (-1, 3) (-1, -3) :=
by
  sorry

end points_symmetric_about_x_axis_l834_83408


namespace sin_x_sin_y_eq_sin_beta_sin_gamma_l834_83439

theorem sin_x_sin_y_eq_sin_beta_sin_gamma
  (A B C M : Type)
  (AM BM CM : ℝ)
  (alpha beta gamma x y : ℝ)
  (h1 : AM * AM = BM * CM)
  (h2 : BM ≠ 0)
  (h3 : CM ≠ 0)
  (hx : AM / BM = Real.sin beta / Real.sin x)
  (hy : AM / CM = Real.sin gamma / Real.sin y) :
  Real.sin x * Real.sin y = Real.sin beta * Real.sin gamma := 
sorry

end sin_x_sin_y_eq_sin_beta_sin_gamma_l834_83439


namespace min_value_frac_sum_l834_83443

variable {a b c : ℝ}

theorem min_value_frac_sum (h1 : 0 < a ∧ a < 1) (h2 : 0 < b ∧ b < 1) (h3 : 0 < c ∧ c < 1) (h4 : a * b + b * c + c * a = 1) : 
  (1 / (1 - a)) + (1 / (1 - b)) + (1 / (1 - c)) = (9 + 3 * Real.sqrt 3) / 2 :=
  sorry

end min_value_frac_sum_l834_83443


namespace person_time_to_walk_without_walkway_l834_83433

def time_to_walk_without_walkway 
  (walkway_length : ℝ) 
  (time_with_walkway : ℝ) 
  (time_against_walkway : ℝ) 
  (correct_time : ℝ) : Prop :=
  ∃ (vp vw : ℝ), 
    ((vp + vw) * time_with_walkway = walkway_length) ∧ 
    ((vp - vw) * time_against_walkway = walkway_length) ∧ 
     correct_time = walkway_length / vp

theorem person_time_to_walk_without_walkway : 
  time_to_walk_without_walkway 120 40 160 64 :=
sorry

end person_time_to_walk_without_walkway_l834_83433


namespace sequence_tenth_term_l834_83438

theorem sequence_tenth_term :
  ∃ (a : ℕ → ℚ), a 1 = 1 ∧ (∀ n : ℕ, n > 0 → a (n + 1) = a n / (1 + a n)) ∧ a 10 = 1 / 10 :=
sorry

end sequence_tenth_term_l834_83438


namespace trig_expression_value_l834_83413

theorem trig_expression_value : 
  (2 * (Real.sin (25 * Real.pi / 180))^2 - 1) / 
  (Real.sin (20 * Real.pi / 180) * Real.cos (20 * Real.pi / 180)) = -2 := 
by
  -- Proof goes here
  sorry

end trig_expression_value_l834_83413


namespace meeting_time_l834_83411

-- Definitions for the problem conditions.
def track_length : ℕ := 1800
def speed_A_kmph : ℕ := 36
def speed_B_kmph : ℕ := 54

-- Conversion factor from kmph to mps.
def kmph_to_mps (speed_kmph : ℕ) : ℕ := (speed_kmph * 1000) / 3600

-- Calculate the speeds in mps.
def speed_A_mps : ℕ := kmph_to_mps speed_A_kmph
def speed_B_mps : ℕ := kmph_to_mps speed_B_kmph

-- Calculate the time to complete one lap for A and B.
def time_lap_A : ℕ := track_length / speed_A_mps
def time_lap_B : ℕ := track_length / speed_B_mps

-- Prove the time to meet at the starting point.
theorem meeting_time : (Nat.lcm time_lap_A time_lap_B) = 360 := by
  -- Skipping the proof with sorry placeholder
  sorry

end meeting_time_l834_83411


namespace find_rate_per_kg_mangoes_l834_83410

-- Definitions based on the conditions
def rate_per_kg_grapes : ℕ := 70
def quantity_grapes : ℕ := 8
def total_payment : ℕ := 1000
def quantity_mangoes : ℕ := 8

-- Proposition stating what we want to prove
theorem find_rate_per_kg_mangoes (r : ℕ) (H : total_payment = (rate_per_kg_grapes * quantity_grapes) + (r * quantity_mangoes)) : r = 55 := sorry

end find_rate_per_kg_mangoes_l834_83410


namespace area_of_given_triangle_is_8_l834_83451

-- Define the vertices of the triangle
def x1 := 2
def y1 := -3
def x2 := -1
def y2 := 6
def x3 := 4
def y3 := -5

-- Define the determinant formula for the area of the triangle
def area_of_triangle (x1 y1 x2 y2 x3 y3 : ℤ) : ℤ :=
  abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2

theorem area_of_given_triangle_is_8 :
  area_of_triangle x1 y1 x2 y2 x3 y3 = 8 := by
  sorry

end area_of_given_triangle_is_8_l834_83451


namespace calculate_xy_l834_83435

theorem calculate_xy (x y : ℝ) (h1 : x + y = 12) (h2 : 3 * x + y = 20) : x * y = 32 :=
by
  sorry

end calculate_xy_l834_83435


namespace remaining_customers_is_13_l834_83419

-- Given conditions
def initial_customers : ℕ := 36
def half_left_customers : ℕ := initial_customers / 2  -- 50% of customers leaving
def remaining_customers_after_half_left : ℕ := initial_customers - half_left_customers

def thirty_percent_of_remaining : ℚ := remaining_customers_after_half_left * 0.30 
def thirty_percent_of_remaining_rounded : ℕ := thirty_percent_of_remaining.floor.toNat  -- rounding down

def final_remaining_customers : ℕ := remaining_customers_after_half_left - thirty_percent_of_remaining_rounded

-- Proof statement without proof
theorem remaining_customers_is_13 : final_remaining_customers = 13 := by
  sorry

end remaining_customers_is_13_l834_83419


namespace change_in_mean_and_median_l834_83476

-- Original attendance data
def original_data : List ℕ := [15, 23, 17, 19, 17, 20]

-- Corrected attendance data
def corrected_data : List ℕ := [15, 23, 17, 19, 17, 25]

-- Function to compute mean
def mean (data: List ℕ) : ℚ := (data.sum : ℚ) / data.length

-- Function to compute median
def median (data: List ℕ) : ℚ :=
  let sorted := data.toArray.qsort (· ≤ ·) |>.toList
  if sorted.length % 2 == 0 then
    (sorted.get! (sorted.length / 2 - 1) + sorted.get! (sorted.length / 2)) / 2
  else
    sorted.get! (sorted.length / 2)

-- Lean statement verifying the expected change in mean and median
theorem change_in_mean_and_median :
  mean corrected_data - mean original_data = 1 ∧ median corrected_data = median original_data :=
by -- Note the use of 'by' to structure the proof
  sorry -- Proof omitted

end change_in_mean_and_median_l834_83476


namespace chord_intersection_probability_l834_83406

noncomputable def probability_chord_intersection : ℚ :=
1 / 3

theorem chord_intersection_probability 
    (A B C D : ℕ) 
    (total_points : ℕ) 
    (adjacent : A + 1 = B ∨ A = B + 1)
    (distinct : ∀ (A B C D : ℕ), A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
    (points_on_circle : total_points = 2023) :
    ∃ p : ℚ, p = probability_chord_intersection :=
by sorry

end chord_intersection_probability_l834_83406


namespace face_value_is_100_l834_83459

-- Definitions based on conditions
def faceValue (F : ℝ) : Prop :=
  let discountedPrice := 0.92 * F
  let brokerageFee := 0.002 * discountedPrice
  let totalCostPrice := discountedPrice + brokerageFee
  totalCostPrice = 92.2

-- The proof statement in Lean
theorem face_value_is_100 : ∃ F : ℝ, faceValue F ∧ F = 100 :=
by
  use 100
  unfold faceValue
  simp
  norm_num
  sorry

end face_value_is_100_l834_83459


namespace solve_for_y_l834_83444

theorem solve_for_y (y : ℝ) (h : (4/7) * (1/5) * y - 2 = 14) : y = 140 := 
sorry

end solve_for_y_l834_83444


namespace number_of_trousers_given_l834_83414

-- Define the conditions
def shirts_given : Nat := 589
def total_clothing_given : Nat := 934

-- Define the expected answer
def expected_trousers_given : Nat := 345

-- The theorem statement to prove the number of trousers given
theorem number_of_trousers_given : total_clothing_given - shirts_given = expected_trousers_given :=
by
  sorry

end number_of_trousers_given_l834_83414


namespace hotpot_total_cost_l834_83405

def table_cost : ℝ := 280
def table_limit : ℕ := 8
def extra_person_cost : ℝ := 29.9
def total_people : ℕ := 12

theorem hotpot_total_cost : 
  total_people > table_limit →
  table_cost + (total_people - table_limit) * extra_person_cost = 369.7 := 
by 
  sorry

end hotpot_total_cost_l834_83405


namespace positive_X_solution_l834_83429

def boxtimes (X Y : ℤ) : ℤ := X^2 - 2 * X + Y^2

theorem positive_X_solution (X : ℤ) (h : boxtimes X 7 = 164) : X = 13 :=
by
  sorry

end positive_X_solution_l834_83429


namespace ratio_B_C_l834_83464

def total_money := 595
def A_share := 420
def B_share := 105
def C_share := 70

-- The main theorem stating the expected ratio
theorem ratio_B_C : (B_share / C_share : ℚ) = 3 / 2 := by
  sorry

end ratio_B_C_l834_83464


namespace ratio_of_autobiographies_to_fiction_l834_83465

theorem ratio_of_autobiographies_to_fiction (total_books fiction_books non_fiction_books picture_books autobiographies: ℕ) 
  (h1 : total_books = 35) 
  (h2 : fiction_books = 5) 
  (h3 : non_fiction_books = fiction_books + 4) 
  (h4 : picture_books = 11) 
  (h5 : autobiographies = total_books - (fiction_books + non_fiction_books + picture_books)) :
  autobiographies / fiction_books = 2 :=
by sorry

end ratio_of_autobiographies_to_fiction_l834_83465


namespace blue_black_pen_ratio_l834_83484

theorem blue_black_pen_ratio (B K R : ℕ) 
  (h1 : B + K + R = 31) 
  (h2 : B = 18) 
  (h3 : K = R + 5) : 
  B / Nat.gcd B K = 2 ∧ K / Nat.gcd B K = 1 := 
by 
  sorry

end blue_black_pen_ratio_l834_83484


namespace number_of_monomials_l834_83431

-- Define the degree of a monomial
def degree (x_deg y_deg z_deg : ℕ) : ℕ := x_deg + y_deg + z_deg

-- Define a condition for the coefficient of the monomial
def monomial_coefficient (coeff : ℤ) : Prop := coeff = -3

-- Define a condition for the presence of the variables x, y, z
def contains_vars (x_deg y_deg z_deg : ℕ) : Prop := x_deg ≥ 1 ∧ y_deg ≥ 1 ∧ z_deg ≥ 1

-- Define the proof for the number of such monomials
theorem number_of_monomials :
  ∃ (x_deg y_deg z_deg : ℕ), contains_vars x_deg y_deg z_deg ∧ monomial_coefficient (-3) ∧ degree x_deg y_deg z_deg = 5 ∧ (6 = 6) :=
by
  sorry

end number_of_monomials_l834_83431


namespace polygon_sides_given_ratio_l834_83426

theorem polygon_sides_given_ratio (n : ℕ) 
  (h : (n - 2) * 180 / 360 = 9 / 2) : n = 11 :=
sorry

end polygon_sides_given_ratio_l834_83426


namespace no_attention_prob_l834_83436

noncomputable def prob_no_attention (p1 p2 p3 : ℝ) : ℝ :=
  (1 - p1) * (1 - p2) * (1 - p3)

theorem no_attention_prob :
  let p1 := 0.9
  let p2 := 0.8
  let p3 := 0.6
  prob_no_attention p1 p2 p3 = 0.008 :=
by
  unfold prob_no_attention
  sorry

end no_attention_prob_l834_83436


namespace extrema_range_of_m_l834_83448

def has_extrema (f : ℝ → ℝ) : Prop :=
  ∃ x y : ℝ, (∀ z : ℝ, z ≤ x → f z ≤ f x) ∧ (∀ z : ℝ, z ≥ y → f z ≤ f y)

noncomputable def f (m x : ℝ) : ℝ :=
  x^3 + m * x^2 + (m + 6) * x + 1

theorem extrema_range_of_m (m : ℝ) :
  has_extrema (f m) ↔ (m ∈ Set.Iic (-3) ∪ Set.Ici 6) :=
by
  sorry

end extrema_range_of_m_l834_83448


namespace total_cost_bicycle_helmet_l834_83400

-- Let h represent the cost of the helmet
def helmet_cost := 40

-- Let b represent the cost of the bicycle
def bicycle_cost := 5 * helmet_cost

-- We need to prove that the total cost (bicycle + helmet) is equal to 240
theorem total_cost_bicycle_helmet : bicycle_cost + helmet_cost = 240 := 
by
  -- This will skip the proof, we only need the statement
  sorry

end total_cost_bicycle_helmet_l834_83400


namespace yellow_not_greater_than_green_l834_83489

theorem yellow_not_greater_than_green
    (G Y S : ℕ)
    (h1 : G + Y + S = 100)
    (h2 : G + S / 2 = 50)
    (h3 : Y + S / 2 = 50) : ¬ Y > G :=
sorry

end yellow_not_greater_than_green_l834_83489


namespace lemonade_water_quarts_l834_83485

theorem lemonade_water_quarts :
  let ratioWaterLemon := (4 : ℕ) / (1 : ℕ)
  let totalParts := 4 + 1
  let totalVolumeInGallons := 3
  let quartsPerGallon := 4
  let totalVolumeInQuarts := totalVolumeInGallons * quartsPerGallon
  let volumePerPart := totalVolumeInQuarts / totalParts
  let volumeWater := 4 * volumePerPart
  volumeWater = 9.6 :=
by
  -- placeholder for actual proof
  sorry

end lemonade_water_quarts_l834_83485


namespace average_speed_l834_83428

theorem average_speed (s₁ s₂ s₃ s₄ s₅ : ℝ) (h₁ : s₁ = 85) (h₂ : s₂ = 45) (h₃ : s₃ = 60) (h₄ : s₄ = 75) (h₅ : s₅ = 50) : 
  (s₁ + s₂ + s₃ + s₄ + s₅) / 5 = 63 := 
by 
  sorry

end average_speed_l834_83428
