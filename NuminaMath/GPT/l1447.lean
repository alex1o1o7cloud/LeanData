import Mathlib

namespace functions_same_l1447_144742

theorem functions_same (x : ℝ) : (∀ x, (y = x) → (∀ x, (y = (x^3 + x) / (x^2 + 1)))) :=
by sorry

end functions_same_l1447_144742


namespace sum_of_first_21_terms_l1447_144769

def is_constant_sum_sequence (a : ℕ → ℕ) (c : ℕ) : Prop :=
  ∀ n, a n + a (n + 1) = c

theorem sum_of_first_21_terms (a : ℕ → ℕ) (h1 : is_constant_sum_sequence a 5) (h2 : a 1 = 2) : (Finset.range 21).sum a = 52 :=
by
  sorry

end sum_of_first_21_terms_l1447_144769


namespace problem_1_problem_2_l1447_144799

noncomputable def a : ℝ := sorry
def m : ℝ := sorry
def n : ℝ := sorry
def k : ℝ := sorry

theorem problem_1 (h1 : a^m = 2) (h2 : a^n = 4) (h3 : a^k = 32) (h4 : a ≠ 0) : 
  a^(3*m + 2*n - k) = 4 := 
sorry

theorem problem_2 (h1 : a^m = 2) (h2 : a^n = 4) (h3 : a^k = 32) (h4 : a ≠ 0) : 
  k - 3*m - n = 0 := 
sorry

end problem_1_problem_2_l1447_144799


namespace find_y_l1447_144793

variable {L B y : ℝ}

theorem find_y (h1 : 2 * ((L + y) + (B + y)) - 2 * (L + B) = 16) : y = 4 :=
by
  sorry

end find_y_l1447_144793


namespace inequality_solution_l1447_144749

theorem inequality_solution (x : ℝ) : 2 * x - 1 ≤ 3 → x ≤ 2 :=
by
  intro h
  -- Here we would perform the solution steps, but we'll skip the proof with sorry.
  sorry

end inequality_solution_l1447_144749


namespace polynomial_solution_l1447_144729

theorem polynomial_solution (P : ℝ → ℝ) (hP : ∀ x : ℝ, (x + 1) * P (x - 1) + (x - 1) * P (x + 1) = 2 * x * P x) :
  ∃ (a d : ℝ), ∀ x : ℝ, P x = a * x^3 - a * x + d := 
sorry

end polynomial_solution_l1447_144729


namespace min_shift_sine_l1447_144797

theorem min_shift_sine (φ : ℝ) (hφ : φ > 0) :
    (∃ k : ℤ, 2 * φ + π / 3 = 2 * k * π) → φ = 5 * π / 6 :=
sorry

end min_shift_sine_l1447_144797


namespace total_games_played_l1447_144786

-- Definition of the conditions
def teams : Nat := 10
def games_per_pair : Nat := 4

-- Statement of the problem
theorem total_games_played (teams games_per_pair : Nat) : 
  teams = 10 → 
  games_per_pair = 4 → 
  ∃ total_games, total_games = 180 :=
by
  intro h1 h2
  sorry

end total_games_played_l1447_144786


namespace work_duration_B_l1447_144717

theorem work_duration_B (x : ℕ) (h : x = 10) : 
  (x * (1 / 15 : ℚ)) + (2 * (1 / 6 : ℚ)) = 1 := 
by 
  rw [h]
  sorry

end work_duration_B_l1447_144717


namespace largest_divisor_of_even_diff_squares_l1447_144778

theorem largest_divisor_of_even_diff_squares (m n : ℤ) (h_m_even : ∃ k : ℤ, m = 2 * k) (h_n_even : ∃ k : ℤ, n = 2 * k) (h_n_lt_m : n < m) : 
  ∃ d : ℤ, d = 16 ∧ ∀ p : ℤ, (p ∣ (m^2 - n^2)) → p ≤ d :=
sorry

end largest_divisor_of_even_diff_squares_l1447_144778


namespace toy_swords_count_l1447_144773

variable (s : ℕ)

def cost_lego := 250
def cost_toy_sword := 120
def cost_play_dough := 35

def total_cost (s : ℕ) :=
  3 * cost_lego + s * cost_toy_sword + 10 * cost_play_dough

theorem toy_swords_count : total_cost s = 1940 → s = 7 := by
  sorry

end toy_swords_count_l1447_144773


namespace algebra_inequality_l1447_144756

theorem algebra_inequality (a : ℝ) : 
  (∃ x : ℝ, 1 < x ∧ x < 4 ∧ 2 * x ^ 2 - 8 * x - 4 - a > 0) → a < -4 :=
by
  sorry

end algebra_inequality_l1447_144756


namespace angle_diff_complement_supplement_l1447_144721

theorem angle_diff_complement_supplement (α : ℝ) : (180 - α) - (90 - α) = 90 := by
  sorry

end angle_diff_complement_supplement_l1447_144721


namespace people_not_in_pool_l1447_144798

-- Define families and their members
def karen_donald_family : ℕ := 2 + 6
def tom_eva_family : ℕ := 2 + 4
def luna_aidan_family : ℕ := 2 + 5
def isabel_jake_family : ℕ := 2 + 3

-- Total number of people
def total_people : ℕ := karen_donald_family + tom_eva_family + luna_aidan_family + isabel_jake_family

-- Number of legs in the pool and people in the pool
def legs_in_pool : ℕ := 34
def people_in_pool : ℕ := legs_in_pool / 2

-- People not in the pool: people who went to store and went to bed
def store_people : ℕ := 2
def bed_people : ℕ := 3
def not_available_people : ℕ := store_people + bed_people

-- Prove (given conditions) number of people not in the pool
theorem people_not_in_pool : total_people - people_in_pool - not_available_people = 4 :=
by
  -- ...proof steps or "sorry"
  sorry

end people_not_in_pool_l1447_144798


namespace vector_field_lines_l1447_144714

noncomputable def vector_lines : Prop :=
  ∃ (C_1 C_2 : ℝ), ∀ (x y z : ℝ), (9 * z^2 + 4 * y^2 = C_1) ∧ (x = C_2)

-- We state the proof goal as follows:
theorem vector_field_lines :
  ∀ (a : ℝ × ℝ × ℝ → ℝ × ℝ × ℝ), 
    (∀ (x y z : ℝ), a (x, y, z) = (0, 9 * z, -4 * y)) →
    vector_lines :=
by
  intro a ha
  sorry

end vector_field_lines_l1447_144714


namespace glass_ball_radius_l1447_144752

theorem glass_ball_radius (x y r : ℝ) (h_parabola : x^2 = 2 * y) (h_touch : y = r) (h_range : 0 ≤ y ∧ y ≤ 20) : 0 < r ∧ r ≤ 1 :=
sorry

end glass_ball_radius_l1447_144752


namespace inequality1_inequality2_l1447_144767

theorem inequality1 (x : ℝ) : 
  x^2 - 2 * x - 1 > 0 -> x > Real.sqrt 2 + 1 ∨ x < -Real.sqrt 2 + 1 := 
by sorry

theorem inequality2 (x : ℝ) : 
  (2 * x - 1) / (x - 3) ≥ 3 -> 3 < x ∧ x <= 8 := 
by sorry

end inequality1_inequality2_l1447_144767


namespace gcd_15_70_l1447_144726

theorem gcd_15_70 : Int.gcd 15 70 = 5 := by
  sorry

end gcd_15_70_l1447_144726


namespace cos_sum_of_arctan_roots_l1447_144753

theorem cos_sum_of_arctan_roots (α β : ℝ) (hα : -π/2 < α ∧ α < 0) (hβ : -π/2 < β ∧ β < 0) 
  (h1 : Real.tan α + Real.tan β = -3 * Real.sqrt 3) 
  (h2 : Real.tan α * Real.tan β = 4) : 
  Real.cos (α + β) = - 1 / 2 :=
sorry

end cos_sum_of_arctan_roots_l1447_144753


namespace residue_of_minus_963_plus_100_mod_35_l1447_144750

-- Defining the problem in Lean 4
theorem residue_of_minus_963_plus_100_mod_35 : 
  ((-963 + 100) % 35) = 12 :=
by
  sorry

end residue_of_minus_963_plus_100_mod_35_l1447_144750


namespace mask_digit_correctness_l1447_144790

noncomputable def elephant_mask_digit : ℕ :=
  6
  
noncomputable def mouse_mask_digit : ℕ :=
  4

noncomputable def guinea_pig_mask_digit : ℕ :=
  8

noncomputable def panda_mask_digit : ℕ :=
  1

theorem mask_digit_correctness :
  (∃ (d1 d2 d3 d4 : ℕ), d1 * d1 = 16 ∧ d2 * d2 = 64 ∧ d3 * d3 = 49 ∧ d4 * d4 = 81) →
  elephant_mask_digit = 6 ∧ mouse_mask_digit = 4 ∧ guinea_pig_mask_digit = 8 ∧ panda_mask_digit = 1 :=
by
  -- skip the proof
  sorry

end mask_digit_correctness_l1447_144790


namespace umar_age_is_ten_l1447_144731

-- Define variables for Ali, Yusaf, and Umar
variables (ali_age yusa_age umar_age : ℕ)

-- Define the conditions from the problem
def ali_is_eight : Prop := ali_age = 8
def ali_older_than_yusaf : Prop := ali_age - yusa_age = 3
def umar_twice_yusaf : Prop := umar_age = 2 * yusa_age

-- The theorem that uses the conditions to assert Umar's age
theorem umar_age_is_ten 
  (h1 : ali_is_eight ali_age)
  (h2 : ali_older_than_yusaf ali_age yusa_age)
  (h3 : umar_twice_yusaf umar_age yusa_age) : 
  umar_age = 10 :=
by
  sorry

end umar_age_is_ten_l1447_144731


namespace merchant_should_choose_option2_l1447_144719

-- Definitions for the initial price and discounts
def P : ℝ := 20000
def d1_1 : ℝ := 0.25
def d1_2 : ℝ := 0.15
def d1_3 : ℝ := 0.05
def d2_1 : ℝ := 0.35
def d2_2 : ℝ := 0.10
def d2_3 : ℝ := 0.05

-- Define the final prices after applying discount options
def finalPrice1 (P : ℝ) (d1_1 d1_2 d1_3 : ℝ) : ℝ :=
  P * (1 - d1_1) * (1 - d1_2) * (1 - d1_3)

def finalPrice2 (P : ℝ) (d2_1 d2_2 d2_3 : ℝ) : ℝ :=
  P * (1 - d2_1) * (1 - d2_2) * (1 - d2_3)

-- Theorem to state the merchant should choose Option 2
theorem merchant_should_choose_option2 : 
  finalPrice1 P d1_1 d1_2 d1_3 = 12112.50 ∧ 
  finalPrice2 P d2_1 d2_2 d2_3 = 11115 ∧ 
  finalPrice1 P d1_1 d1_2 d1_3 - finalPrice2 P d2_1 d2_2 d2_3 = 997.50 :=
by
  -- Placeholder for the proof
  sorry

end merchant_should_choose_option2_l1447_144719


namespace multiples_of_231_l1447_144779

theorem multiples_of_231 (h : ∀ i j : ℕ, 1 ≤ i → i < j → j ≤ 99 → i % 2 = 1 → 231 ∣ 10^j - 10^i) :
  ∃ n, n = 416 :=
by sorry

end multiples_of_231_l1447_144779


namespace f_bound_l1447_144736

noncomputable def f : ℕ+ → ℝ := sorry

axiom f_1 : f 1 = 3 / 2
axiom f_ineq (x y : ℕ+) : f (x + y) ≥ (1 + y / (x + 1)) * f x + (1 + x / (y + 1)) * f y + x^2 * y + x * y + x * y^2

theorem f_bound (x : ℕ+) : f x ≥ 1 / 4 * x * (x + 1) * (2 * x + 1) := sorry

end f_bound_l1447_144736


namespace smallest_perfect_square_greater_l1447_144725

theorem smallest_perfect_square_greater (a : ℕ) (h : ∃ n : ℕ, a = n^2) : 
  ∃ m : ℕ, m^2 > a ∧ ∀ k : ℕ, k^2 > a → m^2 ≤ k^2 :=
  sorry

end smallest_perfect_square_greater_l1447_144725


namespace remaining_subtasks_l1447_144745

def total_problems : ℝ := 72.0
def finished_problems : ℝ := 32.0
def subtasks_per_problem : ℕ := 5

theorem remaining_subtasks :
    (total_problems * subtasks_per_problem - finished_problems * subtasks_per_problem) = 200 := 
by
  sorry

end remaining_subtasks_l1447_144745


namespace percentage_increase_after_decrease_l1447_144782

theorem percentage_increase_after_decrease (P : ℝ) :
  let P_decreased := 0.70 * P
  let P_final := 1.16 * P
  let x := ((P_final / P_decreased) - 1) * 100
  (P_decreased * (1 + x / 100) = P_final) → x = 65.71 := 
by 
  intros
  let P_decreased := 0.70 * P
  let P_final := 1.16 * P
  let x := ((P_final / P_decreased) - 1) * 100
  have h : (P_decreased * (1 + x / 100) = P_final) := by assumption
  sorry

end percentage_increase_after_decrease_l1447_144782


namespace problem_proof_l1447_144785

noncomputable def binomial (n k : ℕ) : ℕ :=
if h : k ≤ n then Nat.choose n k else 0

noncomputable def probability_ratio_pq : ℕ :=
let p := binomial 10 2 * binomial 30 2 * binomial 28 2
let q := binomial 30 3 * binomial 27 3 * binomial 24 3 * binomial 21 3 * binomial 18 3 * binomial 15 3 * binomial 12 3 * binomial 9 3 * binomial 6 3 * binomial 3 3
p / (q / (binomial 30 3 * binomial 27 3 * binomial 24 3 * binomial 21 3 * binomial 18 3 * binomial 15 3 * binomial 12 3 * binomial 9 3 * binomial 6 3 * binomial 3 3))

theorem problem_proof :
  probability_ratio_pq = 7371 :=
sorry

end problem_proof_l1447_144785


namespace sport_tournament_attendance_l1447_144781

theorem sport_tournament_attendance :
  let total_attendance := 500
  let team_A_supporters := 0.35 * total_attendance
  let team_B_supporters := 0.25 * total_attendance
  let team_C_supporters := 0.20 * total_attendance
  let team_D_supporters := 0.15 * total_attendance
  let AB_overlap := 0.10 * team_A_supporters
  let BC_overlap := 0.05 * team_B_supporters
  let CD_overlap := 0.07 * team_C_supporters
  let atmosphere_attendees := 30
  let total_supporters := team_A_supporters + team_B_supporters + team_C_supporters + team_D_supporters
                         - (AB_overlap + BC_overlap + CD_overlap)
  let unsupported_people := total_attendance - total_supporters - atmosphere_attendees
  unsupported_people = 26 :=
by
  sorry

end sport_tournament_attendance_l1447_144781


namespace quadratic_expression_value_l1447_144764

theorem quadratic_expression_value (a b : ℝ) (h₁ : a ≠ 0) (h₂ : a + b - 1 = 1) : (1 - a - b) = -1 :=
sorry

end quadratic_expression_value_l1447_144764


namespace seating_arrangements_total_l1447_144735

def num_round_tables := 3
def num_rect_tables := 4
def num_square_tables := 2
def num_couches := 2
def num_benches := 3
def num_extra_chairs := 5

def seats_per_round_table := 6
def seats_per_rect_table := 7
def seats_per_square_table := 4
def seats_per_couch := 3
def seats_per_bench := 5

def total_seats : Nat :=
  num_round_tables * seats_per_round_table +
  num_rect_tables * seats_per_rect_table +
  num_square_tables * seats_per_square_table +
  num_couches * seats_per_couch +
  num_benches * seats_per_bench +
  num_extra_chairs

theorem seating_arrangements_total :
  total_seats = 80 :=
by
  simp [total_seats, num_round_tables, seats_per_round_table,
        num_rect_tables, seats_per_rect_table, num_square_tables,
        seats_per_square_table, num_couches, seats_per_couch,
        num_benches, seats_per_bench, num_extra_chairs]
  done

end seating_arrangements_total_l1447_144735


namespace how_much_does_c_have_l1447_144712

theorem how_much_does_c_have (A B C : ℝ) (h1 : A + B + C = 400) (h2 : A + C = 300) (h3 : B + C = 150) : C = 50 :=
by
  sorry

end how_much_does_c_have_l1447_144712


namespace min_value_a_plus_3b_l1447_144728

theorem min_value_a_plus_3b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 3 * a * b - 3 = a + 3 * b) :
  ∃ x : ℝ, x = 6 ∧ ∀ y : ℝ, y = a + 3 * b → y ≥ 6 :=
sorry

end min_value_a_plus_3b_l1447_144728


namespace prime_ge_5_div_24_l1447_144794

theorem prime_ge_5_div_24 (p : ℕ) (hp : Prime p) (hp_ge_5 : p ≥ 5) : 24 ∣ p^2 - 1 := 
sorry

end prime_ge_5_div_24_l1447_144794


namespace property_check_l1447_144732

noncomputable def f (x : ℝ) : ℤ := ⌈x⌉ -- Define the ceiling function

theorem property_check :
  (¬ (∀ x : ℝ, f (2 * x) = 2 * f x)) ∧
  (∀ x1 x2 : ℝ, f x1 = f x2 → |x1 - x2| < 1) ∧
  (∀ x1 x2 : ℝ, f (x1 + x2) ≤ f x1 + f x2) ∧
  (¬ (∀ x : ℝ, f x + f (x + 0.5) = f (2 * x))) :=
by
  sorry

end property_check_l1447_144732


namespace false_statements_l1447_144758

variable (a b c : ℝ)

theorem false_statements (a b c : ℝ) :
  ¬(a > b → a^2 > b^2) ∧ ¬((a^2 > b^2) → a > b) ∧ ¬(a > b → a * c^2 > b * c^2) ∧ ¬(a > b ↔ |a| > |b|) :=
by
  sorry

end false_statements_l1447_144758


namespace Thursday_total_rainfall_correct_l1447_144707

def Monday_rainfall : ℝ := 0.9
def Tuesday_rainfall : ℝ := Monday_rainfall - 0.7
def Wednesday_rainfall : ℝ := Tuesday_rainfall + 0.5 * Tuesday_rainfall
def additional_rain : ℝ := 0.3
def decrease_factor : ℝ := 0.2
def Thursday_rainfall_before_addition : ℝ := Wednesday_rainfall - decrease_factor * Wednesday_rainfall
def Thursday_total_rainfall : ℝ := Thursday_rainfall_before_addition + additional_rain

theorem Thursday_total_rainfall_correct :
  Thursday_total_rainfall = 0.54 :=
by
  sorry

end Thursday_total_rainfall_correct_l1447_144707


namespace paul_spent_374_43_l1447_144701

noncomputable def paul_total_cost_after_discounts : ℝ :=
  let dress_shirts := 4 * 15.00
  let discount_dress_shirts := dress_shirts * 0.20
  let cost_dress_shirts := dress_shirts - discount_dress_shirts
  
  let pants := 2 * 40.00
  let discount_pants := pants * 0.30
  let cost_pants := pants - discount_pants
  
  let suit := 150.00
  
  let sweaters := 2 * 30.00
  
  let ties := 3 * 20.00
  let discount_tie := 20.00 * 0.50
  let cost_ties := 20.00 + (20.00 - discount_tie) + 20.00

  let shoes := 80.00
  let discount_shoes := shoes * 0.25
  let cost_shoes := shoes - discount_shoes

  let total_after_discounts := cost_dress_shirts + cost_pants + suit + sweaters + cost_ties + cost_shoes
  
  let total_after_coupon := total_after_discounts * 0.90
  
  let total_after_rewards := total_after_coupon - (500 * 0.05)
  
  let total_after_tax := total_after_rewards * 1.05
  
  total_after_tax

theorem paul_spent_374_43 :
  paul_total_cost_after_discounts = 374.43 :=
by
  sorry

end paul_spent_374_43_l1447_144701


namespace stephanie_quarters_fraction_l1447_144700

/-- Stephanie has a collection containing exactly one of the first 25 U.S. state quarters. 
    The quarters are in the order the states joined the union.
    Suppose 8 states joined the union between 1800 and 1809. -/
theorem stephanie_quarters_fraction :
  (8 / 25 : ℚ) = (8 / 25) :=
by
  sorry

end stephanie_quarters_fraction_l1447_144700


namespace employees_paid_per_shirt_l1447_144768

theorem employees_paid_per_shirt:
  let num_employees := 20
  let shirts_per_employee_per_day := 20
  let hours_per_shift := 8
  let wage_per_hour := 12
  let price_per_shirt := 35
  let nonemployee_expenses_per_day := 1000
  let profit_per_day := 9080
  let total_shirts_made_per_day := num_employees * shirts_per_employee_per_day
  let total_daily_wages := num_employees * hours_per_shift * wage_per_hour
  let total_revenue := total_shirts_made_per_day * price_per_shirt
  let per_shirt_payment := (total_revenue - (total_daily_wages + nonemployee_expenses_per_day)) / total_shirts_made_per_day
  per_shirt_payment = 27.70 :=
sorry

end employees_paid_per_shirt_l1447_144768


namespace robin_hid_150_seeds_l1447_144730

theorem robin_hid_150_seeds
    (x y : ℕ)
    (h1 : 5 * x = 6 * y)
    (h2 : y = x - 5) : 
    5 * x = 150 :=
by
    sorry

end robin_hid_150_seeds_l1447_144730


namespace three_digit_number_550_l1447_144760

theorem three_digit_number_550 (N : ℕ) (a b c : ℕ) (h1 : N = 100 * a + 10 * b + c)
  (h2 : 1 ≤ a ∧ a ≤ 9) (h3 : b ≤ 9) (h4 : c ≤ 9) (h5 : 11 ∣ N)
  (h6 : N / 11 = a^2 + b^2 + c^2) : N = 550 :=
by
  sorry

end three_digit_number_550_l1447_144760


namespace plate_and_roller_acceleration_l1447_144723

noncomputable def m : ℝ := 150
noncomputable def g : ℝ := 10
noncomputable def R : ℝ := 1
noncomputable def r : ℝ := 0.4
noncomputable def alpha : ℝ := Real.arccos 0.68

theorem plate_and_roller_acceleration :
  let sin_alpha_half := Real.sin (alpha / 2)
  sin_alpha_half = 0.4 →
  plate_acceleration == 4 ∧ direction == Real.arcsin 0.4 ∧ rollers_acceleration == 4 :=
by
  sorry

end plate_and_roller_acceleration_l1447_144723


namespace union_complement_eq_universal_l1447_144718

-- Definitions of the sets
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {3, 5}

-- The proof problem
theorem union_complement_eq_universal :
  U = A ∪ (U \ B) :=
by
  sorry

end union_complement_eq_universal_l1447_144718


namespace distance_between_two_girls_after_12_hours_l1447_144744

theorem distance_between_two_girls_after_12_hours :
  let speed1 := 7 -- speed of the first girl (km/hr)
  let speed2 := 3 -- speed of the second girl (km/hr)
  let time := 12 -- time (hours)
  let distance1 := speed1 * time -- distance traveled by the first girl
  let distance2 := speed2 * time -- distance traveled by the second girl
  distance1 + distance2 = 120 := -- total distance
by
  -- Here, we would provide the proof, but we put sorry to skip it
  sorry

end distance_between_two_girls_after_12_hours_l1447_144744


namespace inequality_condition_l1447_144738

theorem inequality_condition (a : ℝ) : 
  (∀ x y : ℝ, x^2 + 2 * x + a ≥ -y^2 - 2 * y) → a ≥ 2 :=
by
  sorry

end inequality_condition_l1447_144738


namespace zero_unique_multiple_prime_l1447_144777

-- Condition: let n be a number
def n : Int := sorry

-- Condition: let p be any prime number
def is_prime (p : Int) : Prop := sorry  -- Predicate definition for prime number

-- Proof problem statement
theorem zero_unique_multiple_prime (n : Int) :
  (∀ p : Int, is_prime p → (∃ k : Int, n * p = k * p)) ↔ (n = 0) := by
  sorry

end zero_unique_multiple_prime_l1447_144777


namespace part1_part2_l1447_144741

-- Problem 1: Given |x| = 9, |y| = 5, x < 0, y > 0, prove x + y = -4
theorem part1 (x y : ℚ) (h1 : |x| = 9) (h2 : |y| = 5) (h3 : x < 0) (h4 : y > 0) : x + y = -4 :=
sorry

-- Problem 2: Given |x| = 9, |y| = 5, |x + y| = x + y, prove x - y = 4 or x - y = 14
theorem part2 (x y : ℚ) (h1 : |x| = 9) (h2 : |y| = 5) (h3 : |x + y| = x + y) : x - y = 4 ∨ x - y = 14 :=
sorry

end part1_part2_l1447_144741


namespace value_of_y_for_absolute_value_eq_zero_l1447_144791

theorem value_of_y_for_absolute_value_eq_zero :
  ∃ (y : ℚ), |(2:ℚ) * y - 3| ≤ 0 ↔ y = 3 / 2 :=
by
  sorry

end value_of_y_for_absolute_value_eq_zero_l1447_144791


namespace students_gold_award_freshmen_l1447_144770

theorem students_gold_award_freshmen 
    (total_students total_award_winners : ℕ)
    (students_selected exchange_meeting : ℕ)
    (freshmen_selected gold_award_selected : ℕ)
    (prop1 : total_award_winners = 120)
    (prop2 : exchange_meeting = 24)
    (prop3 : freshmen_selected = 6)
    (prop4 : gold_award_selected = 4) :
    ∃ (gold_award_students : ℕ), gold_award_students = 4 ∧ gold_award_students ≤ freshmen_selected :=
by
  sorry

end students_gold_award_freshmen_l1447_144770


namespace algebraic_expression_value_l1447_144788

theorem algebraic_expression_value (x y : ℝ) (h1 : x * y = -2) (h2 : x + y = 4) : x^2 * y + x * y^2 = -8 := 
by
  sorry

end algebraic_expression_value_l1447_144788


namespace circumference_given_area_l1447_144716

noncomputable def area_of_circle (r : ℝ) : ℝ := Real.pi * r^2

noncomputable def circumference_of_circle (r : ℝ) : ℝ := 2 * Real.pi * r

theorem circumference_given_area :
  (∃ r : ℝ, area_of_circle r = 616) →
  circumference_of_circle 14 = 2 * Real.pi * 14 :=
by
  sorry

end circumference_given_area_l1447_144716


namespace evaluate_expression_l1447_144792

theorem evaluate_expression : 4 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 3200 :=
by
  sorry

end evaluate_expression_l1447_144792


namespace solve_equation_l1447_144774

theorem solve_equation :
  (∀ x : ℝ, x ≠ 2/3 → (6 * x + 2) / (3 * x^2 + 6 * x - 4) = (3 * x) / (3 * x - 2)) →
  (∀ x : ℝ, x = 1 / Real.sqrt 3 ∨ x = -1 / Real.sqrt 3) :=
by
  sorry

end solve_equation_l1447_144774


namespace holidays_per_month_l1447_144706

theorem holidays_per_month (total_holidays : ℕ) (months_in_year : ℕ) (holidays_in_month : ℕ) 
    (h1 : total_holidays = 48) (h2 : months_in_year = 12) : holidays_in_month = 4 := 
by
  sorry

end holidays_per_month_l1447_144706


namespace three_digit_solutions_modulo_l1447_144795

def three_digit_positive_integers (x : ℕ) : Prop :=
  100 ≤ x ∧ x ≤ 999

theorem three_digit_solutions_modulo (h : ∃ x : ℕ, three_digit_positive_integers x ∧ 
  (2597 * x + 763) % 17 = 1459 % 17) : 
  ∃ (count : ℕ), count = 53 :=
by sorry

end three_digit_solutions_modulo_l1447_144795


namespace difference_before_flipping_l1447_144759

-- Definitions based on the conditions:
variables (Y G : ℕ) -- Number of yellow and green papers

-- Condition: flipping 16 yellow papers changes counts as described
def papers_after_flipping (Y G : ℕ) : Prop :=
  Y - 16 = G + 16

-- Condition: after flipping, there are 64 more green papers than yellow papers.
def green_more_than_yellow_after_flipping (G Y : ℕ) : Prop :=
  G + 16 = (Y - 16) + 64

-- Statement: Prove the difference in the number of green and yellow papers before flipping was 32.
theorem difference_before_flipping (Y G : ℕ) (h1 : papers_after_flipping Y G) (h2 : green_more_than_yellow_after_flipping G Y) :
  (Y - G) = 32 :=
by
  sorry

end difference_before_flipping_l1447_144759


namespace perimeter_of_unshaded_rectangle_l1447_144710

theorem perimeter_of_unshaded_rectangle (length width height base area shaded_area perimeter : ℝ)
  (h1 : length = 12)
  (h2 : width = 9)
  (h3 : height = 3)
  (h4 : base = (2 * shaded_area) / height)
  (h5 : shaded_area = 18)
  (h6 : perimeter = 2 * ((length - base) + width))
  : perimeter = 24 := by
  sorry

end perimeter_of_unshaded_rectangle_l1447_144710


namespace Saheed_earnings_l1447_144746

theorem Saheed_earnings (Vika_earnings : ℕ) (Kayla_earnings : ℕ) (Saheed_earnings : ℕ)
  (h1 : Vika_earnings = 84) (h2 : Kayla_earnings = Vika_earnings - 30) (h3 : Saheed_earnings = 4 * Kayla_earnings) :
  Saheed_earnings = 216 := 
by
  sorry

end Saheed_earnings_l1447_144746


namespace restaurant_total_dishes_l1447_144776

noncomputable def total_couscous_received : ℝ := 15.4 + 45
noncomputable def total_chickpeas_received : ℝ := 19.8 + 33

-- Week 1, ratio of 5:3 (couscous:chickpeas)
noncomputable def sets_of_ratio_week1_couscous : ℝ := total_couscous_received / 5
noncomputable def sets_of_ratio_week1_chickpeas : ℝ := total_chickpeas_received / 3
noncomputable def dishes_week1 : ℝ := min sets_of_ratio_week1_couscous sets_of_ratio_week1_chickpeas

-- Week 2, ratio of 3:2 (couscous:chickpeas)
noncomputable def sets_of_ratio_week2_couscous : ℝ := total_couscous_received / 3
noncomputable def sets_of_ratio_week2_chickpeas : ℝ := total_chickpeas_received / 2
noncomputable def dishes_week2 : ℝ := min sets_of_ratio_week2_couscous sets_of_ratio_week2_chickpeas

-- Total dishes rounded down
noncomputable def total_dishes : ℝ := dishes_week1 + dishes_week2

theorem restaurant_total_dishes :
  ⌊total_dishes⌋ = 32 :=
by {
  sorry
}

end restaurant_total_dishes_l1447_144776


namespace probability_of_consonant_initials_is_10_over_13_l1447_144703

def is_vowel (c : Char) : Prop :=
  c = 'A' ∨ c = 'E' ∨ c = 'I' ∨ c = 'O' ∨ c = 'U' ∨ c = 'Y'

def is_consonant (c : Char) : Prop :=
  ¬(is_vowel c) ∧ c ≠ 'W' 

noncomputable def probability_of_consonant_initials : ℚ :=
  let total_letters := 26
  let number_of_vowels := 6
  let number_of_consonants := total_letters - number_of_vowels
  number_of_consonants / total_letters

theorem probability_of_consonant_initials_is_10_over_13 :
  probability_of_consonant_initials = 10 / 13 :=
by
  sorry

end probability_of_consonant_initials_is_10_over_13_l1447_144703


namespace abigail_total_savings_l1447_144713

def monthly_savings : ℕ := 4000
def months_in_year : ℕ := 12

theorem abigail_total_savings : monthly_savings * months_in_year = 48000 := by
  sorry

end abigail_total_savings_l1447_144713


namespace melanie_mother_dimes_l1447_144784

-- Definitions based on the conditions
variables (initial_dimes : ℕ) (dimes_given_to_dad : ℤ) (current_dimes : ℤ)

-- Conditions
def melanie_conditions := initial_dimes = 7 ∧ dimes_given_to_dad = 8 ∧ current_dimes = 3

-- Question to be proved is equivalent to proving the number of dimes given by her mother
theorem melanie_mother_dimes (initial_dimes : ℕ) (dimes_given_to_dad : ℤ) (current_dimes : ℤ) (dimes_given_by_mother : ℤ) 
  (h : melanie_conditions initial_dimes dimes_given_to_dad current_dimes) : 
  dimes_given_by_mother = 4 :=
by 
  sorry

end melanie_mother_dimes_l1447_144784


namespace dozen_pencils_l1447_144761

-- Define the given conditions
def pencils_total : ℕ := 144
def pencils_per_dozen : ℕ := 12

-- Theorem stating the desired proof
theorem dozen_pencils (h : pencils_total = 144) (hdozen : pencils_per_dozen = 12) : 
  pencils_total / pencils_per_dozen = 12 :=
by
  sorry

end dozen_pencils_l1447_144761


namespace distribution_ways_l1447_144740

-- Define the conditions
def num_papers : ℕ := 7
def num_friends : ℕ := 10

-- Define the theorem to prove the number of ways to distribute the papers
theorem distribution_ways : (num_friends ^ num_papers) = 10000000 := by
  -- This is where the proof would go
  sorry

end distribution_ways_l1447_144740


namespace ephraim_keiko_same_heads_probability_l1447_144775

def coin_toss_probability_same_heads : ℚ :=
  let keiko_prob_0 := 1 / 4
  let keiko_prob_1 := 1 / 2
  let keiko_prob_2 := 1 / 4
  let ephraim_prob_0 := 1 / 8
  let ephraim_prob_1 := 3 / 8
  let ephraim_prob_2 := 3 / 8
  let ephraim_prob_3 := 1 / 8
  (keiko_prob_0 * ephraim_prob_0) 
  + (keiko_prob_1 * ephraim_prob_1) 
  + (keiko_prob_2 * ephraim_prob_2)

theorem ephraim_keiko_same_heads_probability : 
  coin_toss_probability_same_heads = 11 / 32 :=
by 
  unfold coin_toss_probability_same_heads
  norm_num
  sorry

end ephraim_keiko_same_heads_probability_l1447_144775


namespace Manu_takes_12_more_seconds_l1447_144715

theorem Manu_takes_12_more_seconds (P M A : ℕ) 
  (hP : P = 60) 
  (hA1 : A = 36) 
  (hA2 : A = M / 2) : 
  M - P = 12 :=
by
  sorry

end Manu_takes_12_more_seconds_l1447_144715


namespace find_number_l1447_144724

-- Define the hypothesis/condition
def condition (x : ℤ) : Prop := 2 * x + 20 = 8 * x - 4

-- Define the statement to prove
theorem find_number (x : ℤ) (h : condition x) : x = 4 := 
by
  sorry

end find_number_l1447_144724


namespace recommended_daily_serving_l1447_144780

theorem recommended_daily_serving (mg_per_pill : ℕ) (pills_per_week : ℕ) (total_mg_week : ℕ) (days_per_week : ℕ) 
  (h1 : mg_per_pill = 50) (h2 : pills_per_week = 28) (h3 : total_mg_week = pills_per_week * mg_per_pill) 
  (h4 : days_per_week = 7) : 
  total_mg_week / days_per_week = 200 :=
by
  sorry

end recommended_daily_serving_l1447_144780


namespace min_h4_for_ahai_avg_ge_along_avg_plus_4_l1447_144747

-- Definitions from conditions
variables (a1 a2 a3 a4 : ℝ)
variables (h1 h2 h3 h4 : ℝ)

-- Conditions from the problem
axiom a1_gt_80 : a1 > 80
axiom a2_gt_80 : a2 > 80
axiom a3_gt_80 : a3 > 80
axiom a4_gt_80 : a4 > 80

axiom h1_eq_a1_plus_1 : h1 = a1 + 1
axiom h2_eq_a2_plus_2 : h2 = a2 + 2
axiom h3_eq_a3_plus_3 : h3 = a3 + 3

-- Lean 4 statement for the problem
theorem min_h4_for_ahai_avg_ge_along_avg_plus_4 : h4 ≥ 99 :=
by
  sorry

end min_h4_for_ahai_avg_ge_along_avg_plus_4_l1447_144747


namespace sand_cake_probability_is_12_percent_l1447_144727

def total_days : ℕ := 5
def ham_days : ℕ := 3
def cake_days : ℕ := 1

-- Probability of packing a ham sandwich on any given day
def prob_ham_sandwich : ℚ := ham_days / total_days

-- Probability of packing a piece of cake on any given day
def prob_cake : ℚ := cake_days / total_days

-- Calculate the combined probability that Karen packs a ham sandwich and cake on the same day
def combined_probability : ℚ := prob_ham_sandwich * prob_cake

-- Convert the combined probability to a percentage
def combined_probability_as_percentage : ℚ := combined_probability * 100

-- The proof problem to show that the probability that Karen packs a ham sandwich and cake on the same day is 12%
theorem sand_cake_probability_is_12_percent : combined_probability_as_percentage = 12 := 
  by sorry

end sand_cake_probability_is_12_percent_l1447_144727


namespace total_items_given_out_l1447_144755

-- Miss Davis gave 15 popsicle sticks and 20 straws to each group.
def popsicle_sticks_per_group := 15
def straws_per_group := 20
def items_per_group := popsicle_sticks_per_group + straws_per_group

-- There are 10 groups in total.
def number_of_groups := 10

-- Prove the total number of items given out equals 350.
theorem total_items_given_out : items_per_group * number_of_groups = 350 :=
by
  sorry

end total_items_given_out_l1447_144755


namespace find_c_minus_2d_l1447_144722

theorem find_c_minus_2d :
  ∃ (c d : ℕ), (c > d) ∧ (c - 2 * d = 0) ∧ (∀ x : ℕ, (x^2 - 18 * x + 72 = (x - c) * (x - d))) :=
by
  sorry

end find_c_minus_2d_l1447_144722


namespace reciprocal_sum_of_roots_l1447_144704

theorem reciprocal_sum_of_roots :
  (∃ m n : ℝ, (m^2 + 2 * m - 3 = 0) ∧ (n^2 + 2 * n - 3 = 0) ∧ m ≠ n) →
  (∃ m n : ℝ, (1/m + 1/n = 2/3)) :=
by
  sorry

end reciprocal_sum_of_roots_l1447_144704


namespace Frank_initial_savings_l1447_144734

theorem Frank_initial_savings 
  (cost_per_toy : Nat)
  (number_of_toys : Nat)
  (allowance : Nat)
  (total_cost : Nat)
  (initial_savings : Nat)
  (h1 : cost_per_toy = 8)
  (h2 : number_of_tys = 5)
  (h3 : allowance = 37)
  (h4 : total_cost = number_of_toys * cost_per_toy)
  (h5 : initial_savings + allowance = total_cost)
  : initial_savings = 3 := 
by
  sorry

end Frank_initial_savings_l1447_144734


namespace probability_at_least_two_green_l1447_144763

def total_apples := 10
def red_apples := 6
def green_apples := 4
def choose_apples := 3

noncomputable def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_at_least_two_green :
  (binomial green_apples 3 + binomial green_apples 2 * binomial red_apples 1) = 40 ∧ 
  binomial total_apples choose_apples = 120 ∧
  (binomial green_apples 3 + binomial green_apples 2 * binomial red_apples 1) / binomial total_apples choose_apples = 1 / 3 := by
sorry

end probability_at_least_two_green_l1447_144763


namespace vivian_mail_in_august_l1447_144757

-- Conditions
def april_mail : ℕ := 5
def may_mail : ℕ := 2 * april_mail
def june_mail : ℕ := 2 * may_mail
def july_mail : ℕ := 2 * june_mail

-- Question: Prove that Vivian will send 80 pieces of mail in August.
theorem vivian_mail_in_august : 2 * july_mail = 80 :=
by
  -- Sorry to skip the proof
  sorry

end vivian_mail_in_august_l1447_144757


namespace union_set_l1447_144705

def M : Set ℝ := {x | -2 < x ∧ x < 1}
def P : Set ℝ := {x | -2 ≤ x ∧ x < 2}

theorem union_set : M ∪ P = {x : ℝ | -2 ≤ x ∧ x < 2} := by
  sorry

end union_set_l1447_144705


namespace calculate_expression_l1447_144751

/-
We need to prove that the value of 18 * 36 + 54 * 18 + 18 * 9 is equal to 1782.
-/

theorem calculate_expression : (18 * 36 + 54 * 18 + 18 * 9 = 1782) :=
by
  have a1 : Int := 18 * 36
  have a2 : Int := 54 * 18
  have a3 : Int := 18 * 9
  sorry

end calculate_expression_l1447_144751


namespace sum_of_squares_nonnegative_l1447_144787

theorem sum_of_squares_nonnegative (x y z : ℝ) : x^2 + y^2 + z^2 - x * y - x * z - y * z ≥ 0 :=
  sorry

end sum_of_squares_nonnegative_l1447_144787


namespace area_of_circumscribed_circle_eq_48pi_l1447_144765

noncomputable def side_length := 12
noncomputable def radius := (2/3) * (side_length / 2) * (Real.sqrt 3)
noncomputable def area := Real.pi * radius^2

theorem area_of_circumscribed_circle_eq_48pi :
  area = 48 * Real.pi :=
by
  sorry

end area_of_circumscribed_circle_eq_48pi_l1447_144765


namespace degrees_for_combined_research_l1447_144708

-- Define the conditions as constants.
def microphotonics_percentage : ℝ := 0.10
def home_electronics_percentage : ℝ := 0.24
def food_additives_percentage : ℝ := 0.15
def gmo_percentage : ℝ := 0.29
def industrial_lubricants_percentage : ℝ := 0.08
def nanotechnology_percentage : ℝ := 0.07

noncomputable def remaining_percentage : ℝ :=
  1 - (microphotonics_percentage + home_electronics_percentage + food_additives_percentage +
    gmo_percentage + industrial_lubricants_percentage + nanotechnology_percentage)

noncomputable def total_percentage : ℝ :=
  remaining_percentage + nanotechnology_percentage

noncomputable def degrees_in_circle : ℝ := 360

noncomputable def degrees_representing_combined_research : ℝ :=
  total_percentage * degrees_in_circle

-- State the theorem to prove the correct answer
theorem degrees_for_combined_research : degrees_representing_combined_research = 50.4 :=
by
  -- Proof will go here
  sorry

end degrees_for_combined_research_l1447_144708


namespace area_of_triangle_l1447_144754

-- Define the lines as functions
def line1 : ℝ → ℝ := fun x => 3 * x - 4
def line2 : ℝ → ℝ := fun x => -2 * x + 16

-- Define the vertices of the triangle formed by lines and y-axis
def vertex1 : ℝ × ℝ := (0, -4)
def vertex2 : ℝ × ℝ := (0, 16)
def vertex3 : ℝ × ℝ := (4, 8)

-- Define the proof statement
theorem area_of_triangle : 
  let A := vertex1 
  let B := vertex2 
  let C := vertex3 
  -- Compute the area of the triangle using the determinant formula
  let area := (1 / 2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))
  area = 40 := 
by
  sorry

end area_of_triangle_l1447_144754


namespace determine_g1_l1447_144748

variable (g : ℝ → ℝ)
variable (h : ∀ x y : ℝ, g (g x + y) = g (x + y) + x * g y - x^2 * y - x^3 + 1)

theorem determine_g1 : g 1 = 2 := sorry

end determine_g1_l1447_144748


namespace find_a_l1447_144743

theorem find_a (a : ℕ) (h_pos : 0 < a)
  (h_cube : ∀ n : ℕ, 0 < n → ∃ k : ℤ, 4 * ((a : ℤ) ^ n + 1) = k^3) :
  a = 1 :=
sorry

end find_a_l1447_144743


namespace eric_has_9306_erasers_l1447_144709

-- Define the conditions as constants
def number_of_friends := 99
def erasers_per_friend := 94

-- Define the total number of erasers based on the conditions
def total_erasers := number_of_friends * erasers_per_friend

-- Theorem stating the total number of erasers Eric has
theorem eric_has_9306_erasers : total_erasers = 9306 := by
  -- Proof to be filled in
  sorry

end eric_has_9306_erasers_l1447_144709


namespace problem_solution_l1447_144796

noncomputable def verify_solution (x y z : ℝ) : Prop :=
  x = 12 ∧ y = 10 ∧ z = 8 →
  (x > 4) ∧ (y > 4) ∧ (z > 4) →
  ( ( (x + 3)^2 / (y + z - 3) ) + 
    ( (y + 5)^2 / (z + x - 5) ) + 
    ( (z + 7)^2 / (x + y - 7) ) = 45)

theorem problem_solution :
  verify_solution 12 10 8 := by
  sorry

end problem_solution_l1447_144796


namespace complement_M_in_U_l1447_144772

open Set

theorem complement_M_in_U : 
  let U : Set ℕ := {1, 3, 5, 7}
  let M : Set ℕ := {1, 5}
  U \ M = {3, 7} := 
by
  let U : Set ℕ := {1, 3, 5, 7}
  let M : Set ℕ := {1, 5}
  sorry

end complement_M_in_U_l1447_144772


namespace simplify_expression_l1447_144766

theorem simplify_expression (x : ℝ) (h : x = Real.sqrt 2) :
  (x - 1 - (2*x - 2)/(x + 1)) / ((x^2 - x) / (2*x + 2)) = 2 - Real.sqrt 2 := 
by
  -- Here we should include the proof steps, but we skip it with "sorry"
  sorry

end simplify_expression_l1447_144766


namespace sample_size_correct_l1447_144702

variable (total_employees young_employees middle_aged_employees elderly_employees young_in_sample sample_size : ℕ)

-- Conditions
def total_number_of_employees := 75
def number_of_young_employees := 35
def number_of_middle_aged_employees := 25
def number_of_elderly_employees := 15
def number_of_young_in_sample := 7
def stratified_sampling := true

-- The proof problem statement
theorem sample_size_correct :
  total_employees = total_number_of_employees ∧ 
  young_employees = number_of_young_employees ∧ 
  middle_aged_employees = number_of_middle_aged_employees ∧ 
  elderly_employees = number_of_elderly_employees ∧ 
  young_in_sample = number_of_young_in_sample ∧ 
  stratified_sampling → 
  sample_size = 15 := by sorry

end sample_size_correct_l1447_144702


namespace range_of_a_l1447_144737

noncomputable def f (a x : ℝ) : ℝ := a * x ^ 3 - x ^ 2 + x - 5

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, ∃ x_max x_min : ℝ, x_max ≠ x_min ∧
  f a x_max = max (f a x_max) (f a x_min) ∧ f a x_min = min (f a x_max) (f a x_min)) → 
  a < 1 / 3 ∧ a ≠ 0 := sorry

end range_of_a_l1447_144737


namespace jean_to_shirt_ratio_l1447_144720

theorem jean_to_shirt_ratio (shirts_sold jeans_sold shirt_cost total_revenue: ℕ) (h1: shirts_sold = 20) (h2: jeans_sold = 10) (h3: shirt_cost = 10) (h4: total_revenue = 400) : 
(shirt_cost * shirts_sold + jeans_sold * ((total_revenue - (shirt_cost * shirts_sold)) / jeans_sold)) / (total_revenue - (shirt_cost * shirts_sold)) / jeans_sold = 2 := 
sorry

end jean_to_shirt_ratio_l1447_144720


namespace flag_covering_proof_l1447_144733

def grid_covering_flag_ways (m n num_flags cells_per_flag : ℕ) :=
  if m * n / cells_per_flag = num_flags then 2^num_flags else 0

theorem flag_covering_proof :
  grid_covering_flag_ways 9 18 18 9 = 262144 := by
  sorry

end flag_covering_proof_l1447_144733


namespace discounted_price_of_russian_doll_l1447_144783

theorem discounted_price_of_russian_doll (original_price : ℕ) (number_of_dolls_original : ℕ) (number_of_dolls_discounted : ℕ) (discounted_price : ℕ) :
  original_price = 4 →
  number_of_dolls_original = 15 →
  number_of_dolls_discounted = 20 →
  (number_of_dolls_original * original_price) = 60 →
  (number_of_dolls_discounted * discounted_price) = 60 →
  discounted_price = 3 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end discounted_price_of_russian_doll_l1447_144783


namespace expression_multiple_of_five_l1447_144711

theorem expression_multiple_of_five (n : ℕ) (h : n ≥ 10) : 
  (∃ k : ℕ, (n + 2) * (n + 1) = 5 * k) :=
sorry

end expression_multiple_of_five_l1447_144711


namespace height_relationship_l1447_144789

theorem height_relationship
  (r1 h1 r2 h2 : ℝ)
  (volume_eq : π * r1^2 * h1 = π * r2^2 * h2)
  (radius_relation : r2 = 1.2 * r1) :
  h1 = 1.44 * h2 :=
sorry

end height_relationship_l1447_144789


namespace arithmetic_geometric_sum_l1447_144739

theorem arithmetic_geometric_sum (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a)
  (h_arith : 2 * b = a + c) (h_geom : a^2 = b * c) 
  (h_sum : a + 3 * b + c = 10) : a = -4 :=
by
  sorry

end arithmetic_geometric_sum_l1447_144739


namespace mars_colony_cost_l1447_144771

theorem mars_colony_cost :
  let total_cost := 45000000000
  let number_of_people := 300000000
  total_cost / number_of_people = 150 := 
by sorry

end mars_colony_cost_l1447_144771


namespace minnie_more_than_week_l1447_144762

-- Define the variables and conditions
variable (M : ℕ) -- number of horses Minnie mounts per day
variable (mickey_daily : ℕ) -- number of horses Mickey mounts per day

axiom mickey_daily_formula : mickey_daily = 2 * M - 6
axiom mickey_total_per_week : mickey_daily * 7 = 98
axiom days_in_week : 7 = 7

-- Theorem: Minnie mounts 3 more horses per day than there are days in a week
theorem minnie_more_than_week (M : ℕ) 
  (h1 : mickey_daily = 2 * M - 6)
  (h2 : mickey_daily * 7 = 98)
  (h3 : 7 = 7) :
  M - 7 = 3 := 
sorry

end minnie_more_than_week_l1447_144762
