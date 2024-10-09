import Mathlib

namespace max_price_of_most_expensive_product_l59_5974

noncomputable def greatest_possible_price
  (num_products : ℕ)
  (avg_price : ℕ)
  (min_price : ℕ)
  (mid_price : ℕ)
  (higher_price_count : ℕ)
  (total_retail_price : ℕ)
  (least_expensive_total_price : ℕ)
  (remaining_price : ℕ)
  (less_expensive_total_price : ℕ) : ℕ :=
  total_retail_price - least_expensive_total_price - less_expensive_total_price

theorem max_price_of_most_expensive_product :
  greatest_possible_price 20 1200 400 1000 10 (20 * 1200) (10 * 400) (20 * 1200 - 10 * 400) (9 * 1000) = 11000 :=
by
  sorry

end max_price_of_most_expensive_product_l59_5974


namespace stuffed_animal_tickets_correct_l59_5985

-- Define the total tickets spent
def total_tickets : ℕ := 14

-- Define the tickets spent on the hat
def hat_tickets : ℕ := 2

-- Define the tickets spent on the yoyo
def yoyo_tickets : ℕ := 2

-- Define the tickets spent on the stuffed animal
def stuffed_animal_tickets : ℕ := total_tickets - (hat_tickets + yoyo_tickets)

-- The theorem we want to prove.
theorem stuffed_animal_tickets_correct :
  stuffed_animal_tickets = 10 :=
by
  sorry

end stuffed_animal_tickets_correct_l59_5985


namespace volleyball_team_selection_l59_5965

/-- A set representing players on the volleyball team -/
def players : Finset String := {
  "Missy", "Lauren", "Liz", -- triplets
  "Anna", "Mia",           -- twins
  "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10" -- other players
}

/-- The triplets -/
def triplets : Finset String := {"Missy", "Lauren", "Liz"}

/-- The twins -/
def twins : Finset String := {"Anna", "Mia"}

/-- The number of ways to choose 7 starters given the restrictions -/
theorem volleyball_team_selection : 
  let total_ways := (players.card.choose 7)
  let select_3_triplets := (players \ triplets).card.choose 4
  let select_2_twins := (players \ twins).card.choose 5
  let select_all_restriction := (players \ (triplets ∪ twins)).card.choose 2
  total_ways - select_3_triplets - select_2_twins + select_all_restriction = 9778 := by
  sorry

end volleyball_team_selection_l59_5965


namespace cakes_in_november_l59_5927

-- Define the function modeling the number of cakes baked each month
def num_of_cakes (initial: ℕ) (n: ℕ) := initial + 2 * n

-- Given conditions
def cakes_in_october := 19
def cakes_in_december := 23
def cakes_in_january := 25
def cakes_in_february := 27
def monthly_increase := 2

-- Prove that the number of cakes baked in November is 21
theorem cakes_in_november : num_of_cakes cakes_in_october 1 = 21 :=
by
  sorry

end cakes_in_november_l59_5927


namespace y_is_80_percent_less_than_x_l59_5961

theorem y_is_80_percent_less_than_x (x y : ℝ) (h : x = 5 * y) : ((x - y) / x) * 100 = 80 :=
by sorry

end y_is_80_percent_less_than_x_l59_5961


namespace small_panda_bears_count_l59_5968

theorem small_panda_bears_count :
  ∃ (S : ℕ), ∃ (B : ℕ),
    B = 5 ∧ 7 * (25 * S + 40 * B) = 2100 ∧ S = 4 :=
by
  exists 4
  exists 5
  repeat { sorry }

end small_panda_bears_count_l59_5968


namespace goods_train_speed_l59_5977

-- Define the given constants
def train_length : ℕ := 370 -- in meters
def platform_length : ℕ := 150 -- in meters
def crossing_time : ℕ := 26 -- in seconds
def conversion_factor : ℕ := 36 / 10 -- conversion from m/s to km/hr

-- Define the total distance covered
def total_distance : ℕ := train_length + platform_length -- in meters

-- Define the speed of the train in m/s
def speed_m_per_s : ℕ := total_distance / crossing_time

-- Define the speed of the train in km/hr
def speed_km_per_hr : ℕ := speed_m_per_s * conversion_factor

-- The proof problem statement
theorem goods_train_speed : speed_km_per_hr = 72 := 
by 
  -- Placeholder for the proof
  sorry

end goods_train_speed_l59_5977


namespace days_for_B_l59_5991

theorem days_for_B
  (x : ℝ)
  (hA : 15 ≠ 0)
  (h_nonzero_fraction : 0.5833333333333334 ≠ 0)
  (hfraction : 0 <  0.5833333333333334 ∧ 0.5833333333333334 < 1)
  (h_fraction_work_left : 5 * (1 / 15 + 1 / x) = 0.5833333333333334) :
  x = 20 := by
  sorry

end days_for_B_l59_5991


namespace other_root_eq_l59_5960

theorem other_root_eq (b : ℝ) : (∀ x, x^2 + b * x - 2 = 0 → (x = 1 ∨ x = -2)) :=
by
  intro x hx
  have : x = 1 ∨ x = -2 := sorry
  exact this

end other_root_eq_l59_5960


namespace smallest_sum_of_sequence_l59_5987

theorem smallest_sum_of_sequence {
  A B C D k : ℕ
} (h1 : 2 * B = A + C)
  (h2 : D - C = (C - B) ^ 2)
  (h3 : 4 * B = 3 * C)
  (h4 : B = 3 * k)
  (h5 : C = 4 * k)
  (h6 : A = 2 * k)
  (h7 : D = 4 * k + k ^ 2) :
  A + B + C + D = 14 :=
by
  sorry

end smallest_sum_of_sequence_l59_5987


namespace age_sum_is_47_l59_5919

theorem age_sum_is_47 (a b c : ℕ) (b_def : b = 18) 
  (a_def : a = b + 2) (c_def : c = b / 2) : a + b + c = 47 :=
by
  sorry

end age_sum_is_47_l59_5919


namespace smallest_t_eq_3_over_4_l59_5996

theorem smallest_t_eq_3_over_4 (t : ℝ) :
  (∀ t : ℝ,
    (16 * t^3 - 49 * t^2 + 35 * t - 6) / (4 * t - 3) + 7 * t = 8 * t - 2 → t >= (3 / 4)) ∧
  (∃ t₀ : ℝ, (16 * t₀^3 - 49 * t₀^2 + 35 * t₀ - 6) / (4 * t₀ - 3) + 7 * t₀ = 8 * t₀ - 2 ∧ t₀ = (3 / 4)) :=
sorry

end smallest_t_eq_3_over_4_l59_5996


namespace isabella_hourly_rate_l59_5903

def isabella_hours_per_day : ℕ := 5
def isabella_days_per_week : ℕ := 6
def isabella_weeks : ℕ := 7
def isabella_total_earnings : ℕ := 1050

theorem isabella_hourly_rate :
  (isabella_hours_per_day * isabella_days_per_week * isabella_weeks) * x = isabella_total_earnings → x = 5 := by
  sorry

end isabella_hourly_rate_l59_5903


namespace geometric_sequence_first_term_l59_5918

theorem geometric_sequence_first_term (a r : ℝ) 
  (h1 : a * r = 5) 
  (h2 : a * r^3 = 45) : 
  a = 5 / (3^(2/3)) := 
by
  -- proof steps to be filled here
  sorry

end geometric_sequence_first_term_l59_5918


namespace inequality_proof_l59_5967

open Real

theorem inequality_proof (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (hSum : x + y + z = 1) :
  x * y / sqrt (x * y + y * z) + y * z / sqrt (y * z + z * x) + z * x / sqrt (z * x + x * y) ≤ sqrt 2 / 2 := 
sorry

end inequality_proof_l59_5967


namespace find_m_for_parallel_lines_l59_5906

noncomputable def parallel_lines_x_plus_1_plus_m_y_eq_2_minus_m_and_m_x_plus_2_y_plus_8_eq_0 (m : ℝ) : Prop :=
  let l1_slope := -(1 + m) / 1
  let l2_slope := -m / 2
  l1_slope = l2_slope

theorem find_m_for_parallel_lines :
  parallel_lines_x_plus_1_plus_m_y_eq_2_minus_m_and_m_x_plus_2_y_plus_8_eq_0 m →
  m = 1 :=
by
  intro h_parallel
  -- Here we would present the proof steps to show that m = 1 under the given conditions.
  sorry

end find_m_for_parallel_lines_l59_5906


namespace union_of_S_and_T_l59_5946

-- Definitions of the sets S and T
def S : Set ℝ := { y | ∃ x : ℝ, y = Real.exp x - 2 }
def T : Set ℝ := { x | -4 ≤ x ∧ x ≤ 1 }

-- Lean proof problem statement
theorem union_of_S_and_T : (S ∪ T) = { y | -4 ≤ y } :=
by
  sorry

end union_of_S_and_T_l59_5946


namespace monotonically_increasing_interval_l59_5984

noncomputable def f (x : ℝ) : ℝ := 4 * x^2 + 1 / x

theorem monotonically_increasing_interval :
  ∀ x : ℝ, x > 1 / 2 → (∀ y : ℝ, y < x → f y < f x) :=
by
  intro x h
  intro y hy
  sorry

end monotonically_increasing_interval_l59_5984


namespace factorization_of_1386_l59_5935

-- We start by defining the number and the requirements.
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def factors_mult (a b : ℕ) : Prop := a * b = 1386
def factorization_count (count : ℕ) : Prop :=
  ∃ (a b : ℕ), is_two_digit a ∧ is_two_digit b ∧ factors_mult a b ∧ 
  (∀ c d, is_two_digit c ∧ is_two_digit d ∧ factors_mult c d → 
  (c = a ∧ d = b ∨ c = b ∧ d = a) → c = a ∧ d = b ∨ c = b ∧ d = a) ∧
  count = 4

-- Now, we state the theorem.
theorem factorization_of_1386 : factorization_count 4 :=
sorry

end factorization_of_1386_l59_5935


namespace laura_annual_income_l59_5957

variable (p : ℝ) -- percentage p
variable (A T : ℝ) -- annual income A and total income tax T

def tax1 : ℝ := 0.01 * p * 35000
def tax2 : ℝ := 0.01 * (p + 3) * (A - 35000)
def tax3 : ℝ := 0.01 * (p + 5) * (A - 55000)

theorem laura_annual_income (h_cond1 : A > 55000)
  (h_tax : T = 350 * p + 600 + 0.01 * (p + 5) * (A - 55000))
  (h_paid_tax : T = (0.01 * (p + 0.45)) * A):
  A = 75000 := by
  sorry

end laura_annual_income_l59_5957


namespace gcd_g105_g106_l59_5930

def g (x : ℕ) : ℕ := x^2 - x + 2502

theorem gcd_g105_g106 : gcd (g 105) (g 106) = 2 := by
  sorry

end gcd_g105_g106_l59_5930


namespace one_fifth_of_5_times_7_l59_5907

theorem one_fifth_of_5_times_7 : (1 / 5) * (5 * 7) = 7 := by
  sorry

end one_fifth_of_5_times_7_l59_5907


namespace infinite_coprime_pairs_divisibility_l59_5969

theorem infinite_coprime_pairs_divisibility :
  ∃ (S : ℕ → ℕ × ℕ), (∀ n, Nat.gcd (S n).1 (S n).2 = 1 ∧ (S n).1 ∣ (S n).2^2 - 5 ∧ (S n).2 ∣ (S n).1^2 - 5) ∧
  Function.Injective S :=
sorry

end infinite_coprime_pairs_divisibility_l59_5969


namespace hiking_time_l59_5931

-- Define the conditions
def Distance : ℕ := 12
def Pace_up : ℕ := 4
def Pace_down : ℕ := 6

-- Statement to be proved
theorem hiking_time (d : ℕ) (pu : ℕ) (pd : ℕ) (h₁ : d = Distance) (h₂ : pu = Pace_up) (h₃ : pd = Pace_down) :
  d / pu + d / pd = 5 :=
by sorry

end hiking_time_l59_5931


namespace initial_scooter_value_l59_5975

theorem initial_scooter_value (V : ℝ) 
    (h : (9 / 16) * V = 22500) : V = 40000 :=
sorry

end initial_scooter_value_l59_5975


namespace a_7_is_4_l59_5911

-- Define the geometric sequence and its properties
variable {a : ℕ → ℝ}

-- Given conditions
axiom pos_seq : ∀ n, a n > 0
axiom geom_seq : ∀ n m, a (n + m) = a n * a m
axiom specific_condition : a 3 * a 11 = 16

theorem a_7_is_4 : a 7 = 4 :=
by
  sorry

end a_7_is_4_l59_5911


namespace square_roots_of_four_ninths_cube_root_of_neg_sixty_four_l59_5942

theorem square_roots_of_four_ninths : {x : ℚ | x ^ 2 = 4 / 9} = {2 / 3, -2 / 3} :=
by
  sorry

theorem cube_root_of_neg_sixty_four : {y : ℚ | y ^ 3 = -64} = {-4} :=
by
  sorry

end square_roots_of_four_ninths_cube_root_of_neg_sixty_four_l59_5942


namespace C0E_hex_to_dec_l59_5917

theorem C0E_hex_to_dec : 
  let C := 12
  let E := 14 
  let result := C * 16^2 + 0 * 16^1 + E * 16^0
  result = 3086 :=
by 
  let C := 12
  let E := 14 
  let result := C * 16^2 + 0 * 16^1 + E * 16^0
  sorry

end C0E_hex_to_dec_l59_5917


namespace xy_difference_l59_5966

noncomputable def x : ℝ := Real.sqrt 3 + 1
noncomputable def y : ℝ := Real.sqrt 3 - 1

theorem xy_difference : x^2 * y - x * y^2 = 4 := by
  sorry

end xy_difference_l59_5966


namespace tram_speed_l59_5945

/-- 
Given:
1. The pedestrian's speed is 1 km per 10 minutes, which converts to 6 km/h.
2. The speed of the trams is V km/h.
3. The relative speed of oncoming trams is V + 6 km/h.
4. The relative speed of overtaking trams is V - 6 km/h.
5. The ratio of the number of oncoming trams to overtaking trams is 700/300.
Prove:
The speed of the trams V is 15 km/h.
-/
theorem tram_speed (V : ℝ) (h1 : (V + 6) / (V - 6) = 700 / 300) : V = 15 :=
by
  sorry

end tram_speed_l59_5945


namespace james_vegetable_intake_l59_5936

theorem james_vegetable_intake :
  let daily_asparagus := 0.25
  let daily_broccoli := 0.25
  let daily_intake := daily_asparagus + daily_broccoli
  let doubled_daily_intake := daily_intake * 2
  let weekly_intake_asparagus_broccoli := doubled_daily_intake * 7
  let weekly_kale := 3
  let total_weekly_intake := weekly_intake_asparagus_broccoli + weekly_kale
  total_weekly_intake = 10 := 
by
  sorry

end james_vegetable_intake_l59_5936


namespace binomial_coeff_equal_l59_5964

theorem binomial_coeff_equal (n : ℕ) (h₁ : 6 ≤ n) (h₂ : (n.choose 5) * 3^5 = (n.choose 6) * 3^6) :
  n = 7 := sorry

end binomial_coeff_equal_l59_5964


namespace beth_finishes_first_l59_5923

open Real

noncomputable def andy_lawn_area : ℝ := sorry
noncomputable def beth_lawn_area : ℝ := andy_lawn_area / 3
noncomputable def carlos_lawn_area : ℝ := andy_lawn_area / 4

noncomputable def andy_mowing_rate : ℝ := sorry
noncomputable def beth_mowing_rate : ℝ := andy_mowing_rate
noncomputable def carlos_mowing_rate : ℝ := andy_mowing_rate / 2

noncomputable def carlos_break : ℝ := 10

noncomputable def andy_mowing_time := andy_lawn_area / andy_mowing_rate
noncomputable def beth_mowing_time := beth_lawn_area / beth_mowing_rate
noncomputable def carlos_mowing_time := (carlos_lawn_area / carlos_mowing_rate) + carlos_break

theorem beth_finishes_first :
  beth_mowing_time < andy_mowing_time ∧ beth_mowing_time < carlos_mowing_time := by
  sorry

end beth_finishes_first_l59_5923


namespace polar_to_rectangular_l59_5999

theorem polar_to_rectangular :
  let x := 16
  let y := 12
  let r := Real.sqrt (x^2 + y^2)
  let θ := Real.arctan (y / x)
  let new_r := 2 * r
  let new_θ := θ / 2
  let cos_half_θ := Real.sqrt ((1 + (x / r)) / 2)
  let sin_half_θ := Real.sqrt ((1 - (x / r)) / 2)
  let new_x := new_r * cos_half_θ
  let new_y := new_r * sin_half_θ
  new_x = 40 * Real.sqrt 0.9 ∧ new_y = 40 * Real.sqrt 0.1 := by
  sorry

end polar_to_rectangular_l59_5999


namespace min_overlap_l59_5950

variable (P : Set ℕ → ℝ)
variable (B M : Set ℕ)

-- Conditions
def P_B_def : P B = 0.95 := sorry
def P_M_def : P M = 0.85 := sorry

-- To Prove
theorem min_overlap : P (B ∩ M) = 0.80 := sorry

end min_overlap_l59_5950


namespace sum_of_ages_is_12_l59_5913

-- Let Y be the age of the youngest child
def Y : ℝ := 1.5

-- Let the ages of the other children
def age2 : ℝ := Y + 1
def age3 : ℝ := Y + 2
def age4 : ℝ := Y + 3

-- Define the sum of the ages
def sum_of_ages : ℝ := Y + age2 + age3 + age4

-- The theorem to prove the sum of the ages is 12 years
theorem sum_of_ages_is_12 : sum_of_ages = 12 :=
by
  -- The detailed proof is to be filled in later, currently skipped.
  sorry

end sum_of_ages_is_12_l59_5913


namespace henry_age_is_20_l59_5921

open Nat

def sum_ages (H J : ℕ) : Prop := H + J = 33
def age_relation (H J : ℕ) : Prop := H - 6 = 2 * (J - 6)

theorem henry_age_is_20 (H J : ℕ) (h1 : sum_ages H J) (h2 : age_relation H J) : H = 20 :=
by
  -- Proof goes here
  sorry

end henry_age_is_20_l59_5921


namespace running_speed_l59_5900

theorem running_speed (walk_speed total_distance walk_time total_time run_distance : ℝ) 
  (h_walk_speed : walk_speed = 4)
  (h_total_distance : total_distance = 4)
  (h_walk_time : walk_time = 0.5)
  (h_total_time : total_time = 0.75)
  (h_run_distance : run_distance = total_distance / 2) :
  (2 / ((total_time - walk_time) - 2 / walk_speed)) = 8 := 
by
  -- To be proven
  sorry

end running_speed_l59_5900


namespace good_permutations_count_l59_5902

-- Define the main problem and the conditions
theorem good_permutations_count (n : ℕ) (hn : n > 0) : 
  ∃ P : ℕ → ℕ, 
  (P n = (1 / Real.sqrt 5) * (((1 + Real.sqrt 5) / 2) ^ (n + 1) - ((1 - Real.sqrt 5) / 2) ^ (n + 1))) := 
sorry

end good_permutations_count_l59_5902


namespace jack_total_yen_l59_5983

def pounds := 42
def euros := 11
def yen := 3000
def pounds_per_euro := 2
def yen_per_pound := 100

theorem jack_total_yen : (euros * pounds_per_euro + pounds) * yen_per_pound + yen = 9400 := by
  sorry

end jack_total_yen_l59_5983


namespace solve_for_x_l59_5916

theorem solve_for_x (x : ℝ) : x^2 + 6 * x + 8 = -(x + 4) * (x + 6) ↔ x = -4 :=
by {
  sorry
}

end solve_for_x_l59_5916


namespace smallest_resolvable_debt_l59_5988

theorem smallest_resolvable_debt (p g : ℤ) : 
  ∃ p g : ℤ, (500 * p + 350 * g = 50) ∧ ∀ D > 0, (∃ p g : ℤ, 500 * p + 350 * g = D) → 50 ≤ D :=
by {
  sorry
}

end smallest_resolvable_debt_l59_5988


namespace triangle_area_l59_5982

theorem triangle_area (a b c : ℝ) (h1 : a = 9) (h2 : b = 40) (h3 : c = 41) (h4 : a^2 + b^2 = c^2) :
  (1 / 2) * a * b = 180 := 
by 
  -- proof is skipped with sorry
  sorry

end triangle_area_l59_5982


namespace find_beta_l59_5915

variable (α β : ℝ)

theorem find_beta 
  (h1 : Real.cos α = 1 / 7)
  (h2 : Real.cos (α + β) = -11 / 14)
  (h3 : 0 < α ∧ α < Real.pi / 2)
  (h4 : Real.pi / 2 < α + β ∧ α + β < Real.pi) : β = Real.pi / 3 := sorry

end find_beta_l59_5915


namespace student_marks_l59_5998

theorem student_marks (M P C X : ℕ) 
  (h1 : M + P = 60)
  (h2 : C = P + X)
  (h3 : M + C = 80) : X = 20 :=
by sorry

end student_marks_l59_5998


namespace sum_first_five_terms_l59_5972

-- Define the arithmetic sequence {a_n}
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ a1 d : ℝ, ∀ n : ℕ, a n = a1 + (n - 1) * d

-- Define the specific condition a_5 + a_8 - a_10 = 2
def specific_condition (a : ℕ → ℝ) : Prop :=
  a 5 + a 8 - a 10 = 2

-- Define the sum of the first five terms S₅
def S5 (a : ℕ → ℝ) : ℝ :=
  a 1 + a 2 + a 3 + a 4 + a 5 

-- The statement to be proved
theorem sum_first_five_terms (a : ℕ → ℝ) (h₁ : arithmetic_sequence a) (h₂ : specific_condition a) : 
  S5 a = 10 :=
sorry

end sum_first_five_terms_l59_5972


namespace general_solution_of_diff_eq_l59_5944

theorem general_solution_of_diff_eq
  (f : ℝ → ℝ → ℝ)
  (D : Set (ℝ × ℝ))
  (hf : ∀ x y, f x y = x)
  (hD : D = Set.univ) :
  ∃ C : ℝ, ∀ x : ℝ, ∃ y : ℝ, y = (x^2) / 2 + C :=
by
  sorry

end general_solution_of_diff_eq_l59_5944


namespace stuffed_animals_count_l59_5939

theorem stuffed_animals_count
  (total_prizes : ℕ)
  (frisbees : ℕ)
  (yoyos : ℕ)
  (h1 : total_prizes = 50)
  (h2 : frisbees = 18)
  (h3 : yoyos = 18) :
  (total_prizes - (frisbees + yoyos) = 14) :=
by
  sorry

end stuffed_animals_count_l59_5939


namespace B_starts_after_A_l59_5986

theorem B_starts_after_A :
  ∀ (A_walk_speed B_cycle_speed dist_from_start t : ℝ), 
    A_walk_speed = 10 →
    B_cycle_speed = 20 →
    dist_from_start = 80 →
    B_cycle_speed * (dist_from_start - A_walk_speed * t) / A_walk_speed = t →
    t = 4 :=
by 
  intros A_walk_speed B_cycle_speed dist_from_start t hA_speed hB_speed hdist heq;
  sorry

end B_starts_after_A_l59_5986


namespace base9_subtraction_l59_5955

theorem base9_subtraction (a b : Nat) (h1 : a = 256) (h2 : b = 143) : 
  (a - b) = 113 := 
sorry

end base9_subtraction_l59_5955


namespace add_base8_l59_5925

-- Define the base 8 numbers 5_8 and 16_8
def five_base8 : ℕ := 5
def sixteen_base8 : ℕ := 1 * 8 + 6

-- Convert the result to base 8 from the sum in base 10
def sum_base8 (a b : ℕ) : ℕ :=
  let sum_base10 := a + b
  let d1 := sum_base10 / 8
  let d0 := sum_base10 % 8
  d1 * 10 + d0 

theorem add_base8 (x y : ℕ) (hx : x = five_base8) (hy : y = sixteen_base8) :
  sum_base8 x y = 23 :=
by
  sorry

end add_base8_l59_5925


namespace find_z_l59_5956

open Complex

theorem find_z (z : ℂ) : (1 + 2*I) * z = 3 - I → z = (1/5) - (7/5)*I :=
by
  intro h
  sorry

end find_z_l59_5956


namespace no_natural_number_divides_Q_by_x_squared_minus_one_l59_5926

def Q (n : ℕ) (x : ℝ) : ℝ := 1 + 5*x^2 + x^4 - (n - 1) * x^(n - 1) + (n - 8) * x^n

theorem no_natural_number_divides_Q_by_x_squared_minus_one :
  ∀ (n : ℕ), n > 0 → ¬ (x^2 - 1 ∣ Q n x) :=
by
  intros n h
  sorry

end no_natural_number_divides_Q_by_x_squared_minus_one_l59_5926


namespace power_of_five_trailing_zeros_l59_5994

theorem power_of_five_trailing_zeros (n : ℕ) (h : n = 1968) : 
  ∃ k : ℕ, 5^n = 10^k ∧ k ≥ 1968 := 
by 
  sorry

end power_of_five_trailing_zeros_l59_5994


namespace tourists_count_l59_5951

theorem tourists_count (n k : ℤ) (h1 : 2 * k % n = 1) (h2 : 3 * k % n = 13) : n = 23 := 
by
-- Proof is omitted
sorry

end tourists_count_l59_5951


namespace min_value_x_y_xy_l59_5924

theorem min_value_x_y_xy (x y : ℝ) (h : 2 * x^2 + 3 * x * y + 2 * y^2 = 1) :
  x + y + x * y ≥ -9 / 8 :=
sorry

end min_value_x_y_xy_l59_5924


namespace sum_of_acutes_tan_eq_pi_over_4_l59_5920

theorem sum_of_acutes_tan_eq_pi_over_4 {α β : ℝ} (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) 
    (h : (1 + Real.tan α) * (1 + Real.tan β) = 2) : α + β = π / 4 :=
sorry

end sum_of_acutes_tan_eq_pi_over_4_l59_5920


namespace find_circle_center_value_x_plus_y_l59_5922

theorem find_circle_center_value_x_plus_y : 
  ∀ (x y : ℝ), (x^2 + y^2 = 4 * x - 6 * y + 9) → 
    x + y = -1 :=
by
  intros x y h
  sorry

end find_circle_center_value_x_plus_y_l59_5922


namespace probability_odd_number_die_l59_5963

theorem probability_odd_number_die :
  let total_outcomes := 6
  let favorable_outcomes := 3
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ) = 1 / 2 :=
by
  sorry

end probability_odd_number_die_l59_5963


namespace percentage_supports_policy_l59_5947

theorem percentage_supports_policy
    (men_support_percentage : ℝ)
    (women_support_percentage : ℝ)
    (num_men : ℕ)
    (num_women : ℕ)
    (total_surveyed : ℕ)
    (total_supporters : ℕ)
    (overall_percentage : ℝ) :
    (men_support_percentage = 0.70) →
    (women_support_percentage = 0.75) →
    (num_men = 200) →
    (num_women = 800) →
    (total_surveyed = num_men + num_women) →
    (total_supporters = (men_support_percentage * num_men) + (women_support_percentage * num_women)) →
    (overall_percentage = (total_supporters / total_surveyed) * 100) →
    overall_percentage = 74 :=
by
  intros
  sorry

end percentage_supports_policy_l59_5947


namespace committee_count_l59_5937

theorem committee_count (total_students : ℕ) (include_students : ℕ) (choose_students : ℕ) :
  total_students = 8 → include_students = 2 → choose_students = 3 →
  Nat.choose (total_students - include_students) choose_students = 20 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end committee_count_l59_5937


namespace grazing_months_b_l59_5928

theorem grazing_months_b (a_oxen a_months b_oxen c_oxen c_months total_rent c_share : ℕ) (x : ℕ) 
  (h_a : a_oxen = 10) (h_am : a_months = 7) (h_b : b_oxen = 12) 
  (h_c : c_oxen = 15) (h_cm : c_months = 3) (h_tr : total_rent = 105) 
  (h_cs : c_share = 27) : 
  45 * 105 = 27 * (70 + 12 * x + 45) → x = 5 :=
by
  sorry

end grazing_months_b_l59_5928


namespace minimize_fencing_l59_5914

def area_requirement (w : ℝ) : Prop :=
  2 * (w * w) ≥ 800

def length_twice_width (l w : ℝ) : Prop :=
  l = 2 * w

def perimeter (w l : ℝ) : ℝ :=
  2 * l + 2 * w

theorem minimize_fencing (w l : ℝ) (h1 : area_requirement w) (h2 : length_twice_width l w) :
  w = 20 ∧ l = 40 :=
by
  sorry

end minimize_fencing_l59_5914


namespace coconut_grove_nut_yield_l59_5953

/--
In a coconut grove, the trees produce nuts based on some given conditions. Prove that the number of nuts produced by (x + 4) trees per year is 720 when x is 8. The conditions are:

1. (x + 4) trees yield a certain number of nuts per year.
2. x trees yield 120 nuts per year.
3. (x - 4) trees yield 180 nuts per year.
4. The average yield per year per tree is 100.
5. x is 8.
-/

theorem coconut_grove_nut_yield (x : ℕ) (y z w: ℕ) (h₁ : x = 8) (h₂ : y = 120) (h₃ : z = 180) (h₄ : w = 100) :
  ((x + 4) * w) - (x * y + (x - 4) * z) = 720 := 
by
  sorry

end coconut_grove_nut_yield_l59_5953


namespace total_tissues_l59_5933

-- define the number of students in each group
def g1 : Nat := 9
def g2 : Nat := 10
def g3 : Nat := 11

-- define the number of tissues per mini tissue box
def t : Nat := 40

-- state the main theorem
theorem total_tissues : (g1 + g2 + g3) * t = 1200 := by
  sorry

end total_tissues_l59_5933


namespace sum_slope_y_intercept_l59_5990

theorem sum_slope_y_intercept (A B C F : ℝ × ℝ) (midpoint_A_C : F = ((A.1 + C.1) / 2, (A.2 + C.2) / 2)) 
  (coords_A : A = (0, 6)) (coords_B : B = (0, 0)) (coords_C : C = (8, 0)) :
  let slope : ℝ := (F.2 - B.2) / (F.1 - B.1)
  let y_intercept : ℝ := B.2
  slope + y_intercept = 3 / 4 := by
{
  -- proof steps
  sorry
}

end sum_slope_y_intercept_l59_5990


namespace cost_per_item_l59_5978

theorem cost_per_item (total_profit : ℝ) (total_customers : ℕ) (purchase_percentage : ℝ) (pays_advertising : ℝ)
    (H1: total_profit = 1000)
    (H2: total_customers = 100)
    (H3: purchase_percentage = 0.80)
    (H4: pays_advertising = 1000)
    : (total_profit / (total_customers * purchase_percentage)) = 12.50 :=
by
  sorry

end cost_per_item_l59_5978


namespace solution_set_inequality_range_of_t_l59_5958

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x - 1

theorem solution_set_inequality :
  {x : ℝ | f x > 7} = {x : ℝ | x < -4} ∪ {x : ℝ | x > 2} :=
sorry

theorem range_of_t (t : ℝ) :
  (∀ x, 2 ≤ x ∧ x ≤ 4 → f (x - t) ≤ x - 2) ↔ 3 ≤ t ∧ t ≤ 3 + Real.sqrt 2 :=
sorry

end solution_set_inequality_range_of_t_l59_5958


namespace twenty_two_percent_of_three_hundred_l59_5981

theorem twenty_two_percent_of_three_hundred : 
  (22 / 100) * 300 = 66 :=
by
  sorry

end twenty_two_percent_of_three_hundred_l59_5981


namespace square_side_length_l59_5908

theorem square_side_length (radius : ℝ) (s1 s2 : ℝ) (h1 : s1 = s2) (h2 : radius = 2 - Real.sqrt 2):
  s1 = 1 :=
  sorry

end square_side_length_l59_5908


namespace third_even_number_sequence_l59_5948

theorem third_even_number_sequence (x : ℕ) (h_even : x % 2 = 0) (h_sum : x + (x + 2) + (x + 4) + (x + 6) + (x + 8) + (x + 10) = 180) : x + 4 = 30 :=
by
  sorry

end third_even_number_sequence_l59_5948


namespace solution_l59_5993

theorem solution (x y : ℝ) (h₁ : x + 3 * y = -1) (h₂ : x - 3 * y = 5) : x^2 - 9 * y^2 = -5 := 
by
  sorry

end solution_l59_5993


namespace abs_neg_one_ninth_l59_5909

theorem abs_neg_one_ninth : abs (- (1 / 9)) = 1 / 9 := by
  sorry

end abs_neg_one_ninth_l59_5909


namespace B_cycling_speed_l59_5962

theorem B_cycling_speed (v : ℝ) : 
  (∀ (t : ℝ), 10 * t + 30 = B_start_distance) ∧ 
  (B_start_distance = 60) ∧ 
  (t = 3) →
  v = 20 :=
sorry

end B_cycling_speed_l59_5962


namespace parabola_focus_coordinates_l59_5949

-- Define the given conditions
def parabola (a : ℝ) (x : ℝ) : ℝ := a * x^2
def passes_through (a : ℝ) (p : ℝ × ℝ) : Prop := p.snd = parabola a p.fst

-- Main theorem: Prove the coordinates of the focus
theorem parabola_focus_coordinates (a : ℝ) (h : passes_through a (1, 4)) (ha : a = 4) : (0, 1 / 16) = (0, 1 / (4 * a)) :=
by
  rw [ha] -- substitute the value of a
  simp -- simplify the expression
  sorry

end parabola_focus_coordinates_l59_5949


namespace least_number_divisor_l59_5934

theorem least_number_divisor (d : ℕ) (n m : ℕ) 
  (h1 : d = 1081)
  (h2 : m = 1077)
  (h3 : n = 4)
  (h4 : ∃ k, m + n = k * d) :
  d = 1081 :=
by
  sorry

end least_number_divisor_l59_5934


namespace no_coprime_odd_numbers_for_6_8_10_l59_5970

theorem no_coprime_odd_numbers_for_6_8_10 :
  ∀ (m n : ℤ), m > n ∧ n > 0 ∧ (m.gcd n = 1) ∧ (m % 2 = 1) ∧ (n % 2 = 1) →
    (1 / 2 : ℚ) * (m^2 - n^2) ≠ 6 ∨ (m * n) ≠ 8 ∨ (1 / 2 : ℚ) * (m^2 + n^2) ≠ 10 :=
by
  sorry

end no_coprime_odd_numbers_for_6_8_10_l59_5970


namespace time_ratio_upstream_downstream_l59_5938

theorem time_ratio_upstream_downstream (S_boat S_stream D : ℝ) (h1 : S_boat = 72) (h2 : S_stream = 24) :
  let time_upstream := D / (S_boat - S_stream)
  let time_downstream := D / (S_boat + S_stream)
  (time_upstream / time_downstream) = 2 :=
by
  sorry

end time_ratio_upstream_downstream_l59_5938


namespace jeff_cat_shelter_l59_5905

theorem jeff_cat_shelter :
  let initial_cats := 20
  let monday_cats := 2
  let tuesday_cats := 1
  let people_adopted := 3
  let cats_per_person := 2
  let total_cats := initial_cats + monday_cats + tuesday_cats
  let adopted_cats := people_adopted * cats_per_person
  total_cats - adopted_cats = 17 := 
by
  sorry

end jeff_cat_shelter_l59_5905


namespace monkey_swinging_speed_l59_5932

namespace LamplighterMonkey

def running_speed : ℝ := 15
def running_time : ℝ := 5
def swinging_time : ℝ := 10
def total_distance : ℝ := 175

theorem monkey_swinging_speed : 
  (total_distance = running_speed * running_time + (running_speed / swinging_time) * swinging_time) → 
  (running_speed / swinging_time = 10) := 
by 
  intros h
  sorry

end LamplighterMonkey

end monkey_swinging_speed_l59_5932


namespace find_x_l59_5959

theorem find_x (x : ℝ) (a b c : ℝ × ℝ)
  (ha : a = (x, 1))
  (hb : b = (2, x))
  (hc : c = (1, -2))
  (h_perpendicular : (a.1 * (b.1 + c.1) + a.2 * (b.2 + c.2)) = 0) :
  x = 1 / 2 :=
sorry

end find_x_l59_5959


namespace gcd_104_156_l59_5912

theorem gcd_104_156 : Nat.gcd 104 156 = 52 :=
by
  -- the proof steps will go here, but we can use sorry to skip it
  sorry

end gcd_104_156_l59_5912


namespace volume_of_cube_l59_5979

-- Definition of the surface area condition
def surface_area_condition (s : ℝ) : Prop :=
  6 * s^2 = 150

-- The main theorem to prove
theorem volume_of_cube (s : ℝ) (h : surface_area_condition s) : s^3 = 125 :=
by
  sorry

end volume_of_cube_l59_5979


namespace expand_expression_l59_5904

theorem expand_expression (x : ℝ) : 25 * (3 * x - 4) = 75 * x - 100 := 
by 
  sorry

end expand_expression_l59_5904


namespace set_union_example_l59_5952

theorem set_union_example (x : ℕ) (M N : Set ℕ) (h1 : M = {0, x}) (h2 : N = {1, 2}) (h3 : M ∩ N = {2}) :
  M ∪ N = {0, 1, 2} := by
  sorry

end set_union_example_l59_5952


namespace cotangent_identity_l59_5995

noncomputable def cotangent (θ : ℝ) : ℝ := 1 / Real.tan θ

theorem cotangent_identity (x : ℝ) (i : ℂ) (n : ℕ) (k : ℕ) (h : (0 < k) ∧ (k < n)) :
  ((x + i) / (x - i))^n = 1 → x = cotangent (k * Real.pi / n) := 
sorry

end cotangent_identity_l59_5995


namespace cost_of_article_l59_5973

variable {C G : ℝ}

theorem cost_of_article (h : 350 = C * (1 + (G + 5) / 100)) (h' : 340 = C * (1 + G / 100)) : C = 200 := by
  sorry

end cost_of_article_l59_5973


namespace find_three_digit_number_l59_5940

theorem find_three_digit_number (a b c : ℕ) (ha : a ≠ b) (hb : b ≠ c) (hc : a ≠ c)
  (h_sum : 122 * a + 212 * b + 221 * c = 2003) :
  100 * a + 10 * b + c = 345 :=
by
  sorry

end find_three_digit_number_l59_5940


namespace determine_g_function_l59_5910

theorem determine_g_function (t x : ℝ) (g : ℝ → ℝ) 
  (line_eq : ∀ x y : ℝ, y = 2 * x - 40) 
  (param_eq : ∀ t : ℝ, (x, 20 * t - 14) = (g t, 20 * t - 14)) :
  g t = 10 * t + 13 :=
by 
  sorry

end determine_g_function_l59_5910


namespace smallest_n_condition_l59_5992

-- Define the conditions
def condition1 (x : ℤ) : Prop := 2 * x - 3 ≡ 0 [ZMOD 13]
def condition2 (y : ℤ) : Prop := 3 * y + 4 ≡ 0 [ZMOD 13]

-- Problem statement: finding n such that the expression is a multiple of 13
theorem smallest_n_condition (x y : ℤ) (n : ℤ) :
  condition1 x → condition2 y → x^2 - x * y + y^2 + n ≡ 0 [ZMOD 13] → n = 1 := 
by
  sorry

end smallest_n_condition_l59_5992


namespace bryan_samples_l59_5980

noncomputable def initial_samples_per_shelf : ℕ := 128
noncomputable def shelves : ℕ := 13
noncomputable def samples_removed_per_shelf : ℕ := 2
noncomputable def remaining_samples_per_shelf := initial_samples_per_shelf - samples_removed_per_shelf
noncomputable def total_remaining_samples := remaining_samples_per_shelf * shelves

theorem bryan_samples : total_remaining_samples = 1638 := 
by 
  sorry

end bryan_samples_l59_5980


namespace time_to_eat_cereal_l59_5954

noncomputable def MrFatRate : ℝ := 1 / 40
noncomputable def MrThinRate : ℝ := 1 / 15
noncomputable def CombinedRate : ℝ := MrFatRate + MrThinRate
noncomputable def CerealAmount : ℝ := 4
noncomputable def TimeToFinish : ℝ := CerealAmount / CombinedRate
noncomputable def expected_time : ℝ := 96

theorem time_to_eat_cereal :
  TimeToFinish = expected_time :=
by
  sorry

end time_to_eat_cereal_l59_5954


namespace second_term_of_geometric_series_l59_5929

theorem second_term_of_geometric_series (a r S: ℝ) (h_r : r = 1/4) (h_S : S = 40) (h_geom_sum : S = a / (1 - r)) : a * r = 7.5 :=
by
  sorry

end second_term_of_geometric_series_l59_5929


namespace mean_of_remaining_four_numbers_l59_5971

theorem mean_of_remaining_four_numbers 
  (a b c d max_num : ℝ) 
  (h1 : max_num = 105) 
  (h2 : (a + b + c + d + max_num) / 5 = 92) : 
  (a + b + c + d) / 4 = 88.75 :=
by
  sorry

end mean_of_remaining_four_numbers_l59_5971


namespace trader_sold_40_meters_l59_5943

noncomputable def meters_of_cloth_sold (profit_per_meter total_profit : ℕ) : ℕ :=
  total_profit / profit_per_meter

theorem trader_sold_40_meters (profit_per_meter total_profit : ℕ) (h1 : profit_per_meter = 35) (h2 : total_profit = 1400) :
  meters_of_cloth_sold profit_per_meter total_profit = 40 :=
by
  sorry

end trader_sold_40_meters_l59_5943


namespace each_cow_gives_5_liters_per_day_l59_5997

-- Define conditions
def cows : ℕ := 52
def weekly_milk : ℕ := 1820
def days_in_week : ℕ := 7

-- Define daily_milk as the daily milk production
def daily_milk := weekly_milk / days_in_week

-- Define milk_per_cow as the amount of milk each cow produces per day
def milk_per_cow := daily_milk / cows

-- Statement to prove
theorem each_cow_gives_5_liters_per_day : milk_per_cow = 5 :=
by
  -- This is where you would normally fill in the proof steps
  sorry

end each_cow_gives_5_liters_per_day_l59_5997


namespace triangle_angles_30_60_90_l59_5901

-- Definition of the angles based on the given ratio
def angles_ratio (A B C : ℝ) : Prop :=
  A / B = 1 / 2 ∧ B / C = 2 / 3

-- The main statement to be proved
theorem triangle_angles_30_60_90
  (A B C : ℝ)
  (h1 : angles_ratio A B C)
  (h2 : A + B + C = 180) :
  A = 30 ∧ B = 60 ∧ C = 90 := 
sorry

end triangle_angles_30_60_90_l59_5901


namespace hyperbola_equation_l59_5989

-- Define the conditions of the problem
def center_at_origin (x y : ℝ) : Prop := x = 0 ∧ y = 0
def focus_on_y_axis (x : ℝ) : Prop := x = 0
def focal_distance (d : ℝ) : Prop := d = 4
def point_on_hyperbola (x y : ℝ) : Prop := x = 1 ∧ y = -Real.sqrt 3

-- Final statement to prove
theorem hyperbola_equation :
  (center_at_origin 0 0) ∧
  (focus_on_y_axis 0) ∧
  (focal_distance 4) ∧
  (point_on_hyperbola 1 (-Real.sqrt 3))
  → ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (a = Real.sqrt 3 ∧ b = 1) ∧ (∀ x y : ℝ, x^2 - (y^2 / 3) = 1) :=
by
  sorry

end hyperbola_equation_l59_5989


namespace intersection_of_asymptotes_l59_5941

noncomputable def f (x : ℝ) : ℝ := (x^2 - 6 * x + 8) / (x^2 - 6 * x + 9)

theorem intersection_of_asymptotes :
  ∃ (p : ℝ × ℝ), p = (3, 1) ∧
    (∀ (x : ℝ), x ≠ 3 → f x ≠ 1) ∧
    ((∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - 3| ∧ |x - 3| < δ → |f x - 1| < ε) ∧ 
    (∀ ε > 0, ∃ δ > 0, ∀ y, 0 < |y - 1| ∧ |y - 1| < δ → |f (3 + y) - 1| < ε)) :=
by
  sorry

end intersection_of_asymptotes_l59_5941


namespace female_managers_count_l59_5976

-- Definitions based on conditions
def total_employees : Nat := 250
def female_employees : Nat := 90
def total_managers : Nat := 40
def male_associates : Nat := 160

-- Statement to prove
theorem female_managers_count : (total_managers = 40) :=
by
  sorry

end female_managers_count_l59_5976
