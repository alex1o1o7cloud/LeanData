import Mathlib

namespace other_juice_cost_l224_224174

theorem other_juice_cost (total_spent : ℕ := 94)
    (mango_cost_per_glass : ℕ := 5)
    (other_total_spent : ℕ := 54)
    (total_people : ℕ := 17) : 
  other_total_spent / (total_people - (total_spent - other_total_spent) / mango_cost_per_glass) = 6 := 
sorry

end other_juice_cost_l224_224174


namespace bernoulli_inequality_l224_224167

theorem bernoulli_inequality (x : ℝ) (n : ℕ) (hx : x > -1) (hn : n > 0) : 
  (1 + x) ^ n ≥ 1 + n * x := 
sorry

end bernoulli_inequality_l224_224167


namespace total_matches_in_2006_world_cup_l224_224245

-- Define relevant variables and conditions
def teams := 32
def groups := 8
def top2_from_each_group := 16

-- Calculate the number of matches in Group Stage
def matches_in_group_stage :=
  let matches_per_group := 6
  matches_per_group * groups

-- Calculate the number of matches in Knockout Stage
def matches_in_knockout_stage :=
  let first_round_matches := 8
  let quarter_final_matches := 4
  let semi_final_matches := 2
  let final_and_third_place_matches := 2
  first_round_matches + quarter_final_matches + semi_final_matches + final_and_third_place_matches

-- Total number of matches
theorem total_matches_in_2006_world_cup : matches_in_group_stage + matches_in_knockout_stage = 64 := by
  sorry

end total_matches_in_2006_world_cup_l224_224245


namespace distance_traveled_on_fifth_day_equals_12_li_l224_224008

theorem distance_traveled_on_fifth_day_equals_12_li:
  ∀ {a_1 : ℝ},
    (a_1 * ((1 - (1 / 2) ^ 6) / (1 - 1 / 2)) = 378) →
    (a_1 * (1 / 2) ^ 4 = 12) :=
by
  intros a_1 h
  sorry

end distance_traveled_on_fifth_day_equals_12_li_l224_224008


namespace solve_system_of_inequalities_l224_224428

theorem solve_system_of_inequalities (x : ℝ) :
  ( (x - 2) / (x - 1) < 1 ) ∧ ( -x^2 + x + 2 < 0 ) → x > 2 :=
by
  sorry

end solve_system_of_inequalities_l224_224428


namespace y_increase_by_30_when_x_increases_by_12_l224_224846

theorem y_increase_by_30_when_x_increases_by_12
  (h : ∀ x y : ℝ, x = 4 → y = 10)
  (x_increase : ℝ := 12) :
  ∃ y_increase : ℝ, y_increase = 30 :=
by
  -- Here we assume the condition h and x_increase
  let ratio := 10 / 4  -- Establish the ratio of increase
  let expected_y_increase := x_increase * ratio
  exact ⟨expected_y_increase, sorry⟩  -- Prove it is 30

end y_increase_by_30_when_x_increases_by_12_l224_224846


namespace Jose_Raju_Work_Together_l224_224025

-- Definitions for the conditions
def JoseWorkRate : ℚ := 1 / 10
def RajuWorkRate : ℚ := 1 / 40
def CombinedWorkRate : ℚ := JoseWorkRate + RajuWorkRate

-- Theorem statement
theorem Jose_Raju_Work_Together :
  1 / CombinedWorkRate = 8 := by
    sorry

end Jose_Raju_Work_Together_l224_224025


namespace ap_number_of_terms_is_six_l224_224340

noncomputable def arithmetic_progression_number_of_terms (a d : ℕ) (n : ℕ) : Prop :=
  let odd_sum := (n / 2) * (2 * a + (n - 2) * d)
  let even_sum := (n / 2) * (2 * a + n * d)
  let last_term_condition := (n - 1) * d = 15
  n % 2 = 0 ∧ odd_sum = 30 ∧ even_sum = 36 ∧ last_term_condition

theorem ap_number_of_terms_is_six (a d n : ℕ) (h : arithmetic_progression_number_of_terms a d n) :
  n = 6 :=
by sorry

end ap_number_of_terms_is_six_l224_224340


namespace father_l224_224505

variable (R F M : ℕ)
variable (h1 : F = 4 * R)
variable (h2 : 4 * R + 8 = M * (R + 8))
variable (h3 : 4 * R + 16 = 2 * (R + 16))

theorem father's_age_ratio (hR : R = 8) : (F + 8) / (R + 8) = 5 / 2 := by
  sorry

end father_l224_224505


namespace quadratic_inequality_solution_l224_224311

theorem quadratic_inequality_solution :
  {x : ℝ | 2 * x ^ 2 - x - 3 > 0} = {x : ℝ | x > 3 / 2 ∨ x < -1} :=
sorry

end quadratic_inequality_solution_l224_224311


namespace find_f_at_75_l224_224553

variables (f : ℝ → ℝ) (h₀ : ∀ x, f (x + 2) = -f x)
variables (h₁ : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x)
variables (h₂ : ∀ x, f (-x) = -f x)

theorem find_f_at_75 : f 7.5 = -0.5 := by
  sorry

end find_f_at_75_l224_224553


namespace sum_of_first_five_terms_geo_seq_l224_224201

theorem sum_of_first_five_terms_geo_seq 
  (a : ℚ) (r : ℚ) (n : ℕ) 
  (h_a : a = 1 / 3)
  (h_r : r = 1 / 3)
  (h_n : n = 5) :
  (∑ i in Finset.range n, a * r^i) = 121 / 243 :=
by
  sorry

end sum_of_first_five_terms_geo_seq_l224_224201


namespace number_of_sides_of_regular_polygon_l224_224656

theorem number_of_sides_of_regular_polygon (n : ℕ) (h : 0 < n) (h_angle : ∀ i, i < n → (2 * n - 4) * 90 / n = 150) : n = 12 :=
sorry

end number_of_sides_of_regular_polygon_l224_224656


namespace arithmetic_seq_problem_l224_224212

-- Conditions and definitions for the arithmetic sequence
def arithmetic_seq (a_n : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a_n (n+1) = a_n n + d

def sum_seq (a_n S_n : ℕ → ℝ) : Prop :=
  ∀ n, S_n n = (n * (a_n 1 + a_n n)) / 2

def T_plus_K_eq_19 (T K : ℕ) : Prop :=
  T + K = 19

-- The given problem to prove
theorem arithmetic_seq_problem (a_n S_n : ℕ → ℝ) (d : ℝ) (h1 : d > 0)
  (h2 : arithmetic_seq a_n d) (h3 : sum_seq a_n S_n)
  (h4 : ∀ T K, T_plus_K_eq_19 T K → S_n T = S_n K) :
  ∃! n, a_n n - S_n n ≥ 0 := sorry

end arithmetic_seq_problem_l224_224212


namespace mean_cat_weights_l224_224313

-- Define a list representing the weights of the cats from the stem-and-leaf plot
def cat_weights : List ℕ := [12, 13, 14, 20, 21, 21, 25, 25, 28, 30, 31, 32, 32, 36, 38, 39, 39]

-- Function to calculate the sum of elements in a list
def sum_list (l : List ℕ) : ℕ := l.foldr (· + ·) 0

-- Function to calculate the mean of a list of natural numbers
def mean_list (l : List ℕ) : ℚ := (sum_list l : ℚ) / l.length

-- The theorem we need to prove
theorem mean_cat_weights : mean_list cat_weights = 27 := by 
  sorry

end mean_cat_weights_l224_224313


namespace crushing_load_value_l224_224057

-- Define the given formula and values
def T : ℕ := 3
def H : ℕ := 9
def K : ℕ := 2

-- Given formula for L
def L (T H K : ℕ) : ℚ := 50 * T^5 / (K * H^3)

-- Prove that L = 8 + 1/3
theorem crushing_load_value :
  L T H K = 8 + 1 / 3 :=
by
  sorry

end crushing_load_value_l224_224057


namespace susie_rooms_l224_224754

-- Define the conditions
def vacuum_time_per_room : ℕ := 20  -- in minutes
def total_vacuum_time : ℕ := 2 * 60  -- 2 hours in minutes

-- Define the number of rooms in Susie's house
def number_of_rooms (total_time room_time : ℕ) : ℕ := total_time / room_time

-- Prove that Susie has 6 rooms in her house
theorem susie_rooms : number_of_rooms total_vacuum_time vacuum_time_per_room = 6 :=
by
  sorry -- proof goes here

end susie_rooms_l224_224754


namespace largest_prime_factor_of_set_l224_224766

def largest_prime_factor (n : ℕ) : ℕ :=
  -- pseudo-code for determining the largest prime factor of n
  sorry

lemma largest_prime_factor_45 : largest_prime_factor 45 = 5 := sorry
lemma largest_prime_factor_65 : largest_prime_factor 65 = 13 := sorry
lemma largest_prime_factor_85 : largest_prime_factor 85 = 17 := sorry
lemma largest_prime_factor_119 : largest_prime_factor 119 = 17 := sorry
lemma largest_prime_factor_143 : largest_prime_factor 143 = 13 := sorry

theorem largest_prime_factor_of_set :
  max (largest_prime_factor 45)
    (max (largest_prime_factor 65)
      (max (largest_prime_factor 85)
        (max (largest_prime_factor 119)
          (largest_prime_factor 143)))) = 17 :=
by
  rw [largest_prime_factor_45,
      largest_prime_factor_65,
      largest_prime_factor_85,
      largest_prime_factor_119,
      largest_prime_factor_143]
  sorry

end largest_prime_factor_of_set_l224_224766


namespace fraction_power_mult_equality_l224_224803

-- Define the fraction and the power
def fraction := (1 : ℚ) / 3
def power : ℚ := fraction ^ 4

-- Define the multiplication
def result := 8 * power

-- Prove the equality
theorem fraction_power_mult_equality : result = 8 / 81 := by
  sorry

end fraction_power_mult_equality_l224_224803


namespace trigonometric_identity_l224_224209

theorem trigonometric_identity (x : ℝ) (h : Real.sin (x + Real.pi / 4) = 1 / 3) :
  Real.sin (4 * x) - 2 * Real.cos (3 * x) * Real.sin x = -7 / 9 :=
by
  sorry

end trigonometric_identity_l224_224209


namespace find_p_l224_224823

noncomputable def binomial_parameter (n : ℕ) (p : ℚ) (E : ℚ) (D : ℚ) : Prop :=
  E = n * p ∧ D = n * p * (1 - p)

theorem find_p (n : ℕ) (p : ℚ) 
  (hE : n * p = 50)
  (hD : n * p * (1 - p) = 30)
  : p = 2 / 5 :=
sorry

end find_p_l224_224823


namespace cos_60_degrees_is_one_half_l224_224633

-- Define the given conditions in Lean 4
def angle_60_degrees : Real :=
  60 * Real.pi / 180

def unit_circle_point (θ : Real) : Real × Real :=
  (Real.cos θ, Real.sin θ)

-- State the proof problem in Lean 4
theorem cos_60_degrees_is_one_half : 
  (unit_circle_point angle_60_degrees).1 = 1 / 2 := 
sorry

end cos_60_degrees_is_one_half_l224_224633


namespace find_f2009_l224_224646

noncomputable def f : ℝ → ℝ :=
sorry

axiom even_function (x : ℝ) : f x = f (-x)
axiom functional_equation (x : ℝ) : f (2 + x) = -f (2 - x)
axiom initial_condition : f (-3) = -2

theorem find_f2009 : f 2009 = 2 :=
sorry

end find_f2009_l224_224646


namespace rectangle_diagonals_equal_l224_224186

-- Define the properties of a rectangle
def is_rectangle (AB CD AD BC : ℝ) (diagonal1 diagonal2 : ℝ) : Prop :=
  AB = CD ∧ AD = BC ∧ diagonal1 = diagonal2

-- State the theorem to prove that the diagonals of a rectangle are equal
theorem rectangle_diagonals_equal (AB CD AD BC diagonal1 diagonal2 : ℝ) (h : is_rectangle AB CD AD BC diagonal1 diagonal2) :
  diagonal1 = diagonal2 :=
by
  sorry

end rectangle_diagonals_equal_l224_224186


namespace tan_22_5_expression_l224_224894

theorem tan_22_5_expression :
  let a := 2
  let b := 1
  let c := 0
  let d := 0
  let t := Real.tan (Real.pi / 8)
  a ≥ b ∧ b ≥ c ∧ c ≥ d ∧
  t = (Real.sqrt a) - (Real.sqrt b) + (Real.sqrt c) - d → 
  a + b + c + d = 3 :=
by
  intros
  exact sorry

end tan_22_5_expression_l224_224894


namespace radius_of_garden_outer_boundary_l224_224035

-- Definitions based on the conditions from the problem statement
def fountain_diameter : ℝ := 12
def garden_width : ℝ := 10

-- Question translated to a proof statement
theorem radius_of_garden_outer_boundary :
  (fountain_diameter / 2 + garden_width) = 16 := 
by 
  sorry

end radius_of_garden_outer_boundary_l224_224035


namespace problem1_problem2_problem3_l224_224816

-- Definitions of transformations and final sequence S
def transformation (A : List ℕ) : List ℕ := 
  match A with
  | x :: y :: xs => (x + y) :: transformation (y :: xs)
  | _ => []

def nth_transform (A : List ℕ) (n : ℕ) : List ℕ :=
  Nat.iterate (λ L => transformation L) n A

def final_sequence (A : List ℕ) : ℕ :=
  match nth_transform A (A.length - 1) with
  | [x] => x
  | _ => 0

-- Proof Statements

theorem problem1 : final_sequence [1, 2, 3] = 8 := sorry

theorem problem2 (n : ℕ) : final_sequence (List.range (n+1)) = (n + 2) * 2 ^ (n - 1) := sorry

theorem problem3 (A B : List ℕ) (h : A = List.range (B.length)) (h_perm : B.permutations.contains A) : 
  final_sequence B = final_sequence A := by
  sorry

end problem1_problem2_problem3_l224_224816


namespace part1_extreme_value_part2_range_of_a_l224_224228

noncomputable def f (x : ℝ) : ℝ := x * Real.exp (x + 1)

theorem part1_extreme_value :
  ∃ x : ℝ, f x = -1 :=
  sorry

theorem part2_range_of_a :
  ∀ x > 0, ∃ a : ℝ, f x ≥ x + Real.log x + a + 1 → a ≤ 1 :=
  sorry

end part1_extreme_value_part2_range_of_a_l224_224228


namespace avg_hamburgers_per_day_l224_224928

theorem avg_hamburgers_per_day (total_hamburgers : ℕ) (days_in_week : ℕ) (h1 : total_hamburgers = 49) (h2 : days_in_week = 7) :
  total_hamburgers / days_in_week = 7 :=
by {
  sorry
}

end avg_hamburgers_per_day_l224_224928


namespace pow137_mod8_l224_224459

theorem pow137_mod8 : (5 ^ 137) % 8 = 5 := by
  -- Use the provided conditions
  have h1: 5 % 8 = 5 := by norm_num
  have h2: (5 ^ 2) % 8 = 1 := by norm_num
  sorry

end pow137_mod8_l224_224459


namespace electronics_weight_l224_224310

variable (B C E : ℝ)
variable (h1 : B / (B * (4 / 7) - 8) = 2 * (B / (B * (4 / 7))))
variable (h2 : C = B * (4 / 7))
variable (h3 : E = B * (3 / 7))

theorem electronics_weight : E = 12 := by
  sorry

end electronics_weight_l224_224310


namespace equal_expression_exists_l224_224535

-- lean statement for the mathematical problem
theorem equal_expression_exists (a b : ℤ) :
  ∃ (expr : ℤ), expr = 20 * a - 18 * b := by
  sorry

end equal_expression_exists_l224_224535


namespace find_pairs_l224_224193

theorem find_pairs (m n : ℕ) : 
  (20^m - 10 * m^2 + 1 = 19^n ↔ (m = 0 ∧ n = 0) ∨ (m = 2 ∧ n = 2)) :=
by
  sorry

end find_pairs_l224_224193


namespace find_max_value_l224_224368

-- We define the conditions as Lean definitions and hypotheses
def is_distinct_digits (A B C D E F : ℕ) : Prop :=
  (A ≠ B) ∧ (A ≠ C) ∧ (A ≠ D) ∧ (A ≠ E) ∧ (A ≠ F) ∧
  (B ≠ C) ∧ (B ≠ D) ∧ (B ≠ E) ∧ (B ≠ F) ∧
  (C ≠ D) ∧ (C ≠ E) ∧ (C ≠ F) ∧
  (D ≠ E) ∧ (D ≠ F) ∧
  (E ≠ F)

def all_digits_in_range (A B C D E F : ℕ) : Prop :=
  (1 ≤ A) ∧ (A ≤ 8) ∧
  (1 ≤ B) ∧ (B ≤ 8) ∧
  (1 ≤ C) ∧ (C ≤ 8) ∧
  (1 ≤ D) ∧ (D ≤ 8) ∧
  (1 ≤ E) ∧ (E ≤ 8) ∧
  (1 ≤ F) ∧ (F ≤ 8)

def divisible_by_99 (n : ℕ) : Prop :=
  (n % 99 = 0)

theorem find_max_value (A B C D E F : ℕ) :
  is_distinct_digits A B C D E F →
  all_digits_in_range A B C D E F →
  divisible_by_99 (100000 * A + 10000 * B + 1000 * C + 100 * D + 10 * E + F) →
  100000 * A + 10000 * B + 1000 * C + 100 * D + 10 * E + F = 87653412 :=
sorry

end find_max_value_l224_224368


namespace probability_at_least_one_boy_one_girl_l224_224877

def total_members : ℕ := 20
def boys : ℕ := 10
def girls : ℕ := 10
def committee_size : ℕ := 4

theorem probability_at_least_one_boy_one_girl :
  (let total_committees := (Nat.choose total_members committee_size)
   let all_boys_committees := (Nat.choose boys committee_size)
   let all_girls_committees := (Nat.choose girls committee_size)
   let unwanted_committees := all_boys_committees + all_girls_committees
   let desired_probability := 1 - (unwanted_committees / total_committees : ℚ)
   in desired_probability = (295 / 323 : ℚ)) := 
by
  sorry

end probability_at_least_one_boy_one_girl_l224_224877


namespace koala_fiber_absorption_l224_224106

theorem koala_fiber_absorption (x : ℝ) (hx : 0.30 * x = 12) : x = 40 :=
by
  sorry

end koala_fiber_absorption_l224_224106


namespace marshmallow_ratio_l224_224085

theorem marshmallow_ratio:
  (∀ h m b, 
    h = 8 ∧ 
    m = 3 * h ∧ 
    h + m + b = 44
  ) → (1 / 2 = b / m) :=
by
sorry

end marshmallow_ratio_l224_224085


namespace distance_downstream_in_12min_l224_224602

-- Define the given constants
def boat_speed_still_water : ℝ := 15  -- km/hr
def current_speed : ℝ := 3  -- km/hr
def time_minutes : ℝ := 12  -- minutes

-- Prove the distance traveled downstream in 12 minutes
theorem distance_downstream_in_12min
  (b_velocity_still : ℝ)
  (c_velocity : ℝ)
  (time_m : ℝ)
  (h1 : b_velocity_still = boat_speed_still_water)
  (h2 : c_velocity = current_speed)
  (h3 : time_m = time_minutes) :
  let effective_speed := b_velocity_still + c_velocity
  let effective_speed_km_per_min := effective_speed / 60
  let distance := effective_speed_km_per_min * time_m
  distance = 3.6 :=
by
  sorry

end distance_downstream_in_12min_l224_224602


namespace sum_of_quotient_and_reciprocal_l224_224750

theorem sum_of_quotient_and_reciprocal (x y : ℝ) (h1 : x + y = 45) (h2 : x * y = 500) : 
    (x / y + y / x) = 41 / 20 := 
sorry

end sum_of_quotient_and_reciprocal_l224_224750


namespace number_of_ways_to_divide_friends_l224_224838

theorem number_of_ways_to_divide_friends :
  (4 ^ 8 = 65536) := by
  -- result obtained through the multiplication principle
  sorry

end number_of_ways_to_divide_friends_l224_224838


namespace line_equation_through_point_and_area_l224_224037

theorem line_equation_through_point_and_area (b S x y : ℝ) 
  (h1 : ∀ y, (x, y) = (-2*b, 0) → True) 
  (h2 : ∀ p1 p2 p3 : ℝ × ℝ, p1 = (-2*b, 0) → p2 = (0, 0) → 
        ∃ k, p3 = (0, k) ∧ S = 1/2 * (2*b) * k) : 2*S*x - b^2*y + 4*b*S = 0 :=
sorry

end line_equation_through_point_and_area_l224_224037


namespace incorrect_statement_among_props_l224_224716

theorem incorrect_statement_among_props 
    (A: Prop := True)  -- Axioms in mathematics are accepted truths that do not require proof.
    (B: Prop := True)  -- A mathematical proof can proceed in different valid sequences depending on the approach and insights.
    (C: Prop := True)  -- All concepts utilized in a proof must be clearly defined before their use in arguments.
    (D: Prop := False) -- Logical deductions based on false premises can lead to valid conclusions.
    (E: Prop := True): -- Proof by contradiction only needs one assumption to be negated and shown to lead to a contradiction to be valid.
  ¬D := 
by sorry

end incorrect_statement_among_props_l224_224716


namespace regular_polygon_sides_l224_224652

theorem regular_polygon_sides (h : ∀ n, (180 * (n - 2) = 150 * n)) : n = 12 := 
by
  sorry

end regular_polygon_sides_l224_224652


namespace depreciation_rate_l224_224926

theorem depreciation_rate (initial_value final_value : ℝ) (years : ℕ) (r : ℝ)
  (h_initial : initial_value = 128000)
  (h_final : final_value = 54000)
  (h_years : years = 3)
  (h_equation : final_value = initial_value * (1 - r) ^ years) :
  r = 0.247 :=
sorry

end depreciation_rate_l224_224926


namespace like_terms_product_l224_224216

theorem like_terms_product :
  ∀ (m n : ℕ),
    (-x^3 * y^n) = (3 * x^m * y^2) → (m = 3 ∧ n = 2) → m * n = 6 :=
by
  intros m n h1 h2
  sorry

end like_terms_product_l224_224216


namespace sum_of_possible_values_l224_224262

theorem sum_of_possible_values (x y : ℝ) (h : x * y - (2 * x) / (y ^ 3) - (2 * y) / (x ^ 3) = 6) :
  ∃ (a b : ℝ), (a - 2) * (b - 2) = 4 ∧ (a - 2) * (b - 2) = 9 ∧ 4 + 9 = 13 :=
sorry

end sum_of_possible_values_l224_224262


namespace cos_60_eq_sqrt3_div_2_l224_224632

theorem cos_60_eq_sqrt3_div_2 : Real.cos (60 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry

end cos_60_eq_sqrt3_div_2_l224_224632


namespace min_sqrt_diff_l224_224721

theorem min_sqrt_diff (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  ∃ x y : ℕ, x = (p - 1) / 2 ∧ y = (p + 1) / 2 ∧ x ≤ y ∧
    ∀ a b : ℕ, (a ≤ b) → (Real.sqrt (2 * p) - Real.sqrt a - Real.sqrt b ≥ 0) → 
      (Real.sqrt (2 * p) - Real.sqrt x - Real.sqrt y) ≤ (Real.sqrt (2 * p) - Real.sqrt a - Real.sqrt b) := 
by 
  -- Proof to be filled in
  sorry

end min_sqrt_diff_l224_224721


namespace smallest_possible_value_l224_224840

theorem smallest_possible_value (x : ℝ) (hx : 11 = x^2 + 1 / x^2) :
  x + 1 / x = -Real.sqrt 13 :=
by
  sorry

end smallest_possible_value_l224_224840


namespace cos_60_eq_sqrt3_div_2_l224_224631

theorem cos_60_eq_sqrt3_div_2 : Real.cos (60 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry

end cos_60_eq_sqrt3_div_2_l224_224631


namespace cos_A_value_l224_224531

theorem cos_A_value (A B C : ℝ) 
  (A_internal : A + B + C = Real.pi) 
  (cos_B : Real.cos B = 1 / 2)
  (sin_C : Real.sin C = 3 / 5) : 
  Real.cos A = (3 * Real.sqrt 3 - 4) / 10 := 
by
  sorry

end cos_A_value_l224_224531


namespace sequence_a5_l224_224396

/-- In the sequence {a_n}, with a_1 = 1, a_2 = 2, and a_(n+2) = 2 * a_(n+1) + a_n, prove that a_5 = 29. -/
theorem sequence_a5 (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : a 2 = 2) (h_rec : ∀ n, a (n + 2) = 2 * a (n + 1) + a n) :
  a 5 = 29 :=
sorry

end sequence_a5_l224_224396


namespace calc_h_one_l224_224253

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 + 4 * x + 6
noncomputable def g (x : ℝ) : ℝ := Real.exp (f x) - 3
noncomputable def h (x : ℝ) : ℝ := f (g x)

-- the final theorem that we are proving
theorem calc_h_one : h 1 = 3 * Real.exp 26 - 14 * Real.exp 13 + 21 := by
  sorry

end calc_h_one_l224_224253


namespace ways_to_divide_8_friends_l224_224833

theorem ways_to_divide_8_friends : (4 : ℕ) ^ 8 = 65536 := by
  sorry

end ways_to_divide_8_friends_l224_224833


namespace find_f_neg3_l224_224293

theorem find_f_neg3 : 
  (∃ f : ℝ → ℝ, (∀ x : ℝ, x ≠ 0 → 5 * f (1 / x) + 3 * f x / x = 2 * x^2) ∧ f (-3) = 14029 / 72) :=
sorry

end find_f_neg3_l224_224293


namespace find_triangle_sides_l224_224544

theorem find_triangle_sides (x y : ℕ) : 
  (x * y = 200) ∧ (x + 2 * y = 50) → ((x = 40 ∧ y = 5) ∨ (x = 10 ∧ y = 20)) := 
by
  intro h
  sorry

end find_triangle_sides_l224_224544


namespace fixed_point_through_1_neg2_l224_224439

noncomputable def fixed_point (a : ℝ) (x : ℝ) : ℝ :=
a^(x - 1) - 3

-- The statement to prove
theorem fixed_point_through_1_neg2 (a : ℝ) (h : a > 0) (h' : a ≠ 1) :
  fixed_point a 1 = -2 :=
by
  unfold fixed_point
  sorry

end fixed_point_through_1_neg2_l224_224439


namespace largest_divisor_n_l224_224842

theorem largest_divisor_n (n : ℕ) (h₁ : n > 0) (h₂ : 650 ∣ n^3) : 130 ∣ n :=
sorry

end largest_divisor_n_l224_224842


namespace proof_problem_l224_224559

open Real

theorem proof_problem 
  (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) 
  (h_sum : a + b + c + d = 1)
  : (b * c * d / (1 - a)^2) + (c * d * a / (1 - b)^2) + (d * a * b / (1 - c)^2) + (a * b * c / (1 - d)^2) ≤ 1 / 9 := 
   sorry

end proof_problem_l224_224559


namespace power_of_fraction_to_decimal_l224_224763

theorem power_of_fraction_to_decimal : ∃ x : ℕ, (1 / 9 : ℚ) ^ x = 1 / 81 ∧ x = 2 :=
by
  use 2
  simp
  sorry

end power_of_fraction_to_decimal_l224_224763


namespace parabola_focus_coordinates_l224_224436

theorem parabola_focus_coordinates :
  ∀ x y : ℝ, y^2 = -8 * x → (x, y) = (-2, 0) := by
  sorry

end parabola_focus_coordinates_l224_224436


namespace integer_with_exactly_12_integers_to_its_left_l224_224251

theorem integer_with_exactly_12_integers_to_its_left :
  let initial_list := List.range' 1 20
  let first_half := initial_list.take 10
  let second_half := initial_list.drop 10
  let new_list := second_half ++ first_half
  new_list.get! 12 = 3 :=
by
  let initial_list := List.range' 1 20
  let first_half := initial_list.take 10
  let second_half := initial_list.drop 10
  let new_list := second_half ++ first_half
  sorry

end integer_with_exactly_12_integers_to_its_left_l224_224251


namespace gerald_pfennigs_left_l224_224208

theorem gerald_pfennigs_left (cost_of_pie : ℕ) (farthings_initial : ℕ) (farthings_per_pfennig : ℕ) :
  cost_of_pie = 2 → farthings_initial = 54 → farthings_per_pfennig = 6 → 
  (farthings_initial / farthings_per_pfennig) - cost_of_pie = 7 :=
by
  intros h1 h2 h3
  sorry

end gerald_pfennigs_left_l224_224208


namespace path_traveled_is_correct_l224_224022

-- Define the original triangle and the circle.
def side_a : ℝ := 8
def side_b : ℝ := 10
def side_c : ℝ := 12.5
def radius : ℝ := 1.5

-- Define the condition that the circle is rolling inside the triangle.
def new_side (original_side : ℝ) (r : ℝ) : ℝ := original_side - 2 * r

-- Calculate the new sides of the smaller triangle path.
def new_side_a := new_side side_a radius
def new_side_b := new_side side_b radius
def new_side_c := new_side side_c radius

-- Calculate the perimeter of the path traced by the circle's center.
def path_perimeter := new_side_a + new_side_b + new_side_c

-- Prove that this perimeter equals 21.5 units under given conditions.
theorem path_traveled_is_correct : path_perimeter = 21.5 := by
  simp [new_side, new_side_a, new_side_b, new_side_c, path_perimeter]
  sorry

end path_traveled_is_correct_l224_224022


namespace octopus_shoes_needed_l224_224474

-- Defining the basic context: number of legs and current shod legs
def num_legs : ℕ := 8

-- Conditions based on the number of already shod legs for each member
def father_shod_legs : ℕ := num_legs / 2       -- Father-octopus has half of his legs shod
def mother_shod_legs : ℕ := 3                  -- Mother-octopus has 3 legs shod
def son_shod_legs : ℕ := 6                     -- Each son-octopus has 6 legs shod
def num_sons : ℕ := 2                          -- There are 2 sons

-- Calculate unshod legs for each 
def father_unshod_legs : ℕ := num_legs - father_shod_legs
def mother_unshod_legs : ℕ := num_legs - mother_shod_legs
def son_unshod_legs : ℕ := num_legs - son_shod_legs

-- Aggregate the total shoes needed based on unshod legs
def total_shoes_needed : ℕ :=
  father_unshod_legs + 
  mother_unshod_legs + 
  (son_unshod_legs * num_sons)

-- The theorem to prove
theorem octopus_shoes_needed : total_shoes_needed = 13 := 
  by 
    sorry

end octopus_shoes_needed_l224_224474


namespace truth_of_q_l224_224534

variable {p q : Prop}

theorem truth_of_q (hnp : ¬ p) (hpq : p ∨ q) : q :=
  by
  sorry

end truth_of_q_l224_224534


namespace raghu_investment_is_2200_l224_224151

noncomputable def RaghuInvestment : ℝ := 
  let R := 2200
  let T := 0.9 * R
  let V := 1.1 * T
  if R + T + V = 6358 then R else 0

theorem raghu_investment_is_2200 :
  RaghuInvestment = 2200 := by
  sorry

end raghu_investment_is_2200_l224_224151


namespace largest_divisor_same_remainder_l224_224324

theorem largest_divisor_same_remainder (n : ℕ) (h : 17 % n = 30 % n) : n = 13 :=
sorry

end largest_divisor_same_remainder_l224_224324


namespace balance_blue_balls_l224_224278

variable (G Y W B : ℝ)

-- Define the conditions
def condition1 : 4 * G = 8 * B := sorry
def condition2 : 3 * Y = 8 * B := sorry
def condition3 : 4 * B = 3 * W := sorry

-- Prove the required balance of 3G + 4Y + 3W
theorem balance_blue_balls (h1 : 4 * G = 8 * B) (h2 : 3 * Y = 8 * B) (h3 : 4 * B = 3 * W) :
  3 * (2 * B) + 4 * (8 / 3 * B) + 3 * (4 / 3 * B) = 62 / 3 * B := by
  sorry

end balance_blue_balls_l224_224278


namespace problem1_problem2_l224_224963

noncomputable def setA (a : ℝ) : Set ℝ := {x | |x - a| ≤ 1}
def setB : Set ℝ := {x | x^2 - 5 * x + 4 ≤ 0}

theorem problem1 (a : ℝ) (h : a = 1) : setA a ∪ setB = {x | 0 ≤ x ∧ x ≤ 4} := by
  sorry

theorem problem2 (a : ℝ) : (∀ x, x ∈ setA a → x ∈ setB) ↔ (2 ≤ a ∧ a ≤ 3) := by
  sorry

end problem1_problem2_l224_224963


namespace distance_from_center_to_point_l224_224508

theorem distance_from_center_to_point :
  let circle_center := (5, -7)
  let point := (3, -4)
  let distance := Real.sqrt ((3 - 5)^2 + (-4 + 7)^2)
  distance = Real.sqrt 13 := sorry

end distance_from_center_to_point_l224_224508


namespace cos_60_degrees_is_one_half_l224_224636

-- Define the given conditions in Lean 4
def angle_60_degrees : Real :=
  60 * Real.pi / 180

def unit_circle_point (θ : Real) : Real × Real :=
  (Real.cos θ, Real.sin θ)

-- State the proof problem in Lean 4
theorem cos_60_degrees_is_one_half : 
  (unit_circle_point angle_60_degrees).1 = 1 / 2 := 
sorry

end cos_60_degrees_is_one_half_l224_224636


namespace find_fraction_l224_224949

theorem find_fraction (a b : ℝ) (h : a = 2 * b) : (a / (a - b)) = 2 :=
by
  sorry

end find_fraction_l224_224949


namespace triangle_is_isosceles_l224_224329

variable (a b m_a m_b : ℝ)

-- Conditions: 
-- A circle touches two sides of a triangle (denoted as a and b).
-- The circle also touches the medians m_a and m_b drawn to these sides.
-- Given equations:
axiom Eq1 : (1/2) * a + (1/3) * m_b = (1/2) * b + (1/3) * m_a
axiom Eq3 : (1/2) * a + m_b = (1/2) * b + m_a

-- Question: Prove that the triangle is isosceles, i.e., a = b
theorem triangle_is_isosceles : a = b :=
by
  sorry

end triangle_is_isosceles_l224_224329


namespace solve_absolute_value_equation_l224_224735

theorem solve_absolute_value_equation (y : ℝ) : (|y - 4| + 3 * y = 15) ↔ (y = 19 / 4) := by
  sorry

end solve_absolute_value_equation_l224_224735


namespace time_to_cover_escalator_l224_224047

-- Definitions of the rates and length
def escalator_speed : ℝ := 12
def person_speed : ℝ := 2
def escalator_length : ℝ := 210

-- Theorem statement that we need to prove
theorem time_to_cover_escalator :
  (escalator_length / (escalator_speed + person_speed) = 15) :=
by
  sorry

end time_to_cover_escalator_l224_224047


namespace number_of_ways_to_choose_a_pair_of_socks_l224_224239

-- Define the number of socks of each color
def white_socks := 5
def brown_socks := 5
def blue_socks := 5
def green_socks := 5

-- Define the total number of socks
def total_socks := white_socks + brown_socks + blue_socks + green_socks

-- Define the number of ways to choose 2 blue socks from 5 blue socks
def num_ways_choose_two_blue_socks : ℕ := Nat.choose blue_socks 2

-- The proof statement
theorem number_of_ways_to_choose_a_pair_of_socks :
  num_ways_choose_two_blue_socks = 10 :=
sorry

end number_of_ways_to_choose_a_pair_of_socks_l224_224239


namespace angle_between_diagonals_of_adjacent_faces_of_cube_is_90_degrees_l224_224771

-- Define the concept of a cube and the diagonals of its faces.
structure Cube :=
  (faces : Fin 6 → (Fin 4 → ℝ × ℝ × ℝ))    -- Representing each face as a set of four vertices in 3D space

def is_square_face (face : Fin 4 → ℝ × ℝ × ℝ) : Prop :=
  -- A function that checks if a given set of four vertices forms a square face.
  sorry

def are_adjacent_faces_perpendicular_diagonals 
  (face1 face2 : Fin 4 → ℝ × ℝ × ℝ) (c : Cube) : Prop :=
  -- A function that checks if the diagonals of two given adjacent square faces of a cube are perpendicular.
  sorry

-- The theorem stating the required proof:
theorem angle_between_diagonals_of_adjacent_faces_of_cube_is_90_degrees
  (c : Cube)
  (h1 : is_square_face (c.faces 0))
  (h2 : is_square_face (c.faces 1))
  (h_adj: are_adjacent_faces_perpendicular_diagonals (c.faces 0) (c.faces 1) c) :
  ∃ q : ℝ, q = 90 :=
by
  sorry

end angle_between_diagonals_of_adjacent_faces_of_cube_is_90_degrees_l224_224771


namespace bookstore_purchase_prices_equal_l224_224776

variable (x : ℝ)

theorem bookstore_purchase_prices_equal
  (h1 : 500 > 0)
  (h2 : 700 > 0)
  (h3 : x > 0)
  (h4 : x + 4 > 0)
  (h5 : ∃ p₁ p₂ : ℝ, p₁ = 500 / x ∧ p₂ = 700 / (x + 4) ∧ p₁ = p₂) :
  500 / x = 700 / (x + 4) :=
by
  sorry

end bookstore_purchase_prices_equal_l224_224776


namespace find_k_inverse_proportion_l224_224095

theorem find_k_inverse_proportion :
  ∃ k : ℝ, (∀ x y : ℝ, y = (k + 1) / x → (x = 1 ∧ y = -2) → k = -3) :=
by
  sorry

end find_k_inverse_proportion_l224_224095


namespace problem_statement_l224_224352

variable {x y : ℝ}

def star (a b : ℝ) : ℝ := (a + b)^2

theorem problem_statement (x y : ℝ) : star ((x + y)^2) ((y + x)^2) = 4 * (x + y)^4 := by
  sorry

end problem_statement_l224_224352


namespace Trent_traveled_distance_l224_224149

variable (blocks_length : ℕ := 50)
variables (walking_blocks : ℕ := 4) (bus_blocks : ℕ := 7) (bicycle_blocks : ℕ := 5)
variables (walking_round_trip : ℕ := 2 * walking_blocks * blocks_length)
variables (bus_round_trip : ℕ := 2 * bus_blocks * blocks_length)
variables (bicycle_round_trip : ℕ := 2 * bicycle_blocks * blocks_length)

def total_distance_traveleed : ℕ :=
  walking_round_trip + bus_round_trip + bicycle_round_trip

theorem Trent_traveled_distance :
  total_distance_traveleed = 1600 := by
    sorry

end Trent_traveled_distance_l224_224149


namespace max_area_rectangle_shorter_side_l224_224021

theorem max_area_rectangle_shorter_side (side_length : ℕ) (n : ℕ)
  (hsq : side_length = 40) (hn : n = 5) :
  ∃ (shorter_side : ℕ), shorter_side = 8 := by
  sorry

end max_area_rectangle_shorter_side_l224_224021


namespace find_distance_AB_l224_224905

variable (vA vB : ℝ) -- speeds of Person A and Person B
variable (x : ℝ) -- distance between points A and B
variable (t1 t2 : ℝ) -- time variables

-- Conditions
def startTime := 0
def meetDistanceBC := 240
def returnPointBDistantFromA := 120
def doublingSpeedFactor := 2

-- Main questions and conditions
theorem find_distance_AB
  (h1 : vA > vB)
  (h2 : t1 = x / vB)
  (h3 : t2 = 2 * (x - meetDistanceBC) / vA) 
  (h4 : x = meetDistanceBC + returnPointBDistantFromA + (t1 * (doublingSpeedFactor * vB) - t2 * vA) / (doublingSpeedFactor - 1)) :
  x = 420 :=
sorry

end find_distance_AB_l224_224905


namespace tangent_line_at_P_l224_224223

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x
noncomputable def f_prime (x : ℝ) : ℝ := 3 * x^2 - 3
def P : ℝ × ℝ := (2, -6)

theorem tangent_line_at_P :
  ∃ (m b : ℝ), (∀ (x : ℝ), f_prime x = m) ∧ (∀ (x : ℝ), f x - f 2 = m * (x - 2) + b) ∧ (2 : ℝ) = 2 → b = 0 ∧ m = -3 :=
by
  sorry

end tangent_line_at_P_l224_224223


namespace jill_peaches_l224_224399

open Nat

theorem jill_peaches (Jake Steven Jill : ℕ)
  (h1 : Jake = Steven - 6)
  (h2 : Steven = Jill + 18)
  (h3 : Jake = 17) :
  Jill = 5 := 
by
  sorry

end jill_peaches_l224_224399


namespace train_length_l224_224162

theorem train_length (speed : ℝ) (time_seconds : ℝ) (time_hours : ℝ) (distance_km : ℝ) (distance_m : ℝ) 
  (h1 : speed = 60) 
  (h2 : time_seconds = 42) 
  (h3 : time_hours = time_seconds / 3600)
  (h4 : distance_km = speed * time_hours) 
  (h5 : distance_m = distance_km * 1000) :
  distance_m = 700 :=
by 
  sorry

end train_length_l224_224162


namespace total_dress_designs_l224_224781

theorem total_dress_designs:
  let colors := 5
  let patterns := 4
  let sleeve_lengths := 2
  colors * patterns * sleeve_lengths = 40 := 
by
  sorry

end total_dress_designs_l224_224781


namespace QR_value_l224_224291

-- Given conditions for the problem
def QP : ℝ := 15
def sinQ : ℝ := 0.4

-- Define QR based on the given conditions
noncomputable def QR : ℝ := QP / sinQ

-- The theorem to prove that QR = 37.5
theorem QR_value : QR = 37.5 := 
by
  unfold QR QP sinQ
  sorry

end QR_value_l224_224291


namespace cos_sin_combination_l224_224366

theorem cos_sin_combination (x : ℝ) (h : 2 * Real.cos x + 3 * Real.sin x = 4) : 
  3 * Real.cos x - 2 * Real.sin x = 0 := 
by 
  sorry

end cos_sin_combination_l224_224366


namespace UncleVanya_travel_time_l224_224773

-- Define the conditions
variables (x y z : ℝ)
variables (h1 : 2 * x + 3 * y + 20 * z = 66)
variables (h2 : 5 * x + 8 * y + 30 * z = 144)

-- Question: how long will it take to walk 4 km, cycle 5 km, and drive 80 km
theorem UncleVanya_travel_time : 4 * x + 5 * y + 80 * z = 174 :=
sorry

end UncleVanya_travel_time_l224_224773


namespace range_of_k_l224_224371

variable (k : ℝ)
def f (x : ℝ) : ℝ := k * x + 1
def g (x : ℝ) : ℝ := x^2 - 1

theorem range_of_k (h : ∀ x : ℝ, f k x > 0 ∨ g x > 0) : k ∈ Set.Ioo (-1 : ℝ) (1 : ℝ) := 
sorry

end range_of_k_l224_224371


namespace probability_of_region_l224_224178

-- Definition of the bounds
def bounds (x y : ℝ) : Prop := 0 ≤ x ∧ x ≤ 4 ∧ 0 ≤ y ∧ y ≤ 8

-- Definition of the region where x + y <= 5
def region (x y : ℝ) : Prop := x + y ≤ 5

-- The proof statement
theorem probability_of_region : 
  (∃ (x y : ℝ), bounds x y ∧ region x y) →
  ∃ (p : ℚ), p = 3/8 :=
by sorry

end probability_of_region_l224_224178


namespace three_pumps_drain_time_l224_224279

-- Definitions of the rates of each pump
def rate1 := 1 / 9
def rate2 := 1 / 6
def rate3 := 1 / 12

-- Combined rate of all three pumps working together
def combined_rate := rate1 + rate2 + rate3

-- Time to drain the lake with all three pumps working together
def time_to_drain := 1 / combined_rate

-- Theorem: The time it takes for three pumps working together to drain the lake is 36/13 hours
theorem three_pumps_drain_time : time_to_drain = 36 / 13 := by
  sorry

end three_pumps_drain_time_l224_224279


namespace solve_arithmetic_sequence_l224_224734

theorem solve_arithmetic_sequence :
  ∀ (x : ℝ), x > 0 ∧ x^2 = (2^2 + 5^2) / 2 → x = Real.sqrt (29 / 2) :=
by
  intro x
  intro hx
  sorry

end solve_arithmetic_sequence_l224_224734


namespace gabrielle_total_crates_l224_224207

theorem gabrielle_total_crates (monday tuesday wednesday thursday : ℕ)
  (h_monday : monday = 5)
  (h_tuesday : tuesday = 2 * monday)
  (h_wednesday : wednesday = tuesday - 2)
  (h_thursday : thursday = tuesday / 2) :
  monday + tuesday + wednesday + thursday = 28 :=
by
  sorry

end gabrielle_total_crates_l224_224207


namespace equation_of_line_containing_BC_l224_224519

theorem equation_of_line_containing_BC (A B C : ℝ × ℝ)
  (hA : A = (1, 2))
  (altitude_from_CA : ∀ x y : ℝ, 2 * x - 3 * y + 1 = 0)
  (altitude_from_BA : ∀ x y : ℝ, x + y = 1)
  (eq_BC : ∃ k b : ℝ, ∀ x y : ℝ, y = k * x + b) :
  ∃ k b : ℝ, (∀ x y : ℝ, y = k * x + b) ∧ 2 * x + 3 * y + 7 = 0 :=
sorry

end equation_of_line_containing_BC_l224_224519


namespace verify_probabilities_l224_224031

/-- A bag contains 2 red balls, 3 black balls, and 4 white balls, all of the same size.
    A ball is drawn from the bag at a time, and once drawn, it is not replaced. -/
def total_balls := 9
def red_balls := 2
def black_balls := 3
def white_balls := 4

/-- Calculate the probability that the first ball is black and the second ball is white. -/
def prob_first_black_second_white :=
  (black_balls / total_balls) * (white_balls / (total_balls - 1))

/-- Calculate the probability that the number of draws does not exceed 3, 
    given that drawing a red ball means stopping. -/
def prob_draws_not_exceed_3 :=
  (red_balls / total_balls) +
  ((total_balls - red_balls) / total_balls) * (red_balls / (total_balls - 1)) +
  ((total_balls - red_balls - 1) / total_balls) *
  ((total_balls - red_balls) / (total_balls - 1)) *
  (red_balls / (total_balls - 2))

/-- Theorem that verifies the probabilities based on the given conditions. -/
theorem verify_probabilities :
  prob_first_black_second_white = 1 / 6 ∧
  prob_draws_not_exceed_3 = 7 / 12 :=
by
  sorry

end verify_probabilities_l224_224031


namespace coefficient_x2_is_negative_40_l224_224077

noncomputable def x2_coefficient_in_expansion (a : ℕ) : ℤ :=
  (-1)^3 * a^2 * Nat.choose 5 2

theorem coefficient_x2_is_negative_40 :
  x2_coefficient_in_expansion 2 = -40 :=
by
  sorry

end coefficient_x2_is_negative_40_l224_224077


namespace quiz_common_difference_l224_224985

theorem quiz_common_difference 
  (x d : ℕ) 
  (h1 : x + 2 * d = 39) 
  (h2 : 8 * x + 28 * d = 360) 
  : d = 4 := 
  sorry

end quiz_common_difference_l224_224985


namespace correct_option_l224_224158

theorem correct_option : 
  (-(2:ℤ))^3 ≠ -6 ∧ 
  (-(1:ℤ))^10 ≠ -10 ∧ 
  (-(1:ℚ)/3)^3 ≠ -1/9 ∧ 
  -(2:ℤ)^2 = -4 :=
by 
  sorry

end correct_option_l224_224158


namespace worker_late_time_l224_224455

noncomputable def usual_time : ℕ := 60
noncomputable def speed_factor : ℚ := 4 / 5

theorem worker_late_time (T T_new : ℕ) (S : ℚ) :
  T = usual_time →
  T = 60 →
  T_new = (5 / 4) * T →
  T_new - T = 15 :=
by
  intros
  subst T
  sorry

end worker_late_time_l224_224455


namespace categorize_numbers_l224_224506

def numbers : List ℚ := [-16/10, -5/6, 89/10, -7, 1/12, 0, 25]

def is_positive (x : ℚ) : Prop := x > 0
def is_negative_fraction (x : ℚ) : Prop := x < 0 ∧ x.den ≠ 1
def is_negative_integer (x : ℚ) : Prop := x < 0 ∧ x.den = 1

theorem categorize_numbers :
  { x | x ∈ numbers ∧ is_positive x } = { 89 / 10, 1 / 12, 25 } ∧
  { x | x ∈ numbers ∧ is_negative_fraction x } = { -5 / 6 } ∧
  { x | x ∈ numbers ∧ is_negative_integer x } = { -7 } := by
  sorry

end categorize_numbers_l224_224506


namespace find_sum_l224_224161

theorem find_sum (P R : ℝ) (T : ℝ) (hT : T = 3) (h1 : P * (R + 1) * 3 = P * R * 3 + 2500) : 
  P = 2500 := by
  sorry

end find_sum_l224_224161


namespace max_c_magnitude_l224_224217

variables {a b c : ℝ × ℝ}

-- Definitions of the given conditions
def unit_vector (v : ℝ × ℝ) : Prop := ‖v‖ = 1
def orthogonal (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0
def satisfied_c (c a b : ℝ × ℝ) : Prop := ‖c - (a + b)‖ = 2

-- Main theorem to prove
theorem max_c_magnitude (ha : unit_vector a) (hb : unit_vector b) (hab : orthogonal a b) (hc : satisfied_c c a b) : ‖c‖ ≤ 2 + Real.sqrt 2 := 
sorry

end max_c_magnitude_l224_224217


namespace mod_equiv_l224_224942

theorem mod_equiv (a b c d e : ℤ) (n : ℤ) (h1 : a = 101)
                                    (h2 : b = 15)
                                    (h3 : c = 7)
                                    (h4 : d = 9)
                                    (h5 : e = 5)
                                    (h6 : n = 17) :
  (a * b - c * d + e) % n = 7 := by
  sorry

end mod_equiv_l224_224942


namespace sum_le_six_l224_224522

theorem sum_le_six (a b : ℤ) (h1 : a ≠ -1) (h2 : b ≠ -1) 
    (h3 : ∃ (r s : ℤ), r * s = a + b ∧ r + s = ab) : a + b ≤ 6 :=
sorry

end sum_le_six_l224_224522


namespace cristina_nicky_head_start_l224_224275

theorem cristina_nicky_head_start (s_c s_n : ℕ) (t d : ℕ) 
  (h1 : s_c = 5) 
  (h2 : s_n = 3) 
  (h3 : t = 30)
  (h4 : d = s_n * t):
  d = 90 := 
by
  sorry

end cristina_nicky_head_start_l224_224275


namespace sequence_properties_l224_224962

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  a 3 = 3 ∧ ∀ n, a (n + 1) = a n + 2

theorem sequence_properties {a : ℕ → ℤ} (h : arithmetic_sequence a) :
  a 2 + a 4 = 6 ∧ ∀ n, a n = 2 * n - 3 :=
by
  sorry

end sequence_properties_l224_224962


namespace part_I_part_II_l224_224961

-- Definition of functions
def f (x a : ℝ) := |3 * x - a|
def g (x : ℝ) := |x + 1|

-- Part (I): Solution set for f(x) < 3 when a = 4
theorem part_I (x : ℝ) : f x 4 < 3 ↔ (1 / 3 < x ∧ x < 7 / 3) :=
by 
  sorry

-- Part (II): Range of a such that f(x) + g(x) > 1 for all x in ℝ
theorem part_II (a : ℝ) : (∀ x : ℝ, f x a + g x > 1) ↔ (a < -6 ∨ a > 0) :=
by 
  sorry

end part_I_part_II_l224_224961


namespace total_matches_played_l224_224039

def home_team_wins := 3
def home_team_draws := 4
def home_team_losses := 0
def rival_team_wins := 2 * home_team_wins
def rival_team_draws := home_team_draws
def rival_team_losses := 0

theorem total_matches_played :
  home_team_wins + home_team_draws + home_team_losses + rival_team_wins + rival_team_draws + rival_team_losses = 17 :=
by
  sorry

end total_matches_played_l224_224039


namespace olivias_dad_total_spending_l224_224568

def people : ℕ := 5
def meal_cost : ℕ := 12
def drink_cost : ℕ := 3
def dessert_cost : ℕ := 5

theorem olivias_dad_total_spending : 
  (people * meal_cost) + (people * drink_cost) + (people * dessert_cost) = 100 := 
by
  sorry

end olivias_dad_total_spending_l224_224568


namespace simplify_expression_l224_224287

theorem simplify_expression :
  (3 + 4 + 5 + 7) / 3 + (3 * 6 + 9) / 4 = 157 / 12 :=
by
  sorry

end simplify_expression_l224_224287


namespace find_chosen_number_l224_224931

-- Define the conditions
def condition (x : ℝ) : Prop := (3 / 2) * x + 53.4 = -78.9

-- State the theorem
theorem find_chosen_number : ∃ x : ℝ, condition x ∧ x = -88.2 :=
sorry

end find_chosen_number_l224_224931


namespace arithmetic_sequence_value_l224_224246

theorem arithmetic_sequence_value (a : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, a n = a 0 + n * (a 1 - a 0)) -- definition of arithmetic sequence
  (h2 : a 2 + a 10 = -12) -- given that a_2 + a_{10} = -12
  (h3 : a_2 = -6) -- given that a_6 is the average of a_2 and a_{10}
  : a 6 = -6 :=
sorry

end arithmetic_sequence_value_l224_224246


namespace find_complex_z_modulus_of_z_l224_224821

open Complex

theorem find_complex_z (z : ℂ) (h1 : (z - (0 + 3 * I)).im = 0) (h2 : ((z - (0 + 5 * I)) / (2 - I)).re = 0) : 
    z = -1 + 3 * I := by 
  sorry

theorem modulus_of_z (z : ℂ) (h1 : (z - (0 + 3 * I)).im = 0) (h2 : ((z - (0 + 5 * I)) / (2 - I)).re = 0) : 
    Complex.abs (z / (1 - I)) = Real.sqrt 5 := by 
  sorry

end find_complex_z_modulus_of_z_l224_224821


namespace minimum_distance_l224_224715

noncomputable def distance (M Q : ℝ × ℝ) : ℝ :=
  ( (M.1 - Q.1) ^ 2 + (M.2 - Q.2) ^ 2 ) ^ (1 / 2)

theorem minimum_distance (M : ℝ × ℝ) :
  ∃ Q : ℝ × ℝ, ( (Q.1 - 1) ^ 2 + Q.2 ^ 2 = 1 ) ∧ distance M Q = 1 :=
sorry

end minimum_distance_l224_224715


namespace polygon_sides_eq_n_l224_224843

theorem polygon_sides_eq_n
  (sum_except_two_angles : ℝ)
  (angle_equal : ℝ)
  (h1 : sum_except_two_angles = 2970)
  (h2 : angle_equal * 2 < 180)
  : ∃ n : ℕ, 180 * (n - 2) = 2970 + 2 * angle_equal ∧ n = 19 :=
by
  sorry

end polygon_sides_eq_n_l224_224843


namespace tan_22_5_equiv_l224_224893

theorem tan_22_5_equiv : 
  ∃ a b c d : ℕ, a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ a + b + c + d = 4 ∧ 
  (tan (real.pi / 8) = real.sqrt a - real.sqrt b + real.sqrt c - d) :=
by
  have h1 : real.sin (real.pi / 4) = real.sqrt 2 / 2 := sorry,
  have h2 : real.cos (real.pi / 4) = real.sqrt 2 / 2 := sorry,
  have tan_half_angle : tan (real.pi / 8) = (1 - real.sqrt 2 / 2) / (real.sqrt 2 / 2), from sorry,
  have tan_val : tan (real.pi / 8) = real.sqrt 2 - 1, from sorry,
  existsi [2, 1, 0, 1],
  split,
  { -- Verify inequalities
    repeat { split }; linarith },
  split,
  { -- Sum of variables
    norm_num },
  -- Check the expression equivalence
  exact tan_val

end tan_22_5_equiv_l224_224893


namespace rectangle_area_proof_l224_224165

def rectangle_width : ℕ := 5

def rectangle_length (width : ℕ) : ℕ := 3 * width

def rectangle_area (length width : ℕ) : ℕ := length * width

theorem rectangle_area_proof : rectangle_area (rectangle_length rectangle_width) rectangle_width = 75 := by
  sorry -- Proof can be added later

end rectangle_area_proof_l224_224165


namespace time_to_fill_box_correct_l224_224729

def total_toys := 50
def mom_rate := 5
def mia_rate := 3

def time_to_fill_box (total_toys mom_rate mia_rate : ℕ) : ℚ :=
  let net_rate_per_cycle := mom_rate - mia_rate
  let cycles := ((total_toys - 1) / net_rate_per_cycle) + 1
  let total_seconds := cycles * 30
  total_seconds / 60

theorem time_to_fill_box_correct : time_to_fill_box total_toys mom_rate mia_rate = 12.5 :=
by
  sorry

end time_to_fill_box_correct_l224_224729


namespace find_cost_price_l224_224795

-- Define the known data
def cost_price_80kg (C : ℝ) := 80 * C
def cost_price_20kg := 20 * 20
def selling_price_mixed := 2000
def total_cost_price_mixed (C : ℝ) := cost_price_80kg C + cost_price_20kg

-- Using the condition for 25% profit
def selling_price_of_mixed (C : ℝ) := 1.25 * total_cost_price_mixed C

-- The main theorem
theorem find_cost_price (C : ℝ) : selling_price_of_mixed C = selling_price_mixed → C = 15 :=
by
  sorry

end find_cost_price_l224_224795


namespace max_value_l224_224263

theorem max_value (x y z : ℝ) (h : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1) : 
  8 * x + 3 * y + 15 * z ≤ Real.sqrt 298 :=
sorry

end max_value_l224_224263


namespace mandy_used_nutmeg_l224_224115

theorem mandy_used_nutmeg (x : ℝ) (h1 : 0.67 = x + 0.17) : x = 0.50 :=
  by
  sorry

end mandy_used_nutmeg_l224_224115


namespace gabrielle_total_crates_l224_224206

theorem gabrielle_total_crates (monday tuesday wednesday thursday : ℕ)
  (h_monday : monday = 5)
  (h_tuesday : tuesday = 2 * monday)
  (h_wednesday : wednesday = tuesday - 2)
  (h_thursday : thursday = tuesday / 2) :
  monday + tuesday + wednesday + thursday = 28 :=
by
  sorry

end gabrielle_total_crates_l224_224206


namespace intersection_x_value_l224_224941

theorem intersection_x_value :
  (∃ x y, y = 3 * x - 7 ∧ y = 48 - 5 * x) → x = 55 / 8 :=
by
  sorry

end intersection_x_value_l224_224941


namespace largest_angle_of_pentagon_l224_224986

theorem largest_angle_of_pentagon (x : ℕ) (hx : 5 * x + 100 = 540) : x + 40 = 128 := by
  sorry

end largest_angle_of_pentagon_l224_224986


namespace perpendicular_lines_a_value_l224_224690

theorem perpendicular_lines_a_value (a : ℝ) :
  (∀ x y : ℝ, (x + a * y - a = 0) ∧ (a * x - (2 * a - 3) * y - 1 = 0) → x ≠ y) →
  a = 0 ∨ a = 2 :=
sorry

end perpendicular_lines_a_value_l224_224690


namespace spring_work_done_l224_224092

theorem spring_work_done (F : ℝ) (l : ℝ) (stretched_length : ℝ) (k : ℝ) (W : ℝ) 
  (hF : F = 10) (hl : l = 0.1) (hk : k = F / l) (h_stretched_length : stretched_length = 0.06) : 
  W = 0.18 :=
by
  sorry

end spring_work_done_l224_224092


namespace tristan_study_hours_l224_224545

theorem tristan_study_hours :
  let monday_hours := 4
  let tuesday_hours := 2 * monday_hours
  let wednesday_hours := 3
  let thursday_hours := 3
  let friday_hours := 3
  let total_hours := monday_hours + tuesday_hours + wednesday_hours + thursday_hours + friday_hours
  let remaining_hours := 25 - total_hours
  let saturday_hours := remaining_hours / 2
  saturday_hours = 2 := by
{
  let monday_hours := 4
  let tuesday_hours := 2 * monday_hours
  let wednesday_hours := 3
  let thursday_hours := 3
  let friday_hours := 3
  let total_hours := monday_hours + tuesday_hours + wednesday_hours + thursday_hours + friday_hours
  let remaining_hours := 25 - total_hours
  let saturday_hours := remaining_hours / 2
  sorry
}

end tristan_study_hours_l224_224545


namespace probability_male_monday_female_tuesday_l224_224673

structure Volunteers where
  men : ℕ
  women : ℕ
  total : ℕ

def group : Volunteers := {men := 2, women := 2, total := 4}

def permutations (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)
def combinations (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_male_monday_female_tuesday :
  let n := permutations group.total 2
  let m := combinations group.men 1 * combinations group.women 1
  (m / n : ℚ) = 1 / 3 :=
by
  sorry

end probability_male_monday_female_tuesday_l224_224673


namespace jenna_stamp_division_l224_224250

theorem jenna_stamp_division (a b c : ℕ) (h₁ : a = 945) (h₂ : b = 1260) (h₃ : c = 630) :
  Nat.gcd (Nat.gcd a b) c = 105 :=
by
  rw [h₁, h₂, h₃]
  -- Now we need to prove Nat.gcd (Nat.gcd 945 1260) 630 = 105
  sorry

end jenna_stamp_division_l224_224250


namespace final_value_after_three_years_l224_224326

theorem final_value_after_three_years (X : ℝ) :
  (X - 0.40 * X) * (1 - 0.10) * (1 - 0.20) = 0.432 * X := by
  sorry

end final_value_after_three_years_l224_224326


namespace number_of_correct_answers_l224_224711

theorem number_of_correct_answers (C W : ℕ) (h1 : C + W = 100) (h2 : 5 * C - 2 * W = 210) : C = 58 :=
sorry

end number_of_correct_answers_l224_224711


namespace cost_price_of_cricket_bat_for_A_l224_224790

-- Define the cost price of the cricket bat for A as a variable
variable (CP_A : ℝ)

-- Define the conditions given in the problem
def condition1 := CP_A * 1.20 -- B buys at 20% profit
def condition2 := CP_A * 1.20 * 1.25 -- B sells at 25% profit
def totalCost := 231 -- C pays $231

-- The theorem we need to prove
theorem cost_price_of_cricket_bat_for_A : (condition2 = totalCost) → CP_A = 154 := by
  intros h
  sorry

end cost_price_of_cricket_bat_for_A_l224_224790


namespace solve_for_x_l224_224917

theorem solve_for_x (x : ℝ) (h : 5 + 7 / x = 6 - 5 / x) : x = 12 :=
by
  -- reduce the problem to its final steps
  sorry

end solve_for_x_l224_224917


namespace scallops_cost_calculation_l224_224470

def scallops_per_pound : ℕ := 8
def cost_per_pound : ℝ := 24.00
def scallops_per_person : ℕ := 2
def number_of_people : ℕ := 8

def total_cost : ℝ := 
  let total_scallops := number_of_people * scallops_per_person
  let total_pounds := total_scallops / scallops_per_pound
  total_pounds * cost_per_pound

theorem scallops_cost_calculation :
  total_cost = 48.00 :=
by sorry

end scallops_cost_calculation_l224_224470


namespace geometric_sequence_a_l224_224145

theorem geometric_sequence_a (a : ℝ) (h1 : a > 0) (h2 : ∃ r : ℝ, 280 * r = a ∧ a * r = 180 / 49) :
  a = 32.07 :=
by sorry

end geometric_sequence_a_l224_224145


namespace regular_polygon_sides_l224_224649

theorem regular_polygon_sides (h : ∀ n, (180 * (n - 2) = 150 * n)) : n = 12 := 
by
  sorry

end regular_polygon_sides_l224_224649


namespace tomato_plants_count_l224_224087

theorem tomato_plants_count :
  ∀ (sunflowers corn tomatoes total_rows plants_per_row : ℕ),
  sunflowers = 45 →
  corn = 81 →
  plants_per_row = 9 →
  total_rows = (sunflowers / plants_per_row) + (corn / plants_per_row) →
  tomatoes = total_rows * plants_per_row →
  tomatoes = 126 :=
by
  intros sunflowers corn tomatoes total_rows plants_per_row Hs Hc Hp Ht Hm
  rw [Hs, Hc, Hp] at *
  -- Additional calculation steps could go here to prove the theorem if needed
  sorry

end tomato_plants_count_l224_224087


namespace general_term_a_n_sum_b_n_terms_l224_224859

-- Given definitions based on the conditions
def a (n : ℕ) : ℕ := 2^(n-1)

def b (n : ℕ) : ℕ := (2^(2*n-1))^2

def b_sum (n : ℕ) : (ℕ → ℕ) := 
  (fun b : ℕ => match b with 
                | 1 => 4 
                | 2 => 64 
                | _ => (4^(2*(b - 2 + 1) - 1)))

def T (n : ℕ) : ℕ := (4 / 15) * (16^n - 1)

-- First part: Proving the general term of {a_n} is 2^(n-1)
theorem general_term_a_n (n : ℕ) : a n = 2^(n-1) := by
  sorry

-- Second part: Proving the sum of the first n terms of {b_n} is (4/15)*(16^n - 1)
theorem sum_b_n_terms (n : ℕ) : T n = (4 / 15) * (16^n - 1) := by 
  sorry

end general_term_a_n_sum_b_n_terms_l224_224859


namespace counter_represents_number_l224_224420

theorem counter_represents_number (a b : ℕ) : 10 * a + b = 10 * a + b := 
by 
  sorry

end counter_represents_number_l224_224420


namespace circle_radius_l224_224578

theorem circle_radius (r : ℝ) (π : ℝ) (h1 : π > 0) (h2 : ∀ x, π * x^2 = 100*π → x = 10) : r = 10 :=
by
  have : π * r^2 = 100*π → r = 10 := h2 r
  exact sorry

end circle_radius_l224_224578


namespace CaitlinIs24_l224_224487

-- Definition using the given conditions
def AuntAnnaAge : ℕ := 45
def BriannaAge : ℕ := (2 * AuntAnnaAge) / 3
def CaitlinAge : ℕ := BriannaAge - 6

-- Statement to be proved
theorem CaitlinIs24 : CaitlinAge = 24 :=
by
  sorry

end CaitlinIs24_l224_224487


namespace average_water_drunk_l224_224317

theorem average_water_drunk (d1 d2 d3 : ℕ) (h1 : d1 = 215) (h2 : d2 = d1 + 76) (h3 : d3 = d2 - 53) :
  (d1 + d2 + d3) / 3 = 248 :=
by
  -- placeholder for actual proof
  sorry

end average_water_drunk_l224_224317


namespace maria_mushrooms_l224_224727

theorem maria_mushrooms (potatoes carrots onions green_beans bell_peppers mushrooms : ℕ) 
  (h1 : carrots = 6 * potatoes)
  (h2 : onions = 2 * carrots)
  (h3 : green_beans = onions / 3)
  (h4 : bell_peppers = 4 * green_beans)
  (h5 : mushrooms = 3 * bell_peppers)
  (h0 : potatoes = 3) : 
  mushrooms = 144 :=
by
  sorry

end maria_mushrooms_l224_224727


namespace volleyball_height_30_l224_224611

theorem volleyball_height_30 (t : ℝ) : (60 - 9 * t - 4.5 * t^2 = 30) → t = 1.77 :=
by
  intro h_eq
  sorry

end volleyball_height_30_l224_224611


namespace line_equation_l224_224925

theorem line_equation (x y : ℝ) : 
  ((y = 1 → x = 2) ∧ ((x,y) = (1,1) ∨ (x,y) = (3,5)))
  → (2 * x - y - 3 = 0) ∨ (x = 2) :=
sorry

end line_equation_l224_224925


namespace maximum_elevation_l224_224331

-- Define the elevation function
def elevation (t : ℝ) : ℝ := 200 * t - 17 * t^2 - 3 * t^3

-- State that the maximum elevation is 368.1 feet
theorem maximum_elevation :
  ∃ t : ℝ, t > 0 ∧ (∀ t' : ℝ, t' ≠ t → elevation t ≤ elevation t') ∧ elevation t = 368.1 :=
by
  sorry

end maximum_elevation_l224_224331


namespace hospital_cost_minimization_l224_224924

theorem hospital_cost_minimization :
  ∃ (x y : ℕ), (5 * x + 6 * y = 50) ∧ (10 * x + 20 * y = 140) ∧ (2 * x + 3 * y = 23) :=
by
  sorry

end hospital_cost_minimization_l224_224924


namespace find_a_l224_224529
-- Import necessary Lean libraries

-- Define the function and its maximum value condition
def f (a x : ℝ) := -x^2 + 2*a*x + 1 - a

def has_max_value (f : ℝ → ℝ) (M : ℝ) (interval : Set ℝ) : Prop :=
  ∃ x ∈ interval, ∀ y ∈ interval, f y ≤ f x ∧ f x = M

theorem find_a (a : ℝ) :
  has_max_value (f a) 2 (Set.Icc 0 1) → (a = -1 ∨ a = 2) :=
by
  sorry

end find_a_l224_224529


namespace unit_digit_product_is_zero_l224_224762

-- Definitions based on conditions in (a)
def a_1 := 6245
def a_2 := 7083
def a_3 := 9137
def a_4 := 4631
def a_5 := 5278
def a_6 := 3974

-- Helper function to get the unit digit of a number
def unit_digit (n : Nat) : Nat := n % 10

-- Main theorem to prove
theorem unit_digit_product_is_zero :
  unit_digit (a_1 * a_2 * a_3 * a_4 * a_5 * a_6) = 0 := by
  sorry

end unit_digit_product_is_zero_l224_224762


namespace sum_of_center_coordinates_l224_224066

theorem sum_of_center_coordinates (x y : ℝ) :
    (x^2 + y^2 - 6*x + 8*y = 18) → (x = 3) → (y = -4) → x + y = -1 := 
by
    intro h1 hx hy
    rw [hx, hy]
    norm_num

end sum_of_center_coordinates_l224_224066


namespace farmer_land_acres_l224_224475

theorem farmer_land_acres
  (initial_ratio_corn : Nat)
  (initial_ratio_sugar_cane : Nat)
  (initial_ratio_tobacco : Nat)
  (new_ratio_corn : Nat)
  (new_ratio_sugar_cane : Nat)
  (new_ratio_tobacco : Nat)
  (additional_tobacco_acres : Nat)
  (total_land_acres : Nat) :
  initial_ratio_corn = 5 →
  initial_ratio_sugar_cane = 2 →
  initial_ratio_tobacco = 2 →
  new_ratio_corn = 2 →
  new_ratio_sugar_cane = 2 →
  new_ratio_tobacco = 5 →
  additional_tobacco_acres = 450 →
  total_land_acres = 1350 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end farmer_land_acres_l224_224475


namespace people_counted_on_second_day_l224_224050

theorem people_counted_on_second_day (x : ℕ) (H1 : 2 * x + x = 1500) : x = 500 :=
by {
  sorry -- Proof goes here
}

end people_counted_on_second_day_l224_224050


namespace triangle_ABC_is_acute_l224_224994

noncomputable def arithmeticSeqTerm (a1 d : ℝ) (n : ℕ) : ℝ :=
  a1 + (n - 1) * d

noncomputable def geometricSeqTerm (a1 r : ℝ) (n : ℕ) : ℝ :=
  a1 * r^(n - 1)

def tanA_condition (a1 d : ℝ) :=
  arithmeticSeqTerm a1 d 3 = -4 ∧ arithmeticSeqTerm a1 d 7 = 4

def tanB_condition (a1 r : ℝ) :=
  geometricSeqTerm a1 r 3 = 1/3 ∧ geometricSeqTerm a1 r 6 = 9

theorem triangle_ABC_is_acute {A B : ℝ} (a1a da a1b rb : ℝ) 
  (hA : tanA_condition a1a da) 
  (hB : tanB_condition a1b rb) :
  0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ (A + B) < π :=
  sorry

end triangle_ABC_is_acute_l224_224994


namespace zero_exponent_rule_proof_l224_224350

-- Defining the condition for 818 being non-zero
def eight_hundred_eighteen_nonzero : Prop := 818 ≠ 0

-- Theorem statement
theorem zero_exponent_rule_proof (h : eight_hundred_eighteen_nonzero) : 818 ^ 0 = 1 := by
  sorry

end zero_exponent_rule_proof_l224_224350


namespace point_on_x_axis_coordinates_l224_224379

theorem point_on_x_axis_coordinates (a : ℝ) (hx : a - 3 = 0) : (a + 2, a - 3) = (5, 0) :=
by
  sorry

end point_on_x_axis_coordinates_l224_224379


namespace cost_of_first_20_kgs_l224_224049

theorem cost_of_first_20_kgs (l q : ℕ)
  (h1 : 30 * l + 3 * q = 168)
  (h2 : 30 * l + 6 * q = 186) :
  20 * l = 100 :=
by
  sorry

end cost_of_first_20_kgs_l224_224049


namespace train_departure_at_10am_l224_224024

noncomputable def train_departure_time (distance travel_rate : ℕ) (arrival_time_chicago : ℕ) (time_difference : ℤ) : ℕ :=
  let travel_time := distance / travel_rate
  let arrival_time_ny := arrival_time_chicago + 1
  arrival_time_ny - travel_time

theorem train_departure_at_10am :
  train_departure_time 480 60 17 1 = 10 :=
by
  -- implementation of the proof will go here
  -- but we skip the proof as per the instructions
  sorry

end train_departure_at_10am_l224_224024


namespace farmer_initial_days_l224_224036

theorem farmer_initial_days 
  (x : ℕ) 
  (plan_daily : ℕ) 
  (actual_daily : ℕ) 
  (extra_days : ℕ) 
  (left_area : ℕ) 
  (total_area : ℕ)
  (h1 : plan_daily = 120) 
  (h2 : actual_daily = 85) 
  (h3 : extra_days = 2) 
  (h4 : left_area = 40) 
  (h5 : total_area = 720): 
  85 * (x + extra_days) + left_area = total_area → x = 6 :=
by
  intros h
  sorry

end farmer_initial_days_l224_224036


namespace like_term_l224_224868

theorem like_term (a : ℝ) : ∃ (a : ℝ), a * x ^ 5 * y ^ 3 = a * x ^ 5 * y ^ 3 :=
by sorry

end like_term_l224_224868


namespace simplify_expression_l224_224127

theorem simplify_expression (x : ℝ) (h : x = Real.sqrt 2) : 
  (x^2 - x) / (x^2 - 2 * x + 1) = 2 + Real.sqrt 2 :=
by
  sorry

end simplify_expression_l224_224127


namespace vacation_costs_l224_224483

theorem vacation_costs :
  let a := 15
  let b := 22.5
  let c := 22.5
  a + b + c = 45 → b - a = 7.5 := by
sorry

end vacation_costs_l224_224483


namespace product_of_real_roots_l224_224309

theorem product_of_real_roots : 
  (∃ x y : ℝ, (x ^ Real.log x = Real.exp 1) ∧ (y ^ Real.log y = Real.exp 1) ∧ x ≠ y ∧ x * y = 1) :=
by
  sorry

end product_of_real_roots_l224_224309


namespace express_y_in_terms_of_x_l224_224812

variable (x y : ℝ)

theorem express_y_in_terms_of_x (h : x + y = -1) : y = -1 - x := 
by 
  sorry

end express_y_in_terms_of_x_l224_224812


namespace sum_series_eq_three_l224_224495

theorem sum_series_eq_three :
  (∑' k : ℕ, (9^k) / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1)))) = 3 :=
by 
  sorry

end sum_series_eq_three_l224_224495


namespace problem1_problem2_problem3_problem4_l224_224054

-- Problem 1
theorem problem1 : (-3 : ℝ) ^ 2 + (1 / 2) ^ (-1 : ℝ) + (Real.pi - 3) ^ 0 = 12 :=
by
  sorry

-- Problem 2
theorem problem2 (x : ℝ) : (8 * x ^ 4 + 4 * x ^ 3 - x ^ 2) / (-2 * x) ^ 2 = 2 * x ^ 2 + x - 1 / 4 :=
by
  sorry

-- Problem 3
theorem problem3 (x : ℝ) : (2 * x + 1) ^ 2 - (4 * x + 1) * (x + 1) = -x :=
by
  sorry

-- Problem 4
theorem problem4 (x y : ℝ) : (x + 2 * y - 3) * (x - 2 * y + 3) = x ^ 2 - 4 * y ^ 2 + 12 * y - 9 :=
by
  sorry

end problem1_problem2_problem3_problem4_l224_224054


namespace eq_3_solutions_l224_224264

theorem eq_3_solutions (p : ℕ) (hp : Nat.Prime p) :
  ∃! (x y : ℕ), (0 < x) ∧ (0 < y) ∧ ((1 / x) + (1 / y) = (1 / p)) ∧
  ((x = p + 1 ∧ y = p^2 + p) ∨ (x = p + p ∧ y = p + p) ∨ (x = p^2 + p ∧ y = p + 1)) :=
sorry

end eq_3_solutions_l224_224264


namespace cakes_difference_l224_224346

theorem cakes_difference (cakes_bought cakes_sold : ℕ) (h1 : cakes_bought = 139) (h2 : cakes_sold = 145) : cakes_sold - cakes_bought = 6 :=
by
  sorry

end cakes_difference_l224_224346


namespace sum_of_opposite_numbers_is_zero_l224_224090

theorem sum_of_opposite_numbers_is_zero {a b : ℝ} (h : a + b = 0) : a + b = 0 := 
h

end sum_of_opposite_numbers_is_zero_l224_224090


namespace fraction_subtraction_l224_224018

theorem fraction_subtraction : (5 / 6) - (1 / 12) = (3 / 4) := 
by 
  sorry

end fraction_subtraction_l224_224018


namespace jogging_track_circumference_l224_224055

theorem jogging_track_circumference
  (Deepak_speed : ℝ)
  (Wife_speed : ℝ)
  (meet_time_minutes : ℝ)
  (H_deepak_speed : Deepak_speed = 4.5)
  (H_wife_speed : Wife_speed = 3.75)
  (H_meet_time_minutes : meet_time_minutes = 3.84) :
  let meet_time_hours := meet_time_minutes / 60
  let distance_deepak := Deepak_speed * meet_time_hours
  let distance_wife := Wife_speed * meet_time_hours
  let total_distance := distance_deepak + distance_wife
  let circumference := 2 * total_distance
  circumference = 1.056 :=
by
  sorry

end jogging_track_circumference_l224_224055


namespace count_3_digit_numbers_divisible_by_5_l224_224969

theorem count_3_digit_numbers_divisible_by_5 : 
  let three_digit_numbers := {n : ℕ | 100 ≤ n ∧ n ≤ 999}
  let divisible_by_5 := {n : ℕ | n % 5 = 0}
  let count := {n : ℕ | n ∈ three_digit_numbers ∧ n ∈ divisible_by_5}.card
  count = 180 :=
by
  sorry

end count_3_digit_numbers_divisible_by_5_l224_224969


namespace larry_gave_52_apples_l224_224549

-- Define the initial and final count of Joyce's apples
def initial_apples : ℝ := 75.0
def final_apples : ℝ := 127.0

-- Define the number of apples Larry gave Joyce
def apples_given : ℝ := final_apples - initial_apples

-- The theorem stating that Larry gave Joyce 52 apples
theorem larry_gave_52_apples : apples_given = 52 := by
  sorry

end larry_gave_52_apples_l224_224549


namespace chess_tournament_boys_l224_224709

noncomputable def num_boys_in_tournament (n k : ℕ) : Prop :=
  (6 + k * n = (n + 2) * (n + 1) / 2) ∧ (n > 2)

theorem chess_tournament_boys :
  ∃ (n : ℕ), num_boys_in_tournament n (if n = 5 then 3 else if n = 10 then 6 else 0) ∧ (n = 5 ∨ n = 10) :=
by
  sorry

end chess_tournament_boys_l224_224709


namespace cost_proof_l224_224503

-- Given conditions
def total_cost : Int := 190
def working_days : Int := 19
def trips_per_day : Int := 2
def total_trips : Int := working_days * trips_per_day

-- Define the problem to prove
def cost_per_trip : Int := 5

theorem cost_proof : (total_cost / total_trips = cost_per_trip) := 
by 
  -- This is a placeholder to indicate that we're skipping the proof
  sorry

end cost_proof_l224_224503


namespace symmetric_point_l224_224298

theorem symmetric_point (x y : ℝ) : 
  (x - 2 * y + 1 = 0) ∧ (y / x * 1 / 2 = -1) → (x = -2/5 ∧ y = 4/5) :=
by 
  sorry

end symmetric_point_l224_224298


namespace tank_capacity_l224_224792

theorem tank_capacity (x : ℝ) (h : (5/12) * x = 150) : x = 360 :=
by
  sorry

end tank_capacity_l224_224792


namespace find_radius_of_smaller_circles_l224_224328

noncomputable def smaller_circle_radius (r : ℝ) : Prop :=
  ∃ sin72 : ℝ, sin72 = Real.sin (72 * Real.pi / 180) ∧
  r = (2 * sin72) / (1 - sin72)

theorem find_radius_of_smaller_circles (r : ℝ) :
  (smaller_circle_radius r) ↔
  r = (2 * Real.sin (72 * Real.pi / 180)) / (1 - Real.sin (72 * Real.pi / 180)) :=
by
  sorry

end find_radius_of_smaller_circles_l224_224328


namespace area_of_plot_area_in_terms_of_P_l224_224717

-- Conditions and definitions.
variables (P : ℝ) (l w : ℝ)
noncomputable def perimeter := 2 * (l + w)
axiom h_perimeter : perimeter l w = 120
axiom h_equality : l = 2 * w

-- Proofs statements
theorem area_of_plot : l + w = 60 → l = 2 * w → (4 * w)^2 = 6400 := by
  sorry

theorem area_in_terms_of_P : (4 * (P / 6))^2 = (2 * P / 3)^2 → (2 * P / 3)^2 = 4 * P^2 / 9 := by
  sorry

end area_of_plot_area_in_terms_of_P_l224_224717


namespace vanessa_total_earnings_l224_224016

theorem vanessa_total_earnings :
  let num_dresses := 7
  let num_shirts := 4
  let price_per_dress := 7
  let price_per_shirt := 5
  (num_dresses * price_per_dress + num_shirts * price_per_shirt) = 69 :=
by
  sorry

end vanessa_total_earnings_l224_224016


namespace min_x9_minus_x1_l224_224265

theorem min_x9_minus_x1
  (x : Fin 9 → ℕ)
  (h_pos : ∀ i, x i > 0)
  (h_sorted : ∀ i j, i < j → x i < x j)
  (h_sum : (Finset.univ.sum x) = 220) :
    ∃ x1 x2 x3 x4 x5 x6 x7 x8 x9 : ℕ,
    x1 < x2 ∧ x2 < x3 ∧ x3 < x4 ∧ x4 < x5 ∧ x5 < x6 ∧ x6 < x7 ∧ x7 < x8 ∧ x8 < x9 ∧
    (x1 + x2 + x3 + x4 + x5 = 110) ∧
    x1 = x 0 ∧ x2 = x 1 ∧ x3 = x 2 ∧ x4 = x 3 ∧ x5 = x 4 ∧ x6 = x 5 ∧ x7 = x 6 ∧ x8 = x 7 ∧ x9 = x 8
    ∧ (x9 - x1 = 9) :=
sorry

end min_x9_minus_x1_l224_224265


namespace divisible_by_two_of_square_l224_224978

theorem divisible_by_two_of_square {a : ℤ} (h : 2 ∣ a^2) : 2 ∣ a :=
sorry

end divisible_by_two_of_square_l224_224978


namespace common_difference_l224_224243

variable {a : ℕ → ℤ} -- Define the arithmetic sequence

theorem common_difference (h : a 2015 = a 2013 + 6) : 
  ∃ d : ℤ, ∀ n : ℕ, a n = a 1 + (n - 1) * d ∧ d = 3 := 
by
  use 3
  sorry

end common_difference_l224_224243


namespace reflected_point_correct_l224_224581

-- Defining the original point coordinates
def original_point : ℝ × ℝ := (3, -5)

-- Defining the transformation function
def reflect_across_y_axis (point : ℝ × ℝ) : ℝ × ℝ :=
  (-point.1, point.2)

-- Proving the point after reflection is as expected
theorem reflected_point_correct : reflect_across_y_axis original_point = (-3, -5) :=
by
  sorry

end reflected_point_correct_l224_224581


namespace solution_set_quadratic_inequality_l224_224589

theorem solution_set_quadratic_inequality :
  {x : ℝ | -x^2 + 5*x + 6 > 0} = {x : ℝ | -1 < x ∧ x < 6} :=
sorry

end solution_set_quadratic_inequality_l224_224589


namespace smallest_value_of_3a_plus_2_l224_224700

variable (a : ℝ)

theorem smallest_value_of_3a_plus_2 (h : 5 * a^2 + 7 * a + 2 = 1) : 3 * a + 2 = -1 :=
sorry

end smallest_value_of_3a_plus_2_l224_224700


namespace find_m_l224_224220

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x + m
noncomputable def g (x : ℝ) : ℝ := 2 * x - 2

theorem find_m : 
  ∃ m : ℝ, ∀ x : ℝ, f m x = g x → m = -2 := by
  sorry

end find_m_l224_224220


namespace solve_for_N_l224_224841

theorem solve_for_N :
    (481 + 483 + 485 + 487 + 489 + 491 = 3000 - N) → (N = 84) :=
by
    -- Proof is omitted
    sorry

end solve_for_N_l224_224841


namespace profit_per_piece_correct_sales_volume_correct_maximum_monthly_profit_optimum_selling_price_is_130_l224_224886

-- Define the selling price and cost price
def cost_price : ℝ := 60
def sales_price (x : ℝ) := x

-- 1. Prove the profit per piece
def profit_per_piece (x : ℝ) : ℝ := sales_price x - cost_price

theorem profit_per_piece_correct (x : ℝ) : profit_per_piece x = x - 60 :=
by 
  -- it follows directly from the definition of profit_per_piece
  sorry

-- 2. Define the linear function relationship between monthly sales volume and selling price
def sales_volume (x : ℝ) : ℝ := -2 * x + 400

theorem sales_volume_correct (x : ℝ) : sales_volume x = -2 * x + 400 :=
by 
  -- it follows directly from the definition of sales_volume
  sorry

-- 3. Define the monthly profit and prove the maximized profit
def monthly_profit (x : ℝ) : ℝ := profit_per_piece x * sales_volume x

theorem maximum_monthly_profit (x : ℝ) : 
  monthly_profit x = -2 * x^2 + 520 * x - 24000 :=
by 
  -- it follows directly from the definition of monthly_profit
  sorry

theorem optimum_selling_price_is_130 : ∃ (x : ℝ), (monthly_profit x = 9800) ∧ (x = 130) :=
by
  -- solve this using the properties of quadratic functions
  sorry

end profit_per_piece_correct_sales_volume_correct_maximum_monthly_profit_optimum_selling_price_is_130_l224_224886


namespace expand_remains_same_l224_224060

variable (m n : ℤ)

-- Define a function that represents expanding m and n by a factor of 3
def expand_by_factor_3 (m n : ℤ) : ℤ := 
  2 * (3 * m) / (3 * m - 3 * n)

-- Define the original fraction
def original_fraction (m n : ℤ) : ℤ :=
  2 * m / (m - n)

-- Theorem to prove that expanding m and n by a factor of 3 does not change the fraction
theorem expand_remains_same (m n : ℤ) : 
  expand_by_factor_3 m n = original_fraction m n := 
by sorry

end expand_remains_same_l224_224060


namespace one_cow_one_bag_l224_224383

theorem one_cow_one_bag (h : 50 * 1 * 50 = 50 * 50) : 50 = 50 :=
by
  sorry

end one_cow_one_bag_l224_224383


namespace laptop_weight_difference_is_3_67_l224_224719

noncomputable def karen_tote_weight : ℝ := 8
noncomputable def kevin_empty_briefcase_weight : ℝ := karen_tote_weight / 2
noncomputable def umbrella_weight : ℝ := kevin_empty_briefcase_weight / 2
noncomputable def briefcase_full_weight_rainy_day : ℝ := 2 * karen_tote_weight
noncomputable def work_papers_weight : ℝ := (briefcase_full_weight_rainy_day - umbrella_weight) / 6
noncomputable def laptop_weight : ℝ := briefcase_full_weight_rainy_day - umbrella_weight - work_papers_weight
noncomputable def weight_difference : ℝ := laptop_weight - karen_tote_weight

theorem laptop_weight_difference_is_3_67 : weight_difference = 3.67 := by
  sorry

end laptop_weight_difference_is_3_67_l224_224719


namespace pencil_pen_eraser_cost_l224_224882

-- Define the problem conditions and question
theorem pencil_pen_eraser_cost 
  (p q : ℝ)
  (h1 : 3 * p + 2 * q = 4.10)
  (h2 : 2 * p + 3 * q = 3.70) :
  p + q + 0.85 = 2.41 :=
sorry

end pencil_pen_eraser_cost_l224_224882


namespace find_T_value_l224_224367

theorem find_T_value (x y : ℤ) (R : ℤ) (h : R = 30) (h2 : (R / 2) * x * y = 21 * x + 20 * y - 13) :
    x = 3 ∧ y = 2 → x * y = 6 := by
  sorry

end find_T_value_l224_224367


namespace sequence_is_geometric_l224_224397

theorem sequence_is_geometric (a : ℕ → ℕ) (h1 : a 1 = 2) (h2 : a 2 = 3) 
  (h_rec : ∀ n, a (n + 2) = 3 * a (n + 1) - 2 * a n) :
  ∀ n, a n = 2 ^ (n - 1) + 1 := 
by
  sorry

end sequence_is_geometric_l224_224397


namespace probability_diff_colors_l224_224607

theorem probability_diff_colors :
  let total_marbles := 24
  let prob_diff_colors := 
    (4 / 24) * (5 / 23) + 
    (4 / 24) * (12 / 23) + 
    (4 / 24) * (3 / 23) + 
    (5 / 24) * (12 / 23) + 
    (5 / 24) * (3 / 23) + 
    (12 / 24) * (3 / 23)
  prob_diff_colors = 191 / 552 :=
by sorry

end probability_diff_colors_l224_224607


namespace square_area_is_correct_l224_224616

noncomputable def find_area_of_square (x : ℚ) : ℚ :=
  let side := 6 * x - 27
  side * side

theorem square_area_is_correct (x : ℚ) (h1 : 6 * x - 27 = 30 - 2 * x) :
  find_area_of_square x = 248.0625 :=
by
  sorry

end square_area_is_correct_l224_224616


namespace solution_set_even_function_l224_224552

/-- Let f be an even function, and for x in [0, ∞), f(x) = x - 1. Determine the solution set for the inequality f(x) > 1.
We prove that the solution set is {x | x < -2 or x > 2}. -/
theorem solution_set_even_function (f : ℝ → ℝ) (h_even : ∀ x, f x = f (-x))
  (h_def : ∀ x, 0 ≤ x → f x = x - 1) :
  {x : ℝ | f x > 1} = {x | x < -2 ∨ x > 2} :=
by
  sorry  -- Proof steps go here.

end solution_set_even_function_l224_224552


namespace cosine_60_degrees_l224_224642

theorem cosine_60_degrees : Real.cos (60 * Real.pi / 180) = 1 / 2 :=
by sorry

end cosine_60_degrees_l224_224642


namespace average_speed_l224_224013

theorem average_speed (d d1 d2 s1 s2 : ℝ)
    (h1 : d = 100)
    (h2 : d1 = 50)
    (h3 : d2 = 50)
    (h4 : s1 = 20)
    (h5 : s2 = 50) :
    d / ((d1 / s1) + (d2 / s2)) = 28.57 :=
by
  sorry

end average_speed_l224_224013


namespace problem1_problem2_problem3_l224_224080

noncomputable def a_n (n : ℕ) : ℕ := 3 * (2 ^ n) - 3
noncomputable def S_n (n : ℕ) : ℕ := 2 * a_n n - 3 * n

-- 1. Prove a_1 = 3 and a_2 = 9 given S_n = 2a_n - 3n
theorem problem1 (n : ℕ) (h : ∀ n > 0, S_n n = 2 * (a_n n) - 3 * n) :
    a_n 1 = 3 ∧ a_n 2 = 9 :=
  sorry

-- 2. Prove that the sequence {a_n + 3} is a geometric sequence and find the general term formula for the sequence {a_n}.
theorem problem2 (n : ℕ) (h : ∀ n > 0, S_n n = 2 * (a_n n) - 3 * n) :
    ∀ n, (a_n (n + 1) + 3) / (a_n n + 3) = 2 ∧ a_n n = 3 * (2 ^ n) - 3 :=
  sorry

-- 3. Prove {S_{n_k}} is not an arithmetic sequence given S_n = 2a_n - 3n and {n_k} is an arithmetic sequence
theorem problem3 (n_k : ℕ → ℕ) (h_arithmetic : ∃ d, ∀ k, n_k (k + 1) - n_k k = d) :
    ¬ ∃ d, ∀ k, S_n (n_k (k + 1)) - S_n (n_k k) = d :=
  sorry

end problem1_problem2_problem3_l224_224080


namespace solve_inequality_l224_224289

def inequality_solution (x : ℝ) : Prop := |2 * x - 1| - x ≥ 2 

theorem solve_inequality (x : ℝ) : 
  inequality_solution x ↔ (x ≥ 3 ∨ x ≤ -1/3) :=
by sorry

end solve_inequality_l224_224289


namespace probability_of_drawing_black_ball_l224_224390

/-- The bag contains 2 black balls and 3 white balls. 
    The balls are identical except for their colors. 
    A ball is randomly drawn from the bag. -/
theorem probability_of_drawing_black_ball (b w : ℕ) (hb : b = 2) (hw : w = 3) :
    (b + w > 0) → (b / (b + w) : ℚ) = 2 / 5 :=
by
  intros h
  rw [hb, hw]
  norm_num

end probability_of_drawing_black_ball_l224_224390


namespace S6_eq_24_l224_224144

-- Definitions based on the conditions provided
def is_arithmetic_sequence (seq : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, seq (n + 1) - seq n = seq (n + 2) - seq (n + 1)

def S : ℕ → ℝ := sorry  -- Sum of the first n terms of some arithmetic sequence

-- Given conditions
axiom S2_eq_2 : S 2 = 2
axiom S4_eq_10 : S 4 = 10

-- The main theorem to prove
theorem S6_eq_24 : S 6 = 24 :=
by 
  sorry  -- Proof is omitted

end S6_eq_24_l224_224144


namespace pyramid_volume_l224_224946

noncomputable def volume_of_pyramid (a h : ℝ) : ℝ :=
  (a^2 * h) / (4 * Real.sqrt 3)

theorem pyramid_volume (d x y : ℝ) (a h : ℝ) (edge_distance lateral_face_distance : ℝ)
  (H1 : edge_distance = 2) (H2 : lateral_face_distance = Real.sqrt 12)
  (H3 : x = 2) (H4 : y = 2 * Real.sqrt 3) (H5 : d = (a * Real.sqrt 3) / 6)
  (H6 : h = Real.sqrt (48 / 5)) :
  volume_of_pyramid a h = 216 * Real.sqrt 3 := by
  sorry

end pyramid_volume_l224_224946


namespace gamma_minus_alpha_l224_224679

theorem gamma_minus_alpha (α β γ : ℝ) (h1 : 0 < α) (h2 : α < β) (h3 : β < γ) (h4 : γ < 2 * Real.pi)
    (h5 : ∀ x : ℝ, Real.cos (x + α) + Real.cos (x + β) + Real.cos (x + γ) = 0) : 
    γ - α = (4 * Real.pi) / 3 :=
sorry

end gamma_minus_alpha_l224_224679


namespace length_of_plot_l224_224479

theorem length_of_plot (W P C r : ℝ) (hW : W = 65) (hP : P = 2.5) (hC : C = 340) (hr : r = 0.4) :
  let L := (C / r - (W + 2 * P) * P) / (W - 2 * P)
  L = 100 :=
by
  sorry

end length_of_plot_l224_224479


namespace smallest_four_digit_multiple_of_15_l224_224811

theorem smallest_four_digit_multiple_of_15 :
  ∃ n : ℕ, (1000 ≤ n ∧ n ≤ 9999) ∧ (n % 15 = 0) ∧ (∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999) ∧ (m % 15 = 0) → n ≤ m) ∧ n = 1005 :=
sorry

end smallest_four_digit_multiple_of_15_l224_224811


namespace sofia_total_time_l224_224288

-- Definitions for the conditions
def laps : ℕ := 5
def track_length : ℕ := 400  -- in meters
def speed_first_100 : ℕ := 4  -- meters per second
def speed_remaining_300 : ℕ := 5  -- meters per second

-- Times taken for respective distances
def time_first_100 (distance speed : ℕ) : ℕ := distance / speed
def time_remaining_300 (distance speed : ℕ) : ℕ := distance / speed

def time_one_lap : ℕ := time_first_100 100 speed_first_100 + time_remaining_300 300 speed_remaining_300
def total_time_seconds : ℕ := laps * time_one_lap
def total_time_minutes : ℕ := 7
def total_time_extra_seconds : ℕ := 5

-- Problem statement
theorem sofia_total_time :
  total_time_seconds = total_time_minutes * 60 + total_time_extra_seconds :=
by
  sorry

end sofia_total_time_l224_224288


namespace algorithm_find_GCD_Song_Yuan_l224_224540

theorem algorithm_find_GCD_Song_Yuan :
  (∀ method, method = "continuous subtraction" → method_finds_GCD_Song_Yuan) :=
sorry

end algorithm_find_GCD_Song_Yuan_l224_224540


namespace find_N_l224_224002

theorem find_N (a b c N : ℝ) (h1 : a + b + c = 120) (h2 : a - 10 = N) 
               (h3 : b + 10 = N) (h4 : 7 * c = N): N = 56 :=
by
  sorry

end find_N_l224_224002


namespace probability_of_rolling_greater_than_five_l224_224387

def probability_of_greater_than_five (dice_faces : Finset ℕ) (greater_than : ℕ) : ℚ := 
  let favorable_outcomes := dice_faces.filter (λ x => x > greater_than)
  favorable_outcomes.card / dice_faces.card

theorem probability_of_rolling_greater_than_five:
  probability_of_greater_than_five ({1, 2, 3, 4, 5, 6} : Finset ℕ) 5 = 1 / 6 :=
by
  sorry

end probability_of_rolling_greater_than_five_l224_224387


namespace vehicles_traveled_l224_224720

theorem vehicles_traveled (V : ℕ)
  (h1 : 40 * V = 800 * 100000000) : 
  V = 2000000000 := 
sorry

end vehicles_traveled_l224_224720


namespace find_s_at_1_l224_224260

variable (t s : ℝ → ℝ)
variable (x : ℝ)

-- Define conditions
def t_def : t x = 4 * x - 9 := by sorry

def s_def : s (t x) = x^2 + 4 * x - 5 := by sorry

-- Prove the question
theorem find_s_at_1 : s 1 = 11.25 := by
  -- Proof goes here
  sorry

end find_s_at_1_l224_224260


namespace rectangle_width_l224_224304

-- Definitions and Conditions
variables (L W : ℕ)

-- Condition 1: The perimeter of the rectangle is 16 cm
def perimeter_eq : Prop := 2 * (L + W) = 16

-- Condition 2: The width is 2 cm longer than the length
def width_eq : Prop := W = L + 2

-- Proof Statement: Given the above conditions, the width of the rectangle is 5 cm
theorem rectangle_width (h1 : perimeter_eq L W) (h2 : width_eq L W) : W = 5 := 
by
  sorry

end rectangle_width_l224_224304


namespace tangent_line_sum_l224_224081

theorem tangent_line_sum {f : ℝ → ℝ} (h_tangent : ∀ x, f x = (1/2 * x) + 2) :
  (f 1) + (deriv f 1) = 3 :=
by
  -- derive the value at x=1 and the derivative manually based on h_tangent
  sorry

end tangent_line_sum_l224_224081


namespace regular_polygon_sides_l224_224658

theorem regular_polygon_sides (n : ℕ) (h : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l224_224658


namespace permutation_identity_l224_224532

open Nat

theorem permutation_identity (n : ℕ) (h : (Nat.factorial n / Nat.factorial (n - 3)) = 6 * n) : n = 4 := 
by
  sorry

end permutation_identity_l224_224532


namespace distance_from_origin_l224_224743

theorem distance_from_origin (A : ℝ) (h : |A - 0| = 4) : A = 4 ∨ A = -4 :=
by {
  sorry
}

end distance_from_origin_l224_224743


namespace handshake_even_acquaintance_l224_224486

theorem handshake_even_acquaintance (n : ℕ) (hn : n = 225) : 
  ∃ (k : ℕ), k < n ∧ (∀ m < n, k ≠ m) :=
by sorry

end handshake_even_acquaintance_l224_224486


namespace find_a5_a6_l224_224851

-- Define the conditions for the geometric sequence
variables (a : ℕ → ℝ) (q : ℝ)
-- Let {a_n} be a geometric sequence with common ratio q
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = a n * q

-- Assume the given conditions
axiom sum_a1_a2 : a 1 + a 2 = 324
axiom sum_a3_a4 : a 3 + a 4 = 36

-- Prove that a_5 + a_6 = 4
theorem find_a5_a6 (hgeom : geometric_sequence a q) (h1 : sum_a1_a2) (h3 : sum_a3_a4) : a 5 + a 6 = 4 :=
  sorry

end find_a5_a6_l224_224851


namespace males_band_not_orchestra_l224_224738

/-- Define conditions as constants -/
def total_females_band := 150
def total_males_band := 130
def total_females_orchestra := 140
def total_males_orchestra := 160
def females_both := 90
def males_both := 80
def total_students_either := 310

/-- The number of males in the band who are NOT in the orchestra -/
theorem males_band_not_orchestra : total_males_band - males_both = 50 := by
  sorry

end males_band_not_orchestra_l224_224738


namespace proof_problem_l224_224560

open Real

theorem proof_problem 
  (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) 
  (h_sum : a + b + c + d = 1)
  : (b * c * d / (1 - a)^2) + (c * d * a / (1 - b)^2) + (d * a * b / (1 - c)^2) + (a * b * c / (1 - d)^2) ≤ 1 / 9 := 
   sorry

end proof_problem_l224_224560


namespace minimum_apples_l224_224697

theorem minimum_apples (n : ℕ) (A : ℕ) (h1 : A = 25 * n + 24) (h2 : A > 300) : A = 324 :=
sorry

end minimum_apples_l224_224697


namespace slices_with_only_mushrooms_l224_224471

theorem slices_with_only_mushrooms :
  ∀ (T P M n : ℕ),
    T = 16 →
    P = 9 →
    M = 12 →
    (9 - n) + (12 - n) + n = 16 →
    M - n = 7 :=
by
  intros T P M n hT hP hM h_eq
  sorry

end slices_with_only_mushrooms_l224_224471


namespace candy_sold_tuesday_correct_l224_224853

variable (pieces_sold_monday pieces_left_by_wednesday initial_candy total_pieces_sold : ℕ)
variable (pieces_sold_tuesday : ℕ)

-- Conditions
def initial_candy_amount := 80
def candy_sold_on_monday := 15
def candy_left_by_wednesday := 7

-- Total candy sold by Wednesday
def total_candy_sold_by_wednesday := initial_candy_amount - candy_left_by_wednesday

-- Candy sold on Tuesday
def candy_sold_on_tuesday : ℕ := total_candy_sold_by_wednesday - candy_sold_on_monday

-- Proof statement
theorem candy_sold_tuesday_correct : candy_sold_on_tuesday = 58 := sorry

end candy_sold_tuesday_correct_l224_224853


namespace three_digit_numbers_divisible_by_5_l224_224975

theorem three_digit_numbers_divisible_by_5 : ∃ n : ℕ, n = 181 ∧ ∀ x : ℕ, (100 ≤ x ∧ x ≤ 999 ∧ x % 5 = 0) → ∃ k : ℕ, x = 100 + k * 5 ∧ k < n := sorry

end three_digit_numbers_divisible_by_5_l224_224975


namespace triangle_area_l224_224386

theorem triangle_area (a b c : ℝ) (h1 : a = 5) (h2 : a + b = 13) (h3 : c = Real.sqrt (a^2 + b^2)) : 
  (1 / 2) * a * b = 20 :=
by
  sorry

end triangle_area_l224_224386


namespace gum_cost_example_l224_224381

def final_cost (pieces : ℕ) (cost_per_piece : ℕ) (discount_percentage : ℕ) : ℕ :=
  let total_cost := pieces * cost_per_piece
  let discount := total_cost * discount_percentage / 100
  total_cost - discount

theorem gum_cost_example :
  final_cost 1500 2 10 / 100 = 27 :=
by sorry

end gum_cost_example_l224_224381


namespace ella_emma_hotdogs_l224_224563

-- Definitions based on the problem conditions
def hotdogs_each_sister_wants (E : ℕ) :=
  let luke := 2 * E
  let hunter := 3 * E
  E + E + luke + hunter = 14

-- Statement we need to prove
theorem ella_emma_hotdogs (E : ℕ) (h : hotdogs_each_sister_wants E) : E = 2 :=
by
  sorry

end ella_emma_hotdogs_l224_224563


namespace problem1_problem2_l224_224937

-- Problem 1
theorem problem1 (x : ℤ) : (x - 2) ^ 2 - (x - 3) * (x + 3) = -4 * x + 13 := by
  sorry

-- Problem 2
theorem problem2 (x : ℤ) (h₁ : x ≠ 1) : 
  (x^2 + 2 * x) / (x^2 - 1) / (x + 1 + (2 * x + 1) / (x - 1)) = 1 / (x + 1) := by 
  sorry

end problem1_problem2_l224_224937


namespace centers_of_parallelograms_l224_224712

def is_skew_lines (l1 l2 l3 l4 : Line) : Prop :=
  -- A function that checks if 4 lines are pairwise skew and no three of them are parallel to the same plane.
  sorry

def count_centers_of_parallelograms (l1 l2 l3 l4 : Line) : ℕ :=
  -- A function that counts the number of lines through which the centers of parallelograms formed by the intersections of the lines pass.
  sorry

theorem centers_of_parallelograms (l1 l2 l3 l4 : Line) (h_skew: is_skew_lines l1 l2 l3 l4) : count_centers_of_parallelograms l1 l2 l3 l4 = 3 :=
  sorry

end centers_of_parallelograms_l224_224712


namespace forty_percent_of_number_l224_224234

variables {N : ℝ}

theorem forty_percent_of_number (h : (1 / 4) * (1 / 3) * (2 / 5) * N = 10) : 0.40 * N = 120 :=
by sorry

end forty_percent_of_number_l224_224234


namespace find_m_l224_224222

theorem find_m (m : ℝ) : 
  (∀ (x y : ℝ), (y = x + m ∧ x = 0) → y = m) ∧
  (∀ (x y : ℝ), (y = 2 * x - 2 ∧ x = 0) → y = -2) ∧
  (∀ (x : ℝ), (∃ y : ℝ, (y = x + m ∧ x = 0) ∧ (y = 2 * x - 2 ∧ x = 0))) → 
  m = -2 :=
by 
  sorry

end find_m_l224_224222


namespace sum_reciprocals_eq_three_l224_224314

-- Define nonzero real numbers x and y with their given condition
variables (x y : ℝ) (hx : x ≠ 0) (hy: y ≠ 0) (h : x + y = 3 * x * y)

-- State the theorem to prove the sum of reciprocals of x and y is 3
theorem sum_reciprocals_eq_three (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 3 * x * y) : (1 / x) + (1 / y) = 3 :=
sorry

end sum_reciprocals_eq_three_l224_224314


namespace num_3_digit_div_by_5_l224_224973

theorem num_3_digit_div_by_5 : 
  ∃ (n : ℕ), 
  let a := 100 in let d := 5 in let l := 995 in
  (l = a + (n-1) * d) ∧ n = 180 :=
by
  sorry

end num_3_digit_div_by_5_l224_224973


namespace total_amount_distributed_l224_224157

theorem total_amount_distributed (A : ℝ) :
  (∀ A, (A / 14 = A / 18 + 80) → A = 5040) :=
by
  sorry

end total_amount_distributed_l224_224157


namespace simplify_poly_l224_224128

open Polynomial

variable {R : Type*} [CommRing R]

-- Definition of the polynomials
def p1 : Polynomial R := 2 * X ^ 6 + 3 * X ^ 5 + X ^ 4 - X ^ 2 + 15
def p2 : Polynomial R := X ^ 6 + X ^ 5 - 2 * X ^ 4 + X ^ 3 + 5

-- Simplified polynomial
def expected_result : Polynomial R := X ^ 6 + 2 * X ^ 5 + 3 * X ^ 4 - X ^ 3 + X ^ 2 + 10

-- The theorem to state the equivalence
theorem simplify_poly : p1 - p2 = expected_result :=
by sorry

end simplify_poly_l224_224128


namespace triangle_area_l224_224806

/-- 
  Given:
  - A smaller rectangle OABD with OA = 4 cm, AB = 4 cm
  - A larger rectangle ABEC with AB = 12 cm, BC = 12 cm
  - Point O at (0,0)
  - Point A at (4,0)
  - Point B at (16,0)
  - Point C at (16,12)
  - Point D at (4,12)
  - Point E is on the line from A to C
  
  Prove the area of the triangle CDE is 54 cm²
-/
theorem triangle_area (OA AB OB DE DC : ℕ) : 
  OA = 4 ∧ AB = 4 ∧ OB = 16 ∧ DE = 12 - 3 ∧ DC = 12 → (1 / 2) * DE * DC = 54 := by 
  intros h
  sorry

end triangle_area_l224_224806


namespace christmas_bonus_remainder_l224_224566

theorem christmas_bonus_remainder (B P R : ℕ) (hP : P = 8 * B + 5) (hR : (4 * P) % 8 = R) : R = 4 :=
by
  sorry

end christmas_bonus_remainder_l224_224566


namespace steel_ingot_weight_l224_224757

theorem steel_ingot_weight 
  (initial_weight : ℕ)
  (percent_increase : ℚ)
  (ingot_cost : ℚ)
  (discount_threshold : ℕ)
  (discount_percent : ℚ)
  (total_cost : ℚ)
  (added_weight : ℚ)
  (number_of_ingots : ℕ)
  (ingot_weight : ℚ)
  (h1 : initial_weight = 60)
  (h2 : percent_increase = 0.6)
  (h3 : ingot_cost = 5)
  (h4 : discount_threshold = 10)
  (h5 : discount_percent = 0.2)
  (h6 : total_cost = 72)
  (h7 : added_weight = initial_weight * percent_increase)
  (h8 : added_weight = ingot_weight * number_of_ingots)
  (h9 : total_cost = (ingot_cost * number_of_ingots) * (1 - discount_percent)) :
  ingot_weight = 2 := 
by
  sorry

end steel_ingot_weight_l224_224757


namespace burger_share_l224_224454

theorem burger_share (burger_length : ℝ) (brother_share : ℝ) (first_friend_share : ℝ) (second_friend_share : ℝ) (valentina_share : ℝ) :
  burger_length = 12 →
  brother_share = burger_length / 3 →
  first_friend_share = (burger_length - brother_share) / 4 →
  second_friend_share = (burger_length - brother_share - first_friend_share) / 2 →
  valentina_share = burger_length - (brother_share + first_friend_share + second_friend_share) →
  brother_share = 4 ∧ first_friend_share = 2 ∧ second_friend_share = 3 ∧ valentina_share = 3 :=
by
  intros
  sorry

end burger_share_l224_224454


namespace problem_I_problem_II_l224_224826

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 4 * a * x + 1
noncomputable def g (a b : ℝ) (x : ℝ) : ℝ := 6 * a^2 * Real.log x + 2 * b + 1
noncomputable def h (a b : ℝ) (x : ℝ) : ℝ := f a x + g a b x

theorem problem_I (a : ℝ) (ha : a > 0) :
  ∃ b, b = 5 / 2 * a^2 - 3 * a^2 * Real.log a ∧ ∀ b', b' ≤ 3 / 2 * Real.exp (2 / 3) :=
sorry

theorem problem_II (a x₁ x₂ : ℝ) (ha : a ≥ Real.sqrt 3 - 1) (hx : 0 < x₁ ∧ 0 < x₂ ∧ x₁ ≠ x₂) :
  (h a b x₂ - h a b x₁) / (x₂ - x₁) > 8 :=
sorry

end problem_I_problem_II_l224_224826


namespace tangent_line_equation_l224_224584

/-- Prove that the equation of the tangent line to the curve y = x^3 - 4x^2 + 4 at the point (1,1) is y = -5x + 6 -/
theorem tangent_line_equation (x y : ℝ)
  (h_curve : y = x^3 - 4 * x^2 + 4)
  (h_point : x = 1 ∧ y = 1) :
  y = -5 * x + 6 := by
  sorry

end tangent_line_equation_l224_224584


namespace sequence_a2017_l224_224707

theorem sequence_a2017 (a : ℕ → ℚ) (h₁ : a 1 = 1 / 2)
  (h₂ : ∀ n : ℕ, 0 < n → a (n + 1) = 2 * a n / (3 * a n + 2)) :
  a 2017 = 1 / 3026 :=
sorry

end sequence_a2017_l224_224707


namespace problem_fraction_of_complex_numbers_l224_224678

/--
Given \(i\) is the imaginary unit, prove that \(\frac {1-i}{1+i} = -i\).
-/
theorem problem_fraction_of_complex_numbers (i : ℂ) (h_i : i^2 = -1) : 
  ((1 - i) / (1 + i)) = -i := 
sorry

end problem_fraction_of_complex_numbers_l224_224678


namespace exists_same_color_points_one_meter_apart_l224_224306

-- Declare the colors as an enumeration
inductive Color
| red : Color
| black : Color

-- Define the function that assigns a color to each point in the plane
def color (point : ℝ × ℝ) : Color := sorry

-- The theorem to be proven
theorem exists_same_color_points_one_meter_apart :
  ∃ x y : ℝ × ℝ, x ≠ y ∧ dist x y = 1 ∧ color x = color y :=
sorry

end exists_same_color_points_one_meter_apart_l224_224306


namespace remainders_equality_l224_224764

open Nat

theorem remainders_equality (P P' D R R' r r': ℕ) 
  (hP : P > P')
  (hP_R : P % D = R)
  (hP'_R' : P' % D = R')
  (hPP' : (P * P') % D = r)
  (hRR' : (R * R') % D = r') : r = r' := 
sorry

end remainders_equality_l224_224764


namespace binding_cost_is_correct_l224_224623

-- Definitions for the conditions used in the problem
def total_cost : ℝ := 250      -- Total cost to copy and bind 10 manuscripts
def copy_cost_per_page : ℝ := 0.05   -- Cost per page to copy
def pages_per_manuscript : ℕ := 400  -- Number of pages in each manuscript
def num_manuscripts : ℕ := 10      -- Number of manuscripts

-- The target value we want to prove
def binding_cost_per_manuscript : ℝ := 5 

-- The theorem statement proving the binding cost per manuscript
theorem binding_cost_is_correct :
  let copy_cost_per_manuscript := pages_per_manuscript * copy_cost_per_page
  let total_copy_cost := num_manuscripts * copy_cost_per_manuscript
  let total_binding_cost := total_cost - total_copy_cost
  (total_binding_cost / num_manuscripts) = binding_cost_per_manuscript :=
by
  sorry

end binding_cost_is_correct_l224_224623


namespace cos_60_eq_one_half_l224_224640

theorem cos_60_eq_one_half : Real.cos (60 * Real.pi / 180) = 1 / 2 :=
by
  -- Proof steps would go here
  sorry

end cos_60_eq_one_half_l224_224640


namespace expected_value_xi_l224_224147

-- Defining the problem parameters
def n : ℕ := 4
def p : ℝ := 3 / 5

-- Defining the random variable ξ
def ξ : ℕ → PReal := binomial_randvar n p

-- The expected value E(ξ)
theorem expected_value_xi : (expected_value ξ) = 12 / 5 := by
  sorry

end expected_value_xi_l224_224147


namespace coins_problem_l224_224032

theorem coins_problem : 
  ∃ x : ℕ, 
  (x % 8 = 6) ∧ 
  (x % 7 = 5) ∧ 
  (x % 9 = 1) ∧ 
  (x % 11 = 0) := 
by
  -- Proof to be provided here
  sorry

end coins_problem_l224_224032


namespace solve_inequality_l224_224196

theorem solve_inequality (x : ℝ) : (x^2 + 7 * x < 8) ↔ x ∈ (Set.Ioo (-8 : ℝ) 1) := by
  sorry

end solve_inequality_l224_224196


namespace total_pieces_of_candy_l224_224793

-- Define the given conditions
def students : ℕ := 43
def pieces_per_student : ℕ := 8

-- Define the goal, which is proving the total number of pieces of candy is 344
theorem total_pieces_of_candy : students * pieces_per_student = 344 :=
by
  sorry

end total_pieces_of_candy_l224_224793


namespace max_mn_l224_224134

theorem max_mn (m n : ℝ) (h : m + n = 1) : mn ≤ 1 / 4 :=
by
  sorry

end max_mn_l224_224134


namespace regular_polygon_sides_l224_224650

theorem regular_polygon_sides (h : ∀ n, (180 * (n - 2) = 150 * n)) : n = 12 := 
by
  sorry

end regular_polygon_sides_l224_224650


namespace cosine_60_degrees_l224_224644

theorem cosine_60_degrees : Real.cos (60 * Real.pi / 180) = 1 / 2 :=
by sorry

end cosine_60_degrees_l224_224644


namespace max_value_of_abc_expression_l224_224257

noncomputable def max_abc_expression (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c = 1) : ℝ :=
  a^3 * b^2 * c^2

theorem max_value_of_abc_expression (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c = 1) :
  max_abc_expression a b c h1 h2 h3 h4 ≤ 432 / 7^7 :=
sorry

end max_value_of_abc_expression_l224_224257


namespace main_inequality_l224_224557

theorem main_inequality (a b c d : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_c_pos : 0 < c) (h_d_pos : 0 < d) (h_sum : a + b + c + d = 1) :
  (b * c * d) / (1 - a)^2 + (c * d * a) / (1 - b)^2 + (d * a * b) / (1 - c)^2 + (a * b * c) / (1 - d)^2 ≤ 1 / 9 :=
by
  sorry

end main_inequality_l224_224557


namespace solution_of_inequality_l224_224312

theorem solution_of_inequality (x : ℝ) : -2 * x - 1 < -1 → x > 0 :=
by
  sorry

end solution_of_inequality_l224_224312


namespace math_problem_l224_224456

theorem math_problem :
  (50 - (4050 - 450)) * (4050 - (450 - 50)) = -12957500 := 
by
  sorry

end math_problem_l224_224456


namespace mike_payments_total_months_l224_224565

-- Definitions based on conditions
def lower_rate := 295
def higher_rate := 310
def lower_payments := 5
def higher_payments := 7
def total_paid := 3615

-- The statement to prove
theorem mike_payments_total_months : lower_payments + higher_payments = 12 := by
  -- Proof goes here
  sorry

end mike_payments_total_months_l224_224565


namespace max_valid_subset_size_l224_224373

open Finset

-- Definition of the set I
def I : Finset (fin 4 → ℕ) :=
  {x | ∀ i, x i ∈ (Ico 1 12)}

-- Definition of the condition on subset A
def valid_subset (A : Finset (fin 4 → ℕ)) : Prop :=
  A ⊆ I ∧
  ∀ (x y : fin 4 → ℕ), (x ∈ A ∧ y ∈ A) →
  ∃ i j, (1 ≤ i) ∧ (i < j) ∧ (j ≤ 4) ∧ (x i - x j) * (y i - y j) < 0

-- The maximum size of such a subset
def max_size_of_valid_subset : ℕ :=
  891

-- The Lean statement for the proof problem
theorem max_valid_subset_size :
  ∃ A : Finset (fin 4 → ℕ), valid_subset A ∧ A.card = max_size_of_valid_subset :=
sorry

end max_valid_subset_size_l224_224373


namespace like_term_l224_224869

theorem like_term (a : ℝ) : ∃ (a : ℝ), a * x ^ 5 * y ^ 3 = a * x ^ 5 * y ^ 3 :=
by sorry

end like_term_l224_224869


namespace max_closable_companies_l224_224590

def number_of_planets : ℕ := 10 ^ 2015
def number_of_companies : ℕ := 2015

theorem max_closable_companies (k : ℕ) : k = 1007 :=
sorry

end max_closable_companies_l224_224590


namespace Kendall_dimes_l224_224406

theorem Kendall_dimes (total_value : ℝ) (quarters : ℝ) (dimes : ℝ) (nickels : ℝ) 
  (num_quarters : ℕ) (num_nickels : ℕ) 
  (total_amount : total_value = 4)
  (quarter_amount : quarters = num_quarters * 0.25)
  (num_quarters_val : num_quarters = 10)
  (nickel_amount : nickels = num_nickels * 0.05) 
  (num_nickels_val : num_nickels = 6) :
  dimes = 12 := by
  sorry

end Kendall_dimes_l224_224406


namespace dog_revs_l224_224782

theorem dog_revs (r₁ r₂ : ℝ) (n₁ : ℕ) (n₂ : ℕ) (h₁ : r₁ = 48) (h₂ : n₁ = 40) (h₃ : r₂ = 12) :
  n₂ = 160 := 
sorry

end dog_revs_l224_224782


namespace equivalent_expression_l224_224413

theorem equivalent_expression (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) (h1 : a + b + c = 0) :
  (a^4 * b^4 + a^4 * c^4 + b^4 * c^4) / ((a^2 - b*c)^2 * (b^2 - a*c)^2 * (c^2 - a*b)^2) = 
  1 / (a^2 - b*c)^2 :=
by
  sorry

end equivalent_expression_l224_224413


namespace clara_current_age_l224_224485

theorem clara_current_age (a c : ℕ) (h1 : a = 54) (h2 : (c - 41) = 3 * (a - 41)) : c = 80 :=
by
  -- This is where the proof would be constructed.
  sorry

end clara_current_age_l224_224485


namespace find_M_M_superset_N_M_intersection_N_l224_224828

-- Define the set M as per the given condition
def M : Set ℝ := { x : ℝ | x^2 - x - 2 < 0 }

-- Define the set N based on parameters a and b
def N (a b : ℝ) : Set ℝ := { x : ℝ | a < x ∧ x < b }

-- Prove that M = (-1, 2)
theorem find_M : M = { x : ℝ | -1 < x ∧ x < 2 } :=
sorry

-- Prove that if M ⊇ N, then a ≥ -1
theorem M_superset_N (a b : ℝ) (h : M ⊇ N a b) : -1 ≤ a :=
sorry

-- Prove that if M ∩ N = M, then b ≥ 2
theorem M_intersection_N (a b : ℝ) (h : M ∩ (N a b) = M) : 2 ≤ b :=
sorry

end find_M_M_superset_N_M_intersection_N_l224_224828


namespace both_boys_and_girls_selected_probability_l224_224184

theorem both_boys_and_girls_selected_probability :
  let total_students := 5
  let boys := 2
  let girls := 3
  let selected_students := 3
  let total_ways := Nat.choose total_students selected_students
  let only_girls_ways := Nat.choose girls selected_students
  (1 - (only_girls_ways / total_ways : ℚ)) = (9 / 10 : ℚ) :=
by
  let total_students := 5
  let boys := 2
  let girls := 3
  let selected_students := 3
  let total_ways := Nat.choose total_students selected_students
  let only_girls_ways := Nat.choose girls selected_students
  have h : (only_girls_ways / total_ways : ℚ) = (1 / 10 : ℚ) := sorry
  have h1 : (1 - (only_girls_ways / total_ways : ℚ)) = (9 / 10 : ℚ) := by rw [h]; norm_num
  exact h1

end both_boys_and_girls_selected_probability_l224_224184


namespace stella_doll_price_l224_224575

theorem stella_doll_price 
  (dolls_count clocks_count glasses_count : ℕ)
  (price_per_clock price_per_glass cost profit : ℕ)
  (D : ℕ)
  (h1 : dolls_count = 3)
  (h2 : clocks_count = 2)
  (h3 : glasses_count = 5)
  (h4 : price_per_clock = 15)
  (h5 : price_per_glass = 4)
  (h6 : cost = 40)
  (h7 : profit = 25)
  (h8 : 3 * D + 2 * price_per_clock + 5 * price_per_glass = cost + profit) :
  D = 5 :=
by
  sorry

end stella_doll_price_l224_224575


namespace average_infections_per_round_infections_after_three_rounds_l224_224504

-- Define the average number of infections per round such that the total after two rounds is 36 and x > 0
theorem average_infections_per_round :
  ∃ x : ℤ, (1 + x)^2 = 36 ∧ x > 0 :=
by
  sorry

-- Given x = 5, prove that the total number of infections after three rounds exceeds 200
theorem infections_after_three_rounds (x : ℤ) (H : x = 5) :
  (1 + x)^3 > 200 :=
by
  sorry

end average_infections_per_round_infections_after_three_rounds_l224_224504


namespace total_cost_is_135_25_l224_224316

-- defining costs and quantities
def cost_A : ℕ := 9
def num_A : ℕ := 4
def cost_B := cost_A + 5
def num_B : ℕ := 2
def cost_clay_pot := cost_A + 20
def cost_bag_soil := cost_A - 2
def cost_fertilizer := cost_A + (cost_A / 2)
def cost_gardening_tools := cost_clay_pot - (cost_clay_pot / 4)

-- total cost calculation
def total_cost : ℚ :=
  (num_A * cost_A) + 
  (num_B * cost_B) + 
  cost_clay_pot + 
  cost_bag_soil + 
  cost_fertilizer + 
  cost_gardening_tools

theorem total_cost_is_135_25 : total_cost = 135.25 := by
  sorry

end total_cost_is_135_25_l224_224316


namespace probability_of_3_correct_answers_is_31_over_135_expected_value_of_total_score_is_50_l224_224710

noncomputable def probability_correct_answers : ℚ :=
  let pA := (1/5 : ℚ)
  let pB := (3/5 : ℚ)
  let pC := (1/5 : ℚ)
  ((pA * (3/9 : ℚ) * (2/3)^2 * (1/3)) + (pB * (6/9 : ℚ) * (2/3) * (1/3)^2) + (pC * (1/9 : ℚ) * (1/3)^3))

theorem probability_of_3_correct_answers_is_31_over_135 :
  probability_correct_answers = 31 / 135 := by
  sorry

noncomputable def expected_score : ℚ :=
  let E_m := (1/5 * 1 + 3/5 * 2 + 1/5 * 3 : ℚ)
  let E_n := (3 * (2/3 : ℚ))
  (15 * E_m + 10 * E_n)

theorem expected_value_of_total_score_is_50 :
  expected_score = 50 := by
  sorry

end probability_of_3_correct_answers_is_31_over_135_expected_value_of_total_score_is_50_l224_224710


namespace continuous_iff_integral_condition_l224_224415

open Real 

noncomputable section

def is_non_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

def integral_condition (f : ℝ → ℝ) (a : ℝ) (a_seq : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, (∫ x in a..(a + a_seq n), f x) + (∫ x in (a - a_seq n)..a, f x) ≤ (a_seq n) / n

theorem continuous_iff_integral_condition (a : ℝ) (f : ℝ → ℝ)
  (h_nondec : is_non_decreasing f) :
  ContinuousAt f a ↔ ∃ (a_seq : ℕ → ℝ), (∀ n, 0 < a_seq n) ∧ integral_condition f a a_seq := sorry

end continuous_iff_integral_condition_l224_224415


namespace cone_radius_height_ratio_l224_224180

theorem cone_radius_height_ratio 
  (V : ℝ) (π : ℝ) (r h : ℝ)
  (circumference : ℝ) 
  (original_height : ℝ)
  (new_volume : ℝ)
  (volume_formula : V = (1/3) * π * r^2 * h)
  (radius_from_circumference : 2 * π * r = circumference)
  (base_circumference : circumference = 28 * π)
  (original_height_eq : original_height = 45)
  (new_volume_eq : new_volume = 441 * π) :
  (r / h) = 14 / 9 :=
by
  sorry

end cone_radius_height_ratio_l224_224180


namespace sum_of_solutions_l224_224064

theorem sum_of_solutions (x : ℤ) (h : x^4 - 13 * x^2 + 36 = 0) : 
  (finset.sum (finset.filter (λ (x : ℤ), x^4 - 13 * x^2 + 36 = 0) (finset.range 4))) = 0 :=
by
  sorry

end sum_of_solutions_l224_224064


namespace pyramid_height_l224_224333

noncomputable def height_of_pyramid : ℝ :=
  let perimeter := 32
  let pb := 12
  let side := perimeter / 4
  let fb := (side * Real.sqrt 2) / 2
  Real.sqrt (pb^2 - fb^2)

theorem pyramid_height :
  height_of_pyramid = 4 * Real.sqrt 7 :=
by
  sorry

end pyramid_height_l224_224333


namespace original_decimal_l224_224273

theorem original_decimal (x : ℝ) : (10 * x = x + 2.7) → x = 0.3 := 
by
    intro h
    sorry

end original_decimal_l224_224273


namespace pureGalaTrees_l224_224606

theorem pureGalaTrees {T F C : ℕ} (h1 : F + C = 204) (h2 : F = (3 / 4 : ℝ) * T) (h3 : C = (1 / 10 : ℝ) * T) : (0.15 * T : ℝ) = 36 :=
by
  sorry

end pureGalaTrees_l224_224606


namespace password_probability_l224_224348

theorem password_probability :
  let even_digits := [0, 2, 4, 6, 8]
  let vowels := ['A', 'E', 'I', 'O', 'U']
  let non_zero_digits := [1, 2, 3, 4, 5, 6, 7, 8, 9]
  (even_digits.length / 10) * (vowels.length / 26) * (non_zero_digits.length / 10) = 9 / 52 :=
by
  sorry

end password_probability_l224_224348


namespace inequality_arith_geo_mean_l224_224723

variable (a k : ℝ)
variable (h1 : 1 ≤ k)
variable (h2 : k ≤ 3)
variable (h3 : 0 < k)

theorem inequality_arith_geo_mean (h1 : 1 ≤ k) (h2 : k ≤ 3) (h3 : 0 < k):
    ( (a + k * a) / 2 ) ^ 2 ≥ ( (a * (k * a)) ^ (1/2) ) ^ 2 :=
by
  sorry

end inequality_arith_geo_mean_l224_224723


namespace marbles_per_boy_l224_224844

theorem marbles_per_boy (boys marbles : ℕ) (h1 : boys = 5) (h2 : marbles = 35) : marbles / boys = 7 := by
  sorry

end marbles_per_boy_l224_224844


namespace roses_in_centerpiece_l224_224728

variable (r : ℕ)

theorem roses_in_centerpiece (h : 6 * 15 * (3 * r + 6) = 2700) : r = 8 := 
  sorry

end roses_in_centerpiece_l224_224728


namespace fraction_equality_l224_224951

theorem fraction_equality (a b : ℝ) (h : a / b = 2) : a / (a - b) = 2 :=
by
  sorry

end fraction_equality_l224_224951


namespace gazprom_rnd_costs_calc_l224_224491

theorem gazprom_rnd_costs_calc (R_D_t ΔAPL_t1 : ℝ) (h1 : R_D_t = 3157.61) (h2 : ΔAPL_t1 = 0.69) :
  R_D_t / ΔAPL_t1 = 4576 :=
by
  sorry

end gazprom_rnd_costs_calc_l224_224491


namespace area_of_quadrilateral_EFGH_l224_224992

noncomputable def trapezium_ABCD_midpoints_area : ℝ :=
  let A := (0, 0)
  let B := (2, 0)
  let C := (4, 3)
  let D := (0, 3)
  let E := ((B.1 + C.1)/2, (B.2 + C.2)/2) -- midpoint of BC
  let F := ((C.1 + D.1)/2, (C.2 + D.2)/2) -- midpoint of CD
  let G := ((A.1 + D.1)/2, (A.2 + D.2)/2) -- midpoint of AD
  let H := ((G.1 + E.1)/2, (G.2 + E.2)/2) -- midpoint of GE
  let area := (E.1 * F.2 + F.1 * G.2 + G.1 * H.2 + H.1 * E.2 - F.1 * E.2 - G.1 * F.2 - H.1 * G.2 - E.1 * H.2) / 2
  abs area

theorem area_of_quadrilateral_EFGH : trapezium_ABCD_midpoints_area = 0.75 := by
  sorry

end area_of_quadrilateral_EFGH_l224_224992


namespace find_principal_amount_l224_224618

-- Define the parameters
def R : ℝ := 11.67
def T : ℝ := 5
def A : ℝ := 950

-- State the theorem
theorem find_principal_amount : ∃ P : ℝ, A = P * (1 + (R/100) * T) :=
by { 
  use 600, 
  -- Skip the proof 
  sorry 
}

end find_principal_amount_l224_224618


namespace sum_of_two_numbers_l224_224891

theorem sum_of_two_numbers (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h1 : x * y = 9) (h2 : (1 / x) = 4 * (1 / y)) : x + y = 15 / 2 :=
  sorry

end sum_of_two_numbers_l224_224891


namespace quadrilateral_ABCD_is_rectangle_l224_224372

noncomputable def point := (ℤ × ℤ)

def A : point := (-2, 0)
def B : point := (1, 6)
def C : point := (5, 4)
def D : point := (2, -2)

def vector (p1 p2 : point) : point := (p2.1 - p1.1, p2.2 - p1.2)

def dot_product (v1 v2 : point) : ℤ := (v1.1 * v2.1) + (v1.2 * v2.2)

def is_perpendicular (v1 v2 : point) : Prop := dot_product v1 v2 = 0

def is_rectangle (A B C D : point) :=
  vector A B = vector C D ∧ is_perpendicular (vector A B) (vector A D)

theorem quadrilateral_ABCD_is_rectangle : is_rectangle A B C D :=
by
  sorry

end quadrilateral_ABCD_is_rectangle_l224_224372


namespace harper_jack_distance_apart_l224_224374

def total_distance : ℕ := 1000
def distance_jack_run : ℕ := 152
def distance_apart (total_distance : ℕ) (distance_jack_run : ℕ) : ℕ :=
  total_distance - distance_jack_run 

theorem harper_jack_distance_apart :
  distance_apart total_distance distance_jack_run = 848 :=
by
  unfold distance_apart
  sorry

end harper_jack_distance_apart_l224_224374


namespace cos_60_eq_one_half_l224_224638

theorem cos_60_eq_one_half : Real.cos (60 * Real.pi / 180) = 1 / 2 :=
by
  -- Proof steps would go here
  sorry

end cos_60_eq_one_half_l224_224638


namespace trig_inequalities_l224_224748

theorem trig_inequalities :
  let sin_168 := Real.sin (168 * Real.pi / 180)
  let cos_10 := Real.cos (10 * Real.pi / 180)
  let tan_58 := Real.tan (58 * Real.pi / 180)
  let tan_45 := Real.tan (45 * Real.pi / 180)
  sin_168 < cos_10 ∧ cos_10 < tan_58 :=
  sorry

end trig_inequalities_l224_224748


namespace remainder_when_divided_by_15_l224_224116

theorem remainder_when_divided_by_15 (c d : ℤ) (h1 : c % 60 = 47) (h2 : d % 45 = 14) : (c + d) % 15 = 1 :=
  sorry

end remainder_when_divided_by_15_l224_224116


namespace Dave_needs_31_gallons_l224_224940

noncomputable def numberOfGallons (numberOfTanks : ℕ) (height : ℝ) (diameter : ℝ) (coveragePerGallon : ℝ) : ℕ :=
  let radius := diameter / 2
  let lateral_surface_area := 2 * Real.pi * radius * height
  let total_surface_area := lateral_surface_area * numberOfTanks
  let gallons_needed := total_surface_area / coveragePerGallon
  Nat.ceil gallons_needed

theorem Dave_needs_31_gallons :
  numberOfGallons 20 24 8 400 = 31 :=
by
  sorry

end Dave_needs_31_gallons_l224_224940


namespace front_wheel_revolutions_l224_224140

theorem front_wheel_revolutions (P_front P_back : ℕ) (R_back : ℕ) (H1 : P_front = 30) (H2 : P_back = 20) (H3 : R_back = 360) :
  ∃ F : ℕ, F = 240 := by
  sorry

end front_wheel_revolutions_l224_224140


namespace servings_per_day_l224_224546

-- Definitions based on the given problem conditions
def serving_size : ℚ := 0.5
def container_size : ℚ := 32 - 2 -- 1 quart is 32 ounces and the jar is 2 ounces less
def days_last : ℕ := 20

-- The theorem statement to prove
theorem servings_per_day (h1 : serving_size = 0.5) (h2 : container_size = 30) (h3 : days_last = 20) :
  (container_size / days_last) / serving_size = 3 :=
by
  sorry

end servings_per_day_l224_224546


namespace max_quotient_l224_224089

theorem max_quotient (a b : ℕ) 
  (h1 : 400 ≤ a) (h2 : a ≤ 800) 
  (h3 : 400 ≤ b) (h4 : b ≤ 1600) 
  (h5 : a + b ≤ 2000) 
  : b / a ≤ 4 := 
sorry

end max_quotient_l224_224089


namespace alyssas_weekly_allowance_l224_224046

-- Define the constants and parameters
def spent_on_movies (A : ℝ) := 0.5 * A
def spent_on_snacks (A : ℝ) := 0.2 * A
def saved_for_future (A : ℝ) := 0.25 * A

-- Define the remaining allowance after expenses
def remaining_allowance_after_expenses (A : ℝ) := A - spent_on_movies A - spent_on_snacks A - saved_for_future A

-- Define Alyssa's allowance given the conditions
theorem alyssas_weekly_allowance : ∀ (A : ℝ), 
  remaining_allowance_after_expenses A = 12 → 
  A = 240 :=
by
  -- Proof omitted
  sorry

end alyssas_weekly_allowance_l224_224046


namespace Milly_took_extra_balloons_l224_224864

theorem Milly_took_extra_balloons :
  let total_packs := 3 + 2
  let balloons_per_pack := 6
  let total_balloons := total_packs * balloons_per_pack
  let even_split := total_balloons / 2
  let Floretta_balloons := 8
  let Milly_extra_balloons := even_split - Floretta_balloons
  Milly_extra_balloons = 7 := by
  sorry

end Milly_took_extra_balloons_l224_224864


namespace shapes_remaining_after_turns_l224_224466

-- Define the initial conditions for the problem
def initial_shapes : ℕ := 12
def triangles : ℕ := 3
def squares : ℕ := 4
def pentagons : ℕ := 5
def turns_per_player : ℕ := 5

-- Define the main question to prove
theorem shapes_remaining_after_turns 
  (initial_shapes = 12)
  (turns_per_player = 5)
  (petya_starts : Prop)
  (no_shared_sides : Prop)
  (petya_strategy : Prop)
  (vasya_strategy : Prop) :
  ∃ remaining_shapes : ℕ, remaining_shapes = 6 := 
sorry

end shapes_remaining_after_turns_l224_224466


namespace total_hockey_games_l224_224450

theorem total_hockey_games (games_per_month : ℕ) (months_in_season : ℕ) 
(h1 : games_per_month = 13) (h2 : months_in_season = 14) : 
games_per_month * months_in_season = 182 := 
by
  -- we can simplify using the given conditions
  sorry

end total_hockey_games_l224_224450


namespace find_c_l224_224542

theorem find_c (c : ℝ) (h : (-(c / 3) + -(c / 5) = 30)) : c = -56.25 :=
sorry

end find_c_l224_224542


namespace ways_to_divide_8_friends_l224_224831

theorem ways_to_divide_8_friends : (4 : ℕ) ^ 8 = 65536 := by
  sorry

end ways_to_divide_8_friends_l224_224831


namespace value_of_f5_and_f_neg5_l224_224210

noncomputable def f (a b c x : ℝ) : ℝ := a * x^7 - b * x^5 + c * x^3 + 2

theorem value_of_f5_and_f_neg5 (a b c : ℝ) (m : ℝ) (h : f a b c (-5) = m) :
  f a b c 5 + f a b c (-5) = 4 :=
sorry

end value_of_f5_and_f_neg5_l224_224210


namespace harvey_sold_17_steaks_l224_224967

variable (initial_steaks : ℕ) (steaks_left_after_first_sale : ℕ) (steaks_sold_in_second_sale : ℕ)

noncomputable def total_steaks_sold (initial_steaks steaks_left_after_first_sale steaks_sold_in_second_sale : ℕ) : ℕ :=
  (initial_steaks - steaks_left_after_first_sale) + steaks_sold_in_second_sale

theorem harvey_sold_17_steaks :
  initial_steaks = 25 →
  steaks_left_after_first_sale = 12 →
  steaks_sold_in_second_sale = 4 →
  total_steaks_sold initial_steaks steaks_left_after_first_sale steaks_sold_in_second_sale = 17 :=
by
  intros
  sorry

end harvey_sold_17_steaks_l224_224967


namespace students_in_second_class_l224_224448

variable (x : ℕ)

theorem students_in_second_class :
  (∃ x, 30 * 40 + 70 * x = (30 + x) * 58.75) → x = 50 :=
by
  sorry

end students_in_second_class_l224_224448


namespace perimeter_of_star_is_160_l224_224513

-- Define the radius of the circles
def radius := 5 -- in cm

-- Define the diameter based on radius
def diameter := 2 * radius

-- Define the side length of the square
def side_length_square := 2 * diameter

-- Define the side length of each equilateral triangle
def side_length_triangle := side_length_square

-- Define the perimeter of the four-pointed star
def perimeter_star := 8 * side_length_triangle

-- Statement: The perimeter of the star is 160 cm
theorem perimeter_of_star_is_160 :
  perimeter_star = 160 := by
    sorry

end perimeter_of_star_is_160_l224_224513


namespace Andrew_has_5_more_goats_than_twice_Adam_l224_224933

-- Definitions based on conditions
def goats_Adam := 7
def goats_Ahmed := 13
def goats_Andrew := goats_Ahmed + 6
def twice_goats_Adam := 2 * goats_Adam

-- Theorem statement
theorem Andrew_has_5_more_goats_than_twice_Adam :
  goats_Andrew - twice_goats_Adam = 5 :=
by
  sorry

end Andrew_has_5_more_goats_than_twice_Adam_l224_224933


namespace fraction_equality_l224_224950

theorem fraction_equality (a b : ℝ) (h : a / b = 2) : a / (a - b) = 2 :=
by
  sorry

end fraction_equality_l224_224950


namespace min_score_seventh_shot_to_break_record_shots_hitting_10_to_break_record_when_7th_shot_is_8_necessary_shot_of_10_when_7th_shot_is_10_l224_224791

-- Definitions for the problem conditions
def initial_points : ℕ := 52
def record_points : ℕ := 89
def max_shots : ℕ := 10
def points_range : Finset ℕ := Finset.range 11 \ {0}

-- Lean statement for the first question
theorem min_score_seventh_shot_to_break_record (x₇ : ℕ) (h₁: x₇ ∈ points_range) :
  initial_points + x₇ + 30 > record_points ↔ x₇ ≥ 8 :=
by sorry

-- Lean statement for the second question
theorem shots_hitting_10_to_break_record_when_7th_shot_is_8 (x₈ x₉ x₁₀ : ℕ)
  (h₂ : 8 ∈ points_range) 
  (h₃ : x₈ ∈ points_range) (h₄ : x₉ ∈ points_range) (h₅ : x₁₀ ∈ points_range) :
  initial_points + 8 + x₈ + x₉ + x₁₀ > record_points ↔ (x₈ = 10 ∧ x₉ = 10 ∧ x₁₀ = 10) :=
by sorry

-- Lean statement for the third question
theorem necessary_shot_of_10_when_7th_shot_is_10 (x₈ x₉ x₁₀ : ℕ)
  (h₆ : 10 ∈ points_range)
  (h₇ : x₈ ∈ points_range) (h₈ : x₉ ∈ points_range) (h₉ : x₁₀ ∈ points_range) :
  initial_points + 10 + x₈ + x₉ + x₁₀ > record_points ↔ (x₈ = 10 ∨ x₉ = 10 ∨ x₁₀ = 10) :=
by sorry

end min_score_seventh_shot_to_break_record_shots_hitting_10_to_break_record_when_7th_shot_is_8_necessary_shot_of_10_when_7th_shot_is_10_l224_224791


namespace average_white_paper_per_ton_trees_saved_per_ton_l224_224177

-- Define the given conditions
def waste_paper_tons : ℕ := 5
def produced_white_paper_tons : ℕ := 4
def saved_trees : ℕ := 40

-- State the theorems that need to be proved
theorem average_white_paper_per_ton :
  (produced_white_paper_tons : ℚ) / waste_paper_tons = 0.8 := 
sorry

theorem trees_saved_per_ton :
  (saved_trees : ℚ) / waste_paper_tons = 8 := 
sorry

end average_white_paper_per_ton_trees_saved_per_ton_l224_224177


namespace original_pencils_l224_224594

-- Define the conditions
def pencils_added : ℕ := 30
def total_pencils_now : ℕ := 71

-- Define the theorem to prove the original number of pencils
theorem original_pencils (original_pencils : ℕ) :
  total_pencils_now = original_pencils + pencils_added → original_pencils = 41 :=
by
  intros h
  sorry

end original_pencils_l224_224594


namespace inequality_hold_l224_224556

theorem inequality_hold (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a + b + c = 3) :
  1/a^2 + 1/b^2 + 1/c^2 ≥ a^2 + b^2 + c^2 :=
by
  -- Proof goes here
  sorry

end inequality_hold_l224_224556


namespace total_points_first_four_games_l224_224801

-- Define the scores for the first three games
def score1 : ℕ := 10
def score2 : ℕ := 14
def score3 : ℕ := 6

-- Define the score for the fourth game as the average of the first three games
def score4 : ℕ := (score1 + score2 + score3) / 3

-- Define the total points scored in the first four games
def total_points : ℕ := score1 + score2 + score3 + score4

-- State the theorem to prove
theorem total_points_first_four_games : total_points = 40 :=
  sorry

end total_points_first_four_games_l224_224801


namespace odd_expressions_l224_224430

theorem odd_expressions (m n p : ℕ) (hm : m % 2 = 1) (hn : n % 2 = 1) (hp : p % 2 = 0) : 
  ((2 * m * n + 5) ^ 2 % 2 = 1) ∧ (5 * m * n + p % 2 = 1) := 
by
  sorry

end odd_expressions_l224_224430


namespace percentage_increase_l224_224442

theorem percentage_increase (P : ℝ) (x : ℝ) 
(h1 : 1.17 * P = 0.90 * P * (1 + x / 100)) : x = 33.33 :=
by sorry

end percentage_increase_l224_224442


namespace value_of_a_l224_224960

noncomputable def f : ℝ → ℝ 
| x => if x > 0 then 2^x else x + 1

theorem value_of_a (a : ℝ) (h : f a + f 1 = 0) : a = -3 :=
by
  sorry

end value_of_a_l224_224960


namespace constant_term_expansion_l224_224959

theorem constant_term_expansion (x : ℝ) (n : ℕ) (h : (x + 2 + 1/x)^n = 20) : n = 3 :=
by
sorry

end constant_term_expansion_l224_224959


namespace Anil_profit_in_rupees_l224_224048

def cost_scooter (C : ℝ) : Prop := 0.10 * C = 500
def profit (C P : ℝ) : Prop := P = 0.20 * C

theorem Anil_profit_in_rupees (C P : ℝ) (h1 : cost_scooter C) (h2 : profit C P) : P = 1000 :=
by
  sorry

end Anil_profit_in_rupees_l224_224048


namespace red_socks_l224_224249

variable {R : ℕ}

theorem red_socks (h1 : 2 * R + R + 6 * R = 90) : R = 10 := 
by
  sorry

end red_socks_l224_224249


namespace slices_needed_l224_224424

def number_of_sandwiches : ℕ := 5
def slices_per_sandwich : ℕ := 3
def total_slices_required (n : ℕ) (s : ℕ) : ℕ := n * s

theorem slices_needed : total_slices_required number_of_sandwiches slices_per_sandwich = 15 :=
by
  sorry

end slices_needed_l224_224424


namespace inequality_abc_l224_224281

theorem inequality_abc 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) : 
  a * b * c ≥ (b + c - a) * (a + c - b) * (a + b - c) := 
by 
  sorry

end inequality_abc_l224_224281


namespace pyramid_height_l224_224335

theorem pyramid_height (perimeter_side_base : ℝ) (apex_distance_to_vertex : ℝ) (height_peak_to_center_base : ℝ) : 
  (perimeter_side_base = 32) → (apex_distance_to_vertex = 12) → 
  height_peak_to_center_base = 4 * Real.sqrt 7 := 
  by
    sorry

end pyramid_height_l224_224335


namespace product_of_mixed_numbers_l224_224760

theorem product_of_mixed_numbers :
  let fraction1 := (13 : ℚ) / 6
  let fraction2 := (29 : ℚ) / 9
  (fraction1 * fraction2) = 377 / 54 := 
by
  sorry

end product_of_mixed_numbers_l224_224760


namespace remainder_of_division_987543_12_l224_224761

theorem remainder_of_division_987543_12 : 987543 % 12 = 7 := by
  sorry

end remainder_of_division_987543_12_l224_224761


namespace radius_of_circle_l224_224890

theorem radius_of_circle (r : ℝ) (h : 3 * (2 * Real.pi * r) = 2 * (Real.pi * r ^ 2)) : r = 3 := by
  sorry

end radius_of_circle_l224_224890


namespace total_length_of_visible_edges_l224_224338

theorem total_length_of_visible_edges (shortest_side : ℕ) (removed_side : ℕ) (longest_side : ℕ) (new_visible_sides_sum : ℕ) 
  (h1 : shortest_side = 4) 
  (h2 : removed_side = 2 * shortest_side) 
  (h3 : removed_side = longest_side / 2) 
  (h4 : longest_side = 16) 
  (h5 : new_visible_sides_sum = shortest_side + removed_side + removed_side) : 
  new_visible_sides_sum = 20 := by 
sorry

end total_length_of_visible_edges_l224_224338


namespace initial_ants_count_l224_224929

theorem initial_ants_count (n : ℕ) (h1 : ∀ x : ℕ, x ≠ n - 42 → x ≠ 42) : n = 42 :=
sorry

end initial_ants_count_l224_224929


namespace four_digit_numbers_count_l224_224138

theorem four_digit_numbers_count : (3:ℕ) ^ 4 = 81 := by
  sorry

end four_digit_numbers_count_l224_224138


namespace regular_polygon_sides_l224_224665

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, 1 ≤ i → i ≤ n → (interior_angle i = 150) ) 
  (sum_interior_angles : ∑ i in range n, interior_angle i = 180 * (n - 2))
  {interior_angle : ℕ → ℕ} :
  n = 12 :=
by
  sorry

end regular_polygon_sides_l224_224665


namespace travis_discount_percentage_l224_224148

theorem travis_discount_percentage (P D : ℕ) (hP : P = 2000) (hD : D = 1400) :
  ((P - D) / P * 100) = 30 := by
  -- sorry to skip the proof
  sorry

end travis_discount_percentage_l224_224148


namespace sum_of_three_numbers_l224_224432

noncomputable def lcm_three_numbers (a b c : ℕ) : ℕ := Nat.lcm (Nat.lcm a b) c

theorem sum_of_three_numbers 
  (a b c : ℕ)
  (x : ℕ)
  (h1 : lcm_three_numbers a b c = 180)
  (h2 : a = 2 * x)
  (h3 : b = 3 * x)
  (h4 : c = 5 * x) : a + b + c = 60 :=
by
  sorry

end sum_of_three_numbers_l224_224432


namespace train_passes_jogger_in_37_seconds_l224_224023

noncomputable def jogger_speed_kmph : ℝ := 9
noncomputable def train_speed_kmph : ℝ := 45
noncomputable def jogger_lead_m : ℝ := 250
noncomputable def train_length_m : ℝ := 120

noncomputable def jogger_speed_mps : ℝ := jogger_speed_kmph * 1000 / 3600
noncomputable def train_speed_mps : ℝ := train_speed_kmph * 1000 / 3600
noncomputable def relative_speed_mps : ℝ := train_speed_mps - jogger_speed_mps
noncomputable def total_distance_m : ℝ := jogger_lead_m + train_length_m

theorem train_passes_jogger_in_37_seconds :
  total_distance_m / relative_speed_mps = 37 := by
  sorry

end train_passes_jogger_in_37_seconds_l224_224023


namespace ashok_avg_first_five_l224_224344

-- Define the given conditions 
def avg (n : ℕ) (s : ℕ) : ℕ := s / n

def total_marks (average : ℕ) (num_subjects : ℕ) : ℕ := average * num_subjects

variables (avg_six_subjects : ℕ := 76)
variables (sixth_subject_marks : ℕ := 86)
variables (total_six_subjects : ℕ := total_marks avg_six_subjects 6)
variables (total_first_five_subjects : ℕ := total_six_subjects - sixth_subject_marks)
variables (avg_first_five_subjects : ℕ := avg 5 total_first_five_subjects)

-- State the theorem
theorem ashok_avg_first_five 
  (h1 : avg_six_subjects = 76)
  (h2 : sixth_subject_marks = 86)
  (h3 : avg_first_five_subjects = 74)
  : avg 5 (total_marks 76 6 - 86) = 74 := 
sorry

end ashok_avg_first_five_l224_224344


namespace part1_part2_l224_224714

variable {a b c m t y1 y2 : ℝ}

-- Condition: point (2, m) lies on the parabola y = ax^2 + bx + c where axis of symmetry is x = t
def point_lies_on_parabola (a b c m : ℝ) := m = a * 2^2 + b * 2 + c

-- Condition: axis of symmetry x = t
def axis_of_symmetry (a b t : ℝ) := t = -b / (2 * a)

-- Condition: m = c
theorem part1 (a c : ℝ) (h : m = c) (h₀ : point_lies_on_parabola a (-2 * a) c m) :
  axis_of_symmetry a (-2 * a) 1 :=
by sorry

-- Additional Condition: c < m
def c_lt_m (c m : ℝ) := c < m

-- Points (-1, y1) and (3, y2) lie on the parabola y = ax^2 + bx + c
def points_on_parabola (a b c y1 y2 : ℝ) :=
  y1 = a * (-1)^2 + b * (-1) + c ∧ y2 = a * 3^2 + b * 3 + c

-- Comparison result
theorem part2 (a : ℝ) (h₁ : c_lt_m c m) (h₂ : 2 * a + (-2 * a) > 0) (h₂' : points_on_parabola a (-2 * a) c y1 y2) :
  y2 > y1 :=
by sorry

end part1_part2_l224_224714


namespace initial_total_quantity_l224_224382

theorem initial_total_quantity(milk_ratio water_ratio : ℕ) (W : ℕ) (x : ℕ) (h1 : milk_ratio = 3) (h2 : water_ratio = 1) (h3 : W = 100) (h4 : 3 * x / (x + 100) = 1 / 3) :
    4 * x = 50 :=
by
  sorry

end initial_total_quantity_l224_224382


namespace balance_balls_l224_224569

variable (G B Y W R : ℕ)

theorem balance_balls :
  (4 * G = 8 * B) →
  (3 * Y = 7 * B) →
  (8 * B = 5 * W) →
  (2 * R = 6 * B) →
  (5 * G + 3 * Y + 3 * R = 26 * B) :=
by
  intros h₁ h₂ h₃ h₄
  sorry

end balance_balls_l224_224569


namespace bus_capacity_l224_224600

def seats_available_on_left := 15
def seats_available_diff := 3
def people_per_seat := 3
def back_seat_capacity := 7

theorem bus_capacity : 
  (seats_available_on_left * people_per_seat) + 
  ((seats_available_on_left - seats_available_diff) * people_per_seat) + 
  back_seat_capacity = 88 := 
by 
  sorry

end bus_capacity_l224_224600


namespace range_of_m_l224_224102

theorem range_of_m (m : ℝ) : 0 < m ∧ m < 2 ↔ (2 - m > 0 ∧ - (1 / 2) * m < 0) := by
  sorry

end range_of_m_l224_224102


namespace moon_speed_conversion_correct_l224_224918

-- Define the conversions
def kilometers_per_second_to_miles_per_hour (kmps : ℝ) : ℝ :=
  kmps * 0.621371 * 3600

-- Condition: The moon's speed
def moon_speed_kmps : ℝ := 1.02

-- Correct answer in miles per hour
def expected_moon_speed_mph : ℝ := 2281.34

-- Theorem stating the equivalence of converted speed to expected speed
theorem moon_speed_conversion_correct :
  kilometers_per_second_to_miles_per_hour moon_speed_kmps = expected_moon_speed_mph :=
by 
  sorry

end moon_speed_conversion_correct_l224_224918


namespace g_properties_l224_224686

def f (x : ℝ) : ℝ := x

def g (x : ℝ) : ℝ := -f x

theorem g_properties :
  (∀ x : ℝ, g (-x) = -g x) ∧ (∀ x y : ℝ, x < y → g x > g y) :=
by
  sorry

end g_properties_l224_224686


namespace lcm_gcd_product_l224_224910

theorem lcm_gcd_product (a b : ℕ) (ha : a = 36) (hb : b = 60) : 
  Nat.lcm a b * Nat.gcd a b = 2160 :=
by
  rw [ha, hb]
  sorry

end lcm_gcd_product_l224_224910


namespace num_pos_multiples_of_six_is_150_l224_224829

theorem num_pos_multiples_of_six_is_150 : 
  ∃ (n : ℕ), (∀ k, (n = 150) ↔ (102 + (k - 1) * 6 = 996 ∧ 102 ≤ 6 * k ∧ 6 * k ≤ 996)) :=
sorry

end num_pos_multiples_of_six_is_150_l224_224829


namespace skitties_remainder_l224_224670

theorem skitties_remainder (m : ℕ) (h : m % 7 = 5) : (4 * m) % 7 = 6 :=
sorry

end skitties_remainder_l224_224670


namespace seunghwa_express_bus_distance_per_min_l224_224732

noncomputable def distance_per_min_on_express_bus (total_distance : ℝ) (total_time : ℝ) (time_on_general : ℝ) (gasoline_general : ℝ) (distance_per_gallon : ℝ) (gasoline_used : ℝ) : ℝ :=
  let distance_general := (gasoline_used * distance_per_gallon) / gasoline_general
  let distance_express := total_distance - distance_general
  let time_express := total_time - time_on_general
  (distance_express / time_express)

theorem seunghwa_express_bus_distance_per_min :
  distance_per_min_on_express_bus 120 110 (70) 6 (40.8) 14 = 0.62 :=
by
  sorry

end seunghwa_express_bus_distance_per_min_l224_224732


namespace sequence_properties_l224_224675

-- Definitions from conditions
def S (n : ℕ) := n^2 - n
def a (n : ℕ) := if n = 1 then 0 else 2 * (n - 1)
def b (n : ℕ) := 2^(n - 1)
def c (n : ℕ) := a n * b n
def T (n : ℕ) := (n - 2) * 2^(n + 1) + 4

-- Theorem statement proving the required identities
theorem sequence_properties {n : ℕ} (hn : n ≠ 0) :
  (a n = (if n = 1 then 0 else 2 * (n - 1))) ∧ 
  (b 2 = a 2) ∧ 
  (b 4 = a 5) ∧ 
  (T n = (n - 2) * 2^(n + 1) + 4) := by
  sorry

end sequence_properties_l224_224675


namespace distance_between_A_and_B_l224_224322

-- Definitions according to the problem's conditions
def speed_train_A : ℕ := 50
def speed_train_B : ℕ := 60
def distance_difference : ℕ := 100

-- The main theorem statement to prove
theorem distance_between_A_and_B
  (x : ℕ) -- x is the distance traveled by the first train
  (distance_train_A := x)
  (distance_train_B := x + distance_difference)
  (total_distance := distance_train_A + distance_train_B)
  (meet_condition : distance_train_A / speed_train_A = distance_train_B / speed_train_B) :
  total_distance = 1100 := 
sorry

end distance_between_A_and_B_l224_224322


namespace apricot_trees_count_l224_224591

theorem apricot_trees_count (peach_trees apricot_trees : ℕ) 
  (h1 : peach_trees = 300) 
  (h2 : peach_trees = 2 * apricot_trees + 30) : 
  apricot_trees = 135 := 
by 
  sorry

end apricot_trees_count_l224_224591


namespace vitamin_A_supplements_per_pack_l224_224865

theorem vitamin_A_supplements_per_pack {A x y : ℕ} (h1 : A * x = 119) (h2 : 17 * y = 119) : A = 7 :=
by
  sorry

end vitamin_A_supplements_per_pack_l224_224865


namespace max_sum_value_l224_224203

noncomputable def maxSum (i j k : ℤ) (h : i^2 + j^2 + k^2 = 2011) : ℤ :=
  i + j + k

theorem max_sum_value (i j k : ℤ) (h : i^2 + j^2 + k^2 = 2011) : 
  maxSum i j k h ≤ 77 :=
  sorry

end max_sum_value_l224_224203


namespace ratio_of_pats_stick_not_covered_to_sarah_stick_l224_224422

-- Defining the given conditions
def pat_stick_length : ℕ := 30
def dirt_covered : ℕ := 7
def jane_stick_length : ℕ := 22
def two_feet : ℕ := 24

-- Computing Sarah's stick length from Jane's stick length and additional two feet
def sarah_stick_length : ℕ := jane_stick_length + two_feet

-- Computing the portion of Pat's stick not covered in dirt
def portion_not_covered_in_dirt : ℕ := pat_stick_length - dirt_covered

-- The statement we need to prove
theorem ratio_of_pats_stick_not_covered_to_sarah_stick : 
  (portion_not_covered_in_dirt : ℚ) / (sarah_stick_length : ℚ) = 1 / 2 := 
by sorry

end ratio_of_pats_stick_not_covered_to_sarah_stick_l224_224422


namespace eval_expression_l224_224195

theorem eval_expression : 4 * (8 - 3) - 6 = 14 :=
by
  sorry

end eval_expression_l224_224195


namespace smallest_n_l224_224939

-- Definitions for arithmetic sequences with given conditions
def arithmetic_sequence_a (n : ℕ) (x : ℕ) : ℕ := 1 + (n-1) * x
def arithmetic_sequence_b (n : ℕ) (y : ℕ) : ℕ := 1 + (n-1) * y

-- Main theorem statement
theorem smallest_n (x y n : ℕ) (hxy : x < y) (ha1 : arithmetic_sequence_a 1 x = 1) (hb1 : arithmetic_sequence_b 1 y = 1) 
  (h_sum : arithmetic_sequence_a n x + arithmetic_sequence_b n y = 2556) : n = 3 :=
sorry

end smallest_n_l224_224939


namespace pie_slices_l224_224694

theorem pie_slices (total_pies : ℕ) (sold_pies : ℕ) (gifted_pies : ℕ) (left_pieces : ℕ) (eaten_fraction : ℚ) :
  total_pies = 4 →
  sold_pies = 1 →
  gifted_pies = 1 →
  eaten_fraction = 2/3 →
  left_pieces = 4 →
  (total_pies - sold_pies - gifted_pies) * (left_pieces * 3 / (1 - eaten_fraction)) / (total_pies - sold_pies - gifted_pies) = 6 :=
by
  sorry

end pie_slices_l224_224694


namespace initial_bottles_proof_l224_224726

-- Define the conditions as variables and statements
def initial_bottles (X : ℕ) : Prop :=
X - 8 + 45 = 51

-- Theorem stating the proof problem
theorem initial_bottles_proof : initial_bottles 14 :=
by
  -- We need to prove the following:
  -- 14 - 8 + 45 = 51
  sorry

end initial_bottles_proof_l224_224726


namespace roots_of_quadratic_function_l224_224516

variable (a b x : ℝ)

theorem roots_of_quadratic_function (h : a + b = 0) : (b * x * x + a * x = 0) → (x = 0 ∨ x = 1) :=
by {sorry}

end roots_of_quadratic_function_l224_224516


namespace cos_60_eq_one_half_l224_224637

theorem cos_60_eq_one_half : Real.cos (60 * Real.pi / 180) = 1 / 2 :=
by
  -- Proof steps would go here
  sorry

end cos_60_eq_one_half_l224_224637


namespace find_m_l224_224221

theorem find_m (m : ℝ) : 
  (∀ (x y : ℝ), (y = x + m ∧ x = 0) → y = m) ∧
  (∀ (x y : ℝ), (y = 2 * x - 2 ∧ x = 0) → y = -2) ∧
  (∀ (x : ℝ), (∃ y : ℝ, (y = x + m ∧ x = 0) ∧ (y = 2 * x - 2 ∧ x = 0))) → 
  m = -2 :=
by 
  sorry

end find_m_l224_224221


namespace prime_divides_30_l224_224979

theorem prime_divides_30 (p : ℕ) (h_prime : Prime p) (h_ge_7 : p ≥ 7) : 30 ∣ (p^2 - 1) := 
  sorry

end prime_divides_30_l224_224979


namespace problem1_problem2_problem3_problem4_l224_224053

theorem problem1 : (-20 + (-14) - (-18) - 13) = -29 := by
  sorry

theorem problem2 : (-6 * (-2) / (1 / 8)) = 96 := by
  sorry

theorem problem3 : (-24 * (-3 / 4 - 5 / 6 + 7 / 8)) = 17 := by
  sorry

theorem problem4 : (-1^4 - (1 - 0.5) * (1 / 3) * (-3)^2) = -5 / 2 := by
  sorry

end problem1_problem2_problem3_problem4_l224_224053


namespace like_terms_product_l224_224215

theorem like_terms_product :
  ∀ (m n : ℕ),
    (-x^3 * y^n) = (3 * x^m * y^2) → (m = 3 ∧ n = 2) → m * n = 6 :=
by
  intros m n h1 h2
  sorry

end like_terms_product_l224_224215


namespace arithmetic_sequence_formula_l224_224247

theorem arithmetic_sequence_formula :
  ∀ (a : ℕ → ℕ), (a 1 = 2) → (∀ n, a (n + 1) = a n + 2) → ∀ n, a n = 2 * n :=
by
  intro a
  intro h1
  intro hdiff
  sorry

end arithmetic_sequence_formula_l224_224247


namespace find_original_radius_l224_224878

theorem find_original_radius (r n : ℝ) (h : π * (r + n) ^ 2 = 3 * π * r ^ 2) :
  r = n * (Real.sqrt 3 - 2) / 2 :=
by
  sorry

end find_original_radius_l224_224878


namespace population_net_increase_l224_224241

-- Define the birth rate and death rate conditions
def birth_rate := 4 / 2 -- people per second
def death_rate := 2 / 2 -- people per second
def net_increase_per_sec := birth_rate - death_rate -- people per second

-- Define the duration of one day in seconds
def seconds_in_a_day := 24 * 3600 -- seconds

-- Define the problem to prove
theorem population_net_increase :
  net_increase_per_sec * seconds_in_a_day = 86400 :=
by
  sorry

end population_net_increase_l224_224241


namespace find_valid_n_values_l224_224142

open List

def is_median_and_mean (S : List ℝ) (m : ℝ) : Prop :=
  let sorted_S := sort (≤) S
  (sorted_S.length % 2 = 1 ∧ sorted_S.get! (sorted_S.length / 2) = m) ∧
  (sorted_S.sum / sorted_S.length = m)

theorem find_valid_n_values :
  let S := [4, 7, 11, 12]
  ∃ n : ℝ, n ∉ S ∧ 
    (is_median_and_mean (n :: S) 7 ∨
     is_median_and_mean (n :: S) 11) →
    n = 1 ∨ n = 21 ∧ (n = 1 ∨ n = 21) →
    1 + 21 = 22 :=
by
  sorry

end find_valid_n_values_l224_224142


namespace jimmy_exams_l224_224400

theorem jimmy_exams (p l a : ℕ) (h_p : p = 50) (h_l : l = 5) (h_a : a = 5) (x : ℕ) :
  (20 * x - (l + a) ≥ p) ↔ (x ≥ 3) :=
by
  sorry

end jimmy_exams_l224_224400


namespace train_crossing_platform_time_l224_224775

theorem train_crossing_platform_time (train_length : ℝ) (platform_length : ℝ) (time_cross_post : ℝ) :
  train_length = 300 → platform_length = 350 → time_cross_post = 18 → 
  (train_length + platform_length) / (train_length / time_cross_post) = 39 :=
by
  intros
  sorry

end train_crossing_platform_time_l224_224775


namespace boys_count_eq_792_l224_224446

-- Definitions of conditions
variables (B G : ℤ)

-- Total number of students is 1443
axiom total_students : B + G = 1443

-- Number of girls is 141 fewer than the number of boys
axiom girls_fewer_than_boys : G = B - 141

-- Proof statement to show that the number of boys (B) is 792
theorem boys_count_eq_792 (B G : ℤ)
  (h1 : B + G = 1443)
  (h2 : G = B - 141) : B = 792 :=
by
  sorry

end boys_count_eq_792_l224_224446


namespace perfect_squares_from_equation_l224_224071

theorem perfect_squares_from_equation (x y : ℕ) (h : 2 * x^2 + x = 3 * y^2 + y) :
  ∃ a b c : ℕ, x - y = a^2 ∧ 2 * x + 2 * y + 1 = b^2 ∧ 3 * x + 3 * y + 1 = c^2 :=
by
  sorry

end perfect_squares_from_equation_l224_224071


namespace angle_ZAX_pentagon_triangle_common_vertex_l224_224179

theorem angle_ZAX_pentagon_triangle_common_vertex :
  let n_pentagon := 5
  let n_triangle := 3
  let internal_angle_pentagon := (n_pentagon - 2) * 180 / n_pentagon
  let internal_angle_triangle := 60
  let common_angle_A := 360 - (internal_angle_pentagon + internal_angle_pentagon + internal_angle_triangle + internal_angle_triangle) / 2
  common_angle_A = 192 := by
  let n_pentagon := 5
  let n_triangle := 3
  let internal_angle_pentagon := (n_pentagon - 2) * 180 / n_pentagon
  let internal_angle_triangle := 60
  let common_angle_A := 360 - (internal_angle_pentagon + internal_angle_pentagon + internal_angle_triangle + internal_angle_triangle) / 2
  sorry

end angle_ZAX_pentagon_triangle_common_vertex_l224_224179


namespace g_of_x_l224_224699

theorem g_of_x (f g : ℕ → ℕ) (h1 : ∀ x, f x = 2 * x + 3)
  (h2 : ∀ x, g (x + 2) = f x) : ∀ x, g x = 2 * x - 1 :=
by
  sorry

end g_of_x_l224_224699


namespace num_values_passing_through_vertex_l224_224068

-- Define the parabola and line
def parabola (a : ℝ) : ℝ → ℝ := λ x, x^2 + a^2
def line (a : ℝ) : ℝ → ℝ := λ x, 2 * x + a

-- Define the vertex condition 
def passes_through_vertex (a : ℝ) : Prop :=
  parabola a 0 = line a 0

-- Prove there are exactly 2 values of a that satisfy the condition
theorem num_values_passing_through_vertex : 
  {a : ℝ | passes_through_vertex a}.finite ∧ 
  {a : ℝ | passes_through_vertex a}.toFinset.card = 2 := 
sorry

end num_values_passing_through_vertex_l224_224068


namespace pq_or_l224_224561

def p : Prop := 2 % 2 = 0
def q : Prop := 3 % 2 = 0

theorem pq_or : p ∨ q :=
by
  -- proof goes here
  sorry

end pq_or_l224_224561


namespace interest_rate_B_lent_to_C_l224_224477

noncomputable def principal : ℝ := 1500
noncomputable def rate_A : ℝ := 10
noncomputable def time : ℝ := 3
noncomputable def gain_B : ℝ := 67.5
noncomputable def interest_paid_by_B_to_A : ℝ := principal * rate_A * time / 100
noncomputable def interest_received_by_B_from_C : ℝ := interest_paid_by_B_to_A + gain_B
noncomputable def expected_rate : ℝ := 11.5

theorem interest_rate_B_lent_to_C :
  interest_received_by_B_from_C = principal * (expected_rate) * time / 100 := 
by
  -- the proof will go here
  sorry

end interest_rate_B_lent_to_C_l224_224477


namespace probability_of_at_least_one_black_ball_l224_224605

noncomputable def probability_at_least_one_black_ball := 
  let total_outcomes := Nat.choose 4 2
  let favorable_outcomes := (Nat.choose 2 1) * (Nat.choose 2 1) + (Nat.choose 2 2)
  favorable_outcomes / total_outcomes

theorem probability_of_at_least_one_black_ball :
  probability_at_least_one_black_ball = 5 / 6 :=
by
  sorry

end probability_of_at_least_one_black_ball_l224_224605


namespace sector_max_area_l224_224457

theorem sector_max_area (r : ℝ) (α : ℝ) (S : ℝ) :
  (0 < r ∧ r < 10) ∧ (2 * r + r * α = 20) ∧ (S = (1 / 2) * r * (r * α)) →
  (α = 2 ∧ S = 25) :=
by
  sorry

end sector_max_area_l224_224457


namespace correct_option_D_l224_224913

theorem correct_option_D : (-8) / (-4) = 8 / 4 := 
by
  exact (rfl

end correct_option_D_l224_224913


namespace ellipse_eccentricity_l224_224954

theorem ellipse_eccentricity (a b c : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : b = (4/3) * c) (h4 : a^2 - b^2 = c^2) : 
  c / a = 3 / 5 :=
by
  sorry

end ellipse_eccentricity_l224_224954


namespace sales_on_third_day_l224_224171

variable (a m : ℕ)

def first_day_sales : ℕ := a
def second_day_sales : ℕ := 3 * a - 3 * m
def third_day_sales : ℕ := (3 * a - 3 * m) + m

theorem sales_on_third_day 
  (a m : ℕ) : third_day_sales a m = 3 * a - 2 * m :=
by
  -- Assuming the conditions as our definitions:
  let fds := first_day_sales a
  let sds := second_day_sales a m
  let tds := third_day_sales a m

  -- Proof direction:
  show tds = 3 * a - 2 * m
  sorry

end sales_on_third_day_l224_224171


namespace number_of_men_l224_224983

variable (M : ℕ)

-- Define the first condition: M men reaping 80 hectares in 24 days.
def first_work_rate (M : ℕ) : ℚ := (80 : ℚ) / (M * 24)

-- Define the second condition: 36 men reaping 360 hectares in 30 days.
def second_work_rate : ℚ := (360 : ℚ) / (36 * 30)

-- Lean 4 statement: Prove the equivalence given conditions.
theorem number_of_men (h : first_work_rate M = second_work_rate) : M = 45 :=
by
  sorry

end number_of_men_l224_224983


namespace common_difference_of_arithmetic_sequence_l224_224701

noncomputable def smallest_angle : ℝ := 25
noncomputable def largest_angle : ℝ := 105
noncomputable def num_angles : ℕ := 5

theorem common_difference_of_arithmetic_sequence :
  ∃ d : ℝ, (smallest_angle + (num_angles - 1) * d = largest_angle) ∧ d = 20 :=
by
  sorry

end common_difference_of_arithmetic_sequence_l224_224701


namespace locus_midpoint_l224_224684

/-- Given a fixed point A (4, -2) and a moving point B on the curve x^2 + y^2 = 4,
    prove that the locus of the midpoint P of the line segment AB satisfies the equation 
    (x - 2)^2 + (y + 1)^2 = 1. -/
theorem locus_midpoint (A B P : ℝ × ℝ)
  (hA : A = (4, -2))
  (hB : ∃ (x y : ℝ), B = (x, y) ∧ x^2 + y^2 = 4)
  (hP : P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) :
  (P.1 - 2)^2 + (P.2 + 1)^2 = 1 :=
sorry

end locus_midpoint_l224_224684


namespace triangle_area_l224_224197

noncomputable def vector (x y z : ℝ) : ℝ × ℝ × ℝ :=
(x, y, z)

noncomputable def cross_product (v w : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
(v.2.1 * w.2.2 - v.2.2 * w.2.1,
 v.2.2 * w.1 - v.1 * w.2.2,
 v.1 * w.2.1 - v.2.1 * w.1)

noncomputable def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
Real.sqrt (v.1^2 + v.2.1^2 + v.2.2^2)

theorem triangle_area :
  let A := vector 2 1 (-1)
  let B := vector 3 0 3
  let C := vector 7 3 2
  let AB := (B.1 - A.1, B.2.1 - A.2.1, B.2.2 - A.2.2)
  let AC := (C.1 - A.1, C.2.1 - A.2.1, C.2.2 - A.2.2)
  0.5 * magnitude (cross_product AB AC) = (1 / 2) * Real.sqrt 459 :=
by
  -- All the steps needed to prove the theorem here
  sorry

end triangle_area_l224_224197


namespace similar_triangle_legs_l224_224480

theorem similar_triangle_legs {y : ℝ} 
  (h₁ : 12 / y = 9 / 6) : y = 8 :=
sorry

end similar_triangle_legs_l224_224480


namespace incenter_divides_angle_bisector_2_1_l224_224211

def is_incenter_divide_angle_bisector (AB BC AC : ℝ) (O : ℝ) : Prop :=
  AB = 15 ∧ BC = 12 ∧ AC = 18 → O = 2 / 1

theorem incenter_divides_angle_bisector_2_1 :
  is_incenter_divide_angle_bisector 15 12 18 (2 / 1) :=
by
  sorry

end incenter_divides_angle_bisector_2_1_l224_224211


namespace total_cost_for_doughnuts_l224_224489

theorem total_cost_for_doughnuts
  (num_students : ℕ)
  (num_chocolate : ℕ)
  (num_glazed : ℕ)
  (price_chocolate : ℕ)
  (price_glazed : ℕ)
  (H1 : num_students = 25)
  (H2 : num_chocolate = 10)
  (H3 : num_glazed = 15)
  (H4 : price_chocolate = 2)
  (H5 : price_glazed = 1) :
  num_chocolate * price_chocolate + num_glazed * price_glazed = 35 :=
by
  -- Proof steps would go here
  sorry

end total_cost_for_doughnuts_l224_224489


namespace sum_in_base5_correct_l224_224587

-- Defining the integers
def num1 : ℕ := 210
def num2 : ℕ := 72

-- Summing the integers
def sum : ℕ := num1 + num2

-- Converting the resulting sum to base 5
def to_base5 (n : ℕ) : String :=
  let rec aux (n : ℕ) (acc : List Char) : List Char :=
    if n < 5 then Char.ofNat (n + 48) :: acc
    else aux (n / 5) (Char.ofNat (n % 5 + 48) :: acc)
  String.mk (aux n [])

-- The expected sum in base 5
def expected_sum_base5 : String := "2062"

-- The Lean theorem to be proven
theorem sum_in_base5_correct : to_base5 sum = expected_sum_base5 :=
by
  sorry

end sum_in_base5_correct_l224_224587


namespace intersection_of_sets_l224_224689

-- Define the sets M and N
def M : Set ℝ := { x | 2 < x ∧ x < 3 }
def N : Set ℝ := { x | 2 < x ∧ x ≤ 5 / 2 }

-- State the theorem to prove
theorem intersection_of_sets : M ∩ N = { x | 2 < x ∧ x ≤ 5 / 2 } :=
by 
  sorry

end intersection_of_sets_l224_224689


namespace monotonicity_f_l224_224499

open Set

noncomputable def f (a x : ℝ) : ℝ := a * x / (x - 1)

theorem monotonicity_f (a : ℝ) (h : a ≠ 0) :
  (∀ x1 x2 : ℝ, -1 < x1 → x1 < x2 → x2 < 1 → (if a > 0 then f a x1 > f a x2 else if a < 0 then f a x1 < f a x2 else False)) :=
by
  sorry

end monotonicity_f_l224_224499


namespace leftover_grass_seed_coverage_l224_224808

/-
Question: How many extra square feet could the leftover grass seed cover after Drew reseeds his lawn?

Conditions:
- One bag of grass seed covers 420 square feet of lawn.
- The lawn consists of a rectangular section and a triangular section.
- Rectangular section:
    - Length: 32 feet
    - Width: 45 feet
- Triangular section:
    - Base: 25 feet
    - Height: 20 feet
- Triangular section requires 1.5 times the standard coverage rate.
- Drew bought seven bags of seed.

Answer: The leftover grass seed coverage is 1125 square feet.
-/

theorem leftover_grass_seed_coverage
  (bag_coverage : ℕ := 420)
  (rect_length : ℕ := 32)
  (rect_width : ℕ := 45)
  (tri_base : ℕ := 25)
  (tri_height : ℕ := 20)
  (coverage_multiplier : ℕ := 15)  -- Using 15 instead of 1.5 for integer math
  (bags_bought : ℕ := 7) :
  (bags_bought * bag_coverage - 
    (rect_length * rect_width + tri_base * tri_height * coverage_multiplier / 20) = 1125) :=
  by {
    -- Placeholder for proof steps
    sorry
  }

end leftover_grass_seed_coverage_l224_224808


namespace fourth_number_ninth_row_eq_206_l224_224141

-- Define the first number in a given row
def first_number_in_row (i : Nat) : Nat :=
  2 + 4 * 6 * (i - 1)

-- Define the number in the j-th position in the i-th row
def number_in_row (i j : Nat) : Nat :=
  first_number_in_row i + 4 * (j - 1)

-- Define the 9th row and fourth number in it
def fourth_number_ninth_row : Nat :=
  number_in_row 9 4

-- The theorem to prove the fourth number in the 9th row is 206
theorem fourth_number_ninth_row_eq_206 : fourth_number_ninth_row = 206 := by
  sorry

end fourth_number_ninth_row_eq_206_l224_224141


namespace solution_set_of_quadratic_inequality_l224_224001

theorem solution_set_of_quadratic_inequality :
  {x : ℝ | x^2 + 2 * x - 3 < 0} = {x : ℝ | -3 < x ∧ x < 1} :=
sorry

end solution_set_of_quadratic_inequality_l224_224001


namespace product_inequality_l224_224125

theorem product_inequality (n : ℕ) (h : n > 1) : 
  ∏ (k : ℕ) in finset.range (n + 1), k^k < n^(n * (n + 1) / 2) := 
begin
  sorry
end

end product_inequality_l224_224125


namespace graph_properties_l224_224852

theorem graph_properties (k b : ℝ) (hk : k ≠ 0) (hb : b ≠ 0) (positive_kb : k * b > 0) :
  (∃ (f g : ℝ → ℝ),
    (∀ x, f x = k * x + b) ∧
    (∀ x (hx : x ≠ 0), g x = k * b / x) ∧
    -- Under the given conditions, the graphs must match option (B)
    (True)) := sorry

end graph_properties_l224_224852


namespace scheduling_courses_in_non_consecutive_periods_l224_224086

theorem scheduling_courses_in_non_consecutive_periods :
  (∃ (n m : ℕ), n = 4 ∧ m = 7 ∧ 
   (∑ v in (finset.filter (λ s, s.card = n ∧ ∀ x y ∈ s, x ≠ y ∧ abs (x - y) ≠ 1) (finset.powerset (finset.range m))), 
   finset.prod v (λ _, 1)) * nat.factorial n = 96) :=
sorry

end scheduling_courses_in_non_consecutive_periods_l224_224086


namespace oa_dot_ob_eq_neg2_l224_224674

/-!
# Problem Statement
Given AB as the diameter of the smallest radius circle centered at C(0,1) that intersects 
the graph of y = 1 / (|x| - 1), where O is the origin. Prove that the dot product 
\overrightarrow{OA} · \overrightarrow{OB} equals -2.
-/

noncomputable def smallest_radius_circle_eqn (x : ℝ) : ℝ :=
  x^2 + ((1 / (|x| - 1)) - 1)^2

noncomputable def radius_of_circle (x : ℝ) : ℝ :=
  Real.sqrt (smallest_radius_circle_eqn x)

noncomputable def OA (x : ℝ) : ℝ × ℝ :=
  (x, (1 / (|x| - 1)) + 1)

noncomputable def OB (x : ℝ) : ℝ × ℝ :=
  (-x, 1 - (1 / (|x| - 1)))

noncomputable def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

theorem oa_dot_ob_eq_neg2 (x : ℝ) (hx : |x| > 1) :
  let a := OA x
  let b := OB x
  dot_product a b = -2 :=
by
  sorry

end oa_dot_ob_eq_neg2_l224_224674


namespace arithmetic_sequence_common_difference_l224_224848

variable {a : ℕ → ℝ}
variable {d : ℝ}

-- Statement of the problem
theorem arithmetic_sequence_common_difference
  (h1 : a 2 + a 6 = 8)
  (h2 : a 3 + a 4 = 3)
  (h_arith : ∀ n, a (n+1) = a n + d) :
  d = 5 := by
  sorry

end arithmetic_sequence_common_difference_l224_224848


namespace negation_equivalence_l224_224885

-- Define the proposition P stating 'there exists an x in ℝ such that x^2 - 2x + 4 > 0'
def P : Prop := ∃ x : ℝ, x^2 - 2*x + 4 > 0

-- Define the proposition Q which is the negation of proposition P
def Q : Prop := ∀ x : ℝ, x^2 - 2*x + 4 ≤ 0

-- State the proof problem: Prove that the negation of proposition P is equivalent to proposition Q
theorem negation_equivalence : ¬ P ↔ Q := by
  -- Proof to be provided.
  sorry

end negation_equivalence_l224_224885


namespace divisibility_problem_l224_224857

theorem divisibility_problem (n : ℕ) : n-1 ∣ n^n - 7*n + 5*n^2024 + 3*n^2 - 2 := 
by
  sorry

end divisibility_problem_l224_224857


namespace inequality_proof_l224_224323

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a * b * c = 1/8) :
  a^2 + b^2 + c^2 + a^2 * b^2 + b^2 * c^2 + c^2 * a^2 ≥ 15 / 16 :=
by sorry

end inequality_proof_l224_224323


namespace ratio_of_larger_to_smaller_l224_224898

theorem ratio_of_larger_to_smaller (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x > y) (h4 : x + y = 7 * (x - y)) : x / y = 4 / 3 := by
  sorry

end ratio_of_larger_to_smaller_l224_224898


namespace ellipse_area_constant_l224_224520

def ellipse_equation (a b : ℝ) (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

def passes_through (x_a y_a x_b y_b : ℝ) (p : ℝ × ℝ) : Prop := 
  p.1 = x_a ∧ p.2 = y_a ∨ p.1 = x_b ∧ p.2 = y_b

def area_ABNM_constant (x y : ℝ) : Prop :=
  let x_0 := x;
  let y_0 := y;
  let y_M := -2 * y_0 / (x_0 - 2);
  let BM := 1 + 2 * y_0 / (x_0 - 2);
  let x_N := - x_0 / (y_0 - 1);
  let AN := 2 + x_0 / (y_0 - 1);
  (1 / 2) * AN * BM = 2

theorem ellipse_area_constant :
  ∀ (a b : ℝ), (a = 2 ∧ b = 1) → 
  (∀ (x y : ℝ), 
    ellipse_equation a b x y → 
    passes_through 2 0 0 1 (x, y) → 
    (x < 0 ∧ y < 0) →
    area_ABNM_constant x y) :=
by
  intros
  sorry

end ellipse_area_constant_l224_224520


namespace graph_intersection_points_l224_224875

open Function

theorem graph_intersection_points (g : ℝ → ℝ) (h_inv : Involutive (invFun g)) : 
  ∃! (x : ℝ), x = 0 ∨ x = 1 ∨ x = -1 → g (x^2) = g (x^6) :=
by sorry

end graph_intersection_points_l224_224875


namespace l_shaped_area_l224_224671

theorem l_shaped_area (A B C D : Type) (side_abcd: ℝ) (side_small_1: ℝ) (side_small_2: ℝ)
  (area_abcd : side_abcd = 6)
  (area_small_1 : side_small_1 = 2)
  (area_small_2 : side_small_2 = 4)
  (no_overlap : true) :
  side_abcd * side_abcd - (side_small_1 * side_small_1 + side_small_2 * side_small_2) = 16 := by
  sorry

end l_shaped_area_l224_224671


namespace number_of_pens_bought_l224_224608

theorem number_of_pens_bought 
  (P : ℝ) -- Marked price of one pen
  (N : ℝ) -- Number of pens bought
  (discount : ℝ := 0.01)
  (profit_percent : ℝ := 29.130434782608695)
  (Total_Cost := 46 * P)
  (Selling_Price_per_Pen := P * (1 - discount))
  (Total_Revenue := N * Selling_Price_per_Pen)
  (Profit := Total_Revenue - Total_Cost)
  (actual_profit_percent := (Profit / Total_Cost) * 100) :
  actual_profit_percent = profit_percent → N = 60 := 
by 
  intro h
  sorry

end number_of_pens_bought_l224_224608


namespace total_fencing_cost_l224_224440

theorem total_fencing_cost
  (length : ℝ) 
  (breadth : ℝ)
  (cost_per_meter : ℝ)
  (h1 : length = 61)
  (h2 : length = breadth + 22)
  (h3 : cost_per_meter = 26.50) : 
  2 * (length + breadth) * cost_per_meter = 5300 := 
by 
  sorry

end total_fencing_cost_l224_224440


namespace smallest_possible_obscured_number_l224_224794

theorem smallest_possible_obscured_number (a b : ℕ) (cond : 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9) :
  2 * a = b - 9 →
  42 + 25 + 56 + 10 * a + b = 4 * (4 + 2 + 2 + 5 + 5 + 6 + a + b) →
  10 * a + b = 79 :=
sorry

end smallest_possible_obscured_number_l224_224794


namespace cos_60_degrees_is_one_half_l224_224635

-- Define the given conditions in Lean 4
def angle_60_degrees : Real :=
  60 * Real.pi / 180

def unit_circle_point (θ : Real) : Real × Real :=
  (Real.cos θ, Real.sin θ)

-- State the proof problem in Lean 4
theorem cos_60_degrees_is_one_half : 
  (unit_circle_point angle_60_degrees).1 = 1 / 2 := 
sorry

end cos_60_degrees_is_one_half_l224_224635


namespace restock_quantities_correct_l224_224173

-- Definition for the quantities of cans required
def cans_peas : ℕ := 810
def cans_carrots : ℕ := 954
def cans_corn : ℕ := 675

-- Definition for the number of cans per box, pack, and case.
def cans_per_box_peas : ℕ := 4
def cans_per_pack_carrots : ℕ := 6
def cans_per_case_corn : ℕ := 5

-- Define the expected order quantities.
def order_boxes_peas : ℕ := 203
def order_packs_carrots : ℕ := 159
def order_cases_corn : ℕ := 135

-- Proof statement for the quantities required to restock exactly.
theorem restock_quantities_correct :
  (order_boxes_peas = Nat.ceil (cans_peas / cans_per_box_peas))
  ∧ (order_packs_carrots = cans_carrots / cans_per_pack_carrots)
  ∧ (order_cases_corn = cans_corn / cans_per_case_corn) :=
by
  sorry

end restock_quantities_correct_l224_224173


namespace bond_face_value_l224_224999

theorem bond_face_value
  (F : ℝ)
  (S : ℝ)
  (hS : S = 3846.153846153846)
  (hI1 : I = 0.05 * F)
  (hI2 : I = 0.065 * S) :
  F = 5000 :=
by
  sorry

end bond_face_value_l224_224999


namespace total_value_of_coins_l224_224976

theorem total_value_of_coins (num_quarters num_nickels : ℕ) (val_quarter val_nickel : ℝ)
  (h_quarters : num_quarters = 8) (h_nickels : num_nickels = 13)
  (h_total_coins : num_quarters + num_nickels = 21) (h_val_quarter : val_quarter = 0.25)
  (h_val_nickel : val_nickel = 0.05) :
  num_quarters * val_quarter + num_nickels * val_nickel = 2.65 := 
sorry

end total_value_of_coins_l224_224976


namespace regular_polygon_sides_l224_224657

theorem regular_polygon_sides (n : ℕ) (h : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l224_224657


namespace problem_100th_term_of_seq_l224_224934

theorem problem_100th_term_of_seq (f : ℕ → ℕ) (n : ℕ) :
  (∀ n, ∃ k : ℕ, f n = k) ∧ (∀ m < n, f m < f n) →
  f 100 = 981 :=
by
  sorry

end problem_100th_term_of_seq_l224_224934


namespace cake_piece_volume_l224_224182

theorem cake_piece_volume (h : ℝ) (d : ℝ) (n : ℕ) (V_piece : ℝ) : 
  h = 1/2 ∧ d = 16 ∧ n = 8 → V_piece = 4 * Real.pi :=
by
  sorry

end cake_piece_volume_l224_224182


namespace cost_per_ton_ice_correct_l224_224484

variables {a p n s : ℝ}

-- Define the cost per ton of ice received by enterprise A
noncomputable def cost_per_ton_ice_received (a p n s : ℝ) : ℝ :=
  (2.5 * a + p * s) * 1000 / (2000 - n * s)

-- The statement of the theorem
theorem cost_per_ton_ice_correct :
  ∀ a p n s : ℝ,
  2000 - n * s ≠ 0 →
  cost_per_ton_ice_received a p n s = (2.5 * a + p * s) * 1000 / (2000 - n * s) := by
  intros a p n s h
  unfold cost_per_ton_ice_received
  sorry

end cost_per_ton_ice_correct_l224_224484


namespace subtraction_of_negatives_l224_224800

theorem subtraction_of_negatives :
  -2 - (-3) = 1 := 
by
  sorry

end subtraction_of_negatives_l224_224800


namespace age_difference_l224_224026

variable (a b c : ℕ)

theorem age_difference (h : a + b = b + c + 13) : a - c = 13 :=
by
  sorry

end age_difference_l224_224026


namespace liam_homework_probability_l224_224888

theorem liam_homework_probability:
  let p_complete := 5 / 9
  let p_not_complete := 1 - p_complete
  in p_not_complete = 4 / 9 :=
by
  -- steps to prove the theorem
  sorry

end liam_homework_probability_l224_224888


namespace find_m_l224_224219

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x + m
noncomputable def g (x : ℝ) : ℝ := 2 * x - 2

theorem find_m : 
  ∃ m : ℝ, ∀ x : ℝ, f m x = g x → m = -2 := by
  sorry

end find_m_l224_224219


namespace tournament_teams_matches_l224_224296

theorem tournament_teams_matches (teams : Fin 10 → ℕ) 
  (h : ∀ i, teams i ≤ 9) : 
  ∃ i j : Fin 10, i ≠ j ∧ teams i = teams j := 
by 
  sorry

end tournament_teams_matches_l224_224296


namespace final_lights_on_l224_224315

def lights_on_by_children : ℕ :=
  let total_lights := 200
  let flips_x := total_lights / 7
  let flips_y := total_lights / 11
  let lcm_xy := 77  -- since lcm(7, 11) = 7 * 11 = 77
  let flips_both := total_lights / lcm_xy
  flips_x + flips_y - flips_both

theorem final_lights_on : lights_on_by_children = 44 :=
by
  sorry

end final_lights_on_l224_224315


namespace new_rectangle_area_l224_224302

theorem new_rectangle_area (L W : ℝ) (h : L * W = 300) :
  let L_new := 2 * L
  let W_new := 3 * W
  L_new * W_new = 1800 :=
by
  let L_new := 2 * L
  let W_new := 3 * W
  sorry

end new_rectangle_area_l224_224302


namespace probability_of_black_ball_l224_224170

theorem probability_of_black_ball (P_red P_white : ℝ) (h_red : P_red = 0.43) (h_white : P_white = 0.27) : 
  (1 - P_red - P_white) = 0.3 := 
by
  sorry

end probability_of_black_ball_l224_224170


namespace father_present_age_l224_224166

theorem father_present_age (S F : ℕ) 
  (h1 : F = 3 * S + 3) 
  (h2 : F + 3 = 2 * (S + 3) + 10) : 
  F = 33 :=
by
  sorry

end father_present_age_l224_224166


namespace coffee_processing_completed_l224_224798

-- Define the initial conditions
def CoffeeBeansProcessed (m n : ℕ) : Prop :=
  let mass: ℝ := 1
  let days_single_machine: ℕ := 5
  let days_both_machines: ℕ := 4
  let half_mass: ℝ := mass / 2
  let total_ground_by_June_10 := (days_single_machine * m + days_both_machines * (m + n)) = half_mass
  total_ground_by_June_10

-- Define the final proof problem
theorem coffee_processing_completed (m n : ℕ) (h: CoffeeBeansProcessed m n) : ∃ d : ℕ, d = 15 := by
  -- Processed in 15 working days
  sorry

end coffee_processing_completed_l224_224798


namespace power_calculation_l224_224622

theorem power_calculation :
  ((8^5 / 8^3) * 4^6) = 262144 := by
  sorry

end power_calculation_l224_224622


namespace min_mn_value_l224_224072

theorem min_mn_value (m n : ℕ) (hmn : m > n) (hn : n ≥ 1) 
  (hdiv : 1000 ∣ 1978 ^ m - 1978 ^ n) : m + n = 106 :=
sorry

end min_mn_value_l224_224072


namespace sum_of_even_sequence_is_194_l224_224076

theorem sum_of_even_sequence_is_194
  (a b c d : ℕ) 
  (even_a : a % 2 = 0) 
  (even_b : b % 2 = 0) 
  (even_c : c % 2 = 0) 
  (even_d : d % 2 = 0)
  (a_lt_b : a < b) 
  (b_lt_c : b < c) 
  (c_lt_d : c < d)
  (diff_da : d - a = 90)
  (arith_ab_c : 2 * b = a + c)
  (geo_bc_d : c^2 = b * d)
  : a + b + c + d = 194 := 
sorry

end sum_of_even_sequence_is_194_l224_224076


namespace kristin_reading_time_l224_224423

-- Definitions
def total_books : Nat := 20
def peter_time_per_book : ℕ := 18
def reading_speed_ratio : Nat := 3

-- Derived Definitions
def kristin_time_per_book : ℕ := peter_time_per_book * reading_speed_ratio
def kristin_books_to_read : Nat := total_books / 2
def kristin_total_time : ℕ := kristin_time_per_book * kristin_books_to_read

-- Statement to be proved
theorem kristin_reading_time :
  kristin_total_time = 540 :=
  by 
    -- Proof would go here, but we are only required to state the theorem
    sorry

end kristin_reading_time_l224_224423


namespace sum_leq_six_of_quadratic_roots_l224_224524

theorem sum_leq_six_of_quadratic_roots (a b : ℤ) (h1 : a ≠ -1) (h2 : b ≠ -1) 
  (h3 : ∃ r1 r2 : ℤ, r1 ≠ r2 ∧ x^2 + ab * x + (a + b) = 0 ∧ 
         x = r1 ∧ x = r2) : a + b ≤ 6 :=
by
  sorry

end sum_leq_six_of_quadratic_roots_l224_224524


namespace cost_price_equals_selling_price_l224_224582

theorem cost_price_equals_selling_price (C : ℝ) (x : ℝ) (hp : C > 0) (profit : ℝ := 0.25) (h : 30 * C = (1 + profit) * C * x) : x = 24 :=
by
  sorry

end cost_price_equals_selling_price_l224_224582


namespace coloring_equilateral_triangle_l224_224307

theorem coloring_equilateral_triangle :
  ∀ (A B C : Type) (color : A → Type) (d : A → A → ℝ),
  (∀ x y, d x y = 1 → color x = color y) :=
by sorry

end coloring_equilateral_triangle_l224_224307


namespace ratio_solution_l224_224610

theorem ratio_solution (x : ℚ) : (1 : ℚ) / 3 = 5 / 3 / x → x = 5 := 
by
  intro h
  sorry

end ratio_solution_l224_224610


namespace num_nat_numbers_divisible_by_7_between_100_and_250_l224_224376

noncomputable def countNatNumbersDivisibleBy7InRange : ℕ :=
  let smallest := Nat.ceil (100 / 7) * 7
  let largest := Nat.floor (250 / 7) * 7
  (largest - smallest) / 7 + 1

theorem num_nat_numbers_divisible_by_7_between_100_and_250 :
  countNatNumbersDivisibleBy7InRange = 21 :=
by
  -- Placeholder for the proof steps
  sorry

end num_nat_numbers_divisible_by_7_between_100_and_250_l224_224376


namespace tan_x_eq_2_solution_l224_224143

noncomputable def solution_set_tan_2 : Set ℝ := {x | ∃ k : ℤ, x = k * Real.pi + Real.arctan 2}

theorem tan_x_eq_2_solution :
  {x : ℝ | Real.tan x = 2} = solution_set_tan_2 :=
by
  sorry

end tan_x_eq_2_solution_l224_224143


namespace angle_between_diagonals_l224_224744

open Real

theorem angle_between_diagonals
  (a b c : ℝ) :
  ∃ θ : ℝ, θ = arccos (a^2 / sqrt ((a^2 + b^2) * (a^2 + c^2))) :=
by
  -- Placeholder for the proof
  sorry

end angle_between_diagonals_l224_224744


namespace remainder_of_1999_pow_81_mod_7_eq_1_l224_224156

/-- 
  Prove the remainder R when 1999^81 is divided by 7 is equal to 1.
  Conditions:
  - number: 1999
  - divisor: 7
-/
theorem remainder_of_1999_pow_81_mod_7_eq_1 : (1999 ^ 81) % 7 = 1 := 
by 
  sorry

end remainder_of_1999_pow_81_mod_7_eq_1_l224_224156


namespace smallest_y_l224_224020

theorem smallest_y (y : ℕ) :
  (y > 0 ∧ 800 ∣ (540 * y)) ↔ (y = 40) :=
by
  sorry

end smallest_y_l224_224020


namespace broccoli_sales_l224_224564

theorem broccoli_sales (B C S Ca : ℝ) (h1 : C = 2 * B) (h2 : S = B / 2 + 16) (h3 : Ca = 136) (total_sales : B + C + S + Ca = 380) :
  B = 57 :=
by
  sorry

end broccoli_sales_l224_224564


namespace mail_distribution_l224_224785

def pieces_per_block (total_pieces blocks : ℕ) : ℕ := total_pieces / blocks

theorem mail_distribution : pieces_per_block 192 4 = 48 := 
by { 
    -- Proof skipped
    sorry 
}

end mail_distribution_l224_224785


namespace schedule_courses_l224_224696

/-- Definition of valid schedule count where at most one pair of courses is consecutive. -/
def count_valid_schedules : ℕ := 180

/-- Given 7 periods and 3 courses, determine the number of valid schedules 
    where at most one pair of these courses is consecutive. -/
theorem schedule_courses (periods : ℕ) (courses : ℕ) (valid_schedules : ℕ) :
  periods = 7 → courses = 3 → valid_schedules = count_valid_schedules →
  valid_schedules = 180 :=
by
  intros h1 h2 h3
  sorry

end schedule_courses_l224_224696


namespace admin_staff_in_sample_l224_224040

theorem admin_staff_in_sample (total_staff : ℕ) (admin_staff : ℕ) (total_samples : ℕ)
  (probability : ℚ) (h1 : total_staff = 200) (h2 : admin_staff = 24)
  (h3 : total_samples = 50) (h4 : probability = 50 / 200) :
  admin_staff * probability = 6 :=
by
  -- Proof goes here
  sorry

end admin_staff_in_sample_l224_224040


namespace animal_shelter_dogs_l224_224984

theorem animal_shelter_dogs (D C R : ℕ) 
  (h₁ : 15 * C = 7 * D)
  (h₂ : 15 * R = 4 * D)
  (h₃ : 15 * (C + 20) = 11 * D)
  (h₄ : 15 * (R + 10) = 6 * D) : 
  D = 75 :=
by
  -- Proof part is omitted
  sorry

end animal_shelter_dogs_l224_224984


namespace average_value_of_powers_l224_224810

theorem average_value_of_powers (z : ℝ) : 
  (z^2 + 3*z^2 + 6*z^2 + 12*z^2 + 24*z^2) / 5 = 46*z^2 / 5 :=
by
  sorry

end average_value_of_powers_l224_224810


namespace remaining_distance_l224_224453

theorem remaining_distance (S u : ℝ) (h1 : S / (2 * u) + 24 = S) (h2 : S * u / 2 + 15 = S) : ∃ x : ℝ, x = 8 :=
by
  -- Proof steps would go here
  sorry

end remaining_distance_l224_224453


namespace base_video_card_cost_l224_224997

theorem base_video_card_cost
    (cost_computer : ℕ)
    (fraction_monitor_peripherals : ℕ → ℕ → ℕ)
    (twice : ℕ → ℕ)
    (total_spent : ℕ)
    (cost_monitor_peripherals_eq : fraction_monitor_peripherals cost_computer 5 = 300)
    (twice_eq : ∀ x, twice x = 2 * x)
    (eq_total : ∀ (base_video_card : ℕ), cost_computer + fraction_monitor_peripherals cost_computer 5 + twice base_video_card = total_spent)
    : ∃ x, total_spent = 2100 ∧ cost_computer = 1500 ∧ x = 150 :=
by
  sorry

end base_video_card_cost_l224_224997


namespace quadrilateral_is_rhombus_l224_224303

theorem quadrilateral_is_rhombus (a b c d : ℝ)
  (h : a^2 + b^2 + c^2 + d^2 = ab + bc + cd + ad) : a = b ∧ b = c ∧ c = d :=
by
  sorry

end quadrilateral_is_rhombus_l224_224303


namespace problem_l224_224073

theorem problem (α : ℝ) (h : Real.tan α = 2) :
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) + Real.cos α^2 = 16 / 5 :=
sorry

end problem_l224_224073


namespace nat_divides_power_difference_l224_224259

theorem nat_divides_power_difference (n : ℕ) : n ∣ 2 ^ (2 * n.factorial) - 2 ^ n.factorial := by
  sorry

end nat_divides_power_difference_l224_224259


namespace cos_of_60_degrees_is_half_l224_224627

noncomputable theory

def unit_circle_coordinates := (1/2 : ℝ, real.sqrt 3 / 2)

theorem cos_of_60_degrees_is_half :
  ∀ θ : ℝ, θ = 60 → unit_circle_coordinates.fst = real.cos θ :=
begin
  sorry
end

end cos_of_60_degrees_is_half_l224_224627


namespace correct_statement_l224_224160

/-- Given the following statements:
 1. Seeing a rainbow after rain is a random event.
 2. To check the various equipment before a plane takes off, a random sampling survey should be conducted.
 3. When flipping a coin 20 times, it will definitely land heads up 10 times.
 4. The average precipitation in two cities, A and B, over the past 5 years is 800 millimeters. The variances are 3.4 for city A and 4.3 for city B. The city with the most stable annual precipitation is city B.

 Prove that the correct statement is: Seeing a rainbow after rain is a random event.
-/
theorem correct_statement : 
  let statement_A := "Seeing a rainbow after rain is a random event"
  let statement_B := "To check the various equipment before a plane takes off, a random sampling survey should be conducted"
  let statement_C := "When flipping a coin 20 times, it will definitely land heads up 10 times"
  let statement_D := "The average precipitation in two cities, A and B, over the past 5 years is 800 millimeters. The variances are 3.4 for city A and 4.3 for city B. The city with the most stable annual precipitation is city B"
  statement_A = "Seeing a rainbow after rain is a random event" := by
sorry

end correct_statement_l224_224160


namespace solve_for_x_l224_224067

theorem solve_for_x :
  ∃ x : ℝ, ((17.28 / x) / (3.6 * 0.2)) = 2 ∧ x = 12 :=
by
  sorry

end solve_for_x_l224_224067


namespace work_completion_days_l224_224482

-- Definitions based on the conditions
def A_work_days : ℕ := 20
def B_work_days : ℕ := 30
def C_work_days : ℕ := 10  -- Twice as fast as A, and A can do it in 20 days, hence 10 days.
def together_work_days : ℕ := 12
def B_C_half_day_rate : ℚ := (1 / B_work_days) / 2 + (1 / C_work_days) / 2  -- rate per half day for both B and C
def A_full_day_rate : ℚ := 1 / A_work_days  -- rate per full day for A

-- Converting to rate per day when B and C work only half day daily
def combined_rate_per_day_with_BC_half : ℚ := A_full_day_rate + B_C_half_day_rate

-- The main theorem to prove
theorem work_completion_days 
  (A_work_days B_work_days C_work_days together_work_days : ℕ)
  (C_work_days_def : C_work_days = A_work_days / 2) 
  (total_days_def : 1 / combined_rate_per_day_with_BC_half = 60 / 7) :
  (1 / combined_rate_per_day_with_BC_half) = 60 / 7 :=
sorry

end work_completion_days_l224_224482


namespace three_digit_numbers_divisible_by_5_l224_224974

theorem three_digit_numbers_divisible_by_5 : ∃ n : ℕ, n = 181 ∧ ∀ x : ℕ, (100 ≤ x ∧ x ≤ 999 ∧ x % 5 = 0) → ∃ k : ℕ, x = 100 + k * 5 ∧ k < n := sorry

end three_digit_numbers_divisible_by_5_l224_224974


namespace train_speed_l224_224043

theorem train_speed 
  (t1 : ℝ) (t2 : ℝ) (L : ℝ) (v : ℝ) 
  (h1 : t1 = 12) 
  (h2 : t2 = 44) 
  (h3 : L = v * 12)
  (h4 : L + 320 = v * 44) : 
  (v * 3.6 = 36) :=
by
  sorry

end train_speed_l224_224043


namespace giraffe_ratio_l224_224900

theorem giraffe_ratio (g ng : ℕ) (h1 : g = 300) (h2 : g = ng + 290) : g / ng = 30 :=
by
  sorry

end giraffe_ratio_l224_224900


namespace find_a_l224_224736

theorem find_a (a b c : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) 
  (h1 : a^2 / b = 1) (h2 : b^2 / c = 2) (h3 : c^2 / a = 3) : 
  a = 12^(1/7 : ℝ) :=
by
  sorry

end find_a_l224_224736


namespace betty_total_stones_l224_224799

def stones_per_bracelet : ℕ := 14
def number_of_bracelets : ℕ := 10
def total_stones : ℕ := stones_per_bracelet * number_of_bracelets

theorem betty_total_stones : total_stones = 140 := by
  sorry

end betty_total_stones_l224_224799


namespace average_eq_solution_l224_224579

theorem average_eq_solution (x : ℝ) :
  (1 / 3) * ((2 * x + 4) + (4 * x + 6) + (5 * x + 3)) = 3 * x + 5 → x = 1 :=
by
  sorry

end average_eq_solution_l224_224579


namespace find_f2_l224_224824

theorem find_f2 (f : ℝ → ℝ) (h : ∀ x y : ℝ, x * f y = y * f x) (h10 : f 10 = 30) : f 2 = 6 := 
by
  sorry

end find_f2_l224_224824


namespace h_odd_l224_224705

variable (f g : ℝ → ℝ)

-- f is odd and g is even
axiom f_odd : ∀ x, -2 ≤ x ∧ x ≤ 2 → f (-x) = -f x
axiom g_even : ∀ x, -2 ≤ x ∧ x ≤ 2 → g (-x) = g x

-- Prove that h(x) = f(x) * g(x) is odd
theorem h_odd : ∀ x, -2 ≤ x ∧ x ≤ 2 → (f x) * (g x) = (f (-x)) * (g (-x)) := by
  sorry

end h_odd_l224_224705


namespace part1_part2_l224_224958

-- Define all given conditions
variable {A B C AC BC : ℝ}
variable (A_in_range : 0 < A ∧ A < π/2)
variable (B_in_range : 0 < B ∧ B < π/2)
variable (C_in_range : 0 < C ∧ C < π/2)
variable (m_perp_n : (Real.cos (A + π/3) * Real.cos B) + (Real.sin (A + π/3) * Real.sin B) = 0)
variable (cos_B : Real.cos B = 3/5)
variable (AC_value : AC = 8)

-- First part: Prove A - B = π/6
theorem part1 : A - B = π / 6 :=
by
  sorry

-- Second part: Prove BC = 4√3 + 3 given additional conditions
theorem part2 : BC = 4 * Real.sqrt 3 + 3 :=
by
  sorry

end part1_part2_l224_224958


namespace change_color_while_preserving_friendship_l224_224168

-- Definitions
def children := Fin 10000
def colors := Fin 7
def friends (a b : children) : Prop := sorry -- mutual and exactly 11 friends per child
def refuses_to_change (c : children) : Prop := sorry -- only 100 specified children refuse to change color

theorem change_color_while_preserving_friendship :
  ∃ c : children, ¬refuses_to_change c ∧
    ∃ new_color : colors, 
      (∀ friend : children, friends c friend → 
      (∃ current_color current_friend_color : colors, current_color ≠ current_friend_color)) :=
sorry

end change_color_while_preserving_friendship_l224_224168


namespace common_tangent_length_l224_224014

noncomputable def length_of_common_tangent {r m r1 r2 : ℝ} : ℝ := 
  m / r * sqrt ((r + r1) * (r + r2))

theorem common_tangent_length
  {r m r1 r2 : ℝ}
  (hr : 0 < r)
  (hm : 0 < m)
  (hr1 : 0 < r1)
  (hr2 : 0 < r2)
  (ext_tangency : (m = (r + r1) ∨ m = (r - r1)) ∧ (m = (r + r2) ∨ m = (r - r2))):
  length_of_common_tangent r m r1 r2 = (m / r) * sqrt ((r + r1) * (r + r2)) :=
sorry

end common_tangent_length_l224_224014


namespace trapezoid_shorter_base_length_l224_224100

theorem trapezoid_shorter_base_length
  (L B : ℕ)
  (hL : L = 125)
  (hB : B = 5)
  (h : ∀ x, (L - x) / 2 = B → x = 115) :
  ∃ x, x = 115 := by
    sorry

end trapezoid_shorter_base_length_l224_224100


namespace discount_on_shoes_l224_224058

theorem discount_on_shoes (x : ℝ) :
  let shoe_price := 200
  let shirt_price := 80
  let total_spent := 285
  let total_shirt_price := 2 * shirt_price
  let initial_total := shoe_price + total_shirt_price
  let disc_shoe_price := shoe_price - (shoe_price * x / 100)
  let pre_final_total := disc_shoe_price + total_shirt_price
  let final_total := pre_final_total * (1 - 0.05)
  final_total = total_spent → x = 30 :=
by
  intros shoe_price shirt_price total_spent total_shirt_price initial_total disc_shoe_price pre_final_total final_total h
  dsimp [shoe_price, shirt_price, total_spent, total_shirt_price, initial_total, disc_shoe_price, pre_final_total, final_total] at h
  -- Here, we would normally continue the proof, but we'll insert 'sorry' for now as instructed.
  sorry

end discount_on_shoes_l224_224058


namespace equalize_vertex_values_impossible_l224_224332

theorem equalize_vertex_values_impossible 
  (n : ℕ) (h₁ : 2 ≤ n) 
  (vertex_values : Fin n → ℤ) 
  (h₂ : ∃! i : Fin n, vertex_values i = 1 ∧ ∀ j ≠ i, vertex_values j = 0) 
  (k : ℕ) (hk : k ∣ n) :
  ¬ (∃ c : ℤ, ∀ v : Fin n, vertex_values v = c) := 
sorry

end equalize_vertex_values_impossible_l224_224332


namespace age_of_B_l224_224238

theorem age_of_B (A B : ℕ) (h1 : A + 10 = 2 * (B - 10)) (h2 : A = B + 9) : B = 39 := by
  sorry

end age_of_B_l224_224238


namespace evaluate_expression_l224_224947

variable (x y : ℝ)

theorem evaluate_expression (hx : x ≠ 0) (hy : y ≠ 0) (h : x^2 - y^2 = x * y) :
  (1 / x^2) - (1 / y^2) = - (1 / (x * y)) :=
sorry

end evaluate_expression_l224_224947


namespace ceil_floor_difference_l224_224944

theorem ceil_floor_difference : 
  (Int.ceil ((15 : ℚ) / 8 * ((-34 : ℚ) / 4)) - Int.floor (((15 : ℚ) / 8) * Int.ceil ((-34 : ℚ) / 4)) = 0) :=
by 
  sorry

end ceil_floor_difference_l224_224944


namespace bob_more_than_ken_l224_224405

def ken_situps : ℕ := 20

def nathan_situps : ℕ := 2 * ken_situps

def bob_situps : ℕ := (ken_situps + nathan_situps) / 2

theorem bob_more_than_ken : bob_situps - ken_situps = 10 := 
sorry

end bob_more_than_ken_l224_224405


namespace compute_nested_operation_l224_224139

def my_op (a b : ℚ) : ℚ := (a^2 - b^2) / (1 - a * b)

theorem compute_nested_operation : my_op 1 (my_op 2 (my_op 3 4)) = -18 := by
  sorry

end compute_nested_operation_l224_224139


namespace triangle_angles_correct_l224_224104

open Real

noncomputable def angle_triple (A B C : ℝ) : Prop :=
  ∃ (a b c : ℝ),
    a = 2 * b * cos C ∧ 
    sin A * sin (B / 2 + C) = sin C * (sin (B / 2) + sin A)

theorem triangle_angles_correct (A B C : ℝ) (h : angle_triple A B C) :
  A = 5 * π / 9 ∧ B = 2 * π / 9 ∧ C = 2 * π / 9 := 
sorry

end triangle_angles_correct_l224_224104


namespace fraction_calculation_l224_224061

theorem fraction_calculation : ( ( (1/2 : ℚ) + (1/5) ) / ( (3/7) - (1/14) ) * (2/3) ) = 98/75 :=
by
  sorry

end fraction_calculation_l224_224061


namespace minimum_choir_size_l224_224183

theorem minimum_choir_size : ∃ (choir_size : ℕ), 
  (choir_size % 9 = 0) ∧ 
  (choir_size % 11 = 0) ∧ 
  (choir_size % 13 = 0) ∧ 
  (choir_size % 10 = 0) ∧ 
  (choir_size = 12870) :=
by
  sorry

end minimum_choir_size_l224_224183


namespace arithmetic_sequence_sum_l224_224254

noncomputable def a_n (a1 d : ℕ) (n : ℕ) : ℕ := a1 + (n - 1) * d

theorem arithmetic_sequence_sum 
  {a1 d : ℕ} (h_pos_d : d > 0) 
  (h_sum : a1 + (a1 + d) + (a1 + 2 * d) = 15) 
  (h_prod : a1 * (a1 + d) * (a1 + 2 * d) = 80) 
  : a_n a1 d 11 + a_n a1 d 12 + a_n a1 d 13 = 105 :=
sorry

end arithmetic_sequence_sum_l224_224254


namespace regular_polygon_sides_l224_224661

theorem regular_polygon_sides (angle : ℝ) (h1 : angle = 150) : ∃ n : ℕ, (180 * (n - 2)) / n = angle ∧ n = 12 :=
by
  sorry

end regular_polygon_sides_l224_224661


namespace linear_if_abs_k_eq_1_l224_224703

theorem linear_if_abs_k_eq_1 (k : ℤ) : |k| = 1 ↔ (k = 1 ∨ k = -1) := by
  sorry

end linear_if_abs_k_eq_1_l224_224703


namespace problem_statement_l224_224451

theorem problem_statement :
  (3 = 0.25 * x) ∧ (3 = 0.50 * y) → (x - y = 6) ∧ (x + y = 18) :=
by
  sorry

end problem_statement_l224_224451


namespace doughnut_cost_is_35_l224_224488

variable (students : ℕ) (choco_demand : ℕ) (glaze_demand : ℕ)
variable (choco_cost : ℕ) (glaze_cost : ℕ)

def total_doughnut_cost (choco_demand choco_cost glaze_demand glaze_cost : ℕ) : ℕ :=
  (choco_demand * choco_cost) + (glaze_demand * glaze_cost)

theorem doughnut_cost_is_35 : 
  students = 25 → 
  choco_demand = 10 → 
  glaze_demand = 15 → 
  choco_cost = 2 → 
  glaze_cost = 1 → 
  total_doughnut_cost choco_demand choco_cost glaze_demand glaze_cost = 35 :=
by
  intros h1 h2 h3 h4 h5
  subst h2
  subst h3
  subst h4
  subst h5
  unfold total_doughnut_cost
  norm_num
  rfl

end doughnut_cost_is_35_l224_224488


namespace lucy_clean_aquariums_l224_224862

-- Define the conditions
def lucy_cleaning_rate : ℝ := 2 / 3 -- Lucy's rate of cleaning aquariums (aquariums per hour)
def lucy_work_hours : ℕ := 24 -- Lucy's working hours this week

-- Define the goal
theorem lucy_clean_aquariums : (lucy_cleaning_rate * lucy_work_hours) = 16 := by
  sorry

end lucy_clean_aquariums_l224_224862


namespace ratio_2_10_as_percent_l224_224895

-- Define the problem conditions as given
def ratio_2_10 := 2 / 10

-- Express the question which is to show the percentage equivalent of the ratio 2:10
theorem ratio_2_10_as_percent : (ratio_2_10 * 100) = 20 :=
by
  -- Starting statement
  sorry -- Proof is not required here

end ratio_2_10_as_percent_l224_224895


namespace snow_at_least_once_three_days_l224_224443

-- Define the probability of snow on a given day
def prob_snow : ℚ := 2 / 3

-- Define the event that it snows at least once in three days
def prob_snow_at_least_once_in_three_days : ℚ :=
  1 - (1 - prob_snow)^3

-- State the theorem
theorem snow_at_least_once_three_days : prob_snow_at_least_once_in_three_days = 26 / 27 :=
by
  sorry

end snow_at_least_once_three_days_l224_224443


namespace find_usual_time_l224_224465

variables (P D T : ℝ)
variable (h1 : P = D / T)
variable (h2 : 3 / 4 * P = D / (T + 20))

theorem find_usual_time (h1 : P = D / T) (h2 : 3 / 4 * P = D / (T + 20)) : T = 80 := 
  sorry

end find_usual_time_l224_224465


namespace real_and_imaginary_parts_of_z_l224_224554

-- Define the imaginary unit
def i : ℂ := Complex.I

-- Define the complex number z
def z : ℂ := i^2 + i

-- State the theorem
theorem real_and_imaginary_parts_of_z :
  z.re = -1 ∧ z.im = 1 :=
by
  -- Provide the proof or placeholder
  sorry

end real_and_imaginary_parts_of_z_l224_224554


namespace line_equation_l224_224378

-- Given a point and a direction vector
def point : ℝ × ℝ := (3, 4)
def direction_vector : ℝ × ℝ := (-2, 1)

-- Equation of the line passing through the given point with the given direction vector
theorem line_equation (x y : ℝ) : 
  (x = 3 ∧ y = 4) → ∃a b c : ℝ, a = 1 ∧ b = 2 ∧ c = -11 ∧ a*x + b*y + c = 0 :=
by
  sorry

end line_equation_l224_224378


namespace smallest_number_collected_l224_224718

-- Define the numbers collected by each person according to the conditions
def jungkook : ℕ := 6 * 3
def yoongi : ℕ := 4
def yuna : ℕ := 5

-- The statement to prove
theorem smallest_number_collected : yoongi = min (min jungkook yoongi) yuna :=
by sorry

end smallest_number_collected_l224_224718


namespace rationalize_denominator_sum_l224_224571

theorem rationalize_denominator_sum :
  let expr := 1 / (Real.sqrt 5 + Real.sqrt 3 + Real.sqrt 11)
  ∃ (A B C D E F G H I : ℤ), 
    I > 0 ∧
    expr * (Real.sqrt 5 + Real.sqrt 3 - Real.sqrt 11) /
    ((Real.sqrt 5 + Real.sqrt 3)^2 - (Real.sqrt 11)^2) = 
        (A * Real.sqrt B + C * Real.sqrt D + E * Real.sqrt F + 
         G * Real.sqrt H) / I ∧
    (A + B + C + D + E + F + G + H + I) = 225 :=
by
  sorry

end rationalize_denominator_sum_l224_224571


namespace season_duration_l224_224010

-- Define the given conditions.
def games_per_month : ℕ := 7
def games_per_season : ℕ := 14

-- Define the property we want to prove.
theorem season_duration : games_per_season / games_per_month = 2 :=
by
  sorry

end season_duration_l224_224010


namespace cow_problem_l224_224537

noncomputable def problem_statement : Prop :=
  ∃ (F M : ℕ), F + M = 300 ∧
               (∃ S H : ℕ, S = 1/2 * F ∧ H = 1/2 * M ∧ S = H + 50) ∧
               F = 2 * M

theorem cow_problem : problem_statement :=
sorry

end cow_problem_l224_224537


namespace georgie_enter_and_exit_ways_l224_224330

-- Define the number of windows
def num_windows := 8

-- Define the magical barrier window
def barrier_window := 8

-- Define a function to count the number of ways Georgie can enter and exit the house
def count_ways_to_enter_and_exit : Nat :=
  let entry_choices := num_windows
  let exit_choices_from_normal := 6
  let exit_choices_from_barrier := 7
  let ways_from_normal := (entry_choices - 1) * exit_choices_from_normal  -- entering through windows 1 to 7
  let ways_from_barrier := 1 * exit_choices_from_barrier  -- entering through window 8
  ways_from_normal + ways_from_barrier

-- Prove the correct number of ways is 49
theorem georgie_enter_and_exit_ways : count_ways_to_enter_and_exit = 49 :=
by
  -- The calculation details are skipped with 'sorry'
  sorry

end georgie_enter_and_exit_ways_l224_224330


namespace difference_in_cents_l224_224126

-- Given definitions and conditions
def number_of_coins : ℕ := 3030
def min_nickels : ℕ := 3
def ratio_pennies_to_nickels : ℕ := 10

-- Problem statement: Prove that the difference in cents between the maximum and minimum monetary amounts is 1088
theorem difference_in_cents (p n : ℕ) (h1 : p + n = number_of_coins)
  (h2 : p ≥ ratio_pennies_to_nickels * n) (h3 : n ≥ min_nickels) :
  4 * 275 = 1100 ∧ (3030 + 1100) - (3030 + 4 * 3) = 1088 :=
by {
  sorry
}

end difference_in_cents_l224_224126


namespace interest_rate_part2_l224_224284

noncomputable def total_investment : ℝ := 3400
noncomputable def part1 : ℝ := 1300
noncomputable def part2 : ℝ := total_investment - part1
noncomputable def rate1 : ℝ := 0.03
noncomputable def total_interest : ℝ := 144
noncomputable def interest1 : ℝ := part1 * rate1
noncomputable def interest2 : ℝ := total_interest - interest1
noncomputable def rate2 : ℝ := interest2 / part2

theorem interest_rate_part2 : rate2 = 0.05 := sorry

end interest_rate_part2_l224_224284


namespace addition_result_l224_224460

theorem addition_result (x : ℝ) (h : 6 * x = 72) : x + 8 = 20 :=
sorry

end addition_result_l224_224460


namespace function_passes_through_fixed_point_l224_224438

variable (a : ℝ)

theorem function_passes_through_fixed_point (h1 : a > 0) (h2 : a ≠ 1) : (1, 2) ∈ {p : ℝ × ℝ | ∃ x, p = (x, a^(x-1) + 1)} :=
by
  sorry

end function_passes_through_fixed_point_l224_224438


namespace solution_set_f_lt_zero_a_two_solution_set_f_gt_zero_l224_224226

-- Given function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := x^2 - (a - 1) * x - a

-- Problem 1: for a = 2, solution to f(x) < 0
theorem solution_set_f_lt_zero_a_two :
  { x : ℝ | f x 2 < 0 } = { x : ℝ | -1 < x ∧ x < 2 } :=
sorry

-- Problem 2: for any a in ℝ, solution to f(x) > 0
theorem solution_set_f_gt_zero (a : ℝ) :
  { x : ℝ | f x a > 0 } =
  if a > -1 then
    {x : ℝ | x < -1} ∪ {x : ℝ | x > a}
  else if a = -1 then
    {x : ℝ | x ≠ -1}
  else
    {x : ℝ | x < a} ∪ {x : ℝ | x > -1} :=
sorry

end solution_set_f_lt_zero_a_two_solution_set_f_gt_zero_l224_224226


namespace pyramid_height_l224_224334

noncomputable def height_of_pyramid : ℝ :=
  let perimeter := 32
  let pb := 12
  let side := perimeter / 4
  let fb := (side * Real.sqrt 2) / 2
  Real.sqrt (pb^2 - fb^2)

theorem pyramid_height :
  height_of_pyramid = 4 * Real.sqrt 7 :=
by
  sorry

end pyramid_height_l224_224334


namespace total_skips_l224_224347

-- Definitions of the given conditions
def BobsSkipsPerRock := 12
def JimsSkipsPerRock := 15
def NumberOfRocks := 10

-- Statement of the theorem to be proved
theorem total_skips :
  (BobsSkipsPerRock * NumberOfRocks) + (JimsSkipsPerRock * NumberOfRocks) = 270 :=
by
  sorry

end total_skips_l224_224347


namespace range_of_a_l224_224585

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 0 then (x - a)^2 else x + (1 / x) + a

theorem range_of_a (a : ℝ) (x : ℝ) (h : ∀ x : ℝ, f a x ≥ f a 0) : 0 ≤ a ∧ a ≤ 2 :=
sorry

end range_of_a_l224_224585


namespace sum_of_possible_M_l224_224200

theorem sum_of_possible_M (M : ℝ) (h : M * (M - 8) = -8) : M = 4 ∨ M = 4 := 
by sorry

end sum_of_possible_M_l224_224200


namespace not_or_implies_both_false_l224_224980

-- The statement of the problem in Lean
theorem not_or_implies_both_false {p q : Prop} (h : ¬(p ∨ q)) : ¬p ∧ ¬q :=
sorry

end not_or_implies_both_false_l224_224980


namespace women_in_retail_l224_224271

theorem women_in_retail (total_population : ℕ) (half_population : total_population / 2 = women_count) 
  (third_of_women_work_in_retail : women_count / 3 = women_retail_count) :
  women_retail_count = 1000000 :=
by
  let total_population := 6000000
  let women_count := total_population / 2
  let women_retail_count := women_count / 3
  have h1 : women_count = 3000000 := rfl
  have h2 : women_retail_count = 1000000 := by
     rw [h1]
     exact rfl
  exact h2

end women_in_retail_l224_224271


namespace value_of_n_l224_224981

theorem value_of_n (n : ℕ) : (1 / 5 : ℝ) ^ n * (1 / 4 : ℝ) ^ 18 = 1 / (2 * (10 : ℝ) ^ 35) → n = 35 :=
by
  intro h
  sorry

end value_of_n_l224_224981


namespace cost_per_sqft_is_3_l224_224117

def deck_length : ℝ := 30
def deck_width : ℝ := 40
def extra_cost_per_sqft : ℝ := 1
def total_cost : ℝ := 4800

theorem cost_per_sqft_is_3
    (area : ℝ := deck_length * deck_width)
    (sealant_cost : ℝ := area * extra_cost_per_sqft)
    (deck_construction_cost : ℝ := total_cost - sealant_cost) :
    deck_construction_cost / area = 3 :=
by
  sorry

end cost_per_sqft_is_3_l224_224117


namespace equivalent_statement_l224_224914

variable (R G : Prop)

theorem equivalent_statement (h : ¬ R → ¬ G) : G → R := by
  intro hG
  by_contra hR
  exact h hR hG

end equivalent_statement_l224_224914


namespace sum_of_exponents_correct_l224_224461

-- Define the initial expression
def original_expr (a b c : ℤ) : ℤ := 40 * a^6 * b^9 * c^14

-- Define the simplified expression outside the radical
def simplified_outside_expr (a b c : ℤ) : ℤ := a * b^3 * c^3

-- Define the sum of the exponents
def sum_of_exponents : ℕ := 1 + 3 + 3

-- Prove that the given conditions lead to the sum of the exponents being 7
theorem sum_of_exponents_correct (a b c : ℤ) :
  original_expr a b c = 40 * a^6 * b^9 * c^14 →
  simplified_outside_expr a b c = a * b^3 * c^3 →
  sum_of_exponents = 7 :=
by
  intros
  -- Proof goes here
  sorry

end sum_of_exponents_correct_l224_224461


namespace problem_statement_l224_224952

open Complex

noncomputable def a : ℂ := 5 - 3 * I
noncomputable def b : ℂ := 2 + 4 * I

theorem problem_statement : 3 * a - 4 * b = 7 - 25 * I :=
by { sorry }

end problem_statement_l224_224952


namespace matrix_norm_min_l224_224409

-- Definition of the matrix
def matrix_mul (a b c d : ℤ) : Option (ℤ × ℤ × ℤ × ℤ) :=
  if a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 then
    some (a^2 + b * c, a * b + b * d, a * c + c * d, b * c + d^2)
  else
    none

-- Main theorem statement
theorem matrix_norm_min (a b c d : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (hc : c ≠ 0) (hd : d ≠ 0) :
  matrix_mul a b c d = some (8, 0, 0, 5) → 
  |a| + |b| + |c| + |d| = 9 :=
by
  sorry

end matrix_norm_min_l224_224409


namespace composite_sum_of_four_integers_l224_224266

theorem composite_sum_of_four_integers 
  (a b c d : ℕ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_pos_c : 0 < c) 
  (h_pos_d : 0 < d) 
  (h_eq : a^2 + b^2 + a * b = c^2 + d^2 + c * d) : 
  ∃ n m : ℕ, 1 < a + b + c + d ∧ a + b + c + d = n * m ∧ 1 < n ∧ 1 < m := 
sorry

end composite_sum_of_four_integers_l224_224266


namespace only_n_eq_1_solution_l224_224498

theorem only_n_eq_1_solution (n : ℕ) (h : n > 0): 
  (∃ x : ℤ, x^n + (2 + x)^n + (2 - x)^n = 0) ↔ n = 1 :=
by
  sorry

end only_n_eq_1_solution_l224_224498


namespace differential_solution_correct_l224_224510

noncomputable def y (x : ℝ) : ℝ := (x + 1)^2

theorem differential_solution_correct : 
  (∀ x : ℝ, deriv (deriv y) x = 2) ∧ y 0 = 1 ∧ (deriv y 0) = 2 := 
by
  sorry

end differential_solution_correct_l224_224510


namespace total_items_18_l224_224731

-- Define the number of dogs, biscuits per dog, and boots per set
def num_dogs : ℕ := 2
def biscuits_per_dog : ℕ := 5
def boots_per_set : ℕ := 4

-- Calculate the total number of items
def total_items (num_dogs biscuits_per_dog boots_per_set : ℕ) : ℕ :=
  (num_dogs * biscuits_per_dog) + (num_dogs * boots_per_set)

-- Prove that the total number of items is 18
theorem total_items_18 : total_items num_dogs biscuits_per_dog boots_per_set = 18 := by
  -- Proof is not provided
  sorry

end total_items_18_l224_224731


namespace area_of_triangle_ABF_l224_224676

theorem area_of_triangle_ABF (A B F : ℝ × ℝ) (hF : F = (1, 0)) (hA_parabola : A.2^2 = 4 * A.1) (hB_parabola : B.2^2 = 4 * B.1) (h_midpoint_AB : (A.1 + B.1) / 2 = 2 ∧ (A.2 + B.2) / 2 = 2) : 
  ∃ area : ℝ, area = 2 :=
sorry

end area_of_triangle_ABF_l224_224676


namespace numbers_left_on_board_l224_224122

theorem numbers_left_on_board : 
  let initial_numbers := { n ∈ (finset.range 21) | n > 0 },
      without_evens := initial_numbers.filter (λ n, n % 2 ≠ 0),
      final_numbers := without_evens.filter (λ n, n % 5 ≠ 4)
  in final_numbers.card = 8 := by sorry

end numbers_left_on_board_l224_224122


namespace probability_prime_and_multiple_of_11_l224_224866

theorem probability_prime_and_multiple_of_11 (h1 : 11.prime) (h2 : 11 ∣ 11) :
  (1 : ℚ) / 100 = 1 / 100 :=
by
  -- Conditions that are given in the problem
  have h_total_cards : 100 > 0 := by norm_num
  -- Card 11 is the only prime and multiple of 11 in the range 1-100
  have h_unique_card : ∃ (n : ℕ), n = 11 := ⟨11, rfl⟩
  -- Probability calculation
  sorry -- proof is not required

end probability_prime_and_multiple_of_11_l224_224866


namespace sum_of_real_solutions_l224_224511

theorem sum_of_real_solutions (x : ℝ) (h : (x^2 + 2*x + 3)^( (x^2 + 2*x + 3)^( (x^2 + 2*x + 3) )) = 2012) : 
  ∃ (x1 x2 : ℝ), (x1 + x2 = -2) ∧ (x1^2 + 2*x1 + 3 = x2^2 + 2*x2 + 3 ∧ x2^2 + 2*x2 + 3 = x^2 + 2*x + 3) := 
by
  sorry

end sum_of_real_solutions_l224_224511


namespace smallest_m_value_l224_224398

theorem smallest_m_value (a : ℕ → ℕ) (m : ℕ) (h_seq : ∀ n : ℕ, ∀ k : ℕ, 2 ^ k ≤ n ∧ n < 2 ^ (k + 1) ↔ a n = k) 
  (h_cond : a m + a (2 * m) + a (4 * m) + a (8 * m) + a (16 * m) ≥ 52) :
  m = 512 := 
by
  sorry

end smallest_m_value_l224_224398


namespace interior_angle_ratio_l224_224682

variables (α β γ : ℝ)

theorem interior_angle_ratio
  (h1 : 2 * α + 3 * β = 4 * γ)
  (h2 : α = 4 * β - γ) :
  ∃ k : ℝ, k ≠ 0 ∧ 
  (α = 2 * k ∧ β = 9 * k ∧ γ = 4 * k) :=
sorry

end interior_angle_ratio_l224_224682


namespace sum_first_five_terms_geometric_sequence_l224_224202

noncomputable def sum_first_five_geometric (a0 : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a0 * (1 - r^n) / (1 - r)

theorem sum_first_five_terms_geometric_sequence : 
  sum_first_five_geometric (1/3) (1/3) 5 = 121 / 243 := 
by 
  sorry

end sum_first_five_terms_geometric_sequence_l224_224202


namespace eq_implies_neq_neq_not_implies_eq_l224_224395

variable (a b : ℝ)

-- Define the conditions
def condition1 : Prop := a^2 = b^2
def condition2 : Prop := a^2 + b^2 = 2 * a * b

-- Theorem statement representing the problem and conclusion
theorem eq_implies_neq (h : condition2 a b) : condition1 a b :=
by
  sorry

theorem neq_not_implies_eq (h : condition1 a b) : ¬ condition2 a b :=
by
  sorry

end eq_implies_neq_neq_not_implies_eq_l224_224395


namespace regular_polygon_sides_l224_224660

theorem regular_polygon_sides (n : ℕ) (h : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l224_224660


namespace distance_P_to_outer_circle_l224_224034

theorem distance_P_to_outer_circle
  (r_large r_small : ℝ) 
  (h_tangent_inner : true) 
  (h_tangent_diameter : true) 
  (P : ℝ) 
  (O1P : ℝ)
  (O2P : ℝ := r_small)
  (O1O2 : ℝ := r_large - r_small)
  (h_O1O2_eq_680 : O1O2 = 680)
  (h_O2P_eq_320 : O2P = 320) :
  r_large - O1P = 400 :=
by
  sorry

end distance_P_to_outer_circle_l224_224034


namespace exists_fraction_expression_l224_224088

theorem exists_fraction_expression (p : ℕ) (hp_prime : Nat.Prime p) (hp_gt_three : p > 3) :
  ∃ (m : ℕ) (h₀ : 3 ≤ m) (h₁ : m ≤ p - 2) (x y : ℕ), (m : ℚ) / (p^2 : ℚ) = 1 / (x : ℚ) + 1 / (y : ℚ) :=
sorry

end exists_fraction_expression_l224_224088


namespace num_six_year_olds_l224_224751

theorem num_six_year_olds (x : ℕ) 
  (h3 : 13 = 13) 
  (h4 : 20 = 20) 
  (h5 : 15 = 15) 
  (h_sum1 : 13 + 20 = 33) 
  (h_sum2 : 15 + x = 15 + x) 
  (h_avg : 2 * 35 = 70) 
  (h_total : 33 + (15 + x) = 70) : 
  x = 22 :=
by
  sorry

end num_six_year_olds_l224_224751


namespace probability_of_5_blue_marbles_l224_224996

/--
Jane has a bag containing 9 blue marbles and 6 red marbles. 
She draws a marble, records its color, returns it to the bag, and repeats this process 8 times. 
We aim to prove that the probability that she draws exactly 5 blue marbles is \(0.279\).
-/
theorem probability_of_5_blue_marbles :
  let blue_probability := 9 / 15 
  let red_probability := 6 / 15
  let single_combination_prob := (blue_probability^5) * (red_probability^3)
  let combinations := (Nat.choose 8 5)
  let total_probability := combinations * single_combination_prob
  (Float.round (total_probability.toFloat * 1000) / 1000) = 0.279 :=
by
  sorry

end probability_of_5_blue_marbles_l224_224996


namespace clear_time_is_approximately_7_point_1_seconds_l224_224150

-- Constants for the lengths of the trains in meters
def length_train1 : ℕ := 121
def length_train2 : ℕ := 165

-- Constants for the speeds of the trains in km/h
def speed_train1 : ℕ := 80
def speed_train2 : ℕ := 65

-- Kilometer to meter conversion
def km_to_meter (km : ℕ) : ℕ := km * 1000

-- Hour to second conversion
def hour_to_second (h : ℕ) : ℕ := h * 3600

-- Relative speed of the trains in meters per second
noncomputable def relative_speed_m_per_s : ℕ := 
  (km_to_meter (speed_train1 + speed_train2)) / hour_to_second 1

-- Total distance to be covered in meters
def total_distance : ℕ := length_train1 + length_train2

-- Time to be completely clear of each other in seconds
noncomputable def clear_time : ℝ := total_distance / (relative_speed_m_per_s : ℝ)

theorem clear_time_is_approximately_7_point_1_seconds :
  abs (clear_time - 7.1) < 0.01 :=
by
  sorry

end clear_time_is_approximately_7_point_1_seconds_l224_224150


namespace find_sum_of_pqrs_l224_224725

variables (p q r s : ℝ)

-- Defining conditions
def distinct (a b c d : ℝ) : Prop := a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem find_sum_of_pqrs
    (h_distinct : distinct p q r s)
    (h_roots1 : r + s = 8 * p ∧ r * s = -12 * q)
    (h_roots2 : p + q = 8 * r ∧ p * q = -12 * s) :
    p + q + r + s = 864 := 
sorry

end find_sum_of_pqrs_l224_224725


namespace cost_of_expensive_feed_l224_224012

open Lean Real

theorem cost_of_expensive_feed (total_feed : Real)
                              (total_cost_per_pound : Real) 
                              (cheap_feed_weight : Real)
                              (cheap_cost_per_pound : Real)
                              (expensive_feed_weight : Real)
                              (expensive_cost_per_pound : Real):
  total_feed = 35 ∧ 
  total_cost_per_pound = 0.36 ∧ 
  cheap_feed_weight = 17 ∧ 
  cheap_cost_per_pound = 0.18 ∧ 
  expensive_feed_weight = total_feed - cheap_feed_weight →
  total_feed * total_cost_per_pound - cheap_feed_weight * cheap_cost_per_pound = expensive_feed_weight * expensive_cost_per_pound →
  expensive_cost_per_pound = 0.53 :=
by {
  sorry
}

end cost_of_expensive_feed_l224_224012


namespace sum_mod_13_l224_224198

theorem sum_mod_13 :
  (9023 % 13 = 5) → 
  (9024 % 13 = 6) → 
  (9025 % 13 = 7) → 
  (9026 % 13 = 8) → 
  ((9023 + 9024 + 9025 + 9026) % 13 = 0) :=
by
  intros h1 h2 h3 h4
  sorry

end sum_mod_13_l224_224198


namespace cos_half_angle_l224_224957

open Real

theorem cos_half_angle (α : ℝ) (h_sin : sin α = (4 / 9) * sqrt 2) (h_obtuse : π / 2 < α ∧ α < π) :
  cos (α / 2) = 1 / 3 :=
by
  sorry

end cos_half_angle_l224_224957


namespace price_of_second_oil_l224_224603

theorem price_of_second_oil : 
  ∃ x : ℝ, 
    (10 * 50 + 5 * x = 15 * 56) → x = 68 := by
  sorry

end price_of_second_oil_l224_224603


namespace midpoint_locus_l224_224930

theorem midpoint_locus (c : ℝ) (H : 0 < c ∧ c ≤ Real.sqrt 2) :
  ∃ L, L = "curvilinear quadrilateral with arcs forming transitions" :=
sorry

end midpoint_locus_l224_224930


namespace longer_subsegment_length_l224_224588

-- Define the given conditions and proof goal in Lean 4
theorem longer_subsegment_length {DE EF DF DG GF : ℝ} (h1 : 3 * EF < 4 * EF) (h2 : 4 * EF < 5 * EF)
  (ratio_condition : DE / EF = 4 / 5) (DF_length : DF = 12) :
  DG + GF = DF ∧ DE / EF = DG / GF ∧ GF = (5 * 12 / 9) :=
by
  sorry

end longer_subsegment_length_l224_224588


namespace sin_cos_inequality_l224_224570

theorem sin_cos_inequality (x : ℝ) (n : ℕ) : 
  (Real.sin (2 * x))^n + (Real.sin x^n - Real.cos x^n)^2 ≤ 1 := 
by
  sorry

end sin_cos_inequality_l224_224570


namespace fair_coin_toss_consecutive_heads_l224_224783

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem fair_coin_toss_consecutive_heads :
  let total_outcomes := 1024
  let favorable_outcomes := 
    1 + binom 10 1 + binom 9 2 + binom 8 3 + binom 7 4 + binom 6 5
  let prob := favorable_outcomes / total_outcomes
  let i := 9
  let j := 64
  Nat.gcd i j = 1 ∧ (prob = i / j) ∧ i + j = 73 :=
by
  sorry

end fair_coin_toss_consecutive_heads_l224_224783


namespace value_of_w_over_y_l224_224528

theorem value_of_w_over_y (w x y : ℝ) (h1 : w / x = 1 / 3) (h2 : (x + y) / y = 3.25) : w / y = 0.75 :=
sorry

end value_of_w_over_y_l224_224528


namespace sum_of_digits_of_N_l224_224337

theorem sum_of_digits_of_N (N : ℕ) (h : N * (N + 1) / 2 = 2016) : (6 + 3 = 9) :=
by
  sorry

end sum_of_digits_of_N_l224_224337


namespace main_inequality_l224_224558

theorem main_inequality (a b c d : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_c_pos : 0 < c) (h_d_pos : 0 < d) (h_sum : a + b + c + d = 1) :
  (b * c * d) / (1 - a)^2 + (c * d * a) / (1 - b)^2 + (d * a * b) / (1 - c)^2 + (a * b * c) / (1 - d)^2 ≤ 1 / 9 :=
by
  sorry

end main_inequality_l224_224558


namespace extreme_value_of_f_range_of_a_l224_224229

noncomputable def f (x : ℝ) : ℝ := x * real.exp (x + 1)

theorem extreme_value_of_f : f (-1) = -1 := by
  -- The proof would go here
  sorry

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x > 0 → f x ≥ x + real.log x + a + 1) → a ≤ 1 := by
  -- The proof would go here
  sorry

end extreme_value_of_f_range_of_a_l224_224229


namespace untouched_shapes_after_moves_l224_224467

-- Definitions
def num_shapes : ℕ := 12
def num_triangles : ℕ := 3
def num_squares : ℕ := 4
def num_pentagons : ℕ := 5
def total_moves : ℕ := 10
def petya_moves_first : Prop := True
def vasya_strategy : Prop := True  -- Vasya's strategy to minimize untouched shapes
def petya_strategy : Prop := True  -- Petya's strategy to maximize untouched shapes

-- Theorem
theorem untouched_shapes_after_moves : num_shapes = 12 ∧ num_triangles = 3 ∧ num_squares = 4 ∧ num_pentagons = 5 ∧
                                        total_moves = 10 ∧ petya_moves_first ∧ vasya_strategy ∧ petya_strategy → 
                                        num_shapes - 5 = 6 :=
by
  sorry

end untouched_shapes_after_moves_l224_224467


namespace arithmetic_sequence_common_difference_l224_224849

theorem arithmetic_sequence_common_difference
  (a1 a4 : ℤ) (d : ℤ) 
  (h1 : a1 + (a1 + 4 * d) = 10)
  (h2 : a1 + 3 * d = 7) : 
  d = 2 :=
sorry

end arithmetic_sequence_common_difference_l224_224849


namespace solve_quadratic_equation_l224_224873

noncomputable def solve_log_equation (x : ℝ) : Prop :=
  log (1 / 3 : ℝ) (x^2 + 3 * x - 4) = log (1 / 3 : ℝ) (2 * x + 2)

theorem solve_quadratic_equation :
  solve_log_equation 2 :=
by
  sorry

end solve_quadratic_equation_l224_224873


namespace sample_size_correct_l224_224758

-- Define the total number of students in a certain grade.
def total_students : ℕ := 500

-- Define the number of students selected for statistical analysis.
def selected_students : ℕ := 30

-- State the theorem to prove the selected students represent the sample size.
theorem sample_size_correct : selected_students = 30 := by
  -- The proof would go here, but we use sorry to indicate it is skipped.
  sorry

end sample_size_correct_l224_224758


namespace dans_age_l224_224463

variable {x : ℤ}

theorem dans_age (h : x + 20 = 7 * (x - 4)) : x = 8 := by
  sorry

end dans_age_l224_224463


namespace g_at_8_equals_minus_30_l224_224884

noncomputable def g : ℝ → ℝ := sorry

theorem g_at_8_equals_minus_30 :
  (∀ x y : ℝ, g x + g (3 * x + y) + 7 * x * y = g (4 * x - y) + 3 * x^2 + 2) →
  g 8 = -30 :=
by
  intro h
  sorry

end g_at_8_equals_minus_30_l224_224884


namespace find_11th_place_l224_224385

def placement_problem (Amara Bindu Carlos Devi Eshan Farel: ℕ): Prop :=
  (Carlos + 5 = Amara) ∧
  (Bindu = Eshan + 3) ∧
  (Carlos = Devi + 2) ∧
  (Devi = 6) ∧
  (Eshan + 1 = Farel) ∧
  (Bindu + 4 = Amara) ∧
  (Farel = 9)

theorem find_11th_place (Amara Bindu Carlos Devi Eshan Farel: ℕ) 
  (h : placement_problem Amara Bindu Carlos Devi Eshan Farel) : 
  Eshan = 11 := 
sorry

end find_11th_place_l224_224385


namespace solution_concentration_l224_224574

theorem solution_concentration (y z : ℝ) :
  let x_vol := 300
  let y_vol := 2 * z
  let z_vol := z
  let total_vol := x_vol + y_vol + z_vol
  let alcohol_x := 0.10 * x_vol
  let alcohol_y := 0.30 * y_vol
  let alcohol_z := 0.40 * z_vol
  let total_alcohol := alcohol_x + alcohol_y + alcohol_z
  total_vol = 600 ∧ y_vol = 2 * z_vol ∧ y_vol + z_vol = 300 → 
  total_alcohol / total_vol = 21.67 / 100 :=
by
  sorry

end solution_concentration_l224_224574


namespace sam_has_12_nickels_l224_224573

theorem sam_has_12_nickels (n d : ℕ) (h1 : n + d = 30) (h2 : 5 * n + 10 * d = 240) : n = 12 :=
sorry

end sam_has_12_nickels_l224_224573


namespace cos_C_values_l224_224543

theorem cos_C_values (sin_A : ℝ) (cos_B : ℝ) (cos_C : ℝ) 
  (h1 : sin_A = 4 / 5) 
  (h2 : cos_B = 12 / 13) 
  : cos_C = -16 / 65 ∨ cos_C = 56 / 65 :=
by
  sorry

end cos_C_values_l224_224543


namespace part1_part2_l224_224492

-- Problem part (1)
theorem part1 : (Real.sqrt 12 + Real.sqrt (4 / 3)) * Real.sqrt 3 = 8 := 
  sorry

-- Problem part (2)
theorem part2 : Real.sqrt 48 - Real.sqrt 54 / Real.sqrt 2 + (3 - Real.sqrt 3) * (3 + Real.sqrt 3) = Real.sqrt 3 + 6 := 
  sorry

end part1_part2_l224_224492


namespace expected_value_sum_of_three_marbles_l224_224977

def bag := {1, 2, 3, 4, 5, 6, 7}
def selected_subsets := {s : Set ℕ | s.card = 3 ∧ s ⊆ bag}

theorem expected_value_sum_of_three_marbles : 
  (∑ s in selected_subsets, (s.sum : ℚ)) / (selected_subsets.card : ℚ) = 12 :=
by sorry

end expected_value_sum_of_three_marbles_l224_224977


namespace cold_brew_cost_l224_224272

theorem cold_brew_cost :
  let drip_coffee_cost := 2.25
  let espresso_cost := 3.50
  let latte_cost := 4.00
  let vanilla_syrup_cost := 0.50
  let cappuccino_cost := 3.50
  let total_order_cost := 25.00
  let drip_coffee_total := 2 * drip_coffee_cost
  let lattes_total := 2 * latte_cost
  let known_costs := drip_coffee_total + espresso_cost + lattes_total + vanilla_syrup_cost + cappuccino_cost
  total_order_cost - known_costs = 5.00 →
  5.00 / 2 = 2.50 := by sorry

end cold_brew_cost_l224_224272


namespace problem_statement_l224_224515

theorem problem_statement (a b : ℝ) (h : a^2 + |b + 1| = 0) : (a + b)^2015 = -1 := by
  sorry

end problem_statement_l224_224515


namespace sum_le_six_l224_224521

theorem sum_le_six (a b : ℤ) (h1 : a ≠ -1) (h2 : b ≠ -1) 
    (h3 : ∃ (r s : ℤ), r * s = a + b ∧ r + s = ab) : a + b ≤ 6 :=
sorry

end sum_le_six_l224_224521


namespace average_after_discard_l224_224434

theorem average_after_discard (sum_50 : ℝ) (avg_50 : sum_50 = 2200) (a b : ℝ) (h1 : a = 45) (h2 : b = 55) :
  (sum_50 - (a + b)) / 48 = 43.75 :=
by
  -- Given conditions: sum_50 = 2200, a = 45, b = 55
  -- We need to prove (sum_50 - (a + b)) / 48 = 43.75
  sorry

end average_after_discard_l224_224434


namespace identify_nearly_regular_polyhedra_l224_224359

structure Polyhedron :=
  (faces : ℕ)
  (edges : ℕ)
  (vertices : ℕ)

def nearlyRegularPolyhedra : List Polyhedron :=
  [ 
    ⟨8, 12, 6⟩,   -- Properties of Tetrahedron-octahedron intersection
    ⟨14, 24, 12⟩, -- Properties of Cuboctahedron
    ⟨32, 60, 30⟩  -- Properties of Dodecahedron-Icosahedron
  ]

theorem identify_nearly_regular_polyhedra :
  nearlyRegularPolyhedra = [
    ⟨8, 12, 6⟩,  -- Tetrahedron-octahedron intersection
    ⟨14, 24, 12⟩, -- Cuboctahedron
    ⟨32, 60, 30⟩  -- Dodecahedron-icosahedron intersection
  ] :=
by
  sorry

end identify_nearly_regular_polyhedra_l224_224359


namespace faster_train_speed_l224_224919

theorem faster_train_speed (length_train : ℝ) (time_cross : ℝ) (speed_ratio : ℝ) (total_distance : ℝ) (relative_speed : ℝ) :
  length_train = 100 → 
  time_cross = 8 → 
  speed_ratio = 2 → 
  total_distance = 2 * length_train → 
  relative_speed = (1 + speed_ratio) * (total_distance / time_cross) → 
  (1 + speed_ratio) * (total_distance / time_cross) / 3 * 2 = 8.33 := 
by
  intros
  sorry

end faster_train_speed_l224_224919


namespace six_digit_permutation_reverse_div_by_11_l224_224856

theorem six_digit_permutation_reverse_div_by_11 
  (a b c : ℕ)
  (h_a : 1 ≤ a ∧ a ≤ 9)
  (h_b : 0 ≤ b ∧ b ≤ 9)
  (h_c : 0 ≤ c ∧ c ≤ 9)
  (X : ℕ)
  (h_X : X = 100001 * a + 10010 * b + 1100 * c) :
  11 ∣ X :=
by 
  sorry

end six_digit_permutation_reverse_div_by_11_l224_224856


namespace retail_women_in_LA_l224_224270

/-
Los Angeles has 6 million people living in it. If half the population is women 
and 1/3 of the women work in retail, how many women work in retail in Los Angeles?
-/

theorem retail_women_in_LA 
  (total_population : ℕ)
  (half_population_women : total_population / 2 = women_population)
  (third_women_retail : women_population / 3 = retail_women)
  : total_population = 6000000 → retail_women = 1000000 :=
by
  sorry

end retail_women_in_LA_l224_224270


namespace find_number_l224_224357

theorem find_number (x : ℝ) (h : 15 * x = 300) : x = 20 :=
by 
  sorry

end find_number_l224_224357


namespace like_term_exists_l224_224871

variable (a b : ℝ) (x y : ℝ)

theorem like_term_exists : ∃ b : ℝ, b * x^5 * y^3 = 3 * x^5 * y^3 ∧ b ≠ a :=
by
  -- existence of b
  use 3
  -- proof is omitted
  sorry

end like_term_exists_l224_224871


namespace rectangle_area_is_140_l224_224464

noncomputable def area_of_square (a : ℝ) : ℝ := a * a
noncomputable def length_of_rectangle (r : ℝ) : ℝ := (2 / 5) * r
noncomputable def area_of_rectangle (l : ℝ) (b : ℝ) : ℝ := l * b

theorem rectangle_area_is_140 :
  ∃ (a r l b : ℝ), area_of_square a = 1225 ∧ r = a ∧ l = length_of_rectangle r ∧ b = 10 ∧ area_of_rectangle l b = 140 :=
by
  use 35, 35, 14, 10
  simp [area_of_square, length_of_rectangle, area_of_rectangle]
  sorry

end rectangle_area_is_140_l224_224464


namespace circumference_greater_than_100_l224_224901

def running_conditions (A B : ℝ) (C : ℝ) (P : ℝ) : Prop :=
  A > 0 ∧ B > 0 ∧ A ≠ B ∧ P = 0 ∧ C > 0

theorem circumference_greater_than_100 (A B C P : ℝ) (h : running_conditions A B C P):
  C > 100 :=
by
  sorry

end circumference_greater_than_100_l224_224901


namespace probability_not_cash_l224_224345

theorem probability_not_cash (h₁ : 0.45 + 0.15 + pnc = 1) : pnc = 0.4 :=
by
  sorry

end probability_not_cash_l224_224345


namespace consumer_installment_credit_l224_224619

theorem consumer_installment_credit (A C : ℝ) (h1 : A = 0.36 * C) (h2 : 35 = (1 / 3) * A) :
  C = 291.67 :=
by 
  -- The proof should go here
  sorry

end consumer_installment_credit_l224_224619


namespace smallest_positive_period_of_f_range_of_f_in_interval_l224_224225

noncomputable def f (x: ℝ) : ℝ :=
  2 * (Real.sin x) ^ 2 + 2 * (Real.sqrt 3) * (Real.sin x) * (Real.sin (x + Real.pi / 2))

#eval (Real.sin (x + Real.pi / 2)) -- should simplify to Real.cos x

theorem smallest_positive_period_of_f :
  ∃ T > 0, (∀ x, f(x + T) = f x) ∧ (∀ τ > 0, (∀ x, f(x + τ) = f x) → τ ≥ T) := sorry

theorem range_of_f_in_interval :
  ∀ x, 0 ≤ x ∧ x ≤ (2 * Real.pi / 3) → 0 ≤ f x ∧ f x ≤ 3 := sorry

end smallest_positive_period_of_f_range_of_f_in_interval_l224_224225


namespace solution_of_inequalities_l224_224945

theorem solution_of_inequalities (x : ℝ) :
  (2 * x / 5 ≤ 3 + x) ∧ (3 + x < -3 * (1 + x)) ↔ (-5 ≤ x ∧ x < -3 / 2) := by
  sorry

end solution_of_inequalities_l224_224945


namespace extreme_value_at_one_symmetric_points_range_l224_224224

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 0 then
  x^2 + 3 * a * x
else
  2 * Real.exp x - x^2 + 2 * a * x

theorem extreme_value_at_one (a : ℝ) :
  (∀ x > 0, f x a = 2 * Real.exp x - x^2 + 2 * a * x) →
  (∀ x < 0, f x a = x^2 + 3 * a * x) →
  (∀ x > 0, deriv (fun x => f x a) x = 2 * Real.exp x - 2 * x + 2 * a) →
  deriv (fun x => f x a) 1 = 0 →
  a = 1 - Real.exp 1 :=
  sorry

theorem symmetric_points_range (a : ℝ) :
  (∃ x0 > 0, (∃ y0 : ℝ, 
  (f x0 a = y0 ∧ f (-x0) a = -y0))) →
  a ≥ 2 * Real.exp 1 :=
  sorry

end extreme_value_at_one_symmetric_points_range_l224_224224


namespace find_constants_l224_224567

-- Given definitions based on the conditions and conjecture
def S (n : ℕ) : ℕ := 
  match n with
  | 1 => 1
  | 2 => 5
  | 3 => 15
  | 4 => 34
  | 5 => 65
  | _ => 0

noncomputable def conjecture_S (n a b c : ℤ) := (2 * n - 1) * (a * n^2 + b * n + c)

theorem find_constants (a b c : ℤ) (h1 : conjecture_S 1 a b c = 1) (h2 : conjecture_S 2 a b c = 5) (h3 : conjecture_S 3 a b c = 15) : 3 * a + b = 4 :=
by
  -- Proof omitted
  sorry

end find_constants_l224_224567


namespace solution_set_of_inequality_l224_224199

theorem solution_set_of_inequality (x : ℝ) : x * (1 - x) > 0 ↔ 0 < x ∧ x < 1 :=
sorry

end solution_set_of_inequality_l224_224199


namespace find_prime_p_l224_224360

theorem find_prime_p (p x y : ℕ) (hp : Nat.Prime p) (hx : x > 0) (hy : y > 0) :
  (p + 49 = 2 * x^2) ∧ (p^2 + 49 = 2 * y^2) ↔ p = 23 :=
by
  sorry

end find_prime_p_l224_224360


namespace rhombus_area_from_roots_l224_224370

-- Definition of the quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 - 10 * x + 24 = 0

-- Define the roots of the quadratic equation
def roots (a b : ℝ) : Prop := quadratic_eq a ∧ quadratic_eq b

-- Final mathematical statement to prove
theorem rhombus_area_from_roots (a b : ℝ) (h : roots a b) :
  a * b = 24 → (1 / 2) * a * b = 12 := 
by
  sorry

end rhombus_area_from_roots_l224_224370


namespace price_of_turban_l224_224693

theorem price_of_turban : 
  ∃ T : ℝ, (9 / 12) * (90 + T) = 40 + T ↔ T = 110 :=
by
  sorry

end price_of_turban_l224_224693


namespace lucy_cleans_aquariums_l224_224861

theorem lucy_cleans_aquariums :
  (∃ rate : ℕ, rate = 2 / 3) →
  (∃ hours : ℕ, hours = 24) →
  (∃ increments : ℕ, increments = 24 / 3) →
  (∃ aquariums : ℕ, aquariums = (2 * (24 / 3))) →
  aquariums = 16 :=
by
  sorry

end lucy_cleans_aquariums_l224_224861


namespace total_weight_of_rice_l224_224872

theorem total_weight_of_rice :
  (29 * 4) / 16 = 7.25 := by
sorry

end total_weight_of_rice_l224_224872


namespace baker_cakes_l224_224620

theorem baker_cakes (C : ℕ) (h1 : 154 = 78 + 76) (h2 : C = 78) : C = 78 :=
sorry

end baker_cakes_l224_224620


namespace cheerleader_total_l224_224879

theorem cheerleader_total 
  (size2 : ℕ)
  (size6 : ℕ)
  (size12 : ℕ)
  (h1 : size2 = 4)
  (h2 : size6 = 10)
  (h3 : size12 = size6 / 2) :
  size2 + size6 + size12 = 19 :=
by
  sorry

end cheerleader_total_l224_224879


namespace total_distance_karl_drove_l224_224550

theorem total_distance_karl_drove :
  ∀ (consumption_rate miles_per_gallon : ℕ) 
    (tank_capacity : ℕ) 
    (initial_gas : ℕ) 
    (distance_leg1 : ℕ) 
    (purchased_gas : ℕ) 
    (remaining_gas : ℕ)
    (final_gas : ℕ),
  consumption_rate = 25 → 
  tank_capacity = 18 →
  initial_gas = 12 →
  distance_leg1 = 250 →
  purchased_gas = 10 →
  remaining_gas = initial_gas - distance_leg1 / consumption_rate + purchased_gas →
  final_gas = remaining_gas - distance_leg2 / consumption_rate →
  remaining_gas - distance_leg2 / consumption_rate = final_gas →
  distance_leg2 = (initial_gas - remaining_gas + purchased_gas - final_gas) * miles_per_gallon →
  miles_per_gallon = 25 →
  distance_leg2 + distance_leg1 = 475 :=
sorry

end total_distance_karl_drove_l224_224550


namespace cost_per_bottle_l224_224401

theorem cost_per_bottle (cost_3_bottles cost_4_bottles : ℝ) (n_bottles : ℕ) 
  (h1 : cost_3_bottles = 1.50) (h2 : cost_4_bottles = 2) : 
  (cost_3_bottles / 3) = (cost_4_bottles / 4) ∧ (cost_3_bottles / 3) * n_bottles = 0.50 * n_bottles :=
by
  sorry

end cost_per_bottle_l224_224401


namespace first_scenario_machines_l224_224093

theorem first_scenario_machines (M : ℕ) (h1 : 20 = 10 * 2 * M) (h2 : 140 = 20 * 17.5 * 2) : M = 5 :=
by sorry

end first_scenario_machines_l224_224093


namespace total_candidates_l224_224580

def average_marks_all_candidates : ℕ := 35
def average_marks_passed_candidates : ℕ := 39
def average_marks_failed_candidates : ℕ := 15
def passed_candidates : ℕ := 100

theorem total_candidates (T : ℕ) (F : ℕ) 
  (h1 : 35 * T = 39 * passed_candidates + 15 * F)
  (h2 : T = passed_candidates + F) : T = 120 := 
  sorry

end total_candidates_l224_224580


namespace point_on_circle_l224_224078

theorem point_on_circle (a b : ℝ) 
  (h1 : (b + 2) * x + a * y + 4 = 0) 
  (h2 : a * x + (2 - b) * y - 3 = 0) 
  (parallel_lines : ∀ x y : ℝ, ∀ C1 C2 : ℝ, 
    (b + 2) * x + a * y + C1 = 0 ∧ a * x + (2 - b) * y + C2 = 0 → 
    - (b + 2) / a = - a / (2 - b)
  ) : a^2 + b^2 = 4 :=
sorry

end point_on_circle_l224_224078


namespace ratio_of_selling_to_buying_l224_224274

noncomputable def natasha_has_3_times_carla (N C : ℕ) : Prop :=
  N = 3 * C

noncomputable def carla_has_2_times_cosima (C S : ℕ) : Prop :=
  C = 2 * S

noncomputable def total_buying_price (N C S : ℕ) : ℕ :=
  N + C + S

noncomputable def total_selling_price (buying_price profit : ℕ) : ℕ :=
  buying_price + profit

theorem ratio_of_selling_to_buying (N C S buying_price selling_price ratio : ℕ) 
  (h1 : natasha_has_3_times_carla N C)
  (h2 : carla_has_2_times_cosima C S)
  (h3 : N = 60)
  (h4 : buying_price = total_buying_price N C S)
  (h5 : total_selling_price buying_price 36 = selling_price)
  (h6 : 18 * ratio = selling_price * 5): ratio = 7 :=
by
  sorry

end ratio_of_selling_to_buying_l224_224274


namespace read_both_books_l224_224421

theorem read_both_books (B S K N : ℕ) (TOTAL : ℕ)
  (h1 : S = 1/4 * 72)
  (h2 : K = 5/8 * 72)
  (h3 : N = (S - B) - 1)
  (h4 : TOTAL = 72)
  (h5 : TOTAL = (S - B) + (K - B) + B + N)
  : B = 8 :=
by
  sorry

end read_both_books_l224_224421


namespace product_sequence_l224_224502

theorem product_sequence : 
  let seq := [1/3, 9/1, 1/27, 81/1, 1/243, 729/1, 1/2187, 6561/1, 1/19683, 59049/1]
  ((seq[0] * seq[1]) * (seq[2] * seq[3]) * (seq[4] * seq[5]) * (seq[6] * seq[7]) * (seq[8] * seq[9])) = 243 :=
by
  sorry

end product_sequence_l224_224502


namespace min_slope_of_tangent_l224_224797

def f (x : ℝ) : ℝ := x^3 + 3 * x^2 + 6 * x - 10

theorem min_slope_of_tangent : (∀ x : ℝ, 3 * (x + 1)^2 + 3 ≥ 3) :=
by 
  sorry

end min_slope_of_tangent_l224_224797


namespace harvey_sold_17_steaks_l224_224966

variable (initial_steaks : ℕ) (steaks_left_after_first_sale : ℕ) (steaks_sold_in_second_sale : ℕ)

noncomputable def total_steaks_sold (initial_steaks steaks_left_after_first_sale steaks_sold_in_second_sale : ℕ) : ℕ :=
  (initial_steaks - steaks_left_after_first_sale) + steaks_sold_in_second_sale

theorem harvey_sold_17_steaks :
  initial_steaks = 25 →
  steaks_left_after_first_sale = 12 →
  steaks_sold_in_second_sale = 4 →
  total_steaks_sold initial_steaks steaks_left_after_first_sale steaks_sold_in_second_sale = 17 :=
by
  intros
  sorry

end harvey_sold_17_steaks_l224_224966


namespace number_of_parents_l224_224902

theorem number_of_parents (P : ℕ) (h : P + 177 = 238) : P = 61 :=
by
  sorry

end number_of_parents_l224_224902


namespace arithmetic_sequence_general_term_sum_of_first_n_terms_l224_224110

theorem arithmetic_sequence_general_term 
  (a_n : ℕ → ℕ) (S : ℕ → ℕ)
  (d : ℕ) (h_d_nonzero : d ≠ 0)
  (h_arith : ∀ n, a_n = a_n 0 + n * d)
  (h_S9 : S 9 = 90)
  (h_geom : ∃ (a1 a2 a4 : ℕ), a2^2 = a1 * a4)
  (h_common_diff : d = a_n 1 - a_n 0)
  : ∀ n, a_n = 2 * n  := 
sorry

theorem sum_of_first_n_terms
  (b_n : ℕ → ℕ)
  (T : ℕ → ℕ)
  (a_n : ℕ → ℕ) 
  (h_b_def : ∀ n, b_n = 1 / (a_n n * a_n (n+1)))
  (h_a_form : ∀ n, a_n = 2 * n)
  : ∀ n, T n = n / (4 * n + 4) :=
sorry

end arithmetic_sequence_general_term_sum_of_first_n_terms_l224_224110


namespace R_and_D_expenditure_per_unit_increase_l224_224490

theorem R_and_D_expenditure_per_unit_increase :
  (R_t : ℝ) (delta_APL_t_plus_1 : ℝ) (R_t = 3157.61) (delta_APL_t_plus_1 = 0.69) :
  R_t / delta_APL_t_plus_1 = 4576 :=
by
  sorry

end R_and_D_expenditure_per_unit_increase_l224_224490


namespace sum_of_star_angles_l224_224903

theorem sum_of_star_angles :
  let n := 12
  let angle_per_arc := 360 / n
  let arcs_per_tip := 3
  let internal_angle_per_tip := 360 - arcs_per_tip * angle_per_arc
  let sum_of_angles := n * (360 - internal_angle_per_tip)
  sum_of_angles = 1080 :=
by
  sorry

end sum_of_star_angles_l224_224903


namespace candy_groups_l224_224006

theorem candy_groups (total_candies group_size : Nat) (h1 : total_candies = 30) (h2 : group_size = 3) : total_candies / group_size = 10 := by
  sorry

end candy_groups_l224_224006


namespace count_3_digit_numbers_divisible_by_5_l224_224971

theorem count_3_digit_numbers_divisible_by_5 :
  let a := 100
  let l := 995
  let d := 5
  let n := (l - a) / d + 1
  n = 180 :=
by
  sorry

end count_3_digit_numbers_divisible_by_5_l224_224971


namespace value_of_expression_l224_224091

theorem value_of_expression 
  (triangle square : ℝ) 
  (h1 : 3 + triangle = 5) 
  (h2 : triangle + square = 7) : 
  triangle + triangle + triangle + square + square = 16 :=
by
  sorry

end value_of_expression_l224_224091


namespace BP_PA_ratio_l224_224993

section

variable (A B C P : Type)
variable {AC BC PA PB BP : ℕ}

-- Conditions:
-- 1. In triangle ABC, the ratio AC:CB = 2:5.
axiom AC_CB_ratio : 2 * BC = 5 * AC

-- 2. The bisector of the exterior angle at C intersects the extension of BA at P,
--    such that B is between P and A.
axiom Angle_Bisector_Theorem : PA * BC = PB * AC

theorem BP_PA_ratio (h1 : 2 * BC = 5 * AC) (h2 : PA * BC = PB * AC) :
  BP * PA = 5 * PA := sorry

end

end BP_PA_ratio_l224_224993


namespace find_fraction_l224_224601

theorem find_fraction (x y : ℤ) (h1 : x + 2 = y + 1) (h2 : 2 * (x + 4) = y + 2) : 
  x = -5 ∧ y = -4 := 
sorry

end find_fraction_l224_224601


namespace bob_more_than_ken_l224_224403

def ken_situps : ℕ := 20

def nathan_situps : ℕ := 2 * ken_situps

def bob_situps : ℕ := (ken_situps + nathan_situps) / 2

theorem bob_more_than_ken : bob_situps - ken_situps = 10 := by
  -- proof steps to be filled in
  sorry

end bob_more_than_ken_l224_224403


namespace determine_k_range_l224_224825

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * x
noncomputable def g (x : ℝ) : ℝ := (Real.log x) / x
noncomputable def h (x : ℝ) : ℝ := (Real.log x) / (x * x)

theorem determine_k_range :
  (∀ x : ℝ, x ∈ Set.Icc (1 / Real.exp 1) (Real.exp 1) → f k x = g x) →
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 ∈ Set.Icc (1 / Real.exp 1) (Real.exp 1) ∧ x2 ∈ Set.Icc (1 / Real.exp 1) (Real.exp 1)) →
  k ∈ Set.Ico (1 / (Real.exp 1) ^ 2) (1 / (2 * Real.exp 1)) := 
  sorry

end determine_k_range_l224_224825


namespace function_decreasing_range_k_l224_224586

theorem function_decreasing_range_k : 
  ∀ k : ℝ, (∀ x : ℝ, 1 ≤ x → ∀ y : ℝ, 1 ≤ y → x ≤ y → (k * x ^ 2 + (3 * k - 2) * x - 5) ≥ (k * y ^ 2 + (3 * k - 2) * y - 5)) ↔ (k ∈ Set.Iic 0) :=
by sorry

end function_decreasing_range_k_l224_224586


namespace prism_volume_is_correct_l224_224788

noncomputable def prism_volume 
  (a b c : ℝ) 
  (hab : a * b = 15) 
  (hbc : b * c = 18) 
  (hca : c * a = 20) 
  (hc_longest : c = 2 * a) 
  : ℝ :=
  a * b * c

theorem prism_volume_is_correct 
  (a b c : ℝ) 
  (hab : a * b = 15) 
  (hbc : b * c = 18) 
  (hca : c * a = 20) 
  (hc_longest : c = 2 * a) 
  : prism_volume a b c hab hbc hca hc_longest = 30 * Real.sqrt 10 :=
sorry

end prism_volume_is_correct_l224_224788


namespace positive_integer_solutions_l224_224361

theorem positive_integer_solutions :
  ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x + y + x * y = 2008 ∧
  ((x = 6 ∧ y = 286) ∨ (x = 286 ∧ y = 6) ∨ (x = 40 ∧ y = 48) ∨ (x = 48 ∧ y = 40)) :=
by
  sorry

end positive_integer_solutions_l224_224361


namespace find_ratio_l224_224883

theorem find_ratio (f : ℝ → ℝ) (h : ∀ a b : ℝ, b^2 * f a = a^2 * f b) (h3 : f 3 ≠ 0) :
  (f 7 - f 3) / f 3 = 40 / 9 :=
sorry

end find_ratio_l224_224883


namespace willy_crayons_eq_l224_224321

def lucy_crayons : ℕ := 3971
def more_crayons : ℕ := 1121

theorem willy_crayons_eq : 
  ∀ willy_crayons : ℕ, willy_crayons = lucy_crayons + more_crayons → willy_crayons = 5092 :=
by
  sorry

end willy_crayons_eq_l224_224321


namespace pants_cost_l224_224351

def total_cost (P : ℕ) : ℕ := 4 * 8 + 2 * 60 + 2 * P

theorem pants_cost :
  (∃ P : ℕ, total_cost P = 188) →
  ∃ P : ℕ, P = 18 :=
by
  intro h
  sorry

end pants_cost_l224_224351


namespace increase_a1_intervals_of_increase_l224_224687

noncomputable def f (x a : ℝ) : ℝ := x - (a + 1) * Real.log x - a / x

-- Prove that when a = 1, f(x) has no extreme points (i.e., it is monotonically increasing in (0, +∞))
theorem increase_a1 : ∀ x : ℝ, 0 < x → f x 1 = x - 2 * Real.log x - 1 / x :=
sorry

-- Find the intervals of increase for f(x) = x - (a+1) ln x - a/x
theorem intervals_of_increase (a : ℝ) : 
  (a ≤ 0 → ∀ x : ℝ, 1 < x → 0 ≤ (f x a - f 1 a)) ∧ 
  (0 < a ∧ a < 1 → (∀ x : ℝ, 0 < x ∧ x < a → 0 ≤ f x a) ∧ ∀ x : ℝ, 1 < x → 0 ≤ f x a ) ∧ 
  (a = 1 → ∀ x : ℝ, 0 < x → 0 ≤ f x a) ∧ 
  (a > 1 → (∀ x : ℝ, 0 < x ∧ x < 1 → 0 ≤ f x a) ∧ ∀ x : ℝ, a < x → 0 ≤ f x a ) :=
sorry

end increase_a1_intervals_of_increase_l224_224687


namespace price_decrease_l224_224896

theorem price_decrease (P : ℝ) (h₁ : 1.25 * P = P * 1.25) (h₂ : 1.10 * P = P * 1.10) :
  1.25 * P * (1 - 12 / 100) = 1.10 * P :=
by
  sorry

end price_decrease_l224_224896


namespace winning_configurations_for_blake_l224_224617

def isWinningConfigurationForBlake (config : List ℕ) := 
  let nimSum := config.foldl (xor) 0
  nimSum = 0

theorem winning_configurations_for_blake :
  (isWinningConfigurationForBlake [8, 2, 3]) ∧ 
  (isWinningConfigurationForBlake [9, 3, 3]) ∧ 
  (isWinningConfigurationForBlake [9, 5, 2]) :=
by {
  sorry
}

end winning_configurations_for_blake_l224_224617


namespace area_PVZ_is_correct_l224_224393

noncomputable def area_triangle_PVZ : ℝ :=
  let PQ : ℝ := 8
  let QR : ℝ := 4
  let RV : ℝ := 2
  let WS : ℝ := 3
  let VW : ℝ := PQ - (RV + WS)  -- VW is calculated as 3
  let base_PV : ℝ := PQ
  let height_PVZ : ℝ := QR
  1 / 2 * base_PV * height_PVZ

theorem area_PVZ_is_correct : area_triangle_PVZ = 16 :=
  sorry

end area_PVZ_is_correct_l224_224393


namespace equilateral_triangle_isosceles_triangle_l224_224815

noncomputable def is_equilateral (a b c : ℝ) : Prop :=
  a = b ∧ b = c

noncomputable def is_isosceles (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ a = c

theorem equilateral_triangle (a b c : ℝ) (h : abs (a - b) + abs (b - c) = 0) : is_equilateral a b c :=
  sorry

theorem isosceles_triangle (a b c : ℝ) (h : (a - b) * (b - c) = 0) : is_isosceles a b c :=
  sorry

end equilateral_triangle_isosceles_triangle_l224_224815


namespace championship_outcome_count_l224_224604

theorem championship_outcome_count (students championships : ℕ) (h_students : students = 8) (h_championships : championships = 3) : students ^ championships = 512 := by
  rw [h_students, h_championships]
  norm_num

end championship_outcome_count_l224_224604


namespace average_of_first_6_numbers_l224_224130

-- Definitions extracted from conditions
def average_of_11_numbers := 60
def average_of_last_6_numbers := 65
def sixth_number := 258
def total_sum := 11 * average_of_11_numbers
def sum_of_last_6_numbers := 6 * average_of_last_6_numbers

-- Lean 4 statement for the proof problem
theorem average_of_first_6_numbers :
  (∃ A, 6 * A = (total_sum - (sum_of_last_6_numbers - sixth_number))) →
  (∃ A, 6 * A = 528) :=
by
  intro h
  exact h

end average_of_first_6_numbers_l224_224130


namespace constant_temperature_l224_224938

def stable_system (T : ℤ × ℤ × ℤ → ℝ) : Prop :=
  ∀ (a b c : ℤ), T (a, b, c) = (1 / 6) * (T (a + 1, b, c) + T (a - 1, b, c) + T (a, b + 1, c) + T (a, b - 1, c) + T (a, b, c + 1) + T (a, b, c - 1))

theorem constant_temperature (T : ℤ × ℤ × ℤ → ℝ) 
    (h1 : ∀ (x : ℤ × ℤ × ℤ), 0 ≤ T x ∧ T x ≤ 1)
    (h2 : stable_system T) : 
  ∃ c : ℝ, ∀ x : ℤ × ℤ × ℤ, T x = c := 
sorry

end constant_temperature_l224_224938


namespace cylindrical_container_invariant_volume_l224_224780

theorem cylindrical_container_invariant_volume {r h y : ℝ} (hr : r = 6) (hh : h = 5) :
    (∀ y > 0, π * (r + y)^2 * h - π * r^2 * h = π * r^2 * (h + y) - π * r^2 * h) → y = 2.16 :=
by
  intro hy
  sorry

end cylindrical_container_invariant_volume_l224_224780


namespace petya_numbers_l224_224120

theorem petya_numbers (S : Finset ℕ) (T : Finset ℕ) :
  S = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20} →
  T = S.filter (λ n, ¬ (Even n ∨ n % 5 = 4)) →
  T.card = 8 :=
by
  intros hS hT
  have : S = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19} := by sorry
  have : T = {1, 3, 5, 7, 11, 13, 15, 17} := by sorry
  sorry

end petya_numbers_l224_224120


namespace crowdfunding_successful_l224_224341

variable (highest_level second_level lowest_level total_amount : ℕ)
variable (x y z : ℕ)

noncomputable def crowdfunding_conditions (highest_level second_level lowest_level : ℕ) := 
  second_level = highest_level / 10 ∧ lowest_level = second_level / 10

noncomputable def total_raised (highest_level second_level lowest_level x y z : ℕ) :=
  highest_level * x + second_level * y + lowest_level * z

theorem crowdfunding_successful (h1 : highest_level = 5000) 
                                (h2 : crowdfunding_conditions highest_level second_level lowest_level) 
                                (h3 : total_amount = 12000) 
                                (h4 : y = 3) 
                                (h5 : z = 10) :
  total_raised highest_level second_level lowest_level x y z = total_amount → x = 2 := by
  sorry

end crowdfunding_successful_l224_224341


namespace pages_per_chapter_l224_224204

-- Definitions based on conditions
def chapters_in_book : ℕ := 2
def days_to_finish : ℕ := 664
def chapters_per_day : ℕ := 332
def total_chapters_read : ℕ := chapters_per_day * days_to_finish

-- Theorem that states the problem
theorem pages_per_chapter : total_chapters_read / chapters_in_book = 110224 :=
by
  -- Proof is omitted
  sorry

end pages_per_chapter_l224_224204


namespace max_mean_BC_l224_224028

theorem max_mean_BC (A_n B_n C_n A_total_weight B_total_weight C_total_weight : ℕ)
    (hA_mean : A_total_weight = 45 * A_n)
    (hB_mean : B_total_weight = 55 * B_n)
    (hAB_mean : (A_total_weight + B_total_weight) / (A_n + B_n) = 48)
    (hAC_mean : (A_total_weight + C_total_weight) / (A_n + C_n) = 50) :
    ∃ m : ℤ, m = 66 := by
  sorry

end max_mean_BC_l224_224028


namespace speed_of_man_rowing_upstream_l224_224786

theorem speed_of_man_rowing_upstream (V_m V_downstream V_upstream V_s : ℝ) 
  (h1 : V_m = 25) 
  (h2 : V_downstream = 38) :
  V_upstream = V_m - (V_downstream - V_m) :=
by
  sorry

end speed_of_man_rowing_upstream_l224_224786


namespace regular_polygon_sides_l224_224659

theorem regular_polygon_sides (n : ℕ) (h : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l224_224659


namespace cookies_per_person_l224_224621

theorem cookies_per_person (total_cookies : ℕ) (num_people : ℕ) (h1 : total_cookies = 35) (h2 : num_people = 5) :
  total_cookies / num_people = 7 := 
by {
  sorry
}

end cookies_per_person_l224_224621


namespace exists_same_color_points_one_meter_apart_l224_224305

-- Declare the colors as an enumeration
inductive Color
| red : Color
| black : Color

-- Define the function that assigns a color to each point in the plane
def color (point : ℝ × ℝ) : Color := sorry

-- The theorem to be proven
theorem exists_same_color_points_one_meter_apart :
  ∃ x y : ℝ × ℝ, x ≠ y ∧ dist x y = 1 ∧ color x = color y :=
sorry

end exists_same_color_points_one_meter_apart_l224_224305


namespace number_of_ways_to_divide_friends_l224_224839

theorem number_of_ways_to_divide_friends :
  (4 ^ 8 = 65536) := by
  -- result obtained through the multiplication principle
  sorry

end number_of_ways_to_divide_friends_l224_224839


namespace baseball_games_season_duration_l224_224011

theorem baseball_games_season_duration 
    (games_per_month : ℕ) 
    (games_per_season : ℕ) 
    (H1 : games_per_month = 7)
    (H2 : games_per_season = 14) :
    games_per_season / games_per_month = 2 :=
begin
  sorry
end

end baseball_games_season_duration_l224_224011


namespace average_pages_per_book_deshaun_l224_224497

-- Definitions related to the conditions
def summer_days : ℕ := 80
def deshaun_books : ℕ := 60
def person_closest_percentage : ℚ := 0.75
def second_person_daily_pages : ℕ := 180

-- Derived definitions
def second_person_total_pages : ℕ := second_person_daily_pages * summer_days
def deshaun_total_pages : ℚ := second_person_total_pages / person_closest_percentage

-- The final proof statement
theorem average_pages_per_book_deshaun : 
  deshaun_total_pages / deshaun_books = 320 := 
by
  -- We would provide the proof here
  sorry

end average_pages_per_book_deshaun_l224_224497


namespace scallop_cost_equivalence_l224_224469

-- Define given conditions
def scallops_per_pound := 8
def cost_per_pound := 24  -- in dollars
def scallops_per_person := 2
def number_of_people := 8

-- Define and prove the total cost of scallops for serving 8 people
theorem scallop_cost_equivalence (h1 : scallops_per_pound = 8) (h2 : cost_per_pound = 24)
(h3 : scallops_per_person = 2) (h4 : number_of_people = 8) :
  let total_scallops := number_of_people * scallops_per_person in
  let weight_in_pounds := total_scallops / scallops_per_pound in
  let total_cost := weight_in_pounds * cost_per_pound in
  total_cost = 48 :=
by
  sorry

end scallop_cost_equivalence_l224_224469


namespace carol_to_cathy_ratio_l224_224860

-- Define the number of cars owned by Cathy, Lindsey, Carol, and Susan
def cathy_cars : ℕ := 5
def lindsey_cars : ℕ := cathy_cars + 4
def carol_cars : ℕ := cathy_cars
def susan_cars : ℕ := carol_cars - 2

-- Define the total number of cars in the problem statement
def total_cars : ℕ := 32

-- Theorem to prove the ratio of Carol's cars to Cathy's cars is 1:1
theorem carol_to_cathy_ratio : carol_cars = cathy_cars := by
  sorry

end carol_to_cathy_ratio_l224_224860


namespace cos_of_60_degrees_is_half_l224_224625

noncomputable theory

def unit_circle_coordinates := (1/2 : ℝ, real.sqrt 3 / 2)

theorem cos_of_60_degrees_is_half :
  ∀ θ : ℝ, θ = 60 → unit_circle_coordinates.fst = real.cos θ :=
begin
  sorry
end

end cos_of_60_degrees_is_half_l224_224625


namespace susan_spending_ratio_l224_224577

theorem susan_spending_ratio (initial_amount clothes_spent books_left books_spent left_after_clothes gcd_ratio : ℤ)
  (h1 : initial_amount = 600)
  (h2 : clothes_spent = initial_amount / 2)
  (h3 : left_after_clothes = initial_amount - clothes_spent)
  (h4 : books_left = 150)
  (h5 : books_spent = left_after_clothes - books_left)
  (h6 : gcd books_spent left_after_clothes = 150)
  (h7 : books_spent / gcd_ratio = 1)
  (h8 : left_after_clothes / gcd_ratio = 2) :
  books_spent / gcd books_spent left_after_clothes = 1 ∧ left_after_clothes / gcd books_spent left_after_clothes = 2 :=
sorry

end susan_spending_ratio_l224_224577


namespace total_bills_combined_l224_224339

theorem total_bills_combined
  (a b c : ℝ)
  (H1 : 0.15 * a = 3)
  (H2 : 0.25 * b = 5)
  (H3 : 0.20 * c = 4) :
  a + b + c = 60 := 
sorry

end total_bills_combined_l224_224339


namespace freds_sister_borrowed_3_dimes_l224_224814

-- Define the conditions
def original_dimes := 7
def remaining_dimes := 4

-- Define the question and answer
def borrowed_dimes := original_dimes - remaining_dimes

-- Statement to prove
theorem freds_sister_borrowed_3_dimes : borrowed_dimes = 3 := by
  sorry

end freds_sister_borrowed_3_dimes_l224_224814


namespace brother_growth_is_one_l224_224290

-- Define measurements related to Stacy's height.
def Stacy_previous_height : ℕ := 50
def Stacy_current_height : ℕ := 57

-- Define the condition that Stacy's growth is 6 inches more than her brother's growth.
def Stacy_growth := Stacy_current_height - Stacy_previous_height
def Brother_growth := Stacy_growth - 6

-- Prove that Stacy's brother grew 1 inch.
theorem brother_growth_is_one : Brother_growth = 1 :=
by
  sorry

end brother_growth_is_one_l224_224290


namespace remaining_numbers_count_l224_224124

-- Defining the initial set from 1 to 20
def initial_set : finset ℕ := finset.range 21 \ {0}

-- Condition: Erase all even numbers
def without_even : finset ℕ := initial_set.filter (λ n, n % 2 ≠ 0)

-- Condition: Erase numbers that give a remainder of 4 when divided by 5 from the remaining set
def final_set : finset ℕ := without_even.filter (λ n, n % 5 ≠ 4)

-- Statement to prove
theorem remaining_numbers_count : final_set.card = 8 :=
by
  -- admitting the proof
  sorry

end remaining_numbers_count_l224_224124


namespace correct_transformation_l224_224129

theorem correct_transformation (x : ℝ) :
  (6 * ((2 * x + 1) / 3) - 6 * ((10 * x + 1) / 6) = 6) ↔ (4 * x + 2 - 10 * x - 1 = 6) :=
by
  sorry

end correct_transformation_l224_224129


namespace polar_coordinates_to_rectangular_l224_224191

noncomputable def polar_to_rectangular (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem polar_coordinates_to_rectangular :
  polar_to_rectangular 10 (11 * Real.pi / 6) = (5 * Real.sqrt 3, -5) :=
by
  sorry

end polar_coordinates_to_rectangular_l224_224191


namespace hyperbola_equation_sum_l224_224391

theorem hyperbola_equation_sum (h k a c b : ℝ) (h_h : h = 1) (h_k : k = 1) (h_a : a = 3) (h_c : c = 9) (h_c2 : c^2 = a^2 + b^2) :
    h + k + a + b = 5 + 6 * Real.sqrt 2 :=
by
  sorry

end hyperbola_equation_sum_l224_224391


namespace find_ending_number_of_range_l224_224297

theorem find_ending_number_of_range :
  ∃ n : ℕ, (∀ avg_200_400 avg_100_n : ℕ,
    avg_200_400 = (200 + 400) / 2 ∧
    avg_100_n = (100 + n) / 2 ∧
    avg_100_n + 150 = avg_200_400) ∧
    n = 200 :=
sorry

end find_ending_number_of_range_l224_224297


namespace total_tires_parking_lot_l224_224538

-- Definitions for each condition in a)
def four_wheel_drive_cars := 30
def motorcycles := 20
def six_wheel_trucks := 10
def bicycles := 5
def unicycles := 3
def baby_strollers := 2

def extra_roof_tires := 4
def flat_bike_tires_removed := 3
def extra_unicycle_wheel := 1

def tires_per_car := 4 + 1
def tires_per_motorcycle := 2 + 2
def tires_per_truck := 6 + 1
def tires_per_bicycle := 2
def tires_per_unicycle := 1
def tires_per_stroller := 4

-- Define total tires calculation
def total_tires (four_wheel_drive_cars motorcycles six_wheel_trucks bicycles unicycles baby_strollers 
                 extra_roof_tires flat_bike_tires_removed extra_unicycle_wheel : ℕ) :=
  (four_wheel_drive_cars * tires_per_car + extra_roof_tires) +
  (motorcycles * tires_per_motorcycle) +
  (six_wheel_trucks * tires_per_truck) +
  (bicycles * tires_per_bicycle - flat_bike_tires_removed) +
  (unicycles * tires_per_unicycle + extra_unicycle_wheel) +
  (baby_strollers * tires_per_stroller)

-- The Lean statement for the proof problem
theorem total_tires_parking_lot : 
  total_tires four_wheel_drive_cars motorcycles six_wheel_trucks bicycles unicycles baby_strollers 
              extra_roof_tires flat_bike_tires_removed extra_unicycle_wheel = 323 :=
by 
  sorry

end total_tires_parking_lot_l224_224538


namespace min_value_inverse_sum_l224_224218

theorem min_value_inverse_sum (a b : ℝ) (h : a > 0) (k : b > 0) (hab : a + 2 * b = 1) : 
  ∃ (y : ℝ), y = 3 + 2 * Real.sqrt 2 ∧ (∀ x, x = (1 / a) + (1 / b) → y ≤ x) :=
sorry

end min_value_inverse_sum_l224_224218


namespace rhombus_area_from_quadratic_roots_l224_224369

theorem rhombus_area_from_quadratic_roots :
  let eq := λ x : ℝ, x^2 - 10 * x + 24 = 0
  ∃ (d1 d2 : ℝ), eq d1 ∧ eq d2 ∧ d1 ≠ d2 ∧ (1/2) * d1 * d2 = 12 :=
by
  sorry

end rhombus_area_from_quadratic_roots_l224_224369


namespace hyperbola_condition_l224_224702

theorem hyperbola_condition (k : ℝ) : 
  (∃ (x y : ℝ), (x^2 / (4 + k) + y^2 / (1 - k) = 1)) ↔ (k < -4 ∨ k > 1) :=
by 
  sorry

end hyperbola_condition_l224_224702


namespace net_sales_revenue_l224_224343

-- Definition of the conditions
def regression (x : ℝ) : ℝ := 8.5 * x + 17.5

-- Statement of the theorem
theorem net_sales_revenue (x : ℝ) (h : x = 10) : (regression x - x) = 92.5 :=
by {
  -- No proof required as per instruction; use sorry.
  sorry
}

end net_sales_revenue_l224_224343


namespace number_of_ways_to_divide_friends_l224_224837

theorem number_of_ways_to_divide_friends :
  (4 ^ 8 = 65536) := by
  -- result obtained through the multiplication principle
  sorry

end number_of_ways_to_divide_friends_l224_224837


namespace gcd_282_470_l224_224468

theorem gcd_282_470 : Int.gcd 282 470 = 94 :=
by
  sorry

end gcd_282_470_l224_224468


namespace greatest_five_consecutive_odd_integers_l224_224319

theorem greatest_five_consecutive_odd_integers (A B C D E : ℤ) (x : ℤ) 
  (h1 : B = x + 2) 
  (h2 : C = x + 4)
  (h3 : D = x + 6)
  (h4 : E = x + 8)
  (h5 : A + B + C + D + E = 148) :
  E = 33 :=
by {
  sorry -- proof not required
}

end greatest_five_consecutive_odd_integers_l224_224319


namespace total_number_of_digits_l224_224741

theorem total_number_of_digits (n S S₅ S₃ : ℕ) (h1 : S = 20 * n) (h2 : S₅ = 5 * 12) (h3 : S₃ = 3 * 33) : n = 8 :=
by
  sorry

end total_number_of_digits_l224_224741


namespace select_four_person_committee_l224_224789

open Nat

theorem select_four_person_committee 
  (n : ℕ)
  (h1 : (n * (n - 1) * (n - 2)) / 6 = 21) 
  : (n = 9) → Nat.choose n 4 = 126 :=
by
  sorry

end select_four_person_committee_l224_224789


namespace area_to_be_painted_correct_l224_224863

-- Define the dimensions and areas involved
def wall_height : ℕ := 10
def wall_length : ℕ := 15
def window_height : ℕ := 3
def window_length : ℕ := 5
def painting_height : ℕ := 2
def painting_length : ℕ := 2

-- Calculate the areas
def wall_area : ℕ := wall_height * wall_length
def window_area : ℕ := window_height * window_length
def painting_area : ℕ := painting_height * painting_length
def area_not_painted : ℕ := window_area + painting_area
def area_to_be_painted : ℕ := wall_area - area_not_painted

-- Theorem: The area to be painted is 131 square feet
theorem area_to_be_painted_correct : area_to_be_painted = 131 := by
  sorry

end area_to_be_painted_correct_l224_224863


namespace max_daily_sales_revenue_l224_224000

noncomputable def P (t : ℕ) : ℕ :=
  if 1 ≤ t ∧ t ≤ 24 then t + 2 else if 25 ≤ t ∧ t ≤ 30 then 100 - t else 0

noncomputable def Q (t : ℕ) : ℕ :=
  if 1 ≤ t ∧ t ≤ 30 then 40 - t else 0

noncomputable def y (t : ℕ) : ℕ :=
  P t * Q t

theorem max_daily_sales_revenue :
  ∃ t : ℕ, 1 ≤ t ∧ t ≤ 30 ∧ y t = 115 :=
sorry

end max_daily_sales_revenue_l224_224000


namespace sum_leq_six_of_quadratic_roots_l224_224523

theorem sum_leq_six_of_quadratic_roots (a b : ℤ) (h1 : a ≠ -1) (h2 : b ≠ -1) 
  (h3 : ∃ r1 r2 : ℤ, r1 ≠ r2 ∧ x^2 + ab * x + (a + b) = 0 ∧ 
         x = r1 ∧ x = r2) : a + b ≤ 6 :=
by
  sorry

end sum_leq_six_of_quadratic_roots_l224_224523


namespace work_increase_percent_l224_224389

theorem work_increase_percent (W p : ℝ) (p_pos : p > 0) :
  (1 / 3 * p) * W / ((2 / 3) * p) - (W / p) = 0.5 * (W / p) :=
by
  sorry

end work_increase_percent_l224_224389


namespace find_a_plus_k_l224_224114

variable (a k : ℝ)

noncomputable def f (x : ℝ) : ℝ := (a - 1) * x^k

theorem find_a_plus_k
  (h1 : f a k (Real.sqrt 2) = 2)
  (h2 : (Real.sqrt 2)^2 = 2) : a + k = 4 := 
sorry

end find_a_plus_k_l224_224114


namespace break_even_price_l224_224737

noncomputable def initial_investment : ℝ := 1500
noncomputable def cost_per_tshirt : ℝ := 3
noncomputable def num_tshirts_break_even : ℝ := 83
noncomputable def total_cost_equipment_tshirts : ℝ := initial_investment + (cost_per_tshirt * num_tshirts_break_even)
noncomputable def price_per_tshirt := total_cost_equipment_tshirts / num_tshirts_break_even

theorem break_even_price : price_per_tshirt = 21.07 := by
  sorry

end break_even_price_l224_224737


namespace problem_1_problem_2_problem_3_l224_224301

noncomputable def f (x : ℝ) : ℝ := Real.logb 2 (2^x + 1)
noncomputable def f_inv (x : ℝ) : ℝ := Real.logb 2 (2^x - 1)

theorem problem_1 : 
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ f_inv x = m + f x) ↔ 
  m ∈ (Set.Icc (Real.logb 2 (1/3)) (Real.logb 2 (3/5))) :=
sorry

theorem problem_2 : 
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ f_inv x > m + f x) ↔ 
  m ∈ (Set.Iio (Real.logb 2 (3/5))) :=
sorry

theorem problem_3 : 
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f_inv x > m + f x) ↔ 
  m ∈ (Set.Iio (Real.logb 2 (1/3))) :=
sorry

end problem_1_problem_2_problem_3_l224_224301


namespace rhombus_diagonal_BD_equation_rhombus_diagonal_AD_equation_l224_224990

theorem rhombus_diagonal_BD_equation (A C : ℝ × ℝ) (AB_eq : ∀ x y : ℝ, 3 * x - y + 2 = 0) : 
  A = (0, 2) ∧ C = (4, 6) → ∃ k b : ℝ, k = 1 ∧ b = 6 ∧ ∀ x y : ℝ, x + y - 6 = 0 := by
  sorry

theorem rhombus_diagonal_AD_equation (A C : ℝ × ℝ) (AB_eq BD_eq : ∀ x y : ℝ, 3 * x - y + 2 = 0 ∧ x + y - 6 = 0) : 
  A = (0, 2) ∧ C = (4, 6) → ∃ k b : ℝ, k = 3 ∧ b = 14 ∧ ∀ x y : ℝ, x - 3 * y + 14 = 0 := by
  sorry

end rhombus_diagonal_BD_equation_rhombus_diagonal_AD_equation_l224_224990


namespace triangle_max_area_l224_224526

noncomputable def max_triangle_area {A B C : ℝ × ℝ} 
  (hABC_inscribed : ∃ x y, (x, y) = A ∨ (x, y) = B ∨ (x, y) = C ∧ 
    (x ^ 2 / 9) + (y ^ 2 / 4) = 1)
  (hAB_passing_through : ∃ P : ℝ × ℝ, P = (1, 0) ∧ collinear A B P) : ℝ :=
  (16 * Real.sqrt 2) / 3

theorem triangle_max_area
  (A B C : ℝ × ℝ)
  (hABC_inscribed : ∃ x y, ((x, y) = A ∨ (x, y) = B ∨ (x, y) = C) ∧ 
    (x ^ 2 / 9) + (y ^ 2 / 4) = 1)
  (hAB_passing_through : (∃ P : ℝ × ℝ, P = (1, 0) ∧ collinear A B P)) :
  max_triangle_area hABC_inscribed hAB_passing_through = (16 * Real.sqrt 2) / 3 :=
sorry

end triangle_max_area_l224_224526


namespace log_sum_correct_l224_224155

noncomputable def log_sum : Prop :=
  let x := (3/2)
  let y := (5/3)
  (x + y) = (19/6)

theorem log_sum_correct : log_sum :=
by
  sorry

end log_sum_correct_l224_224155


namespace correct_calculation_given_conditions_l224_224916

variable (number : ℤ)

theorem correct_calculation_given_conditions 
  (h : number + 16 = 64) : number - 16 = 32 := by
  sorry

end correct_calculation_given_conditions_l224_224916


namespace fraction_power_mult_equality_l224_224802

-- Define the fraction and the power
def fraction := (1 : ℚ) / 3
def power : ℚ := fraction ^ 4

-- Define the multiplication
def result := 8 * power

-- Prove the equality
theorem fraction_power_mult_equality : result = 8 / 81 := by
  sorry

end fraction_power_mult_equality_l224_224802


namespace women_at_each_table_l224_224045

theorem women_at_each_table (W : ℕ) (h1 : ∃ W, ∀ i : ℕ, (i < 7) → W + 2 = 7 * W + 14) (h2 : 7 * W + 14 = 63) : W = 7 :=
by
  sorry

end women_at_each_table_l224_224045


namespace intercept_sum_equation_l224_224512

theorem intercept_sum_equation (c : ℝ) (h₀ : 3 * x + 4 * y + c = 0)
  (h₁ : (-(c / 3)) + (-(c / 4)) = 28) : c = -48 := 
by
  sorry

end intercept_sum_equation_l224_224512


namespace twenty_five_percent_of_five_hundred_l224_224809

theorem twenty_five_percent_of_five_hundred : 0.25 * 500 = 125 := 
by 
  sorry

end twenty_five_percent_of_five_hundred_l224_224809


namespace gcd_lcm_240_l224_224911

theorem gcd_lcm_240 (a b : ℕ) (h : Nat.gcd a b * Nat.lcm a b = 240) : 
  ∃ n, ∃ gcds : Finset ℕ, (gcds.card = n) ∧ (Nat.gcd a b ∈ gcds) :=
by
  sorry

end gcd_lcm_240_l224_224911


namespace inequality_proof_l224_224688

theorem inequality_proof
  (x y : ℝ)
  (h : x^8 + y^8 ≤ 1) :
  x^12 - y^12 + 2 * x^6 * y^6 ≤ π / 2 :=
sorry

end inequality_proof_l224_224688


namespace function_characterization_l224_224517

noncomputable def f : ℝ → ℝ := sorry

theorem function_characterization (f : ℝ → ℝ) (k : ℝ) :
  (∀ x y : ℝ, f (x^2 + 2*x*y + y^2) = (x + y) * (f x + f y)) →
  (∀ x : ℝ, |f x - k * x| ≤ |x^2 - x|) →
  ∀ x : ℝ, f x = k * x :=
by
  sorry

end function_characterization_l224_224517


namespace equidistant_trajectory_l224_224745

theorem equidistant_trajectory {x y z : ℝ} :
  (x + 1)^2 + (y - 1)^2 + z^2 = (x - 2)^2 + (y + 1)^2 + (z + 1)^2 →
  3 * x - 2 * y - z = 2 :=
sorry

end equidistant_trajectory_l224_224745


namespace average_sequence_x_l224_224132

theorem average_sequence_x (x : ℚ) (h : (5050 + x) / 101 = 50 * x) : x = 5050 / 5049 :=
by
  sorry

end average_sequence_x_l224_224132


namespace indira_cricket_minutes_l224_224426

def totalMinutesSeanPlayed (sean_minutes_per_day : ℕ) (days : ℕ) : ℕ :=
  sean_minutes_per_day * days

def totalMinutesIndiraPlayed (total_minutes_together : ℕ) (total_minutes_sean : ℕ) : ℕ :=
  total_minutes_together - total_minutes_sean

theorem indira_cricket_minutes :
  totalMinutesIndiraPlayed 1512 (totalMinutesSeanPlayed 50 14) = 812 :=
by
  sorry

end indira_cricket_minutes_l224_224426


namespace geometric_sequence_sufficient_not_necessary_l224_224820

theorem geometric_sequence_sufficient_not_necessary (a b c : ℝ) :
  (∃ r : ℝ, a = b * r ∧ b = c * r) → (b^2 = a * c) ∧ ¬ ( (b^2 = a * c) → (∃ r : ℝ, a = b * r ∧ b = c * r) ) :=
by
  sorry

end geometric_sequence_sufficient_not_necessary_l224_224820


namespace point_translation_l224_224541

theorem point_translation :
  ∃ (x_old y_old x_new y_new : ℤ),
  (x_old = 1 ∧ y_old = -2) ∧
  (x_new = x_old + 2) ∧
  (y_new = y_old + 3) ∧
  (x_new = 3) ∧
  (y_new = 1) :=
sorry

end point_translation_l224_224541


namespace like_term_exists_l224_224870

variable (a b : ℝ) (x y : ℝ)

theorem like_term_exists : ∃ b : ℝ, b * x^5 * y^3 = 3 * x^5 * y^3 ∧ b ≠ a :=
by
  -- existence of b
  use 3
  -- proof is omitted
  sorry

end like_term_exists_l224_224870


namespace cos_60_eq_sqrt3_div_2_l224_224629

theorem cos_60_eq_sqrt3_div_2 : Real.cos (60 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry

end cos_60_eq_sqrt3_div_2_l224_224629


namespace sin_pi_div_2_minus_pi_div_6_l224_224899

noncomputable def sin_diff (α β : ℝ) : ℝ := Real.sin (α - β)

theorem sin_pi_div_2_minus_pi_div_6 : sin_diff (Real.pi / 2) (Real.pi / 6) = Real.sqrt 3 / 2 := by
  sorry

end sin_pi_div_2_minus_pi_div_6_l224_224899


namespace inequality_am_gm_holds_l224_224818

theorem inequality_am_gm_holds 
    (a b c : ℝ) 
    (ha : a > 0) 
    (hb : b > 0) 
    (hc : c > 0) 
    (h : a^3 + b^3 = c^3) : 
  a^2 + b^2 - c^2 > 6 * (c - a) * (c - b) := 
sorry

end inequality_am_gm_holds_l224_224818


namespace expand_and_simplify_l224_224356

theorem expand_and_simplify (x : ℝ) : 3 * (x - 3) * (x + 10) + 2 * x = 3 * x^2 + 23 * x - 90 :=
by
  sorry

end expand_and_simplify_l224_224356


namespace find_constant_l224_224932

-- Define the relationship between Fahrenheit and Celsius
def temp_rel (c f k : ℝ) : Prop :=
  f = (9 / 5) * c + k

-- Temperature increases
def temp_increase (c1 c2 f1 f2 : ℝ) : Prop :=
  (f2 - f1 = 30) ∧ (c2 - c1 = 16.666666666666668)

-- Freezing point condition
def freezing_point (f : ℝ) : Prop :=
  f = 32

-- Main theorem to prove
theorem find_constant (k : ℝ) :
  ∃ (c1 c2 f1 f2: ℝ), temp_rel c1 f1 k ∧ temp_rel c2 f2 k ∧ 
  temp_increase c1 c2 f1 f2 ∧ freezing_point f1 → k = 32 :=
by sorry

end find_constant_l224_224932


namespace new_daily_average_wage_l224_224388

theorem new_daily_average_wage (x : ℝ) : 
  (∀ y : ℝ, 25 - x = y) → 
  (∀ z : ℝ, 20 * (25 - x) = 30 * (10)) → 
  x = 10 :=
by
  intro h1 h2
  sorry

end new_daily_average_wage_l224_224388


namespace raghu_investment_approx_l224_224017

-- Define the investments
def investments (R : ℝ) : Prop :=
  let Trishul := 0.9 * R
  let Vishal := 0.99 * R
  let Deepak := 1.188 * R
  R + Trishul + Vishal + Deepak = 8578

-- State the theorem to prove that Raghu invested approximately Rs. 2103.96
theorem raghu_investment_approx : 
  ∃ R : ℝ, investments R ∧ abs (R - 2103.96) < 1 :=
by
  sorry

end raghu_investment_approx_l224_224017


namespace general_formula_l224_224827

open Nat

def a (n : ℕ) : ℚ :=
  if n = 0 then 7/6 else 0 -- Recurrence initialization with dummy else condition

-- Defining the recurrence relation as a function
lemma recurrence_relation {n : ℕ} (h : n > 0) : 
    a n = (1 / 2) * a (n - 1) + (1 / 3) := 
sorry

-- Proof of the general formula
theorem general_formula (n : ℕ) : a n = (1 / (2^n : ℚ)) + (2 / 3) :=
sorry

end general_formula_l224_224827


namespace remaining_numbers_after_erasure_l224_224121

theorem remaining_numbers_after_erasure :
  let initial_list := list.range 20
  let odd_numbers := list.filter (λ x, ¬ (x % 2 = 0)) initial_list
  let final_list := list.filter (λ x, ¬ (x % 5 = 4)) odd_numbers
  list.length final_list = 8 :=
by
  let initial_list := list.range 20
  let odd_numbers := list.filter (λ x, ¬ (x % 2 = 0)) initial_list
  let final_list := list.filter (λ x, ¬ (x % 5 = 4)) odd_numbers
  show list.length final_list = 8 from sorry

end remaining_numbers_after_erasure_l224_224121


namespace number_of_boys_is_12500_l224_224805

-- Define the number of boys and girls in the school
def numberOfBoys (B : ℕ) : ℕ := B
def numberOfGirls : ℕ := 5000

-- Define the total attendance
def totalAttendance (B : ℕ) : ℕ := B + numberOfGirls

-- Define the condition for the percentage increase from boys to total attendance
def percentageIncreaseCondition (B : ℕ) : Prop :=
  totalAttendance B = B + Int.ofNat numberOfGirls

-- Statement to prove
theorem number_of_boys_is_12500 (B : ℕ) (h : totalAttendance B = B + numberOfGirls) : B = 12500 :=
sorry

end number_of_boys_is_12500_l224_224805


namespace no_intersection_points_l224_224647

theorem no_intersection_points :
  ¬ ∃ x y : ℝ, y = |3 * x + 4| ∧ y = -|2 * x + 1| :=
by
  sorry

end no_intersection_points_l224_224647


namespace total_worth_all_crayons_l224_224118

def cost_of_crayons (packs: ℕ) (cost_per_pack: ℝ) : ℝ := packs * cost_per_pack

def discounted_cost (cost: ℝ) (discount_rate: ℝ) : ℝ := cost * (1 - discount_rate)

def tax_amount (cost: ℝ) (tax_rate: ℝ) : ℝ := cost * tax_rate

theorem total_worth_all_crayons : 
  let cost_per_pack := 2.5
  let discount_rate := 0.15
  let tax_rate := 0.07
  let packs_already_have := 4
  let packs_to_buy := 2
  let cost_two_packs := cost_of_crayons packs_to_buy cost_per_pack
  let discounted_two_packs := discounted_cost cost_two_packs discount_rate
  let tax_two_packs := tax_amount cost_two_packs tax_rate
  let total_cost_two_packs := discounted_two_packs + tax_two_packs
  let cost_four_packs := cost_of_crayons packs_already_have cost_per_pack
  cost_four_packs + total_cost_two_packs = 14.60 := 
by 
  sorry

end total_worth_all_crayons_l224_224118


namespace increasing_decreasing_intervals_l224_224807

noncomputable def f (x : ℝ) : ℝ := Real.sin (-2 * x + 3 * Real.pi / 4)

theorem increasing_decreasing_intervals : (∀ k : ℤ, 
    ∀ x, 
      ((k : ℝ) * Real.pi + 5 * Real.pi / 8 ≤ x ∧ x ≤ (k : ℝ) * Real.pi + 9 * Real.pi / 8) 
      → 0 < f x ∧ f x < 1) 
  ∧ 
    (∀ k : ℤ, 
    ∀ x, 
      ((k : ℝ) * Real.pi + Real.pi / 8 ≤ x ∧ x ≤ (k : ℝ) * Real.pi + 5 * Real.pi / 8) 
      → -1 < f x ∧ f x < 0) :=
by
  sorry

end increasing_decreasing_intervals_l224_224807


namespace like_terms_product_l224_224214

theorem like_terms_product :
  ∀ (m n : ℕ),
    (-x^3 * y^n) = (3 * x^m * y^2) → (m = 3 ∧ n = 2) → m * n = 6 :=
by
  intros m n h1 h2
  sorry

end like_terms_product_l224_224214


namespace smallest_whole_number_larger_than_triangle_perimeter_l224_224154

theorem smallest_whole_number_larger_than_triangle_perimeter :
  (∀ s : ℝ, 16 < s ∧ s < 30 → ∃ n : ℕ, n = 60) :=
by
  sorry

end smallest_whole_number_larger_than_triangle_perimeter_l224_224154


namespace find_m_in_arith_seq_l224_224817

noncomputable def arith_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

theorem find_m_in_arith_seq (a : ℕ → ℝ) (d : ℝ) (h_d : d ≠ 0) 
  (h_seq : arith_seq a d) 
  (h_sum : a 3 + a 6 + a 10 + a 13 = 32) 
  (h_am : ∃ m, a m = 8) : 
  ∃ m, m = 8 := 
sorry

end find_m_in_arith_seq_l224_224817


namespace cos_60_eq_one_half_l224_224639

theorem cos_60_eq_one_half : Real.cos (60 * Real.pi / 180) = 1 / 2 :=
by
  -- Proof steps would go here
  sorry

end cos_60_eq_one_half_l224_224639


namespace maya_total_pages_read_l224_224417

def last_week_books : ℕ := 5
def pages_per_book : ℕ := 300
def this_week_multiplier : ℕ := 2

theorem maya_total_pages_read : 
  (last_week_books * pages_per_book * (1 + this_week_multiplier)) = 4500 :=
by
  sorry

end maya_total_pages_read_l224_224417


namespace tangent_line_to_curve_l224_224096

section TangentLine

variables {x m : ℝ}

theorem tangent_line_to_curve (x0 : ℝ) :
  (∀ x : ℝ, x > 0 → y = x * Real.log x) →
  (∀ x : ℝ, y = 2 * x + m) →
  (x0 > 0) →
  (x0 * Real.log x0 = 2 * x0 + m) →
  m = -Real.exp 1 :=
by
  sorry

end TangentLine

end tangent_line_to_curve_l224_224096


namespace raft_drift_time_l224_224004

-- Define the conditions from the problem
variable (distance : ℝ := 1)
variable (steamboat_time : ℝ := 1) -- in hours
variable (motorboat_time : ℝ := 3 / 4) -- 45 minutes in hours
variable (motorboat_speed_ratio : ℝ := 2)

-- Variables for speeds
variable (vs vf : ℝ)

-- Conditions: the speeds and conditions of traveling from one village to another
variable (steamboat_eqn : vs + vf = distance / steamboat_time := by sorry)
variable (motorboat_eqn : (2 * vs) + vf = distance / motorboat_time := by sorry)

-- Time for the raft to travel the distance
theorem raft_drift_time : 90 = (distance / vf) * 60 := by
  -- Proof comes here
  sorry

end raft_drift_time_l224_224004


namespace regular_polygon_sides_l224_224664

theorem regular_polygon_sides (angle : ℝ) (h1 : angle = 150) : ∃ n : ℕ, (180 * (n - 2)) / n = angle ∧ n = 12 :=
by
  sorry

end regular_polygon_sides_l224_224664


namespace total_coins_l224_224572

theorem total_coins (q1 q2 q3 q4 : Nat) (d1 d2 d3 : Nat) (n1 n2 : Nat) (p1 p2 p3 p4 p5 : Nat) :
  q1 = 8 → q2 = 6 → q3 = 7 → q4 = 5 →
  d1 = 7 → d2 = 5 → d3 = 9 →
  n1 = 4 → n2 = 6 →
  p1 = 10 → p2 = 3 → p3 = 8 → p4 = 2 → p5 = 13 →
  q1 + q2 + q3 + q4 + d1 + d2 + d3 + n1 + n2 + p1 + p2 + p3 + p4 + p5 = 93 :=
by
  intros
  sorry

end total_coins_l224_224572


namespace least_8_heavy_three_digit_l224_224612

def is_8_heavy (n : ℕ) : Prop :=
  n % 8 > 6

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

theorem least_8_heavy_three_digit : ∃ n : ℕ, is_three_digit n ∧ is_8_heavy n ∧ ∀ m : ℕ, is_three_digit m ∧ is_8_heavy m → n ≤ m := 
sorry

end least_8_heavy_three_digit_l224_224612


namespace least_clock_equivalent_hour_l224_224277

theorem least_clock_equivalent_hour (h : ℕ) (h_gt_9 : h > 9) (clock_equiv : (h^2 - h) % 12 = 0) : h = 13 :=
sorry

end least_clock_equivalent_hour_l224_224277


namespace average_difference_l224_224131

-- Definitions for the conditions
def set1 : List ℕ := [20, 40, 60]
def set2 : List ℕ := [10, 60, 35]

-- Function to compute the average of a list of numbers
def average (lst : List ℕ) : ℚ :=
  lst.sum / lst.length

-- The main theorem to prove the difference between the averages is 5
theorem average_difference : average set1 - average set2 = 5 := by
  sorry

end average_difference_l224_224131


namespace cone_lateral_area_l224_224706

theorem cone_lateral_area (r l : ℝ) (h_r : r = 3) (h_l : l = 5) : 
  π * r * l = 15 * π := by
  sorry

end cone_lateral_area_l224_224706


namespace bounded_harmonic_is_constant_l224_224414

noncomputable def is_harmonic (f : ℤ × ℤ → ℝ) : Prop :=
  ∀ (x y : ℤ), f (x+1, y) + f (x-1, y) + f (x, y+1) + f (x, y-1) = 4 * f (x, y)

theorem bounded_harmonic_is_constant (f : ℤ × ℤ → ℝ) (M : ℝ) 
  (h_bound : ∀ (x y : ℤ), |f (x, y)| ≤ M)
  (h_harmonic : is_harmonic f) :
  ∃ c : ℝ, ∀ x y : ℤ, f (x, y) = c :=
sorry

end bounded_harmonic_is_constant_l224_224414


namespace number_of_sides_of_regular_polygon_l224_224654

theorem number_of_sides_of_regular_polygon (n : ℕ) (h : 0 < n) (h_angle : ∀ i, i < n → (2 * n - 4) * 90 / n = 150) : n = 12 :=
sorry

end number_of_sides_of_regular_polygon_l224_224654


namespace susie_rooms_l224_224753

-- Define the conditions
def vacuum_time_per_room : ℕ := 20  -- in minutes
def total_vacuum_time : ℕ := 2 * 60  -- 2 hours in minutes

-- Define the number of rooms in Susie's house
def number_of_rooms (total_time room_time : ℕ) : ℕ := total_time / room_time

-- Prove that Susie has 6 rooms in her house
theorem susie_rooms : number_of_rooms total_vacuum_time vacuum_time_per_room = 6 :=
by
  sorry -- proof goes here

end susie_rooms_l224_224753


namespace fill_digits_subtraction_correct_l224_224507

theorem fill_digits_subtraction_correct :
  ∀ (A B : ℕ), A236 - (B*100 + 97) = 5439 → A = 6 ∧ B = 7 :=
by
  sorry

end fill_digits_subtraction_correct_l224_224507


namespace range_f_in_interval_l224_224227

-- Define the function f and the interval
def f (x : ℝ) (f_deriv_neg1 : ℝ) := x^3 + 2 * x * f_deriv_neg1
def interval := Set.Icc (-2 : ℝ) (3 : ℝ)

-- State the theorem
theorem range_f_in_interval :
  ∃ (f_deriv_neg1 : ℝ),
  (∀ x ∈ interval, f x f_deriv_neg1 ∈ Set.Icc (-4 * Real.sqrt 2) 9) :=
sorry

end range_f_in_interval_l224_224227


namespace ratio_of_x_to_y_l224_224458

theorem ratio_of_x_to_y (x y : ℚ) (h : (8*x - 5*y)/(10*x - 3*y) = 4/7) : x/y = 23/16 :=
by 
  sorry

end ratio_of_x_to_y_l224_224458


namespace count_3_digit_numbers_divisible_by_5_l224_224968

theorem count_3_digit_numbers_divisible_by_5 : 
  let three_digit_numbers := {n : ℕ | 100 ≤ n ∧ n ≤ 999}
  let divisible_by_5 := {n : ℕ | n % 5 = 0}
  let count := {n : ℕ | n ∈ three_digit_numbers ∧ n ∈ divisible_by_5}.card
  count = 180 :=
by
  sorry

end count_3_digit_numbers_divisible_by_5_l224_224968


namespace square_sum_l224_224232

theorem square_sum (a b : ℝ) (h1 : a + b = 8) (h2 : a * b = -2) : a^2 + b^2 = 68 := 
by 
  sorry

end square_sum_l224_224232


namespace max_gcd_a_is_25_l224_224749

-- Define the sequence a_n
def a (n : ℕ) : ℕ := 100 + n^2 + 2 * n

-- Define the gcd function
def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

-- Define the theorem to prove the maximum value of d_n as 25
theorem max_gcd_a_is_25 : ∃ n : ℕ, d n = 25 := 
sorry

end max_gcd_a_is_25_l224_224749


namespace indira_cricket_minutes_l224_224425

theorem indira_cricket_minutes (sean_minutes_per_day : ℕ) (days : ℕ) (total_minutes : ℕ) (sean_total_minutes : ℕ) (sean_indira_total : ℕ) :
  sean_minutes_per_day = 50 →
  days = 14 →
  total_minutes = sean_minutes_per_day * days →
  sean_indira_total = 1512 →
  sean_total_minutes = total_minutes →
  ∃ indira_minutes : ℕ, indira_minutes = sean_indira_total - sean_total_minutes ∧ indira_minutes = 812 := 
by
  intros 
  use 812
  split
  { rw [←a_5, ←a_4, ←a_3, a_1, a_2]
    norm_num}
  { refl }

end indira_cricket_minutes_l224_224425


namespace problem_l224_224213

theorem problem {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a^2 + b^2 + c^2 + a*b*c = 4) : 
  a + b + c ≤ 3 := 
sorry

end problem_l224_224213


namespace find_value_of_a_l224_224292

theorem find_value_of_a (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
(h_eq : 7 * a^2 + 14 * a * b = a^3 + 2 * a^2 * b) : a = 7 := 
sorry

end find_value_of_a_l224_224292


namespace triangle_sides_l224_224318
-- Import the entire library mainly used for geometry and algebraic proofs.

-- Define the main problem statement as a theorem.
theorem triangle_sides (a b c : ℕ) (r_incircle : ℕ)
  (r_excircle_a r_excircle_b r_excircle_c : ℕ) (s : ℕ)
  (area : ℕ) : 
  r_incircle = 1 → 
  area = s →
  r_excircle_a * r_excircle_b * r_excircle_c = (s * s * s) →
  s = (a + b + c) / 2 →
  r_excircle_a = s / (s - a) →
  r_excircle_b = s / (s - b) →
  r_excircle_c = s / (s - c) →
  a * b = 12 → 
  a = 3 ∧ b = 4 ∧ c = 5 :=
by {
  -- Placeholder for the proof.
  sorry
}

end triangle_sides_l224_224318


namespace intersection_M_N_l224_224083

def M := { x : ℝ | -1 < x ∧ x < 2 }
def N := { x : ℝ | x ≤ 1 }
def expectedIntersection := { x : ℝ | -1 < x ∧ x ≤ 1 }

theorem intersection_M_N :
  M ∩ N = expectedIntersection :=
by
  sorry

end intersection_M_N_l224_224083


namespace coco_hours_used_l224_224300

noncomputable def electricity_price : ℝ := 0.10
noncomputable def consumption_rate : ℝ := 2.4
noncomputable def total_cost : ℝ := 6.0

theorem coco_hours_used (hours_used : ℝ) : hours_used = total_cost / (consumption_rate * electricity_price) :=
by
  sorry

end coco_hours_used_l224_224300


namespace incenter_x_coord_eq_2_l224_224609

noncomputable def equidistant_point (x y : ℝ) : Prop :=
  abs y = abs x ∧ abs (x + y - 4) / real.sqrt 2 = abs x

theorem incenter_x_coord_eq_2 : ∃ y : ℝ, equidistant_point 2 y :=
by {
  sorry
}

end incenter_x_coord_eq_2_l224_224609


namespace smallest_base_for_100_l224_224153

theorem smallest_base_for_100 :
  ∃ b : ℕ, b^2 ≤ 100 ∧ 100 < b^3 ∧ ∀ c : ℕ, (c^2 ≤ 100 ∧ 100 < c^3) → b ≤ c :=
sorry

end smallest_base_for_100_l224_224153


namespace alice_bob_sitting_arrangements_l224_224847

theorem alice_bob_sitting_arrangements :
  ∃ (n : ℕ), n = 10 
  ∧ ∃ (arrangements : ℕ), arrangements = (10 - 1)! / (10 - 1) * 2
  ∧ arrangements = 80,640 :=
begin
  -- There are 10 people.
  use 10,
  split,
  -- Alice and Bob must sit next to each other.
  exact rfl,
  use ((10 - 1)! / (10 - 1)) * 2,
  split,
  -- Compute the arrangement
  -- (9!) / 9 * 2 = (8!) * 2
  -- 8! = 40320, so the total is 40320 * 2 = 80640
  exact rfl,
  -- The total is 80640
  norm_num,
  sorry,
end

end alice_bob_sitting_arrangements_l224_224847


namespace remaining_pie_portion_l224_224188

theorem remaining_pie_portion (Carlos_takes: ℝ) (fraction_Maria: ℝ) :
  Carlos_takes = 0.60 →
  fraction_Maria = 0.25 →
  (1 - Carlos_takes) * (1 - fraction_Maria) = 0.30 := by
  intros h1 h2
  rw [h1, h2]
  simp
  sorry

end remaining_pie_portion_l224_224188


namespace sin_double_angle_l224_224956

variables (α : ℝ)

-- Defining the condition that α is an acute angle
def is_acute (α : ℝ) : Prop := 0 < α ∧ α < π / 2

-- Given conditions
def given_conditions : Prop :=
  is_acute α ∧ cos (α + π / 6) = 4 / 5

-- The theorem to prove
theorem sin_double_angle (α : ℝ) (h : given_conditions α) : 
  sin (2 * α + π / 3) = 24 / 25 := 
by 
  -- proof omitted
  sorry

end sin_double_angle_l224_224956


namespace tan_periodic_example_l224_224052

theorem tan_periodic_example : Real.tan (13 * Real.pi / 4) = 1 := 
by 
  sorry

end tan_periodic_example_l224_224052


namespace xyz_squared_eq_one_l224_224074

theorem xyz_squared_eq_one (x y z : ℝ) (h_distinct : x ≠ y ∧ y ≠ z ∧ z ≠ x)
    (h_eq : ∃ k, x + (1 / y) = k ∧ y + (1 / z) = k ∧ z + (1 / x) = k) : 
    x^2 * y^2 * z^2 = 1 := 
  sorry

end xyz_squared_eq_one_l224_224074


namespace intersect_count_l224_224874

def g (x : ℝ) : ℝ := sorry
def g_inv (y : ℝ) : ℝ := sorry

axiom g_invertible : ∀ (y : ℝ), g (g_inv y) = y

theorem intersect_count : {x : ℝ | g (x^2) = g (x^6)}.finite.card = 3 := by
  sorry

end intersect_count_l224_224874


namespace find_n_l224_224033

-- Defining the conditions given in the problem
def condition_eq (n : ℝ) : Prop :=
  10 * 1.8 - (n * 1.5 / 0.3) = 50

-- Stating the goal: Prove that the number n is -6.4
theorem find_n : condition_eq (-6.4) :=
by
  -- Proof is omitted
  sorry

end find_n_l224_224033


namespace restaurant_cooks_l224_224936

theorem restaurant_cooks
  (C W : ℕ)
  (h1 : C * 11 = 3 * W)
  (h2 : C * 5 = (W + 12))
  : C = 9 :=
  sorry

end restaurant_cooks_l224_224936


namespace sum_eq_neg_20_div_3_l224_224111
-- Import the necessary libraries

-- The main theoretical statement
theorem sum_eq_neg_20_div_3
    (a b c d : ℝ)
    (h : a + 2 = b + 4 ∧ b + 4 = c + 6 ∧ c + 6 = d + 8 ∧ d + 8 = a + b + c + d + 10) :
    a + b + c + d = -20 / 3 :=
by
  sorry

end sum_eq_neg_20_div_3_l224_224111


namespace identical_graphs_l224_224412

theorem identical_graphs :
  (∃ (b c : ℝ), (∀ (x y : ℝ), 3 * x + b * y + c = 0 ↔ c * x - 2 * y + 12 = 0) ∧
                 ((b, c) = (1, 6) ∨ (b, c) = (-1, -6))) → ∃ n : ℕ, n = 2 :=
by
  sorry

end identical_graphs_l224_224412


namespace largest_square_side_length_largest_rectangle_dimensions_l224_224518

variable (a b : ℝ) (h : a > 0) (k : b > 0)

-- Part (a): Side length of the largest possible square
theorem largest_square_side_length (h : a > 0) (k : b > 0) :
  ∃ (s : ℝ), s = (a * b) / (a + b) := sorry

-- Part (b): Dimensions of the largest possible rectangle
theorem largest_rectangle_dimensions (h : a > 0) (k : b > 0) :
  ∃ (x y : ℝ), x = a / 2 ∧ y = b / 2 := sorry

end largest_square_side_length_largest_rectangle_dimensions_l224_224518


namespace cosine_60_degrees_l224_224641

theorem cosine_60_degrees : Real.cos (60 * Real.pi / 180) = 1 / 2 :=
by sorry

end cosine_60_degrees_l224_224641


namespace number_of_remaining_numbers_problem_solution_l224_224119

def initial_set : List Nat := (List.range 20).map (fun n => n + 1)
def is_even (n : Nat) : Bool := n % 2 = 0
def remainder_4_mod_5 (n : Nat) : Bool := n % 5 = 4

def remaining_numbers (numbers : List Nat) : List Nat :=
  numbers.filter (fun n => ¬ is_even n).filter (fun n => ¬ remainder_4_mod_5 n)

theorem number_of_remaining_numbers : remaining_numbers initial_set = [1, 3, 5, 7, 11, 13, 15, 17] :=
  by {
    sorry
  }

theorem problem_solution : remaining_numbers initial_set = [1, 3, 5, 7, 11, 13, 15, 17] -> length (remaining_numbers initial_set) = 8 :=
  by {
    intro h,
    rw h,
    simp,
  }

end number_of_remaining_numbers_problem_solution_l224_224119


namespace find_distance_CD_l224_224804

-- Define the ellipse and the required points
def ellipse (x y : ℝ) : Prop := 16 * (x-3)^2 + 4 * (y+2)^2 = 64

-- Define the center and the semi-axes lengths
noncomputable def center : (ℝ × ℝ) := (3, -2)
noncomputable def semi_major_axis_length : ℝ := 4
noncomputable def semi_minor_axis_length : ℝ := 2

-- Define the points C and D on the ellipse
def point_C (x y : ℝ) : Prop := ellipse x y ∧ (x = 3 + semi_major_axis_length ∨ x = 3 - semi_major_axis_length) ∧ y = -2
def point_D (x y : ℝ) : Prop := ellipse x y ∧ x = 3 ∧ (y = -2 + semi_minor_axis_length ∨ y = -2 - semi_minor_axis_length)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := 
Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

-- Main theorem to prove
theorem find_distance_CD : 
  ∃ C D : ℝ × ℝ, 
    (point_C C.1 C.2 ∧ point_D D.1 D.2) → 
    distance C D = 2 * Real.sqrt 5 := 
sorry

end find_distance_CD_l224_224804


namespace simplify_expression_l224_224733

theorem simplify_expression :
  (Real.sqrt (Real.sqrt (81)) - Real.sqrt (8 + 1 / 2)) ^ 2 = (35 / 2) - 3 * Real.sqrt 34 :=
by
  sorry

end simplify_expression_l224_224733


namespace cos_of_60_degrees_is_half_l224_224628

noncomputable theory

def unit_circle_coordinates := (1/2 : ℝ, real.sqrt 3 / 2)

theorem cos_of_60_degrees_is_half :
  ∀ θ : ℝ, θ = 60 → unit_circle_coordinates.fst = real.cos θ :=
begin
  sorry
end

end cos_of_60_degrees_is_half_l224_224628


namespace solve_equation_x_squared_eq_16x_l224_224897

theorem solve_equation_x_squared_eq_16x :
  ∀ x : ℝ, x^2 = 16 * x ↔ (x = 0 ∨ x = 16) :=
by 
  intro x
  -- Complete proof here
  sorry

end solve_equation_x_squared_eq_16x_l224_224897


namespace arithmetic_mean_multiplied_correct_l224_224759

-- Define the fractions involved
def frac1 : ℚ := 3 / 4
def frac2 : ℚ := 5 / 8

-- Define the arithmetic mean and the final multiplication result
def mean_and_multiply_result : ℚ := ( (frac1 + frac2) / 2 ) * 3

-- Statement to prove that the calculated result is equal to 33/16
theorem arithmetic_mean_multiplied_correct : mean_and_multiply_result = 33 / 16 := 
by 
  -- Skipping the proof with sorry for the statement only requirement
  sorry

end arithmetic_mean_multiplied_correct_l224_224759


namespace find_A_and_B_l224_224943

theorem find_A_and_B : 
  ∃ A B : ℝ, 
    (∀ x : ℝ, x ≠ 10 ∧ x ≠ -3 → 5*x + 2 = A * (x + 3) + B * (x - 10)) ∧ 
    A = 4 ∧ B = 1 :=
  sorry

end find_A_and_B_l224_224943


namespace ball_redistribution_impossible_l224_224772

noncomputable def white_boxes_initial_ball_count := 31
noncomputable def black_boxes_initial_ball_count := 26
noncomputable def white_boxes_new_ball_count := 21
noncomputable def black_boxes_new_ball_count := 16
noncomputable def white_boxes_target_ball_count := 15
noncomputable def black_boxes_target_ball_count := 10

theorem ball_redistribution_impossible
  (initial_white_boxes : ℕ)
  (initial_black_boxes : ℕ)
  (new_white_boxes : ℕ)
  (new_black_boxes : ℕ)
  (total_white_boxes : ℕ)
  (total_black_boxes : ℕ) :
  initial_white_boxes * white_boxes_initial_ball_count +
    initial_black_boxes * black_boxes_initial_ball_count =
  total_white_boxes * white_boxes_target_ball_count +
    total_black_boxes * black_boxes_target_ball_count →
  (new_white_boxes, new_black_boxes) = (total_white_boxes - initial_white_boxes, total_black_boxes - initial_black_boxes) →
  ¬(∃ total_white_boxes total_black_boxes, 
    total_white_boxes * white_boxes_target_ball_count +
    total_black_boxes * black_boxes_target_ball_count =
    initial_white_boxes * white_boxes_initial_ball_count +
    initial_black_boxes * black_boxes_initial_ball_count) :=
by sorry

end ball_redistribution_impossible_l224_224772


namespace part1_part2_l224_224286

noncomputable def f (m x : ℝ) : ℝ := m - |x - 1| - |x + 1|

theorem part1 (x : ℝ) : -3 / 2 < x ∧ x < 3 / 2 ↔ f 5 x > 2 := by
  sorry

theorem part2 (m : ℝ) : (∀ x : ℝ, ∃ y : ℝ, x^2 + 2 * x + 3 = f m y) ↔ 4 ≤ m := by
  sorry

end part1_part2_l224_224286


namespace eat_five_pounds_in_46_875_min_l224_224730

theorem eat_five_pounds_in_46_875_min
  (fat_rate : ℝ) (thin_rate : ℝ) (combined_rate : ℝ) (total_fruit : ℝ)
  (hf1 : fat_rate = 1 / 15)
  (hf2 : thin_rate = 1 / 25)
  (h_comb : combined_rate = fat_rate + thin_rate)
  (h_fruit : total_fruit = 5) :
  total_fruit / combined_rate = 46.875 :=
by
  sorry

end eat_five_pounds_in_46_875_min_l224_224730


namespace right_triangle_area_l224_224242

/-- Given a right triangle with one leg of length 3 and the hypotenuse of length 5,
    the area of the triangle is 6. -/
theorem right_triangle_area (a b c : ℝ) (h₁ : a = 3) (h₂ : c = 5) (h₃ : c^2 = a^2 + b^2) :
  (1 / 2) * a * b = 6 := 
sorry

end right_triangle_area_l224_224242


namespace avg_of_combined_data_l224_224082

variables (x1 x2 x3 y1 y2 y3 a b : ℝ)

-- condition: average of x1, x2, x3 is a
axiom h1 : (x1 + x2 + x3) / 3 = a

-- condition: average of y1, y2, y3 is b
axiom h2 : (y1 + y2 + y3) / 3 = b

-- Prove that the average of 3x1 + y1, 3x2 + y2, 3x3 + y3 is 3a + b
theorem avg_of_combined_data : 
  ((3 * x1 + y1) + (3 * x2 + y2) + (3 * x3 + y3)) / 3 = 3 * a + b :=
by
  sorry

end avg_of_combined_data_l224_224082


namespace exist_1006_intersecting_permutations_l224_224597

open Equiv

def intersects {n : ℕ} (a b : Perm (Fin n)) : Prop :=
  ∃ k : Fin n, a k = b k

theorem exist_1006_intersecting_permutations :
  ∃ (S : Fin 1006 → Perm (Fin 2010)), ∀ p : Perm (Fin 2010), ∃ i : Fin 1006, intersects (S i) p :=
sorry

end exist_1006_intersecting_permutations_l224_224597


namespace locus_of_Q_is_circle_l224_224349

variables {A B C P Q : ℝ}

def point_on_segment (A B C : ℝ) : Prop := C > A ∧ C < B

def variable_point_on_circle (A B P : ℝ) : Prop := (P - A) * (P - B) = 0

def ratio_condition (C P Q A B : ℝ) : Prop := (P - C) / (C - Q) = (A - C) / (C - B)

def locus_of_Q_circle (A B C P Q : ℝ) : Prop := ∃ B', (C > A ∧ C < B) → (P - A) * (P - B) = 0 → (P - C) / (C - Q) = (A - C) / (C - B) → (Q - B') * (Q - B) = 0

theorem locus_of_Q_is_circle (A B C P Q : ℝ) :
  point_on_segment A B C →
  variable_point_on_circle A B P →
  ratio_condition C P Q A B →
  locus_of_Q_circle A B C P Q :=
by
  sorry

end locus_of_Q_is_circle_l224_224349


namespace seating_circular_table_l224_224172

variable (V : Type) [Fintype V] [DecidableEq V]
variable (G : SimpleGraph V)
variable [Fintype (EdgeSet G)]
variable [DecidableRel G.Adj]

theorem seating_circular_table (P : Fintype (Fin 5))
  (h : ∀ (X : Finset (Fin 5)), X.card = 3 → ∃ (x y z : V), x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ (G.Adj x y ∨ G.Adj y z) ∧ (¬ G.Adj x y ∨ ¬ G.Adj y z)) :
  ∃ (C : Cycle G), ∀ (v : Fin 5), v ∈ C.support → ∃ (u : Fin 5), G.Adj v u :=
begin
  sorry
end

end seating_circular_table_l224_224172


namespace pythagorean_theorem_l224_224539

theorem pythagorean_theorem (a b c : ℝ) : (a^2 + b^2 = c^2) ↔ (a^2 + b^2 = c^2) :=
by sorry

end pythagorean_theorem_l224_224539


namespace lcm_gcd_product_l224_224909

theorem lcm_gcd_product (a b : ℕ) (ha : a = 36) (hb : b = 60) : 
  Nat.lcm a b * Nat.gcd a b = 2160 :=
by
  rw [ha, hb]
  sorry

end lcm_gcd_product_l224_224909


namespace largest_gcd_sum_1071_l224_224445

theorem largest_gcd_sum_1071 (x y: ℕ) (h1: x > 0) (h2: y > 0) (h3: x + y = 1071) : 
  ∃ d, d = Nat.gcd x y ∧ ∀ z, (z ∣ 1071 -> z ≤ d) := 
sorry

end largest_gcd_sum_1071_l224_224445


namespace elberta_money_l224_224965

theorem elberta_money (granny_smith : ℕ) (anjou : ℕ) (elberta : ℕ) 
  (h1 : granny_smith = 120) 
  (h2 : anjou = granny_smith / 4) 
  (h3 : elberta = anjou + 5) : 
  elberta = 35 :=
by {
  sorry
}

end elberta_money_l224_224965


namespace total_marbles_l224_224536

variable (r : ℝ) -- number of red marbles
variable (b g y : ℝ) -- number of blue, green, and yellow marbles

-- Conditions
axiom h1 : r = 1.3 * b
axiom h2 : g = 1.5 * r
axiom h3 : y = 0.8 * g

/-- Theorem: The total number of marbles in the collection is 4.47 times the number of red marbles -/
theorem total_marbles (r b g y : ℝ) (h1 : r = 1.3 * b) (h2 : g = 1.5 * r) (h3 : y = 0.8 * g) :
  b + r + g + y = 4.47 * r :=
sorry

end total_marbles_l224_224536


namespace correct_division_algorithm_l224_224912

theorem correct_division_algorithm : (-8 : ℤ) / (-4 : ℤ) = (8 : ℤ) / (4 : ℤ) := 
by 
  sorry

end correct_division_algorithm_l224_224912


namespace functional_eq_solution_l224_224358

variable (f g : ℝ → ℝ)

theorem functional_eq_solution (h : ∀ x y : ℝ, f (x + y * g x) = g x + x * f y) : f = id := 
sorry

end functional_eq_solution_l224_224358


namespace find_a3_l224_224527

-- Define the geometric sequence and its properties.
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Given conditions
variables (a : ℕ → ℝ) (q : ℝ)
variable (h_GeoSeq : is_geometric_sequence a q)
variable (h_a1 : a 1 = 1)
variable (h_a5 : a 5 = 9)

-- Define what we need to prove
theorem find_a3 : a 3 = 3 :=
sorry

end find_a3_l224_224527


namespace frequency_of_group_5_l224_224987

/-- Let the total number of data points be 50, number of data points in groups 1, 2, 3, and 4 be
  2, 8, 15, and 5 respectively. Prove that the frequency of group 5 is 0.4. -/
theorem frequency_of_group_5 :
  let total_data_points := 50
  let group1_data_points := 2
  let group2_data_points := 8
  let group3_data_points := 15
  let group4_data_points := 5
  let group5_data_points := total_data_points - group1_data_points - group2_data_points - group3_data_points - group4_data_points
  let frequency_group5 := (group5_data_points : ℝ) / total_data_points
  frequency_group5 = 0.4 := 
by
  sorry

end frequency_of_group_5_l224_224987


namespace speed_of_second_train_l224_224472

/-- Given:
1. The first train has a length of 220 meters.
2. The speed of the first train is 120 kilometers per hour.
3. The time taken to cross each other is 9 seconds.
4. The length of the second train is 280.04 meters.

Prove the speed of the second train is 80 kilometers per hour. -/
theorem speed_of_second_train
    (len_first_train : ℝ := 220)
    (speed_first_train_kmph : ℝ := 120)
    (time_to_cross : ℝ := 9)
    (len_second_train : ℝ := 280.04) 
  : (len_first_train / time_to_cross + len_second_train / time_to_cross - (speed_first_train_kmph * 1000 / 3600)) * (3600 / 1000) = 80 := 
by
  sorry

end speed_of_second_train_l224_224472


namespace point_above_line_l224_224681

theorem point_above_line (a : ℝ) : 3 * (-3) - 2 * (-1) - a < 0 ↔ a > -7 :=
by sorry

end point_above_line_l224_224681


namespace ratio_abcd_efgh_l224_224767

variable (a b c d e f g h : ℚ)

theorem ratio_abcd_efgh :
  (a / b = 1 / 3) ->
  (b / c = 2) ->
  (c / d = 1 / 2) ->
  (d / e = 3) ->
  (e / f = 1 / 2) ->
  (f / g = 5 / 3) ->
  (g / h = 4 / 9) ->
  (a * b * c * d) / (e * f * g * h) = 1 / 97 :=
by
  sorry

end ratio_abcd_efgh_l224_224767


namespace team_X_played_24_games_l224_224294

def games_played_X (x : ℕ) : ℕ := x
def games_played_Y (x : ℕ) : ℕ := x + 9
def games_won_X (x : ℕ) : ℚ := 3 / 4 * x
def games_won_Y (x : ℕ) : ℚ := 2 / 3 * (x + 9)

theorem team_X_played_24_games (x : ℕ) 
  (h1 : games_won_Y x = games_won_X x + 4) : games_played_X x = 24 :=
by
  sorry

end team_X_played_24_games_l224_224294


namespace sophia_book_length_l224_224164

variables {P : ℕ}

def total_pages (P : ℕ) : Prop :=
  (2 / 3 : ℝ) * P = (1 / 3 : ℝ) * P + 90

theorem sophia_book_length 
  (h1 : total_pages P) :
  P = 270 :=
sorry

end sophia_book_length_l224_224164


namespace hyperbola_satisfies_conditions_l224_224615

-- Define the equations of the hyperbolas as predicates
def hyperbola_A (x y : ℝ) : Prop := x^2 - (y^2 / 4) = 1
def hyperbola_B (x y : ℝ) : Prop := (x^2 / 4) - y^2 = 1
def hyperbola_C (x y : ℝ) : Prop := (y^2 / 4) - x^2 = 1
def hyperbola_D (x y : ℝ) : Prop := y^2 - (x^2 / 4) = 1

-- Define the conditions on foci and asymptotes
def foci_on_y_axis (h : (ℝ → ℝ → Prop)) : Prop := 
  h = hyperbola_C ∨ h = hyperbola_D

def has_asymptotes (h : (ℝ → ℝ → Prop)) : Prop :=
  ∀ x y, h x y → (y = (1/2) * x ∨ y = -(1/2) * x)

-- The proof statement
theorem hyperbola_satisfies_conditions :
  foci_on_y_axis hyperbola_D ∧ has_asymptotes hyperbola_D ∧ 
    (¬ (foci_on_y_axis hyperbola_A ∧ has_asymptotes hyperbola_A)) ∧ 
    (¬ (foci_on_y_axis hyperbola_B ∧ has_asymptotes hyperbola_B)) ∧ 
    (¬ (foci_on_y_axis hyperbola_C ∧ has_asymptotes hyperbola_C)) := 
by
  sorry

end hyperbola_satisfies_conditions_l224_224615


namespace arithmetic_sequence_general_term_l224_224269

theorem arithmetic_sequence_general_term (a_n S_n : ℕ → ℕ) (d : ℕ) (a1 S1 S5 S7 : ℕ)
  (h1: a_n 3 = 5)
  (h2: ∀ n, S_n n = (n * (a1 * 2 + (n - 1) * d)) / 2)
  (h3: S1 = S_n 1)
  (h4: S5 = S_n 5)
  (h5: S7 = S_n 7)
  (h6: S1 + S7 = 2 * S5):
  ∀ n, a_n n = 2 * n - 1 :=
by
  sorry

end arithmetic_sequence_general_term_l224_224269


namespace candy_sampling_l224_224163

theorem candy_sampling (total_customers caught_sampling not_caught_sampling : ℝ) :
  caught_sampling = 0.22 * total_customers →
  not_caught_sampling = 0.12 * (total_customers * sampling_percent) →
  (sampling_percent * total_customers = caught_sampling / 0.78) :=
by
  intros h1 h2
  sorry

end candy_sampling_l224_224163


namespace inequality_proof_l224_224514

theorem inequality_proof (a b x : ℝ) (h : a > b) : a * 2 ^ x > b * 2 ^ x :=
sorry

end inequality_proof_l224_224514


namespace smallest_integer_to_make_y_perfect_square_l224_224476

-- Define y as given in the problem
def y : ℕ :=
  2^33 * 3^54 * 4^45 * 5^76 * 6^57 * 7^38 * 8^69 * 9^10

-- Smallest integer n such that (y * n) is a perfect square
theorem smallest_integer_to_make_y_perfect_square
  : ∃ n : ℕ, (∀ k : ℕ, y * n = k * k) ∧ (∀ m : ℕ, (∀ k : ℕ, y * m = k * k) → n ≤ m) := 
sorry

end smallest_integer_to_make_y_perfect_square_l224_224476


namespace shelby_scooter_drive_l224_224427

/-- 
Let y be the time (in minutes) Shelby drove when it was not raining.
Speed when not raining is 25 miles per hour, which is 5/12 mile per minute.
Speed when raining is 15 miles per hour, which is 1/4 mile per minute.
Total distance covered is 18 miles.
Total time taken is 36 minutes.
Prove that Shelby drove for 6 minutes when it was not raining.
-/
theorem shelby_scooter_drive
  (y : ℝ)
  (h_not_raining_speed : ∀ t (h : t = (25/60 : ℝ)), t = (5/12 : ℝ))
  (h_raining_speed : ∀ t (h : t = (15/60 : ℝ)), t = (1/4 : ℝ))
  (h_total_distance : ∀ d (h : d = ((5/12 : ℝ) * y + (1/4 : ℝ) * (36 - y))), d = 18)
  (h_total_time : ∀ t (h : t = 36), t = 36) :
  y = 6 :=
sorry

end shelby_scooter_drive_l224_224427


namespace line_curve_intersection_l224_224363

theorem line_curve_intersection (a : ℝ) : 
  (∃! (x y : ℝ), (y = a * (x + 2)) ∧ (x ^ 2 - y * |y| = 1)) ↔ a ∈ Set.Ioo (-Real.sqrt 3 / 3) 1 :=
by
  sorry

end line_curve_intersection_l224_224363


namespace sum_of_thousands_and_units_digit_of_product_l224_224889

def thousands_digit (n : ℕ) : ℕ := (n / 1000) % 10
def units_digit (n : ℕ) : ℕ := n % 10

-- Define the two 102-digit numbers
def num1 : ℕ := 404040404040404040404040404040404040404040404040404040404040404040404040404040404040404040404
def num2 : ℕ := 707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707

-- Define their product
def product : ℕ := num1 * num2

-- Define the conditions for the problem
def A := thousands_digit product
def B := units_digit product

-- Define the problem statement
theorem sum_of_thousands_and_units_digit_of_product : A + B = 13 := 
by
  sorry

end sum_of_thousands_and_units_digit_of_product_l224_224889


namespace frequency_of_third_group_l224_224822

theorem frequency_of_third_group (total_data first_group second_group fourth_group third_group : ℕ) 
    (h1 : total_data = 40)
    (h2 : first_group = 5)
    (h3 : second_group = 12)
    (h4 : fourth_group = 8) :
    third_group = 15 :=
by
  sorry

end frequency_of_third_group_l224_224822


namespace first_quadrant_sin_cos_inequality_l224_224231

def is_first_quadrant_angle (α : ℝ) : Prop :=
  0 < Real.sin α ∧ 0 < Real.cos α

theorem first_quadrant_sin_cos_inequality (α : ℝ) :
  (is_first_quadrant_angle α ↔ Real.sin α + Real.cos α > 1) :=
by
  sorry

end first_quadrant_sin_cos_inequality_l224_224231


namespace fraction_subtraction_l224_224493

theorem fraction_subtraction (x y : ℝ) (h : x ≠ y) : (x + y) / (x - y) - (2 * y) / (x - y) = 1 := by
  sorry

end fraction_subtraction_l224_224493


namespace emery_family_first_hour_distance_l224_224669

noncomputable def total_time : ℝ := 4
noncomputable def remaining_distance : ℝ := 300
noncomputable def first_hour_distance : ℝ := 100

theorem emery_family_first_hour_distance :
  (remaining_distance / (total_time - 1)) = first_hour_distance :=
sorry

end emery_family_first_hour_distance_l224_224669


namespace algebraic_expression_value_l224_224691

theorem algebraic_expression_value (a : ℝ) (h : a^2 + 2023 * a - 1 = 0) : 
  a * (a + 1) * (a - 1) + 2023 * a^2 + 1 = 1 :=
by
  sorry

end algebraic_expression_value_l224_224691


namespace friends_payment_l224_224169

theorem friends_payment
  (num_friends : ℕ) (num_bread : ℕ) (cost_bread : ℕ) 
  (num_hotteok : ℕ) (cost_hotteok : ℕ) (total_cost : ℕ)
  (cost_per_person : ℕ)
  (h1 : num_friends = 4)
  (h2 : num_bread = 5)
  (h3 : cost_bread = 200)
  (h4 : num_hotteok = 7)
  (h5 : cost_hotteok = 800)
  (h6 : total_cost = num_bread * cost_bread + num_hotteok * cost_hotteok)
  (h7 : cost_per_person = total_cost / num_friends) :
  cost_per_person = 1650 := by
  sorry

end friends_payment_l224_224169


namespace new_ratio_after_adding_ten_l224_224904

theorem new_ratio_after_adding_ten 
  (x : ℕ) 
  (h_ratio : 3 * x = 15) 
  (new_smaller : ℕ := x + 10) 
  (new_larger : ℕ := 15) 
  : new_smaller / new_larger = 1 :=
by sorry

end new_ratio_after_adding_ten_l224_224904


namespace david_presents_l224_224192

variables (C B E : ℕ)

def total_presents (C B E : ℕ) : ℕ := C + B + E

theorem david_presents :
  C = 60 →
  B = 3 * E →
  E = (C / 2) - 10 →
  total_presents C B E = 140 :=
by
  intros hC hB hE
  sorry

end david_presents_l224_224192


namespace problem1_problem2_l224_224494

-- Definition and proof statement for Problem 1
theorem problem1 (y : ℝ) : 
  (y + 2) * (y - 2) + (y - 1) * (y + 3) = 2 * y^2 + 2 * y - 7 := 
by sorry

-- Definition and proof statement for Problem 2
theorem problem2 (x : ℝ) (h : x ≠ -1) :
  (1 + 2 / (x + 1)) / ((x^2 + 6 * x + 9) / (x + 1)) = 1 / (x + 3) :=
by sorry

end problem1_problem2_l224_224494


namespace regular_polygon_sides_l224_224651

theorem regular_polygon_sides (h : ∀ n, (180 * (n - 2) = 150 * n)) : n = 12 := 
by
  sorry

end regular_polygon_sides_l224_224651


namespace num_3_digit_div_by_5_l224_224972

theorem num_3_digit_div_by_5 : 
  ∃ (n : ℕ), 
  let a := 100 in let d := 5 in let l := 995 in
  (l = a + (n-1) * d) ∧ n = 180 :=
by
  sorry

end num_3_digit_div_by_5_l224_224972


namespace determine_m_with_opposite_roots_l224_224056

theorem determine_m_with_opposite_roots (c d k : ℝ) (h : c + d ≠ 0):
  (∃ m : ℝ, ∀ x : ℝ, (x^2 - d * x) / (c * x - k) = (m - 2) / (m + 2) ∧ 
            (x = -y ∧ y = -x)) ↔ m = 2 * (c - d) / (c + d) :=
sorry

end determine_m_with_opposite_roots_l224_224056


namespace necessary_but_not_sufficient_l224_224394

theorem necessary_but_not_sufficient (a b : ℝ) (h : a^2 = b^2) : 
  (a^2 + b^2 = 2 * a * b) ↔ (a = b) :=
begin
  sorry
end

end necessary_but_not_sufficient_l224_224394


namespace regular_polygon_sides_l224_224667

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, 1 ≤ i → i ≤ n → (interior_angle i = 150) ) 
  (sum_interior_angles : ∑ i in range n, interior_angle i = 180 * (n - 2))
  {interior_angle : ℕ → ℕ} :
  n = 12 :=
by
  sorry

end regular_polygon_sides_l224_224667


namespace smallest_five_digit_perfect_square_and_cube_l224_224019

theorem smallest_five_digit_perfect_square_and_cube :
  ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ (∃ a : ℕ, n = a^6) ∧ n = 15625 := 
by
  sorry

end smallest_five_digit_perfect_square_and_cube_l224_224019


namespace basketball_team_wins_l224_224921

theorem basketball_team_wins (wins_first_60 : ℕ) (remaining_games : ℕ) (total_games : ℕ) (target_win_percentage : ℚ) (winning_games : ℕ) : 
  wins_first_60 = 45 → remaining_games = 40 → total_games = 100 → target_win_percentage = 0.75 → 
  winning_games = 30 := by
  intros h1 h2 h3 h4
  sorry

end basketball_team_wins_l224_224921


namespace bookstore_floor_l224_224449

theorem bookstore_floor (academy_floor reading_room_floor bookstore_floor : ℤ)
  (h1: academy_floor = 7)
  (h2: reading_room_floor = academy_floor + 4)
  (h3: bookstore_floor = reading_room_floor - 9) :
  bookstore_floor = 2 :=
by
  sorry

end bookstore_floor_l224_224449


namespace percentage_passed_all_three_l224_224101

variable (F_H F_E F_M F_HE F_EM F_HM F_HEM : ℝ)

theorem percentage_passed_all_three :
  F_H = 0.46 →
  F_E = 0.54 →
  F_M = 0.32 →
  F_HE = 0.18 →
  F_EM = 0.12 →
  F_HM = 0.1 →
  F_HEM = 0.06 →
  (100 - (F_H + F_E + F_M - F_HE - F_EM - F_HM + F_HEM)) = 2 :=
by sorry

end percentage_passed_all_three_l224_224101


namespace susie_rooms_l224_224756

theorem susie_rooms
  (house_vacuum_time_hours : ℕ)
  (room_vacuum_time_minutes : ℕ)
  (total_vacuum_time_minutes : ℕ)
  (total_vacuum_time_computed : house_vacuum_time_hours * 60 = total_vacuum_time_minutes)
  (rooms_count : ℕ)
  (rooms_count_computed : total_vacuum_time_minutes / room_vacuum_time_minutes = rooms_count) :
  house_vacuum_time_hours = 2 →
  room_vacuum_time_minutes = 20 →
  rooms_count = 6 :=
by
  intros h1 h2
  sorry

end susie_rooms_l224_224756


namespace circles_equal_or_tangent_l224_224255

theorem circles_equal_or_tangent (a b c : ℝ) 
  (h : (2 * a)^2 - 4 * (b^2 - c * (b - a)) = 0) : 
  a = b ∨ c = a + b :=
by
  -- Will fill the proof later
  sorry

end circles_equal_or_tangent_l224_224255


namespace common_ratio_is_2_l224_224079

noncomputable def common_ratio_of_increasing_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (∀ n, a (n+1) = a n * q) ∧ (∀ m n, m < n → a m < a n)

theorem common_ratio_is_2
  (a : ℕ → ℝ) (q : ℝ)
  (hgeo : common_ratio_of_increasing_geometric_sequence a q)
  (h1 : a 1 + a 5 = 17)
  (h2 : a 2 * a 4 = 16) :
  q = 2 :=
sorry

end common_ratio_is_2_l224_224079


namespace is_exact_time_now_321_l224_224105

noncomputable def current_time_is_321 : Prop :=
  exists t : ℝ, 0 < t ∧ t < 60 ∧ |(6 * t + 48) - (90 + 0.5 * (t - 4))| = 180

theorem is_exact_time_now_321 : current_time_is_321 := 
  sorry

end is_exact_time_now_321_l224_224105


namespace candies_initial_count_l224_224429

theorem candies_initial_count (x : ℕ) (h : (x - 29) / 13 = 15) : x = 224 :=
sorry

end candies_initial_count_l224_224429


namespace sum_a_b_l224_224982

theorem sum_a_b (a b : ℚ) (h1 : a + 3 * b = 27) (h2 : 5 * a + 2 * b = 40) : a + b = 161 / 13 :=
  sorry

end sum_a_b_l224_224982


namespace num_ways_to_divide_friends_l224_224836

theorem num_ways_to_divide_friends :
  (4 : ℕ) ^ 8 = 65536 := by
  sorry

end num_ways_to_divide_friends_l224_224836


namespace range_of_a_l224_224533

theorem range_of_a (a : ℝ) : (∀ x : ℝ, a * x^2 - 2 * a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 1) := 
by sorry

end range_of_a_l224_224533


namespace not_child_age_l224_224418

theorem not_child_age (children_ages : Finset ℕ) (license_plate_number : ℕ) (mr_jones_age : ℕ) :
  children_ages.card = 8 ∧ 9 ∈ children_ages ∧
  (∃ a b, a ≠ b ∧ license_plate_number = 1001*a + 1010*b) ∧
  (∀ age ∈ children_ages, license_plate_number % age = 0) ∧
  mr_jones_age = (license_plate_number % 100) →
  5 ∉ children_ages :=
by
  sorry

end not_child_age_l224_224418


namespace shared_vertex_angle_of_triangle_and_square_l224_224015

theorem shared_vertex_angle_of_triangle_and_square (α β γ δ ε ζ η θ : ℝ) :
  (α = 60 ∧ β = 60 ∧ γ = 60 ∧ δ = 90 ∧ ε = 90 ∧ ζ = 90 ∧ η = 90 ∧ θ = 90) →
  θ = 90 :=
by
  sorry

end shared_vertex_angle_of_triangle_and_square_l224_224015


namespace fibby_numbers_l224_224927

def is_fibby (k : ℕ) : Prop :=
  k ≥ 3 ∧ ∃ (n : ℕ) (d : ℕ → ℕ),
  (∀ j, 1 ≤ j ∧ j ≤ k - 2 → d (j + 2) = d (j + 1) + d j) ∧
  (∀ (j : ℕ), 1 ≤ j ∧ j ≤ k → d j ∣ n) ∧
  (∀ (m : ℕ), m ∣ n → m < d 1 ∨ m > d k)

theorem fibby_numbers : ∀ (k : ℕ), is_fibby k → k = 3 ∨ k = 4 :=
sorry

end fibby_numbers_l224_224927


namespace ellipse_major_minor_axis_l224_224920

theorem ellipse_major_minor_axis (m : ℝ) :
  (∀ x y : ℝ, x^2 + m*y^2 = 1) ∧
  (∃ a b : ℝ, a = 2 * b ∧ b^2 = 1 ∧ a^2 = 1/m) →
  m = 1/4 :=
by {
  sorry
}

end ellipse_major_minor_axis_l224_224920


namespace sixth_oak_placement_l224_224613

structure Point :=
  (x : ℝ)
  (y : ℝ)

def is_aligned (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

noncomputable def intersection_point (p1 p2 p3 p4 : Point) : Point := 
  let m1 := (p2.y - p1.y) / (p2.x - p1.x)
  let m2 := (p4.y - p3.y) / (p4.x - p3.x)
  let c1 := p1.y - (m1 * p1.x)
  let c2 := p3.y - (m2 * p3.x)
  let x := (c2 - c1) / (m1 - m2)
  let y := m1 * x + c1
  ⟨x, y⟩

theorem sixth_oak_placement 
  (A1 A2 A3 B1 B2 B3 : Point) 
  (hA : ¬ is_aligned A1 A2 A3)
  (hB : ¬ is_aligned B1 B2 B3) :
  ∃ P : Point, (∃ (C1 C2 : Point), C1 = A1 ∧ C2 = B1 ∧ is_aligned C1 C2 P) ∧ 
               (∃ (C3 C4 : Point), C3 = A2 ∧ C4 = B2 ∧ is_aligned C3 C4 P) := by
  sorry

end sixth_oak_placement_l224_224613


namespace current_job_wage_l224_224416

variable (W : ℝ) -- Maisy's wage per hour at her current job

-- Define the conditions
def current_job_hours : ℝ := 8
def new_job_hours : ℝ := 4
def new_job_wage_per_hour : ℝ := 15
def new_job_bonus : ℝ := 35
def additional_new_job_earnings : ℝ := 15

-- Assert the given condition
axiom job_earnings_condition : 
  new_job_hours * new_job_wage_per_hour + new_job_bonus 
  = current_job_hours * W + additional_new_job_earnings

-- The theorem we want to prove
theorem current_job_wage : W = 10 := by
  sorry

end current_job_wage_l224_224416


namespace bill_caroline_ratio_l224_224051

theorem bill_caroline_ratio (B C : ℕ) (h1 : B = 17) (h2 : B + C = 26) : B / C = 17 / 9 := by
  sorry

end bill_caroline_ratio_l224_224051


namespace capacities_correct_rental_plan_exists_minimal_rental_cost_exists_l224_224964

-- Step 1: Define the capacities of type A and B cars
def typeACarCapacity := 3
def typeBCarCapacity := 4

-- Step 2: Prove transportation capacities x and y
theorem capacities_correct (x y: ℕ) (h1 : 3 * x + 2 * y = 17) (h2 : 2 * x + 3 * y = 18) :
    x = typeACarCapacity ∧ y = typeBCarCapacity :=
by
  sorry

-- Step 3: Define a rental plan to transport 35 tons
theorem rental_plan_exists (a b : ℕ) : 3 * a + 4 * b = 35 :=
by
  sorry

-- Step 4: Prove the minimal cost solution
def typeACarCost := 300
def typeBCarCost := 320

def rentalCost (a b : ℕ) : ℕ := a * typeACarCost + b * typeBCarCost

theorem minimal_rental_cost_exists :
    ∃ a b, 3 * a + 4 * b = 35 ∧ rentalCost a b = 2860 :=
by
  sorry

end capacities_correct_rental_plan_exists_minimal_rental_cost_exists_l224_224964


namespace min_value_frac_x_y_l224_224364

theorem min_value_frac_x_y (x y : ℝ) (hx : x > 0) (hy : y > -1) (hxy : x + y = 1) :
  ∃ m, m = 2 + Real.sqrt 3 ∧ ∀ x y, x > 0 → y > -1 → x + y = 1 → (x^2 + 3) / x + y^2 / (y + 1) ≥ m :=
sorry

end min_value_frac_x_y_l224_224364


namespace evaluate_expression_l224_224501

theorem evaluate_expression (a b : ℕ) (h_a : a = 15) (h_b : b = 7) :
  (a + b) ^ 2 - (a ^ 2 + b ^ 2) = 210 :=
by 
  rw [h_a, h_b]
  sorry

end evaluate_expression_l224_224501


namespace susie_rooms_l224_224755

theorem susie_rooms
  (house_vacuum_time_hours : ℕ)
  (room_vacuum_time_minutes : ℕ)
  (total_vacuum_time_minutes : ℕ)
  (total_vacuum_time_computed : house_vacuum_time_hours * 60 = total_vacuum_time_minutes)
  (rooms_count : ℕ)
  (rooms_count_computed : total_vacuum_time_minutes / room_vacuum_time_minutes = rooms_count) :
  house_vacuum_time_hours = 2 →
  room_vacuum_time_minutes = 20 →
  rooms_count = 6 :=
by
  intros h1 h2
  sorry

end susie_rooms_l224_224755


namespace abby_correct_percentage_l224_224988

-- Defining the scores and number of problems for each test
def score_test1 := 85 / 100
def score_test2 := 75 / 100
def score_test3 := 60 / 100
def score_test4 := 90 / 100

def problems_test1 := 30
def problems_test2 := 50
def problems_test3 := 20
def problems_test4 := 40

-- Define the total number of problems
def total_problems := problems_test1 + problems_test2 + problems_test3 + problems_test4

-- Calculate the number of problems Abby answered correctly on each test
def correct_problems_test1 := score_test1 * problems_test1
def correct_problems_test2 := score_test2 * problems_test2
def correct_problems_test3 := score_test3 * problems_test3
def correct_problems_test4 := score_test4 * problems_test4

-- Calculate the total number of correctly answered problems
def total_correct_problems := correct_problems_test1 + correct_problems_test2 + correct_problems_test3 + correct_problems_test4

-- Calculate the overall percentage of problems answered correctly
def overall_percentage_correct := (total_correct_problems / total_problems) * 100

-- The theorem to be proved
theorem abby_correct_percentage : overall_percentage_correct = 80 := by
  -- Skipping the actual proof
  sorry

end abby_correct_percentage_l224_224988


namespace compare_negative_fractions_l224_224624

theorem compare_negative_fractions : (-3/4 : ℚ) < (-2/3 : ℚ) :=
by sorry

end compare_negative_fractions_l224_224624


namespace cakes_in_november_l224_224876

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

end cakes_in_november_l224_224876


namespace product_of_real_roots_eq_one_l224_224509

theorem product_of_real_roots_eq_one:
  ∀ x : ℝ, (x ^ (Real.log x / Real.log 5) = 25) → (∀ x1 x2 : ℝ, (x1 ^ (Real.log x1 / Real.log 5) = 25) → (x2 ^ (Real.log x2 / Real.log 5) = 25) → x1 * x2 = 1) :=
by
  sorry

end product_of_real_roots_eq_one_l224_224509


namespace smallest_number_is_1013_l224_224770

def smallest_number_divisible (n : ℕ) : Prop :=
  n - 5 % Nat.lcm 12 (Nat.lcm 16 (Nat.lcm 18 (Nat.lcm 21 28))) = 0

theorem smallest_number_is_1013 : smallest_number_divisible 1013 :=
by
  sorry

end smallest_number_is_1013_l224_224770


namespace product_lcm_gcd_eq_2160_l224_224908

theorem product_lcm_gcd_eq_2160 :
  let a := 36
  let b := 60
  lcm a b * gcd a b = 2160 := by
  sorry

end product_lcm_gcd_eq_2160_l224_224908


namespace Paula_initial_cans_l224_224280

theorem Paula_initial_cans :
  ∀ (cans rooms_lost : ℕ), rooms_lost = 10 → 
  (40 / (rooms_lost / 5) = cans + 5 → cans = 20) :=
by
  intros cans rooms_lost h_rooms_lost h_calculation
  sorry

end Paula_initial_cans_l224_224280


namespace SharonOranges_l224_224854

-- Define the given conditions
def JanetOranges : Nat := 9
def TotalOranges : Nat := 16

-- Define the statement that needs to be proven
theorem SharonOranges (J : Nat) (T : Nat) (S : Nat) (hJ : J = 9) (hT : T = 16) (hS : S = T - J) : S = 7 := by
  -- (proof to be filled in later)
  sorry

end SharonOranges_l224_224854


namespace common_ratio_q_l224_224850

variable {a_n : ℕ → ℝ} {S_n : ℕ → ℝ} {q : ℝ}

axiom a5_condition : a_n 5 = 2 * S_n 4 + 3
axiom a6_condition : a_n 6 = 2 * S_n 5 + 3

theorem common_ratio_q : q = 3 :=
by
  sorry

end common_ratio_q_l224_224850


namespace find_total_price_l224_224285

noncomputable def total_price (p : ℝ) : Prop := 0.20 * p = 240

theorem find_total_price (p : ℝ) (h : total_price p) : p = 1200 :=
by sorry

end find_total_price_l224_224285


namespace count_triangles_in_figure_l224_224830

/-- 
The figure is a rectangle divided into 8 columns and 2 rows with additional diagonal and vertical lines.
We need to prove that there are 76 triangles in total in the figure.
-/
theorem count_triangles_in_figure : 
  let columns := 8 
  let rows := 2 
  let num_triangles := 76 
  ∃ total_triangles, total_triangles = num_triangles :=
by
  sorry

end count_triangles_in_figure_l224_224830


namespace solution_interval_l224_224268

theorem solution_interval (X₀ : ℝ) (h₀ : Real.log (X₀ + 1) = 2 / X₀) : 1 < X₀ ∧ X₀ < 2 :=
by
  admit -- to be proved

end solution_interval_l224_224268


namespace smallest_positive_multiple_of_23_mod_89_is_805_l224_224152

theorem smallest_positive_multiple_of_23_mod_89_is_805 : 
  ∃ a : ℕ, 23 * a ≡ 4 [MOD 89] ∧ 23 * a = 805 := 
by
  sorry

end smallest_positive_multiple_of_23_mod_89_is_805_l224_224152


namespace count_3_digit_numbers_divisible_by_5_l224_224970

theorem count_3_digit_numbers_divisible_by_5 :
  let a := 100
  let l := 995
  let d := 5
  let n := (l - a) / d + 1
  n = 180 :=
by
  sorry

end count_3_digit_numbers_divisible_by_5_l224_224970


namespace food_last_after_join_l224_224009

-- Define the conditions
def initial_men := 760
def additional_men := 2280
def initial_days := 22
def days_before_join := 2
def initial_food := initial_men * initial_days
def remaining_food := initial_food - (initial_men * days_before_join)
def total_men := initial_men + additional_men

-- Define the goal to prove
theorem food_last_after_join :
  (remaining_food / total_men) = 5 :=
by
  sorry

end food_last_after_join_l224_224009


namespace unanswered_questions_count_l224_224276

-- Define the variables: c (correct), w (wrong), u (unanswered)
variables (c w u : ℕ)

-- Define the conditions based on the problem statement.
def total_questions (c w u : ℕ) : Prop := c + w + u = 35
def new_system_score (c u : ℕ) : Prop := 6 * c + 3 * u = 120
def old_system_score (c w : ℕ) : Prop := 5 * c - 2 * w = 55

-- Prove that the number of unanswered questions, u, equals 10
theorem unanswered_questions_count (c w u : ℕ) 
    (h1 : total_questions c w u)
    (h2 : new_system_score c u)
    (h3 : old_system_score c w) : u = 10 :=
by
  sorry

end unanswered_questions_count_l224_224276


namespace tangent_line_min_slope_equation_l224_224672

def curve (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x - 1

theorem tangent_line_min_slope_equation :
  ∃ (k : ℝ) (b : ℝ), (∀ x y, y = curve x → y = k * x + b)
  ∧ (k = 3)
  ∧ (b = -2)
  ∧ (3 * x - y - 2 = 0) :=
by
  sorry

end tangent_line_min_slope_equation_l224_224672


namespace line_intersects_x_axis_at_l224_224175

theorem line_intersects_x_axis_at (a b : ℝ) (h1 : a = 12) (h2 : b = 2)
  (c d : ℝ) (h3 : c = 6) (h4 : d = 6) : 
  ∃ x : ℝ, (x, 0) = (15, 0) := 
by
  -- proof needed here
  sorry

end line_intersects_x_axis_at_l224_224175


namespace subcommittee_count_l224_224030

theorem subcommittee_count :
  let totalWays : ℕ :=
    -- 2 Republicans and 3 Democrats
    (@nat.choose 10 2) * (@nat.choose 8 3) +
    -- 3 Republicans and 2 Democrats
    (@nat.choose 10 3) * (@nat.choose 8 2) +
    -- 4 Republicans and 1 Democrat
    (@nat.choose 10 4) * (@nat.choose 8 1) +
    -- 5 Republicans and 0 Democrats
    (@nat.choose 10 5) * (@nat.choose 8 0)
  in totalWays = 7812 := by
  sorry

end subcommittee_count_l224_224030


namespace locus_equation_l224_224685

-- Defining the fixed point A
def A : ℝ × ℝ := (4, -2)

-- Defining the predicate that B lies on the circle
def on_circle (B : ℝ × ℝ) : Prop := B.1^2 + B.2^2 = 4

-- Defining the locus of the midpoint P
def locus_of_midpoint (P : ℝ × ℝ) : Prop := (P.1 - 2)^2 + (P.2 + 1)^2 = 1

-- Main theorem statement
theorem locus_equation (P : ℝ × ℝ) (B : ℝ × ℝ) :
  on_circle(B) →
  P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  locus_of_midpoint(P) :=
by
  sorry

end locus_equation_l224_224685


namespace find_ab_l224_224075

theorem find_ab (a b c : ℤ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 29) (h3 : a + b + c = 21) : a * b = 10 := 
sorry

end find_ab_l224_224075


namespace cos_60_eq_sqrt3_div_2_l224_224630

theorem cos_60_eq_sqrt3_div_2 : Real.cos (60 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry

end cos_60_eq_sqrt3_div_2_l224_224630


namespace cost_price_per_meter_of_cloth_l224_224042

theorem cost_price_per_meter_of_cloth 
  (total_meters : ℕ)
  (selling_price : ℝ)
  (profit_per_meter : ℝ) 
  (total_profit : ℝ)
  (cp_45 : ℝ)
  (cp_per_meter: ℝ) :
  total_meters = 45 →
  selling_price = 4500 →
  profit_per_meter = 14 →
  total_profit = profit_per_meter * total_meters →
  cp_45 = selling_price - total_profit →
  cp_per_meter = cp_45 / total_meters →
  cp_per_meter = 86 :=
by
  -- your proof here
  sorry

end cost_price_per_meter_of_cloth_l224_224042


namespace weight_of_brand_b_l224_224005

theorem weight_of_brand_b (w_a w_b : ℕ) (vol_a vol_b : ℕ) (total_volume total_weight : ℕ) 
  (h1 : w_a = 950) 
  (h2 : vol_a = 3) 
  (h3 : vol_b = 2) 
  (h4 : total_volume = 4) 
  (h5 : total_weight = 3640) 
  (h6 : vol_a + vol_b = total_volume) 
  (h7 : vol_a * w_a + vol_b * w_b = total_weight) : 
  w_b = 395 := 
by {
  sorry
}

end weight_of_brand_b_l224_224005


namespace perpendicular_lines_condition_l224_224880

theorem perpendicular_lines_condition (a : ℝ) :
  (a = 2) ↔ (∃ m1 m2 : ℝ, (m1 = -1/(4 : ℝ)) ∧ (m2 = (4 : ℝ)) ∧ (m1 * m2 = -1)) :=
by sorry

end perpendicular_lines_condition_l224_224880


namespace correct_option_l224_224765

-- Define the conditions
def c1 (a : ℝ) : Prop := (2 * a^2)^3 ≠ 6 * a^6
def c2 (a : ℝ) : Prop := (a^8) / (a^2) ≠ a^4
def c3 (x y : ℝ) : Prop := (4 * x^2 * y) / (-2 * x * y) ≠ -2
def c4 : Prop := Real.sqrt ((-2)^2) = 2

-- The main statement to be proved
theorem correct_option (a x y : ℝ) (h1 : c1 a) (h2 : c2 a) (h3 : c3 x y) (h4 : c4) : c4 :=
by
  apply h4

end correct_option_l224_224765


namespace sufficient_condition_l224_224599

theorem sufficient_condition (a b c : ℤ) (h1 : a = c) (h2 : b - 1 = a) : 
  a * (a - b) + b * (b - c) + c * (c - a) = 2 :=
by
  sorry

end sufficient_condition_l224_224599


namespace journey_total_distance_l224_224548

def miles_driven : ℕ := 923
def miles_to_go : ℕ := 277
def total_distance : ℕ := 1200

theorem journey_total_distance : miles_driven + miles_to_go = total_distance := by
  sorry

end journey_total_distance_l224_224548


namespace remainder_modulo_l224_224236

theorem remainder_modulo (n : ℤ) (h : n % 50 = 23) : (3 * n - 5) % 15 = 4 := 
by 
  sorry

end remainder_modulo_l224_224236


namespace find_S_200_l224_224365

-- Define the arithmetic sequence a_n
def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := a + d * n

-- Define the sum of the first n terms of the arithmetic sequence
def sum_arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a + (n - 1) * d) / 2

-- Given conditions
variables (a_1 a_200 : ℝ) (S_200 : ℝ)
axiom collinearity_condition : a_1 + a_200 = 1
def arithmetic_sum_condition : S_200 = 200 * (a_1 + a_200) / 2

-- Proof statement
theorem find_S_200 : S_200 = 100 :=
by
  -- Definitions skipping with sorry
  sorry

end find_S_200_l224_224365


namespace Mary_is_2_l224_224325

variable (M J : ℕ)

/-- Given the conditions from the problem, Mary's age can be determined to be 2. -/
theorem Mary_is_2 (h1 : J - 5 = M + 2) (h2 : J = 2 * M + 5) : M = 2 := by
  sorry

end Mary_is_2_l224_224325


namespace abs_neg_one_half_eq_one_half_l224_224739

theorem abs_neg_one_half_eq_one_half : abs (-1/2) = 1/2 := 
by sorry

end abs_neg_one_half_eq_one_half_l224_224739


namespace sin_identity_l224_224698

theorem sin_identity (α : ℝ) (h : Real.cos (75 * Real.pi / 180 + α) = 1 / 3) :
  Real.sin (60 * Real.pi / 180 + 2 * α) = 7 / 9 :=
by
  sorry

end sin_identity_l224_224698


namespace construct_triangle_from_medians_l224_224190

theorem construct_triangle_from_medians
    (s_a s_b s_c : ℝ)
    (h1 : s_a + s_b > s_c)
    (h2 : s_a + s_c > s_b)
    (h3 : s_b + s_c > s_a) :
    ∃ (a b c : ℝ), 
    a + b > c ∧ a + c > b ∧ b + c > a ∧ 
    (∃ (median_a median_b median_c : ℝ), 
        median_a = s_a ∧ 
        median_b = s_b ∧ 
        median_c = s_c) :=
sorry

end construct_triangle_from_medians_l224_224190


namespace sum_of_coefficients_l224_224065

theorem sum_of_coefficients :
  (Nat.choose 50 3 + Nat.choose 50 5) = 2138360 := 
by 
  sorry

end sum_of_coefficients_l224_224065


namespace eval_P_at_4_over_3_eval_P_at_2_l224_224355

noncomputable def P (a : ℚ) : ℚ := (6 * a^2 - 14 * a + 5) * (3 * a - 4)

theorem eval_P_at_4_over_3 : P (4 / 3) = 0 :=
by sorry

theorem eval_P_at_2 : P 2 = 2 :=
by sorry

end eval_P_at_4_over_3_eval_P_at_2_l224_224355


namespace bejgli_slices_l224_224593

theorem bejgli_slices (x : ℕ) (hx : x ≤ 58) 
    (h1 : x * (x - 1) * (x - 2) = 3 * (58 - x) * (57 - x) * x) : 
    58 - x = 21 :=
by
  have hpos1 : 0 < x := sorry  -- x should be strictly positive since it's a count
  have hpos2 : 0 < 58 - x := sorry  -- the remaining slices should be strictly positive
  sorry

end bejgli_slices_l224_224593


namespace count_n_grids_correct_l224_224953

variable {m k n : ℕ}

-- Define the grid and n-grid conditions
def grid (m n : ℕ) := matrix (fin m) (fin n) bool

def is_n_grid (g : grid m n) (n : ℕ) : Prop :=
  ∃ reds : fin m → fin n, function.injective reds ∧
  (∀ i, reds i < n) ∧
  (∀ (i : fin (m*n - (k-1))), 
    ∑ j in finset.range k, ite (g (fin.floor (i + j).val) (fin.mod (i+j).val) = ff) 1 0 < k) ∧
  (∀ (i : fin (m*n - (m-1))), 
    ∑ j in finset.range m, ite (g (fin.floor (i + j).val) (fin.mod (i+j).val) = ff) 1 0 < m)

-- Function f(n) to count n-grids
def count_n_grids (n : ℕ) : ℕ := n!

theorem count_n_grids_correct (n m k : ℕ) (h_pos_n : 0 < n) (h_pos_m : 0 < m) (h_pos_k : 0 < k) : 
  (∀ g : grid m k, is_n_grid g n) → count_n_grids n = n! :=
sorry

end count_n_grids_correct_l224_224953


namespace students_with_uncool_parents_l224_224137

theorem students_with_uncool_parents (class_size : ℕ) (cool_dads : ℕ) (cool_moms : ℕ) (both_cool_parents : ℕ) : 
  class_size = 40 → cool_dads = 18 → cool_moms = 20 → both_cool_parents = 10 → 
  (class_size - (cool_dads - both_cool_parents + cool_moms - both_cool_parents + both_cool_parents) = 12) :=
by
  sorry

end students_with_uncool_parents_l224_224137


namespace truck_distance_l224_224044

theorem truck_distance (V_t : ℝ) (D : ℝ) (h1 : D = V_t * 8) (h2 : D = (V_t + 18) * 5) : D = 240 :=
by
  sorry

end truck_distance_l224_224044


namespace mutually_exclusive_iff_complementary_l224_224955

variables {Ω : Type} (A₁ A₂ : Set Ω) (S : Set Ω)

/-- Proposition A: Events A₁ and A₂ are mutually exclusive. -/
def mutually_exclusive : Prop := A₁ ∩ A₂ = ∅

/-- Proposition B: Events A₁ and A₂ are complementary. -/
def complementary : Prop := A₁ ∩ A₂ = ∅ ∧ A₁ ∪ A₂ = S

/-- Proposition A is a necessary but not sufficient condition for Proposition B. -/
theorem mutually_exclusive_iff_complementary :
  mutually_exclusive A₁ A₂ → (complementary A₁ A₂ S → mutually_exclusive A₁ A₂) ∧
  (¬(mutually_exclusive A₁ A₂ → complementary A₁ A₂ S)) :=
by
  sorry

end mutually_exclusive_iff_complementary_l224_224955


namespace strictly_increasing_and_symmetric_l224_224441

open Real

noncomputable def f1 (x : ℝ) : ℝ := x^((1 : ℝ)/2)
noncomputable def f2 (x : ℝ) : ℝ := x^((1 : ℝ)/3)
noncomputable def f3 (x : ℝ) : ℝ := x^((2 : ℝ)/3)
noncomputable def f4 (x : ℝ) : ℝ := x^(-(1 : ℝ)/3)

theorem strictly_increasing_and_symmetric : 
  ∀ f : ℝ → ℝ,
  (f = f2) ↔ 
  ((∀ x : ℝ, 0 < x → f x = x^((1 : ℝ)/3) ∧ f (-x) = -(f x)) ∧ 
   (∀ x y : ℝ, 0 < x ∧ 0 < y → (x < y → f x < f y))) :=
sorry

end strictly_increasing_and_symmetric_l224_224441


namespace growth_rate_of_yield_l224_224778

-- Let x be the growth rate of the average yield per acre
variable (x : ℝ)

-- Initial conditions
def initial_acres := 10
def initial_yield := 20000
def final_yield := 60000

-- Relationship between the growth rates
def growth_relation := x * initial_acres * (1 + 2 * x) * (1 + x) = final_yield / initial_yield

theorem growth_rate_of_yield (h : growth_relation x) : x = 0.5 :=
  sorry

end growth_rate_of_yield_l224_224778


namespace number_of_sides_of_regular_polygon_l224_224653

theorem number_of_sides_of_regular_polygon (n : ℕ) (h : 0 < n) (h_angle : ∀ i, i < n → (2 * n - 4) * 90 / n = 150) : n = 12 :=
sorry

end number_of_sides_of_regular_polygon_l224_224653


namespace min_value_fraction_geq_3_div_2_l224_224240

theorem min_value_fraction_geq_3_div_2 (a : ℕ → ℝ) (m n : ℕ) (q : ℝ) (h1 : q > 0) 
  (h2 : ∀ k, a (k + 2) = q * a (k + 1)) (h3 : a 2016 = a 2015 + 2 * a 2014) 
  (h4 : a m * a n = 16 * (a 1) ^ 2) :
  (∃ q, q = 2 ∧ m + n = 6) → 4 / m + 1 / n ≥ 3 / 2 :=
by sorry

end min_value_fraction_geq_3_div_2_l224_224240


namespace f_divisible_by_8_l224_224411

-- Define the function f
def f (n : ℕ) : ℕ := 5^n + 2 * 3^(n-1) + 1

-- Theorem statement
theorem f_divisible_by_8 (n : ℕ) (hn : n > 0) : 8 ∣ f n := sorry

end f_divisible_by_8_l224_224411


namespace bob_more_than_ken_l224_224402

def ken_situps : ℕ := 20

def nathan_situps : ℕ := 2 * ken_situps

def bob_situps : ℕ := (ken_situps + nathan_situps) / 2

theorem bob_more_than_ken : bob_situps - ken_situps = 10 := by
  -- proof steps to be filled in
  sorry

end bob_more_than_ken_l224_224402


namespace meal_cost_l224_224133

theorem meal_cost:
  ∀ (s c p k : ℝ), 
  (2 * s + 5 * c + 2 * p + 3 * k = 6.30) →
  (3 * s + 8 * c + 2 * p + 4 * k = 8.40) →
  (s + c + p + k = 3.15) :=
by
  intros s c p k h1 h2
  sorry

end meal_cost_l224_224133


namespace function_equality_l224_224233

theorem function_equality (f : ℝ → ℝ)
  (hf : ∀ x : ℝ, f (2 * x + 1) = 2 * x^2 + 1) :
  ∀ x : ℝ, f x = (1/2) * x^2 - x + (3/2) :=
by
  sorry

end function_equality_l224_224233


namespace parabola_hyperbola_focus_l224_224704

theorem parabola_hyperbola_focus (p : ℝ) :
  let parabolaFocus := (p / 2, 0)
  let hyperbolaRightFocus := (2, 0)
  (parabolaFocus = hyperbolaRightFocus) → p = 4 := 
by
  intro h
  sorry

end parabola_hyperbola_focus_l224_224704


namespace at_least_one_boy_selected_l224_224205

-- Define the number of boys and girls
def boys : ℕ := 6
def girls : ℕ := 2

-- Define the total group and the total selected
def total_people : ℕ := boys + girls
def selected_people : ℕ := 3

-- Statement: In any selection of 3 people from the group, the selection contains at least one boy
theorem at_least_one_boy_selected :
  ∀ (selection : Finset ℕ), selection.card = selected_people → selection.card > girls :=
sorry

end at_least_one_boy_selected_l224_224205


namespace num_ways_to_divide_friends_l224_224835

theorem num_ways_to_divide_friends :
  (4 : ℕ) ^ 8 = 65536 := by
  sorry

end num_ways_to_divide_friends_l224_224835


namespace fraction_interval_l224_224462

theorem fraction_interval :
  (5 / 24 > 1 / 6) ∧ (5 / 24 < 1 / 4) ∧
  (¬ (5 / 12 > 1 / 6 ∧ 5 / 12 < 1 / 4)) ∧
  (¬ (5 / 36 > 1 / 6 ∧ 5 / 36 < 1 / 4)) ∧
  (¬ (5 / 60 > 1 / 6 ∧ 5 / 60 < 1 / 4)) ∧
  (¬ (5 / 48 > 1 / 6 ∧ 5 / 48 < 1 / 4)) :=
by
  sorry

end fraction_interval_l224_224462


namespace beverage_distribution_l224_224194

theorem beverage_distribution (total_cans : ℕ) (number_of_children : ℕ) (hcans : total_cans = 5) (hchildren : number_of_children = 8) :
  (total_cans / number_of_children : ℚ) = 5 / 8 :=
by
  -- Given the conditions
  have htotal_cans : total_cans = 5 := hcans
  have hnumber_of_children : number_of_children = 8 := hchildren
  
  -- we need to show the beverage distribution
  rw [htotal_cans, hnumber_of_children]
  exact by norm_num

end beverage_distribution_l224_224194


namespace common_chord_length_l224_224596

/-- Two circles intersect such that each passes through the other's center.
Prove that the length of their common chord is 8√3 cm. -/
theorem common_chord_length (r : ℝ) (h : r = 8) :
  let chord_length := 2 * (r * (Real.sqrt 3 / 2))
  chord_length = 8 * Real.sqrt 3 := by
  sorry

end common_chord_length_l224_224596


namespace apples_to_pears_l224_224576

theorem apples_to_pears :
  (3 / 4) * 12 = 9 → (2 / 3) * 6 = 4 :=
by {
  sorry
}

end apples_to_pears_l224_224576


namespace lizzy_final_amount_l224_224562

-- Define constants
def m : ℕ := 80   -- cents from mother
def f : ℕ := 40   -- cents from father
def s : ℕ := 50   -- cents spent on candy
def u : ℕ := 70   -- cents from uncle
def t : ℕ := 90   -- cents for the toy
def c : ℕ := 110  -- cents change she received

-- Define the final amount calculation
def final_amount : ℕ := m + f - s + u - t + c

-- Prove the final amount is 160
theorem lizzy_final_amount : final_amount = 160 := by
  sorry

end lizzy_final_amount_l224_224562


namespace find_a_l224_224551

def A (x : ℝ) : Prop := x^2 + 6 * x < 0
def B (a x : ℝ) : Prop := x^2 - (a - 2) * x - 2 * a < 0
def U (x : ℝ) : Prop := -6 < x ∧ x < 5

theorem find_a : (∀ x, A x ∨ ∃ a, B a x) = U x -> a = 5 :=
by
  sorry

end find_a_l224_224551


namespace probability_exists_bc_l224_224107

open Probability

noncomputable def f : ℕ → ℕ := sorry
noncomputable def G (f : ℕ → ℕ) : ℕ := sorry

theorem probability_exists_bc {n : ℕ} (h_n : n > 0) :
  let f := λ (x : Fin n), Fin n in
  let a := (uniform (Fin n)) in
  let event := {a : Fin n | ∃ b c : ℕ, b ≥ 1 ∧ c ≥ 1 ∧ f^[b] 1 = a ∧ f^[c] a = 1 } in
  Pr[event] = 1 / n :=
by
  sorry

end probability_exists_bc_l224_224107


namespace ratio_of_speeds_l224_224742

theorem ratio_of_speeds
  (speed_of_tractor : ℝ)
  (speed_of_bike : ℝ)
  (speed_of_car : ℝ)
  (h1 : speed_of_tractor = 575 / 25)
  (h2 : speed_of_car = 331.2 / 4)
  (h3 : speed_of_bike = 2 * speed_of_tractor) :
  speed_of_car / speed_of_bike = 1.8 :=
by
  sorry

end ratio_of_speeds_l224_224742


namespace PropositionA_PropositionD_l224_224159

-- Proposition A: a > 1 is a sufficient but not necessary condition for 1/a < 1.
theorem PropositionA (a : ℝ) (h : a > 1) : 1 / a < 1 :=
by sorry

-- PropositionD: a ≠ 0 is a necessary but not sufficient condition for ab ≠ 0.
theorem PropositionD (a b : ℝ) (h : a ≠ 0) : a * b ≠ 0 :=
by sorry
 
end PropositionA_PropositionD_l224_224159


namespace cosine_60_degrees_l224_224643

theorem cosine_60_degrees : Real.cos (60 * Real.pi / 180) = 1 / 2 :=
by sorry

end cosine_60_degrees_l224_224643


namespace average_speed_additional_hours_l224_224327

theorem average_speed_additional_hours
  (time_first_part : ℝ) (speed_first_part : ℝ) (total_time : ℝ) (avg_speed_total : ℝ)
  (additional_hours : ℝ) (speed_additional_hours : ℝ) :
  time_first_part = 4 → speed_first_part = 35 → total_time = 24 → avg_speed_total = 50 →
  additional_hours = total_time - time_first_part →
  (time_first_part * speed_first_part + additional_hours * speed_additional_hours) / total_time = avg_speed_total →
  speed_additional_hours = 53 :=
by intros; sorry

end average_speed_additional_hours_l224_224327


namespace hydrogen_atoms_in_compound_l224_224473

theorem hydrogen_atoms_in_compound : 
  ∀ (Al_weight O_weight H_weight : ℕ) (total_weight : ℕ) (num_Al num_O num_H : ℕ),
  Al_weight = 27 →
  O_weight = 16 →
  H_weight = 1 →
  total_weight = 78 →
  num_Al = 1 →
  num_O = 3 →
  (num_Al * Al_weight + num_O * O_weight + num_H * H_weight = total_weight) →
  num_H = 3 := 
by
  intros
  sorry

end hydrogen_atoms_in_compound_l224_224473


namespace rate_of_first_car_l224_224595

theorem rate_of_first_car
  (r : ℕ) (h1 : 3 * r + 30 = 180) : r = 50 :=
sorry

end rate_of_first_car_l224_224595


namespace max_value_of_expression_l224_224256

-- We have three nonnegative real numbers a, b, and c,
-- such that a + b + c = 3.
def nonnegative (x : ℝ) := x ≥ 0

theorem max_value_of_expression (a b c : ℝ) (h1 : nonnegative a) (h2 : nonnegative b) (h3 : nonnegative c) (h4 : a + b + c = 3) :
  a + b^2 + c^4 ≤ 3 :=
  sorry

end max_value_of_expression_l224_224256


namespace allan_has_4_more_balloons_than_jake_l224_224796

namespace BalloonProblem

def initial_balloons_allan : Nat := 6
def initial_balloons_jake : Nat := 2
def additional_balloons_jake : Nat := 3
def additional_balloons_allan : Nat := 4
def given_balloons_jake : Nat := 2
def given_balloons_allan : Nat := 3

def final_balloons_allan : Nat := (initial_balloons_allan + additional_balloons_allan) - given_balloons_allan
def final_balloons_jake : Nat := (initial_balloons_jake + additional_balloons_jake) - given_balloons_jake

theorem allan_has_4_more_balloons_than_jake :
  final_balloons_allan = final_balloons_jake + 4 :=
by
  -- proof is skipped with sorry
  sorry

end BalloonProblem

end allan_has_4_more_balloons_than_jake_l224_224796


namespace abs_g_eq_abs_gx_l224_224353

noncomputable def g (x : ℝ) : ℝ :=
if -3 <= x ∧ x <= 0 then x^2 - 2 else -x + 2

noncomputable def abs_g (x : ℝ) : ℝ :=
if -3 <= x ∧ x <= -Real.sqrt 2 then x^2 - 2
else if -Real.sqrt 2 < x ∧ x <= Real.sqrt 2 then 2 - x^2
else if Real.sqrt 2 < x ∧ x <= 2 then 2 - x
else x - 2

theorem abs_g_eq_abs_gx (x : ℝ) (hx1 : -3 <= x ∧ x <= -Real.sqrt 2) 
  (hx2 : -Real.sqrt 2 < x ∧ x <= Real.sqrt 2)
  (hx3 : Real.sqrt 2 < x ∧ x <= 2)
  (hx4 : 2 < x ∧ x <= 3) :
  abs_g x = |g x| :=
by
  sorry

end abs_g_eq_abs_gx_l224_224353


namespace number_of_sides_of_regular_polygon_l224_224655

theorem number_of_sides_of_regular_polygon (n : ℕ) (h : 0 < n) (h_angle : ∀ i, i < n → (2 * n - 4) * 90 / n = 150) : n = 12 :=
sorry

end number_of_sides_of_regular_polygon_l224_224655


namespace cos_of_60_degrees_is_half_l224_224626

noncomputable theory

def unit_circle_coordinates := (1/2 : ℝ, real.sqrt 3 / 2)

theorem cos_of_60_degrees_is_half :
  ∀ θ : ℝ, θ = 60 → unit_circle_coordinates.fst = real.cos θ :=
begin
  sorry
end

end cos_of_60_degrees_is_half_l224_224626


namespace tree_height_l224_224481

theorem tree_height (BR MH MB MR TB : ℝ)
  (h_cond1 : BR = 5)
  (h_cond2 : MH = 1.8)
  (h_cond3 : MB = 1)
  (h_cond4 : MR = BR - MB)
  (h_sim : TB / BR = MH / MR)
  : TB = 2.25 :=
by sorry

end tree_height_l224_224481


namespace find_X_l224_224109

def r (X Y : ℕ) : ℕ := X^2 + Y^2

theorem find_X (X : ℕ) (h : r X 7 = 338) : X = 17 := by
  sorry

end find_X_l224_224109


namespace count_final_numbers_l224_224123

def initial_numbers : List ℕ := List.range' 1 20

def erased_even_numbers : List ℕ :=
  initial_numbers.filter (λ n => n % 2 ≠ 0)

def final_numbers : List ℕ :=
  erased_even_numbers.filter (λ n => n % 5 ≠ 4)

theorem count_final_numbers : final_numbers.length = 8 :=
by
  -- Here, we would proceed to prove the statement.
  sorry

end count_final_numbers_l224_224123


namespace probability_of_picking_peach_l224_224708

-- Define the counts of each type of fruit
def apples : ℕ := 5
def pears : ℕ := 3
def peaches : ℕ := 2

-- Define the total number of fruits
def total_fruits : ℕ := apples + pears + peaches

-- Define the probability of picking a peach
def probability_of_peach : ℚ := peaches / total_fruits

-- State the theorem
theorem probability_of_picking_peach : probability_of_peach = 1/5 := by
  -- proof goes here
  sorry

end probability_of_picking_peach_l224_224708


namespace axis_of_symmetry_parabola_l224_224683

theorem axis_of_symmetry_parabola (x y : ℝ) :
  x^2 + 2*x*y + y^2 + 3*x + y = 0 → x + y + 1 = 0 :=
by {
  sorry
}

end axis_of_symmetry_parabola_l224_224683


namespace intersection_M_N_l224_224380

def M : Set ℝ := {x : ℝ | |x| < 1}
def N : Set ℝ := {x : ℝ | x^2 - x < 0}

theorem intersection_M_N :
  M ∩ N = {x : ℝ | 0 < x ∧ x < 1} := by
  sorry

end intersection_M_N_l224_224380


namespace all_points_lie_on_circle_l224_224070

theorem all_points_lie_on_circle {s : ℝ} :
  let x := (s^2 - 1) / (s^2 + 1)
  let y := (2 * s) / (s^2 + 1)
  x^2 + y^2 = 1 :=
by
  sorry

end all_points_lie_on_circle_l224_224070


namespace max_non_attacking_rooks_l224_224779

theorem max_non_attacking_rooks (a b : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 299) (h3 : 1 ≤ b) (h4 : b ≤ 299) :
  ∃ max_rooks : ℕ, max_rooks = 400 :=
  sorry

end max_non_attacking_rooks_l224_224779


namespace area_of_triangle_FYG_l224_224452

theorem area_of_triangle_FYG (EF GH : ℝ) 
  (EF_len : EF = 15) 
  (GH_len : GH = 25) 
  (area_trapezoid : 0.5 * (EF + GH) * 10 = 200) 
  (intersection : true) -- Placeholder for intersection condition
  : 0.5 * GH * 3.75 = 46.875 := 
sorry

end area_of_triangle_FYG_l224_224452


namespace min_value_expression_l224_224408

theorem min_value_expression (α β : ℝ) :
  (3 * Real.cos α + 4 * Real.sin β - 10)^2 + (3 * Real.sin α + 4 * Real.cos β - 12)^2 ≥ 215 :=
by
  sorry

end min_value_expression_l224_224408


namespace pyramid_height_l224_224336

theorem pyramid_height (perimeter_side_base : ℝ) (apex_distance_to_vertex : ℝ) (height_peak_to_center_base : ℝ) : 
  (perimeter_side_base = 32) → (apex_distance_to_vertex = 12) → 
  height_peak_to_center_base = 4 * Real.sqrt 7 := 
  by
    sorry

end pyramid_height_l224_224336


namespace least_homeowners_l224_224769

theorem least_homeowners (M W : ℕ) (total_members : M + W = 150)
  (men_homeowners : ∃ n : ℕ, n = 10 * M / 100) 
  (women_homeowners : ∃ n : ℕ, n = 20 * W / 100) : 
  ∃ homeowners : ℕ, homeowners = 16 := 
sorry

end least_homeowners_l224_224769


namespace average_abc_l224_224146

theorem average_abc (A B C : ℚ) 
  (h1 : 2002 * C - 3003 * A = 6006) 
  (h2 : 2002 * B + 4004 * A = 8008) 
  (h3 : B - C = A + 1) :
  (A + B + C) / 3 = 7 / 3 := 
sorry

end average_abc_l224_224146


namespace circle_sum_l224_224248

theorem circle_sum :
  ∃ (a b c d e f : ℕ),
    {a, b, c, d, e, f} = {1, 2, 3, 4, 5, 6} ∧
    a + b + c = 14 ∧ d + e + f = 14 ∧ a + d + e = 14 :=
begin
  sorry
end

end circle_sum_l224_224248


namespace cost_price_percentage_l224_224299

theorem cost_price_percentage (MP CP SP : ℝ) (h1 : SP = 0.88 * MP) (h2 : SP = 1.375 * CP) :
  (CP / MP) * 100 = 64 :=
by
  sorry

end cost_price_percentage_l224_224299


namespace shoes_count_l224_224922

def numberOfShoes (numPairs : Nat) (matchingPairProbability : ℚ) : Nat :=
  let S := numPairs * 2
  if (matchingPairProbability = 1 / (S - 1))
  then S
  else 0

theorem shoes_count 
(numPairs : Nat)
(matchingPairProbability : ℚ)
(hp : numPairs = 9)
(hq : matchingPairProbability = 0.058823529411764705) :
numberOfShoes numPairs matchingPairProbability = 18 := 
by
  -- definition only, the proof is not required
  sorry

end shoes_count_l224_224922


namespace statement_a_statement_b_statement_c_l224_224235

theorem statement_a (a b : ℝ) (h1 : 1 ≤ a + b ∧ a + b ≤ 5) (h2 : -1 ≤ a - b ∧ a - b ≤ 3) :
  0 ≤ a ∧ a ≤ 4 := sorry

theorem statement_b (a b : ℝ) (h1 : 1 ≤ a + b ∧ a + b ≤ 5) (h2 : -1 ≤ a - b ∧ a - b ≤ 3) :
  -1 ≤ b ∧ b ≤ 3 := sorry

theorem statement_c (a b : ℝ) (h1 : 1 ≤ a + b ∧ a + b ≤ 5) (h2 : -1 ≤ a - b ∧ a - b ≤ 3) :
  -2 ≤ 3 * a - 2 * b ∧ 3 * a - 2 * b ≤ 10 := sorry

end statement_a_statement_b_statement_c_l224_224235


namespace save_after_increase_l224_224478

def monthly_saving_initial (salary : ℕ) (saving_percentage : ℕ) : ℕ :=
  salary * saving_percentage / 100

def monthly_expense_initial (salary : ℕ) (saving : ℕ) : ℕ :=
  salary - saving

def increase_by_percentage (amount : ℕ) (percentage : ℕ) : ℕ :=
  amount * percentage / 100

def new_expense (initial_expense : ℕ) (increase : ℕ) : ℕ :=
  initial_expense + increase

def new_saving (salary : ℕ) (expense : ℕ) : ℕ :=
  salary - expense

theorem save_after_increase (salary saving_percentage increase_percentage : ℕ) 
  (H_salary : salary = 5500) 
  (H_saving_percentage : saving_percentage = 20) 
  (H_increase_percentage : increase_percentage = 20) :
  new_saving salary (new_expense (monthly_expense_initial salary (monthly_saving_initial salary saving_percentage)) (increase_by_percentage (monthly_expense_initial salary (monthly_saving_initial salary saving_percentage)) increase_percentage)) = 220 := 
by
  sorry

end save_after_increase_l224_224478


namespace find_f_3_l224_224230

def f (x : ℝ) (a b : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

theorem find_f_3 (a b : ℝ) (h : f (-3) a b = 10) : f 3 a b = -26 :=
by sorry

end find_f_3_l224_224230


namespace train_speed_correct_l224_224185

def train_length : ℝ := 250  -- length of the train in meters
def time_to_pass : ℝ := 18  -- time to pass a tree in seconds
def speed_of_train_km_hr : ℝ := 50  -- speed of the train in km/hr

theorem train_speed_correct :
  (train_length / time_to_pass) * (3600 / 1000) = speed_of_train_km_hr :=
by
  sorry

end train_speed_correct_l224_224185


namespace quadrilateral_area_inequality_l224_224787

theorem quadrilateral_area_inequality 
  (T : ℝ) (a b c d e f : ℝ) (φ : ℝ) 
  (hT : T = (1/2) * e * f * Real.sin φ) 
  (hptolemy : e * f ≤ a * c + b * d) : 
  2 * T ≤ a * c + b * d := 
sorry

end quadrilateral_area_inequality_l224_224787


namespace Maggie_age_l224_224252

theorem Maggie_age (Kate Maggie Sue : ℕ) (h1 : Kate + Maggie + Sue = 48) (h2 : Kate = 19) (h3 : Sue = 12) : Maggie = 17 := by
  sorry

end Maggie_age_l224_224252


namespace determine_value_of_e_l224_224108

theorem determine_value_of_e {a b c d e : ℝ} (h1 : a < b) (h2 : b < c) (h3 : c < d) (h4 : d < e) 
    (h5 : a + b = 32) (h6 : a + c = 36) (h7 : b + c = 37 ∨ a + d = 37) 
    (h8 : c + e = 48) (h9 : d + e = 51) : e = 27.5 :=
sorry

end determine_value_of_e_l224_224108


namespace product_of_possible_b_values_l224_224136

theorem product_of_possible_b_values (b : ℝ) :
  (∀ (y1 y2 x1 x2 : ℝ), y1 = -1 ∧ y2 = 3 ∧ x1 = 2 ∧ (x2 = b) ∧ (y2 - y1 = 4) → 
   (b = 2 + 4 ∨ b = 2 - 4)) → 
  (b = 6 ∨ b = -2) → (b = 6) ∧ (b = -2) → 6 * -2 = -12 :=
sorry

end product_of_possible_b_values_l224_224136


namespace ways_to_divide_8_friends_l224_224832

theorem ways_to_divide_8_friends : (4 : ℕ) ^ 8 = 65536 := by
  sorry

end ways_to_divide_8_friends_l224_224832


namespace number_of_kids_stay_home_l224_224407

def total_kids : ℕ := 313473
def kids_at_camp : ℕ := 38608
def kids_stay_home : ℕ := 274865

theorem number_of_kids_stay_home :
  total_kids - kids_at_camp = kids_stay_home := 
by
  -- Subtracting the number of kids who go to camp from the total number of kids
  sorry

end number_of_kids_stay_home_l224_224407


namespace find_fraction_l224_224948

theorem find_fraction (a b : ℝ) (h : a = 2 * b) : (a / (a - b)) = 2 :=
by
  sorry

end find_fraction_l224_224948


namespace num_ways_to_divide_friends_l224_224834

theorem num_ways_to_divide_friends :
  (4 : ℕ) ^ 8 = 65536 := by
  sorry

end num_ways_to_divide_friends_l224_224834


namespace trees_left_after_typhoon_and_growth_l224_224695

-- Conditions
def initial_trees : ℕ := 9
def trees_died_in_typhoon : ℕ := 4
def new_trees : ℕ := 5

-- Question (Proof Problem)
theorem trees_left_after_typhoon_and_growth : 
  initial_trees - trees_died_in_typhoon + new_trees = 10 := 
by
  sorry

end trees_left_after_typhoon_and_growth_l224_224695


namespace dot_product_u_v_l224_224062

def u : ℝ × ℝ × ℝ × ℝ := (4, -3, 5, -2)
def v : ℝ × ℝ × ℝ × ℝ := (-6, 1, 2, 3)

theorem dot_product_u_v : (4 * -6 + -3 * 1 + 5 * 2 + -2 * 3) = -23 := by
  sorry

end dot_product_u_v_l224_224062


namespace scientific_notation_80000000_l224_224433

-- Define the given number
def number : ℕ := 80000000

-- Define the scientific notation form
def scientific_notation (n k : ℕ) (a : ℝ) : Prop :=
  n = (a * (10 : ℝ) ^ k)

-- The theorem to prove scientific notation of 80,000,000
theorem scientific_notation_80000000 : scientific_notation number 7 8 :=
by {
  sorry
}

end scientific_notation_80000000_l224_224433


namespace jeff_corrected_mean_l224_224547

def initial_scores : List ℕ := [85, 90, 87, 93, 89, 84, 88]

def corrected_scores : List ℕ := [85, 90, 92, 93, 89, 89, 88]

noncomputable def arithmetic_mean (scores : List ℕ) : ℝ :=
  (scores.sum : ℝ) / (scores.length : ℝ)

theorem jeff_corrected_mean :
  arithmetic_mean corrected_scores = 89.42857142857143 := 
by
  sorry

end jeff_corrected_mean_l224_224547


namespace inequality_solution_1_inequality_solution_2_l224_224027

-- Definition for part 1
theorem inequality_solution_1 (x : ℝ) : x^2 + 3*x - 4 > 0 ↔ x > 1 ∨ x < -4 :=
sorry

-- Definition for part 2
theorem inequality_solution_2 (x : ℝ) : (1 - x) / (x - 5) ≥ 1 ↔ 3 ≤ x ∧ x < 5 :=
sorry

end inequality_solution_1_inequality_solution_2_l224_224027


namespace first_digit_base_4_of_853_l224_224906

theorem first_digit_base_4_of_853 : 
  ∃ (d : ℕ), d = 3 ∧ (d * 256 ≤ 853 ∧ 853 < (d + 1) * 256) :=
by
  sorry

end first_digit_base_4_of_853_l224_224906


namespace value_of_N_l224_224362

theorem value_of_N (N : ℕ) (x y z w s : ℕ) (h_pos_x : 0 < x) (h_pos_y : 0 < y)
    (h_pos_z : 0 < z) (h_pos_w : 0 < w) (h_pos_s : 0 < s) (h_sum : x + y + z + w + s = N)
    (h_comb : Nat.choose N 4 = 3003) : N = 18 := 
by
  sorry

end value_of_N_l224_224362


namespace union_A_B_l224_224819

open Set

def A := {x : ℝ | x * (x - 2) < 3}
def B := {x : ℝ | 5 / (x + 1) ≥ 1}
def U := {x : ℝ | -1 < x ∧ x ≤ 4}

theorem union_A_B : A ∪ B = U := 
sorry

end union_A_B_l224_224819


namespace calculate_expression_l224_224113

def f (x : ℝ) := 2 * x^2 - 3 * x + 1
def g (x : ℝ) := x + 2

theorem calculate_expression : f (1 + g 3) = 55 := 
by
  sorry

end calculate_expression_l224_224113


namespace sum_of_series_l224_224267

theorem sum_of_series (a b : ℝ) (h_pos : 0 < a ∧ 0 < b) (h_gt : a > b) :
  ∑' n, 1 / ( ((n - 1) * a + (n - 2) * b) * (n * a + (n - 1) * b)) = 1 / ((a + b) * b) :=
by
  sorry

end sum_of_series_l224_224267


namespace line_passes_through_parabola_vertex_l224_224069

theorem line_passes_through_parabola_vertex :
  ∃ (a : ℝ), (∃ (b : ℝ), b = a ∧ (a = 0 ∨ a = 1)) :=
by
  sorry

end line_passes_through_parabola_vertex_l224_224069


namespace syllogism_correct_l224_224244

-- Hypotheses for each condition
def OptionA := "The first section, the second section, the third section"
def OptionB := "Major premise, minor premise, conclusion"
def OptionC := "Induction, conjecture, proof"
def OptionD := "Dividing the discussion into three sections"

-- Definition of a syllogism in deductive reasoning
def syllogism_def := "A logical argument that applies deductive reasoning to arrive at a conclusion based on two propositions assumed to be true"

-- Theorem stating that a syllogism corresponds to Option B
theorem syllogism_correct :
  syllogism_def = OptionB :=
by
  sorry

end syllogism_correct_l224_224244


namespace emergency_vehicle_reachable_area_l224_224099

theorem emergency_vehicle_reachable_area :
  let speed_roads := 60 -- velocity on roads in miles per hour
    let speed_sand := 10 -- velocity on sand in miles per hour
    let time_limit := 5 / 60 -- time limit in hours
    let max_distance_on_roads := speed_roads * time_limit -- max distance on roads
    let radius_sand_circle := (10 / 12) -- radius on the sand
    -- calculate area covered
  (5 * 5 + 4 * (1 / 4 * Real.pi * (radius_sand_circle)^2)) = (25 + (25 * Real.pi) / 36) :=
by
  sorry

end emergency_vehicle_reachable_area_l224_224099


namespace prove_a_minus_b_plus_c_eq_3_l224_224112

variable {a b c m n : ℝ}

theorem prove_a_minus_b_plus_c_eq_3 
    (h : ∀ x : ℝ, m * x^2 - n * x + 3 = a * (x - 1)^2 + b * (x - 1) + c) :
    a - b + c = 3 :=
sorry

end prove_a_minus_b_plus_c_eq_3_l224_224112


namespace product_lcm_gcd_eq_2160_l224_224907

theorem product_lcm_gcd_eq_2160 :
  let a := 36
  let b := 60
  lcm a b * gcd a b = 2160 := by
  sorry

end product_lcm_gcd_eq_2160_l224_224907


namespace angle_BAC_in_isosceles_triangle_l224_224103

theorem angle_BAC_in_isosceles_triangle
  (A B C D : Type)
  (AB AC : ℝ)
  (BD DC : ℝ)
  (angle_BDA : ℝ)
  (isosceles_triangle : AB = AC)
  (midpoint_D : BD = DC)
  (external_angle_D : angle_BDA = 120) :
  ∃ (angle_BAC : ℝ), angle_BAC = 60 :=
by
  sorry

end angle_BAC_in_isosceles_triangle_l224_224103


namespace f_2009_value_l224_224677

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom even_function (f : ℝ → ℝ) : ∀ x, f x = f (-x)
axiom odd_function (g : ℝ → ℝ) : ∀ x, g x = -g (-x)
axiom f_value : f 1 = 0
axiom g_def : ∀ x, g x = f (x - 1)

theorem f_2009_value : f 2009 = 0 :=
by
  sorry

end f_2009_value_l224_224677


namespace problem_statement_l224_224261

def t (x : ℝ) : ℝ := 4 * x - 9
def s (y : ℝ) : ℝ := (Real.sqrtSq' (1 / 4 * (y + 9)))^2 + 4 * (Real.sqrtSq' (1 / 4 * (y + 9))) - 5

theorem problem_statement : s 1 = 11.25 := by
  sorry

end problem_statement_l224_224261


namespace price_of_first_variety_l224_224431

theorem price_of_first_variety (P : ℝ) (h1 : 1 * P + 1 * 135 + 2 * 175.5 = 153 * 4) : P = 126 :=
sorry

end price_of_first_variety_l224_224431


namespace solve_for_t_l224_224176

variables (V0 V g a t S : ℝ)

-- Given conditions
def velocity_eq : Prop := V = (g + a) * t + V0
def displacement_eq : Prop := S = (1/2) * (g + a) * t^2 + V0 * t

-- The theorem to prove
theorem solve_for_t (h1 : velocity_eq V0 V g a t)
                    (h2 : displacement_eq V0 g a t S) :
  t = 2 * S / (V + V0) :=
sorry

end solve_for_t_l224_224176


namespace cost_price_to_marked_price_l224_224437

theorem cost_price_to_marked_price (MP CP SP : ℝ)
  (h1 : SP = MP * 0.87)
  (h2 : SP = CP * 1.359375) :
  (CP / MP) * 100 = 64 := by
  sorry

end cost_price_to_marked_price_l224_224437


namespace age_of_B_l224_224237

theorem age_of_B (A B : ℕ) (h1 : A + 10 = 2 * (B - 10)) (h2 : A = B + 9) : B = 39 := by
  sorry

end age_of_B_l224_224237


namespace harry_bought_l224_224447

-- Definitions based on the conditions
def initial_bottles := 35
def jason_bought := 5
def final_bottles := 24

-- Theorem stating the number of bottles Harry bought
theorem harry_bought :
  (initial_bottles - jason_bought) - final_bottles = 6 :=
by
  sorry

end harry_bought_l224_224447


namespace base_conversion_sum_correct_l224_224496

theorem base_conversion_sum_correct :
  (253 / 8 / 13 / 3 + 245 / 7 / 35 / 6 : ℚ) = 339 / 23 := sorry

end base_conversion_sum_correct_l224_224496


namespace molecular_weight_of_4_moles_AlCl3_is_correct_l224_224320

/-- The atomic weight of aluminum (Al) is 26.98 g/mol. -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of chlorine (Cl) is 35.45 g/mol. -/
def atomic_weight_Cl : ℝ := 35.45

/-- A molecule of AlCl3 consists of 1 atom of Al and 3 atoms of Cl. -/
def molecular_weight_AlCl3 := (1 * atomic_weight_Al) + (3 * atomic_weight_Cl)

/-- The total weight of 4 moles of AlCl3. -/
def total_weight_4_moles_AlCl3 := 4 * molecular_weight_AlCl3

/-- We prove that the total weight of 4 moles of AlCl3 is 533.32 g. -/
theorem molecular_weight_of_4_moles_AlCl3_is_correct :
  total_weight_4_moles_AlCl3 = 533.32 :=
sorry

end molecular_weight_of_4_moles_AlCl3_is_correct_l224_224320


namespace lake_crystal_frogs_percentage_l224_224855

noncomputable def percentage_fewer_frogs (frogs_in_lassie_lake total_frogs : ℕ) : ℕ :=
  let P := (total_frogs - frogs_in_lassie_lake) * 100 / frogs_in_lassie_lake
  P

theorem lake_crystal_frogs_percentage :
  let frogs_in_lassie_lake := 45
  let total_frogs := 81
  percentage_fewer_frogs frogs_in_lassie_lake total_frogs = 20 :=
by
  sorry

end lake_crystal_frogs_percentage_l224_224855


namespace amoeba_doubling_time_l224_224747

theorem amoeba_doubling_time (H1 : ∀ t : ℕ, t = 60 → 2^(t / 3) = 2^20) :
  ∀ t : ℕ, 2 * 2^(t / 3) = 2^20 → t = 57 :=
by
  intro t
  intro H2
  sorry

end amoeba_doubling_time_l224_224747


namespace raft_drift_time_l224_224003

-- Define the distance between the villages
def distance : ℝ := 1

-- Define the speed of the steamboat in still water (in units/hour)
def v_s : ℝ := sorry

-- Define the time taken by the steamboat to travel the distance (in hours)
def t_steamboat : ℝ := 1

-- Define the speed of the motorboat in still water (in units/hour)
def v_m : ℝ := 2 * v_s

-- Define the time taken by the motorboat to travel the distance (in hours)
def t_motorboat : ℝ := 45 / 60

-- Define the speed of the river's current (in units/hour)
def v_f : ℝ := sorry

-- Equations for steamboat and motorboat effective speeds
def steamboat_equation := v_s + v_f = distance / t_steamboat
def motorboat_equation := v_m + v_f = distance / t_motorboat

-- Solve for the speeds
def v_s_solution : ℝ := 1 - v_f
def v_f_solution : ℝ := (4 / 3) - 2 * v_s_solution

-- Define the time for the raft to drift from Verkhnie Vasyuki to Nizhnie Vasyuki (in hours)
def raft_time_hours : ℝ := distance / v_f_solution

-- Convert the time to minutes
def raft_time_minutes : ℝ := raft_time_hours * 60

-- Theorem statement that proves the raft drifts in 90 minutes
theorem raft_drift_time : raft_time_minutes = 90 :=
by
  unfold distance v_s t_steamboat v_m t_motorboat v_f steamboat_equation motorboat_equation v_s_solution v_f_solution raft_time_hours raft_time_minutes
  rw [←solve_v_s, ←solve_v_f]
  sorry

end raft_drift_time_l224_224003


namespace index_card_area_l224_224614

theorem index_card_area (length width : ℕ) (h_length : length = 5) (h_width : width = 7)
  (h_area_shortened_length : (length - 2) * width = 21) : (length * (width - 2)) = 25 := by
  sorry

end index_card_area_l224_224614


namespace find_r_in_parallelogram_l224_224392

theorem find_r_in_parallelogram 
  (θ : ℝ) 
  (r : ℝ)
  (CAB DBA DBC ACB AOB : ℝ)
  (h1 : CAB = 3 * DBA)
  (h2 : DBC = 2 * DBA)
  (h3 : ACB = r * (t * AOB))
  (h4 : t = 4 / 3)
  (h5 : AOB = 180 - 2 * DBA)
  (h6 : ACB = 180 - 4 * DBA) :
  r = 1 / 3 :=
by
  sorry

end find_r_in_parallelogram_l224_224392


namespace stratified_sampling_workshops_l224_224384

theorem stratified_sampling_workshops (units_A units_B units_C sample_B n : ℕ) 
(hA : units_A = 96) 
(hB : units_B = 84) 
(hC : units_C = 60) 
(hSample_B : sample_B = 7) 
(hn : (sample_B : ℚ) / n = (units_B : ℚ) / (units_A + units_B + units_C)) : 
  n = 70 :=
  by
  sorry

end stratified_sampling_workshops_l224_224384


namespace find_limpet_shells_l224_224500

variable (L L_shells E_shells J_shells totalShells : ℕ)

def Ed_and_Jacob_initial_shells := 2
def Ed_oyster_shells := 2
def Ed_conch_shells := 4
def Jacob_more_shells := 2
def total_shells := 30

def Ed_total_shells := L + Ed_oyster_shells + Ed_conch_shells
def Jacob_total_shells := Ed_total_shells + Jacob_more_shells

theorem find_limpet_shells
  (H : Ed_and_Jacob_initial_shells + Ed_total_shells + Jacob_total_shells = total_shells) :
  L = 7 :=
by
  sorry

end find_limpet_shells_l224_224500


namespace ellipse_transform_circle_l224_224354

theorem ellipse_transform_circle (a b x y : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a ≠ b)
  (h_ellipse : x^2 / a^2 + y^2 / b^2 = 1)
  (y' : ℝ)
  (h_transform : y' = (a / b) * y) :
  x^2 + y'^2 = a^2 :=
by
  sorry

end ellipse_transform_circle_l224_224354


namespace base_k_to_decimal_l224_224094

theorem base_k_to_decimal (k : ℕ) (h : 0 < k ∧ k < 10) : 
  1 * k^2 + 7 * k + 5 = 125 → k = 8 := 
by
  sorry

end base_k_to_decimal_l224_224094


namespace find_a_l224_224680

theorem find_a (a b : ℝ) (h1 : a * (a - 4) = 5) (h2 : b * (b - 4) = 5) (h3 : a ≠ b) (h4 : a + b = 4) : a = -1 :=
by 
sorry

end find_a_l224_224680


namespace regular_polygon_sides_l224_224666

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, 1 ≤ i → i ≤ n → (interior_angle i = 150) ) 
  (sum_interior_angles : ∑ i in range n, interior_angle i = 180 * (n - 2))
  {interior_angle : ℕ → ℕ} :
  n = 12 :=
by
  sorry

end regular_polygon_sides_l224_224666


namespace no_perfect_square_in_range_l224_224530

def isPerfectSquare (x : ℕ) : Prop :=
  ∃ (k : ℕ), k * k = x

theorem no_perfect_square_in_range :
  ∀ (n : ℕ), 4 ≤ n ∧ n ≤ 12 → ¬ isPerfectSquare (2*n*n + 3*n + 2) :=
by
  intro n
  intro h
  sorry

end no_perfect_square_in_range_l224_224530


namespace max_a_value_l224_224525

noncomputable def f (x : ℝ) : ℝ := (1 + Real.log x) / x - 1

theorem max_a_value : 
  ∀ (a : ℝ), 
  (∀ (x : ℝ), x ∈ Set.Icc (1/2) 2 → (a + 1) * x - 1 - Real.log x ≤ 0) → 
  a ≤ 1 - 2 * Real.log 2 := 
by
  sorry

end max_a_value_l224_224525


namespace alternating_intersections_l224_224098

theorem alternating_intersections (n : ℕ)
  (roads : Fin n → ℝ → ℝ) -- Roads are functions from reals to reals
  (h_straight : ∀ (i : Fin n), ∃ (a b : ℝ), ∀ x, roads i x = a * x + b) 
  (h_intersect : ∀ (i j : Fin n), i ≠ j → ∃ x, roads i x = roads j x)
  (h_two_roads : ∀ (x y : ℝ), ∃! (i j : Fin n), i ≠ j ∧ roads i x = roads j y) :
  ∃ (design : ∀ (i : Fin n), ℝ → Prop), 
  -- ensuring alternation, road 'i' alternates crossings with other roads 
  (∀ (i : Fin n) (x y : ℝ), 
    roads i x = roads i y → (design i x ↔ ¬design i y)) := sorry

end alternating_intersections_l224_224098


namespace equation_negative_roots_iff_l224_224583

theorem equation_negative_roots_iff (a : ℝ) :
  (∃ x < 0, 4^x - 2^(x-1) + a = 0) ↔ (-1/2 < a ∧ a ≤ 1/16) := 
sorry

end equation_negative_roots_iff_l224_224583


namespace walnut_trees_total_l224_224592

theorem walnut_trees_total : 33 + 44 = 77 :=
by
  sorry

end walnut_trees_total_l224_224592


namespace Albaszu_machine_productivity_l224_224892

theorem Albaszu_machine_productivity (x : ℝ) 
  (h1 : 1.5 * x = 25) : x = 16 := 
by 
  sorry

end Albaszu_machine_productivity_l224_224892


namespace locus_of_circumcenter_l224_224713

theorem locus_of_circumcenter (θ : ℝ) :
  let M := (3, 3 * Real.tan (θ - Real.pi / 3))
  let N := (3, 3 * Real.tan θ)
  let C := (3 / 2, 3 / 2 * Real.tan θ)
  ∃ (x y : ℝ), (x - 4) ^ 2 / 4 - y ^ 2 / 12 = 1 :=
by
  sorry

end locus_of_circumcenter_l224_224713


namespace evaluate_expression_l224_224059

theorem evaluate_expression : 
  (3^1002 + 7^1003)^2 - (3^1002 - 7^1003)^2 = 1372 * 10^1003 := 
by sorry

end evaluate_expression_l224_224059


namespace average_weight_increase_l224_224435

theorem average_weight_increase (old_weight : ℕ) (new_weight : ℕ) (n : ℕ) (increase : ℕ) :
  old_weight = 45 → new_weight = 93 → n = 8 → increase = (new_weight - old_weight) / n → increase = 6 :=
by
  intros h_old h_new h_n h_increase
  rw [h_old, h_new, h_n] at h_increase
  simp at h_increase
  exact h_increase

end average_weight_increase_l224_224435


namespace expression_rewrite_l224_224282

theorem expression_rewrite :
  ∃ (d r s : ℚ), (∀ k : ℚ, 8*k^2 - 6*k + 16 = d*(k + r)^2 + s) ∧ s / r = -118 / 3 :=
by sorry

end expression_rewrite_l224_224282


namespace find_n_l224_224774

theorem find_n (n : ℕ) : (8 : ℝ)^(1/3) = (2 : ℝ)^n → n = 1 := by
  sorry

end find_n_l224_224774


namespace triangle_is_right_triangle_l224_224097

theorem triangle_is_right_triangle 
  {A B C : ℝ} {a b c : ℝ} 
  (h₁ : b - a * Real.cos B = a * Real.cos C - c) 
  (h₂ : ∀ (angle : ℝ), 0 < angle ∧ angle < π) : A = π / 2 := 
sorry

end triangle_is_right_triangle_l224_224097


namespace fortieth_sequence_number_l224_224935

theorem fortieth_sequence_number :
  (∃ r n : ℕ, ((r * (r + 1)) - 40 = n) ∧ (40 ≤ r * (r + 1)) ∧ (40 > (r - 1) * r) ∧ n = 2 * r) :=
sorry

end fortieth_sequence_number_l224_224935


namespace gen_formula_is_arith_seq_l224_224692

-- Given: The sum of the first n terms of the sequence {a_n} is S_n = n^2 + 2n
def sum_seq (S : ℕ → ℕ) := ∀ n : ℕ, S n = n^2 + 2 * n

-- The general formula for {a_n} is a_n = 2n + 1
theorem gen_formula (S : ℕ → ℕ) (h : sum_seq S) : ∀ n : ℕ,  n > 0 → (∃ a : ℕ → ℕ, a n = 2 * n + 1 ∧ ∀ m : ℕ, m < n → a m = S (m + 1) - S m) :=
by sorry

-- The sequence {a_n} defined by a_n = 2n + 1 is an arithmetic sequence
theorem is_arith_seq : ∀ n : ℕ, n > 0 → (∀ a : ℕ → ℕ, (∀ k, k > 0 → a k = 2 * k + 1) → ∃ d : ℕ, d = 2 ∧ ∀ j > 0, a j - a (j - 1) = d) :=
by sorry

end gen_formula_is_arith_seq_l224_224692


namespace values_of_x_l224_224258

def f (x : ℝ) : ℝ := x^2 - 5 * x

theorem values_of_x (x : ℝ) : f (f x) = f x → x = 0 ∨ x = -2 ∨ x = 5 ∨ x = 6 :=
by {
  sorry
}

end values_of_x_l224_224258


namespace square_side_length_l224_224041

theorem square_side_length (s : ℚ) (h : s^2 = 9/16) : s = 3/4 := 
by
sorry

end square_side_length_l224_224041


namespace price_of_tea_mixture_l224_224295

theorem price_of_tea_mixture 
  (p1 p2 p3 : ℝ) 
  (q1 q2 q3 : ℝ) 
  (h_p1 : p1 = 126) 
  (h_p2 : p2 = 135) 
  (h_p3 : p3 = 173.5) 
  (h_q1 : q1 = 1) 
  (h_q2 : q2 = 1) 
  (h_q3 : q3 = 2) : 
  (p1 * q1 + p2 * q2 + p3 * q3) / (q1 + q2 + q3) = 152 := 
by 
  sorry

end price_of_tea_mixture_l224_224295


namespace sin_min_period_set_l224_224746

-- Function definition
def f (ω : ℝ) (x : ℝ) : ℝ :=
  2 * Real.sin (ω * x - Real.pi / 6)

-- Lean statement
theorem sin_min_period_set (ω : ℝ) (hω : ω > 0) (T : ℝ) (hT : T = 4 * Real.pi) :
  (∀ x : ℝ, f ω x = 2 * Real.sin (ω * x - Real.pi / 6)) →
  T = (2 * Real.pi / ω) →
  {x | ∃ k : ℤ, x = 4 * (k : ℝ) * Real.pi - 2 * Real.pi / 3} =
  {x | f ω x = -2} :=
by
  -- You can add the actual proof here.
  sorry

end sin_min_period_set_l224_224746


namespace count_rhombuses_in_large_triangle_l224_224342

-- Definitions based on conditions
def large_triangle_side_length : ℕ := 10
def small_triangle_side_length : ℕ := 1
def small_triangle_count : ℕ := 100
def rhombuses_of_8_triangles := 84

-- Problem statement in Lean 4
theorem count_rhombuses_in_large_triangle :
  ∀ (large_side small_side small_count : ℕ),
  large_side = large_triangle_side_length →
  small_side = small_triangle_side_length →
  small_count = small_triangle_count →
  (∃ (rhombus_count : ℕ), rhombus_count = rhombuses_of_8_triangles) :=
by
  intros large_side small_side small_count h_large h_small h_count
  use 84
  sorry

end count_rhombuses_in_large_triangle_l224_224342


namespace negation_of_every_function_has_parity_l224_224915

-- Assume the initial proposition
def every_function_has_parity := ∀ f : ℕ → ℕ, ∃ (p : ℕ), p = 0 ∨ p = 1

-- Negation of the original proposition
def exists_function_without_parity := ∃ f : ℕ → ℕ, ∀ p : ℕ, p ≠ 0 ∧ p ≠ 1

-- The theorem to prove
theorem negation_of_every_function_has_parity : 
  ¬ every_function_has_parity ↔ exists_function_without_parity := 
by
  unfold every_function_has_parity exists_function_without_parity
  sorry

end negation_of_every_function_has_parity_l224_224915


namespace additional_male_students_l224_224923

variable (a : ℕ)

theorem additional_male_students (h : a > 20) : a - 20 = (a - 20) := 
by 
  sorry

end additional_male_students_l224_224923


namespace height_difference_is_9_l224_224283

-- Definitions of the height of Petronas Towers and Empire State Building.
def height_Petronas : ℕ := 452
def height_EmpireState : ℕ := 443

-- Definition stating the height difference.
def height_difference := height_Petronas - height_EmpireState

-- Proving the height difference is 9 meters.
theorem height_difference_is_9 : height_difference = 9 :=
by
  -- the proof goes here
  sorry

end height_difference_is_9_l224_224283


namespace checkered_rectangles_containing_one_gray_cell_l224_224375

def total_number_of_rectangles_with_one_gray_cell :=
  let gray_cells := 40
  let blue_cells := 36
  let red_cells := 4
  
  let blue_rectangles_each := 4
  let red_rectangles_each := 8
  
  (blue_cells * blue_rectangles_each) + (red_cells * red_rectangles_each)

theorem checkered_rectangles_containing_one_gray_cell : total_number_of_rectangles_with_one_gray_cell = 176 :=
by 
  sorry

end checkered_rectangles_containing_one_gray_cell_l224_224375


namespace initial_hours_per_day_l224_224029

/-- 
Given:
1. 18 men working a certain number of hours per day dig 30 meters deep.
2. To dig to a depth of 50 meters, working 6 hours per day, 22 extra men should be put to work (total of 40 men).

Prove:
The initial 18 men were working \(\frac{200}{9}\) hours per day.
-/
theorem initial_hours_per_day 
  (h : ℚ)
  (work_done_18_men : 18 * h * 30 = 40 * 6 * 50) :
  h = 200 / 9 :=
by
  sorry

end initial_hours_per_day_l224_224029


namespace percentage_of_democrats_l224_224845

variable (D R : ℝ)

theorem percentage_of_democrats (h1 : D + R = 100) (h2 : 0.75 * D + 0.20 * R = 53) :
  D = 60 :=
by
  sorry

end percentage_of_democrats_l224_224845


namespace sunflower_cans_l224_224995

theorem sunflower_cans (total_seeds seeds_per_can : ℕ) (h_total_seeds : total_seeds = 54) (h_seeds_per_can : seeds_per_can = 6) :
  total_seeds / seeds_per_can = 9 :=
by sorry

end sunflower_cans_l224_224995


namespace election_votes_total_l224_224989

-- Definitions representing the conditions
def CandidateAVotes (V : ℕ) := 45 * V / 100
def CandidateBVotes (V : ℕ) := 35 * V / 100
def CandidateCVotes (V : ℕ) := 20 * V / 100

-- Main theorem statement
theorem election_votes_total (V : ℕ) (h1: CandidateAVotes V = 45 * V / 100) (h2: CandidateBVotes V = 35 * V / 100) (h3: CandidateCVotes V = 20 * V / 100)
  (h4: CandidateAVotes V - CandidateBVotes V = 1800) : V = 18000 :=
  sorry

end election_votes_total_l224_224989


namespace regular_polygon_sides_l224_224662

theorem regular_polygon_sides (angle : ℝ) (h1 : angle = 150) : ∃ n : ℕ, (180 * (n - 2)) / n = angle ∧ n = 12 :=
by
  sorry

end regular_polygon_sides_l224_224662


namespace smallest_value_of_c_l224_224410

theorem smallest_value_of_c :
  ∃ c : ℚ, (3 * c + 4) * (c - 2) = 9 * c ∧ (∀ d : ℚ, (3 * d + 4) * (d - 2) = 9 * d → c ≤ d) ∧ c = -8 / 3 := 
sorry

end smallest_value_of_c_l224_224410


namespace am_hm_inequality_l224_224722

theorem am_hm_inequality (a1 a2 a3 : ℝ) (h1 : 0 < a1) (h2 : 0 < a2) (h3 : 0 < a3) (h_sum : a1 + a2 + a3 = 1) : 
  (1 / a1) + (1 / a2) + (1 / a3) ≥ 9 :=
by
  sorry

end am_hm_inequality_l224_224722


namespace marching_band_total_weight_l224_224991

def weight_trumpet : ℕ := 5
def weight_clarinet : ℕ := 5
def weight_trombone : ℕ := 10
def weight_tuba : ℕ := 20
def weight_drummer : ℕ := 15
def weight_percussionist : ℕ := 8

def uniform_trumpet : ℕ := 3
def uniform_clarinet : ℕ := 3
def uniform_trombone : ℕ := 4
def uniform_tuba : ℕ := 5
def uniform_drummer : ℕ := 6
def uniform_percussionist : ℕ := 3

def count_trumpet : ℕ := 6
def count_clarinet : ℕ := 9
def count_trombone : ℕ := 8
def count_tuba : ℕ := 3
def count_drummer : ℕ := 2
def count_percussionist : ℕ := 4

def total_weight_band : ℕ :=
  (count_trumpet * (weight_trumpet + uniform_trumpet)) +
  (count_clarinet * (weight_clarinet + uniform_clarinet)) +
  (count_trombone * (weight_trombone + uniform_trombone)) +
  (count_tuba * (weight_tuba + uniform_tuba)) +
  (count_drummer * (weight_drummer + uniform_drummer)) +
  (count_percussionist * (weight_percussionist + uniform_percussionist))

theorem marching_band_total_weight : total_weight_band = 393 :=
  by
  sorry

end marching_band_total_weight_l224_224991


namespace gcd_102_238_l224_224135

theorem gcd_102_238 : Nat.gcd 102 238 = 34 :=
by
  sorry

end gcd_102_238_l224_224135


namespace sum_of_special_right_triangle_areas_l224_224181

noncomputable def is_special_right_triangle (a b : ℕ) : Prop :=
  let area := (a * b) / 2
  area = 3 * (a + b)

noncomputable def special_right_triangle_areas : List ℕ :=
  [(18, 9), (9, 18), (15, 10), (10, 15), (12, 12)].map (λ p => (p.1 * p.2) / 2)

theorem sum_of_special_right_triangle_areas : 
  special_right_triangle_areas.eraseDups.sum = 228 := by
  sorry

end sum_of_special_right_triangle_areas_l224_224181


namespace common_integer_root_l224_224813

theorem common_integer_root (a x : ℤ) : (a * x + a = 7) ∧ (3 * x - a = 17) → a = 1 :=
by
    sorry

end common_integer_root_l224_224813


namespace cats_remaining_l224_224419

theorem cats_remaining 
  (n_initial n_given_away : ℝ) 
  (h_initial : n_initial = 17.0) 
  (h_given_away : n_given_away = 14.0) : 
  (n_initial - n_given_away) = 3.0 :=
by
  rw [h_initial, h_given_away]
  norm_num

end cats_remaining_l224_224419


namespace gear_squeak_interval_l224_224784

theorem gear_squeak_interval 
  (N : ℕ) (M : ℕ) (T : ℕ) (hN : N = 12) (hM : M = 32) (hT : T = 3) :
  let lcm_val := Nat.lcm N M in
  (lcm_val / M) * T = 9 :=
by
  sorry

end gear_squeak_interval_l224_224784


namespace set_B_equals_1_4_l224_224084

open Set

def U : Set ℕ := {1, 2, 3, 4}
def C_U_B : Set ℕ := {2, 3}

theorem set_B_equals_1_4 : 
  ∃ B : Set ℕ, B = {1, 4} ∧ U \ B = C_U_B := by
  sorry

end set_B_equals_1_4_l224_224084


namespace bug_total_distance_l224_224777

/-- 
A bug starts at position 3 on a number line. It crawls to -4, then to 7, and finally to 1.
The total distance the bug crawls is 24 units.
-/
theorem bug_total_distance : 
  let start := 3
  let first_stop := -4
  let second_stop := 7
  let final_position := 1
  let distance := abs (first_stop - start) + abs (second_stop - first_stop) + abs (final_position - second_stop)
  distance = 24 := 
by
  sorry

end bug_total_distance_l224_224777


namespace problem_statement_l224_224555

noncomputable def α : ℝ := 3 + Real.sqrt 8
noncomputable def β : ℝ := 3 - Real.sqrt 8
noncomputable def x : ℝ := α ^ 500
noncomputable def n : ℤ := Int.floor x
noncomputable def f : ℝ := x - n

theorem problem_statement : x * (1 - f) = 1 := 
by
  sorry

end problem_statement_l224_224555


namespace artist_painting_time_l224_224187

theorem artist_painting_time (hours_per_week : ℕ) (weeks : ℕ) (total_paintings : ℕ) :
  hours_per_week = 30 → weeks = 4 → total_paintings = 40 →
  ((hours_per_week * weeks) / total_paintings) = 3 := by
  intros h_hours h_weeks h_paintings
  sorry

end artist_painting_time_l224_224187


namespace triangle_perimeter_l224_224887

variable (r A p : ℝ)

-- Define the conditions from the problem
def inradius (r : ℝ) := r = 3
def area (A : ℝ) := A = 30
def perimeter (A r p : ℝ) := A = r * (p / 2)

-- The theorem stating the problem
theorem triangle_perimeter (h1 : inradius r) (h2 : area A) (h3 : perimeter A r p) : p = 20 := 
by
  -- Proof is provided by the user, so we skip it with sorry
  sorry

end triangle_perimeter_l224_224887


namespace probability_prime_and_multiple_of_11_l224_224867

noncomputable def is_prime (n : ℕ) : Prop := nat.prime n
def is_multiple_of_11 (n : ℕ) : Prop := n % 11 = 0

theorem probability_prime_and_multiple_of_11 :
  (1 / 100 : ℝ) = 
  let qualifying_numbers := {n | n ∈ finset.range 101 ∧ is_prime n ∧ is_multiple_of_11 n} in
  let number_of_qualifying := finset.card qualifying_numbers in
  (number_of_qualifying / 100 : ℝ) :=
by
  sorry

end probability_prime_and_multiple_of_11_l224_224867


namespace coloring_equilateral_triangle_l224_224308

theorem coloring_equilateral_triangle :
  ∀ (A B C : Type) (color : A → Type) (d : A → A → ℝ),
  (∀ x y, d x y = 1 → color x = color y) :=
by sorry

end coloring_equilateral_triangle_l224_224308


namespace total_scissors_l224_224007

def initial_scissors : ℕ := 54
def added_scissors : ℕ := 22

theorem total_scissors : initial_scissors + added_scissors = 76 :=
by
  sorry

end total_scissors_l224_224007


namespace regular_polygon_sides_l224_224663

theorem regular_polygon_sides (angle : ℝ) (h1 : angle = 150) : ∃ n : ℕ, (180 * (n - 2)) / n = angle ∧ n = 12 :=
by
  sorry

end regular_polygon_sides_l224_224663


namespace prove_fraction_identity_l224_224377

-- Define the conditions and the entities involved
variables {x y : ℝ} (h₁ : x ≠ 0) (h₂ : y ≠ 0) (h₃ : 3 * x + y / 3 ≠ 0)

-- Formulate the theorem statement
theorem prove_fraction_identity :
  (3 * x + y / 3)⁻¹ * ((3 * x)⁻¹ + (y / 3)⁻¹) = (x * y)⁻¹ :=
sorry

end prove_fraction_identity_l224_224377


namespace bob_more_than_ken_l224_224404

def ken_situps : ℕ := 20

def nathan_situps : ℕ := 2 * ken_situps

def bob_situps : ℕ := (ken_situps + nathan_situps) / 2

theorem bob_more_than_ken : bob_situps - ken_situps = 10 := 
sorry

end bob_more_than_ken_l224_224404


namespace parabola_vertex_l224_224881

theorem parabola_vertex :
  ∃ (x y : ℤ), ((∀ x : ℝ, 2 * x^2 - 4 * x - 7 = y) ∧ x = 1 ∧ y = -9) := 
sorry

end parabola_vertex_l224_224881


namespace sum_of_integer_solutions_l224_224063

theorem sum_of_integer_solutions :
  ∀ x : ℤ, (x^4 - 13 * x^2 + 36 = 0) → (∃ a b c d, x = a + b + c + d ∧ a + b + c + d = 0) :=
begin
  sorry
end

end sum_of_integer_solutions_l224_224063


namespace regular_polygon_sides_l224_224668

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, 1 ≤ i → i ≤ n → (interior_angle i = 150) ) 
  (sum_interior_angles : ∑ i in range n, interior_angle i = 180 * (n - 2))
  {interior_angle : ℕ → ℕ} :
  n = 12 :=
by
  sorry

end regular_polygon_sides_l224_224668


namespace age_of_B_l224_224768

-- Define the ages of A and B
variables (A B : ℕ)

-- The conditions given in the problem
def condition1 (a b : ℕ) : Prop := a + 10 = 2 * (b - 10)
def condition2 (a b : ℕ) : Prop := a = b + 9

theorem age_of_B (A B : ℕ) (h1 : condition1 A B) (h2 : condition2 A B) : B = 39 :=
by
  sorry

end age_of_B_l224_224768


namespace rectangle_width_l224_224740

theorem rectangle_width (side_length_square : ℕ) (length_rectangle : ℕ) (area_equal : side_length_square * side_length_square = length_rectangle * w) : w = 4 := by
  sorry

end rectangle_width_l224_224740


namespace smallest_percentage_boys_correct_l224_224998

noncomputable def smallest_percentage_boys (B : ℝ) : ℝ :=
  if h : 0 ≤ B ∧ B ≤ 1 then B else 0

theorem smallest_percentage_boys_correct :
  ∃ B : ℝ,
    0 ≤ B ∧ B ≤ 1 ∧
    (67.5 / 100 * B * 200 + 25 / 100 * (1 - B) * 200) ≥ 101 ∧
    B = 0.6 :=
by
  sorry

end smallest_percentage_boys_correct_l224_224998


namespace pet_purchase_ways_l224_224038

-- Define the conditions
def number_of_puppies : Nat := 20
def number_of_kittens : Nat := 6
def number_of_hamsters : Nat := 8

def alice_choices : Nat := number_of_puppies

-- Define the problem statement in Lean
theorem pet_purchase_ways : 
  (number_of_puppies = 20) ∧ 
  (number_of_kittens = 6) ∧ 
  (number_of_hamsters = 8) → 
  (alice_choices * 2 * number_of_kittens * number_of_hamsters) = 1920 := 
by
  intros h
  sorry

end pet_purchase_ways_l224_224038


namespace green_peaches_count_l224_224752

def red_peaches : ℕ := 17
def green_peaches (x : ℕ) : Prop := red_peaches = x + 1

theorem green_peaches_count (x : ℕ) (h : green_peaches x) : x = 16 :=
by
  sorry

end green_peaches_count_l224_224752


namespace pairs_characterization_l224_224858

noncomputable def valid_pairs (A : ℝ) : Set (ℕ × ℕ) :=
  { p | ∃ x : ℝ, x > 0 ∧ (1 + x) ^ p.1 = (1 + A * x) ^ p.2 }

theorem pairs_characterization (A : ℝ) (hA : A > 1) :
  valid_pairs A = { p | p.2 < p.1 ∧ p.1 < A * p.2 } :=
by
  sorry

end pairs_characterization_l224_224858


namespace acute_angle_at_9_35_is_77_5_degrees_l224_224598

def degrees_in_acute_angle_formed_by_hands_of_clock_9_35 : ℝ := 77.5

theorem acute_angle_at_9_35_is_77_5_degrees 
  (hour_angle : ℝ := 270 + (35/60 * 30))
  (minute_angle : ℝ := 35/60 * 360) : 
  |hour_angle - minute_angle| < 180 → |hour_angle - minute_angle| = degrees_in_acute_angle_formed_by_hands_of_clock_9_35 := 
by 
  sorry

end acute_angle_at_9_35_is_77_5_degrees_l224_224598


namespace spending_difference_l224_224645

def chocolate_price : ℝ := 7
def candy_bar_price : ℝ := 2
def discount_rate : ℝ := 0.15
def sales_tax_rate : ℝ := 0.08
def gum_price : ℝ := 3

def discounted_chocolate_price : ℝ := chocolate_price * (1 - discount_rate)
def total_before_tax : ℝ := candy_bar_price + gum_price
def tax_amount : ℝ := total_before_tax * sales_tax_rate
def total_after_tax : ℝ := total_before_tax + tax_amount

theorem spending_difference : 
  discounted_chocolate_price - candy_bar_price = 3.95 :=
by 
  -- Apply the necessary calculations
  have discount_chocolate : ℝ := discounted_chocolate_price
  have candy_bar : ℝ := candy_bar_price
  calc
    discounted_chocolate_price - candy_bar_price = _ := sorry

end spending_difference_l224_224645


namespace cos_60_degrees_is_one_half_l224_224634

-- Define the given conditions in Lean 4
def angle_60_degrees : Real :=
  60 * Real.pi / 180

def unit_circle_point (θ : Real) : Real × Real :=
  (Real.cos θ, Real.sin θ)

-- State the proof problem in Lean 4
theorem cos_60_degrees_is_one_half : 
  (unit_circle_point angle_60_degrees).1 = 1 / 2 := 
sorry

end cos_60_degrees_is_one_half_l224_224634


namespace unique_n_divisors_satisfies_condition_l224_224724

theorem unique_n_divisors_satisfies_condition:
  ∃ (n : ℕ), (∃ d1 d2 d3 : ℕ, d1 = 1 ∧ d2 > d1 ∧ d3 > d2 ∧ n = d3 ∧
  n = d2^2 + d3^3) ∧ n = 68 := by
  sorry

end unique_n_divisors_satisfies_condition_l224_224724


namespace ratio_chloe_to_max_l224_224189

/-- Chloe’s wins and Max’s wins -/
def chloe_wins : ℕ := 24
def max_wins : ℕ := 9

/-- The ratio of Chloe's wins to Max's wins is 8:3 -/
theorem ratio_chloe_to_max : (chloe_wins / Nat.gcd chloe_wins max_wins) = 8 ∧ (max_wins / Nat.gcd chloe_wins max_wins) = 3 := by
  sorry

end ratio_chloe_to_max_l224_224189


namespace sum_tripled_numbers_l224_224444

theorem sum_tripled_numbers (x y S : ℝ) (h : x + y = S) : 
  3 * (x + 5) + 3 * (y + 5) = 3 * S + 30 :=
by
  sorry

end sum_tripled_numbers_l224_224444


namespace stratified_sampling_young_employees_l224_224648

-- Given conditions
def total_young : Nat := 350
def total_middle_aged : Nat := 500
def total_elderly : Nat := 150
def total_employees : Nat := total_young + total_middle_aged + total_elderly
def representatives_to_select : Nat := 20
def sampling_ratio : Rat := representatives_to_select / (total_employees : Rat)

-- Proof goal
theorem stratified_sampling_young_employees :
  (total_young : Rat) * sampling_ratio = 7 := 
by
  sorry

end stratified_sampling_young_employees_l224_224648
