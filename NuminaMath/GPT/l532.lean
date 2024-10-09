import Mathlib

namespace black_balls_in_box_l532_53285

theorem black_balls_in_box (B : ℕ) (probability : ℚ) 
  (h1 : probability = 0.38095238095238093) 
  (h2 : B / (14 + B) = probability) : 
  B = 9 := by
  sorry

end black_balls_in_box_l532_53285


namespace shop_owner_pricing_l532_53245

theorem shop_owner_pricing (L C M S : ℝ)
  (h1 : C = 0.75 * L)
  (h2 : S = 1.3 * C)
  (h3 : S = 0.75 * M) : 
  M = 1.3 * L := 
sorry

end shop_owner_pricing_l532_53245


namespace x_squared_plus_y_squared_l532_53232

theorem x_squared_plus_y_squared (x y : ℝ) (h1 : x * y = 3) (h2 : (x - y) ^ 2 = 9) : 
  x ^ 2 + y ^ 2 = 15 := sorry

end x_squared_plus_y_squared_l532_53232


namespace find_five_digit_number_l532_53225

theorem find_five_digit_number : 
  ∃ (A B C D E : ℕ), 
    (0 < A ∧ A ≤ 9) ∧ 
    (0 < B ∧ B ≤ 9) ∧ 
    (0 < C ∧ C ≤ 9) ∧ 
    (0 < D ∧ D ≤ 9) ∧ 
    (0 < E ∧ E ≤ 9) ∧ 
    (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E) ∧ 
    (B ≠ C ∧ B ≠ D ∧ B ≠ E) ∧ 
    (C ≠ D ∧ C ≠ E) ∧ 
    (D ≠ E) ∧ 
    (2016 = (10 * D + E) * A * B) ∧ 
    (¬ (10 * D + E) % 3 = 0) ∧ 
    (10^4 * A + 10^3 * B + 10^2 * C + 10 * D + E = 85132) :=
sorry

end find_five_digit_number_l532_53225


namespace rate_grapes_l532_53289

/-- Given that Bruce purchased 8 kg of grapes at a rate G per kg, 8 kg of mangoes at the rate of 55 per kg, 
and paid a total of 1000 to the shopkeeper, prove that the rate per kg for the grapes (G) is 70. -/
theorem rate_grapes (G : ℝ) (h1 : 8 * G + 8 * 55 = 1000) : G = 70 :=
by 
  sorry

end rate_grapes_l532_53289


namespace sequence_arithmetic_l532_53230

theorem sequence_arithmetic (a : ℕ → Real)
    (h₁ : a 3 = 2)
    (h₂ : a 7 = 1)
    (h₃ : ∃ d, ∀ n, 1 / (1 + a (n + 1)) = 1 / (1 + a n) + d):
    a 11 = 1 / 2 := by
  sorry

end sequence_arithmetic_l532_53230


namespace Stan_pays_magician_l532_53271

theorem Stan_pays_magician :
  let hours_per_day := 3
  let days_per_week := 7
  let weeks := 2
  let hourly_rate := 60
  let total_hours := hours_per_day * days_per_week * weeks
  let total_payment := hourly_rate * total_hours
  total_payment = 2520 := 
by 
  sorry

end Stan_pays_magician_l532_53271


namespace prime_of_form_a2_minus_1_l532_53247

theorem prime_of_form_a2_minus_1 (a : ℕ) (p : ℕ) (ha : a ≥ 2) (hp : p = a^2 - 1) (prime_p : Nat.Prime p) : p = 3 := 
by 
  sorry

end prime_of_form_a2_minus_1_l532_53247


namespace first_digit_base_9_of_y_l532_53260

def base_3_to_base_10 (n : Nat) : Nat := sorry
def base_10_to_base_9_first_digit (n : Nat) : Nat := sorry

theorem first_digit_base_9_of_y :
  let y := 11220022110022112221
  let base_10_y := base_3_to_base_10 y
  base_10_to_base_9_first_digit base_10_y = 4 :=
by
  let y := 11220022110022112221
  let base_10_y := base_3_to_base_10 y
  show base_10_to_base_9_first_digit base_10_y = 4
  sorry

end first_digit_base_9_of_y_l532_53260


namespace person_left_time_l532_53211

theorem person_left_time :
  ∃ (x y : ℚ), 
    0 ≤ x ∧ x < 1 ∧ 
    0 ≤ y ∧ y < 1 ∧ 
    (120 + 30 * x = 360 * y) ∧
    (360 * x = 150 + 30 * y) ∧
    (4 + x = 4 + 64 / 143) := 
by
  sorry

end person_left_time_l532_53211


namespace inequality_sum_leq_three_l532_53254

theorem inequality_sum_leq_three
  (x y z : ℝ)
  (hx : 0 < x)
  (hy : 0 < y)
  (hz : 0 < z)
  (hxyz : x^2 + y^2 + z^2 ≥ 3) :
  (x^2 + y^2 + z^2) / (x^5 + y^2 + z^2) + 
  (x^2 + y^2 + z^2) / (y^5 + x^2 + z^2) + 
  (x^2 + y^2 + z^2) / (z^5 + x^2 + y^2 + z^2) ≤ 3 := 
sorry

end inequality_sum_leq_three_l532_53254


namespace part1_part2_l532_53266

noncomputable def f (x a : ℝ) : ℝ := abs (2 * x - 5 * a) + abs (2 * x + 1)
noncomputable def g (x : ℝ) : ℝ := abs (x - 1) + 3

-- (1)
theorem part1 (x : ℝ) : abs (g x) < 8 → -4 < x ∧ x < 6 :=
by
  sorry

-- (2)
theorem part2 (a : ℝ) : (∀ x1 : ℝ, ∃ x2 : ℝ, f x1 a = g x2) → (a ≥ 0.4 ∨ a ≤ -0.8) :=
by
  sorry

end part1_part2_l532_53266


namespace heather_final_blocks_l532_53276

def heather_initial_blocks : ℝ := 86.0
def jose_shared_blocks : ℝ := 41.0

theorem heather_final_blocks : heather_initial_blocks + jose_shared_blocks = 127.0 :=
by
  sorry

end heather_final_blocks_l532_53276


namespace at_least_one_gt_one_l532_53205

theorem at_least_one_gt_one (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y > 2) : x > 1 ∨ y > 1 :=
sorry

end at_least_one_gt_one_l532_53205


namespace evaluate_expression_l532_53275

theorem evaluate_expression (x y : ℕ) (h1 : x = 4) (h2 : y = 3) :
  5 * x + 2 * y * 3 = 38 :=
by
  sorry

end evaluate_expression_l532_53275


namespace average_student_headcount_is_10983_l532_53238

def student_headcount_fall_03_04 := 11500
def student_headcount_spring_03_04 := 10500
def student_headcount_fall_04_05 := 11600
def student_headcount_spring_04_05 := 10700
def student_headcount_fall_05_06 := 11300
def student_headcount_spring_05_06 := 10300 -- Assume value

def total_student_headcount :=
  student_headcount_fall_03_04 + student_headcount_spring_03_04 +
  student_headcount_fall_04_05 + student_headcount_spring_04_05 +
  student_headcount_fall_05_06 + student_headcount_spring_05_06

def average_student_headcount := total_student_headcount / 6

theorem average_student_headcount_is_10983 :
  average_student_headcount = 10983 :=
by -- Will prove the theorem
sorry

end average_student_headcount_is_10983_l532_53238


namespace slope_of_line_l532_53228

theorem slope_of_line (x₁ y₁ x₂ y₂ : ℝ) (h₁ : 2/x₁ + 3/y₁ = 0) (h₂ : 2/x₂ + 3/y₂ = 0) (h_diff : x₁ ≠ x₂) : 
  (y₂ - y₁) / (x₂ - x₁) = -3/2 :=
sorry

end slope_of_line_l532_53228


namespace cone_height_l532_53207

theorem cone_height (r h : ℝ) (π : ℝ) (Hπ : Real.pi = π) (slant_height : ℝ) (lateral_area : ℝ) (base_area : ℝ) 
  (H1 : slant_height = 2) 
  (H2 : lateral_area = 2 * π * r) 
  (H3 : base_area = π * r^2) 
  (H4 : lateral_area = 4 * base_area) 
  (H5 : r^2 + h^2 = slant_height^2) 
  : h = π / 2 := by 
sorry

end cone_height_l532_53207


namespace value_of_r_minus_p_l532_53293

variable (p q r : ℝ)

-- The conditions given as hypotheses
def arithmetic_mean_pq := (p + q) / 2 = 10
def arithmetic_mean_qr := (q + r) / 2 = 25

-- The goal is to prove that r - p = 30
theorem value_of_r_minus_p (h1: arithmetic_mean_pq p q) (h2: arithmetic_mean_qr q r) :
  r - p = 30 := by
  sorry

end value_of_r_minus_p_l532_53293


namespace solve_system1_solve_system2_l532_53253

-- Define System (1) and prove its solution
theorem solve_system1 (x y : ℝ) (h1 : x = 5 - y) (h2 : x - 3 * y = 1) : x = 4 ∧ y = 1 := by
  sorry

-- Define System (2) and prove its solution
theorem solve_system2 (x y : ℝ) (h1 : x - 2 * y = 6) (h2 : 2 * x + 3 * y = -2) : x = 2 ∧ y = -2 := by
  sorry

end solve_system1_solve_system2_l532_53253


namespace find_erased_number_l532_53263

theorem find_erased_number (x : ℕ) (h : 8 * x = 96) : x = 12 := by
  sorry

end find_erased_number_l532_53263


namespace cube_volume_l532_53274

theorem cube_volume (A V : ℝ) (h : A = 16) : V = 64 :=
by
  -- Here, we would provide the proof, but for now, we end with sorry
  sorry

end cube_volume_l532_53274


namespace transistors_in_2005_l532_53219

theorem transistors_in_2005
  (initial_count : ℕ)
  (doubles_every : ℕ)
  (triples_every : ℕ)
  (years : ℕ) :
  initial_count = 500000 ∧ doubles_every = 2 ∧ triples_every = 6 ∧ years = 15 →
  (initial_count * 2^(years/doubles_every) + initial_count * 3^(years/triples_every)) = 68500000 :=
by
  sorry

end transistors_in_2005_l532_53219


namespace people_who_speak_French_l532_53286

theorem people_who_speak_French (T L N B : ℕ) (hT : T = 25) (hL : L = 13) (hN : N = 6) (hB : B = 9) : 
  ∃ F : ℕ, F = 15 := 
by 
  sorry

end people_who_speak_French_l532_53286


namespace log_sum_geometric_sequence_l532_53227

open Real

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ (n : ℕ), a n ≠ 0 ∧ a (n + 1) / a n = a 1 / a 0

theorem log_sum_geometric_sequence 
  (a : ℕ → ℝ) 
  (h_pos : ∀ n, 0 < a n) 
  (h_geo : geometric_sequence a) 
  (h_eq : a 10 * a 11 + a 9 * a 12 = 2 * exp 5) : 
  log (a 1) + log (a 2) + log (a 3) + log (a 4) + log (a 5) + 
  log (a 6) + log (a 7) + log (a 8) + log (a 9) + log (a 10) + 
  log (a 11) + log (a 12) + log (a 13) + log (a 14) + log (a 15) + 
  log (a 16) + log (a 17) + log (a 18) + log (a 19) + log (a 20) = 50 :=
sorry

end log_sum_geometric_sequence_l532_53227


namespace candy_problem_l532_53265

-- Definitions of conditions
def total_candies : ℕ := 91
def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

-- The minimum number of kinds of candies function
def min_kinds : ℕ := 46

-- Lean statement for the proof problem
theorem candy_problem : 
  ∀ (kinds : ℕ), 
    (∀ c : ℕ, c < total_candies → c % kinds < 2) → (∃ n : ℕ, kinds = min_kinds) := 
sorry

end candy_problem_l532_53265


namespace unique_solution_condition_l532_53246

variable (c d x : ℝ)

-- Define the equation
def equation : Prop := 4 * x - 7 + c = d * x + 3

-- Lean theorem for the proof problem
theorem unique_solution_condition :
  (∃! x, equation c d x) ↔ d ≠ 4 :=
sorry

end unique_solution_condition_l532_53246


namespace glass_pieces_same_color_l532_53217

theorem glass_pieces_same_color (r y b : ℕ) (h : r + y + b = 2002) :
  (∃ k : ℕ, ∀ n, n ≥ k → (r + y + b) = n ∧ (r = 0 ∨ y = 0 ∨ b = 0)) ∧
  (∀ (r1 y1 b1 r2 y2 b2 : ℕ),
    r1 + y1 + b1 = 2002 →
    r2 + y2 + b2 = 2002 →
    (∃ k : ℕ, ∀ n, n ≥ k → (r1 = 0 ∨ y1 = 0 ∨ b1 = 0)) →
    (∃ l : ℕ, ∀ m, m ≥ l → (r2 = 0 ∨ y2 = 0 ∨ b2 = 0)) →
    r1 = r2 ∧ y1 = y2 ∧ b1 = b2):=
by
  sorry

end glass_pieces_same_color_l532_53217


namespace change_received_l532_53249

variable (a : ℝ)

theorem change_received (h : a ≤ 30) : 100 - 3 * a = 100 - 3 * a :=
by
  sorry

end change_received_l532_53249


namespace primes_divisibility_l532_53208

theorem primes_divisibility
  (p1 p2 p3 p4 q1 q2 q3 q4 : ℕ)
  (hp1_lt_p2 : p1 < p2) (hp2_lt_p3 : p2 < p3) (hp3_lt_p4 : p3 < p4)
  (hq1_lt_q2 : q1 < q2) (hq2_lt_q3 : q2 < q3) (hq3_lt_q4 : q3 < q4)
  (hp4_minus_p1 : p4 - p1 = 8) (hq4_minus_q1 : q4 - q1 = 8)
  (hp1_gt_5 : 5 < p1) (hq1_gt_5 : 5 < q1) :
  30 ∣ (p1 - q1) :=
sorry

end primes_divisibility_l532_53208


namespace trapezoid_area_l532_53209

theorem trapezoid_area (outer_triangle_area inner_triangle_area : ℝ) (congruent_trapezoids : ℕ) 
  (h1 : outer_triangle_area = 36) (h2 : inner_triangle_area = 4) (h3 : congruent_trapezoids = 3) :
  (outer_triangle_area - inner_triangle_area) / congruent_trapezoids = 32 / 3 :=
by sorry

end trapezoid_area_l532_53209


namespace polynomial_remainder_l532_53295

theorem polynomial_remainder (x : ℝ) :
  ∃ (Q : ℝ → ℝ) (a b : ℝ),
    (x^150 = (x^2 - 5*x + 6) * Q x + (a*x + b)) ∧
    (2 * a + b = 2^150) ∧
    (3 * a + b = 3^150) ∧ 
    (a = 3^150 - 2^150) ∧ 
    (b = 2^150 - 2 * 3^150 + 2 * 2^150) := sorry

end polynomial_remainder_l532_53295


namespace smallest_n_value_l532_53206

-- Define the conditions as given in the problem
def num_birthdays := 365

-- Formulating the main statement
theorem smallest_n_value : ∃ (n : ℕ), (∀ (group_size : ℕ), group_size = 2 * n - 10 → group_size ≥ 3286) ∧ n = 1648 :=
by
  use 1648
  sorry

end smallest_n_value_l532_53206


namespace income_growth_l532_53270

theorem income_growth (x : ℝ) : 12000 * (1 + x)^2 = 14520 :=
sorry

end income_growth_l532_53270


namespace pool_capacity_l532_53236

variable (C : ℕ)

-- Conditions
def rate_first_valve := C / 120
def rate_second_valve := C / 120 + 50
def combined_rate := C / 48

-- Proof statement
theorem pool_capacity (C_pos : 0 < C) (h1 : rate_first_valve C + rate_second_valve C = combined_rate C) : C = 12000 := by
  sorry

end pool_capacity_l532_53236


namespace possible_values_of_reciprocal_sum_l532_53273

theorem possible_values_of_reciprocal_sum (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + b = 2) :
  ∃ y, y = (1/a + 1/b) ∧ (2 ≤ y ∧ ∀ t, t < y ↔ ¬t < 2) :=
by sorry

end possible_values_of_reciprocal_sum_l532_53273


namespace initial_birds_count_l532_53298

theorem initial_birds_count (current_total_birds birds_joined initial_birds : ℕ) 
  (h1 : current_total_birds = 6) 
  (h2 : birds_joined = 4) : 
  initial_birds = current_total_birds - birds_joined → 
  initial_birds = 2 :=
by 
  intro h3
  rw [h1, h2] at h3
  exact h3

end initial_birds_count_l532_53298


namespace intersection_eq_l532_53294

def A := {x : ℝ | |x| = x}
def B := {x : ℝ | x^2 + x ≥ 0}

theorem intersection_eq : A ∩ B = {x : ℝ | 0 ≤ x} := by
  sorry

end intersection_eq_l532_53294


namespace living_room_floor_area_l532_53290

-- Define the problem conditions
def carpet_length : ℝ := 4
def carpet_width : ℝ := 9
def carpet_area : ℝ := carpet_length * carpet_width    -- Area of the carpet

def percentage_covered_by_carpet : ℝ := 0.75

-- Theorem to prove: the area of the living room floor is 48 square feet
theorem living_room_floor_area (carpet_area : ℝ) (percentage_covered_by_carpet : ℝ) : 
  (A_floor : ℝ) = carpet_area / percentage_covered_by_carpet :=
by
  let carpet_area := 36
  let percentage_covered_by_carpet := 0.75
  let A_floor := 48
  sorry

end living_room_floor_area_l532_53290


namespace repeating_decimal_value_l532_53257

def repeating_decimal : ℝ := 0.0000253253325333 -- Using repeating decimal as given in the conditions

theorem repeating_decimal_value :
  (10^7 - 10^5) * repeating_decimal = 253 / 990 :=
sorry

end repeating_decimal_value_l532_53257


namespace initial_team_sizes_l532_53241

/-- 
On the first day of the sports competition, 1/6 of the boys' team and 1/7 of the girls' team 
did not meet the qualifying standards and were eliminated. During the rest of the competition, 
the same number of athletes from both teams were eliminated for not meeting the standards. 
By the end of the competition, a total of 48 boys and 50 girls did not meet the qualifying standards. 
Moreover, the number of girls who met the qualifying standards was twice the number of boys who did.
We are to prove the initial number of boys and girls in their respective teams.
-/

theorem initial_team_sizes (initial_boys initial_girls : ℕ) :
  (∃ (x : ℕ), 
    initial_boys = x + 48 ∧ 
    initial_girls = 2 * x + 50 ∧ 
    48 - (1 / 6 : ℚ) * (x + 48 : ℚ) = 50 - (1 / 7 : ℚ) * (2 * x + 50 : ℚ) ∧
    initial_girls - 2 * initial_boys = 98 - 2 * 72
  ) ↔ 
  initial_boys = 72 ∧ initial_girls = 98 := 
sorry

end initial_team_sizes_l532_53241


namespace determine_counterfeit_coin_l532_53201

-- Definitions and conditions
def coin_weight (coin : ℕ) : ℕ :=
  match coin with
  | 1 => 1 -- 1-kopek coin weighs 1 gram
  | 2 => 2 -- 2-kopeks coin weighs 2 grams
  | 3 => 3 -- 3-kopeks coin weighs 3 grams
  | 5 => 5 -- 5-kopeks coin weighs 5 grams
  | _ => 0 -- Invalid coin denomination, should not happen

def is_counterfeit (coin : ℕ) (actual_weight : ℕ) : Prop :=
  coin_weight coin ≠ actual_weight

-- Statement of the problem to be proved
theorem determine_counterfeit_coin (coins : List (ℕ × ℕ)) :
   (∀ (coin: ℕ) (weight: ℕ) (h : (coin, weight) ∈ coins),
      coin_weight coin = weight ∨ is_counterfeit coin weight) →
   (∃ (counterfeit_coin: ℕ) (weight: ℕ),
      (counterfeit_coin, weight) ∈ coins ∧ is_counterfeit counterfeit_coin weight) :=
sorry

end determine_counterfeit_coin_l532_53201


namespace betty_wallet_l532_53292

theorem betty_wallet :
  let wallet_cost := 125.75
  let initial_amount := wallet_cost / 2
  let parents_contribution := 45.25
  let grandparents_contribution := 2 * parents_contribution
  let brothers_contribution := 3/4 * grandparents_contribution
  let aunts_contribution := 1/2 * brothers_contribution
  let total_amount := initial_amount + parents_contribution + grandparents_contribution + brothers_contribution + aunts_contribution
  total_amount - wallet_cost = 174.6875 :=
by
  sorry

end betty_wallet_l532_53292


namespace digit_B_divisible_by_9_l532_53244

-- Defining the condition for B making 762B divisible by 9
theorem digit_B_divisible_by_9 (B : ℕ) : (15 + B) % 9 = 0 ↔ B = 3 := 
by
  sorry

end digit_B_divisible_by_9_l532_53244


namespace mn_sum_l532_53277

theorem mn_sum (M N : ℚ) (h1 : (4 : ℚ) / 7 = M / 63) (h2 : (4 : ℚ) / 7 = 84 / N) : M + N = 183 := sorry

end mn_sum_l532_53277


namespace most_likely_sum_exceeding_twelve_l532_53252

-- Define a die with faces 0, 1, 2, 3, 4, 5
def die_faces : List ℕ := [0, 1, 2, 3, 4, 5]

-- Define a function to get the sum of rolled results exceeding 12
noncomputable def sum_exceeds_twelve (rolls : List ℕ) : ℕ :=
  let sum := rolls.foldl (· + ·) 0
  if sum > 12 then sum else 0

-- Define a function to simulate the die roll until the sum exceeds 12
noncomputable def roll_die_until_exceeds_twelve : ℕ :=
  sorry -- This would contain the logic to simulate the rolling process

-- The theorem statement that the most likely value of the sum exceeding 12 is 13
theorem most_likely_sum_exceeding_twelve : roll_die_until_exceeds_twelve = 13 :=
  sorry

end most_likely_sum_exceeding_twelve_l532_53252


namespace largest_prime_divisor_of_36_squared_plus_49_squared_l532_53202

theorem largest_prime_divisor_of_36_squared_plus_49_squared :
  Nat.gcd (36^2 + 49^2) 3697 = 3697 :=
by
  -- Since 3697 is prime, and the calculation shows 36^2 + 49^2 is 3697
  sorry

end largest_prime_divisor_of_36_squared_plus_49_squared_l532_53202


namespace sum_opposite_numbers_correct_opposite_sum_numbers_correct_l532_53262

def opposite (x : Int) : Int := -x

def sum_opposite_numbers (a b : Int) : Int := opposite a + opposite b

def opposite_sum_numbers (a b : Int) : Int := opposite (a + b)

theorem sum_opposite_numbers_correct (a b : Int) : sum_opposite_numbers (-6) 4 = 2 := 
by sorry

theorem opposite_sum_numbers_correct (a b : Int) : opposite_sum_numbers (-6) 4 = 2 := 
by sorry

end sum_opposite_numbers_correct_opposite_sum_numbers_correct_l532_53262


namespace same_graph_iff_same_function_D_l532_53248

theorem same_graph_iff_same_function_D :
  ∀ x : ℝ, (|x| = if x ≥ 0 then x else -x) :=
by
  intro x
  sorry

end same_graph_iff_same_function_D_l532_53248


namespace simplify_expression_l532_53291

theorem simplify_expression (m : ℤ) : 
  ((7 * m + 3) - 3 * m * 2) * 4 + (5 - 2 / 4) * (8 * m - 12) = 40 * m - 42 :=
by 
  sorry

end simplify_expression_l532_53291


namespace fraction_of_upgraded_sensors_l532_53200

theorem fraction_of_upgraded_sensors (N U : ℕ) (h1 : N = U / 6) :
  (U / (24 * N + U)) = 1 / 5 :=
by
  sorry

end fraction_of_upgraded_sensors_l532_53200


namespace circle_equation_focus_parabola_origin_l532_53281

noncomputable def parabola_focus (p : ℝ) (x y : ℝ) : Prop :=
  y^2 = 4 * p * x

def passes_through_origin (x y : ℝ) : Prop :=
  (0 - x)^2 + (0 - y)^2 = x^2 + y^2

theorem circle_equation_focus_parabola_origin :
  (∃ x y : ℝ, parabola_focus 1 x y ∧ passes_through_origin x y)
    → ∃ k : ℝ, (x^2 - 2 * x + y^2 = k) :=
sorry

end circle_equation_focus_parabola_origin_l532_53281


namespace h_inv_f_neg3_does_not_exist_real_l532_53288

noncomputable def h : ℝ → ℝ := sorry
noncomputable def f : ℝ → ℝ := sorry

theorem h_inv_f_neg3_does_not_exist_real (h_inv : ℝ → ℝ)
  (h_cond : ∀ (x : ℝ), f (h_inv (h x)) = 7 * x ^ 2 + 4) :
  ¬ ∃ x : ℝ, h_inv (f (-3)) = x :=
by 
  sorry

end h_inv_f_neg3_does_not_exist_real_l532_53288


namespace solve_for_m_l532_53278

theorem solve_for_m (x y m : ℤ) (h1 : x - 2 * y = -3) (h2 : 2 * x + 3 * y = m - 1) (h3 : x = -y) : m = 2 :=
by
  sorry

end solve_for_m_l532_53278


namespace coins_division_remainder_l532_53214

theorem coins_division_remainder :
  ∃ n : ℕ, (n % 8 = 6 ∧ n % 7 = 5 ∧ n % 9 = 0) :=
sorry

end coins_division_remainder_l532_53214


namespace factorize_polynomial_1_factorize_polynomial_2_triangle_shape_l532_53231

-- Proof for (1)
theorem factorize_polynomial_1 (a : ℝ) : 2*a^2 - 8*a + 8 = 2*(a - 2)^2 :=
by
  sorry

-- Proof for (2)
theorem factorize_polynomial_2 (x y : ℝ) : x^2 - y^2 + 3*x - 3*y = (x - y)*(x + y + 3) :=
by
  sorry

-- Proof for (3)
theorem triangle_shape (a b c : ℝ) (h : a^2 - ab - ac + bc = 0) : 
  (a = b ∨ a = c) :=
by
  sorry

end factorize_polynomial_1_factorize_polynomial_2_triangle_shape_l532_53231


namespace common_area_approximation_l532_53261

noncomputable def elliptical_domain (x y : ℝ) : Prop :=
  (x^2 / 3 + y^2 / 2) ≤ 1

noncomputable def circular_domain (x y : ℝ) : Prop :=
  (x^2 + y^2) ≤ 2

noncomputable def intersection_area : ℝ :=
  7.27

theorem common_area_approximation :
  ∃ area, 
    elliptical_domain x y ∧ circular_domain x y →
    abs (area - intersection_area) < 0.01 :=
sorry

end common_area_approximation_l532_53261


namespace planar_graph_edge_vertex_inequality_l532_53224

def planar_graph (G : Type _) : Prop := -- Placeholder for planar graph property
  sorry

variables {V E : ℕ}

theorem planar_graph_edge_vertex_inequality (G : Type _) (h : planar_graph G) :
  E ≤ 3 * V - 6 :=
sorry

end planar_graph_edge_vertex_inequality_l532_53224


namespace canister_ratio_l532_53203

variable (C D : ℝ) -- Define capacities of canister C and canister D
variable (hC_half : 1/2 * C) -- Canister C is 1/2 full of water
variable (hD_third : 1/3 * D) -- Canister D is 1/3 full of water
variable (hD_after : 1/12 * D) -- Canister D contains 1/12 after pouring

theorem canister_ratio (h : 1/2 * C = 1/4 * D) : D / C = 2 :=
by
  sorry

end canister_ratio_l532_53203


namespace exists_common_ratio_of_geometric_progression_l532_53218

theorem exists_common_ratio_of_geometric_progression (a r : ℝ) (h_pos : 0 < r) 
(h_eq: a = a * r + a * r^2 + a * r^3) : ∃ r : ℝ, r^3 + r^2 + r - 1 = 0 :=
by sorry

end exists_common_ratio_of_geometric_progression_l532_53218


namespace tangent_line_equation_l532_53282

theorem tangent_line_equation 
    (h_perpendicular : ∃ m1 m2 : ℝ, m1 * m2 = -1 ∧ (∀ y, x + m1 * y = 4) ∧ (x + 4 * y = 4)) 
    (h_tangent : ∀ x : ℝ, y = 2 * x ^ 2 ∧ (∀ y', y' = 4 * x)) :
    ∃ a b c : ℝ, (4 * a - b - c = 0) ∧ (∀ (t : ℝ), a * t + b * (2 * t ^ 2) = 1) :=
sorry

end tangent_line_equation_l532_53282


namespace fraction_equality_l532_53284

variables (x y : ℝ)

theorem fraction_equality (h : y / 2 = (2 * y - x) / 3) : y / x = 2 :=
sorry

end fraction_equality_l532_53284


namespace compute_x2_y2_l532_53269

theorem compute_x2_y2 (x y : ℝ) (h1 : 1 < x) (h2 : 1 < y)
  (h3 : (Real.log x / Real.log 4)^3 + (Real.log y / Real.log 5)^3 + 27 = 9 * (Real.log x / Real.log 4) * (Real.log y / Real.log 5)) :
  x^2 + y^2 = 189 := 
by sorry

end compute_x2_y2_l532_53269


namespace masha_wins_l532_53251

def num_matches : Nat := 111

-- Define a function for Masha's optimal play strategy
-- In this problem, we'll denote both players' move range and the condition for winning.
theorem masha_wins (n : Nat := num_matches) (conditions : n > 0 ∧ n % 11 = 0 ∧ (∀ k : Nat, 1 ≤ k ∧ k ≤ 10 → ∃ new_n : Nat, n = k + new_n)) : True :=
  sorry

end masha_wins_l532_53251


namespace seq_20_l532_53299

noncomputable def seq (n : ℕ) : ℝ := 
  if n = 0 then 0
  else if n = 1 then 1
  else if n = 2 then 1/2
  else sorry -- The actual function definition based on the recurrence relation is omitted for brevity

lemma seq_recurrence (n : ℕ) (hn : n ≥ 1) :
  2 / seq (n + 1) = (seq n + seq (n + 2)) / (seq n * seq (n + 2)) :=
sorry

theorem seq_20 : seq 20 = 1/20 :=
sorry

end seq_20_l532_53299


namespace probability_of_type_I_error_l532_53240

theorem probability_of_type_I_error 
  (K_squared : ℝ)
  (alpha : ℝ)
  (critical_val : ℝ)
  (h1 : K_squared = 4.05)
  (h2 : alpha = 0.05)
  (h3 : critical_val = 3.841)
  (h4 : 4.05 > 3.841) :
  alpha = 0.05 := 
sorry

end probability_of_type_I_error_l532_53240


namespace math_problem_l532_53242

noncomputable def log_base (base x : ℝ) : ℝ := Real.log x / Real.log base

theorem math_problem (a b c : ℝ) (h1 : ∃ k : ℤ, log_base c b = k)
  (h2 : log_base a (1 / b) > log_base a (Real.sqrt b) ∧ log_base a (Real.sqrt b) > log_base b (a^2)) :
  (∃ n : ℕ, n = 1 ∧ 
    ((1 / b > Real.sqrt b ∧ Real.sqrt b > a^2) ∨ 
    (Real.log b + log_base a a = 0) ∨ 
    (0 < a ∧ a < b ∧ b < 1) ∨ 
    (a * b = 1))) :=
by sorry

end math_problem_l532_53242


namespace find_a_l532_53216

theorem find_a (a : ℝ) (hne : a ≠ 1) (eq_sets : ∀ x : ℝ, (a-1) * x < a + 5 ↔ 2 * x < 4) : a = 7 :=
sorry

end find_a_l532_53216


namespace right_rect_prism_volume_l532_53222

theorem right_rect_prism_volume (a b c : ℝ) 
  (h1 : a * b = 56) 
  (h2 : b * c = 63) 
  (h3 : a * c = 36) : 
  a * b * c = 504 := by
  sorry

end right_rect_prism_volume_l532_53222


namespace max_group_size_l532_53280

theorem max_group_size 
  (students_class1 : ℕ) (students_class2 : ℕ) 
  (leftover_class1 : ℕ) (leftover_class2 : ℕ) 
  (h_class1 : students_class1 = 69) 
  (h_class2 : students_class2 = 86) 
  (h_leftover1 : leftover_class1 = 5) 
  (h_leftover2 : leftover_class2 = 6) : 
  Nat.gcd (students_class1 - leftover_class1) (students_class2 - leftover_class2) = 16 :=
by
  sorry

end max_group_size_l532_53280


namespace sqrt_sum_simplify_l532_53268

theorem sqrt_sum_simplify :
  Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 := 
by
  sorry

end sqrt_sum_simplify_l532_53268


namespace triangle_angle_B_max_sin_A_plus_sin_C_l532_53221

theorem triangle_angle_B (a b c : ℝ) (A B C : ℝ) (h1 : (a - c) * Real.sin A + c * Real.sin C - b * Real.sin B = 0) 
  (h2 : a / Real.sin A = b / Real.sin B) (h3 : b / Real.sin B = c / Real.sin C) : 
  B = Real.arccos (1/2) := 
sorry

theorem max_sin_A_plus_sin_C (a b c : ℝ) (A B C : ℝ) (h1 : (a - c) * Real.sin A + c * Real.sin C - b * Real.sin B = 0) 
  (h2 : a / Real.sin A = b / Real.sin B) (h3 : b / Real.sin B = c / Real.sin C) 
  (hB : B = Real.arccos (1/2)) : 
  Real.sin A + Real.sin C = Real.sqrt 3 :=
sorry

end triangle_angle_B_max_sin_A_plus_sin_C_l532_53221


namespace joe_new_average_l532_53258

def joe_tests_average (a b c d : ℝ) : Prop :=
  ((a + b + c + d) / 4 = 35) ∧ (min a (min b (min c d)) = 20)

theorem joe_new_average (a b c d : ℝ) (h : joe_tests_average a b c d) :
  ((a + b + c + d - min a (min b (min c d))) / 3 = 40) :=
sorry

end joe_new_average_l532_53258


namespace area_of_region_ABCDEFGHIJ_l532_53239

/-- 
  Given:
  1. Region ABCDEFGHIJ consists of 13 equal squares.
  2. Region ABCDEFGHIJ is inscribed in rectangle PQRS.
  3. Point A is on line PQ, B is on line QR, E is on line RS, and H is on line SP.
  4. PQ has length 28 and QR has length 26.

  Prove that the area of region ABCDEFGHIJ is 338 square units.
-/
theorem area_of_region_ABCDEFGHIJ 
  (squares : ℕ)             -- Number of squares in region ABCDEFGHIJ
  (len_PQ len_QR : ℕ)       -- Lengths of sides PQ and QR
  (area : ℕ)                 -- Area of region ABCDEFGHIJ
  (h1 : squares = 13)
  (h2 : len_PQ = 28)
  (h3 : len_QR = 26)
  : area = 338 :=
sorry

end area_of_region_ABCDEFGHIJ_l532_53239


namespace conic_section_hyperbola_l532_53283

theorem conic_section_hyperbola (x y : ℝ) :
  (x - 3) ^ 2 = 9 * (y + 2) ^ 2 - 81 → conic_section := by
  sorry

end conic_section_hyperbola_l532_53283


namespace problem_l532_53234

def isRightTriangle (a b c : ℝ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

def CannotFormRightTriangle (lst : List ℝ) : Prop :=
  ¬isRightTriangle lst.head! lst.tail.head! lst.tail.tail.head!

theorem problem :
  (¬isRightTriangle 3 4 5 ∧ ¬isRightTriangle 5 12 13 ∧ ¬isRightTriangle 2 3 (Real.sqrt 13)) ∧ CannotFormRightTriangle [4, 6, 8] :=
by
  sorry

end problem_l532_53234


namespace negation_exists_negation_proposition_l532_53250

theorem negation_exists (P : ℝ → Prop) :
  (∃ x : ℝ, P x) ↔ ¬ (∀ x : ℝ, ¬ P x) :=
by sorry

theorem negation_proposition :
  (¬ ∃ x : ℝ, x^2 + 1 < 0) ↔ (∀ x : ℝ, x^2 + 1 ≥ 0) :=
by sorry

end negation_exists_negation_proposition_l532_53250


namespace solve_quadratic_eq_l532_53256

theorem solve_quadratic_eq : ∀ x : ℝ, (12 - 3 * x)^2 = x^2 ↔ x = 6 ∨ x = 3 :=
by
  intro x
  sorry

end solve_quadratic_eq_l532_53256


namespace shortest_routes_l532_53267

theorem shortest_routes
  (side_length : ℝ)
  (refuel_distance : ℝ)
  (total_distance : ℝ)
  (shortest_paths : ℕ) :
  side_length = 10 ∧
  refuel_distance = 30 ∧
  total_distance = 180 →
  shortest_paths = 18 :=
sorry

end shortest_routes_l532_53267


namespace lana_eats_fewer_candies_l532_53229

-- Definitions based on conditions
def canEatNellie : ℕ := 12
def canEatJacob : ℕ := canEatNellie / 2
def candiesBeforeLanaCries : ℕ := 6 -- This is the derived answer for Lana
def initialCandies : ℕ := 30
def remainingCandies : ℕ := 3 * 3 -- After division, each gets 3 candies and they are 3 people

-- Statement to prove how many fewer candies Lana can eat compared to Jacob
theorem lana_eats_fewer_candies :
  canEatJacob = 6 → 
  (initialCandies - remainingCandies = 12 + canEatJacob + candiesBeforeLanaCries) →
  canEatJacob - candiesBeforeLanaCries = 3 :=
by
  intros hJacobEats hCandiesAte
  sorry

end lana_eats_fewer_candies_l532_53229


namespace solve_phi_l532_53212

noncomputable def find_phi (phi : ℝ) : Prop :=
  2 * Real.cos phi - Real.sin phi = Real.sqrt 3 * Real.sin (20 / 180 * Real.pi)

theorem solve_phi (phi : ℝ) :
  find_phi phi ↔ (phi = 140 / 180 * Real.pi ∨ phi = 40 / 180 * Real.pi) :=
sorry

end solve_phi_l532_53212


namespace total_number_of_workers_l532_53255

variables (W N : ℕ)
variables (average_salary_workers average_salary_techs average_salary_non_techs : ℤ)
variables (num_techs total_salary total_salary_techs total_salary_non_techs : ℤ)

theorem total_number_of_workers (h1 : average_salary_workers = 8000)
                               (h2 : average_salary_techs = 14000)
                               (h3 : num_techs = 7)
                               (h4 : average_salary_non_techs = 6000)
                               (h5 : total_salary = W * 8000)
                               (h6 : total_salary_techs = 7 * 14000)
                               (h7 : total_salary_non_techs = N * 6000)
                               (h8 : total_salary = total_salary_techs + total_salary_non_techs)
                               (h9 : W = 7 + N) : 
                               W = 28 :=
sorry

end total_number_of_workers_l532_53255


namespace neither_cable_nor_vcr_fraction_l532_53226

variable (T : ℕ) -- Let T be the total number of housing units

def cableTV_fraction : ℚ := 1 / 5
def VCR_fraction : ℚ := 1 / 10
def both_fraction_given_cable : ℚ := 1 / 4

theorem neither_cable_nor_vcr_fraction : 
  (T : ℚ) * (1 - ((1 / 5) + ((1 / 10) - ((1 / 4) * (1 / 5))))) = (T : ℚ) * (3 / 4) :=
by sorry

end neither_cable_nor_vcr_fraction_l532_53226


namespace hyperbola_equation_l532_53279

theorem hyperbola_equation 
  (h k a c : ℝ)
  (center_cond : (h, k) = (3, -1))
  (vertex_cond : a = abs (2 - (-1)))
  (focus_cond : c = abs (7 - (-1)))
  (b : ℝ)
  (b_square : c^2 = a^2 + b^2) :
  h + k + a + b = 5 + Real.sqrt 55 := 
by
  -- Prove that given the conditions, the value of h + k + a + b is 5 + √55.
  sorry

end hyperbola_equation_l532_53279


namespace age_of_B_l532_53297

theorem age_of_B (A B C : ℕ) (h1 : A = B + 2) (h2 : B = 2 * C) (h3 : A + B + C = 37) : B = 14 :=
by sorry

end age_of_B_l532_53297


namespace solution_set_l532_53259

-- Define the intervals for the solution set
def interval1 : Set ℝ := Set.Ico (5/3) 2
def interval2 : Set ℝ := Set.Ico 2 3

-- Define the function that we need to prove
def equation_holds (x : ℝ) : Prop := Int.floor (Int.floor (3 * x) - 1 / 3) = Int.floor (x + 3)

theorem solution_set :
  { x : ℝ | equation_holds x } = interval1 ∪ interval2 :=
by
  -- Placeholder for the proof
  sorry

end solution_set_l532_53259


namespace prob_all_four_even_dice_l532_53235

noncomputable def probability_even (n : ℕ) : ℚ := (3 / 6)^n

theorem prob_all_four_even_dice : probability_even 4 = 1 / 16 := 
by
  sorry

end prob_all_four_even_dice_l532_53235


namespace hotdogs_total_l532_53223

theorem hotdogs_total:
  let e := 2.5
  let l := 2 * (e * 2)
  let m := 7
  let h := 1.5 * (e * 2)
  let z := 0.5
  (e * 2 + l + m + h + z) = 30 := 
by
  sorry

end hotdogs_total_l532_53223


namespace similar_area_ratios_l532_53287

theorem similar_area_ratios (a₁ a₂ s₁ s₂ : ℝ) (h₁ : a₁ = s₁^2) (h₂ : a₂ = s₂^2) (h₃ : a₁ / a₂ = 1 / 9) (h₄ : s₁ = 4) : s₂ = 12 :=
by
  sorry

end similar_area_ratios_l532_53287


namespace max_value_of_expression_l532_53213

open Real

theorem max_value_of_expression (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1) : 
  2 * x * y + y * z + 2 * z * x ≤ 4 / 7 := 
sorry

end max_value_of_expression_l532_53213


namespace ledi_age_10_in_years_l532_53243

-- Definitions of ages of Duoduo and Ledi
def duoduo_current_age : ℝ := 10
def years_ago : ℝ := 12.3
def sum_ages_years_ago : ℝ := 12

-- Function to calculate Ledi's current age
def ledi_current_age :=
  (sum_ages_years_ago + years_ago + years_ago) + (duoduo_current_age - years_ago)

-- Function to calculate years from now for Ledi to be 10 years old
def years_until_ledi_age_10 (ledi_age_now : ℝ) : ℝ :=
  10 - ledi_age_now

-- Main statement we need to prove
theorem ledi_age_10_in_years : years_until_ledi_age_10 ledi_current_age = 6.3 :=
by
  -- Proof goes here
  sorry

end ledi_age_10_in_years_l532_53243


namespace math_problem_l532_53296

theorem math_problem (x y : ℚ) (h1 : 1/x + 1/y = 4) (h2 : 1/x - 1/y = -5) : x + y = -16/9 := 
sorry

end math_problem_l532_53296


namespace evaluate_fraction_l532_53233

theorem evaluate_fraction : (1 / (2 + (1 / (3 + (1 / 4))))) = 13 / 30 :=
by
  sorry

end evaluate_fraction_l532_53233


namespace average_runs_per_game_l532_53237

-- Define the number of games
def games : ℕ := 6

-- Define the list of runs scored in each game
def runs : List ℕ := [1, 4, 4, 5, 5, 5]

-- The sum of the runs
def total_runs : ℕ := List.sum runs

-- The average runs per game
def avg_runs : ℚ := total_runs / games

-- The theorem to prove
theorem average_runs_per_game : avg_runs = 4 := by sorry

end average_runs_per_game_l532_53237


namespace is_isosceles_right_triangle_l532_53215

theorem is_isosceles_right_triangle 
  {a b c : ℝ}
  (h : |c^2 - a^2 - b^2| + (a - b)^2 = 0) : 
  a = b ∧ c^2 = a^2 + b^2 :=
sorry

end is_isosceles_right_triangle_l532_53215


namespace sufficient_but_not_necessary_condition_l532_53204

theorem sufficient_but_not_necessary_condition (x y : ℝ) :
  (x > 1 ∧ y > 1) → (x + y > 2) ∧ ¬((x + y > 2) → (x > 1 ∧ y > 1)) := 
by
  sorry

end sufficient_but_not_necessary_condition_l532_53204


namespace max_ab_sum_l532_53272

theorem max_ab_sum (a b: ℤ) (h1: a ≠ b) (h2: a * b = -132) (h3: a ≤ b): a + b = -1 :=
sorry

end max_ab_sum_l532_53272


namespace inequality_abc_l532_53264

theorem inequality_abc 
  (a b c : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_order : a ≥ b ∧ b ≥ c) 
  (h_sum : a + b + c ≤ 1) : 
  a^2 + 3 * b^2 + 5 * c^2 ≤ 1 := 
by sorry

end inequality_abc_l532_53264


namespace marie_packs_construction_paper_l532_53210

theorem marie_packs_construction_paper (marie_glue_sticks : ℕ) (allison_glue_sticks : ℕ) (total_allison_items : ℕ)
    (glue_sticks_difference : allison_glue_sticks = marie_glue_sticks + 8)
    (marie_glue_sticks_count : marie_glue_sticks = 15)
    (total_items_allison : total_allison_items = 28)
    (marie_construction_paper_multiplier : ℕ)
    (construction_paper_ratio : marie_construction_paper_multiplier = 6) : 
    ∃ (marie_construction_paper_packs : ℕ), marie_construction_paper_packs = 30 := 
by
  sorry

end marie_packs_construction_paper_l532_53210


namespace square_of_sum_possible_l532_53220

theorem square_of_sum_possible (a b c : ℝ) : 
  ∃ d : ℝ, d = (a + b + c)^2 :=
sorry

end square_of_sum_possible_l532_53220
