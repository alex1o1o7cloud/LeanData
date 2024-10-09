import Mathlib

namespace no_real_y_for_common_solution_l1214_121420

theorem no_real_y_for_common_solution :
  ∀ (x y : ℝ), x^2 + y^2 = 25 → x^2 + 3 * y = 45 → false :=
by 
sorry

end no_real_y_for_common_solution_l1214_121420


namespace quadratic_fraction_formula_l1214_121484

theorem quadratic_fraction_formula (p q α β : ℝ) 
  (h1 : α + β = p) 
  (h2 : α * β = 6) 
  (h3 : p^2 ≠ 12) 
  (h4 : ∃ x : ℝ, x^2 - p * x + q = 0) :
  (α + β) / (α^2 + β^2) = p / (p^2 - 12) :=
sorry

end quadratic_fraction_formula_l1214_121484


namespace total_cost_of_color_drawing_l1214_121429

def cost_bwch_drawing : ℕ := 160
def bwch_to_color_cost_multiplier : ℝ := 1.5

theorem total_cost_of_color_drawing 
  (cost_bwch : ℕ)
  (bwch_to_color_mult : ℝ)
  (h₁ : cost_bwch = 160)
  (h₂ : bwch_to_color_mult = 1.5) :
  cost_bwch * bwch_to_color_mult = 240 := 
  by
    sorry

end total_cost_of_color_drawing_l1214_121429


namespace incorrect_fraction_addition_l1214_121491

theorem incorrect_fraction_addition (a b x y : ℤ) (h1 : 0 < b) (h2 : 0 < y) (h3 : (a + x) * (b * y) = (a * y + b * x) * (b + y)) :
  ∃ k : ℤ, x = -a * k^2 ∧ y = b * k :=
by
  sorry

end incorrect_fraction_addition_l1214_121491


namespace cube_partition_exists_l1214_121408

theorem cube_partition_exists : ∃ (n_0 : ℕ), (0 < n_0) ∧ (∀ (n : ℕ), n ≥ n_0 → ∃ k : ℕ, n = k) := sorry

end cube_partition_exists_l1214_121408


namespace basketball_weight_l1214_121475

theorem basketball_weight (b k : ℝ) (h1 : 6 * b = 4 * k) (h2 : 3 * k = 72) : b = 16 :=
by
  sorry

end basketball_weight_l1214_121475


namespace number_of_subsets_of_five_element_set_is_32_l1214_121445

theorem number_of_subsets_of_five_element_set_is_32 (M : Finset ℕ) (h : M.card = 5) :
    (2 : ℕ) ^ 5 = 32 :=
by
  sorry

end number_of_subsets_of_five_element_set_is_32_l1214_121445


namespace toys_profit_l1214_121482

theorem toys_profit (sp cp : ℕ) (x : ℕ) (h1 : sp = 25200) (h2 : cp = 1200) (h3 : 18 * cp + x * cp = sp) :
  x = 3 :=
by
  sorry

end toys_profit_l1214_121482


namespace S4_equals_15_l1214_121478

noncomputable def S_n (q : ℝ) (n : ℕ) := (1 - q^n) / (1 - q)

theorem S4_equals_15 (q : ℝ) (n : ℕ) (h1 : S_n q 1 = 1) (h2 : S_n q 5 = 5 * S_n q 3 - 4) : 
  S_n q 4 = 15 :=
by
  sorry

end S4_equals_15_l1214_121478


namespace total_crayons_correct_l1214_121443

-- Define the number of crayons each child has
def crayons_per_child : ℕ := 12

-- Define the number of children
def number_of_children : ℕ := 18

-- Define the total number of crayons
def total_crayons : ℕ := crayons_per_child * number_of_children

-- State the theorem
theorem total_crayons_correct : total_crayons = 216 :=
by
  -- Proof goes here
  sorry

end total_crayons_correct_l1214_121443


namespace num_solutions_even_pairs_l1214_121427

theorem num_solutions_even_pairs : ∃ n : ℕ, n = 25 ∧ ∀ (x y : ℕ),
  x % 2 = 0 ∧ y % 2 = 0 ∧ 4 * x + 6 * y = 600 → n = 25 :=
by
  sorry

end num_solutions_even_pairs_l1214_121427


namespace main_theorem_l1214_121417

noncomputable def exists_coprime_integers (a b p : ℤ) : Prop :=
  ∃ (m n : ℤ), Int.gcd m n = 1 ∧ p ∣ (a * m + b * n)

theorem main_theorem (a b p : ℤ) : exists_coprime_integers a b p := 
  sorry

end main_theorem_l1214_121417


namespace closest_point_on_line_l1214_121490

open Real

theorem closest_point_on_line (x y : ℝ) (h_line : y = 4 * x - 3) (h_closest : ∀ p : ℝ × ℝ, (p.snd - -1)^2 + (p.fst - 2)^2 ≥ (y - -1)^2 + (x - 2)^2) :
  x = 10 / 17 ∧ y = 31 / 17 :=
sorry

end closest_point_on_line_l1214_121490


namespace measure_angle_ACB_l1214_121446

-- Definitions of angles and the conditions
def angle_ABD := 140
def angle_BAC := 105
def supplementary_angle (α β : ℕ) := α + β = 180
def angle_sum_property (α β γ : ℕ) := α + β + γ = 180

-- Theorem to prove the measure of angle ACB
theorem measure_angle_ACB (angle_ABD : ℕ) 
                         (angle_BAC : ℕ) 
                         (h1 : supplementary_angle angle_ABD 40)
                         (h2 : angle_sum_property 40 angle_BAC 35) :
  angle_sum_property 40 105 35 :=
sorry

end measure_angle_ACB_l1214_121446


namespace range_of_a_l1214_121494

theorem range_of_a (a : ℝ) (h : a > 0) :
  let A := {x : ℝ | x^2 + 2 * x - 8 > 0}
  let B := {x : ℝ | x^2 - 2 * a * x + 4 ≤ 0}
  (∃! x : ℤ, (x : ℝ) ∈ A ∩ B) → (13 / 6 ≤ a ∧ a < 5 / 2) :=
by
  sorry

end range_of_a_l1214_121494


namespace _l1214_121426

noncomputable def urn_marble_theorem (r w b g y : Nat) : Prop :=
  let n := r + w + b + g + y
  ∃ k : Nat, 
  (k * r * (r-1) * (r-2) * (r-3) * (r-4) / 120 = w * r * (r-1) * (r-2) * (r-3) / 24)
  ∧ (w * r * (r-1) * (r-2) * (r-3) / 24 = w * b * r * (r-1) * (r-2) / 6)
  ∧ (w * b * r * (r-1) * (r-2) / 6 = w * b * g * r * (r-1) / 2)
  ∧ (w * b * g * r * (r-1) / 2 = w * b * g * r * y)
  ∧ n = 55

example : ∃ (r w b g y : Nat), urn_marble_theorem r w b g y := sorry

end _l1214_121426


namespace cos_value_l1214_121437

theorem cos_value (α : ℝ) (h : Real.sin (π / 6 - α) = 1 / 3) : Real.cos (2 * π / 3 + 2 * α) = -7 / 9 :=
by
  sorry

end cos_value_l1214_121437


namespace max_value_2019m_2020n_l1214_121424

theorem max_value_2019m_2020n (m n : ℤ) (h1 : 0 ≤ m - n) (h2 : m - n ≤ 1) (h3 : 2 ≤ m + n) (h4 : m + n ≤ 4) :
  (∀ (m' n' : ℤ), (0 ≤ m' - n') → (m' - n' ≤ 1) → (2 ≤ m' + n') → (m' + n' ≤ 4) → (m - 2 * n ≥ m' - 2 * n')) →
  2019 * m + 2020 * n = 2019 :=
by
  sorry

end max_value_2019m_2020n_l1214_121424


namespace probability_ratio_l1214_121400

theorem probability_ratio :
  let draws := 4
  let total_slips := 40
  let numbers := 10
  let slips_per_number := 4
  let p := 10 / (Nat.choose total_slips draws)
  let q := (Nat.choose numbers 2) * (Nat.choose slips_per_number 2) * (Nat.choose slips_per_number 2) / (Nat.choose total_slips draws)
  p ≠ 0 →
  (q / p) = 162 :=
by
  sorry

end probability_ratio_l1214_121400


namespace altitude_line_equation_equal_distance_lines_l1214_121434

-- Define the points A, B, and C
def A : ℝ × ℝ := (4, 0)
def B : ℝ × ℝ := (8, 10)
def C : ℝ × ℝ := (0, 6)

-- The equation of the line for the altitude from A to BC
theorem altitude_line_equation :
  ∃ (a b c : ℝ), 2 * a - 3 * b + 14 = 0 :=
sorry

-- The equations of the line passing through B such that the distances from A and C are equal
theorem equal_distance_lines :
  ∃ (a b c : ℝ), (7 * a - 6 * b + 4 = 0) ∧ (3 * a + 2 * b - 44 = 0) :=
sorry

end altitude_line_equation_equal_distance_lines_l1214_121434


namespace probability_correct_l1214_121419

-- Define the set and the probability calculation
def set : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Function to check if the difference condition holds
def valid_triplet (a b c: ℕ) : Prop := a < b ∧ b < c ∧ c - a = 4

-- Total number of ways to pick 3 numbers and ways that fit the condition
noncomputable def total_ways : ℕ := Nat.choose 9 3
noncomputable def valid_ways : ℕ := 5 * 2

-- Calculate the probability
noncomputable def probability : ℚ := valid_ways / total_ways

-- The theorem statement
theorem probability_correct : probability = 5 / 42 := by sorry

end probability_correct_l1214_121419


namespace max_sum_abs_coeff_l1214_121474

theorem max_sum_abs_coeff (a b c : ℝ) (f : ℝ → ℝ)
  (h1 : ∀ x, f x = a * x^2 + b * x + c)
  (h2 : |f 1| ≤ 1)
  (h3 : |f (1/2)| ≤ 1)
  (h4 : |f 0| ≤ 1) :
  |a| + |b| + |c| ≤ 17 :=
sorry

end max_sum_abs_coeff_l1214_121474


namespace identical_answers_l1214_121480
-- Import necessary libraries

-- Define the entities and conditions
structure Person :=
  (name : String)
  (always_tells_truth : Bool)

def Fyodor : Person := { name := "Fyodor", always_tells_truth := true }
def Sasha : Person := { name := "Sasha", always_tells_truth := false }

def answer (p : Person) : String :=
  if p.always_tells_truth then "Yes" else "No"

-- The theorem statement
theorem identical_answers :
  answer Fyodor = answer Sasha :=
by
  -- Proof steps will be filled in later
  sorry

end identical_answers_l1214_121480


namespace inverse_proportion_m_range_l1214_121461

theorem inverse_proportion_m_range (m : ℝ) :
  (∀ x : ℝ, x < 0 → ∀ y1 y2 : ℝ, y1 = (1 - 2 * m) / x → y2 = (1 - 2 * m) / (x + 1) → y1 < y2) 
  ↔ (m > 1 / 2) :=
by sorry

end inverse_proportion_m_range_l1214_121461


namespace count_ordered_pairs_l1214_121450

theorem count_ordered_pairs (d n : ℕ) (h₁ : d ≥ 35) (h₂ : n > 0) 
    (h₃ : 45 + 2 * n < 120)
    (h₄ : ∃ a b : ℕ, 10 * a + b = 30 + n ∧ 10 * b + a = 35 + n ∧ a ≤ 9 ∧ b ≤ 9) :
    ∃ k : ℕ, -- number of valid ordered pairs (d, n)
    sorry := sorry

end count_ordered_pairs_l1214_121450


namespace arithmetic_sequences_integer_ratio_count_l1214_121409

theorem arithmetic_sequences_integer_ratio_count 
  (a_n b_n : ℕ → ℕ)
  (A_n B_n : ℕ → ℕ)
  (h₁ : ∀ n, A_n n = n * (a_n 1 + a_n (2 * n - 1)) / 2)
  (h₂ : ∀ n, B_n n = n * (b_n 1 + b_n (2 * n - 1)) / 2)
  (h₃ : ∀ n, A_n n / B_n n = (7 * n + 41) / (n + 3)) :
  ∃ (cnt : ℕ), cnt = 3 ∧ ∀ n, (∃ k, n = 1 + 3 * k) → (a_n n) / (b_n n) = 7 + (10 / (n + 1)) :=
by
  sorry

end arithmetic_sequences_integer_ratio_count_l1214_121409


namespace ratio_Andrea_Jude_l1214_121416

-- Definitions
def number_of_tickets := 100
def tickets_left := 40
def tickets_sold := number_of_tickets - tickets_left

def Jude_tickets := 16
def Sandra_tickets := 4 + 1/2 * Jude_tickets
def Andrea_tickets := tickets_sold - (Jude_tickets + Sandra_tickets)

-- Assertion that needs proof
theorem ratio_Andrea_Jude : 
  (Andrea_tickets / Jude_tickets) = 2 := by
  sorry

end ratio_Andrea_Jude_l1214_121416


namespace concert_tickets_l1214_121418

theorem concert_tickets : ∃ (A B : ℕ), 8 * A + 425 * B = 3000000 ∧ A + B = 4500 ∧ A = 2900 := by
  sorry

end concert_tickets_l1214_121418


namespace sphere_surface_area_ratios_l1214_121473

theorem sphere_surface_area_ratios
  (s : ℝ)
  (r1 : ℝ)
  (r2 : ℝ)
  (r3 : ℝ)
  (h1 : r1 = s / 4 * Real.sqrt 6)
  (h2 : r2 = s / 4 * Real.sqrt 2)
  (h3 : r3 = s / 12 * Real.sqrt 6) :
  (4 * Real.pi * r1^2) / (4 * Real.pi * r3^2) = 9 ∧
  (4 * Real.pi * r2^2) / (4 * Real.pi * r3^2) = 3 ∧
  (4 * Real.pi * r3^2) / (4 * Real.pi * r3^2) = 1 := 
by
  sorry

end sphere_surface_area_ratios_l1214_121473


namespace correct_transformation_C_l1214_121457

-- Define the conditions as given in the problem
def condition_A (x : ℝ) : Prop := 4 + x = 3 ∧ x = 3 - 4
def condition_B (x : ℝ) : Prop := (1 / 3) * x = 0 ∧ x = 0
def condition_C (y : ℝ) : Prop := 5 * y = -4 * y + 2 ∧ 5 * y + 4 * y = 2
def condition_D (a : ℝ) : Prop := (1 / 2) * a - 1 = 3 * a ∧ a - 2 = 6 * a

-- The theorem to prove that condition_C is correctly transformed
theorem correct_transformation_C : condition_C 1 := 
by sorry

end correct_transformation_C_l1214_121457


namespace university_theater_ticket_sales_l1214_121423

theorem university_theater_ticket_sales (total_tickets : ℕ) (adult_price : ℕ) (senior_price : ℕ) (senior_tickets : ℕ) 
  (h1 : total_tickets = 510) (h2 : adult_price = 21) (h3 : senior_price = 15) (h4 : senior_tickets = 327) : 
  (total_tickets - senior_tickets) * adult_price + senior_tickets * senior_price = 8748 :=
by 
  -- Proof skipped
  sorry

end university_theater_ticket_sales_l1214_121423


namespace find_mistaken_divisor_l1214_121477

-- Define the conditions
def remainder : ℕ := 0
def quotient_correct : ℕ := 32
def divisor_correct : ℕ := 21
def quotient_mistaken : ℕ := 56
def dividend : ℕ := quotient_correct * divisor_correct + remainder

-- Prove the mistaken divisor
theorem find_mistaken_divisor : ∃ x : ℕ, dividend = quotient_mistaken * x + remainder ∧ x = 12 :=
by
  -- We leave this as an exercise to the prover
  sorry

end find_mistaken_divisor_l1214_121477


namespace coin_ratio_l1214_121498

theorem coin_ratio (coins_1r coins_50p coins_25p : ℕ) (value_1r value_50p value_25p : ℕ) :
  coins_1r = 120 → coins_50p = 120 → coins_25p = 120 →
  value_1r = coins_1r * 1 → value_50p = coins_50p * 50 → value_25p = coins_25p * 25 →
  value_1r + value_50p + value_25p = 210 →
  (coins_1r : ℚ) / (coins_50p + coins_25p : ℚ) = (1 / 1) :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end coin_ratio_l1214_121498


namespace train_speed_kmph_l1214_121433

noncomputable def train_length : ℝ := 200
noncomputable def crossing_time : ℝ := 3.3330666879982935

theorem train_speed_kmph : (train_length / crossing_time) * 3.6 = 216.00072 := by
  sorry

end train_speed_kmph_l1214_121433


namespace supplement_of_angle_l1214_121449

theorem supplement_of_angle (θ : ℝ) 
  (h_complement: θ = 90 - 30) : 180 - θ = 120 :=
by
  sorry

end supplement_of_angle_l1214_121449


namespace solution_inequality_l1214_121421

theorem solution_inequality {x : ℝ} : x - 1 > 0 ↔ x > 1 := 
by
  sorry

end solution_inequality_l1214_121421


namespace total_thread_needed_l1214_121406

def keychain_length : Nat := 12
def friends_in_classes : Nat := 10
def multiplier_for_club_friends : Nat := 2
def thread_per_class_friend : Nat := 16
def thread_per_club_friend : Nat := 20

theorem total_thread_needed :
  10 * thread_per_class_friend + (10 * multiplier_for_club_friends) * thread_per_club_friend = 560 := by
  sorry

end total_thread_needed_l1214_121406


namespace value_of_f_minus_3_l1214_121458

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * Real.sin x + b * Real.tan x + x^3 + 1

theorem value_of_f_minus_3 (a b : ℝ) (h : f 3 a b = 7) : f (-3) a b = -5 := 
by
  sorry

end value_of_f_minus_3_l1214_121458


namespace b_is_dk_squared_l1214_121431

theorem b_is_dk_squared (a b : ℤ) (h : ∃ r1 r2 r3 : ℤ, (r1 * r2 * r3 = b) ∧ (r1 + r2 + r3 = a) ∧ (r1 * r2 + r1 * r3 + r2 * r3 = 0))
  : ∃ d k : ℤ, (b = d * k^2) ∧ (d ∣ a) := 
sorry

end b_is_dk_squared_l1214_121431


namespace more_green_peaches_than_red_l1214_121456

theorem more_green_peaches_than_red : 
  let red_peaches := 7
  let green_peaches := 8
  green_peaches - red_peaches = 1 := 
by
  let red_peaches := 7
  let green_peaches := 8
  show green_peaches - red_peaches = 1 
  sorry

end more_green_peaches_than_red_l1214_121456


namespace raine_steps_l1214_121432

theorem raine_steps (steps_per_trip : ℕ) (num_days : ℕ) (total_steps : ℕ) : 
  steps_per_trip = 150 → 
  num_days = 5 → 
  total_steps = steps_per_trip * 2 * num_days → 
  total_steps = 1500 := 
by 
  intros h1 h2 h3
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end raine_steps_l1214_121432


namespace math_problem_l1214_121436
-- Import necessary modules

-- Define the condition as a hypothesis and state the theorem
theorem math_problem (x : ℝ) (h : 8 * x - 6 = 10) : 50 * (1 / x) + 150 = 175 :=
sorry

end math_problem_l1214_121436


namespace engineer_walk_duration_l1214_121412

variables (D : ℕ) (S : ℕ) (v : ℕ) (t : ℕ) (t1 : ℕ)

-- Stating the conditions
-- The time car normally takes to travel distance D
-- Speed (S) times the time (t) equals distance (D)
axiom speed_distance_relation : S * t = D

-- Engineer arrives at station at 7:00 AM and walks towards the car
-- They meet at t1 minutes past 7:00 AM, and the car covers part of the distance
-- Engineer reaches factory 20 minutes earlier than usual
-- Therefore, the car now meets the engineer covering less distance and time
axiom car_meets_engineer : S * t1 + v * t1 = D

-- The total travel time to the factory is reduced by 20 minutes
axiom travel_time_reduction : t - t1 = (t - 20 / 60)

-- Mathematically equivalent proof problem
theorem engineer_walk_duration : t1 = 50 := by
  sorry

end engineer_walk_duration_l1214_121412


namespace intersection_m_zero_range_of_m_l1214_121452

def A (x : ℝ) : Prop := x^2 - 2 * x - 3 < 0
def B (x : ℝ) (m : ℝ) : Prop := (x - m + 1) * (x - m - 1) ≥ 0

theorem intersection_m_zero : 
  ∀ x : ℝ, A x → B x 0 ↔ (1 ≤ x ∧ x < 3) :=
sorry

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, A x → B x m) ∧ (∃ x : ℝ, B x m ∧ ¬A x) → (m ≤ -2 ∨ m ≥ 4) :=
sorry

end intersection_m_zero_range_of_m_l1214_121452


namespace probability_of_winning_plan1_is_2_over_5_probability_of_winning_plan2_is_11_over_36_choose_plan1_l1214_121485

-- Definition of the total number of outcomes and outcomes where a player wins for Plan 1
def total_outcomes_plan1 := 15
def winning_outcomes_plan1 := 6
def probability_plan1 : ℚ := winning_outcomes_plan1 / total_outcomes_plan1

-- Definition of the total number of outcomes and outcomes where a player wins for Plan 2
def total_outcomes_plan2 := 36
def winning_outcomes_plan2 := 11
def probability_plan2 : ℚ := winning_outcomes_plan2 / total_outcomes_plan2

-- Statements to prove
theorem probability_of_winning_plan1_is_2_over_5 : probability_plan1 = 2 / 5 :=
by sorry

theorem probability_of_winning_plan2_is_11_over_36 : probability_plan2 = 11 / 36 :=
by sorry

theorem choose_plan1 : probability_plan1 > probability_plan2 :=
by sorry

end probability_of_winning_plan1_is_2_over_5_probability_of_winning_plan2_is_11_over_36_choose_plan1_l1214_121485


namespace total_worth_all_crayons_l1214_121414

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

end total_worth_all_crayons_l1214_121414


namespace sin_A_value_l1214_121444

variables {A B C a b c : ℝ}
variables {sin cos : ℝ → ℝ}

-- Conditions
axiom triangle_sides : ∀ (A B C: ℝ), ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0
axiom sin_cos_conditions : 3 * b * sin A = c * cos A + a * cos C

-- Proof statement
theorem sin_A_value (h : 3 * b * sin A = c * cos A + a * cos C) : sin A = 1 / 3 :=
by 
  sorry

end sin_A_value_l1214_121444


namespace algebraic_identity_l1214_121467

theorem algebraic_identity (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2 * m^2 - 2001 = -2000 :=
by
  sorry

end algebraic_identity_l1214_121467


namespace joe_spent_255_minutes_l1214_121442

-- Define the time taken to cut hair for women, men, and children
def time_per_woman : Nat := 50
def time_per_man : Nat := 15
def time_per_child : Nat := 25

-- Define the number of haircuts for each category
def women_haircuts : Nat := 3
def men_haircuts : Nat := 2
def children_haircuts : Nat := 3

-- Compute the total time spent cutting hair
def total_time_spent : Nat :=
  (women_haircuts * time_per_woman) +
  (men_haircuts * time_per_man) +
  (children_haircuts * time_per_child)

-- The theorem stating the total time spent is equal to 255 minutes
theorem joe_spent_255_minutes : total_time_spent = 255 := by
  sorry

end joe_spent_255_minutes_l1214_121442


namespace remainder_modulo_l1214_121441

theorem remainder_modulo (n : ℤ) (h : n % 50 = 23) : (3 * n - 5) % 15 = 4 := 
by 
  sorry

end remainder_modulo_l1214_121441


namespace find_unique_positive_integers_l1214_121479

theorem find_unique_positive_integers (x y : ℕ) (hx : x > 0) (hy : y > 0) :
  3 ^ x + 7 = 2 ^ y → x = 2 ∧ y = 4 :=
by
  -- Proof will go here
  sorry

end find_unique_positive_integers_l1214_121479


namespace hexagon_perimeter_l1214_121430

def side_length : ℕ := 10
def num_sides : ℕ := 6

theorem hexagon_perimeter : num_sides * side_length = 60 := by
  sorry

end hexagon_perimeter_l1214_121430


namespace cross_number_puzzle_hundreds_digit_l1214_121440

theorem cross_number_puzzle_hundreds_digit :
  ∃ a b : ℕ, a ≥ 5 ∧ a ≤ 6 ∧ b = 3 ∧ (3^a / 100 = 7 ∨ 7^b / 100 = 7) :=
sorry

end cross_number_puzzle_hundreds_digit_l1214_121440


namespace simplest_quadratic_radicals_same_type_l1214_121469

theorem simplest_quadratic_radicals_same_type (m n : ℕ)
  (h : ∀ {a : ℕ}, (a = m - 1 → a = 2) ∧ (a = 4 * n - 1 → a = 7)) :
  m + n = 5 :=
sorry

end simplest_quadratic_radicals_same_type_l1214_121469


namespace general_term_of_sequence_l1214_121410

theorem general_term_of_sequence
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (h_pos_a : ∀ n, 0 < a n)
  (h_pos_b : ∀ n, 0 < b n)
  (h_arith : ∀ n, 2 * b n = a n + a (n + 1))
  (h_geom : ∀ n, (a (n + 1))^2 = b n * b (n + 1))
  (h_a1 : a 1 = 1)
  (h_a2 : a 2 = 3)
  : ∀ n, a n = (n^2 + n) / 2 :=
by
  sorry

end general_term_of_sequence_l1214_121410


namespace min_guests_at_banquet_l1214_121460

-- Definitions based on conditions
def total_food : ℕ := 675
def vegetarian_food : ℕ := 195
def pescatarian_food : ℕ := 220
def carnivorous_food : ℕ := 260

def max_vegetarian_per_guest : ℚ := 3
def max_pescatarian_per_guest : ℚ := 2.5
def max_carnivorous_per_guest : ℚ := 4

-- Definition based on the question and the correct answer
def minimum_number_of_guests : ℕ := 218

-- Lean statement to prove the problem
theorem min_guests_at_banquet :
  195 / 3 + 220 / 2.5 + 260 / 4 = 218 :=
by sorry

end min_guests_at_banquet_l1214_121460


namespace line_equation_l1214_121472

theorem line_equation (a b : ℝ) (h_intercept_eq : a = b) (h_pass_through : 3 * a + 2 * b = 2 * a + 5) : (3 + 2 = 5) ↔ (a = 5 ∧ b = 5) :=
sorry

end line_equation_l1214_121472


namespace inverse_function_ratio_l1214_121403

noncomputable def g (x : ℚ) : ℚ := (3 * x + 2) / (2 * x - 5)

noncomputable def g_inv (x : ℚ) : ℚ := (-5 * x + 2) / (-2 * x + 3)

theorem inverse_function_ratio :
  ∀ x : ℚ, g (g_inv x) = x ∧ (∃ a b c d : ℚ, a = -5 ∧ b = 2 ∧ c = -2 ∧ d = 3 ∧ a / c = 2.5) :=
by
  sorry

end inverse_function_ratio_l1214_121403


namespace sum_n_k_l1214_121438

theorem sum_n_k (n k : ℕ) (h1 : 3 * (k + 1) = n - k) (h2 : 2 * (k + 2) = n - k - 1) : n + k = 13 := by
  sorry

end sum_n_k_l1214_121438


namespace complex_addition_l1214_121401

theorem complex_addition :
  (⟨6, -5⟩ : ℂ) + (⟨3, 2⟩ : ℂ) = ⟨9, -3⟩ := 
sorry

end complex_addition_l1214_121401


namespace simplify_expression_l1214_121428

theorem simplify_expression (x : ℝ) : 3 * x + 5 * x ^ 2 + 2 - (9 - 4 * x - 5 * x ^ 2) = 10 * x ^ 2 + 7 * x - 7 :=
by
  sorry

end simplify_expression_l1214_121428


namespace quadratic_roots_r6_s6_l1214_121407

theorem quadratic_roots_r6_s6 (r s : ℝ) (h1 : r + s = 3 * Real.sqrt 2) (h2 : r * s = 4) : r^6 + s^6 = 648 := by
  sorry

end quadratic_roots_r6_s6_l1214_121407


namespace socks_probability_l1214_121492

theorem socks_probability :
  let total_socks := 18
  let total_pairs := (total_socks.choose 2)
  let gray_socks := 12
  let white_socks := 6
  let gray_pairs := (gray_socks.choose 2)
  let white_pairs := (white_socks.choose 2)
  let same_color_pairs := gray_pairs + white_pairs
  same_color_pairs / total_pairs = (81 / 153) :=
by
  sorry

end socks_probability_l1214_121492


namespace unique_positive_x_for_volume_l1214_121497

variable (x : ℕ)

def prism_volume (x : ℕ) : ℕ :=
  (x + 5) * (x - 5) * (x ^ 2 + 25)

theorem unique_positive_x_for_volume {x : ℕ} (h : prism_volume x < 700) (h_pos : 0 < x) :
  ∃! x, (prism_volume x < 700) ∧ (x - 5 > 0) :=
by
  sorry

end unique_positive_x_for_volume_l1214_121497


namespace function_takes_negative_values_l1214_121495

def f (x a : ℝ) : ℝ := x^2 - a * x + 1

theorem function_takes_negative_values {a : ℝ} :
  (∃ x : ℝ, f x a < 0) ↔ (a > 2 ∨ a < -2) :=
by
  sorry

end function_takes_negative_values_l1214_121495


namespace smallest_sum_3x3_grid_l1214_121462

-- Define the given conditions
def numbers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9] -- List of numbers used in the grid
def total_sum : ℕ := 45 -- Total sum of numbers from 1 to 9
def grid_size : ℕ := 3 -- Size of the grid
def corners_ids : List Nat := [0, 2, 6, 8] -- Indices of the corners in the grid
def remaining_sum : ℕ := 25 -- Sum of the remaining 5 numbers (after excluding the corners)

-- Define the goal: Prove that the smallest sum s is achieved
theorem smallest_sum_3x3_grid : ∃ s : ℕ, 
  (∀ (r : Fin grid_size) (c : Fin grid_size),
    r + c = s) → (s = 12) :=
by
  sorry

end smallest_sum_3x3_grid_l1214_121462


namespace participants_are_multiple_of_7_l1214_121496

theorem participants_are_multiple_of_7 (P : ℕ) (h1 : P % 2 = 0)
  (h2 : ∀ p, p = P / 2 → P + p / 7 = (4 * P) / 7)
  (h3 : (4 * P) / 7 * 7 = 4 * P) : ∃ k : ℕ, P = 7 * k := 
by
  sorry

end participants_are_multiple_of_7_l1214_121496


namespace soda_cost_l1214_121486

variable {b s f : ℕ}

theorem soda_cost :
    5 * b + 3 * s + 2 * f = 520 ∧
    3 * b + 2 * s + f = 340 →
    s = 80 :=
by
  sorry

end soda_cost_l1214_121486


namespace find_a21_l1214_121459

def seq_a (n : ℕ) : ℝ := sorry  -- This should define the sequence a_n
def seq_b (n : ℕ) : ℝ := sorry  -- This should define the sequence b_n

theorem find_a21 (h1 : seq_a 1 = 2)
  (h2 : ∀ n, seq_b n = seq_a (n + 1) / seq_a n)
  (h3 : ∀ n m, seq_b n = seq_b m * r^(n - m)) 
  (h4 : seq_b 10 * seq_b 11 = 2) :
  seq_a 21 = 2 ^ 11 :=
sorry

end find_a21_l1214_121459


namespace no_partition_of_integers_l1214_121415

theorem no_partition_of_integers (A B C : Set ℕ) :
  (A ≠ ∅ ∧ B ≠ ∅ ∧ C ≠ ∅) ∧
  (∀ a b, a ∈ A ∧ b ∈ B → (a^2 - a * b + b^2) ∈ C) ∧
  (∀ a b, a ∈ B ∧ b ∈ C → (a^2 - a * b + b^2) ∈ A) ∧
  (∀ a b, a ∈ C ∧ b ∈ A → (a^2 - a * b + b^2) ∈ B) →
  False := 
sorry

end no_partition_of_integers_l1214_121415


namespace find_x_l1214_121448

theorem find_x (x : ℕ) 
  (h : (744 + 745 + 747 + 748 + 749 + 752 + 752 + 753 + 755 + x) / 10 = 750) : 
  x = 1255 := 
sorry

end find_x_l1214_121448


namespace cube_and_difference_of_squares_l1214_121468

theorem cube_and_difference_of_squares (x : ℤ) (h : x^3 = 9261) : (x + 1) * (x - 1) = 440 :=
by {
  sorry
}

end cube_and_difference_of_squares_l1214_121468


namespace min_phi_l1214_121435

theorem min_phi
  (ϕ : ℝ) (hϕ : ϕ > 0)
  (h_symm : ∃ k : ℤ, 2 * (π / 6) - 2 * ϕ = k * π + π / 2) :
  ϕ = 5 * π / 12 :=
sorry

end min_phi_l1214_121435


namespace rope_length_after_100_cuts_l1214_121422

noncomputable def rope_cut (initial_length : ℝ) (num_cuts : ℕ) (cut_fraction : ℝ) : ℝ :=
  initial_length * (1 - cut_fraction) ^ num_cuts

theorem rope_length_after_100_cuts :
  rope_cut 1 100 (3 / 4) = (1 / 4) ^ 100 :=
by
  sorry

end rope_length_after_100_cuts_l1214_121422


namespace f_is_odd_range_of_x_l1214_121487

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation : ∀ x₁ x₂ : ℝ, f (x₁ + x₂) = f x₁ + f x₂
axiom f_3 : f 3 = 1
axiom f_increase_nonneg : ∀ x₁ x₂ : ℝ, (0 ≤ x₁ ∧ x₁ ≤ x₂) → f x₁ ≤ f x₂
axiom f_lt_2 : ∀ x : ℝ, f (x - 1) < 2

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x :=
sorry

theorem range_of_x : {x : ℝ | f (x - 1) < 2} =
{s : ℝ | sorry } :=
sorry

end f_is_odd_range_of_x_l1214_121487


namespace sin_pi_minus_alpha_l1214_121453

theorem sin_pi_minus_alpha (α : ℝ) (h : Real.sin α = 1 / 2) : Real.sin (π - α) = 1 / 2 :=
by
  sorry

end sin_pi_minus_alpha_l1214_121453


namespace solution_l1214_121488

theorem solution (y q : ℝ) (h1 : |y - 3| = q) (h2 : y < 3) : y - 2 * q = 3 - 3 * q :=
by
  sorry

end solution_l1214_121488


namespace find_a_l1214_121411

open Real

noncomputable def valid_solutions (a b : ℝ) : Prop :=
  a + 2 / b = 17 ∧ b + 2 / a = 1 / 3

theorem find_a (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : valid_solutions a b) :
  a = 6 ∨ a = 17 :=
by sorry

end find_a_l1214_121411


namespace noodles_initial_count_l1214_121413

theorem noodles_initial_count (noodles_given : ℕ) (noodles_now : ℕ) (initial_noodles : ℕ) 
  (h_given : noodles_given = 12) (h_now : noodles_now = 54) (h_initial_noodles : initial_noodles = noodles_now + noodles_given) : 
  initial_noodles = 66 :=
by 
  rw [h_now, h_given] at h_initial_noodles
  exact h_initial_noodles

-- Adding 'sorry' since the solution steps are not required

end noodles_initial_count_l1214_121413


namespace largest_n_exists_l1214_121451

theorem largest_n_exists :
  ∃ (n : ℕ), 
  (∀ (x y z : ℕ), n^2 = 2*x^2 + 2*y^2 + 2*z^2 + 4*x*y + 4*y*z + 4*z*x + 6*x + 6*y + 6*z - 14) → n = 9 :=
sorry

end largest_n_exists_l1214_121451


namespace sunny_weather_prob_correct_l1214_121481

def rain_prob : ℝ := 0.45
def cloudy_prob : ℝ := 0.20
def sunny_prob : ℝ := 1 - rain_prob - cloudy_prob

theorem sunny_weather_prob_correct : sunny_prob = 0.35 := by
  sorry

end sunny_weather_prob_correct_l1214_121481


namespace total_spent_is_195_l1214_121404

def hoodie_cost : ℝ := 80
def flashlight_cost : ℝ := 0.2 * hoodie_cost
def boots_original_cost : ℝ := 110
def boots_discount : ℝ := 0.1
def boots_discounted_cost : ℝ := boots_original_cost * (1 - boots_discount)
def total_cost : ℝ := hoodie_cost + flashlight_cost + boots_discounted_cost

theorem total_spent_is_195 : total_cost = 195 := by
  sorry

end total_spent_is_195_l1214_121404


namespace sum_of_roots_of_polynomials_l1214_121402

theorem sum_of_roots_of_polynomials :
  ∃ (a b : ℝ), (a^4 - 16 * a^3 + 40 * a^2 - 50 * a + 25 = 0) ∧ (b^4 - 24 * b^3 + 216 * b^2 - 720 * b + 625 = 0) ∧ (a + b = 7 ∨ a + b = 3) :=
by 
  sorry

end sum_of_roots_of_polynomials_l1214_121402


namespace hotel_loss_l1214_121455

theorem hotel_loss (operations_expenses : ℝ) (payment_fraction : ℝ) (total_payment : ℝ) (loss : ℝ) 
  (hOpExp : operations_expenses = 100) 
  (hPayFr : payment_fraction = 3 / 4)
  (hTotalPay : total_payment = payment_fraction * operations_expenses) 
  (hLossCalc : loss = operations_expenses - total_payment) : 
  loss = 25 := 
by 
  sorry

end hotel_loss_l1214_121455


namespace shaded_area_percentage_l1214_121447

theorem shaded_area_percentage (n_shaded : ℕ) (n_total : ℕ) (hn_shaded : n_shaded = 21) (hn_total : n_total = 36) :
  ((n_shaded : ℚ) / (n_total : ℚ)) * 100 = 58.33 :=
by
  sorry

end shaded_area_percentage_l1214_121447


namespace probability_of_at_most_3_heads_l1214_121439

-- Definitions based on conditions
def binom : ℕ → ℕ → ℕ
| 0, 0       => 1
| 0, (k + 1) => 0
| (n + 1), 0 => 1
| (n + 1), (k + 1) => binom n k + binom n (k + 1)

def favorable_outcomes : ℕ :=
binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3

def total_outcomes : ℕ := 2 ^ 10

def probability (favorable : ℕ) (total : ℕ) : ℚ :=
favorable / total

-- Theorem statement
theorem probability_of_at_most_3_heads : 
  probability favorable_outcomes total_outcomes = 11 / 64 :=
sorry

end probability_of_at_most_3_heads_l1214_121439


namespace team_leader_and_deputy_choice_l1214_121493

def TeamLeaderSelection : Type := {x : Fin 5 // true}
def DeputyLeaderSelection (TL : TeamLeaderSelection) : Type := {x : Fin 5 // x ≠ TL.val}

theorem team_leader_and_deputy_choice : 
  (Σ TL : TeamLeaderSelection, DeputyLeaderSelection TL) → Fin 20 :=
by sorry

end team_leader_and_deputy_choice_l1214_121493


namespace gcd_8251_6105_l1214_121466

theorem gcd_8251_6105 : Nat.gcd 8251 6105 = 37 := by
  sorry

end gcd_8251_6105_l1214_121466


namespace smallest_b_for_perfect_square_l1214_121499

theorem smallest_b_for_perfect_square : ∃ (b : ℤ), b > 4 ∧ (∃ (n : ℤ), 4 * b + 5 = n ^ 2) ∧ b = 5 := 
sorry

end smallest_b_for_perfect_square_l1214_121499


namespace angle_in_quadrant_l1214_121405

-- Define the problem statement as a theorem to prove
theorem angle_in_quadrant (α : ℝ) (k : ℤ) 
  (hα : 2 * (k:ℝ) * Real.pi + Real.pi < α ∧ α < 2 * (k:ℝ) * Real.pi + 3 * Real.pi / 2) :
  (k:ℝ) * Real.pi + Real.pi / 2 < α / 2 ∧ α / 2 < (k:ℝ) * Real.pi + 3 * Real.pi / 4 := 
sorry

end angle_in_quadrant_l1214_121405


namespace cubic_roots_sum_of_cubes_l1214_121476

def cube_root (x : ℝ) : ℝ := x^(1/3)

theorem cubic_roots_sum_of_cubes :
  let α := cube_root 17
  let β := cube_root 73
  let γ := cube_root 137
  ∀ (a b c : ℝ),
    (a - α) * (a - β) * (a - γ) = 1/2 ∧
    (b - α) * (b - β) * (b - γ) = 1/2 ∧
    (c - α) * (c - β) * (c - γ) = 1/2 →
    a^3 + b^3 + c^3 = 228.5 :=
by {
  sorry
}

end cubic_roots_sum_of_cubes_l1214_121476


namespace transform_graph_of_g_to_f_l1214_121454

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 - Real.sqrt 3 * Real.sin (2 * x)
noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (2 * x) + 1

theorem transform_graph_of_g_to_f :
  ∀ (x : ℝ), f x = g (x + (5 * Real.pi) / 12) :=
by
  sorry

end transform_graph_of_g_to_f_l1214_121454


namespace courtyard_width_l1214_121489

theorem courtyard_width 
  (L : ℝ) (N : ℕ) (brick_length brick_width : ℝ) (courtyard_area : ℝ)
  (hL : L = 18)
  (hN : N = 30000)
  (hbrick_length : brick_length = 0.12)
  (hbrick_width : brick_width = 0.06)
  (hcourtyard_area : courtyard_area = (N : ℝ) * (brick_length * brick_width)) :
  (courtyard_area / L) = 12 :=
by
  sorry

end courtyard_width_l1214_121489


namespace remainder_when_dividing_698_by_13_is_9_l1214_121483

theorem remainder_when_dividing_698_by_13_is_9 :
  ∃ k m : ℤ, 242 = k * 13 + 8 ∧
             698 = m * 13 + 9 ∧
             (k + m) * 13 + 4 = 940 :=
by {
  sorry
}

end remainder_when_dividing_698_by_13_is_9_l1214_121483


namespace additional_people_needed_l1214_121465

theorem additional_people_needed (h₁ : ∀ p h : ℕ, (p * h = 40)) (h₂ : 5 * 8 = 40) : 7 - 5 = 2 :=
by
  sorry

end additional_people_needed_l1214_121465


namespace TwentyFifthMultipleOfFour_l1214_121471

theorem TwentyFifthMultipleOfFour (n : ℕ) (h : ∀ k, 0 <= k ∧ k <= 24 → n = 16 + 4 * k) : n = 112 :=
by
  sorry

end TwentyFifthMultipleOfFour_l1214_121471


namespace calculator_transform_implication_l1214_121463

noncomputable def transform (x n S : ℕ) : Prop :=
  (S > x^n + 1)

theorem calculator_transform_implication (x n S : ℕ) (hx : 0 < x) (hn : 0 < n) (hS : 0 < S) 
  (h_transform: transform x n S) : S > x^n + x - 1 := by
  sorry

end calculator_transform_implication_l1214_121463


namespace total_cost_correct_l1214_121470

-- Define the individual costs and quantities
def pumpkin_cost : ℝ := 2.50
def tomato_cost : ℝ := 1.50
def chili_pepper_cost : ℝ := 0.90

def pumpkin_quantity : ℕ := 3
def tomato_quantity : ℕ := 4
def chili_pepper_quantity : ℕ := 5

-- Define the total cost calculation
def total_cost : ℝ :=
  pumpkin_quantity * pumpkin_cost +
  tomato_quantity * tomato_cost +
  chili_pepper_quantity * chili_pepper_cost

-- Prove the total cost is $18.00
theorem total_cost_correct : total_cost = 18.00 := by
  sorry

end total_cost_correct_l1214_121470


namespace j_h_five_l1214_121425

-- Define the functions h and j
def h (x : ℤ) : ℤ := 4 * x + 5
def j (x : ℤ) : ℤ := 6 * x - 11

-- State the theorem to prove j(h(5)) = 139
theorem j_h_five : j (h 5) = 139 := by
  sorry

end j_h_five_l1214_121425


namespace num_ways_to_distribute_balls_into_boxes_l1214_121464

theorem num_ways_to_distribute_balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls = 243) := 
by
  sorry

end num_ways_to_distribute_balls_into_boxes_l1214_121464
