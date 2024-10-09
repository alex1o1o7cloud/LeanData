import Mathlib

namespace xiaohua_amount_paid_l2374_237444

def cost_per_bag : ℝ := 18
def discount_rate : ℝ := 0.1
def price_difference : ℝ := 36

theorem xiaohua_amount_paid (x : ℝ) 
  (h₁ : 18 * (x+1) * (1 - 0.1) = 18 * x - 36) :
  18 * (x + 1) * (1 - 0.1) = 486 := 
sorry

end xiaohua_amount_paid_l2374_237444


namespace hypotenuse_length_l2374_237427

theorem hypotenuse_length (a b c : ℝ) (h₁ : a + b + c = 40) (h₂ : 0.5 * a * b = 24) (h₃ : a^2 + b^2 = c^2) : c = 18.8 := sorry

end hypotenuse_length_l2374_237427


namespace biscuits_initial_l2374_237432

theorem biscuits_initial (F M A L B : ℕ) 
  (father_gave : F = 13) 
  (mother_gave : M = 15) 
  (brother_ate : A = 20) 
  (left_with : L = 40) 
  (remaining : B + F + M - A = L) :
  B = 32 := 
by 
  subst father_gave
  subst mother_gave
  subst brother_ate
  subst left_with
  simp at remaining
  linarith

end biscuits_initial_l2374_237432


namespace order_of_a_b_c_l2374_237481

noncomputable def a : ℝ := Real.log 3 / Real.log 2
noncomputable def b : ℝ := Real.log 2 / Real.log 3
noncomputable def c : ℝ := (1 / 2) * (Real.log 5 / Real.log 2)

theorem order_of_a_b_c : a > c ∧ c > b :=
by
  -- proof here
  sorry

end order_of_a_b_c_l2374_237481


namespace halt_duration_l2374_237470

theorem halt_duration (avg_speed : ℝ) (distance : ℝ) (start_time end_time : ℝ) (halt_duration : ℝ) :
  avg_speed = 87 ∧ distance = 348 ∧ start_time = 9 ∧ end_time = 13.75 →
  halt_duration = (end_time - start_time) - (distance / avg_speed) → 
  halt_duration = 0.75 :=
by
  sorry

end halt_duration_l2374_237470


namespace power_equality_l2374_237423

theorem power_equality (n : ℝ) : (9:ℝ)^4 = (27:ℝ)^n → n = (8:ℝ) / 3 :=
by
  sorry

end power_equality_l2374_237423


namespace muffin_banana_cost_ratio_l2374_237491

variables (m b c : ℕ) -- costs of muffin, banana, and cookie respectively
variables (susie_cost calvin_cost : ℕ)

-- Conditions
def susie_cost_eq : Prop := susie_cost = 5 * m + 4 * b + 2 * c
def calvin_cost_eq : Prop := calvin_cost = 3 * (5 * m + 4 * b + 2 * c)
def calvin_cost_eq_reduced : Prop := calvin_cost = 3 * m + 20 * b + 6 * c
def cookie_cost_eq : Prop := c = 2 * b

-- Question and Answer
theorem muffin_banana_cost_ratio
  (h1 : susie_cost_eq m b c susie_cost)
  (h2 : calvin_cost_eq m b c calvin_cost)
  (h3 : calvin_cost_eq_reduced m b c calvin_cost)
  (h4 : cookie_cost_eq b c)
  : m = 4 * b / 3 :=
sorry

end muffin_banana_cost_ratio_l2374_237491


namespace isosceles_triangle_sides_l2374_237471

theorem isosceles_triangle_sides (length_rope : ℝ) (one_side : ℝ) (a b : ℝ) :
  length_rope = 18 ∧ one_side = 5 ∧ a + a + one_side = length_rope ∧ b = one_side ∨ b + b + one_side = length_rope -> (a = 6.5 ∨ a = 5) ∧ (b = 6.5 ∨ b = 5) :=
by
  sorry

end isosceles_triangle_sides_l2374_237471


namespace solve_system_of_eq_l2374_237454

noncomputable def system_of_eq (x y z : ℝ) : Prop :=
  y = x^3 * (3 - 2 * x) ∧
  z = y^3 * (3 - 2 * y) ∧
  x = z^3 * (3 - 2 * z)

theorem solve_system_of_eq (x y z : ℝ) :
  system_of_eq x y z →
  (x = 0 ∧ y = 0 ∧ z = 0) ∨
  (x = 1 ∧ y = 1 ∧ z = 1) ∨
  (x = -1/2 ∧ y = -1/2 ∧ z = -1/2) :=
sorry

end solve_system_of_eq_l2374_237454


namespace hamburgers_second_day_l2374_237458

theorem hamburgers_second_day (x H D : ℕ) (h1 : 3 * H + 4 * D = 10) (h2 : x * H + 3 * D = 7) (h3 : D = 1) (h4 : H = 2) :
  x = 2 :=
by
  sorry

end hamburgers_second_day_l2374_237458


namespace Paula_initial_cans_l2374_237415

theorem Paula_initial_cans :
  ∀ (cans rooms_lost : ℕ), rooms_lost = 10 → 
  (40 / (rooms_lost / 5) = cans + 5 → cans = 20) :=
by
  intros cans rooms_lost h_rooms_lost h_calculation
  sorry

end Paula_initial_cans_l2374_237415


namespace num_nonnegative_real_values_l2374_237434

theorem num_nonnegative_real_values :
  ∃ n : ℕ, ∀ x : ℝ, (x ≥ 0) → (∃ k : ℕ, (169 - (x^(1/3))) = k^2) → n = 27 := 
sorry

end num_nonnegative_real_values_l2374_237434


namespace power_identity_l2374_237430

theorem power_identity (x y a b : ℝ) (h1 : 10^x = a) (h2 : 10^y = b) : 10^(3*x + 2*y) = a^3 * b^2 := 
by 
  sorry

end power_identity_l2374_237430


namespace lex_reads_in_12_days_l2374_237496

theorem lex_reads_in_12_days
  (total_pages : ℕ)
  (pages_per_day : ℕ)
  (h1 : total_pages = 240)
  (h2 : pages_per_day = 20) :
  total_pages / pages_per_day = 12 :=
by
  sorry

end lex_reads_in_12_days_l2374_237496


namespace cabbage_count_l2374_237490

theorem cabbage_count 
  (length : ℝ)
  (width : ℝ)
  (density : ℝ)
  (h_length : length = 16)
  (h_width : width = 12)
  (h_density : density = 9) : 
  length * width * density = 1728 := 
by
  rw [h_length, h_width, h_density]
  norm_num
  done

end cabbage_count_l2374_237490


namespace inscribed_circles_radii_sum_l2374_237460

noncomputable def sum_of_radii (d : ℝ) (r1 r2 : ℝ) : Prop :=
  r1 + r2 = d / 2

theorem inscribed_circles_radii_sum (d : ℝ) (h : d = 23) (r1 r2 : ℝ) (h1 : r1 + r2 = d / 2) :
  r1 + r2 = 23 / 2 :=
by
  rw [h] at h1
  exact h1

end inscribed_circles_radii_sum_l2374_237460


namespace original_deck_card_count_l2374_237440

variable (r b : ℕ)

theorem original_deck_card_count (h1 : r / (r + b) = 1 / 4) (h2 : r / (r + b + 6) = 1 / 6) : r + b = 12 :=
by
  -- The proof goes here
  sorry

end original_deck_card_count_l2374_237440


namespace quadratic_roots_identity_l2374_237448

variable (α β : ℝ)
variable (h1 : α^2 + 3*α - 7 = 0)
variable (h2 : β^2 + 3*β - 7 = 0)

-- The problem is to prove that α^2 + 4*α + β = 4
theorem quadratic_roots_identity :
  α^2 + 4*α + β = 4 :=
sorry

end quadratic_roots_identity_l2374_237448


namespace find_distance_l2374_237420

-- Definitions of given conditions
def speed : ℝ := 65 -- km/hr
def time  : ℝ := 3  -- hr

-- Statement: The distance is 195 km given the speed and time.
theorem find_distance (speed : ℝ) (time : ℝ) : (speed * time = 195) :=
by
  sorry

end find_distance_l2374_237420


namespace cheese_pops_count_l2374_237492

-- Define the number of hotdogs, chicken nuggets, and total portions
def hotdogs : ℕ := 30
def chicken_nuggets : ℕ := 40
def total_portions : ℕ := 90

-- Define the number of bite-sized cheese pops
def cheese_pops : ℕ := total_portions - hotdogs - chicken_nuggets

-- Theorem to prove that the number of bite-sized cheese pops Andrew brought is 20
theorem cheese_pops_count :
  cheese_pops = 20 :=
by
  -- The following proof is omitted
  sorry

end cheese_pops_count_l2374_237492


namespace part1_part2_l2374_237411

theorem part1 (m : ℝ) : 
  (∀ x y : ℝ, (x^2 + y^2 - 2 * x + 4 * y - 4 = 0 ∧ y = x + m) → -3 - 3 * Real.sqrt 2 < m ∧ m < -3 + 3 * Real.sqrt 2) :=
sorry

theorem part2 (m x1 x2 y1 y2 : ℝ) (h1 : x1 + x2 = -(m + 1)) (h2 : x1 * x2 = (m^2 + 4 * m - 4) / 2) 
(h3 : (x - x1) * (x - x2) + (x1 + m) * (x2 + m) = 0) : 
  m = -4 ∨ m = 1 →
  (∀ x y : ℝ, y = x + m ↔ x - y - 4 = 0 ∨ x - y + 1 = 0) :=
sorry

end part1_part2_l2374_237411


namespace no_polygon_with_half_parallel_diagonals_l2374_237424

open Set

noncomputable def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

def is_parallel_diagonal (n i j : ℕ) : Bool := 
  -- Here, you should define the mathematical condition of a diagonal being parallel to a side
  ((j - i) % n = 0) -- This is a placeholder; the actual condition would depend on the precise geometric definition.

theorem no_polygon_with_half_parallel_diagonals (n : ℕ) (h1 : n ≥ 3) :
  ¬(∃ (k : ℕ), k = num_diagonals n ∧ (∀ (i j : ℕ), i < j ∧ is_parallel_diagonal n i j = true → k = num_diagonals n / 2)) :=
by
  sorry

end no_polygon_with_half_parallel_diagonals_l2374_237424


namespace union_A_B_l2374_237405

def A : Set ℝ := {x | x^2 - 2 * x < 0}
def B : Set ℝ := {x | ∃ y, y = Real.log (x - 1)}

theorem union_A_B : A ∪ B = {x | x > 0} :=
by
  sorry

end union_A_B_l2374_237405


namespace bromine_atoms_in_compound_l2374_237462

theorem bromine_atoms_in_compound
  (atomic_weight_H : ℕ := 1)
  (atomic_weight_Br : ℕ := 80)
  (atomic_weight_O : ℕ := 16)
  (total_molecular_weight : ℕ := 129) :
  ∃ (n : ℕ), total_molecular_weight = atomic_weight_H + n * atomic_weight_Br + 3 * atomic_weight_O ∧ n = 1 := 
by
  sorry

end bromine_atoms_in_compound_l2374_237462


namespace cost_per_remaining_ticket_is_seven_l2374_237429

def total_tickets : ℕ := 29
def nine_dollar_tickets : ℕ := 11
def total_cost : ℕ := 225
def nine_dollar_ticket_cost : ℕ := 9
def remaining_tickets : ℕ := total_tickets - nine_dollar_tickets

theorem cost_per_remaining_ticket_is_seven :
  (total_cost - nine_dollar_tickets * nine_dollar_ticket_cost) / remaining_tickets = 7 :=
  sorry

end cost_per_remaining_ticket_is_seven_l2374_237429


namespace min_value_n_l2374_237409

noncomputable def minN : ℕ :=
  5

theorem min_value_n :
  ∀ (S : Finset ℕ), (∀ n ∈ S, 1 ≤ n ∧ n ≤ 9) ∧ S.card = minN → 
    (∃ T ⊆ S, T ≠ ∅ ∧ 10 ∣ (T.sum id)) :=
by
  sorry

end min_value_n_l2374_237409


namespace triangle_right_angle_solution_l2374_237451

def is_right_angle (a b : ℝ × ℝ) : Prop := (a.1 * b.1 + a.2 * b.2 = 0)

def vector_sub (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)

theorem triangle_right_angle_solution (x : ℝ) (h1 : (2, -1) = (2, -1)) (h2 : (x, 3) = (x, 3)) : 
  is_right_angle (2, -1) (x, 3) ∨ 
  is_right_angle (2, -1) (vector_sub (x, 3) (2, -1)) ∨ 
  is_right_angle (x, 3) (vector_sub (x, 3) (2, -1)) → 
  x = 3 / 2 ∨ x = 4 :=
sorry

end triangle_right_angle_solution_l2374_237451


namespace Bruce_paid_l2374_237435

noncomputable def total_paid : ℝ :=
  let grapes_price := 9 * 70 * (1 - 0.10)
  let mangoes_price := 7 * 55 * (1 - 0.05)
  let oranges_price := 5 * 45 * (1 - 0.15)
  let apples_price := 3 * 80 * (1 - 0.20)
  grapes_price + mangoes_price + oranges_price + apples_price

theorem Bruce_paid (h : total_paid = 1316.25) : true :=
by
  -- This is where the proof would be
  sorry

end Bruce_paid_l2374_237435


namespace zero_in_interval_l2374_237416

theorem zero_in_interval {b : ℝ} (f : ℝ → ℝ)
  (h₁ : ∀ x, f x = 2 * b * x - 3 * b + 1)
  (h₂ : b > 1/5)
  (h₃ : b < 1) :
  ∃ x, -1 < x ∧ x < 1 ∧ f x = 0 :=
by
  sorry

end zero_in_interval_l2374_237416


namespace problem_decimal_parts_l2374_237499

theorem problem_decimal_parts :
  let a := 5 + Real.sqrt 7 - 7
  let b := 5 - Real.sqrt 7 - 2
  (a + b) ^ 2023 = 1 :=
by
  sorry

end problem_decimal_parts_l2374_237499


namespace ned_initial_video_games_l2374_237450

theorem ned_initial_video_games : ∀ (w t : ℕ), 7 * w = 63 ∧ t = w + 6 → t = 15 := by
  intro w t
  intro h
  sorry

end ned_initial_video_games_l2374_237450


namespace evaluate_polynomial_at_three_l2374_237468

def polynomial (x : ℕ) : ℕ :=
  x^6 + 2 * x^5 + 4 * x^3 + 5 * x^2 + 6 * x + 12

theorem evaluate_polynomial_at_three :
  polynomial 3 = 588 :=
by
  sorry

end evaluate_polynomial_at_three_l2374_237468


namespace winner_percentage_l2374_237442

variable (votes_winner : ℕ) (win_by : ℕ)
variable (total_votes : ℕ)
variable (percentage_winner : ℕ)

-- Conditions
def conditions : Prop :=
  votes_winner = 930 ∧
  win_by = 360 ∧
  total_votes = votes_winner + (votes_winner - win_by) ∧
  percentage_winner = (votes_winner * 100) / total_votes

-- Theorem to prove
theorem winner_percentage (h : conditions votes_winner win_by total_votes percentage_winner) : percentage_winner = 62 :=
sorry

end winner_percentage_l2374_237442


namespace common_difference_is_3_l2374_237488

variables {a : ℕ → ℝ} {d a1 : ℝ}

-- Define the arithmetic sequence
def arithmetic_sequence (a_n : ℕ → ℝ) (a1 d : ℝ) : Prop := 
  ∀ n, a_n n = a1 + (n - 1) * d

-- Conditions
def a2_eq : a 2 = 3 := sorry
def a5_eq : a 5 = 12 := sorry

-- Theorem to prove the common difference is 3
theorem common_difference_is_3 :
  ∀ {a : ℕ → ℝ} {a1 d : ℝ},
  (arithmetic_sequence a a1 d)
  → a 2 = 3 
  → a 5 = 12 
  → d = 3 :=
  by
  intros a a1 d h_seq h_a2 h_a5
  sorry

end common_difference_is_3_l2374_237488


namespace find_number_l2374_237473

theorem find_number (N : ℝ) (h : 0.4 * (3 / 5) * N = 36) : N = 150 :=
sorry

end find_number_l2374_237473


namespace first_prime_year_with_digit_sum_8_l2374_237433

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |> List.sum

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

theorem first_prime_year_with_digit_sum_8 :
  ∃ y : ℕ, y > 2015 ∧ sum_of_digits y = 8 ∧ is_prime y ∧
  ∀ z : ℕ, z > 2015 ∧ sum_of_digits z = 8 ∧ is_prime z → y ≤ z :=
sorry

end first_prime_year_with_digit_sum_8_l2374_237433


namespace Mary_younger_than_Albert_l2374_237400

-- Define the basic entities and conditions
def Betty_age : ℕ := 11
def Albert_age : ℕ := 4 * Betty_age
def Mary_age : ℕ := Albert_age / 2

-- Define the property to prove
theorem Mary_younger_than_Albert : Albert_age - Mary_age = 22 :=
by 
  sorry

end Mary_younger_than_Albert_l2374_237400


namespace calc3aMinus4b_l2374_237419

theorem calc3aMinus4b (a b : ℤ) (h1 : a * 1 - b * 2 = -1) (h2 : a * 1 + b * 2 = 7) : 3 * a - 4 * b = 1 :=
by
  /- Proof goes here -/
  sorry

end calc3aMinus4b_l2374_237419


namespace prob1_prob2_l2374_237455

-- Problem 1
theorem prob1 (x y : ℝ) : 3 * x^2 * y * (-2 * x * y)^3 = -24 * x^5 * y^4 :=
sorry

-- Problem 2
theorem prob2 (x y : ℝ) : (5 * x + 2 * y) * (3 * x - 2 * y) = 15 * x^2 - 4 * x * y - 4 * y^2 :=
sorry

end prob1_prob2_l2374_237455


namespace car_dealer_bmw_sales_l2374_237425

theorem car_dealer_bmw_sales (total_cars : ℕ)
  (vw_percentage : ℝ)
  (toyota_percentage : ℝ)
  (acura_percentage : ℝ)
  (bmw_count : ℕ) :
  total_cars = 300 →
  vw_percentage = 0.10 →
  toyota_percentage = 0.25 →
  acura_percentage = 0.20 →
  bmw_count = total_cars * (1 - (vw_percentage + toyota_percentage + acura_percentage)) →
  bmw_count = 135 :=
by
  intros
  sorry

end car_dealer_bmw_sales_l2374_237425


namespace geometric_series_sum_l2374_237467

theorem geometric_series_sum :
  let a := -2
  let r := 4
  let n := 10
  let S := (a * (r^n - 1)) / (r - 1)
  S = -699050 :=
by
  sorry

end geometric_series_sum_l2374_237467


namespace abe_age_equation_l2374_237459

theorem abe_age_equation (a : ℕ) (x : ℕ) (h1 : a = 19) (h2 : a + (a - x) = 31) : x = 7 :=
by
  sorry

end abe_age_equation_l2374_237459


namespace solve_quadratic_l2374_237452

theorem solve_quadratic :
  ∀ x : ℝ, (x^2 - 3 * x + 2 = 0) → (x = 1 ∨ x = 2) :=
by sorry

end solve_quadratic_l2374_237452


namespace popsicle_sticks_sum_l2374_237403

-- Define the number of popsicle sticks each person has
def Gino_popsicle_sticks : Nat := 63
def my_popsicle_sticks : Nat := 50

-- Formulate the theorem stating the sum of popsicle sticks
theorem popsicle_sticks_sum : Gino_popsicle_sticks + my_popsicle_sticks = 113 := by
  sorry

end popsicle_sticks_sum_l2374_237403


namespace B_takes_6_days_to_complete_work_alone_l2374_237413

theorem B_takes_6_days_to_complete_work_alone 
    (work_duration_A : ℕ) 
    (work_payment : ℚ)
    (work_days_with_C : ℕ) 
    (payment_C : ℚ) 
    (combined_work_rate_A_B_C : ℚ)
    (amount_to_be_shared_A_B : ℚ) 
    (combined_daily_earning_A_B : ℚ) :
  work_duration_A = 6 ∧
  work_payment = 3360 ∧ 
  work_days_with_C = 3 ∧ 
  payment_C = 420.00000000000017 ∧ 
  combined_work_rate_A_B_C = 1 / 3 ∧ 
  amount_to_be_shared_A_B = 2940 ∧ 
  combined_daily_earning_A_B = 980 → 
  work_duration_A = 6 ∧
  (∃ (work_duration_B : ℕ), work_duration_B = 6) :=
by 
  sorry

end B_takes_6_days_to_complete_work_alone_l2374_237413


namespace quadratic_common_root_inverse_other_roots_l2374_237407

variables (p q r s : ℝ)
variables (hq : q ≠ -1) (hs : s ≠ -1)

theorem quadratic_common_root_inverse_other_roots :
  (∃ a b : ℝ, (a ≠ b) ∧ (a^2 + p * a + q = 0) ∧ (a * b = 1) ∧ (b^2 + r * b + s = 0)) ↔ 
  (p * r = (q + 1) * (s + 1) ∧ p * (q + 1) * s = r * (s + 1) * q) :=
sorry

end quadratic_common_root_inverse_other_roots_l2374_237407


namespace simplified_sum_l2374_237472

theorem simplified_sum :
  (-2^2003) + (2^2004) + (-2^2005) - (2^2006) = 5 * (2^2003) :=
by
  sorry

end simplified_sum_l2374_237472


namespace arithmetic_sequence_sum_l2374_237464

variable {α : Type} [LinearOrderedField α]

noncomputable def a_n (a1 d n : α) := a1 + (n - 1) * d

theorem arithmetic_sequence_sum (a1 d : α) (h1 : a_n a1 d 3 * a_n a1 d 11 = 5)
  (h2 : a_n a1 d 3 + a_n a1 d 11 = 3) : a_n a1 d 5 + a_n a1 d 6 + a_n a1 d 10 = 9 / 2 :=
by
  sorry

end arithmetic_sequence_sum_l2374_237464


namespace semi_circle_radius_l2374_237487

theorem semi_circle_radius (π : ℝ) (hπ : Real.pi = π) (P : ℝ) (hP : P = 180) : 
  ∃ r : ℝ, r = 180 / (π + 2) :=
by
  sorry

end semi_circle_radius_l2374_237487


namespace extra_charge_per_wand_l2374_237486

theorem extra_charge_per_wand
  (cost_per_wand : ℕ)
  (num_wands : ℕ)
  (total_collected : ℕ)
  (num_wands_sold : ℕ)
  (h_cost : cost_per_wand = 60)
  (h_num_wands : num_wands = 3)
  (h_total_collected : total_collected = 130)
  (h_num_wands_sold : num_wands_sold = 2) :
  ((total_collected / num_wands_sold) - cost_per_wand) = 5 :=
by
  -- Proof goes here
  sorry

end extra_charge_per_wand_l2374_237486


namespace problem_statement_l2374_237437

variables {Line Plane : Type}

-- Defining the perpendicular relationship between a line and a plane
def perp (a : Line) (α : Plane) : Prop := sorry

-- Defining the parallel relationship between two planes
def para (α β : Plane) : Prop := sorry

-- The main statement to prove
theorem problem_statement (a : Line) (α β : Plane) (h1 : perp a α) (h2 : perp a β) : para α β := 
sorry

end problem_statement_l2374_237437


namespace sufficient_but_not_necessary_condition_l2374_237465

theorem sufficient_but_not_necessary_condition (a : ℝ) (h : a ≠ 0) :
  (a > 2 ↔ |a - 1| > 1) ↔ (a > 2 → |a - 1| > 1) ∧ (a < 0 → |a - 1| > 1) ∧ (∃ x : ℝ, (|x - 1| > 1) ∧ x < 0 ∧ x ≠ a) :=
by
  sorry

end sufficient_but_not_necessary_condition_l2374_237465


namespace technician_round_trip_completion_percentage_l2374_237431

theorem technician_round_trip_completion_percentage :
  ∀ (d total_d : ℝ),
  d = 1 + (0.75 * 1) + (0.5 * 1) + (0.25 * 1) →
  total_d = 4 * 2 →
  (d / total_d) * 100 = 31.25 :=
by
  intros d total_d h1 h2
  sorry

end technician_round_trip_completion_percentage_l2374_237431


namespace tangent_and_normal_lines_l2374_237478

noncomputable def x (t : ℝ) := 2 * Real.exp t
noncomputable def y (t : ℝ) := Real.exp (-t)

theorem tangent_and_normal_lines (t0 : ℝ) (x0 y0 : ℝ) (m_tangent m_normal : ℝ)
  (hx0 : x0 = x t0)
  (hy0 : y0 = y t0)
  (hm_tangent : m_tangent = -(1 / 2))
  (hm_normal : m_normal = 2) :
  (∀ x y : ℝ, y = m_tangent * x + 2) ∧ (∀ x y : ℝ, y = m_normal * x - 3) :=
by
  sorry

end tangent_and_normal_lines_l2374_237478


namespace faces_of_prism_with_24_edges_l2374_237441

theorem faces_of_prism_with_24_edges (L : ℕ) (h1 : 3 * L = 24) : L + 2 = 10 := by
  sorry

end faces_of_prism_with_24_edges_l2374_237441


namespace find_a_value_l2374_237474

theorem find_a_value :
  (∀ y : ℝ, y ∈ Set.Ioo (-3/2 : ℝ) 4 → y * (2 * y - 3) < (12 : ℝ)) ↔ (12 = 12) := 
by 
  sorry

end find_a_value_l2374_237474


namespace parabola_focus_l2374_237418

theorem parabola_focus (h : ∀ x y : ℝ, y ^ 2 = -12 * x → True) : (-3, 0) = (-3, 0) :=
  sorry

end parabola_focus_l2374_237418


namespace vishal_investment_more_than_trishul_l2374_237477

theorem vishal_investment_more_than_trishul :
  ∀ (V T R : ℝ), R = 2000 → T = R - 0.10 * R → V + T + R = 5780 → (V - T) / T * 100 = 10 :=
by
  intros V T R hR hT hSum
  sorry

end vishal_investment_more_than_trishul_l2374_237477


namespace greatest_prime_factor_of_144_l2374_237443

-- Define the number 144
def num : ℕ := 144

-- Define what it means for a number to be a prime factor of num
def is_prime_factor (p n : ℕ) : Prop :=
  Prime p ∧ p ∣ n

-- Define what it means to be the greatest prime factor
def greatest_prime_factor (p n : ℕ) : Prop :=
  is_prime_factor p n ∧ (∀ q, is_prime_factor q n → q ≤ p)

-- Prove that the greatest prime factor of 144 is 3
theorem greatest_prime_factor_of_144 : greatest_prime_factor 3 num :=
sorry

end greatest_prime_factor_of_144_l2374_237443


namespace cost_of_each_soda_l2374_237438

def initial_money := 20
def change_received := 14
def number_of_sodas := 3

theorem cost_of_each_soda :
  (initial_money - change_received) / number_of_sodas = 2 :=
by
  sorry

end cost_of_each_soda_l2374_237438


namespace accurate_place_24000_scientific_notation_46400000_l2374_237410

namespace MathProof

def accurate_place (n : ℕ) : String :=
  if n = 24000 then "hundred's place" else "unknown"

def scientific_notation (n : ℕ) : String :=
  if n = 46400000 then "4.64 × 10^7" else "unknown"

theorem accurate_place_24000 : accurate_place 24000 = "hundred's place" :=
by
  sorry

theorem scientific_notation_46400000 : scientific_notation 46400000 = "4.64 × 10^7" :=
by
  sorry

end MathProof

end accurate_place_24000_scientific_notation_46400000_l2374_237410


namespace factorize_quadratic_l2374_237485

theorem factorize_quadratic : ∀ x : ℝ, x^2 - 7*x + 10 = (x - 2)*(x - 5) :=
by
  sorry

end factorize_quadratic_l2374_237485


namespace find_length_CD_m_plus_n_l2374_237428

noncomputable def lengthAB : ℝ := 7
noncomputable def lengthBD : ℝ := 11
noncomputable def lengthBC : ℝ := 9

axiom angle_BAD_ADC : Prop
axiom angle_ABD_BCD : Prop

theorem find_length_CD_m_plus_n :
  ∃ (m n : ℕ), gcd m n = 1 ∧ (CD = m / n) ∧ (m + n = 67) :=
sorry  -- Proof would be provided here

end find_length_CD_m_plus_n_l2374_237428


namespace pumps_fill_time_l2374_237426

-- Definitions for the rates and the time calculation
def small_pump_rate : ℚ := 1 / 3
def large_pump_rate : ℚ := 4
def third_pump_rate : ℚ := 1 / 2

def total_pump_rate : ℚ := small_pump_rate + large_pump_rate + third_pump_rate

theorem pumps_fill_time :
  1 / total_pump_rate = 6 / 29 :=
by
  -- Definition of the rates has already been given.
  -- Here we specify the calculation for the combined rate and filling time.
  sorry

end pumps_fill_time_l2374_237426


namespace focus_of_parabola_l2374_237479

theorem focus_of_parabola (y : ℝ → ℝ) (h : ∀ x, y x = 16 * x^2) : 
    ∃ p, p = (0, 1/64) := 
by
    existsi (0, 1/64)
    -- The proof would go here, but we are adding sorry to skip it 
    sorry

end focus_of_parabola_l2374_237479


namespace circle_radius_6_l2374_237495

theorem circle_radius_6 (k : ℝ) : 
  (∀ x y : ℝ, x^2 + 10*x + y^2 + 6*y - k = 0 ↔ (x + 5)^2 + (y + 3)^2 = 36) → k = 2 :=
by
  sorry

end circle_radius_6_l2374_237495


namespace minjeong_walk_distance_l2374_237497

noncomputable def park_side_length : ℕ := 40
noncomputable def square_sides : ℕ := 4

theorem minjeong_walk_distance (side_length : ℕ) (sides : ℕ) (h : side_length = park_side_length) (h2 : sides = square_sides) : 
  side_length * sides = 160 := by
  sorry

end minjeong_walk_distance_l2374_237497


namespace cookies_left_l2374_237408

theorem cookies_left (initial_cookies : ℕ) (cookies_eaten : ℕ) (cookies_left : ℕ) :
  initial_cookies = 28 → cookies_eaten = 21 → cookies_left = initial_cookies - cookies_eaten → cookies_left = 7 :=
by
  intros h_initial h_eaten h_left
  rw [h_initial, h_eaten] at h_left
  exact h_left

end cookies_left_l2374_237408


namespace matthew_hotdogs_l2374_237498

-- Definitions based on conditions
def hotdogs_ella_emma : ℕ := 2 + 2
def hotdogs_luke : ℕ := 2 * hotdogs_ella_emma
def hotdogs_hunter : ℕ := (3 * hotdogs_ella_emma) / 2  -- Multiplying by 1.5 

-- Theorem statement to prove the total number of hotdogs
theorem matthew_hotdogs : hotdogs_ella_emma + hotdogs_luke + hotdogs_hunter = 18 := by
  sorry

end matthew_hotdogs_l2374_237498


namespace one_inch_cubes_with_two_or_more_painted_faces_l2374_237494

def original_cube_length : ℕ := 4

def total_one_inch_cubes : ℕ := original_cube_length ^ 3

def corners_count : ℕ := 8

def edges_minus_corners_count : ℕ := 12 * 2

theorem one_inch_cubes_with_two_or_more_painted_faces
  (painted_faces_on_each_face : ∀ i : ℕ, i < total_one_inch_cubes → ℕ) : 
  ∃ n : ℕ, n = corners_count + edges_minus_corners_count ∧ n = 32 := 
by
  simp only [corners_count, edges_minus_corners_count, total_one_inch_cubes]
  sorry

end one_inch_cubes_with_two_or_more_painted_faces_l2374_237494


namespace a_pow_5_mod_11_l2374_237446

theorem a_pow_5_mod_11 (a : ℕ) : (a^5) % 11 = 0 ∨ (a^5) % 11 = 1 ∨ (a^5) % 11 = 10 :=
sorry

end a_pow_5_mod_11_l2374_237446


namespace triangle_isosceles_or_right_l2374_237447

theorem triangle_isosceles_or_right (a b c : ℝ) (h_triangle : a > 0 ∧ b > 0 ∧ c > 0)
  (h_side_constraint : a + b > c ∧ a + c > b ∧ b + c > a)
  (h_condition: a^2 * c^2 - b^2 * c^2 = a^4 - b^4) :
  (a = b) ∨ (a^2 + b^2 = c^2) :=
by {
  sorry
}

end triangle_isosceles_or_right_l2374_237447


namespace inverse_g_neg138_l2374_237463

def g (x : ℝ) : ℝ := 5 * x^3 - 3

theorem inverse_g_neg138 :
  g (-3) = -138 :=
by
  sorry

end inverse_g_neg138_l2374_237463


namespace height_on_hypotenuse_of_right_triangle_l2374_237406

theorem height_on_hypotenuse_of_right_triangle (a b : ℝ) (h_a : a = 2) (h_b : b = 3) :
  ∃ h : ℝ, h = (6 * Real.sqrt 13) / 13 :=
by
  sorry

end height_on_hypotenuse_of_right_triangle_l2374_237406


namespace rectangle_length_width_l2374_237461

-- Given conditions
variables (L W : ℕ)

-- Condition 1: The area of the rectangular field is 300 square meters
def area_condition : Prop := L * W = 300

-- Condition 2: The perimeter of the rectangular field is 70 meters
def perimeter_condition : Prop := 2 * (L + W) = 70

-- Condition 3: One side of the rectangle is 20 meters
def side_condition : Prop := L = 20

-- Conclusion
def length_width_proof : Prop :=
  L = 20 ∧ W = 15

-- The final mathematical proof problem statement
theorem rectangle_length_width (L W : ℕ) 
  (h1 : area_condition L W) 
  (h2 : perimeter_condition L W) 
  (h3 : side_condition L) : 
  length_width_proof L W :=
sorry

end rectangle_length_width_l2374_237461


namespace initial_albums_in_cart_l2374_237480

theorem initial_albums_in_cart (total_songs : ℕ) (songs_per_album : ℕ) (removed_albums : ℕ) 
  (h_total: total_songs = 42) 
  (h_songs_per_album: songs_per_album = 7)
  (h_removed: removed_albums = 2): 
  (total_songs / songs_per_album) + removed_albums = 8 := 
by
  sorry

end initial_albums_in_cart_l2374_237480


namespace fraction_value_l2374_237469

theorem fraction_value (x : ℝ) (h : 1 - 5 / x + 6 / x^3 = 0) : 3 / x = 3 / 2 :=
by
  sorry

end fraction_value_l2374_237469


namespace isosceles_trapezoid_fewest_axes_l2374_237402

def equilateral_triangle_axes : Nat := 3
def isosceles_trapezoid_axes : Nat := 1
def rectangle_axes : Nat := 2
def regular_pentagon_axes : Nat := 5

theorem isosceles_trapezoid_fewest_axes :
  isosceles_trapezoid_axes < equilateral_triangle_axes ∧
  isosceles_trapezoid_axes < rectangle_axes ∧
  isosceles_trapezoid_axes < regular_pentagon_axes :=
by
  sorry

end isosceles_trapezoid_fewest_axes_l2374_237402


namespace find_a_l2374_237476

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^(-x) * (1 - a^x)

theorem find_a (a : ℝ) : 
  (∀ x : ℝ, f a (-x) = -f a x) ∧ a > 0 ∧ a ≠ 1 → a = 4 :=
by
  sorry

end find_a_l2374_237476


namespace quadratic_expression_value_l2374_237422

variables (α β : ℝ)
noncomputable def quadratic_root_sum (α β : ℝ) (h1 : α^2 + 2*α - 1 = 0) (h2 : β^2 + 2*β - 1 = 0) : Prop :=
  α + β = -2

theorem quadratic_expression_value (α β : ℝ) (h1 : α^2 + 2*α - 1 = 0) (h2 : β^2 + 2*β - 1 = 0) (h3 : α + β = -2) :
  α^2 + 3*α + β = -1 :=
sorry

end quadratic_expression_value_l2374_237422


namespace boys_in_biology_is_25_l2374_237453

-- Definition of the total number of students in the Physics class
def physics_class_students : ℕ := 200

-- Definition of the total number of students in the Biology class
def biology_class_students : ℕ := physics_class_students / 2

-- Condition that there are three times as many girls as boys in the Biology class
def girls_boys_ratio : ℕ := 3

-- Calculate the total number of "parts" in the Biology class (3 parts girls + 1 part boys)
def total_parts : ℕ := girls_boys_ratio + 1

-- The number of boys in the Biology class
def boys_in_biology : ℕ := biology_class_students / total_parts

-- The statement to prove the number of boys in the Biology class is 25
theorem boys_in_biology_is_25 : boys_in_biology = 25 := by
  sorry

end boys_in_biology_is_25_l2374_237453


namespace cube_painting_equiv_1260_l2374_237482

def num_distinguishable_paintings_of_cube : Nat :=
  1260

theorem cube_painting_equiv_1260 :
  ∀ (colors : Fin 8 → Color), -- assuming we have a type Color representing colors
    (∀ i j : Fin 6, i ≠ j → colors i ≠ colors j) →  -- each face has a different color
    ∃ f : Cube × Fin 8 → Cube × Fin 8, -- considering symmetry transformations (rotations)
      num_distinguishable_paintings_of_cube = 1260 :=
by
  -- Proof would go here
  sorry

end cube_painting_equiv_1260_l2374_237482


namespace percent_of_x_is_y_l2374_237404

theorem percent_of_x_is_y 
    (x y : ℝ) 
    (h : 0.30 * (x - y) = 0.20 * (x + y)) : 
    y / x = 0.2 :=
  sorry

end percent_of_x_is_y_l2374_237404


namespace ratio_roses_to_lilacs_l2374_237412

theorem ratio_roses_to_lilacs
  (L: ℕ) -- number of lilacs sold
  (G: ℕ) -- number of gardenias sold
  (R: ℕ) -- number of roses sold
  (hL: L = 10) -- defining lilacs sold as 10
  (hG: G = L / 2) -- defining gardenias sold as half the lilacs
  (hTotal: R + L + G = 45) -- defining total flowers sold as 45
  : R / L = 3 :=
by {
  -- The actual proof would go here, but we skip it as per instructions
  sorry
}

end ratio_roses_to_lilacs_l2374_237412


namespace p_sufficient_but_not_necessary_for_q_l2374_237436

-- Definitions corresponding to conditions
def p (x : ℝ) : Prop := x > 1
def q (x : ℝ) : Prop := 1 / x < 1

-- Theorem stating the relationship between p and q
theorem p_sufficient_but_not_necessary_for_q :
  (∀ x : ℝ, p x → q x) ∧ ¬(∀ x : ℝ, q x → p x) := 
by
  sorry

end p_sufficient_but_not_necessary_for_q_l2374_237436


namespace find_x_l2374_237414

variables (a b c : ℝ)

theorem find_x (h : a ≥ 0) (h' : b ≥ 0) (h'' : c ≥ 0) : 
  ∃ x ≥ 0, x = Real.sqrt ((b - c)^2 - a^2) :=
by
  use Real.sqrt ((b - c)^2 - a^2)
  sorry

end find_x_l2374_237414


namespace least_n_froods_l2374_237483

def froods_score (n : ℕ) : ℕ := n * (n + 1) / 2
def eating_score (n : ℕ) : ℕ := n ^ 2

theorem least_n_froods :
    ∃ n : ℕ, 0 < n ∧ (froods_score n > eating_score n) ∧ (∀ m : ℕ, 0 < m ∧ m < n → froods_score m ≤ eating_score m) :=
  sorry

end least_n_froods_l2374_237483


namespace log_ab_eq_l2374_237493

-- Definition and conditions
variables (a b x : ℝ)
variables (ha : 0 < a) (hb : 0 < b) (hx : 0 < x)

-- The theorem to prove
theorem log_ab_eq (a b x : ℝ) (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) :
  Real.log (x) / Real.log (a * b) = (Real.log (x) / Real.log (a)) * (Real.log (x) / Real.log (b)) / ((Real.log (x) / Real.log (a)) + (Real.log (x) / Real.log (b))) :=
sorry

end log_ab_eq_l2374_237493


namespace probability_of_event_B_l2374_237457

def fair_dice := { n : ℕ // 1 ≤ n ∧ n ≤ 8 }

def event_B (x y : fair_dice) : Prop := x.val = y.val + 2

def total_outcomes : ℕ := 64

def favorable_outcomes : ℕ := 6

theorem probability_of_event_B : (favorable_outcomes : ℚ) / total_outcomes = 3/32 := by
  have h1 : (64 : ℚ) = 8 * 8 := by norm_num
  have h2 : (6 : ℚ) / 64 = 3 / 32 := by norm_num
  sorry

end probability_of_event_B_l2374_237457


namespace find_k_l2374_237489

theorem find_k (k : ℝ) (h : ∃ (k : ℝ), 3 = k * (-1) - 2) : k = -5 :=
by
  rcases h with ⟨k, hk⟩
  sorry

end find_k_l2374_237489


namespace sample_size_product_A_l2374_237484

theorem sample_size_product_A 
  (ratio_A : ℕ)
  (ratio_B : ℕ)
  (ratio_C : ℕ)
  (total_ratio : ℕ)
  (sample_size : ℕ) 
  (h_ratio : ratio_A = 2 ∧ ratio_B = 3 ∧ ratio_C = 5)
  (h_total_ratio : total_ratio = ratio_A + ratio_B + ratio_C)
  (h_sample_size : sample_size = 80) :
  (80 * (ratio_A : ℚ) / total_ratio) = 16 :=
by
  sorry

end sample_size_product_A_l2374_237484


namespace computation_correct_l2374_237401

theorem computation_correct : 12 * ((216 / 3) + (36 / 6) + (16 / 8) + 2) = 984 := 
by 
  sorry

end computation_correct_l2374_237401


namespace cottonwood_fiber_scientific_notation_l2374_237449

theorem cottonwood_fiber_scientific_notation :
  0.0000108 = 1.08 * 10^(-5)
:= by
  sorry

end cottonwood_fiber_scientific_notation_l2374_237449


namespace parallelogram_base_length_l2374_237445

theorem parallelogram_base_length
  (height : ℝ) (area : ℝ) (base : ℝ) 
  (h1 : height = 18) 
  (h2 : area = 576) 
  (h3 : area = base * height) : 
  base = 32 :=
by
  rw [h1, h2] at h3
  sorry

end parallelogram_base_length_l2374_237445


namespace k_value_for_z_perfect_square_l2374_237466

theorem k_value_for_z_perfect_square (Z K : ℤ) (h1 : 500 < Z ∧ Z < 1000) (h2 : K > 1) (h3 : Z = K * K^2) :
  ∃ K : ℤ, Z = 729 ∧ K = 9 :=
by {
  sorry
}

end k_value_for_z_perfect_square_l2374_237466


namespace length_of_boat_l2374_237439

-- Define Josie's jogging variables and problem conditions
variables (L J B : ℝ)
axiom eqn1 : 130 * J = L + 130 * B
axiom eqn2 : 70 * J = L - 70 * B

-- The theorem to prove that the length of the boat L equals 91 steps (i.e., 91 * J)
theorem length_of_boat : L = 91 * J :=
by
  sorry

end length_of_boat_l2374_237439


namespace watch_arrangement_count_l2374_237456

noncomputable def number_of_satisfying_watch_arrangements : Nat :=
  let dial_arrangements := Nat.factorial 2
  let strap_arrangements := Nat.factorial 3
  dial_arrangements * strap_arrangements

theorem watch_arrangement_count :
  number_of_satisfying_watch_arrangements = 12 :=
by
-- Proof omitted
sorry

end watch_arrangement_count_l2374_237456


namespace multiple_of_B_share_l2374_237475

theorem multiple_of_B_share (A B C : ℝ) (k : ℝ) 
    (h1 : 3 * A = k * B) 
    (h2 : k * B = 7 * 84) 
    (h3 : C = 84)
    (h4 : A + B + C = 427) :
    k = 4 :=
by
  -- We do not need the detailed proof steps here.
  sorry

end multiple_of_B_share_l2374_237475


namespace solve_rational_numbers_l2374_237417

theorem solve_rational_numbers 
  (a b c d : ℚ)
  (h₁ : a + b + c = -1)
  (h₂ : a + b + d = -3)
  (h₃ : a + c + d = 2)
  (h₄ : b + c + d = 17) :
  a = 6 ∧ b = 8 ∧ c = 3 ∧ d = -12 := 
by
  sorry

end solve_rational_numbers_l2374_237417


namespace total_employees_l2374_237421

-- Definitions based on the conditions:
variables (N S : ℕ)
axiom condition1 : 75 % 100 * S = 75 / 100 * S
axiom condition2 : 65 % 100 * S = 65 / 100 * S
axiom condition3 : N - S = 40
axiom condition4 : 5 % 6 * N = 5 / 6 * N

-- The statement to be proven:
theorem total_employees (N S : ℕ)
    (h1 : 75 % 100 * S = 75 / 100 * S)
    (h2 : 65 % 100 * S = 65 / 100 * S)
    (h3 : N - S = 40)
    (h4 : 5 % 6 * N = 5 / 6 * N)
    : N = 240 :=
sorry

end total_employees_l2374_237421
