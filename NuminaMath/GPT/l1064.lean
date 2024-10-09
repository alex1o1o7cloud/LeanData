import Mathlib

namespace initial_amount_l1064_106477

theorem initial_amount (P : ℝ) :
  (P * 1.0816 - P * 1.08 = 3.0000000000002274) → P = 1875.0000000001421 :=
by
  sorry

end initial_amount_l1064_106477


namespace roots_expression_l1064_106400

theorem roots_expression (p q : ℝ) (hpq : (∀ x, 3*x^2 + 9*x - 21 = 0 → x = p ∨ x = q)) 
  (sum_roots : p + q = -3) 
  (prod_roots : p * q = -7) : (3*p - 4) * (6*q - 8) = 122 :=
by
  sorry

end roots_expression_l1064_106400


namespace Tim_weekly_earnings_l1064_106418

def number_of_tasks_per_day : ℕ := 100
def pay_per_task : ℝ := 1.2
def working_days_per_week : ℕ := 6

theorem Tim_weekly_earnings :
  (number_of_tasks_per_day * pay_per_task) * working_days_per_week = 720 := by
  sorry

end Tim_weekly_earnings_l1064_106418


namespace christopher_strolled_5_miles_l1064_106411

theorem christopher_strolled_5_miles (s t : ℝ) (hs : s = 4) (ht : t = 1.25) : s * t = 5 :=
by
  rw [hs, ht]
  norm_num

end christopher_strolled_5_miles_l1064_106411


namespace fountains_fill_pool_together_l1064_106480

-- Define the times in hours for each fountain to fill the pool
def time_fountain1 : ℚ := 5 / 2  -- 2.5 hours
def time_fountain2 : ℚ := 15 / 4 -- 3.75 hours

-- Define the rates at which each fountain can fill the pool
def rate_fountain1 : ℚ := 1 / time_fountain1
def rate_fountain2 : ℚ := 1 / time_fountain2

-- Calculate the combined rate
def combined_rate : ℚ := rate_fountain1 + rate_fountain2

-- Define the time for both fountains working together to fill the pool
def combined_time : ℚ := 1 / combined_rate

-- Prove that the combined time is indeed 1.5 hours
theorem fountains_fill_pool_together : combined_time = 3 / 2 := by
  sorry

end fountains_fill_pool_together_l1064_106480


namespace distance_A_B_l1064_106415

variable (x : ℚ)

def pointA := x
def pointB := 1
def pointC := -1

theorem distance_A_B : |pointA x - pointB| = |x - 1| := by
  sorry

end distance_A_B_l1064_106415


namespace line_through_circle_center_l1064_106427

theorem line_through_circle_center
  (C : ℝ × ℝ)
  (hC : C = (-1, 0))
  (hCircle : ∀ (x y : ℝ), x^2 + 2 * x + y^2 = 0 → (x, y) = (-1, 0))
  (hPerpendicular : ∀ (m₁ m₂ : ℝ), (m₁ * m₂ = -1) → m₁ = -1 → m₂ = 1)
  (line_eq : ∀ (x y : ℝ), y = x + 1)
  : ∀ (x y : ℝ), x - y + 1 = 0 :=
sorry

end line_through_circle_center_l1064_106427


namespace ratio_of_turtles_l1064_106464

noncomputable def initial_turtles_owen : ℕ := 21
noncomputable def initial_turtles_johanna : ℕ := initial_turtles_owen - 5
noncomputable def turtles_johanna_after_month : ℕ := initial_turtles_johanna / 2
noncomputable def turtles_owen_after_month : ℕ := 50 - turtles_johanna_after_month

theorem ratio_of_turtles (a b : ℕ) (h1 : a = 21) (h2 : b = 5) (h3 : initial_turtles_owen = a) (h4 : initial_turtles_johanna = initial_turtles_owen - b) 
(h5 : turtles_johanna_after_month = initial_turtles_johanna / 2) (h6 : turtles_owen_after_month = 50 - turtles_johanna_after_month) : 
turtles_owen_after_month / initial_turtles_owen = 2 := by
  sorry

end ratio_of_turtles_l1064_106464


namespace difference_between_a_b_l1064_106405

theorem difference_between_a_b (a b : ℝ) (d : ℝ) : 
  (a - b = d) → (a ^ 2 + b ^ 2 = 150) → (a * b = 25) → d = 10 :=
by
  sorry

end difference_between_a_b_l1064_106405


namespace acute_angles_complementary_l1064_106435

-- Given conditions
variables (α β : ℝ)
variables (α_acute : 0 < α ∧ α < π / 2) (β_acute : 0 < β ∧ β < π / 2)
variables (h : (sin α) ^ 2 + (sin β) ^ 2 = sin (α + β))

-- Statement we want to prove
theorem acute_angles_complementary : α + β = π / 2 :=
  sorry

end acute_angles_complementary_l1064_106435


namespace interval_length_of_solutions_l1064_106439

theorem interval_length_of_solutions (a b : ℝ) :
  (∃ x : ℝ, a ≤ 3*x + 6 ∧ 3*x + 6 ≤ b) ∧ (∃ (l : ℝ), l = (b - a) / 3 ∧ l = 15) → b - a = 45 :=
by sorry

end interval_length_of_solutions_l1064_106439


namespace value_of_f_neg_a_l1064_106472

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.sin x + 1

theorem value_of_f_neg_a (a : ℝ) (h : f a = 2) : f (-a) = 0 :=
by
  sorry

end value_of_f_neg_a_l1064_106472


namespace range_of_m_l1064_106438

theorem range_of_m (m : ℝ) : 
  (∀ x y : ℝ, (x - (m^2 - 2 * m + 4) * y - 6 > 0) ↔ (x, y) ≠ (-1, -1)) →
  -1 ≤ m ∧ m ≤ 3 :=
by
  intro h
  sorry

end range_of_m_l1064_106438


namespace kendra_packs_l1064_106492

/-- Kendra has some packs of pens. Tony has 2 packs of pens. There are 3 pens in each pack. 
Kendra and Tony decide to keep two pens each and give the remaining pens to their friends 
one pen per friend. They give pens to 14 friends. Prove that Kendra has 4 packs of pens. --/
theorem kendra_packs : ∀ (kendra_pens tony_pens pens_per_pack pens_kept pens_given friends : ℕ),
  tony_pens = 2 →
  pens_per_pack = 3 →
  pens_kept = 2 →
  pens_given = 14 →
  tony_pens * pens_per_pack - pens_kept + kendra_pens - pens_kept = pens_given →
  kendra_pens / pens_per_pack = 4 :=
by
  intros kendra_pens tony_pens pens_per_pack pens_kept pens_given friends
  intro h1
  intro h2
  intro h3
  intro h4
  intro h5
  sorry

end kendra_packs_l1064_106492


namespace trigonometric_identity_proof_l1064_106401

noncomputable def m : ℝ := 2 * Real.sin (Real.pi / 10)
noncomputable def n : ℝ := 4 - m^2

theorem trigonometric_identity_proof :
  (m = 2 * Real.sin (Real.pi / 10)) →
  (m^2 + n = 4) →
  (m * Real.sqrt n) / (2 * Real.cos (3 * Real.pi / 20)^2 - 1) = 2 :=
by
  intros h1 h2
  sorry

end trigonometric_identity_proof_l1064_106401


namespace realize_ancient_dreams_only_C_l1064_106410

-- Define the available options
inductive Options
| A : Options
| B : Options
| C : Options
| D : Options

-- Define the ancient dreams condition
def realize_ancient_dreams (o : Options) : Prop :=
  o = Options.C

-- The theorem states that only Geographic Information Technology (option C) can realize the ancient dreams
theorem realize_ancient_dreams_only_C :
  realize_ancient_dreams Options.C :=
by
  -- skip the exact proof
  sorry

end realize_ancient_dreams_only_C_l1064_106410


namespace find_a_extreme_value_l1064_106437

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log (x + 1) - x - a * x

theorem find_a_extreme_value :
  (∃ a : ℝ, ∀ x, f x a = Real.log (x + 1) - x - a * x ∧ (∃ m : ℝ, ∀ y : ℝ, f y a ≤ m)) ↔ a = -1 / 2 :=
by
  sorry

end find_a_extreme_value_l1064_106437


namespace greatest_x_for_quadratic_inequality_l1064_106452

theorem greatest_x_for_quadratic_inequality (x : ℝ) (h : x^2 - 12 * x + 35 ≤ 0) : x ≤ 7 :=
sorry

end greatest_x_for_quadratic_inequality_l1064_106452


namespace sum_is_402_3_l1064_106450

def sum_of_numbers := 3 + 33 + 333 + 33.3

theorem sum_is_402_3 : sum_of_numbers = 402.3 := by
  sorry

end sum_is_402_3_l1064_106450


namespace mean_of_remaining_two_numbers_l1064_106475

/-- 
Given seven numbers:
a = 1870, b = 1995, c = 2020, d = 2026, e = 2110, f = 2124, g = 2500
and the condition that the mean of five of these numbers is 2100,
prove that the mean of the remaining two numbers is 2072.5.
-/
theorem mean_of_remaining_two_numbers :
  let a := 1870
  let b := 1995
  let c := 2020
  let d := 2026
  let e := 2110
  let f := 2124
  let g := 2500
  a + b + c + d + e + f + g = 14645 →
  (a + b + c + d + e + f + g) = 14645 →
  (a + b + c + d + e) / 5 = 2100 →
  (f + g) / 2 = 2072.5 :=
by
  let a := 1870
  let b := 1995
  let c := 2020
  let d := 2026
  let e := 2110
  let f := 2124
  let g := 2500
  sorry

end mean_of_remaining_two_numbers_l1064_106475


namespace intersection_complement_l1064_106422

-- Definitions and conditions
def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 2, 3}
def B : Set ℕ := {1, 3, 4}
def C_U (B : Set ℕ) : Set ℕ := {x ∈ U | x ∉ B}

-- Theorem statement
theorem intersection_complement :
  (C_U B) ∩ A = {0, 2} := 
by
  -- Proof is not required, so we use sorry
  sorry

end intersection_complement_l1064_106422


namespace bridge_length_is_115_meters_l1064_106468

noncomputable def length_of_bridge (length_of_train : ℝ) (speed_km_per_hr : ℝ) (time_to_pass : ℝ) : ℝ :=
  let speed_m_per_s := speed_km_per_hr * (1000 / 3600)
  let total_distance := speed_m_per_s * time_to_pass
  total_distance - length_of_train

theorem bridge_length_is_115_meters :
  length_of_bridge 300 35 42.68571428571429 = 115 :=
by
  -- Here the proof has to show the steps for converting speed and calculating distances
  sorry

end bridge_length_is_115_meters_l1064_106468


namespace state_B_more_candidates_l1064_106432

theorem state_B_more_candidates (appeared : ℕ) (selected_A_pct selected_B_pct : ℝ)
  (h1 : appeared = 8000)
  (h2 : selected_A_pct = 0.06)
  (h3 : selected_B_pct = 0.07) :
  (selected_B_pct * appeared - selected_A_pct * appeared = 80) :=
by
  sorry

end state_B_more_candidates_l1064_106432


namespace cos_54_deg_l1064_106446

-- Define cosine function
noncomputable def cos_deg (θ : ℝ) : ℝ := Real.cos (θ * Real.pi / 180)

-- The main theorem statement
theorem cos_54_deg : cos_deg 54 = (-1 + Real.sqrt 5) / 4 :=
  sorry

end cos_54_deg_l1064_106446


namespace arcsin_range_l1064_106460

theorem arcsin_range (α : ℝ ) (x : ℝ ) (h₁ : x = Real.cos α) (h₂ : -Real.pi / 4 ≤ α ∧ α ≤ 3 * Real.pi / 4) : 
-Real.pi / 4 ≤ Real.arcsin x ∧ Real.arcsin x ≤ Real.pi / 2 :=
sorry

end arcsin_range_l1064_106460


namespace solve_equation_l1064_106465

theorem solve_equation (a b : ℕ) : 
  (a^2 = b * (b + 7) ∧ a ≥ 0 ∧ b ≥ 0) ↔ (a = 0 ∧ b = 0) ∨ (a = 12 ∧ b = 9) := by
  sorry

end solve_equation_l1064_106465


namespace ratio_of_building_heights_l1064_106466

theorem ratio_of_building_heights (F_h F_s A_s B_s : ℝ) (hF_h : F_h = 18) (hF_s : F_s = 45)
  (hA_s : A_s = 60) (hB_s : B_s = 72) :
  let h_A := (F_h / F_s) * A_s
  let h_B := (F_h / F_s) * B_s
  (h_A / h_B) = 5 / 6 :=
by
  sorry

end ratio_of_building_heights_l1064_106466


namespace train_speed_l1064_106419

theorem train_speed (train_length bridge_length : ℕ) (time : ℝ)
  (h_train_length : train_length = 110)
  (h_bridge_length : bridge_length = 290)
  (h_time : time = 23.998080153587715) :
  (train_length + bridge_length) / time * 3.6 = 60 := 
by
  rw [h_train_length, h_bridge_length, h_time]
  sorry

end train_speed_l1064_106419


namespace inequality_solution_l1064_106444

theorem inequality_solution (x : ℝ) : 
  3 - 1 / (3 * x + 4) < 5 ↔ x < -4 / 3 ∨ -3 / 2 < x := 
sorry

end inequality_solution_l1064_106444


namespace max_parts_by_rectangles_l1064_106473

theorem max_parts_by_rectangles (n : ℕ) : 
  ∃ S : ℕ, S = 2 * n^2 - 2 * n + 2 :=
by
  sorry

end max_parts_by_rectangles_l1064_106473


namespace math_problem_l1064_106431

theorem math_problem (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x^3 + y^3 = x - y) : x^2 + 4 * y^2 < 1 := 
sorry

end math_problem_l1064_106431


namespace rice_amount_previously_l1064_106474

variables (P X : ℝ) (hP : P > 0) (h : 0.8 * P * 50 = P * X)

theorem rice_amount_previously (hP : P > 0) (h : 0.8 * P * 50 = P * X) : X = 40 := 
by 
  sorry

end rice_amount_previously_l1064_106474


namespace kelsey_total_distance_l1064_106409

-- Define the constants and variables involved
def total_distance (total_time : ℕ) (speed1 speed2 half_dist1 half_dist2 : ℕ) : ℕ :=
  let T1 := half_dist1 / speed1
  let T2 := half_dist2 / speed2
  let T := T1 + T2
  total_time

-- Prove the equivalency given the conditions
theorem kelsey_total_distance (total_time : ℕ) (speed1 speed2 : ℕ) : 
  (total_time = 10) ∧ (speed1 = 25) ∧ (speed2 = 40)  →
  ∃ D, D = 307 ∧ (10 = D / 50 + D / 80) :=
by 
  intro h
  have h_total_time := h.1
  have h_speed1 := h.2.1
  have h_speed2 := h.2.2
  -- Need to prove the statement using provided conditions
  let D := 307
  sorry

end kelsey_total_distance_l1064_106409


namespace some_number_value_l1064_106469

theorem some_number_value (some_number : ℝ) (h : (some_number * 14) / 100 = 0.045388) :
  some_number = 0.3242 :=
sorry

end some_number_value_l1064_106469


namespace find_nat_numbers_for_divisibility_l1064_106488

theorem find_nat_numbers_for_divisibility :
  ∃ (a b : ℕ), (7^3 ∣ a^2 + a * b + b^2) ∧ (¬ 7 ∣ a) ∧ (¬ 7 ∣ b) ∧ (a = 1) ∧ (b = 18) := by
  sorry

end find_nat_numbers_for_divisibility_l1064_106488


namespace not_always_product_greater_l1064_106486

-- Define the premise and the conclusion
theorem not_always_product_greater (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b < 1) : a * b < a :=
sorry

end not_always_product_greater_l1064_106486


namespace find_coordinates_of_Q_l1064_106493

theorem find_coordinates_of_Q (x y : ℝ) (P : ℝ × ℝ) (hP : P = (1, 2))
    (perp : x + 2 * y = 0) (length : x^2 + y^2 = 5) :
    (x, y) = (-2, 1) :=
by
  -- Proof should go here
  sorry

end find_coordinates_of_Q_l1064_106493


namespace time_correct_l1064_106457

theorem time_correct {t : ℝ} (h : 0 < t ∧ t < 60) :
  |6 * (t + 5) - (90 + 0.5 * (t - 4))| = 180 → t = 43 := by
  sorry

end time_correct_l1064_106457


namespace deer_meat_distribution_l1064_106408

theorem deer_meat_distribution :
  ∃ (a1 a2 a3 a4 a5 : ℕ), a1 > a2 ∧ a2 > a3 ∧ a3 > a4 ∧ a4 > a5 ∧
  (a1 + a2 + a3 + a4 + a5 = 500) ∧
  (a2 + a3 + a4 = 300) :=
sorry

end deer_meat_distribution_l1064_106408


namespace crows_and_trees_l1064_106414

theorem crows_and_trees : ∃ (x y : ℕ), 3 * y + 5 = x ∧ 5 * (y - 1) = x ∧ x = 20 ∧ y = 5 :=
by
  sorry

end crows_and_trees_l1064_106414


namespace quadratic_real_roots_iff_l1064_106428

-- Define the statement of the problem in Lean
theorem quadratic_real_roots_iff (m : ℝ) :
  (∃ x : ℂ, m * x^2 + 2 * x - 1 = 0) ↔ (m ≥ -1 ∧ m ≠ 0) := 
by
  sorry

end quadratic_real_roots_iff_l1064_106428


namespace find_a3_minus_b3_l1064_106487

theorem find_a3_minus_b3 (a b : ℝ) (h1 : a - b = 7) (h2 : a^2 + b^2 = 47) : a^3 - b^3 = 322 :=
by
  sorry

end find_a3_minus_b3_l1064_106487


namespace train_length_l1064_106445

noncomputable def speed_kph := 56  -- speed in km/hr
def time_crossing := 9  -- time in seconds
noncomputable def speed_mps := speed_kph * 1000 / 3600  -- converting km/hr to m/s

theorem train_length : speed_mps * time_crossing = 140 := by
  -- conversion and result approximation
  sorry

end train_length_l1064_106445


namespace post_height_l1064_106496

theorem post_height 
  (circumference : ℕ) 
  (rise_per_circuit : ℕ) 
  (travel_distance : ℕ)
  (circuits : ℕ := travel_distance / circumference) 
  (total_rise : ℕ := circuits * rise_per_circuit) 
  (c : circumference = 3)
  (r : rise_per_circuit = 4)
  (t : travel_distance = 9) :
  total_rise = 12 := by
  sorry

end post_height_l1064_106496


namespace b_is_some_even_number_l1064_106489

noncomputable def factorable_b (b : ℤ) : Prop :=
  ∃ (m n p q : ℤ), 
    (m * p = 15 ∧ n * q = 15) ∧ 
    (b = m * q + n * p)

theorem b_is_some_even_number (b : ℤ) 
  (h : factorable_b b) : ∃ k : ℤ, b = 2 * k := 
by
  sorry

end b_is_some_even_number_l1064_106489


namespace tailor_trim_amount_l1064_106407

variable (x : ℝ)

def original_side : ℝ := 22
def trimmed_side : ℝ := original_side - x
def fixed_trimmed_side : ℝ := original_side - 5
def remaining_area : ℝ := 120

theorem tailor_trim_amount :
  (original_side - x) * 17 = remaining_area → x = 15 :=
by
  intro h
  sorry

end tailor_trim_amount_l1064_106407


namespace gcd_779_209_589_l1064_106416

theorem gcd_779_209_589 : Int.gcd (Int.gcd 779 209) 589 = 19 := 
by 
  sorry

end gcd_779_209_589_l1064_106416


namespace left_handed_jazz_lovers_count_l1064_106461

noncomputable def club_members := 30
noncomputable def left_handed := 11
noncomputable def like_jazz := 20
noncomputable def right_handed_dislike_jazz := 4

theorem left_handed_jazz_lovers_count : 
  ∃ x, x + (left_handed - x) + (like_jazz - x) + right_handed_dislike_jazz = club_members ∧ x = 5 :=
by
  sorry

end left_handed_jazz_lovers_count_l1064_106461


namespace mod_inverse_5_221_l1064_106420

theorem mod_inverse_5_221 : ∃ x : ℤ, 0 ≤ x ∧ x < 221 ∧ (5 * x) % 221 = 1 % 221 :=
by
  use 177
  sorry

end mod_inverse_5_221_l1064_106420


namespace product_of_averages_is_125000_l1064_106423

-- Define the problem from step a
def sum_1_to_99 : ℕ := (99 * (1 + 99)) / 2
def average_of_group (x : ℕ) : Prop := 3 * 33 * x = sum_1_to_99

-- Define the goal to prove
theorem product_of_averages_is_125000 (x : ℕ) (h : average_of_group x) : x^3 = 125000 :=
by
  sorry

end product_of_averages_is_125000_l1064_106423


namespace quadratic_intersects_x_axis_only_once_l1064_106436

theorem quadratic_intersects_x_axis_only_once (a : ℝ) :
  (∀ x : ℝ, (a * x^2 - a * x + 3 * x + 1 = 0) → a = 1 ∨ a = 9) :=
sorry

end quadratic_intersects_x_axis_only_once_l1064_106436


namespace Marty_combination_count_l1064_106471

theorem Marty_combination_count :
  let num_colors := 4
  let num_methods := 3
  num_colors * num_methods = 12 :=
by
  let num_colors := 4
  let num_methods := 3
  sorry

end Marty_combination_count_l1064_106471


namespace incorrect_statement_l1064_106402

theorem incorrect_statement : 
  ¬(∀ (p q : Prop), (¬p ∧ ¬q) → (¬p ∧ ¬q)) := 
    sorry

end incorrect_statement_l1064_106402


namespace binary_arithmetic_l1064_106462

theorem binary_arithmetic :
  let a := 0b11101
  let b := 0b10011
  let c := 0b101
  (a * b) / c = 0b11101100 :=
by
  sorry

end binary_arithmetic_l1064_106462


namespace minimum_value_of_function_l1064_106490

theorem minimum_value_of_function (x : ℝ) (h : x > 1) : 
  (x + (1 / x) + (16 * x) / (x^2 + 1)) ≥ 8 :=
sorry

end minimum_value_of_function_l1064_106490


namespace complement_of_union_l1064_106449

def U : Set ℝ := Set.univ
def A : Set ℝ := { x | (x - 2) * (x + 1) ≤ 0 }
def B : Set ℝ := { x | 0 ≤ x ∧ x < 3 }

theorem complement_of_union :
  Set.compl (A ∪ B) = { x : ℝ | x < -1 } ∪ { x | x ≥ 3 } := by
  sorry

end complement_of_union_l1064_106449


namespace simple_random_sampling_correct_statements_l1064_106447

theorem simple_random_sampling_correct_statements :
  let N : ℕ := 10
  -- Conditions for simple random sampling
  let is_finite (N : ℕ) := N > 0
  let is_non_sequential (N : ℕ) := N > 0 -- represents sampling does not require sequential order
  let without_replacement := true
  let equal_probability := true
  -- Verification
  (is_finite N) ∧ 
  (¬ is_non_sequential N) ∧ 
  without_replacement ∧ 
  equal_probability = true :=
by
  sorry

end simple_random_sampling_correct_statements_l1064_106447


namespace sum_of_numbers_l1064_106412

theorem sum_of_numbers (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) 
  (h1 : x^2 + y^2 + z^2 = 52) (h2 : x * y + y * z + z * x = 24) : 
  x + y + z = 10 :=
by
  sorry

end sum_of_numbers_l1064_106412


namespace sum_of_squares_of_four_consecutive_even_numbers_eq_344_l1064_106433

theorem sum_of_squares_of_four_consecutive_even_numbers_eq_344 (n : ℤ) 
  (h : n + (n + 2) + (n + 4) + (n + 6) = 36) : 
  n^2 + (n + 2)^2 + (n + 4)^2 + (n + 6)^2 = 344 :=
by sorry

end sum_of_squares_of_four_consecutive_even_numbers_eq_344_l1064_106433


namespace sufficient_not_necessary_condition_l1064_106448

variable (x y : ℝ)

theorem sufficient_not_necessary_condition (h : x + y ≤ 1) : x ≤ 1/2 ∨ y ≤ 1/2 := 
  sorry

end sufficient_not_necessary_condition_l1064_106448


namespace primes_between_2_and_100_l1064_106443

open Nat

theorem primes_between_2_and_100 :
  { p : ℕ | 2 ≤ p ∧ p ≤ 100 ∧ Nat.Prime p } = 
  {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97} :=
by
  sorry

end primes_between_2_and_100_l1064_106443


namespace find_number_l1064_106406

theorem find_number (X : ℝ) (h : 0.8 * X = 0.7 * 60.00000000000001 + 30) : X = 90.00000000000001 :=
sorry

end find_number_l1064_106406


namespace smallest_value_N_l1064_106494

theorem smallest_value_N (l m n N : ℕ) (h1 : (l - 1) * (m - 1) * (n - 1) = 143) (h2 : N = l * m * n) :
  N = 336 :=
sorry

end smallest_value_N_l1064_106494


namespace min_packs_for_126_cans_l1064_106463

-- Definition of pack sizes
def pack_sizes : List ℕ := [15, 18, 36]

-- The given total cans of soda
def total_cans : ℕ := 126

-- The minimum number of packs needed to buy exactly 126 cans of soda
def min_packs_needed (total : ℕ) (packs : List ℕ) : ℕ :=
  -- Function definition to calculate the minimum packs needed
  -- This function needs to be implemented or proven
  sorry

-- The proof that the minimum number of packs needed to buy exactly 126 cans of soda is 4
theorem min_packs_for_126_cans : min_packs_needed total_cans pack_sizes = 4 :=
  -- Proof goes here
  sorry

end min_packs_for_126_cans_l1064_106463


namespace sum_of_possible_values_for_a_l1064_106442

-- Define the conditions
variables (a b c d : ℤ)
variables (h1 : a > b) (h2 : b > c) (h3 : c > d)
variables (h4 : a + b + c + d = 52)
variables (differences : finset ℤ)

-- Hypotheses about the pairwise differences
variable (h_diff : differences = {2, 3, 5, 6, 8, 11})
variable (h_ad : a - d = 11)

-- The pairs of differences adding up to 11
variable (h_pairs1 : a - b + b - d = 11)
variable (h_pairs2 : a - c + c - d = 11)

-- The theorem to be proved
theorem sum_of_possible_values_for_a : a = 19 :=
by
-- Implemented variables and conditions correctly, and the proof is outlined.
sorry

end sum_of_possible_values_for_a_l1064_106442


namespace point_P_quadrant_l1064_106440

theorem point_P_quadrant 
  (h1 : Real.sin (θ / 2) = 3 / 5) 
  (h2 : Real.cos (θ / 2) = -4 / 5) : 
  (0 < Real.cos θ) ∧ (Real.sin θ < 0) :=
by
  sorry

end point_P_quadrant_l1064_106440


namespace min_value_frac_sum_l1064_106483

theorem min_value_frac_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 1) : 
  (1 / x) + (4 / y) ≥ 9 :=
sorry

end min_value_frac_sum_l1064_106483


namespace factor_expression_l1064_106453

theorem factor_expression (x : ℝ) : 5 * x * (x - 2) + 9 * (x - 2) = (x - 2) * (5 * x + 9) := 
by 
sorry

end factor_expression_l1064_106453


namespace triangle_perimeter_sqrt_l1064_106470

theorem triangle_perimeter_sqrt :
  let a := Real.sqrt 8
  let b := Real.sqrt 18
  let c := Real.sqrt 32
  a + b + c = 9 * Real.sqrt 2 :=
by
  sorry

end triangle_perimeter_sqrt_l1064_106470


namespace both_not_divisible_by_7_l1064_106413

theorem both_not_divisible_by_7 {a b : ℝ} (h : ¬ (∃ k : ℤ, ab = 7 * k)) : ¬ (∃ m : ℤ, a = 7 * m) ∧ ¬ (∃ n : ℤ, b = 7 * n) :=
sorry

end both_not_divisible_by_7_l1064_106413


namespace B_k_largest_at_45_l1064_106485

def B_k (k : ℕ) : ℝ := (Nat.choose 500 k) * (0.1)^k

theorem B_k_largest_at_45 : ∀ k : ℕ, k = 45 → ∀ m : ℕ, m ≠ 45 → B_k 45 > B_k m :=
by
  intro k h_k m h_m
  sorry

end B_k_largest_at_45_l1064_106485


namespace right_angled_trapezoid_base_height_l1064_106425

theorem right_angled_trapezoid_base_height {a b : ℝ} (h : a = b) :
  ∃ (base height : ℝ), base = a ∧ height = b := 
by
  sorry

end right_angled_trapezoid_base_height_l1064_106425


namespace arithmetic_sequence_l1064_106459

noncomputable def a_n (a1 d : ℝ) (n : ℕ) : ℝ := a1 + (n - 1) * d

theorem arithmetic_sequence (a1 d : ℝ) (h_d : d ≠ 0) 
  (h1 : a1 + (a1 + 2 * d) = 8) 
  (h2 : (a1 + d) * (a1 + 8 * d) = (a1 + 3 * d) * (a1 + 3 * d)) :
  a_n a1 d 5 = 13 := 
by 
  sorry

end arithmetic_sequence_l1064_106459


namespace negation_of_proposition_l1064_106484

-- Conditions
variable {x : ℝ}

-- The proposition
def proposition : Prop := ∃ x : ℝ, Real.exp x > x

-- The proof problem: proving the negation of the proposition
theorem negation_of_proposition : (¬ proposition) ↔ ∀ x : ℝ, Real.exp x ≤ x := by
  sorry

end negation_of_proposition_l1064_106484


namespace find_additional_fuel_per_person_l1064_106421

def num_passengers : ℕ := 30
def num_crew : ℕ := 5
def num_people : ℕ := num_passengers + num_crew
def num_bags_per_person : ℕ := 2
def num_bags : ℕ := num_people * num_bags_per_person
def fuel_empty_plane : ℕ := 20
def fuel_per_bag : ℕ := 2
def total_trip_fuel : ℕ := 106000
def trip_distance : ℕ := 400
def fuel_per_mile : ℕ := total_trip_fuel / trip_distance

def additional_fuel_per_person (x : ℕ) : Prop :=
  fuel_empty_plane + num_people * x + num_bags * fuel_per_bag = fuel_per_mile

theorem find_additional_fuel_per_person : additional_fuel_per_person 3 :=
  sorry

end find_additional_fuel_per_person_l1064_106421


namespace grape_juice_problem_l1064_106404

noncomputable def grape_juice_amount (initial_mixture_volume : ℕ) (initial_concentration : ℝ) (final_concentration : ℝ) : ℝ :=
  let initial_grape_juice := initial_mixture_volume * initial_concentration
  let total_volume := initial_mixture_volume + final_concentration * (final_concentration - initial_grape_juice) / (1 - final_concentration) -- Total volume after adding x gallons
  let added_grape_juice := total_volume - initial_mixture_volume -- x gallons added
  added_grape_juice

theorem grape_juice_problem :
  grape_juice_amount 40 0.20 0.36 = 10 := 
by
  sorry

end grape_juice_problem_l1064_106404


namespace sum_fractions_eq_l1064_106497

theorem sum_fractions_eq (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end sum_fractions_eq_l1064_106497


namespace f_correct_l1064_106458

noncomputable def f : ℕ → ℝ
| 0       => 0 -- undefined for 0, start from 1
| (n + 1) => if n = 0 then 1/2 else sorry -- recursion undefined for now

theorem f_correct : ∀ n ≥ 1, f n = (3^(n-1) / (3^(n-1) + 1)) :=
by
  -- Initial conditions
  have h0 : f 1 = 1/2 := sorry
  -- Recurrence relations
  have h1 : ∀ n, n ≥ 1 → f (n + 1) ≥ (3 * f n) / (2 * f n + 1) := sorry
  -- Prove the function form
  sorry

end f_correct_l1064_106458


namespace find_t_l1064_106481

-- Define the logarithm base 3 function
noncomputable def log_base_3 (x : ℝ) : ℝ := Real.log x / Real.log 3

-- Given Condition
def condition (t : ℝ) : Prop := 4 * log_base_3 t = log_base_3 (4 * t) + 2

-- Theorem stating if the given condition holds, then t must be 6
theorem find_t (t : ℝ) (ht : condition t) : t = 6 := 
by
  sorry

end find_t_l1064_106481


namespace only_n_equal_1_is_natural_number_for_which_2_pow_n_plus_n_pow_2016_is_prime_l1064_106499

theorem only_n_equal_1_is_natural_number_for_which_2_pow_n_plus_n_pow_2016_is_prime (n : ℕ) : 
  Prime (2^n + n^2016) ↔ n = 1 := by
  sorry

end only_n_equal_1_is_natural_number_for_which_2_pow_n_plus_n_pow_2016_is_prime_l1064_106499


namespace expected_total_cost_of_removing_blocks_l1064_106456

/-- 
  There are six blocks in a row labeled 1 through 6, each with weight 1.
  Two blocks x ≤ y are connected if for all x ≤ z ≤ y, block z has not been removed.
  While there is at least one block remaining, a block is chosen uniformly at random and removed.
  The cost of removing a block is the sum of the weights of the blocks that are connected to it.
  Prove that the expected total cost of removing all blocks is 163 / 10.
-/
theorem expected_total_cost_of_removing_blocks : (6:ℚ) + 5 + 8/3 + 3/2 + 4/5 + 1/3 = 163 / 10 := sorry

end expected_total_cost_of_removing_blocks_l1064_106456


namespace initial_milk_water_ratio_l1064_106430

theorem initial_milk_water_ratio
  (M W : ℕ)
  (h1 : M + W = 40000)
  (h2 : (M : ℚ) / (W + 1600) = 3 / 1) :
  (M : ℚ) / W = 3.55 :=
by
  sorry

end initial_milk_water_ratio_l1064_106430


namespace count_triangles_l1064_106498

-- Assuming the conditions are already defined and given as parameters  
-- Let's define a proposition to prove the solution

noncomputable def total_triangles_in_figure : ℕ := 68

-- Create the theorem statement:
theorem count_triangles : total_triangles_in_figure = 68 := 
by
  sorry

end count_triangles_l1064_106498


namespace scientific_notation_120_million_l1064_106495

theorem scientific_notation_120_million :
  120000000 = 1.2 * 10^7 :=
by
  sorry

end scientific_notation_120_million_l1064_106495


namespace minimum_P_ge_37_l1064_106426

noncomputable def minimum_P (x y z : ℝ) : ℝ := 
  (x / y + y / z + z / x) * (y / x + z / y + x / z)

theorem minimum_P_ge_37 (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h : (x / y + y / z + z / x) + (y / x + z / y + x / z) = 10) : 
  minimum_P x y z ≥ 37 :=
sorry

end minimum_P_ge_37_l1064_106426


namespace upstream_distance_l1064_106424

theorem upstream_distance (v : ℝ) 
  (H1 : ∀ d : ℝ, (10 + v) * 2 = 28) 
  (H2 : (10 - v) * 2 = d) : d = 12 := by
  sorry

end upstream_distance_l1064_106424


namespace chess_or_basketball_students_l1064_106482

-- Definitions based on the conditions
def percentage_likes_basketball : ℝ := 0.4
def percentage_likes_chess : ℝ := 0.1
def total_students : ℕ := 250

-- Main statement to prove
theorem chess_or_basketball_students : 
  (percentage_likes_basketball + percentage_likes_chess) * total_students = 125 :=
by
  sorry

end chess_or_basketball_students_l1064_106482


namespace find_k_l1064_106429

noncomputable def series (k : ℝ) : ℝ := ∑' n, (7 * n - 2) / k^n

theorem find_k (k : ℝ) (h₁ : 1 < k) (h₂ : series k = 17 / 2) : k = 17 / 7 :=
by
  sorry

end find_k_l1064_106429


namespace sum_squares_l1064_106455

theorem sum_squares {a b c : ℝ} (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a + b + c = 0) 
  (h5 : a^4 + b^4 + c^4 = a^6 + b^6 + c^6) : 
  a^2 + b^2 + c^2 = 6 / 5 := 
by sorry

end sum_squares_l1064_106455


namespace smallest_positive_integer_x_l1064_106454

theorem smallest_positive_integer_x :
  ∃ x : ℕ, 42 * x + 14 ≡ 4 [MOD 26] ∧ x ≡ 3 [MOD 5] ∧ x = 38 := 
by
  sorry

end smallest_positive_integer_x_l1064_106454


namespace find_number_of_packs_l1064_106476

-- Define the cost of a pack of Digimon cards
def cost_pack_digimon : ℝ := 4.45

-- Define the cost of the deck of baseball cards
def cost_deck_baseball : ℝ := 6.06

-- Define the total amount spent
def total_spent : ℝ := 23.86

-- Define the number of packs of Digimon cards Keith bought
def number_of_packs (D : ℝ) : Prop :=
  cost_pack_digimon * D + cost_deck_baseball = total_spent

-- Prove the number of packs is 4
theorem find_number_of_packs : ∃ D, number_of_packs D ∧ D = 4 :=
by
  -- the proof will be inserted here
  sorry

end find_number_of_packs_l1064_106476


namespace arithmetic_mean_18_27_45_l1064_106491

theorem arithmetic_mean_18_27_45 : 
  (18 + 27 + 45) / 3 = 30 :=
by
  -- skipping proof
  sorry

end arithmetic_mean_18_27_45_l1064_106491


namespace simplification_problem_l1064_106403

theorem simplification_problem :
  (3^2015 - 3^2013 + 3^2011) / (3^2015 + 3^2013 - 3^2011) = 73 / 89 :=
  sorry

end simplification_problem_l1064_106403


namespace sum_of_areas_l1064_106467

def base_width : ℕ := 3
def lengths : List ℕ := [1, 8, 27, 64, 125, 216]
def area (w l : ℕ) : ℕ := w * l
def total_area : ℕ := (lengths.map (area base_width)).sum

theorem sum_of_areas : total_area = 1323 := 
by sorry

end sum_of_areas_l1064_106467


namespace abs_eq_of_unique_solution_l1064_106478

theorem abs_eq_of_unique_solution (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
    (unique_solution : ∃! x : ℝ, a * (x - a) ^ 2 + b * (x - b) ^ 2 = 0) :
    |a| = |b| :=
sorry

end abs_eq_of_unique_solution_l1064_106478


namespace tangent_line_parallel_to_given_line_l1064_106417

theorem tangent_line_parallel_to_given_line 
  (x : ℝ) (y : ℝ) (tangent_line : ℝ → ℝ) :
  (tangent_line y = x^2 - 1) → 
  (tangent_line = 4) → 
  (4 * x - y - 5 = 0) :=
by 
  sorry

end tangent_line_parallel_to_given_line_l1064_106417


namespace find_angle_A_l1064_106441

variable (a b c : ℝ)
variable (A : ℝ)

axiom triangle_ABC : a = Real.sqrt 3 ∧ b = 1 ∧ c = 2

theorem find_angle_A : a = Real.sqrt 3 ∧ b = 1 ∧ c = 2 → A = Real.pi / 3 :=
by
  intro h
  sorry

end find_angle_A_l1064_106441


namespace rectangular_field_area_l1064_106434

-- Given a rectangle with one side 4 meters and diagonal 5 meters, prove that its area is 12 square meters.
theorem rectangular_field_area
  (w l d : ℝ)
  (h_w : w = 4)
  (h_d : d = 5)
  (h_pythagoras : w^2 + l^2 = d^2) :
  w * l = 12 := 
by
  sorry

end rectangular_field_area_l1064_106434


namespace percentage_correct_l1064_106479

theorem percentage_correct (x : ℕ) (h : x > 0) : 
  (4 * x / (6 * x) * 100 = 200 / 3) :=
by
  sorry

end percentage_correct_l1064_106479


namespace books_initially_l1064_106451

theorem books_initially (A B : ℕ) (h1 : A = 3) (h2 : B = (A + 2) + 2) : B = 7 :=
by
  -- Using the given facts, we need to show B = 7
  sorry

end books_initially_l1064_106451
