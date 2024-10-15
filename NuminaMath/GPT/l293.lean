import Mathlib

namespace NUMINAMATH_GPT_probability_of_at_least_one_l293_29362

theorem probability_of_at_least_one (P_1 P_2 : ℝ) (h1 : 0 ≤ P_1 ∧ P_1 ≤ 1) (h2 : 0 ≤ P_2 ∧ P_2 ≤ 1) :
  1 - (1 - P_1) * (1 - P_2) = P_1 + P_2 - P_1 * P_2 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_at_least_one_l293_29362


namespace NUMINAMATH_GPT_ratio_of_DN_NF_l293_29352

theorem ratio_of_DN_NF (D E F N : Type) (DE EF DF DN NF p q: ℕ) (h1 : DE = 18) (h2 : EF = 28) (h3 : DF = 34) 
(h4 : DN + NF = DF) (h5 : DN = 22) (h6 : NF = 11) (h7 : p = 101) (h8 : q = 50) : p + q = 151 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_of_DN_NF_l293_29352


namespace NUMINAMATH_GPT_regular_polygon_sides_l293_29365

theorem regular_polygon_sides (n : ℕ) : (360 : ℝ) / n = 18 → n = 20 :=
by
  intros h
  -- Start the proof here
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l293_29365


namespace NUMINAMATH_GPT_outerCircumference_is_correct_l293_29334

noncomputable def π : ℝ := Real.pi  
noncomputable def innerCircumference : ℝ := 352 / 7
noncomputable def width : ℝ := 4.001609997739084

noncomputable def radius_inner : ℝ := innerCircumference / (2 * π)
noncomputable def radius_outer : ℝ := radius_inner + width
noncomputable def outerCircumference : ℝ := 2 * π * radius_outer

theorem outerCircumference_is_correct : outerCircumference = 341.194 := by
  sorry

end NUMINAMATH_GPT_outerCircumference_is_correct_l293_29334


namespace NUMINAMATH_GPT_weight_of_each_dumbbell_l293_29317

-- Definitions based on conditions
def initial_dumbbells : Nat := 4
def added_dumbbells : Nat := 2
def total_dumbbells : Nat := initial_dumbbells + added_dumbbells -- 6
def total_weight : Nat := 120

-- Theorem statement
theorem weight_of_each_dumbbell (h : total_dumbbells = 6) (w : total_weight = 120) :
  total_weight / total_dumbbells = 20 :=
by
  -- Proof is to be written here
  sorry

end NUMINAMATH_GPT_weight_of_each_dumbbell_l293_29317


namespace NUMINAMATH_GPT_trig_identity_proof_l293_29345

noncomputable def cos_30 := Real.cos (Real.pi / 6)
noncomputable def sin_60 := Real.sin (Real.pi / 3)
noncomputable def sin_30 := Real.sin (Real.pi / 6)
noncomputable def cos_60 := Real.cos (Real.pi / 3)

theorem trig_identity_proof :
  (1 - (1 / cos_30)) * (1 + (2 / sin_60)) * (1 - (1 / sin_30)) * (1 + (2 / cos_60)) = (25 - 10 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_GPT_trig_identity_proof_l293_29345


namespace NUMINAMATH_GPT_original_price_l293_29305

-- Definitions based on the conditions
def selling_price : ℝ := 1080
def gain_percent : ℝ := 80

-- The proof problem: Prove that the cost price is Rs. 600
theorem original_price (CP : ℝ) (h_sp : CP + CP * (gain_percent / 100) = selling_price) : CP = 600 :=
by
  -- We skip the proof itself
  sorry

end NUMINAMATH_GPT_original_price_l293_29305


namespace NUMINAMATH_GPT_trajectory_of_M_l293_29360

-- Define the conditions: P moves on the circle, and Q is fixed
variable (P Q M : ℝ × ℝ)
variable (P_moves_on_circle : P.1^2 + P.2^2 = 1)
variable (Q_fixed : Q = (3, 0))
variable (M_midpoint : M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2))

-- Theorem statement
theorem trajectory_of_M :
  (2 * M.1 - 3)^2 + 4 * M.2^2 = 1 :=
sorry

end NUMINAMATH_GPT_trajectory_of_M_l293_29360


namespace NUMINAMATH_GPT_fred_grew_38_cantelopes_l293_29387

def total_cantelopes : Nat := 82
def tim_cantelopes : Nat := 44
def fred_cantelopes : Nat := total_cantelopes - tim_cantelopes

theorem fred_grew_38_cantelopes : fred_cantelopes = 38 :=
by
  sorry

end NUMINAMATH_GPT_fred_grew_38_cantelopes_l293_29387


namespace NUMINAMATH_GPT_add_salt_solution_l293_29332

theorem add_salt_solution
  (initial_amount : ℕ) (added_concentration : ℕ) (desired_concentration : ℕ)
  (initial_concentration : ℝ) :
  initial_amount = 50 ∧ initial_concentration = 0.4 ∧ added_concentration = 10 ∧ desired_concentration = 25 →
  (∃ (x : ℕ), x = 50 ∧ 
    (initial_concentration * initial_amount + 0.1 * x) / (initial_amount + x) = 0.25) :=
by
  sorry

end NUMINAMATH_GPT_add_salt_solution_l293_29332


namespace NUMINAMATH_GPT_find_principal_amount_l293_29399

theorem find_principal_amount
  (r : ℝ := 0.05)  -- Interest rate (5% per annum)
  (t : ℕ := 2)    -- Time period (2 years)
  (diff : ℝ := 20) -- Given difference between CI and SI
  (P : ℝ := 8000) -- Principal amount to prove
  : P * (1 + r) ^ t - P - P * r * t = diff :=
by
  sorry

end NUMINAMATH_GPT_find_principal_amount_l293_29399


namespace NUMINAMATH_GPT_minimum_value_of_ratio_l293_29358

theorem minimum_value_of_ratio 
  {a b c : ℝ} (h_a : a ≠ 0) 
  (h_f'0 : 2 * a * 0 + b > 0)
  (h_f_nonneg : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0) :
  (∃ x : ℝ, a * x^2 + b * x + c ≥ 0) ∧ (1 + (a + c) / b = 2) := sorry

end NUMINAMATH_GPT_minimum_value_of_ratio_l293_29358


namespace NUMINAMATH_GPT_proof_of_intersection_l293_29372

open Set

theorem proof_of_intersection :
  let U := ℝ
  let M := compl { x : ℝ | x^2 > 4 }
  let N := { x : ℝ | 1 < x ∧ x ≤ 3 }
  M ∩ N = { x | 1 < x ∧ x ≤ 2 } := by
sorry

end NUMINAMATH_GPT_proof_of_intersection_l293_29372


namespace NUMINAMATH_GPT_bus_ride_cost_l293_29300

theorem bus_ride_cost (B T : ℝ) (h1 : T = B + 6.85) (h2 : T + B = 9.65) : B = 1.40 :=
sorry

end NUMINAMATH_GPT_bus_ride_cost_l293_29300


namespace NUMINAMATH_GPT_find_number_l293_29397

-- Define the main condition and theorem.
theorem find_number (x : ℤ) : 45 - (x - (37 - (15 - 19))) = 58 ↔ x = 28 :=
by
  sorry  -- placeholder for the proof

end NUMINAMATH_GPT_find_number_l293_29397


namespace NUMINAMATH_GPT_rectangle_area_l293_29307

theorem rectangle_area (x : ℝ) (h : (x - 3) * (2 * x + 3) = 4 * x - 9) : x = 7 / 2 :=
sorry

end NUMINAMATH_GPT_rectangle_area_l293_29307


namespace NUMINAMATH_GPT_dvd_count_correct_l293_29375

def total_dvds (store_dvds online_dvds : Nat) : Nat :=
  store_dvds + online_dvds

theorem dvd_count_correct :
  total_dvds 8 2 = 10 :=
by
  sorry

end NUMINAMATH_GPT_dvd_count_correct_l293_29375


namespace NUMINAMATH_GPT_find_numbers_l293_29396

theorem find_numbers (x y : ℕ) (h1 : 100 ≤ x ∧ x ≤ 999) (h2 : 100 ≤ y ∧ y ≤ 999) (h3 : 1000 * x + y = 7 * x * y) :
  x = 143 ∧ y = 143 :=
by
  sorry

end NUMINAMATH_GPT_find_numbers_l293_29396


namespace NUMINAMATH_GPT_simplify_nested_fraction_l293_29325

theorem simplify_nested_fraction :
  (1 : ℚ) / (1 + (1 / (3 + (1 / 4)))) = 13 / 17 :=
by
  sorry

end NUMINAMATH_GPT_simplify_nested_fraction_l293_29325


namespace NUMINAMATH_GPT_numberOfWaysToChoose4Cards_l293_29339

-- Define the total number of ways to choose 4 cards of different suits from a standard deck.
def waysToChoose4Cards : ℕ := 13^4

-- Prove that the calculated number of ways is equal to 28561
theorem numberOfWaysToChoose4Cards : waysToChoose4Cards = 28561 :=
by
  sorry

end NUMINAMATH_GPT_numberOfWaysToChoose4Cards_l293_29339


namespace NUMINAMATH_GPT_num_positive_divisors_of_720_multiples_of_5_l293_29366

theorem num_positive_divisors_of_720_multiples_of_5 :
  (∃ (a b c : ℕ), 0 ≤ a ∧ a ≤ 4 ∧ 0 ≤ b ∧ b ≤ 2 ∧ c = 1) →
  ∃ (n : ℕ), n = 15 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_num_positive_divisors_of_720_multiples_of_5_l293_29366


namespace NUMINAMATH_GPT_gcd_mn_mn_squared_l293_29357

theorem gcd_mn_mn_squared (m n : ℕ) (h : Nat.gcd m n = 1) : ({d : ℕ | d = Nat.gcd (m + n) (m ^ 2 + n ^ 2)} ⊆ {1, 2}) := 
sorry

end NUMINAMATH_GPT_gcd_mn_mn_squared_l293_29357


namespace NUMINAMATH_GPT_next_in_step_distance_l293_29348

theorem next_in_step_distance
  (jack_stride jill_stride : ℕ)
  (h1 : jack_stride = 64)
  (h2 : jill_stride = 56) :
  Nat.lcm jack_stride jill_stride = 448 := by
  sorry

end NUMINAMATH_GPT_next_in_step_distance_l293_29348


namespace NUMINAMATH_GPT_map_length_representation_l293_29301

theorem map_length_representation (a b : ℕ) (h : a = 15 ∧ b = 90) : b * (20 / a) = 120 :=
by
  sorry

end NUMINAMATH_GPT_map_length_representation_l293_29301


namespace NUMINAMATH_GPT_bob_paid_24_percent_of_SRP_l293_29353

theorem bob_paid_24_percent_of_SRP
  (P : ℝ) -- Suggested Retail Price (SRP)
  (MP : ℝ) -- Marked Price (MP)
  (price_bob_paid : ℝ) -- Price Bob Paid
  (h1 : MP = 0.60 * P) -- Condition 1: MP is 60% of SRP
  (h2 : price_bob_paid = 0.40 * MP) -- Condition 2: Bob paid 40% of the MP
  : (price_bob_paid / P) * 100 = 24 := -- Bob paid 24% of the SRP
by
  sorry

end NUMINAMATH_GPT_bob_paid_24_percent_of_SRP_l293_29353


namespace NUMINAMATH_GPT_ratio_cereal_A_to_B_l293_29315

-- Definitions translated from conditions
def sugar_percentage_A : ℕ := 10
def sugar_percentage_B : ℕ := 2
def desired_sugar_percentage : ℕ := 6

-- The theorem based on the question and correct answer
theorem ratio_cereal_A_to_B :
  let difference_A := sugar_percentage_A - desired_sugar_percentage
  let difference_B := desired_sugar_percentage - sugar_percentage_B
  difference_A = 4 ∧ difference_B = 4 → 
  difference_B / difference_A = 1 :=
by
  intros
  sorry

end NUMINAMATH_GPT_ratio_cereal_A_to_B_l293_29315


namespace NUMINAMATH_GPT_calculate_expression_l293_29320

theorem calculate_expression : 15 * 35 + 45 * 15 = 1200 :=
by {
  -- hint to the Lean prover to consider associative property
  sorry
}

end NUMINAMATH_GPT_calculate_expression_l293_29320


namespace NUMINAMATH_GPT_job_completion_l293_29361

theorem job_completion (x y z : ℝ) 
  (h1 : 1/x + 1/y = 1/2) 
  (h2 : 1/y + 1/z = 1/4) 
  (h3 : 1/z + 1/x = 1/2.4) 
  (h4 : 1/x + 1/y + 1/z = 7/12) : 
  x = 3 := 
sorry

end NUMINAMATH_GPT_job_completion_l293_29361


namespace NUMINAMATH_GPT_mice_meet_after_three_days_l293_29303

theorem mice_meet_after_three_days 
  (thickness : ℕ) 
  (first_day_distance : ℕ) 
  (big_mouse_double_progress : ℕ → ℕ) 
  (small_mouse_half_remain_distance : ℕ → ℕ) 
  (days : ℕ) 
  (big_mouse_distance : ℚ) : 
  thickness = 5 ∧ 
  first_day_distance = 1 ∧ 
  (∀ n, big_mouse_double_progress n = 2 ^ (n - 1)) ∧ 
  (∀ n, small_mouse_half_remain_distance n = 5 - (5 / 2 ^ (n - 1))) ∧ 
  days = 3 → 
  big_mouse_distance = 3 + 8 / 17 := 
by
  sorry

end NUMINAMATH_GPT_mice_meet_after_three_days_l293_29303


namespace NUMINAMATH_GPT_div_five_times_eight_by_ten_l293_29316

theorem div_five_times_eight_by_ten : (5 * 8) / 10 = 4 := by
  sorry

end NUMINAMATH_GPT_div_five_times_eight_by_ten_l293_29316


namespace NUMINAMATH_GPT_more_radishes_correct_l293_29354

def total_radishes : ℕ := 88
def radishes_first_basket : ℕ := 37

def more_radishes_in_second_basket := total_radishes - radishes_first_basket - radishes_first_basket

theorem more_radishes_correct : more_radishes_in_second_basket = 14 :=
by
  sorry

end NUMINAMATH_GPT_more_radishes_correct_l293_29354


namespace NUMINAMATH_GPT_mixed_number_subtraction_l293_29380

theorem mixed_number_subtraction :
  2 + 5 / 6 - (1 + 1 / 3) = 3 / 2 := by
sorry

end NUMINAMATH_GPT_mixed_number_subtraction_l293_29380


namespace NUMINAMATH_GPT_quadratic_real_roots_l293_29359

theorem quadratic_real_roots (k : ℝ) :
  (∃ x : ℝ, k * x^2 + 2 * x + 1 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_real_roots_l293_29359


namespace NUMINAMATH_GPT_red_gumballs_count_l293_29385

def gumballs_problem (R B G : ℕ) : Prop :=
  B = R / 2 ∧
  G = 4 * B ∧
  R + B + G = 56

theorem red_gumballs_count (R B G : ℕ) (h : gumballs_problem R B G) : R = 16 :=
by
  rcases h with ⟨h1, h2, h3⟩
  sorry

end NUMINAMATH_GPT_red_gumballs_count_l293_29385


namespace NUMINAMATH_GPT_festival_second_day_attendance_l293_29389

-- Define the conditions
variables (X Y Z A : ℝ)
variables (h1 : A = 2700) (h2 : Y = X / 2) (h3 : Z = 3 * X) (h4 : A = X + Y + Z)

-- Theorem stating the question and the conditions result in the correct answer
theorem festival_second_day_attendance (X Y Z A : ℝ) 
  (h1 : A = 2700) (h2 : Y = X / 2) (h3 : Z = 3 * X) (h4 : A = X + Y + Z) : 
  Y = 300 :=
sorry

end NUMINAMATH_GPT_festival_second_day_attendance_l293_29389


namespace NUMINAMATH_GPT_total_flowers_l293_29363

def pieces (f : String) : Nat :=
  if f == "roses" ∨ f == "lilies" ∨ f == "sunflowers" ∨ f == "daisies" then 40 else 0

theorem total_flowers : 
  pieces "roses" + pieces "lilies" + pieces "sunflowers" + pieces "daisies" = 160 := 
by
  sorry


end NUMINAMATH_GPT_total_flowers_l293_29363


namespace NUMINAMATH_GPT_average_children_per_grade_average_girls_per_grade_average_boys_per_grade_average_club_members_per_grade_l293_29337

theorem average_children_per_grade (G3_girls G3_boys G3_club : ℕ) 
                                  (G4_girls G4_boys G4_club : ℕ) 
                                  (G5_girls G5_boys G5_club : ℕ) 
                                  (H1 : G3_girls = 28) 
                                  (H2 : G3_boys = 35) 
                                  (H3 : G3_club = 12) 
                                  (H4 : G4_girls = 45) 
                                  (H5 : G4_boys = 42) 
                                  (H6 : G4_club = 15) 
                                  (H7 : G5_girls = 38) 
                                  (H8 : G5_boys = 51) 
                                  (H9 : G5_club = 10) :
   (63 + 87 + 89) / 3 = 79.67 :=
by sorry

theorem average_girls_per_grade (G3_girls G4_girls G5_girls : ℕ) 
                                (H1 : G3_girls = 28) 
                                (H2 : G4_girls = 45) 
                                (H3 : G5_girls = 38) :
   (28 + 45 + 38) / 3 = 37 :=
by sorry

theorem average_boys_per_grade (G3_boys G4_boys G5_boys : ℕ)
                               (H1 : G3_boys = 35) 
                               (H2 : G4_boys = 42) 
                               (H3 : G5_boys = 51) :
   (35 + 42 + 51) / 3 = 42.67 :=
by sorry

theorem average_club_members_per_grade (G3_club G4_club G5_club : ℕ) 
                                       (H1 : G3_club = 12)
                                       (H2 : G4_club = 15)
                                       (H3 : G5_club = 10) :
   (12 + 15 + 10) / 3 = 12.33 :=
by sorry

end NUMINAMATH_GPT_average_children_per_grade_average_girls_per_grade_average_boys_per_grade_average_club_members_per_grade_l293_29337


namespace NUMINAMATH_GPT_rowing_speed_downstream_correct_l293_29304

/-- Given:
- The speed of the man upstream V_upstream is 20 kmph.
- The speed of the man in still water V_man is 40 kmph.
Prove:
- The speed of the man rowing downstream V_downstream is 60 kmph.
-/
def rowing_speed_downstream : Prop :=
  let V_upstream := 20
  let V_man := 40
  let V_s := V_man - V_upstream
  let V_downstream := V_man + V_s
  V_downstream = 60

theorem rowing_speed_downstream_correct : rowing_speed_downstream := by
  sorry

end NUMINAMATH_GPT_rowing_speed_downstream_correct_l293_29304


namespace NUMINAMATH_GPT_minimum_value_of_reciprocal_product_l293_29310

theorem minimum_value_of_reciprocal_product (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + a * b + 2 * b = 30) : 
  ∃ m : ℝ, m = 1 / (a * b) ∧ m = 1 / 18 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_reciprocal_product_l293_29310


namespace NUMINAMATH_GPT_perfect_square_trinomial_m_l293_29368

theorem perfect_square_trinomial_m (m : ℝ) :
  (∀ x : ℝ, ∃ b : ℝ, x^2 + 2 * (m - 3) * x + 16 = (1 * x + b)^2) → (m = 7 ∨ m = -1) :=
by 
  intro h
  sorry

end NUMINAMATH_GPT_perfect_square_trinomial_m_l293_29368


namespace NUMINAMATH_GPT_shaded_square_ratio_l293_29371

theorem shaded_square_ratio (side_length : ℝ) (H : side_length = 5) :
  let large_square_area := side_length ^ 2
  let shaded_square_area := (side_length / 2) ^ 2
  shaded_square_area / large_square_area = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_shaded_square_ratio_l293_29371


namespace NUMINAMATH_GPT_alex_buys_15_pounds_of_corn_l293_29384

theorem alex_buys_15_pounds_of_corn:
  ∃ (c b : ℝ), c + b = 30 ∧ 1.20 * c + 0.60 * b = 27.00 ∧ c = 15.0 :=
by
  sorry

end NUMINAMATH_GPT_alex_buys_15_pounds_of_corn_l293_29384


namespace NUMINAMATH_GPT_games_lost_l293_29392

theorem games_lost (total_games won_games : ℕ) (h_total : total_games = 12) (h_won : won_games = 8) :
  (total_games - won_games) = 4 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_games_lost_l293_29392


namespace NUMINAMATH_GPT_find_w_l293_29347

theorem find_w {w : ℝ} : (3, w^3) ∈ {p : ℝ × ℝ | ∃ x, p = (x, x^2 - 1)} → w = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_w_l293_29347


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_range_m_l293_29338

namespace problem

variable (m x y : ℝ)

/-- Propositions for m -/
def P := (1 < m ∧ m < 4) 
def Q := (2 < m ∧ m < 3) ∨ (3 < m ∧  m < 4)

/-- Statements that P => Q is necessary but not sufficient -/
theorem necessary_but_not_sufficient (hP : 1 < m ∧ m < 4) : 
  ((m-1) * (m-4) < 0) ∧ (Q m) :=
by 
  sorry

theorem range_m (h1 : ¬ (P m ∧ Q m)) (h2 : P m ∨ Q m) : 
  1 < m ∧ m ≤ 2 ∨ m = 3 :=
by
  sorry

end problem

end NUMINAMATH_GPT_necessary_but_not_sufficient_range_m_l293_29338


namespace NUMINAMATH_GPT_kishore_savings_l293_29324

noncomputable def total_expenses : ℝ :=
  5000 + 1500 + 4500 + 2500 + 2000 + 5200

def percentage_saved : ℝ := 0.10

theorem kishore_savings (salary : ℝ) :
  (total_expenses + percentage_saved * salary) = salary → 
  (percentage_saved * salary = 2077.78) :=
by
  intros h
  rw [← h]
  sorry

end NUMINAMATH_GPT_kishore_savings_l293_29324


namespace NUMINAMATH_GPT_solve_equation_l293_29333

theorem solve_equation (x : ℝ) :
    x^6 - 22 * x^2 - Real.sqrt 21 = 0 ↔ x = Real.sqrt ((Real.sqrt 21 + 5) / 2) ∨ x = -Real.sqrt ((Real.sqrt 21 + 5) / 2) := by
  sorry

end NUMINAMATH_GPT_solve_equation_l293_29333


namespace NUMINAMATH_GPT_units_digit_of_exponentiated_product_l293_29388

theorem units_digit_of_exponentiated_product :
  (2 ^ 2101 * 5 ^ 2102 * 11 ^ 2103) % 10 = 0 := 
sorry

end NUMINAMATH_GPT_units_digit_of_exponentiated_product_l293_29388


namespace NUMINAMATH_GPT_yellow_highlighters_l293_29306

def highlighters (pink blue yellow total : Nat) : Prop :=
  (pink + blue + yellow = total)

theorem yellow_highlighters (h : highlighters 3 5 y 15) : y = 7 :=
by 
  sorry

end NUMINAMATH_GPT_yellow_highlighters_l293_29306


namespace NUMINAMATH_GPT_emily_initial_cards_l293_29344

theorem emily_initial_cards (x : ℤ) (h1 : x + 7 = 70) : x = 63 :=
by
  sorry

end NUMINAMATH_GPT_emily_initial_cards_l293_29344


namespace NUMINAMATH_GPT_minimum_value_of_expression_l293_29335

theorem minimum_value_of_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) : 
  ∃ m : ℝ, (m = 8) ∧ (∀ z : ℝ, z = (y / x) + (4 / y) → z ≥ m) :=
sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l293_29335


namespace NUMINAMATH_GPT_range_of_a_l293_29314

noncomputable def p (a : ℝ) : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a > 0

noncomputable def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

theorem range_of_a (a : ℝ) (h1 : p a) (h2 : q a) : a ≤ -2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l293_29314


namespace NUMINAMATH_GPT_twelve_edge_cubes_painted_faces_l293_29391

theorem twelve_edge_cubes_painted_faces :
  let painted_faces_per_edge_cube := 2
  let num_edge_cubes := 12
  painted_faces_per_edge_cube * num_edge_cubes = 24 :=
by
  sorry

end NUMINAMATH_GPT_twelve_edge_cubes_painted_faces_l293_29391


namespace NUMINAMATH_GPT_one_and_one_third_of_x_is_36_l293_29383

theorem one_and_one_third_of_x_is_36 (x : ℝ) (h : (4 / 3) * x = 36) : x = 27 := 
sorry

end NUMINAMATH_GPT_one_and_one_third_of_x_is_36_l293_29383


namespace NUMINAMATH_GPT_line_representation_l293_29394

variable {R : Type*} [Field R]
variable (f : R → R → R)
variable (x0 y0 : R)

def not_on_line (P : R × R) (f : R → R → R) : Prop :=
  f P.1 P.2 ≠ 0

theorem line_representation (P : R × R) (hP : not_on_line P f) :
  ∃ l : R → R → Prop, (∀ x y, l x y ↔ f x y - f P.1 P.2 = 0) ∧ (l P.1 P.2) ∧ 
  ∀ x y, f x y = 0 → ∃ n : R, ∀ x1 y1, (l x1 y1 → f x1 y1 = n * (f x y)) :=
sorry

end NUMINAMATH_GPT_line_representation_l293_29394


namespace NUMINAMATH_GPT_find_smallest_number_l293_29327

theorem find_smallest_number (a b c : ℕ) 
  (h1 : a ≤ b) (h2 : b ≤ c)
  (h3 : b = 31)
  (h4 : c = b + 6)
  (h5 : (a + b + c) / 3 = 30) :
  a = 22 := 
sorry

end NUMINAMATH_GPT_find_smallest_number_l293_29327


namespace NUMINAMATH_GPT_find_number_l293_29351

theorem find_number : ∃ n : ℕ, n = (15 * 6) + 5 := 
by sorry

end NUMINAMATH_GPT_find_number_l293_29351


namespace NUMINAMATH_GPT_original_rectangle_area_l293_29355

theorem original_rectangle_area : 
  ∃ (a b : ℤ), (a + b = 20) ∧ (a * b = 96) := by
  sorry

end NUMINAMATH_GPT_original_rectangle_area_l293_29355


namespace NUMINAMATH_GPT_jason_retirement_age_l293_29364

def age_at_retirement (initial_age years_to_chief extra_years_ratio years_after_masterchief : ℕ) : ℕ :=
  initial_age + years_to_chief + (years_to_chief * extra_years_ratio / 100) + years_after_masterchief

theorem jason_retirement_age :
  age_at_retirement 18 8 25 10 = 46 :=
by
  sorry

end NUMINAMATH_GPT_jason_retirement_age_l293_29364


namespace NUMINAMATH_GPT_count_valid_numbers_l293_29386

def is_valid_number (n : ℕ) : Prop :=
  n < 100000 ∧
  ∃ (d₁ d₂ : ℕ), d₁ < 10 ∧ d₂ < 10 ∧ d₁ ≠ d₂ ∧
    (∀ k, (n / 10^k % 10 = d₁ ∨ n / 10^k % 10 = d₂))

theorem count_valid_numbers : 
  ∃ (k : ℕ), k = 2151 ∧ (∀ n, is_valid_number n → n < 100000 → n ≤ k) :=
by
  sorry

end NUMINAMATH_GPT_count_valid_numbers_l293_29386


namespace NUMINAMATH_GPT_total_packs_sold_l293_29350

theorem total_packs_sold (lucy_packs : ℕ) (robyn_packs : ℕ) (h1 : lucy_packs = 19) (h2 : robyn_packs = 16) : lucy_packs + robyn_packs = 35 :=
by
  sorry

end NUMINAMATH_GPT_total_packs_sold_l293_29350


namespace NUMINAMATH_GPT_cubic_root_identity_l293_29309

theorem cubic_root_identity (a b c : ℝ) 
  (h1 : a + b + c = 0) 
  (h2 : a * b + a * c + b * c = -3)
  (h3 : a * b * c = -2) : 
  a * (b + c) ^ 2 + b * (c + a) ^ 2 + c * (a + b) ^ 2 = -6 := 
by
  sorry

end NUMINAMATH_GPT_cubic_root_identity_l293_29309


namespace NUMINAMATH_GPT_third_side_length_of_triangle_l293_29393

theorem third_side_length_of_triangle {a b c : ℝ} (h1 : a^2 - 7 * a + 12 = 0) (h2 : b^2 - 7 * b + 12 = 0) 
  (h3 : a ≠ b) (h4 : a = 3 ∨ a = 4) (h5 : b = 3 ∨ b = 4) : 
  (c = 5 ∨ c = Real.sqrt 7) := by
  sorry

end NUMINAMATH_GPT_third_side_length_of_triangle_l293_29393


namespace NUMINAMATH_GPT_function_symmetric_and_monotonic_l293_29395

noncomputable def f (x : ℝ) : ℝ := (Real.cos x)^4 - 2 * Real.sin x * Real.cos x - (Real.sin x)^4

theorem function_symmetric_and_monotonic :
  (∀ x, f (x + (3/8) * π) = f (x - (3/8) * π)) ∧
  (∀ x y, x ∈  Set.Icc (-(π / 8)) ((3 * π) / 8) → y ∈  Set.Icc (-(π / 8)) ((3 * π) / 8) → x < y → f x > f y) :=
by
  sorry

end NUMINAMATH_GPT_function_symmetric_and_monotonic_l293_29395


namespace NUMINAMATH_GPT_integer_solution_unique_l293_29367

theorem integer_solution_unique (w x y z : ℤ) :
  w^2 + 11 * x^2 - 8 * y^2 - 12 * y * z - 10 * z^2 = 0 →
  w = 0 ∧ x = 0 ∧ y = 0 ∧ z = 0 :=
by
  sorry
 
end NUMINAMATH_GPT_integer_solution_unique_l293_29367


namespace NUMINAMATH_GPT_quadratic_solution_l293_29326

theorem quadratic_solution (a b : ℚ) (h : a * 1^2 + b * 1 + 1 = 0) : 3 - a - b = 4 := 
by
  sorry

end NUMINAMATH_GPT_quadratic_solution_l293_29326


namespace NUMINAMATH_GPT_mother_gave_80_cents_l293_29329

theorem mother_gave_80_cents (father_uncles_gift : Nat) (spent_on_candy current_amount : Nat) (gift_from_father gift_from_uncle add_gift_from_uncle : Nat) (x : Nat) :
  father_uncles_gift = gift_from_father + gift_from_uncle ∧
  father_uncles_gift = 110 ∧
  spent_on_candy = 50 ∧
  current_amount = 140 ∧
  gift_from_father = 40 ∧
  gift_from_uncle = 70 ∧
  add_gift_from_uncle = 70 ∧
  x = current_amount + spent_on_candy - father_uncles_gift ∧
  x = 190 - 110 ∨
  x = 80 :=
  sorry

end NUMINAMATH_GPT_mother_gave_80_cents_l293_29329


namespace NUMINAMATH_GPT_greater_solution_of_quadratic_eq_l293_29322

theorem greater_solution_of_quadratic_eq (x : ℝ) : 
  (∀ y : ℝ, y^2 + 20 * y - 96 = 0 → (y = 4)) :=
sorry

end NUMINAMATH_GPT_greater_solution_of_quadratic_eq_l293_29322


namespace NUMINAMATH_GPT_difference_in_areas_l293_29374

def S1 (x y : ℝ) : Prop :=
  Real.log (3 + x ^ 2 + y ^ 2) / Real.log 2 ≤ 2 + Real.log (x + y) / Real.log 2

def S2 (x y : ℝ) : Prop :=
  Real.log (3 + x ^ 2 + y ^ 2) / Real.log 2 ≤ 3 + Real.log (x + y) / Real.log 2

theorem difference_in_areas : 
  let area_S1 := π * 1 ^ 2
  let area_S2 := π * (Real.sqrt 13) ^ 2
  area_S2 - area_S1 = 12 * π :=
by
  sorry

end NUMINAMATH_GPT_difference_in_areas_l293_29374


namespace NUMINAMATH_GPT_handshake_count_l293_29342

def gathering_handshakes (total_people : ℕ) (know_each_other : ℕ) (know_no_one : ℕ) : ℕ :=
  let group2_handshakes := know_no_one * (total_people - 1)
  group2_handshakes / 2

theorem handshake_count :
  gathering_handshakes 30 20 10 = 145 :=
by
  sorry

end NUMINAMATH_GPT_handshake_count_l293_29342


namespace NUMINAMATH_GPT_sin_cos_sum_inequality_l293_29381

theorem sin_cos_sum_inequality (A B C : ℝ) (h : A + B + C = Real.pi) :
  (Real.sin A + Real.sin B + Real.sin C) / (Real.cos A + Real.cos B + Real.cos C) < 2 := 
sorry

end NUMINAMATH_GPT_sin_cos_sum_inequality_l293_29381


namespace NUMINAMATH_GPT_unique_bounded_sequence_exists_l293_29377

variable (a : ℝ) (n : ℕ) (hn_pos : n > 0)

theorem unique_bounded_sequence_exists :
  ∃! (x : ℕ → ℝ), (x 0 = 0) ∧ (x (n+1) = 0) ∧
                   (∀ i, 1 ≤ i ∧ i ≤ n → (1/2) * (x (i+1) + x (i-1)) = x i + x i ^ 3 - a ^ 3) ∧
                   (∀ i, i ≤ n + 1 → |x i| ≤ |a|) := by
  sorry

end NUMINAMATH_GPT_unique_bounded_sequence_exists_l293_29377


namespace NUMINAMATH_GPT_combined_degrees_l293_29373

theorem combined_degrees (summer_degrees jolly_degrees : ℕ) (h1 : summer_degrees = jolly_degrees + 5) (h2 : summer_degrees = 150) : summer_degrees + jolly_degrees = 295 := 
by
  sorry

end NUMINAMATH_GPT_combined_degrees_l293_29373


namespace NUMINAMATH_GPT_rate_of_interest_is_8_l293_29330

def principal_B : ℕ := 5000
def time_B : ℕ := 2
def principal_C : ℕ := 3000
def time_C : ℕ := 4
def total_interest : ℕ := 1760

theorem rate_of_interest_is_8 :
  ∃ (R : ℝ), ((principal_B * R * time_B) / 100 + (principal_C * R * time_C) / 100 = total_interest) → R = 8 := 
by
  sorry

end NUMINAMATH_GPT_rate_of_interest_is_8_l293_29330


namespace NUMINAMATH_GPT_aira_rubber_bands_l293_29382

theorem aira_rubber_bands (total_bands : ℕ) (bands_each : ℕ) (samantha_extra : ℕ) (aira_fewer : ℕ)
  (h1 : total_bands = 18) 
  (h2 : bands_each = 6) 
  (h3 : samantha_extra = 5) 
  (h4 : aira_fewer = 1) : 
  ∃ x : ℕ, x + (x + samantha_extra) + (x + aira_fewer) = total_bands ∧ x = 4 :=
by
  sorry

end NUMINAMATH_GPT_aira_rubber_bands_l293_29382


namespace NUMINAMATH_GPT_find_number_l293_29323

theorem find_number (x : ℤ) (h : 22 * (x - 36) = 748) : x = 70 :=
sorry

end NUMINAMATH_GPT_find_number_l293_29323


namespace NUMINAMATH_GPT_probability_of_color_difference_l293_29379

noncomputable def probability_of_different_colors (n m : ℕ) : ℚ :=
  (Nat.choose n m : ℚ) * (1/2)^n

theorem probability_of_color_difference :
  probability_of_different_colors 8 4 = 35/128 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_color_difference_l293_29379


namespace NUMINAMATH_GPT_Romeo_bars_of_chocolate_l293_29343

theorem Romeo_bars_of_chocolate 
  (cost_per_bar : ℕ) (packaging_cost : ℕ) (total_sale : ℕ) (profit : ℕ) (x : ℕ) :
  cost_per_bar = 5 →
  packaging_cost = 2 →
  total_sale = 90 →
  profit = 55 →
  (total_sale - (cost_per_bar + packaging_cost) * x = profit) →
  x = 5 :=
by
  sorry

end NUMINAMATH_GPT_Romeo_bars_of_chocolate_l293_29343


namespace NUMINAMATH_GPT_people_in_first_group_l293_29331

-- Conditions
variables (P W : ℕ) (people_work_rate same_work_rate : ℕ)

-- Given conditions as Lean definitions
-- P people can do 3W in 3 days implies the work rate of the group is W per day
def first_group_work_rate : ℕ := 3 * W / 3

-- 9 people can do 9W in 3 days implies the work rate of these 9 people is 3W per day
def second_group_work_rate : ℕ := 9 * W / 3

-- The work rates are proportional to the number of people
def proportional_work_rate : Prop := P / 9 = first_group_work_rate / second_group_work_rate

-- Lean theorem statement for proof
theorem people_in_first_group (h1 : first_group_work_rate = W) (h2 : second_group_work_rate = 3 * W) :
  P = 3 :=
by
  sorry

end NUMINAMATH_GPT_people_in_first_group_l293_29331


namespace NUMINAMATH_GPT_range_of_eccentricity_l293_29370

theorem range_of_eccentricity
  (a b c : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : c^2 = a^2 - b^2)
  (h4 : c^2 - b^2 + a * c < 0) :
  0 < c / a ∧ c / a < 1 / 2 :=
sorry

end NUMINAMATH_GPT_range_of_eccentricity_l293_29370


namespace NUMINAMATH_GPT_alice_expected_games_l293_29313

-- Defining the initial conditions
def skill_levels := Fin 21

def initial_active_player := 0

-- Defining Alice's skill level
def Alice_skill_level := 11

-- Define the tournament structure and conditions
def tournament_round (active: skill_levels) (inactive: Set skill_levels) : skill_levels :=
  sorry

-- Define the expected number of games Alice plays
noncomputable def expected_games_Alice_plays : ℚ :=
  sorry

-- Statement of the problem proving the expected number of games Alice plays
theorem alice_expected_games : expected_games_Alice_plays = 47 / 42 :=
sorry

end NUMINAMATH_GPT_alice_expected_games_l293_29313


namespace NUMINAMATH_GPT_brad_trips_to_fill_barrel_l293_29356

noncomputable def bucket_volume (r : ℝ) : ℝ :=
  (2 / 3) * Real.pi * r^3

noncomputable def barrel_volume (r h : ℝ) : ℝ :=
  Real.pi * r^2 * h

theorem brad_trips_to_fill_barrel :
  let r_bucket := 8  -- radius of the hemisphere bucket in inches
  let r_barrel := 8  -- radius of the cylindrical barrel in inches
  let h_barrel := 20 -- height of the cylindrical barrel in inches
  let V_bucket := bucket_volume r_bucket
  let V_barrel := barrel_volume r_barrel h_barrel
  (Nat.ceil (V_barrel / V_bucket) = 4) :=
by
  sorry

end NUMINAMATH_GPT_brad_trips_to_fill_barrel_l293_29356


namespace NUMINAMATH_GPT_acute_angles_theorem_l293_29308

open Real

variable (α β : ℝ)

-- Given conditions
def conditions : Prop :=
  0 < α ∧ α < π / 2 ∧
  0 < β ∧ β < π / 2 ∧
  tan α = 1 / 7 ∧
  sin β = sqrt 10 / 10

-- Proof goal
def proof_goal : Prop :=
  α + 2 * β = π / 4

-- The final theorem
theorem acute_angles_theorem (h : conditions α β) : proof_goal α β :=
  sorry

end NUMINAMATH_GPT_acute_angles_theorem_l293_29308


namespace NUMINAMATH_GPT_largest_divisor_of_expression_l293_29319

theorem largest_divisor_of_expression (n : ℤ) : 6 ∣ (n^4 - n) := 
sorry

end NUMINAMATH_GPT_largest_divisor_of_expression_l293_29319


namespace NUMINAMATH_GPT_min_average_annual_growth_rate_l293_29376

theorem min_average_annual_growth_rate (M : ℝ) (x : ℝ) (h : M * (1 + x)^2 = 2 * M) : x = Real.sqrt 2 - 1 :=
by
  sorry

end NUMINAMATH_GPT_min_average_annual_growth_rate_l293_29376


namespace NUMINAMATH_GPT_triangle_angle_not_greater_than_60_l293_29321

theorem triangle_angle_not_greater_than_60 (A B C : Real) (h1 : A + B + C = 180) 
  : A ≤ 60 ∨ B ≤ 60 ∨ C ≤ 60 :=
by {
  sorry
}

end NUMINAMATH_GPT_triangle_angle_not_greater_than_60_l293_29321


namespace NUMINAMATH_GPT_radius_of_inscribed_circle_l293_29390

variable (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] (triangle : Triangle A B C)

-- Given conditions
def AC : ℝ := 24
def BC : ℝ := 10
def AB : ℝ := 26

-- Statement to be proved
theorem radius_of_inscribed_circle (hAC : triangle.side_length A C = AC)
                                   (hBC : triangle.side_length B C = BC)
                                   (hAB : triangle.side_length A B = AB) :
  triangle.incircle_radius = 4 :=
by sorry

end NUMINAMATH_GPT_radius_of_inscribed_circle_l293_29390


namespace NUMINAMATH_GPT_max_abs_value_inequality_l293_29346

theorem max_abs_value_inequality (a b : ℝ)
  (h : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |a * x + b| ≤ 1) :
  ∃ (a b : ℝ), (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |a * x + b| ≤ 1) ∧ |20 * a + 14 * b| + |20 * a - 14 * b| = 80 := 
sorry

end NUMINAMATH_GPT_max_abs_value_inequality_l293_29346


namespace NUMINAMATH_GPT_ratio_solves_for_x_l293_29311

theorem ratio_solves_for_x (x : ℝ) (h : 0.60 / x = 6 / 4) : x = 0.4 :=
by
  -- The formal proof would go here.
  sorry

end NUMINAMATH_GPT_ratio_solves_for_x_l293_29311


namespace NUMINAMATH_GPT_Jerry_paid_more_last_month_l293_29349

def Debt_total : ℕ := 50
def Debt_remaining : ℕ := 23
def Paid_2_months_ago : ℕ := 12
def Paid_last_month : ℕ := 27 - Paid_2_months_ago

theorem Jerry_paid_more_last_month :
  Paid_last_month - Paid_2_months_ago = 3 :=
by
  -- Calculation for Paid_last_month
  have h : Paid_last_month = 27 - 12 := by rfl
  -- Compute the difference
  have diff : 15 - 12 = 3 := by rfl
  exact diff

end NUMINAMATH_GPT_Jerry_paid_more_last_month_l293_29349


namespace NUMINAMATH_GPT_find_a6_plus_a7_plus_a8_l293_29341

noncomputable def geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ (n : ℕ), a n = a 0 * r ^ n

theorem find_a6_plus_a7_plus_a8 
  (a : ℕ → ℝ) (r : ℝ)
  (h_geom_seq : geometric_sequence_sum a r)
  (h_sum_1 : a 0 + a 1 + a 2 = 1)
  (h_sum_2 : a 1 + a 2 + a 3 = 2) :
  a 5 + a 6 + a 7 = 32 :=
sorry

end NUMINAMATH_GPT_find_a6_plus_a7_plus_a8_l293_29341


namespace NUMINAMATH_GPT_John_walked_miles_to_park_l293_29336

theorem John_walked_miles_to_park :
  ∀ (total_skateboarded_miles skateboarded_first_leg skateboarded_return_leg walked_miles : ℕ),
    total_skateboarded_miles = 24 →
    skateboarded_first_leg = 10 →
    skateboarded_return_leg = 10 →
    total_skateboarded_miles = skateboarded_first_leg + skateboarded_return_leg + walked_miles →
    walked_miles = 4 :=
by
  intros total_skateboarded_miles skateboarded_first_leg skateboarded_return_leg walked_miles
  intro h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_John_walked_miles_to_park_l293_29336


namespace NUMINAMATH_GPT_student_contribution_is_4_l293_29302

-- Definitions based on the conditions in the problem statement
def total_contribution := 90
def available_class_funds := 14
def number_of_students := 19

-- The theorem statement to be proven
theorem student_contribution_is_4 : 
  (total_contribution - available_class_funds) / number_of_students = 4 :=
by
  sorry  -- Proof is not required as per the instructions

end NUMINAMATH_GPT_student_contribution_is_4_l293_29302


namespace NUMINAMATH_GPT_calculate_difference_of_squares_l293_29340

theorem calculate_difference_of_squares : (640^2 - 360^2) = 280000 := by
  sorry

end NUMINAMATH_GPT_calculate_difference_of_squares_l293_29340


namespace NUMINAMATH_GPT_remainder_eval_at_4_l293_29398

def p : ℚ → ℚ := sorry

def r (x : ℚ) : ℚ := sorry

theorem remainder_eval_at_4 :
  (p 1 = 2) →
  (p 3 = 5) →
  (p (-2) = -2) →
  (∀ x, ∃ q : ℚ → ℚ, p x = (x - 1) * (x - 3) * (x + 2) * q x + r x) →
  r 4 = 38 / 7 :=
sorry

end NUMINAMATH_GPT_remainder_eval_at_4_l293_29398


namespace NUMINAMATH_GPT_find_theta_ratio_l293_29378

def det2x2 (a b c d : ℝ) : ℝ := a * d - b * c

theorem find_theta_ratio (θ : ℝ) 
  (h : det2x2 (Real.sin θ) 2 (Real.cos θ) 3 = 0) : 
  (3 * Real.sin θ + 2 * Real.cos θ) / (3 * Real.sin θ - Real.cos θ) = 4 := 
by 
  sorry

end NUMINAMATH_GPT_find_theta_ratio_l293_29378


namespace NUMINAMATH_GPT_find_points_l293_29328

def clubsuit (a b : ℝ) : ℝ := a^3 * b - a * b^3

theorem find_points (x y : ℝ) :
  clubsuit x y = clubsuit y x ↔ (y = x ∨ y = -x) := by
  sorry

end NUMINAMATH_GPT_find_points_l293_29328


namespace NUMINAMATH_GPT_murtha_pebbles_after_20_days_l293_29369

/- Define the sequence function for the pebbles collected each day -/
def pebbles_collected_day (n : ℕ) : ℕ :=
  if (n = 0) then 0 else 1 + pebbles_collected_day (n - 1)

/- Define the total pebbles collected by the nth day -/
def total_pebbles_collected (n : ℕ) : ℕ :=
  (n * (pebbles_collected_day n)) / 2

/- Define the total pebbles given away by the nth day -/
def pebbles_given_away (n : ℕ) : ℕ :=
  (n / 5) * 3

/- Define the net total of pebbles Murtha has on the nth day -/
def pebbles_net (n : ℕ) : ℕ :=
  total_pebbles_collected (n + 1) - pebbles_given_away (n + 1)

/- The main theorem about the pebbles Murtha has after the 20th day -/
theorem murtha_pebbles_after_20_days : pebbles_net 19 = 218 := 
  by sorry

end NUMINAMATH_GPT_murtha_pebbles_after_20_days_l293_29369


namespace NUMINAMATH_GPT_work_completion_days_l293_29318

-- We assume D is a certain number of days and W is some amount of work
variables (D W : ℕ)

-- Define the rate at which 3 people can do 3W work in D days
def rate_3_people : ℚ := 3 * W / D

-- Define the rate at which 5 people can do 5W work in D days
def rate_5_people : ℚ := 5 * W / D

-- The problem states that both rates must be equal
theorem work_completion_days : (3 * D) = D / 3 :=
by sorry

end NUMINAMATH_GPT_work_completion_days_l293_29318


namespace NUMINAMATH_GPT_problem1_problem2_l293_29312

theorem problem1 (n : ℕ) : 2^n + 3 = k * k → n = 0 :=
by
  intros
  sorry 

theorem problem2 (n : ℕ) : 2^n + 1 = x * x → n = 3 :=
by
  intros
  sorry 

end NUMINAMATH_GPT_problem1_problem2_l293_29312
