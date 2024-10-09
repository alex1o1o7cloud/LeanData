import Mathlib

namespace smallest_non_factor_product_of_factors_of_60_l1564_156425

theorem smallest_non_factor_product_of_factors_of_60 :
  ∃ x y : ℕ, x ≠ y ∧ x ∣ 60 ∧ y ∣ 60 ∧ ¬ (x * y ∣ 60) ∧ ∀ x' y' : ℕ, x' ≠ y' → x' ∣ 60 → y' ∣ 60 → ¬(x' * y' ∣ 60) → x * y ≤ x' * y' := 
sorry

end smallest_non_factor_product_of_factors_of_60_l1564_156425


namespace find_number_l1564_156422

theorem find_number (x : ℝ) : (30 / 100) * x = (60 / 100) * 150 + 120 ↔ x = 700 :=
by
  sorry

end find_number_l1564_156422


namespace bob_final_total_score_l1564_156469

theorem bob_final_total_score 
  (points_per_correct : ℕ := 5)
  (points_per_incorrect : ℕ := 2)
  (correct_answers : ℕ := 18)
  (incorrect_answers : ℕ := 2) :
  (points_per_correct * correct_answers - points_per_incorrect * incorrect_answers) = 86 :=
by 
  sorry

end bob_final_total_score_l1564_156469


namespace four_digit_numbers_sum_30_l1564_156438

-- Definitions of the variables and constraints
def valid_digit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9

-- The main statement we aim to prove
theorem four_digit_numbers_sum_30 : 
  ∃ (count : ℕ), 
  count = 20 ∧ 
  ∃ (a b c d : ℕ), 
  (1 ≤ a ∧ valid_digit a) ∧ 
  (valid_digit b) ∧ 
  (valid_digit c) ∧ 
  (valid_digit d) ∧ 
  a + b + c + d = 30 := sorry

end four_digit_numbers_sum_30_l1564_156438


namespace no_real_solution_l1564_156479

theorem no_real_solution :
  ¬ ∃ x : ℝ, (1 / (x + 2) + 8 / (x + 6) ≥ 2) ∧ (5 / (x + 1) - 2 ≤ 1) :=
by
  sorry

end no_real_solution_l1564_156479


namespace smallest_sum_of_four_distinct_numbers_l1564_156497

theorem smallest_sum_of_four_distinct_numbers 
  (S : Finset ℤ) 
  (h : S = {8, 26, -2, 13, -4, 0}) :
  ∃ (a b c d : ℤ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ a + b + c + d = 2 :=
sorry

end smallest_sum_of_four_distinct_numbers_l1564_156497


namespace smallest_positive_integer_x_l1564_156439

theorem smallest_positive_integer_x (x : ℕ) (h900 : ∃ a b c : ℕ, 900 = (2^a) * (3^b) * (5^c) ∧ a = 2 ∧ b = 2 ∧ c = 2) (h1152 : ∃ a b : ℕ, 1152 = (2^a) * (3^b) ∧ a = 7 ∧ b = 2) : x = 32 :=
by
  sorry

end smallest_positive_integer_x_l1564_156439


namespace avg_difference_l1564_156488

def avg (a b c : ℕ) := (a + b + c) / 3

theorem avg_difference : avg 14 32 53 - avg 21 47 22 = 3 :=
by
  sorry

end avg_difference_l1564_156488


namespace range_of_a_l1564_156435

theorem range_of_a {a : ℝ} :
  (∃ (x y : ℝ), (x - a)^2 + (y - a)^2 = 4 ∧ x^2 + y^2 = 4) ↔ (-2*Real.sqrt 2 < a ∧ a < 2*Real.sqrt 2 ∧ a ≠ 0) :=
sorry

end range_of_a_l1564_156435


namespace number_of_boxes_l1564_156471

variable (boxes : ℕ) -- number of boxes
variable (mangoes_per_box : ℕ) -- mangoes per box
variable (total_mangoes : ℕ) -- total mangoes

def dozen : ℕ := 12

-- Condition: each box contains 10 dozen mangoes
def condition1 : mangoes_per_box = 10 * dozen := by 
  sorry

-- Condition: total mangoes in all boxes together is 4320
def condition2 : total_mangoes = 4320 := by
  sorry

-- Proof problem: prove that the number of boxes is 36
theorem number_of_boxes (h1 : mangoes_per_box = 10 * dozen) 
                        (h2 : total_mangoes = 4320) :
  boxes = 4320 / (10 * dozen) :=
  by
  sorry

end number_of_boxes_l1564_156471


namespace mixed_oil_rate_l1564_156436

theorem mixed_oil_rate :
  let v₁ := 10
  let p₁ := 50
  let v₂ := 5
  let p₂ := 68
  let v₃ := 8
  let p₃ := 42
  let v₄ := 7
  let p₄ := 62
  let v₅ := 12
  let p₅ := 55
  let v₆ := 6
  let p₆ := 75
  let total_cost := v₁ * p₁ + v₂ * p₂ + v₃ * p₃ + v₄ * p₄ + v₅ * p₅ + v₆ * p₆
  let total_volume := v₁ + v₂ + v₃ + v₄ + v₅ + v₆
  let rate := total_cost / total_volume
  rate = 56.67 :=
by
  sorry

end mixed_oil_rate_l1564_156436


namespace beth_wins_if_arjun_plays_first_l1564_156467

/-- 
In the game where players take turns removing one, two adjacent, or two non-adjacent bricks from 
walls, given certain configurations, the configuration where Beth has a guaranteed winning 
strategy if Arjun plays first is (7, 3, 1).
-/
theorem beth_wins_if_arjun_plays_first :
  let nim_value_1 := 1
  let nim_value_2 := 2
  let nim_value_3 := 3
  let nim_value_7 := 2 -- computed as explained in the solution
  ∀ config : List ℕ,
    config = [7, 1, 1] ∨ config = [7, 2, 1] ∨ config = [7, 2, 2] ∨ config = [7, 3, 1] ∨ config = [7, 3, 2] →
    match config with
    | [7, 3, 1] => true
    | _ => false :=
by
  sorry

end beth_wins_if_arjun_plays_first_l1564_156467


namespace number_of_three_digit_integers_congruent_to_2_mod_4_l1564_156499

theorem number_of_three_digit_integers_congruent_to_2_mod_4 : 
  ∃ (count : ℕ), count = 225 ∧ ∀ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ n % 4 = 2 ↔ (∃ k : ℕ, 25 ≤ k ∧ k ≤ 249 ∧ n = 4 * k + 2) := 
by {
  sorry
}

end number_of_three_digit_integers_congruent_to_2_mod_4_l1564_156499


namespace product_of_roots_l1564_156468

noncomputable def is_root (a b c x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

theorem product_of_roots :
  ∀ (x1 x2 : ℝ), is_root 1 (-4) 3 x1 ∧ is_root 1 (-4) 3 x2 ∧ x1 ≠ x2 → x1 * x2 = 3 :=
by
  intros x1 x2 h
  sorry

end product_of_roots_l1564_156468


namespace power_function_increasing_l1564_156490

theorem power_function_increasing (m : ℝ) : 
  (∀ x : ℝ, 0 < x → (m^2 - 2*m - 2) * x^(-4*m - 2) > 0) ↔ m = -1 :=
by sorry

end power_function_increasing_l1564_156490


namespace polar_to_cartesian_l1564_156403

theorem polar_to_cartesian (ρ θ : ℝ) : (ρ * Real.cos θ = 0) → ρ = 0 ∨ θ = π/2 :=
by 
  sorry

end polar_to_cartesian_l1564_156403


namespace general_term_seq_l1564_156447

universe u

-- Define the sequence
def seq (a : ℕ → ℕ) : Prop :=
  (a 1 = 1) ∧ (∀ n, a (n + 1) = 2 * a n + 1)

-- State the theorem
theorem general_term_seq (a : ℕ → ℕ) (h : seq a) : ∀ n, a n = 2^n - 1 :=
by
  sorry

end general_term_seq_l1564_156447


namespace prop_A_l1564_156444

theorem prop_A (x : ℝ) (h : x > 1) : (x + (1 / (x - 1)) >= 3) :=
sorry

end prop_A_l1564_156444


namespace smaller_circle_radius_l1564_156420

-- Given conditions
def larger_circle_radius : ℝ := 10
def number_of_smaller_circles : ℕ := 7

-- The goal
theorem smaller_circle_radius :
  ∃ r : ℝ, (∃ D : ℝ, D = 2 * larger_circle_radius ∧ D = 4 * r) ∧ r = 2.5 :=
by
  sorry

end smaller_circle_radius_l1564_156420


namespace pablo_books_read_l1564_156401

noncomputable def pages_per_book : ℕ := 150
noncomputable def cents_per_page : ℕ := 1
noncomputable def cost_of_candy : ℕ := 1500    -- $15 in cents
noncomputable def leftover_money : ℕ := 300    -- $3 in cents
noncomputable def total_money := cost_of_candy + leftover_money
noncomputable def earnings_per_book := pages_per_book * cents_per_page

theorem pablo_books_read : total_money / earnings_per_book = 12 := by
  sorry

end pablo_books_read_l1564_156401


namespace abs_eq_two_l1564_156483

theorem abs_eq_two (m : ℤ) (h : |m| = 2) : m = 2 ∨ m = -2 :=
sorry

end abs_eq_two_l1564_156483


namespace sum_digits_500_l1564_156449

noncomputable def sum_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_digits_500 (k : ℕ) (h : k = 55) :
  sum_digits (63 * 10^k - 64) = 500 :=
by
  sorry

end sum_digits_500_l1564_156449


namespace no_real_solutions_l1564_156461

theorem no_real_solutions :
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 4 → ¬(3 * x^2 - 15 * x) / (x^2 - 4 * x) = x - 2) :=
by
  sorry

end no_real_solutions_l1564_156461


namespace value_of_M_l1564_156474

theorem value_of_M (x y z M : ℚ) 
  (h1 : x + y + z = 120)
  (h2 : x - 10 = M)
  (h3 : y + 10 = M)
  (h4 : 10 * z = M) : 
  M = 400 / 7 :=
sorry

end value_of_M_l1564_156474


namespace distance_A_to_C_through_B_l1564_156442

-- Define the distances on the map
def Distance_AB_map : ℝ := 20
def Distance_BC_map : ℝ := 10

-- Define the scale of the map
def scale : ℝ := 5

-- Define the actual distances
def Distance_AB := Distance_AB_map * scale
def Distance_BC := Distance_BC_map * scale

-- Define the total distance from A to C through B
def Distance_AC_through_B := Distance_AB + Distance_BC

-- Theorem to be proved
theorem distance_A_to_C_through_B : Distance_AC_through_B = 150 := by
  sorry

end distance_A_to_C_through_B_l1564_156442


namespace JimSiblings_l1564_156457

-- Define the students and their characteristics.
structure Student :=
  (name : String)
  (eyeColor : String)
  (hairColor : String)
  (wearsGlasses : Bool)

def Benjamin : Student := ⟨"Benjamin", "Blue", "Blond", true⟩
def Jim : Student := ⟨"Jim", "Brown", "Blond", false⟩
def Nadeen : Student := ⟨"Nadeen", "Brown", "Black", true⟩
def Austin : Student := ⟨"Austin", "Blue", "Black", false⟩
def Tevyn : Student := ⟨"Tevyn", "Blue", "Blond", true⟩
def Sue : Student := ⟨"Sue", "Brown", "Blond", false⟩

-- Define the condition that students from the same family share at least one characteristic.
def shareCharacteristic (s1 s2 : Student) : Bool :=
  (s1.eyeColor = s2.eyeColor) ∨
  (s1.hairColor = s2.hairColor) ∨
  (s1.wearsGlasses = s2.wearsGlasses)

-- Define what it means to be siblings of a student.
def areSiblings (s1 s2 s3 : Student) : Bool :=
  shareCharacteristic s1 s2 ∧
  shareCharacteristic s1 s3 ∧
  shareCharacteristic s2 s3

-- The theorem we are trying to prove.
theorem JimSiblings : areSiblings Jim Sue Benjamin = true := 
  by sorry

end JimSiblings_l1564_156457


namespace five_digit_number_is_40637_l1564_156411

theorem five_digit_number_is_40637 
  (A B C D E F G : ℕ) 
  (h1 : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧ 
        B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧ 
        C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧ 
        D ≠ E ∧ D ≠ F ∧ D ≠ G ∧
        E ≠ F ∧ E ≠ G ∧ 
        F ≠ G)
  (h2 : 0 < A ∧ 0 < B ∧ 0 < C ∧ 0 < D ∧ 0 < E ∧ 0 < F ∧ 0 < G)
  (h3 : A + 11 * A = 2 * (10 * B + A))
  (h4 : A + 10 * C + D = 2 * (10 * A + B))
  (h5 : 10 * C + D = 20 * A)
  (h6 : 20 + 62 = 2 * (10 * C + A)) -- for sequences formed by AB, CA, EF
  (h7 : 21 + 63 = 2 * (10 * G + A)) -- for sequences formed by BA, CA, GA
  : ∃ (C D E F G : ℕ), C * 10000 + D * 1000 + E * 100 + F * 10 + G = 40637 := 
sorry

end five_digit_number_is_40637_l1564_156411


namespace number_of_integer_chords_through_point_l1564_156478

theorem number_of_integer_chords_through_point {r : ℝ} {c : ℝ} 
    (hr: r = 13) (hc : c = 12) : 
    ∃ n : ℕ, n = 17 :=
by
  -- Suppose O is the center and P is a point inside the circle such that OP = 12
  -- Given radius r = 13, we need to show there are 17 different integer chord lengths
  sorry  -- Proof is omitted

end number_of_integer_chords_through_point_l1564_156478


namespace min_value_f_at_0_l1564_156406

noncomputable def f (a x : ℝ) : ℝ :=
if x ≤ 0 then (x - a)^2 else x + 1/x + a

theorem min_value_f_at_0 (a : ℝ) : (∀ x : ℝ, f a 0 ≤ f a x) ↔ 0 ≤ a ∧ a ≤ 2 :=
by
  sorry

end min_value_f_at_0_l1564_156406


namespace sum_of_coordinates_l1564_156434

-- Define the conditions for m and n
def m : ℤ := -3
def n : ℤ := 2

-- State the proposition based on the conditions
theorem sum_of_coordinates : m + n = -1 := 
by 
  -- Provide an incomplete proof skeleton with "sorry" to skip the proof
  sorry

end sum_of_coordinates_l1564_156434


namespace rancher_monetary_loss_l1564_156448

def rancher_head_of_cattle := 500
def market_rate_per_head := 700
def sick_cattle := 350
def additional_cost_per_sick_animal := 80
def reduced_price_per_head := 450

def expected_revenue := rancher_head_of_cattle * market_rate_per_head
def loss_from_death := sick_cattle * market_rate_per_head
def additional_sick_cost := sick_cattle * additional_cost_per_sick_animal
def remaining_cattle := rancher_head_of_cattle - sick_cattle
def revenue_from_remaining_cattle := remaining_cattle * reduced_price_per_head

def total_loss := (expected_revenue - revenue_from_remaining_cattle) + additional_sick_cost

theorem rancher_monetary_loss : total_loss = 310500 := by
  sorry

end rancher_monetary_loss_l1564_156448


namespace find_a_plus_c_l1564_156498

theorem find_a_plus_c {a b c d : ℝ} 
  (h1 : ∀ x, -|x - a| + b = |x - c| + d → x = 4 ∧ -|4 - a| + b = 7 ∨ x = 10 ∧ -|10 - a| + b = 3)
  (h2 : b + d = 12): a + c = 14 := by
  sorry

end find_a_plus_c_l1564_156498


namespace fraction_zero_implies_x_eq_one_l1564_156472

theorem fraction_zero_implies_x_eq_one (x : ℝ) (h : (x - 1) / (x + 1) = 0) : x = 1 :=
sorry

end fraction_zero_implies_x_eq_one_l1564_156472


namespace max_earnings_l1564_156446

section MaryEarnings

def regular_rate : ℝ := 10
def first_period_hours : ℕ := 40
def second_period_hours : ℕ := 10
def third_period_hours : ℕ := 10
def weekend_days : ℕ := 2
def weekend_bonus_per_day : ℝ := 50
def bonus_threshold_hours : ℕ := 55
def overtime_multiplier_second_period : ℝ := 0.25
def overtime_multiplier_third_period : ℝ := 0.5
def milestone_bonus : ℝ := 100

def regular_pay := regular_rate * first_period_hours
def second_period_pay := (regular_rate * (1 + overtime_multiplier_second_period)) * second_period_hours
def third_period_pay := (regular_rate * (1 + overtime_multiplier_third_period)) * third_period_hours
def weekend_bonus := weekend_days * weekend_bonus_per_day
def milestone_pay := milestone_bonus

def total_earnings := regular_pay + second_period_pay + third_period_pay + weekend_bonus + milestone_pay

theorem max_earnings : total_earnings = 875 := by
  sorry

end MaryEarnings

end max_earnings_l1564_156446


namespace blocks_added_l1564_156404

theorem blocks_added (original_blocks new_blocks added_blocks : ℕ) 
  (h1 : original_blocks = 35) 
  (h2 : new_blocks = 65) 
  (h3 : new_blocks = original_blocks + added_blocks) : 
  added_blocks = 30 :=
by
  -- We use the given conditions to prove the statement
  sorry

end blocks_added_l1564_156404


namespace simplify_expression_l1564_156409

theorem simplify_expression (k : ℤ) : 
  let a := 1
  let b := 3
  (6 * k + 18) / 6 = k + 3 ∧ a / b = 1 / 3 :=
by
  sorry

end simplify_expression_l1564_156409


namespace length_of_other_train_is_correct_l1564_156415

noncomputable def length_of_second_train 
  (length_first_train : ℝ) 
  (speed_first_train : ℝ) 
  (speed_second_train : ℝ) 
  (time_to_cross : ℝ) 
  : ℝ := 
  let speed_first_train_m_s := speed_first_train * (1000 / 3600)
  let speed_second_train_m_s := speed_second_train * (1000 / 3600)
  let relative_speed := speed_first_train_m_s + speed_second_train_m_s
  let total_distance := relative_speed * time_to_cross
  total_distance - length_first_train

theorem length_of_other_train_is_correct :
  length_of_second_train 250 120 80 9 = 249.95 :=
by
  unfold length_of_second_train
  simp
  sorry

end length_of_other_train_is_correct_l1564_156415


namespace train_crossing_time_l1564_156481

-- Definitions for the conditions
def speed_kmph : Float := 72
def speed_mps : Float := speed_kmph * (1000 / 3600)
def length_train_m : Float := 240.0416
def length_platform_m : Float := 280
def total_distance_m : Float := length_train_m + length_platform_m

-- The problem statement
theorem train_crossing_time :
  (total_distance_m / speed_mps) = 26.00208 :=
by
  sorry

end train_crossing_time_l1564_156481


namespace james_weekly_pistachio_cost_l1564_156428

def cost_per_can : ℕ := 10
def ounces_per_can : ℕ := 5
def consumption_per_5_days : ℕ := 30
def days_per_week : ℕ := 7

theorem james_weekly_pistachio_cost : (days_per_week / 5 * consumption_per_5_days) / ounces_per_can * cost_per_can = 90 := 
by
  sorry

end james_weekly_pistachio_cost_l1564_156428


namespace proof_problem_l1564_156402

theorem proof_problem
  (n : ℕ)
  (h : n = 16^3018) :
  n / 8 = 2^9032 := by
  sorry

end proof_problem_l1564_156402


namespace expected_total_rain_l1564_156475

noncomputable def expected_daily_rain : ℝ :=
  (0.50 * 0) + (0.30 * 3) + (0.20 * 8)

theorem expected_total_rain :
  (5 * expected_daily_rain) = 12.5 :=
by
  sorry

end expected_total_rain_l1564_156475


namespace ship_speed_in_still_water_eq_25_l1564_156491

-- Definitions and conditions
variable (x : ℝ) (h1 : 81 / (x + 2) = 69 / (x - 2)) (h2 : x ≠ -2) (h3 : x ≠ 2)

-- Theorem statement
theorem ship_speed_in_still_water_eq_25 : x = 25 :=
by
  sorry

end ship_speed_in_still_water_eq_25_l1564_156491


namespace number_of_bikes_l1564_156470

theorem number_of_bikes (total_wheels : ℕ) (car_wheels : ℕ) (tricycle_wheels : ℕ) (roller_skate_wheels : ℕ) (trash_can_wheels : ℕ) (bike_wheels : ℕ) (num_bikes : ℕ) :
  total_wheels = 25 →
  car_wheels = 2 * 4 →
  tricycle_wheels = 3 →
  roller_skate_wheels = 4 →
  trash_can_wheels = 2 →
  bike_wheels = 2 →
  (total_wheels - (car_wheels + tricycle_wheels + roller_skate_wheels + trash_can_wheels)) = bike_wheels * num_bikes →
  num_bikes = 4 := 
by
  intros total_wheels_eq total_car_wheels_eq tricycle_wheels_eq roller_skate_wheels_eq trash_can_wheels_eq bike_wheels_eq remaining_wheels_eq
  sorry

end number_of_bikes_l1564_156470


namespace k_values_for_perpendicular_lines_l1564_156414

-- Definition of perpendicular condition for lines
def perpendicular_lines (k : ℝ) : Prop :=
  k * (k - 1) + (1 - k) * (2 * k + 3) = 0

-- Lean 4 statement representing the math proof problem
theorem k_values_for_perpendicular_lines (k : ℝ) :
  perpendicular_lines k ↔ k = -3 ∨ k = 1 :=
by
  sorry

end k_values_for_perpendicular_lines_l1564_156414


namespace units_digit_expression_l1564_156486

theorem units_digit_expression: 
  (8 * 19 * 1981 + 6^3 - 2^5) % 10 = 6 := 
by
  sorry

end units_digit_expression_l1564_156486


namespace length_of_one_side_of_regular_pentagon_l1564_156418

-- Define the conditions
def is_regular_pentagon (P : ℝ) (n : ℕ) : Prop := n = 5 ∧ P = 23.4

-- State the theorem
theorem length_of_one_side_of_regular_pentagon (P : ℝ) (n : ℕ) 
  (h : is_regular_pentagon P n) : P / n = 4.68 :=
by
  sorry

end length_of_one_side_of_regular_pentagon_l1564_156418


namespace two_point_four_times_eight_point_two_l1564_156445

theorem two_point_four_times_eight_point_two (x y z : ℝ) (hx : x = 2.4) (hy : y = 8.2) (hz : z = 4.8 + 5.2) :
  x * y * z = 2.4 * 8.2 * 10 ∧ abs (x * y * z - 200) < abs (x * y * z - 150) ∧
  abs (x * y * z - 200) < abs (x * y * z - 250) ∧
  abs (x * y * z - 200) < abs (x * y * z - 300) ∧
  abs (x * y * z - 200) < abs (x * y * z - 350) := by
  sorry

end two_point_four_times_eight_point_two_l1564_156445


namespace total_consultation_time_l1564_156450

-- Define the times in which each chief finishes a pipe
def chief1_time := 10
def chief2_time := 30
def chief3_time := 60

theorem total_consultation_time : 
  ∃ (t : ℕ), (∃ x, ((x / chief1_time) + (x / chief2_time) + (x / chief3_time) = 1) ∧ t = 3 * x) ∧ t = 20 :=
sorry

end total_consultation_time_l1564_156450


namespace problem1_l1564_156452

theorem problem1 (A B C : Prop) : (A ∨ (B ∧ C)) ↔ ((A ∨ B) ∧ (A ∨ C)) :=
sorry 

end problem1_l1564_156452


namespace relationship_among_abc_l1564_156429

noncomputable def a : ℝ := (1 / 3) ^ (2 / 5)
noncomputable def b : ℝ := (2 / 3) ^ (2 / 5)
noncomputable def c : ℝ := Real.log (1 / 5) / Real.log (1 / 3)

theorem relationship_among_abc : c > b ∧ b > a :=
by
  have h1 : a = (1 / 3) ^ (2 / 5) := rfl
  have h2 : b = (2 / 3) ^ (2 / 5) := rfl
  have h3 : c = Real.log (1 / 5) / Real.log (1 / 3) := rfl
  sorry

end relationship_among_abc_l1564_156429


namespace calvin_score_l1564_156494

theorem calvin_score (C : ℚ) (h_paislee_score : (3/4) * C = 125) : C = 167 := 
  sorry

end calvin_score_l1564_156494


namespace simplify_expression_l1564_156489

variables (a b : ℝ)
noncomputable def x := (1 / 2) * (Real.sqrt (a / b) - Real.sqrt (b / a))

theorem simplify_expression (ha : a > 0) (hb : b > 0) :
  (2 * a * Real.sqrt (1 + x a b ^ 2)) / (x a b + Real.sqrt (1 + x a b ^ 2)) = a + b :=
sorry

end simplify_expression_l1564_156489


namespace concave_number_probability_l1564_156430

/-- Definition of a concave number -/
def is_concave_number (a b c : ℕ) : Prop :=
  a > b ∧ c > b

/-- Set of possible digits -/
def digits : Finset ℕ := {4, 5, 6, 7, 8}

 /-- Total number of distinct three-digit combinations -/
def total_combinations : ℕ := 60

 /-- Number of concave numbers -/
def concave_numbers : ℕ := 20

 /-- Probability that a randomly chosen three-digit number is a concave number -/
def probability_concave : ℚ := concave_numbers / total_combinations

theorem concave_number_probability :
  probability_concave = 1 / 3 :=
by
  sorry

end concave_number_probability_l1564_156430


namespace imaginary_part_of_complex_l1564_156417

theorem imaginary_part_of_complex :
  let i := Complex.I
  let z := 10 * i / (3 + i)
  z.im = 3 :=
by
  sorry

end imaginary_part_of_complex_l1564_156417


namespace determine_a_l1564_156466

-- Define the function f(x)
def f (x a : ℝ) : ℝ := x^2 - a * x + 3

-- Define the condition that f(x) >= a for all x in the interval [-1, +∞)
def condition (a : ℝ) : Prop := ∀ x : ℝ, x ≥ -1 → f x a ≥ a

-- The theorem to prove:
theorem determine_a : ∀ a : ℝ, condition a ↔ a ≤ 2 :=
by
  sorry

end determine_a_l1564_156466


namespace calculate_change_l1564_156482

theorem calculate_change : 
  let bracelet_cost := 15
  let necklace_cost := 10
  let mug_cost := 20
  let num_bracelets := 3
  let num_necklaces := 2
  let num_mugs := 1
  let discount := 0.10
  let total_cost := (num_bracelets * bracelet_cost) + (num_necklaces * necklace_cost) + (num_mugs * mug_cost)
  let discount_amount := total_cost * discount
  let final_amount := total_cost - discount_amount
  let payment := 100
  let change := payment - final_amount
  change = 23.50 :=
by
  -- Intentionally skipping the proof
  sorry

end calculate_change_l1564_156482


namespace expand_expression_l1564_156487

theorem expand_expression (a b : ℤ) : (-1 + a * b^2)^2 = 1 - 2 * a * b^2 + a^2 * b^4 :=
by sorry

end expand_expression_l1564_156487


namespace linear_regression_forecast_l1564_156496

variable (x : ℝ) (y : ℝ)
variable (b : ℝ) (a : ℝ) (center_x : ℝ) (center_y : ℝ)

theorem linear_regression_forecast :
  b=-2 → center_x=4 → center_y=50 → (center_y = b * center_x + a) →
  (a = 58) → (x = 6) → y = b * x + a → y = 46 :=
by
  intros hb hcx hcy heq ha hx hy
  sorry

end linear_regression_forecast_l1564_156496


namespace gcf_54_81_l1564_156453

theorem gcf_54_81 : Nat.gcd 54 81 = 27 :=
by sorry

end gcf_54_81_l1564_156453


namespace average_percentage_for_all_students_l1564_156456

-- Definitions of the variables
def students1 : Nat := 15
def average1 : Nat := 75
def students2 : Nat := 10
def average2 : Nat := 90
def total_students : Nat := students1 + students2
def total_percentage1 : Nat := students1 * average1
def total_percentage2 : Nat := students2 * average2
def total_percentage : Nat := total_percentage1 + total_percentage2

-- Main theorem stating the average percentage for all students.
theorem average_percentage_for_all_students :
  total_percentage / total_students = 81 := by
  sorry

end average_percentage_for_all_students_l1564_156456


namespace find_a_l1564_156441

noncomputable def has_exactly_one_solution_in_x (a : ℝ) : Prop :=
  ∀ x : ℝ, |x^2 + 2*a*x + a + 5| = 3 → x = -a

theorem find_a (a : ℝ) : has_exactly_one_solution_in_x a ↔ (a = 4 ∨ a = -2) :=
by
  sorry

end find_a_l1564_156441


namespace triangle_side_relation_l1564_156465

variable {α β γ : ℝ} -- angles in the triangle
variable {a b c : ℝ} -- sides opposite to the angles

theorem triangle_side_relation
  (h1 : α = 3 * β)
  (h2 : α = 6 * γ)
  (h_sum : α + β + γ = 180)
  : b * c^2 = (a + b) * (a - b)^2 := 
by
  sorry

end triangle_side_relation_l1564_156465


namespace central_angle_of_sector_l1564_156459

theorem central_angle_of_sector (P : ℝ) (x : ℝ) (h : P = 1 / 8) : x = 45 :=
by
  sorry

end central_angle_of_sector_l1564_156459


namespace total_education_duration_l1564_156455

-- Definitions from the conditions
def high_school_duration : ℕ := 4 - 1
def tertiary_education_duration : ℕ := 3 * high_school_duration

-- The theorem statement
theorem total_education_duration : high_school_duration + tertiary_education_duration = 12 :=
by
  sorry

end total_education_duration_l1564_156455


namespace alex_walking_distance_l1564_156400

theorem alex_walking_distance
  (distance : ℝ)
  (time_45 : ℝ)
  (walking_rate : distance = 1.5 ∧ time_45 = 45):
  ∃ distance_90, distance_90 = 3 :=
by 
  sorry

end alex_walking_distance_l1564_156400


namespace incorrect_conclusion_l1564_156458

theorem incorrect_conclusion (b x : ℂ) (h : x^2 - b * x + 1 = 0) : x = 1 ∨ x = -1
  ↔ (b = 2 ∨ b = -2) :=
by sorry

end incorrect_conclusion_l1564_156458


namespace complement_A_l1564_156437

-- Definitions for the conditions
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x < 1}

-- Proof statement
theorem complement_A : (U \ A) = {x | x ≥ 1} := by
  sorry

end complement_A_l1564_156437


namespace bicycle_cost_price_l1564_156412

theorem bicycle_cost_price (CP_A : ℝ) (CP_B : ℝ) (SP_C : ℝ)
    (h1 : CP_B = 1.60 * CP_A)
    (h2 : SP_C = 1.25 * CP_B)
    (h3 : SP_C = 225) :
    CP_A = 225 / 2.00 :=
by
  sorry -- the proof steps will follow here

end bicycle_cost_price_l1564_156412


namespace Cindy_initial_marbles_l1564_156440

theorem Cindy_initial_marbles (M : ℕ) 
  (h1 : 4 * (M - 320) = 720) : M = 500 :=
by
  sorry

end Cindy_initial_marbles_l1564_156440


namespace helga_ratio_l1564_156416

variable (a b c d : ℕ)

def helga_shopping (a b c d total_shoes pairs_first_three : ℕ) : Prop :=
  a = 7 ∧
  b = a + 2 ∧
  c = 0 ∧
  a + b + c + d = total_shoes ∧
  pairs_first_three = a + b + c ∧
  total_shoes = 48 ∧
  (d : ℚ) / (pairs_first_three : ℚ) = 2

theorem helga_ratio : helga_shopping 7 9 0 32 48 16 := by
  sorry

end helga_ratio_l1564_156416


namespace work_problem_l1564_156432

theorem work_problem 
  (A_work_time : ℤ) 
  (B_work_time : ℤ) 
  (x : ℤ)
  (A_work_rate : ℚ := 1 / 15 )
  (work_left : ℚ := 0.18333333333333335)
  (worked_together_for : ℚ := 7)
  (work_done : ℚ := 1 - work_left) :
  (7 * (1 / 15 + 1 / x) = work_done) → x = 20 :=
by sorry

end work_problem_l1564_156432


namespace angle_between_adjacent_triangles_l1564_156431

-- Define the setup of the problem
def five_nonoverlapping_equilateral_triangles (angles : Fin 5 → ℝ) :=
  ∀ i, angles i = 60

def angles_between_adjacent_triangles (angles : Fin 5 → ℝ) :=
  ∀ i j, i ≠ j → angles i = angles j

-- State the main theorem
theorem angle_between_adjacent_triangles :
  ∀ (angles : Fin 5 → ℝ),
    five_nonoverlapping_equilateral_triangles angles →
    angles_between_adjacent_triangles angles →
    ((360 - 5 * 60) / 5) = 12 :=
by
  intros angles h1 h2
  sorry

end angle_between_adjacent_triangles_l1564_156431


namespace all_statements_correct_l1564_156408

theorem all_statements_correct :
  (∀ (b h : ℝ), (3 * b * h = 3 * (b * h))) ∧
  (∀ (b h : ℝ), (1/2 * b * (1/2 * h) = 1/2 * (1/2 * b * h))) ∧
  (∀ (r : ℝ), (π * (2 * r) ^ 2 = 4 * (π * r ^ 2))) ∧
  (∀ (r : ℝ), (π * (3 * r) ^ 2 = 9 * (π * r ^ 2))) ∧
  (∀ (s : ℝ), ((2 * s) ^ 2 = 4 * (s ^ 2)))
  → False := 
by 
  intros h
  sorry

end all_statements_correct_l1564_156408


namespace hours_l1564_156419

def mechanic_hours_charged (h : ℕ) : Prop :=
  45 * h + 225 = 450

theorem hours (h : ℕ) : mechanic_hours_charged h → h = 5 :=
by
  intro h_eq
  have : 45 * h + 225 = 450 := h_eq
  sorry

end hours_l1564_156419


namespace length_of_second_train_l1564_156421

theorem length_of_second_train 
  (length_first_train : ℝ) 
  (speed_first_train_kmph : ℝ) 
  (speed_second_train_kmph : ℝ) 
  (time_to_cross : ℝ) 
  (h1 : length_first_train = 400)
  (h2 : speed_first_train_kmph = 72)
  (h3 : speed_second_train_kmph = 36)
  (h4 : time_to_cross = 69.99440044796417) :
  let speed_first_train := speed_first_train_kmph * (1000 / 3600)
  let speed_second_train := speed_second_train_kmph * (1000 / 3600)
  let relative_speed := speed_first_train - speed_second_train
  let distance := relative_speed * time_to_cross
  let length_second_train := distance - length_first_train
  length_second_train = 299.9440044796417 :=
  by
    sorry

end length_of_second_train_l1564_156421


namespace log_sum_l1564_156405

variable (m a b : ℝ)
variable (m_pos : 0 < m)
variable (m_ne_one : m ≠ 1)
variable (h1 : m^2 = a)
variable (h2 : m^3 = b)

theorem log_sum (m_pos : 0 < m) (m_ne_one : m ≠ 1) (h1 : m^2 = a) (h2 : m^3 = b) :
  2 * Real.log (a) / Real.log (m) + Real.log (b) / Real.log (m) = 7 := 
sorry

end log_sum_l1564_156405


namespace smallest_intersection_value_l1564_156464

theorem smallest_intersection_value (a b : ℝ) (f g : ℝ → ℝ)
    (Hf : ∀ x, f x = x^4 - 6 * x^3 + 11 * x^2 - 6 * x + a)
    (Hg : ∀ x, g x = x + b)
    (Hinter : ∀ x, f x = g x → true):
  ∃ x₀, x₀ = 0 :=
by
  intros
  -- Further steps would involve proving roots and conditions stated but omitted here.
  sorry

end smallest_intersection_value_l1564_156464


namespace neg_neg_eq_l1564_156484

theorem neg_neg_eq (n : ℤ) : -(-n) = n :=
  sorry

example : -(-2023) = 2023 :=
by apply neg_neg_eq

end neg_neg_eq_l1564_156484


namespace emily_necklaces_l1564_156492

theorem emily_necklaces (total_beads : ℤ) (beads_per_necklace : ℤ) 
(h_total_beads : total_beads = 16) (h_beads_per_necklace : beads_per_necklace = 8) : 
  total_beads / beads_per_necklace = 2 := 
by
  sorry

end emily_necklaces_l1564_156492


namespace lydia_eats_apple_age_l1564_156485

-- Define the conditions
def years_to_bear_fruit : ℕ := 7
def age_when_planted : ℕ := 4
def current_age : ℕ := 9

-- Define the theorem statement
theorem lydia_eats_apple_age : 
  (age_when_planted + years_to_bear_fruit = 11) :=
by
  sorry

end lydia_eats_apple_age_l1564_156485


namespace shifted_polynomial_sum_l1564_156476

theorem shifted_polynomial_sum (a b c : ℝ) :
  (∀ x : ℝ, (3 * x^2 + 2 * x + 5) = (a * (x + 5)^2 + b * (x + 5) + c)) →
  a + b + c = 125 :=
by
  sorry

end shifted_polynomial_sum_l1564_156476


namespace parallel_lines_condition_iff_l1564_156477

def line_parallel (a : ℝ) : Prop :=
  let l1_slope := -1 / -a
  let l2_slope := -(a - 1) / -12
  l1_slope = l2_slope

theorem parallel_lines_condition_iff (a : ℝ) :
  (a = 4) ↔ line_parallel a := by
  sorry

end parallel_lines_condition_iff_l1564_156477


namespace arithmetic_sum_l1564_156463

theorem arithmetic_sum :
  ∀ (a : ℕ → ℝ),
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) →
  (∃ x : ℝ, ∃ y : ℝ, x^2 - 6 * x - 1 = 0 ∧ y^2 - 6 * y - 1 = 0 ∧ x = a 3 ∧ y = a 15) →
  a 7 + a 8 + a 9 + a 10 + a 11 = 15 :=
by
  intros a h_arith_seq h_roots
  sorry

end arithmetic_sum_l1564_156463


namespace train_speed_l1564_156495

def train_length : ℝ := 110
def bridge_length : ℝ := 265
def crossing_time : ℝ := 30
def conversion_factor : ℝ := 3.6

theorem train_speed (train_length bridge_length crossing_time conversion_factor : ℝ) :
  (train_length + bridge_length) / crossing_time * conversion_factor = 45 :=
by
  sorry

end train_speed_l1564_156495


namespace sum_original_and_correct_value_l1564_156443

theorem sum_original_and_correct_value (x : ℕ) (h : x + 14 = 68) :
  x + (x + 41) = 149 := by
  sorry

end sum_original_and_correct_value_l1564_156443


namespace second_card_is_three_l1564_156462

theorem second_card_is_three (a b c d : ℕ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
                             (h_sum : a + b + c + d = 30)
                             (h_increasing : a < b ∧ b < c ∧ c < d)
                             (h_dennis : ∀ x y z, x = a → (y ≠ b ∨ z ≠ c ∨ d ≠ 30 - a - y - z))
                             (h_mandy : ∀ x y z, x = b → (y ≠ a ∨ z ≠ c ∨ d ≠ 30 - x - y - z))
                             (h_sandy : ∀ x y z, x = c → (y ≠ a ∨ z ≠ b ∨ d ≠ 30 - x - y - z))
                             (h_randy : ∀ x y z, x = d → (y ≠ a ∨ z ≠ b ∨ c ≠ 30 - x - y - z)) :
  b = 3 := 
sorry

end second_card_is_three_l1564_156462


namespace number_of_unsold_items_l1564_156460

theorem number_of_unsold_items (v k : ℕ) (hv : v ≤ 53) (havg_int : ∃ n : ℕ, k = n * v)
  (hk_eq : k = 130*v - 1595) 
  (hnew_avg : (k + 2505) / (v + 7) = 130) :
  60 - (v + 7) = 24 :=
by
  sorry

end number_of_unsold_items_l1564_156460


namespace initial_girls_are_11_l1564_156493

variable {n : ℕ}  -- Assume n (the total number of students initially) is a natural number

def initial_num_girls (n : ℕ) : ℕ := (n / 2)

def total_students_after_changes (n : ℕ) : ℕ := n - 2

def num_girls_after_changes (n : ℕ) : ℕ := (n / 2) - 3

def is_40_percent_girls (n : ℕ) : Prop := (num_girls_after_changes n) * 10 = 4 * (total_students_after_changes n)

theorem initial_girls_are_11 :
  is_40_percent_girls 22 → initial_num_girls 22 = 11 :=
by
  sorry

end initial_girls_are_11_l1564_156493


namespace at_least_one_tails_up_l1564_156454

-- Define propositions p and q
variable (p q : Prop)

-- The theorem statement
theorem at_least_one_tails_up : (¬p ∨ ¬q) ↔ ¬(p ∧ q) := by
  sorry

end at_least_one_tails_up_l1564_156454


namespace circle_center_radius_l1564_156413

open Real

theorem circle_center_radius (x y : ℝ) :
  x^2 + y^2 - 6*x = 0 ↔ (x - 3)^2 + y^2 = 9 :=
by sorry

end circle_center_radius_l1564_156413


namespace Mehki_is_10_years_older_than_Jordyn_l1564_156433

def Zrinka_age : Nat := 6
def Mehki_age : Nat := 22
def Jordyn_age : Nat := 2 * Zrinka_age

theorem Mehki_is_10_years_older_than_Jordyn : Mehki_age - Jordyn_age = 10 :=
by
  sorry

end Mehki_is_10_years_older_than_Jordyn_l1564_156433


namespace estimate_larger_than_difference_l1564_156451

theorem estimate_larger_than_difference (x y z : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : z > 0) :
    (x + z) - (y - z) > x - y :=
    sorry

end estimate_larger_than_difference_l1564_156451


namespace arithmetic_seq_sum_l1564_156407

/-- Given an arithmetic sequence {a_n} such that a_5 + a_6 + a_7 = 15,
prove that the sum of the first 11 terms of the sequence S_11 is 55. -/
theorem arithmetic_seq_sum (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h₁ : a 5 + a 6 + a 7 = 15)
  (h₂ : ∀ n, S n = n * (a 1 + a n) / 2) :
  S 11 = 55 :=
sorry

end arithmetic_seq_sum_l1564_156407


namespace library_books_difference_l1564_156480

theorem library_books_difference (total_books : ℕ) (borrowed_percentage : ℕ) 
  (initial_books : total_books = 400) 
  (percentage_borrowed : borrowed_percentage = 30) :
  (total_books - (borrowed_percentage * total_books / 100)) = 280 :=
by
  sorry

end library_books_difference_l1564_156480


namespace average_speed_lila_l1564_156473

-- Definitions
def distance1 : ℝ := 50 -- miles
def speed1 : ℝ := 20 -- miles per hour
def distance2 : ℝ := 20 -- miles
def speed2 : ℝ := 40 -- miles per hour
def break_time : ℝ := 0.5 -- hours

-- Question to prove: Lila's average speed for the entire ride is 20 miles per hour
theorem average_speed_lila (d1 d2 s1 s2 bt : ℝ) 
  (h1 : d1 = distance1) (h2 : s1 = speed1) (h3 : d2 = distance2) (h4 : s2 = speed2) (h5 : bt = break_time) :
  (d1 + d2) / (d1 / s1 + d2 / s2 + bt) = 20 :=
by
  sorry

end average_speed_lila_l1564_156473


namespace least_number_of_roots_l1564_156410

variable {g : ℝ → ℝ}

-- Conditions
axiom g_defined (x : ℝ) : g x = g x
axiom g_symmetry_1 (x : ℝ) : g (3 + x) = g (3 - x)
axiom g_symmetry_2 (x : ℝ) : g (5 + x) = g (5 - x)
axiom g_at_1 : g 1 = 0

-- Root count in the interval
theorem least_number_of_roots : ∃ (n : ℕ), n >= 250 ∧ (∀ m, -1000 ≤ (1 + 8 * m:ℝ) ∧ (1 + 8 * m:ℝ) ≤ 1000 → g (1 + 8 * m) = 0) :=
sorry

end least_number_of_roots_l1564_156410


namespace tetrahedron_edge_length_l1564_156423

-- Define the problem as a Lean theorem statement
theorem tetrahedron_edge_length (r : ℝ) (a : ℝ) (h : r = 1) :
  a = 2 * Real.sqrt 2 :=
sorry

end tetrahedron_edge_length_l1564_156423


namespace max_trees_cut_l1564_156424

theorem max_trees_cut (n : ℕ) (h : n = 2001) :
  (∃ m : ℕ, m = n * n ∧ ∀ (x y : ℕ), x < n ∧ y < n → (x % 2 = 0 ∧ y % 2 = 0 → m = 1001001)) := sorry

end max_trees_cut_l1564_156424


namespace value_of_m_squared_plus_2m_minus_3_l1564_156426

theorem value_of_m_squared_plus_2m_minus_3 (m : ℤ) : 
  (∀ x : ℤ, 4 * (x - 1) - m * x + 6 = 8 → x = 3) →
  m^2 + 2 * m - 3 = 5 :=
by
  sorry

end value_of_m_squared_plus_2m_minus_3_l1564_156426


namespace coursework_materials_spending_l1564_156427

def budget : ℝ := 1000
def food_percentage : ℝ := 0.30
def accommodation_percentage : ℝ := 0.15
def entertainment_percentage : ℝ := 0.25

theorem coursework_materials_spending : 
    budget - (budget * food_percentage + budget * accommodation_percentage + budget * entertainment_percentage) = 300 := 
by 
  -- steps you would use to prove this
  sorry

end coursework_materials_spending_l1564_156427
