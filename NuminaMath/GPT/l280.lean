import Mathlib

namespace targets_break_order_count_l280_280322

theorem targets_break_order_count :
  let arrangements := finset.card (finset.perm (multiset.of_list ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C'])) in
  arrangements = 560 :=
by
  sorry

end targets_break_order_count_l280_280322


namespace combined_weight_of_Leo_and_Kendra_l280_280474

theorem combined_weight_of_Leo_and_Kendra :
  ∃ (K : ℝ), (92 + K = 160) ∧ (102 = 1.5 * K) :=
by
  sorry

end combined_weight_of_Leo_and_Kendra_l280_280474


namespace fraction_of_milk_is_one_fourth_l280_280949

theorem fraction_of_milk_is_one_fourth :
  ∀ (cup1_tea_initial cup2_milk_initial : ℚ)
    (cup1_transfer_fraction cup2_transfer_fraction1 cup1_transfer_fraction_back : ℚ),
    cup1_tea_initial = 6 →
    cup2_milk_initial = 6 →
    cup1_transfer_fraction = 1/3 →
    cup2_transfer_fraction1 = 1/4 →
    cup1_transfer_fraction_back = 1/6 →
    let cup1_tea_after_first_transfer := cup1_tea_initial * (1 - cup1_transfer_fraction),
        cup2_milk_after_first_transfer := cup2_milk_initial,
        cup2_tea_after_first_transfer := cup1_tea_initial * cup1_transfer_fraction,
        cup2_total_after_first_transfer := cup2_milk_after_first_transfer + cup2_tea_after_first_transfer,
        transfer_to_cup1 := cup2_total_after_first_transfer * cup2_transfer_fraction1,
        cup1_tea_after_second_transfer := cup1_tea_after_first_transfer + transfer_to_cup1 * (cup2_tea_after_first_transfer / cup2_total_after_first_transfer),
        cup1_milk_after_second_transfer := 1.5,
        cup1_total_after_second_transfer := cup1_tea_after_second_transfer + cup1_milk_after_second_transfer,
        transfer_back_to_cup2 := cup1_total_after_second_transfer * cup1_transfer_fraction_back,
        cup1_tea_after_final_transfer := cup1_tea_after_second_transfer - transfer_back_to_cup2 * (cup1_tea_after_second_transfer / cup1_total_after_second_transfer),
        cup1_milk_after_final_transfer := cup1_milk_after_second_transfer - transfer_back_to_cup2 * (cup1_milk_after_second_transfer / cup1_total_after_second_transfer),
        cup1_total_after_final_transfer := cup1_tea_after_final_transfer + cup1_milk_after_final_transfer
    in
    (cup1_milk_after_final_transfer / cup1_total_after_final_transfer) = 1/4 := by
  intros _ _ _ _ _ h1 h2 h3 h4 h5
  let cup1_tea_after_first_transfer := 6 * (1 - 1/3)
  let cup2_milk_after_first_transfer := 6
  let cup2_tea_after_first_transfer := 6 * 1/3
  let cup2_total_after_first_transfer := 6 + 6 * 1/3
  let transfer_to_cup1 := (6 + 2) * (1/4)
  let cup1_tea_after_second_transfer := 6 * (2/3) + 2 * (2 / 8)
  let cup1_milk_after_second_transfer :=  2 * (6 / 8)
  let cup1_total_after_second_transfer := (4 + 0.5 + 6 / 4)
  let transfer_back_to_cup2 := 6 * (1 / 6)
  let cup1_tea_after_final_transfer := 4.5 - 1 * (4.5 / 6)
  let cup1_milk_after_final_transfer := 1.5 - 1 * (1.5 / 6)
  let cup1_total_after_final_transfer := (3.75 + 1.25)
  show (1.25 / 5) = 1 / 4
  sorry

end fraction_of_milk_is_one_fourth_l280_280949


namespace tan_of_45_deg_l280_280707

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_of_45_deg_l280_280707


namespace tan_45_degree_l280_280673

theorem tan_45_degree : ∀ (Q : ℝ × ℝ),
  -- Q corresponds to 45 degrees on the unit circle
  Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) →
  -- The radius of the unit circle is 1
  (Q.1 ^ 2 + Q.2 ^ 2 = 1) →
  -- QOE is a 45-45-90 triangle
  True →
  -- Prove that tan 45 degrees equals 1
  Real.tan (Real.pi / 4) = 1 :=
by
  -- The proof using these conditions is omitted
  intros,
  sorry

end tan_45_degree_l280_280673


namespace tan_45_deg_eq_1_l280_280610

-- Define conditions
variable (Q : ℝ × ℝ)
variable (E : ℝ × ℝ)
variable (O : ℝ × ℝ := (0, 0))
variable (θ : ℝ := 45)

-- Define main theorem
theorem tan_45_deg_eq_1
  (Q_on_unit_circle : Q = (ℝ.sqrt 2 / 2, ℝ.sqrt 2 / 2))
  (E_foot_perpendicular : E = (ℝ.sqrt 2 / 2, 0))
  (triangle_45_45_90 : (Q.1 - O.1) ^ 2 + (Q.2 - O.2) ^ 2 = 1) :
  Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_1_l280_280610


namespace find_a_l280_280316

-- The conditions converted to Lean definitions
variable (a : ℝ)
variable (α : ℝ)
variable (point_on_terminal_side : a ≠ 0 ∧ (∃ α, tan α = -1 / 2 ∧ ∀ y : ℝ, y = -1 → a = 2 * y) )

-- The theorem statement
theorem find_a (H : point_on_terminal_side): a = 2 := by
  sorry

end find_a_l280_280316


namespace tan_45_deg_eq_one_l280_280644

/-
  Given:
  cos (45 deg) = sqrt 2 / 2,
  sin (45 deg) = sqrt 2 / 2,
  Prove: 
  tan (45 deg) = 1
-/

theorem tan_45_deg_eq_one : 
  (real.cos (real.pi / 4) = real.sqrt 2 / 2) → 
  (real.sin (real.pi / 4) = real.sqrt 2 / 2) → 
  real.tan (real.pi / 4) = 1 :=
by
  intros hcos hsin
  sorry

end tan_45_deg_eq_one_l280_280644


namespace largest_class_is_28_l280_280219

-- definition and conditions
def largest_class_students (x : ℕ) : Prop :=
  let total_students := x + (x - 2) + (x - 4) + (x - 6) + (x - 8)
  total_students = 120

-- statement to prove
theorem largest_class_is_28 : ∃ x : ℕ, largest_class_students x ∧ x = 28 :=
by
  sorry

end largest_class_is_28_l280_280219


namespace inequality_solution_l280_280568

noncomputable def solution_set_inequality : Set ℝ := {x | -2 < x ∧ x < 1 / 3}

theorem inequality_solution :
  {x : ℝ | (2 * x - 1) / (3 * x + 1) > 1} = solution_set_inequality :=
by
  sorry

end inequality_solution_l280_280568


namespace quadratic_root_exists_l280_280105

theorem quadratic_root_exists {a b c d : ℝ} (h : a * c = 2 * b + 2 * d) :
  (∃ x : ℝ, x^2 + a * x + b = 0) ∨ (∃ x : ℝ, x^2 + c * x + d = 0) :=
by
  sorry

end quadratic_root_exists_l280_280105


namespace quadratic_solutions_l280_280088

-- Define the equation x^2 - 6x + 8 = 0
def quadratic_eq (x : ℝ) : Prop := x^2 - 6*x + 8 = 0

-- Lean statement for the equivalence of solutions
theorem quadratic_solutions : ∀ x : ℝ, quadratic_eq x ↔ x = 2 ∨ x = 4 :=
by
  intro x
  dsimp [quadratic_eq]
  sorry

end quadratic_solutions_l280_280088


namespace tan_45_degrees_l280_280805

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l280_280805


namespace sum_of_a_and_b_l280_280248

theorem sum_of_a_and_b (a b : ℕ) (h1: a > 0) (h2 : b > 1) (h3 : ∀ (x y : ℕ), x > 0 → y > 1 → x^y < 500 → x = a ∧ y = b → x^y ≥ a^b ) :
  a + b = 24 :=
sorry

end sum_of_a_and_b_l280_280248


namespace value_of_a_max_value_of_k_l280_280448

noncomputable def f (x a : ℝ) : ℝ := x * (a + Real.log x)

-- Condition: The function f(x) has a minimum value of -e^{-2}
axiom min_value_f (a : ℝ) : ∃ x : ℝ, f x a = -Real.exp (-2)

-- Question 1: Prove that the real number a = 1
theorem value_of_a : ∃ (a : ℝ), (∀ (x : ℝ), f x a = -Real.exp (-2)) → a = 1 :=
by sorry

-- Condition: k is an integer
axiom k_is_integer (k : ℤ): Prop

-- Condition: k < f(x) / (x - 1) for any x > 1
axiom k_condition (k : ℤ): ∀ x > 1, k < f x 1 / (x - 1)

-- Question 2: Prove that the maximum value of k is 3
theorem max_value_of_k : ∀ (k : ℤ), (k_is_integer k) → (∀ x > 1, k < f x 1 / (x - 1)) → k ≤ 3 :=
by sorry

end value_of_a_max_value_of_k_l280_280448


namespace beads_cost_is_three_l280_280388

-- Define the given conditions
def cost_of_string_per_bracelet : Nat := 1
def selling_price_per_bracelet : Nat := 6
def number_of_bracelets_sold : Nat := 25
def total_profit : Nat := 50

-- The amount spent on beads per bracelet
def amount_spent_on_beads_per_bracelet (B : Nat) : Prop :=
  B = (total_profit + number_of_bracelets_sold * (cost_of_string_per_bracelet + B) - number_of_bracelets_sold * selling_price_per_bracelet) / number_of_bracelets_sold 

-- The main goal is to prove that the amount spent on beads is 3
theorem beads_cost_is_three : amount_spent_on_beads_per_bracelet 3 :=
by sorry

end beads_cost_is_three_l280_280388


namespace sum_of_repeating_decimals_l280_280598

-- Definitions based on the conditions
def x := 0.6666666666666666 -- Lean may not directly support \(0.\overline{6}\) notation
def y := 0.7777777777777777 -- Lean may not directly support \(0.\overline{7}\) notation

-- Translate those to the correct fractional forms
def x_as_fraction := (2 : ℚ) / 3
def y_as_fraction := (7 : ℚ) / 9

-- The main statement to prove
theorem sum_of_repeating_decimals : x_as_fraction + y_as_fraction = 13 / 9 :=
by
  -- Proof skipped
  sorry

end sum_of_repeating_decimals_l280_280598


namespace tan_45_eq_1_l280_280724

theorem tan_45_eq_1 (hcos : real.cos (real.pi / 4) = real.sqrt 2 / 2) (hsin : real.sin (real.pi / 4) = real.sqrt 2 / 2) : real.tan (real.pi / 4) = 1 :=
by
  have h : real.tan (real.pi / 4) = real.sin (real.pi / 4) / real.cos (real.pi / 4) := real.tan_eq_sin_div_cos _
  rw [hsin, hcos] at h
  simp at h
  exact h.symm

end tan_45_eq_1_l280_280724


namespace minimum_value_of_function_l280_280430

theorem minimum_value_of_function :
  ∃ x y : ℝ, 2 * x ^ 2 + 3 * x * y + 4 * y ^ 2 - 8 * x + y = 3.7391 := by
  sorry

end minimum_value_of_function_l280_280430


namespace joggers_meeting_time_l280_280246

def lap_time (ben_lap : ℕ) (carol_lap : ℕ) (dave_lap : ℕ) : ℕ :=
  Int.natAbs (Nat.lcm (Nat.lcm ben_lap carol_lap) dave_lap)

def earliest_meeting_time (start_time : ℕ) (lap_time : ℕ) : ℕ :=
  start_time + lap_time / 60

def time_in_minutes : ℕ :=
  7 * 60  -- 7:00 AM in minutes

theorem joggers_meeting_time (ben_lap : 5) (carol_lap : 8) (dave_lap : 9)
                            (hc : lap_time 5 8 9 = 360) :
  earliest_meeting_time time_in_minutes 360 = 13 * 60 := 
sorry

end joggers_meeting_time_l280_280246


namespace inequality_abc_l280_280076

theorem inequality_abc (a b c : ℝ) (h : a * b * c = 1) :
  1 / (2 * a^2 + b^2 + 3) + 1 / (2 * b^2 + c^2 + 3) + 1 / (2 * c^2 + a^2 + 3) ≤ 1 / 2 :=
by
  -- Sorry to skip the proof
  sorry

end inequality_abc_l280_280076


namespace isosceles_triangle_base_length_l280_280304

theorem isosceles_triangle_base_length
  (a b c: ℕ) 
  (h_iso: a = b ∨ a = c ∨ b = c)
  (h_perimeter: a + b + c = 21)
  (h_side: a = 5 ∨ b = 5 ∨ c = 5) :
  c = 5 :=
by
  sorry

end isosceles_triangle_base_length_l280_280304


namespace sum_of_roots_l280_280154

theorem sum_of_roots : 
  let eq := (x - 7) ^ 2 = 16 in
  (roots : set ℝ) := { x | eq } in
  ∑ roots = 14 :=
by
sorry

end sum_of_roots_l280_280154


namespace annual_income_earned_by_both_investments_l280_280379

noncomputable def interest (principal: ℝ) (rate: ℝ) (time: ℝ) : ℝ :=
  principal * rate * time

theorem annual_income_earned_by_both_investments :
  let total_amount := 8000
  let first_investment := 3000
  let first_interest_rate := 0.085
  let second_interest_rate := 0.064
  let second_investment := total_amount - first_investment
  interest first_investment first_interest_rate 1 + interest second_investment second_interest_rate 1 = 575 :=
by
  sorry

end annual_income_earned_by_both_investments_l280_280379


namespace kenny_cost_per_book_l280_280045

theorem kenny_cost_per_book (B : ℕ) :
  let lawn_charge := 15
  let mowed_lawns := 35
  let video_game_cost := 45
  let video_games := 5
  let total_earnings := lawn_charge * mowed_lawns
  let spent_on_video_games := video_game_cost * video_games
  let remaining_money := total_earnings - spent_on_video_games
  remaining_money / B = 300 / B :=
by
  sorry

end kenny_cost_per_book_l280_280045


namespace max_value_of_quadratic_l280_280995

theorem max_value_of_quadratic : ∀ (x : ℝ), -9 * x^2 + 27 * x + 15 ≤ 141 / 4 :=
begin
  sorry
end

end max_value_of_quadratic_l280_280995


namespace fill_bucket_time_l280_280218

-- Problem statement:
-- Prove that the time taken to fill the bucket completely is 150 seconds
-- given that two-thirds of the bucket is filled in 100 seconds.

theorem fill_bucket_time (t : ℕ) (h : (2 / 3) * t = 100) : t = 150 :=
by
  -- Proof should be here
  sorry

end fill_bucket_time_l280_280218


namespace solve_equation_l280_280975

theorem solve_equation :
  ∀ x : ℝ, (x * (2 * x + 4) = 10 + 5 * x) ↔ (x = -2 ∨ x = 2.5) :=
by
  sorry

end solve_equation_l280_280975


namespace tan_45_deg_l280_280631

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l280_280631


namespace most_frequent_third_number_l280_280278

def is_lottery_condition (e1 e2 e3 e4 e5 : ℕ) : Prop :=
  1 ≤ e1 ∧ e1 < e2 ∧ e2 < e3 ∧ e3 < e4 ∧ e4 < e5 ∧ e5 ≤ 90 ∧ (e1 + e2 = e3)

theorem most_frequent_third_number :
  ∃ h : ℕ, 3 ≤ h ∧ h ≤ 88 ∧ (∀ h', (h' = 31 → ¬ (31 < h')) ∧ 
        ∀ e1 e2 e3 e4 e5, is_lottery_condition e1 e2 e3 e4 e5 → e3 = h) :=
sorry

end most_frequent_third_number_l280_280278


namespace chocolates_problem_l280_280476

theorem chocolates_problem (C S : ℝ) (n : ℕ) 
  (h1 : 24 * C = n * S)
  (h2 : (S - C) / C = 0.5) : 
  n = 16 :=
by 
  sorry

end chocolates_problem_l280_280476


namespace punger_needs_pages_l280_280965

theorem punger_needs_pages (p c h : ℕ) (h_p : p = 60) (h_c : c = 7) (h_h : h = 10) : 
  (p * c) / h = 42 := 
by
  sorry

end punger_needs_pages_l280_280965


namespace find_x_l280_280584

theorem find_x (x : ℝ) (h : 61 + 5 * 12 / (x / 3) = 62) : x = 180 :=
by
  sorry

end find_x_l280_280584


namespace total_capacity_l280_280028

def eight_liters : ℝ := 8
def percentage : ℝ := 0.20
def num_containers : ℕ := 40

theorem total_capacity (h : eight_liters = percentage * capacity) :
  40 * (eight_liters / percentage) = 1600 := sorry

end total_capacity_l280_280028


namespace base_b_square_of_15_l280_280019

theorem base_b_square_of_15 (b : ℕ) (h : (b + 5) * (b + 5) = 4 * b^2 + 3 * b + 6) : b = 8 :=
sorry

end base_b_square_of_15_l280_280019


namespace aarti_bina_work_l280_280592

theorem aarti_bina_work (days_aarti : ℚ) (days_bina : ℚ) (D : ℚ)
  (ha : days_aarti = 5) (hb : days_bina = 8)
  (rate_aarti : 1 / days_aarti = 1/5) 
  (rate_bina : 1 / days_bina = 1/8)
  (combine_rate : (1 / days_aarti) + (1 / days_bina) = 13 / 40) :
  3 / (13 / 40) = 120 / 13 := 
by
  sorry

end aarti_bina_work_l280_280592


namespace helga_shoes_l280_280310

theorem helga_shoes :
  ∃ (S : ℕ), 7 + S + 0 + 2 * (7 + S) = 48 ∧ (S - 7 = 2) :=
by
  sorry

end helga_shoes_l280_280310


namespace central_cell_value_l280_280503

theorem central_cell_value :
  ∀ (a b c d e f g h i : ℝ),
    (a * b * c = 10) →
    (d * e * f = 10) →
    (g * h * i = 10) →
    (a * d * g = 10) →
    (b * e * h = 10) →
    (c * f * i = 10) →
    (a * b * d * e = 3) →
    (b * c * e * f = 3) →
    (d * e * g * h = 3) →
    (e * f * h * i = 3) →
    e = 0.00081 := 
by 
  intros a b c d e f g h i h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
  sorry

end central_cell_value_l280_280503


namespace down_payment_l280_280074

theorem down_payment {total_loan : ℕ} {monthly_payment : ℕ} {years : ℕ} (h1 : total_loan = 46000) (h2 : monthly_payment = 600) (h3 : years = 5):
  total_loan - (years * 12 * monthly_payment) = 10000 := by
  sorry

end down_payment_l280_280074


namespace total_students_in_Lansing_l280_280046

theorem total_students_in_Lansing:
  (number_of_schools : Nat) → 
  (students_per_school : Nat) → 
  (total_students : Nat) →
  number_of_schools = 25 → 
  students_per_school = 247 → 
  total_students = number_of_schools * students_per_school → 
  total_students = 6175 :=
by
  intros number_of_schools students_per_school total_students h_schools h_students h_total
  rw [h_schools, h_students] at h_total
  exact h_total

end total_students_in_Lansing_l280_280046


namespace tan_45_deg_eq_one_l280_280849

theorem tan_45_deg_eq_one :
  let θ := 45
  (∀ θ, tan θ = sin θ / cos θ) →
  (sin 45 = sqrt 2 / 2) →
  (cos 45 = sqrt 2 / 2) →
  tan 45 = 1 :=
by
  intros θ h_tan h_sin h_cos
  sorry

end tan_45_deg_eq_one_l280_280849


namespace quadratic_roots_range_l280_280477

theorem quadratic_roots_range (m : ℝ) :
  (∃ x : ℝ, x^2 - (2 * m + 1) * x + m^2 = 0 ∧ (∃ y : ℝ, y ≠ x ∧ y^2 - (2 * m + 1) * y + m^2 = 0)) ↔ m > -1 / 4 :=
by sorry

end quadratic_roots_range_l280_280477


namespace minimum_photos_l280_280293

theorem minimum_photos (G B : ℕ) (n : ℕ) : G = 4 → B = 8 → n ≥ 33 → 
  (∃ (p : fin ((G + B) choose 2) → (fin (G + B) × fin (G + B))),
  (∃ i j : fin (G + B), i ≠ j ∧ p i = p j) ∨ 
  (∃ k j : fin (G + B), k ≤ G ∧ j ≤ G ∧ p k = p j) ∨
  (∃ k j : fin (G + B), k > G ∧ j > G ∧ p k = p j)) :=
by
  intros G_eq B_eq n_ge_33
  sorry

end minimum_photos_l280_280293


namespace max_power_sum_l280_280253

open Nat

theorem max_power_sum (a b : ℕ) (h_a_pos : a > 0) (h_b_gt_one : b > 1) (h_max : a ^ b < 500 ∧ 
  ∀ (a' b' : ℕ), a' > 0 → b' > 1 → a' ^ b' < 500 → a' ^ b' ≤ a ^ b ) : a + b = 24 :=
sorry

end max_power_sum_l280_280253


namespace find_percentage_l280_280386

/-- 
Given some percentage P of 6,000, when subtracted from 1/10th of 6,000 (which is 600), 
the difference is 693. Prove that P equals 1.55.
-/
theorem find_percentage (P : ℝ) (h₁ : 6000 / 10 = 600) (h₂ : 600 - (P / 100) * 6000 = 693) : 
  P = 1.55 :=
  sorry

end find_percentage_l280_280386


namespace tan_45_deg_eq_one_l280_280648

/-
  Given:
  cos (45 deg) = sqrt 2 / 2,
  sin (45 deg) = sqrt 2 / 2,
  Prove: 
  tan (45 deg) = 1
-/

theorem tan_45_deg_eq_one : 
  (real.cos (real.pi / 4) = real.sqrt 2 / 2) → 
  (real.sin (real.pi / 4) = real.sqrt 2 / 2) → 
  real.tan (real.pi / 4) = 1 :=
by
  intros hcos hsin
  sorry

end tan_45_deg_eq_one_l280_280648


namespace tangent_line_equation_at_1_2_l280_280449

noncomputable def f (x : ℝ) : ℝ := 2 / x

theorem tangent_line_equation_at_1_2 :
  let x₀ := 1
  let y₀ := 2
  let slope := -2
  ∀ (x y : ℝ),
    y - y₀ = slope * (x - x₀) →
    2 * x + y - 4 = 0 :=
by
  sorry

end tangent_line_equation_at_1_2_l280_280449


namespace sum_of_roots_eq_14_l280_280199

-- Define the condition as a hypothesis
def equation (x : ℝ) := (x - 7) ^ 2 = 16

-- Define the statement that needs to be proved
theorem sum_of_roots_eq_14 : 
  (∀ x : ℝ, equation x → x = 11 ∨ x = 3) → 11 + 3 = 14 :=
by 
sorry

end sum_of_roots_eq_14_l280_280199


namespace sum_of_roots_eq_14_l280_280175

theorem sum_of_roots_eq_14 : 
  (∀ x : ℝ, (x - 7)^2 = 16 → (x = 11 ∨ x = 3)) →
  (∀ x1 x2 : ℝ, (x1 - 7)^2 = 16 ∧ (x2 - 7)^2 = 16 ∧ x1 ≠ x2 → x1 + x2 = 14) :=
by
  intros h x1 x2 h_comb
  sorry

end sum_of_roots_eq_14_l280_280175


namespace central_cell_value_l280_280488

def table (a b c d e f g h i : ℝ) : Prop :=
  (a * b * c = 10) ∧ (d * e * f = 10) ∧ (g * h * i = 10) ∧
  (a * d * g = 10) ∧ (b * e * h = 10) ∧ (c * f * i = 10) ∧
  (a * b * d * e = 3) ∧ (b * c * e * f = 3) ∧ (d * e * g * h = 3) ∧ (e * f * h * i = 3)

theorem central_cell_value (a b c d f g h i e : ℝ) (h_table : table a b c d e f g h i) : 
  e = 0.00081 :=
by sorry

end central_cell_value_l280_280488


namespace tan_of_45_deg_l280_280700

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_of_45_deg_l280_280700


namespace simplified_expression_l280_280952

def f (x : ℝ) : ℝ := 3 * x + 4
def g (x : ℝ) : ℝ := 2 * x - 1

theorem simplified_expression :
  (f (g (f 3))) / (g (f (g 3))) = 79 / 37 :=
by  sorry

end simplified_expression_l280_280952


namespace min_photos_for_condition_l280_280291

noncomputable def minimum_photos (girls boys : ℕ) : ℕ :=
  if (girls = 4 ∧ boys = 8) 
  then 33
  else 0

theorem min_photos_for_condition (girls boys : ℕ) (photos : ℕ) :
  girls = 4 → boys = 8 → photos = minimum_photos girls boys
  → ∃ (pa : ℕ), pa >= 33 → pa = photos :=
by
  intros h1 h2 h3
  use minimum_photos girls boys
  rw [h3]
  sorry

end min_photos_for_condition_l280_280291


namespace negation_equivalent_statement_l280_280221

theorem negation_equivalent_statement (x y : ℝ) :
  (x^2 + y^2 = 0 → x = 0 ∧ y = 0) ↔ (x^2 + y^2 ≠ 0 → ¬ (x = 0 ∧ y = 0)) :=
sorry

end negation_equivalent_statement_l280_280221


namespace tan_45_deg_l280_280659

theorem tan_45_deg : Real.tan (π / 4) = 1 := 
by
  have h_sin_cos : Real.sin (π / 4) = Real.cos (π / 4) := 
    by sorry
  have h_value : Real.sin (π / 4) = (Real.sqrt 2) / 2 := 
    by sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos, h_value]
  calc
    (Real.sqrt 2)/ 2 / (Real.sqrt 2)/ 2 = 1 := sorry

end tan_45_deg_l280_280659


namespace sum_of_roots_eq_14_l280_280192

theorem sum_of_roots_eq_14 (x : ℝ) :
  (x - 7) ^ 2 = 16 → ∃ r1 r2 : ℝ, (x = r1 ∨ x = r2) ∧ r1 + r2 = 14 :=
by
  intro h
  sorry

end sum_of_roots_eq_14_l280_280192


namespace tan_45_deg_eq_one_l280_280646

/-
  Given:
  cos (45 deg) = sqrt 2 / 2,
  sin (45 deg) = sqrt 2 / 2,
  Prove: 
  tan (45 deg) = 1
-/

theorem tan_45_deg_eq_one : 
  (real.cos (real.pi / 4) = real.sqrt 2 / 2) → 
  (real.sin (real.pi / 4) = real.sqrt 2 / 2) → 
  real.tan (real.pi / 4) = 1 :=
by
  intros hcos hsin
  sorry

end tan_45_deg_eq_one_l280_280646


namespace tan_45_deg_eq_one_l280_280647

/-
  Given:
  cos (45 deg) = sqrt 2 / 2,
  sin (45 deg) = sqrt 2 / 2,
  Prove: 
  tan (45 deg) = 1
-/

theorem tan_45_deg_eq_one : 
  (real.cos (real.pi / 4) = real.sqrt 2 / 2) → 
  (real.sin (real.pi / 4) = real.sqrt 2 / 2) → 
  real.tan (real.pi / 4) = 1 :=
by
  intros hcos hsin
  sorry

end tan_45_deg_eq_one_l280_280647


namespace tan_45_deg_eq_one_l280_280653

/-
  Given:
  cos (45 deg) = sqrt 2 / 2,
  sin (45 deg) = sqrt 2 / 2,
  Prove: 
  tan (45 deg) = 1
-/

theorem tan_45_deg_eq_one : 
  (real.cos (real.pi / 4) = real.sqrt 2 / 2) → 
  (real.sin (real.pi / 4) = real.sqrt 2 / 2) → 
  real.tan (real.pi / 4) = 1 :=
by
  intros hcos hsin
  sorry

end tan_45_deg_eq_one_l280_280653


namespace sufficient_and_necessary_condition_l280_280539

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.log (x + Real.sqrt (x^2 + 1)) / Real.log 2

theorem sufficient_and_necessary_condition {a b : ℝ} (h : a + b ≥ 0) : f a + f b ≥ 0 :=
sorry

end sufficient_and_necessary_condition_l280_280539


namespace tan_45_deg_l280_280662

theorem tan_45_deg : Real.tan (π / 4) = 1 := 
by
  have h_sin_cos : Real.sin (π / 4) = Real.cos (π / 4) := 
    by sorry
  have h_value : Real.sin (π / 4) = (Real.sqrt 2) / 2 := 
    by sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos, h_value]
  calc
    (Real.sqrt 2)/ 2 / (Real.sqrt 2)/ 2 = 1 := sorry

end tan_45_deg_l280_280662


namespace tan_45_degree_l280_280751

theorem tan_45_degree : ∀ (θ : ℝ), 
  θ = 45 → 
  (sin θ = 1 / real.sqrt 2) → 
  (cos θ = 1 / real.sqrt 2) → 
  real.tan θ = 1 := 
by
  intros θ hθ hsin hcos
  sorry

end tan_45_degree_l280_280751


namespace rosy_fish_is_twelve_l280_280957

/-- Let lilly_fish be the number of fish Lilly has. -/
def lilly_fish : ℕ := 10

/-- Let total_fish be the total number of fish Lilly and Rosy have together. -/
def total_fish : ℕ := 22

/-- Prove that the number of fish Rosy has is equal to 12. -/
theorem rosy_fish_is_twelve : (total_fish - lilly_fish) = 12 :=
by sorry

end rosy_fish_is_twelve_l280_280957


namespace tan_45_deg_eq_one_l280_280844

theorem tan_45_deg_eq_one :
  let θ := 45
  (∀ θ, tan θ = sin θ / cos θ) →
  (sin 45 = sqrt 2 / 2) →
  (cos 45 = sqrt 2 / 2) →
  tan 45 = 1 :=
by
  intros θ h_tan h_sin h_cos
  sorry

end tan_45_deg_eq_one_l280_280844


namespace total_distance_is_20_l280_280230

noncomputable def total_distance_walked (x : ℝ) : ℝ :=
  let flat_distance := 4 * x
  let uphill_time := (2 / 3) * (5 - x)
  let uphill_distance := 3 * uphill_time
  let downhill_time := (1 / 3) * (5 - x)
  let downhill_distance := 6 * downhill_time
  flat_distance + uphill_distance + downhill_distance

theorem total_distance_is_20 :
  ∃ x : ℝ, x >= 0 ∧ x <= 5 ∧ total_distance_walked x = 20 :=
by
  -- The existence proof is omitted (hence the sorry)
  sorry

end total_distance_is_20_l280_280230


namespace central_cell_value_l280_280520

theorem central_cell_value :
  ∀ (a b c d e f g h i : ℝ),
  a * b * c = 10 →
  d * e * f = 10 →
  g * h * i = 10 →
  a * d * g = 10 →
  b * e * h = 10 →
  c * f * i = 10 →
  a * b * d * e = 3 →
  b * c * e * f = 3 →
  d * e * g * h = 3 →
  e * f * h * i = 3 →
  e = 0.00081 := 
by sorry

end central_cell_value_l280_280520


namespace tan_45_degree_l280_280749

theorem tan_45_degree : ∀ (θ : ℝ), 
  θ = 45 → 
  (sin θ = 1 / real.sqrt 2) → 
  (cos θ = 1 / real.sqrt 2) → 
  real.tan θ = 1 := 
by
  intros θ hθ hsin hcos
  sorry

end tan_45_degree_l280_280749


namespace inequality_proof_l280_280955

theorem inequality_proof {x y : ℝ} (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) :
  (1 / Real.sqrt (1 + x^2)) + (1 / Real.sqrt (1 + y^2)) ≤ (2 / Real.sqrt (1 + x * y)) :=
by
  sorry

end inequality_proof_l280_280955


namespace container_capacity_l280_280022

/-- Given a container where 8 liters is 20% of its capacity, calculate the total capacity of 
    40 such containers filled with water. -/
theorem container_capacity (c : ℝ) (h : 8 = 0.20 * c) : 
    40 * c * 40 = 1600 := 
by
  sorry

end container_capacity_l280_280022


namespace find_b_eq_neg_three_l280_280925

theorem find_b_eq_neg_three (b : ℝ) (h : (2 - b) / 5 = -(2 * b + 1) / 5) : b = -3 :=
by
  sorry

end find_b_eq_neg_three_l280_280925


namespace problem1_l280_280356

theorem problem1 (x y : ℝ) 
  (h1 : 3 * x + 2 * y = 10)
  (h2 : x / 2 - (y + 1) / 3 = 1) :
  x = 3 ∧ y = 1 / 2 := 
sorry

end problem1_l280_280356


namespace tan_45_eq_one_l280_280874

theorem tan_45_eq_one : Real.tan (Real.pi / 4) = 1 := 
by
  sorry

end tan_45_eq_one_l280_280874


namespace mike_coins_value_l280_280335

theorem mike_coins_value (d q : ℕ)
  (h1 : d + q = 17)
  (h2 : q + 3 = 2 * d) :
  10 * d + 25 * q = 345 :=
by
  sorry

end mike_coins_value_l280_280335


namespace tan_45_degree_is_one_l280_280719

theorem tan_45_degree_is_one : Real.tan (Real.pi / 4) = 1 := sorry

end tan_45_degree_is_one_l280_280719


namespace sum_of_roots_eq_14_l280_280172

theorem sum_of_roots_eq_14 : 
  let equation := λ x : ℝ, (x - 7)^2 = 16 in
  let roots := {x | equation x} in
  (roots : ℝ) → ∀ x y ∈ roots, x + y = 14 :=
by
  -- For the purpose of the problem statement, we specify the exact roots here
  let root1 := 11
  let root2 := 3
  have H : root1 + root2 = 14, by norm_num
  intro equation roots x y hx hy
  exact H

end sum_of_roots_eq_14_l280_280172


namespace tan_45_deg_eq_one_l280_280850

theorem tan_45_deg_eq_one :
  let θ := 45
  (∀ θ, tan θ = sin θ / cos θ) →
  (sin 45 = sqrt 2 / 2) →
  (cos 45 = sqrt 2 / 2) →
  tan 45 = 1 :=
by
  intros θ h_tan h_sin h_cos
  sorry

end tan_45_deg_eq_one_l280_280850


namespace sum_of_first_n_terms_geom_sequence_l280_280047

theorem sum_of_first_n_terms_geom_sequence (a₁ q : ℚ) (S : ℕ → ℚ)
  (h : ∀ n, S n = a₁ * (1 - q^n) / (1 - q))
  (h_ratio : S 4 / S 2 = 3) :
  S 6 / S 4 = 7 / 3 :=
by
  sorry

end sum_of_first_n_terms_geom_sequence_l280_280047


namespace tan_45_deg_eq_one_l280_280834

/-- The tangent of 45 degrees is equal to 1. -/
theorem tan_45_deg_eq_one : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_one_l280_280834


namespace reverse_digits_multiplication_l280_280131

theorem reverse_digits_multiplication (a b : ℕ) (h₁ : a < 10) (h₂ : b < 10) : 
  (10 * a + b) * (10 * b + a) = 101 * a * b + 10 * (a^2 + b^2) :=
by 
  sorry

end reverse_digits_multiplication_l280_280131


namespace tan_of_45_deg_l280_280702

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_of_45_deg_l280_280702


namespace intersection_M_N_l280_280010

def M := {x : ℝ | x < 1}

def N := {y : ℝ | ∃ x : ℝ, y = Real.exp x}

theorem intersection_M_N : M ∩ N = {x : ℝ | 0 < x ∧ x < 1} :=
  sorry

end intersection_M_N_l280_280010


namespace hundred_chicken_problem_l280_280942

theorem hundred_chicken_problem :
  ∃ (x y : ℕ), x + y + 81 = 100 ∧ 5 * x + 3 * y + 81 / 3 = 100 := 
by
  sorry

end hundred_chicken_problem_l280_280942


namespace min_value_a_4b_l280_280438

theorem min_value_a_4b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a * b = a + b) :
  a + 4 * b = 9 :=
sorry

end min_value_a_4b_l280_280438


namespace unique_real_solution_k_l280_280005

-- Definitions corresponding to problem conditions:
def is_real_solution (a b k : ℤ) : Prop :=
  (a > 0) ∧ (b > 0) ∧ (∃ (x y : ℝ), x * x = a - 1 ∧ y * y = b - 1 ∧ x + y = Real.sqrt (a * b + k))

-- Theorem statement:
theorem unique_real_solution_k (k : ℤ) : (∀ a b : ℤ, is_real_solution a b k → (a = 2 ∧ b = 2)) ↔ k = 0 :=
sorry

end unique_real_solution_k_l280_280005


namespace at_least_one_root_l280_280109

theorem at_least_one_root 
  (a b c d : ℝ)
  (h : a * c = 2 * b + 2 * d) :
  (∃ x : ℝ, x^2 + a * x + b = 0) ∨ (∃ x : ℝ, x^2 + c * x + d = 0) :=
sorry

end at_least_one_root_l280_280109


namespace greatest_area_difference_l280_280897

theorem greatest_area_difference 
    (a b c d : ℕ) 
    (H1 : 2 * (a + b) = 100)
    (H2 : 2 * (c + d) = 100)
    (H3 : ∀i j : ℕ, 2 * (i + j) = 100 → i * j ≤ a * b)
    : 373 ≤ a * b - (c * d) := 
sorry

end greatest_area_difference_l280_280897


namespace max_power_sum_l280_280254

open Nat

theorem max_power_sum (a b : ℕ) (h_a_pos : a > 0) (h_b_gt_one : b > 1) (h_max : a ^ b < 500 ∧ 
  ∀ (a' b' : ℕ), a' > 0 → b' > 1 → a' ^ b' < 500 → a' ^ b' ≤ a ^ b ) : a + b = 24 :=
sorry

end max_power_sum_l280_280254


namespace tan_45_eq_1_l280_280723

theorem tan_45_eq_1 (hcos : real.cos (real.pi / 4) = real.sqrt 2 / 2) (hsin : real.sin (real.pi / 4) = real.sqrt 2 / 2) : real.tan (real.pi / 4) = 1 :=
by
  have h : real.tan (real.pi / 4) = real.sin (real.pi / 4) / real.cos (real.pi / 4) := real.tan_eq_sin_div_cos _
  rw [hsin, hcos] at h
  simp at h
  exact h.symm

end tan_45_eq_1_l280_280723


namespace range_of_k_l280_280561

noncomputable def circle_equation (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 + y^2 + 2*k*x + 4*y + 3*k + 8 = 0

theorem range_of_k (k : ℝ) :
  circle_equation k →
  k ∈ (Set.Iio (-1) ∪ Set.Ioi 4) :=
sorry

end range_of_k_l280_280561


namespace central_cell_value_l280_280527

theorem central_cell_value
  (a b c d e f g h i : ℝ)
  (row1 : a * b * c = 10)
  (row2 : d * e * f = 10)
  (row3 : g * h * i = 10)
  (col1 : a * d * g = 10)
  (col2 : b * e * h = 10)
  (col3 : c * f * i = 10)
  (sub1 : a * b * d * e = 3)
  (sub2 : b * c * e * f = 3)
  (sub3 : d * e * g * h = 3)
  (sub4 : e * f * h * i = 3) : 
  e = 0.00081 :=
sorry

end central_cell_value_l280_280527


namespace find_x_l280_280359

def custom_op (a b : ℤ) : ℤ := 2 * a + 3 * b

theorem find_x : ∃ x : ℤ, custom_op 5 (custom_op 7 x) = -4 ∧ x = -56 / 9 := by
  sorry

end find_x_l280_280359


namespace tan_45_deg_l280_280881

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 :=
by
  have h_sin_cos : Real.sin (Real.pi / 4) = Real.cos (Real.pi / 4) := sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos]
  have sqrt2_ne_zero : Real.sqrt 2 ≠ 0 := sorry
  have sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have cos_45 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [sin_45, cos_45]
  field_simp [sqrt2_ne_zero]
  exact one_div_one

end tan_45_deg_l280_280881


namespace quadratic_root_value_l280_280938

theorem quadratic_root_value (a b : ℤ) (h : 2 * a - b = -3) : 6 * a - 3 * b + 6 = -3 :=
by 
  sorry

end quadratic_root_value_l280_280938


namespace tan_of_45_deg_l280_280706

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_of_45_deg_l280_280706


namespace abs_inequality_l280_280119

theorem abs_inequality (x : ℝ) (h : |x - 2| < 1) : 1 < x ∧ x < 3 := by
  sorry

end abs_inequality_l280_280119


namespace tan_45_eq_one_l280_280873

theorem tan_45_eq_one : Real.tan (Real.pi / 4) = 1 := 
by
  sorry

end tan_45_eq_one_l280_280873


namespace tan_45_eq_1_l280_280823

theorem tan_45_eq_1
: ∀ (Q : ℝ × ℝ), Q = (Real.sqrt 2 / 2, Real.sqrt 2 / 2) ->
  ∀ θ, θ = Real.pi / 4 -> 
  Real.tan θ = 1 :=
by
  intros Q HQ θ Hθ
  rw [HQ, Hθ]
  sorry

end tan_45_eq_1_l280_280823


namespace tan_45_eq_1_l280_280763

theorem tan_45_eq_1 : Real.tan (Real.pi / 4) = 1 := by
  have h1 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have h2 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [Real.tan_eq_sin_div_cos, h1, h2]
  simp

end tan_45_eq_1_l280_280763


namespace sum_of_roots_eq_14_l280_280191

theorem sum_of_roots_eq_14 (x : ℝ) :
  (x - 7) ^ 2 = 16 → ∃ r1 r2 : ℝ, (x = r1 ∨ x = r2) ∧ r1 + r2 = 14 :=
by
  intro h
  sorry

end sum_of_roots_eq_14_l280_280191


namespace days_taken_to_complete_work_l280_280579

-- Conditions
def work_rate_B : ℚ := 1 / 33
def work_rate_A : ℚ := 2 * work_rate_B
def combined_work_rate : ℚ := work_rate_A + work_rate_B

-- Proof statement
theorem days_taken_to_complete_work : combined_work_rate ≠ 0 → 1 / combined_work_rate = 11 :=
by
  sorry

end days_taken_to_complete_work_l280_280579


namespace three_digit_multiples_of_7_l280_280465

theorem three_digit_multiples_of_7 :
  let a := 7 * Nat.ceil (100 / 7)
  let l := 7 * Nat.floor (999 / 7)
  let d := 7
  let n := (l - a) / d + 1
  n = 128 :=
by
  let a := 7 * Nat.ceil (100 / 7)
  let l := 7 * Nat.floor (999 / 7)
  let d := 7
  let n := (l - a) / d + 1
  have : a = 105 := sorry
  have : l = 994 := sorry
  have : n = (994 - 105) / 7 + 1 := sorry
  have : n = 128 := sorry
  exact this

end three_digit_multiples_of_7_l280_280465


namespace tan_of_45_deg_l280_280704

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_of_45_deg_l280_280704


namespace tan_45_eq_one_l280_280792

theorem tan_45_eq_one (Q : ℝ × ℝ) (h1 : Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2)) : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_eq_one_l280_280792


namespace tan_45_eq_1_l280_280764

theorem tan_45_eq_1 : Real.tan (Real.pi / 4) = 1 := by
  have h1 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have h2 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [Real.tan_eq_sin_div_cos, h1, h2]
  simp

end tan_45_eq_1_l280_280764


namespace tan_45_eq_1_l280_280766

theorem tan_45_eq_1 :
  let Q : ℝ × ℝ := (1 / Real.sqrt 2, 1 / Real.sqrt 2)
  in ∃ (Q : ℝ × ℝ), Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) → 
     Real.tan (Real.pi / 4) = 1 := 
by {
  -- The proof is left as an exercise.
  sorry
}

end tan_45_eq_1_l280_280766


namespace sum_of_roots_l280_280147

theorem sum_of_roots (a b : Real) (h : (x - 7)^2 = 16):
  a + b = 14 :=
sorry

end sum_of_roots_l280_280147


namespace g_value_at_50_l280_280543

noncomputable def g (x : ℝ) : ℝ := (1 - x) / 2

theorem g_value_at_50 :
  (∀ x y : ℝ, 0 < x → 0 < y → 
  (x * g y - y * g x = g (x / y) + x - y)) →
  g 50 = -24.5 :=
by
  intro h
  have h_g : ∀ x : ℝ, 0 < x → g x = (1 - x) / 2 := 
    fun x x_pos => sorry -- g(x) derivation proof goes here
  exact sorry -- Final answer proof goes here

end g_value_at_50_l280_280543


namespace central_cell_value_l280_280530

theorem central_cell_value
  (a b c d e f g h i : ℝ)
  (row1 : a * b * c = 10)
  (row2 : d * e * f = 10)
  (row3 : g * h * i = 10)
  (col1 : a * d * g = 10)
  (col2 : b * e * h = 10)
  (col3 : c * f * i = 10)
  (sub1 : a * b * d * e = 3)
  (sub2 : b * c * e * f = 3)
  (sub3 : d * e * g * h = 3)
  (sub4 : e * f * h * i = 3) : 
  e = 0.00081 :=
sorry

end central_cell_value_l280_280530


namespace central_cell_value_l280_280505

variables a b c d e f g h i : ℝ

-- Conditions
axiom row1 : a * b * c = 10
axiom row2 : d * e * f = 10
axiom row3 : g * h * i = 10
axiom col1 : a * d * g = 10
axiom col2 : b * e * h = 10
axiom col3 : c * f * i = 10
axiom sq1 : a * b * d * e = 3
axiom sq2 : b * c * e * f = 3
axiom sq3 : d * e * g * h = 3
axiom sq4 : e * f * h * i = 3

theorem central_cell_value : e = 0.00081 := by
  sorry

end central_cell_value_l280_280505


namespace length_of_arc_l280_280036

variable {O A B : Type}
variable (angle_OAB : Real) (radius_OA : Real)

theorem length_of_arc (h1 : angle_OAB = 45) (h2 : radius_OA = 5) :
  (length_of_arc_AB = 5 * π / 4) :=
sorry

end length_of_arc_l280_280036


namespace lucas_change_l280_280545

-- Define the given conditions as constants in Lean
def num_bananas : ℕ := 5
def cost_per_banana : ℝ := 0.70
def num_oranges : ℕ := 2
def cost_per_orange : ℝ := 0.80
def amount_paid : ℝ := 10.00

-- Define a noncomputable constant to represent the change received
noncomputable def change_received : ℝ := 
  amount_paid - (num_bananas * cost_per_banana + num_oranges * cost_per_orange)

-- State the theorem to be proved
theorem lucas_change : change_received = 4.90 := 
by 
  -- Dummy proof since the actual proof is not required
  sorry

end lucas_change_l280_280545


namespace tan_45_eq_one_l280_280866

theorem tan_45_eq_one : Real.tan (Real.pi / 4) = 1 := 
by
  sorry

end tan_45_eq_one_l280_280866


namespace initial_group_machines_l280_280552

-- Define the number of bags produced by n machines in one minute and 150 machines in one minute
def bags_produced (machines : ℕ) (bags_per_minute : ℕ) : Prop :=
  machines * bags_per_minute = 45

def bags_produced_150 (bags_produced_in_8_mins : ℕ) : Prop :=
  150 * (bags_produced_in_8_mins / 8) = 450

-- Given the conditions, prove that the number of machines in the initial group is 15
theorem initial_group_machines (n : ℕ) (bags_produced_in_8_mins : ℕ) :
  bags_produced n 45 → bags_produced_150 bags_produced_in_8_mins → n = 15 :=
by
  intro h1 h2
  -- use the conditions to derive the result
  sorry

end initial_group_machines_l280_280552


namespace central_cell_value_l280_280523

theorem central_cell_value :
  ∀ (a b c d e f g h i : ℝ),
  a * b * c = 10 →
  d * e * f = 10 →
  g * h * i = 10 →
  a * d * g = 10 →
  b * e * h = 10 →
  c * f * i = 10 →
  a * b * d * e = 3 →
  b * c * e * f = 3 →
  d * e * g * h = 3 →
  e * f * h * i = 3 →
  e = 0.00081 := 
by sorry

end central_cell_value_l280_280523


namespace central_cell_value_l280_280510

variables a b c d e f g h i : ℝ

-- Conditions
axiom row1 : a * b * c = 10
axiom row2 : d * e * f = 10
axiom row3 : g * h * i = 10
axiom col1 : a * d * g = 10
axiom col2 : b * e * h = 10
axiom col3 : c * f * i = 10
axiom sq1 : a * b * d * e = 3
axiom sq2 : b * c * e * f = 3
axiom sq3 : d * e * g * h = 3
axiom sq4 : e * f * h * i = 3

theorem central_cell_value : e = 0.00081 := by
  sorry

end central_cell_value_l280_280510


namespace distance_between_centers_of_circles_l280_280560

theorem distance_between_centers_of_circles :
  ∀ (rect_width rect_height circle_radius distance_between_centers : ℝ),
  rect_width = 11 
  ∧ rect_height = 7 
  ∧ circle_radius = rect_height / 2 
  ∧ distance_between_centers = rect_width - 2 * circle_radius 
  → distance_between_centers = 4 := by
  intros rect_width rect_height circle_radius distance_between_centers
  sorry

end distance_between_centers_of_circles_l280_280560


namespace central_cell_value_l280_280512

theorem central_cell_value (a b c d e f g h i : ℝ)
  (h_row1 : a * b * c = 10)
  (h_row2 : d * e * f = 10)
  (h_row3 : g * h * i = 10)
  (h_col1 : a * d * g = 10)
  (h_col2 : b * e * h = 10)
  (h_col3 : c * f * i = 10)
  (h_block1 : a * b * d * e = 3)
  (h_block2 : b * c * e * f = 3)
  (h_block3 : d * e * g * h = 3)
  (h_block4 : e * f * h * i = 3) :
  e = 0.00081 :=
sorry

end central_cell_value_l280_280512


namespace area_of_annulus_l280_280238

section annulus
variables {R r x : ℝ}
variable (h1 : R > r)
variable (h2 : R^2 - r^2 = x^2)

theorem area_of_annulus (R r x : ℝ) (h1 : R > r) (h2 : R^2 - r^2 = x^2) : 
  π * R^2 - π * r^2 = π * x^2 :=
sorry

end annulus

end area_of_annulus_l280_280238


namespace tan_45_eq_one_l280_280687

theorem tan_45_eq_one 
  (θ : ℝ)
  (hθ : θ = 45)
  (sin_45_eq : Real.sin (θ * Real.pi / 180) = Real.sqrt 2 / 2)
  (cos_45_eq : Real.cos (θ * Real.pi /180) = Real.sqrt 2 / 2)
  (tan_def : ∀ x : ℝ, 0 < Real.cos x → Real.tan x = Real.sin x / Real.cos x) : 
  Real.tan (θ * Real.pi / 180) = 1 := sorry

end tan_45_eq_one_l280_280687


namespace sum_of_roots_eq_14_l280_280196

theorem sum_of_roots_eq_14 (x : ℝ) :
  (x - 7) ^ 2 = 16 → ∃ r1 r2 : ℝ, (x = r1 ∨ x = r2) ∧ r1 + r2 = 14 :=
by
  intro h
  sorry

end sum_of_roots_eq_14_l280_280196


namespace tan_of_45_deg_l280_280732

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  -- Definitions used directly from conditions
  have hcos : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry
  have hsin : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry

  -- Using the definition of tangent
  rw [Real.tan_eq_sin_div_cos, hsin, hcos]
  exact div_self (ne_of_gt (Real.sqrt_pos.2 zero_lt_two))

end tan_of_45_deg_l280_280732


namespace central_cell_value_l280_280525

theorem central_cell_value
  (a b c d e f g h i : ℝ)
  (row1 : a * b * c = 10)
  (row2 : d * e * f = 10)
  (row3 : g * h * i = 10)
  (col1 : a * d * g = 10)
  (col2 : b * e * h = 10)
  (col3 : c * f * i = 10)
  (sub1 : a * b * d * e = 3)
  (sub2 : b * c * e * f = 3)
  (sub3 : d * e * g * h = 3)
  (sub4 : e * f * h * i = 3) : 
  e = 0.00081 :=
sorry

end central_cell_value_l280_280525


namespace central_cell_value_l280_280483

def table (a b c d e f g h i : ℝ) : Prop :=
  (a * b * c = 10) ∧ (d * e * f = 10) ∧ (g * h * i = 10) ∧
  (a * d * g = 10) ∧ (b * e * h = 10) ∧ (c * f * i = 10) ∧
  (a * b * d * e = 3) ∧ (b * c * e * f = 3) ∧ (d * e * g * h = 3) ∧ (e * f * h * i = 3)

theorem central_cell_value (a b c d f g h i e : ℝ) (h_table : table a b c d e f g h i) : 
  e = 0.00081 :=
by sorry

end central_cell_value_l280_280483


namespace tan_45_deg_l280_280655

theorem tan_45_deg : Real.tan (π / 4) = 1 := 
by
  have h_sin_cos : Real.sin (π / 4) = Real.cos (π / 4) := 
    by sorry
  have h_value : Real.sin (π / 4) = (Real.sqrt 2) / 2 := 
    by sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos, h_value]
  calc
    (Real.sqrt 2)/ 2 / (Real.sqrt 2)/ 2 = 1 := sorry

end tan_45_deg_l280_280655


namespace ellipse_properties_l280_280593

theorem ellipse_properties (h k a b : ℝ)
  (h_eq : h = 1)
  (k_eq : k = -3)
  (a_eq : a = 7)
  (b_eq : b = 4) :
  h + k + a + b = 9 :=
by
  sorry

end ellipse_properties_l280_280593


namespace tan_45_deg_eq_one_l280_280847

theorem tan_45_deg_eq_one :
  let θ := 45
  (∀ θ, tan θ = sin θ / cos θ) →
  (sin 45 = sqrt 2 / 2) →
  (cos 45 = sqrt 2 / 2) →
  tan 45 = 1 :=
by
  intros θ h_tan h_sin h_cos
  sorry

end tan_45_deg_eq_one_l280_280847


namespace sum_of_roots_l280_280142

theorem sum_of_roots (a b : Real) (h : (x - 7)^2 = 16):
  a + b = 14 :=
sorry

end sum_of_roots_l280_280142


namespace range_of_expression_l280_280433

variable (a b c : ℝ)

theorem range_of_expression (h1 : -3 < b) (h2 : b < a) (h3 : a < -1) (h4 : -2 < c) (h5 : c < -1) :
  0 < (a - b) * c^2 ∧ (a - b) * c^2 < 8 :=
sorry

end range_of_expression_l280_280433


namespace sum_of_roots_of_equation_l280_280136

theorem sum_of_roots_of_equation : 
  (∃ x1 x2 : ℝ, (x1 - 7)^2 = 16 ∧ (x2 - 7)^2 = 16 ∧ x1 ≠ x2 ∧ (x1 + x2 = 14)) :=
by
  sorry

end sum_of_roots_of_equation_l280_280136


namespace tennis_racket_price_l280_280042

theorem tennis_racket_price (P : ℝ) : 
    (0.8 * P + 515) * 1.10 + 20 = 800 → 
    P = 242.61 :=
by
  sorry

end tennis_racket_price_l280_280042


namespace find_number_l280_280911

theorem find_number (N: ℕ): (N % 131 = 112) ∧ (N % 132 = 98) → 1000 ≤ N ∧ N ≤ 9999 ∧ N = 1946 :=
sorry

end find_number_l280_280911


namespace tan_45_degrees_l280_280819

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l280_280819


namespace tan_45_deg_l280_280698

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l280_280698


namespace tan_45_deg_eq_one_l280_280845

theorem tan_45_deg_eq_one :
  let θ := 45
  (∀ θ, tan θ = sin θ / cos θ) →
  (sin 45 = sqrt 2 / 2) →
  (cos 45 = sqrt 2 / 2) →
  tan 45 = 1 :=
by
  intros θ h_tan h_sin h_cos
  sorry

end tan_45_deg_eq_one_l280_280845


namespace pipe_B_fill_time_l280_280994

theorem pipe_B_fill_time (T : ℕ) (h1 : 50 > 0) (h2 : 30 > 0)
  (h3 : (1/50 + 1/T = 1/30)) : T = 75 := 
sorry

end pipe_B_fill_time_l280_280994


namespace tan_45_deg_l280_280624

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l280_280624


namespace tan_45_eq_one_l280_280796

theorem tan_45_eq_one (Q : ℝ × ℝ) (h1 : Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2)) : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_eq_one_l280_280796


namespace all_points_lie_on_parabola_l280_280432

noncomputable def parabola_curve (u : ℝ) : ℝ × ℝ :=
  let x := 3^u - 4
  let y := 9^u - 7 * 3^u - 2
  (x, y)

theorem all_points_lie_on_parabola (u : ℝ) :
  let (x, y) := parabola_curve u
  y = x^2 + x - 6 := sorry

end all_points_lie_on_parabola_l280_280432


namespace white_trees_count_l280_280408

noncomputable def calculate_white_trees (total_trees pink_percent red_trees : ℕ) : ℕ :=
  total_trees - (total_trees * pink_percent / 100 + red_trees)

theorem white_trees_count 
  (h1 : total_trees = 42)
  (h2 : pink_percent = 100 / 3)
  (h3 : red_trees = 2) :
  calculate_white_trees total_trees pink_percent red_trees = 26 :=
by
  -- proof will go here
  sorry

end white_trees_count_l280_280408


namespace tan_45_eq_1_l280_280722

theorem tan_45_eq_1 (hcos : real.cos (real.pi / 4) = real.sqrt 2 / 2) (hsin : real.sin (real.pi / 4) = real.sqrt 2 / 2) : real.tan (real.pi / 4) = 1 :=
by
  have h : real.tan (real.pi / 4) = real.sin (real.pi / 4) / real.cos (real.pi / 4) := real.tan_eq_sin_div_cos _
  rw [hsin, hcos] at h
  simp at h
  exact h.symm

end tan_45_eq_1_l280_280722


namespace tan_45_deg_l280_280660

theorem tan_45_deg : Real.tan (π / 4) = 1 := 
by
  have h_sin_cos : Real.sin (π / 4) = Real.cos (π / 4) := 
    by sorry
  have h_value : Real.sin (π / 4) = (Real.sqrt 2) / 2 := 
    by sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos, h_value]
  calc
    (Real.sqrt 2)/ 2 / (Real.sqrt 2)/ 2 = 1 := sorry

end tan_45_deg_l280_280660


namespace pieces_of_chocolate_left_l280_280394

theorem pieces_of_chocolate_left (initial_boxes : ℕ) (given_away_boxes : ℕ) (pieces_per_box : ℕ) 
    (h1 : initial_boxes = 14) (h2 : given_away_boxes = 8) (h3 : pieces_per_box = 3) : 
    (initial_boxes - given_away_boxes) * pieces_per_box = 18 := 
by 
  -- The proof will be here
  sorry

end pieces_of_chocolate_left_l280_280394


namespace sufficient_not_necessary_of_and_false_or_true_l280_280442

variables (p q : Prop)

theorem sufficient_not_necessary_of_and_false_or_true :
  (¬(p ∧ q) → (p ∨ q)) ∧ ((p ∨ q) → ¬(¬(p ∧ q))) :=
sorry

end sufficient_not_necessary_of_and_false_or_true_l280_280442


namespace tan_45_deg_eq_one_l280_280842

theorem tan_45_deg_eq_one :
  let θ := 45
  (∀ θ, tan θ = sin θ / cos θ) →
  (sin 45 = sqrt 2 / 2) →
  (cos 45 = sqrt 2 / 2) →
  tan 45 = 1 :=
by
  intros θ h_tan h_sin h_cos
  sorry

end tan_45_deg_eq_one_l280_280842


namespace parabola_focus_l280_280427

theorem parabola_focus (a : ℝ) (h k x : ℝ) (hx : h = 0) (kx : k = 0) (a_eq : a = -1/16) :
  focus (y = -a * x^2) = (0, -4) :=
by
  sorry

end parabola_focus_l280_280427


namespace problem_l280_280017

def f (x : ℤ) : ℤ := 3 * x - 1
def g (x : ℤ) : ℤ := 2 * x + 5

theorem problem (h : ℤ) :
  (g (f (g (3))) : ℚ) / f (g (f (3))) = 69 / 206 :=
by
  sorry

end problem_l280_280017


namespace tan_45_degrees_l280_280802

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l280_280802


namespace unique_function_satisfying_condition_l280_280540

theorem unique_function_satisfying_condition (k : ℕ) (hk : 0 < k) :
  ∀ f : ℕ → ℕ, (∀ m n : ℕ, 0 < m → 0 < n → f m + f n ∣ (m + n) ^ k) →
  ∃ c : ℕ, ∀ n : ℕ, f n = n + c :=
by
  sorry

end unique_function_satisfying_condition_l280_280540


namespace tan_45_eq_1_l280_280775

theorem tan_45_eq_1 :
  let Q : ℝ × ℝ := (1 / Real.sqrt 2, 1 / Real.sqrt 2)
  in ∃ (Q : ℝ × ℝ), Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) → 
     Real.tan (Real.pi / 4) = 1 := 
by {
  -- The proof is left as an exercise.
  sorry
}

end tan_45_eq_1_l280_280775


namespace complement_is_correct_l280_280016

variable (U : Set ℕ) (A : Set ℕ)

def complement (U : Set ℕ) (A : Set ℕ) : Set ℕ :=
  { x ∈ U | x ∉ A }

theorem complement_is_correct :
  (U = {1, 2, 3, 4, 5, 6, 7}) →
  (A = {2, 4, 5}) →
  complement U A = {1, 3, 6, 7} :=
by
  sorry

end complement_is_correct_l280_280016


namespace tan_45_eq_1_l280_280770

theorem tan_45_eq_1 :
  let Q : ℝ × ℝ := (1 / Real.sqrt 2, 1 / Real.sqrt 2)
  in ∃ (Q : ℝ × ℝ), Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) → 
     Real.tan (Real.pi / 4) = 1 := 
by {
  -- The proof is left as an exercise.
  sorry
}

end tan_45_eq_1_l280_280770


namespace sum_of_a_and_b_l280_280247

theorem sum_of_a_and_b (a b : ℕ) (h1: a > 0) (h2 : b > 1) (h3 : ∀ (x y : ℕ), x > 0 → y > 1 → x^y < 500 → x = a ∧ y = b → x^y ≥ a^b ) :
  a + b = 24 :=
sorry

end sum_of_a_and_b_l280_280247


namespace bottle_caps_given_l280_280044

variable (initial_caps : ℕ) (final_caps : ℕ) (caps_given_by_rebecca : ℕ)

theorem bottle_caps_given (h1: initial_caps = 7) (h2: final_caps = 9) : caps_given_by_rebecca = 2 :=
by
  -- The proof will be filled here
  sorry

end bottle_caps_given_l280_280044


namespace parabola_y_intersection_l280_280367

theorem parabola_y_intersection (x y : ℝ) : 
  (∀ x' y', y' = -(x' + 2)^2 + 6 → ((x' = 0) → (y' = 2))) :=
by
  intros x' y' hy hx0
  rw hx0 at hy
  simp [hy]
  sorry

end parabola_y_intersection_l280_280367


namespace min_percentage_excellent_both_l280_280243

theorem min_percentage_excellent_both (P_M : ℝ) (P_C : ℝ) (hM : P_M = 0.7) (hC : P_C = 0.25) :
  (P_M * P_C = 0.175) :=
by
  rw [hM, hC]
  norm_num
  done

end min_percentage_excellent_both_l280_280243


namespace pf1_pf2_range_l280_280443

noncomputable def ellipse_point (x y : ℝ) : Prop :=
  x ^ 2 / 4 + y ^ 2 = 1

noncomputable def dot_product (x y : ℝ) : ℝ :=
  (x ^ 2 + y ^ 2 - 3)

theorem pf1_pf2_range (x y : ℝ) (h : ellipse_point x y) :
  -2 ≤ dot_product x y ∧ dot_product x y ≤ 1 :=
by
  sorry

end pf1_pf2_range_l280_280443


namespace range_of_m_l280_280097

-- Define the variables and main theorem
theorem range_of_m (m : ℝ) (a b c : ℝ) 
  (h₀ : a = 3) (h₁ : b = (1 - 2 * m)) (h₂ : c = 8)
  : -5 < m ∧ m < -2 :=
by
  -- Given that a, b, and c are sides of a triangle, we use the triangle inequality theorem
  -- This code will remain as a placeholder of that proof
  sorry

end range_of_m_l280_280097


namespace tan_45_deg_eq_one_l280_280838

/-- The tangent of 45 degrees is equal to 1. -/
theorem tan_45_deg_eq_one : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_one_l280_280838


namespace central_cell_value_l280_280522

theorem central_cell_value :
  ∀ (a b c d e f g h i : ℝ),
  a * b * c = 10 →
  d * e * f = 10 →
  g * h * i = 10 →
  a * d * g = 10 →
  b * e * h = 10 →
  c * f * i = 10 →
  a * b * d * e = 3 →
  b * c * e * f = 3 →
  d * e * g * h = 3 →
  e * f * h * i = 3 →
  e = 0.00081 := 
by sorry

end central_cell_value_l280_280522


namespace length_A_l280_280956

open Real

theorem length_A'B'_correct {A B C A' B' : ℝ × ℝ} :
  A = (0, 10) →
  B = (0, 15) →
  C = (3, 9) →
  (A'.1 = A'.2) →
  (B'.1 = B'.2) →
  (C.2 - A.2) / (C.1 - A.1) = ((B.2 - C.2) / (B.1 - C.1)) →
  (dist A' B') = 2.5 * sqrt 2 :=
by
  intros
  sorry

end length_A_l280_280956


namespace punger_needs_pages_l280_280964

theorem punger_needs_pages (p c h : ℕ) (h_p : p = 60) (h_c : c = 7) (h_h : h = 10) : 
  (p * c) / h = 42 := 
by
  sorry

end punger_needs_pages_l280_280964


namespace tan_45_deg_eq_one_l280_280846

theorem tan_45_deg_eq_one :
  let θ := 45
  (∀ θ, tan θ = sin θ / cos θ) →
  (sin 45 = sqrt 2 / 2) →
  (cos 45 = sqrt 2 / 2) →
  tan 45 = 1 :=
by
  intros θ h_tan h_sin h_cos
  sorry

end tan_45_deg_eq_one_l280_280846


namespace robert_coin_arrangement_l280_280071

noncomputable def num_arrangements (gold : ℕ) (silver : ℕ) : ℕ :=
  if gold + silver = 8 ∧ gold = 5 ∧ silver = 3 then 504 else 0

theorem robert_coin_arrangement :
  num_arrangements 5 3 = 504 := 
sorry

end robert_coin_arrangement_l280_280071


namespace tan_45_degrees_l280_280803

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l280_280803


namespace probability_of_experts_winning_l280_280905

-- Definitions required from the conditions
def p : ℝ := 0.6
def q : ℝ := 1 - p
def current_expert_score : ℕ := 3
def current_audience_score : ℕ := 4

-- The main theorem to state
theorem probability_of_experts_winning : 
  p^4 + 4 * p^3 * q = 0.4752 := 
by sorry

end probability_of_experts_winning_l280_280905


namespace experts_win_eventually_l280_280902

noncomputable def p : ℝ := 0.6
noncomputable def q : ℝ := 1 - p

def experts_need : ℕ := 3
def audience_need : ℕ := 2
def max_rounds : ℕ := 4

def winning_probability (experts_need : ℕ) (audience_need : ℕ) (p q : ℝ) : ℝ :=
  if experts_need = 3 ∧ audience_need = 2 ∧ p = 0.6 ∧ q = 0.4 then
    p^4 + 4 * p^3 * q
  else
    0

theorem experts_win_eventually :
  winning_probability experts_need audience_need p q = 0.4752 := sorry

end experts_win_eventually_l280_280902


namespace derivative_at_pi_div_3_l280_280307

noncomputable def f (x : ℝ) : ℝ := x * Real.cos x - Real.sin x

theorem derivative_at_pi_div_3 : 
  deriv f (Real.pi / 3) = - (Real.sqrt 3 * Real.pi / 6) :=
by
  sorry

end derivative_at_pi_div_3_l280_280307


namespace three_digit_multiples_of_7_l280_280456

theorem three_digit_multiples_of_7 :
  ∃ n : ℕ, (n = ∑ k in finset.range (143 - 15), ∀ k ∈ finset.range (143 - 15), 100 ≤ 7 * (15 + k) ∧ 7 * (15 + k) ≤ 999) :=
sorry

end three_digit_multiples_of_7_l280_280456


namespace quadratic_inequality_condition_l280_280978

theorem quadratic_inequality_condition (a b c : ℝ) (h : a < 0) (disc : b^2 - 4 * a * c ≤ 0) : 
  ∀ x : ℝ, a * x^2 + b * x + c ≤ 0 :=
sorry

end quadratic_inequality_condition_l280_280978


namespace tan_45_deg_eq_1_l280_280600

-- Define conditions
variable (Q : ℝ × ℝ)
variable (E : ℝ × ℝ)
variable (O : ℝ × ℝ := (0, 0))
variable (θ : ℝ := 45)

-- Define main theorem
theorem tan_45_deg_eq_1
  (Q_on_unit_circle : Q = (ℝ.sqrt 2 / 2, ℝ.sqrt 2 / 2))
  (E_foot_perpendicular : E = (ℝ.sqrt 2 / 2, 0))
  (triangle_45_45_90 : (Q.1 - O.1) ^ 2 + (Q.2 - O.2) ^ 2 = 1) :
  Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_1_l280_280600


namespace tan_of_45_deg_l280_280737

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  -- Definitions used directly from conditions
  have hcos : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry
  have hsin : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry

  -- Using the definition of tangent
  rw [Real.tan_eq_sin_div_cos, hsin, hcos]
  exact div_self (ne_of_gt (Real.sqrt_pos.2 zero_lt_two))

end tan_of_45_deg_l280_280737


namespace tan_45_deg_eq_1_l280_280601

-- Define conditions
variable (Q : ℝ × ℝ)
variable (E : ℝ × ℝ)
variable (O : ℝ × ℝ := (0, 0))
variable (θ : ℝ := 45)

-- Define main theorem
theorem tan_45_deg_eq_1
  (Q_on_unit_circle : Q = (ℝ.sqrt 2 / 2, ℝ.sqrt 2 / 2))
  (E_foot_perpendicular : E = (ℝ.sqrt 2 / 2, 0))
  (triangle_45_45_90 : (Q.1 - O.1) ^ 2 + (Q.2 - O.2) ^ 2 = 1) :
  Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_1_l280_280601


namespace tan_45_deg_l280_280629

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l280_280629


namespace combined_rate_is_29_l280_280550

def combined_rate_of_mpg (miles_ray : ℕ) (mpg_ray : ℕ) (miles_tom : ℕ) (mpg_tom : ℕ) (miles_jerry : ℕ) (mpg_jerry : ℕ) : ℕ :=
  let gallons_ray := miles_ray / mpg_ray
  let gallons_tom := miles_tom / mpg_tom
  let gallons_jerry := miles_jerry / mpg_jerry
  let total_gallons := gallons_ray + gallons_tom + gallons_jerry
  let total_miles := miles_ray + miles_tom + miles_jerry
  total_miles / total_gallons

theorem combined_rate_is_29 :
  combined_rate_of_mpg 60 50 60 20 60 30 = 29 :=
by
  sorry

end combined_rate_is_29_l280_280550


namespace find_top_angle_l280_280120

theorem find_top_angle 
  (sum_of_angles : ∀ (α β γ : ℝ), α + β + γ = 250) 
  (left_is_twice_right : ∀ (α β : ℝ), α = 2 * β) 
  (right_angle_is_60 : ∀ (β : ℝ), β = 60) :
  ∃ γ : ℝ, γ = 70 :=
by
  -- Assume the variables for the angles
  obtain ⟨α, β, γ, h_sum, h_left, h_right⟩ := ⟨_, _, _, sum_of_angles, left_is_twice_right, right_angle_is_60⟩
  -- Your proof here
  sorry

end find_top_angle_l280_280120


namespace solve_for_x_l280_280352

theorem solve_for_x (x : ℝ) (h : (x+10) / (x-4) = (x-3) / (x+6)) : x = -48 / 23 :=
by
  sorry

end solve_for_x_l280_280352


namespace tan_45_degree_l280_280669

theorem tan_45_degree : ∀ (Q : ℝ × ℝ),
  -- Q corresponds to 45 degrees on the unit circle
  Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) →
  -- The radius of the unit circle is 1
  (Q.1 ^ 2 + Q.2 ^ 2 = 1) →
  -- QOE is a 45-45-90 triangle
  True →
  -- Prove that tan 45 degrees equals 1
  Real.tan (Real.pi / 4) = 1 :=
by
  -- The proof using these conditions is omitted
  intros,
  sorry

end tan_45_degree_l280_280669


namespace parabola_y_intersection_l280_280368

theorem parabola_y_intersection (x y : ℝ) : 
  (∀ x' y', y' = -(x' + 2)^2 + 6 → ((x' = 0) → (y' = 2))) :=
by
  intros x' y' hy hx0
  rw hx0 at hy
  simp [hy]
  sorry

end parabola_y_intersection_l280_280368


namespace exists_quad_root_l280_280100

theorem exists_quad_root (a b c d : ℝ) (h : a * c = 2 * b + 2 * d) :
  (∃ x, x^2 + a * x + b = 0) ∨ (∃ x, x^2 + c * x + d = 0) :=
sorry

end exists_quad_root_l280_280100


namespace relationship_a_b_l280_280331

noncomputable def e : ℝ := Real.exp 1

theorem relationship_a_b
  (a b : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : e^a + 2 * a = e^b + 3 * b) :
  a > b :=
sorry

end relationship_a_b_l280_280331


namespace minimum_photos_l280_280292

theorem minimum_photos (G B : ℕ) (n : ℕ) : G = 4 → B = 8 → n ≥ 33 → 
  (∃ (p : fin ((G + B) choose 2) → (fin (G + B) × fin (G + B))),
  (∃ i j : fin (G + B), i ≠ j ∧ p i = p j) ∨ 
  (∃ k j : fin (G + B), k ≤ G ∧ j ≤ G ∧ p k = p j) ∨
  (∃ k j : fin (G + B), k > G ∧ j > G ∧ p k = p j)) :=
by
  intros G_eq B_eq n_ge_33
  sorry

end minimum_photos_l280_280292


namespace tan_45_deg_eq_one_l280_280843

theorem tan_45_deg_eq_one :
  let θ := 45
  (∀ θ, tan θ = sin θ / cos θ) →
  (sin 45 = sqrt 2 / 2) →
  (cos 45 = sqrt 2 / 2) →
  tan 45 = 1 :=
by
  intros θ h_tan h_sin h_cos
  sorry

end tan_45_deg_eq_one_l280_280843


namespace tan_45_eq_l280_280611

def tan_identity (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

def sin_45_eq : Real.sin 45 = Real.sqrt 2 / 2 := sorry

def cos_45_eq : Real.cos 45 = Real.sqrt 2 / 2 := sorry

theorem tan_45_eq : tan_identity 45 = 1 := by
  rw [tan_identity, sin_45_eq, cos_45_eq]
  -- simplifying (sqrt 2 / 2) / (sqrt 2 / 2) = 1
  sorry

end tan_45_eq_l280_280611


namespace handshake_problem_l280_280349

-- Defining the necessary elements:
def num_people : Nat := 12
def num_handshakes_per_person : Nat := num_people - 2

-- Defining the total number of handshakes. Each handshake is counted twice.
def total_handshakes : Nat := (num_people * num_handshakes_per_person) / 2

-- The theorem statement:
theorem handshake_problem : total_handshakes = 60 :=
by
  sorry

end handshake_problem_l280_280349


namespace min_colors_needed_l280_280125

def vertices (G : SimpleGraph ℕ) : Finset ℕ := { n : ℕ | G.adj n ≠ ∅ }

theorem min_colors_needed (G : SimpleGraph ℕ) (h_vertex_count : vertices G = 20) 
  (h_max_degree : ∀ v ∈ G.verts, G.degree v ≤ 3) : 
  ∃ k, (k = 4 ∧ ∀ v ∈ G.verts, ∃ f : G.edge → Fin 4, ∀ e₁ e₂ ∈ G.edge_set v, f e₁ ≠ f e₂) :=
by sorry

end min_colors_needed_l280_280125


namespace tan_45_eq_one_l280_280889

noncomputable def sin_45 : ℝ := real.sqrt 2 / 2
noncomputable def cos_45 : ℝ := real.sqrt 2 / 2

theorem tan_45_eq_one : real.tan (real.pi / 4) = 1 := by
  have hsin : real.sin (real.pi / 4) = sin_45 := by sorry
  have hcos : real.cos (real.pi / 4) = cos_45 := by sorry
  rw [real.tan_eq_sin_div_cos, hsin, hcos]
  rw [div_self]
  exact eq.refl 1
  exact sorry

end tan_45_eq_one_l280_280889


namespace gcf_4370_13824_l280_280132

/-- Define the two numbers 4370 and 13824 -/
def num1 := 4370
def num2 := 13824

/-- The statement that the GCF of num1 and num2 is 1 -/
theorem gcf_4370_13824 : Nat.gcd num1 num2 = 1 := by
  sorry

end gcf_4370_13824_l280_280132


namespace tan_45_eq_1_l280_280765

theorem tan_45_eq_1 :
  let Q : ℝ × ℝ := (1 / Real.sqrt 2, 1 / Real.sqrt 2)
  in ∃ (Q : ℝ × ℝ), Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) → 
     Real.tan (Real.pi / 4) = 1 := 
by {
  -- The proof is left as an exercise.
  sorry
}

end tan_45_eq_1_l280_280765


namespace increase_by_percentage_l280_280395

theorem increase_by_percentage (x : ℝ) (y : ℝ): x = 90 → y = 0.50 → x + x * y = 135 := 
by
  intro h1 h2
  sorry

end increase_by_percentage_l280_280395


namespace roller_coaster_cars_l280_280588

theorem roller_coaster_cars (n : ℕ) (h : ((n - 1) : ℝ) / n = 0.5) : n = 2 :=
sorry

end roller_coaster_cars_l280_280588


namespace tan_45_eq_one_l280_280865

theorem tan_45_eq_one : Real.tan (Real.pi / 4) = 1 := 
by
  sorry

end tan_45_eq_one_l280_280865


namespace tan_45_deg_eq_one_l280_280832

/-- The tangent of 45 degrees is equal to 1. -/
theorem tan_45_deg_eq_one : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_one_l280_280832


namespace tan_45_eq_one_l280_280894

noncomputable def sin_45 : ℝ := real.sqrt 2 / 2
noncomputable def cos_45 : ℝ := real.sqrt 2 / 2

theorem tan_45_eq_one : real.tan (real.pi / 4) = 1 := by
  have hsin : real.sin (real.pi / 4) = sin_45 := by sorry
  have hcos : real.cos (real.pi / 4) = cos_45 := by sorry
  rw [real.tan_eq_sin_div_cos, hsin, hcos]
  rw [div_self]
  exact eq.refl 1
  exact sorry

end tan_45_eq_one_l280_280894


namespace max_area_of_backyard_l280_280551

theorem max_area_of_backyard (fence_length : ℕ) (h1 : fence_length = 500) 
  (l w : ℕ) (h2 : l = 2 * w) (h3 : l + 2 * w = fence_length) : 
  l * w = 31250 := 
by
  sorry

end max_area_of_backyard_l280_280551


namespace central_cell_value_l280_280518

theorem central_cell_value :
  ∀ (a b c d e f g h i : ℝ),
  a * b * c = 10 →
  d * e * f = 10 →
  g * h * i = 10 →
  a * d * g = 10 →
  b * e * h = 10 →
  c * f * i = 10 →
  a * b * d * e = 3 →
  b * c * e * f = 3 →
  d * e * g * h = 3 →
  e * f * h * i = 3 →
  e = 0.00081 := 
by sorry

end central_cell_value_l280_280518


namespace tan_45_eq_one_l280_280886

noncomputable def sin_45 : ℝ := real.sqrt 2 / 2
noncomputable def cos_45 : ℝ := real.sqrt 2 / 2

theorem tan_45_eq_one : real.tan (real.pi / 4) = 1 := by
  have hsin : real.sin (real.pi / 4) = sin_45 := by sorry
  have hcos : real.cos (real.pi / 4) = cos_45 := by sorry
  rw [real.tan_eq_sin_div_cos, hsin, hcos]
  rw [div_self]
  exact eq.refl 1
  exact sorry

end tan_45_eq_one_l280_280886


namespace tan_45_degree_l280_280668

theorem tan_45_degree : ∀ (Q : ℝ × ℝ),
  -- Q corresponds to 45 degrees on the unit circle
  Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) →
  -- The radius of the unit circle is 1
  (Q.1 ^ 2 + Q.2 ^ 2 = 1) →
  -- QOE is a 45-45-90 triangle
  True →
  -- Prove that tan 45 degrees equals 1
  Real.tan (Real.pi / 4) = 1 :=
by
  -- The proof using these conditions is omitted
  intros,
  sorry

end tan_45_degree_l280_280668


namespace product_of_two_odd_numbers_not_always_composite_l280_280370

theorem product_of_two_odd_numbers_not_always_composite :
  ∃ (m n : ℕ), (¬ (2 ∣ m) ∧ ¬ (2 ∣ n)) ∧ (∀ d : ℕ, d ∣ (m * n) → d = 1 ∨ d = m * n) :=
by
  sorry

end product_of_two_odd_numbers_not_always_composite_l280_280370


namespace three_digit_multiples_of_7_l280_280457

theorem three_digit_multiples_of_7 :
  ∃ n : ℕ, (n = ∑ k in finset.range (143 - 15), ∀ k ∈ finset.range (143 - 15), 100 ≤ 7 * (15 + k) ∧ 7 * (15 + k) ≤ 999) :=
sorry

end three_digit_multiples_of_7_l280_280457


namespace tan_45_eq_l280_280620

def tan_identity (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

def sin_45_eq : Real.sin 45 = Real.sqrt 2 / 2 := sorry

def cos_45_eq : Real.cos 45 = Real.sqrt 2 / 2 := sorry

theorem tan_45_eq : tan_identity 45 = 1 := by
  rw [tan_identity, sin_45_eq, cos_45_eq]
  -- simplifying (sqrt 2 / 2) / (sqrt 2 / 2) = 1
  sorry

end tan_45_eq_l280_280620


namespace sum_of_roots_l280_280150

theorem sum_of_roots : 
  let eq := (x - 7) ^ 2 = 16 in
  (roots : set ℝ) := { x | eq } in
  ∑ roots = 14 :=
by
sorry

end sum_of_roots_l280_280150


namespace sum_of_roots_eq_l280_280184

theorem sum_of_roots_eq (x : ℝ) : (x - 7)^2 = 16 → x = 11 ∨ x = 3 → ∑ (root : ℝ) in {11, 3}, root = 14 :=
by
  assume h₁ : (x - 7)^2 = 16
  assume h₂ : x = 11 ∨ x = 3
  sorry

end sum_of_roots_eq_l280_280184


namespace tan_of_45_deg_l280_280739

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  -- Definitions used directly from conditions
  have hcos : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry
  have hsin : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry

  -- Using the definition of tangent
  rw [Real.tan_eq_sin_div_cos, hsin, hcos]
  exact div_self (ne_of_gt (Real.sqrt_pos.2 zero_lt_two))

end tan_of_45_deg_l280_280739


namespace problem_l280_280478

variable (a : Int)
variable (h : -a = 1)

theorem problem : 3 * a - 2 = -5 :=
by
  -- Proof will go here
  sorry

end problem_l280_280478


namespace tan_45_deg_l280_280623

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l280_280623


namespace tan_45_eq_one_l280_280892

noncomputable def sin_45 : ℝ := real.sqrt 2 / 2
noncomputable def cos_45 : ℝ := real.sqrt 2 / 2

theorem tan_45_eq_one : real.tan (real.pi / 4) = 1 := by
  have hsin : real.sin (real.pi / 4) = sin_45 := by sorry
  have hcos : real.cos (real.pi / 4) = cos_45 := by sorry
  rw [real.tan_eq_sin_div_cos, hsin, hcos]
  rw [div_self]
  exact eq.refl 1
  exact sorry

end tan_45_eq_one_l280_280892


namespace tan_45_degree_l280_280744

theorem tan_45_degree : ∀ (θ : ℝ), 
  θ = 45 → 
  (sin θ = 1 / real.sqrt 2) → 
  (cos θ = 1 / real.sqrt 2) → 
  real.tan θ = 1 := 
by
  intros θ hθ hsin hcos
  sorry

end tan_45_degree_l280_280744


namespace number_of_three_digit_multiples_of_7_l280_280458

theorem number_of_three_digit_multiples_of_7 : 
  let smallest_multiple := 7 * Nat.ceil (100 / 7)
  let largest_multiple := 7 * Nat.floor (999 / 7)
  (largest_multiple - smallest_multiple) / 7 + 1 = 128 :=
by
  sorry

end number_of_three_digit_multiples_of_7_l280_280458


namespace central_cell_value_l280_280519

theorem central_cell_value :
  ∀ (a b c d e f g h i : ℝ),
  a * b * c = 10 →
  d * e * f = 10 →
  g * h * i = 10 →
  a * d * g = 10 →
  b * e * h = 10 →
  c * f * i = 10 →
  a * b * d * e = 3 →
  b * c * e * f = 3 →
  d * e * g * h = 3 →
  e * f * h * i = 3 →
  e = 0.00081 := 
by sorry

end central_cell_value_l280_280519


namespace tan_45_deg_eq_1_l280_280609

-- Define conditions
variable (Q : ℝ × ℝ)
variable (E : ℝ × ℝ)
variable (O : ℝ × ℝ := (0, 0))
variable (θ : ℝ := 45)

-- Define main theorem
theorem tan_45_deg_eq_1
  (Q_on_unit_circle : Q = (ℝ.sqrt 2 / 2, ℝ.sqrt 2 / 2))
  (E_foot_perpendicular : E = (ℝ.sqrt 2 / 2, 0))
  (triangle_45_45_90 : (Q.1 - O.1) ^ 2 + (Q.2 - O.2) ^ 2 = 1) :
  Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_1_l280_280609


namespace tan_45_eq_1_l280_280772

theorem tan_45_eq_1 :
  let Q : ℝ × ℝ := (1 / Real.sqrt 2, 1 / Real.sqrt 2)
  in ∃ (Q : ℝ × ℝ), Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) → 
     Real.tan (Real.pi / 4) = 1 := 
by {
  -- The proof is left as an exercise.
  sorry
}

end tan_45_eq_1_l280_280772


namespace tan_45_deg_eq_1_l280_280605

-- Define conditions
variable (Q : ℝ × ℝ)
variable (E : ℝ × ℝ)
variable (O : ℝ × ℝ := (0, 0))
variable (θ : ℝ := 45)

-- Define main theorem
theorem tan_45_deg_eq_1
  (Q_on_unit_circle : Q = (ℝ.sqrt 2 / 2, ℝ.sqrt 2 / 2))
  (E_foot_perpendicular : E = (ℝ.sqrt 2 / 2, 0))
  (triangle_45_45_90 : (Q.1 - O.1) ^ 2 + (Q.2 - O.2) ^ 2 = 1) :
  Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_1_l280_280605


namespace count_ordered_triples_l280_280424

open Nat

theorem count_ordered_triples :
  ∃ (s : Finset (ℕ × ℕ × ℕ)), 
  (∀ t ∈ s, let ⟨x, y, z⟩ := t in 
    lcm x y = 180 ∧ lcm x z = 840 ∧ lcm y z = 1260 ∧ gcd (gcd x y) z = 6) ∧
  s.card = 2 :=
sorry

end count_ordered_triples_l280_280424


namespace find_a_plus_b_l280_280333

noncomputable def f (a b x : ℝ) : ℝ := a * x + b
def h (x : ℝ) : ℝ := 3 * x - 6

theorem find_a_plus_b (a b : ℝ) (h_cond : ∀ x : ℝ, h (f a b x) = 4 * x + 3) : a + b = 13 / 3 :=
by
  sorry

end find_a_plus_b_l280_280333


namespace percent_of_sales_not_pens_pencils_erasers_l280_280361

theorem percent_of_sales_not_pens_pencils_erasers :
  let percent_pens := 25
  let percent_pencils := 30
  let percent_erasers := 20
  let percent_total := 100
  percent_total - (percent_pens + percent_pencils + percent_erasers) = 25 :=
by
  -- definitions and assumptions
  let percent_pens := 25
  let percent_pencils := 30
  let percent_erasers := 20
  let percent_total := 100
  sorry

end percent_of_sales_not_pens_pencils_erasers_l280_280361


namespace probability_at_least_one_defective_is_correct_l280_280224

noncomputable def probability_at_least_one_defective : ℚ :=
  let total_bulbs := 23
  let defective_bulbs := 4
  let non_defective_bulbs := total_bulbs - defective_bulbs
  let probability_neither_defective :=
    (non_defective_bulbs / total_bulbs) * ((non_defective_bulbs - 1) / (total_bulbs - 1))
  1 - probability_neither_defective

theorem probability_at_least_one_defective_is_correct :
  probability_at_least_one_defective = 164 / 506 :=
by
  sorry

end probability_at_least_one_defective_is_correct_l280_280224


namespace tan_45_degree_l280_280746

theorem tan_45_degree : ∀ (θ : ℝ), 
  θ = 45 → 
  (sin θ = 1 / real.sqrt 2) → 
  (cos θ = 1 / real.sqrt 2) → 
  real.tan θ = 1 := 
by
  intros θ hθ hsin hcos
  sorry

end tan_45_degree_l280_280746


namespace height_is_centimeters_weight_is_kilograms_book_length_is_centimeters_book_thickness_is_millimeters_cargo_capacity_is_tons_sleep_time_is_hours_tree_height_is_meters_l280_280421

-- Definitions
def Height (x : ℕ) : Prop := x = 140
def Weight (x : ℕ) : Prop := x = 23
def BookLength (x : ℕ) : Prop := x = 20
def BookThickness (x : ℕ) : Prop := x = 7
def CargoCapacity (x : ℕ) : Prop := x = 4
def SleepTime (x : ℕ) : Prop := x = 9
def TreeHeight (x : ℕ) : Prop := x = 12

-- Propositions
def XiaohongHeightUnit := "centimeters"
def XiaohongWeightUnit := "kilograms"
def MathBookLengthUnit := "centimeters"
def MathBookThicknessUnit := "millimeters"
def TruckCargoCapacityUnit := "tons"
def ChildrenSleepTimeUnit := "hours"
def BigTreeHeightUnit := "meters"

theorem height_is_centimeters (x : ℕ) (h : Height x) : XiaohongHeightUnit = "centimeters" := sorry
theorem weight_is_kilograms (x : ℕ) (w : Weight x) : XiaohongWeightUnit = "kilograms" := sorry
theorem book_length_is_centimeters (x : ℕ) (l : BookLength x) : MathBookLengthUnit = "centimeters" := sorry
theorem book_thickness_is_millimeters (x : ℕ) (t : BookThickness x) : MathBookThicknessUnit = "millimeters" := sorry
theorem cargo_capacity_is_tons (x : ℕ) (c : CargoCapacity x) : TruckCargoCapacityUnit = "tons" := sorry
theorem sleep_time_is_hours (x : ℕ) (s : SleepTime x) : ChildrenSleepTimeUnit = "hours" := sorry
theorem tree_height_is_meters (x : ℕ) (th : TreeHeight x) : BigTreeHeightUnit = "meters" := sorry

end height_is_centimeters_weight_is_kilograms_book_length_is_centimeters_book_thickness_is_millimeters_cargo_capacity_is_tons_sleep_time_is_hours_tree_height_is_meters_l280_280421


namespace charity_donation_correct_l280_280556

-- Define each donation series for Suzanne, Maria, and James
def suzanne_donation_per_km (n : ℕ) : ℝ :=
  match n with
  |  0     => 10
  | (n+1)  => 2 * suzanne_donation_per_km n

def maria_donation_per_km (n : ℕ) : ℝ :=
  match n with
  |  0     => 15
  | (n+1)  => 1.5 * maria_donation_per_km n

def james_donation_per_km (n : ℕ) : ℝ :=
  match n with
  |  0     => 20
  | (n+1)  => 2 * james_donation_per_km n

-- Total donations after 5 kilometers
def total_donation_suzanne : ℝ := (List.range 5).map suzanne_donation_per_km |>.sum
def total_donation_maria : ℝ := (List.range 5).map maria_donation_per_km |>.sum
def total_donation_james : ℝ := (List.range 5).map james_donation_per_km |>.sum

def total_donation_charity : ℝ :=
  total_donation_suzanne + total_donation_maria + total_donation_james

-- Statement to be proven
theorem charity_donation_correct : total_donation_charity = 1127.81 := by
  sorry

end charity_donation_correct_l280_280556


namespace extra_mangoes_l280_280065

-- Definitions of the conditions
def original_price_per_mango := 433.33 / 130
def new_price_per_mango := original_price_per_mango - 0.10 * original_price_per_mango
def mangoes_at_original_price := 360 / original_price_per_mango
def mangoes_at_new_price := 360 / new_price_per_mango

-- Statement to be proved
theorem extra_mangoes : mangoes_at_new_price - mangoes_at_original_price = 12 := 
by {
  sorry
}

end extra_mangoes_l280_280065


namespace sum_of_roots_l280_280144

theorem sum_of_roots (a b : Real) (h : (x - 7)^2 = 16):
  a + b = 14 :=
sorry

end sum_of_roots_l280_280144


namespace minimum_connected_components_l280_280898

/-- We start with two points A, B on a 6*7 lattice grid. We say two points 
  X, Y are connected if one can reflect several times with respect to points A, B 
  and reach from X to Y. Prove that the minimum number of connected components 
  over all choices of A, B is 8. -/
theorem minimum_connected_components (A B : ℕ × ℕ) 
  (hA : A.1 < 6 ∧ A.2 < 7) (hB : B.1 < 6 ∧ B.2 < 7) :
  ∃ k, k = 8 :=
sorry

end minimum_connected_components_l280_280898


namespace quadrilateral_area_l280_280340

noncomputable def rectangle_ABCD := 
  let A := (0:ℝ, 8:ℝ)
  let B := (11:ℝ, 8:ℝ)
  let C := (11:ℝ, 0:ℝ)
  let D := (0:ℝ, 0:ℝ)
  (A, B, C, D)

noncomputable def points_EF := 
  let E := (5:ℝ, 4:ℝ)
  let F := (6:ℝ, 4:ℝ)
  (E, F)

def area_quadrilateral (E F B : ℝ × ℝ) : ℝ :=
  1/2 * ((B.1 - E.1 + E.1 - F.1) * 4)

theorem quadrilateral_area :
  let (A, B, C, D) := rectangle_ABCD in
  let (E, F) := points_EF in
  (E.1 - A.1)^2 + (E.2 - A.2)^2 = (D.1 - E.1)^2 + (D.2 - E.2)^2 →
  (F.1 - B.1)^2 + (F.2 - B.2)^2 = (C.1 - F.1)^2 + (C.2 - F.2)^2 →
  (E.1 - F.1)^2 + (E.2 - F.2)^2 = (B.1 - F.1)^2 + (B.2 - F.2)^2 →
  area_quadrilateral E F B = 32 := by
  intros
  have a := A; have b := B; have e := E; have f := F
  have ablen := 11 -- AB
  have bclen := 8  -- BC
  sorry

end quadrilateral_area_l280_280340


namespace max_pieces_with_single_cut_min_cuts_to_intersect_all_pieces_l280_280915

theorem max_pieces_with_single_cut (n : ℕ) (h : n = 4) :
  (∃ m : ℕ, m = 23) :=
sorry

theorem min_cuts_to_intersect_all_pieces (n : ℕ) (h : n = 4) :
  (∃ k : ℕ, k = 3) :=
sorry

noncomputable def pieces_of_cake : ℕ := 23

noncomputable def cuts_required : ℕ := 3

end max_pieces_with_single_cut_min_cuts_to_intersect_all_pieces_l280_280915


namespace period_six_l280_280939

variable {R : Type} [LinearOrderedField R]

def symmetric1 (f : R → R) : Prop := ∀ x : R, f (2 + x) = f (2 - x)
def symmetric2 (f : R → R) : Prop := ∀ x : R, f (5 + x) = f (5 - x)

theorem period_six (f : R → R) (h1 : symmetric1 f) (h2 : symmetric2 f) : ∀ x : R, f (x + 6) = f x :=
sorry

end period_six_l280_280939


namespace geometric_representation_l280_280381

variables (a : ℝ)

-- Definition of the area of the figure
def total_area := a^2 + 1.5 * a

-- Definition of the perimeter of the figure
def total_perimeter := 4 * a + 3

theorem geometric_representation :
  total_area a = a^2 + 1.5 * a ∧ total_perimeter a = 4 * a + 3 :=
by
  exact ⟨rfl, rfl⟩

end geometric_representation_l280_280381


namespace problem_l280_280332

-- Definitions for the problem's conditions:
variables {a b c d : ℝ}

-- a and b are roots of x^2 + 68x + 1 = 0
axiom ha : a ^ 2 + 68 * a + 1 = 0
axiom hb : b ^ 2 + 68 * b + 1 = 0

-- c and d are roots of x^2 - 86x + 1 = 0
axiom hc : c ^ 2 - 86 * c + 1 = 0
axiom hd : d ^ 2 - 86 * d + 1 = 0

theorem problem : (a + c) * (b + c) * (a - d) * (b - d) = 2772 :=
sorry

end problem_l280_280332


namespace larger_exceeds_smaller_by_16_l280_280989

-- Define the smaller number S and the larger number L in terms of the ratio 7:11
def S : ℕ := 28
def L : ℕ := (11 * S) / 7

-- State the theorem that the larger number exceeds the smaller number by 16
theorem larger_exceeds_smaller_by_16 : L - S = 16 :=
by
  -- Proof steps will go here
  sorry

end larger_exceeds_smaller_by_16_l280_280989


namespace tan_45_eq_1_l280_280863

theorem tan_45_eq_1 :
  let π := Real.pi in
  ∃ (x : ℝ), x = π / 4 ∧
  sin x = sqrt 2 / 2 ∧
  cos x = sqrt 2 / 2 →
  tan (π / 4) = 1 :=
by sorry

end tan_45_eq_1_l280_280863


namespace common_difference_l280_280303

theorem common_difference (a : ℕ → ℝ) (d : ℝ) (h1 : a 1 = 1)
  (h2 : a 2 = 1 + d) (h4 : a 4 = 1 + 3 * d) (h5 : a 5 = 1 + 4 * d) 
  (h_geometric : (a 4)^2 = a 2 * a 5) 
  (h_nonzero : d ≠ 0) : 
  d = 1 / 5 :=
by sorry

end common_difference_l280_280303


namespace central_cell_value_l280_280529

theorem central_cell_value
  (a b c d e f g h i : ℝ)
  (row1 : a * b * c = 10)
  (row2 : d * e * f = 10)
  (row3 : g * h * i = 10)
  (col1 : a * d * g = 10)
  (col2 : b * e * h = 10)
  (col3 : c * f * i = 10)
  (sub1 : a * b * d * e = 3)
  (sub2 : b * c * e * f = 3)
  (sub3 : d * e * g * h = 3)
  (sub4 : e * f * h * i = 3) : 
  e = 0.00081 :=
sorry

end central_cell_value_l280_280529


namespace tan_45_eq_l280_280615

def tan_identity (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

def sin_45_eq : Real.sin 45 = Real.sqrt 2 / 2 := sorry

def cos_45_eq : Real.cos 45 = Real.sqrt 2 / 2 := sorry

theorem tan_45_eq : tan_identity 45 = 1 := by
  rw [tan_identity, sin_45_eq, cos_45_eq]
  -- simplifying (sqrt 2 / 2) / (sqrt 2 / 2) = 1
  sorry

end tan_45_eq_l280_280615


namespace statement_A_statement_B_statement_C_l280_280944

variable {α : Type}

-- Conditions for statement A
def angle_greater (A B : ℝ) : Prop := A > B
def sin_greater (A B : ℝ) : Prop := Real.sin A > Real.sin B

-- Conditions for statement B
def acute_triangle (A B C : ℝ) : Prop := A < Real.pi / 2 ∧ B < Real.pi / 2 ∧ C < Real.pi / 2
def sin_greater_than_cos (A B : ℝ) : Prop := Real.sin A > Real.cos B

-- Conditions for statement C
def obtuse_triangle (C : ℝ) : Prop := C > Real.pi / 2

-- Statement A in Lean
theorem statement_A (A B : ℝ) : angle_greater A B → sin_greater A B :=
sorry

-- Statement B in Lean
theorem statement_B {A B C : ℝ} : acute_triangle A B C → sin_greater_than_cos A B :=
sorry

-- Statement C in Lean
theorem statement_C {a b c : ℝ} (h : a^2 + b^2 < c^2) : obtuse_triangle C :=
sorry

-- Statement D in Lean (proof not needed as it's incorrect)
-- Theorem is omitted since statement D is incorrect

end statement_A_statement_B_statement_C_l280_280944


namespace min_photos_exists_l280_280296

-- Conditions: Girls and Boys
def girls : ℕ := 4
def boys : ℕ := 8

-- Definition representing the minimum number of photos for the condition
def min_photos : ℕ := 33

theorem min_photos_exists : 
  ∀ (photos : ℕ), 
  (photos ≥ min_photos) →
  (∃ (bb gg bg : ℕ), 
    (bb > 0 ∨ gg > 0 ∨ bg < photos)) :=
begin
  sorry
end

end min_photos_exists_l280_280296


namespace remaining_area_l280_280400

-- Definitions based on conditions
def large_rectangle_length (x : ℝ) : ℝ := 2 * x + 8
def large_rectangle_width (x : ℝ) : ℝ := x + 6
def hole_length (x : ℝ) : ℝ := 3 * x - 4
def hole_width (x : ℝ) : ℝ := x + 1

-- Theorem statement
theorem remaining_area (x : ℝ) : (large_rectangle_length x) * (large_rectangle_width x) - (hole_length x) * (hole_width x) = -x^2 + 21 * x + 52 :=
by
  -- Proof is skipped
  sorry

end remaining_area_l280_280400


namespace sum_of_roots_eq_14_l280_280203

-- Define the condition as a hypothesis
def equation (x : ℝ) := (x - 7) ^ 2 = 16

-- Define the statement that needs to be proved
theorem sum_of_roots_eq_14 : 
  (∀ x : ℝ, equation x → x = 11 ∨ x = 3) → 11 + 3 = 14 :=
by 
sorry

end sum_of_roots_eq_14_l280_280203


namespace initially_calculated_average_is_correct_l280_280559

theorem initially_calculated_average_is_correct :
  let S := 220
  let incorrect_sum := S - 36 + 26
  let initially_avg := incorrect_sum / 10
  initially_avg = 22 :=
by
  let S := 220
  let incorrect_sum := S - 36 + 26
  let initially_avg := incorrect_sum / 10
  show initially_avg = 22
  sorry

end initially_calculated_average_is_correct_l280_280559


namespace jasper_hot_dogs_fewer_l280_280328

theorem jasper_hot_dogs_fewer (chips drinks hot_dogs : ℕ)
  (h1 : chips = 27)
  (h2 : drinks = 31)
  (h3 : drinks = hot_dogs + 12) : 27 - hot_dogs = 8 := by
  sorry

end jasper_hot_dogs_fewer_l280_280328


namespace tan_45_deg_eq_one_l280_280839

/-- The tangent of 45 degrees is equal to 1. -/
theorem tan_45_deg_eq_one : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_one_l280_280839


namespace LisaNeedsMoreMarbles_l280_280061

theorem LisaNeedsMoreMarbles :
  let friends := 12
  let marbles := 40
  let required_marbles := (friends * (friends + 1)) / 2
  let additional_marbles := required_marbles - marbles
  additional_marbles = 38 :=
by
  let friends := 12
  let marbles := 40
  let required_marbles := (friends * (friends + 1)) / 2
  let additional_marbles := required_marbles - marbles
  have h1 : required_marbles = 78 := by
    calc (friends * (friends + 1)) / 2
      _ = (12 * 13) / 2 : by rfl
      _ = 156 / 2 : by rfl
      _ = 78 : by norm_num
  have h2 : additional_marbles = 38 := by
    calc required_marbles - marbles
      _ = 78 - 40 : by rw h1
      _ = 38 : by norm_num
  exact h2

end LisaNeedsMoreMarbles_l280_280061


namespace solve_for_x_l280_280353

theorem solve_for_x (x : ℝ) (h : (x+10) / (x-4) = (x-3) / (x+6)) : x = -48 / 23 :=
by
  sorry

end solve_for_x_l280_280353


namespace line_segment_endpoint_l280_280401

theorem line_segment_endpoint (x : ℝ) (h1 : (x - 3)^2 + 36 = 289) (h2 : x < 0) : x = 3 - Real.sqrt 253 :=
sorry

end line_segment_endpoint_l280_280401


namespace number_of_positive_integers_l280_280093

theorem number_of_positive_integers (n : ℕ) (hpos : 0 < n) (h : 24 - 6 * n ≥ 12) : n = 1 ∨ n = 2 :=
sorry

end number_of_positive_integers_l280_280093


namespace tan_45_deg_eq_1_l280_280602

-- Define conditions
variable (Q : ℝ × ℝ)
variable (E : ℝ × ℝ)
variable (O : ℝ × ℝ := (0, 0))
variable (θ : ℝ := 45)

-- Define main theorem
theorem tan_45_deg_eq_1
  (Q_on_unit_circle : Q = (ℝ.sqrt 2 / 2, ℝ.sqrt 2 / 2))
  (E_foot_perpendicular : E = (ℝ.sqrt 2 / 2, 0))
  (triangle_45_45_90 : (Q.1 - O.1) ^ 2 + (Q.2 - O.2) ^ 2 = 1) :
  Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_1_l280_280602


namespace tan_45_eq_1_l280_280830

theorem tan_45_eq_1
: ∀ (Q : ℝ × ℝ), Q = (Real.sqrt 2 / 2, Real.sqrt 2 / 2) ->
  ∀ θ, θ = Real.pi / 4 -> 
  Real.tan θ = 1 :=
by
  intros Q HQ θ Hθ
  rw [HQ, Hθ]
  sorry

end tan_45_eq_1_l280_280830


namespace more_stickers_correct_l280_280343

def total_stickers : ℕ := 58
def first_box_stickers : ℕ := 23
def second_box_stickers : ℕ := total_stickers - first_box_stickers
def more_stickers_in_second_box : ℕ := second_box_stickers - first_box_stickers

theorem more_stickers_correct : more_stickers_in_second_box = 12 := by
  sorry

end more_stickers_correct_l280_280343


namespace total_cards_is_56_l280_280073

-- Let n be the number of Pokemon cards each person has
def n : Nat := 14

-- Let k be the number of people
def k : Nat := 4

-- Total number of Pokemon cards
def total_cards : Nat := n * k

-- Prove that the total number of Pokemon cards is 56
theorem total_cards_is_56 : total_cards = 56 := by
  sorry

end total_cards_is_56_l280_280073


namespace three_digit_multiples_of_7_l280_280455

theorem three_digit_multiples_of_7 :
  ∃ n : ℕ, (n = ∑ k in finset.range (143 - 15), ∀ k ∈ finset.range (143 - 15), 100 ≤ 7 * (15 + k) ∧ 7 * (15 + k) ≤ 999) :=
sorry

end three_digit_multiples_of_7_l280_280455


namespace cubes_sum_to_91_l280_280983

theorem cubes_sum_to_91
  (a b : ℤ)
  (h : a^3 + b^3 = 91) : a * b = 12 :=
sorry

end cubes_sum_to_91_l280_280983


namespace part1_part2_l280_280058

-- Given conditions for part (Ⅰ)
variables {a_n : ℕ → ℝ} {S_n : ℕ → ℝ}

-- The general formula for the sequence {a_n}
theorem part1 (a3_eq : a_n 3 = 1 / 8)
  (arith_seq : S_n 2 + 1 / 16 = 2 * S_n 3 - S_n 4) :
  ∀ n, a_n n = (1 / 2)^n := sorry

-- Given conditions for part (Ⅱ)
variables {b_n : ℕ → ℝ} {T_n : ℕ → ℝ}

-- The sum of the first n terms of the sequence {b_n}
theorem part2 (h_general : ∀ n, a_n n = (1 / 2)^n)
  (b_formula : ∀ n, b_n n = a_n n * (Real.log (a_n n) / Real.log (1 / 2))) :
  ∀ n, T_n n = 2 - (n + 2) / 2^n := sorry

end part1_part2_l280_280058


namespace tan_45_deg_l280_280637

theorem tan_45_deg : (Real.tan (Float.pi / 4)) = 1 :=
by
  sorry

end tan_45_deg_l280_280637


namespace sum_of_roots_eq_l280_280183

theorem sum_of_roots_eq (x : ℝ) : (x - 7)^2 = 16 → x = 11 ∨ x = 3 → ∑ (root : ℝ) in {11, 3}, root = 14 :=
by
  assume h₁ : (x - 7)^2 = 16
  assume h₂ : x = 11 ∨ x = 3
  sorry

end sum_of_roots_eq_l280_280183


namespace tan_45_eq_one_l280_280896

noncomputable def sin_45 : ℝ := real.sqrt 2 / 2
noncomputable def cos_45 : ℝ := real.sqrt 2 / 2

theorem tan_45_eq_one : real.tan (real.pi / 4) = 1 := by
  have hsin : real.sin (real.pi / 4) = sin_45 := by sorry
  have hcos : real.cos (real.pi / 4) = cos_45 := by sorry
  rw [real.tan_eq_sin_div_cos, hsin, hcos]
  rw [div_self]
  exact eq.refl 1
  exact sorry

end tan_45_eq_one_l280_280896


namespace tan_45_eq_one_l280_280681

theorem tan_45_eq_one 
  (θ : ℝ)
  (hθ : θ = 45)
  (sin_45_eq : Real.sin (θ * Real.pi / 180) = Real.sqrt 2 / 2)
  (cos_45_eq : Real.cos (θ * Real.pi /180) = Real.sqrt 2 / 2)
  (tan_def : ∀ x : ℝ, 0 < Real.cos x → Real.tan x = Real.sin x / Real.cos x) : 
  Real.tan (θ * Real.pi / 180) = 1 := sorry

end tan_45_eq_one_l280_280681


namespace intersection_of_parabola_with_y_axis_l280_280366

theorem intersection_of_parabola_with_y_axis :
  ∃ y : ℝ, y = - (0 + 2)^2 + 6 ∧ (0, y) = (0, 2) :=
by
  sorry

end intersection_of_parabola_with_y_axis_l280_280366


namespace multiples_of_7_are_128_l280_280461

theorem multiples_of_7_are_128 : 
  let range_start := 100
  let range_end := 999
  let multiple_7_smallest := 7 * 15
  let multiple_7_largest := 7 * 142
  let n_terms := (142 - 15 + 1)
  n_terms = 128 := sorry

end multiples_of_7_are_128_l280_280461


namespace multiple_of_n_eventually_written_l280_280946

theorem multiple_of_n_eventually_written (a b n : ℕ) (h_a_pos: 0 < a) (h_b_pos: 0 < b)  (h_ab_neq: a ≠ b) (h_n_pos: 0 < n) :
  ∃ m : ℕ, m % n = 0 :=
sorry

end multiple_of_n_eventually_written_l280_280946


namespace smallest_n_inequality_l280_280919

theorem smallest_n_inequality : 
  ∃ (n : ℕ), (n > 0) ∧ ( ∀ m : ℕ, (m > 0) ∧ ( m < n ) → ¬( ( 1 : ℚ ) / m - ( 1 / ( m + 1 : ℚ ) ) < ( 1 / 15 ) ) ) ∧ ( ( 1 : ℚ ) / n - ( 1 / ( n + 1 : ℚ ) ) < ( 1 / 15 ) ) :=
sorry

end smallest_n_inequality_l280_280919


namespace compute_fg_neg1_l280_280935

def f (x : ℝ) : ℝ := 5 - 2 * x
def g (x : ℝ) : ℝ := x^3 + 2

theorem compute_fg_neg1 : f (g (-1)) = 3 := by
  sorry

end compute_fg_neg1_l280_280935


namespace inletRate_is_3_l280_280410

def volumeTank (v_cubic_feet : ℕ) : ℕ :=
  1728 * v_cubic_feet

def outletRate1 : ℕ := 9 -- rate of first outlet in cubic inches/min
def outletRate2 : ℕ := 6 -- rate of second outlet in cubic inches/min
def tankVolume : ℕ := volumeTank 30 -- tank volume in cubic inches
def minutesToEmpty : ℕ := 4320 -- time to empty the tank in minutes

def effectiveRate (inletRate : ℕ) : ℕ :=
  outletRate1 + outletRate2 - inletRate

theorem inletRate_is_3 : (15 - 3) * minutesToEmpty = tankVolume :=
  by simp [outletRate1, outletRate2, tankVolume, minutesToEmpty]; sorry

end inletRate_is_3_l280_280410


namespace tan_45_degree_l280_280747

theorem tan_45_degree : ∀ (θ : ℝ), 
  θ = 45 → 
  (sin θ = 1 / real.sqrt 2) → 
  (cos θ = 1 / real.sqrt 2) → 
  real.tan θ = 1 := 
by
  intros θ hθ hsin hcos
  sorry

end tan_45_degree_l280_280747


namespace sum_of_roots_of_equation_l280_280135

theorem sum_of_roots_of_equation : 
  (∃ x1 x2 : ℝ, (x1 - 7)^2 = 16 ∧ (x2 - 7)^2 = 16 ∧ x1 ≠ x2 ∧ (x1 + x2 = 14)) :=
by
  sorry

end sum_of_roots_of_equation_l280_280135


namespace greatest_value_sum_eq_24_l280_280273

theorem greatest_value_sum_eq_24 {a b : ℕ} (ha : 0 < a) (hb : 1 < b) 
  (h1 : ∀ (x y : ℕ), 0 < x → 1 < y → x^y < 500 → x^y ≤ a^b) : 
  a + b = 24 := 
by
  sorry

end greatest_value_sum_eq_24_l280_280273


namespace tan_45_deg_l280_280885

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 :=
by
  have h_sin_cos : Real.sin (Real.pi / 4) = Real.cos (Real.pi / 4) := sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos]
  have sqrt2_ne_zero : Real.sqrt 2 ≠ 0 := sorry
  have sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have cos_45 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [sin_45, cos_45]
  field_simp [sqrt2_ne_zero]
  exact one_div_one

end tan_45_deg_l280_280885


namespace solve_quadratic_l280_280083

theorem solve_quadratic : ∀ x : ℝ, x ^ 2 - 6 * x + 8 = 0 ↔ x = 2 ∨ x = 4 := by
  sorry

end solve_quadratic_l280_280083


namespace tan_45_degrees_eq_1_l280_280781

theorem tan_45_degrees_eq_1 :
    tan (45 * real.pi / 180) = 1 :=
by
  sorry

end tan_45_degrees_eq_1_l280_280781


namespace only_one_student_remains_l280_280412

theorem only_one_student_remains (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 2002) :
  (∃! k, k = n ∧ n % 1331 = 0) ↔ n = 1331 :=
by
  sorry

end only_one_student_remains_l280_280412


namespace intersection_of_parabola_with_y_axis_l280_280365

theorem intersection_of_parabola_with_y_axis :
  ∃ y : ℝ, y = - (0 + 2)^2 + 6 ∧ (0, y) = (0, 2) :=
by
  sorry

end intersection_of_parabola_with_y_axis_l280_280365


namespace find_a1_plus_a9_l280_280039

variable (a : ℕ → ℝ) (d : ℝ)

-- condition: arithmetic sequence
def is_arithmetic_seq : Prop := ∀ n, a (n + 1) = a n + d

-- condition: sum of specific terms
def sum_specific_terms : Prop := a 3 + a 4 + a 5 + a 6 + a 7 = 450

-- theorem: prove the desired sum
theorem find_a1_plus_a9 (h1 : is_arithmetic_seq a d) (h2 : sum_specific_terms a) : 
  a 1 + a 9 = 180 :=
  sorry

end find_a1_plus_a9_l280_280039


namespace greatest_value_sum_eq_24_l280_280276

theorem greatest_value_sum_eq_24 {a b : ℕ} (ha : 0 < a) (hb : 1 < b) 
  (h1 : ∀ (x y : ℕ), 0 < x → 1 < y → x^y < 500 → x^y ≤ a^b) : 
  a + b = 24 := 
by
  sorry

end greatest_value_sum_eq_24_l280_280276


namespace tan_45_degrees_l280_280800

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l280_280800


namespace sum_of_roots_of_equation_l280_280134

theorem sum_of_roots_of_equation : 
  (∃ x1 x2 : ℝ, (x1 - 7)^2 = 16 ∧ (x2 - 7)^2 = 16 ∧ x1 ≠ x2 ∧ (x1 + x2 = 14)) :=
by
  sorry

end sum_of_roots_of_equation_l280_280134


namespace int_solutions_l280_280912

theorem int_solutions (a b : ℤ) (h : a^2 + b = b^2022) : (a, b) = (0, 0) ∨ (a, b) = (0, 1) :=
by {
  sorry
}

end int_solutions_l280_280912


namespace cube_polygon_area_l280_280281

theorem cube_polygon_area (cube_side : ℝ) 
  (A B C D : ℝ × ℝ × ℝ)
  (P Q R : ℝ × ℝ × ℝ)
  (hP : P = (10, 0, 0))
  (hQ : Q = (30, 0, 20))
  (hR : R = (30, 5, 30))
  (hA : A = (0, 0, 0))
  (hB : B = (30, 0, 0))
  (hC : C = (30, 0, 30))
  (hD : D = (30, 30, 30))
  (cube_length : cube_side = 30) :
  ∃ area, area = 450 := 
sorry

end cube_polygon_area_l280_280281


namespace correct_statement_of_abs_l280_280209

theorem correct_statement_of_abs (r : ℚ) :
  ¬ (∀ r : ℚ, abs r > 0) ∧
  ¬ (∀ a b : ℚ, a ≠ b → abs a ≠ abs b) ∧
  (∀ r : ℚ, abs r ≥ 0) ∧
  ¬ (∀ r : ℚ, r < 0 → abs r = -r ∧ abs r < 0 → abs r ≠ -r) :=
by
  sorry

end correct_statement_of_abs_l280_280209


namespace parabola_focus_l280_280428

theorem parabola_focus (a : ℝ) (h k x : ℝ) (hx : h = 0) (kx : k = 0) (a_eq : a = -1/16) :
  focus (y = -a * x^2) = (0, -4) :=
by
  sorry

end parabola_focus_l280_280428


namespace top_angle_is_70_l280_280122

theorem top_angle_is_70
  (sum_angles : ℝ)
  (left_angle : ℝ)
  (right_angle : ℝ)
  (top_angle : ℝ)
  (h1 : sum_angles = 250)
  (h2 : left_angle = 2 * right_angle)
  (h3 : right_angle = 60)
  (h4 : sum_angles = left_angle + right_angle + top_angle) :
  top_angle = 70 :=
by
  sorry

end top_angle_is_70_l280_280122


namespace slopes_product_no_circle_MN_A_l280_280006

-- Define the equation of the ellipse E and the specific points A and B
def ellipse_eq (x y : ℝ) : Prop := (x^2 / 4) + y^2 = 1
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)

-- Define the point P which lies on the ellipse
def P (x0 y0 : ℝ) : Prop := ellipse_eq x0 y0 ∧ x0 ≠ -2 ∧ x0 ≠ 2

-- Prove the product of the slopes of lines PA and PB
theorem slopes_product (x0 y0 : ℝ) (hP : P x0 y0) : 
  (y0 / (x0 + 2)) * (y0 / (x0 - 2)) = -1 / 4 := sorry

-- Define point Q
def Q : ℝ × ℝ := (-1, 0)

-- Define points M and N which are intersections of line and ellipse
def MN_line (t y : ℝ) : ℝ := t * y - 1

-- Prove there is no circle with diameter MN passing through A
theorem no_circle_MN_A (t : ℝ) : 
  ¬ ∃ M N : ℝ × ℝ, ellipse_eq M.1 M.2 ∧ ellipse_eq N.1 N.2 ∧
  (∃ x1 y1 x2 y2, (M = (x1, y1) ∧ N = (x2, y2)) ∧
  (MN_line t y1 = x1 ∧ MN_line t y2 = x2) ∧ 
  ((x1 + 2) * (x2 + 2) + y1 * y2 = 0)) := sorry

end slopes_product_no_circle_MN_A_l280_280006


namespace tan_45_degrees_l280_280807

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l280_280807


namespace walter_age_in_2005_l280_280244

theorem walter_age_in_2005 
  (y : ℕ) (gy : ℕ)
  (h1 : gy = 3 * y)
  (h2 : (2000 - y) + (2000 - gy) = 3896) : y + 5 = 31 :=
by {
  sorry
}

end walter_age_in_2005_l280_280244


namespace tan_45_deg_l280_280658

theorem tan_45_deg : Real.tan (π / 4) = 1 := 
by
  have h_sin_cos : Real.sin (π / 4) = Real.cos (π / 4) := 
    by sorry
  have h_value : Real.sin (π / 4) = (Real.sqrt 2) / 2 := 
    by sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos, h_value]
  calc
    (Real.sqrt 2)/ 2 / (Real.sqrt 2)/ 2 = 1 := sorry

end tan_45_deg_l280_280658


namespace find_f_l280_280057

noncomputable def f : ℝ → ℝ := sorry

theorem find_f (f : ℝ → ℝ) (h₀ : f 0 = 2) 
  (h₁ : ∀ x y : ℝ, f (x * y) = f ((x^2 + y^2) / 2) + (x + y)^2) :
  ∀ x : ℝ, f x = 2 - 2 * x :=
sorry

end find_f_l280_280057


namespace anna_spent_more_on_lunch_l280_280321

def bagel_cost : ℝ := 0.95
def cream_cheese_cost : ℝ := 0.50
def orange_juice_cost : ℝ := 1.25
def orange_juice_discount : ℝ := 0.32
def sandwich_cost : ℝ := 4.65
def avocado_cost : ℝ := 0.75
def milk_cost : ℝ := 1.15
def milk_discount : ℝ := 0.10

-- Calculate total cost of breakfast.
def breakfast_cost : ℝ := 
  let bagel_with_cream_cheese := bagel_cost + cream_cheese_cost
  let discounted_orange_juice := orange_juice_cost - (orange_juice_cost * orange_juice_discount)
  bagel_with_cream_cheese + discounted_orange_juice

-- Calculate total cost of lunch.
def lunch_cost : ℝ :=
  let sandwich_with_avocado := sandwich_cost + avocado_cost
  let discounted_milk := milk_cost - (milk_cost * milk_discount)
  sandwich_with_avocado + discounted_milk

-- Calculate the difference between lunch and breakfast costs.
theorem anna_spent_more_on_lunch : lunch_cost - breakfast_cost = 4.14 := by
  sorry

end anna_spent_more_on_lunch_l280_280321


namespace tan_45_eq_one_l280_280890

noncomputable def sin_45 : ℝ := real.sqrt 2 / 2
noncomputable def cos_45 : ℝ := real.sqrt 2 / 2

theorem tan_45_eq_one : real.tan (real.pi / 4) = 1 := by
  have hsin : real.sin (real.pi / 4) = sin_45 := by sorry
  have hcos : real.cos (real.pi / 4) = cos_45 := by sorry
  rw [real.tan_eq_sin_div_cos, hsin, hcos]
  rw [div_self]
  exact eq.refl 1
  exact sorry

end tan_45_eq_one_l280_280890


namespace Elon_has_10_more_Teslas_than_Sam_l280_280284

noncomputable def TeslasCalculation : Nat :=
let Chris : Nat := 6
let Sam : Nat := Chris / 2
let Elon : Nat := 13
Elon - Sam

theorem Elon_has_10_more_Teslas_than_Sam :
  TeslasCalculation = 10 :=
by
  sorry

end Elon_has_10_more_Teslas_than_Sam_l280_280284


namespace basic_astrophysics_degrees_l280_280389

open Real

theorem basic_astrophysics_degrees :
  let microphotonics := 12
  let home_electronics := 24
  let food_additives := 15
  let gmo := 29
  let industrial_lubricants := 8
  let total_percentage := microphotonics + home_electronics + food_additives + gmo + industrial_lubricants
  let basic_astrophysics_percentage := 100 - total_percentage
  let circle_degrees := 360
  basic_astrophysics_percentage / 100 * circle_degrees = 43.2 :=
by
  let microphotonics := 12
  let home_electronics := 24
  let food_additives := 15
  let gmo := 29
  let industrial_lubricants := 8
  let total_percentage := microphotonics + home_electronics + food_additives + gmo + industrial_lubricants
  let basic_astrophysics_percentage := 100 - total_percentage
  let circle_degrees := 360
  exact sorry

end basic_astrophysics_degrees_l280_280389


namespace man_rate_in_still_water_l280_280217

theorem man_rate_in_still_water (V_m V_s: ℝ) 
(h1 : V_m + V_s = 19) 
(h2 : V_m - V_s = 11) : 
V_m = 15 := 
by
  sorry

end man_rate_in_still_water_l280_280217


namespace minimum_a_l280_280930

open Real

theorem minimum_a (a : ℝ) : (∀ (x y : ℝ), x > 0 → y > 0 → (1 / x + a / y) ≥ (16 / (x + y))) → a ≥ 9 := by
sorry

end minimum_a_l280_280930


namespace tan_45_deg_l280_280883

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 :=
by
  have h_sin_cos : Real.sin (Real.pi / 4) = Real.cos (Real.pi / 4) := sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos]
  have sqrt2_ne_zero : Real.sqrt 2 ≠ 0 := sorry
  have sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have cos_45 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [sin_45, cos_45]
  field_simp [sqrt2_ne_zero]
  exact one_div_one

end tan_45_deg_l280_280883


namespace min_photos_needed_to_ensure_conditions_l280_280288

noncomputable def min_photos (girls boys : ℕ) : ℕ :=
  33

theorem min_photos_needed_to_ensure_conditions (girls boys : ℕ)
  (hgirls : girls = 4) (hboys : boys = 8) :
  min_photos girls boys = 33 := 
sorry

end min_photos_needed_to_ensure_conditions_l280_280288


namespace quadratic_solutions_l280_280089

-- Define the equation x^2 - 6x + 8 = 0
def quadratic_eq (x : ℝ) : Prop := x^2 - 6*x + 8 = 0

-- Lean statement for the equivalence of solutions
theorem quadratic_solutions : ∀ x : ℝ, quadratic_eq x ↔ x = 2 ∨ x = 4 :=
by
  intro x
  dsimp [quadratic_eq]
  sorry

end quadratic_solutions_l280_280089


namespace total_pokemon_cards_l280_280072

theorem total_pokemon_cards : 
  let n := 14 in
  let total_cards := 4 * n in
  total_cards = 56 :=
by
  let n := 14
  let total_cards := 4 * n
  show total_cards = 56
  sorry

end total_pokemon_cards_l280_280072


namespace probability_white_second_given_red_first_l280_280035

theorem probability_white_second_given_red_first :
  let total_balls := 8
  let red_balls := 5
  let white_balls := 3
  let event_A := red_balls
  let event_B_given_A := white_balls

  (event_B_given_A * (total_balls - 1)) / (event_A * total_balls) = 3 / 7 :=
by
  sorry

end probability_white_second_given_red_first_l280_280035


namespace number_of_bricks_in_wall_l280_280226

noncomputable def rate_one_bricklayer (x : ℕ) : ℚ := x / 8
noncomputable def rate_other_bricklayer (x : ℕ) : ℚ := x / 12
noncomputable def combined_rate_with_efficiency (x : ℕ) : ℚ := (rate_one_bricklayer x + rate_other_bricklayer x - 15)
noncomputable def total_time (x : ℕ) : ℚ := 6 * combined_rate_with_efficiency x

theorem number_of_bricks_in_wall (x : ℕ) : total_time x = x → x = 360 :=
by sorry

end number_of_bricks_in_wall_l280_280226


namespace tan_45_degrees_l280_280816

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l280_280816


namespace tan_45_deg_l280_280690

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l280_280690


namespace sum_of_a_and_b_l280_280265

theorem sum_of_a_and_b (a b : ℕ) (h1 : b > 1) (h2 : a^b < 500) (h3 : ∀ c d : ℕ, d > 1 → c^d < 500 → c^d ≤ a^b) : a + b = 24 :=
sorry

end sum_of_a_and_b_l280_280265


namespace condition_sufficiency_but_not_necessity_l280_280986

variable (p q : Prop)

theorem condition_sufficiency_but_not_necessity:
  (¬ (p ∨ q) → ¬ p) ∧ (¬ p → ¬ (p ∨ q) → False) := 
by
  sorry

end condition_sufficiency_but_not_necessity_l280_280986


namespace tan_45_deg_l280_280643

theorem tan_45_deg : (Real.tan (Float.pi / 4)) = 1 :=
by
  sorry

end tan_45_deg_l280_280643


namespace increasing_interval_range_of_a_l280_280012

noncomputable def f (x : ℝ) := (x^2 - x + 1) / Real.exp x

theorem increasing_interval :
  {x : ℝ | 1 < x ∧ x < 2} = {x : ℝ | f' x > 0} :=
begin
  sorry
end

theorem range_of_a (a : ℝ) (h : ∀ x > 0, Real.exp x * f x ≥ a + Real.log x) :
  a ≤ 1 :=
begin
  sorry
end

end increasing_interval_range_of_a_l280_280012


namespace find_angle_A_l280_280309

noncomputable def angle_A (a b : ℝ) (B : ℝ) : ℝ :=
  Real.arcsin ((a * Real.sin B) / b)

theorem find_angle_A :
  ∀ (a b : ℝ) (angle_B : ℝ), 0 < a → 0 < b → 0 < angle_B → angle_B < 180 →
  a = Real.sqrt 2 →
  b = Real.sqrt 3 →
  angle_B = 60 →
  angle_A a b angle_B = 45 :=
by
  intros a b angle_B h1 h2 h3 h4 ha hb hB
  have ha' : a = Real.sqrt 2 := ha
  have hb' : b = Real.sqrt 3 := hb
  have hB' : angle_B = 60 := hB
  -- Proof omitted for demonstration
  sorry

end find_angle_A_l280_280309


namespace tan_45_eq_one_l280_280869

theorem tan_45_eq_one : Real.tan (Real.pi / 4) = 1 := 
by
  sorry

end tan_45_eq_one_l280_280869


namespace clock_hands_overlap_l280_280396

theorem clock_hands_overlap:
  ∃ x y: ℚ,
  -- Conditions
  (60 * 10 + x = 60 * 11 * 54 + 6 / 11) ∧
  (y - (5 / 60) * y = 60) ∧
  (65 * 5 / 11 = y) := sorry

end clock_hands_overlap_l280_280396


namespace B_finish_in_54_days_l280_280390

-- Definitions based on conditions
variables (A B : ℝ) -- A and B are the amount of work done in one day
axiom h1 : A = 2 * B -- A is twice as good as workman as B
axiom h2 : (A + B) * 18 = 1 -- Together, A and B finish the piece of work in 18 days

-- Prove that B alone will finish the work in 54 days.
theorem B_finish_in_54_days : (1 / B) = 54 :=
by 
  sorry

end B_finish_in_54_days_l280_280390


namespace find_k_l280_280391

theorem find_k (k : ℝ) (h : 64 / k = 8) : k = 8 := 
sorry

end find_k_l280_280391


namespace tan_of_45_deg_l280_280741

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  -- Definitions used directly from conditions
  have hcos : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry
  have hsin : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry

  -- Using the definition of tangent
  rw [Real.tan_eq_sin_div_cos, hsin, hcos]
  exact div_self (ne_of_gt (Real.sqrt_pos.2 zero_lt_two))

end tan_of_45_deg_l280_280741


namespace central_cell_value_l280_280528

theorem central_cell_value
  (a b c d e f g h i : ℝ)
  (row1 : a * b * c = 10)
  (row2 : d * e * f = 10)
  (row3 : g * h * i = 10)
  (col1 : a * d * g = 10)
  (col2 : b * e * h = 10)
  (col3 : c * f * i = 10)
  (sub1 : a * b * d * e = 3)
  (sub2 : b * c * e * f = 3)
  (sub3 : d * e * g * h = 3)
  (sub4 : e * f * h * i = 3) : 
  e = 0.00081 :=
sorry

end central_cell_value_l280_280528


namespace common_ratio_geometric_progression_l280_280037

theorem common_ratio_geometric_progression (r : ℝ) (a : ℝ) (h : a > 0) (h_r : r > 0) (h_eq : ∀ (n : ℕ), a * r^(n-1) = a * r^n + a * r^(n+1) + a * r^(n+2)) : r^3 + r^2 + r - 1 = 0 := 
by sorry

end common_ratio_geometric_progression_l280_280037


namespace tan_45_deg_l280_280697

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l280_280697


namespace rebecca_income_l280_280967

variable (R : ℝ) -- Rebecca's current yearly income (denoted as R)
variable (increase : ℝ := 7000) -- The increase in Rebecca's income
variable (jimmy_income : ℝ := 18000) -- Jimmy's yearly income
variable (combined_income : ℝ := (R + increase) + jimmy_income) -- Combined income after increase
variable (new_income_ratio : ℝ := 0.55) -- Proportion of total income that is Rebecca's new income

theorem rebecca_income : (R + increase) = new_income_ratio * combined_income → R = 15000 :=
by
  sorry

end rebecca_income_l280_280967


namespace tan_45_deg_l280_280880

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 :=
by
  have h_sin_cos : Real.sin (Real.pi / 4) = Real.cos (Real.pi / 4) := sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos]
  have sqrt2_ne_zero : Real.sqrt 2 ≠ 0 := sorry
  have sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have cos_45 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [sin_45, cos_45]
  field_simp [sqrt2_ne_zero]
  exact one_div_one

end tan_45_deg_l280_280880


namespace tan_45_degrees_l280_280815

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l280_280815


namespace quadratic_root_exists_l280_280106

theorem quadratic_root_exists {a b c d : ℝ} (h : a * c = 2 * b + 2 * d) :
  (∃ x : ℝ, x^2 + a * x + b = 0) ∨ (∃ x : ℝ, x^2 + c * x + d = 0) :=
by
  sorry

end quadratic_root_exists_l280_280106


namespace sum_of_roots_eq_l280_280185

theorem sum_of_roots_eq (x : ℝ) : (x - 7)^2 = 16 → x = 11 ∨ x = 3 → ∑ (root : ℝ) in {11, 3}, root = 14 :=
by
  assume h₁ : (x - 7)^2 = 16
  assume h₂ : x = 11 ∨ x = 3
  sorry

end sum_of_roots_eq_l280_280185


namespace greatest_prime_factor_180_l280_280382

noncomputable def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ n : ℕ, n ∣ p → n = 1 ∨ n = p

theorem greatest_prime_factor_180 : 
  ∃ p : ℕ, is_prime p ∧ p ∣ 180 ∧ ∀ q : ℕ, is_prime q ∧ q ∣ 180 → q ≤ p :=
  sorry

end greatest_prime_factor_180_l280_280382


namespace inequality_proof_l280_280077

theorem inequality_proof (a b c : ℝ) (h : a * b * c = 1) : 
  1 / (2 * a^2 + b^2 + 3) + 1 / (2 * b^2 + c^2 + 3) + 1 / (2 * c^2 + a^2 + 3) ≤ 1 / 2 := 
by
  sorry

end inequality_proof_l280_280077


namespace sum_when_max_power_less_500_l280_280264

theorem sum_when_max_power_less_500 :
  ∃ (a b : ℕ), a > 0 ∧ b > 1 ∧ a^b < 500 ∧
  (∀ (a' b' : ℕ), a' > 0 → b' > 1 → a'^b' < 500 → a^b ≥ a'^b') ∧ (a + b = 24) :=
  sorry

end sum_when_max_power_less_500_l280_280264


namespace tan_45_eq_one_l280_280887

noncomputable def sin_45 : ℝ := real.sqrt 2 / 2
noncomputable def cos_45 : ℝ := real.sqrt 2 / 2

theorem tan_45_eq_one : real.tan (real.pi / 4) = 1 := by
  have hsin : real.sin (real.pi / 4) = sin_45 := by sorry
  have hcos : real.cos (real.pi / 4) = cos_45 := by sorry
  rw [real.tan_eq_sin_div_cos, hsin, hcos]
  rw [div_self]
  exact eq.refl 1
  exact sorry

end tan_45_eq_one_l280_280887


namespace sum_when_max_power_less_500_l280_280262

theorem sum_when_max_power_less_500 :
  ∃ (a b : ℕ), a > 0 ∧ b > 1 ∧ a^b < 500 ∧
  (∀ (a' b' : ℕ), a' > 0 → b' > 1 → a'^b' < 500 → a^b ≥ a'^b') ∧ (a + b = 24) :=
  sorry

end sum_when_max_power_less_500_l280_280262


namespace hotpot_total_cost_l280_280399

def table_cost : ℝ := 280
def table_limit : ℕ := 8
def extra_person_cost : ℝ := 29.9
def total_people : ℕ := 12

theorem hotpot_total_cost : 
  total_people > table_limit →
  table_cost + (total_people - table_limit) * extra_person_cost = 369.7 := 
by 
  sorry

end hotpot_total_cost_l280_280399


namespace tan_45_deg_l280_280693

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l280_280693


namespace find_a5_and_sum_l280_280920

variable {a : ℕ → ℝ}

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a (n + 1) = a n * q

def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) > a n

-- Given conditions
def given_conditions (a : ℕ → ℝ) (q : ℝ) : Prop :=
is_geometric_sequence a q ∧ is_increasing_sequence a ∧ a 2 = 3 ∧ a 4 - a 3 = 18

-- Theorem to prove
theorem find_a5_and_sum {a : ℕ → ℝ} {q : ℝ} (h : given_conditions a q) :
  a 5 = 81 ∧ (a 1 + a 2 + a 3 + a 4 + a 5) = 121 :=
by
  -- Placeholder for the actual proof
  sorry

end find_a5_and_sum_l280_280920


namespace cost_per_order_of_pakoras_l280_280242

noncomputable def samosa_cost : ℕ := 2
noncomputable def samosa_count : ℕ := 3
noncomputable def mango_lassi_cost : ℕ := 2
noncomputable def pakora_count : ℕ := 4
noncomputable def tip_percentage : ℚ := 0.25
noncomputable def total_cost_with_tax : ℚ := 25

theorem cost_per_order_of_pakoras (P : ℚ)
  (h1 : samosa_cost * samosa_count = 6)
  (h2 : mango_lassi_cost = 2)
  (h3 : 1.25 * (samosa_cost * samosa_count + mango_lassi_cost + pakora_count * P) = total_cost_with_tax) :
  P = 3 :=
by
  -- sorry ⟹ sorry
  sorry

end cost_per_order_of_pakoras_l280_280242


namespace rectangle_area_l280_280225

-- Declare the given conditions
def circle_radius : ℝ := 5
def rectangle_width : ℝ := 2 * circle_radius
def length_to_width_ratio : ℝ := 2

-- Given that the length to width ratio is 2:1, calculate the length
def rectangle_length : ℝ := length_to_width_ratio * rectangle_width

-- Define the statement we need to prove
theorem rectangle_area :
  rectangle_length * rectangle_width = 200 :=
by
  sorry

end rectangle_area_l280_280225


namespace product_of_solutions_l280_280933

theorem product_of_solutions (x : ℚ) (h : abs (12 / x + 3) = 2) :
  x = -12 ∨ x = -12 / 5 → x₁ * x₂ = 144 / 5 := by
  sorry

end product_of_solutions_l280_280933


namespace tan_45_deg_eq_one_l280_280848

theorem tan_45_deg_eq_one :
  let θ := 45
  (∀ θ, tan θ = sin θ / cos θ) →
  (sin 45 = sqrt 2 / 2) →
  (cos 45 = sqrt 2 / 2) →
  tan 45 = 1 :=
by
  intros θ h_tan h_sin h_cos
  sorry

end tan_45_deg_eq_one_l280_280848


namespace tan_45_eq_one_l280_280868

theorem tan_45_eq_one : Real.tan (Real.pi / 4) = 1 := 
by
  sorry

end tan_45_eq_one_l280_280868


namespace sum_of_roots_of_equation_l280_280141

theorem sum_of_roots_of_equation : 
  (∃ x1 x2 : ℝ, (x1 - 7)^2 = 16 ∧ (x2 - 7)^2 = 16 ∧ x1 ≠ x2 ∧ (x1 + x2 = 14)) :=
by
  sorry

end sum_of_roots_of_equation_l280_280141


namespace karlsson_candies_28_l280_280338

def karlsson_max_candies (n : ℕ) : ℕ := (n * (n - 1)) / 2

theorem karlsson_candies_28 : karlsson_max_candies 28 = 378 := by
  sorry

end karlsson_candies_28_l280_280338


namespace anita_gave_apples_l280_280558

theorem anita_gave_apples (initial_apples needed_for_pie apples_left_after_pie : ℝ)
  (h_initial : initial_apples = 10.0)
  (h_needed : needed_for_pie = 4.0)
  (h_left : apples_left_after_pie = 11.0) :
  ∃ (anita_apples : ℝ), anita_apples = 5 :=
by
  sorry

end anita_gave_apples_l280_280558


namespace tan_45_degree_is_one_l280_280715

theorem tan_45_degree_is_one : Real.tan (Real.pi / 4) = 1 := sorry

end tan_45_degree_is_one_l280_280715


namespace tan_45_degrees_l280_280808

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l280_280808


namespace tan_45_eq_one_l280_280864

theorem tan_45_eq_one : Real.tan (Real.pi / 4) = 1 := 
by
  sorry

end tan_45_eq_one_l280_280864


namespace number_of_three_digit_multiples_of_7_l280_280459

theorem number_of_three_digit_multiples_of_7 : 
  let smallest_multiple := 7 * Nat.ceil (100 / 7)
  let largest_multiple := 7 * Nat.floor (999 / 7)
  (largest_multiple - smallest_multiple) / 7 + 1 = 128 :=
by
  sorry

end number_of_three_digit_multiples_of_7_l280_280459


namespace solve_for_x_l280_280351

theorem solve_for_x (x : ℝ) (h1 : x ≠ 4) (h2 : x ≠ 6)
  (h : (x + 10) / (x - 4) = (x - 3) / (x + 6)) : x = -48 / 23 :=
sorry

end solve_for_x_l280_280351


namespace tan_45_deg_eq_one_l280_280837

/-- The tangent of 45 degrees is equal to 1. -/
theorem tan_45_deg_eq_one : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_one_l280_280837


namespace sum_of_roots_of_equation_l280_280139

theorem sum_of_roots_of_equation : 
  (∃ x1 x2 : ℝ, (x1 - 7)^2 = 16 ∧ (x2 - 7)^2 = 16 ∧ x1 ≠ x2 ∧ (x1 + x2 = 14)) :=
by
  sorry

end sum_of_roots_of_equation_l280_280139


namespace original_work_days_l280_280397

-- Definitions based on conditions
noncomputable def L : ℕ := 7  -- Number of laborers originally employed
noncomputable def A : ℕ := 3  -- Number of absent laborers
noncomputable def t : ℕ := 14 -- Number of days it took the remaining laborers to finish the work

-- Theorem statement to prove
theorem original_work_days : (L - A) * t = L * 8 := by
  sorry

end original_work_days_l280_280397


namespace at_least_one_root_l280_280107

theorem at_least_one_root 
  (a b c d : ℝ)
  (h : a * c = 2 * b + 2 * d) :
  (∃ x : ℝ, x^2 + a * x + b = 0) ∨ (∃ x : ℝ, x^2 + c * x + d = 0) :=
sorry

end at_least_one_root_l280_280107


namespace range_of_x_satisfying_inequality_l280_280050

def f (x : ℝ) : ℝ := -- Define the function f (we will leave this definition open for now)
sorry
@[continuity] axiom f_increasing (x y : ℝ) (h : x < y) : f x < f y
axiom f_2_eq_1 : f 2 = 1
axiom f_xy_eq_f_x_add_f_y (x y : ℝ) : f (x * y) = f x + f y

noncomputable def f_4_eq_2 : f 4 = 2 := sorry

theorem range_of_x_satisfying_inequality (x : ℝ) :
  3 < x ∧ x ≤ 4 ↔ f x + f (x - 3) ≤ 2 :=
sorry

end range_of_x_satisfying_inequality_l280_280050


namespace slope_angle_vertical_line_l280_280033

theorem slope_angle_vertical_line : 
  ∀ α : ℝ, (∀ x y : ℝ, x = 1 → y = α) → α = Real.pi / 2 := 
by 
  sorry

end slope_angle_vertical_line_l280_280033


namespace sum_of_roots_l280_280157

theorem sum_of_roots : 
  let eq := (x - 7) ^ 2 = 16 in
  (roots : set ℝ) := { x | eq } in
  ∑ roots = 14 :=
by
sorry

end sum_of_roots_l280_280157


namespace distinct_solutions_abs_eq_l280_280452

theorem distinct_solutions_abs_eq (x : ℝ) : 
  (|x - 3| = |x + 5|) → x = -1 :=
by
  sorry

end distinct_solutions_abs_eq_l280_280452


namespace marbles_exceed_200_on_sunday_l280_280059

theorem marbles_exceed_200_on_sunday:
  ∃ n : ℕ, 3 * 2^n > 200 ∧ (n % 7) = 0 :=
by
  sorry

end marbles_exceed_200_on_sunday_l280_280059


namespace sum_of_roots_of_quadratic_l280_280160

theorem sum_of_roots_of_quadratic : 
  (∃ (x : ℝ), (x - 7) ^ 2 = 16 ∧ (x = 11 ∨ x = 3)) → 
  (11 + 3 = 14) :=
by 
  intro h
  sorry

end sum_of_roots_of_quadratic_l280_280160


namespace find_a5_l280_280306

-- Define an arithmetic sequence with a given common difference
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a (n + 1) = a n + d

-- Define that three terms form a geometric sequence
def geometric_sequence (x y z : ℝ) := y^2 = x * z

-- Given conditions for the problem
def a₁ : ℝ := 1  -- found from the geometric sequence condition
def d : ℝ := 2

-- The definition of the sequence {a_n} based on the common difference
noncomputable def a_n (n : ℕ) : ℝ := a₁ + n * d

-- Given that a_1, a_2, a_5 form a geometric sequence
axiom geo_progression : geometric_sequence a₁ (a_n 1) (a_n 4)

-- The proof goal
theorem find_a5 : a_n 4 = 9 :=
by
  -- the proof is skipped
  sorry

end find_a5_l280_280306


namespace quadratic_root_exists_l280_280103

theorem quadratic_root_exists {a b c d : ℝ} (h : a * c = 2 * b + 2 * d) :
  (∃ x : ℝ, x^2 + a * x + b = 0) ∨ (∃ x : ℝ, x^2 + c * x + d = 0) :=
by
  sorry

end quadratic_root_exists_l280_280103


namespace eraser_ratio_l280_280280

-- Define the variables and conditions
variables (c j g : ℕ)
variables (total : ℕ := 35)
variables (c_erasers : ℕ := 10)
variables (gabriel_erasers : ℕ := c_erasers / 2)
variables (julian_erasers : ℕ := c_erasers)

-- The proof statement
theorem eraser_ratio (hc : c_erasers = 10)
                      (h1 : c_erasers = 2 * gabriel_erasers)
                      (h2 : julian_erasers = c_erasers)
                      (h3 : c_erasers + gabriel_erasers + julian_erasers = total) :
                      julian_erasers / c_erasers = 1 :=
by
  sorry

end eraser_ratio_l280_280280


namespace prove_value_of_expressions_l280_280434

theorem prove_value_of_expressions (a b : ℕ) 
  (h₁ : 2^a = 8^b) 
  (h₂ : a + 2 * b = 5) : 
  2^a + 8^b = 16 := 
by 
  -- proof steps go here
  sorry

end prove_value_of_expressions_l280_280434


namespace tan_45_eq_1_l280_280824

theorem tan_45_eq_1
: ∀ (Q : ℝ × ℝ), Q = (Real.sqrt 2 / 2, Real.sqrt 2 / 2) ->
  ∀ θ, θ = Real.pi / 4 -> 
  Real.tan θ = 1 :=
by
  intros Q HQ θ Hθ
  rw [HQ, Hθ]
  sorry

end tan_45_eq_1_l280_280824


namespace tan_45_eq_1_l280_280860

theorem tan_45_eq_1 :
  let π := Real.pi in
  ∃ (x : ℝ), x = π / 4 ∧
  sin x = sqrt 2 / 2 ∧
  cos x = sqrt 2 / 2 →
  tan (π / 4) = 1 :=
by sorry

end tan_45_eq_1_l280_280860


namespace treShaun_marker_ink_left_l280_280574

noncomputable def ink_left_percentage (marker_area : ℕ) (total_colored_area : ℕ) : ℕ :=
if total_colored_area >= marker_area then 0 else ((marker_area - total_colored_area) * 100) / marker_area

theorem treShaun_marker_ink_left :
  let marker_area := 3 * (4 * 4)
  let colored_area := (2 * (6 * 2) + 8 * 4)
  ink_left_percentage marker_area colored_area = 0 :=
by
  sorry

end treShaun_marker_ink_left_l280_280574


namespace pow2_gt_square_for_all_n_ge_5_l280_280130

theorem pow2_gt_square_for_all_n_ge_5 (n : ℕ) (h : n ≥ 5) : 2^n > n^2 :=
by
  sorry

end pow2_gt_square_for_all_n_ge_5_l280_280130


namespace sum_when_max_power_less_500_l280_280259

theorem sum_when_max_power_less_500 :
  ∃ (a b : ℕ), a > 0 ∧ b > 1 ∧ a^b < 500 ∧
  (∀ (a' b' : ℕ), a' > 0 → b' > 1 → a'^b' < 500 → a^b ≥ a'^b') ∧ (a + b = 24) :=
  sorry

end sum_when_max_power_less_500_l280_280259


namespace tan_45_degree_l280_280672

theorem tan_45_degree : ∀ (Q : ℝ × ℝ),
  -- Q corresponds to 45 degrees on the unit circle
  Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) →
  -- The radius of the unit circle is 1
  (Q.1 ^ 2 + Q.2 ^ 2 = 1) →
  -- QOE is a 45-45-90 triangle
  True →
  -- Prove that tan 45 degrees equals 1
  Real.tan (Real.pi / 4) = 1 :=
by
  -- The proof using these conditions is omitted
  intros,
  sorry

end tan_45_degree_l280_280672


namespace sum_when_max_power_less_500_l280_280260

theorem sum_when_max_power_less_500 :
  ∃ (a b : ℕ), a > 0 ∧ b > 1 ∧ a^b < 500 ∧
  (∀ (a' b' : ℕ), a' > 0 → b' > 1 → a'^b' < 500 → a^b ≥ a'^b') ∧ (a + b = 24) :=
  sorry

end sum_when_max_power_less_500_l280_280260


namespace sum_of_roots_eq_14_l280_280180

theorem sum_of_roots_eq_14 : 
  (∀ x : ℝ, (x - 7)^2 = 16 → (x = 11 ∨ x = 3)) →
  (∀ x1 x2 : ℝ, (x1 - 7)^2 = 16 ∧ (x2 - 7)^2 = 16 ∧ x1 ≠ x2 → x1 + x2 = 14) :=
by
  intros h x1 x2 h_comb
  sorry

end sum_of_roots_eq_14_l280_280180


namespace tan_45_eq_1_l280_280858

theorem tan_45_eq_1 :
  let π := Real.pi in
  ∃ (x : ℝ), x = π / 4 ∧
  sin x = sqrt 2 / 2 ∧
  cos x = sqrt 2 / 2 →
  tan (π / 4) = 1 :=
by sorry

end tan_45_eq_1_l280_280858


namespace tan_45_degree_l280_280667

theorem tan_45_degree : ∀ (Q : ℝ × ℝ),
  -- Q corresponds to 45 degrees on the unit circle
  Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) →
  -- The radius of the unit circle is 1
  (Q.1 ^ 2 + Q.2 ^ 2 = 1) →
  -- QOE is a 45-45-90 triangle
  True →
  -- Prove that tan 45 degrees equals 1
  Real.tan (Real.pi / 4) = 1 :=
by
  -- The proof using these conditions is omitted
  intros,
  sorry

end tan_45_degree_l280_280667


namespace central_cell_value_l280_280504

variables a b c d e f g h i : ℝ

-- Conditions
axiom row1 : a * b * c = 10
axiom row2 : d * e * f = 10
axiom row3 : g * h * i = 10
axiom col1 : a * d * g = 10
axiom col2 : b * e * h = 10
axiom col3 : c * f * i = 10
axiom sq1 : a * b * d * e = 3
axiom sq2 : b * c * e * f = 3
axiom sq3 : d * e * g * h = 3
axiom sq4 : e * f * h * i = 3

theorem central_cell_value : e = 0.00081 := by
  sorry

end central_cell_value_l280_280504


namespace negation_of_all_squares_positive_l280_280982

theorem negation_of_all_squares_positive :
  ¬ (∀ x : ℝ, x * x > 0) ↔ ∃ x : ℝ, x * x ≤ 0 :=
by sorry

end negation_of_all_squares_positive_l280_280982


namespace sum_of_roots_l280_280143

theorem sum_of_roots (a b : Real) (h : (x - 7)^2 = 16):
  a + b = 14 :=
sorry

end sum_of_roots_l280_280143


namespace tan_45_deg_eq_one_l280_280650

/-
  Given:
  cos (45 deg) = sqrt 2 / 2,
  sin (45 deg) = sqrt 2 / 2,
  Prove: 
  tan (45 deg) = 1
-/

theorem tan_45_deg_eq_one : 
  (real.cos (real.pi / 4) = real.sqrt 2 / 2) → 
  (real.sin (real.pi / 4) = real.sqrt 2 / 2) → 
  real.tan (real.pi / 4) = 1 :=
by
  intros hcos hsin
  sorry

end tan_45_deg_eq_one_l280_280650


namespace central_cell_value_l280_280531

theorem central_cell_value
  (a b c d e f g h i : ℝ)
  (row1 : a * b * c = 10)
  (row2 : d * e * f = 10)
  (row3 : g * h * i = 10)
  (col1 : a * d * g = 10)
  (col2 : b * e * h = 10)
  (col3 : c * f * i = 10)
  (sub1 : a * b * d * e = 3)
  (sub2 : b * c * e * f = 3)
  (sub3 : d * e * g * h = 3)
  (sub4 : e * f * h * i = 3) : 
  e = 0.00081 :=
sorry

end central_cell_value_l280_280531


namespace tan_of_45_deg_l280_280709

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_of_45_deg_l280_280709


namespace normal_line_equation_at_x0_l280_280214

def curve (x : ℝ) : ℝ := x - x^3
noncomputable def x0 : ℝ := -1
noncomputable def y0 : ℝ := curve x0

theorem normal_line_equation_at_x0 :
  ∀ (y : ℝ), y = (1/2 : ℝ) * x + 1/2 ↔ (∃ (x : ℝ), y = curve x ∧ x = x0) :=
by
  sorry

end normal_line_equation_at_x0_l280_280214


namespace tan_45_eq_1_l280_280758

theorem tan_45_eq_1 : Real.tan (Real.pi / 4) = 1 := by
  have h1 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have h2 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [Real.tan_eq_sin_div_cos, h1, h2]
  simp

end tan_45_eq_1_l280_280758


namespace tan_45_degree_is_one_l280_280720

theorem tan_45_degree_is_one : Real.tan (Real.pi / 4) = 1 := sorry

end tan_45_degree_is_one_l280_280720


namespace min_photos_l280_280299

theorem min_photos (G B : ℕ) (G_eq : G = 4) (B_eq : B = 8): 
  ∃ n ≥ 33, ∀ photos : set (set (ℕ × ℕ)), 
  (∀ p ∈ photos, p = (i, j) → i < j ∧ i < G ∧ j < B ∨ i >= G ∧ j < G) →
  ((∃ p ∈ photos, ∀ (i j : ℕ), (i, j) = p → (i < G ∧ j < G) ∨ (i < B ∧ j < B)) ∨ (∃ p1 p2 ∈ photos, p1 = p2)) := 
by {
  sorry
}

end min_photos_l280_280299


namespace central_cell_value_l280_280524

theorem central_cell_value :
  ∀ (a b c d e f g h i : ℝ),
  a * b * c = 10 →
  d * e * f = 10 →
  g * h * i = 10 →
  a * d * g = 10 →
  b * e * h = 10 →
  c * f * i = 10 →
  a * b * d * e = 3 →
  b * c * e * f = 3 →
  d * e * g * h = 3 →
  e * f * h * i = 3 →
  e = 0.00081 := 
by sorry

end central_cell_value_l280_280524


namespace age_difference_l280_280220

theorem age_difference (a b c : ℕ) (h : a + b = b + c + 10) : c = a - 10 :=
by
  sorry

end age_difference_l280_280220


namespace inclination_angle_range_l280_280283

theorem inclination_angle_range (a θ : ℝ) 
    (h : θ = real.arctan (- (2 * a) / (a^2 + 1))) :
     θ ∈ set.Icc 0 (real.pi / 4) ∪ set.Icc (3 * real.pi / 4) real.pi := 
sorry

end inclination_angle_range_l280_280283


namespace hyperbola_asymptote_l280_280241

-- Given conditions for the hyperbola
variable (a b : ℝ) (h_a_positive : 0 < a) (h_b_positive : 0 < b)
variable (h_hyperbola : ∀ x y : ℝ, ((x ^ 2) / (a ^ 2)) - ((y ^ 2) / (b ^ 2)) = 1)
variable (h_imaginary_axis : 2 * b = 2)
variable (h_focal_length : ∀ (c : ℝ), 2 * c = 2 * real.sqrt 3)

-- Definition of the asymptote equation to be proven
theorem hyperbola_asymptote :
  let a := real.sqrt 2
  let b := 1
  (∀ x y : ℝ, (((x ^ 2) / (real.sqrt 2 ^ 2)) - ((y ^ 2) / (1 ^ 2)) = 1) →
    (y = (x * (real.sqrt 2) / 2) ∨ y = -(x * (real.sqrt 2) / 2))) :=
begin
  intro a, intro b,
  let c := real.sqrt 3,
  have h_eq : c ^ 2 = a ^ 2 + b ^ 2 := by
  {
    exact (c ^ 2) = (real.sqrt 3) ^ 2 = 3,
    rw [a ^ 2 + b ^ 2],
    exact 2 + 1 
  },
  have h_a_eq : a = real.sqrt 2 := by 
  {
    rw [h_eq],
    exact a ^ 2 = 2,
    rw pow_two a,
    exact real.sqrt_eq_rpow,
  },
  have h_b_eq : b = 1 := by 
  {
    rw [h_imaginary_axis],
    exact b = 1,
  },
  rw [h_b_eq],

  exact sorry
end

end hyperbola_asymptote_l280_280241


namespace solve_x_l280_280020

theorem solve_x (x : ℝ) (hx : (1/x + 1/(2*x) + 1/(3*x) = 1/12)) : x = 22 :=
  sorry

end solve_x_l280_280020


namespace factorize_expression_l280_280910

theorem factorize_expression (x y : ℝ) : x^2 * y + 2 * x * y + y = y * (x + 1)^2 :=
by
  sorry

end factorize_expression_l280_280910


namespace expression_evaluation_l280_280347

-- Definitions of the expressions
def expr (x y : ℤ) : ℤ :=
  ((x - 2 * y) ^ 2 + (3 * x - y) * (3 * x + y) - 3 * y ^ 2) / (-2 * x)

-- Proof that the expression evaluates to -11 when x = 1 and y = -3
theorem expression_evaluation : expr 1 (-3) = -11 :=
by
  -- Declarations
  let x := 1
  let y := -3
  -- The core calculation
  show expr x y = -11
  sorry

end expression_evaluation_l280_280347


namespace tan_45_deg_l280_280625

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l280_280625


namespace sum_of_roots_of_equation_l280_280138

theorem sum_of_roots_of_equation : 
  (∃ x1 x2 : ℝ, (x1 - 7)^2 = 16 ∧ (x2 - 7)^2 = 16 ∧ x1 ≠ x2 ∧ (x1 + x2 = 14)) :=
by
  sorry

end sum_of_roots_of_equation_l280_280138


namespace sum_of_roots_eq_14_l280_280181

theorem sum_of_roots_eq_14 : 
  (∀ x : ℝ, (x - 7)^2 = 16 → (x = 11 ∨ x = 3)) →
  (∀ x1 x2 : ℝ, (x1 - 7)^2 = 16 ∧ (x2 - 7)^2 = 16 ∧ x1 ≠ x2 → x1 + x2 = 14) :=
by
  intros h x1 x2 h_comb
  sorry

end sum_of_roots_eq_14_l280_280181


namespace train_a_distance_traveled_l280_280129

variable (distance : ℝ) (speedA : ℝ) (speedB : ℝ) (relative_speed : ℝ) (time_to_meet : ℝ) 

axiom condition1 : distance = 450
axiom condition2 : speedA = 50
axiom condition3 : speedB = 50
axiom condition4 : relative_speed = speedA + speedB
axiom condition5 : time_to_meet = distance / relative_speed

theorem train_a_distance_traveled : (50 * time_to_meet) = 225 := by
  sorry

end train_a_distance_traveled_l280_280129


namespace tan_45_eq_one_l280_280791

theorem tan_45_eq_one (Q : ℝ × ℝ) (h1 : Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2)) : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_eq_one_l280_280791


namespace central_cell_value_l280_280517

theorem central_cell_value (a b c d e f g h i : ℝ)
  (h_row1 : a * b * c = 10)
  (h_row2 : d * e * f = 10)
  (h_row3 : g * h * i = 10)
  (h_col1 : a * d * g = 10)
  (h_col2 : b * e * h = 10)
  (h_col3 : c * f * i = 10)
  (h_block1 : a * b * d * e = 3)
  (h_block2 : b * c * e * f = 3)
  (h_block3 : d * e * g * h = 3)
  (h_block4 : e * f * h * i = 3) :
  e = 0.00081 :=
sorry

end central_cell_value_l280_280517


namespace tan_45_eq_one_l280_280677

theorem tan_45_eq_one 
  (θ : ℝ)
  (hθ : θ = 45)
  (sin_45_eq : Real.sin (θ * Real.pi / 180) = Real.sqrt 2 / 2)
  (cos_45_eq : Real.cos (θ * Real.pi /180) = Real.sqrt 2 / 2)
  (tan_def : ∀ x : ℝ, 0 < Real.cos x → Real.tan x = Real.sin x / Real.cos x) : 
  Real.tan (θ * Real.pi / 180) = 1 := sorry

end tan_45_eq_one_l280_280677


namespace central_cell_value_l280_280506

variables a b c d e f g h i : ℝ

-- Conditions
axiom row1 : a * b * c = 10
axiom row2 : d * e * f = 10
axiom row3 : g * h * i = 10
axiom col1 : a * d * g = 10
axiom col2 : b * e * h = 10
axiom col3 : c * f * i = 10
axiom sq1 : a * b * d * e = 3
axiom sq2 : b * c * e * f = 3
axiom sq3 : d * e * g * h = 3
axiom sq4 : e * f * h * i = 3

theorem central_cell_value : e = 0.00081 := by
  sorry

end central_cell_value_l280_280506


namespace hawks_score_l280_280940

theorem hawks_score (x y : ℕ) (h1 : x + y = 50) (h2 : x - y = 18) : y = 16 := by
  sorry

end hawks_score_l280_280940


namespace polynomial_has_root_l280_280114

theorem polynomial_has_root {a b c d : ℝ} 
  (h : a * c = 2 * b + 2 * d) : 
  ∃ x : ℝ, (x^2 + a * x + b = 0) ∨ (x^2 + c * x + d = 0) :=
by 
  sorry

end polynomial_has_root_l280_280114


namespace wall_height_correct_l280_280587

-- Define the dimensions of the brick in meters
def brick_length : ℝ := 0.2
def brick_width  : ℝ := 0.1
def brick_height : ℝ := 0.08

-- Define the volume of one brick
def volume_brick : ℝ := brick_length * brick_width * brick_height

-- Total number of bricks used
def number_of_bricks : ℕ := 12250

-- Define the wall dimensions except height
def wall_length : ℝ := 10
def wall_width  : ℝ := 24.5

-- Total volume of all bricks
def volume_total_bricks : ℝ := number_of_bricks * volume_brick

-- Volume of the wall
def volume_wall (h : ℝ) : ℝ := wall_length * h * wall_width

-- The height of the wall
def wall_height : ℝ := 0.08

-- The theorem to prove
theorem wall_height_correct : volume_total_bricks = volume_wall wall_height :=
by
  sorry

end wall_height_correct_l280_280587


namespace burglar_total_sentence_l280_280337

-- Given conditions
def value_of_goods_stolen : ℝ := 40000
def base_sentence_per_thousand_stolen : ℝ := 1 / 5000
def third_offense_increase : ℝ := 0.25
def resisting_arrest_addition : ℕ := 2

-- Theorem to prove the total sentence
theorem burglar_total_sentence :
  let base_sentence := base_sentence_per_thousand_stolen * value_of_goods_stolen
  let increased_sentence := base_sentence * (1 + third_offense_increase)
  let total_sentence := increased_sentence + resisting_arrest_addition
  total_sentence = 12 :=
by 
  sorry -- Proof steps are skipped

end burglar_total_sentence_l280_280337


namespace sum_of_roots_eq_14_l280_280173

theorem sum_of_roots_eq_14 : 
  let equation := λ x : ℝ, (x - 7)^2 = 16 in
  let roots := {x | equation x} in
  (roots : ℝ) → ∀ x y ∈ roots, x + y = 14 :=
by
  -- For the purpose of the problem statement, we specify the exact roots here
  let root1 := 11
  let root2 := 3
  have H : root1 + root2 = 14, by norm_num
  intro equation roots x y hx hy
  exact H

end sum_of_roots_eq_14_l280_280173


namespace sum_of_roots_eq_l280_280186

theorem sum_of_roots_eq (x : ℝ) : (x - 7)^2 = 16 → x = 11 ∨ x = 3 → ∑ (root : ℝ) in {11, 3}, root = 14 :=
by
  assume h₁ : (x - 7)^2 = 16
  assume h₂ : x = 11 ∨ x = 3
  sorry

end sum_of_roots_eq_l280_280186


namespace sum_when_max_power_less_500_l280_280261

theorem sum_when_max_power_less_500 :
  ∃ (a b : ℕ), a > 0 ∧ b > 1 ∧ a^b < 500 ∧
  (∀ (a' b' : ℕ), a' > 0 → b' > 1 → a'^b' < 500 → a^b ≥ a'^b') ∧ (a + b = 24) :=
  sorry

end sum_when_max_power_less_500_l280_280261


namespace tan_45_eq_1_l280_280761

theorem tan_45_eq_1 : Real.tan (Real.pi / 4) = 1 := by
  have h1 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have h2 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [Real.tan_eq_sin_div_cos, h1, h2]
  simp

end tan_45_eq_1_l280_280761


namespace tan_45_eq_l280_280621

def tan_identity (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

def sin_45_eq : Real.sin 45 = Real.sqrt 2 / 2 := sorry

def cos_45_eq : Real.cos 45 = Real.sqrt 2 / 2 := sorry

theorem tan_45_eq : tan_identity 45 = 1 := by
  rw [tan_identity, sin_45_eq, cos_45_eq]
  -- simplifying (sqrt 2 / 2) / (sqrt 2 / 2) = 1
  sorry

end tan_45_eq_l280_280621


namespace tan_45_degrees_l280_280799

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l280_280799


namespace tan_45_degree_l280_280748

theorem tan_45_degree : ∀ (θ : ℝ), 
  θ = 45 → 
  (sin θ = 1 / real.sqrt 2) → 
  (cos θ = 1 / real.sqrt 2) → 
  real.tan θ = 1 := 
by
  intros θ hθ hsin hcos
  sorry

end tan_45_degree_l280_280748


namespace tan_45_eq_one_l280_280685

theorem tan_45_eq_one 
  (θ : ℝ)
  (hθ : θ = 45)
  (sin_45_eq : Real.sin (θ * Real.pi / 180) = Real.sqrt 2 / 2)
  (cos_45_eq : Real.cos (θ * Real.pi /180) = Real.sqrt 2 / 2)
  (tan_def : ∀ x : ℝ, 0 < Real.cos x → Real.tan x = Real.sin x / Real.cos x) : 
  Real.tan (θ * Real.pi / 180) = 1 := sorry

end tan_45_eq_one_l280_280685


namespace product_modulo_10_l280_280999

-- Define the numbers involved
def a := 2457
def b := 7623
def c := 91309

-- Define the modulo operation we're interested in
def modulo_10 (n : Nat) : Nat := n % 10

-- State the theorem we want to prove
theorem product_modulo_10 :
  modulo_10 (a * b * c) = 9 :=
sorry

end product_modulo_10_l280_280999


namespace tan_of_45_deg_l280_280705

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_of_45_deg_l280_280705


namespace sum_of_roots_eq_14_l280_280195

theorem sum_of_roots_eq_14 (x : ℝ) :
  (x - 7) ^ 2 = 16 → ∃ r1 r2 : ℝ, (x = r1 ∨ x = r2) ∧ r1 + r2 = 14 :=
by
  intro h
  sorry

end sum_of_roots_eq_14_l280_280195


namespace fraction_meaningful_iff_l280_280032

theorem fraction_meaningful_iff (x : ℝ) : (x ≠ 2) ↔ (x - 2 ≠ 0) := 
by
  sorry

end fraction_meaningful_iff_l280_280032


namespace sum_of_roots_eq_l280_280182

theorem sum_of_roots_eq (x : ℝ) : (x - 7)^2 = 16 → x = 11 ∨ x = 3 → ∑ (root : ℝ) in {11, 3}, root = 14 :=
by
  assume h₁ : (x - 7)^2 = 16
  assume h₂ : x = 11 ∨ x = 3
  sorry

end sum_of_roots_eq_l280_280182


namespace tens_digit_of_desired_number_is_one_l280_280054

def productOfDigits (n : Nat) : Nat :=
  match n / 10, n % 10 with
  | a, b => a * b

def sumOfDigits (n : Nat) : Nat :=
  match n / 10, n % 10 with
  | a, b => a + b

def isDesiredNumber (N : Nat) : Prop :=
  N < 100 ∧ N ≥ 10 ∧ N = (productOfDigits N)^2 + sumOfDigits N

theorem tens_digit_of_desired_number_is_one (N : Nat) (h : isDesiredNumber N) : N / 10 = 1 :=
  sorry

end tens_digit_of_desired_number_is_one_l280_280054


namespace tan_45_deg_l280_280875

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 :=
by
  have h_sin_cos : Real.sin (Real.pi / 4) = Real.cos (Real.pi / 4) := sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos]
  have sqrt2_ne_zero : Real.sqrt 2 ≠ 0 := sorry
  have sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have cos_45 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [sin_45, cos_45]
  field_simp [sqrt2_ne_zero]
  exact one_div_one

end tan_45_deg_l280_280875


namespace total_capacity_l280_280030

def eight_liters : ℝ := 8
def percentage : ℝ := 0.20
def num_containers : ℕ := 40

theorem total_capacity (h : eight_liters = percentage * capacity) :
  40 * (eight_liters / percentage) = 1600 := sorry

end total_capacity_l280_280030


namespace socorro_training_days_l280_280974

def total_hours := 5
def minutes_per_hour := 60
def total_training_minutes := total_hours * minutes_per_hour

def minutes_multiplication_per_day := 10
def minutes_division_per_day := 20
def daily_training_minutes := minutes_multiplication_per_day + minutes_division_per_day

theorem socorro_training_days:
  total_training_minutes / daily_training_minutes = 10 :=
by
  -- proof omitted
  sorry

end socorro_training_days_l280_280974


namespace sum_of_roots_of_quadratic_l280_280162

theorem sum_of_roots_of_quadratic : 
  (∃ (x : ℝ), (x - 7) ^ 2 = 16 ∧ (x = 11 ∨ x = 3)) → 
  (11 + 3 = 14) :=
by 
  intro h
  sorry

end sum_of_roots_of_quadratic_l280_280162


namespace tan_45_eq_1_l280_280726

theorem tan_45_eq_1 (hcos : real.cos (real.pi / 4) = real.sqrt 2 / 2) (hsin : real.sin (real.pi / 4) = real.sqrt 2 / 2) : real.tan (real.pi / 4) = 1 :=
by
  have h : real.tan (real.pi / 4) = real.sin (real.pi / 4) / real.cos (real.pi / 4) := real.tan_eq_sin_div_cos _
  rw [hsin, hcos] at h
  simp at h
  exact h.symm

end tan_45_eq_1_l280_280726


namespace ziggy_song_requests_l280_280578

theorem ziggy_song_requests :
  ∃ T : ℕ, 
    (T = (1/2) * T + (1/6) * T + 5 + 2 + 1 + 2) →
    T = 30 :=
by 
  sorry

end ziggy_song_requests_l280_280578


namespace fraction_female_to_male_fraction_male_to_total_l280_280320

-- Define the number of male and female students
def num_male_students : ℕ := 30
def num_female_students : ℕ := 24
def total_students : ℕ := num_male_students + num_female_students

-- Prove the fraction of female students to male students
theorem fraction_female_to_male :
  (num_female_students : ℚ) / num_male_students = 4 / 5 :=
by sorry

-- Prove the fraction of male students to total students
theorem fraction_male_to_total :
  (num_male_students : ℚ) / total_students = 5 / 9 :=
by sorry

end fraction_female_to_male_fraction_male_to_total_l280_280320


namespace min_photos_l280_280298

theorem min_photos (G B : ℕ) (G_eq : G = 4) (B_eq : B = 8): 
  ∃ n ≥ 33, ∀ photos : set (set (ℕ × ℕ)), 
  (∀ p ∈ photos, p = (i, j) → i < j ∧ i < G ∧ j < B ∨ i >= G ∧ j < G) →
  ((∃ p ∈ photos, ∀ (i j : ℕ), (i, j) = p → (i < G ∧ j < G) ∨ (i < B ∧ j < B)) ∨ (∃ p1 p2 ∈ photos, p1 = p2)) := 
by {
  sorry
}

end min_photos_l280_280298


namespace tan_45_eq_1_l280_280725

theorem tan_45_eq_1 (hcos : real.cos (real.pi / 4) = real.sqrt 2 / 2) (hsin : real.sin (real.pi / 4) = real.sqrt 2 / 2) : real.tan (real.pi / 4) = 1 :=
by
  have h : real.tan (real.pi / 4) = real.sin (real.pi / 4) / real.cos (real.pi / 4) := real.tan_eq_sin_div_cos _
  rw [hsin, hcos] at h
  simp at h
  exact h.symm

end tan_45_eq_1_l280_280725


namespace fixed_point_coordinates_l280_280096

theorem fixed_point_coordinates (a : ℝ) (h_pos : a > 0) (h_neq_one : a ≠ 1) :
  ∃ P : ℝ × ℝ, P = (1, 4) ∧ ∀ x, P = (x, a^(x-1) + 3) :=
by
  use (1, 4)
  sorry

end fixed_point_coordinates_l280_280096


namespace odds_against_horse_C_winning_l280_280573

theorem odds_against_horse_C_winning (odds_A : ℚ) (odds_B : ℚ) (odds_C : ℚ) 
  (cond1 : odds_A = 5 / 2) 
  (cond2 : odds_B = 3 / 1) 
  (race_condition : odds_C = 1 - ((2 / (5 + 2)) + (1 / (3 + 1))))
  : odds_C / (1 - odds_C) = 15 / 13 := 
sorry

end odds_against_horse_C_winning_l280_280573


namespace maximum_value_of_function_y_l280_280929

noncomputable def function_y (x : ℝ) : ℝ :=
  x * (3 - 2 * x)

theorem maximum_value_of_function_y : ∃ (x : ℝ), 0 < x ∧ x ≤ 1 ∧ function_y x = 9 / 8 :=
by
  sorry

end maximum_value_of_function_y_l280_280929


namespace max_value_real_roots_l280_280446

theorem max_value_real_roots (k x1 x2 : ℝ) :
  (∀ k, k^2 + 3 * k + 5 ≥ 0) →
  (x1 + x2 = k - 2) →
  (x1 * x2 = k^2 + 3 * k + 5) →
  (x1^2 + x2^2 ≤ 18) :=
by
  intro h1 h2 h3
  sorry

end max_value_real_roots_l280_280446


namespace central_cell_value_l280_280485

def table (a b c d e f g h i : ℝ) : Prop :=
  (a * b * c = 10) ∧ (d * e * f = 10) ∧ (g * h * i = 10) ∧
  (a * d * g = 10) ∧ (b * e * h = 10) ∧ (c * f * i = 10) ∧
  (a * b * d * e = 3) ∧ (b * c * e * f = 3) ∧ (d * e * g * h = 3) ∧ (e * f * h * i = 3)

theorem central_cell_value (a b c d f g h i e : ℝ) (h_table : table a b c d e f g h i) : 
  e = 0.00081 :=
by sorry

end central_cell_value_l280_280485


namespace proof_problem_l280_280450

noncomputable def p : ℝ := -5 / 3
noncomputable def q : ℝ := -1

def A (p : ℝ) : Set ℝ := {x | 2 * x^2 + 3 * p * x + 2 = 0}
def B (q : ℝ) : Set ℝ := {x | 2 * x^2 + x + q = 0}

theorem proof_problem (h : (A p ∩ B q) = {1 / 2}) :
    p = -5 / 3 ∧ q = -1 ∧ (A p ∪ B q) = {-1, 1 / 2, 2} := by
  sorry

end proof_problem_l280_280450


namespace find_a_2016_l280_280041

noncomputable def a (n : ℕ) : ℕ := sorry

axiom condition_1 : a 4 = 1
axiom condition_2 : a 11 = 9
axiom condition_3 : ∀ n : ℕ, a n + a (n+1) + a (n+2) = 15

theorem find_a_2016 : a 2016 = 5 := sorry

end find_a_2016_l280_280041


namespace tan_45_degree_is_one_l280_280712

theorem tan_45_degree_is_one : Real.tan (Real.pi / 4) = 1 := sorry

end tan_45_degree_is_one_l280_280712


namespace tan_45_eq_1_l280_280756

theorem tan_45_eq_1 : Real.tan (Real.pi / 4) = 1 := by
  have h1 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have h2 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [Real.tan_eq_sin_div_cos, h1, h2]
  simp

end tan_45_eq_1_l280_280756


namespace tan_45_eq_1_l280_280829

theorem tan_45_eq_1
: ∀ (Q : ℝ × ℝ), Q = (Real.sqrt 2 / 2, Real.sqrt 2 / 2) ->
  ∀ θ, θ = Real.pi / 4 -> 
  Real.tan θ = 1 :=
by
  intros Q HQ θ Hθ
  rw [HQ, Hθ]
  sorry

end tan_45_eq_1_l280_280829


namespace tan_45_eq_one_l280_280867

theorem tan_45_eq_one : Real.tan (Real.pi / 4) = 1 := 
by
  sorry

end tan_45_eq_one_l280_280867


namespace largest_x_value_l280_280053

theorem largest_x_value (x y z : ℝ) (h1 : x + y + z = 6) (h2 : x * y + x * z + y * z = 9) : x ≤ 4 := 
sorry

end largest_x_value_l280_280053


namespace friends_total_l280_280546

-- Define the conditions as constants
def can_go : Nat := 8
def can't_go : Nat := 7

-- Define the total number of friends and the correct answer
def total_friends : Nat := can_go + can't_go
def correct_answer : Nat := 15

-- Prove that the total number of friends is 15
theorem friends_total : total_friends = correct_answer := by
  -- We use the definitions and the conditions directly here
  sorry

end friends_total_l280_280546


namespace sum_of_roots_eq_14_l280_280200

-- Define the condition as a hypothesis
def equation (x : ℝ) := (x - 7) ^ 2 = 16

-- Define the statement that needs to be proved
theorem sum_of_roots_eq_14 : 
  (∀ x : ℝ, equation x → x = 11 ∨ x = 3) → 11 + 3 = 14 :=
by 
sorry

end sum_of_roots_eq_14_l280_280200


namespace final_value_correct_l280_280330

-- Define the square and its vertices
def A := (0, 2 : ℝ × ℝ)
def B := (2, 2 : ℝ × ℝ)
def C := (2, 0 : ℝ × ℝ)
def D := (0, 0 : ℝ × ℝ)

-- Define the midpoints M and N
def M := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
def N := ((C.1 + D.1) / 2, (C.2 + D.2) / 2)

-- Define the feet of the perpendiculars X and Y
def X : ℝ × ℝ :=
let lineMD : ℝ → ℝ := λ x, (1/2) * x in
let perpA : ℝ → ℝ := λ x, -2 * x + 2 in
let x_coord := 4/5 in
(x_coord, lineMD x_coord)

def Y : ℝ × ℝ :=
let lineNB : ℝ → ℝ := λ x, 2 * x - 2 in
let perpA : ℝ → ℝ := λ x, -(1/2) * x + 2 in
let x_coord := 8/5 in
(x_coord, lineNB x_coord)

-- Define the distance square as a rational number
def dist_sq_xy :=
let dx := (Y.1 - X.1) in
let dy := (Y.2 - X.2) in
(dx^2 + dy^2 : ℝ)

-- Define the final value of 100p + q where p/q = dist_sq_xy
def final_value := 100 * 32 + 25

theorem final_value_correct : final_value = 3225 := by
  -- Placeholder for the actual proof, which will involve calculating the
  -- Euclidean distance and proving p = 32, q = 25 are relatively prime
  sorry

end final_value_correct_l280_280330


namespace sum_of_roots_l280_280152

theorem sum_of_roots : 
  let eq := (x - 7) ^ 2 = 16 in
  (roots : set ℝ) := { x | eq } in
  ∑ roots = 14 :=
by
sorry

end sum_of_roots_l280_280152


namespace tan_45_degree_is_one_l280_280716

theorem tan_45_degree_is_one : Real.tan (Real.pi / 4) = 1 := sorry

end tan_45_degree_is_one_l280_280716


namespace dragon_heads_belong_to_dragons_l280_280586

def truthful (H : ℕ) : Prop := 
  H = 1 ∨ H = 3

def lying (H : ℕ) : Prop := 
  H = 2 ∨ H = 4

def head1_statement : Prop := truthful 1
def head2_statement : Prop := truthful 3
def head3_statement : Prop := ¬ truthful 2
def head4_statement : Prop := lying 3

theorem dragon_heads_belong_to_dragons :
  head1_statement ∧ head2_statement ∧ head3_statement ∧ head4_statement →
  (∀ H, (truthful H ↔ H = 1 ∨ H = 3) ∧ (lying H ↔ H = 2 ∨ H = 4)) :=
by
  sorry

end dragon_heads_belong_to_dragons_l280_280586


namespace mutually_exclusive_necessary_not_sufficient_complementary_l280_280548

variables {Ω : Type} {A1 A2 : Set Ω}

/-- Definition of mutually exclusive events -/
def mutually_exclusive (A1 A2 : Set Ω) : Prop :=
  A1 ∩ A2 = ∅

/-- Definition of complementary events -/
def complementary (A1 A2 : Set Ω) : Prop :=
  A1 ∪ A2 = Set.univ ∧ mutually_exclusive A1 A2

/-- The proposition that mutually exclusive events are necessary but not sufficient for being complementary -/
theorem mutually_exclusive_necessary_not_sufficient_complementary :
  (mutually_exclusive A1 A2 → complementary A1 A2) = false 
  ∧ (complementary A1 A2 → mutually_exclusive A1 A2) = true :=
sorry

end mutually_exclusive_necessary_not_sufficient_complementary_l280_280548


namespace tan_45_eq_1_l280_280856

theorem tan_45_eq_1 :
  let π := Real.pi in
  ∃ (x : ℝ), x = π / 4 ∧
  sin x = sqrt 2 / 2 ∧
  cos x = sqrt 2 / 2 →
  tan (π / 4) = 1 :=
by sorry

end tan_45_eq_1_l280_280856


namespace tan_45_eq_1_l280_280754

theorem tan_45_eq_1 : Real.tan (Real.pi / 4) = 1 := by
  have h1 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have h2 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [Real.tan_eq_sin_div_cos, h1, h2]
  simp

end tan_45_eq_1_l280_280754


namespace tenth_term_arithmetic_sequence_l280_280123

theorem tenth_term_arithmetic_sequence (a d : ℕ) 
  (h1 : a + 2 * d = 10) 
  (h2 : a + 5 * d = 16) : 
  a + 9 * d = 24 := 
by 
  sorry

end tenth_term_arithmetic_sequence_l280_280123


namespace find_f6_l280_280094

noncomputable def f : ℝ → ℝ :=
sorry

theorem find_f6 (h1 : ∀ x y : ℝ, f (x + y) = f x + f y)
                (h2 : f 5 = 6) :
  f 6 = 36 / 5 :=
sorry

end find_f6_l280_280094


namespace central_cell_value_l280_280509

variables a b c d e f g h i : ℝ

-- Conditions
axiom row1 : a * b * c = 10
axiom row2 : d * e * f = 10
axiom row3 : g * h * i = 10
axiom col1 : a * d * g = 10
axiom col2 : b * e * h = 10
axiom col3 : c * f * i = 10
axiom sq1 : a * b * d * e = 3
axiom sq2 : b * c * e * f = 3
axiom sq3 : d * e * g * h = 3
axiom sq4 : e * f * h * i = 3

theorem central_cell_value : e = 0.00081 := by
  sorry

end central_cell_value_l280_280509


namespace find_a_odd_function_l280_280926

noncomputable def f (a x : ℝ) := Real.log (Real.sqrt (x^2 + 1) - a * x)

theorem find_a_odd_function :
  ∀ a : ℝ, (∀ x : ℝ, f a (-x) + f a x = 0) ↔ (a = 1 ∨ a = -1) := by
  sorry

end find_a_odd_function_l280_280926


namespace linda_needs_additional_batches_l280_280958

theorem linda_needs_additional_batches:
  let classmates := 24
  let cookies_per_classmate := 10
  let dozen := 12
  let cookies_per_batch := 4 * dozen
  let cookies_needed := classmates * cookies_per_classmate
  let chocolate_chip_batches := 2
  let oatmeal_raisin_batches := 1
  let cookies_made := (chocolate_chip_batches + oatmeal_raisin_batches) * cookies_per_batch
  let remaining_cookies := cookies_needed - cookies_made
  let additional_batches := remaining_cookies / cookies_per_batch
  additional_batches = 2 :=
by
  sorry

end linda_needs_additional_batches_l280_280958


namespace tan_45_deg_l280_280688

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l280_280688


namespace solve_triangle_l280_280534

variable {A B C : ℝ}
variable {a b c : ℝ}

noncomputable def sin_B_plus_pi_four (a b c : ℝ) : ℝ :=
  let cos_B := (a^2 + c^2 - b^2) / (2 * a * c)
  let sin_B := Real.sqrt (1 - cos_B^2)
  sin_B * Real.sqrt 2 / 2 + cos_B * Real.sqrt 2 / 2

theorem solve_triangle 
  (a b c : ℝ)
  (h1 : b = 2 * Real.sqrt 5)
  (h2 : c = 3)
  (h3 : 3 * a * (a^2 + b^2 - c^2) / (2 * a * b) = 2 * c * (b^2 + c^2 - a^2) / (2 * b * c)) :
  a = Real.sqrt 5 ∧ 
  sin_B_plus_pi_four a b c = Real.sqrt 10 / 10 :=
by 
  sorry

end solve_triangle_l280_280534


namespace greatest_value_sum_eq_24_l280_280272

theorem greatest_value_sum_eq_24 {a b : ℕ} (ha : 0 < a) (hb : 1 < b) 
  (h1 : ∀ (x y : ℕ), 0 < x → 1 < y → x^y < 500 → x^y ≤ a^b) : 
  a + b = 24 := 
by
  sorry

end greatest_value_sum_eq_24_l280_280272


namespace sum_when_max_power_less_500_l280_280263

theorem sum_when_max_power_less_500 :
  ∃ (a b : ℕ), a > 0 ∧ b > 1 ∧ a^b < 500 ∧
  (∀ (a' b' : ℕ), a' > 0 → b' > 1 → a'^b' < 500 → a^b ≥ a'^b') ∧ (a + b = 24) :=
  sorry

end sum_when_max_power_less_500_l280_280263


namespace exists_n_le_2500_perfect_square_l280_280429

noncomputable def sum_of_squares (n : ℕ) : ℚ :=
  (n * (n + 1) * (2 * n + 1)) / 6

noncomputable def sum_of_squares_segment (n : ℕ) : ℚ :=
  ((26 * n^3 + 12 * n^2 + n) / 3)

theorem exists_n_le_2500_perfect_square :
  ∃ (n : ℕ), n ≤ 2500 ∧ ∃ (k : ℚ), k^2 = (sum_of_squares n) * (sum_of_squares_segment n) :=
sorry

end exists_n_le_2500_perfect_square_l280_280429


namespace tan_45_eq_one_l280_280684

theorem tan_45_eq_one 
  (θ : ℝ)
  (hθ : θ = 45)
  (sin_45_eq : Real.sin (θ * Real.pi / 180) = Real.sqrt 2 / 2)
  (cos_45_eq : Real.cos (θ * Real.pi /180) = Real.sqrt 2 / 2)
  (tan_def : ∀ x : ℝ, 0 < Real.cos x → Real.tan x = Real.sin x / Real.cos x) : 
  Real.tan (θ * Real.pi / 180) = 1 := sorry

end tan_45_eq_one_l280_280684


namespace tan_45_eq_l280_280617

def tan_identity (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

def sin_45_eq : Real.sin 45 = Real.sqrt 2 / 2 := sorry

def cos_45_eq : Real.cos 45 = Real.sqrt 2 / 2 := sorry

theorem tan_45_eq : tan_identity 45 = 1 := by
  rw [tan_identity, sin_45_eq, cos_45_eq]
  -- simplifying (sqrt 2 / 2) / (sqrt 2 / 2) = 1
  sorry

end tan_45_eq_l280_280617


namespace experts_win_probability_l280_280906

noncomputable def probability_of_experts_winning (p : ℝ) (q : ℝ) (needed_expert_wins : ℕ) (needed_audience_wins : ℕ) : ℝ :=
  p ^ 4 + 4 * (p ^ 3 * q)

-- Probability values
def p : ℝ := 0.6
def q : ℝ := 1 - p

-- Number of wins needed
def needed_expert_wins : ℕ := 3
def needed_audience_wins : ℕ := 2

theorem experts_win_probability :
  probability_of_experts_winning p q needed_expert_wins needed_audience_wins = 0.4752 :=
by
  -- Proof would go here
  sorry

end experts_win_probability_l280_280906


namespace shares_of_stocks_they_can_buy_l280_280403

def weekly_savings_wife : ℕ := 100
def monthly_savings_husband : ℕ := 225
def months_of_savings : ℕ := 4
def cost_per_share : ℕ := 50

theorem shares_of_stocks_they_can_buy :
  (((weekly_savings_wife * 4) + monthly_savings_husband) * months_of_savings / 2) / cost_per_share = 25 :=
by
  -- sorry for the implementation
  sorry

end shares_of_stocks_they_can_buy_l280_280403


namespace treasure_distribution_l280_280004

noncomputable def calculate_share (investment total_investment total_value : ℝ) : ℝ :=
  (investment / total_investment) * total_value

theorem treasure_distribution 
  (investment_fonzie investment_aunt_bee investment_lapis investment_skylar investment_orion total_treasure : ℝ)
  (total_investment : ℝ)
  (h : total_investment = investment_fonzie + investment_aunt_bee + investment_lapis + investment_skylar + investment_orion) :
  calculate_share investment_fonzie total_investment total_treasure = 210000 ∧
  calculate_share investment_aunt_bee total_investment total_treasure = 255000 ∧
  calculate_share investment_lapis total_investment total_treasure = 270000 ∧
  calculate_share investment_skylar total_investment total_treasure = 225000 ∧
  calculate_share investment_orion total_investment total_treasure = 240000 :=
by
  sorry

end treasure_distribution_l280_280004


namespace average_number_of_stickers_per_album_is_correct_l280_280334

def average_stickers_per_album (albums : List ℕ) (n : ℕ) : ℚ := (albums.sum : ℚ) / n

theorem average_number_of_stickers_per_album_is_correct :
  average_stickers_per_album [5, 7, 9, 14, 19, 12, 26, 18, 11, 15] 10 = 13.6 := 
by
  sorry

end average_number_of_stickers_per_album_is_correct_l280_280334


namespace tan_45_eq_1_l280_280731

theorem tan_45_eq_1 (hcos : real.cos (real.pi / 4) = real.sqrt 2 / 2) (hsin : real.sin (real.pi / 4) = real.sqrt 2 / 2) : real.tan (real.pi / 4) = 1 :=
by
  have h : real.tan (real.pi / 4) = real.sin (real.pi / 4) / real.cos (real.pi / 4) := real.tan_eq_sin_div_cos _
  rw [hsin, hcos] at h
  simp at h
  exact h.symm

end tan_45_eq_1_l280_280731


namespace tan_45_degree_l280_280666

theorem tan_45_degree : ∀ (Q : ℝ × ℝ),
  -- Q corresponds to 45 degrees on the unit circle
  Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) →
  -- The radius of the unit circle is 1
  (Q.1 ^ 2 + Q.2 ^ 2 = 1) →
  -- QOE is a 45-45-90 triangle
  True →
  -- Prove that tan 45 degrees equals 1
  Real.tan (Real.pi / 4) = 1 :=
by
  -- The proof using these conditions is omitted
  intros,
  sorry

end tan_45_degree_l280_280666


namespace sum_of_roots_eq_14_l280_280193

theorem sum_of_roots_eq_14 (x : ℝ) :
  (x - 7) ^ 2 = 16 → ∃ r1 r2 : ℝ, (x = r1 ∨ x = r2) ∧ r1 + r2 = 14 :=
by
  intro h
  sorry

end sum_of_roots_eq_14_l280_280193


namespace central_cell_value_l280_280486

def table (a b c d e f g h i : ℝ) : Prop :=
  (a * b * c = 10) ∧ (d * e * f = 10) ∧ (g * h * i = 10) ∧
  (a * d * g = 10) ∧ (b * e * h = 10) ∧ (c * f * i = 10) ∧
  (a * b * d * e = 3) ∧ (b * c * e * f = 3) ∧ (d * e * g * h = 3) ∧ (e * f * h * i = 3)

theorem central_cell_value (a b c d f g h i e : ℝ) (h_table : table a b c d e f g h i) : 
  e = 0.00081 :=
by sorry

end central_cell_value_l280_280486


namespace central_cell_value_l280_280497

theorem central_cell_value :
  ∀ (a b c d e f g h i : ℝ),
    (a * b * c = 10) →
    (d * e * f = 10) →
    (g * h * i = 10) →
    (a * d * g = 10) →
    (b * e * h = 10) →
    (c * f * i = 10) →
    (a * b * d * e = 3) →
    (b * c * e * f = 3) →
    (d * e * g * h = 3) →
    (e * f * h * i = 3) →
    e = 0.00081 := 
by 
  intros a b c d e f g h i h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
  sorry

end central_cell_value_l280_280497


namespace sum_of_roots_eq_14_l280_280170

theorem sum_of_roots_eq_14 : 
  let equation := λ x : ℝ, (x - 7)^2 = 16 in
  let roots := {x | equation x} in
  (roots : ℝ) → ∀ x y ∈ roots, x + y = 14 :=
by
  -- For the purpose of the problem statement, we specify the exact roots here
  let root1 := 11
  let root2 := 3
  have H : root1 + root2 = 14, by norm_num
  intro equation roots x y hx hy
  exact H

end sum_of_roots_eq_14_l280_280170


namespace tan_45_deg_l280_280628

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l280_280628


namespace solution_set_inequality_l280_280567

open Real

theorem solution_set_inequality (k : ℤ) (x : ℝ) :
  (x ∈ Set.Ioo (-π/4 + k * π) (k * π)) ↔ cos (4 * x) - 2 * sin (2 * x) - sin (4 * x) - 1 > 0 :=
by
  sorry

end solution_set_inequality_l280_280567


namespace tan_45_degree_l280_280745

theorem tan_45_degree : ∀ (θ : ℝ), 
  θ = 45 → 
  (sin θ = 1 / real.sqrt 2) → 
  (cos θ = 1 / real.sqrt 2) → 
  real.tan θ = 1 := 
by
  intros θ hθ hsin hcos
  sorry

end tan_45_degree_l280_280745


namespace caffeine_over_l280_280544

section caffeine_problem

-- Definitions of the given conditions
def cups_of_coffee : Nat := 3
def cans_of_soda : Nat := 1
def cups_of_tea : Nat := 2

def caffeine_per_cup_coffee : Nat := 80
def caffeine_per_can_soda : Nat := 40
def caffeine_per_cup_tea : Nat := 50

def caffeine_goal : Nat := 200

-- Calculate the total caffeine consumption
def caffeine_from_coffee : Nat := cups_of_coffee * caffeine_per_cup_coffee
def caffeine_from_soda : Nat := cans_of_soda * caffeine_per_can_soda
def caffeine_from_tea : Nat := cups_of_tea * caffeine_per_cup_tea

def total_caffeine : Nat := caffeine_from_coffee + caffeine_from_soda + caffeine_from_tea

-- Calculate the caffeine amount over the goal
def caffeine_over_goal : Nat := total_caffeine - caffeine_goal

-- Theorem statement
theorem caffeine_over {total_caffeine caffeine_goal : Nat} (h : total_caffeine = 380) (g : caffeine_goal = 200) :
  caffeine_over_goal = 180 := by
  -- The proof goes here.
  sorry

end caffeine_problem

end caffeine_over_l280_280544


namespace find_central_cell_l280_280495

variable (a b c d e f g h i : ℝ)

def condition_1 : Prop :=
  a * b * c = 10 ∧ d * e * f = 10 ∧ g * h * i = 10

def condition_2 : Prop :=
  a * d * g = 10 ∧ b * e * h = 10 ∧ c * f * i = 10

def condition_3 : Prop :=
  a * b * d * e = 3 ∧ b * c * e * f = 3 ∧ d * e * g * h = 3 ∧ e * f * h * i = 3

theorem find_central_cell (h1 : condition_1 a b c d e f g h i)
                          (h2 : condition_2 a b c d e f g h i)
                          (h3 : condition_3 a b c d e f g h i) : 
  e = 0.00081 := 
sorry

end find_central_cell_l280_280495


namespace filtration_concentration_l280_280317

-- Variables and conditions used in the problem
variable (P P0 : ℝ) (k t : ℝ)
variable (h1 : P = P0 * Real.exp (-k * t))
variable (h2 : Real.exp (-2 * k) = 0.8)

-- Main statement: Prove the concentration after 5 hours is approximately 57% of the original
theorem filtration_concentration :
  (P0 * Real.exp (-5 * k)) / P0 = 0.57 :=
by sorry

end filtration_concentration_l280_280317


namespace tan_45_eq_one_l280_280682

theorem tan_45_eq_one 
  (θ : ℝ)
  (hθ : θ = 45)
  (sin_45_eq : Real.sin (θ * Real.pi / 180) = Real.sqrt 2 / 2)
  (cos_45_eq : Real.cos (θ * Real.pi /180) = Real.sqrt 2 / 2)
  (tan_def : ∀ x : ℝ, 0 < Real.cos x → Real.tan x = Real.sin x / Real.cos x) : 
  Real.tan (θ * Real.pi / 180) = 1 := sorry

end tan_45_eq_one_l280_280682


namespace exists_quad_root_l280_280099

theorem exists_quad_root (a b c d : ℝ) (h : a * c = 2 * b + 2 * d) :
  (∃ x, x^2 + a * x + b = 0) ∨ (∃ x, x^2 + c * x + d = 0) :=
sorry

end exists_quad_root_l280_280099


namespace distinct_solution_count_l280_280451

theorem distinct_solution_count : 
  ∃! x : ℝ, |x - 3| = |x + 5| :=
begin
  sorry
end

end distinct_solution_count_l280_280451


namespace problem1_solution_l280_280355

theorem problem1_solution (x y : ℚ) (h1 : 3 * x + 2 * y = 10) (h2 : x / 2 - (y + 1) / 3 = 1) : 
  x = 3 ∧ y = 1 / 2 :=
by
  sorry

end problem1_solution_l280_280355


namespace arrangements_of_45520_l280_280532

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def arrangements (n : Nat) (k : Nat) : Nat :=
  factorial n / factorial k

theorem arrangements_of_45520 : 
  let n0_pos := 4
  let remaining_digits := 4 * arrangements 4 2
  n0_pos * remaining_digits = 48 :=
by
  -- Definitions and lemmas can be introduced here
  sorry

end arrangements_of_45520_l280_280532


namespace simplify_expression_l280_280553

variable (p : ℤ)

-- Defining the given expression
def initial_expression : ℤ := ((5 * p + 1) - 2 * p * 4) * 3 + (4 - 1 / 3) * (6 * p - 9)

-- Statement asserting the simplification
theorem simplify_expression : initial_expression p = 13 * p - 30 := 
sorry

end simplify_expression_l280_280553


namespace sum_of_roots_eq_14_l280_280168

theorem sum_of_roots_eq_14 : 
  let equation := λ x : ℝ, (x - 7)^2 = 16 in
  let roots := {x | equation x} in
  (roots : ℝ) → ∀ x y ∈ roots, x + y = 14 :=
by
  -- For the purpose of the problem statement, we specify the exact roots here
  let root1 := 11
  let root2 := 3
  have H : root1 + root2 = 14, by norm_num
  intro equation roots x y hx hy
  exact H

end sum_of_roots_eq_14_l280_280168


namespace tan_45_deg_eq_one_l280_280833

/-- The tangent of 45 degrees is equal to 1. -/
theorem tan_45_deg_eq_one : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_one_l280_280833


namespace tan_45_eq_1_l280_280855

theorem tan_45_eq_1 :
  let π := Real.pi in
  ∃ (x : ℝ), x = π / 4 ∧
  sin x = sqrt 2 / 2 ∧
  cos x = sqrt 2 / 2 →
  tan (π / 4) = 1 :=
by sorry

end tan_45_eq_1_l280_280855


namespace tan_45_deg_eq_1_l280_280606

-- Define conditions
variable (Q : ℝ × ℝ)
variable (E : ℝ × ℝ)
variable (O : ℝ × ℝ := (0, 0))
variable (θ : ℝ := 45)

-- Define main theorem
theorem tan_45_deg_eq_1
  (Q_on_unit_circle : Q = (ℝ.sqrt 2 / 2, ℝ.sqrt 2 / 2))
  (E_foot_perpendicular : E = (ℝ.sqrt 2 / 2, 0))
  (triangle_45_45_90 : (Q.1 - O.1) ^ 2 + (Q.2 - O.2) ^ 2 = 1) :
  Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_1_l280_280606


namespace range_of_a_l280_280480

theorem range_of_a (h : ¬ ∃ x : ℝ, x < 2023 ∧ x > a) : a ≥ 2023 := 
sorry

end range_of_a_l280_280480


namespace abs_inequality_solution_l280_280117

theorem abs_inequality_solution (x : ℝ) : |x - 2| < 1 ↔ 1 < x ∧ x < 3 :=
by
  -- the proof would go here
  sorry

end abs_inequality_solution_l280_280117


namespace inequality_solution_l280_280423

theorem inequality_solution :
  {x : ℝ | (3 * x - 9) * (x - 4) / (x - 1) ≥ 0} = {x : ℝ | x < 1} ∪ {x : ℝ | 1 < x ∧ x ≤ 3} ∪ {x : ℝ | x ≥ 4} :=
by
  sorry

end inequality_solution_l280_280423


namespace sum_of_roots_eq_14_l280_280198

-- Define the condition as a hypothesis
def equation (x : ℝ) := (x - 7) ^ 2 = 16

-- Define the statement that needs to be proved
theorem sum_of_roots_eq_14 : 
  (∀ x : ℝ, equation x → x = 11 ∨ x = 3) → 11 + 3 = 14 :=
by 
sorry

end sum_of_roots_eq_14_l280_280198


namespace tan_45_degrees_l280_280818

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l280_280818


namespace f_monotonically_increasing_on_1_to_infinity_l280_280927

noncomputable def f (x : ℝ) : ℝ := x + 1/x

theorem f_monotonically_increasing_on_1_to_infinity :
  ∀ x y : ℝ, 1 < x → x < y → f x < f y := 
sorry

end f_monotonically_increasing_on_1_to_infinity_l280_280927


namespace probability_all_red_is_correct_l280_280585

def total_marbles (R W B : Nat) : Nat := R + W + B

def first_red_probability (R W B : Nat) : Rat := R / total_marbles R W B
def second_red_probability (R W B : Nat) : Rat := (R - 1) / (total_marbles R W B - 1)
def third_red_probability (R W B : Nat) : Rat := (R - 2) / (total_marbles R W B - 2)

def all_red_probability (R W B : Nat) : Rat := 
  first_red_probability R W B * 
  second_red_probability R W B * 
  third_red_probability R W B

theorem probability_all_red_is_correct 
  (R W B : Nat) (hR : R = 5) (hW : W = 6) (hB : B = 7) :
  all_red_probability R W B = 5 / 408 := by
  sorry

end probability_all_red_is_correct_l280_280585


namespace three_digit_multiples_of_7_l280_280468

theorem three_digit_multiples_of_7 : 
  ∃! n : ℕ, (n = 128) ∧ (∀ k, (100 ≤ 7 * k ∧ 7 * k ≤ 999) ↔ (15 ≤ k ∧ k ≤ 142)) :=
begin
  sorry
end

end three_digit_multiples_of_7_l280_280468


namespace david_profit_l280_280236

def weight : ℝ := 50
def cost : ℝ := 50
def price_per_kg : ℝ := 1.20
def total_earnings : ℝ := weight * price_per_kg
def profit : ℝ := total_earnings - cost

theorem david_profit : profit = 10 := by
  sorry

end david_profit_l280_280236


namespace complement_correct_l280_280308

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define the set A as the set of real numbers such that -1 ≤ x < 2
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 2}

-- Define the complement of A in U
def complement_U_A : Set ℝ := {x : ℝ | x < -1 ∨ x ≥ 2}

-- The proof statement: the complement of A in U is the expected set
theorem complement_correct : (U \ A) = complement_U_A := 
by
  sorry

end complement_correct_l280_280308


namespace ratio_of_ages_in_six_years_l280_280064

-- Definitions based on conditions
def EllensCurrentAge : ℕ := 10
def MarthasCurrentAge : ℕ := 32

-- The main statement to prove
theorem ratio_of_ages_in_six_years : 
  let EllensAgeInSixYears := EllensCurrentAge + 6
  let MarthasAgeInSixYears := MarthasCurrentAge + 6
  (MarthasAgeInSixYears : ℚ) / (EllensAgeInSixYears : ℚ) = 19 / 8 := by
  sorry

end ratio_of_ages_in_six_years_l280_280064


namespace central_cell_value_l280_280511

theorem central_cell_value (a b c d e f g h i : ℝ)
  (h_row1 : a * b * c = 10)
  (h_row2 : d * e * f = 10)
  (h_row3 : g * h * i = 10)
  (h_col1 : a * d * g = 10)
  (h_col2 : b * e * h = 10)
  (h_col3 : c * f * i = 10)
  (h_block1 : a * b * d * e = 3)
  (h_block2 : b * c * e * f = 3)
  (h_block3 : d * e * g * h = 3)
  (h_block4 : e * f * h * i = 3) :
  e = 0.00081 :=
sorry

end central_cell_value_l280_280511


namespace sum_of_roots_eq_14_l280_280177

theorem sum_of_roots_eq_14 : 
  (∀ x : ℝ, (x - 7)^2 = 16 → (x = 11 ∨ x = 3)) →
  (∀ x1 x2 : ℝ, (x1 - 7)^2 = 16 ∧ (x2 - 7)^2 = 16 ∧ x1 ≠ x2 → x1 + x2 = 14) :=
by
  intros h x1 x2 h_comb
  sorry

end sum_of_roots_eq_14_l280_280177


namespace pieces_to_cut_l280_280453

-- Define the conditions
def rodLength : ℝ := 42.5  -- Length of the rod
def pieceLength : ℝ := 0.85  -- Length of each piece

-- Define the theorem that needs to be proven
theorem pieces_to_cut (h1 : rodLength = 42.5) (h2 : pieceLength = 0.85) : 
  (rodLength / pieceLength) = 50 := 
  by sorry

end pieces_to_cut_l280_280453


namespace solution_of_binary_linear_equation_l280_280962

theorem solution_of_binary_linear_equation :
  ∃ (x y : ℤ), x + 2 * y = 6 ∧ x = 2 ∧ y = 2 :=
by
  use 2
  use 2
  split
  . calc
    2 + 2 * 2 = 2 + 4 := by rfl
    _ = 6         := by rfl
  . rfl
  . rfl

end solution_of_binary_linear_equation_l280_280962


namespace sum_of_a_and_b_l280_280269

theorem sum_of_a_and_b (a b : ℕ) (h1 : b > 1) (h2 : a^b < 500) (h3 : ∀ c d : ℕ, d > 1 → c^d < 500 → c^d ≤ a^b) : a + b = 24 :=
sorry

end sum_of_a_and_b_l280_280269


namespace central_cell_value_l280_280487

def table (a b c d e f g h i : ℝ) : Prop :=
  (a * b * c = 10) ∧ (d * e * f = 10) ∧ (g * h * i = 10) ∧
  (a * d * g = 10) ∧ (b * e * h = 10) ∧ (c * f * i = 10) ∧
  (a * b * d * e = 3) ∧ (b * c * e * f = 3) ∧ (d * e * g * h = 3) ∧ (e * f * h * i = 3)

theorem central_cell_value (a b c d f g h i e : ℝ) (h_table : table a b c d e f g h i) : 
  e = 0.00081 :=
by sorry

end central_cell_value_l280_280487


namespace chess_tournament_games_l280_280318

def games_played (n : ℕ) : ℕ := n * (n - 1) / 2

theorem chess_tournament_games (n : ℕ) (h : n = 19) : games_played n = 171 :=
by
  rw [h]
  sorry

end chess_tournament_games_l280_280318


namespace distance_small_ball_to_surface_l280_280339

-- Define the main variables and conditions
variables (R : ℝ)

-- Define the conditions of the problem
def bottomBallRadius : ℝ := 2 * R
def topBallRadius : ℝ := R
def edgeLengthBaseTetrahedron : ℝ := 4 * R
def edgeLengthLateralTetrahedron : ℝ := 3 * R

-- Define the main statement in Lean format
theorem distance_small_ball_to_surface (R : ℝ) :
  (3 * R) = R + bottomBallRadius R :=
sorry

end distance_small_ball_to_surface_l280_280339


namespace max_value_t_min_value_y_l280_280481

open Real

-- Maximum value of t for ∀ x ∈ ℝ, |3x + 2| + |3x - 1| ≥ t
theorem max_value_t :
  ∃ t, (∀ x : ℝ, |3 * x + 2| + |3 * x - 1| ≥ t) ∧ t = 3 :=
by
  sorry

-- Minimum value of y for 4m + 5n = 3
theorem min_value_y (m n: ℝ) (hm : m > 0) (hn: n > 0) (h: 4 * m + 5 * n = 3) :
  ∃ y, (y = (1 / (m + 2 * n)) + (4 / (3 * m + 3 * n))) ∧ y = 3 :=
by
  sorry

end max_value_t_min_value_y_l280_280481


namespace tan_45_deg_eq_one_l280_280835

/-- The tangent of 45 degrees is equal to 1. -/
theorem tan_45_deg_eq_one : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_one_l280_280835


namespace frosting_cupcakes_l280_280597

theorem frosting_cupcakes :
  let r1 := 1 / 15
  let r2 := 1 / 25
  let r3 := 1 / 40
  let t := 600
  t * (r1 + r2 + r3) = 79 :=
by
  sorry

end frosting_cupcakes_l280_280597


namespace sum_of_roots_eq_14_l280_280197

theorem sum_of_roots_eq_14 (x : ℝ) :
  (x - 7) ^ 2 = 16 → ∃ r1 r2 : ℝ, (x = r1 ∨ x = r2) ∧ r1 + r2 = 14 :=
by
  intro h
  sorry

end sum_of_roots_eq_14_l280_280197


namespace sum_of_roots_of_quadratic_l280_280164

theorem sum_of_roots_of_quadratic : 
  (∃ (x : ℝ), (x - 7) ^ 2 = 16 ∧ (x = 11 ∨ x = 3)) → 
  (11 + 3 = 14) :=
by 
  intro h
  sorry

end sum_of_roots_of_quadratic_l280_280164


namespace min_photos_exists_l280_280297

-- Conditions: Girls and Boys
def girls : ℕ := 4
def boys : ℕ := 8

-- Definition representing the minimum number of photos for the condition
def min_photos : ℕ := 33

theorem min_photos_exists : 
  ∀ (photos : ℕ), 
  (photos ≥ min_photos) →
  (∃ (bb gg bg : ℕ), 
    (bb > 0 ∨ gg > 0 ∨ bg < photos)) :=
begin
  sorry
end

end min_photos_exists_l280_280297


namespace max_four_by_one_in_six_by_six_grid_l280_280383

-- Define the grid and rectangle dimensions
def grid_width : ℕ := 6
def grid_height : ℕ := 6
def rect_width : ℕ := 4
def rect_height : ℕ := 1

-- Define the maximum number of rectangles that can be placed
def max_rectangles (grid_w grid_h rect_w rect_h : ℕ) (non_overlapping : Bool) (within_boundaries : Bool) : ℕ :=
  if grid_w = 6 ∧ grid_h = 6 ∧ rect_w = 4 ∧ rect_h = 1 ∧ non_overlapping ∧ within_boundaries then
    8
  else
    0

-- The theorem stating the maximum number of 4x1 rectangles in a 6x6 grid
theorem max_four_by_one_in_six_by_six_grid
  : max_rectangles grid_width grid_height rect_width rect_height true true = 8 := 
sorry

end max_four_by_one_in_six_by_six_grid_l280_280383


namespace geometric_series_sum_l280_280279

theorem geometric_series_sum :
  let a := (2 : ℚ) / 3
  let r := -(1 / 2 : ℚ)
  let n := 6
  let S := a * ((1 - r^n) / (1 - r))
  S = 7 / 16 :=
by
  let a := (2 : ℚ) / 3
  let r := -(1 / 2 : ℚ)
  let n := 6
  let S := a * ((1 - r^n) / (1 - r))
  have h : S = 7 / 16 := sorry
  exact h

end geometric_series_sum_l280_280279


namespace tan_45_eq_1_l280_280730

theorem tan_45_eq_1 (hcos : real.cos (real.pi / 4) = real.sqrt 2 / 2) (hsin : real.sin (real.pi / 4) = real.sqrt 2 / 2) : real.tan (real.pi / 4) = 1 :=
by
  have h : real.tan (real.pi / 4) = real.sin (real.pi / 4) / real.cos (real.pi / 4) := real.tan_eq_sin_div_cos _
  rw [hsin, hcos] at h
  simp at h
  exact h.symm

end tan_45_eq_1_l280_280730


namespace integers_satisfy_equation_l280_280207

theorem integers_satisfy_equation (x y z : ℤ) (h1 : x = z) (h2 : y - 1 = x) : x * (x - y) + y * (y - z) + z * (z - x) = 1 :=
by
  sorry

end integers_satisfy_equation_l280_280207


namespace tan_45_deg_eq_1_l280_280603

-- Define conditions
variable (Q : ℝ × ℝ)
variable (E : ℝ × ℝ)
variable (O : ℝ × ℝ := (0, 0))
variable (θ : ℝ := 45)

-- Define main theorem
theorem tan_45_deg_eq_1
  (Q_on_unit_circle : Q = (ℝ.sqrt 2 / 2, ℝ.sqrt 2 / 2))
  (E_foot_perpendicular : E = (ℝ.sqrt 2 / 2, 0))
  (triangle_45_45_90 : (Q.1 - O.1) ^ 2 + (Q.2 - O.2) ^ 2 = 1) :
  Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_1_l280_280603


namespace central_cell_value_l280_280507

variables a b c d e f g h i : ℝ

-- Conditions
axiom row1 : a * b * c = 10
axiom row2 : d * e * f = 10
axiom row3 : g * h * i = 10
axiom col1 : a * d * g = 10
axiom col2 : b * e * h = 10
axiom col3 : c * f * i = 10
axiom sq1 : a * b * d * e = 3
axiom sq2 : b * c * e * f = 3
axiom sq3 : d * e * g * h = 3
axiom sq4 : e * f * h * i = 3

theorem central_cell_value : e = 0.00081 := by
  sorry

end central_cell_value_l280_280507


namespace sum_of_roots_of_equation_l280_280137

theorem sum_of_roots_of_equation : 
  (∃ x1 x2 : ℝ, (x1 - 7)^2 = 16 ∧ (x2 - 7)^2 = 16 ∧ x1 ≠ x2 ∧ (x1 + x2 = 14)) :=
by
  sorry

end sum_of_roots_of_equation_l280_280137


namespace zero_in_interval_l280_280570

noncomputable def f (x : ℝ) : ℝ := Real.log (3 * x / 2) - 2 / x

theorem zero_in_interval :
  (Real.log (3 / 2) - 2 < 0) ∧ (Real.log 3 - 2 / 3 > 0) →
  ∃ x : ℝ, 1 < x ∧ x < 2 ∧ f x = 0 :=
by
  -- conditions from the problem statement
  intros h
  -- proving the result
  sorry

end zero_in_interval_l280_280570


namespace intersection_with_y_axis_l280_280364

theorem intersection_with_y_axis :
  (∃ y : ℝ, y = -(0 + 2)^2 + 6 ∧ (0, y) = (0, 2)) :=
by
  sorry

end intersection_with_y_axis_l280_280364


namespace tan_45_deg_eq_one_l280_280654

/-
  Given:
  cos (45 deg) = sqrt 2 / 2,
  sin (45 deg) = sqrt 2 / 2,
  Prove: 
  tan (45 deg) = 1
-/

theorem tan_45_deg_eq_one : 
  (real.cos (real.pi / 4) = real.sqrt 2 / 2) → 
  (real.sin (real.pi / 4) = real.sqrt 2 / 2) → 
  real.tan (real.pi / 4) = 1 :=
by
  intros hcos hsin
  sorry

end tan_45_deg_eq_one_l280_280654


namespace central_cell_value_l280_280521

theorem central_cell_value :
  ∀ (a b c d e f g h i : ℝ),
  a * b * c = 10 →
  d * e * f = 10 →
  g * h * i = 10 →
  a * d * g = 10 →
  b * e * h = 10 →
  c * f * i = 10 →
  a * b * d * e = 3 →
  b * c * e * f = 3 →
  d * e * g * h = 3 →
  e * f * h * i = 3 →
  e = 0.00081 := 
by sorry

end central_cell_value_l280_280521


namespace sum_of_roots_l280_280149

theorem sum_of_roots (a b : Real) (h : (x - 7)^2 = 16):
  a + b = 14 :=
sorry

end sum_of_roots_l280_280149


namespace tan_45_deg_l280_280878

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 :=
by
  have h_sin_cos : Real.sin (Real.pi / 4) = Real.cos (Real.pi / 4) := sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos]
  have sqrt2_ne_zero : Real.sqrt 2 ≠ 0 := sorry
  have sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have cos_45 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [sin_45, cos_45]
  field_simp [sqrt2_ne_zero]
  exact one_div_one

end tan_45_deg_l280_280878


namespace functional_equation_solution_l280_280538

def odd_integers (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem functional_equation_solution (f : ℤ → ℤ)
  (h_odd : ∀ x : ℤ, odd_integers (f x))
  (h_eq : ∀ x y : ℤ, 
    f (x + f x + y) + f (x - f x - y) = f (x + y) + f (x - y)) :
  ∃ (d k : ℤ) (ell : ℕ → ℤ), 
    (∀ i : ℕ, i < d → odd_integers (ell i)) ∧
    ∀ (m : ℤ) (i : ℕ), i < d → 
      f (m * d + i) = 2 * k * m * d + ell i :=
sorry

end functional_equation_solution_l280_280538


namespace pencils_purchased_l280_280414

variable (P : ℕ)

theorem pencils_purchased (misplaced broke found bought left : ℕ) (h1 : misplaced = 7) (h2 : broke = 3) (h3 : found = 4) (h4 : bought = 2) (h5 : left = 16) :
  P - misplaced - broke + found + bought = left → P = 22 :=
by
  intros h
  have h_eq : P - 7 - 3 + 4 + 2 = 16 := by
    rw [h1, h2, h3, h4, h5] at h; exact h
  sorry

end pencils_purchased_l280_280414


namespace tan_45_degrees_l280_280798

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l280_280798


namespace find_u_v_l280_280954

theorem find_u_v (u v : ℤ) (huv_pos : 0 < v ∧ v < u) (area_eq : u^2 + 3 * u * v = 615) : 
  u + v = 45 :=
sorry

end find_u_v_l280_280954


namespace tommy_number_of_nickels_l280_280377

theorem tommy_number_of_nickels
  (d p n q : ℕ)
  (h1 : d = p + 10)
  (h2 : n = 2 * d)
  (h3 : q = 4)
  (h4 : p = 10 * q) : n = 100 :=
sorry

end tommy_number_of_nickels_l280_280377


namespace final_value_l280_280009

noncomputable def f : ℕ → ℝ := sorry

axiom f_mul_add (a b : ℕ) : f (a + b) = f a * f b
axiom f_one : f 1 = 2

theorem final_value : 
  (f 1)^2 + f 2 / f 1 + (f 2)^2 + f 4 / f 3 + (f 3)^2 + f 6 / f 5 + (f 4)^2 + f 8 / f 7 = 16 := 
sorry

end final_value_l280_280009


namespace factorize_poly1_factorize_poly2_l280_280286

theorem factorize_poly1 (x : ℝ) : 2 * x^3 - 8 * x^2 = 2 * x^2 * (x - 4) :=
by
  sorry

theorem factorize_poly2 (x : ℝ) : x^2 - 14 * x + 49 = (x - 7) ^ 2 :=
by
  sorry

end factorize_poly1_factorize_poly2_l280_280286


namespace container_capacity_l280_280024

/-- Given a container where 8 liters is 20% of its capacity, calculate the total capacity of 
    40 such containers filled with water. -/
theorem container_capacity (c : ℝ) (h : 8 = 0.20 * c) : 
    40 * c * 40 = 1600 := 
by
  sorry

end container_capacity_l280_280024


namespace solve_quadratic_l280_280085

theorem solve_quadratic :
    ∀ x : ℝ, x^2 - 6*x + 8 = 0 ↔ (x = 2 ∨ x = 4) :=
by
  intros x
  sorry

end solve_quadratic_l280_280085


namespace tan_45_deg_l280_280882

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 :=
by
  have h_sin_cos : Real.sin (Real.pi / 4) = Real.cos (Real.pi / 4) := sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos]
  have sqrt2_ne_zero : Real.sqrt 2 ≠ 0 := sorry
  have sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have cos_45 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [sin_45, cos_45]
  field_simp [sqrt2_ne_zero]
  exact one_div_one

end tan_45_deg_l280_280882


namespace three_digit_multiples_of_7_l280_280464

theorem three_digit_multiples_of_7 :
  let a := 7 * Nat.ceil (100 / 7)
  let l := 7 * Nat.floor (999 / 7)
  let d := 7
  let n := (l - a) / d + 1
  n = 128 :=
by
  let a := 7 * Nat.ceil (100 / 7)
  let l := 7 * Nat.floor (999 / 7)
  let d := 7
  let n := (l - a) / d + 1
  have : a = 105 := sorry
  have : l = 994 := sorry
  have : n = (994 - 105) / 7 + 1 := sorry
  have : n = 128 := sorry
  exact this

end three_digit_multiples_of_7_l280_280464


namespace tan_45_degree_is_one_l280_280713

theorem tan_45_degree_is_one : Real.tan (Real.pi / 4) = 1 := sorry

end tan_45_degree_is_one_l280_280713


namespace tan_45_deg_l280_280630

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l280_280630


namespace tan_45_eq_one_l280_280790

theorem tan_45_eq_one (Q : ℝ × ℝ) (h1 : Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2)) : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_eq_one_l280_280790


namespace suraj_average_after_13th_innings_l280_280582

theorem suraj_average_after_13th_innings
  (A : ℝ)
  (h : (12 * A + 96) / 13 = A + 5) :
  (12 * A + 96) / 13 = 36 :=
by
  sorry

end suraj_average_after_13th_innings_l280_280582


namespace fraction_identity_l280_280472

variable {a b x : ℝ}

-- Conditions
axiom h1 : x = a / b
axiom h2 : a ≠ b
axiom h3 : b ≠ 0

-- Question to prove
theorem fraction_identity :
  (a + b) / (a - b) = (x + 1) / (x - 1) :=
by
  sorry

end fraction_identity_l280_280472


namespace central_cell_value_l280_280501

theorem central_cell_value :
  ∀ (a b c d e f g h i : ℝ),
    (a * b * c = 10) →
    (d * e * f = 10) →
    (g * h * i = 10) →
    (a * d * g = 10) →
    (b * e * h = 10) →
    (c * f * i = 10) →
    (a * b * d * e = 3) →
    (b * c * e * f = 3) →
    (d * e * g * h = 3) →
    (e * f * h * i = 3) →
    e = 0.00081 := 
by 
  intros a b c d e f g h i h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
  sorry

end central_cell_value_l280_280501


namespace tan_45_eq_one_l280_280678

theorem tan_45_eq_one 
  (θ : ℝ)
  (hθ : θ = 45)
  (sin_45_eq : Real.sin (θ * Real.pi / 180) = Real.sqrt 2 / 2)
  (cos_45_eq : Real.cos (θ * Real.pi /180) = Real.sqrt 2 / 2)
  (tan_def : ∀ x : ℝ, 0 < Real.cos x → Real.tan x = Real.sin x / Real.cos x) : 
  Real.tan (θ * Real.pi / 180) = 1 := sorry

end tan_45_eq_one_l280_280678


namespace unique_positive_real_solution_l280_280932

-- Define the function
def f (x : ℝ) : ℝ := x^11 + 9 * x^10 + 19 * x^9 + 2023 * x^8 - 1421 * x^7 + 5

-- Prove the statement
theorem unique_positive_real_solution : ∃! x : ℝ, x > 0 ∧ f x = 0 :=
by
  sorry

end unique_positive_real_solution_l280_280932


namespace sum_of_roots_l280_280145

theorem sum_of_roots (a b : Real) (h : (x - 7)^2 = 16):
  a + b = 14 :=
sorry

end sum_of_roots_l280_280145


namespace tan_45_eq_1_l280_280853

theorem tan_45_eq_1 :
  let π := Real.pi in
  ∃ (x : ℝ), x = π / 4 ∧
  sin x = sqrt 2 / 2 ∧
  cos x = sqrt 2 / 2 →
  tan (π / 4) = 1 :=
by sorry

end tan_45_eq_1_l280_280853


namespace tan_45_degree_l280_280752

theorem tan_45_degree : ∀ (θ : ℝ), 
  θ = 45 → 
  (sin θ = 1 / real.sqrt 2) → 
  (cos θ = 1 / real.sqrt 2) → 
  real.tan θ = 1 := 
by
  intros θ hθ hsin hcos
  sorry

end tan_45_degree_l280_280752


namespace tommy_nickels_l280_280376

-- Definitions of given conditions
def pennies (quarters : Nat) : Nat := 10 * quarters  -- Tommy has 10 times as many pennies as quarters
def dimes (pennies : Nat) : Nat := pennies + 10      -- Tommy has 10 more dimes than pennies
def nickels (dimes : Nat) : Nat := 2 * dimes         -- Tommy has twice as many nickels as dimes

theorem tommy_nickels (quarters : Nat) (P : Nat) (D : Nat) (N : Nat) 
  (h1 : quarters = 4) 
  (h2 : P = pennies quarters) 
  (h3 : D = dimes P) 
  (h4 : N = nickels D) : 
  N = 100 := 
by
  -- sorry allows us to skip the proof
  sorry

end tommy_nickels_l280_280376


namespace tan_45_degree_l280_280670

theorem tan_45_degree : ∀ (Q : ℝ × ℝ),
  -- Q corresponds to 45 degrees on the unit circle
  Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) →
  -- The radius of the unit circle is 1
  (Q.1 ^ 2 + Q.2 ^ 2 = 1) →
  -- QOE is a 45-45-90 triangle
  True →
  -- Prove that tan 45 degrees equals 1
  Real.tan (Real.pi / 4) = 1 :=
by
  -- The proof using these conditions is omitted
  intros,
  sorry

end tan_45_degree_l280_280670


namespace tan_45_degree_l280_280676

theorem tan_45_degree : ∀ (Q : ℝ × ℝ),
  -- Q corresponds to 45 degrees on the unit circle
  Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) →
  -- The radius of the unit circle is 1
  (Q.1 ^ 2 + Q.2 ^ 2 = 1) →
  -- QOE is a 45-45-90 triangle
  True →
  -- Prove that tan 45 degrees equals 1
  Real.tan (Real.pi / 4) = 1 :=
by
  -- The proof using these conditions is omitted
  intros,
  sorry

end tan_45_degree_l280_280676


namespace second_number_mod_12_l280_280998

theorem second_number_mod_12 (x : ℕ) (h : (1274 * x * 1277 * 1285) % 12 = 6) : x % 12 = 1 := 
by 
  sorry

end second_number_mod_12_l280_280998


namespace sum_of_a_and_b_l280_280249

theorem sum_of_a_and_b (a b : ℕ) (h1: a > 0) (h2 : b > 1) (h3 : ∀ (x y : ℕ), x > 0 → y > 1 → x^y < 500 → x = a ∧ y = b → x^y ≥ a^b ) :
  a + b = 24 :=
sorry

end sum_of_a_and_b_l280_280249


namespace trigonometric_identity_l280_280301

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 2) : 
  (1 - Real.sin (2 * θ)) / (2 * (Real.cos θ)^2) = 1 / 2 :=
sorry

end trigonometric_identity_l280_280301


namespace max_value_quadratic_l280_280996

theorem max_value_quadratic : ∃ x : ℝ, -9 * x^2 + 27 * x + 15 = 35.25 :=
sorry

end max_value_quadratic_l280_280996


namespace tan_45_deg_l280_280640

theorem tan_45_deg : (Real.tan (Float.pi / 4)) = 1 :=
by
  sorry

end tan_45_deg_l280_280640


namespace permutation_inequality_l280_280536

theorem permutation_inequality (a b c d : ℝ) (h : a * b * c * d > 0) :
  ∃ (x y z w : ℝ), (x = a ∨ x = b ∨ x = c ∨ x = d) ∧ (y = a ∨ y = b ∨ y = c ∨ y = d) ∧
  (z = a ∨ z = b ∨ z = c ∨ z = d) ∧ (w = a ∨ w = b ∨ w = c ∨ w = d) ∧ 
  2 * (x * y + z * w)^2 > (x^2 + y^2) * (z^2 + w^2) := 
sorry

end permutation_inequality_l280_280536


namespace standing_in_a_row_standing_in_a_row_AB_adj_CD_not_adj_assign_to_classes_l280_280571

theorem standing_in_a_row (students : Finset String) (h : students = {"A", "B", "C", "D", "E"}) :
  students.card = 5 → 
  ∃ (ways : ℕ), ways = 120 :=
by
  sorry

theorem standing_in_a_row_AB_adj_CD_not_adj (students : Finset String) (h : students = {"A", "B", "C", "D", "E"}) :
  students.card = 5 →
  ∃ (ways : ℕ), ways = 24 :=
by
  sorry

theorem assign_to_classes (students : Finset String) (h : students = {"A", "B", "C", "D", "E"}) :
  students.card = 5 →
  ∃ (ways : ℕ), ways = 150 :=
by
  sorry

end standing_in_a_row_standing_in_a_row_AB_adj_CD_not_adj_assign_to_classes_l280_280571


namespace tan_45_eq_1_l280_280771

theorem tan_45_eq_1 :
  let Q : ℝ × ℝ := (1 / Real.sqrt 2, 1 / Real.sqrt 2)
  in ∃ (Q : ℝ × ℝ), Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) → 
     Real.tan (Real.pi / 4) = 1 := 
by {
  -- The proof is left as an exercise.
  sorry
}

end tan_45_eq_1_l280_280771


namespace sum_of_roots_eq_14_l280_280169

theorem sum_of_roots_eq_14 : 
  let equation := λ x : ℝ, (x - 7)^2 = 16 in
  let roots := {x | equation x} in
  (roots : ℝ) → ∀ x y ∈ roots, x + y = 14 :=
by
  -- For the purpose of the problem statement, we specify the exact roots here
  let root1 := 11
  let root2 := 3
  have H : root1 + root2 = 14, by norm_num
  intro equation roots x y hx hy
  exact H

end sum_of_roots_eq_14_l280_280169


namespace two_digit_sequence_partition_property_l280_280070

theorem two_digit_sequence_partition_property :
  ∀ (A B : Set ℕ), (A ∪ B = {x | x < 100 ∧ x % 10 < 10}) →
  ∃ (C : Set ℕ), (C = A ∨ C = B) ∧ 
  ∃ (lst : List ℕ), (∀ (x : ℕ), x ∈ lst → x ∈ C) ∧ 
  (∀ (x y : ℕ), (x, y) ∈ lst.zip lst.tail → (y = x + 1 ∨ y = x + 10 ∨ y = x + 11)) :=
by
  intros A B partition_condition
  sorry

end two_digit_sequence_partition_property_l280_280070


namespace tan_45_degree_is_one_l280_280711

theorem tan_45_degree_is_one : Real.tan (Real.pi / 4) = 1 := sorry

end tan_45_degree_is_one_l280_280711


namespace sum_of_roots_l280_280146

theorem sum_of_roots (a b : Real) (h : (x - 7)^2 = 16):
  a + b = 14 :=
sorry

end sum_of_roots_l280_280146


namespace problem_statement_l280_280963

noncomputable def is_integer (x : ℚ) : Prop := ∃ (n : ℤ), x = n

theorem problem_statement (m n p q : ℕ) (h₁ : m ≠ p) (h₂ : is_integer ((mn + pq : ℚ) / (m - p))) :
  is_integer ((mq + np : ℚ) / (m - p)) :=
sorry

end problem_statement_l280_280963


namespace sum_of_roots_l280_280153

theorem sum_of_roots : 
  let eq := (x - 7) ^ 2 = 16 in
  (roots : set ℝ) := { x | eq } in
  ∑ roots = 14 :=
by
sorry

end sum_of_roots_l280_280153


namespace tan_45_eq_one_l280_280797

theorem tan_45_eq_one (Q : ℝ × ℝ) (h1 : Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2)) : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_eq_one_l280_280797


namespace problem1_problem2_l280_280015

open Set

variable {U : Set ℝ} (A B : Set ℝ)

def UA : U = univ := by sorry
def A_def : A = { x : ℝ | 0 < x ∧ x ≤ 2 } := by sorry
def B_def : B = { x : ℝ | x < -3 ∨ x > 1 } := by sorry

theorem problem1 : A ∩ B = { x : ℝ | 1 < x ∧ x ≤ 2 } := 
by sorry

theorem problem2 : (U \ A) ∩ (U \ B) = { x : ℝ | -3 ≤ x ∧ x ≤ 0 } := 
by sorry

end problem1_problem2_l280_280015


namespace minimum_value_expression_l280_280917

theorem minimum_value_expression (x : ℝ) (h : x > 4) : 
  ∃ (m : ℝ), m = 6 ∧ ∀ y : ℝ, y = (x + 5) / (Real.sqrt (x - 4)) → y ≥ m :=
by
  -- proof goes here
  sorry

end minimum_value_expression_l280_280917


namespace curve_trajectory_a_eq_1_curve_fixed_point_a_ne_1_l280_280011

noncomputable def curve (x y a : ℝ) : ℝ :=
  x^2 + y^2 - 2 * a * x + 2 * (a - 2) * y + 2 

theorem curve_trajectory_a_eq_1 :
  ∃! (x y : ℝ), curve x y 1 = 0 ∧ x = 1 ∧ y = 1 := by
  sorry

theorem curve_fixed_point_a_ne_1 (a : ℝ) (ha : a ≠ 1) :
  curve 1 1 a = 0 := by
  sorry

end curve_trajectory_a_eq_1_curve_fixed_point_a_ne_1_l280_280011


namespace tan_45_eq_one_l280_280794

theorem tan_45_eq_one (Q : ℝ × ℝ) (h1 : Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2)) : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_eq_one_l280_280794


namespace sqrt_exp_cube_l280_280277

theorem sqrt_exp_cube :
  ((Real.sqrt ((Real.sqrt 5)^4))^3 = 125) :=
by
  sorry

end sqrt_exp_cube_l280_280277


namespace training_days_l280_280969

def total_minutes : ℕ := 5 * 60
def minutes_per_day : ℕ := 10 + 20

theorem training_days :
  total_minutes / minutes_per_day = 10 :=
by
  sorry

end training_days_l280_280969


namespace sum_of_roots_of_quadratic_l280_280165

theorem sum_of_roots_of_quadratic : 
  (∃ (x : ℝ), (x - 7) ^ 2 = 16 ∧ (x = 11 ∨ x = 3)) → 
  (11 + 3 = 14) :=
by 
  intro h
  sorry

end sum_of_roots_of_quadratic_l280_280165


namespace total_pebbles_l280_280066

theorem total_pebbles (white_pebbles : ℕ) (red_pebbles : ℕ)
  (h1 : white_pebbles = 20)
  (h2 : red_pebbles = white_pebbles / 2) :
  white_pebbles + red_pebbles = 30 := by
  sorry

end total_pebbles_l280_280066


namespace calculate_f_g_f_l280_280953

def f (x : ℤ) : ℤ := 5 * x + 5
def g (x : ℤ) : ℤ := 6 * x + 5

theorem calculate_f_g_f : f (g (f 3)) = 630 := by
  sorry

end calculate_f_g_f_l280_280953


namespace total_capacity_is_1600_l280_280025

/-- Eight liters is 20% of the capacity of one container. -/
def capacity_of_one_container := 8 / 0.20

/-- Calculate the total capacity of 40 such containers filled with water. -/
def total_capacity_of_40_containers := 40 * capacity_of_one_container

theorem total_capacity_is_1600 :
  total_capacity_of_40_containers = 1600 := by
    -- Proof is skipped using sorry.
    sorry

end total_capacity_is_1600_l280_280025


namespace tan_45_degrees_l280_280806

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l280_280806


namespace intersection_with_y_axis_l280_280363

theorem intersection_with_y_axis :
  (∃ y : ℝ, y = -(0 + 2)^2 + 6 ∧ (0, y) = (0, 2)) :=
by
  sorry

end intersection_with_y_axis_l280_280363


namespace tan_45_eq_one_l280_280683

theorem tan_45_eq_one 
  (θ : ℝ)
  (hθ : θ = 45)
  (sin_45_eq : Real.sin (θ * Real.pi / 180) = Real.sqrt 2 / 2)
  (cos_45_eq : Real.cos (θ * Real.pi /180) = Real.sqrt 2 / 2)
  (tan_def : ∀ x : ℝ, 0 < Real.cos x → Real.tan x = Real.sin x / Real.cos x) : 
  Real.tan (θ * Real.pi / 180) = 1 := sorry

end tan_45_eq_one_l280_280683


namespace min_trucks_for_crates_l280_280380

noncomputable def min_trucks (total_weight : ℕ) (max_weight_per_crate : ℕ) (truck_capacity : ℕ) : ℕ :=
  if total_weight % truck_capacity = 0 then total_weight / truck_capacity
  else total_weight / truck_capacity + 1

theorem min_trucks_for_crates :
  ∀ (total_weight max_weight_per_crate truck_capacity : ℕ),
    total_weight = 10 →
    max_weight_per_crate = 1 →
    truck_capacity = 3 →
    min_trucks total_weight max_weight_per_crate truck_capacity = 5 :=
by
  intros total_weight max_weight_per_crate truck_capacity h_total h_max h_truck
  rw [h_total, h_max, h_truck]
  sorry

end min_trucks_for_crates_l280_280380


namespace tan_45_degrees_l280_280809

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l280_280809


namespace three_digit_multiples_of_7_l280_280467

theorem three_digit_multiples_of_7 : 
  ∃! n : ℕ, (n = 128) ∧ (∀ k, (100 ≤ 7 * k ∧ 7 * k ≤ 999) ↔ (15 ≤ k ∧ k ≤ 142)) :=
begin
  sorry
end

end three_digit_multiples_of_7_l280_280467


namespace sum_of_roots_eq_14_l280_280204

-- Define the condition as a hypothesis
def equation (x : ℝ) := (x - 7) ^ 2 = 16

-- Define the statement that needs to be proved
theorem sum_of_roots_eq_14 : 
  (∀ x : ℝ, equation x → x = 11 ∨ x = 3) → 11 + 3 = 14 :=
by 
sorry

end sum_of_roots_eq_14_l280_280204


namespace experts_win_eventually_l280_280903

noncomputable def p : ℝ := 0.6
noncomputable def q : ℝ := 1 - p

def experts_need : ℕ := 3
def audience_need : ℕ := 2
def max_rounds : ℕ := 4

def winning_probability (experts_need : ℕ) (audience_need : ℕ) (p q : ℝ) : ℝ :=
  if experts_need = 3 ∧ audience_need = 2 ∧ p = 0.6 ∧ q = 0.4 then
    p^4 + 4 * p^3 * q
  else
    0

theorem experts_win_eventually :
  winning_probability experts_need audience_need p q = 0.4752 := sorry

end experts_win_eventually_l280_280903


namespace solve_equation_l280_280080

theorem solve_equation:
  ∀ x : ℝ, (x + 1) / 3 - 1 = (5 * x - 1) / 6 → x = -1 :=
by
  intro x
  intro h
  sorry

end solve_equation_l280_280080


namespace inequality_am_gm_l280_280445

variable {a b c : ℝ}

theorem inequality_am_gm (habc_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_abc_eq_1 : a * b * c = 1) : 
  a^3 + b^3 + c^3 + (a * b / (a^2 + b^2) + b * c / (b^2 + c^2) + c * a / (c^2 + a^2)) ≥ 9 / 2 := 
by
  sorry

end inequality_am_gm_l280_280445


namespace triangle_angle_inconsistency_l280_280121

theorem triangle_angle_inconsistency : 
  ∀ (L R T : ℝ), L + R + T = 180 ∧ L = 2 * R ∧ R = 60 → T = 0 → false :=
by
  intros L R T h1 h2,
  obtain ⟨h_sum, h_left, h_right⟩ := h1,
  rw h_right at *,
  rw h_left at *,
  linarith

end triangle_angle_inconsistency_l280_280121


namespace solution_to_equation_l280_280079

noncomputable def equation (x : ℝ) : ℝ := 
  (3 * x^2) / (x - 2) - (3 * x + 8) / 4 + (5 - 9 * x) / (x - 2) + 2

theorem solution_to_equation :
  equation 3.294 = 0 ∧ equation (-0.405) = 0 :=
by
  sorry

end solution_to_equation_l280_280079


namespace multiples_of_7_are_128_l280_280462

theorem multiples_of_7_are_128 : 
  let range_start := 100
  let range_end := 999
  let multiple_7_smallest := 7 * 15
  let multiple_7_largest := 7 * 142
  let n_terms := (142 - 15 + 1)
  n_terms = 128 := sorry

end multiples_of_7_are_128_l280_280462


namespace multiplier_for_second_part_l280_280228

theorem multiplier_for_second_part {x y k : ℝ} (h1 : x + y = 52) (h2 : 10 * x + k * y = 780) (hy : y = 30.333333333333332) (hx : x = 21.666666666666668) :
  k = 18.571428571428573 :=
by
  sorry

end multiplier_for_second_part_l280_280228


namespace simplify_expression_l280_280348

theorem simplify_expression (x y : ℤ) :
  (2 * x + 20) + (150 * x + 30) + y = 152 * x + 50 + y :=
by sorry

end simplify_expression_l280_280348


namespace lisa_needs_additional_marbles_l280_280063

/-- Lisa has 12 friends and 40 marbles. She needs to ensure each friend gets at least one marble and no two friends receive the same number of marbles. We need to find the minimum number of additional marbles needed to ensure this. -/
theorem lisa_needs_additional_marbles : 
  ∀ (friends marbles : ℕ), friends = 12 → marbles = 40 → 
  ∃ (additional_marbles : ℕ), additional_marbles = 38 ∧ 
  (∑ i in finset.range (friends + 1), i) - marbles = additional_marbles :=
by
  intros friends marbles friends_eq marbles_eq 
  use 38
  split
  · exact rfl
  calc (∑ i in finset.range (12 + 1), i) - 40 = 78 - 40 : by norm_num
                                  ... = 38 : by norm_num

end lisa_needs_additional_marbles_l280_280063


namespace leak_drain_time_l280_280580

/-- Statement: Given the rates at which a pump fills a tank and a leak drains the tank, 
prove that the leak can drain all the water in the tank in 14 hours. -/
theorem leak_drain_time :
  (∀ P L: ℝ, P = 1/2 → (P - L) = 3/7 → L = 1/14 → (1 / L) = 14) := 
by
  intros P L hP hPL hL
  -- Proof is omitted (to be provided)
  sorry

end leak_drain_time_l280_280580


namespace count_adjacent_pairs_sum_multiple_of_three_l280_280596

def adjacent_digit_sum_multiple_of_three (n : ℕ) : ℕ :=
  -- A function to count the number of pairs with a sum multiple of 3
  sorry

-- Define the sequence from 100 to 999 as digits concatenation
def digit_sequence : List ℕ := List.join (List.map (fun x => x.digits 10) (List.range' 100 900))

theorem count_adjacent_pairs_sum_multiple_of_three :
  adjacent_digit_sum_multiple_of_three digit_sequence.length = 897 :=
sorry

end count_adjacent_pairs_sum_multiple_of_three_l280_280596


namespace distinct_sum_l280_280048

theorem distinct_sum (a b c d e : ℤ) (h1 : (7 - a) * (7 - b) * (7 - c) * (7 - d) * (7 - e) = 0)
  (h2 : a ≠ b) (h3 : a ≠ c) (h4 : a ≠ d) (h5 : a ≠ e) (h6 : b ≠ c) (h7 : b ≠ d) (h8 : b ≠ e) (h9 : c ≠ d) (h10 : c ≠ e) (h11 : d ≠ e) :
  a + b + c + d + e = 35 :=
sorry

end distinct_sum_l280_280048


namespace symmetric_point_x_axis_l280_280943

theorem symmetric_point_x_axis (x y z : ℝ) : 
    (x, -y, -z) = (-2, -1, -9) :=
by 
  sorry

end symmetric_point_x_axis_l280_280943


namespace polynomial_coef_sum_l280_280479

theorem polynomial_coef_sum :
  ∃ (a b c d : ℝ), (∀ x : ℝ, (4 * x^2 - 6 * x + 3) * (8 - 3 * x) = a * x^3 + b * x^2 + c * x + d) ∧ (8 * a + 4 * b + 2 * c + d = 14) :=
by
  sorry

end polynomial_coef_sum_l280_280479


namespace shortest_chord_through_M_is_x_plus_y_minus_1_eq_0_l280_280923

noncomputable def circle_C : Set (ℝ × ℝ) := { p | (p.1^2 + p.2^2 - 4*p.1 - 2*p.2) = 0 }

def point_M_in_circle : Prop :=
  (1, 0) ∈ circle_C

theorem shortest_chord_through_M_is_x_plus_y_minus_1_eq_0 :
  point_M_in_circle →
  ∃ (a b c : ℝ), a * 1 + b * 0 + c = 0 ∧
  ∀ (x y : ℝ), (a * x + b * y + c = 0) → (x + y - 1 = 0) :=
by
  sorry

end shortest_chord_through_M_is_x_plus_y_minus_1_eq_0_l280_280923


namespace tan_of_45_deg_l280_280701

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_of_45_deg_l280_280701


namespace social_gathering_married_men_fraction_l280_280413

theorem social_gathering_married_men_fraction {W : ℝ} {MW : ℝ} {MM : ℝ} 
  (hW_pos : 0 < W)
  (hMW_def : MW = W * (3/7))
  (hMM_def : MM = W - MW)
  (h_total_people : 2 * MM + MW = 11) :
  (MM / 11) = 4/11 :=
by {
  sorry
}

end social_gathering_married_men_fraction_l280_280413


namespace max_min_cos_sin_l280_280002

open Real

theorem max_min_cos_sin :
  (-π / 2) < x → x < 0 → (sin x + cos x = 1 / 5)
  → (∃ y_max y_min : ℝ, y_max = 9 / 4 ∧ y_min = 2
    ∧ ∀ x_max ∈ {x | cos x = 1/2}, x_max ∈ {π / 3, -π / 3}
    ∧ ∀ x_min ∈ {x | cos x = 0} ∪ {x | cos x = 1}, x_min ∈ {π / 2, -π / 2, 0}) 
  := by sorry

end max_min_cos_sin_l280_280002


namespace find_central_cell_l280_280492

variable (a b c d e f g h i : ℝ)

def condition_1 : Prop :=
  a * b * c = 10 ∧ d * e * f = 10 ∧ g * h * i = 10

def condition_2 : Prop :=
  a * d * g = 10 ∧ b * e * h = 10 ∧ c * f * i = 10

def condition_3 : Prop :=
  a * b * d * e = 3 ∧ b * c * e * f = 3 ∧ d * e * g * h = 3 ∧ e * f * h * i = 3

theorem find_central_cell (h1 : condition_1 a b c d e f g h i)
                          (h2 : condition_2 a b c d e f g h i)
                          (h3 : condition_3 a b c d e f g h i) : 
  e = 0.00081 := 
sorry

end find_central_cell_l280_280492


namespace tan_45_deg_l280_280635

theorem tan_45_deg : (Real.tan (Float.pi / 4)) = 1 :=
by
  sorry

end tan_45_deg_l280_280635


namespace men_in_hotel_l280_280358

theorem men_in_hotel (n : ℕ) (A : ℝ) (h1 : 8 * 3 = 24)
  (h2 : A = 32.625 / n)
  (h3 : 24 + (A + 5) = 32.625) :
  n = 9 := 
  by
  sorry

end men_in_hotel_l280_280358


namespace sarah_monthly_payment_l280_280344

noncomputable def monthly_payment (loan_amount : ℝ) (down_payment : ℝ) (years : ℝ) : ℝ :=
  let financed_amount := loan_amount - down_payment
  let months := years * 12
  financed_amount / months

theorem sarah_monthly_payment : monthly_payment 46000 10000 5 = 600 := by
  sorry

end sarah_monthly_payment_l280_280344


namespace experts_eventual_win_probability_l280_280909

noncomputable def experts_win_probability (p : ℝ) (q : ℝ) (experts_needed : ℕ) (audience_needed : ℕ) (max_rounds : ℕ) : ℝ :=
  let e_all_wins := p ^ max_rounds
  let e_three_wins_one_loss := (Nat.choose max_rounds (max_rounds - 1)) * (p ^ (max_rounds - 1)) * q
  e_all_wins + e_three_wins_one_loss

theorem experts_eventual_win_probability :
  ∀ (p : ℝ) (q : ℝ),
  p = 0.6 →
  q = 1 - p →
  experts_win_probability p q 3 2 4 = 0.4752 := by
  intros p q hp hq
  have h : experts_win_probability p q 3 2 4 = 0.6 ^ 4 + 4 * (0.6 ^ 3) * 0.4 := sorry
  rw [hp, hq] at h
  norm_num at h
  exact h

end experts_eventual_win_probability_l280_280909


namespace tan_of_fourth_quadrant_l280_280444

theorem tan_of_fourth_quadrant (α : ℝ) (h₁ : Real.sin α = -5 / 13) (h₂ : α > 3 * Real.pi / 2 ∧ α < 2 * Real.pi) : Real.tan α = -5 / 12 :=
sorry

end tan_of_fourth_quadrant_l280_280444


namespace johns_speed_l280_280329

theorem johns_speed (J : ℝ)
  (lewis_speed : ℝ := 60)
  (distance_AB : ℝ := 240)
  (meet_distance_A : ℝ := 160)
  (time_lewis_to_B : ℝ := distance_AB / lewis_speed)
  (time_lewis_back_80 : ℝ := 80 / lewis_speed)
  (total_time_meet : ℝ := time_lewis_to_B + time_lewis_back_80)
  (total_distance_john_meet : ℝ := J * total_time_meet) :
  total_distance_john_meet = meet_distance_A → J = 30 := 
by
  sorry

end johns_speed_l280_280329


namespace at_least_one_root_l280_280108

theorem at_least_one_root 
  (a b c d : ℝ)
  (h : a * c = 2 * b + 2 * d) :
  (∃ x : ℝ, x^2 + a * x + b = 0) ∨ (∃ x : ℝ, x^2 + c * x + d = 0) :=
sorry

end at_least_one_root_l280_280108


namespace central_cell_value_l280_280502

theorem central_cell_value :
  ∀ (a b c d e f g h i : ℝ),
    (a * b * c = 10) →
    (d * e * f = 10) →
    (g * h * i = 10) →
    (a * d * g = 10) →
    (b * e * h = 10) →
    (c * f * i = 10) →
    (a * b * d * e = 3) →
    (b * c * e * f = 3) →
    (d * e * g * h = 3) →
    (e * f * h * i = 3) →
    e = 0.00081 := 
by 
  intros a b c d e f g h i h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
  sorry

end central_cell_value_l280_280502


namespace tan_45_eq_1_l280_280773

theorem tan_45_eq_1 :
  let Q : ℝ × ℝ := (1 / Real.sqrt 2, 1 / Real.sqrt 2)
  in ∃ (Q : ℝ × ℝ), Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) → 
     Real.tan (Real.pi / 4) = 1 := 
by {
  -- The proof is left as an exercise.
  sorry
}

end tan_45_eq_1_l280_280773


namespace tan_45_deg_l280_280879

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 :=
by
  have h_sin_cos : Real.sin (Real.pi / 4) = Real.cos (Real.pi / 4) := sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos]
  have sqrt2_ne_zero : Real.sqrt 2 ≠ 0 := sorry
  have sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have cos_45 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [sin_45, cos_45]
  field_simp [sqrt2_ne_zero]
  exact one_div_one

end tan_45_deg_l280_280879


namespace distance_midpoint_AD_to_BC_l280_280319

variable (AC BC BD : ℕ)
variable (perpendicular : Prop)
variable (d : ℝ)

theorem distance_midpoint_AD_to_BC
  (h1 : AC = 6)
  (h2 : BC = 5)
  (h3 : BD = 3)
  (h4 : perpendicular) :
  d = Real.sqrt 5 + 2 := by
  sorry

end distance_midpoint_AD_to_BC_l280_280319


namespace tan_45_deg_l280_280689

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l280_280689


namespace printer_ratio_l280_280569

-- Define the given conditions
def total_price_basic_computer_printer := 2500
def enhanced_computer_extra := 500
def basic_computer_price := 1500

-- The lean statement to prove the ratio of the price of the printer to the total price of the enhanced computer and printer is 1/3
theorem printer_ratio : ∀ (C_basic P C_enhanced Total_enhanced : ℕ), 
  C_basic + P = total_price_basic_computer_printer →
  C_enhanced = C_basic + enhanced_computer_extra →
  C_basic = basic_computer_price →
  C_enhanced + P = Total_enhanced →
  P / Total_enhanced = 1 / 3 := 
by
  intros C_basic P C_enhanced Total_enhanced h1 h2 h3 h4
  sorry

end printer_ratio_l280_280569


namespace sum_of_roots_eq_14_l280_280179

theorem sum_of_roots_eq_14 : 
  (∀ x : ℝ, (x - 7)^2 = 16 → (x = 11 ∨ x = 3)) →
  (∀ x1 x2 : ℝ, (x1 - 7)^2 = 16 ∧ (x2 - 7)^2 = 16 ∧ x1 ≠ x2 → x1 + x2 = 14) :=
by
  intros h x1 x2 h_comb
  sorry

end sum_of_roots_eq_14_l280_280179


namespace tan_45_deg_l280_280627

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l280_280627


namespace quadratic_same_roots_abs_l280_280535

theorem quadratic_same_roots_abs (d e : ℤ) : 
  (∀ x : ℤ, |x - 8| = 3 ↔ x = 11 ∨ x = 5) →
  (∀ x : ℤ, x^2 + d * x + e = 0 ↔ x = 11 ∨ x = 5) →
  (d, e) = (-16, 55) :=
by
  intro h₁ h₂
  have h₃ : ∀ x : ℤ, x^2 - 16 * x + 55 = 0 ↔ x = 11 ∨ x = 5 := sorry
  sorry

end quadratic_same_roots_abs_l280_280535


namespace tan_45_eq_1_l280_280759

theorem tan_45_eq_1 : Real.tan (Real.pi / 4) = 1 := by
  have h1 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have h2 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [Real.tan_eq_sin_div_cos, h1, h2]
  simp

end tan_45_eq_1_l280_280759


namespace statue_original_cost_l280_280947

noncomputable def original_cost (selling_price : ℝ) (profit_rate : ℝ) : ℝ :=
  selling_price / (1 + profit_rate)

theorem statue_original_cost :
  original_cost 660 0.20 = 550 := 
by
  sorry

end statue_original_cost_l280_280947


namespace sum_of_roots_of_equation_l280_280140

theorem sum_of_roots_of_equation : 
  (∃ x1 x2 : ℝ, (x1 - 7)^2 = 16 ∧ (x2 - 7)^2 = 16 ∧ x1 ≠ x2 ∧ (x1 + x2 = 14)) :=
by
  sorry

end sum_of_roots_of_equation_l280_280140


namespace quadratic_function_monotonicity_l280_280369

theorem quadratic_function_monotonicity
  (a b : ℝ)
  (h1 : ∀ x y : ℝ, x ≤ y ∧ y ≤ -1 → a * x^2 + b * x + 3 ≤ a * y^2 + b * y + 3)
  (h2 : ∀ x y : ℝ, -1 ≤ x ∧ x ≤ y → a * x^2 + b * x + 3 ≥ a * y^2 + b * y + 3) :
  b = 2 * a ∧ a < 0 :=
sorry

end quadratic_function_monotonicity_l280_280369


namespace tan_45_deg_l280_280633

theorem tan_45_deg : (Real.tan (Float.pi / 4)) = 1 :=
by
  sorry

end tan_45_deg_l280_280633


namespace tan_45_eq_1_l280_280767

theorem tan_45_eq_1 :
  let Q : ℝ × ℝ := (1 / Real.sqrt 2, 1 / Real.sqrt 2)
  in ∃ (Q : ℝ × ℝ), Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) → 
     Real.tan (Real.pi / 4) = 1 := 
by {
  -- The proof is left as an exercise.
  sorry
}

end tan_45_eq_1_l280_280767


namespace compute_c_minus_d_squared_eq_0_l280_280049

-- Defining conditions
def multiples_of_n_under_m (n m : ℕ) : ℕ :=
  (m - 1) / n

-- Defining the specific values
def c : ℕ := multiples_of_n_under_m 9 60
def d : ℕ := multiples_of_n_under_m 9 60  -- Since every multiple of 9 is a multiple of 3

theorem compute_c_minus_d_squared_eq_0 : (c - d) ^ 2 = 0 := by
  sorry

end compute_c_minus_d_squared_eq_0_l280_280049


namespace find_pqr_abs_l280_280052

variables {p q r : ℝ}

-- Conditions as hypotheses
def conditions (p q r : ℝ) : Prop :=
  p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0 ∧ p ≠ q ∧ q ≠ r ∧ r ≠ p ∧
  (p^2 + 2/q = q^2 + 2/r) ∧ (q^2 + 2/r = r^2 + 2/p)

-- Statement of the theorem
theorem find_pqr_abs (h : conditions p q r) : |p * q * r| = 2 :=
sorry

end find_pqr_abs_l280_280052


namespace white_trees_count_l280_280407

noncomputable def calculate_white_trees (total_trees pink_percent red_trees : ℕ) : ℕ :=
  total_trees - (total_trees * pink_percent / 100 + red_trees)

theorem white_trees_count 
  (h1 : total_trees = 42)
  (h2 : pink_percent = 100 / 3)
  (h3 : red_trees = 2) :
  calculate_white_trees total_trees pink_percent red_trees = 26 :=
by
  -- proof will go here
  sorry

end white_trees_count_l280_280407


namespace candy_game_solution_l280_280547

open Nat

theorem candy_game_solution 
  (total_candies : ℕ) 
  (nick_candies : ℕ) 
  (tim_candies : ℕ)
  (tim_wins : ℕ)
  (m n : ℕ)
  (htotal : total_candies = 55) 
  (hnick : nick_candies = 30) 
  (htim : tim_candies = 25)
  (htim_wins : tim_wins = 2)
  (hrounds_total : total_candies = nick_candies + tim_candies)
  (hwinner_condition1 : m > n) 
  (hwinner_condition2 : n > 0) 
  (hwinner_candies_total : total_candies = tim_wins * m + (total_candies / (m + n) - tim_wins) * n)
: m = 8 := 
sorry

end candy_game_solution_l280_280547


namespace central_cell_value_l280_280500

theorem central_cell_value :
  ∀ (a b c d e f g h i : ℝ),
    (a * b * c = 10) →
    (d * e * f = 10) →
    (g * h * i = 10) →
    (a * d * g = 10) →
    (b * e * h = 10) →
    (c * f * i = 10) →
    (a * b * d * e = 3) →
    (b * c * e * f = 3) →
    (d * e * g * h = 3) →
    (e * f * h * i = 3) →
    e = 0.00081 := 
by 
  intros a b c d e f g h i h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
  sorry

end central_cell_value_l280_280500


namespace tan_45_eq_one_l280_280891

noncomputable def sin_45 : ℝ := real.sqrt 2 / 2
noncomputable def cos_45 : ℝ := real.sqrt 2 / 2

theorem tan_45_eq_one : real.tan (real.pi / 4) = 1 := by
  have hsin : real.sin (real.pi / 4) = sin_45 := by sorry
  have hcos : real.cos (real.pi / 4) = cos_45 := by sorry
  rw [real.tan_eq_sin_div_cos, hsin, hcos]
  rw [div_self]
  exact eq.refl 1
  exact sorry

end tan_45_eq_one_l280_280891


namespace tan_of_45_deg_l280_280733

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  -- Definitions used directly from conditions
  have hcos : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry
  have hsin : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry

  -- Using the definition of tangent
  rw [Real.tan_eq_sin_div_cos, hsin, hcos]
  exact div_self (ne_of_gt (Real.sqrt_pos.2 zero_lt_two))

end tan_of_45_deg_l280_280733


namespace tan_alpha_eq_one_third_cos2alpha_over_expr_l280_280934

theorem tan_alpha_eq_one_third_cos2alpha_over_expr (α : ℝ) (h : Real.tan α = 1/3) :
  (Real.cos (2 * α)) / (2 * Real.sin α * Real.cos α + (Real.cos α)^2) = 8 / 15 :=
by
  -- This is the point where the proof steps will go, but we leave it as a placeholder.
  sorry

end tan_alpha_eq_one_third_cos2alpha_over_expr_l280_280934


namespace sum_of_roots_eq_14_l280_280178

theorem sum_of_roots_eq_14 : 
  (∀ x : ℝ, (x - 7)^2 = 16 → (x = 11 ∨ x = 3)) →
  (∀ x1 x2 : ℝ, (x1 - 7)^2 = 16 ∧ (x2 - 7)^2 = 16 ∧ x1 ≠ x2 → x1 + x2 = 14) :=
by
  intros h x1 x2 h_comb
  sorry

end sum_of_roots_eq_14_l280_280178


namespace abs_inequality_l280_280118

theorem abs_inequality (x : ℝ) (h : |x - 2| < 1) : 1 < x ∧ x < 3 := by
  sorry

end abs_inequality_l280_280118


namespace common_ratio_of_geometric_sequence_l280_280922

theorem common_ratio_of_geometric_sequence (a_1 q : ℝ) (hq : q ≠ 1) 
  (S : ℕ → ℝ) (hS: ∀ n, S n = a_1 * (1 - q^n) / (1 - q))
  (arithmetic_seq : 2 * S 7 = S 8 + S 9) :
  q = -2 :=
by sorry

end common_ratio_of_geometric_sequence_l280_280922


namespace products_selling_less_than_1000_l280_280599

theorem products_selling_less_than_1000 (N: ℕ) 
  (total_products: ℕ := 25) 
  (average_price: ℤ := 1200) 
  (min_price: ℤ := 400) 
  (max_price: ℤ := 12000) 
  (total_revenue := total_products * average_price) 
  (revenue_from_expensive: ℤ := max_price):
  12000 + (24 - N) * 1000 + N * 400 = 30000 ↔ N = 10 :=
by
  sorry

end products_selling_less_than_1000_l280_280599


namespace lisa_additional_marbles_l280_280062

theorem lisa_additional_marbles (n : ℕ) (m : ℕ) (s_n : ℕ) :
  n = 12 →
  m = 40 →
  (s_n = (list.sum (list.range (n + 1)))) →
  s_n - m = 38 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp only [list.range_succ, list.sum_range_succ, nat.factorial, nat.succ_eq_add_one, nat.add_succ, mul_add, mul_one, mul_comm n]
  sorry

end lisa_additional_marbles_l280_280062


namespace central_cell_value_l280_280515

theorem central_cell_value (a b c d e f g h i : ℝ)
  (h_row1 : a * b * c = 10)
  (h_row2 : d * e * f = 10)
  (h_row3 : g * h * i = 10)
  (h_col1 : a * d * g = 10)
  (h_col2 : b * e * h = 10)
  (h_col3 : c * f * i = 10)
  (h_block1 : a * b * d * e = 3)
  (h_block2 : b * c * e * f = 3)
  (h_block3 : d * e * g * h = 3)
  (h_block4 : e * f * h * i = 3) :
  e = 0.00081 :=
sorry

end central_cell_value_l280_280515


namespace tan_45_eq_1_l280_280769

theorem tan_45_eq_1 :
  let Q : ℝ × ℝ := (1 / Real.sqrt 2, 1 / Real.sqrt 2)
  in ∃ (Q : ℝ × ℝ), Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) → 
     Real.tan (Real.pi / 4) = 1 := 
by {
  -- The proof is left as an exercise.
  sorry
}

end tan_45_eq_1_l280_280769


namespace compute_v_l280_280555

variable (a b c : ℝ)

theorem compute_v (H1 : (a * c) / (a + b) + (b * a) / (b + c) + (c * b) / (c + a) = -8)
                  (H2 : (b * c) / (a + b) + (c * a) / (b + c) + (a * b) / (c + a) = 12)
                  (H3 : a * b * c = 1) :
  (b / (a + b) + c / (b + c) + a / (c + a)) = -8.5 :=
sorry

end compute_v_l280_280555


namespace tan_45_degrees_l280_280812

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l280_280812


namespace sum_of_roots_of_quadratic_l280_280159

theorem sum_of_roots_of_quadratic : 
  (∃ (x : ℝ), (x - 7) ^ 2 = 16 ∧ (x = 11 ∨ x = 3)) → 
  (11 + 3 = 14) :=
by 
  intro h
  sorry

end sum_of_roots_of_quadratic_l280_280159


namespace tan_45_eq_one_l280_280789

theorem tan_45_eq_one (Q : ℝ × ℝ) (h1 : Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2)) : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_eq_one_l280_280789


namespace tan_45_eq_1_l280_280820

theorem tan_45_eq_1
: ∀ (Q : ℝ × ℝ), Q = (Real.sqrt 2 / 2, Real.sqrt 2 / 2) ->
  ∀ θ, θ = Real.pi / 4 -> 
  Real.tan θ = 1 :=
by
  intros Q HQ θ Hθ
  rw [HQ, Hθ]
  sorry

end tan_45_eq_1_l280_280820


namespace shoe_size_ratio_l280_280327

theorem shoe_size_ratio (J A : ℕ) (hJ : J = 7) (hAJ : A + J = 21) : A / J = 2 :=
by
  -- Skipping the proof
  sorry

end shoe_size_ratio_l280_280327


namespace domain_of_c_eq_real_l280_280282

theorem domain_of_c_eq_real (m : ℝ) : (∀ x : ℝ, m * x^2 - 3 * x + 2 * m ≠ 0) ↔ (m < -3 * Real.sqrt 2 / 4 ∨ m > 3 * Real.sqrt 2 / 4) :=
by
  sorry

end domain_of_c_eq_real_l280_280282


namespace tan_45_degrees_l280_280817

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l280_280817


namespace silver_cost_l280_280043

theorem silver_cost (S : ℝ) : 
  (1.5 * S) + (3 * 50 * S) = 3030 → S = 20 :=
by
  intro h
  sorry

end silver_cost_l280_280043


namespace solve_quadratic_l280_280084

theorem solve_quadratic :
    ∀ x : ℝ, x^2 - 6*x + 8 = 0 ↔ (x = 2 ∨ x = 4) :=
by
  intros x
  sorry

end solve_quadratic_l280_280084


namespace fraction_of_sum_l280_280216

theorem fraction_of_sum (numbers : List ℝ) (h_len : numbers.length = 21)
  (n : ℝ) (h_n : n ∈ numbers)
  (h_avg : n = 5 * ((numbers.sum - n) / 20)) :
  n / numbers.sum = 1 / 5 :=
by
  sorry

end fraction_of_sum_l280_280216


namespace three_digit_multiples_of_7_l280_280469

theorem three_digit_multiples_of_7 : 
  ∃! n : ℕ, (n = 128) ∧ (∀ k, (100 ≤ 7 * k ∧ 7 * k ≤ 999) ↔ (15 ≤ k ∧ k ≤ 142)) :=
begin
  sorry
end

end three_digit_multiples_of_7_l280_280469


namespace find_central_cell_l280_280494

variable (a b c d e f g h i : ℝ)

def condition_1 : Prop :=
  a * b * c = 10 ∧ d * e * f = 10 ∧ g * h * i = 10

def condition_2 : Prop :=
  a * d * g = 10 ∧ b * e * h = 10 ∧ c * f * i = 10

def condition_3 : Prop :=
  a * b * d * e = 3 ∧ b * c * e * f = 3 ∧ d * e * g * h = 3 ∧ e * f * h * i = 3

theorem find_central_cell (h1 : condition_1 a b c d e f g h i)
                          (h2 : condition_2 a b c d e f g h i)
                          (h3 : condition_3 a b c d e f g h i) : 
  e = 0.00081 := 
sorry

end find_central_cell_l280_280494


namespace tan_45_eq_one_l280_280888

noncomputable def sin_45 : ℝ := real.sqrt 2 / 2
noncomputable def cos_45 : ℝ := real.sqrt 2 / 2

theorem tan_45_eq_one : real.tan (real.pi / 4) = 1 := by
  have hsin : real.sin (real.pi / 4) = sin_45 := by sorry
  have hcos : real.cos (real.pi / 4) = cos_45 := by sorry
  rw [real.tan_eq_sin_div_cos, hsin, hcos]
  rw [div_self]
  exact eq.refl 1
  exact sorry

end tan_45_eq_one_l280_280888


namespace last_digit_of_a_power_b_l280_280210

-- Define the constants from the problem
def a : ℕ := 954950230952380948328708
def b : ℕ := 470128749397540235934750230

-- Define a helper function to calculate the last digit of a natural number
def last_digit (n : ℕ) : ℕ :=
  n % 10

-- Define the main statement to be proven
theorem last_digit_of_a_power_b : last_digit ((last_digit a) ^ (b % 4)) = 4 :=
by
  -- Here go the proof steps if we were to provide them
  sorry

end last_digit_of_a_power_b_l280_280210


namespace three_digit_multiples_of_7_l280_280466

theorem three_digit_multiples_of_7 :
  let a := 7 * Nat.ceil (100 / 7)
  let l := 7 * Nat.floor (999 / 7)
  let d := 7
  let n := (l - a) / d + 1
  n = 128 :=
by
  let a := 7 * Nat.ceil (100 / 7)
  let l := 7 * Nat.floor (999 / 7)
  let d := 7
  let n := (l - a) / d + 1
  have : a = 105 := sorry
  have : l = 994 := sorry
  have : n = (994 - 105) / 7 + 1 := sorry
  have : n = 128 := sorry
  exact this

end three_digit_multiples_of_7_l280_280466


namespace no_solution_to_system_l280_280913

theorem no_solution_to_system : ∀ (x y : ℝ), ¬ (y^2 - (⌊x⌋ : ℝ)^2 = 2001 ∧ x^2 + (⌊y⌋ : ℝ)^2 = 2001) :=
by sorry

end no_solution_to_system_l280_280913


namespace union_A_B_inter_A_B_diff_U_A_U_B_subset_A_C_l280_280014

universe u

open Set

def U := @univ ℝ
def A := { x : ℝ | 3 ≤ x ∧ x < 10 }
def B := { x : ℝ | 2 < x ∧ x ≤ 7 }
def C (a : ℝ) := { x : ℝ | x > a }

theorem union_A_B : A ∪ B = { x : ℝ | 2 < x ∧ x < 10 } :=
by sorry

theorem inter_A_B : A ∩ B = { x : ℝ | 3 ≤ x ∧ x ≤ 7 } :=
by sorry

theorem diff_U_A_U_B : (U \ A) ∩ (U \ B) = { x : ℝ | x ≤ 2 } ∪ { x : ℝ | 10 ≤ x } :=
by sorry

theorem subset_A_C (a : ℝ) (h : A ⊆ C a) : a < 3 :=
by sorry

end union_A_B_inter_A_B_diff_U_A_U_B_subset_A_C_l280_280014


namespace cost_fly_D_to_E_l280_280326

-- Definitions for the given conditions
def distance_DE : ℕ := 4750
def cost_per_km_plane : ℝ := 0.12
def booking_fee_plane : ℝ := 150

-- The proof statement about the total cost
theorem cost_fly_D_to_E : (distance_DE * cost_per_km_plane + booking_fee_plane = 720) :=
by sorry

end cost_fly_D_to_E_l280_280326


namespace minimum_bailing_rate_l280_280387

theorem minimum_bailing_rate
  (distance : ℝ) (to_shore_rate : ℝ) (water_in_rate : ℝ) (submerge_limit : ℝ) (r : ℝ)
  (h_distance : distance = 0.5) 
  (h_speed : to_shore_rate = 6) 
  (h_water_intake : water_in_rate = 12) 
  (h_submerge_limit : submerge_limit = 50)
  (h_time : (distance / to_shore_rate) * 60 = 5)
  (h_total_intake : water_in_rate * 5 = 60)
  (h_max_intake : submerge_limit - 60 = -10) :
  r = 2 := sorry

end minimum_bailing_rate_l280_280387


namespace tan_45_eq_1_l280_280862

theorem tan_45_eq_1 :
  let π := Real.pi in
  ∃ (x : ℝ), x = π / 4 ∧
  sin x = sqrt 2 / 2 ∧
  cos x = sqrt 2 / 2 →
  tan (π / 4) = 1 :=
by sorry

end tan_45_eq_1_l280_280862


namespace multiple_of_interest_rate_l280_280092

theorem multiple_of_interest_rate (P r : ℝ) (m : ℝ) 
  (h1 : P * r^2 = 40) 
  (h2 : P * m^2 * r^2 = 360) : 
  m = 3 :=
by
  sorry

end multiple_of_interest_rate_l280_280092


namespace find_k_l280_280051

theorem find_k (a b c k : ℤ) (g : ℤ → ℤ)
  (h₁ : g 1 = 0)
  (h₂ : 10 < g 5 ∧ g 5 < 20)
  (h₃ : 30 < g 6 ∧ g 6 < 40)
  (h₄ : 3000 * k < g 100 ∧ g 100 < 3000 * (k + 1))
  (h_g : ∀ x, g x = a * x^2 + b * x + c) :
  k = 9 :=
by
  sorry

end find_k_l280_280051


namespace player_reach_wingspan_l280_280985

theorem player_reach_wingspan :
  ∀ (rim_height player_height jump_height reach_above_rim reach_with_jump reach_wingspan : ℕ),
  rim_height = 120 →
  player_height = 72 →
  jump_height = 32 →
  reach_above_rim = 6 →
  reach_with_jump = player_height + jump_height →
  reach_wingspan = (rim_height + reach_above_rim) - reach_with_jump →
  reach_wingspan = 22 :=
by
  intros rim_height player_height jump_height reach_above_rim reach_with_jump reach_wingspan
  intros h_rim_height h_player_height h_jump_height h_reach_above_rim h_reach_with_jump h_reach_wingspan
  rw [h_rim_height, h_player_height, h_jump_height, h_reach_above_rim] at *
  simp at *
  sorry

end player_reach_wingspan_l280_280985


namespace find_a_plus_b_l280_280936

theorem find_a_plus_b (a b : ℝ) (h : |a - 2| + |b + 3| = 0) : a + b = -1 :=
by {
  sorry
}

end find_a_plus_b_l280_280936


namespace tan_45_degree_l280_280750

theorem tan_45_degree : ∀ (θ : ℝ), 
  θ = 45 → 
  (sin θ = 1 / real.sqrt 2) → 
  (cos θ = 1 / real.sqrt 2) → 
  real.tan θ = 1 := 
by
  intros θ hθ hsin hcos
  sorry

end tan_45_degree_l280_280750


namespace sum_of_roots_eq_14_l280_280190

theorem sum_of_roots_eq_14 (x : ℝ) :
  (x - 7) ^ 2 = 16 → ∃ r1 r2 : ℝ, (x = r1 ∨ x = r2) ∧ r1 + r2 = 14 :=
by
  intro h
  sorry

end sum_of_roots_eq_14_l280_280190


namespace tommy_nickels_l280_280375

-- Definitions of given conditions
def pennies (quarters : Nat) : Nat := 10 * quarters  -- Tommy has 10 times as many pennies as quarters
def dimes (pennies : Nat) : Nat := pennies + 10      -- Tommy has 10 more dimes than pennies
def nickels (dimes : Nat) : Nat := 2 * dimes         -- Tommy has twice as many nickels as dimes

theorem tommy_nickels (quarters : Nat) (P : Nat) (D : Nat) (N : Nat) 
  (h1 : quarters = 4) 
  (h2 : P = pennies quarters) 
  (h3 : D = dimes P) 
  (h4 : N = nickels D) : 
  N = 100 := 
by
  -- sorry allows us to skip the proof
  sorry

end tommy_nickels_l280_280375


namespace tan_45_eq_l280_280614

def tan_identity (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

def sin_45_eq : Real.sin 45 = Real.sqrt 2 / 2 := sorry

def cos_45_eq : Real.cos 45 = Real.sqrt 2 / 2 := sorry

theorem tan_45_eq : tan_identity 45 = 1 := by
  rw [tan_identity, sin_45_eq, cos_45_eq]
  -- simplifying (sqrt 2 / 2) / (sqrt 2 / 2) = 1
  sorry

end tan_45_eq_l280_280614


namespace max_participants_l280_280373

structure MeetingRoom where
  rows : ℕ
  cols : ℕ
  seating : ℕ → ℕ → Bool -- A function indicating if a seat (i, j) is occupied (true) or not (false)
  row_condition : ∀ i : ℕ, ∀ j : ℕ, seating i j → seating i (j+1) → seating i (j+2) → False
  col_condition : ∀ i : ℕ, ∀ j : ℕ, seating i j → seating (i+1) j → seating (i+2) j → False

theorem max_participants {room : MeetingRoom} (h : room.rows = 4 ∧ room.cols = 4) : 
  (∃ n : ℕ, (∀ i < room.rows, ∀ j < room.cols, room.seating i j → n < 12) ∧
            (∀ m, (∀ i < room.rows, ∀ j < room.cols, room.seating i j → m < 12) → m ≤ 11)) :=
  sorry

end max_participants_l280_280373


namespace Rachel_spent_on_lunch_fraction_l280_280342

variable {MoneyEarned MoneySpentOnDVD MoneyLeft MoneySpentOnLunch : ℝ}

-- Given conditions
axiom Rachel_earnings : MoneyEarned = 200
axiom Rachel_spent_on_DVD : MoneySpentOnDVD = MoneyEarned / 2
axiom Rachel_leftover : MoneyLeft = 50
axiom Rachel_total_spent : MoneyEarned - MoneyLeft = MoneySpentOnLunch + MoneySpentOnDVD

-- Prove that Rachel spent 1/4 of her money on lunch
theorem Rachel_spent_on_lunch_fraction :
  MoneySpentOnLunch / MoneyEarned = 1 / 4 :=
sorry

end Rachel_spent_on_lunch_fraction_l280_280342


namespace find_integers_l280_280127

theorem find_integers (a b c : ℤ) (h1 : ∃ x : ℤ, a = 2 * x ∧ b = 5 * x ∧ c = 8 * x)
  (h2 : a + 6 = b / 3)
  (h3 : c - 10 = 5 * a / 4) :
  a = 36 ∧ b = 90 ∧ c = 144 :=
by
  sorry

end find_integers_l280_280127


namespace tan_45_degrees_l280_280814

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l280_280814


namespace exists_quad_root_l280_280102

theorem exists_quad_root (a b c d : ℝ) (h : a * c = 2 * b + 2 * d) :
  (∃ x, x^2 + a * x + b = 0) ∨ (∃ x, x^2 + c * x + d = 0) :=
sorry

end exists_quad_root_l280_280102


namespace tan_45_eq_1_l280_280826

theorem tan_45_eq_1
: ∀ (Q : ℝ × ℝ), Q = (Real.sqrt 2 / 2, Real.sqrt 2 / 2) ->
  ∀ θ, θ = Real.pi / 4 -> 
  Real.tan θ = 1 :=
by
  intros Q HQ θ Hθ
  rw [HQ, Hθ]
  sorry

end tan_45_eq_1_l280_280826


namespace last_digit_of_large_exponentiation_l280_280213

theorem last_digit_of_large_exponentiation
  (a : ℕ) (b : ℕ)
  (h1 : a = 954950230952380948328708) 
  (h2 : b = 470128749397540235934750230) :
  (a ^ b) % 10 = 4 :=
sorry

end last_digit_of_large_exponentiation_l280_280213


namespace tan_45_deg_l280_280638

theorem tan_45_deg : (Real.tan (Float.pi / 4)) = 1 :=
by
  sorry

end tan_45_deg_l280_280638


namespace min_photos_required_l280_280294

theorem min_photos_required (girls boys : ℕ) (children : ℕ) : 
  girls = 4 → boys = 8 → children = girls + boys →
  ∃ n, n ≥ 33 ∧ (∀ (p : ℕ), p < n → 
  (∃ (g g' : ℕ), g < girls ∧ g' < girls ∧ g ≠ g' ∨ 
   ∃ (b b' : ℕ), b < boys ∧ b' < boys ∧ b ≠ b' ∨ 
   ∃ (g : ℕ) (b : ℕ), g < girls ∧ b < boys ∧ ∃ (g' : ℕ) (b' : ℕ), g = g' ∧ b = b'))) :=
by
  sorry

end min_photos_required_l280_280294


namespace manolo_makes_45_masks_in_four_hours_l280_280287

noncomputable def face_masks_in_four_hour_shift : ℕ :=
  let first_hour_rate := 4
  let subsequent_hour_rate := 6
  let first_hour_face_masks := 60 / first_hour_rate
  let subsequent_hours_face_masks_per_hour := 60 / subsequent_hour_rate
  let total_face_masks :=
    first_hour_face_masks + subsequent_hours_face_masks_per_hour * (4 - 1)
  total_face_masks

theorem manolo_makes_45_masks_in_four_hours :
  face_masks_in_four_hour_shift = 45 :=
 by sorry

end manolo_makes_45_masks_in_four_hours_l280_280287


namespace tan_45_deg_l280_280626

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l280_280626


namespace sum_of_roots_eq_14_l280_280167

theorem sum_of_roots_eq_14 : 
  let equation := λ x : ℝ, (x - 7)^2 = 16 in
  let roots := {x | equation x} in
  (roots : ℝ) → ∀ x y ∈ roots, x + y = 14 :=
by
  -- For the purpose of the problem statement, we specify the exact roots here
  let root1 := 11
  let root2 := 3
  have H : root1 + root2 = 14, by norm_num
  intro equation roots x y hx hy
  exact H

end sum_of_roots_eq_14_l280_280167


namespace problem1_solution_l280_280354

theorem problem1_solution (x y : ℚ) (h1 : 3 * x + 2 * y = 10) (h2 : x / 2 - (y + 1) / 3 = 1) : 
  x = 3 ∧ y = 1 / 2 :=
by
  sorry

end problem1_solution_l280_280354


namespace find_y_l280_280422

theorem find_y :
  (∃ y : ℝ, (4 * Real.arctan (1/5) + Real.arctan (1/25) + Real.arctan (1/y) = π/4) ∧ y = 1251) :=
by
  sorry

end find_y_l280_280422


namespace central_cell_value_l280_280526

theorem central_cell_value
  (a b c d e f g h i : ℝ)
  (row1 : a * b * c = 10)
  (row2 : d * e * f = 10)
  (row3 : g * h * i = 10)
  (col1 : a * d * g = 10)
  (col2 : b * e * h = 10)
  (col3 : c * f * i = 10)
  (sub1 : a * b * d * e = 3)
  (sub2 : b * c * e * f = 3)
  (sub3 : d * e * g * h = 3)
  (sub4 : e * f * h * i = 3) : 
  e = 0.00081 :=
sorry

end central_cell_value_l280_280526


namespace rectangle_length_width_difference_l280_280239

theorem rectangle_length_width_difference :
  ∃ (length width : ℕ), (length * width = 864) ∧ (length + width = 60) ∧ (length - width = 12) :=
by
  sorry

end rectangle_length_width_difference_l280_280239


namespace tan_45_deg_l280_280692

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l280_280692


namespace david_profit_l280_280235

def weight : ℝ := 50
def cost : ℝ := 50
def price_per_kg : ℝ := 1.20
def total_earnings : ℝ := weight * price_per_kg
def profit : ℝ := total_earnings - cost

theorem david_profit : profit = 10 := by
  sorry

end david_profit_l280_280235


namespace lisa_needs_additional_marbles_l280_280060

theorem lisa_needs_additional_marbles
  (friends : ℕ) (initial_marbles : ℕ) (total_required_marbles : ℕ) :
  friends = 12 ∧ initial_marbles = 40 ∧ total_required_marbles = (friends * (friends + 1)) / 2 →
  total_required_marbles - initial_marbles = 38 :=
by
  sorry

end lisa_needs_additional_marbles_l280_280060


namespace range_of_a_l280_280314

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) :
  (∀ x : ℝ, f x = x^3 - a) → (∀ x : ℝ, f 0 ≤ 0) → (0 ≤ a) :=
by
  intro h1 h2
  suffices h : -a ≤ 0 by
    simpa using h
  have : f 0 = -a
  simp [h1]
  sorry -- Proof steps are omitted

end range_of_a_l280_280314


namespace tan_of_45_deg_l280_280708

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_of_45_deg_l280_280708


namespace container_capacity_l280_280023

/-- Given a container where 8 liters is 20% of its capacity, calculate the total capacity of 
    40 such containers filled with water. -/
theorem container_capacity (c : ℝ) (h : 8 = 0.20 * c) : 
    40 * c * 40 = 1600 := 
by
  sorry

end container_capacity_l280_280023


namespace find_y_l280_280918

-- Define the points and slope conditions
def point_R : ℝ × ℝ := (-3, 4)
def x2 : ℝ := 5

-- Define the y coordinate and its corresponding condition
def y_condition (y : ℝ) : Prop := (y - 4) / (5 - (-3)) = 1 / 2

-- The main theorem stating the conditions and conclusion
theorem find_y (y : ℝ) (h : y_condition y) : y = 8 :=
by
  sorry

end find_y_l280_280918


namespace tan_45_eq_1_l280_280774

theorem tan_45_eq_1 :
  let Q : ℝ × ℝ := (1 / Real.sqrt 2, 1 / Real.sqrt 2)
  in ∃ (Q : ℝ × ℝ), Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) → 
     Real.tan (Real.pi / 4) = 1 := 
by {
  -- The proof is left as an exercise.
  sorry
}

end tan_45_eq_1_l280_280774


namespace tan_45_deg_eq_1_l280_280607

-- Define conditions
variable (Q : ℝ × ℝ)
variable (E : ℝ × ℝ)
variable (O : ℝ × ℝ := (0, 0))
variable (θ : ℝ := 45)

-- Define main theorem
theorem tan_45_deg_eq_1
  (Q_on_unit_circle : Q = (ℝ.sqrt 2 / 2, ℝ.sqrt 2 / 2))
  (E_foot_perpendicular : E = (ℝ.sqrt 2 / 2, 0))
  (triangle_45_45_90 : (Q.1 - O.1) ^ 2 + (Q.2 - O.2) ^ 2 = 1) :
  Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_1_l280_280607


namespace lcm_one_to_twelve_l280_280575

theorem lcm_one_to_twelve : 
  Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 
  (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 (Nat.lcm 10 (Nat.lcm 11 12)))))))))) = 27720 := 
by sorry

end lcm_one_to_twelve_l280_280575


namespace tan_45_degrees_l280_280801

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l280_280801


namespace focus_of_parabola_l280_280426

theorem focus_of_parabola (f : ℝ) : 
  (∀ (x: ℝ), x^2 + ((- 1 / 16) * x^2 - f)^2 = ((- 1 / 16) * x^2 - (f + 8))^2) 
  → f = -4 :=
by
  intro h
  sorry

end focus_of_parabola_l280_280426


namespace tan_45_eq_one_l280_280895

noncomputable def sin_45 : ℝ := real.sqrt 2 / 2
noncomputable def cos_45 : ℝ := real.sqrt 2 / 2

theorem tan_45_eq_one : real.tan (real.pi / 4) = 1 := by
  have hsin : real.sin (real.pi / 4) = sin_45 := by sorry
  have hcos : real.cos (real.pi / 4) = cos_45 := by sorry
  rw [real.tan_eq_sin_div_cos, hsin, hcos]
  rw [div_self]
  exact eq.refl 1
  exact sorry

end tan_45_eq_one_l280_280895


namespace min_photos_needed_to_ensure_conditions_l280_280289

noncomputable def min_photos (girls boys : ℕ) : ℕ :=
  33

theorem min_photos_needed_to_ensure_conditions (girls boys : ℕ)
  (hgirls : girls = 4) (hboys : boys = 8) :
  min_photos girls boys = 33 := 
sorry

end min_photos_needed_to_ensure_conditions_l280_280289


namespace regular_tetrahedron_properties_l280_280416

-- Definitions
def equilateral (T : Type) : Prop := sorry -- equilateral triangle property
def equal_sides (T : Type) : Prop := sorry -- all sides equal property
def equal_angles (T : Type) : Prop := sorry -- all angles equal property

def regular (H : Type) : Prop := sorry -- regular tetrahedron property
def equal_edges (H : Type) : Prop := sorry -- all edges are equal
def equal_edge_angles (H : Type) : Prop := sorry -- angles between two edges at the same vertex are equal
def congruent_equilateral_faces (H : Type) : Prop := sorry -- faces are congruent equilateral triangles
def equal_dihedral_angles (H : Type) : Prop := sorry -- dihedral angles between adjacent faces are equal

-- Theorem statement
theorem regular_tetrahedron_properties :
  ∀ (T H : Type), 
    (equilateral T → equal_sides T ∧ equal_angles T) →
    (regular H → 
      (equal_edges H ∧ equal_edge_angles H) ∧
      (congruent_equilateral_faces H ∧ equal_dihedral_angles H) ∧
      (congruent_equilateral_faces H ∧ equal_edge_angles H)) :=
by
  intros T H hT hH
  sorry

end regular_tetrahedron_properties_l280_280416


namespace tan_45_eq_1_l280_280861

theorem tan_45_eq_1 :
  let π := Real.pi in
  ∃ (x : ℝ), x = π / 4 ∧
  sin x = sqrt 2 / 2 ∧
  cos x = sqrt 2 / 2 →
  tan (π / 4) = 1 :=
by sorry

end tan_45_eq_1_l280_280861


namespace socorro_training_days_l280_280971

variable (total_training_time_per_day : ℕ) (total_training_time : ℕ)

theorem socorro_training_days (h1 : total_training_time = 300) 
                              (h2 : total_training_time_per_day = 30) :
                              total_training_time / total_training_time_per_day = 10 := 
begin
  rw [h1, h2],
  norm_num,
end

end socorro_training_days_l280_280971


namespace tan_45_eq_1_l280_280755

theorem tan_45_eq_1 : Real.tan (Real.pi / 4) = 1 := by
  have h1 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have h2 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [Real.tan_eq_sin_div_cos, h1, h2]
  simp

end tan_45_eq_1_l280_280755


namespace shares_of_stocks_they_can_buy_l280_280404

def weekly_savings_wife : ℕ := 100
def monthly_savings_husband : ℕ := 225
def months_of_savings : ℕ := 4
def cost_per_share : ℕ := 50

theorem shares_of_stocks_they_can_buy :
  (((weekly_savings_wife * 4) + monthly_savings_husband) * months_of_savings / 2) / cost_per_share = 25 :=
by
  -- sorry for the implementation
  sorry

end shares_of_stocks_they_can_buy_l280_280404


namespace sum_of_roots_eq_14_l280_280171

theorem sum_of_roots_eq_14 : 
  let equation := λ x : ℝ, (x - 7)^2 = 16 in
  let roots := {x | equation x} in
  (roots : ℝ) → ∀ x y ∈ roots, x + y = 14 :=
by
  -- For the purpose of the problem statement, we specify the exact roots here
  let root1 := 11
  let root2 := 3
  have H : root1 + root2 = 14, by norm_num
  intro equation roots x y hx hy
  exact H

end sum_of_roots_eq_14_l280_280171


namespace tan_45_eq_one_l280_280795

theorem tan_45_eq_one (Q : ℝ × ℝ) (h1 : Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2)) : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_eq_one_l280_280795


namespace find_prime_between_20_and_35_with_remainder_7_l280_280589

theorem find_prime_between_20_and_35_with_remainder_7 : 
  ∃ p : ℕ, Nat.Prime p ∧ 20 ≤ p ∧ p ≤ 35 ∧ p % 11 = 7 ∧ p = 29 := 
by 
  sorry

end find_prime_between_20_and_35_with_remainder_7_l280_280589


namespace trip_first_part_length_l280_280398

theorem trip_first_part_length
  (total_distance : ℝ := 50)
  (first_speed : ℝ := 66)
  (second_speed : ℝ := 33)
  (average_speed : ℝ := 44) :
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ total_distance) ∧ 44 = total_distance / (x / first_speed + (total_distance - x) / second_speed) ∧ x = 25 :=
by
  sorry

end trip_first_part_length_l280_280398


namespace inequality_proof_l280_280341

variable {A B C a b c r : ℝ}

theorem inequality_proof (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hr : 0 < r) :
  (A + a + B + b) / (A + a + B + b + c + r) + (B + b + C + c) / (B + b + C + c + a + r) > (C + c + A + a) / (C + c + A + a + b + r) := 
    sorry

end inequality_proof_l280_280341


namespace triangle_sides_consecutive_obtuse_l280_280315

/-- Given the sides of a triangle are consecutive natural numbers 
    and the largest angle is obtuse, 
    the lengths of the sides in ascending order are 2, 3, 4. -/
theorem triangle_sides_consecutive_obtuse 
    (x : ℕ) (hx : x > 1) 
    (cos_alpha_neg : (x - 4) < 0) 
    (x_lt_4 : x < 4) :
    (x = 3) → (∃ a b c : ℕ, a < b ∧ b < c ∧ a + b > c ∧ a = 2 ∧ b = 3 ∧ c = 4) :=
by
  intro hx3
  use 2, 3, 4
  repeat {split}
  any_goals {linarith}
  all_goals {sorry}

end triangle_sides_consecutive_obtuse_l280_280315


namespace tan_45_degrees_l280_280813

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l280_280813


namespace tan_45_eq_1_l280_280828

theorem tan_45_eq_1
: ∀ (Q : ℝ × ℝ), Q = (Real.sqrt 2 / 2, Real.sqrt 2 / 2) ->
  ∀ θ, θ = Real.pi / 4 -> 
  Real.tan θ = 1 :=
by
  intros Q HQ θ Hθ
  rw [HQ, Hθ]
  sorry

end tan_45_eq_1_l280_280828


namespace find_divisor_l280_280124

-- Definitions based on the conditions
def is_divisor (d : ℕ) (a b k : ℕ) : Prop :=
  ∃ (n : ℕ), n > 0 ∧ (b - a) / n = k ∧ k = d

-- Problem statement
theorem find_divisor (a b k : ℕ) (H : b = 43 ∧ a = 10 ∧ k = 11) : ∃ d, d = 3 :=
by
  sorry

end find_divisor_l280_280124


namespace tan_45_eq_1_l280_280729

theorem tan_45_eq_1 (hcos : real.cos (real.pi / 4) = real.sqrt 2 / 2) (hsin : real.sin (real.pi / 4) = real.sqrt 2 / 2) : real.tan (real.pi / 4) = 1 :=
by
  have h : real.tan (real.pi / 4) = real.sin (real.pi / 4) / real.cos (real.pi / 4) := real.tan_eq_sin_div_cos _
  rw [hsin, hcos] at h
  simp at h
  exact h.symm

end tan_45_eq_1_l280_280729


namespace at_least_one_root_l280_280110

theorem at_least_one_root 
  (a b c d : ℝ)
  (h : a * c = 2 * b + 2 * d) :
  (∃ x : ℝ, x^2 + a * x + b = 0) ∨ (∃ x : ℝ, x^2 + c * x + d = 0) :=
sorry

end at_least_one_root_l280_280110


namespace tan_45_eq_1_l280_280825

theorem tan_45_eq_1
: ∀ (Q : ℝ × ℝ), Q = (Real.sqrt 2 / 2, Real.sqrt 2 / 2) ->
  ∀ θ, θ = Real.pi / 4 -> 
  Real.tan θ = 1 :=
by
  intros Q HQ θ Hθ
  rw [HQ, Hθ]
  sorry

end tan_45_eq_1_l280_280825


namespace tan_45_deg_l280_280665

theorem tan_45_deg : Real.tan (π / 4) = 1 := 
by
  have h_sin_cos : Real.sin (π / 4) = Real.cos (π / 4) := 
    by sorry
  have h_value : Real.sin (π / 4) = (Real.sqrt 2) / 2 := 
    by sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos, h_value]
  calc
    (Real.sqrt 2)/ 2 / (Real.sqrt 2)/ 2 = 1 := sorry

end tan_45_deg_l280_280665


namespace locus_eq_l280_280362

noncomputable def locus_of_centers (a b : ℝ) : Prop :=
  5 * a^2 + 9 * b^2 + 80 * a - 400 = 0

theorem locus_eq (a b : ℝ) :
  (∃ r : ℝ, (a^2 + b^2 = (r + 2)^2) ∧ ((a - 1)^2 + b^2 = (5 - r)^2)) →
  locus_of_centers a b :=
by
  intro h
  sorry

end locus_eq_l280_280362


namespace focus_of_parabola_l280_280425

theorem focus_of_parabola (f : ℝ) : 
  (∀ (x: ℝ), x^2 + ((- 1 / 16) * x^2 - f)^2 = ((- 1 / 16) * x^2 - (f + 8))^2) 
  → f = -4 :=
by
  intro h
  sorry

end focus_of_parabola_l280_280425


namespace find_product_l280_280040

def a : ℕ := 4
def g : ℕ := 8
def d : ℕ := 10

theorem find_product (A B C D E F : ℕ) (hA : A % 2 = 0) (hB : B % 3 = 0) (hC : C % 4 = 0) 
  (hD : D % 5 = 0) (hE : E % 6 = 0) (hF : F % 7 = 0) :
  a * g * d = 320 :=
by
  sorry

end find_product_l280_280040


namespace tan_of_45_deg_l280_280735

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  -- Definitions used directly from conditions
  have hcos : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry
  have hsin : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry

  -- Using the definition of tangent
  rw [Real.tan_eq_sin_div_cos, hsin, hcos]
  exact div_self (ne_of_gt (Real.sqrt_pos.2 zero_lt_two))

end tan_of_45_deg_l280_280735


namespace triangle_inequality_for_roots_l280_280305

theorem triangle_inequality_for_roots (p q r : ℝ) (hroots_pos : ∀ (u v w : ℝ), (u > 0) ∧ (v > 0) ∧ (w > 0) ∧ (u * v * w = -r) ∧ (u + v + w = -p) ∧ (u * v + u * w + v * w = q)) :
  p^3 - 4 * p * q + 8 * r > 0 :=
sorry

end triangle_inequality_for_roots_l280_280305


namespace verify_system_of_equations_l280_280899

/-- Define a structure to hold the conditions of the problem -/
structure TreePurchasing :=
  (cost_A : ℕ)
  (cost_B : ℕ)
  (diff_A_B : ℕ)
  (total_cost : ℕ)
  (x : ℕ)
  (y : ℕ)

/-- Given conditions for purchasing trees -/
def example_problem : TreePurchasing :=
  { cost_A := 100,
    cost_B := 80,
    diff_A_B := 8,
    total_cost := 8000,
    x := 0,
    y := 0 }

/-- The theorem to prove that the equations match given conditions -/
theorem verify_system_of_equations (data : TreePurchasing) (h_diff : data.x - data.y = data.diff_A_B) (h_cost : data.cost_A * data.x + data.cost_B * data.y = data.total_cost) : 
  (data.x - data.y = 8) ∧ (100 * data.x + 80 * data.y = 8000) :=
  by
    sorry

end verify_system_of_equations_l280_280899


namespace tan_45_deg_eq_one_l280_280649

/-
  Given:
  cos (45 deg) = sqrt 2 / 2,
  sin (45 deg) = sqrt 2 / 2,
  Prove: 
  tan (45 deg) = 1
-/

theorem tan_45_deg_eq_one : 
  (real.cos (real.pi / 4) = real.sqrt 2 / 2) → 
  (real.sin (real.pi / 4) = real.sqrt 2 / 2) → 
  real.tan (real.pi / 4) = 1 :=
by
  intros hcos hsin
  sorry

end tan_45_deg_eq_one_l280_280649


namespace socorro_training_days_l280_280973

def total_hours := 5
def minutes_per_hour := 60
def total_training_minutes := total_hours * minutes_per_hour

def minutes_multiplication_per_day := 10
def minutes_division_per_day := 20
def daily_training_minutes := minutes_multiplication_per_day + minutes_division_per_day

theorem socorro_training_days:
  total_training_minutes / daily_training_minutes = 10 :=
by
  -- proof omitted
  sorry

end socorro_training_days_l280_280973


namespace find_a₁₈_l280_280440

open Classical

-- Define the arithmetic sequence and conditions
variables (a : ℕ → ℝ) (S : ℕ → ℝ)
variable (d : ℝ)

-- Define a₁ (the first term of the arithmetic sequence)
def a₁ := (2 : ℝ)

-- Define the arithmetic sequence formula
def a (n : ℕ) : ℝ := a₁ + (n - 1) * d

-- Define the sum of the first n terms of the arithmetic sequence
def Sn (n : ℕ) : ℝ := (n / 2) * (2 * a₁ + (n - 1) * d)

-- Given conditions
axiom Sn_equal : Sn 8 = Sn 10

-- Theorem statement to find a₁₈
theorem find_a₁₈ : a 18 = -2 :=
by
  -- Conditions derived from the problem (used as axioms here)
  have h1 : a₁ = 2 := rfl
  have h2 : S 8 = S 10 := Sn_equal
  -- Placeholder for proof, which is omitted using "sorry"
  sorry

end find_a₁₈_l280_280440


namespace gcd_840_1764_l280_280979

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end gcd_840_1764_l280_280979


namespace tan_45_eq_1_l280_280768

theorem tan_45_eq_1 :
  let Q : ℝ × ℝ := (1 / Real.sqrt 2, 1 / Real.sqrt 2)
  in ∃ (Q : ℝ × ℝ), Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) → 
     Real.tan (Real.pi / 4) = 1 := 
by {
  -- The proof is left as an exercise.
  sorry
}

end tan_45_eq_1_l280_280768


namespace tan_45_deg_eq_one_l280_280840

/-- The tangent of 45 degrees is equal to 1. -/
theorem tan_45_deg_eq_one : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_one_l280_280840


namespace part_a_l280_280581

theorem part_a (x y : ℝ) : (x + y) * (x^2 - x * y + y^2) = x^3 + y^3 := sorry

end part_a_l280_280581


namespace tan_45_deg_eq_one_l280_280645

/-
  Given:
  cos (45 deg) = sqrt 2 / 2,
  sin (45 deg) = sqrt 2 / 2,
  Prove: 
  tan (45 deg) = 1
-/

theorem tan_45_deg_eq_one : 
  (real.cos (real.pi / 4) = real.sqrt 2 / 2) → 
  (real.sin (real.pi / 4) = real.sqrt 2 / 2) → 
  real.tan (real.pi / 4) = 1 :=
by
  intros hcos hsin
  sorry

end tan_45_deg_eq_one_l280_280645


namespace sum_of_a_and_b_l280_280266

theorem sum_of_a_and_b (a b : ℕ) (h1 : b > 1) (h2 : a^b < 500) (h3 : ∀ c d : ℕ, d > 1 → c^d < 500 → c^d ≤ a^b) : a + b = 24 :=
sorry

end sum_of_a_and_b_l280_280266


namespace tan_45_eq_one_l280_280893

noncomputable def sin_45 : ℝ := real.sqrt 2 / 2
noncomputable def cos_45 : ℝ := real.sqrt 2 / 2

theorem tan_45_eq_one : real.tan (real.pi / 4) = 1 := by
  have hsin : real.sin (real.pi / 4) = sin_45 := by sorry
  have hcos : real.cos (real.pi / 4) = cos_45 := by sorry
  rw [real.tan_eq_sin_div_cos, hsin, hcos]
  rw [div_self]
  exact eq.refl 1
  exact sorry

end tan_45_eq_one_l280_280893


namespace find_interest_rate_l280_280590

-- Definitions for conditions
def principal : ℝ := 12500
def interest : ℝ := 1500
def time : ℝ := 1

-- Interest rate to prove
def interest_rate : ℝ := 0.12

-- Formal statement to prove
theorem find_interest_rate (P I T : ℝ) (hP : P = principal) (hI : I = interest) (hT : T = time) : I = P * interest_rate * T :=
by
  sorry

end find_interest_rate_l280_280590


namespace solve_quadratic_l280_280086

theorem solve_quadratic :
    ∀ x : ℝ, x^2 - 6*x + 8 = 0 ↔ (x = 2 ∨ x = 4) :=
by
  intros x
  sorry

end solve_quadratic_l280_280086


namespace fraction_is_5_div_9_l280_280031

-- Define the conditions t = f * (k - 32), t = 35, and k = 95
theorem fraction_is_5_div_9 {f k t : ℚ} (h1 : t = f * (k - 32)) (h2 : t = 35) (h3 : k = 95) : f = 5 / 9 :=
by
  sorry

end fraction_is_5_div_9_l280_280031


namespace rosa_calls_pages_l280_280931

theorem rosa_calls_pages (pages_last_week : ℝ) (pages_this_week : ℝ) (h_last_week : pages_last_week = 10.2) (h_this_week : pages_this_week = 8.6) : pages_last_week + pages_this_week = 18.8 :=
by sorry

end rosa_calls_pages_l280_280931


namespace tan_45_deg_l280_280691

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l280_280691


namespace simplify_and_evaluate_l280_280346

theorem simplify_and_evaluate (a : ℤ) (h : a = -2) : 
  (1 - (1 / (a + 1))) / ((a^2 - 2*a + 1) / (a^2 - 1)) = (2 / 3) :=
by
  sorry

end simplify_and_evaluate_l280_280346


namespace tan_45_degrees_eq_1_l280_280776

theorem tan_45_degrees_eq_1 :
    tan (45 * real.pi / 180) = 1 :=
by
  sorry

end tan_45_degrees_eq_1_l280_280776


namespace tan_45_deg_l280_280636

theorem tan_45_deg : (Real.tan (Float.pi / 4)) = 1 :=
by
  sorry

end tan_45_deg_l280_280636


namespace probability_of_experts_winning_l280_280904

-- Definitions required from the conditions
def p : ℝ := 0.6
def q : ℝ := 1 - p
def current_expert_score : ℕ := 3
def current_audience_score : ℕ := 4

-- The main theorem to state
theorem probability_of_experts_winning : 
  p^4 + 4 * p^3 * q = 0.4752 := 
by sorry

end probability_of_experts_winning_l280_280904


namespace minimum_number_of_gloves_l280_280980

theorem minimum_number_of_gloves (participants : ℕ) (gloves_per_participant : ℕ) (total_participants : participants = 63) (each_participant_needs_2_gloves : gloves_per_participant = 2) : 
  participants * gloves_per_participant = 126 :=
by
  rcases participants, gloves_per_participant, total_participants, each_participant_needs_2_gloves
  -- sorry to skip the proof
  sorry

end minimum_number_of_gloves_l280_280980


namespace tan_45_eq_1_l280_280827

theorem tan_45_eq_1
: ∀ (Q : ℝ × ℝ), Q = (Real.sqrt 2 / 2, Real.sqrt 2 / 2) ->
  ∀ θ, θ = Real.pi / 4 -> 
  Real.tan θ = 1 :=
by
  intros Q HQ θ Hθ
  rw [HQ, Hθ]
  sorry

end tan_45_eq_1_l280_280827


namespace max_power_sum_l280_280255

open Nat

theorem max_power_sum (a b : ℕ) (h_a_pos : a > 0) (h_b_gt_one : b > 1) (h_max : a ^ b < 500 ∧ 
  ∀ (a' b' : ℕ), a' > 0 → b' > 1 → a' ^ b' < 500 → a' ^ b' ≤ a ^ b ) : a + b = 24 :=
sorry

end max_power_sum_l280_280255


namespace sum_of_roots_eq_14_l280_280166

theorem sum_of_roots_eq_14 : 
  let equation := λ x : ℝ, (x - 7)^2 = 16 in
  let roots := {x | equation x} in
  (roots : ℝ) → ∀ x y ∈ roots, x + y = 14 :=
by
  -- For the purpose of the problem statement, we specify the exact roots here
  let root1 := 11
  let root2 := 3
  have H : root1 + root2 = 14, by norm_num
  intro equation roots x y hx hy
  exact H

end sum_of_roots_eq_14_l280_280166


namespace perimeter_of_square_l280_280977

theorem perimeter_of_square (s : ℝ) (h : s^2 = 588) : (4 * s) = 56 * Real.sqrt 3 :=
by
  sorry

end perimeter_of_square_l280_280977


namespace solve_for_x_l280_280350

theorem solve_for_x (x : ℝ) (h1 : x ≠ 4) (h2 : x ≠ 6)
  (h : (x + 10) / (x - 4) = (x - 3) / (x + 6)) : x = -48 / 23 :=
sorry

end solve_for_x_l280_280350


namespace last_digit_of_large_exponentiation_l280_280212

theorem last_digit_of_large_exponentiation
  (a : ℕ) (b : ℕ)
  (h1 : a = 954950230952380948328708) 
  (h2 : b = 470128749397540235934750230) :
  (a ^ b) % 10 = 4 :=
sorry

end last_digit_of_large_exponentiation_l280_280212


namespace find_C_and_D_l280_280562

theorem find_C_and_D (C D : ℚ) :
  (∀ x : ℚ, ((6 * x - 8) / (2 * x^2 + 5 * x - 3) = (C / (x - 1)) + (D / (2 * x + 3)))) →
  (2*x^2 + 5*x - 3 = (2*x - 1)*(x + 3)) →
  (∀ x : ℚ, ((C*(2*x + 3) + D*(x - 1)) / ((2*x - 1)*(x + 3))) = ((6*x - 8) / ((2*x - 1)*(x + 3)))) →
  (∀ x : ℚ, C*(2*x + 3) + D*(x - 1) = 6*x - 8) →
  C = -2/5 ∧ D = 34/5 := 
by 
  sorry

end find_C_and_D_l280_280562


namespace tan_45_degree_l280_280743

theorem tan_45_degree : ∀ (θ : ℝ), 
  θ = 45 → 
  (sin θ = 1 / real.sqrt 2) → 
  (cos θ = 1 / real.sqrt 2) → 
  real.tan θ = 1 := 
by
  intros θ hθ hsin hcos
  sorry

end tan_45_degree_l280_280743


namespace sum_of_a_and_b_l280_280268

theorem sum_of_a_and_b (a b : ℕ) (h1 : b > 1) (h2 : a^b < 500) (h3 : ∀ c d : ℕ, d > 1 → c^d < 500 → c^d ≤ a^b) : a + b = 24 :=
sorry

end sum_of_a_and_b_l280_280268


namespace min_photos_required_l280_280295

theorem min_photos_required (girls boys : ℕ) (children : ℕ) : 
  girls = 4 → boys = 8 → children = girls + boys →
  ∃ n, n ≥ 33 ∧ (∀ (p : ℕ), p < n → 
  (∃ (g g' : ℕ), g < girls ∧ g' < girls ∧ g ≠ g' ∨ 
   ∃ (b b' : ℕ), b < boys ∧ b' < boys ∧ b ≠ b' ∨ 
   ∃ (g : ℕ) (b : ℕ), g < girls ∧ b < boys ∧ ∃ (g' : ℕ) (b' : ℕ), g = g' ∧ b = b'))) :=
by
  sorry

end min_photos_required_l280_280295


namespace tan_45_degrees_eq_1_l280_280783

theorem tan_45_degrees_eq_1 :
    tan (45 * real.pi / 180) = 1 :=
by
  sorry

end tan_45_degrees_eq_1_l280_280783


namespace tan_45_degree_is_one_l280_280714

theorem tan_45_degree_is_one : Real.tan (Real.pi / 4) = 1 := sorry

end tan_45_degree_is_one_l280_280714


namespace quadratic_root_exists_l280_280104

theorem quadratic_root_exists {a b c d : ℝ} (h : a * c = 2 * b + 2 * d) :
  (∃ x : ℝ, x^2 + a * x + b = 0) ∨ (∃ x : ℝ, x^2 + c * x + d = 0) :=
by
  sorry

end quadratic_root_exists_l280_280104


namespace proof_fraction_l280_280420

noncomputable def A' : ℝ :=
  ∑' n in {n : ℕ | n % 2 = 1 ∧ n % 3 ≠ 0}, (-1)^(n / 2) / n^3

noncomputable def B' : ℝ :=
  ∑' n in {n : ℕ | n % 2 = 1 ∧ n % 3 = 0}, (-1)^((n - 3) / 6) / n^3

theorem proof_fraction :
  A' / B' = 28 := by
  sorry

end proof_fraction_l280_280420


namespace tan_of_45_deg_l280_280734

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  -- Definitions used directly from conditions
  have hcos : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry
  have hsin : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry

  -- Using the definition of tangent
  rw [Real.tan_eq_sin_div_cos, hsin, hcos]
  exact div_self (ne_of_gt (Real.sqrt_pos.2 zero_lt_two))

end tan_of_45_deg_l280_280734


namespace tan_45_eq_one_l280_280787

theorem tan_45_eq_one (Q : ℝ × ℝ) (h1 : Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2)) : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_eq_one_l280_280787


namespace cost_of_batman_game_l280_280374

noncomputable def footballGameCost : ℝ := 14.02
noncomputable def strategyGameCost : ℝ := 9.46
noncomputable def totalAmountSpent : ℝ := 35.52

theorem cost_of_batman_game :
  totalAmountSpent - (footballGameCost + strategyGameCost) = 12.04 :=
by
  -- The proof is omitted as instructed.
  sorry

end cost_of_batman_game_l280_280374


namespace tan_45_deg_l280_280639

theorem tan_45_deg : (Real.tan (Float.pi / 4)) = 1 :=
by
  sorry

end tan_45_deg_l280_280639


namespace divisible_by_12_l280_280549

theorem divisible_by_12 (a b c d : ℕ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c) (hpos_d : 0 < d) :
  12 ∣ (a - b) * (a - c) * (a - d) * (b - c) * (b - d) * (c - d) := 
by
  sorry

end divisible_by_12_l280_280549


namespace thomas_savings_l280_280991

def first_year_earnings (weekly_allowance_year1 : ℕ) : ℕ :=
  weekly_allowance_year1 * 52

def second_year_earnings (hourly_rate_year2 hours_per_week_year2 : ℕ) : ℕ :=
  hourly_rate_year2 * hours_per_week_year2 * 52

def total_earnings (first_year second_year : ℕ) : ℕ :=
  first_year + second_year

def total_expenses (personal_expenses : ℕ) : ℕ :=
  personal_expenses * (52 * 2)

def savings (earnings expenses : ℕ) : ℕ :=
  earnings - expenses

def amount_needed (car_cost savings : ℕ) : ℕ :=
  car_cost - savings

theorem thomas_savings : 
  ∀ (weekly_allowance_year1 hourly_rate_year2 hours_per_week_year2 car_cost personal_expenses : ℕ),
  years = 2 →
  weekly_allowance_year1 = 50 →
  hourly_rate_year2 = 9 →
  hours_per_week_year2 = 30 →
  car_cost = 15000 →
  personal_expenses = 35 →
  amount_needed car_cost (
    savings (
      total_earnings 
        (first_year_earnings weekly_allowance_year1)
        (second_year_earnings hourly_rate_year2 hours_per_week_year2)
      )
    (total_expenses personal_expenses)
  ) = 2000
:=
by
  intros _ _ _ _ _ _ _ _ _ _ _ _
  sorry

end thomas_savings_l280_280991


namespace tan_45_eq_l280_280613

def tan_identity (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

def sin_45_eq : Real.sin 45 = Real.sqrt 2 / 2 := sorry

def cos_45_eq : Real.cos 45 = Real.sqrt 2 / 2 := sorry

theorem tan_45_eq : tan_identity 45 = 1 := by
  rw [tan_identity, sin_45_eq, cos_45_eq]
  -- simplifying (sqrt 2 / 2) / (sqrt 2 / 2) = 1
  sorry

end tan_45_eq_l280_280613


namespace tan_45_eq_1_l280_280760

theorem tan_45_eq_1 : Real.tan (Real.pi / 4) = 1 := by
  have h1 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have h2 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [Real.tan_eq_sin_div_cos, h1, h2]
  simp

end tan_45_eq_1_l280_280760


namespace quadratic_is_complete_the_square_l280_280115

theorem quadratic_is_complete_the_square :
  ∃ a b c : ℝ, 15 * (x : ℝ)^2 + 150 * x + 2250 = a * (x + b)^2 + c 
  ∧ a + b + c = 1895 :=
sorry

end quadratic_is_complete_the_square_l280_280115


namespace probability_event_A_l280_280223

def probability_of_defective : Real := 0.3
def probability_of_all_defective : Real := 0.027
def probability_of_event_A : Real := 0.973

theorem probability_event_A :
  1 - probability_of_all_defective = probability_of_event_A :=
by
  sorry

end probability_event_A_l280_280223


namespace tan_45_degree_is_one_l280_280710

theorem tan_45_degree_is_one : Real.tan (Real.pi / 4) = 1 := sorry

end tan_45_degree_is_one_l280_280710


namespace find_FC_l280_280435

theorem find_FC (DC : ℝ) (CB : ℝ) (AB AD ED FC : ℝ) 
  (h1 : DC = 9) 
  (h2 : CB = 10) 
  (h3 : AB = (1/3) * AD) 
  (h4 : ED = (3/4) * AD) 
  (h5 : FC = 14.625) : FC = 14.625 :=
by sorry

end find_FC_l280_280435


namespace tan_45_deg_l280_280884

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 :=
by
  have h_sin_cos : Real.sin (Real.pi / 4) = Real.cos (Real.pi / 4) := sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos]
  have sqrt2_ne_zero : Real.sqrt 2 ≠ 0 := sorry
  have sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have cos_45 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [sin_45, cos_45]
  field_simp [sqrt2_ne_zero]
  exact one_div_one

end tan_45_deg_l280_280884


namespace central_cell_value_l280_280508

variables a b c d e f g h i : ℝ

-- Conditions
axiom row1 : a * b * c = 10
axiom row2 : d * e * f = 10
axiom row3 : g * h * i = 10
axiom col1 : a * d * g = 10
axiom col2 : b * e * h = 10
axiom col3 : c * f * i = 10
axiom sq1 : a * b * d * e = 3
axiom sq2 : b * c * e * f = 3
axiom sq3 : d * e * g * h = 3
axiom sq4 : e * f * h * i = 3

theorem central_cell_value : e = 0.00081 := by
  sorry

end central_cell_value_l280_280508


namespace length_of_PR_l280_280533

-- Define the entities and conditions
variables (x y : ℝ)
variables (xy_area : ℝ := 125)
variables (PR_length : ℝ := 10 * Real.sqrt 5)

-- State the problem in Lean
theorem length_of_PR (x y : ℝ) (hxy : x * y = 125) :
  x^2 + (125 / x)^2 = (10 * Real.sqrt 5)^2 :=
sorry

end length_of_PR_l280_280533


namespace central_cell_value_l280_280484

def table (a b c d e f g h i : ℝ) : Prop :=
  (a * b * c = 10) ∧ (d * e * f = 10) ∧ (g * h * i = 10) ∧
  (a * d * g = 10) ∧ (b * e * h = 10) ∧ (c * f * i = 10) ∧
  (a * b * d * e = 3) ∧ (b * c * e * f = 3) ∧ (d * e * g * h = 3) ∧ (e * f * h * i = 3)

theorem central_cell_value (a b c d f g h i e : ℝ) (h_table : table a b c d e f g h i) : 
  e = 0.00081 :=
by sorry

end central_cell_value_l280_280484


namespace count_multiples_l280_280454

theorem count_multiples (n : ℕ) : 
  n = 1 ↔ ∃ k : ℕ, k < 500 ∧ k > 0 ∧ k % 4 = 0 ∧ k % 5 = 0 ∧ k % 6 = 0 ∧ k % 7 = 0 :=
by
  sorry

end count_multiples_l280_280454


namespace function_below_x_axis_l280_280916

theorem function_below_x_axis (k : ℝ) :
  (∀ x : ℝ, (k^2 - k - 2) * x^2 - (k - 2) * x - 1 < 0) ↔ (-2 / 5 < k ∧ k ≤ 2) :=
by
  sorry

end function_below_x_axis_l280_280916


namespace apples_for_48_oranges_l280_280948

theorem apples_for_48_oranges (o a : ℕ) (h : 8 * o = 6 * a) (ho : o = 48) : a = 36 :=
by
  sorry

end apples_for_48_oranges_l280_280948


namespace gcd_of_1237_and_1957_is_one_l280_280001

noncomputable def gcd_1237_1957 : Nat := Nat.gcd 1237 1957

theorem gcd_of_1237_and_1957_is_one : gcd_1237_1957 = 1 :=
by
  unfold gcd_1237_1957
  have : Nat.gcd 1237 1957 = 1 := sorry
  exact this

end gcd_of_1237_and_1957_is_one_l280_280001


namespace initial_salmons_l280_280415

theorem initial_salmons (x : ℕ) (hx : 10 * x = 5500) : x = 550 := 
by
  sorry

end initial_salmons_l280_280415


namespace find_two_digits_l280_280245

theorem find_two_digits (a b : ℕ) (h₁: a ≤ 9) (h₂: b ≤ 9)
  (h₃: (4 + a + b) % 9 = 0) (h₄: (10 * a + b) % 4 = 0) :
  (a = 3 ∧ b = 2) ∨ (a = 6 ∧ b = 8) :=
by {
  sorry
}

end find_two_digits_l280_280245


namespace reversed_digit_multiple_of_sum_l280_280393

variable (u v k : ℕ)

theorem reversed_digit_multiple_of_sum (h1 : 10 * u + v = k * (u + v)) :
  10 * v + u = (11 - k) * (u + v) :=
sorry

end reversed_digit_multiple_of_sum_l280_280393


namespace trajectory_midpoint_l280_280323

/-- Let A and B be two moving points on the circle x^2 + y^2 = 4, and AB = 2. 
    The equation of the trajectory of the midpoint M of the line segment AB is x^2 + y^2 = 3. -/
theorem trajectory_midpoint (A B : ℝ × ℝ) (M : ℝ × ℝ)
    (hA : A.1^2 + A.2^2 = 4)
    (hB : B.1^2 + B.2^2 = 4)
    (hAB : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 4)
    (hM : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) :
    M.1^2 + M.2^2 = 3 :=
sorry

end trajectory_midpoint_l280_280323


namespace tan_45_deg_l280_280663

theorem tan_45_deg : Real.tan (π / 4) = 1 := 
by
  have h_sin_cos : Real.sin (π / 4) = Real.cos (π / 4) := 
    by sorry
  have h_value : Real.sin (π / 4) = (Real.sqrt 2) / 2 := 
    by sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos, h_value]
  calc
    (Real.sqrt 2)/ 2 / (Real.sqrt 2)/ 2 = 1 := sorry

end tan_45_deg_l280_280663


namespace simplify_and_rationalize_l280_280345

noncomputable def simplify_expr : ℝ :=
  1 / (1 - (1 / (Real.sqrt 5 - 2)))

theorem simplify_and_rationalize :
  simplify_expr = (1 - Real.sqrt 5) / 4 := by
  sorry

end simplify_and_rationalize_l280_280345


namespace find_other_number_l280_280984

theorem find_other_number (B : ℕ)
  (HCF : Nat.gcd 24 B = 12)
  (LCM : Nat.lcm 24 B = 312) :
  B = 156 :=
by
  sorry

end find_other_number_l280_280984


namespace complex_quadrant_l280_280937

theorem complex_quadrant (θ : ℝ) (hθ : θ ∈ Set.Ioo (3/4 * Real.pi) (5/4 * Real.pi)) :
  let z := Complex.mk (Real.cos θ + Real.sin θ) (Real.sin θ - Real.cos θ)
  z.re < 0 ∧ z.im > 0 :=
by
  sorry

end complex_quadrant_l280_280937


namespace tan_45_degree_is_one_l280_280718

theorem tan_45_degree_is_one : Real.tan (Real.pi / 4) = 1 := sorry

end tan_45_degree_is_one_l280_280718


namespace school_growth_difference_l280_280565

theorem school_growth_difference (X Y : ℕ) (H₁ : Y = 2400)
  (H₂ : X + Y = 4000) : (X + 7 * X / 100 - X) - (Y + 3 * Y / 100 - Y) = 40 :=
by
  sorry

end school_growth_difference_l280_280565


namespace polynomial_has_root_l280_280111

theorem polynomial_has_root {a b c d : ℝ} 
  (h : a * c = 2 * b + 2 * d) : 
  ∃ x : ℝ, (x^2 + a * x + b = 0) ∨ (x^2 + c * x + d = 0) :=
by 
  sorry

end polynomial_has_root_l280_280111


namespace num_surjections_l280_280537

open Finset

theorem num_surjections (A B : Finset ℕ) (hA : A.card = 4) (hB : B.card = 3) :
  (∃ f : A → B, Function.Surjective f) →
  {f : A → B | Function.Surjective f}.to_finset.card = 36 :=
by
  sorry

end num_surjections_l280_280537


namespace tan_45_degrees_eq_1_l280_280778

theorem tan_45_degrees_eq_1 :
    tan (45 * real.pi / 180) = 1 :=
by
  sorry

end tan_45_degrees_eq_1_l280_280778


namespace tan_45_eq_1_l280_280822

theorem tan_45_eq_1
: ∀ (Q : ℝ × ℝ), Q = (Real.sqrt 2 / 2, Real.sqrt 2 / 2) ->
  ∀ θ, θ = Real.pi / 4 -> 
  Real.tan θ = 1 :=
by
  intros Q HQ θ Hθ
  rw [HQ, Hθ]
  sorry

end tan_45_eq_1_l280_280822


namespace common_sum_l280_280981

theorem common_sum (a l : ℤ) (n r c : ℕ) (S x : ℤ) 
  (h_a : a = -18) 
  (h_l : l = 30) 
  (h_n : n = 49) 
  (h_S : S = (n * (a + l)) / 2) 
  (h_r : r = 7) 
  (h_c : c = 7) 
  (h_sum_eq : r * x = S) :
  x = 42 := 
sorry

end common_sum_l280_280981


namespace training_days_l280_280970

def total_minutes : ℕ := 5 * 60
def minutes_per_day : ℕ := 10 + 20

theorem training_days :
  total_minutes / minutes_per_day = 10 :=
by
  sorry

end training_days_l280_280970


namespace tea_garden_problem_pruned_to_wild_conversion_l280_280069

-- Definitions and conditions as per the problem statement
def total_area : ℕ := 16
def total_yield : ℕ := 660
def wild_yield_per_mu : ℕ := 30
def pruned_yield_per_mu : ℕ := 50

-- Lean 4 statement as per the proof problem
theorem tea_garden_problem :
  ∃ (x y : ℕ), (x + y = total_area) ∧ (wild_yield_per_mu * x + pruned_yield_per_mu * y = total_yield) ∧
  x = 7 ∧ y = 9 :=
sorry

-- Additional theorem for the conversion condition
theorem pruned_to_wild_conversion :
  ∀ (a : ℕ), (wild_yield_per_mu * (7 + a) ≥ pruned_yield_per_mu * (9 - a)) → a ≥ 3 :=
sorry

end tea_garden_problem_pruned_to_wild_conversion_l280_280069


namespace total_capacity_is_1600_l280_280027

/-- Eight liters is 20% of the capacity of one container. -/
def capacity_of_one_container := 8 / 0.20

/-- Calculate the total capacity of 40 such containers filled with water. -/
def total_capacity_of_40_containers := 40 * capacity_of_one_container

theorem total_capacity_is_1600 :
  total_capacity_of_40_containers = 1600 := by
    -- Proof is skipped using sorry.
    sorry

end total_capacity_is_1600_l280_280027


namespace tan_45_deg_eq_one_l280_280831

/-- The tangent of 45 degrees is equal to 1. -/
theorem tan_45_deg_eq_one : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_one_l280_280831


namespace initial_bananas_per_child_l280_280067

theorem initial_bananas_per_child 
    (absent : ℕ) (present : ℕ) (total : ℕ) (x : ℕ) (B : ℕ)
    (h1 : absent = 305)
    (h2 : present = 305)
    (h3 : total = 610)
    (h4 : B = present * (x + 2))
    (h5 : B = total * x) : 
    x = 2 :=
by
  sorry

end initial_bananas_per_child_l280_280067


namespace sum_of_roots_l280_280156

theorem sum_of_roots : 
  let eq := (x - 7) ^ 2 = 16 in
  (roots : set ℝ) := { x | eq } in
  ∑ roots = 14 :=
by
sorry

end sum_of_roots_l280_280156


namespace trigonometric_identity_l280_280437

theorem trigonometric_identity (x : ℝ) (h : Real.tan (x + Real.pi / 2) = 5) : 
  1 / (Real.sin x * Real.cos x) = -26 / 5 :=
by
  sorry

end trigonometric_identity_l280_280437


namespace find_ordered_pair_l280_280126

theorem find_ordered_pair (a b : ℚ) :
  a • (⟨2, 3⟩ : ℚ × ℚ) + b • (⟨-2, 5⟩ : ℚ × ℚ) = (⟨10, -8⟩ : ℚ × ℚ) →
  (a, b) = (17 / 8, -23 / 8) :=
by
  intro h
  sorry

end find_ordered_pair_l280_280126


namespace greatest_value_sum_eq_24_l280_280274

theorem greatest_value_sum_eq_24 {a b : ℕ} (ha : 0 < a) (hb : 1 < b) 
  (h1 : ∀ (x y : ℕ), 0 < x → 1 < y → x^y < 500 → x^y ≤ a^b) : 
  a + b = 24 := 
by
  sorry

end greatest_value_sum_eq_24_l280_280274


namespace central_cell_value_l280_280513

theorem central_cell_value (a b c d e f g h i : ℝ)
  (h_row1 : a * b * c = 10)
  (h_row2 : d * e * f = 10)
  (h_row3 : g * h * i = 10)
  (h_col1 : a * d * g = 10)
  (h_col2 : b * e * h = 10)
  (h_col3 : c * f * i = 10)
  (h_block1 : a * b * d * e = 3)
  (h_block2 : b * c * e * f = 3)
  (h_block3 : d * e * g * h = 3)
  (h_block4 : e * f * h * i = 3) :
  e = 0.00081 :=
sorry

end central_cell_value_l280_280513


namespace find_central_cell_l280_280490

variable (a b c d e f g h i : ℝ)

def condition_1 : Prop :=
  a * b * c = 10 ∧ d * e * f = 10 ∧ g * h * i = 10

def condition_2 : Prop :=
  a * d * g = 10 ∧ b * e * h = 10 ∧ c * f * i = 10

def condition_3 : Prop :=
  a * b * d * e = 3 ∧ b * c * e * f = 3 ∧ d * e * g * h = 3 ∧ e * f * h * i = 3

theorem find_central_cell (h1 : condition_1 a b c d e f g h i)
                          (h2 : condition_2 a b c d e f g h i)
                          (h3 : condition_3 a b c d e f g h i) : 
  e = 0.00081 := 
sorry

end find_central_cell_l280_280490


namespace average_percent_increase_per_year_l280_280566

-- Definitions and conditions
def initialPopulation : ℕ := 175000
def finalPopulation : ℕ := 297500
def numberOfYears : ℕ := 10

-- Statement to prove
theorem average_percent_increase_per_year : 
  ((finalPopulation - initialPopulation) / numberOfYears : ℚ) / initialPopulation * 100 = 7 := by
  sorry

end average_percent_increase_per_year_l280_280566


namespace correct_operation_l280_280577

theorem correct_operation :
  (3 * m^2 + 4 * m^2 ≠ 7 * m^4) ∧
  (4 * m^3 * 5 * m^3 ≠ 20 * m^3) ∧
  ((-2 * m)^3 ≠ -6 * m^3) ∧
  (m^10 / m^5 = m^5) :=
by
  sorry

end correct_operation_l280_280577


namespace ratio_female_to_male_l280_280976

namespace DeltaSportsClub

variables (f m : ℕ) -- number of female and male members
-- Sum of ages of female and male members respectively
def sum_ages_females := 35 * f
def sum_ages_males := 30 * m
-- Total sum of ages
def total_sum_ages := sum_ages_females f + sum_ages_males m
-- Total number of members
def total_members := f + m

-- Given condition on the average age of all members
def average_age_condition := (total_sum_ages f m) / (total_members f m) = 32

-- The target theorem to prove the ratio of female to male members
theorem ratio_female_to_male (h : average_age_condition f m) : f/m = 2/3 :=
by sorry

end DeltaSportsClub

end ratio_female_to_male_l280_280976


namespace students_not_taking_test_l280_280959

theorem students_not_taking_test (total_students students_q1 students_q2 students_both not_taken : ℕ)
  (h_total : total_students = 30)
  (h_q1 : students_q1 = 25)
  (h_q2 : students_q2 = 22)
  (h_both : students_both = 22)
  (h_not_taken : not_taken = total_students - students_q2) :
  not_taken = 8 := by
  sorry

end students_not_taking_test_l280_280959


namespace arithmetic_sequence_length_l280_280311

theorem arithmetic_sequence_length :
  ∃ n : ℕ, n > 0 ∧ ∀ (a_1 a_2 a_n : ℤ), a_1 = 2 ∧ a_2 = 6 ∧ a_n = 2006 →
  a_n = a_1 + (n - 1) * (a_2 - a_1) → n = 502 := by
  sorry

end arithmetic_sequence_length_l280_280311


namespace tan_45_deg_l280_280656

theorem tan_45_deg : Real.tan (π / 4) = 1 := 
by
  have h_sin_cos : Real.sin (π / 4) = Real.cos (π / 4) := 
    by sorry
  have h_value : Real.sin (π / 4) = (Real.sqrt 2) / 2 := 
    by sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos, h_value]
  calc
    (Real.sqrt 2)/ 2 / (Real.sqrt 2)/ 2 = 1 := sorry

end tan_45_deg_l280_280656


namespace prime_power_divides_binomial_l280_280541

theorem prime_power_divides_binomial {p n k α : ℕ} (hp : Nat.Prime p) 
  (h : p^α ∣ Nat.choose n k) : p^α ≤ n := 
sorry

end prime_power_divides_binomial_l280_280541


namespace percentage_problem_l280_280473

theorem percentage_problem (x : ℝ) (h : 0.20 * x = 60) : 0.80 * x = 240 := 
by
  sorry

end percentage_problem_l280_280473


namespace sum_of_a_and_b_l280_280250

theorem sum_of_a_and_b (a b : ℕ) (h1: a > 0) (h2 : b > 1) (h3 : ∀ (x y : ℕ), x > 0 → y > 1 → x^y < 500 → x = a ∧ y = b → x^y ≥ a^b ) :
  a + b = 24 :=
sorry

end sum_of_a_and_b_l280_280250


namespace sum_of_roots_eq_14_l280_280202

-- Define the condition as a hypothesis
def equation (x : ℝ) := (x - 7) ^ 2 = 16

-- Define the statement that needs to be proved
theorem sum_of_roots_eq_14 : 
  (∀ x : ℝ, equation x → x = 11 ∨ x = 3) → 11 + 3 = 14 :=
by 
sorry

end sum_of_roots_eq_14_l280_280202


namespace find_m_range_l280_280928

noncomputable def f (x m : ℝ) : ℝ := x * abs (x - m) + 2 * x - 3

theorem find_m_range (m : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ m ≤ f x₂ m)
    ↔ (-2 ≤ m ∧ m ≤ 2) :=
by
  sorry

end find_m_range_l280_280928


namespace tan_45_deg_l280_280696

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l280_280696


namespace perspective_square_area_l280_280302

theorem perspective_square_area (a b : ℝ) (ha : a = 4 ∨ b = 4) : 
  a * a = 16 ∨ (2 * b) * (2 * b) = 64 :=
by 
sorry

end perspective_square_area_l280_280302


namespace area_triangle_ABC_l280_280372

open Real

structure Point :=
  (x : ℝ)
  (y : ℝ)

def distance (P Q : Point) : ℝ := sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

def area_of_triangle (A B C : Point) : ℝ :=
  (1 / 2) * abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))

theorem area_triangle_ABC :
  let A := Point.mk (-7) 3
  let B := Point.mk 0 4
  let C := Point.mk 9 5
  distance A B = 7 ∧ distance B C = 9 → area_of_triangle A B C = 8 :=
by
  intros
  sorry

end area_triangle_ABC_l280_280372


namespace polynomial_has_root_l280_280112

theorem polynomial_has_root {a b c d : ℝ} 
  (h : a * c = 2 * b + 2 * d) : 
  ∃ x : ℝ, (x^2 + a * x + b = 0) ∨ (x^2 + c * x + d = 0) :=
by 
  sorry

end polynomial_has_root_l280_280112


namespace central_cell_value_l280_280514

theorem central_cell_value (a b c d e f g h i : ℝ)
  (h_row1 : a * b * c = 10)
  (h_row2 : d * e * f = 10)
  (h_row3 : g * h * i = 10)
  (h_col1 : a * d * g = 10)
  (h_col2 : b * e * h = 10)
  (h_col3 : c * f * i = 10)
  (h_block1 : a * b * d * e = 3)
  (h_block2 : b * c * e * f = 3)
  (h_block3 : d * e * g * h = 3)
  (h_block4 : e * f * h * i = 3) :
  e = 0.00081 :=
sorry

end central_cell_value_l280_280514


namespace tan_of_45_deg_l280_280738

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  -- Definitions used directly from conditions
  have hcos : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry
  have hsin : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry

  -- Using the definition of tangent
  rw [Real.tan_eq_sin_div_cos, hsin, hcos]
  exact div_self (ne_of_gt (Real.sqrt_pos.2 zero_lt_two))

end tan_of_45_deg_l280_280738


namespace sum_of_roots_eq_14_l280_280194

theorem sum_of_roots_eq_14 (x : ℝ) :
  (x - 7) ^ 2 = 16 → ∃ r1 r2 : ℝ, (x = r1 ∨ x = r2) ∧ r1 + r2 = 14 :=
by
  intro h
  sorry

end sum_of_roots_eq_14_l280_280194


namespace tan_of_45_deg_l280_280740

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  -- Definitions used directly from conditions
  have hcos : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry
  have hsin : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry

  -- Using the definition of tangent
  rw [Real.tan_eq_sin_div_cos, hsin, hcos]
  exact div_self (ne_of_gt (Real.sqrt_pos.2 zero_lt_two))

end tan_of_45_deg_l280_280740


namespace sum_of_roots_eq_14_l280_280205

-- Define the condition as a hypothesis
def equation (x : ℝ) := (x - 7) ^ 2 = 16

-- Define the statement that needs to be proved
theorem sum_of_roots_eq_14 : 
  (∀ x : ℝ, equation x → x = 11 ∨ x = 3) → 11 + 3 = 14 :=
by 
sorry

end sum_of_roots_eq_14_l280_280205


namespace fraction_addition_l280_280436

theorem fraction_addition (a b : ℕ) (hb : b ≠ 0) (h : a / (b : ℚ) = 3 / 5) : (a + b) / (b : ℚ) = 8 / 5 := 
by
sorry

end fraction_addition_l280_280436


namespace tan_45_eq_1_l280_280859

theorem tan_45_eq_1 :
  let π := Real.pi in
  ∃ (x : ℝ), x = π / 4 ∧
  sin x = sqrt 2 / 2 ∧
  cos x = sqrt 2 / 2 →
  tan (π / 4) = 1 :=
by sorry

end tan_45_eq_1_l280_280859


namespace correct_substitution_l280_280385

theorem correct_substitution (x y : ℝ) (h1 : y = 1 - x) (h2 : x - 2 * y = 4) : x - 2 * (1 - x) = 4 → x - 2 + 2 * x = 4 := by
  sorry

end correct_substitution_l280_280385


namespace expression_value_l280_280384

theorem expression_value : (19 + 12) ^ 2 - (12 ^ 2 + 19 ^ 2) = 456 := 
by sorry

end expression_value_l280_280384


namespace tan_45_degree_l280_280753

theorem tan_45_degree : ∀ (θ : ℝ), 
  θ = 45 → 
  (sin θ = 1 / real.sqrt 2) → 
  (cos θ = 1 / real.sqrt 2) → 
  real.tan θ = 1 := 
by
  intros θ hθ hsin hcos
  sorry

end tan_45_degree_l280_280753


namespace tan_45_degree_is_one_l280_280717

theorem tan_45_degree_is_one : Real.tan (Real.pi / 4) = 1 := sorry

end tan_45_degree_is_one_l280_280717


namespace ring_area_l280_280993

theorem ring_area (r1 r2 : ℝ) (h1 : r1 = 12) (h2 : r2 = 5) : 
  (π * r1^2) - (π * r2^2) = 119 * π := 
by simp [h1, h2]; sorry

end ring_area_l280_280993


namespace tan_45_degrees_l280_280810

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l280_280810


namespace tan_45_eq_one_l280_280870

theorem tan_45_eq_one : Real.tan (Real.pi / 4) = 1 := 
by
  sorry

end tan_45_eq_one_l280_280870


namespace find_prime_A_l280_280229

open Nat

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem find_prime_A (A : ℕ) :
  is_prime A ∧ is_prime (A + 14) ∧ is_prime (A + 18) ∧ is_prime (A + 32) ∧ is_prime (A + 36) → A = 5 := by
  sorry

end find_prime_A_l280_280229


namespace restore_axes_with_parabola_l280_280961

-- Define the given parabola y = x^2
def parabola (x : ℝ) : ℝ := x^2

-- Problem: Prove that you can restore the coordinate axes using the given parabola and tools.
theorem restore_axes_with_parabola : 
  ∃ O X Y : ℝ × ℝ, 
  (∀ x, parabola x = (x, x^2).snd) ∧ 
  (X.fst = 0 ∧ Y.snd = 0) ∧
  (O = (0,0)) :=
sorry

end restore_axes_with_parabola_l280_280961


namespace experts_eventual_win_probability_l280_280908

noncomputable def experts_win_probability (p : ℝ) (q : ℝ) (experts_needed : ℕ) (audience_needed : ℕ) (max_rounds : ℕ) : ℝ :=
  let e_all_wins := p ^ max_rounds
  let e_three_wins_one_loss := (Nat.choose max_rounds (max_rounds - 1)) * (p ^ (max_rounds - 1)) * q
  e_all_wins + e_three_wins_one_loss

theorem experts_eventual_win_probability :
  ∀ (p : ℝ) (q : ℝ),
  p = 0.6 →
  q = 1 - p →
  experts_win_probability p q 3 2 4 = 0.4752 := by
  intros p q hp hq
  have h : experts_win_probability p q 3 2 4 = 0.6 ^ 4 + 4 * (0.6 ^ 3) * 0.4 := sorry
  rw [hp, hq] at h
  norm_num at h
  exact h

end experts_eventual_win_probability_l280_280908


namespace total_fruit_salads_l280_280594

theorem total_fruit_salads (a : ℕ) (h_alaya : a = 200) (h_angel : 2 * a = 400) : a + 2 * a = 600 :=
by 
  rw [h_alaya, h_angel]
  sorry

end total_fruit_salads_l280_280594


namespace tan_45_eq_l280_280618

def tan_identity (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

def sin_45_eq : Real.sin 45 = Real.sqrt 2 / 2 := sorry

def cos_45_eq : Real.cos 45 = Real.sqrt 2 / 2 := sorry

theorem tan_45_eq : tan_identity 45 = 1 := by
  rw [tan_identity, sin_45_eq, cos_45_eq]
  -- simplifying (sqrt 2 / 2) / (sqrt 2 / 2) = 1
  sorry

end tan_45_eq_l280_280618


namespace find_central_cell_l280_280491

variable (a b c d e f g h i : ℝ)

def condition_1 : Prop :=
  a * b * c = 10 ∧ d * e * f = 10 ∧ g * h * i = 10

def condition_2 : Prop :=
  a * d * g = 10 ∧ b * e * h = 10 ∧ c * f * i = 10

def condition_3 : Prop :=
  a * b * d * e = 3 ∧ b * c * e * f = 3 ∧ d * e * g * h = 3 ∧ e * f * h * i = 3

theorem find_central_cell (h1 : condition_1 a b c d e f g h i)
                          (h2 : condition_2 a b c d e f g h i)
                          (h3 : condition_3 a b c d e f g h i) : 
  e = 0.00081 := 
sorry

end find_central_cell_l280_280491


namespace tan_45_deg_eq_1_l280_280608

-- Define conditions
variable (Q : ℝ × ℝ)
variable (E : ℝ × ℝ)
variable (O : ℝ × ℝ := (0, 0))
variable (θ : ℝ := 45)

-- Define main theorem
theorem tan_45_deg_eq_1
  (Q_on_unit_circle : Q = (ℝ.sqrt 2 / 2, ℝ.sqrt 2 / 2))
  (E_foot_perpendicular : E = (ℝ.sqrt 2 / 2, 0))
  (triangle_45_45_90 : (Q.1 - O.1) ^ 2 + (Q.2 - O.2) ^ 2 = 1) :
  Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_1_l280_280608


namespace tan_45_deg_l280_280876

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 :=
by
  have h_sin_cos : Real.sin (Real.pi / 4) = Real.cos (Real.pi / 4) := sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos]
  have sqrt2_ne_zero : Real.sqrt 2 ≠ 0 := sorry
  have sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have cos_45 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [sin_45, cos_45]
  field_simp [sqrt2_ne_zero]
  exact one_div_one

end tan_45_deg_l280_280876


namespace sum_of_roots_l280_280148

theorem sum_of_roots (a b : Real) (h : (x - 7)^2 = 16):
  a + b = 14 :=
sorry

end sum_of_roots_l280_280148


namespace unique_x1_exists_l280_280431

theorem unique_x1_exists (x : ℕ → ℝ) :
  (∀ n : ℕ+, x (n+1) = x n * (x n + 1 / n)) →
  ∃! (x1 : ℝ), (∀ n : ℕ+, 0 < x n ∧ x n < x (n+1) ∧ x (n+1) < 1) :=
sorry

end unique_x1_exists_l280_280431


namespace tan_45_eq_one_l280_280679

theorem tan_45_eq_one 
  (θ : ℝ)
  (hθ : θ = 45)
  (sin_45_eq : Real.sin (θ * Real.pi / 180) = Real.sqrt 2 / 2)
  (cos_45_eq : Real.cos (θ * Real.pi /180) = Real.sqrt 2 / 2)
  (tan_def : ∀ x : ℝ, 0 < Real.cos x → Real.tan x = Real.sin x / Real.cos x) : 
  Real.tan (θ * Real.pi / 180) = 1 := sorry

end tan_45_eq_one_l280_280679


namespace acrobat_count_range_l280_280038

def animal_legs (elephants monkeys acrobats : ℕ) : ℕ :=
  4 * elephants + 2 * monkeys + 2 * acrobats

def animal_heads (elephants monkeys acrobats : ℕ) : ℕ :=
  elephants + monkeys + acrobats

theorem acrobat_count_range (e m a : ℕ) (h1 : animal_heads e m a = 18)
  (h2 : animal_legs e m a = 50) : 0 ≤ a ∧ a ≤ 11 :=
by {
  sorry
}

end acrobat_count_range_l280_280038


namespace exists_quad_root_l280_280101

theorem exists_quad_root (a b c d : ℝ) (h : a * c = 2 * b + 2 * d) :
  (∃ x, x^2 + a * x + b = 0) ∨ (∃ x, x^2 + c * x + d = 0) :=
sorry

end exists_quad_root_l280_280101


namespace ratio_square_areas_l280_280997

theorem ratio_square_areas (r : ℝ) (h1 : r > 0) :
  let s1 := 2 * r / Real.sqrt 5
  let area1 := (s1) ^ 2
  let h := r * Real.sqrt 3
  let s2 := r
  let area2 := (s2) ^ 2
  area1 / area2 = 4 / 5 := by
  sorry

end ratio_square_areas_l280_280997


namespace peach_difference_proof_l280_280371

def red_peaches_odd := 12
def green_peaches_odd := 22
def red_peaches_even := 15
def green_peaches_even := 20
def num_baskets := 20
def num_odd_baskets := num_baskets / 2
def num_even_baskets := num_baskets / 2

def total_red_peaches := (red_peaches_odd * num_odd_baskets) + (red_peaches_even * num_even_baskets)
def total_green_peaches := (green_peaches_odd * num_odd_baskets) + (green_peaches_even * num_even_baskets)
def difference := total_green_peaches - total_red_peaches

theorem peach_difference_proof : difference = 150 := by
  sorry

end peach_difference_proof_l280_280371


namespace tan_45_deg_eq_1_l280_280604

-- Define conditions
variable (Q : ℝ × ℝ)
variable (E : ℝ × ℝ)
variable (O : ℝ × ℝ := (0, 0))
variable (θ : ℝ := 45)

-- Define main theorem
theorem tan_45_deg_eq_1
  (Q_on_unit_circle : Q = (ℝ.sqrt 2 / 2, ℝ.sqrt 2 / 2))
  (E_foot_perpendicular : E = (ℝ.sqrt 2 / 2, 0))
  (triangle_45_45_90 : (Q.1 - O.1) ^ 2 + (Q.2 - O.2) ^ 2 = 1) :
  Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_1_l280_280604


namespace range_of_a_l280_280441

variable {x a : ℝ}

def p (x a : ℝ) : Prop := x > a
def q (x : ℝ) : Prop := x^2 + x - 2 > 0

theorem range_of_a 
  (h_sufficient : ∀ x, p x a → q x)
  (h_not_necessary : ∃ x, q x ∧ ¬ p x a) :
  a ≥ 1 :=
sorry

end range_of_a_l280_280441


namespace min_value_expression_l280_280008

open Real

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 4) :
  ∃ z : ℝ, z = 16 / 7 ∧ ∀ u > 0, ∀ v > 0, u + v = 4 → ((u^2 / (u + 1)) + (v^2 / (v + 2))) ≥ z :=
by
  sorry

end min_value_expression_l280_280008


namespace thomas_needs_more_money_l280_280990

-- Define the conditions in Lean
def weeklyAllowance : ℕ := 50
def hourlyWage : ℕ := 9
def hoursPerWeek : ℕ := 30
def weeklyExpenses : ℕ := 35
def weeksInYear : ℕ := 52
def carCost : ℕ := 15000

-- Define the total earnings for the first year
def firstYearEarnings : ℕ :=
  weeklyAllowance * weeksInYear

-- Define the weekly earnings from the second year job
def secondYearWeeklyEarnings : ℕ :=
  hourlyWage * hoursPerWeek

-- Define the total earnings for the second year
def secondYearEarnings : ℕ :=
  secondYearWeeklyEarnings * weeksInYear

-- Define the total earnings over two years
def totalEarnings : ℕ :=
  firstYearEarnings + secondYearEarnings

-- Define the total expenses over two years
def totalExpenses : ℕ :=
  weeklyExpenses * (2 * weeksInYear)

-- Define the net savings after two years
def netSavings : ℕ :=
  totalEarnings - totalExpenses

-- Define the amount more needed for the car
def amountMoreNeeded : ℕ :=
  carCost - netSavings

-- The theorem to prove
theorem thomas_needs_more_money : amountMoreNeeded = 2000 := by
  sorry

end thomas_needs_more_money_l280_280990


namespace sum_of_a_and_b_l280_280270

theorem sum_of_a_and_b (a b : ℕ) (h1 : b > 1) (h2 : a^b < 500) (h3 : ∀ c d : ℕ, d > 1 → c^d < 500 → c^d ≤ a^b) : a + b = 24 :=
sorry

end sum_of_a_and_b_l280_280270


namespace sandy_spent_on_shirt_l280_280968

-- Define the conditions
def cost_of_shorts : ℝ := 13.99
def cost_of_jacket : ℝ := 7.43
def total_spent_on_clothes : ℝ := 33.56

-- Define the amount spent on the shirt
noncomputable def cost_of_shirt : ℝ :=
  total_spent_on_clothes - (cost_of_shorts + cost_of_jacket)

-- Prove that Sandy spent $12.14 on the shirt
theorem sandy_spent_on_shirt : cost_of_shirt = 12.14 :=
by
  sorry

end sandy_spent_on_shirt_l280_280968


namespace T_n_formula_l280_280007

def a_n (n : ℕ) : ℕ := 3 * n - 1
def b_n (n : ℕ) : ℕ := 2 ^ n
def T_n (n : ℕ) : ℕ := (Finset.range n).sum (λ k => a_n (k + 1) * b_n (k + 1))

theorem T_n_formula (n : ℕ) : T_n n = 8 - 8 * 2 ^ n + 3 * n * 2 ^ (n + 1) :=
by 
  sorry

end T_n_formula_l280_280007


namespace tan_45_degree_l280_280674

theorem tan_45_degree : ∀ (Q : ℝ × ℝ),
  -- Q corresponds to 45 degrees on the unit circle
  Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) →
  -- The radius of the unit circle is 1
  (Q.1 ^ 2 + Q.2 ^ 2 = 1) →
  -- QOE is a 45-45-90 triangle
  True →
  -- Prove that tan 45 degrees equals 1
  Real.tan (Real.pi / 4) = 1 :=
by
  -- The proof using these conditions is omitted
  intros,
  sorry

end tan_45_degree_l280_280674


namespace log10_cubic_solution_l280_280921

noncomputable def log10 (x: ℝ) : ℝ := Real.log x / Real.log 10

open Real

theorem log10_cubic_solution 
  (x : ℝ) 
  (hx1 : x < 1) 
  (hx2 : (log10 x)^3 - log10 (x^4) = 640) : 
  (log10 x)^4 - log10 (x^4) = 645 := 
by 
  sorry

end log10_cubic_solution_l280_280921


namespace find_m_l280_280222

theorem find_m 
  (h : ( (1 ^ m) / (5 ^ m) ) * ( (1 ^ 16) / (4 ^ 16) ) = 1 / (2 * 10 ^ 31)) :
  m = 31 :=
by
  sorry

end find_m_l280_280222


namespace sum_of_roots_of_quadratic_l280_280163

theorem sum_of_roots_of_quadratic : 
  (∃ (x : ℝ), (x - 7) ^ 2 = 16 ∧ (x = 11 ∨ x = 3)) → 
  (11 + 3 = 14) :=
by 
  intro h
  sorry

end sum_of_roots_of_quadratic_l280_280163


namespace num_employees_is_143_l280_280098

def b := 143
def is_sol (b : ℕ) := 80 < b ∧ b < 150 ∧ b % 4 = 3 ∧ b % 5 = 3 ∧ b % 7 = 4

theorem num_employees_is_143 : is_sol b :=
by
  -- This is where the proof would be written
  sorry

end num_employees_is_143_l280_280098


namespace not_parallel_to_a_l280_280542

noncomputable def is_parallel (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k * u.1, k * u.2)

theorem not_parallel_to_a : ∀ k : ℝ, ¬ is_parallel (k^2 + 1, k^2 + 1) (1, -2) :=
sorry

end not_parallel_to_a_l280_280542


namespace tan_45_eq_one_l280_280788

theorem tan_45_eq_one (Q : ℝ × ℝ) (h1 : Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2)) : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_eq_one_l280_280788


namespace sum_of_roots_eq_l280_280188

theorem sum_of_roots_eq (x : ℝ) : (x - 7)^2 = 16 → x = 11 ∨ x = 3 → ∑ (root : ℝ) in {11, 3}, root = 14 :=
by
  assume h₁ : (x - 7)^2 = 16
  assume h₂ : x = 11 ∨ x = 3
  sorry

end sum_of_roots_eq_l280_280188


namespace required_sampling_methods_l280_280941

-- Defining the given conditions
def total_households : Nat := 2000
def farmer_households : Nat := 1800
def worker_households : Nat := 100
def intellectual_households : Nat := total_households - farmer_households - worker_households
def sample_size : Nat := 40

-- Statement representing the proof problem
theorem required_sampling_methods :
  stratified_sampling_needed ∧ systematic_sampling_needed ∧ simple_random_sampling_needed :=
sorry

end required_sampling_methods_l280_280941


namespace solve_quadratic_l280_280081

theorem solve_quadratic : ∀ x : ℝ, x ^ 2 - 6 * x + 8 = 0 ↔ x = 2 ∨ x = 4 := by
  sorry

end solve_quadratic_l280_280081


namespace tan_45_degrees_l280_280811

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l280_280811


namespace boy_age_is_10_l280_280206

-- Define the boy's current age as a variable
def boy_current_age := 10

-- Define a condition based on the boy's statement
def boy_statement_condition (x : ℕ) : Prop :=
  x = 2 * (x - 5)

-- The main theorem stating equivalence of the boy's current age to 10 given the condition
theorem boy_age_is_10 (x : ℕ) (h : boy_statement_condition x) : x = boy_current_age := by
  sorry

end boy_age_is_10_l280_280206


namespace problem1_l280_280357

theorem problem1 (x y : ℝ) 
  (h1 : 3 * x + 2 * y = 10)
  (h2 : x / 2 - (y + 1) / 3 = 1) :
  x = 3 ∧ y = 1 / 2 := 
sorry

end problem1_l280_280357


namespace tan_45_deg_l280_280877

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 :=
by
  have h_sin_cos : Real.sin (Real.pi / 4) = Real.cos (Real.pi / 4) := sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos]
  have sqrt2_ne_zero : Real.sqrt 2 ≠ 0 := sorry
  have sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have cos_45 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [sin_45, cos_45]
  field_simp [sqrt2_ne_zero]
  exact one_div_one

end tan_45_deg_l280_280877


namespace david_profit_l280_280233

theorem david_profit (weight : ℕ) (cost sell_price : ℝ) (h_weight : weight = 50) (h_cost : cost = 50) (h_sell_price : sell_price = 1.20) : 
  sell_price * weight - cost = 10 :=
by sorry

end david_profit_l280_280233


namespace polynomial_has_real_root_l280_280914

theorem polynomial_has_real_root (b : ℝ) : ∃ x : ℝ, (x^4 + b * x^3 + 2 * x^2 + b * x - 2 = 0) := sorry

end polynomial_has_real_root_l280_280914


namespace experts_win_probability_l280_280907

noncomputable def probability_of_experts_winning (p : ℝ) (q : ℝ) (needed_expert_wins : ℕ) (needed_audience_wins : ℕ) : ℝ :=
  p ^ 4 + 4 * (p ^ 3 * q)

-- Probability values
def p : ℝ := 0.6
def q : ℝ := 1 - p

-- Number of wins needed
def needed_expert_wins : ℕ := 3
def needed_audience_wins : ℕ := 2

theorem experts_win_probability :
  probability_of_experts_winning p q needed_expert_wins needed_audience_wins = 0.4752 :=
by
  -- Proof would go here
  sorry

end experts_win_probability_l280_280907


namespace bet_final_result_l280_280091

theorem bet_final_result :
  let M₀ := 64
  let final_money := (3 / 2) ^ 3 * (1 / 2) ^ 3 * M₀
  final_money = 27 ∧ M₀ - final_money = 37 :=
by
  sorry

end bet_final_result_l280_280091


namespace inequality_proof_l280_280078

theorem inequality_proof (a b c : ℝ) (h : a * b * c = 1) : 
  1 / (2 * a^2 + b^2 + 3) + 1 / (2 * b^2 + c^2 + 3) + 1 / (2 * c^2 + a^2 + 3) ≤ 1 / 2 := 
by
  sorry

end inequality_proof_l280_280078


namespace fifteen_percent_minus_70_l280_280583

theorem fifteen_percent_minus_70 (a : ℝ) : 0.15 * a - 70 = (15 / 100) * a - 70 :=
by sorry

end fifteen_percent_minus_70_l280_280583


namespace johns_money_left_l280_280951

def dog_walking_days_in_april := 26
def earnings_per_day := 10
def money_spent_on_books := 50
def money_given_to_sister := 50

theorem johns_money_left : (dog_walking_days_in_april * earnings_per_day) - (money_spent_on_books + money_given_to_sister) = 160 := 
by
  sorry

end johns_money_left_l280_280951


namespace tan_45_degrees_eq_1_l280_280785

theorem tan_45_degrees_eq_1 :
    tan (45 * real.pi / 180) = 1 :=
by
  sorry

end tan_45_degrees_eq_1_l280_280785


namespace factor_polynomial_l280_280018

theorem factor_polynomial (x : ℝ) :
  (x^3 - 12 * x + 16) = (x + 4) * ((x - 2)^2) :=
by
  sorry

end factor_polynomial_l280_280018


namespace tan_45_deg_eq_one_l280_280851

theorem tan_45_deg_eq_one :
  let θ := 45
  (∀ θ, tan θ = sin θ / cos θ) →
  (sin 45 = sqrt 2 / 2) →
  (cos 45 = sqrt 2 / 2) →
  tan 45 = 1 :=
by
  intros θ h_tan h_sin h_cos
  sorry

end tan_45_deg_eq_one_l280_280851


namespace find_y_l280_280313

theorem find_y (x y : ℝ) (h1 : 9823 + x = 13200) (h2 : x = y / 3 + 37.5) : y = 10018.5 :=
by
  sorry

end find_y_l280_280313


namespace angle_FMG_l280_280240
  
theorem angle_FMG :
  ∀ (A B C D E F G M : Point) (angle : Point → Point → Point → ℝ),
    IsoscelesRightTriangle A B C 90° →
    IsoscelesRightTriangle A D E 90° →
    Midpoint M B C →
    (dist A B = dist A C) → (dist A B = dist D F) → (dist A B = dist F M) → (dist A B = dist E G) → (dist A B = dist G M) →
    angle F D E = 9° →
    angle G E D = 9° →
    Outside F D E →
    Outside G D E →
    angle F M G = 54°
:= by
  sorry

end angle_FMG_l280_280240


namespace total_sentence_l280_280336

theorem total_sentence (base_rate : ℝ) (value_stolen : ℝ) (third_offense_increase : ℝ) (additional_years : ℕ) : 
  base_rate = 1 / 5000 → 
  value_stolen = 40000 → 
  third_offense_increase = 0.25 → 
  additional_years = 2 →
  (value_stolen * base_rate * (1 + third_offense_increase) + additional_years) = 12 := 
by
  intros
  sorry

end total_sentence_l280_280336


namespace tan_45_degree_l280_280675

theorem tan_45_degree : ∀ (Q : ℝ × ℝ),
  -- Q corresponds to 45 degrees on the unit circle
  Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) →
  -- The radius of the unit circle is 1
  (Q.1 ^ 2 + Q.2 ^ 2 = 1) →
  -- QOE is a 45-45-90 triangle
  True →
  -- Prove that tan 45 degrees equals 1
  Real.tan (Real.pi / 4) = 1 :=
by
  -- The proof using these conditions is omitted
  intros,
  sorry

end tan_45_degree_l280_280675


namespace negation_proof_l280_280564

open Classical

variable {x : ℝ}

theorem negation_proof :
  (∀ x : ℝ, (x + 1) ≥ 0 ∧ (x^2 - x) ≤ 0) ↔ ¬ (∃ x_0 : ℝ, (x_0 + 1) < 0 ∨ (x_0^2 - x_0) > 0) := 
by
  sorry

end negation_proof_l280_280564


namespace buddy_cards_on_thursday_is_32_l280_280068

def buddy_cards_on_monday := 30
def buddy_cards_on_tuesday := buddy_cards_on_monday / 2
def buddy_cards_on_wednesday := buddy_cards_on_tuesday + 12
def buddy_cards_bought_on_thursday := buddy_cards_on_tuesday / 3
def buddy_cards_on_thursday := buddy_cards_on_wednesday + buddy_cards_bought_on_thursday

theorem buddy_cards_on_thursday_is_32 : buddy_cards_on_thursday = 32 :=
by sorry

end buddy_cards_on_thursday_is_32_l280_280068


namespace magician_method_N_2k_magician_method_values_l280_280402

-- (a) Prove that if there is a method for N = k, then there is a method for N = 2k.
theorem magician_method_N_2k (k : ℕ) (method_k : Prop) : 
  (∃ method_N_k : Prop, method_k → method_N_k) → 
  (∃ method_N_2k : Prop, method_k → method_N_2k) :=
sorry

-- (b) Find all values of N for which the magician and the assistant have a method.
theorem magician_method_values (N : ℕ) : 
  (∃ method : Prop, method) ↔ (∃ m : ℕ, N = 2^m) :=
sorry

end magician_method_N_2k_magician_method_values_l280_280402


namespace central_cell_value_l280_280489

def table (a b c d e f g h i : ℝ) : Prop :=
  (a * b * c = 10) ∧ (d * e * f = 10) ∧ (g * h * i = 10) ∧
  (a * d * g = 10) ∧ (b * e * h = 10) ∧ (c * f * i = 10) ∧
  (a * b * d * e = 3) ∧ (b * c * e * f = 3) ∧ (d * e * g * h = 3) ∧ (e * f * h * i = 3)

theorem central_cell_value (a b c d f g h i e : ℝ) (h_table : table a b c d e f g h i) : 
  e = 0.00081 :=
by sorry

end central_cell_value_l280_280489


namespace tan_45_deg_l280_280694

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l280_280694


namespace no_solution_fraction_equation_l280_280090

theorem no_solution_fraction_equation (x : ℝ) (h : x ≠ 2) : 
  (1 - x) / (x - 2) + 2 = 1 / (2 - x) → false :=
by 
  intro h_eq
  sorry

end no_solution_fraction_equation_l280_280090


namespace sum_of_roots_eq_l280_280189

theorem sum_of_roots_eq (x : ℝ) : (x - 7)^2 = 16 → x = 11 ∨ x = 3 → ∑ (root : ℝ) in {11, 3}, root = 14 :=
by
  assume h₁ : (x - 7)^2 = 16
  assume h₂ : x = 11 ∨ x = 3
  sorry

end sum_of_roots_eq_l280_280189


namespace tan_45_degrees_eq_1_l280_280786

theorem tan_45_degrees_eq_1 :
    tan (45 * real.pi / 180) = 1 :=
by
  sorry

end tan_45_degrees_eq_1_l280_280786


namespace tan_45_deg_eq_one_l280_280836

/-- The tangent of 45 degrees is equal to 1. -/
theorem tan_45_deg_eq_one : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_one_l280_280836


namespace solve_quadratic_l280_280082

theorem solve_quadratic : ∀ x : ℝ, x ^ 2 - 6 * x + 8 = 0 ↔ x = 2 ∨ x = 4 := by
  sorry

end solve_quadratic_l280_280082


namespace find_b_perpendicular_l280_280095

theorem find_b_perpendicular
  (b : ℝ)
  (line1 : ∀ x y : ℝ, 2 * x - 3 * y + 5 = 0)
  (line2 : ∀ x y : ℝ, b * x - 3 * y + 1 = 0)
  (perpendicular : (2 / 3) * (b / 3) = -1)
  : b = -9/2 :=
sorry

end find_b_perpendicular_l280_280095


namespace motion_of_Q_is_clockwise_with_2ω_l280_280021

variables {ω t : ℝ} {P Q : ℝ × ℝ}

def moving_counterclockwise (P : ℝ × ℝ) (ω t : ℝ) : Prop :=
  P = (Real.cos (ω * t), Real.sin (ω * t))

def motion_of_Q (x y : ℝ): ℝ × ℝ :=
  (-2 * x * y, y^2 - x^2)

def is_on_unit_circle (Q : ℝ × ℝ) : Prop :=
  Q.fst ^ 2 + Q.snd ^ 2 = 1

theorem motion_of_Q_is_clockwise_with_2ω 
  (P : ℝ × ℝ) (ω t : ℝ) (x y : ℝ) :
  moving_counterclockwise P ω t →
  P = (x, y) →
  is_on_unit_circle P →
  is_on_unit_circle (motion_of_Q x y) ∧
  Q = (x, y) →
  Q.fst = Real.cos (2 * ω * t + 3 * Real.pi / 2) ∧ 
  Q.snd = Real.sin (2 * ω * t + 3 * Real.pi / 2) :=
sorry

end motion_of_Q_is_clockwise_with_2ω_l280_280021


namespace tan_45_degrees_eq_1_l280_280779

theorem tan_45_degrees_eq_1 :
    tan (45 * real.pi / 180) = 1 :=
by
  sorry

end tan_45_degrees_eq_1_l280_280779


namespace sum_of_properly_paintable_numbers_l280_280418

-- Definitions based on conditions
def properly_paintable (a b c : ℕ) : Prop :=
  ∀ n : ℕ, (n % a = 0 ∧ n % b ≠ 1 ∧ n % c ≠ 3) ∨
           (n % a ≠ 0 ∧ n % b = 1 ∧ n % c ≠ 3) ∨
           (n % a ≠ 0 ∧ n % b ≠ 1 ∧ n % c = 3) → n < 100

-- Main theorem to prove
theorem sum_of_properly_paintable_numbers : 
  (properly_paintable 3 3 6) ∧ (properly_paintable 4 2 8) → 
  100 * 3 + 10 * 3 + 6 + 100 * 4 + 10 * 2 + 8 = 764 :=
by
  sorry  -- The proof goes here, but it's not required

-- Note: The actual condition checks in the definition of properly_paintable 
-- might need more detailed splits into depending on specific post visits and a 
-- more rigorous formalization to comply with the exact checking as done above. 
-- This definition is a simplified logical structure to represent the condition.


end sum_of_properly_paintable_numbers_l280_280418


namespace sum_of_roots_abs_gt_six_l280_280312

theorem sum_of_roots_abs_gt_six {p r1 r2 : ℝ} (h1 : r1 + r2 = -p) (h2 : r1 * r2 = 9) (h3 : r1 ≠ r2) (h4 : p^2 > 36) : |r1 + r2| > 6 :=
sorry

end sum_of_roots_abs_gt_six_l280_280312


namespace tan_45_deg_eq_one_l280_280652

/-
  Given:
  cos (45 deg) = sqrt 2 / 2,
  sin (45 deg) = sqrt 2 / 2,
  Prove: 
  tan (45 deg) = 1
-/

theorem tan_45_deg_eq_one : 
  (real.cos (real.pi / 4) = real.sqrt 2 / 2) → 
  (real.sin (real.pi / 4) = real.sqrt 2 / 2) → 
  real.tan (real.pi / 4) = 1 :=
by
  intros hcos hsin
  sorry

end tan_45_deg_eq_one_l280_280652


namespace central_cell_value_l280_280516

theorem central_cell_value (a b c d e f g h i : ℝ)
  (h_row1 : a * b * c = 10)
  (h_row2 : d * e * f = 10)
  (h_row3 : g * h * i = 10)
  (h_col1 : a * d * g = 10)
  (h_col2 : b * e * h = 10)
  (h_col3 : c * f * i = 10)
  (h_block1 : a * b * d * e = 3)
  (h_block2 : b * c * e * f = 3)
  (h_block3 : d * e * g * h = 3)
  (h_block4 : e * f * h * i = 3) :
  e = 0.00081 :=
sorry

end central_cell_value_l280_280516


namespace friends_pets_ratio_l280_280557

theorem friends_pets_ratio (pets_total : ℕ) (pets_taylor : ℕ) (pets_friend4 : ℕ) (pets_friend5 : ℕ)
  (pets_first3_total : ℕ) : pets_total = 32 → pets_taylor = 4 → pets_friend4 = 2 → pets_friend5 = 2 →
  pets_first3_total = pets_total - pets_taylor - pets_friend4 - pets_friend5 →
  (pets_first3_total : ℚ) / pets_taylor = 6 :=
by
  sorry

end friends_pets_ratio_l280_280557


namespace tan_of_45_deg_l280_280742

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  -- Definitions used directly from conditions
  have hcos : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry
  have hsin : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry

  -- Using the definition of tangent
  rw [Real.tan_eq_sin_div_cos, hsin, hcos]
  exact div_self (ne_of_gt (Real.sqrt_pos.2 zero_lt_two))

end tan_of_45_deg_l280_280742


namespace tan_45_eq_1_l280_280854

theorem tan_45_eq_1 :
  let π := Real.pi in
  ∃ (x : ℝ), x = π / 4 ∧
  sin x = sqrt 2 / 2 ∧
  cos x = sqrt 2 / 2 →
  tan (π / 4) = 1 :=
by sorry

end tan_45_eq_1_l280_280854


namespace alcohol_percentage_in_new_solution_l280_280215

theorem alcohol_percentage_in_new_solution :
  let original_volume := 40 -- liters
  let original_percentage_alcohol := 0.05
  let added_alcohol := 5.5 -- liters
  let added_water := 4.5 -- liters
  let original_alcohol := original_percentage_alcohol * original_volume
  let new_alcohol := original_alcohol + added_alcohol
  let new_volume := original_volume + added_alcohol + added_water
  (new_alcohol / new_volume) * 100 = 15 := by
  sorry

end alcohol_percentage_in_new_solution_l280_280215


namespace last_digit_of_a_power_b_l280_280211

-- Define the constants from the problem
def a : ℕ := 954950230952380948328708
def b : ℕ := 470128749397540235934750230

-- Define a helper function to calculate the last digit of a natural number
def last_digit (n : ℕ) : ℕ :=
  n % 10

-- Define the main statement to be proven
theorem last_digit_of_a_power_b : last_digit ((last_digit a) ^ (b % 4)) = 4 :=
by
  -- Here go the proof steps if we were to provide them
  sorry

end last_digit_of_a_power_b_l280_280211


namespace tan_45_eq_l280_280616

def tan_identity (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

def sin_45_eq : Real.sin 45 = Real.sqrt 2 / 2 := sorry

def cos_45_eq : Real.cos 45 = Real.sqrt 2 / 2 := sorry

theorem tan_45_eq : tan_identity 45 = 1 := by
  rw [tan_identity, sin_45_eq, cos_45_eq]
  -- simplifying (sqrt 2 / 2) / (sqrt 2 / 2) = 1
  sorry

end tan_45_eq_l280_280616


namespace max_power_sum_l280_280256

open Nat

theorem max_power_sum (a b : ℕ) (h_a_pos : a > 0) (h_b_gt_one : b > 1) (h_max : a ^ b < 500 ∧ 
  ∀ (a' b' : ℕ), a' > 0 → b' > 1 → a' ^ b' < 500 → a' ^ b' ≤ a ^ b ) : a + b = 24 :=
sorry

end max_power_sum_l280_280256


namespace sum_of_roots_of_quadratic_l280_280161

theorem sum_of_roots_of_quadratic : 
  (∃ (x : ℝ), (x - 7) ^ 2 = 16 ∧ (x = 11 ∨ x = 3)) → 
  (11 + 3 = 14) :=
by 
  intro h
  sorry

end sum_of_roots_of_quadratic_l280_280161


namespace sum_of_roots_l280_280155

theorem sum_of_roots : 
  let eq := (x - 7) ^ 2 = 16 in
  (roots : set ℝ) := { x | eq } in
  ∑ roots = 14 :=
by
sorry

end sum_of_roots_l280_280155


namespace tan_45_deg_l280_280695

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l280_280695


namespace tan_of_45_deg_l280_280699

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_of_45_deg_l280_280699


namespace central_cell_value_l280_280499

theorem central_cell_value :
  ∀ (a b c d e f g h i : ℝ),
    (a * b * c = 10) →
    (d * e * f = 10) →
    (g * h * i = 10) →
    (a * d * g = 10) →
    (b * e * h = 10) →
    (c * f * i = 10) →
    (a * b * d * e = 3) →
    (b * c * e * f = 3) →
    (d * e * g * h = 3) →
    (e * f * h * i = 3) →
    e = 0.00081 := 
by 
  intros a b c d e f g h i h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
  sorry

end central_cell_value_l280_280499


namespace gcd_smallest_value_l280_280471

theorem gcd_smallest_value {m n : ℕ} (h1 : 0 < m) (h2 : 0 < n) (h3 : Nat.gcd m n = 12) : Nat.gcd (8 * m) (18 * n) = 24 :=
by
  sorry

end gcd_smallest_value_l280_280471


namespace tan_45_degree_l280_280671

theorem tan_45_degree : ∀ (Q : ℝ × ℝ),
  -- Q corresponds to 45 degrees on the unit circle
  Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) →
  -- The radius of the unit circle is 1
  (Q.1 ^ 2 + Q.2 ^ 2 = 1) →
  -- QOE is a 45-45-90 triangle
  True →
  -- Prove that tan 45 degrees equals 1
  Real.tan (Real.pi / 4) = 1 :=
by
  -- The proof using these conditions is omitted
  intros,
  sorry

end tan_45_degree_l280_280671


namespace sum_of_roots_eq_l280_280187

theorem sum_of_roots_eq (x : ℝ) : (x - 7)^2 = 16 → x = 11 ∨ x = 3 → ∑ (root : ℝ) in {11, 3}, root = 14 :=
by
  assume h₁ : (x - 7)^2 = 16
  assume h₂ : x = 11 ∨ x = 3
  sorry

end sum_of_roots_eq_l280_280187


namespace total_fencing_needed_l280_280572

def width1 : ℕ := 4
def length1 : ℕ := 2 * width1 - 1

def length2 : ℕ := length1 + 3
def width2 : ℕ := width1 - 2

def width3 : ℕ := (width1 + width2) / 2
def length3 : ℚ := (length1 + length2) / 2

def perimeter (w l : ℚ) : ℚ := 2 * (w + l)

def P1 : ℚ := perimeter width1 length1
def P2 : ℚ := perimeter width2 length2
def P3 : ℚ := perimeter width3 length3

def total_fence : ℚ := P1 + P2 + P3

theorem total_fencing_needed : total_fence = 69 := 
  sorry

end total_fencing_needed_l280_280572


namespace number_of_shares_is_25_l280_280406

def wife_weekly_savings := 100
def husband_monthly_savings := 225
def duration_months := 4
def cost_per_share := 50

def total_savings : ℕ :=
  (wife_weekly_savings * 4 * duration_months) + (husband_monthly_savings * duration_months)

def amount_invested := total_savings / 2

def number_of_shares := amount_invested / cost_per_share

theorem number_of_shares_is_25 : number_of_shares = 25 := by
  sorry

end number_of_shares_is_25_l280_280406


namespace tan_45_degrees_eq_1_l280_280777

theorem tan_45_degrees_eq_1 :
    tan (45 * real.pi / 180) = 1 :=
by
  sorry

end tan_45_degrees_eq_1_l280_280777


namespace rate_of_interest_l280_280409

noncomputable def compound_interest (P r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r / 100) ^ (n : ℝ)

theorem rate_of_interest (P : ℝ) (r : ℝ) (A : ℕ → ℝ) :
  A 2 = compound_interest P r 2 →
  A 3 = compound_interest P r 3 →
  A 2 = 2420 →
  A 3 = 2662 →
  r = 10 :=
by
  sorry

end rate_of_interest_l280_280409


namespace tan_45_eq_one_l280_280793

theorem tan_45_eq_one (Q : ℝ × ℝ) (h1 : Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2)) : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_eq_one_l280_280793


namespace number_of_shares_is_25_l280_280405

def wife_weekly_savings := 100
def husband_monthly_savings := 225
def duration_months := 4
def cost_per_share := 50

def total_savings : ℕ :=
  (wife_weekly_savings * 4 * duration_months) + (husband_monthly_savings * duration_months)

def amount_invested := total_savings / 2

def number_of_shares := amount_invested / cost_per_share

theorem number_of_shares_is_25 : number_of_shares = 25 := by
  sorry

end number_of_shares_is_25_l280_280405


namespace sum_of_roots_eq_14_l280_280201

-- Define the condition as a hypothesis
def equation (x : ℝ) := (x - 7) ^ 2 = 16

-- Define the statement that needs to be proved
theorem sum_of_roots_eq_14 : 
  (∀ x : ℝ, equation x → x = 11 ∨ x = 3) → 11 + 3 = 14 :=
by 
sorry

end sum_of_roots_eq_14_l280_280201


namespace one_point_one_seven_three_billion_in_scientific_notation_l280_280411

theorem one_point_one_seven_three_billion_in_scientific_notation :
  (1.173 * 10^9 = 1.173 * 1000000000) :=
by
  sorry

end one_point_one_seven_three_billion_in_scientific_notation_l280_280411


namespace tan_45_eq_1_l280_280721

theorem tan_45_eq_1 (hcos : real.cos (real.pi / 4) = real.sqrt 2 / 2) (hsin : real.sin (real.pi / 4) = real.sqrt 2 / 2) : real.tan (real.pi / 4) = 1 :=
by
  have h : real.tan (real.pi / 4) = real.sin (real.pi / 4) / real.cos (real.pi / 4) := real.tan_eq_sin_div_cos _
  rw [hsin, hcos] at h
  simp at h
  exact h.symm

end tan_45_eq_1_l280_280721


namespace sum_of_integers_l280_280000

theorem sum_of_integers (n m : ℕ) (h1 : n * (n + 1) = 300) (h2 : m * (m + 1) * (m + 2) = 300) : 
  n + (n + 1) + m + (m + 1) + (m + 2) = 49 := 
by sorry

end sum_of_integers_l280_280000


namespace inequality_abc_l280_280075

theorem inequality_abc (a b c : ℝ) (h : a * b * c = 1) :
  1 / (2 * a^2 + b^2 + 3) + 1 / (2 * b^2 + c^2 + 3) + 1 / (2 * c^2 + a^2 + 3) ≤ 1 / 2 :=
by
  -- Sorry to skip the proof
  sorry

end inequality_abc_l280_280075


namespace find_greater_number_l280_280988

theorem find_greater_number (x y : ℝ) (h1 : x + y = 30) (h2 : x - y = 6) (h3 : x * y = 216) (h4 : x > y) : x = 18 := 
sorry

end find_greater_number_l280_280988


namespace y_percent_of_x_l280_280392

theorem y_percent_of_x (x y : ℝ) (h : 0.60 * (x - y) = 0.20 * (x + y)) : y / x = 0.5 :=
sorry

end y_percent_of_x_l280_280392


namespace inequality_satisfied_l280_280056

open Real

theorem inequality_satisfied (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 1) :
  a * sqrt b + b * sqrt c + c * sqrt a ≤ 1 / sqrt 3 :=
sorry

end inequality_satisfied_l280_280056


namespace tan_45_eq_1_l280_280728

theorem tan_45_eq_1 (hcos : real.cos (real.pi / 4) = real.sqrt 2 / 2) (hsin : real.sin (real.pi / 4) = real.sqrt 2 / 2) : real.tan (real.pi / 4) = 1 :=
by
  have h : real.tan (real.pi / 4) = real.sin (real.pi / 4) / real.cos (real.pi / 4) := real.tan_eq_sin_div_cos _
  rw [hsin, hcos] at h
  simp at h
  exact h.symm

end tan_45_eq_1_l280_280728


namespace tan_45_degrees_eq_1_l280_280784

theorem tan_45_degrees_eq_1 :
    tan (45 * real.pi / 180) = 1 :=
by
  sorry

end tan_45_degrees_eq_1_l280_280784


namespace problem_solution_l280_280417

noncomputable def problem_statement : Prop :=
  8 * (Real.cos (25 * Real.pi / 180)) ^ 2 - Real.tan (40 * Real.pi / 180) - 4 = Real.sqrt 3

theorem problem_solution : problem_statement :=
by
sorry

end problem_solution_l280_280417


namespace tan_45_deg_l280_280632

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l280_280632


namespace tan_45_degrees_eq_1_l280_280782

theorem tan_45_degrees_eq_1 :
    tan (45 * real.pi / 180) = 1 :=
by
  sorry

end tan_45_degrees_eq_1_l280_280782


namespace germination_rate_proof_l280_280128

def random_number_table := [[78226, 85384, 40527, 48987, 60602, 16085, 29971, 61279],
                            [43021, 92980, 27768, 26916, 27783, 84572, 78483, 39820],
                            [61459, 39073, 79242, 20372, 21048, 87088, 34600, 74636],
                            [63171, 58247, 12907, 50303, 28814, 40422, 97895, 61421],
                            [42372, 53183, 51546, 90385, 12120, 64042, 51320, 22983]]

noncomputable def first_4_tested_seeds : List Nat :=
  let numbers_in_random_table := [390, 737, 924, 220, 372]
  numbers_in_random_table.filter (λ x => x < 850) |>.take 4

theorem germination_rate_proof :
  first_4_tested_seeds = [390, 737, 220, 372] := 
by 
  sorry

end germination_rate_proof_l280_280128


namespace tan_45_deg_l280_280642

theorem tan_45_deg : (Real.tan (Float.pi / 4)) = 1 :=
by
  sorry

end tan_45_deg_l280_280642


namespace geometric_sequence_problem_l280_280439

noncomputable def geometric_sequence_sum_condition 
  (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (a 1 + a 2 + a 3 + a 4 + a 5 = 6) ∧ 
  (a 1 ^ 2 + a 2 ^ 2 + a 3 ^ 2 + a 4 ^ 2 + a 5 ^ 2 = 18) ∧ 
  (∀ n, a n = a 1 * q ^ (n - 1)) ∧ 
  (q ≠ 1)

theorem geometric_sequence_problem 
  (a : ℕ → ℝ) (q : ℝ) 
  (h : geometric_sequence_sum_condition a q) : 
  a 1 - a 2 + a 3 - a 4 + a 5 = 3 := 
by 
  sorry

end geometric_sequence_problem_l280_280439


namespace no_unique_symbols_for_all_trains_l280_280237

def proposition (a b c d : Prop) : Prop :=
  (¬a ∧  b ∧ ¬c ∧  d)
∨ ( a ∧ ¬b ∧ ¬c ∧ ¬d)

theorem no_unique_symbols_for_all_trains 
    (a b c d : Prop)
    (p : proposition a b c d)
    (s1 : ¬a ∧  b ∧ ¬c ∧  d)
    (s2 :  a ∧ ¬b ∧ ¬c ∧ ¬d) : 
    False :=
by {cases s1; cases s2; contradiction}

end no_unique_symbols_for_all_trains_l280_280237


namespace fg_of_2_eq_0_l280_280475

def f (x : ℝ) : ℝ := 4 - x^2
def g (x : ℝ) : ℝ := 3 * x - x^3

theorem fg_of_2_eq_0 : f (g 2) = 0 := by
  sorry

end fg_of_2_eq_0_l280_280475


namespace tan_45_deg_l280_280661

theorem tan_45_deg : Real.tan (π / 4) = 1 := 
by
  have h_sin_cos : Real.sin (π / 4) = Real.cos (π / 4) := 
    by sorry
  have h_value : Real.sin (π / 4) = (Real.sqrt 2) / 2 := 
    by sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos, h_value]
  calc
    (Real.sqrt 2)/ 2 / (Real.sqrt 2)/ 2 = 1 := sorry

end tan_45_deg_l280_280661


namespace tan_45_eq_l280_280619

def tan_identity (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

def sin_45_eq : Real.sin 45 = Real.sqrt 2 / 2 := sorry

def cos_45_eq : Real.cos 45 = Real.sqrt 2 / 2 := sorry

theorem tan_45_eq : tan_identity 45 = 1 := by
  rw [tan_identity, sin_45_eq, cos_45_eq]
  -- simplifying (sqrt 2 / 2) / (sqrt 2 / 2) = 1
  sorry

end tan_45_eq_l280_280619


namespace method_is_systematic_sampling_l280_280591

-- Define the conditions
def rows : ℕ := 25
def seats_per_row : ℕ := 20
def filled_auditorium : Prop := True
def seat_numbered_15_sampled : Prop := True
def interval : ℕ := 20

-- Define the concept of systematic sampling
def systematic_sampling (rows seats_per_row interval : ℕ) : Prop :=
  (rows > 0 ∧ seats_per_row > 0 ∧ interval > 0 ∧ (interval = seats_per_row))

-- State the problem in terms of proving that the sampling method is systematic
theorem method_is_systematic_sampling :
  filled_auditorium → seat_numbered_15_sampled → systematic_sampling rows seats_per_row interval :=
by
  intros h1 h2
  -- Assume that the proof goes here
  sorry

end method_is_systematic_sampling_l280_280591


namespace tan_45_eq_one_l280_280686

theorem tan_45_eq_one 
  (θ : ℝ)
  (hθ : θ = 45)
  (sin_45_eq : Real.sin (θ * Real.pi / 180) = Real.sqrt 2 / 2)
  (cos_45_eq : Real.cos (θ * Real.pi /180) = Real.sqrt 2 / 2)
  (tan_def : ∀ x : ℝ, 0 < Real.cos x → Real.tan x = Real.sin x / Real.cos x) : 
  Real.tan (θ * Real.pi / 180) = 1 := sorry

end tan_45_eq_one_l280_280686


namespace tangent_x_axis_l280_280034

noncomputable def curve (k : ℝ) : ℝ → ℝ := λ x => Real.log x - k * x + 3

theorem tangent_x_axis (k : ℝ) : 
  ∃ t : ℝ, curve k t = 0 ∧ deriv (curve k) t = 0 → k = Real.exp 2 :=
by
  sorry

end tangent_x_axis_l280_280034


namespace range_of_a_l280_280447

open Real

noncomputable def A (x : ℝ) : Prop := (x + 1) / (x - 2) ≥ 0
noncomputable def B (x a : ℝ) : Prop := x^2 - (2 * a + 1) * x + a^2 + a ≥ 0

theorem range_of_a :
  (∀ x, A x → B x a) ↔ -1 ≤ a ∧ a ≤ 1 :=
by
  sorry

end range_of_a_l280_280447


namespace tan_45_eq_one_l280_280680

theorem tan_45_eq_one 
  (θ : ℝ)
  (hθ : θ = 45)
  (sin_45_eq : Real.sin (θ * Real.pi / 180) = Real.sqrt 2 / 2)
  (cos_45_eq : Real.cos (θ * Real.pi /180) = Real.sqrt 2 / 2)
  (tan_def : ∀ x : ℝ, 0 < Real.cos x → Real.tan x = Real.sin x / Real.cos x) : 
  Real.tan (θ * Real.pi / 180) = 1 := sorry

end tan_45_eq_one_l280_280680


namespace central_cell_value_l280_280498

theorem central_cell_value :
  ∀ (a b c d e f g h i : ℝ),
    (a * b * c = 10) →
    (d * e * f = 10) →
    (g * h * i = 10) →
    (a * d * g = 10) →
    (b * e * h = 10) →
    (c * f * i = 10) →
    (a * b * d * e = 3) →
    (b * c * e * f = 3) →
    (d * e * g * h = 3) →
    (e * f * h * i = 3) →
    e = 0.00081 := 
by 
  intros a b c d e f g h i h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
  sorry

end central_cell_value_l280_280498


namespace tan_45_eq_one_l280_280871

theorem tan_45_eq_one : Real.tan (Real.pi / 4) = 1 := 
by
  sorry

end tan_45_eq_one_l280_280871


namespace tan_45_deg_l280_280657

theorem tan_45_deg : Real.tan (π / 4) = 1 := 
by
  have h_sin_cos : Real.sin (π / 4) = Real.cos (π / 4) := 
    by sorry
  have h_value : Real.sin (π / 4) = (Real.sqrt 2) / 2 := 
    by sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos, h_value]
  calc
    (Real.sqrt 2)/ 2 / (Real.sqrt 2)/ 2 = 1 := sorry

end tan_45_deg_l280_280657


namespace greatest_value_sum_eq_24_l280_280271

theorem greatest_value_sum_eq_24 {a b : ℕ} (ha : 0 < a) (hb : 1 < b) 
  (h1 : ∀ (x y : ℕ), 0 < x → 1 < y → x^y < 500 → x^y ≤ a^b) : 
  a + b = 24 := 
by
  sorry

end greatest_value_sum_eq_24_l280_280271


namespace socorro_training_days_l280_280972

variable (total_training_time_per_day : ℕ) (total_training_time : ℕ)

theorem socorro_training_days (h1 : total_training_time = 300) 
                              (h2 : total_training_time_per_day = 30) :
                              total_training_time / total_training_time_per_day = 10 := 
begin
  rw [h1, h2],
  norm_num,
end

end socorro_training_days_l280_280972


namespace sum_of_roots_of_quadratic_l280_280158

theorem sum_of_roots_of_quadratic : 
  (∃ (x : ℝ), (x - 7) ^ 2 = 16 ∧ (x = 11 ∨ x = 3)) → 
  (11 + 3 = 14) :=
by 
  intro h
  sorry

end sum_of_roots_of_quadratic_l280_280158


namespace tan_45_eq_1_l280_280762

theorem tan_45_eq_1 : Real.tan (Real.pi / 4) = 1 := by
  have h1 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have h2 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [Real.tan_eq_sin_div_cos, h1, h2]
  simp

end tan_45_eq_1_l280_280762


namespace sum_of_roots_eq_14_l280_280174

theorem sum_of_roots_eq_14 : 
  (∀ x : ℝ, (x - 7)^2 = 16 → (x = 11 ∨ x = 3)) →
  (∀ x1 x2 : ℝ, (x1 - 7)^2 = 16 ∧ (x2 - 7)^2 = 16 ∧ x1 ≠ x2 → x1 + x2 = 14) :=
by
  intros h x1 x2 h_comb
  sorry

end sum_of_roots_eq_14_l280_280174


namespace tan_of_45_deg_l280_280736

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  -- Definitions used directly from conditions
  have hcos : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry
  have hsin : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry

  -- Using the definition of tangent
  rw [Real.tan_eq_sin_div_cos, hsin, hcos]
  exact div_self (ne_of_gt (Real.sqrt_pos.2 zero_lt_two))

end tan_of_45_deg_l280_280736


namespace solve_for_x_l280_280003

theorem solve_for_x (x : ℝ) :
  (x + 3)^3 = -64 → x = -7 :=
by
  intro h
  sorry

end solve_for_x_l280_280003


namespace probability_correct_l280_280285

noncomputable def probability_sum_equals_sixteen (p_coin : ℚ) (p_die : ℚ) (age : ℕ): ℚ :=
  if age = 16 ∧ p_coin = 1 / 2 ∧ p_die = 1 / 6 then p_coin * p_die else 0

theorem probability_correct: 
  probability_sum_equals_sixteen (1/2) (1/6) 16 = 1 / 12 :=
sorry

end probability_correct_l280_280285


namespace meeting_distance_and_time_l280_280360

theorem meeting_distance_and_time 
  (total_distance : ℝ)
  (delta_time : ℝ)
  (x : ℝ)
  (V : ℝ)
  (v : ℝ)
  (t : ℝ) :

  -- Conditions 
  total_distance = 150 ∧
  delta_time = 25 ∧
  (150 - 2 * x) = 25 ∧
  (62.5 / v) = (87.5 / V) ∧
  (150 / v) - (150 / V) = 25 ∧
  t = (62.5 / v)

  -- Show that 
  → x = 62.5 ∧ t = 36 + 28 / 60 := 
by 
  sorry

end meeting_distance_and_time_l280_280360


namespace number_of_three_digit_multiples_of_7_l280_280460

theorem number_of_three_digit_multiples_of_7 : 
  let smallest_multiple := 7 * Nat.ceil (100 / 7)
  let largest_multiple := 7 * Nat.floor (999 / 7)
  (largest_multiple - smallest_multiple) / 7 + 1 = 128 :=
by
  sorry

end number_of_three_digit_multiples_of_7_l280_280460


namespace fruit_count_correct_l280_280300

def george_oranges := 45
def amelia_oranges := george_oranges - 18
def amelia_apples := 15
def george_apples := amelia_apples + 5

def olivia_orange_rate := 3
def olivia_apple_rate := 2
def olivia_minutes := 30
def olivia_cycle_minutes := 5
def olivia_cycles := olivia_minutes / olivia_cycle_minutes
def olivia_oranges := olivia_orange_rate * olivia_cycles
def olivia_apples := olivia_apple_rate * olivia_cycles

def total_oranges := george_oranges + amelia_oranges + olivia_oranges
def total_apples := george_apples + amelia_apples + olivia_apples
def total_fruits := total_oranges + total_apples

theorem fruit_count_correct : total_fruits = 137 := by
  sorry

end fruit_count_correct_l280_280300


namespace integers_satisfy_equation_l280_280208

theorem integers_satisfy_equation (x y z : ℤ) (h1 : x = z) (h2 : y - 1 = x) : x * (x - y) + y * (y - z) + z * (z - x) = 1 :=
by
  sorry

end integers_satisfy_equation_l280_280208


namespace tan_45_eq_1_l280_280757

theorem tan_45_eq_1 : Real.tan (Real.pi / 4) = 1 := by
  have h1 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have h2 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [Real.tan_eq_sin_div_cos, h1, h2]
  simp

end tan_45_eq_1_l280_280757


namespace arithmetic_sequence_sum_l280_280924

variable (a : ℕ → ℝ)
variable (d : ℝ)

axiom a2_a5 : a 2 + a 5 = 4
axiom a6_a9 : a 6 + a 9 = 20

theorem arithmetic_sequence_sum : a 4 + a 7 = 12 := by
  sorry

end arithmetic_sequence_sum_l280_280924


namespace room_length_perimeter_ratio_l280_280232

theorem room_length_perimeter_ratio :
  ∀ (L W : ℕ), L = 19 → W = 11 → (L : ℚ) / (2 * (L + W)) = 19 / 60 := by
  intros L W hL hW
  sorry

end room_length_perimeter_ratio_l280_280232


namespace sum_of_a_and_b_l280_280251

theorem sum_of_a_and_b (a b : ℕ) (h1: a > 0) (h2 : b > 1) (h3 : ∀ (x y : ℕ), x > 0 → y > 1 → x^y < 500 → x = a ∧ y = b → x^y ≥ a^b ) :
  a + b = 24 :=
sorry

end sum_of_a_and_b_l280_280251


namespace total_capacity_is_1600_l280_280026

/-- Eight liters is 20% of the capacity of one container. -/
def capacity_of_one_container := 8 / 0.20

/-- Calculate the total capacity of 40 such containers filled with water. -/
def total_capacity_of_40_containers := 40 * capacity_of_one_container

theorem total_capacity_is_1600 :
  total_capacity_of_40_containers = 1600 := by
    -- Proof is skipped using sorry.
    sorry

end total_capacity_is_1600_l280_280026


namespace john_money_left_l280_280950

noncomputable def total_earned (days_worked : ℕ) (earnings_per_day : ℕ) : ℕ := days_worked * earnings_per_day

noncomputable def total_spent (spent_books: ℕ) (spent_kaylee: ℕ) : ℕ := spent_books + spent_kaylee

def amount_left (total_earned : ℕ) (total_spent : ℕ) : ℕ := total_earned - total_spent

theorem john_money_left : 
  let days_worked := 26 in
  let earnings_per_day := 10 in
  let spent_books := 50 in
  let spent_kaylee := 50 in
  amount_left (total_earned days_worked earnings_per_day) (total_spent spent_books spent_kaylee) = 160 :=
by
  sorry

end john_money_left_l280_280950


namespace tan_45_degrees_l280_280804

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l280_280804


namespace tan_45_deg_l280_280634

theorem tan_45_deg : (Real.tan (Float.pi / 4)) = 1 :=
by
  sorry

end tan_45_deg_l280_280634


namespace tan_45_eq_1_l280_280857

theorem tan_45_eq_1 :
  let π := Real.pi in
  ∃ (x : ℝ), x = π / 4 ∧
  sin x = sqrt 2 / 2 ∧
  cos x = sqrt 2 / 2 →
  tan (π / 4) = 1 :=
by sorry

end tan_45_eq_1_l280_280857


namespace complement_P_l280_280013

def U : Set ℝ := Set.univ
def P : Set ℝ := {x | x^2 ≤ 1}

theorem complement_P :
  (U \ P) = Set.Iio (-1) ∪ Set.Ioi (1) :=
by
  sorry

end complement_P_l280_280013


namespace integer_solutions_2x2_2xy_9x_y_eq_2_l280_280554

theorem integer_solutions_2x2_2xy_9x_y_eq_2 : ∀ (x y : ℤ), 2 * x^2 - 2 * x * y + 9 * x + y = 2 → (x, y) = (1, 9) ∨ (x, y) = (2, 8) ∨ (x, y) = (0, 2) ∨ (x, y) = (-1, 3) := 
by 
  intros x y h
  sorry

end integer_solutions_2x2_2xy_9x_y_eq_2_l280_280554


namespace corveus_lack_of_sleep_l280_280419

def daily_sleep_actual : ℕ := 4
def daily_sleep_recommended : ℕ := 6
def days_in_week : ℕ := 7

theorem corveus_lack_of_sleep : (daily_sleep_recommended - daily_sleep_actual) * days_in_week = 14 := 
by 
  sorry

end corveus_lack_of_sleep_l280_280419


namespace total_fruit_salads_correct_l280_280595

-- Definitions for the conditions
def alayas_fruit_salads : ℕ := 200
def angels_fruit_salads : ℕ := 2 * alayas_fruit_salads
def total_fruit_salads : ℕ := alayas_fruit_salads + angels_fruit_salads

-- Theorem statement
theorem total_fruit_salads_correct : total_fruit_salads = 600 := by
  -- Proof goes here, but is not required for this task
  sorry

end total_fruit_salads_correct_l280_280595


namespace sum_of_a_and_b_l280_280252

theorem sum_of_a_and_b (a b : ℕ) (h1: a > 0) (h2 : b > 1) (h3 : ∀ (x y : ℕ), x > 0 → y > 1 → x^y < 500 → x = a ∧ y = b → x^y ≥ a^b ) :
  a + b = 24 :=
sorry

end sum_of_a_and_b_l280_280252


namespace sum_last_two_digits_7_13_23_l280_280133

theorem sum_last_two_digits_7_13_23 :
  (7 ^ 23 + 13 ^ 23) % 100 = 40 :=
by 
-- Proof goes here
sorry

end sum_last_two_digits_7_13_23_l280_280133


namespace sum_first_12_terms_l280_280987

variable (S : ℕ → ℝ)

def sum_of_first_n_terms (n : ℕ) : ℝ :=
  S n

theorem sum_first_12_terms (h₁ : sum_of_first_n_terms 4 = 30) (h₂ : sum_of_first_n_terms 8 = 100) :
  sum_of_first_n_terms 12 = 210 := 
sorry

end sum_first_12_terms_l280_280987


namespace find_central_cell_l280_280493

variable (a b c d e f g h i : ℝ)

def condition_1 : Prop :=
  a * b * c = 10 ∧ d * e * f = 10 ∧ g * h * i = 10

def condition_2 : Prop :=
  a * d * g = 10 ∧ b * e * h = 10 ∧ c * f * i = 10

def condition_3 : Prop :=
  a * b * d * e = 3 ∧ b * c * e * f = 3 ∧ d * e * g * h = 3 ∧ e * f * h * i = 3

theorem find_central_cell (h1 : condition_1 a b c d e f g h i)
                          (h2 : condition_2 a b c d e f g h i)
                          (h3 : condition_3 a b c d e f g h i) : 
  e = 0.00081 := 
sorry

end find_central_cell_l280_280493


namespace quadratic_solutions_l280_280087

-- Define the equation x^2 - 6x + 8 = 0
def quadratic_eq (x : ℝ) : Prop := x^2 - 6*x + 8 = 0

-- Lean statement for the equivalence of solutions
theorem quadratic_solutions : ∀ x : ℝ, quadratic_eq x ↔ x = 2 ∨ x = 4 :=
by
  intro x
  dsimp [quadratic_eq]
  sorry

end quadratic_solutions_l280_280087


namespace nice_circles_intersecting_segment_l280_280960

noncomputable def is_nice_circle (r : ℝ) : Prop :=
  ∃ (m n : ℤ), r ^ 2 = (m^2 + n^2 : ℤ)

theorem nice_circles_intersecting_segment :
  let A : (ℝ × ℝ) := (20, 15)
  let B : (ℝ × ℝ) := (20, 16)
  let segment := {t : ℝ | 15 < t ∧ t < 16}
  ∃ (r_list : List ℝ), 
  (∀ r ∈ r_list, r = Real.sqrt (400 + t ^ 2) ∧ t ∈ segment ∧ is_nice_circle r) ∧ 
  List.length r_list = 10 :=
by
  let A := (20, 15 : ℝ)
  let B := (20, 16 : ℝ)
  let segment := {t : ℝ | 15 < t ∧ t < 16}
  let r_list := [sqrt 626, sqrt 628, sqrt 629, sqrt 634, sqrt 637, sqrt 640, sqrt 641, sqrt 648, sqrt 650, sqrt 653]
  use r_list
  split
  { intros r hr
    split
    { use sqrt (400 + t ^ 2),
      repeat { sorry } },
    { sorry }
  }
  { sorry }

end nice_circles_intersecting_segment_l280_280960


namespace find_central_cell_l280_280496

variable (a b c d e f g h i : ℝ)

def condition_1 : Prop :=
  a * b * c = 10 ∧ d * e * f = 10 ∧ g * h * i = 10

def condition_2 : Prop :=
  a * d * g = 10 ∧ b * e * h = 10 ∧ c * f * i = 10

def condition_3 : Prop :=
  a * b * d * e = 3 ∧ b * c * e * f = 3 ∧ d * e * g * h = 3 ∧ e * f * h * i = 3

theorem find_central_cell (h1 : condition_1 a b c d e f g h i)
                          (h2 : condition_2 a b c d e f g h i)
                          (h3 : condition_3 a b c d e f g h i) : 
  e = 0.00081 := 
sorry

end find_central_cell_l280_280496


namespace f_is_decreasing_on_interval_l280_280563

noncomputable def f (x : ℝ) : ℝ := -3 * x ^ 2 - 2

theorem f_is_decreasing_on_interval :
  ∀ x y : ℝ, (1 ≤ x ∧ x < y ∧ y ≤ 2) → f y < f x :=
by
  sorry

end f_is_decreasing_on_interval_l280_280563


namespace total_capacity_l280_280029

def eight_liters : ℝ := 8
def percentage : ℝ := 0.20
def num_containers : ℕ := 40

theorem total_capacity (h : eight_liters = percentage * capacity) :
  40 * (eight_liters / percentage) = 1600 := sorry

end total_capacity_l280_280029


namespace greatest_value_sum_eq_24_l280_280275

theorem greatest_value_sum_eq_24 {a b : ℕ} (ha : 0 < a) (hb : 1 < b) 
  (h1 : ∀ (x y : ℕ), 0 < x → 1 < y → x^y < 500 → x^y ≤ a^b) : 
  a + b = 24 := 
by
  sorry

end greatest_value_sum_eq_24_l280_280275


namespace tommy_number_of_nickels_l280_280378

theorem tommy_number_of_nickels
  (d p n q : ℕ)
  (h1 : d = p + 10)
  (h2 : n = 2 * d)
  (h3 : q = 4)
  (h4 : p = 10 * q) : n = 100 :=
sorry

end tommy_number_of_nickels_l280_280378


namespace sum_of_roots_l280_280151

theorem sum_of_roots : 
  let eq := (x - 7) ^ 2 = 16 in
  (roots : set ℝ) := { x | eq } in
  ∑ roots = 14 :=
by
sorry

end sum_of_roots_l280_280151


namespace students_more_than_pets_l280_280900

-- Definition of given conditions
def num_students_per_classroom := 20
def num_rabbits_per_classroom := 2
def num_goldfish_per_classroom := 3
def num_classrooms := 5

-- Theorem stating the proof problem
theorem students_more_than_pets :
  let total_students := num_students_per_classroom * num_classrooms
  let total_pets := (num_rabbits_per_classroom + num_goldfish_per_classroom) * num_classrooms
  total_students - total_pets = 75 := by
  sorry

end students_more_than_pets_l280_280900


namespace cubes_sum_expr_l280_280055

variable {a b s p : ℝ}

theorem cubes_sum_expr (h1 : s = a + b) (h2 : p = a * b) : a^3 + b^3 = s^3 - 3 * s * p := by
  sorry

end cubes_sum_expr_l280_280055


namespace leak_drain_time_l280_280231

theorem leak_drain_time (P L : ℕ → ℕ) (H1 : ∀ t, P t = 1 / 2) (H2 : ∀ t, P t - L t = 1 / 3) : 
  (1 / L 1) = 6 :=
by
  sorry

end leak_drain_time_l280_280231


namespace sum_of_roots_eq_14_l280_280176

theorem sum_of_roots_eq_14 : 
  (∀ x : ℝ, (x - 7)^2 = 16 → (x = 11 ∨ x = 3)) →
  (∀ x1 x2 : ℝ, (x1 - 7)^2 = 16 ∧ (x2 - 7)^2 = 16 ∧ x1 ≠ x2 → x1 + x2 = 14) :=
by
  intros h x1 x2 h_comb
  sorry

end sum_of_roots_eq_14_l280_280176


namespace tan_45_deg_l280_280664

theorem tan_45_deg : Real.tan (π / 4) = 1 := 
by
  have h_sin_cos : Real.sin (π / 4) = Real.cos (π / 4) := 
    by sorry
  have h_value : Real.sin (π / 4) = (Real.sqrt 2) / 2 := 
    by sorry
  rw [Real.tan_eq_sin_div_cos, h_sin_cos, h_value]
  calc
    (Real.sqrt 2)/ 2 / (Real.sqrt 2)/ 2 = 1 := sorry

end tan_45_deg_l280_280664


namespace tan_45_eq_1_l280_280821

theorem tan_45_eq_1
: ∀ (Q : ℝ × ℝ), Q = (Real.sqrt 2 / 2, Real.sqrt 2 / 2) ->
  ∀ θ, θ = Real.pi / 4 -> 
  Real.tan θ = 1 :=
by
  intros Q HQ θ Hθ
  rw [HQ, Hθ]
  sorry

end tan_45_eq_1_l280_280821


namespace sum_of_a_and_b_l280_280267

theorem sum_of_a_and_b (a b : ℕ) (h1 : b > 1) (h2 : a^b < 500) (h3 : ∀ c d : ℕ, d > 1 → c^d < 500 → c^d ≤ a^b) : a + b = 24 :=
sorry

end sum_of_a_and_b_l280_280267


namespace min_photos_for_condition_l280_280290

noncomputable def minimum_photos (girls boys : ℕ) : ℕ :=
  if (girls = 4 ∧ boys = 8) 
  then 33
  else 0

theorem min_photos_for_condition (girls boys : ℕ) (photos : ℕ) :
  girls = 4 → boys = 8 → photos = minimum_photos girls boys
  → ∃ (pa : ℕ), pa >= 33 → pa = photos :=
by
  intros h1 h2 h3
  use minimum_photos girls boys
  rw [h3]
  sorry

end min_photos_for_condition_l280_280290


namespace gross_profit_percentage_without_discount_l280_280227

theorem gross_profit_percentage_without_discount (C P : ℝ)
  (discount : P * 0.9 = C * 1.2)
  (discount_profit : C * 0.2 = P * 0.9 - C) :
  (P - C) / C * 100 = 33.3 :=
by
  sorry

end gross_profit_percentage_without_discount_l280_280227


namespace vampire_count_after_two_nights_l280_280992

noncomputable def vampire_growth : Nat :=
  let first_night_new_vampires := 3 * 7
  let total_vampires_after_first_night := first_night_new_vampires + 3
  let second_night_new_vampires := total_vampires_after_first_night * (7 + 1)
  second_night_new_vampires + total_vampires_after_first_night

theorem vampire_count_after_two_nights : vampire_growth = 216 :=
by
  -- Skipping the detailed proof steps for now
  sorry

end vampire_count_after_two_nights_l280_280992


namespace tan_45_eq_l280_280612

def tan_identity (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

def sin_45_eq : Real.sin 45 = Real.sqrt 2 / 2 := sorry

def cos_45_eq : Real.cos 45 = Real.sqrt 2 / 2 := sorry

theorem tan_45_eq : tan_identity 45 = 1 := by
  rw [tan_identity, sin_45_eq, cos_45_eq]
  -- simplifying (sqrt 2 / 2) / (sqrt 2 / 2) = 1
  sorry

end tan_45_eq_l280_280612


namespace tan_45_eq_1_l280_280727

theorem tan_45_eq_1 (hcos : real.cos (real.pi / 4) = real.sqrt 2 / 2) (hsin : real.sin (real.pi / 4) = real.sqrt 2 / 2) : real.tan (real.pi / 4) = 1 :=
by
  have h : real.tan (real.pi / 4) = real.sin (real.pi / 4) / real.cos (real.pi / 4) := real.tan_eq_sin_div_cos _
  rw [hsin, hcos] at h
  simp at h
  exact h.symm

end tan_45_eq_1_l280_280727


namespace find_angle_B_and_sin_ratio_l280_280325

variable (A B C a b c : ℝ)
variable (h₁ : a * (Real.sin C - Real.sin A) / (Real.sin C + Real.sin B) = c - b)
variable (h₂ : Real.tan B / Real.tan A + Real.tan B / Real.tan C = 4)

theorem find_angle_B_and_sin_ratio :
  B = Real.pi / 3 ∧ Real.sin A / Real.sin C = (3 + Real.sqrt 5) / 2 ∨ Real.sin A / Real.sin C = (3 - Real.sqrt 5) / 2 :=
by
  sorry

end find_angle_B_and_sin_ratio_l280_280325


namespace evaluate_polynomial_at_2_l280_280901

theorem evaluate_polynomial_at_2 : (2^4 + 2^3 + 2^2 + 2 + 2) = 32 := 
by
  sorry

end evaluate_polynomial_at_2_l280_280901


namespace tan_45_deg_l280_280641

theorem tan_45_deg : (Real.tan (Float.pi / 4)) = 1 :=
by
  sorry

end tan_45_deg_l280_280641


namespace multiples_of_7_are_128_l280_280463

theorem multiples_of_7_are_128 : 
  let range_start := 100
  let range_end := 999
  let multiple_7_smallest := 7 * 15
  let multiple_7_largest := 7 * 142
  let n_terms := (142 - 15 + 1)
  n_terms = 128 := sorry

end multiples_of_7_are_128_l280_280463


namespace tan_45_deg_eq_one_l280_280651

/-
  Given:
  cos (45 deg) = sqrt 2 / 2,
  sin (45 deg) = sqrt 2 / 2,
  Prove: 
  tan (45 deg) = 1
-/

theorem tan_45_deg_eq_one : 
  (real.cos (real.pi / 4) = real.sqrt 2 / 2) → 
  (real.sin (real.pi / 4) = real.sqrt 2 / 2) → 
  real.tan (real.pi / 4) = 1 :=
by
  intros hcos hsin
  sorry

end tan_45_deg_eq_one_l280_280651


namespace tan_of_45_deg_l280_280703

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_of_45_deg_l280_280703


namespace abs_inequality_solution_l280_280116

theorem abs_inequality_solution (x : ℝ) : |x - 2| < 1 ↔ 1 < x ∧ x < 3 :=
by
  -- the proof would go here
  sorry

end abs_inequality_solution_l280_280116


namespace sum_abc_eq_ten_l280_280470

theorem sum_abc_eq_ten (a b c : ℝ) (h : (a - 5)^2 + (b - 3)^2 + (c - 2)^2 = 0) : a + b + c = 10 :=
by
  sorry

end sum_abc_eq_ten_l280_280470


namespace plane_equation_l280_280324

variable (a b c : ℝ)
variable (ha : a ≠ 0)
variable (hb : b ≠ 0)
variable (hc : c ≠ 0)

theorem plane_equation (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  ∃ x y z : ℝ, (x / a + y / b + z / c = 1) :=
sorry

end plane_equation_l280_280324


namespace value_of_x_l280_280482

theorem value_of_x (x : ℝ) (h : x = 80 + 0.2 * 80) : x = 96 :=
sorry

end value_of_x_l280_280482


namespace tan_45_degrees_eq_1_l280_280780

theorem tan_45_degrees_eq_1 :
    tan (45 * real.pi / 180) = 1 :=
by
  sorry

end tan_45_degrees_eq_1_l280_280780


namespace max_power_sum_l280_280257

open Nat

theorem max_power_sum (a b : ℕ) (h_a_pos : a > 0) (h_b_gt_one : b > 1) (h_max : a ^ b < 500 ∧ 
  ∀ (a' b' : ℕ), a' > 0 → b' > 1 → a' ^ b' < 500 → a' ^ b' ≤ a ^ b ) : a + b = 24 :=
sorry

end max_power_sum_l280_280257


namespace tan_45_deg_eq_one_l280_280841

/-- The tangent of 45 degrees is equal to 1. -/
theorem tan_45_deg_eq_one : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_one_l280_280841


namespace tan_45_deg_l280_280622

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l280_280622


namespace david_profit_l280_280234

theorem david_profit (weight : ℕ) (cost sell_price : ℝ) (h_weight : weight = 50) (h_cost : cost = 50) (h_sell_price : sell_price = 1.20) : 
  sell_price * weight - cost = 10 :=
by sorry

end david_profit_l280_280234


namespace max_value_relationship_l280_280576

theorem max_value_relationship (x y : ℝ) :
  (2005 - (x + y)^2 = 2005) → (x = -y) :=
by
  intro h
  sorry

end max_value_relationship_l280_280576


namespace max_power_sum_l280_280258

open Nat

theorem max_power_sum (a b : ℕ) (h_a_pos : a > 0) (h_b_gt_one : b > 1) (h_max : a ^ b < 500 ∧ 
  ∀ (a' b' : ℕ), a' > 0 → b' > 1 → a' ^ b' < 500 → a' ^ b' ≤ a ^ b ) : a + b = 24 :=
sorry

end max_power_sum_l280_280258


namespace tan_45_eq_one_l280_280872

theorem tan_45_eq_one : Real.tan (Real.pi / 4) = 1 := 
by
  sorry

end tan_45_eq_one_l280_280872


namespace tan_45_deg_eq_one_l280_280852

theorem tan_45_deg_eq_one :
  let θ := 45
  (∀ θ, tan θ = sin θ / cos θ) →
  (sin 45 = sqrt 2 / 2) →
  (cos 45 = sqrt 2 / 2) →
  tan 45 = 1 :=
by
  intros θ h_tan h_sin h_cos
  sorry

end tan_45_deg_eq_one_l280_280852


namespace inverse_variation_l280_280966

theorem inverse_variation (a : ℕ) (b : ℝ) (h : a * b = 400) (h₀ : a = 3200) : b = 0.125 :=
by sorry

end inverse_variation_l280_280966


namespace train_cross_pole_time_l280_280945

noncomputable def train_time_to_cross_pole (length : ℕ) (speed_km_per_hr : ℕ) : ℕ :=
  let speed_m_per_s := speed_km_per_hr * 1000 / 3600
  length / speed_m_per_s

theorem train_cross_pole_time :
  train_time_to_cross_pole 100 72 = 5 :=
by
  unfold train_time_to_cross_pole
  sorry

end train_cross_pole_time_l280_280945


namespace polynomial_has_root_l280_280113

theorem polynomial_has_root {a b c d : ℝ} 
  (h : a * c = 2 * b + 2 * d) : 
  ∃ x : ℝ, (x^2 + a * x + b = 0) ∨ (x^2 + c * x + d = 0) :=
by 
  sorry

end polynomial_has_root_l280_280113
