import Mathlib

namespace surface_area_increase_l638_63899

theorem surface_area_increase :
  let l := 4
  let w := 3
  let h := 2
  let side_cube := 1
  let original_surface := 2 * (l * w + l * h + w * h)
  let additional_surface := 6 * side_cube * side_cube
  let new_surface := original_surface + additional_surface
  new_surface = original_surface + 6 :=
by
  sorry

end surface_area_increase_l638_63899


namespace rectangle_area_change_l638_63897

theorem rectangle_area_change (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  let A := L * B
  let L' := 1.15 * L
  let B' := 0.80 * B
  let A' := L' * B'
  A' = 0.92 * A :=
by
  let A := L * B
  let L' := 1.15 * L
  let B' := 0.80 * B
  let A' := L' * B'
  show A' = 0.92 * A
  sorry

end rectangle_area_change_l638_63897


namespace sequence_term_and_k_value_l638_63858

/-- Given a sequence {a_n} whose sum of the first n terms is S_n = n^2 - 9n,
    prove the sequence term a_n = 2n - 10, and if 5 < a_k < 8, then k = 8. -/
theorem sequence_term_and_k_value (S : ℕ → ℤ)
  (a : ℕ → ℤ)
  (hS : ∀ n, S n = n^2 - 9 * n) :
  (∀ n, a n = if n = 1 then S 1 else S n - S (n - 1)) →
  (∀ n, a n = 2 * n - 10) ∧ (∀ k, 5 < a k ∧ a k < 8 → k = 8) :=
by {
  -- Given S_n = n^2 - 9n, we need to show a_n = 2n - 10 and verify when 5 < a_k < 8, then k = 8
  sorry
}

end sequence_term_and_k_value_l638_63858


namespace trigonometric_identity_l638_63808

theorem trigonometric_identity
  (α : ℝ)
  (h : Real.tan α = Real.sqrt 2) :
  2 * (Real.sin α)^2 - (Real.sin α) * (Real.cos α) + (Real.cos α)^2 = (5 - Real.sqrt 2) / 3 := 
by
  sorry

end trigonometric_identity_l638_63808


namespace difference_between_numbers_l638_63893

theorem difference_between_numbers (x y : ℕ) (h : x - y = 9) :
  (10 * x + y) - (10 * y + x) = 81 :=
by
  sorry

end difference_between_numbers_l638_63893


namespace flower_beds_fraction_l638_63838

-- Define the main problem parameters
def leg_length := (30 - 18) / 2
def triangle_area := (1 / 2) * (leg_length ^ 2)
def total_flower_bed_area := 2 * triangle_area
def yard_area := 30 * 6
def fraction_of_yard_occupied := total_flower_bed_area / yard_area

-- The theorem to be proved
theorem flower_beds_fraction :
  fraction_of_yard_occupied = 1/5 := by
  sorry

end flower_beds_fraction_l638_63838


namespace three_digit_diff_l638_63845

theorem three_digit_diff (a b : ℕ) (ha : 100 ≤ a ∧ a < 1000) (hb : 100 ≤ b ∧ b < 1000) :
  ∃ d : ℕ, d = a - b ∧ (d < 10 ∨ (10 ≤ d ∧ d < 100) ∨ (100 ≤ d ∧ d < 1000)) :=
sorry

end three_digit_diff_l638_63845


namespace algebraic_expression_correct_l638_63846

-- Definition of the problem
def algebraic_expression (x : ℝ) : ℝ :=
  2 * x + 3

-- Theorem statement
theorem algebraic_expression_correct (x : ℝ) :
  algebraic_expression x = 2 * x + 3 :=
by
  sorry

end algebraic_expression_correct_l638_63846


namespace partner_q_investment_time_l638_63831

theorem partner_q_investment_time 
  (P Q R : ℝ)
  (Profit_p Profit_q Profit_r : ℝ)
  (Tp Tq Tr : ℝ)
  (h1 : P / Q = 7 / 5)
  (h2 : Q / R = 5 / 3)
  (h3 : Profit_p / Profit_q = 7 / 14)
  (h4 : Profit_q / Profit_r = 14 / 9)
  (h5 : Tp = 5)
  (h6 : Tr = 9) :
  Tq = 14 :=
by
  sorry

end partner_q_investment_time_l638_63831


namespace value_of_expression_l638_63882

theorem value_of_expression (a b : ℝ) (h : a + b = 3) : 2 * a^2 + 4 * a * b + 2 * b^2 - 6 = 12 :=
by
  sorry

end value_of_expression_l638_63882


namespace items_per_charge_is_five_l638_63884

-- Define the number of dog treats, chew toys, rawhide bones, and credit cards as constants.
def num_dog_treats := 8
def num_chew_toys := 2
def num_rawhide_bones := 10
def num_credit_cards := 4

-- Define the total number of items.
def total_items := num_dog_treats + num_chew_toys + num_rawhide_bones

-- Prove that the number of items per credit card charge is 5.
theorem items_per_charge_is_five :
  (total_items / num_credit_cards) = 5 :=
by
  -- Proof goes here (we use sorry to skip the actual proof)
  sorry

end items_per_charge_is_five_l638_63884


namespace smallest_k_divides_polynomial_l638_63809

theorem smallest_k_divides_polynomial :
  ∃ (k : ℕ), k > 0 ∧ (∀ z : ℂ, z ≠ 0 → 
    (z ^ 11 + z ^ 9 + z ^ 7 + z ^ 6 + z ^ 5 + z ^ 2 + 1) ∣ (z ^ k - 1)) ∧ k = 11 := by
  sorry

end smallest_k_divides_polynomial_l638_63809


namespace probability_one_painted_face_and_none_painted_l638_63879

-- Define the total number of smaller unit cubes
def total_cubes : ℕ := 125

-- Define the number of cubes with exactly one painted face
def one_painted_face : ℕ := 25

-- Define the number of cubes with no painted faces
def no_painted_faces : ℕ := 125 - 25 - 12

-- Define the total number of ways to select two cubes uniformly at random
def total_pairs : ℕ := (total_cubes * (total_cubes - 1)) / 2

-- Define the number of successful outcomes
def successful_outcomes : ℕ := one_painted_face * no_painted_faces

-- Define the sought probability
def desired_probability : ℚ := (successful_outcomes : ℚ) / (total_pairs : ℚ)

-- Lean statement to prove the probability
theorem probability_one_painted_face_and_none_painted :
  desired_probability = 44 / 155 :=
by
  sorry

end probability_one_painted_face_and_none_painted_l638_63879


namespace percentage_increase_weekends_l638_63888

def weekday_price : ℝ := 18
def weekend_price : ℝ := 27

theorem percentage_increase_weekends : 
  (weekend_price - weekday_price) / weekday_price * 100 = 50 := by
  sorry

end percentage_increase_weekends_l638_63888


namespace quadratic_root_m_l638_63843

theorem quadratic_root_m (m : ℝ) : (∃ x : ℝ, x^2 + x + m^2 - 1 = 0 ∧ x = 0) → (m = 1 ∨ m = -1) :=
by 
  sorry

end quadratic_root_m_l638_63843


namespace angle_measure_supplement_complement_l638_63835

theorem angle_measure_supplement_complement (x : ℝ) 
    (h1 : 180 - x = 7 * (90 - x)) : 
    x = 75 := by
  sorry

end angle_measure_supplement_complement_l638_63835


namespace infinite_solutions_of_linear_eq_l638_63806

theorem infinite_solutions_of_linear_eq (a b : ℝ) : 
  (∃ b : ℝ, ∃ a : ℝ, 5 * a - 11 * b = 21) := sorry

end infinite_solutions_of_linear_eq_l638_63806


namespace range_of_f_l638_63848

noncomputable def f (x y z : ℝ) := ((x * y + y * z + z * x) * (x + y + z)) / ((x + y) * (y + z) * (z + x))

theorem range_of_f :
  ∃ x y z : ℝ, (0 < x ∧ 0 < y ∧ 0 < z) ∧ f x y z = r ↔ 1 ≤ r ∧ r ≤ 9 / 8 :=
sorry

end range_of_f_l638_63848


namespace converse_not_true_without_negatives_l638_63836

theorem converse_not_true_without_negatives (a b c d : ℕ) (h : a + d = b + c) : ¬(a - c = b - d) :=
by
  sorry

end converse_not_true_without_negatives_l638_63836


namespace total_birds_from_monday_to_wednesday_l638_63814

def birds_monday := 70
def birds_tuesday := birds_monday / 2
def birds_wednesday := birds_tuesday + 8
def total_birds := birds_monday + birds_tuesday + birds_wednesday

theorem total_birds_from_monday_to_wednesday : total_birds = 148 :=
by
  -- sorry is used here to skip the actual proof
  sorry

end total_birds_from_monday_to_wednesday_l638_63814


namespace x_y_sum_l638_63852

theorem x_y_sum (x y : ℝ) (h1 : |x| - 2 * x + y = 1) (h2 : x - |y| + y = 8) :
  x + y = 17 ∨ x + y = 1 :=
by
  sorry

end x_y_sum_l638_63852


namespace virginia_ends_up_with_93_eggs_l638_63823

-- Define the initial and subtracted number of eggs as conditions
def initial_eggs : ℕ := 96
def taken_eggs : ℕ := 3

-- The theorem we want to prove
theorem virginia_ends_up_with_93_eggs : (initial_eggs - taken_eggs) = 93 :=
by
  sorry

end virginia_ends_up_with_93_eggs_l638_63823


namespace car_speed_l638_63829

theorem car_speed (distance time speed : ℝ)
  (h_const_speed : ∀ t : ℝ, t = time → speed = distance / t)
  (h_distance : distance = 48)
  (h_time : time = 8) :
  speed = 6 :=
by
  sorry

end car_speed_l638_63829


namespace percentage_discount_four_friends_l638_63830

theorem percentage_discount_four_friends 
  (num_friends : ℕ)
  (original_price : ℝ)
  (total_spent : ℝ)
  (item_per_friend : ℕ)
  (total_items : ℕ)
  (each_spent : ℝ)
  (discount_percentage : ℝ):
  num_friends = 4 →
  original_price = 20 →
  total_spent = 40 →
  item_per_friend = 1 →
  total_items = num_friends * item_per_friend →
  each_spent = total_spent / num_friends →
  discount_percentage = ((original_price - each_spent) / original_price) * 100 →
  discount_percentage = 50 :=
by
  sorry

end percentage_discount_four_friends_l638_63830


namespace cost_of_fencing_is_8750_rsquare_l638_63818

variable (l w : ℝ)
variable (area : ℝ := 7500)
variable (cost_per_meter : ℝ := 0.25)
variable (ratio_lw : ℝ := 4/3)

theorem cost_of_fencing_is_8750_rsquare :
  (l / w = ratio_lw) → 
  (l * w = area) → 
  (2 * (l + w) * cost_per_meter = 87.50) :=
by 
  intros h1 h2
  sorry

end cost_of_fencing_is_8750_rsquare_l638_63818


namespace parabola_vertex_parabola_point_condition_l638_63805

-- Define the parabola function 
def parabola (x m : ℝ) : ℝ := x^2 - 2*m*x + m^2 - 1

-- 1. Prove the vertex of the parabola
theorem parabola_vertex (m : ℝ) : ∃ x y, (∀ x m, parabola x m = (x - m)^2 - 1) ∧ (x = m ∧ y = -1) :=
by
  sorry

-- 2. Prove the range of values for m given the conditions on points A and B
theorem parabola_point_condition (m : ℝ) (y1 y2 : ℝ) :
  (y1 > y2) ∧ 
  (parabola (1 - 2*m) m = y1) ∧ 
  (parabola (m + 1) m = y2) → m < 0 ∨ m > 2/3 :=
by
  sorry

end parabola_vertex_parabola_point_condition_l638_63805


namespace joshua_finishes_after_malcolm_l638_63844

def time_difference_between_runners
  (race_length : ℕ)
  (malcolm_speed : ℕ)
  (joshua_speed : ℕ)
  (malcolm_finish_time : ℕ := malcolm_speed * race_length)
  (joshua_finish_time : ℕ := joshua_speed * race_length) : ℕ :=
joshua_finish_time - malcolm_finish_time

theorem joshua_finishes_after_malcolm
  (race_length : ℕ)
  (malcolm_speed : ℕ)
  (joshua_speed : ℕ)
  (h_race_length : race_length = 12)
  (h_malcolm_speed : malcolm_speed = 7)
  (h_joshua_speed : joshua_speed = 9) : time_difference_between_runners race_length malcolm_speed joshua_speed = 24 :=
by 
  subst h_race_length
  subst h_malcolm_speed
  subst h_joshua_speed
  rfl

#print joshua_finishes_after_malcolm

end joshua_finishes_after_malcolm_l638_63844


namespace option_D_is_correct_l638_63834

noncomputable def correct_operation : Prop := 
  (∀ x : ℝ, x + x ≠ 2 * x^2) ∧
  (∀ y : ℝ, 2 * y^3 + 3 * y^2 ≠ 5 * y^5) ∧
  (∀ x : ℝ, 2 * x - x ≠ 1) ∧
  (∀ x y : ℝ, 4 * x^3 * y^2 - (-2)^2 * x^3 * y^2 = 0)

theorem option_D_is_correct : correct_operation :=
by {
  -- We'll complete the proofs later
  sorry
}

end option_D_is_correct_l638_63834


namespace at_least_two_squares_same_size_l638_63875

theorem at_least_two_squares_same_size (S : ℝ) : 
  ∃ a b : ℝ, a = b ∧ 
  (∀ i : ℕ, i < 10 → 
   ∀ j : ℕ, j < 10 → 
   (∃ k : ℕ, k < 9 ∧ 
    ((∃ (x y : ℕ), x < 10 ∧ y < 10 ∧ x ≠ y → 
          (i = x ∧ j = y)) → 
        ((S / 10) = (a * k)) ∨ ((S / 10) = (b * k))))) := sorry

end at_least_two_squares_same_size_l638_63875


namespace parallel_planes_of_perpendicular_lines_l638_63885

-- Definitions of planes and lines
variable (Plane Line : Type)
variable (α β γ : Plane)
variable (m n : Line)

-- Relations between planes and lines
variable (perpendicular : Plane → Line → Prop)
variable (parallel : Plane → Plane → Prop)
variable (line_parallel : Line → Line → Prop)

-- Conditions for the proof
variable (m_perp_α : perpendicular α m)
variable (n_perp_β : perpendicular β n)
variable (m_par_n : line_parallel m n)

-- Statement of the theorem
theorem parallel_planes_of_perpendicular_lines :
  parallel α β :=
sorry

end parallel_planes_of_perpendicular_lines_l638_63885


namespace fraction_of_bread_slices_eaten_l638_63812

theorem fraction_of_bread_slices_eaten
    (total_slices : ℕ)
    (slices_used_for_sandwich : ℕ)
    (remaining_slices : ℕ)
    (slices_eaten_for_breakfast : ℕ)
    (h1 : total_slices = 12)
    (h2 : slices_used_for_sandwich = 2)
    (h3 : remaining_slices = 6)
    (h4 : total_slices - slices_used_for_sandwich - remaining_slices = slices_eaten_for_breakfast) :
    slices_eaten_for_breakfast / total_slices = 1 / 3 :=
sorry

end fraction_of_bread_slices_eaten_l638_63812


namespace fraction_sum_equals_mixed_number_l638_63801

theorem fraction_sum_equals_mixed_number :
  (3 / 5 : ℚ) + (2 / 3) + (16 / 15) = (7 / 3) :=
by sorry

end fraction_sum_equals_mixed_number_l638_63801


namespace min_value_of_trig_expression_l638_63826

open Real

theorem min_value_of_trig_expression (α : ℝ) (h₁ : sin α ≠ 0) (h₂ : cos α ≠ 0) : 
  (9 / (sin α)^2 + 1 / (cos α)^2) ≥ 16 :=
  sorry

end min_value_of_trig_expression_l638_63826


namespace calculate_p_p1_neg1_p_neg5_neg2_l638_63855

def p (x y : ℤ) : ℤ :=
  if x ≥ 0 ∧ y ≥ 0 then
    x + y
  else if x < 0 ∧ y < 0 then
    x - 2 * y
  else
    3 * x + y

theorem calculate_p_p1_neg1_p_neg5_neg2 :
  p (p 1 (-1)) (p (-5) (-2)) = 5 :=
by
  sorry

end calculate_p_p1_neg1_p_neg5_neg2_l638_63855


namespace sequence_conjecture_l638_63877

theorem sequence_conjecture (a : ℕ → ℝ) (h₁ : a 1 = 7)
  (h₂ : ∀ n, a (n + 1) = 7 * a n / (a n + 7)) :
  ∀ n, a n = 7 / n :=
by
  sorry

end sequence_conjecture_l638_63877


namespace red_yellow_flowers_l638_63803

theorem red_yellow_flowers
  (total : ℕ)
  (yellow_white : ℕ)
  (red_white : ℕ)
  (extra_red_over_white : ℕ)
  (H1 : total = 44)
  (H2 : yellow_white = 13)
  (H3 : red_white = 14)
  (H4 : extra_red_over_white = 4) :
  ∃ (red_yellow : ℕ), red_yellow = 17 := by
  sorry

end red_yellow_flowers_l638_63803


namespace triangle_table_distinct_lines_l638_63816

theorem triangle_table_distinct_lines (a : ℕ) (h : a > 1) : 
  ∀ (n : ℕ) (line : ℕ → ℕ), 
  (line 0 = a) → 
  (∀ k, line (2*k + 1) = line k ^ 2 ∧ line (2*k + 2) = line k + 1) → 
  ∀ i j, i < 2^n → j < 2^n → (i ≠ j → line i ≠ line j) := 
by {
  sorry
}

end triangle_table_distinct_lines_l638_63816


namespace circle_diameter_of_circumscribed_square_l638_63869

theorem circle_diameter_of_circumscribed_square (r : ℝ) (s : ℝ) (h1 : s = 2 * r) (h2 : 4 * s = π * r^2) : 2 * r = 16 / π := by
  sorry

end circle_diameter_of_circumscribed_square_l638_63869


namespace distance_after_3rd_turn_l638_63837

theorem distance_after_3rd_turn (d1 d2 d4 total_distance : ℕ) 
  (h1 : d1 = 5) 
  (h2 : d2 = 8) 
  (h4 : d4 = 0) 
  (h_total : total_distance = 23) : 
  total_distance - (d1 + d2 + d4) = 10 := 
  sorry

end distance_after_3rd_turn_l638_63837


namespace painters_work_days_l638_63854

theorem painters_work_days 
  (six_painters_days : ℝ) (number_six_painters : ℝ) (total_work_units : ℝ)
  (number_four_painters : ℝ) 
  (h1 : number_six_painters = 6)
  (h2 : six_painters_days = 1.4)
  (h3 : total_work_units = number_six_painters * six_painters_days) 
  (h4 : number_four_painters = 4) :
  2 + 1 / 10 = total_work_units / number_four_painters :=
by
  rw [h3, h1, h2, h4]
  sorry

end painters_work_days_l638_63854


namespace volunteer_arrangement_l638_63820

theorem volunteer_arrangement (volunteers : Fin 5) (elderly : Fin 2) 
  (h1 : elderly.1 ≠ 0 ∧ elderly.1 ≠ 6) : 
  ∃ arrangements : ℕ, arrangements = 960 := 
sorry

end volunteer_arrangement_l638_63820


namespace find_a_l638_63815

theorem find_a (a : ℝ) (h : ∃ x, x = -1 ∧ 4 * x^3 + 2 * a * x = 8) : a = -6 :=
sorry

end find_a_l638_63815


namespace avg_ABC_l638_63856

variables (A B C : Set ℕ) -- Sets of people
variables (a b c : ℕ) -- Numbers of people in sets A, B, and C respectively
variables (sum_A sum_B sum_C : ℕ) -- Sums of the ages of people in sets A, B, and C respectively

-- Given conditions
axiom avg_A : sum_A / a = 30
axiom avg_B : sum_B / b = 20
axiom avg_C : sum_C / c = 45

axiom avg_AB : (sum_A + sum_B) / (a + b) = 25
axiom avg_AC : (sum_A + sum_C) / (a + c) = 40
axiom avg_BC : (sum_B + sum_C) / (b + c) = 32

theorem avg_ABC : (sum_A + sum_B + sum_C) / (a + b + c) = 35 :=
by
  sorry

end avg_ABC_l638_63856


namespace Elina_garden_area_l638_63822

theorem Elina_garden_area :
  ∀ (L W: ℝ),
    (30 * L = 1500) →
    (12 * (2 * (L + W)) = 1500) →
    (L * W = 625) :=
by
  intros L W h1 h2
  sorry

end Elina_garden_area_l638_63822


namespace max_cars_per_div_100_is_20_l638_63810

theorem max_cars_per_div_100_is_20 :
  let m : ℕ := Nat.succ (Nat.succ 0) -- represents m going to infinity
  let car_length : ℕ := 5
  let speed_factor : ℕ := 10
  let sensor_distance_per_hour : ℕ := speed_factor * 1000 * m
  let separation_distance : ℕ := car_length * (m + 1)
  let max_cars : ℕ := (sensor_distance_per_hour / separation_distance) * m
  Nat.floor ((2 * (max_cars : ℝ)) / 100) = 20 :=
by
  sorry

end max_cars_per_div_100_is_20_l638_63810


namespace square_area_720_l638_63891

noncomputable def length_squared {α : Type*} [EuclideanDomain α] (a b : α) := a * a + b * b

theorem square_area_720
  (side x : ℝ)
  (h1 : BE = 20) (h2 : EF = 20) (h3 : FD = 20)
  (h4 : AE = 2 * ED) (h5 : BF = 2 * FC)
  : x * x = 720 :=
by
  let AE := 2/3 * side
  let ED := 1/3 * side
  let BF := 2/3 * side
  let FC := 1/3 * side
  have h6 : length_squared BF EF = BE * BE := sorry
  have h7 : x * x = 720 := sorry
  exact h7

end square_area_720_l638_63891


namespace necessary_but_not_sufficient_l638_63849

noncomputable def f (a x : ℝ) : ℝ := x^2 - 2*(a+1)*x + 3

theorem necessary_but_not_sufficient (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x → f a x - f a 1 ≥ 0) ↔ (a ≤ -2) :=
sorry

end necessary_but_not_sufficient_l638_63849


namespace number_of_milkshakes_l638_63859

-- Define the amounts and costs
def initial_money : ℕ := 132
def remaining_money : ℕ := 70
def hamburger_cost : ℕ := 4
def milkshake_cost : ℕ := 5
def hamburgers_bought : ℕ := 8

-- Defining the money spent calculations
def hamburgers_spent : ℕ := hamburgers_bought * hamburger_cost
def total_spent : ℕ := initial_money - remaining_money
def milkshake_spent : ℕ := total_spent - hamburgers_spent

-- The final theorem to prove
theorem number_of_milkshakes : (milkshake_spent / milkshake_cost) = 6 :=
by
  sorry

end number_of_milkshakes_l638_63859


namespace sequence_sum_l638_63871

variable (P Q R S T U V : ℤ)
variable (hR : R = 7)
variable (h1 : P + Q + R = 36)
variable (h2 : Q + R + S = 36)
variable (h3 : R + S + T = 36)
variable (h4 : S + T + U = 36)
variable (h5 : T + U + V = 36)

theorem sequence_sum (P Q R S T U V : ℤ)
  (hR : R = 7)
  (h1 : P + Q + R = 36)
  (h2 : Q + R + S = 36)
  (h3 : R + S + T = 36)
  (h4 : S + T + U = 36)
  (h5 : T + U + V = 36) :
  P + V = 29 := 
sorry

end sequence_sum_l638_63871


namespace distance_between_locations_l638_63840

theorem distance_between_locations
  (d_AC d_BC : ℚ)
  (d : ℚ)
  (meet_C : d_AC + d_BC = d)
  (travel_A_B : 150 + 150 + 540 = 840)
  (distance_ratio : 840 / 540 = 14 / 9)
  (distance_ratios : d_AC / d_BC = 14 / 9)
  (C_D : 540 = 5 * d / 23) :
  d = 2484 :=
by
  sorry

end distance_between_locations_l638_63840


namespace sum_of_roots_abs_gt_six_l638_63866

theorem sum_of_roots_abs_gt_six {p r1 r2 : ℝ} (h1 : r1 + r2 = -p) (h2 : r1 * r2 = 9) (h3 : r1 ≠ r2) (h4 : p^2 > 36) : |r1 + r2| > 6 :=
sorry

end sum_of_roots_abs_gt_six_l638_63866


namespace proof_l638_63894

variable (p : ℕ) (ε : ℤ)
variable (RR NN NR RN : ℕ)

-- Conditions
axiom h1 : ∀ n ≤ p - 2, 
  (n % 2 = 0 ∧ (n + 1) % 2 = 0) ∨ 
  (n % 2 ≠ 0 ∧ (n + 1) % 2 ≠ 0) ∨ 
  (n % 2 ≠ 0 ∧ (n + 1) % 2 = 0 ) ∨ 
  (n % 2 = 0 ∧ (n + 1) % 2 ≠ 0) 

axiom h2 :  RR + NN - RN - NR = 1

axiom h3 : ε = (-1) ^ ((p - 1) / 2)

axiom h4 : RR + RN = (p - 2 - ε) / 2

axiom h5 : RR + NR = (p - 1) / 2 - 1

axiom h6 : NR + NN = (p - 2 + ε) / 2

axiom h7 : RN + NN = (p - 1) / 2  

-- To prove
theorem proof : 
  RR = (p / 4) - (ε + 4) / 4 ∧ 
  RN = (p / 4) - (ε) / 4 ∧ 
  NN = (p / 4) + (ε - 2) / 4 ∧ 
  NR = (p / 4) + (ε - 2) / 4 := 
sorry

end proof_l638_63894


namespace total_carrots_computation_l638_63873

-- Definitions
def initial_carrots : ℕ := 19
def thrown_out_carrots : ℕ := 4
def next_day_carrots : ℕ := 46

def total_carrots (c1 c2 t : ℕ) : ℕ := (c1 - t) + c2

-- The statement to prove
theorem total_carrots_computation :
  total_carrots initial_carrots next_day_carrots thrown_out_carrots = 61 :=
by sorry

end total_carrots_computation_l638_63873


namespace Tim_sleep_hours_l638_63878

theorem Tim_sleep_hours (x : ℕ) : 
  (x + x + 10 + 10 = 32) → x = 6 :=
by
  intro h
  sorry

end Tim_sleep_hours_l638_63878


namespace circumferences_ratio_l638_63862

theorem circumferences_ratio (r1 r2 : ℝ) (h : (π * r1 ^ 2) / (π * r2 ^ 2) = 49 / 64) : r1 / r2 = 7 / 8 :=
sorry

end circumferences_ratio_l638_63862


namespace solution_to_system_l638_63889

theorem solution_to_system (x y a b : ℝ) 
  (h1 : x = 1) (h2 : y = 2) 
  (h3 : a * x + b * y = 4) 
  (h4 : b * x - a * y = 7) : 
  a + b = 1 :=
by
  sorry

end solution_to_system_l638_63889


namespace problem1_simplified_problem2_simplified_l638_63898

-- Definition and statement for the first problem
def problem1_expression (x y : ℝ) : ℝ := 
  -3 * x * y - 3 * x^2 + 4 * x * y + 2 * x^2

theorem problem1_simplified (x y : ℝ) : 
  problem1_expression x y = x * y - x^2 := 
by
  sorry

-- Definition and statement for the second problem
def problem2_expression (a b : ℝ) : ℝ := 
  3 * (a^2 - 2 * a * b) - 5 * (a^2 + 4 * a * b)

theorem problem2_simplified (a b : ℝ) : 
  problem2_expression a b = -2 * a^2 - 26 * a * b :=
by
  sorry

end problem1_simplified_problem2_simplified_l638_63898


namespace function_monotonicity_l638_63800

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)

theorem function_monotonicity :
  ∀ x₁ x₂, -Real.pi / 6 ≤ x₁ → x₁ ≤ x₂ → x₂ ≤ Real.pi / 3 → f x₁ ≤ f x₂ :=
by
  sorry

end function_monotonicity_l638_63800


namespace max_product_xyz_l638_63887

theorem max_product_xyz : ∃ (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ x + y + z = 12 ∧ z ≤ 3 * x ∧ ∀ (a b c : ℕ), a + b + c = 12 → c ≤ 3 * a → 0 < a ∧ 0 < b ∧ 0 < c → a * b * c ≤ 48 :=
by
  sorry

end max_product_xyz_l638_63887


namespace rectangle_perimeter_l638_63868

theorem rectangle_perimeter :
  ∃ (a b : ℤ), a ≠ b ∧ a * b = 2 * (2 * a + 2 * b) ∧ 2 * (a + b) = 36 :=
by
  sorry

end rectangle_perimeter_l638_63868


namespace max_value_of_quadratic_l638_63860

def quadratic_func (x : ℝ) : ℝ := -3 * (x - 2) ^ 2 - 3

theorem max_value_of_quadratic : 
  ∃ x : ℝ, quadratic_func x = -3 :=
by
  sorry

end max_value_of_quadratic_l638_63860


namespace polynomial_operation_correct_l638_63825

theorem polynomial_operation_correct :
    ∀ (s t : ℝ), (s * t + 0.25 * s * t = 0) :=
by
  intros s t
  sorry

end polynomial_operation_correct_l638_63825


namespace find_angle_degree_l638_63833

theorem find_angle_degree (x : ℝ) (h : 90 - x = (1 / 3) * (180 - x) + 20) : x = 75 := by
    sorry

end find_angle_degree_l638_63833


namespace only_pair_2_2_satisfies_l638_63865

theorem only_pair_2_2_satisfies :
  ∀ a b : ℕ, (∀ n : ℕ, ∃ c : ℕ, a ^ n + b ^ n = c ^ (n + 1)) → (a = 2 ∧ b = 2) :=
by sorry

end only_pair_2_2_satisfies_l638_63865


namespace eval_expression_l638_63841

def square_avg (a b : ℚ) : ℚ := (a^2 + b^2) / 2
def custom_avg (a b c : ℚ) : ℚ := (a + b + 2 * c) / 3

theorem eval_expression : 
  custom_avg (custom_avg 2 (-1) 1) (square_avg 2 3) 1 = 19 / 6 :=
by
  sorry

end eval_expression_l638_63841


namespace Sarah_is_26_l638_63857

noncomputable def Sarah_age (mark_age billy_age ana_age : ℕ): ℕ :=
  3 * mark_age - 4

def Mark_age (billy_age : ℕ): ℕ :=
  billy_age + 4

def Billy_age (ana_age : ℕ): ℕ :=
  ana_age / 2

def Ana_age : ℕ := 15 - 3

theorem Sarah_is_26 : Sarah_age (Mark_age (Billy_age Ana_age)) (Billy_age Ana_age) Ana_age = 26 := 
by
  sorry

end Sarah_is_26_l638_63857


namespace cost_per_item_l638_63813

theorem cost_per_item (total_cost : ℝ) (num_items : ℕ) (cost_per_item : ℝ) 
                      (h1 : total_cost = 26) (h2 : num_items = 8) : 
                      cost_per_item = total_cost / num_items := 
by
  sorry

end cost_per_item_l638_63813


namespace maurice_needs_7_letters_l638_63870
noncomputable def prob_no_job (n : ℕ) : ℝ := (4 / 5) ^ n

theorem maurice_needs_7_letters :
  ∃ n : ℕ, (prob_no_job n) ≤ 1 / 4 ∧ n = 7 :=
by
  sorry

end maurice_needs_7_letters_l638_63870


namespace green_beads_in_pattern_l638_63861

noncomputable def G : ℕ := 3
def P : ℕ := 5
def R (G : ℕ) : ℕ := 2 * G
def total_beads (G : ℕ) (P : ℕ) (R : ℕ) : ℕ := 3 * (G + P + R) + 10 * 5 * (G + P + R)

theorem green_beads_in_pattern :
  total_beads 3 5 (R 3) = 742 :=
by
  sorry

end green_beads_in_pattern_l638_63861


namespace pentagon_area_l638_63850

theorem pentagon_area {a b c d e : ℕ} (split: ℕ) (non_parallel1 non_parallel2 parallel1 parallel2 : ℕ)
  (h1 : a = 16) (h2 : b = 25) (h3 : c = 30) (h4 : d = 26) (h5 : e = 25)
  (split_condition : a + b + c + d + e = 5 * split)
  (np_condition1: non_parallel1 = c) (np_condition2: non_parallel2 = a)
  (p_condition1: parallel1 = d) (p_condition2: parallel2 = e)
  (area_triangle: 1 / 2 * b * a = 200)
  (area_trapezoid: 1 / 2 * (parallel1 + parallel2) * non_parallel1 = 765) :
  a + b + c + d + e = 965 := by
  sorry

end pentagon_area_l638_63850


namespace reservoir_fullness_before_storm_l638_63832

-- Definition of the conditions as Lean definitions
def storm_deposits : ℝ := 120 -- in billion gallons
def reservoir_percentage_after_storm : ℝ := 85 -- percentage
def original_contents : ℝ := 220 -- in billion gallons

-- The proof statement
theorem reservoir_fullness_before_storm (storm_deposits reservoir_percentage_after_storm original_contents : ℝ) : 
    (169 / 340) * 100 = 49.7 := 
  sorry

end reservoir_fullness_before_storm_l638_63832


namespace ratio_to_percentage_l638_63895

theorem ratio_to_percentage (x y : ℚ) (h : (2/3 * x) / (4/5 * y) = 5 / 6) : (5 / 6 : ℚ) * 100 = 83.33 :=
by
  sorry

end ratio_to_percentage_l638_63895


namespace abs_condition_iff_range_l638_63824

theorem abs_condition_iff_range (x : ℝ) : 
  (|x-1| + |x+2| ≤ 5) ↔ (-3 ≤ x ∧ x ≤ 2) := 
sorry

end abs_condition_iff_range_l638_63824


namespace max_area_of_triangle_ABC_l638_63874

noncomputable def max_triangle_area (a b c : ℝ) (A B C : ℝ) := 
  1 / 2 * b * c * Real.sin A

theorem max_area_of_triangle_ABC :
  ∀ (a b c A B C : ℝ)
  (ha : a = 2)
  (hTrig : a = Real.sqrt (b^2 + c^2 - 2 * b * c * Real.cos A))
  (hCondition: 3 * b * Real.sin C - 5 * c * Real.sin B * Real.cos A = 0),
  max_triangle_area a b c A B C ≤ 2 := 
by
  intros a b c A B C ha hTrig hCondition
  sorry

end max_area_of_triangle_ABC_l638_63874


namespace evaluate_expression_l638_63896

theorem evaluate_expression :
  (4^1001 * 9^1002) / (6^1002 * 4^1000) = (3^1002) / (2^1000) :=
by sorry

end evaluate_expression_l638_63896


namespace polynomial_value_at_minus_two_l638_63807

def f (x : ℝ) : ℝ := x^5 + 4*x^4 + x^2 + 20*x + 16

theorem polynomial_value_at_minus_two : f (-2) = 12 := by 
  sorry

end polynomial_value_at_minus_two_l638_63807


namespace students_total_l638_63867

theorem students_total (scavenger_hunt_students : ℕ) (ski_trip_students : ℕ) 
  (h1 : ski_trip_students = 2 * scavenger_hunt_students) 
  (h2 : scavenger_hunt_students = 4000) : 
  scavenger_hunt_students + ski_trip_students = 12000 := 
by
  sorry

end students_total_l638_63867


namespace largest_quotient_is_25_l638_63811

def largest_quotient_set : Set ℤ := {-25, -4, -1, 1, 3, 9}

theorem largest_quotient_is_25 :
  ∃ (a b : ℤ), a ∈ largest_quotient_set ∧ b ∈ largest_quotient_set ∧ b ≠ 0 ∧ (a : ℚ) / b = 25 := by
  sorry

end largest_quotient_is_25_l638_63811


namespace number_of_members_l638_63821

noncomputable def club_members (n O N : ℕ) : Prop :=
  (3 * n = O - N) ∧ (O - N = 15)

theorem number_of_members (n O N : ℕ) (h : club_members n O N) : n = 5 :=
  by
    sorry

end number_of_members_l638_63821


namespace wire_ratio_l638_63828

theorem wire_ratio (a b : ℝ) (h : (a / 4) ^ 2 = (b / (2 * Real.pi)) ^ 2 * Real.pi) : a / b = 2 / Real.sqrt Real.pi := by
  sorry

end wire_ratio_l638_63828


namespace find_x_l638_63804

theorem find_x (k : ℝ) (x : ℝ) (h : x ≠ 4) :
  (x = (1 - k) / 2) ↔ ((x^2 - 3 * x - 4) / (x - 4) = 3 * x + k) :=
by sorry

end find_x_l638_63804


namespace sum_divisible_by_100_l638_63827

theorem sum_divisible_by_100 (S : Finset ℤ) (hS : S.card = 200) : 
  ∃ T : Finset ℤ, T ⊆ S ∧ T.card = 100 ∧ (T.sum id) % 100 = 0 := 
  sorry

end sum_divisible_by_100_l638_63827


namespace train_speeds_l638_63853

theorem train_speeds (v t : ℕ) (h1 : t = 1)
  (h2 : v + v * t = 90)
  (h3 : 90 * t = 90) :
  v = 45 := by
  sorry

end train_speeds_l638_63853


namespace problem_statement_l638_63863

-- Problem statement in Lean 4
theorem problem_statement (a b : ℝ) (h : b < a ∧ a < 0) : 7 - a > b :=
by 
  sorry

end problem_statement_l638_63863


namespace points_per_member_l638_63892

theorem points_per_member
  (total_members : ℕ)
  (members_didnt_show : ℕ)
  (total_points : ℕ)
  (H1 : total_members = 14)
  (H2 : members_didnt_show = 7)
  (H3 : total_points = 35) :
  total_points / (total_members - members_didnt_show) = 5 :=
by
  sorry

end points_per_member_l638_63892


namespace second_order_derivative_l638_63817

-- Define the parameterized functions x and y
noncomputable def x (t : ℝ) : ℝ := 1 / t
noncomputable def y (t : ℝ) : ℝ := 1 / (1 + t ^ 2)

-- Define the second-order derivative of y with respect to x
noncomputable def d2y_dx2 (t : ℝ) : ℝ := (2 * (t^2 - 3) * t^4) / (1 + t^2) ^ 3

-- Prove the relationship based on given conditions
theorem second_order_derivative :
  ∀ t : ℝ, (∃ x y : ℝ, x = 1 / t ∧ y = 1 / (1 + t ^ 2)) → 
    (d2y_dx2 t) = (2 * (t^2 - 3) * t^4) / (1 + t^2) ^ 3 :=
by
  intros t ht
  -- Proof omitted
  sorry

end second_order_derivative_l638_63817


namespace jericho_money_left_l638_63851

/--
Given:
1. Twice the money Jericho has is 60.
2. Jericho owes Annika $14.
3. Jericho owes Manny half as much as he owes Annika.

Prove:
Jericho will be left with $9 after paying off all his debts.
-/
theorem jericho_money_left (j_money : ℕ) (annika_owes : ℕ) (manny_multiplier : ℕ) (debt : ℕ) (remaining_money : ℕ) :
  2 * j_money = 60 →
  annika_owes = 14 →
  manny_multiplier = 1 / 2 →
  debt = annika_owes + manny_multiplier * annika_owes →
  remaining_money = j_money - debt →
  remaining_money = 9 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end jericho_money_left_l638_63851


namespace rabbit_speed_final_result_l638_63886

def rabbit_speed : ℕ := 45

def double_speed (speed : ℕ) : ℕ := speed * 2

def add_four (n : ℕ) : ℕ := n + 4

def final_operation : ℕ := double_speed (add_four (double_speed rabbit_speed))

theorem rabbit_speed_final_result : final_operation = 188 := 
by
  sorry

end rabbit_speed_final_result_l638_63886


namespace max_lateral_surface_area_l638_63872

theorem max_lateral_surface_area : ∀ (x y : ℝ), 6 * x + 3 * y = 12 → (3 * x * y) ≤ 6 :=
by
  intros x y h
  have xy_le_2 : x * y ≤ 2 :=
    by
      sorry
  have max_area_6 : 3 * x * y ≤ 6 :=
    by
      sorry
  exact max_area_6

end max_lateral_surface_area_l638_63872


namespace geometric_series_sum_l638_63883

theorem geometric_series_sum :
  let a := 1 / 4
  let r := - (1 / 4)
  ∃ S : ℚ, S = (a * (1 - r^6)) / (1 - r) ∧ S = 4095 / 81920 :=
by
  let a := 1 / 4
  let r := - (1 / 4)
  exists (a * (1 - r^6)) / (1 - r)
  sorry

end geometric_series_sum_l638_63883


namespace find_pairs_l638_63890

theorem find_pairs (p n : ℕ) (hp : Nat.Prime p) (h1 : n ≤ 2 * p) (h2 : n^(p-1) ∣ (p-1)^n + 1) : 
    (p = 2 ∧ n = 2) ∨ (p = 3 ∧ n = 3) ∨ (n = 1) :=
by
  sorry

end find_pairs_l638_63890


namespace g_neither_even_nor_odd_l638_63839

noncomputable def g (x : ℝ) : ℝ := (1 / (3^x - 2)) + 1

theorem g_neither_even_nor_odd : ¬ (∀ x : ℝ, g x = g (-x)) ∧ ¬ (∀ x : ℝ, g x = -g (-x)) := 
by sorry

end g_neither_even_nor_odd_l638_63839


namespace divisibility_equiv_l638_63876

theorem divisibility_equiv (n : ℕ) : (7 ∣ 3^n + n^3) ↔ (7 ∣ 3^n * n^3 + 1) :=
by sorry

end divisibility_equiv_l638_63876


namespace graph_of_equation_is_two_intersecting_lines_l638_63802

theorem graph_of_equation_is_two_intersecting_lines :
  ∀ x y : ℝ, (x + 3 * y) ^ 3 = x ^ 3 + 9 * y ^ 3 ↔ (x = 0 ∨ y = 0 ∨ x + 3 * y = 0) :=
by
  sorry

end graph_of_equation_is_two_intersecting_lines_l638_63802


namespace room_length_l638_63847

-- Defining conditions
def room_height : ℝ := 5
def room_width : ℝ := 7
def door_height : ℝ := 3
def door_width : ℝ := 1
def num_doors : ℝ := 2
def window1_height : ℝ := 1.5
def window1_width : ℝ := 2
def window2_height : ℝ := 1.5
def window2_width : ℝ := 1
def num_window2 : ℝ := 2
def paint_cost_per_sq_m : ℝ := 3
def total_paint_cost : ℝ := 474

-- Defining the problem as a statement to prove x (room length) is 10 meters
theorem room_length {x : ℝ} 
  (H1 : total_paint_cost = paint_cost_per_sq_m * ((2 * (x * room_height) + 2 * (room_width * room_height)) - (num_doors * (door_height * door_width) + (window1_height * window1_width) + num_window2 * (window2_height * window2_width)))) 
  : x = 10 :=
by 
  sorry

end room_length_l638_63847


namespace eugene_initial_pencils_l638_63881

theorem eugene_initial_pencils (e given left : ℕ) (h1 : given = 6) (h2 : left = 45) (h3 : e = given + left) : e = 51 := by
  sorry

end eugene_initial_pencils_l638_63881


namespace final_price_correct_l638_63880

def cost_price : ℝ := 20
def profit_percentage : ℝ := 0.30
def sale_discount_percentage : ℝ := 0.50
def local_tax_percentage : ℝ := 0.10
def packaging_fee : ℝ := 2

def selling_price_before_discount : ℝ := cost_price * (1 + profit_percentage)
def sale_discount : ℝ := sale_discount_percentage * selling_price_before_discount
def price_after_discount : ℝ := selling_price_before_discount - sale_discount
def tax : ℝ := local_tax_percentage * price_after_discount
def price_with_tax : ℝ := price_after_discount + tax
def final_price : ℝ := price_with_tax + packaging_fee

theorem final_price_correct : final_price = 16.30 :=
by
  sorry

end final_price_correct_l638_63880


namespace transform_expression_to_product_l638_63819

open Real

noncomputable def transform_expression (α : ℝ) : ℝ :=
  4.66 * sin (5 * π / 2 + 4 * α) - (sin (5 * π / 2 + 2 * α)) ^ 6 + (cos (7 * π / 2 - 2 * α)) ^ 6

theorem transform_expression_to_product (α : ℝ) :
  transform_expression α = (1 / 8) * sin (4 * α) * sin (8 * α) :=
by
  sorry

end transform_expression_to_product_l638_63819


namespace Jean_calls_thursday_l638_63864

theorem Jean_calls_thursday :
  ∃ (thursday_calls : ℕ), thursday_calls = 61 ∧ 
  (∃ (mon tue wed fri : ℕ),
    mon = 35 ∧ 
    tue = 46 ∧ 
    wed = 27 ∧ 
    fri = 31 ∧ 
    (mon + tue + wed + thursday_calls + fri = 40 * 5)) :=
sorry

end Jean_calls_thursday_l638_63864


namespace find_divisor_l638_63842

theorem find_divisor (d : ℕ) : ((23 = (d * 7) + 2) → d = 3) :=
by
  sorry

end find_divisor_l638_63842
