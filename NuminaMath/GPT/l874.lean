import Mathlib

namespace people_who_own_neither_l874_87470

theorem people_who_own_neither (total_people cat_owners cat_and_dog_owners dog_owners non_cat_dog_owners: ℕ)
        (h1: total_people = 522)
        (h2: 20 * cat_and_dog_owners = cat_owners)
        (h3: 7 * dog_owners = 10 * (dog_owners + cat_and_dog_owners))
        (h4: 2 * non_cat_dog_owners = (non_cat_dog_owners + dog_owners)):
    non_cat_dog_owners = 126 := 
by
  sorry

end people_who_own_neither_l874_87470


namespace quadrilateral_area_l874_87457

theorem quadrilateral_area {d o1 o2 : ℝ} (hd : d = 15) (ho1 : o1 = 6) (ho2 : o2 = 4) :
  (d * (o1 + o2)) / 2 = 75 := by
  sorry

end quadrilateral_area_l874_87457


namespace renu_suma_combined_work_days_l874_87465

theorem renu_suma_combined_work_days :
  (1 / (1 / 8 + 1 / 4.8)) = 3 :=
by
  sorry

end renu_suma_combined_work_days_l874_87465


namespace find_ratio_PS_SR_l874_87414

variable {P Q R S : Type}
variable [MetricSpace P]
variable [MetricSpace Q]
variable [MetricSpace R]
variable [MetricSpace S]

-- Given conditions
variable (PQ QR PR : ℝ)
variable (hPQ : PQ = 6)
variable (hQR : QR = 8)
variable (hPR : PR = 10)
variable (QS : ℝ)
variable (hQS : QS = 6)

-- Points on the segments
variable (PS : ℝ)
variable (SR : ℝ)

-- The theorem to be proven: the ratio PS : SR = 0 : 1
theorem find_ratio_PS_SR (hPQ : PQ = 6) (hQR : QR = 8) (hPR : PR = 10) (hQS : QS = 6) :
    PS = 0 ∧ SR = 10 → PS / SR = 0 :=
by
  sorry

end find_ratio_PS_SR_l874_87414


namespace printer_fraction_l874_87479

noncomputable def basic_computer_price : ℝ := 2000
noncomputable def total_basic_price : ℝ := 2500
noncomputable def printer_price : ℝ := total_basic_price - basic_computer_price -- inferred as 500

noncomputable def enhanced_computer_price : ℝ := basic_computer_price + 500
noncomputable def total_enhanced_price : ℝ := enhanced_computer_price + printer_price -- inferred as 3000

theorem printer_fraction  (h1 : basic_computer_price + printer_price = total_basic_price)
                          (h2 : basic_computer_price = 2000)
                          (h3 : enhanced_computer_price = basic_computer_price + 500) :
  printer_price / total_enhanced_price = 1 / 6 :=
  sorry

end printer_fraction_l874_87479


namespace inequality_proof_l874_87492

theorem inequality_proof (x y : ℝ) (hx : x > -1) (hy : y > -1) (hxy : x + y = 1) : 
    (x / (y + 1) + y / (x + 1) ≥ 2 / 3) := 
  sorry

end inequality_proof_l874_87492


namespace part1_part2_l874_87421

noncomputable def f (x : ℝ) : ℝ := |2 * x - 1| + |2 * x + 3|

theorem part1 (x : ℝ) : f x ≥ 6 ↔ x ≥ 1 ∨ x ≤ -2 := by
  sorry

theorem part2 (a b : ℝ) (m : ℝ) 
  (a_pos : a > 0) (b_pos : b > 0) 
  (fmin : m = 4) 
  (condition : 2 * a * b + a + 2 * b = m) : 
  a + 2 * b = 2 * Real.sqrt 5 - 2 := by
  sorry

end part1_part2_l874_87421


namespace inequality_ge_one_l874_87426

theorem inequality_ge_one {x y z : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by sorry

end inequality_ge_one_l874_87426


namespace abs_neg_one_div_three_l874_87443

open Real

theorem abs_neg_one_div_three : abs (-1 / 3) = 1 / 3 :=
by
  sorry

end abs_neg_one_div_three_l874_87443


namespace sufficient_and_necessary_condition_l874_87419

theorem sufficient_and_necessary_condition {a : ℝ} :
  (∀ x, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≤ 0) ↔ a ≥ 4 :=
sorry

end sufficient_and_necessary_condition_l874_87419


namespace probability_five_cards_one_from_each_suit_and_extra_l874_87441

/--
Given five cards chosen with replacement from a standard 52-card deck, 
the probability of having exactly one card from each suit, plus one 
additional card from any suit, is 3/32.
-/
theorem probability_five_cards_one_from_each_suit_and_extra 
  (cards : ℕ) (total_suits : ℕ)
  (prob_first_diff_suit : ℚ) 
  (prob_second_diff_suit : ℚ) 
  (prob_third_diff_suit : ℚ) 
  (prob_fourth_diff_suit : ℚ) 
  (prob_any_suit : ℚ) 
  (total_prob : ℚ) :
  cards = 5 ∧ total_suits = 4 ∧ 
  prob_first_diff_suit = 3 / 4 ∧ 
  prob_second_diff_suit = 1 / 2 ∧ 
  prob_third_diff_suit = 1 / 4 ∧ 
  prob_fourth_diff_suit = 1 ∧ 
  prob_any_suit = 1 →
  total_prob = 3 / 32 :=
by {
  sorry
}

end probability_five_cards_one_from_each_suit_and_extra_l874_87441


namespace sarah_bottle_caps_total_l874_87412

def initial_caps : ℕ := 450
def first_day_caps : ℕ := 175
def second_day_caps : ℕ := 95
def third_day_caps : ℕ := 220
def total_caps : ℕ := 940

theorem sarah_bottle_caps_total : 
    initial_caps + first_day_caps + second_day_caps + third_day_caps = total_caps :=
by
  sorry

end sarah_bottle_caps_total_l874_87412


namespace product_ge_one_l874_87474

variable (a b : ℝ)
variable (x1 x2 x3 x4 x5 : ℝ)

theorem product_ge_one
  (ha : 0 < a)
  (hb : 0 < b)
  (h_ab : a + b = 1)
  (hx1 : 0 < x1)
  (hx2 : 0 < x2)
  (hx3 : 0 < x3)
  (hx4 : 0 < x4)
  (hx5 : 0 < x5)
  (h_prod_xs : x1 * x2 * x3 * x4 * x5 = 1) :
  (a * x1 + b) * (a * x2 + b) * (a * x3 + b) * (a * x4 + b) * (a * x5 + b) ≥ 1 :=
by
  sorry

end product_ge_one_l874_87474


namespace find_f_2010_l874_87485

open Nat

variable (f : ℕ → ℕ)

axiom strictly_increasing : ∀ m n : ℕ, m < n → f m < f n

axiom function_condition : ∀ n : ℕ, f (f n) = 3 * n

theorem find_f_2010 : f 2010 = 3015 := sorry

end find_f_2010_l874_87485


namespace koala_fiber_l874_87478

theorem koala_fiber (absorption_percent: ℝ) (absorbed_fiber: ℝ) (total_fiber: ℝ) 
  (h1: absorption_percent = 0.25) 
  (h2: absorbed_fiber = 10.5) 
  (h3: absorbed_fiber = absorption_percent * total_fiber) : 
  total_fiber = 42 :=
by
  rw [h1, h2] at h3
  have h : 10.5 = 0.25 * total_fiber := h3
  sorry

end koala_fiber_l874_87478


namespace triangle_interior_angle_leq_60_l874_87467

theorem triangle_interior_angle_leq_60 (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (angle_sum : A + B + C = 180)
  (all_gt_60 : A > 60 ∧ B > 60 ∧ C > 60) :
  false :=
by
  sorry

end triangle_interior_angle_leq_60_l874_87467


namespace sin_270_eq_neg_one_l874_87407

theorem sin_270_eq_neg_one : Real.sin (270 * Real.pi / 180) = -1 := 
by
  sorry

end sin_270_eq_neg_one_l874_87407


namespace solution_form_l874_87489

noncomputable def required_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 < x → 0 < y → f (x * y) ≤ (x * f y + y * f x) / 2

theorem solution_form (f : ℝ → ℝ) (h : ∀ x : ℝ, 0 < x → 0 < f x) : required_function f → ∃ a : ℝ, 0 < a ∧ ∀ x : ℝ, 0 < x → f x = a * x :=
by
  intros
  sorry

end solution_form_l874_87489


namespace boy_usual_time_reach_school_l874_87409

theorem boy_usual_time_reach_school (R T : ℝ) (h : (7 / 6) * R * (T - 3) = R * T) : T = 21 := by
  sorry

end boy_usual_time_reach_school_l874_87409


namespace count_100_digit_numbers_divisible_by_3_l874_87435

def num_100_digit_numbers_divisible_by_3 : ℕ := (4^50 + 2) / 3

theorem count_100_digit_numbers_divisible_by_3 :
  ∃ n : ℕ, n = num_100_digit_numbers_divisible_by_3 :=
by
  use (4^50 + 2) / 3
  sorry

end count_100_digit_numbers_divisible_by_3_l874_87435


namespace stockholm_to_uppsala_distance_l874_87475

-- Definitions based on conditions
def map_distance_cm : ℝ := 3
def scale_cm_to_km : ℝ := 80

-- Theorem statement based on the question and correct answer
theorem stockholm_to_uppsala_distance : 
  (map_distance_cm * scale_cm_to_km = 240) :=
by 
  -- This is where the proof would go
  sorry

end stockholm_to_uppsala_distance_l874_87475


namespace part_I_part_II_l874_87473

def f (x : ℝ) (m : ℕ) : ℝ := |x - m| + |x|

theorem part_I (m : ℕ) (hm : m = 1) : ∃ x : ℝ, f x m < 2 :=
by sorry

theorem part_II (α β : ℝ) (hα : 1 < α) (hβ : 1 < β) (h : f α 1 + f β 1 = 2) :
  (4 / α) + (1 / β) ≥ 9 / 2 :=
by sorry

end part_I_part_II_l874_87473


namespace cats_left_l874_87423

def initial_siamese_cats : ℕ := 12
def initial_house_cats : ℕ := 20
def cats_sold : ℕ := 20

theorem cats_left : (initial_siamese_cats + initial_house_cats - cats_sold) = 12 :=
by
sorry

end cats_left_l874_87423


namespace batsman_running_percentage_l874_87447

theorem batsman_running_percentage (total_runs boundary_runs six_runs : ℕ) 
  (h1 : total_runs = 120) (h2 : boundary_runs = 3 * 4) (h3 : six_runs = 8 * 6) : 
  (total_runs - (boundary_runs + six_runs)) * 100 / total_runs = 50 := 
sorry

end batsman_running_percentage_l874_87447


namespace connie_correct_answer_l874_87466

theorem connie_correct_answer 
  (x : ℝ) 
  (h1 : 2 * x = 80) 
  (correct_ans : ℝ := x / 3) :
  correct_ans = 40 / 3 :=
by
  sorry

end connie_correct_answer_l874_87466


namespace area_of_stripe_l874_87494

def cylindrical_tank.diameter : ℝ := 40
def cylindrical_tank.height : ℝ := 100
def green_stripe.width : ℝ := 4
def green_stripe.revolutions : ℝ := 3

theorem area_of_stripe :
  let diameter := cylindrical_tank.diameter
  let height := cylindrical_tank.height
  let width := green_stripe.width
  let revolutions := green_stripe.revolutions
  let circumference := Real.pi * diameter
  let length := revolutions * circumference
  let area := length * width
  area = 480 * Real.pi := by
  sorry

end area_of_stripe_l874_87494


namespace discount_rate_on_pony_jeans_is_15_l874_87400

noncomputable def discountProblem : Prop :=
  ∃ (F P : ℝ),
    (15 * 3 * F / 100 + 18 * 2 * P / 100 = 8.55) ∧ 
    (F + P = 22) ∧ 
    (P = 15)

theorem discount_rate_on_pony_jeans_is_15 : discountProblem :=
sorry

end discount_rate_on_pony_jeans_is_15_l874_87400


namespace monomial_exponents_l874_87499

theorem monomial_exponents (m n : ℕ) 
  (h1 : m + 1 = 3)
  (h2 : n - 1 = 3) : 
  m^n = 16 := by
  sorry

end monomial_exponents_l874_87499


namespace base_six_to_base_ten_equivalent_l874_87436

theorem base_six_to_base_ten_equivalent :
  let n := 12345
  (5 * 6^0 + 4 * 6^1 + 3 * 6^2 + 2 * 6^3 + 1 * 6^4) = 1865 :=
by
  sorry

end base_six_to_base_ten_equivalent_l874_87436


namespace problem_l874_87460

variable (g : ℝ → ℝ)
variables (x y : ℝ)

noncomputable def cond1 : Prop := ∀ x y : ℝ, 0 < x → 0 < y → g (x^2 * y) = g x / y^2
noncomputable def cond2 : Prop := g 800 = 4

-- The statement to be proved
theorem problem (h1 : cond1 g) (h2 : cond2 g) : g 7200 = 4 / 81 :=
by
  sorry

end problem_l874_87460


namespace geometric_sequence_strictly_increasing_iff_l874_87480

noncomputable def geometric_sequence (a_1 q : ℝ) (n : ℕ) : ℝ :=
  a_1 * q^(n-1)

theorem geometric_sequence_strictly_increasing_iff (a_1 q : ℝ) :
  (∀ n : ℕ, geometric_sequence a_1 q (n+2) > geometric_sequence a_1 q n) ↔ 
  (∀ n : ℕ, geometric_sequence a_1 q (n+1) > geometric_sequence a_1 q n) := 
by
  sorry

end geometric_sequence_strictly_increasing_iff_l874_87480


namespace cost_difference_of_dolls_proof_l874_87487

-- Define constants
def cost_large_doll : ℝ := 7
def total_spent : ℝ := 350
def additional_dolls : ℝ := 20

-- Define the function for the cost of small dolls
def cost_small_doll (S : ℝ) : Prop :=
  total_spent / S = total_spent / cost_large_doll + additional_dolls

-- The statement given the conditions and solving for the difference in cost
theorem cost_difference_of_dolls_proof : 
  ∃ S, cost_small_doll S ∧ (cost_large_doll - S = 2) :=
by
  sorry

end cost_difference_of_dolls_proof_l874_87487


namespace exponential_function_passes_through_01_l874_87449

theorem exponential_function_passes_through_01 (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : (a^0 = 1) :=
by
  sorry

end exponential_function_passes_through_01_l874_87449


namespace remainder_when_s_div_6_is_5_l874_87453

theorem remainder_when_s_div_6_is_5 (s t : ℕ) (h1 : s > t) (Rs Rt : ℕ) (h2 : s % 6 = Rs) (h3 : t % 6 = Rt) (h4 : (s - t) % 6 = 5) : Rs = 5 := 
by
  sorry

end remainder_when_s_div_6_is_5_l874_87453


namespace product_of_solutions_l874_87429

theorem product_of_solutions (x : ℝ) (hx : |x - 5| - 5 = 0) :
  ∃ a b : ℝ, (|a - 5| - 5 = 0 ∧ |b - 5| - 5 = 0) ∧ a * b = 0 := by
  sorry

end product_of_solutions_l874_87429


namespace sin_135_eq_sqrt2_div_2_l874_87418

theorem sin_135_eq_sqrt2_div_2 :
  Real.sin (135 * Real.pi / 180) = (Real.sqrt 2) / 2 := 
sorry

end sin_135_eq_sqrt2_div_2_l874_87418


namespace impossible_coins_l874_87462

theorem impossible_coins (p1 p2 : ℝ) : 
  ¬ ((1 - p1) * (1 - p2) = p1 * p2 ∧ p1 * p2 = p1 * (1 - p2) + p2 * (1 - p1)) :=
by
  sorry

end impossible_coins_l874_87462


namespace games_planned_to_attend_this_month_l874_87451

theorem games_planned_to_attend_this_month (T A_l P_l M_l P_m : ℕ) 
  (h1 : T = 12) 
  (h2 : P_l = 17) 
  (h3 : M_l = 16) 
  (h4 : A_l = P_l - M_l) 
  (h5 : T = A_l + P_m) : P_m = 11 :=
by 
  sorry

end games_planned_to_attend_this_month_l874_87451


namespace proof_no_solution_l874_87424

noncomputable def no_solution (a b : ℕ) : Prop :=
  2 * a^2 + 1 ≠ 4 * b^2

theorem proof_no_solution (a b : ℕ) : no_solution a b := by
  sorry

end proof_no_solution_l874_87424


namespace avg_salary_officers_correct_l874_87482

def total_employees := 465
def avg_salary_employees := 120
def non_officers := 450
def avg_salary_non_officers := 110
def officers := 15

theorem avg_salary_officers_correct : (15 * 420) = ((total_employees * avg_salary_employees) - (non_officers * avg_salary_non_officers)) := by
  sorry

end avg_salary_officers_correct_l874_87482


namespace polygon_sides_l874_87464

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 1080) : n = 8 :=
by
  sorry

end polygon_sides_l874_87464


namespace trigonometric_identity_tangent_line_l874_87401

theorem trigonometric_identity_tangent_line 
  (α : ℝ) 
  (h_tan : Real.tan α = 4) 
  : Real.cos α ^ 2 - Real.sin (2 * α) = - 7 / 17 := 
by sorry

end trigonometric_identity_tangent_line_l874_87401


namespace find_a_l874_87491

noncomputable def unique_quad_solution (a : ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1^2 - a * x1 + a = 1 → x2^2 - a * x2 + a = 1 → x1 = x2

theorem find_a (a : ℝ) (h : unique_quad_solution a) : a = 2 :=
sorry

end find_a_l874_87491


namespace sufficient_but_not_necessary_condition_l874_87455

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≤ 0) ↔ (a ≥ 5) :=
by
  sorry

end sufficient_but_not_necessary_condition_l874_87455


namespace bianca_points_per_bag_l874_87498

theorem bianca_points_per_bag (total_bags : ℕ) (not_recycled : ℕ) (total_points : ℕ) 
  (h1 : total_bags = 17) 
  (h2 : not_recycled = 8) 
  (h3 : total_points = 45) : 
  total_points / (total_bags - not_recycled) = 5 :=
by
  sorry 

end bianca_points_per_bag_l874_87498


namespace arithmetic_sequence_sixtieth_term_l874_87428

theorem arithmetic_sequence_sixtieth_term (a₁ a₂₁ a₆₀ d : ℕ) 
  (h1 : a₁ = 7)
  (h2 : a₂₁ = 47)
  (h3 : a₂₁ = a₁ + 20 * d) : 
  a₆₀ = a₁ + 59 * d := 
  by
  have HD : d = 2 := by 
    rw [h1] at h3
    rw [h2] at h3
    linarith
  rw [HD]
  rw [h1]
  sorry

end arithmetic_sequence_sixtieth_term_l874_87428


namespace axis_of_symmetry_eq_l874_87444

theorem axis_of_symmetry_eq : 
  ∃ k : ℤ, (λ x => 2 * Real.cos (2 * x)) = (λ x => 2 * Real.sin (2 * (x + π / 3) - π / 6)) ∧
            x = (1/2) * k * π ∧ x = -π / 2 := 
by
  sorry

end axis_of_symmetry_eq_l874_87444


namespace jacks_walking_rate_l874_87411

theorem jacks_walking_rate :
  let distance := 8
  let time_in_minutes := 1 * 60 + 15
  let time := time_in_minutes / 60.0
  let rate := distance / time
  rate = 6.4 :=
by
  sorry

end jacks_walking_rate_l874_87411


namespace largest_possible_median_l874_87410

theorem largest_possible_median (l : List ℕ) (h1 : l.length = 10) 
  (h2 : ∀ x ∈ l, 0 < x) (exists6l : ∃ l1 : List ℕ, l1 = [3, 4, 5, 7, 8, 9]) :
  ∃ median_val : ℝ, median_val = 8.5 := 
sorry

end largest_possible_median_l874_87410


namespace squirrel_walnut_count_l874_87497

-- Lean 4 statement
theorem squirrel_walnut_count :
  let initial_boy_walnuts := 12
  let gathered_walnuts := 6
  let dropped_walnuts := 1
  let initial_girl_walnuts := 0
  let brought_walnuts := 5
  let eaten_walnuts := 2
  (initial_boy_walnuts + gathered_walnuts - dropped_walnuts + initial_girl_walnuts + brought_walnuts - eaten_walnuts) = 20 :=
by
  -- Proof goes here
  sorry

end squirrel_walnut_count_l874_87497


namespace khalil_dogs_l874_87493

theorem khalil_dogs (D : ℕ) (cost_dog cost_cat : ℕ) (num_cats total_cost : ℕ) 
  (h1 : cost_dog = 60)
  (h2 : cost_cat = 40)
  (h3 : num_cats = 60)
  (h4 : total_cost = 3600) :
  (num_cats * cost_cat + D * cost_dog = total_cost) → D = 20 :=
by
  intros h
  sorry

end khalil_dogs_l874_87493


namespace value_of_expression_l874_87404

def delta (a b : ℕ) : ℕ := a * a - b

theorem value_of_expression :
  delta (5 ^ (delta 6 17)) (2 ^ (delta 7 11)) = 5 ^ 38 - 2 ^ 38 :=
by
  sorry

end value_of_expression_l874_87404


namespace ap_80th_term_l874_87495

/--
If the sum of the first 20 terms of an arithmetic progression is 200,
and the sum of the first 60 terms is 180, then the 80th term is -573/40.
-/
theorem ap_80th_term (S : ℤ → ℚ) (a d : ℚ)
  (h1 : S 20 = 200)
  (h2 : S 60 = 180)
  (hS : ∀ n, S n = n / 2 * (2 * a + (n - 1) * d)) :
  a + 79 * d = -573 / 40 :=
by {
  sorry
}

end ap_80th_term_l874_87495


namespace right_triangle_side_length_l874_87458

theorem right_triangle_side_length (r f : ℝ) (h : f < 2 * r) :
  let c := 2 * r
  let a := (f / (4 * c)) * (f + Real.sqrt (f^2 + 8 * c^2))
  a = (f / (4 * (2 * r))) * (f + Real.sqrt (f^2 + 8 * (2 * r)^2)) :=
by
  let c := 2 * r
  let a := (f / (4 * c)) * (f + Real.sqrt (f^2 + 8 * c^2))
  have acalc : a = (f / (4 * (2 * r))) * (f + Real.sqrt (f^2 + 8 * (2 * r)^2)) := by sorry
  exact acalc

end right_triangle_side_length_l874_87458


namespace non_powers_of_a_meet_condition_l874_87439

-- Definitions used directly from the conditions detailed in the problem:
def Sa (a x : ℕ) : ℕ := sorry -- S_{a}(x): sum of the digits of x in base a
def Fa (a x : ℕ) : ℕ := sorry -- F_{a}(x): number of digits of x in base a
def fa (a x : ℕ) : ℕ := sorry -- f_{a}(x): position of the first non-zero digit from the right in base a

theorem non_powers_of_a_meet_condition (a M : ℕ) (h₁: a > 1) (h₂ : M ≥ 2020) :
  ∀ n : ℕ, (n > 0) → (∀ k : ℕ, (k > 0) → (Sa a (k * n) = Sa a n ∧ Fa a (k * n) - fa a (k * n) > M)) ↔ (∃ α : ℕ, n = a ^ α) :=
sorry

end non_powers_of_a_meet_condition_l874_87439


namespace zeros_of_f_l874_87433

noncomputable def f (a : ℝ) (x : ℝ) :=
if x ≤ 1 then a + 2^x else (1/2) * x + a

theorem zeros_of_f (a : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0) ↔ a ∈ Set.Ico (-2) (-1/2) :=
sorry

end zeros_of_f_l874_87433


namespace total_area_of_strips_l874_87483

def strip1_length := 12
def strip1_width := 1
def strip2_length := 8
def strip2_width := 2
def num_strips1 := 2
def num_strips2 := 2
def overlap_area_per_strip := 2
def num_overlaps := 4
def total_area_covered := 48

theorem total_area_of_strips : 
  num_strips1 * (strip1_length * strip1_width) + 
  num_strips2 * (strip2_length * strip2_width) - 
  num_overlaps * overlap_area_per_strip = total_area_covered := sorry

end total_area_of_strips_l874_87483


namespace first_pipe_time_l874_87405

noncomputable def pool_filling_time (T : ℝ) : Prop :=
  (1 / T + 1 / 12 = 1 / 4.8) → (T = 8)

theorem first_pipe_time :
  ∃ T : ℝ, pool_filling_time T := by
  use 8
  sorry

end first_pipe_time_l874_87405


namespace correct_proportion_l874_87477

theorem correct_proportion {a b c x y : ℝ} 
  (h1 : x + y = b)
  (h2 : x * c = y * a) :
  y / a = b / (a + c) :=
sorry

end correct_proportion_l874_87477


namespace complex_number_quadrant_l874_87496

noncomputable def complex_quadrant : ℂ → String
| z => if z.re > 0 ∧ z.im > 0 then "First quadrant"
      else if z.re < 0 ∧ z.im > 0 then "Second quadrant"
      else if z.re < 0 ∧ z.im < 0 then "Third quadrant"
      else if z.re > 0 ∧ z.im < 0 then "Fourth quadrant"
      else "On the axis"

theorem complex_number_quadrant (z : ℂ) (h : z = (5 : ℂ) / (2 + I)) : complex_quadrant z = "Fourth quadrant" :=
by
  sorry

end complex_number_quadrant_l874_87496


namespace no_a_satisfies_condition_l874_87438

noncomputable def M : Set ℝ := {0, 1}
noncomputable def N (a : ℝ) : Set ℝ := {11 - a, Real.log a / Real.log 1, 2^a, a}

theorem no_a_satisfies_condition :
  ¬ ∃ a : ℝ, M ∩ N a = {1} :=
by
  sorry

end no_a_satisfies_condition_l874_87438


namespace complement_union_eq_l874_87408

namespace SetComplementUnion

-- Defining the universal set U, set M and set N.
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x < 2}
def N : Set ℝ := {x | -1 < x ∧ x < 3}

-- Proving the desired equality
theorem complement_union_eq :
  (U \ M) ∪ N = {x | x > -1} :=
sorry

end SetComplementUnion

end complement_union_eq_l874_87408


namespace arithmetic_expression_l874_87488

theorem arithmetic_expression :
  4 * 6 * 8 + 24 / 4 - 2^3 = 190 := by
  sorry

end arithmetic_expression_l874_87488


namespace max_coconuts_needed_l874_87427

theorem max_coconuts_needed (goats : ℕ) (coconuts_per_crab : ℕ) (crabs_per_goat : ℕ) 
  (final_goats : ℕ) : 
  goats = 19 ∧ coconuts_per_crab = 3 ∧ crabs_per_goat = 6 →
  ∃ coconuts, coconuts = 342 :=
by
  sorry

end max_coconuts_needed_l874_87427


namespace min_value_of_sum_l874_87484

theorem min_value_of_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 4 / x + 1 / y = 1) : x + y = 9 :=
by
  -- sorry used to skip the proof
  sorry

end min_value_of_sum_l874_87484


namespace rationalize_denominator_sum_l874_87425

theorem rationalize_denominator_sum :
  let A := -4
  let B := 7
  let C := 3
  let D := 13
  let E := 1
  A + B + C + D + E = 20 := by
    sorry

end rationalize_denominator_sum_l874_87425


namespace calculate_green_paint_l874_87476

theorem calculate_green_paint {green white : ℕ} (ratio_white_to_green : 5 * green = 3 * white) (use_white_paint : white = 15) : green = 9 :=
by
  sorry

end calculate_green_paint_l874_87476


namespace weight_of_new_person_l874_87472

-- Definition of the problem
def average_weight_increases (W : ℝ) (N : ℝ) : Prop :=
  let increase := 2.5
  W - 45 + N = W + 8 * increase

-- The main statement we need to prove
theorem weight_of_new_person (W : ℝ) : ∃ N, average_weight_increases W N ∧ N = 65 := 
by
  use 65
  unfold average_weight_increases
  sorry

end weight_of_new_person_l874_87472


namespace arithmetic_geometric_sequence_l874_87430

theorem arithmetic_geometric_sequence (x y z : ℤ) :
  (x ≠ y ∧ y ≠ z ∧ x ≠ z) ∧
  ((x + y + z = 6) ∧ (y - x = z - y) ∧ (y^2 = x * z)) →
  (x = -4 ∧ y = 2 ∧ z = 8 ∨ x = 8 ∧ y = 2 ∧ z = -4) :=
by
  intros h
  sorry

end arithmetic_geometric_sequence_l874_87430


namespace students_in_ms_delmont_class_l874_87448

-- Let us define the necessary conditions

def total_cupcakes : Nat := 40
def students_mrs_donnelly_class : Nat := 16
def adults_count : Nat := 4 -- Ms. Delmont, Mrs. Donnelly, the school nurse, and the school principal
def leftover_cupcakes : Nat := 2

-- Define the number of students in Ms. Delmont's class
def students_ms_delmont_class : Nat := 18

-- The statement to prove
theorem students_in_ms_delmont_class :
  total_cupcakes - adults_count - students_mrs_donnelly_class - leftover_cupcakes = students_ms_delmont_class :=
by
  sorry

end students_in_ms_delmont_class_l874_87448


namespace max_value_l874_87416

noncomputable def max_expression (x : ℝ) : ℝ :=
  3^x - 2 * 9^x

theorem max_value : ∃ x : ℝ, max_expression x = 1 / 8 :=
sorry

end max_value_l874_87416


namespace probability_product_positive_correct_l874_87481

noncomputable def probability_product_positive : ℚ :=
  let length_total := 45
  let length_negative := 30
  let length_positive := 15
  let prob_negative := (length_negative : ℚ) / length_total
  let prob_positive := (length_positive : ℚ) / length_total
  let prob_product_positive := prob_negative^2 + prob_positive^2
  prob_product_positive

theorem probability_product_positive_correct :
  probability_product_positive = 5 / 9 :=
by
  sorry

end probability_product_positive_correct_l874_87481


namespace add_like_terms_l874_87434

variable (a : ℝ)

theorem add_like_terms : a^2 + 2 * a^2 = 3 * a^2 := 
by sorry

end add_like_terms_l874_87434


namespace arithmetic_sequence_initial_term_l874_87437

theorem arithmetic_sequence_initial_term (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ)
  (h_seq : ∀ n, a (n + 1) = a n + d)
  (h_sum : ∀ n, S n = n * (a 1 + n * d / 2))
  (h_product : a 2 * a 3 = a 4 * a 5)
  (h_sum_9 : S 9 = 27)
  (h_d_nonzero : d ≠ 0) :
  a 1 = -5 :=
sorry

end arithmetic_sequence_initial_term_l874_87437


namespace mass_percentage_O_in_N2O3_l874_87471

variable (m_N : ℝ := 14.01)  -- Molar mass of nitrogen (N) in g/mol
variable (m_O : ℝ := 16.00)  -- Molar mass of oxygen (O) in g/mol
variable (n_N : ℕ := 2)      -- Number of nitrogen (N) atoms in N2O3
variable (n_O : ℕ := 3)      -- Number of oxygen (O) atoms in N2O3

theorem mass_percentage_O_in_N2O3 :
  let molar_mass_N2O3 := (n_N * m_N) + (n_O * m_O)
  let mass_O_in_N2O3 := n_O * m_O
  let percentage_O := (mass_O_in_N2O3 / molar_mass_N2O3) * 100
  percentage_O = 63.15 :=
by
  -- Formal proof here
  sorry

end mass_percentage_O_in_N2O3_l874_87471


namespace arrangement_books_l874_87406

def combination (n k : ℕ) := n.factorial / (k.factorial * (n - k).factorial)

theorem arrangement_books : combination 9 4 = 126 := by
  sorry

end arrangement_books_l874_87406


namespace transition_term_l874_87486

theorem transition_term (k : ℕ) : (2 * k + 2) + (2 * k + 3) = (2 * (k + 1) + 1) + (2 * k + 2) :=
by
  sorry

end transition_term_l874_87486


namespace donuts_left_for_coworkers_l874_87403

theorem donuts_left_for_coworkers :
  ∀ (total_donuts gluten_free regular gluten_free_chocolate gluten_free_plain regular_chocolate regular_plain consumed_gluten_free consumed_regular afternoon_gluten_free_chocolate afternoon_gluten_free_plain afternoon_regular_chocolate afternoon_regular_plain left_gluten_free_chocolate left_gluten_free_plain left_regular_chocolate left_regular_plain),
  total_donuts = 30 →
  gluten_free = 12 →
  regular = 18 →
  gluten_free_chocolate = 6 →
  gluten_free_plain = 6 →
  regular_chocolate = 11 →
  regular_plain = 7 →
  consumed_gluten_free = 1 →
  consumed_regular = 1 →
  afternoon_gluten_free_chocolate = 2 →
  afternoon_gluten_free_plain = 1 →
  afternoon_regular_chocolate = 2 →
  afternoon_regular_plain = 1 →
  left_gluten_free_chocolate = gluten_free_chocolate - consumed_gluten_free * 0.5 - afternoon_gluten_free_chocolate →
  left_gluten_free_plain = gluten_free_plain - consumed_gluten_free * 0.5 - afternoon_gluten_free_plain →
  left_regular_chocolate = regular_chocolate - consumed_regular * 1 - afternoon_regular_chocolate →
  left_regular_plain = regular_plain - consumed_regular * 0 - afternoon_regular_plain →
  left_gluten_free_chocolate + left_gluten_free_plain + left_regular_chocolate + left_regular_plain = 23 :=
by
  intros
  sorry

end donuts_left_for_coworkers_l874_87403


namespace circle_center_radius_l874_87459

theorem circle_center_radius (x y : ℝ) :
  x^2 + y^2 - 4 * x + 2 * y - 4 = 0 ↔ (x - 2)^2 + (y + 1)^2 = 3 :=
by
  sorry

end circle_center_radius_l874_87459


namespace sum_of_consecutive_odds_mod_16_l874_87431

theorem sum_of_consecutive_odds_mod_16 :
  (12001 + 12003 + 12005 + 12007 + 12009 + 12011 + 12013) % 16 = 1 :=
by
  sorry

end sum_of_consecutive_odds_mod_16_l874_87431


namespace allan_balloons_l874_87422

def jak_balloons : ℕ := 11
def diff_balloons : ℕ := 6

theorem allan_balloons (jake_allan_diff : jak_balloons = diff_balloons + 5) : jak_balloons - diff_balloons = 5 :=
by
  sorry

end allan_balloons_l874_87422


namespace total_saplings_l874_87445

theorem total_saplings (a_efficiency b_efficiency : ℝ) (A B T n : ℝ) 
  (h1 : a_efficiency = (3/4))
  (h2 : b_efficiency = 1)
  (h3 : B = n + 36)
  (h4 : T = 2 * n + 36)
  (h5 : n * (4/3) = n + 36)
  : T = 252 :=
by {
  sorry
}

end total_saplings_l874_87445


namespace compute_expression_l874_87440

theorem compute_expression : 9 * (1 / 13) * 26 = 18 :=
by
  sorry

end compute_expression_l874_87440


namespace rectangle_area_l874_87450

theorem rectangle_area (a b c: ℝ) (h₁ : a = 7.1) (h₂ : b = 8.9) (h₃ : c = 10.0) (L W: ℝ)
  (h₄ : L = 2 * W) (h₅ : 2 * (L + W) = a + b + c) : L * W = 37.54 :=
by
  sorry

end rectangle_area_l874_87450


namespace height_of_cone_l874_87420

theorem height_of_cone (e : ℝ) (bA : ℝ) (v : ℝ) :
  e = 6 ∧ bA = 54 ∧ v = e^3 → ∃ h : ℝ, (1/3) * bA * h = v ∧ h = 12 := by
  sorry

end height_of_cone_l874_87420


namespace simplify_expression_l874_87442

noncomputable def term1 : ℝ := 3 / (Real.sqrt 2 + 2)
noncomputable def term2 : ℝ := 4 / (Real.sqrt 5 - 2)
noncomputable def simplifiedExpression : ℝ := 1 / (term1 + term2)
noncomputable def finalExpression : ℝ := 1 / (11 + 4 * Real.sqrt 5 - 3 * Real.sqrt 2 / 2)

theorem simplify_expression : simplifiedExpression = finalExpression := by
  sorry

end simplify_expression_l874_87442


namespace find_d_l874_87413

noncomputable def area_triangle (base height : ℝ) : ℝ := (1 / 2) * base * height

theorem find_d (d : ℝ) (h₁ : 0 ≤ d ∧ d ≤ 2) (h₂ : 6 - ((1 / 2) * (2 - d) * 2) = 2 * ((1 / 2) * (2 - d) * 2)) : 
  d = 0 :=
sorry

end find_d_l874_87413


namespace find_taller_tree_height_l874_87432

-- Define the known variables and conditions
variables (H : ℕ) (ratio : ℚ) (difference : ℕ)

-- Specify the conditions from the problem
def taller_tree_height (H difference : ℕ) := H
def shorter_tree_height (H difference : ℕ) := H - difference
def height_ratio (H : ℕ) (ratio : ℚ) (difference : ℕ) :=
  (shorter_tree_height H difference : ℚ) / (taller_tree_height H difference : ℚ) = ratio

-- Prove the height of the taller tree given the conditions
theorem find_taller_tree_height (H : ℕ) (h_ratio : height_ratio H (2/3) 20) : 
  taller_tree_height H 20 = 60 :=
  sorry

end find_taller_tree_height_l874_87432


namespace sandy_grew_watermelons_l874_87468

-- Definitions for the conditions
def jason_grew_watermelons : ℕ := 37
def total_watermelons : ℕ := 48

-- Define what we want to prove
theorem sandy_grew_watermelons : total_watermelons - jason_grew_watermelons = 11 := by
  sorry

end sandy_grew_watermelons_l874_87468


namespace simplify_division_l874_87469

theorem simplify_division (x : ℝ) : 2 * x^8 / x^4 = 2 * x^4 := 
by sorry

end simplify_division_l874_87469


namespace pennies_to_quarters_ratio_l874_87417

-- Define the given conditions as assumptions
variables (pennies dimes nickels quarters: ℕ)

-- Given conditions
axiom cond1 : dimes = pennies + 10
axiom cond2 : nickels = 2 * dimes
axiom cond3 : quarters = 4
axiom cond4 : nickels = 100

-- Theorem stating the final result should be a certain ratio
theorem pennies_to_quarters_ratio (hpn : pennies = 40) : pennies / quarters = 10 := 
by sorry

end pennies_to_quarters_ratio_l874_87417


namespace total_carrots_l874_87446

theorem total_carrots (sandy_carrots: Nat) (sam_carrots: Nat) (h1: sandy_carrots = 6) (h2: sam_carrots = 3) : sandy_carrots + sam_carrots = 9 :=
by
  sorry

end total_carrots_l874_87446


namespace bread_calories_l874_87402

theorem bread_calories (total_calories : Nat) (pb_calories : Nat) (pb_servings : Nat) (bread_pieces : Nat) (bread_calories : Nat)
  (h1 : total_calories = 500)
  (h2 : pb_calories = 200)
  (h3 : pb_servings = 2)
  (h4 : bread_pieces = 1)
  (h5 : total_calories = pb_servings * pb_calories + bread_pieces * bread_calories) : 
  bread_calories = 100 :=
by
  sorry

end bread_calories_l874_87402


namespace positive_solution_in_interval_l874_87452

def quadratic (x : ℝ) := x^2 + 3 * x - 5

theorem positive_solution_in_interval :
  ∃ x : ℝ, 1 < x ∧ x < 2 ∧ quadratic x = 0 :=
sorry

end positive_solution_in_interval_l874_87452


namespace remainder_is_zero_l874_87490

theorem remainder_is_zero :
  (86 * 87 * 88 * 89 * 90 * 91 * 92) % 7 = 0 := 
by 
  sorry

end remainder_is_zero_l874_87490


namespace problem_statement_l874_87454

theorem problem_statement :
  ∀ x a k n : ℤ, 
  (3 * x + 2) * (2 * x - 7) = a * x^2 + k * x + n → a - n + k = 3 :=
by  
  sorry

end problem_statement_l874_87454


namespace train_speed_in_kmh_l874_87415

-- Definitions of conditions
def time_to_cross_platform := 30  -- in seconds
def time_to_cross_man := 17  -- in seconds
def length_of_platform := 260  -- in meters

-- Conversion factor from m/s to km/h
def meters_per_second_to_kilometers_per_hour (v : ℕ) : ℕ :=
  v * 36 / 10

-- The theorem statement
theorem train_speed_in_kmh :
  (∃ (L V : ℕ),
    L = V * time_to_cross_man ∧
    L + length_of_platform = V * time_to_cross_platform ∧
    meters_per_second_to_kilometers_per_hour V = 72) :=
sorry

end train_speed_in_kmh_l874_87415


namespace fixed_point_1_3_l874_87456

noncomputable def fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : Prop :=
  (f (1) = 3) where f x := a^(x-1) + 2

theorem fixed_point_1_3 (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : fixed_point a h1 h2 :=
by
  unfold fixed_point
  sorry

end fixed_point_1_3_l874_87456


namespace scientists_nobel_greater_than_not_nobel_by_three_l874_87461

-- Definitions of the given conditions
def total_scientists := 50
def wolf_prize_laureates := 31
def nobel_prize_laureates := 25
def wolf_and_nobel_laureates := 14

-- Derived quantities
def no_wolf_prize := total_scientists - wolf_prize_laureates
def only_wolf_prize := wolf_prize_laureates - wolf_and_nobel_laureates
def only_nobel_prize := nobel_prize_laureates - wolf_and_nobel_laureates
def nobel_no_wolf := only_nobel_prize
def no_wolf_no_nobel := no_wolf_prize - nobel_no_wolf
def difference := nobel_no_wolf - no_wolf_no_nobel

-- The theorem to be proved
theorem scientists_nobel_greater_than_not_nobel_by_three :
  difference = 3 := 
sorry

end scientists_nobel_greater_than_not_nobel_by_three_l874_87461


namespace exist_two_numbers_with_GCD_and_LCM_l874_87463

def GCD (a b : ℕ) : ℕ := Nat.gcd a b
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

theorem exist_two_numbers_with_GCD_and_LCM :
  ∃ A B : ℕ, GCD A B = 21 ∧ LCM A B = 3969 ∧ ((A = 21 ∧ B = 3969) ∨ (A = 147 ∧ B = 567)) :=
by
  sorry

end exist_two_numbers_with_GCD_and_LCM_l874_87463
