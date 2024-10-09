import Mathlib

namespace max_pieces_in_8x8_grid_l300_30085

theorem max_pieces_in_8x8_grid : 
  ∃ m n : ℕ, (m = 8) ∧ (n = 9) ∧ 
  (∀ H V : ℕ, (H ≤ n) → (V ≤ n) → 
   (H + V + 1 ≤ 16)) := sorry

end max_pieces_in_8x8_grid_l300_30085


namespace math_problem_l300_30054

def foo (a b : ℝ) (h : a + b > 0) : Prop :=
  (a^5 * b^2 + a^4 * b^3 ≥ 0) ∧
  ¬ (a^4 * b^3 + a^3 * b^4 ≥ 0) ∧
  (a^21 + b^21 > 0) ∧
  ((a + 2) * (b + 2) > a * b) ∧
  ¬ ((a - 3) * (b - 3) < a * b) ∧
  ¬ ((a + 2) * (b + 3) > a * b + 5)

theorem math_problem (a b : ℝ) (h : a + b > 0) : foo a b h :=
by
  -- The proof will be here
  sorry

end math_problem_l300_30054


namespace number_of_BMWs_sold_l300_30097

-- Defining the percentages of Mercedes, Toyota, and Acura cars sold
def percentageMercedes : ℕ := 18
def percentageToyota  : ℕ := 25
def percentageAcura   : ℕ := 15

-- Defining the total number of cars sold
def totalCars : ℕ := 250

-- The theorem to be proved
theorem number_of_BMWs_sold : (totalCars * (100 - (percentageMercedes + percentageToyota + percentageAcura)) / 100) = 105 := by
  sorry -- Proof to be filled in later

end number_of_BMWs_sold_l300_30097


namespace hall_ratio_l300_30095

variable (w l : ℝ)

theorem hall_ratio
  (h1 : w * l = 200)
  (h2 : l - w = 10) :
  w / l = 1 / 2 := 
by
  sorry

end hall_ratio_l300_30095


namespace gopi_servant_salary_l300_30010

theorem gopi_servant_salary (S : ℝ) (h1 : 9 / 12 * S + 110 = 150) : S = 200 :=
by
  sorry

end gopi_servant_salary_l300_30010


namespace race_time_l300_30076

theorem race_time (t : ℝ) (h1 : 100 / t = 66.66666666666667 / 45) : t = 67.5 :=
by
  sorry

end race_time_l300_30076


namespace adam_deleted_items_l300_30080

theorem adam_deleted_items (initial_items deleted_items remaining_items : ℕ)
  (h1 : initial_items = 100) (h2 : remaining_items = 20) 
  (h3 : remaining_items = initial_items - deleted_items) : 
  deleted_items = 80 :=
by
  sorry

end adam_deleted_items_l300_30080


namespace time_to_fill_by_B_l300_30060

/-- 
Assume a pool with two taps, A and B, fills in 30 minutes when both are open.
When both are open for 10 minutes, and then only B is open for another 40 minutes, the pool fills up.
Prove that if only tap B is opened, it would take 60 minutes to fill the pool.
-/
theorem time_to_fill_by_B
  (r_A r_B : ℝ)
  (H1 : (r_A + r_B) * 30 = 1)
  (H2 : ((r_A + r_B) * 10 + r_B * 40) = 1) :
  1 / r_B = 60 :=
by
  sorry

end time_to_fill_by_B_l300_30060


namespace corresponding_angles_equal_l300_30086

-- Define what it means for two angles to be corresponding angles
def corresponding_angles (a b : ℝ) : Prop :=
  -- Hypothetical definition
  sorry

-- Lean 4 statement of the problem
theorem corresponding_angles_equal (a b : ℝ) (h : corresponding_angles a b) : a = b :=
by
  sorry

end corresponding_angles_equal_l300_30086


namespace triangle_non_existence_no_solution_max_value_expression_l300_30099

-- Define sides and angles
variables {A B C : ℝ} -- Angles of the triangle
variables {a b c : ℝ} -- Corresponding opposite sides

-- Define the triangle conditions
def triangle_sides_angles (a b c A B C : ℝ) : Prop := 
  (a^2 = (1 - Real.cos A) / (1 - Real.cos B)) ∧ 
  (b = 1) ∧ 
  -- Additional properties ensuring we have a valid triangle can be added here
  (A ≠ B) -- Non-isosceles condition (equivalent to angles being different).

-- Prove non-existence under given conditions
theorem triangle_non_existence_no_solution (h : triangle_sides_angles a b c A B C) : false := 
sorry 

-- Define the maximization problem
theorem max_value_expression (h : a^2 = (1 - Real.cos A) / (1 - Real.cos B)) : 
(∃ b c, (b = 1) → ∀ a, a > 0 → (c > 0) ∧ ((1/c) * (1/b - 1/a)) ≤ (3 - 2 * Real.sqrt 2)) := 
sorry

end triangle_non_existence_no_solution_max_value_expression_l300_30099


namespace angle_bisector_equation_intersection_l300_30037

noncomputable def slope_of_angle_bisector (m1 m2 : ℝ) : ℝ :=
  (m1 + m2 - Real.sqrt (1 + m1^2 + m2^2)) / (1 - m1 * m2)

noncomputable def equation_of_angle_bisector (x : ℝ) : ℝ :=
  (Real.sqrt 21 - 6) / 7 * x

theorem angle_bisector_equation_intersection :
  let m1 := 2
  let m2 := 4
  slope_of_angle_bisector m1 m2 = (Real.sqrt 21 - 6) / 7 ∧
  equation_of_angle_bisector 1 = (Real.sqrt 21 - 6) / 7 :=
by
  sorry

end angle_bisector_equation_intersection_l300_30037


namespace inequality_solution_l300_30005

theorem inequality_solution (x y z : ℝ) (h1 : x + 3 * y + 2 * z = 6) :
  (z = 3 - 1/2 * x - 3/2 * y) ∧ (x - 2)^2 + (3 * y - 2)^2 ≤ 4 ∧ 0 ≤ x ∧ x ≤ 4 :=
sorry

end inequality_solution_l300_30005


namespace card_d_total_percent_change_l300_30090

noncomputable def card_d_initial_value : ℝ := 250
noncomputable def card_d_percent_changes : List ℝ := [0.05, -0.15, 0.30, -0.10, 0.20]

noncomputable def final_value (initial_value : ℝ) (changes : List ℝ) : ℝ :=
  changes.foldl (λ acc change => acc * (1 + change)) initial_value

theorem card_d_total_percent_change :
  let final_val := final_value card_d_initial_value card_d_percent_changes
  let total_percent_change := ((final_val - card_d_initial_value) / card_d_initial_value) * 100
  total_percent_change = 25.307 := by
  sorry

end card_d_total_percent_change_l300_30090


namespace arithmetic_series_sum_l300_30035

theorem arithmetic_series_sum (k : ℤ) : 
  let a₁ := k^2 + k + 1 
  let n := 2 * k + 3 
  let d := 1 
  let aₙ := a₁ + (n - 1) * d 
  let S_n := n / 2 * (a₁ + aₙ)
  S_n = 2 * k^3 + 7 * k^2 + 10 * k + 6 := 
by {
  sorry
}

end arithmetic_series_sum_l300_30035


namespace log_relationship_l300_30063

theorem log_relationship :
  let a := Real.logb 0.3 0.2
  let b := Real.logb 3 2
  let c := Real.logb 0.2 3
  c < b ∧ b < a :=
by
  let a := Real.logb 0.3 0.2
  let b := Real.logb 3 2
  let c := Real.logb 0.2 3
  sorry

end log_relationship_l300_30063


namespace ratio_lcm_gcf_eq_55_l300_30034

theorem ratio_lcm_gcf_eq_55 : 
  ∀ (a b : ℕ), a = 210 → b = 462 →
  (Nat.lcm a b / Nat.gcd a b) = 55 :=
by
  intros a b ha hb
  rw [ha, hb]
  sorry

end ratio_lcm_gcf_eq_55_l300_30034


namespace cost_of_bananas_and_cantaloupe_l300_30006

-- Define prices for different items
variables (a b c d e : ℝ)

-- Define the conditions as hypotheses
theorem cost_of_bananas_and_cantaloupe (h1 : a + b + c + d + e = 30)
    (h2 : d = 3 * a) (h3 : c = a - b) (h4 : e = a + b) :
    b + c = 5 := 
by 
  -- Initial proof setup
  sorry

end cost_of_bananas_and_cantaloupe_l300_30006


namespace fraction_to_decimal_l300_30073

-- We define the fraction and its simplified form
def fraction : ℚ := 58 / 160
def simplified_fraction : ℚ := 29 / 80

-- We state that the fraction simplifies correctly
lemma simplify_fraction : fraction = simplified_fraction := by
  sorry

-- Define the factorization of the denominator
def denominator_factorization : ℕ := 2^4 * 5

-- Verify the fraction when multiplied by 125/125
def equalized_fraction : ℚ := 29 * 125 / 10000

-- State the final result as a decimal
theorem fraction_to_decimal : fraction = 0.3625 := by
  sorry

end fraction_to_decimal_l300_30073


namespace find_f1_plus_gneg1_l300_30043

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- Conditions
axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom g_even : ∀ x : ℝ, g (-x) = g x
axiom relation : ∀ x : ℝ, f x - g x = (1 / 2) ^ x

-- Proof statement
theorem find_f1_plus_gneg1 : f 1 + g (-1) = -2 :=
by
  -- Proof goes here
  sorry

end find_f1_plus_gneg1_l300_30043


namespace gcd_765432_654321_l300_30016

theorem gcd_765432_654321 :
  Int.gcd 765432 654321 = 3 := 
sorry

end gcd_765432_654321_l300_30016


namespace ratio_of_shares_l300_30075

theorem ratio_of_shares 
    (sheila_share : ℕ → ℕ)
    (rose_share : ℕ)
    (total_rent : ℕ) 
    (h1 : ∀ P, sheila_share P = 5 * P)
    (h2 : rose_share = 1800)
    (h3 : ∀ P, sheila_share P + P + rose_share = total_rent) 
    (h4 : total_rent = 5400) :
    ∃ P, 1800 / P = 3 := 
by 
  sorry

end ratio_of_shares_l300_30075


namespace system_solution_l300_30040

theorem system_solution :
  ∀ (a1 b1 c1 a2 b2 c2 : ℝ),
  (a1 * 8 + b1 * 5 = c1) ∧ (a2 * 8 + b2 * 5 = c2) →
  ∃ (x y : ℝ), (4 * a1 * x - 5 * b1 * y = 3 * c1) ∧ (4 * a2 * x - 5 * b2 * y = 3 * c2) ∧ 
               (x = 6) ∧ (y = -3) :=
by
  sorry

end system_solution_l300_30040


namespace remainder_when_divided_by_29_l300_30067

theorem remainder_when_divided_by_29 (N : ℤ) (h : N % 899 = 63) : N % 29 = 10 :=
sorry

end remainder_when_divided_by_29_l300_30067


namespace hilary_total_kernels_l300_30048

-- Define the conditions given in the problem
def ears_per_stalk : ℕ := 4
def total_stalks : ℕ := 108
def kernels_per_ear_first_half : ℕ := 500
def additional_kernels_second_half : ℕ := 100

-- Express the main problem as a theorem in Lean
theorem hilary_total_kernels : 
  let total_ears := ears_per_stalk * total_stalks
  let half_ears := total_ears / 2
  let kernels_first_half := half_ears * kernels_per_ear_first_half
  let kernels_per_ear_second_half := kernels_per_ear_first_half + additional_kernels_second_half
  let kernels_second_half := half_ears * kernels_per_ear_second_half
  kernels_first_half + kernels_second_half = 237600 :=
by
  sorry

end hilary_total_kernels_l300_30048


namespace mike_gave_12_pears_l300_30015

variable (P M K N : ℕ)

def initial_pears := 46
def pears_given_to_keith := 47
def pears_left := 11

theorem mike_gave_12_pears (M : ℕ) : 
  initial_pears - pears_given_to_keith + M = pears_left → M = 12 :=
by
  intro h
  sorry

end mike_gave_12_pears_l300_30015


namespace birds_on_branch_l300_30039

theorem birds_on_branch (initial_parrots remaining_parrots remaining_crows total_birds : ℕ) (h₁ : initial_parrots = 7) (h₂ : remaining_parrots = 2) (h₃ : remaining_crows = 1) (h₄ : initial_parrots - remaining_parrots = total_birds - remaining_crows - initial_parrots) : total_birds = 13 :=
sorry

end birds_on_branch_l300_30039


namespace ab_leq_one_l300_30042

theorem ab_leq_one (a b : ℝ) (h : (a + b) * (a + b + a + b) = 9) : a * b ≤ 1 := 
  sorry

end ab_leq_one_l300_30042


namespace initial_pigeons_l300_30030

theorem initial_pigeons (n : ℕ) (h : n + 1 = 2) : n = 1 := 
sorry

end initial_pigeons_l300_30030


namespace intersection_A_B_l300_30078

-- Defining sets A and B based on the given conditions.
def A : Set ℝ := {x | ∃ y, y = Real.log x ∧ x > 0}
def B : Set ℝ := {x | x^2 - 2 * x - 3 < 0}

-- Stating the theorem that A ∩ B = {x | 0 < x ∧ x < 3}.
theorem intersection_A_B : A ∩ B = {x | 0 < x ∧ x < 3} :=
by
  sorry

end intersection_A_B_l300_30078


namespace ellipse_condition_l300_30093

theorem ellipse_condition (m : ℝ) : 
  (∃ x y : ℝ, m * (x^2 + y^2 + 2*y + 1) = (x - 2*y + 3)^2) → m > 5 :=
by
  intro h
  sorry

end ellipse_condition_l300_30093


namespace line_tangent_to_ellipse_l300_30027

theorem line_tangent_to_ellipse (m : ℝ) (a : ℝ) (b : ℝ) (h_a : a = 3) (h_b : b = 1) :
  m^2 = 1 / 3 := by
  sorry

end line_tangent_to_ellipse_l300_30027


namespace math_problem_l300_30047

open Real

theorem math_problem
  (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : a - b + c = 0) :
  (a^2 * b^2 / ((a^2 + b * c) * (b^2 + a * c)) +
   a^2 * c^2 / ((a^2 + b * c) * (c^2 + a * b)) +
   b^2 * c^2 / ((b^2 + a * c) * (c^2 + a * b))) = 1 := by
  sorry

end math_problem_l300_30047


namespace percentage_deficit_of_second_side_l300_30012

theorem percentage_deficit_of_second_side
  (L W : Real)
  (h1 : ∃ (L' : Real), L' = 1.16 * L)
  (h2 : ∃ (W' : Real), (L' * W') = 1.102 * (L * W))
  (h3 : ∃ (x : Real), W' = W * (1 - x / 100)) :
  x = 5 := 
  sorry

end percentage_deficit_of_second_side_l300_30012


namespace combined_weight_chihuahua_pitbull_greatdane_l300_30077

noncomputable def chihuahua_pitbull_greatdane_combined_weight (C P G : ℕ) : ℕ :=
  C + P + G

theorem combined_weight_chihuahua_pitbull_greatdane :
  ∀ (C P G : ℕ), P = 3 * C → G = 3 * P + 10 → G = 307 → chihuahua_pitbull_greatdane_combined_weight C P G = 439 :=
by
  intros C P G h1 h2 h3
  sorry

end combined_weight_chihuahua_pitbull_greatdane_l300_30077


namespace average_age_increase_l300_30033

theorem average_age_increase (average_age_students : ℕ) (num_students : ℕ) (teacher_age : ℕ) (new_avg_age : ℕ)
                             (h1 : average_age_students = 26) (h2 : num_students = 25) (h3 : teacher_age = 52)
                             (h4 : new_avg_age = (650 + teacher_age) / (num_students + 1))
                             (h5 : 650 = average_age_students * num_students) :
  new_avg_age - average_age_students = 1 := 
by
  sorry

end average_age_increase_l300_30033


namespace find_height_of_cuboid_l300_30044

-- Definitions and given conditions
def length : ℕ := 22
def width : ℕ := 30
def total_edges : ℕ := 224

-- Proof statement
theorem find_height_of_cuboid (h : ℕ) (H : 4 * length + 4 * width + 4 * h = total_edges) : h = 4 :=
by
  sorry

end find_height_of_cuboid_l300_30044


namespace circle_and_tangent_lines_l300_30092

open Real

noncomputable def equation_of_circle_center_on_line (x y : ℝ) : Prop :=
  ∃ (a b : ℝ), (x - a)^2 + (y - (a + 1))^2 = 2 ∧ (a = 4) ∧ (b = 5)

noncomputable def tangent_line_through_point (x y : ℝ) : Prop :=
  y = x - 1 ∨ y = (23 / 7) * x - (23 / 7)

theorem circle_and_tangent_lines :
  (∃ (a b : ℝ), (a = 4) ∧ (b = 5) ∧ (∀ x y : ℝ, equation_of_circle_center_on_line x y)) ∧
  (∀ x y : ℝ, tangent_line_through_point x y) := 
  by
  sorry

end circle_and_tangent_lines_l300_30092


namespace leak_empty_time_l300_30020

variable (inlet_rate : ℕ := 6) -- litres per minute
variable (total_capacity : ℕ := 12960) -- litres
variable (empty_time_with_inlet_open : ℕ := 12) -- hours

def inlet_rate_per_hour := inlet_rate * 60 -- litres per hour
def net_emptying_rate := total_capacity / empty_time_with_inlet_open -- litres per hour
def leak_rate := net_emptying_rate + inlet_rate_per_hour -- litres per hour

theorem leak_empty_time : total_capacity / leak_rate = 9 := by
  sorry

end leak_empty_time_l300_30020


namespace ellipse_major_axis_length_l300_30068

-- Conditions
def cylinder_radius : ℝ := 2
def minor_axis (r : ℝ) := 2 * r
def major_axis (minor: ℝ) := minor + 0.6 * minor

-- Problem
theorem ellipse_major_axis_length :
  major_axis (minor_axis cylinder_radius) = 6.4 :=
by
  sorry

end ellipse_major_axis_length_l300_30068


namespace inequality_proof_l300_30025

theorem inequality_proof
  (a b c : ℝ)
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_sum : a + b + c = 1) :
  (a^2 + b^2 + c^2) * ((a / (b + c)) + (b / (a + c)) + (c / (a + b))) ≥ 1/2 := by
  sorry

end inequality_proof_l300_30025


namespace sum_of_two_longest_altitudes_l300_30018

theorem sum_of_two_longest_altitudes (a b c : ℕ) (h : a^2 + b^2 = c^2) (h1: a = 7) (h2: b = 24) (h3: c = 25) : 
  (a + b = 31) :=
by {
  sorry
}

end sum_of_two_longest_altitudes_l300_30018


namespace shara_shells_after_vacation_l300_30004

-- Definitions based on conditions
def initial_shells : ℕ := 20
def shells_per_day : ℕ := 5
def days : ℕ := 3
def shells_fourth_day : ℕ := 6

-- Statement of the proof problem
theorem shara_shells_after_vacation : 
  initial_shells + (shells_per_day * days) + shells_fourth_day = 41 := by
  sorry

end shara_shells_after_vacation_l300_30004


namespace Emily_GRE_Exam_Date_l300_30059

theorem Emily_GRE_Exam_Date : 
  ∃ (exam_date : ℕ) (exam_month : String), 
  exam_date = 5 ∧ exam_month = "September" ∧
  ∀ study_days break_days start_day_cycles start_break_cycles start_month_june total_days S_june_remaining S_remaining_july S_remaining_august September_start_day, 
    study_days = 15 ∧ 
    break_days = 5 ∧ 
    start_day_cycles = 5 ∧ 
    start_break_cycles = 4 ∧ 
    start_month_june = 1 ∧
    total_days = start_day_cycles * study_days + start_break_cycles * break_days ∧ 
    S_june_remaining = 30 - start_month_june ∧ 
    S_remaining = total_days - S_june_remaining ∧ 
    S_remaining_july = S_remaining - 31 ∧ 
    S_remaining_august = S_remaining_july - 31 ∧ 
    September_start_day = S_remaining_august + 1 ∧
    exam_date = September_start_day ∧ 
    exam_month = "September" := by 
  sorry

end Emily_GRE_Exam_Date_l300_30059


namespace remainder_102_104_plus_6_div_9_l300_30079

theorem remainder_102_104_plus_6_div_9 :
  ((102 * 104 + 6) % 9) = 3 :=
by
  sorry

end remainder_102_104_plus_6_div_9_l300_30079


namespace ellipse_m_range_l300_30045

theorem ellipse_m_range (m : ℝ) 
  (h1 : m + 9 > 25 - m) 
  (h2 : 25 - m > 0) 
  (h3 : m + 9 > 0) : 
  8 < m ∧ m < 25 := 
by
  sorry

end ellipse_m_range_l300_30045


namespace find_linear_function_l300_30023

noncomputable def functional_equation (f : ℝ → ℝ) : Prop :=
(∀ (a b c : ℝ), a + b + c ≥ 0 → f (a^3) + f (b^3) + f (c^3) ≥ 3 * f (a * b * c))
∧ (∀ (a b c : ℝ), a + b + c ≤ 0 → f (a^3) + f (b^3) + f (c^3) ≤ 3 * f (a * b * c))

theorem find_linear_function (f : ℝ → ℝ) (h : functional_equation f) : ∃ c : ℝ, ∀ x : ℝ, f x = c * x :=
sorry

end find_linear_function_l300_30023


namespace initial_selling_price_l300_30007

theorem initial_selling_price (P : ℝ) : 
    (∀ (c_i c_m p_m r : ℝ),
        c_i = 3 ∧
        c_m = 20 ∧
        p_m = 4 ∧
        r = 50 ∧
        (15 * P + 5 * p_m - 20 * c_i = r)
    ) → 
    P = 6 := by 
    sorry

end initial_selling_price_l300_30007


namespace additional_flowers_grew_l300_30096

-- Define the initial conditions
def initial_flowers : ℕ := 10  -- Dane’s two daughters planted 5 flowers each (5 + 5).
def flowers_died : ℕ := 10     -- 10 flowers died.
def baskets : ℕ := 5
def flowers_per_basket : ℕ := 4

-- Total flowers harvested (from the baskets)
def total_harvested : ℕ := baskets * flowers_per_basket  -- 5 * 4 = 20

-- The proof to show additional flowers grown
theorem additional_flowers_grew : (total_harvested - initial_flowers + flowers_died) = 10 :=
by
  -- The final number of flowers and the initial number of flowers are known
  have final_flowers : ℕ := total_harvested
  have initial_plus_grown : ℕ := initial_flowers + (total_harvested - initial_flowers)
  -- Show the equality that defines the additional flowers grown
  show (total_harvested - initial_flowers + flowers_died) = 10
  sorry

end additional_flowers_grew_l300_30096


namespace quadratic_distinct_real_roots_l300_30082

theorem quadratic_distinct_real_roots (k : ℝ) :
  (∀ x : ℝ, k * x^2 - 2 * x - 1 = 0 → 
  (k ≠ 0 ∧ ((-2)^2 - 4 * k * (-1) > 0))) ↔ (k > -1 ∧ k ≠ 0) := 
sorry

end quadratic_distinct_real_roots_l300_30082


namespace Melies_money_left_l300_30050

variable (meat_weight : ℕ)
variable (meat_cost_per_kg : ℕ)
variable (initial_money : ℕ)

def money_left_after_purchase (meat_weight : ℕ) (meat_cost_per_kg : ℕ) (initial_money : ℕ) : ℕ :=
  initial_money - (meat_weight * meat_cost_per_kg)

theorem Melies_money_left : 
  money_left_after_purchase 2 82 180 = 16 :=
by
  sorry

end Melies_money_left_l300_30050


namespace parallel_line_through_point_l300_30056

theorem parallel_line_through_point (x y c : ℝ) (h1 : c = -1) :
  ∃ c, (x-2*y+c = 0 ∧ x = 1 ∧ y = 0) ∧ ∃ k b, k = 1 ∧ b = -2 ∧ k*x-2*y+b=0 → c = -1 := by
  sorry

end parallel_line_through_point_l300_30056


namespace dwarfs_truthful_count_l300_30009

theorem dwarfs_truthful_count :
  ∃ (T L : ℕ), T + L = 10 ∧
    (∀ t : ℕ, t = 10 → t + ((10 - T) * 2 - T) = 16) ∧
    T = 4 :=
by
  sorry

end dwarfs_truthful_count_l300_30009


namespace ball_hits_ground_time_l300_30062

theorem ball_hits_ground_time :
  ∃ t : ℝ, -16 * t^2 - 30 * t + 180 = 0 ∧ t = 1.25 := by
  sorry

end ball_hits_ground_time_l300_30062


namespace pentagon_perimeter_even_l300_30074

noncomputable def dist_sq (A B : ℤ × ℤ) : ℤ :=
  (A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2

theorem pentagon_perimeter_even (A B C D E : ℤ × ℤ) (h1 : dist_sq A B % 2 = 1) (h2 : dist_sq B C % 2 = 1) 
  (h3 : dist_sq C D % 2 = 1) (h4 : dist_sq D E % 2 = 1) (h5 : dist_sq E A % 2 = 1) : 
  (dist_sq A B + dist_sq B C + dist_sq C D + dist_sq D E + dist_sq E A) % 2 = 0 := 
by 
  sorry

end pentagon_perimeter_even_l300_30074


namespace jackie_eligible_for_free_shipping_l300_30002

def shampoo_cost : ℝ := 2 * 12.50
def conditioner_cost : ℝ := 3 * 15.00
def face_cream_cost : ℝ := 20.00  -- Considering the buy-one-get-one-free deal

def subtotal : ℝ := shampoo_cost + conditioner_cost + face_cream_cost
def discount : ℝ := 0.10 * subtotal
def total_after_discount : ℝ := subtotal - discount

theorem jackie_eligible_for_free_shipping : total_after_discount >= 75 := by
  sorry

end jackie_eligible_for_free_shipping_l300_30002


namespace evaluate_g_at_5_l300_30001

noncomputable def g (x : ℝ) : ℝ := 2 * x ^ 4 - 15 * x ^ 3 + 24 * x ^ 2 - 18 * x - 72

theorem evaluate_g_at_5 : g 5 = -7 := by
  sorry

end evaluate_g_at_5_l300_30001


namespace find_m_and_star_l300_30070

-- Definitions from conditions
def star (x y m : ℚ) : ℚ := (x * y) / (m * x + 2 * y)

-- Given conditions
def given_star (x y : ℚ) (m : ℚ) : Prop := star x y m = 2 / 5

-- Target: Proving m = 1 and 2 * 6 = 6 / 7 given the conditions
theorem find_m_and_star :
  ∀ m : ℚ, 
  (given_star 1 2 m) → 
  (m = 1 ∧ star 2 6 m = 6 / 7) := 
sorry

end find_m_and_star_l300_30070


namespace profit_calculation_l300_30051

theorem profit_calculation
  (P : ℝ)
  (h1 : 9 > 0)  -- condition that there are 9 employees
  (h2 : 0 < 0.10 ∧ 0.10 < 1) -- 10 percent profit is between 0 and 100%
  (h3 : 5 > 0)  -- condition that each employee gets $5
  (h4 : 9 * 5 = 45) -- total amount distributed among employees
  (h5 : 0.90 * P = 45) -- remaining profit to be distributed
  : P = 50 :=
sorry

end profit_calculation_l300_30051


namespace solve_system_eqn_l300_30057

theorem solve_system_eqn :
  ∃ x y : ℚ, 7 * x = -9 - 3 * y ∧ 2 * x = 5 * y - 30 ∧ x = -135 / 41 ∧ y = 192 / 41 :=
by 
  sorry

end solve_system_eqn_l300_30057


namespace total_balloons_correct_l300_30094

-- Define the number of balloons each person has
def dan_balloons : ℕ := 29
def tim_balloons : ℕ := 7 * dan_balloons
def molly_balloons : ℕ := 5 * dan_balloons

-- Define the total number of balloons
def total_balloons : ℕ := dan_balloons + tim_balloons + molly_balloons

-- The theorem to prove
theorem total_balloons_correct : total_balloons = 377 :=
by
  -- This part is where the proof will go
  sorry

end total_balloons_correct_l300_30094


namespace ratio_of_books_l300_30098

theorem ratio_of_books (longest_pages : ℕ) (middle_pages : ℕ) (shortest_pages : ℕ) :
  longest_pages = 396 ∧ middle_pages = 297 ∧ shortest_pages = longest_pages / 4 →
  (middle_pages / shortest_pages = 3) :=
by
  intros h
  obtain ⟨h_longest, h_middle, h_shortest⟩ := h
  sorry

end ratio_of_books_l300_30098


namespace calculate_expression_l300_30089

theorem calculate_expression :
  18 - ((-16) / (2 ^ 3)) = 20 :=
by
  sorry

end calculate_expression_l300_30089


namespace problem1_problem2_l300_30028

open Nat

def seq (a : ℕ → ℕ) :=
  ∀ n : ℕ, n > 0 → a n < a (n + 1) ∧ a n > 0

def b_seq (a : ℕ → ℕ) (n : ℕ) :=
  a (a n)

def c_seq (a : ℕ → ℕ) (n : ℕ) :=
  a (a n + 1)

theorem problem1 (a : ℕ → ℕ) (h_seq : seq a) (h_bseq : ∀ n, n > 0 → b_seq a n = 3 * n) : a 1 = 2 ∧ c_seq a 1 = 6 :=
  sorry

theorem problem2 (a : ℕ → ℕ) (h_seq : seq a) (h_cseq : ∀ n, n > 0 → c_seq a (n + 1) - c_seq a n = 1) : 
  ∀ n, n > 0 → a (n + 1) - a n = 1 :=
  sorry

end problem1_problem2_l300_30028


namespace cans_of_chili_beans_ordered_l300_30008

theorem cans_of_chili_beans_ordered (T C : ℕ) (h1 : 2 * T = C) (h2 : T + C = 12) : C = 8 := by
  sorry

end cans_of_chili_beans_ordered_l300_30008


namespace problem1_problem2_l300_30066

-- Problem 1
theorem problem1 :
  (1 : ℝ) * (2 * Real.sqrt 12 - (1 / 2) * Real.sqrt 18) - (Real.sqrt 75 - (1 / 4) * Real.sqrt 32)
  = -Real.sqrt 3 - (Real.sqrt 2) / 2 :=
by
  sorry

-- Problem 2
theorem problem2 :
  (2 : ℝ) * (Real.sqrt 5 + 2) * (Real.sqrt 5 - 2) + Real.sqrt 48 / (2 * Real.sqrt (1 / 2)) - Real.sqrt 30 / Real.sqrt 5
  = 1 + Real.sqrt 6 :=
by
  sorry

end problem1_problem2_l300_30066


namespace roots_ellipse_condition_l300_30088

theorem roots_ellipse_condition (m n : ℝ) : 
  (∃ x1 x2 : ℝ, x1 > 0 ∧ x2 > 0 ∧ x1^2 - m*x1 + n = 0 ∧ x2^2 - m*x2 + n = 0) 
  ↔ (m > 0 ∧ n > 0 ∧ m ≠ n) :=
sorry

end roots_ellipse_condition_l300_30088


namespace probability_both_asian_selected_probability_A1_but_not_B1_selected_l300_30014

noncomputable def choose (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_both_asian_selected (A1 A2 A3 B1 B2 B3 : Prop) :
  let total_ways := choose 6 2
  let asian_ways := choose 3 2
  asian_ways / total_ways = 1 / 5 := by
  let total_ways := choose 6 2
  let asian_ways := choose 3 2
  sorry

theorem probability_A1_but_not_B1_selected (A1 A2 A3 B1 B2 B3 : Prop) :
  let total_ways := 9
  let valid_ways := 2
  valid_ways / total_ways = 2 / 9 := by
  let total_ways := 9
  let valid_ways := 2
  sorry

end probability_both_asian_selected_probability_A1_but_not_B1_selected_l300_30014


namespace smallest_n_7770_l300_30084

theorem smallest_n_7770 (n : ℕ) 
  (h1 : ∀ d ∈ n.digits 10, d = 0 ∨ d = 7)
  (h2 : 15 ∣ n) : 
  n = 7770 := 
sorry

end smallest_n_7770_l300_30084


namespace largest_of_seven_consecutive_numbers_l300_30091

theorem largest_of_seven_consecutive_numbers (a b c d e f g : ℤ) (h1 : a + 1 = b)
                                             (h2 : b + 1 = c) (h3 : c + 1 = d)
                                             (h4 : d + 1 = e) (h5 : e + 1 = f)
                                             (h6 : f + 1 = g)
                                             (h_avg : (a + b + c + d + e + f + g) / 7 = 20) :
    g = 23 :=
by
  sorry

end largest_of_seven_consecutive_numbers_l300_30091


namespace claudia_candle_choices_l300_30083

-- Claudia can choose 4 different candles
def num_candles : ℕ := 4

-- Claudia can choose 8 out of 9 different flowers
def num_ways_to_choose_flowers : ℕ := Nat.choose 9 8

-- The total number of groupings is given as 54
def total_groupings : ℕ := 54

-- Prove the main theorem using the conditions
theorem claudia_candle_choices :
  num_ways_to_choose_flowers = 9 ∧ num_ways_to_choose_flowers * C = total_groupings → C = 6 :=
by
  sorry

end claudia_candle_choices_l300_30083


namespace prove_inequality_l300_30055

variable (f : ℝ → ℝ)

def isEvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def isMonotonicOnInterval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≥ f y

theorem prove_inequality
  (h1 : isEvenFunction f)
  (h2 : isMonotonicOnInterval f 0 5)
  (h3 : f (-3) < f 1) :
  f 0 > f 1 :=
sorry

end prove_inequality_l300_30055


namespace max_value_of_f_l300_30046

def f (x : ℝ) : ℝ := x^2 - 2 * x - 5

theorem max_value_of_f : ∃ x ∈ (Set.Icc (-2:ℝ) 2), ∀ y ∈ (Set.Icc (-2:ℝ) 2), f y ≤ f x ∧ f x = 3 := by
  sorry

end max_value_of_f_l300_30046


namespace plane_contains_points_l300_30064

def point := (ℝ × ℝ × ℝ)

def is_plane (A B C D : ℝ) (p : point) : Prop :=
  ∃ x y z, p = (x, y, z) ∧ A * x + B * y + C * z + D = 0

theorem plane_contains_points :
  ∃ A B C D : ℤ,
    A > 0 ∧
    Int.gcd (Int.natAbs A) (Int.gcd (Int.natAbs B) (Int.gcd (Int.natAbs C) (Int.natAbs D))) = 1 ∧
    is_plane A B C D (2, -1, 3) ∧
    is_plane A B C D (0, -1, 5) ∧
    is_plane A B C D (-2, -3, 4) ∧
    A = 2 ∧ B = 5 ∧ C = -2 ∧ D = 7 :=
  sorry

end plane_contains_points_l300_30064


namespace problem_condition_l300_30038

theorem problem_condition (x y : ℝ) (h : x^2 + y^2 - x * y = 1) : 
  x + y ≥ -2 ∧ x^2 + y^2 ≤ 2 :=
by
  sorry

end problem_condition_l300_30038


namespace solve_xy_l300_30017

theorem solve_xy (x y : ℝ) (hx: x ≠ 0) (hxy: x + y ≠ 0) : 
  (x + y) / x = 2 * y / (x + y) + 1 → (x = y ∨ x = -3 * y) := 
by 
  intros h 
  sorry

end solve_xy_l300_30017


namespace range_of_a_l300_30072

theorem range_of_a (a : ℝ) :
  (∃ x y : ℝ, x^2 + 4 * (y - a)^2 = 4 ∧ x^2 = 2 * y) ↔ -1 ≤ a ∧ a ≤ 17 / 8 :=
by
  sorry

end range_of_a_l300_30072


namespace slope_of_line_l300_30022

theorem slope_of_line (a b c : ℝ) (h : 3 * a = 4 * b - 9) : a = 4 / 3 * b - 3 :=
by
  sorry

end slope_of_line_l300_30022


namespace lucy_reads_sixty_pages_l300_30087

-- Define the number of pages Carter, Lucy, and Oliver can read in an hour.
def pages_carter : ℕ := 30
def pages_oliver : ℕ := 40

-- Carter reads half as many pages as Lucy.
def reads_half_as_much_as (a b : ℕ) : Prop := a = b / 2

-- Lucy reads more pages than Oliver.
def reads_more_than (a b : ℕ) : Prop := a > b

-- The goal is to show that Lucy can read 60 pages in an hour.
theorem lucy_reads_sixty_pages (pages_lucy : ℕ) (h1 : reads_half_as_much_as pages_carter pages_lucy)
  (h2 : reads_more_than pages_lucy pages_oliver) : pages_lucy = 60 :=
sorry

end lucy_reads_sixty_pages_l300_30087


namespace total_books_in_series_l300_30021

-- Definitions for the conditions
def books_read : ℕ := 8
def books_to_read : ℕ := 6

-- Statement to be proved
theorem total_books_in_series : books_read + books_to_read = 14 := by
  sorry

end total_books_in_series_l300_30021


namespace symmetric_line_equation_l300_30029

theorem symmetric_line_equation 
  (l1 : ∀ x y : ℝ, x - 2 * y - 2 = 0) 
  (l2 : ∀ x y : ℝ, x + y = 0) : 
  ∀ x y : ℝ, 2 * x - y - 2 = 0 :=
sorry

end symmetric_line_equation_l300_30029


namespace cuboid_volume_l300_30026

/-- Define the ratio condition for the dimensions of the cuboid. -/
def ratio (l w h : ℕ) : Prop :=
  (∃ x : ℕ, l = 2*x ∧ w = x ∧ h = 3*x)

/-- Define the total surface area condition for the cuboid. -/
def surface_area (l w h sa : ℕ) : Prop :=
  2*(l*w + l*h + w*h) = sa

/-- Volume of the cuboid given the ratio and surface area conditions. -/
theorem cuboid_volume (l w h : ℕ) (sa : ℕ) (h_ratio : ratio l w h) (h_surface : surface_area l w h sa) :
  ∃ v : ℕ, v = l * w * h ∧ v = 48 :=
by
  sorry

end cuboid_volume_l300_30026


namespace bobby_candy_left_l300_30011

def initial_candy := 22
def eaten_candy1 := 9
def eaten_candy2 := 5

theorem bobby_candy_left : initial_candy - eaten_candy1 - eaten_candy2 = 8 :=
by
  sorry

end bobby_candy_left_l300_30011


namespace elena_earnings_l300_30058

theorem elena_earnings (hourly_wage : ℝ) (hours_worked : ℝ) (h_wage : hourly_wage = 13.25) (h_hours : hours_worked = 4) : 
  hourly_wage * hours_worked = 53.00 := by
sorry

end elena_earnings_l300_30058


namespace fx_solution_l300_30024

theorem fx_solution (f : ℝ → ℝ) (x : ℝ) (h₀ : x ≠ 0) (h₁ : x ≠ 1)
  (h_assumption : f (1 / x) = x / (1 - x)) : f x = 1 / (x - 1) :=
by
  sorry

end fx_solution_l300_30024


namespace area_diff_of_rectangle_l300_30071

theorem area_diff_of_rectangle (a : ℝ) : 
  let length_increased := 1.40 * a
  let breadth_increased := 1.30 * a
  let original_area := a * a
  let new_area := length_increased * breadth_increased
  (new_area - original_area) = 0.82 * (a * a) :=
by 
sorry

end area_diff_of_rectangle_l300_30071


namespace faster_train_cross_time_l300_30031

noncomputable def time_to_cross (speed_fast_kmph : ℝ) (speed_slow_kmph : ℝ) (length_fast_m : ℝ) : ℝ :=
  let speed_diff_kmph := speed_fast_kmph - speed_slow_kmph
  let speed_diff_mps := (speed_diff_kmph * 1000) / 3600
  length_fast_m / speed_diff_mps

theorem faster_train_cross_time :
  time_to_cross 72 36 120 = 12 :=
by
  sorry

end faster_train_cross_time_l300_30031


namespace add_three_digits_l300_30053

theorem add_three_digits (x : ℕ) :
  (x = 152 ∨ x = 656) →
  (523000 + x) % 504 = 0 := 
by
  sorry

end add_three_digits_l300_30053


namespace intersection_of_A_and_CU_B_l300_30065

open Set Real

noncomputable def U : Set ℝ := univ
noncomputable def A : Set ℝ := {-1, 0, 1, 2, 3}
noncomputable def B : Set ℝ := { x : ℝ | x ≥ 2 }
noncomputable def CU_B : Set ℝ := { x : ℝ | x < 2 }

theorem intersection_of_A_and_CU_B :
  A ∩ CU_B = {-1, 0, 1} :=
by
  sorry

end intersection_of_A_and_CU_B_l300_30065


namespace stock_rise_in_morning_l300_30069

theorem stock_rise_in_morning (x : ℕ) (V : ℕ → ℕ) (h0 : V 0 = 100)
  (h100 : V 100 = 200) (h_recurrence : ∀ n, V n = 100 + n * x - n) :
  x = 2 :=
  by
  sorry

end stock_rise_in_morning_l300_30069


namespace part_1_odd_function_part_2_decreasing_l300_30052

noncomputable def f (x : ℝ) : ℝ := (1 - 2^x) / (1 + 2^x)

theorem part_1_odd_function : ∀ x : ℝ, f (-x) = -f x := by
  intro x
  sorry

theorem part_2_decreasing : ∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2 := by
  intros x1 x2 h
  sorry

end part_1_odd_function_part_2_decreasing_l300_30052


namespace evaluate_expression_l300_30061

def operation_star (A B : ℕ) : ℕ := (A + B) / 2
def operation_ominus (A B : ℕ) : ℕ := A - B

theorem evaluate_expression :
  operation_ominus (operation_star 6 10) (operation_star 2 4) = 5 := 
by 
  sorry

end evaluate_expression_l300_30061


namespace ab_zero_l300_30000

theorem ab_zero (a b : ℝ) (x : ℝ) (h : ∀ x : ℝ, a * x + b * x ^ 2 = -(a * (-x) + b * (-x) ^ 2)) : a * b = 0 :=
by
  sorry

end ab_zero_l300_30000


namespace triangle_perimeter_l300_30003

/-
  A square piece of paper with side length 2 has vertices A, B, C, and D. 
  The paper is folded such that vertex A meets edge BC at point A', 
  and A'C = 1/2. Prove that the perimeter of triangle A'BD is (3 + sqrt(17))/2 + 2sqrt(2).
-/
theorem triangle_perimeter
  (A B C D A' : ℝ × ℝ)
  (side_length : ℝ)
  (BC_length : ℝ)
  (CA'_length : ℝ)
  (BA'_length : ℝ)
  (BD_length : ℝ)
  (DA'_length : ℝ)
  (perimeter_correct : ℝ) :
  side_length = 2 ∧
  BC_length = 2 ∧
  CA'_length = 1/2 ∧
  BA'_length = 3/2 ∧
  BD_length = 2 * Real.sqrt 2 ∧
  DA'_length = Real.sqrt 17 / 2 →
  perimeter_correct = (3 + Real.sqrt 17) / 2 + 2 * Real.sqrt 2 →
  (side_length ≠ 0 ∧ BC_length = side_length ∧ 
   CA'_length ≠ 0 ∧ BA'_length ≠ 0 ∧ 
   BD_length ≠ 0 ∧ DA'_length ≠ 0) →
  (BA'_length + BD_length + DA'_length = perimeter_correct) :=
  sorry

end triangle_perimeter_l300_30003


namespace no_integer_solutions_l300_30041

theorem no_integer_solutions (x y z : ℤ) : x^3 + y^6 ≠ 7 * z + 3 :=
by sorry

end no_integer_solutions_l300_30041


namespace ratio_of_area_to_perimeter_l300_30032

noncomputable def altitude_of_equilateral_triangle (s : ℝ) : ℝ :=
  s * (Real.sqrt 3 / 2)

noncomputable def area_of_equilateral_triangle (s : ℝ) : ℝ :=
  1 / 2 * s * altitude_of_equilateral_triangle s

noncomputable def perimeter_of_equilateral_triangle (s : ℝ) : ℝ :=
  3 * s

theorem ratio_of_area_to_perimeter (s : ℝ) (h : s = 10) :
    (area_of_equilateral_triangle s) / (perimeter_of_equilateral_triangle s) = 5 * Real.sqrt 3 / 6 :=
  by
  rw [h]
  sorry

end ratio_of_area_to_perimeter_l300_30032


namespace simplify_expression_l300_30019

theorem simplify_expression (w : ℤ) : 
  (-2 * w + 3 - 4 * w + 7 + 6 * w - 5 - 8 * w + 8) = (-8 * w + 13) :=
by {
  sorry
}

end simplify_expression_l300_30019


namespace number_of_triangles_with_one_side_five_not_shortest_l300_30049

theorem number_of_triangles_with_one_side_five_not_shortest (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_one_side_five : a = 5 ∨ b = 5 ∨ c = 5)
  (h_not_shortest : a = 5 ∧ a > b ∧ a > c ∨ b = 5 ∧ b > a ∧ b > c ∨ c = 5 ∧ c > a ∧ c > b ∨ a ≠ 5 ∧ b = 5 ∧ b > c ∨ a ≠ 5 ∧ c = 5 ∧ c > b) :
  (∃ n, n = 10) :=
by
  sorry

end number_of_triangles_with_one_side_five_not_shortest_l300_30049


namespace prove_condition_for_equality_l300_30081

noncomputable def condition_for_equality (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : Prop :=
  c = (b * (a ^ 3 - 1)) / a

theorem prove_condition_for_equality (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (∃ (c' : ℕ), (c' = (b * (a ^ 3 - 1)) / a) ∧ 
      c' > 0 ∧ 
      (a + b / c' = a ^ 3 * (b / c')) ) → 
  c = (b * (a ^ 3 - 1)) / a := 
sorry

end prove_condition_for_equality_l300_30081


namespace coprime_sum_product_l300_30013

theorem coprime_sum_product (a b : ℤ) (h : Int.gcd a b = 1) : Int.gcd (a + b) (a * b) = 1 := by
  sorry

end coprime_sum_product_l300_30013


namespace area_on_larger_sphere_l300_30036

-- Define the radii of the spheres
def r_small : ℝ := 1
def r_in : ℝ := 4
def r_out : ℝ := 6

-- Given the area on the smaller sphere
def A_small_sphere_area : ℝ := 37

-- Statement: Find the area on the larger sphere
theorem area_on_larger_sphere :
  (A_small_sphere_area * (r_out / r_in) ^ 2 = 83.25) := by
  sorry

end area_on_larger_sphere_l300_30036
