import Mathlib

namespace NUMINAMATH_GPT_find_digit_sum_l387_38766

theorem find_digit_sum (A B X D C Y : ℕ) :
  (A * 100 + B * 10 + X) + (C * 100 + D * 10 + Y) = Y * 1010 + X * 1010 →
  A + D = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_digit_sum_l387_38766


namespace NUMINAMATH_GPT_polynomial_evaluation_x_eq_4_l387_38716

theorem polynomial_evaluation_x_eq_4 : 
  (4 ^ 4 + 4 ^ 3 + 4 ^ 2 + 4 + 1 = 341) := 
by 
  sorry

end NUMINAMATH_GPT_polynomial_evaluation_x_eq_4_l387_38716


namespace NUMINAMATH_GPT_parabola_tangent_midpoint_l387_38770

theorem parabola_tangent_midpoint (p : ℝ) (h : p > 0) :
    (∃ M : ℝ × ℝ, M = (2, -2*p)) ∧ 
    (∃ A B : ℝ × ℝ, A ≠ B ∧ 
                      (∃ yA yB : ℝ, yA = (A.1^2)/(2*p) ∧ yB = (B.1^2)/(2*p)) ∧ 
                      (0.5 * (A.2 + B.2) = 6)) → p = 1 := by sorry

end NUMINAMATH_GPT_parabola_tangent_midpoint_l387_38770


namespace NUMINAMATH_GPT_other_person_time_to_complete_job_l387_38722

-- Define the conditions
def SureshTime : ℕ := 15
def SureshWorkHours : ℕ := 9
def OtherPersonWorkHours : ℕ := 4

-- The proof problem: Prove that the other person can complete the job in 10 hours.
theorem other_person_time_to_complete_job (x : ℕ) 
  (h1 : ∀ SureshWorkHours SureshTime, SureshWorkHours * (1 / SureshTime) = (SureshWorkHours / SureshTime) ∧ 
       4 * (SureshWorkHours / SureshTime / 4) = 1) : 
  (x = 10) :=
sorry

end NUMINAMATH_GPT_other_person_time_to_complete_job_l387_38722


namespace NUMINAMATH_GPT_triple_solution_unique_l387_38705

theorem triple_solution_unique (a b c n : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hn : 0 < n) :
  (a^2 + b^2 = n * Nat.lcm a b + n^2) ∧
  (b^2 + c^2 = n * Nat.lcm b c + n^2) ∧
  (c^2 + a^2 = n * Nat.lcm c a + n^2) →
  (a = n ∧ b = n ∧ c = n) :=
by
  sorry

end NUMINAMATH_GPT_triple_solution_unique_l387_38705


namespace NUMINAMATH_GPT_find_x_l387_38787

variable {a b x r : ℝ}
variable (h₀ : 0 < a)
variable (h₁ : 0 < b)
variable (h₂ : r = (4 * a)^(2 * b))
variable (h₃ : r = (a^b * x^b)^2)
variable (h₄ : 0 < x)

theorem find_x : x = 4 := by
  sorry

end NUMINAMATH_GPT_find_x_l387_38787


namespace NUMINAMATH_GPT_initial_cell_count_l387_38745

-- Defining the constants and parameters given in the problem
def doubling_time : ℕ := 20 -- minutes
def culture_time : ℕ := 240 -- minutes (4 hours converted to minutes)
def final_bacterial_cells : ℕ := 4096

-- Definition to find the number of doublings
def num_doublings (culture_time doubling_time : ℕ) : ℕ :=
  culture_time / doubling_time

-- Definition for exponential growth formula
def exponential_growth (initial_cells : ℕ) (doublings : ℕ) : ℕ :=
  initial_cells * (2 ^ doublings)

-- The main theorem to be proven
theorem initial_cell_count :
  exponential_growth 1 (num_doublings culture_time doubling_time) = final_bacterial_cells :=
  sorry

end NUMINAMATH_GPT_initial_cell_count_l387_38745


namespace NUMINAMATH_GPT_trains_meet_in_approx_17_45_seconds_l387_38737

noncomputable def train_meet_time
  (length1 length2 distance_between : ℕ)
  (speed1_kmph speed2_kmph : ℕ)
  : ℕ :=
  let speed1_mps := (speed1_kmph * 1000) / 3600
  let speed2_mps := (speed2_kmph * 1000) / 3600
  let relative_speed := speed1_mps + speed2_mps
  let total_distance := distance_between + length1 + length2
  total_distance / relative_speed

theorem trains_meet_in_approx_17_45_seconds :
  train_meet_time 100 200 660 90 108 = 17 := by
  sorry

end NUMINAMATH_GPT_trains_meet_in_approx_17_45_seconds_l387_38737


namespace NUMINAMATH_GPT_rest_stop_location_l387_38765

theorem rest_stop_location (km_A km_B : ℕ) (fraction : ℚ) (difference := km_B - km_A) 
  (rest_stop_distance := fraction * difference) : 
  km_A = 30 → km_B = 210 → fraction = 4 / 5 → rest_stop_distance + km_A = 174 :=
by 
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_rest_stop_location_l387_38765


namespace NUMINAMATH_GPT_maximal_inradius_of_tetrahedron_l387_38741

-- Define the properties and variables
variables (A B C D : ℝ) (h_A h_B h_C h_D : ℝ) (V r : ℝ)

-- Assumptions
variable (h_A_ge_1 : h_A ≥ 1)
variable (h_B_ge_1 : h_B ≥ 1)
variable (h_C_ge_1 : h_C ≥ 1)
variable (h_D_ge_1 : h_D ≥ 1)

-- Volume expressed in terms of altitudes and face areas
axiom vol_eq_Ah : V = (1 / 3) * A * h_A
axiom vol_eq_Bh : V = (1 / 3) * B * h_B
axiom vol_eq_Ch : V = (1 / 3) * C * h_C
axiom vol_eq_Dh : V = (1 / 3) * D * h_D

-- Volume expressed in terms of inradius and sum of face areas
axiom vol_eq_inradius : V = (1 / 3) * (A + B + C + D) * r

-- The theorem to prove
theorem maximal_inradius_of_tetrahedron : r = 1 / 4 :=
sorry

end NUMINAMATH_GPT_maximal_inradius_of_tetrahedron_l387_38741


namespace NUMINAMATH_GPT_a_plus_c_eq_neg_300_l387_38794

namespace Polynomials

variable {α : Type*} [LinearOrderedField α]

def f (a b x : α) := x^2 + a * x + b
def g (c d x : α) := x^2 + c * x + d

theorem a_plus_c_eq_neg_300 
  {a b c d : α}
  (h1 : ∀ x, f a b x ≥ -144) 
  (h2 : ∀ x, g c d x ≥ -144)
  (h3 : f a b 150 = -200) 
  (h4 : g c d 150 = -200)
  (h5 : ∃ x, (2*x + a = 0) ∧ g c d x = 0)
  (h6 : ∃ x, (2*x + c = 0) ∧ f a b x = 0) :
  a + c = -300 := 
sorry

end Polynomials

end NUMINAMATH_GPT_a_plus_c_eq_neg_300_l387_38794


namespace NUMINAMATH_GPT_set_intersection_nonempty_implies_m_le_neg1_l387_38788

theorem set_intersection_nonempty_implies_m_le_neg1
  (m : ℝ)
  (A : Set ℝ := {x | x^2 - 4 * m * x + 2 * m + 6 = 0})
  (B : Set ℝ := {x | x < 0}) :
  (A ∩ B).Nonempty → m ≤ -1 := 
sorry

end NUMINAMATH_GPT_set_intersection_nonempty_implies_m_le_neg1_l387_38788


namespace NUMINAMATH_GPT_part1_part2_part3_l387_38774

/-- Proof for part (1): If the point P lies on the x-axis, then m = -1. -/
theorem part1 (m : ℝ) (hx : 3 * m + 3 = 0) : m = -1 := 
by {
  sorry
}

/-- Proof for part (2): If point P lies on a line passing through A(-5, 1) and parallel to the y-axis, 
then the coordinates of point P are (-5, -12). -/
theorem part2 (m : ℝ) (hy : 2 * m + 5 = -5) : (2 * m + 5, 3 * m + 3) = (-5, -12) := 
by {
  sorry
}

/-- Proof for part (3): If point P is moved 2 right and 3 up to point M, 
and point M lies in the third quadrant with a distance of 7 from the y-axis, then the coordinates of M are (-7, -15). -/
theorem part3 (m : ℝ) 
  (hc : 2 * m + 7 = -7)
  (config : 3 * m + 6 < 0) : (2 * m + 7, 3 * m + 6) = (-7, -15) := 
by {
  sorry
}

end NUMINAMATH_GPT_part1_part2_part3_l387_38774


namespace NUMINAMATH_GPT_star_evaluation_l387_38719

def star (a b : ℕ) : ℕ := 3 + b^(a + 1)

theorem star_evaluation : star (star 2 3) 2 = 3 + 2^31 :=
by {
  sorry
}

end NUMINAMATH_GPT_star_evaluation_l387_38719


namespace NUMINAMATH_GPT_incorrect_option_C_l387_38740

theorem incorrect_option_C (a b : ℝ) (h1 : a > b) (h2 : b > a + b) : ¬ (ab > (a + b)^2) :=
by {
  sorry
}

end NUMINAMATH_GPT_incorrect_option_C_l387_38740


namespace NUMINAMATH_GPT_intersection_complement_eq_l387_38742

open Set

variable (U A B : Set ℕ)

theorem intersection_complement_eq :
  (U = {1, 2, 3, 4, 5, 6}) →
  (A = {1, 3}) →
  (B = {3, 4, 5}) →
  A ∩ (U \ B) = {1} :=
by
  intros hU hA hB
  subst hU
  subst hA
  subst hB
  sorry

end NUMINAMATH_GPT_intersection_complement_eq_l387_38742


namespace NUMINAMATH_GPT_s9_s3_ratio_l387_38778

variable {a_n : ℕ → ℝ}
variable {s_n : ℕ → ℝ}
variable {a : ℝ}

-- Conditions
axiom h_s6_s3_ratio : s_n 6 / s_n 3 = 1 / 2

-- Theorem to prove
theorem s9_s3_ratio (h : s_n 3 = a) : s_n 9 / s_n 3 = 3 / 4 := 
sorry

end NUMINAMATH_GPT_s9_s3_ratio_l387_38778


namespace NUMINAMATH_GPT_stock_yield_percentage_l387_38714

theorem stock_yield_percentage
  (annual_dividend : ℝ)
  (market_price : ℝ)
  (face_value : ℝ)
  (yield_percentage : ℝ)
  (H1 : annual_dividend = 0.14 * face_value)
  (H2 : market_price = 175)
  (H3 : face_value = 100)
  (H4 : yield_percentage = (annual_dividend / market_price) * 100) :
  yield_percentage = 8 := sorry

end NUMINAMATH_GPT_stock_yield_percentage_l387_38714


namespace NUMINAMATH_GPT_mean_of_set_l387_38758

theorem mean_of_set (n : ℤ) (h_median : n + 7 = 14) : (n + (n + 4) + (n + 7) + (n + 10) + (n + 14)) / 5 = 14 := by
  sorry

end NUMINAMATH_GPT_mean_of_set_l387_38758


namespace NUMINAMATH_GPT_bookcase_length_in_inches_l387_38755

theorem bookcase_length_in_inches (feet_length : ℕ) (inches_per_foot : ℕ) (h1 : feet_length = 4) (h2 : inches_per_foot = 12) : (feet_length * inches_per_foot) = 48 :=
by
  sorry

end NUMINAMATH_GPT_bookcase_length_in_inches_l387_38755


namespace NUMINAMATH_GPT_inequality_solution_l387_38724

theorem inequality_solution {x : ℝ} (h : |x + 3| - |x - 1| > 0) : x > -1 :=
sorry

end NUMINAMATH_GPT_inequality_solution_l387_38724


namespace NUMINAMATH_GPT_initial_birds_179_l387_38727

theorem initial_birds_179 (B : ℕ) (h1 : B + 38 = 217) : B = 179 :=
sorry

end NUMINAMATH_GPT_initial_birds_179_l387_38727


namespace NUMINAMATH_GPT_savings_value_l387_38760

def total_cost_individual (g : ℕ) (s : ℕ) : ℝ :=
  let cost_per_window := 120
  let cost (n : ℕ) : ℝ := 
    let paid_windows := n - (n / 6) -- one free window per five
    cost_per_window * paid_windows
  let discount (amount : ℝ) : ℝ :=
    if s > 10 then 0.95 * amount else amount
  discount (cost g) + discount (cost s)

def total_cost_joint (g : ℕ) (s : ℕ) : ℝ :=
  let cost_per_window := 120
  let n := g + s
  let paid_windows := n - (n / 6) -- one free window per five
  let joint_cost := cost_per_window * paid_windows
  if n > 10 then 0.95 * joint_cost else joint_cost

def savings (g : ℕ) (s : ℕ) : ℝ :=
  total_cost_individual g s - total_cost_joint g s

theorem savings_value (g s : ℕ) (hg : g = 9) (hs : s = 13) : savings g s = 162 := 
by 
  simp [savings, total_cost_individual, total_cost_joint, hg, hs]
  -- Detailed calculation is omitted, since it's not required according to the instructions.
  sorry

end NUMINAMATH_GPT_savings_value_l387_38760


namespace NUMINAMATH_GPT_original_people_l387_38744

-- Declare the original number of people in the room
variable (x : ℕ)

-- Conditions
-- One third of the people in the room left
def remaining_after_one_third_left (x : ℕ) : ℕ := (2 * x) / 3

-- One quarter of the remaining people started to dance
def dancers (remaining : ℕ) : ℕ := remaining / 4

-- Number of people not dancing
def non_dancers (remaining : ℕ) (dancers : ℕ) : ℕ := remaining - dancers

-- Given that there are 18 people not dancing
variable (remaining : ℕ) (dancers : ℕ)
axiom non_dancers_number : non_dancers remaining dancers = 18

-- Theorem to prove
theorem original_people (h_rem: remaining = remaining_after_one_third_left x) 
(h_dancers: dancers = remaining / 4) : x = 36 := by
  sorry

end NUMINAMATH_GPT_original_people_l387_38744


namespace NUMINAMATH_GPT_victor_weight_is_correct_l387_38747

-- Define the given conditions
def bear_daily_food : ℕ := 90
def victors_food_in_3_weeks : ℕ := 15
def days_in_3_weeks : ℕ := 21

-- Define the equivalent weight of Victor based on the given conditions
def victor_weight : ℕ := bear_daily_food * days_in_3_weeks / victors_food_in_3_weeks

-- Prove that the weight of Victor is 126 pounds
theorem victor_weight_is_correct : victor_weight = 126 := by
  sorry

end NUMINAMATH_GPT_victor_weight_is_correct_l387_38747


namespace NUMINAMATH_GPT_calc_3_pow_6_mul_4_pow_6_l387_38700

theorem calc_3_pow_6_mul_4_pow_6 : (3^6) * (4^6) = 2985984 :=
by 
  sorry

end NUMINAMATH_GPT_calc_3_pow_6_mul_4_pow_6_l387_38700


namespace NUMINAMATH_GPT_dance_boys_count_l387_38736

theorem dance_boys_count (d b : ℕ) (h1 : b = 2 * d) (h2 : b = d - 1 + 8) : b = 14 :=
by
  -- The proof is omitted, denoted by 'sorry'
  sorry

end NUMINAMATH_GPT_dance_boys_count_l387_38736


namespace NUMINAMATH_GPT_train_speed_l387_38721

def train_length : ℝ := 800
def crossing_time : ℝ := 12
def expected_speed : ℝ := 66.67 

theorem train_speed (h_len : train_length = 800) (h_time : crossing_time = 12) : 
  train_length / crossing_time = expected_speed := 
by {
  sorry
}

end NUMINAMATH_GPT_train_speed_l387_38721


namespace NUMINAMATH_GPT_solution_per_beaker_l387_38743

theorem solution_per_beaker (solution_per_tube : ℕ) (num_tubes : ℕ) (num_beakers : ℕ)
    (h1 : solution_per_tube = 7) (h2 : num_tubes = 6) (h3 : num_beakers = 3) :
    (solution_per_tube * num_tubes) / num_beakers = 14 :=
by
  sorry

end NUMINAMATH_GPT_solution_per_beaker_l387_38743


namespace NUMINAMATH_GPT_discount_per_issue_l387_38731

theorem discount_per_issue
  (normal_subscription_cost : ℝ) (months : ℕ) (issues_per_month : ℕ) 
  (promotional_discount : ℝ) :
  normal_subscription_cost = 34 →
  months = 18 →
  issues_per_month = 2 →
  promotional_discount = 9 →
  (normal_subscription_cost - promotional_discount) / (months * issues_per_month) = 0.25 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_discount_per_issue_l387_38731


namespace NUMINAMATH_GPT_triangle_sides_proportional_l387_38763

theorem triangle_sides_proportional (a b c r d : ℝ)
  (h1 : 2 * r < a) 
  (h2 : a < b) 
  (h3 : b < c) 
  (h4 : a = 2 * r + d)
  (h5 : b = 2 * r + 2 * d)
  (h6 : c = 2 * r + 3 * d)
  (hr_pos : r > 0)
  (hd_pos : d > 0) :
  ∃ k : ℝ, k > 0 ∧ a = 3 * k ∧ b = 4 * k ∧ c = 5 * k :=
sorry

end NUMINAMATH_GPT_triangle_sides_proportional_l387_38763


namespace NUMINAMATH_GPT_train_length_is_correct_l387_38786

noncomputable def length_of_train (speed_train_kmph : ℕ) (speed_man_kmph : ℕ) (time_seconds : ℕ) : ℝ :=
  let relative_speed_kmph := (speed_train_kmph + speed_man_kmph)
  let relative_speed_mps := (relative_speed_kmph : ℝ) * (5 / 18)
  relative_speed_mps * (time_seconds : ℝ)

theorem train_length_is_correct :
  length_of_train 60 6 3 = 54.99 := 
by
  sorry

end NUMINAMATH_GPT_train_length_is_correct_l387_38786


namespace NUMINAMATH_GPT_evaluate_expression_l387_38796

theorem evaluate_expression : (827 * 827) - (826 * 828) + 2 = 3 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l387_38796


namespace NUMINAMATH_GPT_gabriel_month_days_l387_38768

theorem gabriel_month_days (forgot_days took_days : ℕ) (h_forgot : forgot_days = 3) (h_took : took_days = 28) : 
  forgot_days + took_days = 31 :=
by
  sorry

end NUMINAMATH_GPT_gabriel_month_days_l387_38768


namespace NUMINAMATH_GPT_domain_M_complement_domain_M_l387_38772

noncomputable def f (x : ℝ) : ℝ :=
  1 / Real.sqrt (1 - x)

noncomputable def g (x : ℝ) : ℝ :=
  Real.log (1 + x)

def M : Set ℝ :=
  {x | 1 - x > 0}

def N : Set ℝ :=
  {x | 1 + x > 0}

def complement_M : Set ℝ :=
  {x | 1 - x ≤ 0}

theorem domain_M :
  M = {x | x < 1} := by
  sorry

theorem complement_domain_M :
  complement_M = {x | x ≥ 1} := by
  sorry

end NUMINAMATH_GPT_domain_M_complement_domain_M_l387_38772


namespace NUMINAMATH_GPT_find_parabola_equation_l387_38797

noncomputable def parabola_equation (a b c : ℝ) : Prop :=
  ∀ x y : ℝ, 
    (y = a * x ^ 2 + b * x + c) ∧ 
    (y = (x - 3) ^ 2 - 2) ∧
    (a * (4 - 3) ^ 2 - 2 = 2)

theorem find_parabola_equation :
  ∃ (a b c : ℝ), parabola_equation a b c ∧ a = 4 ∧ b = -24 ∧ c = 34 :=
sorry

end NUMINAMATH_GPT_find_parabola_equation_l387_38797


namespace NUMINAMATH_GPT_number_of_questions_in_test_l387_38710

-- Definitions based on the conditions:
def marks_per_question : ℕ := 2
def jose_wrong_questions : ℕ := 5  -- number of questions Jose got wrong
def total_combined_score : ℕ := 210  -- total score of Meghan, Jose, and Alisson combined

-- Let A be Alisson's score
variables (A Jose Meghan : ℕ)

-- Conditions
axiom joe_more_than_alisson : Jose = A + 40
axiom megh_less_than_jose : Meghan = Jose - 20
axiom combined_scores : A + Jose + Meghan = total_combined_score

-- Function to compute the total possible score for Jose without wrong answers:
noncomputable def jose_improvement_score : ℕ := Jose + (jose_wrong_questions * marks_per_question)

-- Proof problem statement
theorem number_of_questions_in_test :
  (jose_improvement_score Jose) / marks_per_question = 50 :=
by
  -- Sorry is used here to indicate that the proof is omitted.
  sorry

end NUMINAMATH_GPT_number_of_questions_in_test_l387_38710


namespace NUMINAMATH_GPT_infinite_sum_problem_l387_38769

theorem infinite_sum_problem :
  (∑' n : ℕ, if n = 0 then 0 else (3^n) / (1 + 3^n + 3^(n + 1) + 3^(2 * n + 1))) = (1 / 4) := 
by
  sorry

end NUMINAMATH_GPT_infinite_sum_problem_l387_38769


namespace NUMINAMATH_GPT_find_third_angle_of_triangle_l387_38728

theorem find_third_angle_of_triangle (a b c : ℝ) (h₁ : a = 40) (h₂ : b = 3 * c) (h₃ : a + b + c = 180) : c = 35 := 
by sorry

end NUMINAMATH_GPT_find_third_angle_of_triangle_l387_38728


namespace NUMINAMATH_GPT_parallelogram_side_length_l387_38723

theorem parallelogram_side_length (a b : ℕ) (h1 : 2 * (a + b) = 16) (h2 : a = 5) : b = 3 :=
by
  sorry

end NUMINAMATH_GPT_parallelogram_side_length_l387_38723


namespace NUMINAMATH_GPT_rachel_left_24_brownies_at_home_l387_38704

-- Defining the conditions
def total_brownies : ℕ := 40
def brownies_brought_to_school : ℕ := 16

-- Formulation of the theorem
theorem rachel_left_24_brownies_at_home : (total_brownies - brownies_brought_to_school = 24) :=
by
  sorry

end NUMINAMATH_GPT_rachel_left_24_brownies_at_home_l387_38704


namespace NUMINAMATH_GPT_b5_plus_b9_l387_38781

variable {a : ℕ → ℕ} -- Geometric sequence
variable {b : ℕ → ℕ} -- Arithmetic sequence

axiom geom_progression {r x y : ℕ} : a x = a 1 * r^(x - 1) ∧ a y = a 1 * r^(y - 1)
axiom arith_progression {d x y : ℕ} : b x = b 1 + d * (x - 1) ∧ b y = b 1 + d * (y - 1)

axiom a3a11_equals_4a7 : a 3 * a 11 = 4 * a 7
axiom a7_equals_b7 : a 7 = b 7

theorem b5_plus_b9 : b 5 + b 9 = 8 := by
  apply sorry

end NUMINAMATH_GPT_b5_plus_b9_l387_38781


namespace NUMINAMATH_GPT_quadratic_equation_in_one_variable_l387_38779

def is_quadratic_in_one_variable (eq : String) : Prop :=
  match eq with
  | "2x^2 + 5y + 1 = 0" => False
  | "ax^2 + bx - c = 0" => ∃ (a b c : ℝ), a ≠ 0
  | "1/x^2 + x = 2" => False
  | "x^2 = 0" => True
  | _ => False

theorem quadratic_equation_in_one_variable :
  is_quadratic_in_one_variable "x^2 = 0" := by
  sorry

end NUMINAMATH_GPT_quadratic_equation_in_one_variable_l387_38779


namespace NUMINAMATH_GPT_curves_intersect_condition_l387_38708

noncomputable def curves_intersect_exactly_three_points (a : ℝ) : Prop :=
  ∃ x y : ℝ, 
    (x^2 + y^2 = a^2) ∧ (y = x^2 + a) ∧ 
    (y = a → x = 0) ∧ 
    ((2 * a + 1 < 0) → y = -(2 * a + 1) - 1)

theorem curves_intersect_condition (a : ℝ) : 
  curves_intersect_exactly_three_points a ↔ a < -1/2 :=
sorry

end NUMINAMATH_GPT_curves_intersect_condition_l387_38708


namespace NUMINAMATH_GPT_constant_is_arithmetic_l387_38701

def is_constant_sequence (a : ℕ → ℝ) : Prop :=
  ∃ c : ℝ, ∀ n : ℕ, a n = c

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem constant_is_arithmetic (a : ℕ → ℝ) (h : is_constant_sequence a) : is_arithmetic_sequence a := by
  sorry

end NUMINAMATH_GPT_constant_is_arithmetic_l387_38701


namespace NUMINAMATH_GPT_circumradius_of_triangle_l387_38752

theorem circumradius_of_triangle (a b S : ℝ) (A : a = 2) (B : b = 3) (Area : S = 3 * Real.sqrt 15 / 4)
  (median_cond : ∃ c m, m = (a^2 + b^2 - c^2) / (2*a*b) ∧ m < c / 2) :
  ∃ R, R = 8 / Real.sqrt 15 :=
by
  sorry

end NUMINAMATH_GPT_circumradius_of_triangle_l387_38752


namespace NUMINAMATH_GPT_smallest_positive_integer_l387_38757

theorem smallest_positive_integer (n : ℕ) : 3 * n ≡ 568 [MOD 34] → n = 18 := 
sorry

end NUMINAMATH_GPT_smallest_positive_integer_l387_38757


namespace NUMINAMATH_GPT_area_curve_is_correct_l387_38767

-- Define the initial conditions
structure Rectangle :=
  (vertices : Fin 4 → ℝ × ℝ)
  (point : ℝ × ℝ)

-- Define the rotation transformation
def rotate_clockwise_90 (center : ℝ × ℝ) (point : ℝ × ℝ) : ℝ × ℝ :=
  let (cx, cy) := center
  let (px, py) := point
  (cx + (py - cy), cy - (px - cx))

-- Given initial rectangle and the point to track
def initial_rectangle : Rectangle :=
  { vertices := ![(0, 0), (2, 0), (0, 3), (2, 3)],
    point := (1, 1) }

-- Perform the four specified rotations
def rotated_points : List (ℝ × ℝ) :=
  let r1 := rotate_clockwise_90 (2, 0) initial_rectangle.point
  let r2 := rotate_clockwise_90 (5, 0) r1
  let r3 := rotate_clockwise_90 (7, 0) r2
  let r4 := rotate_clockwise_90 (10, 0) r3
  [initial_rectangle.point, r1, r2, r3, r4]

-- Calculate the area below the curve and above the x-axis
noncomputable def area_below_curve : ℝ :=
  6 + (7 * Real.pi / 2)

-- The theorem statement
theorem area_curve_is_correct : 
  area_below_curve = 6 + (7 * Real.pi / 2) :=
  by trivial

end NUMINAMATH_GPT_area_curve_is_correct_l387_38767


namespace NUMINAMATH_GPT_quadratic_is_complete_the_square_l387_38711

theorem quadratic_is_complete_the_square :
  ∃ a b c : ℝ, 15 * (x : ℝ)^2 + 150 * x + 2250 = a * (x + b)^2 + c 
  ∧ a + b + c = 1895 :=
sorry

end NUMINAMATH_GPT_quadratic_is_complete_the_square_l387_38711


namespace NUMINAMATH_GPT_find_x_l387_38732

noncomputable def is_solution (x : ℝ) : Prop :=
   (⌊x * ⌊x⌋⌋ = 29)

theorem find_x (x : ℝ) (h : is_solution x) : 5.8 ≤ x ∧ x < 6 :=
sorry

end NUMINAMATH_GPT_find_x_l387_38732


namespace NUMINAMATH_GPT_valid_base6_number_2015_l387_38777

def is_valid_base6_digit (d : Nat) : Prop :=
  d = 0 ∨ d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 5

def is_base6_number (n : Nat) : Prop :=
  ∀ (digit : Nat), digit ∈ (n.digits 10) → is_valid_base6_digit digit

theorem valid_base6_number_2015 : is_base6_number 2015 := by
  sorry

end NUMINAMATH_GPT_valid_base6_number_2015_l387_38777


namespace NUMINAMATH_GPT_MarlySoupBags_l387_38782

theorem MarlySoupBags :
  ∀ (milk chicken_stock vegetables bag_capacity total_soup total_bags : ℚ),
    milk = 6 ∧
    chicken_stock = 3 * milk ∧
    vegetables = 3 ∧
    bag_capacity = 2 ∧
    total_soup = milk + chicken_stock + vegetables ∧
    total_bags = total_soup / bag_capacity ∧
    total_bags.ceil = 14 :=
by
  intros
  sorry

end NUMINAMATH_GPT_MarlySoupBags_l387_38782


namespace NUMINAMATH_GPT_tom_climbing_time_l387_38702

theorem tom_climbing_time (elizabeth_time : ℕ) (multiplier : ℕ) 
  (h1 : elizabeth_time = 30) (h2 : multiplier = 4) : (elizabeth_time * multiplier) / 60 = 2 :=
by
  sorry

end NUMINAMATH_GPT_tom_climbing_time_l387_38702


namespace NUMINAMATH_GPT_find_C_l387_38764

variable (A B C : ℕ)

theorem find_C (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : B + C = 310) : 
  C = 10 := 
by
  sorry

end NUMINAMATH_GPT_find_C_l387_38764


namespace NUMINAMATH_GPT_housewife_more_kgs_l387_38789

theorem housewife_more_kgs (P R money more_kgs : ℝ)
  (hR: R = 40)
  (hReduction: R = P - 0.25 * P)
  (hMoney: money = 800)
  (hMoreKgs: more_kgs = (money / R) - (money / P)) :
  more_kgs = 5 :=
  by
    sorry

end NUMINAMATH_GPT_housewife_more_kgs_l387_38789


namespace NUMINAMATH_GPT_find_t_l387_38759

theorem find_t :
  ∃ (B : ℝ × ℝ) (t : ℝ), 
  B.1^2 + B.2^2 = 100 ∧ 
  B.1 - 2 * B.2 + 10 = 0 ∧ 
  B.1 > 0 ∧ B.2 > 0 ∧ 
  t = 20 ∧ 
  (∃ m : ℝ, 
    m = -2 ∧ 
    B.2 = m * B.1 + (8 + 2 * B.1 - m * B.1)) := 
by
  sorry

end NUMINAMATH_GPT_find_t_l387_38759


namespace NUMINAMATH_GPT_ex1_ex2_l387_38717

-- Definition of the "multiplication-subtraction" operation.
def mult_sub (a b : ℚ) : ℚ :=
  if a = 0 then abs b else if b = 0 then abs a else if abs a = abs b then 0 else
  if (a > 0 ∧ b > 0) ∨ (a < 0 ∧ b < 0) then abs a - abs b else -(abs a - abs b)

theorem ex1 : mult_sub (mult_sub (3) (-2)) (mult_sub (-9) 0) = -8 :=
  sorry

theorem ex2 : ∃ (a b c : ℚ), (mult_sub (mult_sub a b) c) ≠ (mult_sub a (mult_sub b c)) :=
  ⟨3, -2, 4, by simp [mult_sub]; sorry⟩

end NUMINAMATH_GPT_ex1_ex2_l387_38717


namespace NUMINAMATH_GPT_fraction_to_decimal_l387_38718

theorem fraction_to_decimal : (17 : ℝ) / 50 = 0.34 := 
by 
  sorry

end NUMINAMATH_GPT_fraction_to_decimal_l387_38718


namespace NUMINAMATH_GPT_earnings_in_total_l387_38730

-- Defining the conditions
def hourly_wage : ℝ := 12.50
def hours_per_week : ℝ := 40
def earnings_per_widget : ℝ := 0.16
def widgets_per_week : ℝ := 1250

-- Theorem statement
theorem earnings_in_total : 
  (hours_per_week * hourly_wage) + (widgets_per_week * earnings_per_widget) = 700 := 
by
  sorry

end NUMINAMATH_GPT_earnings_in_total_l387_38730


namespace NUMINAMATH_GPT_remainder_proof_l387_38785

-- Definitions and conditions
variables {x y u v : ℕ}
variables (hx : x = u * y + v)

-- Problem statement in Lean 4
theorem remainder_proof (hx : x = u * y + v) : ((x + 3 * u * y + y) % y) = v :=
sorry

end NUMINAMATH_GPT_remainder_proof_l387_38785


namespace NUMINAMATH_GPT_find_min_value_l387_38735

theorem find_min_value (a x y : ℝ) (h : y = -x^2 + 3 * Real.log x) : ∃ x, ∃ y, (a - x)^2 + (a + 2 - y)^2 = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_min_value_l387_38735


namespace NUMINAMATH_GPT_three_digit_number_l387_38792

theorem three_digit_number (a b c : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 9) (h3 : 1 ≤ b) (h4 : b ≤ 9) (h5 : 0 ≤ c) (h6 : c ≤ 9) 
  (h : 100 * a + 10 * b + c = 3 * (10 * (a + b) + c)) : 100 * a + 10 * b + c = 135 :=
  sorry

end NUMINAMATH_GPT_three_digit_number_l387_38792


namespace NUMINAMATH_GPT_liam_markers_liam_first_markers_over_500_l387_38790

def seq (n : ℕ) : ℕ := 5 * 3^n

theorem liam_markers (n : ℕ) (h1 : seq 0 = 5) (h2 : seq 1 = 10) (h3 : ∀ k < n, 5 * 3^k ≤ 500) : 
  seq n > 500 := by sorry

theorem liam_first_markers_over_500 (h1 : seq 0 = 5) (h2 : seq 1 = 10) :
  ∃ n, seq n > 500 ∧ ∀ k < n, seq k ≤ 500 := by sorry

end NUMINAMATH_GPT_liam_markers_liam_first_markers_over_500_l387_38790


namespace NUMINAMATH_GPT_find_function_g_l387_38746

noncomputable def g (x : ℝ) : ℝ := (5^x - 3^x) / 8

theorem find_function_g (x y : ℝ) (h1 : g 2 = 2) (h2 : ∀ x y : ℝ, g (x + y) = 5^y * g x + 3^x * g y) :
  g x = (5^x - 3^x) / 8 :=
by
  sorry

end NUMINAMATH_GPT_find_function_g_l387_38746


namespace NUMINAMATH_GPT_butterflies_in_the_garden_l387_38748

variable (total_butterflies : Nat) (fly_away : Nat)

def butterflies_left (total_butterflies : Nat) (fly_away : Nat) : Nat :=
  total_butterflies - fly_away

theorem butterflies_in_the_garden :
  (total_butterflies = 9) → (fly_away = 1 / 3 * total_butterflies) → butterflies_left total_butterflies fly_away = 6 :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_butterflies_in_the_garden_l387_38748


namespace NUMINAMATH_GPT_evaluate_expression_l387_38791

theorem evaluate_expression : 3 - 5 * (6 - 2^3) / 2 = 8 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l387_38791


namespace NUMINAMATH_GPT_gold_bars_total_worth_l387_38799

theorem gold_bars_total_worth :
  let rows := 4
  let bars_per_row := 20
  let worth_per_bar : ℕ := 20000
  let total_bars := rows * bars_per_row
  let total_worth := total_bars * worth_per_bar
  total_worth = 1600000 :=
by
  sorry

end NUMINAMATH_GPT_gold_bars_total_worth_l387_38799


namespace NUMINAMATH_GPT_evaluate_expression_x_eq_3_l387_38771

theorem evaluate_expression_x_eq_3 : (3^5 - 5 * 3 + 7 * 3^3) = 417 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_x_eq_3_l387_38771


namespace NUMINAMATH_GPT_min_value_quadratic_expr_l387_38749

-- Define the quadratic function
def quadratic_expr (x : ℝ) : ℝ := 8 * x^2 - 24 * x + 1729

-- State the theorem to prove the minimum value
theorem min_value_quadratic_expr : (∃ x : ℝ, ∀ y : ℝ, quadratic_expr y ≥ quadratic_expr x) ∧ ∃ x : ℝ, quadratic_expr x = 1711 :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_min_value_quadratic_expr_l387_38749


namespace NUMINAMATH_GPT_minimum_value_fraction_l387_38720

theorem minimum_value_fraction (m n : ℝ) (h_m_pos : 0 < m) (h_n_pos : 0 < n) 
  (h_parallel : m / (4 - n) = 1 / 2) : 
  (1 / m + 8 / n) ≥ 9 / 2 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_fraction_l387_38720


namespace NUMINAMATH_GPT_velocity_ratio_proof_l387_38783

noncomputable def velocity_ratio (V U : ℝ) : ℝ := V / U

-- The conditions:
-- 1. A smooth horizontal surface.
-- 2. The speed of the ball is perpendicular to the face of the block.
-- 3. The mass of the ball is much smaller than the mass of the block.
-- 4. The collision is elastic.
-- 5. After the collision, the ball’s speed is halved and it moves in the opposite direction.

def ball_block_collision 
    (V U U_final : ℝ) 
    (smooth_surface : Prop) 
    (perpendicular_impact : Prop) 
    (ball_much_smaller : Prop) 
    (elastic_collision : Prop) 
    (speed_halved : Prop) : Prop :=
  U_final = U ∧ V / U = 4

theorem velocity_ratio_proof : 
  ∀ (V U U_final : ℝ)
    (smooth_surface : Prop)
    (perpendicular_impact : Prop)
    (ball_much_smaller : Prop)
    (elastic_collision : Prop)
    (speed_halved : Prop),
    ball_block_collision V U U_final smooth_surface perpendicular_impact ball_much_smaller elastic_collision speed_halved := 
sorry

end NUMINAMATH_GPT_velocity_ratio_proof_l387_38783


namespace NUMINAMATH_GPT_initial_men_work_count_l387_38734

-- Define conditions given in the problem
def work_rate (M : ℕ) := 1 / (40 * M)
def initial_men_can_complete_work_in_40_days (M : ℕ) : Prop := M * work_rate M * 40 = 1
def work_done_by_initial_men_in_16_days (M : ℕ) := (M * 16) * work_rate M
def remaining_work_done_by_remaining_men_in_40_days (M : ℕ) := ((M - 14) * 40) * work_rate M

-- Define the main theorem to prove
theorem initial_men_work_count (M : ℕ) :
  initial_men_can_complete_work_in_40_days M →
  work_done_by_initial_men_in_16_days M = 2 / 5 →
  3 / 5 = (remaining_work_done_by_remaining_men_in_40_days M) →
  M = 15 :=
by
  intros h_initial h_16_days h_remaining
  have rate := h_initial
  sorry

end NUMINAMATH_GPT_initial_men_work_count_l387_38734


namespace NUMINAMATH_GPT_total_handshakes_l387_38761

theorem total_handshakes (team1 team2 refs : ℕ) (players_per_team : ℕ) :
  team1 = 11 → team2 = 11 → refs = 3 → players_per_team = 11 →
  (players_per_team * players_per_team + (players_per_team * 2 * refs) = 187) :=
by
  intros h_team1 h_team2 h_refs h_players_per_team
  -- Now we want to prove that
  -- 11 * 11 + (11 * 2 * 3) = 187
  -- However, we can just add sorry here as the purpose is to write the statement
  sorry

end NUMINAMATH_GPT_total_handshakes_l387_38761


namespace NUMINAMATH_GPT_medicine_liquid_poured_l387_38725

theorem medicine_liquid_poured (x : ℝ) (h : 63 * (1 - x / 63) * (1 - x / 63) = 28) : x = 18 :=
by
  sorry

end NUMINAMATH_GPT_medicine_liquid_poured_l387_38725


namespace NUMINAMATH_GPT_shift_down_equation_l387_38793

def f (x : ℝ) : ℝ := 2 * x + 3
def g (x : ℝ) : ℝ := f x - 3

theorem shift_down_equation : ∀ x : ℝ, g x = 2 * x := by
  sorry

end NUMINAMATH_GPT_shift_down_equation_l387_38793


namespace NUMINAMATH_GPT_georgia_total_carnation_cost_l387_38773

-- Define the cost of one carnation
def cost_of_single_carnation : ℝ := 0.50

-- Define the cost of one dozen carnations
def cost_of_dozen_carnations : ℝ := 4.00

-- Define the number of teachers
def number_of_teachers : ℕ := 5

-- Define the number of friends
def number_of_friends : ℕ := 14

-- Calculate the cost for teachers
def cost_for_teachers : ℝ :=
  (number_of_teachers : ℝ) * cost_of_dozen_carnations

-- Calculate the cost for friends
def cost_for_friends : ℝ :=
  cost_of_dozen_carnations + (2 * cost_of_single_carnation)

-- Calculate the total cost
def total_cost : ℝ := cost_for_teachers + cost_for_friends

-- Theorem stating the total cost
theorem georgia_total_carnation_cost : total_cost = 25 := by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_georgia_total_carnation_cost_l387_38773


namespace NUMINAMATH_GPT_intersection_M_N_l387_38762

-- Define the set M
def M : Set ℤ := {-2, -1, 0, 1, 2}

-- Define the condition for set N
def N : Set ℤ := {x | x + 2 ≥ x^2}

-- State the theorem to prove the intersection
theorem intersection_M_N :
  M ∩ N = {-1, 0, 1, 2} :=
sorry

end NUMINAMATH_GPT_intersection_M_N_l387_38762


namespace NUMINAMATH_GPT_max_students_on_field_trip_l387_38706

theorem max_students_on_field_trip 
  (bus_cost : ℕ := 100)
  (bus_capacity : ℕ := 25)
  (student_admission_cost_high : ℕ := 10)
  (student_admission_cost_low : ℕ := 8)
  (discount_threshold : ℕ := 20)
  (teacher_cost : ℕ := 0)
  (budget : ℕ := 350) :
  max_students ≤ bus_capacity ↔ bus_cost + 
  (if max_students ≥ discount_threshold then max_students * student_admission_cost_low
  else max_students * student_admission_cost_high) 
   ≤ budget := 
sorry

end NUMINAMATH_GPT_max_students_on_field_trip_l387_38706


namespace NUMINAMATH_GPT_prob_white_ball_is_0_25_l387_38795

-- Let's define the conditions and the statement for the proof
variable (P_red P_white P_yellow : ℝ)

-- The given conditions 
def prob_red_or_white : Prop := P_red + P_white = 0.65
def prob_yellow_or_white : Prop := P_yellow + P_white = 0.6

-- The statement we want to prove
theorem prob_white_ball_is_0_25 (h1 : prob_red_or_white P_red P_white)
                               (h2 : prob_yellow_or_white P_yellow P_white) :
  P_white = 0.25 :=
sorry

end NUMINAMATH_GPT_prob_white_ball_is_0_25_l387_38795


namespace NUMINAMATH_GPT_tabitha_item_cost_l387_38713

theorem tabitha_item_cost :
  ∀ (start_money gave_mom invest fraction_remain spend item_count remain_money item_cost : ℝ),
    start_money = 25 →
    gave_mom = 8 →
    invest = (start_money - gave_mom) / 2 →
    fraction_remain = start_money - gave_mom - invest →
    spend = fraction_remain - remain_money →
    item_count = 5 →
    remain_money = 6 →
    item_cost = spend / item_count →
    item_cost = 0.5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_tabitha_item_cost_l387_38713


namespace NUMINAMATH_GPT_triangle_inequality_min_diff_l387_38784

theorem triangle_inequality_min_diff
  (DE EF FD : ℕ) 
  (h1 : DE + EF + FD = 398)
  (h2 : DE < EF ∧ EF ≤ FD) : 
  EF - DE = 1 :=
by
  sorry

end NUMINAMATH_GPT_triangle_inequality_min_diff_l387_38784


namespace NUMINAMATH_GPT_num_two_digit_math_representation_l387_38703

-- Define the problem space
def unique_digits (n : ℕ) : Prop := 
  n >= 1 ∧ n <= 9

-- Representation of the characters' assignment
def representation (x y z w : ℕ) : Prop :=
  unique_digits x ∧ unique_digits y ∧ unique_digits z ∧ unique_digits w ∧
  x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w ∧ 
  x = z ∧ 3 * (10 * y + z) = 10 * w + x

-- The main theorem to prove
theorem num_two_digit_math_representation : 
  ∃ x y z w, representation x y z w :=
sorry

end NUMINAMATH_GPT_num_two_digit_math_representation_l387_38703


namespace NUMINAMATH_GPT_running_laps_l387_38780

theorem running_laps (A B : ℕ)
  (h_ratio : ∀ t : ℕ, (A * t) = 5 * (B * t) / 3)
  (h_start : A = 5 ∧ B = 3 ∧ ∀ t : ℕ, (A * t) - (B * t) = 4) :
  (B * 2 = 6) ∧ (A * 2 = 10) :=
by
  sorry

end NUMINAMATH_GPT_running_laps_l387_38780


namespace NUMINAMATH_GPT_cn_squared_eq_28_l387_38775

theorem cn_squared_eq_28 (n : ℕ) (h : n * (n - 1) / 2 = 28) : n = 8 :=
sorry

end NUMINAMATH_GPT_cn_squared_eq_28_l387_38775


namespace NUMINAMATH_GPT_union_of_M_and_N_l387_38753

open Set

theorem union_of_M_and_N :
  let M := {x : ℝ | x^2 - 4 * x < 0}
  let N := {x : ℝ | |x| ≤ 2}
  M ∪ N = {x : ℝ | -2 ≤ x ∧ x < 4} :=
by
  sorry

end NUMINAMATH_GPT_union_of_M_and_N_l387_38753


namespace NUMINAMATH_GPT_math_proof_statement_l387_38756

open Real

noncomputable def proof_problem (x : ℝ) : Prop :=
  let a := (cos x, sin x)
  let b := (sqrt 2, sqrt 2)
  (a.1 * b.1 + a.2 * b.2 = 8 / 5) ∧ (π / 4 < x ∧ x < π / 2) ∧ 
  (cos (x - π / 4) = 4 / 5) ∧ (tan (x - π / 4) = 3 / 4) ∧ 
  (sin (2 * x) * (1 - tan x) / (1 + tan x) = -21 / 100)

theorem math_proof_statement (x : ℝ) : proof_problem x := 
by
  unfold proof_problem
  sorry

end NUMINAMATH_GPT_math_proof_statement_l387_38756


namespace NUMINAMATH_GPT_find_value_of_N_l387_38712

theorem find_value_of_N :
  (2 * ((3.6 * 0.48 * 2.5) / (0.12 * 0.09 * 0.5)) = 1600.0000000000002) :=
by {
  sorry
}

end NUMINAMATH_GPT_find_value_of_N_l387_38712


namespace NUMINAMATH_GPT_sequence_arithmetic_and_find_an_l387_38726

theorem sequence_arithmetic_and_find_an (a : ℕ → ℝ)
  (h1 : a 9 = 1 / 7)
  (h2 : ∀ n, a (n + 1) = a n / (3 * a n + 1)) :
  (∀ n, 1 / a (n + 1) = 3 + 1 / a n) ∧ (∀ n, a n = 1 / (3 * n - 20)) :=
by
  sorry

end NUMINAMATH_GPT_sequence_arithmetic_and_find_an_l387_38726


namespace NUMINAMATH_GPT_sum_of_two_digit_factors_is_162_l387_38715

-- Define the number
def num := 6545

-- Define the condition: num can be written as a product of two two-digit numbers
def are_two_digit_numbers (a b : ℕ) : Prop :=
  10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = num

-- The theorem to prove
theorem sum_of_two_digit_factors_is_162 : ∃ a b : ℕ, are_two_digit_numbers a b ∧ a + b = 162 :=
sorry

end NUMINAMATH_GPT_sum_of_two_digit_factors_is_162_l387_38715


namespace NUMINAMATH_GPT_fraction_zero_l387_38798

theorem fraction_zero (x : ℝ) (h : x ≠ -1) (h₀ : (x^2 - 1) / (x + 1) = 0) : x = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_fraction_zero_l387_38798


namespace NUMINAMATH_GPT_problem_solution_l387_38738

def problem_conditions : Prop :=
  (∃ (students_total excellent_students: ℕ) 
     (classA_excellent classB_not_excellent: ℕ),
     students_total = 110 ∧
     excellent_students = 30 ∧
     classA_excellent = 10 ∧
     classB_not_excellent = 30)

theorem problem_solution
  (students_total excellent_students: ℕ)
  (classA_excellent classB_not_excellent: ℕ)
  (h : problem_conditions) :
  ∃ classA_not_excellent classB_excellent: ℕ,
    classA_not_excellent = 50 ∧
    classB_excellent = 20 ∧
    ((∃ χ_squared: ℝ, χ_squared = 7.5 ∧ χ_squared > 6.635) → true) ∧
    (∃ selectA selectB: ℕ, selectA = 5 ∧ selectB = 3) :=
by {
  sorry
}

end NUMINAMATH_GPT_problem_solution_l387_38738


namespace NUMINAMATH_GPT_contractor_absent_days_l387_38709

theorem contractor_absent_days :
  ∃ (x y : ℝ), x + y = 30 ∧ 25 * x - 7.5 * y = 490 ∧ y = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_contractor_absent_days_l387_38709


namespace NUMINAMATH_GPT_determine_p_l387_38776

def is_tangent (circle_eq : ℝ → ℝ → Prop) (parabola_eq : ℝ → ℝ → Prop) (p : ℝ) : Prop :=
  ∃ x y : ℝ, parabola_eq x y ∧ circle_eq x y ∧ x = -p / 2 

noncomputable def circle_eq (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 16
noncomputable def parabola_eq (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x

theorem determine_p (p : ℝ) (hpos : p > 0) :
  (is_tangent circle_eq (parabola_eq p) p) ↔ p = 2 := 
sorry

end NUMINAMATH_GPT_determine_p_l387_38776


namespace NUMINAMATH_GPT_sin_cos_fourth_power_l387_38733

theorem sin_cos_fourth_power (θ : ℝ) (h : Real.sin (2 * θ) = 1 / 4) : Real.sin θ ^ 4 + Real.cos θ ^ 4 = 63 / 64 :=
by
  sorry

end NUMINAMATH_GPT_sin_cos_fourth_power_l387_38733


namespace NUMINAMATH_GPT_cube_face_sharing_l387_38707

theorem cube_face_sharing (n : ℕ) :
  (∃ W B : ℕ, (W + B = n^3) ∧ (3 * W = 3 * B) ∧ W = B ∧ W = n^3 / 2) ↔ n % 2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_cube_face_sharing_l387_38707


namespace NUMINAMATH_GPT_min_value_reciprocal_sum_l387_38754

theorem min_value_reciprocal_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x + y + z = 3) :
  (1 / x) + (1 / y) + (1 / z) ≥ 3 :=
sorry

end NUMINAMATH_GPT_min_value_reciprocal_sum_l387_38754


namespace NUMINAMATH_GPT_johns_final_amount_l387_38729

def initial_amount : ℝ := 45.7
def deposit_amount : ℝ := 18.6
def withdrawal_amount : ℝ := 20.5

theorem johns_final_amount : initial_amount + deposit_amount - withdrawal_amount = 43.8 :=
by
  sorry

end NUMINAMATH_GPT_johns_final_amount_l387_38729


namespace NUMINAMATH_GPT_inequality_inequality_l387_38739

theorem inequality_inequality (x y z : ℝ) (hx : x > -1) (hy : y > -1) (hz : z > -1) : 
  (1 + x^2) / (1 + y + z^2) + (1 + y^2) / (1 + z + x^2) + (1 + z^2) / (1 + x + y^2) ≥ 2 :=
by sorry

end NUMINAMATH_GPT_inequality_inequality_l387_38739


namespace NUMINAMATH_GPT_range_m_n_l387_38750

noncomputable def f (m n x: ℝ) : ℝ := m * Real.exp x + x^2 + n * x

theorem range_m_n (m n: ℝ) :
  (∃ x, f m n x = 0) ∧ (∀ x, f m n x = 0 ↔ f m n (f m n x) = 0) →
  0 ≤ m + n ∧ m + n < 4 :=
by
  sorry

end NUMINAMATH_GPT_range_m_n_l387_38750


namespace NUMINAMATH_GPT_probability_point_closer_to_7_than_0_l387_38751

noncomputable def segment_length (a b : ℝ) : ℝ := b - a
noncomputable def closer_segment (a c b : ℝ) : ℝ := segment_length c b

theorem probability_point_closer_to_7_than_0 :
  let a := 0
  let b := 10
  let c := 7
  let midpoint := (a + c) / 2
  let total_length := b - a
  let closer_length := segment_length midpoint b
  (closer_length / total_length) = 0.7 :=
by
  sorry

end NUMINAMATH_GPT_probability_point_closer_to_7_than_0_l387_38751
