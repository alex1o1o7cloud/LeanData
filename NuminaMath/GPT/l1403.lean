import Mathlib

namespace probability_exactly_one_six_probability_at_least_one_six_probability_at_most_one_six_l1403_140311

-- Considering a die with 6 faces
def die_faces := 6

-- Total number of possible outcomes when rolling 3 dice
def total_outcomes := die_faces^3

-- 1. Probability of having exactly one die showing a 6 when rolling 3 dice
def prob_exactly_one_six : ℚ :=
  have favorable_outcomes := 3 * 5^2 -- 3 ways to choose which die shows 6, and 25 ways for others to not show 6
  favorable_outcomes / total_outcomes

-- Proof statement
theorem probability_exactly_one_six : prob_exactly_one_six = 25/72 := by 
  sorry

-- 2. Probability of having at least one die showing a 6 when rolling 3 dice
def prob_at_least_one_six : ℚ :=
  have no_six_outcomes := 5^3
  (total_outcomes - no_six_outcomes) / total_outcomes

-- Proof statement
theorem probability_at_least_one_six : prob_at_least_one_six = 91/216 := by 
  sorry

-- 3. Probability of having at most one die showing a 6 when rolling 3 dice
def prob_at_most_one_six : ℚ :=
  have no_six_probability := 125 / total_outcomes
  have one_six_probability := 75 / total_outcomes
  no_six_probability + one_six_probability

-- Proof statement
theorem probability_at_most_one_six : prob_at_most_one_six = 25/27 := by 
  sorry

end probability_exactly_one_six_probability_at_least_one_six_probability_at_most_one_six_l1403_140311


namespace eugene_pencils_after_giving_l1403_140314

-- Define Eugene's initial number of pencils and the number of pencils given away.
def initial_pencils : ℝ := 51.0
def pencils_given : ℝ := 6.0

-- State the theorem that should be proved.
theorem eugene_pencils_after_giving : initial_pencils - pencils_given = 45.0 :=
by
  -- We would normally provide the proof steps here, but as per instructions, we'll use "sorry" to skip it.
  sorry

end eugene_pencils_after_giving_l1403_140314


namespace trajectory_of_Q_l1403_140395

variables {P Q M : ℝ × ℝ}

-- Define the conditions as Lean predicates
def is_midpoint (M P Q : ℝ × ℝ) : Prop :=
  M = (0, 4) ∧ M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

def point_on_line (P : ℝ × ℝ) : Prop :=
  P.1 + P.2 - 2 = 0

-- Define the theorem that needs to be proven
theorem trajectory_of_Q :
  (∃ P Q M : ℝ × ℝ, is_midpoint M P Q ∧ point_on_line P) →
  ∃ Q : ℝ × ℝ, (∀ P : ℝ × ℝ, point_on_line P → is_midpoint (0,4) P Q → Q.1 + Q.2 - 6 = 0) :=
by sorry

end trajectory_of_Q_l1403_140395


namespace WR_eq_35_l1403_140305

theorem WR_eq_35 (PQ ZY SX : ℝ) (hPQ : PQ = 30) (hZY : ZY = 15) (hSX : SX = 10) :
    let WS := ZY - SX
    let SR := PQ
    let WR := WS + SR
    WR = 35 := by
  sorry

end WR_eq_35_l1403_140305


namespace term_of_sequence_l1403_140379

def S (n : ℕ) : ℚ := n^2 + 2/3

def a (n : ℕ) : ℚ :=
  if n = 1 then 5/3
  else 2 * n - 1

theorem term_of_sequence (n : ℕ) : a n = 
  if n = 1 then S n 
  else S n - S (n - 1) :=
by
  sorry

end term_of_sequence_l1403_140379


namespace number_of_girls_l1403_140354

theorem number_of_girls (total_children boys girls : ℕ) 
    (total_children_eq : total_children = 60)
    (boys_eq : boys = 22)
    (compute_girls : girls = total_children - boys) : 
    girls = 38 :=
by
    rw [total_children_eq, boys_eq] at compute_girls
    simp at compute_girls
    exact compute_girls

end number_of_girls_l1403_140354


namespace positive_reals_inequality_l1403_140332

variable {a b c : ℝ}

theorem positive_reals_inequality (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_prod : a * b * c = 1) :
  (a * b)^(1/4) + (b * c)^(1/4) + (c * a)^(1/4) < 1/4 := 
sorry

end positive_reals_inequality_l1403_140332


namespace find_soma_cubes_for_shape_l1403_140324

def SomaCubes (n : ℕ) : Type := 
  if n = 1 
  then Fin 3 
  else if 2 ≤ n ∧ n ≤ 7 
       then Fin 4 
       else Fin 0

theorem find_soma_cubes_for_shape :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  SomaCubes a = Fin 3 ∧ SomaCubes b = Fin 4 ∧ SomaCubes c = Fin 4 ∧ 
  a + b + c = 11 ∧ ((a, b, c) = (1, 3, 5) ∨ (a, b, c) = (1, 3, 6)) := 
by
  sorry

end find_soma_cubes_for_shape_l1403_140324


namespace parabolic_points_l1403_140340

noncomputable def A (x1 : ℝ) (y1 : ℝ) : Prop := y1 = x1^2 - 3
noncomputable def B (x2 : ℝ) (y2 : ℝ) : Prop := y2 = x2^2 - 3

theorem parabolic_points (x1 x2 y1 y2 : ℝ) (h1 : 0 < x1) (h2 : x1 < x2)
  (hA : A x1 y1) (hB : B x2 y2) : y1 < y2 :=
by
  sorry

end parabolic_points_l1403_140340


namespace function_relation_l1403_140382

def f (x : ℝ) (c : ℝ) : ℝ := x^2 + 4*x + c

theorem function_relation (c : ℝ) :
  f 1 c > f 0 c ∧ f 0 c > f (-2) c := by
  sorry

end function_relation_l1403_140382


namespace f_g_of_3_l1403_140383

def f (x : ℝ) := 4 * x - 3
def g (x : ℝ) := x^2 + 2 * x + 1

theorem f_g_of_3 : f (g 3) = 61 :=
by
  sorry

end f_g_of_3_l1403_140383


namespace roundness_of_hundred_billion_l1403_140380

def roundness (n : ℕ) : ℕ :=
  let pf := n.factorization
  pf 2 + pf 5

theorem roundness_of_hundred_billion : roundness 100000000000 = 22 := by
  sorry

end roundness_of_hundred_billion_l1403_140380


namespace correct_exponentiation_l1403_140375

theorem correct_exponentiation (a : ℝ) : (-2 * a^3) ^ 4 = 16 * a ^ 12 :=
by sorry

end correct_exponentiation_l1403_140375


namespace smallest_arithmetic_geometric_seq_sum_l1403_140310

variable (A B C D : ℕ)

noncomputable def arithmetic_seq (A B C : ℕ) (d : ℕ) : Prop :=
  B - A = d ∧ C - B = d

noncomputable def geometric_seq (B C D : ℕ) : Prop :=
  C = (5 / 3) * B ∧ D = (25 / 9) * B

theorem smallest_arithmetic_geometric_seq_sum :
  ∃ A B C D : ℕ, 
    arithmetic_seq A B C 12 ∧ 
    geometric_seq B C D ∧ 
    (A + B + C + D = 104) :=
sorry

end smallest_arithmetic_geometric_seq_sum_l1403_140310


namespace inequality_proof_l1403_140396

theorem inequality_proof (a b : ℤ) (ha : a > 0) (hb : b > 0) : a + b ≤ 1 + a * b :=
by
  sorry

end inequality_proof_l1403_140396


namespace reciprocal_of_one_fifth_l1403_140394

theorem reciprocal_of_one_fifth : (∃ x : ℚ, (1/5) * x = 1 ∧ x = 5) :=
by
  -- The proof goes here, for now we assume it with sorry
  sorry

end reciprocal_of_one_fifth_l1403_140394


namespace max_value_is_one_l1403_140388

noncomputable def max_value (x y z : ℝ) : ℝ :=
  (x^2 - 2 * x * y + y^2) * (x^2 - 2 * x * z + z^2) * (y^2 - 2 * y * z + z^2)

theorem max_value_is_one :
  ∀ (x y z : ℝ), 0 ≤ x → 0 ≤ y → 0 ≤ z → x + y + z = 3 →
  max_value x y z ≤ 1 :=
by sorry

end max_value_is_one_l1403_140388


namespace greatest_x_value_l1403_140377

theorem greatest_x_value : 
  ∃ x : ℝ, (∀ y : ℝ, (y = (4 * x - 16) / (3 * x - 4)) → (y^2 + y = 12)) ∧ (x = 2) := by
  sorry

end greatest_x_value_l1403_140377


namespace squares_difference_sum_l1403_140345

theorem squares_difference_sum : 
  23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 288 :=
by 
  sorry

end squares_difference_sum_l1403_140345


namespace aluminum_percentage_range_l1403_140363

variable (x1 x2 x3 y : ℝ)

theorem aluminum_percentage_range:
  (0.15 * x1 + 0.3 * x2 = 0.2) →
  (x1 + x2 + x3 = 1) →
  y = 0.6 * x1 + 0.45 * x3 →
  (1/3 ≤ x2 ∧ x2 ≤ 2/3) →
  (0.15 ≤ y ∧ y ≤ 0.4) := by
  sorry

end aluminum_percentage_range_l1403_140363


namespace mrs_hilt_walks_240_feet_l1403_140369

-- Define the distances and trips as given conditions
def distance_to_fountain : ℕ := 30
def trips_to_fountain : ℕ := 4
def round_trip_distance : ℕ := 2 * distance_to_fountain
def total_distance_walked (round_trip_distance trips_to_fountain : ℕ) : ℕ :=
  round_trip_distance * trips_to_fountain

-- State the theorem
theorem mrs_hilt_walks_240_feet :
  total_distance_walked round_trip_distance trips_to_fountain = 240 :=
by
  sorry

end mrs_hilt_walks_240_feet_l1403_140369


namespace max_value_b_minus_inv_a_is_minus_one_min_value_inv_3a_plus_1_plus_inv_a_plus_b_is_one_l1403_140317

open Real

noncomputable def max_value_b_minus_inv_a (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 4 * a + b = 3) : ℝ :=
b - (1 / a)

noncomputable def min_value_inv_3a_plus_1_plus_inv_a_plus_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 4 * a + b = 3) : ℝ :=
(1 / (3 * a + 1)) + (1 / (a + b))

theorem max_value_b_minus_inv_a_is_minus_one (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 4 * a + b = 3) : 
  max_value_b_minus_inv_a a b ha hb h = -1 :=
sorry

theorem min_value_inv_3a_plus_1_plus_inv_a_plus_b_is_one (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 4 * a + b = 3) : 
  min_value_inv_3a_plus_1_plus_inv_a_plus_b a b ha hb h = 1 :=
sorry

end max_value_b_minus_inv_a_is_minus_one_min_value_inv_3a_plus_1_plus_inv_a_plus_b_is_one_l1403_140317


namespace phil_final_quarters_l1403_140359

-- Define the conditions
def initial_quarters : ℕ := 50
def doubled_initial_quarters : ℕ := 2 * initial_quarters
def quarters_collected_each_month : ℕ := 3
def months_in_year : ℕ := 12
def quarters_collected_in_a_year : ℕ := quarters_collected_each_month * months_in_year
def quarters_collected_every_third_month : ℕ := 1
def quarters_collected_in_third_months : ℕ := months_in_year / 3 * quarters_collected_every_third_month
def total_before_losing : ℕ := doubled_initial_quarters + quarters_collected_in_a_year + quarters_collected_in_third_months
def lost_quarter_of_total : ℕ := total_before_losing / 4
def quarters_left : ℕ := total_before_losing - lost_quarter_of_total

-- Prove the final result
theorem phil_final_quarters : quarters_left = 105 := by
  sorry

end phil_final_quarters_l1403_140359


namespace plywood_perimeter_difference_l1403_140303

theorem plywood_perimeter_difference :
  let l := 10
  let w := 6
  let n := 6
  ∃ p_max p_min, 
    (l * w) % n = 0 ∧
    (p_max = 24) ∧
    (p_min = 12.66) ∧
    p_max - p_min = 11.34 := 
by
  sorry

end plywood_perimeter_difference_l1403_140303


namespace certain_event_is_A_l1403_140348

def conditions (option_A option_B option_C option_D : Prop) : Prop :=
  option_A ∧ ¬option_B ∧ ¬option_C ∧ ¬option_D

theorem certain_event_is_A 
  (option_A option_B option_C option_D : Prop)
  (hconditions : conditions option_A option_B option_C option_D) : 
  ∀ e, (e = option_A) := 
by
  sorry

end certain_event_is_A_l1403_140348


namespace find_u_values_l1403_140374

namespace MathProof

variable (u v : ℝ)
variable (h1 : u ≠ 0) (h2 : v ≠ 0)
variable (h3 : u + 1/v = 8) (h4 : v + 1/u = 16/3)

theorem find_u_values : u = 4 + Real.sqrt 232 / 4 ∨ u = 4 - Real.sqrt 232 / 4 :=
by {
  sorry
}

end MathProof

end find_u_values_l1403_140374


namespace train_length_l1403_140361

noncomputable def speed_kmph := 90
noncomputable def time_sec := 5
noncomputable def speed_mps := speed_kmph * 1000 / 3600

theorem train_length : (speed_mps * time_sec) = 125 := by
  -- We need to assert and prove this theorem
  sorry

end train_length_l1403_140361


namespace fraction_of_paint_first_week_l1403_140362

-- Definitions based on conditions
def total_paint := 360
def fraction_first_week (f : ℚ) : ℚ := f * total_paint
def paint_remaining_first_week (f : ℚ) : ℚ := total_paint - fraction_first_week f
def fraction_second_week (f : ℚ) : ℚ := (1 / 5) * paint_remaining_first_week f
def total_paint_used (f : ℚ) : ℚ := fraction_first_week f + fraction_second_week f
def total_paint_used_value := 104

-- Proof problem statement
theorem fraction_of_paint_first_week (f : ℚ) (h : total_paint_used f = total_paint_used_value) : f = 1 / 9 := 
sorry

end fraction_of_paint_first_week_l1403_140362


namespace line_intersects_ellipse_two_points_l1403_140350

theorem line_intersects_ellipse_two_points {m n : ℝ} (h1 : ¬∃ x y : ℝ, m*x + n*y = 4 ∧ x^2 + y^2 = 4)
  (h2 : m^2 + n^2 < 4) : 
  ∃ p1 p2 : ℝ × ℝ, p1 ≠ p2 ∧ (m * p1.1 + n * p1.2 = 4) ∧ (m * p2.1 + n * p2.2 = 4) ∧ 
  (p1.1^2 / 9 + p1.2^2 / 4 = 1) ∧ (p2.1^2 / 9 + p2.2^2 / 4 = 1) :=
sorry

end line_intersects_ellipse_two_points_l1403_140350


namespace jamal_bought_4_half_dozens_l1403_140320

/-- Given that each crayon costs $2, the total cost is $48, and a half dozen is 6 crayons,
    prove that Jamal bought 4 half dozens of crayons. -/
theorem jamal_bought_4_half_dozens (cost_per_crayon : ℕ) (total_cost : ℕ) (half_dozen : ℕ) 
  (h1 : cost_per_crayon = 2) (h2 : total_cost = 48) (h3 : half_dozen = 6) : 
  (total_cost / cost_per_crayon) / half_dozen = 4 := 
by 
  sorry

end jamal_bought_4_half_dozens_l1403_140320


namespace c_investment_ratio_l1403_140366

-- Conditions as definitions
variables (x : ℕ) (m : ℕ) (total_profit a_share : ℕ)
variables (h_total_profit : total_profit = 19200)
variables (h_a_share : a_share = 6400)

-- Definition of total investment (investments weighted by time)
def total_investment (x m : ℕ) : ℕ :=
  (12 * x) + (6 * 2 * x) + (4 * m * x)

-- Definition of A's share in terms of total investment
def a_share_in_terms_of_total_investment (x : ℕ) (total_investment : ℕ) (total_profit : ℕ) : ℕ :=
  (12 * x * total_profit) / total_investment

-- The theorem stating the ratio of C's investment to A's investment
theorem c_investment_ratio (x m total_profit a_share : ℕ) (h_total_profit : total_profit = 19200)
  (h_a_share : a_share = 6400) (h_a_share_eq : a_share_in_terms_of_total_investment x (total_investment x m) total_profit = a_share) :
  m = 3 :=
by sorry

end c_investment_ratio_l1403_140366


namespace multiplication_simplification_l1403_140336

theorem multiplication_simplification :
  let y := 6742
  let z := 397778
  let approx_mult (a b : ℕ) := 60 * a - a
  z = approx_mult y 59 := sorry

end multiplication_simplification_l1403_140336


namespace range_of_m_l1403_140372

theorem range_of_m (m : ℝ) (x : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → ((m - 2023) * x₁ + m + 2023) > ((m - 2023) * x₂ + m + 2023)) → m < 2023 :=
by
  sorry

end range_of_m_l1403_140372


namespace additional_students_needed_l1403_140349

theorem additional_students_needed 
  (n : ℕ) 
  (r : ℕ) 
  (t : ℕ) 
  (h_n : n = 82) 
  (h_r : r = 2) 
  (h_t : t = 49) : 
  (t - n / r) * r = 16 := 
by 
  sorry

end additional_students_needed_l1403_140349


namespace diagonal_length_of_regular_hexagon_l1403_140302

-- Define a structure for the hexagon with a given side length
structure RegularHexagon (s : ℝ) :=
(side_length : ℝ := s)

-- Prove that the length of diagonal DB in a regular hexagon with side length 12 is 12√3
theorem diagonal_length_of_regular_hexagon (H : RegularHexagon 12) : 
  ∃ DB : ℝ, DB = 12 * Real.sqrt 3 :=
by
  sorry

end diagonal_length_of_regular_hexagon_l1403_140302


namespace adam_has_9_apples_l1403_140368

def jackie_apples : ℕ := 6
def difference : ℕ := 3

def adam_apples (j : ℕ) (d : ℕ) : ℕ := 
  j + d

theorem adam_has_9_apples : adam_apples jackie_apples difference = 9 := 
by 
  sorry

end adam_has_9_apples_l1403_140368


namespace problem_solution_l1403_140307

-- Define the main theorem
theorem problem_solution (x : ℝ) : (x + 2) ^ 2 + 2 * (x + 2) * (4 - x) + (4 - x) ^ 2 = 36 := 
by
  sorry

end problem_solution_l1403_140307


namespace JulieCompletesInOneHour_l1403_140315

-- Define conditions
def JuliePeelsIn : ℕ := 10
def TedPeelsIn : ℕ := 8
def TimeTogether : ℕ := 4

-- Define their respective rates
def JulieRate : ℚ := 1 / JuliePeelsIn
def TedRate : ℚ := 1 / TedPeelsIn

-- Define the task completion in 4 hours together
def TaskCompletedTogether : ℚ := (JulieRate * TimeTogether) + (TedRate * TimeTogether)

-- Define remaining task after working together
def RemainingTask : ℚ := 1 - TaskCompletedTogether

-- Define time for Julie to complete the remaining task
def TimeForJulieToComplete : ℚ := RemainingTask / JulieRate

-- The theorem statement
theorem JulieCompletesInOneHour :
  TimeForJulieToComplete = 1 := by
  sorry

end JulieCompletesInOneHour_l1403_140315


namespace max_value_4x_plus_3y_l1403_140342

theorem max_value_4x_plus_3y :
  ∃ x y : ℝ, (x^2 + y^2 = 16 * x + 8 * y + 8) ∧ (∀ w, w = 4 * x + 3 * y → w ≤ 64) ∧ ∃ x y, 4 * x + 3 * y = 64 :=
sorry

end max_value_4x_plus_3y_l1403_140342


namespace solve_quadratic_eq_l1403_140329

theorem solve_quadratic_eq (x : ℝ) :
  x^2 - 4 * x + 2 = 0 ↔ x = 2 + Real.sqrt 2 ∨ x = 2 - Real.sqrt 2 := by
  sorry

end solve_quadratic_eq_l1403_140329


namespace triangle_AC_range_l1403_140335

noncomputable def length_AB : ℝ := 12
noncomputable def length_CD : ℝ := 6

def is_valid_AC (AC : ℝ) : Prop :=
  AC > 6 ∧ AC < 24

theorem triangle_AC_range :
  ∃ m n : ℝ, 
    (6 < m ∧ m < 24) ∧ (6 < n ∧ n < 24) ∧
    m + n = 30 ∧
    ∀ AC : ℝ, is_valid_AC AC →
      6 < AC ∧ AC < 24 :=
by
  use 6
  use 24
  simp
  sorry

end triangle_AC_range_l1403_140335


namespace max_zoo_area_l1403_140399

theorem max_zoo_area (length width x y : ℝ) (h1 : length = 16) (h2 : width = 8 - x) (h3 : y = x * (8 - x)) : 
  ∃ M, ∀ x, 0 < x ∧ x < 8 → y ≤ M ∧ M = 16 :=
by
  sorry

end max_zoo_area_l1403_140399


namespace will_money_left_l1403_140358

theorem will_money_left (initial sweater tshirt shoes refund_percentage : ℕ) 
  (h_initial : initial = 74)
  (h_sweater : sweater = 9)
  (h_tshirt : tshirt = 11)
  (h_shoes : shoes = 30)
  (h_refund_percentage : refund_percentage = 90) : 
  initial - (sweater + tshirt + (100 - refund_percentage) * shoes / 100) = 51 := by
  sorry

end will_money_left_l1403_140358


namespace find_expression_value_l1403_140381

theorem find_expression_value (x y : ℕ) (h₁ : x = 3) (h₂ : y = 4) :
  (x^3 + 3 * y^3) / 9 = 73 / 3 :=
by
  sorry

end find_expression_value_l1403_140381


namespace efficiency_ratio_l1403_140301

theorem efficiency_ratio (A B : ℝ) (h1 : A ≠ B)
  (h2 : A + B = 1 / 7)
  (h3 : B = 1 / 21) :
  A / B = 2 :=
by
  sorry

end efficiency_ratio_l1403_140301


namespace jennie_rental_cost_is_306_l1403_140325

-- Definitions for the given conditions
def weekly_rate_mid_size : ℕ := 190
def daily_rate_mid_size_upto10 : ℕ := 25
def total_rental_days : ℕ := 13
def coupon_discount : ℝ := 0.10

-- Define the cost calculation
def rental_cost (days : ℕ) : ℕ :=
  let weeks := days / 7
  let extra_days := days % 7
  let cost_weeks := weeks * weekly_rate_mid_size
  let cost_extra := extra_days * daily_rate_mid_size_upto10
  cost_weeks + cost_extra

def discount (total : ℝ) (rate : ℝ) : ℝ := total * rate

def final_amount (initial_amount : ℝ) (discount_amount : ℝ) : ℝ := initial_amount - discount_amount

-- Main theorem to prove the final payment amount
theorem jennie_rental_cost_is_306 : 
  final_amount (rental_cost total_rental_days) (discount (rental_cost total_rental_days) coupon_discount) = 306 := 
by
  sorry

end jennie_rental_cost_is_306_l1403_140325


namespace regular_price_per_can_l1403_140387

variable (P : ℝ) -- Regular price per can

-- Condition: The regular price per can is discounted 15 percent when the soda is purchased in 24-can cases
def discountedPricePerCan (P : ℝ) : ℝ :=
  0.85 * P

-- Condition: The price of 72 cans purchased in 24-can cases is $18.36
def priceOf72CansInDollars : ℝ :=
  18.36

-- Predicate describing the condition that the price of 72 cans is 18.36
axiom h : (72 * discountedPricePerCan P) = priceOf72CansInDollars

theorem regular_price_per_can (P : ℝ) (h : (72 * discountedPricePerCan P) = priceOf72CansInDollars) : P = 0.30 :=
by
  sorry

end regular_price_per_can_l1403_140387


namespace find_number_divided_l1403_140316

theorem find_number_divided (x : ℝ) (h : x / 1.33 = 48) : x = 63.84 :=
by
  sorry

end find_number_divided_l1403_140316


namespace sum_A_B_C_zero_l1403_140389

noncomputable def poly : Polynomial ℝ := Polynomial.X^3 - 16 * Polynomial.X^2 + 72 * Polynomial.X - 27

noncomputable def exists_real_A_B_C 
  (p q r: ℝ) (hpqr: p ≠ q ∧ q ≠ r ∧ p ≠ r) 
  (hrootsp: Polynomial.eval p poly = 0) (hrootsq: Polynomial.eval q poly = 0)
  (hrootsr: Polynomial.eval r poly = 0) :
  ∃ (A B C: ℝ), (∀ s, s ≠ p → s ≠ q → s ≠ r → (1 / (s^3 - 16*s^2 + 72*s - 27) = (A / (s - p)) + (B / (s - q)) + (C / (s - r)))) := sorry

theorem sum_A_B_C_zero 
  {p q r: ℝ} (hpqr: p ≠ q ∧ q ≠ r ∧ p ≠ r) 
  (hrootsp: Polynomial.eval p poly = 0) (hrootsq: Polynomial.eval q poly = 0)
  (hrootsr: Polynomial.eval r poly = 0) 
  (hABC: ∃ (A B C: ℝ), (∀ s, s ≠ p → s ≠ q → s ≠ r → (1 / (s^3 - 16*s^2 + 72*s - 27) = (A / (s - p)) + (B / (s - q)) + (C / (s - r))))) :
  ∀ A B C, A + B + C = 0 := sorry

end sum_A_B_C_zero_l1403_140389


namespace apples_per_slice_is_two_l1403_140355

def number_of_apples_per_slice (total_apples : ℕ) (total_pies : ℕ) (slices_per_pie : ℕ) : ℕ :=
  total_apples / total_pies / slices_per_pie

theorem apples_per_slice_is_two (total_apples : ℕ) (total_pies : ℕ) (slices_per_pie : ℕ) :
  total_apples = 48 → total_pies = 4 → slices_per_pie = 6 → number_of_apples_per_slice total_apples total_pies slices_per_pie = 2 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end apples_per_slice_is_two_l1403_140355


namespace remaining_sum_avg_l1403_140308

variable (a b : ℕ → ℝ)
variable (h1 : 1 / 6 * (a 1 + a 2 + a 3 + a 4 + a 5 + a 6) = 2.5)
variable (h2 : 1 / 2 * (a 1 + a 2) = 1.1)
variable (h3 : 1 / 2 * (a 3 + a 4) = 1.4)

theorem remaining_sum_avg :
  1 / 2 * (a 5 + a 6) = 5 :=
by
  sorry

end remaining_sum_avg_l1403_140308


namespace grandpa_max_movies_l1403_140344

-- Definition of the conditions
def movie_duration : ℕ := 90

def tuesday_total_minutes : ℕ := 4 * 60 + 30

def tuesday_movies_watched : ℕ := tuesday_total_minutes / movie_duration

def wednesday_movies_watched : ℕ := 2 * tuesday_movies_watched

def total_movies_watched : ℕ := tuesday_movies_watched + wednesday_movies_watched

theorem grandpa_max_movies : total_movies_watched = 9 := by
  sorry

end grandpa_max_movies_l1403_140344


namespace composite_function_evaluation_l1403_140397

def f (x : ℕ) : ℕ := x * x
def g (x : ℕ) : ℕ := x + 2

theorem composite_function_evaluation : f (g 3) = 25 := by
  sorry

end composite_function_evaluation_l1403_140397


namespace inequality1_inequality2_l1403_140385

noncomputable def f (x : ℝ) := abs (x + 1 / 2) + abs (x - 3 / 2)

theorem inequality1 (x : ℝ) : 
  (f x ≤ 3) ↔ (-1 ≤ x ∧ x ≤ 2) := by
sorry

theorem inequality2 (a : ℝ) :
  (∀ x, f x ≥ 1 / 2 * abs (1 - a)) ↔ (-3 ≤ a ∧ a ≤ 5) := by
sorry

end inequality1_inequality2_l1403_140385


namespace k_cubed_divisible_l1403_140351

theorem k_cubed_divisible (k : ℕ) (h : k = 84) : ∃ n : ℕ, k ^ 3 = 592704 * n :=
by
  sorry

end k_cubed_divisible_l1403_140351


namespace inequality_solution_set_l1403_140341

open Set -- Open the Set namespace to work with sets in Lean

theorem inequality_solution_set (x : ℝ) :
  (3 * x - 1) / (2 - x) ≥ 1 ↔ (x ∈ Icc (3 / 4) 2 \ {2}) := 
by
  sorry

end inequality_solution_set_l1403_140341


namespace sum_of_ages_l1403_140330

theorem sum_of_ages (J L : ℕ) (h1 : J = L + 8) (h2 : J + 5 = 3 * (L - 6)) : (J + L) = 39 :=
by {
  -- Proof steps would go here, but are omitted for this task per instructions
  sorry
}

end sum_of_ages_l1403_140330


namespace intersection_of_sets_l1403_140321

def set_A (x : ℝ) : Prop := |x - 1| < 3
def set_B (x : ℝ) : Prop := (x - 1) / (x - 5) < 0

theorem intersection_of_sets : ∀ x : ℝ, (set_A x ∧ set_B x) ↔ 1 < x ∧ x < 4 := 
by sorry

end intersection_of_sets_l1403_140321


namespace bounded_sequence_is_constant_two_l1403_140356

def is_bounded (l : ℕ → ℕ) := ∃ (M : ℕ), ∀ (n : ℕ), l n ≤ M

def satisfies_condition (a : ℕ → ℕ) : Prop :=
∀ n ≥ 3, a n = (a n.pred + a (n.pred.pred)) / (Nat.gcd (a n.pred) (a (n.pred.pred)))

theorem bounded_sequence_is_constant_two (a : ℕ → ℕ) 
  (h1 : is_bounded a) 
  (h2 : satisfies_condition a) : 
  ∀ n : ℕ, a n = 2 :=
sorry

end bounded_sequence_is_constant_two_l1403_140356


namespace remainder_when_divided_l1403_140391

theorem remainder_when_divided (P D Q R D'' Q'' R'' : ℕ) (h1 : P = Q * D + R) (h2 : Q = D'' * Q'' + R'') :
  P % (2 * D * D'') = D * R'' + R := sorry

end remainder_when_divided_l1403_140391


namespace quadratic_sum_constants_l1403_140365

theorem quadratic_sum_constants (a b c : ℝ) 
  (h_eq : ∀ x, a * x^2 + b * x + c = 0 → x = -3 ∨ x = 5)
  (h_min : ∀ x, a * x^2 + b * x + c ≥ 36) 
  (h_at : a * 1^2 + b * 1 + c = 36) :
  a + b + c = 36 :=
sorry

end quadratic_sum_constants_l1403_140365


namespace fraction_simplest_sum_l1403_140323

theorem fraction_simplest_sum (a b : ℕ) (ha : a > 0) (hb : b > 0) 
  (h : (3975 : ℚ) / 10000 = (a : ℚ) / b) 
  (simp : ∀ (c : ℕ), c ∣ a ∧ c ∣ b → c = 1) : a + b = 559 :=
sorry

end fraction_simplest_sum_l1403_140323


namespace sum_of_remainders_l1403_140304

theorem sum_of_remainders (n : ℤ) (h₁ : n % 12 = 5) (h₂ : n % 3 = 2) (h₃ : n % 4 = 1) : 2 + 1 = 3 := by
  sorry

end sum_of_remainders_l1403_140304


namespace average_speed_of_trip_l1403_140331

theorem average_speed_of_trip :
  let distance_local := 60
  let speed_local := 20
  let distance_highway := 120
  let speed_highway := 60
  let total_distance := distance_local + distance_highway
  let time_local := distance_local / speed_local
  let time_highway := distance_highway / speed_highway
  let total_time := time_local + time_highway
  let average_speed := total_distance / total_time
  average_speed = 36 := 
by 
  sorry

end average_speed_of_trip_l1403_140331


namespace general_term_a_general_term_b_sum_c_l1403_140353

-- Problem 1: General term formula for the sequence {a_n}
theorem general_term_a (a : ℕ → ℝ) (S : ℕ → ℝ) (hS : ∀ n, S n = 2 - a n) :
  ∀ n, a n = (1 / 2) ^ (n - 1) := 
sorry

-- Problem 2: General term formula for the sequence {b_n}
theorem general_term_b (b : ℕ → ℝ) (a : ℕ → ℝ) (h_b1 : b 1 = 1)
  (h_b : ∀ n, b (n + 1) = b n + a n) (h_a : ∀ n, a n = (1 / 2) ^ (n - 1)) :
  ∀ n, b n = 3 - 2 * (1 / 2) ^ (n - 1) := 
sorry

-- Problem 3: Sum of the first n terms for the sequence {c_n}
theorem sum_c (c : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ)
  (h_b : ∀ n, b n = 3 - 2 * (1 / 2) ^ (n - 1)) (h_c : ∀ n, c n = n * (3 - b n)) :
  ∀ n, T n = 8 - (8 + 4 * n) * (1 / 2) ^ n := 
sorry

end general_term_a_general_term_b_sum_c_l1403_140353


namespace solve_for_x_l1403_140318

-- Define the problem with the given conditions
def sum_of_triangle_angles (x : ℝ) : Prop := x + 2 * x + 30 = 180

-- State the theorem
theorem solve_for_x : ∀ (x : ℝ), sum_of_triangle_angles x → x = 50 :=
by
  intro x
  intro h
  sorry

end solve_for_x_l1403_140318


namespace share_of_A_l1403_140371

-- Definitions corresponding to the conditions
variables (A B C : ℝ)
variable (total : ℝ := 578)
variable (share_ratio_B_C : ℝ := 1 / 4)
variable (share_ratio_A_B : ℝ := 2 / 3)

-- Conditions
def condition1 : B = share_ratio_B_C * C := by sorry
def condition2 : A = share_ratio_A_B * B := by sorry
def condition3 : A + B + C = total := by sorry

-- The equivalent math proof problem statement
theorem share_of_A :
  A = 68 :=
by sorry

end share_of_A_l1403_140371


namespace remainder_of_n_div_1000_l1403_140322

noncomputable def setS : Set ℕ := {x | 1 ≤ x ∧ x ≤ 15}

def n : ℕ :=
  let T := {x | 4 ≤ x ∧ x ≤ 15}
  (3^12 - 2^12) / 2

theorem remainder_of_n_div_1000 : (n % 1000) = 672 := 
  by sorry

end remainder_of_n_div_1000_l1403_140322


namespace efficiency_ratio_l1403_140347

variable (A_eff B_eff : ℝ)

-- Condition 1: A and B together finish a piece of work in 36 days
def combined_efficiency := A_eff + B_eff = 1 / 36

-- Condition 2: B alone finishes the work in 108 days
def B_efficiency := B_eff = 1 / 108

-- Theorem: Prove that the ratio of A's efficiency to B's efficiency is 2:1
theorem efficiency_ratio (h1 : combined_efficiency A_eff B_eff) (h2 : B_efficiency B_eff) : (A_eff / B_eff) = 2 := by
  sorry

end efficiency_ratio_l1403_140347


namespace determine_range_of_a_l1403_140386

theorem determine_range_of_a (a : ℝ) (h : ∃ x y : ℝ, x ≠ y ∧ a * x^2 - x + 2 = 0 ∧ a * y^2 - y + 2 = 0) : 
  a < 1 / 8 ∧ a ≠ 0 :=
sorry

end determine_range_of_a_l1403_140386


namespace gcd_of_sum_and_squares_l1403_140370

theorem gcd_of_sum_and_squares {a b : ℤ} (h : Int.gcd a b = 1) : 
  Int.gcd (a^2 + b^2) (a + b) = 1 ∨ Int.gcd (a^2 + b^2) (a + b) = 2 := 
by
  sorry

end gcd_of_sum_and_squares_l1403_140370


namespace inequality_transformation_l1403_140343

theorem inequality_transformation (a b : ℝ) (h : a > b) : -3 * a < -3 * b := by
  sorry

end inequality_transformation_l1403_140343


namespace laura_annual_income_l1403_140352

theorem laura_annual_income (I T : ℝ) (q : ℝ)
  (h1 : I > 50000) 
  (h2 : T = 0.01 * q * 30000 + 0.01 * (q + 3) * 20000 + 0.01 * (q + 5) * (I - 50000))
  (h3 : T = 0.01 * (q + 0.5) * I) : I = 56000 := 
by sorry

end laura_annual_income_l1403_140352


namespace associates_more_than_two_years_l1403_140326

-- Definitions based on the given conditions
def total_associates := 100
def second_year_associates_percent := 25
def not_first_year_associates_percent := 75

-- The theorem to prove
theorem associates_more_than_two_years :
  not_first_year_associates_percent - second_year_associates_percent = 50 :=
by
  -- The proof is omitted
  sorry

end associates_more_than_two_years_l1403_140326


namespace sin_neg_1740_eq_sqrt3_div_2_l1403_140334

theorem sin_neg_1740_eq_sqrt3_div_2 : Real.sin (-1740 * Real.pi / 180) = Real.sqrt 3 / 2 :=
by
  sorry

end sin_neg_1740_eq_sqrt3_div_2_l1403_140334


namespace pizza_cost_is_correct_l1403_140367

noncomputable def total_pizza_cost : ℝ :=
  let triple_cheese_pizza_cost := (3 * 10) + (6 * 2 * 2.5)
  let meat_lovers_pizza_cost := (3 * 8) + (4 * 3 * 2.5)
  let veggie_delight_pizza_cost := (6 * 5) + (10 * 1 * 2.5)
  triple_cheese_pizza_cost + meat_lovers_pizza_cost + veggie_delight_pizza_cost

theorem pizza_cost_is_correct : total_pizza_cost = 169 := by
  sorry

end pizza_cost_is_correct_l1403_140367


namespace cube_edge_length_close_to_six_l1403_140346

theorem cube_edge_length_close_to_six
  (a V S : ℝ)
  (h1 : V = a^3)
  (h2 : S = 6 * a^2)
  (h3 : V = S + 1) : abs (a - 6) < 1 :=
by
  sorry

end cube_edge_length_close_to_six_l1403_140346


namespace problem_statement_l1403_140384

-- We begin by stating the variables x and y with the given conditions
variables (x y : ℝ)

-- Given conditions
axiom h1 : x - 2 * y = 3
axiom h2 : (x - 2) * (y + 1) = 2

-- The theorem to prove
theorem problem_statement : (x^2 - 2) * (2 * y^2 - 1) = -9 :=
by
  sorry

end problem_statement_l1403_140384


namespace Matthias_fewer_fish_l1403_140398

-- Define the number of fish Micah has
def Micah_fish : ℕ := 7

-- Define the number of fish Kenneth has
def Kenneth_fish : ℕ := 3 * Micah_fish

-- Define the total number of fish
def total_fish : ℕ := 34

-- Define the number of fish Matthias has
def Matthias_fish : ℕ := total_fish - (Micah_fish + Kenneth_fish)

-- State the theorem for the number of fewer fish Matthias has compared to Kenneth
theorem Matthias_fewer_fish : Kenneth_fish - Matthias_fish = 15 := by
  -- Proof goes here
  sorry

end Matthias_fewer_fish_l1403_140398


namespace necessarily_positive_expressions_l1403_140328

theorem necessarily_positive_expressions
  (a b c : ℝ)
  (ha : 0 < a ∧ a < 2)
  (hb : -2 < b ∧ b < 0)
  (hc : 0 < c ∧ c < 3) :
  (b + b^2 > 0) ∧ (b + 3 * b^2 > 0) :=
sorry

end necessarily_positive_expressions_l1403_140328


namespace purchase_price_of_first_commodity_l1403_140300

-- Define the conditions
variable (price_first price_second : ℝ)
variable (h1 : price_first - price_second = 127)
variable (h2 : price_first + price_second = 827)

-- Prove the purchase price of the first commodity is $477
theorem purchase_price_of_first_commodity : price_first = 477 :=
by
  sorry

end purchase_price_of_first_commodity_l1403_140300


namespace asymptotes_of_hyperbola_l1403_140339

-- Definition of hyperbola
def hyperbola_eq (x y : ℝ) : Prop :=
  (x^2 / 4) - y^2 = 1

-- The main theorem to prove
theorem asymptotes_of_hyperbola (x y : ℝ) :
  hyperbola_eq x y → (y = (1/2) * x ∨ y = -(1/2) * x) :=
by 
  sorry

end asymptotes_of_hyperbola_l1403_140339


namespace price_per_pound_of_peanuts_is_2_40_l1403_140319

-- Assume the conditions
def peanuts_price_per_pound (P : ℝ) : Prop :=
  let cashews_price := 6.00
  let mixture_weight := 60
  let mixture_price_per_pound := 3.00
  let cashews_weight := 10
  let total_mixture_price := mixture_weight * mixture_price_per_pound
  let total_cashews_price := cashews_weight * cashews_price
  let total_peanuts_price := total_mixture_price - total_cashews_price
  let peanuts_weight := mixture_weight - cashews_weight
  let P := total_peanuts_price / peanuts_weight
  P = 2.40

-- Prove the price per pound of peanuts
theorem price_per_pound_of_peanuts_is_2_40 (P : ℝ) : peanuts_price_per_pound P :=
by
  sorry

end price_per_pound_of_peanuts_is_2_40_l1403_140319


namespace number_of_tables_l1403_140360

-- Defining the given parameters
def linen_cost : ℕ := 25
def place_setting_cost : ℕ := 10
def rose_cost : ℕ := 5
def lily_cost : ℕ := 4
def num_place_settings : ℕ := 4
def num_roses : ℕ := 10
def num_lilies : ℕ := 15
def total_decoration_cost : ℕ := 3500

-- Defining the cost per table
def cost_per_table : ℕ := linen_cost + (num_place_settings * place_setting_cost) + (num_roses * rose_cost) + (num_lilies * lily_cost)

-- Proof problem statement: Proving number of tables is 20
theorem number_of_tables : (total_decoration_cost / cost_per_table) = 20 :=
by
  sorry

end number_of_tables_l1403_140360


namespace fraction_zero_implies_x_is_minus_5_l1403_140393

theorem fraction_zero_implies_x_is_minus_5 (x : ℝ) (h1 : (x + 5) / (x - 2) = 0) (h2 : x ≠ 2) : x = -5 := 
by
  sorry

end fraction_zero_implies_x_is_minus_5_l1403_140393


namespace min_value_x_squared_plus_y_squared_plus_z_squared_l1403_140378

theorem min_value_x_squared_plus_y_squared_plus_z_squared (x y z : ℝ) (h : x + y + z = 1) : 
  x^2 + y^2 + z^2 ≥ 1/3 :=
by
  sorry

end min_value_x_squared_plus_y_squared_plus_z_squared_l1403_140378


namespace fundamental_disagreement_l1403_140313

-- Definitions based on conditions
def represents_materialism (s : String) : Prop :=
  s = "Without scenery, where does emotion come from?"

def represents_idealism (s : String) : Prop :=
  s = "Without emotion, where does scenery come from?"

-- Theorem statement
theorem fundamental_disagreement :
  ∀ (s1 s2 : String),
  (represents_materialism s1 ∧ represents_idealism s2) →
  (∃ disagreement : String,
    disagreement = "Acknowledging whether the essence of the world is material or consciousness") :=
by
  intros s1 s2 h
  existsi "Acknowledging whether the essence of the world is material or consciousness"
  sorry

end fundamental_disagreement_l1403_140313


namespace union_of_sets_l1403_140309

def M : Set ℝ := {x | x^2 + 2 * x = 0}

def N : Set ℝ := {x | x^2 - 2 * x = 0}

theorem union_of_sets : M ∪ N = {x | x = -2 ∨ x = 0 ∨ x = 2} := sorry

end union_of_sets_l1403_140309


namespace min_value_of_function_l1403_140337

theorem min_value_of_function (x : ℝ) (h : x > 5 / 4) : 
  ∃ ymin : ℝ, ymin = 7 ∧ ∀ y : ℝ, y = 4 * x + 1 / (4 * x - 5) → y ≥ ymin := 
sorry

end min_value_of_function_l1403_140337


namespace veggie_patty_percentage_l1403_140357

-- Let's define the weights
def weight_total : ℕ := 150
def weight_additives : ℕ := 45

-- Let's express the proof statement as a theorem
theorem veggie_patty_percentage : (weight_total - weight_additives) * 100 / weight_total = 70 := by
  sorry

end veggie_patty_percentage_l1403_140357


namespace max_square_plots_l1403_140392
-- Lean 4 statement for the equivalent math problem

theorem max_square_plots (w l f s : ℕ) (h₁ : w = 40) (h₂ : l = 60) 
                         (h₃ : f = 2400) (h₄ : s ≠ 0) (h₅ : 2400 - 100 * s ≤ 2400)
                         (h₆ : w % s = 0) (h₇ : l % s = 0) :
  (w * l) / (s * s) = 6 :=
by {
  sorry
}

end max_square_plots_l1403_140392


namespace correct_systematic_sampling_method_l1403_140327

inductive SamplingMethod
| A
| B
| C
| D

def most_suitable_for_systematic_sampling (A B C D : SamplingMethod) : SamplingMethod :=
SamplingMethod.C

theorem correct_systematic_sampling_method : 
    most_suitable_for_systematic_sampling SamplingMethod.A SamplingMethod.B SamplingMethod.C SamplingMethod.D = SamplingMethod.C :=
by
  sorry

end correct_systematic_sampling_method_l1403_140327


namespace geometric_sequence_a4_l1403_140333

theorem geometric_sequence_a4 (a : ℕ → ℝ) (q : ℝ) 
  (h1 : ∀ n, a (n + 1) = a n * q)
  (h2 : a 2 = 4)
  (h3 : a 6 = 16) : 
  a 4 = 8 :=
by
  -- Proof goes here
  sorry

end geometric_sequence_a4_l1403_140333


namespace sequence_polynomial_l1403_140312

theorem sequence_polynomial (f : ℕ → ℤ) :
  (f 0 = 3 ∧ f 1 = 7 ∧ f 2 = 21 ∧ f 3 = 51) ↔ (∀ n, f n = n^3 + 2 * n^2 + n + 3) :=
by
  sorry

end sequence_polynomial_l1403_140312


namespace paris_total_study_hours_semester_l1403_140306

-- Definitions
def weeks_in_semester := 15
def weekday_study_hours_per_day := 3
def weekdays_per_week := 5
def saturday_study_hours := 4
def sunday_study_hours := 5

-- Theorem statement
theorem paris_total_study_hours_semester :
  weeks_in_semester * (weekday_study_hours_per_day * weekdays_per_week + saturday_study_hours + sunday_study_hours) = 360 := 
sorry

end paris_total_study_hours_semester_l1403_140306


namespace caterpillar_prob_A_l1403_140364

-- Define the probabilities involved
def prob_move_to_A_from_1 (x y z : ℚ) : ℚ :=
  (1/3 : ℚ) * 1 + (1/3 : ℚ) * y + (1/3 : ℚ) * z

def prob_move_to_A_from_2 (x y u : ℚ) : ℚ :=
  (1/3 : ℚ) * 0 + (1/3 : ℚ) * x + (1/3 : ℚ) * u

def prob_move_to_A_from_0 (x y : ℚ) : ℚ :=
  (2/3 : ℚ) * x + (1/3 : ℚ) * y

def prob_move_to_A_from_3 (y u : ℚ) : ℚ :=
  (2/3 : ℚ) * y + (1/3 : ℚ) * u

theorem caterpillar_prob_A :
  exists (x y z u : ℚ), 
    x = prob_move_to_A_from_1 x y z ∧
    y = prob_move_to_A_from_2 x y y ∧
    z = prob_move_to_A_from_0 x y ∧
    u = prob_move_to_A_from_3 y y ∧
    u = y ∧
    x = 9/14 :=
sorry

end caterpillar_prob_A_l1403_140364


namespace smallest_sector_angle_3_l1403_140338

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_angles_is_360 (a : ℕ → ℕ) : Prop :=
  (Finset.range 15).sum a = 360

def smallest_possible_angle (a : ℕ → ℕ) (x : ℕ) : Prop :=
  ∀ i : ℕ, a i ≥ x

theorem smallest_sector_angle_3 :
  ∃ a : ℕ → ℕ,
    is_arithmetic_sequence a ∧
    sum_of_angles_is_360 a ∧
    smallest_possible_angle a 3 :=
sorry

end smallest_sector_angle_3_l1403_140338


namespace student_B_more_consistent_l1403_140373

noncomputable def standard_deviation_A := 5.09
noncomputable def standard_deviation_B := 3.72
def games_played := 7
noncomputable def average_score_A := 16
noncomputable def average_score_B := 16

theorem student_B_more_consistent :
  standard_deviation_B < standard_deviation_A :=
sorry

end student_B_more_consistent_l1403_140373


namespace eight_is_100_discerning_nine_is_not_100_discerning_l1403_140390

-- Define what it means to be b-discerning
def is_b_discerning (n b : ℕ) : Prop :=
  ∃ S : Finset ℕ, S.card = n ∧ (∀ (U V : Finset ℕ), U ≠ V ∧ U ⊆ S ∧ V ⊆ S → U.sum id ≠ V.sum id)

-- Prove that 8 is 100-discerning
theorem eight_is_100_discerning : is_b_discerning 8 100 :=
sorry

-- Prove that 9 is not 100-discerning
theorem nine_is_not_100_discerning : ¬is_b_discerning 9 100 :=
sorry

end eight_is_100_discerning_nine_is_not_100_discerning_l1403_140390


namespace floor_abs_sum_eq_501_l1403_140376

open Int

theorem floor_abs_sum_eq_501 (x : Fin 1004 → ℝ) (h : ∀ i, x i + (i : ℝ) + 1 = (Finset.univ.sum x) + 1005) : 
  Int.floor (abs (Finset.univ.sum x)) = 501 :=
by
  -- Proof steps will go here
  sorry

end floor_abs_sum_eq_501_l1403_140376
