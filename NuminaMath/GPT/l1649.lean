import Mathlib

namespace NUMINAMATH_GPT_final_result_is_four_l1649_164956

theorem final_result_is_four (x : ℕ) (h1 : x = 208) (y : ℕ) (h2 : y = x / 2) (z : ℕ) (h3 : z = y - 100) : z = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_final_result_is_four_l1649_164956


namespace NUMINAMATH_GPT_class_trip_contributions_l1649_164970

theorem class_trip_contributions (x y : ℕ) :
  (x + 5) * (y + 6) = x * y + 792 ∧ (x - 4) * (y + 4) = x * y - 388 → x = 213 ∧ y = 120 := 
by
  sorry

end NUMINAMATH_GPT_class_trip_contributions_l1649_164970


namespace NUMINAMATH_GPT_fraction_addition_l1649_164990

theorem fraction_addition (a b c d : ℚ) (ha : a = 2/5) (hb : b = 3/8) (hc : c = 31/40) :
  a + b = c :=
by
  rw [ha, hb, hc]
  -- The proof part is skipped here as per instructions
  sorry

end NUMINAMATH_GPT_fraction_addition_l1649_164990


namespace NUMINAMATH_GPT_face_value_of_shares_l1649_164958

theorem face_value_of_shares (investment : ℝ) (premium_rate : ℝ) (dividend_rate : ℝ) (dividend_received : ℝ) (F : ℝ)
  (h1 : investment = 14400)
  (h2 : premium_rate = 0.20)
  (h3 : dividend_rate = 0.06)
  (h4 : dividend_received = 720) :
  (1.20 * F = investment) ∧ (0.06 * F = dividend_received) ∧ (F = 12000) :=
by
  sorry

end NUMINAMATH_GPT_face_value_of_shares_l1649_164958


namespace NUMINAMATH_GPT_solve_equation_l1649_164974

theorem solve_equation (x : ℝ) (h₀ : x ≠ 0) (h₁ : x ≠ 1) :
  (x / (x - 1) - 2 / x = 1) ↔ x = 2 :=
sorry

end NUMINAMATH_GPT_solve_equation_l1649_164974


namespace NUMINAMATH_GPT_ratio_children_to_adults_l1649_164959

variable (f m c : ℕ)

-- Conditions
def average_age_female (f : ℕ) := 35
def average_age_male (m : ℕ) := 30
def average_age_child (c : ℕ) := 10
def overall_average_age (f m c : ℕ) := 25

-- Total age sums based on given conditions
def total_age_sum_female (f : ℕ) := 35 * f
def total_age_sum_male (m : ℕ) := 30 * m
def total_age_sum_child (c : ℕ) := 10 * c

-- Total sum and average conditions
def total_age_sum (f m c : ℕ) := total_age_sum_female f + total_age_sum_male m + total_age_sum_child c
def total_members (f m c : ℕ) := f + m + c

theorem ratio_children_to_adults (f m c : ℕ) (h : (total_age_sum f m c) / (total_members f m c) = 25) :
  (c : ℚ) / (f + m) = 2 / 3 := sorry

end NUMINAMATH_GPT_ratio_children_to_adults_l1649_164959


namespace NUMINAMATH_GPT_factorization_correct_l1649_164944

theorem factorization_correct (x y : ℝ) : x^2 * y - x * y^2 = x * y * (x - y) :=
by
  sorry

end NUMINAMATH_GPT_factorization_correct_l1649_164944


namespace NUMINAMATH_GPT_solve_for_x_l1649_164972

def f (x : ℝ) : ℝ := 3 * x - 4

noncomputable def f_inv (x : ℝ) : ℝ := (x + 4) / 3

theorem solve_for_x : ∃ x : ℝ, f x = f_inv x ∧ x = 2 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1649_164972


namespace NUMINAMATH_GPT_max_sum_of_cubes_l1649_164996

open Real

theorem max_sum_of_cubes (a b c d e : ℝ) (h : a^2 + b^2 + c^2 + d^2 + e^2 = 5) :
  a^3 + b^3 + c^3 + d^3 + e^3 ≤ 5 * sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_max_sum_of_cubes_l1649_164996


namespace NUMINAMATH_GPT_fraction_division_l1649_164975

theorem fraction_division :
  (1/4) / 2 = 1/8 :=
by
  sorry

end NUMINAMATH_GPT_fraction_division_l1649_164975


namespace NUMINAMATH_GPT_outfit_count_correct_l1649_164918

def total_shirts : ℕ := 8
def total_pants : ℕ := 4
def total_hats : ℕ := 6
def shirt_colors : Set (String) := {"tan", "black", "blue", "gray", "white", "yellow"}
def hat_colors : Set (String) := {"tan", "black", "blue", "gray", "white", "yellow"}
def conflict_free_outfits (total_shirts total_pants total_hats : ℕ) : ℕ :=
  let total_outfits := total_shirts * total_pants * total_hats
  let matching_outfits := (2 * 1 * 4) * total_pants
  total_outfits - matching_outfits

theorem outfit_count_correct :
  conflict_free_outfits total_shirts total_pants total_hats = 160 :=
by
  unfold conflict_free_outfits
  norm_num
  sorry

end NUMINAMATH_GPT_outfit_count_correct_l1649_164918


namespace NUMINAMATH_GPT_angle_C_of_triangle_l1649_164960

theorem angle_C_of_triangle (A B C : ℝ) (hA : A = 90) (hB : B = 50) (h_sum : A + B + C = 180) : C = 40 := 
by
  sorry

end NUMINAMATH_GPT_angle_C_of_triangle_l1649_164960


namespace NUMINAMATH_GPT_eval_expr_l1649_164926

def a := -1
def b := 1 / 7
def expr := (3 * a^3 - 2 * a * b + b^2) - 2 * (-a^3 - a * b + 4 * b^2)

theorem eval_expr : expr = -36 / 7 := by
  -- Inserting the proof using the original mathematical solution steps is not required here.
  sorry

end NUMINAMATH_GPT_eval_expr_l1649_164926


namespace NUMINAMATH_GPT_number_of_shares_is_25_l1649_164986

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

end NUMINAMATH_GPT_number_of_shares_is_25_l1649_164986


namespace NUMINAMATH_GPT_air_quality_probability_l1649_164925

variable (p_good_day : ℝ) (p_good_two_days : ℝ)

theorem air_quality_probability
  (h1 : p_good_day = 0.75)
  (h2 : p_good_two_days = 0.6) :
  (p_good_two_days / p_good_day = 0.8) :=
by
  rw [h1, h2]
  norm_num

end NUMINAMATH_GPT_air_quality_probability_l1649_164925


namespace NUMINAMATH_GPT_average_of_25_results_l1649_164908

theorem average_of_25_results (first12_avg : ℕ -> ℕ -> ℕ)
                             (last12_avg : ℕ -> ℕ -> ℕ) 
                             (res13 : ℕ)
                             (avg_of_25 : ℕ) :
                             first12_avg 12 10 = 120
                             ∧ last12_avg 12 20 = 240
                             ∧ res13 = 90
                             ∧ avg_of_25 = (first12_avg 12 10 + last12_avg 12 20 + res13) / 25
                             → avg_of_25 = 18 := by
  sorry

end NUMINAMATH_GPT_average_of_25_results_l1649_164908


namespace NUMINAMATH_GPT_opposite_neg_two_l1649_164911

theorem opposite_neg_two : -(-2) = 2 := by
  sorry

end NUMINAMATH_GPT_opposite_neg_two_l1649_164911


namespace NUMINAMATH_GPT_cell_chain_length_l1649_164964

theorem cell_chain_length (d n : ℕ) (h₁ : d = 5 * 10^2) (h₂ : n = 2 * 10^3) : d * n = 10^6 :=
by
  sorry

end NUMINAMATH_GPT_cell_chain_length_l1649_164964


namespace NUMINAMATH_GPT_sum_reciprocals_of_roots_l1649_164982

theorem sum_reciprocals_of_roots (p q x₁ x₂ : ℝ) (h₀ : x₁ + x₂ = -p) (h₁ : x₁ * x₂ = q) :
  (1 / x₁ + 1 / x₂) = -p / q :=
by 
  sorry

end NUMINAMATH_GPT_sum_reciprocals_of_roots_l1649_164982


namespace NUMINAMATH_GPT_difference_in_students_and_guinea_pigs_l1649_164939

def num_students (classrooms : ℕ) (students_per_classroom : ℕ) : ℕ := classrooms * students_per_classroom
def num_guinea_pigs (classrooms : ℕ) (guinea_pigs_per_classroom : ℕ) : ℕ := classrooms * guinea_pigs_per_classroom
def difference_students_guinea_pigs (students : ℕ) (guinea_pigs : ℕ) : ℕ := students - guinea_pigs

theorem difference_in_students_and_guinea_pigs :
  ∀ (classrooms : ℕ) (students_per_classroom : ℕ) (guinea_pigs_per_classroom : ℕ),
  classrooms = 6 →
  students_per_classroom = 24 →
  guinea_pigs_per_classroom = 3 →
  difference_students_guinea_pigs (num_students classrooms students_per_classroom) (num_guinea_pigs classrooms guinea_pigs_per_classroom) = 126 :=
by
  intros
  sorry

end NUMINAMATH_GPT_difference_in_students_and_guinea_pigs_l1649_164939


namespace NUMINAMATH_GPT_train_average_speed_l1649_164955

theorem train_average_speed :
  let start_time := 9.0 -- Start time in hours (9:00 am)
  let end_time := 13.75 -- End time in hours (1:45 pm)
  let total_distance := 348.0 -- Total distance in km
  let halt_time := 0.75 -- Halt time in hours (45 minutes)
  let scheduled_time := end_time - start_time -- Total scheduled time in hours
  let actual_travel_time := scheduled_time - halt_time -- Actual travel time in hours
  let average_speed := total_distance / actual_travel_time -- Average speed formula
  average_speed = 87.0 := sorry

end NUMINAMATH_GPT_train_average_speed_l1649_164955


namespace NUMINAMATH_GPT_average_speed_sf_l1649_164941

variables
  (v d t : ℝ)  -- Representing the average speed to SF, the distance, and time to SF
  (h1 : 42 = (2 * d) / (3 * t))  -- Condition: Average speed of the round trip is 42 mph
  (h2 : t = d / v)  -- Definition of time t in terms of distance and speed

theorem average_speed_sf : v = 63 :=
by
  sorry

end NUMINAMATH_GPT_average_speed_sf_l1649_164941


namespace NUMINAMATH_GPT_part1_part2_l1649_164943

theorem part1 (n : Nat) (hn : 0 < n) : 
  (∃ k, -5^4 + 5^5 + 5^n = k^2) -> n = 5 :=
by
  sorry

theorem part2 (n : Nat) (hn : 0 < n) : 
  (∃ m, 2^4 + 2^7 + 2^n = m^2) -> n = 8 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l1649_164943


namespace NUMINAMATH_GPT_geometric_sequence_sum_eq_five_l1649_164914

/-- Given that {a_n} is a geometric sequence where each a_n > 0
    and the equation a_2 * a_4 + 2 * a_3 * a_5 + a_4 * a_6 = 25 holds,
    we want to prove that a_3 + a_5 = 5. -/
theorem geometric_sequence_sum_eq_five
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geom : ∀ n, a n = a 1 * r ^ (n - 1))
  (h_pos : ∀ n, a n > 0)
  (h_eq : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25) : a 3 + a 5 = 5 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_eq_five_l1649_164914


namespace NUMINAMATH_GPT_number_of_boys_in_second_group_l1649_164916

noncomputable def daily_work_done_by_man (M : ℝ) (B : ℝ) : Prop :=
  M = 2 * B

theorem number_of_boys_in_second_group
  (M B : ℝ)
  (h1 : (12 * M + 16 * B) * 5 = (13 * M + 24 * B) * 4)
  (h2 : daily_work_done_by_man M B) :
  24 = 24 :=
by
  -- The proof is omitted.
  sorry

end NUMINAMATH_GPT_number_of_boys_in_second_group_l1649_164916


namespace NUMINAMATH_GPT_Ted_has_15_bags_l1649_164993

-- Define the parameters
def total_candy_bars : ℕ := 75
def candy_per_bag : ℝ := 5.0

-- Define the assertion to be proved
theorem Ted_has_15_bags : total_candy_bars / candy_per_bag = 15 := 
by
  sorry

end NUMINAMATH_GPT_Ted_has_15_bags_l1649_164993


namespace NUMINAMATH_GPT_nine_otimes_three_l1649_164919

def otimes (a b : ℤ) : ℤ := a + (4 * a) / (3 * b)

theorem nine_otimes_three : otimes 9 3 = 13 := by
  sorry

end NUMINAMATH_GPT_nine_otimes_three_l1649_164919


namespace NUMINAMATH_GPT_discount_percentage_l1649_164971

theorem discount_percentage (coach_cost sectional_cost other_cost paid : ℕ) 
  (h1 : coach_cost = 2500) 
  (h2 : sectional_cost = 3500) 
  (h3 : other_cost = 2000) 
  (h4 : paid = 7200) : 
  ((coach_cost + sectional_cost + other_cost - paid) * 100) / (coach_cost + sectional_cost + other_cost) = 10 :=
by
  sorry

end NUMINAMATH_GPT_discount_percentage_l1649_164971


namespace NUMINAMATH_GPT_minimal_q_for_fraction_l1649_164980

theorem minimal_q_for_fraction :
  ∃ p q : ℕ, 0 < p ∧ 0 < q ∧ 
  (3/5 : ℚ) < p / q ∧ p / q < (5/8 : ℚ) ∧
  (∀ r : ℕ, 0 < r ∧ (3/5 : ℚ) < p / r ∧ p / r < (5/8 : ℚ) → q ≤ r) ∧
  p + q = 21 :=
by
  sorry

end NUMINAMATH_GPT_minimal_q_for_fraction_l1649_164980


namespace NUMINAMATH_GPT_production_difference_correct_l1649_164936

variable (w t M T : ℕ)

-- Condition: w = 2t
def condition_w := w = 2 * t

-- Widgets produced on Monday
def widgets_monday := M = w * t

-- Widgets produced on Tuesday
def widgets_tuesday := T = (w + 5) * (t - 3)

-- Difference in production
def production_difference := M - T = t + 15

theorem production_difference_correct
  (h1 : condition_w w t)
  (h2 : widgets_monday M w t)
  (h3 : widgets_tuesday T w t) :
  production_difference M T t :=
sorry

end NUMINAMATH_GPT_production_difference_correct_l1649_164936


namespace NUMINAMATH_GPT_baseball_cards_start_count_l1649_164947

theorem baseball_cards_start_count (X : ℝ) 
  (h1 : ∃ (x : ℝ), x = (X + 1) / 2)
  (h2 : ∃ (x' : ℝ), x' = X - ((X + 1) / 2) - 1)
  (h3 : ∃ (y : ℝ), y = 3 * (X - ((X + 1) / 2) - 1))
  (h4 : ∃ (z : ℝ), z = 18) : 
  X = 15 :=
by
  sorry

end NUMINAMATH_GPT_baseball_cards_start_count_l1649_164947


namespace NUMINAMATH_GPT_sum_abc_l1649_164995

noncomputable def polynomial : Polynomial ℝ :=
  Polynomial.C (-6) + Polynomial.X * (Polynomial.C 11 + Polynomial.X * (Polynomial.C (-6) + Polynomial.X))

def t (k : ℕ) : ℝ :=
  match k with
  | 0 => 3
  | 1 => 6
  | 2 => 14
  | _ => 0 -- placeholder, as only t_0, t_1, t_2 are given explicitly

def a := 6
def b := -11
def c := 18

def t_rec (k : ℕ) : ℝ :=
  match k with
  | 0 => 3
  | 1 => 6
  | 2 => 14
  | n + 3 => a * t (n + 2) + b * t (n + 1) + c * t n

theorem sum_abc : a + b + c = 13 := by
  sorry

end NUMINAMATH_GPT_sum_abc_l1649_164995


namespace NUMINAMATH_GPT_maximum_y_coordinate_l1649_164969

variable (x y b : ℝ)

def hyperbola (x y b : ℝ) : Prop := (x^2) / 4 - (y^2) / b = 1

def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

def op_condition (x y b : ℝ) : Prop := (x^2 + y^2) = 4 + b

noncomputable def eccentricity (b : ℝ) : ℝ := (Real.sqrt (4 + b)) / 2

theorem maximum_y_coordinate (hb : b > 0) 
                            (h_ec : 1 < eccentricity b ∧ eccentricity b ≤ 2) 
                            (h_hyp : hyperbola x y b) 
                            (h_first : first_quadrant x y) 
                            (h_op : op_condition x y b) 
                            : y ≤ 3 :=
sorry

end NUMINAMATH_GPT_maximum_y_coordinate_l1649_164969


namespace NUMINAMATH_GPT_intersecting_lines_solution_l1649_164935

theorem intersecting_lines_solution (x y b : ℝ) 
  (h₁ : y = 2 * x - 5)
  (h₂ : y = 3 * x + b)
  (hP : x = 1 ∧ y = -3) : 
  b = -6 ∧ x = 1 ∧ y = -3 := by
  sorry

end NUMINAMATH_GPT_intersecting_lines_solution_l1649_164935


namespace NUMINAMATH_GPT_arithmetic_identity_l1649_164997

theorem arithmetic_identity :
  65 * 1515 - 25 * 1515 + 1515 = 62115 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_identity_l1649_164997


namespace NUMINAMATH_GPT_div_by_66_l1649_164922

theorem div_by_66 :
  (43 ^ 23 + 23 ^ 43) % 66 = 0 := 
sorry

end NUMINAMATH_GPT_div_by_66_l1649_164922


namespace NUMINAMATH_GPT_jessica_seashells_l1649_164981

theorem jessica_seashells (joan jessica total : ℕ) (h1 : joan = 6) (h2 : total = 14) (h3 : total = joan + jessica) : jessica = 8 :=
by
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_jessica_seashells_l1649_164981


namespace NUMINAMATH_GPT_initial_fish_count_l1649_164952

theorem initial_fish_count (F T : ℕ) 
  (h1 : T = 3 * F)
  (h2 : T / 2 = (F - 7) + 32) : F = 50 :=
by
  sorry

end NUMINAMATH_GPT_initial_fish_count_l1649_164952


namespace NUMINAMATH_GPT_complement_A_in_U_l1649_164954

open Set

-- Definitions for sets
def U : Set ℕ := {x | 1 < x ∧ x < 5}
def A : Set ℕ := {2, 3}

-- The proof goal: prove that the complement of A in U is {4}
theorem complement_A_in_U : (U \ A) = {4} := by
  sorry

end NUMINAMATH_GPT_complement_A_in_U_l1649_164954


namespace NUMINAMATH_GPT_value_of_b_div_a_l1649_164942

theorem value_of_b_div_a (a b : ℝ) (h : |5 - a| + (b + 3)^2 = 0) : b / a = -3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_value_of_b_div_a_l1649_164942


namespace NUMINAMATH_GPT_harriet_trip_time_l1649_164921

theorem harriet_trip_time :
  ∀ (t1 : ℝ) (s1 s2 t2 d : ℝ), 
  t1 = 2.8 ∧ 
  s1 = 110 ∧ 
  s2 = 140 ∧ 
  d = s1 * t1 ∧ 
  t2 = d / s2 → 
  t1 + t2 = 5 :=
by intros t1 s1 s2 t2 d
   sorry

end NUMINAMATH_GPT_harriet_trip_time_l1649_164921


namespace NUMINAMATH_GPT_intersection_A_B_l1649_164924

open Set

def A : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def B : Set ℕ := {x | 2 ≤ x ∧ x < 6}

theorem intersection_A_B : A ∩ B = {2, 3, 4, 5} := by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l1649_164924


namespace NUMINAMATH_GPT_exists_irrational_an_l1649_164931

theorem exists_irrational_an (a : ℕ → ℝ) (h1 : ∀ n, a n > 0)
  (h2 : ∀ n ≥ 1, a (n + 1)^2 = a n + 1) :
  ∃ n, ¬ ∃ q : ℚ, a n = q :=
sorry

end NUMINAMATH_GPT_exists_irrational_an_l1649_164931


namespace NUMINAMATH_GPT_strawberries_picking_problem_l1649_164945

noncomputable def StrawberriesPicked : Prop :=
  let kg_to_lb := 2.2
  let marco_pounds := 1 + 3 * kg_to_lb
  let sister_pounds := 1.5 * marco_pounds
  let father_pounds := 2 * sister_pounds
  marco_pounds = 7.6 ∧ sister_pounds = 11.4 ∧ father_pounds = 22.8

theorem strawberries_picking_problem : StrawberriesPicked :=
  sorry

end NUMINAMATH_GPT_strawberries_picking_problem_l1649_164945


namespace NUMINAMATH_GPT_proof_cost_A_B_schools_proof_renovation_plans_l1649_164957

noncomputable def cost_A_B_schools : Prop :=
  ∃ (x y : ℝ), 2 * x + 3 * y = 78 ∧ 3 * x + y = 54 ∧ x = 12 ∧ y = 18

noncomputable def renovation_plans : Prop :=
  ∃ (a : ℕ), 3 ≤ a ∧ a ≤ 5 ∧ 
    (1200 - 300) * a + (1800 - 500) * (10 - a) ≤ 11800 ∧
    300 * a + 500 * (10 - a) ≥ 4000

theorem proof_cost_A_B_schools : cost_A_B_schools :=
sorry

theorem proof_renovation_plans : renovation_plans :=
sorry

end NUMINAMATH_GPT_proof_cost_A_B_schools_proof_renovation_plans_l1649_164957


namespace NUMINAMATH_GPT_cakes_sold_to_baked_ratio_l1649_164973

theorem cakes_sold_to_baked_ratio
  (cakes_per_day : ℕ) 
  (days : ℕ)
  (cakes_left : ℕ)
  (total_cakes : ℕ := cakes_per_day * days)
  (cakes_sold : ℕ := total_cakes - cakes_left) :
  cakes_per_day = 20 → 
  days = 9 → 
  cakes_left = 90 → 
  cakes_sold * 2 = total_cakes := 
by 
  intros 
  sorry

end NUMINAMATH_GPT_cakes_sold_to_baked_ratio_l1649_164973


namespace NUMINAMATH_GPT_distinct_m_value_l1649_164991

theorem distinct_m_value (a b : ℝ) (m : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0)
    (h_b_eq_2a : b = 2 * a) (h_m_eq_neg2a_b : m = -2 * a / b) : 
    ∃! (m : ℝ), m = -1 :=
by sorry

end NUMINAMATH_GPT_distinct_m_value_l1649_164991


namespace NUMINAMATH_GPT_cost_of_green_lettuce_l1649_164998

-- Definitions based on the conditions given in the problem
def cost_per_pound := 2
def weight_red_lettuce := 6 / cost_per_pound
def total_weight := 7
def weight_green_lettuce := total_weight - weight_red_lettuce

-- Problem statement: Prove that the cost of green lettuce is $8
theorem cost_of_green_lettuce : (weight_green_lettuce * cost_per_pound) = 8 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_green_lettuce_l1649_164998


namespace NUMINAMATH_GPT_expected_score_particular_player_l1649_164989

-- Define types of dice
inductive DiceType : Type
| A | B | C

-- Define the faces of each dice type
def DiceFaces : DiceType → List ℕ
| DiceType.A => [2, 2, 4, 4, 9, 9]
| DiceType.B => [1, 1, 6, 6, 8, 8]
| DiceType.C => [3, 3, 5, 5, 7, 7]

-- Define a function to calculate the score of a player given their roll and opponents' rolls
def player_score (p_roll : ℕ) (opp_rolls : List ℕ) : ℕ :=
  opp_rolls.foldl (λ acc roll => if roll < p_roll then acc + 1 else acc) 0

-- Define a function to calculate the expected score of a player
noncomputable def expected_score (dice_choice : DiceType) : ℚ :=
  let rolls := DiceFaces dice_choice
  let total_possibilities := (rolls.length : ℚ) ^ 3
  let score_sum := rolls.foldl (λ acc p_roll =>
    acc + rolls.foldl (λ acc1 opp1_roll =>
        acc1 + rolls.foldl (λ acc2 opp2_roll =>
            acc2 + player_score p_roll [opp1_roll, opp2_roll]
          ) 0
      ) 0
    ) 0
  score_sum / total_possibilities

-- The main theorem statement
theorem expected_score_particular_player : (expected_score DiceType.A + expected_score DiceType.B + expected_score DiceType.C) / 3 = 
(8 : ℚ) / 9 := sorry

end NUMINAMATH_GPT_expected_score_particular_player_l1649_164989


namespace NUMINAMATH_GPT_compute_a1d1_a2d2_a3d3_l1649_164910

theorem compute_a1d1_a2d2_a3d3
  (a1 a2 a3 d1 d2 d3 : ℝ)
  (h : ∀ x : ℝ, x^6 + 2 * x^5 + x^4 + x^3 + x^2 + 2 * x + 1 = (x^2 + a1*x + d1) * (x^2 + a2*x + d2) * (x^2 + a3*x + d3)) :
  a1 * d1 + a2 * d2 + a3 * d3 = 2 :=
by
  sorry

end NUMINAMATH_GPT_compute_a1d1_a2d2_a3d3_l1649_164910


namespace NUMINAMATH_GPT_kevin_hops_7_times_l1649_164928

noncomputable def distance_hopped_after_n_hops (n : ℕ) : ℚ :=
  4 * (1 - (3 / 4) ^ n)

theorem kevin_hops_7_times :
  distance_hopped_after_n_hops 7 = 7086 / 2048 := 
by
  sorry

end NUMINAMATH_GPT_kevin_hops_7_times_l1649_164928


namespace NUMINAMATH_GPT_lino_shells_total_l1649_164965

def picked_up_shells : Float := 324.0
def put_back_shells : Float := 292.0

theorem lino_shells_total : picked_up_shells - put_back_shells = 32.0 :=
by
  sorry

end NUMINAMATH_GPT_lino_shells_total_l1649_164965


namespace NUMINAMATH_GPT_average_of_ABC_l1649_164963

theorem average_of_ABC (A B C : ℝ) 
  (h1 : 2002 * C - 1001 * A = 8008) 
  (h2 : 2002 * B + 3003 * A = 7007) 
  (h3 : A = 2) : (A + B + C) / 3 = 2.33 := 
by 
  sorry

end NUMINAMATH_GPT_average_of_ABC_l1649_164963


namespace NUMINAMATH_GPT_binary_1101_to_decimal_l1649_164927

theorem binary_1101_to_decimal : (1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0) = 13 := by
  -- To convert a binary number to its decimal equivalent, we multiply each digit by its corresponding power of 2 based on its position and then sum the results.
  sorry

end NUMINAMATH_GPT_binary_1101_to_decimal_l1649_164927


namespace NUMINAMATH_GPT_min_value_fraction_l1649_164930

theorem min_value_fraction (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + 2 * b + 3 * c = 1) : 
  (1/a + 2/b + 3/c) ≥ 36 := 
sorry

end NUMINAMATH_GPT_min_value_fraction_l1649_164930


namespace NUMINAMATH_GPT_roots_of_polynomial_l1649_164940

theorem roots_of_polynomial : {x : ℝ | (x^2 - 5*x + 6)*(x - 1)*(x - 6) = 0} = {1, 2, 3, 6} :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_roots_of_polynomial_l1649_164940


namespace NUMINAMATH_GPT_factorization_of_x4_plus_16_l1649_164994

theorem factorization_of_x4_plus_16 :
  (x : ℝ) → x^4 + 16 = (x^2 + 2 * x + 2) * (x^2 - 2 * x + 2) :=
by
  intro x
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_factorization_of_x4_plus_16_l1649_164994


namespace NUMINAMATH_GPT_expression_divisibility_l1649_164999

theorem expression_divisibility (x y : ℤ) (k_1 k_2 : ℤ) (h1 : 2 * x + 3 * y = 17 * k_1) :
    ∃ k_2 : ℤ, 9 * x + 5 * y = 17 * k_2 :=
by
  sorry

end NUMINAMATH_GPT_expression_divisibility_l1649_164999


namespace NUMINAMATH_GPT_bingley_bracelets_final_l1649_164985

-- Definitions
def initial_bingley_bracelets : Nat := 5
def kelly_bracelets_given : Nat := 16 / 4
def bingley_bracelets_after_kelly : Nat := initial_bingley_bracelets + kelly_bracelets_given
def bingley_bracelets_given_to_sister : Nat := bingley_bracelets_after_kelly / 3
def bingley_remaining_bracelets : Nat := bingley_bracelets_after_kelly - bingley_bracelets_given_to_sister

-- Theorem
theorem bingley_bracelets_final : bingley_remaining_bracelets = 6 := by
  sorry

end NUMINAMATH_GPT_bingley_bracelets_final_l1649_164985


namespace NUMINAMATH_GPT_quadratic_root_ratio_eq_l1649_164988

theorem quadratic_root_ratio_eq (k : ℝ) :
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ (x = 3 * y ∨ y = 3 * x) ∧ x + y = -10 ∧ x * y = k) → k = 18.75 := by
  sorry

end NUMINAMATH_GPT_quadratic_root_ratio_eq_l1649_164988


namespace NUMINAMATH_GPT_first_tier_tax_rate_l1649_164951

theorem first_tier_tax_rate (price : ℕ) (total_tax : ℕ) (tier1_limit : ℕ) (tier2_rate : ℝ) (tier1_tax_rate : ℝ) :
  price = 18000 →
  total_tax = 1950 →
  tier1_limit = 11000 →
  tier2_rate = 0.09 →
  ((price - tier1_limit) * tier2_rate + tier1_tax_rate * tier1_limit = total_tax) →
  tier1_tax_rate = 0.12 :=
by
  intros hprice htotal htier1 hrate htax_eq
  sorry

end NUMINAMATH_GPT_first_tier_tax_rate_l1649_164951


namespace NUMINAMATH_GPT_no_natural_number_such_that_n_pow_2012_minus_1_is_power_of_two_l1649_164906

theorem no_natural_number_such_that_n_pow_2012_minus_1_is_power_of_two :
  ¬ ∃ (n : ℕ), ∃ (k : ℕ), n^2012 - 1 = 2^k :=
by
  sorry  

end NUMINAMATH_GPT_no_natural_number_such_that_n_pow_2012_minus_1_is_power_of_two_l1649_164906


namespace NUMINAMATH_GPT_tan_triple_angle_l1649_164937

theorem tan_triple_angle (θ : ℝ) (h : Real.tan θ = 3) : Real.tan (3 * θ) = 9 / 13 := by
  sorry

end NUMINAMATH_GPT_tan_triple_angle_l1649_164937


namespace NUMINAMATH_GPT_sally_spent_eur_l1649_164948

-- Define the given conditions
def coupon_value : ℝ := 3
def peaches_total_usd : ℝ := 12.32
def cherries_original_usd : ℝ := 11.54
def discount_rate : ℝ := 0.1
def conversion_rate : ℝ := 0.85

-- Define the intermediate calculations
def cherries_discount_usd : ℝ := cherries_original_usd * discount_rate
def cherries_final_usd : ℝ := cherries_original_usd - cherries_discount_usd
def total_usd : ℝ := peaches_total_usd + cherries_final_usd
def total_eur : ℝ := total_usd * conversion_rate

-- The final statement to be proven
theorem sally_spent_eur : total_eur = 19.30 := by
  sorry

end NUMINAMATH_GPT_sally_spent_eur_l1649_164948


namespace NUMINAMATH_GPT_james_earnings_l1649_164909

theorem james_earnings :
  let jan_earn : ℕ := 4000
  let feb_earn := 2 * jan_earn
  let total_earnings : ℕ := 18000
  let earnings_jan_feb := jan_earn + feb_earn
  let mar_earn := total_earnings - earnings_jan_feb
  (feb_earn - mar_earn) = 2000 := by
  sorry

end NUMINAMATH_GPT_james_earnings_l1649_164909


namespace NUMINAMATH_GPT_train_probability_at_station_l1649_164977

-- Define time intervals
def t0 := 0 -- Train arrival start time in minutes after 1:00 PM
def t1 := 60 -- Train arrival end time in minutes after 1:00 PM
def a0 := 0 -- Alex arrival start time in minutes after 1:00 PM
def a1 := 120 -- Alex arrival end time in minutes after 1:00 PM

-- Define the probability calculation problem
theorem train_probability_at_station :
  let total_area := (t1 - t0) * (a1 - a0)
  let overlap_area := (1/2 * 50 * 50) + (10 * 55)
  (overlap_area / total_area) = 1/4 := 
by
  sorry

end NUMINAMATH_GPT_train_probability_at_station_l1649_164977


namespace NUMINAMATH_GPT_smallest_three_digit_divisible_l1649_164934

theorem smallest_three_digit_divisible :
  ∃ (A B C : Nat), A ≠ 0 ∧ 100 ≤ (100 * A + 10 * B + C) ∧ (100 * A + 10 * B + C) < 1000 ∧
  (10 * A + B) > 9 ∧ (10 * B + C) > 9 ∧ 
  (100 * A + 10 * B + C) % (10 * A + B) = 0 ∧ (100 * A + 10 * B + C) % (10 * B + C) = 0 ∧
  (100 * A + 10 * B + C) = 110 :=
by
  sorry

end NUMINAMATH_GPT_smallest_three_digit_divisible_l1649_164934


namespace NUMINAMATH_GPT_range_of_m_l1649_164979

theorem range_of_m (h : ¬ (∀ x : ℝ, ∃ m : ℝ, 4 ^ x - 2 ^ (x + 1) + m = 0) → false) : 
  ∀ m : ℝ, m ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1649_164979


namespace NUMINAMATH_GPT_functional_equation_solution_l1649_164905

theorem functional_equation_solution :
  ∀ (f : ℚ → ℝ), (∀ x y : ℚ, f (x + y) = f x + f y + 2 * x * y) →
  ∃ k : ℝ, ∀ x : ℚ, f x = x^2 + k * x :=
by
  sorry

end NUMINAMATH_GPT_functional_equation_solution_l1649_164905


namespace NUMINAMATH_GPT_geometric_sequence_general_term_formula_no_arithmetic_sequence_l1649_164983

-- Assume we have a sequence {a_n} and its sum of the first n terms S_n where S_n = 2a_n - n (for n ∈ ℕ*)
variable {a_n : ℕ → ℕ}
variable {S_n : ℕ → ℕ}

-- Condition 1: S_n = 2a_n - n
axiom Sn_condition (n : ℕ) (h : n > 0) : S_n n = 2 * a_n n - n

-- 1. Prove that the sequence {a_n + 1} is a geometric sequence with first term and common ratio equal to 2
theorem geometric_sequence (n : ℕ) (h : n > 0) : ∃ r : ℕ, r = 2 ∧ ∀ m : ℕ, a_n (m + 1) + 1 = r * (a_n m + 1) :=
by
  sorry

-- 2. Prove the general term formula an = 2^n - 1
theorem general_term_formula (n : ℕ) (h : n > 0) : a_n n = 2^n - 1 :=
by
  sorry

-- 3. Prove that there do not exist three consecutive terms in {a_n} that form an arithmetic sequence
theorem no_arithmetic_sequence (n k : ℕ) (h : n > 0 ∧ k > 0 ∧ k + 2 < n) : ¬(a_n k + a_n (k + 2) = 2 * a_n (k + 1)) :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_general_term_formula_no_arithmetic_sequence_l1649_164983


namespace NUMINAMATH_GPT_measure_angle_Z_l1649_164976

-- Given conditions
def triangle_condition (X Y Z : ℝ) :=
   X = 78 ∧ Y = 4 * Z - 14

-- Triangle angle sum property
def triangle_angle_sum (X Y Z : ℝ) :=
   X + Y + Z = 180

-- Prove the measure of angle Z
theorem measure_angle_Z (X Y Z : ℝ) (h1 : triangle_condition X Y Z) (h2 : triangle_angle_sum X Y Z) : 
  Z = 23.2 :=
by
  -- Lean will expect proof steps here, ‘sorry’ is used to denote unproven parts.
  sorry

end NUMINAMATH_GPT_measure_angle_Z_l1649_164976


namespace NUMINAMATH_GPT_circle_center_coordinates_l1649_164929

theorem circle_center_coordinates (h k r : ℝ) :
  (∀ x y : ℝ, (x - 2)^2 + (y + 3)^2 = 1 → (x - h)^2 + (y - k)^2 = r^2) →
  (h, k) = (2, -3) :=
by
  intro H
  sorry

end NUMINAMATH_GPT_circle_center_coordinates_l1649_164929


namespace NUMINAMATH_GPT_hotel_room_assignment_even_hotel_room_assignment_odd_l1649_164949

def smallest_n_even (k : ℕ) (m : ℕ) (h1 : k = 2 * m) : ℕ :=
  100 * (m + 1)

def smallest_n_odd (k : ℕ) (m : ℕ) (h1 : k = 2 * m + 1) : ℕ :=
  100 * (m + 1) + 1

theorem hotel_room_assignment_even (k m : ℕ) (h1 : k = 2 * m) :
  ∃ n, n = smallest_n_even k m h1 ∧ n >= 100 :=
  by
  sorry

theorem hotel_room_assignment_odd (k m : ℕ) (h1 : k = 2 * m + 1) :
  ∃ n, n = smallest_n_odd k m h1 ∧ n >= 100 :=
  by
  sorry

end NUMINAMATH_GPT_hotel_room_assignment_even_hotel_room_assignment_odd_l1649_164949


namespace NUMINAMATH_GPT_factor_quadratic_l1649_164932

theorem factor_quadratic : ∀ (x : ℝ), 4 * x^2 - 20 * x + 25 = (2 * x - 5)^2 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_factor_quadratic_l1649_164932


namespace NUMINAMATH_GPT_range_of_m_l1649_164978

-- Define the conditions
theorem range_of_m (m : ℝ) : 
    (∀ x : ℝ, (m-1) * x^2 + 2 * x + 1 = 0 → 
     (m-1 ≠ 0) ∧ 
     (4 - 4 * (m - 1) > 0)) ↔ 
    (m < 2 ∧ m ≠ 1) :=
sorry

end NUMINAMATH_GPT_range_of_m_l1649_164978


namespace NUMINAMATH_GPT_linear_eq_conditions_l1649_164903

theorem linear_eq_conditions (m : ℤ) (h : abs m = 1) (h₂ : m + 1 ≠ 0) : m = 1 :=
by
  sorry

end NUMINAMATH_GPT_linear_eq_conditions_l1649_164903


namespace NUMINAMATH_GPT_total_height_increase_in_4_centuries_l1649_164984

def height_increase_per_decade : ℕ := 75
def years_per_century : ℕ := 100
def years_per_decade : ℕ := 10
def centuries : ℕ := 4

theorem total_height_increase_in_4_centuries :
  height_increase_per_decade * (centuries * years_per_century / years_per_decade) = 3000 := by
  sorry

end NUMINAMATH_GPT_total_height_increase_in_4_centuries_l1649_164984


namespace NUMINAMATH_GPT_sequence_first_term_l1649_164953

theorem sequence_first_term (a : ℕ → ℤ) 
  (h1 : a 3 = 5) 
  (h2 : ∀ n : ℕ, 0 < n → a (n + 1) = 2 * a n - 1) : 
  a 1 = 2 := 
sorry

end NUMINAMATH_GPT_sequence_first_term_l1649_164953


namespace NUMINAMATH_GPT_eleven_pow_2048_mod_17_l1649_164904

theorem eleven_pow_2048_mod_17 : 11^2048 % 17 = 1 := by
  sorry

end NUMINAMATH_GPT_eleven_pow_2048_mod_17_l1649_164904


namespace NUMINAMATH_GPT_lorry_empty_weight_l1649_164961

-- Define variables for the weights involved
variable (lw : ℕ)  -- weight of the lorry when empty
variable (bl : ℕ)  -- number of bags of apples
variable (bw : ℕ)  -- weight of each bag of apples
variable (total_weight : ℕ)  -- total loaded weight of the lorry

-- Given conditions
axiom lorry_loaded_weight : bl = 20 ∧ bw = 60 ∧ total_weight = 1700

-- The theorem we want to prove
theorem lorry_empty_weight : (∀ lw bw, total_weight - bl * bw = lw) → lw = 500 :=
by
  intro h
  rw [←h lw bw]
  sorry

end NUMINAMATH_GPT_lorry_empty_weight_l1649_164961


namespace NUMINAMATH_GPT_rectangular_prism_width_l1649_164992

theorem rectangular_prism_width 
  (l : ℝ) (h : ℝ) (d : ℝ) (w : ℝ)
  (hl : l = 5) (hh : h = 7) (hd : d = 14) :
  d = Real.sqrt (l^2 + w^2 + h^2) → w = Real.sqrt 122 :=
by 
  sorry

end NUMINAMATH_GPT_rectangular_prism_width_l1649_164992


namespace NUMINAMATH_GPT_find_m_l1649_164923

theorem find_m (a b c m x : ℂ) :
  ( (2 * m + 1) * (x^2 - (b + 1) * x) = (2 * m - 3) * (2 * a * x - c) )
  →
  (x = (b + 1)) 
  →
  m = 1.5 := by
  sorry

end NUMINAMATH_GPT_find_m_l1649_164923


namespace NUMINAMATH_GPT_intersection_A_B_l1649_164967

def A := {x : ℝ | 2 * x - 1 ≤ 0}
def B := {x : ℝ | 1 / x > 1}

theorem intersection_A_B : A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 1 / 2} :=
  sorry

end NUMINAMATH_GPT_intersection_A_B_l1649_164967


namespace NUMINAMATH_GPT_cos_double_angle_l1649_164907

theorem cos_double_angle (α : ℝ) (h : Real.tan α = 3) : Real.cos (2 * α) = -4 / 5 := 
  sorry

end NUMINAMATH_GPT_cos_double_angle_l1649_164907


namespace NUMINAMATH_GPT_part1_min_value_part2_max_value_k_lt_part2_max_value_k_geq_l1649_164987

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem part1_min_value : ∀ (x : ℝ), x > 0 → f x ≥ -1 / Real.exp 1 := 
by sorry

noncomputable def g (x k : ℝ) : ℝ := f x - k * (x - 1)

theorem part2_max_value_k_lt : ∀ (k : ℝ), k < Real.exp 1 / (Real.exp 1 - 1) → 
  ∀ (x : ℝ), 1 ≤ x ∧ x ≤ Real.exp 1 → g x k ≤ Real.exp 1 - k * Real.exp 1 + k :=
by sorry

theorem part2_max_value_k_geq : ∀ (k : ℝ), k ≥ Real.exp 1 / (Real.exp 1 - 1) → 
  ∀ (x : ℝ), 1 ≤ x ∧ x ≤ Real.exp 1 → g x k ≤ 0 :=
by sorry

end NUMINAMATH_GPT_part1_min_value_part2_max_value_k_lt_part2_max_value_k_geq_l1649_164987


namespace NUMINAMATH_GPT_correct_decimal_multiplication_l1649_164920

theorem correct_decimal_multiplication : 0.085 * 3.45 = 0.29325 := 
by 
  sorry

end NUMINAMATH_GPT_correct_decimal_multiplication_l1649_164920


namespace NUMINAMATH_GPT_total_vegetables_l1649_164950

-- Definitions for the conditions in the problem
def cucumbers := 58
def carrots := cucumbers - 24
def tomatoes := cucumbers + 49
def radishes := carrots

-- Statement for the proof problem
theorem total_vegetables :
  cucumbers + carrots + tomatoes + radishes = 233 :=
by sorry

end NUMINAMATH_GPT_total_vegetables_l1649_164950


namespace NUMINAMATH_GPT_find_function_l1649_164966

theorem find_function (f : ℝ → ℝ) (c : ℝ) :
  (∀ x y : ℝ, (x - y) * f (x + y) - (x + y) * f (x - y) = 4 * x * y * (x ^ 2 - y ^ 2)) →
  (∀ x : ℝ, f x = x ^ 3 + c * x) :=
by
  -- The proof details will be filled here.
  sorry

end NUMINAMATH_GPT_find_function_l1649_164966


namespace NUMINAMATH_GPT_travel_time_seattle_to_lasvegas_l1649_164913

def distance_seattle_boise : ℝ := 640
def distance_boise_saltlakecity : ℝ := 400
def distance_saltlakecity_phoenix : ℝ := 750
def distance_phoenix_lasvegas : ℝ := 300

def speed_highway_seattle_boise : ℝ := 80
def speed_city_seattle_boise : ℝ := 35

def speed_highway_boise_saltlakecity : ℝ := 65
def speed_city_boise_saltlakecity : ℝ := 25

def speed_highway_saltlakecity_denver : ℝ := 75
def speed_city_saltlakecity_denver : ℝ := 30

def speed_highway_denver_phoenix : ℝ := 70
def speed_city_denver_phoenix : ℝ := 20

def speed_highway_phoenix_lasvegas : ℝ := 50
def speed_city_phoenix_lasvegas : ℝ := 30

def city_distance_estimate : ℝ := 10

noncomputable def total_time : ℝ :=
  let time_seattle_boise := ((distance_seattle_boise - city_distance_estimate) / speed_highway_seattle_boise) + (city_distance_estimate / speed_city_seattle_boise)
  let time_boise_saltlakecity := ((distance_boise_saltlakecity - city_distance_estimate) / speed_highway_boise_saltlakecity) + (city_distance_estimate / speed_city_boise_saltlakecity)
  let time_saltlakecity_phoenix := ((distance_saltlakecity_phoenix - city_distance_estimate) / speed_highway_saltlakecity_denver) + (city_distance_estimate / speed_city_saltlakecity_denver)
  let time_phoenix_lasvegas := ((distance_phoenix_lasvegas - city_distance_estimate) / speed_highway_phoenix_lasvegas) + (city_distance_estimate / speed_city_phoenix_lasvegas)
  time_seattle_boise + time_boise_saltlakecity + time_saltlakecity_phoenix + time_phoenix_lasvegas

theorem travel_time_seattle_to_lasvegas :
  total_time = 30.89 :=
sorry

end NUMINAMATH_GPT_travel_time_seattle_to_lasvegas_l1649_164913


namespace NUMINAMATH_GPT_percentage_discount_is_12_l1649_164933

noncomputable def cost_price : ℝ := 47.50
noncomputable def list_price : ℝ := 67.47
noncomputable def desired_selling_price : ℝ := cost_price + 0.25 * cost_price
noncomputable def actual_selling_price : ℝ := 59.375

theorem percentage_discount_is_12 :
  ∃ D : ℝ, desired_selling_price = list_price - (list_price * D) ∧ D = 0.12 := 
by 
  sorry

end NUMINAMATH_GPT_percentage_discount_is_12_l1649_164933


namespace NUMINAMATH_GPT_roots_of_polynomial_l1649_164902

noncomputable def polynomial (m z : ℝ) : ℝ :=
  z^3 - (m^2 - m + 7) * z - (3 * m^2 - 3 * m - 6)

theorem roots_of_polynomial (m z : ℝ) (h : polynomial m (-1) = 0) :
  (m = 3 ∧ z = 4 ∨ z = -3) ∨ (m = -2 ∧ sorry) :=
sorry

end NUMINAMATH_GPT_roots_of_polynomial_l1649_164902


namespace NUMINAMATH_GPT_max_vouchers_with_680_l1649_164901

def spend_to_voucher (spent : ℕ) : ℕ := (spent / 100) * 20

theorem max_vouchers_with_680 : spend_to_voucher 680 = 160 := by
  sorry

end NUMINAMATH_GPT_max_vouchers_with_680_l1649_164901


namespace NUMINAMATH_GPT_students_brought_apples_l1649_164962

theorem students_brought_apples (A B C D : ℕ) (h1 : B = 8) (h2 : C = 10) (h3 : D = 5) (h4 : A - D + B - D = C) : A = 12 :=
by {
  sorry
}

end NUMINAMATH_GPT_students_brought_apples_l1649_164962


namespace NUMINAMATH_GPT_seq_a10_eq_90_l1649_164938

noncomputable def seq (a : ℕ → ℕ) : Prop :=
  a 1 = 0 ∧ ∀ n, a (n + 1) = a n + 2 * n

theorem seq_a10_eq_90 {a : ℕ → ℕ} (h : seq a) : a 10 = 90 :=
  sorry

end NUMINAMATH_GPT_seq_a10_eq_90_l1649_164938


namespace NUMINAMATH_GPT_next_month_eggs_l1649_164968

-- Given conditions definitions
def eggs_left_last_month : ℕ := 27
def eggs_after_buying : ℕ := 58
def eggs_eaten_this_month : ℕ := 48

-- Calculate number of eggs mother buys each month
def eggs_bought_each_month : ℕ := eggs_after_buying - eggs_left_last_month

-- Remaining eggs before next purchase
def eggs_left_before_next_purchase : ℕ := eggs_after_buying - eggs_eaten_this_month

-- Final amount of eggs after mother buys next month's supply
def total_eggs_next_month : ℕ := eggs_left_before_next_purchase + eggs_bought_each_month

-- Prove the total number of eggs next month equals 41
theorem next_month_eggs : total_eggs_next_month = 41 := by
  sorry

end NUMINAMATH_GPT_next_month_eggs_l1649_164968


namespace NUMINAMATH_GPT_maria_total_baggies_l1649_164917

def choc_chip_cookies := 33
def oatmeal_cookies := 2
def cookies_per_bag := 5

def total_cookies := choc_chip_cookies + oatmeal_cookies

def total_baggies (total_cookies : Nat) (cookies_per_bag : Nat) : Nat :=
  total_cookies / cookies_per_bag

theorem maria_total_baggies : total_baggies total_cookies cookies_per_bag = 7 :=
  by
    -- Steps proving the equivalence can be done here
    sorry

end NUMINAMATH_GPT_maria_total_baggies_l1649_164917


namespace NUMINAMATH_GPT_direction_vector_of_line_m_l1649_164915

noncomputable def projectionMatrix : Matrix (Fin 3) (Fin 3) ℚ :=
  ![
    ![ 5 / 21, -2 / 21, -2 / 7 ],
    ![ -2 / 21, 1 / 42, 1 / 14 ],
    ![ -2 / 7,  1 / 14, 4 / 7 ]
  ]

noncomputable def vectorI : Fin 3 → ℚ
  | 0 => 1
  | _ => 0

noncomputable def projectedVector : Fin 3 → ℚ :=
  fun i => (projectionMatrix.mulVec vectorI) i

theorem direction_vector_of_line_m :
  (projectedVector 0 = 5 / 21) ∧ 
  (projectedVector 1 = -2 / 21) ∧
  (projectedVector 2 = -6 / 21) ∧
  Nat.gcd (Nat.gcd 5 2) 6 = 1 :=
by
  sorry

end NUMINAMATH_GPT_direction_vector_of_line_m_l1649_164915


namespace NUMINAMATH_GPT_largest_four_digit_number_divisible_by_33_l1649_164912

theorem largest_four_digit_number_divisible_by_33 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (33 ∣ n) ∧ ∀ m : ℕ, (1000 ≤ m ∧ m < 10000 ∧ 33 ∣ m → m ≤ 9999) :=
by
  sorry

end NUMINAMATH_GPT_largest_four_digit_number_divisible_by_33_l1649_164912


namespace NUMINAMATH_GPT_sum_of_consecutive_integers_l1649_164946

theorem sum_of_consecutive_integers (a b c : ℤ) (h1 : a + 1 = b) (h2 : b + 1 = c) (h3 : c = 7) : a + b + c = 18 := by
  sorry

end NUMINAMATH_GPT_sum_of_consecutive_integers_l1649_164946


namespace NUMINAMATH_GPT_susie_rooms_l1649_164900

theorem susie_rooms
  (house_vacuum_time_hours : ℕ)
  (room_vacuum_time_minutes : ℕ)
  (total_vacuum_time_minutes : ℕ)
  (total_vacuum_time_computed : house_vacuum_time_hours * 60 = total_vacuum_time_minutes)
  (rooms_count : ℕ)
  (rooms_count_computed : total_vacuum_time_minutes / room_vacuum_time_minutes = rooms_count) :
  house_vacuum_time_hours = 2 →
  room_vacuum_time_minutes = 20 →
  rooms_count = 6 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_susie_rooms_l1649_164900
