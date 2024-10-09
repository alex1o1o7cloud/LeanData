import Mathlib

namespace parallel_condition_l2189_218930

-- Define the vectors a and b
def vector_a (x : ℝ) : ℝ × ℝ := (1, x)
def vector_b (x : ℝ) : ℝ × ℝ := (x^2, 4 * x)

-- Define the condition for parallelism for two-dimensional vectors
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

-- Define the theorem to prove
theorem parallel_condition (x : ℝ) :
  parallel (vector_a x) (vector_b x) ↔ |x| = 2 :=
by {
  sorry
}

end parallel_condition_l2189_218930


namespace solve_for_x_l2189_218963

theorem solve_for_x (x : ℝ) (h : 5 + 7 / x = 6 - 5 / x) : x = 12 := 
sorry

end solve_for_x_l2189_218963


namespace savings_of_person_l2189_218910

-- Definitions as given in the problem
def income := 18000
def ratio_income_expenditure := 5 / 4

-- Implied definitions based on the conditions and problem context
noncomputable def expenditure := income * (4/5)
noncomputable def savings := income - expenditure

-- Theorem statement
theorem savings_of_person : savings = 3600 :=
by
  -- Placeholder for proof
  sorry

end savings_of_person_l2189_218910


namespace factor_expression_l2189_218993

variable (a : ℝ)

theorem factor_expression : 45 * a^2 + 135 * a + 90 = 45 * a * (a + 5) :=
by
  sorry

end factor_expression_l2189_218993


namespace equivalent_single_discount_l2189_218914

noncomputable def original_price : ℝ := 50
noncomputable def first_discount : ℝ := 0.30
noncomputable def second_discount : ℝ := 0.15
noncomputable def third_discount : ℝ := 0.10

theorem equivalent_single_discount :
  let discount_1 := original_price * (1 - first_discount)
  let discount_2 := discount_1 * (1 - second_discount)
  let discount_3 := discount_2 * (1 - third_discount)
  let final_price := discount_3
  (1 - (final_price / original_price)) = 0.4645 :=
by
  let discount_1 := original_price * (1 - first_discount)
  let discount_2 := discount_1 * (1 - second_discount)
  let discount_3 := discount_2 * (1 - third_discount)
  let final_price := discount_3
  sorry

end equivalent_single_discount_l2189_218914


namespace equal_chords_divide_equally_l2189_218920

theorem equal_chords_divide_equally 
  {A B C D M : ℝ} 
  (in_circle : ∃ (O : ℝ), (dist O A = dist O B) ∧ (dist O C = dist O D) ∧ (dist O M < dist O A))
  (chords_equal : dist A B = dist C D)
  (intersection_M : dist A M + dist M B = dist C M + dist M D ∧ dist A M = dist C M ∧ dist B M = dist D M) :
  dist A M = dist M B ∧ dist C M = dist M D := 
sorry

end equal_chords_divide_equally_l2189_218920


namespace christine_commission_rate_l2189_218905

theorem christine_commission_rate (C : ℝ) (H1 : 24000 ≠ 0) (H2 : 0.4 * (C / 100 * 24000) = 1152) :
  C = 12 :=
by
  sorry

end christine_commission_rate_l2189_218905


namespace abs_sum_fraction_le_sum_abs_fraction_l2189_218954

variable (a b : ℝ)

theorem abs_sum_fraction_le_sum_abs_fraction (a b : ℝ) :
  (|a + b| / (1 + |a + b|)) ≤ (|a| / (1 + |a|)) + (|b| / (1 + |b|)) :=
sorry

end abs_sum_fraction_le_sum_abs_fraction_l2189_218954


namespace odd_function_value_sum_l2189_218929

theorem odd_function_value_sum
  (f : ℝ → ℝ)
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_fneg1 : f (-1) = 2) :
  f 0 + f 1 = -2 := by
  sorry

end odd_function_value_sum_l2189_218929


namespace units_digit_n_l2189_218934

theorem units_digit_n (m n : ℕ) (hm : m % 10 = 9) (h : m * n = 18^5) : n % 10 = 2 :=
sorry

end units_digit_n_l2189_218934


namespace solve_system_of_equations_l2189_218926

theorem solve_system_of_equations (x y : ℝ) (h1 : x + y = 7) (h2 : 2 * x - y = 2) :
  x = 3 ∧ y = 4 :=
by
  sorry

end solve_system_of_equations_l2189_218926


namespace sixth_grader_count_l2189_218966

theorem sixth_grader_count : 
  ∃ x y : ℕ, (3 / 7) * x = (1 / 3) * y ∧ x + y = 140 ∧ x = 61 :=
by {
  sorry  -- Proof not required
}

end sixth_grader_count_l2189_218966


namespace four_x_sq_plus_nine_y_sq_l2189_218972

theorem four_x_sq_plus_nine_y_sq (x y : ℝ) 
  (h1 : 2 * x + 3 * y = 9)
  (h2 : x * y = -12) : 
  4 * x^2 + 9 * y^2 = 225 := 
by
  sorry

end four_x_sq_plus_nine_y_sq_l2189_218972


namespace part_a_part_b_l2189_218982

theorem part_a (a : Fin 10 → ℤ) : ∃ i j : Fin 10, i ≠ j ∧ 27 ∣ (a i)^3 - (a j)^3 := sorry
theorem part_b (b : Fin 8 → ℤ) : ∃ i j : Fin 8, i ≠ j ∧ 27 ∣ (b i)^3 - (b j)^3 := sorry

end part_a_part_b_l2189_218982


namespace xyz_square_sum_l2189_218939

theorem xyz_square_sum {x y z a b c d : ℝ} (h1 : x * y = a) (h2 : x * z = b) (h3 : y * z = c) (h4 : x + y + z = d) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0):
  x^2 + y^2 + z^2 = d^2 - 2 * (a + b + c) :=
sorry

end xyz_square_sum_l2189_218939


namespace quadratic_inequality_solution_l2189_218997

theorem quadratic_inequality_solution :
  { x : ℝ | x^2 + 3 * x - 4 < 0 } = { x : ℝ | -4 < x ∧ x < 1 } :=
by
  sorry

end quadratic_inequality_solution_l2189_218997


namespace time_difference_180_div_vc_l2189_218908

open Real

theorem time_difference_180_div_vc
  (V_A V_B V_C : ℝ)
  (h_ratio : V_A / V_C = 5 ∧ V_B / V_C = 4)
  (start_A start_B start_C : ℝ)
  (h_start_A : start_A = 100)
  (h_start_B : start_B = 80)
  (h_start_C : start_C = 0)
  (race_distance : ℝ)
  (h_race_distance : race_distance = 1200) :
  (race_distance - start_A) / V_A - race_distance / V_C = 180 / V_C := 
sorry

end time_difference_180_div_vc_l2189_218908


namespace wicket_keeper_age_l2189_218937

/-- The cricket team consists of 11 members with an average age of 22 years.
    One member is 25 years old, and the wicket keeper is W years old.
    Excluding the 25-year-old and the wicket keeper, the average age of the remaining players is 21 years.
    Prove that the wicket keeper is 6 years older than the average age of the team. -/
theorem wicket_keeper_age (W : ℕ) (team_avg_age : ℕ := 22) (total_team_members : ℕ := 11) 
                          (other_member_age : ℕ := 25) (remaining_avg_age : ℕ := 21) :
    W = 28 → W - team_avg_age = 6 :=
by
  intros
  sorry

end wicket_keeper_age_l2189_218937


namespace distance_ratio_l2189_218999

variable (d_RB d_BC : ℝ)

theorem distance_ratio
    (h1 : d_RB / 60 + d_BC / 20 ≠ 0)
    (h2 : 36 * (d_RB / 60 + d_BC / 20) = d_RB + d_BC) : 
    d_RB / d_BC = 2 := 
sorry

end distance_ratio_l2189_218999


namespace sum_of_first_11_terms_is_minus_66_l2189_218995

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d 

def sum_of_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (n * (a n + a 1)) / 2

theorem sum_of_first_11_terms_is_minus_66 
  (a : ℕ → ℤ) 
  (h_seq : arithmetic_sequence a)
  (h_roots : ∃ a2 a10, (a2 = a 2 ∧ a10 = a 10) ∧ (a2 + a10 = -12) ∧ (a2 * a10 = -8)) 
  : sum_of_first_n_terms a 11 = -66 :=
by
  sorry

end sum_of_first_11_terms_is_minus_66_l2189_218995


namespace percent_same_grades_l2189_218974

theorem percent_same_grades 
    (total_students same_A same_B same_C same_D same_E : ℕ)
    (h_total_students : total_students = 40)
    (h_same_A : same_A = 3)
    (h_same_B : same_B = 5)
    (h_same_C : same_C = 6)
    (h_same_D : same_D = 2)
    (h_same_E : same_E = 1):
    ((same_A + same_B + same_C + same_D + same_E : ℚ) / total_students * 100) = 42.5 :=
by
  sorry

end percent_same_grades_l2189_218974


namespace cost_of_two_books_and_one_magazine_l2189_218901

-- Definitions of the conditions
def condition1 (x y : ℝ) : Prop := 3 * x + 2 * y = 18.40
def condition2 (x y : ℝ) : Prop := 2 * x + 3 * y = 17.60

-- Proof problem
theorem cost_of_two_books_and_one_magazine (x y : ℝ) 
  (h1 : condition1 x y) 
  (h2 : condition2 x y) : 
  2 * x + y = 11.20 :=
sorry

end cost_of_two_books_and_one_magazine_l2189_218901


namespace original_number_is_80_l2189_218916

variable (e : ℝ)

def increased_value := 1.125 * e
def decreased_value := 0.75 * e
def difference_condition := increased_value e - decreased_value e = 30

theorem original_number_is_80 (h : difference_condition e) : e = 80 :=
sorry

end original_number_is_80_l2189_218916


namespace probability_donation_to_A_l2189_218912

-- Define population proportions
def prob_O : ℝ := 0.50
def prob_A : ℝ := 0.15
def prob_B : ℝ := 0.30
def prob_AB : ℝ := 0.05

-- Define blood type compatibility predicate
def can_donate_to_A (blood_type : ℝ) : Prop := 
  blood_type = prob_O ∨ blood_type = prob_A

-- Theorem statement
theorem probability_donation_to_A : 
  prob_O + prob_A = 0.65 :=
by
  -- proof skipped
  sorry

end probability_donation_to_A_l2189_218912


namespace sequence_m_l2189_218900

noncomputable def a (n : ℕ) : ℕ :=
  if n = 0 then 0  -- We usually start sequences from n = 1; hence, a_0 is irrelevant
  else (n * n) - n + 1

theorem sequence_m (m : ℕ) (h_positive : m > 0) (h_bound : 43 < a m ∧ a m < 73) : m = 8 :=
by {
  sorry
}

end sequence_m_l2189_218900


namespace find_y_l2189_218940

-- Define the points and slope conditions
def point_R : ℝ × ℝ := (-3, 4)
def x2 : ℝ := 5

-- Define the y coordinate and its corresponding condition
def y_condition (y : ℝ) : Prop := (y - 4) / (5 - (-3)) = 1 / 2

-- The main theorem stating the conditions and conclusion
theorem find_y (y : ℝ) (h : y_condition y) : y = 8 :=
by
  sorry

end find_y_l2189_218940


namespace initial_tomatoes_l2189_218960

def t_picked : ℕ := 83
def t_left : ℕ := 14
def t_total : ℕ := t_picked + t_left

theorem initial_tomatoes : t_total = 97 := by
  rw [t_total]
  rfl

end initial_tomatoes_l2189_218960


namespace college_application_distributions_l2189_218975

theorem college_application_distributions : 
  let total_students := 6
  let colleges := 3
  ∃ n : ℕ, n = 540 ∧ 
    (n = (colleges^total_students - colleges * (2^total_students) + 
      (colleges.choose 2) * 1)) := sorry

end college_application_distributions_l2189_218975


namespace unique_suwy_product_l2189_218907

def letter_value (c : Char) : Nat :=
  if 'A' ≤ c ∧ c ≤ 'Z' then Char.toNat c - Char.toNat 'A' + 1 else 0

def product_of_chars (l : List Char) : Nat :=
  l.foldr (λ c acc => letter_value c * acc) 1

theorem unique_suwy_product :
  ∀ (l : List Char), l.length = 4 → product_of_chars l = 19 * 21 * 23 * 25 → l = ['S', 'U', 'W', 'Y'] := 
by
  intro l hlen hproduct
  sorry

end unique_suwy_product_l2189_218907


namespace paint_floor_cost_l2189_218973

theorem paint_floor_cost :
  ∀ (L : ℝ) (rate : ℝ)
  (condition1 : L = 3 * (L / 3))
  (condition2 : L = 19.595917942265423)
  (condition3 : rate = 5),
  rate * (L * (L / 3)) = 640 :=
by
  intros L rate condition1 condition2 condition3
  sorry

end paint_floor_cost_l2189_218973


namespace geometric_sequence_sum_l2189_218911

theorem geometric_sequence_sum (a1 r : ℝ) (S : ℕ → ℝ) :
  S 2 = 3 → S 4 = 15 →
  (∀ n, S n = a1 * (1 - r^n) / (1 - r)) → S 6 = 63 :=
by
  intros hS2 hS4 hSn
  sorry

end geometric_sequence_sum_l2189_218911


namespace sarah_correct_answer_percentage_l2189_218947

theorem sarah_correct_answer_percentage
  (q1 q2 q3 : ℕ)   -- Number of questions in the first, second, and third tests.
  (p1 p2 p3 : ℕ → ℝ)   -- Percentages of questions Sarah got right in the first, second, and third tests.
  (m : ℕ)   -- Number of calculation mistakes:
  (h_q1 : q1 = 30) (h_q2 : q2 = 20) (h_q3 : q3 = 50)
  (h_p1 : p1 q1 = 0.85) (h_p2 : p2 q2 = 0.75) (h_p3 : p3 q3 = 0.90)
  (h_m : m = 3) :
  ∃ pct_correct : ℝ, pct_correct = 83 :=
by
  sorry

end sarah_correct_answer_percentage_l2189_218947


namespace zero_points_in_intervals_l2189_218979

noncomputable def f (x : ℝ) : ℝ :=
  (1 / 3) * x - Real.log x

theorem zero_points_in_intervals :
  (∀ x : ℝ, x ∈ Set.Ioo (1 / Real.exp 1) 1 → f x ≠ 0) ∧
  (∃ x : ℝ, x ∈ Set.Ioo 1 (Real.exp 1) ∧ f x = 0) :=
by
  sorry

end zero_points_in_intervals_l2189_218979


namespace probability_both_truth_l2189_218970

variable (P_A : ℝ) (P_B : ℝ)

theorem probability_both_truth (hA : P_A = 0.55) (hB : P_B = 0.60) :
  P_A * P_B = 0.33 :=
by
  sorry

end probability_both_truth_l2189_218970


namespace mrs_lee_earnings_percentage_l2189_218902

noncomputable def percentage_earnings_june (T : ℝ) : ℝ :=
  let L := 0.5 * T
  let L_June := 1.2 * L
  let total_income_june := T
  (L_June / total_income_june) * 100

theorem mrs_lee_earnings_percentage (T : ℝ) (hT : T ≠ 0) : percentage_earnings_june T = 60 :=
by
  sorry

end mrs_lee_earnings_percentage_l2189_218902


namespace carwash_num_cars_l2189_218935

variable (C : ℕ)

theorem carwash_num_cars 
    (h1 : 5 * 7 + 5 * 6 + C * 5 = 100)
    : C = 7 := 
by
    sorry

end carwash_num_cars_l2189_218935


namespace fraction_equality_l2189_218985
-- Import the necessary library

-- The proof statement
theorem fraction_equality : (16 + 8) / (4 - 2) = 12 := 
by {
  -- Inserting 'sorry' to indicate that the proof is omitted
  sorry
}

end fraction_equality_l2189_218985


namespace age_of_participant_who_left_l2189_218919

theorem age_of_participant_who_left
  (avg_age_first_room : ℕ)
  (num_people_first_room : ℕ)
  (avg_age_second_room : ℕ)
  (num_people_second_room : ℕ)
  (increase_in_avg_age : ℕ)
  (total_num_people : ℕ)
  (final_avg_age : ℕ)
  (initial_avg_age : ℕ)
  (sum_ages : ℕ)
  (person_left : ℕ) :
  avg_age_first_room = 20 ∧ 
  num_people_first_room = 8 ∧
  avg_age_second_room = 45 ∧
  num_people_second_room = 12 ∧
  increase_in_avg_age = 1 ∧
  total_num_people = num_people_first_room + num_people_second_room ∧
  final_avg_age = initial_avg_age + increase_in_avg_age ∧
  initial_avg_age = (sum_ages) / total_num_people ∧
  sum_ages = (avg_age_first_room * num_people_first_room + avg_age_second_room * num_people_second_room) ∧
  19 * final_avg_age = sum_ages - person_left
  → person_left = 16 :=
by sorry

end age_of_participant_who_left_l2189_218919


namespace food_bank_remaining_after_four_weeks_l2189_218978

def week1_donated : ℝ := 40
def week1_given_out : ℝ := 0.6 * week1_donated
def week1_remaining : ℝ := week1_donated - week1_given_out

def week2_donated : ℝ := 1.5 * week1_donated
def week2_given_out : ℝ := 0.7 * week2_donated
def week2_remaining : ℝ := week2_donated - week2_given_out
def total_remaining_after_week2 : ℝ := week1_remaining + week2_remaining

def week3_donated : ℝ := 1.25 * week2_donated
def week3_given_out : ℝ := 0.8 * week3_donated
def week3_remaining : ℝ := week3_donated - week3_given_out
def total_remaining_after_week3 : ℝ := total_remaining_after_week2 + week3_remaining

def week4_donated : ℝ := 0.9 * week3_donated
def week4_given_out : ℝ := 0.5 * week4_donated
def week4_remaining : ℝ := week4_donated - week4_given_out
def total_remaining_after_week4 : ℝ := total_remaining_after_week3 + week4_remaining

theorem food_bank_remaining_after_four_weeks : total_remaining_after_week4 = 82.75 := by
  sorry

end food_bank_remaining_after_four_weeks_l2189_218978


namespace distance_from_origin_l2189_218977

theorem distance_from_origin :
  ∃ (m : ℝ), m = Real.sqrt (108 + 8 * Real.sqrt 10) ∧
              (∃ (x y : ℝ), y = 8 ∧ 
                            (x - 2)^2 + (y - 5)^2 = 49 ∧ 
                            x = 2 + 2 * Real.sqrt 10 ∧ 
                            m = Real.sqrt ((x^2) + (y^2))) :=
by
  sorry

end distance_from_origin_l2189_218977


namespace mod_37_5_l2189_218951

theorem mod_37_5 : 37 % 5 = 2 :=
by
  sorry

end mod_37_5_l2189_218951


namespace cos_squared_diff_tan_l2189_218921

theorem cos_squared_diff_tan (α : ℝ) (h : Real.tan α = 3) :
  Real.cos (α + π/4) ^ 2 - Real.cos (α - π/4) ^ 2 = -3 / 5 :=
by
  sorry

end cos_squared_diff_tan_l2189_218921


namespace max_balls_in_cube_l2189_218958

theorem max_balls_in_cube 
  (radius : ℝ) (side_length : ℝ) 
  (ball_volume : ℝ := (4 / 3) * Real.pi * (radius^3)) 
  (cube_volume : ℝ := side_length^3) 
  (max_balls : ℝ := cube_volume / ball_volume) :
  radius = 3 ∧ side_length = 8 → Int.floor max_balls = 4 := 
by
  intro h
  rw [h.left, h.right]
  -- further proof would use numerical evaluation
  sorry

end max_balls_in_cube_l2189_218958


namespace fraction_cost_of_raisins_l2189_218998

variable (cost_raisins cost_nuts total_cost_raisins total_cost_nuts total_cost : ℝ)

theorem fraction_cost_of_raisins (h1 : cost_nuts = 3 * cost_raisins)
                                 (h2 : total_cost_raisins = 4 * cost_raisins)
                                 (h3 : total_cost_nuts = 4 * cost_nuts)
                                 (h4 : total_cost = total_cost_raisins + total_cost_nuts) :
                                 (total_cost_raisins / total_cost) = (1 / 4) :=
by
  sorry

end fraction_cost_of_raisins_l2189_218998


namespace max_even_integers_with_odd_product_l2189_218917

theorem max_even_integers_with_odd_product (a b c d e f : ℕ) (h_pos: 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f) (h_odd_product : (a * b * c * d * e * f) % 2 = 1) : 
  (a % 2 = 1) ∧ (b % 2 = 1) ∧ (c % 2 = 1) ∧ (d % 2 = 1) ∧ (e % 2 = 1) ∧ (f % 2 = 1) := 
sorry

end max_even_integers_with_odd_product_l2189_218917


namespace joan_paid_230_l2189_218956

theorem joan_paid_230 (J K : ℝ) (h1 : J + K = 600) (h2 : 2 * J = K + 90) : J = 230 := 
by 
  sorry

end joan_paid_230_l2189_218956


namespace geom_seq_min_value_l2189_218903

open Real

/-- 
Theorem: For a geometric sequence {a_n} where a_n > 0 and a_7 = √2/2, 
the minimum value of 1/a_3 + 2/a_11 is 4.
-/
theorem geom_seq_min_value (a : ℕ → ℝ) (a_pos : ∀ n, 0 < a n) (h7 : a 7 = (sqrt 2) / 2) :
  (1 / (a 3) + 2 / (a 11) >= 4) :=
sorry

end geom_seq_min_value_l2189_218903


namespace average_number_of_fish_is_75_l2189_218925

-- Define the number of fish in Boast Pool and conditions for other bodies of water
def Boast_Pool_fish : ℕ := 75
def Onum_Lake_fish : ℕ := Boast_Pool_fish + 25
def Riddle_Pond_fish : ℕ := Onum_Lake_fish / 2

-- Define the average number of fish in all three bodies of water
def average_fish : ℕ := (Onum_Lake_fish + Boast_Pool_fish + Riddle_Pond_fish) / 3

-- Prove that the average number of fish in all three bodies of water is 75
theorem average_number_of_fish_is_75 : average_fish = 75 := by
  sorry

end average_number_of_fish_is_75_l2189_218925


namespace least_number_divisible_by_11_l2189_218904

theorem least_number_divisible_by_11 (n : ℕ) (k : ℕ) (h₁ : n = 2520 * k + 1) (h₂ : 11 ∣ n) : n = 12601 :=
sorry

end least_number_divisible_by_11_l2189_218904


namespace child_ticket_price_correct_l2189_218948

-- Definitions based on conditions
def total_collected := 104
def price_adult := 6
def total_tickets := 21
def children_tickets := 11

-- Derived conditions
def adult_tickets := total_tickets - children_tickets
def total_revenue_child (C : ℕ) := children_tickets * C
def total_revenue_adult := adult_tickets * price_adult

-- Main statement to prove
theorem child_ticket_price_correct (C : ℕ) 
  (h1 : total_revenue_child C + total_revenue_adult = total_collected) : 
  C = 4 :=
by
  sorry

end child_ticket_price_correct_l2189_218948


namespace product_fraction_l2189_218996

open Int

def first_six_composites : List ℕ := [4, 6, 8, 9, 10, 12]
def first_three_primes : List ℕ := [2, 3, 5]
def next_three_composites : List ℕ := [14, 15, 16]

def product (l : List ℕ) : ℕ := l.foldl (· * ·) 1

theorem product_fraction :
  (product first_six_composites : ℚ) / (product (first_three_primes ++ next_three_composites) : ℚ) = 24 / 7 :=
by 
  sorry

end product_fraction_l2189_218996


namespace original_price_of_trouser_l2189_218945

theorem original_price_of_trouser (sale_price : ℝ) (discount_rate : ℝ) (original_price : ℝ) 
  (h1 : sale_price = 50) (h2 : discount_rate = 0.50) (h3 : sale_price = (1 - discount_rate) * original_price) : 
  original_price = 100 :=
sorry

end original_price_of_trouser_l2189_218945


namespace parallel_vectors_imply_x_value_l2189_218933

theorem parallel_vectors_imply_x_value (x : ℝ) : 
    let a := (1, 2)
    let b := (-1, x)
    (1 / -1:ℝ) = (2 / x) → x = -2 := 
by
  intro h
  sorry

end parallel_vectors_imply_x_value_l2189_218933


namespace carter_drum_sticks_l2189_218967

def sets_per_show (used : ℕ) (tossed : ℕ) : ℕ := used + tossed

def total_sets (sets_per_show : ℕ) (num_shows : ℕ) : ℕ := sets_per_show * num_shows

theorem carter_drum_sticks :
  sets_per_show 8 10 * 45 = 810 :=
by
  sorry

end carter_drum_sticks_l2189_218967


namespace parallel_vectors_l2189_218955

noncomputable def vector_a : (ℤ × ℤ) := (1, 3)
noncomputable def vector_b (m : ℤ) : (ℤ × ℤ) := (-2, m)

theorem parallel_vectors (m : ℤ) (h : vector_a = (1, 3) ∧ vector_b m = (-2, m))
  (hp: ∃ k : ℤ, ∀ (a1 a2 b1 b2 : ℤ), (a1, a2) = vector_a ∧ (b1, b2) = (1 + k * (-2), 3 + k * m)):
  m = -6 :=
by
  sorry

end parallel_vectors_l2189_218955


namespace simplify_and_evaluate_l2189_218943

theorem simplify_and_evaluate (x : ℝ) (h : x = Real.sqrt 2 - 1) : 
  ((x + 3) * (x - 3)) - (x * (x - 2)) = 2 * Real.sqrt 2 - 11 := by
  sorry

end simplify_and_evaluate_l2189_218943


namespace extremum_is_not_unique_l2189_218906

-- Define the extremum conditionally in terms of unique extremum within an interval for a function
def isExtremum {α : Type*} [Preorder α] (f : α → ℝ) (x : α) :=
  ∀ y, f y ≤ f x ∨ f x ≤ f y

theorem extremum_is_not_unique (α : Type*) [Preorder α] (f : α → ℝ) :
  ¬ ∀ x, isExtremum f x → (∀ y, isExtremum f y → x = y) :=
by
  sorry

end extremum_is_not_unique_l2189_218906


namespace find_y_l2189_218984

theorem find_y (x y : ℕ) (h_pos_y : 0 < y) (h_rem : x % y = 7) (h_div : x = 86 * y + (1 / 10) * y) :
  y = 70 :=
sorry

end find_y_l2189_218984


namespace total_books_on_shelves_l2189_218949

theorem total_books_on_shelves (shelves books_per_shelf : ℕ) (h_shelves : shelves = 350) (h_books_per_shelf : books_per_shelf = 25) :
  shelves * books_per_shelf = 8750 :=
by {
  sorry
}

end total_books_on_shelves_l2189_218949


namespace exactly_one_correct_proposition_l2189_218932

variables (l1 l2 : Line) (alpha : Plane)

-- Definitions for the conditions
def perpendicular_lines (l1 l2 : Line) : Prop := -- definition of perpendicular lines
sorry

def perpendicular_to_plane (l : Line) (alpha : Plane) : Prop := -- definition of line perpendicular to plane
sorry

def line_in_plane (l : Line) (alpha : Plane) : Prop := -- definition of line in a plane
sorry

-- Problem statement
theorem exactly_one_correct_proposition 
  (h1 : perpendicular_lines l1 l2) 
  (h2 : perpendicular_to_plane l1 alpha) 
  (h3 : line_in_plane l2 alpha) : 
  (¬(perpendicular_lines l1 l2 ∧ perpendicular_to_plane l1 alpha → line_in_plane l2 alpha) ∧
   ¬(perpendicular_lines l1 l2 ∧ line_in_plane l2 alpha → perpendicular_to_plane l1 alpha) ∧
   (perpendicular_to_plane l1 alpha ∧ line_in_plane l2 alpha → perpendicular_lines l1 l2)) :=
sorry

end exactly_one_correct_proposition_l2189_218932


namespace worker_original_daily_wage_l2189_218942

-- Given Conditions
def increases : List ℝ := [0.20, 0.30, 0.40, 0.50, 0.60]
def new_total_weekly_salary : ℝ := 1457

-- Define the sum of the weekly increases
def total_increase : ℝ := (1 + increases.get! 0) + (1 + increases.get! 1) + (1 + increases.get! 2) + (1 + increases.get! 3) + (1 + increases.get! 4)

-- Main Theorem
theorem worker_original_daily_wage : ∀ (W : ℝ), total_increase * W = new_total_weekly_salary → W = 242.83 :=
by
  intro W h
  sorry

end worker_original_daily_wage_l2189_218942


namespace sakshi_days_l2189_218962

theorem sakshi_days (Sakshi_efficiency Tanya_efficiency : ℝ) (Sakshi_days Tanya_days : ℝ) (h_efficiency : Tanya_efficiency = 1.25 * Sakshi_efficiency) (h_days : Tanya_days = 8) : Sakshi_days = 10 :=
by
  sorry

end sakshi_days_l2189_218962


namespace dodecagon_area_l2189_218994

theorem dodecagon_area (s : ℝ) (n : ℕ) (angles : ℕ → ℝ)
  (h_s : s = 10) (h_n : n = 12) 
  (h_angles : ∀ i, angles i = if i % 3 == 2 then 270 else 90) :
  ∃ area : ℝ, area = 500 := 
sorry

end dodecagon_area_l2189_218994


namespace probability_of_two_red_balls_l2189_218980

-- Define the total number of balls, number of red balls, and number of white balls
def total_balls := 6
def red_balls := 4
def white_balls := 2
def drawn_balls := 2

-- Define the combination formula
def choose (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The number of ways to choose 2 red balls from 4
def ways_to_choose_red := choose 4 2

-- The number of ways to choose any 2 balls from the total of 6
def ways_to_choose_any := choose 6 2

-- The corresponding probability
def probability := ways_to_choose_red / ways_to_choose_any

-- The theorem we want to prove
theorem probability_of_two_red_balls :
  probability = 2 / 5 :=
by
  sorry

end probability_of_two_red_balls_l2189_218980


namespace vaishali_total_stripes_l2189_218988

def total_stripes (hats_with_3_stripes hats_with_4_stripes hats_with_no_stripes : ℕ) 
  (hats_with_5_stripes hats_with_7_stripes hats_with_1_stripe : ℕ) 
  (hats_with_10_stripes hats_with_2_stripes : ℕ)
  (stripes_per_hat_with_3 stripes_per_hat_with_4 stripes_per_hat_with_no : ℕ)
  (stripes_per_hat_with_5 stripes_per_hat_with_7 stripes_per_hat_with_1 : ℕ)
  (stripes_per_hat_with_10 stripes_per_hat_with_2 : ℕ) : ℕ :=
  hats_with_3_stripes * stripes_per_hat_with_3 +
  hats_with_4_stripes * stripes_per_hat_with_4 +
  hats_with_no_stripes * stripes_per_hat_with_no +
  hats_with_5_stripes * stripes_per_hat_with_5 +
  hats_with_7_stripes * stripes_per_hat_with_7 +
  hats_with_1_stripe * stripes_per_hat_with_1 +
  hats_with_10_stripes * stripes_per_hat_with_10 +
  hats_with_2_stripes * stripes_per_hat_with_2

#eval total_stripes 4 3 6 2 1 4 2 3 3 4 0 5 7 1 10 2 -- 71

theorem vaishali_total_stripes : (total_stripes 4 3 6 2 1 4 2 3 3 4 0 5 7 1 10 2) = 71 :=
by
  sorry

end vaishali_total_stripes_l2189_218988


namespace remainder_14_div_5_l2189_218965

theorem remainder_14_div_5 : 14 % 5 = 4 := by
  sorry

end remainder_14_div_5_l2189_218965


namespace last_week_profit_min_selling_price_red_beauty_l2189_218936

theorem last_week_profit (x kgs_of_red_beauty x_green : ℕ) 
  (purchase_cost_red_beauty_per_kg selling_cost_red_beauty_per_kg 
  purchase_cost_xiangshan_green_per_kg selling_cost_xiangshan_green_per_kg
  total_weight total_cost all_fruits_profit : ℕ) :
  purchase_cost_red_beauty_per_kg = 20 ->
  selling_cost_red_beauty_per_kg = 35 ->
  purchase_cost_xiangshan_green_per_kg = 5 ->
  selling_cost_xiangshan_green_per_kg = 10 ->
  total_weight = 300 ->
  total_cost = 3000 ->
  x * purchase_cost_red_beauty_per_kg + (total_weight - x) * purchase_cost_xiangshan_green_per_kg = total_cost ->
  all_fruits_profit = x * (selling_cost_red_beauty_per_kg - purchase_cost_red_beauty_per_kg) +
  (total_weight - x) * (selling_cost_xiangshan_green_per_kg - purchase_cost_xiangshan_green_per_kg) -> 
  all_fruits_profit = 2500 := sorry

theorem min_selling_price_red_beauty (last_week_profit : ℕ) (x kgs_of_red_beauty x_green damaged_ratio : ℝ) 
  (purchase_cost_red_beauty_per_kg profit_last_week selling_cost_xiangshan_per_kg 
  total_weight total_cost : ℝ) :
  purchase_cost_red_beauty_per_kg = 20 ->
  profit_last_week = 2500 ->
  damaged_ratio = 0.1 ->
  x = 100 ->
  (profit_last_week = 
    x * (35 - purchase_cost_red_beauty_per_kg) + (total_weight - x) * (10 - 5)) ->
  90 * (purchase_cost_red_beauty_per_kg + (last_week_profit - 15 * (total_weight - x) / 90)) ≥ 1500 ->
  profit_last_week / (90 * (90 * (purchase_cost_red_beauty_per_kg + (2500 - 15 * (300 - x) / 90)))) >=
  (36.7 - 20 / purchase_cost_red_beauty_per_kg) :=
  sorry

end last_week_profit_min_selling_price_red_beauty_l2189_218936


namespace volume_of_wedge_l2189_218931

theorem volume_of_wedge (r : ℝ) (V : ℝ) (sphere_wedges : ℝ) 
  (h_circumference : 2 * Real.pi * r = 18 * Real.pi)
  (h_volume : V = (4 / 3) * Real.pi * r ^ 3) 
  (h_sphere_wedges : sphere_wedges = 6) : 
  V / sphere_wedges = 162 * Real.pi :=
by
  sorry

end volume_of_wedge_l2189_218931


namespace absolute_value_inequality_l2189_218946

variable (a b c d : ℝ)

theorem absolute_value_inequality (h₁ : a + b + c + d > 0) (h₂ : a > c) (h₃ : b > d) : 
  |a + b| > |c + d| := sorry

end absolute_value_inequality_l2189_218946


namespace cross_section_area_correct_l2189_218913

noncomputable def cross_section_area (a : ℝ) : ℝ :=
  (3 * a^2 * Real.sqrt 33) / 8

theorem cross_section_area_correct
  (AB CC1 : ℝ)
  (h1 : AB = a)
  (h2 : CC1 = 2 * a) :
  cross_section_area a = (3 * a^2 * Real.sqrt 33) / 8 :=
by
  sorry

end cross_section_area_correct_l2189_218913


namespace ayse_guarantee_win_l2189_218950

def can_ayse_win (m n k : ℕ) : Prop :=
  -- Function defining the winning strategy for Ayşe
  sorry -- The exact strategy definition would be here

theorem ayse_guarantee_win :
  ((can_ayse_win 1 2012 2014) ∧ 
   (can_ayse_win 2011 2011 2012) ∧ 
   (can_ayse_win 2011 2012 2013) ∧ 
   (can_ayse_win 2011 2012 2014) ∧ 
   (can_ayse_win 2011 2013 2013)) = true :=
sorry -- Proof goes here

end ayse_guarantee_win_l2189_218950


namespace length_of_box_l2189_218957

theorem length_of_box (rate : ℕ) (width : ℕ) (depth : ℕ) (time : ℕ) (volume : ℕ) (length : ℕ) :
  rate = 4 →
  width = 6 →
  depth = 2 →
  time = 21 →
  volume = rate * time →
  length = volume / (width * depth) →
  length = 7 :=
by
  intros
  sorry

end length_of_box_l2189_218957


namespace solve_for_xy_l2189_218968

theorem solve_for_xy (x y : ℝ) (h1 : 3 * x ^ 2 - 9 * y ^ 2 = 0) (h2 : x + y = 5) :
    (x = (15 - 5 * Real.sqrt 3) / 2 ∧ y = (5 * Real.sqrt 3 - 5) / 2) ∨
    (x = (-15 - 5 * Real.sqrt 3) / 2 ∧ y = (5 + 5 * Real.sqrt 3) / 2) :=
by
  sorry

end solve_for_xy_l2189_218968


namespace minimum_y_value_y_at_4_eq_6_l2189_218964

noncomputable def y (x : ℝ) : ℝ := x + 4 / (x - 2)

theorem minimum_y_value (x : ℝ) (h : x > 2) : y x ≥ 6 :=
sorry

theorem y_at_4_eq_6 : y 4 = 6 :=
sorry

end minimum_y_value_y_at_4_eq_6_l2189_218964


namespace triangle_angle_A_eq_60_l2189_218969

theorem triangle_angle_A_eq_60 (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : A + B + C = π)
  (h_tan : (Real.tan A) / (Real.tan B) = (2 * c - b) / b) : 
  A = π / 3 :=
by
  sorry

end triangle_angle_A_eq_60_l2189_218969


namespace red_bowling_balls_count_l2189_218986

theorem red_bowling_balls_count (G R : ℕ) (h1 : G = R + 6) (h2 : R + G = 66) : R = 30 :=
by
  sorry

end red_bowling_balls_count_l2189_218986


namespace golf_money_l2189_218959

-- Definitions based on conditions
def cost_per_round : ℤ := 80
def number_of_rounds : ℤ := 5

-- The theorem/problem statement
theorem golf_money : cost_per_round * number_of_rounds = 400 := 
by {
  -- Proof steps would go here, but to skip the proof, we use sorry
  sorry
}

end golf_money_l2189_218959


namespace inequality_proof_l2189_218927

variable (a b c : ℝ)
variable (h1 : 0 < a)
variable (h2 : 0 < b)
variable (h3 : 0 < c)

theorem inequality_proof :
  (2 * a + b + c)^2 / (2 * a^2 + (b + c)^2) +
  (a + 2 * b + c)^2 / (2 * b^2 + (c + a)^2) +
  (a + b + 2 * c)^2 / (2 * c^2 + (a + b)^2) ≤ 8 := sorry

end inequality_proof_l2189_218927


namespace slope_of_line_l2189_218961

theorem slope_of_line {x y : ℝ} (hx : x ≠ 0) (hy : y ≠ 0) (h : 5 / x + 4 / y = 0) :
  ∃ x₁ x₂ y₁ y₂, (5 / x₁ + 4 / y₁ = 0) ∧ (5 / x₂ + 4 / y₂ = 0) ∧ 
  (y₂ - y₁) / (x₂ - x₁) = -4 / 5 :=
sorry

end slope_of_line_l2189_218961


namespace quilt_cost_proof_l2189_218971

-- Definitions for conditions
def length := 7
def width := 8
def cost_per_sq_foot := 40

-- Definition for the calculation of the area
def area := length * width

-- Definition for the calculation of the cost
def total_cost := area * cost_per_sq_foot

-- Theorem stating the final proof
theorem quilt_cost_proof : total_cost = 2240 := by
  sorry

end quilt_cost_proof_l2189_218971


namespace trapezoid_two_heights_l2189_218990

-- Define trivially what a trapezoid is, in terms of having two parallel sides.
structure Trapezoid :=
(base1 base2 : ℝ)
(height1 height2 : ℝ)
(has_two_heights : height1 = height2)

theorem trapezoid_two_heights (T : Trapezoid) : ∃ h1 h2 : ℝ, h1 = h2 :=
by
  use T.height1
  use T.height2
  exact T.has_two_heights

end trapezoid_two_heights_l2189_218990


namespace cheerleader_total_l2189_218989

theorem cheerleader_total 
  (size2 : ℕ)
  (size6 : ℕ)
  (size12 : ℕ)
  (h1 : size2 = 4)
  (h2 : size6 = 10)
  (h3 : size12 = size6 / 2) :
  size2 + size6 + size12 = 19 :=
by
  sorry

end cheerleader_total_l2189_218989


namespace largest_class_students_l2189_218991

theorem largest_class_students (x : ℕ)
  (h1 : x + (x - 2) + (x - 4) + (x - 6) + (x - 8) = 95) :
  x = 23 :=
by
  sorry

end largest_class_students_l2189_218991


namespace caochong_weighing_equation_l2189_218941

-- Definitions for porter weight, stone weight, and the counts in the respective steps
def porter_weight : ℝ := 120
def stone_weight (x : ℝ) : ℝ := x
def first_step_weight (x : ℝ) : ℝ := 20 * stone_weight x + 3 * porter_weight
def second_step_weight (x : ℝ) : ℝ := (20 + 1) * stone_weight x + 1 * porter_weight

-- Theorem stating the equality condition ensuring the same water level
theorem caochong_weighing_equation (x : ℝ) :
  first_step_weight x = second_step_weight x :=
by
  sorry

end caochong_weighing_equation_l2189_218941


namespace P_positive_l2189_218915

variable (P : ℕ → ℝ)

axiom P_cond_0 : P 0 > 0
axiom P_cond_1 : P 1 > P 0
axiom P_cond_2 : P 2 > 2 * P 1 - P 0
axiom P_cond_3 : P 3 > 3 * P 2 - 3 * P 1 + P 0
axiom P_cond_n : ∀ n, P (n + 4) > 4 * P (n + 3) - 6 * P (n + 2) + 4 * P (n + 1) - P n

theorem P_positive (n : ℕ) (h : n > 0) : P n > 0 := by
  sorry

end P_positive_l2189_218915


namespace subproblem1_l2189_218981

theorem subproblem1 (a : ℝ) : a^3 * a + (2 * a^2)^2 = 5 * a^4 := 
by sorry

end subproblem1_l2189_218981


namespace tensor_A_B_eq_l2189_218909

-- Define sets A and B
def A : Set ℕ := {0, 2}
def B : Set ℕ := {x | x^2 - 3 * x + 2 = 0}

-- Define set operation ⊗
def tensor (A B : Set ℕ) : Set ℕ := {z | ∃ x y, x ∈ A ∧ y ∈ B ∧ z = x * y}

-- Prove that A ⊗ B = {0, 2, 4}
theorem tensor_A_B_eq : tensor A B = {0, 2, 4} :=
by
  sorry

end tensor_A_B_eq_l2189_218909


namespace investment_doubles_in_9_years_l2189_218923

noncomputable def years_to_double (initial_amount : ℕ) (interest_rate : ℕ) : ℕ :=
  72 / interest_rate

theorem investment_doubles_in_9_years :
  ∀ (initial_amount : ℕ) (interest_rate : ℕ) (investment_period_val : ℕ) (expected_value : ℕ),
  initial_amount = 8000 ∧ interest_rate = 8 ∧ investment_period_val = 18 ∧ expected_value = 32000 →
  years_to_double initial_amount interest_rate = 9 :=
by
  intros initial_amount interest_rate investment_period_val expected_value h
  sorry

end investment_doubles_in_9_years_l2189_218923


namespace solve_system_eq_l2189_218953

theorem solve_system_eq (x y z : ℤ) :
  (x^2 - 23 * y + 66 * z + 612 = 0) ∧ 
  (y^2 + 62 * x - 20 * z + 296 = 0) ∧ 
  (z^2 - 22 * x + 67 * y + 505 = 0) →
  (x = -20) ∧ (y = -22) ∧ (z = -23) :=
by {
  sorry
}

end solve_system_eq_l2189_218953


namespace volume_of_rectangular_solid_l2189_218944

theorem volume_of_rectangular_solid : 
  let l := 100 -- length in cm
  let w := 20  -- width in cm
  let h := 50  -- height in cm
  let V := l * w * h
  V = 100000 :=
by
  rfl

end volume_of_rectangular_solid_l2189_218944


namespace number_of_bags_needed_l2189_218928

def cost_corn_seeds : ℕ := 50
def cost_fertilizers_pesticides : ℕ := 35
def cost_labor : ℕ := 15
def profit_percentage : ℝ := 0.10
def price_per_bag : ℝ := 11

theorem number_of_bags_needed (total_cost : ℕ) (total_revenue : ℝ) (num_bags : ℝ) :
  total_cost = cost_corn_seeds + cost_fertilizers_pesticides + cost_labor →
  total_revenue = ↑total_cost + (↑total_cost * profit_percentage) →
  num_bags = total_revenue / price_per_bag →
  num_bags = 10 := 
by
  sorry

end number_of_bags_needed_l2189_218928


namespace harmonic_mean_closest_integer_l2189_218983

theorem harmonic_mean_closest_integer (a b : ℝ) (ha : a = 1) (hb : b = 2016) :
  abs ((2 * a * b) / (a + b) - 2) < 1 :=
by
  sorry

end harmonic_mean_closest_integer_l2189_218983


namespace rope_segments_after_folding_l2189_218924

theorem rope_segments_after_folding (n : ℕ) (h : n = 6) : 2^n + 1 = 65 :=
by
  rw [h]
  norm_num

end rope_segments_after_folding_l2189_218924


namespace mimi_spent_on_clothes_l2189_218952

theorem mimi_spent_on_clothes :
  let total_spent := 8000
  let adidas_cost := 600
  let skechers_cost := 5 * adidas_cost
  let nike_cost := 3 * adidas_cost
  let total_sneakers_cost := adidas_cost + skechers_cost + nike_cost
  total_spent - total_sneakers_cost = 2600 :=
by
  let total_spent := 8000
  let adidas_cost := 600
  let skechers_cost := 5 * adidas_cost
  let nike_cost := 3 * adidas_cost
  let total_sneakers_cost := adidas_cost + skechers_cost + nike_cost
  show total_spent - total_sneakers_cost = 2600
  sorry

end mimi_spent_on_clothes_l2189_218952


namespace PropA_neither_sufficient_nor_necessary_for_PropB_l2189_218938

variable (a b : ℤ)

-- Proposition A
def PropA : Prop := a + b ≠ 4

-- Proposition B
def PropB : Prop := a ≠ 1 ∧ b ≠ 3

-- The required statement
theorem PropA_neither_sufficient_nor_necessary_for_PropB : ¬(PropA a b → PropB a b) ∧ ¬(PropB a b → PropA a b) :=
by
  sorry

end PropA_neither_sufficient_nor_necessary_for_PropB_l2189_218938


namespace compound_interest_amount_l2189_218987

/-
Given:
- Principal amount P = 5000
- Annual interest rate r = 0.07
- Time period t = 15 years

We aim to prove:
A = 5000 * (1 + 0.07) ^ 15 = 13795.15
-/
theorem compound_interest_amount :
  let P : ℝ := 5000
  let r : ℝ := 0.07
  let t : ℝ := 15
  let A : ℝ := P * (1 + r) ^ t
  A = 13795.15 :=
by
  sorry

end compound_interest_amount_l2189_218987


namespace reusable_bag_trips_correct_lowest_carbon_solution_l2189_218918

open Real

-- Conditions definitions
def canvas_CO2 := 600 -- in pounds
def polyester_CO2 := 250 -- in pounds
def recycled_plastic_CO2 := 150 -- in pounds
def CO2_per_plastic_bag := 4 / 16 -- 4 ounces per bag, converted to pounds
def bags_per_trip := 8

-- Total CO2 per trip using plastic bags
def CO2_per_trip := CO2_per_plastic_bag * bags_per_trip

-- Proof of correct number of trips
theorem reusable_bag_trips_correct :
  canvas_CO2 / CO2_per_trip = 300 ∧
  polyester_CO2 / CO2_per_trip = 125 ∧
  recycled_plastic_CO2 / CO2_per_trip = 75 :=
by
  -- Here we would provide proofs for each part,
  -- ensuring we are fulfilling the conditions provided
  -- Skipping the proof with sorry
  sorry

-- Proof that recycled plastic bag is the lowest-carbon solution
theorem lowest_carbon_solution :
  min (canvas_CO2 / CO2_per_trip) (min (polyester_CO2 / CO2_per_trip) (recycled_plastic_CO2 / CO2_per_trip)) = recycled_plastic_CO2 / CO2_per_trip :=
by
  -- Here we would provide proofs for each part,
  -- ensuring we are fulfilling the conditions provided
  -- Skipping the proof with sorry
  sorry

end reusable_bag_trips_correct_lowest_carbon_solution_l2189_218918


namespace carls_garden_area_is_correct_l2189_218976

-- Define the conditions
def isRectangle (length width : ℕ) : Prop :=
∃ l w, l * w = length * width

def validFencePosts (shortSidePosts longSidePosts totalPosts : ℕ) : Prop :=
∃ x, totalPosts = 2 * x + 2 * (2 * x) - 4 ∧ x = shortSidePosts

def validSpacing (shortSideSpaces longSideSpaces : ℕ) : Prop :=
shortSideSpaces = 4 * (shortSideSpaces - 1) ∧ longSideSpaces = 4 * (longSideSpaces - 1)

def correctArea (shortSide longSide expectedArea : ℕ) : Prop :=
shortSide * longSide = expectedArea

-- Prove the conditions lead to the expected area
theorem carls_garden_area_is_correct :
  ∃ shortSide longSide,
  isRectangle shortSide longSide ∧
  validFencePosts 5 10 24 ∧
  validSpacing 5 10 ∧
  correctArea (4 * (5-1)) (4 * (10-1)) 576 :=
by
  sorry

end carls_garden_area_is_correct_l2189_218976


namespace Chris_had_before_birthday_l2189_218922

-- Define the given amounts
def grandmother_money : ℕ := 25
def aunt_uncle_money : ℕ := 20
def parents_money : ℕ := 75
def total_money_now : ℕ := 279

-- Define the total birthday money received
def birthday_money : ℕ := grandmother_money + aunt_uncle_money + parents_money

-- Define the amount of money Chris had before his birthday
def money_before_birthday (total_now birthday_money : ℕ) : ℕ := total_now - birthday_money

-- Proposition to prove
theorem Chris_had_before_birthday : money_before_birthday total_money_now birthday_money = 159 := by
  sorry

end Chris_had_before_birthday_l2189_218922


namespace player5_points_combination_l2189_218992

theorem player5_points_combination :
  ∃ (two_point_shots three_pointers free_throws : ℕ), 
  (two_point_shots * 2 + three_pointers * 3 + free_throws * 1 = 14) :=
sorry

end player5_points_combination_l2189_218992
