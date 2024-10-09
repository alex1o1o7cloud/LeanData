import Mathlib

namespace find_n_satisfying_conditions_l368_36851

noncomputable def exists_set_satisfying_conditions (n : ℕ) : Prop :=
  ∃ S : Finset ℕ, S.card = n ∧
  (∀ x ∈ S, x < 2^(n-1)) ∧
  ∀ A B : Finset ℕ, A ⊆ S → B ⊆ S → A ≠ B → A ≠ ∅ → B ≠ ∅ → A.sum id ≠ B.sum id

theorem find_n_satisfying_conditions : ∀ n : ℕ, (n ≥ 4) ↔ exists_set_satisfying_conditions n :=
sorry

end find_n_satisfying_conditions_l368_36851


namespace common_root_sum_k_l368_36869

theorem common_root_sum_k :
  (∃ x : ℝ, (x^2 - 4 * x + 3 = 0) ∧ (x^2 - 6 * x + k = 0)) → 
  (∃ (k₁ k₂ : ℝ), (k₁ = 5) ∧ (k₂ = 9) ∧ (k₁ + k₂ = 14)) :=
by
  sorry

end common_root_sum_k_l368_36869


namespace quadratic_real_roots_condition_l368_36878

theorem quadratic_real_roots_condition (m : ℝ) :
  (∃ x : ℝ, x^2 + x + m = 0) → m ≤ 1/4 :=
by
  sorry

end quadratic_real_roots_condition_l368_36878


namespace problem_solution_l368_36864

noncomputable def f (x : ℝ) := 2 * Real.sin x + x^3 + 1

theorem problem_solution (a : ℝ) (h : f a = 3) : f (-a) = -1 := by
  sorry

end problem_solution_l368_36864


namespace find_divisor_l368_36853

theorem find_divisor (x y : ℕ) (h1 : (x - 5) / 7 = 7) (h2 : (x - 14) / y = 4) : y = 10 :=
sorry

end find_divisor_l368_36853


namespace initial_discount_l368_36879

theorem initial_discount (total_amount price_after_initial_discount additional_disc_percent : ℝ)
  (H1 : total_amount = 1000)
  (H2 : price_after_initial_discount = total_amount - 280)
  (H3 : additional_disc_percent = 0.20) :
  let additional_discount := additional_disc_percent * price_after_initial_discount
  let price_after_additional_discount := price_after_initial_discount - additional_discount
  let total_discount := total_amount - price_after_additional_discount
  let initial_discount := total_discount - additional_discount
  initial_discount = 280 := by
  sorry

end initial_discount_l368_36879


namespace convince_jury_l368_36865

-- Define predicates for being a criminal, normal man, guilty, or a knight
def Criminal : Prop := sorry
def NormalMan : Prop := sorry
def Guilty : Prop := sorry
def Knight : Prop := sorry

-- Define your status
variable (you : Prop)

-- Assumptions as per given conditions
axiom criminal_not_normal_man : Criminal → ¬NormalMan
axiom you_not_guilty : ¬Guilty
axiom you_not_knight : ¬Knight

-- The statement to prove
theorem convince_jury : ¬Guilty ∧ ¬Knight := by
  exact And.intro you_not_guilty you_not_knight

end convince_jury_l368_36865


namespace martha_gingers_amount_l368_36887

theorem martha_gingers_amount (G : ℚ) (h : G = 0.43 * (G + 3)) : G = 2 := by
  sorry

end martha_gingers_amount_l368_36887


namespace percentage_boys_not_attended_college_l368_36880

/-
Define the constants and given conditions.
-/
def number_of_boys : ℕ := 300
def number_of_girls : ℕ := 240
def total_students : ℕ := number_of_boys + number_of_girls
def percentage_class_attended_college : ℝ := 0.70
def percentage_girls_not_attended_college : ℝ := 0.30

/-
The proof problem statement: 
Prove the percentage of the boys class that did not attend college.
-/
theorem percentage_boys_not_attended_college :
  let students_attended_college := percentage_class_attended_college * total_students
  let not_attended_college_students := total_students - students_attended_college
  let not_attended_college_girls := percentage_girls_not_attended_college * number_of_girls
  let not_attended_college_boys := not_attended_college_students - not_attended_college_girls
  let percentage_boys_not_attended_college := (not_attended_college_boys / number_of_boys) * 100
  percentage_boys_not_attended_college = 30 := by
  sorry

end percentage_boys_not_attended_college_l368_36880


namespace person_c_completion_time_l368_36875

def job_completion_days (Ra Rb Rc : ℚ) (total_earnings b_earnings : ℚ) : ℚ :=
  Rc

theorem person_c_completion_time (Ra Rb Rc : ℚ)
  (hRa : Ra = 1 / 6)
  (hRb : Rb = 1 / 8)
  (total_earnings : ℚ)
  (b_earnings : ℚ)
  (earnings_ratio : b_earnings / total_earnings = Rb / (Ra + Rb + Rc))
  : Rc = 1 / 12 :=
sorry

end person_c_completion_time_l368_36875


namespace area_ratio_triangle_MNO_XYZ_l368_36871

noncomputable def triangle_area_ratio (XY YZ XZ p q r : ℝ) : ℝ := sorry

theorem area_ratio_triangle_MNO_XYZ : 
  ∀ (p q r: ℝ),
  p > 0 → q > 0 → r > 0 →
  p + q + r = 3 / 4 →
  p ^ 2 + q ^ 2 + r ^ 2 = 1 / 2 →
  triangle_area_ratio 12 16 20 p q r = 9 / 32 :=
sorry

end area_ratio_triangle_MNO_XYZ_l368_36871


namespace students_failed_to_get_degree_l368_36838

/-- 
Out of 1,500 senior high school students, 70% passed their English exams,
80% passed their Mathematics exams, and 65% passed their Science exams.
To get their degree, a student must pass in all three subjects.
Assume independence of passing rates. This Lean proof shows that
the number of students who failed to get their degree is 954.
-/
theorem students_failed_to_get_degree :
  let total_students := 1500
  let p_english := 0.70
  let p_math := 0.80
  let p_science := 0.65
  let p_all_pass := p_english * p_math * p_science
  let students_all_pass := p_all_pass * total_students
  total_students - students_all_pass = 954 :=
by
  sorry

end students_failed_to_get_degree_l368_36838


namespace intervals_of_positivity_l368_36826

theorem intervals_of_positivity :
  {x : ℝ | (x + 1) * (x - 1) * (x - 2) > 0} = {x : ℝ | (-1 < x ∧ x < 1) ∨ (2 < x)} :=
by
  sorry

end intervals_of_positivity_l368_36826


namespace trapezium_distance_parallel_sides_l368_36804

theorem trapezium_distance_parallel_sides (a b A : ℝ) (h : ℝ) (h1 : a = 20) (h2 : b = 18) (h3 : A = 380) :
  A = (1 / 2) * (a + b) * h → h = 20 :=
by
  intro h4
  rw [h1, h2, h3] at h4
  sorry

end trapezium_distance_parallel_sides_l368_36804


namespace compute_series_l368_36884

noncomputable def sum_series (c d : ℝ) : ℝ :=
  ∑' n, 1 / ((n-1) * d - (n-2) * c) / (n * d - (n-1) * c)

theorem compute_series (c d : ℝ) (hc_pos : 0 < c) (hd_pos : 0 < d) (hcd : d < c) : 
  sum_series c d = 1 / ((d - c) * c) :=
sorry

end compute_series_l368_36884


namespace can_use_bisection_method_l368_36835

noncomputable def f1 (x : ℝ) : ℝ := x^2
noncomputable def f2 (x : ℝ) : ℝ := x⁻¹
noncomputable def f3 (x : ℝ) : ℝ := abs x
noncomputable def f4 (x : ℝ) : ℝ := x^3

theorem can_use_bisection_method : ∃ (a b : ℝ), a < b ∧ (f4 a) * (f4 b) < 0 := 
sorry

end can_use_bisection_method_l368_36835


namespace completing_the_square_l368_36855

theorem completing_the_square (x : ℝ) :
  x^2 + 4 * x + 1 = 0 ↔ (x + 2)^2 = 3 :=
by
  sorry

end completing_the_square_l368_36855


namespace find_value_of_M_l368_36894

theorem find_value_of_M (a b M : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = M) (h4 : ∀ x y : ℝ, x > 0 → y > 0 → (x + y = M) → x * y ≤ (M^2) / 4) (h5 : ∀ x y : ℝ, x > 0 → y > 0 → (x + y = M) → x * y = 2) :
  M = 2 * Real.sqrt 2 :=
by
  sorry

end find_value_of_M_l368_36894


namespace smallest_internal_angle_l368_36868

theorem smallest_internal_angle (α : ℝ) (β : ℝ) (γ : ℝ)
  (h1 : α = 2 * β) (h2 : α = 3 * γ)
  (h3 : α + β + γ = π) :
  α = π / 6 :=
by
  sorry

end smallest_internal_angle_l368_36868


namespace quadratic_vertex_ordinate_l368_36811

theorem quadratic_vertex_ordinate :
  let a := 2
  let b := -4
  let c := -1
  let vertex_x := -b / (2 * a)
  let vertex_y := a * vertex_x ^ 2 + b * vertex_x + c
  vertex_y = -3 :=
by
  sorry

end quadratic_vertex_ordinate_l368_36811


namespace next_wednesday_l368_36818
open Nat

/-- Prove that the next year after 2010 when April 16 falls on a Wednesday is 2014,
    given the conditions:
    1. 2010 is a non-leap year.
    2. The day advances by 1 day for a non-leap year and 2 days for a leap year.
    3. April 16, 2010 was a Friday. -/
theorem next_wednesday (initial_year : ℕ) (initial_day : String) (target_day : String) : 
  (initial_year = 2010) ∧
  (initial_day = "Friday") ∧ 
  (target_day = "Wednesday") →
  2014 = 2010 + 4 :=
by
  sorry

end next_wednesday_l368_36818


namespace probability_green_l368_36837

def total_marbles : ℕ := 100

def P_white : ℚ := 1 / 4

def P_red_or_blue : ℚ := 0.55

def P_sum : ℚ := 1

theorem probability_green :
  P_sum = P_white + P_red_or_blue + P_green →
  P_green = 0.2 :=
sorry

end probability_green_l368_36837


namespace largest_consecutive_odd_integers_sum_255_l368_36841

theorem largest_consecutive_odd_integers_sum_255 : 
  ∃ (n : ℤ), (n + (n + 2) + (n + 4) + (n + 6) + (n + 8) = 255) ∧ (n + 8 = 55) :=
by
  sorry

end largest_consecutive_odd_integers_sum_255_l368_36841


namespace canadian_olympiad_2008_inequality_l368_36866

variable (a b c : ℝ)
variables (positive_a : 0 < a) (positive_b : 0 < b) (positive_c : 0 < c)
variable (sum_abc : a + b + c = 1)

theorem canadian_olympiad_2008_inequality :
  (ab / ((b + c) * (c + a))) + (bc / ((c + a) * (a + b))) + (ca / ((a + b) * (b + c))) ≥ 3 / 4 :=
sorry

end canadian_olympiad_2008_inequality_l368_36866


namespace remainder_zero_when_x_divided_by_y_l368_36845

theorem remainder_zero_when_x_divided_by_y :
  ∀ (x y : ℝ), 
    0 < x ∧ 0 < y ∧ x / y = 6.12 ∧ y = 49.99999999999996 → 
      x % y = 0 := by
  sorry

end remainder_zero_when_x_divided_by_y_l368_36845


namespace value_of_expression_l368_36822

theorem value_of_expression (x : ℝ) (h : 3 * x + 2 = 11) : 6 * x + 4 = 22 :=
by
  sorry

end value_of_expression_l368_36822


namespace find_a_plus_b_l368_36842

theorem find_a_plus_b (a b : ℝ) : (3 = 1/3 * 1 + a) → (1 = 1/3 * 3 + b) → a + b = 8/3 :=
by
  intros h1 h2
  sorry

end find_a_plus_b_l368_36842


namespace min_val_f_l368_36815

noncomputable def f (x : ℝ) : ℝ :=
  4 / (x - 2) + x

theorem min_val_f (x : ℝ) (h : x > 2) : ∃ y, y = f x ∧ y ≥ 6 :=
by {
  sorry
}

end min_val_f_l368_36815


namespace root_in_interval_l368_36803

noncomputable def f (x : ℝ) := Real.log x + x - 2

theorem root_in_interval : ∃ c ∈ Set.Ioo 1 2, f c = 0 := 
sorry

end root_in_interval_l368_36803


namespace election_result_l368_36806

theorem election_result (total_votes : ℕ) (invalid_vote_percentage valid_vote_percentage : ℚ) 
  (candidate_A_percentage : ℚ) (hv: valid_vote_percentage = 1 - invalid_vote_percentage) 
  (ht: total_votes = 560000) 
  (hi: invalid_vote_percentage = 0.15) 
  (hc: candidate_A_percentage = 0.80) : 
  (candidate_A_percentage * valid_vote_percentage * total_votes = 380800) :=
by 
  sorry

end election_result_l368_36806


namespace crayons_count_l368_36821

def crayons_per_box : ℕ := 8
def number_of_boxes : ℕ := 10
def total_crayons : ℕ := crayons_per_box * number_of_boxes

theorem crayons_count : total_crayons = 80 := by
  sorry

end crayons_count_l368_36821


namespace fewer_trombone_than_trumpet_l368_36820

theorem fewer_trombone_than_trumpet 
  (flute_players : ℕ)
  (trumpet_players : ℕ)
  (trombone_players : ℕ)
  (drummers : ℕ)
  (clarinet_players : ℕ)
  (french_horn_players : ℕ)
  (total_members : ℕ) :
  flute_players = 5 →
  trumpet_players = 3 * flute_players →
  clarinet_players = 2 * flute_players →
  drummers = trombone_players + 11 →
  french_horn_players = trombone_players + 3 →
  total_members = flute_players + clarinet_players + trumpet_players + trombone_players + drummers + french_horn_players →
  total_members = 65 →
  trombone_players = 7 ∧ trumpet_players - trombone_players = 8 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2, h3] at h6
  sorry

end fewer_trombone_than_trumpet_l368_36820


namespace no_real_roots_implies_negative_l368_36847

theorem no_real_roots_implies_negative (m : ℝ) : (¬ ∃ x : ℝ, x^2 = m) → m < 0 :=
sorry

end no_real_roots_implies_negative_l368_36847


namespace weight_of_one_pencil_l368_36861

theorem weight_of_one_pencil (total_weight : ℝ) (num_pencils : ℕ) (H : total_weight = 141.5) (H' : num_pencils = 5) : (total_weight / num_pencils) = 28.3 :=
by sorry

end weight_of_one_pencil_l368_36861


namespace impossible_to_achieve_25_percent_grape_juice_l368_36831

theorem impossible_to_achieve_25_percent_grape_juice (x y : ℝ) 
  (h1 : ∀ a b : ℝ, (8 / (8 + 32) = 2 / 10) → (6 / (6 + 24) = 2 / 10))
  (h2 : (8 * x + 6 * y) / (40 * x + 30 * y) = 1 / 4) : false :=
by
  sorry

end impossible_to_achieve_25_percent_grape_juice_l368_36831


namespace polynomial_real_root_l368_36809

theorem polynomial_real_root (a : ℝ) :
  (∃ x : ℝ, x^5 + a * x^4 - x^3 + a * x^2 + x + 1 = 0) ↔
  (a ∈ (Set.Iic (-1/2)) ∨ a ∈ (Set.Ici (1/2))) :=
by
  sorry

end polynomial_real_root_l368_36809


namespace rectangle_side_ratio_square_l368_36828

noncomputable def ratio_square (a b : ℝ) : ℝ :=
(a / b) ^ 2

theorem rectangle_side_ratio_square (a b : ℝ) (h : (a - b) / (a + b) = 1 / 3) : 
  ratio_square a b = 4 := by
  sorry

end rectangle_side_ratio_square_l368_36828


namespace cone_lateral_area_l368_36843

theorem cone_lateral_area (cos_ASB : ℝ)
  (angle_SA_base : ℝ)
  (triangle_SAB_area : ℝ) :
  cos_ASB = 7 / 8 →
  angle_SA_base = 45 →
  triangle_SAB_area = 5 * Real.sqrt 15 →
  (lateral_area : ℝ) = 40 * Real.sqrt 2 * Real.pi :=
by
  intros h1 h2 h3
  sorry

end cone_lateral_area_l368_36843


namespace softball_players_l368_36813

theorem softball_players (cricket hockey football total : ℕ) (h1 : cricket = 12) (h2 : hockey = 17) (h3 : football = 11) (h4 : total = 50) : 
  total - (cricket + hockey + football) = 10 :=
by
  sorry

end softball_players_l368_36813


namespace larger_fraction_l368_36859

theorem larger_fraction :
  (22222222221 : ℚ) / 22222222223 > (33333333331 : ℚ) / 33333333334 := by sorry

end larger_fraction_l368_36859


namespace tangent_line_at_one_f_gt_one_l368_36876

noncomputable def f (x : ℝ) : ℝ :=
  Real.exp x * Real.log x + (2 * Real.exp (x - 1)) / x

theorem tangent_line_at_one : 
  let y := f 1 + (Real.exp 1) * (x - 1)
  y = Real.exp (1 : ℝ) * (x - 1) + 2 := 
sorry

theorem f_gt_one (x : ℝ) (hx : 0 < x) : f x > 1 := 
sorry

end tangent_line_at_one_f_gt_one_l368_36876


namespace total_number_of_cottages_is_100_l368_36839

noncomputable def total_cottages
    (x : ℕ) (n : ℕ) 
    (h1 : 2 * x = number_of_two_room_cottages)
    (h2 : n * x = number_of_three_room_cottages)
    (h3 : 3 * (n * x) = 2 * x + 25) 
    (h4 : x + 2 * x + n * x ≥ 70) : ℕ :=
x + 2 * x + n * x

theorem total_number_of_cottages_is_100 
    (x n : ℕ)
    (h1 : 2 * x = number_of_two_room_cottages)
    (h2 : n * x = number_of_three_room_cottages)
    (h3 : 3 * (n * x) = 2 * x + 25)
    (h4 : x + 2 * x + n * x ≥ 70)
    (h5 : ∃ m : ℕ, m = (x + 2 * x + n * x)) :
  total_cottages x n h1 h2 h3 h4 = 100 :=
by
  sorry

end total_number_of_cottages_is_100_l368_36839


namespace find_three_digit_number_l368_36854

theorem find_three_digit_number : 
  ∃ x : ℕ, (x >= 100 ∧ x < 1000) ∧ (2 * x = 3 * x - 108) :=
by
  have h : ∀ x : ℕ, 100 ≤ x → x < 1000 → 2 * x = 3 * x - 108 → x = 108 := sorry
  exact ⟨108, by sorry⟩

end find_three_digit_number_l368_36854


namespace sum_of_integer_solutions_l368_36808

theorem sum_of_integer_solutions (n_values : List ℤ) : 
  (∀ n ∈ n_values, ∃ (k : ℤ), 2 * n - 3 = k ∧ k ∣ 18) → (n_values.sum = 11) := 
by
  sorry

end sum_of_integer_solutions_l368_36808


namespace sets_of_laces_needed_l368_36872

-- Define the conditions as constants
def teams := 4
def members_per_team := 10
def pairs_per_member := 2
def skates_per_pair := 2
def sets_of_laces_per_skate := 3

-- Formulate and state the theorem to be proven
theorem sets_of_laces_needed : 
  sets_of_laces_per_skate * (teams * members_per_team * (pairs_per_member * skates_per_pair)) = 480 :=
by sorry

end sets_of_laces_needed_l368_36872


namespace solve_for_x_l368_36852

theorem solve_for_x : (2 / 5 : ℚ) - (1 / 7) = 1 / (35 / 9) :=
by
  sorry

end solve_for_x_l368_36852


namespace Ferris_wheel_ticket_cost_l368_36802

theorem Ferris_wheel_ticket_cost
  (cost_rc : ℕ) (rides_rc : ℕ) (cost_c : ℕ) (rides_c : ℕ) (total_tickets : ℕ) (rides_fw : ℕ)
  (H1 : cost_rc = 4) (H2 : rides_rc = 3) (H3 : cost_c = 4) (H4 : rides_c = 2) (H5 : total_tickets = 21) (H6 : rides_fw = 1) :
  21 - (3 * 4 + 2 * 4) = 1 :=
by
  sorry

end Ferris_wheel_ticket_cost_l368_36802


namespace percentage_of_female_students_l368_36814

theorem percentage_of_female_students {F : ℝ} (h1 : 200 > 0): ((200 * (F / 100)) * 0.5 * 0.5 = 30) → (F = 60) :=
by
  sorry

end percentage_of_female_students_l368_36814


namespace john_taller_than_lena_l368_36846

-- Define the heights of John, Lena, and Rebeca.
variables (J L R : ℕ)

-- Given conditions:
-- 1. John has a height of 152 cm
axiom john_height : J = 152

-- 2. John is 6 cm shorter than Rebeca
axiom john_shorter_rebeca : J = R - 6

-- 3. The height of Lena and Rebeca together is 295 cm
axiom lena_rebeca_together : L + R = 295

-- Prove that John is 15 cm taller than Lena
theorem john_taller_than_lena : (J - L) = 15 := by
  sorry

end john_taller_than_lena_l368_36846


namespace dave_paid_3_more_than_doug_l368_36849

theorem dave_paid_3_more_than_doug :
  let total_slices := 10
  let plain_pizza_cost := 10
  let anchovy_fee := 3
  let total_cost := plain_pizza_cost + anchovy_fee
  let cost_per_slice := total_cost / total_slices
  let slices_with_anchovies := total_slices / 3
  let dave_slices := slices_with_anchovies + 2
  let doug_slices := total_slices - dave_slices
  let doug_pay := doug_slices * plain_pizza_cost / total_slices
  let dave_pay := total_cost - doug_pay
  dave_pay - doug_pay = 3 :=
by
  sorry

end dave_paid_3_more_than_doug_l368_36849


namespace basketball_game_count_l368_36829

noncomputable def total_games_played (teams games_each_opp : ℕ) : ℕ :=
  (teams * (teams - 1) / 2) * games_each_opp

theorem basketball_game_count (n : ℕ) (g : ℕ) (h_n : n = 10) (h_g : g = 4) : total_games_played n g = 180 :=
by
  -- Use 'h_n' and 'h_g' as hypotheses
  rw [h_n, h_g]
  show (10 * 9 / 2) * 4 = 180
  sorry

end basketball_game_count_l368_36829


namespace find_n_l368_36881

def sum_for (x : ℕ) : ℕ :=
  if x > 1 then (List.range (2*x)).sum else 0

theorem find_n (n : ℕ) (h : n * (sum_for 4) = 360) : n = 10 :=
by
  sorry

end find_n_l368_36881


namespace undefined_denominator_values_l368_36860

theorem undefined_denominator_values (a : ℝ) : a = 3 ∨ a = -3 ↔ ∃ b : ℝ, (a - b) * (a + b) = 0 := by
  sorry

end undefined_denominator_values_l368_36860


namespace final_values_comparison_l368_36897

theorem final_values_comparison :
  let AA_initial : ℝ := 100
  let BB_initial : ℝ := 100
  let CC_initial : ℝ := 100
  let AA_year1 := AA_initial * 1.20
  let BB_year1 := BB_initial * 0.75
  let CC_year1 := CC_initial
  let AA_year2 := AA_year1 * 0.80
  let BB_year2 := BB_year1 * 1.25
  let CC_year2 := CC_year1
  AA_year2 = 96 ∧ BB_year2 = 93.75 ∧ CC_year2 = 100 ∧ BB_year2 < AA_year2 ∧ AA_year2 < CC_year2 :=
by {
  -- Definitions from conditions
  let AA_initial : ℝ := 100;
  let BB_initial : ℝ := 100;
  let CC_initial : ℝ := 100;
  let AA_year1 := AA_initial * 1.20;
  let BB_year1 := BB_initial * 0.75;
  let CC_year1 := CC_initial;
  let AA_year2 := AA_year1 * 0.80;
  let BB_year2 := BB_year1 * 1.25;
  let CC_year2 := CC_year1;

  -- Use sorry to skip the actual proof
  sorry
}

end final_values_comparison_l368_36897


namespace greatest_divisor_four_consecutive_l368_36801

open Nat

theorem greatest_divisor_four_consecutive (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_four_consecutive_l368_36801


namespace g_at_10_is_300_l368_36840

-- Define the function g and the given condition about g
def g: ℕ → ℤ := sorry

axiom g_cond (m n: ℕ) (h: m ≥ n): g (m + n) + g (m - n) = 2 * g m + 3 * g n
axiom g_1: g 1 = 3

-- Statement to be proved
theorem g_at_10_is_300 : g 10 = 300 := by
  sorry

end g_at_10_is_300_l368_36840


namespace fraction_zero_implies_x_zero_l368_36812

theorem fraction_zero_implies_x_zero (x : ℝ) (h : x / (2 * x - 1) = 0) : x = 0 := 
by {
  sorry
}

end fraction_zero_implies_x_zero_l368_36812


namespace range_of_m_l368_36817

theorem range_of_m (m : ℝ) : 
  ((m - 1) * x^2 - 4 * x + 1 = 0) → 
  ((20 - 4 * m ≥ 0) ∧ (m ≠ 1)) :=
by
  sorry

end range_of_m_l368_36817


namespace ratio_of_shaded_to_white_l368_36886

theorem ratio_of_shaded_to_white (A : ℝ) : 
  let shaded_area := 5 * A
  let unshaded_area := 3 * A
  shaded_area / unshaded_area = 5 / 3 := by
  sorry

end ratio_of_shaded_to_white_l368_36886


namespace solution_set_f_geq_3_range_of_a_for_f_geq_abs_a_minus_4_l368_36819

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 3| - |x - 2|

-- Proof Problem 1 Statement:
theorem solution_set_f_geq_3 :
  {x : ℝ | f x ≥ 3} = {x : ℝ | x ≥ 1} :=
sorry

-- Proof Problem 2 Statement:
theorem range_of_a_for_f_geq_abs_a_minus_4 (a : ℝ) :
  (∃ x : ℝ, f x ≥ |a - 4|) ↔ -1 ≤ a ∧ a ≤ 9 :=
sorry

end solution_set_f_geq_3_range_of_a_for_f_geq_abs_a_minus_4_l368_36819


namespace sequence_form_l368_36885

theorem sequence_form {a : ℕ → ℚ} (h_eq : ∀ n : ℕ, a n * x ^ 2 - a (n + 1) * x + 1 = 0) 
  (h_roots : ∀ α β : ℚ, 6 * α - 2 * α * β + 6 * β = 3 ) (h_a1 : a 1 = 7 / 6) :
  ∀ n : ℕ, a n = (1 / 2) ^ n + 2 / 3 :=
by
  sorry

end sequence_form_l368_36885


namespace soccer_tournament_games_l368_36823

-- Define the single-elimination tournament problem
def single_elimination_games (teams : ℕ) : ℕ :=
  teams - 1

-- Define the specific problem instance
def teams := 20

-- State the theorem
theorem soccer_tournament_games : single_elimination_games teams = 19 :=
  sorry

end soccer_tournament_games_l368_36823


namespace lineup_count_l368_36800

def total_players : ℕ := 15
def out_players : ℕ := 3  -- Alice, Max, and John
def lineup_size : ℕ := 6

-- Define the binomial coefficient in Lean
def binom (n k : ℕ) : ℕ :=
  if h : n ≥ k then
    Nat.choose n k
  else
    0

theorem lineup_count (total_players out_players lineup_size : ℕ) :
  let remaining_with_alice := total_players - out_players + 1 
  let remaining_without_alice := total_players - out_players + 1 
  let remaining_without_both := total_players - out_players 
  binom remaining_with_alice (lineup_size-1) + binom remaining_without_alice (lineup_size-1) + binom remaining_without_both lineup_size = 3498 :=
by
  sorry

end lineup_count_l368_36800


namespace curves_intersect_on_x_axis_l368_36805

theorem curves_intersect_on_x_axis (t θ a : ℝ) (h : a > 0) :
  (∃ t, (t + 1, 1 - 2 * t).snd = 0) →
  (∃ θ, (a * Real.cos θ, 3 * Real.cos θ).snd = 0) →
  (t + 1 = a * Real.cos θ) →
  a = 3 / 2 :=
by
  intro h1 h2 h3
  sorry

end curves_intersect_on_x_axis_l368_36805


namespace ratio_proof_l368_36816

theorem ratio_proof (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : (4 * x + y) / (x - 4 * y) = 3) :
    (x + 4 * y) / (4 * x - y) = 9 / 53 :=
  sorry

end ratio_proof_l368_36816


namespace scientific_notation_of_virus_diameter_l368_36890

theorem scientific_notation_of_virus_diameter :
  0.00000012 = 1.2 * 10 ^ (-7) :=
by
  sorry

end scientific_notation_of_virus_diameter_l368_36890


namespace find_rate_l368_36899

def simple_interest_rate (P A T : ℕ) : ℕ :=
  ((A - P) * 100) / (P * T)

theorem find_rate :
  simple_interest_rate 750 1200 5 = 12 :=
by
  -- This is the statement of equality we need to prove
  sorry

end find_rate_l368_36899


namespace lcm_36_125_l368_36895

-- Define the prime factorizations
def factorization_36 : List (ℕ × ℕ) := [(2, 2), (3, 2)]
def factorization_125 : List (ℕ × ℕ) := [(5, 3)]

-- Least common multiple definition
noncomputable def my_lcm (a b : ℕ) : ℕ :=
  a * b / (Nat.gcd a b)

-- Theorem to prove
theorem lcm_36_125 : my_lcm 36 125 = 4500 :=
by
  sorry

end lcm_36_125_l368_36895


namespace probability_both_blue_buttons_l368_36889

theorem probability_both_blue_buttons :
  let initial_red_C := 6
  let initial_blue_C := 12
  let initial_total_C := initial_red_C + initial_blue_C
  let remaining_fraction_C := 2 / 3
  let remaining_total_C := initial_total_C * remaining_fraction_C
  let removed_buttons := initial_total_C - remaining_total_C
  let removed_red := removed_buttons / 2
  let removed_blue := removed_buttons / 2
  let remaining_blue_C := initial_blue_C - removed_blue
  let total_remaining_C := remaining_total_C
  let probability_blue_C := remaining_blue_C / total_remaining_C
  let probability_blue_D := removed_blue / removed_buttons
  probability_blue_C * probability_blue_D = 3 / 8 :=
by
  sorry

end probability_both_blue_buttons_l368_36889


namespace intersection_A_B_l368_36832

open Set

-- Conditions given in the problem
def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | 0 < x ∧ x < 3}

-- Statement to prove, no proof needed
theorem intersection_A_B : A ∩ B = {1, 2} := 
sorry

end intersection_A_B_l368_36832


namespace smallest_b_l368_36862

theorem smallest_b (b: ℕ) (h1: b > 3) (h2: ∃ n: ℕ, n^3 = 2 * b + 3) : b = 12 :=
sorry

end smallest_b_l368_36862


namespace appetizer_cost_per_person_l368_36874

theorem appetizer_cost_per_person
    (cost_per_bag: ℕ)
    (num_bags: ℕ)
    (cost_creme_fraiche: ℕ)
    (cost_caviar: ℕ)
    (num_people: ℕ)
    (h1: cost_per_bag = 1)
    (h2: num_bags = 3)
    (h3: cost_creme_fraiche = 5)
    (h4: cost_caviar = 73)
    (h5: num_people = 3):
    (cost_per_bag * num_bags + cost_creme_fraiche + cost_caviar) / num_people = 27 := 
  by
    sorry

end appetizer_cost_per_person_l368_36874


namespace range_of_a_l368_36891

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : 3 * a ≥ 1) (h3 : 4 * a ≤ 3 / 2) : 
  (1 / 3) ≤ a ∧ a ≤ (3 / 8) :=
by
  sorry

end range_of_a_l368_36891


namespace length_of_LN_l368_36824

theorem length_of_LN 
  (sinN : ℝ)
  (LM LN : ℝ)
  (h1 : sinN = 3 / 5)
  (h2 : LM = 20)
  (h3 : sinN = LM / LN) :
  LN = 100 / 3 :=
by
  sorry

end length_of_LN_l368_36824


namespace quadruples_solution_l368_36882

noncomputable
def valid_quadruples (x1 x2 x3 x4 : ℝ) : Prop :=
  (x1 + x2 * x3 * x4 = 2) ∧
  (x2 + x1 * x3 * x4 = 2) ∧
  (x3 + x1 * x2 * x4 = 2) ∧
  (x4 + x1 * x2 * x3 = 2) ∧
  (x1 ≠ 0) ∧ (x2 ≠ 0) ∧ (x3 ≠ 0) ∧ (x4 ≠ 0)

theorem quadruples_solution (x1 x2 x3 x4 : ℝ) :
  valid_quadruples x1 x2 x3 x4 ↔ 
  (x1 = 1 ∧ x2 = 1 ∧ x3 = 1 ∧ x4 = 1) ∨
  (x1 = -1 ∧ x2 = -1 ∧ x3 = -1 ∧ x4 = 3) ∨
  (x1 = -1 ∧ x2 = -1 ∧ x3 = 3 ∧ x4 = -1) ∨
  (x1 = -1 ∧ x2 = 3 ∧ x3 = -1 ∧ x4 = -1) ∨
  (x1 = 3 ∧ x2 = -1 ∧ x3 = -1 ∧ x4 = -1) := 
by sorry

end quadruples_solution_l368_36882


namespace existence_of_committees_l368_36867

noncomputable def committeesExist : Prop :=
∃ (C : Fin 1990 → Fin 11 → Fin 3), 
  (∀ i j, i ≠ j → C i ≠ C j) ∧
  (∀ i j, i = j + 1 ∨ (i = 0 ∧ j = 1990 - 1) → ∃ k, C i k = C j k)

theorem existence_of_committees : committeesExist :=
sorry

end existence_of_committees_l368_36867


namespace probability_both_boys_probability_exactly_one_girl_probability_at_least_one_girl_l368_36830

noncomputable def total_outcomes : ℕ := Nat.choose 6 2

noncomputable def prob_both_boys : ℚ := (Nat.choose 4 2 : ℚ) / total_outcomes

noncomputable def prob_exactly_one_girl : ℚ := ((Nat.choose 4 1) * (Nat.choose 2 1) : ℚ) / total_outcomes

noncomputable def prob_at_least_one_girl : ℚ := 1 - prob_both_boys

theorem probability_both_boys : prob_both_boys = 2 / 5 := by sorry
theorem probability_exactly_one_girl : prob_exactly_one_girl = 8 / 15 := by sorry
theorem probability_at_least_one_girl : prob_at_least_one_girl = 3 / 5 := by sorry

end probability_both_boys_probability_exactly_one_girl_probability_at_least_one_girl_l368_36830


namespace value_of_f_l368_36844

noncomputable def f (a b x : ℝ) := a * Real.exp x + b * x 
noncomputable def f' (a b x : ℝ) := a * Real.exp x + b

theorem value_of_f'_at_1 (a b : ℝ)
  (h₁ : f a b 0 = 1)
  (h₂ : f' (a := a) (b := b) 0 = 0) :
  f' (a := a) (b := b) 1 = Real.exp 1 - 1 :=
by
  sorry

end value_of_f_l368_36844


namespace total_distance_craig_walked_l368_36898

theorem total_distance_craig_walked :
  0.2 + 0.7 = 0.9 :=
by sorry

end total_distance_craig_walked_l368_36898


namespace miles_run_on_tuesday_l368_36836

-- Defining the distances run on specific days
def distance_monday : ℝ := 4.2
def distance_wednesday : ℝ := 3.6
def distance_thursday : ℝ := 4.4

-- Average distance run on each of the days Terese runs
def average_distance : ℝ := 4
-- Number of days Terese runs
def running_days : ℕ := 4

-- Defining the total distance calculated using the average distance and number of days
def total_distance : ℝ := average_distance * running_days

-- Defining the total distance run on Monday, Wednesday, and Thursday
def total_other_days : ℝ := distance_monday + distance_wednesday + distance_thursday

-- The distance run on Tuesday can be defined as the difference between the total distance and the total distance on other days
theorem miles_run_on_tuesday : 
  total_distance - total_other_days = 3.8 :=
by
  sorry

end miles_run_on_tuesday_l368_36836


namespace commute_times_l368_36888

theorem commute_times (x y : ℝ) 
  (h1 : x + y = 20)
  (h2 : (x - 10)^2 + (y - 10)^2 = 8) : |x - y| = 4 := 
sorry

end commute_times_l368_36888


namespace tan_C_value_l368_36848

theorem tan_C_value (A B C : ℝ)
  (h_cos_A : Real.cos A = 4/5)
  (h_tan_A_minus_B : Real.tan (A - B) = -1/2) :
  Real.tan C = 11/2 :=
sorry

end tan_C_value_l368_36848


namespace incorrect_statement_A_l368_36893

-- Definitions for the conditions
def conditionA (x : ℝ) : Prop := -3 * x > 9
def conditionB (x : ℝ) : Prop := 2 * x - 1 < 0
def conditionC (x : ℤ) : Prop := x < 10
def conditionD (x : ℤ) : Prop := x < 2

-- Formal theorem statement
theorem incorrect_statement_A : ¬ (∀ x : ℝ, conditionA x ↔ x < -3) :=
by 
  sorry

end incorrect_statement_A_l368_36893


namespace fundraiser_price_per_item_l368_36833

theorem fundraiser_price_per_item
  (students_brownies : ℕ)
  (brownies_per_student : ℕ)
  (students_cookies : ℕ)
  (cookies_per_student : ℕ)
  (students_donuts : ℕ)
  (donuts_per_student : ℕ)
  (total_amount_raised : ℕ)
  (total_brownies : ℕ := students_brownies * brownies_per_student)
  (total_cookies : ℕ := students_cookies * cookies_per_student)
  (total_donuts : ℕ := students_donuts * donuts_per_student)
  (total_items : ℕ := total_brownies + total_cookies + total_donuts)
  (price_per_item : ℕ := total_amount_raised / total_items) :
  students_brownies = 30 →
  brownies_per_student = 12 →
  students_cookies = 20 →
  cookies_per_student = 24 →
  students_donuts = 15 →
  donuts_per_student = 12 →
  total_amount_raised = 2040 →
  price_per_item = 2 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2, h3, h4, h5, h6, h7]
  sorry

end fundraiser_price_per_item_l368_36833


namespace rectangle_dimensions_l368_36856

theorem rectangle_dimensions (w l : ℚ) (h1 : 2 * l + 2 * w = 2 * l * w) (h2 : l = 3 * w) :
  w = 4 / 3 ∧ l = 4 :=
by
  sorry

end rectangle_dimensions_l368_36856


namespace alice_favorite_number_l368_36870

def is_multiple (x y : ℕ) : Prop := ∃ k : ℕ, k * y = x
def digit_sum (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem alice_favorite_number 
  (n : ℕ) 
  (h1 : 90 ≤ n ∧ n ≤ 150) 
  (h2 : is_multiple n 13) 
  (h3 : ¬ is_multiple n 4) 
  (h4 : is_multiple (digit_sum n) 4) : 
  n = 143 := 
by 
  sorry

end alice_favorite_number_l368_36870


namespace find_a_plus_b_l368_36810

noncomputable def f (a b x : ℝ) : ℝ := a * x + b
def h (x : ℝ) : ℝ := 3 * x - 6

theorem find_a_plus_b (a b : ℝ) (h_cond : ∀ x : ℝ, h (f a b x) = 4 * x + 3) : a + b = 13 / 3 :=
by
  sorry

end find_a_plus_b_l368_36810


namespace red_balls_count_l368_36892

theorem red_balls_count (white_balls_ratio : ℕ) (red_balls_ratio : ℕ) (total_white_balls : ℕ)
  (h_ratio : white_balls_ratio = 3 ∧ red_balls_ratio = 2)
  (h_white_balls : total_white_balls = 9) :
  ∃ (total_red_balls : ℕ), total_red_balls = 6 :=
by
  sorry

end red_balls_count_l368_36892


namespace distance_to_place_l368_36825

theorem distance_to_place (rowing_speed still_water : ℝ) (downstream_speed : ℝ)
                         (upstream_speed : ℝ) (total_time : ℝ) (distance : ℝ) :
  rowing_speed = 10 → downstream_speed = 2 → upstream_speed = 3 →
  total_time = 10 → distance = 44.21 → 
  (distance / (rowing_speed + downstream_speed) + distance / (rowing_speed - upstream_speed)) = 10 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3]
  field_simp
  sorry

end distance_to_place_l368_36825


namespace vector_subtraction_l368_36883

def a : ℝ × ℝ × ℝ := (1, -2, 1)
def b : ℝ × ℝ × ℝ := (1, 0, 2)

theorem vector_subtraction : a - b = (0, -2, -1) := 
by 
  unfold a b
  simp
  sorry

end vector_subtraction_l368_36883


namespace distance_from_origin_l368_36807

theorem distance_from_origin (x y : ℝ) :
  (x, y) = (12, -5) →
  (0, 0) = (0, 0) →
  Real.sqrt ((x - 0)^2 + (y - 0)^2) = 13 :=
by
  -- Please note, the proof steps go here, but they are omitted as per instructions.
  -- Typically we'd use sorry to indicate the proof is missing.
  sorry

end distance_from_origin_l368_36807


namespace janet_better_condition_count_l368_36850

noncomputable def janet_initial := 10
noncomputable def janet_sells := 6
noncomputable def janet_remaining := janet_initial - janet_sells
noncomputable def brother_gives := 2 * janet_remaining
noncomputable def janet_after_brother := janet_remaining + brother_gives
noncomputable def janet_total := 24

theorem janet_better_condition_count : 
  janet_total - janet_after_brother = 12 := by
  sorry

end janet_better_condition_count_l368_36850


namespace sin_neg_seven_pi_over_three_correct_l368_36863

noncomputable def sin_neg_seven_pi_over_three : Prop :=
  (Real.sin (-7 * Real.pi / 3) = - (Real.sqrt 3 / 2))

theorem sin_neg_seven_pi_over_three_correct : sin_neg_seven_pi_over_three := 
by
  sorry

end sin_neg_seven_pi_over_three_correct_l368_36863


namespace inequality_solution_l368_36834

theorem inequality_solution (m : ℝ) : (∀ x : ℝ, m * x^2 - m * x + 1/2 > 0) ↔ (0 ≤ m ∧ m < 2) :=
by
  sorry

end inequality_solution_l368_36834


namespace greatest_award_correct_l368_36827

-- Definitions and constants
def total_prize : ℕ := 600
def num_winners : ℕ := 15
def min_award : ℕ := 15
def prize_fraction_num : ℕ := 2
def prize_fraction_den : ℕ := 5
def winners_fraction_num : ℕ := 3
def winners_fraction_den : ℕ := 5

-- Conditions (translated and simplified)
def num_specific_winners : ℕ := (winners_fraction_num * num_winners) / winners_fraction_den
def specific_prize : ℕ := (prize_fraction_num * total_prize) / prize_fraction_den
def remaining_winners : ℕ := num_winners - num_specific_winners
def min_total_award_remaining : ℕ := remaining_winners * min_award
def remaining_prize : ℕ := total_prize - min_total_award_remaining
def min_award_specific : ℕ := num_specific_winners - 1
def sum_min_awards_specific : ℕ := min_award_specific * min_award

-- Correct answer
def greatest_award : ℕ := remaining_prize - sum_min_awards_specific

-- Theorem statement (Proof skipped with sorry)
theorem greatest_award_correct :
  greatest_award = 390 := sorry

end greatest_award_correct_l368_36827


namespace total_cost_is_660_l368_36857

def total_material_cost : ℝ :=
  let velvet_area := (12 * 4) * 3
  let velvet_cost := velvet_area * 3
  let silk_cost := 2 * 6
  let lace_cost := 5 * 2 * 10
  let bodice_cost := silk_cost + lace_cost
  let satin_area := 2.5 * 1.5
  let satin_cost := satin_area * 4
  let leather_area := 1 * 1.5 * 2
  let leather_cost := leather_area * 5
  let wool_area := 5 * 2
  let wool_cost := wool_area * 8
  let ribbon_cost := 3 * 2
  velvet_cost + bodice_cost + satin_cost + leather_cost + wool_cost + ribbon_cost

theorem total_cost_is_660 : total_material_cost = 660 := by
  sorry

end total_cost_is_660_l368_36857


namespace track_length_l368_36877

theorem track_length (L : ℕ)
  (h1 : ∃ B S : ℕ, B = 120 ∧ (L - B) = S ∧ (S + 200) - B = (L + 80) - B)
  (h2 : L + 80 = 440 - L) : L = 180 := 
  by
    sorry

end track_length_l368_36877


namespace am_gm_example_l368_36873

variable {x y z : ℝ}

theorem am_gm_example (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  x / y + y / z + z / x + y / x + z / y + x / z ≥ 6 :=
sorry

end am_gm_example_l368_36873


namespace train_people_count_l368_36896

theorem train_people_count :
  let initial := 48
  let after_first_stop := initial - 13 + 5
  let after_second_stop := after_first_stop - 9 + 10 - 2
  let after_third_stop := after_second_stop - 7 + 4 - 3
  let after_fourth_stop := after_third_stop - 16 + 7 - 5
  let after_fifth_stop := after_fourth_stop - 8 + 15
  after_fifth_stop = 26 := sorry

end train_people_count_l368_36896


namespace sum_q_p_values_l368_36858

def p (x : ℤ) : ℤ := x^2 - 4

def q (x : ℤ) : ℤ := -abs x

theorem sum_q_p_values : 
  (q (p (-3)) + q (p (-2)) + q (p (-1)) + q (p (0)) + q (p (1)) + q (p (2)) + q (p (3))) = -20 :=
by
  sorry

end sum_q_p_values_l368_36858
