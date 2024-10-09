import Mathlib

namespace pool_students_count_l1295_129571

noncomputable def total_students (total_women : ℕ) (female_students : ℕ) (extra_men : ℕ) (non_student_men : ℕ) : ℕ := 
  let total_men := total_women + extra_men
  let male_students := total_men - non_student_men
  female_students + male_students

theorem pool_students_count
  (total_women : ℕ := 1518)
  (female_students : ℕ := 536)
  (extra_men : ℕ := 525)
  (non_student_men : ℕ := 1257) :
  total_students total_women female_students extra_men non_student_men = 1322 := 
by
  sorry

end pool_students_count_l1295_129571


namespace man_and_son_work_together_l1295_129587

theorem man_and_son_work_together (man_days son_days : ℕ) (h_man : man_days = 15) (h_son : son_days = 10) :
  (1 / (1 / man_days + 1 / son_days) = 6) :=
by
  rw [h_man, h_son]
  sorry

end man_and_son_work_together_l1295_129587


namespace sum_is_seventeen_l1295_129596

variable (x y : ℕ)

def conditions (x y : ℕ) : Prop :=
  x > y ∧ x - y = 3 ∧ x * y = 56

theorem sum_is_seventeen (x y : ℕ) (h: conditions x y) : x + y = 17 :=
by
  sorry

end sum_is_seventeen_l1295_129596


namespace ordered_pair_solution_l1295_129572

theorem ordered_pair_solution :
  ∃ x y : ℚ, 7 * x - 50 * y = 3 ∧ 3 * y - x = 5 ∧ x = -259 / 29 ∧ y = -38 / 29 :=
by sorry

end ordered_pair_solution_l1295_129572


namespace arithmetic_sequence_angles_sum_l1295_129539

theorem arithmetic_sequence_angles_sum (A B C : ℝ) (h₁ : A + B + C = 180) (h₂ : 2 * B = A + C) :
  A + C = 120 :=
by
  sorry

end arithmetic_sequence_angles_sum_l1295_129539


namespace candy_store_food_colouring_amount_l1295_129511

theorem candy_store_food_colouring_amount :
  let lollipop_colour := 5 -- each lollipop uses 5ml of food colouring
  let hard_candy_colour := 20 -- each hard candy uses 20ml of food colouring
  let num_lollipops := 100 -- the candy store makes 100 lollipops in one day
  let num_hard_candies := 5 -- the candy store makes 5 hard candies in one day
  (num_lollipops * lollipop_colour) + (num_hard_candies * hard_candy_colour) = 600 :=
by
  let lollipop_colour := 5
  let hard_candy_colour := 20
  let num_lollipops := 100
  let num_hard_candies := 5
  show (num_lollipops * lollipop_colour) + (num_hard_candies * hard_candy_colour) = 600
  sorry

end candy_store_food_colouring_amount_l1295_129511


namespace sufficient_condition_l1295_129517

theorem sufficient_condition (p q r : Prop) (hpq : p → q) (hqr : q → r) : p → r :=
by
  intro hp
  apply hqr
  apply hpq
  exact hp

end sufficient_condition_l1295_129517


namespace third_team_cups_l1295_129583

theorem third_team_cups (required_cups : ℕ) (first_team : ℕ) (second_team : ℕ) (third_team : ℕ) :
  required_cups = 280 ∧ first_team = 90 ∧ second_team = 120 →
  third_team = required_cups - (first_team + second_team) :=
by
  intro h
  rcases h with ⟨h1, h2, h3⟩
  sorry

end third_team_cups_l1295_129583


namespace number_of_tricycles_l1295_129545

def num_bicycles : Nat := 24
def wheels_per_bicycle : Nat := 2
def wheels_per_tricycle : Nat := 3
def total_wheels : Nat := 90

theorem number_of_tricycles : ∃ T : Nat, (wheels_per_bicycle * num_bicycles) + (wheels_per_tricycle * T) = total_wheels ∧ T = 14 := by
  sorry

end number_of_tricycles_l1295_129545


namespace find_k_l1295_129575

variable (x y k : ℝ)

-- Definition: the line equations and the intersection condition
def line1_eq (x y k : ℝ) : Prop := 3 * x - 2 * y = k
def line2_eq (x y : ℝ) : Prop := x - 0.5 * y = 10
def intersect_at_x (x : ℝ) : Prop := x = -6

-- The theorem we need to prove
theorem find_k (h1 : line1_eq x y k)
               (h2 : line2_eq x y)
               (h3 : intersect_at_x x) :
               k = 46 :=
sorry

end find_k_l1295_129575


namespace minimum_days_l1295_129543

theorem minimum_days (n : ℕ) (rain_afternoon : ℕ) (sunny_afternoon : ℕ) (sunny_morning : ℕ) :
  rain_afternoon + sunny_afternoon = 7 ∧
  sunny_afternoon <= 5 ∧
  sunny_morning <= 6 ∧
  sunny_morning + rain_afternoon = 7 ∧
  n = 11 :=
by
  sorry

end minimum_days_l1295_129543


namespace union_of_M_N_l1295_129563

-- Define the sets M and N
def M : Set ℕ := {0, 2, 3}
def N : Set ℕ := {1, 3}

-- State the theorem to prove that M ∪ N = {0, 1, 2, 3}
theorem union_of_M_N : M ∪ N = {0, 1, 2, 3} :=
by
  sorry -- Proof goes here

end union_of_M_N_l1295_129563


namespace charlie_paints_140_square_feet_l1295_129581

-- Define the conditions
def total_area : ℕ := 320
def ratio_allen : ℕ := 4
def ratio_ben : ℕ := 5
def ratio_charlie : ℕ := 7
def total_parts : ℕ := ratio_allen + ratio_ben + ratio_charlie
def area_per_part := total_area / total_parts
def charlie_parts := 7

-- Prove the main statement
theorem charlie_paints_140_square_feet : charlie_parts * area_per_part = 140 := by
  sorry

end charlie_paints_140_square_feet_l1295_129581


namespace average_first_two_l1295_129577

theorem average_first_two (a b c d e f : ℝ)
  (h1 : (a + b + c + d + e + f) = 16.8)
  (h2 : (c + d) = 4.6)
  (h3 : (e + f) = 7.4) : 
  (a + b) / 2 = 2.4 :=
by
  sorry

end average_first_two_l1295_129577


namespace ratio_used_to_total_apples_l1295_129523

noncomputable def total_apples_bonnie : ℕ := 8
noncomputable def total_apples_samuel : ℕ := total_apples_bonnie + 20
noncomputable def eaten_apples_samuel : ℕ := total_apples_samuel / 2
noncomputable def used_for_pie_samuel : ℕ := total_apples_samuel - eaten_apples_samuel - 10

theorem ratio_used_to_total_apples : used_for_pie_samuel / (Nat.gcd used_for_pie_samuel total_apples_samuel) = 1 ∧
                                     total_apples_samuel / (Nat.gcd used_for_pie_samuel total_apples_samuel) = 7 := by
  sorry

end ratio_used_to_total_apples_l1295_129523


namespace parabola_focus_eq_l1295_129585

theorem parabola_focus_eq (focus : ℝ × ℝ) (hfocus : focus = (0, 1)) :
  ∃ (p : ℝ), p = 1 ∧ ∀ (x y : ℝ), x^2 = 4 * p * y → x^2 = 4 * y :=
by { sorry }

end parabola_focus_eq_l1295_129585


namespace floor_sqrt_80_l1295_129561

theorem floor_sqrt_80 : ∀ (x : ℝ), 8 ^ 2 < 80 ∧ 80 < 9 ^ 2 → x = 8 :=
by
  intros x h
  sorry

end floor_sqrt_80_l1295_129561


namespace construct_rectangle_l1295_129505

-- Define the essential properties of the rectangles
structure Rectangle where
  length : ℕ
  width : ℕ 

-- Define the given rectangles
def r1 : Rectangle := ⟨7, 1⟩
def r2 : Rectangle := ⟨6, 1⟩
def r3 : Rectangle := ⟨5, 1⟩
def r4 : Rectangle := ⟨4, 1⟩
def r5 : Rectangle := ⟨3, 1⟩
def r6 : Rectangle := ⟨2, 1⟩
def s  : Rectangle := ⟨1, 1⟩

-- Hypothesis for condition that length of each side of resulting rectangle should be > 1
def validSide (rect : Rectangle) : Prop :=
  rect.length > 1 ∧ rect.width > 1

-- The proof statement
theorem construct_rectangle : 
  (∃ rect1 rect2 rect3 rect4 : Rectangle, 
      rect1 = ⟨7, 1⟩ ∧ rect2 = ⟨6, 1⟩ ∧ rect3 = ⟨5, 1⟩ ∧ rect4 = ⟨4, 1⟩) →
  (∃ rect5 rect6 : Rectangle, 
      rect5 = ⟨3, 1⟩ ∧ rect6 = ⟨2, 1⟩) →
  (∃ square : Rectangle, 
      square = ⟨1, 1⟩) →
  (∃ compositeRect : Rectangle, 
      compositeRect.length = 7 ∧ 
      compositeRect.width = 4 ∧ 
      validSide compositeRect) :=
sorry

end construct_rectangle_l1295_129505


namespace problem_1_problem_2_l1295_129538

open Real

-- Part 1
theorem problem_1 (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 4) : 
  (1 / a + 1 / (b + 1) ≥ 4 / 5) :=
sorry

-- Part 2
theorem problem_2 : 
  ∃ (a b : ℝ), 0 < a ∧ 0 < b ∧ a + b = 4 ∧ (4 / (a * b) + a / b = (1 + sqrt 5) / 2) :=
sorry

end problem_1_problem_2_l1295_129538


namespace find_Z_l1295_129515

open Complex

-- Definitions
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0

theorem find_Z (Z : ℂ) (h1 : abs Z = 3) (h2 : is_pure_imaginary (Z + (3 * Complex.I))) : Z = 3 * Complex.I :=
by
  sorry

end find_Z_l1295_129515


namespace yard_area_l1295_129541

theorem yard_area (posts : Nat) (spacing : Real) (longer_factor : Nat) (shorter_side_posts longer_side_posts : Nat)
  (h1 : posts = 24)
  (h2 : spacing = 3)
  (h3 : longer_factor = 3)
  (h4 : 2 * (shorter_side_posts + longer_side_posts) = posts - 4)
  (h5 : longer_side_posts = 3 * shorter_side_posts + 2) :
  (spacing * (shorter_side_posts - 1)) * (spacing * (longer_side_posts - 1)) = 144 :=
by
  sorry

end yard_area_l1295_129541


namespace complex_in_second_quadrant_l1295_129598

-- Define the complex number z based on the problem conditions.
def z : ℂ := Complex.I + (Complex.I^6)

-- State the condition to check whether z is in the second quadrant.
def is_in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

-- Formulate the theorem stating that the complex number z is in the second quadrant.
theorem complex_in_second_quadrant : is_in_second_quadrant z :=
by
  sorry

end complex_in_second_quadrant_l1295_129598


namespace profit_percentage_mobile_l1295_129576

-- Definitions derived from conditions
def cost_price_grinder : ℝ := 15000
def cost_price_mobile : ℝ := 8000
def loss_percentage_grinder : ℝ := 0.05
def total_profit : ℝ := 50
def selling_price_grinder := cost_price_grinder * (1 - loss_percentage_grinder)
def total_cost_price := cost_price_grinder + cost_price_mobile
def total_selling_price := total_cost_price + total_profit
def selling_price_mobile := total_selling_price - selling_price_grinder
def profit_mobile := selling_price_mobile - cost_price_mobile

-- The theorem to prove the profit percentage on the mobile phone is 10%
theorem profit_percentage_mobile : (profit_mobile / cost_price_mobile) * 100 = 10 :=
by
  sorry

end profit_percentage_mobile_l1295_129576


namespace negation_of_implication_l1295_129551

theorem negation_of_implication {r p q : Prop} :
  ¬ (r → (p ∨ q)) ↔ (¬ r → (¬ p ∧ ¬ q)) :=
by sorry

end negation_of_implication_l1295_129551


namespace find_teacher_age_l1295_129584

/-- Given conditions: 
1. The class initially has 30 students with an average age of 10.
2. One student aged 11 leaves the class.
3. The average age of the remaining 29 students plus the teacher is 11.
Prove that the age of the teacher is 30 years.
-/
theorem find_teacher_age (total_students : ℕ) (avg_age : ℕ) (left_student_age : ℕ) 
  (remaining_avg_age : ℕ) (teacher_age : ℕ) :
  total_students = 30 →
  avg_age = 10 →
  left_student_age = 11 →
  remaining_avg_age = 11 →
  289 + teacher_age = 29 * remaining_avg_age + teacher_age →
  teacher_age = 30 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end find_teacher_age_l1295_129584


namespace number_of_polynomials_l1295_129580

-- Define conditions
def is_positive_integer (n : ℤ) : Prop :=
  5 * 151 * n > 0

-- Define the main theorem
theorem number_of_polynomials (n : ℤ) (h : is_positive_integer n) : 
  ∃ k : ℤ, k = ⌊n / 2⌋ + 1 :=
by
  sorry

end number_of_polynomials_l1295_129580


namespace simplify_expression_l1295_129597

theorem simplify_expression :
  (-2 : ℝ) ^ 2005 + (-2) ^ 2006 + (3 : ℝ) ^ 2007 - (2 : ℝ) ^ 2008 =
  -7 * (2 : ℝ) ^ 2005 + (3 : ℝ) ^ 2007 := 
by
    sorry

end simplify_expression_l1295_129597


namespace train_length_is_correct_l1295_129537

noncomputable def speed_of_train_kmph : ℝ := 77.993280537557

noncomputable def speed_of_man_kmph : ℝ := 6

noncomputable def conversion_factor : ℝ := 5 / 18

noncomputable def speed_of_train_mps : ℝ := speed_of_train_kmph * conversion_factor

noncomputable def speed_of_man_mps : ℝ := speed_of_man_kmph * conversion_factor

noncomputable def relative_speed : ℝ := speed_of_train_mps + speed_of_man_mps

noncomputable def time_to_pass_man : ℝ := 6

noncomputable def length_of_train : ℝ := relative_speed * time_to_pass_man

theorem train_length_is_correct : length_of_train = 139.99 := by
  sorry

end train_length_is_correct_l1295_129537


namespace number_of_solutions_l1295_129567

-- Define the equation
def equation (x : ℝ) : Prop := (3 * x^2 - 15 * x) / (x^2 - 7 * x + 10) = x - 4

-- State the problem with conditions and conclusion
theorem number_of_solutions : (∀ x : ℝ, x ≠ 2 ∧ x ≠ 5 → equation x) ↔ (∃ x1 x2 : ℝ, x1 ≠ 2 ∧ x1 ≠ 5 ∧ x2 ≠ 2 ∧ x2 ≠ 5 ∧ equation x1 ∧ equation x2) :=
by
  sorry

end number_of_solutions_l1295_129567


namespace fleas_initial_minus_final_l1295_129591

theorem fleas_initial_minus_final (F : ℕ) (h : F / 16 = 14) :
  F - 14 = 210 :=
sorry

end fleas_initial_minus_final_l1295_129591


namespace problem_1_minimum_value_problem_2_range_of_a_l1295_129534

noncomputable def e : ℝ := Real.exp 1  -- Definition of e as exp(1)

-- Question I:
-- Prove that the minimum value of the function f(x) = e^x - e*x - e is -e.
theorem problem_1_minimum_value :
  ∃ x : ℝ, (∀ y : ℝ, (Real.exp x - e * x - e) ≤ (Real.exp y - e * y - e))
  ∧ (Real.exp x - e * x - e) = -e := 
sorry

-- Question II:
-- Prove that the range of values for a such that f(x) = e^x - a*x - a >= 0 for all x is [0, 1].
theorem problem_2_range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, (Real.exp x - a * x - a) ≥ 0) ↔ 0 ≤ a ∧ a ≤ 1 :=
sorry

end problem_1_minimum_value_problem_2_range_of_a_l1295_129534


namespace machine_does_not_print_13824_l1295_129559

-- Definitions corresponding to the conditions:
def machine_property (S : Set ℕ) : Prop :=
  ∀ n ∈ S, (2 * n) ∉ S ∧ (3 * n) ∉ S

def machine_prints_2 (S : Set ℕ) : Prop :=
  2 ∈ S

-- Statement to be proved
theorem machine_does_not_print_13824 (S : Set ℕ) 
  (H1 : machine_property S) 
  (H2 : machine_prints_2 S) : 
  13824 ∉ S :=
sorry

end machine_does_not_print_13824_l1295_129559


namespace sample_size_is_13_l1295_129531

noncomputable def stratified_sample_size : ℕ :=
  let A := 120
  let B := 80
  let C := 60
  let total_units := A + B + C
  let sampled_C_units := 3
  let sampling_fraction := sampled_C_units / C
  let n := sampling_fraction * total_units
  n

theorem sample_size_is_13 :
  stratified_sample_size = 13 := by
  sorry

end sample_size_is_13_l1295_129531


namespace man_rate_still_water_l1295_129560

def speed_with_stream : ℝ := 6
def speed_against_stream : ℝ := 2

theorem man_rate_still_water : (speed_with_stream + speed_against_stream) / 2 = 4 := by
  sorry

end man_rate_still_water_l1295_129560


namespace values_of_a_l1295_129574

noncomputable def M : Set ℝ := {x | x^2 = 1}

noncomputable def N (a : ℝ) : Set ℝ := 
  if a = 0 then ∅ else {x | a * x = 1}

theorem values_of_a (a : ℝ) : (N a ⊆ M) ↔ (a = -1 ∨ a = 0 ∨ a = 1) := by
  sorry

end values_of_a_l1295_129574


namespace speed_of_second_train_l1295_129522

theorem speed_of_second_train
  (t₁ : ℕ := 2)  -- Time the first train sets off (2:00 pm in hours)
  (s₁ : ℝ := 70) -- Speed of the first train in km/h
  (t₂ : ℕ := 3)  -- Time the second train sets off (3:00 pm in hours)
  (t₃ : ℕ := 10) -- Time when the second train catches the first train (10:00 pm in hours)
  : ∃ S : ℝ, S = 80 := sorry

end speed_of_second_train_l1295_129522


namespace abs_inequality_solution_l1295_129513

theorem abs_inequality_solution (x : ℝ) : 
  3 < |x + 2| ∧ |x + 2| ≤ 6 ↔ (1 < x ∧ x ≤ 4) ∨ (-8 ≤ x ∧ x < -5) := 
by
  sorry

end abs_inequality_solution_l1295_129513


namespace fraction_meaningful_l1295_129586

theorem fraction_meaningful (x : ℝ) : (x ≠ 2) ↔ (x - 2 ≠ 0) :=
by
  sorry

end fraction_meaningful_l1295_129586


namespace john_unanswered_questions_l1295_129556

theorem john_unanswered_questions (c w u : ℕ) 
  (h1 : 25 + 5 * c - 2 * w = 95) 
  (h2 : 6 * c - w + 3 * u = 105) 
  (h3 : c + w + u = 30) : 
  u = 2 := 
sorry

end john_unanswered_questions_l1295_129556


namespace find_abc_l1295_129568

noncomputable def a_b_c_exist : Prop :=
  ∃ (a b c : ℝ), 
    (a + b + c = 21/4) ∧ 
    (1/a + 1/b + 1/c = 21/4) ∧ 
    (a * b * c = 1) ∧ 
    (a < b) ∧ (b < c) ∧ 
    (a = 1/4) ∧ (b = 1) ∧ (c = 4)

theorem find_abc : a_b_c_exist :=
sorry

end find_abc_l1295_129568


namespace sum_consecutive_evens_l1295_129508

theorem sum_consecutive_evens (n k : ℕ) (hn : 2 < n) (hk : 2 < k) : 
  ∃ (m : ℕ), n * (n - 1)^(k - 1) = n * (2 * m + (n - 1)) :=
by
  sorry

end sum_consecutive_evens_l1295_129508


namespace jennifer_money_left_over_l1295_129564

theorem jennifer_money_left_over :
  let original_amount := 120
  let sandwich_cost := original_amount / 5
  let museum_ticket_cost := original_amount / 6
  let book_cost := original_amount / 2
  let total_spent := sandwich_cost + museum_ticket_cost + book_cost
  let money_left := original_amount - total_spent
  money_left = 16 :=
by
  let original_amount := 120
  let sandwich_cost := original_amount / 5
  let museum_ticket_cost := original_amount / 6
  let book_cost := original_amount / 2
  let total_spent := sandwich_cost + museum_ticket_cost + book_cost
  let money_left := original_amount - total_spent
  exact sorry

end jennifer_money_left_over_l1295_129564


namespace complementary_angle_decrease_l1295_129553

theorem complementary_angle_decrease :
  (ratio : ℚ := 3 / 7) →
  let total_angle := 90
  let small_angle := (ratio * total_angle) / (1+ratio)
  let large_angle := total_angle - small_angle
  let new_small_angle := small_angle * 1.2
  let new_large_angle := total_angle - new_small_angle
  let decrease_percent := (large_angle - new_large_angle) / large_angle * 100
  decrease_percent = 8.57 :=
by
  sorry

end complementary_angle_decrease_l1295_129553


namespace hose_removal_rate_l1295_129565

theorem hose_removal_rate (w l d : ℝ) (capacity_fraction : ℝ) (drain_time : ℝ) 
  (h_w : w = 60) 
  (h_l : l = 150) 
  (h_d : d = 10) 
  (h_capacity_fraction : capacity_fraction = 0.80) 
  (h_drain_time : drain_time = 1200) : 
  ((w * l * d * capacity_fraction) / drain_time) = 60 :=
by
  -- the proof is omitted here
  sorry

end hose_removal_rate_l1295_129565


namespace arithmetic_operations_correct_l1295_129562

theorem arithmetic_operations_correct :
  (3 + (3 / 3) = (77 / 7) - 7) :=
by
  sorry

end arithmetic_operations_correct_l1295_129562


namespace diameter_of_circle_A_l1295_129502

theorem diameter_of_circle_A (r_B r_C : ℝ) (h1 : r_B = 12) (h2 : r_C = 3)
  (area_relation : ∀ (r_A : ℝ), π * (r_B^2 - r_A^2) = 4 * (π * r_C^2)) :
  ∃ r_A : ℝ, 2 * r_A = 12 * Real.sqrt 3 := by
  -- We will club the given conditions and logical sequence here
  sorry

end diameter_of_circle_A_l1295_129502


namespace m_range_positive_solution_l1295_129542

theorem m_range_positive_solution (m : ℝ) : (∃ x : ℝ, x > 0 ∧ (2 * x + m) / (x - 2) + (x - 1) / (2 - x) = 3) ↔ (m > -7 ∧ m ≠ -3) := by
  sorry

end m_range_positive_solution_l1295_129542


namespace find_point_P_l1295_129509

noncomputable def tangent_at (f : ℝ → ℝ) (x : ℝ) : ℝ := (deriv f) x

theorem find_point_P :
  ∃ (x₀ y₀ : ℝ), (y₀ = (1 / x₀)) 
  ∧ (0 < x₀)
  ∧ (tangent_at (fun x => x^2) 2 = 4)
  ∧ (tangent_at (fun x => (1 / x)) x₀ = -1 / 4) 
  ∧ (x₀ = 2)
  ∧ (y₀ = 1 / 2) :=
sorry

end find_point_P_l1295_129509


namespace find_fraction_squares_l1295_129529

theorem find_fraction_squares (x y z a b c : ℝ) 
  (h1 : x / a + y / b + z / c = 4) 
  (h2 : a / x + b / y + c / z = 0) : 
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 16 := 
by
  sorry

end find_fraction_squares_l1295_129529


namespace percent_error_l1295_129552

theorem percent_error (x : ℝ) (h : x > 0) :
  (abs ((12 * x) - (x / 3)) / (x / 3)) * 100 = 3500 :=
by
  sorry

end percent_error_l1295_129552


namespace sum_smallest_largest_3_digit_numbers_made_up_of_1_2_5_l1295_129521

theorem sum_smallest_largest_3_digit_numbers_made_up_of_1_2_5 :
  let smallest := 125
  let largest := 521
  smallest + largest = 646 := by
  sorry

end sum_smallest_largest_3_digit_numbers_made_up_of_1_2_5_l1295_129521


namespace sum_infinite_series_l1295_129549

noncomputable def series_term (n : ℕ) : ℚ := 
  (2 * n + 3) / (n * (n + 1) * (n + 2))

noncomputable def partial_fractions (n : ℕ) : ℚ := 
  (3 / 2) / n - 1 / (n + 1) - (1 / 2) / (n + 2)

theorem sum_infinite_series : 
  (∑' n : ℕ, series_term (n + 1)) = 5 / 4 := 
by
  sorry

end sum_infinite_series_l1295_129549


namespace range_of_x_squared_f_x_lt_x_squared_minus_f_1_l1295_129547

noncomputable def even_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f x = f (-x)

noncomputable def satisfies_inequality (f f' : ℝ → ℝ) : Prop :=
∀ x : ℝ, 2 * f x + x * f' x < 2

theorem range_of_x_squared_f_x_lt_x_squared_minus_f_1 (f f' : ℝ → ℝ)
  (h_even : even_function f)
  (h_ineq : satisfies_inequality f f')
  : {x : ℝ | x^2 * f x - f 1 < x^2 - 1} = {x : ℝ | x < -1} ∪ {x : ℝ | x > 1} :=
sorry

end range_of_x_squared_f_x_lt_x_squared_minus_f_1_l1295_129547


namespace unique_root_value_l1295_129507

theorem unique_root_value {x n : ℝ} (h : (15 - n) = 15 - (35 / 4)) :
  (x + 5) * (x + 3) = n + 3 * x → n = 35 / 4 :=
sorry

end unique_root_value_l1295_129507


namespace problem_statement_l1295_129550

-- Definition of the arithmetic sequence {a_n}
def a (n : ℕ) : ℕ := n

-- Definition of the geometric sequence {b_n}
def b (n : ℕ) : ℕ := 2^n

-- Definition of the sequence {c_n}
def c (n : ℕ) : ℕ := a n + b n

-- Sum of first n terms of the sequence {c_n}
def S (n : ℕ) : ℕ := (n * (n + 1)) / 2 + 2^(n + 1) - 2

-- Prove the problem statement
theorem problem_statement :
  (a 1 + a 2 = 3) ∧
  (a 4 - a 3 = 1) ∧
  (b 2 = a 4) ∧
  (b 3 = a 8) ∧
  (∀ n : ℕ, c n = a n + b n) ∧
  (∀ n : ℕ, S n = (n * (n + 1)) / 2 + 2^(n + 1) - 2) :=
by {
  sorry -- Proof goes here
}

end problem_statement_l1295_129550


namespace sam_collected_42_cans_l1295_129500

noncomputable def total_cans_collected (bags_saturday : ℕ) (bags_sunday : ℕ) (cans_per_bag : ℕ) : ℕ :=
  bags_saturday + bags_sunday * cans_per_bag

theorem sam_collected_42_cans :
  total_cans_collected 4 3 6 = 42 :=
by
  sorry

end sam_collected_42_cans_l1295_129500


namespace inequality_proof_l1295_129599

noncomputable def x : ℝ := Real.exp (-1/2)
noncomputable def y : ℝ := Real.log 2 / Real.log 5
noncomputable def z : ℝ := Real.log 3

theorem inequality_proof : z > x ∧ x > y := by
  -- Conditions defined as follows:
  -- x = exp(-1/2)
  -- y = log(2) / log(5)
  -- z = log(3)
  -- To be proved:
  -- z > x > y
  sorry

end inequality_proof_l1295_129599


namespace trigonometric_identity_l1295_129557

noncomputable def trigonometric_identity_proof : Prop :=
  let cos_30 := Real.sqrt 3 / 2;
  let sin_60 := Real.sqrt 3 / 2;
  let sin_30 := 1 / 2;
  let cos_60 := 1 / 2;
  (1 - 1 / cos_30) * (1 + 1 / sin_60) * (1 - 1 / sin_30) * (1 + 1 / cos_60) = 1

theorem trigonometric_identity : trigonometric_identity_proof :=
  sorry

end trigonometric_identity_l1295_129557


namespace diagonal_perimeter_ratio_l1295_129510

theorem diagonal_perimeter_ratio
    (b : ℝ)
    (h : b ≠ 0) -- To ensure the garden has non-zero side lengths
    (a : ℝ) (h1: a = 3 * b) 
    (d : ℝ) (h2: d = (Real.sqrt (b^2 + a^2)))
    (P : ℝ) (h3: P = 2 * a + 2 * b)
    (h4 : d = b * (Real.sqrt 10)) :
  d / P = (Real.sqrt 10) / 8 := by
    sorry

end diagonal_perimeter_ratio_l1295_129510


namespace arrangements_15_cents_l1295_129525

def numArrangements (n : ℕ) : ℕ :=
  sorry  -- Function definition which outputs the number of arrangements for sum n

theorem arrangements_15_cents : numArrangements 15 = X :=
  sorry  -- Replace X with the correct calculated number

end arrangements_15_cents_l1295_129525


namespace franks_daily_reading_l1295_129530

-- Define the conditions
def total_pages : ℕ := 612
def days_to_finish : ℕ := 6

-- State the theorem we want to prove
theorem franks_daily_reading : (total_pages / days_to_finish) = 102 :=
by
  sorry

end franks_daily_reading_l1295_129530


namespace line_passing_through_M_l1295_129506

-- Define the point M
def M : ℝ × ℝ := (-3, 4)

-- Define the predicate for a line equation having equal intercepts and passing through point M
def line_eq (x y : ℝ) (a b : ℝ) : Prop :=
  ∃ c : ℝ, ((a = 0 ∧ b = 0 ∧ 4 * x + 3 * y = 0) ∨ (a ≠ 0 ∧ b ≠ 0 ∧ a = b ∧ x + y = 1)) 

theorem line_passing_through_M (x y : ℝ) (a b : ℝ) (h₀ : (-3, 4) = M) (h₁ : ∃ c : ℝ, (a = 0 ∧ b = 0 ∧ 4 * x + 3 * y = 0) ∨ (a ≠ 0 ∧ b ≠ 0 ∧ a = b ∧ x + y = 1)) :
  (4 * x + 3 * y = 0) ∨ (x + y = 1) :=
by
  -- We add 'sorry' to skip the proof
  sorry

end line_passing_through_M_l1295_129506


namespace cover_square_floor_l1295_129540

theorem cover_square_floor (x : ℕ) (h : 2 * x - 1 = 37) : x^2 = 361 :=
by
  sorry

end cover_square_floor_l1295_129540


namespace factorize_x4_minus_4x2_l1295_129535

theorem factorize_x4_minus_4x2 (x : ℝ) : 
  x^4 - 4 * x^2 = x^2 * (x - 2) * (x + 2) :=
by
  sorry

end factorize_x4_minus_4x2_l1295_129535


namespace division_and_subtraction_l1295_129570

theorem division_and_subtraction : (23 ^ 11 / 23 ^ 8) - 15 = 12152 := by
  sorry

end division_and_subtraction_l1295_129570


namespace stadium_surface_area_correct_l1295_129524

noncomputable def stadium_length_yards : ℝ := 62
noncomputable def stadium_width_yards : ℝ := 48
noncomputable def stadium_height_yards : ℝ := 30

noncomputable def stadium_length_feet : ℝ := stadium_length_yards * 3
noncomputable def stadium_width_feet : ℝ := stadium_width_yards * 3
noncomputable def stadium_height_feet : ℝ := stadium_height_yards * 3

def total_surface_area_stadium (length : ℝ) (width : ℝ) (height : ℝ) : ℝ :=
  2 * (length * width + width * height + height * length)

theorem stadium_surface_area_correct :
  total_surface_area_stadium stadium_length_feet stadium_width_feet stadium_height_feet = 110968 := by
  sorry

end stadium_surface_area_correct_l1295_129524


namespace count_divisible_by_25_l1295_129593

-- Define the conditions
def is_positive_four_digit (n : ℕ) : Prop := n >= 1000 ∧ n < 10000

def ends_in_25 (n : ℕ) : Prop := n % 100 = 25

-- Define the main statement to prove
theorem count_divisible_by_25 : 
  (∃ (count : ℕ), count = 90 ∧
  ∀ n, is_positive_four_digit n ∧ ends_in_25 n → count = 90) :=
by {
  -- Outline the proof
  sorry
}

end count_divisible_by_25_l1295_129593


namespace B_days_finish_work_l1295_129514

theorem B_days_finish_work :
  ∀ (W : ℝ) (A_work B_work B_days : ℝ),
  (A_work = W / 9) → 
  (B_work = W / B_days) →
  (3 * (W / 9) + 10 * (W / B_days) = W) →
  B_days = 15 :=
by
  intros W A_work B_work B_days hA_work hB_work hTotal
  sorry

end B_days_finish_work_l1295_129514


namespace cost_of_gravelling_the_path_l1295_129504

-- Define the problem conditions
def plot_length : ℝ := 110
def plot_width : ℝ := 65
def path_width : ℝ := 2.5
def cost_per_sq_meter : ℝ := 0.70

-- Define the dimensions of the grassy area without the path
def grassy_length : ℝ := plot_length - 2 * path_width
def grassy_width : ℝ := plot_width - 2 * path_width

-- Define the area of the entire plot and the grassy area without the path
def area_entire_plot : ℝ := plot_length * plot_width
def area_grassy_area : ℝ := grassy_length * grassy_width

-- Define the area of the path
def area_path : ℝ := area_entire_plot - area_grassy_area

-- Define the cost of gravelling the path
def cost_gravelling_path : ℝ := area_path * cost_per_sq_meter

-- State the theorem
theorem cost_of_gravelling_the_path : cost_gravelling_path = 595 := 
by
  -- The proof is omitted
  sorry

end cost_of_gravelling_the_path_l1295_129504


namespace count_correct_propositions_l1295_129582

def line_parallel_plane (a : Line) (M : Plane) : Prop := sorry
def line_perpendicular_plane (a : Line) (M : Plane) : Prop := sorry
def line_parallel_line (a b : Line) : Prop := sorry
def line_perpendicular_line (a b : Line) : Prop := sorry
def plane_perpendicular_plane (M N : Plane) : Prop := sorry

theorem count_correct_propositions 
  (a b c : Line) 
  (M N : Plane) 
  (h1 : ¬ (line_parallel_plane a M ∧ line_parallel_plane b M → line_parallel_line a b)) 
  (h2 : line_parallel_plane a M ∧ line_perpendicular_plane b M → line_perpendicular_line b a) 
  (h3 : ¬ ((line_parallel_plane a M ∧ line_perpendicular_plane b M ∧ line_perpendicular_line c a ∧ line_perpendicular_line c b) → line_perpendicular_plane c M))
  (h4 : line_perpendicular_plane a M ∧ line_parallel_plane a N → plane_perpendicular_plane M N) :
  (0 + 1 + 0 + 1) = 2 :=
sorry

end count_correct_propositions_l1295_129582


namespace weight_of_B_l1295_129518

noncomputable def A : ℝ := sorry
noncomputable def B : ℝ := sorry
noncomputable def C : ℝ := sorry

theorem weight_of_B :
  (A + B + C) / 3 = 45 → 
  (A + B) / 2 = 40 → 
  (B + C) / 2 = 43 → 
  B = 31 :=
by
  intros h1 h2 h3
  -- detailed proof steps omitted
  sorry

end weight_of_B_l1295_129518


namespace opposite_of_three_l1295_129555

theorem opposite_of_three :
  ∃ x : ℤ, 3 + x = 0 ∧ x = -3 :=
by
  sorry

end opposite_of_three_l1295_129555


namespace negation_of_proposition_l1295_129566

noncomputable def original_proposition :=
  ∀ a b : ℝ, (a * b = 0) → (a = 0)

theorem negation_of_proposition :
  ¬ original_proposition ↔ ∃ a b : ℝ, (a * b = 0) ∧ (a ≠ 0) :=
by
  sorry

end negation_of_proposition_l1295_129566


namespace lcm_25_35_50_l1295_129588

theorem lcm_25_35_50 : Nat.lcm (Nat.lcm 25 35) 50 = 350 := by
  sorry

end lcm_25_35_50_l1295_129588


namespace basketball_team_first_competition_games_l1295_129519

-- Definitions given the conditions
def first_competition_games (x : ℕ) := x
def second_competition_games (x : ℕ) := (5 * x) / 8
def third_competition_games (x : ℕ) := x + (5 * x) / 8
def total_games (x : ℕ) := x + (5 * x) / 8 + (x + (5 * x) / 8)

-- Lean 4 statement to prove the correct answer
theorem basketball_team_first_competition_games : 
  ∃ x : ℕ, total_games x = 130 ∧ first_competition_games x = 40 :=
by
  sorry

end basketball_team_first_competition_games_l1295_129519


namespace mouse_jump_distance_l1295_129573

theorem mouse_jump_distance
  (g f m : ℕ)
  (hg : g = 25)
  (hf : f = g + 32)
  (hm : m = f - 26) :
  m = 31 := by
  sorry

end mouse_jump_distance_l1295_129573


namespace pow_two_div_factorial_iff_exists_l1295_129590

theorem pow_two_div_factorial_iff_exists (n : ℕ) (hn : n > 0) : 
  (∃ k : ℕ, k > 0 ∧ n = 2^(k-1)) ↔ 2^(n-1) ∣ n! := 
by {
  sorry
}

end pow_two_div_factorial_iff_exists_l1295_129590


namespace int_squares_l1295_129579

theorem int_squares (n : ℕ) (h : ∃ k : ℕ, n^4 - n^3 + 3 * n^2 + 5 = k^2) : n = 2 := by
  sorry

end int_squares_l1295_129579


namespace find_number_l1295_129512

theorem find_number (x : ℝ) (h : 0.85 * x = (4 / 5) * 25 + 14) : x = 40 :=
sorry

end find_number_l1295_129512


namespace librarian_took_books_l1295_129520

-- Define variables and conditions
def total_books : ℕ := 46
def books_per_shelf : ℕ := 4
def shelves_needed : ℕ := 9

-- Define the number of books Oliver has left to put away
def books_left : ℕ := shelves_needed * books_per_shelf

-- Define the number of books the librarian took
def books_taken : ℕ := total_books - books_left

-- State the theorem
theorem librarian_took_books : books_taken = 10 := by
  sorry

end librarian_took_books_l1295_129520


namespace no_perfect_square_after_swap_l1295_129558

def is_consecutive_digits (a b c d : ℕ) : Prop := 
  (b = a + 1) ∧ (c = b + 1) ∧ (d = c + 1)

def swap_hundreds_tens (n : ℕ) : ℕ := 
  let d4 := n / 1000
  let d3 := (n % 1000) / 100
  let d2 := (n % 100) / 10
  let d1 := n % 10
  d4 * 1000 + d2 * 100 + d3 * 10 + d1

theorem no_perfect_square_after_swap : ¬ ∃ (n : ℕ), 
  1000 ≤ n ∧ n < 10000 ∧ 
  (let d4 := n / 1000
   let d3 := (n % 1000) / 100
   let d2 := (n % 100) / 10
   let d1 := n % 10
   is_consecutive_digits d4 d3 d2 d1) ∧ 
  let new_number := swap_hundreds_tens n
  (∃ m : ℕ, m * m = new_number) := 
sorry

end no_perfect_square_after_swap_l1295_129558


namespace pablo_puzzle_pieces_per_hour_l1295_129501

theorem pablo_puzzle_pieces_per_hour
  (num_300_puzzles : ℕ)
  (num_500_puzzles : ℕ)
  (pieces_per_300_puzzle : ℕ)
  (pieces_per_500_puzzle : ℕ)
  (max_hours_per_day : ℕ)
  (total_days : ℕ)
  (total_pieces_completed : ℕ)
  (total_hours_spent : ℕ)
  (P : ℕ)
  (h1 : num_300_puzzles = 8)
  (h2 : num_500_puzzles = 5)
  (h3 : pieces_per_300_puzzle = 300)
  (h4 : pieces_per_500_puzzle = 500)
  (h5 : max_hours_per_day = 7)
  (h6 : total_days = 7)
  (h7 : total_pieces_completed = (num_300_puzzles * pieces_per_300_puzzle + num_500_puzzles * pieces_per_500_puzzle))
  (h8 : total_hours_spent = max_hours_per_day * total_days)
  (h9 : P = total_pieces_completed / total_hours_spent) :
  P = 100 :=
sorry

end pablo_puzzle_pieces_per_hour_l1295_129501


namespace mod_2_200_sub_3_l1295_129533

theorem mod_2_200_sub_3 (h1 : 2^1 % 7 = 2) (h2 : 2^2 % 7 = 4) (h3 : 2^3 % 7 = 1) : (2^200 - 3) % 7 = 1 := 
by
  sorry

end mod_2_200_sub_3_l1295_129533


namespace simplify_expression_l1295_129548

theorem simplify_expression (a b : ℕ) (h : a / b = 1 / 3) : 
    1 - (a - b) / (a - 2 * b) / ((a ^ 2 - b ^ 2) / (a ^ 2 - 4 * a * b + 4 * b ^ 2)) = 3 / 4 := 
by sorry

end simplify_expression_l1295_129548


namespace regression_total_sum_of_squares_l1295_129578

variables (y : Fin 10 → ℝ) (y_hat : Fin 10 → ℝ)
variables (residual_sum_of_squares : ℝ) 

-- Given conditions
def R_squared := 0.95
def RSS := 120.53

-- The total sum of squares is what we need to prove
noncomputable def total_sum_of_squares := 2410.6

-- Statement to prove
theorem regression_total_sum_of_squares :
  1 - RSS / total_sum_of_squares = R_squared := by
sorry

end regression_total_sum_of_squares_l1295_129578


namespace equation_of_line_l1295_129516

theorem equation_of_line (θ : ℝ) (b : ℝ) (k : ℝ) (y x : ℝ) :
  θ = Real.pi / 4 ∧ b = 2 ∧ k = Real.tan θ ∧ k = 1 ∧ y = k * x + b ↔ y = x + 2 :=
by
  intros
  sorry

end equation_of_line_l1295_129516


namespace average_lifespan_is_28_l1295_129526

-- Define the given data
def batteryLifespans : List ℕ := [30, 35, 25, 25, 30, 34, 26, 25, 29, 21]

-- Define a function to calculate the average of a list of natural numbers
def average (lst : List ℕ) : ℚ :=
  (lst.sum : ℚ) / lst.length

-- State the theorem to be proved
theorem average_lifespan_is_28 :
  average batteryLifespans = 28 := by
  sorry

end average_lifespan_is_28_l1295_129526


namespace option_C_true_l1295_129589

theorem option_C_true (a b : ℝ):
    (a^2 + b^2 ≥ 2 * a * b) ↔ ((a^2 + b^2 > 2 * a * b) ∨ (a^2 + b^2 = 2 * a * b)) :=
by
  sorry

end option_C_true_l1295_129589


namespace episode_length_l1295_129532

/-- Subject to the conditions provided, we prove the length of each episode watched by Maddie. -/
theorem episode_length
  (total_episodes : ℕ)
  (monday_minutes : ℕ)
  (thursday_minutes : ℕ)
  (weekend_minutes : ℕ)
  (episodes_length : ℕ)
  (monday_watch : monday_minutes = 138)
  (thursday_watch : thursday_minutes = 21)
  (weekend_watch : weekend_minutes = 105)
  (total_episodes_watch : total_episodes = 8)
  (total_minutes : monday_minutes + thursday_minutes + weekend_minutes = total_episodes * episodes_length) :
  episodes_length = 33 := 
by 
  sorry

end episode_length_l1295_129532


namespace division_remainder_l1295_129528

theorem division_remainder (x y : ℕ) (hx : 0 < x) (hy : 0 < y) 
  (hrem : x % y = 3) (hdiv : (x : ℚ) / y = 96.15) : y = 20 :=
sorry

end division_remainder_l1295_129528


namespace polynomial_roots_cubed_l1295_129569

noncomputable def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 5*x - 3
noncomputable def g (x : ℝ) : ℝ := x^3 - 2*x^2 - 5*x + 3

theorem polynomial_roots_cubed {r : ℝ} (h : f r = 0) :
  g (r^3) = 0 := by
  sorry

end polynomial_roots_cubed_l1295_129569


namespace insurance_slogan_equivalence_l1295_129546

variables (H I : Prop)

theorem insurance_slogan_equivalence :
  (∀ x, x → H → I) ↔ (∀ y, y → ¬I → ¬H) :=
sorry

end insurance_slogan_equivalence_l1295_129546


namespace vasya_lowest_position_l1295_129544

theorem vasya_lowest_position
  (n_cyclists : ℕ) (n_stages : ℕ) 
  (stage_positions : ℕ → ℕ → ℕ) -- a function that takes a stage and a cyclist and returns the position (e.g., stage_positions(stage, cyclist) = position)
  (total_time : ℕ → ℕ)  -- a function that takes a cyclist and returns their total time
  (distinct_times : ∀ (c1 c2 : ℕ), c1 ≠ c2 → (total_time c1 ≠ total_time c2) ∧ 
                   ∀ (s : ℕ), stage_positions s c1 ≠ stage_positions s c2)
  (vasya_position : ℕ) (hv : ∀ (s : ℕ), s < n_stages → stage_positions s vasya_position = 7) :
  vasya_position = 91 :=
sorry

end vasya_lowest_position_l1295_129544


namespace range_of_m_l1295_129554

theorem range_of_m (m : ℝ) : 
    (∀ x : ℝ, mx^2 - 6 * m * x + m + 8 ≥ 0) ↔ (0 ≤ m ∧ m ≤ 1) :=
sorry

end range_of_m_l1295_129554


namespace proof_correct_props_l1295_129536

variable (p1 : Prop) (p2 : Prop) (p3 : Prop) (p4 : Prop)

def prop1 : Prop := ∃ (x₀ : ℝ), 0 < x₀ ∧ (1 / 2) * x₀ < (1 / 3) * x₀
def prop2 : Prop := ∃ (x₀ : ℝ), 0 < x₀ ∧ x₀ < 1 ∧ Real.log x₀ / Real.log (1 / 2) > Real.log x₀ / Real.log (1 / 3)
def prop3 : Prop := ∀ (x : ℝ), 0 < x ∧ (1 / 2) ^ x > Real.log x / Real.log (1 / 2)
def prop4 : Prop := ∀ (x : ℝ), 0 < x ∧ x < 1 / 3 ∧ (1 / 2) ^ x < Real.log x / Real.log (1 / 3)

theorem proof_correct_props : prop2 ∧ prop4 :=
by
  sorry -- Proof goes here

end proof_correct_props_l1295_129536


namespace carrie_fourth_day_miles_l1295_129592

theorem carrie_fourth_day_miles (d1 d2 d3 d4: ℕ) (charge_interval charges: ℕ) 
  (h1: d1 = 135) 
  (h2: d2 = d1 + 124) 
  (h3: d3 = 159) 
  (h4: charge_interval = 106) 
  (h5: charges = 7):
  d4 = 742 - (d1 + d2 + d3) :=
by
  sorry

end carrie_fourth_day_miles_l1295_129592


namespace calculate_two_squared_l1295_129594

theorem calculate_two_squared : 2^2 = 4 :=
by
  sorry

end calculate_two_squared_l1295_129594


namespace juniors_score_l1295_129527

theorem juniors_score (n : ℕ) (j s : ℕ) (avg_score students_avg seniors_avg : ℕ)
  (h1 : 0 < n)
  (h2 : j = n / 5)
  (h3 : s = 4 * n / 5)
  (h4 : avg_score = 80)
  (h5 : seniors_avg = 78)
  (h6 : students_avg = avg_score)
  (h7 : n * students_avg = n * avg_score)
  (h8 : s * seniors_avg = 78 * s) :
  (800 - 624) / j = 88 := by
  sorry

end juniors_score_l1295_129527


namespace book_price_increase_percentage_l1295_129595

theorem book_price_increase_percentage :
  let P_original := 300
  let P_new := 480
  (P_new - P_original : ℝ) / P_original * 100 = 60 :=
by
  sorry

end book_price_increase_percentage_l1295_129595


namespace values_of_x_l1295_129503

theorem values_of_x (x : ℝ) : (-2 < x ∧ x < 2) ↔ (x^2 < |x| + 2) := by
  sorry

end values_of_x_l1295_129503
