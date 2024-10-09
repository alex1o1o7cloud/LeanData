import Mathlib

namespace abs_val_equality_l1334_133481

theorem abs_val_equality (m : ℝ) (h : |m| = |(-3 : ℝ)|) : m = 3 ∨ m = -3 :=
sorry

end abs_val_equality_l1334_133481


namespace quadratic_root_k_l1334_133465

theorem quadratic_root_k (k : ℝ) : (∃ x : ℝ, x^2 - 2 * x + k = 0 ∧ x = 1) → k = 1 :=
by
  sorry

end quadratic_root_k_l1334_133465


namespace binomial_minus_floor_divisible_by_seven_l1334_133474

theorem binomial_minus_floor_divisible_by_seven (n : ℕ) (h : n > 7) :
  ((Nat.choose n 7 : ℤ) - ⌊(n : ℤ) / 7⌋) % 7 = 0 :=
  sorry

end binomial_minus_floor_divisible_by_seven_l1334_133474


namespace monotonic_decreasing_interval_range_of_a_l1334_133421

noncomputable def f (a x : ℝ) : ℝ := Real.exp (a * x) * ((a / x) + a + 1)

theorem monotonic_decreasing_interval (a : ℝ) (h : a ≥ -1) :
  (a = -1 → ∀ x, x < -1 → f a x < f a (x + 1)) ∧
  (a ≠ -1 → (∀ x, -1 < a ∧ x < -1 ∨ x > 1 / (a + 1) → f a x < f a (x + 1)) ∧
                (∀ x, -1 < a ∧ -1 < x ∧ x < 0 ∨ 0 < x ∧ x < 1 / (a + 1) → f a x < f a (x + 1)))
:= sorry

theorem range_of_a (a : ℝ) (h : a ≥ -1) :
  (∃ x1 x2, x1 > 0 ∧ x2 < 0 ∧ f a x1 < f a x2 → -1 ≤ a ∧ a < 0)
:= sorry

end monotonic_decreasing_interval_range_of_a_l1334_133421


namespace actual_average_height_correct_l1334_133401

noncomputable def actual_average_height (n : ℕ) (average_height : ℝ) (wrong_height : ℝ) (actual_height : ℝ) : ℝ :=
  let total_height := average_height * n
  let difference := wrong_height - actual_height
  let correct_total_height := total_height - difference
  correct_total_height / n

theorem actual_average_height_correct :
  actual_average_height 35 184 166 106 = 182.29 :=
by
  sorry

end actual_average_height_correct_l1334_133401


namespace find_k_l1334_133419

theorem find_k (k : ℝ) (h : ∀ x : ℝ, x^2 + 10 * x + k = 0 → (∃ a : ℝ, a > 0 ∧ (x = -3 * a ∨ x = -a))) :
  k = 18.75 :=
sorry

end find_k_l1334_133419


namespace set_swept_by_all_lines_l1334_133455

theorem set_swept_by_all_lines
  (a c x y : ℝ)
  (h1 : 0 < a)
  (h2 : 0 < c)
  (h3 : c < a)
  (h4 : x^2 + y^2 ≤ a^2) : 
  (c^2 - a^2) * x^2 - a^2 * y^2 ≤ (c^2 - a^2) * c^2 :=
sorry

end set_swept_by_all_lines_l1334_133455


namespace find_n_l1334_133407

theorem find_n (n : ℕ) 
  (hM : ∀ M, M = n - 7 → 1 ≤ M)
  (hA : ∀ A, A = n - 2 → 1 ≤ A)
  (hT : ∀ M A, M = n - 7 → A = n - 2 → M + A < n) :
  n = 8 :=
by
  sorry

end find_n_l1334_133407


namespace sum_of_first_five_terms_l1334_133432

theorem sum_of_first_five_terms : 
  ∀ (S : ℕ → ℕ) (a : ℕ → ℕ), 
    (a 1 = 1) ∧ 
    (∀ n ≥ 2, S n = S (n - 1) + n + 2) → 
    S 5 = 23 :=
by
  sorry

end sum_of_first_five_terms_l1334_133432


namespace min_value_expr_l1334_133412

theorem min_value_expr :
  ∀ x y : ℝ, 3 * x^2 + 3 * x * y + y^2 - 3 * x + 3 * y + 9 ≥ 9 :=
by sorry

end min_value_expr_l1334_133412


namespace diversity_values_l1334_133462

theorem diversity_values (k : ℕ) (h : 1 ≤ k ∧ k ≤ 4) :
  ∃ (D : ℕ), D = 1000 * (k - 1) := by
  sorry

end diversity_values_l1334_133462


namespace total_guests_l1334_133445

theorem total_guests (G : ℕ) 
  (hwomen: ∃ n, n = G / 2)
  (hmen: 15 = 15)
  (hchildren: ∃ n, n = G - (G / 2 + 15))
  (men_leaving: ∃ n, n = 1/5 * 15)
  (children_leaving: 4 = 4)
  (people_stayed: 43 = G - ((1/5 * 15) + 4))
  : G = 50 := by
  sorry

end total_guests_l1334_133445


namespace length_more_than_breadth_l1334_133442

theorem length_more_than_breadth (b : ℝ) (x : ℝ) 
  (h1 : b + x = 55) 
  (h2 : 4 * b + 2 * x = 200) 
  (h3 : (5300 : ℝ) / 26.5 = 200)
  : x = 10 := 
by
  sorry

end length_more_than_breadth_l1334_133442


namespace find_y_z_l1334_133487

theorem find_y_z (x y z : ℚ) (h1 : (x + y) / (z - x) = 9 / 2) (h2 : (y + z) / (y - x) = 5) (h3 : x = 43 / 4) :
  y = 12 / 17 + 17 ∧ z = 5 / 68 + 17 := 
by sorry

end find_y_z_l1334_133487


namespace notecard_area_new_dimension_l1334_133406

theorem notecard_area_new_dimension :
  ∀ (length : ℕ) (width : ℕ) (shortened : ℕ),
    length = 7 →
    width = 5 →
    shortened = 2 →
    (width - shortened) * length = 21 →
    (length - shortened) * (width - shortened + shortened) = 25 :=
by
  intros length width shortened h_length h_width h_shortened h_area
  sorry

end notecard_area_new_dimension_l1334_133406


namespace number_of_six_digit_integers_l1334_133478

-- Define the problem conditions
def digits := [1, 1, 3, 3, 7, 8]

-- State the theorem
theorem number_of_six_digit_integers : 
  (List.permutations digits).length = 180 := 
by sorry

end number_of_six_digit_integers_l1334_133478


namespace daily_rate_is_three_l1334_133431

theorem daily_rate_is_three (r : ℝ) : 
  (∀ (initial bedbugs : ℝ), initial = 30 ∧ 
  (∀ days later_bedbugs, days = 4 ∧ later_bedbugs = 810 →
  later_bedbugs = initial * r ^ days)) → r = 3 :=
by
  intros h
  sorry

end daily_rate_is_three_l1334_133431


namespace smallest_N_l1334_133418

theorem smallest_N (N : ℕ) (hN : N > 70) (hdv : 70 ∣ 21 * N) : N = 80 :=
sorry

end smallest_N_l1334_133418


namespace determine_angle_B_l1334_133486

noncomputable def problem_statement (A B C : ℝ) (a b c : ℝ) : Prop :=
  (2 * (Real.cos ((A - B) / 2))^2 * Real.cos B - Real.sin (A - B) * Real.sin B + Real.cos (A + C) = -3 / 5)
  ∧ (a = 8)
  ∧ (b = Real.sqrt 3)

theorem determine_angle_B (A B C : ℝ) (a b c : ℝ)
  (h : problem_statement A B C a b c) : 
  B = Real.arcsin (Real.sqrt 3 / 10) :=
by 
  sorry

end determine_angle_B_l1334_133486


namespace train_speed_l1334_133454

-- Define the conditions
def train_length : ℝ := 50 -- Length of the train in meters
def crossing_time : ℝ := 3 -- Time to cross the pole in seconds

-- Define the speed in meters per second and convert it to km/hr
noncomputable def speed_mps : ℝ := train_length / crossing_time
noncomputable def speed_kmph : ℝ := speed_mps * 3.6 -- Conversion factor

-- Theorem statement: Prove that the calculated speed in km/hr is 60 km/hr
theorem train_speed : speed_kmph = 60 := by
  sorry

end train_speed_l1334_133454


namespace marble_theorem_l1334_133404

noncomputable def marble_problem (M : ℝ) : Prop :=
  let M_Pedro : ℝ := 0.7 * M
  let M_Ebony : ℝ := 0.85 * M_Pedro
  let M_Jimmy : ℝ := 0.7 * M_Ebony
  (M_Jimmy / M) * 100 = 41.65

theorem marble_theorem (M : ℝ) : marble_problem M := 
by
  sorry

end marble_theorem_l1334_133404


namespace complement_union_l1334_133485

def U : Set Int := {-2, -1, 0, 1, 2}

def A : Set Int := {-1, 2}

def B : Set Int := {-1, 0, 1}

theorem complement_union :
  (U \ B) ∪ A = {-2, -1, 2} :=
by
  sorry

end complement_union_l1334_133485


namespace pentagon_area_l1334_133483

noncomputable def area_of_pentagon (a b c d e : ℕ) : ℕ := 
  let area_triangle := (1/2) * a * b
  let area_trapezoid := (1/2) * (c + e) * d
  area_triangle + area_trapezoid

theorem pentagon_area : area_of_pentagon 18 25 30 28 25 = 995 :=
by sorry

end pentagon_area_l1334_133483


namespace bounded_sequence_iff_l1334_133413

theorem bounded_sequence_iff (x : ℕ → ℝ) (h : ∀ n, x (n + 1) = (n^2 + 1) * x n ^ 2 / (x n ^ 3 + n^2)) :
  (∃ C, ∀ n, x n < C) ↔ (0 < x 0 ∧ x 0 ≤ (Real.sqrt 5 - 1) / 2) ∨ x 0 ≥ 1 := sorry

end bounded_sequence_iff_l1334_133413


namespace solve_for_y_l1334_133492

theorem solve_for_y (y : ℤ) : 
  7 * (4 * y + 3) - 3 = -3 * (2 - 9 * y) → y = -24 :=
by
  intro h
  sorry

end solve_for_y_l1334_133492


namespace intercepts_sum_eq_eight_l1334_133468

def parabola_x_y (x y : ℝ) := x = 3 * y^2 - 9 * y + 5

theorem intercepts_sum_eq_eight :
  ∃ (a b c : ℝ), parabola_x_y a 0 ∧ parabola_x_y 0 b ∧ parabola_x_y 0 c ∧ a + b + c = 8 :=
sorry

end intercepts_sum_eq_eight_l1334_133468


namespace each_client_selected_cars_l1334_133484

theorem each_client_selected_cars (cars clients selections : ℕ) (h1 : cars = 16) (h2 : selections = 3 * cars) (h3 : clients = 24) :
  selections / clients = 2 :=
by
  sorry

end each_client_selected_cars_l1334_133484


namespace third_group_members_l1334_133490

theorem third_group_members (total_members first_group second_group : ℕ) (h₁ : total_members = 70) (h₂ : first_group = 25) (h₃ : second_group = 30) : (total_members - (first_group + second_group)) = 15 :=
sorry

end third_group_members_l1334_133490


namespace min_value_of_expression_l1334_133464

theorem min_value_of_expression {x y z : ℝ} 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h : 2 * x * (x + 1 / y + 1 / z) = y * z) : 
  (x + 1 / y) * (x + 1 / z) >= Real.sqrt 2 :=
by
  sorry

end min_value_of_expression_l1334_133464


namespace solution_set_of_quadratic_inequality_l1334_133443

theorem solution_set_of_quadratic_inequality (x : ℝ) : x^2 < x + 6 ↔ -2 < x ∧ x < 3 := 
by
  sorry

end solution_set_of_quadratic_inequality_l1334_133443


namespace number_of_ways_to_choose_one_person_l1334_133430

-- Definitions for the conditions
def people_using_first_method : ℕ := 3
def people_using_second_method : ℕ := 5

-- Definition of the total number of ways to choose one person
def total_ways_to_choose_one_person : ℕ :=
  people_using_first_method + people_using_second_method

-- Statement of the theorem to be proved
theorem number_of_ways_to_choose_one_person :
  total_ways_to_choose_one_person = 8 :=
by 
  sorry

end number_of_ways_to_choose_one_person_l1334_133430


namespace marcy_needs_6_tubs_of_lip_gloss_l1334_133416

theorem marcy_needs_6_tubs_of_lip_gloss (people tubes_per_person tubes_per_tub : ℕ) 
  (h1 : people = 36) (h2 : tubes_per_person = 3) (h3 : tubes_per_tub = 2) :
  (people / tubes_per_person) / tubes_per_tub = 6 :=
by
  -- The proof goes here
  sorry

end marcy_needs_6_tubs_of_lip_gloss_l1334_133416


namespace ratio_of_dad_to_jayson_l1334_133422

-- Define the conditions
def JaysonAge : ℕ := 10
def MomAgeWhenBorn : ℕ := 28
def MomCurrentAge (JaysonAge : ℕ) (MomAgeWhenBorn : ℕ) : ℕ := MomAgeWhenBorn + JaysonAge
def DadCurrentAge (MomCurrentAge : ℕ) : ℕ := MomCurrentAge + 2

-- Define the proof problem
theorem ratio_of_dad_to_jayson (JaysonAge : ℕ) (MomAgeWhenBorn : ℕ)
  (h1 : JaysonAge = 10) (h2 : MomAgeWhenBorn = 28) :
  DadCurrentAge (MomCurrentAge JaysonAge MomAgeWhenBorn) / JaysonAge = 4 :=
by 
  sorry

end ratio_of_dad_to_jayson_l1334_133422


namespace pages_left_l1334_133403

variable (a b : ℕ)

theorem pages_left (a b : ℕ) : a - 8 * b = a - 8 * b :=
by
  sorry

end pages_left_l1334_133403


namespace eq_has_infinite_solutions_l1334_133473

theorem eq_has_infinite_solutions (b : ℝ) (x : ℝ) :
  5 * (3 * x - b) = 3 * (5 * x + 15) → b = -9 := by
sorry

end eq_has_infinite_solutions_l1334_133473


namespace total_number_of_students_l1334_133459

theorem total_number_of_students 
    (T : ℕ)
    (h1 : ∃ a, a = T / 5) 
    (h2 : ∃ b, b = T / 4) 
    (h3 : ∃ c, c = T / 2) 
    (h4 : T - (T / 5 + T / 4 + T / 2) = 25) : 
  T = 500 := by 
  sorry

end total_number_of_students_l1334_133459


namespace relationship_between_3a_3b_4a_l1334_133475

variable (a b : ℝ)
variable (h : a > b)
variable (hb : b > 0)

theorem relationship_between_3a_3b_4a (a b : ℝ) (h : a > b) (hb : b > 0) :
  3 * b < 3 * a ∧ 3 * a < 4 * a := 
by
  sorry

end relationship_between_3a_3b_4a_l1334_133475


namespace mutually_exclusive_event_l1334_133497

theorem mutually_exclusive_event (A B C D: Prop) 
  (h_A: ¬ (A ∧ (¬D)) ∧ ¬ ¬ D)
  (h_B: ¬ (B ∧ (¬D)) ∧ ¬ ¬ D)
  (h_C: ¬ (C ∧ (¬D)) ∧ ¬ ¬ D)
  (h_D: ¬ (D ∧ (¬D)) ∧ ¬ ¬ D) :
  D :=
sorry

end mutually_exclusive_event_l1334_133497


namespace determine_k_linear_l1334_133424

theorem determine_k_linear (k : ℝ) : |k| = 1 ∧ k + 1 ≠ 0 ↔ k = 1 := by
  sorry

end determine_k_linear_l1334_133424


namespace moles_of_KCl_formed_l1334_133498

variables (NaCl KNO3 KCl NaNO3 : Type) 

-- Define the moles of each compound
variables (moles_NaCl moles_KNO3 moles_KCl moles_NaNO3 : ℕ)

-- Initial conditions
axiom initial_NaCl_condition : moles_NaCl = 2
axiom initial_KNO3_condition : moles_KNO3 = 2

-- Reaction definition
axiom reaction : moles_KCl = moles_NaCl

theorem moles_of_KCl_formed :
  moles_KCl = 2 :=
by sorry

end moles_of_KCl_formed_l1334_133498


namespace gcd_18_30_l1334_133488

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l1334_133488


namespace real_roots_of_quadratics_l1334_133495

theorem real_roots_of_quadratics {p1 p2 q1 q2 : ℝ} (h : p1 * p2 = 2 * (q1 + q2)) :
  (∃ x : ℝ, x^2 + p1 * x + q1 = 0) ∨ (∃ x : ℝ, x^2 + p2 * x + q2 = 0) :=
by
  have D1 := p1^2 - 4 * q1
  have D2 := p2^2 - 4 * q2
  sorry

end real_roots_of_quadratics_l1334_133495


namespace find_a_l1334_133448

theorem find_a (a x : ℝ) (h : x = 1) (h_eq : 2 - 3 * (a + x) = 2 * x) : a = -1 := by
  sorry

end find_a_l1334_133448


namespace sum_of_consecutive_odd_integers_l1334_133450

-- Definitions of conditions
def consecutive_odd_integers (a b : ℤ) : Prop :=
  b = a + 2 ∧ (a % 2 = 1) ∧ (b % 2 = 1)

def five_times_smaller_minus_two_condition (a b : ℤ) : Prop :=
  b = 5 * a - 2

-- Theorem statement
theorem sum_of_consecutive_odd_integers (a b : ℤ)
  (h1 : consecutive_odd_integers a b)
  (h2 : five_times_smaller_minus_two_condition a b) : a + b = 4 :=
by
  sorry

end sum_of_consecutive_odd_integers_l1334_133450


namespace additional_telephone_lines_l1334_133405

def telephone_lines_increase : ℕ :=
  let lines_six_digits := 9 * 10^5
  let lines_seven_digits := 9 * 10^6
  lines_seven_digits - lines_six_digits

theorem additional_telephone_lines : telephone_lines_increase = 81 * 10^5 :=
by
  sorry

end additional_telephone_lines_l1334_133405


namespace max_bishops_on_chessboard_l1334_133477

theorem max_bishops_on_chessboard (N : ℕ) (N_pos: 0 < N) : 
  ∃ max_number : ℕ, max_number = 2 * N - 2 :=
sorry

end max_bishops_on_chessboard_l1334_133477


namespace salary_increase_l1334_133476

variable (S : ℝ) (P : ℝ)

theorem salary_increase (h1 : 0.65 * S = 0.5 * S + (P / 100) * (0.5 * S)) : P = 30 := 
by
  -- proof goes here
  sorry

end salary_increase_l1334_133476


namespace buyers_of_cake_mix_l1334_133479

/-
  A certain manufacturer of cake, muffin, and bread mixes has 100 buyers,
  of whom some purchase cake mix, 40 purchase muffin mix, and 17 purchase both cake mix and muffin mix.
  If a buyer is to be selected at random from the 100 buyers, the probability that the buyer selected will be one who purchases 
  neither cake mix nor muffin mix is 0.27.
  Prove that the number of buyers who purchase cake mix is 50.
-/

theorem buyers_of_cake_mix (C M B total : ℕ) (hM : M = 40) (hB : B = 17) (hTotal : total = 100)
    (hProb : (total - (C + M - B) : ℝ) / total = 0.27) : C = 50 :=
by
  -- Definition of the proof is required here
  sorry

end buyers_of_cake_mix_l1334_133479


namespace sum_fractions_correct_l1334_133494

def sum_of_fractions : Prop :=
  (3 / 15 + 5 / 150 + 7 / 1500 + 9 / 15000 = 0.2386)

theorem sum_fractions_correct : sum_of_fractions :=
by
  sorry

end sum_fractions_correct_l1334_133494


namespace ellipse_eccentricity_proof_l1334_133463

theorem ellipse_eccentricity_proof (a b c : ℝ) 
  (ha_gt_hb : a > b) (hb_gt_zero : b > 0) (hc_gt_zero : c > 0)
  (h_ellipse : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1)
  (h_r : ∃ r : ℝ, r = (Real.sqrt 2 / 6) * c) :
  (Real.sqrt (1 - b^2 / a^2)) = (2 * Real.sqrt 5 / 5) := by {
  sorry
}

end ellipse_eccentricity_proof_l1334_133463


namespace polygon_perimeter_l1334_133449

theorem polygon_perimeter :
  let AB := 2
  let BC := 2
  let CD := 2
  let DE := 2
  let EF := 2
  let FG := 3
  let GH := 3
  let HI := 3
  let IJ := 3
  let JA := 4
  AB + BC + CD + DE + EF + FG + GH + HI + IJ + JA = 26 :=
by {
  sorry
}

end polygon_perimeter_l1334_133449


namespace min_positive_value_l1334_133472

theorem min_positive_value (c d : ℤ) (h : c > d) : 
  ∃ x : ℝ, x = (c + 2 * d) / (c - d) + (c - d) / (c + 2 * d) ∧ x = 2 :=
by {
  sorry
}

end min_positive_value_l1334_133472


namespace number_of_student_tickets_sold_l1334_133460

variable (A S : ℝ)

theorem number_of_student_tickets_sold
  (h1 : A + S = 59)
  (h2 : 4 * A + 2.5 * S = 222.50) :
  S = 9 :=
by sorry

end number_of_student_tickets_sold_l1334_133460


namespace father_ate_oranges_l1334_133444

theorem father_ate_oranges (initial_oranges : ℝ) (remaining_oranges : ℝ) (eaten_oranges : ℝ) : 
  initial_oranges = 77.0 → remaining_oranges = 75 → eaten_oranges = initial_oranges - remaining_oranges → eaten_oranges = 2.0 :=
by
  intros h1 h2 h3
  sorry

end father_ate_oranges_l1334_133444


namespace least_possible_integer_l1334_133480

theorem least_possible_integer (N : ℕ) :
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 30 ∧ n ≠ 28 ∧ n ≠ 29 → n ∣ N) ∧
  (∀ m : ℕ, (∀ n : ℕ, 1 ≤ n ∧ n ≤ 30 ∧ n ≠ 28 ∧ n ≠ 29 → n ∣ m) → N ≤ m) →
  N = 2329089562800 :=
sorry

end least_possible_integer_l1334_133480


namespace arithmetic_sequence_sum_l1334_133439

theorem arithmetic_sequence_sum (S : ℕ → ℤ) (a_1 d : ℤ) 
  (h1: S 3 = (3 * a_1) + (3 * (2 * d) / 2))
  (h2: S 7 = (7 * a_1) + (7 * (6 * d) / 2)) :
  S 5 = (5 * a_1) + (5 * (4 * d) / 2) := by
  sorry

end arithmetic_sequence_sum_l1334_133439


namespace value_of_expression_l1334_133433

theorem value_of_expression (x : ℝ) (hx : x = -2) : (3 * x + 4) ^ 2 = 4 :=
by
  sorry

end value_of_expression_l1334_133433


namespace remainder_when_divided_by_385_l1334_133429

theorem remainder_when_divided_by_385 (x : ℤ)
  (h1 : 2 + x ≡ 4 [ZMOD 125])
  (h2 : 3 + x ≡ 9 [ZMOD 343])
  (h3 : 4 + x ≡ 25 [ZMOD 1331]) :
  x ≡ 307 [ZMOD 385] :=
sorry

end remainder_when_divided_by_385_l1334_133429


namespace circles_intersect_l1334_133499

noncomputable def circle1 := {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}
noncomputable def circle2 := {p : ℝ × ℝ | (p.1 - 2)^2 + p.2^2 = 9}

theorem circles_intersect :
  ∃ p : ℝ × ℝ, p ∈ circle1 ∧ p ∈ circle2 :=
sorry

end circles_intersect_l1334_133499


namespace a_eq_zero_iff_purely_imaginary_l1334_133414

open Complex

noncomputable def purely_imaginary (z : ℂ) : Prop :=
  z.re = 0

theorem a_eq_zero_iff_purely_imaginary (a b : ℝ) :
  (a = 0) ↔ purely_imaginary (a + b * Complex.I) :=
by
  sorry

end a_eq_zero_iff_purely_imaginary_l1334_133414


namespace find_a_if_line_passes_through_center_l1334_133458

-- Define the given circle equation
def circle_eqn (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y = 0

-- Define the given line equation
def line_eqn (x y a : ℝ) : Prop := 3*x + y + a = 0

-- The coordinates of the center of the circle
def center_of_circle : (ℝ × ℝ) := (-1, 2)

-- Prove that a = 1 if the line passes through the center of the circle
theorem find_a_if_line_passes_through_center (a : ℝ) :
  line_eqn (-1) 2 a → a = 1 :=
by
  sorry

end find_a_if_line_passes_through_center_l1334_133458


namespace number_of_B_students_l1334_133461

theorem number_of_B_students (x : ℝ) (h1 : 0.8 * x + x + 1.2 * x = 40) : x = 13 :=
  sorry

end number_of_B_students_l1334_133461


namespace unique_solution_l1334_133489

theorem unique_solution (x : ℝ) : (3 : ℝ)^x + (4 : ℝ)^x + (5 : ℝ)^x = (6 : ℝ)^x ↔ x = 3 := by
  sorry

end unique_solution_l1334_133489


namespace expression_1_expression_2_expression_3_expression_4_l1334_133456

section problem1

variable {x : ℝ}

theorem expression_1:
  (x^2 - 1 + x)*(x^2 - 1 + 3*x) + x^2  = x^4 + 4*x^3 + 4*x^2 - 4*x - 1 :=
sorry

end problem1

section problem2

variable {x a : ℝ}

theorem expression_2:
  (x - a)^4 + 4*a^4 = (x^2 + a^2)*(x^2 - 4*a*x + 5*a^2) :=
sorry

end problem2

section problem3

variable {a : ℝ}

theorem expression_3:
  (a + 1)^4 + 2*(a + 1)^3 + a*(a + 2) = (a + 1)^4 + 2*(a + 1)^3 + 1 :=
sorry

end problem3

section problem4

variable {p : ℝ}

theorem expression_4:
  (p + 2)^4 + 2*(p^2 - 4)^2 + (p - 2)^4 = 4*p^4 :=
sorry

end problem4

end expression_1_expression_2_expression_3_expression_4_l1334_133456


namespace hitting_probability_l1334_133452

theorem hitting_probability (A_hit B_hit : ℚ) (hA : A_hit = 4/5) (hB : B_hit = 5/6) :
  1 - ((1 - A_hit) * (1 - B_hit)) = 29/30 :=
by 
  sorry

end hitting_probability_l1334_133452


namespace ms_emily_inheritance_l1334_133496

theorem ms_emily_inheritance :
  ∃ (y : ℝ), 
    (0.25 * y + 0.15 * (y - 0.25 * y) = 19500) ∧
    (y = 53800) :=
by
  sorry

end ms_emily_inheritance_l1334_133496


namespace tom_and_jerry_drank_80_ounces_l1334_133402

theorem tom_and_jerry_drank_80_ounces
    (T J : ℝ) 
    (initial_T : T = 40)
    (initial_J : J = 2 * T)
    (T_drank J_drank : ℝ)
    (T_remaining J_remaining : ℝ)
    (T_after_pour J_after_pour : ℝ)
    (T_final J_final : ℝ)
    (H1 : T_drank = (2 / 3) * T)
    (H2 : J_drank = (2 / 3) * J)
    (H3 : T_remaining = T - T_drank)
    (H4 : J_remaining = J - J_drank)
    (H5 : T_after_pour = T_remaining + (1 / 4) * J_remaining)
    (H6 : J_after_pour = J_remaining - (1 / 4) * J_remaining)
    (H7 : T_final = T_after_pour - 5)
    (H8 : J_final = J_after_pour + 5)
    (H9 : T_final = J_final + 4)
    : T_drank + J_drank = 80 :=
by
  sorry

end tom_and_jerry_drank_80_ounces_l1334_133402


namespace find_a3_l1334_133469

variable {α : Type*} [AddCommGroup α] [Module ℤ α]

noncomputable
def arithmetic_sequence (a : ℕ → α) (d : α) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

theorem find_a3 {a : ℕ → ℤ} (d : ℤ) (h6 : a 6 = 6) (h9 : a 9 = 9) :
  (∃ d : ℤ, arithmetic_sequence a d) →
  a 3 = 3 :=
by
  intro h_arith_seq
  sorry

end find_a3_l1334_133469


namespace frank_oranges_correct_l1334_133426

def betty_oranges : ℕ := 12
def sandra_oranges : ℕ := 3 * betty_oranges
def emily_oranges : ℕ := 7 * sandra_oranges
def frank_oranges : ℕ := 5 * emily_oranges

theorem frank_oranges_correct : frank_oranges = 1260 := by
  sorry

end frank_oranges_correct_l1334_133426


namespace solve_equation_l1334_133438

theorem solve_equation : ∀ (x : ℝ), (2 * x + 5 = 3 * x - 2) → (x = 7) :=
by
  intro x
  intro h
  sorry

end solve_equation_l1334_133438


namespace two_pow_n_minus_one_divisible_by_seven_l1334_133427

theorem two_pow_n_minus_one_divisible_by_seven (n : ℕ) : (2^n - 1) % 7 = 0 ↔ ∃ k : ℕ, n = 3 * k := 
sorry

end two_pow_n_minus_one_divisible_by_seven_l1334_133427


namespace find_c_plus_d_l1334_133411

noncomputable def f (c d : ℝ) (x : ℝ) : ℝ :=
  if x < 3 then 2 * c * x + d else 9 - 2 * x

theorem find_c_plus_d (c d : ℝ) (h : ∀ x : ℝ, f c d (f c d x) = x) : c + d = 4.25 :=
by
  sorry

end find_c_plus_d_l1334_133411


namespace balloon_arrangement_count_l1334_133491

theorem balloon_arrangement_count :
  let total_permutations := (Nat.factorial 7) / (Nat.factorial 2 * Nat.factorial 3)
  let ways_to_arrange_L_and_O := Nat.choose 4 1 * (Nat.factorial 3)
  let valid_arrangements := ways_to_arrange_L_and_O * total_permutations
  valid_arrangements = 10080 :=
by
  sorry

end balloon_arrangement_count_l1334_133491


namespace sign_of_slope_equals_sign_of_correlation_l1334_133451

-- Definitions for conditions
def linear_relationship (x y : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), ∀ t, y t = a + b * x t

def correlation_coefficient (x y : ℝ → ℝ) (r : ℝ) : Prop :=
  r > -1 ∧ r < 1 ∧ ∀ t t', (y t - y t').sign = (x t - x t').sign

def regression_line_slope (b : ℝ) : Prop := True

-- Theorem to prove the sign of b is equal to the sign of r
theorem sign_of_slope_equals_sign_of_correlation (x y : ℝ → ℝ) (r b : ℝ) 
  (h1 : linear_relationship x y) 
  (h2 : correlation_coefficient x y r) 
  (h3 : regression_line_slope b) : 
  b.sign = r.sign := 
sorry

end sign_of_slope_equals_sign_of_correlation_l1334_133451


namespace volume_increase_factor_l1334_133423

   variable (π : ℝ) (r h : ℝ)

   def original_volume : ℝ := π * r^2 * h

   def new_height : ℝ := 3 * h

   def new_radius : ℝ := 2.5 * r

   def new_volume : ℝ := π * (new_radius r)^2 * (new_height h)

   theorem volume_increase_factor :
     new_volume π r h = 18.75 * original_volume π r h := 
   by
     sorry
   
end volume_increase_factor_l1334_133423


namespace initial_items_in_cart_l1334_133400

theorem initial_items_in_cart (deleted_items : ℕ) (items_left : ℕ) (initial_items : ℕ) 
  (h1 : deleted_items = 10) (h2 : items_left = 8) : initial_items = 18 :=
by 
  -- Proof goes here
  sorry

end initial_items_in_cart_l1334_133400


namespace count_integers_satisfy_inequality_l1334_133435

theorem count_integers_satisfy_inequality : 
  ∃ l : List Int, (∀ n ∈ l, (n - 3) * (n + 5) < 0) ∧ l.length = 7 :=
by
  sorry

end count_integers_satisfy_inequality_l1334_133435


namespace find_a4_l1334_133446

noncomputable def geometric_sequence (a_n : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a_n (n + 1) = a_n n * q

theorem find_a4 (a_n : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a_n q →
  a_n 1 + a_n 2 = -1 →
  a_n 1 - a_n 3 = -3 →
  a_n 4 = -8 :=
by 
  sorry

end find_a4_l1334_133446


namespace average_speed_of_trip_l1334_133440

theorem average_speed_of_trip (d1 d2 s1 s2 : ℕ)
  (h1 : d1 = 30) (h2 : d2 = 30)
  (h3 : s1 = 60) (h4 : s2 = 30) :
  (d1 + d2) / (d1 / s1 + d2 / s2) = 40 :=
by sorry

end average_speed_of_trip_l1334_133440


namespace arithmetic_mean_of_fractions_l1334_133482

theorem arithmetic_mean_of_fractions :
  (3/8 + 5/9) / 2 = 67 / 144 :=
by
  sorry

end arithmetic_mean_of_fractions_l1334_133482


namespace keanu_total_spending_l1334_133408

-- Definitions based on conditions
def dog_fish : Nat := 40
def cat_fish : Nat := dog_fish / 2
def total_fish : Nat := dog_fish + cat_fish
def cost_per_fish : Nat := 4
def total_cost : Nat := total_fish * cost_per_fish

-- Theorem statement
theorem keanu_total_spending : total_cost = 240 :=
by 
    sorry

end keanu_total_spending_l1334_133408


namespace sum_of_fourth_powers_l1334_133453

theorem sum_of_fourth_powers (n : ℤ) (h : (n - 2)^2 + n^2 + (n + 2)^2 = 2450) :
  (n - 2)^4 + n^4 + (n + 2)^4 = 1881632 :=
sorry

end sum_of_fourth_powers_l1334_133453


namespace tennis_tournament_possible_l1334_133417

theorem tennis_tournament_possible (p : ℕ) : 
  (∀ i j : ℕ, i ≠ j → ∃ a b c d : ℕ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  i = a ∨ i = b ∨ i = c ∨ i = d ∧ j = a ∨ j = b ∨ j = c ∨ j = d) → 
  ∃ k : ℕ, p = 8 * k + 1 := by
  sorry

end tennis_tournament_possible_l1334_133417


namespace smaller_integer_is_49_l1334_133493

theorem smaller_integer_is_49 (m n : ℕ) (hm : 10 ≤ m ∧ m < 100) (hn : 10 ≤ n ∧ n < 100)
  (h : (m + n) / 2 = m + n / 100) : min m n = 49 :=
by
  sorry

end smaller_integer_is_49_l1334_133493


namespace percent_of_a_is_b_l1334_133434

theorem percent_of_a_is_b (a b c : ℝ) (h1 : c = 0.30 * a) (h2 : c = 0.25 * b) : b = 1.2 * a :=
by
  -- proof 
  sorry

end percent_of_a_is_b_l1334_133434


namespace cube_dimension_ratio_l1334_133457

theorem cube_dimension_ratio (V1 V2 : ℕ) (h1 : V1 = 27) (h2 : V2 = 216) :
  ∃ r : ℕ, r = 2 ∧ (∃ l1 l2 : ℕ, l1 * l1 * l1 = V1 ∧ l2 * l2 * l2 = V2 ∧ l2 = r * l1) :=
by
  sorry

end cube_dimension_ratio_l1334_133457


namespace icosahedron_to_octahedron_l1334_133470

theorem icosahedron_to_octahedron : 
  ∃ (f : Finset (Fin 20)), f.card = 8 ∧ 
  (∀ {o : Finset (Fin 8)}, (True ∧ True)) ∧
  (∃ n : ℕ, n = 5) := by
  sorry

end icosahedron_to_octahedron_l1334_133470


namespace polynomial_coeff_divisible_by_5_l1334_133437

theorem polynomial_coeff_divisible_by_5 (a b c d : ℤ) 
  (h : ∀ (x : ℤ), (a * x^3 + b * x^2 + c * x + d) % 5 = 0) : 
  a % 5 = 0 ∧ b % 5 = 0 ∧ c % 5 = 0 ∧ d % 5 = 0 := 
by
  sorry

end polynomial_coeff_divisible_by_5_l1334_133437


namespace arithmetic_progression_sum_l1334_133471

theorem arithmetic_progression_sum (a d : ℝ) (n : ℕ) : 
  a + 10 * d = 5.25 → 
  a + 6 * d = 3.25 → 
  (n : ℝ) / 2 * (2 * a + (n - 1) * d) = 56.25 → 
  n = 15 :=
by
  intros h1 h2 h3
  sorry

end arithmetic_progression_sum_l1334_133471


namespace total_notebooks_distributed_l1334_133466

/-- Define the parameters for children in Class A and Class B and the conditions given. -/
def ClassAChildren : ℕ := 64
def ClassBChildren : ℕ := 13

/-- Define the conditions as per the problem -/
def notebooksPerChildInClassA (A : ℕ) : ℕ := A / 8
def notebooksPerChildInClassB (A : ℕ) : ℕ := 2 * A
def totalChildrenClasses (A B : ℕ) : ℕ := A + B
def totalChildrenCondition (A : ℕ) : ℕ := 6 * A / 5

/-- Theorem to state the number of notebooks distributed between the two classes -/
theorem total_notebooks_distributed (A : ℕ) (B : ℕ) (H : A = 64) (H1 : B = 13) : 
  (A * (A / 8) + B * (2 * A)) = 2176 := by
  -- Conditions from the problem
  have conditionA : A = 64 := H
  have conditionB : B = 13 := H1
  have classA_notebooks : ℕ := (notebooksPerChildInClassA A) * A
  have classB_notebooks : ℕ := (notebooksPerChildInClassB A) * B
  have total_notebooks : ℕ := classA_notebooks + classB_notebooks
  -- Proof that total notebooks equals 2176
  sorry

end total_notebooks_distributed_l1334_133466


namespace zoe_has_47_nickels_l1334_133415

theorem zoe_has_47_nickels (x : ℕ) 
  (h1 : 5 * x + 10 * x + 50 * x = 3050) : 
  x = 47 := 
sorry

end zoe_has_47_nickels_l1334_133415


namespace geometric_series_common_ratio_l1334_133441

theorem geometric_series_common_ratio (a S r : ℝ) (ha : a = 400) (hS : S = 2500) (hS_eq : S = a / (1 - r)) : r = 21 / 25 :=
by
  rw [ha, hS] at hS_eq
  -- This statement follows from algebraic manipulation outlined in the solution steps.
  sorry

end geometric_series_common_ratio_l1334_133441


namespace total_pencils_l1334_133447

def initial_pencils : ℕ := 9
def additional_pencils : ℕ := 56

theorem total_pencils : initial_pencils + additional_pencils = 65 :=
by
  -- proof steps are not required, so we use sorry
  sorry

end total_pencils_l1334_133447


namespace boats_left_l1334_133467

def initial_boats : ℕ := 30
def percentage_eaten_by_fish : ℕ := 20
def boats_shot_with_arrows : ℕ := 2
def boats_blown_by_wind : ℕ := 3
def boats_sank : ℕ := 4

def boats_eaten_by_fish : ℕ := (initial_boats * percentage_eaten_by_fish) / 100

theorem boats_left : initial_boats - boats_eaten_by_fish - boats_shot_with_arrows - boats_blown_by_wind - boats_sank = 15 := by
  sorry

end boats_left_l1334_133467


namespace solution_set_of_inequality_l1334_133436

theorem solution_set_of_inequality (x : ℝ) : -x^2 + 2*x + 3 > 0 ↔ (-1 < x ∧ x < 3) :=
sorry

end solution_set_of_inequality_l1334_133436


namespace negation_of_universal_is_existential_l1334_133420

theorem negation_of_universal_is_existential :
  ¬ (∀ x : ℝ, x^2 - 2 * x + 4 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2 * x + 4 > 0) :=
by
  sorry

end negation_of_universal_is_existential_l1334_133420


namespace calculate_rent_is_correct_l1334_133428

noncomputable def requiredMonthlyRent 
  (purchase_cost : ℝ) 
  (monthly_set_aside_percent : ℝ)
  (annual_property_tax : ℝ)
  (annual_insurance : ℝ)
  (annual_return_percent : ℝ) : ℝ :=
  let annual_return := annual_return_percent * purchase_cost
  let total_yearly_expenses := annual_return + annual_property_tax + annual_insurance
  let monthly_expenses := total_yearly_expenses / 12
  let retention_rate := 1 - monthly_set_aside_percent
  monthly_expenses / retention_rate

theorem calculate_rent_is_correct 
  (purchase_cost : ℝ := 200000)
  (monthly_set_aside_percent : ℝ := 0.2)
  (annual_property_tax : ℝ := 5000)
  (annual_insurance : ℝ := 2400)
  (annual_return_percent : ℝ := 0.08) :
  requiredMonthlyRent purchase_cost monthly_set_aside_percent annual_property_tax annual_insurance annual_return_percent = 2437.50 :=
by
  sorry

end calculate_rent_is_correct_l1334_133428


namespace at_least_one_gt_one_l1334_133409

variable (a b : ℝ)

theorem at_least_one_gt_one (h : a + b > 2) : a > 1 ∨ b > 1 :=
by
  sorry

end at_least_one_gt_one_l1334_133409


namespace alcohol_quantity_l1334_133425

theorem alcohol_quantity (A W : ℝ) (h1 : A / W = 2 / 5) (h2 : A / (W + 10) = 2 / 7) : A = 10 :=
by
  sorry

end alcohol_quantity_l1334_133425


namespace mirror_side_length_l1334_133410

theorem mirror_side_length (width length : ℝ) (area_wall : ℝ) (area_mirror : ℝ) (side_length : ℝ) 
  (h1 : width = 28) 
  (h2 : length = 31.5) 
  (h3 : area_wall = width * length)
  (h4 : area_mirror = area_wall / 2) 
  (h5 : area_mirror = side_length ^ 2) : 
  side_length = 21 := 
by 
  sorry

end mirror_side_length_l1334_133410
