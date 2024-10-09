import Mathlib

namespace simplify_expression_l2405_240584

theorem simplify_expression :
  let a := 2
  let b := -3
  10 * a^2 * b - (2 * a * b^2 - 2 * (a * b - 5 * a^2 * b)) = -48 := sorry

end simplify_expression_l2405_240584


namespace ages_of_people_l2405_240557

-- Define types
variable (A M B C : ℕ)

-- Define conditions as hypotheses
def conditions : Prop :=
  A = 2 * M ∧
  A = 4 * B ∧
  M = A - 10 ∧
  C = B + 3 ∧
  C = M / 2

-- Define what we want to prove
theorem ages_of_people :
  (conditions A M B C) →
  A = 20 ∧
  M = 10 ∧
  B = 2 ∧
  C = 5 :=
by
  sorry

end ages_of_people_l2405_240557


namespace flower_counts_l2405_240579

theorem flower_counts (R G Y : ℕ) : (R + G = 62) → (R + Y = 49) → (G + Y = 77) → R = 17 ∧ G = 45 ∧ Y = 32 :=
by
  intros h1 h2 h3
  sorry

end flower_counts_l2405_240579


namespace cone_cannot_have_rectangular_projection_l2405_240510

def orthographic_projection (solid : Type) : Type := sorry

theorem cone_cannot_have_rectangular_projection :
  (∀ (solid : Type), orthographic_projection solid = Rectangle → solid ≠ Cone) :=
sorry

end cone_cannot_have_rectangular_projection_l2405_240510


namespace three_digit_numbers_distinct_base_l2405_240514

theorem three_digit_numbers_distinct_base (b : ℕ) (h : (b - 1) ^ 2 * (b - 2) = 250) : b = 8 :=
sorry

end three_digit_numbers_distinct_base_l2405_240514


namespace proposition_p_proposition_q_l2405_240591

theorem proposition_p : ∅ ≠ ({∅} : Set (Set Empty)) := by
  sorry

theorem proposition_q (A : Set ℕ) (B : Set (Set ℕ)) (hA : A = {1, 2})
    (hB : B = {x | x ⊆ A}) : A ∈ B := by
  sorry

end proposition_p_proposition_q_l2405_240591


namespace min_xyz_value_l2405_240558

theorem min_xyz_value (x y z : ℝ) (h1 : x + y + z = 1) (h2 : z = 2 * y) (h3 : y ≤ (1 / 3)) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  (∀ a b c : ℝ, (a + b + c = 1) → (c = 2 * b) → (b ≤ (1 / 3)) → 0 < a → 0 < b → 0 < c → (a * b * c) ≥ (x * y * z) → (a * b * c) = (8 / 243)) :=
by sorry

end min_xyz_value_l2405_240558


namespace other_root_is_minus_5_l2405_240573

-- conditions
def polynomial (x : ℝ) := x^4 - x^3 - 18 * x^2 + 52 * x + (-40 : ℝ)
def r1 := 2
def f_of_r1_eq_zero : polynomial r1 = 0 := by sorry -- given condition

-- the proof problem
theorem other_root_is_minus_5 : ∃ r, polynomial r = 0 ∧ r ≠ r1 ∧ r = -5 :=
by
  sorry

end other_root_is_minus_5_l2405_240573


namespace breadth_of_landscape_l2405_240554

noncomputable def landscape_breadth (L : ℕ) (playground_area : ℕ) (total_area : ℕ) (B : ℕ) : Prop :=
  B = 6 * L ∧ playground_area = 4200 ∧ playground_area = (1 / 7) * total_area ∧ total_area = L * B

theorem breadth_of_landscape : ∃ (B : ℕ), ∀ (L : ℕ), landscape_breadth L 4200 29400 B → B = 420 :=
by
  intros
  sorry

end breadth_of_landscape_l2405_240554


namespace baskets_delivered_l2405_240518

theorem baskets_delivered 
  (peaches_per_basket : ℕ := 25)
  (boxes : ℕ := 8)
  (peaches_per_box : ℕ := 15)
  (peaches_eaten : ℕ := 5)
  (peaches_in_boxes := boxes * peaches_per_box) 
  (total_peaches := peaches_in_boxes + peaches_eaten) : 
  total_peaches / peaches_per_basket = 5 :=
by
  sorry

end baskets_delivered_l2405_240518


namespace find_x_minus_y_l2405_240519

theorem find_x_minus_y (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 16) : x - y = 2 :=
by
  have h3 : x^2 - y^2 = (x + y) * (x - y) := by sorry
  have h4 : (x + y) * (x - y) = 8 * (x - y) := by sorry
  have h5 : 16 = 8 * (x - y) := by sorry
  have h6 : 16 = 8 * (x - y) := by sorry
  have h7 : x - y = 2 := by sorry
  exact h7

end find_x_minus_y_l2405_240519


namespace cocoa_powder_total_l2405_240569

variable (already_has : ℕ) (still_needs : ℕ)

theorem cocoa_powder_total (h₁ : already_has = 259) (h₂ : still_needs = 47) : already_has + still_needs = 306 :=
by
  sorry

end cocoa_powder_total_l2405_240569


namespace range_f_1_range_m_l2405_240545

noncomputable def f (x : ℝ) : ℝ := (Real.log x / Real.log 2 - 2) * (Real.log x / (2 * Real.log 2) - 1/2)

theorem range_f_1 (x : ℝ) (h1 : 2 ≤ x) (h2 : x ≤ 4) : 
  -1/8 ≤ f x ∧ f x ≤ 0 :=
sorry

theorem range_m (m : ℝ) (x : ℝ) (h1 : 4 ≤ x) (h2 : x ≤ 16) (h3 : f x ≥ m * Real.log x / Real.log 2) :
  m ≤ 0 :=
sorry

end range_f_1_range_m_l2405_240545


namespace intersection_M_N_l2405_240564

def M : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }
def N : Set ℝ := { y | ∃ x, y = x^2 ∧ -1 ≤ x ∧ x ≤ 1 }

theorem intersection_M_N :
  M ∩ N = { z | 0 ≤ z ∧ z ≤ 1 } := by
  sorry

end intersection_M_N_l2405_240564


namespace cos_angle_of_vectors_l2405_240503

variables (a b : EuclideanSpace ℝ (Fin 2))

theorem cos_angle_of_vectors (h1 : ‖a‖ = 2) (h2 : ‖b‖ = 1) (h3 : ‖a - b‖ = 2) :
  (inner a b) / (‖a‖ * ‖b‖) = 1/4 :=
by
  sorry

end cos_angle_of_vectors_l2405_240503


namespace smallest_n_exists_l2405_240506

theorem smallest_n_exists (n : ℤ) (r : ℝ) : 
  (∃ m : ℤ, m = (↑n + r) ^ 3 ∧ r > 0 ∧ r < 1 / 1000) ∧ n > 0 → n = 19 := 
by sorry

end smallest_n_exists_l2405_240506


namespace hallie_made_100_per_painting_l2405_240520

-- Define conditions
def num_paintings : ℕ := 3
def total_money_made : ℕ := 300

-- Define the goal
def money_per_painting : ℕ := total_money_made / num_paintings

theorem hallie_made_100_per_painting :
  money_per_painting = 100 :=
sorry

end hallie_made_100_per_painting_l2405_240520


namespace tom_and_eva_children_count_l2405_240505

theorem tom_and_eva_children_count (karen_donald_children : ℕ)
  (total_legs_in_pool : ℕ) (people_not_in_pool : ℕ) 
  (total_legs_each_person : ℕ) (karen_donald : ℕ) (tom_eva : ℕ) 
  (total_people_in_pool : ℕ) (total_people : ℕ) :
  karen_donald_children = 6 ∧ total_legs_in_pool = 16 ∧ people_not_in_pool = 6 ∧ total_legs_each_person = 2 ∧
  karen_donald = 2 ∧ tom_eva = 2 ∧ total_people_in_pool = total_legs_in_pool / total_legs_each_person ∧ 
  total_people = total_people_in_pool + people_not_in_pool ∧ 
  total_people - (karen_donald + karen_donald_children + tom_eva) = 4 :=
by
  intros
  sorry

end tom_and_eva_children_count_l2405_240505


namespace find_lengths_of_DE_and_HJ_l2405_240587

noncomputable def lengths_consecutive_segments (BD DE EF FG GH HJ : ℝ) (BC : ℝ) : Prop :=
  BD = 5 ∧ EF = 11 ∧ FG = 7 ∧ GH = 3 ∧ BC = 29 ∧ BD + DE + EF + FG + GH + HJ = BC ∧ DE = HJ

theorem find_lengths_of_DE_and_HJ (x : ℝ) : lengths_consecutive_segments 5 x 11 7 3 x 29 → x = 1.5 :=
by
  intros h
  sorry

end find_lengths_of_DE_and_HJ_l2405_240587


namespace a100_gt_two_pow_99_l2405_240513

theorem a100_gt_two_pow_99 (a : ℕ → ℤ) (h_pos : ∀ n, 0 < a n) 
  (h1 : a 1 > a 0) (h_rec : ∀ n ≥ 2, a n = 3 * a (n - 1) - 2 * a (n - 2)) :
  a 100 > 2 ^ 99 :=
sorry

end a100_gt_two_pow_99_l2405_240513


namespace road_repair_equation_l2405_240594

variable (x : ℝ) 

-- Original problem conditions
def total_road_length := 150
def extra_repair_per_day := 5
def days_ahead := 5

-- The proof problem to show that the schedule differential equals 5 days ahead
theorem road_repair_equation :
  (total_road_length / x) - (total_road_length / (x + extra_repair_per_day)) = days_ahead :=
sorry

end road_repair_equation_l2405_240594


namespace problem1_problem2_l2405_240537

noncomputable def f (x a b : ℝ) : ℝ := (x + a) / (x + b)

-- Problem (1): Prove the inequality f(x-1) > 0 given b = 1.
theorem problem1 (a x : ℝ) : f (x - 1) a 1 > 0 := sorry

-- Problem (2): Prove the values of a and b such that the range of f(x) for x ∈ [-1, 2] is [5/4, 2].
theorem problem2 (a b : ℝ) (H₁ : f (-1) a b = 5 / 4) (H₂ : f 2 a b = 2) :
    (a = 3 ∧ b = 2) ∨ (a = -4 ∧ b = -3) := sorry

end problem1_problem2_l2405_240537


namespace count_4_digit_numbers_with_property_l2405_240509

noncomputable def count_valid_4_digit_numbers : ℕ :=
  let valid_units (t : ℕ) : List ℕ := List.filter (λ u => u ≥ 3 * t) [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  let choices_for_tu : ℕ := (List.length (valid_units 0)) + (List.length (valid_units 1)) + (List.length (valid_units 2))
  choices_for_tu * 9 * 9

theorem count_4_digit_numbers_with_property : count_valid_4_digit_numbers = 1701 := by
  sorry

end count_4_digit_numbers_with_property_l2405_240509


namespace no_distributive_laws_hold_l2405_240526

def tripledAfterAdding (a b : ℝ) : ℝ := 3 * (a + b)

theorem no_distributive_laws_hold (x y z : ℝ) :
  ¬ (tripledAfterAdding x (y + z) = tripledAfterAdding (tripledAfterAdding x y) (tripledAfterAdding x z)) ∧
  ¬ (x + (tripledAfterAdding y z) = tripledAfterAdding (x + y) (x + z)) ∧
  ¬ (tripledAfterAdding x (tripledAfterAdding y z) = tripledAfterAdding (tripledAfterAdding x y) (tripledAfterAdding x z)) :=
by sorry

end no_distributive_laws_hold_l2405_240526


namespace minimum_value_of_fraction_l2405_240571

theorem minimum_value_of_fraction (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) : 
  (4 / a + 9 / b) ≥ 25 :=
by
  sorry

end minimum_value_of_fraction_l2405_240571


namespace speed_of_mans_train_is_80_kmph_l2405_240548

-- Define the given constants
def length_goods_train : ℤ := 280 -- length in meters
def time_to_pass : ℤ := 9 -- time in seconds
def speed_goods_train : ℤ := 32 -- speed in km/h

-- Define the conversion factor from km/h to m/s
def kmh_to_ms (v : ℤ) : ℤ := v * 1000 / 3600

-- Define the speed of the goods train in m/s
def speed_goods_train_ms := kmh_to_ms speed_goods_train

-- Define the speed of the man's train in km/h
def speed_mans_train : ℤ := 80

-- Prove that the speed of the man's train is 80 km/h given the conditions
theorem speed_of_mans_train_is_80_kmph :
  ∃ V : ℤ,
    (V + speed_goods_train) * 1000 / 3600 = length_goods_train / time_to_pass → 
    V = speed_mans_train :=
by
  sorry

end speed_of_mans_train_is_80_kmph_l2405_240548


namespace mean_of_six_numbers_l2405_240512

theorem mean_of_six_numbers (sum_of_six: ℚ) (H: sum_of_six = 3 / 4) : sum_of_six / 6 = 1 / 8 := by
  sorry

end mean_of_six_numbers_l2405_240512


namespace fraction_diff_l2405_240597

open Real

theorem fraction_diff (x y : ℝ) (hx : x = sqrt 5 - 1) (hy : y = sqrt 5 + 1) :
  (1 / x - 1 / y) = 1 / 2 := sorry

end fraction_diff_l2405_240597


namespace fraction_identity_l2405_240542

theorem fraction_identity :
  (1721^2 - 1714^2 : ℚ) / (1728^2 - 1707^2) = 1 / 3 :=
by
  sorry

end fraction_identity_l2405_240542


namespace division_by_fraction_l2405_240521

theorem division_by_fraction :
  (5 / (8 / 15) : ℚ) = 75 / 8 :=
by
  sorry

end division_by_fraction_l2405_240521


namespace mike_baseball_cards_l2405_240593

theorem mike_baseball_cards :
  let InitialCards : ℕ := 87
  let BoughtCards : ℕ := 13
  (InitialCards - BoughtCards = 74)
:= by
  sorry

end mike_baseball_cards_l2405_240593


namespace xy_condition_l2405_240528

theorem xy_condition : (∀ x y : ℝ, x^2 + y^2 = 0 → xy = 0) ∧ ¬ (∀ x y : ℝ, xy = 0 → x^2 + y^2 = 0) := 
by
  sorry

end xy_condition_l2405_240528


namespace train_speed_correct_l2405_240540

-- Definitions for the given conditions
def train_length : ℝ := 320
def time_to_cross : ℝ := 6

-- The speed of the train
def train_speed : ℝ := 53.33

-- The proof statement
theorem train_speed_correct : train_speed = train_length / time_to_cross :=
by
  sorry

end train_speed_correct_l2405_240540


namespace minimum_questions_to_identify_white_ball_l2405_240561

theorem minimum_questions_to_identify_white_ball (n : ℕ) (even_white : ℕ) 
  (h₁ : n = 2004) 
  (h₂ : even_white % 2 = 0) 
  (h₃ : 1 ≤ even_white ∧ even_white ≤ n) :
  ∃ m : ℕ, m = 2003 := 
sorry

end minimum_questions_to_identify_white_ball_l2405_240561


namespace salary_increase_l2405_240529

theorem salary_increase (x : ℝ) (y : ℝ) :
  (1000 : ℝ) * 80 + 50 = y → y - (50 + 80 * x) = 80 :=
by
  intros h
  sorry

end salary_increase_l2405_240529


namespace exists_group_of_four_l2405_240530

-- Define the given conditions
variables (students : Finset ℕ) (h_size : students.card = 21)
variables (done_homework : Finset ℕ → Prop)
variables (hw_unique : ∀ (s : Finset ℕ), s.card = 3 → done_homework s)

-- Define the theorem with the assertion to be proved
theorem exists_group_of_four (students : Finset ℕ) (h_size : students.card = 21)
  (done_homework : Finset ℕ → Prop)
  (hw_unique : ∀ s, s.card = 3 → done_homework s) :
  ∃ (grp : Finset ℕ), grp.card = 4 ∧ 
    (∀ (s : Finset ℕ), s ⊆ grp ∧ s.card = 3 → done_homework s) :=
sorry

end exists_group_of_four_l2405_240530


namespace steven_set_aside_9_grapes_l2405_240531

-- Define the conditions based on the problem statement
def total_seeds_needed : ℕ := 60
def average_seeds_per_apple : ℕ := 6
def average_seeds_per_pear : ℕ := 2
def average_seeds_per_grape : ℕ := 3
def apples_set_aside : ℕ := 4
def pears_set_aside : ℕ := 3
def additional_seeds_needed : ℕ := 3

-- Calculate the number of seeds from apples and pears
def seeds_from_apples : ℕ := apples_set_aside * average_seeds_per_apple
def seeds_from_pears : ℕ := pears_set_aside * average_seeds_per_pear

-- Calculate the number of seeds that Steven already has from apples and pears
def seeds_from_apples_and_pears : ℕ := seeds_from_apples + seeds_from_pears

-- Calculate the remaining seeds needed from grapes
def seeds_needed_from_grapes : ℕ := total_seeds_needed - seeds_from_apples_and_pears - additional_seeds_needed

-- Calculate the number of grapes set aside
def grapes_set_aside : ℕ := seeds_needed_from_grapes / average_seeds_per_grape

theorem steven_set_aside_9_grapes : grapes_set_aside = 9 :=
by 
  sorry

end steven_set_aside_9_grapes_l2405_240531


namespace square_root_unique_l2405_240596

theorem square_root_unique (x : ℝ) (h1 : x + 3 ≥ 0) (h2 : 2 * x - 6 ≥ 0)
  (h : (x + 3)^2 = (2 * x - 6)^2) :
  x = 1 ∧ (x + 3)^2 = 16 := 
by
  sorry

end square_root_unique_l2405_240596


namespace ratio_WX_XY_l2405_240552

theorem ratio_WX_XY (p q : ℝ) (h : 3 * p = 4 * q) : (4 * q) / (3 * p) = 12 / 7 := by
  sorry

end ratio_WX_XY_l2405_240552


namespace sufficient_condition_for_parallel_l2405_240543

-- Definitions for lines and planes
variables {Line Plane : Type}

-- Definitions of parallelism and perpendicularity
variable {Parallel Perpendicular : Line → Plane → Prop}
variable {ParallelLines : Line → Line → Prop}

-- Definition of subset relation
variable {Subset : Line → Plane → Prop}

-- Theorems or conditions
variables (a b : Line) (α β : Plane)

-- Assertion of the theorem
theorem sufficient_condition_for_parallel (h1 : ParallelLines a b) (h2 : Parallel b α) (h3 : ¬ Subset a α) : Parallel a α :=
sorry

end sufficient_condition_for_parallel_l2405_240543


namespace value_of_collection_l2405_240582

theorem value_of_collection (n : ℕ) (v : ℕ → ℕ) (h1 : n = 20) 
    (h2 : v 5 = 20) (h3 : ∀ k1 k2, v k1 = v k2) : v n = 80 :=
by
  sorry

end value_of_collection_l2405_240582


namespace find_range_of_a_l2405_240562

variable {a : ℝ}
variable {x : ℝ}

theorem find_range_of_a (h₁ : x ∈ Set.Ioo (-2:ℝ) (-1:ℝ)) :
  ∃ a, a ∈ Set.Icc (1:ℝ) (2:ℝ) ∧ (x + 1)^2 < Real.log (|x|) / Real.log a :=
by
  sorry

end find_range_of_a_l2405_240562


namespace length_of_GH_l2405_240534

theorem length_of_GH (AB CD GH : ℤ) (h_parallel : AB = 240 ∧ CD = 160 ∧ (AB + CD) = GH*2) : GH = 320 / 3 :=
by sorry

end length_of_GH_l2405_240534


namespace solve_for_A_l2405_240580

theorem solve_for_A (A : ℚ) : 80 - (5 - (6 + A * (7 - 8 - 5))) = 89 → A = -4/3 :=
by
  sorry

end solve_for_A_l2405_240580


namespace smallest_possible_obscured_number_l2405_240568

theorem smallest_possible_obscured_number (a b : ℕ) (cond : 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9) :
  2 * a = b - 9 →
  42 + 25 + 56 + 10 * a + b = 4 * (4 + 2 + 2 + 5 + 5 + 6 + a + b) →
  10 * a + b = 79 :=
sorry

end smallest_possible_obscured_number_l2405_240568


namespace pie_chart_degrees_for_cherry_pie_l2405_240546

theorem pie_chart_degrees_for_cherry_pie :
  ∀ (total_students chocolate_pie apple_pie blueberry_pie : ℕ)
    (remaining_students cherry_pie_students lemon_pie_students : ℕ),
    total_students = 40 →
    chocolate_pie = 15 →
    apple_pie = 10 →
    blueberry_pie = 7 →
    remaining_students = total_students - chocolate_pie - apple_pie - blueberry_pie →
    cherry_pie_students = remaining_students / 2 →
    lemon_pie_students = remaining_students / 2 →
    (cherry_pie_students : ℝ) / (total_students : ℝ) * 360 = 36 :=
by
  sorry

end pie_chart_degrees_for_cherry_pie_l2405_240546


namespace two_and_four_digit_singular_numbers_six_digit_singular_number_exists_twenty_digit_singular_number_at_most_ten_singular_numbers_with_100_digits_exists_thirty_digit_singular_number_l2405_240599

def is_singular_number (n : ℕ) (num : ℕ) : Prop :=
  let first_n_digits := num / 10^n;
  let last_n_digits := num % 10^n;
  (num > 0) ∧
  (first_n_digits > 0) ∧
  (last_n_digits > 0) ∧
  (first_n_digits < 10^n) ∧
  (last_n_digits < 10^n) ∧
  (num = first_n_digits * 10^n + last_n_digits) ∧
  (∃ k, num = k^2) ∧
  (∃ k, first_n_digits = k^2) ∧
  (∃ k, last_n_digits = k^2)

-- (1) Prove that 49 is a two-digit singular number and 1681 is a four-digit singular number
theorem two_and_four_digit_singular_numbers :
  is_singular_number 1 49 ∧ is_singular_number 2 1681 :=
sorry

-- (2) Prove that 256036 is a six-digit singular number
theorem six_digit_singular_number :
  is_singular_number 3 256036 :=
sorry

-- (3) Prove the existence of a 20-digit singular number
theorem exists_twenty_digit_singular_number :
  ∃ num, is_singular_number 10 num :=
sorry

-- (4) Prove that there are at most 10 singular numbers with 100 digits
theorem at_most_ten_singular_numbers_with_100_digits :
  ∃! n, n <= 10 ∧ ∀ num, num < 10^100 → is_singular_number 50 num → num < 10 ∧ num > 0 :=
sorry

-- (5) Prove the existence of a 30-digit singular number
theorem exists_thirty_digit_singular_number :
  ∃ num, is_singular_number 15 num :=
sorry

end two_and_four_digit_singular_numbers_six_digit_singular_number_exists_twenty_digit_singular_number_at_most_ten_singular_numbers_with_100_digits_exists_thirty_digit_singular_number_l2405_240599


namespace p_q_r_cubic_sum_l2405_240563

theorem p_q_r_cubic_sum (p q r : ℚ) (h1 : p + q + r = 4) (h2 : p * q + p * r + q * r = 6) (h3 : p * q * r = -8) : 
  p^3 + q^3 + r^3 = 8 := by
  sorry

end p_q_r_cubic_sum_l2405_240563


namespace problem_statement_l2405_240583

theorem problem_statement (x : ℝ) (h : 8 * x - 6 = 10) : 200 * (1 / x) = 100 := by
  sorry

end problem_statement_l2405_240583


namespace nat_digit_problem_l2405_240504

theorem nat_digit_problem :
  ∀ n : Nat, (n % 10 = (2016 * (n / 2016)) % 10) → (n = 4032 ∨ n = 8064 ∨ n = 12096 ∨ n = 16128) :=
by
  sorry

end nat_digit_problem_l2405_240504


namespace intersection_of_sets_l2405_240511

-- Definitions from the conditions.
def A := { x : ℝ | x^2 - 2 * x ≤ 0 }
def B := { x : ℝ | x > 1 }

-- The proof problem statement.
theorem intersection_of_sets :
  A ∩ B = { x : ℝ | 1 < x ∧ x ≤ 2 } :=
sorry

end intersection_of_sets_l2405_240511


namespace fred_initial_cards_l2405_240567

variables {n : ℕ}

theorem fred_initial_cards (h : n - 22 = 18) : n = 40 :=
by {
  sorry
}

end fred_initial_cards_l2405_240567


namespace remainder_of_max_6_multiple_no_repeated_digits_l2405_240547

theorem remainder_of_max_6_multiple_no_repeated_digits (M : ℕ) 
  (hM : ∃ n, M = 6 * n) 
  (h_unique_digits : ∀ (d : ℕ), d ∈ (M.digits 10) → (M.digits 10).count d = 1) 
  (h_max_M : ∀ (k : ℕ), (∃ n, k = 6 * n) ∧ (∀ (d : ℕ), d ∈ (k.digits 10) → (k.digits 10).count d = 1) → k ≤ M) :
  M % 100 = 78 := 
sorry

end remainder_of_max_6_multiple_no_repeated_digits_l2405_240547


namespace repeating_decimal_sum_l2405_240522

-- Definitions from conditions
def repeating_decimal_1_3 : ℚ := 1 / 3
def repeating_decimal_2_99 : ℚ := 2 / 99

-- Statement to prove
theorem repeating_decimal_sum : repeating_decimal_1_3 + repeating_decimal_2_99 = 35 / 99 :=
by sorry

end repeating_decimal_sum_l2405_240522


namespace johns_overall_average_speed_l2405_240585

open Real

noncomputable def johns_average_speed (scooter_time_min : ℝ) (scooter_speed_mph : ℝ) 
    (jogging_time_min : ℝ) (jogging_speed_mph : ℝ) : ℝ :=
  let scooter_time_hr := scooter_time_min / 60
  let jogging_time_hr := jogging_time_min / 60
  let distance_scooter := scooter_speed_mph * scooter_time_hr
  let distance_jogging := jogging_speed_mph * jogging_time_hr
  let total_distance := distance_scooter + distance_jogging
  let total_time := scooter_time_hr + jogging_time_hr
  total_distance / total_time

theorem johns_overall_average_speed :
  johns_average_speed 40 20 60 6 = 11.6 :=
by
  sorry

end johns_overall_average_speed_l2405_240585


namespace fraction_of_yard_occupied_by_flower_beds_l2405_240500

theorem fraction_of_yard_occupied_by_flower_beds :
  let leg_length := (36 - 26) / 3
  let triangle_area := (1 / 2) * leg_length^2
  let total_flower_bed_area := 3 * triangle_area
  let yard_area := 36 * 6
  (total_flower_bed_area / yard_area) = 25 / 324
  := by
  let leg_length := (36 - 26) / 3
  let triangle_area := (1 / 2) * leg_length^2
  let total_flower_bed_area := 3 * triangle_area
  let yard_area := 36 * 6
  have h1 : leg_length = 10 / 3 := by sorry
  have h2 : triangle_area = (1 / 2) * (10 / 3)^2 := by sorry
  have h3 : total_flower_bed_area = 3 * ((1 / 2) * (10 / 3)^2) := by sorry
  have h4 : yard_area = 216 := by sorry
  have h5 : total_flower_bed_area / yard_area = 25 / 324 := by sorry
  exact h5

end fraction_of_yard_occupied_by_flower_beds_l2405_240500


namespace ratio_Nicolai_to_Charliz_l2405_240590

-- Definitions based on conditions
def Haylee_guppies := 3 * 12
def Jose_guppies := Haylee_guppies / 2
def Charliz_guppies := Jose_guppies / 3
def Total_guppies := 84
def Nicolai_guppies := Total_guppies - (Haylee_guppies + Jose_guppies + Charliz_guppies)

-- Proof statement
theorem ratio_Nicolai_to_Charliz : Nicolai_guppies / Charliz_guppies = 4 := by
  sorry

end ratio_Nicolai_to_Charliz_l2405_240590


namespace simplify_fraction_l2405_240576

theorem simplify_fraction (n : ℕ) (h : 2 ^ n ≠ 0) : 
  (2 ^ (n + 5) - 3 * 2 ^ n) / (3 * 2 ^ (n + 4)) = 29 / 48 := 
by
  sorry

end simplify_fraction_l2405_240576


namespace shirt_to_pants_ratio_l2405_240535

noncomputable def cost_uniforms
  (pants_cost shirt_ratio socks_price total_spending : ℕ) : Prop :=
  ∃ (shirt_cost tie_cost : ℕ),
    shirt_cost = shirt_ratio * pants_cost ∧
    tie_cost = shirt_cost / 5 ∧
    5 * (pants_cost + shirt_cost + tie_cost + socks_price) = total_spending

theorem shirt_to_pants_ratio 
  (pants_cost socks_price total_spending : ℕ)
  (h1 : pants_cost = 20)
  (h2 : socks_price = 3)
  (h3 : total_spending = 355)
  (shirt_ratio : ℕ)
  (h4 : cost_uniforms pants_cost shirt_ratio socks_price total_spending) :
  shirt_ratio = 2 := by
  sorry

end shirt_to_pants_ratio_l2405_240535


namespace bus_stop_time_per_hour_l2405_240536

theorem bus_stop_time_per_hour 
  (speed_without_stoppages : ℝ)
  (speed_with_stoppages : ℝ)
  (h1 : speed_without_stoppages = 64)
  (h2 : speed_with_stoppages = 48) : 
  ∃ t : ℝ, t = 15 := 
by
  sorry

end bus_stop_time_per_hour_l2405_240536


namespace distance_between_closest_points_l2405_240517

noncomputable def distance_closest_points :=
  let center1 : ℝ × ℝ := (5, 3)
  let center2 : ℝ × ℝ := (20, 7)
  let radius1 := center1.2  -- radius of first circle is y-coordinate of its center
  let radius2 := center2.2  -- radius of second circle is y-coordinate of its center
  let distance_centers := Real.sqrt ((center2.1 - center1.1)^2 + (center2.2 - center1.2)^2)
  distance_centers - radius1 - radius2

theorem distance_between_closest_points :
  distance_closest_points = Real.sqrt 241 - 10 :=
sorry

end distance_between_closest_points_l2405_240517


namespace stock_increase_l2405_240508

theorem stock_increase (x : ℝ) (h₁ : x > 0) :
  (1.25 * (0.85 * x) - x) / x * 100 = 6.25 :=
by 
  -- {proof steps would go here}
  sorry

end stock_increase_l2405_240508


namespace triangle_inequality_condition_l2405_240592

variable (a b c : ℝ)
variable (α : ℝ) -- angle in radians

-- Define the condition where c must be less than a + b
theorem triangle_inequality_condition : c < a + b := by
  sorry

end triangle_inequality_condition_l2405_240592


namespace length_MN_proof_l2405_240502

-- Declare a noncomputable section to avoid computational requirements
noncomputable section

-- Define the quadrilateral ABCD with given sides
structure Quadrilateral :=
  (BC AD AB CD : ℕ)
  (BC_AD_parallel : Prop)

-- Define a theorem to calculate the length MN
theorem length_MN_proof (ABCD : Quadrilateral) 
  (M N : ℝ) (BisectorsIntersect_M : Prop) (BisectorsIntersect_N : Prop) : 
  ABCD.BC = 26 → ABCD.AD = 5 → ABCD.AB = 10 → ABCD.CD = 17 → 
  (MN = 2 ↔ (BC + AD - AB - CD) / 2 = 2) :=
by
  sorry

end length_MN_proof_l2405_240502


namespace remainder_of_expression_l2405_240544

theorem remainder_of_expression :
  (8 * 7^19 + 1^19) % 9 = 3 :=
  by
    sorry

end remainder_of_expression_l2405_240544


namespace smallest_n_interval_l2405_240525

theorem smallest_n_interval :
  ∃ n : ℕ, (∃ x : ℤ, ⌊10 ^ n / x⌋ = 2006) ∧ 7 ≤ n ∧ n ≤ 12 :=
sorry

end smallest_n_interval_l2405_240525


namespace decimal_representation_of_fraction_l2405_240556

theorem decimal_representation_of_fraction :
  (47 : ℝ) / (2^3 * 5^4) = 0.0094 :=
by
  sorry

end decimal_representation_of_fraction_l2405_240556


namespace calculate_expression_l2405_240541

theorem calculate_expression :
  15^2 + 2 * 15 * 5 + 5^2 + 5^3 = 525 := 
sorry

end calculate_expression_l2405_240541


namespace parallelogram_area_l2405_240581

-- Define a plane rectangular coordinate system
structure PlaneRectangularCoordinateSystem :=
(axis : ℝ)

-- Define the properties of a square
structure Square :=
(side_length : ℝ)

-- Define the properties of a parallelogram in a perspective drawing
structure Parallelogram :=
(side_length: ℝ)

-- Define the conditions of the problem
def problem_conditions (s : Square) (p : Parallelogram) :=
  s.side_length = 4 ∨ s.side_length = 8 ∧ 
  p.side_length = 4

-- Statement of the problem
theorem parallelogram_area (s : Square) (p : Parallelogram)
  (h : problem_conditions s p) :
  p.side_length * p.side_length = 16 ∨ p.side_length * p.side_length = 64 :=
by {
  sorry
}

end parallelogram_area_l2405_240581


namespace model1_best_fitting_effect_l2405_240524

-- Definitions for the correlation coefficients of the models
def R1 : ℝ := 0.98
def R2 : ℝ := 0.80
def R3 : ℝ := 0.50
def R4 : ℝ := 0.25

-- Main theorem stating Model 1 has the best fitting effect
theorem model1_best_fitting_effect : |R1| > |R2| ∧ |R1| > |R3| ∧ |R1| > |R4| :=
by sorry

end model1_best_fitting_effect_l2405_240524


namespace wall_length_is_7_5_meters_l2405_240572

noncomputable def brick_volume : ℚ := 25 * 11.25 * 6

noncomputable def total_brick_volume : ℚ := 6000 * brick_volume

noncomputable def wall_cross_section : ℚ := 600 * 22.5

noncomputable def wall_length (total_volume : ℚ) (cross_section : ℚ) : ℚ := total_volume / cross_section

theorem wall_length_is_7_5_meters :
  wall_length total_brick_volume wall_cross_section = 7.5 := by
sorry

end wall_length_is_7_5_meters_l2405_240572


namespace rate_of_A_is_8_l2405_240560

noncomputable def rate_of_A (a b : ℕ) : ℕ :=
  if b = a + 4 ∧ 48 * b = 72 * a then a else 0

theorem rate_of_A_is_8 {a b : ℕ} 
  (h1 : b = a + 4)
  (h2 : 48 * b = 72 * a) : 
  rate_of_A a b = 8 :=
by
  -- proof steps can be added here
  sorry

end rate_of_A_is_8_l2405_240560


namespace approx_num_fish_in_pond_l2405_240577

noncomputable def numFishInPond (tagged_in_second: ℕ) (total_second: ℕ) (tagged: ℕ) : ℕ :=
  tagged * total_second / tagged_in_second

theorem approx_num_fish_in_pond :
  numFishInPond 2 50 50 = 1250 := by
  sorry

end approx_num_fish_in_pond_l2405_240577


namespace calculate_value_l2405_240549

theorem calculate_value :
  (1 / 3) * 9 * (1 / 27) * 81 * (1 / 243) * 729 * (1 / 2187) * 6561 * (1 / 19683) * 59049 = 243 :=
by
  sorry

end calculate_value_l2405_240549


namespace problem_solution_l2405_240588

theorem problem_solution (a b c d : ℝ) (h1 : a = 5 * b) (h2 : b = 3 * c) (h3 : c = 6 * d) :
  (a + b * c) / (c + d * b) = (3 * (5 + 6 * d)) / (1 + 3 * d) :=
by
  sorry

end problem_solution_l2405_240588


namespace max_n_value_l2405_240589

noncomputable def f (x : ℝ) : ℝ := Real.sin (3 * x)

theorem max_n_value (m : ℝ) (x_i : ℕ → ℝ) (n : ℕ) (h1 : ∀ i, i < n → f (x_i i) / (x_i i) = m)
  (h2 : ∀ i, i < n → -2 * Real.pi ≤ x_i i ∧ x_i i ≤ 2 * Real.pi) :
  n ≤ 12 :=
sorry

end max_n_value_l2405_240589


namespace total_cost_verification_l2405_240570

def sandwich_cost : ℝ := 2.45
def soda_cost : ℝ := 0.87
def num_sandwiches : ℕ := 2
def num_sodas : ℕ := 4
def total_cost : ℝ := 8.38

theorem total_cost_verification 
  (sc : sandwich_cost = 2.45)
  (sd : soda_cost = 0.87)
  (ns : num_sandwiches = 2)
  (nd : num_sodas = 4) :
  num_sandwiches * sandwich_cost + num_sodas * soda_cost = total_cost := 
sorry

end total_cost_verification_l2405_240570


namespace books_withdrawn_is_15_l2405_240565

-- Define the initial condition
def initial_books : ℕ := 250

-- Define the books taken out on Tuesday
def books_taken_out_tuesday : ℕ := 120

-- Define the books returned on Wednesday
def books_returned_wednesday : ℕ := 35

-- Define the books left in library on Thursday
def books_left_thursday : ℕ := 150

-- Define the problem: Determine the number of books withdrawn on Thursday
def books_withdrawn_thursday : ℕ :=
  (initial_books - books_taken_out_tuesday + books_returned_wednesday) - books_left_thursday

-- The statement we want to prove
theorem books_withdrawn_is_15 : books_withdrawn_thursday = 15 := by sorry

end books_withdrawn_is_15_l2405_240565


namespace graph_not_in_first_quadrant_l2405_240507

theorem graph_not_in_first_quadrant (a b : ℝ) (h_a_pos : a > 0) (h_a_ne_one : a ≠ 1) 
  (h_not_in_first_quadrant : ∀ x : ℝ, a^x + b - 1 ≤ 0) : 
  0 < a ∧ a < 1 ∧ b ≤ 0 :=
sorry

end graph_not_in_first_quadrant_l2405_240507


namespace range_of_a_l2405_240533

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, 7 * x1^2 - (a + 13) * x1 + a^2 - a - 2 = 0 ∧
                 7 * x2^2 - (a + 13) * x2 + a^2 - a - 2 = 0 ∧
                 0 < x1 ∧ x1 < 1 ∧ 1 < x2 ∧ x2 < 2) →
  (-2 < a ∧ a < -1) ∨ (3 < a ∧ a < 4) :=
by
  intro h
  sorry

end range_of_a_l2405_240533


namespace helga_shoes_l2405_240527

theorem helga_shoes (x : ℕ) : 
  (x + (x + 2) + 0 + 2 * (x + (x + 2) + 0) = 48) → x = 7 := 
by
  sorry

end helga_shoes_l2405_240527


namespace original_number_of_motorcycles_l2405_240578

theorem original_number_of_motorcycles (x y : ℕ) 
  (h1 : x + 2 * y = 42) 
  (h2 : x > y) 
  (h3 : 2 * (x - 3) + 4 * y = 3 * (x + y - 3)) : x = 16 := 
sorry

end original_number_of_motorcycles_l2405_240578


namespace distance_Owlford_Highcastle_l2405_240539

open Complex

theorem distance_Owlford_Highcastle :
  let Highcastle := (0 : ℂ)
  let Owlford := (900 + 1200 * I : ℂ)
  dist Highcastle Owlford = 1500 := by
  sorry

end distance_Owlford_Highcastle_l2405_240539


namespace stuart_segments_return_l2405_240551

theorem stuart_segments_return (r1 r2 : ℝ) (tangent_chord : ℝ)
  (angle_ABC : ℝ) (h1 : r1 < r2) (h2 : tangent_chord = r1 * 2)
  (h3 : angle_ABC = 75) :
  ∃ (n : ℕ), n = 24 ∧ tangent_chord * n = 360 * (n / 24) :=
by {
  sorry
}

end stuart_segments_return_l2405_240551


namespace find_integer_for_perfect_square_l2405_240532

theorem find_integer_for_perfect_square :
  ∃ (n : ℤ), ∃ (m : ℤ), n^2 + 20 * n + 11 = m^2 ∧ n = 35 := by
  sorry

end find_integer_for_perfect_square_l2405_240532


namespace cos_double_angle_sum_l2405_240550

theorem cos_double_angle_sum
  (α β : ℝ)
  (h1 : Real.sin (α - β) = 1 / 3)
  (h2 : Real.cos α * Real.sin β = 1 / 6) :
  Real.cos (2 * α + 2 * β) = 1 / 9 := by
  sorry

end cos_double_angle_sum_l2405_240550


namespace points_meet_every_720_seconds_l2405_240516

theorem points_meet_every_720_seconds
    (v1 v2 : ℝ) 
    (h1 : v1 - v2 = 1/720) 
    (h2 : (1/v2) - (1/v1) = 10) :
    v1 = 1/80 ∧ v2 = 1/90 :=
by
  sorry

end points_meet_every_720_seconds_l2405_240516


namespace time_for_A_to_complete_work_l2405_240555

theorem time_for_A_to_complete_work (W : ℝ) (A B C : ℝ) (W_pos : 0 < W) (B_work : B = W / 40) (C_work : C = W / 20) : 
  (10 * (W / A) + 10 * (W / B) + 10 * (W / C) = W) → A = W / 40 :=
by 
  sorry

end time_for_A_to_complete_work_l2405_240555


namespace percent_less_than_l2405_240559

-- Definitions based on the given conditions.
variable (y q w z : ℝ)
variable (h1 : w = 0.60 * q)
variable (h2 : q = 0.60 * y)
variable (h3 : z = 1.50 * w)

-- The theorem that the percentage by which z is less than y is 46%.
theorem percent_less_than (y q w z : ℝ) (h1 : w = 0.60 * q) (h2 : q = 0.60 * y) (h3 : z = 1.50 * w) :
  100 - (z / y * 100) = 46 :=
sorry

end percent_less_than_l2405_240559


namespace find_a_plus_d_l2405_240566

theorem find_a_plus_d (a b c d : ℝ) (h1 : a + b = 5) (h2 : b + c = 6) (h3 : c + d = 3) : a + d = -1 := 
by 
  -- omit proof
  sorry

end find_a_plus_d_l2405_240566


namespace total_balls_l2405_240553

theorem total_balls (r b g : ℕ) (ratio : r = 2 * k ∧ b = 4 * k ∧ g = 6 * k) (green_balls : g = 36) : r + b + g = 72 :=
by
  sorry

end total_balls_l2405_240553


namespace full_seasons_already_aired_l2405_240595

variable (days_until_premiere : ℕ)
variable (episodes_per_day : ℕ)
variable (episodes_per_season : ℕ)

theorem full_seasons_already_aired (h_days : days_until_premiere = 10)
                                  (h_episodes_day : episodes_per_day = 6)
                                  (h_episodes_season : episodes_per_season = 15) :
  (days_until_premiere * episodes_per_day) / episodes_per_season = 4 := by
  sorry

end full_seasons_already_aired_l2405_240595


namespace kat_boxing_trainings_per_week_l2405_240598

noncomputable def strength_training_hours_per_week : ℕ := 3
noncomputable def boxing_training_hours (x : ℕ) : ℚ := 1.5 * x
noncomputable def total_training_hours : ℕ := 9

theorem kat_boxing_trainings_per_week (x : ℕ) (h : total_training_hours = strength_training_hours_per_week + boxing_training_hours x) : x = 4 :=
by
  sorry

end kat_boxing_trainings_per_week_l2405_240598


namespace Olly_needs_24_shoes_l2405_240538

def dogs := 3
def cats := 2
def ferrets := 1
def paws_per_dog := 4
def paws_per_cat := 4
def paws_per_ferret := 4

theorem Olly_needs_24_shoes : (dogs * paws_per_dog) + (cats * paws_per_cat) + (ferrets * paws_per_ferret) = 24 :=
by
  sorry

end Olly_needs_24_shoes_l2405_240538


namespace neg_ten_plus_three_l2405_240501

theorem neg_ten_plus_three :
  -10 + 3 = -7 := by
  sorry

end neg_ten_plus_three_l2405_240501


namespace divide_L_shaped_plaque_into_four_equal_parts_l2405_240523

-- Definition of an "L"-shaped plaque and the condition of symmetric cuts
def L_shaped_plaque (a b : ℕ) : Prop := (a > 0) ∧ (b > 0)

-- Statement of the proof problem
theorem divide_L_shaped_plaque_into_four_equal_parts (a b : ℕ) (h : L_shaped_plaque a b) :
  ∃ (p1 p2 : ℕ → ℕ → Prop),
    (∀ x y, p1 x y ↔ (x < a/2 ∧ y < b/2)) ∧
    (∀ x y, p2 x y ↔ (x < a/2 ∧ y >= b/2) ∨ (x >= a/2 ∧ y < b/2) ∨ (x >= a/2 ∧ y >= b/2)) :=
sorry

end divide_L_shaped_plaque_into_four_equal_parts_l2405_240523


namespace plaza_area_increase_l2405_240575

theorem plaza_area_increase (a : ℝ) : 
  ((a + 2)^2 - a^2 = 4 * a + 4) :=
sorry

end plaza_area_increase_l2405_240575


namespace gcd_max_value_l2405_240515

theorem gcd_max_value (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1008) : 
  ∃ d, d = Nat.gcd a b ∧ d = 504 :=
by
  sorry

end gcd_max_value_l2405_240515


namespace third_vertex_l2405_240586

/-- Two vertices of a right triangle are located at (4, 3) and (0, 0).
The third vertex of the triangle lies on the positive branch of the x-axis.
Determine the coordinates of the third vertex if the area of the triangle is 24 square units. -/
theorem third_vertex (x : ℝ) (h : x > 0) : 
  (1 / 2 * |x| * 3 = 24) → (x, 0) = (16, 0) :=
by
  intro h_area
  sorry

end third_vertex_l2405_240586


namespace car_total_distance_l2405_240574

noncomputable def distance_first_segment (speed1 : ℝ) (time1 : ℝ) : ℝ :=
  speed1 * time1

noncomputable def distance_second_segment (speed2 : ℝ) (time2 : ℝ) : ℝ :=
  speed2 * time2

noncomputable def distance_final_segment (speed3 : ℝ) (time3 : ℝ) : ℝ :=
  speed3 * time3

noncomputable def total_distance (d1 d2 d3 : ℝ) : ℝ :=
  d1 + d2 + d3

theorem car_total_distance :
  let d1 := distance_first_segment 65 2
  let d2 := distance_second_segment 80 1.5
  let d3 := distance_final_segment 50 2
  total_distance d1 d2 d3 = 350 :=
by
  sorry

end car_total_distance_l2405_240574
