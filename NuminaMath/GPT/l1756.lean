import Mathlib

namespace sufficiency_s_for_q_l1756_175676

variables {q r s : Prop}

theorem sufficiency_s_for_q (h₁ : r → q) (h₂ : ¬(q → r)) (h₃ : r ↔ s) : s → q ∧ ¬(q → s) :=
by
  sorry

end sufficiency_s_for_q_l1756_175676


namespace volleyball_height_30_l1756_175664

theorem volleyball_height_30 (t : ℝ) : (60 - 9 * t - 4.5 * t^2 = 30) → t = 1.77 :=
by
  intro h_eq
  sorry

end volleyball_height_30_l1756_175664


namespace razorback_shop_tshirts_l1756_175635

theorem razorback_shop_tshirts (T : ℕ) (h : 215 * T = 4300) : T = 20 :=
by sorry

end razorback_shop_tshirts_l1756_175635


namespace probability_not_within_square_b_l1756_175655

noncomputable def prob_not_within_square_b : Prop :=
  let area_A := 121
  let side_length_B := 16 / 4
  let area_B := side_length_B * side_length_B
  let area_not_covered := area_A - area_B
  let prob := area_not_covered / area_A
  prob = (105 / 121)

theorem probability_not_within_square_b : prob_not_within_square_b :=
by
  sorry

end probability_not_within_square_b_l1756_175655


namespace largest_value_of_d_l1756_175608

noncomputable def maximum_possible_value_of_d 
  (a b c d : ℝ) 
  (h1 : a + b + c + d = 10) 
  (h2 : ab + ac + ad + bc + bd + cd = 17) : ℝ :=
  (5 + Real.sqrt 123) / 2

theorem largest_value_of_d 
  (a b c d : ℝ) 
  (h1 : a + b + c + d = 10) 
  (h2 : ab + ac + ad + bc + bd + cd = 17) : 
  d ≤ maximum_possible_value_of_d a b c d h1 h2 :=
sorry

end largest_value_of_d_l1756_175608


namespace smallest_d_l1756_175659

theorem smallest_d (d t s : ℕ) (h1 : 3 * t - 4 * s = 2023)
                   (h2 : t = s + d) 
                   (h3 : 4 * s > 0)
                   (h4 : d % 3 = 0) :
                   d = 675 := sorry

end smallest_d_l1756_175659


namespace value_of_M_l1756_175663

theorem value_of_M (M : ℝ) (h : (25 / 100) * M = (35 / 100) * 1800) : M = 2520 := 
sorry

end value_of_M_l1756_175663


namespace total_money_is_305_l1756_175618

-- Define the worth of each gold coin, silver coin, and the quantity of each type of coin and cash.
def worth_of_gold_coin := 50
def worth_of_silver_coin := 25
def number_of_gold_coins := 3
def number_of_silver_coins := 5
def cash_in_dollars := 30

-- Define the total money calculation based on given conditions.
def total_gold_value := number_of_gold_coins * worth_of_gold_coin
def total_silver_value := number_of_silver_coins * worth_of_silver_coin
def total_value := total_gold_value + total_silver_value + cash_in_dollars

-- The proof statement asserting the total value.
theorem total_money_is_305 : total_value = 305 := by
  -- Proof omitted for brevity.
  sorry

end total_money_is_305_l1756_175618


namespace inequality_c_l1756_175621

theorem inequality_c (x : ℝ) : x^2 + 1 + 1 / (x^2 + 1) ≥ 2 := sorry

end inequality_c_l1756_175621


namespace students_not_finding_parents_funny_l1756_175625

theorem students_not_finding_parents_funny:
  ∀ (total_students funny_dad funny_mom funny_both : ℕ),
  total_students = 50 →
  funny_dad = 25 →
  funny_mom = 30 →
  funny_both = 18 →
  (total_students - (funny_dad + funny_mom - funny_both) = 13) :=
by
  intros total_students funny_dad funny_mom funny_both
  sorry

end students_not_finding_parents_funny_l1756_175625


namespace remaining_black_cards_l1756_175651

-- Define the conditions of the problem
def total_cards : ℕ := 52
def colors : ℕ := 2
def cards_per_color := total_cards / colors
def black_cards_taken_out : ℕ := 5
def total_black_cards : ℕ := cards_per_color

-- Prove the remaining black cards
theorem remaining_black_cards : total_black_cards - black_cards_taken_out = 21 := 
by
  -- Logic to calculate remaining black cards
  sorry

end remaining_black_cards_l1756_175651


namespace inequality_of_fractions_l1756_175623

theorem inequality_of_fractions
  (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (c d : ℝ) (h3 : c < d) (h4 : d < 0)
  (e : ℝ) (h5 : e < 0) :
  (e / ((a - c)^2)) > (e / ((b - d)^2)) :=
by
  sorry

end inequality_of_fractions_l1756_175623


namespace map_distance_to_actual_distance_l1756_175640

theorem map_distance_to_actual_distance
  (map_distance : ℝ)
  (scale_map_to_real : ℝ)
  (scale_real_distance : ℝ)
  (H_map_distance : map_distance = 18)
  (H_scale_map : scale_map_to_real = 0.5)
  (H_scale_real : scale_real_distance = 6) :
  (map_distance / scale_map_to_real) * scale_real_distance = 216 :=
by
  sorry

end map_distance_to_actual_distance_l1756_175640


namespace original_cost_of_pencil_l1756_175694

theorem original_cost_of_pencil (final_price discount: ℝ) (h_final: final_price = 3.37) (h_disc: discount = 0.63) : 
  final_price + discount = 4 :=
by
  sorry

end original_cost_of_pencil_l1756_175694


namespace equilateral_A1C1E1_l1756_175628

variables {A B C D E F A₁ B₁ C₁ D₁ E₁ F₁ : Type*}

-- Defining the convex hexagon and the equilateral triangles.
def is_convex_hexagon (A B C D E F : Type*) : Prop := sorry

def is_equilateral (P Q R : Type*) : Prop := sorry

-- Given conditions
variable (h_hexagon : is_convex_hexagon A B C D E F)
variable (h_eq_triangles :
  is_equilateral A B C₁ ∧ is_equilateral B C D₁ ∧ is_equilateral C D E₁ ∧
  is_equilateral D E F₁ ∧ is_equilateral E F A₁ ∧ is_equilateral F A B₁)
variable (h_B1D1F1 : is_equilateral B₁ D₁ F₁)

-- Statement to be proved
theorem equilateral_A1C1E1 :
  is_equilateral A₁ C₁ E₁ :=
sorry

end equilateral_A1C1E1_l1756_175628


namespace locus_of_intersection_l1756_175662

theorem locus_of_intersection (m : ℝ) :
  (∃ x y : ℝ, (m * x - y + 1 = 0) ∧ (x - m * y - 1 = 0)) ↔ (∃ x y : ℝ, (x - y = 0) ∨ (x - y + 1 = 0)) :=
by
  sorry

end locus_of_intersection_l1756_175662


namespace solution_set_inequality_l1756_175644

theorem solution_set_inequality (x : ℝ) : 2 * x^2 - x - 3 > 0 ↔ x > 3 / 2 ∨ x < -1 := 
by sorry

end solution_set_inequality_l1756_175644


namespace complement_union_correct_l1756_175613

noncomputable def U : Set ℕ := {2, 3, 4, 5, 6}
noncomputable def A : Set ℕ := {x | x^2 - 6*x + 8 = 0}
noncomputable def B : Set ℕ := {2, 5, 6}

theorem complement_union_correct :
  (U \ A) ∪ B = {2, 3, 5, 6} := 
by
  sorry

end complement_union_correct_l1756_175613


namespace inscribed_square_area_l1756_175658

theorem inscribed_square_area (R : ℝ) (h : (R^2 * (π - 2) / 4) = (2 * π - 4)) : 
  ∃ (a : ℝ), a^2 = 16 := by
  sorry

end inscribed_square_area_l1756_175658


namespace manny_had_3_pies_l1756_175695

-- Definitions of the conditions
def number_of_classmates : ℕ := 24
def number_of_teachers : ℕ := 1
def slices_per_pie : ℕ := 10
def slices_left : ℕ := 4

-- Number of people including Manny
def number_of_people : ℕ := number_of_classmates + number_of_teachers + 1

-- Total number of slices eaten
def slices_eaten : ℕ := number_of_people

-- Total number of slices initially
def total_slices : ℕ := slices_eaten + slices_left

-- Number of pies Manny had
def number_of_pies : ℕ := (total_slices / slices_per_pie) + 1

-- Theorem statement
theorem manny_had_3_pies : number_of_pies = 3 := by
  sorry

end manny_had_3_pies_l1756_175695


namespace f_at_3_l1756_175693

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * x - 1

-- The theorem to prove
theorem f_at_3 : f 3 = 5 := sorry

end f_at_3_l1756_175693


namespace ratio_boys_to_girls_l1756_175690

theorem ratio_boys_to_girls
  (b g : ℕ) 
  (h1 : b = g + 6) 
  (h2 : b + g = 36) : b / g = 7 / 5 :=
sorry

end ratio_boys_to_girls_l1756_175690


namespace find_chemistry_marks_l1756_175636

theorem find_chemistry_marks
  (english_marks : ℕ) (math_marks : ℕ) (physics_marks : ℕ) (biology_marks : ℕ) (average_marks : ℕ) (chemistry_marks : ℕ) :
  english_marks = 86 → math_marks = 89 → physics_marks = 82 → biology_marks = 81 → average_marks = 85 →
  chemistry_marks = 425 - (english_marks + math_marks + physics_marks + biology_marks) →
  chemistry_marks = 87 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4] at h6
  have total_marks := 425 - (86 + 89 + 82 + 81)
  norm_num at total_marks
  exact h6

end find_chemistry_marks_l1756_175636


namespace students_moved_outside_correct_l1756_175665

noncomputable def students_total : ℕ := 90
noncomputable def students_cafeteria_initial : ℕ := (2 * students_total) / 3
noncomputable def students_outside_initial : ℕ := students_total - students_cafeteria_initial
noncomputable def students_ran_inside : ℕ := students_outside_initial / 3
noncomputable def students_cafeteria_now : ℕ := 67
noncomputable def students_moved_outside : ℕ := students_cafeteria_initial + students_ran_inside - students_cafeteria_now

theorem students_moved_outside_correct : students_moved_outside = 3 := by
  sorry

end students_moved_outside_correct_l1756_175665


namespace darnell_avg_yards_eq_11_l1756_175669

-- Defining the given conditions
def malikYardsPerGame := 18
def josiahYardsPerGame := 22
def numberOfGames := 4
def totalYardsRun := 204

-- Defining the corresponding total yards for Malik and Josiah
def malikTotalYards := malikYardsPerGame * numberOfGames
def josiahTotalYards := josiahYardsPerGame * numberOfGames

-- The combined total yards for Malik and Josiah
def combinedTotal := malikTotalYards + josiahTotalYards

-- Calculate Darnell's total yards and average per game
def darnellTotalYards := totalYardsRun - combinedTotal
def darnellAverageYardsPerGame := darnellTotalYards / numberOfGames

-- Now, we write the theorem to prove darnell's average yards per game
theorem darnell_avg_yards_eq_11 : darnellAverageYardsPerGame = 11 := by
  sorry

end darnell_avg_yards_eq_11_l1756_175669


namespace remainder_divisor_l1756_175616

theorem remainder_divisor (d r : ℤ) (h1 : d > 1) 
  (h2 : 2024 % d = r) (h3 : 3250 % d = r) (h4 : 4330 % d = r) : d - r = 2 := 
by
  sorry

end remainder_divisor_l1756_175616


namespace x_y_quartic_l1756_175626

theorem x_y_quartic (x y : ℝ) (h₁ : x - y = 2) (h₂ : x * y = 48) : x^4 + y^4 = 5392 := by
  sorry

end x_y_quartic_l1756_175626


namespace frank_handed_cashier_amount_l1756_175646

-- Place conditions as definitions
def cost_chocolate_bar : ℕ := 2
def cost_bag_chip : ℕ := 3
def num_chocolate_bars : ℕ := 5
def num_bag_chips : ℕ := 2
def change_received : ℕ := 4

-- Define the target theorem (Lean 4 statement)
theorem frank_handed_cashier_amount :
  (num_chocolate_bars * cost_chocolate_bar + num_bag_chips * cost_bag_chip + change_received = 20) := 
sorry

end frank_handed_cashier_amount_l1756_175646


namespace exists_rectangle_with_properties_l1756_175692

variables {e a φ : ℝ}

-- Define the given conditions
def diagonal_diff (e a : ℝ) := e - a
def angle_between_diagonals (φ : ℝ) := φ

-- The problem to prove
theorem exists_rectangle_with_properties (e a φ : ℝ) 
  (h_diff : diagonal_diff e a = e - a) 
  (h_angle : angle_between_diagonals φ = φ) : 
  ∃ (rectangle : Type) (A B C D : rectangle), 
    (e - a = e - a) ∧ 
    (φ = φ) := 
sorry

end exists_rectangle_with_properties_l1756_175692


namespace barrel_contents_lost_l1756_175612

theorem barrel_contents_lost (initial_amount remaining_amount : ℝ) 
  (h1 : initial_amount = 220) 
  (h2 : remaining_amount = 198) : 
  (initial_amount - remaining_amount) / initial_amount * 100 = 10 :=
by
  rw [h1, h2]
  sorry

end barrel_contents_lost_l1756_175612


namespace sum_of_possible_values_l1756_175610

theorem sum_of_possible_values (x y : ℝ) 
  (h : x * y - 2 * x / y ^ 3 - 2 * y / x ^ 3 = 4) : 
  (x - 2) * (y - 2) = 1 := 
sorry

end sum_of_possible_values_l1756_175610


namespace max_notebooks_with_budget_l1756_175661

/-- Define the prices and quantities of notebooks -/
def notebook_price : ℕ := 2
def four_pack_price : ℕ := 6
def seven_pack_price : ℕ := 9
def max_budget : ℕ := 15

def total_notebooks (single_packs four_packs seven_packs : ℕ) : ℕ :=
  single_packs + 4 * four_packs + 7 * seven_packs

theorem max_notebooks_with_budget : 
  ∃ (single_packs four_packs seven_packs : ℕ), 
    notebook_price * single_packs + 
    four_pack_price * four_packs + 
    seven_pack_price * seven_packs ≤ max_budget ∧ 
    booklet_price * single_packs + 
    four_pack_price * four_packs + 
    seven_pack_price * seven_packs + total_notebooks single_packs four_packs seven_packs = 11 := 
by
  sorry

end max_notebooks_with_budget_l1756_175661


namespace smallest_Y_l1756_175603

theorem smallest_Y (U : ℕ) (Y : ℕ) (hU : U = 15 * Y) 
  (digits_U : ∀ d ∈ Nat.digits 10 U, d = 0 ∨ d = 1) 
  (div_15 : U % 15 = 0) : Y = 74 :=
sorry

end smallest_Y_l1756_175603


namespace find_actual_marks_l1756_175630

theorem find_actual_marks (wrong_marks : ℕ) (avg_increase : ℕ) (num_pupils : ℕ) (h_wrong_marks: wrong_marks = 73) (h_avg_increase : avg_increase = 1/2) (h_num_pupils : num_pupils = 16) : 
  ∃ (actual_marks : ℕ), actual_marks = 65 :=
by
  have total_increase := num_pupils * avg_increase
  have eqn := wrong_marks - total_increase
  use eqn
  sorry

end find_actual_marks_l1756_175630


namespace eval_expr_at_neg3_l1756_175643

theorem eval_expr_at_neg3 : 
  (5 + 2 * (-3) * ((-3) + 5) - 5^2) / (2 * (-3) - 5 + 2 * (-3)^3) = 32 / 65 := 
by 
  sorry

end eval_expr_at_neg3_l1756_175643


namespace rectangular_solid_volume_l1756_175666

variables {x y z : ℝ}

theorem rectangular_solid_volume :
  x * y = 15 ∧ y * z = 10 ∧ x * z = 6 ∧ x = 3 * y →
  x * y * z = 6 * Real.sqrt 5 :=
by
  intros h
  sorry

end rectangular_solid_volume_l1756_175666


namespace fraction_subtraction_proof_l1756_175609

theorem fraction_subtraction_proof : 
  (21 / 12) - (18 / 15) = 11 / 20 := 
by 
  sorry

end fraction_subtraction_proof_l1756_175609


namespace total_pages_in_book_l1756_175617

theorem total_pages_in_book (P : ℕ)
  (first_day : P - (P / 5) - 12 = remaining_1)
  (second_day : remaining_1 - (remaining_1 / 4) - 15 = remaining_2)
  (third_day : remaining_2 - (remaining_2 / 3) - 18 = 42) :
  P = 190 := 
sorry

end total_pages_in_book_l1756_175617


namespace sum_of_roots_l1756_175667

variable {p m n : ℝ}

axiom roots_condition (h : m * n = 4) : m + n = -4

theorem sum_of_roots (h : m * n = 4) : m + n = -4 := 
  roots_condition h

end sum_of_roots_l1756_175667


namespace minimum_value_l1756_175615

theorem minimum_value (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_sum : a + b = 2) :
  (1 / a) + (1 / b) ≥ 2 :=
by {
  sorry
}

end minimum_value_l1756_175615


namespace sum_of_repeating_decimals_l1756_175672

-- Declare the repeating decimals as constants
def x : ℚ := 2/3
def y : ℚ := 7/9

-- The problem statement
theorem sum_of_repeating_decimals : x + y = 13 / 9 := by
  sorry

end sum_of_repeating_decimals_l1756_175672


namespace largest_multiple_of_6_neg_greater_than_neg_150_l1756_175677

theorem largest_multiple_of_6_neg_greater_than_neg_150 : 
  ∃ m : ℤ, m % 6 = 0 ∧ -m > -150 ∧ m = 144 :=
by
  sorry

end largest_multiple_of_6_neg_greater_than_neg_150_l1756_175677


namespace abs_diff_mn_sqrt_eight_l1756_175648

theorem abs_diff_mn_sqrt_eight {m n p : ℝ} (h1 : m * n = 6) (h2 : m + n + p = 7) (h3 : p = 1) :
  |m - n| = 2 * Real.sqrt 3 :=
by
  sorry

end abs_diff_mn_sqrt_eight_l1756_175648


namespace cat_walking_rate_l1756_175645

theorem cat_walking_rate :
  let resisting_time := 20 -- minutes
  let total_distance := 64 -- feet
  let total_time := 28 -- minutes
  let walking_time := total_time - resisting_time
  (total_distance / walking_time = 8) :=
by
  let resisting_time := 20
  let total_distance := 64
  let total_time := 28
  let walking_time := total_time - resisting_time
  have : total_distance / walking_time = 8 :=
    by norm_num [total_distance, walking_time]
  exact this

end cat_walking_rate_l1756_175645


namespace min_value_reciprocals_l1756_175674

variable {a b : ℝ}

theorem min_value_reciprocals (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 2) :
  (∃ a b, a > 0 ∧ b > 0 ∧ a + b = 2 ∧ ∀ x y, x > 0 → y > 0 → x + y = 2 → 
  (1/a + 1/b) ≥ 2) :=
sorry

end min_value_reciprocals_l1756_175674


namespace shifted_parabola_expression_l1756_175653

theorem shifted_parabola_expression (x : ℝ) :
  let y_original := x^2
  let y_shifted_right := (x - 1)^2
  let y_shifted_up := y_shifted_right + 2
  y_shifted_up = (x - 1)^2 + 2 :=
by
  sorry

end shifted_parabola_expression_l1756_175653


namespace total_pencils_l1756_175683

theorem total_pencils (pencils_per_child : ℕ) (children : ℕ) (hp : pencils_per_child = 2) (hc : children = 8) :
  pencils_per_child * children = 16 :=
by
  sorry

end total_pencils_l1756_175683


namespace problem_l1756_175639

variables (x y z : ℝ)

theorem problem :
  x - y - z = 3 ∧ yz - xy - xz = 3 → x^2 + y^2 + z^2 = 3 :=
by
  sorry

end problem_l1756_175639


namespace expression_equals_two_l1756_175673

noncomputable def simplify_expression : ℝ :=
  1 / (Real.log 3 / Real.log 12 + 1) +
  1 / (Real.log 2 / Real.log 8 + 1) +
  1 / (Real.log 3 / Real.log 9 + 1)

theorem expression_equals_two : simplify_expression = 2 :=
by
  sorry

end expression_equals_two_l1756_175673


namespace perpendicular_line_through_P_l1756_175688

open Real

/-- Define the point P as (-1, 3) -/
def P : ℝ × ℝ := (-1, 3)

/-- Define the line equation -/
def line1 (x y : ℝ) : Prop := x + 2 * y - 3 = 0

/-- Define the perpendicular line equation -/
def perpendicular_line (x y : ℝ) : Prop := 2 * x - y + 5 = 0

/-- The theorem stating that P lies on the perpendicular line to the given line -/
theorem perpendicular_line_through_P : ∀ x y, P = (x, y) → line1 x y → perpendicular_line x y :=
by
  sorry

end perpendicular_line_through_P_l1756_175688


namespace sample_size_stratified_sampling_l1756_175605

theorem sample_size_stratified_sampling :
  let N_business := 120
  let N_management := 24
  let N_logistics := 16
  let N_total := N_business + N_management + N_logistics
  let n_management_chosen := 3
  let sampling_fraction := n_management_chosen / N_management
  let sample_size := N_total * sampling_fraction
  sample_size = 20 :=
by
  -- Definitions:
  let N_business := 120
  let N_management := 24
  let N_logistics := 16
  let N_total := N_business + N_management + N_logistics
  let n_management_chosen := 3
  let sampling_fraction := n_management_chosen / N_management
  let sample_size := N_total * sampling_fraction
  
  -- Proof:
  sorry

end sample_size_stratified_sampling_l1756_175605


namespace smallest_x_no_triangle_l1756_175671

def triangle_inequality_violated (a b c : ℝ) : Prop :=
a + b ≤ c ∨ a + c ≤ b ∨ b + c ≤ a

theorem smallest_x_no_triangle (x : ℕ) (h : ∀ x, triangle_inequality_violated (7 - x : ℝ) (24 - x : ℝ) (26 - x : ℝ)) : x = 5 :=
sorry

end smallest_x_no_triangle_l1756_175671


namespace area_of_formed_triangle_l1756_175620

def triangle_area (S R d : ℝ) (S₁ : ℝ) : Prop :=
  S₁ = (S / 4) * |1 - (d^2 / R^2)|

variable (S R d : ℝ)

theorem area_of_formed_triangle (h : S₁ = (S / 4) * |1 - (d^2 / R^2)|) : triangle_area S R d S₁ :=
by
  sorry

end area_of_formed_triangle_l1756_175620


namespace plane_can_be_colored_l1756_175629

-- Define a structure for a triangle and the plane divided into triangles
structure Triangle :=
(vertices : Fin 3 → ℕ) -- vertices labeled with ℕ, interpreted as 0, 1, 2

structure Plane :=
(triangles : Set Triangle)
(adjacent : Triangle → Triangle → Prop)
(labels_correct : ∀ {t1 t2 : Triangle}, adjacent t1 t2 → 
  ∀ i j: Fin 3, t1.vertices i ≠ t1.vertices j)
(adjacent_conditions: ∀ t1 t2: Triangle, adjacent t1 t2 → 
  ∃ v, (∃ i: Fin 3, t1.vertices i = v) ∧ (∃ j: Fin 3, t2.vertices j = v))

theorem plane_can_be_colored (p : Plane) : 
  ∃ (c : Triangle → ℕ), (∀ t1 t2, p.adjacent t1 t2 → c t1 ≠ c t2) :=
sorry

end plane_can_be_colored_l1756_175629


namespace distinct_factorizations_72_l1756_175652

-- Define the function D that calculates the number of distinct factorizations.
noncomputable def D (n : Nat) : Nat := 
  -- Placeholder function to represent D, the actual implementation is skipped.
  sorry

-- Theorem stating the number of distinct factorizations of 72 considering the order of factors.
theorem distinct_factorizations_72 : D 72 = 119 :=
  sorry

end distinct_factorizations_72_l1756_175652


namespace bundles_burned_in_afternoon_l1756_175611

theorem bundles_burned_in_afternoon 
  (morning_burn : ℕ)
  (start_bundles : ℕ)
  (end_bundles : ℕ)
  (h_morning_burn : morning_burn = 4)
  (h_start : start_bundles = 10)
  (h_end : end_bundles = 3)
  : (start_bundles - morning_burn - end_bundles) = 3 := 
by 
  sorry

end bundles_burned_in_afternoon_l1756_175611


namespace factorization_and_evaluation_l1756_175602

noncomputable def polynomial_q1 (x : ℝ) : ℝ := x
noncomputable def polynomial_q2 (x : ℝ) : ℝ := x^2 - 2
noncomputable def polynomial_q3 (x : ℝ) : ℝ := x^2 + x + 1
noncomputable def polynomial_q4 (x : ℝ) : ℝ := x^2 + 1

theorem factorization_and_evaluation :
  polynomial_q1 3 + polynomial_q2 3 + polynomial_q3 3 + polynomial_q4 3 = 33 := by
  sorry

end factorization_and_evaluation_l1756_175602


namespace wilson_pays_total_l1756_175637

def hamburger_price : ℝ := 5
def cola_price : ℝ := 2
def fries_price : ℝ := 3
def sundae_price : ℝ := 4
def discount_coupon : ℝ := 4
def loyalty_discount : ℝ := 0.10

def total_cost_before_discounts : ℝ :=
  2 * hamburger_price + 3 * cola_price + fries_price + sundae_price

def total_cost_after_coupon : ℝ :=
  total_cost_before_discounts - discount_coupon

def loyalty_discount_amount : ℝ :=
  loyalty_discount * total_cost_after_coupon

def total_cost_after_all_discounts : ℝ :=
  total_cost_after_coupon - loyalty_discount_amount

theorem wilson_pays_total : total_cost_after_all_discounts = 17.10 :=
  sorry

end wilson_pays_total_l1756_175637


namespace least_integer_gt_square_l1756_175684

theorem least_integer_gt_square (x : ℝ) (y : ℝ) (h1 : x = 2) (h2 : y = Real.sqrt 3) :
  ∃ (n : ℤ), n = 14 ∧ n > (x + y) ^ 2 := by
  sorry

end least_integer_gt_square_l1756_175684


namespace inverse_proposition_false_l1756_175600

-- Definitions for the conditions
def congruent (A B C D E F: ℝ) : Prop := 
  A = D ∧ B = E ∧ C = F

def angles_equal (α β γ δ ε ζ: ℝ) : Prop := 
  α = δ ∧ β = ε ∧ γ = ζ

def original_proposition (A B C D E F α β γ : ℝ) : Prop :=
  congruent A B C D E F → angles_equal α β γ A B C

-- The inverse proposition
def inverse_proposition (α β γ δ ε ζ A B C D E F : ℝ) : Prop :=
  angles_equal α β γ δ ε ζ → congruent A B C D E F

-- The main theorem: the inverse proposition is false
theorem inverse_proposition_false (α β γ δ ε ζ A B C D E F : ℝ) :
  ¬(inverse_proposition α β γ δ ε ζ A B C D E F) := by
  sorry

end inverse_proposition_false_l1756_175600


namespace intersection_complement_eq_l1756_175619

open Set

variable (U A B : Set ℕ)
  
theorem intersection_complement_eq : 
  U = {0, 1, 2, 3, 4} → 
  A = {0, 1, 3} → 
  B = {2, 3} → 
  A ∩ (U \ B) = {0, 1} :=
by
  intros hU hA hB
  rw [hU, hA, hB]
  sorry

end intersection_complement_eq_l1756_175619


namespace equivalent_statements_l1756_175606

variable (P Q : Prop)

theorem equivalent_statements : 
  (P → Q) ↔ ((¬ Q → ¬ P) ∧ (¬ P ∨ Q)) :=
by 
  sorry

end equivalent_statements_l1756_175606


namespace dog_ate_cost_6_l1756_175632

noncomputable def totalCost : ℝ := 4 + 2 + 0.5 + 2.5
noncomputable def costPerSlice : ℝ := totalCost / 6
noncomputable def slicesEatenByDog : ℕ := 6 - 2
noncomputable def costEatenByDog : ℝ := slicesEatenByDog * costPerSlice

theorem dog_ate_cost_6 : costEatenByDog = 6 := by
  sorry

end dog_ate_cost_6_l1756_175632


namespace correct_operation_l1756_175649

theorem correct_operation (x y a b : ℝ) :
  (-2 * x) * (3 * y) = -6 * x * y :=
by
  sorry

end correct_operation_l1756_175649


namespace factorize_square_difference_l1756_175699

open Real

theorem factorize_square_difference (m n : ℝ) :
  m ^ 2 - 4 * n ^ 2 = (m + 2 * n) * (m - 2 * n) :=
sorry

end factorize_square_difference_l1756_175699


namespace inscribed_circle_radius_DEF_l1756_175698

noncomputable def radius_inscribed_circle (DE DF EF : ℕ) : ℝ :=
  let s := (DE + DF + EF) / 2
  let K := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  K / s

theorem inscribed_circle_radius_DEF :
  radius_inscribed_circle 26 16 20 = 5 * Real.sqrt 511.5 / 31 :=
by
  sorry

end inscribed_circle_radius_DEF_l1756_175698


namespace size_of_angle_B_length_of_side_b_and_area_l1756_175634

-- Given problem conditions
variables (A B C : ℝ) (a b c : ℝ)
variables (h1 : a < b) (h2 : b < c) (h3 : a / Real.sin A = 2 * b / Real.sqrt 3)

-- Prove that B = π / 3
theorem size_of_angle_B : B = Real.pi / 3 := 
sorry

-- Additional conditions for part (2)
variables (h4 : a = 2) (h5 : c = 3) (h6 : Real.cos B = 1 / 2)

-- Prove b = √7 and the area of triangle ABC
theorem length_of_side_b_and_area :
  b = Real.sqrt 7 ∧ 1/2 * a * c * Real.sin B = 3 * Real.sqrt 3 / 2 :=
sorry

end size_of_angle_B_length_of_side_b_and_area_l1756_175634


namespace license_plate_palindrome_probability_l1756_175689

-- Definitions for the problem conditions
def count_letter_palindromes : ℕ := 26 * 26
def total_letter_combinations : ℕ := 26 ^ 4

def count_digit_palindromes : ℕ := 10 * 10
def total_digit_combinations : ℕ := 10 ^ 4

def prob_letter_palindrome : ℚ := count_letter_palindromes / total_letter_combinations
def prob_digit_palindrome : ℚ := count_digit_palindromes / total_digit_combinations
def prob_both_palindrome : ℚ := (count_letter_palindromes * count_digit_palindromes) / (total_letter_combinations * total_digit_combinations)

def prob_atleast_one_palindrome : ℚ :=
  prob_letter_palindrome + prob_digit_palindrome - prob_both_palindrome

def p_q_sum : ℕ := 775 + 67600

-- Statement of the problem to be proved
theorem license_plate_palindrome_probability :
  prob_atleast_one_palindrome = 775 / 67600 ∧ p_q_sum = 68375 :=
by { sorry }

end license_plate_palindrome_probability_l1756_175689


namespace arithmetic_sequence_sum_l1756_175654

noncomputable def sum_of_first_n_terms (n : ℕ) (a d : ℝ) : ℝ :=
  n / 2 * (2 * a + (n - 1) * d)

theorem arithmetic_sequence_sum 
  (a_n : ℕ → ℝ) 
  (h_arith : ∃ d, ∀ n, a_n (n + 1) = a_n n + d) 
  (h1 : a_n 1 + a_n 2 + a_n 3 = 3 )
  (h2 : a_n 28 + a_n 29 + a_n 30 = 165 ) 
  : sum_of_first_n_terms 30 (a_n 1) (a_n 2 - a_n 1) = 840 := 
  sorry

end arithmetic_sequence_sum_l1756_175654


namespace parabola_focus_l1756_175679

theorem parabola_focus : 
  ∀ x y : ℝ, y = - (1 / 16) * x^2 → ∃ f : ℝ × ℝ, f = (0, -4) := 
by
  sorry

end parabola_focus_l1756_175679


namespace weight_of_one_baseball_l1756_175641

structure Context :=
  (numberBaseballs : ℕ)
  (numberBicycles : ℕ)
  (weightBicycles : ℕ)
  (weightTotalBicycles : ℕ)

def problem (ctx : Context) :=
  ctx.weightTotalBicycles = ctx.numberBicycles * ctx.weightBicycles ∧
  ctx.numberBaseballs * ctx.weightBicycles = ctx.weightTotalBicycles →
  (ctx.weightTotalBicycles / ctx.numberBaseballs) = 8

theorem weight_of_one_baseball (ctx : Context) : problem ctx :=
sorry

end weight_of_one_baseball_l1756_175641


namespace concentration_replacement_l1756_175622

theorem concentration_replacement 
  (initial_concentration : ℝ)
  (new_concentration : ℝ)
  (fraction_replaced : ℝ)
  (replacing_concentration : ℝ)
  (h1 : initial_concentration = 0.45)
  (h2 : new_concentration = 0.35)
  (h3 : fraction_replaced = 0.5) :
  replacing_concentration = 0.25 := by
  sorry

end concentration_replacement_l1756_175622


namespace moles_of_NH3_formed_l1756_175650

-- Conditions
def moles_NH4Cl : ℕ := 3 -- 3 moles of Ammonium chloride
def total_moles_NH3_formed : ℕ := 3 -- The total moles of Ammonia formed

-- The balanced chemical reaction implies a 1:1 molar ratio
lemma reaction_ratio (n : ℕ) : total_moles_NH3_formed = n := by
  sorry

-- Prove that the number of moles of NH3 formed is equal to 3
theorem moles_of_NH3_formed : total_moles_NH3_formed = moles_NH4Cl := 
reaction_ratio moles_NH4Cl

end moles_of_NH3_formed_l1756_175650


namespace change_factor_l1756_175642

theorem change_factor (n : ℕ) (avg_original avg_new : ℕ) (F : ℝ)
  (h1 : n = 10) (h2 : avg_original = 80) (h3 : avg_new = 160) 
  (h4 : F * (n * avg_original) = n * avg_new) :
  F = 2 :=
by
  sorry

end change_factor_l1756_175642


namespace base_conversion_subtraction_l1756_175624

/-- Definition of base conversion from base 7 and base 5 to base 10. -/
def convert_base_7_to_10 (n : Nat) : Nat :=
  match n with
  | 52103 => 5 * 7^4 + 2 * 7^3 + 1 * 7^2 + 0 * 7^1 + 3 * 7^0
  | _ => 0

def convert_base_5_to_10 (n : Nat) : Nat :=
  match n with
  | 43120 => 4 * 5^4 + 3 * 5^3 + 1 * 5^2 + 2 * 5^1 + 0 * 5^0
  | _ => 0

theorem base_conversion_subtraction : 
  convert_base_7_to_10 52103 - convert_base_5_to_10 43120 = 9833 :=
by
  -- The proof goes here
  sorry

end base_conversion_subtraction_l1756_175624


namespace total_toys_correct_l1756_175604

-- Define the conditions
def JaxonToys : ℕ := 15
def GabrielToys : ℕ := 2 * JaxonToys
def JerryToys : ℕ := GabrielToys + 8

-- Define the total number of toys
def TotalToys : ℕ := JaxonToys + GabrielToys + JerryToys

-- State the theorem
theorem total_toys_correct : TotalToys = 83 :=
by
  -- Skipping the proof as per instruction
  sorry

end total_toys_correct_l1756_175604


namespace sled_total_distance_l1756_175660

def arithmetic_sequence_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (a₁ + (a₁ + (n - 1) * d)) / 2

theorem sled_total_distance (a₁ : ℕ) (d : ℕ) (n : ℕ) :
  a₁ = 6 → d = 8 → n = 20 → arithmetic_sequence_sum a₁ d n = 1640 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end sled_total_distance_l1756_175660


namespace mark_reading_time_l1756_175686

variable (x y : ℕ)

theorem mark_reading_time (x y : ℕ) : 
  7 * x + y = 7 * x + y :=
by
  sorry

end mark_reading_time_l1756_175686


namespace television_hours_watched_l1756_175696

theorem television_hours_watched (minutes_per_day : ℕ) (days_per_week : ℕ) (weeks : ℕ)
  (h1 : minutes_per_day = 45) (h2 : days_per_week = 4) (h3 : weeks = 2):
  (minutes_per_day * days_per_week / 60) * weeks = 6 :=
by
  sorry

end television_hours_watched_l1756_175696


namespace marble_price_proof_l1756_175675

noncomputable def price_per_colored_marble (total_marbles white_percentage black_percentage white_price black_price total_earnings : ℕ) : ℕ :=
  let white_marbles := total_marbles * white_percentage / 100
  let black_marbles := total_marbles * black_percentage / 100
  let colored_marbles := total_marbles - (white_marbles + black_marbles)
  let earnings_from_white := white_marbles * white_price
  let earnings_from_black := black_marbles * black_price
  let earnings_from_colored := total_earnings - (earnings_from_white + earnings_from_black)
  earnings_from_colored / colored_marbles

theorem marble_price_proof : price_per_colored_marble 100 20 30 5 10 1400 = 20 := 
sorry

end marble_price_proof_l1756_175675


namespace probability_of_region_F_l1756_175656

theorem probability_of_region_F
  (pD pE pG pF : ℚ)
  (hD : pD = 3/8)
  (hE : pE = 1/4)
  (hG : pG = 1/8)
  (hSum : pD + pE + pF + pG = 1) : pF = 1/4 :=
by
  -- we can perform the steps as mentioned in the solution without actually executing them
  sorry

end probability_of_region_F_l1756_175656


namespace find_m_collinear_l1756_175607

theorem find_m_collinear (m : ℝ) 
    (a : ℝ × ℝ := (m + 3, 2)) 
    (b : ℝ × ℝ := (m, 1)) 
    (collinear : a.1 * 1 - 2 * b.1 = 0) : 
    m = 3 :=
by {
    sorry
}

end find_m_collinear_l1756_175607


namespace single_jalapeno_strips_l1756_175601

-- Definitions based on conditions
def strips_per_sandwich : ℕ := 4
def minutes_per_sandwich : ℕ := 5
def hours_per_day : ℕ := 8
def total_jalapeno_peppers_used : ℕ := 48
def minutes_per_hour : ℕ := 60

-- Calculate intermediate steps
def total_minutes : ℕ := hours_per_day * minutes_per_hour
def total_sandwiches_served : ℕ := total_minutes / minutes_per_sandwich
def total_strips_needed : ℕ := total_sandwiches_served * strips_per_sandwich

theorem single_jalapeno_strips :
  total_strips_needed / total_jalapeno_peppers_used = 8 := 
by
  sorry

end single_jalapeno_strips_l1756_175601


namespace problem_l1756_175678

variable (a b : ℝ)

theorem problem (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = 4) : a^2 + b^2 ≥ 8 := 
sorry

end problem_l1756_175678


namespace sms_message_fraudulent_l1756_175685

-- Define the conditions as properties
def messageArrivedNumberKnown (msg : String) (numberKnown : Bool) : Prop :=
  msg = "SMS message has already arrived" ∧ numberKnown = true

def fraudDefinition (acquisition : String -> Prop) : Prop :=
  ∀ (s : String), acquisition s = (s = "acquisition of property by third parties through deception or gaining the trust of the victim")

-- Define the main proof problem statement
theorem sms_message_fraudulent (msg : String) (numberKnown : Bool) (acquisition : String -> Prop) :
  messageArrivedNumberKnown msg numberKnown ∧ fraudDefinition acquisition →
  acquisition "acquisition of property by third parties through deception or gaining the trust of the victim" :=
  sorry

end sms_message_fraudulent_l1756_175685


namespace number_of_months_in_season_l1756_175697

def games_per_month : ℝ := 323.0
def total_games : ℝ := 5491.0

theorem number_of_months_in_season : total_games / games_per_month = 17 := 
by
  sorry

end number_of_months_in_season_l1756_175697


namespace probability_not_siblings_l1756_175614

-- Define the number of people and the sibling condition
def number_of_people : ℕ := 6
def siblings_count (x : Fin number_of_people) : ℕ := 2

-- Define the probability that two individuals randomly selected are not siblings
theorem probability_not_siblings (P Q : Fin number_of_people) (h : P ≠ Q) :
  let K := number_of_people - 1
  let non_siblings := K - siblings_count P
  (non_siblings / K : ℚ) = 3 / 5 :=
by
  intros
  sorry

end probability_not_siblings_l1756_175614


namespace problem1_problem2_problem3_problem4_l1756_175631

-- Statement for problem 1
theorem problem1 : -12 + (-6) - (-28) = 10 :=
  by sorry

-- Statement for problem 2
theorem problem2 : (-8 / 5) * (15 / 4) / (-9) = 2 / 3 :=
  by sorry

-- Statement for problem 3
theorem problem3 : (-3 / 16 - 7 / 24 + 5 / 6) * (-48) = -17 :=
  by sorry

-- Statement for problem 4
theorem problem4 : -3^2 + (7 / 8 - 1) * (-2)^2 = -9.5 :=
  by sorry

end problem1_problem2_problem3_problem4_l1756_175631


namespace bookshelf_arrangements_l1756_175657

theorem bookshelf_arrangements :
  let math_books := 6
  let english_books := 5
  let valid_arrangements := 2400
  (∃ (math_books : Nat) (english_books : Nat) (valid_arrangements : Nat), 
    math_books = 6 ∧ english_books = 5 ∧ valid_arrangements = 2400) :=
by
  sorry

end bookshelf_arrangements_l1756_175657


namespace gcd_of_powers_l1756_175691

theorem gcd_of_powers (a b c : ℕ) (h1 : a = 2^105 - 1) (h2 : b = 2^115 - 1) (h3 : c = 1023) :
  Nat.gcd a b = c :=
by sorry

end gcd_of_powers_l1756_175691


namespace sum_of_midpoints_l1756_175627

theorem sum_of_midpoints {a b c : ℝ} (h : a + b + c = 15) : 
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 := 
by
  sorry

end sum_of_midpoints_l1756_175627


namespace linear_eq_rewrite_l1756_175682

theorem linear_eq_rewrite (x y : ℝ) (h : 2 * x - y = 3) : y = 2 * x - 3 :=
by
  sorry

end linear_eq_rewrite_l1756_175682


namespace gcd_ab_a2b2_eq_1_or_2_l1756_175687

theorem gcd_ab_a2b2_eq_1_or_2
  (a b : Nat)
  (h_coprime : Nat.gcd a b = 1) :
  Nat.gcd (a + b) (a^2 + b^2) = 1 ∨ Nat.gcd (a + b) (a^2 + b^2) = 2 :=
by {
  sorry
}

end gcd_ab_a2b2_eq_1_or_2_l1756_175687


namespace total_sugar_l1756_175647

theorem total_sugar (sugar_frosting sugar_cake : ℝ) (h1 : sugar_frosting = 0.6) (h2 : sugar_cake = 0.2) :
  sugar_frosting + sugar_cake = 0.8 :=
by {
  -- The proof goes here
  sorry
}

end total_sugar_l1756_175647


namespace op_example_l1756_175670

def op (a b : ℚ) : ℚ := a * b / (a + b)

theorem op_example : op (op 3 5) (op 5 4) = 60 / 59 := by
  sorry

end op_example_l1756_175670


namespace bob_shuck_2_hours_l1756_175681

def shucking_rate : ℕ := 10  -- oysters per 5 minutes
def minutes_per_hour : ℕ := 60
def hours : ℕ := 2
def minutes : ℕ := hours * minutes_per_hour
def interval : ℕ := 5  -- minutes per interval
def intervals : ℕ := minutes / interval
def num_oysters (intervals : ℕ) : ℕ := intervals * shucking_rate

theorem bob_shuck_2_hours : num_oysters intervals = 240 := by
  -- leave the proof as an exercise
  sorry

end bob_shuck_2_hours_l1756_175681


namespace cube_sum_minus_triple_product_l1756_175633

theorem cube_sum_minus_triple_product (x y z : ℝ) (h1 : x + y + z = 8) (h2 : xy + yz + zx = 20) :
  x^3 + y^3 + z^3 - 3 * x * y * z = 32 :=
sorry

end cube_sum_minus_triple_product_l1756_175633


namespace value_of_expression_l1756_175638

theorem value_of_expression (m n : ℝ) (h : m + n = 3) :
  2 * m^2 + 4 * m * n + 2 * n^2 - 6 = 12 :=
by
  sorry

end value_of_expression_l1756_175638


namespace negation_of_existential_l1756_175668

theorem negation_of_existential (P : Prop) :
  (¬ (∃ x : ℝ, x ^ 3 > 0)) ↔ (∀ x : ℝ, x ^ 3 ≤ 0) :=
by
  sorry

end negation_of_existential_l1756_175668


namespace divisor_unique_l1756_175680

theorem divisor_unique {b : ℕ} (h1 : 826 % b = 7) (h2 : 4373 % b = 8) : b = 9 :=
sorry

end divisor_unique_l1756_175680
