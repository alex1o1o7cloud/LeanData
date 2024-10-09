import Mathlib

namespace number_of_solutions_l2225_222570

theorem number_of_solutions :
  ∃ (s : Finset (ℤ × ℤ)), (∀ (a : ℤ × ℤ), a ∈ s ↔ (a.1^4 + a.2^4 = 4 * a.2)) ∧ s.card = 3 :=
by
  sorry

end number_of_solutions_l2225_222570


namespace oranges_ratio_l2225_222512

theorem oranges_ratio (T : ℕ) (h1 : 100 + T + 70 = 470) : T / 100 = 3 := by
  -- The solution steps are omitted.
  sorry

end oranges_ratio_l2225_222512


namespace diff_present_students_l2225_222530

theorem diff_present_students (T A1 A2 A3 P1 P2 : ℕ) 
  (hT : T = 280)
  (h_total_absent : A1 + A2 + A3 = 240)
  (h_absent_ratio : A2 = 2 * A3)
  (h_absent_third_day : A3 = 280 / 7) 
  (hP1 : P1 = T - A1)
  (hP2 : P2 = T - A2) :
  P2 - P1 = 40 :=
sorry

end diff_present_students_l2225_222530


namespace ship_length_in_emilys_steps_l2225_222559

variable (L E S : ℝ)

-- Conditions from the problem:
variable (cond1 : 240 * E = L + 240 * S)
variable (cond2 : 60 * E = L - 60 * S)

-- Theorem to prove:
theorem ship_length_in_emilys_steps (cond1 : 240 * E = L + 240 * S) (cond2 : 60 * E = L - 60 * S) : 
  L = 96 * E := 
sorry

end ship_length_in_emilys_steps_l2225_222559


namespace range_of_a_l2225_222595

variable (a : ℝ)

def a_n (n : ℕ) : ℝ :=
if n = 1 then a else 4 * ↑n + (-1 : ℝ) ^ n * (8 - 2 * a)

theorem range_of_a (h : ∀ n : ℕ, n > 0 → a_n a n < a_n a (n + 1)) : 3 < a ∧ a < 5 :=
by
  sorry

end range_of_a_l2225_222595


namespace frustum_volume_correct_l2225_222568

-- Definitions of pyramids and their properties
structure Pyramid :=
  (base_edge : ℕ)
  (altitude : ℕ)
  (volume : ℚ)

-- Definition of the original pyramid and smaller pyramid
def original_pyramid : Pyramid := {
  base_edge := 20,
  altitude := 10,
  volume := (1 / 3 : ℚ) * (20 ^ 2) * 10
}

def smaller_pyramid : Pyramid := {
  base_edge := 8,
  altitude := 5,
  volume := (1 / 3 : ℚ) * (8 ^ 2) * 5
}

-- Definition and calculation of the volume of the frustum 
def volume_frustum (p1 p2 : Pyramid) : ℚ :=
  p1.volume - p2.volume

-- Main theorem to be proved
theorem frustum_volume_correct :
  volume_frustum original_pyramid smaller_pyramid = 992 := by
  sorry

end frustum_volume_correct_l2225_222568


namespace hyperbola_parabola_focus_l2225_222521

open Classical

theorem hyperbola_parabola_focus :
  ∃ a : ℝ, (a > 0) ∧ (∃ c > 0, (c = 2) ∧ (a^2 + 3 = c^2)) → a = 1 :=
sorry

end hyperbola_parabola_focus_l2225_222521


namespace stratified_sampling_correct_l2225_222580

def total_employees : ℕ := 150
def senior_titles : ℕ := 15
def intermediate_titles : ℕ := 45
def general_staff : ℕ := 90
def sample_size : ℕ := 30

def stratified_sampling (total_employees senior_titles intermediate_titles general_staff sample_size : ℕ) : (ℕ × ℕ × ℕ) :=
  (senior_titles * sample_size / total_employees, 
   intermediate_titles * sample_size / total_employees, 
   general_staff * sample_size / total_employees)

theorem stratified_sampling_correct :
  stratified_sampling total_employees senior_titles intermediate_titles general_staff sample_size = (3, 9, 18) :=
  by sorry

end stratified_sampling_correct_l2225_222580


namespace num_distinct_pos_factors_81_l2225_222523

-- Define what it means to be a factor
def is_factor (n d : ℕ) : Prop := d ∣ n

-- 81 is 3^4 by definition
def eighty_one : ℕ := 3 ^ 4

-- Define the list of positive factors of 81
def pos_factors_81 : List ℕ := [1, 3, 9, 27, 81]

-- Formal statement that 81 has 5 distinct positive factors
theorem num_distinct_pos_factors_81 : (pos_factors_81.length = 5) :=
by sorry

end num_distinct_pos_factors_81_l2225_222523


namespace operation_value_l2225_222518

variable (a b : ℤ)

theorem operation_value (h : (21 - 1) * (9 - 1) = 160) : a = 21 :=
by
  sorry

end operation_value_l2225_222518


namespace initial_men_count_l2225_222511

theorem initial_men_count (M : ℕ) (P : ℕ) :
  P = M * 20 →
  P = (M + 650) * 109 / 9 →
  M = 1000 :=
by
  sorry

end initial_men_count_l2225_222511


namespace quadratic_inequality_solution_l2225_222546

theorem quadratic_inequality_solution (x : ℝ) : x^2 - 5 * x + 6 < 0 ↔ 2 < x ∧ x < 3 :=
by
  sorry

end quadratic_inequality_solution_l2225_222546


namespace julieta_total_spent_l2225_222526

def original_price_backpack : ℕ := 50
def original_price_ring_binder : ℕ := 20
def quantity_ring_binders : ℕ := 3
def price_increase_backpack : ℕ := 5
def price_decrease_ring_binder : ℕ := 2

def total_spent (original_price_backpack original_price_ring_binder quantity_ring_binders price_increase_backpack price_decrease_ring_binder : ℕ) : ℕ :=
  let new_price_backpack := original_price_backpack + price_increase_backpack
  let new_price_ring_binder := original_price_ring_binder - price_decrease_ring_binder
  new_price_backpack + (new_price_ring_binder * quantity_ring_binders)

theorem julieta_total_spent :
  total_spent original_price_backpack original_price_ring_binder quantity_ring_binders price_increase_backpack price_decrease_ring_binder = 109 :=
by 
  -- Proof steps are omitted intentionally
  sorry

end julieta_total_spent_l2225_222526


namespace mary_more_candy_initially_l2225_222569

-- Definitions of the conditions
def Megan_initial_candy : ℕ := 5
def Mary_candy_after_addition : ℕ := 25
def additional_candy_Mary_adds : ℕ := 10

-- The proof problem statement
theorem mary_more_candy_initially :
  (Mary_candy_after_addition - additional_candy_Mary_adds) / Megan_initial_candy = 3 :=
by
  sorry

end mary_more_candy_initially_l2225_222569


namespace find_x_l2225_222574

variables {x : ℝ}
def vector_parallel (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.2 = v1.2 * v2.1

theorem find_x
  (h1 : (6, 1) = (6, 1))
  (h2 : (x, -3) = (x, -3))
  (h3 : vector_parallel (6, 1) (x, -3)) :
  x = -18 := by
  sorry

end find_x_l2225_222574


namespace field_length_l2225_222508

-- Definitions of the conditions
def pond_area : ℝ := 25  -- area of the square pond
def width_to_length_ratio (w l : ℝ) : Prop := l = 2 * w  -- length is double the width
def pond_to_field_ratio (pond_area field_area : ℝ) : Prop := pond_area = (1/8) * field_area  -- pond area is 1/8 of field area

-- Statement to prove
theorem field_length (w l : ℝ) (h1 : width_to_length_ratio w l) (h2 : pond_to_field_ratio pond_area (l * w)) : l = 20 :=
by sorry

end field_length_l2225_222508


namespace ratio_a_to_b_l2225_222517

theorem ratio_a_to_b (a b : ℝ) (h : (a - 3 * b) / (2 * a - b) = 0.14285714285714285) : a / b = 4 :=
by 
  -- The proof goes here
  sorry

end ratio_a_to_b_l2225_222517


namespace number_of_coaches_l2225_222504

theorem number_of_coaches (r : ℕ) (v : ℕ) (c : ℕ) (h1 : r = 60) (h2 : v = 3) (h3 : c * 5 = 60 * 3) : c = 36 :=
by
  -- We skip the proof as per instructions
  sorry

end number_of_coaches_l2225_222504


namespace tiffany_lives_after_bonus_stage_l2225_222505

theorem tiffany_lives_after_bonus_stage :
  let initial_lives := 250
  let lives_lost := 58
  let remaining_lives := initial_lives - lives_lost
  let additional_lives := 3 * remaining_lives
  let final_lives := remaining_lives + additional_lives
  final_lives = 768 :=
by
  let initial_lives := 250
  let lives_lost := 58
  let remaining_lives := initial_lives - lives_lost
  let additional_lives := 3 * remaining_lives
  let final_lives := remaining_lives + additional_lives
  exact sorry

end tiffany_lives_after_bonus_stage_l2225_222505


namespace value_of_y_l2225_222541

theorem value_of_y (x y : ℤ) (h1 : x - y = 6) (h2 : x + y = 12) : y = 3 := 
by
  sorry

end value_of_y_l2225_222541


namespace find_x_tan_identity_l2225_222500

theorem find_x_tan_identity : 
  ∀ x : ℝ, (0 < x ∧ x < 180) → 
  (Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) → 
  x = 105 :=
by
  intro x hx htan
  sorry

end find_x_tan_identity_l2225_222500


namespace melanie_missed_games_l2225_222531

-- Define the total number of games and the number of games attended by Melanie
def total_games : ℕ := 7
def games_attended : ℕ := 3

-- Define the number of games missed as total games minus games attended
def games_missed : ℕ := total_games - games_attended

-- Theorem stating the number of games missed by Melanie
theorem melanie_missed_games : games_missed = 4 := by
  -- The proof is omitted
  sorry

end melanie_missed_games_l2225_222531


namespace complement_intersection_l2225_222510

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {2, 4, 5, 7}
def B : Set ℕ := {3, 4, 5}

theorem complement_intersection :
  (U \ A) ∩ (U \ B) = {1, 6} :=
by
  sorry

end complement_intersection_l2225_222510


namespace arithmetic_sequence_150th_term_l2225_222515

theorem arithmetic_sequence_150th_term :
  let a1 := 3
  let d := 5
  let n := 150
  a1 + (n - 1) * d = 748 :=
by
  sorry

end arithmetic_sequence_150th_term_l2225_222515


namespace drew_got_wrong_19_l2225_222592

theorem drew_got_wrong_19 :
  ∃ (D_wrong C_wrong : ℕ), 
    (20 + D_wrong = 52) ∧
    (14 + C_wrong = 52) ∧
    (C_wrong = 2 * D_wrong) ∧
    D_wrong = 19 :=
by
  sorry

end drew_got_wrong_19_l2225_222592


namespace positive_difference_of_two_numbers_l2225_222555

theorem positive_difference_of_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : |x - y| = 4 :=
by
  sorry

end positive_difference_of_two_numbers_l2225_222555


namespace xyz_value_l2225_222565

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 40)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) + 2 * x * y * z = 20) 
  : x * y * z = 20 := 
by
  sorry

end xyz_value_l2225_222565


namespace first_pump_rate_is_180_l2225_222534

-- Define the known conditions
variables (R : ℕ) -- The rate of the first pump in gallons per hour
def second_pump_rate : ℕ := 250 -- The rate of the second pump in gallons per hour
def second_pump_time : ℕ := 35 / 10 -- 3.5 hours represented as a fraction
def total_pump_time : ℕ := 60 / 10 -- 6 hours represented as a fraction
def total_volume : ℕ := 1325 -- Total volume pumped by both pumps in gallons

-- Define derived conditions from the problem
def second_pump_volume : ℕ := second_pump_rate * second_pump_time -- Volume pumped by the second pump
def first_pump_volume : ℕ := total_volume - second_pump_volume -- Volume pumped by the first pump
def first_pump_time : ℕ := total_pump_time - second_pump_time -- Time the first pump was used

-- The main theorem to prove that the rate of the first pump is 180 gallons per hour
theorem first_pump_rate_is_180 : R = 180 :=
by
  -- The proof would go here
  sorry

end first_pump_rate_is_180_l2225_222534


namespace perpendicular_exists_l2225_222548

-- Definitions for geometric entities involved

structure Point where
  x : ℝ
  y : ℝ

structure Line where
  p1 : Point
  p2 : Point

structure Circle where
  center : Point
  radius : ℝ

-- Definitions for conditions in the problem

-- Condition 1: Point C is not on the circle
def point_not_on_circle (C : Point) (circle : Circle) : Prop :=
  (C.x - circle.center.x)^2 + (C.y - circle.center.y)^2 ≠ circle.radius^2

-- Condition 2: Point C is on the circle
def point_on_circle (C : Point) (circle : Circle) : Prop :=
  (C.x - circle.center.x)^2 + (C.y - circle.center.y)^2 = circle.radius^2

-- Definitions for lines and perpendicularity
def is_perpendicular (line1 : Line) (line2 : Line) : Prop :=
  (line1.p1.x - line1.p2.x) * (line2.p1.x - line2.p2.x) +
  (line1.p1.y - line1.p2.y) * (line2.p1.y - line2.p2.y) = 0

noncomputable def perpendicular_from_point_to_line (C : Point) (line : Line) (circle : Circle) 
  (h₁ : point_not_on_circle C circle ∨ point_on_circle C circle) : Line := 
  sorry

-- The Lean statement for part (a) and (b) combined into one proof.
theorem perpendicular_exists (C : Point) (lineAB : Line) (circle : Circle) 
  (h₁ : point_not_on_circle C circle ∨ point_on_circle C circle) : 
  ∃ (line_perpendicular : Line), is_perpendicular line_perpendicular lineAB ∧ 
  (line_perpendicular.p1 = C ∨ line_perpendicular.p2 = C) :=
  sorry

end perpendicular_exists_l2225_222548


namespace find_two_digit_number_l2225_222524

def is_positive (n : ℕ) := n > 0
def is_even (n : ℕ) := n % 2 = 0
def is_multiple_of_9 (n : ℕ) := n % 9 = 0
def product_of_digits_is_square (n : ℕ) : Prop :=
  let tens := n / 10
  let units := n % 10
  ∃ k : ℕ, (tens * units) = k * k

theorem find_two_digit_number (N : ℕ) 
  (h_pos : is_positive N) 
  (h_ev : is_even N) 
  (h_mult_9 : is_multiple_of_9 N)
  (h_prod_square : product_of_digits_is_square N) 
: N = 90 := by 
  sorry

end find_two_digit_number_l2225_222524


namespace length_of_AB_l2225_222573

theorem length_of_AB
  (P Q : ℝ) (AB : ℝ)
  (hP : P = 3 / 7 * AB)
  (hQ : Q = 4 / 9 * AB)
  (hPQ : abs (Q - P) = 3) :
  AB = 189 :=
by
  sorry

end length_of_AB_l2225_222573


namespace painted_cubes_on_two_faces_l2225_222567

theorem painted_cubes_on_two_faces (n : ℕ) (painted_faces_all : Prop) (equal_smaller_cubes : n = 27) : ∃ k : ℕ, k = 12 :=
by
  -- We only need the statement, not the proof
  sorry

end painted_cubes_on_two_faces_l2225_222567


namespace find_n_l2225_222575

theorem find_n (a b c : ℤ) (m n p : ℕ)
  (h1 : a = 3)
  (h2 : b = -7)
  (h3 : c = -6)
  (h4 : m > 0)
  (h5 : n > 0)
  (h6 : p > 0)
  (h7 : Nat.gcd m p = 1)
  (h8 : Nat.gcd m n = 1)
  (h9 : Nat.gcd n p = 1)
  (h10 : ∃ x1 x2 : ℤ, x1 = (m + Int.sqrt n) / p ∧ x2 = (m - Int.sqrt n) / p)
  : n = 121 :=
sorry

end find_n_l2225_222575


namespace sqrt_fraction_expression_l2225_222554

theorem sqrt_fraction_expression : 
  (Real.sqrt (9 / 4) - Real.sqrt (4 / 9) + (Real.sqrt (9 / 4) + Real.sqrt (4 / 9))^2) = (199 / 36) := 
by
  sorry

end sqrt_fraction_expression_l2225_222554


namespace remembers_umbrella_prob_l2225_222549

theorem remembers_umbrella_prob 
    (P_forgets : ℚ) 
    (h_forgets : P_forgets = 5 / 8) : 
    ∃ P_remembers : ℚ, P_remembers = 3 / 8 := 
by
    sorry

end remembers_umbrella_prob_l2225_222549


namespace minimum_x_y_l2225_222520

theorem minimum_x_y (x y : ℝ) (hx_pos : x > 0) (hy_pos : y > 0) (h_eq : 2 * x + 8 * y = x * y) : x + y ≥ 18 :=
by sorry

end minimum_x_y_l2225_222520


namespace max_area_circle_between_parallel_lines_l2225_222543

theorem max_area_circle_between_parallel_lines : 
  ∀ (l₁ l₂ : ℝ → ℝ → Prop), 
    (∀ x y, l₁ x y ↔ 3*x - 4*y = 0) → 
    (∀ x y, l₂ x y ↔ 3*x - 4*y - 20 = 0) → 
  ∃ A, A = 4 * Real.pi :=
by 
  sorry

end max_area_circle_between_parallel_lines_l2225_222543


namespace ethan_arianna_apart_l2225_222557

def ethan_distance := 1000 -- the distance Ethan ran
def arianna_distance := 184 -- the distance Arianna ran

theorem ethan_arianna_apart : ethan_distance - arianna_distance = 816 := by
  sorry

end ethan_arianna_apart_l2225_222557


namespace sequence_exists_l2225_222527

theorem sequence_exists
  {a_0 b_0 c_0 a b c : ℤ}
  (gcd1 : Int.gcd (Int.gcd a_0 b_0) c_0 = 1)
  (gcd2 : Int.gcd (Int.gcd a b) c = 1) :
  ∃ (n : ℕ) (a_seq b_seq c_seq : Fin (n + 1) → ℤ),
    a_seq 0 = a_0 ∧ b_seq 0 = b_0 ∧ c_seq 0 = c_0 ∧ 
    a_seq n = a ∧ b_seq n = b ∧ c_seq n = c ∧
    ∀ (i : Fin n), (a_seq i) * (a_seq i.succ) + (b_seq i) * (b_seq i.succ) + (c_seq i) * (c_seq i.succ) = 1 :=
sorry

end sequence_exists_l2225_222527


namespace purely_imaginary_complex_number_l2225_222513

theorem purely_imaginary_complex_number (a : ℝ) 
  (h1 : (a^2 - 4 * a + 3 = 0))
  (h2 : a ≠ 1) 
  : a = 3 := 
sorry

end purely_imaginary_complex_number_l2225_222513


namespace value_of_v3_at_2_l2225_222529

def f (x : ℝ) : ℝ := x^5 - 2 * x^4 + 3 * x^3 - 7 * x^2 + 6 * x - 3

def v3 (x : ℝ) := (x - 2) * x + 3 
def v3_eval_at_2 : ℝ := (2 - 2) * 2 + 3

theorem value_of_v3_at_2 : v3 2 - 7 = -1 := by
    sorry

end value_of_v3_at_2_l2225_222529


namespace condition_for_ellipse_l2225_222583

-- Definition of the problem conditions
def is_ellipse (m : ℝ) : Prop :=
  (m - 2 > 0) ∧ (5 - m > 0) ∧ (m - 2 ≠ 5 - m)

noncomputable def necessary_not_sufficient_condition (m : ℝ) : Prop :=
  (2 < m) ∧ (m < 5)

-- The theorem to be proved
theorem condition_for_ellipse (m : ℝ) : 
  (necessary_not_sufficient_condition m) → (is_ellipse m) :=
by
  -- proof to be written here
  sorry

end condition_for_ellipse_l2225_222583


namespace construct_circle_feasible_l2225_222584

theorem construct_circle_feasible (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : b^2 > (a^2 + c^2) / 2) :
  ∃ x y d : ℝ, 
  d > 0 ∧ 
  (d / 2)^2 = y^2 + (a / 2)^2 ∧ 
  (d / 2)^2 = (y - x)^2 + (b / 2)^2 ∧ 
  (d / 2)^2 = (y - 2 * x)^2 + (c / 2)^2 :=
sorry

end construct_circle_feasible_l2225_222584


namespace sum_of_values_not_satisfying_eq_l2225_222590

variable {A B C x : ℝ}

theorem sum_of_values_not_satisfying_eq (h : (∀ x, ∃ C, ∃ B, A = 3 ∧ ((x + B) * (A * x + 36) = 3 * (x + C) * (x + 9)) ∧ (x ≠ -9))):
  ∃ y, y = -9 := sorry

end sum_of_values_not_satisfying_eq_l2225_222590


namespace weight_of_white_ring_l2225_222594

def weight_orange := 0.08333333333333333
def weight_purple := 0.3333333333333333
def total_weight := 0.8333333333

def weight_white := 0.41666666663333337

theorem weight_of_white_ring :
  weight_white + weight_orange + weight_purple = total_weight :=
by
  sorry

end weight_of_white_ring_l2225_222594


namespace children_more_than_adults_l2225_222578

-- Definitions based on given conditions
def price_per_child : ℚ := 4.50
def price_per_adult : ℚ := 6.75
def total_receipts : ℚ := 405
def number_of_children : ℕ := 48

-- Goal: Prove the number of children is 20 more than the number of adults.
theorem children_more_than_adults :
  ∃ (A : ℕ), (number_of_children - A) = 20 ∧ (price_per_child * number_of_children) + (price_per_adult * A) = total_receipts := by
  sorry

end children_more_than_adults_l2225_222578


namespace John_finishes_at_610PM_l2225_222598

def TaskTime : Nat := 55
def StartTime : Nat := 14 * 60 + 30 -- 2:30 PM in minutes
def EndSecondTask : Nat := 16 * 60 + 20 -- 4:20 PM in minutes

theorem John_finishes_at_610PM (h1 : TaskTime * 2 = EndSecondTask - StartTime) : 
  (EndSecondTask + TaskTime * 2) = (18 * 60 + 10) :=
by
  sorry

end John_finishes_at_610PM_l2225_222598


namespace balloon_arrangement_count_l2225_222550

theorem balloon_arrangement_count :
  let n := 7
  let l := 2
  let o := 2
  n.factorial / (l.factorial * o.factorial) = 1260 :=
by
  sorry

end balloon_arrangement_count_l2225_222550


namespace mari_buttons_l2225_222591

/-- 
Given that:
1. Sue made 6 buttons
2. Sue made half as many buttons as Kendra.
3. Mari made 4 more than five times as many buttons as Kendra.

We are to prove that Mari made 64 buttons.
-/
theorem mari_buttons (sue_buttons : ℕ) (kendra_buttons : ℕ) (mari_buttons : ℕ) 
  (h1 : sue_buttons = 6)
  (h2 : sue_buttons = kendra_buttons / 2)
  (h3 : mari_buttons = 5 * kendra_buttons + 4) :
  mari_buttons = 64 :=
  sorry

end mari_buttons_l2225_222591


namespace quadratic_equal_roots_iff_a_eq_4_l2225_222502

theorem quadratic_equal_roots_iff_a_eq_4 (a : ℝ) (h : ∃ x : ℝ, (a * x^2 - 4 * x + 1 = 0) ∧ (a * x^2 - 4 * x + 1 = 0)) :
  a = 4 :=
by
  sorry

end quadratic_equal_roots_iff_a_eq_4_l2225_222502


namespace f_at_one_f_increasing_f_range_for_ineq_l2225_222536

-- Define the function f with its properties
noncomputable def f : ℝ → ℝ := sorry

-- Properties of f
axiom f_domain : ∀ x, 0 < x → f x ≠ 0 
axiom f_property_additive : ∀ x y, f (x * y) = f x + f y
axiom f_property_positive : ∀ x, (1 < x) → (0 < f x)
axiom f_property_fract : f (1/3) = -1

-- Proofs to be completed
theorem f_at_one : f 1 = 0 :=
sorry

theorem f_increasing : ∀ (x₁ x₂ : ℝ), (0 < x₁) → (0 < x₂) → (x₁ < x₂) → (f x₁ < f x₂) :=
sorry

theorem f_range_for_ineq : {x : ℝ | 2 < x ∧ x ≤ 9/4} = {x : ℝ | f x - f (x - 2) ≥ 2} :=
sorry

end f_at_one_f_increasing_f_range_for_ineq_l2225_222536


namespace sufficient_not_necessary_condition_l2225_222528

theorem sufficient_not_necessary_condition {x : ℝ} (h : 1 < x ∧ x < 2) : x < 2 ∧ ¬(∀ x, x < 2 → (1 < x ∧ x < 2)) :=
by
  sorry

end sufficient_not_necessary_condition_l2225_222528


namespace factorization_correct_l2225_222540

theorem factorization_correct (x : ℝ) : 
  x^4 - 5*x^2 - 36 = (x^2 + 4)*(x + 3)*(x - 3) :=
sorry

end factorization_correct_l2225_222540


namespace double_sum_evaluation_l2225_222522

theorem double_sum_evaluation :
  ∑' m:ℕ, ∑' n:ℕ, (if m > 0 ∧ n > 0 then 1 / (m * n * (m + n + 2)) else 0) = -Real.pi^2 / 6 :=
sorry

end double_sum_evaluation_l2225_222522


namespace alice_age_l2225_222553

theorem alice_age (a m : ℕ) (h1 : a = m - 18) (h2 : a + m = 50) : a = 16 := by
  sorry

end alice_age_l2225_222553


namespace average_mileage_correct_l2225_222542

def total_distance : ℕ := 150 * 2
def sedan_mileage : ℕ := 25
def hybrid_mileage : ℕ := 50
def sedan_gas_used : ℕ := 150 / sedan_mileage
def hybrid_gas_used : ℕ := 150 / hybrid_mileage
def total_gas_used : ℕ := sedan_gas_used + hybrid_gas_used
def average_gas_mileage : ℚ := total_distance / total_gas_used

theorem average_mileage_correct :
  average_gas_mileage = 33 + 1 / 3 :=
by
  sorry

end average_mileage_correct_l2225_222542


namespace not_p_and_p_or_q_implies_q_l2225_222545

theorem not_p_and_p_or_q_implies_q (p q : Prop) (h1 : ¬ p) (h2 : p ∨ q) : q :=
by
  have h3 : p := sorry
  have h4 : false := sorry
  exact sorry

end not_p_and_p_or_q_implies_q_l2225_222545


namespace range_of_x_l2225_222558

noncomputable def T (x : ℝ) : ℝ := |(2 * x - 1)|

theorem range_of_x (x : ℝ) (h : ∀ a : ℝ, T x ≥ |1 + a| - |2 - a|) : 
  x ≤ -1 ∨ 2 ≤ x :=
by
  sorry

end range_of_x_l2225_222558


namespace average_math_score_l2225_222538

theorem average_math_score (scores : Fin 4 → ℕ) (other_avg : ℕ) (num_students : ℕ) (num_other_students : ℕ)
  (h1 : scores 0 = 90) (h2 : scores 1 = 85) (h3 : scores 2 = 88) (h4 : scores 3 = 80)
  (h5 : other_avg = 82) (h6 : num_students = 30) (h7 : num_other_students = 26) :
  (90 + 85 + 88 + 80 + 26 * 82) / 30 = 82.5 :=
by
  sorry

end average_math_score_l2225_222538


namespace quadratic_inequality_solution_set_l2225_222599

theorem quadratic_inequality_solution_set (p q : ℝ) :
  (∀ x : ℝ, 2 < x ∧ x < 3 → x^2 - p * x - q < 0) →
  p = 5 ∧ q = -6 ∧
  (∀ x : ℝ, - (1 : ℝ) / 2 < x ∧ x < - (1 : ℝ) / 3 → 6 * x^2 + 5 * x + 1 < 0) :=
by
  sorry

end quadratic_inequality_solution_set_l2225_222599


namespace time_for_A_to_complete_race_l2225_222537

theorem time_for_A_to_complete_race
  (V_A V_B : ℝ) (T_A : ℝ)
  (h1 : V_B = 975 / T_A) (h2 : V_B = 2.5) :
  T_A = 390 :=
by
  sorry

end time_for_A_to_complete_race_l2225_222537


namespace empty_bidon_weight_l2225_222589

theorem empty_bidon_weight (B M : ℝ) 
  (h1 : B + M = 34) 
  (h2 : B + M / 2 = 17.5) : 
  B = 1 := 
by {
  -- The proof steps would go here, but we just add sorry
  sorry
}

end empty_bidon_weight_l2225_222589


namespace problem_l2225_222597

noncomputable def f (x : ℝ) : ℝ := 3 / (x + 1)

theorem problem (x : ℝ) (h1 : 3 ≤ x) (h2 : x ≤ 5) :
  (∀ x₁ x₂, 3 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 5 → f x₁ > f x₁) ∧
  f 3 = 3/4 ∧
  f 5 = 1/2 :=
by sorry

end problem_l2225_222597


namespace Joshua_share_correct_l2225_222539

noncomputable def Joshua_share (J : ℝ) : ℝ :=
  3 * J

noncomputable def Jasmine_share (J : ℝ) : ℝ :=
  J / 2

theorem Joshua_share_correct (J : ℝ) (h : J + 3 * J + J / 2 = 120) :
  Joshua_share J = 80.01 := by
  sorry

end Joshua_share_correct_l2225_222539


namespace intersection_complement_l2225_222571

open Set

-- Defining sets A, B and universal set U
def A : Set ℕ := {1, 2, 3, 5, 7}
def B : Set ℕ := {x | 1 < x ∧ x ≤ 6}
def U : Set ℕ := A ∪ B

-- Statement of the proof problem
theorem intersection_complement :
  A ∩ (U \ B) = {1, 7} :=
by
  sorry

end intersection_complement_l2225_222571


namespace solve_x_in_equation_l2225_222566

theorem solve_x_in_equation (x : ℕ) (h : x + (x + 1) + (x + 2) + (x + 3) = 18) : x = 3 :=
by
  sorry

end solve_x_in_equation_l2225_222566


namespace find_all_quartets_l2225_222514

def is_valid_quartet (a b c d : ℕ) : Prop :=
  a + b = c * d ∧
  a * b = c + d ∧
  a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d

theorem find_all_quartets :
  ∀ (a b c d : ℕ),
  is_valid_quartet a b c d ↔
  (a, b, c, d) = (1, 5, 3, 2) ∨ 
  (a, b, c, d) = (1, 5, 2, 3) ∨ 
  (a, b, c, d) = (5, 1, 3, 2) ∨
  (a, b, c, d) = (5, 1, 2, 3) ∨ 
  (a, b, c, d) = (2, 3, 1, 5) ∨ 
  (a, b, c, d) = (3, 2, 1, 5) ∨ 
  (a, b, c, d) = (2, 3, 5, 1) ∨ 
  (a, b, c, d) = (3, 2, 5, 1) := by
  sorry

end find_all_quartets_l2225_222514


namespace arithmetic_mean_l2225_222586

theorem arithmetic_mean (a b c : ℚ) (h₁ : a = 8 / 12) (h₂ : b = 10 / 12) (h₃ : c = 9 / 12) :
  c = (a + b) / 2 :=
by
  sorry

end arithmetic_mean_l2225_222586


namespace remainder_4873_div_29_l2225_222564

theorem remainder_4873_div_29 : 4873 % 29 = 1 := 
by sorry

end remainder_4873_div_29_l2225_222564


namespace train_length_l2225_222556

theorem train_length (speed : ℝ) (time : ℝ) (bridge_length : ℝ) (total_distance : ℝ) (train_length : ℝ) 
  (h1 : speed = 48) (h2 : time = 45) (h3 : bridge_length = 300)
  (h4 : total_distance = speed * time) (h5 : train_length = total_distance - bridge_length) : 
  train_length = 1860 :=
sorry

end train_length_l2225_222556


namespace factorial_mod_5_l2225_222547

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_mod_5 :
  (factorial 1 + factorial 2 + factorial 3 + factorial 4 + factorial 5 +
   factorial 6 + factorial 7 + factorial 8 + factorial 9 + factorial 10) % 5 = 3 :=
by
  sorry

end factorial_mod_5_l2225_222547


namespace part_a_sequence_l2225_222593

def circle_sequence (n m : ℕ) : List ℕ :=
  List.replicate m 1 -- Placeholder: Define the sequence computation properly

theorem part_a_sequence :
  circle_sequence 5 12 = [1, 6, 11, 4, 9, 2, 7, 12, 5, 10, 3, 8, 1] := 
sorry

end part_a_sequence_l2225_222593


namespace difference_of_squares_l2225_222587

theorem difference_of_squares (a b : ℕ) (h1 : a + b = 60) (h2 : a - b = 20) : a^2 - b^2 = 1200 := 
sorry

end difference_of_squares_l2225_222587


namespace sam_distance_walked_l2225_222506

variable (d : ℝ := 40) -- initial distance between Fred and Sam
variable (v_f : ℝ := 4) -- Fred's constant speed in miles per hour
variable (v_s : ℝ := 4) -- Sam's constant speed in miles per hour

theorem sam_distance_walked :
  (d / (v_f + v_s)) * v_s = 20 :=
by
  sorry

end sam_distance_walked_l2225_222506


namespace simplify_sqrt_expression_l2225_222582

theorem simplify_sqrt_expression :
  (Real.sqrt 726 / Real.sqrt 242) + (Real.sqrt 484 / Real.sqrt 121) = Real.sqrt 3 + 2 :=
by
  -- Proof goes here
  sorry

end simplify_sqrt_expression_l2225_222582


namespace find_m_collinear_l2225_222572

structure Point :=
  (x : ℝ)
  (y : ℝ)

def isCollinear (A B C : Point) : Prop :=
  (B.y - A.y) * (C.x - A.x) = (C.y - A.y) * (B.x - A.x)

theorem find_m_collinear :
  ∀ (m : ℝ),
  let A := Point.mk (-2) 3
  let B := Point.mk 3 (-2)
  let C := Point.mk (1 / 2) m
  isCollinear A B C → m = 1 / 2 :=
by
  -- Placeholder for the proof
  sorry

end find_m_collinear_l2225_222572


namespace A_times_B_correct_l2225_222588

noncomputable def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}
noncomputable def B : Set ℝ := {y | y > 1}
noncomputable def A_times_B : Set ℝ := {x | (x ∈ A ∪ B) ∧ ¬(x ∈ A ∩ B)}

theorem A_times_B_correct : A_times_B = {x | (0 ≤ x ∧ x ≤ 1) ∨ x > 2} := 
sorry

end A_times_B_correct_l2225_222588


namespace binomial_odd_sum_l2225_222551

theorem binomial_odd_sum (n : ℕ) (h : (2:ℕ)^(n - 1) = 64) : n = 7 :=
by
  sorry

end binomial_odd_sum_l2225_222551


namespace evaporation_rate_is_200_ml_per_hour_l2225_222563

-- Definitions based on the given conditions
def faucet_drip_rate : ℕ := 40 -- ml per minute
def running_time : ℕ := 9 -- hours
def dumped_water : ℕ := 12000 -- ml (converted from liters)
def water_left : ℕ := 7800 -- ml

-- Alias for total water dripped in running_time
noncomputable def total_dripped_water : ℕ := faucet_drip_rate * 60 * running_time

-- Total water that should have been in the bathtub without evaporation
noncomputable def total_without_evaporation : ℕ := total_dripped_water - dumped_water

-- Water evaporated
noncomputable def evaporated_water : ℕ := total_without_evaporation - water_left

-- Evaporation rate in ml/hour
noncomputable def evaporation_rate : ℕ := evaporated_water / running_time

-- The goal theorem statement
theorem evaporation_rate_is_200_ml_per_hour : evaporation_rate = 200 := by
  -- proof here
  sorry

end evaporation_rate_is_200_ml_per_hour_l2225_222563


namespace largest_prime_factor_always_37_l2225_222535

-- We define the cyclic sequence conditions
def cyclic_shift (seq : List ℕ) : Prop :=
  ∀ i, seq.get! (i % seq.length) % 10 = seq.get! ((i + 1) % seq.length) / 100 ∧
       (seq.get! ((i + 1) % seq.length) / 10 % 10 = seq.get! ((i + 2) % seq.length) % 10) ∧
       (seq.get! ((i + 2) % seq.length) / 10 % 10 = seq.get! ((i + 3) % seq.length) / 100)

-- Summing all elements of a list
def sum (l : List ℕ) : ℕ := l.foldr (· + ·) 0

-- Prove that 37 is always a factor of the sum T
theorem largest_prime_factor_always_37 (seq : List ℕ) (h : cyclic_shift seq) : 
  37 ∣ sum seq := 
sorry

end largest_prime_factor_always_37_l2225_222535


namespace salary_May_l2225_222560

theorem salary_May
  (J F M A M' : ℝ)
  (h1 : (J + F + M + A) / 4 = 8000)
  (h2 : (F + M + A + M') / 4 = 8400)
  (h3 : J = 4900) :
  M' = 6500 :=
  by
  sorry

end salary_May_l2225_222560


namespace min_value_of_g_function_l2225_222509

noncomputable def g (x : Real) := x + (x + 1) / (x^2 + 1) + (x * (x + 3)) / (x^2 + 3) + (3 * (x + 1)) / (x * (x^2 + 3))

theorem min_value_of_g_function : ∀ x : ℝ, x > 0 → g x ≥ 3 := sorry

end min_value_of_g_function_l2225_222509


namespace find_a_l2225_222585

theorem find_a (a x y : ℝ) (h1 : a^(3*x - 1) * 3^(4*y - 3) = 49^x * 27^y) (h2 : x + y = 4) : a = 7 := by
  sorry

end find_a_l2225_222585


namespace compare_negatives_l2225_222552

theorem compare_negatives : -2 < -3 / 2 :=
by sorry

end compare_negatives_l2225_222552


namespace complete_square_equation_l2225_222576

theorem complete_square_equation (b c : ℤ) (h : (x : ℝ) → x^2 - 6 * x + 5 = (x + b)^2 - c) : b + c = 1 :=
by
  sorry  -- This is where the proof would go

end complete_square_equation_l2225_222576


namespace inequality_for_a_l2225_222503

noncomputable def f (x : ℝ) : ℝ :=
  2^x + (Real.log x) / (Real.log 2)

theorem inequality_for_a (n : ℕ) (a : ℝ) (h₁ : 2 < n) (h₂ : 0 < a) (h₃ : 2^a + Real.log a / Real.log 2 = n^2) :
  2 * Real.log n / Real.log 2 > a ∧ a > 2 * Real.log n / Real.log 2 - 1 / n :=
by
  sorry

end inequality_for_a_l2225_222503


namespace fraction_subtraction_l2225_222532

theorem fraction_subtraction :
  ((2 + 4 + 6 : ℚ) / (1 + 3 + 5) - (1 + 3 + 5) / (2 + 4 + 6) = 7 / 12) :=
by
  sorry

end fraction_subtraction_l2225_222532


namespace remainder_when_divided_by_x_minus_2_l2225_222533

-- Define the polynomial
def f (x : ℕ) : ℕ := x^3 - x^2 + 4 * x - 1

-- Statement of the problem: Prove f(2) = 11 using the Remainder Theorem
theorem remainder_when_divided_by_x_minus_2 : f 2 = 11 :=
by
  sorry

end remainder_when_divided_by_x_minus_2_l2225_222533


namespace scientific_notation_101_49_billion_l2225_222579

-- Define the term "one hundred and one point four nine billion"
def billion (n : ℝ) := n * 10^9

-- Axiomatization of the specific number in question
def hundredOnePointFourNineBillion := billion 101.49

-- Theorem stating that the scientific notation for 101.49 billion is 1.0149 × 10^10
theorem scientific_notation_101_49_billion : hundredOnePointFourNineBillion = 1.0149 * 10^10 :=
by
  sorry

end scientific_notation_101_49_billion_l2225_222579


namespace solve_percentage_of_X_in_B_l2225_222596

variable (P : ℝ)

def liquid_X_in_A_percentage : ℝ := 0.008
def mass_of_A : ℝ := 200
def mass_of_B : ℝ := 700
def mixed_solution_percentage_of_X : ℝ := 0.0142
def target_percentage_of_P_in_B : ℝ := 0.01597

theorem solve_percentage_of_X_in_B (P : ℝ) 
  (h1 : mass_of_A * liquid_X_in_A_percentage + mass_of_B * P = (mass_of_A + mass_of_B) * mixed_solution_percentage_of_X) :
  P = target_percentage_of_P_in_B :=
sorry

end solve_percentage_of_X_in_B_l2225_222596


namespace enlarged_sticker_height_l2225_222516

theorem enlarged_sticker_height (original_width original_height new_width : ℕ) 
  (h1 : original_width = 3) 
  (h2 : original_height = 2) 
  (h3 : new_width = 12) : (new_width / original_width) * original_height = 8 := 
by 
  -- Prove the height of the enlarged sticker is 8 inches
  sorry

end enlarged_sticker_height_l2225_222516


namespace lollipops_left_l2225_222562

def problem_conditions : Prop :=
  ∃ (lollipops_bought lollipops_eaten lollipops_left : ℕ),
    lollipops_bought = 12 ∧
    lollipops_eaten = 5 ∧
    lollipops_left = lollipops_bought - lollipops_eaten

theorem lollipops_left (lollipops_bought lollipops_eaten lollipops_left : ℕ) 
  (hb : lollipops_bought = 12) (he : lollipops_eaten = 5) (hl : lollipops_left = lollipops_bought - lollipops_eaten) : 
  lollipops_left = 7 := 
by 
  sorry

end lollipops_left_l2225_222562


namespace problem_solution_l2225_222525

-- Definitions of sets
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3, 4}

-- Complement within U
def complement_U (A : Set ℕ) : Set ℕ := {x ∈ U | x ∉ A}

-- The proof goal
theorem problem_solution : (complement_U A) ∪ B = {2, 3, 4, 5} := by
  sorry

end problem_solution_l2225_222525


namespace find_height_of_tank_A_l2225_222561

noncomputable def height_of_tank_A (C_A C_B h_B ratio V_ratio : ℝ) : ℝ :=
  let r_A := C_A / (2 * Real.pi)
  let r_B := C_B / (2 * Real.pi)
  let V_A := Real.pi * (r_A ^ 2) * ratio
  let V_B := Real.pi * (r_B ^ 2) * h_B
  (V_ratio * V_B) / (Real.pi * (r_A ^ 2))

theorem find_height_of_tank_A :
  height_of_tank_A 8 10 8 10 0.8000000000000001 = 10 :=
by
  sorry

end find_height_of_tank_A_l2225_222561


namespace min_cut_length_no_triangle_l2225_222577

theorem min_cut_length_no_triangle (a b c x : ℝ) 
  (h_y : a = 7) 
  (h_z : b = 24) 
  (h_w : c = 25) 
  (h1 : a - x > 0)
  (h2 : b - x > 0)
  (h3 : c - x > 0)
  (h4 : (a - x) + (b - x) ≤ (c - x)) :
  x = 6 :=
by
  sorry

end min_cut_length_no_triangle_l2225_222577


namespace tyrone_gives_non_integer_marbles_to_eric_l2225_222519

theorem tyrone_gives_non_integer_marbles_to_eric
  (T_init : ℕ) (E_init : ℕ) (x : ℚ)
  (hT : T_init = 120) (hE : E_init = 18)
  (h_eq : T_init - x = 3 * (E_init + x)) :
  ¬ (∃ n : ℕ, x = n) :=
by
  sorry

end tyrone_gives_non_integer_marbles_to_eric_l2225_222519


namespace range_of_z_l2225_222544

theorem range_of_z (x y : ℝ) (h : x^2 / 16 + y^2 / 9 = 1) : -5 ≤ x + y ∧ x + y ≤ 5 :=
sorry

end range_of_z_l2225_222544


namespace worker_new_wage_after_increase_l2225_222581

theorem worker_new_wage_after_increase (initial_wage : ℝ) (increase_percentage : ℝ) (new_wage : ℝ) 
  (h1 : initial_wage = 34) (h2 : increase_percentage = 0.50) 
  (h3 : new_wage = initial_wage + (increase_percentage * initial_wage)) : new_wage = 51 := 
by
  sorry

end worker_new_wage_after_increase_l2225_222581


namespace no_solution_if_and_only_if_zero_l2225_222501

theorem no_solution_if_and_only_if_zero (n : ℝ) :
  ¬(∃ (x y z : ℝ), 2 * n * x + y = 2 ∧ 3 * n * y + z = 3 ∧ x + 2 * n * z = 2) ↔ n = 0 := 
  by
  sorry

end no_solution_if_and_only_if_zero_l2225_222501


namespace find_a_l2225_222507

def F (a : ℚ) (b : ℚ) (c : ℚ) : ℚ := a * b^3 + c

theorem find_a (a : ℚ) : F a 2 3 = F a 3 8 → a = -5 / 19 :=
by
  sorry

end find_a_l2225_222507
