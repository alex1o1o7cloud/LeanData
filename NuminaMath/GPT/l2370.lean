import Mathlib

namespace can_measure_all_weights_l2370_237019

theorem can_measure_all_weights (a b c : ℕ) 
  (h_sum : a + b + c = 10) 
  (h_unique : (a = 1 ∧ b = 2 ∧ c = 7) ∨ (a = 1 ∧ b = 3 ∧ c = 6)) : 
  ∀ w : ℕ, 1 ≤ w ∧ w ≤ 10 → 
    ∃ (k l m : ℤ), w = k * a + l * b + m * c ∨ w = k * -a + l * -b + m * -c :=
  sorry

end can_measure_all_weights_l2370_237019


namespace neg_sin_leq_one_l2370_237073

theorem neg_sin_leq_one (p : Prop) :
  (∀ x : ℝ, Real.sin x ≤ 1) → (¬(∀ x : ℝ, Real.sin x ≤ 1) ↔ ∃ x : ℝ, Real.sin x > 1) :=
by
  sorry

end neg_sin_leq_one_l2370_237073


namespace ways_to_distribute_balls_l2370_237072

theorem ways_to_distribute_balls :
  let balls : Finset ℕ := {0, 1, 2, 3, 4, 5, 6}
  let boxes : Finset ℕ := {0, 1, 2, 3}
  let choose_distinct (n k : ℕ) : ℕ := Nat.choose n k
  let distribution_patterns : List (ℕ × ℕ × ℕ × ℕ) := 
    [(6,0,0,0), (5,1,0,0), (4,2,0,0), (4,1,1,0), 
     (3,3,0,0), (3,2,1,0), (3,1,1,1), (2,2,2,0), (2,2,1,1)]
  let ways_to_pattern (pattern : ℕ × ℕ × ℕ × ℕ) : ℕ :=
    match pattern with
    | (6,0,0,0) => 1
    | (5,1,0,0) => choose_distinct 6 5
    | (4,2,0,0) => choose_distinct 6 4 * choose_distinct 2 2
    | (4,1,1,0) => choose_distinct 6 4
    | (3,3,0,0) => choose_distinct 6 3 * choose_distinct 3 3 / 2
    | (3,2,1,0) => choose_distinct 6 3 * choose_distinct 3 2 * choose_distinct 1 1
    | (3,1,1,1) => choose_distinct 6 3
    | (2,2,2,0) => choose_distinct 6 2 * choose_distinct 4 2 * choose_distinct 2 2 / 6
    | (2,2,1,1) => choose_distinct 6 2 * choose_distinct 4 2 / 2
    | _ => 0
  let total_ways : ℕ := distribution_patterns.foldl (λ acc x => acc + ways_to_pattern x) 0
  total_ways = 182 := by
  sorry

end ways_to_distribute_balls_l2370_237072


namespace point_in_second_quadrant_l2370_237075

-- Definitions for the coordinates of the points
def A : ℤ × ℤ := (3, 2)
def B : ℤ × ℤ := (-3, -2)
def C : ℤ × ℤ := (3, -2)
def D : ℤ × ℤ := (-3, 2)

-- Definition for the second quadrant condition
def isSecondQuadrant (p : ℤ × ℤ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

-- The theorem we need to prove
theorem point_in_second_quadrant : isSecondQuadrant D :=
by
  sorry

end point_in_second_quadrant_l2370_237075


namespace sum_of_coordinates_D_l2370_237058

theorem sum_of_coordinates_D (M C D : ℝ × ℝ)
  (h1 : M = (5, 5))
  (h2 : C = (10, 10))
  (h3 : M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)) :
  D.1 + D.2 = 0 := 
sorry

end sum_of_coordinates_D_l2370_237058


namespace tan_sum_inequality_l2370_237027

noncomputable def pi : ℝ := Real.pi

theorem tan_sum_inequality (x α : ℝ) (hx1 : 0 ≤ x) (hx2 : x ≤ pi / 2) (hα1 : pi / 6 < α) (hα2 : α < pi / 3) :
  Real.tan (pi * (Real.sin x) / (4 * Real.sin α)) + Real.tan (pi * (Real.cos x) / (4 * Real.cos α)) > 1 :=
by
  sorry

end tan_sum_inequality_l2370_237027


namespace price_per_foot_of_fencing_l2370_237025

theorem price_per_foot_of_fencing
  (area : ℝ) (total_cost : ℝ) (price_per_foot : ℝ)
  (h1 : area = 36) (h2 : total_cost = 1392) :
  price_per_foot = 58 :=
by
  sorry

end price_per_foot_of_fencing_l2370_237025


namespace sufficient_but_not_necessary_condition_not_neccessary_condition_l2370_237065

theorem sufficient_but_not_necessary_condition (x y : ℝ) :
  ((x + 3)^2 + (y - 4)^2 = 0) → ((x + 3) * (y - 4) = 0) :=
by { sorry }

theorem not_neccessary_condition (x y : ℝ) :
  ((x + 3) * (y - 4) = 0) ↔ ((x + 3)^2 + (y - 4)^2 = 0) :=
by { sorry }

end sufficient_but_not_necessary_condition_not_neccessary_condition_l2370_237065


namespace cos_five_pi_over_six_l2370_237009

theorem cos_five_pi_over_six :
  Real.cos (5 * Real.pi / 6) = -(Real.sqrt 3 / 2) :=
sorry

end cos_five_pi_over_six_l2370_237009


namespace KHSO4_formed_l2370_237080

-- Define the reaction condition and result using moles
def KOH_moles : ℕ := 2
def H2SO4_moles : ℕ := 2

-- The balanced chemical reaction in terms of moles
-- 1 mole of KOH reacts with 1 mole of H2SO4 to produce 
-- 1 mole of KHSO4
def react (koh : ℕ) (h2so4 : ℕ) : ℕ := 
  -- stoichiometry 1:1 ratio of KOH and H2SO4 to KHSO4
  if koh ≤ h2so4 then koh else h2so4

-- The proof statement that verifies the expected number of moles of KHSO4
theorem KHSO4_formed (koh : ℕ) (h2so4 : ℕ) (hrs : react koh h2so4 = koh) : 
  koh = KOH_moles → h2so4 = H2SO4_moles → react koh h2so4 = 2 := 
by
  intros 
  sorry

end KHSO4_formed_l2370_237080


namespace find_integer_l2370_237082

theorem find_integer (n : ℕ) (h1 : 0 < n) (h2 : 200 % n = 2) (h3 : 398 % n = 2) : n = 6 :=
sorry

end find_integer_l2370_237082


namespace tan_150_eq_neg_inv_sqrt_3_l2370_237045

theorem tan_150_eq_neg_inv_sqrt_3 : Real.tan (150 * Real.pi / 180) = -1 / Real.sqrt 3 := by
  -- Using the given conditions
  sorry

end tan_150_eq_neg_inv_sqrt_3_l2370_237045


namespace solution_set_of_inequality_l2370_237007

theorem solution_set_of_inequality (x : ℝ) : (0 < x ∧ x < 1/3) ↔ (1/x > 3) := 
sorry

end solution_set_of_inequality_l2370_237007


namespace determine_f_101_l2370_237043

theorem determine_f_101 (f : ℕ → ℕ) (h : ∀ m n : ℕ, m * n + 1 ∣ f m * f n + 1) : 
  ∃ k : ℕ, k % 2 = 1 ∧ f 101 = 101 ^ k :=
sorry

end determine_f_101_l2370_237043


namespace tiles_needed_correct_l2370_237040

noncomputable def tiles_needed (floor_length : ℝ) (floor_width : ℝ) (tile_length_inch : ℝ) (tile_width_inch : ℝ) (border_width : ℝ) : ℝ :=
  let tile_length := tile_length_inch / 12
  let tile_width := tile_width_inch / 12
  let main_length := floor_length - 2 * border_width
  let main_width := floor_width - 2 * border_width
  let main_area := main_length * main_width
  let tile_area := tile_length * tile_width
  main_area / tile_area

theorem tiles_needed_correct :
  tiles_needed 15 20 3 9 1 = 1248 := 
by 
  sorry -- Proof skipped.

end tiles_needed_correct_l2370_237040


namespace problem_statement_l2370_237047

theorem problem_statement (x : ℝ) (h : x = -3) :
  (5 + x * (5 + x) - 5^2) / (x - 5 + x^2) = -26 := by
  rw [h]
  sorry

end problem_statement_l2370_237047


namespace four_positive_reals_inequality_l2370_237014

theorem four_positive_reals_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  a^3 + b^3 + c^3 + d^3 ≥ a^2 * b + b^2 * c + c^2 * d + d^2 * a :=
sorry

end four_positive_reals_inequality_l2370_237014


namespace isosceles_triangle_perimeter_l2370_237017

theorem isosceles_triangle_perimeter (a b : ℝ) (h₁ : a = 6) (h₂ : b = 5) :
  ∃ p : ℝ, (p = a + a + b ∨ p = b + b + a) ∧ (p = 16 ∨ p = 17) :=
by
  sorry

end isosceles_triangle_perimeter_l2370_237017


namespace ratio_of_numbers_l2370_237077

theorem ratio_of_numbers (A B : ℕ) (hA : A = 45) (hLCM : Nat.lcm A B = 180) : A / Nat.lcm A B = 45 / 4 :=
by
  sorry

end ratio_of_numbers_l2370_237077


namespace devin_basketball_chances_l2370_237060

theorem devin_basketball_chances 
  (initial_chances : ℝ := 0.1) 
  (base_height : ℕ := 66) 
  (chance_increase_per_inch : ℝ := 0.1)
  (initial_height : ℕ := 65) 
  (growth : ℕ := 3) :
  initial_chances + (growth + initial_height - base_height) * chance_increase_per_inch = 0.3 := 
by 
  sorry

end devin_basketball_chances_l2370_237060


namespace bill_weight_training_l2370_237064

theorem bill_weight_training (jugs : ℕ) (gallons_per_jug : ℝ) (percent_filled : ℝ) (density : ℝ) 
  (h_jugs : jugs = 2)
  (h_gallons_per_jug : gallons_per_jug = 2)
  (h_percent_filled : percent_filled = 0.70)
  (h_density : density = 5) :
  jugs * gallons_per_jug * percent_filled * density = 14 := 
by
  subst h_jugs
  subst h_gallons_per_jug
  subst h_percent_filled
  subst h_density
  norm_num
  done

end bill_weight_training_l2370_237064


namespace problem_1_problem_2_l2370_237094

noncomputable def is_positive_real (x : ℝ) : Prop := x > 0

theorem problem_1 (a b : ℝ) (ha : is_positive_real a) (hb : is_positive_real b)
  (h : 1 / a + 1 / b = 2 * Real.sqrt 2) : 
  a^2 + b^2 ≥ 1 := by
  sorry

theorem problem_2 (a b : ℝ) (ha : is_positive_real a) (hb : is_positive_real b)
  (h : 1 / a + 1 / b = 2 * Real.sqrt 2) (h_extra : (a - b)^2 ≥ 4 * (a * b)^3) : 
  a * b = 1 := by
  sorry

end problem_1_problem_2_l2370_237094


namespace eggs_per_hen_l2370_237081

theorem eggs_per_hen (total_eggs : Float) (num_hens : Float) (h1 : total_eggs = 303.0) (h2 : num_hens = 28.0) : 
  total_eggs / num_hens = 10.821428571428571 :=
by 
  sorry

end eggs_per_hen_l2370_237081


namespace oblique_projection_intuitive_diagrams_correct_l2370_237056

-- Definitions based on conditions
structure ObliqueProjection :=
  (lines_parallel_x_axis_same_length : Prop)
  (lines_parallel_y_axis_halved_length : Prop)
  (perpendicular_relationship_becomes_45_angle : Prop)

-- Definitions based on statements
def intuitive_triangle_projection (P : ObliqueProjection) : Prop :=
  P.lines_parallel_x_axis_same_length ∧ 
  P.lines_parallel_y_axis_halved_length ∧ 
  P.perpendicular_relationship_becomes_45_angle

def intuitive_parallelogram_projection (P : ObliqueProjection) : Prop := 
  P.lines_parallel_x_axis_same_length ∧ 
  P.lines_parallel_y_axis_halved_length ∧ 
  P.perpendicular_relationship_becomes_45_angle

def intuitive_square_projection (P : ObliqueProjection) : Prop := 
  P.lines_parallel_x_axis_same_length ∧ 
  P.lines_parallel_y_axis_halved_length ∧ 
  P.perpendicular_relationship_becomes_45_angle

def intuitive_rhombus_projection (P : ObliqueProjection) : Prop := 
  P.lines_parallel_x_axis_same_length ∧ 
  P.lines_parallel_y_axis_halved_length ∧ 
  P.perpendicular_relationship_becomes_45_angle

-- Theorem stating which intuitive diagrams are correctly represented under the oblique projection method.
theorem oblique_projection_intuitive_diagrams_correct : 
  ∀ (P : ObliqueProjection), 
    intuitive_triangle_projection P ∧ 
    intuitive_parallelogram_projection P ∧
    ¬intuitive_square_projection P ∧
    ¬intuitive_rhombus_projection P :=
by 
  sorry

end oblique_projection_intuitive_diagrams_correct_l2370_237056


namespace school_supply_cost_l2370_237013

theorem school_supply_cost (num_students : ℕ) (pens_per_student : ℕ) (pen_cost : ℝ) 
  (notebooks_per_student : ℕ) (notebook_cost : ℝ) 
  (binders_per_student : ℕ) (binder_cost : ℝ) 
  (highlighters_per_student : ℕ) (highlighter_cost : ℝ) 
  (teacher_discount : ℝ) : 
  num_students = 30 →
  pens_per_student = 5 →
  pen_cost = 0.50 →
  notebooks_per_student = 3 →
  notebook_cost = 1.25 →
  binders_per_student = 1 →
  binder_cost = 4.25 →
  highlighters_per_student = 2 →
  highlighter_cost = 0.75 →
  teacher_discount = 100 →
  (num_students * 
    (pens_per_student * pen_cost + notebooks_per_student * notebook_cost + 
    binders_per_student * binder_cost + highlighters_per_student * highlighter_cost) - 
    teacher_discount) = 260 :=
by
  intros _ _ _ _ _ _ _ _ _ _

  -- Sorry added to skip the proof
  sorry

end school_supply_cost_l2370_237013


namespace evaluate_h_j_l2370_237032

def h (x : ℝ) : ℝ := 3 * x - 4
def j (x : ℝ) : ℝ := x - 2

theorem evaluate_h_j : h (2 + j 3) = 5 := by
  sorry

end evaluate_h_j_l2370_237032


namespace remainder_sum_of_squares_25_mod_6_l2370_237031

def sum_of_squares (n : ℕ) : ℕ :=
  n * (n + 1) * (2 * n + 1) / 6

theorem remainder_sum_of_squares_25_mod_6 :
  (sum_of_squares 25) % 6 = 5 :=
by
  sorry

end remainder_sum_of_squares_25_mod_6_l2370_237031


namespace find_x_in_terms_of_z_l2370_237063

variable (z : ℝ)
variable (x y : ℝ)

theorem find_x_in_terms_of_z (h1 : 0.35 * (400 + y) = 0.20 * x) 
                             (h2 : x = 2 * z^2) 
                             (h3 : y = 3 * z - 5) : 
  x = 2 * z^2 :=
by
  exact h2

end find_x_in_terms_of_z_l2370_237063


namespace solve_sum_of_squares_l2370_237069

theorem solve_sum_of_squares
  (k l m n a b c : ℕ)
  (h_cond1 : k ≠ l ∧ k ≠ m ∧ k ≠ n ∧ l ≠ m ∧ l ≠ n ∧ m ≠ n)
  (h_cond2 : a * k^2 - b * k + c = 0)
  (h_cond3 : a * l^2 - b * l + c = 0)
  (h_cond4 : c * m^2 - 16 * b * m + 256 * a = 0)
  (h_cond5 : c * n^2 - 16 * b * n + 256 * a = 0) :
  k^2 + l^2 + m^2 + n^2 = 325 :=
by
  sorry

end solve_sum_of_squares_l2370_237069


namespace min_distance_point_to_line_l2370_237052

theorem min_distance_point_to_line :
    ∀ (x y : ℝ), (x^2 + y^2 - 6 * x - 4 * y + 12 = 0) -> 
    (3 * x + 4 * y - 2 = 0) -> 
    ∃ d: ℝ, d = 2 :=
by sorry

end min_distance_point_to_line_l2370_237052


namespace sum_of_variables_l2370_237049

theorem sum_of_variables (x y z : ℝ) (h₁ : x + y = 1) (h₂ : y + z = 1) (h₃ : z + x = 1) : x + y + z = 3 / 2 := 
sorry

end sum_of_variables_l2370_237049


namespace simplify_expression_l2370_237039

theorem simplify_expression :
  (3 / 4 : ℚ) * 60 - (8 / 5 : ℚ) * 60 + x = 12 → x = 63 :=
by
  intro h
  sorry

end simplify_expression_l2370_237039


namespace total_people_attended_l2370_237068

theorem total_people_attended (A C : ℕ) (ticket_price_adult ticket_price_child : ℕ) (total_receipts : ℕ) 
  (number_of_children : ℕ) (h_ticket_prices : ticket_price_adult = 60 ∧ ticket_price_child = 25)
  (h_total_receipts : total_receipts = 140 * 100) (h_children : C = 80) 
  (h_equation : ticket_price_adult * A + ticket_price_child * C = total_receipts) : 
  A + C = 280 :=
by
  sorry

end total_people_attended_l2370_237068


namespace intersection_A_B_l2370_237026

def A := {x : ℤ | ∃ k : ℤ, x = 2 * k + 1}
def B := {x : ℤ | 0 < x ∧ x < 5}

theorem intersection_A_B : A ∩ B = {1, 3} :=
by
  sorry

end intersection_A_B_l2370_237026


namespace beans_in_jar_l2370_237087

theorem beans_in_jar (B : ℕ) 
  (h1 : B / 4 = number_of_red_beans)
  (h2 : number_of_red_beans = B / 4)
  (h3 : number_of_white_beans = (B * 3 / 4) / 3)
  (h4 : number_of_white_beans = B / 4)
  (h5 : number_of_remaining_beans_after_white = B / 2)
  (h6 : 143 = B / 4):
  B = 572 :=
by
  sorry

end beans_in_jar_l2370_237087


namespace coeff_a_zero_l2370_237018

theorem coeff_a_zero
  (a b c : ℝ)
  (h : ∀ p : ℝ, 0 < p → ∀ (x : ℝ), (a * x^2 + b * x + c + p = 0) → x > 0) :
  a = 0 :=
sorry

end coeff_a_zero_l2370_237018


namespace routes_from_A_to_B_l2370_237048

-- Definitions based on conditions given in the problem
variables (A B C D E F : Type)
variables (AB AD AE BC BD CD DE EF : Prop) 

-- Theorem statement
theorem routes_from_A_to_B (route_criteria : AB ∧ AD ∧ AE ∧ BC ∧ BD ∧ CD ∧ DE ∧ EF)
  : ∃ n : ℕ, n = 16 :=
sorry

end routes_from_A_to_B_l2370_237048


namespace lowest_score_for_average_l2370_237024

theorem lowest_score_for_average
  (score1 score2 score3 : ℕ)
  (h1 : score1 = 81)
  (h2 : score2 = 72)
  (h3 : score3 = 93)
  (max_score : ℕ := 100)
  (desired_average : ℕ := 86)
  (number_of_exams : ℕ := 5) :
  ∃ x y : ℕ, x ≤ 100 ∧ y ≤ 100 ∧ (score1 + score2 + score3 + x + y) / number_of_exams = desired_average ∧ min x y = 84 :=
by
  sorry

end lowest_score_for_average_l2370_237024


namespace ellipse_eccentricity_l2370_237016

-- Define the geometric sequence condition and the ellipse properties
theorem ellipse_eccentricity :
  ∀ (a b c e : ℝ), 
  (b^2 = a * c) ∧ (a^2 - c^2 = b^2) ∧ (e = c / a) ∧ (0 < e ∧ e < 1) →
  e = (Real.sqrt 5 - 1) / 2 := 
by 
  sorry

end ellipse_eccentricity_l2370_237016


namespace total_number_of_students_l2370_237067

theorem total_number_of_students 
  (b g : ℕ) 
  (ratio_condition : 5 * g = 8 * b) 
  (girls_count : g = 160) : 
  b + g = 260 := by
  sorry

end total_number_of_students_l2370_237067


namespace number_equals_fifty_l2370_237059

def thirty_percent_less_than_ninety : ℝ := 0.7 * 90

theorem number_equals_fifty (x : ℝ) (h : (5 / 4) * x = thirty_percent_less_than_ninety) : x = 50 :=
by
  sorry

end number_equals_fifty_l2370_237059


namespace instantaneous_velocity_at_t2_l2370_237066

def displacement (t : ℝ) : ℝ := 2 * (1 - t) ^ 2

theorem instantaneous_velocity_at_t2 :
  (deriv displacement 2) = 4 :=
by
  sorry

end instantaneous_velocity_at_t2_l2370_237066


namespace tan_alpha_fraction_l2370_237042

theorem tan_alpha_fraction (α : ℝ) (hα1 : 0 < α ∧ α < (Real.pi / 2)) (hα2 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) :
  Real.tan α = Real.sqrt 15 / 15 :=
by
  sorry

end tan_alpha_fraction_l2370_237042


namespace mark_has_24_dollars_l2370_237071

theorem mark_has_24_dollars
  (small_bag_cost : ℕ := 4)
  (small_bag_balloons : ℕ := 50)
  (medium_bag_cost : ℕ := 6)
  (medium_bag_balloons : ℕ := 75)
  (large_bag_cost : ℕ := 12)
  (large_bag_balloons : ℕ := 200)
  (total_balloons : ℕ := 400) :
  total_balloons / large_bag_balloons = 2 ∧ 2 * large_bag_cost = 24 := by
  sorry

end mark_has_24_dollars_l2370_237071


namespace angle_B_l2370_237008

open Set

variables {Point Line : Type}

variable (l m n p : Line)
variable (A B C D : Point)
variable (angle : Point → Point → Point → ℝ)

-- Definitions of the conditions
def parallel (x y : Line) : Prop := sorry
def intersects (x y : Line) (P : Point) : Prop := sorry
def measure_angle (P Q R : Point) : ℝ := sorry

-- Assumptions based on conditions
axiom parallel_lm : parallel l m
axiom intersection_n_l : intersects n l A
axiom angle_A : measure_angle B A D = 140
axiom intersection_p_m : intersects p m C
axiom angle_C : measure_angle A C B = 70
axiom intersection_p_l : intersects p l D
axiom not_parallel_np : ¬ parallel n p

-- Proof goal
theorem angle_B : measure_angle C B D = 140 := sorry

end angle_B_l2370_237008


namespace area_of_inscribed_octagon_l2370_237055

-- Define the given conditions and required proof
theorem area_of_inscribed_octagon (r : ℝ) (h : π * r^2 = 400 * π) :
  let A := r^2 * (1 + Real.sqrt 2)
  A = 20^2 * (1 + Real.sqrt 2) :=
by 
  sorry

end area_of_inscribed_octagon_l2370_237055


namespace geometric_sequence_a4_l2370_237083

theorem geometric_sequence_a4 (a : ℕ → ℝ) (q : ℝ) 
    (h_geom : ∀ n, a (n + 1) = a n * q)
    (h_a2 : a 2 = 1)
    (h_q : q = 2) : 
    a 4 = 4 :=
by
  -- Skip the proof as instructed
  sorry

end geometric_sequence_a4_l2370_237083


namespace lawn_length_l2370_237078

-- Defining the main conditions
def area : ℕ := 20
def width : ℕ := 5

-- The proof statement (goal)
theorem lawn_length : (area / width) = 4 := by
  sorry

end lawn_length_l2370_237078


namespace darnell_phone_minutes_l2370_237029

theorem darnell_phone_minutes
  (unlimited_cost : ℕ)
  (text_cost : ℕ)
  (call_cost : ℕ)
  (texts_per_dollar : ℕ)
  (minutes_per_dollar : ℕ)
  (total_texts : ℕ)
  (cost_difference : ℕ)
  (alternative_total_cost : ℕ)
  (M : ℕ)
  (text_cost_condition : unlimited_cost - cost_difference = alternative_total_cost)
  (text_formula : M / minutes_per_dollar * call_cost + total_texts / texts_per_dollar * text_cost = alternative_total_cost)
  : M = 60 :=
sorry

end darnell_phone_minutes_l2370_237029


namespace find_p_fifth_plus_3_l2370_237004

theorem find_p_fifth_plus_3 (p : ℕ) (hp : Nat.Prime p) (h : Nat.Prime (p^4 + 3)) :
  p^5 + 3 = 35 :=
sorry

end find_p_fifth_plus_3_l2370_237004


namespace smallest_a_such_that_sqrt_50a_is_integer_l2370_237050

theorem smallest_a_such_that_sqrt_50a_is_integer : ∃ a : ℕ, (∀ b : ℕ, (b > 0 ∧ (∃ k : ℕ, 50 * b = k^2)) → (a ≤ b)) ∧ (∃ k : ℕ, 50 * a = k^2) ∧ a = 2 := 
by
  sorry

end smallest_a_such_that_sqrt_50a_is_integer_l2370_237050


namespace third_stack_shorter_by_five_l2370_237092

theorem third_stack_shorter_by_five
    (first_stack second_stack third_stack fourth_stack : ℕ)
    (h1 : first_stack = 5)
    (h2 : second_stack = first_stack + 2)
    (h3 : fourth_stack = third_stack + 5)
    (h4 : first_stack + second_stack + third_stack + fourth_stack = 21) :
    second_stack - third_stack = 5 :=
by
  sorry

end third_stack_shorter_by_five_l2370_237092


namespace parallel_lines_l2370_237005

theorem parallel_lines (a : ℝ) :
  (∀ x y : ℝ, a * x + 2 * y + a + 3 = 0) ∧ (∀ x y : ℝ, x + (a + 1) * y + 4 = 0) 
  → a = -2 :=
sorry

end parallel_lines_l2370_237005


namespace symmetric_circle_l2370_237028

theorem symmetric_circle
    (x y : ℝ)
    (circle_eq : x^2 + y^2 + 4 * x - 1 = 0) :
    (x - 2)^2 + y^2 = 5 :=
sorry

end symmetric_circle_l2370_237028


namespace percentage_paid_l2370_237089

/-- 
Given the marked price is 80% of the suggested retail price,
and Alice paid 60% of the marked price,
prove that the percentage of the suggested retail price Alice paid is 48%.
-/
theorem percentage_paid (P : ℝ) (MP : ℝ) (price_paid : ℝ)
  (h1 : MP = 0.80 * P)
  (h2 : price_paid = 0.60 * MP) :
  (price_paid / P) * 100 = 48 := 
sorry

end percentage_paid_l2370_237089


namespace total_pay_is_correct_l2370_237044

-- Define the weekly pay for employee B
def pay_B : ℝ := 228

-- Define the multiplier for employee A's pay relative to employee B's pay
def multiplier_A : ℝ := 1.5

-- Define the weekly pay for employee A
def pay_A : ℝ := multiplier_A * pay_B

-- Define the total weekly pay for both employees
def total_pay : ℝ := pay_A + pay_B

-- Prove the total pay
theorem total_pay_is_correct : total_pay = 570 := by
  -- Use the definitions and compute the total pay
  sorry

end total_pay_is_correct_l2370_237044


namespace smallest_x_for_multiple_l2370_237003

theorem smallest_x_for_multiple (x : ℕ) (h : x > 0) :
  (450 * x) % 500 = 0 ↔ x = 10 := by
  sorry

end smallest_x_for_multiple_l2370_237003


namespace problem_l2370_237085

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos (x - Real.pi / 2)

theorem problem 
: (∃ T > 0, ∀ x, f (x + T) = f x) ∧ (∃ c, c = (Real.pi / 2) ∧ f c = 0) → (T = Real.pi ∧ c = (Real.pi / 2)) :=
sorry

end problem_l2370_237085


namespace dot_product_AB_BC_l2370_237074

variable (AB AC : ℝ × ℝ)

def BC (AB AC : ℝ × ℝ) : ℝ × ℝ := (AC.1 - AB.1, AC.2 - AB.2)

def dot_product (u v : ℝ × ℝ) : ℝ := (u.1 * v.1) + (u.2 * v.2)

theorem dot_product_AB_BC :
  ∀ (AB AC : ℝ × ℝ), AB = (2, 3) → AC = (3, 4) →
  dot_product AB (BC AB AC) = 5 :=
by
  intros
  unfold BC
  unfold dot_product
  sorry

end dot_product_AB_BC_l2370_237074


namespace element_in_set_l2370_237001

open Set

noncomputable def A : Set ℝ := { x | x < 2 * Real.sqrt 3 }
def a : ℝ := 2

theorem element_in_set : a ∈ A := by
  sorry

end element_in_set_l2370_237001


namespace amount_spent_on_marbles_l2370_237090

-- Definitions of conditions
def cost_of_football : ℝ := 5.71
def total_spent_on_toys : ℝ := 12.30

-- Theorem statement
theorem amount_spent_on_marbles : (total_spent_on_toys - cost_of_football) = 6.59 :=
by
  sorry

end amount_spent_on_marbles_l2370_237090


namespace andy_tomatoes_l2370_237054

theorem andy_tomatoes (P : ℕ) (h1 : ∀ P, 7 * P / 3 = 42) : P = 18 := by
  sorry

end andy_tomatoes_l2370_237054


namespace emily_required_sixth_score_is_99_l2370_237033

/-- Emily's quiz scores and the required mean score -/
def emily_scores : List ℝ := [85, 90, 88, 92, 98]
def required_mean_score : ℝ := 92

/-- The function to calculate the required sixth quiz score for Emily -/
def required_sixth_score (scores : List ℝ) (mean : ℝ) : ℝ :=
  let sum_current := scores.sum
  let total_required := mean * (scores.length + 1)
  total_required - sum_current

/-- Emily needs to score 99 on her sixth quiz for an average of 92 -/
theorem emily_required_sixth_score_is_99 : 
  required_sixth_score emily_scores required_mean_score = 99 :=
by
  sorry

end emily_required_sixth_score_is_99_l2370_237033


namespace jill_trips_to_fill_tank_l2370_237037

def tank_capacity : ℕ := 600
def bucket_volume : ℕ := 5
def jack_buckets_per_trip : ℕ := 2
def jill_buckets_per_trip : ℕ := 1
def jack_to_jill_trip_ratio : ℕ := 3 / 2

theorem jill_trips_to_fill_tank : (tank_capacity / bucket_volume) = 120 → 
                                   ((jack_to_jill_trip_ratio * jack_buckets_per_trip) + 2 * jill_buckets_per_trip) = 8 →
                                   15 * 2 = 30 :=
by
  intros h1 h2
  sorry

end jill_trips_to_fill_tank_l2370_237037


namespace sequence_general_formula_l2370_237020

theorem sequence_general_formula (a : ℕ → ℝ) (h₁ : a 1 = 1)
  (h₂ : ∀ n, a (n + 1) = a n / (1 + 2 * a n)) :
  ∀ n, a n = 1 / (2 * n - 1) :=
sorry

end sequence_general_formula_l2370_237020


namespace find_f_neg_3_l2370_237057

theorem find_f_neg_3
    (a : ℝ)
    (f : ℝ → ℝ)
    (h : ∀ x, f x = a^2 * x^3 + a * Real.sin x + abs x + 1)
    (h_f3 : f 3 = 5) :
    f (-3) = 3 :=
by
    sorry

end find_f_neg_3_l2370_237057


namespace ratio_of_y_to_x_l2370_237095

theorem ratio_of_y_to_x (c x y : ℝ) (hx : x = 0.90 * c) (hy : y = 1.20 * c) :
  y / x = 4 / 3 := 
sorry

end ratio_of_y_to_x_l2370_237095


namespace solve_frac_eq_l2370_237051

theorem solve_frac_eq (x : ℝ) (h : 3 - 5 / x + 2 / (x^2) = 0) : 
  ∃ y : ℝ, (y = 3 / x ∧ (y = 9 / 2 ∨ y = 3)) :=
sorry

end solve_frac_eq_l2370_237051


namespace part_a_39x55_5x11_l2370_237006

theorem part_a_39x55_5x11 :
  ¬ (∃ (a1 a2 b1 b2 : ℕ), 
    39 = 5 * a1 + 11 * b1 ∧ 
    55 = 5 * a2 + 11 * b2) := 
  by sorry

end part_a_39x55_5x11_l2370_237006


namespace cat_moves_on_circular_arc_l2370_237091

theorem cat_moves_on_circular_arc (L : ℝ) (x y : ℝ)
  (h : x^2 + y^2 = L^2) :
  (x / 2)^2 + (y / 2)^2 = (L / 2)^2 :=
  by sorry

end cat_moves_on_circular_arc_l2370_237091


namespace derivative_f_l2370_237099

noncomputable def f (x : ℝ) : ℝ := 1 + Real.cos x

theorem derivative_f (x : ℝ) : deriv f x = -Real.sin x := 
by 
  sorry

end derivative_f_l2370_237099


namespace rhombus_diagonal_l2370_237000

/-- Given a rhombus with one diagonal being 11 cm and the area of the rhombus being 88 cm²,
prove that the length of the other diagonal is 16 cm. -/
theorem rhombus_diagonal 
  (d1 : ℝ) (d2 : ℝ) (area : ℝ)
  (h_d1 : d1 = 11)
  (h_area : area = 88)
  (h_area_eq : area = (d1 * d2) / 2) : d2 = 16 :=
sorry

end rhombus_diagonal_l2370_237000


namespace smaller_angle_between_hands_at_3_40_pm_larger_angle_between_hands_at_3_40_pm_l2370_237079

def degree_movement_per_minute_of_minute_hand : ℝ := 6
def degree_movement_per_hour_of_hour_hand : ℝ := 30
def degree_movement_per_minute_of_hour_hand : ℝ := 0.5

def minute_position_at_3_40_pm : ℝ := 40 * degree_movement_per_minute_of_minute_hand
def hour_position_at_3_40_pm : ℝ := 3 * degree_movement_per_hour_of_hour_hand + 40 * degree_movement_per_minute_of_hour_hand

def clockwise_angle_between_hands_at_3_40_pm : ℝ := minute_position_at_3_40_pm - hour_position_at_3_40_pm
def counterclockwise_angle_between_hands_at_3_40_pm : ℝ := 360 - clockwise_angle_between_hands_at_3_40_pm

theorem smaller_angle_between_hands_at_3_40_pm : clockwise_angle_between_hands_at_3_40_pm = 130.0 := 
by
  sorry

theorem larger_angle_between_hands_at_3_40_pm : counterclockwise_angle_between_hands_at_3_40_pm = 230.0 := 
by
  sorry

end smaller_angle_between_hands_at_3_40_pm_larger_angle_between_hands_at_3_40_pm_l2370_237079


namespace eccentricity_of_ellipse_l2370_237038

theorem eccentricity_of_ellipse (a c : ℝ) (h : 4 * a = 7 * 2 * (a - c)) : 
    c / a = 5 / 7 :=
by {
  sorry
}

end eccentricity_of_ellipse_l2370_237038


namespace div_poly_iff_l2370_237084

-- Definitions from conditions
def P (x : ℂ) (n : ℕ) := x^(4*n) + x^(3*n) + x^(2*n) + x^n + 1
def Q (x : ℂ) := x^4 + x^3 + x^2 + x + 1

-- The main theorem stating the problem
theorem div_poly_iff (n : ℕ) : 
  ∀ x : ℂ, (P x n) ∣ (Q x) ↔ n % 5 ≠ 0 :=
by sorry

end div_poly_iff_l2370_237084


namespace probability_perfect_square_l2370_237053

theorem probability_perfect_square (choose_numbers : Finset (Fin 49)) (ticket : Finset (Fin 49))
  (h_choose_size : choose_numbers.card = 6) 
  (h_ticket_size : ticket.card = 6)
  (h_choose_square : ∃ (n : ℕ), (choose_numbers.prod id = n * n))
  (h_ticket_square : ∃ (m : ℕ), (ticket.prod id = m * m)) :
  ∃ T, (1 / T = 1 / T) :=
by
  sorry

end probability_perfect_square_l2370_237053


namespace salt_mixture_problem_l2370_237088

theorem salt_mixture_problem :
  ∃ (m : ℝ), 0.20 = (150 + 0.05 * m) / (600 + m) :=
by
  sorry

end salt_mixture_problem_l2370_237088


namespace value_divided_by_l2370_237061

theorem value_divided_by {x : ℝ} : (5 / x) * 12 = 10 → x = 6 :=
by
  sorry

end value_divided_by_l2370_237061


namespace heartsuit_calc_l2370_237096

-- Define the operation x ♡ y = 4x + 6y
def heartsuit (x y : ℝ) : ℝ := 4 * x + 6 * y

-- State the theorem
theorem heartsuit_calc : heartsuit 5 3 = 38 := by
  -- Proof omitted
  sorry

end heartsuit_calc_l2370_237096


namespace football_hits_ground_l2370_237062

theorem football_hits_ground :
  ∃ t : ℚ, -16 * t^2 + 18 * t + 60 = 0 ∧ 0 < t ∧ t = 41 / 16 :=
by
  sorry

end football_hits_ground_l2370_237062


namespace correct_total_cost_l2370_237097

noncomputable def total_cost_after_discount : ℝ :=
  let sandwich_cost := 4
  let soda_cost := 3
  let sandwich_count := 7
  let soda_count := 5
  let total_items := sandwich_count + soda_count
  let total_cost := sandwich_count * sandwich_cost + soda_count * soda_cost
  let discount := if total_items ≥ 10 then 0.1 * total_cost else 0
  total_cost - discount

theorem correct_total_cost :
  total_cost_after_discount = 38.7 :=
by
  -- The proof would go here
  sorry

end correct_total_cost_l2370_237097


namespace nicky_cards_value_l2370_237021

theorem nicky_cards_value 
  (x : ℝ)
  (h : 21 = 2 * x + 5) : 
  x = 8 := by
  sorry

end nicky_cards_value_l2370_237021


namespace arithmetic_progression_15th_term_l2370_237035

theorem arithmetic_progression_15th_term :
  let a := 2
  let d := 3
  let n := 15
  a + (n - 1) * d = 44 :=
by
  let a := 2
  let d := 3
  let n := 15
  sorry

end arithmetic_progression_15th_term_l2370_237035


namespace lines_intersect_and_not_perpendicular_l2370_237011

theorem lines_intersect_and_not_perpendicular (a : ℝ) :
  (∃ (x y : ℝ), 3 * x + 3 * y + a = 0 ∧ 3 * x - 2 * y + 1 = 0) ∧ 
  ¬ (∃ k1 k2 : ℝ, k1 = -1 ∧ k2 = 3 / 2 ∧ k1 ≠ k2 ∧ k1 * k2 = -1) :=
by
  sorry

end lines_intersect_and_not_perpendicular_l2370_237011


namespace largest_multiple_11_lt_neg85_l2370_237002

-- Define the conditions: a multiple of 11 and smaller than -85
def largest_multiple_lt (m n : Int) : Int :=
  let k := (m / n) - 1
  n * k

-- Define our specific problem
theorem largest_multiple_11_lt_neg85 : largest_multiple_lt (-85) 11 = -88 := 
  by
  sorry

end largest_multiple_11_lt_neg85_l2370_237002


namespace monotonic_interval_range_l2370_237034

noncomputable def f (a x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

theorem monotonic_interval_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 1 < x₁ → x₁ < 2 → 1 < x₂ → x₂ < 2 → x₁ < x₂ → f a x₁ ≤ f a x₂ ∨ f a x₁ ≥ f a x₂) ↔
  (a ∈ Set.Iic (-1) ∪ Set.Ici 0) :=
sorry

end monotonic_interval_range_l2370_237034


namespace choir_members_count_l2370_237012

theorem choir_members_count (n : ℕ) (h1 : n % 10 = 4) (h2 : n % 11 = 5) (h3 : 200 ≤ n) (h4 : n ≤ 300) : n = 234 := 
sorry

end choir_members_count_l2370_237012


namespace symmetric_circle_equation_l2370_237030

theorem symmetric_circle_equation :
  ∀ (x y : ℝ), (x + 2) ^ 2 + y ^ 2 = 5 → (x - 2) ^ 2 + y ^ 2 = 5 :=
by 
  sorry

end symmetric_circle_equation_l2370_237030


namespace op_example_l2370_237086

def myOp (c d : Int) : Int :=
  c * (d + 1) + c * d

theorem op_example : myOp 5 (-2) = -15 := 
  by
    sorry

end op_example_l2370_237086


namespace score_order_l2370_237022

variables (L N O P : ℕ)

def conditions : Prop := 
  O = L ∧ 
  N < max O P ∧ 
  P > L

theorem score_order (h : conditions L N O P) : N < O ∧ O < P :=
by
  sorry

end score_order_l2370_237022


namespace range_of_a_for_inequality_l2370_237070

noncomputable def has_solution_in_interval (a : ℝ) : Prop :=
  ∃ x : ℝ, 1 ≤ x ∧ x ≤ 4 ∧ (x^2 + a*x - 2 < 0)

theorem range_of_a_for_inequality : ∀ a : ℝ, has_solution_in_interval a ↔ a < 1 :=
by sorry

end range_of_a_for_inequality_l2370_237070


namespace problem_statement_l2370_237098

def f (n : ℕ) : ℕ :=
if n < 5 then n^2 + 1 else 2 * n - 3

theorem problem_statement : f (f (f 3)) = 31 :=
by
  sorry

end problem_statement_l2370_237098


namespace difference_students_guinea_pigs_l2370_237046

-- Define the conditions as constants
def students_per_classroom : Nat := 20
def guinea_pigs_per_classroom : Nat := 3
def number_of_classrooms : Nat := 6

-- Calculate the total number of students
def total_students : Nat := students_per_classroom * number_of_classrooms

-- Calculate the total number of guinea pigs
def total_guinea_pigs : Nat := guinea_pigs_per_classroom * number_of_classrooms

-- Define the theorem to prove the equality
theorem difference_students_guinea_pigs :
  total_students - total_guinea_pigs = 102 :=
by
  sorry -- Proof to be filled in

end difference_students_guinea_pigs_l2370_237046


namespace range_of_k_l2370_237023

-- Define the set M
def M := {x : ℝ | -1 ≤ x ∧ x ≤ 7}

-- Define the set N based on k
def N (k : ℝ) := {x : ℝ | k + 1 ≤ x ∧ x ≤ 2 * k - 1}

-- The main statement to prove
theorem range_of_k (k : ℝ) : M ∩ N k = ∅ → 6 < k :=
by
  -- skipping the proof as instructed
  sorry

end range_of_k_l2370_237023


namespace find_moles_of_NaCl_l2370_237036

-- Define the chemical reaction as an equation
def chemical_reaction (NaCl KNO3 NaNO3 KCl : ℕ) : Prop :=
  NaCl + KNO3 = NaNO3 + KCl

-- Define the problem conditions
def problem_conditions (naCl : ℕ) : Prop :=
  ∃ (kno3 naNo3 kcl : ℕ),
    kno3 = 3 ∧
    naNo3 = 3 ∧
    chemical_reaction naCl kno3 naNo3 kcl

-- Define the goal statement
theorem find_moles_of_NaCl (naCl : ℕ) : problem_conditions naCl → naCl = 3 :=
by
  sorry -- proof to be filled in later

end find_moles_of_NaCl_l2370_237036


namespace least_number_subtracted_divisible_by_six_l2370_237076

theorem least_number_subtracted_divisible_by_six :
  ∃ d : ℕ, d = 6 ∧ (427398 - 6) % d = 0 := by
sorry

end least_number_subtracted_divisible_by_six_l2370_237076


namespace final_cost_cooking_gear_sets_l2370_237010

-- Definitions based on conditions
def hand_mitts_cost : ℕ := 14
def apron_cost : ℕ := 16
def utensils_cost : ℕ := 10
def knife_cost : ℕ := 2 * utensils_cost
def discount_rate : ℚ := 0.25
def sales_tax_rate : ℚ := 0.08
def number_of_recipients : ℕ := 3 + 5

-- Proof statement: calculate the final cost
theorem final_cost_cooking_gear_sets :
  let total_cost_before_discount := hand_mitts_cost + apron_cost + utensils_cost + knife_cost
  let discounted_cost_per_set := (total_cost_before_discount : ℚ) * (1 - discount_rate)
  let total_cost_for_recipients := (discounted_cost_per_set * number_of_recipients : ℚ)
  let final_cost := total_cost_for_recipients * (1 + sales_tax_rate)
  final_cost = 388.80 :=
by
  sorry

end final_cost_cooking_gear_sets_l2370_237010


namespace pepperoni_slices_left_l2370_237015

theorem pepperoni_slices_left :
  ∀ (total_friends : ℕ) (total_slices : ℕ) (cheese_left : ℕ),
    (total_friends = 4) →
    (total_slices = 16) →
    (cheese_left = 7) →
    (∃ p_slices_left : ℕ, p_slices_left = 4) :=
by
  intros total_friends total_slices cheese_left h_friends h_slices h_cheese
  sorry

end pepperoni_slices_left_l2370_237015


namespace bunchkin_total_distance_l2370_237093

theorem bunchkin_total_distance
  (a b c d e : ℕ)
  (ha : a = 17)
  (hb : b = 43)
  (hc : c = 56)
  (hd : d = 66)
  (he : e = 76) :
  (a + b + c + d + e) / 2 = 129 :=
by
  sorry

end bunchkin_total_distance_l2370_237093


namespace prob_first_diamond_second_ace_or_face_l2370_237041

theorem prob_first_diamond_second_ace_or_face :
  let deck_size := 52
  let first_card_diamonds := 13 / deck_size
  let prob_ace_after_diamond := 4 / (deck_size - 1)
  let prob_face_after_diamond := 12 / (deck_size - 1)
  first_card_diamonds * (prob_ace_after_diamond + prob_face_after_diamond) = 68 / 867 :=
by
  let deck_size := 52
  let first_card_diamonds := 13 / deck_size
  let prob_ace_after_diamond := 4 / (deck_size - 1)
  let prob_face_after_diamond := 12 / (deck_size - 1)
  sorry

end prob_first_diamond_second_ace_or_face_l2370_237041
