import Mathlib

namespace machine_shirts_per_minute_l1480_148090

def shirts_made_yesterday : ℕ := 13
def shirts_made_today : ℕ := 3
def minutes_worked : ℕ := 2
def total_shirts_made : ℕ := shirts_made_yesterday + shirts_made_today
def shirts_per_minute : ℕ := total_shirts_made / minutes_worked

theorem machine_shirts_per_minute :
  shirts_per_minute = 8 := by
  sorry

end machine_shirts_per_minute_l1480_148090


namespace minimize_triangle_expression_l1480_148009

theorem minimize_triangle_expression :
  ∃ (a b c : ℤ), a < b ∧ b < c ∧ a + b + c = 30 ∧
  ∀ (x y z : ℤ), x < y ∧ y < z ∧ x + y + z = 30 → (z^2 + 18*x + 18*y - 446) ≥ 17 ∧ 
  ∃ (p q r : ℤ), p < q ∧ q < r ∧ p + q + r = 30 ∧ (r^2 + 18*p + 18*q - 446 = 17) := 
sorry

end minimize_triangle_expression_l1480_148009


namespace valid_paths_in_grid_l1480_148005

theorem valid_paths_in_grid : 
  let total_paths := Nat.choose 15 4;
  let paths_through_EF := (Nat.choose 7 2) * (Nat.choose 7 2);
  let valid_paths := total_paths - 2 * paths_through_EF;
  grid_size == (11, 4) ∧
  blocked_segments == [((5, 2), (5, 3)), ((6, 2), (6, 3))] 
  → valid_paths = 483 :=
by
  sorry

end valid_paths_in_grid_l1480_148005


namespace distance_traveled_l1480_148008

theorem distance_traveled
  (D : ℝ) (T : ℝ)
  (h1 : D = 10 * T)
  (h2 : D + 20 = 14 * T)
  : D = 50 := sorry

end distance_traveled_l1480_148008


namespace solve_for_p_l1480_148041

variable (p q : ℝ)
noncomputable def binomial_third_term : ℝ := 55 * p^9 * q^2
noncomputable def binomial_fourth_term : ℝ := 165 * p^8 * q^3

theorem solve_for_p (h1 : p + q = 1) (h2 : binomial_third_term p q = binomial_fourth_term p q) : p = 3 / 4 :=
by sorry

end solve_for_p_l1480_148041


namespace typing_speed_ratio_l1480_148074

-- Define Tim's and Tom's typing speeds
variables (T t : ℝ)

-- Conditions from the problem
def condition1 : Prop := T + t = 15
def condition2 : Prop := T + 1.6 * t = 18

-- The proposition to prove: the ratio of Tom's typing speed to Tim's is 1:2
theorem typing_speed_ratio (h1 : condition1 T t) (h2 : condition2 T t) : t / T = 1 / 2 :=
sorry

end typing_speed_ratio_l1480_148074


namespace train_speed_l1480_148091

theorem train_speed
  (cross_time : ℝ := 5)
  (train_length : ℝ := 111.12)
  (conversion_factor : ℝ := 3.6)
  (speed : ℝ := (train_length / cross_time) * conversion_factor) :
  speed = 80 :=
by
  sorry

end train_speed_l1480_148091


namespace pyramid_volume_l1480_148020

theorem pyramid_volume
  (s : ℝ) (h : ℝ) (base_area : ℝ) (triangular_face_area : ℝ) (surface_area : ℝ)
  (h_base_area : base_area = s * s)
  (h_triangular_face_area : triangular_face_area = (1 / 3) * base_area)
  (h_surface_area : surface_area = base_area + 4 * triangular_face_area)
  (h_surface_area_value : surface_area = 768)
  (h_vol : h = 7.78) :
  (1 / 3) * base_area * h = 853.56 :=
by
  sorry

end pyramid_volume_l1480_148020


namespace point_on_graph_l1480_148094

variable (x y : ℝ)

-- Define the condition for a point to be on the graph of the function y = 6/x
def is_on_graph (x y : ℝ) : Prop :=
  x * y = 6

-- State the theorem to be proved
theorem point_on_graph : is_on_graph (-2) (-3) :=
  by
  sorry

end point_on_graph_l1480_148094


namespace find_percentage_l1480_148028

theorem find_percentage (P : ℝ) (h : (P / 100) * 600 = (50 / 100) * 720) : P = 60 :=
by
  sorry

end find_percentage_l1480_148028


namespace students_last_year_l1480_148018

theorem students_last_year (students_this_year : ℝ) (increase_percent : ℝ) (last_year_students : ℝ) 
  (h1 : students_this_year = 960) 
  (h2 : increase_percent = 0.20) 
  (h3 : students_this_year = last_year_students * (1 + increase_percent)) : 
  last_year_students = 800 :=
by 
  sorry

end students_last_year_l1480_148018


namespace birch_tree_count_l1480_148025

theorem birch_tree_count:
  let total_trees := 8000
  let spruces := 0.12 * total_trees
  let pines := 0.15 * total_trees
  let maples := 0.18 * total_trees
  let cedars := 0.09 * total_trees
  let oaks := spruces + pines
  let calculated_trees := spruces + pines + maples + cedars + oaks
  let birches := total_trees - calculated_trees
  spruces = 960 → pines = 1200 → maples = 1440 → cedars = 720 → oaks = 2160 →
  birches = 1520 :=
by
  intros
  sorry

end birch_tree_count_l1480_148025


namespace sum_of_coefficients_l1480_148080

theorem sum_of_coefficients (a b : ℝ) (h1 : a = 1 * 5) (h2 : -b = 1 + 5) : a + b = -1 :=
by
  sorry

end sum_of_coefficients_l1480_148080


namespace supplements_of_congruent_angles_are_congruent_l1480_148039

-- Define the concept of supplementary angles
def is_supplementary (α β : ℝ) : Prop := α + β = 180

-- Statement of the problem
theorem supplements_of_congruent_angles_are_congruent :
  ∀ {α β γ δ : ℝ},
  is_supplementary α β →
  is_supplementary γ δ →
  β = δ →
  α = γ :=
by
  intros α β γ δ h1 h2 h3
  sorry

end supplements_of_congruent_angles_are_congruent_l1480_148039


namespace total_interest_percentage_l1480_148026

theorem total_interest_percentage (inv_total : ℝ) (rate1 rate2 : ℝ) (inv2 : ℝ)
  (h_inv_total : inv_total = 100000)
  (h_rate1 : rate1 = 0.09)
  (h_rate2 : rate2 = 0.11)
  (h_inv2 : inv2 = 24999.999999999996) :
  (rate1 * (inv_total - inv2) + rate2 * inv2) / inv_total * 100 = 9.5 := 
sorry

end total_interest_percentage_l1480_148026


namespace find_A_l1480_148082

theorem find_A (A B C : ℕ) (h1 : A ≠ B) (h2 : A ≠ C) (h3 : B ≠ C) (h4 : A < 10) (h5 : B < 10) (h6 : C < 10) (h7 : 10 * A + B + 10 * B + C = 101 * B + 10 * C) : A = 9 :=
sorry

end find_A_l1480_148082


namespace problem_1_problem_2_l1480_148027

def f (x : ℝ) : ℝ := x^2 + 4 * x
def g (a : ℝ) : ℝ := |a - 2| + |a + 1|

theorem problem_1 (x : ℝ) :
    (f x ≥ g 3) ↔ (x ≥ 1 ∨ x ≤ -5) :=
  sorry

theorem problem_2 (a : ℝ) :
    (∃ x : ℝ, f x + g a = 0) → (-3 / 2 ≤ a ∧ a ≤ 5 / 2) :=
  sorry

end problem_1_problem_2_l1480_148027


namespace product_of_roots_l1480_148054

theorem product_of_roots (a b c : ℝ) (h_eq : 24 * a^2 + 36 * a - 648 = 0) : a * c = -27 := 
by
  have h_root_product : (24 * a^2 + 36 * a - 648) = 0 ↔ a = -27 := sorry
  exact sorry

end product_of_roots_l1480_148054


namespace seventh_term_l1480_148030

def nth_term (n : ℕ) (a : ℝ) : ℝ :=
  (-2) ^ n * a ^ (2 * n - 1)

theorem seventh_term (a : ℝ) : nth_term 7 a = -128 * a ^ 13 :=
by sorry

end seventh_term_l1480_148030


namespace age_of_fourth_child_l1480_148034

theorem age_of_fourth_child (c1 c2 c3 c4 : ℕ) (h1 : c1 = 15)
  (h2 : c2 = c1 - 1) (h3 : c3 = c2 - 4)
  (h4 : c4 = c3 - 2) : c4 = 8 :=
by
  sorry

end age_of_fourth_child_l1480_148034


namespace possible_arrangements_count_l1480_148000

-- Define students as a type
inductive Student
| A | B | C | D | E | F

open Student

-- Define Club as a type
inductive Club
| A | B | C

open Club

-- Define the arrangement constraints
structure Arrangement :=
(assignment : Student → Club)
(club_size : Club → Nat)
(A_and_B_same_club : assignment A = assignment B)
(C_and_D_diff_clubs : assignment C ≠ assignment D)
(club_A_size : club_size A = 3)
(all_clubs_nonempty : ∀ c : Club, club_size c > 0)

-- Define the possible number of arrangements
def arrangement_count (a : Arrangement) : Nat := sorry

-- Theorem stating the number of valid arrangements
theorem possible_arrangements_count : ∃ a : Arrangement, arrangement_count a = 24 := sorry

end possible_arrangements_count_l1480_148000


namespace required_percentage_to_pass_l1480_148055

theorem required_percentage_to_pass
  (marks_obtained : ℝ)
  (marks_failed_by : ℝ)
  (max_marks : ℝ)
  (passing_marks := marks_obtained + marks_failed_by)
  (required_percentage : ℝ := (passing_marks / max_marks) * 100)
  (h : marks_obtained = 80)
  (h' : marks_failed_by = 40)
  (h'' : max_marks = 200) :
  required_percentage = 60 := 
by
  sorry

end required_percentage_to_pass_l1480_148055


namespace albert_earnings_l1480_148050

theorem albert_earnings (E P : ℝ) 
  (h1 : E * 1.20 = 660) 
  (h2 : E * (1 + P) = 693) : 
  P = 0.26 :=
sorry

end albert_earnings_l1480_148050


namespace yao_ming_shots_l1480_148077

-- Defining the conditions
def total_shots_made : ℕ := 14
def total_points_scored : ℕ := 28
def three_point_shots_made : ℕ := 3
def two_point_shots (x : ℕ) : ℕ := x
def free_throws_made (x : ℕ) : ℕ := total_shots_made - three_point_shots_made - x

-- The theorem we want to prove
theorem yao_ming_shots :
  ∃ (x y : ℕ),
    (total_shots_made = three_point_shots_made + x + y) ∧ 
    (total_points_scored = 3 * three_point_shots_made + 2 * x + y) ∧
    (x = 8) ∧
    (y = 3) :=
sorry

end yao_ming_shots_l1480_148077


namespace find_p_for_quadratic_l1480_148049

theorem find_p_for_quadratic (p : ℝ) (h : p ≠ 0) 
  (h_eq : ∀ x : ℝ, p * x^2 - 10 * x + 2 = 0 → x = 5 / p) : p = 12.5 :=
sorry

end find_p_for_quadratic_l1480_148049


namespace kody_half_mohamed_years_ago_l1480_148052

-- Definitions of initial conditions
def current_age_mohamed : ℕ := 2 * 30
def current_age_kody : ℕ := 32

-- Proof statement
theorem kody_half_mohamed_years_ago : ∃ x : ℕ, (current_age_kody - x) = (1 / 2 : ℕ) * (current_age_mohamed - x) ∧ x = 4 := 
by 
  sorry

end kody_half_mohamed_years_ago_l1480_148052


namespace solve_x_l1480_148043

theorem solve_x :
  (1 / 4 - 1 / 6) = 1 / (12 : ℝ) :=
by sorry

end solve_x_l1480_148043


namespace range_of_m_is_increasing_l1480_148093

noncomputable def f (x : ℝ) (m: ℝ) := x^2 + m*x + m

theorem range_of_m_is_increasing :
  { m : ℝ // ∀ x y : ℝ, -2 ≤ x → x ≤ y → f x m ≤ f y m } = {m | 4 ≤ m} :=
by
  sorry

end range_of_m_is_increasing_l1480_148093


namespace quadruplet_zero_solution_l1480_148059

theorem quadruplet_zero_solution (a b c d : ℝ)
  (h1 : (a + b) * (a^2 + b^2) = (c + d) * (c^2 + d^2))
  (h2 : (a + c) * (a^2 + c^2) = (b + d) * (b^2 + d^2))
  (h3 : (a + d) * (a^2 + d^2) = (b + c) * (b^2 + c^2)) :
  a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 := 
sorry

end quadruplet_zero_solution_l1480_148059


namespace original_number_of_people_l1480_148099

variable (x : ℕ)
-- Conditions
axiom one_third_left : x / 3 > 0
axiom half_dancing : 18 = x / 3

-- Theorem Statement
theorem original_number_of_people (x : ℕ) (one_third_left : x / 3 > 0) (half_dancing : 18 = x / 3) : x = 54 := sorry

end original_number_of_people_l1480_148099


namespace ratio_of_kids_waiting_for_slide_to_swings_final_ratio_of_kids_waiting_l1480_148035

-- Define the conditions
def W : ℕ := 3
def wait_time_swing : ℕ := 120 * W
def wait_time_slide (S : ℕ) : ℕ := 15 * S
def wait_diff_condition (S : ℕ) : Prop := wait_time_swing - wait_time_slide S = 270

theorem ratio_of_kids_waiting_for_slide_to_swings (S : ℕ) (h : wait_diff_condition S) : S = 6 :=
by
  -- placeholder proof
  sorry

theorem final_ratio_of_kids_waiting (S : ℕ) (h : wait_diff_condition S) : S / W = 2 :=
by
  -- placeholder proof
  sorry

end ratio_of_kids_waiting_for_slide_to_swings_final_ratio_of_kids_waiting_l1480_148035


namespace A_inter_B_complement_l1480_148075

def A : Set ℝ := {x : ℝ | -4 < x^2 - 5*x + 2 ∧ x^2 - 5*x + 2 < 26}
def B : Set ℝ := {x : ℝ | -x^2 + 4*x - 3 < 0}

theorem A_inter_B_complement :
  A ∩ B = {x : ℝ | (-3 < x ∧ x < 1) ∨ (3 < x ∧ x < 8)} ∧
  {x | x ∉ A ∩ B} = {x : ℝ | x ≤ -3 ∨ (1 ≤ x ∧ x ≤ 3) ∨ x ≥ 8 } :=
by
  sorry

end A_inter_B_complement_l1480_148075


namespace part_I_part_II_l1480_148051

-- Part (I): If a = 1, prove that q implies p
theorem part_I (x : ℝ) (h : 3 < x ∧ x < 4) : (1 < x) ∧ (x < 4) :=
by sorry

-- Part (II): Prove the range of a for which p is necessary but not sufficient for q
theorem part_II (a : ℝ) (h1 : a > 0) (h2 : ∀ x : ℝ, (a < x ∧ x < 4 * a) → (3 < x ∧ x < 4)) : 1 < a ∧ a ≤ 3 :=
by sorry

end part_I_part_II_l1480_148051


namespace molecular_weight_CO_l1480_148057

theorem molecular_weight_CO : 
  let molecular_weight_C := 12.01
  let molecular_weight_O := 16.00
  molecular_weight_C + molecular_weight_O = 28.01 :=
by
  sorry

end molecular_weight_CO_l1480_148057


namespace diagonal_not_perpendicular_l1480_148053

open Real

theorem diagonal_not_perpendicular (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) 
  (h_a_ne_b : a ≠ b) (h_c_ne_d : c ≠ d) (h_a_ne_c : a ≠ c) (h_b_ne_d : b ≠ d): 
  ¬ ((d - b) * (b - a) = - (c - a) * (d - c)) :=
by
  sorry

end diagonal_not_perpendicular_l1480_148053


namespace rearrangement_impossible_l1480_148038

-- Definition of an 8x8 chessboard's cell numbering.
def cell_number (i j : ℕ) : ℕ := i + j - 1

-- The initial placement of pieces, represented as a permutation on {1, 2, ..., 8}
def initial_placement (p: Fin 8 → Fin 8) := True -- simplify for definition purposes

-- The rearranged placement of pieces
def rearranged_placement (q: Fin 8 → Fin 8) := True -- simplify for definition purposes

-- Condition for each piece: cell number increases
def cell_increase_condition (p q: Fin 8 → Fin 8) : Prop :=
  ∀ i, cell_number (q i).val (i.val + 1) > cell_number (p i).val (i.val + 1)

-- The main theorem to state it's impossible to rearrange under the given conditions and question
theorem rearrangement_impossible 
  (p q: Fin 8 → Fin 8) 
  (h_initial : initial_placement p) 
  (h_rearranged : rearranged_placement q) 
  (h_increase : cell_increase_condition p q) : False := 
sorry

end rearrangement_impossible_l1480_148038


namespace find_divisor_l1480_148072

theorem find_divisor (n d k : ℤ) (h1 : n = k * d + 3) (h2 : n^2 % d = 4) : d = 5 :=
by
  sorry

end find_divisor_l1480_148072


namespace smallest_n_for_perfect_square_and_cube_l1480_148060

theorem smallest_n_for_perfect_square_and_cube :
  ∃ n : ℕ, (∃ a : ℕ, 4 * n = a^2) ∧ (∃ b : ℕ, 5 * n = b^3) ∧ n = 125 :=
by
  sorry

end smallest_n_for_perfect_square_and_cube_l1480_148060


namespace distance_to_plane_l1480_148062

variable (V : ℝ) (A : ℝ) (r : ℝ) (d : ℝ)

-- Assume the volume of the sphere and area of the cross-section
def sphere_volume := V = 4 * Real.sqrt 3 * Real.pi
def cross_section_area := A = Real.pi

-- Define radius of sphere and cross-section
def sphere_radius := r = Real.sqrt 3
def cross_section_radius := Real.sqrt A = 1

-- Define distance as per Pythagorean theorem
def distance_from_center := d = Real.sqrt (r^2 - 1^2)

-- Main statement to prove
theorem distance_to_plane (V A : ℝ)
  (h1 : sphere_volume V) 
  (h2 : cross_section_area A) 
  (h3: sphere_radius r) 
  (h4: cross_section_radius A) : 
  distance_from_center r d :=
sorry

end distance_to_plane_l1480_148062


namespace determinant_of_matrix_l1480_148089

def mat : Matrix (Fin 3) (Fin 3) ℤ := 
  ![![3, 0, 2],![8, 5, -2],![3, 3, 6]]

theorem determinant_of_matrix : Matrix.det mat = 90 := 
by 
  sorry

end determinant_of_matrix_l1480_148089


namespace cos_pi_minus_alpha_l1480_148032

theorem cos_pi_minus_alpha (α : ℝ) (h : Real.sin (Real.pi / 2 + α) = 1 / 7) : Real.cos (Real.pi - α) = - (1 / 7) := by
  sorry

end cos_pi_minus_alpha_l1480_148032


namespace max_number_of_pies_l1480_148040

def total_apples := 250
def apples_given_to_students := 42
def apples_used_for_juice := 75
def apples_per_pie := 8

theorem max_number_of_pies (h1 : total_apples = 250)
                           (h2 : apples_given_to_students = 42)
                           (h3 : apples_used_for_juice = 75)
                           (h4 : apples_per_pie = 8) :
  ((total_apples - apples_given_to_students - apples_used_for_juice) / apples_per_pie) ≥ 16 :=
by
  sorry

end max_number_of_pies_l1480_148040


namespace range_of_k_l1480_148085

theorem range_of_k (k : ℝ) (f : ℝ → ℝ) (f' : ℝ → ℝ) (h_f : ∀ x, f x = x^3 - 3 * x^2 - k)
  (h_f' : ∀ x, f' x = 3 * x^2 - 6 * x) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₃ ≠ x₁ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ f x₃ = 0) ↔ -4 < k ∧ k < 0 :=
sorry

end range_of_k_l1480_148085


namespace evaluate_expression_l1480_148098

theorem evaluate_expression :
  (24^36) / (72^18) = 8^18 :=
by
  sorry

end evaluate_expression_l1480_148098


namespace company_employee_count_l1480_148017

/-- 
 Given the employees are divided into three age groups: A, B, and C, with a ratio of 5:4:1,
 a stratified sampling method is used to draw a sample of size 20 from the population,
 and the probability of selecting both person A and person B from group C is 1/45.
 Prove the total number of employees in the company is 100.
-/
theorem company_employee_count :
  ∃ (total_employees : ℕ),
    (∃ (ratio_A : ℕ) (ratio_B : ℕ) (ratio_C : ℕ),
      ratio_A = 5 ∧ 
      ratio_B = 4 ∧ 
      ratio_C = 1 ∧
      ∃ (sample_size : ℕ), 
        sample_size = 20 ∧
        ∃ (prob_selecting_two_from_C : ℚ),
          prob_selecting_two_from_C = 1 / 45 ∧
          total_employees = 100) :=
sorry

end company_employee_count_l1480_148017


namespace cost_of_song_book_l1480_148064

-- Define the given constants: cost of trumpet, cost of music tool, and total spent at the music store.
def cost_of_trumpet : ℝ := 149.16
def cost_of_music_tool : ℝ := 9.98
def total_spent_at_store : ℝ := 163.28

-- The goal is to prove that the cost of the song book is $4.14.
theorem cost_of_song_book :
  total_spent_at_store - (cost_of_trumpet + cost_of_music_tool) = 4.14 :=
by
  sorry

end cost_of_song_book_l1480_148064


namespace initial_forks_l1480_148019

variables (forks knives spoons teaspoons : ℕ)
variable (F : ℕ)

-- Conditions as given
def num_knives := F + 9
def num_spoons := 2 * (F + 9)
def num_teaspoons := F / 2
def total_cutlery := (F + 2) + (F + 11) + (2 * (F + 9) + 2) + (F / 2 + 2)

-- Problem statement to prove
theorem initial_forks :
  (total_cutlery = 62) ↔ (F = 6) :=
by {
  sorry
}

end initial_forks_l1480_148019


namespace neither_sufficient_nor_necessary_condition_l1480_148011

-- Given conditions
def p (a : ℝ) : Prop := ∃ (x y : ℝ), a * x + y + 1 = 0 ∧ a * x - y + 2 = 0
def q : Prop := ∃ (a : ℝ), a = 1

-- The proof problem
theorem neither_sufficient_nor_necessary_condition : 
  ¬ ((∀ a, p a → q) ∧ (∀ a, q → p a)) :=
sorry

end neither_sufficient_nor_necessary_condition_l1480_148011


namespace ram_money_l1480_148095

variable (R G K : ℕ)

theorem ram_money (h1 : R / G = 7 / 17) (h2 : G / K = 7 / 17) (h3 : K = 2890) : R = 490 :=
by
  sorry

end ram_money_l1480_148095


namespace problem1_problem2_l1480_148081

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 / x

def is_increasing_on (f : ℝ → ℝ) (domain : Set ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ∈ domain → x₂ ∈ domain → x₁ < x₂ → f x₁ < f x₂

theorem problem1 : is_increasing_on f {x | 1 ≤ x} := 
by sorry

def is_decreasing (g : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → g x₁ > g x₂

theorem problem2 (g : ℝ → ℝ) (h_decreasing : is_decreasing g)
  (h_inequality : ∀ x : ℝ, 1 ≤ x → g (x^3 + 2) < g ((a^2 - 2 * a) * x)) :
  -1 < a ∧ a < 3 :=
by sorry

end problem1_problem2_l1480_148081


namespace number_subtracted_l1480_148058

theorem number_subtracted (t k x : ℝ) (h1 : t = (5 / 9) * (k - x)) (h2 : t = 105) (h3 : k = 221) : x = 32 :=
by
  sorry

end number_subtracted_l1480_148058


namespace comparison_of_logs_l1480_148065

noncomputable def a : ℝ := Real.log 6 / Real.log 3
noncomputable def b : ℝ := Real.log 12 / Real.log 6
noncomputable def c : ℝ := Real.log 16 / Real.log 8

theorem comparison_of_logs : a > b ∧ b > c :=
by
  sorry

end comparison_of_logs_l1480_148065


namespace eddy_travel_time_l1480_148047

theorem eddy_travel_time (T : ℝ) (S_e S_f : ℝ) (Freddy_time : ℝ := 4)
  (distance_AB : ℝ := 540) (distance_AC : ℝ := 300) (speed_ratio : ℝ := 2.4) :
  (distance_AB / T = 2.4 * (distance_AC / Freddy_time)) -> T = 3 :=
by
  sorry

end eddy_travel_time_l1480_148047


namespace complete_square_solution_l1480_148076

theorem complete_square_solution
  (x : ℝ)
  (h : x^2 + 4*x + 2 = 0):
  ∃ c : ℝ, (x + 2)^2 = c ∧ c = 2 :=
by
  sorry

end complete_square_solution_l1480_148076


namespace initial_pencils_count_l1480_148003

variables {pencils_taken : ℕ} {pencils_left : ℕ} {initial_pencils : ℕ}

theorem initial_pencils_count 
  (h1 : pencils_taken = 4)
  (h2 : pencils_left = 5) :
  initial_pencils = 9 :=
by 
  sorry

end initial_pencils_count_l1480_148003


namespace base8_units_digit_l1480_148002

theorem base8_units_digit (n m : ℕ) (h1 : n = 348) (h2 : m = 27) : 
  (n * m % 8) = 4 := sorry

end base8_units_digit_l1480_148002


namespace quadratic_inequality_no_solution_l1480_148070

theorem quadratic_inequality_no_solution (a b c : ℝ) (h : a ≠ 0)
  (hnsol : ∀ x : ℝ, ¬(a * x^2 + b * x + c ≥ 0)) :
  a < 0 ∧ b^2 - 4 * a * c < 0 :=
sorry

end quadratic_inequality_no_solution_l1480_148070


namespace average_of_remaining_numbers_l1480_148033

variable (numbers : List ℝ) (x y : ℝ)

theorem average_of_remaining_numbers
  (h_length_15 : numbers.length = 15)
  (h_avg_15 : (numbers.sum / 15) = 90)
  (h_x : x = 80)
  (h_y : y = 85)
  (h_members : x ∈ numbers ∧ y ∈ numbers) :
  ((numbers.sum - x - y) / 13) = 91.15 :=
sorry

end average_of_remaining_numbers_l1480_148033


namespace interval_solution_l1480_148029

theorem interval_solution (x : ℝ) : 2 ≤ x / (2 * x - 5) ∧ x / (2 * x - 5) < 7 ↔ (35 / 13 : ℝ) < x ∧ x ≤ 10 / 3 :=
by
  sorry

end interval_solution_l1480_148029


namespace tourists_left_l1480_148042

noncomputable def tourists_remaining {initial remaining poisoned recovered : ℕ} 
  (h1 : initial = 30)
  (h2 : remaining = initial - 2)
  (h3 : poisoned = remaining / 2)
  (h4 : recovered = poisoned / 7)
  (h5 : remaining % 2 = 0) -- ensuring even division for / 2
  (h6 : poisoned % 7 = 0) -- ensuring even division for / 7
  : ℕ :=
  remaining - poisoned + recovered

theorem tourists_left 
  (initial remaining poisoned recovered : ℕ) 
  (h1 : initial = 30)
  (h2 : remaining = initial - 2)
  (h3 : poisoned = remaining / 2)
  (h4 : recovered = poisoned / 7)
  (h5 : remaining % 2 = 0) -- ensuring even division for / 2
  (h6 : poisoned % 7 = 0) -- ensuring even division for / 7
  : tourists_remaining h1 h2 h3 h4 h5 h6 = 16 :=
  by
  sorry

end tourists_left_l1480_148042


namespace total_cups_l1480_148087

theorem total_cups (n : ℤ) (h_rainy_days : n = 8) :
  let tea_cups := 6 * 3
  let total_cups := tea_cups + n
  total_cups = 26 :=
by
  let tea_cups := 6 * 3
  let total_cups := tea_cups + n
  exact sorry

end total_cups_l1480_148087


namespace geometric_sequence_nine_l1480_148067

theorem geometric_sequence_nine (a : ℕ → ℝ) (h_geo : ∀ n, a (n + 1) / a n = a 1 / a 0) 
  (h_a1 : a 1 = 2) (h_a5: a 5 = 4) : a 9 = 8 := 
by
  sorry

end geometric_sequence_nine_l1480_148067


namespace seating_arrangement_l1480_148012

variable {M I P A : Prop}

def first_fact : ¬ M := sorry
def second_fact : ¬ A := sorry
def third_fact : ¬ M → I := sorry
def fourth_fact : I → P := sorry

theorem seating_arrangement : ¬ M → (I ∧ P) :=
by
  intros hM
  have hI : I := third_fact hM
  have hP : P := fourth_fact hI
  exact ⟨hI, hP⟩

end seating_arrangement_l1480_148012


namespace no_solution_range_of_a_l1480_148001

noncomputable def range_of_a : Set ℝ := {a | ∀ x : ℝ, ¬(abs (x - 1) + abs (x - 2) ≤ a^2 + a + 1)}

theorem no_solution_range_of_a :
  range_of_a = {a | -1 < a ∧ a < 0} :=
by
  sorry

end no_solution_range_of_a_l1480_148001


namespace division_of_difference_squared_l1480_148014

theorem division_of_difference_squared :
  ((2222 - 2121)^2) / 196 = 52 := 
sorry

end division_of_difference_squared_l1480_148014


namespace discriminant_of_quadratic_equation_l1480_148013

theorem discriminant_of_quadratic_equation :
  let a := 5
  let b := -9
  let c := 4
  (b^2 - 4 * a * c = 1) :=
by {
  sorry
}

end discriminant_of_quadratic_equation_l1480_148013


namespace find_interest_rate_l1480_148048

noncomputable def interest_rate (A P T : ℚ) : ℚ := (A - P) / (P * T) * 100

theorem find_interest_rate :
  let A := 1120
  let P := 921.0526315789474
  let T := 2.4
  interest_rate A P T = 9 := 
by
  sorry

end find_interest_rate_l1480_148048


namespace total_money_raised_l1480_148024

-- Assume there are 30 students in total
def total_students := 30

-- Assume 10 students raised $20 each
def students_raising_20 := 10
def money_raised_per_20 := 20

-- The rest of the students raised $30 each
def students_raising_30 := total_students - students_raising_20
def money_raised_per_30 := 30

-- Prove that the total amount raised is $800
theorem total_money_raised :
  (students_raising_20 * money_raised_per_20) +
  (students_raising_30 * money_raised_per_30) = 800 :=
by
  sorry

end total_money_raised_l1480_148024


namespace problem_statement_l1480_148079

noncomputable def least_period (f : ℝ → ℝ) (P : ℝ) :=
  ∀ x : ℝ, f (x + P) = f x

theorem problem_statement (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x + 5) + f (x - 5) = f x) :
  least_period f 30 :=
sorry

end problem_statement_l1480_148079


namespace second_number_is_72_l1480_148097

theorem second_number_is_72 
  (sum_eq_264 : ∀ (x : ℝ), 2 * x + x + (2 / 3) * x = 264) 
  (first_eq_2_second : ∀ (x : ℝ), first = 2 * x)
  (third_eq_1_3_first : ∀ (first : ℝ), third = 1 / 3 * first) :
  second = 72 :=
by
  sorry

end second_number_is_72_l1480_148097


namespace num_5_letter_words_with_at_least_two_consonants_l1480_148063

open Finset

def letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}
def consonants : Finset Char := {'B', 'C', 'D', 'F'}
def vowels : Finset Char := {'A', 'E'}

def total_5_letter_words : ℕ := 6^5
def words_with_0_consonants : ℕ := 2^5
def words_with_1_consonant : ℕ := 5 * 4 * 2^4

theorem num_5_letter_words_with_at_least_two_consonants : 
  total_5_letter_words - (words_with_0_consonants + words_with_1_consonant) = 7424 := by
  sorry

end num_5_letter_words_with_at_least_two_consonants_l1480_148063


namespace find_x_l1480_148088

def f (x : ℝ) := 2 * x - 3

theorem find_x : ∃ x, 2 * (f x) - 11 = f (x - 2) ∧ x = 5 :=
by 
  unfold f
  exists 5
  sorry

end find_x_l1480_148088


namespace range_of_t_max_radius_circle_eq_l1480_148056

-- Definitions based on conditions
def circle_equation (x y t : ℝ) := x^2 + y^2 - 2 * x + t^2 = 0

-- Statement for the range of values of t
theorem range_of_t (t : ℝ) (h : ∃ x y : ℝ, circle_equation x y t) : -1 < t ∧ t < 1 := sorry

-- Statement for the equation of the circle when t = 0
theorem max_radius_circle_eq (x y : ℝ) (h : circle_equation x y 0) : (x - 1)^2 + y^2 = 1 := sorry

end range_of_t_max_radius_circle_eq_l1480_148056


namespace force_exerted_by_pulley_on_axis_l1480_148016

-- Define the basic parameters given in the problem
def m1 : ℕ := 3 -- mass 1 in kg
def m2 : ℕ := 6 -- mass 2 in kg
def g : ℕ := 10 -- acceleration due to gravity in m/s^2

-- From the problem, we know that:
def F1 : ℕ := m1 * g -- gravitational force on mass 1
def F2 : ℕ := m2 * g -- gravitational force on mass 2

-- To find the tension, setup the equations
def a := (F2 - F1) / (m1 + m2) -- solving for acceleration between the masses

def T := (m1 * a) + F1 -- solving for the tension in the rope considering mass 1

-- Define the proof statement to find the force exerted by the pulley on its axis
theorem force_exerted_by_pulley_on_axis : 2 * T = 80 :=
by
  -- Annotations or calculations can go here
  sorry

end force_exerted_by_pulley_on_axis_l1480_148016


namespace integer_1000_column_l1480_148036

def column_sequence (n : ℕ) : String :=
  let sequence := ["A", "B", "C", "D", "E", "F", "E", "D", "C", "B"]
  sequence.get! (n % 10)

theorem integer_1000_column : column_sequence 999 = "C" :=
by
  sorry

end integer_1000_column_l1480_148036


namespace trapezoid_circumcircle_radius_l1480_148061

theorem trapezoid_circumcircle_radius :
  ∀ (BC AD height midline R : ℝ), 
  (BC / AD = (5 / 12)) →
  (height = 17) →
  (midline = height) →
  (midline = (BC + AD) / 2) →
  (BC = 10) →
  (AD = 24) →
  R = 13 :=
by
  intro BC AD height midline R
  intros h_ratio h_height h_midline_eq_height h_midline_eq_avg_bases h_BC h_AD
  -- Proof would go here, but it's skipped for now.
  sorry

end trapezoid_circumcircle_radius_l1480_148061


namespace mixed_number_multiplication_equiv_l1480_148069

theorem mixed_number_multiplication_equiv :
  (-3 - 1 / 2) * (5 / 7) = -3.5 * (5 / 7) := 
by 
  sorry

end mixed_number_multiplication_equiv_l1480_148069


namespace number_of_elements_in_A_l1480_148044

theorem number_of_elements_in_A (a b : ℕ) (h1 : a = 3 * b)
  (h2 : a + b - 100 = 500) (h3 : 100 = 100) (h4 : a - 100 = b - 100 + 50) : a = 450 := by
  sorry

end number_of_elements_in_A_l1480_148044


namespace max_n_no_constant_term_l1480_148004

theorem max_n_no_constant_term (n : ℕ) (h : n < 10 ∧ n ≠ 3 ∧ n ≠ 6 ∧ n ≠ 9 ∧ n ≠ 2 ∧ n ≠ 5 ∧ n ≠ 8): n ≤ 7 :=
by {
  sorry
}

end max_n_no_constant_term_l1480_148004


namespace distance_between_trees_l1480_148023

theorem distance_between_trees (num_trees : ℕ) (total_length : ℕ) (num_spaces : ℕ) (distance_per_space : ℕ) 
  (h_num_trees : num_trees = 11) (h_total_length : total_length = 180)
  (h_num_spaces : num_spaces = num_trees - 1) (h_distance_per_space : distance_per_space = total_length / num_spaces) :
  distance_per_space = 18 := 
  by 
    sorry

end distance_between_trees_l1480_148023


namespace bailey_chew_toys_l1480_148015

theorem bailey_chew_toys (dog_treats rawhide_bones: ℕ) (cards items_per_card : ℕ)
  (h1 : dog_treats = 8)
  (h2 : rawhide_bones = 10)
  (h3 : cards = 4)
  (h4 : items_per_card = 5) :
  ∃ chew_toys : ℕ, chew_toys = 2 :=
by
  sorry

end bailey_chew_toys_l1480_148015


namespace area_of_triangle_l1480_148045

theorem area_of_triangle {a b c : ℝ} (S : ℝ) (h1 : (a^2) * (Real.sin C) = 4 * (Real.sin A))
                          (h2 : (a + c)^2 = 12 + b^2)
                          (h3 : S = Real.sqrt ((1/4) * (a^2 * c^2 - ( (a^2 + c^2 - b^2)/2 )^2))) :
  S = Real.sqrt 3 :=
by
  sorry

end area_of_triangle_l1480_148045


namespace salary_for_january_l1480_148078

theorem salary_for_january (J F M A May : ℝ)
  (h1 : (J + F + M + A) / 4 = 8000)
  (h2 : (F + M + A + May) / 4 = 8700)
  (h_may : May = 6500) :
  J = 3700 :=
by
  sorry

end salary_for_january_l1480_148078


namespace determinant_condition_l1480_148037

theorem determinant_condition (a b c d : ℤ)
    (H : ∀ m n : ℤ, ∃ x y : ℤ, a * x + b * y = m ∧ c * x + d * y = n) :
    a * d - b * c = 1 ∨ a * d - b * c = -1 :=
by 
  sorry

end determinant_condition_l1480_148037


namespace rectangular_garden_area_l1480_148084

theorem rectangular_garden_area (w l : ℝ) 
  (h1 : l = 3 * w + 30) 
  (h2 : 2 * (l + w) = 800) : w * l = 28443.75 := 
by
  sorry

end rectangular_garden_area_l1480_148084


namespace simplify_expr1_simplify_expr2_simplify_expr3_l1480_148086

-- For the first expression
theorem simplify_expr1 (a b : ℝ) : 2 * a - 3 * b + a - 5 * b = 3 * a - 8 * b :=
by
  sorry

-- For the second expression
theorem simplify_expr2 (a : ℝ) : (a^2 - 6 * a) - 3 * (a^2 - 2 * a + 1) + 3 = -2 * a^2 :=
by
  sorry

-- For the third expression
theorem simplify_expr3 (x y : ℝ) : 4*(x^2*y - 2*x*y^2) - 3*(-x*y^2 + 2*x^2*y) = -2*x^2*y - 5*x*y^2 :=
by
  sorry

end simplify_expr1_simplify_expr2_simplify_expr3_l1480_148086


namespace pascal_tenth_number_in_hundred_row_l1480_148007

def pascal_row (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem pascal_tenth_number_in_hundred_row :
  pascal_row 99 9 = Nat.choose 99 9 :=
by
  sorry

end pascal_tenth_number_in_hundred_row_l1480_148007


namespace maximum_value_problem_l1480_148096

theorem maximum_value_problem (a b c : ℝ) (h : a^2 + b^2 + c^2 = 1) :
  (a^2 - b * c) * (b^2 - c * a) * (c^2 - a * b) ≤ 1 / 8 :=
sorry

end maximum_value_problem_l1480_148096


namespace solve_for_a_l1480_148046

-- Given the equation is quadratic, meaning the highest power of x in the quadratic term equals 2
theorem solve_for_a (a : ℚ) : (2 * a - 1 = 2) -> a = 3 / 2 :=
by
  sorry

end solve_for_a_l1480_148046


namespace books_problem_l1480_148068

variable (L W : ℕ) -- L for Li Ming's initial books, W for Wang Hong's initial books

theorem books_problem (h1 : L = W + 26) (h2 : L - 14 = W + 14 - 2) : 14 = 14 :=
by
  sorry

end books_problem_l1480_148068


namespace find_k_l1480_148031

theorem find_k (x1 x2 : ℝ) (r : ℝ) (h1 : x1 = 3 * r) (h2 : x2 = r) (h3 : x1 + x2 = -8) (h4 : x1 * x2 = k) : k = 12 :=
by
  -- proof steps here
  sorry

end find_k_l1480_148031


namespace total_boys_went_down_slide_l1480_148010

-- Definitions according to the conditions given
def boys_went_down_slide1 : ℕ := 22
def boys_went_down_slide2 : ℕ := 13

-- The statement to be proved
theorem total_boys_went_down_slide : boys_went_down_slide1 + boys_went_down_slide2 = 35 := 
by 
  sorry

end total_boys_went_down_slide_l1480_148010


namespace parabola_expression_l1480_148073

theorem parabola_expression:
  (∀ x : ℝ, y = a * (x + 3) * (x - 1)) →
  a * (0 + 3) * (0 - 1) = 2 →
  a = -2 / 3 →
  (∀ x : ℝ, y = -2 / 3 * x^2 - 4 / 3 * x + 2) :=
by
  sorry

end parabola_expression_l1480_148073


namespace percentage_HNO3_final_l1480_148066

-- Define the initial conditions
def initial_volume_solution : ℕ := 60 -- 60 liters of solution
def initial_percentage_HNO3 : ℝ := 0.45 -- 45% HNO3
def added_pure_HNO3 : ℕ := 6 -- 6 liters of pure HNO3

-- Define the volume of HNO3 in the initial solution
def hno3_initial := initial_percentage_HNO3 * initial_volume_solution

-- Define the total volume of the final solution
def total_volume_final := initial_volume_solution + added_pure_HNO3

-- Define the total amount of HNO3 in the final solution
def total_hno3_final := hno3_initial + added_pure_HNO3

-- The main theorem: prove the final percentage is 50%
theorem percentage_HNO3_final :
  (total_hno3_final / total_volume_final) * 100 = 50 :=
by
  -- proof is omitted
  sorry

end percentage_HNO3_final_l1480_148066


namespace circle_O2_tangent_circle_O2_intersect_l1480_148092

-- Condition: The equation of circle O_1 is \(x^2 + (y + 1)^2 = 4\)
def circle_O1 (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 4

-- Condition: The center of circle O_2 is \(O_2(2, 1)\)
def center_O2 : (ℝ × ℝ) := (2, 1)

-- Prove the equation of circle O_2 if it is tangent to circle O_1
theorem circle_O2_tangent : 
  ∀ (x y : ℝ), circle_O1 x y → (x - 2)^2 + (y - 1)^2 = 12 - 8 * Real.sqrt 2 :=
sorry

-- Prove the equations of circle O_2 if it intersects circle O_1 and \(|AB| = 2\sqrt{2}\)
theorem circle_O2_intersect :
  ∀ (x y : ℝ), 
  circle_O1 x y → 
  (2 * Real.sqrt 2 = |(x - 2)^2 + (y - 1)^2 - 4| ∨ (x - 2)^2 + (y - 1)^2 = 20) :=
sorry

end circle_O2_tangent_circle_O2_intersect_l1480_148092


namespace quadratic_equation_correct_form_l1480_148021

theorem quadratic_equation_correct_form :
  ∀ (a b c x : ℝ), a = 3 → b = -6 → c = 1 → a * x^2 + c = b * x :=
by
  intros a b c x ha hb hc
  rw [ha, hb, hc]
  sorry

end quadratic_equation_correct_form_l1480_148021


namespace evaluate_expression_l1480_148022

theorem evaluate_expression :
  (2 + 3 / (4 + 5 / (6 + 7 / 8))) = 137 / 52 :=
by
  sorry

end evaluate_expression_l1480_148022


namespace range_of_a_l1480_148071

noncomputable def f (x a : ℝ) : ℝ := (x^2 + (a - 1) * x + 1) * Real.exp x

theorem range_of_a :
  (∀ x, f x a + Real.exp 2 ≥ 0) ↔ (-2 ≤ a ∧ a ≤ Real.exp 3 + 3) :=
sorry

end range_of_a_l1480_148071


namespace no_a_b_not_divide_bn_minus_n_l1480_148006

theorem no_a_b_not_divide_bn_minus_n :
  ∀ (a b : ℕ), 0 < a → 0 < b → ∃ (n : ℕ), 0 < n ∧ a ∣ (b^n - n) :=
by
  sorry

end no_a_b_not_divide_bn_minus_n_l1480_148006


namespace speed_ratio_l1480_148083

theorem speed_ratio (v_A v_B : ℝ) (L t : ℝ) 
  (h1 : v_A * t = (1 - 0.11764705882352941) * L)
  (h2 : v_B * t = L) : 
  v_A / v_B = 1.11764705882352941 := 
by 
  sorry

end speed_ratio_l1480_148083
