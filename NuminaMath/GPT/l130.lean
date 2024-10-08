import Mathlib

namespace arithmetic_expression_evaluation_l130_130952

theorem arithmetic_expression_evaluation :
  4 * (7 * 24) / 3 + 5 * (13 * 15) - 2 * (6 * 28) + 7 * (3 * 19) / 2 = 1062.5 := 
by
  -- Skipping the proof.
  sorry

end arithmetic_expression_evaluation_l130_130952


namespace annual_yield_range_l130_130336

-- Here we set up the conditions as definitions in Lean 4
def last_year_range : ℝ := 10000
def improvement_rate : ℝ := 0.15

-- Theorems that are based on the conditions and need proving
theorem annual_yield_range (last_year_range : ℝ) (improvement_rate : ℝ) : 
  last_year_range * (1 + improvement_rate) = 11500 := 
sorry

end annual_yield_range_l130_130336


namespace read_books_correct_l130_130143

namespace CrazySillySchool

-- Definitions from conditions
def total_books : Nat := 20
def unread_books : Nat := 5
def read_books : Nat := total_books - unread_books

-- Theorem statement
theorem read_books_correct : read_books = 15 :=
by
  -- Mathematical statement that follows from conditions and correct answer
  sorry

end CrazySillySchool

end read_books_correct_l130_130143


namespace marie_ends_with_755_l130_130123

def erasers_end (initial lost packs erasers_per_pack : ℕ) : ℕ :=
  initial - lost + packs * erasers_per_pack

theorem marie_ends_with_755 :
  erasers_end 950 420 3 75 = 755 :=
by
  sorry

end marie_ends_with_755_l130_130123


namespace isosceles_right_triangle_contains_probability_l130_130838

noncomputable def isosceles_right_triangle_probability : ℝ :=
  let leg_length := 2
  let triangle_area := (leg_length * leg_length) / 2
  let distance_radius := 1
  let quarter_circle_area := (Real.pi * (distance_radius * distance_radius)) / 4
  quarter_circle_area / triangle_area

theorem isosceles_right_triangle_contains_probability :
  isosceles_right_triangle_probability = (Real.pi / 8) :=
by
  sorry

end isosceles_right_triangle_contains_probability_l130_130838


namespace money_left_after_expenditures_l130_130634

variable (initial_amount : ℝ) (P : initial_amount = 15000)
variable (gas_percentage food_fraction clothing_fraction entertainment_percentage : ℝ) 
variable (H1 : gas_percentage = 0.35) (H2 : food_fraction = 0.2) (H3 : clothing_fraction = 0.25) (H4 : entertainment_percentage = 0.15)

theorem money_left_after_expenditures
  (money_left : ℝ):
  money_left = initial_amount * (1 - gas_percentage) *
                (1 - food_fraction) * 
                (1 - clothing_fraction) * 
                (1 - entertainment_percentage) → 
  money_left = 4972.50 :=
by
  sorry

end money_left_after_expenditures_l130_130634


namespace find_c_plus_inv_b_l130_130049

variable (a b c : ℝ)

def conditions := 
  (a * b * c = 1) ∧ 
  (a + 1/c = 7) ∧ 
  (b + 1/a = 16)

theorem find_c_plus_inv_b (h : conditions a b c) : 
  c + 1/b = 25 / 111 :=
sorry

end find_c_plus_inv_b_l130_130049


namespace largest_multiple_of_8_less_than_100_l130_130011

theorem largest_multiple_of_8_less_than_100 : ∃ x, x < 100 ∧ 8 ∣ x ∧ ∀ y, y < 100 ∧ 8 ∣ y → y ≤ x :=
by sorry

end largest_multiple_of_8_less_than_100_l130_130011


namespace total_balls_estimation_l130_130160

theorem total_balls_estimation 
  (num_red_balls : ℕ)
  (total_trials : ℕ)
  (red_ball_draws : ℕ)
  (red_ball_ratio : ℚ)
  (total_balls_estimate : ℕ)
  (h1 : num_red_balls = 5)
  (h2 : total_trials = 80)
  (h3 : red_ball_draws = 20)
  (h4 : red_ball_ratio = 1 / 4)
  (h5 : red_ball_ratio = red_ball_draws / total_trials)
  (h6 : red_ball_ratio = num_red_balls / total_balls_estimate)
  : total_balls_estimate = 20 := 
sorry

end total_balls_estimation_l130_130160


namespace m_le_n_l130_130652

theorem m_le_n (k m n : ℕ) (hk_pos : 0 < k) (hm_pos : 0 < m) (hn_pos : 0 < n) (h : m^2 + n = k^2 + k) : m ≤ n := 
sorry

end m_le_n_l130_130652


namespace geometric_sequence_formula_l130_130601

noncomputable def a_n (a_1 : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a_1 * q^n

theorem geometric_sequence_formula
  (a_1 q : ℝ)
  (h_pos : ∀ n : ℕ, a_n a_1 q n > 0)
  (h_4_eq : a_n a_1 q 4 = (a_n a_1 q 2)^2)
  (h_2_4_sum : a_n a_1 q 2 + a_n a_1 q 4 = 5 / 16) :
  ∀ n : ℕ, a_n a_1 q n = ((1 : ℝ) / 2) ^ n :=
sorry

end geometric_sequence_formula_l130_130601


namespace trains_cross_time_l130_130091

def L : ℕ := 120 -- Length of each train in meters

def t1 : ℕ := 10 -- Time for the first train to cross the telegraph post in seconds
def t2 : ℕ := 12 -- Time for the second train to cross the telegraph post in seconds

def V1 : ℕ := L / t1 -- Speed of the first train (in m/s)
def V2 : ℕ := L / t2 -- Speed of the second train (in m/s)

def Vr : ℕ := V1 + V2 -- Relative speed when traveling in opposite directions

def TotalDistance : ℕ := 2 * L -- Total distance when both trains cross each other

def T : ℚ := TotalDistance / Vr -- Time for the trains to cross each other

theorem trains_cross_time : T = 11 := sorry

end trains_cross_time_l130_130091


namespace trisha_collects_4_dozen_less_l130_130458

theorem trisha_collects_4_dozen_less (B C T : ℕ) 
  (h1 : B = 6) 
  (h2 : C = 3 * B) 
  (h3 : B + C + T = 26) : 
  B - T = 4 := 
by 
  sorry

end trisha_collects_4_dozen_less_l130_130458


namespace range_of_a_l130_130926

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (x ∈ {x : ℝ | x ≥ 3 ∨ x ≤ -1} ∩ {x : ℝ | x ≤ a} ↔ x ∈ {x : ℝ | x ≤ a})) ↔ a ≤ -1 :=
by sorry

end range_of_a_l130_130926


namespace minimum_value_of_expression_l130_130629

theorem minimum_value_of_expression {a c : ℝ} (h_pos : a > 0)
  (h_range : ∀ x, a * x ^ 2 - 4 * x + c ≥ 1) :
  ∃ a c, a > 0 ∧ (∀ x, a * x ^ 2 - 4 * x + c ≥ 1) ∧ (∃ a, a > 0 ∧ ∃ c, c - 1 = 4 / a ∧ (a / 4 + 9 / a = 3)) :=
by sorry

end minimum_value_of_expression_l130_130629


namespace bears_per_shelf_l130_130570

def bears_initial : ℕ := 6

def shipment : ℕ := 18

def shelves : ℕ := 4

theorem bears_per_shelf : (bears_initial + shipment) / shelves = 6 := by
  sorry

end bears_per_shelf_l130_130570


namespace find_n_l130_130012

open Nat

theorem find_n (d : ℕ → ℕ) (n : ℕ) (h1 : ∀ j, d (j + 1) > d j) (h2 : n = d 13 + d 14 + d 15) (h3 : (d 5 + 1)^3 = d 15 + 1) : 
  n = 1998 :=
by
  sorry

end find_n_l130_130012


namespace product_of_roots_proof_l130_130757

noncomputable def product_of_roots : ℚ :=
  let leading_coeff_poly1 := 3
  let leading_coeff_poly2 := 4
  let constant_term_poly1 := -15
  let constant_term_poly2 := 9
  let a := leading_coeff_poly1 * leading_coeff_poly2
  let b := constant_term_poly1 * constant_term_poly2
  (b : ℚ) / a

theorem product_of_roots_proof :
  product_of_roots = -45/4 :=
by
  sorry

end product_of_roots_proof_l130_130757


namespace probability_of_3_tails_in_8_flips_l130_130847

open ProbabilityTheory

/-- The probability of getting exactly 3 tails out of 8 flips of an unfair coin, where the probability of tails is 4/5 and the probability of heads is 1/5, is 3584/390625. -/
theorem probability_of_3_tails_in_8_flips :
  let p_heads := 1 / 5
  let p_tails := 4 / 5
  let n_trials := 8
  let k_successes := 3
  let binomial_coefficient := Nat.choose n_trials k_successes
  let probability := binomial_coefficient * (p_tails ^ k_successes) * (p_heads ^ (n_trials - k_successes))
  probability = (3584 : ℚ) / 390625 := 
by 
  sorry

end probability_of_3_tails_in_8_flips_l130_130847


namespace chemistry_club_student_count_l130_130412

theorem chemistry_club_student_count (x : ℕ) (h1 : x % 3 = 0)
  (h2 : x % 4 = 0) (h3 : x % 6 = 0)
  (h4 : (x / 3) = (x / 4) + 3) :
  (x / 6) = 6 :=
by {
  -- Proof goes here
  sorry
}

end chemistry_club_student_count_l130_130412


namespace john_max_questions_correct_l130_130782

variable (c w b : ℕ)

theorem john_max_questions_correct (H1 : c + w + b = 20) (H2 : 5 * c - 2 * w = 48) : c ≤ 12 := sorry

end john_max_questions_correct_l130_130782


namespace seq_le_n_squared_l130_130691

theorem seq_le_n_squared (a : ℕ → ℕ) (h_increasing : ∀ n, a n < a (n + 1))
  (h_positive : ∀ n, 0 < a n)
  (h_property : ∀ t, ∃ i j, t = a i ∨ t = a i + a j) :
  ∀ n, a n ≤ n^2 :=
by {
  sorry
}

end seq_le_n_squared_l130_130691


namespace ten_row_geometric_figure_has_286_pieces_l130_130086

noncomputable def rods (rows : ℕ) : ℕ := 3 * rows * (rows + 1) / 2
noncomputable def connectors (rows : ℕ) : ℕ := (rows +1) * (rows + 2) / 2
noncomputable def squares (rows : ℕ) : ℕ := rows * (rows + 1) / 2

theorem ten_row_geometric_figure_has_286_pieces :
    rods 10 + connectors 10 + squares 10 = 286 := by
  sorry

end ten_row_geometric_figure_has_286_pieces_l130_130086


namespace problem_1_problem_2_l130_130380

def f (x : ℝ) : ℝ := abs (x - 2)
def g (x m : ℝ) : ℝ := -abs (x + 7) + 3 * m

theorem problem_1 (x : ℝ) : f x + x^2 - 4 > 0 ↔ (x > 2 ∨ x < -1) := sorry

theorem problem_2 {m : ℝ} (h : m > 3) : ∃ x : ℝ, f x < g x m := sorry

end problem_1_problem_2_l130_130380


namespace modular_arithmetic_proof_l130_130233

open Nat

theorem modular_arithmetic_proof (m : ℕ) (h0 : 0 ≤ m ∧ m < 37) (h1 : 4 * m ≡ 1 [MOD 37]) :
  (3^m)^4 ≡ 27 + 3 [MOD 37] :=
by
  -- Although some parts like modular inverse calculation or finding specific m are skipped,
  -- the conclusion directly should reflect (3^m)^4 ≡ 27 + 3 [MOD 37]
  -- Considering (3^m)^4 - 3 ≡ 24 [MOD 37] translates to the above statement
  sorry

end modular_arithmetic_proof_l130_130233


namespace sqrt_six_plus_s_cubed_l130_130899

theorem sqrt_six_plus_s_cubed (s : ℝ) : 
    Real.sqrt (s^6 + s^3) = |s| * Real.sqrt (s * (s^3 + 1)) :=
sorry

end sqrt_six_plus_s_cubed_l130_130899


namespace clothing_store_earnings_l130_130008

-- Definitions for the given conditions
def num_shirts : ℕ := 20
def num_jeans : ℕ := 10
def cost_per_shirt : ℕ := 10
def cost_per_jeans : ℕ := 2 * cost_per_shirt

-- Theorem statement
theorem clothing_store_earnings : 
  (num_shirts * cost_per_shirt + num_jeans * cost_per_jeans = 400) := 
sorry

end clothing_store_earnings_l130_130008


namespace find_x1_l130_130815

theorem find_x1 (x1 x2 x3 : ℝ) (h1 : 0 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1)
  (h2 : (1-x1)^2 + 2*(x1-x2)^2 + (x2-x3)^2 + x3^2 = 1/2) :
  x1 = (3*Real.sqrt 2 - 3)/7 :=
by
  sorry

end find_x1_l130_130815


namespace at_least_one_irrational_l130_130662

theorem at_least_one_irrational (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) 
  (h₃ : a^2 * b^2 * (a^2 * b^2 + 4) = 2 * (a^6 + b^6)) : 
  ¬ (∀ a b : ℚ, a ≠ 0 ∧ b ≠ 0 → a^2 * b^2 * (a^2 * b^2 + 4) = 2 * (a^6 + b^6)) :=
by sorry

end at_least_one_irrational_l130_130662


namespace logan_passengers_count_l130_130434

noncomputable def passengers_used_Kennedy_Airport : ℝ := (1 / 3) * 38.3
noncomputable def passengers_used_Miami_Airport : ℝ := (1 / 2) * passengers_used_Kennedy_Airport
noncomputable def passengers_used_Logan_Airport : ℝ := passengers_used_Miami_Airport / 4

theorem logan_passengers_count : abs (passengers_used_Logan_Airport - 1.6) < 0.01 := by
  sorry

end logan_passengers_count_l130_130434


namespace solve_for_c_l130_130171

noncomputable def f (c : ℝ) (x : ℝ) : ℝ := (c * x) / (2 * x + 3)

theorem solve_for_c {c : ℝ} (hc : ∀ x ≠ (-3/2), f c (f c x) = x) : c = -3 :=
by
  intros
  -- The proof steps will go here
  sorry

end solve_for_c_l130_130171


namespace determine_k_for_circle_l130_130777

theorem determine_k_for_circle (x y k : ℝ) (h : x^2 + 14*x + y^2 + 8*y - k = 0) (r : ℝ) :
  r = 5 → k = 40 :=
by
  intros radius_eq_five
  sorry

end determine_k_for_circle_l130_130777


namespace circle_line_intersection_l130_130060

theorem circle_line_intersection (x y a : ℝ) (A B C O : ℝ × ℝ) :
  (x + y = 1) ∧ ((x^2 + y^2) = a) ∧ 
  (O = (0, 0)) ∧ 
  (x^2 + y^2 = a ∧ (A.1^2 + A.2^2 = a) ∧ (B.1^2 + B.2^2 = a) ∧ (C.1^2 + C.2^2 = a) ∧ 
  (A.1 + B.1 = C.1) ∧ (A.2 + B.2 = C.2)) -> 
  a = 2 := 
sorry

end circle_line_intersection_l130_130060


namespace trips_needed_l130_130058

def barbieCapacity : Nat := 4
def brunoCapacity : Nat := 8
def totalCoconuts : Nat := 144

theorem trips_needed : (totalCoconuts / (barbieCapacity + brunoCapacity)) = 12 := by
  sorry

end trips_needed_l130_130058


namespace exponent_division_l130_130223

theorem exponent_division : (23 ^ 11) / (23 ^ 8) = 12167 := 
by {
  sorry
}

end exponent_division_l130_130223


namespace pizzasServedDuringDinner_l130_130077

-- Definitions based on the conditions
def pizzasServedDuringLunch : ℕ := 9
def totalPizzasServedToday : ℕ := 15

-- Theorem statement
theorem pizzasServedDuringDinner : 
  totalPizzasServedToday - pizzasServedDuringLunch = 6 := 
  by 
    sorry

end pizzasServedDuringDinner_l130_130077


namespace cost_of_fencing_l130_130936

theorem cost_of_fencing (d : ℝ) (rate : ℝ) (C : ℝ) (cost : ℝ) : 
  d = 22 → rate = 3 → C = Real.pi * d → cost = C * rate → cost = 207 :=
by
  intros
  sorry

end cost_of_fencing_l130_130936


namespace opposite_face_of_X_is_Y_l130_130753

-- Define the labels for the cube faces
inductive Label
| X | V | Z | W | U | Y

-- Define adjacency relations
def adjacent (a b : Label) : Prop :=
  (a = Label.X ∧ (b = Label.V ∨ b = Label.Z ∨ b = Label.W ∨ b = Label.U)) ∨
  (b = Label.X ∧ (a = Label.V ∨ a = Label.Z ∨ a = Label.W ∨ a = Label.U))

-- Define the theorem to prove the face opposite to X
theorem opposite_face_of_X_is_Y : ∀ l1 l2 l3 l4 l5 l6 : Label,
  l1 = Label.X →
  l2 = Label.V →
  l3 = Label.Z →
  l4 = Label.W →
  l5 = Label.U →
  l6 = Label.Y →
  ¬ adjacent l1 l6 →
  ¬ adjacent l2 l6 →
  ¬ adjacent l3 l6 →
  ¬ adjacent l4 l6 →
  ¬ adjacent l5 l6 →
  ∃ (opposite : Label), opposite = Label.Y ∧ opposite = l6 :=
by sorry

end opposite_face_of_X_is_Y_l130_130753


namespace new_selling_price_l130_130205

theorem new_selling_price (C : ℝ) (h1 : 1.10 * C = 88) :
  1.15 * C = 92 :=
sorry

end new_selling_price_l130_130205


namespace no_solution_xy_in_nat_star_l130_130752

theorem no_solution_xy_in_nat_star (x y : ℕ) (hx : 0 < x) (hy : 0 < y) : ¬ (x * (x + 1) = 4 * y * (y + 1)) :=
by
  -- The proof would go here, but we'll leave it out for now.
  sorry

end no_solution_xy_in_nat_star_l130_130752


namespace car_and_bicycle_distances_l130_130795

noncomputable def train_speed : ℝ := 100 -- speed of the train in mph
noncomputable def car_speed : ℝ := (2 / 3) * train_speed -- speed of the car in mph
noncomputable def bicycle_speed : ℝ := (1 / 5) * train_speed -- speed of the bicycle in mph
noncomputable def travel_time_hours : ℝ := 30 / 60 -- travel time in hours, which is 0.5 hours

noncomputable def car_distance : ℝ := car_speed * travel_time_hours
noncomputable def bicycle_distance : ℝ := bicycle_speed * travel_time_hours

theorem car_and_bicycle_distances :
  car_distance = 100 / 3 ∧ bicycle_distance = 10 :=
by
  sorry

end car_and_bicycle_distances_l130_130795


namespace z_value_l130_130547

theorem z_value (z : ℝ) (h : |z + 2| = |z - 3|) : z = 1 / 2 := 
sorry

end z_value_l130_130547


namespace books_per_shelf_l130_130755

theorem books_per_shelf (mystery_shelves picture_shelves total_books : ℕ)
    (h₁ : mystery_shelves = 5)
    (h₂ : picture_shelves = 3)
    (h₃ : total_books = 32) :
    (total_books / (mystery_shelves + picture_shelves) = 4) :=
by
    sorry

end books_per_shelf_l130_130755


namespace find_number_l130_130519

theorem find_number (x : ℝ) (h : (2 * x - 37 + 25) / 8 = 5) : x = 26 :=
sorry

end find_number_l130_130519


namespace fgf_one_l130_130311

/-- Define the function f(x) = 5x + 2 --/
def f (x : ℝ) := 5 * x + 2

/-- Define the function g(x) = 3x - 1 --/
def g (x : ℝ) := 3 * x - 1

/-- Prove that f(g(f(1))) = 102 given the definitions of f and g --/
theorem fgf_one : f (g (f 1)) = 102 := by
  sorry

end fgf_one_l130_130311


namespace function_two_common_points_with_xaxis_l130_130390

theorem function_two_common_points_with_xaxis (c : ℝ) :
  (∀ x : ℝ, x^3 - 3 * x + c = 0 → x = -1 ∨ x = 1) → (c = -2 ∨ c = 2) :=
by
  sorry

end function_two_common_points_with_xaxis_l130_130390


namespace faculty_reduction_l130_130363

theorem faculty_reduction (x : ℝ) (h1 : 0.75 * x = 195) : x = 260 :=
by sorry

end faculty_reduction_l130_130363


namespace find_positive_integer_l130_130027

theorem find_positive_integer (x : ℕ) (h1 : (10 * x + 4) % (x + 4) = 0) (h2 : (10 * x + 4) / (x + 4) = x - 23) : x = 32 :=
by
  sorry

end find_positive_integer_l130_130027


namespace system_of_equations_correct_l130_130430

def weight_system (x y : ℝ) : Prop :=
  (5 * x + 6 * y = 1) ∧ (3 * x = y)

theorem system_of_equations_correct (x y : ℝ) :
  weight_system x y ↔ 
    (5 * x + 6 * y = 1) ∧ (4 * x + 7 * y = 5 * x + 6 * y) :=
by sorry

end system_of_equations_correct_l130_130430


namespace total_mile_times_l130_130873

theorem total_mile_times (t_Tina t_Tony t_Tom t_Total : ℕ) 
  (h1 : t_Tina = 6) 
  (h2 : t_Tony = t_Tina / 2) 
  (h3 : t_Tom = t_Tina / 3) 
  (h4 : t_Total = t_Tina + t_Tony + t_Tom) : t_Total = 11 := 
sorry

end total_mile_times_l130_130873


namespace inequality_proof_l130_130217

variable (a b c : ℝ)
variable (h1 : a > 0)
variable (h2 : b > 0)
variable (h3 : c > 0)
variable (h4 : a + b + c = 1)

theorem inequality_proof : 
  (a + 1/a)^3 + (b + 1/b)^3 + (c + 1/c)^3 ≥ 1000/9 := 
by 
  sorry

end inequality_proof_l130_130217


namespace lowest_test_score_dropped_l130_130887

theorem lowest_test_score_dropped (S L : ℕ)
  (h1 : S = 5 * 42) 
  (h2 : S - L = 4 * 48) : 
  L = 18 :=
by
  sorry

end lowest_test_score_dropped_l130_130887


namespace running_distance_l130_130281

theorem running_distance (D : ℕ) 
  (hA_time : ∀ (A_time : ℕ), A_time = 28) 
  (hB_time : ∀ (B_time : ℕ), B_time = 32) 
  (h_lead : ∀ (lead : ℕ), lead = 28) 
  (hA_speed : ∀ (A_speed : ℚ), A_speed = D / 28) 
  (hB_speed : ∀ (B_speed : ℚ), B_speed = D / 32) 
  (hB_dist : ∀ (B_dist : ℚ), B_dist = D - 28) 
  (h_eq : ∀ (B_dist : ℚ), B_dist = D * (28 / 32)) :
  D = 224 :=
by 
  sorry

end running_distance_l130_130281


namespace sufficient_not_necessary_l130_130227

theorem sufficient_not_necessary (a : ℝ) :
  a > 1 → (a^2 > 1) ∧ (∀ a : ℝ, a^2 > 1 → a = -1 ∨ a > 1 → false) :=
by {
  sorry
}

end sufficient_not_necessary_l130_130227


namespace eccentricity_of_ellipse_equilateral_triangle_l130_130259

theorem eccentricity_of_ellipse_equilateral_triangle (c b a e : ℝ)
  (h1 : b = Real.sqrt (3 * c))
  (h2 : a = Real.sqrt (b^2 + c^2)) 
  (h3 : e = c / a) :
  e = 1 / 2 :=
by {
  sorry
}

end eccentricity_of_ellipse_equilateral_triangle_l130_130259


namespace arithmetic_seq_min_S19_l130_130998

noncomputable def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n * (a 1 + a n) / 2

theorem arithmetic_seq_min_S19
  (a : ℕ → ℝ) (d : ℝ)
  (h_seq : ∀ n, a (n + 1) = a n + d)
  (h_S8 : S a 8 ≤ 6)
  (h_S11 : S a 11 ≥ 27) :
  S a 19 ≥ 133 :=
sorry

end arithmetic_seq_min_S19_l130_130998


namespace find_integer_n_cos_l130_130625

theorem find_integer_n_cos : ∃ n : ℤ, 0 ≤ n ∧ n ≤ 180 ∧ (Real.cos (n * Real.pi / 180) = Real.cos (1124 * Real.pi / 180)) ∧ n = 44 := by
  sorry

end find_integer_n_cos_l130_130625


namespace subtract_from_sum_base8_l130_130965

def add_in_base_8 (a b : ℕ) : ℕ :=
  ((a % 8) + (b % 8)) % 8
  + (((a / 8) % 8 + (b / 8) % 8 + ((a % 8) + (b % 8)) / 8) % 8) * 8
  + (((((a / 8) % 8 + (b / 8) % 8 + ((a % 8) + (b % 8)) / 8) / 8) + ((a / 64) % 8 + (b / 64) % 8)) % 8) * 64

def subtract_in_base_8 (a b : ℕ) : ℕ :=
  ((a % 8) - (b % 8) + 8) % 8
  + (((a / 8) % 8 - (b / 8) % 8 - if (a % 8) < (b % 8) then 1 else 0 + 8) % 8) * 8
  + (((a / 64) - (b / 64) - if (a / 8) % 8 < (b / 8) % 8 then 1 else 0) % 8) * 64

theorem subtract_from_sum_base8 :
  subtract_in_base_8 (add_in_base_8 652 147) 53 = 50 := by
  sorry

end subtract_from_sum_base8_l130_130965


namespace pages_left_l130_130835

-- Define the conditions
def initial_books := 10
def pages_per_book := 100
def books_lost := 2

-- The total pages Phil had initially
def initial_pages := initial_books * pages_per_book

-- The number of books left after losing some during the move
def books_left := initial_books - books_lost

-- Prove the number of pages worth of books Phil has left
theorem pages_left : books_left * pages_per_book = 800 := by
  sorry

end pages_left_l130_130835


namespace find_percent_l130_130062

theorem find_percent (x y z : ℝ) (h1 : z * (x - y) = 0.15 * (x + y)) (h2 : y = 0.25 * x) : 
  z = 0.25 := 
sorry

end find_percent_l130_130062


namespace rowing_speed_l130_130282

theorem rowing_speed (V_m V_w V_upstream V_downstream : ℝ)
  (h1 : V_upstream = 25)
  (h2 : V_downstream = 65)
  (h3 : V_w = 5) :
  V_m = 45 :=
by
  -- Lean will verify the theorem given the conditions
  sorry

end rowing_speed_l130_130282


namespace two_beta_plus_alpha_eq_pi_div_two_l130_130762

theorem two_beta_plus_alpha_eq_pi_div_two
  (α β : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2)
  (hβ1 : 0 < β) (hβ2 : β < π / 2)
  (h : Real.tan α + Real.tan β = 1 / Real.cos α) :
  2 * β + α = π / 2 :=
sorry

end two_beta_plus_alpha_eq_pi_div_two_l130_130762


namespace batsman_average_after_11th_inning_l130_130147

theorem batsman_average_after_11th_inning
  (x : ℝ)  -- the average score of the batsman before the 11th inning
  (h1 : 10 * x + 85 = 11 * (x + 5))  -- given condition from the problem
  : x + 5 = 35 :=   -- goal statement proving the new average
by
  -- We need to prove that new average after the 11th inning is 35
  sorry

end batsman_average_after_11th_inning_l130_130147


namespace edward_initial_money_l130_130669

theorem edward_initial_money (cars qty : Nat) (car_cost race_track_cost left_money initial_money : ℝ) 
    (h1 : cars = 4) 
    (h2 : car_cost = 0.95) 
    (h3 : race_track_cost = 6.00)
    (h4 : left_money = 8.00)
    (h5 : initial_money = (cars * car_cost) + race_track_cost + left_money) :
  initial_money = 17.80 := sorry

end edward_initial_money_l130_130669


namespace individual_weight_l130_130922

def total_students : ℕ := 1500
def sampled_students : ℕ := 100

def individual := "the weight of each student"

theorem individual_weight :
  (total_students = 1500) →
  (sampled_students = 100) →
  individual = "the weight of each student" :=
by
  intros h1 h2
  sorry

end individual_weight_l130_130922


namespace solve_problem_l130_130254

def spadesuit (x y : ℝ) : ℝ := x^2 + y^2

theorem solve_problem : spadesuit (spadesuit 3 5) 4 = 1172 := by
  sorry

end solve_problem_l130_130254


namespace price_difference_l130_130263

/-- Given an original price, two successive price increases, and special deal prices for a fixed number of items, 
    calculate the difference between the final retail price and the average special deal price. -/
theorem price_difference
  (original_price : ℝ) (first_increase_percent: ℝ) (second_increase_percent: ℝ)
  (special_deal_percent_1: ℝ) (num_items_1: ℕ) (special_deal_percent_2: ℝ) (num_items_2: ℕ)
  (final_retail_price : ℝ) (average_special_deal_price : ℝ) :
  original_price = 50 →
  first_increase_percent = 0.30 →
  second_increase_percent = 0.15 →
  special_deal_percent_1 = 0.70 →
  num_items_1 = 50 →
  special_deal_percent_2 = 0.85 →
  num_items_2 = 100 →
  final_retail_price = original_price * (1 + first_increase_percent) * (1 + second_increase_percent) →
  average_special_deal_price = 
    (num_items_1 * (special_deal_percent_1 * final_retail_price) + 
    num_items_2 * (special_deal_percent_2 * final_retail_price)) / 
    (num_items_1 + num_items_2) →
  final_retail_price - average_special_deal_price = 14.95 :=
by
  intros
  sorry

end price_difference_l130_130263


namespace white_roses_per_bouquet_l130_130098

/-- Mrs. Dunbar needs to make 5 bouquets and 7 table decorations. -/
def number_of_bouquets : ℕ := 5
def number_of_table_decorations : ℕ := 7
/-- She uses 12 white roses in each table decoration. -/
def white_roses_per_table_decoration : ℕ := 12
/-- She needs a total of 109 white roses to complete all bouquets and table decorations. -/
def total_white_roses_needed : ℕ := 109

/-- Prove that the number of white roses used in each bouquet is 5. -/
theorem white_roses_per_bouquet : ∃ (white_roses_per_bouquet : ℕ),
  number_of_bouquets * white_roses_per_bouquet + number_of_table_decorations * white_roses_per_table_decoration = total_white_roses_needed
  ∧ white_roses_per_bouquet = 5 := 
by
  sorry

end white_roses_per_bouquet_l130_130098


namespace volume_of_rectangular_box_l130_130087

theorem volume_of_rectangular_box (x y z : ℝ) 
  (h1 : x * y = 15) 
  (h2 : y * z = 20) 
  (h3 : x * z = 12) : 
  x * y * z = 60 := 
sorry

end volume_of_rectangular_box_l130_130087


namespace find_f2_l130_130017

noncomputable def f (x : ℝ) : ℝ := (4*x + 2/x + 3) / 3

theorem find_f2 (f : ℝ → ℝ) (h : ∀ x : ℝ, 2 * f x - f (1 / x) = 2 * x + 1) : f 2 = 4 :=
  by
  sorry

end find_f2_l130_130017


namespace polar_equation_C1_intersection_C2_C1_distance_l130_130740

noncomputable def parametric_to_cartesian (α : ℝ) : Prop :=
  ∃ (x y : ℝ), x = 2 + 2 * Real.cos α ∧ y = 4 + 2 * Real.sin α

noncomputable def cartesian_to_polar (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 4)^2 = 4

noncomputable def polar_equation_of_C1 (ρ θ : ℝ) : Prop :=
  ρ^2 - 4 * ρ * Real.cos θ - 8 * ρ * Real.sin θ + 16 = 0

noncomputable def C2_line_polar (θ : ℝ) : Prop :=
  θ = Real.pi / 4

theorem polar_equation_C1 (α : ℝ) (ρ θ : ℝ) :
  parametric_to_cartesian α →
  cartesian_to_polar (2 + 2 * Real.cos α) (4 + 2 * Real.sin α) →
  polar_equation_of_C1 ρ θ :=
by
  sorry

theorem intersection_C2_C1_distance (ρ θ : ℝ) (t1 t2 : ℝ) :
  C2_line_polar θ →
  polar_equation_of_C1 ρ θ →
  (t1 + t2 = 6 * Real.sqrt 2) ∧ (t1 * t2 = 16) →
  |t1 - t2| = 2 * Real.sqrt 2 :=
by
  sorry

end polar_equation_C1_intersection_C2_C1_distance_l130_130740


namespace complex_number_modulus_l130_130370

open Complex

theorem complex_number_modulus :
  ∀ x : ℂ, x + I = (2 - I) / I → abs x = Real.sqrt 10 := by
  sorry

end complex_number_modulus_l130_130370


namespace remainder_when_divided_l130_130696

theorem remainder_when_divided (L S R : ℕ) (h1: L - S = 1365) (h2: S = 270) (h3: L = 6 * S + R) : 
  R = 15 := 
by 
  sorry

end remainder_when_divided_l130_130696


namespace pool_filling_time_l130_130901

theorem pool_filling_time (rate_jim rate_sue rate_tony : ℝ) (h1 : rate_jim = 1 / 30) (h2 : rate_sue = 1 / 45) (h3 : rate_tony = 1 / 90) : 
     1 / (rate_jim + rate_sue + rate_tony) = 15 := by
  sorry

end pool_filling_time_l130_130901


namespace beth_marbles_left_l130_130410

theorem beth_marbles_left :
  let T := 72
  let C := T / 3
  let L_red := 5
  let L_blue := 2 * L_red
  let L_yellow := 3 * L_red
  T - (L_red + L_blue + L_yellow) = 42 :=
by
  let T := 72
  let C := T / 3
  let L_red := 5
  let L_blue := 2 * L_red
  let L_yellow := 3 * L_red
  have h1 : T - (L_red + L_blue + L_yellow) = 42 := rfl
  exact h1

end beth_marbles_left_l130_130410


namespace surface_area_hemisphere_radius_1_l130_130093

noncomputable def surface_area_hemisphere (r : ℝ) : ℝ :=
  2 * Real.pi * r^2 + Real.pi * r^2

theorem surface_area_hemisphere_radius_1 :
  surface_area_hemisphere 1 = 3 * Real.pi :=
by
  sorry

end surface_area_hemisphere_radius_1_l130_130093


namespace sum_of_first_15_terms_of_arithmetic_sequence_l130_130216

theorem sum_of_first_15_terms_of_arithmetic_sequence 
  (a d : ℕ) 
  (h1 : (5 * (2 * a + 4 * d)) / 2 = 10) 
  (h2 : (10 * (2 * a + 9 * d)) / 2 = 50) :
  (15 * (2 * a + 14 * d)) / 2 = 120 :=
sorry

end sum_of_first_15_terms_of_arithmetic_sequence_l130_130216


namespace Q_polynomial_l130_130746

def cos3x_using_cos2x (cos_α : ℝ) := (2 * cos_α^2 - 1) * cos_α - 2 * (1 - cos_α^2) * cos_α

def Q (x : ℝ) := 4 * x^3 - 3 * x

theorem Q_polynomial (α : ℝ) : Q (Real.cos α) = Real.cos (3 * α) := by
  rw [Real.cos_three_mul]
  sorry

end Q_polynomial_l130_130746


namespace friday_vs_tuesday_l130_130502

def tuesday_amount : ℝ := 8.5
def wednesday_amount : ℝ := 5.5 * tuesday_amount
def thursday_amount : ℝ := wednesday_amount + 0.10 * wednesday_amount
def friday_amount : ℝ := 0.75 * thursday_amount

theorem friday_vs_tuesday :
  friday_amount - tuesday_amount = 30.06875 :=
sorry

end friday_vs_tuesday_l130_130502


namespace min_value_of_frac_expr_l130_130733

theorem min_value_of_frac_expr (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  (3 / a) + (2 / b) ≥ 5 + 2 * Real.sqrt 6 :=
sorry

end min_value_of_frac_expr_l130_130733


namespace farmer_ducks_sold_l130_130876

theorem farmer_ducks_sold (D : ℕ) (earnings : ℕ) :
  (earnings = (10 * D) + (5 * 8)) →
  ((earnings / 2) * 2 = 60) →
  D = 2 := by
  sorry

end farmer_ducks_sold_l130_130876


namespace infection_never_covers_grid_l130_130174

theorem infection_never_covers_grid (n : ℕ) (H : n > 0) :
  exists (non_infected_cell : ℕ × ℕ), (non_infected_cell.1 < n ∧ non_infected_cell.2 < n) :=
by
  sorry

end infection_never_covers_grid_l130_130174


namespace eq_iff_solution_l130_130854

theorem eq_iff_solution (x y : ℕ) (hx : 0 < x) (hy : 0 < y) :
  x^y + y^x = x^x + y^y ↔ x = y :=
by sorry

end eq_iff_solution_l130_130854


namespace inverse_proportion_l130_130427

theorem inverse_proportion (α β k : ℝ) (h1 : α * β = k) (h2 : α = 5) (h3 : β = 10) : (α = 25 / 2) → (β = 4) := by sorry

end inverse_proportion_l130_130427


namespace partition_displacement_l130_130496

variables (l : ℝ) (R T : ℝ) (initial_V1 initial_V2 final_Vleft final_Vright initial_P1 initial_P2 : ℝ)

-- Conditions
def initial_conditions (initial_V1 initial_V2 : ℝ) : Prop :=
  initial_V1 + initial_V2 = l ∧
  initial_V2 = 2 * initial_V1 ∧
  initial_P1 * initial_V1 = R * T ∧
  initial_P2 * initial_V2 = 2 * R * T ∧
  initial_P1 = initial_P2

-- Final volumes
def final_volumes (final_Vleft final_Vright : ℝ) : Prop :=
  final_Vleft = l / 2 ∧ final_Vright = l / 2 

-- Displacement of the partition
def displacement (initial_position final_position : ℝ) : ℝ :=
  initial_position - final_position

-- Theorem statement: the displacement of the partition is l / 6
theorem partition_displacement (l R T initial_V1 initial_V2 final_Vleft final_Vright initial_P1 initial_P2 : ℝ)
  (h_initial_cond : initial_conditions l R T initial_V1 initial_V2 initial_P1 initial_P2)
  (h_final_vol : final_volumes l final_Vleft final_Vright) 
  (initial_position final_position : ℝ)
  (initial_position_def : initial_position = 2 * l / 3)
  (final_position_def : final_position = l / 2) :
  displacement initial_position final_position = l / 6 := 
by sorry

end partition_displacement_l130_130496


namespace xiao_li_estimate_l130_130364

variable (x y z : ℝ)

theorem xiao_li_estimate (h1 : x > y) (h2 : y > 0) (h3 : 0 < z):
    (x + z) + (y - z) = x + y := 
by 
sorry

end xiao_li_estimate_l130_130364


namespace cube_polygon_area_l130_130727

theorem cube_polygon_area (cube_side : ℝ) 
  (A B C D : ℝ × ℝ × ℝ)
  (P Q R : ℝ × ℝ × ℝ)
  (hP : P = (10, 0, 0))
  (hQ : Q = (30, 0, 20))
  (hR : R = (30, 5, 30))
  (hA : A = (0, 0, 0))
  (hB : B = (30, 0, 0))
  (hC : C = (30, 0, 30))
  (hD : D = (30, 30, 30))
  (cube_length : cube_side = 30) :
  ∃ area, area = 450 := 
sorry

end cube_polygon_area_l130_130727


namespace minimum_omega_l130_130200

theorem minimum_omega (ω : ℝ) (k : ℤ) (hω : ω > 0) 
  (h_symmetry : ∃ k : ℤ, ω * (π / 12) + π / 6 = k * π + π / 2) : ω = 4 :=
sorry

end minimum_omega_l130_130200


namespace sqrt_domain_l130_130056

theorem sqrt_domain (x : ℝ) : 3 * x - 6 ≥ 0 ↔ x ≥ 2 := sorry

end sqrt_domain_l130_130056


namespace polynomial_coeff_sum_l130_130636

theorem polynomial_coeff_sum :
  let p1 : Polynomial ℝ := Polynomial.C 4 * Polynomial.X ^ 2 - Polynomial.C 6 * Polynomial.X + Polynomial.C 5
  let p2 : Polynomial ℝ := Polynomial.C 8 - Polynomial.C 3 * Polynomial.X
  let product : Polynomial ℝ := p1 * p2
  let a : ℝ := - (product.coeff 3)
  let b : ℝ := (product.coeff 2)
  let c : ℝ := - (product.coeff 1)
  let d : ℝ := (product.coeff 0)
  8 * a + 4 * b + 2 * c + d = 18 := sorry

end polynomial_coeff_sum_l130_130636


namespace find_a_l130_130864

theorem find_a (a b c : ℕ) (h1 : a + b = c) (h2 : b + c = 5) (h3 : c = 3) : a = 1 := by
  sorry

end find_a_l130_130864


namespace q1_monotonic_increasing_intervals_q2_proof_l130_130383

noncomputable def f (a x : ℝ) : ℝ := (1 / 2) * a * x^2 - (2 * a + 1) * x + 2 * Real.log x

theorem q1_monotonic_increasing_intervals (a : ℝ) (h : a > 0) :
  (a > 1/2 ∧ (∀ x, (0 < x ∧ x < 1/a) ∨ (2 < x) → f a x > 0)) ∨
  (a = 1/2 ∧ (∀ x, 0 < x → f a x ≥ 0)) ∨
  (0 < a ∧ a < 1/2 ∧ (∀ x, (0 < x ∧ x < 2) ∨ (1/a < x) → f a x > 0)) := sorry

theorem q2_proof (x : ℝ) :
  (a = 0 ∧ x > 0 → f 0 x < 2 * Real.exp x - x - 4) := sorry

end q1_monotonic_increasing_intervals_q2_proof_l130_130383


namespace Dylan_needs_two_trays_l130_130175

noncomputable def ice_cubes_glass : ℕ := 8
noncomputable def ice_cubes_pitcher : ℕ := 2 * ice_cubes_glass
noncomputable def tray_capacity : ℕ := 12
noncomputable def total_ice_cubes_used : ℕ := ice_cubes_glass + ice_cubes_pitcher
noncomputable def number_of_trays : ℕ := total_ice_cubes_used / tray_capacity

theorem Dylan_needs_two_trays : number_of_trays = 2 := by
  sorry

end Dylan_needs_two_trays_l130_130175


namespace product_of_three_consecutive_integers_is_square_l130_130517

theorem product_of_three_consecutive_integers_is_square (x : ℤ) : 
  ∃ n : ℤ, x * (x + 1) * (x + 2) = n^2 → x = 0 ∨ x = -1 ∨ x = -2 :=
by
  sorry

end product_of_three_consecutive_integers_is_square_l130_130517


namespace test_unanswered_one_way_l130_130393

theorem test_unanswered_one_way (Q A : ℕ) (hQ : Q = 4) (hA : A = 5):
  ∀ (unanswered : ℕ), (unanswered = 1) :=
by
  intros
  sorry

end test_unanswered_one_way_l130_130393


namespace value_of_x_l130_130340

theorem value_of_x {x y z w v : ℝ} 
  (h1 : y * x = 3)
  (h2 : z = 3)
  (h3 : w = z * y)
  (h4 : v = w * z)
  (h5 : v = 18)
  (h6 : w = 6) :
  x = 3 / 2 :=
by
  sorry

end value_of_x_l130_130340


namespace projectile_reaches_75_feet_l130_130671

def projectile_height (t : ℝ) : ℝ := -16 * t^2 + 80 * t

theorem projectile_reaches_75_feet :
  ∃ t : ℝ, projectile_height t = 75 ∧ t = 1.25 :=
by
  -- Skipping the proof as instructed
  sorry

end projectile_reaches_75_feet_l130_130671


namespace parallel_lines_slope_l130_130870

theorem parallel_lines_slope (m : ℝ) 
  (h1 : ∀ x y : ℝ, x + 2 * y - 1 = 0 → x = -2 * y + 1)
  (h2 : ∀ x y : ℝ, m * x - y = 0 → y = m * x) : 
  m = -1 / 2 :=
by
  sorry

end parallel_lines_slope_l130_130870


namespace evaluate_T_l130_130645

def T (a b : ℤ) : ℤ := 4 * a - 7 * b

theorem evaluate_T : T 6 3 = 3 := by
  sorry

end evaluate_T_l130_130645


namespace remaining_distance_l130_130195

theorem remaining_distance (total_depth distance_traveled remaining_distance : ℕ) (h_total_depth : total_depth = 1218) 
  (h_distance_traveled : distance_traveled = 849) : remaining_distance = total_depth - distance_traveled := 
by
  sorry

end remaining_distance_l130_130195


namespace slope_of_tangent_at_1_l130_130173

noncomputable def f (x : ℝ) : ℝ := Real.sqrt x

theorem slope_of_tangent_at_1 : (deriv f 1) = 1 / 2 :=
  by
  sorry

end slope_of_tangent_at_1_l130_130173


namespace minimum_value_l130_130731

noncomputable def polynomial_expr (x y : ℝ) : ℝ :=
  3 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 4 * y + 5

theorem minimum_value : ∃ x y : ℝ, (polynomial_expr x y = 8) := 
sorry

end minimum_value_l130_130731


namespace jerry_needs_money_l130_130001

theorem jerry_needs_money
  (jerry_has : ℕ := 7)
  (total_needed : ℕ := 16)
  (cost_per_figure : ℕ := 8) :
  (total_needed - jerry_has) * cost_per_figure = 72 :=
by
  sorry

end jerry_needs_money_l130_130001


namespace total_fish_l130_130038

theorem total_fish (fish_Lilly fish_Rosy : ℕ) (hL : fish_Lilly = 10) (hR : fish_Rosy = 8) : fish_Lilly + fish_Rosy = 18 := 
by 
  sorry

end total_fish_l130_130038


namespace subset_condition_for_a_l130_130588

theorem subset_condition_for_a (a : ℝ) : 
  (∀ x y : ℝ, (x - 1)^2 + (y - 2)^2 ≤ 5 / 4 → (|x - 1| + 2 * |y - 2| ≤ a)) → a ≥ 5 / 2 :=
by
  intro H
  sorry

end subset_condition_for_a_l130_130588


namespace value_of_fraction_l130_130190

theorem value_of_fraction (y : ℝ) (h : 4 - 9 / y + 9 / (y^2) = 0) : 3 / y = 2 :=
sorry

end value_of_fraction_l130_130190


namespace members_who_play_both_l130_130338

theorem members_who_play_both (N B T Neither : ℕ) (hN : N = 30) (hB : B = 16) (hT : T = 19) (hNeither : Neither = 2) : 
  B + T - (N - Neither) = 7 :=
by
  sorry

end members_who_play_both_l130_130338


namespace collinear_vectors_parallel_right_angle_triangle_abc_l130_130713

def vec_ab (k : ℝ) : ℝ × ℝ := (2 - k, -1)
def vec_ac (k : ℝ) : ℝ × ℝ := (1, k)

-- Prove that if vectors AB and AC are collinear, then k = 1 ± √2
theorem collinear_vectors_parallel (k : ℝ) :
  (2 - k) * k - 1 = 0 ↔ k = 1 + Real.sqrt 2 ∨ k = 1 - Real.sqrt 2 :=
by
  sorry

def vec_bc (k : ℝ) : ℝ × ℝ := (k - 1, k + 1)

-- Prove that if triangle ABC is right-angled, then k = 1 or k = -1 ± √2
theorem right_angle_triangle_abc (k : ℝ) :
  ( (2 - k) * 1 + (-1) * k = 0 ∨ (k - 1) * 1 + (k + 1) * k = 0 ) ↔ 
  k = 1 ∨ k = -1 + Real.sqrt 2 ∨ k = -1 - Real.sqrt 2 :=
by
  sorry

end collinear_vectors_parallel_right_angle_triangle_abc_l130_130713


namespace infinite_non_expressible_integers_l130_130270

theorem infinite_non_expressible_integers :
  ∃ (S : Set ℤ), S.Infinite ∧ (∀ n ∈ S, ∀ a b c : ℕ, n ≠ 2^a + 3^b - 5^c) :=
sorry

end infinite_non_expressible_integers_l130_130270


namespace triangular_array_sum_of_digits_l130_130277

def triangular_sum (N : ℕ) : ℕ := N * (N + 1) / 2

def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem triangular_array_sum_of_digits :
  ∃ N : ℕ, triangular_sum N = 2080 ∧ sum_of_digits N = 10 :=
by
  sorry

end triangular_array_sum_of_digits_l130_130277


namespace bridge_extension_length_l130_130122

theorem bridge_extension_length (river_width bridge_length : ℕ) (h_river : river_width = 487) (h_bridge : bridge_length = 295) : river_width - bridge_length = 192 :=
by
  sorry

end bridge_extension_length_l130_130122


namespace find_quadrant_372_degrees_l130_130975

theorem find_quadrant_372_degrees : 
  ∃ q : ℕ, q = 1 ↔ (372 % 360 = 12 ∧ (0 ≤ 12 ∧ 12 < 90)) :=
by
  sorry

end find_quadrant_372_degrees_l130_130975


namespace complete_task_in_3_days_l130_130385

theorem complete_task_in_3_days (x y z w v : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0) (hv : v > 0)
  (h1 : 1 / x + 1 / y + 1 / z = 1 / 7.5)
  (h2 : 1 / x + 1 / z + 1 / v = 1 / 5)
  (h3 : 1 / x + 1 / z + 1 / w = 1 / 6)
  (h4 : 1 / y + 1 / w + 1 / v = 1 / 4) :
  1 / (1 / x + 1 / z + 1 / v + 1 / w + 1 / y) = 3 :=
sorry

end complete_task_in_3_days_l130_130385


namespace lengths_AC_CB_ratio_GJ_JH_coords_F_on_DE_values_p_q_KL_l130_130179

-- Problem 1 - Lengths of AC and CB are 15 and 5 respectively.
theorem lengths_AC_CB (x1 y1 x2 y2 x3 y3 : ℝ) :
  (x1, y1) = (1,2) ∧ (x2, y2) = (17,14) ∧ (x3, y3) = (13,11) →
  ∃ (AC CB : ℝ), AC = 15 ∧ CB = 5 :=
by
  sorry

-- Problem 2 - Ratio of GJ and JH is 3:2.
theorem ratio_GJ_JH (x1 y1 x2 y2 x3 y3 : ℝ) :
  (x1, y1) = (11,2) ∧ (x2, y2) = (1,7) ∧ (x3, y3) = (5,5) →
  ∃ (GJ JH : ℝ), GJ / JH = 3 / 2 :=
by
  sorry

-- Problem 3 - Coordinates of point F on DE with ratio 1:2 is (3,7).
theorem coords_F_on_DE (x1 y1 x2 y2 : ℝ) :
  (x1, y1) = (1,6) ∧ (x2, y2) = (7,9) →
  ∃ (x y : ℝ), (x, y) = (3,7) :=
by
  sorry

-- Problem 4 - Values of p and q for point M on KL with ratio 3:4 are p = 15 and q = 2.
theorem values_p_q_KL (x1 y1 x2 y2 x3 y3 : ℝ) :
  (x1, y1) = (1, q) ∧ (x2, y2) = (p, 9) ∧ (x3, y3) = (7,5) →
  ∃ (p q : ℝ), p = 15 ∧ q = 2 :=
by
  sorry

end lengths_AC_CB_ratio_GJ_JH_coords_F_on_DE_values_p_q_KL_l130_130179


namespace maximum_s_squared_l130_130745

-- Definitions based on our conditions
def semicircle_radius : ℝ := 5
def diameter_length : ℝ := 10

-- Statement of the problem (no proof, statement only)
theorem maximum_s_squared (A B C : ℝ×ℝ) (AC BC : ℝ) (h : AC + BC = s) :
    (A.2 = 0) ∧ (B.2 = 0) ∧ (dist A B = diameter_length) ∧
    (dist C (5,0) = semicircle_radius) ∧ (s = AC + BC) →
    s^2 ≤ 200 :=
sorry

end maximum_s_squared_l130_130745


namespace difference_between_numbers_l130_130708

-- Given definitions based on conditions
def sum_of_two_numbers (x y : ℝ) : Prop := x + y = 15
def difference_of_two_numbers (x y : ℝ) : Prop := x - y = 10
def difference_of_squares (x y : ℝ) : Prop := x^2 - y^2 = 150

theorem difference_between_numbers (x y : ℝ) 
  (h1 : sum_of_two_numbers x y) 
  (h2 : difference_of_two_numbers x y) 
  (h3 : difference_of_squares x y) :
  x - y = 10 :=
by
  sorry

end difference_between_numbers_l130_130708


namespace symmetric_circle_eq_l130_130230

theorem symmetric_circle_eq {x y : ℝ} :
  (∃ x y : ℝ, (x+2)^2 + (y-1)^2 = 5) →
  (x - 1)^2 + (y + 2)^2 = 5 :=
sorry

end symmetric_circle_eq_l130_130230


namespace find_number_l130_130809

theorem find_number {x : ℝ} (h : (1/3) * x = 130.00000000000003) : x = 390 := 
sorry

end find_number_l130_130809


namespace michelle_scored_30_l130_130927

-- Define the total team points
def team_points : ℕ := 72

-- Define the number of other players
def num_other_players : ℕ := 7

-- Define the average points scored by the other players
def avg_points_other_players : ℕ := 6

-- Calculate the total points scored by the other players
def total_points_other_players : ℕ := num_other_players * avg_points_other_players

-- Define the points scored by Michelle
def michelle_points : ℕ := team_points - total_points_other_players

-- Prove that the points scored by Michelle is 30
theorem michelle_scored_30 : michelle_points = 30 :=
by
  -- Here would be the proof, but we skip it with sorry.
  sorry

end michelle_scored_30_l130_130927


namespace necessary_but_not_sufficient_l130_130248

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  2 * Real.sin (ω * x - (Real.pi / 3))

theorem necessary_but_not_sufficient (ω : ℝ) :
  (∀ x : ℝ, f ω (x + Real.pi) = f ω x) ↔ (ω = 2) ∨ (∃ ω ≠ 2, ∀ x : ℝ, f ω (x + Real.pi) = f ω x) :=
by
  sorry

end necessary_but_not_sufficient_l130_130248


namespace never_consecutive_again_l130_130165

theorem never_consecutive_again (n : ℕ) (seq : ℕ → ℕ) :
  (∀ k, seq k = seq 0 + k) → 
  ∀ seq' : ℕ → ℕ,
    (∀ i j, i < j → seq' (2*i) = seq i + seq (j) ∧ seq' (2*i+1) = seq i - seq (j)) →
    ¬ (∀ k, seq' k = seq' 0 + k) :=
by
  sorry

end never_consecutive_again_l130_130165


namespace aria_analysis_time_l130_130896

-- Definitions for the number of bones in each section
def skull_bones : ℕ := 29
def spine_bones : ℕ := 33
def thorax_bones : ℕ := 37
def upper_limb_bones : ℕ := 64
def lower_limb_bones : ℕ := 62

-- Definitions for the time spent per bone in each section (in minutes)
def time_per_skull_bone : ℕ := 15
def time_per_spine_bone : ℕ := 10
def time_per_thorax_bone : ℕ := 12
def time_per_upper_limb_bone : ℕ := 8
def time_per_lower_limb_bone : ℕ := 10

-- Definition for the total time needed in minutes
def total_time_in_minutes : ℕ :=
  (skull_bones * time_per_skull_bone) +
  (spine_bones * time_per_spine_bone) +
  (thorax_bones * time_per_thorax_bone) +
  (upper_limb_bones * time_per_upper_limb_bone) +
  (lower_limb_bones * time_per_lower_limb_bone)

-- Definition for the total time needed in hours
def total_time_in_hours : ℚ := total_time_in_minutes / 60

-- Theorem to prove the total time needed in hours is approximately 39.02
theorem aria_analysis_time : abs (total_time_in_hours - 39.02) < 0.01 :=
by
  sorry

end aria_analysis_time_l130_130896


namespace trigonometric_expression_result_l130_130918

variable (α : ℝ)
variable (line_eq : ∀ x y : ℝ, 6 * x - 2 * y - 5 = 0)
variable (tan_alpha : Real.tan α = 3)

theorem trigonometric_expression_result :
  (Real.sin (Real.pi - α) + Real.cos (-α)) / (Real.sin (-α) - Real.cos (Real.pi + α)) = -2 := 
by
  sorry

end trigonometric_expression_result_l130_130918


namespace compute_sum_of_products_of_coefficients_l130_130843

theorem compute_sum_of_products_of_coefficients (b1 b2 b3 b4 c1 c2 c3 c4 : ℝ)
  (h : ∀ x : ℝ, (x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1) =
    (x^2 + b1 * x + c1) * (x^2 + b2 * x + c2) * (x^2 + b3 * x + c3) * (x^2 + b4 * x + c4)) :
  b1 * c1 + b2 * c2 + b3 * c3 + b4 * c4 = -1 :=
by
  -- Proof would go here
  sorry

end compute_sum_of_products_of_coefficients_l130_130843


namespace cylinder_ellipse_eccentricity_l130_130313

noncomputable def eccentricity_of_ellipse (diameter : ℝ) (angle : ℝ) : ℝ :=
  let r := diameter / 2
  let b := r
  let a := r / (Real.cos angle)
  let c := Real.sqrt (a^2 - b^2)
  c / a

theorem cylinder_ellipse_eccentricity :
  eccentricity_of_ellipse 12 (Real.pi / 6) = 1 / 2 :=
by
  sorry

end cylinder_ellipse_eccentricity_l130_130313


namespace fewer_onions_correct_l130_130759

-- Define the quantities
def tomatoes : ℕ := 2073
def corn : ℕ := 4112
def onions : ℕ := 985

-- Calculate the total number of tomatoes and corn
def tomatoes_and_corn : ℕ := tomatoes + corn

-- Calculate the number of fewer onions
def fewer_onions : ℕ := tomatoes_and_corn - onions

-- State the theorem and provide the proof
theorem fewer_onions_correct : fewer_onions = 5200 :=
by
  -- The statement is proved directly by the calculations above
  -- Providing the actual proof is not necessary as per the guidelines
  sorry

end fewer_onions_correct_l130_130759


namespace john_taking_pictures_years_l130_130991

-- Definitions based on the conditions
def pictures_per_day : ℕ := 10
def images_per_card : ℕ := 50
def cost_per_card : ℕ := 60
def total_spent : ℕ := 13140
def days_per_year : ℕ := 365

-- Theorem statement
theorem john_taking_pictures_years : total_spent / cost_per_card * images_per_card / pictures_per_day / days_per_year = 3 :=
by
  sorry

end john_taking_pictures_years_l130_130991


namespace exists_coprime_positive_sum_le_m_l130_130428

theorem exists_coprime_positive_sum_le_m (m : ℕ) (a b : ℤ) 
  (ha : 0 < a) (hb : 0 < b) (hcoprime : Int.gcd a b = 1)
  (h1 : a ∣ (m + b^2)) (h2 : b ∣ (m + a^2)) 
  : ∃ a' b', 0 < a' ∧ 0 < b' ∧ Int.gcd a' b' = 1 ∧ a' ∣ (m + b'^2) ∧ b' ∣ (m + a'^2) ∧ a' + b' ≤ m + 1 :=
by
  sorry

end exists_coprime_positive_sum_le_m_l130_130428


namespace distance_between_centers_same_side_distance_between_centers_opposite_side_l130_130748

open Real

noncomputable def distance_centers_same_side (r : ℝ) : ℝ := (r * (sqrt 6 + sqrt 2)) / 2

noncomputable def distance_centers_opposite_side (r : ℝ) : ℝ := (r * (sqrt 6 - sqrt 2)) / 2

theorem distance_between_centers_same_side (r : ℝ):
  ∃ dist, dist = distance_centers_same_side r :=
sorry

theorem distance_between_centers_opposite_side (r : ℝ):
  ∃ dist, dist = distance_centers_opposite_side r :=
sorry

end distance_between_centers_same_side_distance_between_centers_opposite_side_l130_130748


namespace find_x_l130_130411

theorem find_x (x : ℝ) (h : (0.4 + x) / 2 = 0.2025) : x = 0.005 :=
by
  sorry

end find_x_l130_130411


namespace units_digit_2008_pow_2008_l130_130623

theorem units_digit_2008_pow_2008 : (2008 ^ 2008) % 10 = 6 := 
by
  -- The units digits of powers of 8 repeat in a cycle: 8, 4, 2, 6
  -- 2008 mod 4 = 0 which implies it falls on the 4th position in the pattern cycle
  sorry

end units_digit_2008_pow_2008_l130_130623


namespace opposite_of_8_is_neg_8_l130_130590

theorem opposite_of_8_is_neg_8 : - (8 : ℤ) = -8 :=
by
  sorry

end opposite_of_8_is_neg_8_l130_130590


namespace solution_l130_130771

theorem solution (x y : ℝ) (h1 : x * y = 6) (h2 : x^2 * y + x * y^2 + x + y = 63) : x^2 + y^2 = 69 := 
by 
  -- Insert proof here
  sorry

end solution_l130_130771


namespace car_gas_tank_capacity_l130_130462

theorem car_gas_tank_capacity
  (initial_mileage : ℕ)
  (final_mileage : ℕ)
  (miles_per_gallon : ℕ)
  (tank_fills : ℕ)
  (usage : initial_mileage = 1728)
  (usage_final : final_mileage = 2928)
  (car_efficiency : miles_per_gallon = 30)
  (fills : tank_fills = 2):
  (final_mileage - initial_mileage) / miles_per_gallon / tank_fills = 20 :=
by
  sorry

end car_gas_tank_capacity_l130_130462


namespace find_radius_l130_130164

-- Definitions based on conditions
def circle_radius (r : ℝ) : Prop := r = 2

-- Specification based on the question and conditions
theorem find_radius (r : ℝ) : circle_radius r :=
by
  -- Skip the proof
  sorry

end find_radius_l130_130164


namespace least_xy_l130_130326

theorem least_xy (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 1 / (3 * y) = 1 / 8) : x * y = 96 :=
sorry

end least_xy_l130_130326


namespace shooter_prob_l130_130764

variable (hit_prob : ℝ)
variable (miss_prob : ℝ := 1 - hit_prob)
variable (p1 : hit_prob = 0.85)
variable (independent_shots : true)

theorem shooter_prob :
  miss_prob * miss_prob * hit_prob = 0.019125 :=
by
  rw [p1]
  sorry

end shooter_prob_l130_130764


namespace pencils_in_drawer_l130_130260

/-- 
If there were originally 2 pencils in the drawer and there are now 5 pencils in total, 
then Tim must have placed 3 pencils in the drawer.
-/
theorem pencils_in_drawer (original_pencils tim_pencils total_pencils : ℕ) 
  (h1 : original_pencils = 2) 
  (h2 : total_pencils = 5) 
  (h3 : total_pencils = original_pencils + tim_pencils) : 
  tim_pencils = 3 := 
by
  rw [h1, h2] at h3
  linarith

end pencils_in_drawer_l130_130260


namespace solution_count_l130_130066

/-- There are 91 solutions to the equation x + y + z = 15 given that x, y, z are all positive integers. -/
theorem solution_count (x y z : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 15) : 
  ∃! n, n = 91 := 
by sorry

end solution_count_l130_130066


namespace polynomial_divisibility_l130_130611

-- Definitions
def f (k l m n : ℕ) (x : ℂ) : ℂ :=
  x^(4 * k) + x^(4 * l + 1) + x^(4 * m + 2) + x^(4 * n + 3)

def g (x : ℂ) : ℂ :=
  x^3 + x^2 + x + 1

-- Theorem statement
theorem polynomial_divisibility (k l m n : ℕ) : ∀ x : ℂ, g x ∣ f k l m n x :=
  sorry

end polynomial_divisibility_l130_130611


namespace average_of_D_E_F_l130_130859

theorem average_of_D_E_F (D E F : ℝ) 
  (h1 : 2003 * F - 4006 * D = 8012) 
  (h2 : 2003 * E + 6009 * D = 10010) : 
  (D + E + F) / 3 = 3 := 
by 
  sorry

end average_of_D_E_F_l130_130859


namespace simplify_expression_l130_130382

theorem simplify_expression (x y : ℝ) :
  3 * (x + y) ^ 2 - 7 * (x + y) + 8 * (x + y) ^ 2 + 6 * (x + y) = 
  11 * (x + y) ^ 2 - (x + y) :=
by
  sorry

end simplify_expression_l130_130382


namespace monthly_installments_l130_130408

theorem monthly_installments (cash_price deposit installment saving : ℕ) (total_paid installments_made : ℕ) :
  cash_price = 8000 →
  deposit = 3000 →
  installment = 300 →
  saving = 4000 →
  total_paid = cash_price + saving →
  installments_made = (total_paid - deposit) / installment →
  installments_made = 30 :=
by
  intros h_cash_price h_deposit h_installment h_saving h_total_paid h_installments_made
  sorry

end monthly_installments_l130_130408


namespace matrix_expression_solution_l130_130063

theorem matrix_expression_solution (x : ℝ) :
  let a := 3 * x + 1
  let b := x + 1
  let c := 2
  let d := 2 * x
  ab - cd = 5 :=
by
  sorry

end matrix_expression_solution_l130_130063


namespace find_n_range_l130_130589

theorem find_n_range (m n : ℝ) 
  (h_m : -Real.sqrt 3 ≤ m ∧ m ≤ Real.sqrt 3) :
  (∀ x y z : ℝ, 0 ≤ x^2 + 2 * y^2 + 3 * z^2 + 2 * x * y + 2 * m * z * x + 2 * n * y * z) ↔ 
  (m - Real.sqrt (3 - m^2) ≤ n ∧ n ≤ m + Real.sqrt (3 - m^2)) :=
by
  sorry

end find_n_range_l130_130589


namespace trig_identity_and_perimeter_l130_130680

theorem trig_identity_and_perimeter
  (a b c : ℝ) (A B C : ℝ)
  (h1 : Real.sin C * Real.sin (A - B) = Real.sin B * Real.sin (C - A))
  (a_val : a = 5)
  (cos_A : Real.cos A = 25 / 31) :
  (2 * a^2 = b^2 + c^2) ∧ (a + b + c = 14) :=
by
  sorry

end trig_identity_and_perimeter_l130_130680


namespace simplify_expression_l130_130962

theorem simplify_expression (r : ℝ) : 100*r - 48*r + 10 = 52*r + 10 :=
by
  sorry

end simplify_expression_l130_130962


namespace pair1_equivalent_pair2_non_equivalent_pair3_equivalent_pair4_equivalent_pair5_non_equivalent_pair6_equivalent_l130_130300

theorem pair1_equivalent (x : ℝ) : (x^2 + 5 * x < 4) ↔ (x^2 + 5 * x + 3 * x < 4 + 3 * x) :=
sorry

theorem pair2_non_equivalent (x : ℝ) (hx : x ≠ 0) : (x^2 + 5 * x < 4) ↔ (x^2 + 5 * x + 1 / x < 4 + 1 / x) :=
sorry

theorem pair3_equivalent (x : ℝ) (hx : x ≥ 3) : (x ≥ 3) ↔ (x * (x + 5)^2 ≥ 3 * (x + 5)^2) :=
sorry

theorem pair4_equivalent (x : ℝ) (hx : x ≥ 3) : (x ≥ 3) ↔ (x * (x - 5)^2 ≥ 3 * (x - 5)^2) :=
sorry

theorem pair5_non_equivalent (x : ℝ) (hx : x ≠ -1) : (x + 3 > 0) ↔ ( (x + 3) * (x + 1) / (x + 1) > 0) :=
sorry

theorem pair6_equivalent (x : ℝ) (hx : x ≠ -2) : (x - 3 > 0) ↔ ( (x + 2) * (x - 3) / (x + 2) > 0) :=
sorry

end pair1_equivalent_pair2_non_equivalent_pair3_equivalent_pair4_equivalent_pair5_non_equivalent_pair6_equivalent_l130_130300


namespace complex_expression_evaluation_l130_130075

theorem complex_expression_evaluation (i : ℂ) (h : i^2 = -1) : i^3 * (1 - i)^2 = -2 :=
by
  -- Placeholder for the actual proof which is skipped here
  sorry

end complex_expression_evaluation_l130_130075


namespace negation_of_exists_prop_l130_130317

variable (n : ℕ)

theorem negation_of_exists_prop :
  (¬ (∃ n : ℕ, n^2 > 2^n)) ↔ (∀ n : ℕ, n^2 ≤ 2^n) :=
by
  sorry

end negation_of_exists_prop_l130_130317


namespace raffle_tickets_sold_l130_130283

theorem raffle_tickets_sold (total_amount : ℕ) (ticket_cost : ℕ) (tickets_sold : ℕ) 
    (h1 : total_amount = 620) (h2 : ticket_cost = 4) : tickets_sold = 155 :=
by {
  sorry
}

end raffle_tickets_sold_l130_130283


namespace cos_75_degree_identity_l130_130938

theorem cos_75_degree_identity :
  Real.cos (75 * Real.pi / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 :=
by
  sorry

end cos_75_degree_identity_l130_130938


namespace calculate_molar_mass_l130_130726

-- Definitions from the conditions
def number_of_moles : ℝ := 8
def weight_in_grams : ℝ := 1600

-- Goal: Prove that the molar mass is 200 grams/mole
theorem calculate_molar_mass : (weight_in_grams / number_of_moles) = 200 :=
by
  sorry

end calculate_molar_mass_l130_130726


namespace part1_part2_l130_130584

-- Problem part (1)
theorem part1 : (Real.sqrt 12 + Real.sqrt (4 / 3)) * Real.sqrt 3 = 8 := 
  sorry

-- Problem part (2)
theorem part2 : Real.sqrt 48 - Real.sqrt 54 / Real.sqrt 2 + (3 - Real.sqrt 3) * (3 + Real.sqrt 3) = Real.sqrt 3 + 6 := 
  sorry

end part1_part2_l130_130584


namespace solve_expression_l130_130450

theorem solve_expression (a b c : ℝ) (ha : a^3 - 2020*a^2 + 1010 = 0) (hb : b^3 - 2020*b^2 + 1010 = 0) (hc : c^3 - 2020*c^2 + 1010 = 0) (habc_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
    (1 / (a * b) + 1 / (b * c) + 1 / (a * c) = -2) := 
sorry

end solve_expression_l130_130450


namespace magnitude_of_z_l130_130945

theorem magnitude_of_z (z : ℂ) (h : z * (2 - Complex.I) = Complex.I) : Complex.abs z = 1 / Real.sqrt 5 := by
  sorry

end magnitude_of_z_l130_130945


namespace original_proposition_true_converse_proposition_false_l130_130851

theorem original_proposition_true (a b : ℝ) : 
  a + b ≥ 2 → (a ≥ 1 ∨ b ≥ 1) := 
sorry

theorem converse_proposition_false : 
  ¬ (∀ a b : ℝ, (a ≥ 1 ∨ b ≥ 1) → a + b ≥ 2) :=
sorry

end original_proposition_true_converse_proposition_false_l130_130851


namespace fractions_sum_l130_130460

theorem fractions_sum (a : ℝ) (h : a ≠ 0) : (1 / a) + (2 / a) = 3 / a := 
by 
  sorry

end fractions_sum_l130_130460


namespace train_length_correct_l130_130924

noncomputable def train_length (time : ℝ) (platform_length : ℝ) (speed_kmph : ℝ) : ℝ :=
  let speed_mps := speed_kmph * 1000 / 3600
  let total_distance := speed_mps * time
  total_distance - platform_length

theorem train_length_correct :
  train_length 17.998560115190784 200 90 = 249.9640028797696 :=
by
  sorry

end train_length_correct_l130_130924


namespace bill_property_taxes_l130_130606

theorem bill_property_taxes 
  (take_home_salary sales_taxes gross_salary : ℕ)
  (income_tax_rate : ℚ)
  (take_home_salary_eq : take_home_salary = 40000)
  (sales_taxes_eq : sales_taxes = 3000)
  (gross_salary_eq : gross_salary = 50000)
  (income_tax_rate_eq : income_tax_rate = 0.1) :
  let income_taxes := (income_tax_rate * gross_salary) 
  let property_taxes := gross_salary - (income_taxes + sales_taxes + take_home_salary)
  property_taxes = 2000 := by
  sorry

end bill_property_taxes_l130_130606


namespace integer_solutions_to_cube_sum_eq_2_pow_30_l130_130275

theorem integer_solutions_to_cube_sum_eq_2_pow_30 (x y : ℤ) :
  x^3 + y^3 = 2^30 → (x = 0 ∧ y = 2^10) ∨ (x = 2^10 ∧ y = 0) :=
by
  sorry

end integer_solutions_to_cube_sum_eq_2_pow_30_l130_130275


namespace find_new_bottle_caps_l130_130550

theorem find_new_bottle_caps (initial caps_thrown current : ℕ) (h_initial : initial = 69)
  (h_thrown : caps_thrown = 60) (h_current : current = 67) :
  ∃ n, initial - caps_thrown + n = current ∧ n = 58 := by
sorry

end find_new_bottle_caps_l130_130550


namespace alicia_art_left_l130_130425

-- Definition of the problem conditions.
def initial_pieces : ℕ := 70
def donated_pieces : ℕ := 46

-- The theorem to prove the number of art pieces left is 24.
theorem alicia_art_left : initial_pieces - donated_pieces = 24 := 
by
  sorry

end alicia_art_left_l130_130425


namespace simon_age_is_10_l130_130931

def alvin_age : ℕ := 30

def simon_age (alvin_age : ℕ) : ℕ :=
  (alvin_age / 2) - 5

theorem simon_age_is_10 : simon_age alvin_age = 10 := by
  sorry

end simon_age_is_10_l130_130931


namespace inequalities_no_solution_l130_130633

theorem inequalities_no_solution (x n : ℝ) (h1 : x ≤ 1) (h2 : x ≥ n) : n > 1 :=
sorry

end inequalities_no_solution_l130_130633


namespace distance_A_B_l130_130399

noncomputable def distance_between_points (v_A v_B : ℝ) (t : ℝ) : ℝ := 5 * (6 * t / (2 / 3 * t))

theorem distance_A_B
  (v_A v_B : ℝ)
  (t : ℝ)
  (h1 : v_A = 1.2 * v_B)
  (h2 : ∃ distance_broken, distance_broken = 5)
  (h3 : ∃ delay, delay = (1 / 6) * 6 * t)
  (h4 : ∃ v_B_new, v_B_new = 1.6 * v_B)
  (h5 : distance_between_points v_A v_B t = 45) :
  distance_between_points v_A v_B t = 45 :=
sorry

end distance_A_B_l130_130399


namespace estimate_total_number_of_fish_l130_130386

-- Define the conditions
variables (totalMarked : ℕ) (secondSample : ℕ) (markedInSecondSample : ℕ) (N : ℕ)

-- Assume the conditions
axiom condition1 : totalMarked = 60
axiom condition2 : secondSample = 80
axiom condition3 : markedInSecondSample = 5

-- Lean theorem statement proving N = 960 given the conditions
theorem estimate_total_number_of_fish (totalMarked secondSample markedInSecondSample N : ℕ)
  (h1 : totalMarked = 60)
  (h2 : secondSample = 80)
  (h3 : markedInSecondSample = 5) :
  N = 960 :=
sorry

end estimate_total_number_of_fish_l130_130386


namespace minimum_value_nine_l130_130500

noncomputable def min_value (a b c k : ℝ) : ℝ :=
  (a + b + k) / c + (a + c + k) / b + (b + c + k) / a

theorem minimum_value_nine (a b c k : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hk : 0 < k) :
  min_value a b c k ≥ 9 :=
sorry

end minimum_value_nine_l130_130500


namespace kevin_exchanges_l130_130118

variables (x y : ℕ)

def R (x y : ℕ) := 100 - 3 * x + 2 * y
def B (x y : ℕ) := 100 + 2 * x - 4 * y

theorem kevin_exchanges :
  (∃ x y, R x y >= 3 ∧ B x y >= 4 ∧ x + y = 132) :=
sorry

end kevin_exchanges_l130_130118


namespace area_enclosed_by_circle_l130_130297

theorem area_enclosed_by_circle : 
  (∀ x y : ℝ, x^2 + y^2 + 10 * x + 24 * y = 0) → 
  (π * 13^2 = 169 * π):=
by
  intro h
  sorry

end area_enclosed_by_circle_l130_130297


namespace production_increase_l130_130307

theorem production_increase (h1 : ℝ) (h2 : ℝ) (h3 : h1 = 0.75) (h4 : h2 = 0.5) :
  (h1 + h2 - 1) = 0.25 := by
  sorry

end production_increase_l130_130307


namespace original_price_l130_130034

variable (a : ℝ)

theorem original_price (h : 0.6 * x = a) : x = (5 / 3) * a :=
sorry

end original_price_l130_130034


namespace find_p_l130_130374

theorem find_p (m n p : ℚ) 
  (h1 : m = n / 6 - 2 / 5)
  (h2 : m + p = (n + 18) / 6 - 2 / 5) : 
  p = 3 := 
by 
  sorry

end find_p_l130_130374


namespace calc_problem_l130_130530

def odot (a b : ℕ) : ℕ := a * b - (a + b)

theorem calc_problem : odot 6 (odot 5 4) = 49 :=
by
  sorry

end calc_problem_l130_130530


namespace percentage_increase_second_year_l130_130522

theorem percentage_increase_second_year 
  (initial_population : ℝ)
  (first_year_increase : ℝ) 
  (population_after_2_years : ℝ) 
  (final_population : ℝ)
  (H_initial_population : initial_population = 800)
  (H_first_year_increase : first_year_increase = 0.22)
  (H_population_after_2_years : final_population = 1220) :
  ∃ P : ℝ, P = 25 := 
by
  -- Define the population after the first year
  let population_after_first_year := initial_population * (1 + first_year_increase)
  -- Define the equation relating populations and solve for P
  let second_year_increase := (final_population / population_after_first_year - 1) * 100
  -- Show P equals 25
  use second_year_increase
  sorry

end percentage_increase_second_year_l130_130522


namespace max_a_correct_answers_l130_130600

theorem max_a_correct_answers : 
  ∃ (a b c x y z w : ℕ), 
  a + b + c + x + y + z + w = 39 ∧
  a = b + c ∧
  (a + x + y + w) = a + 5 + (x + y + w) ∧
  b + z = 2 * (c + z) ∧
  23 ≤ a :=
sorry

end max_a_correct_answers_l130_130600


namespace simplify_complex_l130_130459

open Complex

theorem simplify_complex : (5 : ℂ) / (I - 2) = -2 - I := by
  sorry

end simplify_complex_l130_130459


namespace not_unique_equilateral_by_one_angle_and_opposite_side_l130_130329

-- Definitions related to triangles
structure Triangle :=
  (a b c : ℝ) -- sides
  (alpha beta gamma : ℝ) -- angles

-- Definition of triangle types
def isIsosceles (t : Triangle) : Prop :=
  (t.a = t.b ∨ t.b = t.c ∨ t.a = t.c)

def isRight (t : Triangle) : Prop :=
  (t.alpha = 90 ∨ t.beta = 90 ∨ t.gamma = 90)

def isEquilateral (t : Triangle) : Prop :=
  (t.a = t.b ∧ t.b = t.c ∧ t.alpha = 60 ∧ t.beta = 60 ∧ t.gamma = 60)

def isScalene (t : Triangle) : Prop :=
  (t.a ≠ t.b ∧ t.b ≠ t.c ∧ t.a ≠ t.c)

-- Proof that having one angle and the side opposite it does not determine an equilateral triangle.
theorem not_unique_equilateral_by_one_angle_and_opposite_side :
  ¬ ∀ (t1 t2 : Triangle), (isEquilateral t1 ∧ isEquilateral t2 →
    t1.alpha = t2.alpha ∧ t1.a = t2.a →
    t1 = t2) := sorry

end not_unique_equilateral_by_one_angle_and_opposite_side_l130_130329


namespace biology_exam_students_l130_130181

theorem biology_exam_students :
  let students := 200
  let score_A := (1 / 4) * students
  let remaining_students := students - score_A
  let score_B := (1 / 5) * remaining_students
  let score_C := (1 / 3) * remaining_students
  let score_D := (5 / 12) * remaining_students
  let score_F := students - (score_A + score_B + score_C + score_D)
  let re_assessed_C := (3 / 5) * score_C
  let final_score_B := score_B + re_assessed_C
  let final_score_C := score_C - re_assessed_C
  score_A = 50 ∧ 
  final_score_B = 60 ∧ 
  final_score_C = 20 ∧ 
  score_D = 62 ∧ 
  score_F = 8 :=
by {
  sorry
}

end biology_exam_students_l130_130181


namespace fg_value_l130_130969

def g (x : ℕ) : ℕ := 4 * x + 10
def f (x : ℕ) : ℕ := 6 * x - 12

theorem fg_value : f (g 10) = 288 := by
  sorry

end fg_value_l130_130969


namespace total_pencil_length_l130_130684

-- Definitions from the conditions
def purple_length : ℕ := 3
def black_length : ℕ := 2
def blue_length : ℕ := 1

-- Proof statement
theorem total_pencil_length :
  purple_length + black_length + blue_length = 6 :=
by
  sorry

end total_pencil_length_l130_130684


namespace martin_waste_time_l130_130153

theorem martin_waste_time : 
  let waiting_traffic := 2
  let trying_off_freeway := 4 * waiting_traffic
  let detours := 3 * 30 / 60
  let meal := 45 / 60
  let delays := (20 + 40) / 60
  waiting_traffic + trying_off_freeway + detours + meal + delays = 13.25 := 
by
  sorry

end martin_waste_time_l130_130153


namespace find_subtracted_number_l130_130775

-- Given conditions
def t : ℕ := 50
def k : ℕ := 122
def eq_condition (n : ℤ) : Prop := t = (5 / 9 : ℚ) * (k - n)

-- The proof problem proving the number subtracted from k is 32
theorem find_subtracted_number : eq_condition 32 :=
by
  -- implementation here will demonstrate that t = 50 implies the number is 32
  sorry

end find_subtracted_number_l130_130775


namespace angle_A_is_correct_l130_130130

-- Define the given conditions and the main theorem.
theorem angle_A_is_correct (A : ℝ) (m n : ℝ × ℝ) 
  (h_m : m = (Real.sin (A / 2), Real.cos (A / 2)))
  (h_n : n = (Real.cos (A / 2), -Real.cos (A / 2)))
  (h_eq : 2 * ((Prod.fst m * Prod.fst n) + (Prod.snd m * Prod.snd n)) + (Real.sqrt ((Prod.fst m)^2 + (Prod.snd m)^2)) = Real.sqrt 2 / 2) 
  : A = 5 * Real.pi / 12 := by
  sorry

end angle_A_is_correct_l130_130130


namespace bombardiers_shots_l130_130644

theorem bombardiers_shots (x y z : ℕ) :
  x + y = z + 26 →
  x + y + 38 = y + z →
  x + z = y + 24 →
  x = 25 ∧ y = 64 ∧ z = 63 := by
  sorry

end bombardiers_shots_l130_130644


namespace max_value_expression_l130_130327

theorem max_value_expression (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : a + b + c + d ≤ 4) :
  (Real.sqrt (Real.sqrt (a^2 * (a + b))) +
   Real.sqrt (Real.sqrt (b^2 * (b + c))) +
   Real.sqrt (Real.sqrt (c^2 * (c + d))) +
   Real.sqrt (Real.sqrt (d^2 * (d + a)))) ≤ 4 * Real.sqrt (Real.sqrt 2) := by
  sorry

end max_value_expression_l130_130327


namespace total_prep_time_l130_130352

-- Definitions:
def jack_time_to_put_shoes_on : ℕ := 4
def additional_time_per_toddler : ℕ := 3
def number_of_toddlers : ℕ := 2

-- Total time calculation
def total_time : ℕ :=
  let time_per_toddler := jack_time_to_put_shoes_on + additional_time_per_toddler
  let total_toddler_time := time_per_toddler * number_of_toddlers
  total_toddler_time + jack_time_to_put_shoes_on

-- Theorem:
theorem total_prep_time :
  total_time = 18 :=
sorry

end total_prep_time_l130_130352


namespace minimum_force_to_submerge_cube_l130_130339

-- Definitions and given conditions
def volume_cube : ℝ := 10e-6 -- 10 cm^3 in m^3
def density_cube : ℝ := 700 -- in kg/m^3
def density_water : ℝ := 1000 -- in kg/m^3
def gravity : ℝ := 10 -- in m/s^2

-- Prove the minimum force required to submerge the cube completely
theorem minimum_force_to_submerge_cube : 
  (density_water * volume_cube * gravity - density_cube * volume_cube * gravity) = 0.03 :=
by
  sorry

end minimum_force_to_submerge_cube_l130_130339


namespace find_c_l130_130350

-- Define the polynomial f(x)
def f (c : ℚ) (x : ℚ) : ℚ := 2 * c * x^3 + 14 * x^2 - 6 * c * x + 25

-- State the problem in Lean 4
theorem find_c (c : ℚ) : (∀ x : ℚ, f c x = 0 ↔ x = (-5)) → c = 75 / 44 := 
by sorry

end find_c_l130_130350


namespace find_b_age_l130_130296

theorem find_b_age (a b : ℕ) (h1 : a + 10 = 2 * (b - 10)) (h2 : a = b + 9) : b = 39 :=
sorry

end find_b_age_l130_130296


namespace quadratic_inequality_l130_130095

noncomputable def quadratic_inequality_solution : Set ℝ :=
  {x | x < 2} ∪ {x | x > 4}

theorem quadratic_inequality (x : ℝ) : (x^2 - 6 * x + 8 > 0) ↔ (x ∈ quadratic_inequality_solution) :=
by
  sorry

end quadratic_inequality_l130_130095


namespace polynomial_expansion_l130_130715

-- Define the polynomial expressions
def poly1 (s : ℝ) : ℝ := 3 * s^3 - 4 * s^2 + 5 * s - 2
def poly2 (s : ℝ) : ℝ := 2 * s^2 - 3 * s + 4

-- Define the expanded form of the product of the two polynomials
def expanded_poly (s : ℝ) : ℝ :=
  6 * s^5 - 17 * s^4 + 34 * s^3 - 35 * s^2 + 26 * s - 8

-- The theorem to prove the equivalence
theorem polynomial_expansion (s : ℝ) :
  (poly1 s) * (poly2 s) = expanded_poly s :=
sorry -- proof goes here

end polynomial_expansion_l130_130715


namespace chords_from_nine_points_l130_130565

theorem chords_from_nine_points : 
  ∀ (n r : ℕ), n = 9 → r = 2 → (Nat.choose n r) = 36 :=
by
  intros n r hn hr
  rw [hn, hr]
  -- Goal: Nat.choose 9 2 = 36
  sorry

end chords_from_nine_points_l130_130565


namespace years_ago_twice_age_l130_130674

variables (H J x : ℕ)

def henry_age : ℕ := 20
def jill_age : ℕ := 13

axiom age_sum : H + J = 33
axiom age_difference : H - x = 2 * (J - x)

theorem years_ago_twice_age (H := henry_age) (J := jill_age) : x = 6 :=
by sorry

end years_ago_twice_age_l130_130674


namespace students_accommodated_l130_130009

theorem students_accommodated 
  (total_students : ℕ)
  (total_workstations : ℕ)
  (workstations_accommodating_x_students : ℕ)
  (x : ℕ)
  (workstations_accommodating_3_students : ℕ)
  (workstation_capacity_10 : ℕ)
  (workstation_capacity_6 : ℕ) :
  total_students = 38 → 
  total_workstations = 16 → 
  workstations_accommodating_x_students = 10 → 
  workstations_accommodating_3_students = 6 → 
  workstation_capacity_10 = 10 * x → 
  workstation_capacity_6 = 6 * 3 → 
  10 * x + 18 = 38 → 
  10 * 2 = 20 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end students_accommodated_l130_130009


namespace population_difference_is_16_l130_130481

def total_birds : ℕ := 250

def pigeons_percent : ℕ := 30
def sparrows_percent : ℕ := 25
def crows_percent : ℕ := 20
def swans_percent : ℕ := 15
def parrots_percent : ℕ := 10

def black_pigeons_percent : ℕ := 60
def white_pigeons_percent : ℕ := 40
def black_male_pigeons_percent : ℕ := 20
def white_female_pigeons_percent : ℕ := 50

def female_sparrows_percent : ℕ := 60
def male_sparrows_percent : ℕ := 40

def female_crows_percent : ℕ := 30
def male_crows_percent : ℕ := 70

def male_parrots_percent : ℕ := 65
def female_parrots_percent : ℕ := 35

noncomputable
def black_male_pigeons : ℕ := (pigeons_percent * total_birds / 100) * (black_pigeons_percent * (black_male_pigeons_percent / 100)) / 100
noncomputable
def white_female_pigeons : ℕ := (pigeons_percent * total_birds / 100) * (white_pigeons_percent * (white_female_pigeons_percent / 100)) / 100
noncomputable
def male_sparrows : ℕ := (sparrows_percent * total_birds / 100) * (male_sparrows_percent / 100)
noncomputable
def female_crows : ℕ := (crows_percent * total_birds / 100) * (female_crows_percent / 100)
noncomputable
def male_parrots : ℕ := (parrots_percent * total_birds / 100) * (male_parrots_percent / 100)

noncomputable
def max_population : ℕ := max (max (max (max black_male_pigeons white_female_pigeons) male_sparrows) female_crows) male_parrots
noncomputable
def min_population : ℕ := min (min (min (min black_male_pigeons white_female_pigeons) male_sparrows) female_crows) male_parrots

noncomputable
def population_difference : ℕ := max_population - min_population

theorem population_difference_is_16 : population_difference = 16 :=
sorry

end population_difference_is_16_l130_130481


namespace range_of_a_l130_130845

noncomputable def p (x: ℝ) : Prop := |4 * x - 1| ≤ 1
noncomputable def q (x a: ℝ) : Prop := x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0

theorem range_of_a (a: ℝ) :
  (¬ (∀ x, p x) → (¬ (∀ x, q x a))) ∧ (¬ (¬ (∀ x, p x) → (¬ (∀ x, q x a))))
  ↔ (-1 / 2 ≤ a ∧ a ≤ 0) :=
sorry

end range_of_a_l130_130845


namespace jack_evening_emails_l130_130322

theorem jack_evening_emails
  (emails_afternoon : ℕ := 3)
  (emails_morning : ℕ := 6)
  (emails_total : ℕ := 10) :
  emails_total - emails_afternoon - emails_morning = 1 :=
by
  sorry

end jack_evening_emails_l130_130322


namespace binomial_inequality_l130_130194

theorem binomial_inequality (n : ℤ) (x : ℝ) (hn : n ≥ 2) (hx : |x| < 1) : 
  2^n > (1 - x)^n + (1 + x)^n := 
sorry

end binomial_inequality_l130_130194


namespace find_original_number_l130_130349

variable (x : ℕ)

theorem find_original_number (h : 3 * (2 * x + 9) = 69) : x = 7 :=
by
  sorry

end find_original_number_l130_130349


namespace solve_trig_eq_l130_130081

open Real

theorem solve_trig_eq (n : ℤ) (x : ℝ) : 
  (sin x) ^ 4 + (cos x) ^ 4 = (sin (2 * x)) ^ 4 + (cos (2 * x)) ^ 4 ↔ x = (n : ℝ) * π / 6 :=
by
  sorry

end solve_trig_eq_l130_130081


namespace polynomial_power_degree_l130_130597

noncomputable def polynomial_degree (p : Polynomial ℝ) : ℕ := p.natDegree

theorem polynomial_power_degree : 
  polynomial_degree ((5 * X^3 - 4 * X + 7)^10) = 30 := by
  sorry

end polynomial_power_degree_l130_130597


namespace garden_furniture_costs_l130_130023

theorem garden_furniture_costs (B T U : ℝ)
    (h1 : T + B + U = 765)
    (h2 : T = 2 * B)
    (h3 : U = 3 * B) :
    B = 127.5 ∧ T = 255 ∧ U = 382.5 :=
by
  sorry

end garden_furniture_costs_l130_130023


namespace find_f_expression_l130_130542

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_expression (x : ℝ) (h₀ : x ≠ 0) (h₁ : x ≠ 1) : 
  f (x) = (1 / (x - 1)) :=
by sorry

example (x : ℝ) (h₀ : x ≠ 0) (h₁ : x ≠ 1) (hx: f (1 / x) = x / (1 - x)) :
  f x = 1 / (x - 1) :=
find_f_expression x h₀ h₁

end find_f_expression_l130_130542


namespace train_length_calculation_l130_130568

theorem train_length_calculation (len1 : ℝ) (speed1_kmph : ℝ) (speed2_kmph : ℝ) (crossing_time : ℝ) (len2 : ℝ) :
  len1 = 120.00001 → 
  speed1_kmph = 120 → 
  speed2_kmph = 80 → 
  crossing_time = 9 → 
  (len1 + len2) = ((speed1_kmph * 1000 / 3600 + speed2_kmph * 1000 / 3600) * crossing_time) → 
  len2 = 379.99949 :=
by
  intros hlen1 hspeed1 hspeed2 htime hdistance
  sorry

end train_length_calculation_l130_130568


namespace sum_first_n_terms_l130_130310

variable (a : ℕ → ℕ)

axiom a1_condition : a 1 = 2
axiom diff_condition : ∀ n : ℕ, a (n + 1) - a n = 2^n

-- Define the sum of the first n terms of the sequence
noncomputable def S : ℕ → ℕ
| 0 => 0
| (n + 1) => S n + a (n + 1)

theorem sum_first_n_terms (n : ℕ) : S a n = 2^(n + 1) - 2 :=
by
  sorry

end sum_first_n_terms_l130_130310


namespace knight_probability_sum_l130_130110

def num_knights := 30
def chosen_knights := 4

-- Calculate valid placements where no knights are adjacent
def valid_placements : ℕ := 26 * 24 * 22 * 20
-- Calculate total unrestricted placements
def total_placements : ℕ := 26 * 27 * 28 * 29
-- Calculate probability
def P : ℚ := 1 - (valid_placements : ℚ) / total_placements

-- Simplify the fraction P to its lowest terms: 553/1079
def simplified_num := 553
def simplified_denom := 1079

-- Sum of the numerator and denominator of simplified P
def sum_numer_denom := simplified_num + simplified_denom

theorem knight_probability_sum :
  sum_numer_denom = 1632 :=
by
  -- Proof is omitted
  sorry

end knight_probability_sum_l130_130110


namespace probability_of_at_least_one_solving_l130_130676

variable (P1 P2 : ℝ)

theorem probability_of_at_least_one_solving : 
  (1 - (1 - P1) * (1 - P2)) = P1 + P2 - P1 * P2 := 
sorry

end probability_of_at_least_one_solving_l130_130676


namespace transformed_curve_eq_l130_130949

-- Define the original ellipse curve
def ellipse (x y : ℝ) : Prop := (x^2) / 4 + y^2 = 1

-- Define the transformation
def transform (x y x' y' : ℝ) : Prop := x' = x ∧ y' = 2 * y

-- Prove the transformed curve satisfies x'^2 + y'^2 = 4
theorem transformed_curve_eq :
  ∀ (x y x' y' : ℝ), ellipse x y → transform x y x' y' → (x'^2 + y'^2 = 4) :=
by
  intros x y x' y' h_ellipse h_transform
  simp [ellipse, transform] at *
  sorry

end transformed_curve_eq_l130_130949


namespace star_four_three_l130_130963

def star (x y : ℕ) : ℕ := x^2 - x*y + y^2

theorem star_four_three : star 4 3 = 13 := by
  sorry

end star_four_three_l130_130963


namespace correct_card_assignment_l130_130177

theorem correct_card_assignment :
  ∃ (cards : Fin 4 → Fin 4), 
    (¬ (cards 1 = 3 ∨ cards 2 = 3) ∧
     ¬ (cards 0 = 2 ∨ cards 2 = 2) ∧
     ¬ (cards 0 = 1) ∧
     ¬ (cards 0 = 3)) →
    (cards 0 = 4 ∧ cards 1 = 2 ∧ cards 2 = 1 ∧ cards 3 = 3) := 
by {
  sorry
}

end correct_card_assignment_l130_130177


namespace value_of_x_when_y_equals_8_l130_130504

noncomputable def inverse_variation(cube_root : ℝ → ℝ) (k : ℝ) (x y : ℝ) : Prop :=
  y * (cube_root x) = k

noncomputable def cube_root (x : ℝ) : ℝ := x^(1/3)

theorem value_of_x_when_y_equals_8 : 
  ∃ k : ℝ, (inverse_variation cube_root k 8 2) → 
  (inverse_variation cube_root k (1 / 8) 8) := 
sorry

end value_of_x_when_y_equals_8_l130_130504


namespace last_passenger_probability_l130_130947

noncomputable def probability_last_passenger_gets_seat {n : ℕ} (h : n > 0) : ℚ :=
  if n = 1 then 1 else 1/2

theorem last_passenger_probability
  (n : ℕ) (h : n > 0) :
  probability_last_passenger_gets_seat h = 1/2 :=
  sorry

end last_passenger_probability_l130_130947


namespace lily_pad_cover_entire_lake_l130_130980

-- Definitions per the conditions
def doublesInSizeEveryDay (P : ℕ → ℝ) : Prop :=
  ∀ n, P (n + 1) = 2 * P n

-- The initial state that it takes 36 days to cover the lake
def coversEntireLakeIn36Days (P : ℕ → ℝ) (L : ℝ) : Prop :=
  P 36 = L

-- The main theorem to prove
theorem lily_pad_cover_entire_lake (P : ℕ → ℝ) (L : ℝ) (h1 : doublesInSizeEveryDay P) (h2 : coversEntireLakeIn36Days P L) :
  ∃ n, n = 36 := 
by
  sorry

end lily_pad_cover_entire_lake_l130_130980


namespace janine_total_pages_l130_130993

-- Define the conditions
def books_last_month : ℕ := 5
def books_this_month : ℕ := 2 * books_last_month
def books_per_page : ℕ := 10

-- Define the total number of pages she read in two months
def total_pages : ℕ :=
  let total_books := books_last_month + books_this_month
  total_books * books_per_page

-- State the theorem to be proven
theorem janine_total_pages : total_pages = 150 :=
by
  sorry

end janine_total_pages_l130_130993


namespace train_cross_pole_time_l130_130249

-- Definitions based on the conditions
def train_speed_kmh := 54
def train_length_m := 105
def train_speed_ms := (train_speed_kmh * 1000) / 3600
def expected_time := train_length_m / train_speed_ms

-- Theorem statement, encapsulating the problem
theorem train_cross_pole_time : expected_time = 7 := by
  sorry

end train_cross_pole_time_l130_130249


namespace total_pages_l130_130206

theorem total_pages (x : ℕ) (h : 9 + 180 + 3 * (x - 99) = 1392) : x = 500 :=
by
  sorry

end total_pages_l130_130206


namespace star_number_of_intersections_2018_25_l130_130201

-- Definitions for the conditions
def rel_prime (a b : ℕ) : Prop := Nat.gcd a b = 1

def star_intersections (n k : ℕ) : ℕ := 
  n * (k - 1)

-- The main theorem
theorem star_number_of_intersections_2018_25 :
  2018 ≥ 5 ∧ 25 < 2018 / 2 ∧ rel_prime 2018 25 → 
  star_intersections 2018 25 = 48432 :=
by
  intros h
  sorry

end star_number_of_intersections_2018_25_l130_130201


namespace ratio_result_l130_130774

theorem ratio_result (p q r s : ℚ) 
(h1 : p / q = 2) 
(h2 : q / r = 4 / 5) 
(h3 : r / s = 3) : 
  s / p = 5 / 24 :=
sorry

end ratio_result_l130_130774


namespace distinct_9_pointed_stars_l130_130422

-- Define a function to count the distinct n-pointed stars for a given n
def count_distinct_stars (n : ℕ) : ℕ :=
  -- Functionality to count distinct stars will be implemented here
  sorry

-- Theorem stating the number of distinct 9-pointed stars
theorem distinct_9_pointed_stars : count_distinct_stars 9 = 2 :=
  sorry

end distinct_9_pointed_stars_l130_130422


namespace functional_expression_point_M_coordinates_l130_130648

variables (x y : ℝ) (k : ℝ)

-- Given conditions
def proportional_relation : Prop := y + 4 = k * (x - 3)
def initial_condition : Prop := (x = 1 → y = 0)
def point_M : Prop := ∃ m : ℝ, (m + 1, 2 * m) = (1, 0)

-- Proof of the functional expression
theorem functional_expression (h1 : proportional_relation x y k) (h2 : initial_condition x y) :
  ∃ k : ℝ, k = -2 ∧ y = -2 * x + 2 := 
sorry

-- Proof of the coordinates of point M
theorem point_M_coordinates (h : ∀ m : ℝ, (m + 1, 2 * m) = (1, 0)) :
  ∃ m : ℝ, m = 0 ∧ (m + 1, 2 * m) = (1, 0) := 
sorry

end functional_expression_point_M_coordinates_l130_130648


namespace sequence_an_formula_l130_130209

theorem sequence_an_formula (a : ℕ → ℝ) (h₀ : a 1 = 2) (h₁ : ∀ n : ℕ, a (n + 1) = a n^2 - n * a n + 1) :
  ∀ n : ℕ, a n = n + 1 :=
sorry

end sequence_an_formula_l130_130209


namespace profit_percent_l130_130262

theorem profit_percent (marked_price : ℝ) (num_bought : ℝ) (num_payed_price : ℝ) (discount_percent : ℝ) : 
  num_bought = 56 → 
  num_payed_price = 46 → 
  discount_percent = 0.01 →
  marked_price = 1 →
  let cost_price := num_payed_price
  let selling_price_per_pen := marked_price * (1 - discount_percent)
  let total_selling_price := num_bought * selling_price_per_pen
  let profit := total_selling_price - cost_price
  let profit_percent := (profit / cost_price) * 100
  profit_percent = 20.52 :=
by 
  intro hnum_bought hnum_payed_price hdiscount_percent hmarked_price 
  let cost_price := num_payed_price
  let selling_price_per_pen := marked_price * (1 - discount_percent)
  let total_selling_price := num_bought * selling_price_per_pen
  let profit := total_selling_price - cost_price
  let profit_percent := (profit / cost_price) * 100
  sorry

end profit_percent_l130_130262


namespace distinct_digit_S_problem_l130_130222

theorem distinct_digit_S_problem :
  ∃! (S : ℕ), S < 10 ∧ 
  ∃ (P Q R : ℕ), P ≠ Q ∧ Q ≠ R ∧ P ≠ R ∧ P ≠ S ∧ Q ≠ S ∧ R ≠ S ∧ 
  P < 10 ∧ Q < 10 ∧ R < 10 ∧
  ((P + Q = S) ∨ (P + Q = S + 10)) ∧
  (R = 0) :=
sorry

end distinct_digit_S_problem_l130_130222


namespace largest_c_in_range_of_f_l130_130906

theorem largest_c_in_range_of_f (c : ℝ) :
  (∃ x : ℝ, x^2 - 6 * x + c = 2) -> c ≤ 11 :=
by
  sorry

end largest_c_in_range_of_f_l130_130906


namespace geometric_progression_sum_eq_l130_130337

theorem geometric_progression_sum_eq
  (a q b : ℝ) (n : ℕ)
  (hq : q ≠ 1)
  (h : (a * (q^2^n - 1)) / (q - 1) = (b * (q^(2*n) - 1)) / (q^2 - 1)) :
  b = a + a * q :=
by
  sorry

end geometric_progression_sum_eq_l130_130337


namespace nancy_crayons_l130_130436

theorem nancy_crayons (p c t : ℕ) (h1 : p = 41) (h2 : c = 15) (h3 : t = p * c) : t = 615 :=
by
  sorry

end nancy_crayons_l130_130436


namespace George_says_365_l130_130016

-- Definitions based on conditions
def skips_Alice (n : Nat) : Prop :=
  ∃ k, n = 3 * k - 1

def skips_Barbara (n : Nat) : Prop :=
  ∃ k, n = 3 * (3 * k - 1) - 1
  
def skips_Candice (n : Nat) : Prop :=
  ∃ k, n = 3 * (3 * (3 * k - 1) - 1) - 1

def skips_Debbie (n : Nat) : Prop :=
  ∃ k, n = 3 * (3 * (3 * (3 * k - 1) - 1) - 1) - 1

def skips_Eliza (n : Nat) : Prop :=
  ∃ k, n = 3 * (3 * (3 * (3 * (3 * k - 1) - 1) - 1) - 1) - 1

def skips_Fatima (n : Nat) : Prop :=
  ∃ k, n = 3 * (3 * (3 * (3 * (3 * (3 * k - 1) - 1) - 1) - 1) - 1) - 1

def numbers_said_by_students (n : Nat) : Prop :=
  skips_Alice n ∨ skips_Barbara n ∨ skips_Candice n ∨ skips_Debbie n ∨ skips_Eliza n ∨ skips_Fatima n

-- The proof statement
theorem George_says_365 : ¬numbers_said_by_students 365 :=
sorry

end George_says_365_l130_130016


namespace roller_coaster_people_l130_130744

def num_cars : ℕ := 7
def seats_per_car : ℕ := 2
def num_runs : ℕ := 6
def total_seats_per_run : ℕ := num_cars * seats_per_car
def total_people : ℕ := total_seats_per_run * num_runs

theorem roller_coaster_people:
  total_people = 84 := 
by
  sorry

end roller_coaster_people_l130_130744


namespace circle_divides_CD_in_ratio_l130_130046

variable (A B C D : Point)
variable (BC a : ℝ)
variable (AD : ℝ := (1 + Real.sqrt 15) * BC)
variable (radius : ℝ := (2 / 3) * BC)
variable (EF : ℝ := (Real.sqrt 7 / 3) * BC)
variable (is_isosceles_trapezoid : is_isosceles_trapezoid A B C D)
variable (circle_centered_at_C : circle_centered_at C radius)
variable (chord_EF : chord_intersects_base EF AD)

theorem circle_divides_CD_in_ratio (CD DK KC : ℝ) (H1 : CD = 2 * a)
  (H2 : DK + KC = CD) (H3 : KC = CD - DK) : DK / KC = 2 :=
sorry

end circle_divides_CD_in_ratio_l130_130046


namespace coins_remainder_l130_130457

theorem coins_remainder (n : ℕ) (h₁ : n % 8 = 6) (h₂ : n % 7 = 5) : n % 9 = 1 := by
  sorry

end coins_remainder_l130_130457


namespace perimeter_ratio_l130_130316

theorem perimeter_ratio (w l : ℕ) (hfold : w = 8) (lfold : l = 6) 
(folded_w : w / 2 = 4) (folded_l : l / 2 = 3) 
(hcut : w / 4 = 1) (lcut : l / 2 = 3) 
(perimeter_small : ℕ) (perimeter_large : ℕ)
(hperim_small : perimeter_small = 2 * (3 + 4)) 
(hperim_large : perimeter_large = 2 * (6 + 4)) :
(perimeter_small : ℕ) / (perimeter_large : ℕ) = 7 / 10 := sorry

end perimeter_ratio_l130_130316


namespace alex_ride_time_l130_130619

theorem alex_ride_time
  (T : ℝ) -- time on flat ground
  (flat_speed : ℝ := 20) -- flat ground speed
  (uphill_speed : ℝ := 12) -- uphill speed
  (uphill_time : ℝ := 2.5) -- uphill time
  (downhill_speed : ℝ := 24) -- downhill speed
  (downhill_time : ℝ := 1.5) -- downhill time
  (walk_distance : ℝ := 8) -- distance walked
  (total_distance : ℝ := 164) -- total distance to the town
  (hup : uphill_speed * uphill_time = 30)
  (hdown : downhill_speed * downhill_time = 36)
  (hwalk : walk_distance = 8) :
  flat_speed * T + 30 + 36 + 8 = total_distance → T = 4.5 :=
by
  intros h
  sorry

end alex_ride_time_l130_130619


namespace function_unique_l130_130855

open Function

-- Define the domain and codomain
def NatPos : Type := {n : ℕ // n > 0}

-- Define the function f from positive integers to positive integers
noncomputable def f : NatPos → NatPos := sorry

-- Provide the main theorem
theorem function_unique (f : NatPos → NatPos) :
  (∀ (m n : NatPos), (m.val ^ 2 + (f n).val) ∣ ((m.val * (f m).val) + n.val)) →
  (∀ n : NatPos, f n = n) :=
by
  sorry

end function_unique_l130_130855


namespace older_brother_has_17_stamps_l130_130198

def stamps_problem (y : ℕ) : Prop := y + (2 * y + 1) = 25

theorem older_brother_has_17_stamps (y : ℕ) (h : stamps_problem y) : 2 * y + 1 = 17 :=
by
  sorry

end older_brother_has_17_stamps_l130_130198


namespace plane_eq_l130_130042

def gcd4 (a b c d : ℤ) : ℤ := Int.gcd (Int.gcd (Int.gcd (abs a) (abs b)) (abs c)) (abs d)

theorem plane_eq (A B C D : ℤ) (A_pos : A > 0) 
  (gcd_1 : gcd4 A B C D = 1) 
  (H_parallel : (A, B, C) = (3, 2, -4)) 
  (H_point : A * 2 + B * 3 + C * (-1) + D = 0) : 
  A = 3 ∧ B = 2 ∧ C = -4 ∧ D = -16 := 
sorry

end plane_eq_l130_130042


namespace cost_of_each_sale_puppy_l130_130972

-- Conditions
def total_cost (total: ℚ) : Prop := total = 800
def non_sale_puppy_cost (cost: ℚ) : Prop := cost = 175
def num_puppies (num: ℕ) : Prop := num = 5

-- Question to Prove
theorem cost_of_each_sale_puppy (total cost : ℚ) (num: ℕ):
  total_cost total →
  non_sale_puppy_cost cost →
  num_puppies num →
  (total - 2 * cost) / (num - 2) = 150 := 
sorry

end cost_of_each_sale_puppy_l130_130972


namespace average_cost_per_pencil_proof_l130_130829

noncomputable def average_cost_per_pencil (pencils_qty: ℕ) (price: ℝ) (discount_percent: ℝ) (shipping_cost: ℝ) : ℝ :=
  let discounted_price := price * (1 - discount_percent / 100)
  let total_cost := discounted_price + shipping_cost
  let cost_in_cents := total_cost * 100
  cost_in_cents / pencils_qty

theorem average_cost_per_pencil_proof :
  average_cost_per_pencil 300 29.85 10 7.50 = 11 :=
by
  sorry

end average_cost_per_pencil_proof_l130_130829


namespace Guido_costs_42840_l130_130798

def LightningMcQueenCost : ℝ := 140000
def MaterCost : ℝ := 0.1 * LightningMcQueenCost
def SallyCostBeforeModifications : ℝ := 3 * MaterCost
def SallyCostAfterModifications : ℝ := SallyCostBeforeModifications + 0.2 * SallyCostBeforeModifications
def GuidoCost : ℝ := SallyCostAfterModifications - 0.15 * SallyCostAfterModifications

theorem Guido_costs_42840 :
  GuidoCost = 42840 :=
sorry

end Guido_costs_42840_l130_130798


namespace seq_product_l130_130976

theorem seq_product (S : ℕ → ℕ) (a : ℕ → ℕ)
  (hSn : ∀ n, S n = 2^n - 1)
  (ha : ∀ n, a n = if n = 1 then 1 else 2^(n-1)) :
  a 2 * a 6 = 64 :=
by 
  sorry

end seq_product_l130_130976


namespace ratio_of_second_to_first_l130_130452

theorem ratio_of_second_to_first:
  ∀ (x y z : ℕ), 
  (y = 90) → 
  (z = 4 * y) → 
  ((x + y + z) / 3 = 165) → 
  (y / x = 2) := 
by 
  intros x y z h1 h2 h3
  sorry

end ratio_of_second_to_first_l130_130452


namespace M_intersect_N_l130_130789

def M : Set ℝ := {x | (x + 3) * (x - 2) < 0}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def intersection (M N : Set ℝ) : Set ℝ := {x | x ∈ M ∧ x ∈ N}

theorem M_intersect_N :
  intersection M N = {x | 1 ≤ x ∧ x < 2} := 
sorry

end M_intersect_N_l130_130789


namespace intersection_of_set_M_with_complement_of_set_N_l130_130785

theorem intersection_of_set_M_with_complement_of_set_N (U M N : Set ℕ) (hU : U = {1, 2, 3, 4, 5})
  (hM : M = {1, 4, 5}) (hN : N = {1, 3}) : M ∩ (U \ N) = {4, 5} :=
by
  sorry

end intersection_of_set_M_with_complement_of_set_N_l130_130785


namespace rectangle_perimeter_l130_130069

theorem rectangle_perimeter (x y : ℝ) (h1 : 2 * x + y = 44) (h2 : x + 2 * y = 40) : 2 * (x + y) = 56 := 
by
  sorry

end rectangle_perimeter_l130_130069


namespace Kristy_baked_cookies_l130_130608

theorem Kristy_baked_cookies 
  (ate_by_Kristy : ℕ) (given_to_brother : ℕ) 
  (taken_by_first_friend : ℕ) (taken_by_second_friend : ℕ)
  (taken_by_third_friend : ℕ) (cookies_left : ℕ) 
  (h_K : ate_by_Kristy = 2) (h_B : given_to_brother = 1) 
  (h_F1 : taken_by_first_friend = 3) (h_F2 : taken_by_second_friend = 5)
  (h_F3 : taken_by_third_friend = 5) (h_L : cookies_left = 6) :
  ate_by_Kristy + given_to_brother 
  + taken_by_first_friend + taken_by_second_friend 
  + taken_by_third_friend + cookies_left = 22 := 
by
  sorry

end Kristy_baked_cookies_l130_130608


namespace sum_first_twelve_terms_of_arithmetic_sequence_l130_130368

theorem sum_first_twelve_terms_of_arithmetic_sequence :
    let a1 := -3
    let a12 := 48
    let n := 12
    let Sn := (n * (a1 + a12)) / 2
    Sn = 270 := 
by
  sorry

end sum_first_twelve_terms_of_arithmetic_sequence_l130_130368


namespace cori_age_l130_130618

theorem cori_age (C A : ℕ) (hA : A = 19) (hEq : C + 5 = (A + 5) / 3) : C = 3 := by
  rw [hA] at hEq
  norm_num at hEq
  linarith

end cori_age_l130_130618


namespace compare_y1_y2_l130_130366

theorem compare_y1_y2 (a : ℝ) (y1 y2 : ℝ) (h₁ : a < 0) (h₂ : y1 = a * (-1 - 1)^2 + 3) (h₃ : y2 = a * (2 - 1)^2 + 3) : 
  y1 < y2 :=
by
  sorry

end compare_y1_y2_l130_130366


namespace two_circles_with_tangents_l130_130231

theorem two_circles_with_tangents
  (a b : ℝ)                -- radii of the circles
  (length_PQ length_AB : ℝ) -- lengths of the tangents PQ and AB
  (h1 : length_PQ = 14)     -- condition: length of PQ is 14
  (h2 : length_AB = 16)     -- condition: length of AB is 16
  (h3 : length_AB^2 + (a - b)^2 = length_PQ^2 + (a + b)^2) -- from the Pythagorean theorem
  : a * b = 15 := 
sorry

end two_circles_with_tangents_l130_130231


namespace roof_length_width_diff_l130_130812

theorem roof_length_width_diff (w l : ℕ) (h1 : l = 4 * w) (h2 : 784 = l * w) : l - w = 42 := by
  sorry

end roof_length_width_diff_l130_130812


namespace ratio_of_radii_of_circles_l130_130706

theorem ratio_of_radii_of_circles 
  (a b : ℝ) 
  (h1 : a = 6) 
  (h2 : b = 8) 
  (h3 : ∃ (c : ℝ), c = Real.sqrt (a^2 + b^2)) 
  (h4 : ∃ (r R : ℝ), R = c / 2 ∧ r = 24 / (a + b + c)) : R / r = 5 / 2 :=
by
  sorry

end ratio_of_radii_of_circles_l130_130706


namespace total_money_l130_130866

namespace MoneyProof

variables (B J T : ℕ)

-- Given conditions
def condition_beth : Prop := B + 35 = 105
def condition_jan : Prop := J - 10 = B
def condition_tom : Prop := T = 3 * (J - 10)

-- Proof that the total money is $360
theorem total_money (h1 : condition_beth B) (h2 : condition_jan B J) (h3 : condition_tom J T) :
  B + J + T = 360 :=
by
  sorry

end MoneyProof

end total_money_l130_130866


namespace mostSuitableSampleSurvey_l130_130515

-- Conditions
def conditionA := "Security check for passengers before boarding a plane"
def conditionB := "Understanding the amount of physical exercise each classmate does per week"
def conditionC := "Interviewing job applicants for a company's recruitment process"
def conditionD := "Understanding the lifespan of a batch of light bulbs"

-- Define a predicate to determine the most suitable for a sample survey
def isMostSuitableForSampleSurvey (s : String) : Prop :=
  s = conditionD

-- Theorem statement
theorem mostSuitableSampleSurvey :
  isMostSuitableForSampleSurvey conditionD :=
by
  -- Skipping the proof for now
  sorry

end mostSuitableSampleSurvey_l130_130515


namespace intersection_of_A_B_C_l130_130979

-- Define the sets A, B, and C as given conditions:
def A : Set ℕ := { x | ∃ n : ℕ, x = 2 * n }
def B : Set ℕ := { x | ∃ n : ℕ, x = 3 * n }
def C : Set ℕ := { x | ∃ n : ℕ, x = n ^ 2 }

-- Prove that A ∩ B ∩ C = { x | ∃ n : ℕ, x = 36 * n ^ 2 }
theorem intersection_of_A_B_C :
  (A ∩ B ∩ C) = { x | ∃ n : ℕ, x = 36 * n ^ 2 } :=
sorry

end intersection_of_A_B_C_l130_130979


namespace find_circle_center_l130_130191

-- The statement to prove that the center of the given circle equation is (1, -2)
theorem find_circle_center : 
  ∃ (h k : ℝ), 3 * x^2 - 6 * x + 3 * y^2 + 12 * y - 75 = 0 → (h, k) = (1, -2) := 
by
  sorry

end find_circle_center_l130_130191


namespace find_other_number_l130_130438

theorem find_other_number (lcm_ab hcf_ab : ℕ) (A : ℕ) (h_lcm: Nat.lcm A (B) = lcm_ab)
  (h_hcf : Nat.gcd A (B) = hcf_ab) (h_a : A = 48) (h_lcm_value: lcm_ab = 192) (h_hcf_value: hcf_ab = 16) :
  B = 64 :=
by
  sorry

end find_other_number_l130_130438


namespace polygon_sides_eq_7_l130_130801

theorem polygon_sides_eq_7 (n : ℕ) (h : n * (n - 3) / 2 = 2 * n) : n = 7 := 
by 
  sorry

end polygon_sides_eq_7_l130_130801


namespace distance_to_place_l130_130996

theorem distance_to_place (rowing_speed : ℝ) (current_speed : ℝ) (total_time : ℝ) (D : ℝ) :
  rowing_speed = 5 ∧ current_speed = 1 ∧ total_time = 1 →
  D = 2.4 :=
by
  -- Rowing Parameters
  let V_d := rowing_speed + current_speed
  let V_u := rowing_speed - current_speed
  
  -- Time Variables
  let T_d := total_time / (V_d + V_u)
  let T_u := total_time - T_d

  -- Distance Calculations
  let D1 := V_d * T_d
  let D2 := V_u * T_u

  -- Prove D is the same distance both upstream and downstream
  sorry

end distance_to_place_l130_130996


namespace find_sqrt_abc_sum_l130_130308

theorem find_sqrt_abc_sum (a b c : ℝ) (h1 : b + c = 20) (h2 : c + a = 22) (h3 : a + b = 24) :
    Real.sqrt (a * b * c * (a + b + c)) = 206.1 := by
  sorry

end find_sqrt_abc_sum_l130_130308


namespace equation_D_is_linear_l130_130358

-- Definitions according to the given conditions
def equation_A (x y : ℝ) := x + 2 * y = 3
def equation_B (x : ℝ) := 3 * x - 2
def equation_C (x : ℝ) := x^2 + x = 6
def equation_D (x : ℝ) := (1 / 3) * x - 2 = 3

-- Properties of a linear equation
def is_linear (eq : ℝ → Prop) : Prop :=
∃ a b c : ℝ, (∃ x : ℝ, eq x = (a * x + b = c)) ∧ a ≠ 0

-- Specifying that equation_D is linear
theorem equation_D_is_linear : is_linear equation_D :=
by
  sorry

end equation_D_is_linear_l130_130358


namespace complete_the_square_correct_l130_130387

noncomputable def complete_the_square (x : ℝ) : Prop :=
  x^2 - 2 * x - 1 = 0 ↔ (x - 1)^2 = 2

theorem complete_the_square_correct : ∀ x : ℝ, complete_the_square x := by
  sorry

end complete_the_square_correct_l130_130387


namespace seats_per_bus_l130_130794

-- Conditions
def total_students : ℕ := 180
def total_buses : ℕ := 3

-- Theorem Statement
theorem seats_per_bus : (total_students / total_buses) = 60 := 
by 
  sorry

end seats_per_bus_l130_130794


namespace trig_product_identity_l130_130984

theorem trig_product_identity :
  (1 + Real.sin (Real.pi / 12)) * (1 + Real.sin (5 * Real.pi / 12)) *
  (1 + Real.sin (7 * Real.pi / 12)) * (1 + Real.sin (11 * Real.pi / 12)) =
  (1 + Real.sin (Real.pi / 12))^2 * (1 + Real.sin (5 * Real.pi / 12))^2 :=
by
  sorry

end trig_product_identity_l130_130984


namespace least_five_digit_congruent_to_7_mod_18_l130_130686

theorem least_five_digit_congruent_to_7_mod_18 :
  ∃ (n : ℕ), 10000 ≤ n ∧ n < 100000 ∧ n % 18 = 7 ∧ ∀ m : ℕ, 10000 ≤ m ∧ m < n → m % 18 ≠ 7 :=
by
  sorry

end least_five_digit_congruent_to_7_mod_18_l130_130686


namespace min_value_fraction_sum_l130_130637

theorem min_value_fraction_sum (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1) : 
  ∃ (x y : ℝ), x = 2/5 ∧ y = 3/5 ∧ (∃ (k : ℝ), k = 4/x + 9/y ∧ k = 25) :=
by
  sorry

end min_value_fraction_sum_l130_130637


namespace find_a_l130_130563

def A (a : ℤ) : Set ℤ := {-4, 2 * a - 1, a * a}
def B (a : ℤ) : Set ℤ := {a - 5, 1 - a, 9}

theorem find_a (a : ℤ) : (9 ∈ (A a ∩ B a)) ∧ (A a ∩ B a = {9}) ↔ a = -3 :=
by
  sorry

end find_a_l130_130563


namespace chess_tournament_max_N_l130_130309

theorem chess_tournament_max_N :
  ∃ (N : ℕ), N = 120 ∧
  ∀ (S T : Finset ℕ), S.card = 15 ∧ T.card = 15 ∧
  (∀ s ∈ S, ∀ t ∈ T, (s, t) ∈ (S.product T)) ∧
  (∀ s, ∃! t, (s, t) ∈ (S.product T)) → 
  ∃ (ways_one_game : ℕ), ways_one_game = N ∧ ways_one_game = 120 :=
by
  sorry

end chess_tournament_max_N_l130_130309


namespace neg_exists_lt_1000_l130_130051

open Nat

theorem neg_exists_lt_1000 : (¬ ∃ n : ℕ, 2^n < 1000) = ∀ n : ℕ, 2^n ≥ 1000 := by
  sorry

end neg_exists_lt_1000_l130_130051


namespace constant_term_value_l130_130620

theorem constant_term_value :
  ∀ (x y z k : ℤ), (4 * x + y + z = 80) → (2 * x - y - z = 40) → (x = 20) → (3 * x + y - z = k) → (k = 60) :=
by 
  intros x y z k h₁ h₂ hx h₃
  sorry

end constant_term_value_l130_130620


namespace abby_bridget_adjacent_probability_l130_130127

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def probability_adjacent : ℚ :=
  let total_seats := 9
  let ab_adj_same_row_pairs := 9
  let ab_adj_diagonal_pairs := 4
  let favorable_outcomes := (ab_adj_same_row_pairs + ab_adj_diagonal_pairs) * 2 * factorial 7
  let total_outcomes := factorial total_seats
  favorable_outcomes / total_outcomes

theorem abby_bridget_adjacent_probability :
  probability_adjacent = 13 / 36 :=
by
  sorry

end abby_bridget_adjacent_probability_l130_130127


namespace sticker_price_of_laptop_l130_130943

variable (x : ℝ)

-- Conditions
noncomputable def price_store_A : ℝ := 0.90 * x - 100
noncomputable def price_store_B : ℝ := 0.80 * x
noncomputable def savings : ℝ := price_store_B x - price_store_A x

-- Theorem statement
theorem sticker_price_of_laptop (x : ℝ) (h : savings x = 20) : x = 800 :=
by
  sorry

end sticker_price_of_laptop_l130_130943


namespace car_bus_initial_speed_l130_130544

theorem car_bus_initial_speed {d : ℝ} {t : ℝ} {s_c : ℝ} {s_b : ℝ}
    (h1 : t = 4) 
    (h2 : s_c = s_b + 8) 
    (h3 : d = 384)
    (h4 : ∀ t, 0 ≤ t → t ≤ 2 → d = s_c * t + s_b * t) 
    (h5 : ∀ t, 2 < t → t ≤ 4 → d = (s_c - 10) * (t - 2) + s_b * (t - 2)) 
    : s_b = 46.5 ∧ s_c = 54.5 := 
by 
    sorry

end car_bus_initial_speed_l130_130544


namespace negation_of_not_both_are_not_even_l130_130054

variables {a b : ℕ}

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem negation_of_not_both_are_not_even :
  ¬ (¬ is_even a ∧ ¬ is_even b) ↔ (is_even a ∨ is_even b) :=
by
  sorry

end negation_of_not_both_are_not_even_l130_130054


namespace tan_sub_pi_over_4_l130_130832

variables (α : ℝ)
axiom tan_alpha : Real.tan α = 1 / 6

theorem tan_sub_pi_over_4 : Real.tan (α - Real.pi / 4) = -5 / 7 := by
  sorry

end tan_sub_pi_over_4_l130_130832


namespace total_fish_count_l130_130919

theorem total_fish_count (num_fishbowls : ℕ) (fish_per_bowl : ℕ)
  (h1 : num_fishbowls = 261) (h2 : fish_per_bowl = 23) : 
  num_fishbowls * fish_per_bowl = 6003 := 
  by 
    sorry

end total_fish_count_l130_130919


namespace horse_distribution_l130_130537

variable (b₁ b₂ b₃ : ℕ) 
variable (a : Matrix (Fin 3) (Fin 3) ℝ)
variable (h1 : a 0 0 > a 0 1 ∧ a 0 0 > a 0 2)
variable (h2 : a 1 1 > a 1 0 ∧ a 1 1 > a 1 2)
variable (h3 : a 2 2 > a 2 0 ∧ a 2 2 > a 2 1)

theorem horse_distribution :
  ∃ n : ℕ, ∀ (b₁ b₂ b₃ : ℕ), min b₁ (min b₂ b₃) > n → 
  ∃ (x1 y1 x2 y2 x3 y3 : ℕ), 3*x1 + y1 = b₁ ∧ 3*x2 + y2 = b₂ ∧ 3*x3 + y3 = b₃ ∧
  y1*a 0 0 > y2*a 0 1 ∧ y1*a 0 0 > y3*a 0 2 ∧
  y2*a 1 1 > y1*a 1 0 ∧ y2*a 1 1 > y3*a 1 2 ∧
  y3*a 2 2 > y1*a 2 0 ∧ y3*a 2 2 > y2*a 2 1 :=
sorry

end horse_distribution_l130_130537


namespace pascal_elements_sum_l130_130554

theorem pascal_elements_sum :
  (Nat.choose 20 4 + Nat.choose 20 5) = 20349 :=
by
  sorry

end pascal_elements_sum_l130_130554


namespace perfect_square_of_division_l130_130235

theorem perfect_square_of_division (a b : ℤ) (ha : 0 < a) (hb : 0 < b) 
  (h : (a * b + 1) ∣ (a^2 + b^2)) : ∃ k : ℤ, 0 < k ∧ k^2 = (a^2 + b^2) / (a * b + 1) :=
by
  sorry

end perfect_square_of_division_l130_130235


namespace inequality_solution_l130_130237

def solutionSetInequality (x : ℝ) : Prop :=
  (x > 1 ∨ x < -2)

theorem inequality_solution (x : ℝ) : 
  (x+2)/(x-1) > 0 ↔ solutionSetInequality x := 
  sorry

end inequality_solution_l130_130237


namespace value_of_expression_l130_130320

theorem value_of_expression (x : ℝ) (h : x + 3 = 10) : 5 * x + 15 = 50 := by
  sorry

end value_of_expression_l130_130320


namespace liquid_X_percentage_in_B_l130_130538

noncomputable def percentage_of_solution_B (X_A : ℝ) (w_A w_B total_X : ℝ) : ℝ :=
  let X_B := (total_X - (w_A * (X_A / 100))) / w_B 
  X_B * 100

theorem liquid_X_percentage_in_B :
  percentage_of_solution_B 0.8 500 700 19.92 = 2.274 := by
  sorry

end liquid_X_percentage_in_B_l130_130538


namespace problem_acute_angles_l130_130531

theorem problem_acute_angles (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) 
  (h1 : 3 * (Real.sin α) ^ 2 + 2 * (Real.sin β) ^ 2 = 1)
  (h2 : 3 * Real.sin (2 * α) - 2 * Real.sin (2 * β) = 0) :
  α + 2 * β = π / 2 := 
by 
  sorry

end problem_acute_angles_l130_130531


namespace perpendicular_vectors_m_solution_l130_130981

theorem perpendicular_vectors_m_solution (m : ℝ) 
  (a : ℝ × ℝ := (1, 2)) 
  (b : ℝ × ℝ := (m, 1)) 
  (h_perp : a.1 * b.1 + a.2 * b.2 = 0) : m = -2 := by
  sorry

end perpendicular_vectors_m_solution_l130_130981


namespace log_50_between_consecutive_integers_l130_130244

theorem log_50_between_consecutive_integers :
    (∃ (m n : ℤ), m < n ∧ m < Real.log 50 / Real.log 10 ∧ Real.log 50 / Real.log 10 < n ∧ m + n = 3) :=
by
  have log_10_eq_1 : Real.log 10 / Real.log 10 = 1 := by sorry
  have log_100_eq_2 : Real.log 100 / Real.log 10 = 2 := by sorry
  have log_increasing : ∀ (x y : ℝ), x < y → Real.log x / Real.log 10 < Real.log y / Real.log 10 := by sorry
  have interval : 10 < 50 ∧ 50 < 100 := by sorry
  use 1
  use 2
  sorry

end log_50_between_consecutive_integers_l130_130244


namespace sqrt_interval_l130_130149

theorem sqrt_interval :
  let expr := (Real.sqrt 18) / 3 - (Real.sqrt 2) * (Real.sqrt (1 / 2))
  0 < expr ∧ expr < 1 :=
by
  let expr := (Real.sqrt 18) / 3 - (Real.sqrt 2) * (Real.sqrt (1 / 2))
  sorry

end sqrt_interval_l130_130149


namespace number_of_questions_in_test_l130_130732

variable (n : ℕ) -- the total number of questions
variable (correct_answers : ℕ) -- the number of correct answers
variable (sections : ℕ) -- number of sections in the test
variable (questions_per_section : ℕ) -- number of questions per section
variable (percentage_correct : ℚ) -- percentage of correct answers

-- Given conditions
def conditions := 
  correct_answers = 32 ∧ 
  sections = 5 ∧ 
  questions_per_section * sections = n ∧ 
  (70 : ℚ) < percentage_correct ∧ 
  percentage_correct < 77 ∧ 
  percentage_correct * n = 3200

-- The main statement to prove
theorem number_of_questions_in_test : conditions n correct_answers sections questions_per_section percentage_correct → 
  n = 45 :=
by
  sorry

end number_of_questions_in_test_l130_130732


namespace david_english_marks_l130_130186

def david_marks (math physics chemistry biology avg : ℕ) : ℕ :=
  avg * 5 - (math + physics + chemistry + biology)

theorem david_english_marks :
  let math := 95
  let physics := 82
  let chemistry := 97
  let biology := 95
  let avg := 93
  david_marks math physics chemistry biology avg = 96 :=
by
  -- Proof is skipped
  sorry

end david_english_marks_l130_130186


namespace parallelogram_area_increase_l130_130596
open Real

/-- The area of the parallelogram increases by 600 square meters when the base is increased by 20 meters. -/
theorem parallelogram_area_increase :
  ∀ (base height new_base : ℝ), 
    base = 65 → height = 30 → new_base = base + 20 → 
    (new_base * height - base * height) = 600 := 
by
  sorry

end parallelogram_area_increase_l130_130596


namespace square_garden_perimeter_l130_130039

theorem square_garden_perimeter (q p : ℝ) (h : q = 2 * p + 20) : p = 40 :=
sorry

end square_garden_perimeter_l130_130039


namespace doughnuts_served_initially_l130_130247

def initial_doughnuts_served (staff_count : Nat) (doughnuts_per_staff : Nat) (doughnuts_left : Nat) : Nat :=
  staff_count * doughnuts_per_staff + doughnuts_left

theorem doughnuts_served_initially :
  ∀ (staff_count doughnuts_per_staff doughnuts_left : Nat), staff_count = 19 → doughnuts_per_staff = 2 → doughnuts_left = 12 →
  initial_doughnuts_served staff_count doughnuts_per_staff doughnuts_left = 50 :=
by
  intros staff_count doughnuts_per_staff doughnuts_left hstaff hdonuts hleft
  rw [hstaff, hdonuts, hleft]
  rfl

#check doughnuts_served_initially

end doughnuts_served_initially_l130_130247


namespace product_of_numbers_l130_130028

theorem product_of_numbers (a b : ℝ) 
  (h1 : a + b = 5 * (a - b))
  (h2 : a * b = 18 * (a - b)) : 
  a * b = 54 :=
by
  sorry

end product_of_numbers_l130_130028


namespace clubs_popularity_order_l130_130431

theorem clubs_popularity_order (chess drama art science : ℚ)
  (h_chess: chess = 14/35) (h_drama: drama = 9/28) (h_art: art = 11/21) (h_science: science = 8/15) :
  science > art ∧ art > chess ∧ chess > drama :=
by {
  -- Place proof steps here (optional)
  sorry
}

end clubs_popularity_order_l130_130431


namespace eight_diamond_five_l130_130461

def diamond (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

theorem eight_diamond_five : diamond 8 5 = 160 :=
by sorry

end eight_diamond_five_l130_130461


namespace max_equal_product_l130_130804

theorem max_equal_product (a b c d e f : ℕ) (h1 : a = 10) (h2 : b = 15) (h3 : c = 20) (h4 : d = 30) (h5 : e = 40) (h6 : f = 60) :
  ∃ S, (a * b * c * d * e * f) * 450 = S^3 ∧ S = 18000 := 
by
  sorry

end max_equal_product_l130_130804


namespace slope_ge_one_sum_pq_eq_17_l130_130278

noncomputable def Q_prob_satisfaction : ℚ := 1/16

theorem slope_ge_one_sum_pq_eq_17 :
  let p := 1
  let q := 16
  p + q = 17 := by
  sorry

end slope_ge_one_sum_pq_eq_17_l130_130278


namespace find_b_l130_130694

theorem find_b (a b : ℝ) (h1 : 2 * a + b = 6) (h2 : -2 * a + b = 2) : b = 4 :=
sorry

end find_b_l130_130694


namespace average_temperature_MTWT_l130_130640

theorem average_temperature_MTWT (T_TWTF : ℝ) (T_M : ℝ) (T_F : ℝ) (T_MTWT : ℝ) :
    T_TWTF = 40 →
    T_M = 42 →
    T_F = 10 →
    T_MTWT = ((4 * T_TWTF - T_F + T_M) / 4) →
    T_MTWT = 48 := 
by
  intros hT_TWTF hT_M hT_F hT_MTWT
  rw [hT_TWTF, hT_M, hT_F] at hT_MTWT
  norm_num at hT_MTWT
  exact hT_MTWT

end average_temperature_MTWT_l130_130640


namespace greatest_sum_l130_130621

-- stating the conditions
def condition1 (x y : ℝ) := x^2 + y^2 = 130
def condition2 (x y : ℝ) := x * y = 45

-- proving the result
theorem greatest_sum (x y : ℝ) 
  (h1 : condition1 x y) 
  (h2 : condition2 x y) : 
  x + y = 10 * Real.sqrt 2.2 :=
sorry

end greatest_sum_l130_130621


namespace sara_staircase_l130_130631

theorem sara_staircase (n : ℕ) (h : 2 * n * (n + 1) = 360) : n = 13 :=
sorry

end sara_staircase_l130_130631


namespace brenda_age_correct_l130_130649

open Nat

noncomputable def brenda_age_proof : Prop :=
  ∃ (A B J : ℚ), 
  (A = 4 * B) ∧ 
  (J = B + 8) ∧ 
  (A = J) ∧ 
  (B = 8 / 3)

theorem brenda_age_correct : brenda_age_proof := 
  sorry

end brenda_age_correct_l130_130649


namespace abc_minus_def_l130_130169

def f (x y z : ℕ) : ℕ := 5^x * 2^y * 3^z

theorem abc_minus_def {a b c d e f : ℕ} (ha : a = d) (hb : b = e) (hc : c = f + 1) : 
  (100 * a + 10 * b + c) - (100 * d + 10 * e + f) = 1 :=
by
  -- Proof omitted
  sorry

end abc_minus_def_l130_130169


namespace roots_of_quadratic_l130_130234

theorem roots_of_quadratic (a b : ℝ) (h : ab ≠ 0) : 
  (a + b = -2 * b) ∧ (a * b = a) → (a = -3 ∧ b = 1) :=
by
  sorry

end roots_of_quadratic_l130_130234


namespace discount_rate_on_pony_jeans_l130_130113

theorem discount_rate_on_pony_jeans 
  (F P : ℝ) 
  (H1 : F + P = 22) 
  (H2 : 45 * F + 36 * P = 882) : 
  P = 12 :=
by
  sorry

end discount_rate_on_pony_jeans_l130_130113


namespace inequality_proof_problem_l130_130743

theorem inequality_proof_problem (a b : ℝ) (h : a < b ∧ b < 0) : (1 / (a - b) ≤ 1 / a) :=
sorry

end inequality_proof_problem_l130_130743


namespace percentage_fertilizer_in_second_solution_l130_130916

theorem percentage_fertilizer_in_second_solution 
    (v1 v2 v3 : ℝ) 
    (p1 p2 p3 : ℝ) 
    (h1 : v1 = 20) 
    (h2 : v2 + v1 = 42) 
    (h3 : p1 = 74 / 100) 
    (h4 : p2 = 63 / 100) 
    (h5 : v3 = (63 * 42 - 74 * 20) / 22) 
    : p3 = (53 / 100) :=
by
  sorry

end percentage_fertilizer_in_second_solution_l130_130916


namespace time_to_traverse_nth_mile_l130_130526

theorem time_to_traverse_nth_mile (n : ℕ) (h : n ≥ 2) :
  (∃ k : ℝ, (∀ d : ℝ, d = n - 1 → (s_n = k / d)) ∧ (s_2 = 1 / 2)) → 
  t_n = 2 * (n - 1) :=
by 
  sorry

end time_to_traverse_nth_mile_l130_130526


namespace amy_lily_tie_l130_130848

noncomputable def tie_probability : ℚ :=
    let amy_win := (2 / 5 : ℚ)
    let lily_win := (1 / 4 : ℚ)
    let total_win := amy_win + lily_win
    1 - total_win

theorem amy_lily_tie (h1 : (2 / 5 : ℚ) = 2 / 5) 
                     (h2 : (1 / 4 : ℚ) = 1 / 4)
                     (h3 : (2 / 5 : ℚ) ≥ 2 * (1 / 4 : ℚ) ∨ (1 / 4 : ℚ) ≥ 2 * (2 / 5 : ℚ)) :
    tie_probability = 7 / 20 :=
by
  sorry

end amy_lily_tie_l130_130848


namespace space_mission_contribution_l130_130967

theorem space_mission_contribution 
  (mission_cost_million : ℕ := 30000) 
  (combined_population_million : ℕ := 350) : 
  mission_cost_million / combined_population_million = 86 := by
  sorry

end space_mission_contribution_l130_130967


namespace bottles_more_than_apples_l130_130738

-- Definitions given in the conditions
def apples : ℕ := 36
def regular_soda_bottles : ℕ := 80
def diet_soda_bottles : ℕ := 54

-- Theorem statement representing the question
theorem bottles_more_than_apples : (regular_soda_bottles + diet_soda_bottles) - apples = 98 :=
by
  sorry

end bottles_more_than_apples_l130_130738


namespace quadratic_sum_roots_l130_130840

theorem quadratic_sum_roots {a b : ℝ}
  (h1 : ∀ x, x^2 - a * x + b < 0 ↔ -1 < x ∧ x < 3) :
  a + b = -1 :=
sorry

end quadratic_sum_roots_l130_130840


namespace p_arithmetic_fibonacci_term_correct_l130_130443

noncomputable def p_arithmetic_fibonacci_term (p : ℕ) : ℝ :=
  5 ^ ((p - 1) / 2)

theorem p_arithmetic_fibonacci_term_correct (p : ℕ) : p_arithmetic_fibonacci_term p = 5 ^ ((p - 1) / 2) := 
by 
  rfl -- direct application of the definition

#check p_arithmetic_fibonacci_term_correct

end p_arithmetic_fibonacci_term_correct_l130_130443


namespace problem_solution_l130_130628

universe u

variable (U : Set Nat) (M : Set Nat)
variable (complement_U_M : Set Nat)

axiom U_def : U = {1, 2, 3, 4, 5}
axiom complement_U_M_def : complement_U_M = {1, 3}
axiom M_def : M = U \ complement_U_M

theorem problem_solution : 2 ∈ M := by
  sorry

end problem_solution_l130_130628


namespace pond_volume_l130_130108

theorem pond_volume (L W H : ℝ) (hL : L = 20) (hW : W = 10) (hH : H = 5) : 
  L * W * H = 1000 :=
by
  rw [hL, hW, hH]
  norm_num

end pond_volume_l130_130108


namespace inverse_proportion_increasing_implication_l130_130348

theorem inverse_proportion_increasing_implication (m x : ℝ) (h1 : x > 0) (h2 : ∀ x1 x2, x1 > 0 → x2 > 0 → x1 < x2 → (m + 3) / x1 < (m + 3) / x2) : m < -3 :=
by
  sorry

end inverse_proportion_increasing_implication_l130_130348


namespace cylinder_volume_calc_l130_130664

def cylinder_volume (r h : ℝ) (π : ℝ) : ℝ := π * r^2 * h

theorem cylinder_volume_calc :
    cylinder_volume 5 (5 + 3) 3.14 = 628 :=
by
  -- We set r = 5, h = 8 (since h = r + 3), and π = 3.14 to calculate the volume
  sorry

end cylinder_volume_calc_l130_130664


namespace exponent_multiplication_l130_130048

variable (a : ℝ) (m : ℤ)

theorem exponent_multiplication (a : ℝ) (m : ℤ) : a^(2 * m + 2) = a^(2 * m) * a^2 := 
sorry

end exponent_multiplication_l130_130048


namespace functional_equation_solution_l130_130453

theorem functional_equation_solution (f : ℕ+ → ℕ+) :
  (∀ n : ℕ+, f (f (f n)) + f (f n) + f n = 3 * n) →
  ∀ n : ℕ+, f n = n :=
by
  intro h
  sorry

end functional_equation_solution_l130_130453


namespace visible_during_metaphase_l130_130415

-- Define the structures which could be present in a plant cell during mitosis.
inductive Structure
| Chromosomes
| Spindle
| CellWall
| MetaphasePlate
| CellMembrane
| Nucleus
| Nucleolus

open Structure

-- Define what structures are visible during metaphase.
def visibleStructures (phase : String) : Set Structure :=
  if phase = "metaphase" then
    {Chromosomes, Spindle, CellWall}
  else
    ∅

-- The proof statement
theorem visible_during_metaphase :
  visibleStructures "metaphase" = {Chromosomes, Spindle, CellWall} :=
by
  sorry

end visible_during_metaphase_l130_130415


namespace arithmetic_sequence_sum_l130_130592

noncomputable def S (n : ℕ) : ℤ :=
  n * (-2012) + n * (n - 1) / 2 * (1 : ℤ)

theorem arithmetic_sequence_sum :
  (S 2012) / 2012 - (S 10) / 10 = 2002 → S 2017 = 2017 :=
by
  sorry

end arithmetic_sequence_sum_l130_130592


namespace plot_area_in_acres_l130_130816

theorem plot_area_in_acres :
  let scale_cm_to_miles : ℝ := 3
  let base1_cm : ℝ := 20
  let base2_cm : ℝ := 25
  let height_cm : ℝ := 15
  let miles_to_acres : ℝ := 640
  let area_trapezoid_cm2 := (1 / 2) * (base1_cm + base2_cm) * height_cm
  let area_trapezoid_miles2 := area_trapezoid_cm2 * (scale_cm_to_miles ^ 2)
  let area_trapezoid_acres := area_trapezoid_miles2 * miles_to_acres
  area_trapezoid_acres = 1944000 := by
    sorry

end plot_area_in_acres_l130_130816


namespace tan_value_l130_130188

variable (a : ℕ → ℝ) (b : ℕ → ℝ)
variable (a_geom : ∀ m n : ℕ, a m / a n = a (m - n))
variable (b_arith : ∃ c d : ℝ, ∀ n : ℕ, b n = c + n * d)
variable (ha : a 1 * a 6 * a 11 = -3 * Real.sqrt 3)
variable (hb : b 1 + b 6 + b 11 = 7 * Real.pi)

theorem tan_value : Real.tan ((b 3 + b 9) / (1 - a 4 * a 8)) = -Real.sqrt 3 := by
  sorry

end tan_value_l130_130188


namespace value_of_m_l130_130555

theorem value_of_m : 
  (2 ^ 1999 - 2 ^ 1998 - 2 ^ 1997 + 2 ^ 1996 - 2 ^ 1995 = m * 2 ^ 1995) -> m = 5 :=
by 
  sorry

end value_of_m_l130_130555


namespace proportion_of_capacity_filled_l130_130959

noncomputable def milk_proportion_8cup_bottle : ℚ := 16 / 3
noncomputable def total_milk := 8

theorem proportion_of_capacity_filled :
  ∃ p : ℚ, (8 * p = milk_proportion_8cup_bottle) ∧ (4 * p = total_milk - milk_proportion_8cup_bottle) ∧ (p = 2 / 3) :=
by
  sorry

end proportion_of_capacity_filled_l130_130959


namespace directrix_of_parabola_l130_130486

theorem directrix_of_parabola :
  ∀ (x y : ℝ), y = - (1 / 8) * x^2 → y = 2 :=
by
  sorry

end directrix_of_parabola_l130_130486


namespace div_by_7_iff_sum_div_by_7_l130_130151

theorem div_by_7_iff_sum_div_by_7 (a b : ℕ) : 
  (101 * a + 10 * b) % 7 = 0 ↔ (a + b) % 7 = 0 := 
by
  sorry

end div_by_7_iff_sum_div_by_7_l130_130151


namespace a5_value_l130_130758

def sequence_sum (n : ℕ) (a : ℕ → ℤ) : ℤ :=
  (Finset.range n).sum a

theorem a5_value (a : ℕ → ℤ) (h : ∀ n : ℕ, 0 < n → sequence_sum n a = (1 / 2 : ℚ) * (a n : ℚ) + 1) :
  a 5 = 2 := by
  sorry

end a5_value_l130_130758


namespace bucket_volume_l130_130941

theorem bucket_volume :
  ∃ (V : ℝ), -- The total volume of the bucket
    (∀ (rate_A rate_B rate_combined : ℝ),
      rate_A = 3 ∧ 
      rate_B = V / 60 ∧ 
      rate_combined = V / 10 ∧ 
      rate_A + rate_B = rate_combined) →
    V = 36 :=
by
  sorry

end bucket_volume_l130_130941


namespace complete_the_square_result_l130_130236

-- Define the equation
def initial_eq (x : ℝ) : Prop := x^2 + 4 * x + 3 = 0

-- State the theorem based on the condition and required to prove the question equals the answer
theorem complete_the_square_result (x : ℝ) : initial_eq x → (x + 2) ^ 2 = 1 := 
by
  intro h
  -- Proof is to be skipped
  sorry

end complete_the_square_result_l130_130236


namespace intersection_eq_set_l130_130925

-- Define set A based on the inequality
def A : Set ℝ := {x | x^2 + x - 2 < 0}

-- Define set B based on the inequality
def B : Set ℝ := {x | 0 ≤ Real.log (x + 1) / Real.log 2 ∧ Real.log (x + 1) / Real.log 2 < 2}

-- Translate the question to a lean theorem
theorem intersection_eq_set : (A ∩ B) = {x | 0 ≤ x ∧ x < 1} := 
sorry

end intersection_eq_set_l130_130925


namespace distinct_solutions_difference_l130_130376

theorem distinct_solutions_difference (r s : ℝ) (h1 : r ≠ s) (h2 : (6 * r - 18) / (r ^ 2 + 3 * r - 18) = r + 3) (h3 : (6 * s - 18) / (s ^ 2 + 3 * s - 18) = s + 3) (h4 : r > s) : r - s = 3 :=
sorry

end distinct_solutions_difference_l130_130376


namespace sufficient_condition_for_inequality_l130_130306

theorem sufficient_condition_for_inequality (x : ℝ) : (1 - 1/x > 0) → (x > 1) :=
by
  sorry

end sufficient_condition_for_inequality_l130_130306


namespace trace_bag_weight_l130_130064

-- Definitions for the given problem
def weight_gordon_bag1 := 3
def weight_gordon_bag2 := 7
def total_weight_gordon := weight_gordon_bag1 + weight_gordon_bag2

noncomputable def weight_trace_one_bag : ℕ :=
  sorry

-- Theorem for what we need to prove
theorem trace_bag_weight :
  total_weight_gordon = 10 ∧
  weight_trace_one_bag = total_weight_gordon / 5 :=
sorry

end trace_bag_weight_l130_130064


namespace solution_set_l130_130885

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 1 else -1

theorem solution_set :
  {x : ℝ | x + (x + 2) * f (x + 2) ≤ 5} = {x : ℝ | x ≤ 3 / 2} :=
by {
  sorry
}

end solution_set_l130_130885


namespace range_of_x_l130_130154

-- Defining the conditions
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = - (f x)

def monotonically_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f y ≤ f x

-- Given conditions in Lean
axiom f : ℝ → ℝ
axiom h_odd : odd_function f
axiom h_decreasing_pos : ∀ x y, 0 < x ∧ x < y → f y ≤ f x
axiom h_f4 : f 4 = 0

-- To prove the range of x for which f(x-3) ≤ 0
theorem range_of_x :
    {x : ℝ | f (x - 3) ≤ 0} = {x : ℝ | -1 ≤ x ∧ x < 3} ∪ {x : ℝ | 7 ≤ x} :=
by
  sorry

end range_of_x_l130_130154


namespace tan_alpha_l130_130045

variable (α : Real)
-- Condition 1: α is an angle in the second quadrant
-- This implies that π/2 < α < π and sin α = 4 / 5
variable (h1 : π / 2 < α ∧ α < π) 
variable (h2 : Real.sin α = 4 / 5)

theorem tan_alpha : Real.tan α = -4 / 3 :=
by
  sorry

end tan_alpha_l130_130045


namespace find_greater_solution_of_quadratic_l130_130015

theorem find_greater_solution_of_quadratic:
  (x^2 + 14 * x - 88 = 0 → x = -22 ∨ x = 4) → (∀ x₁ x₂, (x₁ = -22 ∨ x₁ = 4) ∧ (x₂ = -22 ∨ x₂ = 4) → max x₁ x₂ = 4) :=
by
  intros h x₁ x₂ hx1x2
  -- proof omitted
  sorry

end find_greater_solution_of_quadratic_l130_130015


namespace missing_files_correct_l130_130610

def total_files : ℕ := 60
def files_in_morning : ℕ := total_files / 2
def files_in_afternoon : ℕ := 15
def missing_files : ℕ := total_files - (files_in_morning + files_in_afternoon)

theorem missing_files_correct : missing_files = 15 := by
  sorry

end missing_files_correct_l130_130610


namespace smallest_sum_of_three_l130_130893

open Finset

-- Define the set of numbers
def my_set : Finset ℤ := {10, 2, -4, 15, -7}

-- Statement of the problem: Prove the smallest sum of any three different numbers from the set is -9
theorem smallest_sum_of_three :
  ∃ (a b c : ℤ), a ∈ my_set ∧ b ∈ my_set ∧ c ∈ my_set ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b + c = -9 :=
sorry

end smallest_sum_of_three_l130_130893


namespace parabola_directrix_equation_l130_130280

theorem parabola_directrix_equation :
  ∀ (x y : ℝ),
  y = -4 * x^2 - 16 * x + 1 →
  ∃ d : ℝ, d = 273 / 16 ∧ y = d :=
by
  sorry

end parabola_directrix_equation_l130_130280


namespace symmetric_point_with_respect_to_x_axis_l130_130342

-- Definition of point M
def point_M : ℝ × ℝ := (3, -4)

-- Define the symmetry condition with respect to the x-axis
def symmetric_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

-- Statement that the symmetric point to point M with respect to the x-axis is (3, 4)
theorem symmetric_point_with_respect_to_x_axis : symmetric_x point_M = (3, 4) :=
by
  -- This is the statement of the theorem; the proof will be added here.
  sorry

end symmetric_point_with_respect_to_x_axis_l130_130342


namespace measure_of_angle_F_l130_130539

theorem measure_of_angle_F {D E F : ℝ}
  (isosceles : D = E)
  (angle_F_condition : F = D + 40)
  (sum_of_angles : D + E + F = 180) :
  F = 260 / 3 :=
by
  sorry

end measure_of_angle_F_l130_130539


namespace plum_balances_pear_l130_130202

variable (A G S : ℕ)

-- Definitions as per the problem conditions
axiom condition1 : 3 * A + G = 10 * S
axiom condition2 : A + 6 * S = G

-- The goal is to prove the following statement
theorem plum_balances_pear : G = 7 * S :=
by
  -- Skipping the proof as only statement is needed
  sorry

end plum_balances_pear_l130_130202


namespace geom_series_first_term_l130_130690

theorem geom_series_first_term (a r : ℝ) 
  (h1 : a / (1 - r) = 30)
  (h2 : a^2 / (1 - r^2) = 120) : 
  a = 120 / 17 :=
by
  sorry

end geom_series_first_term_l130_130690


namespace length_AC_l130_130721
open Real

-- Define the conditions and required proof
theorem length_AC (AB DC AD : ℝ) (h1 : AB = 17) (h2 : DC = 25) (h3 : AD = 8) : 
  abs (sqrt ((AD + DC - AD)^2 + (DC - sqrt (AB^2 - AD^2))^2) - 33.6) < 0.1 := 
  by
  -- The proof is omitted for brevity
  sorry

end length_AC_l130_130721


namespace traditionalist_fraction_l130_130284

theorem traditionalist_fraction (T P : ℕ) 
  (h1 : ∀ prov : ℕ, prov < 6 → T = P / 9) 
  (h2 : P + 6 * T > 0) :
  6 * T / (P + 6 * T) = 2 / 5 := 
by
  sorry

end traditionalist_fraction_l130_130284


namespace translate_sin_eq_cos_l130_130485

theorem translate_sin_eq_cos (φ : ℝ) (hφ : 0 ≤ φ ∧ φ < 2 * Real.pi) :
  (∀ x, Real.cos (x - Real.pi / 6) = Real.sin (x + φ)) → φ = Real.pi / 3 :=
by
  sorry

end translate_sin_eq_cos_l130_130485


namespace plywood_cut_difference_l130_130347

theorem plywood_cut_difference :
  ∀ (length width : ℕ) (n : ℕ) (perimeter_greatest perimeter_least : ℕ),
    length = 8 ∧ width = 4 ∧ n = 4 ∧
    (∀ l w, (l = (length / 2) ∧ w = width) ∨ (l = length ∧ w = (width / 2)) → (perimeter_greatest = 2 * (l + w))) ∧
    (∀ l w, (l = (length / n) ∧ w = width) ∨ (l = length ∧ w = (width / n)) → (perimeter_least = 2 * (l + w))) →
    length = 8 ∧ width = 4 ∧ n = 4 ∧ perimeter_greatest = 18 ∧ perimeter_least = 12 →
    (perimeter_greatest - perimeter_least) = 6 :=
by
  intros length width n perimeter_greatest perimeter_least h1 h2
  sorry

end plywood_cut_difference_l130_130347


namespace total_votes_election_l130_130880

theorem total_votes_election (total_votes fiona_votes elena_votes devin_votes : ℝ) 
  (Fiona_fraction : fiona_votes = (4/15) * total_votes)
  (Elena_fiona : elena_votes = fiona_votes + 15)
  (Devin_elena : devin_votes = 2 * elena_votes)
  (total_eq : total_votes = fiona_votes + elena_votes + devin_votes) :
  total_votes = 675 := 
sorry

end total_votes_election_l130_130880


namespace value_range_f_at_4_l130_130765

noncomputable def f (x : ℝ) : ℝ := sorry

theorem value_range_f_at_4 (f : ℝ → ℝ)
  (h1 : 1 ≤ f (-1) ∧ f (-1) ≤ 2)
  (h2 : 1 ≤ f (1) ∧ f (1) ≤ 3)
  (h3 : 2 ≤ f (2) ∧ f (2) ≤ 4)
  (h4 : -1 ≤ f (3) ∧ f (3) ≤ 1) :
  -21.75 ≤ f 4 ∧ f 4 ≤ 1 :=
sorry

end value_range_f_at_4_l130_130765


namespace minimum_value_of_expression_l130_130133

theorem minimum_value_of_expression (x y : ℝ) (hx : x > 1) (hy : y > 1) :
    (x^4 / (y - 1)) + (y^4 / (x - 1)) ≥ 12 := 
sorry

end minimum_value_of_expression_l130_130133


namespace problem_l130_130167

theorem problem (a : ℕ → ℝ) (h0 : a 1 = 0) (h9 : a 9 = 0)
  (h2_8 : ∃ i, 2 ≤ i ∧ i ≤ 8 ∧ a i > 0) (h_nonneg : ∀ n, 1 ≤ n ∧ n ≤ 9 → a n ≥ 0) : 
  (∃ i, 2 ≤ i ∧ i ≤ 8 ∧ a (i-1) + a (i+1) < 2 * a i) ∧ (∃ i, 2 ≤ i ∧ i ≤ 8 ∧ a (i-1) + a (i+1) < 1.9 * a i) := 
sorry

end problem_l130_130167


namespace solve_for_x_l130_130933

theorem solve_for_x (x : ℝ) (h : 12 - 2 * x = 6) : x = 3 :=
sorry

end solve_for_x_l130_130933


namespace net_profit_calc_l130_130750

theorem net_profit_calc:
  ∃ (x y : ℕ), x + y = 25 ∧ 1700 * x + 1800 * y = 44000 ∧ 2400 * x + 2600 * y = 63000 := by
  sorry

end net_profit_calc_l130_130750


namespace price_after_two_reductions_l130_130447

-- Define the two reductions as given in the conditions
def first_day_reduction (P : ℝ) : ℝ := P * 0.88
def second_day_reduction (P : ℝ) : ℝ := first_day_reduction P * 0.9

-- Main theorem: Price on the second day is 79.2% of the original price
theorem price_after_two_reductions (P : ℝ) : second_day_reduction P = 0.792 * P :=
by
  sorry

end price_after_two_reductions_l130_130447


namespace negative_number_zero_exponent_l130_130958

theorem negative_number_zero_exponent (a : ℤ) (h : a ≠ 0) : a^0 = 1 :=
by sorry

end negative_number_zero_exponent_l130_130958


namespace price_difference_is_correct_l130_130273

-- Definitions from the problem conditions
def list_price : ℝ := 58.80
def tech_shop_discount : ℝ := 12.00
def value_mart_discount_rate : ℝ := 0.20

-- Calculating the sale prices from definitions
def tech_shop_sale_price : ℝ := list_price - tech_shop_discount
def value_mart_sale_price : ℝ := list_price * (1 - value_mart_discount_rate)

-- The proof problem statement
theorem price_difference_is_correct :
  value_mart_sale_price - tech_shop_sale_price = 0.24 :=
by
  sorry

end price_difference_is_correct_l130_130273


namespace _l130_130330

noncomputable def X := 0
noncomputable def Y := 1
noncomputable def Z := 2

noncomputable def angle_XYZ (X Y Z : ℝ) : ℝ := 90 -- Triangle XYZ where ∠X = 90°

noncomputable def length_YZ := 10 -- YZ = 10 units
noncomputable def length_XY := 6 -- XY = 6 units
noncomputable def length_XZ : ℝ := Real.sqrt (length_YZ^2 - length_XY^2) -- Pythagorean theorem to find XZ
noncomputable def cos_Z : ℝ := length_XZ / length_YZ -- cos Z = adjacent/hypotenuse

example : cos_Z = 0.8 :=
by {
  sorry
}

end _l130_130330


namespace quadratic_roots_equal_integral_l130_130888

theorem quadratic_roots_equal_integral (c : ℝ) (h : (6^2 - 4 * 3 * c) = 0) : 
  ∃ x : ℝ, (3 * x^2 - 6 * x + c = 0) ∧ (x = 1) := 
by sorry

end quadratic_roots_equal_integral_l130_130888


namespace rect_area_perimeter_l130_130520

def rect_perimeter (Length Width : ℕ) : ℕ :=
  2 * (Length + Width)

theorem rect_area_perimeter (Area Length : ℕ) (hArea : Area = 192) (hLength : Length = 24) :
  ∃ (Width Perimeter : ℕ), Width = Area / Length ∧ Perimeter = rect_perimeter Length Width ∧ Perimeter = 64 :=
by
  sorry

end rect_area_perimeter_l130_130520


namespace fraction_meaningful_l130_130089

theorem fraction_meaningful (a : ℝ) : (∃ b, b = 2 / (a + 1)) → a ≠ -1 :=
by
  sorry

end fraction_meaningful_l130_130089


namespace hitting_probability_l130_130659

theorem hitting_probability (P_miss : ℝ) (P_6 P_7 P_8 P_9 P_10 : ℝ) :
  P_miss = 0.2 →
  P_6 = 0.1 →
  P_7 = 0.2 →
  P_8 = 0.3 →
  P_9 = 0.15 →
  P_10 = 0.05 →
  1 - P_miss = 0.8 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end hitting_probability_l130_130659


namespace probability_divisible_by_25_is_zero_l130_130377

-- Definitions of spinner outcomes and the function to generate four-digit numbers
def is_valid_spinner_outcome (n : ℕ) : Prop := n = 1 ∨ n = 2 ∨ n = 3

def generate_four_digit_number (spin1 spin2 spin3 spin4 : ℕ) : ℕ :=
  spin1 * 1000 + spin2 * 100 + spin3 * 10 + spin4

-- Condition stating that all outcomes of each spin are equally probable among {1, 2, 3}
def valid_outcome_condition (spin1 spin2 spin3 spin4 : ℕ) : Prop :=
  is_valid_spinner_outcome spin1 ∧ is_valid_spinner_outcome spin2 ∧
  is_valid_spinner_outcome spin3 ∧ is_valid_spinner_outcome spin4

-- Probability condition for the number being divisible by 25
def is_divisible_by_25 (n : ℕ) : Prop := n % 25 = 0

-- Main theorem: proving the probability is 0
theorem probability_divisible_by_25_is_zero :
  ∀ spin1 spin2 spin3 spin4,
    valid_outcome_condition spin1 spin2 spin3 spin4 →
    ¬ is_divisible_by_25 (generate_four_digit_number spin1 spin2 spin3 spin4) :=
by
  intros spin1 spin2 spin3 spin4 h
  -- Sorry for the proof details
  sorry

end probability_divisible_by_25_is_zero_l130_130377


namespace PP1_length_l130_130831

open Real

theorem PP1_length (AB AC : ℝ) (h₁ : AB = 5) (h₂ : AC = 3)
  (h₃ : ∃ γ : ℝ, γ = 90)  -- a right angle at A
  (BC : ℝ) (h₄ : BC = sqrt (AB^2 - AC^2))
  (A1B : ℝ) (A1C : ℝ) (h₅ : BC = A1B + A1C)
  (h₆ : A1B / A1C = AB / AC)
  (PQ : ℝ) (h₇ : PQ = A1B)
  (PR : ℝ) (h₈ : PR = A1C)
  (PP1 : ℝ) :
  PP1 = (3 * sqrt 5) / 4 :=
sorry

end PP1_length_l130_130831


namespace angle_C_triangle_area_l130_130658

theorem angle_C 
  (a b c : ℝ) (A B C : ℝ)
  (h1 : a * Real.cos B + b * Real.cos A = -2 * c * Real.cos C) :
  C = 2 * Real.pi / 3 :=
sorry

theorem triangle_area 
  (a b c : ℝ) (C : ℝ)
  (h1 : a * Real.cos B + b * Real.cos A = -2 * c * Real.cos C)
  (h2 : c = Real.sqrt 7)
  (h3 : b = 2) :
  1 / 2 * a * b * Real.sin C = Real.sqrt 3 / 2 :=
sorry

end angle_C_triangle_area_l130_130658


namespace rowing_upstream_speed_l130_130071

theorem rowing_upstream_speed 
  (V_m : ℝ) (V_downstream : ℝ) (V_upstream : ℝ)
  (hyp1 : V_m = 30)
  (hyp2 : V_downstream = 35) :
  V_upstream = V_m - (V_downstream - V_m) := 
  sorry

end rowing_upstream_speed_l130_130071


namespace reduce_to_one_piece_l130_130204

-- Definitions representing the conditions:
def plane_divided_into_unit_triangles : Prop := sorry
def initial_configuration (n : ℕ) : Prop := sorry
def possible_moves : Prop := sorry

-- Main theorem statement:
theorem reduce_to_one_piece (n : ℕ) 
  (H1 : plane_divided_into_unit_triangles) 
  (H2 : initial_configuration n) 
  (H3 : possible_moves) : 
  ∃ k : ℕ, k * 3 = n :=
sorry

end reduce_to_one_piece_l130_130204


namespace smallest_n_area_gt_2500_l130_130250

noncomputable def triangle_area (n : ℕ) : ℝ :=
  (1/2 : ℝ) * (|(n : ℝ) * (2 * n) + (n^2 - 1 : ℝ) * (3 * n^2 - 1) + (n^3 - 3 * n) * 1
  - (1 : ℝ) * (n^2 - 1) - (2 * n) * (n^3 - 3 * n) - (3 * n^2 - 1) * (n : ℝ)|)

theorem smallest_n_area_gt_2500 : ∃ n : ℕ, (∀ m : ℕ, 0 < m ∧ m < n → triangle_area m <= 2500) ∧ triangle_area n > 2500 :=
by
  sorry

end smallest_n_area_gt_2500_l130_130250


namespace kris_age_l130_130856

theorem kris_age (kris_age herbert_age : ℕ) (h1 : herbert_age + 1 = 15) (h2 : herbert_age + 10 = kris_age) : kris_age = 24 :=
by
  sorry

end kris_age_l130_130856


namespace bronson_cost_per_bushel_is_12_l130_130736

noncomputable def cost_per_bushel 
  (sale_price_per_apple : ℝ := 0.40)
  (apples_per_bushel : ℕ := 48)
  (profit_from_100_apples : ℝ := 15)
  (number_of_apples_sold : ℕ := 100) 
  : ℝ :=
  let revenue := number_of_apples_sold * sale_price_per_apple
  let cost := revenue - profit_from_100_apples
  let number_of_bushels := (number_of_apples_sold : ℝ) / apples_per_bushel
  cost / number_of_bushels

theorem bronson_cost_per_bushel_is_12 :
  cost_per_bushel = 12 :=
by
  sorry

end bronson_cost_per_bushel_is_12_l130_130736


namespace wrapping_paper_l130_130343

theorem wrapping_paper (total_used_per_roll : ℚ) (number_of_presents : ℕ) (fraction_used : ℚ) (fraction_left : ℚ) 
  (h1 : total_used_per_roll = 2 / 5) 
  (h2 : number_of_presents = 5) 
  (h3 : fraction_used = total_used_per_roll / number_of_presents) 
  (h4 : fraction_left = 1 - total_used_per_roll) : 
  fraction_used = 2 / 25 ∧ fraction_left = 3 / 5 := 
by 
  sorry

end wrapping_paper_l130_130343


namespace quadrilateral_angle_cosine_proof_l130_130163

variable (AB BC CD AD : ℝ)
variable (ϕ B C : ℝ)

theorem quadrilateral_angle_cosine_proof :
  AD^2 = AB^2 + BC^2 + CD^2 - 2 * (AB * BC * Real.cos B + BC * CD * Real.cos C + CD * AB * Real.cos ϕ) :=
by
  sorry

end quadrilateral_angle_cosine_proof_l130_130163


namespace isosceles_triangle_perimeter_l130_130598

theorem isosceles_triangle_perimeter (a b : ℝ)
  (h1 : b = 7)
  (h2 : a^2 - 8 * a + 15 = 0)
  (h3 : a * 2 > b)
  : 2 * a + b = 17 :=
by
  sorry

end isosceles_triangle_perimeter_l130_130598


namespace rhombus_side_length_l130_130862

-- Define the rhombus properties and the problem conditions
variables (p q x : ℝ)

-- State the problem as a theorem in Lean 4
theorem rhombus_side_length (h : x^2 = p * q) : x = Real.sqrt (p * q) :=
sorry

end rhombus_side_length_l130_130862


namespace part1_part2_l130_130197

def A (x : ℝ) : Prop := x^2 + 2*x - 3 < 0
def B (x : ℝ) (a : ℝ) : Prop := abs (x + a) < 1

theorem part1 (a : ℝ) (h : a = 3) : (∃ x : ℝ, (A x ∨ B x a)) ↔ (∃ x : ℝ, -4 < x ∧ x < 1) :=
by {
  sorry
}

theorem part2 : (∀ x : ℝ, B x a → A x) ∧ (¬ ∀ x : ℝ, A x → B x a) ↔ 0 ≤ a ∧ a ≤ 2 :=
by {
  sorry
}

end part1_part2_l130_130197


namespace min_value_of_sum_eq_l130_130977

theorem min_value_of_sum_eq : ∀ (a b : ℝ), a > 0 ∧ b > 0 ∧ 2 * a + b = a * b - 1 → a + 2 * b = 5 + 2 * Real.sqrt 6 :=
by
  intros a b h
  sorry

end min_value_of_sum_eq_l130_130977


namespace median_of_first_15_integers_l130_130074

theorem median_of_first_15_integers :
  150 * (8 / 100 : ℝ) = 12.0 :=
by
  sorry

end median_of_first_15_integers_l130_130074


namespace correct_equation_l130_130513

-- Define the conditions
variables {x : ℝ}

-- Condition 1: The unit price of a notebook is 2 yuan less than that of a water-based pen.
def notebook_price (water_pen_price : ℝ) : ℝ := water_pen_price - 2

-- Condition 2: Xiaogang bought 5 notebooks and 3 water-based pens for exactly 14 yuan.
def total_cost (notebook_price water_pen_price : ℝ) : ℝ :=
  5 * notebook_price + 3 * water_pen_price

-- Question restated as a theorem: Verify the given equation is correct
theorem correct_equation (water_pen_price : ℝ) (h : total_cost (notebook_price water_pen_price) water_pen_price = 14) :
  5 * (water_pen_price - 2) + 3 * water_pen_price = 14 :=
  by
    -- Introduce the assumption
    intros
    -- Sorry to skip the proof
    sorry

end correct_equation_l130_130513


namespace max_gcd_of_13n_plus_3_and_7n_plus_1_l130_130413

theorem max_gcd_of_13n_plus_3_and_7n_plus_1 (n : ℕ) (hn : 0 < n) :
  ∃ d, d = Nat.gcd (13 * n + 3) (7 * n + 1) ∧ ∀ m, m = Nat.gcd (13 * n + 3) (7 * n + 1) → m ≤ 8 := 
sorry

end max_gcd_of_13n_plus_3_and_7n_plus_1_l130_130413


namespace max_n_value_l130_130869

theorem max_n_value (a b c : ℝ) (n : ℕ) (h1 : a > b) (h2 : b > c) (h_ineq : 1/(a - b) + 1/(b - c) ≥ n/(a - c)) : n ≤ 4 := 
sorry

end max_n_value_l130_130869


namespace find_x_l130_130540

theorem find_x (x : ℕ) (h : 220030 = (x + 445) * (2 * (x - 445)) + 30) : x = 555 := 
sorry

end find_x_l130_130540


namespace solve_equation_l130_130693

theorem solve_equation : ∀ x : ℝ, 2 * x - 6 = 3 * x * (x - 3) ↔ (x = 3 ∨ x = 2 / 3) := by sorry

end solve_equation_l130_130693


namespace find_values_and_properties_l130_130476

variable (f : ℝ → ℝ)

axiom f_neg1 : f (-1) = 2
axiom f_pos_x : ∀ x, x < 0 → f x > 1
axiom f_add : ∀ x y : ℝ, f (x + y) = f x * f y

theorem find_values_and_properties :
  f 0 = 1 ∧
  f (-4) = 16 ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂) ∧
  (∀ x : ℝ, f (-4 * x^2) * f (10 * x) ≥ 1/16 ↔ x ≤ 1/2 ∨ x ≥ 2) :=
sorry

end find_values_and_properties_l130_130476


namespace students_in_both_clubs_l130_130211

theorem students_in_both_clubs :
  ∀ (total_students drama_club science_club either_club both_club : ℕ),
  total_students = 300 →
  drama_club = 100 →
  science_club = 140 →
  either_club = 220 →
  (drama_club + science_club - both_club = either_club) →
  both_club = 20 :=
by
  intros total_students drama_club science_club either_club both_club
  intros h1 h2 h3 h4 h5
  sorry

end students_in_both_clubs_l130_130211


namespace words_per_page_eq_106_l130_130268

-- Definition of conditions as per the problem statement
def pages : ℕ := 224
def max_words_per_page : ℕ := 150
def total_words_congruence : ℕ := 156
def modulus : ℕ := 253

theorem words_per_page_eq_106 (p : ℕ) : 
  (224 * p % 253 = 156) ∧ (p ≤ 150) → p = 106 :=
by 
  sorry

end words_per_page_eq_106_l130_130268


namespace sum_of_consecutive_odds_eq_power_l130_130505

theorem sum_of_consecutive_odds_eq_power (n : ℕ) (k : ℕ) (hn : n > 0) (hk : k ≥ 2) :
  ∃ a : ℤ, n * (2 * a + n) = n^k ∧
            (∀ i : ℕ, i < n → 2 * a + 2 * (i : ℤ) + 1 = 2 * a + 1 + 2 * i) :=
by
  sorry

end sum_of_consecutive_odds_eq_power_l130_130505


namespace distance_P_to_y_axis_l130_130106

-- Define the Point structure
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Condition: Point P with coordinates (-3, 5)
def P : Point := ⟨-3, 5⟩

-- Definition of distance from a point to the y-axis
def distance_to_y_axis (p : Point) : ℝ :=
  abs p.x

-- Proof problem statement
theorem distance_P_to_y_axis : distance_to_y_axis P = 3 := 
  sorry

end distance_P_to_y_axis_l130_130106


namespace total_kids_l130_130790

theorem total_kids (girls boys: ℕ) (h1: girls = 3) (h2: boys = 6) : girls + boys = 9 :=
by
  sorry

end total_kids_l130_130790


namespace tennis_tournament_matches_l130_130491

theorem tennis_tournament_matches (n : ℕ) (h₁ : n = 128) (h₂ : ∃ m : ℕ, m = 32) (h₃ : ∃ k : ℕ, k = 96) (h₄ : ∀ i : ℕ, i > 1 → i ≤ n → ∃ j : ℕ, j = 1 + (i - 1)) :
  ∃ total_matches : ℕ, total_matches = 127 := 
by 
  sorry

end tennis_tournament_matches_l130_130491


namespace max_value_l130_130912

-- Definitions for the given conditions
def point_A := (3, 1)
def line_equation (m n : ℝ) := 3 * m + n + 1 = 0
def positive_product (m n : ℝ) := m * n > 0

-- The main statement to be proved
theorem max_value (m n : ℝ) (h1 : line_equation m n) (h2 : positive_product m n) : 
  (3 / m + 1 / n) ≤ -16 :=
sorry

end max_value_l130_130912


namespace smallest_positive_e_l130_130541

-- Define the polynomial and roots condition
def polynomial (a b c d e : ℤ) (x : ℝ) : ℝ :=
  a * x^4 + b * x^3 + c * x^2 + d * x + e

def has_integer_roots (p : ℝ → ℝ) (roots : List ℝ) : Prop :=
  ∀ r ∈ roots, p r = 0

def polynomial_with_given_roots (a b c d e : ℤ) : Prop :=
  has_integer_roots (polynomial a b c d e) [-3, 4, 11, -(1/4)]

-- Main theorem to prove the smallest positive integer e
theorem smallest_positive_e (a b c d : ℤ) :
  ∃ e : ℤ, e > 0 ∧ polynomial_with_given_roots a b c d e ∧
            (∀ e' : ℤ, e' > 0 ∧ polynomial_with_given_roots a b c d e' → e ≤ e') :=
  sorry

end smallest_positive_e_l130_130541


namespace price_of_other_stamp_l130_130208

-- Define the conditions
def total_stamps : ℕ := 75
def total_value_cents : ℕ := 480
def known_stamp_price : ℕ := 8
def known_stamp_count : ℕ := 40
def unknown_stamp_count : ℕ := total_stamps - known_stamp_count

-- The problem to solve
theorem price_of_other_stamp (x : ℕ) :
  (known_stamp_count * known_stamp_price) + (unknown_stamp_count * x) = total_value_cents → x = 5 :=
by
  sorry

end price_of_other_stamp_l130_130208


namespace easter_eggs_problem_l130_130019

noncomputable def mia_rate : ℕ := 24
noncomputable def billy_rate : ℕ := 10
noncomputable def total_hours : ℕ := 5
noncomputable def total_eggs : ℕ := 170

theorem easter_eggs_problem :
  (mia_rate + billy_rate) * total_hours = total_eggs :=
by
  sorry

end easter_eggs_problem_l130_130019


namespace trapezoid_other_base_possible_lengths_l130_130435

-- Definition of the trapezoid problem.
structure Trapezoid where
  height : ℕ
  leg1 : ℕ
  leg2 : ℕ
  base1 : ℕ

-- The given conditions
def trapezoid_data : Trapezoid :=
{ height := 12, leg1 := 20, leg2 := 15, base1 := 42 }

-- The proof problem in Lean 4 statement
theorem trapezoid_other_base_possible_lengths (t : Trapezoid) :
  t = trapezoid_data → (∃ b : ℕ, (b = 17 ∨ b = 35)) :=
by
  intro h_data_eq
  sorry

end trapezoid_other_base_possible_lengths_l130_130435


namespace find_rectangle_dimensions_area_30_no_rectangle_dimensions_area_32_l130_130406

noncomputable def length_width_rectangle_area_30 : Prop :=
∃ (x y : ℝ), x * y = 30 ∧ 2 * (x + y) = 22 ∧ x = 6 ∧ y = 5

noncomputable def impossible_rectangle_area_32 : Prop :=
¬(∃ (x y : ℝ), x * y = 32 ∧ 2 * (x + y) = 22)

-- Proof statements (without proofs)
theorem find_rectangle_dimensions_area_30 : length_width_rectangle_area_30 :=
sorry

theorem no_rectangle_dimensions_area_32 : impossible_rectangle_area_32 :=
sorry

end find_rectangle_dimensions_area_30_no_rectangle_dimensions_area_32_l130_130406


namespace number_of_jars_good_for_sale_l130_130013

def numberOfGoodJars (initialCartons : Nat) (cartonsNotDelivered : Nat) (jarsPerCarton : Nat)
  (damagedJarsPerCarton : Nat) (numberOfDamagedCartons : Nat) (oneTotallyDamagedCarton : Nat) : Nat := 
  let deliveredCartons := initialCartons - cartonsNotDelivered
  let totalJars := deliveredCartons * jarsPerCarton
  let damagedJars := (damagedJarsPerCarton * numberOfDamagedCartons) + oneTotallyDamagedCarton
  totalJars - damagedJars

theorem number_of_jars_good_for_sale : 
  numberOfGoodJars 50 20 20 3 5 20 = 565 :=
by
  sorry

end number_of_jars_good_for_sale_l130_130013


namespace find_angle_A_find_cos2C_minus_pi_over_6_l130_130908

noncomputable def triangle_area_formula (a b c : ℝ) (C : ℝ) : ℝ :=
  (1 / 2) * a * b * Real.sin C

noncomputable def given_area_formula (b c : ℝ) (S : ℝ) (a : ℝ) (C : ℝ) : Prop :=
  S = (Real.sqrt 3 / 6) * b * (b + c - a * Real.cos C)

noncomputable def angle_A (S b c a C : ℝ) (h : given_area_formula b c S a C) : ℝ :=
  Real.arcsin ((Real.sqrt 3 / 3) * (b + c - a * Real.cos C))

theorem find_angle_A (a b c S C : ℝ) (h : given_area_formula b c S a C) :
  angle_A S b c a C h = π / 3 :=
sorry

-- Part 2 related definitions
noncomputable def cos2C_minus_pi_over_6 (b c a C : ℝ) : ℝ :=
  let cos_C := (b^2 + c^2 - a^2) / (2 * b * c)
  let sin_C := Real.sqrt (1 - cos_C^2)
  let cos_2C := 2 * cos_C^2 - 1
  let sin_2C := 2 * sin_C * cos_C
  cos_2C * (Real.sqrt 3 / 2) + sin_2C * (1 / 2)

theorem find_cos2C_minus_pi_over_6 (b c a C : ℝ) (hb : b = 1) (hc : c = 3) (ha : a = Real.sqrt 7) :
  cos2C_minus_pi_over_6 b c a C = - (4 * Real.sqrt 3 / 7) :=
sorry

end find_angle_A_find_cos2C_minus_pi_over_6_l130_130908


namespace hcf_of_36_and_x_is_12_l130_130440

theorem hcf_of_36_and_x_is_12 (x : ℕ) (h : Nat.gcd 36 x = 12) : x = 48 :=
sorry

end hcf_of_36_and_x_is_12_l130_130440


namespace polynomial_degree_l130_130545

noncomputable def polynomial1 : Polynomial ℤ := 3 * Polynomial.monomial 5 1 + 2 * Polynomial.monomial 4 1 - Polynomial.monomial 1 1 + Polynomial.C 5
noncomputable def polynomial2 : Polynomial ℤ := 4 * Polynomial.monomial 11 1 - 2 * Polynomial.monomial 8 1 + 5 * Polynomial.monomial 5 1 - Polynomial.C 9
noncomputable def polynomial3 : Polynomial ℤ := (Polynomial.monomial 2 1 - Polynomial.C 3) ^ 9

theorem polynomial_degree :
  (polynomial1 * polynomial2 - polynomial3).degree = 18 := by
  sorry

end polynomial_degree_l130_130545


namespace sum_of_two_squares_l130_130992

theorem sum_of_two_squares (n : ℕ) (k m : ℤ) : 2 * n = k^2 + m^2 → ∃ a b : ℤ, n = a^2 + b^2 := 
by
  sorry

end sum_of_two_squares_l130_130992


namespace prime_range_for_integer_roots_l130_130473

theorem prime_range_for_integer_roots (p : ℕ) (h_prime : Prime p) 
  (h_int_roots : ∃ (a b : ℤ), a + b = -p ∧ a * b = -300 * p) : 
  1 < p ∧ p ≤ 11 :=
sorry

end prime_range_for_integer_roots_l130_130473


namespace sum_consecutive_even_integers_l130_130111

theorem sum_consecutive_even_integers (n : ℕ) (h : 2 * n + 4 = 156) : 
  n + (n + 2) + (n + 4) = 234 := 
by
  sorry

end sum_consecutive_even_integers_l130_130111


namespace perimeter_proof_l130_130663

noncomputable def perimeter (x : ℝ) : ℝ :=
  if x ≥ 0 ∧ x ≤ (Real.sqrt 3) / 3 then 3 * Real.sqrt 6 * x
  else if x > (Real.sqrt 3) / 3 ∧ x ≤ (2 * Real.sqrt 3) / 3 then 3 * Real.sqrt 2
  else if x > (2 * Real.sqrt 3) / 3 ∧ x ≤ Real.sqrt 3 then 3 * Real.sqrt 6 * (Real.sqrt 3 - x)
  else 0

theorem perimeter_proof (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x ≤ Real.sqrt 3) :
  perimeter x = 
    if x ≤ (Real.sqrt 3) / 3 then 3 * Real.sqrt 6 * x
    else if x ≤ (2 * Real.sqrt 3) / 3 then 3 * Real.sqrt 2
    else 3 * Real.sqrt 6 * (Real.sqrt 3 - x) :=
by 
  sorry

end perimeter_proof_l130_130663


namespace wine_consumption_correct_l130_130454

-- Definitions based on conditions
def drank_after_first_pound : ℚ := 1
def drank_after_second_pound : ℚ := 1
def drank_after_third_pound : ℚ := 1 / 2
def drank_after_fourth_pound : ℚ := 1 / 4
def drank_after_fifth_pound : ℚ := 1 / 8
def drank_after_sixth_pound : ℚ := 1 / 16

-- Total wine consumption
def total_wine_consumption : ℚ :=
  drank_after_first_pound + drank_after_second_pound +
  drank_after_third_pound + drank_after_fourth_pound +
  drank_after_fifth_pound + drank_after_sixth_pound

-- Theorem statement
theorem wine_consumption_correct :
  total_wine_consumption = 47 / 16 :=
by
  sorry

end wine_consumption_correct_l130_130454


namespace minimum_value_l130_130562

noncomputable def condition (x : ℝ) : Prop := (2 * x - 1) / 3 - 1 ≥ x - (5 - 3 * x) / 2

noncomputable def target_function (x : ℝ) : ℝ := abs (x - 1) - abs (x + 3)

theorem minimum_value :
  ∃ x : ℝ, condition x ∧ ∀ y : ℝ, condition y → target_function y ≥ target_function x :=
sorry

end minimum_value_l130_130562


namespace eleven_power_2023_mod_50_l130_130245

theorem eleven_power_2023_mod_50 :
  11^2023 % 50 = 31 :=
by
  sorry

end eleven_power_2023_mod_50_l130_130245


namespace cost_of_child_ticket_is_4_l130_130939

def cost_of_child_ticket (cost_adult cost_total tickets_sold tickets_child receipts_total : ℕ) : ℕ :=
  let tickets_adult := tickets_sold - tickets_child
  let receipts_adult := tickets_adult * cost_adult
  let receipts_child := receipts_total - receipts_adult
  receipts_child / tickets_child

theorem cost_of_child_ticket_is_4 (cost_adult : ℕ) (cost_total : ℕ)
  (tickets_sold : ℕ) (tickets_child : ℕ) (receipts_total : ℕ) :
  cost_of_child_ticket 12 4 130 90 840 = 4 := by
  sorry

end cost_of_child_ticket_is_4_l130_130939


namespace dogwood_trees_l130_130067

/-- There are 7 dogwood trees currently in the park. 
Park workers will plant 5 dogwood trees today. 
The park will have 16 dogwood trees when the workers are finished.
Prove that 4 dogwood trees will be planted tomorrow. --/
theorem dogwood_trees (x : ℕ) : 7 + 5 + x = 16 → x = 4 :=
by
  sorry

end dogwood_trees_l130_130067


namespace sale_in_second_month_l130_130890

def sale_first_month : ℕ := 6435
def sale_third_month : ℕ := 6855
def sale_fourth_month : ℕ := 7230
def sale_fifth_month : ℕ := 6562
def sale_sixth_month : ℕ := 6191
def average_sale : ℕ := 6700

theorem sale_in_second_month : 
  ∀ (sale_second_month : ℕ), 
    (sale_first_month + sale_second_month + sale_third_month + sale_fourth_month + sale_fifth_month + sale_sixth_month = 6700 * 6) → 
    sale_second_month = 6927 :=
by
  intro sale_second_month h
  sorry

end sale_in_second_month_l130_130890


namespace sequence_remainder_4_l130_130953

def sequence_of_numbers (n : ℕ) : ℕ :=
  7 * n + 4

theorem sequence_remainder_4 (n : ℕ) : (sequence_of_numbers n) % 7 = 4 := by
  sorry

end sequence_remainder_4_l130_130953


namespace find_number_l130_130213

theorem find_number (x : ℝ) (h : 140 = 3.5 * x) : x = 40 :=
by
  sorry

end find_number_l130_130213


namespace find_a1_range_a1_l130_130323

variables (a_1 : ℤ) (d : ℤ := -1) (S : ℕ → ℤ)

-- Definition of sum of first n terms in an arithmetic sequence
def arithmetic_sum (n : ℕ) : ℤ := n * (2 * a_1 + (n - 1) * d) / 2

-- Definition of nth term in an arithmetic sequence
def arithmetic_nth_term (n : ℕ) : ℤ := a_1 + (n - 1) * d

-- Given conditions for the problems
axiom S_def : ∀ n, S n = arithmetic_sum a_1 d n

-- Problem 1: Proving a1 = 1 given S_5 = -5
theorem find_a1 (h : S 5 = -5) : a_1 = 1 :=
by
  sorry

-- Problem 2: Proving range of a1 given S_n ≤ a_n for any positive integer n
theorem range_a1 (h : ∀ n : ℕ, n > 0 → S n ≤ arithmetic_nth_term a_1 d n) : a_1 ≤ 0 :=
by
  sorry

end find_a1_range_a1_l130_130323


namespace average_age_condition_l130_130145

theorem average_age_condition (n : ℕ) 
  (h1 : (↑n * 14) / n = 14) 
  (h2 : ((↑n * 14) + 34) / (n + 1) = 16) : 
  n = 9 := 
by 
-- Proof goes here
sorry

end average_age_condition_l130_130145


namespace tank_A_is_60_percent_of_tank_B_capacity_l130_130140

-- Conditions
def height_A : ℝ := 10
def circumference_A : ℝ := 6
def height_B : ℝ := 6
def circumference_B : ℝ := 10

-- Statement
theorem tank_A_is_60_percent_of_tank_B_capacity (V_A V_B : ℝ) (radius_A radius_B : ℝ)
  (hA : radius_A = circumference_A / (2 * Real.pi))
  (hB : radius_B = circumference_B / (2 * Real.pi))
  (vol_A : V_A = Real.pi * radius_A^2 * height_A)
  (vol_B : V_B = Real.pi * radius_B^2 * height_B) :
  (V_A / V_B) * 100 = 60 :=
by
  sorry

end tank_A_is_60_percent_of_tank_B_capacity_l130_130140


namespace B_is_345_complement_U_A_inter_B_is_3_l130_130961

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}

-- Define set A
def A : Set ℕ := {2, 4, 5}

-- Define set B as given in the conditions
def B : Set ℕ := {x ∈ U | 2 < x ∧ x < 6}

-- Prove that B is {3, 4, 5}
theorem B_is_345 : B = {3, 4, 5} := by
  sorry

-- Define the complement of A with respect to U
def complement_U_A : Set ℕ := U \ A

-- Prove the intersection of the complement of A and B is {3}
theorem complement_U_A_inter_B_is_3 : (complement_U_A ∩ B) = {3} := by
  sorry

end B_is_345_complement_U_A_inter_B_is_3_l130_130961


namespace min_value_of_f_l130_130302

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem min_value_of_f : ∃ x : ℝ, (f x = -(1 / Real.exp 1)) ∧ (∀ y : ℝ, f y ≥ f x) := by
  sorry

end min_value_of_f_l130_130302


namespace cricket_bat_cost_l130_130467

noncomputable def CP_A_sol : ℝ := 444.96 / 1.95

theorem cricket_bat_cost (CP_A : ℝ) (SP_B : ℝ) (SP_C : ℝ) (SP_D : ℝ) :
  (SP_B = 1.20 * CP_A) →
  (SP_C = 1.25 * SP_B) →
  (SP_D = 1.30 * SP_C) →
  (SP_D = 444.96) →
  CP_A = CP_A_sol :=
by
  intros h1 h2 h3 h4
  sorry

end cricket_bat_cost_l130_130467


namespace pasture_rent_share_l130_130551

theorem pasture_rent_share (x : ℕ) (H1 : (45 / (10 * x + 60 + 45)) * 245 = 63) : 
  x = 7 :=
by {
  sorry
}

end pasture_rent_share_l130_130551


namespace pure_imaginary_number_l130_130560

open Complex -- Use the Complex module for complex numbers

theorem pure_imaginary_number (a : ℝ) (h : (a - 1 : ℂ).re = 0) : a = 1 :=
by
  -- This part of the proof is omitted hence we put sorry
  sorry

end pure_imaginary_number_l130_130560


namespace red_knights_fraction_magic_l130_130614

theorem red_knights_fraction_magic (total_knights red_knights blue_knights magical_knights : ℕ)
  (h1 : red_knights = (3 / 8 : ℚ) * total_knights)
  (h2 : blue_knights = total_knights - red_knights)
  (h3 : magical_knights = (1 / 4 : ℚ) * total_knights)
  (fraction_red_magic fraction_blue_magic : ℚ) 
  (h4 : fraction_red_magic = 3 * fraction_blue_magic)
  (h5 : magical_knights = red_knights * fraction_red_magic + blue_knights * fraction_blue_magic) :
  fraction_red_magic = 3 / 7 := 
by
  sorry

end red_knights_fraction_magic_l130_130614


namespace solution_l130_130681

noncomputable def problem (x : ℝ) (h : x ≠ 3) : ℝ :=
  (3 * x / (x - 3)) + ((x + 6) / (3 - x))

theorem solution (x : ℝ) (h : x ≠ 3) : problem x h = 2 :=
by
  sorry

end solution_l130_130681


namespace yarn_total_length_l130_130828

/-- The green yarn is 156 cm long, the red yarn is 8 cm more than three times the green yarn,
    prove that the total length of the two pieces of yarn is 632 cm. --/
theorem yarn_total_length : 
  let green_yarn := 156
  let red_yarn := 3 * green_yarn + 8
  green_yarn + red_yarn = 632 :=
by
  let green_yarn := 156
  let red_yarn := 3 * green_yarn + 8
  sorry

end yarn_total_length_l130_130828


namespace sufficient_but_not_necessary_l130_130709

theorem sufficient_but_not_necessary (x : ℝ) :
  (x > 1 → x > 0) ∧ ¬ (x > 0 → x > 1) :=
by
  sorry

end sufficient_but_not_necessary_l130_130709


namespace number_of_routes_l130_130295

variable {City : Type}
variable (A B C D E : City)
variable (AB_N AB_S AD AE BC BD CD DE : City → City → Prop)
  
theorem number_of_routes 
  (hAB_N : AB_N A B) (hAB_S : AB_S A B)
  (hAD : AD A D) (hAE : AE A E)
  (hBC : BC B C) (hBD : BD B D)
  (hCD : CD C D) (hDE : DE D E) :
  ∃ r : ℕ, r = 16 := 
sorry

end number_of_routes_l130_130295


namespace tomatoes_eaten_l130_130463

theorem tomatoes_eaten 
  (initial_tomatoes : ℕ) 
  (final_tomatoes : ℕ) 
  (half_given : ℕ) 
  (B : ℕ) 
  (h_initial : initial_tomatoes = 127) 
  (h_final : final_tomatoes = 54) 
  (h_half : half_given = final_tomatoes * 2) 
  (h_remaining : initial_tomatoes - half_given = B)
  : B = 19 := 
by
  sorry

end tomatoes_eaten_l130_130463


namespace alpha_and_2beta_l130_130144

theorem alpha_and_2beta (α β : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2) 
  (h_tan_alpha : Real.tan α = 1 / 8) (h_sin_beta : Real.sin β = 1 / 3) :
  α + 2 * β = Real.arctan (15 / 56) := by
  sorry

end alpha_and_2beta_l130_130144


namespace prism_volume_l130_130398

theorem prism_volume
  (l w h : ℝ)
  (h1 : l * w = 6.5)
  (h2 : w * h = 8)
  (h3 : l * h = 13) :
  l * w * h = 26 :=
by
  sorry

end prism_volume_l130_130398


namespace find_number_l130_130705

theorem find_number (N M : ℕ) 
  (h1 : N + M = 3333) (h2 : N - M = 693) :
  N = 2013 :=
sorry

end find_number_l130_130705


namespace radius_of_cookie_l130_130569

theorem radius_of_cookie : 
  ∀ x y : ℝ, (x^2 + y^2 - 6.5 = x + 3 * y) → 
  ∃ (c : ℝ × ℝ) (r : ℝ), r = 3 ∧ (x - c.1)^2 + (y - c.2)^2 = r^2 :=
by {
  sorry
}

end radius_of_cookie_l130_130569


namespace pure_gala_trees_l130_130824

variable (T F G : ℝ)

theorem pure_gala_trees (h1 : F + 0.1 * T = 170) (h2 : F = 0.75 * T): G = T - F -> G = 50 :=
by
  sorry

end pure_gala_trees_l130_130824


namespace exists_x_for_integer_conditions_l130_130315

-- Define the conditions as functions in Lean
def is_int_div (a b : Int) : Prop := ∃ k : Int, a = b * k

-- The target statement in Lean 4
theorem exists_x_for_integer_conditions :
  ∃ t_1 : Int, ∃ x : Int, (x = 105 * t_1 + 52) ∧ 
    (is_int_div (x - 3) 7) ∧ 
    (is_int_div (x - 2) 5) ∧ 
    (is_int_div (x - 4) 3) :=
by 
  sorry

end exists_x_for_integer_conditions_l130_130315


namespace geometric_sequence_a10_a11_l130_130718

noncomputable def a (n : ℕ) : ℝ := sorry  -- define the geometric sequence {a_n}

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n m, a (n + m) = a n * q^m

variables (a : ℕ → ℝ) (q : ℝ)

-- Conditions given in the problem
axiom h1 : a 1 + a 5 = 5
axiom h2 : a 4 + a 5 = 15
axiom geom_seq : is_geometric_sequence a q

theorem geometric_sequence_a10_a11 : a 10 + a 11 = 135 :=
by {
  sorry
}

end geometric_sequence_a10_a11_l130_130718


namespace prince_cd_total_spent_l130_130622

theorem prince_cd_total_spent (total_cds : ℕ)
    (pct_20 : ℕ) (pct_15 : ℕ) (pct_10 : ℕ)
    (bought_20_pct : ℕ) (bought_15_pct : ℕ)
    (bought_10_pct : ℕ) (bought_6_pct : ℕ)
    (discount_cnt_4 : ℕ) (discount_amount_4 : ℕ)
    (discount_cnt_5 : ℕ) (discount_amount_5 : ℕ)
    (total_cost_no_discount : ℕ) (total_discount : ℕ) (total_spent : ℕ) :
    total_cds = 400 ∧
    pct_20 = 25 ∧ pct_15 = 30 ∧ pct_10 = 20 ∧
    bought_20_pct = 70 ∧ bought_15_pct = 40 ∧
    bought_10_pct = 80 ∧ bought_6_pct = 100 ∧
    discount_cnt_4 = 4 ∧ discount_amount_4 = 5 ∧
    discount_cnt_5 = 5 ∧ discount_amount_5 = 3 ∧
    total_cost_no_discount - total_discount = total_spent ∧
    total_spent = 3119 := by
  sorry

end prince_cd_total_spent_l130_130622


namespace tim_prank_combinations_l130_130701

def number_of_combinations : Nat :=
  let monday_choices := 1
  let tuesday_choices := 2
  let wednesday_choices := 6
  let thursday_choices := 5
  let friday_choices := 1
  monday_choices * tuesday_choices * wednesday_choices * thursday_choices * friday_choices

theorem tim_prank_combinations : number_of_combinations = 60 :=
by
  sorry

end tim_prank_combinations_l130_130701


namespace green_square_area_percentage_l130_130612

noncomputable def flag_side_length (k: ℝ) : ℝ := k
noncomputable def cross_area_fraction : ℝ := 0.49
noncomputable def cross_area (k: ℝ) : ℝ := cross_area_fraction * k^2
noncomputable def cross_width (t: ℝ) : ℝ := t
noncomputable def green_square_side (x: ℝ) : ℝ := x
noncomputable def green_square_area (x: ℝ) : ℝ := x^2

theorem green_square_area_percentage (k: ℝ) (t: ℝ) (x: ℝ)
  (h1: x = 2 * t)
  (h2: 4 * t * (k - t) + x^2 = cross_area k)
  : green_square_area x / (k^2) * 100 = 6.01 :=
by
  sorry

end green_square_area_percentage_l130_130612


namespace interest_rate_for_4000_investment_l130_130823

theorem interest_rate_for_4000_investment
      (total_money : ℝ := 9000)
      (invested_at_9_percent : ℝ := 5000)
      (total_interest : ℝ := 770)
      (invested_at_unknown_rate : ℝ := 4000) :
  ∃ r : ℝ, invested_at_unknown_rate * r = total_interest - (invested_at_9_percent * 0.09) ∧ r = 0.08 :=
by {
  -- Proof is not required based on instruction, so we use sorry.
  sorry
}

end interest_rate_for_4000_investment_l130_130823


namespace abs_diff_squares_l130_130126

theorem abs_diff_squares (a b : ℤ) (h_a : a = 105) (h_b : b = 95):
  |a^2 - b^2| = 2000 := by
  sorry

end abs_diff_squares_l130_130126


namespace problem1_problem2_l130_130420

-- We define a point P(x, y) on the circle x^2 + y^2 = 2y.
variables {x y a : ℝ}

-- Condition for the point P to be on the circle
def on_circle (x y : ℝ) : Prop := x^2 + y^2 = 2 * y

-- Definition for 2x + y range
def range_2x_plus_y (x y : ℝ) : Prop := - Real.sqrt 5 + 1 ≤ 2 * x + y ∧ 2 * x + y ≤ Real.sqrt 5 + 1

-- Definition for the range of a given x + y + a ≥ 0
def range_a (x y a : ℝ) : Prop := x + y + a ≥ 0 → a ≥ Real.sqrt 2 - 1

-- Main statements to prove
theorem problem1 (hx : on_circle x y) : range_2x_plus_y x y := sorry

theorem problem2 (hx : on_circle x y) (h : ∀ θ, x = Real.cos θ ∧ y = 1 + Real.sin θ) : range_a x y a := sorry

end problem1_problem2_l130_130420


namespace units_digit_of_sum_is_4_l130_130712

-- Definitions and conditions based on problem
def base_8_add (a b : List Nat) : List Nat :=
    sorry -- Function to perform addition in base 8, returning result as a list of digits

def units_digit (a : List Nat) : Nat :=
    a.headD 0  -- Function to get the units digit of the result

-- The list representation for the digits of 65 base 8 and 37 base 8
def sixty_five_base8 := [6, 5]
def thirty_seven_base8 := [3, 7]

-- The theorem that asserts the final result
theorem units_digit_of_sum_is_4 : units_digit (base_8_add sixty_five_base8 thirty_seven_base8) = 4 :=
    sorry

end units_digit_of_sum_is_4_l130_130712


namespace train_crosses_tunnel_in_45_sec_l130_130484

/-- Given the length of the train, the length of the platform, the length of the tunnel, 
and the time taken to cross the platform, prove the time taken for the train to cross the tunnel is 45 seconds. -/
theorem train_crosses_tunnel_in_45_sec (l_train : ℕ) (l_platform : ℕ) (t_platform : ℕ) (l_tunnel : ℕ)
  (h_train_length : l_train = 330)
  (h_platform_length : l_platform = 180)
  (h_time_platform : t_platform = 15)
  (h_tunnel_length : l_tunnel = 1200) :
  (l_train + l_tunnel) / ((l_train + l_platform) / t_platform) = 45 :=
by
  -- placeholder for the actual proof
  sorry

end train_crosses_tunnel_in_45_sec_l130_130484


namespace shortest_distance_midpoint_parabola_chord_l130_130982

theorem shortest_distance_midpoint_parabola_chord
  (A B : ℝ × ℝ)
  (hA : A.1 ^ 2 = 4 * A.2)
  (hB : B.1 ^ 2 = 4 * B.2)
  (cord_length : dist A B = 6)
  : dist ((A.1 + B.1) / 2, (A.2 + B.2) / 2) (0, 0) = 2 :=
sorry

end shortest_distance_midpoint_parabola_chord_l130_130982


namespace determine_point_T_l130_130956

noncomputable def point : Type := ℝ × ℝ

def is_square (O P Q R : point) : Prop :=
  O.1 = 0 ∧ O.2 = 0 ∧
  Q.1 = 3 ∧ Q.2 = 3 ∧
  P.1 = 3 ∧ P.2 = 0 ∧
  R.1 = 0 ∧ R.2 = 3

def twice_area_square_eq_area_triangle (O P Q T : point) : Prop :=
  2 * (3 * 3) = abs ((P.1 * Q.2 + Q.1 * T.2 + T.1 * P.2 - P.2 * Q.1 - Q.2 * T.1 - T.2 * P.1) / 2)

theorem determine_point_T (O P Q R T : point) (h1 : is_square O P Q R) : 
  twice_area_square_eq_area_triangle O P Q T ↔ T = (3, 12) :=
sorry

end determine_point_T_l130_130956


namespace stratified_sampling_example_l130_130312

noncomputable def sample_proportion := 70 / 3500
noncomputable def total_students := 3500 + 1500
noncomputable def sample_size := total_students * sample_proportion

theorem stratified_sampling_example 
  (high_school_students : ℕ := 3500)
  (junior_high_students : ℕ := 1500)
  (sampled_high_school_students : ℕ := 70)
  (proportion_of_sampling : ℝ := sampled_high_school_students / high_school_students)
  (total_number_of_students : ℕ := high_school_students + junior_high_students)
  (calculated_sample_size : ℝ := total_number_of_students * proportion_of_sampling) :
  calculated_sample_size = 100 :=
by
  sorry

end stratified_sampling_example_l130_130312


namespace complete_square_eq_l130_130818

theorem complete_square_eq (x : ℝ) :
  x^2 - 8 * x + 15 = 0 →
  (x - 4)^2 = 1 :=
by sorry

end complete_square_eq_l130_130818


namespace find_probability_l130_130155

noncomputable def probability_distribution (X : ℕ → ℝ) := ∀ k, X k = 1 / (2^k)

theorem find_probability (X : ℕ → ℝ) (h : probability_distribution X) :
  X 3 + X 4 = 3 / 16 :=
by
  sorry

end find_probability_l130_130155


namespace range_of_m_l130_130360

noncomputable def proposition_p (x m : ℝ) := (x - m) ^ 2 > 3 * (x - m)
noncomputable def proposition_q (x : ℝ) := x ^ 2 + 3 * x - 4 < 0

theorem range_of_m (m : ℝ) : 
  (∀ x, proposition_p x m → proposition_q x) → 
  (1 ≤ m ∨ m ≤ -7) :=
sorry

end range_of_m_l130_130360


namespace soccer_players_l130_130470

/-- 
If the total number of socks in the washing machine is 16,
and each player wears a pair of socks (2 socks per player), 
then the number of players is 8. 
-/
theorem soccer_players (total_socks : ℕ) (socks_per_player : ℕ) (h1 : total_socks = 16) (h2 : socks_per_player = 2) : total_socks / socks_per_player = 8 :=
by
  -- Proof goes here
  sorry

end soccer_players_l130_130470


namespace tomato_red_flesh_probability_l130_130566

theorem tomato_red_flesh_probability :
  (P_yellow_skin : ℝ) = 3 / 8 →
  (P_red_flesh_given_yellow_skin : ℝ) = 8 / 15 →
  (P_yellow_skin_given_not_red_flesh : ℝ) = 7 / 30 →
  (P_red_flesh : ℝ) = 1 / 4 := 
by
  intros h1 h2 h3
  sorry

end tomato_red_flesh_probability_l130_130566


namespace smallest_int_k_for_64_pow_k_l130_130528

theorem smallest_int_k_for_64_pow_k (k : ℕ) (base : ℕ) (h₁ : k = 7) : 
  64^k > base^20 → base = 4 := by
  sorry

end smallest_int_k_for_64_pow_k_l130_130528


namespace Kara_books_proof_l130_130655

-- Let's define the conditions and the proof statement in Lean 4

def Candice_books : ℕ := 18
def Amanda_books := Candice_books / 3
def Kara_books := Amanda_books / 2

theorem Kara_books_proof : Kara_books = 3 := by
  -- setting up the conditions based on the given problem.
  have Amanda_books_correct : Amanda_books = 6 := by
    exact Nat.div_eq_of_eq_mul_right (Nat.zero_lt_succ 2) (rfl) -- 18 / 3 = 6

  have Kara_books_correct : Kara_books = 3 := by
    exact Nat.div_eq_of_eq_mul_right (Nat.zero_lt_succ 1) Amanda_books_correct -- 6 / 2 = 3

  exact Kara_books_correct

end Kara_books_proof_l130_130655


namespace tangent_line_l130_130134

noncomputable def f (x : ℝ) : ℝ := Real.sqrt x
noncomputable def g (x : ℝ) : ℝ := 1 / x

theorem tangent_line (x y : ℝ) (h_inter : y = f x ∧ y = g x) :
  (x - 2 * y + 1 = 0) :=
by
  sorry

end tangent_line_l130_130134


namespace lengths_equal_l130_130061

-- a rhombus AFCE inscribed in a rectangle ABCD
variables {A B C D E F : Type}
variables {width length perimeter side_BF side_DE : ℝ}
variables {AF CE FC AF_side FC_side : ℝ}
variables {h1 : width = 20} {h2 : length = 25} {h3 : perimeter = 82}
variables {h4 : side_BF = (82 / 4 - 20)} {h5 : side_DE = (82 / 4 - 20)} 

-- prove that the lengths of BF and DE are equal
theorem lengths_equal :
  side_BF = side_DE :=
by
  sorry

end lengths_equal_l130_130061


namespace does_not_represent_right_triangle_l130_130820

/-- In triangle ABC, the sides opposite to angles A, B, and C are a, b, and c respectively. Given:
  - a:b:c = 6:8:10
  - ∠A:∠B:∠C = 1:1:3
  - a^2 + c^2 = b^2
  - ∠A + ∠B = ∠C

Prove that the condition ∠A:∠B:∠C = 1:1:3 does not represent a right triangle ABC. -/
theorem does_not_represent_right_triangle
  (a b c : ℝ) (A B C : ℝ)
  (h1 : a / b = 6 / 8 ∧ b / c = 8 / 10)
  (h2 : A / B = 1 / 1 ∧ B / C = 1 / 3)
  (h3 : a^2 + c^2 = b^2)
  (h4 : A + B = C) :
  ¬ (B = 90) :=
sorry

end does_not_represent_right_triangle_l130_130820


namespace locus_of_moving_point_l130_130084

open Real

theorem locus_of_moving_point
  (M N P Q T E : ℝ × ℝ)
  (a b : ℝ)
  (h_ellipse_M : M.1^2 / 48 + M.2^2 / 16 = 1)
  (h_P : P = (-M.1, M.2))
  (h_Q : Q = (-M.1, -M.2))
  (h_T : T = (M.1, -M.2))
  (h_ellipse_N : N.1^2 / 48 + N.2^2 / 16 = 1)
  (h_perp : (M.1 - N.1) * (M.1 + N.1) + (M.2 - N.2) * (M.2 + N.2) = 0)
  (h_intersection : ∃ x y : ℝ, (y - Q.2) = (N.2 - Q.2)/(N.1 - Q.1) * (x - Q.1) ∧ (y - P.2) = (T.2 - P.2)/(T.1 - P.1) * (x - P.1) ∧ E = (x, y)) : 
  (E.1^2 / 12 + E.2^2 / 4 = 1) :=
  sorry

end locus_of_moving_point_l130_130084


namespace find_a_and_b_l130_130224

theorem find_a_and_b (a b : ℤ) (h1 : 3 * (b + a^2) = 99) (h2 : 3 * a * b^2 = 162) : a = 6 ∧ b = -3 :=
sorry

end find_a_and_b_l130_130224


namespace min_expression_value_l130_130910

theorem min_expression_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  ∃ x : ℝ, x = 5 ∧ ∀ y, (y = (b / (3 * a) + 3 / b)) → x ≤ y :=
by
  sorry

end min_expression_value_l130_130910


namespace pencils_per_student_l130_130372

theorem pencils_per_student (total_pencils : ℕ) (students : ℕ) (pencils_per_student : ℕ)
    (h1 : total_pencils = 125)
    (h2 : students = 25)
    (h3 : pencils_per_student = total_pencils / students) :
    pencils_per_student = 5 :=
by
  sorry

end pencils_per_student_l130_130372


namespace tedra_harvested_2000kg_l130_130595

noncomputable def totalTomatoesHarvested : ℕ :=
  let wednesday : ℕ := 400
  let thursday : ℕ := wednesday / 2
  let total_wednesday_thursday := wednesday + thursday
  let remaining_friday : ℕ := 700
  let given_away_friday : ℕ := 700
  let friday := remaining_friday + given_away_friday
  total_wednesday_thursday + friday

theorem tedra_harvested_2000kg :
  totalTomatoesHarvested = 2000 := by
  sorry

end tedra_harvested_2000kg_l130_130595


namespace ratio_of_books_sold_l130_130932

theorem ratio_of_books_sold
  (T W R : ℕ)
  (hT : T = 7)
  (hW : W = 3 * T)
  (hTotal : T + W + R = 91) :
  R / W = 3 :=
by
  sorry

end ratio_of_books_sold_l130_130932


namespace sin_2012_equals_neg_sin_32_l130_130475

theorem sin_2012_equals_neg_sin_32 : Real.sin (2012 * Real.pi / 180) = -Real.sin (32 * Real.pi / 180) := by
  sorry

end sin_2012_equals_neg_sin_32_l130_130475


namespace sum_of_first_half_of_numbers_l130_130448

theorem sum_of_first_half_of_numbers 
  (avg_total : ℝ) 
  (total_count : ℕ) 
  (avg_second_half : ℝ) 
  (sum_total : ℝ)
  (sum_second_half : ℝ)
  (sum_first_half : ℝ) 
  (h1 : total_count = 8)
  (h2 : avg_total = 43.1)
  (h3 : avg_second_half = 46.6)
  (h4 : sum_total = avg_total * total_count)
  (h5 : sum_second_half = 4 * avg_second_half)
  (h6 : sum_first_half = sum_total - sum_second_half)
  :
  sum_first_half = 158.4 := 
sorry

end sum_of_first_half_of_numbers_l130_130448


namespace part_a_part_b_l130_130100

-- Define the parabola y = x^2
def parabola (x : ℝ) : ℝ := x^2

-- Prove that (1, 1) lies on the parabola
theorem part_a : parabola 1 = 1 := by
  sorry

-- Prove that for any t, (t, t^2) lies on the parabola
theorem part_b (t : ℝ) : parabola t = t^2 := by
  sorry

end part_a_part_b_l130_130100


namespace find_a_l130_130103

def set_A : Set ℝ := { x | abs (x - 1) > 2 }
def set_B (a : ℝ) : Set ℝ := { x | x^2 - (a + 1) * x + a < 0 }
def intersection (A B : Set ℝ) : Set ℝ := {x | x ∈ A ∧ x ∈ B}

theorem find_a (a : ℝ) : (intersection set_A (set_B a)) = { x | 3 < x ∧ x < 5 } → a = 5 :=
by
  sorry

end find_a_l130_130103


namespace sufficient_but_not_necessary_sufficient_but_not_necessary_rel_l130_130781

theorem sufficient_but_not_necessary (a b : ℝ) (h : 0 < a ∧ a < b) : (1 / a) > (1 / b) :=
by
  sorry

theorem sufficient_but_not_necessary_rel (a b : ℝ) : 0 < a ∧ a < b ↔ (1 / a) > (1 / b) :=
by
  sorry

end sufficient_but_not_necessary_sufficient_but_not_necessary_rel_l130_130781


namespace solve_for_y_l130_130607

theorem solve_for_y (y : ℚ) (h : (4 / 7) * (1 / 5) * y - 2 = 10) : y = 105 := by
  sorry

end solve_for_y_l130_130607


namespace people_present_l130_130905

-- Number of parents, pupils, teachers, staff members, and volunteers
def num_parents : ℕ := 105
def num_pupils : ℕ := 698
def num_teachers : ℕ := 35
def num_staff_members : ℕ := 20
def num_volunteers : ℕ := 50

-- The total number of people present in the program
def total_people : ℕ := num_parents + num_pupils + num_teachers + num_staff_members + num_volunteers

-- Proof statement
theorem people_present : total_people = 908 := by
  -- Proof goes here, but adding sorry for now
  sorry

end people_present_l130_130905


namespace find_intersection_complement_find_value_m_l130_130808

-- (1) Problem Statement
theorem find_intersection_complement (A : Set ℝ) (B : Set ℝ) (x : ℝ) :
  (A = {x | x^2 - 4 * x - 5 ≤ 0}) →
  (B = {x | x^2 - 2 * x - 3 < 0}) →
  (x ∈ A ∩ (Bᶜ : Set ℝ)) ↔ (x = -1 ∨ 3 ≤ x ∧ x ≤ 5) :=
by
  sorry

-- (2) Problem Statement
theorem find_value_m (A B : Set ℝ) (m : ℝ) :
  (A = {x | x^2 - 4 * x - 5 ≤ 0}) →
  (B = {x | x^2 - 2 * x - m < 0}) →
  (A ∩ B = {x | -1 ≤ x ∧ x < 4}) →
  m = 8 :=
by
  sorry

end find_intersection_complement_find_value_m_l130_130808


namespace locus_of_vertices_l130_130582

theorem locus_of_vertices (t : ℝ) (x y : ℝ) (h : y = x^2 + t * x + 1) : y = 1 - x^2 :=
by
  sorry

end locus_of_vertices_l130_130582


namespace doris_weeks_to_meet_expenses_l130_130656

def doris_weekly_hours : Nat := 5 * 3 + 5 -- 5 weekdays (3 hours each) + 5 hours on Saturday
def doris_hourly_rate : Nat := 20 -- Doris earns $20 per hour
def doris_weekly_earnings : Nat := doris_weekly_hours * doris_hourly_rate -- The total earnings per week
def doris_monthly_expenses : Nat := 1200 -- Doris's monthly expense

theorem doris_weeks_to_meet_expenses : ∃ w : Nat, doris_weekly_earnings * w ≥ doris_monthly_expenses ∧ w = 3 :=
by
  sorry

end doris_weeks_to_meet_expenses_l130_130656


namespace compute_expression_l130_130837

theorem compute_expression :
  ( (12^4 + 324) * (24^4 + 324) * (36^4 + 324) * (48^4 + 324) * (60^4 + 324) )
  /
  ( (6^4 + 324) * (18^4 + 324) * (30^4 + 324) * (42^4 + 324) * (54^4 + 324) )
  = 221 := 
by sorry

end compute_expression_l130_130837


namespace math_problem_proof_l130_130935

theorem math_problem_proof (a b x y : ℝ) 
  (h1: x = a) 
  (h2: y = b)
  (h3: a + a = b * a)
  (h4: y = a)
  (h5: a * a = a + a)
  (h6: b = 3) : 
  x * y = 4 := 
by 
  sorry

end math_problem_proof_l130_130935


namespace quadratic_root_condition_l130_130836

theorem quadratic_root_condition (m n : ℝ) (h : m * (-1)^2 - n * (-1) - 2023 = 0) :
  m + n = 2023 :=
sorry

end quadratic_root_condition_l130_130836


namespace patrol_streets_in_one_hour_l130_130328

-- Definitions of the given conditions
def streets_patrolled_by_A := 36
def hours_by_A := 4
def rate_A := streets_patrolled_by_A / hours_by_A

def streets_patrolled_by_B := 55
def hours_by_B := 5
def rate_B := streets_patrolled_by_B / hours_by_B

def streets_patrolled_by_C := 42
def hours_by_C := 6
def rate_C := streets_patrolled_by_C / hours_by_C

-- Proof statement 
theorem patrol_streets_in_one_hour : rate_A + rate_B + rate_C = 27 := by
  sorry

end patrol_streets_in_one_hour_l130_130328


namespace find_absolute_cd_l130_130334

noncomputable def polynomial_solution (c d : ℤ) (root1 root2 root3 : ℤ) : Prop :=
  c ≠ 0 ∧ d ≠ 0 ∧ 
  root1 = root2 ∧
  (root3 ≠ root1 ∨ root3 ≠ root2) ∧
  (root1^3 + root2^2 * root3 + (c * root1^2) + (d * root1) + 16 * c = 0) ∧ 
  (root2^3 + root1^2 * root3 + (c * root2^2) + (d * root2) + 16 * c = 0) ∧
  (root3^3 + root1^2 * root3 + (c * root3^2) + (d * root3) + 16 * c = 0)

theorem find_absolute_cd : ∃ c d root1 root2 root3 : ℤ,
  polynomial_solution c d root1 root2 root3 ∧ (|c * d| = 2560) :=
sorry

end find_absolute_cd_l130_130334


namespace find_integer_n_l130_130449

theorem find_integer_n (n : ℤ) : (⌊(n^2 : ℤ) / 4⌋ - ⌊n / 2⌋ ^ 2 = 3) → n = 7 :=
by sorry

end find_integer_n_l130_130449


namespace intersection_points_l130_130994

def parabola1 (x : ℝ) : ℝ := 3 * x ^ 2 - 12 * x - 5
def parabola2 (x : ℝ) : ℝ := x ^ 2 - 2 * x + 3

theorem intersection_points :
  { p : ℝ × ℝ | p.snd = parabola1 p.fst ∧ p.snd = parabola2 p.fst } =
  { (1, -14), (4, -5) } :=
by
  sorry

end intersection_points_l130_130994


namespace customer_can_receive_exact_change_l130_130990

theorem customer_can_receive_exact_change (k : ℕ) (hk : k ≤ 1000) :
  ∃ change : ℕ, change + k = 1000 ∧ change ≤ 1999 :=
by
  sorry

end customer_can_receive_exact_change_l130_130990


namespace geometric_sequence_arithmetic_Sn_l130_130184

theorem geometric_sequence_arithmetic_Sn (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (a1 : ℝ) (n : ℕ) :
  (∀ n, a n = a1 * q ^ (n - 1)) →
  (∀ n, S n = a1 * (1 - q ^ n) / (1 - q)) →
  (∀ n, S (n + 1) - S n = S n - S (n - 1)) →
  q = 1 :=
by
  sorry

end geometric_sequence_arithmetic_Sn_l130_130184


namespace ratio_lcm_gcf_256_162_l130_130388

theorem ratio_lcm_gcf_256_162 : (Nat.lcm 256 162) / (Nat.gcd 256 162) = 10368 := 
by 
  sorry

end ratio_lcm_gcf_256_162_l130_130388


namespace KeatonAnnualEarnings_l130_130525

-- Keaton's conditions for oranges
def orangeHarvestInterval : ℕ := 2
def orangeSalePrice : ℕ := 50

-- Keaton's conditions for apples
def appleHarvestInterval : ℕ := 3
def appleSalePrice : ℕ := 30

-- Annual earnings calculation
def annualEarnings (monthsInYear : ℕ) : ℕ :=
  let orangeEarnings := (monthsInYear / orangeHarvestInterval) * orangeSalePrice
  let appleEarnings := (monthsInYear / appleHarvestInterval) * appleSalePrice
  orangeEarnings + appleEarnings

-- Prove the total annual earnings is 420
theorem KeatonAnnualEarnings : annualEarnings 12 = 420 :=
  by 
    -- We skip the proof details here.
    sorry

end KeatonAnnualEarnings_l130_130525


namespace math_problem_l130_130833
-- Import the entire mathlib library for necessary mathematical definitions and notations

-- Define the conditions and the statement to prove
theorem math_problem (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a ≠ b) : 
  a^3 + b^3 > a^2 * b + a * b^2 :=
by 
  -- place a sorry as a placeholder for the proof
  sorry

end math_problem_l130_130833


namespace value_of_x_l130_130041

theorem value_of_x (x y z : ℝ) (h1 : x = (1 / 2) * y) (h2 : y = (1 / 4) * z) (h3 : z = 80) : x = 10 := by
  sorry

end value_of_x_l130_130041


namespace value_of_expression_l130_130903

theorem value_of_expression :
  (3^2 - 3) - (4^2 - 4) + (5^2 - 5) - (6^2 - 6) = -16 :=
by
  sorry

end value_of_expression_l130_130903


namespace sum_powers_divisible_by_5_iff_l130_130082

theorem sum_powers_divisible_by_5_iff (n : ℕ) (h_pos : n > 0) :
  (1^n + 2^n + 3^n + 4^n) % 5 = 0 ↔ n % 4 ≠ 0 := 
sorry

end sum_powers_divisible_by_5_iff_l130_130082


namespace factor_x_squared_minus_169_l130_130472

theorem factor_x_squared_minus_169 (x : ℝ) : x^2 - 169 = (x - 13) * (x + 13) := 
by
  -- Recognize that 169 is a perfect square
  have h : 169 = 13^2 := by norm_num
  -- Use the difference of squares formula
  -- Sorry is used to skip the proof part
  sorry

end factor_x_squared_minus_169_l130_130472


namespace fruit_salad_mixture_l130_130251

theorem fruit_salad_mixture :
  ∃ (A P G : ℝ), A / P = 12 / 8 ∧ A / G = 12 / 7 ∧ P / G = 8 / 7 ∧ A = G + 10 ∧ A + P + G = 54 :=
by
  sorry

end fruit_salad_mixture_l130_130251


namespace square_field_area_l130_130293

noncomputable def area_of_square_field(speed_kph : ℝ) (time_hrs : ℝ) : ℝ :=
  let speed_mps := (speed_kph * 1000) / 3600
  let distance := speed_mps * (time_hrs * 3600)
  let side_length := distance / Real.sqrt 2
  side_length ^ 2

theorem square_field_area 
  (speed_kph : ℝ := 2.4)
  (time_hrs : ℝ := 3.0004166666666667) :
  area_of_square_field speed_kph time_hrs = 25939764.41 := 
by 
  -- This is a placeholder for the proof. 
  sorry

end square_field_area_l130_130293


namespace B_can_win_with_initial_config_B_l130_130378

def initial_configuration_B := (6, 2, 1)

def A_starts_and_B_wins (config : (Nat × Nat × Nat)) : Prop := sorry

theorem B_can_win_with_initial_config_B : A_starts_and_B_wins initial_configuration_B :=
sorry

end B_can_win_with_initial_config_B_l130_130378


namespace transistor_length_scientific_notation_l130_130942

theorem transistor_length_scientific_notation :
  0.000000006 = 6 * 10^(-9) := 
sorry

end transistor_length_scientific_notation_l130_130942


namespace algebra_expression_value_l130_130900

theorem algebra_expression_value (x : ℝ) (h : x^2 + 3 * x + 5 = 11) : 3 * x^2 + 9 * x + 12 = 30 := 
by
  sorry

end algebra_expression_value_l130_130900


namespace stratified_sampling_junior_teachers_l130_130581

theorem stratified_sampling_junior_teachers 
    (total_teachers : ℕ) (senior_teachers : ℕ) 
    (intermediate_teachers : ℕ) (junior_teachers : ℕ) 
    (sample_size : ℕ) 
    (H1 : total_teachers = 200)
    (H2 : senior_teachers = 20)
    (H3 : intermediate_teachers = 100)
    (H4 : junior_teachers = 80) 
    (H5 : sample_size = 50)
    : (junior_teachers * sample_size / total_teachers = 20) := 
  by 
    sorry

end stratified_sampling_junior_teachers_l130_130581


namespace range_of_k_l130_130647

theorem range_of_k (k : ℝ) 
  (h1 : ∀ x y : ℝ, (x^2 / (k-3) + y^2 / (2-k) = 1) → (k-3 < 0) ∧ (2-k > 0)) : 
  k < 2 := by
  sorry

end range_of_k_l130_130647


namespace daria_needs_to_earn_more_money_l130_130000

noncomputable def moneyNeeded (ticket_cost : ℕ) (discount : ℕ) (gift_card : ℕ) 
  (transport_cost : ℕ) (parking_cost : ℕ) (tshirt_cost : ℕ) (current_money : ℕ) (tickets : ℕ) : ℕ :=
  let discounted_ticket_price := ticket_cost - (ticket_cost * discount / 100)
  let total_ticket_cost := discounted_ticket_price * tickets
  let ticket_cost_after_gift_card := total_ticket_cost - gift_card
  let total_cost := ticket_cost_after_gift_card + transport_cost + parking_cost + tshirt_cost
  total_cost - current_money

theorem daria_needs_to_earn_more_money :
  moneyNeeded 90 10 50 20 10 25 189 6 = 302 :=
by
  sorry

end daria_needs_to_earn_more_money_l130_130000


namespace age_of_15th_student_l130_130024

theorem age_of_15th_student (avg_age_15_students avg_age_5_students avg_age_9_students : ℕ)
  (total_students total_age_15_students total_age_5_students total_age_9_students : ℕ)
  (h1 : total_students = 15)
  (h2 : avg_age_15_students = 15)
  (h3 : avg_age_5_students = 14)
  (h4 : avg_age_9_students = 16)
  (h5 : total_age_15_students = total_students * avg_age_15_students)
  (h6 : total_age_5_students = 5 * avg_age_5_students)
  (h7 : total_age_9_students = 9 * avg_age_9_students):
  total_age_15_students = total_age_5_students + total_age_9_students + 11 :=
by
  sorry

end age_of_15th_student_l130_130024


namespace tina_days_to_use_pink_pens_tina_total_pens_l130_130210

-- Definitions based on the problem conditions.
def pink_pens : ℕ := 15
def green_pens : ℕ := pink_pens - 9
def blue_pens : ℕ := green_pens + 3
def total_pink_green := pink_pens + green_pens
def yellow_pens : ℕ := total_pink_green - 5
def pink_pens_per_day := 4

-- Prove the two statements based on the definitions.
theorem tina_days_to_use_pink_pens 
  (h1 : pink_pens = 15)
  (h2 : pink_pens_per_day = 4) :
  4 = 4 :=
by sorry

theorem tina_total_pens 
  (h1 : pink_pens = 15)
  (h2 : green_pens = pink_pens - 9)
  (h3 : blue_pens = green_pens + 3)
  (h4 : yellow_pens = total_pink_green - 5) :
  pink_pens + green_pens + blue_pens + yellow_pens = 46 :=
by sorry

end tina_days_to_use_pink_pens_tina_total_pens_l130_130210


namespace solve_pos_int_a_l130_130860

theorem solve_pos_int_a :
  ∀ a : ℕ, (0 < a) →
  (∀ n : ℕ, (n ≥ 5) → ((2^n - n^2) ∣ (a^n - n^a))) →
  (a = 2 ∨ a = 4) :=
by
  sorry

end solve_pos_int_a_l130_130860


namespace unique_injective_f_solution_l130_130891

noncomputable def unique_injective_function (f : ℝ → ℝ) : Prop := 
  (∀ x y : ℝ, x ≠ y → f ((x + y) / (x - y)) = (f x + f y) / (f x - f y))

theorem unique_injective_f_solution :
  ∀ f : ℝ → ℝ, (∀ x y : ℝ, x ≠ y → f ((x + y) / (x - y)) = (f x + f y) / (f x - f y))
  → (∀ x y : ℝ, f x = f y → x = y) -- injectivity condition
  → ∀ x : ℝ, f x = x :=
sorry

end unique_injective_f_solution_l130_130891


namespace value_fraction_eq_three_l130_130037

namespace Problem

variable {R : Type} [Field R]

theorem value_fraction_eq_three (a b c : R) (h : a / 2 = b / 3 ∧ b / 3 = c / 4) :
  (a + b + c) / (2 * a + b - c) = 3 := by
  sorry

end Problem

end value_fraction_eq_three_l130_130037


namespace trigonometric_problem_l130_130243

theorem trigonometric_problem
  (α : ℝ)
  (h1 : Real.tan α = Real.sqrt 3)
  (h2 : π < α)
  (h3 : α < 3 * π / 2) :
  Real.cos (2 * π - α) - Real.sin α = (Real.sqrt 3 - 1) / 2 := by
  sorry

end trigonometric_problem_l130_130243


namespace second_lady_distance_l130_130232

theorem second_lady_distance (x : ℕ) 
  (h1 : ∃ y, y = 2 * x) 
  (h2 : x + 2 * x = 12) : x = 4 := 
by 
  sorry

end second_lady_distance_l130_130232


namespace man_l130_130180

-- Constants and conditions
def V_down : ℝ := 18  -- downstream speed in km/hr
def V_c : ℝ := 3.4    -- speed of the current in km/hr

-- Main statement to prove
theorem man's_speed_against_the_current : (V_down - V_c - V_c) = 11.2 := by
  sorry

end man_l130_130180


namespace trigonometric_identity_l130_130698

theorem trigonometric_identity :
  (2 * Real.sin (46 * Real.pi / 180) - Real.sqrt 3 * Real.cos (74 * Real.pi / 180)) / Real.cos (16 * Real.pi / 180) = 1 := 
by
  sorry

end trigonometric_identity_l130_130698


namespace area_of_inscribed_square_l130_130125

theorem area_of_inscribed_square (D : ℝ) (h : D = 10) : 
  ∃ A : ℝ, A = 50 :=
by
  sorry

end area_of_inscribed_square_l130_130125


namespace x_values_l130_130861

theorem x_values (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h1 : x + 1 / y = 7) (h2 : y + 1 / x = 1 / 3) :
  x = 3 ∨ x = 4 :=
by
  sorry

end x_values_l130_130861


namespace max_pieces_with_3_cuts_l130_130199

theorem max_pieces_with_3_cuts (cake : Type) : 
  (∀ (cuts : ℕ), cuts = 3 → (∃ (max_pieces : ℕ), max_pieces = 8)) := by
  sorry

end max_pieces_with_3_cuts_l130_130199


namespace f_is_periodic_f_nat_exact_l130_130092

noncomputable def f : ℝ → ℝ := sorry

axiom f_functional_eq (x y : ℝ) : f x + f y = 2 * f ((x + y) / 2) * f ((x - y) / 2)
axiom f_0_nonzero : f 0 ≠ 0
axiom f_1_zero : f 1 = 0

theorem f_is_periodic : ∃ T > 0, ∀ x : ℝ, f (x + T) = f x :=
  by
    use 4
    sorry

theorem f_nat_exact (n : ℕ) : f n = Real.cos (n * Real.pi / 2) :=
  by
    sorry

end f_is_periodic_f_nat_exact_l130_130092


namespace problem_statement_l130_130564

-- Given that f(x) is an even function.
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Definition of the main condition f(x) + f(2 - x) = 0.
def special_condition (f : ℝ → ℝ) : Prop := ∀ x, f x + f (2 - x) = 0

-- Theorem: Given the conditions, show that f(x) has a period of 4 and f(x-1) is odd.
theorem problem_statement {f : ℝ → ℝ} (h_even : is_even f) (h_cond : special_condition f) :
  (∀ x, f (4 + x) = f x) ∧ (∀ x, f (-x - 1) = -f (x - 1)) :=
by
  sorry

end problem_statement_l130_130564


namespace simplify_expression_l130_130940

-- Define the variables x and y
variables (x y : ℝ)

-- State the theorem
theorem simplify_expression (x y : ℝ) (hy : y ≠ 0) :
  ((x + 3 * y)^2 - (x + y) * (x - y)) / (2 * y) = 3 * x + 5 * y := 
by 
  -- skip the proof
  sorry

end simplify_expression_l130_130940


namespace find_cos_F1PF2_l130_130654

noncomputable def cos_angle_P_F1_F2 : ℝ :=
  let F1 := (-(4:ℝ), 0)
  let F2 := ((4:ℝ), 0)
  let a := (5:ℝ)
  let b := (3:ℝ)
  let P : ℝ × ℝ := sorry -- P is a point on the ellipse
  let area_triangle : ℝ := 3 * Real.sqrt 3
  let cos_angle : ℝ := 1 / 2
  cos_angle

def cos_angle_F1PF2_lemma (F1 F2 : ℝ × ℝ) (ellipse_Area : ℝ) (cos_angle : ℝ) : Prop :=
  cos_angle = 1/2

theorem find_cos_F1PF2 (a b : ℝ) (F1 F2 : ℝ × ℝ) (P : ℝ × ℝ) (Area_PF1F2 : ℝ) :
  (F1 = (-(4:ℝ), 0) ∧ F2 = ((4:ℝ), 0)) ∧ (Area_PF1F2 = 3 * Real.sqrt 3) ∧
  (P.1^2 / (a^2) + P.2^2 / (b^2) = 1) → cos_angle_F1PF2_lemma F1 F2 Area_PF1F2 (cos_angle_P_F1_F2)
:=
  sorry

end find_cos_F1PF2_l130_130654


namespace number_of_books_is_8_l130_130739

def books_and_albums (x y p_a p_b : ℕ) : Prop :=
  (x * p_b = 1056) ∧ (p_b = p_a + 100) ∧ (x = y + 6)

theorem number_of_books_is_8 (y p_a p_b : ℕ) (h : books_and_albums 8 y p_a p_b) : 8 = 8 :=
by
  sorry

end number_of_books_is_8_l130_130739


namespace supplementary_angles_difference_l130_130723

theorem supplementary_angles_difference 
  (x : ℝ) 
  (h1 : 5 * x + 3 * x = 180) 
  (h2 : 0 < x) : 
  abs (5 * x - 3 * x) = 45 :=
by sorry

end supplementary_angles_difference_l130_130723


namespace find_2theta_plus_phi_l130_130455

variable (θ φ : ℝ)
variable (hθ : 0 < θ ∧ θ < π / 2)
variable (hφ : 0 < φ ∧ φ < π / 2)
variable (tan_hθ : Real.tan θ = 2 / 5)
variable (cos_hφ : Real.cos φ = 1 / 2)

theorem find_2theta_plus_phi : 2 * θ + φ = π / 4 := by
  sorry

end find_2theta_plus_phi_l130_130455


namespace james_puzzle_completion_time_l130_130800

theorem james_puzzle_completion_time :
  let num_puzzles := 2
  let pieces_per_puzzle := 2000
  let pieces_per_10_minutes := 100
  let total_pieces := num_puzzles * pieces_per_puzzle
  let sets_of_100_pieces := total_pieces / pieces_per_10_minutes
  let total_minutes := sets_of_100_pieces * 10
  total_minutes = 400 :=
by
  -- Definitions based on conditions
  let num_puzzles := 2
  let pieces_per_puzzle := 2000
  let pieces_per_10_minutes := 100
  let total_pieces := num_puzzles * pieces_per_puzzle
  let sets_of_100_pieces := total_pieces / pieces_per_10_minutes
  let total_minutes := sets_of_100_pieces * 10

  -- Using sorry to skip proof
  sorry

end james_puzzle_completion_time_l130_130800


namespace trees_per_day_l130_130288

def blocks_per_tree := 3
def total_blocks := 30
def days := 5

theorem trees_per_day : (total_blocks / days) / blocks_per_tree = 2 := by
  sorry

end trees_per_day_l130_130288


namespace proof_problem_l130_130189

theorem proof_problem (x : ℝ) (a : ℝ) :
  (0 < x) → 
  (x + 1 / x ≥ 2) →
  (x + 4 / x^2 ≥ 3) →
  (x + 27 / x^3 ≥ 4) →
  a = 4^4 → 
  x + a / x^4 ≥ 5 :=
  sorry

end proof_problem_l130_130189


namespace real_roots_range_of_k_l130_130274

theorem real_roots_range_of_k (k : ℝ) : 
  (∃ x : ℝ, (k - 1) * x^2 - 2 * k * x + (k + 3) = 0) ↔ (k ≤ 3 / 2) :=
sorry

end real_roots_range_of_k_l130_130274


namespace days_to_clear_messages_l130_130849

theorem days_to_clear_messages 
  (initial_messages : ℕ)
  (messages_read_per_day : ℕ)
  (new_messages_per_day : ℕ) 
  (net_messages_cleared_per_day : ℕ)
  (d : ℕ) :
  initial_messages = 98 →
  messages_read_per_day = 20 →
  new_messages_per_day = 6 →
  net_messages_cleared_per_day = messages_read_per_day - new_messages_per_day →
  d = initial_messages / net_messages_cleared_per_day →
  d = 7 :=
by
  intros h_initial h_read h_new h_net h_days
  sorry

end days_to_clear_messages_l130_130849


namespace sine_beta_value_l130_130630

variable (α β : ℝ)
variable (h1 : 0 < α ∧ α < (π / 2))
variable (h2 : 0 < β ∧ β < (π / 2))
variable (h3 : Real.cos α = 4 / 5)
variable (h4 : Real.cos (α + β) = 3 / 5)

theorem sine_beta_value : Real.sin β = 7 / 25 :=
by
  -- The proof will go here
  sorry

end sine_beta_value_l130_130630


namespace jerry_liters_of_mustard_oil_l130_130033

-- Definitions
def cost_per_liter_mustard_oil : ℕ := 13
def cost_per_pound_penne_pasta : ℕ := 4
def cost_per_pound_pasta_sauce : ℕ := 5
def total_money_jerry_had : ℕ := 50
def money_left_with_jerry : ℕ := 7
def pounds_of_penne_pasta : ℕ := 3
def pounds_of_pasta_sauce : ℕ := 1

-- Our goal is to calculate how many liters of mustard oil Jerry bought
theorem jerry_liters_of_mustard_oil : ℕ :=
  let cost_of_penne_pasta := pounds_of_penne_pasta * cost_per_pound_penne_pasta
  let cost_of_pasta_sauce := pounds_of_pasta_sauce * cost_per_pound_pasta_sauce
  let total_spent := total_money_jerry_had - money_left_with_jerry
  let spent_on_pasta_and_sauce := cost_of_penne_pasta + cost_of_pasta_sauce
  let spent_on_mustard_oil := total_spent - spent_on_pasta_and_sauce
  spent_on_mustard_oil / cost_per_liter_mustard_oil

example : jerry_liters_of_mustard_oil = 2 := by
  unfold jerry_liters_of_mustard_oil
  simp
  sorry

end jerry_liters_of_mustard_oil_l130_130033


namespace average_marks_l130_130511

/--
Given:
1. The average marks in physics (P) and mathematics (M) is 90.
2. The average marks in physics (P) and chemistry (C) is 70.
3. The student scored 110 marks in physics (P).

Prove that the average marks the student scored in the 3 subjects (P, C, M) is 70.
-/
theorem average_marks (P C M : ℝ) 
  (h1 : (P + M) / 2 = 90)
  (h2 : (P + C) / 2 = 70)
  (h3 : P = 110) : 
  (P + C + M) / 3 = 70 :=
sorry

end average_marks_l130_130511


namespace eccentricity_hyperbola_l130_130104

-- Conditions
def is_eccentricity_ellipse (a b : ℝ) (h1 : a > b) (h2 : b > 0) : Prop :=
  let e := (Real.sqrt 2) / 2
  (Real.sqrt (1 - b^2 / a^2) = e)

-- Objective: Find the eccentricity of the given the hyperbola.
theorem eccentricity_hyperbola (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : is_eccentricity_ellipse a b h1 h2) : 
  Real.sqrt (1 + b^2 / a^2) = Real.sqrt 3 :=
sorry

end eccentricity_hyperbola_l130_130104


namespace aaron_earnings_l130_130444

def monday_hours : ℚ := 7 / 4
def tuesday_hours : ℚ := 1 + 10 / 60
def wednesday_hours : ℚ := 3 + 15 / 60
def friday_hours : ℚ := 45 / 60

def total_hours_worked : ℚ := monday_hours + tuesday_hours + wednesday_hours + friday_hours
def hourly_rate : ℚ := 4

def total_earnings : ℚ := total_hours_worked * hourly_rate

theorem aaron_earnings : total_earnings = 27 := by
  sorry

end aaron_earnings_l130_130444


namespace intersection_A_B_l130_130047

-- Define the set A
def A := {y : ℝ | ∃ x : ℝ, y = Real.sin x}

-- Define the set B
def B := {x : ℝ | x^2 - x < 0}

-- The proof problem statement in Lean 4
theorem intersection_A_B : A ∩ B = {y : ℝ | 0 < y ∧ y < 1} :=
by
  sorry

end intersection_A_B_l130_130047


namespace least_possible_value_of_y_l130_130369

theorem least_possible_value_of_y
  (x y z : ℤ)
  (hx : Even x)
  (hy : Odd y)
  (hz : Odd z)
  (h1 : y - x > 5)
  (h2 : ∀ z', z' - x ≥ 9 → z' ≥ 9) :
  y ≥ 7 :=
by
  -- Proof is not required here
  sorry

end least_possible_value_of_y_l130_130369


namespace optimal_play_results_in_draw_l130_130478

-- Define the concept of an optimal player, and a game state in Tic-Tac-Toe
structure Game :=
(board : Fin 3 × Fin 3 → Option Bool) -- Option Bool represents empty, O, or X
(turn : Bool) -- False for O's turn, True for X's turn

def draw (g : Game) : Bool :=
-- Implementation of checking for a draw will go here
sorry

noncomputable def optimal_move (g : Game) : Game :=
-- Implementation of finding the optimal move for the current player
sorry

theorem optimal_play_results_in_draw :
  ∀ (g : Game) (h : ∀ g, optimal_move g = g),
    draw (optimal_move g) = true :=
by
  -- The proof will be provided here
  sorry

end optimal_play_results_in_draw_l130_130478


namespace white_balls_count_l130_130395

theorem white_balls_count (n : ℕ) (h : 8 / (8 + n : ℝ) = 0.4) : n = 12 := by
  sorry

end white_balls_count_l130_130395


namespace find_f2_l130_130255

-- Define f as an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define g based on f
def g (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  f x + 9

-- Given conditions
variables (f : ℝ → ℝ) (h_odd : odd_function f)
variable (h_g_neg2 : g f (-2) = 3)

-- Theorem statement
theorem find_f2 : f 2 = 6 :=
by
  sorry

end find_f2_l130_130255


namespace system_of_equations_solution_l130_130287

theorem system_of_equations_solution :
  ∃ (x y : ℚ), 
    (2 * x - 3 * y = 1) ∧ 
    (5 * x + 4 * y = 6) ∧ 
    (x + 2 * y = 2) ∧
    x = 2 / 3 ∧ y = 2 / 3 :=
by {
  sorry
}

end system_of_equations_solution_l130_130287


namespace students_in_sample_l130_130627

theorem students_in_sample (T : ℕ) (S : ℕ) (F : ℕ) (J : ℕ) (se : ℕ)
  (h1 : J = 22 * T / 100)
  (h2 : S = 25 * T / 100)
  (h3 : se = 160)
  (h4 : F = S + 64)
  (h5 : ∀ x, x ∈ ({F, S, J, se} : Finset ℕ) → x ≤ T ∧  x ≥ 0):
  T = 800 :=
by
  have h6 : T = F + S + J + se := sorry
  sorry

end students_in_sample_l130_130627


namespace problem_1_problem_2_l130_130587

noncomputable def a : ℝ := Real.sqrt 7 + 2
noncomputable def b : ℝ := Real.sqrt 7 - 2

theorem problem_1 : a^2 * b + b^2 * a = 6 * Real.sqrt 7 := by
  sorry

theorem problem_2 : a^2 + a * b + b^2 = 25 := by
  sorry

end problem_1_problem_2_l130_130587


namespace total_distance_covered_l130_130219

theorem total_distance_covered :
  let speed1 := 40 -- miles per hour
  let speed2 := 50 -- miles per hour
  let speed3 := 30 -- miles per hour
  let time1 := 1.5 -- hours
  let time2 := 1 -- hour
  let time3 := 2.25 -- hours
  let distance1 := speed1 * time1 -- distance covered in the first part of the trip
  let distance2 := speed2 * time2 -- distance covered in the second part of the trip
  let distance3 := speed3 * time3 -- distance covered in the third part of the trip
  distance1 + distance2 + distance3 = 177.5 := 
by
  sorry

end total_distance_covered_l130_130219


namespace find_number_l130_130902

theorem find_number (x : ℝ) : 3 * (2 * x + 9) = 75 → x = 8 :=
by {
  sorry
}

end find_number_l130_130902


namespace expected_value_of_smallest_seven_selected_from_sixty_three_l130_130469

noncomputable def expected_value_smallest_selected (n r : ℕ) : ℕ :=
  (n + 1) / (r + 1)

theorem expected_value_of_smallest_seven_selected_from_sixty_three :
  expected_value_smallest_selected 63 7 = 8 :=
by
  sorry -- Proof is omitted as per instructions

end expected_value_of_smallest_seven_selected_from_sixty_three_l130_130469


namespace estate_problem_l130_130424

def totalEstateValue (E a b : ℝ) : Prop :=
  (a + b = (3/5) * E) ∧ 
  (a = 2 * b) ∧ 
  (3 * b = (3/5) * E) ∧ 
  (E = a + b + (3 * b) + 4000)

theorem estate_problem (E : ℝ) (a b : ℝ) :
  totalEstateValue E a b → E = 20000 :=
by
  -- The proof will be filled here
  sorry

end estate_problem_l130_130424


namespace horner_v3_at_2_l130_130043

-- Defining the polynomial f(x).
def f (x : ℝ) := 2 * x^5 + 3 * x^3 - 2 * x^2 + x - 1

-- Defining the Horner's method evaluation up to v3 at x = 2.
def horner_eval (x : ℝ) := (((2 * x + 0) * x + 3) * x - 2) * x + 1

-- The proof statement we need to show.
theorem horner_v3_at_2 : horner_eval 2 = 20 := sorry

end horner_v3_at_2_l130_130043


namespace suitable_for_census_l130_130291

-- Definitions based on the conditions in a)
def survey_A := "The service life of a batch of batteries"
def survey_B := "The height of all classmates in the class"
def survey_C := "The content of preservatives in a batch of food"
def survey_D := "The favorite mathematician of elementary and middle school students in the city"

-- The main statement to prove
theorem suitable_for_census : survey_B = "The height of all classmates in the class" := by
  -- We assert that the height of all classmates is the suitable survey for a census based on given conditions
  sorry

end suitable_for_census_l130_130291


namespace sequence_a_2011_l130_130488

noncomputable def sequence_a : ℕ → ℕ
| 0       => 2
| 1       => 3
| (n+2)   => (sequence_a (n+1) * sequence_a n) % 10

theorem sequence_a_2011 : sequence_a 2010 = 2 :=
by
  sorry

end sequence_a_2011_l130_130488


namespace smallest_four_digit_multiple_of_17_is_1013_l130_130446

-- Lean definition to state the problem
def smallest_four_digit_multiple_of_17 : ℕ :=
  1013

-- Main Lean theorem to assert the correctness
theorem smallest_four_digit_multiple_of_17_is_1013 :
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 17 ∣ n ∧ n = smallest_four_digit_multiple_of_17 :=
by
  -- proof here
  sorry

end smallest_four_digit_multiple_of_17_is_1013_l130_130446


namespace frequency_of_third_group_l130_130381

theorem frequency_of_third_group (total_data first_group second_group fourth_group third_group : ℕ) 
    (h1 : total_data = 40)
    (h2 : first_group = 5)
    (h3 : second_group = 12)
    (h4 : fourth_group = 8) :
    third_group = 15 :=
by
  sorry

end frequency_of_third_group_l130_130381


namespace not_partition_1985_1987_partition_1987_1989_l130_130026

-- Define the number of squares in an L-shape
def squares_in_lshape : ℕ := 3

-- Question 1: Can 1985 x 1987 be partitioned into L-shapes?
def partition_1985_1987 (m n : ℕ) (L_shape_size : ℕ) : Prop :=
  ∃ k : ℕ, m * n = k * L_shape_size ∧ (m % L_shape_size = 0 ∨ n % L_shape_size = 0)

theorem not_partition_1985_1987 :
  ¬ partition_1985_1987 1985 1987 squares_in_lshape :=
by {
  -- Proof omitted
  sorry
}

-- Question 2: Can 1987 x 1989 be partitioned into L-shapes?
theorem partition_1987_1989 :
  partition_1985_1987 1987 1989 squares_in_lshape :=
by {
  -- Proof omitted
  sorry
}

end not_partition_1985_1987_partition_1987_1989_l130_130026


namespace marcus_calzones_total_time_l130_130573

theorem marcus_calzones_total_time :
  let saute_onions_time := 20
  let saute_garlic_peppers_time := (1 / 4 : ℚ) * saute_onions_time
  let knead_time := 30
  let rest_time := 2 * knead_time
  let assemble_time := (1 / 10 : ℚ) * (knead_time + rest_time)
  let total_time := saute_onions_time + saute_garlic_peppers_time + knead_time + rest_time + assemble_time
  total_time = 124 :=
by
  let saute_onions_time := 20
  let saute_garlic_peppers_time := (1 / 4 : ℚ) * saute_onions_time
  let knead_time := 30
  let rest_time := 2 * knead_time
  let assemble_time := (1 / 10 : ℚ) * (knead_time + rest_time)
  let total_time := saute_onions_time + saute_garlic_peppers_time + knead_time + rest_time + assemble_time
  sorry

end marcus_calzones_total_time_l130_130573


namespace solution_set_of_inequalities_l130_130974

theorem solution_set_of_inequalities :
  {x : ℝ | 2 ≤ x / (3 * x - 5) ∧ x / (3 * x - 5) < 9} = {x : ℝ | x > 45 / 26} :=
by sorry

end solution_set_of_inequalities_l130_130974


namespace probability_of_passing_l130_130196

theorem probability_of_passing (pA pB pC : ℝ) (hA : pA = 0.8) (hB : pB = 0.6) (hC : pC = 0.5) :
  1 - (1 - pA) * (1 - pB) * (1 - pC) = 0.96 := by
  sorry

end probability_of_passing_l130_130196


namespace max_unique_solution_l130_130418

theorem max_unique_solution (x y : ℕ) (m : ℕ) (h : 2005 * x + 2007 * y = m) : 
  m = 2 * 2005 * 2007 ↔ ∃! (x y : ℕ), 2005 * x + 2007 * y = m :=
sorry

end max_unique_solution_l130_130418


namespace mixtape_first_side_songs_l130_130365

theorem mixtape_first_side_songs (total_length : ℕ) (second_side_songs : ℕ) (song_length : ℕ) :
  total_length = 40 → second_side_songs = 4 → song_length = 4 → (total_length - second_side_songs * song_length) / song_length = 6 := 
by
  intros h1 h2 h3
  sorry

end mixtape_first_side_songs_l130_130365


namespace b_in_terms_of_a_axis_of_symmetry_and_a_value_range_of_a_l130_130632

-- Definitions based on problem conditions
def parabola (a b x : ℝ) : ℝ := a * x^2 + b * x
def passes_through_A (a b : ℝ) : Prop := parabola a b 3 = 3
def points_on_parabola (a b x1 x2 : ℝ) : Prop := x1 < x2 ∧ x1 + x2 = 2
def equal_y_values (a b x1 x2 : ℝ) : Prop := parabola a b x1 = parabola a b x2
def less_than_y_values (a b x1 x2 : ℝ) : Prop := parabola a b x1 < parabola a b x2

-- 1) Express b in terms of a
theorem b_in_terms_of_a (a : ℝ) (h : passes_through_A a (1 - 3 * a)) : True := sorry

-- 2) Axis of symmetry and the value of a when y1 = y2
theorem axis_of_symmetry_and_a_value (a : ℝ) (x1 x2 : ℝ) 
    (h1 : points_on_parabola a (1 - 3 * a) x1 x2)
    (h2 : equal_y_values a (1 - 3 * a) x1 x2) 
    : a = 1 ∧ -1 / 2 * (1 - 3 * a) / a = 1 := sorry

-- 3) Range of values for a when y1 < y2
theorem range_of_a (a x1 x2 : ℝ) 
    (h1 : points_on_parabola a (1 - 3 * a) x1 x2)
    (h2 : less_than_y_values a (1 - 3 * a) x1 x2) 
    (h3 : a ≠ 0) : 0 < a ∧ a < 1 := sorry

end b_in_terms_of_a_axis_of_symmetry_and_a_value_range_of_a_l130_130632


namespace not_m_gt_132_l130_130392

theorem not_m_gt_132 (m : ℕ) (hm : 0 < m)
  (H : ∃ (k : ℕ), 1 / 2 + 1 / 3 + 1 / 11 + 1 / (m:ℚ) = k) :
  m ≤ 132 :=
sorry

end not_m_gt_132_l130_130392


namespace find_b_l130_130898

def h (x : ℝ) : ℝ := 5 * x + 6

theorem find_b : ∃ b : ℝ, h b = 0 ∧ b = -6 / 5 :=
by
  sorry

end find_b_l130_130898


namespace interest_amount_eq_750_l130_130904

-- Definitions
def P : ℕ := 3000
def R : ℕ := 5
def T : ℕ := 5

-- Condition
def interest_less_than_sum := 2250

-- Simple interest formula
def simple_interest (P R T : ℕ) := (P * R * T) / 100

-- Theorem
theorem interest_amount_eq_750 : simple_interest P R T = P - interest_less_than_sum :=
by
  -- We assert that we need to prove the equality holds.
  sorry

end interest_amount_eq_750_l130_130904


namespace find_a5_plus_a7_l130_130786

variable {a : ℕ → ℝ}

theorem find_a5_plus_a7 (h : a 3 + a 9 = 16) : a 5 + a 7 = 16 := 
sorry

end find_a5_plus_a7_l130_130786


namespace part_a_part_b_l130_130624

-- Define the natural numbers m and n
variable (m n : Nat)

-- Condition: m * n is divisible by m + n
def divisible_condition : Prop :=
  ∃ (k : Nat), m * n = k * (m + n)

-- Define prime number
def is_prime (p : Nat) : Prop :=
  p > 1 ∧ ∀ d : Nat, d ∣ p → d = 1 ∨ d = p

-- Define n as the product of two distinct primes
def is_product_of_two_distinct_primes (n : Nat) : Prop :=
  ∃ (p₁ p₂ : Nat), is_prime p₁ ∧ is_prime p₂ ∧ p₁ ≠ p₂ ∧ n = p₁ * p₂

-- Problem (a): Prove that m is divisible by n when n is a prime number and m * n is divisible by m + n
theorem part_a (prime_n : is_prime n) (h : divisible_condition m n) : n ∣ m := sorry

-- Problem (b): Prove that m is not necessarily divisible by n when n is a product of two distinct prime numbers
theorem part_b (prod_of_primes_n : is_product_of_two_distinct_primes n) (h : divisible_condition m n) :
  ¬ (n ∣ m) := sorry

end part_a_part_b_l130_130624


namespace second_divisor_27_l130_130396

theorem second_divisor_27 (N : ℤ) (D : ℤ) (k : ℤ) (q : ℤ) (h1 : N = 242 * k + 100) (h2 : N = D * q + 19) : D = 27 := by
  sorry

end second_divisor_27_l130_130396


namespace units_digit_of_n_l130_130875

theorem units_digit_of_n (m n : ℕ) (h1 : m * n = 14^8) (h2 : m % 10 = 4) : n % 10 = 4 :=
by
  sorry

end units_digit_of_n_l130_130875


namespace proof_problem_l130_130677

open Set

variable {U : Set ℕ} {A : Set ℕ} {B : Set ℕ}

def problem_statement (U A B : Set ℕ) : Prop :=
  ((U \ A) ∪ B) = {2, 3}

theorem proof_problem :
  problem_statement {0, 1, 2, 3} {0, 1, 2} {2, 3} :=
by
  unfold problem_statement
  simp
  sorry

end proof_problem_l130_130677


namespace boys_and_girls_original_total_l130_130286

theorem boys_and_girls_original_total (b g : ℕ) 
(h1 : b = 3 * g) 
(h2 : b - 4 = 5 * (g - 4)) : 
b + g = 32 := 
sorry

end boys_and_girls_original_total_l130_130286


namespace johns_bakery_fraction_l130_130894

theorem johns_bakery_fraction :
  ∀ (M : ℝ), 
  (M / 4 + M / 3 + 6 + (24 - (M / 4 + M / 3 + 6)) = 24) →
  (24 : ℝ) = M →
  (4 + 8 + 6 = 18) →
  (24 - 18 = 6) →
  (6 / 24 = (1 / 6 : ℝ)) :=
by
  intros M h1 h2 h3 h4
  sorry

end johns_bakery_fraction_l130_130894


namespace burglary_charge_sentence_l130_130853

theorem burglary_charge_sentence (B : ℕ) 
  (arson_counts : ℕ := 3) 
  (arson_sentence : ℕ := 36)
  (burglary_charges : ℕ := 2)
  (petty_larceny_factor : ℕ := 6)
  (total_jail_time : ℕ := 216) :
  arson_counts * arson_sentence + burglary_charges * B + (burglary_charges * petty_larceny_factor) * (B / 3) = total_jail_time → B = 18 := 
by
  sorry

end burglary_charge_sentence_l130_130853


namespace total_bushels_needed_l130_130729

def cows := 5
def sheep := 4
def chickens := 8
def pigs := 6
def horses := 2

def cow_bushels := 3.5
def sheep_bushels := 1.75
def chicken_bushels := 1.25
def pig_bushels := 4.5
def horse_bushels := 5.75

theorem total_bushels_needed
  (cows : ℕ) (sheep : ℕ) (chickens : ℕ) (pigs : ℕ) (horses : ℕ)
  (cow_bushels: ℝ) (sheep_bushels: ℝ) (chicken_bushels: ℝ) (pig_bushels: ℝ) (horse_bushels: ℝ) :
  cows * cow_bushels + sheep * sheep_bushels + chickens * chicken_bushels + pigs * pig_bushels + horses * horse_bushels = 73 :=
by
  -- Skipping the proof
  sorry

end total_bushels_needed_l130_130729


namespace matrix_B3_is_zero_unique_l130_130417

theorem matrix_B3_is_zero_unique (B : Matrix (Fin 2) (Fin 2) ℝ) (h : B^4 = 0) :
  ∃! (B3 : Matrix (Fin 2) (Fin 2) ℝ), B3 = B^3 ∧ B3 = 0 := sorry

end matrix_B3_is_zero_unique_l130_130417


namespace probability_of_neither_red_nor_purple_l130_130988

theorem probability_of_neither_red_nor_purple :
  let total_balls := 100
  let white_balls := 20
  let green_balls := 30
  let yellow_balls := 10
  let red_balls := 37
  let purple_balls := 3
  let neither_red_nor_purple_balls := white_balls + green_balls + yellow_balls
  (neither_red_nor_purple_balls : ℝ) / (total_balls : ℝ) = 0.6 :=
by
  sorry

end probability_of_neither_red_nor_purple_l130_130988


namespace expected_value_abs_diff_HT_l130_130923

noncomputable def expected_abs_diff_HT : ℚ :=
  let F : ℕ → ℚ := sorry -- Recurrence relation omitted for brevity
  F 0

theorem expected_value_abs_diff_HT :
  expected_abs_diff_HT = 24 / 7 :=
sorry

end expected_value_abs_diff_HT_l130_130923


namespace pipeline_problem_l130_130821

theorem pipeline_problem 
  (length_pipeline : ℕ) 
  (extra_meters : ℕ) 
  (days_saved : ℕ) 
  (x : ℕ)
  (h1 : length_pipeline = 4000) 
  (h2 : extra_meters = 10) 
  (h3 : days_saved = 20) 
  (h4 : (4000:ℕ) / (x - extra_meters) - (4000:ℕ) / x = days_saved) :
  x = 4000 / ((4000 / (x - extra_meters) + 20)) + extra_meters :=
by
  -- The proof goes here
  sorry

end pipeline_problem_l130_130821


namespace woman_completion_days_l130_130877

variable (M W : ℚ)
variable (work_days_man work_days_total : ℚ)

-- Given conditions
def condition1 : Prop :=
  (10 * M + 15 * W) * 7 = 1

def condition2 : Prop :=
  M * 100 = 1

-- To prove
def one_woman_days : ℚ := 350

theorem woman_completion_days (h1 : condition1 M W) (h2 : condition2 M) :
  1 / W = one_woman_days :=
by
  sorry

end woman_completion_days_l130_130877


namespace fraction_pizza_covered_by_pepperoni_l130_130770

theorem fraction_pizza_covered_by_pepperoni :
  (∀ (r_pizz : ℝ) (n_pepp : ℕ) (d_pepp : ℝ),
      r_pizz = 8 ∧ n_pepp = 32 ∧ d_pepp = 2 →
      (n_pepp * π * (d_pepp / 2)^2) / (π * r_pizz^2) = 1 / 2) :=
sorry

end fraction_pizza_covered_by_pepperoni_l130_130770


namespace max_number_ahn_can_get_l130_130604

theorem max_number_ahn_can_get :
  ∃ n : ℤ, (10 ≤ n ∧ n ≤ 99) ∧ ∀ m : ℤ, (10 ≤ m ∧ m ≤ 99) → (3 * (300 - n) ≥ 3 * (300 - m)) ∧ 3 * (300 - n) = 870 :=
by sorry

end max_number_ahn_can_get_l130_130604


namespace gcd_840_1764_l130_130819

-- Define the numbers according to the conditions
def a : ℕ := 1764
def b : ℕ := 840

-- The goal is to prove that the GCD of a and b is 84
theorem gcd_840_1764 : Nat.gcd a b = 84 := 
by
  -- The proof steps would normally go here
  sorry

end gcd_840_1764_l130_130819


namespace min_distinct_values_l130_130642

theorem min_distinct_values (n : ℕ) (mode_freq : ℕ) (total : ℕ)
  (h1 : total = 3000) (h2 : mode_freq = 15) :
  n = 215 :=
by
  sorry

end min_distinct_values_l130_130642


namespace range_of_m_l130_130687

theorem range_of_m (m : ℝ) : (-1 : ℝ) ≤ m ∧ m ≤ 3 ∧ ∀ x y : ℝ, x - ((m^2) - 2 * m + 4) * y - 6 > 0 → (x, y) ≠ (-1, -1) := 
by sorry

end range_of_m_l130_130687


namespace divisor_of_product_of_four_consecutive_integers_l130_130806

theorem divisor_of_product_of_four_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end divisor_of_product_of_four_consecutive_integers_l130_130806


namespace volume_of_box_l130_130527

theorem volume_of_box (l w h : ℝ) (hlw : l * w = 36) (hwh : w * h = 18) (hlh : l * h = 8) : 
    l * w * h = 72 := 
by 
    sorry

end volume_of_box_l130_130527


namespace degrees_to_radians_750_l130_130267

theorem degrees_to_radians_750 (π : ℝ) (deg_750 : ℝ) 
  (h : 180 = π) : 
  750 * (π / 180) = 25 / 6 * π :=
by
  sorry

end degrees_to_radians_750_l130_130267


namespace initial_points_count_l130_130602

theorem initial_points_count (k : ℕ) (h : (4 * k - 3) = 101): k = 26 :=
by 
  sorry

end initial_points_count_l130_130602


namespace find_s_l130_130616

def f (x s : ℝ) : ℝ := 3 * x^3 - 2 * x^2 + 4 * x + s

theorem find_s (s : ℝ) : f (-1) s = 0 → s = 9 :=
by
  sorry

end find_s_l130_130616


namespace number_of_integer_solutions_l130_130319

theorem number_of_integer_solutions (x : ℤ) :
  (∃ n : ℤ, n^2 = x^4 + 8*x^3 + 18*x^2 + 8*x + 36) ↔ x = -1 :=
sorry

end number_of_integer_solutions_l130_130319


namespace eq_of_divides_l130_130226

theorem eq_of_divides (a b : ℕ) (h : (4 * a * b - 1) ∣ (4 * a^2 - 1)^2) : a = b :=
sorry

end eq_of_divides_l130_130226


namespace james_carrot_sticks_l130_130031

theorem james_carrot_sticks (carrots_before : ℕ) (carrots_after : ℕ) 
(h_before : carrots_before = 22) (h_after : carrots_after = 15) : 
carrots_before + carrots_after = 37 := 
by 
  -- Placeholder for proof
  sorry

end james_carrot_sticks_l130_130031


namespace number_of_factors_of_60_l130_130567

theorem number_of_factors_of_60 : 
  ∃ n, n = 12 ∧ 
  (∀ p k : ℕ, p ∈ [2, 3, 5] → 60 = 2^2 * 3^1 * 5^1 → (∃ d : ℕ, d = (2 + 1) * (1 + 1) * (1 + 1) ∧ n = d)) :=
by sorry

end number_of_factors_of_60_l130_130567


namespace building_height_l130_130239

theorem building_height (h : ℕ) 
  (flagpole_height flagpole_shadow building_shadow : ℕ)
  (h_flagpole : flagpole_height = 18)
  (s_flagpole : flagpole_shadow = 45)
  (s_building : building_shadow = 70) 
  (condition : flagpole_height / flagpole_shadow = h / building_shadow) :
  h = 28 := by
  sorry

end building_height_l130_130239


namespace slope_of_line_is_neg_one_l130_130895

theorem slope_of_line_is_neg_one (y : ℝ) (h : (y - 5) / (5 - (-3)) = -1) : y = -3 :=
by
  sorry

end slope_of_line_is_neg_one_l130_130895


namespace hyperbola_dot_product_zero_l130_130892

theorem hyperbola_dot_product_zero
  (a b x y : ℝ)
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_hyperbola : (x^2 / a^2) - (y^2 / b^2) = 1)
  (h_ecc : (Real.sqrt (a^2 + b^2)) / a = Real.sqrt 2) :
  let B := (-x, y)
  let C := (x, y)
  let A := (a, 0)
  let AB := (B.1 - A.1, B.2 - A.2)
  let AC := (C.1 - A.1, C.2 - A.2)
  (AB.1 * AC.1 + AB.2 * AC.2) = 0 :=
by
  sorry

end hyperbola_dot_product_zero_l130_130892


namespace find_z_l130_130401

theorem find_z (z : ℝ) 
    (cos_angle : (2 + 2 * z) / ((Real.sqrt (1 + z^2)) * 3) = 2 / 3) : 
    z = 0 := 
sorry

end find_z_l130_130401


namespace solve_equation_l130_130261

-- Define the equation and the conditions
def problem_equation (x : ℝ) : Prop :=
  (3 * x + 6) / (x^2 + 5 * x - 6) = (3 - x) / (x - 2)

def valid_solution (x : ℝ) : Prop :=
  x ≠ 2 ∧ x ≠ 1 ∧ x ≠ -6

-- State the theorem that solutions x = 3 and x = -4 solve the problem under the conditions
theorem solve_equation : ∀ x : ℝ, valid_solution x → (x = 3 ∨ x = -4 ∧ problem_equation x) :=
by
  sorry

end solve_equation_l130_130261


namespace total_cases_l130_130187

-- Define the number of boys' high schools and girls' high schools
def boys_high_schools : Nat := 4
def girls_high_schools : Nat := 3

-- Theorem to be proven
theorem total_cases (B G : Nat) (hB : B = boys_high_schools) (hG : G = girls_high_schools) : 
  B + G = 7 :=
by
  rw [hB, hG]
  exact rfl

end total_cases_l130_130187


namespace muffins_sugar_l130_130971

theorem muffins_sugar (cups_muffins_ratio : 24 * 3 = 72 * s / 9) : s = 9 := by
  sorry

end muffins_sugar_l130_130971


namespace ali_initial_money_l130_130178

theorem ali_initial_money (X : ℝ) (h1 : X / 2 - (1 / 3) * (X / 2) = 160) : X = 480 :=
by sorry

end ali_initial_money_l130_130178


namespace factorize_expression_l130_130314

theorem factorize_expression (a : ℝ) : 2 * a^2 - 8 = 2 * (a + 2) * (a - 2) :=
by
  -- skipping the proof
  sorry

end factorize_expression_l130_130314


namespace solve_for_x_l130_130114

theorem solve_for_x :
  (48 = 5 * x + 3) → x = 9 :=
by
  sorry

end solve_for_x_l130_130114


namespace smallest_m_for_integral_solutions_l130_130433

theorem smallest_m_for_integral_solutions (p q : ℤ) (h : p * q = 42) (h0 : p + q = m / 15) : 
  0 < m ∧ 15 * p * p - m * p + 630 = 0 ∧ 15 * q * q - m * q + 630 = 0 →
  m = 195 :=
by 
  sorry

end smallest_m_for_integral_solutions_l130_130433


namespace negation_of_p_l130_130558

open Classical

-- Define proposition p
def p : Prop := ∀ x : ℝ, x^2 + x > 2

-- Define the negation of proposition p
def not_p : Prop := ∃ x : ℝ, x^2 + x ≤ 2

theorem negation_of_p : ¬p ↔ not_p :=
by sorry

end negation_of_p_l130_130558


namespace total_books_sum_l130_130834

-- Given conditions
def Joan_books := 10
def Tom_books := 38
def Lisa_books := 27
def Steve_books := 45
def Kim_books := 14
def Alex_books := 48

-- Define the total number of books
def total_books := Joan_books + Tom_books + Lisa_books + Steve_books + Kim_books + Alex_books

-- Proof statement
theorem total_books_sum : total_books = 182 := by
  sorry

end total_books_sum_l130_130834


namespace simplify_expression_l130_130052

theorem simplify_expression (a b : ℝ) (h1 : 2 * b - a < 3) (h2 : 2 * a - b < 5) : 
  -abs (2 * b - a - 7) - abs (b - 2 * a + 8) + abs (a + b - 9) = -6 :=
by
  sorry

end simplify_expression_l130_130052


namespace prove_value_of_question_l130_130710

theorem prove_value_of_question :
  let a := 9548
  let b := 7314
  let c := 3362
  let value_of_question : ℕ := by 
    sorry -- Proof steps to show the computation.

  (a + b = value_of_question) ∧ (c + 13500 = value_of_question) :=
by {
  let a := 9548
  let b := 7314
  let c := 3362
  let sum_of_a_b := a + b
  let computed_question := sum_of_a_b - c
  sorry -- Proof steps to show sum_of_a_b and the final result.
}

end prove_value_of_question_l130_130710


namespace range_of_a_l130_130711

def proposition_p (a : ℝ) : Prop := ∀ x : ℝ, ¬ (x^2 + (a-1)*x + 1 ≤ 0)
def proposition_q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (a-1)^x < (a-1)^y

theorem range_of_a (a : ℝ) :
  ¬ (proposition_p a ∧ proposition_q a) ∧ (proposition_p a ∨ proposition_q a) →
  (-1 < a ∧ a ≤ 2) ∨ (3 ≤ a) :=
by sorry

end range_of_a_l130_130711


namespace complement_of_M_l130_130170

open Set

-- Define the universal set
def U : Set ℝ := univ

-- Define the set M
def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 - 1}

-- The theorem stating the complement of M in U
theorem complement_of_M : (U \ M) = {y | y < -1} :=
by
  sorry

end complement_of_M_l130_130170


namespace solve_problem_l130_130881
noncomputable def is_solution (n : ℕ) : Prop :=
  ∀ (a b c : ℕ), (0 < a) → (0 < b) → (0 < c) → (a + b + c ∣ a^2 + b^2 + c^2) → (a + b + c ∣ a^n + b^n + c^n)

theorem solve_problem : {n : ℕ // is_solution (3 * n - 1) ∧ is_solution (3 * n - 2)} :=
sorry

end solve_problem_l130_130881


namespace values_of_cos_0_45_l130_130747

-- Define the interval and the condition for the cos function
def in_interval (x : ℝ) : Prop := 0 ≤ x ∧ x < 360
def cos_condition (x : ℝ) : Prop := Real.cos x = 0.45

-- Final theorem statement
theorem values_of_cos_0_45 : 
  ∃ (n : ℕ), n = 2 ∧ ∀ (x : ℝ), in_interval x ∧ cos_condition x ↔ x = 1 ∨ x = 2 := 
sorry

end values_of_cos_0_45_l130_130747


namespace auditorium_shared_days_l130_130068

theorem auditorium_shared_days :
  let drama_club_days := 3
  let choir_days := 5
  let debate_team_days := 7
  Nat.lcm (Nat.lcm drama_club_days choir_days) debate_team_days = 105 :=
by
  let drama_club_days := 3
  let choir_days := 5
  let debate_team_days := 7
  sorry

end auditorium_shared_days_l130_130068


namespace min_value_expr_l130_130683

theorem min_value_expr (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  8 * a^3 + 12 * b^3 + 27 * c^3 + 1 / (9 * a * b * c) ≥ 4 := 
sorry

end min_value_expr_l130_130683


namespace bell_pepper_slices_l130_130907

theorem bell_pepper_slices :
  ∀ (num_peppers : ℕ) (slices_per_pepper : ℕ) (total_slices_pieces : ℕ) (half_slices : ℕ),
  num_peppers = 5 → slices_per_pepper = 20 → total_slices_pieces = 200 →
  half_slices = (num_peppers * slices_per_pepper) / 2 →
  (total_slices_pieces - (num_peppers * slices_per_pepper)) / half_slices = 2 :=
by
  intros num_peppers slices_per_pepper total_slices_pieces half_slices h1 h2 h3 h4
  -- skip the proof with sorry as instructed
  sorry

end bell_pepper_slices_l130_130907


namespace expr1_val_expr2_val_l130_130271

noncomputable def expr1 : ℝ :=
  (1 / Real.sin (10 * Real.pi / 180)) - (Real.sqrt 3 / Real.cos (10 * Real.pi / 180))

theorem expr1_val : expr1 = 4 :=
  sorry

noncomputable def expr2 : ℝ :=
  (Real.sin (50 * Real.pi / 180) * (1 + Real.sqrt 3 * Real.tan (10 * Real.pi / 180)) - Real.cos (20 * Real.pi / 180)) /
  (Real.cos (80 * Real.pi / 180) * Real.sqrt (1 - Real.cos (20 * Real.pi / 180)))

theorem expr2_val : expr2 = Real.sqrt 2 :=
  sorry

end expr1_val_expr2_val_l130_130271


namespace member_pays_48_percent_of_SRP_l130_130136

theorem member_pays_48_percent_of_SRP
  (P : ℝ)
  (h₀ : P > 0)
  (basic_discount : ℝ := 0.40)
  (additional_discount : ℝ := 0.20) :
  ((1 - additional_discount) * (1 - basic_discount) * P) / P * 100 = 48 := by
  sorry

end member_pays_48_percent_of_SRP_l130_130136


namespace sequence_property_l130_130080

theorem sequence_property (a : ℕ → ℕ) (h1 : ∀ n, n ≥ 1 → a n ∈ { x | x ≥ 1 }) 
  (h2 : ∀ n, n ≥ 1 → a (a n) + a n = 2 * n) : ∀ n, n ≥ 1 → a n = n :=
by
  sorry

end sequence_property_l130_130080


namespace smallest_k_for_inequality_l130_130700

theorem smallest_k_for_inequality : 
  ∃ k : ℕ,  k > 0 ∧ ( (k-10) ^ 5026 ≥ 2013 ^ 2013 ) ∧ 
  (∀ m : ℕ, m > 0 ∧ ((m-10) ^ 5026) ≥ 2013 ^ 2013 → m ≥ 55) :=
sorry

end smallest_k_for_inequality_l130_130700


namespace minimize_distance_l130_130109

noncomputable def f (x : ℝ) := 9 * x^3
noncomputable def g (x : ℝ) := Real.log x

theorem minimize_distance :
  ∃ m > 0, (∀ x > 0, |f m - g m| ≤ |f x - g x|) ∧ m = 1/3 :=
sorry

end minimize_distance_l130_130109


namespace mary_initial_blue_crayons_l130_130124

/-- **Mathematically equivalent proof problem**:
  Given that Mary has 5 green crayons and gives away 3 green crayons and 1 blue crayon,
  and she has 9 crayons left, prove that she initially had 8 blue crayons. 
  -/
theorem mary_initial_blue_crayons (initial_green_crayons : ℕ) (green_given_away : ℕ) (blue_given_away : ℕ)
  (crayons_left : ℕ) (initial_crayons : ℕ) :
  initial_green_crayons = 5 →
  green_given_away = 3 →
  blue_given_away = 1 →
  crayons_left = 9 →
  initial_crayons = crayons_left + (green_given_away + blue_given_away) →
  initial_crayons - initial_green_crayons = 8 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end mary_initial_blue_crayons_l130_130124


namespace problem_l130_130078

def f (x : ℝ) : ℝ := x^2 - 3 * x + 7
def g (x : ℝ) : ℝ := 2 * x + 4
theorem problem : f (g 5) - g (f 5) = 123 := 
by 
  sorry

end problem_l130_130078


namespace commodity_price_difference_l130_130578

theorem commodity_price_difference (r : ℝ) (t : ℕ) :
  let P_X (t : ℕ) := 4.20 * (1 + (2*r + 10)/100)^(t - 2001)
  let P_Y (t : ℕ) := 4.40 * (1 + (r + 15)/100)^(t - 2001)
  P_X t = P_Y t + 0.90  ->
  ∃ t : ℕ, true :=
by
  sorry

end commodity_price_difference_l130_130578


namespace right_handed_total_l130_130156

theorem right_handed_total (total_players throwers : Nat) (h1 : total_players = 70) (h2 : throwers = 37) :
  let non_throwers := total_players - throwers
  let left_handed := non_throwers / 3
  let right_handed_non_throwers := non_throwers - left_handed
  let right_handed := right_handed_non_throwers + throwers
  right_handed = 59 :=
by
  sorry

end right_handed_total_l130_130156


namespace checkered_rectangles_containing_one_gray_cell_l130_130872

theorem checkered_rectangles_containing_one_gray_cell 
  (num_gray_cells : ℕ) 
  (num_blue_cells : ℕ) 
  (num_red_cells : ℕ)
  (blue_containing_rectangles : ℕ) 
  (red_containing_rectangles : ℕ) :
  num_gray_cells = 40 →
  num_blue_cells = 36 →
  num_red_cells = 4 →
  blue_containing_rectangles = 4 →
  red_containing_rectangles = 8 →
  num_blue_cells * blue_containing_rectangles + num_red_cells * red_containing_rectangles = 176 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end checkered_rectangles_containing_one_gray_cell_l130_130872


namespace cylinder_volume_triple_radius_quadruple_height_l130_130978

open Real

theorem cylinder_volume_triple_radius_quadruple_height (r h : ℝ) (V : ℝ) (hV : V = π * r^2 * h) :
  (3 * r) ^ 2 * 4 * h * π = 360 :=
by
  sorry

end cylinder_volume_triple_radius_quadruple_height_l130_130978


namespace minimum_value_of_f_l130_130356

def f (x a : ℝ) : ℝ := abs (x + 1) + abs (a * x + 1)

theorem minimum_value_of_f (a : ℝ) :
  (∀ x : ℝ, f x a ≥ 3 / 2) →
  (∃ x : ℝ, f x a = 3 / 2) →
  (a = -1 / 2 ∨ a = -2) :=
by
  intros h1 h2
  sorry

end minimum_value_of_f_l130_130356


namespace work_rate_combined_l130_130094

theorem work_rate_combined (a b c : ℝ) (ha : a = 21) (hb : b = 6) (hc : c = 12) :
  (1 / ((1 / a) + (1 / b) + (1 / c))) = 84 / 25 := by
  sorry

end work_rate_combined_l130_130094


namespace angle_EHG_65_l130_130571

/-- Quadrilateral $EFGH$ has $EF = FG = GH$, $\angle EFG = 80^\circ$, and $\angle FGH = 150^\circ$; and hence the degree measure of $\angle EHG$ is $65^\circ$. -/
theorem angle_EHG_65 {EF FG GH : ℝ} (h1 : EF = FG) (h2 : FG = GH) 
  (EFG : ℝ) (FGH : ℝ) (h3 : EFG = 80) (h4 : FGH = 150) : 
  ∃ EHG : ℝ, EHG = 65 :=
by
  sorry

end angle_EHG_65_l130_130571


namespace sasha_made_an_error_l130_130050

theorem sasha_made_an_error :
  ∀ (f : ℕ → ℤ), 
  (∀ n, 1 ≤ n → n ≤ 9 → f n = n ∨ f n = -n) →
  (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 = 21) →
  (f 1 + f 2 + f 3 + f 4 + f 5 + f 6 + f 7 + f 8 + f 9 = 20) →
  false :=
by
  intros f h_cons h_volodya_sum h_sasha_sum
  sorry

end sasha_made_an_error_l130_130050


namespace div_iff_div_l130_130371

theorem div_iff_div {a b : ℤ} : (29 ∣ (3 * a + 2 * b)) ↔ (29 ∣ (11 * a + 17 * b)) := 
by sorry

end div_iff_div_l130_130371


namespace problem_solution_l130_130116

noncomputable def inequality_holds (a b : ℝ) (n : ℕ) : Prop :=
  (a + b)^n - a^n - b^n ≥ 2^(2 * n) - 2^(n - 1)

theorem problem_solution (a b : ℝ) (n : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (1 / a + 1 / b = 1)) (h4 : 0 < n):
  inequality_holds a b n :=
by
  sorry

end problem_solution_l130_130116


namespace radius_of_circle_zero_l130_130599

theorem radius_of_circle_zero :
  (∃ x y : ℝ, x^2 + 8 * x + y^2 - 10 * y + 41 = 0) →
  (0 : ℝ) = 0 :=
by
  intro h
  sorry

end radius_of_circle_zero_l130_130599


namespace combination_permutation_value_l130_130768

theorem combination_permutation_value (n : ℕ) (h : (n * (n - 1)) = 42) : (Nat.factorial n) / (Nat.factorial 3 * Nat.factorial (n - 3)) = 35 := 
by
  sorry

end combination_permutation_value_l130_130768


namespace total_distance_walked_l130_130879

-- Define the conditions
def home_to_school : ℕ := 750
def half_distance : ℕ := home_to_school / 2
def return_home : ℕ := half_distance
def home_to_school_again : ℕ := home_to_school

-- Define the theorem statement
theorem total_distance_walked : 
  half_distance + return_home + home_to_school_again = 1500 := by
  sorry

end total_distance_walked_l130_130879


namespace difference_between_20th_and_first_15_l130_130085

def grains_on_square (k : ℕ) : ℕ := 2^k

def total_grains_on_first_15_squares : ℕ :=
  (Finset.range 15).sum (λ k => grains_on_square (k + 1))

def grains_on_20th_square : ℕ := grains_on_square 20

theorem difference_between_20th_and_first_15 :
  grains_on_20th_square - total_grains_on_first_15_squares = 983042 :=
by
  sorry

end difference_between_20th_and_first_15_l130_130085


namespace fixed_costs_16699_50_l130_130937

noncomputable def fixed_monthly_costs (production_cost shipping_cost units_sold price_per_unit : ℝ) : ℝ :=
  let total_variable_cost := (production_cost + shipping_cost) * units_sold
  let total_revenue := price_per_unit * units_sold
  total_revenue - total_variable_cost

theorem fixed_costs_16699_50 :
  fixed_monthly_costs 80 7 150 198.33 = 16699.5 :=
by
  sorry

end fixed_costs_16699_50_l130_130937


namespace diet_soda_bottles_l130_130029

theorem diet_soda_bottles (r d l t : Nat) (h1 : r = 49) (h2 : l = 6) (h3 : t = 89) (h4 : t = r + d) : d = 40 :=
by
  sorry

end diet_soda_bottles_l130_130029


namespace zero_cleverly_numbers_l130_130400

theorem zero_cleverly_numbers (n : ℕ) : 
  (1000 ≤ n ∧ n < 10000) ∧ (∃ a b c, n = 1000 * a + 10 * b + c ∧ b = 0 ∧ 9 * (100 * a + 10 * b + c) = n) ↔ (n = 2025 ∨ n = 4050 ∨ n = 6075) := 
sorry

end zero_cleverly_numbers_l130_130400


namespace p_expression_l130_130289

theorem p_expression (m n p : ℤ) (r1 r2 : ℝ) 
  (h1 : r1 + r2 = m) 
  (h2 : r1 * r2 = n) 
  (h3 : r1^2 + r2^2 = p) : 
  p = m^2 - 2 * n := by
  sorry

end p_expression_l130_130289


namespace A_finishes_remaining_work_in_6_days_l130_130651

-- Definitions for conditions
def A_workdays : ℕ := 18
def B_workdays : ℕ := 15
def B_worked_days : ℕ := 10

-- Proof problem statement
theorem A_finishes_remaining_work_in_6_days (A_workdays B_workdays B_worked_days : ℕ) :
  let rate_A := 1 / A_workdays
  let rate_B := 1 / B_workdays
  let work_done_by_B := B_worked_days * rate_B
  let remaining_work := 1 - work_done_by_B
  let days_A_needs := remaining_work / rate_A
  days_A_needs = 6 :=
by
  sorry

end A_finishes_remaining_work_in_6_days_l130_130651


namespace contradiction_assumption_l130_130439

theorem contradiction_assumption (a b : ℝ) (h : |a - 1| * |b - 1| = 0) : ¬ (a ≠ 1 ∧ b ≠ 1) :=
  sorry

end contradiction_assumption_l130_130439


namespace company_l130_130928

-- Define conditions
def initial_outlay : ℝ := 10000

def material_cost_per_set_first_300 : ℝ := 20
def material_cost_per_set_beyond_300 : ℝ := 15

def exchange_rate : ℝ := 1.1

def import_tax_rate : ℝ := 0.10

def sales_price_per_set_first_400 : ℝ := 50
def sales_price_per_set_beyond_400 : ℝ := 45

def export_tax_threshold : ℕ := 500
def export_tax_rate : ℝ := 0.05

def production_and_sales : ℕ := 800

-- Helper functions for the problem
def material_cost_first_300_sets : ℝ :=
  300 * material_cost_per_set_first_300 * exchange_rate

def material_cost_next_500_sets : ℝ :=
  (production_and_sales - 300) * material_cost_per_set_beyond_300 * exchange_rate

def total_material_cost : ℝ :=
  material_cost_first_300_sets + material_cost_next_500_sets

def import_tax : ℝ := total_material_cost * import_tax_rate

def total_manufacturing_cost : ℝ :=
  initial_outlay + total_material_cost + import_tax

def sales_revenue_first_400_sets : ℝ :=
  400 * sales_price_per_set_first_400

def sales_revenue_next_400_sets : ℝ :=
  (production_and_sales - 400) * sales_price_per_set_beyond_400

def total_sales_revenue_before_export_tax : ℝ :=
  sales_revenue_first_400_sets + sales_revenue_next_400_sets

def sales_revenue_beyond_threshold : ℝ :=
  (production_and_sales - export_tax_threshold) * sales_price_per_set_beyond_400

def export_tax : ℝ := sales_revenue_beyond_threshold * export_tax_rate

def total_sales_revenue_after_export_tax : ℝ :=
  total_sales_revenue_before_export_tax - export_tax

def profit : ℝ :=
  total_sales_revenue_after_export_tax - total_manufacturing_cost

-- Lean 4 statement for the proof problem
theorem company's_profit_is_10990 :
  profit = 10990 := by
  sorry

end company_l130_130928


namespace series_sum_eq_five_l130_130714

open Nat Real

noncomputable def sum_series : ℝ := ∑' (n : ℕ), (2 * n ^ 2 - n) / (n * (n + 1) * (n + 2))

theorem series_sum_eq_five : sum_series = 5 :=
sorry

end series_sum_eq_five_l130_130714


namespace determine_solution_set_inequality_l130_130238

-- Definitions based on given conditions
def quadratic_inequality_solution (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c > 0
def new_quadratic_inequality_solution (c b a : ℝ) (x : ℝ) := c * x^2 + b * x + a < 0

-- The proof statement
theorem determine_solution_set_inequality (a b c : ℝ):
  (∀ x : ℝ, -1/3 < x ∧ x < 2 → quadratic_inequality_solution a b c x) →
  (∀ x : ℝ, -3 < x ∧ x < 1/2 ↔ new_quadratic_inequality_solution c b a x) := sorry

end determine_solution_set_inequality_l130_130238


namespace f_11_f_2021_eq_neg_one_l130_130846

def f (n : ℕ) : ℚ := sorry

axiom recurrence_relation (n : ℕ) : f (n + 3) = (f n - 1) / (f n + 1)
axiom f1_ne_zero : f 1 ≠ 0
axiom f1_ne_one : f 1 ≠ 1
axiom f1_ne_neg_one : f 1 ≠ -1

theorem f_11_f_2021_eq_neg_one : f 11 * f 2021 = -1 := 
by
  sorry

end f_11_f_2021_eq_neg_one_l130_130846


namespace time_away_is_43point64_minutes_l130_130112

theorem time_away_is_43point64_minutes :
  ∃ (n1 n2 : ℝ), 
    (195 + n1 / 2 - 6 * n1 = 120 ∨ 195 + n1 / 2 - 6 * n1 = -120) ∧
    (195 + n2 / 2 - 6 * n2 = 120 ∨ 195 + n2 / 2 - 6 * n2 = -120) ∧
    n1 ≠ n2 ∧
    n1 < 60 ∧
    n2 < 60 ∧
    |n2 - n1| = 43.64 :=
sorry

end time_away_is_43point64_minutes_l130_130112


namespace roots_abs_less_than_one_l130_130355

theorem roots_abs_less_than_one {a b : ℝ} 
    (h : |a| + |b| < 1) 
    (x1 x2 : ℝ) 
    (h_roots : x1 * x1 + a * x1 + b = 0) 
    (h_roots' : x2 * x2 + a * x2 + b = 0) 
    : |x1| < 1 ∧ |x2| < 1 := 
sorry

end roots_abs_less_than_one_l130_130355


namespace bunny_burrows_l130_130240

theorem bunny_burrows (x : ℕ) (h1 : 20 * x * 600 = 36000) : x = 3 :=
by
  -- Skipping proof using sorry
  sorry

end bunny_burrows_l130_130240


namespace necessary_and_sufficient_condition_l130_130883

variable (a : ℝ)

theorem necessary_and_sufficient_condition :
  (-16 ≤ a ∧ a ≤ 0) ↔ ∀ x : ℝ, ¬(x^2 + a * x - 4 * a < 0) :=
by
  sorry

end necessary_and_sufficient_condition_l130_130883


namespace bounce_height_less_than_two_l130_130105

theorem bounce_height_less_than_two (k : ℕ) (h₀ : ℝ) (r : ℝ) (ε : ℝ) 
    (h₀_pos : h₀ = 20) (r_pos : r = 1/2) (ε_pos : ε = 2): 
  (h₀ * (r ^ k) < ε) ↔ k >= 4 := by
  sorry

end bounce_height_less_than_two_l130_130105


namespace sum_geometric_sequence_l130_130182

theorem sum_geometric_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (a1 : ℝ)
  (h1 : a 5 = -2) (h2 : a 8 = 16)
  (hq : q^3 = a 8 / a 5) (ha1 : a 1 = a1)
  (hS : S n = a1 * (1 - q^n) / (1 - q))
  : S 6 = 21 / 8 :=
sorry

end sum_geometric_sequence_l130_130182


namespace range_of_a_l130_130132

noncomputable def f (a x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * a * x^2 + (a - 1) * x + 1
noncomputable def f' (a x : ℝ) : ℝ := x^2 - a * x + a - 1

theorem range_of_a (a : ℝ) :
  (∀ x, 1 < x ∧ x < 4 → f' a x ≤ 0) ∧ (∀ x, 6 < x → f' a x ≥ 0) ↔ 5 ≤ a ∧ a ≤ 7 :=
by
  sorry

end range_of_a_l130_130132


namespace max_remainder_l130_130456

theorem max_remainder (y : ℕ) : 
  ∃ q r : ℕ, y = 11 * q + r ∧ r < 11 ∧ r = 10 := by sorry

end max_remainder_l130_130456


namespace max_pancake_pieces_3_cuts_l130_130792

open Nat

def P : ℕ → ℕ
| 0 => 1
| n => n * (n + 1) / 2 + 1

theorem max_pancake_pieces_3_cuts : P 3 = 7 := by
  have h0: P 0 = 1 := by rfl
  have h1: P 1 = 2 := by rfl
  have h2: P 2 = 4 := by rfl
  show P 3 = 7
  calc
    P 3 = 3 * (3 + 1) / 2 + 1 := by rfl
    _ = 3 * 4 / 2 + 1 := by rfl
    _ = 6 + 1 := by norm_num
    _ = 7 := by norm_num

end max_pancake_pieces_3_cuts_l130_130792


namespace project_presentation_periods_l130_130717

def students : ℕ := 32
def period_length : ℕ := 40
def presentation_time_per_student : ℕ := 5

theorem project_presentation_periods : 
  (students * presentation_time_per_student) / period_length = 4 := by
  sorry

end project_presentation_periods_l130_130717


namespace find_lisa_speed_l130_130353

theorem find_lisa_speed (Distance : ℕ) (Time : ℕ) (h1 : Distance = 256) (h2 : Time = 8) : Distance / Time = 32 := 
by {
  sorry
}

end find_lisa_speed_l130_130353


namespace sin_double_angle_sin_multiple_angle_l130_130657

-- Prove that |sin(2x)| <= 2|sin(x)| for any value of x
theorem sin_double_angle (x : ℝ) : |Real.sin (2 * x)| ≤ 2 * |Real.sin x| := 
by sorry

-- Prove that |sin(nx)| <= n|sin(x)| for any positive integer n and any value of x
theorem sin_multiple_angle (n : ℕ) (x : ℝ) (h : 0 < n) : |Real.sin (n * x)| ≤ n * |Real.sin x| :=
by sorry

end sin_double_angle_sin_multiple_angle_l130_130657


namespace journey_distance_l130_130559

theorem journey_distance (D : ℝ) (h1 : (D / 40) + (D / 60) = 40) : D = 960 :=
by
  sorry

end journey_distance_l130_130559


namespace speed_ratio_l130_130915

variable (d_A d_B : ℝ) (t_A t_B : ℝ)

-- Define the conditions
def condition1 : Prop := d_A = (1 + 1/5) * d_B
def condition2 : Prop := t_B = (1 - 1/11) * t_A

-- State the theorem that the speed ratio is 12:11
theorem speed_ratio (h1 : condition1 d_A d_B) (h2 : condition2 t_A t_B) :
  (d_A / t_A) / (d_B / t_B) = 12 / 11 :=
sorry

end speed_ratio_l130_130915


namespace infinite_pairs_m_n_l130_130878

theorem infinite_pairs_m_n :
  ∃ (f : ℕ → ℕ × ℕ), (∀ k, (f k).1 > 0 ∧ (f k).2 > 0 ∧ ((f k).1 ∣ (f k).2 ^ 2 + 1) ∧ ((f k).2 ∣ (f k).1 ^ 2 + 1)) :=
sorry

end infinite_pairs_m_n_l130_130878


namespace new_class_average_l130_130040

theorem new_class_average (total_students : ℕ) (students_group1 : ℕ) (avg1 : ℝ) (students_group2 : ℕ) (avg2 : ℝ) : 
  total_students = 40 → students_group1 = 28 → avg1 = 68 → students_group2 = 12 → avg2 = 77 → 
  ((students_group1 * avg1 + students_group2 * avg2) / total_students) = 70.7 :=
by
  sorry

end new_class_average_l130_130040


namespace sandra_remaining_money_l130_130032

def sandra_savings : ℝ := 10
def mother_contribution : ℝ := 4
def father_contribution : ℝ := 2 * mother_contribution
def candy_cost : ℝ := 0.5
def jelly_bean_cost : ℝ := 0.2
def num_candies : ℝ := 14
def num_jelly_beans : ℝ := 20

theorem sandra_remaining_money : (sandra_savings + mother_contribution + father_contribution) - (num_candies * candy_cost + num_jelly_beans * jelly_bean_cost) = 11 :=
by
  sorry

end sandra_remaining_money_l130_130032


namespace three_two_three_zero_zero_zero_zero_in_scientific_notation_l130_130574

theorem three_two_three_zero_zero_zero_zero_in_scientific_notation :
  3230000 = 3.23 * 10^6 :=
sorry

end three_two_three_zero_zero_zero_zero_in_scientific_notation_l130_130574


namespace smallest_n_to_make_183_divisible_by_11_l130_130737

theorem smallest_n_to_make_183_divisible_by_11 : ∃ n : ℕ, 183 + n % 11 = 0 ∧ n = 4 :=
by
  have h1 : 183 % 11 = 7 := 
    sorry
  let n := 11 - (183 % 11)
  have h2 : 183 + n % 11 = 0 :=
    sorry
  exact ⟨n, h2, sorry⟩

end smallest_n_to_make_183_divisible_by_11_l130_130737


namespace simple_interest_rate_l130_130579

theorem simple_interest_rate :
  ∀ (P T F : ℝ), P = 1000 → T = 3 → F = 1300 → (F - P) = P * 0.1 * T :=
by
  intros P T F hP hT hF
  sorry

end simple_interest_rate_l130_130579


namespace student_tickets_sold_l130_130225

theorem student_tickets_sold (S NS : ℕ) (h1 : 9 * S + 11 * NS = 20960) (h2 : S + NS = 2000) : S = 520 :=
by
  sorry

end student_tickets_sold_l130_130225


namespace smallest_n_for_symmetry_property_l130_130344

-- Define the setup for the problem
def has_required_symmetry (n : ℕ) : Prop :=
∀ (S : Finset (Fin n)), S.card = 5 →
∃ (l : Fin n → Fin n), (∀ v ∈ S, l v ≠ v) ∧ (∀ v ∈ S, l v ∉ S)

-- The main lemma we are proving
theorem smallest_n_for_symmetry_property : ∃ n : ℕ, (∀ m < n, ¬ has_required_symmetry m) ∧ has_required_symmetry 14 :=
by
  sorry

end smallest_n_for_symmetry_property_l130_130344


namespace sum_ages_l130_130503

theorem sum_ages (a b c : ℕ) (h1 : a = b) (h2 : a * b * c = 72) : a + b + c = 14 := 
by 
  sorry

end sum_ages_l130_130503


namespace stairs_climbed_l130_130359

theorem stairs_climbed (s v r : ℕ) 
  (h_s: s = 318) 
  (h_v: v = 18 + s / 2) 
  (h_r: r = 2 * v) 
  : s + v + r = 849 :=
by {
  sorry
}

end stairs_climbed_l130_130359


namespace find_f_7_l130_130514

noncomputable def f (a b c d x : ℝ) : ℝ :=
  a * x^8 + b * x^7 + c * x^3 + d * x - 6

theorem find_f_7 (a b c d : ℝ) (h : f a b c d (-7) = 10) :
  f a b c d 7 = 11529580 * a - 22 :=
sorry

end find_f_7_l130_130514


namespace lowest_score_of_14_scores_l130_130741

theorem lowest_score_of_14_scores (mean_14 : ℝ) (new_mean_12 : ℝ) (highest_score : ℝ) (lowest_score : ℝ) :
  mean_14 = 85 ∧ new_mean_12 = 88 ∧ highest_score = 105 → lowest_score = 29 :=
by
  sorry

end lowest_score_of_14_scores_l130_130741


namespace quadratic_equation_roots_sum_and_difference_l130_130556

theorem quadratic_equation_roots_sum_and_difference :
  ∃ (p q : ℝ), 
    p + q = 7 ∧ 
    |p - q| = 9 ∧ 
    (∀ x, (x - p) * (x - q) = x^2 - 7 * x - 8) :=
sorry

end quadratic_equation_roots_sum_and_difference_l130_130556


namespace length_RS_14_l130_130096

-- Definitions of conditions
def edges : List ℕ := [8, 14, 19, 28, 37, 42]
def PQ_length : ℕ := 42

-- Problem statement
theorem length_RS_14 (edges : List ℕ) (PQ_length : ℕ) (h : PQ_length = 42) (h_edges : edges = [8, 14, 19, 28, 37, 42]) :
  ∃ RS_length : ℕ, RS_length ∈ edges ∧ RS_length = 14 :=
by
  sorry

end length_RS_14_l130_130096


namespace fifth_friend_payment_l130_130279

/-- 
Five friends bought a piece of furniture for $120.
The first friend paid one third of the sum of the amounts paid by the other four;
the second friend paid one fourth of the sum of the amounts paid by the other four;
the third friend paid one fifth of the sum of the amounts paid by the other four;
and the fourth friend paid one sixth of the sum of the amounts paid by the other four.
Prove that the fifth friend paid $41.33.
-/
theorem fifth_friend_payment :
  ∀ (a b c d e : ℝ),
    a = 1/3 * (b + c + d + e) →
    b = 1/4 * (a + c + d + e) →
    c = 1/5 * (a + b + d + e) →
    d = 1/6 * (a + b + c + e) →
    a + b + c + d + e = 120 →
    e = 41.33 :=
by
  intros a b c d e ha hb hc hd he_sum
  sorry

end fifth_friend_payment_l130_130279


namespace machine_present_value_l130_130695

/-- A machine depreciates at a certain rate annually.
    Given the future value after a certain number of years and the depreciation rate,
    prove the present value of the machine. -/
theorem machine_present_value
  (depreciation_rate : ℝ := 0.25)
  (future_value : ℝ := 54000)
  (years : ℕ := 3)
  (pv : ℝ := 128000) :
  (future_value = pv * (1 - depreciation_rate) ^ years) :=
sorry

end machine_present_value_l130_130695


namespace hours_per_day_is_8_l130_130613

-- Define the conditions
def hire_two_bodyguards (day_count : ℕ) (total_payment : ℕ) (hourly_rate : ℕ) (daily_hours : ℕ) : Prop :=
  2 * hourly_rate * day_count * daily_hours = total_payment

-- Define the correct answer
theorem hours_per_day_is_8 :
  hire_two_bodyguards 7 2240 20 8 :=
by
  -- Here, you would provide the step-by-step justification, but we use sorry since no proof is required.
  sorry

end hours_per_day_is_8_l130_130613


namespace speech_competition_sequences_l130_130499

theorem speech_competition_sequences
    (contestants : Fin 5 → Prop)
    (girls boys : Fin 5 → Prop)
    (girl_A : Fin 5)
    (not_girl_A_first : ¬contestants 0)
    (no_consecutive_boys : ∀ i, boys i → ¬boys (i + 1))
    (count_girls : ∀ x, girls x → x = girl_A ∨ (contestants x ∧ ¬boys x))
    (count_boys : ∀ x, (boys x) → contestants x)
    (total_count : Fin 5 → Fin 5 → ℕ)
    (correct_answer : total_count = 276) : 
    ∃ seq_count, seq_count = 276 := 
sorry

end speech_competition_sequences_l130_130499


namespace sqrt_floor_square_l130_130419

theorem sqrt_floor_square (h1 : 7 < Real.sqrt 50) (h2 : Real.sqrt 50 < 8) :
  Int.floor (Real.sqrt 50) ^ 2 = 49 := by
  sorry

end sqrt_floor_square_l130_130419


namespace P_plus_Q_l130_130784

theorem P_plus_Q (P Q : ℝ) (h : ∀ x : ℝ, x ≠ 4 → (P / (x - 4) + Q * (x + 2) = (-4 * x^2 + 16 * x + 30) / (x - 4))) : P + Q = 42 :=
sorry

end P_plus_Q_l130_130784


namespace correct_operation_result_l130_130429

-- Define the conditions
def original_number : ℤ := 231
def incorrect_result : ℤ := 13

-- Define the two incorrect operations and the intended corrections
def reverse_subtract : ℤ := incorrect_result + 20
def reverse_division : ℤ := reverse_subtract * 7

-- Define the intended operations
def intended_multiplication : ℤ := original_number * 7
def intended_addition : ℤ := intended_multiplication + 20

-- The theorem we need to prove
theorem correct_operation_result :
  original_number = reverse_division →
  intended_addition > 1100 :=
by
  intros h
  sorry

end correct_operation_result_l130_130429


namespace square_feet_per_acre_l130_130839

-- Define the conditions
def rent_per_acre_per_month : ℝ := 60
def total_rent_per_month : ℝ := 600
def length_of_plot : ℝ := 360
def width_of_plot : ℝ := 1210

-- Translate the problem to a Lean theorem
theorem square_feet_per_acre :
  (length_of_plot * width_of_plot) / (total_rent_per_month / rent_per_acre_per_month) = 43560 :=
by {
  -- skipping the proof steps
  sorry
}

end square_feet_per_acre_l130_130839


namespace fraction_unshaded_area_l130_130465

theorem fraction_unshaded_area (s : ℝ) :
  let P := (s / 2, 0)
  let Q := (s, s / 2)
  let top_left := (0, s)
  let area_triangle : ℝ := 1 / 2 * (s / 2) * (s / 2)
  let area_square : ℝ := s * s
  let unshaded_area : ℝ := area_square - area_triangle
  let fraction_unshaded : ℝ := unshaded_area / area_square
  fraction_unshaded = 7 / 8 := 
by 
  sorry

end fraction_unshaded_area_l130_130465


namespace total_yearly_car_leasing_cost_l130_130666

-- Define mileage per day
def mileage_per_day (day : String) : ℕ :=
  if day = "Monday" ∨ day = "Wednesday" ∨ day = "Friday" ∨ day = "Sunday" then 50
  else if day = "Tuesday" ∨ day = "Thursday" then 80
  else if day = "Saturday" then 120
  else 0

-- Define weekly mileage
def weekly_mileage : ℕ := 4 * 50 + 2 * 80 + 120

-- Define cost parameters
def cost_per_mile : ℕ := 1 / 10
def weekly_fee : ℕ := 100
def monthly_toll_parking_fees : ℕ := 50
def discount_every_5th_week : ℕ := 30
def number_of_weeks_in_year : ℕ := 52

-- Define total yearly cost
def total_cost_yearly : ℕ :=
  let total_weekly_cost := (weekly_mileage * cost_per_mile + weekly_fee)
  let total_yearly_cost := total_weekly_cost * number_of_weeks_in_year
  let total_discounts := (number_of_weeks_in_year / 5) * discount_every_5th_week
  let annual_cost_without_tolls := total_yearly_cost - total_discounts
  let total_toll_fees := monthly_toll_parking_fees * 12
  annual_cost_without_tolls + total_toll_fees

-- Define the main theorem
theorem total_yearly_car_leasing_cost : total_cost_yearly = 7996 := 
  by
    -- Proof omitted
    sorry

end total_yearly_car_leasing_cost_l130_130666


namespace relationship_y1_y2_y3_l130_130480

-- Define the quadratic function
def quadratic_function (x : ℝ) : ℝ := -3 * x^2 + 2

-- Define the points and their coordinates
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨-1, quadratic_function (-1)⟩
def B : Point := ⟨1, quadratic_function 1⟩
def C : Point := ⟨2, quadratic_function 2⟩

-- Prove the relationship between y1, y2, and y3
theorem relationship_y1_y2_y3 :
  A.y = B.y ∧ A.y > C.y :=
by
  sorry

end relationship_y1_y2_y3_l130_130480


namespace multiple_of_first_number_is_eight_l130_130361

theorem multiple_of_first_number_is_eight 
  (a b c k : ℤ)
  (h1 : a = 7) 
  (h2 : b = a + 2) 
  (h3 : c = b + 2) 
  (h4 : 7 * k = 3 * c + (2 * b + 5)) : 
  k = 8 :=
by
  sorry

end multiple_of_first_number_is_eight_l130_130361


namespace find_N_l130_130468

theorem find_N (N : ℤ) :
  (10 + 11 + 12) / 3 = (2010 + 2011 + 2012 + N) / 4 → N = -5989 :=
by
  sorry

end find_N_l130_130468


namespace female_democrats_l130_130252

theorem female_democrats (F M D_f: ℕ) 
  (h1 : F + M = 780)
  (h2 : D_f = (1/2) * F)
  (h3 : (1/3) * 780 = 260)
  (h4 : 260 = (1/2) * F + (1/4) * M) : 
  D_f = 130 := 
by
  sorry

end female_democrats_l130_130252


namespace geometric_sequence_ratio_l130_130506

/-
Given a geometric sequence {a_n} with common ratio q ≠ -1 and q ≠ 1,
and S_n is the sum of the first n terms of the geometric sequence.
Given S_{12} = 7 S_{4}, prove:
S_{8}/S_{4} = 3
-/

theorem geometric_sequence_ratio {a_n : ℕ → ℝ} (q : ℝ) (h₁ : q ≠ -1) (h₂ : q ≠ 1)
  (S : ℕ → ℝ) (hSn : ∀ n, S n = a_n 0 * (1 - q ^ n) / (1 - q)) (h : S 12 = 7 * S 4) :
  S 8 / S 4 = 3 :=
by
  sorry

end geometric_sequence_ratio_l130_130506


namespace workers_production_l130_130638

theorem workers_production
    (x y : ℝ)
    (h1 : x + y = 72)
    (h2 : 1.15 * x + 1.25 * y = 86) :
    1.15 * x = 46 ∧ 1.25 * y = 40 :=
by {
  sorry
}

end workers_production_l130_130638


namespace jesse_started_with_l130_130035

-- Define the conditions
variables (g e : ℕ)

-- Theorem stating that given the conditions, Jesse started with 78 pencils
theorem jesse_started_with (g e : ℕ) (h1 : g = 44) (h2 : e = 34) : e + g = 78 :=
by sorry

end jesse_started_with_l130_130035


namespace general_term_of_sequence_l130_130796

theorem general_term_of_sequence (a : ℕ → ℝ)
  (h1 : a 1 = 1)
  (h2 : a 2 = 1)
  (h_rec : ∀ n ≥ 2, a (n + 1) = (n^2 * (a n)^2 + 5) / ((n^2 - 1) * a (n - 1))) :
  ∀ n : ℕ, a n = 
    if n = 0 then 0 else
    (1 / n) * ( (63 - 13 * Real.sqrt 21) / 42 * ((5 + Real.sqrt 21) / 2) ^ n + 
                (63 + 13 * Real.sqrt 21) / 42 * ((5 - Real.sqrt 21) / 2) ^ n) :=
by
  sorry

end general_term_of_sequence_l130_130796


namespace distance_midpoint_parabola_y_axis_l130_130685

theorem distance_midpoint_parabola_y_axis (x1 x2 : ℝ) (y1 y2 : ℝ) 
  (hA : y1 ^ 2 = x1) (hB : y2 ^ 2 = x2) 
  (h_focus : ∀ {p : ℝ × ℝ}, p = (x1, y1) ∨ p = (x2, y2) → |p.1 - 1/4| = |p.1 + 1/4|)
  (h_dist : |x1 - 1/4| + |x2 - 1/4| = 3) :
  abs ((x1 + x2) / 2) = 5 / 4 :=
by sorry

end distance_midpoint_parabola_y_axis_l130_130685


namespace geometric_sequence_solution_l130_130466

open Real

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n m, a (n + m) = a n * q ^ m

theorem geometric_sequence_solution :
  ∃ (a : ℕ → ℝ) (q : ℝ), geometric_sequence a q ∧
    (∀ n, 1 ≤ n ∧ n ≤ 5 → 10^8 ≤ a n ∧ a n < 10^9) ∧
    (∀ n, 6 ≤ n ∧ n ≤ 10 → 10^9 ≤ a n ∧ a n < 10^10) ∧
    (∀ n, 11 ≤ n ∧ n ≤ 14 → 10^10 ≤ a n ∧ a n < 10^11) ∧
    (∀ n, 15 ≤ n ∧ n ≤ 16 → 10^11 ≤ a n ∧ a n < 10^12) ∧
    (∀ i, a i = 7 * 3^(16-i) * 5^(i-1)) := sorry

end geometric_sequence_solution_l130_130466


namespace parallel_condition_l130_130639

theorem parallel_condition (a : ℝ) : (a = -2) ↔ (∀ x y : ℝ, ax + 2 * y = 0 → (-a / 2) = 1) :=
by
  sorry

end parallel_condition_l130_130639


namespace tip_percentage_is_30_l130_130404

theorem tip_percentage_is_30
  (appetizer_cost : ℝ)
  (entree_cost : ℝ)
  (num_entrees : ℕ)
  (dessert_cost : ℝ)
  (total_price_including_tip : ℝ)
  (h_appetizer : appetizer_cost = 9.0)
  (h_entree : entree_cost = 20.0)
  (h_num_entrees : num_entrees = 2)
  (h_dessert : dessert_cost = 11.0)
  (h_total : total_price_including_tip = 78.0) :
  let total_before_tip := appetizer_cost + num_entrees * entree_cost + dessert_cost
  let tip_amount := total_price_including_tip - total_before_tip
  let tip_percentage := (tip_amount / total_before_tip) * 100
  tip_percentage = 30 :=
by
  sorry

end tip_percentage_is_30_l130_130404


namespace arithmetic_sequence_sum_l130_130107

theorem arithmetic_sequence_sum :
  ∀ (a : ℕ → ℤ), (∀ n : ℕ, a (n+1) - a n = 2) → a 2 = 5 → (a 0 + a 1 + a 2 + a 3) = 24 :=
by
  sorry

end arithmetic_sequence_sum_l130_130107


namespace min_sum_of_squares_of_y_coords_l130_130166

def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def line_through_point (m : ℝ) (x y : ℝ) : Prop := x = m * y + 4

theorem min_sum_of_squares_of_y_coords :
  ∃ (m : ℝ), ∀ (x1 y1 x2 y2 : ℝ),
  (line_through_point m x1 y1) →
  (parabola x1 y1) →
  (line_through_point m x2 y2) →
  (parabola x2 y2) →
  x1 ≠ x2 → 
  ((y1 + y2)^2 - 2 * y1 * y2) = 32 :=
sorry

end min_sum_of_squares_of_y_coords_l130_130166


namespace maria_total_cost_l130_130152

-- Define the conditions as variables in the Lean environment
def daily_rental_rate : ℝ := 35
def mileage_rate : ℝ := 0.25
def rental_days : ℕ := 3
def miles_driven : ℕ := 500

-- Now, state the theorem that Maria’s total payment should be $230
theorem maria_total_cost : (daily_rental_rate * rental_days) + (mileage_rate * miles_driven) = 230 := 
by
  -- no proof required, just state as sorry
  sorry

end maria_total_cost_l130_130152


namespace find_e_l130_130266

theorem find_e (d e f : ℕ) (hd : d > 1) (he : e > 1) (hf : f > 1) :
  (∀ M : ℝ, M ≠ 1 → (M^(1/d) * (M^(1/e) * (M^(1/f)))^(1/e)^(1/d)) = (M^(17/24))^(1/24)) → e = 4 :=
by
  sorry

end find_e_l130_130266


namespace calculate_x_l130_130779

variable (a b x : ℝ)
variable (h1 : r = (3 * a) ^ (3 * b))
variable (h2 : r = a ^ b * x ^ b)
variable (h3 : x > 0)

theorem calculate_x (a b x : ℝ) (h1 : r = (3 * a) ^ (3 * b)) (h2 : r = a ^ b * x ^ b) (h3 : x > 0) : x = 27 * a ^ 2 := by
  sorry

end calculate_x_l130_130779


namespace necessary_and_sufficient_condition_l130_130303

def f (x a : ℝ) : ℝ := x^3 + 3 * a * x

def slope_tangent_at_one (a : ℝ) : ℝ := 3 * 1^2 + 3 * a

def are_perpendicular (a : ℝ) : Prop := -a = -1

theorem necessary_and_sufficient_condition (a : ℝ) :
  (slope_tangent_at_one a = 6) ↔ (are_perpendicular a) :=
by
  sorry

end necessary_and_sufficient_condition_l130_130303


namespace total_number_of_coins_is_324_l130_130572

noncomputable def total_coins (total_sum : ℕ) (coins_20p : ℕ) (coins_25p_value : ℕ) : ℕ :=
    coins_20p + (coins_25p_value / 25)

theorem total_number_of_coins_is_324 (h_sum: 7100 = 71 * 100) (h_coins_20p: 200 * 20 = 4000) :
  total_coins 7100 200 3100 = 324 := by
  sorry

end total_number_of_coins_is_324_l130_130572


namespace parts_of_alloys_l130_130215

def ratio_of_metals_in_alloy (a1 a2 a3 b1 b2 : ℚ) (x y : ℚ) : Prop :=
  let first_metal := (1 / a3) * x + (a1 / b2) * y
  let second_metal := (2 / a3) * x + (b1 / b2) * y
  (first_metal / second_metal) = (17 / 27)

theorem parts_of_alloys
  (x y : ℚ)
  (a1 a2 a3 b1 b2 : ℚ)
  (h1 : a1 = 1)
  (h2 : a2 = 2)
  (h3 : a3 = 3)
  (h4 : b1 = 2)
  (h5 : b2 = 5)
  (h6 : ratio_of_metals_in_alloy a1 a2 a3 b1 b2 x y) :
  x = 9 ∧ y = 35 :=
sorry

end parts_of_alloys_l130_130215


namespace arctan_addition_formula_l130_130946

noncomputable def arctan_add : ℝ :=
  Real.arctan (1 / 3) + Real.arctan (3 / 8)

theorem arctan_addition_formula :
  arctan_add = Real.arctan (17 / 21) :=
by
  sorry

end arctan_addition_formula_l130_130946


namespace part1_part2_l130_130185

noncomputable def f (x : ℝ) : ℝ := abs (2 * x - 1) - abs (x + 2)

theorem part1 {x : ℝ} : f x > 0 ↔ (x < -1 / 3 ∨ x > 3) := sorry

theorem part2 {m : ℝ} (h : ∃ x₀ : ℝ, f x₀ + 2 * m^2 < 4 * m) : -1 / 2 < m ∧ m < 5 / 2 := sorry

end part1_part2_l130_130185


namespace find_number_l130_130707

def single_digit (n : ℕ) : Prop := n < 10
def greater_than_zero (n : ℕ) : Prop := n > 0
def less_than_two (n : ℕ) : Prop := n < 2

theorem find_number (n : ℕ) : 
  single_digit n ∧ greater_than_zero n ∧ less_than_two n → n = 1 :=
by
  sorry

end find_number_l130_130707


namespace tangent_line_circle_l130_130987

theorem tangent_line_circle (m : ℝ) :
  (∀ x y : ℝ, x - 2*y + m = 0 ↔ (x^2 + y^2 - 4*x + 6*y + 8 = 0)) →
  m = -3 ∨ m = -13 :=
sorry

end tangent_line_circle_l130_130987


namespace distinct_real_roots_l130_130290

theorem distinct_real_roots (k : ℝ) :
  (∀ x : ℝ, (k - 2) * x^2 + 2 * x - 1 = 0 → ∃ y : ℝ, (k - 2) * y^2 + 2 * y - 1 = 0 ∧ y ≠ x) ↔
  (k > 1 ∧ k ≠ 2) := 
by sorry

end distinct_real_roots_l130_130290


namespace trajectory_is_line_segment_l130_130549

theorem trajectory_is_line_segment : 
  ∃ (P : ℝ × ℝ) (F1 F2: ℝ × ℝ), 
    F1 = (-3, 0) ∧ F2 = (3, 0) ∧ (|F1.1 - P.1|^2 + |F1.2 - P.2|^2).sqrt + (|F2.1 - P.1|^2 + |F2.2 - P.2|^2).sqrt = 6
  → (P.1 = F1.1 ∨ P.1 = F2.1) ∧ (P.2 = F1.2 ∨ P.2 = F2.2) :=
by sorry

end trajectory_is_line_segment_l130_130549


namespace intersection_M_N_l130_130929

def M (x : ℝ) : Prop := -2 < x ∧ x < 2
def N (x : ℝ) : Prop := |x - 1| ≤ 2

theorem intersection_M_N :
  {x : ℝ | M x ∧ N x} = {x : ℝ | -1 ≤ x ∧ x < 2} :=
sorry

end intersection_M_N_l130_130929


namespace june_walked_miles_l130_130650

theorem june_walked_miles
  (step_counter_reset : ℕ)
  (resets_per_year : ℕ)
  (final_steps : ℕ)
  (steps_per_mile : ℕ)
  (h1 : step_counter_reset = 100000)
  (h2 : resets_per_year = 52)
  (h3 : final_steps = 30000)
  (h4 : steps_per_mile = 2000) :
  (resets_per_year * step_counter_reset + final_steps) / steps_per_mile = 2615 := 
by 
  sorry

end june_walked_miles_l130_130650


namespace number_of_dimes_l130_130985

theorem number_of_dimes (x : ℕ) (h1 : 10 * x + 25 * x + 50 * x = 2040) : x = 24 :=
by {
  -- The proof will go here if you need to fill it out.
  sorry
}

end number_of_dimes_l130_130985


namespace negation_of_universal_proposition_l130_130660

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 - x ≤ 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - x > 0) :=
by
  sorry

end negation_of_universal_proposition_l130_130660


namespace speed_in_still_water_l130_130886

theorem speed_in_still_water (u d s : ℝ) (hu : u = 20) (hd : d = 60) (hs : s = (u + d) / 2) : s = 40 := 
by 
  sorry

end speed_in_still_water_l130_130886


namespace sum_of_n_l130_130553

theorem sum_of_n (n : ℤ) (h : (36 : ℤ) % (2 * n - 1) = 0) :
  (n = 1 ∨ n = 2 ∨ n = 5) → 1 + 2 + 5 = 8 :=
by
  intros hn
  have h1 : n = 1 ∨ n = 2 ∨ n = 5 := hn
  sorry

end sum_of_n_l130_130553


namespace Bryan_has_more_skittles_l130_130479

-- Definitions for conditions
def Bryan_skittles : ℕ := 50
def Ben_mms : ℕ := 20

-- Main statement to be proven
theorem Bryan_has_more_skittles : Bryan_skittles > Ben_mms ∧ Bryan_skittles - Ben_mms = 30 :=
by
  sorry

end Bryan_has_more_skittles_l130_130479


namespace find_x_l130_130897

theorem find_x (x : ℝ) (h : 6 * x + 7 * x + 3 * x + 2 * x + 4 * x = 360) : 
  x = 180 / 11 := 
by
  sorry

end find_x_l130_130897


namespace line_through_A_parallel_y_axis_l130_130088

theorem line_through_A_parallel_y_axis (x y: ℝ) (A: ℝ × ℝ) (h1: A = (-3, 1)) : 
  (∀ P: ℝ × ℝ, P ∈ {p : ℝ × ℝ | p.1 = -3} → (P = A ∨ P.1 = -3)) :=
by
  sorry

end line_through_A_parallel_y_axis_l130_130088


namespace max_ratio_xy_l130_130930

def two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem max_ratio_xy (x y : ℕ) (hx : two_digit x) (hy : two_digit y) (hmean : (x + y) / 2 = 60) : x / y ≤ 33 / 7 :=
by
  sorry

end max_ratio_xy_l130_130930


namespace truck_distance_l130_130018

theorem truck_distance (V_t : ℝ) (D : ℝ) (h1 : D = V_t * 8) (h2 : D = (V_t + 18) * 5) : D = 240 :=
by
  sorry

end truck_distance_l130_130018


namespace friends_gcd_l130_130973

theorem friends_gcd {a b : ℤ} (h : ∃ n : ℤ, a * b = n * n) : 
  ∃ m : ℤ, a * Int.gcd a b = m * m :=
sorry

end friends_gcd_l130_130973


namespace cars_with_neither_l130_130101

theorem cars_with_neither (total_cars air_bag power_windows both : ℕ) 
                          (h1 : total_cars = 65) (h2 : air_bag = 45)
                          (h3 : power_windows = 30) (h4 : both = 12) : 
                          (total_cars - (air_bag + power_windows - both) = 2) :=
by
  sorry

end cars_with_neither_l130_130101


namespace sqrt_expression_equals_l130_130529

theorem sqrt_expression_equals : Real.sqrt (5^2 * 7^4) = 245 :=
by
  sorry

end sqrt_expression_equals_l130_130529


namespace product_of_two_numbers_l130_130842

theorem product_of_two_numbers (x y : ℝ) 
  (h1 : x + y = 60) 
  (h2 : x - y = 16) : 
  x * y = 836 := 
by
  sorry

end product_of_two_numbers_l130_130842


namespace units_digit_of_k_squared_plus_2_to_k_l130_130389

theorem units_digit_of_k_squared_plus_2_to_k (k : ℕ) (h : k = 2012 ^ 2 + 2 ^ 2014) : (k ^ 2 + 2 ^ k) % 10 = 5 := by
  sorry

end units_digit_of_k_squared_plus_2_to_k_l130_130389


namespace find_a_20_l130_130423

-- Arithmetic sequence definition and known conditions
def arithmetic_sequence (a : ℕ → ℤ) :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

variable (a : ℕ → ℤ)

-- Conditions
def condition1 : Prop := a 1 + a 2 + a 3 = 6
def condition2 : Prop := a 5 = 8

-- The main statement to prove
theorem find_a_20 (h_arith : arithmetic_sequence a) (h_cond1 : condition1 a) (h_cond2 : condition2 a) : 
  a 20 = 38 := by
  sorry

end find_a_20_l130_130423


namespace right_triangle_arithmetic_sequence_side_length_l130_130298

theorem right_triangle_arithmetic_sequence_side_length :
  ∃ (a b c : ℕ), (a < b ∧ b < c) ∧ (b - a = c - b) ∧ (a^2 + b^2 = c^2) ∧ (b = 81) :=
sorry

end right_triangle_arithmetic_sequence_side_length_l130_130298


namespace logical_equivalence_l130_130002

variable (R S T : Prop)

theorem logical_equivalence :
  (R → ¬S ∧ ¬T) ↔ ((S ∨ T) → ¬R) :=
by
  sorry

end logical_equivalence_l130_130002


namespace blipblish_modulo_l130_130807

-- Definitions from the conditions
inductive Letter
| B | I | L

def is_consonant (c : Letter) : Bool :=
  match c with
  | Letter.B | Letter.L => true
  | _ => false

def is_vowel (v : Letter) : Bool :=
  match v with
  | Letter.I => true
  | _ => false

def is_valid_blipblish_word (word : List Letter) : Bool :=
  -- Check if between any two I's there at least three consonants
  let rec check (lst : List Letter) (cnt : Nat) (during_vowels : Bool) : Bool :=
    match lst with
    | [] => true
    | Letter.I :: xs =>
        if during_vowels then cnt >= 3 && check xs 0 false
        else check xs 0 true
    | x :: xs =>
        if is_consonant x then check xs (cnt + 1) during_vowels
        else check xs cnt during_vowels
  check word 0 false

def number_of_valid_words (n : Nat) : Nat :=
  -- Placeholder function to compute the number of valid Blipblish words of length n
  sorry

-- Statement of the proof problem
theorem blipblish_modulo : number_of_valid_words 12 % 1000 = 312 :=
by sorry

end blipblish_modulo_l130_130807


namespace tan_alpha_plus_pi_over_4_l130_130351

theorem tan_alpha_plus_pi_over_4 
  (α β : ℝ)
  (h1 : Real.tan (α + β) = 2 / 5)
  (h2 : Real.tan (β - Real.pi / 4) = 1 / 4) :
  Real.tan (α + Real.pi / 4) = 3 / 22 :=
sorry

end tan_alpha_plus_pi_over_4_l130_130351


namespace cars_on_river_road_l130_130679

theorem cars_on_river_road (B C : ℕ) (h1 : B = C - 60) (h2 : B * 13 = C) : C = 65 :=
sorry

end cars_on_river_road_l130_130679


namespace original_dining_bill_l130_130214

theorem original_dining_bill (B : ℝ) (h1 : B * 1.15 / 5 = 48.53) : B = 211 := 
sorry

end original_dining_bill_l130_130214


namespace bucket_p_fill_time_l130_130070

theorem bucket_p_fill_time (capacity_P capacity_Q drum_capacity turns : ℕ)
  (h1 : capacity_P = 3 * capacity_Q)
  (h2 : drum_capacity = 45 * (capacity_P + capacity_Q))
  (h3 : bucket_fill_turns = drum_capacity / capacity_P) :
  bucket_fill_turns = 60 :=
by
  sorry

end bucket_p_fill_time_l130_130070


namespace prove_inequality_l130_130432

noncomputable def inequality_problem :=
  ∀ (x y z : ℝ),
    0 < x ∧ 0 < y ∧ 0 < z ∧ x^2 + y^2 + z^2 = 3 → 
      (x ^ 2009 - 2008 * (x - 1)) / (y + z) + 
      (y ^ 2009 - 2008 * (y - 1)) / (x + z) + 
      (z ^ 2009 - 2008 * (z - 1)) / (x + y) ≥ 
      (x + y + z) / 2

theorem prove_inequality : inequality_problem := 
  by 
    sorry

end prove_inequality_l130_130432


namespace kit_costs_more_l130_130646

-- Defining the individual prices of the filters and the kit price
def price_filter1 := 16.45
def price_filter2 := 14.05
def price_filter3 := 19.50
def kit_price := 87.50

-- Calculating the total price of the filters if bought individually
def total_individual_price := (2 * price_filter1) + (2 * price_filter2) + price_filter3

-- Calculate the amount saved
def amount_saved := total_individual_price - kit_price

-- The theorem to show the amount saved 
theorem kit_costs_more : amount_saved = -7.00 := by
  sorry

end kit_costs_more_l130_130646


namespace number_of_cows_in_farm_l130_130810

-- Definitions relating to the conditions
def total_bags_consumed := 20
def bags_per_cow := 1
def days := 20

-- Question and proof of the answer
theorem number_of_cows_in_farm : (total_bags_consumed / bags_per_cow) = 20 := by
  -- proof goes here
  sorry

end number_of_cows_in_farm_l130_130810


namespace overall_average_score_l130_130276

def students_monday := 24
def students_tuesday := 4
def total_students := 28
def mean_score_monday := 82
def mean_score_tuesday := 90

theorem overall_average_score :
  (students_monday * mean_score_monday + students_tuesday * mean_score_tuesday) / total_students = 83 := by
sorry

end overall_average_score_l130_130276


namespace last_digit_sum_chessboard_segments_l130_130258

theorem last_digit_sum_chessboard_segments {N : ℕ} (tile_count : ℕ) (segment_count : ℕ := 112) (dominos_per_tiling : ℕ := 32) (segments_per_domino : ℕ := 2) (N := tile_count / N) :
  (80 * N) % 10 = 0 :=
by
  sorry

end last_digit_sum_chessboard_segments_l130_130258


namespace m_is_perfect_square_l130_130482

theorem m_is_perfect_square
  (m n k : ℕ) 
  (h1 : 0 < m) 
  (h2 : 0 < n) 
  (h3 : 0 < k) 
  (h4 : 1 + m + n * Real.sqrt 3 = (2 + Real.sqrt 3) ^ (2 * k + 1)) : 
  ∃ a : ℕ, m = a ^ 2 :=
by 
  sorry

end m_is_perfect_square_l130_130482


namespace hair_cut_first_day_l130_130763

theorem hair_cut_first_day 
  (total_hair_cut : ℝ) 
  (hair_cut_second_day : ℝ) 
  (h_total : total_hair_cut = 0.875) 
  (h_second : hair_cut_second_day = 0.5) : 
  total_hair_cut - hair_cut_second_day = 0.375 := 
  by
  simp [h_total, h_second]
  sorry

end hair_cut_first_day_l130_130763


namespace chocolate_bar_cost_l130_130702

theorem chocolate_bar_cost 
  (x : ℝ)  -- cost of each bar in dollars
  (total_bars : ℕ)  -- total number of bars in the box
  (sold_bars : ℕ)  -- number of bars sold
  (amount_made : ℝ)  -- amount made in dollars
  (h1 : total_bars = 9)  -- condition: total bars in the box is 9
  (h2 : sold_bars = total_bars - 3)  -- condition: Wendy sold all but 3 bars
  (h3 : amount_made = 18)  -- condition: Wendy made $18
  (h4 : amount_made = sold_bars * x)  -- condition: amount made from selling sold bars
  : x = 3 := 
sorry

end chocolate_bar_cost_l130_130702


namespace E_runs_is_20_l130_130332

-- Definitions of runs scored by each batsman as multiples of 4
def a := 28
def e := 20
def d := e + 12
def b := d + e
def c := 107 - b
def total_runs := a + b + c + d + e

-- Adding conditions
axiom A_max: a > b ∧ a > c ∧ a > d ∧ a > e
axiom runs_multiple_of_4: ∀ (x : ℕ), x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e → x % 4 = 0
axiom average_runs: total_runs = 180
axiom d_condition: d = e + 12
axiom e_condition: e = a - 8
axiom b_condition: b = d + e
axiom bc_condition: b + c = 107

theorem E_runs_is_20 : e = 20 := by
  sorry

end E_runs_is_20_l130_130332


namespace find_x_value_l130_130097

theorem find_x_value (b x : ℝ) (hb : b > 1) (hx : x > 0) 
    (heq: (3 * x)^(Real.log 3 / Real.log b) - (5 * x)^(Real.log 5 / Real.log b) = 0) : 
    x = 1 / 5 := 
sorry

end find_x_value_l130_130097


namespace arrange_books_l130_130617

-- We define the conditions about the number of books
def num_algebra_books : ℕ := 4
def num_calculus_books : ℕ := 5
def total_books : ℕ := num_algebra_books + num_calculus_books

-- The combination function which calculates binomial coefficients
def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The theorem stating that there are 126 ways to arrange the books
theorem arrange_books : combination total_books num_algebra_books = 126 :=
  by
    sorry

end arrange_books_l130_130617


namespace average_bull_weight_l130_130626

def ratioA : ℚ := 7 / 28  -- Ratio of cows to total cattle in section A
def ratioB : ℚ := 5 / 20  -- Ratio of cows to total cattle in section B
def ratioC : ℚ := 3 / 12  -- Ratio of cows to total cattle in section C

def total_cattle : ℕ := 1220  -- Total cattle on the farm
def total_bull_weight : ℚ := 200000  -- Total weight of bulls in kg

theorem average_bull_weight :
  ratioA = 7 / 28 ∧
  ratioB = 5 / 20 ∧
  ratioC = 3 / 12 ∧
  total_cattle = 1220 ∧
  total_bull_weight = 200000 →
  ∃ avg_weight : ℚ, avg_weight = 218.579 :=
sorry

end average_bull_weight_l130_130626


namespace inequality_correct_l130_130146

variable {a b c : ℝ}

theorem inequality_correct (h : a * b < 0) : |a - c| ≤ |a - b| + |b - c| :=
sorry

end inequality_correct_l130_130146


namespace pencil_cost_l130_130557

theorem pencil_cost (P : ℝ) (h1 : 24 * P + 18 = 30) : P = 0.5 :=
by
  sorry

end pencil_cost_l130_130557


namespace complex_imag_part_of_z_l130_130882

theorem complex_imag_part_of_z (z : ℂ) (h : z * (2 + ⅈ) = 3 - 6 * ⅈ) : z.im = -3 := by
  sorry

end complex_imag_part_of_z_l130_130882


namespace true_propositions_l130_130264

-- Defining the propositions as functions for clarity
def proposition1 (L1 L2 P: Prop) : Prop := 
  (L1 ∧ L2 → P) → (P)

def proposition2 (plane1 plane2 line: Prop) : Prop := 
  (line → (plane1 ∧ plane2)) → (plane1 ∧ plane2)

def proposition3 (L1 L2 L3: Prop) : Prop := 
  (L1 ∧ L2 → L3) → L1

def proposition4 (plane1 plane2 line: Prop) : Prop := 
  (plane1 ∧ plane2 → (line → ¬ (plane1 ∧ plane2)))

-- Assuming the required mathematical hypothesis was valid within our formal system 
theorem true_propositions : proposition2 plane1 plane2 line ∧ proposition4 plane1 plane2 line := 
by sorry

end true_propositions_l130_130264


namespace question_true_l130_130405
noncomputable def a := (1/2) * Real.cos (7 * Real.pi / 180) - (Real.sqrt 3 / 2) * Real.sin (7 * Real.pi / 180)
noncomputable def b := (2 * Real.tan (12 * Real.pi / 180)) / (1 + Real.tan (12 * Real.pi / 180)^2)
noncomputable def c := Real.sqrt ((1 - Real.cos (44 * Real.pi / 180)) / 2)

theorem question_true :
  b > a ∧ a > c :=
by
  sorry

end question_true_l130_130405


namespace product_B_original_price_l130_130493

variable (a b : ℝ)

theorem product_B_original_price (h1 : a = 1.2 * b) (h2 : 0.9 * a = 198) : b = 183.33 :=
by
  sorry

end product_B_original_price_l130_130493


namespace elias_purchased_50cent_items_l130_130379

theorem elias_purchased_50cent_items :
  ∃ (a b c : ℕ), a + b + c = 50 ∧ (50 * a + 250 * b + 400 * c = 5000) ∧ (a = 40) :=
by {
  sorry
}

end elias_purchased_50cent_items_l130_130379


namespace union_of_sets_l130_130951

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B : Set ℝ := {x | 1 < x ∧ x < 3}

-- State the proof problem
theorem union_of_sets : A ∪ B = {x | -1 < x ∧ x < 3} := by
  sorry

end union_of_sets_l130_130951


namespace notebook_cost_l130_130867

theorem notebook_cost
  (students : ℕ)
  (majority_students : ℕ)
  (cost : ℕ)
  (notebooks : ℕ)
  (h1 : students = 36)
  (h2 : majority_students > 18)
  (h3 : notebooks > 1)
  (h4 : cost > notebooks)
  (h5 : majority_students * cost * notebooks = 2079) :
  cost = 11 :=
by
  sorry

end notebook_cost_l130_130867


namespace sqrt_sum_eq_six_l130_130766

theorem sqrt_sum_eq_six (x : ℝ) :
  (Real.sqrt ((3 + 2 * Real.sqrt 2) ^ x) + Real.sqrt ((3 - 2 * Real.sqrt 2) ^ x) = 6) ↔ (x = 2 ∨ x = -2) := 
sorry

end sqrt_sum_eq_six_l130_130766


namespace not_q_true_l130_130464

theorem not_q_true (p q : Prop) (hp : p = true) (hq : q = false) : ¬q = true :=
by
  sorry

end not_q_true_l130_130464


namespace area_ratio_l130_130492

variables {A B C D: Type} [LinearOrderedField A]
variables {AB AD AR AE : A}

-- Conditions
axiom cond1 : AR = (2 / 3) * AB
axiom cond2 : AE = (1 / 3) * AD

theorem area_ratio (h : A) (h1 : A) (S_ABCD : A) (S_ARE : A)
  (h_eq : S_ABCD = AD * h)
  (h1_eq : S_ARE = (1 / 2) * AE * h1)
  (ratio_heights : h / h1 = 3 / 2) :
  S_ABCD / S_ARE = 9 :=
by {
  sorry
}

end area_ratio_l130_130492


namespace value_of_f_at_2_l130_130192

def f (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 + 5 * x - 4

theorem value_of_f_at_2 : f 2 = 62 :=
by
  -- The proof will be inserted here, it follows Horner's method steps shown in the solution
  sorry

end value_of_f_at_2_l130_130192


namespace neither_probability_l130_130534

-- Definitions of the probabilities P(A), P(B), and P(A ∩ B)
def P_A : ℝ := 0.63
def P_B : ℝ := 0.49
def P_A_and_B : ℝ := 0.32

-- Definition stating the probability of neither event
theorem neither_probability :
  (1 - (P_A + P_B - P_A_and_B)) = 0.20 := 
sorry

end neither_probability_l130_130534


namespace integer_mod_105_l130_130497

theorem integer_mod_105 (x : ℤ) :
  (4 + x ≡ 2 * 2 [ZMOD 3^3]) →
  (6 + x ≡ 3 * 3 [ZMOD 5^3]) →
  (8 + x ≡ 5 * 5 [ZMOD 7^3]) →
  x % 105 = 3 :=
by
  sorry

end integer_mod_105_l130_130497


namespace mailman_total_pieces_l130_130523

def piecesOfMailFirstHouse := 6 + 5 + 3 + 4 + 2
def piecesOfMailSecondHouse := 4 + 7 + 2 + 5 + 3
def piecesOfMailThirdHouse := 8 + 3 + 4 + 6 + 1

def totalPiecesOfMail := piecesOfMailFirstHouse + piecesOfMailSecondHouse + piecesOfMailThirdHouse

theorem mailman_total_pieces : totalPiecesOfMail = 63 := by
  sorry

end mailman_total_pieces_l130_130523


namespace triangle_area_l130_130719

theorem triangle_area (a b c : ℝ) (h1 : a = 9) (h2 : b = 40) (h3 : c = 41) (h4 : c^2 = a^2 + b^2) :
  (1 / 2) * a * b = 180 :=
by
  sorry

end triangle_area_l130_130719


namespace second_customer_headphones_l130_130583

theorem second_customer_headphones
  (H : ℕ)
  (M : ℕ)
  (x : ℕ)
  (H_eq : H = 30)
  (eq1 : 5 * M + 8 * H = 840)
  (eq2 : 3 * M + x * H = 480) :
  x = 4 :=
by
  sorry

end second_customer_headphones_l130_130583


namespace gumball_probability_l130_130533

theorem gumball_probability :
  let total_gumballs : ℕ := 25
  let orange_gumballs : ℕ := 10
  let green_gumballs : ℕ := 6
  let yellow_gumballs : ℕ := 9
  let total_gumballs_after_first : ℕ := total_gumballs - 1
  let total_gumballs_after_second : ℕ := total_gumballs - 2
  let orange_probability_first : ℚ := orange_gumballs / total_gumballs
  let green_or_yellow_probability_second : ℚ := (green_gumballs + yellow_gumballs) / total_gumballs_after_first
  let orange_probability_third : ℚ := (orange_gumballs - 1) / total_gumballs_after_second
  orange_probability_first * green_or_yellow_probability_second * orange_probability_third = 9 / 92 :=
by
  sorry

end gumball_probability_l130_130533


namespace arithmetic_sequence_n_is_17_l130_130889

theorem arithmetic_sequence_n_is_17
  (a : ℕ → ℤ)  -- An arithmetic sequence a_n
  (h1 : a 1 = 5)  -- First term is 5
  (h5 : a 5 = -3)  -- Fifth term is -3
  (hn : a n = -27) : n = 17 := sorry

end arithmetic_sequence_n_is_17_l130_130889


namespace irrationals_l130_130403

open Classical

variable (x : ℝ)

theorem irrationals (h : x^3 + 2 * x^2 + 10 * x = 20) : Irrational x ∧ Irrational (x^2) :=
by
  sorry

end irrationals_l130_130403


namespace distance_to_store_l130_130203

noncomputable def D : ℝ := 4

theorem distance_to_store :
  (1/3) * (D/2 + D/10 + D/10) = 56/60 :=
by
  sorry

end distance_to_store_l130_130203


namespace sum_of_angles_of_parallelepiped_diagonal_lt_pi_l130_130776

/-- In a rectangular parallelepiped, if the main diagonal forms angles α, β, and γ with the three edges meeting at a vertex, then the sum of these angles is less than π. -/
theorem sum_of_angles_of_parallelepiped_diagonal_lt_pi {α β γ : ℝ} (hα : 0 < α) (hβ : 0 < β) (hγ : 0 < γ)
  (h_sum : 2 * α + 2 * β + 2 * γ < 2 * π) :
  α + β + γ < π := by
sorry

end sum_of_angles_of_parallelepiped_diagonal_lt_pi_l130_130776


namespace find_sp_l130_130442

theorem find_sp (s p : ℝ) (t x y : ℝ) (h1 : x = 3 + 5 * t) (h2 : y = 3 + p * t) 
  (h3 : y = 4 * x - 9) : 
  s = 3 ∧ p = 20 := 
by
  -- Proof goes here
  sorry

end find_sp_l130_130442


namespace smallest_PR_minus_QR_l130_130335

theorem smallest_PR_minus_QR :
  ∃ (PQ QR PR : ℤ), 
    PQ + QR + PR = 2023 ∧ PQ ≤ QR ∧ QR < PR ∧ PR - QR = 13 :=
by
  sorry

end smallest_PR_minus_QR_l130_130335


namespace adi_baller_prob_l130_130716

theorem adi_baller_prob (a b : ℕ) (p : ℝ) (h_prime: Nat.Prime a) (h_pos_b: 0 < b)
  (h_p: p = (1 / 2) ^ (1 / 35)) : a + b = 37 :=
sorry

end adi_baller_prob_l130_130716


namespace pure_imaginary_solution_l130_130653

theorem pure_imaginary_solution (b : ℝ) (z : ℂ) 
  (H : z = (b + Complex.I) / (2 + Complex.I))
  (H_imaginary : z.im = z ∧ z.re = 0) :
  b = -1 / 2 := 
by 
  sorry

end pure_imaginary_solution_l130_130653


namespace simplify_expression_l130_130921

variable (a b : ℝ)

theorem simplify_expression : 
  (a^(2/3) * b^(1/2)) * (-3 * a^(1/2) * b^(1/3)) / (1/3 * a^(1/6) * b^(5/6)) = -9 * a := 
  sorry

end simplify_expression_l130_130921


namespace problem_statement_l130_130667

variables (x y : ℝ)

def p : Prop := x > 1 ∧ y > 1
def q : Prop := x + y > 2

theorem problem_statement : (p x y → q x y) ∧ ¬(q x y → p x y) := sorry

end problem_statement_l130_130667


namespace proof_statement_l130_130120

noncomputable def problem_statement (a b : ℤ) : ℤ :=
  (a^3 + b^3) / (a^2 - a * b + b^2)

theorem proof_statement : problem_statement 5 4 = 9 := by
  sorry

end proof_statement_l130_130120


namespace min_abc_sum_l130_130487

theorem min_abc_sum (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 8) : a + b + c ≥ 6 :=
by {
  sorry
}

end min_abc_sum_l130_130487


namespace tan_zero_l130_130548

theorem tan_zero : Real.tan 0 = 0 := 
by
  sorry

end tan_zero_l130_130548


namespace proportional_sets_l130_130997

/-- Prove that among the sets of line segments, the ones that are proportional are: -/
theorem proportional_sets : 
  let A := (3, 6, 7, 9)
  let B := (2, 5, 6, 8)
  let C := (3, 6, 9, 18)
  let D := (1, 2, 3, 4)
  ∃ a b c d, (a, b, c, d) = C ∧ (a * d = b * c) :=
by
  let A := (3, 6, 7, 9)
  let B := (2, 5, 6, 8)
  let C := (3, 6, 9, 18)
  let D := (1, 2, 3, 4)
  sorry

end proportional_sets_l130_130997


namespace plus_one_eq_next_plus_l130_130826

theorem plus_one_eq_next_plus (m : ℕ) (h : m > 1) : (m^2 + m) + 1 = ((m + 1)^2 + (m + 1)) := by
  sorry

end plus_one_eq_next_plus_l130_130826


namespace crickets_total_l130_130830

noncomputable def initial_amount : ℝ := 7.5
noncomputable def additional_amount : ℝ := 11.25
noncomputable def total_amount : ℝ := 18.75

theorem crickets_total : initial_amount + additional_amount = total_amount :=
by
  sorry

end crickets_total_l130_130830


namespace range_of_a_l130_130586

theorem range_of_a (a : ℝ) : 
  (∀ x y z : ℝ, x^2 + y^2 + z^2 = 1 → |a - 1| ≥ x + 2 * y + 2 * z) →
  a ∈ Set.Iic (-2) ∪ Set.Ici 4 :=
by
  sorry

end range_of_a_l130_130586


namespace problem2_l130_130083

noncomputable def problem1 (a b c : ℝ) (A B C : ℝ) (h1 : 2 * (Real.sin A)^2 + (Real.sin B)^2 = (Real.sin C)^2)
    (h2 : b = 2 * a) (h3 : a = 2) : (1 / 2) * a * b * Real.sin C = Real.sqrt 15 :=
by
  sorry

theorem problem2 (a b c : ℝ) (h : 2 * a^2 + b^2 = c^2) :
  ∃ m : ℝ, (m = 2 * Real.sqrt 2) ∧ (∀ x y z : ℝ, 2 * x^2 + y^2 = z^2 → (z^2 / (x * y)) ≥ m) ∧ ((c / a) = 2) :=
by
  sorry

end problem2_l130_130083


namespace complement_of_A_l130_130044

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {1, 3, 5, 7}

theorem complement_of_A : U \ A = {2, 4, 6} := 
by 
  sorry

end complement_of_A_l130_130044


namespace no_solution_for_floor_x_plus_x_eq_15_point_3_l130_130073

theorem no_solution_for_floor_x_plus_x_eq_15_point_3 : ¬ ∃ (x : ℝ), (⌊x⌋ : ℝ) + x = 15.3 := by
  sorry

end no_solution_for_floor_x_plus_x_eq_15_point_3_l130_130073


namespace injective_of_comp_injective_surjective_of_comp_surjective_l130_130501

section FunctionProperties

variables {X Y V : Type} (f : X → Y) (g : Y → V)

-- Proof for part (i) if g ∘ f is injective, then f is injective
theorem injective_of_comp_injective (h : Function.Injective (g ∘ f)) : Function.Injective f :=
  sorry

-- Proof for part (ii) if g ∘ f is surjective, then g is surjective
theorem surjective_of_comp_surjective (h : Function.Surjective (g ∘ f)) : Function.Surjective g :=
  sorry

end FunctionProperties

end injective_of_comp_injective_surjective_of_comp_surjective_l130_130501


namespace even_function_derivative_at_zero_l130_130944

variable (f : ℝ → ℝ)
variable (hf_even : ∀ x, f x = f (-x))
variable (hf_diff : Differentiable ℝ f)

theorem even_function_derivative_at_zero : deriv f 0 = 0 :=
by 
  -- proof omitted
  sorry

end even_function_derivative_at_zero_l130_130944


namespace distance_between_points_l130_130787

theorem distance_between_points :
  let (x1, y1) := (1, 2)
  let (x2, y2) := (6, 5)
  let d := Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)
  d = Real.sqrt 34 :=
by
  sorry

end distance_between_points_l130_130787


namespace smallest_q_exists_l130_130817

theorem smallest_q_exists (p q : ℕ) (h : 0 < q) (h_eq : (p : ℚ) / q = 123456789 / 100000000000) :
  q = 10989019 :=
sorry

end smallest_q_exists_l130_130817


namespace ratio_of_A_to_B_l130_130474

-- Definitions of the conditions.
def amount_A : ℕ := 200
def total_amount : ℕ := 600
def amount_B : ℕ := total_amount - amount_A

-- The proof statement.
theorem ratio_of_A_to_B :
  amount_A / amount_B = 1 / 2 := 
sorry

end ratio_of_A_to_B_l130_130474


namespace buns_distribution_not_equal_for_all_cases_l130_130354

theorem buns_distribution_not_equal_for_all_cases :
  ∀ (initial_buns : Fin 30 → ℕ),
  (∃ (p : ℕ → Fin 30 → Fin 30), 
    (∀ t, 
      (∀ i, 
        (initial_buns (p t i) = initial_buns i ∨ 
         initial_buns (p t i) = initial_buns i + 2 ∨ 
         initial_buns (p t i) = initial_buns i - 2))) → 
    ¬ ∀ n : Fin 30, initial_buns n = 2) := 
sorry

end buns_distribution_not_equal_for_all_cases_l130_130354


namespace ways_to_divide_day_l130_130914

theorem ways_to_divide_day : 
  ∃ nm_count: ℕ, nm_count = 72 ∧ ∀ n m: ℕ, 0 < n ∧ 0 < m ∧ n * m = 72000 → 
  ∃ nm_pairs: ℕ, nm_pairs = 72 * 2 :=
sorry

end ways_to_divide_day_l130_130914


namespace find_distance_MF_l130_130139

-- Define the parabola and point conditions
def parabola (x y : ℝ) := y^2 = 8 * x

-- Define the focus of the parabola
def F : ℝ × ℝ := (2, 0)

-- Define the origin
def O : ℝ × ℝ := (0, 0)

-- Define the distance squared between two points
def dist_squared (A B : ℝ × ℝ) : ℝ :=
  (A.1 - B.1)^2 + (A.2 - B.2)^2

-- Prove the required statement
theorem find_distance_MF (x y : ℝ) (hM : parabola x y) (h_dist: dist_squared (x, y) O = 3 * (x + 2)) :
  dist_squared (x, y) F = 9 := by
  sorry

end find_distance_MF_l130_130139


namespace total_pencils_l130_130576

theorem total_pencils (pencils_per_child : ℕ) (children : ℕ) (h1 : pencils_per_child = 2) (h2 : children = 11) : (pencils_per_child * children = 22) := 
by
  sorry

end total_pencils_l130_130576


namespace solve_for_y_l130_130161

theorem solve_for_y (y : ℝ) (h : 5 * (y ^ (1/3)) - 3 * (y / (y ^ (2/3))) = 10 + (y ^ (1/3))) :
  y = 1000 :=
by {
  sorry
}

end solve_for_y_l130_130161


namespace find_ab_and_m_l130_130871

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 + b * x^2

theorem find_ab_and_m (a b m : ℝ) (P : ℝ × ℝ)
  (h1 : P = (-1, -2))
  (h2 : ∀ (x : ℝ), (3 * a * x^2 + 2 * b * x) = -1/3 ↔ x = -1)
  (h3 : ∀ (x : ℝ), f a b x = a * x ^ 3 + b * x ^ 2)
  : (a = -13/3 ∧ b = -19/3) ∧ (0 < m ∧ m < 38/39) :=
sorry

end find_ab_and_m_l130_130871


namespace gift_cost_l130_130728

theorem gift_cost (C F : ℕ) (hF : F = 15) (h_eq : C / (F - 4) = C / F + 12) : C = 495 :=
by
  -- Using the conditions given, we need to show that C computes to 495.
  -- Details are skipped using sorry.
  sorry

end gift_cost_l130_130728


namespace find_s_for_g_neg1_zero_l130_130099

def g (x s : ℝ) : ℝ := 3 * x^4 + x^3 - 2 * x^2 - 4 * x + s

theorem find_s_for_g_neg1_zero (s : ℝ) : g (-1) s = 0 ↔ s = -4 := by
  sorry

end find_s_for_g_neg1_zero_l130_130099


namespace max_ladder_height_reached_l130_130689

def distance_from_truck_to_building : ℕ := 5
def ladder_extension : ℕ := 13

theorem max_ladder_height_reached :
  (ladder_extension ^ 2 - distance_from_truck_to_building ^ 2) = 144 :=
by
  -- This is where the proof should go
  sorry

end max_ladder_height_reached_l130_130689


namespace max_s_value_l130_130778

theorem max_s_value (p q r s : ℝ) (h1 : p + q + r + s = 10) (h2 : (p * q) + (p * r) + (p * s) + (q * r) + (q * s) + (r * s) = 20) :
  s ≤ (5 * (1 + Real.sqrt 21)) / 2 :=
sorry

end max_s_value_l130_130778


namespace savings_calculation_l130_130964

noncomputable def weekly_rate_peak : ℕ := 10
noncomputable def weekly_rate_non_peak : ℕ := 8
noncomputable def monthly_rate_peak : ℕ := 40
noncomputable def monthly_rate_non_peak : ℕ := 35
noncomputable def non_peak_duration_weeks : ℝ := 17.33
noncomputable def peak_duration_weeks : ℝ := 52 - non_peak_duration_weeks
noncomputable def non_peak_duration_months : ℕ := 4
noncomputable def peak_duration_months : ℕ := 12 - non_peak_duration_months

noncomputable def total_weekly_cost := (non_peak_duration_weeks * weekly_rate_non_peak) 
                                     + (peak_duration_weeks * weekly_rate_peak)

noncomputable def total_monthly_cost := (non_peak_duration_months * monthly_rate_non_peak) 
                                      + (peak_duration_months * monthly_rate_peak)

noncomputable def savings := total_weekly_cost - total_monthly_cost

theorem savings_calculation 
  : savings = 25.34 := by
  sorry

end savings_calculation_l130_130964


namespace good_games_count_l130_130865

theorem good_games_count :
  ∀ (g1 g2 b : ℕ), g1 = 50 → g2 = 27 → b = 74 → g1 + g2 - b = 3 := by
  intros g1 g2 b hg1 hg2 hb
  sorry

end good_games_count_l130_130865


namespace sequence_periodic_l130_130799

theorem sequence_periodic (a : ℕ → ℝ) (h1 : a 1 = 0) (h2 : ∀ n, a n + a (n + 1) = 2) : a 2011 = 0 := by
  sorry

end sequence_periodic_l130_130799


namespace line_segment_is_symmetric_l130_130661

def is_axial_symmetric (shape : Type) : Prop := sorry
def is_central_symmetric (shape : Type) : Prop := sorry

def equilateral_triangle : Type := sorry
def isosceles_triangle : Type := sorry
def parallelogram : Type := sorry
def line_segment : Type := sorry

theorem line_segment_is_symmetric : 
  is_axial_symmetric line_segment ∧ is_central_symmetric line_segment := 
by
  sorry

end line_segment_is_symmetric_l130_130661


namespace volume_of_tetrahedron_OABC_l130_130138

-- Definitions of side lengths and their squared values
def side_length_A_B := 7
def side_length_B_C := 8
def side_length_C_A := 9

-- Squared values of coordinates
def a_sq := 33
def b_sq := 16
def c_sq := 48

-- Main statement to prove the volume
theorem volume_of_tetrahedron_OABC :
  (1/6) * (Real.sqrt a_sq) * (Real.sqrt b_sq) * (Real.sqrt c_sq) = 2 * Real.sqrt 176 :=
by
  -- Proof steps would go here
  sorry

end volume_of_tetrahedron_OABC_l130_130138


namespace toilet_paper_production_per_day_l130_130575

theorem toilet_paper_production_per_day 
    (total_production_march : ℕ)
    (days_in_march : ℕ)
    (increase_factor : ℕ)
    (total_production : ℕ)
    (days : ℕ)
    (increase : ℕ)
    (production : ℕ) :
    total_production_march = total_production →
    days_in_march = days →
    increase_factor = increase →
    total_production = 868000 →
    days = 31 →
    increase = 3 →
    production = total_production / days →
    production / increase = 9333
:= by
  intros h1 h2 h3 h4 h5 h6 h7

  sorry

end toilet_paper_production_per_day_l130_130575


namespace two_packs_remainder_l130_130934

theorem two_packs_remainder (m : ℕ) (h : m % 7 = 5) : (2 * m) % 7 = 3 :=
by {
  sorry
}

end two_packs_remainder_l130_130934


namespace necessary_but_not_sufficient_for_inequality_l130_130968

variables (a b : ℝ)

theorem necessary_but_not_sufficient_for_inequality (h : a ≠ b) (hab_pos : a * b > 0) :
  (b/a + a/b > 2) :=
sorry

end necessary_but_not_sufficient_for_inequality_l130_130968


namespace exists_increasing_seq_with_sum_square_diff_l130_130272

/-- There exists an increasing sequence of natural numbers in which
  the sum of any two consecutive terms is equal to the square of their
  difference. -/
theorem exists_increasing_seq_with_sum_square_diff :
  ∃ (a : ℕ → ℕ), (∀ n : ℕ, a n < a (n + 1)) ∧ (∀ n : ℕ, a n + a (n + 1) = (a (n + 1) - a n) ^ 2) :=
sorry

end exists_increasing_seq_with_sum_square_diff_l130_130272


namespace find_PQ_length_l130_130735

-- Defining the problem parameters
variables {X Y Z P Q R : Type}
variables (dXY dXZ dPQ dPR : ℝ)
variable (angle_common : ℝ)

-- Conditions:
def angle_XYZ_PQR_common : Prop :=
  angle_common = 150 ∧ 
  dXY = 10 ∧
  dXZ = 20 ∧
  dPQ = 5 ∧
  dPR = 12

-- Question: Prove PQ = 2.5 given the conditions
theorem find_PQ_length
  (h : angle_XYZ_PQR_common dXY dXZ dPQ dPR angle_common) :
  dPQ = 2.5 :=
sorry

end find_PQ_length_l130_130735


namespace each_friend_eats_six_slices_l130_130724

-- Definitions
def slices_per_loaf : ℕ := 15
def loaves_bought : ℕ := 4
def friends : ℕ := 10
def total_slices : ℕ := loaves_bought * slices_per_loaf
def slices_per_friend : ℕ := total_slices / friends

-- Theorem to prove
theorem each_friend_eats_six_slices (h1 : slices_per_loaf = 15) (h2 : loaves_bought = 4) (h3 : friends = 10) : slices_per_friend = 6 :=
by
  sorry

end each_friend_eats_six_slices_l130_130724


namespace geometric_seq_sum_l130_130754

theorem geometric_seq_sum (a : ℝ) (q : ℝ) (ha : a ≠ 0) (hq : q ≠ 1) 
    (hS4 : a * (1 - q^4) / (1 - q) = 1) 
    (hS12 : a * (1 - q^12) / (1 - q) = 13) 
    : a * q^12 * (1 + q + q^2 + q^3) = 27 := 
by
  sorry

end geometric_seq_sum_l130_130754


namespace shift_sin_to_cos_l130_130535

open Real

theorem shift_sin_to_cos:
  ∀ x: ℝ, 3 * cos (2 * x) = 3 * sin (2 * (x + π / 6) - π / 6) :=
by 
  sorry

end shift_sin_to_cos_l130_130535


namespace bobbit_worm_fish_count_l130_130852

theorem bobbit_worm_fish_count 
  (initial_fish : ℕ)
  (fish_eaten_per_day : ℕ)
  (days_before_adding_fish : ℕ)
  (additional_fish : ℕ)
  (days_after_adding_fish : ℕ) :
  days_before_adding_fish = 14 →
  days_after_adding_fish = 7 →
  fish_eaten_per_day = 2 →
  initial_fish = 60 →
  additional_fish = 8 →
  (initial_fish - days_before_adding_fish * fish_eaten_per_day + additional_fish - days_after_adding_fish * fish_eaten_per_day) = 26 :=
by
  intros 
  -- sorry proof goes here
  sorry

end bobbit_worm_fish_count_l130_130852


namespace transformed_quadratic_equation_l130_130814

theorem transformed_quadratic_equation (u v: ℝ) :
  (u + v = -5 / 2) ∧ (u * v = 3 / 2) ↔ (∃ y : ℝ, y^2 - y + 6 = 0) := sorry

end transformed_quadratic_equation_l130_130814


namespace uncle_ben_parking_probability_l130_130543

theorem uncle_ben_parking_probability :
  let total_spaces := 20
  let cars := 15
  let rv_spaces := 3
  let total_combinations := Nat.choose total_spaces cars
  let non_adjacent_empty_combinations := Nat.choose (total_spaces - rv_spaces) cars
  (1 - (non_adjacent_empty_combinations / total_combinations)) = (232 / 323) := by
  sorry

end uncle_ben_parking_probability_l130_130543


namespace deposit_correct_l130_130220

-- Define the conditions
def monthly_income : ℝ := 10000
def deposit_percentage : ℝ := 0.25

-- Define the deposit calculation based on the conditions
def deposit_amount (income : ℝ) (percentage : ℝ) : ℝ :=
  percentage * income

-- Theorem: Prove that the deposit amount is Rs. 2500
theorem deposit_correct :
    deposit_amount monthly_income deposit_percentage = 2500 :=
  sorry

end deposit_correct_l130_130220


namespace perpendicular_lines_slope_l130_130471

theorem perpendicular_lines_slope (a : ℝ)
  (h : (a * (a + 2)) = -1) : a = -1 :=
sorry

end perpendicular_lines_slope_l130_130471


namespace no_such_decreasing_h_exists_l130_130218

-- Define the interval [0, ∞)
def nonneg_reals := {x : ℝ // 0 ≤ x}

-- Define a decreasing function h on [0, ∞)
def is_decreasing (h : ℝ → ℝ) : Prop := ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x < y → h x ≥ h y

-- Define the function f based on h
def f (h : ℝ → ℝ) (x : ℝ) : ℝ := (x^2 - x + 1) * h x

-- Define the increasing property for f on [0, ∞)
def is_increasing_on_nonneg_reals (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x < y → f x ≤ f y

theorem no_such_decreasing_h_exists :
  ¬ ∃ h : ℝ → ℝ, is_decreasing h ∧ is_increasing_on_nonneg_reals (f h) :=
by sorry

end no_such_decreasing_h_exists_l130_130218


namespace inequality_solution_l130_130725

theorem inequality_solution (x : ℝ) : (2 * x - 1) / 3 > (3 * x - 2) / 2 - 1 → x < 2 :=
by
  intro h
  sorry

end inequality_solution_l130_130725


namespace eval_36_pow_five_over_two_l130_130954

theorem eval_36_pow_five_over_two : (36 : ℝ)^(5/2) = 7776 := by
  sorry

end eval_36_pow_five_over_two_l130_130954


namespace systematic_sampling_method_l130_130665

-- Defining the conditions of the problem as lean definitions
def sampling_interval_is_fixed (interval : ℕ) : Prop :=
  interval = 10

def production_line_uniformly_flowing : Prop :=
  true  -- Assumption

-- The main theorem formulation
theorem systematic_sampling_method :
  ∀ (interval : ℕ), sampling_interval_is_fixed interval → production_line_uniformly_flowing →
  (interval = 10 → true) :=
by {
  sorry
}

end systematic_sampling_method_l130_130665


namespace sin_symmetry_value_l130_130022

theorem sin_symmetry_value (ϕ : ℝ) (hϕ₀ : 0 < ϕ) (hϕ₁ : ϕ < π / 2) :
  ϕ = 5 * π / 12 :=
sorry

end sin_symmetry_value_l130_130022


namespace find_vertex_C_l130_130373

def A : ℝ × ℝ := (2, 0)
def B : ℝ × ℝ := (0, 4)
def euler_line (x y : ℝ) : Prop := x - y + 2 = 0

theorem find_vertex_C 
  (C : ℝ × ℝ)
  (h_centroid : (2 + C.1) / 3 = (4 + C.2) / 3)
  (h_euler_line : euler_line ((2 + C.1) / 3) ((4 + C.2) / 3))
  (h_circumcenter : (C.1 + 1)^2 + (C.2 - 1)^2 = 10) :
  C = (-4, 0) :=
sorry

end find_vertex_C_l130_130373


namespace sufficiency_of_inequality_l130_130635

theorem sufficiency_of_inequality (x : ℝ) (h : x > 5) : x^2 > 25 :=
sorry

end sufficiency_of_inequality_l130_130635


namespace visual_range_increase_percent_l130_130158

theorem visual_range_increase_percent :
  let original_visual_range := 100
  let new_visual_range := 150
  ((new_visual_range - original_visual_range) / original_visual_range) * 100 = 50 :=
by
  sorry

end visual_range_increase_percent_l130_130158


namespace dog_group_division_l130_130751

theorem dog_group_division:
  let total_dogs := 12
  let group1_size := 4
  let group2_size := 5
  let group3_size := 3
  let Rocky_in_group1 := true
  let Bella_in_group2 := true
  (total_dogs == 12 ∧ group1_size == 4 ∧ group2_size == 5 ∧ group3_size == 3 ∧ Rocky_in_group1 ∧ Bella_in_group2) →
  (∃ ways: ℕ, ways = 4200)
  :=
  sorry

end dog_group_division_l130_130751


namespace chi_squared_test_expectation_correct_distribution_table_correct_l130_130675

-- Given data for the contingency table
def male_good := 52
def male_poor := 8
def female_good := 28
def female_poor := 12
def total := 100

-- Define the $\chi^2$ calculation
def chi_squared_value : ℚ :=
  (total * (male_good * female_poor - male_poor * female_good)^2) / 
  ((male_good + male_poor) * (female_good + female_poor) * (male_good + female_good) * (male_poor + female_poor))

-- The $\chi^2$ value to compare against for 99% confidence
def critical_value_99 : ℚ := 6.635

-- Prove that $\chi^2$ value is less than the critical value for 99% confidence
theorem chi_squared_test :
  chi_squared_value < critical_value_99 :=
by
  -- Sorry to skip the proof as instructed
  sorry

-- Probability data and expectations for successful shots
def prob_male_success : ℚ := 2 / 3
def prob_female_success : ℚ := 1 / 2

-- Probabilities of the number of successful shots
def prob_X_0 : ℚ := (1 - prob_male_success) ^ 2 * (1 - prob_female_success)
def prob_X_1 : ℚ := 2 * prob_male_success * (1 - prob_male_success) * (1 - prob_female_success) +
                    (1 - prob_male_success) ^ 2 * prob_female_success
def prob_X_2 : ℚ := prob_male_success ^ 2 * (1 - prob_female_success) +
                    2 * prob_male_success * (1 - prob_male_success) * prob_female_success
def prob_X_3 : ℚ := prob_male_success ^ 2 * prob_female_success

def expectation_X : ℚ :=
  0 * prob_X_0 + 
  1 * prob_X_1 + 
  2 * prob_X_2 + 
  3 * prob_X_3

-- The expected value of X
def expected_value_X : ℚ := 11 / 6

-- Prove the expected value is as calculated
theorem expectation_correct :
  expectation_X = expected_value_X :=
by
  -- Sorry to skip the proof as instructed
  sorry

-- Define the distribution table based on calculated probabilities
def distribution_table : List (ℚ × ℚ) :=
  [(0, prob_X_0), (1, prob_X_1), (2, prob_X_2), (3, prob_X_3)]

-- The correct distribution table
def correct_distribution_table : List (ℚ × ℚ) :=
  [(0, 1 / 18), (1, 5 / 18), (2, 4 / 9), (3, 2 / 9)]

-- Prove the distribution table is as calculated
theorem distribution_table_correct :
  distribution_table = correct_distribution_table :=
by
  -- Sorry to skip the proof as instructed
  sorry

end chi_squared_test_expectation_correct_distribution_table_correct_l130_130675


namespace sum_first_5n_eq_630_l130_130495

theorem sum_first_5n_eq_630 (n : ℕ)
  (h : (4 * n * (4 * n + 1)) / 2 = (2 * n * (2 * n + 1)) / 2 + 300) :
  (5 * n * (5 * n + 1)) / 2 = 630 :=
sorry

end sum_first_5n_eq_630_l130_130495


namespace daniel_practices_each_school_day_l130_130490

-- Define the conditions
def total_minutes : ℕ := 135
def school_days : ℕ := 5
def weekend_days : ℕ := 2

-- Define the variables
def x : ℕ := 15

-- Define the practice time equations
def school_week_practice_time (x : ℕ) := school_days * x
def weekend_practice_time (x : ℕ) := weekend_days * 2 * x
def total_practice_time (x : ℕ) := school_week_practice_time x + weekend_practice_time x

-- The proof goal
theorem daniel_practices_each_school_day :
  total_practice_time x = total_minutes := by
  sorry

end daniel_practices_each_school_day_l130_130490


namespace fraction_of_fraction_l130_130065

theorem fraction_of_fraction:
  let a := (3:ℚ) / 4
  let b := (5:ℚ) / 12
  b / a = (5:ℚ) / 9 := by
  sorry

end fraction_of_fraction_l130_130065


namespace find_alpha_after_five_operations_l130_130007

def returns_to_starting_point_after_operations (α : Real) (n : Nat) : Prop :=
  (n * α) % 360 = 0

theorem find_alpha_after_five_operations (α : Real) 
  (hα1 : 0 < α)
  (hα2 : α < 180)
  (h_return : returns_to_starting_point_after_operations α 5) :
  α = 72 ∨ α = 144 :=
sorry

end find_alpha_after_five_operations_l130_130007


namespace multiplication_of_negative_and_positive_l130_130813

theorem multiplication_of_negative_and_positive :
  (-3) * 5 = -15 :=
by
  sorry

end multiplication_of_negative_and_positive_l130_130813


namespace inequality_solution_l130_130137

-- Define the condition for the denominator being positive
def denom_positive (x : ℝ) : Prop :=
  x^2 + 2*x + 7 > 0

-- Statement of the problem
theorem inequality_solution (x : ℝ) (h : denom_positive x) :
  (x + 6) / (x^2 + 2*x + 7) ≥ 0 ↔ x ≥ -6 :=
sorry

end inequality_solution_l130_130137


namespace monotonicity_of_f_odd_function_a_value_l130_130407

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 1 / (2 ^ x + 1)

-- Part 1: Prove that f(x) is monotonically increasing
theorem monotonicity_of_f (a : ℝ) : 
  ∀ x1 x2 : ℝ, x1 < x2 → f a x1 < f a x2 := by
  intro x1 x2 hx
  sorry

-- Part 2: If f(x) is an odd function, find the value of a
theorem odd_function_a_value (a : ℝ) (h_odd : ∀ x : ℝ, f a (-x) = -f a x) : 
  f a 0 = 0 → a = 1 / 2 := by
  intro h
  sorry

end monotonicity_of_f_odd_function_a_value_l130_130407


namespace meaningful_sqrt_range_l130_130722

theorem meaningful_sqrt_range (x : ℝ) (h : 1 - x ≥ 0) : x ≤ 1 :=
sorry

end meaningful_sqrt_range_l130_130722


namespace cricket_bat_weight_l130_130591

-- Define the conditions as Lean definitions
def weight_of_basketball : ℕ := 36
def weight_of_basketballs (n : ℕ) := n * weight_of_basketball
def weight_of_cricket_bats (m : ℕ) := m * (weight_of_basketballs 4 / 8)

-- State the theorem and skip the proof
theorem cricket_bat_weight :
  weight_of_cricket_bats 1 = 18 :=
by
  sorry

end cricket_bat_weight_l130_130591


namespace quadrilateral_is_parallelogram_l130_130536

theorem quadrilateral_is_parallelogram (a b c d : ℝ) (h : (a - c) ^ 2 + (b - d) ^ 2 = 0) : 
  -- The theorem states that if lengths a, b, c, d of a quadrilateral satisfy the given equation,
  -- then the quadrilateral must be a parallelogram.
  a = c ∧ b = d :=
by {
  sorry
}

end quadrilateral_is_parallelogram_l130_130536


namespace odd_solution_exists_l130_130131

theorem odd_solution_exists (k m n : ℕ) (h : m * n = k^2 + k + 3) : 
∃ (x y : ℤ), (x^2 + 11 * y^2 = 4 * m ∨ x^2 + 11 * y^2 = 4 * n) ∧ (x % 2 ≠ 0 ∧ y % 2 ≠ 0) :=
sorry

end odd_solution_exists_l130_130131


namespace lucky_numbers_count_l130_130863

def isLuckyNumber (n : ℕ) : Bool :=
  let d1 := n / 100
  let d2 := (n % 100) / 10
  let d3 := n % 10
  (d1 + d2 + d3 = 6) && (100 ≤ n) && (n < 1000)

def countLuckyNumbers : ℕ :=
  (List.range' 100 900).filter isLuckyNumber |>.length

theorem lucky_numbers_count : countLuckyNumbers = 21 := 
  sorry

end lucky_numbers_count_l130_130863


namespace sum_of_first_ten_terms_l130_130102

theorem sum_of_first_ten_terms (a1 d : ℝ) (h1 : 3 * (a1 + d) = 15) 
  (h2 : (a1 + d - 1) ^ 2 = (a1 - 1) * (a1 + 2 * d + 1)) : 
  (10 / 2) * (2 * a1 + (10 - 1) * d) = 120 := 
by 
  sorry

end sum_of_first_ten_terms_l130_130102


namespace perfect_square_trinomial_l130_130734

theorem perfect_square_trinomial (m : ℝ) : 
  (∃ a : ℝ, (∀ x : ℝ, x^2 + 2*(m - 1)*x + 16 = (x + a)^2) ∨ (∀ x : ℝ, x^2 + 2*(m - 1)*x + 16 = (x - a)^2)) ↔ m = 5 ∨ m = -3 :=
sorry

end perfect_square_trinomial_l130_130734


namespace max_monthly_profit_l130_130494

theorem max_monthly_profit (x : ℝ) (h : 0 < x ∧ x ≤ 15) :
  let C := 100 + 4 * x
  let p := 76 + 15 * x - x^2
  let L := p * x - C
  L = -x^3 + 15 * x^2 + 72 * x - 100 ∧
  (∀ x, 0 < x ∧ x ≤ 15 → L ≤ -12^3 + 15 * 12^2 + 72 * 12 - 100) :=
by
  sorry

end max_monthly_profit_l130_130494


namespace shanmukham_total_payment_l130_130909

noncomputable def total_price_shanmukham_pays : Real :=
  let itemA_price : Real := 6650
  let itemA_rebate : Real := 6 -- percentage
  let itemA_tax : Real := 10 -- percentage

  let itemB_price : Real := 8350
  let itemB_rebate : Real := 4 -- percentage
  let itemB_tax : Real := 12 -- percentage

  let itemC_price : Real := 9450
  let itemC_rebate : Real := 8 -- percentage
  let itemC_tax : Real := 15 -- percentage

  let final_price (price : Real) (rebate : Real) (tax : Real) : Real :=
    let rebate_amt := (rebate / 100) * price
    let price_after_rebate := price - rebate_amt
    let tax_amt := (tax / 100) * price_after_rebate
    price_after_rebate + tax_amt

  final_price itemA_price itemA_rebate itemA_tax +
  final_price itemB_price itemB_rebate itemB_tax +
  final_price itemC_price itemC_rebate itemC_tax

theorem shanmukham_total_payment :
  total_price_shanmukham_pays = 25852.12 := by
  sorry

end shanmukham_total_payment_l130_130909


namespace find_a_l130_130783

noncomputable def coefficient_of_x3_in_expansion (a : ℝ) : ℝ :=
  6 * a^2 - 15 * a + 20 

theorem find_a (a : ℝ) (h : coefficient_of_x3_in_expansion a = 56) : a = 6 ∨ a = -1 :=
  sorry

end find_a_l130_130783


namespace find_two_digit_number_l130_130673

theorem find_two_digit_number (N : ℕ) (a b c : ℕ) 
  (h_end_digits : N % 1000 = c + 10 * b + 100 * a)
  (hN2_end_digits : N^2 % 1000 = c + 10 * b + 100 * a)
  (h_nonzero : a ≠ 0) :
  10 * a + b = 24 := 
by
  sorry

end find_two_digit_number_l130_130673


namespace frac_left_handed_l130_130822

variable (x : ℕ)

def red_participants := 10 * x
def blue_participants := 5 * x
def total_participants := red_participants x + blue_participants x

def left_handed_red := (1 / 3 : ℚ) * red_participants x
def left_handed_blue := (2 / 3 : ℚ) * blue_participants x
def total_left_handed := left_handed_red x + left_handed_blue x

theorem frac_left_handed :
  total_left_handed x / total_participants x = (4 / 9 : ℚ) := by
  sorry

end frac_left_handed_l130_130822


namespace odd_numbers_le_twice_switch_pairs_l130_130265

-- Number of odd elements in row n is denoted as numOdd n
def numOdd (n : ℕ) : ℕ := -- Definition of numOdd function
sorry

-- Number of switch pairs in row n is denoted as numSwitchPairs n
def numSwitchPairs (n : ℕ) : ℕ := -- Definition of numSwitchPairs function
sorry

-- Definition of Pascal's Triangle and conditions
def binom (n k : ℕ) : ℕ := if k > n then 0 else if k = 0 ∨ k = n then 1 else binom (n-1) (k-1) + binom (n-1) k

-- Check even or odd
def isOdd (n : ℕ) : Bool := n % 2 = 1

-- Definition of switch pair check
def isSwitchPair (a b : ℕ) : Prop := (isOdd a ∧ ¬isOdd b) ∨ (¬isOdd a ∧ isOdd b)

theorem odd_numbers_le_twice_switch_pairs (n : ℕ) :
  numOdd n ≤ 2 * numSwitchPairs (n-1) :=
sorry

end odd_numbers_le_twice_switch_pairs_l130_130265


namespace combined_final_selling_price_correct_l130_130394

def itemA_cost : Float := 180.0
def itemB_cost : Float := 220.0
def itemC_cost : Float := 130.0

def itemA_profit_margin : Float := 0.15
def itemB_profit_margin : Float := 0.20
def itemC_profit_margin : Float := 0.25

def itemA_tax_rate : Float := 0.05
def itemB_discount_rate : Float := 0.10
def itemC_tax_rate : Float := 0.08

def itemA_selling_price_before_tax := itemA_cost * (1 + itemA_profit_margin)
def itemB_selling_price_before_discount := itemB_cost * (1 + itemB_profit_margin)
def itemC_selling_price_before_tax := itemC_cost * (1 + itemC_profit_margin)

def itemA_final_price := itemA_selling_price_before_tax * (1 + itemA_tax_rate)
def itemB_final_price := itemB_selling_price_before_discount * (1 - itemB_discount_rate)
def itemC_final_price := itemC_selling_price_before_tax * (1 + itemC_tax_rate)

def combined_final_price := itemA_final_price + itemB_final_price + itemC_final_price

theorem combined_final_selling_price_correct : 
  combined_final_price = 630.45 :=
by
  -- proof would go here
  sorry

end combined_final_selling_price_correct_l130_130394


namespace find_x_min_construction_cost_l130_130253

-- Define the conditions for Team A and Team B
def Team_A_Daily_Construction (x : ℕ) : ℕ := x + 300
def Team_A_Daily_Cost : ℕ := 3600
def Team_B_Daily_Construction (x : ℕ) : ℕ := x
def Team_B_Daily_Cost : ℕ := 2200

-- Condition: The number of days Team A needs to construct 1800m^2 is equal to the number of days Team B needs to construct 1200m^2
def construction_days (x : ℕ) : Prop := 
  1800 / (x + 300) = 1200 / x

-- Define the total days worked and the minimum construction area condition
def total_days : ℕ := 22
def min_construction_area : ℕ := 15000

-- Define the construction cost function given the number of days each team works
def construction_cost (m : ℕ) : ℕ := 
  3600 * m + 2200 * (total_days - m)

-- Main theorem: Prove that x = 600 satisfies the conditions
theorem find_x (x : ℕ) (h : x = 600) : construction_days x := by sorry

-- Second theorem: Prove that the minimum construction cost is 56800 yuan
theorem min_construction_cost (m : ℕ) (h : m ≥ 6) : construction_cost m = 56800 := by sorry

end find_x_min_construction_cost_l130_130253


namespace volume_ratio_l130_130397

variable (A B : ℝ)

theorem volume_ratio (h1 : (3 / 4) * A = (5 / 8) * B) :
  A / B = 5 / 6 :=
by
  sorry

end volume_ratio_l130_130397


namespace distance_from_origin_is_correct_l130_130346

-- Define the point (x, y) with given conditions
variables (x y : ℝ)

-- Given conditions
axiom h1 : y = 20
axiom h2 : dist (x, y) (2, 15) = 15
axiom h3 : x > 2

-- The theorem to prove
theorem distance_from_origin_is_correct :
  dist (x, y) (0, 0) = Real.sqrt (604 + 40 * Real.sqrt 2) :=
by
  -- Set h1, h2, and h3 as our constraints
  sorry

end distance_from_origin_is_correct_l130_130346


namespace train_seat_count_l130_130162

theorem train_seat_count (t : ℝ)
  (h1 : ∃ (t : ℝ), t = 36 + 0.2 * t + 0.5 * t) :
  t = 120 :=
by
  sorry

end train_seat_count_l130_130162


namespace umbrella_cost_l130_130516

theorem umbrella_cost (house_umbrellas car_umbrellas cost_per_umbrella : ℕ) (h1 : house_umbrellas = 2) (h2 : car_umbrellas = 1) (h3 : cost_per_umbrella = 8) : 
  (house_umbrellas + car_umbrellas) * cost_per_umbrella = 24 := 
by
  sorry

end umbrella_cost_l130_130516


namespace smallest_positive_integer_div_conditions_l130_130509

theorem smallest_positive_integer_div_conditions :
  ∃ x : ℕ, x > 0 ∧ x % 5 = 2 ∧ x % 7 = 3 ∧ ∀ y : ℕ, (y > 0 ∧ y % 5 = 2 ∧ y % 7 = 3) → x ≤ y :=
  sorry

end smallest_positive_integer_div_conditions_l130_130509


namespace speedster_convertibles_proof_l130_130362

-- Definitions based on conditions
def total_inventory (T : ℕ) : Prop := 2 / 3 * T = 2 / 3 * T
def not_speedsters (T : ℕ) : Prop := 1 / 3 * T = 60
def speedsters (T : ℕ) (S : ℕ) : Prop := S = 2 / 3 * T
def speedster_convertibles (S : ℕ) (C : ℕ) : Prop := C = 4 / 5 * S

theorem speedster_convertibles_proof (T S C : ℕ) (hT : total_inventory T) (hNS : not_speedsters T) (hS : speedsters T S) (hSC : speedster_convertibles S C) : C = 96 :=
by
  -- Proof goes here
  sorry

end speedster_convertibles_proof_l130_130362


namespace length_FJ_is_35_l130_130006

noncomputable def length_of_FJ (h : ℝ) : ℝ :=
  let FG := 50
  let HI := 20
  let trapezium_area := (1 / 2) * (FG + HI) * h
  let half_trapezium_area := trapezium_area / 2
  let JI_area := (1 / 2) * 35 * h
  35

theorem length_FJ_is_35 (h : ℝ) : length_of_FJ h = 35 :=
  sorry

end length_FJ_is_35_l130_130006


namespace side_length_of_square_l130_130451

theorem side_length_of_square (d : ℝ) (s : ℝ) (h1 : d = 2 * Real.sqrt 2) (h2 : d = s * Real.sqrt 2) : s = 2 :=
by
  sorry

end side_length_of_square_l130_130451


namespace total_worth_of_produce_is_630_l130_130772

def bundles_of_asparagus : ℕ := 60
def price_per_bundle_asparagus : ℝ := 3.00

def boxes_of_grapes : ℕ := 40
def price_per_box_grapes : ℝ := 2.50

def num_apples : ℕ := 700
def price_per_apple : ℝ := 0.50

def total_worth : ℝ :=
  bundles_of_asparagus * price_per_bundle_asparagus +
  boxes_of_grapes * price_per_box_grapes +
  num_apples * price_per_apple

theorem total_worth_of_produce_is_630 : 
  total_worth = 630 := by
  sorry

end total_worth_of_produce_is_630_l130_130772


namespace sarees_original_price_l130_130292

theorem sarees_original_price (P : ℝ) (h : 0.75 * 0.85 * P = 248.625) : P = 390 :=
by
  sorry

end sarees_original_price_l130_130292


namespace option_A_is_correct_l130_130409

theorem option_A_is_correct (a b : ℝ) (h : a ≠ 0) : (a^2 / (a * b)) = (a / b) :=
by
  -- Proof will be filled in here
  sorry

end option_A_is_correct_l130_130409


namespace repeating_decimal_sum_num_denom_l130_130802

noncomputable def repeating_decimal_to_fraction (n d : ℕ) (rep : ℚ) : ℚ :=
(rep * (10^d) - rep) / ((10^d) - 1)

theorem repeating_decimal_sum_num_denom
  (x : ℚ)
  (h1 : x = repeating_decimal_to_fraction 45 2 0.45)
  (h2 : repeating_decimal_to_fraction 45 2 0.45 = 5/11) : 
  (5 + 11) = 16 :=
by 
  sorry

end repeating_decimal_sum_num_denom_l130_130802


namespace find_number_l130_130246

theorem find_number :
  ∃ (N : ℝ), (5 / 4) * N = (4 / 5) * N + 45 ∧ N = 100 :=
by
  sorry

end find_number_l130_130246


namespace workers_contribution_l130_130148

theorem workers_contribution (N C : ℕ) 
(h1 : N * C = 300000) 
(h2 : N * (C + 50) = 360000) : 
N = 1200 :=
sorry

end workers_contribution_l130_130148


namespace fraction_div_subtract_l130_130920

theorem fraction_div_subtract : 
  (5 / 6 : ℚ) / (9 / 10) - (1 / 15) = 116 / 135 := 
by 
  sorry

end fraction_div_subtract_l130_130920


namespace max_sum_seq_l130_130868

theorem max_sum_seq (a : ℕ → ℝ) (h1 : a 1 = 0)
  (h2 : abs (a 2) = abs (a 1 - 1)) 
  (h3 : abs (a 3) = abs (a 2 - 1)) 
  (h4 : abs (a 4) = abs (a 3 - 1)) 
  : ∃ M, (∀ (b : ℕ → ℝ), b 1 = 0 → abs (b 2) = abs (b 1 - 1) → abs (b 3) = abs (b 2 - 1) → abs (b 4) = abs (b 3 - 1) → (b 1 + b 2 + b 3 + b 4) ≤ M) 
    ∧ (a 1 + a 2 + a 3 + a 4 = M) :=
  sorry

end max_sum_seq_l130_130868


namespace decrease_in_area_of_equilateral_triangle_l130_130518

noncomputable def equilateral_triangle_area (s : ℝ) : ℝ := (s^2 * Real.sqrt 3) / 4

theorem decrease_in_area_of_equilateral_triangle :
  (equilateral_triangle_area 20 - equilateral_triangle_area 14) = 51 * Real.sqrt 3 := by
  sorry

end decrease_in_area_of_equilateral_triangle_l130_130518


namespace meaningful_sqrt_range_l130_130421

theorem meaningful_sqrt_range (x : ℝ) (h : 2 * x - 3 ≥ 0) : x ≥ 3 / 2 :=
sorry

end meaningful_sqrt_range_l130_130421


namespace estimate_ratio_l130_130561

theorem estimate_ratio (A B : ℕ) (A_def : A = 1 * 2 * 7 + 2 * 4 * 14 + 3 * 6 * 21 + 4 * 8 * 28)
  (B_def : B = 1 * 3 * 5 + 2 * 6 * 10 + 3 * 9 * 15 + 4 * 12 * 20) : 0 < A / B ∧ A / B < 1 := by
  sorry

end estimate_ratio_l130_130561


namespace find_certain_amount_l130_130827

theorem find_certain_amount :
  ∀ (A : ℝ), (160 * 8 * 12.5 / 100 = A * 8 * 4 / 100) → 
            (A = 500) :=
  by
    intros A h
    sorry

end find_certain_amount_l130_130827


namespace area_of_triangle_l130_130955

theorem area_of_triangle (a b c : ℝ) (C : ℝ) 
  (h1 : c^2 = (a - b)^2 + 6)
  (h2 : C = Real.pi / 3) 
  : (1/2 * a * b * Real.sin C = 3 * Real.sqrt 3 / 2) :=
sorry

end area_of_triangle_l130_130955


namespace perpendicular_line_through_point_l130_130402

theorem perpendicular_line_through_point
  (a b x1 y1 : ℝ)
  (h_line : a * 3 - b * 6 = 9)
  (h_point : (x1, y1) = (2, -3)) :
  ∃ m b, 
    (∀ x y, y = m * x + b ↔ a * x - y * b = a * 2 + b * 3) ∧ 
    m = -2 ∧ 
    b = 1 :=
by sorry

end perpendicular_line_through_point_l130_130402


namespace percentage_passed_eng_students_l130_130318

variable (total_male_students : ℕ := 120)
variable (total_female_students : ℕ := 100)
variable (total_international_students : ℕ := 70)
variable (total_disabilities_students : ℕ := 30)

variable (male_eng_percentage : ℕ := 25)
variable (female_eng_percentage : ℕ := 20)
variable (intern_eng_percentage : ℕ := 15)
variable (disab_eng_percentage : ℕ := 10)

variable (male_pass_percentage : ℕ := 20)
variable (female_pass_percentage : ℕ := 25)
variable (intern_pass_percentage : ℕ := 30)
variable (disab_pass_percentage : ℕ := 35)

def total_engineering_students : ℕ :=
  (total_male_students * male_eng_percentage / 100) +
  (total_female_students * female_eng_percentage / 100) +
  (total_international_students * intern_eng_percentage / 100) +
  (total_disabilities_students * disab_eng_percentage / 100)

def total_passed_engineering_students : ℕ :=
  (total_male_students * male_eng_percentage / 100 * male_pass_percentage / 100) +
  (total_female_students * female_eng_percentage / 100 * female_pass_percentage / 100) +
  (total_international_students * intern_eng_percentage / 100 * intern_pass_percentage / 100) +
  (total_disabilities_students * disab_eng_percentage / 100 * disab_pass_percentage / 100)

def passed_eng_students_percentage : ℕ :=
  total_passed_engineering_students * 100 / total_engineering_students

theorem percentage_passed_eng_students :
  passed_eng_students_percentage = 23 :=
sorry

end percentage_passed_eng_students_l130_130318


namespace nadia_flower_shop_l130_130168

theorem nadia_flower_shop :
  let roses := 20
  let lilies := (3 / 4) * roses
  let cost_per_rose := 5
  let cost_per_lily := 2 * cost_per_rose
  let total_cost := roses * cost_per_rose + lilies * cost_per_lily
  total_cost = 250 := by
    sorry

end nadia_flower_shop_l130_130168


namespace wendy_lost_lives_l130_130670

theorem wendy_lost_lives (L : ℕ) (h1 : 10 - L + 37 = 41) : L = 6 :=
by
  sorry

end wendy_lost_lives_l130_130670


namespace find_m_l130_130697

variable (a : ℝ) (m : ℝ)

theorem find_m (h : a^(m + 1) * a^(2 * m - 1) = a^9) : m = 3 := 
by
  sorry

end find_m_l130_130697


namespace new_light_wattage_is_143_l130_130010

-- Define the original wattage and the percentage increase
def original_wattage : ℕ := 110
def percentage_increase : ℕ := 30

-- Compute the increase in wattage
noncomputable def increase : ℕ := (percentage_increase * original_wattage) / 100

-- The new wattage should be the original wattage plus the increase
noncomputable def new_wattage : ℕ := original_wattage + increase

-- State the theorem that proves the new wattage is 143 watts
theorem new_light_wattage_is_143 : new_wattage = 143 := by
  unfold new_wattage
  unfold increase
  sorry

end new_light_wattage_is_143_l130_130010


namespace number_of_pieces_from_rod_l130_130760

theorem number_of_pieces_from_rod (rod_length_m : ℕ) (piece_length_cm : ℕ) (meter_to_cm : ℕ) 
  (h1 : rod_length_m = 34) (h2 : piece_length_cm = 85) (h3 : meter_to_cm = 100) : 
  rod_length_m * meter_to_cm / piece_length_cm = 40 := by
  sorry

end number_of_pieces_from_rod_l130_130760


namespace generatrix_length_of_cone_l130_130030

theorem generatrix_length_of_cone (r : ℝ) (l : ℝ) (h1 : r = 4) (h2 : (2 * Real.pi * r) = (Real.pi / 2) * l) : l = 16 := 
by
  sorry

end generatrix_length_of_cone_l130_130030


namespace balancing_point_is_vertex_l130_130508

-- Define a convex polygon and its properties
structure ConvexPolygon (n : ℕ) :=
(vertices : Fin n → Point)

-- Define a balancing point for a convex polygon
def is_balancing_point (P : ConvexPolygon n) (Q : Point) : Prop :=
  -- Placeholder for the actual definition that the areas formed by drawing lines from Q to vertices of P are equal
  sorry

-- Define the uniqueness of the balancing point
def unique_balancing_point (P : ConvexPolygon n) (Q : Point) : Prop :=
  ∀ R : Point, is_balancing_point P R → R = Q

-- Main theorem statement
theorem balancing_point_is_vertex (P : ConvexPolygon n) (Q : Point) 
  (h_balance : is_balancing_point P Q) (h_unique : unique_balancing_point P Q) : 
  ∃ i : Fin n, Q = P.vertices i :=
sorry

end balancing_point_is_vertex_l130_130508


namespace brian_needs_some_cartons_l130_130414

def servings_per_person : ℕ := sorry -- This should be defined with the actual number of servings per person.
def family_members : ℕ := 8
def us_cup_in_ml : ℕ := 250
def ml_per_serving : ℕ := us_cup_in_ml / 2
def ml_per_liter : ℕ := 1000

def total_milk_needed (servings_per_person : ℕ) : ℕ :=
  family_members * servings_per_person * ml_per_serving

def cartons_of_milk_needed (servings_per_person : ℕ) : ℕ :=
  total_milk_needed servings_per_person / ml_per_liter + if total_milk_needed servings_per_person % ml_per_liter = 0 then 0 else 1

theorem brian_needs_some_cartons (servings_per_person : ℕ) : 
  cartons_of_milk_needed servings_per_person = (family_members * servings_per_person * ml_per_serving / ml_per_liter + 
  if (family_members * servings_per_person * ml_per_serving) % ml_per_liter = 0 then 0 else 1) := 
by 
  sorry

end brian_needs_some_cartons_l130_130414


namespace base_five_equals_base_b_l130_130761

theorem base_five_equals_base_b : ∃ (b : ℕ), b > 0 ∧ (2 * 5^1 + 4 * 5^0) = (1 * b^2 + 0 * b^1 + 1 * b^0) := by
  sorry

end base_five_equals_base_b_l130_130761


namespace fruit_store_initial_quantities_l130_130756

-- Definitions from conditions:
def total_fruit (a b c : ℕ) := a + b + c = 275
def sold_apples (a : ℕ) := a - 30
def added_peaches (b : ℕ) := b + 45
def sold_pears (c : ℕ) := c - c / 4
def final_ratio (a b c : ℕ) := (sold_apples a) / 4 = (added_peaches b) / 3 ∧ (added_peaches b) / 3 = (sold_pears c) / 2

-- The proof problem:
theorem fruit_store_initial_quantities (a b c : ℕ) (h1 : total_fruit a b c) 
  (h2 : final_ratio a b c) : a = 150 ∧ b = 45 ∧ c = 80 :=
sorry

end fruit_store_initial_quantities_l130_130756


namespace find_general_term_l130_130844

theorem find_general_term (S a : ℕ → ℤ) (n : ℕ) (h_sum : S n = 2 * a n + 1) : a n = -2 * n - 1 := sorry

end find_general_term_l130_130844


namespace sum_of_two_integers_eq_sqrt_466_l130_130986

theorem sum_of_two_integers_eq_sqrt_466
  (x y : ℝ)
  (hx : x^2 + y^2 = 250)
  (hy : x * y = 108) :
  x + y = Real.sqrt 466 :=
sorry

end sum_of_two_integers_eq_sqrt_466_l130_130986


namespace smallest_number_is_minus_three_l130_130911

theorem smallest_number_is_minus_three :
  ∀ (a b c d : ℤ), (a = 0) → (b = -3) → (c = 1) → (d = -1) → b < d ∧ d < a ∧ a < c → b = -3 :=
by
  intros a b c d ha hb hc hd h
  exact hb

end smallest_number_is_minus_three_l130_130911


namespace max_value_of_b_minus_a_l130_130183

theorem max_value_of_b_minus_a (a b : ℝ) (h1 : a < 0) (h2 : ∀ x : ℝ, a < x ∧ x < b → (3 * x^2 + a) * (2 * x + b) ≥ 0) : b - a ≤ 1 / 3 :=
by
  sorry

end max_value_of_b_minus_a_l130_130183


namespace johns_number_l130_130241

theorem johns_number (n : ℕ) (h1 : ∃ k₁ : ℤ, n = 125 * k₁) (h2 : ∃ k₂ : ℤ, n = 180 * k₂) (h3 : 1000 < n) (h4 : n < 3000) : n = 1800 :=
sorry

end johns_number_l130_130241


namespace number_of_members_in_league_l130_130142

-- Define the conditions
def pair_of_socks_cost := 4
def t_shirt_cost := pair_of_socks_cost + 6
def cap_cost := t_shirt_cost - 3
def total_cost_per_member := 2 * (pair_of_socks_cost + t_shirt_cost + cap_cost)
def league_total_expenditure := 3144

-- Prove that the number of members in the league is 75
theorem number_of_members_in_league : 
  (∃ (n : ℕ), total_cost_per_member * n = league_total_expenditure) → 
  (∃ (n : ℕ), n = 75) :=
by
  sorry

end number_of_members_in_league_l130_130142


namespace cosA_sinB_value_l130_130593

theorem cosA_sinB_value (A B : ℝ) (hA1 : 0 < A ∧ A < π / 2) (hB1 : 0 < B ∧ B < π / 2)
  (h_tan_eq : (4 + (Real.tan A)^2) * (5 + (Real.tan B)^2) = Real.sqrt 320 * Real.tan A * Real.tan B) :
  Real.cos A * Real.sin B = Real.sqrt 6 / 6 := sorry

end cosA_sinB_value_l130_130593


namespace problem_statement_l130_130970

def U : Set ℕ := {x | x < 6}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4, 5}

theorem problem_statement : ((U \ A) ∪ (U \ B)) = {0, 1, 3, 4, 5} := by
  sorry

end problem_statement_l130_130970


namespace linear_function_y1_greater_y2_l130_130546

theorem linear_function_y1_greater_y2 :
  ∀ (y_1 y_2 : ℝ), 
    (y_1 = -(-1) + 6) → (y_2 = -(2) + 6) → y_1 > y_2 :=
by
  intros y_1 y_2 h1 h2
  sorry

end linear_function_y1_greater_y2_l130_130546


namespace root_expression_value_l130_130005

theorem root_expression_value
  (r s : ℝ)
  (h1 : 3 * r^2 - 4 * r - 8 = 0)
  (h2 : 3 * s^2 - 4 * s - 8 = 0) :
  (9 * r^3 - 9 * s^3) * (r - s)⁻¹ = 40 := 
sorry

end root_expression_value_l130_130005


namespace find_breadth_of_landscape_l130_130580

theorem find_breadth_of_landscape (L B A : ℕ) 
  (h1 : B = 8 * L)
  (h2 : 3200 = A / 9)
  (h3 : 3200 * 9 = A) :
  B = 480 :=
by 
  sorry

end find_breadth_of_landscape_l130_130580


namespace bills_are_fake_bart_can_give_exact_amount_l130_130119

-- Problem (a)
theorem bills_are_fake : 
  (∀ x, x = 17 ∨ x = 19 → false) :=
sorry

-- Problem (b)
theorem bart_can_give_exact_amount (n : ℕ) :
  (∀ m, m = 323  → (n ≥ m → ∃ a b : ℕ, n = 17 * a + 19 * b)) :=
sorry

end bills_are_fake_bart_can_give_exact_amount_l130_130119


namespace solve_fraction_equation_l130_130507

theorem solve_fraction_equation (x : ℝ) (h : x ≠ 3) : (x^2 - 9) / (x - 3) = 0 ↔ x = -3 :=
sorry

end solve_fraction_equation_l130_130507


namespace log_base_30_of_8_l130_130269

theorem log_base_30_of_8 (a b : Real) (h1 : Real.log 5 = a) (h2 : Real.log 3 = b) : 
    Real.logb 30 8 = 3 * (1 - a) / (b + 1) := 
  sorry

end log_base_30_of_8_l130_130269


namespace iron_wire_square_rectangle_l130_130055

theorem iron_wire_square_rectangle 
  (total_length : ℕ) 
  (rect_length : ℕ) 
  (h1 : total_length = 28) 
  (h2 : rect_length = 12) :
  (total_length / 4 = 7) ∧
  ((total_length / 2) - rect_length = 2) :=
by 
  sorry

end iron_wire_square_rectangle_l130_130055


namespace sum_of_squares_of_coefficients_l130_130532

theorem sum_of_squares_of_coefficients :
  let p := 3 * (X^5 + 4 * X^3 + 2 * X + 1)
  let coeffs := [3, 12, 6, 3, 0, 0]
  let sum_squares := coeffs.map (λ c => c * c) |>.sum
  sum_squares = 198 := by
  sorry

end sum_of_squares_of_coefficients_l130_130532


namespace max_area_of_fencing_l130_130357

theorem max_area_of_fencing (P : ℕ) (hP : P = 150) 
  (x y : ℕ) (h1 : x + y = P / 2) : (x * y) ≤ 1406 :=
sorry

end max_area_of_fencing_l130_130357


namespace log_w_u_value_l130_130884

noncomputable def log (base x : ℝ) : ℝ := Real.log x / Real.log base

theorem log_w_u_value (u v w : ℝ) (hu : u > 0) (hv : v > 0) (hw : w > 0) (hu1 : u ≠ 1) (hv1 : v ≠ 1) (hw1 : w ≠ 1)
    (h1 : log u (v * w) + log v w = 5) (h2 : log v u + log w v = 3) : 
    log w u = 4 / 5 := 
sorry

end log_w_u_value_l130_130884


namespace division_remainder_l130_130793

theorem division_remainder (n : ℕ) (h : n = 8 * 8 + 0) : n % 5 = 4 := by
  sorry

end division_remainder_l130_130793


namespace expected_value_range_of_p_l130_130767

theorem expected_value_range_of_p (p : ℝ) (X : ℕ → ℝ) :
  (∀ n, (n = 1 → X n = p) ∧ 
        (n = 2 → X n = p * (1 - p)) ∧ 
        (n = 3 → X n = (1 - p) ^ 2)) →
  (p^2 - 3 * p + 3 > 1.75) → 
  0 < p ∧ p < 0.5 := by
  intros hprob hexp
  -- Proof would be filled in here
  sorry

end expected_value_range_of_p_l130_130767


namespace greatest_integer_difference_l130_130121

theorem greatest_integer_difference (x y : ℤ) (hx : 3 < x ∧ x < 6) (hy : 6 < y ∧ y < 8) : 
  ∃ d, d = y - x ∧ d = 2 := 
by
  sorry

end greatest_integer_difference_l130_130121


namespace find_k_l130_130703

theorem find_k : 
  (∃ y, -3 * (-15) + y = k ∧ 0.3 * (-15) + y = 10) → k = 59.5 :=
by
  sorry

end find_k_l130_130703


namespace total_hats_purchased_l130_130521

theorem total_hats_purchased (B G : ℕ) (h1 : G = 38) (h2 : 6 * B + 7 * G = 548) : B + G = 85 := 
by 
  sorry

end total_hats_purchased_l130_130521


namespace minimum_value_f_b_eq_neg_a_maximum_value_ab_inequality_for_f_and_f_l130_130483

noncomputable def f (a b x : ℝ) := Real.exp x - a * x - b

theorem minimum_value_f_b_eq_neg_a (a : ℝ) (h : 0 < a) :
  ∃ m, m = 2 * a - a * Real.log a ∧ ∀ x : ℝ, f a (-a) x ≥ m :=
sorry

theorem maximum_value_ab (a b : ℝ) (h : ∀ x : ℝ, f a b x + a ≥ 0) :
  ab ≤ (1 / 2) * Real.exp 3 :=
sorry

theorem inequality_for_f_and_f' (a x1 x2 : ℝ) (h1 : 0 < a) (h2 : b = -a) (h3 : f a b x1 = 0) (h4 : f a b x2 = 0) (h5 : x1 < x2)
  : f a (-a) (3 * Real.log a) > (Real.exp ((2 * x1 * x2) / (x1 + x2)) - a) :=
sorry

end minimum_value_f_b_eq_neg_a_maximum_value_ab_inequality_for_f_and_f_l130_130483


namespace ThreePowFifteenModFive_l130_130445

def rem_div_3_pow_15_by_5 : ℕ :=
  let base := 3
  let mod := 5
  let exp := 15
  
  base^exp % mod

theorem ThreePowFifteenModFive (h1: 3^4 ≡ 1 [MOD 5]) : rem_div_3_pow_15_by_5 = 2 := by
  sorry

end ThreePowFifteenModFive_l130_130445


namespace lowest_price_is_six_l130_130345

def single_package_cost : ℝ := 7
def eight_oz_package_cost : ℝ := 4
def four_oz_package_original_cost : ℝ := 2
def discount_rate : ℝ := 0.5

theorem lowest_price_is_six
  (cost_single : single_package_cost = 7)
  (cost_eight : eight_oz_package_cost = 4)
  (cost_four : four_oz_package_original_cost = 2)
  (discount : discount_rate = 0.5) :
  min single_package_cost (eight_oz_package_cost + 2 * (four_oz_package_original_cost * discount_rate)) = 6 := by
  sorry

end lowest_price_is_six_l130_130345


namespace StacyBoughtPacks_l130_130305

theorem StacyBoughtPacks (sheets_per_pack days daily_printed_sheets total_packs : ℕ) 
  (h1 : sheets_per_pack = 240)
  (h2 : days = 6)
  (h3 : daily_printed_sheets = 80) 
  (h4 : total_packs = (days * daily_printed_sheets) / sheets_per_pack) : total_packs = 2 :=
by 
  sorry

end StacyBoughtPacks_l130_130305


namespace min_value_l130_130299

noncomputable def min_expression_value (x y z k : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hk : 2 ≤ k) : ℝ :=
  (x^2 + k*x + 1) * (y^2 + k*y + 1) * (z^2 + k*z + 1) / (x * y * z)

theorem min_value (x y z k : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hk : 2 ≤ k) :
  min_expression_value x y z k hx hy hz hk ≥ (2 + k)^3 :=
by
  sorry

end min_value_l130_130299


namespace tangent_line_parallel_points_l130_130014

-- Define the function f(x) = x^3 + x - 2
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Prove the points where the derivative equals 4
theorem tangent_line_parallel_points :
  ∃ (P0 : ℝ × ℝ), P0 = (1, 0) ∨ P0 = (-1, -4) ∧ (f' P0.fst = 4) :=
by
  sorry

end tangent_line_parallel_points_l130_130014


namespace find_f2_l130_130577

-- Define the conditions
variable {f g : ℝ → ℝ} {a : ℝ}

-- Assume f is an odd function
axiom odd_f : ∀ x : ℝ, f (-x) = -f x

-- Assume g is an even function
axiom even_g : ∀ x : ℝ, g (-x) = g x

-- Condition given in the problem
axiom f_g_relation : ∀ x : ℝ, f x + g x = a^x - a^(-x) + 2

-- Condition that g(2) = a
axiom g_at_2 : g 2 = a

-- Condition for a
axiom a_cond : a > 0 ∧ a ≠ 1

-- Proof problem
theorem find_f2 : f 2 = 15 / 4 := by
  sorry

end find_f2_l130_130577


namespace perpendicular_vector_solution_l130_130294

theorem perpendicular_vector_solution 
    (a b : ℝ × ℝ) (m : ℝ) 
    (h_a : a = (1, -1)) 
    (h_b : b = (-2, 3)) 
    (h_perp : a.1 * (a.1 + m * b.1) + a.2 * (a.2 + m * b.2) = 0) 
    : m = 2 / 5 := 
sorry

end perpendicular_vector_solution_l130_130294


namespace circle_equation_l130_130603

-- Define the conditions
def chord_length_condition (a b r : ℝ) : Prop := r^2 = a^2 + 1
def arc_length_condition (b r : ℝ) : Prop := r^2 = 2 * b^2
def min_distance_condition (a b : ℝ) : Prop := a = b

-- The main theorem stating the final answer
theorem circle_equation (a b r : ℝ) (h1 : chord_length_condition a b r)
    (h2 : arc_length_condition b r) (h3 : min_distance_condition a b) :
    ((x - a)^2 + (y - a)^2 = 2) ∨ ((x + a)^2 + (y + a)^2 = 2) :=
sorry

end circle_equation_l130_130603


namespace sequence_an_sum_sequence_Tn_l130_130025

theorem sequence_an (k c : ℝ) (S : ℕ → ℝ) (a : ℕ → ℝ)
  (hS : ∀ n, S n = k * c ^ n - k) (ha2 : a 2 = 4) (ha6 : a 6 = 8 * a 3) :
  ∀ n, a n = 2 ^ n :=
by
  -- Proof is assumed to be given
  sorry

theorem sum_sequence_Tn (a : ℕ → ℝ) (T : ℕ → ℝ)
  (ha : ∀ n, a n = 2 ^ n) :
  ∀ n, T n = (n - 1) * 2 ^ (n + 1) + 2 :=
by
  -- Proof is assumed to be given
  sorry

end sequence_an_sum_sequence_Tn_l130_130025


namespace total_number_of_people_l130_130437

theorem total_number_of_people (L F LF N T : ℕ) (hL : L = 13) (hF : F = 15) (hLF : LF = 9) (hN : N = 6) : 
  T = (L + F - LF) + N → T = 25 :=
by
  intros h
  rw [hL, hF, hLF, hN] at h
  exact h

end total_number_of_people_l130_130437


namespace problem_statement_l130_130999

theorem problem_statement (a b c : ℝ) (h1 : a - b = 2) (h2 : b - c = -3) : a - c = -1 := 
by
  sorry

end problem_statement_l130_130999


namespace average_children_in_families_with_children_l130_130692

theorem average_children_in_families_with_children :
  (15 * 3 = 45) ∧ (15 - 3 = 12) →
  (45 / (15 - 3) = 3.75) →
  (Float.round 3.75) = 3.8 :=
by
  intros h1 h2
  sorry

end average_children_in_families_with_children_l130_130692


namespace find_x_l130_130742

variable {a b x : ℝ}
variable (h₁ : b ≠ 0)
variable (h₂ : (3 * a) ^ (2 * b) = a ^ b * x ^ b)

theorem find_x (h₁ : b ≠ 0) (h₂ : (3 * a) ^ (2 * b) = a ^ b * x ^ b) : 
  x = 9 * a :=
by
  sorry

end find_x_l130_130742


namespace pure_imaginary_b_eq_two_l130_130229

theorem pure_imaginary_b_eq_two (b : ℝ) : (∃ (im_part : ℝ), (1 + b * Complex.I) / (2 - Complex.I) = im_part * Complex.I) ↔ b = 2 :=
by
  sorry

end pure_imaginary_b_eq_two_l130_130229


namespace ratio_of_medians_to_sides_l130_130331

theorem ratio_of_medians_to_sides (a b c : ℝ) (m_a m_b m_c : ℝ) 
  (h1: m_a = 1/2 * (2 * b^2 + 2 * c^2 - a^2)^(1/2))
  (h2: m_b = 1/2 * (2 * a^2 + 2 * c^2 - b^2)^(1/2))
  (h3: m_c = 1/2 * (2 * a^2 + 2 * b^2 - c^2)^(1/2)) :
  (m_a*m_a + m_b*m_b + m_c*m_c) / (a*a + b*b + c*c) = 3/4 := 
by 
  sorry

end ratio_of_medians_to_sides_l130_130331


namespace line_eq_l130_130117

variables {x x1 x2 y y1 y2 : ℝ}

theorem line_eq (h : x2 ≠ x1 ∧ y2 ≠ y1) : 
  (x - x1) / (x2 - x1) = (y - y1) / (y2 - y1) :=
sorry

end line_eq_l130_130117


namespace correct_statements_l130_130285

-- Define the universal set U as ℤ (integers)
noncomputable def U : Set ℤ := Set.univ

-- Conditions
def is_subset_of_int : Prop := {0} ⊆ (Set.univ : Set ℤ)

def counterexample_subsets (A B : Set ℤ) : Prop :=
  (A = {1, 2} ∧ B = {1, 2, 3}) ∧ (B ∩ (U \ A) ≠ ∅)

def negation_correct_1 : Prop :=
  ¬(∀ x : ℤ, x^2 > 0) ↔ ∃ x : ℤ, x^2 ≤ 0

def negation_correct_2 : Prop :=
  ¬(∀ x : ℤ, x^2 > 0) ↔ ¬(∀ x : ℤ, x^2 < 0)

-- The theorem to prove the equivalence of correct statements
theorem correct_statements :
  (is_subset_of_int ∧
   ∀ A B : Set ℤ, A ⊆ U → B ⊆ U → (A ⊆ B → counterexample_subsets A B) ∧
   negation_correct_1 ∧
   ¬negation_correct_2) ↔
  (true) :=
by 
  sorry

end correct_statements_l130_130285


namespace ribbon_tying_length_l130_130325

theorem ribbon_tying_length :
  let l1 := 36
  let l2 := 42
  let l3 := 48
  let cut1 := l1 / 6
  let cut2 := l2 / 6
  let cut3 := l3 / 6
  let rem1 := l1 - cut1
  let rem2 := l2 - cut2
  let rem3 := l3 - cut3
  let total_rem := rem1 + rem2 + rem3
  let final_length := 97
  let tying_length := total_rem - final_length
  tying_length = 8 :=
by
  sorry

end ribbon_tying_length_l130_130325


namespace age_condition_l130_130416

theorem age_condition (x y z : ℕ) (h1 : x > y) : 
  (z > y) ↔ (y + z > 2 * x) ∧ (∀ x y z, y + z > 2 * x → z > y) := sorry

end age_condition_l130_130416


namespace single_line_points_l130_130678

theorem single_line_points (S : ℝ) (h1 : 6 * S + 4 * (8 * S) = 38000) : S = 1000 :=
by
  sorry

end single_line_points_l130_130678


namespace triangle_side_m_l130_130524

theorem triangle_side_m (a b m : ℝ) (ha : a = 2) (hb : b = 3) (h1 : a + b > m) (h2 : a + m > b) (h3 : b + m > a) :
  (1 < m ∧ m < 5) → m = 3 :=
by
  sorry

end triangle_side_m_l130_130524


namespace min_value_x3y3z2_is_1_over_27_l130_130489

noncomputable def min_value_x3y3z2 (x y z : ℝ) (h : 0 < x ∧ 0 < y ∧ 0 < z) (h' : 1 / x + 1 / y + 1 / z = 9) : ℝ :=
  x^3 * y^3 * z^2

theorem min_value_x3y3z2_is_1_over_27 (x y z : ℝ) (h : 0 < x ∧ 0 < y ∧ 0 < z)
  (h' : 1 / x + 1 / y + 1 / z = 9) : min_value_x3y3z2 x y z h h' = 1 / 27 :=
sorry

end min_value_x3y3z2_is_1_over_27_l130_130489


namespace percentage_of_employees_in_manufacturing_l130_130668

theorem percentage_of_employees_in_manufacturing (d total_degrees : ℝ) (h1 : d = 144) (h2 : total_degrees = 360) :
    (d / total_degrees) * 100 = 40 :=
by
  sorry

end percentage_of_employees_in_manufacturing_l130_130668


namespace evaluate_fraction_l130_130811

theorem evaluate_fraction (a b : ℕ) (h₁ : a = 250) (h₂ : b = 240) :
  1800^2 / (a^2 - b^2) = 660 :=
by 
  sorry

end evaluate_fraction_l130_130811


namespace sum_is_18_less_than_abs_sum_l130_130059

theorem sum_is_18_less_than_abs_sum : 
  (-5 + -4) = (|-5| + |-4| - 18) :=
by
  sorry

end sum_is_18_less_than_abs_sum_l130_130059


namespace parallel_lines_l130_130256

-- Definitions of lines and plane
variable {Line : Type}
variable {Plane : Type}
variable (a b c : Line)
variable (α : Plane)

-- Parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (parallelPlane : Line → Plane → Prop)

-- Given conditions
variable (h1 : parallel a c)
variable (h2 : parallel b c)

-- Theorem statement
theorem parallel_lines (a b c : Line) 
                       (α : Plane) 
                       (parallel : Line → Line → Prop) 
                       (perpendicular : Line → Line → Prop) 
                       (parallelPlane : Line → Plane → Prop)
                       (h1 : parallel a c) 
                       (h2 : parallel b c) : 
                       parallel a b :=
sorry

end parallel_lines_l130_130256


namespace person_walks_distance_l130_130301

theorem person_walks_distance {D t : ℝ} (h1 : 5 * t = D) (h2 : 10 * t = D + 20) : D = 20 :=
by
  sorry

end person_walks_distance_l130_130301


namespace quadratic_distinct_roots_l130_130699

theorem quadratic_distinct_roots (p q : ℚ) :
  (∀ x : ℚ, x^2 + p * x + q = 0 ↔ x = 2 * p ∨ x = p + q) →
  (p = 2 / 3 ∧ q = -8 / 3) :=
by sorry

end quadratic_distinct_roots_l130_130699


namespace equilateral_triangle_area_ratio_l130_130788

theorem equilateral_triangle_area_ratio :
  let side_small := 1
  let perim_small := 3 * side_small
  let total_fencing := 6 * perim_small
  let side_large := total_fencing / 3
  let area_small := (Real.sqrt 3) / 4 * side_small ^ 2
  let area_large := (Real.sqrt 3) / 4 * side_large ^ 2
  let total_area_small := 6 * area_small
  total_area_small / area_large = 1 / 6 :=
by
  sorry

end equilateral_triangle_area_ratio_l130_130788


namespace arithmetic_sequence_seventh_term_l130_130857

theorem arithmetic_sequence_seventh_term
  (a d : ℝ)
  (h_sum : 4 * a + 6 * d = 20)
  (h_fifth : a + 4 * d = 8) :
  a + 6 * d = 10.4 :=
by
  sorry -- proof to be provided

end arithmetic_sequence_seventh_term_l130_130857


namespace smallest_base_l130_130020

theorem smallest_base (b : ℕ) (h1 : b^2 ≤ 125) (h2 : 125 < b^3) : b = 6 := by
  sorry

end smallest_base_l130_130020


namespace sequence_is_periodic_l130_130129

open Nat

def is_periodic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ p > 0, ∀ i, a (i + p) = a i

theorem sequence_is_periodic (a : ℕ → ℕ)
  (h1 : ∀ n, a n < 1988)
  (h2 : ∀ m n, a m + a n ∣ a (m + n)) : is_periodic_sequence a :=
by
  sorry

end sequence_is_periodic_l130_130129


namespace find_different_mass_part_l130_130257

-- Definitions for the parts a1, a2, a3, a4 and their masses
variable {α : Type}
variables (a₁ a₂ a₃ a₄ : α)
variable [LinearOrder α]

-- Definition of the problem conditions
def different_mass_part (a₁ a₂ a₃ a₄ : α) : Prop :=
  (a₁ ≠ a₂ ∨ a₁ ≠ a₃ ∨ a₁ ≠ a₄ ∨ a₂ ≠ a₃ ∨ a₂ ≠ a₄ ∨ a₃ ≠ a₄)

-- Theorem statement assuming we can identify the differing part using two weighings on a pan balance
theorem find_different_mass_part (h : different_mass_part a₁ a₂ a₃ a₄) :
  ∃ (part : α), part = a₁ ∨ part = a₂ ∨ part = a₃ ∨ part = a₄ :=
sorry

end find_different_mass_part_l130_130257


namespace jessica_remaining_time_after_penalties_l130_130128

-- Definitions for the given conditions
def questions_answered : ℕ := 16
def total_questions : ℕ := 80
def time_used_minutes : ℕ := 12
def exam_duration_minutes : ℕ := 60
def penalty_per_incorrect_answer_minutes : ℕ := 2

-- Define the rate of answering questions
def answering_rate : ℚ := questions_answered / time_used_minutes

-- Define the total time needed to answer all questions
def total_time_needed : ℚ := total_questions / answering_rate

-- Define the remaining time after penalties
def remaining_time_after_penalties (x : ℕ) : ℤ :=
  max 0 (0 - penalty_per_incorrect_answer_minutes * x)

-- The theorem to prove
theorem jessica_remaining_time_after_penalties (x : ℕ) : 
  remaining_time_after_penalties x = max 0 (0 - penalty_per_incorrect_answer_minutes * x) := 
by
  sorry

end jessica_remaining_time_after_penalties_l130_130128


namespace sum_of_midpoint_coords_l130_130825

theorem sum_of_midpoint_coords :
  let x1 := 10
  let y1 := 7
  let x2 := 4
  let y2 := 1
  let xm := (x1 + x2) / 2
  let ym := (y1 + y2) / 2
  xm + ym = 11 :=
by
  let x1 := 10
  let y1 := 7
  let x2 := 4
  let y2 := 1
  let xm := (x1 + x2) / 2
  let ym := (y1 + y2) / 2
  sorry

end sum_of_midpoint_coords_l130_130825


namespace toms_restaurant_bill_l130_130090

theorem toms_restaurant_bill (num_adults num_children : ℕ) (meal_cost : ℕ) (total_meals : ℕ) (bill : ℕ) :
  num_adults = 2 ∧ num_children = 5 ∧ meal_cost = 8 ∧ total_meals = num_adults + num_children ∧ bill = total_meals * meal_cost → bill = 56 :=
by sorry

end toms_restaurant_bill_l130_130090


namespace Basel_series_l130_130510

theorem Basel_series :
  (∑' (n : ℕ+), 1 / (n : ℝ)^2) = π^2 / 6 := by sorry

end Basel_series_l130_130510


namespace power_expression_l130_130321

variable {a b : ℝ}

theorem power_expression : (-2 * a^2 * b^3)^3 = -8 * a^6 * b^9 := 
by 
  sorry

end power_expression_l130_130321


namespace find_M_value_l130_130730

-- Statements of the problem conditions and the proof goal
theorem find_M_value (a b c M : ℤ) (h1 : a + b + c = 75) (h2 : a + 4 = M) (h3 : b - 5 = M) (h4 : 3 * c = M) : M = 31 := 
by
  sorry

end find_M_value_l130_130730


namespace expected_value_l130_130072

noncomputable def p : ℝ := 0.25
noncomputable def P_xi_1 : ℝ := 0.24
noncomputable def P_black_bag_b : ℝ := 0.8
noncomputable def P_xi_0 : ℝ := (1 - p) * (1 - P_black_bag_b) * (1 - P_black_bag_b)
noncomputable def P_xi_2 : ℝ := p * (1 - P_black_bag_b) * (1 - P_black_bag_b) + (1 - p) * P_black_bag_b * P_black_bag_b
noncomputable def P_xi_3 : ℝ := p * P_black_bag_b + p * (1 - P_black_bag_b) * P_black_bag_b
noncomputable def E_xi : ℝ := 0 * P_xi_0 + 1 * P_xi_1 + 2 * P_xi_2 + 3 * P_xi_3

theorem expected_value : E_xi = 1.94 := by
  sorry

end expected_value_l130_130072


namespace age_difference_l130_130995

theorem age_difference (a b c : ℕ) (h : a + b = b + c + 18) : a - c = 18 :=
by
  sorry

end age_difference_l130_130995


namespace mother_daughter_age_equality_l130_130228

theorem mother_daughter_age_equality :
  ∀ (x : ℕ), (24 * 12 + 3) + x = 12 * ((-5 : ℤ) + x) → x = 32 := 
by
  intros x h
  sorry

end mother_daughter_age_equality_l130_130228


namespace farmer_profit_l130_130585

noncomputable def profit_earned : ℕ :=
  let pigs := 6
  let sale_price := 300
  let food_cost_per_month := 10
  let months_group1 := 12
  let months_group2 := 16
  let pigs_group1 := 3
  let pigs_group2 := 3
  let total_food_cost := (pigs_group1 * months_group1 * food_cost_per_month) + 
                         (pigs_group2 * months_group2 * food_cost_per_month)
  let total_revenue := pigs * sale_price
  total_revenue - total_food_cost

theorem farmer_profit : profit_earned = 960 := by
  unfold profit_earned
  sorry

end farmer_profit_l130_130585


namespace prism_edges_l130_130021

theorem prism_edges (n : ℕ) (h1 : n > 310) (h2 : n < 320) (h3 : n % 2 = 1) : n = 315 := by
  sorry

end prism_edges_l130_130021


namespace bethany_saw_16_portraits_l130_130605

variable (P S : ℕ)

def bethany_conditions : Prop :=
  S = 4 * P ∧ P + S = 80

theorem bethany_saw_16_portraits (P S : ℕ) (h : bethany_conditions P S) : P = 16 := by
  sorry

end bethany_saw_16_portraits_l130_130605


namespace check_correct_l130_130324

-- Given the conditions
variable (x y : ℕ) (H1 : 10 ≤ x ∧ x ≤ 81) (H2 : y = x + 18)

-- Rewrite the problem and correct answer for verification in Lean
theorem check_correct (Hx : 10 ≤ x ∧ x ≤ 81) (Hy : y = x + 18) : 
  y = 2 * x ↔ x = 18 := 
by
  sorry

end check_correct_l130_130324


namespace classrooms_student_rabbit_difference_l130_130141

-- Definitions from conditions
def students_per_classroom : Nat := 20
def rabbits_per_classroom : Nat := 3
def number_of_classrooms : Nat := 6

-- Theorem statement
theorem classrooms_student_rabbit_difference :
  (students_per_classroom * number_of_classrooms) - (rabbits_per_classroom * number_of_classrooms) = 102 := by
  sorry

end classrooms_student_rabbit_difference_l130_130141


namespace selection_plans_l130_130641

-- Definitions for the students
inductive Student
| A | B | C | D | E | F

open Student

-- Definitions for the subjects
inductive Subject
| Mathematics | Physics | Chemistry | Biology

open Subject

-- A function to count the number of valid selections such that A and B do not participate in Biology.
def countValidSelections : Nat :=
  let totalWays := Nat.factorial 6 / Nat.factorial 2 / Nat.factorial (6 - 4)
  let forbiddenWays := 2 * (Nat.factorial 5 / Nat.factorial 2 / Nat.factorial (5 - 3))
  totalWays - forbiddenWays

theorem selection_plans :
  countValidSelections = 240 :=
by
  sorry

end selection_plans_l130_130641


namespace least_beans_l130_130150

-- Define the conditions 
variables (r b : ℕ)

-- State the theorem 
theorem least_beans (h1 : r ≥ 2 * b + 8) (h2 : r ≤ 3 * b) : b ≥ 8 :=
by
  sorry

end least_beans_l130_130150


namespace fleas_cannot_reach_final_positions_l130_130159

structure Point2D :=
  (x : ℝ)
  (y : ℝ)

def initial_A : Point2D := ⟨0, 0⟩
def initial_B : Point2D := ⟨1, 0⟩
def initial_C : Point2D := ⟨0, 1⟩

def area (A B C : Point2D) : ℝ :=
  0.5 * abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))

def final_A : Point2D := ⟨1, 0⟩
def final_B : Point2D := ⟨-1, 0⟩
def final_C : Point2D := ⟨0, 1⟩

theorem fleas_cannot_reach_final_positions : 
    ¬ (∃ (flea_move_sequence : List (Point2D → Point2D)), 
    area initial_A initial_B initial_C = area final_A final_B final_C) :=
by 
  sorry

end fleas_cannot_reach_final_positions_l130_130159


namespace missing_number_is_twelve_l130_130172

theorem missing_number_is_twelve
  (x : ℤ)
  (h : 10010 - x * 3 * 2 = 9938) :
  x = 12 :=
sorry

end missing_number_is_twelve_l130_130172


namespace minor_axis_length_l130_130791

theorem minor_axis_length {x y : ℝ} (h : x^2 / 16 + y^2 / 9 = 1) : 6 = 6 :=
by
  sorry

end minor_axis_length_l130_130791


namespace john_needs_29_planks_for_house_wall_l130_130704

def total_number_of_planks (large_planks small_planks : ℕ) : ℕ :=
  large_planks + small_planks

theorem john_needs_29_planks_for_house_wall :
  total_number_of_planks 12 17 = 29 :=
by
  sorry

end john_needs_29_planks_for_house_wall_l130_130704


namespace rational_iff_geometric_progression_l130_130720

theorem rational_iff_geometric_progression :
  (∃ x a b c : ℤ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ (x + a)*(x + c) = (x + b)^2) ↔
  (∃ x : ℚ, ∃ a b c : ℤ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ (x + (a : ℚ))*(x + (c : ℚ)) = (x + (b : ℚ))^2) :=
sorry

end rational_iff_geometric_progression_l130_130720


namespace broken_seashells_count_l130_130948

def total_seashells : ℕ := 7
def unbroken_seashells : ℕ := 3

theorem broken_seashells_count : (total_seashells - unbroken_seashells) = 4 := by
  sorry

end broken_seashells_count_l130_130948


namespace number_of_candidates_l130_130913

theorem number_of_candidates
  (n : ℕ)
  (h : n * (n - 1) = 132) : 
  n = 12 :=
sorry

end number_of_candidates_l130_130913


namespace evaluate_expression_l130_130036

theorem evaluate_expression : (36 + 12) / (6 - (2 + 1)) = 16 := by
  sorry

end evaluate_expression_l130_130036


namespace pairs_satisfaction_l130_130858

-- Definitions for the conditions given
def condition1 (x y : ℝ) : Prop := y = (x + 2)^2
def condition2 (x y : ℝ) : Prop := x * y + 2 * y = 2

-- The statement that we need to prove
theorem pairs_satisfaction : 
  (∃ x y : ℝ, condition1 x y ∧ condition2 x y) ∧ 
  (∃ x1 x2 : ℂ, x^2 + -2*x + 1 = 0 ∧ ¬∃ (y : ℝ), y = (x1 + 2)^2 ∨ y = (x2 + 2)^2) :=
by
  sorry

end pairs_satisfaction_l130_130858


namespace find_number_of_each_coin_l130_130135

-- Define the number of coins
variables (n d q : ℕ)

-- Given conditions
axiom twice_as_many_nickels_as_quarters : n = 2 * q
axiom same_number_of_dimes_as_quarters : d = q
axiom total_value_of_coins : 5 * n + 10 * d + 25 * q = 1520

-- Statement to prove
theorem find_number_of_each_coin :
  q = 304 / 9 ∧
  n = 2 * (304 / 9) ∧
  d = 304 / 9 :=
sorry

end find_number_of_each_coin_l130_130135


namespace not_equal_d_l130_130477

def frac_14_over_6 : ℚ := 14 / 6
def mixed_2_and_1_3rd : ℚ := 2 + 1 / 3
def mixed_neg_2_and_1_3rd : ℚ := -(2 + 1 / 3)
def mixed_3_and_1_9th : ℚ := 3 + 1 / 9
def mixed_2_and_4_12ths : ℚ := 2 + 4 / 12
def target_fraction : ℚ := 7 / 3

theorem not_equal_d : mixed_3_and_1_9th ≠ target_fraction :=
by sorry

end not_equal_d_l130_130477


namespace PS_length_correct_l130_130441

variable {Triangle : Type}

noncomputable def PR := 15

noncomputable def PS_length (PS SR : ℝ) (PR : ℝ) : Prop :=
  PS + SR = PR ∧ (PS / SR) = (3 / 4)

theorem PS_length_correct : 
  ∃ PS SR : ℝ, PS_length PS SR PR ∧ PS = (45 / 7) :=
sorry

end PS_length_correct_l130_130441


namespace algorithm_characteristics_l130_130780

theorem algorithm_characteristics (finiteness : Prop) (definiteness : Prop) (output_capability : Prop) (unique : Prop) 
  (h1 : finiteness = true) 
  (h2 : definiteness = true) 
  (h3 : output_capability = true) 
  (h4 : unique = false) : 
  incorrect_statement = unique := 
by
  sorry

end algorithm_characteristics_l130_130780


namespace find_m_parallel_l130_130615

noncomputable def is_parallel (A1 B1 C1 A2 B2 C2 : ℝ) : Prop :=
  -(A1 / B1) = -(A2 / B2)

theorem find_m_parallel : ∃ m : ℝ, is_parallel (m-1) 3 m 1 (m+1) 2 ∧ m = -2 :=
by
  unfold is_parallel
  exists (-2 : ℝ)
  sorry

end find_m_parallel_l130_130615


namespace polar_curve_is_parabola_l130_130682

theorem polar_curve_is_parabola (ρ θ : ℝ) (h : 3 * ρ * Real.sin θ ^ 2 + Real.cos θ = 0) : ∃ (x y : ℝ), x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ ∧ 3 * y ^ 2 + x = 0 :=
by
  sorry

end polar_curve_is_parabola_l130_130682


namespace sam_bought_17_mystery_books_l130_130805

def adventure_books := 13
def used_books := 15
def new_books := 15
def total_books := used_books + new_books
def mystery_books := total_books - adventure_books

theorem sam_bought_17_mystery_books : mystery_books = 17 := by
  sorry

end sam_bought_17_mystery_books_l130_130805


namespace equivalence_negation_l130_130341

-- Define irrational numbers
def is_irrational (x : ℝ) : Prop :=
  ¬ (∃ q : ℚ, x = q)

-- Define rational numbers
def is_rational (x : ℝ) : Prop :=
  ∃ q : ℚ, x = q

-- Original proposition: There exists an irrational number whose square is rational
def original_proposition : Prop :=
  ∃ x : ℝ, is_irrational x ∧ is_rational (x * x)

-- Negation of the original proposition
def negation_of_proposition : Prop :=
  ∀ x : ℝ, is_irrational x → ¬is_rational (x * x)

-- Proof statement that the negation of the original proposition is equivalent to "Every irrational number has a square that is not rational"
theorem equivalence_negation :
  (¬ original_proposition) ↔ negation_of_proposition :=
sorry

end equivalence_negation_l130_130341


namespace quadratic_no_real_roots_range_l130_130874

theorem quadratic_no_real_roots_range (k : ℝ) : 
  (∀ x : ℝ, ¬ (x^2 + 2 * x - k = 0)) ↔ k < -1 :=
by
  sorry

end quadratic_no_real_roots_range_l130_130874


namespace oxygen_atoms_in_compound_l130_130609

theorem oxygen_atoms_in_compound (K_weight Br_weight O_weight molecular_weight : ℕ) 
    (hK : K_weight = 39) (hBr : Br_weight = 80) (hO : O_weight = 16) (hMW : molecular_weight = 168) 
    (n : ℕ) :
    168 = 39 + 80 + n * 16 → n = 3 :=
by
  intros h
  sorry

end oxygen_atoms_in_compound_l130_130609


namespace value_of_a_plus_b_l130_130375

theorem value_of_a_plus_b (a b x y : ℝ) 
  (h1 : 2 * x + 4 * y = 20)
  (h2 : a * x + b * y = 1)
  (h3 : 2 * x - y = 5)
  (h4 : b * x + a * y = 6) : a + b = 1 := 
sorry

end value_of_a_plus_b_l130_130375


namespace cost_option1_eq_cost_option2_eq_option1_final_cost_eq_option2_final_cost_eq_option1_more_cost_effective_optimal_cost_eq_l130_130643

section price_calculations

variables {x : ℕ} (hx : x > 20)

-- Definitions based on the problem statement.
def suit_price : ℕ := 400
def tie_price : ℕ := 80

def option1_cost (x : ℕ) : ℕ :=
  20 * suit_price + tie_price * (x - 20)

def option2_cost (x : ℕ) : ℕ :=
  (20 * suit_price + tie_price * x) * 9 / 10

def option1_final_cost := option1_cost 30
def option2_final_cost := option2_cost 30

def optimal_cost : ℕ := 20 * suit_price + tie_price * 10 * 9 / 10

-- Proof obligations
theorem cost_option1_eq : option1_cost x = 80 * x + 6400 :=
by sorry

theorem cost_option2_eq : option2_cost x = 72 * x + 7200 :=
by sorry

theorem option1_final_cost_eq : option1_final_cost = 8800 :=
by sorry

theorem option2_final_cost_eq : option2_final_cost = 9360 :=
by sorry

theorem option1_more_cost_effective : option1_final_cost < option2_final_cost :=
by sorry

theorem optimal_cost_eq : optimal_cost = 8720 :=
by sorry

end price_calculations

end cost_option1_eq_cost_option2_eq_option1_final_cost_eq_option2_final_cost_eq_option1_more_cost_effective_optimal_cost_eq_l130_130643


namespace probability_of_event_B_given_A_l130_130850

-- Definition of events and probability
noncomputable def prob_event_B_given_A : ℝ :=
  let total_outcomes := 36
  let outcomes_A := 30
  let outcomes_B_given_A := 10
  outcomes_B_given_A / outcomes_A

-- Theorem statement
theorem probability_of_event_B_given_A : prob_event_B_given_A = 1 / 3 := by
  sorry

end probability_of_event_B_given_A_l130_130850


namespace geom_seq_common_ratio_l130_130207

theorem geom_seq_common_ratio (a1 : ℤ) (S3 : ℚ) (q : ℚ) (hq : -2 * (1 + q + q^2) = - (7 / 2)) : 
  q = 1 / 2 ∨ q = -3 / 2 :=
sorry

end geom_seq_common_ratio_l130_130207


namespace rounding_estimation_correct_l130_130304

theorem rounding_estimation_correct (a b d : ℕ)
  (ha : a > 0) (hb : b > 0) (hd : d > 0)
  (a_round : ℕ) (b_round : ℕ) (d_round : ℕ)
  (h_round_a : a_round ≥ a) (h_round_b : b_round ≤ b) (h_round_d : d_round ≤ d) :
  (Real.sqrt (a_round / b_round) - Real.sqrt d_round) > (Real.sqrt (a / b) - Real.sqrt d) :=
by
  sorry

end rounding_estimation_correct_l130_130304


namespace missing_number_unique_l130_130176

theorem missing_number_unique (x : ℤ) 
  (h : |9 - x * (3 - 12)| - |5 - 11| = 75) : 
  x = 8 :=
sorry

end missing_number_unique_l130_130176


namespace cookie_weight_l130_130917

theorem cookie_weight :
  ∀ (pounds_per_box cookies_per_box ounces_per_pound : ℝ),
    pounds_per_box = 40 →
    cookies_per_box = 320 →
    ounces_per_pound = 16 →
    (pounds_per_box * ounces_per_pound) / cookies_per_box = 2 := 
by 
  intros pounds_per_box cookies_per_box ounces_per_pound hpounds hcookies hounces
  rw [hpounds, hcookies, hounces]
  norm_num

end cookie_weight_l130_130917


namespace prob_correct_l130_130384

-- Define percentages as ratio values
def prob_beginner_excel : ℝ := 0.35
def prob_intermediate_excel : ℝ := 0.25
def prob_advanced_excel : ℝ := 0.20
def prob_no_excel : ℝ := 0.20

def prob_day_shift : ℝ := 0.70
def prob_night_shift : ℝ := 0.30

def prob_weekend : ℝ := 0.40
def prob_not_weekend : ℝ := 0.60

-- Define the target probability calculation
def prob_intermediate_or_advanced_excel : ℝ := prob_intermediate_excel + prob_advanced_excel
def prob_combined : ℝ := prob_intermediate_or_advanced_excel * prob_night_shift * prob_not_weekend

-- The proof problem statement
theorem prob_correct : prob_combined = 0.081 :=
by
  sorry

end prob_correct_l130_130384


namespace total_amount_l130_130960

theorem total_amount (A B C : ℤ) (S : ℤ) (h_ratio : 100 * B = 45 * A ∧ 100 * C = 30 * A) (h_B : B = 6300) : S = 24500 := by
  sorry

end total_amount_l130_130960


namespace least_multiple_of_11_not_lucky_l130_130212

-- Define what it means for a number to be a lucky integer
def is_lucky (n : ℕ) : Prop :=
  n % (n.digits 10).sum = 0

-- Define what it means for a number to be a multiple of 11
def is_multiple_of_11 (n : ℕ) : Prop :=
  n % 11 = 0

-- State the problem: the least positive multiple of 11 that is not a lucky integer is 132
theorem least_multiple_of_11_not_lucky :
  ∃ n : ℕ, is_multiple_of_11 n ∧ ¬ is_lucky n ∧ n = 132 :=
sorry

end least_multiple_of_11_not_lucky_l130_130212


namespace problem_solution_l130_130769
open Real

theorem problem_solution (a b c : ℝ) (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hc : 0 < c ∧ c < 1) :
  a * (1 - b) ≤ 1 / 4 ∨ b * (1 - c) ≤ 1 / 4 ∨ c * (1 - a) ≤ 1 / 4 :=
by
  sorry

end problem_solution_l130_130769


namespace log_x_y_eq_sqrt_3_l130_130115

variable (x y z : ℝ)
variable (hx : 1 < x) (hy : 1 < y) (hz : 1 < z)
variable (h1 : x ^ (Real.log z / Real.log y) = 2)
variable (h2 : y ^ (Real.log x / Real.log y) = 4)
variable (h3 : z ^ (Real.log y / Real.log x) = 8)

theorem log_x_y_eq_sqrt_3 : Real.log y / Real.log x = Real.sqrt 3 :=
by
  sorry

end log_x_y_eq_sqrt_3_l130_130115


namespace minimum_value_l130_130803

noncomputable def f : ℝ → ℝ
| x => if h : 0 < x ∧ x ≤ 1 then x^2 - x else
         if h : 1 < x ∧ x ≤ 2 then -2 * (x - 1)^2 + 6 * (x - 1) - 5
         else 0 -- extend as appropriate outside given ranges

noncomputable def g (x : ℝ) : ℝ := x - 1

theorem minimum_value (x_1 x_2 : ℝ) (h1 : 1 < x_1 ∧ x_1 ≤ 2) : 
  (x_1 - x_2)^2 + (f x_1 - g x_2)^2 = 49 / 128 :=
sorry

end minimum_value_l130_130803


namespace general_term_formula_l130_130057

noncomputable def S (n : ℕ) : ℕ := 2^n - 1
noncomputable def a (n : ℕ) : ℕ := 2^(n-1)

theorem general_term_formula (n : ℕ) (hn : n > 0) : 
    a n = S n - S (n - 1) := 
by 
  sorry

end general_term_formula_l130_130057


namespace total_hours_difference_l130_130053

-- Definitions based on conditions
def hours_learning_english := 6
def hours_learning_chinese := 2
def hours_learning_spanish := 3
def hours_learning_french := 1

-- Calculation of total time spent on English and Chinese
def total_hours_english_chinese := hours_learning_english + hours_learning_chinese

-- Calculation of total time spent on Spanish and French
def total_hours_spanish_french := hours_learning_spanish + hours_learning_french

-- Calculation of the difference in hours spent
def hours_difference := total_hours_english_chinese - total_hours_spanish_french

-- Statement to prove
theorem total_hours_difference : hours_difference = 4 := by
  sorry

end total_hours_difference_l130_130053


namespace sequence_term_number_l130_130079

theorem sequence_term_number (n : ℕ) (a_n : ℕ) (h : a_n = 2 * n ^ 2 - 3) : a_n = 125 → n = 8 :=
by
  sorry

end sequence_term_number_l130_130079


namespace kids_outside_l130_130498

theorem kids_outside (s t n c : ℕ)
  (h1 : s = 644997)
  (h2 : t = 893835)
  (h3 : n = 1538832)
  (h4 : (n - s) = t) : c = 0 :=
by {
  sorry
}

end kids_outside_l130_130498


namespace range_of_f_when_a_neg_2_is_0_to_4_and_bounded_range_of_a_if_f_bounded_by_4_l130_130688

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 + a * Real.sin x - Real.cos x ^ 2

theorem range_of_f_when_a_neg_2_is_0_to_4_and_bounded :
  (∀ x : ℝ, 0 ≤ f (-2) x ∧ f (-2) x ≤ 4) :=
sorry

theorem range_of_a_if_f_bounded_by_4 :
  (∀ x : ℝ, abs (f a x) ≤ 4) → (-2 ≤ a ∧ a ≤ 2) :=
sorry

end range_of_f_when_a_neg_2_is_0_to_4_and_bounded_range_of_a_if_f_bounded_by_4_l130_130688


namespace solve_equation_l130_130552

theorem solve_equation (x : ℝ) : (x + 2)^2 - 5 * (x + 2) = 0 ↔ (x = -2 ∨ x = 3) :=
by sorry

end solve_equation_l130_130552


namespace min_area_OBX_l130_130950

structure Point : Type :=
  (x : ℤ)
  (y : ℤ)

def O : Point := ⟨0, 0⟩
def B : Point := ⟨11, 8⟩

def area_triangle (A B C : Point) : ℚ :=
  1 / 2 * abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))

def in_rectangle (X : Point) : Prop :=
  0 ≤ X.x ∧ X.x ≤ 11 ∧ 0 ≤ X.y ∧ X.y ≤ 8

theorem min_area_OBX : ∃ (X : Point), in_rectangle X ∧ area_triangle O B X = 1 / 2 :=
sorry

end min_area_OBX_l130_130950


namespace triple_sum_of_45_point_2_and_one_fourth_l130_130773

theorem triple_sum_of_45_point_2_and_one_fourth : 
  (3 * (45.2 + 0.25)) = 136.35 :=
by
  sorry

end triple_sum_of_45_point_2_and_one_fourth_l130_130773


namespace problem1_problem2_solution_l130_130841

noncomputable def trig_expr : ℝ :=
  3 * Real.tan (30 * Real.pi / 180) - (Real.tan (45 * Real.pi / 180))^2 + 2 * Real.sin (60 * Real.pi / 180)

theorem problem1 : trig_expr = 2 * Real.sqrt 3 - 1 :=
by
  -- Proof omitted
  sorry

noncomputable def quad_eq (x : ℝ) : Prop := 
  (3*x - 1) * (x + 2) = 11*x - 4

theorem problem2_solution (x : ℝ) : quad_eq x ↔ (x = (3 + Real.sqrt 3) / 3 ∨ x = (3 - Real.sqrt 3) / 3) :=
by
  -- Proof omitted
  sorry

end problem1_problem2_solution_l130_130841


namespace pieces_of_chocolate_left_l130_130242

theorem pieces_of_chocolate_left (initial_boxes : ℕ) (given_away_boxes : ℕ) (pieces_per_box : ℕ) 
    (h1 : initial_boxes = 14) (h2 : given_away_boxes = 8) (h3 : pieces_per_box = 3) : 
    (initial_boxes - given_away_boxes) * pieces_per_box = 18 := 
by 
  -- The proof will be here
  sorry

end pieces_of_chocolate_left_l130_130242


namespace square_same_area_as_rectangle_l130_130957

theorem square_same_area_as_rectangle (l w : ℝ) (rect_area sq_side : ℝ) :
  l = 25 → w = 9 → rect_area = l * w → sq_side^2 = rect_area → sq_side = 15 :=
by
  intros h_l h_w h_rect_area h_sq_area
  rw [h_l, h_w] at h_rect_area
  sorry

end square_same_area_as_rectangle_l130_130957


namespace gcf_75_90_l130_130004

theorem gcf_75_90 : Nat.gcd 75 90 = 15 :=
by
  sorry

end gcf_75_90_l130_130004


namespace three_legged_extraterrestrials_l130_130966

-- Define the conditions
variables (x y : ℕ)

-- Total number of heads
def heads_equation := x + y = 300

-- Total number of legs
def legs_equation := 3 * x + 4 * y = 846

theorem three_legged_extraterrestrials : heads_equation x y ∧ legs_equation x y → x = 246 :=
by
  sorry

end three_legged_extraterrestrials_l130_130966


namespace no_prime_divisible_by_77_l130_130989

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_prime_divisible_by_77 :
  ∀ p : ℕ, is_prime p → ¬ (77 ∣ p) :=
by
  intros p h_prime h_div
  sorry

end no_prime_divisible_by_77_l130_130989


namespace ratio_of_speeds_l130_130076

def eddy_time := 3
def eddy_distance := 480
def freddy_time := 4
def freddy_distance := 300

def eddy_speed := eddy_distance / eddy_time
def freddy_speed := freddy_distance / freddy_time

theorem ratio_of_speeds : (eddy_speed / freddy_speed) = 32 / 15 :=
by
  sorry

end ratio_of_speeds_l130_130076


namespace smallest_positive_integer_x_l130_130193

theorem smallest_positive_integer_x :
  ∃ (x : ℕ), 0 < x ∧ (45 * x + 13) % 17 = 5 % 17 ∧ ∀ y : ℕ, 0 < y ∧ (45 * y + 13) % 17 = 5 % 17 → y ≥ x := 
sorry

end smallest_positive_integer_x_l130_130193


namespace unique_decomposition_of_two_reciprocals_l130_130797

theorem unique_decomposition_of_two_reciprocals (p : ℕ) (hp : Nat.Prime p) (hp_ne_two : p ≠ 2) :
  ∃ (x y : ℕ), x ≠ y ∧ (1 / (x : ℝ) + 1 / (y : ℝ) = 2 / (p : ℝ)) := sorry

end unique_decomposition_of_two_reciprocals_l130_130797


namespace solving_inequality_l130_130391

theorem solving_inequality (x : ℝ) : 
  (x > 2 ∨ x < -2 ∨ (-1 < x ∧ x < 1)) ↔ ((x^2 - 4) / (x^2 - 1) > 0) :=
by 
  sorry

end solving_inequality_l130_130391


namespace quadratic_real_solutions_l130_130367

theorem quadratic_real_solutions (m : ℝ) :
  (∃ (x : ℝ), m * x^2 + 2 * x + 1 = 0) ↔ (m ≤ 1 ∧ m ≠ 0) :=
by
  sorry

end quadratic_real_solutions_l130_130367


namespace find_replaced_weight_l130_130157

-- Define the conditions and the hypothesis
def replaced_weight (W : ℝ) : Prop :=
  let avg_increase := 2.5
  let num_persons := 8
  let new_weight := 85
  (new_weight - W) = num_persons * avg_increase

-- Define the statement we aim to prove
theorem find_replaced_weight : replaced_weight 65 :=
by
  -- proof goes here
  sorry

end find_replaced_weight_l130_130157


namespace erica_pie_percentage_l130_130983

theorem erica_pie_percentage (a c : ℚ) (ha : a = 1/5) (hc : c = 3/4) : 
  (a + c) * 100 = 95 := 
sorry

end erica_pie_percentage_l130_130983


namespace algebra_correct_option_B_l130_130003

theorem algebra_correct_option_B (a b c : ℝ) (h : b * (c^2 + 1) ≠ 0) : 
  (a * (c^2 + 1)) / (b * (c^2 + 1)) = a / b := 
by
  -- Skipping the proof to focus on the statement
  sorry

end algebra_correct_option_B_l130_130003


namespace charlie_age_l130_130594

variable (J C B : ℝ)

def problem_statement :=
  J = C + 12 ∧ C = B + 7 ∧ J = 3 * B → C = 18

theorem charlie_age : problem_statement J C B :=
by
  sorry

end charlie_age_l130_130594


namespace percentage_of_books_returned_l130_130749

theorem percentage_of_books_returned
  (initial_books : ℕ) (end_books : ℕ) (loaned_books : ℕ) (returned_books_percentage : ℚ) 
  (h1 : initial_books = 75) 
  (h2 : end_books = 68) 
  (h3 : loaned_books = 20)
  (h4 : returned_books_percentage = (end_books - (initial_books - loaned_books)) * 100 / loaned_books):
  returned_books_percentage = 65 := 
by
  sorry

end percentage_of_books_returned_l130_130749


namespace quotient_remainder_l130_130333

theorem quotient_remainder (x y : ℕ) (hx : 0 ≤ x) (hy : 0 < y) : 
  ∃ q r : ℕ, q ≥ 0 ∧ 0 ≤ r ∧ r < y ∧ x = q * y + r := by
  sorry

end quotient_remainder_l130_130333


namespace distance_calculation_l130_130672

-- Define the given constants
def time_minutes : ℕ := 30
def average_speed : ℕ := 1
def seconds_per_minute : ℕ := 60

-- Define the total time in seconds
def time_seconds : ℕ := time_minutes * seconds_per_minute

-- The proof goal: that the distance covered is 1800 meters
theorem distance_calculation :
  time_seconds * average_speed = 1800 := by
  -- Calculation steps (using axioms and known values)
  sorry

end distance_calculation_l130_130672


namespace find_a_l130_130512

theorem find_a (a : ℝ) 
  (h1 : ∀ x y : ℝ, 2*x + y - 2 = 0)
  (h2 : ∀ x y : ℝ, a*x + 4*y + 1 = 0)
  (perpendicular : ∀ (m1 m2 : ℝ), m1 = -2 → m2 = -a/4 → m1 * m2 = -1) :
  a = -2 :=
sorry

end find_a_l130_130512


namespace proportion_of_triumphal_arch_photographs_l130_130426

-- Define the constants
variables (x y z t : ℕ) -- x = castles, y = triumphal arches, z = waterfalls, t = cathedrals

-- The conditions
axiom half_photographed : t + x + y + z = (3*y + 2*x + 2*z + y) / 2
axiom three_times_cathedrals : ∃ (a : ℕ), t = 3 * a ∧ y = a
axiom same_castles_waterfalls : ∃ (b : ℕ), t + z = x + y
axiom quarter_photographs_castles : x = (t + x + y + z) / 4
axiom second_castle_frequency : t + z = 2 * x
axiom every_triumphal_arch_photographed : ∀ (c : ℕ), y = c ∧ y = c

theorem proportion_of_triumphal_arch_photographs : 
  ∃ (p : ℚ), p = 1 / 4 ∧ p = y / ((t + x + y + z) / 2) :=
sorry

end proportion_of_triumphal_arch_photographs_l130_130426


namespace shaded_region_area_l130_130221

def area_of_square (side : ℕ) : ℕ := side * side

def area_of_triangle (base height : ℕ) : ℕ := (base * height) / 2

def combined_area_of_triangles (base height : ℕ) : ℕ := 2 * area_of_triangle base height

def shaded_area (square_side : ℕ) (triangle_base triangle_height : ℕ) : ℕ :=
  area_of_square square_side - combined_area_of_triangles triangle_base triangle_height

theorem shaded_region_area (h₁ : area_of_square 40 = 1600)
                          (h₂ : area_of_triangle 30 30 = 450)
                          (h₃ : combined_area_of_triangles 30 30 = 900) :
  shaded_area 40 30 30 = 700 :=
by
  sorry

end shaded_region_area_l130_130221
