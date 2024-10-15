import Mathlib

namespace NUMINAMATH_GPT_domain_of_function_l1515_151597

theorem domain_of_function :
  (∀ x : ℝ, (2 * Real.sin x - 1 > 0) ∧ (1 - 2 * Real.cos x ≥ 0) ↔
    ∃ k : ℤ, 2 * k * Real.pi + Real.pi / 3 ≤ x ∧ x < 2 * k * Real.pi + 5 * Real.pi / 6) :=
sorry

end NUMINAMATH_GPT_domain_of_function_l1515_151597


namespace NUMINAMATH_GPT_fraction_equality_l1515_151534

theorem fraction_equality :
  (2 - (1 / 2) * (1 - (1 / 4))) / (2 - (1 - (1 / 3))) = 39 / 32 := 
  sorry

end NUMINAMATH_GPT_fraction_equality_l1515_151534


namespace NUMINAMATH_GPT_number_of_possible_measures_l1515_151553

theorem number_of_possible_measures (A B : ℕ) (h1 : A > 0) (h2 : B > 0) (h3 : A + B = 180) (h4 : ∃ k : ℕ, k ≥ 1 ∧ A = k * B) : 
  ∃ n : ℕ, n = 17 :=
sorry

end NUMINAMATH_GPT_number_of_possible_measures_l1515_151553


namespace NUMINAMATH_GPT_initial_value_amount_l1515_151518

theorem initial_value_amount (P : ℝ) 
  (h1 : ∀ t, t ≥ 0 → t = P * (1 + (1/8)) ^ t) 
  (h2 : P * (1 + (1/8)) ^ 2 = 105300) : 
  P = 83200 := 
sorry

end NUMINAMATH_GPT_initial_value_amount_l1515_151518


namespace NUMINAMATH_GPT_part1_part2_l1515_151501

def f (x m : ℝ) : ℝ := |x - 1| - |2 * x + m|

theorem part1 (x : ℝ) (m : ℝ) (h : m = -4) : 
    f x m < 0 ↔ x < 5 / 3 ∨ x > 3 := 
by 
  sorry

theorem part2 (x : ℝ) (h : 1 < x) (h' : ∀ x, 1 < x → f x m < 0) : 
    m ≥ -2 :=
by 
  sorry

end NUMINAMATH_GPT_part1_part2_l1515_151501


namespace NUMINAMATH_GPT_Ahmad_eight_steps_l1515_151504

def reach_top (n : Nat) (holes : List Nat) : Nat := sorry

theorem Ahmad_eight_steps (h : reach_top 8 [6] = 8) : True := by 
  trivial

end NUMINAMATH_GPT_Ahmad_eight_steps_l1515_151504


namespace NUMINAMATH_GPT_find_x_l1515_151552

theorem find_x (x : ℝ) :
  (x^2 - 7 * x + 12) / (x^2 - 9 * x + 20) = (x^2 - 4 * x - 21) / (x^2 - 5 * x - 24) -> x = 11 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1515_151552


namespace NUMINAMATH_GPT_range_of_m_l1515_151526

theorem range_of_m (f : ℝ → ℝ) (m : ℝ) (h_even : ∀ x, f x = f (-x)) 
 (h_decreasing : ∀ {x y}, 0 ≤ x → 0 ≤ y → x ≤ y → f y ≤ f x)
 (h_condition : ∀ x, 1 ≤ x → x ≤ 3 → f (2 * m * x - Real.log x - 3) ≥ 2 * f 3 - f (Real.log x + 3 - 2 * m * x)) :
  m ∈ Set.Icc (1 / (2 * Real.exp 1)) ((Real.log 3 + 6) / 6) :=
sorry

end NUMINAMATH_GPT_range_of_m_l1515_151526


namespace NUMINAMATH_GPT_complement_unions_subset_condition_l1515_151575

open Set

-- Condition Definitions
def A : Set ℝ := {x | 1 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 3 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | x < a + 1}

-- Questions Translated to Lean Statements
theorem complement_unions (U : Set ℝ)
  (hU : U = univ) : (compl A ∪ compl B) = compl (A ∩ B) := by sorry

theorem subset_condition (a : ℝ)
  (h : B ⊆ C a) : a ≥ 8 := by sorry

end NUMINAMATH_GPT_complement_unions_subset_condition_l1515_151575


namespace NUMINAMATH_GPT_find_sin_E_floor_l1515_151537

variable {EF GH EH FG : ℝ}
variable (E G : ℝ)

-- Conditions from the problem
def is_convex_quadrilateral (EF GH EH FG : ℝ) : Prop := true
def angles_congruent (E G : ℝ) : Prop := E = G
def sides_equal (EF GH : ℝ) : Prop := EF = GH ∧ EF = 200
def sides_not_equal (EH FG : ℝ) : Prop := EH ≠ FG
def perimeter (EF GH EH FG : ℝ) : Prop := EF + GH + EH + FG = 800

-- The theorem to be proved
theorem find_sin_E_floor (h_convex : is_convex_quadrilateral EF GH EH FG)
                         (h_angles : angles_congruent E G)
                         (h_sides : sides_equal EF GH)
                         (h_sides_ne : sides_not_equal EH FG)
                         (h_perimeter : perimeter EF GH EH FG) :
  ⌊ 1000 * Real.sin E ⌋ = 0 := by
  sorry

end NUMINAMATH_GPT_find_sin_E_floor_l1515_151537


namespace NUMINAMATH_GPT_range_of_k_l1515_151550

open Real

noncomputable def f (x : ℝ) : ℝ := x^2 + x - 1

noncomputable def g (x : ℝ) : ℝ := x^2 - 1

noncomputable def h (x : ℝ) : ℝ := x

theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, x ≠ 0 → g (k * x + k / x) < g (x^2 + 1 / x^2 + 1)) ↔ (-3 / 2 < k ∧ k < 3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_l1515_151550


namespace NUMINAMATH_GPT_total_area_of_forest_and_fields_l1515_151503

theorem total_area_of_forest_and_fields (r p k : ℝ) (h1 : k = 12) 
  (h2 : r^2 + 4 * p^2 + 45 = 12 * k) :
  (r^2 + 4 * p^2 + 12 * k = 135) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_total_area_of_forest_and_fields_l1515_151503


namespace NUMINAMATH_GPT_z_in_second_quadrant_l1515_151582

def is_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

theorem z_in_second_quadrant (z : ℂ) (i : ℂ) (hi : i^2 = -1) (h : z * (1 + i^3) = i) : 
  is_second_quadrant z := by
  sorry

end NUMINAMATH_GPT_z_in_second_quadrant_l1515_151582


namespace NUMINAMATH_GPT_teresa_age_when_michiko_born_l1515_151506

def conditions (T M Michiko K Yuki : ℕ) : Prop := 
  T = 59 ∧ 
  M = 71 ∧ 
  M - Michiko = 38 ∧ 
  K = Michiko - 4 ∧ 
  Yuki = K - 3 ∧ 
  (Yuki + 3) - (26 - 25) = 25

theorem teresa_age_when_michiko_born :
  ∃ T M Michiko K Yuki, conditions T M Michiko K Yuki → T - Michiko = 26 :=
  by
  sorry

end NUMINAMATH_GPT_teresa_age_when_michiko_born_l1515_151506


namespace NUMINAMATH_GPT_sin_cos_identity_l1515_151587

variables (α : ℝ)

def tan_pi_add_alpha (α : ℝ) : Prop := Real.tan (Real.pi + α) = 3

theorem sin_cos_identity (h : tan_pi_add_alpha α) : 
  Real.sin (-α) * Real.cos (Real.pi - α) = 3 / 10 :=
sorry

end NUMINAMATH_GPT_sin_cos_identity_l1515_151587


namespace NUMINAMATH_GPT_iron_balls_molded_l1515_151581

-- Define the dimensions of the iron bar
def length_bar : ℝ := 12
def width_bar : ℝ := 8
def height_bar : ℝ := 6

-- Define the volume calculations
def volume_iron_bar : ℝ := length_bar * width_bar * height_bar
def number_of_bars : ℝ := 10
def total_volume_bars : ℝ := volume_iron_bar * number_of_bars
def volume_iron_ball : ℝ := 8

-- Define the goal statement
theorem iron_balls_molded : total_volume_bars / volume_iron_ball = 720 :=
by
  -- Proof is to be filled in here
  sorry

end NUMINAMATH_GPT_iron_balls_molded_l1515_151581


namespace NUMINAMATH_GPT_cost_of_lamps_and_bulbs_l1515_151519

theorem cost_of_lamps_and_bulbs : 
    let lamp_cost := 7
    let bulb_cost := lamp_cost - 4
    let total_cost := (lamp_cost * 2) + (bulb_cost * 6)
    total_cost = 32 := by
  let lamp_cost := 7
  let bulb_cost := lamp_cost - 4
  let total_cost := (lamp_cost * 2) + (bulb_cost * 6)
  sorry

end NUMINAMATH_GPT_cost_of_lamps_and_bulbs_l1515_151519


namespace NUMINAMATH_GPT_simplest_square_root_l1515_151565

theorem simplest_square_root :
  let a := Real.sqrt (1 / 2)
  let b := Real.sqrt 11
  let c := Real.sqrt 27
  let d := Real.sqrt 0.3
  (b < a ∧ b < c ∧ b < d) :=
sorry

end NUMINAMATH_GPT_simplest_square_root_l1515_151565


namespace NUMINAMATH_GPT_travel_times_either_24_or_72_l1515_151511

variable (A B C : String)
variable (travel_time : String → String → Float)
variable (current : Float)

-- Conditions:
-- 1. Travel times are 24 minutes or 72 minutes
-- 2. Traveling from dock B cannot be balanced with current constraints
-- 3. A 3 km travel with the current is 24 minutes
-- 4. A 3 km travel against the current is 72 minutes

theorem travel_times_either_24_or_72 :
  (∀ (P Q : String), P = A ∨ P = B ∨ P = C ∧ Q = A ∨ Q = B ∨ Q = C →
  (travel_time A C = 72 ∨ travel_time C A = 24)) :=
by
  intros
  sorry

end NUMINAMATH_GPT_travel_times_either_24_or_72_l1515_151511


namespace NUMINAMATH_GPT_sin_double_angle_l1515_151502

theorem sin_double_angle
  (α : ℝ) (h1 : Real.sin (3 * Real.pi / 2 - α) = 3 / 5) (h2 : α ∈ Set.Ioo Real.pi (3 * Real.pi / 2)) :
  Real.sin (2 * α) = 24 / 25 :=
sorry

end NUMINAMATH_GPT_sin_double_angle_l1515_151502


namespace NUMINAMATH_GPT_find_smaller_root_l1515_151510

theorem find_smaller_root :
  ∀ x : ℝ, (x - 2 / 3) ^ 2 + (x - 2 / 3) * (x - 1 / 3) = 0 → x = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_smaller_root_l1515_151510


namespace NUMINAMATH_GPT_percentage_of_loss_is_25_l1515_151521

-- Definitions from conditions
def CP : ℝ := 2800
def SP : ℝ := 2100

-- Proof statement
theorem percentage_of_loss_is_25 : ((CP - SP) / CP) * 100 = 25 := by
  sorry

end NUMINAMATH_GPT_percentage_of_loss_is_25_l1515_151521


namespace NUMINAMATH_GPT_complex_expression_l1515_151580

theorem complex_expression (i : ℂ) (h : i^2 = -1) : ( (1 + i) / (1 - i) )^2006 = -1 :=
by {
  sorry
}

end NUMINAMATH_GPT_complex_expression_l1515_151580


namespace NUMINAMATH_GPT_abs_eq_neg_of_nonpos_l1515_151576

theorem abs_eq_neg_of_nonpos (a : ℝ) (h : |a| = -a) : a ≤ 0 :=
by
  have ha : |a| ≥ 0 := abs_nonneg a
  rw [h] at ha
  exact neg_nonneg.mp ha

end NUMINAMATH_GPT_abs_eq_neg_of_nonpos_l1515_151576


namespace NUMINAMATH_GPT_assistant_professor_pencils_l1515_151561

theorem assistant_professor_pencils :
  ∀ (A B P : ℕ), 
    A + B = 7 →
    2 * A + P * B = 10 →
    A + 2 * B = 11 →
    P = 1 :=
by 
  sorry

end NUMINAMATH_GPT_assistant_professor_pencils_l1515_151561


namespace NUMINAMATH_GPT_condition_1_valid_for_n_condition_2_valid_for_n_l1515_151588

-- Definitions from the conditions
def is_cube_root_of_unity (ω : ℂ) : Prop := ω^3 = 1

def roots_of_polynomial (ω : ℂ) (ω2 : ℂ) : Prop :=
  ω^2 + ω + 1 = 0 ∧ is_cube_root_of_unity ω ∧ is_cube_root_of_unity ω2

-- Problem statements
theorem condition_1_valid_for_n (n : ℕ) (ω : ℂ) (ω2 : ℂ) (h : roots_of_polynomial ω ω2) :
  (x^2 + x + 1) ∣ (x+1)^n - x^n - 1 ↔ ∃ k : ℕ, n = 6 * k + 1 ∨ n = 6 * k - 1 := sorry

theorem condition_2_valid_for_n (n : ℕ) (ω : ℂ) (ω2 : ℂ) (h : roots_of_polynomial ω ω2) :
  (x^2 + x + 1) ∣ (x+1)^n + x^n + 1 ↔ ∃ k : ℕ, n = 6 * k + 2 ∨ n = 6 * k - 2 := sorry

end NUMINAMATH_GPT_condition_1_valid_for_n_condition_2_valid_for_n_l1515_151588


namespace NUMINAMATH_GPT_product_of_tangents_is_constant_l1515_151549

theorem product_of_tangents_is_constant (a b : ℝ) (h_ab : a > b) (P : ℝ × ℝ)
  (hP_on_ellipse : P.1^2 / a^2 + P.2^2 / b^2 = 1)
  (A1 A2 : ℝ × ℝ)
  (hA1 : A1 = (-a, 0))
  (hA2 : A2 = (a, 0)) :
  ∃ (Q1 Q2 : ℝ × ℝ),
  (A1.1 - Q1.1, A2.1 - Q2.1) = (b^2, b^2) :=
sorry

end NUMINAMATH_GPT_product_of_tangents_is_constant_l1515_151549


namespace NUMINAMATH_GPT_brenda_peaches_remaining_l1515_151541

theorem brenda_peaches_remaining (total_peaches : ℕ) (percent_fresh : ℚ) (thrown_away : ℕ) (fresh_peaches : ℕ) (remaining_peaches : ℕ) :
    total_peaches = 250 → 
    percent_fresh = 0.60 → 
    thrown_away = 15 → 
    fresh_peaches = total_peaches * percent_fresh → 
    remaining_peaches = fresh_peaches - thrown_away → 
    remaining_peaches = 135 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_brenda_peaches_remaining_l1515_151541


namespace NUMINAMATH_GPT_solve_quadratic_eq_l1515_151596

theorem solve_quadratic_eq (x : ℝ) (h : x > 0) (eq : 4 * x^2 + 8 * x - 20 = 0) : 
  x = Real.sqrt 6 - 1 :=
sorry

end NUMINAMATH_GPT_solve_quadratic_eq_l1515_151596


namespace NUMINAMATH_GPT_part_a_part_b_part_c_part_d_l1515_151546

-- (a)
theorem part_a : ∃ x y : ℤ, x > 0 ∧ y > 0 ∧ x ≤ 5 ∧ x^2 - 2 * y^2 = 1 :=
by
  -- proof here
  sorry

-- (b)
theorem part_b : ∃ u v : ℤ, (3 + 2 * Real.sqrt 2)^2 = u + v * Real.sqrt 2 ∧ u^2 - 2 * v^2 = 1 :=
by
  -- proof here
  sorry

-- (c)
theorem part_c : ∀ a b c d : ℤ, a^2 - 2 * b^2 = 1 → (a + b * Real.sqrt 2) * (3 + 2 * Real.sqrt 2) = c + d * Real.sqrt 2
                  → c^2 - 2 * d^2 = 1 :=
by
  -- proof here
  sorry

-- (d)
theorem part_d : ∃ x y : ℤ, y > 100 ∧ x^2 - 2 * y^2 = 1 :=
by
  -- proof here
  sorry

end NUMINAMATH_GPT_part_a_part_b_part_c_part_d_l1515_151546


namespace NUMINAMATH_GPT_valid_integer_lattice_points_count_l1515_151567

def point := (ℤ × ℤ)
def A : point := (-4, 3)
def B : point := (4, -3)

def manhattan_distance (p1 p2 : point) : ℤ :=
  abs (p2.1 - p1.1) + abs (p2.2 - p1.2)

def valid_path_length (p1 p2 : point) : Prop :=
  manhattan_distance p1 p2 ≤ 18

def does_not_cross_y_eq_x (p1 p2 : point) : Prop :=
  ∀ x y, (x, y) ∈ [(p1, p2)] → y ≠ x

def integer_lattice_points_on_path (p1 p2 : point) : ℕ := sorry

theorem valid_integer_lattice_points_count :
  integer_lattice_points_on_path A B = 112 :=
sorry

end NUMINAMATH_GPT_valid_integer_lattice_points_count_l1515_151567


namespace NUMINAMATH_GPT_median_of_consecutive_integers_l1515_151593

def sum_of_consecutive_integers (n : ℕ) (a : ℤ) : ℤ :=
  n * (2*a + (n - 1)) / 2

theorem median_of_consecutive_integers (a : ℤ) : 
  (sum_of_consecutive_integers 25 a = 5^5) -> 
  (a + 12 = 125) := 
by
  sorry

end NUMINAMATH_GPT_median_of_consecutive_integers_l1515_151593


namespace NUMINAMATH_GPT_direction_vector_of_line_l1515_151514

noncomputable def direction_vector_of_line_eq : Prop :=
  ∃ u v, ∀ x y, (x / 4) + (y / 2) = 1 → (u, v) = (-2, 1)

theorem direction_vector_of_line :
  direction_vector_of_line_eq := sorry

end NUMINAMATH_GPT_direction_vector_of_line_l1515_151514


namespace NUMINAMATH_GPT_johns_cookies_left_l1515_151584

def dozens_to_cookies (d : ℕ) : ℕ := d * 12 -- Definition to convert dozens to actual cookie count

def cookies_left (initial_cookies : ℕ) (eaten_cookies : ℕ) : ℕ := initial_cookies - eaten_cookies -- Definition to calculate remaining cookies

theorem johns_cookies_left : cookies_left (dozens_to_cookies 2) 3 = 21 :=
by
  -- Given that John buys 2 dozen cookies
  -- And he eats 3 cookies
  -- We need to prove that he has 21 cookies left
  sorry  -- Proof is omitted as per instructions

end NUMINAMATH_GPT_johns_cookies_left_l1515_151584


namespace NUMINAMATH_GPT_go_game_prob_l1515_151591

theorem go_game_prob :
  ∀ (pA pB : ℝ),
    (pA = 0.6) →
    (pB = 0.4) →
    ((pA ^ 2) + (pB ^ 2) = 0.52) :=
by
  intros pA pB hA hB
  rw [hA, hB]
  sorry

end NUMINAMATH_GPT_go_game_prob_l1515_151591


namespace NUMINAMATH_GPT_function_translation_l1515_151547

def translateLeft (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ := λ x => f (x + a)
def translateUp (f : ℝ → ℝ) (b : ℝ) : ℝ → ℝ := λ x => (f x) + b

theorem function_translation :
  (translateUp (translateLeft (λ x => 2 * x^2) 1) 3) = λ x => 2 * (x + 1)^2 + 3 :=
by
  sorry

end NUMINAMATH_GPT_function_translation_l1515_151547


namespace NUMINAMATH_GPT_other_root_of_quadratic_l1515_151540

theorem other_root_of_quadratic (k : ℝ) :
  (∃ x : ℝ, 3 * x^2 + k * x - 5 = 0 ∧ x = 3) →
  ∃ r : ℝ, 3 * r * 3 = -5 / 3 ∧ r = -5 / 9 :=
by
  sorry

end NUMINAMATH_GPT_other_root_of_quadratic_l1515_151540


namespace NUMINAMATH_GPT_number_of_articles_l1515_151563

theorem number_of_articles (C S : ℝ) (h_gain : S = 1.4285714285714286 * C) (h_cost : ∃ X : ℝ, X * C = 35 * S) : ∃ X : ℝ, X = 50 :=
by
  -- Define the specific existence and equality proof here
  sorry

end NUMINAMATH_GPT_number_of_articles_l1515_151563


namespace NUMINAMATH_GPT_sum_of_geometric_sequence_l1515_151579

theorem sum_of_geometric_sequence :
  ∀ (a : ℕ → ℝ) (r : ℝ),
  (∃ a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℝ,
   a 1 = a_1 ∧ a 2 = a_2 ∧ a 3 = a_3 ∧ a 4 = a_4 ∧ a 5 = a_5 ∧ a 6 = a_6 ∧ a 7 = a_7 ∧ a 8 = a_8 ∧ a 9 = a_9 ∧
   a_1 * r^1 = a_2 ∧ a_1 * r^2 = a_3 ∧ a_1 * r^3 = a_4 ∧ a_1 * r^4 = a_5 ∧ a_1 * r^5 = a_6 ∧ a_1 * r^6 = a_7 ∧ a_1 * r^7 = a_8 ∧ a_1 * r^8 = a_9 ∧
   a_1 + a_2 + a_3 = 8 ∧
   a_4 + a_5 + a_6 = -4) →
  a 7 + a 8 + a 9 = 2 :=
sorry

end NUMINAMATH_GPT_sum_of_geometric_sequence_l1515_151579


namespace NUMINAMATH_GPT_inequality_solution_l1515_151555

theorem inequality_solution (a x : ℝ) : 
  (ax^2 + (2 - a) * x - 2 < 0) → 
  ((a = 0) → x < 1) ∧ 
  ((a > 0) → (-2/a < x ∧ x < 1)) ∧ 
  ((a < 0) → 
    ((-2 < a ∧ a < 0) → (x < 1 ∨ x > -2/a)) ∧
    (a = -2 → (x ≠ 1)) ∧
    (a < -2 → (x < -2/a ∨ x > 1)))
:=
sorry

end NUMINAMATH_GPT_inequality_solution_l1515_151555


namespace NUMINAMATH_GPT_complement_intersection_l1515_151554

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

theorem complement_intersection : 
  U = {1, 2, 3, 4, 5} → 
  A = {1, 2, 3} → 
  B = {2, 3, 4} → 
  (U \ (A ∩ B) = {1, 4, 5}) := 
by
  sorry

end NUMINAMATH_GPT_complement_intersection_l1515_151554


namespace NUMINAMATH_GPT_union_of_sets_l1515_151548

open Set

variable (a : ℤ)

def setA : Set ℤ := {1, 3}
def setB (a : ℤ) : Set ℤ := {a + 2, 5}

theorem union_of_sets (h : {3} = setA ∩ setB a) : setA ∪ setB a = {1, 3, 5} :=
by
  sorry

end NUMINAMATH_GPT_union_of_sets_l1515_151548


namespace NUMINAMATH_GPT_root_relationship_l1515_151573

theorem root_relationship (m n a b : ℝ) 
  (h_eq : ∀ x, 3 - (x - m) * (x - n) = 0 ↔ x = a ∨ x = b) : a < m ∧ m < n ∧ n < b :=
by
  sorry

end NUMINAMATH_GPT_root_relationship_l1515_151573


namespace NUMINAMATH_GPT_consecutive_integer_sum_l1515_151522

theorem consecutive_integer_sum (n : ℕ) (h1 : n * (n + 1) = 2720) : n + (n + 1) = 103 :=
sorry

end NUMINAMATH_GPT_consecutive_integer_sum_l1515_151522


namespace NUMINAMATH_GPT_number_of_glass_bottles_l1515_151538

theorem number_of_glass_bottles (total_litter : ℕ) (aluminum_cans : ℕ) (glass_bottles : ℕ) : 
  total_litter = 18 → aluminum_cans = 8 → glass_bottles = total_litter - aluminum_cans → glass_bottles = 10 :=
by
  intros h_total h_aluminum h_glass
  rw [h_total, h_aluminum] at h_glass
  exact h_glass.trans rfl


end NUMINAMATH_GPT_number_of_glass_bottles_l1515_151538


namespace NUMINAMATH_GPT_minimum_value_squared_sum_minimum_value_squared_sum_equality_l1515_151592

theorem minimum_value_squared_sum (a b c t : ℝ) (h : a + b + c = t) : 
  a^2 + b^2 + c^2 ≥ t^2 / 3 := by
  sorry

theorem minimum_value_squared_sum_equality (a b c t : ℝ) (h : a + b + c = t) 
  (ha : a = t / 3) (hb : b = t / 3) (hc : c = t / 3) : 
  a^2 + b^2 + c^2 = t^2 / 3 := by
  sorry

end NUMINAMATH_GPT_minimum_value_squared_sum_minimum_value_squared_sum_equality_l1515_151592


namespace NUMINAMATH_GPT_question1_perpendicular_question2_parallel_l1515_151562

structure Vector2D :=
  (x : ℝ)
  (y : ℝ)

def dot_product (v1 v2 : Vector2D) : ℝ :=
  v1.x * v2.x + v1.y * v2.y

noncomputable def vector_k_a_plus_2_b (k : ℝ) (a b : Vector2D) : Vector2D :=
  ⟨k * a.x + 2 * b.x, k * a.y + 2 * b.y⟩

noncomputable def vector_2_a_minus_4_b (a b : Vector2D) : Vector2D :=
  ⟨2 * a.x - 4 * b.x, 2 * a.y - 4 * b.y⟩

def perpendicular (v1 v2 : Vector2D) : Prop :=
  dot_product v1 v2 = 0

def parallel (v1 v2 : Vector2D) : Prop :=
  v1.x * v2.y = v1.y * v2.x

def opposite_direction (v1 v2 : Vector2D) : Prop :=
  parallel v1 v2 ∧ v1.x * v2.x + v1.y * v2.y < 0

noncomputable def vector_a : Vector2D := ⟨1, 1⟩
noncomputable def vector_b : Vector2D := ⟨2, 3⟩

theorem question1_perpendicular (k : ℝ) : 
  perpendicular (vector_k_a_plus_2_b k vector_a vector_b) (vector_2_a_minus_4_b vector_a vector_b) ↔ 
  k = -21 / 4 :=
sorry

theorem question2_parallel (k : ℝ) :
  (parallel (vector_k_a_plus_2_b k vector_a vector_b) (vector_2_a_minus_4_b vector_a vector_b) ∧
  opposite_direction (vector_k_a_plus_2_b k vector_a vector_b) (vector_2_a_minus_4_b vector_a vector_b)) ↔ 
  k = -1 / 2 :=
sorry

end NUMINAMATH_GPT_question1_perpendicular_question2_parallel_l1515_151562


namespace NUMINAMATH_GPT_combined_jail_time_in_weeks_l1515_151500

-- Definitions based on conditions
def days_of_protest : ℕ := 30
def number_of_cities : ℕ := 21
def daily_arrests_per_city : ℕ := 10
def days_in_jail_pre_trial : ℕ := 4
def sentence_weeks : ℕ := 2
def jail_fraction_of_sentence : ℕ := 1 / 2

-- Calculate the combined weeks of jail time
theorem combined_jail_time_in_weeks : 
  (days_of_protest * daily_arrests_per_city * number_of_cities) * 
  (days_in_jail_pre_trial + (sentence_weeks * 7 * jail_fraction_of_sentence)) / 
  7 = 9900 := 
by sorry

end NUMINAMATH_GPT_combined_jail_time_in_weeks_l1515_151500


namespace NUMINAMATH_GPT_unit_digit_4137_pow_754_l1515_151523

theorem unit_digit_4137_pow_754 : (4137 ^ 754) % 10 = 9 := by
  sorry

end NUMINAMATH_GPT_unit_digit_4137_pow_754_l1515_151523


namespace NUMINAMATH_GPT_tournament_total_games_l1515_151572

def total_number_of_games (num_teams : ℕ) (group_size : ℕ) (num_groups : ℕ) (teams_for_knockout : ℕ) : ℕ :=
  let games_per_group := (group_size * (group_size - 1)) / 2
  let group_stage_games := num_groups * games_per_group
  let knockout_teams := num_groups * teams_for_knockout
  let knockout_games := knockout_teams - 1
  group_stage_games + knockout_games

theorem tournament_total_games : total_number_of_games 32 4 8 2 = 63 := by
  sorry

end NUMINAMATH_GPT_tournament_total_games_l1515_151572


namespace NUMINAMATH_GPT_expression_equals_8_l1515_151513

-- Define the expression we are interested in.
def expression : ℚ :=
  (1 + 1 / 2) * (1 + 1 / 3) * (1 + 1 / 4) * (1 + 1 / 5) * (1 + 1 / 6) * (1 + 1 / 7)

-- Statement we need to prove
theorem expression_equals_8 : expression = 8 := by
  sorry

end NUMINAMATH_GPT_expression_equals_8_l1515_151513


namespace NUMINAMATH_GPT_lolita_milk_per_week_l1515_151528

def weekday_milk : ℕ := 3
def saturday_milk : ℕ := 2 * weekday_milk
def sunday_milk : ℕ := 3 * weekday_milk
def total_milk_week : ℕ := 5 * weekday_milk + saturday_milk + sunday_milk

theorem lolita_milk_per_week : total_milk_week = 30 := 
by 
  sorry

end NUMINAMATH_GPT_lolita_milk_per_week_l1515_151528


namespace NUMINAMATH_GPT_isosceles_triangle_base_length_l1515_151585

theorem isosceles_triangle_base_length
  (a b : ℝ) (h₁ : a = 4) (h₂ : b = 8) (h₃ : a ≠ b)
  (triangle_inequality : ∀ x y z : ℝ, x + y > z) :
  ∃ base : ℝ, base = 8 := by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_base_length_l1515_151585


namespace NUMINAMATH_GPT_domain_of_function_l1515_151571

theorem domain_of_function:
  {x : ℝ | x^2 - 5*x + 6 > 0 ∧ x ≠ 3} = {x : ℝ | x < 2 ∨ x > 3} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_function_l1515_151571


namespace NUMINAMATH_GPT_molecular_weight_CaCO3_is_100_09_l1515_151569

-- Declare the atomic weights
def atomic_weight_Ca : ℝ := 40.08
def atomic_weight_C : ℝ := 12.01
def atomic_weight_O : ℝ := 16.00

-- Define the molecular weight constant for calcium carbonate
def molecular_weight_CaCO3 : ℝ :=
  (1 * atomic_weight_Ca) + (1 * atomic_weight_C) + (3 * atomic_weight_O)

-- Prove that the molecular weight of calcium carbonate is 100.09 g/mol
theorem molecular_weight_CaCO3_is_100_09 :
  molecular_weight_CaCO3 = 100.09 :=
by
  -- Proof goes here, placeholder for now
  sorry

end NUMINAMATH_GPT_molecular_weight_CaCO3_is_100_09_l1515_151569


namespace NUMINAMATH_GPT_place_mat_length_l1515_151533

theorem place_mat_length (r : ℝ) (n : ℕ) (w : ℝ) (x : ℝ)
  (table_is_round : r = 3)
  (number_of_mats : n = 8)
  (mat_width : w = 1)
  (mat_length : ∀ (k: ℕ), 0 ≤ k ∧ k < n → (2 * r * Real.sin (Real.pi / n) = x)) :
  x = (3 * Real.sqrt 35) / 10 + 1 / 2 :=
sorry

end NUMINAMATH_GPT_place_mat_length_l1515_151533


namespace NUMINAMATH_GPT_distance_between_neg2_and_3_l1515_151589
-- Import the necessary Lean libraries

-- State the theorem to prove the distance between -2 and 3 is 5
theorem distance_between_neg2_and_3 : abs (3 - (-2)) = 5 := by
  sorry

end NUMINAMATH_GPT_distance_between_neg2_and_3_l1515_151589


namespace NUMINAMATH_GPT_tan_neg405_deg_l1515_151543

theorem tan_neg405_deg : Real.tan (-405 * Real.pi / 180) = -1 := by
  -- This is a placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_tan_neg405_deg_l1515_151543


namespace NUMINAMATH_GPT_james_hours_worked_l1515_151599

variable (x : ℝ) (y : ℝ)

theorem james_hours_worked (h1: 18 * x + 16 * (1.5 * x) = 40 * x + (y - 40) * (2 * x)) : y = 41 :=
by
  sorry

end NUMINAMATH_GPT_james_hours_worked_l1515_151599


namespace NUMINAMATH_GPT_num_solution_pairs_l1515_151577

theorem num_solution_pairs (x y : ℕ) (hx : x > 0) (hy : y > 0) :
  4 * x + 7 * y = 600 → ∃ n : ℕ, n = 21 :=
by
  sorry

end NUMINAMATH_GPT_num_solution_pairs_l1515_151577


namespace NUMINAMATH_GPT_negation_of_p_l1515_151520

namespace ProofProblem

variable (x : ℝ)

def p : Prop := ∃ x : ℝ, x^2 + x - 1 ≥ 0

def neg_p : Prop := ∀ x : ℝ, x^2 + x - 1 < 0

theorem negation_of_p : ¬p = neg_p := sorry

end ProofProblem

end NUMINAMATH_GPT_negation_of_p_l1515_151520


namespace NUMINAMATH_GPT_find_x_converges_to_l1515_151570

noncomputable def series_sum (x : ℝ) : ℝ := ∑' n : ℕ, (4 * (n + 1) - 2) * x^n

theorem find_x_converges_to (x : ℝ) (h : |x| < 1) :
  series_sum x = 60 → x = 29 / 30 :=
by
  sorry

end NUMINAMATH_GPT_find_x_converges_to_l1515_151570


namespace NUMINAMATH_GPT_isosceles_triangle_problem_l1515_151586

theorem isosceles_triangle_problem
  (BT CT : Real) (BC : Real) (BZ CZ TZ : Real) :
  BT = 20 →
  CT = 20 →
  BC = 24 →
  TZ^2 + 2 * BZ * CZ = 478 →
  BZ = CZ →
  BZ * CZ = 144 :=
by 
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_isosceles_triangle_problem_l1515_151586


namespace NUMINAMATH_GPT_quadratic_sum_l1515_151544

theorem quadratic_sum (x : ℝ) (h : x^2 = 16*x - 9) : x = 8 ∨ x = 9 := sorry

end NUMINAMATH_GPT_quadratic_sum_l1515_151544


namespace NUMINAMATH_GPT_absolute_difference_rectangle_l1515_151524

theorem absolute_difference_rectangle 
  (x y r k : ℝ)
  (h1 : 2 * x + 2 * y = 4 * r)
  (h2 : (x^2 + y^2) = (k * x)^2) :
  |x - y| = k * x :=
by
  sorry

end NUMINAMATH_GPT_absolute_difference_rectangle_l1515_151524


namespace NUMINAMATH_GPT_range_of_a_if_exists_x_l1515_151556

variable {a x : ℝ}

theorem range_of_a_if_exists_x :
  (∃ x : ℝ, -1 < x ∧ x < 1 ∧ (a * x^2 - 1 ≥ 0)) → (a > 1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_if_exists_x_l1515_151556


namespace NUMINAMATH_GPT_point_reflection_x_axis_l1515_151564

-- Definition of the original point P
def P : ℝ × ℝ := (-2, 5)

-- Function to reflect a point across the x-axis
def reflect_x_axis (point : ℝ × ℝ) : ℝ × ℝ :=
  (point.1, -point.2)

-- Our theorem
theorem point_reflection_x_axis :
  reflect_x_axis P = (-2, -5) := by
  sorry

end NUMINAMATH_GPT_point_reflection_x_axis_l1515_151564


namespace NUMINAMATH_GPT_geom_seq_thm_l1515_151536

noncomputable def geom_seq (a : ℕ → ℝ) :=
  a 1 = 2 ∧ (a 2 * a 4 = a 6)

noncomputable def b_seq (a : ℕ → ℝ) (n : ℕ) :=
  1 / (Real.logb 2 (a (2 * n - 1)) * Real.logb 2 (a (2 * n + 1)))

noncomputable def sn_sum (b : ℕ → ℝ) (n : ℕ) :=
  (Finset.range (n + 1)).sum b

theorem geom_seq_thm (a : ℕ → ℝ) (n : ℕ) (b : ℕ → ℝ) :
  geom_seq a →
  ∀ n, a n = 2 ^ n ∧ sn_sum (b_seq a) n = n / (2 * n + 1) :=
by
  sorry

end NUMINAMATH_GPT_geom_seq_thm_l1515_151536


namespace NUMINAMATH_GPT_LovelyCakeSlices_l1515_151527

/-- Lovely cuts her birthday cake into some equal pieces.
    One-fourth of the cake was eaten by her visitors.
    Nine slices of cake were kept, representing three-fourths of the total number of slices.
    Prove: Lovely cut her birthday cake into 12 equal pieces. -/
theorem LovelyCakeSlices (totalSlices : ℕ) 
  (h1 : (3 / 4 : ℚ) * totalSlices = 9) : totalSlices = 12 := by
  sorry

end NUMINAMATH_GPT_LovelyCakeSlices_l1515_151527


namespace NUMINAMATH_GPT_inequality_solution_l1515_151590

theorem inequality_solution (x : ℝ) : 1 - (2 * x - 2) / 5 < (3 - 4 * x) / 2 → x < 1 / 16 := by
  sorry

end NUMINAMATH_GPT_inequality_solution_l1515_151590


namespace NUMINAMATH_GPT_third_restaurant_meals_per_day_l1515_151557

-- Define the daily meals served by the first two restaurants
def meals_first_restaurant_per_day : ℕ := 20
def meals_second_restaurant_per_day : ℕ := 40

-- Define the total meals served by all three restaurants per week
def total_meals_per_week : ℕ := 770

-- Define the weekly meals served by the first two restaurants
def meals_first_restaurant_per_week : ℕ := meals_first_restaurant_per_day * 7
def meals_second_restaurant_per_week : ℕ := meals_second_restaurant_per_day * 7

-- Total weekly meals served by the first two restaurants
def total_meals_first_two_restaurants_per_week : ℕ := meals_first_restaurant_per_week + meals_second_restaurant_per_week

-- Weekly meals served by the third restaurant
def meals_third_restaurant_per_week : ℕ := total_meals_per_week - total_meals_first_two_restaurants_per_week

-- Convert weekly meals served by the third restaurant to daily meals
def meals_third_restaurant_per_day : ℕ := meals_third_restaurant_per_week / 7

-- Goal: Prove the third restaurant serves 50 meals per day
theorem third_restaurant_meals_per_day : meals_third_restaurant_per_day = 50 := by
  -- proof skipped
  sorry

end NUMINAMATH_GPT_third_restaurant_meals_per_day_l1515_151557


namespace NUMINAMATH_GPT_number_of_sixth_graders_who_bought_more_pens_than_seventh_graders_l1515_151559

theorem number_of_sixth_graders_who_bought_more_pens_than_seventh_graders 
  (p : ℕ) (h1 : 178 % p = 0) (h2 : 252 % p = 0) :
  (252 / p) - (178 / p) = 5 :=
sorry

end NUMINAMATH_GPT_number_of_sixth_graders_who_bought_more_pens_than_seventh_graders_l1515_151559


namespace NUMINAMATH_GPT_pet_shop_dogs_l1515_151525

theorem pet_shop_dogs (D C B : ℕ) (x : ℕ) (h1 : D = 3 * x) (h2 : C = 5 * x) (h3 : B = 9 * x) (h4 : D + B = 204) : D = 51 := by
  -- omitted proof
  sorry

end NUMINAMATH_GPT_pet_shop_dogs_l1515_151525


namespace NUMINAMATH_GPT_geometric_sequence_sum_ratio_l1515_151583

noncomputable def geometric_sum (a q : ℝ) (n : ℕ) : ℝ := a * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_ratio (a q : ℝ) (h : a * q^2 = 8 * a * q^5) :
  (geometric_sum a q 4) / (geometric_sum a q 2) = 5 / 4 :=
by
  -- The proof will go here.
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_ratio_l1515_151583


namespace NUMINAMATH_GPT_c_put_15_oxen_l1515_151515

theorem c_put_15_oxen (x : ℕ):
  (10 * 7 + 12 * 5 + 3 * x = 130 + 3 * x) →
  (175 * 3 * x / (130 + 3 * x) = 45) →
  x = 15 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_c_put_15_oxen_l1515_151515


namespace NUMINAMATH_GPT_belts_count_l1515_151578

-- Definitions based on conditions
variable (shoes belts hats : ℕ)

-- Conditions from the problem
axiom shoes_eq_14 : shoes = 14
axiom hat_count : hats = 5
axiom shoes_double_of_belts : shoes = 2 * belts

-- Definition of the theorem to prove the number of belts
theorem belts_count : belts = 7 :=
by
  sorry

end NUMINAMATH_GPT_belts_count_l1515_151578


namespace NUMINAMATH_GPT_circle_radius_l1515_151551

-- Define the main geometric scenario in Lean 4
theorem circle_radius 
  (O P A B : Type) 
  (r OP PA PB : ℝ)
  (h1 : PA * PB = 24)
  (h2 : OP = 5)
  : r = 7 
:= sorry

end NUMINAMATH_GPT_circle_radius_l1515_151551


namespace NUMINAMATH_GPT_find_b_l1515_151594

theorem find_b
  (a b c : ℚ)
  (h1 : (4 : ℚ) * a = 12)
  (h2 : (4 * (4 * b) = - (14:ℚ) + 3 * a)) :
  b = -(7:ℚ) / 2 :=
by sorry

end NUMINAMATH_GPT_find_b_l1515_151594


namespace NUMINAMATH_GPT_solution_of_inequality_answer_A_incorrect_answer_B_incorrect_answer_C_incorrect_D_is_correct_l1515_151560

theorem solution_of_inequality (a b x : ℝ) :
    (b - a * x > 0) ↔
    (a > 0 ∧ x < b / a ∨ 
     a < 0 ∧ x > b / a ∨ 
     a = 0 ∧ false) :=
by sorry

-- Additional theorems to rule out incorrect answers
theorem answer_A_incorrect (a b : ℝ) :
    (∀ x : ℝ, b - a * x > 0 → x > |b| / |a|) → false :=
by sorry

theorem answer_B_incorrect (a b : ℝ) :
    (∀ x : ℝ, b - a * x > 0 → x < |b| / |a|) → false :=
by sorry

theorem answer_C_incorrect (a b : ℝ) :
    (∀ x : ℝ, b - a * x > 0 → x > -|b| / |a|) → false :=
by sorry

theorem D_is_correct (a b : ℝ) :
    (∀ x : ℝ, b - a * x > 0 → x > |b| / |a| ∨ x < |b| / |a| ∨ x > -|b| / |a|) → false :=
by sorry

end NUMINAMATH_GPT_solution_of_inequality_answer_A_incorrect_answer_B_incorrect_answer_C_incorrect_D_is_correct_l1515_151560


namespace NUMINAMATH_GPT_simplify_expression_l1515_151508

theorem simplify_expression (a1 a2 a3 a4 : ℝ) (h1 : 1 - a1 ≠ 0) (h2 : 1 - a2 ≠ 0) (h3 : 1 - a3 ≠ 0) (h4 : 1 - a4 ≠ 0) :
  1 + a1 / (1 - a1) + a2 / ((1 - a1) * (1 - a2)) + a3 / ((1 - a1) * (1 - a2) * (1 - a3)) + 
  (a4 - a1) / ((1 - a1) * (1 - a2) * (1 - a3) * (1 - a4)) = 
  1 / ((1 - a2) * (1 - a3) * (1 - a4)) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1515_151508


namespace NUMINAMATH_GPT_bottles_produced_by_twenty_machines_l1515_151507

-- Definitions corresponding to conditions
def bottles_per_machine_per_minute (total_machines : ℕ) (total_bottles : ℕ) : ℕ :=
  total_bottles / total_machines

def bottles_produced (machines : ℕ) (rate : ℕ) (time : ℕ) : ℕ :=
  machines * rate * time

-- Given conditions
axiom six_machines_rate : ∀ (machines total_bottles : ℕ), machines = 6 → total_bottles = 270 →
  bottles_per_machine_per_minute machines total_bottles = 45

-- Prove the question == answer given conditions
theorem bottles_produced_by_twenty_machines :
  bottles_produced 20 45 4 = 3600 :=
by sorry

end NUMINAMATH_GPT_bottles_produced_by_twenty_machines_l1515_151507


namespace NUMINAMATH_GPT_total_students_count_l1515_151535

-- Define the conditions
def num_rows : ℕ := 8
def students_per_row : ℕ := 6
def students_last_row : ℕ := 5
def rows_with_six_students : ℕ := 7

-- Define the total students
def total_students : ℕ :=
  (rows_with_six_students * students_per_row) + students_last_row

-- The theorem to prove
theorem total_students_count : total_students = 47 := by
  sorry

end NUMINAMATH_GPT_total_students_count_l1515_151535


namespace NUMINAMATH_GPT_monotonicity_range_of_a_l1515_151516

noncomputable def f (x a : ℝ) : ℝ := Real.log x + a * (1 - x)
noncomputable def f' (x a : ℝ) : ℝ := 1 / x - a

-- 1. Monotonicity discussion
theorem monotonicity (a x : ℝ) (h : 0 < x) : 
  (a ≤ 0 → ∀ x, 0 < x → f' x a > 0) ∧
  (a > 0 → (∀ x, 0 < x ∧ x < 1 / a → f' x a > 0) ∧ (∀ x, x > 1 / a → f' x a < 0)) :=
sorry

-- 2. Range of a for maximum value condition
noncomputable def g (a : ℝ) : ℝ := Real.log a + a - 1

theorem range_of_a (a : ℝ) : 
  (0 < a) ∧ (a < 1) ↔ g a < 0 :=
sorry

end NUMINAMATH_GPT_monotonicity_range_of_a_l1515_151516


namespace NUMINAMATH_GPT_fixed_point_min_value_l1515_151568

theorem fixed_point_min_value {a m n : ℝ} (ha_pos : 0 < a) (ha_ne_one : a ≠ 1) (hm_pos : 0 < m) (hn_pos : 0 < n)
  (h : 3 * m + n = 1) : (1 / m + 3 / n) = 12 := sorry

end NUMINAMATH_GPT_fixed_point_min_value_l1515_151568


namespace NUMINAMATH_GPT_ratio_of_ages_l1515_151539

variable (D R : ℕ)

theorem ratio_of_ages : (D = 9) → (R + 6 = 18) → (R / D = 4 / 3) :=
by
  intros hD hR
  -- proof goes here
  sorry

end NUMINAMATH_GPT_ratio_of_ages_l1515_151539


namespace NUMINAMATH_GPT_largest_common_term_arith_seq_l1515_151512

theorem largest_common_term_arith_seq :
  ∃ a, a < 90 ∧ (∃ n : ℤ, a = 3 + 8 * n) ∧ (∃ m : ℤ, a = 5 + 9 * m) ∧ a = 59 :=
by
  sorry

end NUMINAMATH_GPT_largest_common_term_arith_seq_l1515_151512


namespace NUMINAMATH_GPT_sum_of_terms_l1515_151517

noncomputable def u1 := 8
noncomputable def r := 2

def first_geometric (u2 u3 : ℝ) (u1 r : ℝ) : Prop := 
  u2 = r * u1 ∧ u3 = r^2 * u1

def last_arithmetic (u2 u3 u4 : ℝ) : Prop := 
  u3 - u2 = u4 - u3

def terms (u1 u2 u3 u4 : ℝ) (r : ℝ) : Prop :=
  first_geometric u2 u3 u1 r ∧
  last_arithmetic u2 u3 u4 ∧
  u4 = u1 + 40

theorem sum_of_terms (u1 u2 u3 u4 : ℝ)
  (h : terms u1 u2 u3 u4 r) : u1 + u2 + u3 + u4 = 104 :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_terms_l1515_151517


namespace NUMINAMATH_GPT_cab_base_price_l1515_151598

theorem cab_base_price (base_price : ℝ) (total_cost : ℝ) (cost_per_mile : ℝ) (distance : ℝ) 
  (H1 : total_cost = 23) 
  (H2 : cost_per_mile = 4) 
  (H3 : distance = 5) 
  (H4 : base_price = total_cost - cost_per_mile * distance) : 
  base_price = 3 :=
by 
  sorry

end NUMINAMATH_GPT_cab_base_price_l1515_151598


namespace NUMINAMATH_GPT_min_value_on_top_layer_l1515_151566

-- Definitions reflecting conditions
def bottom_layer : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def block_value (layer : List ℕ) (i : ℕ) : ℕ :=
  layer.getD (i-1) 0 -- assuming 1-based indexing

def second_layer_values : List ℕ :=
  [block_value bottom_layer 1 + block_value bottom_layer 2 + block_value bottom_layer 3,
   block_value bottom_layer 2 + block_value bottom_layer 3 + block_value bottom_layer 4,
   block_value bottom_layer 4 + block_value bottom_layer 5 + block_value bottom_layer 6,
   block_value bottom_layer 5 + block_value bottom_layer 6 + block_value bottom_layer 7,
   block_value bottom_layer 7 + block_value bottom_layer 8 + block_value bottom_layer 9,
   block_value bottom_layer 8 + block_value bottom_layer 9 + block_value bottom_layer 10]

def third_layer_values : List ℕ :=
  [second_layer_values.getD 0 0 + second_layer_values.getD 1 0 + second_layer_values.getD 2 0,
   second_layer_values.getD 1 0 + second_layer_values.getD 2 0 + second_layer_values.getD 3 0,
   second_layer_values.getD 3 0 + second_layer_values.getD 4 0 + second_layer_values.getD 5 0]

def top_layer_value : ℕ :=
  third_layer_values.getD 0 0 + third_layer_values.getD 1 0 + third_layer_values.getD 2 0

theorem min_value_on_top_layer : top_layer_value = 114 :=
by
  have h0 := block_value bottom_layer 1 -- intentionally leaving this incomplete as we're skipping the actual proof
  sorry

end NUMINAMATH_GPT_min_value_on_top_layer_l1515_151566


namespace NUMINAMATH_GPT_run_time_difference_l1515_151558

variables (distance duration_injured : ℝ) (initial_speed : ℝ)

theorem run_time_difference (H1 : distance = 20) 
                            (H2 : duration_injured = 22) 
                            (H3 : initial_speed = distance * 2 / duration_injured) :
                            duration_injured - (distance / initial_speed) = 11 :=
by
  sorry

end NUMINAMATH_GPT_run_time_difference_l1515_151558


namespace NUMINAMATH_GPT_sequence_converges_l1515_151505

open Real

theorem sequence_converges (x : ℕ → ℝ) (h₀ : ∀ n, x (n + 1) = 1 + x n - 0.5 * (x n) ^ 2) (h₁ : 1 < x 1 ∧ x 1 < 2) :
  ∀ n ≥ 3, |x n - sqrt 2| < 2 ^ (-n : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_sequence_converges_l1515_151505


namespace NUMINAMATH_GPT_spending_percentage_A_l1515_151542

def combined_salary (S_A S_B : ℝ) : Prop := S_A + S_B = 7000
def A_salary (S_A : ℝ) : Prop := S_A = 5250
def B_salary (S_B : ℝ) : Prop := S_B = 1750
def B_spending (P_B : ℝ) : Prop := P_B = 0.85
def same_savings (S_A S_B P_A P_B : ℝ) : Prop := S_A * (1 - P_A) = S_B * (1 - P_B)
def A_spending (P_A : ℝ) : Prop := P_A = 0.95

theorem spending_percentage_A (S_A S_B P_A P_B : ℝ) 
  (h1: combined_salary S_A S_B) 
  (h2: A_salary S_A) 
  (h3: B_salary S_B) 
  (h4: B_spending P_B) 
  (h5: same_savings S_A S_B P_A P_B) : A_spending P_A :=
sorry

end NUMINAMATH_GPT_spending_percentage_A_l1515_151542


namespace NUMINAMATH_GPT_triangle_side_length_sum_l1515_151530

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def distance_squared (p1 p2 : Point3D) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2

structure Triangle where
  D : Point3D
  E : Point3D
  F : Point3D

noncomputable def centroid (t : Triangle) : Point3D :=
  let D := t.D
  let E := t.E
  let F := t.F
  { x := (D.x + E.x + F.x) / 3,
    y := (D.y + E.y + F.y) / 3,
    z := (D.z + E.z + F.z) / 3 }

noncomputable def sum_of_squares_centroid_distances (t : Triangle) : ℝ :=
  let G := centroid t
  distance_squared G t.D + distance_squared G t.E + distance_squared G t.F

noncomputable def sum_of_squares_side_lengths (t : Triangle) : ℝ :=
  distance_squared t.D t.E + distance_squared t.D t.F + distance_squared t.E t.F

theorem triangle_side_length_sum (t : Triangle) (h : sum_of_squares_centroid_distances t = 72) :
  sum_of_squares_side_lengths t = 216 :=
sorry

end NUMINAMATH_GPT_triangle_side_length_sum_l1515_151530


namespace NUMINAMATH_GPT_cos_alpha_beta_value_l1515_151532

theorem cos_alpha_beta_value
  (α β : ℝ)
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : -π / 2 < β ∧ β < 0)
  (h3 : Real.cos (π / 4 + α) = 1 / 3)
  (h4 : Real.cos (π / 4 - β) = Real.sqrt 3 / 3) :
  Real.cos (α + β) = (5 * Real.sqrt 3) / 9 := 
by
  sorry

end NUMINAMATH_GPT_cos_alpha_beta_value_l1515_151532


namespace NUMINAMATH_GPT_michael_current_chickens_l1515_151509

-- Defining variables and constants
variable (initial_chickens final_chickens annual_increase : ℕ)

-- Given conditions
def chicken_increase_condition : Prop :=
  final_chickens = initial_chickens + annual_increase * 9

-- Question to answer
def current_chickens (final_chickens annual_increase : ℕ) : ℕ :=
  final_chickens - annual_increase * 9

-- Proof problem
theorem michael_current_chickens
  (initial_chickens : ℕ)
  (final_chickens : ℕ)
  (annual_increase : ℕ)
  (h1 : chicken_increase_condition final_chickens initial_chickens annual_increase) :
  initial_chickens = 550 :=
by
  -- Formal proof would go here.
  sorry

end NUMINAMATH_GPT_michael_current_chickens_l1515_151509


namespace NUMINAMATH_GPT_sequence_form_l1515_151545

theorem sequence_form (c : ℕ) (a : ℕ → ℕ) :
  (∀ n : ℕ, 0 < n →
    (∃! i : ℕ, 0 < i ∧ a i ≤ a (n + 1) + c)) ↔
  (∀ n : ℕ, 0 < n → a n = n + (c + 1)) :=
by
  sorry

end NUMINAMATH_GPT_sequence_form_l1515_151545


namespace NUMINAMATH_GPT_sum_S5_l1515_151529

-- Geometric sequence definitions and conditions
noncomputable def geometric_sequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r^n

noncomputable def sum_of_geometric_sequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

variables (a r : ℝ)

-- Given conditions translated into Lean:
-- a2 * a3 = 2 * a1
def condition1 := (geometric_sequence a r 1) * (geometric_sequence a r 2) = 2 * a

-- Arithmetic mean of a4 and 2 * a7 is 5/4
def condition2 := (geometric_sequence a r 3 + 2 * geometric_sequence a r 6) / 2 = 5 / 4

-- The final goal proving that S5 = 31
theorem sum_S5 (h1 : condition1 a r) (h2 : condition2 a r) : sum_of_geometric_sequence a r 5 = 31 := by
  apply sorry

end NUMINAMATH_GPT_sum_S5_l1515_151529


namespace NUMINAMATH_GPT_line_passes_through_fixed_point_minimum_area_triangle_l1515_151595

theorem line_passes_through_fixed_point (k : ℝ) :
  ∃ P : ℝ × ℝ, P = (2, 1) ∧ ∀ k : ℝ, (k * 2 - 1 + 1 - 2 * k = 0) :=
sorry

theorem minimum_area_triangle (k : ℝ) :
  ∀ k: ℝ, k < 0 → 1/2 * (2 - 1/k) * (1 - 2*k) ≥ 4 ∧ 
           (1/2 * (2 - 1/k) * (1 - 2*k) = 4 ↔ k = -1/2) :=
sorry

end NUMINAMATH_GPT_line_passes_through_fixed_point_minimum_area_triangle_l1515_151595


namespace NUMINAMATH_GPT_simplify_expression_l1515_151574

theorem simplify_expression (r : ℝ) (h1 : r^2 ≠ 0) (h2 : r^4 > 16) :
  ( ( ( (r^2 + 4) ^ (3 : ℝ) ) ^ (1 / 3 : ℝ) * (1 + 4 / r^2) ^ (1 / 2 : ℝ) ^ (1 / 3 : ℝ)
    - ( (r^2 - 4) ^ (3 : ℝ) ) ^ (1 / 3 : ℝ) * (1 - 4 / r^2) ^ (1 / 2 : ℝ) ^ (1 / 3 : ℝ) ) ^ 2 )
  / ( r^2 - (r^4 - 16) ^ (1 / 2 : ℝ) )
  = 2 * r ^ (-(2 / 3 : ℝ)) := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1515_151574


namespace NUMINAMATH_GPT_sum_of_cubes_eq_twice_product_of_roots_l1515_151531

theorem sum_of_cubes_eq_twice_product_of_roots (m : ℝ) :
  (∃ a b : ℝ, (3*a^2 + 6*a + m = 0) ∧ (3*b^2 + 6*b + m = 0) ∧ (a ≠ b)) → 
  (a^3 + b^3 = 2 * a * b) → 
  m = 6 :=
by
  intros h_exists sum_eq_twice_product
  sorry

end NUMINAMATH_GPT_sum_of_cubes_eq_twice_product_of_roots_l1515_151531
