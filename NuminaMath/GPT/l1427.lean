import Mathlib

namespace NUMINAMATH_GPT_goals_even_more_likely_l1427_142721

theorem goals_even_more_likely (p_1 : ℝ) (q_1 : ℝ) (h1 : p_1 + q_1 = 1) :
  let p := p_1^2 + q_1^2 
  let q := 2 * p_1 * q_1
  p ≥ q := by
    sorry

end NUMINAMATH_GPT_goals_even_more_likely_l1427_142721


namespace NUMINAMATH_GPT_congruence_from_overlap_l1427_142722

-- Definitions used in the conditions
def figure := Type
def equal_area (f1 f2 : figure) : Prop := sorry
def equal_perimeter (f1 f2 : figure) : Prop := sorry
def equilateral_triangle (f : figure) : Prop := sorry
def can_completely_overlap (f1 f2 : figure) : Prop := sorry

-- Theorem that should be proven
theorem congruence_from_overlap (f1 f2 : figure) (h: can_completely_overlap f1 f2) : f1 = f2 := sorry

end NUMINAMATH_GPT_congruence_from_overlap_l1427_142722


namespace NUMINAMATH_GPT_original_acid_percentage_l1427_142716

variables (a w : ℝ)

-- Conditions from the problem
def cond1 : Prop := a / (a + w + 2) = 0.18
def cond2 : Prop := (a + 2) / (a + w + 4) = 0.36

-- The Lean statement to prove
theorem original_acid_percentage (hc1 : cond1 a w) (hc2 : cond2 a w) : (a / (a + w)) * 100 = 19 :=
sorry

end NUMINAMATH_GPT_original_acid_percentage_l1427_142716


namespace NUMINAMATH_GPT_matt_total_points_l1427_142718

variable (n2_successful_shots : Nat) (n3_successful_shots : Nat)

def total_points (n2 : Nat) (n3 : Nat) : Nat :=
  2 * n2 + 3 * n3

theorem matt_total_points :
  total_points 4 2 = 14 :=
by
  sorry

end NUMINAMATH_GPT_matt_total_points_l1427_142718


namespace NUMINAMATH_GPT_min_students_wearing_both_l1427_142757

theorem min_students_wearing_both (n : ℕ) (H1 : n % 3 = 0) (H2 : n % 6 = 0) (H3 : n = 6) :
  ∃ x : ℕ, x = 1 ∧ 
           (∃ b : ℕ, b = n / 3) ∧
           (∃ r : ℕ, r = 5 * n / 6) ∧
           6 = b + r - x :=
by sorry

end NUMINAMATH_GPT_min_students_wearing_both_l1427_142757


namespace NUMINAMATH_GPT_cows_in_herd_l1427_142741

theorem cows_in_herd (n : ℕ) (h1 : n / 3 + n / 6 + n / 7 < n) (h2 : 15 = n * 5 / 14) : n = 42 :=
sorry

end NUMINAMATH_GPT_cows_in_herd_l1427_142741


namespace NUMINAMATH_GPT_remainder_b_div_6_l1427_142762

theorem remainder_b_div_6 (a b : ℕ) (r_a r_b : ℕ) 
  (h1 : a ≡ r_a [MOD 6]) 
  (h2 : b ≡ r_b [MOD 6]) 
  (h3 : a > b) 
  (h4 : (a - b) % 6 = 5) 
  : b % 6 = 0 := 
sorry

end NUMINAMATH_GPT_remainder_b_div_6_l1427_142762


namespace NUMINAMATH_GPT_proof_problem_l1427_142755

variable (x y : ℕ) -- define x and y as natural numbers

-- Define the problem-specific variables m and n
variable (m n : ℕ)

-- Assume the conditions given in the problem
axiom H1 : 2 = m
axiom H2 : n = 3

-- The goal is to prove that -m^n equals -8 given the conditions H1 and H2
theorem proof_problem : - (m^n : ℤ) = -8 :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l1427_142755


namespace NUMINAMATH_GPT_overtime_rate_is_correct_l1427_142711

/-
Define the parameters:
ordinary_rate: Rate per hour for ordinary time in dollars
total_hours: Total hours worked in a week
overtime_hours: Overtime hours worked in a week
total_earnings: Total earnings for the week in dollars
-/

def ordinary_rate : ℝ := 0.60
def total_hours : ℝ := 50
def overtime_hours : ℝ := 8
def total_earnings : ℝ := 32.40

noncomputable def overtime_rate : ℝ :=
(total_earnings - ordinary_rate * (total_hours - overtime_hours)) / overtime_hours

theorem overtime_rate_is_correct :
  overtime_rate = 0.90 :=
by
  sorry

end NUMINAMATH_GPT_overtime_rate_is_correct_l1427_142711


namespace NUMINAMATH_GPT_proof_by_contradiction_example_l1427_142747

theorem proof_by_contradiction_example (a b : ℝ) (h : a + b ≥ 0) : ¬ (a < 0 ∧ b < 0) :=
by
  sorry

end NUMINAMATH_GPT_proof_by_contradiction_example_l1427_142747


namespace NUMINAMATH_GPT_total_cost_of_dresses_l1427_142760

-- Define the costs of each dress
variables (patty_cost ida_cost jean_cost pauline_cost total_cost : ℕ)

-- Given conditions
axiom pauline_cost_is_30 : pauline_cost = 30
axiom jean_cost_is_10_less_than_pauline : jean_cost = pauline_cost - 10
axiom ida_cost_is_30_more_than_jean : ida_cost = jean_cost + 30
axiom patty_cost_is_10_more_than_ida : patty_cost = ida_cost + 10

-- Statement to prove total cost
theorem total_cost_of_dresses : total_cost = pauline_cost + jean_cost + ida_cost + patty_cost 
                                 → total_cost = 160 :=
by {
  -- Proof is left as an exercise
  sorry
}

end NUMINAMATH_GPT_total_cost_of_dresses_l1427_142760


namespace NUMINAMATH_GPT_coin_stack_height_l1427_142710

def alpha_thickness : ℝ := 1.25
def beta_thickness : ℝ := 2.00
def gamma_thickness : ℝ := 0.90
def delta_thickness : ℝ := 1.60
def stack_height : ℝ := 18.00

theorem coin_stack_height :
  (∃ n : ℕ, stack_height = n * beta_thickness) ∨ (∃ n : ℕ, stack_height = n * gamma_thickness) :=
sorry

end NUMINAMATH_GPT_coin_stack_height_l1427_142710


namespace NUMINAMATH_GPT_quadratic_has_equal_roots_l1427_142759

-- Proposition: If the quadratic equation 3x^2 + 6x + m = 0 has two equal real roots, then m = 3.

theorem quadratic_has_equal_roots (m : ℝ) : 3 * 6 - 12 * m = 0 → m = 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_quadratic_has_equal_roots_l1427_142759


namespace NUMINAMATH_GPT_find_multiple_of_A_l1427_142797

def shares_division_problem (A B C : ℝ) (x : ℝ) : Prop :=
  C = 160 ∧
  x * A = 5 * B ∧
  x * A = 10 * C ∧
  A + B + C = 880

theorem find_multiple_of_A (A B C x : ℝ) (h : shares_division_problem A B C x) : x = 4 :=
by sorry

end NUMINAMATH_GPT_find_multiple_of_A_l1427_142797


namespace NUMINAMATH_GPT_maria_average_speed_l1427_142742

theorem maria_average_speed:
  let distance1 := 180
  let time1 := 4.5
  let distance2 := 270
  let time2 := 5.25
  let total_distance := distance1 + distance2
  let total_time := time1 + time2
  total_distance / total_time = 46.15 := by
  -- Sorry to skip the proof
  sorry

end NUMINAMATH_GPT_maria_average_speed_l1427_142742


namespace NUMINAMATH_GPT_minimum_value_y_l1427_142785

theorem minimum_value_y (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) :
  (∀ x : ℝ, x = (1 / a + 4 / b) → x ≥ 9 / 2) :=
sorry

end NUMINAMATH_GPT_minimum_value_y_l1427_142785


namespace NUMINAMATH_GPT_find_d_l1427_142726

theorem find_d {x d : ℤ} (h : (x + (x + 2) + (x + 4) + (x + 7) + (x + d)) / 5 = (x + 4) + 6) : d = 37 :=
sorry

end NUMINAMATH_GPT_find_d_l1427_142726


namespace NUMINAMATH_GPT_max_m_n_sq_l1427_142761

theorem max_m_n_sq (m n : ℕ) (hm : 1 ≤ m ∧ m ≤ 1981) (hn : 1 ≤ n ∧ n ≤ 1981)
  (h : (n^2 - m * n - m^2)^2 = 1) : m^2 + n^2 ≤ 3524578 :=
sorry

end NUMINAMATH_GPT_max_m_n_sq_l1427_142761


namespace NUMINAMATH_GPT_imaginary_unit_problem_l1427_142779

variable {a b : ℝ}

theorem imaginary_unit_problem (h : i * (a + i) = b + 2 * i) : a + b = 1 :=
sorry

end NUMINAMATH_GPT_imaginary_unit_problem_l1427_142779


namespace NUMINAMATH_GPT_stickers_per_student_l1427_142707

theorem stickers_per_student (G S B N: ℕ) (hG: G = 50) (hS: S = 2 * G) (hB: B = S - 20) (hN: N = 5) : 
  (G + S + B) / N = 46 := by
  sorry

end NUMINAMATH_GPT_stickers_per_student_l1427_142707


namespace NUMINAMATH_GPT_calculate_gain_percentage_l1427_142754

theorem calculate_gain_percentage (CP SP : ℝ) (h1 : 0.9 * CP = 450) (h2 : SP = 550) : 
  (SP - CP) / CP * 100 = 10 :=
by
  sorry

end NUMINAMATH_GPT_calculate_gain_percentage_l1427_142754


namespace NUMINAMATH_GPT_ratio_third_to_first_second_l1427_142713

-- Define the times spent on each step
def time_first_step : ℕ := 30
def time_second_step : ℕ := time_first_step / 2
def time_total : ℕ := 90
def time_third_step : ℕ := time_total - (time_first_step + time_second_step)

-- Define the combined time for the first two steps
def time_combined_first_second : ℕ := time_first_step + time_second_step

-- The goal is to prove that the ratio of the time spent on the third step to the combined time spent on the first and second steps is 1:1
theorem ratio_third_to_first_second : time_third_step = time_combined_first_second :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_ratio_third_to_first_second_l1427_142713


namespace NUMINAMATH_GPT_find_tan_theta_l1427_142730

theorem find_tan_theta
  (θ : ℝ)
  (h1 : θ ∈ Set.Ioc 0 (Real.pi / 4))
  (h2 : Real.sin θ + Real.cos θ = 17 / 13) :
  Real.tan θ = 5 / 12 :=
sorry

end NUMINAMATH_GPT_find_tan_theta_l1427_142730


namespace NUMINAMATH_GPT_two_times_koi_minus_X_is_64_l1427_142792

-- Definitions based on the conditions
def n : ℕ := 39
def X : ℕ := 14

-- Main proof statement
theorem two_times_koi_minus_X_is_64 : 2 * n - X = 64 :=
by
  sorry

end NUMINAMATH_GPT_two_times_koi_minus_X_is_64_l1427_142792


namespace NUMINAMATH_GPT_smaller_circle_y_coordinate_l1427_142733

theorem smaller_circle_y_coordinate 
  (center : ℝ × ℝ) 
  (P : ℝ × ℝ)
  (S : ℝ × ℝ) 
  (QR : ℝ)
  (r_large : ℝ):
    center = (0, 0) → P = (5, 12) → QR = 2 → S.1 = 0 → S.2 = k → r_large = 13 → k = 11 := 
by
  intros h_center hP hQR hSx hSy hr_large
  sorry

end NUMINAMATH_GPT_smaller_circle_y_coordinate_l1427_142733


namespace NUMINAMATH_GPT_common_difference_ne_3_l1427_142729

theorem common_difference_ne_3 
  (d : ℕ) (hd_pos : d > 0) 
  (exists_n : ∃ n : ℕ, 81 = 1 + (n - 1) * d) : 
  d ≠ 3 :=
by sorry

end NUMINAMATH_GPT_common_difference_ne_3_l1427_142729


namespace NUMINAMATH_GPT_quadratic_roster_method_l1427_142789

theorem quadratic_roster_method :
  {x : ℝ | x^2 - 3 * x + 2 = 0} = {1, 2} :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roster_method_l1427_142789


namespace NUMINAMATH_GPT_trig_relation_l1427_142786

theorem trig_relation : (Real.pi/4 < 1) ∧ (1 < Real.pi/2) → Real.tan 1 > Real.sin 1 ∧ Real.sin 1 > Real.cos 1 := 
by 
  intro h
  sorry

end NUMINAMATH_GPT_trig_relation_l1427_142786


namespace NUMINAMATH_GPT_range_of_a_l1427_142701

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 1 < x ∧ x ≤ 2}
def B (a : ℝ) : Set ℝ := {x : ℝ | x < a}

-- Lean statement asserting the requirement
theorem range_of_a (a : ℝ) (h : A ⊆ B a ∧ A ≠ B a) : 2 < a := by
  sorry

end NUMINAMATH_GPT_range_of_a_l1427_142701


namespace NUMINAMATH_GPT_find_value_l1427_142723

variable (a b : ℝ)

def quadratic_equation_roots : Prop :=
  a^2 - 4 * a - 1 = 0 ∧ b^2 - 4 * b - 1 = 0

def sum_of_roots : Prop :=
  a + b = 4

def product_of_roots : Prop :=
  a * b = -1

theorem find_value (ha : quadratic_equation_roots a b) (hs : sum_of_roots a b) (hp : product_of_roots a b) :
  2 * a^2 + 3 / b + 5 * b = 22 :=
sorry

end NUMINAMATH_GPT_find_value_l1427_142723


namespace NUMINAMATH_GPT_g_is_even_l1427_142795

noncomputable def g (x : ℝ) : ℝ := Real.log (Real.cos x + Real.sqrt (1 + Real.sin x ^ 2))

theorem g_is_even : ∀ x : ℝ, g (-x) = g (x) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_g_is_even_l1427_142795


namespace NUMINAMATH_GPT_set_intersection_complement_l1427_142708

def U : Set ℝ := Set.univ
def A : Set ℝ := { y | ∃ x, x > 0 ∧ y = 4 / x }
def B : Set ℝ := { y | ∃ x, x < 1 ∧ y = 2^x }
def comp_B : Set ℝ := { y | y ≤ 0 } ∪ { y | y ≥ 2 }
def intersection : Set ℝ := { y | y ≥ 2 }

theorem set_intersection_complement :
  A ∩ comp_B = intersection :=
by
  sorry

end NUMINAMATH_GPT_set_intersection_complement_l1427_142708


namespace NUMINAMATH_GPT_find_k_l1427_142719

-- Defining the vectors
def a (k : ℝ) : ℝ × ℝ := (k, -2)
def b : ℝ × ℝ := (2, 2)

-- Condition 1: a + b is not the zero vector
def non_zero_sum (k : ℝ) := (a k).1 + b.1 ≠ 0 ∨ (a k).2 + b.2 ≠ 0

-- Condition 2: a is perpendicular to a + b
def perpendicular (k : ℝ) := (a k).1 * ((a k).1 + b.1) + (a k).2 * ((a k).2 + b.2) = 0

-- The theorem to prove
theorem find_k (k : ℝ) (cond1 : non_zero_sum k) (cond2 : perpendicular k) : k = 0 := 
sorry

end NUMINAMATH_GPT_find_k_l1427_142719


namespace NUMINAMATH_GPT_triangle_area_inequality_l1427_142752

variables {a b c S x y z T : ℝ}

-- Definitions based on the given conditions
def side_lengths_of_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def area_of_triangle (a b c S : ℝ) : Prop :=
  16 * S * S = (a + b + c) * (a + b - c) * (a - b + c) * (-a + b + c)

def new_side_lengths (a b c : ℝ) (x y z : ℝ) : Prop :=
  x = a + b / 2 ∧ y = b + c / 2 ∧ z = c + a / 2

def area_condition (S T : ℝ) : Prop :=
  T ≥ 9 / 4 * S

-- Main theorem statement
theorem triangle_area_inequality
  (h_triangle: side_lengths_of_triangle a b c)
  (h_area: area_of_triangle a b c S)
  (h_new_sides: new_side_lengths a b c x y z) :
  ∃ T : ℝ, side_lengths_of_triangle x y z ∧ area_condition S T :=
sorry

end NUMINAMATH_GPT_triangle_area_inequality_l1427_142752


namespace NUMINAMATH_GPT_cos_2alpha_minus_pi_over_6_l1427_142705

theorem cos_2alpha_minus_pi_over_6 (α : ℝ) (hα : 0 < α ∧ α < π / 2) (hSin : Real.sin (α + π / 6) = 3 / 5) :
  Real.cos (2 * α - π / 6) = 24 / 25 :=
sorry

end NUMINAMATH_GPT_cos_2alpha_minus_pi_over_6_l1427_142705


namespace NUMINAMATH_GPT_find_k_parallel_vectors_l1427_142798

def vector_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

theorem find_k_parallel_vectors (k : ℝ) :
  let a := (1, k)
  let b := (-2, 6)
  vector_parallel a b → k = -3 :=
by
  sorry

end NUMINAMATH_GPT_find_k_parallel_vectors_l1427_142798


namespace NUMINAMATH_GPT_ConfuciusBirthYear_l1427_142720

-- Definitions based on the conditions provided
def birthYearAD (year : Int) : Int := year

def birthYearBC (year : Int) : Int := -year

theorem ConfuciusBirthYear :
  birthYearBC 551 = -551 :=
by
  sorry

end NUMINAMATH_GPT_ConfuciusBirthYear_l1427_142720


namespace NUMINAMATH_GPT_select_and_swap_ways_l1427_142768

theorem select_and_swap_ways :
  let n := 8
  let k := 3
  Nat.choose n k * 2 = 112 := 
by
  let n := 8
  let k := 3
  sorry

end NUMINAMATH_GPT_select_and_swap_ways_l1427_142768


namespace NUMINAMATH_GPT_hexagon_circle_radius_l1427_142743

noncomputable def hexagon_radius (sides : List ℝ) (probability : ℝ) : ℝ :=
  let total_angle := 360.0
  let visible_angle := probability * total_angle
  let side_length_average := (sides.sum / sides.length : ℝ)
  let theta := (visible_angle / 6 : ℝ) -- assuming θ approximately splits equally among 6 gaps
  side_length_average / Real.sin (theta / 2 * Real.pi / 180.0)

theorem hexagon_circle_radius :
  hexagon_radius [3, 2, 4, 3, 2, 4] (1 / 3) = 17.28 :=
by
  sorry

end NUMINAMATH_GPT_hexagon_circle_radius_l1427_142743


namespace NUMINAMATH_GPT_min_xyz_product_l1427_142780

open Real

theorem min_xyz_product
  (x y z : ℝ)
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h_sum : x + y + z = 1)
  (h_no_more_than_twice : x ≤ 2 * y ∧ y ≤ 2 * x ∧ y ≤ 2 * z ∧ z ≤ 2 * y) :
  ∃ p : ℝ, (∀ x y z : ℝ, 0 < x → 0 < y → 0 < z → x + y + z = 1 → x ≤ 2 * y ∧ y ≤ 2 * x ∧ y ≤ 2 * z ∧ z ≤ 2 * y → x * y * z ≥ p) ∧ p = 1 / 32 :=
by
  sorry

end NUMINAMATH_GPT_min_xyz_product_l1427_142780


namespace NUMINAMATH_GPT_ratio_x_to_w_as_percentage_l1427_142732

theorem ratio_x_to_w_as_percentage (x y z w : ℝ) 
    (h1 : x = 1.20 * y) 
    (h2 : y = 0.30 * z) 
    (h3 : z = 1.35 * w) : 
    (x / w) * 100 = 48.6 := 
by sorry

end NUMINAMATH_GPT_ratio_x_to_w_as_percentage_l1427_142732


namespace NUMINAMATH_GPT_maximum_value_at_vertex_l1427_142717

-- Defining the parabola as a function
def parabola (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Defining the vertex condition
def vertex_condition (a b c : ℝ) := ∀ x : ℝ, parabola a b c x = a * x^2 + b * x + c

-- Defining the condition that the parabola opens downward
def opens_downward (a : ℝ) := a < 0

-- Defining the vertex coordinates condition
def vertex_coordinates (a b c : ℝ) := 
  ∃ (x₀ y₀ : ℝ), x₀ = 2 ∧ y₀ = -3 ∧ parabola a b c x₀ = y₀

-- The main theorem statement
theorem maximum_value_at_vertex (a b c : ℝ) (h1 : opens_downward a) (h2 : vertex_coordinates a b c) : ∃ y₀, y₀ = -3 ∧ ∀ x : ℝ, parabola a b c x ≤ y₀ :=
by
  sorry

end NUMINAMATH_GPT_maximum_value_at_vertex_l1427_142717


namespace NUMINAMATH_GPT_solve_for_x_l1427_142769

variable (x : ℝ)

theorem solve_for_x (h : 2 * x - 3 * x + 4 * x = 150) : x = 50 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1427_142769


namespace NUMINAMATH_GPT_range_of_m_l1427_142745

theorem range_of_m (m : ℝ) :
  (1 - 2 * m > 0) ∧ (m + 1 > 0) → -1 < m ∧ m < 1/2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1427_142745


namespace NUMINAMATH_GPT_find_y_is_90_l1427_142714

-- Definitions for given conditions
def angle_ABC : ℝ := 120
def angle_ABD : ℝ := 180 - angle_ABC
def angle_BDA : ℝ := 30

-- The theorem to prove y = 90 degrees
theorem find_y_is_90 :
  ∃ y : ℝ, angle_ABD = 60 ∧ angle_BDA = 30 ∧ (30 + 60 + y = 180) → y = 90 :=
by
  sorry

end NUMINAMATH_GPT_find_y_is_90_l1427_142714


namespace NUMINAMATH_GPT_distinct_valid_c_values_l1427_142703

theorem distinct_valid_c_values : 
  let is_solution (c : ℤ) (x : ℚ) := (5 * ⌊x⌋₊ + 3 * ⌈x⌉₊ = c) 
  ∃ s : Finset ℤ, (∀ c ∈ s, (∃ x : ℚ, is_solution c x)) ∧ s.card = 500 :=
by sorry

end NUMINAMATH_GPT_distinct_valid_c_values_l1427_142703


namespace NUMINAMATH_GPT_arithmetic_progression_rth_term_l1427_142753

open Nat

theorem arithmetic_progression_rth_term (n r : ℕ) (Sn : ℕ → ℕ) 
  (h : ∀ n, Sn n = 5 * n + 4 * n^2) : Sn r - Sn (r - 1) = 8 * r + 1 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_progression_rth_term_l1427_142753


namespace NUMINAMATH_GPT_number_of_integers_covered_l1427_142709

-- Define the number line and the length condition
def unit_length_cm (p : ℝ) := p = 1
def length_AB_cm (length : ℝ) := length = 2009

-- Statement of the proof problem in Lean
theorem number_of_integers_covered (ab_length : ℝ) (unit_length : ℝ) 
    (h1 : unit_length_cm unit_length) (h2 : length_AB_cm ab_length) :
    ∃ n : ℕ, n = 2009 ∨ n = 2010 :=
by
  sorry

end NUMINAMATH_GPT_number_of_integers_covered_l1427_142709


namespace NUMINAMATH_GPT_percentage_change_difference_l1427_142776

-- Define initial and final percentages
def initial_yes : ℝ := 0.4
def initial_no : ℝ := 0.6
def final_yes : ℝ := 0.6
def final_no : ℝ := 0.4

-- Definition for the percentage of students who changed their opinion
def y_min : ℝ := 0.2 -- 20%
def y_max : ℝ := 0.6 -- 60%

-- Calculate the difference
def difference_y : ℝ := y_max - y_min

theorem percentage_change_difference :
  difference_y = 0.4 := by
  sorry

end NUMINAMATH_GPT_percentage_change_difference_l1427_142776


namespace NUMINAMATH_GPT_minimum_glue_drops_to_prevent_37_gram_subset_l1427_142725

def stones : List ℕ := List.range' 1 36  -- List of stones with masses from 1 to 36 grams

def glue_drop_combination_invalid (stones : List ℕ) : Prop :=
  ¬ (∃ (subset : List ℕ), subset.sum = 37 ∧ (∀ s ∈ subset, s ∈ stones))

def min_glue_drops (stones : List ℕ) : ℕ := 
  9 -- as per the solution

theorem minimum_glue_drops_to_prevent_37_gram_subset :
  ∀ (s : List ℕ), s = stones → glue_drop_combination_invalid s → min_glue_drops s = 9 :=
by intros; sorry

end NUMINAMATH_GPT_minimum_glue_drops_to_prevent_37_gram_subset_l1427_142725


namespace NUMINAMATH_GPT_calculate_lives_lost_l1427_142740

-- Define the initial number of lives
def initial_lives : ℕ := 98

-- Define the remaining number of lives
def remaining_lives : ℕ := 73

-- Define the number of lives lost
def lives_lost : ℕ := initial_lives - remaining_lives

-- Prove that Kaleb lost 25 lives
theorem calculate_lives_lost : lives_lost = 25 := 
by {
  -- The proof would go here, but we'll skip it
  sorry
}

end NUMINAMATH_GPT_calculate_lives_lost_l1427_142740


namespace NUMINAMATH_GPT_sleeping_bag_selling_price_l1427_142700

def wholesale_cost : ℝ := 24.56
def gross_profit_percentage : ℝ := 0.14

def gross_profit (x : ℝ) : ℝ := gross_profit_percentage * x

def selling_price (x y : ℝ) : ℝ := x + y

theorem sleeping_bag_selling_price :
  selling_price wholesale_cost (gross_profit wholesale_cost) = 28 := by
  sorry

end NUMINAMATH_GPT_sleeping_bag_selling_price_l1427_142700


namespace NUMINAMATH_GPT_sin_105_mul_sin_15_eq_one_fourth_l1427_142727

noncomputable def sin_105_deg := Real.sin (105 * Real.pi / 180)
noncomputable def sin_15_deg := Real.sin (15 * Real.pi / 180)

theorem sin_105_mul_sin_15_eq_one_fourth :
  sin_105_deg * sin_15_deg = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_sin_105_mul_sin_15_eq_one_fourth_l1427_142727


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1427_142799

variable (S : ℕ → ℕ)   -- S is a function that gives the sum of the first k*n terms

theorem arithmetic_sequence_sum
  (n : ℕ)
  (h1 : S n = 45)
  (h2 : S (2 * n) = 60) :
  S (3 * n) = 65 := sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1427_142799


namespace NUMINAMATH_GPT_disjoint_subsets_exist_l1427_142712

theorem disjoint_subsets_exist (n : ℕ) (h : 0 < n) 
  (A : Fin (n + 1) → Set (Fin n)) (hA : ∀ i : Fin (n + 1), A i ≠ ∅) :
  ∃ (I J : Finset (Fin (n + 1))), I ≠ ∅ ∧ J ≠ ∅ ∧ Disjoint I J ∧ 
    (⋃ i ∈ I, A i) = (⋃ j ∈ J, A j) :=
sorry

end NUMINAMATH_GPT_disjoint_subsets_exist_l1427_142712


namespace NUMINAMATH_GPT_segment_length_l1427_142781

theorem segment_length (A B C : ℝ) (hAB : abs (A - B) = 3) (hBC : abs (B - C) = 5) :
  abs (A - C) = 2 ∨ abs (A - C) = 8 := by
  sorry

end NUMINAMATH_GPT_segment_length_l1427_142781


namespace NUMINAMATH_GPT_gcd_of_90_and_405_l1427_142744

def gcd_90_405 : ℕ := Nat.gcd 90 405

theorem gcd_of_90_and_405 : gcd_90_405 = 45 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_gcd_of_90_and_405_l1427_142744


namespace NUMINAMATH_GPT_tunnel_depth_l1427_142790

theorem tunnel_depth (topWidth : ℝ) (bottomWidth : ℝ) (area : ℝ) (h : ℝ)
  (h1 : topWidth = 15)
  (h2 : bottomWidth = 5)
  (h3 : area = 400)
  (h4 : area = (1 / 2) * (topWidth + bottomWidth) * h) :
  h = 40 := 
sorry

end NUMINAMATH_GPT_tunnel_depth_l1427_142790


namespace NUMINAMATH_GPT_orthographic_projection_area_l1427_142772

theorem orthographic_projection_area (s : ℝ) (h : s = 1) : 
  let S := (Real.sqrt 3) / 4 
  let factor := (Real.sqrt 2) / 2
  let S' := (factor ^ 2) * S
  S' = (Real.sqrt 6) / 16 :=
by
  let S := (Real.sqrt 3) / 4
  let factor := (Real.sqrt 2) / 2
  let S' := (factor ^ 2) * S
  sorry

end NUMINAMATH_GPT_orthographic_projection_area_l1427_142772


namespace NUMINAMATH_GPT_radius_of_tangent_circle_l1427_142738

theorem radius_of_tangent_circle (side_length : ℝ) (num_semicircles : ℕ)
  (r_s : ℝ) (r : ℝ)
  (h1 : side_length = 4)
  (h2 : num_semicircles = 16)
  (h3 : r_s = side_length / 4 / 2)
  (h4 : r = (9 : ℝ) / (2 * Real.sqrt 5)) :
  r = (9 * Real.sqrt 5) / 10 :=
by
  rw [h4]
  sorry

end NUMINAMATH_GPT_radius_of_tangent_circle_l1427_142738


namespace NUMINAMATH_GPT_ratio_of_savings_to_earnings_l1427_142764

-- Definitions based on the given conditions
def earnings_washing_cars : ℤ := 20
def earnings_walking_dogs : ℤ := 40
def total_savings : ℤ := 150
def months : ℤ := 5

-- Statement to prove the ratio of savings per month to total earnings per month
theorem ratio_of_savings_to_earnings :
  (total_savings / months) = (earnings_washing_cars + earnings_walking_dogs) / 2 := by
  sorry

end NUMINAMATH_GPT_ratio_of_savings_to_earnings_l1427_142764


namespace NUMINAMATH_GPT_minimum_value_f_on_interval_l1427_142784

noncomputable def f (x : ℝ) : ℝ := (Real.cos x)^3 / (Real.sin x) + (Real.sin x)^3 / (Real.cos x)

theorem minimum_value_f_on_interval : ∃ x ∈ Set.Ioo 0 (Real.pi / 2), f x = 1 ∧ ∀ y ∈ Set.Ioo 0 (Real.pi / 2), f y ≥ 1 :=
by sorry

end NUMINAMATH_GPT_minimum_value_f_on_interval_l1427_142784


namespace NUMINAMATH_GPT_number_added_is_10_l1427_142767

theorem number_added_is_10 (x y a : ℕ) (h1 : y = 40) 
  (h2 : x * 4 = 3 * y) 
  (h3 : (x + a) * 5 = 4 * (y + a)) : a = 10 := 
by
  sorry

end NUMINAMATH_GPT_number_added_is_10_l1427_142767


namespace NUMINAMATH_GPT_problem_statement_l1427_142748

theorem problem_statement (n : ℕ) : 2 ^ n ∣ (1 + ⌊(3 + Real.sqrt 5) ^ n⌋) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1427_142748


namespace NUMINAMATH_GPT_minimum_value_f_condition_f_geq_zero_l1427_142778

noncomputable def f (x a : ℝ) := Real.exp x - a * x - 1

theorem minimum_value_f (a : ℝ) (h : 0 < a) : 
  (∀ x : ℝ, f x a ≥ f (Real.log a) a) ∧ f (Real.log a) a = a - a * Real.log a - 1 :=
by 
  sorry

theorem condition_f_geq_zero (a : ℝ) (h : 0 < a) : 
  (∀ x : ℝ, f x a ≥ 0) ↔ a = 1 :=
by 
  sorry

end NUMINAMATH_GPT_minimum_value_f_condition_f_geq_zero_l1427_142778


namespace NUMINAMATH_GPT_aunt_may_milk_leftover_l1427_142763

noncomputable def milk_leftover : Real :=
let morning_milk := 5 * 13 + 4 * 0.5 + 10 * 0.25
let evening_milk := 5 * 14 + 4 * 0.6 + 10 * 0.2

let morning_spoiled := morning_milk * 0.1
let cheese_produced := morning_milk * 0.15
let remaining_morning_milk := morning_milk - morning_spoiled - cheese_produced
let ice_cream_sale := remaining_morning_milk * 0.7

let evening_spoiled := evening_milk * 0.05
let remaining_evening_milk := evening_milk - evening_spoiled
let cheese_shop_sale := remaining_evening_milk * 0.8

let leftover_previous_day := 15
let remaining_morning_after_sale := remaining_morning_milk - ice_cream_sale
let remaining_evening_after_sale := remaining_evening_milk - cheese_shop_sale

leftover_previous_day + remaining_morning_after_sale + remaining_evening_after_sale

theorem aunt_may_milk_leftover : 
  milk_leftover = 44.7735 := 
sorry

end NUMINAMATH_GPT_aunt_may_milk_leftover_l1427_142763


namespace NUMINAMATH_GPT_intersection_of_sets_l1427_142773

def A := { x : ℝ | x^2 - 2 * x - 8 < 0 }
def B := { x : ℝ | x >= 0 }
def intersection := { x : ℝ | 0 <= x ∧ x < 4 }

theorem intersection_of_sets : (A ∩ B) = intersection := 
sorry

end NUMINAMATH_GPT_intersection_of_sets_l1427_142773


namespace NUMINAMATH_GPT_jane_savings_l1427_142771

noncomputable def cost_promotion_A (price: ℝ) : ℝ :=
  price + (price / 2)

noncomputable def cost_promotion_B (price: ℝ) : ℝ :=
  price + (price - (price * 0.25))

theorem jane_savings (price : ℝ) (h_price_pos : 0 < price) : 
  cost_promotion_B price - cost_promotion_A price = 12.5 :=
by
  let price := 50
  unfold cost_promotion_A
  unfold cost_promotion_B
  norm_num
  sorry

end NUMINAMATH_GPT_jane_savings_l1427_142771


namespace NUMINAMATH_GPT_triangle_angle_bisector_YE_l1427_142793

noncomputable def triangle_segs_YE : ℝ := (36 : ℝ) / 7

theorem triangle_angle_bisector_YE
  (XYZ: Type)
  (XY XZ YZ YE EZ: ℝ)
  (YZ_length : YZ = 12)
  (side_ratios : XY / XZ = 3 / 4 ∧ XY / YZ  = 3 / 5 ∧ XZ / YZ = 4 / 5)
  (angle_bisector : YE / EZ = XY / XZ)
  (seg_sum : YE + EZ = YZ) :
  YE = (36 : ℝ) / 7 :=
by sorry

end NUMINAMATH_GPT_triangle_angle_bisector_YE_l1427_142793


namespace NUMINAMATH_GPT_new_average_weight_l1427_142782

theorem new_average_weight (original_players : ℕ) (new_players : ℕ) 
  (average_weight_original : ℝ) (weight_new_player1 : ℝ) (weight_new_player2 : ℝ) : 
  original_players = 7 → 
  new_players = 2 →
  average_weight_original = 76 → 
  weight_new_player1 = 110 → 
  weight_new_player2 = 60 → 
  (original_players * average_weight_original + weight_new_player1 + weight_new_player2) / (original_players + new_players) = 78 :=
by 
  intros h1 h2 h3 h4 h5;
  sorry

end NUMINAMATH_GPT_new_average_weight_l1427_142782


namespace NUMINAMATH_GPT_sum_equals_one_l1427_142731

noncomputable def sum_proof (x y z : ℝ) (h : x * y * z = 1) : ℝ :=
  (1 / (1 + x + x * y)) + (1 / (1 + y + y * z)) + (1 / (1 + z + z * x))

theorem sum_equals_one (x y z : ℝ) (h : x * y * z = 1) : 
  sum_proof x y z h = 1 := sorry

end NUMINAMATH_GPT_sum_equals_one_l1427_142731


namespace NUMINAMATH_GPT_max_pies_without_ingredients_l1427_142788

theorem max_pies_without_ingredients :
  let total_pies := 36
  let chocolate_pies := total_pies / 3
  let marshmallow_pies := total_pies / 4
  let cayenne_pies := total_pies / 2
  let soy_nuts_pies := total_pies / 8
  let max_ingredient_pies := max (max chocolate_pies marshmallow_pies) (max cayenne_pies soy_nuts_pies)
  total_pies - max_ingredient_pies = 18 :=
by
  sorry

end NUMINAMATH_GPT_max_pies_without_ingredients_l1427_142788


namespace NUMINAMATH_GPT_total_games_played_l1427_142766

theorem total_games_played (n : ℕ) (h : n = 7) : (n.choose 2) = 21 := by
  sorry

end NUMINAMATH_GPT_total_games_played_l1427_142766


namespace NUMINAMATH_GPT_brother_age_in_5_years_l1427_142702

theorem brother_age_in_5_years
  (nick_age : ℕ)
  (sister_age : ℕ)
  (brother_age : ℕ)
  (h_nick : nick_age = 13)
  (h_sister : sister_age = nick_age + 6)
  (h_brother : brother_age = (nick_age + sister_age) / 2) :
  brother_age + 5 = 21 := 
by 
  sorry

end NUMINAMATH_GPT_brother_age_in_5_years_l1427_142702


namespace NUMINAMATH_GPT_primes_with_no_sum_of_two_cubes_l1427_142787

theorem primes_with_no_sum_of_two_cubes (p : ℕ) [Fact (Nat.Prime p)] :
  (∃ n : ℤ, ∀ x y : ℤ, x^3 + y^3 ≠ n % p) ↔ p = 7 :=
sorry

end NUMINAMATH_GPT_primes_with_no_sum_of_two_cubes_l1427_142787


namespace NUMINAMATH_GPT_combined_dog_years_difference_l1427_142758

theorem combined_dog_years_difference 
  (Max_age : ℕ) 
  (small_breed_rate medium_breed_rate large_breed_rate : ℕ) 
  (Max_turns_age : ℕ) 
  (small_breed_diff medium_breed_diff large_breed_diff combined_diff : ℕ) :
  Max_age = 3 →
  small_breed_rate = 5 →
  medium_breed_rate = 7 →
  large_breed_rate = 9 →
  Max_turns_age = 6 →
  small_breed_diff = small_breed_rate * Max_turns_age - Max_turns_age →
  medium_breed_diff = medium_breed_rate * Max_turns_age - Max_turns_age →
  large_breed_diff = large_breed_rate * Max_turns_age - Max_turns_age →
  combined_diff = small_breed_diff + medium_breed_diff + large_breed_diff →
  combined_diff = 108 :=
by
  intros
  sorry

end NUMINAMATH_GPT_combined_dog_years_difference_l1427_142758


namespace NUMINAMATH_GPT_trigonometric_proof_l1427_142783

noncomputable def cos30 : ℝ := Real.sqrt 3 / 2
noncomputable def tan60 : ℝ := Real.sqrt 3
noncomputable def sin45 : ℝ := Real.sqrt 2 / 2
noncomputable def cos45 : ℝ := Real.sqrt 2 / 2

theorem trigonometric_proof :
  2 * cos30 - tan60 + sin45 * cos45 = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_proof_l1427_142783


namespace NUMINAMATH_GPT_count_consecutive_integers_l1427_142706

theorem count_consecutive_integers : 
  ∃ n : ℕ, (∀ x : ℕ, (1 < x ∧ x < 111) → (x - 1) + x + (x + 1) < 333) ∧ n = 109 := 
  by
    sorry

end NUMINAMATH_GPT_count_consecutive_integers_l1427_142706


namespace NUMINAMATH_GPT_distance_from_P_to_y_axis_l1427_142737

theorem distance_from_P_to_y_axis (P : ℝ × ℝ) :
  (P.2 ^ 2 = -12 * P.1) → (dist P (-3, 0) = 9) → abs P.1 = 6 :=
by
  sorry

end NUMINAMATH_GPT_distance_from_P_to_y_axis_l1427_142737


namespace NUMINAMATH_GPT_basketball_player_possible_scores_l1427_142728

-- Define the conditions
def isValidBasketCount (n : Nat) : Prop := n = 7
def isValidBasketValue (v : Nat) : Prop := v = 1 ∨ v = 2 ∨ v = 3

-- Define the theorem statement
theorem basketball_player_possible_scores :
  ∃ (s : Finset ℕ), s = {n | ∃ n1 n2 n3 : Nat, 
                                n1 + n2 + n3 = 7 ∧ 
                                n = 1 * n1 + 2 * n2 + 3 * n3 ∧ 
                                n1 + n2 + n3 = 7 ∧ 
                                n >= 7 ∧ n <= 21} ∧
                                s.card = 15 :=
by
  sorry

end NUMINAMATH_GPT_basketball_player_possible_scores_l1427_142728


namespace NUMINAMATH_GPT_largest_is_D_l1427_142734

-- Definitions based on conditions
def A : ℕ := 27
def B : ℕ := A + 7
def C : ℕ := B - 9
def D : ℕ := 2 * C

-- Theorem stating D is the largest
theorem largest_is_D : D = max (max A B) (max C D) :=
by
  -- Inserting sorry because the proof is not required.
  sorry

end NUMINAMATH_GPT_largest_is_D_l1427_142734


namespace NUMINAMATH_GPT_ratio_of_perimeters_l1427_142794

theorem ratio_of_perimeters (s₁ s₂ : ℝ) (h : (s₁^2 / s₂^2) = (16 / 49)) : (4 * s₁) / (4 * s₂) = 4 / 7 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_ratio_of_perimeters_l1427_142794


namespace NUMINAMATH_GPT_stockholm_to_uppsala_distance_l1427_142746

theorem stockholm_to_uppsala_distance :
  let map_distance_cm : ℝ := 45
  let map_scale_cm_to_km : ℝ := 10
  (map_distance_cm * map_scale_cm_to_km = 450) :=
by
  sorry

end NUMINAMATH_GPT_stockholm_to_uppsala_distance_l1427_142746


namespace NUMINAMATH_GPT_simple_interest_problem_l1427_142775

theorem simple_interest_problem (P : ℝ) (R : ℝ) (T : ℝ) : T = 10 → 
  ((P * R * T) / 100 = (4 / 5) * P) → R = 8 :=
by
  intros hT hsi
  sorry

end NUMINAMATH_GPT_simple_interest_problem_l1427_142775


namespace NUMINAMATH_GPT_boat_travel_distance_downstream_l1427_142777

def boat_speed : ℝ := 22 -- Speed of boat in still water in km/hr
def stream_speed : ℝ := 5 -- Speed of the stream in km/hr
def time_downstream : ℝ := 7 -- Time taken to travel downstream in hours
def effective_speed_downstream : ℝ := boat_speed + stream_speed -- Effective speed downstream

theorem boat_travel_distance_downstream : effective_speed_downstream * time_downstream = 189 := by
  -- Since effective_speed_downstream = 27 (22 + 5)
  -- Distance = Speed * Time
  -- Hence, Distance = 27 km/hr * 7 hours = 189 km
  sorry

end NUMINAMATH_GPT_boat_travel_distance_downstream_l1427_142777


namespace NUMINAMATH_GPT_coprime_integers_exist_l1427_142791

theorem coprime_integers_exist (a b c : ℚ) (t : ℤ) (h1 : a + b + c = t) (h2 : a^2 + b^2 + c^2 = t) (h3 : t ≥ 0) : 
  ∃ (u v : ℤ), Int.gcd u v = 1 ∧ abc = (u^2 : ℚ) / (v^3 : ℚ) :=
by sorry

end NUMINAMATH_GPT_coprime_integers_exist_l1427_142791


namespace NUMINAMATH_GPT_water_level_function_l1427_142770

def water_level (x : ℝ) (h : 0 ≤ x ∧ x ≤ 5) : ℝ :=
  0.3 * x + 6

theorem water_level_function :
  ∀ (x : ℝ) (h : 0 ≤ x ∧ x ≤ 5), water_level x h = 6 + 0.3 * x :=
by
  intros
  unfold water_level
  sorry -- Proof skipped

end NUMINAMATH_GPT_water_level_function_l1427_142770


namespace NUMINAMATH_GPT_max_sum_first_n_terms_formula_sum_terms_abs_l1427_142739

theorem max_sum_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) :
  a 1 = 29 ∧ S 10 = S 20 →
  ∃ (n : ℕ), n = 15 ∧ S 15 = 225 := by
  sorry

theorem formula_sum_terms_abs (a : ℕ → ℤ) (S T : ℕ → ℤ) :
  a 1 = 29 ∧ S 10 = S 20 →
  (∀ n, n ≤ 15 → T n = 30 * n - n * n) ∧
  (∀ n, n ≥ 16 → T n = n * n - 30 * n + 450) := by
  sorry

end NUMINAMATH_GPT_max_sum_first_n_terms_formula_sum_terms_abs_l1427_142739


namespace NUMINAMATH_GPT_find_geometric_progression_l1427_142750

theorem find_geometric_progression (a b c : ℚ)
  (h1 : a * c = b * b)
  (h2 : a + c = 2 * (b + 8))
  (h3 : a * (c + 64) = (b + 8) * (b + 8)) :
  (a = 4/9 ∧ b = -20/9 ∧ c = 100/9) ∨ (a = 4 ∧ b = 12 ∧ c = 36) :=
sorry

end NUMINAMATH_GPT_find_geometric_progression_l1427_142750


namespace NUMINAMATH_GPT_problem_proof_l1427_142774

theorem problem_proof (n : ℕ) 
  (h : ∃ k, 2 * k = n) :
  4 ∣ n :=
sorry

end NUMINAMATH_GPT_problem_proof_l1427_142774


namespace NUMINAMATH_GPT_product_of_solutions_eq_zero_l1427_142735

theorem product_of_solutions_eq_zero : 
  (∀ x : ℝ, (x + 3) / (2 * x + 3) = (4 * x + 4) / (7 * x + 4)) → 
  ∃ (x1 x2 : ℝ), (x1 = 0 ∨ x1 = 5) ∧ (x2 = 0 ∨ x2 = 5) ∧ x1 * x2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_product_of_solutions_eq_zero_l1427_142735


namespace NUMINAMATH_GPT_isosceles_triangle_base_angle_l1427_142765

/-- In an isosceles triangle, if one angle is 110 degrees, then each base angle measures 35 degrees. -/
theorem isosceles_triangle_base_angle (α β γ : ℝ) (h1 : α + β + γ = 180)
  (h2 : α = β ∨ α = γ ∨ β = γ) (h3 : α = 110 ∨ β = 110 ∨ γ = 110) :
  β = 35 ∨ γ = 35 :=
sorry

end NUMINAMATH_GPT_isosceles_triangle_base_angle_l1427_142765


namespace NUMINAMATH_GPT_book_pages_l1427_142736

theorem book_pages (P : ℝ) (h1 : P / 2 + 0.15 * (P / 2) + 210 = P) : P = 600 := 
sorry

end NUMINAMATH_GPT_book_pages_l1427_142736


namespace NUMINAMATH_GPT_range_of_a_l1427_142704

open Real

/-- Proposition p: x^2 + 2*a*x + 4 > 0 for all x in ℝ -/
def p (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + 2*a*x + 4 > 0

/-- Proposition q: the exponential function (3 - 2*a)^x is increasing -/
def q (a : ℝ) : Prop :=
  3 - 2*a > 1

/-- Given p ∧ q, prove that -2 < a < 1 -/
theorem range_of_a (a : ℝ) (hp : p a) (hq : q a) : -2 < a ∧ a < 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1427_142704


namespace NUMINAMATH_GPT_cost_per_book_l1427_142749

-- Definitions and conditions
def number_of_books : ℕ := 8
def amount_tommy_has : ℕ := 13
def amount_tommy_needs_to_save : ℕ := 27

-- Total money Tommy needs to buy the books
def total_amount_needed : ℕ := amount_tommy_has + amount_tommy_needs_to_save

-- Proven statement
theorem cost_per_book : (total_amount_needed / number_of_books) = 5 := by
  -- Skip proof
  sorry

end NUMINAMATH_GPT_cost_per_book_l1427_142749


namespace NUMINAMATH_GPT_ratio_x_y_l1427_142796

noncomputable def side_length_x (x : ℝ) : Prop :=
  ∃ (a b c : ℝ), 
    a = 5 ∧ b = 12 ∧ c = 13 ∧ 
    (12 - x) / x = 5 / 12 ∧
    12 * x = 5 * x + 60 ∧
    7 * x = 60

noncomputable def side_length_y (y : ℝ) : Prop :=
  ∃ (a b c : ℝ), 
    a = 5 ∧ b = 12 ∧ c = 13 ∧
    y = 60 / 17

theorem ratio_x_y (x y : ℝ) (hx : side_length_x x) (hy : side_length_y y) : x / y = 17 / 7 :=
by
  sorry

end NUMINAMATH_GPT_ratio_x_y_l1427_142796


namespace NUMINAMATH_GPT_yogurt_calories_per_ounce_l1427_142751

variable (calories_strawberries_per_unit : ℕ)
variable (calories_yogurt_total : ℕ)
variable (calories_total : ℕ)
variable (strawberries_count : ℕ)
variable (yogurt_ounces_count : ℕ)

theorem yogurt_calories_per_ounce (h1: strawberries_count = 12)
                                   (h2: yogurt_ounces_count = 6)
                                   (h3: calories_strawberries_per_unit = 4)
                                   (h4: calories_total = 150)
                                   (h5: calories_yogurt_total = calories_total - strawberries_count * calories_strawberries_per_unit):
                                   calories_yogurt_total / yogurt_ounces_count = 17 :=
by
  -- We conjecture that this is correct based on given conditions.
  sorry

end NUMINAMATH_GPT_yogurt_calories_per_ounce_l1427_142751


namespace NUMINAMATH_GPT_carrie_total_spend_l1427_142715

def cost_per_tshirt : ℝ := 9.15
def number_of_tshirts : ℝ := 22

theorem carrie_total_spend : (cost_per_tshirt * number_of_tshirts) = 201.30 := by 
  sorry

end NUMINAMATH_GPT_carrie_total_spend_l1427_142715


namespace NUMINAMATH_GPT_positive_difference_is_9107_03_l1427_142756

noncomputable def Cedric_balance : ℝ :=
  15000 * (1 + 0.06) ^ 20

noncomputable def Daniel_balance : ℝ :=
  15000 * (1 + 20 * 0.08)

noncomputable def Elaine_balance : ℝ :=
  15000 * (1 + 0.055 / 2) ^ 40

-- Positive difference between highest and lowest balances.
noncomputable def positive_difference : ℝ :=
  let highest := max Cedric_balance (max Daniel_balance Elaine_balance)
  let lowest := min Cedric_balance (min Daniel_balance Elaine_balance)
  highest - lowest

theorem positive_difference_is_9107_03 :
  positive_difference = 9107.03 := by
  sorry

end NUMINAMATH_GPT_positive_difference_is_9107_03_l1427_142756


namespace NUMINAMATH_GPT_solve_for_x_l1427_142724

theorem solve_for_x (x : ℝ) (h : 4 * x - 5 = 3) : x = 2 :=
by sorry

end NUMINAMATH_GPT_solve_for_x_l1427_142724
