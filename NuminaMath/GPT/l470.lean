import Mathlib

namespace distance_between_A_and_B_l470_47080

-- Definitions and conditions
variables {A B C : Type}    -- Locations
variables {v1 v2 : ℕ}       -- Speeds of person A and person B
variables {distanceAB : ℕ}  -- Distance we want to find

noncomputable def first_meet_condition (v1 v2 : ℕ) : Prop :=
  ∃ t : ℕ, (v1 * t - 108 = v2 * t - 100)

noncomputable def second_meet_condition (v1 v2 distanceAB : ℕ) : Prop :=
  distanceAB = 3750

-- Theorem statement
theorem distance_between_A_and_B (v1 v2 distanceAB : ℕ) :
  first_meet_condition v1 v2 → second_meet_condition v1 v2 distanceAB →
  distanceAB = 3750 :=
by
  intros _ _ 
  sorry

end distance_between_A_and_B_l470_47080


namespace tom_sold_4_books_l470_47091

-- Definitions based on conditions from the problem
def initial_books : ℕ := 5
def new_books : ℕ := 38
def final_books : ℕ := 39

-- The number of books Tom sold
def books_sold (S : ℕ) : Prop := initial_books - S + new_books = final_books

-- Our goal is to prove that Tom sold 4 books
theorem tom_sold_4_books : books_sold 4 :=
  by
    -- Implicitly here would be the proof, but we use sorry to skip it
    sorry

end tom_sold_4_books_l470_47091


namespace value_of_power_l470_47068

theorem value_of_power (a b : ℝ) (h : |a - 1| + (b + 2)^2 = 0) : (a + b) ^ 2014 = 1 :=
by
  sorry

end value_of_power_l470_47068


namespace pattern_D_cannot_form_tetrahedron_l470_47094

theorem pattern_D_cannot_form_tetrahedron :
  (¬ ∃ (f : ℝ × ℝ → ℝ × ℝ),
      f (0, 0) = (1, 1) ∧ f (1, 0) = (1, -1) ∧ f (2, 0) = (-1, 1) ∧ f (3, 0) = (-1, -1)) :=
by
  -- proof will go here
  sorry

end pattern_D_cannot_form_tetrahedron_l470_47094


namespace fuel_consumption_new_model_l470_47084

variable (d_old : ℝ) (d_new : ℝ) (c_old : ℝ) (c_new : ℝ)

theorem fuel_consumption_new_model :
  (d_new = d_old + 4.4) →
  (c_new = c_old - 2) →
  (c_old = 100 / d_old) →
  d_old = 12.79 →
  c_new = 5.82 :=
by
  intro h1 h2 h3 h4
  sorry

end fuel_consumption_new_model_l470_47084


namespace sum_of_first_ten_terms_l470_47066

theorem sum_of_first_ten_terms (S : ℕ → ℕ) (h : ∀ n, S n = n^2 - 4 * n + 1) : S 10 = 61 :=
by
  sorry

end sum_of_first_ten_terms_l470_47066


namespace can_be_divided_into_two_triangles_l470_47034

-- Definitions and properties of geometrical shapes
def is_triangle (sides : ℕ) (vertices : ℕ) : Prop :=
  sides = 3 ∧ vertices = 3

def is_pentagon (sides : ℕ) (vertices : ℕ) : Prop :=
  sides = 5 ∧ vertices = 5

def is_hexagon (sides : ℕ) (vertices : ℕ) : Prop :=
  sides = 6 ∧ vertices = 6

def is_heptagon (sides : ℕ) (vertices : ℕ) : Prop :=
  sides = 7 ∧ vertices = 7

-- The theorem we need to prove
theorem can_be_divided_into_two_triangles :
  ∀ sides vertices,
  (is_pentagon sides vertices → is_triangle sides vertices ∧ is_triangle sides vertices) ∧
  (is_hexagon sides vertices → is_triangle sides vertices ∧ is_triangle sides vertices) ∧
  (is_heptagon sides vertices → ¬ (is_triangle sides vertices ∧ is_triangle sides vertices)) :=
by sorry

end can_be_divided_into_two_triangles_l470_47034


namespace equation_one_solution_equation_two_solution_l470_47061

theorem equation_one_solution (x : ℝ) (h : 2 * (2 - x) - 5 * (2 - x) = 9) : x = 5 :=
sorry

theorem equation_two_solution (x : ℝ) (h : x / 3 - (3 * x - 1) / 6 = 1) : x = -5 :=
sorry

end equation_one_solution_equation_two_solution_l470_47061


namespace value_taken_away_l470_47043

theorem value_taken_away (n x : ℕ) (h1 : n = 4) (h2 : 2 * n + 20 = 8 * n - x) : x = 4 :=
by
  sorry

end value_taken_away_l470_47043


namespace planes_parallel_l470_47069

variables (α β : Type)
variables (n : ℝ → ℝ → ℝ → Prop) (u v : ℝ × ℝ × ℝ)

-- Conditions: 
def normal_vector_plane_alpha (u : ℝ × ℝ × ℝ) := u = (1, 2, -1)
def normal_vector_plane_beta (v : ℝ × ℝ × ℝ) := v = (-3, -6, 3)

-- Proof Problem: Prove that alpha is parallel to beta
theorem planes_parallel (h1 : normal_vector_plane_alpha u)
                        (h2 : normal_vector_plane_beta v) :
  v = -3 • u :=
by sorry

end planes_parallel_l470_47069


namespace william_ends_with_18_tickets_l470_47059

-- Define the initial number of tickets
def initialTickets : ℕ := 15

-- Define the tickets bought
def ticketsBought : ℕ := 3

-- Prove the total number of tickets William ends with
theorem william_ends_with_18_tickets : initialTickets + ticketsBought = 18 := by
  sorry

end william_ends_with_18_tickets_l470_47059


namespace solve_quadratic_inequality_l470_47082

-- To express that a real number x is in the interval (0, 2)
def in_interval (x : ℝ) : Prop := 0 < x ∧ x < 2

theorem solve_quadratic_inequality :
  { x : ℝ | x^2 < 2 * x } = { x : ℝ | in_interval x } :=
by
  sorry

end solve_quadratic_inequality_l470_47082


namespace inequality_solution_l470_47021

theorem inequality_solution 
  (a x : ℝ) : 
  (a = 2 ∨ a = -2 → x > 1 / 4) ∧ 
  (a > 2 → x > 1 / (a + 2) ∨ x < 1 / (2 - a)) ∧ 
  (a < -2 → x < 1 / (a + 2) ∨ x > 1 / (2 - a)) ∧ 
  (-2 < a ∧ a < 2 → 1 / (a + 2) < x ∧ x < 1 / (2 - a)) 
  :=
by
  sorry

end inequality_solution_l470_47021


namespace problem1_l470_47036

theorem problem1 (x y : ℝ) (h : |x + 1| + (2 * x - y)^2 = 0) : x^2 - y = 3 :=
sorry

end problem1_l470_47036


namespace bagel_spending_l470_47012

theorem bagel_spending (B D : ℝ) (h1 : D = 0.5 * B) (h2 : B = D + 15) : B + D = 45 := by
  sorry

end bagel_spending_l470_47012


namespace sum_of_numbers_l470_47071

theorem sum_of_numbers : 1234 + 2341 + 3412 + 4123 = 11110 := by
  sorry

end sum_of_numbers_l470_47071


namespace diana_statues_painted_l470_47051

theorem diana_statues_painted :
  let paint_remaining := (1 : ℚ) / 2
  let paint_per_statue := (1 : ℚ) / 4
  (paint_remaining / paint_per_statue) = 2 :=
by
  sorry

end diana_statues_painted_l470_47051


namespace minimum_distance_from_lattice_point_to_line_l470_47083

theorem minimum_distance_from_lattice_point_to_line :
  let distance (x y : ℤ) := |25 * x - 15 * y + 12| / (5 * Real.sqrt 34)
  ∃ (x y : ℤ), distance x y = Real.sqrt 34 / 85 :=
sorry

end minimum_distance_from_lattice_point_to_line_l470_47083


namespace arccos_neg_one_eq_pi_l470_47085

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = Real.pi :=
by
  sorry

end arccos_neg_one_eq_pi_l470_47085


namespace circle_center_radius_l470_47075

theorem circle_center_radius
    (x y : ℝ)
    (eq_circle : (x - 2)^2 + y^2 = 4) :
    (2, 0) = (2, 0) ∧ 2 = 2 :=
by
  sorry

end circle_center_radius_l470_47075


namespace find_side_length_a_l470_47047

noncomputable def length_of_a (A B : ℝ) (b : ℝ) : ℝ :=
  b * Real.sin A / Real.sin B

theorem find_side_length_a :
  ∀ (a b c : ℝ) (A B C : ℝ),
  A = Real.pi / 3 → B = Real.pi / 4 → b = Real.sqrt 6 →
  a = length_of_a A B b →
  a = 3 :=
by
  intros a b c A B C hA hB hb ha
  rw [hA, hB, hb] at ha
  sorry

end find_side_length_a_l470_47047


namespace smallest_integer_k_condition_l470_47001

theorem smallest_integer_k_condition :
  ∃ k : ℤ, k > 1 ∧ k % 12 = 1 ∧ k % 5 = 1 ∧ k % 3 = 1 ∧ k = 61 :=
by
  sorry

end smallest_integer_k_condition_l470_47001


namespace algebraic_comparison_l470_47027

theorem algebraic_comparison (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (a^2 / b + b^2 / a ≥ a + b) :=
by
  sorry

end algebraic_comparison_l470_47027


namespace tanx_eq_2_sin2cos2_tanx_eq_2_cos_sin_ratio_l470_47020

theorem tanx_eq_2_sin2cos2 (x : ℝ) (h : Real.tan x = 2) : 
  (2 / 3) * (Real.sin x) ^ 2 + (1 / 4) * (Real.cos x) ^ 2 = 7 / 12 := 
by 
  sorry

theorem tanx_eq_2_cos_sin_ratio (x : ℝ) (h : Real.tan x = 2) : 
  (Real.cos x + Real.sin x) / (Real.cos x - Real.sin x) = -3 := 
by 
  sorry

end tanx_eq_2_sin2cos2_tanx_eq_2_cos_sin_ratio_l470_47020


namespace cost_per_kg_mixture_l470_47067

variables (C1 C2 R Cm : ℝ)

-- Statement of the proof problem
theorem cost_per_kg_mixture :
  C1 = 6 → C2 = 8.75 → R = 5 / 6 → Cm = C1 * R + C2 * (1 - R) → Cm = 6.458333333333333 :=
by intros hC1 hC2 hR hCm; sorry

end cost_per_kg_mixture_l470_47067


namespace geometric_sequence_term_l470_47078

/-
Prove that the 303rd term in a geometric sequence with the first term a1 = 5 and the second term a2 = -10 is 5 * 2^302.
-/

theorem geometric_sequence_term :
  let a1 := 5
  let a2 := -10
  let r := a2 / a1
  let n := 303
  let a_n := a1 * r^(n-1)
  a_n = 5 * 2^302 :=
by
  let a1 := 5
  let a2 := -10
  let r := a2 / a1
  let n := 303
  have h1 : a1 * r^(n-1) = 5 * 2^302 := sorry
  exact h1

end geometric_sequence_term_l470_47078


namespace symmetric_pattern_count_l470_47052

noncomputable def number_of_symmetric_patterns (n : ℕ) : ℕ :=
  let regions := 12
  let total_patterns := 2^regions
  total_patterns - 2

theorem symmetric_pattern_count : number_of_symmetric_patterns 8 = 4094 :=
by
  sorry

end symmetric_pattern_count_l470_47052


namespace cost_of_carpeting_l470_47092

noncomputable def cost_per_meter_in_paise (cost : ℝ) (length_in_meters : ℝ) : ℝ :=
  cost * 100 / length_in_meters

theorem cost_of_carpeting (room_length room_breadth carpet_width_m cost_total : ℝ) (h1 : room_length = 15) 
  (h2 : room_breadth = 6) (h3 : carpet_width_m = 0.75) (h4 : cost_total = 36) :
  cost_per_meter_in_paise cost_total (room_length * room_breadth / carpet_width_m) = 30 :=
by
  sorry

end cost_of_carpeting_l470_47092


namespace yao_ming_mcgrady_probability_l470_47076

theorem yao_ming_mcgrady_probability
        (p : ℝ) (q : ℝ)
        (h1 : p = 0.8)
        (h2 : q = 0.7) :
        (2 * p * (1 - p)) * (2 * q * (1 - q)) = 0.1344 := 
by
  sorry

end yao_ming_mcgrady_probability_l470_47076


namespace compute_expression_l470_47039

-- Define the conditions
variables (ω : ℂ) (hω_nonreal : ω^3 = 1) (hω_eq : ω^2 + ω + 1 = 0)

-- State the theorem to be proved
theorem compute_expression (ω : ℂ) (hω_nonreal : ω^3 = 1) (hω_eq : ω^2 + ω + 1 = 0) :
  (1 - ω + ω^2)^6 + (1 + ω - ω^2)^6 = 128 := 
sorry

end compute_expression_l470_47039


namespace find_10th_integer_l470_47074

-- Defining the conditions
def avg_20_consecutive_integers (avg : ℝ) : Prop :=
  avg = 23.65

def consecutive_integer_sequence (n : ℤ) (a : ℤ) : Prop :=
  a = n + 9

-- The main theorem statement
theorem find_10th_integer (n : ℤ) (avg : ℝ) (h_avg : avg_20_consecutive_integers avg) (h_seq : consecutive_integer_sequence n 23) :
  n = 14 :=
sorry

end find_10th_integer_l470_47074


namespace nearest_integer_to_a_plus_b_l470_47044

theorem nearest_integer_to_a_plus_b
  (a b : ℝ)
  (h1 : |a| + b = 5)
  (h2 : |a| * b + a^3 = -8) :
  abs (a + b - 3) ≤ 0.5 :=
sorry

end nearest_integer_to_a_plus_b_l470_47044


namespace Vlad_score_l470_47064

-- Defining the initial conditions of the problem
def total_rounds : ℕ := 30
def points_per_win : ℕ := 5
def total_points : ℕ := total_rounds * points_per_win

-- Taro's score as described in the problem
def Taros_score := (3 * total_points / 5) - 4

-- Prove that Vlad's score is 64 points
theorem Vlad_score : total_points - Taros_score = 64 := by
  sorry

end Vlad_score_l470_47064


namespace pq_implies_q_l470_47041

theorem pq_implies_q (p q : Prop) (h₁ : p ∨ q) (h₂ : ¬p) : q :=
by
  sorry

end pq_implies_q_l470_47041


namespace annual_interest_rate_l470_47049

theorem annual_interest_rate (P A : ℝ) (n : ℕ) (t r : ℝ) 
  (hP : P = 5000) 
  (hA : A = 5202) 
  (hn : n = 4) 
  (ht : t = 1 / 2)
  (compound_interest : A = P * (1 + r / n)^ (n * t)) : 
  r = 0.080392 :=
by
  sorry

end annual_interest_rate_l470_47049


namespace journey_time_l470_47089

theorem journey_time 
  (d1 d2 T : ℝ)
  (h1 : d1 / 30 + (150 - d1) / 10 = T)
  (h2 : d1 / 30 + d2 / 30 + (150 - (d1 - d2)) / 30 = T)
  (h3 : (d1 - d2) / 10 + (150 - (d1 - d2)) / 30 = T) :
  T = 5 := 
sorry

end journey_time_l470_47089


namespace coloring_scheme_formula_l470_47081

noncomputable def number_of_coloring_schemes (m n : ℕ) : ℕ :=
  if h : (m ≥ 2) ∧ (n ≥ 2) then
    m * ((-1 : ℤ)^n * (m - 2 : ℤ)).natAbs + (m - 2)^n
  else 0

-- Formal statement verifying the formula for coloring schemes
theorem coloring_scheme_formula (m n : ℕ) (h1 : m ≥ 2) (h2 : n ≥ 2) :
  number_of_coloring_schemes m n = m * ((-1 : ℤ)^n * (m - 2 : ℤ)).natAbs + (m - 2)^n :=
by sorry

end coloring_scheme_formula_l470_47081


namespace cos_alpha_plus_two_pi_over_three_l470_47062

theorem cos_alpha_plus_two_pi_over_three (α : ℝ) (h : Real.sin (α + π / 6) = 1 / 3) :
  Real.cos (α + 2 * π / 3) = -1 / 3 :=
by
  sorry

end cos_alpha_plus_two_pi_over_three_l470_47062


namespace steve_correct_operations_l470_47090

theorem steve_correct_operations (x : ℕ) (h1 : x / 8 - 20 = 12) : ((x * 8) + 20) = 2068 :=
by
  sorry

end steve_correct_operations_l470_47090


namespace pairs_divisible_by_three_l470_47024

theorem pairs_divisible_by_three (P T : ℕ) (h : 5 * P = 3 * T) : ∃ k : ℕ, P = 3 * k := 
sorry

end pairs_divisible_by_three_l470_47024


namespace circle_intersection_exists_l470_47005

theorem circle_intersection_exists (a b : ℝ) :
  ∃ (m n : ℤ), (m - a)^2 + (n - b)^2 ≤ (1 / 14)^2 →
  ∀ x y, (x - a)^2 + (y - b)^2 = 100^2 :=
sorry

end circle_intersection_exists_l470_47005


namespace number_of_selected_in_interval_l470_47079

noncomputable def systematic_sampling_group := (420: ℕ)
noncomputable def selected_people := (21: ℕ)
noncomputable def interval_start := (241: ℕ)
noncomputable def interval_end := (360: ℕ)
noncomputable def sampling_interval := systematic_sampling_group / selected_people
noncomputable def interval_length := interval_end - interval_start + 1

theorem number_of_selected_in_interval :
  interval_length / sampling_interval = 6 :=
by
  -- Placeholder for the proof
  sorry

end number_of_selected_in_interval_l470_47079


namespace yoongi_correct_calculation_l470_47006

theorem yoongi_correct_calculation (x : ℕ) (h : x + 9 = 30) : x - 7 = 14 :=
sorry

end yoongi_correct_calculation_l470_47006


namespace provider_assignment_ways_l470_47002

theorem provider_assignment_ways (total_providers : ℕ) (children : ℕ) (h1 : total_providers = 15) (h2 : children = 4) : 
  (Finset.range total_providers).card.factorial / (Finset.range (total_providers - children)).card.factorial = 32760 :=
by
  rw [h1, h2]
  norm_num
  sorry

end provider_assignment_ways_l470_47002


namespace wall_volume_is_128512_l470_47086

noncomputable def wall_volume (width : ℝ) (height : ℝ) (length : ℝ) : ℝ :=
  width * height * length

theorem wall_volume_is_128512 : 
  ∀ (w : ℝ) (h : ℝ) (l : ℝ), 
  h = 6 * w ∧ l = 7 * h ∧ w = 8 → 
  wall_volume w h l = 128512 := 
by
  sorry

end wall_volume_is_128512_l470_47086


namespace tan_sum_pi_over_4_x_l470_47072

theorem tan_sum_pi_over_4_x (x : ℝ) (h1 : x > -π/2 ∧ x < 0) (h2 : Real.cos x = 4/5) :
  Real.tan (π/4 + x) = 1/7 :=
by
  sorry

end tan_sum_pi_over_4_x_l470_47072


namespace scientific_notation_of_11090000_l470_47037

theorem scientific_notation_of_11090000 :
  ∃ (x : ℝ) (n : ℤ), 11090000 = x * 10^n ∧ x = 1.109 ∧ n = 7 :=
by
  -- skip the proof
  sorry

end scientific_notation_of_11090000_l470_47037


namespace gcd_square_product_l470_47098

theorem gcd_square_product (x y z : ℕ) (h : 1 / (x : ℝ) - 1 / (y : ℝ) = 1 / (z : ℝ)) : 
    ∃ n : ℕ, gcd x (gcd y z) * x * y * z = n * n := 
sorry

end gcd_square_product_l470_47098


namespace min_b_over_a_l470_47017

theorem min_b_over_a (a b : ℝ) (h : ∀ x : ℝ, (Real.log a + b) * Real.exp x - a^2 * Real.exp x ≥ 0) : b / a ≥ 1 := by
  sorry

end min_b_over_a_l470_47017


namespace back_seat_tickets_sold_l470_47011

def total_tickets : ℕ := 20000
def main_seat_price : ℕ := 55
def back_seat_price : ℕ := 45
def total_revenue : ℕ := 955000

theorem back_seat_tickets_sold :
  ∃ (M B : ℕ), 
    M + B = total_tickets ∧ 
    main_seat_price * M + back_seat_price * B = total_revenue ∧ 
    B = 14500 :=
by
  sorry

end back_seat_tickets_sold_l470_47011


namespace additional_men_joined_l470_47060

theorem additional_men_joined (men_initial : ℕ) (days_initial : ℕ)
  (days_new : ℕ) (additional_men : ℕ) :
  men_initial = 600 →
  days_initial = 20 →
  days_new = 15 →
  (men_initial * days_initial) = ((men_initial + additional_men) * days_new) →
  additional_men = 200 := 
by
  intros h1 h2 h3 h4
  sorry

end additional_men_joined_l470_47060


namespace derivative_at_neg_one_l470_47035

variable (a b : ℝ)

-- Define the function f(x)
def f (x : ℝ) : ℝ := a * x^4 + b * x^2 + 6

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 4 * a * x^3 + 2 * b * x

-- Given condition f'(1) = 2
axiom h : f' a b 1 = 2

-- Statement to prove f'(-1) = -2
theorem derivative_at_neg_one : f' a b (-1) = -2 :=
by 
  sorry

end derivative_at_neg_one_l470_47035


namespace sequence_term_l470_47070

noncomputable def S (n : ℕ) : ℤ := n^2 - 3 * n

theorem sequence_term (n : ℕ) (h : n ≥ 1) : 
  ∃ a : ℕ → ℤ, a n = 2 * n - 4 := 
  sorry

end sequence_term_l470_47070


namespace maximum_value_of_A_l470_47057

theorem maximum_value_of_A (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) :
    (a^4 + b^4 + c^4) / ((a + b + c)^4 - 80 * (a * b * c)^(4 / 3)) ≤ 3 :=
sorry

end maximum_value_of_A_l470_47057


namespace carpool_commute_distance_l470_47099

theorem carpool_commute_distance :
  (∀ (D : ℕ),
    4 * 5 * ((2 * D : ℝ) / 30) * 2.50 = 5 * 14 →
    D = 21) :=
by
  intro D
  intro h
  sorry

end carpool_commute_distance_l470_47099


namespace unique_k_linear_equation_l470_47053

theorem unique_k_linear_equation :
  (∀ x y k : ℝ, (2 : ℝ) * x^|k| + (k - 1) * y = 3 → (|k| = 1 ∧ k ≠ 1) → k = -1) :=
by
  sorry

end unique_k_linear_equation_l470_47053


namespace fish_count_total_l470_47007

def Jerk_Tuna_fish : ℕ := 144
def Tall_Tuna_fish : ℕ := 2 * Jerk_Tuna_fish
def Total_fish_together : ℕ := Jerk_Tuna_fish + Tall_Tuna_fish

theorem fish_count_total :
  Total_fish_together = 432 :=
by
  sorry

end fish_count_total_l470_47007


namespace valid_x_for_expression_l470_47013

theorem valid_x_for_expression :
  (∃ x : ℝ, x = 8 ∧ (10 - x ≥ 0) ∧ (x - 4 ≠ 0)) ↔ (∃ x : ℝ, x = 8) :=
by
  sorry

end valid_x_for_expression_l470_47013


namespace ratio_of_areas_l470_47014

theorem ratio_of_areas (AB BC O : ℝ) (h_diameter : AB = 4) (h_BC : BC = 3)
  (ABD DBE ABDeqDBE : Prop) (x y : ℝ) 
  (h_area_ABCD : x = 7 * y) :
  (x / y) = 7 :=
by
  sorry

end ratio_of_areas_l470_47014


namespace base_rate_first_company_proof_l470_47042

noncomputable def base_rate_first_company : ℝ := 8.00
def charge_per_minute_first_company : ℝ := 0.25
def base_rate_second_company : ℝ := 12.00
def charge_per_minute_second_company : ℝ := 0.20
def minutes : ℕ := 80

theorem base_rate_first_company_proof :
  base_rate_first_company = 8.00 :=
sorry

end base_rate_first_company_proof_l470_47042


namespace exists_square_no_visible_points_l470_47033

-- Define visibility from the origin
def visible_from_origin (x y : ℤ) : Prop :=
  Int.gcd x y = 1

-- Main theorem statement
theorem exists_square_no_visible_points (n : ℕ) (hn : 0 < n) :
  ∃ (a b : ℤ), 
    (∀ (x y : ℤ), a ≤ x ∧ x ≤ a + n ∧ b ≤ y ∧ y ≤ b + n ∧ (x ≠ 0 ∨ y ≠ 0) → ¬visible_from_origin x y) :=
sorry

end exists_square_no_visible_points_l470_47033


namespace find_x_l470_47022

theorem find_x (x : ℚ) (h : (3 * x - 7) / 4 = 15) : x = 67 / 3 :=
sorry

end find_x_l470_47022


namespace sandra_tickets_relation_l470_47009

def volleyball_game : Prop :=
  ∃ (tickets_total tickets_left tickets_jude tickets_andrea tickets_sandra : ℕ),
    tickets_total = 100 ∧
    tickets_left = 40 ∧
    tickets_jude = 16 ∧
    tickets_andrea = 2 * tickets_jude ∧
    tickets_total - tickets_left = tickets_jude + tickets_andrea + tickets_sandra ∧
    tickets_sandra = tickets_jude - 4

theorem sandra_tickets_relation : volleyball_game :=
  sorry

end sandra_tickets_relation_l470_47009


namespace product_of_possible_values_l470_47065

theorem product_of_possible_values (x : ℝ) (h : (x + 3) * (x - 4) = 18) : ∃ a b, x = a ∨ x = b ∧ a * b = -30 :=
by 
  sorry

end product_of_possible_values_l470_47065


namespace crushing_load_example_l470_47038

noncomputable def crushing_load (T H : ℝ) : ℝ :=
  (30 * T^5) / H^3

theorem crushing_load_example : crushing_load 5 10 = 93.75 := by
  sorry

end crushing_load_example_l470_47038


namespace arith_seq_sum_correct_l470_47093

-- Define the arithmetic sequence given the first term and common difference
def arith_seq (a₁ d : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * d

-- Define the sum of the first n terms of an arithmetic sequence
def arith_seq_sum (a₁ d : ℤ) (n : ℕ) : ℤ :=
  (n * (2 * a₁ + (n - 1) * d)) / 2

-- Given Problem Conditions
def a₁ := -5
def d := 3
def n := 20

-- Theorem: Sum of the first 20 terms of the arithmetic sequence is 470
theorem arith_seq_sum_correct : arith_seq_sum a₁ d n = 470 :=
  sorry

end arith_seq_sum_correct_l470_47093


namespace sufficient_but_not_necessary_condition_l470_47073

-- Define the condition
variable (a : ℝ)

-- Theorem statement: $a > 0$ is a sufficient but not necessary condition for $a^2 > 0$
theorem sufficient_but_not_necessary_condition : 
  (a > 0 → a^2 > 0) ∧ (¬ (a > 0) → a^2 > 0) :=
  by
    sorry

end sufficient_but_not_necessary_condition_l470_47073


namespace percent_decrease_l470_47097

theorem percent_decrease (original_price sale_price : ℝ) (h₀ : original_price = 100) (h₁ : sale_price = 30) :
  (original_price - sale_price) / original_price * 100 = 70 :=
by
  rw [h₀, h₁]
  norm_num

end percent_decrease_l470_47097


namespace debate_team_boys_l470_47096

theorem debate_team_boys (total_groups : ℕ) (members_per_group : ℕ) (num_girls : ℕ) (total_members : ℕ) :
  total_groups = 8 →
  members_per_group = 4 →
  num_girls = 4 →
  total_members = total_groups * members_per_group →
  total_members - num_girls = 28 :=
by
  sorry

end debate_team_boys_l470_47096


namespace find_m_l470_47025

variables {m : ℝ}
def vec_a : ℝ × ℝ := (-2, 3)
def vec_b (m : ℝ) : ℝ × ℝ := (3, m)
def perpendicular (a b : ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 = 0

theorem find_m (m : ℝ) (h : perpendicular vec_a (vec_b m)) : m = 2 :=
by
  sorry

end find_m_l470_47025


namespace work_completion_time_l470_47004

theorem work_completion_time (B_rate A_rate Combined_rate : ℝ) (B_time : ℝ) :
  (B_rate = 1 / 60) →
  (A_rate = 4 * B_rate) →
  (Combined_rate = A_rate + B_rate) →
  (B_time = 1 / Combined_rate) →
  B_time = 12 :=
by sorry

end work_completion_time_l470_47004


namespace A_min_votes_for_victory_l470_47088

theorem A_min_votes_for_victory:
  ∀ (initial_votes_A initial_votes_B initial_votes_C total_votes remaining_votes min_votes_A: ℕ),
  initial_votes_A = 350 →
  initial_votes_B = 370 →
  initial_votes_C = 280 →
  total_votes = 1500 →
  remaining_votes = 500 →
  min_votes_A = 261 →
  initial_votes_A + min_votes_A > initial_votes_B + (remaining_votes - min_votes_A) :=
by
  intros _ _ _ _ _ _
  sorry

end A_min_votes_for_victory_l470_47088


namespace length_of_new_section_l470_47040

-- Definitions from the conditions
def area : ℕ := 35
def width : ℕ := 7

-- The problem statement
theorem length_of_new_section (h : area = 35 ∧ width = 7) : 35 / 7 = 5 :=
by
  -- We'll provide the proof later
  sorry

end length_of_new_section_l470_47040


namespace gcd_f_l470_47008

def f (x: ℤ) : ℤ := x^2 - x + 2023

theorem gcd_f (x y : ℤ) (hx : x = 105) (hy : y = 106) : Int.gcd (f x) (f y) = 7 := by
  sorry

end gcd_f_l470_47008


namespace boys_bound_l470_47045

open Nat

noncomputable def num_students := 1650
noncomputable def num_rows := 22
noncomputable def num_cols := 75
noncomputable def max_pairs_same_sex := 11

-- Assume we have a function that gives the number of boys.
axiom number_of_boys : ℕ
axiom col_pairs_property : ∀ (c1 c2 : ℕ), ∀ (r : ℕ), c1 ≠ c2 → r ≤ num_rows → 
  (number_of_boys ≤ max_pairs_same_sex)

theorem boys_bound : number_of_boys ≤ 920 :=
sorry

end boys_bound_l470_47045


namespace number_of_stones_l470_47087

theorem number_of_stones (hall_length_m : ℕ) (hall_breadth_m : ℕ)
  (stone_length_dm : ℕ) (stone_breadth_dm : ℕ)
  (hall_length_dm_eq : hall_length_m * 10 = 360)
  (hall_breadth_dm_eq : hall_breadth_m * 10 = 150)
  (stone_length_eq : stone_length_dm = 6)
  (stone_breadth_eq : stone_breadth_dm = 5) :
  ((hall_length_m * 10) * (hall_breadth_m * 10)) / (stone_length_dm * stone_breadth_dm) = 1800 :=
by
  sorry

end number_of_stones_l470_47087


namespace monotone_on_interval_and_extreme_values_l470_47054

noncomputable def f (x : ℝ) : ℝ := x + 4 / x

theorem monotone_on_interval_and_extreme_values :
  (∀ x1 x2 : ℝ, (1 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 2) → f x1 > f x2) ∧ (f 1 = 5 ∧ f 2 = 4) := 
by
  sorry

end monotone_on_interval_and_extreme_values_l470_47054


namespace minimum_sum_am_gm_l470_47015

theorem minimum_sum_am_gm (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (3 * b)) + (b / (6 * c)) + (c / (9 * a)) ≥ (1 / 2) :=
sorry

end minimum_sum_am_gm_l470_47015


namespace inequality_holds_for_all_real_l470_47046

theorem inequality_holds_for_all_real (k : ℝ) :
  (∀ x : ℝ, k * x ^ 2 - 6 * k * x + k + 8 ≥ 0) ↔ (0 ≤ k ∧ k ≤ 1) :=
sorry

end inequality_holds_for_all_real_l470_47046


namespace count_solutions_l470_47056

noncomputable def num_solutions : ℕ :=
  let eq1 (x y : ℝ) := 2 * x + 5 * y = 10
  let eq2 (x y : ℝ) := abs (abs (x + 1) - abs (y - 1)) = 1
  sorry

theorem count_solutions : num_solutions = 2 := by
  sorry

end count_solutions_l470_47056


namespace cube_volume_proof_l470_47030

-- Define the conditions
def len_inch : ℕ := 48
def width_inch : ℕ := 72
def total_surface_area_inch : ℕ := len_inch * width_inch
def num_faces : ℕ := 6
def area_one_face_inch : ℕ := total_surface_area_inch / num_faces
def inches_to_feet (length_in_inches : ℕ) : ℕ := length_in_inches / 12

-- Define the key elements of the proof problem
def side_length_inch : ℕ := Int.natAbs (Nat.sqrt area_one_face_inch)
def side_length_ft : ℕ := inches_to_feet side_length_inch
def volume_ft3 : ℕ := side_length_ft ^ 3

-- State the proof problem
theorem cube_volume_proof : volume_ft3 = 8 := by
  -- The proof would be implemented here
  sorry

end cube_volume_proof_l470_47030


namespace inequality1_inequality2_l470_47050

variables (Γ B P : ℕ)

def convex_polyhedron : Prop :=
  Γ - B + P = 2

theorem inequality1 (h : convex_polyhedron Γ B P) : 
  3 * Γ ≥ 6 + P :=
sorry

theorem inequality2 (h : convex_polyhedron Γ B P) : 
  3 * B ≥ 6 + P :=
sorry

end inequality1_inequality2_l470_47050


namespace f_irreducible_l470_47028

noncomputable def f (n : ℕ) (x : ℤ) : ℤ := x^n + 5 * x^(n-1) + 3

theorem f_irreducible (n : ℕ) (hn : n > 1) : Irreducible (f n) :=
sorry

end f_irreducible_l470_47028


namespace average_speed_correct_l470_47019

variable (t1 t2 : ℝ) -- time components in hours
variable (v1 v2 : ℝ) -- speed components in km/h

-- conditions
def time1 := 20 / 60 -- 20 minutes converted to hours
def time2 := 40 / 60 -- 40 minutes converted to hours
def speed1 := 60 -- speed in km/h for the first segment
def speed2 := 90 -- speed in km/h for the second segment

-- total distance traveled
def distance1 := speed1 * time1
def distance2 := speed2 * time2
def total_distance := distance1 + distance2

-- total time taken
def total_time := time1 + time2

-- average speed
def average_speed := total_distance / total_time

-- proof statement
theorem average_speed_correct : average_speed = 80 := by
  sorry

end average_speed_correct_l470_47019


namespace map_scale_l470_47055

theorem map_scale (cm12_km90 : 12 * (1 / 90) = 1) : 20 * (90 / 12) = 150 :=
by
  sorry

end map_scale_l470_47055


namespace total_cost_of_lollipops_l470_47063

/-- Given Sarah bought 12 lollipops and shared one-quarter of them, 
    and Julie reimbursed Sarah 75 cents for the shared lollipops,
    Prove that the total cost of the lollipops in dollars is $3. --/
theorem total_cost_of_lollipops 
(Sarah_lollipops : ℕ) 
(shared_fraction : ℚ) 
(Julie_paid : ℚ) 
(total_lollipops_cost : ℚ)
(h1 : Sarah_lollipops = 12) 
(h2 : shared_fraction = 1/4) 
(h3 : Julie_paid = 75 / 100) 
(h4 : total_lollipops_cost = 
        ((Julie_paid / (Sarah_lollipops * shared_fraction)) * Sarah_lollipops / 100)) :
total_lollipops_cost = 3 := 
sorry

end total_cost_of_lollipops_l470_47063


namespace triangles_combined_area_is_96_l470_47077

noncomputable def combined_area_of_triangles : Prop :=
  let length_rectangle : ℝ := 6
  let width_rectangle : ℝ := 4
  let area_rectangle : ℝ := length_rectangle * width_rectangle
  let ratio_rectangle_to_first_triangle : ℝ := 2 / 5
  let area_first_triangle : ℝ := (5 / 2) * area_rectangle
  let x : ℝ := area_first_triangle / 5
  let base_second_triangle : ℝ := 8
  let height_second_triangle : ℝ := 9  -- calculated height based on the area ratio
  let area_second_triangle : ℝ := (base_second_triangle * height_second_triangle) / 2
  let combined_area : ℝ := area_first_triangle + area_second_triangle
  combined_area = 96

theorem triangles_combined_area_is_96 : combined_area_of_triangles := by
  sorry

end triangles_combined_area_is_96_l470_47077


namespace discount_percentage_l470_47010

theorem discount_percentage (number_of_tshirts : ℕ) (cost_per_tshirt amount_paid : ℝ)
  (h1 : number_of_tshirts = 6)
  (h2 : cost_per_tshirt = 20)
  (h3 : amount_paid = 60) : 
  ((number_of_tshirts * cost_per_tshirt - amount_paid) / (number_of_tshirts * cost_per_tshirt) * 100) = 50 := by
  -- The proof will go here
  sorry

end discount_percentage_l470_47010


namespace base7_digits_l470_47032

theorem base7_digits (D E F : ℕ) (h1 : D ≠ 0) (h2 : E ≠ 0) (h3 : F ≠ 0) (h4 : D < 7) (h5 : E < 7) (h6 : F < 7)
  (h_diff1 : D ≠ E) (h_diff2 : D ≠ F) (h_diff3 : E ≠ F)
  (h_eq : (49 * D + 7 * E + F) + (49 * E + 7 * F + D) + (49 * F + 7 * D + E) = 400 * D) :
  E + F = 6 :=
by
  sorry

end base7_digits_l470_47032


namespace cards_net_cost_equivalence_l470_47026

-- Define the purchase amount
def purchase_amount : ℝ := 10000

-- Define cashback percentages
def debit_card_cashback : ℝ := 0.01
def credit_card_cashback : ℝ := 0.005

-- Define interest rate for keeping money in the debit account
def interest_rate : ℝ := 0.005

-- A function to calculate the net cost after 1 month using the debit card
def net_cost_debit_card (purchase_amount : ℝ) (cashback_percentage : ℝ) : ℝ :=
  purchase_amount - purchase_amount * cashback_percentage

-- A function to calculate the net cost after 1 month using the credit card
def net_cost_credit_card (purchase_amount : ℝ) (cashback_percentage : ℝ) (interest_rate : ℝ) : ℝ :=
  purchase_amount - purchase_amount * cashback_percentage - purchase_amount * interest_rate

-- Final theorem stating that the net cost using both cards is the same
theorem cards_net_cost_equivalence : 
  net_cost_debit_card purchase_amount debit_card_cashback = 
  net_cost_credit_card purchase_amount credit_card_cashback interest_rate :=
by
  sorry

end cards_net_cost_equivalence_l470_47026


namespace min_band_members_exists_l470_47029

theorem min_band_members_exists (n : ℕ) :
  (∃ n, (∃ k : ℕ, n = 9 * k) ∧ (∃ m : ℕ, n = 10 * m) ∧ (∃ p : ℕ, n = 11 * p)) → n = 990 :=
by
  sorry

end min_band_members_exists_l470_47029


namespace monotonic_increasing_implies_range_a_l470_47003

-- Definition of the function f(x) = ax^3 - x^2 + x - 5
def f (a x : ℝ) : ℝ := a * x^3 - x^2 + x - 5

-- Derivative of f(x) with respect to x
def f_prime (a x : ℝ) : ℝ := 3 * a * x^2 - 2 * x + 1

-- The statement that proves the monotonicity condition implies the range for a
theorem monotonic_increasing_implies_range_a (a : ℝ) : 
  ( ∀ x, f_prime a x ≥ 0 ) → a ≥ (1:ℝ) / 3 := by
  sorry

end monotonic_increasing_implies_range_a_l470_47003


namespace inequality_proof_l470_47023

theorem inequality_proof (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 1) :
  (a / (b^2 * (c + 1))) + (b / (c^2 * (a + 1))) + (c / (a^2 * (b + 1))) ≥ 3 / 2 :=
sorry

end inequality_proof_l470_47023


namespace abc_plus_ab_plus_a_div_4_l470_47000

noncomputable def prob_abc_div_4 (a b c : ℕ) (isPositive_a : 0 < a) (isPositive_b : 0 < b) (isPositive_c : 0 < c) (a_in_range : a ∈ {k | 1 ≤ k ∧ k ≤ 2009}) (b_in_range : b ∈ {k | 1 ≤ k ∧ k ≤ 2009}) (c_in_range : c ∈ {k | 1 ≤ k ∧ k ≤ 2009}) : ℚ :=
  let total_elements : ℚ := 2009
  let multiples_of_4 := 502
  let non_multiples_of_4 := total_elements - multiples_of_4
  let prob_a_div_4 : ℚ := multiples_of_4 / total_elements
  let prob_a_not_div_4 : ℚ := non_multiples_of_4 / total_elements
  sorry

theorem abc_plus_ab_plus_a_div_4 : ∃ P : ℚ, prob_abc_div_4 a b c isPositive_a isPositive_b isPositive_c a_in_range b_in_range c_in_range = P :=
by sorry

end abc_plus_ab_plus_a_div_4_l470_47000


namespace calculate_expression_l470_47018

theorem calculate_expression :
  (50 - (2050 - 250)) + (2050 - (250 - 50)) = 100 := by
  sorry

end calculate_expression_l470_47018


namespace milk_problem_l470_47016

theorem milk_problem (x : ℕ) (hx : 0 < x)
    (total_cost_wednesday : 10 = x * (10 / x))
    (price_reduced : ∀ x, 0.5 = (10 / x - (10 / x) + 0.5))
    (extra_bags : 2 = (x + 2) - x)
    (extra_cost : 2 + 10 = x * (10 / x) + 2) :
    x^2 + 6 * x - 40 = 0 := by
  sorry

end milk_problem_l470_47016


namespace naomi_stickers_l470_47048

theorem naomi_stickers :
  ∃ S : ℕ, S > 1 ∧
    (S % 5 = 2) ∧
    (S % 9 = 2) ∧
    (S % 11 = 2) ∧
    S = 497 :=
by
  sorry

end naomi_stickers_l470_47048


namespace solve_ellipse_correct_m_l470_47031

noncomputable def ellipse_is_correct_m : Prop :=
  ∃ (m : ℝ), 
    (m > 6) ∧
    ((m - 2) - (10 - m) = 4) ∧
    (m = 8)

theorem solve_ellipse_correct_m : ellipse_is_correct_m :=
sorry

end solve_ellipse_correct_m_l470_47031


namespace balance_equation_l470_47095

variable (G Y W B : ℝ)
variable (balance1 : 4 * G = 8 * B)
variable (balance2 : 3 * Y = 7.5 * B)
variable (balance3 : 8 * B = 6 * W)

theorem balance_equation : 5 * G + 3 * Y + 4 * W = 23.5 * B := by
  sorry

end balance_equation_l470_47095


namespace find_a_b_sum_l470_47058

theorem find_a_b_sum
  (a b : ℝ)
  (h1 : 2 * a = -6)
  (h2 : a ^ 2 - b = 1) :
  a + b = 5 :=
by
  sorry

end find_a_b_sum_l470_47058
